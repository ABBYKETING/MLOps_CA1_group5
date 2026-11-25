"""PBI 02 - Decision Tree fraud classifier (DT_safe and DT_full).

Run from the repository root, for example:

    python src/decision_tree.py                       # DT_safe + DT_full with the selected settings
    python src/decision_tree.py --tune                # small search on safe train/validation first
    python src/decision_tree.py --sample-frac 0.05 --no-mlflow   # quick smoke test
    python src/decision_tree.py --datasets safe --params "{\"max_depth\": 8}"

Why no scaling? A decision tree splits on "feature <= value", so the scale of a
feature does not change the result. All features are already numeric
(log transforms and one-hot columns were built in data_processing.py with fixed
rules), so this model needs no extra preprocessing.
"""

import itertools
import time

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

import model_utils as mu

ALGORITHM = "DecisionTreeClassifier"
SHORT_NAME = "DT"

# Internal sklearn type MLflow must trust when saving the tree (we trained it ourselves)
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# Settings selected by the validation search (see notebooks/decision_tree.ipynb).
# Used for DT_safe, DT_full and future automated retraining (PBI 05).
SELECTED_PARAMS = {
    "criterion": "entropy",
    "max_depth": 10,
    "min_samples_split": 100,
    "min_samples_leaf": 50,
    "class_weight": None,
    "random_state": mu.RANDOM_SEED,
}

# Small search: 2 x 3 x 2 x 2 = 24 trees (a few seconds each).
# min_samples_split is fixed: min_samples_leaf already stops tiny splits.
PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [6, 10, 14],
    "min_samples_leaf": [50, 200],
    "class_weight": [None, "balanced"],
}


def create_model(params):
    """Build an untrained DecisionTreeClassifier from a parameter dict."""
    return DecisionTreeClassifier(**params)


def train_model(X_train, y_train, X_val, y_val, params):
    """Fit the tree on training data only. X_val is not used by the tree itself."""
    model = create_model(params)
    model.fit(X_train, y_train)
    info = {"metrics": {"tree_depth": model.get_depth(), "tree_leaves": model.get_n_leaves()}}
    return model, info


def feature_importance(model, X_val, y_val, features):
    """Built-in impurity importance (how much each feature reduces Gini/entropy)."""
    return (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def tune(df_safe, grid=PARAM_GRID, base_params=SELECTED_PARAMS, sample_frac=None):
    """Try each combination on SAFE train data, score on SAFE validation data.

    The test set is not touched. Ranked by validation average precision,
    which does not depend on a threshold and suits imbalanced data.
    """
    parts, _ = mu.split_data(df_safe, sample_frac=sample_frac)
    X_train, y_train = parts["Train"]
    X_val, y_val = parts["Validation"]
    rows = []
    keys = list(grid)
    for values in itertools.product(*grid.values()):
        params = {**base_params, **dict(zip(keys, values))}
        start = time.perf_counter()
        model = create_model(params).fit(X_train, y_train)
        seconds = time.perf_counter() - start
        prob = model.predict_proba(X_val)[:, 1]
        rows.append({**dict(zip(keys, values)),
                     "val_average_precision": average_precision_score(y_val, prob),
                     "val_roc_auc": roc_auc_score(y_val, prob),
                     "train_average_precision": average_precision_score(
                         y_train, model.predict_proba(X_train)[:, 1]),
                     "depth": model.get_depth(), "leaves": model.get_n_leaves(),
                     "train_seconds": round(seconds, 2)})
    table = pd.DataFrame(rows).sort_values("val_average_precision", ascending=False)
    best = {**base_params, **{k: table.iloc[0][k] for k in keys}}
    best["max_depth"] = int(best["max_depth"])
    best["min_samples_leaf"] = int(best["min_samples_leaf"])
    if pd.isna(best["class_weight"]):
        best["class_weight"] = None
    return table.reset_index(drop=True), best


def run(params=None, **kwargs):
    """Train DT_safe and DT_full (see model_utils.run_safe_and_full for options)."""
    return mu.run_safe_and_full(ALGORITHM, SHORT_NAME, train_model, feature_importance,
                                params or SELECTED_PARAMS, trusted_types=TRUSTED_TYPES, **kwargs)


def main():
    parser = mu.build_arg_parser("Train Decision Tree fraud models (PBI 02).", SELECTED_PARAMS)
    args = parser.parse_args()
    params = mu.parse_params(args.params, SELECTED_PARAMS)

    if args.tune:
        df_safe = mu.load_feature_data(mu.dataset_path("safe", args.data_dir))
        table, params = tune(df_safe, base_params=params, sample_frac=args.sample_frac)
        print(table.head(10).to_string(index=False))
        print("Selected:", params)
        if not args.no_mlflow:
            mu.log_tuning_table(table, "decision_tree_tuning", args.experiment, args.tracking_uri, args.developer)

    start = time.perf_counter()
    results, _ = run(params, datasets=args.datasets, data_dir=args.data_dir,
                     output_dir=args.output_dir, sample_frac=args.sample_frac,
                     use_mlflow=not args.no_mlflow, experiment=args.experiment,
                     tracking_uri=args.tracking_uri, developer=args.developer)
    print(mu.comparison_table(results).round(4).to_string())
    print(f"Total time (load + train + evaluate + log): {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
