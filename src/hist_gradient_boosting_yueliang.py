"""PBI 02 - Hist Gradient Boosting fraud classifier (HGB_safe and HGB_full).

Run from the repository root, for example:

    python src/hist_gradient_boosting_yueliang.py                       # HGB_safe + HGB_full with the selected settings
    python src/hist_gradient_boosting_yueliang.py --tune                # small search on safe train/validation first
    python src/hist_gradient_boosting_yueliang.py --sample-frac 0.05 --no-mlflow   # quick smoke test

How it works: the model builds many small decision trees one after another.
Each new tree focuses on the orders the previous trees got wrong, and the
final score adds all the trees together. Unlike Logistic Regression, trees
can learn combinations of features (e.g. a new account AND a large order).

No scaling is needed: a tree only asks "is this value above or below X",
so the size of the numbers does not matter.

Early stopping: trees are added until the validation log-loss stops
improving for n_iter_no_change rounds. The chronological validation period is
used (same idea as the team's MLP), so the number of trees is chosen on
orders that come after the training period. The test set is never used.

Threshold rule: highest recall while validation precision >= PRECISION_TARGET,
because the PO prioritises high precision with reasonable recall.
"""

import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import model_utils as mu

ALGORITHM = "HistGradientBoostingClassifier"
SHORT_NAME = "HGB"
DEVELOPER = os.environ.get("MODEL_DEVELOPER", "yueliang")

# MLflow saves models in the safe skops format, which only loads approved types.
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]

PRECISION_TARGET = 0.70
THRESHOLD_RULE = f"highest recall with validation precision >= {PRECISION_TARGET:.0%}"

# Settings selected by the validation search (see notebooks/hgb_yueliang.ipynb).
# Used for HGB_safe, HGB_full and future automated retraining (PBI 05).
SELECTED_PARAMS = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "l2_regularization": 1.0,
    "class_weight": None,
    "min_samples_leaf": 50,
    "max_iter": 500,             # upper limit; early stopping usually stops earlier
    "early_stopping": True,
    "scoring": "loss",            # log-loss on validation; a scorer object cannot be saved in MLflow's safe format
    "n_iter_no_change": 20,
    "random_state": mu.RANDOM_SEED,
}

# Small search: 2 x 2 x 2 x 2 = 16 models.
PARAM_GRID = {
    "learning_rate": [0.05, 0.1],        # how big a step each new tree takes
    "max_leaf_nodes": [31, 63],          # how complex each tree can be
    "l2_regularization": [0.0, 1.0],     # penalty that keeps the trees simpler
    "class_weight": [None, "balanced"],  # extra weight on fraud orders or not
}


def choose_threshold_min_precision(y_val, prob_val, min_precision=PRECISION_TARGET):
    """Lowest cut-off (so highest recall) whose VALIDATION precision is at least min_precision."""
    precision, recall, thresholds = precision_recall_curve(y_val, prob_val)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    meets_target = precision >= min_precision
    if not meets_target.any():
        raise ValueError(f"No threshold reaches {min_precision:.0%} precision on validation data")
    best = int(np.argmax(np.where(meets_target, recall, -1)))
    return float(thresholds[best])


def recall_at_precision(y_true, prob, min_precision=PRECISION_TARGET):
    """Best recall reachable while precision stays at or above min_precision (0 if never)."""
    precision, recall, _ = precision_recall_curve(y_true, prob)
    ok = precision[:-1] >= min_precision
    return float(recall[:-1][ok].max()) if ok.any() else 0.0


def create_model(params):
    """Build an untrained Hist Gradient Boosting model from a parameter dict."""
    return HistGradientBoostingClassifier(**params)


def train_model(X_train, y_train, X_val, y_val, params):
    """Fit on training data. Validation data is only used to decide when to stop adding trees."""
    model = create_model(params)
    start = time.perf_counter()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    info = {"metrics": {"trees_used": int(model.n_iter_),
                        "fit_seconds": round(time.perf_counter() - start, 2)}}
    return model, info


def feature_importance(model, X_val, y_val, features):
    """Permutation importance on a validation sample.

    Trees have no coefficients, so we measure how much validation average
    precision drops when each feature is randomly shuffled. A big drop means
    the model relies on that feature.
    """
    return mu.permutation_importance_table(model, X_val, y_val)


def tune(df_safe, grid=PARAM_GRID, base_params=SELECTED_PARAMS, sample_frac=None):
    """Try each combination on SAFE train data, score on SAFE validation data.

    The test set is not touched. Ranked by validation average precision (PR-AUC),
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
        model, info = train_model(X_train, y_train, X_val, y_val, params)
        seconds = time.perf_counter() - start
        prob = model.predict_proba(X_val)[:, 1]
        rows.append({**dict(zip(keys, values)),
                     "trees_used": info["metrics"]["trees_used"],
                     "val_average_precision": average_precision_score(y_val, prob),
                     "val_roc_auc": roc_auc_score(y_val, prob),
                     f"val_recall_at_{PRECISION_TARGET:.0%}_precision": recall_at_precision(y_val, prob),
                     "train_average_precision": average_precision_score(
                         y_train, model.predict_proba(X_train)[:, 1]),
                     "train_seconds": round(seconds, 2)})
    table = pd.DataFrame(rows).sort_values("val_average_precision", ascending=False)
    table["class_weight"] = table["class_weight"].fillna("None")
    best_row = table.iloc[0]
    best = {**base_params,
            "learning_rate": float(best_row["learning_rate"]),
            "max_leaf_nodes": int(best_row["max_leaf_nodes"]),
            "l2_regularization": float(best_row["l2_regularization"]),
            "class_weight": None if best_row["class_weight"] == "None" else best_row["class_weight"]}
    return table.reset_index(drop=True), best


def run(params=None, **kwargs):
    """Train HGB_safe and HGB_full (see model_utils.run_safe_and_full for options)."""
    kwargs.setdefault("developer", DEVELOPER)
    return mu.run_safe_and_full(ALGORITHM, SHORT_NAME, train_model, feature_importance,
                                params or SELECTED_PARAMS, trusted_types=TRUSTED_TYPES,
                                threshold_fn=choose_threshold_min_precision,
                                threshold_rule=THRESHOLD_RULE, **kwargs)


def main():
    parser = mu.build_arg_parser("Train Hist Gradient Boosting fraud models (PBI 02).", SELECTED_PARAMS)
    parser.set_defaults(developer=DEVELOPER)
    args = parser.parse_args()
    params = mu.parse_params(args.params, SELECTED_PARAMS)

    if args.tune:
        df_safe = mu.load_feature_data(mu.dataset_path("safe", args.data_dir))
        table, params = tune(df_safe, base_params=params, sample_frac=args.sample_frac)
        print(table.to_string(index=False))
        print("Selected:", params)
        if not args.no_mlflow:
            mu.log_tuning_table(table, "hist_gradient_boosting_tuning", args.experiment,
                                args.tracking_uri, args.developer)

    start = time.perf_counter()
    results, _ = run(params, datasets=args.datasets, data_dir=args.data_dir,
                     output_dir=args.output_dir, sample_frac=args.sample_frac,
                     use_mlflow=not args.no_mlflow, experiment=args.experiment,
                     tracking_uri=args.tracking_uri, developer=args.developer)
    print(mu.comparison_table(results).round(4).to_string())
    print(f"Total time (load + train + evaluate + log): {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
