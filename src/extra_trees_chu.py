"""Extra Trees fraud model: reusable training, validation search and CLI.

Run ``python src/extra_trees_chu.py`` to train Safe/Full with the saved
validation-selected settings. Add ``--tune`` to repeat the search. Hyperparameters are selected on Safe validation average precision;
test scores never influence model selection. No feature scaling is needed.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

import model_utils as mu

ALGORITHM = "ExtraTreesClassifier"
SHORT_NAME = "extra_trees_chu"
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]
DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": mu.RANDOM_SEED,
    "n_jobs": -1,
}
# Best Safe validation AP among the eight configurations in the executed notebook.
# Keep the original DEFAULT_PARAMS above as the search baseline.
SELECTED_PARAMS = {**DEFAULT_PARAMS, "max_depth": 25, "class_weight": None}

# Eight configurations, including the original defaults.
PARAM_GRID = {
    "max_depth": [15, 25],
    "min_samples_leaf": [5, 25],
    "class_weight": [None, "balanced"],
}
SELECTED_PARAMS_PATH = mu.DEFAULT_OUTPUT_DIR / SHORT_NAME / "selected_params.json"


def create_model(params=None):
    config = {**DEFAULT_PARAMS, **(params or {})}
    return ExtraTreesClassifier(**config)


def train_model(X_train, y_train, X_val, y_val, params):
    """Fit only on Train; validation is handled by the shared pipeline."""
    model = create_model(params).fit(X_train, y_train)
    return model, {"metrics": {
        "mean_tree_depth": float(np.mean([t.get_depth() for t in model.estimators_])),
        "mean_tree_leaves": float(np.mean([t.get_n_leaves() for t in model.estimators_])),
    }}


def feature_importance(model, X_val, y_val, features):
    """Impurity importance: model reliance, not evidence of causality or leakage."""
    return (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def tune(df_safe, grid=None, base_params=None, sample_frac=None):
    """Rank candidates by Safe validation AP without accessing Test features.

    Only training and validation matrices are materialized. All candidates use
    identical rows. A fixed seed makes sampling and tree construction repeatable.
    """
    from data_split import create_split_labels
    labels = create_split_labels(df_safe)
    features = mu.get_feature_columns(df_safe)
    parts = {}
    for name in ["Train", "Validation"]:
        frame = df_safe.loc[labels == name]
        if sample_frac is not None:
            if not 0 < sample_frac <= 1:
                raise ValueError("sample_frac must be in (0, 1].")
            frame = frame.sample(frac=sample_frac, random_state=mu.RANDOM_SEED)
        if frame[mu.TARGET].nunique() != 2:
            raise ValueError(f"{name} must contain both target classes.")
        parts[name] = (frame[features].astype("float32"), frame[mu.TARGET].to_numpy())
    X_train, y_train = parts["Train"]
    X_val, y_val = parts["Validation"]
    configs = [{**DEFAULT_PARAMS, **(base_params or {}), **candidate}
               for candidate in ParameterGrid(PARAM_GRID if grid is None else grid)]
    if not configs:
        raise ValueError("The parameter grid must contain a candidate.")
    rows = []
    for index, params in enumerate(configs):
        start = time.perf_counter()
        model = create_model(params).fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        prob = model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, prob)
        rows.append({"candidate": index, **params,
                     "val_average_precision": float(ap),
                     "val_roc_auc": float(roc_auc_score(y_val, prob)),
                     "train_average_precision": float(average_precision_score(
                         y_train, model.predict_proba(X_train)[:, 1])),
                     "train_seconds": train_seconds})
        print(f"Candidate {index + 1}/{len(configs)} | validation AP {ap:.5f} | "
              f"fit {train_seconds:.1f}s | depth={params['max_depth']} "
              f"leaf={params['min_samples_leaf']} weight={params['class_weight']}", flush=True)
        del model
    table = (pd.DataFrame(rows).sort_values("val_average_precision", ascending=False,
                                           kind="stable").reset_index(drop=True))
    # Recover the original Python parameter dict (not pandas scalar conversions).
    best = configs[int(table.iloc[0]["candidate"])].copy()
    return table, best


def load_selected_params(path=SELECTED_PARAMS_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No selected configuration at {path}. Run tuning first.")
    return {**DEFAULT_PARAMS, **json.loads(path.read_text(encoding="utf-8"))}


def run(params=None, **kwargs):
    kwargs.setdefault("developer", "chu")
    return mu.run_safe_and_full(ALGORITHM, SHORT_NAME, train_model, feature_importance,
                               (SELECTED_PARAMS.copy() if params is None else params), trusted_types=TRUSTED_TYPES, **kwargs)


def main():
    parser = mu.build_arg_parser("Train Extra Trees fraud models.", SELECTED_PARAMS)
    parser.set_defaults(developer="chu")
    parser.add_argument("--selected-params", help="JSON file saved by the notebook/search")
    args = parser.parse_args()
    base = load_selected_params(args.selected_params) if args.selected_params else SELECTED_PARAMS
    params = mu.parse_params(args.params, base)
    if args.tune:
        df = mu.load_feature_data(mu.dataset_path("safe", args.data_dir))
        table, params = tune(df, base_params=params, sample_frac=args.sample_frac)
        folder = Path(args.output_dir) / SHORT_NAME
        folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(folder / "tuning_results.csv", index=False)
        (folder / "selected_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        if not args.no_mlflow:
            mu.log_tuning_table(table, "extra_trees_tuning", args.experiment,
                                args.tracking_uri, args.developer)
        del df
    results, _ = run(params, datasets=args.datasets, data_dir=args.data_dir,
                     output_dir=args.output_dir, sample_frac=args.sample_frac,
                     use_mlflow=not args.no_mlflow, experiment=args.experiment,
                     tracking_uri=args.tracking_uri, developer=args.developer)
    print(mu.comparison_table(results).to_string())


if __name__ == "__main__":
    main()
