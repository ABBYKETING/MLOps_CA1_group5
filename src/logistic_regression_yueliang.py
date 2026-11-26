"""PBI 02 - Logistic Regression fraud classifier (LR_safe and LR_full).

Run from the repository root, for example:

    python src/logistic_regression.py                       # LR_safe + LR_full with the selected settings
    python src/logistic_regression.py --tune                # small search on safe train/validation first
    python src/logistic_regression.py --sample-frac 0.05 --no-mlflow   # quick smoke test

Why scaling? Logistic Regression multiplies each feature by a weight, so a feature
with large values (e.g. account_age_days) would dominate one with small values.
StandardScaler puts every feature on the same scale. It sits inside the Pipeline,
so it is fitted on the training data only and the coefficients can be compared.

Threshold rule: highest recall while validation precision >= PRECISION_TARGET,
because the PO prioritises high precision with reasonable recall.
"""

import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import model_utils as mu

ALGORITHM = "LogisticRegression"
SHORT_NAME = "LR"
DEVELOPER = os.environ.get("MODEL_DEVELOPER", "yueliang")

# Pipeline, StandardScaler and LogisticRegression are trusted by default,
# so MLflow's safe (skops) format needs no extra types for this model.
TRUSTED_TYPES = []

PRECISION_TARGET = 0.70
THRESHOLD_RULE = f"highest recall with validation precision >= {PRECISION_TARGET:.0%}"

# Settings selected by the validation search (see notebooks/03_logreg_yueliang.ipynb).
# Used for LR_safe, LR_full and future automated retraining (PBI 05).
SELECTED_PARAMS = {
    "C": 0.01,
    "class_weight": "balanced",
    "max_iter": 2000,
    "solver": "lbfgs",
    "random_state": mu.RANDOM_SEED,
}

# Small search: 4 x 2 = 8 models (about 2 seconds each).
# C = how strongly the weights are kept small (smaller C = stronger regularisation).
PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
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
    """Build an untrained scaler + Logistic Regression pipeline from a parameter dict."""
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(**params))])


def train_model(X_train, y_train, X_val, y_val, params):
    """Fit on training data only. X_val is not used by the model itself."""
    model = create_model(params)
    model.fit(X_train, y_train)
    info = {"metrics": {"iterations": int(model.named_steps["model"].n_iter_[0])}}
    return model, info


def feature_importance(model, X_val, y_val, features):
    """Coefficients on the scaled features.

    Positive = raises the fraud probability, negative = lowers it.
    Ranked by size (absolute value) because both directions matter.
    """
    coef = model.named_steps["model"].coef_[0]
    return (pd.DataFrame({"feature": features, "coefficient": coef, "importance": np.abs(coef)})
            .sort_values("importance", ascending=False).reset_index(drop=True))


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
        model = create_model(params).fit(X_train, y_train)
        seconds = time.perf_counter() - start
        prob = model.predict_proba(X_val)[:, 1]
        rows.append({**dict(zip(keys, values)),
                     "val_average_precision": average_precision_score(y_val, prob),
                     "val_roc_auc": roc_auc_score(y_val, prob),
                     f"val_recall_at_{PRECISION_TARGET:.0%}_precision": recall_at_precision(y_val, prob),
                     "train_average_precision": average_precision_score(
                         y_train, model.predict_proba(X_train)[:, 1]),
                     "train_seconds": round(seconds, 2)})
    table = pd.DataFrame(rows).sort_values("val_average_precision", ascending=False)
    table["class_weight"] = table["class_weight"].fillna("None")
    best_row = table.iloc[0]
    best = {**base_params, "C": float(best_row["C"]),
            "class_weight": None if best_row["class_weight"] == "None" else best_row["class_weight"]}
    return table.reset_index(drop=True), best


def run(params=None, **kwargs):
    """Train LR_safe and LR_full (see model_utils.run_safe_and_full for options)."""
    kwargs.setdefault("developer", DEVELOPER)
    return mu.run_safe_and_full(ALGORITHM, SHORT_NAME, train_model, feature_importance,
                                params or SELECTED_PARAMS, trusted_types=TRUSTED_TYPES,
                                threshold_fn=choose_threshold_min_precision,
                                threshold_rule=THRESHOLD_RULE, **kwargs)


def main():
    parser = mu.build_arg_parser("Train Logistic Regression fraud models (PBI 02).", SELECTED_PARAMS)
    parser.set_defaults(developer=DEVELOPER)
    args = parser.parse_args()
    params = mu.parse_params(args.params, SELECTED_PARAMS)

    if args.tune:
        df_safe = mu.load_feature_data(mu.dataset_path("safe", args.data_dir))
        table, params = tune(df_safe, base_params=params, sample_frac=args.sample_frac)
        print(table.to_string(index=False))
        print("Selected:", params)
        if not args.no_mlflow:
            mu.log_tuning_table(table, "logistic_regression_tuning", args.experiment,
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
