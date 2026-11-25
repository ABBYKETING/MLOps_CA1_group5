"""Chu's LightGBM baseline for Safe and Full fraud feature datasets.

One fixed configuration, two independently fitted models. No tuning or early
stopping. The shared pipeline chooses a threshold on Safe validation by max F1
and reuses it for Full. Full is exploratory, not eligible for production.

Examples (from repository root):
    python src/lightgbm_chu.py --no-mlflow
    python src/lightgbm_chu.py --tracking-uri http://localhost:5000
    python src/lightgbm_chu.py --sample-frac 0.02 --no-mlflow
"""
import argparse
import os

import pandas as pd
from lightgbm import LGBMClassifier

import model_utils as mu

ALGORITHM = "LGBMClassifier"
SHORT_NAME = "lightgbm_chu"
DEVELOPER = "chu"
# Required for skops reloads of models trained by this code. Never infer trust
# automatically from an external model file.
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]
DEFAULT_PARAMS = {
    "objective": "binary",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "random_state": mu.RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def create_model(params=None):
    """Construct the baseline; caller overrides do not mutate DEFAULT_PARAMS."""
    return LGBMClassifier(**{**DEFAULT_PARAMS, **(params or {})})


def train_model(X_train, y_train, X_val, y_val, params):
    """Fit on Train only; neither validation nor test controls tree fitting."""
    model = create_model(params).fit(X_train, y_train)
    return model, {"metrics": {
        "boosting_iterations": float(model.n_iter_),
        "tree_count": float(model.booster_.num_trees()),
    }}


def feature_importance(model, X_val, y_val, features):
    """Total split gain per feature, normalized for Safe/Full visualization."""
    gains = model.booster_.feature_importance(importance_type="gain")
    total = float(gains.sum())
    return (pd.DataFrame({"feature": features,
                          "importance": gains / total if total else gains})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def run(params=None, **kwargs):
    """Train the fixed baseline with shared splitting, evaluation and logging."""
    kwargs.setdefault("developer", DEVELOPER)
    config = {**DEFAULT_PARAMS, **(params or {})}
    return mu.run_safe_and_full(
        ALGORITHM, SHORT_NAME, train_model, feature_importance, config,
        trusted_types=TRUSTED_TYPES, **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Chu's LightGBM baseline on Safe and Full.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(mu.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--datasets", nargs="+", choices=["safe", "full"], default=["safe", "full"])
    parser.add_argument("--sample-frac", type=float, default=None, help="Development smoke checks only")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", mu.DEFAULT_EXPERIMENT))
    parser.add_argument("--developer", default=DEVELOPER)
    args = parser.parse_args()
    if args.sample_frac is not None and not 0 < args.sample_frac <= 1:
        parser.error("--sample-frac must be in (0, 1].")
    datasets = tuple(d for d in ["safe", "full"] if d in args.datasets)
    results, _ = run(
        datasets=datasets, data_dir=args.data_dir, output_dir=args.output_dir,
        sample_frac=args.sample_frac, use_mlflow=not args.no_mlflow,
        tracking_uri=args.tracking_uri, experiment=args.experiment, developer=args.developer,
    )
    print(mu.comparison_table(results).round(4).to_string())


if __name__ == "__main__":
    main()
