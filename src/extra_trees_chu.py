"""Chu's Extra Trees baseline for Safe and Full fraud feature datasets.

One fixed configuration, two independently fitted models. No hyperparameter
tuning or improved configuration. The shared pipeline chooses a threshold on
Safe validation by max F1 and reuses it for Full. Full is exploratory, not eligible for production.

Examples (from repository root):
    python src/extra_trees_chu.py --no-mlflow
    python src/extra_trees_chu.py --tracking-uri http://localhost:5000
    python src/extra_trees_chu.py --sample-frac 0.02 --no-mlflow
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

import model_utils as mu

ALGORITHM = "ExtraTreesClassifier"
SHORT_NAME = "extra_trees_chu"
DEVELOPER = "chu"
# Required for skops reloads of models trained by this code. Never infer trust
# automatically from an external model file.
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


def create_model(params=None):
    """Construct the baseline; caller overrides do not mutate DEFAULT_PARAMS."""
    return ExtraTreesClassifier(**{**DEFAULT_PARAMS, **(params or {})})


def train_model(X_train, y_train, X_val, y_val, params):
    """Fit on Train only; neither validation nor test controls tree fitting."""
    model = create_model(params).fit(X_train, y_train)
    return model, {"metrics": {
        "tree_count": float(len(model.estimators_)),
        "mean_tree_depth": float(np.mean([tree.get_depth() for tree in model.estimators_])),
        "mean_tree_leaves": float(np.mean([tree.get_n_leaves() for tree in model.estimators_])),
    }}


def feature_importance(model, X_val, y_val, features):
    """Mean normalized impurity reduction across trees, not causal importance."""
    return (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
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
    parser = argparse.ArgumentParser(description="Train Chu's Extra Trees baseline on Safe and Full.")
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
