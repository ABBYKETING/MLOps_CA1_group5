"""Abby's fixed Logistic Regression baseline (Safe v1 and Full v2).

Run from the repository root:
    python src/logistic_regression_abby.py --sample-frac 0.02 --no-mlflow
    python src/logistic_regression_abby.py --no-mlflow
    python src/logistic_regression_abby.py --tracking-uri http://127.0.0.1:5004

The original notebook uses a fixed 0.5 threshold and a train-fitted scaler.
The Full dataset is exploratory; Safe is only a candidate for further review.
Shared pipeline standardisation is a subsequent PBI 05 task.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The existing shared helpers use top-level imports from src.
# Keep this entry point usable from both the CLI and a root-level import.
if __package__:
    source_dir = str(Path(__file__).resolve().parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

import model_utils as mu
from data_split import create_split_labels

ALGORITHM = "LogisticRegression"
SHORT_NAME = "logistic_regression_abby"
DEVELOPER = "abby"
THRESHOLD = 0.5
THRESHOLD_RULE = "fixed 0.5 (notebook baseline)"
TRUSTED_TYPES = []
SELECTED_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": None,
    "random_state": mu.RANDOM_SEED,
}


def create_model(params=None):
    """Build a fresh pipeline; fit the scaler on training data only."""
    config = {**SELECTED_PARAMS, **(params or {})}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**config)),
    ])


def train_model(X_train, y_train, X_val, y_val, params):
    """Match the team's training interface; validation is not used for fitting."""
    model = create_model(params)
    model.fit(X_train, y_train)
    iterations = int(model.named_steps["classifier"].n_iter_.max())
    return model, {"metrics": {"iterations_used": iterations}}


def feature_importance(model, X_val, y_val, features):
    """Rank standardised coefficients by absolute magnitude, not causality."""
    coef = model.named_steps["classifier"].coef_[0]
    return (
        pd.DataFrame({
            "feature": features,
            "coefficient": coef,
            "importance": np.abs(coef),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def split_data(df, sample_frac=None):
    """Use the shared chronological split without forcing float32 conversion."""
    labels = create_split_labels(df)
    features = mu.get_feature_columns(df)
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in features):
        raise ValueError("All prediction features must be numeric.")
    if not labels.isin(["Train", "Validation", "Test"]).all():
        raise ValueError("Unexpected chronological split labels.")
    parts = {}
    for name in ("Train", "Validation", "Test"):
        part = df.loc[labels == name]
        if sample_frac is not None:
            part = part.sample(frac=sample_frac, random_state=mu.RANDOM_SEED)
        y = part[mu.TARGET].astype(int)
        if set(y.unique()) != {0, 1}:
            raise ValueError(f"{name} split must contain both classes.")
        parts[name] = (part[features], y)
    return parts, features


def _connect_tracking(experiment, tracking_uri):
    """Require an explicit URI or environment configuration for MLflow."""
    import mlflow_utils as tracking
    if tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    tracking.connect_mlflow(experiment)
    return tracking


def run(params=None, *, datasets=("safe", "full"), data_dir=None,
        output_dir=mu.DEFAULT_OUTPUT_DIR, sample_frac=None, use_mlflow=True,
        experiment=mu.DEFAULT_EXPERIMENT, tracking_uri=None, developer=DEVELOPER):
    """Return (results, models) with the same overall interface as teammates."""
    if not datasets or set(datasets) - {"safe", "full"}:
        raise ValueError("datasets must contain safe and/or full.")
    if sample_frac is not None and not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1].")
    if sample_frac is not None and use_mlflow:
        raise ValueError("Use --no-mlflow for sampled smoke tests.")
    config = {**SELECTED_PARAMS, **(params or {})}
    versions = tuple(d for d in ("safe", "full") if d in datasets)
    paths = {d: mu.dataset_path(d, data_dir) for d in versions}
    frames = {d: mu.load_feature_data(paths[d]) for d in versions}
    if len(frames) == 2:
        mu.check_same_orders(frames["safe"], frames["full"])
        shared = mu.get_feature_columns(frames["safe"])
        if not set(shared).issubset(mu.get_feature_columns(frames["full"])):
            raise ValueError("Safe features must be a subset of Full features.")
        if not frames["safe"][shared].equals(frames["full"][shared]):
            raise ValueError("Shared feature values differ between datasets.")

    tracking = _connect_tracking(experiment, tracking_uri) if use_mlflow else None
    results, models = [], {}
    for dataset in versions:
        parts, features = split_data(frames[dataset], sample_frac)
        X_train, y_train = parts["Train"]
        X_val, y_val = parts["Validation"]
        start = time.perf_counter()
        model, info = train_model(X_train, y_train, X_val, y_val, config)
        train_seconds = time.perf_counter() - start
        predictions, evaluations = {}, {}
        inference_seconds = 0.0

        for name, split in (("Train", "train"), ("Validation", "val"), ("Test", "test")):
            X, y = parts[name]
            prob, seconds = mu.timed_predict_proba(model, X)
            predictions[split] = (y, prob)
            evaluations[name] = mu.evaluate(y, prob, THRESHOLD)
            if split == "test":
                inference_seconds = seconds

        version = "v1" if dataset == "safe" else "v2"
        run_name = f"{SHORT_NAME}_{dataset}"
        candidate = dataset == "safe" and sample_frac is None
        result = {
            "run_name": run_name, "algorithm": ALGORITHM,
            "dataset_type": dataset, "model_version": version,
            "developer": developer, "params": config, "features": features,
            "feature_count": len(features),
            "dataset_sha256": mu.file_sha256(paths[dataset]),
            "rows": {name: len(y) for name, (_, y) in parts.items()},
            "threshold": THRESHOLD, "threshold_rule": THRESHOLD_RULE,
            "train": evaluations["Train"],
            "validation": evaluations["Validation"],
            "test": evaluations["Test"],
            "train_seconds": train_seconds,
            "inference_seconds_test": inference_seconds,
            "extra_metrics": info["metrics"],
            "selection_candidate": candidate,
            "production_eligible": False,
            "sample_frac": sample_frac,
        }

        importance = feature_importance(model, X_val, y_val, features)
        run_dir = Path(output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        importance.to_csv(run_dir / "feature_importance.csv", index=False)
        joblib.dump(model, run_dir / "model.joblib")

        if tracking is not None:
            import mlflow
            with tracking.start_model_run(
                member=developer, algorithm="logistic_regression",
                dataset_version=dataset, model_version=version
            ) as active_run:
                mlflow.set_tags({
                    "pbi": "PBI-02",
                    "run_type": "training_script",
                    "exploratory": str(dataset == "full").lower(),
                    "selection_candidate": str(candidate).lower(),
                    "production_eligible": "false",
                    "snapshot_availability": "unconfirmed",
                })
                tracking.log_model_parameters(config, THRESHOLD, len(features))
                mlflow.log_params({
                    "scaler": "StandardScaler",
                    "threshold_rule": THRESHOLD_RULE,
                    "dataset_sha256": result["dataset_sha256"],
                    "training_records": len(y_train),
                    "validation_records": len(y_val),
                    "test_records": len(parts["Test"][1]),
                })
                mlflow.log_metrics({
                    "training_time_seconds": train_seconds,
                    "inference_seconds_test": inference_seconds,
                    **info["metrics"],
                })
                for split, (y, prob) in predictions.items():
                    tracking.log_classification_metrics(y, prob, THRESHOLD, split)
                    if split != "train":
                        tracking.log_evaluation_artifacts(
                            y, prob, features, THRESHOLD, split
                        )
                model_info = tracking.log_trained_model(
                    model, X_train.head(5),
                    model_name=f"logistic_regression_{dataset}",
                    skops_trusted_types=TRUSTED_TYPES or None,
                )
                mlflow.log_artifact(str(run_dir / "feature_importance.csv"))
                result["mlflow_run_id"] = active_run.info.run_id
                result["model_uri"] = model_info.model_uri

        (run_dir / "result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        result["importance"] = importance
        results.append(result)
        models[dataset] = model

    return results, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(mu.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--datasets", nargs="+", choices=["safe", "full"],
                        default=["safe", "full"])
    parser.add_argument("--params", default=None, help="JSON hyperparameter overrides")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME",
                                                         mu.DEFAULT_EXPERIMENT))
    parser.add_argument("--developer", default=DEVELOPER)
    args = parser.parse_args()
    params = mu.parse_params(args.params, SELECTED_PARAMS)
    results, _ = run(
        params, datasets=args.datasets, data_dir=args.data_dir,
        output_dir=args.output_dir, sample_frac=args.sample_frac,
        use_mlflow=not args.no_mlflow, experiment=args.experiment,
        tracking_uri=args.tracking_uri, developer=args.developer,
    )
    print(mu.comparison_table(results).round(4).to_string())
    for result in results:
        print(result["run_name"], "| selection candidate:",
              result["selection_candidate"], "| version:", result["model_version"])


if __name__ == "__main__":
    main()
