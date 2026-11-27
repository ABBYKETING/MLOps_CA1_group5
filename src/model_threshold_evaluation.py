"""Evaluate Safe models under a consistent validation precision policy."""

import argparse
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

import model_utils as mu

from model_comparison import (
    DEFAULT_EXPERIMENT,
    DEFAULT_OUTPUT_DIR,
    METRICS,
    fetch_official_runs,
    extract_model_metrics,
)


PRECISION_TARGETS = (0.60, 0.70, 0.80)
PRIMARY_TARGET = 0.70


def choose_precision_threshold(y_true, probabilities, target):
    """Maximise validation recall subject to minimum precision."""

    if not 0 < target <= 1:
        raise ValueError("Precision target must be in (0, 1].")

    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    # The final PR point has no decision threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    feasible = np.flatnonzero(precision >= target)

    if len(feasible) == 0:
        return None

    # Prefer highest recall, then highest precision.
    best = max(
        feasible,
        key=lambda i: (
            recall[i],
            precision[i],
            thresholds[i],
        ),
    )

    return float(thresholds[best])


def load_validation_data(safe_runs, data_dir=None):
    """Verify and load the original Safe validation dataset."""

    path = mu.dataset_path("safe", data_dir)
    actual_hash = mu.file_sha256(path)

    expected_hashes = set(safe_runs["dataset_sha256"])

    if expected_hashes != {actual_hash}:
        raise ValueError("Safe dataset SHA-256 mismatch.")

    df = mu.load_feature_data(path)
    features = mu.get_feature_columns(df)

    if len(features) != 31:
        raise ValueError("Expected 31 Safe features.")

    labels = mu.create_split_labels(df)
    validation = df.loc[labels == "Validation"]

    X = validation[features].copy()
    y = validation["is_fraud"].to_numpy()

    if len(y) != 139874 or int(y.sum()) != 18993:
        raise ValueError("Unexpected validation dataset counts.")

    print("Safe dataset SHA-256:", actual_hash)
    print("Validation orders:", len(y))
    print("Validation fraud orders:", int(y.sum()))
    print("Validation feature count:", len(features))

    return X, y


def verify_original_metrics(row, y, probabilities):
    """Ensure reloaded predictions reproduce the logged metrics."""

    threshold = float(row["decision_threshold"])

    calculated = mu.evaluate(
        y, probabilities, threshold
    )

    for metric in METRICS:
        source_key = (
            "average_precision"
            if metric == "pr_auc"
            else metric
        )

        expected = float(row[f"val_{metric}"])
        actual = float(calculated[source_key])

        if not np.isclose(
            expected,
            actual,
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{row['run_name']}: {metric} mismatch. "
                f"Logged={expected}, calculated={actual}"
            )


def evaluate_safe_models(safe_runs, X_val, y_val):
    """Evaluate every Safe model at identical precision targets."""

    rows = []

    for _, run in safe_runs.iterrows():

        print("\nEvaluating:", run["run_name"])

        # Preserve the original training input representation.
        X = X_val.copy()

        if run["team_member"] != "abby":
            X = X.astype("float32")

        with tempfile.TemporaryDirectory() as temp_dir:

            model = mlflow.sklearn.load_model(
                run["model_uri"],
                dst_path=temp_dir,
            )

            if model.n_features_in_ != 31:
                raise ValueError("Model feature count mismatch.")

            if list(model.classes_) != [0, 1]:
                raise ValueError("Unexpected model classes.")

            if hasattr(model, "feature_names_in_"):
                model_features = list(model.feature_names_in_)

                if set(model_features) != set(X.columns):
                    raise ValueError("Model feature names mismatch.")

                X = X[model_features]

            probabilities, inference_seconds = (
                mu.timed_predict_proba(model, X)
            )

            probabilities = np.asarray(
                probabilities, dtype=np.float64
            )

            if (
                probabilities.shape != (len(y_val),)
                or not np.isfinite(probabilities).all()
                or not (
                    (probabilities >= 0)
                    & (probabilities <= 1)
                ).all()
            ):
                raise ValueError("Invalid predicted probabilities.")

            verify_original_metrics(
                run, y_val, probabilities
            )

            print("Original validation metrics: PASS")

            for target in PRECISION_TARGETS:

                threshold = choose_precision_threshold(
                    y_val, probabilities, target
                )

                result = {
                    "run_name": run["run_name"],
                    "run_id": run["run_id"],
                    "model_uri": run["model_uri"],
                    "team_member": run["team_member"],
                    "algorithm": run["algorithm"],
                    "dataset_version": "safe",
                    "dataset_sha256": run["dataset_sha256"],
                    "precision_target": target,
                    "target_achieved": threshold is not None,
                    "original_threshold": float(
                        run["decision_threshold"]
                    ),
                    "selected_threshold": threshold,
                    "val_pr_auc": float(run["val_pr_auc"]),
                    "val_roc_auc": float(run["val_roc_auc"]),
                    "val_inference_seconds": inference_seconds,
                }

                if threshold is None:

                    print(
                        f"  Target {target:.0%}: NOT ACHIEVABLE"
                    )

                else:

                    metrics = mu.evaluate(
                        y_val, probabilities, threshold
                    )

                    if metrics["precision"] < target - 1e-12:
                        raise ValueError(
                            "Selected threshold fails precision target."
                        )

                    result.update({
                        "val_accuracy": metrics["accuracy"],
                        "val_precision": metrics["precision"],
                        "val_recall": metrics["recall"],
                        "val_f1": metrics["f1"],
                        "val_tp": metrics["tp"],
                        "val_fp": metrics["fp"],
                        "val_fn": metrics["fn"],
                        "val_tn": metrics["tn"],
                        "val_alerts": (
                            metrics["tp"] + metrics["fp"]
                        ),
                    })

                    print(
                        f"  Target {target:.0%} | "
                        f"Threshold {threshold:.6f} | "
                        f"Precision {metrics['precision']:.4f} | "
                        f"Recall {metrics['recall']:.4f} | "
                        f"FP {metrics['fp']}"
                    )

                rows.append(result)

            del model

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Consistent validation threshold evaluation."
    )

    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
    )

    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
    )

    parser.add_argument("--data-dir", default=None)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if (
        not args.tracking_uri
        or not args.tracking_uri.startswith("https://")
    ):
        parser.error("A shared HTTPS tracking URI is required.")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()

    runs = fetch_official_runs(
        client, args.experiment
    )

    table = extract_model_metrics(runs)

    safe_runs = table[
        (table["dataset_version"] == "safe")
        & table["selection_candidate"]
    ].copy()

    if len(safe_runs) != 8:
        raise ValueError("Expected eight Safe candidates.")

    X_val, y_val = load_validation_data(
        safe_runs, args.data_dir
    )

    sensitivity = evaluate_safe_models(
        safe_runs, X_val, y_val
    )

    if len(sensitivity) != 24:
        raise ValueError("Expected 24 threshold evaluations.")

    primary = sensitivity[
        sensitivity["precision_target"] == PRIMARY_TARGET
    ].copy()

    primary = primary.sort_values(
        [
            "target_achieved",
            "val_recall",
            "val_precision",
            "val_pr_auc",
            "run_name",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    )

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    sensitivity.to_csv(
        args.output_dir / "threshold_sensitivity.csv",
        index=False,
    )

    primary.to_csv(
        args.output_dir / "safe_model_comparison.csv",
        index=False,
    )

    print("\n=== PRIMARY PRECISION TARGET: 70% ===")

    print(
        primary[
            [
                "team_member",
                "algorithm",
                "target_achieved",
                "selected_threshold",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_pr_auc",
                "val_fp",
                "val_alerts",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== SENSITIVITY ANALYSIS: VALIDATION RECALL ===")

    print(
        sensitivity.pivot(
            index=["team_member", "algorithm"],
            columns="precision_target",
            values="val_recall",
        ).round(4).to_string()
    )

    print("\nSafe models evaluated:", len(safe_runs))
    print("Threshold evaluations:", len(sensitivity))
    print("Original validation metric verification: PASS")
    print("Threshold evaluation: PASS")
    print("Output directory:", args.output_dir)


if __name__ == "__main__":
    main()
