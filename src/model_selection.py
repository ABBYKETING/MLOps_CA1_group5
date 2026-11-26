"""Lock the validation-selected top two models and evaluate on test data."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

import model_utils as mu

from model_comparison import (
    DEFAULT_EXPERIMENT,
    DEFAULT_OUTPUT_DIR,
    METRICS,
    fetch_official_runs,
    extract_model_metrics,
)

from model_threshold_evaluation import PRIMARY_TARGET


def select_top_two(comparison):
    """Rank eligible models using validation results only."""

    if len(comparison) != 8:
        raise ValueError("Expected eight Safe candidates.")

    if comparison["run_id"].nunique() != 8:
        raise ValueError("Duplicate candidate runs.")

    if not comparison["dataset_version"].eq("safe").all():
        raise ValueError("Only Safe models may be selected.")

    if not comparison["precision_target"].eq(PRIMARY_TARGET).all():
        raise ValueError("Unexpected precision policy.")

    eligible = comparison[
        comparison["target_achieved"].eq(True)
        & comparison["val_precision"].ge(PRIMARY_TARGET)
        & comparison["val_recall"].gt(0)
    ].copy()

    if len(eligible) < 2:
        raise ValueError("Fewer than two eligible models.")

    ranked = eligible.sort_values(
        [
            "val_recall",
            "val_precision",
            "val_pr_auc",
            "run_name",
        ],
        ascending=[False, False, False, True],
    )

    return ranked.head(2).reset_index(drop=True)


def lock_selection(top_two, output_dir):
    """Save the selection before evaluating the test predictions."""

    candidates = []

    for i, row in top_two.iterrows():

        threshold = float(row["selected_threshold"])

        if not 0 <= threshold <= 1:
            raise ValueError("Invalid selected threshold.")

        candidates.append({
            "rank": i + 1,
            "proposed_role": (
                "champion" if i == 0 else "challenger"
            ),
            "run_name": row["run_name"],
            "run_id": row["run_id"],
            "model_uri": row["model_uri"],
            "team_member": row["team_member"],
            "algorithm": row["algorithm"],
            "dataset_version": "safe",
            "dataset_sha256": row["dataset_sha256"],
            "decision_threshold": threshold,
            "original_threshold": float(
                row["original_threshold"]
            ),
            "val_precision": float(row["val_precision"]),
            "val_recall": float(row["val_recall"]),
            "val_f1": float(row["val_f1"]),
            "val_pr_auc": float(row["val_pr_auc"]),
            "val_alerts": int(row["val_alerts"]),
            "val_tp": int(row["val_tp"]),
            "val_fp": int(row["val_fp"]),
            "production_eligible": False,
        })

    selection = {
        "selection_basis": "validation_only",
        "precision_target": PRIMARY_TARGET,
        "precision_target_source": (
            "Proposed technical comparison policy; "
            "not a stakeholder-mandated numeric requirement."
        ),
        "ranking_policy": (
            "Maximum validation recall subject to minimum "
            "precision; ties broken by precision, AP and run name."
        ),
        "production_eligible": False,
        "models": candidates,
    }

    path = output_dir / "top_two_models.json"

    if path.exists():

        existing = json.loads(
            path.read_text(encoding="utf-8")
        )

        if existing != selection:
            raise ValueError(
                "Existing locked selection differs. "
                "Review before replacing it."
            )

    else:

        path.write_text(
            json.dumps(selection, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    print("\n=== LOCKED MODEL SELECTION ===")

    for candidate in candidates:

        print(
            f"Rank {candidate['rank']}: "
            f"{candidate['run_name']} | "
            f"Threshold={candidate['decision_threshold']!r}"
        )

    print("Selection saved:", path)

    return selection


def load_test_data(selection, data_dir=None):
    """Load the original Safe test split and verify its identity."""

    path = mu.dataset_path("safe", data_dir)
    actual_hash = mu.file_sha256(path)

    for candidate in selection["models"]:

        if candidate["dataset_sha256"] != actual_hash:
            raise ValueError("Safe dataset SHA-256 mismatch.")

    df = mu.load_feature_data(path)
    features = mu.get_feature_columns(df)

    if len(features) != 31:
        raise ValueError("Expected 31 Safe features.")

    labels = mu.create_split_labels(df)
    test = df.loc[labels == "Test"]

    X = test[features].copy()
    y = test["is_fraud"].to_numpy()

    if len(y) != 144122 or int(y.sum()) != 17835:
        raise ValueError("Unexpected test dataset counts.")

    print("\nTest orders:", len(y))
    print("Test fraud orders:", int(y.sum()))
    print("Test feature count:", len(features))

    return X, y


def verify_original_test_metrics(row, y, probabilities):

    original = mu.evaluate(
        y, probabilities, float(row["decision_threshold"])
    )

    for metric in METRICS:

        key = (
            "average_precision"
            if metric == "pr_auc"
            else metric
        )

        if not np.isclose(
            original[key],
            float(row[f"test_{metric}"]),
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{row['run_name']}: original test "
                f"{metric} does not match MLflow."
            )


def evaluate_selected_models(selection, inventory, X_test, y_test):

    results = []

    for candidate in selection["models"]:

        print("\nEvaluating:", candidate["run_name"])

        original = inventory.loc[
            inventory["run_id"] == candidate["run_id"]
        ]

        if len(original) != 1:
            raise ValueError("Selected source run not found.")

        original = original.iloc[0]

        if (
            original["model_uri"] != candidate["model_uri"]
            or original["dataset_version"] != "safe"
            or not original["selection_candidate"]
            or original["dataset_sha256"] != candidate["dataset_sha256"]
            or original["run_name"] != candidate["run_name"]
        ):
            raise ValueError("Selected run metadata mismatch.")

        X = X_test.copy()

        if candidate["team_member"] != "abby":
            X = X.astype("float32")

        with tempfile.TemporaryDirectory() as temp_dir:

            model = mlflow.sklearn.load_model(
                candidate["model_uri"],
                dst_path=temp_dir,
            )

            if model.n_features_in_ != 31:
                raise ValueError("Model feature count mismatch.")

            if list(model.classes_) != [0, 1]:
                raise ValueError("Unexpected model classes.")

            if hasattr(model, "feature_names_in_"):

                model_features = list(model.feature_names_in_)

                if set(model_features) != set(X.columns):
                    raise ValueError("Feature names mismatch.")

                X = X[model_features]

            probabilities, inference_seconds = (
                mu.timed_predict_proba(model, X)
            )

            probabilities = np.asarray(
                probabilities, dtype=np.float64
            )

            if (
                probabilities.shape != (len(y_test),)
                or not np.isfinite(probabilities).all()
                or not (
                    (probabilities >= 0)
                    & (probabilities <= 1)
                ).all()
            ):
                raise ValueError("Invalid model probabilities.")

            verify_original_test_metrics(
                original, y_test, probabilities
            )

            print("Original test metrics: PASS")

            metrics = mu.evaluate(
                y_test,
                probabilities,
                candidate["decision_threshold"],
            )

            result = {
                **candidate,
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_pr_auc": metrics["average_precision"],
                "test_roc_auc": metrics["roc_auc"],
                "test_tp": metrics["tp"],
                "test_fp": metrics["fp"],
                "test_fn": metrics["fn"],
                "test_tn": metrics["tn"],
                "test_alerts": metrics["tp"] + metrics["fp"],
                "test_inference_seconds": inference_seconds,
                "test_precision_target_achieved": (
                    metrics["precision"] >= PRIMARY_TARGET
                ),
                "precision_delta": (
                    metrics["precision"]
                    - candidate["val_precision"]
                ),
                "recall_delta": (
                    metrics["recall"]
                    - candidate["val_recall"]
                ),
                "pr_auc_delta": (
                    metrics["average_precision"]
                    - candidate["val_pr_auc"]
                ),
            }

            results.append(result)

            del model

    return pd.DataFrame(results)


def main():

    parser = argparse.ArgumentParser(
        description="Lock and evaluate the top two Safe models."
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

    inventory = extract_model_metrics(
        fetch_official_runs(client, args.experiment)
    )

    comparison = pd.read_csv(
        args.output_dir / "safe_model_comparison.csv",
        float_precision="round_trip",
    )

    top_two = select_top_two(comparison)

    selection = lock_selection(
        top_two, args.output_dir
    )

    X_test, y_test = load_test_data(
        selection, args.data_dir
    )

    results = evaluate_selected_models(
        selection, inventory, X_test, y_test
    )

    results.to_csv(
        args.output_dir / "top_two_test_results.csv",
        index=False,
    )

    print("\n=== LOCKED THRESHOLD: TEST RESULTS ===")

    print(
        results[[
            "rank",
            "algorithm",
            "decision_threshold",
            "val_precision",
            "test_precision",
            "val_recall",
            "test_recall",
            "test_f1",
            "test_pr_auc",
            "test_fp",
            "test_alerts",
            "test_precision_target_achieved",
        ]].round(4).to_string(index=False)
    )

    print("\n=== GENERALISATION ===")

    print(
        results[[
            "algorithm",
            "precision_delta",
            "recall_delta",
            "pr_auc_delta",
            "test_inference_seconds",
        ]].round(4).to_string(index=False)
    )

    print("\nTop two selection: LOCKED")
    print("Original test metric verification: PASS")
    print("Test evaluation: PASS")
    print("Model selection unchanged after test evaluation.")
    print("Output directory:", args.output_dir)


if __name__ == "__main__":
    main()
