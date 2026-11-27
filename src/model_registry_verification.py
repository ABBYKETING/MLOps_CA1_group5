"""Verify registered model predictions against the original logged models."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np

import model_utils as mu

from model_registry import (
    REGISTERED_MODEL_NAME,
    DEFAULT_MANIFEST,
    load_manifest,
    validate_source_models,
    plan_registration,
    verify_version,
)

from model_comparison import DEFAULT_EXPERIMENT
from model_selection import load_test_data


DEFAULT_OUTPUT_DIR = Path("outputs/model_registry")


def predict_model(model_uri, X):
    """Load a model and return verified fraud probabilities."""

    with tempfile.TemporaryDirectory() as temp_dir:

        model = mlflow.sklearn.load_model(
            model_uri,
            dst_path=temp_dir,
        )

        if model.n_features_in_ != 31:
            raise ValueError("Unexpected model feature count.")

        if list(model.classes_) != [0, 1]:
            raise ValueError("Unexpected model classes.")

        X_input = X

        if hasattr(model, "feature_names_in_"):

            features = list(model.feature_names_in_)

            if set(features) != set(X.columns):
                raise ValueError("Model feature names mismatch.")

            X_input = X[features]

        probabilities, inference_seconds = (
            mu.timed_predict_proba(model, X_input)
        )

        probabilities = np.asarray(
            probabilities, dtype=np.float64
        )

        if (
            probabilities.shape != (len(X),)
            or not np.isfinite(probabilities).all()
            or not (
                (probabilities >= 0)
                & (probabilities <= 1)
            ).all()
        ):
            raise ValueError("Invalid predicted probabilities.")

        del model

    return probabilities, inference_seconds


def verify_predictions(original, registered, threshold):
    """Ensure probabilities and classification decisions match."""

    if original.shape != registered.shape:
        raise ValueError("Prediction shape mismatch.")

    if not np.allclose(
        original,
        registered,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Registered model probabilities differ from source."
        )

    original_predictions = (
        original >= threshold
    ).astype(int)

    registered_predictions = (
        registered >= threshold
    ).astype(int)

    if not np.array_equal(
        original_predictions,
        registered_predictions,
    ):
        raise ValueError(
            "Registered model classification decisions differ."
        )

    return float(
        np.max(np.abs(original - registered))
    )


def verify_registered_models(
    client,
    selection,
    X_test,
    y_test,
    use_aliases=False,
):

    name = REGISTERED_MODEL_NAME

    plan = plan_registration(
        client, name, selection
    )

    if len(plan) != 2 or any(
        version is None for _, version in plan
    ):
        raise ValueError(
            "Both selected models must be registered."
        )

    results = []

    for candidate, version in plan:

        verify_version(client, version, candidate)

        role = candidate["proposed_role"]

        if use_aliases:

            resolved = client.get_model_version_by_alias(
                name, role
            )

            if str(resolved.version) != str(version.version):
                raise ValueError(
                    f"Alias {role} points to the wrong version."
                )

            verify_version(client, resolved, candidate)

            registry_uri = f"models:/{name}@{role}"

        else:

            registry_uri = (
                f"models:/{name}/{version.version}"
            )

        threshold = float(
            version.tags["decision_threshold"]
        )

        if threshold != candidate["decision_threshold"]:
            raise ValueError("Registered threshold mismatch.")

        X = X_test.copy()

        # Preserve the original training input representation.
        if candidate["team_member"] != "abby":
            X = X.astype("float32")

        print("\n=== MODEL VERIFICATION ===")
        print("Algorithm:", candidate["algorithm"])
        print("Registered version:", version.version)
        print("Proposed role:", role)
        print("Original URI:", candidate["model_uri"])
        print("Registry URI:", registry_uri)
        print("Threshold:", repr(threshold))

        source_prob, source_seconds = predict_model(
            candidate["model_uri"], X
        )

        registry_prob, registry_seconds = predict_model(
            registry_uri, X
        )

        max_difference = verify_predictions(
            source_prob,
            registry_prob,
            threshold,
        )

        source_metrics = mu.evaluate(
            y_test, source_prob, threshold
        )

        registry_metrics = mu.evaluate(
            y_test, registry_prob, threshold
        )

        for key, value in source_metrics.items():

            if not np.isclose(
                value,
                registry_metrics[key],
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Evaluation metric mismatch: {key}"
                )

        result = {
            "registered_model": name,
            "version": str(version.version),
            "role": role,
            "algorithm": candidate["algorithm"],
            "team_member": candidate["team_member"],
            "source_run_id": candidate["run_id"],
            "source_model_uri": candidate["model_uri"],
            "registry_uri": registry_uri,
            "decision_threshold": threshold,
            "dataset_sha256": candidate["dataset_sha256"],
            "test_orders": len(y_test),
            "production_eligible": False,
            "max_probability_difference": max_difference,
            "probabilities_match": True,
            "classifications_match": True,
            "metrics_match": True,
            "source_inference_seconds": source_seconds,
            "registry_inference_seconds": registry_seconds,
            "test_metrics": {
                key: (
                    int(value)
                    if key in {"tp", "tn", "fp", "fn"}
                    else float(value)
                )
                for key, value in registry_metrics.items()
            },
        }

        results.append(result)

        print("Test predictions:", len(registry_prob))
        print("Max probability difference:", max_difference)
        print("Probability consistency: PASS")
        print("Classification consistency: PASS")
        print("Metric consistency: PASS")
        print(
            "Test precision:",
            f"{registry_metrics['precision']:.6f}",
        )
        print(
            "Test recall:",
            f"{registry_metrics['recall']:.6f}",
        )
        print(
            "Test AP:",
            f"{registry_metrics['average_precision']:.6f}",
        )

    return results


def main():

    parser = argparse.ArgumentParser(
        description="Verify registered CA1 fraud models."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
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

    parser.add_argument(
        "--aliases",
        action="store_true",
        help="Load models through champion/challenger aliases.",
    )

    args = parser.parse_args()

    if (
        not args.tracking_uri
        or not args.tracking_uri.startswith("https://")
    ):
        parser.error("Shared HTTPS MLflow tracking URI required.")

    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI")

    if registry_uri and registry_uri != args.tracking_uri:
        parser.error("Tracking and registry URI must match.")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)

    client = mlflow.MlflowClient()

    selection = load_manifest(args.manifest)

    validate_source_models(
        client,
        selection,
        args.experiment,
    )

    X_test, y_test = load_test_data(
        selection,
        args.data_dir,
    )

    results = verify_registered_models(
        client,
        selection,
        X_test,
        y_test,
        use_aliases=args.aliases,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        "verification_aliases.json"
        if args.aliases
        else "verification_versions.json"
    )

    report_path = args.output_dir / filename

    report = {
        "verification_mode": (
            "aliases" if args.aliases else "versions"
        ),
        "registered_model": REGISTERED_MODEL_NAME,
        "models_verified": len(results),
        "total_test_predictions": sum(
            result["test_orders"] for result in results
        ),
        "registry_modified": False,
        "results": results,
    }

    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== VERIFICATION SUMMARY ===")
    print("Models verified:", len(results))
    print(
        "Total test predictions:",
        report["total_test_predictions"],
    )
    print("Source lineage verification: PASS")
    print("Prediction consistency: PASS")
    print("Threshold consistency: PASS")
    print("Metric consistency: PASS")
    print("Registry changes: none")
    print("Verification report:", report_path)


if __name__ == "__main__":
    main()
