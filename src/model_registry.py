"""Register the locked CA1 baseline models in shared MLflow."""

import argparse
import json
import math
import os
from pathlib import Path

import mlflow
from mlflow.entities import LoggedModelStatus

from model_comparison import (
    DEFAULT_EXPERIMENT,
    fetch_official_runs,
    extract_model_metrics,
)


REGISTERED_MODEL_NAME = "SP-buy-Fraud-Detector"

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "top_two_models.json"
)


def load_manifest(path):
    """Read and validate the locked baseline selection."""

    selection = json.loads(
        Path(path).read_text(encoding="utf-8-sig")
    )

    if selection["selection_basis"] != "validation_only":
        raise ValueError("Unexpected selection methodology.")

    if selection["precision_target"] != 0.70:
        raise ValueError("Unexpected baseline precision policy.")

    if selection["production_eligible"] is not False:
        raise ValueError("Production eligibility must be false.")

    candidates = selection["models"]

    if len(candidates) != 2:
        raise ValueError("Expected exactly two selected models.")

    if len({m["run_id"] for m in candidates}) != 2:
        raise ValueError("Duplicate source run IDs.")

    if len({m["model_uri"] for m in candidates}) != 2:
        raise ValueError("Duplicate logged model URIs.")

    for rank, candidate in enumerate(candidates, start=1):

        expected_role = (
            "champion" if rank == 1 else "challenger"
        )

        if candidate["rank"] != rank:
            raise ValueError("Unexpected candidate ranking.")

        if candidate["proposed_role"] != expected_role:
            raise ValueError("Unexpected candidate role.")

        if candidate["dataset_version"] != "safe":
            raise ValueError("Only Safe models can be registered.")

        if candidate["production_eligible"] is not False:
            raise ValueError("Model is not approved for production.")

        threshold = candidate["decision_threshold"]

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("Invalid decision threshold.")

        if (
            not math.isfinite(candidate["val_precision"])
            or not 0.70 <= candidate["val_precision"] <= 1
            or not math.isfinite(candidate["val_recall"])
            or not 0 < candidate["val_recall"] <= 1
        ):
            raise ValueError("Invalid validation performance.")

        if not candidate["model_uri"].startswith("models:/m-"):
            raise ValueError("Expected an MLflow 3 logged model URI.")

    return selection


def validate_source_models(client, selection, experiment):
    """Verify both source runs and logged models before registration."""

    inventory = extract_model_metrics(
        fetch_official_runs(client, experiment)
    )

    for candidate in selection["models"]:

        matches = inventory.loc[
            inventory["run_id"] == candidate["run_id"]
        ]

        if len(matches) != 1:
            raise ValueError("Selected source run not found.")

        original = matches.iloc[0]

        for key in [
            "run_name",
            "model_uri",
            "team_member",
            "algorithm",
            "dataset_version",
            "dataset_sha256",
        ]:

            if original[key] != candidate[key]:
                raise ValueError(f"Source metadata mismatch: {key}")

        if not original["selection_candidate"]:
            raise ValueError("Source model is not a selection candidate.")

        if original["production_eligible"]:
            raise ValueError("Unexpected production eligibility.")

        if original["exploratory"]:
            raise ValueError("Exploratory model cannot be registered.")

        if not math.isclose(
            float(original["decision_threshold"]),
            candidate["original_threshold"],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Original threshold mismatch.")

        model_id = candidate["model_uri"].removeprefix("models:/")

        logged = client.get_logged_model(model_id)

        if logged.status != LoggedModelStatus.READY:
            raise ValueError("Logged model is not READY.")

        if logged.source_run_id != candidate["run_id"]:
            raise ValueError("Logged model source run mismatch.")

        if str(logged.experiment_id) != "2":
            raise ValueError("Unexpected source experiment ID.")

        print("\nSource model:", candidate["run_name"])
        print("Run ID:", candidate["run_id"])
        print("Logged model ID:", model_id)
        print("Status: READY")
        print("Dataset: safe")
        print("Decision threshold:", repr(candidate["decision_threshold"]))
        print("Source validation: PASS")


def version_tags(candidate):
    """Preserve the selection decision as model version metadata."""

    return {
        "team_member": candidate["team_member"],
        "algorithm": candidate["algorithm"],
        "dataset_version": candidate["dataset_version"],
        "dataset_sha256": candidate["dataset_sha256"],
        "source_run_id": candidate["run_id"],
        "source_model_uri": candidate["model_uri"],
        "source_run_name": candidate["run_name"],
        "selection_rank": str(candidate["rank"]),
        "proposed_role": candidate["proposed_role"],
        "selection_basis": "validation_only",
        "selection_policy": "max_recall_at_min_precision",
        "precision_target": "0.7",
        "decision_threshold": repr(candidate["decision_threshold"]),
        "original_decision_threshold": repr(
            candidate["original_threshold"]
        ),
        "val_precision_at_selected_threshold": repr(
            candidate["val_precision"]
        ),
        "val_recall_at_selected_threshold": repr(
            candidate["val_recall"]
        ),
        "val_f1_at_selected_threshold": repr(
            candidate["val_f1"]
        ),
        "val_average_precision": repr(candidate["val_pr_auc"]),
        "selection_candidate": "true",
        "production_eligible": "false",
        "exploratory": "false",
        "registration_scope": "ca1_baseline_demo",
    }


def existing_versions(client, name):
    """Retrieve existing versions for duplicate protection."""

    versions = client.search_model_versions(
        filter_string=f"name = '{name}'",
        max_results=100,
    )

    if getattr(versions, "token", None):
        raise ValueError("Registry contains additional version pages.")

    return [
        client.get_model_version(name, version.version)
        for version in versions
    ]


def verify_version(client, version, candidate):
    """Check registered model identity and required version tags."""

    expected_id = candidate["model_uri"].removeprefix("models:/")

    # The registry may omit the optional model_id field.
    # Reject conflicting IDs and verify the actual source lineage.
    if version.model_id not in (None, expected_id):
        raise ValueError("Registered model ID mismatch.")

    if version.source != candidate["model_uri"]:
        raise ValueError("Registered model source URI mismatch.")

    logged = client.get_logged_model(expected_id)

    if logged.status != LoggedModelStatus.READY:
        raise ValueError("Source logged model is not READY.")

    if logged.source_run_id != candidate["run_id"]:
        raise ValueError("Logged model source run mismatch.")

    download_uri = client.get_model_version_download_uri(
        version.name, version.version
    )

    if download_uri != logged.artifact_location:
        raise ValueError("Registered model artifact location mismatch.")

    if version.run_id != candidate["run_id"]:
        raise ValueError("Registered model source run mismatch.")

    if version.status != "READY":
        raise ValueError("Registered version is not READY.")

    for key, value in version_tags(candidate).items():

        if version.tags.get(key) != value:
            raise ValueError(f"Version tag mismatch: {key}")


def plan_registration(client, name, selection):
    """Reuse matching baseline versions; reject unexpected versions."""

    versions = existing_versions(client, name)

    candidates = selection["models"]
    expected_sources = {
        c["model_uri"] for c in candidates
    }

    if any(v.source not in expected_sources for v in versions):
        raise ValueError(
            "Unexpected registered versions found. "
            "This baseline script will not modify them."
        )

    plan = []

    for candidate in candidates:

        matches = [
            v for v in versions
            if v.source == candidate["model_uri"]
        ]

        if len(matches) > 1:
            raise ValueError("Duplicate registered versions found.")

        existing = matches[0] if matches else None

        if existing is not None:
            verify_version(client, existing, candidate)

        plan.append((candidate, existing))

    return plan


def register_models(client, name, plan, apply=False):
    """Register missing models, or preview the planned changes."""

    registered = []

    for candidate, existing in plan:

        if existing is not None:

            print(
                f"REUSE: {candidate['run_name']} "
                f"-> Version {existing.version}"
            )

            registered.append(existing)
            continue

        if not apply:

            print(
                f"WOULD REGISTER: {candidate['run_name']}"
            )

            continue

        created = mlflow.register_model(
            model_uri=candidate["model_uri"],
            name=name,
            await_registration_for=300,
            tags=version_tags(candidate),
        )

        version = client.get_model_version(
            name, created.version
        )

        verify_version(client, version, candidate)

        registered.append(version)

        print(
            f"REGISTERED: {candidate['run_name']} "
            f"-> Version {version.version}"
        )

    return registered


def main():

    parser = argparse.ArgumentParser(
        description="Register the two locked CA1 baseline models."
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

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply registration changes. Default is read-only.",
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

    experiment = client.get_experiment_by_name(args.experiment)

    if experiment is None or str(experiment.experiment_id) != "2":
        raise ValueError("Expected shared baseline experiment ID 2.")

    selection = load_manifest(args.manifest)

    print("Registered model name:", REGISTERED_MODEL_NAME)
    print("Mode:", "APPLY" if args.apply else "DRY RUN")

    validate_source_models(
        client,
        selection,
        args.experiment,
    )

    plan = plan_registration(
        client,
        REGISTERED_MODEL_NAME,
        selection,
    )

    print("\n=== REGISTRATION PLAN ===")

    versions = register_models(
        client,
        REGISTERED_MODEL_NAME,
        plan,
        apply=args.apply,
    )

    if args.apply:

        if len(versions) != 2:
            raise ValueError("Expected two registered versions.")

        print("\nRegistered versions verified:", len(versions))

    else:

        print("\nDry run complete. No registry changes made.")

    print("Source verification: PASS")
    print("Registration plan: PASS")
    print("Alias changes: none")


if __name__ == "__main__":
    main()
