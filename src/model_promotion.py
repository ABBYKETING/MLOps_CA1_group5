"""Promote the locked CA1 baseline models using MLflow aliases."""

import argparse
import os
from pathlib import Path

import mlflow

from model_comparison import DEFAULT_EXPERIMENT

from model_registry import (
    REGISTERED_MODEL_NAME,
    DEFAULT_MANIFEST,
    load_manifest,
    validate_source_models,
    plan_registration,
    verify_version,
)


def plan_promotion(client, name, selection):
    """Validate registered versions and protect existing aliases."""

    registration_plan = plan_registration(
        client, name, selection
    )

    if len(registration_plan) != 2 or any(
        version is None
        for _, version in registration_plan
    ):
        raise ValueError(
            "Both selected models must be registered."
        )

    registered = client.get_registered_model(name)

    current_aliases = {
        alias: str(version)
        for alias, version in registered.aliases.items()
    }

    promotion_plan = []

    for candidate, version in registration_plan:

        verify_version(client, version, candidate)

        role = candidate["proposed_role"]
        target_version = str(version.version)
        current_version = current_aliases.get(role)

        if (
            current_version is not None
            and current_version != target_version
        ):
            raise ValueError(
                f"Alias {role} already points to "
                f"Version {current_version}. "
                "Baseline promotion will not overwrite it."
            )

        promotion_plan.append({
            "alias": role,
            "version": target_version,
            "run_id": candidate["run_id"],
            "algorithm": candidate["algorithm"],
            "decision_threshold": candidate[
                "decision_threshold"
            ],
            "already_assigned": (
                current_version == target_version
            ),
        })

    return promotion_plan


def promote_models(client, name, promotion_plan, apply=False):
    """Assign missing aliases without creating model versions."""

    for item in promotion_plan:

        alias = item["alias"]
        version = item["version"]

        # Recheck before writing to avoid overwriting a changed alias.
        current = client.get_registered_model(name).aliases
        current_version = current.get(alias)

        if current_version is not None:
            current_version = str(current_version)

        if (
            current_version is not None
            and current_version != version
        ):
            raise ValueError(
                f"Alias {alias} has changed. "
                "Promotion stopped without overwriting it."
            )

        if current_version == version:

            print(
                f"REUSE: @{alias} -> Version {version}"
            )

            continue

        if not apply:

            print(
                f"WOULD ASSIGN: @{alias} -> Version {version}"
            )

            continue

        client.set_registered_model_alias(
            name,
            alias,
            version,
        )

        resolved = client.get_model_version_by_alias(
            name, alias
        )

        if str(resolved.version) != version:
            raise ValueError("Alias assignment mismatch.")

        if resolved.run_id != item["run_id"]:
            raise ValueError("Alias source run mismatch.")

        print(
            f"ASSIGNED: @{alias} -> Version {version}"
        )


def verify_aliases(client, name, selection):
    """Verify both aliases and their complete model metadata."""

    plan = plan_registration(
        client, name, selection
    )

    for candidate, version in plan:

        if version is None:
            raise ValueError("Registered version missing.")

        alias = candidate["proposed_role"]

        resolved = client.get_model_version_by_alias(
            name, alias
        )

        if str(resolved.version) != str(version.version):
            raise ValueError(
                f"Alias {alias} resolves to an unexpected version."
            )

        verify_version(client, resolved, candidate)

        threshold = float(
            resolved.tags["decision_threshold"]
        )

        if threshold != candidate["decision_threshold"]:
            raise ValueError("Alias threshold mismatch.")

        print(f"\nAlias: @{alias}")
        print("Version:", resolved.version)
        print("Algorithm:", resolved.tags["algorithm"])
        print("Source run ID:", resolved.run_id)
        print("Decision threshold:", repr(threshold))
        print(
            "Production eligible:",
            resolved.tags["production_eligible"],
        )
        print("Alias verification: PASS")


def main():

    parser = argparse.ArgumentParser(
        description="Promote the locked CA1 baseline models."
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
        help="Assign aliases. Default is read-only.",
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

    print("Registered model:", REGISTERED_MODEL_NAME)
    print("Mode:", "APPLY" if args.apply else "DRY RUN")

    validate_source_models(
        client,
        selection,
        args.experiment,
    )

    plan = plan_promotion(
        client,
        REGISTERED_MODEL_NAME,
        selection,
    )

    print("\n=== PROMOTION PLAN ===")

    promote_models(
        client,
        REGISTERED_MODEL_NAME,
        plan,
        apply=args.apply,
    )

    if args.apply:

        verify_aliases(
            client,
            REGISTERED_MODEL_NAME,
            selection,
        )

        print("\n=== PROMOTION SUMMARY ===")
        print("Champion promotion: PASS")
        print("Challenger assignment: PASS")
        print("Source lineage verification: PASS")
        print("Threshold metadata verification: PASS")
        print("Production eligibility: false")
        print("New model versions created: 0")

    else:

        print("\nDry run complete.")
        print("Promotion plan: PASS")
        print("Registry changes: none")


if __name__ == "__main__":
    main()
