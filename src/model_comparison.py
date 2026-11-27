"""Compare the SP-buy fraud detection models recorded in shared MLflow."""

import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT = "SP-buy-Fraud-Detection"
DEFAULT_OUTPUT_DIR = Path("outputs/model_comparison")

MODEL_CONFIGS = {
    "abby": ["logistic_regression", "random_forest"],
    "yueliang": ["logistic_regression", "hist_gradient_boosting"],
    "emi": ["decision_tree", "mlp"],
    "chu": ["extra_trees", "lightgbm"],
}

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
]


def expected_run_names():
    """Identify the 16 official model configurations."""
    names = set()

    for member, algorithms in MODEL_CONFIGS.items():
        for algorithm in algorithms:
            for dataset in ["safe", "full"]:
                version = (
                    "v2"
                    if member == "abby" and dataset == "full"
                    else "v1"
                )

                names.add(
                    f"{member}_{algorithm}_{dataset}_{version}"
                )

    return names


def fetch_official_runs(client, experiment_name):
    """Retrieve and validate the official experiment runs."""
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment not found: {experiment_name}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000,
    )

    expected = expected_run_names()

    official = [
        run for run in runs
        if run.info.run_name in expected
        and not run.data.tags.get("pipeline_run_id")
    ]

    actual = {
        run.info.run_name for run in official
    }

    if actual != expected or len(official) != 16:
        raise ValueError(
            "Missing or duplicate official runs. "
            f"Missing: {sorted(expected - actual)}; "
            f"Matched runs: {len(official)}"
        )

    if any(
        run.info.status != "FINISHED"
        for run in official
    ):
        raise ValueError(
            "All official runs must be FINISHED."
        )

    print("Experiment:", experiment.name)
    print("Experiment ID:", experiment.experiment_id)
    print("Official runs:", len(official))
    print("Other active runs:", len(runs) - len(official))

    return official


def extract_model_metrics(runs):
    """Convert MLflow run metadata and metrics into a table."""
    rows = []

    for run in runs:
        tags = run.data.tags
        params = run.data.params
        metrics = run.data.metrics

        dataset = tags["dataset_version"]
        member = tags["team_member"]
        algorithm = tags["algorithm"]
        version = tags["model_version"]

        if dataset not in {"safe", "full"}:
            raise ValueError("Unexpected dataset version")

        if algorithm not in MODEL_CONFIGS.get(member, []):
            raise ValueError("Unexpected model configuration")

        expected_name = (
            f"{member}_{algorithm}_{dataset}_{version}"
        )

        if run.info.run_name != expected_name:
            raise ValueError("Run name and metadata mismatch")

        candidate = tags["selection_candidate"] == "true"

        if candidate != (dataset == "safe"):
            raise ValueError("Invalid selection eligibility")

        if tags["production_eligible"] != "false":
            raise ValueError("Unexpected production eligibility")

        if tags["exploratory"] != str(dataset == "full").lower():
            raise ValueError("Invalid exploratory tag")

        threshold = float(params["decision_threshold"])

        if not 0 <= threshold <= 1:
            raise ValueError("Invalid decision threshold")

        row = {
            "run_name": run.info.run_name,
            "run_id": run.info.run_id,
            "team_member": member,
            "algorithm": algorithm,
            "dataset_version": dataset,
            "model_version": version,
            "model_uri": tags["logged_model_uri"],
            "dataset_sha256": params["dataset_sha256"],
            "feature_count": int(params["feature_count"]),
            "decision_threshold": threshold,
            "threshold_rule": params["threshold_rule"],
            "selection_candidate": candidate,
            "production_eligible": False,
            "exploratory": dataset == "full",
        }

        expected_features = 31 if dataset == "safe" else 41

        if row["feature_count"] != expected_features:
            raise ValueError("Unexpected feature count")

        for split in ["val", "test"]:
            for metric in METRICS:
                key = f"{split}_{metric}"

                if key not in metrics:
                    raise ValueError(
                        f"{run.info.run_name}: missing {key}"
                    )

                value = float(metrics[key])

                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid metric: {key}")

                row[key] = value

        rows.append(row)

    table = pd.DataFrame(rows)

    for dataset in ["safe", "full"]:
        subset = table[
            table["dataset_version"] == dataset
        ]

        if len(subset) != 8:
            raise ValueError("Unexpected dataset run count")

        if subset["dataset_sha256"].nunique() != 1:
            raise ValueError(
                f"Inconsistent {dataset} dataset hashes"
            )

    if table["dataset_sha256"].nunique() != 2:
        raise ValueError("Expected distinct Safe and Full hashes")

    if table["model_uri"].nunique() != 16:
        raise ValueError("Expected 16 unique logged models")

    return table.sort_values(
        ["team_member", "algorithm", "dataset_version"]
    ).reset_index(drop=True)


def compare_safe_full(table):
    """Compare Safe and Full variants for each developer/algorithm."""
    keys = ["team_member", "algorithm"]

    safe = table[table["dataset_version"] == "safe"]
    full = table[table["dataset_version"] == "full"]

    metric_columns = [
        f"{split}_{metric}"
        for split in ["val", "test"]
        for metric in METRICS
    ]

    columns = keys + metric_columns

    comparison = safe[columns].merge(
        full[columns],
        on=keys,
        suffixes=("_safe", "_full"),
        validate="one_to_one",
    )

    for metric in metric_columns:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_full"]
            - comparison[f"{metric}_safe"]
        )

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare official SP-buy MLflow model runs."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if not args.tracking_uri:
        parser.error("A shared MLflow tracking URI is required.")

    if not args.tracking_uri.startswith("https://"):
        parser.error("Use the shared HTTPS MLflow server.")

    client = mlflow.MlflowClient(
        tracking_uri=args.tracking_uri
    )

    runs = fetch_official_runs(
        client,
        args.experiment,
    )

    table = extract_model_metrics(runs)
    comparison = compare_safe_full(table)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        args.output_dir / "all_model_metrics.csv",
        index=False,
    )

    comparison.to_csv(
        args.output_dir / "safe_full_comparison.csv",
        index=False,
    )

    print("\n=== ORIGINAL VALIDATION RESULTS ===")

    safe = table[table["selection_candidate"]]

    print(
        safe[
            [
                "team_member",
                "algorithm",
                "decision_threshold",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_pr_auc",
                "val_roc_auc",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== SAFE VS FULL: TEST AP ===")

    print(
        comparison[
            [
                "team_member",
                "algorithm",
                "test_pr_auc_safe",
                "test_pr_auc_full",
                "test_pr_auc_delta",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nOfficial model inventory: PASS")
    print("Dataset consistency: PASS")
    print("Metadata consistency: PASS")
    print("Metrics export: PASS")
    print("Output directory:", args.output_dir)


if __name__ == "__main__":
    main()
