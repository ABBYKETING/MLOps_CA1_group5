"""Task #51: verify an execution's recorded MLflow runs, metrics and artifacts.

Reads the existing server records and downloads only small metadata artifacts.
It does not retrain models, create new runs or modify previous experiments.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import mlflow
import requests

SOURCE_DIR = str(Path(__file__).resolve().parent)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from retraining_tracking import resolve_tracking_uri, safe_error


def verify_logged_execution(summary_path, *, tracking_uri=None, local_mlflow=False):
    path = Path(summary_path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("mlflow", {}).get("status") != "completed":
        raise ValueError("Choose a completed retraining execution with completed MLflow logging")
    info = report["mlflow"]
    # The configured destination is authoritative; a report file must not choose
    # where the user's authentication credentials are sent.
    uri = resolve_tracking_uri(tracking_uri, local=local_mlflow)
    if uri != info["tracking_uri"]:
        raise ValueError("Verification URI must match the execution's tracking server")
    client = mlflow.MlflowClient(tracking_uri=uri)
    parent = client.get_run(info["parent_run_id"])
    if parent.info.status != "FINISHED" or parent.data.tags.get("pipeline_run_id") != report["run_id"]:
        raise ValueError("Pipeline run identity/status does not match the local report")
    if "execution/summary.json" not in {item.path for item in client.list_artifacts(parent.info.run_id, "execution")}:
        raise ValueError("Pipeline execution summary artifact is missing")
    auth = None
    if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        auth = (os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"])
    response = requests.get(uri + "/", auth=auth, timeout=20)
    response.raise_for_status()
    if "mlflow" not in response.text.lower():
        raise ValueError("Tracking address did not serve the MLflow UI HTML")
    child_runs = client.search_runs([info["experiment_id"]],
                                    filter_string=f"tags.mlflow.parentRunId = '{info['parent_run_id']}'")
    expected_ids = {info["parent_run_id"]}
    verified = []
    with tempfile.TemporaryDirectory() as temporary:
        for task in report["tasks"]:
            if task["status"] != "completed" or len(task["results"]) != 1:
                raise ValueError("Every requested Safe model task must be completed")
            result = task["results"][0]
            run_id = result["mlflow_run_id"]
            if run_id in expected_ids:
                raise ValueError("Model tasks unexpectedly reused an MLflow run ID")
            expected_ids.add(run_id)
            stored = client.get_run(run_id)
            tags = stored.data.tags
            if (stored.info.status != "FINISHED" or tags.get("mlflow.parentRunId") != info["parent_run_id"]
                    or tags.get("pipeline_run_id") != report["run_id"]
                    or tags.get("team_member") != task["member"]
                    or tags.get("model_key") != task["model_key"]
                    or tags.get("dataset_version") != "safe"):
                raise ValueError(f"Run identity/status mismatch for {task['model_key']}")
            if stored.data.params.get("dataset_sha256") != report["input"]["sha256"]:
                raise ValueError(f"Dataset checksum mismatch for {task['model_key']}")
            for key, value in result["params"].items():
                representations = {str(value)}
                if isinstance(value, list):
                    # Tuple parameters become JSON lists in the local summary.
                    representations.add(str(tuple(value)))
                if stored.data.params.get(key) not in representations:
                    raise ValueError(f"Parameter mismatch for {task['model_key']}: {key}")
            if abs(float(stored.data.params["decision_threshold"]) - result["threshold"]) > 1e-12:
                raise ValueError(f"Decision threshold mismatch for {task['model_key']}")
            for split, field in (("val", "validation"), ("test", "test")):
                for metric in ("accuracy", "precision", "recall", "f1", "average_precision", "roc_auc",
                               "tp", "tn", "fp", "fn"):
                    name = f"{split}_{'pr_auc' if metric == 'average_precision' else metric}"
                    actual = stored.data.metrics.get(name)
                    if actual is None or abs(actual - result[field][metric]) > 1e-9:
                        raise ValueError(f"Metric mismatch for {task['model_key']}: {name}")
            evaluation = {item.path for item in client.list_artifacts(run_id, "evaluation")}
            required_plots = {f"evaluation/{split}_{plot}.png" for split in ("val", "test")
                              for plot in ("confusion_matrix", "pr_curve")}
            if not required_plots.issubset(evaluation):
                raise ValueError(f"Missing evaluation artifacts for {task['model_key']}")
            metadata_paths = {item.path for item in client.list_artifacts(run_id, "metadata")}
            if not {"metadata/features.json", "metadata/input.json", "metadata/result.json"}.issubset(metadata_paths):
                raise ValueError(f"Missing metadata artifacts for {task['model_key']}")
            if "feature_importance.csv" not in {item.path for item in client.list_artifacts(run_id)}:
                raise ValueError(f"Missing feature importance for {task['model_key']}")
            model_uri = result["model_uri"]
            if tags.get("logged_model_uri") != model_uri or not model_uri.startswith("models:/m-"):
                raise ValueError(f"Missing/mismatched MLflow 3 trained model for {task['model_key']}")
            model = client.get_logged_model(model_uri.removeprefix("models:/"))
            if model.source_run_id != run_id:
                raise ValueError(f"Trained model belongs to another run for {task['model_key']}")
            # Download the model descriptor, without executing a deserialized model.
            model_file = mlflow.artifacts.download_artifacts(
                artifact_uri=model.artifact_location.rstrip("/") + "/MLmodel",
                tracking_uri=uri, dst_path=str(Path(temporary) / run_id))
            if "sklearn" not in Path(model_file).read_text(encoding="utf-8"):
                raise ValueError(f"Invalid model descriptor for {task['model_key']}")
            verified.append({"model_key": task["model_key"], "run_id": run_id,
                             "model_uri": model_uri, "run_url": result["mlflow_run_url"]})
    if {parent.info.run_id, *[run.info.run_id for run in child_runs]} != expected_ids:
        raise ValueError("Server records for this execution differ from the expected pipeline/model runs")
    evidence = {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
                "pipeline_run_id": report["run_id"], "tracking_uri": uri,
                "tracking_location": info["tracking_location"], "ui_html_accessible": True,
                "visual_ui_confirmation": "Open the generated run URLs in the browser to confirm visibility",
                "parent_run_id": info["parent_run_id"], "parent_run_url": info["parent_run_url"],
                "models_verified": verified}
    (path.parent / "mlflow_verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--local-mlflow", action="store_true")
    args = parser.parse_args()
    try:
        evidence = verify_logged_execution(args.summary, tracking_uri=args.tracking_uri,
                                           local_mlflow=args.local_mlflow)
    except Exception as error:
        print(f"MLflow verification failed: {type(error).__name__}: {safe_error(error)}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
