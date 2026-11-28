"""Task #53: verify saved pipeline evidence and the ETL-only 180-second target.

Reads an existing execution; it does not run ETL, fit models or promote models.
Shared MLflow checks use the configured account and tracking URI.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from automated_pipeline import STAGE_NAMES
from retraining_tracking import safe_error
from training_interface import MODEL_REGISTRY
from verify_retraining_logging import verify_logged_execution


def _duration(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Execution durations must be finite non-negative numbers")
    return value


def verify_pipeline_execution(summary_path, *, verify_shared=True, local_mlflow=False):
    path = Path(summary_path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    stages = report["stages"]
    if report["status"] != "completed" or [s["name"] for s in stages] != list(STAGE_NAMES):
        raise ValueError("Choose a completed integrated pipeline with all four required stages")
    previous = None
    for stage in stages:
        if stage["status"] != "completed":
            raise ValueError("Every required pipeline stage must be completed")
        start, end = (datetime.fromisoformat(stage[k]) for k in ("started_at", "finished_at"))
        if start.tzinfo is None or end.tzinfo is None or end < start or (previous is not None and start < previous):
            raise ValueError("Stage timestamps must be ordered and include timezone information")
        previous = end
        _duration(stage["duration_seconds"])
        if abs((end - start).total_seconds() - stage["duration_seconds"]) > 1:
            raise ValueError("Stage duration differs from its recorded timestamps")
    etl = report["etl"]
    if etl.get("dry_run") is not False or etl["batch_name"] != report["batch_name"]:
        raise ValueError("ETL must confirm an actual load of the requested batch")
    counts = [etl[k] for k in ("labels_inserted", "labels_updated")]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts) or sum(counts) == 0:
        raise ValueError("ETL must contain accepted inserted or updated labels")
    loaded = report["loaded_batch"]
    if (loaded["status"] != "completed" or loaded["batch_name"] != report["batch_name"]
            or loaded["accepted_labels"] != sum(counts)):
        raise ValueError("Committed-batch evidence does not match the ETL result")
    training_path = Path(report["retraining"]["summary_path"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if (training["run_id"] != report["pipeline_run_id"] or training["status"] != "completed"
            or training["mlflow"]["status"] != "completed" or training["dataset_type"] != "safe"
            or not training["from_database"] or training.get("sample_frac") is not None):
        raise ValueError("Training must be a completed, unsampled Safe execution from combined RDS data")
    tasks = training["tasks"]
    if len(tasks) != len(MODEL_REGISTRY) or {t["model_key"] for t in tasks} != set(MODEL_REGISTRY):
        raise ValueError("All eight required models must be included exactly once")
    if any(t["status"] != "completed" or len(t["results"]) != 1 for t in tasks):
        raise ValueError("All eight Safe model tasks must complete successfully")
    info = training["input"]
    if info["status"] != "completed" or info["feature_count"] != 31:
        raise ValueError("Expected the completed 31-feature Safe snapshot")
    with Path(info["snapshot"]).open("rb") as snapshot:
        actual = hashlib.file_digest(snapshot, "sha256").hexdigest()
    if actual != info["sha256"]:
        raise ValueError("The prepared input snapshot checksum has changed")
    shared = "not_checked"
    if verify_shared:
        check = verify_logged_execution(training_path, local_mlflow=local_mlflow)
        if check["status"] != "passed" or len(check["models_verified"]) != 8:
            raise ValueError("Shared model read-back did not pass for all eight models")
        shared = "passed"
    seconds = stages[1]["duration_seconds"]
    evidence = {
        "status": ("passed" if verify_shared else "local_checks_passed") if seconds <= 180 else "performance_failed",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_id": report["pipeline_run_id"], "pipeline_summary": str(path),
        "etl_duration_seconds": seconds, "etl_limit_seconds": 180,
        "etl_within_three_minutes": seconds <= 180,
        "total_pipeline_seconds": _duration(report["duration_seconds"]),
        "performance_scope": "ETL stage; total pipeline time is reported separately",
        "models_completed": len(tasks), "safe_features": info["feature_count"],
        "partition_rows": info["rows"], "snapshot_sha256": actual,
        "shared_mlflow_read_back": shared,
    }
    (path.parent / "pipeline_verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--skip-mlflow", action="store_true", help="Only local evidence checks; not a shared-server pass")
    parser.add_argument("--local-mlflow", action="store_true", help="Explicit isolated loopback verification")
    args = parser.parse_args()
    try:
        evidence = verify_pipeline_execution(args.summary, verify_shared=not args.skip_mlflow,
                                             local_mlflow=args.local_mlflow)
    except Exception as error:
        print(f"Pipeline verification failed: {type(error).__name__}: {safe_error(error)}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2))
    return 1 if evidence["status"] == "performance_failed" else 0


if __name__ == "__main__":
    sys.exit(main())
