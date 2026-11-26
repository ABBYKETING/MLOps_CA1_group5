"""PBI 5 task #49: execute all eight Safe model tasks with one command.

Consumes prepared features; RDS preparation, new split rules and ETL triggering
remain tasks #50/#52. Each invocation keeps its own input snapshot and report.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from uuid import uuid4

SOURCE_DIR = str(Path(__file__).resolve().parent)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

import model_utils as mu
from data_processing import SNAPSHOT_FEATURES
from data_split import create_split_labels
from training_interface import MODEL_REGISTRY, run_model

DEFAULT_OUTPUT_DIR = mu.DEFAULT_OUTPUT_DIR / "retraining"


class RetrainingFailed(RuntimeError):
    """Training failed; report and summary_path retain execution evidence."""

    def __init__(self, report, summary_path):
        self.report = report
        self.summary_path = str(summary_path)
        failed = [task["model_key"] for task in report["tasks"] if task["status"] == "failed"]
        detail = ", ".join(failed) if failed else "input preparation"
        reasons = [task["error"] for task in report["tasks"]
                   if task["status"] == "failed" and task.get("error")]
        if not failed and report.get("input", {}).get("error"):
            reasons.append(report["input"]["error"])
        reason = f" {'; '.join(reasons)}." if reasons else ""
        super().__init__(f"Retraining run {report['run_id']} failed: {detail}.{reason} Report: {summary_path}")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save_report(report, path):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _prepare_input(data_dir, run_dir):
    """Freeze and validate one Safe feature file for every model in this run."""
    source = mu.dataset_path("safe", data_dir).resolve()
    before = mu.file_sha256(source)
    input_dir = run_dir / "input"
    input_dir.mkdir()
    snapshot = input_dir / "features_base_safe.csv"
    shutil.copyfile(source, snapshot)
    checksum = mu.file_sha256(snapshot)
    if checksum != before or mu.file_sha256(source) != before:
        raise ValueError("Safe input changed while being copied; retry with a stable prepared dataset")
    frame = mu.load_feature_data(snapshot)
    snapshots = sorted(set(frame.columns).intersection(SNAPSHOT_FEATURES))
    if snapshots:
        raise ValueError(f"Safe input contains excluded snapshot features: {snapshots}")
    labels = create_split_labels(frame)
    counts = {}
    for name in ("Train", "Validation", "Test"):
        y = frame.loc[labels == name, mu.TARGET]
        if y.nunique() != 2:
            raise ValueError(f"{name} must contain both classes under the existing chronological split")
        counts[name] = len(y)
    return input_dir, {
        "status": "completed", "source": str(source), "snapshot": str(snapshot),
        "sha256": checksum, "rows": counts, "feature_count": len(mu.get_feature_columns(frame)),
    }


def run_retraining(*, data_dir=None, output_dir=DEFAULT_OUTPUT_DIR,
                   model_keys=None, run_id=None, sample_frac=None,
                   stop_on_error=False, use_mlflow=False,
                   experiment=mu.DEFAULT_EXPERIMENT, tracking_uri=None):
    """Run all registered Safe models, returning a completed execution report.

    All eight algorithms run by default. model_keys optionally selects models
    for targeted checks. Model failures are recorded and remaining tasks still
    run, unless stop_on_error=True. Any failed input/model task makes the run
    fail and raises RetrainingFailed after saving summary.json. CLI exit codes:
    0 success, 1 execution failure, 2 invalid arguments.

    No fitted models are retained in memory across tasks: artifacts are saved by
    task #48. A fresh directory is required for each run_id to avoid overwrites.
    """
    if isinstance(model_keys, str):
        raise ValueError("model_keys must be a sequence, not a string")
    keys = list(MODEL_REGISTRY) if model_keys is None else list(model_keys)
    if not keys or len(set(keys)) != len(keys) or set(keys) - set(MODEL_REGISTRY):
        raise ValueError("model_keys must be nonempty, unique registered model keys")
    if sample_frac is not None:
        if (isinstance(sample_frac, bool) or not isinstance(sample_frac, (int, float))
                or not math.isfinite(sample_frac) or not 0 < sample_frac <= 1):
            raise ValueError("sample_frac must be a finite number in (0, 1]")
        if use_mlflow:
            raise ValueError("Sampled development runs must disable MLflow")
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:12]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id):
        raise ValueError("run_id must start with a letter/digit and contain only letters, digits, _, -, . (max 100)")
    root = Path(output_dir).resolve()
    run_dir = root / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"Run directory already exists; choose a new run_id: {run_dir}") from error
    summary_path = run_dir / "summary.json"
    started = time.perf_counter()
    report = {
        "run_id": run_id, "status": "running", "started_at": _now(), "finished_at": None,
        "dataset_type": "safe", "sample_frac": sample_frac, "use_mlflow": use_mlflow,
        "output_dir": str(run_dir), "input": {"status": "pending"},
        "tasks": [{"model_key": key, "member": MODEL_REGISTRY[key].member,
                   "status": "pending", "started_at": None, "finished_at": None,
                   "duration_seconds": None, "results": [], "error": None} for key in keys],
    }
    _save_report(report, summary_path)
    print(f"Retraining {run_id}: {len(keys)} Safe model tasks", flush=True)
    try:
        input_dir, input_info = _prepare_input(data_dir, run_dir)
        report["input"] = input_info
    except Exception as error:
        error_file = run_dir / "input_error.txt"
        report["input"] = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                           "error_file": str(error_file)}
        error_file.write_text(traceback.format_exc(), encoding="utf-8")
        for task in report["tasks"]:
            task["status"] = "skipped"
            task["skip_reason"] = "Input preparation failed"
        report.update(status="failed", finished_at=_now(), duration_seconds=time.perf_counter() - started)
        _save_report(report, summary_path)
        raise RetrainingFailed(report, summary_path) from error
    _save_report(report, summary_path)

    failed = False
    for index, task in enumerate(report["tasks"]):
        key = task["model_key"]
        if failed and stop_on_error:
            task.update(status="skipped", skip_reason="Stopped after a previous model failure")
            _save_report(report, summary_path)
            continue
        task.update(status="running", started_at=_now())
        _save_report(report, summary_path)
        task_started = time.perf_counter()
        print(f"[{index + 1}/{len(keys)}] {key}: running", flush=True)
        try:
            results, models = run_model(
                key, data_dir=input_dir, output_dir=run_dir / "models",
                sample_frac=sample_frac, use_mlflow=use_mlflow,
                experiment=experiment, tracking_uri=tracking_uri,
            )
            task["results"] = [{k: v for k, v in result.items() if k != "importance"} for result in results]
            task["status"] = "completed"
            del results, models
            print(f"[{index + 1}/{len(keys)}] {key}: completed", flush=True)
        except Exception as error:
            failed = True
            error_file = run_dir / f"{key}_error.txt"
            error_file.write_text(traceback.format_exc(), encoding="utf-8")
            task.update(status="failed", error=f"{type(error).__name__}: {error}", error_file=str(error_file))
            print(f"[{index + 1}/{len(keys)}] {key}: FAILED ({task['error']})", flush=True)
        finally:
            task.update(finished_at=_now(), duration_seconds=time.perf_counter() - task_started)
            _save_report(report, summary_path)
    report.update(status="failed" if failed else "completed", finished_at=_now(),
                  duration_seconds=time.perf_counter() - started)
    _save_report(report, summary_path)
    print(f"Retraining {report['status']}. Report: {summary_path}", flush=True)
    if failed:
        raise RetrainingFailed(report, summary_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--models", nargs="+", choices=MODEL_REGISTRY)
    parser.add_argument("--run-id")
    parser.add_argument("--sample-frac", type=float)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default=mu.DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    try:
        run_retraining(data_dir=args.data_dir, output_dir=args.output_dir, model_keys=args.models,
                       run_id=args.run_id, sample_frac=args.sample_frac, stop_on_error=args.stop_on_error,
                       use_mlflow=args.use_mlflow, tracking_uri=args.tracking_uri, experiment=args.experiment)
    except ValueError as error:
        parser.error(str(error))
    except RetrainingFailed as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
