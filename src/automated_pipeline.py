"""Task #52: CSV URLs -> PBI 4 ETL/RDS -> Safe retraining -> shared MLflow.

One invocation runs the stages in order. Retraining always reads the committed
combined RDS state, never the raw/new CSV alone. Registered models and aliases
are not changed. Connection credentials come from the existing DB_*/MLFLOW_*
environment variables.
"""

import argparse
from datetime import datetime, timezone
import json
from numbers import Integral
from pathlib import Path
import re
import sys
import time
import traceback
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from sqlalchemy import text

SOURCE_DIR = str(Path(__file__).resolve().parent)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

import database as db
import etl_pipeline as etl
import model_utils as mu
from retraining_tracking import RetrainingTracker, safe_error
from retraining_workflow import RetrainingFailed, run_retraining
from training_interface import MODEL_REGISTRY

DEFAULT_OUTPUT_DIR = mu.DEFAULT_OUTPUT_DIR / "pipeline"
STAGE_NAMES = ("preflight", "etl", "load_validation", "retraining")


class PipelineFailed(RuntimeError):
    def __init__(self, report, summary_path):
        self.report = report
        self.summary_path = str(summary_path)
        failed = next(stage for stage in report["stages"] if stage["status"] == "failed")
        super().__init__(f"Pipeline {report['pipeline_run_id']} failed at {failed['name']}: "
                         f"{failed['error']}. Report: {summary_path}")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(report, path):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _safe_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _clean_error(error, sources):
    message = safe_error(error)
    for url in sources.values():
        message = message.replace(url, _safe_url(url))
    return message


def _validate_sources(sources, batch_name):
    if not isinstance(sources, dict) or set(sources) != set(etl.FILE_KINDS):
        raise ValueError("sources must contain exactly customers, orders and labels URLs")
    for kind, url in sources.items():
        if not isinstance(url, str):
            raise ValueError(f"{kind} source must be an HTTP(S) URL")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"{kind} source must be an HTTP(S) URL")
        if parts.username is not None or parts.password is not None:
            raise ValueError("Provide public CSV URLs without embedded usernames/passwords")
    if batch_name is None:
        names = {etl.batch_name_from_source(url) for url in sources.values()}
        if len(names) != 1:
            raise ValueError("CSV postfixes differ; provide --batch-name explicitly")
        batch_name = names.pop()
    if not isinstance(batch_name, str) or batch_name in (".", ".."):
        raise ValueError("Invalid batch name")
    return etl.check_batch_name(batch_name)


def _check_loaded_batch(summary, batch_name, engine):
    if not isinstance(summary, dict) or summary.get("dry_run") is not False:
        raise ValueError("ETL did not confirm an actual database load; retraining is blocked")
    if summary.get("batch_name") != batch_name:
        raise ValueError("ETL returned a different batch identifier; retraining is blocked")
    counts = [summary.get(key) for key in ("labels_inserted", "labels_updated")]
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in counts):
        raise ValueError("ETL did not return valid accepted-label counts")
    with engine.connect() as connection:
        status = connection.execute(text("SELECT status FROM etl_batches WHERE batch_name = :batch"),
                                    {"batch": batch_name}).scalar_one_or_none()
    if status != "completed":
        raise ValueError(f"RDS batch status is {status!r}, not completed; retraining is blocked")
    if sum(counts) == 0:
        raise ValueError("No valid labelled records were loaded from this batch; retraining is blocked")
    return {"batch_name": batch_name, "status": status, "accepted_labels": int(sum(counts))}


def _training_reference(report):
    return {"run_id": report["run_id"], "status": report["status"],
            "summary_path": str(Path(report["output_dir"]) / "summary.json"),
            "mlflow": report.get("mlflow", {}),
            "models": [{"model_key": task["model_key"], "status": task["status"]}
                       for task in report["tasks"]]}


def run_pipeline(sources, *, batch_name=None, output_dir=DEFAULT_OUTPUT_DIR,
                 run_id=None, engine=None, tracking_uri=None,
                 experiment=mu.DEFAULT_EXPERIMENT, local_mlflow=False,
                 verification_run=False, model_keys=None):
    """Process one URL batch, then automatically retrain/log all eight Safe models.

    A nonempty accepted-label set and a committed completed batch are required.
    Existing row-level ETL rejection rules are preserved. Any failed stage skips
    later stages. A completed ETL commit is retained if later retraining fails.
    Model subsets are permitted only for explicitly labelled verification runs.
    """
    batch_name = _validate_sources(sources, batch_name)
    if model_keys is not None:
        if isinstance(model_keys, str):
            raise ValueError("model_keys must be a sequence")
        model_keys = list(model_keys)
        if not model_keys or len(set(model_keys)) != len(model_keys) or set(model_keys) - set(MODEL_REGISTRY):
            raise ValueError("model_keys must be nonempty unique registered model keys")
        if set(model_keys) != set(MODEL_REGISTRY) and not (verification_run or local_mlflow):
            raise ValueError("Model subsets require verification_run=True; official pipeline runs use all eight")
    if run_id is None:
        run_id = "pipeline_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:12]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id):
        raise ValueError("run_id must be a safe directory name of 1-100 characters")
    folder = Path(output_dir).resolve() / run_id
    folder.mkdir(parents=True, exist_ok=False)
    summary_path = folder / "summary.json"
    report = {
        "pipeline_run_id": run_id, "batch_name": batch_name, "status": "running",
        "started_at": _now(), "finished_at": None, "output_dir": str(folder),
        "sources": {kind: _safe_url(url) for kind, url in sources.items()},
        "dataset_type": "safe", "use_mlflow": True,
        "verification_run": bool(verification_run or local_mlflow),
        "requested_models": list(MODEL_REGISTRY) if model_keys is None else model_keys,
        "stages": [{"name": name, "status": "pending", "started_at": None,
                    "finished_at": None, "duration_seconds": None, "error": None}
                   for name in STAGE_NAMES],
    }
    _save(report, summary_path)
    owned_engine = engine is None
    started = time.perf_counter()
    current = None
    stage_started = started
    try:
        for current in report["stages"]:
            current.update(status="running", started_at=_now())
            stage_started = time.perf_counter()
            _save(report, summary_path)
            print(f"Pipeline {run_id}: {current['name']} running", flush=True)
            if current["name"] == "preflight":
                if engine is None:
                    engine = db.get_engine()
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                tracker = RetrainingTracker(run_id=run_id, experiment=experiment,
                                            tracking_uri=tracking_uri, local=local_mlflow,
                                            verification_run=verification_run, batch_name=batch_name)
                try:
                    if tracker.client.search_runs([tracker.experiment_id],
                                                   filter_string=f"tags.pipeline_run_id = '{run_id}'",
                                                   max_results=1):
                        raise ValueError("This pipeline ID already exists in MLflow; choose a new run_id")
                    tracking_uri = tracker.uri
                    report["tracking"] = {"tracking_uri": tracker.uri,
                                          "experiment_id": tracker.experiment_id}
                finally:
                    tracker.close()
            elif current["name"] == "etl":
                report["etl"] = etl.run_etl(sources, batch_name=batch_name, engine=engine,
                                             raw_dir=folder / "raw", summary_dir=folder / "etl",
                                             dry_run=False)
            elif current["name"] == "load_validation":
                report["loaded_batch"] = _check_loaded_batch(report["etl"], batch_name, engine)
            else:
                training = run_retraining(
                    from_database=True, engine=engine, use_mlflow=True, tracking_uri=tracking_uri,
                    experiment=experiment, run_id=run_id, output_dir=folder / "retraining",
                    model_keys=model_keys, stop_on_error=True, local_mlflow=local_mlflow,
                    verification_run=verification_run, batch_name=batch_name,
                )
                report["retraining"] = _training_reference(training)
                if training["status"] != "completed" or training.get("mlflow", {}).get("status") != "completed":
                    raise ValueError("Retraining/shared logging did not complete successfully")
            current.update(status="completed", finished_at=_now(),
                           duration_seconds=time.perf_counter() - stage_started)
            _save(report, summary_path)
        report["status"] = "completed"
    except Exception as error:
        detail = f"{type(error).__name__}: {_clean_error(etl.short_error(error), sources)}"
        current.update(status="failed", finished_at=_now(), error=detail,
                       duration_seconds=time.perf_counter() - stage_started)
        if isinstance(error, RetrainingFailed):
            report["retraining"] = _training_reference(error.report)
        for stage in report["stages"]:
            if stage["status"] == "pending":
                stage.update(status="skipped", skip_reason=f"Earlier stage {current['name']} failed")
        report["status"] = "failed"
        error_file = folder / "error.txt"
        error_file.write_text(_clean_error(traceback.format_exc(), sources), encoding="utf-8")
        report["error_file"] = str(error_file)
        report.update(finished_at=_now(), duration_seconds=time.perf_counter() - started)
        _save(report, summary_path)
        raise PipelineFailed(report, summary_path) from error
    finally:
        if owned_engine and engine is not None:
            engine.dispose()
    report.update(finished_at=_now(), duration_seconds=time.perf_counter() - started)
    _save(report, summary_path)
    print(f"Pipeline completed. Report: {summary_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", required=True, help="customers_XXX.csv URL")
    parser.add_argument("--orders", required=True, help="orders_XXX.csv URL")
    parser.add_argument("--labels", required=True, help="labels_XXX.csv URL")
    parser.add_argument("--batch-name")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default=mu.DEFAULT_EXPERIMENT)
    parser.add_argument("--local-mlflow", action="store_true", help="Explicit isolated loopback MLflow verification")
    parser.add_argument("--verification-run", action="store_true", help="Exclude practice runs from model selection")
    parser.add_argument("--models", nargs="+", choices=MODEL_REGISTRY,
                        help="Verification-only subset; default is all eight Safe models")
    args = parser.parse_args()
    try:
        run_pipeline({kind: getattr(args, kind) for kind in etl.FILE_KINDS},
                     batch_name=args.batch_name, output_dir=args.output_dir, run_id=args.run_id,
                     tracking_uri=args.tracking_uri, experiment=args.experiment,
                     local_mlflow=args.local_mlflow, verification_run=args.verification_run,
                     model_keys=args.models)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    except PipelineFailed as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
