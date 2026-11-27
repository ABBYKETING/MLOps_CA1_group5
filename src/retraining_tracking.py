"""Task #51: link existing member MLflow runs to one retraining execution.

Shared mode requires the team's configured HTTPS server/account and existing
experiment. Local verification is explicit; unavailable shared logging never
falls back to a local store. Existing model settings/logging helpers are reused.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import mlflow
from mlflow.entities import Metric, Param, RunTag

import mlflow_utils as tracking
from data_split import TRAIN_END, VAL_END


def safe_error(error):
    """Keep authentication values out of summaries and saved tracebacks."""
    message = str(error)
    for key in ("MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN", "DB_PASSWORD"):
        value = os.environ.get(key)
        if value:
            message = message.replace(value, "[REDACTED]")
    return message


def resolve_tracking_uri(uri=None, *, local=False):
    uri = uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError("Set MLFLOW_TRACKING_URI or pass --tracking-uri; "
                         "automated retraining has no implicit local fallback")
    parsed = urlparse(uri)
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("Use a tracking URL without embedded credentials, query or fragment; "
                         "configure authentication through environment variables")
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if local:
        if parsed.scheme not in {"http", "https"} or not loopback:
            raise ValueError("--local-mlflow requires an HTTP(S) loopback server")
    else:
        if parsed.scheme != "https" or not parsed.hostname or loopback:
            raise ValueError("Shared retraining logging requires the PBI 3 private HTTPS tracking URI; "
                             "use --local-mlflow only for local verification")
        if not all(os.environ.get(key) for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")):
            raise ValueError("Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD "
                             "to your individual shared-server account")
    return uri.rstrip("/")


class RetrainingTracker:
    def __init__(self, *, run_id, experiment, tracking_uri=None, local=False,
                 verification_run=False, batch_name=None):
        self.uri = resolve_tracking_uri(tracking_uri, local=local)
        if mlflow.active_run() is not None or os.environ.get("MLFLOW_RUN_ID"):
            raise ValueError("End the active MLflow run and unset MLFLOW_RUN_ID before retraining; "
                             "the workflow always creates new experiment records")
        self.run_id = run_id
        self.local = local
        self.verification = verification_run or local
        self.batch_name = batch_name
        self.experiment_name = experiment
        self.client = mlflow.MlflowClient(tracking_uri=self.uri)
        existing = self.client.get_experiment_by_name(experiment)
        if not local and (existing is None or existing.lifecycle_stage != "active"):
            raise ValueError("The shared experiment is missing or inactive; ask the PBI 3 owner "
                             "to grant access to the existing experiment")
        os.environ["MLFLOW_TRACKING_URI"] = self.uri
        tracking.connect_mlflow(experiment)
        self.experiment_id = self.client.get_experiment_by_name(experiment).experiment_id
        self.parent_id = None
        self.model_runs = {}
        self.input_metadata = {}
        # MLflow prints emoji run links when terminating HTTP runs. Windows
        # redirected consoles can use cp1252 and raise after a successful upload.
        # The workflow already saves explicit UI URLs, so suppress only the SDK's
        # decorative output for this execution and restore the caller's setting.
        self._previous_url_output = os.environ.get("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT")
        os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"

    def close(self):
        if self._previous_url_output is None:
            os.environ.pop("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", None)
        else:
            os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = self._previous_url_output

    def run_url(self, run_id):
        return f"{self.uri}/#/experiments/{self.experiment_id}/runs/{run_id}"

    def _tags(self):
        tags = {
            "pbi": "PBI-05", "pipeline_run_id": self.run_id,
            "dataset_version": "safe", "tracking_location": "local" if self.local else "shared",
            "verification_run": str(self.verification).lower(),
            "production_eligible": "false", "exploratory": "false",
        }
        if self.batch_name is not None:
            tags["batch_name"] = self.batch_name
        return tags

    def start(self):
        existing = self.client.search_runs([self.experiment_id],
                                           filter_string=f"tags.pipeline_run_id = '{self.run_id}'",
                                           max_results=1)
        if existing:
            raise ValueError("This pipeline_run_id already exists on the tracking server; choose a new run_id")
        tags = {**self._tags(), "run_type": "retraining_pipeline", "selection_candidate": "false"}
        run = self.client.create_run(self.experiment_id, run_name=f"retraining_{self.run_id}", tags=tags)
        self.parent_id = run.info.run_id
        return {"status": "running", "tracking_uri": self.uri,
                "tracking_location": tags["tracking_location"], "experiment_id": self.experiment_id,
                "parent_run_id": self.parent_id, "parent_run_url": self.run_url(self.parent_id),
                "verification_run": self.verification}

    def record_input(self, info):
        # Log provenance, never the entire CSV containing order/customer identifiers.
        preparation = info.get("preparation", {})
        self.input_metadata = {
            "source": preparation.get("source", "prepared_features"),
            "sha256": info["sha256"], "feature_count": info["feature_count"], "rows": info["rows"],
            "split": preparation.get("split", {"name": "historical_fixed",
                                                "train_end": TRAIN_END.date().isoformat(),
                                                "validation_end": VAL_END.date().isoformat()}),
            "partitions": preparation.get("partitions"),
            "features": info.get("features", preparation.get("features")),
        }
        self.client.log_dict(self.parent_id, self.input_metadata, "metadata/input.json")
        self.client.log_batch(self.parent_id, params=[
            Param("dataset_sha256", info["sha256"]),
            Param("feature_count", str(info["feature_count"])),
            *[Param(f"{name.lower()}_rows", str(count)) for name, count in info["rows"].items()],
        ])

    @contextmanager
    def model_context(self, model_key):
        ids = self.model_runs.setdefault(model_key, [])
        tags = {**self._tags(), "model_key": model_key,
                "run_type": "retraining_model", "data_snapshot_sha256": self.input_metadata["sha256"]}
        with tracking.retraining_run_context(experiment_id=self.experiment_id,
                                            parent_run_id=self.parent_id, tags=tags, on_start=ids.append):
            yield

    def model_links(self, model_key):
        return [{"run_id": run_id, "run_url": self.run_url(run_id)}
                for run_id in self.model_runs.get(model_key, [])]

    def complete_model(self, model_key, results):
        ids = self.model_runs.get(model_key, [])
        if len(ids) != 1 or len(results) != 1 or results[0].get("mlflow_run_id") != ids[0]:
            raise RuntimeError(f"{model_key} did not create exactly one new Safe MLflow run")
        result = results[0]
        child_id = ids[0]
        stored = self.client.get_run(child_id)
        if stored.info.status != "FINISHED":
            raise RuntimeError(f"{model_key} MLflow run is not FINISHED")
        model_uri = stored.data.tags.get("logged_model_uri")
        if not model_uri:
            raise RuntimeError(f"{model_key} did not log a trained model artifact")
        # Member scripts add PBI 2 tags while logging. Apply final PBI 5 identity
        # here, preserving their original run names and threshold policies.
        tags = {**self._tags(), "model_key": model_key, "run_type": "retraining_model",
                "pipeline_task_status": "completed",
                "selection_candidate": str(bool(result.get("selection_candidate")) and not self.verification).lower()}
        now = int(time.time() * 1000)
        metrics = []
        for split, field in (("val", "validation"), ("test", "test")):
            metrics.extend(Metric(f"{split}_{key}", float(result[field][key]), now, 0)
                           for key in ("tp", "tn", "fp", "fn"))
        self.client.log_batch(child_id, tags=[RunTag(key, str(value)) for key, value in tags.items()],
                              metrics=metrics, params=[Param("model_key", model_key),
                                                       Param("pipeline_run_id", self.run_id)])
        self.client.log_dict(child_id, self.input_metadata, "metadata/input.json")
        result.update(pipeline_run_id=self.run_id, mlflow_parent_run_id=self.parent_id,
                      mlflow_run_url=self.run_url(child_id), model_uri=model_uri,
                      verification_run=self.verification)
        if self.verification:
            result["selection_candidate"] = False
        path = Path(result["artifact_dir"]) / "result.json"
        path.write_text(json.dumps({k: v for k, v in result.items() if k != "importance"},
                                   indent=2, default=str), encoding="utf-8")
        self.client.log_artifact(child_id, str(path), "metadata")

    def fail_model(self, model_key, error_file):
        errors = []
        for run_id in self.model_runs.get(model_key, []):
            try:
                self.client.set_tag(run_id, "pipeline_task_status", "failed")
                self.client.log_artifact(run_id, str(error_file), "execution")
                self.client.set_terminated(run_id, status="FAILED")
            except Exception as error:
                errors.append(safe_error(error))
        return errors

    def finish(self, report):
        self.client.log_dict(self.parent_id, report, "execution/summary.json")
        now = int(time.time() * 1000)
        counts = {state: sum(task["status"] == state for task in report["tasks"])
                  for state in ("completed", "failed", "skipped")}
        self.client.log_batch(self.parent_id, metrics=[
            Metric(f"models_{state}", float(count), now, 0) for state, count in counts.items()
        ] + [Metric("execution_seconds", float(report["duration_seconds"]), now, 0)])
        self.client.set_terminated(self.parent_id,
                                   status="FINISHED" if report["status"] == "completed" else "FAILED")
