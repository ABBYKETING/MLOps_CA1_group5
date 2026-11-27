"""Task #51: real HTTP MLflow logging, repetition, artifacts and failure paths.

Starts an isolated loopback server with temporary SQLite/artifact storage.
No AWS, shared MLflow account, or real fraud-training data is used.
"""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow_utils as shared
from model_comparison import expected_run_names, fetch_official_runs
import retraining_tracking as logging
import retraining_workflow as workflow
from test_training_interface import write_features
from verify_retraining_logging import verify_logged_execution


class TrackingConfigurationTests(unittest.TestCase):
    def test_shared_configuration_requires_explicit_https_and_individual_account(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "no implicit local fallback"):
                logging.resolve_tracking_uri()
            for uri in ("sqlite:///test.db", "http://shared.example", "http://127.0.0.1:5000"):
                with self.subTest(uri=uri), self.assertRaisesRegex(ValueError, "HTTPS"):
                    logging.resolve_tracking_uri(uri)
            with self.assertRaisesRegex(ValueError, "individual"):
                logging.resolve_tracking_uri("https://shared.example")
            with patch.dict(os.environ, {"MLFLOW_TRACKING_USERNAME": "chu", "MLFLOW_TRACKING_PASSWORD": "test-secret"}):
                self.assertEqual(logging.resolve_tracking_uri("https://shared.example/"), "https://shared.example")
            for uri in ("https://user:password@shared.example", "https://shared.example?token=secret"):
                with self.assertRaisesRegex(ValueError, "embedded credentials"):
                    logging.resolve_tracking_uri(uri)
            self.assertEqual(logging.resolve_tracking_uri("http://127.0.0.1:5000", local=True),
                             "http://127.0.0.1:5000")
            with self.assertRaisesRegex(ValueError, "loopback"):
                logging.resolve_tracking_uri("http://remote.example", local=True)

    def test_retraining_runs_do_not_break_historical_baseline_comparison(self):
        baseline = [SimpleNamespace(info=SimpleNamespace(run_name=name, status="FINISHED"),
                                    data=SimpleNamespace(tags={})) for name in expected_run_names()]
        extra = SimpleNamespace(info=baseline[0].info,
                                data=SimpleNamespace(tags={"pipeline_run_id": "execution-1"}))
        client = Mock()
        client.get_experiment_by_name.return_value = SimpleNamespace(name="baseline", experiment_id="1")
        client.search_runs.return_value = baseline + [extra]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(fetch_official_runs(client, "baseline"), baseline)


class RetrainingTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cls.port = listener.getsockname()[1]
        cls.uri = f"http://127.0.0.1:{cls.port}"
        cls.log = (cls.root / "server.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(MLFLOW_DISABLE_TELEMETRY="true", MLFLOW_DISABLE_AGENT_HINT="1")
        for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
                    "MLFLOW_TRACKING_TOKEN", "MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID"):
            env.pop(key, None)
        cls.server = subprocess.Popen([
            sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(cls.port),
            "--workers", "1", "--backend-store-uri", f"sqlite:///{(cls.root / 'tracking.db').as_posix()}",
            "--artifacts-destination", str(cls.root / "artifacts"),
        ], stdout=cls.log, stderr=subprocess.STDOUT, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt")
        try:
            deadline = time.monotonic() + 75
            while time.monotonic() < deadline:
                if cls.server.poll() is not None:
                    break
                try:
                    with urlopen(cls.uri + "/health", timeout=1) as response:
                        if response.status == 200:
                            return
                except Exception:
                    time.sleep(0.25)
            raise RuntimeError("Local MLflow server did not start: "
                               + (cls.root / "server.log").read_text(encoding="utf-8")[-2000:])
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.server.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(cls.server.pid), "/T", "/F"], capture_output=True)
            else:
                import signal
                os.killpg(cls.server.pid, signal.SIGTERM)
            cls.server.wait(timeout=15)
        cls.log.close()
        cls.folder.cleanup()

    def setUp(self):
        self.env = patch.dict(os.environ, {"MLFLOW_TRACKING_URI": self.uri,
                                          "MLFLOW_DISABLE_TELEMETRY": "true",
                                          "MLFLOW_DISABLE_AGENT_HINT": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.frame = write_features(self.path)
        self.client = mlflow.MlflowClient(tracking_uri=self.uri)

    def run_workflow(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return workflow.run_retraining(data_dir=self.path, output_dir=self.path / "runs",
                                           use_mlflow=True, local_mlflow=True,
                                           tracking_uri=self.uri, experiment="Task51-HTTP-Test", **kwargs)

    def test_all_eight_models_log_twice_with_distinct_executions_and_reloadable_artifacts(self):
        first = self.run_workflow(batch_name="school_test")
        second = self.run_workflow(batch_name="school_test")
        parents = set()
        children = set()
        for report in (first, second):
            self.assertEqual(report["status"], "completed")
            parent_id = report["mlflow"]["parent_run_id"]
            parents.add(parent_id)
            parent = self.client.get_run(parent_id)
            self.assertEqual(parent.info.status, "FINISHED")
            self.assertEqual(parent.data.metrics["models_completed"], 8)
            self.assertEqual(parent.data.tags["pipeline_run_id"], report["run_id"])
            runs = self.client.search_runs([report["mlflow"]["experiment_id"]],
                                           filter_string=f"tags.pipeline_run_id = '{report['run_id']}'")
            self.assertEqual(len(runs), 9)
            with urlopen(report["mlflow"]["parent_run_url"], timeout=10) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("mlflow", response.read().decode("utf-8").lower())
            self.assertIn("execution/summary.json", {item.path for item in self.client.list_artifacts(parent_id, "execution")})
            for task in report["tasks"]:
                result = task["results"][0]
                run_id = result["mlflow_run_id"]
                self.assertNotIn(run_id, children)
                children.add(run_id)
                stored = self.client.get_run(run_id)
                self.assertEqual(stored.info.status, "FINISHED")
                self.assertEqual(stored.data.tags["mlflow.parentRunId"], parent_id)
                self.assertEqual(stored.data.tags["pbi"], "PBI-05")
                self.assertEqual(stored.data.tags["team_member"], task["member"])
                self.assertEqual(stored.data.tags["model_key"], task["model_key"])
                self.assertEqual(stored.data.tags["batch_name"], "school_test")
                self.assertEqual(stored.data.tags["selection_candidate"], "false")
                self.assertEqual(stored.data.tags["verification_run"], "true")
                self.assertEqual(stored.data.params["pipeline_run_id"], report["run_id"])
                self.assertEqual(stored.data.params["dataset_sha256"], report["input"]["sha256"])
                self.assertEqual(stored.data.params["feature_count"], "4")
                self.assertAlmostEqual(float(stored.data.params["decision_threshold"]), result["threshold"])
                for prefix, field in (("val", "validation"), ("test", "test")):
                    for metric in ("precision", "recall", "f1", "roc_auc", "tp", "tn", "fp", "fn"):
                        self.assertAlmostEqual(stored.data.metrics[f"{prefix}_{metric}"], result[field][metric])
                    self.assertAlmostEqual(stored.data.metrics[f"{prefix}_pr_auc"], result[field]["average_precision"])
                for split in ("val", "test"):
                    files = {item.path for item in self.client.list_artifacts(run_id, "evaluation")}
                    self.assertIn(f"evaluation/{split}_confusion_matrix.png", files)
                    self.assertIn(f"evaluation/{split}_pr_curve.png", files)
                self.assertIn("metadata/result.json", {item.path for item in self.client.list_artifacts(run_id, "metadata")})
                # Check artifact upload/download and every model's skops trust list.
                if report is first:
                    restored = mlflow.sklearn.load_model(result["model_uri"])
                    original = joblib.load(Path(result["artifact_dir"]) / "model.joblib")
                    sample = self.frame[result["features"]].iloc[:12]
                    np.testing.assert_allclose(restored.predict_proba(sample), original.predict_proba(sample))
            self.assertIsNone(mlflow.active_run())
        self.assertEqual(len(parents), 2)
        self.assertEqual(len(children), 16)
        verification = verify_logged_execution(Path(first["output_dir"]) / "summary.json", local_mlflow=True)
        self.assertEqual(verification["status"], "passed")
        self.assertEqual(len(verification["models_verified"]), 8)
        self.assertIsNone(shared._RETRAINING_CONTEXT.get())
        with patch.object(workflow, "run_model") as runner, redirect_stdout(io.StringIO()):
            with self.assertRaises(workflow.RetrainingFailed) as reused:
                workflow.run_retraining(data_dir=self.path, output_dir=self.path / "another-root",
                                        run_id=first["run_id"], use_mlflow=True, local_mlflow=True,
                                        tracking_uri=self.uri, experiment="Task51-HTTP-Test")
        runner.assert_not_called()
        self.assertIn("already exists on the tracking server", str(reused.exception))
        evidence = {"scope": "Isolated loopback HTTP MLflow; synthetic input; no shared server",
                    "executions": [{"pipeline_run_id": r["run_id"], "parent_run_id": r["mlflow"]["parent_run_id"],
                                     "model_run_ids": [task["results"][0]["mlflow_run_id"] for task in r["tasks"]]}
                                   for r in (first, second)],
                    "all_eight_models_reloaded": True, "ui_html_accessible": True,
                    "shared_server_verified": False}
        (ROOT / "outputs").mkdir(exist_ok=True)
        (ROOT / "outputs" / "task51_logging_verification.json").write_text(json.dumps(evidence, indent=2))

    def test_missing_shared_configuration_skips_training_without_fallback(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(workflow, "run_model") as runner, \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(workflow.RetrainingFailed) as failed:
                workflow.run_retraining(data_dir=self.path, output_dir=self.path / "runs", use_mlflow=True)
        runner.assert_not_called()
        self.assertEqual(failed.exception.report["mlflow"]["status"], "failed")
        self.assertIn("no implicit local fallback", str(failed.exception))
        self.assertTrue(all(task["status"] == "skipped" for task in failed.exception.report["tasks"]))

    def test_legacy_shared_logging_works_without_pipeline_context(self):
        from test_mlflow_logging import TestMLflowLogging
        with redirect_stdout(io.StringIO()):
            TestMLflowLogging("test_complete_logging_workflow").test_complete_logging_workflow()
        self.assertIsNone(shared._RETRAINING_CONTEXT.get())
        self.assertIsNone(mlflow.active_run())

    def test_model_logging_failure_marks_child_and_pipeline_failed_and_keeps_other_tasks(self):
        original = shared.log_trained_model
        calls = []
        def fail_first(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("controlled model artifact upload failure")
            return original(*args, **kwargs)
        with patch.object(shared, "log_trained_model", side_effect=fail_first):
            with self.assertRaises(workflow.RetrainingFailed) as failed:
                self.run_workflow(model_keys=["chu_extra_trees", "abby_logistic_regression"])
        report = failed.exception.report
        self.assertEqual([task["status"] for task in report["tasks"]], ["failed", "completed"])
        self.assertEqual(self.client.get_run(report["mlflow"]["parent_run_id"]).info.status, "FAILED")
        child_id = report["tasks"][0]["mlflow_runs"][0]["run_id"]
        self.assertEqual(self.client.get_run(child_id).info.status, "FAILED")
        self.assertIsNone(mlflow.active_run())

    def test_input_failure_finalizes_parent_and_secret_values_are_redacted(self):
        secret = "dummy-private-value-for-redaction"
        with patch.dict(os.environ, {"MLFLOW_TRACKING_PASSWORD": secret}), \
             patch.object(workflow, "_prepare_input", side_effect=ValueError(f"controlled failure {secret}")):
            with self.assertRaises(workflow.RetrainingFailed) as failed:
                self.run_workflow(model_keys=["chu_extra_trees"])
        report = failed.exception.report
        self.assertEqual(self.client.get_run(report["mlflow"]["parent_run_id"]).info.status, "FAILED")
        self.assertNotIn(secret, Path(failed.exception.summary_path).read_text())
        self.assertNotIn(secret, Path(report["input"]["error_file"]).read_text())
        self.assertIsNone(shared._RETRAINING_CONTEXT.get())

    def test_final_report_logging_failure_fails_execution(self):
        with patch.object(logging.RetrainingTracker, "finish", side_effect=RuntimeError("final upload failed")):
            with self.assertRaises(workflow.RetrainingFailed) as failed:
                self.run_workflow(model_keys=["chu_extra_trees"])
        self.assertEqual(failed.exception.report["status"], "failed")
        self.assertEqual(failed.exception.report["mlflow"]["status"], "failed")
        self.assertIn("final upload failed", str(failed.exception))
        self.assertEqual(self.client.get_run(failed.exception.report["mlflow"]["parent_run_id"]).info.status, "FAILED")

    def test_existing_mlflow_run_is_not_reused(self):
        with patch.dict(os.environ, {"MLFLOW_RUN_ID": "must-not-resume"}), \
             patch.object(workflow, "run_model") as runner:
            with self.assertRaises(workflow.RetrainingFailed) as failed:
                self.run_workflow(model_keys=["chu_extra_trees"])
        runner.assert_not_called()
        self.assertIn("always creates new", str(failed.exception))

    def test_prepared_safe_snapshot_logs_moving_split_and_provenance(self):
        from sqlalchemy import create_engine
        from retraining_data import prepare_retraining_data
        from test_retraining_data import populate_database
        engine = create_engine("sqlite://")
        try:
            populate_database(engine)
            folder = self.path / "prepared"
            metadata = prepare_retraining_data(output_dir=folder, engine=engine)
        finally:
            engine.dispose()
        with redirect_stdout(io.StringIO()):
            report = workflow.run_retraining(data_dir=folder, output_dir=self.path / "runs",
                                             model_keys=["chu_extra_trees"], use_mlflow=True,
                                             local_mlflow=True, tracking_uri=self.uri,
                                             experiment="Task51-HTTP-Test")
        result = report["tasks"][0]["results"][0]
        self.assertEqual(result["feature_count"], 31)
        self.assertEqual(result["data_preparation"], metadata)
        for run_id in (report["mlflow"]["parent_run_id"], result["mlflow_run_id"]):
            downloaded = self.client.download_artifacts(run_id, "metadata/input.json", str(self.path))
            saved = json.loads(Path(downloaded).read_text())
            self.assertEqual(saved["split"], metadata["split"])
            self.assertEqual(saved["partitions"], metadata["partitions"])
            self.assertEqual(saved["sha256"], metadata["sha256"])
            self.assertEqual(saved["features"], metadata["features"])
        evidence = verify_logged_execution(Path(report["output_dir"]) / "summary.json", local_mlflow=True)
        self.assertEqual(evidence["status"], "passed")

    def test_windows_cp1252_console_does_not_break_http_logging(self):
        previous = os.environ.get("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT")
        buffer = io.BytesIO()
        console = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
        try:
            with redirect_stdout(console):
                report = workflow.run_retraining(data_dir=self.path, output_dir=self.path / "runs",
                                                 model_keys=["chu_lightgbm"], use_mlflow=True,
                                                 local_mlflow=True, tracking_uri=self.uri,
                                                 experiment="Task51-HTTP-Test")
            console.flush()
            self.assertEqual(report["status"], "completed")
            self.assertIn("completed", buffer.getvalue().decode("cp1252"))
            self.assertEqual(os.environ.get("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"), previous)
            parent_id = report["mlflow"]["parent_run_id"]
            self.assertEqual(self.client.get_run(parent_id).info.status, "FINISHED")
        finally:
            console.detach()

    def test_unavailable_tracking_server_skips_models_without_fallback(self):
        with patch.object(logging.mlflow, "MlflowClient", side_effect=RuntimeError("tracking server unavailable")), \
             patch.object(workflow, "run_model") as runner:
            with self.assertRaises(workflow.RetrainingFailed) as failed:
                self.run_workflow(model_keys=["chu_extra_trees"])
        runner.assert_not_called()
        self.assertEqual(failed.exception.report["tasks"][0]["status"], "skipped")
        self.assertIn("tracking server unavailable", str(failed.exception))


if __name__ == "__main__":
    unittest.main()
