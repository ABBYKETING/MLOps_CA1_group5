"""Task #52 stage contracts and actual HTTP extraction/cleaning.

SQLite holds batch-status evidence. Training/logging and MySQL-specific loading
are replaced here; their real components have separate tests and live checks.
"""

from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import automated_pipeline as pipeline
import etl_pipeline as etl
from retraining_workflow import RetrainingFailed
from test_etl_pipeline import write_batch, GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS

SOURCES = {kind: f"https://files.example/{kind}_practice.csv" for kind in etl.FILE_KINDS}


class AutomatedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = create_engine("sqlite://")
        self.addCleanup(self.engine.dispose)
        with self.engine.begin() as connection:
            connection.execute(text("CREATE TABLE etl_batches (batch_name TEXT PRIMARY KEY, status TEXT)"))
        self.tracker = Mock()
        self.tracker.uri = "https://shared.example"
        self.tracker.experiment_id = "2"
        self.tracker.client.search_runs.return_value = []
        patcher = patch.object(pipeline, "RetrainingTracker", return_value=self.tracker)
        self.preflight = patcher.start()
        self.addCleanup(patcher.stop)
        self.events = []

    def store_status(self, batch="practice", status="completed"):
        with self.engine.begin() as connection:
            connection.execute(text("INSERT OR REPLACE INTO etl_batches VALUES (:batch, :status)"),
                               {"batch": batch, "status": status})

    def etl_success(self, sources, **kwargs):
        self.events.append("etl")
        self.assertFalse(kwargs["dry_run"])
        self.assertIs(kwargs["engine"], self.engine)
        self.store_status(kwargs["batch_name"])
        return {"batch_name": kwargs["batch_name"], "dry_run": False,
                "labels_inserted": 2, "labels_updated": 0, "rows_rejected": 0}

    def training_success(self, **kwargs):
        self.events.append("retraining")
        self.assertTrue(kwargs["from_database"])
        self.assertTrue(kwargs["use_mlflow"])
        self.assertTrue(kwargs["stop_on_error"])
        self.assertIs(kwargs["engine"], self.engine)
        with self.engine.connect() as connection:
            self.assertEqual(connection.execute(text("SELECT status FROM etl_batches WHERE batch_name=:batch"),
                                                {"batch": kwargs["batch_name"]}).scalar(), "completed")
        keys = list(pipeline.MODEL_REGISTRY) if kwargs["model_keys"] is None else kwargs["model_keys"]
        return {"run_id": kwargs["run_id"], "status": "completed",
                "output_dir": str(Path(kwargs["output_dir"]) / kwargs["run_id"]),
                "mlflow": {"status": "completed", "parent_run_id": "new-parent-id"},
                "tasks": [{"model_key": key, "status": "completed"} for key in keys]}

    def invoke(self, sources=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return pipeline.run_pipeline(sources or SOURCES, output_dir=self.root / "runs",
                                         engine=self.engine, **kwargs)

    def test_success_automatically_runs_all_eight_after_committed_load(self):
        with patch.object(etl, "run_etl", side_effect=self.etl_success) as loader, \
             patch.object(pipeline, "run_retraining", side_effect=self.training_success) as trainer:
            report = self.invoke()
        self.assertEqual(self.events, ["etl", "retraining"])
        self.assertEqual([stage["status"] for stage in report["stages"]], ["completed"] * 4)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["requested_models"], list(pipeline.MODEL_REGISTRY))
        self.assertEqual(len(report["retraining"]["models"]), 8)
        self.assertEqual(report["loaded_batch"]["accepted_labels"], 2)
        self.assertEqual(trainer.call_args.kwargs["batch_name"], "practice")
        self.assertEqual(trainer.call_args.kwargs["run_id"], report["pipeline_run_id"])
        self.assertTrue(Path(report["output_dir"], "summary.json").exists())
        self.tracker.close.assert_called_once()
        loader.assert_called_once()

    def test_preflight_failure_skips_etl_and_training(self):
        self.preflight.side_effect = ValueError("shared logging unavailable")
        with patch.object(etl, "run_etl") as loader, patch.object(pipeline, "run_retraining") as trainer:
            with self.assertRaises(pipeline.PipelineFailed) as failed:
                self.invoke()
        loader.assert_not_called()
        trainer.assert_not_called()
        self.assertEqual([stage["status"] for stage in failed.exception.report["stages"]],
                         ["failed", "skipped", "skipped", "skipped"])

    def test_etl_failure_blocks_retraining_even_if_an_old_batch_marker_is_completed(self):
        self.store_status()
        with patch.object(etl, "run_etl", side_effect=ValueError("missing CSV column")), \
             patch.object(pipeline, "run_retraining") as trainer:
            with self.assertRaises(pipeline.PipelineFailed) as failed:
                self.invoke()
        trainer.assert_not_called()
        self.assertEqual([stage["status"] for stage in failed.exception.report["stages"]],
                         ["completed", "failed", "skipped", "skipped"])
        self.assertIn("missing CSV column", str(failed.exception))

    def test_incomplete_dry_run_empty_labels_and_mismatched_batches_are_blocked(self):
        good = {"batch_name": "practice", "dry_run": False, "labels_inserted": 2, "labels_updated": 0}
        cases = [
            ({**good, "dry_run": True}, "completed"),
            ({**good, "batch_name": "different"}, "completed"),
            ({**good, "labels_inserted": -1}, "completed"),
            ({**good, "labels_inserted": True}, "completed"),
            ({**good, "labels_inserted": None}, "completed"),
            ({**good, "labels_inserted": 0}, "completed"),
            (good, "failed"), (good, "running"), (good, None),
        ]
        for index, (summary, status) in enumerate(cases):
            with self.subTest(case=index):
                with self.engine.begin() as connection:
                    connection.execute(text("DELETE FROM etl_batches"))
                if status is not None:
                    self.store_status(status=status)
                with patch.object(etl, "run_etl", return_value=summary), \
                     patch.object(pipeline, "run_retraining") as trainer:
                    with self.assertRaises(pipeline.PipelineFailed) as failed:
                        self.invoke(run_id=f"bad-load-{index}")
                trainer.assert_not_called()
                self.assertEqual(failed.exception.report["stages"][2]["status"], "failed")
                self.assertEqual(failed.exception.report["stages"][3]["status"], "skipped")

    def test_retraining_failure_retains_completed_etl_and_error_reference(self):
        def fail(**kwargs):
            report = self.training_success(**kwargs)
            report["status"] = "failed"
            report["tasks"][0].update(status="failed", error="controlled model failure")
            raise RetrainingFailed(report, Path(report["output_dir"]) / "summary.json")
        with patch.object(etl, "run_etl", side_effect=self.etl_success), \
             patch.object(pipeline, "run_retraining", side_effect=fail):
            with self.assertRaises(pipeline.PipelineFailed) as failed:
                self.invoke()
        report = failed.exception.report
        self.assertEqual([stage["status"] for stage in report["stages"]],
                         ["completed", "completed", "completed", "failed"])
        self.assertEqual(report["retraining"]["status"], "failed")
        with self.engine.connect() as connection:
            self.assertEqual(connection.execute(text("SELECT status FROM etl_batches")).scalar(), "completed")

    def test_unsuccessful_training_return_is_not_reported_as_pipeline_success(self):
        def fail(**kwargs):
            report = self.training_success(**kwargs)
            report["mlflow"]["status"] = "failed"
            return report
        with patch.object(etl, "run_etl", side_effect=self.etl_success), \
             patch.object(pipeline, "run_retraining", side_effect=fail):
            with self.assertRaises(pipeline.PipelineFailed):
                self.invoke()

    def test_corrected_existing_labels_also_trigger_retraining(self):
        def load_updates(sources, **kwargs):
            self.store_status(kwargs["batch_name"])
            return {"batch_name": kwargs["batch_name"], "dry_run": False,
                    "labels_inserted": 0, "labels_updated": 2}
        with patch.object(etl, "run_etl", side_effect=load_updates), \
             patch.object(pipeline, "run_retraining", side_effect=self.training_success) as trainer:
            report = self.invoke()
        trainer.assert_called_once()
        self.assertEqual(report["loaded_batch"]["accepted_labels"], 2)
        self.assertEqual(report["status"], "completed")

    def test_owned_database_engine_is_closed_after_execution(self):
        with patch.object(pipeline.db, "get_engine", return_value=self.engine), \
             patch.object(self.engine, "dispose", wraps=self.engine.dispose) as dispose, \
             patch.object(etl, "run_etl", side_effect=self.etl_success), \
             patch.object(pipeline, "run_retraining", side_effect=self.training_success), \
             redirect_stdout(io.StringIO()):
            pipeline.run_pipeline(SOURCES, output_dir=self.root / "runs")
        dispose.assert_called_once()

    def test_repeated_execution_keeps_separate_reports_and_refuses_existing_ids(self):
        with patch.object(etl, "run_etl", side_effect=self.etl_success), \
             patch.object(pipeline, "run_retraining", side_effect=self.training_success):
            first = self.invoke()
            original = Path(first["output_dir"], "summary.json").read_bytes()
            second = self.invoke()
            self.assertNotEqual(first["pipeline_run_id"], second["pipeline_run_id"])
            self.assertEqual(Path(first["output_dir"], "summary.json").read_bytes(), original)
            with self.assertRaises(FileExistsError):
                self.invoke(run_id=first["pipeline_run_id"])
            self.tracker.client.search_runs.return_value = [object()]
            with self.assertRaisesRegex(pipeline.PipelineFailed, "already exists in MLflow"):
                self.invoke(run_id="already-on-server")

    def test_invalid_sources_options_and_cli_failure(self):
        bad_sources = [{}, {**SOURCES, "extra": "https://files.example/extra.csv"},
                       {**SOURCES, "orders": "orders_practice.csv"},
                       {**SOURCES, "orders": "https://user:password@files.example/orders_practice.csv"},
                       {**SOURCES, "orders": "https://files.example/orders_other.csv"}]
        for sources in bad_sources:
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                pipeline.run_pipeline(sources, engine=self.engine, output_dir=self.root / "bad")
        opaque = {kind: f"https://files.example/download?id={kind}" for kind in etl.FILE_KINDS}
        self.assertEqual(pipeline._validate_sources(opaque, "practice"), "practice")
        for name in (".", "..", "../../escape", "has spaces"):
            with self.assertRaises(ValueError):
                self.invoke(batch_name=name)
        with self.assertRaises(ValueError):
            self.invoke(model_keys=["chu_extra_trees"])
        with self.assertRaises(ValueError):
            self.invoke(model_keys="chu_extra_trees", verification_run=True)
        report = {"pipeline_run_id": "bad", "stages": [{"name": "etl", "status": "failed", "error": "bad CSV"}]}
        failure = pipeline.PipelineFailed(report, "summary.json")
        argv = ["automated_pipeline.py", "--customers", SOURCES["customers"],
                "--orders", SOURCES["orders"], "--labels", SOURCES["labels"]]
        with patch.object(sys, "argv", argv), patch.object(pipeline, "run_pipeline", side_effect=failure), \
             redirect_stderr(io.StringIO()):
            self.assertEqual(pipeline.main(), 1)

    def test_passwords_and_signed_url_queries_are_redacted_from_errors(self):
        secret = "test-only-secret-value"
        sources = {kind: url + "?signature=private-query-value" for kind, url in SOURCES.items()}
        self.preflight.side_effect = RuntimeError(secret + " " + sources["orders"])
        with patch.dict(os.environ, {"DB_PASSWORD": secret}):
            with self.assertRaises(pipeline.PipelineFailed) as failed:
                self.invoke(sources=sources)
        folder = Path(failed.exception.report["output_dir"])
        for name in ("summary.json", "error.txt"):
            content = (folder / name).read_text(encoding="utf-8")
            self.assertNotIn(secret, content)
            self.assertNotIn("private-query-value", content)

    def test_actual_http_download_and_cleaning_feed_the_automatic_handoff(self):
        write_batch(self.root, "practice", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS)
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(self.root)))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        urls = {kind: f"http://127.0.0.1:{server.server_port}/{kind}_practice.csv" for kind in etl.FILE_KINDS}
        # Only the MySQL-specific write operations are substituted. The existing
        # extract/clean/validate/summary code and HTTP downloads execute for real.
        def start_batch(engine, batch):
            self.store_status(batch, "running")
        def load(engine, batch, cleaned, rejected, raw):
            self.assertTrue(cleaned["customers"]["mobile_verified"].all())
            self.assertEqual(cleaned["orders"]["collect_type"].tolist(), ["Delivery", "Pickup"])
            self.assertEqual(cleaned["orders"]["payment_method"].tolist(), ["credit card", "payment on delivery"])
            self.assertEqual(rejected, [])
            self.store_status(batch)
            counts = {"customers_inserted": 2, "customers_updated": 0, "orders_inserted": 2,
                      "orders_updated": 0, "labels_inserted": 2, "labels_updated": 0, "rows_rejected": 0}
            return counts, rejected
        try:
            with patch.object(etl, "start_batch", side_effect=start_batch), \
                 patch.object(etl, "load", side_effect=load), \
                 patch.object(pipeline, "run_retraining", side_effect=self.training_success):
                report = self.invoke(sources=urls)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["etl"]["rows_read"], {"customers": 2, "orders": 2, "labels": 2})
        self.assertEqual(report["etl"]["labels_inserted"], 2)
        self.assertTrue(Path(report["output_dir"], "etl", "practice_summary.json").exists())
        for kind in etl.FILE_KINDS:
            self.assertTrue(Path(report["output_dir"], "raw", "practice", f"{kind}_practice.csv").exists())


if __name__ == "__main__":
    unittest.main()
