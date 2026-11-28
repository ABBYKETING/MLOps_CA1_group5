"""Task #53: real HTTP input failures with isolated batch-state fixtures.

No production database faults are injected. MySQL writes and training are
substituted; extraction, schema validation and pipeline failure propagation run.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_automated_pipeline as fixtures
from test_etl_pipeline import GOOD_CUSTOMERS, GOOD_LABELS, GOOD_ORDERS, write_batch


class PipelineReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.AutomatedPipelineTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        write_batch(self.f.root, "practice", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS)
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(self.f.root)))
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.stop_server)
        self.urls = {k: f"http://127.0.0.1:{self.server.server_port}/{k}_practice.csv"
                     for k in fixtures.etl.FILE_KINDS}

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)

    def assert_blocked(self):
        with patch.object(fixtures.etl, "start_batch") as start, \
             patch.object(fixtures.etl, "load") as load, \
             patch.object(fixtures.pipeline, "run_retraining") as train:
            with self.assertRaises(fixtures.pipeline.PipelineFailed) as failed:
                self.f.invoke(sources=self.urls)
        start.assert_not_called()
        load.assert_not_called()
        train.assert_not_called()
        self.assertEqual([s["status"] for s in failed.exception.report["stages"]],
                         ["completed", "failed", "skipped", "skipped"])
        return failed.exception

    def test_http_404_retries_then_blocks_all_database_writes_and_training(self):
        (self.f.root / "orders_practice.csv").unlink()
        with patch.object(fixtures.etl.time, "sleep") as backoff:
            failure = self.assert_blocked()
        self.assertEqual(backoff.call_count, 2)
        self.assertIn("404", str(failure))

    def test_downloaded_csv_with_missing_columns_blocks_load_and_training(self):
        (self.f.root / "customers_practice.csv").write_text("unrelated_column\nvalue\n", encoding="utf-8")
        failure = self.assert_blocked()
        self.assertIn("missing", str(failure).lower())

    def test_database_write_failure_marks_batch_failed_and_skips_training(self):
        with patch.object(fixtures.etl, "start_batch", side_effect=lambda e, b: self.f.store_status(b, "running")), \
             patch.object(fixtures.etl, "load", side_effect=RuntimeError("simulated database write failure")), \
             patch.object(fixtures.etl, "fail_batch", side_effect=lambda e, b, err: self.f.store_status(b, "failed")), \
             patch.object(fixtures.pipeline, "run_retraining") as train:
            with self.assertRaises(fixtures.pipeline.PipelineFailed) as failed:
                self.f.invoke(sources=self.urls)
        train.assert_not_called()
        self.assertIn("simulated database write failure", str(failed.exception))
        self.assertEqual(failed.exception.report["stages"][-1]["status"], "skipped")

    def test_cli_invalid_url_returns_argument_error_without_services(self):
        result = subprocess.run([sys.executable, "src/automated_pipeline.py", "--customers", "not-a-url",
                                 "--orders", self.urls["orders"], "--labels", self.urls["labels"]],
                                capture_output=True, text=True, timeout=25,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 2)
        self.assertIn("HTTP(S)", result.stderr)


if __name__ == "__main__":
    unittest.main()
