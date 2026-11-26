"""Task #49: real repeated runs plus controlled orchestration failures."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retraining_workflow as workflow
from test_training_interface import write_features


class RetrainingWorkflowTests(unittest.TestCase):
    def test_all_eight_models_run_twice_without_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            write_features(folder)
            original = (Path(folder) / "features_base_safe.csv").read_bytes()
            with redirect_stdout(io.StringIO()):
                first = workflow.run_retraining(data_dir=folder, output_dir=Path(folder) / "runs")
                second = workflow.run_retraining(data_dir=folder, output_dir=Path(folder) / "runs")
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertNotEqual(first["output_dir"], second["output_dir"])
            self.assertEqual((Path(folder) / "features_base_safe.csv").read_bytes(), original)
            for report in [first, second]:
                self.assertEqual(report["status"], "completed")
                self.assertEqual([t["model_key"] for t in report["tasks"]], list(workflow.MODEL_REGISTRY))
                self.assertEqual(report["input"]["rows"], {"Train": 1200, "Validation": 400, "Test": 400})
                self.assertEqual(Path(report["input"]["snapshot"]).read_bytes(), original)
                self.assertFalse((Path(report["input"]["snapshot"]).parent / "features_base_full.csv").exists())
                saved = json.loads((Path(report["output_dir"]) / "summary.json").read_text())
                self.assertEqual(saved["status"], "completed")
                for task in report["tasks"]:
                    self.assertEqual(task["status"], "completed")
                    self.assertIsNotNone(task["started_at"])
                    self.assertIsNotNone(task["finished_at"])
                    self.assertGreaterEqual(task["duration_seconds"], 0)
                    self.assertEqual(task["results"][0]["dataset_type"], "safe")
                    artifacts = Path(task["results"][0]["artifact_dir"])
                    self.assertTrue((artifacts / "model.joblib").is_file())
                    self.assertTrue((artifacts / "result.json").is_file())
            before = (Path(first["output_dir"]) / "summary.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                workflow.run_retraining(data_dir=folder, output_dir=Path(folder) / "runs", run_id=first["run_id"])
            self.assertEqual((Path(first["output_dir"]) / "summary.json").read_bytes(), before)
            evidence = {
                "data": "synthetic Safe features, no Full CSV or database",
                "executions": [{"run_id": r["run_id"], "status": r["status"],
                                "duration_seconds": r["duration_seconds"],
                                "models_completed": [t["model_key"] for t in r["tasks"]]}
                               for r in [first, second]],
                "original_input_unchanged": True,
                "existing_run_overwrite_rejected": True,
            }
        (ROOT / "outputs").mkdir(exist_ok=True)
        (ROOT / "outputs" / "task49_workflow_verification.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8")

    def test_model_failure_is_recorded_and_other_models_continue(self):
        with tempfile.TemporaryDirectory() as folder:
            write_features(folder)
            keys = list(workflow.MODEL_REGISTRY)[:3]
            def train(key, **kwargs):
                report = json.loads((Path(folder) / "runs" / "failure" / "summary.json").read_text())
                current = next(t for t in report["tasks"] if t["model_key"] == key)
                self.assertEqual(current["status"], "running")
                if key == keys[0]:
                    raise ValueError("simulated model failure")
                return [{"dataset_type": "safe"}], {}
            with patch.object(workflow, "run_model", side_effect=train) as runner, redirect_stdout(io.StringIO()):
                with self.assertRaises(workflow.RetrainingFailed) as failed:
                    workflow.run_retraining(data_dir=folder, output_dir=Path(folder) / "runs",
                                            model_keys=keys, run_id="failure")
            report = failed.exception.report
            self.assertEqual(runner.call_count, 3)
            self.assertEqual(report["status"], "failed")
            self.assertEqual([t["status"] for t in report["tasks"]], ["failed", "completed", "completed"])
            self.assertIn("simulated model failure", report["tasks"][0]["error"])
            self.assertIn("ValueError", Path(report["tasks"][0]["error_file"]).read_text())
            self.assertEqual(json.loads(Path(failed.exception.summary_path).read_text())["status"], "failed")

    def test_stop_on_error_marks_remaining_models_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            write_features(folder)
            with patch.object(workflow, "run_model", side_effect=RuntimeError("model stopped")) as runner, redirect_stdout(io.StringIO()):
                with self.assertRaises(workflow.RetrainingFailed) as failed:
                    workflow.run_retraining(data_dir=folder, output_dir=Path(folder) / "runs", stop_on_error=True)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(failed.exception.report["tasks"][0]["status"], "failed")
            self.assertTrue(all(t["status"] == "skipped" for t in failed.exception.report["tasks"][1:]))

    def test_bad_input_prevents_every_training_task(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = write_features(folder)
            frame["refund_value"] = 0
            frame.to_csv(Path(folder) / "features_base_safe.csv", index=False)
            with patch.object(workflow, "run_model") as runner, redirect_stdout(io.StringIO()):
                with self.assertRaises(workflow.RetrainingFailed) as failed:
                    workflow.run_retraining(data_dir=folder, output_dir=Path(folder) / "runs")
            runner.assert_not_called()
            self.assertEqual(failed.exception.report["input"]["status"], "failed")
            self.assertIn("refund_value", failed.exception.report["input"]["error"])
            self.assertTrue(all(t["status"] == "skipped" for t in failed.exception.report["tasks"]))

    def test_all_tasks_read_the_same_snapshot_if_source_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            write_features(folder)
            source = Path(folder) / "features_base_safe.csv"
            original = source.read_bytes()
            def train(key, **kwargs):
                snapshot = Path(kwargs["data_dir"]) / "features_base_safe.csv"
                self.assertEqual(snapshot.read_bytes(), original)
                self.assertNotEqual(snapshot, source)
                source.write_text("changed original file", encoding="utf-8")
                return [{"dataset_type": "safe"}], {}
            with patch.object(workflow, "run_model", side_effect=train), redirect_stdout(io.StringIO()):
                report = workflow.run_retraining(data_dir=folder, output_dir=Path(folder) / "runs")
            self.assertEqual(report["status"], "completed")

    def test_invalid_options_and_cli_exit_code(self):
        for keys in [[], ["missing"], ["emi_mlp", "emi_mlp"], "emi_mlp"]:
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                workflow.run_retraining(model_keys=keys)
        for run_id in ["../escape", "..", "bad/name", "", "bad name"]:
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                workflow.run_retraining(run_id=run_id)
        with self.assertRaises(ValueError):
            workflow.run_retraining(sample_frac=0.1, use_mlflow=True)
        report = {"run_id": "failed_test", "tasks": [{"model_key": "emi_mlp", "status": "failed"}]}
        failure = workflow.RetrainingFailed(report, "summary.json")
        with patch.object(sys, "argv", ["retraining_workflow.py"]), patch.object(
            workflow, "run_retraining", side_effect=failure
        ), redirect_stderr(io.StringIO()):
            self.assertEqual(workflow.main(), 1)


if __name__ == "__main__":
    unittest.main()
