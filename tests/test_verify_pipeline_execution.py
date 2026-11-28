"""Task #53: evidence validation must not report incomplete/slow runs as passing."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import verify_pipeline_execution as verification


class PipelineEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "safe.csv"
        self.snapshot.write_text("frozen input for checksum verification\n", encoding="utf-8")
        self.training_path = self.root / "training.json"
        self.path = self.root / "summary.json"
        self.save()

    def save(self, etl_seconds=8):
        self.training = {
            "run_id": "test-run", "status": "completed", "dataset_type": "safe",
            "from_database": True, "sample_frac": None, "mlflow": {"status": "completed"},
            "tasks": [{"model_key": key, "status": "completed", "results": [{}]}
                      for key in verification.MODEL_REGISTRY],
            "input": {"status": "completed", "feature_count": 31,
                      "snapshot": str(self.snapshot), "sha256": hashlib.sha256(self.snapshot.read_bytes()).hexdigest(),
                      "rows": {"Train": 100, "Validation": 30, "Test": 30}},
        }
        current = datetime(2026, 11, 28, tzinfo=timezone.utc)
        stages = []
        for name, seconds in zip(verification.STAGE_NAMES, [1, etl_seconds, 1, 500]):
            end = current + timedelta(seconds=seconds)
            stages.append({"name": name, "status": "completed", "started_at": current.isoformat(),
                           "finished_at": end.isoformat(), "duration_seconds": seconds})
            current = end
        self.report = {"status": "completed", "pipeline_run_id": "test-run", "batch_name": "test",
                       "stages": stages, "duration_seconds": etl_seconds + 502,
                       "etl": {"dry_run": False, "batch_name": "test", "labels_inserted": 0, "labels_updated": 3},
                       "loaded_batch": {"status": "completed", "batch_name": "test", "accepted_labels": 3},
                       "retraining": {"summary_path": str(self.training_path)}}
        self.persist()

    def persist(self):
        self.path.write_text(json.dumps(self.report), encoding="utf-8")
        self.training_path.write_text(json.dumps(self.training), encoding="utf-8")

    def check(self):
        return verification.verify_pipeline_execution(self.path, verify_shared=False)

    def test_etl_target_is_separate_from_longer_total_pipeline_duration(self):
        evidence = self.check()
        self.assertEqual(evidence["status"], "local_checks_passed")
        self.assertTrue(evidence["etl_within_three_minutes"])
        self.assertGreater(evidence["total_pipeline_seconds"], 180)
        self.assertEqual(evidence["shared_mlflow_read_back"], "not_checked")

    def test_exactly_180_seconds_passes_but_exceeding_it_fails(self):
        self.save(180)
        self.assertTrue(self.check()["etl_within_three_minutes"])
        self.save(180.001)
        self.assertEqual(self.check()["status"], "performance_failed")

    def test_incomplete_or_wrong_model_set_is_rejected(self):
        self.training["tasks"].pop()
        self.persist()
        with self.assertRaisesRegex(ValueError, "eight"):
            self.check()

    def test_changed_snapshot_is_rejected(self):
        self.snapshot.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.check()

    def test_stage_overlap_or_unreliable_duration_is_rejected(self):
        self.report["stages"][2]["started_at"] = self.report["stages"][0]["started_at"]
        self.persist()
        with self.assertRaisesRegex(ValueError, "timestamps"):
            self.check()
        self.save()
        self.report["stages"][1]["duration_seconds"] = 0
        self.persist()
        with self.assertRaisesRegex(ValueError, "duration"):
            self.check()

    def test_empty_or_uncommitted_load_is_rejected(self):
        self.report["etl"]["labels_updated"] = 0
        self.persist()
        with self.assertRaisesRegex(ValueError, "accepted"):
            self.check()

    def test_full_verification_requires_successful_shared_readback(self):
        with patch.object(verification, "verify_logged_execution", side_effect=RuntimeError("tracking unavailable")):
            with self.assertRaisesRegex(RuntimeError, "tracking unavailable"):
                verification.verify_pipeline_execution(self.path)
        with patch.object(verification, "verify_logged_execution", return_value={"status": "passed", "models_verified": [None] * 8}):
            self.assertEqual(verification.verify_pipeline_execution(self.path)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
