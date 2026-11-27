"""Task #50: combined database records, moving windows and all-model handoff.

SQLite exercises the actual shared SQL join without an AWS connection.
Synthetic labels are separable; these tests verify plumbing, not model quality.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_processing as dp
from data_split import create_split_labels, rolling_split_policy
import model_utils as mu
import retraining_data as preparation
import retraining_workflow as workflow


def populate_database(engine, *, start="2024-01-01", days=100, append=False):
    """Two countries reuse order IDs; twelve labelled orders per date."""
    n = days * 12
    offset = (pd.Timestamp(start) - pd.Timestamp("2024-01-01")).days * 12
    indices = np.arange(offset, offset + n)
    countries = np.where(indices % 2 == 0, "MY", "PK")
    y = (indices // 2) % 2
    customers = pd.DataFrame({
        "country_code": ["MY", "PK"], "customer_id": ["c1", "c1"],
        "mobile_verified": [True, False], "num_orders_last_50days": [20, 10],
        "num_cancelled_orders_last_50days": [1, 0], "num_refund_orders_last_50days": [0, 1],
        "total_payment_last_50days": [50., 100.], "num_associated_customers": [1, 2],
        "first_order_datetime": pd.to_datetime(["2022-01-01", None]),
    })
    orders = pd.DataFrame({
        "country_code": countries, "order_id": [f"{i // 2:05d}" for i in indices],
        "collect_type": "Delivery", "payment_method": "credit card",
        "order_value": np.where(y == 1, 1000., 1.), "num_items_ordered": 2,
        "refund_value": np.where(y == 1, 500., 0.),
        "order_date": np.repeat(pd.date_range(start, periods=days).to_numpy(), 12),
    })
    labels = orders[["country_code", "order_id"]].assign(customer_id="c1", is_fraud=y)
    if not append:
        customers.to_sql("customers", engine, index=False)
    mode = "append" if append else "fail"
    orders.to_sql("orders", engine, index=False, if_exists=mode)
    labels.to_sql("fraud_labels", engine, index=False, if_exists=mode)


class RetrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        self.addCleanup(self.engine.dispose)
        populate_database(self.engine)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare(self, name="prepared"):
        folder = self.root / name
        metadata = preparation.prepare_retraining_data(output_dir=folder, engine=self.engine)
        return folder, metadata, mu.load_feature_data(folder / "features_base_safe.csv")

    def test_calendar_boundaries_and_historical_defaults(self):
        dates = ["2024-04-05 23:59:59", "2024-04-06", "2024-05-03 23:59:59",
                 "2024-05-04", "2024-05-31 23:59:59"]
        frame = pd.DataFrame({"order_date": pd.to_datetime(dates, format="mixed")})
        policy = rolling_split_policy(frame)
        self.assertEqual(policy["anchor_date"], "2024-05-31")
        self.assertEqual(policy["train_end"], "2024-04-05")
        self.assertEqual(policy["validation_start"], "2024-04-06")
        self.assertEqual(policy["validation_end"], "2024-05-03")
        self.assertEqual(policy["test_start"], "2024-05-04")
        frame.attrs["retraining_split"] = policy
        self.assertEqual(create_split_labels(frame).tolist(),
                         ["Train", "Validation", "Validation", "Test", "Test"])
        historical = pd.DataFrame({"order_date": ["2023-03-15", "2023-03-16", "2023-04-14"]})
        self.assertEqual(create_split_labels(historical).tolist(), ["Train", "Validation", "Test"])
        self.assertEqual(create_split_labels(historical, train_end="2023-03-16",
                                            val_end="2023-04-14").tolist(), ["Train", "Train", "Validation"])
        with self.assertRaisesRegex(ValueError, "override"):
            create_split_labels(frame, train_end="2024-04-01")

    def test_combined_join_safe_features_and_counts(self):
        folder, metadata, frame = self.prepare()
        self.assertEqual(len(frame), 1200)
        self.assertTrue(frame.order_id.str.startswith("0").all())
        self.assertTrue(frame.order_id.duplicated().any())  # valid across countries
        self.assertFalse(frame.duplicated(["country_code", "order_id"]).any())
        expected = {"Train": 528, "Validation": 336, "Test": 336}
        self.assertEqual({k: v["rows"] for k, v in metadata["partitions"].items()}, expected)
        self.assertEqual(metadata["feature_count"], 31)
        self.assertEqual(metadata["sha256"], mu.file_sha256(folder / "features_base_safe.csv"))
        self.assertEqual(metadata["split"], frame.attrs["retraining_split"])
        for stats in metadata["partitions"].values():
            self.assertEqual(stats["fraud"], stats["non_fraud"])
        self.assertTrue(set(metadata["features"]).isdisjoint(dp.SNAPSHOT_FEATURES + mu.FORBIDDEN_FEATURES))
        self.assertFalse((folder / "features_base_full.csv").exists())
        self.assertEqual(set(folder.iterdir()), {folder / "features_base_safe.csv",
                                                folder / "features_base_safe.metadata.json"})
        with self.assertRaises(FileExistsError):
            preparation.prepare_retraining_data(output_dir=folder, engine=self.engine)

    def test_new_rows_move_windows_and_current_label_replaces_old(self):
        _, old, previous = self.prepare("before")
        populate_database(self.engine, start="2024-04-10", days=28, append=True)
        with self.engine.begin() as connection:
            connection.execute(text("UPDATE fraud_labels SET is_fraud=1 "
                                    "WHERE country_code='MY' AND order_id='00000'"))
        _, new, current = self.prepare("after")
        self.assertEqual(len(current), 1536)
        self.assertEqual(new["total_rows"], old["total_rows"] + 336)
        self.assertEqual(current.is_fraud.sum(), previous.is_fraud.sum() + 168 + 1)
        self.assertEqual(new["split"]["anchor_date"], "2024-05-07")
        before = dict(zip(zip(previous.country_code, previous.order_id), create_split_labels(previous)))
        after = dict(zip(zip(current.country_code, current.order_id), create_split_labels(current)))
        for date, old_split, new_split in [("2024-03-12", "Validation", "Train"),
                                           ("2024-04-09", "Test", "Validation"),
                                           ("2024-05-07", None, "Test")]:
            row = current.loc[pd.to_datetime(current.order_date).dt.normalize() == pd.Timestamp(date)].iloc[0]
            key = (row.country_code, row.order_id)
            self.assertEqual(after[key], new_split)
            if old_split is not None:
                self.assertEqual(before[key], old_split)

    def test_invalid_or_insufficient_database_rows_stop_preparation(self):
        for name, query, message in [
            ("empty", "DELETE FROM fraud_labels", "No labelled orders"),
            ("one_class", "UPDATE fraud_labels SET is_fraud=0", "both fraud"),
            ("missing_dates", "UPDATE orders SET order_date=NULL", "valid order dates"),
            ("duplicate", "INSERT INTO orders SELECT * FROM orders LIMIT 1", "Duplicate"),
            ("short_history", "UPDATE orders SET order_date='2024-04-01'", "Train must contain"),
        ]:
            # A transaction rollback restores the fixture for the next case.
            with self.subTest(case=name), self.engine.connect() as connection:
                transaction = connection.begin()
                connection.execute(text(query))
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        # Connection is accepted by read_sql; preserve uncommitted test state.
                        with patch.object(preparation.db, "load_training_data", wraps=lambda engine:
                                          pd.read_sql(text(preparation.db.TRAINING_QUERY), connection,
                                                      parse_dates=["order_date", "first_order_datetime"])):
                            preparation.prepare_retraining_data(output_dir=self.root / name, engine=self.engine)
                finally:
                    transaction.rollback()
                self.assertFalse((self.root / name / "features_base_safe.metadata.json").exists())

    def test_metadata_and_data_tampering_are_rejected(self):
        folder, metadata, _ = self.prepare()
        path = folder / "features_base_safe.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            mu.load_feature_data(path)
        path.write_bytes(original)
        sidecar = path.with_suffix(".metadata.json")
        for field in ("split", "partitions"):
            altered = json.loads(json.dumps(metadata))
            if field == "split":
                altered[field]["test_start"] = "2024-03-01"
            else:
                altered[field]["Train"]["fraud"] += 1
            sidecar.write_text(json.dumps(altered), encoding="utf-8")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "metadata|counts"):
                mu.load_feature_data(path)

    def test_prepared_csv_and_metadata_are_frozen_together(self):
        folder, metadata, _ = self.prepare()
        original = (folder / "features_base_safe.metadata.json").read_bytes()
        def train(key, **kwargs):
            path = Path(kwargs["data_dir"]) / "features_base_safe.csv"
            self.assertEqual(path.with_suffix(".metadata.json").read_bytes(), original)
            self.assertEqual(mu.load_feature_data(path).attrs["retraining_split"], metadata["split"])
            (folder / "features_base_safe.metadata.json").write_text("changed source")
            return [{"dataset_type": "safe"}], {}
        with patch.object(workflow, "run_model", side_effect=train), redirect_stdout(io.StringIO()):
            report = workflow.run_retraining(data_dir=folder, output_dir=self.root / "runs")
        self.assertEqual(report["input"]["preparation"], metadata)

    def test_database_failure_skips_all_models_and_reports_error(self):
        with patch.object(preparation.db, "load_training_data", side_effect=RuntimeError("database unavailable")), \
             patch.object(workflow, "run_model") as runner, redirect_stdout(io.StringIO()):
            with self.assertRaises(workflow.RetrainingFailed) as failed:
                workflow.run_retraining(from_database=True, engine=self.engine, output_dir=self.root / "runs")
        runner.assert_not_called()
        self.assertIn("database unavailable", failed.exception.report["input"]["error"])
        self.assertTrue(all(task["status"] == "skipped" for task in failed.exception.report["tasks"]))

    def test_database_to_all_eight_real_safe_models(self):
        with redirect_stdout(io.StringIO()):
            report = workflow.run_retraining(from_database=True, engine=self.engine, output_dir=self.root / "runs")
        self.assertEqual(report["status"], "completed")
        metadata = report["input"]["preparation"]
        expected_counts = {k: v["rows"] for k, v in metadata["partitions"].items()}
        self.assertEqual(len(report["tasks"]), 8)
        for task in report["tasks"]:
            result = task["results"][0]
            self.assertEqual(task["status"], "completed")
            self.assertEqual(result["rows"], expected_counts)
            self.assertEqual(result["feature_count"], 31)
            self.assertEqual(result["data_preparation"], metadata)
            saved = json.loads((Path(result["artifact_dir"]) / "result.json").read_text())
            self.assertEqual(saved["data_preparation"], metadata)
            self.assertEqual(result["dataset_sha256"], metadata["sha256"])
        evidence = {"source": "local SQLite; synthetic data; no AWS/MLflow connection",
                    "status": report["status"], "preparation": metadata,
                    "models_completed": [task["model_key"] for task in report["tasks"]]}
        (ROOT / "outputs").mkdir(exist_ok=True)
        (ROOT / "outputs" / "task50_data_verification.json").write_text(json.dumps(evidence, indent=2))

    def test_invalid_source_options_and_cli_error(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            workflow.run_retraining(data_dir="data", from_database=True)
        with self.assertRaisesRegex(ValueError, "requires"):
            workflow.run_retraining(engine=self.engine)
        with patch.object(sys, "argv", ["retraining_workflow.py", "--from-database", "--data-dir", "data"]), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            workflow.main()
        self.assertEqual(stopped.exception.code, 2)
        with patch.object(sys, "argv", ["retraining_data.py", "--output-dir", "unused"]), \
             patch.object(preparation, "prepare_retraining_data", side_effect=RuntimeError("test failure")), \
             redirect_stderr(io.StringIO()):
            self.assertEqual(preparation.main(), 1)


if __name__ == "__main__":
    unittest.main()
