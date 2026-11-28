"""Tests for data quality monitoring (PBI 11, issue #85, Task #91).

Run from the repository root:

    python -m unittest tests.test_data_quality -v

The sample batches contain valid records, missing values, duplicates, invalid types
and schema errors (#91). No database is needed: the ETL runs with --dry-run
behaviour, except the opt-in database test, which uses the same ETL_TEST_DATABASE
practice database as tests/test_etl_pipeline.py (never the team's "spbuy").
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_quality as dq  # noqa: E402
import etl_pipeline as etl  # noqa: E402
from test_etl_pipeline import GOOD_CUSTOMERS, GOOD_LABELS, GOOD_ORDERS, write_batch  # noqa: E402


class SampleBatch:
    """Writes a batch, extracts it with the ETL's reader and runs the quality checks."""

    def __init__(self, test, customers=GOOD_CUSTOMERS, orders=GOOD_ORDERS, labels=GOOD_LABELS, postfix="t"):
        self.folder = Path(test.folder.name)
        self.sources = write_batch(self.folder, postfix, customers, orders, labels)
        self.raw = etl.extract(self.sources, postfix, raw_dir=self.folder / "raw")
        self.quality = dq.inspect_batch(self.raw)
        self.issues = self.quality["issues"]
        self.counts = self.quality["counts"].set_index("dataset")

    def find(self, dataset, check, column=None):
        rows = self.issues[(self.issues["dataset"] == dataset) & (self.issues["check"] == check)]
        if column is not None:
            rows = rows[rows["column"] == column]
        return rows


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_valid_batch_has_no_issues(self):
        batch = SampleBatch(self)
        self.assertTrue(batch.issues.empty)
        self.assertEqual(dq.batch_status(batch.issues), "passed")
        self.assertEqual(batch.counts.loc["orders", "records_with_issues"], 0)

    def test_missing_values_required_are_errors_optional_are_warnings(self):
        orders = GOOD_ORDERS + ["MY,o3,Delivery,,5,,0.0,2023/05/03",          # payment + items empty: filled
                                "MY,o4,Delivery,credit card,,1,0.0,2023/05/03"]  # order_value empty: rejected
        customers = GOOD_CUSTOMERS + ["MY,c3,True,1,0,0,1.0,1,"]                # first order date may be empty
        batch = SampleBatch(self, customers=customers, orders=orders)
        self.assertEqual(batch.find("orders", "missing required value", "order_value")["severity"].item(), "error")
        self.assertEqual(batch.find("orders", "missing optional value", "payment_method")["severity"].item(), "warning")
        self.assertEqual(batch.find("orders", "missing optional value", "num_items_ordered")["affected_records"].item(), 1)
        self.assertEqual(batch.find("customers", "missing optional value", "first_order_datetime")["severity"].item(),
                         "warning")
        self.assertEqual(batch.counts.loc["orders", "records_with_issues"], 2)
        self.assertEqual(batch.counts.loc["orders", "records_with_errors"], 1)

    def test_exact_duplicate_rows_are_reported(self):
        batch = SampleBatch(self, orders=GOOD_ORDERS + [GOOD_ORDERS[0]])           # identical copy
        self.assertEqual(batch.find("orders", "exact duplicate row")["affected_records"].item(), 1)
        self.assertTrue(batch.find("orders", "repeated record key, different values").empty)
        _, rejected = etl.transform(batch.raw)                  # the ETL rejects the earlier copy
        self.assertEqual(batch.counts.loc["orders", "records_with_errors"], len(rejected))

    def test_repeated_key_after_normalisation_is_detected(self):
        orders = GOOD_ORDERS + ["pk,o2,Delivery,credit card,8.0,1,0.0,2023-05-02"]   # PK-o2 again, lower case
        batch = SampleBatch(self, orders=orders)
        repeated = batch.find("orders", "repeated record key, different values")
        self.assertEqual(repeated["affected_records"].item(), 1)
        self.assertIn("PK-o2", repeated["examples"].item())

    def test_invalid_types_and_values(self):
        customers = GOOD_CUSTOMERS + ["MY,c3,maybe,1,0,0,1.0,1,2022-01-01 00:00:00",     # boolean
                                      "MY,c4,True,1.5,0,0,1.0,1,2022-01-01 00:00:00",    # not whole
                                      "MY,c5,True,1,0,0,abc,1,2022-01-01 00:00:00",      # not a number
                                      "MY,c6,True,1,0,0,1.0,1,someday"]                  # date/time
        orders = GOOD_ORDERS + ["MY,o3,drone,credit card,5,1,0.0,2023/05/03",            # collect type
                                "MY,o4,Delivery,credit card,-5,1,0.0,2023/05/03",        # negative
                                "MY,o5,Delivery,credit card,5,0,0.0,2023/05/03",         # items < 1
                                "MY,o6,Delivery,credit card,5,1,0.0,03-05-2023",         # date format
                                "XX,o7,Delivery,crypto,5,1,0.0,2023/05/03"]              # unknown country, method
        labels = GOOD_LABELS + ["MY,o3,c1,2"]                                             # label not 0/1
        batch = SampleBatch(self, customers=customers, orders=orders, labels=labels)
        expected = [("customers", "invalid boolean", "error"), ("customers", "not a whole number", "error"),
                    ("customers", "not a number", "error"), ("customers", "invalid date/time", "error"),
                    ("orders", "unknown collect_type", "error"), ("orders", "negative value", "error"),
                    ("orders", "invalid item count", "error"), ("orders", "invalid date", "error"),
                    ("orders", "unknown country code", "warning"), ("orders", "new payment method", "warning"),
                    ("labels", "invalid fraud label", "error")]
        for dataset, check, severity in expected:
            found = batch.find(dataset, check)
            self.assertFalse(found.empty, f"{dataset}: {check} not detected")
            self.assertEqual(found["severity"].iloc[0], severity)
        self.assertIn("drone", batch.find("orders", "unknown collect_type")["examples"].item())

    def test_schema_extra_column_and_wrong_number_of_fields(self):
        batch = SampleBatch(self, orders=GOOD_ORDERS + ["MY,o4,Delivery,credit card,5,1,0.0,2023/05/03,EXTRA"])
        self.assertEqual(batch.find("orders", "wrong number of fields")["affected_records"].item(), 1)
        self.assertEqual(batch.counts.loc["orders", "input_records"], 3)
        Path(batch.sources["labels"]).write_text("country_code,order_id,customer_id,is_fraud,note\n"
                                                 + "\n".join(f"{row},x" for row in GOOD_LABELS) + "\n")
        raw = etl.extract(batch.sources, "t", raw_dir=self.folder.name)
        issues = dq.inspect_batch(raw)["issues"]
        extra = issues[issues["check"] == "unexpected column"]
        self.assertEqual((extra["column"].item(), extra["severity"].item()), ("note", "warning"))

    def test_records_flagged_as_errors_are_the_ones_the_etl_rejects(self):
        orders = GOOD_ORDERS + ["MY,o3,drone,credit card,5,1,0.0,2023/05/03",
                                "MY,o4,Delivery,credit card,-5,1,0.0,2023/05/03",
                                "MY,o5,Delivery,,5,,0.0,2023/05/03"]                     # warnings only
        customers = GOOD_CUSTOMERS + ["MY,c3,maybe,1,0,0,1.0,1,2022-01-01 00:00:00"]
        batch = SampleBatch(self, customers=customers, orders=orders)
        _, rejected = etl.transform(batch.raw)
        by_table = pd.Series([r["source_table"] for r in rejected]).value_counts()
        self.assertEqual(batch.counts.loc["orders", "records_with_errors"], by_table["orders"])
        self.assertEqual(batch.counts.loc["customers", "records_with_errors"], by_table["customers"])


class PipelineReportTests(unittest.TestCase):
    """The checks run automatically inside run_etl and write a report per batch (#85 AC3-AC6)."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.report_dir = self.base / "quality"

    def run_dry(self, postfix, customers=GOOD_CUSTOMERS, orders=GOOD_ORDERS, labels=GOOD_LABELS):
        sources = write_batch(self.base, postfix, customers, orders, labels)
        return etl.run_etl(sources, raw_dir=self.base / "raw", dry_run=True,
                           summary_dir=self.base / "summary", quality_dir=self.report_dir)

    def test_report_is_generated_with_processing_summary(self):
        orders = GOOD_ORDERS + ["MY,o3,drone,credit card,5,1,0.0,2023/05/03", "MY,o5,Delivery,,5,,0.0,2023/05/03"]
        result = self.run_dry("b1", orders=orders)
        folder = Path(result["data_quality"]["report_dir"])
        for name in ("quality_issues.csv", "processing_summary.csv", "report.md"):
            self.assertTrue((folder / name).is_file(), name)
        summary = pd.read_csv(folder / "processing_summary.csv").set_index("dataset")
        self.assertEqual(summary.loc["orders", ["input_records", "records_with_issues", "records_with_errors",
                                                "records_rejected", "records_processed"]].tolist(), [4, 2, 1, 1, 3])
        self.assertIn("unrecognised collect_type: 1", summary.loc["orders", "rejection_reasons"])
        self.assertEqual(result["data_quality"]["status"], "issues found: affected records rejected")

    def test_each_batch_gets_its_own_report_and_a_history_row(self):
        self.run_dry("b1")
        self.run_dry("b2", orders=GOOD_ORDERS + ["MY,o5,Delivery,,5,,0.0,2023/05/03"])   # warnings only
        self.assertTrue((self.report_dir / "b1" / "report.md").is_file())
        self.assertTrue((self.report_dir / "b2" / "report.md").is_file())
        history = pd.read_csv(self.report_dir / "history.csv")
        self.assertEqual(sorted(history["batch_name"].unique()), ["b1", "b2"])
        self.assertEqual(len(history), 6)                           # 3 datasets x 2 batches
        statuses = history.groupby("batch_name")["status"].first().to_dict()
        self.assertEqual(statuses, {"b1": "passed", "b2": "passed with warnings"})

    def test_missing_required_column_is_critical_and_stops_the_batch(self):
        sources = write_batch(self.base, "bad", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS)
        Path(sources["labels"]).write_text("country_code,order_id\nMY,o1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            etl.run_etl(sources, raw_dir=self.base / "raw", dry_run=True, summary_dir=self.base / "summary",
                        quality_dir=self.report_dir)
        issues = pd.read_csv(self.report_dir / "bad" / "quality_issues.csv")
        critical = issues[issues["severity"] == "critical"]
        self.assertEqual(critical["dataset"].tolist(), ["labels"])
        self.assertIn("customer_id", critical["column"].item())
        self.assertIn("critical issue", (self.report_dir / "bad" / "report.md").read_text())
        self.assertFalse((self.base / "summary" / "bad_summary.json").exists())   # nothing went further

    def test_unreadable_or_empty_file_is_reported_as_critical(self):
        sources = write_batch(self.base, "bin", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS)
        Path(sources["orders"]).write_bytes("country_code".encode("utf-16"))
        with self.assertRaises(ValueError):
            etl.run_etl(sources, raw_dir=self.base / "raw", dry_run=True, summary_dir=self.base / "summary",
                        quality_dir=self.report_dir)
        issues = pd.read_csv(self.report_dir / "bin" / "quality_issues.csv")
        self.assertEqual(issues.loc[issues["severity"] == "critical", ["dataset", "check"]].values.tolist(),
                         [["orders", "unreadable file"]])

    def test_failed_cleaned_data_validation_still_writes_a_report(self):
        original = etl.dp.validate_or_raise

        def always_fail(*args):
            raise ValueError("Cleaned-data validation failed: test")
        etl.dp.validate_or_raise = always_fail
        self.addCleanup(setattr, etl.dp, "validate_or_raise", original)
        with self.assertRaises(ValueError):
            self.run_dry("val")
        report = (self.report_dir / "val" / "report.md").read_text()
        self.assertIn("pipeline stopped", report)
        self.assertIn("critical issue", report)


@unittest.skipUnless(os.environ.get("ETL_TEST_DATABASE"), "set ETL_TEST_DATABASE to run the database test")
class DatabaseReportTests(unittest.TestCase):
    """A loaded batch: 'records_processed' comes from the rows actually inserted or updated in RDS."""

    @classmethod
    def setUpClass(cls):
        import database as db
        if os.environ["ETL_TEST_DATABASE"] == db.DEFAULT_DB_NAME:
            raise unittest.SkipTest("refusing to run tests against the team database 'spbuy'")
        os.environ["DB_NAME"] = os.environ["ETL_TEST_DATABASE"]
        db.create_database()
        cls.engine = db.get_engine()
        from sqlalchemy import text
        with cls.engine.begin() as conn:
            for table in ("fraud_labels", "orders", "customers", "rejected_records", "etl_batches"):
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        db.create_tables(cls.engine)

    def test_loaded_batch_report_uses_database_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            labels = GOOD_LABELS + ["MY,o9,c1,0"]                       # orphan label: rejected during load
            sources = write_batch(folder, "dq1", GOOD_CUSTOMERS, GOOD_ORDERS, labels)
            result = etl.run_etl(sources, raw_dir=Path(folder) / "raw", engine=self.engine,
                                 summary_dir=Path(folder) / "summary", quality_dir=Path(folder) / "quality")
            summary = pd.read_csv(Path(result["data_quality"]["report_dir"]) / "processing_summary.csv")
            labels_row = summary.set_index("dataset").loc["labels"]
            self.assertEqual((labels_row["records_processed"], labels_row["records_rejected"]), (2, 1))
            self.assertIn("order not found", labels_row["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
