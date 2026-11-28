"""Tests for the new customer risk analytics (Task #62) and its Prefect flow (Task #68).

The unit tests need no database:

    python -m unittest tests.test_analytics_new_customer_emi -v

The MySQL test checks that the counting done inside MySQL gives exactly the same
numbers as the pandas version. It runs only when ETL_TEST_DATABASE names a SEPARATE
practice database (never the team's "spbuy"), with DB_HOST, DB_USER, DB_PASSWORD set:

    $env:ETL_TEST_DATABASE = "spbuy_practice"
    python -m unittest tests.test_analytics_new_customer_emi -v
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "flows"))

import analytics_new_customer_emi as analytics  # noqa: E402


def make_orders():
    """12 labelled orders: 6 from new customers, 5 established, 1 with an invalid first-order date."""
    rows = [
        # customer, first order, order date, value, verified, fraud
        ("c1", "2023-05-01 09:00:00", "2023-05-01", 20.0, True, 1),    # new, same day, high value
        ("c2", "2023-04-25 23:59:00", "2023-05-01", 19.0, True, 1),    # new, 6 days, high value
        ("c3", "2023-04-10 10:00:00", "2023-05-01", 2.0, True, 1),     # new, 21 days
        ("c4", "2023-04-10 10:00:00", "2023-05-01", 3.0, True, 0),     # new, 21 days
        ("c5", "2023-04-05 10:00:00", "2023-05-01", 4.0, False, 0),    # new, 26 days, not verified
        ("c6", "2023-04-01 10:00:00", "2023-05-01", 5.0, True, 0),     # new, exactly 30 days
        ("c7", "2023-03-31 10:00:00", "2023-05-01", 18.0, True, 1),    # established, 31 days, high value
        ("c8", "2022-01-01 10:00:00", "2023-05-01", 2.0, True, 0),     # established
        ("c9", "2022-01-01 10:00:00", "2023-05-01", 3.0, True, 0),     # established
        ("c10", "2022-01-01 10:00:00", "2023-05-01", 4.0, True, 0),    # established
        ("c11", None, "2023-05-02", 6.0, True, 0),                     # missing first order: unknown
        ("c12", "2023-06-01 10:00:00", "2023-05-01", 7.0, True, 0),    # first order AFTER this order: unknown
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "first_order_datetime", "order_date",
                                     "order_value", "mobile_verified", "is_fraud"])
    df["country_code"] = "MY"
    df["order_id"] = [f"o{i}" for i in range(1, len(df) + 1)]
    df["first_order_datetime"] = pd.to_datetime(df["first_order_datetime"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def group(counts, **where):
    rows = counts
    for column, value in where.items():
        rows = rows[rows[column] == value]
    return int(rows["orders"].sum()), int(rows["frauds"].sum())


class CountTests(unittest.TestCase):
    def setUp(self):
        self.counts, self.overview, self.cutoffs = analytics.count_orders(make_orders())

    def test_new_established_and_unknown(self):
        self.assertEqual(group(self.counts, customer_segment=analytics.NEW), (6, 3))           # day 0-30
        self.assertEqual(group(self.counts, customer_segment=analytics.ESTABLISHED), (4, 1))   # day 31+
        self.assertEqual(group(self.counts, customer_segment=analytics.UNKNOWN), (2, 0))       # missing / negative

    def test_age_bands(self):
        self.assertEqual(group(self.counts, age_band="Same day"), (1, 1))
        self.assertEqual(group(self.counts, age_band="1-7 days"), (1, 1))
        self.assertEqual(group(self.counts, age_band="8-30 days"), (4, 1))       # 21, 21, 26 and 30 days
        self.assertEqual(group(self.counts, age_band="31-90 days"), (1, 1))
        self.assertEqual(group(self.counts, age_band="Over 1 year"), (3, 0))

    def test_high_value_is_top_quarter_within_the_country(self):
        self.assertEqual(group(self.counts, value_segment=analytics.HIGH), (3, 3))   # 18, 19 and 20
        self.assertAlmostEqual(self.cutoffs["high_value_cutoff"].iloc[0], 9.75)

    def test_overview(self):
        self.assertEqual((self.overview["orders"], self.overview["frauds"]), (12, 4))
        self.assertEqual(self.overview["last_order_date"], pd.Timestamp("2023-05-02"))


class QuestionTests(unittest.TestCase):
    def setUp(self):
        self.counts, self.overview, self.cutoffs = analytics.count_orders(make_orders())

    def test_q1_new_customers_fraud_rate(self):
        findings = analytics.q1_account_age(self.counts)["findings"]
        self.assertAlmostEqual(findings["new_fraud_rate_pct"], 50.0)          # 3 of 6
        self.assertAlmostEqual(findings["established_fraud_rate_pct"], 25.0)  # 1 of 4
        self.assertAlmostEqual(findings["times_higher"], 2.0)

    def test_q2_mobile_verification(self):
        findings = analytics.q2_mobile_verification(self.counts)["findings"]
        self.assertEqual(findings["new_unverified_orders"], 1)
        self.assertAlmostEqual(findings["new_verified_fraud_rate_pct"], 60.0)  # 3 of 5

    def test_q3_high_value_orders(self):
        findings = analytics.q3_high_value_orders(self.counts, self.cutoffs)["findings"]
        self.assertAlmostEqual(findings["new_high_value_fraud_rate_pct"], 100.0)
        self.assertEqual(findings["new_high_value_orders"], 2)

    def test_confidence_interval_contains_the_rate(self):
        low, high = analytics.wilson_interval([3], [6])
        self.assertLess(low[0], 0.5)
        self.assertGreater(high[0], 0.5)

    def test_analyse_saves_tables_charts_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            result = analytics.analyse(self.counts, self.overview, self.cutoffs, folder)
            for path in result["files"].values():
                self.assertTrue(Path(path).is_file() and Path(path).stat().st_size > 0, path)
            summary = Path(result["files"]["summary"]).read_text(encoding="utf-8")
            self.assertIn("50.0% fraud rate vs 25.0%", summary)


@unittest.skipUnless(os.environ.get("ETL_TEST_DATABASE"),
                     "set ETL_TEST_DATABASE (a separate practice database) to run the MySQL test")
class MySqlCountTests(unittest.TestCase):
    def test_mysql_counts_match_pandas(self):
        import database as db
        name = os.environ["ETL_TEST_DATABASE"]
        if name == db.DEFAULT_DB_NAME:
            self.skipTest("refusing to run tests against the team database 'spbuy'")
        config = {**db.get_db_config(), "database": name}
        db.create_database(config)
        engine = db.get_engine(config=config)
        db.create_tables(engine)
        if any(db.check_tables(engine).values()):
            self.skipTest(f"practice database {name!r} is not empty")
        orders = make_orders()
        batch = {"first_loaded_batch": "test", "last_updated_batch": "test"}
        try:
            orders.assign(num_orders_last_50days=1, num_cancelled_orders_last_50days=0,
                          num_refund_orders_last_50days=0, total_payment_last_50days=5.0,
                          num_associated_customers=1, **batch)[
                ["country_code", "customer_id", "mobile_verified", "num_orders_last_50days",
                 "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
                 "total_payment_last_50days", "num_associated_customers", "first_order_datetime",
                 "first_loaded_batch", "last_updated_batch"]].to_sql("customers", engine, if_exists="append", index=False)
            orders.assign(collect_type="Delivery", payment_method="balance", num_items_ordered=1,
                          refund_value=0.0, order_date=orders["order_date"].dt.date, **batch)[
                ["country_code", "order_id", "collect_type", "payment_method", "order_value",
                 "num_items_ordered", "refund_value", "order_date", "first_loaded_batch",
                 "last_updated_batch"]].to_sql("orders", engine, if_exists="append", index=False)
            orders.assign(**batch)[["country_code", "order_id", "customer_id", "is_fraud",
                                    "first_loaded_batch", "last_updated_batch"]].to_sql(
                "fraud_labels", engine, if_exists="append", index=False)

            counts, overview, cutoffs = analytics.load_counts(engine)
            expected_counts, expected_overview, expected_cutoffs = analytics.count_orders(orders)
            pd.testing.assert_frame_equal(counts, expected_counts)
            self.assertEqual(overview, expected_overview)
            self.assertAlmostEqual(cutoffs["high_value_cutoff"].iloc[0], expected_cutoffs["high_value_cutoff"].iloc[0])
        finally:                       # leave the practice database empty again
            from sqlalchemy import text
            with engine.begin() as conn:
                for table in ("fraud_labels", "orders", "customers"):
                    conn.execute(text(f"DELETE FROM {table}"))


@unittest.skipUnless(importlib.util.find_spec("prefect"), "prefect is not installed")
class FlowTests(unittest.TestCase):
    def test_flow_runs_all_tasks(self):
        from prefect.testing.utilities import prefect_test_harness
        import new_customer_risk_flow_emi as flow_module

        counts, overview, cutoffs = analytics.count_orders(make_orders())
        originals = analytics.high_value_cutoffs, analytics.count_orders_in_db
        analytics.high_value_cutoffs = lambda engine=None: cutoffs
        analytics.count_orders_in_db = lambda cut, engine=None: (counts, overview)
        try:
            with prefect_test_harness(), tempfile.TemporaryDirectory() as folder:
                summary = flow_module.new_customer_risk_flow(output_dir=folder)
                self.assertIn("Q1. Are customers in their first 30 days riskier?", summary)
                self.assertTrue((Path(folder) / "q3_fraud_by_order_value.png").is_file())
        finally:
            analytics.high_value_cutoffs, analytics.count_orders_in_db = originals


if __name__ == "__main__":
    unittest.main()
