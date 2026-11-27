"""Tests for the new customer risk analytics (Task #62) and its Prefect flow (Task #68).

They use a small in-memory SQLite database, so no RDS connection is needed:

    python -m unittest tests.test_analytics_new_customer_emi -v
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "flows"))

import analytics_new_customer_emi as analytics  # noqa: E402


def make_orders():
    """12 labelled orders: 6 from new customers, 5 established, 1 with an invalid first-order date."""
    rows = [
        # customer, first order, order date, value, verified, fraud
        ("c1", "2023-05-01", "2023-05-01", 20.0, True, 1),    # new, same day, high value
        ("c2", "2023-04-25", "2023-05-01", 19.0, True, 1),    # new, 6 days, high value
        ("c3", "2023-04-10", "2023-05-01", 2.0, True, 1),     # new, 21 days
        ("c4", "2023-04-10", "2023-05-01", 3.0, True, 0),     # new, 21 days
        ("c5", "2023-04-05", "2023-05-01", 4.0, False, 0),    # new, 26 days, not verified
        ("c6", "2023-04-01", "2023-05-01", 5.0, True, 0),     # new, exactly 30 days
        ("c7", "2023-03-31", "2023-05-01", 18.0, True, 1),    # established, 31 days, high value
        ("c8", "2022-01-01", "2023-05-01", 2.0, True, 0),     # established
        ("c9", "2022-01-01", "2023-05-01", 3.0, True, 0),     # established
        ("c10", "2022-01-01", "2023-05-01", 4.0, True, 0),    # established
        ("c11", "2022-01-01", "2023-05-01", 6.0, True, 0),    # established
        ("c12", "2023-06-01", "2023-05-01", 7.0, True, 0),    # first order AFTER this order: unknown
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "first_order_datetime", "order_date",
                                     "order_value", "mobile_verified", "is_fraud"])
    df["country_code"] = "MY"
    df["order_id"] = [f"o{i}" for i in range(1, len(df) + 1)]
    df["first_order_datetime"] = pd.to_datetime(df["first_order_datetime"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.df = analytics.add_customer_segments(make_orders()).set_index("customer_id")

    def test_new_established_and_unknown(self):
        segment = self.df["customer_segment"]
        self.assertEqual(segment["c1"], analytics.NEW)
        self.assertEqual(segment["c6"], analytics.NEW)              # day 30 is still new
        self.assertEqual(segment["c7"], analytics.ESTABLISHED)      # day 31 is established
        self.assertEqual(segment["c12"], analytics.UNKNOWN)         # negative account age
        self.assertEqual((segment == analytics.NEW).sum(), 6)

    def test_age_bands(self):
        self.assertEqual(self.df.loc["c1", "age_band"], "Same day")
        self.assertEqual(self.df.loc["c2", "age_band"], "1-7 days")
        self.assertEqual(self.df.loc["c6", "age_band"], "8-30 days")
        self.assertEqual(self.df.loc["c7", "age_band"], "31-90 days")
        self.assertEqual(self.df.loc["c8", "age_band"], "Over 1 year")

    def test_high_value_is_top_quarter_within_the_country(self):
        high = self.df.index[self.df["value_segment"] == analytics.HIGH]
        self.assertEqual(sorted(high), ["c1", "c2", "c7"])


class QuestionTests(unittest.TestCase):
    def setUp(self):
        self.df = analytics.add_customer_segments(make_orders())

    def test_q1_new_customers_fraud_rate(self):
        findings = analytics.q1_account_age(self.df)["findings"]
        self.assertAlmostEqual(findings["new_fraud_rate_pct"], 50.0)          # 3 of 6
        self.assertAlmostEqual(findings["established_fraud_rate_pct"], 20.0)  # 1 of 5
        self.assertAlmostEqual(findings["times_higher"], 2.5)

    def test_q2_mobile_verification(self):
        findings = analytics.q2_mobile_verification(self.df)["findings"]
        self.assertEqual(findings["new_unverified_orders"], 1)
        self.assertAlmostEqual(findings["new_verified_fraud_rate_pct"], 60.0)  # 3 of 5

    def test_q3_high_value_orders(self):
        result = analytics.q3_high_value_orders(self.df)
        self.assertAlmostEqual(result["findings"]["new_high_value_fraud_rate_pct"], 100.0)
        self.assertEqual(result["findings"]["new_high_value_orders"], 2)

    def test_confidence_interval_contains_the_rate(self):
        low, high = analytics.wilson_interval([3], [6])
        self.assertLess(low[0], 0.5)
        self.assertGreater(high[0], 0.5)


class OutputTests(unittest.TestCase):
    def test_run_analysis_reads_the_database_and_saves_everything(self):
        engine = create_engine("sqlite://")
        orders = make_orders()
        orders[["country_code", "order_id", "customer_id", "is_fraud"]].to_sql("fraud_labels", engine, index=False)
        orders.assign(collect_type="Delivery", payment_method="balance", num_items_ordered=1,
                      refund_value=0.0)[["country_code", "order_id", "collect_type", "payment_method",
                                         "order_value", "num_items_ordered", "refund_value",
                                         "order_date"]].to_sql("orders", engine, index=False)
        orders.assign(num_orders_last_50days=1, num_cancelled_orders_last_50days=0,
                      num_refund_orders_last_50days=0, total_payment_last_50days=5.0,
                      num_associated_customers=1)[["country_code", "customer_id", "mobile_verified",
                                                   "num_orders_last_50days", "num_cancelled_orders_last_50days",
                                                   "num_refund_orders_last_50days", "total_payment_last_50days",
                                                   "num_associated_customers", "first_order_datetime"]
                                                  ].to_sql("customers", engine, index=False)
        with tempfile.TemporaryDirectory() as folder:
            result = analytics.run_analysis(folder, engine=engine)
            self.assertEqual(len(result["data"]), 12)
            for path in result["files"].values():
                self.assertTrue(Path(path).is_file() and Path(path).stat().st_size > 0, path)
            summary = Path(result["files"]["summary"]).read_text(encoding="utf-8")
            self.assertIn("50.0% fraud rate vs 20.0%", summary)


@unittest.skipUnless(importlib.util.find_spec("prefect"), "prefect is not installed")
class FlowTests(unittest.TestCase):
    def test_flow_runs_all_tasks(self):
        from prefect.testing.utilities import prefect_test_harness
        import new_customer_risk_flow_emi as flow_module

        original = analytics.load_orders
        analytics.load_orders = lambda engine=None: make_orders()
        try:
            with prefect_test_harness(), tempfile.TemporaryDirectory() as folder:
                summary = flow_module.new_customer_risk_flow(output_dir=folder)
                self.assertIn("Q1. Are customers in their first 30 days riskier?", summary)
                self.assertTrue((Path(folder) / "q3_fraud_by_order_value.png").is_file())
        finally:
            analytics.load_orders = original


if __name__ == "__main__":
    unittest.main()
