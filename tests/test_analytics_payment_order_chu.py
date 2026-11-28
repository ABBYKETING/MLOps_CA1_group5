"""Task #63: actual aggregate SQL on isolated SQLite fixtures; no RDS writes."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analytics_payment_order_chu as analytics


class PaymentAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        self.addCleanup(self.engine.dispose)
        rows = [
            ("MY", "same", "credit card", "Delivery", 0, 0),
            ("PH", "same", "credit card", "Pickup", 10, 1),
            ("MY", "b", "credit card", "Delivery", 10.01, 0),
            ("MY", "c", "credit card", "Pickup", 25, 1),
            ("MY", "d", "credit card", "Delivery", 25.01, 0),
            ("MY", "e", "payment on delivery", "Delivery", 50, 1),
            ("MY", "f", "payment on delivery", "Pickup", 50.01, 0),
            ("PH", "g", "payment on delivery", "Delivery", 100, 1),
            ("PH", "h", "unknown", "Delivery", 100.01, 0),
            ("MY", "i", "no payment", "Pickup", 0, 1),
            ("PH", "j", "credit card", "Delivery", 101, 1),
            ("MY", "unlabelled", "credit card", "Delivery", 30, None),
        ]
        with self.engine.begin() as c:
            c.execute(text("CREATE TABLE orders(country_code TEXT, order_id TEXT, payment_method TEXT, "
                           "collect_type TEXT, order_value REAL, order_date TEXT, PRIMARY KEY(country_code,order_id))"))
            c.execute(text("CREATE TABLE fraud_labels(country_code TEXT, order_id TEXT, is_fraud INTEGER, "
                           "PRIMARY KEY(country_code,order_id))"))
            for country, oid, method, collection, value, fraud in rows:
                c.execute(text("INSERT INTO orders VALUES(:country,:oid,:method,:collection,:value,'2023-05-01')"),
                          dict(country=country, oid=oid, method=method, collection=collection, value=value))
                if fraud is not None:
                    c.execute(text("INSERT INTO fraud_labels VALUES(:country,:oid,:fraud)"),
                              dict(country=country, oid=oid, fraud=fraud))
        self.groups, self.overview = analytics.load_counts(self.engine)

    def test_composite_keys_and_labelled_order_denominator(self):
        self.assertEqual(self.overview["orders"], 11)
        self.assertEqual(self.overview["frauds"], 6)
        self.assertEqual(self.overview["unlabelled_orders"], 1)
        table = analytics.query_payment_methods(self.groups).set_index("payment_method")
        self.assertEqual(table.loc["credit card", "orders"], 6)
        self.assertEqual(table.loc["credit card", "frauds"], 3)
        self.assertEqual(table.loc["credit card", "fraud_rate_pct"], 50)
        self.assertAlmostEqual(table.loc["credit card", "risk_lift"], 0.5/(6/11))
        self.assertAlmostEqual(table["share_fraud_pct"].sum(), 100)

    def test_sgd_boundaries_and_valid_zero_orders(self):
        observed = self.groups.groupby("value_band")["orders"].sum().to_dict()
        self.assertEqual(observed, dict(zip(analytics.VALUE_BANDS, [2, 1, 2, 2, 2, 2])))
        tables = analytics.analyse(self.groups)
        for name in ["q1_payment_methods", "q2_payment_value", "q3_payment_collection",
                     "q1_country_payment", "q2_country_payment_value", "q3_country_payment_collection"]:
            self.assertEqual(tables[name]["orders"].sum(), 11, name)
            self.assertEqual(tables[name]["frauds"].sum(), 6, name)

    def test_q2_lift_uses_the_same_payment_method_baseline(self):
        q2 = analytics.query_order_values(self.groups)
        card = q2[(q2.payment_method=="credit card") & (q2.value_band=="(0, 10]")].iloc[0]
        self.assertEqual(card.fraud_rate_pct, 100)
        self.assertEqual(card.method_baseline_pct, 50)
        self.assertEqual(card.within_method_lift, 2)

    def test_collection_gap_and_missing_group_are_not_zero_risk(self):
        gaps = analytics.analyse(self.groups)["q3_delivery_pickup_gaps"].set_index("payment_method")
        self.assertEqual(gaps.loc["credit card", "delivery_minus_pickup_pp"], -75)
        self.assertEqual(gaps.loc["payment on delivery", "delivery_minus_pickup_pp"], 100)
        self.assertEqual(gaps.loc["no payment", "orders_delivery"], 0)
        self.assertTrue(pd.isna(gaps.loc["no payment", "fraud_rate_pct_delivery"]))
        self.assertTrue(pd.isna(gaps.loc["no payment", "delivery_minus_pickup_pp"]))

    def test_uncertainty_handles_zero_and_all_fraud(self):
        low, high = analytics.wilson_interval([0, 6, 3, 0], [6, 6, 6, 0])
        self.assertTrue(0 <= low[0] < high[0] < 1)
        self.assertTrue(0 < low[1] < high[1] <= 1)
        self.assertLess(low[2], 0.5)
        self.assertGreater(high[2], 0.5)
        self.assertTrue(np.isnan(low[3]) and np.isnan(high[3]))

    def test_each_call_reads_updated_database_state(self):
        with self.engine.begin() as c:
            c.execute(text("UPDATE fraud_labels SET is_fraud=1 WHERE country_code='MY' AND order_id='b'"))
        groups, overview = analytics.load_counts(self.engine)
        self.assertEqual(overview["frauds"], 7)
        self.assertEqual(analytics.query_payment_methods(groups).set_index("payment_method").loc["credit card", "frauds"], 4)

    def test_invalid_values_and_unmatched_labels_stop_analysis(self):
        with self.engine.begin() as c:
            c.execute(text("UPDATE orders SET order_value=-1 WHERE country_code='MY' AND order_id='b'"))
        with self.assertRaisesRegex(ValueError, "invalid"):
            analytics.load_counts(self.engine)
        with self.engine.begin() as c:
            c.execute(text("UPDATE orders SET order_value=10 WHERE country_code='MY' AND order_id='b'"))
            c.execute(text("INSERT INTO fraud_labels VALUES('MY','missing',1)"))
        with self.assertRaisesRegex(ValueError, "unmatched"):
            analytics.load_counts(self.engine)

    def test_insights_change_with_data_and_exports_include_three_figures(self):
        groups = self.groups.copy()
        groups[["orders", "frauds"]] *= 100
        overview = {**self.overview, "orders": 1100, "frauds": 600}
        first = analytics.insights(analytics.analyse(groups), overview)
        groups["frauds"] = np.where(groups.payment_method=="payment on delivery", groups.orders, 0)
        changed = analytics.insights(analytics.analyse(groups), {**overview, "frauds": int(groups.frauds.sum())})
        self.assertNotEqual(first, changed)
        self.assertIn("**payment on delivery**", changed)
        with tempfile.TemporaryDirectory() as folder:
            result = analytics.run_analysis(folder, self.engine)
            self.assertEqual(result["overview"]["orders"], 11)
            for name in ["q1_payment_methods.png", "q2_payment_value.png", "q3_payment_collection.png", "summary.md", "metadata.json"]:
                self.assertGreater((Path(result["output_dir"])/name).stat().st_size, 0)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT 1")).scalar(), 1)


if __name__ == "__main__":
    unittest.main()
