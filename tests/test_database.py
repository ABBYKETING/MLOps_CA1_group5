"""Tests for database.load_training_data() (PBI 5, Task #50).

They use a small in-memory SQLite database, so no RDS connection is needed:

    python -m unittest tests.test_database -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_processing as dp  # noqa: E402
import database as db  # noqa: E402

CUSTOMERS = pd.DataFrame({
    "country_code": ["MY", "PK", "TH"], "customer_id": ["c1", "c2", "c3"],
    "mobile_verified": ["True", "False", "True"], "num_orders_last_50days": [3, 1, 2],
    "num_cancelled_orders_last_50days": [0, 1, 0], "num_refund_orders_last_50days": [0, 0, 1],
    "total_payment_last_50days": [25.5, 9.0, 12.0], "num_associated_customers": [1, 2, 150],
    "first_order_datetime": ["2022-01-05 10:00:00", None, "2023-04-01 08:00:00"],
})
ORDERS = pd.DataFrame({
    "country_code": ["MY", "PK", "TH", "TH"], "order_id": ["o1", "o2", "o3", "o4"],
    "collect_type": ["Delivery", "Pickup", "Delivery", "Delivery"],
    "payment_method": ["credit card", "balance", "paypal", "paypal"],
    "order_value": [12.5, 7.0, 30.0, 5.0], "num_items_ordered": [2, 1, 3, 1],
    "refund_value": [0.0, 0.0, 0.0, 0.0],
    "order_date": ["2023/05/03", "2023/05/01", "2023/05/02", "2023/05/04"],
})
LABELS = pd.DataFrame({            # o4 has no label, so it is not a training row
    "country_code": ["MY", "PK", "TH"], "order_id": ["o1", "o2", "o3"],
    "customer_id": ["c1", "c2", "c3"], "is_fraud": [0, 1, 0],
})


class LoadTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.customers, _ = dp.clean_customers(CUSTOMERS)
        self.orders, _ = dp.clean_orders(ORDERS)
        self.labels, _ = dp.clean_labels(LABELS)
        self.engine = create_engine("sqlite://")
        self.customers.to_sql("customers", self.engine, index=False)
        self.orders.to_sql("orders", self.engine, index=False)
        self.labels.to_sql("fraud_labels", self.engine, index=False)

    def test_matches_merge_datasets(self):
        loaded = db.load_training_data(self.engine)
        expected = dp.merge_datasets(self.customers, self.orders, self.labels)
        self.assertEqual(list(loaded.columns), list(expected.columns))
        keys = ["country_code", "order_id"]
        pd.testing.assert_frame_equal(
            loaded.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True),
            check_dtype=False)

    def test_only_labelled_orders_sorted_by_date(self):
        loaded = db.load_training_data(self.engine)
        self.assertEqual(list(loaded["order_id"]), ["o2", "o3", "o1"])     # o4 has no label
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["order_date"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["first_order_datetime"]))
        self.assertEqual(loaded["mobile_verified"].dtype, bool)

    def test_ready_for_feature_engineering(self):
        X_full, X_safe, y = dp.build_feature_sets(db.load_training_data(self.engine))
        self.assertEqual(len(X_safe), 3)
        self.assertEqual(list(y), [1, 0, 0])
        self.assertNotIn("refund_value", X_full.columns)


if __name__ == "__main__":
    unittest.main()
