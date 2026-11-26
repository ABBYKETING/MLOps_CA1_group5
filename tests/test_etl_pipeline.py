"""Tests for the ETL pipeline (PBI 4, Task #47).

Run from the repository root:

    python -m unittest tests.test_etl_pipeline -v

The unit tests need no database. The database tests run only when you opt in by
setting ETL_TEST_DATABASE to a SEPARATE database name (never the team's "spbuy"),
together with DB_HOST, DB_USER and DB_PASSWORD, e.g. in PowerShell:

    $env:ETL_TEST_DATABASE = "spbuy_practice"
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import etl_pipeline as etl  # noqa: E402

CUSTOMERS_HEADER = ("country_code,customer_id,mobile_verified,num_orders_last_50days,"
                    "num_cancelled_orders_last_50days,num_refund_orders_last_50days,"
                    "total_payment_last_50days,num_associated_customers,first_order_datetime")
ORDERS_HEADER = ("country_code,order_id,collect_type,payment_method,order_value,"
                 "num_items_ordered,refund_value,order_date")
LABELS_HEADER = "country_code,order_id,customer_id,is_fraud"


def write_batch(folder, postfix, customers, orders, labels):
    """Write a small batch of CSV files (rows are CSV lines without the header)."""
    folder = Path(folder)
    for kind, header, rows in (("customers", CUSTOMERS_HEADER, customers),
                               ("orders", ORDERS_HEADER, orders),
                               ("labels", LABELS_HEADER, labels)):
        (folder / f"{kind}_{postfix}.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return {kind: folder / f"{kind}_{postfix}.csv" for kind in etl.FILE_KINDS}


GOOD_CUSTOMERS = ["MY,c1,True,3,0,0,25.5,1,2022-01-05 10:00:00",
                  "pk,c2,yes,1,0,0,9.0,2,2022/03/01 08:30:00"]           # lowercase country, 'yes'
GOOD_ORDERS = ["MY,o1,Delivery,credit card,12.5,2,0.0,2023/05/01",
               "PK,o2,pickup,Payment on Delivery,7.0,1,0.0,2023-05-02"]   # lowercase collect type
GOOD_LABELS = ["MY,o1,c1,0", "PK,o2,c2,1"]


class BatchNameTests(unittest.TestCase):
    def test_batch_name_comes_from_the_file_postfix(self):
        self.assertEqual(etl.batch_name_from_source("data/orders_XXX.csv"), "XXX")
        self.assertEqual(etl.batch_name_from_source("https://example.com/files/labels_week2.csv"), "week2")
        self.assertEqual(etl.batch_name_from_source("https://example.com/customers_b1.csv?x=1"), "b1")

    def test_unexpected_file_names_and_unsafe_batch_names_are_refused(self):
        with self.assertRaises(ValueError):
            etl.batch_name_from_source("data/random.csv")
        with self.assertRaises(ValueError):
            etl.check_batch_name("bad name; DROP TABLE")


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def extract_and_transform(self, customers, orders, labels):
        sources = write_batch(self.folder.name, "t", customers, orders, labels)
        raw = etl.extract(sources, "t", raw_dir=Path(self.folder.name) / "raw")
        return raw, *etl.transform(raw)

    def test_good_rows_are_cleaned_with_the_team_rules(self):
        _, cleaned, rejected = self.extract_and_transform(GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS)
        self.assertEqual(rejected, [])
        self.assertEqual(set(cleaned["customers"]["country_code"]), {"MY", "PK"})
        self.assertEqual(set(cleaned["orders"]["collect_type"]), {"Delivery", "Pickup"})
        self.assertTrue(cleaned["customers"]["mobile_verified"].all())

    def test_bad_rows_are_rejected_with_reasons_and_original_values(self):
        orders = GOOD_ORDERS + ["MY,o3,Delivery,credit card,-5,1,0.0,2023/05/03",      # negative value
                                "MY,o4,drone,credit card,5,1,0.0,2023/05/03"]          # unknown collect type
        customers = GOOD_CUSTOMERS + ["MY,c3,maybe,1,0,0,1.0,1,2022-01-01 00:00:00"]  # bad yes/no
        _, cleaned, rejected = self.extract_and_transform(customers, orders, GOOD_LABELS)
        reasons = {r["record_id"]: r["reason"] for r in rejected}
        self.assertEqual(reasons, {"o3": "negative value", "o4": "unrecognised collect_type",
                                   "c3": "unrecognised mobile_verified"})
        raw_o4 = json.loads(next(r for r in rejected if r["record_id"] == "o4")["raw_record"])
        self.assertEqual(raw_o4["collect_type"], "drone")          # stored exactly as received
        self.assertEqual(len(cleaned["orders"]), 2)

    def test_values_too_long_for_the_database_are_rejected_not_crashing(self):
        orders = GOOD_ORDERS + [f"MY,{'x' * 40},Delivery,credit card,5,1,0.0,2023/05/03"]
        _, cleaned, rejected = self.extract_and_transform(GOOD_CUSTOMERS, orders, GOOD_LABELS)
        self.assertEqual(len(rejected), 1)
        self.assertIn("does not fit the database column", rejected[0]["reason"])
        self.assertEqual(len(cleaned["orders"]), 2)

    def test_rows_with_wrong_number_of_fields_are_rejected_not_stopping(self):
        orders = GOOD_ORDERS + ['MY,o3,Delivery,"credit, card",5,1,0.0,2023/05/03',      # quoted comma is fine
                                "MY,o4,Delivery,credit card,5,1,0.0,2023/05/03,EXTRA"]   # one field too many
        _, cleaned, rejected = self.extract_and_transform(GOOD_CUSTOMERS, orders, GOOD_LABELS)
        self.assertEqual([(r["record_id"], r["reason"]) for r in rejected],
                         [("o4", "row has 9 fields, expected 8")])
        self.assertEqual(json.loads(rejected[0]["raw_record"])["fields"][-1], "EXTRA")
        self.assertEqual(sorted(cleaned["orders"]["order_id"]), ["o1", "o2", "o3"])

    def test_future_dates_are_rejected(self):
        orders = GOOD_ORDERS + ["MY,o3,Delivery,credit card,5,1,0.0,2100/01/01"]
        customers = GOOD_CUSTOMERS + ["MY,c3,True,1,0,0,1.0,1,2100-01-01 00:00:00"]
        _, cleaned, rejected = self.extract_and_transform(customers, orders, GOOD_LABELS)
        self.assertEqual({r["record_id"]: r["reason"] for r in rejected},
                         {"o3": "date is in the future", "c3": "date is in the future"})
        self.assertEqual(len(cleaned["orders"]), 2)

    def test_counts_too_large_for_the_database_are_rejected(self):
        customers = GOOD_CUSTOMERS + ["MY,c3,True,99999999999,0,0,1.0,1,2022-01-01 00:00:00"]
        _, cleaned, rejected = self.extract_and_transform(customers, GOOD_ORDERS, GOOD_LABELS)
        self.assertEqual(len(rejected), 1)
        self.assertIn("count too large", rejected[0]["reason"])
        self.assertEqual(len(cleaned["customers"]), 2)

    def test_header_names_are_trimmed_and_lowercased(self):
        sources = write_batch(self.folder.name, "t", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS)
        Path(sources["labels"]).write_text(" Country_Code ,ORDER_ID,Customer_ID,is_fraud\n"
                                           + "\n".join(GOOD_LABELS) + "\n", encoding="utf-8-sig")
        raw = etl.extract(sources, "t", raw_dir=Path(self.folder.name) / "raw")
        cleaned, rejected = etl.transform(raw)
        self.assertEqual(rejected, [])
        self.assertEqual(len(cleaned["fraud_labels"]), 2)

    def test_empty_or_non_text_files_stop_with_a_clear_message(self):
        sources = write_batch(self.folder.name, "t", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS)
        Path(sources["labels"]).write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "is empty"):
            etl.extract(sources, "t", raw_dir=Path(self.folder.name) / "raw")
        Path(sources["labels"]).write_bytes("country_code".encode("utf-16"))
        with self.assertRaisesRegex(ValueError, "not a UTF-8"):
            etl.extract(sources, "t", raw_dir=Path(self.folder.name) / "raw")

    def test_missing_columns_stop_the_pipeline(self):
        sources = write_batch(self.folder.name, "t", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS)
        Path(sources["labels"]).write_text("country_code,order_id\nMY,o1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            etl.extract(sources, "t", raw_dir=Path(self.folder.name) / "raw")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):          # keep test output clean
        pass


class UrlExtractTests(unittest.TestCase):
    """The demo gives URLs, so test downloading from a small local web server."""

    def test_files_are_downloaded_from_urls(self):
        with tempfile.TemporaryDirectory() as served, tempfile.TemporaryDirectory() as raw_dir:
            write_batch(served, "web", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS)
            handler = partial(QuietHandler, directory=served)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                sources = {k: f"{base}/{k}_web.csv" for k in etl.FILE_KINDS}
                self.assertEqual(etl.batch_name_from_source(sources["orders"]), "web")
                raw = etl.extract(sources, "web", raw_dir=raw_dir)
            finally:
                server.shutdown()
                server.server_close()
            self.assertEqual({k: len(v) for k, v in raw.items()}, {"customers": 2, "orders": 2, "labels": 2})
            self.assertTrue((Path(raw_dir) / "web" / "orders_web.csv").is_file())  # raw copy kept

    def test_unreachable_url_fails_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            with self.assertRaises(RuntimeError) as ctx:
                etl.download("http://127.0.0.1:9/orders_x.csv", Path(raw_dir) / "orders_x.csv",
                             retries=1, timeout=2)
            self.assertIn("Could not download", str(ctx.exception))


@unittest.skipUnless(os.environ.get("ETL_TEST_DATABASE"),
                     "set ETL_TEST_DATABASE (a separate practice database) to run database tests")
class DatabaseTests(unittest.TestCase):
    """End-to-end load into a separate practice database (never the team's 'spbuy')."""

    @classmethod
    def setUpClass(cls):
        import database as db
        name = os.environ["ETL_TEST_DATABASE"]
        if name == db.DEFAULT_DB_NAME:
            raise unittest.SkipTest("refusing to run tests against the team database 'spbuy'")
        os.environ["DB_NAME"] = name
        db.create_database()
        cls.engine = db.get_engine()
        from sqlalchemy import text
        with cls.engine.begin() as conn:                     # start from empty tables
            for table in ("fraud_labels", "orders", "customers", "rejected_records", "etl_batches"):
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        db.create_tables(cls.engine)

    def run_batch(self, postfix, customers, orders, labels):
        with tempfile.TemporaryDirectory() as folder:
            sources = write_batch(folder, postfix, customers, orders, labels)
            return etl.run_etl(sources, raw_dir=Path(folder) / "raw", engine=self.engine,
                               summary_dir=Path(folder) / "summary")

    def query(self, sql):
        from sqlalchemy import text
        with self.engine.connect() as conn:
            return conn.execute(text(sql)).fetchall()

    def test_load_rerun_latest_wins_and_orphans(self):
        first = self.run_batch("b1", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS + ["MY,o9,c1,0"])
        self.assertEqual((first["orders_inserted"], first["labels_inserted"], first["rows_rejected"]), (2, 2, 1))

        again = self.run_batch("b1", GOOD_CUSTOMERS, GOOD_ORDERS, GOOD_LABELS + ["MY,o9,c1,0"])
        self.assertEqual((again["orders_inserted"], again["orders_updated"]), (0, 2))
        self.assertEqual(self.query("SELECT COUNT(*) FROM rejected_records WHERE batch_name='b1'")[0][0], 1)

        # Later batch: changed fraud label and order value replace the old ones
        later = self.run_batch("b2", [], ["MY,o1,Delivery,credit card,99.9,2,0.0,2023/05/01"], ["MY,o1,c1,1"])
        self.assertEqual((later["orders_updated"], later["labels_updated"]), (1, 1))
        row = self.query("SELECT o.order_value, l.is_fraud, l.first_loaded_batch, l.last_updated_batch "
                         "FROM orders o JOIN fraud_labels l USING (country_code, order_id) WHERE order_id='o1'")[0]
        self.assertEqual(tuple(row), (99.9, 1, "b1", "b2"))
        statuses = {r[0]: r[1] for r in self.query("SELECT batch_name, status FROM etl_batches")}
        self.assertEqual(statuses, {"b1": "completed", "b2": "completed"})


if __name__ == "__main__":
    unittest.main()
