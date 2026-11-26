"""Data integrity check for the SP-buy RDS database (PBI 4, Task #47).

Run after an ETL load to prove the database is consistent:

    python src/check_data_integrity.py

Prints an overview (row counts, fraud rate, date range, ETL runs, rejected rows)
and a list of checks that must all PASS. Exits with code 1 if any check fails,
so it can also be used as a step in an automated pipeline.
Connection settings come from DB_HOST, DB_USER, DB_PASSWORD (and DB_NAME).
"""

import sys
from datetime import date

import pandas as pd
from sqlalchemy import text

import database as db

# Each check is a query that counts PROBLEM rows: 0 means PASS.
CHECKS = {
    "Every ETL batch finished successfully":
        "SELECT COUNT(*) FROM etl_batches WHERE status <> 'completed'",
    "Every fraud label has its order":
        """SELECT COUNT(*) FROM fraud_labels l LEFT JOIN orders o
           ON o.country_code = l.country_code AND o.order_id = l.order_id WHERE o.order_id IS NULL""",
    "Every fraud label has its customer":
        """SELECT COUNT(*) FROM fraud_labels l LEFT JOIN customers c
           ON c.country_code = l.country_code AND c.customer_id = l.customer_id WHERE c.customer_id IS NULL""",
    "No duplicate orders (country_code + order_id)":
        "SELECT COUNT(*) - COUNT(DISTINCT country_code, order_id) FROM orders",
    "No duplicate customers (country_code + customer_id)":
        "SELECT COUNT(*) - COUNT(DISTINCT country_code, customer_id) FROM customers",
    "is_fraud is only 0 or 1":
        "SELECT COUNT(*) FROM fraud_labels WHERE is_fraud NOT IN (0, 1)",
    "collect_type is only Delivery or Pickup":
        "SELECT COUNT(*) FROM orders WHERE collect_type NOT IN ('Delivery', 'Pickup')",
    "No negative order, refund or payment values":
        """SELECT (SELECT COUNT(*) FROM orders WHERE order_value < 0 OR refund_value < 0)
                + (SELECT COUNT(*) FROM customers WHERE total_payment_last_50days < 0)""",
    "Every order has at least 1 item":
        "SELECT COUNT(*) FROM orders WHERE num_items_ordered < 1",
    "No order dates in the future":
        "SELECT COUNT(*) FROM orders WHERE order_date > :today",
    "Country codes are 2 upper-case letters":
        # 'c' = case-sensitive match, so lower-case codes like 'my' are caught
        """SELECT (SELECT COUNT(*) FROM customers WHERE NOT REGEXP_LIKE(country_code, '^[A-Z]{2}$', 'c'))
                + (SELECT COUNT(*) FROM orders WHERE NOT REGEXP_LIKE(country_code, '^[A-Z]{2}$', 'c'))""",
}


def overview(engine):
    """Counts and summaries (information, not pass/fail)."""
    with engine.connect() as conn:
        one = lambda sql: conn.execute(text(sql)).scalar()          # noqa: E731
        print("Rows per table")
        for table in ("customers", "orders", "fraud_labels", "rejected_records", "etl_batches"):
            print(f"  {table:<18} {one(f'SELECT COUNT(*) FROM {table}'):>10,}")
        print(f"  orders without a label    {one('''SELECT COUNT(*) FROM orders o LEFT JOIN fraud_labels l
              ON l.country_code = o.country_code AND l.order_id = o.order_id WHERE l.order_id IS NULL'''):>10,}")
        rate = one("SELECT AVG(is_fraud) FROM fraud_labels")
        print(f"  fraud rate (labelled orders) {0 if rate is None else float(rate) * 100:.2f}%")
        print(f"  order dates                {one('SELECT MIN(order_date) FROM orders')} to "
              f"{one('SELECT MAX(order_date) FROM orders')}")
    print("\nETL runs")
    print(pd.read_sql("""SELECT batch_name, status, started_at, finished_at, customers_inserted AS cust_new,
                         customers_updated AS cust_upd, orders_inserted AS ord_new, orders_updated AS ord_upd,
                         labels_inserted AS lab_new, labels_updated AS lab_upd, rows_rejected AS rejected
                         FROM etl_batches ORDER BY started_at""", engine).to_string(index=False))
    rejected = pd.read_sql("""SELECT batch_name, source_table, reason, COUNT(*) AS rows_rejected
                              FROM rejected_records GROUP BY batch_name, source_table, reason
                              ORDER BY batch_name, rows_rejected DESC""", engine)
    print("\nRejected rows by reason")
    print("  none" if rejected.empty else rejected.to_string(index=False))


def run_checks(engine):
    """Run every check; returns True only if all pass."""
    print("\nIntegrity checks")
    all_passed = True
    with engine.connect() as conn:
        for name, sql in CHECKS.items():
            try:
                problems = conn.execute(text(sql), {"today": date.today()}).scalar() or 0
            except Exception as err:                       # report it, keep checking the rest
                all_passed = False
                print(f"  ERROR {name}  ({getattr(err, 'orig', err)})")
                conn.rollback()
                continue
            all_passed &= problems == 0
            print(f"  {'PASS' if problems == 0 else 'FAIL'}  {name}"
                  + ("" if problems == 0 else f"  ({problems:,} problem rows)"))
    print("\nAll checks passed." if all_passed else "\nSOME CHECKS FAILED.")
    return all_passed


def main():
    config = db.get_db_config()
    print(f"Database: {config['database']} on {config['host']}\n")
    engine = db.get_engine(config=config)
    overview(engine)
    passed = run_checks(engine)
    engine.dispose()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
