"""AWS RDS MySQL connection and table setup (PBI 4, Task #42).

Creates the SP-buy database tables defined in sql/schema.sql and provides
``get_engine()``, which the ETL pipeline, analytics and Prefect flows can reuse
to connect to the same database. ``load_training_data()`` reads the latest
labelled rows for model retraining (PBI 5, Task #50).

Connection settings come from environment variables, never from the code:

    DB_HOST      RDS endpoint, e.g. spbuy-db.xxxx.us-east-1.rds.amazonaws.com
    DB_PORT      optional, default 3306
    DB_USER      database username
    DB_PASSWORD  database password
    DB_NAME      optional, default "spbuy"

Usage (from the repository root):

    python src/database.py            # create the database and tables, then check them
    python src/database.py --check    # only check the connection and list the tables
    python src/database.py --reset    # rebuild the tables after a schema change (only if all are empty)
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_FILE = REPO_ROOT / "sql" / "schema.sql"
DEFAULT_DB_NAME = "spbuy"
EXPECTED_TABLES = ["customers", "orders", "fraud_labels", "rejected_records", "etl_batches"]


def get_db_config():
    """Read the connection settings from environment variables."""
    missing = [name for name in ("DB_HOST", "DB_USER", "DB_PASSWORD") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}.\n"
            "Set them before running, e.g. in PowerShell:\n"
            '  $env:DB_HOST = "<your RDS endpoint>"\n'
            '  $env:DB_USER = "<username>"\n'
            '  $env:DB_PASSWORD = "<password>"'
        )
    name = os.environ.get("DB_NAME", DEFAULT_DB_NAME)
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"DB_NAME may only contain letters, digits and underscores, got {name!r}")
    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ.get("DB_PORT", 3306)),
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "database": name,
    }


def get_engine(with_database=True, config=None):
    """SQLAlchemy engine for the RDS database (or the server only, to create the database)."""
    config = config or get_db_config()
    url = URL.create(
        "mysql+pymysql",
        username=config["user"],
        password=config["password"],      # URL.create escapes special characters safely
        host=config["host"],
        port=config["port"],
        database=config["database"] if with_database else None,
        query={"charset": "utf8mb4"},
    )
    # pool_pre_ping reconnects automatically if RDS closed an idle connection
    return create_engine(url, pool_pre_ping=True)


def create_database(config=None):
    """Create the database on the RDS server if it does not exist yet."""
    config = config or get_db_config()
    engine = get_engine(with_database=False, config=config)
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{config['database']}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
        ))
    engine.dispose()


def read_schema_statements(schema_file=SCHEMA_FILE):
    """Split sql/schema.sql into separate CREATE TABLE statements (comments removed)."""
    sql = Path(schema_file).read_text(encoding="utf-8")
    sql = re.sub(r"--[^\n]*", "", sql)    # the schema file has no "--" inside strings
    return [s.strip() for s in sql.split(";") if s.strip()]


def create_tables(engine, schema_file=SCHEMA_FILE):
    """Run every CREATE TABLE IF NOT EXISTS statement in the schema file."""
    statements = read_schema_statements(schema_file)
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
    return len(statements)


def reset_empty_tables(engine):
    """Drop the project tables so they can be recreated - only allowed while they hold no data."""
    existing = set(inspect(engine).get_table_names())
    with engine.connect() as conn:
        not_empty = [t for t in EXPECTED_TABLES if t in existing
                     and conn.execute(text(f"SELECT COUNT(*) FROM `{t}`")).scalar() > 0]
    if not_empty:
        raise RuntimeError(f"Refusing to reset: these tables contain data: {not_empty}")
    with engine.begin() as conn:
        # fraud_labels first, because it references orders and customers
        for table in ["fraud_labels", "orders", "customers", "rejected_records", "etl_batches"]:
            conn.execute(text(f"DROP TABLE IF EXISTS `{table}`"))


def check_tables(engine):
    """Return {table: row count} for the expected tables; raise if any is missing."""
    existing = set(inspect(engine).get_table_names())
    missing = [t for t in EXPECTED_TABLES if t not in existing]
    if missing:
        raise RuntimeError(f"Tables missing from the database: {missing}")
    with engine.connect() as conn:
        return {t: conn.execute(text(f"SELECT COUNT(*) FROM `{t}`")).scalar() for t in EXPECTED_TABLES}


# Same columns, in the same order, as data_processing.merge_datasets() (labels, then
# order details, then customer details), so the result can go straight into
# data_processing.build_feature_sets().
TRAINING_QUERY = """
    SELECT l.country_code, l.order_id, l.customer_id, l.is_fraud,
           o.collect_type, o.payment_method, o.order_value, o.num_items_ordered,
           o.refund_value, o.order_date,
           c.mobile_verified, c.num_orders_last_50days, c.num_cancelled_orders_last_50days,
           c.num_refund_orders_last_50days, c.total_payment_last_50days,
           c.num_associated_customers, c.first_order_datetime
    FROM fraud_labels l
    JOIN orders o    ON o.country_code = l.country_code AND o.order_id = l.order_id
    JOIN customers c ON c.country_code = l.country_code AND c.customer_id = l.customer_id
    ORDER BY o.order_date, l.country_code, l.order_id
"""


def load_training_data(engine=None):
    """Return every labelled order with its order and customer details (PBI 5, Task #50).

    The database already holds the base data plus every loaded batch, with one
    current record per customer, order and label ("latest wins"), so this is the
    combined historical + new data. Rows come back sorted by order_date.

    The result matches data_processing.merge_datasets(), so it can be used as is:

        merged = db.load_training_data()
        X_full, X_safe, y = dp.build_feature_sets(merged)

    refund_value is included only because merge_datasets() includes it; it is
    never a model feature (build_features() does not use it).
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(TRAINING_QUERY), conn,
                         parse_dates=["order_date", "first_order_datetime"])
    # MySQL returns BOOLEAN as 0/1 and TINYINT as small ints; match the cleaned data types
    df["mobile_verified"] = df["mobile_verified"].astype(bool)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Create and check the SP-buy RDS tables (Task #42).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Only check the connection and tables")
    group.add_argument("--reset", action="store_true",
                       help="Drop and recreate the tables (refused if any table has data)")
    args = parser.parse_args()

    config = get_db_config()
    print(f"Connecting to {config['host']}:{config['port']} as {config['user']} (database: {config['database']})")

    if not args.check:
        create_database(config)
        engine = get_engine(config=config)
        if args.reset:
            reset_empty_tables(engine)
            print("Dropped the empty tables so they can be recreated.")
        count = create_tables(engine)
        print(f"Ran {count} CREATE TABLE IF NOT EXISTS statements from {SCHEMA_FILE.name}")
    else:
        engine = get_engine(config=config)

    with engine.connect() as conn:
        version = conn.execute(text("SELECT VERSION()")).scalar()
    print(f"MySQL version: {version}")
    for table, rows in check_tables(engine).items():
        print(f"  {table:<18} {rows:>10,} rows")
    print("All tables present.")
    engine.dispose()


if __name__ == "__main__":
    main()
