"""SP-buy automated ETL pipeline (PBI 4, Tasks #43-#46).

Extract   (#43) download customers_XXX.csv, orders_XXX.csv and labels_XXX.csv from
                their URLs (or read local files) and keep a raw copy in data/raw/<batch>/.
Transform (#44) clean them with the team's shared rules in data_processing.py
                and stop if the cleaned data fails the team's validation checks.
                Rows that still cannot be stored (wrong number of fields, values too
                long or too large for the columns, dates in the future) are rejected
                with a reason instead of stopping the batch.
Load (#45-#46)  write everything to the RDS tables in sql/schema.sql:
                - customers: new ones inserted, existing ones updated ("latest wins")
                - orders and fraud labels: new ones inserted, repeated ones replace
                  the older record ("latest wins", PO answers Q4 and Q5)
                - labels whose order or customer does not exist are not loaded
                - every row that could not be used goes to rejected_records with a reason
                - the run is recorded in etl_batches

The whole load runs in ONE database transaction: if anything fails, nothing is
half-loaded and the batch is marked "failed" in etl_batches.

Usage (from the repository root, with DB_HOST, DB_USER and DB_PASSWORD set):

    # New batch from the tutor's URLs (the batch name is taken from the file names)
    python src/etl_pipeline.py --customers <url> --orders <url> --labels <url>

    # Local files, e.g. the historical base data
    python src/etl_pipeline.py --data-dir path/to/folder --postfix base

    # Check extraction and cleaning only, without touching the database
    python src/etl_pipeline.py --data-dir path/to/folder --postfix test --dry-run
"""

import argparse
import json
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import text

import data_processing as dp

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"          # git-ignored
DEFAULT_SUMMARY_DIR = REPO_ROOT / "outputs" / "etl"    # git-ignored
FILE_KINDS = ("customers", "orders", "labels")
INSERT_CHUNK = 10000                                   # rows per INSERT statement batch

# Columns stored in each table (same names as the CSV files and data_processing.py)
TABLE_COLUMNS = {
    "customers": ["country_code", "customer_id", "mobile_verified", "num_orders_last_50days",
                  "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
                  "total_payment_last_50days", "num_associated_customers", "first_order_datetime"],
    "orders": ["country_code", "order_id", "collect_type", "payment_method", "order_value",
               "num_items_ordered", "refund_value", "order_date"],
    "fraud_labels": ["country_code", "order_id", "customer_id", "is_fraud"],
}
TABLE_KEYS = {
    "customers": ["country_code", "customer_id"],
    "orders": ["country_code", "order_id"],
    "fraud_labels": ["country_code", "order_id"],
}
# Temporary staging tables (one per run, deleted automatically when the connection closes).
# _row is the row's position in the original CSV file, used to store rejected rows exactly as received.
STAGING_DDL = {
    "customers": """CREATE TEMPORARY TABLE stg_customers (
        _row INT NOT NULL, country_code CHAR(2) NOT NULL, customer_id VARCHAR(32) NOT NULL,
        mobile_verified BOOLEAN NOT NULL, num_orders_last_50days INT UNSIGNED NOT NULL,
        num_cancelled_orders_last_50days INT UNSIGNED NOT NULL,
        num_refund_orders_last_50days INT UNSIGNED NOT NULL, total_payment_last_50days DOUBLE NOT NULL,
        num_associated_customers INT UNSIGNED NOT NULL, first_order_datetime DATETIME NULL,
        PRIMARY KEY (country_code, customer_id))""",
    "orders": """CREATE TEMPORARY TABLE stg_orders (
        _row INT NOT NULL, country_code CHAR(2) NOT NULL, order_id VARCHAR(32) NOT NULL,
        collect_type VARCHAR(10) NOT NULL, payment_method VARCHAR(40) NOT NULL,
        order_value DOUBLE NOT NULL, num_items_ordered INT UNSIGNED NOT NULL,
        refund_value DOUBLE NOT NULL, order_date DATE NOT NULL,
        PRIMARY KEY (country_code, order_id))""",
    "fraud_labels": """CREATE TEMPORARY TABLE stg_fraud_labels (
        _row INT NOT NULL, country_code CHAR(2) NOT NULL, order_id VARCHAR(32) NOT NULL,
        customer_id VARCHAR(32) NOT NULL, is_fraud TINYINT NOT NULL,
        PRIMARY KEY (country_code, order_id))""",
}


# ---------------------------------------------------------------------------
# Extract (#43)
# ---------------------------------------------------------------------------
def is_url(source):
    return str(source).lower().startswith(("http://", "https://"))


def batch_name_from_source(source):
    """'.../orders_XXX.csv' -> 'XXX' (the postfix that identifies the batch)."""
    name = Path(urlparse(str(source)).path if is_url(source) else str(source)).name
    match = re.fullmatch(r"(?:customers|orders|labels)_(.+)\.csv", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot read a batch name from {name!r}; expected e.g. orders_XXX.csv "
                         "(or pass --batch-name)")
    return match.group(1)


def check_batch_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
        raise ValueError(f"Batch name {name!r} may only use letters, digits, '_', '-', '.' (max 64)")
    return name


def download(url, destination, retries=3, timeout=60):
    """Download one file, retrying a few times (network hiccups during the demo)."""
    temporary = destination.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, open(temporary, "wb") as out:
                shutil.copyfileobj(response, out)
            if temporary.stat().st_size == 0:
                raise ValueError("downloaded file is empty")
            temporary.replace(destination)
            return
        except Exception as err:
            if attempt == retries:
                raise RuntimeError(f"Could not download {url} after {retries} attempts: {err}") from err
            print(f"  download attempt {attempt} failed ({err}); retrying...")
            time.sleep(2 * attempt)


TEXT_COLUMNS = {"customers": dp.CUSTOMER_TEXT_COLS, "orders": dp.ORDER_TEXT_COLS,
                "labels": dp.LABEL_TEXT_COLS}


def read_batch_file(path, kind):
    """Read one CSV file the same way as dp.load_raw, but more forgiving:
    - header names are trimmed and lower-cased ("Order_ID " -> "order_id"), BOM removed
    - a row with too many fields is set aside (rejected later) instead of stopping the batch
    - unreadable files stop the batch with a clear message
    The bad rows are kept in frame.attrs["bad_lines"]."""
    name = Path(path).name
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    except pd.errors.EmptyDataError:
        raise ValueError(f"{name} is empty (no header row)") from None
    except UnicodeDecodeError:
        raise ValueError(f"{name} is not a UTF-8 text CSV file") from None
    clean_names = {col: str(col).strip().lower() for col in header}
    text_dtype = {col: str for col, new in clean_names.items() if new in TEXT_COLUMNS[kind]}
    bad_lines = []

    def keep_bad_line(fields):
        bad_lines.append(fields)
        return None                                    # drop it from the table

    options = dict(dtype=text_dtype, encoding="utf-8-sig")
    try:
        frame = pd.read_csv(path, **options)            # fast C parser for normal files
    except pd.errors.ParserError:
        frame = pd.read_csv(path, engine="python", on_bad_lines=keep_bad_line, **options)
    except UnicodeDecodeError:
        raise ValueError(f"{name} is not a UTF-8 text CSV file") from None
    frame = frame.rename(columns=clean_names)
    frame.attrs["bad_lines"] = bad_lines
    return frame


def extract(sources, batch_name, raw_dir=DEFAULT_RAW_DIR):
    """Fetch the 3 files into raw_dir/<batch>/ and read them (see read_batch_file)."""
    folder = Path(raw_dir) / batch_name
    folder.mkdir(parents=True, exist_ok=True)
    for kind in FILE_KINDS:
        source = sources[kind]
        destination = folder / f"{kind}_{batch_name}.csv"
        if is_url(source):
            download(source, destination)
        else:
            source = Path(source)
            if not source.is_file():
                raise FileNotFoundError(f"{kind} file not found: {source}")
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
    customers, orders, labels = (read_batch_file(folder / f"{kind}_{batch_name}.csv", kind)
                                 for kind in FILE_KINDS)
    expected = {"customers": TABLE_COLUMNS["customers"],
                "orders": TABLE_COLUMNS["orders"],
                "labels": TABLE_COLUMNS["fraud_labels"]}
    for kind, frame in zip(FILE_KINDS, (customers, orders, labels)):
        missing = [c for c in expected[kind] if c not in frame.columns]
        if missing:
            raise ValueError(f"{kind}_{batch_name}.csv is missing columns {missing}; "
                             "new files must have the same schema as the base data")
    return {"customers": customers, "orders": orders, "labels": labels}


# ---------------------------------------------------------------------------
# Transform (#44)
# ---------------------------------------------------------------------------
def json_record(row):
    """One raw CSV row as JSON text, with missing values as null."""
    values = {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    return json.dumps(values, default=str)


def rejected_from_cleaning(rejected, raw, source_table, id_column):
    """Turn data_processing's rejected rows into rejected_records rows (original values kept)."""
    if rejected.empty:
        return []
    raw_rows = raw.loc[rejected["_row"].to_numpy()]
    return [{"source_table": source_table,
             "country_code": None if pd.isna(r["country_code"]) else str(r["country_code"])[:16],
             "record_id": None if pd.isna(r[id_column]) else str(r[id_column])[:64],
             "reason": str(reason)[:255],
             "raw_record": json_record(r)}
            for (_, r), reason in zip(raw_rows.iterrows(), rejected["reason"])]


def rejected_bad_lines(bad_lines, columns, source_table, id_column):
    """Rows with too many fields: rejected with every field kept as received."""
    rejected = []
    for fields in bad_lines:
        named = dict(zip(columns, fields))
        rejected.append({"source_table": source_table,
                         "country_code": (named.get("country_code") or None) and str(named["country_code"])[:16],
                         "record_id": (named.get(id_column) or None) and str(named[id_column])[:64],
                         "reason": f"row has {len(fields)} fields, expected {len(columns)}",
                         "raw_record": json.dumps({"fields": fields})})
    return rejected


# Longest value each text column can hold in the RDS tables (sql/schema.sql)
COLUMN_LIMITS = {"country_code": 2, "customer_id": 32, "order_id": 32, "payment_method": 40}
# Count columns are INT UNSIGNED in the RDS tables
INT_UNSIGNED_MAX = 4_294_967_295
INT_COLUMNS = ["num_orders_last_50days", "num_cancelled_orders_last_50days",
               "num_refund_orders_last_50days", "num_associated_customers", "num_items_ordered"]
DATE_COLUMNS = ["order_date", "first_order_datetime"]


def reject_unfit_for_schema(cleaned_frame):
    """Rows whose text values cannot fit the database columns are rejected (with a reason)
    instead of making the whole batch fail. Returns (kept rows, rejected rows)."""
    too_long = pd.Series(False, index=cleaned_frame.index)
    for column, limit in COLUMN_LIMITS.items():
        if column in cleaned_frame:
            values = cleaned_frame[column].astype(str)
            bad = values.str.len() > limit
            if column == "country_code":
                bad |= values.str.len() != limit
            too_long |= bad
    too_big = pd.Series(False, index=cleaned_frame.index)
    for column in INT_COLUMNS:
        if column in cleaned_frame:
            too_big |= cleaned_frame[column] > INT_UNSIGNED_MAX
    reason = pd.Series(None, index=cleaned_frame.index, dtype=object)
    reason[too_big] = f"count too large for the database column (max {INT_UNSIGNED_MAX:,})"
    reason[too_long] = ("value does not fit the database column (country_code must be 2 letters, "
                        "IDs at most 32 characters, payment_method at most 40)")
    bad = reason.notna()
    return cleaned_frame[~bad], cleaned_frame[bad].assign(reason=reason[bad])


def reject_future_dates(cleaned_frame, now=None):
    """Orders or first-order dates later than now cannot be real, so they are rejected
    (otherwise the integrity check "no future dates" would fail after loading)."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    future = pd.Series(False, index=cleaned_frame.index)
    for column in DATE_COLUMNS:
        if column in cleaned_frame:
            future |= pd.to_datetime(cleaned_frame[column]) > now
    return cleaned_frame[~future], cleaned_frame[future].assign(reason="date is in the future")


def transform(raw, now=None):
    """Clean the 3 raw tables with the team's shared rules (same as the historical data),
    then reject rows that cannot be stored. `now` is only set by the tests."""
    tagged = {kind: frame.assign(_row=range(len(frame))) for kind, frame in raw.items()}
    customers, rej_c = dp.clean_customers(tagged["customers"])
    orders, rej_o = dp.clean_orders(tagged["orders"])
    labels, rej_l = dp.clean_labels(tagged["labels"])
    dp.validate_or_raise(customers, orders, labels)     # stop before loading anything bad
    customers, unfit_c = reject_unfit_for_schema(customers)
    orders, unfit_o = reject_unfit_for_schema(orders)
    labels, unfit_l = reject_unfit_for_schema(labels)
    customers, future_c = reject_future_dates(customers, now)
    orders, future_o = reject_future_dates(orders, now)
    rej_c, rej_o, rej_l = (pd.concat([f for f in group if not f.empty] or [group[0]], ignore_index=True)
                           for group in ((rej_c, unfit_c, future_c), (rej_o, unfit_o, future_o),
                                         (rej_l, unfit_l)))
    rejected = []
    for kind, table, id_column, rej in (("customers", "customers", "customer_id", rej_c),
                                        ("orders", "orders", "order_id", rej_o),
                                        ("labels", "fraud_labels", "order_id", rej_l)):
        rejected += rejected_from_cleaning(rej, raw[kind], table, id_column)
        rejected += rejected_bad_lines(raw[kind].attrs.get("bad_lines", []), list(raw[kind].columns),
                                       table, id_column)
    return {"customers": customers, "orders": orders, "fraud_labels": labels}, rejected


# ---------------------------------------------------------------------------
# Load (#45, #46)
# ---------------------------------------------------------------------------
def to_records(frame, columns):
    """DataFrame -> list of dicts with plain Python values (None for missing)."""
    out = frame[columns].copy()
    if "order_date" in out:
        out["order_date"] = pd.to_datetime(out["order_date"]).dt.date
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict("records")


def insert_many(conn, sql, records):
    for start in range(0, len(records), INSERT_CHUNK):
        conn.execute(text(sql), records[start:start + INSERT_CHUNK])


def stage(conn, table, frame):
    """Copy cleaned rows into a temporary staging table."""
    columns = ["_row"] + TABLE_COLUMNS[table]
    conn.execute(text(f"DROP TEMPORARY TABLE IF EXISTS stg_{table}"))
    conn.execute(text(STAGING_DDL[table]))
    sql = f"INSERT INTO stg_{table} ({', '.join(columns)}) VALUES ({', '.join(':' + c for c in columns)})"
    insert_many(conn, sql, to_records(frame, columns))


def upsert_from_stage(conn, table, batch_name):
    """Latest wins: insert new rows, replace existing ones. Returns (inserted, updated)."""
    keys, columns = TABLE_KEYS[table], TABLE_COLUMNS[table]
    join = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    total = conn.execute(text(f"SELECT COUNT(*) FROM stg_{table}")).scalar()
    existing = conn.execute(text(
        f"SELECT COUNT(*) FROM stg_{table} s JOIN {table} t ON {join}")).scalar()
    updates = ", ".join(f"{c} = s.{c}" for c in columns if c not in keys)
    conn.execute(text(
        f"INSERT INTO {table} ({', '.join(columns)}, first_loaded_batch, last_updated_batch) "
        f"SELECT {', '.join('s.' + c for c in columns)}, :batch, :batch FROM stg_{table} s "
        f"ON DUPLICATE KEY UPDATE {updates}, last_updated_batch = :batch"), {"batch": batch_name})
    return total - existing, existing


def reject_orphan_labels(conn, raw_labels):
    """Labels whose order or customer is not in the database cannot form a complete row
    (team Assumption 14): remove them from staging and return them as rejected rows."""
    orphans = conn.execute(text("""
        SELECT s._row, (o.order_id IS NULL) AS no_order, (c.customer_id IS NULL) AS no_customer
        FROM stg_fraud_labels s
        LEFT JOIN orders o    ON o.country_code = s.country_code AND o.order_id = s.order_id
        LEFT JOIN customers c ON c.country_code = s.country_code AND c.customer_id = s.customer_id
        WHERE o.order_id IS NULL OR c.customer_id IS NULL""")).fetchall()
    if not orphans:
        return []
    rejected = []
    for row, no_order, no_customer in orphans:
        reason = ("order and customer not found" if no_order and no_customer
                  else "order not found" if no_order else "customer not found")
        raw = raw_labels.iloc[int(row)]
        rejected.append({"source_table": "fraud_labels",
                         "country_code": None if pd.isna(raw["country_code"]) else str(raw["country_code"])[:16],
                         "record_id": None if pd.isna(raw["order_id"]) else str(raw["order_id"])[:64],
                         "reason": reason, "raw_record": json_record(raw)})
    # One statement removes them all (deleting row by row is very slow on large batches)
    conn.execute(text("""
        DELETE s FROM stg_fraud_labels s
        LEFT JOIN orders o    ON o.country_code = s.country_code AND o.order_id = s.order_id
        LEFT JOIN customers c ON c.country_code = s.country_code AND c.customer_id = s.customer_id
        WHERE o.order_id IS NULL OR c.customer_id IS NULL"""))
    return rejected


def start_batch(engine, batch_name):
    """Mark the batch as running (a re-run of the same batch resets its row)."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO etl_batches (batch_name, status, started_at) VALUES (:b, 'running', :t)
            ON DUPLICATE KEY UPDATE status = 'running', started_at = :t, finished_at = NULL,
                customers_inserted = 0, customers_updated = 0, orders_inserted = 0,
                orders_updated = 0, labels_inserted = 0, labels_updated = 0,
                rows_rejected = 0, error_message = NULL"""),
            {"b": batch_name, "t": datetime.now()})


def short_error(error):
    """The database's own message, without the (possibly huge) list of rows SQLAlchemy appends."""
    return str(getattr(error, "orig", error))[:2000]


def fail_batch(engine, batch_name, error):
    with engine.begin() as conn:
        conn.execute(text("UPDATE etl_batches SET status = 'failed', finished_at = :t, "
                          "error_message = :e WHERE batch_name = :b"),
                     {"b": batch_name, "t": datetime.now(), "e": short_error(error)})


def load(engine, batch_name, cleaned, rejected, raw):
    """Write one batch to RDS in a single transaction. Returns the counts."""
    counts = {}
    with engine.begin() as conn:
        # A re-run of the same batch replaces its earlier rejected rows
        conn.execute(text("DELETE FROM rejected_records WHERE batch_name = :b"), {"b": batch_name})
        for table in ("customers", "orders"):                      # parents first
            stage(conn, table, cleaned[table])
            counts[f"{table}_inserted"], counts[f"{table}_updated"] = upsert_from_stage(conn, table, batch_name)
        stage(conn, "fraud_labels", cleaned["fraud_labels"])
        rejected = rejected + reject_orphan_labels(conn, raw["labels"])
        counts["labels_inserted"], counts["labels_updated"] = upsert_from_stage(conn, "fraud_labels", batch_name)
        insert_many(conn, "INSERT INTO rejected_records (batch_name, source_table, country_code, "
                          "record_id, reason, raw_record) VALUES (:batch_name, :source_table, "
                          ":country_code, :record_id, :reason, :raw_record)",
                    [{**r, "batch_name": batch_name} for r in rejected])
        counts["rows_rejected"] = len(rejected)
        conn.execute(text("""
            UPDATE etl_batches SET status = 'completed', finished_at = :t,
                customers_inserted = :customers_inserted, customers_updated = :customers_updated,
                orders_inserted = :orders_inserted, orders_updated = :orders_updated,
                labels_inserted = :labels_inserted, labels_updated = :labels_updated,
                rows_rejected = :rows_rejected
            WHERE batch_name = :b"""), {**counts, "b": batch_name, "t": datetime.now()})
    return counts, rejected


# ---------------------------------------------------------------------------
# Whole pipeline
# ---------------------------------------------------------------------------
def run_etl(sources, batch_name=None, raw_dir=DEFAULT_RAW_DIR, dry_run=False, engine=None,
            summary_dir=DEFAULT_SUMMARY_DIR):
    """Extract -> transform -> load one batch. Returns a summary dict (for Prefect / retraining)."""
    if batch_name is None:
        names = {batch_name_from_source(sources[kind]) for kind in FILE_KINDS}
        if len(names) != 1:
            raise ValueError(f"The 3 files have different postfixes {sorted(names)}; pass --batch-name")
        batch_name = names.pop()
    batch_name = check_batch_name(batch_name)
    timings, start = {}, time.perf_counter()
    print(f"=== ETL batch '{batch_name}'")

    step = time.perf_counter()
    raw = extract(sources, batch_name, raw_dir)
    timings["extract_seconds"] = time.perf_counter() - step
    print("Extracted:", {k: len(v) for k, v in raw.items()}, f"({timings['extract_seconds']:.1f}s)")

    step = time.perf_counter()
    cleaned, rejected = transform(raw)
    timings["transform_seconds"] = time.perf_counter() - step
    print("Cleaned:", {k: len(v) for k, v in cleaned.items()},
          f"| rejected by cleaning: {len(rejected)} ({timings['transform_seconds']:.1f}s)")

    summary = {"batch_name": batch_name, "rows_read": {k: len(v) for k, v in raw.items()},
               "rows_cleaned": {k: len(v) for k, v in cleaned.items()}, "dry_run": dry_run}
    if dry_run:
        summary["rows_rejected"] = len(rejected)
        summary["rejected_reasons"] = pd.Series([r["reason"] for r in rejected], dtype=object).value_counts().to_dict()
    else:
        if engine is None:
            import database as db
            engine = db.get_engine()
        start_batch(engine, batch_name)
        step = time.perf_counter()
        try:
            counts, all_rejected = load(engine, batch_name, cleaned, rejected, raw)
        except Exception as err:
            fail_batch(engine, batch_name, err)
            raise
        timings["load_seconds"] = time.perf_counter() - step
        summary.update(counts)
        summary["rejected_reasons"] = pd.Series([r["reason"] for r in all_rejected], dtype=object).value_counts().to_dict()
        print("Loaded:", counts, f"({timings['load_seconds']:.1f}s)")

    timings["total_seconds"] = time.perf_counter() - start
    summary.update({k: round(v, 2) for k, v in timings.items()})
    if summary["rejected_reasons"]:
        print("Rejected rows by reason:", summary["rejected_reasons"])
    print(f"Total time: {summary['total_seconds']:.1f}s")

    Path(summary_dir).mkdir(parents=True, exist_ok=True)
    (Path(summary_dir) / f"{batch_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="SP-buy ETL: CSV files -> cleaned -> AWS RDS MySQL")
    parser.add_argument("--customers", help="URL or path of customers_XXX.csv")
    parser.add_argument("--orders", help="URL or path of orders_XXX.csv")
    parser.add_argument("--labels", help="URL or path of labels_XXX.csv")
    parser.add_argument("--data-dir", help="Folder with the 3 files (instead of --customers/--orders/--labels)")
    parser.add_argument("--postfix", help="File postfix with --data-dir, e.g. base or XXX")
    parser.add_argument("--batch-name", help="Override the batch name (default: the file postfix)")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Where raw copies are kept")
    parser.add_argument("--dry-run", action="store_true", help="Extract and clean only; do not touch the database")
    args = parser.parse_args()

    if args.data_dir:
        if not args.postfix:
            parser.error("--data-dir needs --postfix (e.g. --postfix base)")
        sources = {k: Path(args.data_dir) / f"{k}_{args.postfix}.csv" for k in FILE_KINDS}
    elif args.customers and args.orders and args.labels:
        sources = {"customers": args.customers, "orders": args.orders, "labels": args.labels}
    else:
        parser.error("give --customers, --orders and --labels, or --data-dir with --postfix")

    try:
        run_etl(sources, batch_name=args.batch_name or (args.postfix if args.data_dir else None),
                raw_dir=args.raw_dir, dry_run=args.dry_run)
    except Exception as err:
        print(f"ETL FAILED: {short_error(err)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
