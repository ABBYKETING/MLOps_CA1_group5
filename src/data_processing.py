"""Reusable SP-buy historical/new-batch data processing (ST1519 CA1, PBI 1).

Source: 01_data_cleaning(1).ipynb, uploaded 2026-11-22.
The cleaning, validation, merging and feature engineering logic below is
extracted from the notebook, preserving its business assumptions and the
original order of engineered columns.

The loader's personal Windows path was removed for portability. Call
``load_raw(data_dir=..., postfix=...)`` with local batch files, or pass
DataFrames from the ETL URL-download step into the cleaning functions.

This is a processing module, NOT the complete ETL pipeline: it neither
connects to RDS nor triggers training/MLflow. Run data validation before
writing to your database. ``SNAPSHOT_FEATURES`` are excluded from the
restricted candidate; this does not certify that all remaining features
are available at prediction time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Read historical/new-batch CSV files from a configurable location
CUSTOMER_TEXT_COLS = ["country_code", "customer_id", "mobile_verified", "first_order_datetime"]
ORDER_TEXT_COLS = ["country_code", "order_id", "collect_type", "payment_method", "order_date"]
LABEL_TEXT_COLS = ["country_code", "order_id", "customer_id"]


def load_raw(data_dir, postfix="base"):
    """Load the three raw csv files for one batch."""
    data_dir = Path(data_dir)
    customers = pd.read_csv(data_dir / f"customers_{postfix}.csv",
                            dtype={c: str for c in CUSTOMER_TEXT_COLS})
    orders = pd.read_csv(data_dir / f"orders_{postfix}.csv",
                         dtype={c: str for c in ORDER_TEXT_COLS})
    labels = pd.read_csv(data_dir / f"labels_{postfix}.csv",
                         dtype={c: str for c in LABEL_TEXT_COLS})
    return customers, orders, labels


# General cleaning helpers
def clean_text(s):
    """Strip spaces and turn empty strings into missing values."""
    s = s.astype("object").where(s.notna(), None)
    s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    return s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})


def clean_country(s):
    """Assumption 1: country codes in uppercase."""
    return clean_text(s).str.upper()


def to_number(s):
    """Convert to numbers; text and infinite values become missing."""
    result = pd.to_numeric(s, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


BOOL_MAP = {"true": True, "yes": True, "y": True, "1": True,
            "false": False, "no": False, "n": False, "0": False}


def clean_bool(s):
    """Assumption 2: map text such as 'yes' or 'TRUE' to True or False."""
    return clean_text(s).astype(str).str.lower().map(BOOL_MAP)


DATE_FORMATS = ["%Y/%m/%d", "%Y-%m-%d", "%Y/%b/%d"]


def parse_dates(s, formats=DATE_FORMATS):
    """Assumption 7: try each known date format; values matching none stay missing."""
    s = clean_text(s)
    result = pd.Series(pd.NaT, index=s.index)
    for fmt in formats:
        missing = result.isna() & s.notna()
        result[missing] = pd.to_datetime(s[missing], format=fmt, errors="coerce")
    return result


def reject(df, mask, reason, rejected):
    """Assumption 18: move rows that cannot be fixed into the rejected list with a reason."""
    if mask.any():
        rejected.append(df[mask].assign(reason=reason))
    return df[~mask]


def combine_rejected(rejected, columns):
    """Join all rejected rows into one table (empty table if none)."""
    if rejected:
        return pd.concat(rejected, ignore_index=True)
    return pd.DataFrame(columns=list(columns) + ["reason"])


# Customer cleaning
CUSTOMER_COUNT_COLS = ["num_orders_last_50days", "num_cancelled_orders_last_50days",
                       "num_refund_orders_last_50days", "num_associated_customers"]
CUSTOMER_NUM_COLS = CUSTOMER_COUNT_COLS + ["total_payment_last_50days"]


def clean_customers(df):
    """Return (cleaned customers, rejected customers with a reason)."""
    df = df.copy()
    rejected = []
    df["country_code"] = clean_country(df["country_code"])
    df["customer_id"] = clean_text(df["customer_id"])
    df = reject(df, df["country_code"].isna() | df["customer_id"].isna(), "missing country_code or customer_id", rejected)
    df["mobile_verified"] = clean_bool(df["mobile_verified"])
    df = reject(df, df["mobile_verified"].isna(), "unrecognised mobile_verified", rejected)
    for col in CUSTOMER_NUM_COLS:
        df[col] = to_number(df[col])
    df = reject(df, df[CUSTOMER_NUM_COLS].isna().any(axis=1), "missing or non-numeric value", rejected)
    df = reject(df, (df[CUSTOMER_NUM_COLS] < 0).any(axis=1), "negative value", rejected)
    # Counts must be whole numbers (total payment can have decimals)
    df = reject(df, (df[CUSTOMER_COUNT_COLS] % 1 != 0).any(axis=1), "count is not a whole number", rejected)
    # Assumption 8: a missing first order date is allowed, but a date that cannot be read is rejected
    raw_date = clean_text(df["first_order_datetime"])
    parsed = pd.to_datetime(raw_date, format="ISO8601", errors="coerce")
    df = reject(df, raw_date.notna() & parsed.isna(), "unparseable first_order_datetime", rejected)
    df["first_order_datetime"] = parsed[df.index]
    df = reject(df, df.duplicated(["country_code", "customer_id"], keep="last"), "duplicate customer (kept last in file)", rejected)
    return df.reset_index(drop=True), combine_rejected(rejected, df.columns)


# Order cleaning
COLLECT_MAP = {"delivery": "Delivery", "pickup": "Pickup"}
PAYMENT_MAP = {"on delivery": "payment on delivery"}
ITEMS_FILL_VALUE = 2  # median of num_items_ordered in the historical data
ORDER_NUM_COLS = ["order_value", "num_items_ordered", "refund_value"]


def clean_orders(df, items_fill_value=ITEMS_FILL_VALUE):
    """Return (cleaned orders, rejected orders with a reason)."""
    df = df.copy()
    rejected = []
    df["country_code"] = clean_country(df["country_code"])
    df["order_id"] = clean_text(df["order_id"])
    df = reject(df, df["country_code"].isna() | df["order_id"].isna(), "missing country_code or order_id", rejected)
    # Assumption 3: standardise collect type
    df["collect_type"] = clean_text(df["collect_type"]).str.lower().map(COLLECT_MAP)
    df = reject(df, df["collect_type"].isna(), "unrecognised collect_type", rejected)
    # Assumptions 4 and 6: merge payment variants, fill missing with "unknown"
    payment = clean_text(df["payment_method"]).str.lower().replace(PAYMENT_MAP)
    df["payment_method"] = payment.fillna("unknown")
    raw_items = clean_text(df["num_items_ordered"])
    for col in ORDER_NUM_COLS:
        df[col] = to_number(df[col])
    # Text such as "abc" is rejected; only a truly empty item count is filled
    df = reject(df, raw_items.notna() & df["num_items_ordered"].isna(), "non-numeric item count", rejected)
    # Assumption 9: fill missing item count with the historical median
    df["num_items_ordered"] = df["num_items_ordered"].fillna(items_fill_value)
    df = reject(df, df[ORDER_NUM_COLS].isna().any(axis=1), "missing or non-numeric value", rejected)
    df = reject(df, (df[ORDER_NUM_COLS] < 0).any(axis=1), "negative value", rejected)
    # An item count must be a whole number of at least 1
    df = reject(df, (df["num_items_ordered"] % 1 != 0) | (df["num_items_ordered"] < 1),
                "item count not a whole number of at least 1", rejected)
    df["num_items_ordered"] = df["num_items_ordered"].astype(int)
    df["order_date"] = parse_dates(df["order_date"])
    df = reject(df, df["order_date"].isna(), "missing or unknown order_date format", rejected)
    # Assumption 11: an order is identified by country_code and order_id together
    df = reject(df, df.duplicated(["country_code", "order_id"], keep="last"),
                "duplicate order (kept last in file)", rejected)
    return df.reset_index(drop=True), combine_rejected(rejected, df.columns)


# Fraud-label cleaning
def clean_labels(df):
    """Return (cleaned labels, rejected labels with a reason)."""
    df = df.copy()
    rejected = []
    df["country_code"] = clean_country(df["country_code"])
    df["order_id"] = clean_text(df["order_id"])
    df["customer_id"] = clean_text(df["customer_id"])
    df = reject(df, df[["country_code", "order_id", "customer_id"]].isna().any(axis=1), "missing ID", rejected)
    df["is_fraud"] = to_number(df["is_fraud"])
    df = reject(df, ~df["is_fraud"].isin([0, 1]), "is_fraud not 0 or 1", rejected)
    df["is_fraud"] = df["is_fraud"].astype(int)
    df = reject(df, df.duplicated(["country_code", "order_id"], keep="last"), "duplicate order (kept last in file)", rejected)
    return df.reset_index(drop=True), combine_rejected(rejected, df.columns)


# Cleaned table validation
def validate(customers, orders, labels):
    """Count the rows that fail each check. A check passes when 0 rows fail."""
    checks = {
        "customers: country codes uppercase": (customers["country_code"] != customers["country_code"].str.upper()).sum(),
        "customers: mobile_verified True/False only": (~customers["mobile_verified"].isin([True, False])).sum(),
        "customers: no duplicate customers": customers.duplicated(["country_code", "customer_id"]).sum(),
        "customers: no missing numbers": customers[CUSTOMER_NUM_COLS].isna().sum().sum(),
        "customers: no negative numbers": (customers[CUSTOMER_NUM_COLS] < 0).sum().sum(),
        "customers: counts are whole numbers": (customers[CUSTOMER_COUNT_COLS] % 1 != 0).sum().sum(),
        "orders: country codes uppercase": (orders["country_code"] != orders["country_code"].str.upper()).sum(),
        "orders: collect_type Delivery/Pickup only": (~orders["collect_type"].isin(["Delivery", "Pickup"])).sum(),
        "orders: no missing payment_method": orders["payment_method"].isna().sum(),
        "orders: item count whole number of at least 1": ((orders["num_items_ordered"] < 1) | (orders["num_items_ordered"] % 1 != 0)).sum(),
        "orders: all order_date parsed": orders["order_date"].isna().sum(),
        "orders: no missing numbers": orders[ORDER_NUM_COLS].isna().sum().sum(),
        "orders: no negative numbers": (orders[ORDER_NUM_COLS] < 0).sum().sum(),
        "orders: no duplicate orders": orders.duplicated(["country_code", "order_id"]).sum(),
        "labels: country codes uppercase": (labels["country_code"] != labels["country_code"].str.upper()).sum(),
        "labels: is_fraud 0/1 only": (~labels["is_fraud"].isin([0, 1])).sum(),
        "labels: no duplicate orders": labels.duplicated(["country_code", "order_id"]).sum(),
    }
    report = pd.DataFrame({"failed_rows": pd.Series(checks)})
    report["passed"] = report["failed_rows"] == 0
    return report


# Label/order/customer merging
def merge_datasets(customers, orders, labels):
    """Assumption 14: keep only orders that have a label, order details and customer details."""
    matched = labels.merge(orders, on=["country_code", "order_id"], how="inner", validate="one_to_one")
    merged = matched.merge(customers, on=["country_code", "customer_id"], how="inner", validate="many_to_one")
    print("Labels without a matching order:   ", len(labels) - len(matched))
    print("Orders without a matching customer:", len(matched) - len(merged))
    return merged


# Feature engineering and fixed schema
COUNTRIES = ["BD", "MY", "PH", "PK", "TH"]
COLLECT_TYPES = ["Delivery", "Pickup"]
PAYMENT_METHODS = ["payment on delivery", "cybersource", "online banking", "credit card",
                   "antfinancial gcash", "generic creditcard", "balance", "unknown",
                   "antfinancial bkash", "antfinancial truemoney", "boost wallet",
                   "jazzcash wallet", "invoice", "adyen hpp molpay", "no payment",
                   "cybersource applepay", "paypal"]
SKEWED_COLS = ["order_value", "num_items_ordered", "num_orders_last_50days",
               "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
               "total_payment_last_50days", "num_associated_customers"]

# Features from the customer snapshot, whose timing is unconfirmed (Assumption 19)
SNAPSHOT_FEATURES = ["cancel_rate", "refund_rate", "avg_payment_per_order",
                     "mobile_verified", "associated_capped",
                     "log_num_orders_last_50days", "log_num_cancelled_orders_last_50days",
                     "log_num_refund_orders_last_50days", "log_total_payment_last_50days",
                     "log_num_associated_customers"]


def safe_divide(a, b):
    """Divide a by b, giving 0 where b is 0."""
    return np.where(b > 0, a / b.where(b > 0, 1), 0.0)


def build_features(df):
    out = pd.DataFrame(index=df.index)

    # Behaviour ratios
    out["value_per_item"] = safe_divide(df["order_value"], df["num_items_ordered"])
    out["cancel_rate"] = safe_divide(df["num_cancelled_orders_last_50days"], df["num_orders_last_50days"])
    out["refund_rate"] = safe_divide(df["num_refund_orders_last_50days"], df["num_orders_last_50days"])
    out["avg_payment_per_order"] = safe_divide(df["total_payment_last_50days"], df["num_orders_last_50days"])

    # Account age (Assumptions 8 and 17)
    age = (
        df["order_date"] -
        df["first_order_datetime"].dt.normalize()
    ).dt.days

    # Flag missing first-order dates
    out["first_order_missing"] = age.isna().astype(int)

    # Flag invalid negative account ages
    out["account_age_invalid"] = (age < 0).astype(int)

    # Replace missing/negative ages with 0 for numerical processing
    out["account_age_days"] = age.clip(lower=0).fillna(0)

    # Flags (Assumption 12)
    out["mobile_verified"] = df["mobile_verified"].astype(int)
    out["associated_capped"] = (df["num_associated_customers"] >= 100).astype(int)

    # Log transform to reduce skewness (Assumption 13)
    for col in SKEWED_COLS:
        out["log_" + col] = np.log1p(df[col])

    # One-hot encoding with fixed categories, unseen values go to "other"
    for col, cats in [("country_code", COUNTRIES), ("collect_type", COLLECT_TYPES),
                      ("payment_method", PAYMENT_METHODS)]:
        values = df[col].where(df[col].isin(cats), "other")
        for cat in cats + (["other"] if col == "payment_method" else []):
            out[f"{col}_{cat.replace(' ', '_')}"] = (values == cat).astype(int)

    return out


# Convenience helpers for ETL integration (not present in the EDA notebook).
def validate_or_raise(customers, orders, labels):
    """Return validation report, or raise to stop ETL before loading bad data."""
    report = validate(customers, orders, labels)
    failed = report.loc[~report["passed"], "failed_rows"]
    if not failed.empty:
        raise ValueError("Cleaned-data validation failed: " + failed.to_dict().__repr__())
    return report


def build_feature_sets(merged):
    """Return (X_full, X_restricted, y) using the notebook's feature rules.

    ``X_full`` is for *exploratory comparison only*. ``X_restricted`` drops
    the 10 snapshot features of Assumption 19. Verify the real-world
    availability of all remaining features before final deployment.
    """
    X = build_features(merged)
    y = merged["is_fraud"]
    missing_names = [name for name in SNAPSHOT_FEATURES if name not in X.columns]
    if missing_names:
        raise ValueError(f"Snapshot feature names absent from X: {missing_names}")
    if not X.columns.is_unique:
        raise ValueError("Duplicate feature names")
    if "refund_value" in X.columns or "is_fraud" in X.columns:
        raise ValueError("Leakage column in features")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Missing or infinite engineered features")
    X_safe = X.drop(columns=SNAPSHOT_FEATURES)
    if set(SNAPSHOT_FEATURES).intersection(X_safe.columns):
        raise ValueError("Snapshot features remain in the restricted feature set")
    return X, X_safe, y


__all__ = [
    "load_raw", "clean_text", "clean_country", "to_number", "clean_bool",
    "parse_dates", "reject", "combine_rejected", "clean_customers",
    "clean_orders", "clean_labels", "validate", "validate_or_raise",
    "merge_datasets", "safe_divide", "build_features", "build_feature_sets",
    "CUSTOMER_TEXT_COLS", "ORDER_TEXT_COLS", "LABEL_TEXT_COLS",
    "BOOL_MAP", "DATE_FORMATS", "CUSTOMER_COUNT_COLS", "CUSTOMER_NUM_COLS",
    "COLLECT_MAP", "PAYMENT_MAP", "ITEMS_FILL_VALUE", "ORDER_NUM_COLS",
    "COUNTRIES", "COLLECT_TYPES", "PAYMENT_METHODS", "SKEWED_COLS",
    "SNAPSHOT_FEATURES",
]
