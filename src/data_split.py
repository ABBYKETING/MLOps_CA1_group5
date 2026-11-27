
"""
ST1519 Agile MLOps CA1
Shared Chronological Data Split

Uses the same chronological boundaries for all team members.
Prepared PBI 5 data carries a validated expanding/28-day/28-day policy.
Existing PBI 2 data retains the historical boundaries below.

Training:   Up to 2023-03-15
Validation:  2023-03-16 to 2023-04-13
Testing:     2023-04-14 onwards
"""

import pandas as pd

# Shared historical cutoff dates
TRAIN_END = pd.Timestamp("2023-03-15")
VAL_END = pd.Timestamp("2023-04-13")


def rolling_split_policy(df):
    """PBI 5: expanding Train, followed by 28 calendar days each of Val/Test.

    Anchor on the latest labelled order date, never the machine's current date.
    Whole dates stay together, including orders with a time component.
    """
    dates = pd.to_datetime(df["order_date"], errors="raise").dt.normalize()
    if dates.empty or dates.isna().any():
        raise ValueError("Retraining requires nonempty, valid order dates")
    anchor = dates.max()
    test_start = anchor - pd.Timedelta(days=27)
    validation_start = test_start - pd.Timedelta(days=28)
    return {
        "name": "expanding_28d_v1",
        "anchor_date": anchor.date().isoformat(),
        "train_end": (validation_start - pd.Timedelta(days=1)).date().isoformat(),
        "validation_start": validation_start.date().isoformat(),
        "validation_end": (test_start - pd.Timedelta(days=1)).date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": anchor.date().isoformat(),
        "validation_days": 28, "test_days": 28,
    }


def create_split_labels(df, train_end=None, val_end=None):
    """Use historical dates, or the validated retraining policy in df.attrs.

    model_utils.load_feature_data attaches the policy from the CSV's checked
    metadata file. No process-wide defaults are changed for notebooks/PBI 2.
    """
    policy = df.attrs.get("retraining_split")
    if policy is not None:
        if train_end is not None or val_end is not None:
            raise ValueError("Cannot override the prepared retraining split")
        if policy != rolling_split_policy(df):
            raise ValueError("Retraining split metadata does not match the order dates")
        train_end = pd.Timestamp(policy["train_end"])
        val_end = pd.Timestamp(policy["validation_end"])
    else:
        train_end = TRAIN_END if train_end is None else pd.Timestamp(train_end)
        val_end = VAL_END if val_end is None else pd.Timestamp(val_end)

    dates = pd.to_datetime(
        df["order_date"], errors="raise"
    ).dt.normalize()

    if dates.isna().any():
        raise ValueError("Missing order dates found")
    if train_end >= val_end:
        raise ValueError("Invalid cutoff dates")

    labels = pd.Series(index=df.index, dtype="string")

    labels.loc[dates <= train_end] = "Train"

    labels.loc[
        (dates > train_end) & (dates <= val_end)
    ] = "Validation"

    labels.loc[dates > val_end] = "Test"

    if labels.isna().any():
        raise ValueError("Unassigned orders found")

    return labels


def split_summary(df):
    """Validate both classes and record actual date ranges and class counts."""
    labels = create_split_labels(df)
    dates = pd.to_datetime(df["order_date"], errors="raise").dt.normalize()
    summary = {}
    for name in ("Train", "Validation", "Test"):
        mask = labels == name
        y = df.loc[mask, "is_fraud"]
        if set(y.unique()) != {0, 1}:
            raise ValueError(f"{name} must contain both fraud and non-fraud orders; "
                             "check data coverage before retraining")
        fraud = int(y.sum())
        summary[name] = {
            "rows": len(y), "fraud": fraud, "non_fraud": len(y) - fraud,
            "fraud_rate": fraud / len(y),
            "first_order_date": dates[mask].min().date().isoformat(),
            "last_order_date": dates[mask].max().date().isoformat(),
        }
    return summary
