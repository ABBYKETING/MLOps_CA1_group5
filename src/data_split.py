
"""
ST1519 Agile MLOps CA1
Shared Chronological Data Split

Uses the same chronological boundaries for all team members
and both feature datasets.

Training:   Up to 2023-03-15
Validation:  2023-03-16 to 2023-04-13
Testing:     2023-04-14 onwards
"""

import pandas as pd

# Shared historical cutoff dates
TRAIN_END = pd.Timestamp("2023-03-15")
VAL_END = pd.Timestamp("2023-04-13")


def create_split_labels(df, train_end=TRAIN_END, val_end=VAL_END):
    """Assign Train, Validation or Test using order dates."""

    dates = pd.to_datetime(
        df["order_date"], errors="raise"
    ).dt.normalize()

    assert dates.notna().all(), "Missing order dates found."
    assert train_end < val_end, "Invalid cutoff dates."

    labels = pd.Series(index=df.index, dtype="string")

    labels.loc[dates <= train_end] = "Train"

    labels.loc[
        (dates > train_end) & (dates <= val_end)
    ] = "Validation"

    labels.loc[dates > val_end] = "Test"

    assert labels.notna().all(), "Unassigned orders found."

    return labels
