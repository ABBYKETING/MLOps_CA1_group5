"""PBI 5 task #50: prepare combined RDS records for Safe model retraining.

The PBI 4 ETL has already cleaned/upserted these records. Read the current state
once, reuse PBI 1 feature engineering, and save one shared chronological split.
This stage reads the database; it never appends a CSV batch or writes to RDS.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

SOURCE_DIR = str(Path(__file__).resolve().parent)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

import data_processing as dp
from data_split import rolling_split_policy, split_summary
import database as db
import model_utils as mu


def prepare_retraining_data(*, output_dir, engine=None):
    """Read latest labelled RDS rows and save Safe CSV + checked JSON metadata.

    output_dir must be new to prevent overwriting a previous data snapshot.
    An injected SQLAlchemy engine supports local tests and caller-owned sessions.
    Without one, DB_* environment variables configure the RDS connection.
    """
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    owned_engine = engine is None
    if owned_engine:
        engine = db.get_engine()
    try:
        merged = db.load_training_data(engine)
    finally:
        if owned_engine:
            engine.dispose()
    required = mu.META_COLS + [mu.TARGET]
    missing = sorted(set(required) - set(merged.columns))
    if missing:
        raise ValueError(f"RDS training rows missing required columns: {missing}")
    if merged.empty:
        raise ValueError("No labelled orders available in the combined database")
    for name in ("country_code", "order_id", "customer_id"):
        if merged[name].isna().any() or merged[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {name} in RDS training rows")
    if merged.duplicated(["country_code", "order_id"]).any():
        raise ValueError("Duplicate (country_code, order_id) in RDS training rows; check the join/schema")
    if not merged[mu.TARGET].isin([0, 1]).all():
        raise ValueError("RDS labels must contain only 0 and 1")
    merged = merged.sort_values(["order_date", "country_code", "order_id"]).reset_index(drop=True)
    policy = rolling_split_policy(merged)
    merged.attrs["retraining_split"] = policy
    partitions = split_summary(merged)

    # Reuse the validated team's rules. Only the Safe set is persisted/trained.
    full, safe, y = dp.build_feature_sets(merged)
    del full
    if set(safe.columns).intersection(dp.SNAPSHOT_FEATURES + mu.FORBIDDEN_FEATURES):
        raise ValueError("Excluded feature found in the Safe retraining dataset")
    frame = merged[mu.META_COLS].copy()
    # country_code is metadata; its fixed dummy columns are numeric features.
    frame = frame.join(safe)
    frame[mu.TARGET] = y.astype(int)
    path = folder / "features_base_safe.csv"
    frame.to_csv(path, index=False)
    metadata = {
        "schema_version": 1, "dataset_type": "safe", "source": "rds_current_state",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "sha256": mu.file_sha256(path), "total_rows": len(frame),
        "features": list(safe.columns), "feature_count": len(safe.columns),
        "split": policy, "partitions": partitions,
        "excluded_features": ["refund_value"] + list(dp.SNAPSHOT_FEATURES),
    }
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New directory for the prepared snapshot")
    args = parser.parse_args()
    try:
        metadata = prepare_retraining_data(output_dir=args.output_dir)
    except Exception as error:
        print(f"Data preparation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
