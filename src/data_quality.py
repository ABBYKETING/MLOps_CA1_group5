"""Data quality monitoring for incoming ETL batches (PBI 11, issue #85, Yue Liang).

Every batch processed by src/etl_pipeline.py is checked BEFORE cleaning, so the
report describes the files exactly as they arrived (business rule 4 of #85):

    schema       missing required columns (critical), unexpected extra columns
    missing      missing values in required and optional columns
    duplicates   exact duplicate rows and repeated record keys
    types        values that are not valid numbers, whole numbers, booleans, dates,
                 0/1 labels or known categories; negative amounts
    structure    rows with the wrong number of fields

Severity:
    critical  the batch cannot be processed (unreadable or empty file, missing
              required columns, file not found / not downloadable). The ETL stops
              before anything is loaded and a critical report is still written.
    error     the record cannot be stored as it is; the ETL rejects it into
              rejected_records with a reason.
    warning   the record is still loaded, but something is unusual (an optional
              value is missing and filled, a new category appears, a repeated
              key where the last copy wins, an extra column that is ignored).

The checks mirror the team's cleaning rules in data_processing.py (PBI 1), so a
record flagged as "error" here is the same record the ETL later rejects.

Reports (git-ignored, one folder per batch, plus a history file for comparison):

    outputs/data_quality/<batch>/quality_issues.csv     dataset, column, check, severity,
                                                         affected records, examples
    outputs/data_quality/<batch>/processing_summary.csv  per dataset: input records,
                                                         records with issues, rejected,
                                                         successfully processed
    outputs/data_quality/<batch>/report.md               the same, readable
    outputs/data_quality/history.csv                     one row per dataset per run

Usage: called automatically by run_etl(); it can also check files on their own:

    python src/data_quality.py --data-dir path/to/folder --postfix XXX
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_processing as dp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO_ROOT / "outputs" / "data_quality"     # git-ignored
FILE_KINDS = ("customers", "orders", "labels")
TABLE_FOR_KIND = {"customers": "customers", "orders": "orders", "labels": "fraud_labels"}
EXAMPLES = 3                                                     # example values per issue

CRITICAL, ERROR, WARNING = "critical", "error", "warning"

# Expected schema of each incoming file: column -> (value type, required?)
# "required" columns must have a value; optional ones may be empty (the cleaning
# rules fill or allow them: Assumptions 6, 8 and 9).
SCHEMA = {
    "customers": {
        "key": ["country_code", "customer_id"],
        "columns": {
            "country_code": ("country", True),
            "customer_id": ("id", True),
            "mobile_verified": ("boolean", True),
            "num_orders_last_50days": ("count", True),
            "num_cancelled_orders_last_50days": ("count", True),
            "num_refund_orders_last_50days": ("count", True),
            "total_payment_last_50days": ("amount", True),
            "num_associated_customers": ("count", True),
            "first_order_datetime": ("datetime", False),
        },
    },
    "orders": {
        "key": ["country_code", "order_id"],
        "columns": {
            "country_code": ("country", True),
            "order_id": ("id", True),
            "collect_type": ("collect_type", True),
            "payment_method": ("payment_method", False),
            "order_value": ("amount", True),
            "num_items_ordered": ("item_count", False),
            "refund_value": ("amount", True),
            "order_date": ("date", True),
        },
    },
    "labels": {
        "key": ["country_code", "order_id"],
        "columns": {
            "country_code": ("country", True),
            "order_id": ("id", True),
            "customer_id": ("id", True),
            "is_fraud": ("label", True),
        },
    },
}
OPTIONAL_NOTE = {
    "payment_method": "filled with 'unknown' (Assumption 6)",
    "num_items_ordered": f"filled with the historical median {dp.ITEMS_FILL_VALUE} (Assumption 9)",
    "first_order_datetime": "allowed to be empty (Assumption 8)",
}


# ---------------------------------------------------------------------------
# Individual checks. Each returns a list of (issue dict, boolean row mask or None).
# ---------------------------------------------------------------------------
def _issue(dataset, column, check, severity, mask_or_count, examples="", detail=""):
    affected = int(mask_or_count.sum()) if isinstance(mask_or_count, pd.Series) else int(mask_or_count)
    return {"dataset": dataset, "column": column, "check": check, "severity": severity,
            "affected_records": affected, "examples": examples, "detail": detail}


def _examples(values):
    shown = pd.Series(values).dropna().astype(str).drop_duplicates().head(EXAMPLES)
    return "; ".join(shown)


def check_schema(frame, kind):
    """Missing required columns stop the batch; extra columns are only reported."""
    expected = SCHEMA[kind]["columns"]
    issues = []
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        issues.append((_issue(kind, ", ".join(missing), "missing required column", CRITICAL, len(frame),
                              detail="the file must have the same schema as the base data"), None))
    extra = [c for c in frame.columns if c not in expected and not str(c).startswith("_")]
    if extra:
        issues.append((_issue(kind, ", ".join(map(str, extra)), "unexpected column", WARNING, 0,
                              detail="ignored: not stored in the database"), None))
    return issues


def check_structure(frame, kind):
    """Rows with more fields than the header (set aside by read_batch_file)."""
    bad = frame.attrs.get("bad_lines", [])
    if not bad:
        return []
    examples = _examples([",".join(map(str, fields))[:60] for fields in bad])
    return [(_issue(kind, "(whole row)", "wrong number of fields", ERROR, len(bad), examples,
                    "row has more fields than the header; it is rejected"), None)]


def check_missing(frame, kind):
    """Missing values (empty, spaces only, 'nan', 'None') per column."""
    issues = []
    for column, (_, required) in SCHEMA[kind]["columns"].items():
        if column not in frame.columns:
            continue
        mask = dp.clean_text(frame[column]).isna()
        if not mask.any():
            continue
        if required:
            issues.append((_issue(kind, column, "missing required value", ERROR, mask,
                                  detail="record is rejected"), mask))
        else:
            issues.append((_issue(kind, column, "missing optional value", WARNING, mask,
                                  detail=OPTIONAL_NOTE.get(column, "allowed")), mask))
    return issues


def check_duplicates(frame, kind):
    """Exact duplicate rows, and repeated keys (the ETL keeps the last copy in the file)."""
    issues = []
    columns = [c for c in SCHEMA[kind]["columns"] if c in frame.columns]
    exact = frame.duplicated(subset=columns, keep="last")
    if exact.any():
        issues.append((_issue(kind, "(whole row)", "exact duplicate row", ERROR, exact,
                              detail="identical copy: the earlier copy is rejected as a duplicate, "
                                     "the record is still loaded once"), exact))
    key = SCHEMA[kind]["key"]
    if all(c in frame.columns for c in key):
        normalised = pd.DataFrame({"country": dp.clean_country(frame[key[0]]), "id": dp.clean_text(frame[key[1]])})
        repeated = normalised.duplicated(keep="last") & normalised.notna().all(axis=1) & ~exact
        if repeated.any():
            ids = (normalised["country"] + "-" + normalised["id"])[repeated]
            issues.append((_issue(kind, " + ".join(key), "repeated record key, different values", ERROR, repeated,
                                  _examples(ids), "earlier copy rejected, last copy in the file is kept"),
                           repeated))
    return issues


def _type_problems(values, value_type):
    """Return (mask of present-but-invalid values, check name, severity, detail) items for one column."""
    present = dp.clean_text(values).notna()
    if value_type == "country":
        unknown = present & ~dp.clean_country(values).isin(dp.COUNTRIES)
        return [(unknown, "unknown country code", WARNING, f"not one of {', '.join(dp.COUNTRIES)}; still loaded")]
    if value_type == "boolean":
        return [(present & dp.clean_bool(values).isna(), "invalid boolean", ERROR,
                 "expected True/False (or yes/no, 1/0)")]
    if value_type in ("count", "amount", "item_count"):
        number = dp.to_number(dp.clean_text(values))
        problems = [(present & number.isna(), "not a number", ERROR, "text or infinite value")]
        problems.append((number < 0, "negative value", ERROR, "must be 0 or more"))
        if value_type == "count":
            problems.append((number.notna() & (number % 1 != 0), "not a whole number", ERROR, "counts must be integers"))
        if value_type == "item_count":
            problems.append((number.notna() & (number >= 0) & ((number % 1 != 0) | (number < 1)),
                             "invalid item count", ERROR, "must be a whole number of at least 1"))
        return problems
    if value_type == "datetime":
        parsed = pd.to_datetime(dp.clean_text(values), format="ISO8601", errors="coerce")
        return [(present & parsed.isna(), "invalid date/time", ERROR, "cannot be read as a date and time")]
    if value_type == "date":
        return [(present & dp.parse_dates(values).isna(), "invalid date", ERROR,
                 "not in a known format (YYYY/MM/DD, YYYY-MM-DD, YYYY/Mon/DD)")]
    if value_type == "collect_type":
        known = dp.clean_text(values).str.lower().isin(dp.COLLECT_MAP)
        return [(present & ~known, "unknown collect_type", ERROR, "expected Delivery or Pickup")]
    if value_type == "payment_method":
        normalised = dp.clean_text(values).str.lower().replace(dp.PAYMENT_MAP)
        return [(present & ~normalised.isin(dp.PAYMENT_METHODS), "new payment method", WARNING,
                 "not seen in the historical data; still loaded")]
    if value_type == "label":
        number = dp.to_number(dp.clean_text(values))
        return [(present & ~number.isin([0, 1]), "invalid fraud label", ERROR, "is_fraud must be 0 or 1")]
    return []


def check_types(frame, kind):
    """Invalid data types and values, per column."""
    issues = []
    for column, (value_type, _) in SCHEMA[kind]["columns"].items():
        if column not in frame.columns:
            continue
        for mask, check, severity, detail in _type_problems(frame[column], value_type):
            mask = mask.fillna(False).astype(bool)
            if mask.any():
                issues.append((_issue(kind, column, check, severity, mask, _examples(frame.loc[mask, column]),
                                      detail), mask))
    return issues


CHECKS = (check_schema, check_structure, check_missing, check_duplicates, check_types)


# ---------------------------------------------------------------------------
# Whole batch
# ---------------------------------------------------------------------------
def inspect_dataset(frame, kind):
    """Run every check on one raw file. Returns (issues table, per-record summary counts)."""
    issues, error_rows, issue_rows = [], pd.Series(False, index=frame.index), pd.Series(False, index=frame.index)
    for check in CHECKS:
        for issue, mask in check(frame, kind):
            issues.append(issue)
            if mask is not None:
                issue_rows |= mask
                if issue["severity"] == ERROR:
                    error_rows |= mask
    bad_lines = len(frame.attrs.get("bad_lines", []))
    counts = {"dataset": kind,
              "input_records": len(frame) + bad_lines,
              "records_with_issues": int(issue_rows.sum()) + bad_lines,
              "records_with_errors": int(error_rows.sum()) + bad_lines}
    return issues, counts


def inspect_batch(raw):
    """Check the 3 raw files of one batch (DataFrames from etl_pipeline.extract)."""
    issues, counts = [], []
    for kind in FILE_KINDS:
        found, summary = inspect_dataset(raw[kind], kind)
        issues += found
        counts.append(summary)
    columns = ["dataset", "column", "check", "severity", "affected_records", "examples", "detail"]
    return {"issues": pd.DataFrame(issues, columns=columns), "counts": pd.DataFrame(counts)}


def inspect_files(folder, batch_name):
    """Used when a batch could not be read: find which file is unreadable or incomplete."""
    issues = []
    for kind in FILE_KINDS:
        path = Path(folder) / f"{kind}_{batch_name}.csv"
        if not path.is_file():
            issues.append(_issue(kind, "(file)", "file not found or not downloaded", CRITICAL, 0, path.name))
            continue
        try:
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
        except pd.errors.EmptyDataError:
            issues.append(_issue(kind, "(file)", "empty file", CRITICAL, 0, path.name, "no header row"))
            continue
        except (UnicodeDecodeError, pd.errors.ParserError):
            issues.append(_issue(kind, "(file)", "unreadable file", CRITICAL, 0, path.name, "not a UTF-8 CSV file"))
            continue
        names = [str(c).strip().lower() for c in header]
        missing = [c for c in SCHEMA[kind]["columns"] if c not in names]
        if missing:
            issues.append(_issue(kind, ", ".join(missing), "missing required column", CRITICAL, 0, path.name,
                                 "the file must have the same schema as the base data"))
    return issues


def processing_summary(quality, rows_read, rows_cleaned, rejected, loaded=None):
    """Per dataset: input records, records with issues, rejected and successfully processed (#85 AC3).

    rejected: the ETL's rejected rows (dicts with 'source_table' and 'reason').
    loaded:   the ETL load counts (inserted + updated per table), or None for a dry run."""
    rejected_by_table = pd.Series([r["source_table"] for r in rejected], dtype=object).value_counts()
    rows = []
    for _, row in quality["counts"].iterrows():
        kind = row["dataset"]
        table = TABLE_FOR_KIND[kind]
        prefix = "labels" if kind == "labels" else kind
        processed = (loaded[f"{prefix}_inserted"] + loaded[f"{prefix}_updated"]) if loaded else rows_cleaned[table]
        reasons = pd.Series([r["reason"] for r in rejected if r["source_table"] == table], dtype=object)
        rows.append({**row.to_dict(),
                     "records_rejected": int(rejected_by_table.get(table, 0)),
                     "records_processed": int(processed),
                     "processed_pct": round(100 * processed / row["input_records"], 2) if row["input_records"] else 0.0,
                     "rejection_reasons": "; ".join(f"{k}: {v}" for k, v in reasons.value_counts().items())})
    return pd.DataFrame(rows)


def batch_status(issues):
    severities = set(issues["severity"]) if len(issues) else set()
    if CRITICAL in severities:
        return "critical issue: batch stopped, nothing loaded"
    if ERROR in severities:
        return "issues found: affected records rejected"
    if WARNING in severities:
        return "passed with warnings"
    return "passed"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _markdown_table(frame):
    if frame.empty:
        return "_none_"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in columns) + " |")
    return "\n".join(lines)


def write_report(batch_name, issues, summary, status, report_dir=DEFAULT_REPORT_DIR, dry_run=False, error=None):
    """Write the batch report folder and append to the history file (#85 AC2, AC5, AC6)."""
    generated = datetime.now()
    folder = Path(report_dir) / batch_name
    folder.mkdir(parents=True, exist_ok=True)
    order = {CRITICAL: 0, ERROR: 1, WARNING: 2}
    issues = issues.sort_values(["severity", "dataset", "affected_records"], key=lambda s: s.map(order) if s.name == "severity" else s,
                                ascending=[True, True, False]) if len(issues) else issues
    issues.to_csv(folder / "quality_issues.csv", index=False)
    summary.to_csv(folder / "processing_summary.csv", index=False)

    lines = [f"# Data quality report: batch {batch_name}", "",
             f"- Generated: {generated:%Y-%m-%d %H:%M:%S}",
             f"- Status: **{status}**",
             f"- Mode: {'dry run (nothing loaded)' if dry_run else 'loaded to RDS' if error is None else 'not loaded'}"]
    if error is not None:
        lines.append(f"- Pipeline stopped with: `{str(error)[:300]}`")
    lines += ["", "## Processing summary", "", _markdown_table(summary),
              "", "## Detected issues", "", _markdown_table(issues),
              "", "Severity: critical = batch stopped before loading; error = record rejected into "
                  "rejected_records; warning = record loaded but unusual.", ""]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")

    history = summary.assign(batch_name=batch_name, generated_at=f"{generated:%Y-%m-%d %H:%M:%S}", status=status,
                             dry_run=dry_run, critical_issues=int((issues["severity"] == CRITICAL).sum()) if len(issues) else 0,
                             error_issues=int((issues["severity"] == ERROR).sum()) if len(issues) else 0,
                             warning_issues=int((issues["severity"] == WARNING).sum()) if len(issues) else 0)
    first = ["generated_at", "batch_name", "status", "dry_run"]
    history = history[first + [c for c in history.columns if c not in first]]
    history_path = Path(report_dir) / "history.csv"
    history.to_csv(history_path, mode="a", header=not history_path.exists(), index=False)
    return folder


def report_batch(batch_name, quality, rows_read, rows_cleaned, rejected, loaded=None,
                 report_dir=DEFAULT_REPORT_DIR, dry_run=False):
    """Normal end of a batch: issues found before cleaning + what the ETL actually did."""
    summary = processing_summary(quality, rows_read, rows_cleaned, rejected, loaded)
    status = batch_status(quality["issues"])
    folder = write_report(batch_name, quality["issues"], summary, status, report_dir, dry_run)
    return {"status": status, "report_dir": str(folder), "issues": len(quality["issues"]),
            "records_with_issues": summary.set_index("dataset")["records_with_issues"].to_dict()}


def report_critical(batch_name, error, raw_folder, quality=None, report_dir=DEFAULT_REPORT_DIR, dry_run=False):
    """The batch stopped (unreadable file, missing columns, failed validation or load): still write a report."""
    issues = quality["issues"].copy() if quality is not None else pd.DataFrame(
        columns=["dataset", "column", "check", "severity", "affected_records", "examples", "detail"])
    file_issues = pd.DataFrame(inspect_files(raw_folder, batch_name))
    if quality is None and not file_issues.empty:
        issues = file_issues
    if not (issues["severity"] == CRITICAL).any():
        issues = pd.concat([issues, pd.DataFrame([_issue("(batch)", "(pipeline)", "pipeline stopped", CRITICAL, 0,
                                                         detail=str(error)[:200])])], ignore_index=True)
    if quality is not None:
        summary = quality["counts"].assign(records_rejected=0, records_processed=0, processed_pct=0.0,
                                           rejection_reasons="batch stopped before loading")
    else:
        summary = pd.DataFrame({"dataset": list(FILE_KINDS), "input_records": 0, "records_with_issues": 0,
                                "records_with_errors": 0, "records_rejected": 0, "records_processed": 0,
                                "processed_pct": 0.0, "rejection_reasons": "batch stopped before loading"})
    status = batch_status(issues)
    folder = write_report(batch_name, issues, summary, status, report_dir, dry_run, error=error)
    print(f"DATA QUALITY: {status}. Report: {folder}", file=sys.stderr)
    return {"status": status, "report_dir": str(folder), "issues": len(issues)}


def console_summary(quality):
    """A few lines for the ETL log."""
    issues = quality["issues"]
    lines = [f"Data quality: {batch_status(issues)}"]
    for _, row in quality["counts"].iterrows():
        lines.append(f"  {row['dataset']}: {row['input_records']} records, {row['records_with_issues']} with issues "
                     f"({row['records_with_errors']} to be rejected)")
    for _, issue in issues.head(8).iterrows():
        lines.append(f"  [{issue['severity']}] {issue['dataset']}.{issue['column']}: {issue['check']} "
                     f"({issue['affected_records']})")
    if len(issues) > 8:
        lines.append(f"  ... {len(issues) - 8} more in the report")
    return "\n".join(lines)


def main():
    """Check a batch of files without running the ETL (nothing is loaded)."""
    parser = argparse.ArgumentParser(description="Check the data quality of one batch of SP-buy CSV files")
    parser.add_argument("--data-dir", required=True, help="Folder with customers_X.csv, orders_X.csv, labels_X.csv")
    parser.add_argument("--postfix", required=True, help="File postfix, e.g. XXX")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()
    import etl_pipeline as etl                      # reuse the ETL's forgiving CSV reader
    folder = Path(args.data_dir)
    critical = inspect_files(folder, args.postfix)
    if critical:
        result = report_critical(args.postfix, "file check failed", folder, report_dir=args.report_dir, dry_run=True)
        sys.exit(f"Critical issue found; see {result['report_dir']}")
    raw = {kind: etl.read_batch_file(folder / f"{kind}_{args.postfix}.csv", kind) for kind in FILE_KINDS}
    quality = inspect_batch(raw)
    print(console_summary(quality))
    cleaned, rejected = etl.transform(raw)
    result = report_batch(args.postfix, quality, {k: len(v) for k, v in raw.items()},
                          {k: len(v) for k, v in cleaned.items()}, rejected, report_dir=args.report_dir, dry_run=True)
    print(f"Report: {result['report_dir']}")


if __name__ == "__main__":
    main()
