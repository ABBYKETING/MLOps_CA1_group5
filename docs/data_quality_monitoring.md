# Data Quality Monitoring for the ETL Pipeline (PBI 11)

User story #85 (Yue Liang): *As an AIML engineer, I want the ETL pipeline to automatically monitor
and report data quality issues in incoming customer, order and fraud label data so that I can
identify issues early and prevent unreliable data from affecting fraud detection models.*

| Task | Where |
|---|---|
| #86 Identify the required checks | Section 1 below; `SCHEMA` in `src/data_quality.py` |
| #87 Reusable validation functions | `check_schema`, `check_structure`, `check_missing`, `check_duplicates`, `check_types` |
| #88 Record issues and statistics | `inspect_dataset`, `inspect_batch`, `processing_summary` |
| #89 Integrate into the PBI 4 ETL | `run_etl()` in `src/etl_pipeline.py` (about 25 added lines; ETL logic unchanged) |
| #90 Structured reports | `write_report`: `quality_issues.csv`, `processing_summary.csv`, `report.md`, `history.csv` |
| #91 Test and document | `tests/test_data_quality.py`, Sections 4 and 5 below |

## 1. Checks (Task #86)

The checks run on the files **as received, before cleaning** (business rule 4), and mirror the
team's cleaning rules in `src/data_processing.py` (PBI 1). A record flagged as an *error* is the
same record the ETL later rejects into `rejected_records`.

| Check | Applies to | Severity | What happens to the data |
|---|---|---|---|
| File not found / not downloadable, empty, not a UTF-8 CSV | each file | critical | batch stops, nothing loaded |
| Missing required column | each file | critical | batch stops, nothing loaded |
| Unexpected extra column | each file | warning | ignored (not stored) |
| Wrong number of fields in a row | each file | error | row rejected |
| Missing required value | IDs, country, mobile_verified, counts, amounts, collect_type, order_date, is_fraud | error | record rejected |
| Missing optional value | payment_method, num_items_ordered, first_order_datetime | warning | filled or allowed (Assumptions 6, 8, 9) |
| Exact duplicate row | each file | error | earlier copy rejected, record loaded once |
| Repeated record key with different values | (country_code, customer_id / order_id) | error | earlier copy rejected, last copy kept (Assumption 11) |
| Not a number, negative, not a whole number | counts and amounts | error | record rejected |
| Invalid item count | num_items_ordered < 1 or not whole | error | record rejected |
| Invalid boolean | mobile_verified | error | record rejected |
| Invalid date / date-time | order_date, first_order_datetime | error | record rejected |
| Unknown collect_type | not Delivery/Pickup | error | record rejected |
| Invalid fraud label | is_fraud not 0/1 | error | record rejected |
| Unknown country code | not BD, MY, PH, PK, TH | warning | still loaded |
| New payment method | not seen in the historical data | warning | still loaded |

Severity meaning: **critical** = the batch cannot be processed and stops before the database is
touched (#85 AC4); **error** = the record is rejected with a reason; **warning** = the record is
loaded, but something unusual should be reviewed.

## 2. How it runs (Task #89)

Nothing extra to run: every ETL command now checks the batch and writes a report.

```powershell
python src/etl_pipeline.py --customers <url> --orders <url> --labels <url>
```

Inside `run_etl()`:

1. `extract()` downloads and reads the 3 files. If that fails (unreadable file, missing column,
   download error), `report_critical()` inspects the files, writes a critical report and the ETL stops.
2. `inspect_batch()` runs all checks on the raw files and prints a short summary in the ETL log.
3. `transform()` cleans the data. If the team's validation of the cleaned data fails, a critical
   report is written and the ETL stops before loading.
4. `load()` writes to RDS in one transaction (a failure rolls back and writes a critical report).
5. `report_batch()` combines the issues with what the ETL actually did and writes the report.
   The report location is also stored in `outputs/etl/<batch>_summary.json` (`data_quality`).

The automated retraining pipeline (`src/automated_pipeline.py`) calls `run_etl()`, so the same
checks and report run in every automated batch as well; the report location is part of the ETL
section of its `summary.json`.

These checks look at the **incoming files before loading**. They complement
`src/check_data_integrity.py` (PBI 4), which checks the **database after loading** (no duplicate
keys, every label has its order and customer, valid values).

Files can also be checked on their own, without running the ETL or touching the database:

```powershell
python src/data_quality.py --data-dir <folder> --postfix XXX
```

## 3. Reports (Task #90)

One folder per batch (git-ignored), plus a history file for comparing batches (#85 AC6):

| File | Contents |
|---|---|
| `outputs/data_quality/<batch>/quality_issues.csv` | dataset, column, check, severity, affected records, example values, detail (#85 AC2) |
| `outputs/data_quality/<batch>/processing_summary.csv` | per dataset: input records, records with issues, records with errors, records rejected, records successfully processed, % processed, rejection reasons (#85 AC3) |
| `outputs/data_quality/<batch>/report.md` | status, both tables, and the error message if the batch stopped |
| `outputs/data_quality/history.csv` | one row per dataset per run: time, batch, status, the counts above and the number of critical / error / warning issues |

"Records successfully processed" is the number of rows inserted or updated in RDS for a loaded
batch (rows that pass cleaning for a `--dry-run`).

Batch status: `passed`, `passed with warnings`, `issues found: affected records rejected`, or
`critical issue: batch stopped, nothing loaded`.

## 4. Handling procedures

| Status / issue | What to do |
|---|---|
| passed / passed with warnings | No action. Review warnings when they grow between batches in `history.csv` (e.g. many new payment methods or unknown country codes may mean the source system changed). |
| issues found | The batch is loaded without the affected records. Check `processing_summary.csv` for the rejected share; the original rows are in `rejected_records` (`SELECT * FROM rejected_records WHERE batch_name = '<batch>'`). If the share is unusually high, ask the data provider for corrected files, then re-run the same batch (the re-run replaces its rejected rows and "latest wins" updates records). |
| critical: missing column / unreadable file | Nothing was loaded. Open `report.md` to see which file and which columns. Ask for corrected files (same schema as the base data), then re-run the same command. |
| critical: file not found / download failed | Check the URL or path, then re-run. |
| critical: pipeline stopped (validation or database error) | Nothing was loaded (the load is one transaction). Read the error in `report.md`, fix the cause, re-run. |

## 5. Test results (Task #91)

`python -m unittest tests.test_data_quality -v` uses sample batches with valid records, missing
values, duplicates, invalid types and schema errors. Results on 28 Nov 2026: **13 tests, all passed**
(the database test was run against a separate practice database, never `spbuy`). The existing
`tests.test_etl_pipeline` suite (14 tests, including its database test) also still passes, so the
ETL behaviour is unchanged.

| Test | Sample data | Result |
|---|---|---|
| valid batch | good customers, orders, labels | status `passed`, no issues |
| missing values | empty order_value, payment_method, num_items_ordered, first_order_datetime | required missing = error (rejected); optional missing = warning (loaded) |
| exact duplicates | identical order row twice | error; matches the ETL's rejected rows |
| repeated key | `pk,o2` after `PK,o2` (same key after normalisation) | error, example `PK-o2` |
| invalid types | 'maybe', 1.5, 'abc', 'someday', 'drone', -5, 0 items, '03-05-2023', 'XX', 'crypto', is_fraud 2 | each detected with the right severity |
| schema | extra column `note`; a row with 9 fields | warning; error |
| errors = rejected rows | mixed bad rows | records flagged as errors equal the ETL's rejected rows per dataset |
| report generated | batch with issues, run through `run_etl` | 3 report files; counts input 4 / issues 2 / errors 1 / rejected 1 / processed 3 |
| repeatable | two batches | separate folders, `history.csv` has 6 rows (3 datasets x 2 batches) |
| missing column | labels file without customer_id, is_fraud | critical report names the file and columns; ETL stops, no summary written |
| unreadable file | UTF-16 orders file | critical `unreadable file` for orders |
| failed validation | cleaned-data validation forced to fail | critical report still written |
| database (opt-in) | orphan fraud label loaded into practice RDS | processed 2 / rejected 1 with reason `order not found` |

Performance: checking a full base-size batch (2.85 million rows across the 3 files) takes about
17 seconds; a tutor demo batch takes well under a second.

### Example report (sample batch with mixed problems, dry run)

| dataset | input_records | records_with_issues | records_with_errors | records_rejected | records_processed | processed_pct | rejection_reasons |
|---|---|---|---|---|---|---|---|
| customers | 4 | 2 | 1 | 1 | 3 | 75.0 | unrecognised mobile_verified: 1 |
| orders | 8 | 6 | 5 | 5 | 3 | 37.5 | duplicate order (kept last in file): 2; unrecognised collect_type: 1; negative value: 1; row has 9 fields, expected 8: 1 |
| labels | 3 | 1 | 1 | 1 | 2 | 66.67 | is_fraud not 0 or 1: 1 |

Records with errors equal the records the ETL rejected for every dataset, which confirms the
checks and the cleaning rules agree.
