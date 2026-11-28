"""Prefect flow: geographic and account behaviour fraud analytics (PBI 8, Yue Liang).

Every run reads the LATEST labelled orders from the team's AWS RDS database and
refreshes the three analyses in src/analytics_geo_account_yueliang.py (PBI 7, Task #64).

Tasks, in the order they run:

    1. count_orders_in_rds        one grouped pass over all labelled orders in RDS (retries twice)
    2. load_review_list           confirmed-fraud accounts with many associated accounts (retries twice)
    3. load_batch_log             which ETL batches the data includes (retries twice)
    4. check_data_quality         stop the flow with a clear message if the data cannot be trusted
    5. analyse_geography          Q1: fraud rate, volume, value and recent trend per country
       analyse_associated_accounts  Q2: fraud rate by number of associated accounts
       analyse_cancellations      Q3: fraud rate by cancellation rate, and the combined signal
                                  (5a-5c are independent, so they are submitted to run concurrently)
    6. save_reports               CSV tables, PNG charts, summary.md and report.html
    7. publish_results            findings in the flow logs and as Prefect artifacts

How users see the results:
    - Prefect UI -> Runs -> the flow run -> Artifacts: the findings summary and one table per question
    - Prefect UI -> the flow run -> Logs: the same findings as text, plus timings
    - outputs/analytics_yueliang/report.html: one page with all charts and tables, refreshed every run

Usage (from the repository root, with DB_HOST, DB_USER and DB_PASSWORD set):

    python -m prefect server start                         # terminal 1, UI at http://127.0.0.1:4200
    $env:PREFECT_API_URL = "http://127.0.0.1:4200/api"     # terminal 2 (plus the DB_* variables)
    python flows/geo_account_risk_flow_yueliang.py         # run once now
    python flows/geo_account_risk_flow_yueliang.py --serve # keep running; refresh daily at 07:30 (Singapore)

As a separate flow service (PBI 9 Task 4, #75): prefect_server/flow_service/ runs this file
with --serve --webserver in its own Docker container on AWS Elastic Beanstalk. It registers
the deployment "daily-geo-account-fraud-yueliang" on the AWS Prefect server, runs it on the
schedule, and also runs it whenever someone clicks Run in the Prefect UI. --webserver opens a
small health endpoint (/health on PREFECT_RUNNER_SERVER_PORT) so Elastic Beanstalk can tell
that the service is alive.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.cache_policies import NO_CACHE
from prefect.schedules import Cron

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analytics_geo_account_yueliang as analytics  # noqa: E402

DAILY_SCHEDULE = "30 7 * * *"      # 07:30 every day, before analysts start reviewing (only with --serve)
SCHEDULE_TIMEZONE = "Asia/Singapore"
DEMO_LIMIT_SECONDS = 180           # the interview demo must finish within 3 minutes
DB_TASK = {"retries": 2, "retry_delay_seconds": 10, "cache_policy": NO_CACHE}
NO_REUSE = {"cache_policy": NO_CACHE}   # always recompute from the latest data, never reuse old results


@task(**DB_TASK)
def count_orders_in_rds():
    """One grouped pass over every labelled order; RDS returns only summary rows."""
    started = time.perf_counter()
    counts = analytics.count_orders_in_db()
    get_run_logger().info("RDS returned %s summary groups covering %s labelled orders in %.1f s",
                          f"{len(counts):,}", f"{counts['orders'].sum():,}", time.perf_counter() - started)
    return counts


@task(**DB_TASK)
def load_review_list():
    """Confirmed-fraud accounts with many associated accounts, for linked-account review."""
    review_list = analytics.query_associated_review_list()
    get_run_logger().info("Review list: %d accounts with at least %d associated accounts",
                          len(review_list), analytics.REVIEW_LIST_MIN_ASSOCIATED)
    return review_list


@task(**DB_TASK)
def load_batch_log():
    """ETL batches recorded in RDS, so the report states which data it covers."""
    batches = analytics.load_batches()
    get_run_logger().info("ETL batches in RDS: %s",
                          ", ".join(f"{b} ({s})" for b, s in zip(batches["batch_name"], batches["status"])))
    return batches


@task(**NO_REUSE)
def check_data_quality(counts, batches):
    """Fail early with a clear reason instead of publishing misleading insights."""
    status = analytics.data_status(counts, batches)
    problems = []
    if status["orders"] == 0:
        problems.append("no labelled orders found in RDS")
    elif not 0 < status["frauds"] < status["orders"]:
        problems.append("the data must contain both fraud and non-fraud orders")
    if not (batches["status"] == "completed").any():
        problems.append("no completed ETL batch in etl_batches")
    if problems:
        raise ValueError("Data quality check failed: " + "; ".join(problems))
    issues = counts.loc[counts["cancel_segment"] == analytics.DATA_ISSUE, "orders"].sum()
    logger = get_run_logger()
    logger.info("Data quality check passed: %s orders (%s fraud), order dates %s to %s",
                f"{status['orders']:,}", f"{status['frauds']:,}",
                status["first_order_date"], status["last_order_date"])
    if issues:
        logger.warning("%s orders belong to customers with more cancellations than orders", f"{issues:,}")
    return status


@task(**NO_REUSE)
def analyse_geography(counts):
    """Q1: which countries have the highest fraud rates and fraud volumes."""
    return analytics.analyse_countries(analytics.country_weeks(counts))


@task(**NO_REUSE)
def analyse_associated_accounts(counts, review_list):
    """Q2: fraud rate by number of associated customer accounts."""
    return analytics.analyse_associated(analytics.associated_groups(counts), review_list)


@task(**NO_REUSE)
def analyse_cancellations(counts):
    """Q3: fraud rate by cancellation rate, and cancellation combined with refunds."""
    return analytics.analyse_cancellation(analytics.cancellation_groups(counts))


@task(**NO_REUSE)
def save_reports(status, q1, q2, q3, output_dir):
    """Write the CSV tables, PNG charts, summary.md and report.html."""
    result = analytics.save_outputs(status, q1, q2, q3, output_dir)
    get_run_logger().info("Saved %d tables and %d charts; open %s", len(result["tables"]),
                          len(result["charts"]), (Path(output_dir) / "report.html").resolve())
    return result


def _rows(table, columns):
    """A results table as plain rows for a Prefect table artifact."""
    shown = table[columns].copy()
    for column in shown.select_dtypes("number"):
        shown[column] = shown[column].round(2)
    return shown.astype(str).to_dict(orient="records")


def _md_table(table, columns):
    """A compact markdown table (column -> header) for the summary artifact."""
    lines = ["| " + " | ".join(columns.values()) + " |", "|" + "---|" * len(columns)]
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:,.0f}" if column == "fraud_order_value" else f"{value:,.2f}")
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _summary_with_tables(result, q1, q2, q3):
    """The findings, with each question's key table placed under its findings."""
    rate = {"orders": "Orders", "frauds": "Fraud orders", "fraud_rate_pct": "Fraud rate %",
            "ci_low_pct": "CI low %", "ci_high_pct": "CI high %", "lift": "Lift"}
    shares = {"share_of_orders_pct": "% of orders", "share_of_fraud_pct": "% of fraud"}
    tables = {
        "## Q2": _md_table(q1["countries"], {"country": "Country", **rate, **shares,
                                             "fraud_order_value": "Fraud value (SGD)",
                                             "recent_fraud_rate_pct": "Latest 4 weeks %",
                                             "change_pp": "Change (pp)"}),
        "## Q3": _md_table(q2["bands"], {"segment": "Associated accounts", **rate, **shares}),
        "## Caveat": (_md_table(q3["bands"], {"segment": "Cancellation rate", **rate, **shares})
                      + "\n\n" + _md_table(q3["combined"], {"signal": "Combined signal", **rate, **shares})),
    }
    text = result["summary"]
    for next_heading, markdown in tables.items():      # insert each table before the next section
        text = text.replace(next_heading, f"{markdown}\n\n{next_heading}", 1)
    return text


@task(**NO_REUSE)
def publish_results(result, q1, q2, q3, seconds_so_far):
    """Findings in the logs, plus a summary and one table per question as Prefect artifacts."""
    logger = get_run_logger()
    for line in result["summary"].splitlines():
        if line.strip():
            logger.info(line)
    report = (Path(result["output_dir"]) / "report.html").resolve()
    create_markdown_artifact(
        key="geo-account-fraud-summary",
        markdown=(f"{_summary_with_tables(result, q1, q2, q3)}\n\n## Full report\n\nCharts and tables: `{report}`\n\n"
                  f"Run time before publishing: {seconds_so_far:.1f} s\n"),
        description="Geographic and account behaviour fraud analytics: findings")
    rate = ["orders", "frauds", "fraud_rate_pct", "ci_low_pct", "ci_high_pct", "lift"]
    create_table_artifact(
        key="q1-fraud-by-country",
        table=_rows(q1["countries"], ["country"] + rate + ["share_of_fraud_pct", "fraud_order_value",
                                                           "recent_fraud_rate_pct", "change_pp"]),
        description="Q1: fraud rate, volume, value and latest-weeks change by country")
    create_table_artifact(
        key="q2-fraud-by-associated-accounts",
        table=_rows(q2["bands"], ["segment"] + rate + ["share_of_orders_pct", "share_of_fraud_pct"]),
        description="Q2: fraud rate by number of associated customer accounts")
    create_table_artifact(
        key="q2-linked-account-review-list",
        table=_rows(q2["review_list"], ["country", "customer_id", "num_associated_customers", "orders", "frauds"]),
        description="Q2: confirmed-fraud accounts with the most associated accounts (review their links)")
    create_table_artifact(
        key="q3-fraud-by-cancellation-rate",
        table=_rows(q3["bands"], ["segment"] + rate + ["share_of_orders_pct", "share_of_fraud_pct"]),
        description="Q3: fraud rate by cancellation rate in the last 50 days")
    create_table_artifact(
        key="q3-combined-cancellation-refund-signal",
        table=_rows(q3["combined"], ["signal"] + rate + ["share_of_orders_pct", "share_of_fraud_pct"]),
        description="Q3: high cancellation and/or high refund rate")
    return str(report)


@flow(name="geo-account-fraud-analytics-yueliang", log_prints=True)
def geo_account_fraud_flow(output_dir: str = str(analytics.DEFAULT_OUTPUT_DIR)):
    """Refresh the geographic and account behaviour fraud analytics from the latest RDS data."""
    started = time.perf_counter()
    counts = count_orders_in_rds()
    review_list = load_review_list()
    batches = load_batch_log()
    status = check_data_quality(counts, batches)

    q1_future = analyse_geography.submit(counts)
    q2_future = analyse_associated_accounts.submit(counts, review_list)
    q3_future = analyse_cancellations.submit(counts)
    q1, q2, q3 = q1_future.result(), q2_future.result(), q3_future.result()

    result = save_reports(status, q1, q2, q3, output_dir)
    report = publish_results(result, q1, q2, q3, time.perf_counter() - started)

    seconds = time.perf_counter() - started
    logger = get_run_logger()
    if seconds > DEMO_LIMIT_SECONDS:
        logger.warning("Flow took %.1f s, over the %d s demo limit", seconds, DEMO_LIMIT_SECONDS)
    else:
        logger.info("Flow finished in %.1f s (demo limit %d s)", seconds, DEMO_LIMIT_SECONDS)
    return report


def main():
    parser = argparse.ArgumentParser(description="Run the geographic and account behaviour fraud analytics flow")
    parser.add_argument("--serve", action="store_true",
                        help=f"keep running and refresh on a schedule ({DAILY_SCHEDULE} {SCHEDULE_TIMEZONE}); "
                             "stop with Ctrl+C")
    parser.add_argument("--webserver", action="store_true",
                        help="with --serve: also open a /health endpoint (used by the AWS flow service)")
    parser.add_argument("--output-dir", default=str(analytics.DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.serve:
        geo_account_fraud_flow.serve(name="daily-geo-account-fraud-yueliang",
                                     schedule=Cron(DAILY_SCHEDULE, timezone=SCHEDULE_TIMEZONE),
                                     parameters={"output_dir": args.output_dir},
                                     tags=["yueliang", "analytics"], webserver=args.webserver)
    else:
        geo_account_fraud_flow(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
