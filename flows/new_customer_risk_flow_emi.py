"""Prefect flow: automated new customer risk analytics (PBI 8, Task #68, Emi).

Every run reads the LATEST data from the team's AWS RDS database and refreshes
the three analyses from src/analytics_new_customer_emi.py (Task #62).

Tasks, in the order they run:

    1. load_latest_orders           read labelled orders + customer details from RDS (retries twice)
    2. segment_customers            add account age, New/Established and high-value labels
    3. analyse_account_age          Q1: first 30 days vs established customers
    4. analyse_mobile_verification  Q2: mobile verification among new customers
    5. analyse_high_value_orders    Q3: high-value orders from new vs established customers
    6. save_reports                 write CSV tables, PNG charts and summary.md
    7. publish_results              log the findings and publish them as Prefect artifacts

How users see the results:
    - Prefect UI -> Runs -> the flow run -> Artifacts: the written summary and one table per question
    - Prefect UI -> the flow run -> Logs: the same findings as text
    - outputs/analytics_emi/: the charts (PNG) and tables (CSV), refreshed on every run

Usage (from the repository root, with DB_HOST, DB_USER and DB_PASSWORD set):

    prefect server start                                  # terminal 1, UI at http://127.0.0.1:4200
    $env:PREFECT_API_URL = "http://127.0.0.1:4200/api"    # terminal 2
    python flows/new_customer_risk_flow_emi.py            # run once now
    python flows/new_customer_risk_flow_emi.py --serve    # keep running, refresh every day at 08:00
"""

import argparse
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.cache_policies import NO_CACHE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analytics_new_customer_emi as analytics  # noqa: E402

DAILY_SCHEDULE = "0 8 * * *"     # every day at 08:00 (only with --serve)
# Never reuse old results: every run must recompute from the latest RDS data
FRESH = {"cache_policy": NO_CACHE}


@task(retries=2, retry_delay_seconds=10, **FRESH)
def load_latest_orders():
    """Read the latest labelled orders from RDS (retried if the connection drops)."""
    orders = analytics.load_orders()
    get_run_logger().info("Loaded %s labelled orders from RDS (order dates %s to %s)", f"{len(orders):,}",
                          orders["order_date"].min().date(), orders["order_date"].max().date())
    return orders


@task(**FRESH)
def segment_customers(orders):
    """Label each order New / Established and High / Normal value."""
    segmented = analytics.add_customer_segments(orders)
    counts = segmented["customer_segment"].value_counts().to_dict()
    get_run_logger().info("Customer segments: %s", counts)
    return segmented


@task(**FRESH)
def analyse_account_age(df):
    """Q1: fraud rate for customers in their first 30 days vs established customers."""
    return analytics.q1_account_age(df)


@task(**FRESH)
def analyse_mobile_verification(df):
    """Q2: fraud rate by mobile verification among new and established customers."""
    return analytics.q2_mobile_verification(df)


@task(**FRESH)
def analyse_high_value_orders(df):
    """Q3: fraud rate of high-value orders from new vs established customers."""
    return analytics.q3_high_value_orders(df)


@task(**FRESH)
def save_reports(df, q1, q2, q3, output_dir):
    """Write tables (CSV), charts (PNG) and summary.md to the output folder."""
    files = analytics.save_outputs(df, q1, q2, q3, output_dir)
    get_run_logger().info("Saved %d files to %s", len(files), Path(output_dir).resolve())
    return {name: str(path) for name, path in files.items()}


def _table_rows(table):
    """A results table as plain rows for a Prefect table artifact."""
    return table.round(2).astype(object).where(table.notna(), None).astype(str).to_dict(orient="records")


@task(**FRESH)
def publish_results(df, q1, q2, q3, files):
    """Log the findings and publish them as artifacts in the Prefect UI."""
    summary = analytics.summary_text(df, q1, q2, q3)
    logger = get_run_logger()
    for line in summary.splitlines():
        if line.strip():
            logger.info(line)
    chart_list = "\n".join(f"- `{files[key]}`" for key in ("q1_chart", "q2_chart", "q3_chart"))
    create_markdown_artifact(
        key="new-customer-risk-summary",
        markdown=f"{summary}\n\n## Charts\n{chart_list}\n",
        description="New customer risk analytics: findings and decisions")
    create_table_artifact(key="q1-fraud-by-account-age", table=_table_rows(q1["by_age_band"]),
                          description="Q1: fraud rate by customer account age")
    create_table_artifact(key="q2-fraud-by-mobile-verification", table=_table_rows(q2["table"]),
                          description="Q2: fraud rate by mobile verification")
    create_table_artifact(key="q3-fraud-by-order-value", table=_table_rows(q3["table"]),
                          description="Q3: fraud rate of high-value vs normal orders")
    return summary


@flow(name="new-customer-risk-analytics-emi", log_prints=True)
def new_customer_risk_flow(output_dir: str = str(analytics.DEFAULT_OUTPUT_DIR)):
    """Refresh the three new customer risk analyses from the latest RDS data."""
    orders = load_latest_orders()
    df = segment_customers(orders)
    q1 = analyse_account_age(df)
    q2 = analyse_mobile_verification(df)
    q3 = analyse_high_value_orders(df)
    files = save_reports(df, q1, q2, q3, output_dir)
    return publish_results(df, q1, q2, q3, files)


def main():
    parser = argparse.ArgumentParser(description="Run the new customer risk analytics flow (Task #68)")
    parser.add_argument("--serve", action="store_true",
                        help=f"Keep running and refresh on a schedule ({DAILY_SCHEDULE}); stop with Ctrl+C")
    args = parser.parse_args()
    if args.serve:
        new_customer_risk_flow.serve(name="daily-new-customer-risk-emi", cron=DAILY_SCHEDULE)
    else:
        new_customer_risk_flow()


if __name__ == "__main__":
    main()
