"""Prefect flow: automated new customer risk analytics (PBI 8, Task #68, Emi).

Every run reads the LATEST data from the team's AWS RDS database and refreshes
the three analyses from src/analytics_new_customer_emi.py (Task #62). MySQL does
the counting, so only a few dozen rows travel from RDS and a run takes seconds.

Tasks, in the order they run:

    1. load_high_value_cutoffs      ask RDS for the top-25% order value per country (retries twice)
    2. count_orders_in_rds          let RDS count orders and frauds per account-age band,
                                    mobile verification and value group (retries twice)
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
def load_high_value_cutoffs():
    """The top-25% order value per country, computed by RDS (retried if the connection drops)."""
    cutoffs = analytics.high_value_cutoffs()
    get_run_logger().info("High-value cut-offs: %s",
                          dict(zip(cutoffs["country_code"], cutoffs["high_value_cutoff"].round(2))))
    return cutoffs


@task(retries=2, retry_delay_seconds=10, **FRESH)
def count_orders_in_rds(cutoffs):
    """Orders and frauds per age band / verification / value group, counted by RDS."""
    counts, overview = analytics.count_orders_in_db(cutoffs)
    segments = counts.groupby("customer_segment")["orders"].sum().to_dict()
    get_run_logger().info("Counted %s labelled orders in RDS (order dates %s to %s); segments: %s",
                          f"{overview['orders']:,}", overview["first_order_date"].date(),
                          overview["last_order_date"].date(), segments)
    return counts, overview


@task(**FRESH)
def analyse_account_age(counts):
    """Q1: fraud rate for customers in their first 30 days vs established customers."""
    return analytics.q1_account_age(counts)


@task(**FRESH)
def analyse_mobile_verification(counts):
    """Q2: fraud rate by mobile verification among new and established customers."""
    return analytics.q2_mobile_verification(counts)


@task(**FRESH)
def analyse_high_value_orders(counts, cutoffs):
    """Q3: fraud rate of high-value orders from new vs established customers."""
    return analytics.q3_high_value_orders(counts, cutoffs)


@task(**FRESH)
def save_reports(counts, overview, q1, q2, q3, output_dir):
    """Write tables (CSV), charts (PNG) and summary.md to the output folder."""
    files = analytics.save_outputs(counts, overview, q1, q2, q3, output_dir)
    get_run_logger().info("Saved %d files to %s", len(files), Path(output_dir).resolve())
    return {name: str(path) for name, path in files.items()}


def _table_rows(table):
    """A results table as plain rows for a Prefect table artifact."""
    return table.round(2).astype(object).where(table.notna(), None).astype(str).to_dict(orient="records")


@task(**FRESH)
def publish_results(counts, overview, q1, q2, q3, files):
    """Log the findings and publish them as artifacts in the Prefect UI."""
    summary = analytics.summary_text(counts, overview, q1, q2, q3)
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
    cutoffs = load_high_value_cutoffs()
    counts, overview = count_orders_in_rds(cutoffs)
    q1 = analyse_account_age(counts)
    q2 = analyse_mobile_verification(counts)
    q3 = analyse_high_value_orders(counts, cutoffs)
    files = save_reports(counts, overview, q1, q2, q3, output_dir)
    return publish_results(counts, overview, q1, q2, q3, files)


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
