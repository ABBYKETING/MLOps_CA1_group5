"""Geographic and account behaviour fraud analytics (PBI 7, Task #64, Yue Liang).

Reads the latest labelled orders from the team's AWS RDS database and answers
three questions for Trust & Safety analysts:

    Q1  Which countries have the highest fraud rates and fraudulent transaction
        volumes, and is any country getting worse recently?
    Q2  Do customers with more associated customer accounts have higher fraud
        rates, and which confirmed-fraud accounts have the largest linked networks?
    Q3  Does a higher cancellation rate over the last 50 days correspond to a
        higher fraud rate, and does combining it with the refund rate find more fraud?

Unit of analysis: one labelled order (fraud_labels joined to its order and its
customer on (country_code, ID), as in the team's RDS design). Fraud rate =
fraud orders / orders in the group.

Review workload view (all three questions): analysts have limited time, so each
segment also reports its share of all orders (review effort), its share of all
fraud (fraud caught), its lift (fraud rate / overall fraud rate) and a 95%
Wilson confidence interval, so that small groups are not over-interpreted.

Definitions:
    Recent period        latest RECENT_WEEKS order weeks (weeks start on Monday);
                         the newest week may be incomplete
    Fraud order value    sum of order_value of fraud orders (SGD): the money at risk
    Fraud refunds        sum of refund_value on fraud orders (SGD), kept in the tables;
                         refund_value is used for analytics only, never as a model feature
    Associated accounts  customers.num_associated_customers, grouped into ASSOCIATED_BANDS
                         (the source data caps it at 100, cleaning Assumption 12)
    Cancellation rate    num_cancelled_orders_last_50days / num_orders_last_50days
    Refund rate          num_refund_orders_last_50days / num_orders_last_50days
                         Customers with no orders in the last 50 days form their own
                         group (a rate cannot be computed), and customers with more
                         cancellations than orders are reported as a data issue.

Caveat: the customer columns (associated accounts, last-50-day counts) are the
latest snapshot of each customer ("latest wins" in the ETL), and their timing
relative to each order is unconfirmed (cleaning Assumption 19). Q2 and Q3 show
associations that help rank accounts for review, not causes of fraud.

How the data is read: MySQL does the grouping and counting, so only summary rows
(a few hundred at most) travel from RDS instead of ~950,000 orders.

Usage (from the repository root, with DB_HOST, DB_USER and DB_PASSWORD set):

    python src/analytics_geo_account_yueliang.py
    python src/analytics_geo_account_yueliang.py --output-dir outputs/analytics_yueliang

Outputs: CSV tables, PNG charts, summary.md and report.html in the output folder.
All queries are read-only (SELECT).
"""

import argparse
import base64
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save charts without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sqlalchemy import text  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import database as db  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analytics_yueliang"     # git-ignored

COUNTRY_NAMES = {"BD": "Bangladesh", "MY": "Malaysia", "PH": "Philippines",
                 "PK": "Pakistan", "TH": "Thailand"}
RECENT_WEEKS = 4
MIN_WEEKLY_ORDERS = 30            # weekly points with fewer orders are not plotted

# (upper limit, label): a band covers values above the previous limit up to its own limit
# Bands chosen from the distribution in RDS (28 Nov 2026): 1-3 links are all near or below
# the overall rate, the rate roughly doubles from 6 links and passes 50% from 9 links.
# The minimum value in the data is 1; 100 means 100 or more (cleaning Assumption 12).
ASSOCIATED_BANDS = [(3, "1-3"), (5, "4-5"), (8, "6-8"), (19, "9-19"), (99, "20-99"), (None, "100+")]
REVIEW_LIST_MIN_ASSOCIATED = 9    # review list: fraud accounts from the band where the rate passes 50%
REVIEW_LIST_SIZE = 20

NO_RECENT, DATA_ISSUE = "No recent orders", "Data issue (more cancelled than orders)"
# The fraud rate rises steadily with the cancellation rate up to about 30%, then levels off
# (and individual 5% steps above 35% hold too few orders to be reliable), so the top band is ">30%".
CANCEL_BANDS = [(0, "0%"), (0.10, ">0-10%"), (0.20, ">10-20%"), (0.30, ">20-30%"), (None, ">30%")]
# "High" for the combined signal: from 20% both rates go with a fraud rate of about 2.5x the
# overall rate, while still covering enough orders (about 2% and 3%) to be a usable review rule.
HIGH_CANCEL_RATE = 0.20
HIGH_REFUND_RATE = 0.20
LIFT_FOCUS = 2.0                  # findings: segments with at least this lift form the "focus" group

PALETTE = {"main": "#3b6ea5", "accent": "#c8553d", "muted": "#9aa5b1"}


# ---------------------------------------------------------------------------
# SQL (read-only)
# ---------------------------------------------------------------------------
LABELLED_ORDERS = """
    FROM fraud_labels l
    JOIN orders o    ON o.country_code = l.country_code AND o.order_id = l.order_id
    JOIN customers c ON c.country_code = l.country_code AND c.customer_id = l.customer_id
"""
FRAUD_VALUE = "SUM(CASE WHEN l.is_fraud = 1 THEN o.order_value ELSE 0 END)"
FRAUD_REFUND = "SUM(CASE WHEN l.is_fraud = 1 THEN o.refund_value ELSE 0 END)"


def band_sql(expr, bands):
    """SQL CASE that puts a numeric expression into the given (upper, label) bands."""
    whens = " ".join(f"WHEN {expr} <= {upper} THEN '{label}'" for upper, label in bands if upper is not None)
    last = next(label for upper, label in bands if upper is None)
    return f"CASE {whens} ELSE '{last}' END"


def rate_sql(count_column):
    """count / num_orders_last_50days; NULL when the customer had no orders in the period."""
    return f"c.{count_column} / NULLIF(c.num_orders_last_50days, 0)"


CANCEL_RATE = rate_sql("num_cancelled_orders_last_50days")
REFUND_RATE = rate_sql("num_refund_orders_last_50days")

CANCEL_SEGMENT = (f"CASE WHEN c.num_orders_last_50days = 0 THEN '{NO_RECENT}' "
                  f"WHEN c.num_cancelled_orders_last_50days > c.num_orders_last_50days THEN '{DATA_ISSUE}' "
                  f"ELSE {band_sql(CANCEL_RATE, CANCEL_BANDS)} END")

# ONE pass over the ~950,000 labelled orders. MySQL groups them by every dimension the three
# questions need (country, order week, associated-account band, cancellation band and the two
# "high" flags), so RDS returns a few thousand summary rows; pandas then adds these groups up
# differently for each question. Scanning the join once instead of once per question is what
# keeps the run within the 3-minute demo limit on the small RDS instance.
COUNTS_QUERY = f"""
    SELECT o.country_code,
           DATE_SUB(o.order_date, INTERVAL WEEKDAY(o.order_date) DAY) AS week_start,
           {band_sql("c.num_associated_customers", ASSOCIATED_BANDS)} AS associated_segment,
           {CANCEL_SEGMENT} AS cancel_segment,
           COALESCE({CANCEL_RATE} >= {HIGH_CANCEL_RATE}, 0) AS high_cancel,
           COALESCE({REFUND_RATE} >= {HIGH_REFUND_RATE}, 0) AS high_refund,
           COUNT(*) AS orders, SUM(l.is_fraud) AS frauds,
           {FRAUD_VALUE} AS fraud_order_value, {FRAUD_REFUND} AS fraud_refund_value,
           MIN(o.order_date) AS first_order_date, MAX(o.order_date) AS last_order_date
    {LABELLED_ORDERS}
    GROUP BY o.country_code, week_start, associated_segment, cancel_segment, high_cancel, high_refund
"""

# Starts from the (few) customers with many associated accounts, then finds their labelled orders,
# so MySQL does not have to scan every order again.
Q2_REVIEW_LIST_QUERY = f"""
    SELECT c.country_code, c.customer_id, c.num_associated_customers,
           COUNT(*) AS orders, SUM(l.is_fraud) AS frauds, {FRAUD_VALUE} AS fraud_order_value
    FROM customers c
    STRAIGHT_JOIN fraud_labels l ON l.country_code = c.country_code AND l.customer_id = c.customer_id
    STRAIGHT_JOIN orders o       ON o.country_code = l.country_code AND o.order_id = l.order_id
    WHERE c.num_associated_customers >= :min_associated
    GROUP BY c.country_code, c.customer_id, c.num_associated_customers
    HAVING SUM(l.is_fraud) > 0
    ORDER BY frauds DESC, c.num_associated_customers DESC, fraud_order_value DESC
    LIMIT :list_size
"""

BATCH_QUERY = "SELECT batch_name, status, finished_at FROM etl_batches ORDER BY started_at"


def _read(query, engine, params=None):
    with engine.connect() as conn:
        frame = pd.read_sql(text(query), conn, params=params or {})
    # MySQL returns SUM() as DECIMAL; convert counts and money to plain numbers
    for column in ("orders", "frauds", "num_associated_customers", "high_cancel", "high_refund"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column]).fillna(0).astype(int)
    for column in ("fraud_order_value", "fraud_refund_value"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column]).fillna(0.0)
    return frame


def count_orders_in_db(engine=None):
    """The single grouped count of all labelled orders (see COUNTS_QUERY)."""
    counts = _read(COUNTS_QUERY, engine or db.get_engine())
    counts["week_start"] = pd.to_datetime(counts["week_start"])
    return counts


def load_batches(engine=None):
    return _read(BATCH_QUERY, engine or db.get_engine())


def query_associated_review_list(engine=None):
    params = {"min_associated": REVIEW_LIST_MIN_ASSOCIATED, "list_size": REVIEW_LIST_SIZE}
    return _read(Q2_REVIEW_LIST_QUERY, engine or db.get_engine(), params)


def data_status(counts, batches):
    """How much labelled data RDS holds now, its date range, and the batches loaded."""
    return {"orders": int(counts["orders"].sum()), "frauds": int(counts["frauds"].sum()),
            "first_order_date": pd.to_datetime(counts["first_order_date"]).min().date(),
            "last_order_date": pd.to_datetime(counts["last_order_date"]).max().date(),
            "batches": batches}


def _sum_by(counts, by, rename=None):
    columns = ["orders", "frauds", "fraud_order_value", "fraud_refund_value"]
    table = counts.groupby(by, observed=True)[columns].sum().reset_index()
    return table.rename(columns=rename or {})


# ---------------------------------------------------------------------------
# Analysis (pandas)
# ---------------------------------------------------------------------------
def wilson_interval(frauds, orders, z=1.96):
    """95% Wilson score interval for a proportion; stays sensible for small groups."""
    k, n = np.asarray(frauds, float), np.asarray(orders, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denominator = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
    return np.clip(centre - margin, 0, 1), np.clip(centre + margin, 0, 1)


def segment_table(counts, by, total_orders, total_frauds):
    """Orders, frauds, fraud rate with 95% CI, lift, and review-workload shares per segment."""
    table = counts.groupby(by, observed=True, sort=False)[["orders", "frauds"]].sum().reset_index()
    low, high = wilson_interval(table["frauds"], table["orders"])
    overall_rate = total_frauds / total_orders
    table["fraud_rate_pct"] = 100 * table["frauds"] / table["orders"]
    table["ci_low_pct"], table["ci_high_pct"] = 100 * low, 100 * high
    table["lift"] = (table["frauds"] / table["orders"]) / overall_rate
    table["share_of_orders_pct"] = 100 * table["orders"] / total_orders
    table["share_of_fraud_pct"] = 100 * table["frauds"] / total_frauds
    return table


def review_priority(table, label):
    """Review segments from the highest fraud rate down: cumulative effort vs fraud caught."""
    ranked = table.sort_values("fraud_rate_pct", ascending=False).reset_index(drop=True)
    ranked.insert(0, "priority", range(1, len(ranked) + 1))
    ranked["cum_share_of_orders_pct"] = ranked["share_of_orders_pct"].cumsum()
    ranked["cum_share_of_fraud_pct"] = ranked["share_of_fraud_pct"].cumsum()
    return ranked[["priority", label, "orders", "frauds", "fraud_rate_pct", "lift",
                   "cum_share_of_orders_pct", "cum_share_of_fraud_pct"]]


def analyse_countries(weeks):
    """Q1: fraud rate, fraud volume and fraud order value per country; latest weeks vs earlier."""
    total_orders, total_frauds = weeks["orders"].sum(), weeks["frauds"].sum()
    countries = segment_table(weeks, "country_code", total_orders, total_frauds)
    money = weeks.groupby("country_code")[["fraud_order_value", "fraud_refund_value"]].sum()
    countries["fraud_order_value"] = countries["country_code"].map(money["fraud_order_value"])
    total_value = money["fraud_order_value"].sum()
    countries["share_of_fraud_value_pct"] = (100 * countries["fraud_order_value"] / total_value
                                             if total_value else 0.0)
    countries["fraud_refund_value"] = countries["country_code"].map(money["fraud_refund_value"])
    countries.insert(1, "country", countries["country_code"].map(COUNTRY_NAMES).fillna(countries["country_code"]))

    recent_start = weeks["week_start"].max() - pd.Timedelta(weeks=RECENT_WEEKS - 1)
    period = np.where(weeks["week_start"] >= recent_start, "recent", "earlier")
    by_period = weeks.assign(period=period).groupby(["country_code", "period"])[["orders", "frauds"]].sum()
    rate = (100 * by_period["frauds"] / by_period["orders"]).unstack("period")
    countries["earlier_fraud_rate_pct"] = countries["country_code"].map(rate.get("earlier"))
    countries["recent_fraud_rate_pct"] = countries["country_code"].map(rate.get("recent"))
    countries["change_pp"] = countries["recent_fraud_rate_pct"] - countries["earlier_fraud_rate_pct"]
    countries = countries.sort_values("fraud_rate_pct", ascending=False).reset_index(drop=True)

    weekly = weeks.copy()
    weekly["fraud_rate_pct"] = 100 * weekly["frauds"] / weekly["orders"]
    weekly = weekly.sort_values(["country_code", "week_start"]).reset_index(drop=True)
    return {"countries": countries, "weekly": weekly, "recent_start": recent_start,
            "priority": review_priority(countries, "country_code")}


def _ordered(table, labels):
    order = [label for label in labels if label in set(table["segment"])]
    table["segment"] = pd.Categorical(table["segment"], categories=order, ordered=True)
    return table.sort_values("segment").reset_index(drop=True)


def analyse_associated(bands, review_list):
    """Q2: fraud rate by number of associated accounts, plus the review list."""
    total_orders, total_frauds = bands["orders"].sum(), bands["frauds"].sum()
    table = segment_table(bands, "segment", total_orders, total_frauds)
    table = _ordered(table, [label for _, label in ASSOCIATED_BANDS])
    review = review_list.copy()
    review.insert(0, "country", review["country_code"].map(COUNTRY_NAMES).fillna(review["country_code"]))
    return {"bands": table, "priority": review_priority(table, "segment"), "review_list": review}


def analyse_cancellation(counts):
    """Q3: fraud rate by cancellation band, and the combined cancellation + refund signal."""
    total_orders, total_frauds = counts["orders"].sum(), counts["frauds"].sum()
    labels = [NO_RECENT] + [label for _, label in CANCEL_BANDS] + [DATA_ISSUE]
    bands = _ordered(segment_table(counts, "segment", total_orders, total_frauds), labels)

    with_rate = counts[~counts["segment"].isin([NO_RECENT, DATA_ISSUE])].copy()
    names = {(0, 0): "Neither high", (1, 0): "High cancellation only",
             (0, 1): "High refund only", (1, 1): "Both high"}
    with_rate["signal"] = [names[(c, r)] for c, r in zip(with_rate["high_cancel"], with_rate["high_refund"])]
    combined = segment_table(with_rate, "signal", total_orders, total_frauds)
    combined["signal"] = pd.Categorical(combined["signal"], categories=list(names.values()), ordered=True)
    combined = combined.sort_values("signal").reset_index(drop=True)
    usable = bands[bands["segment"] != DATA_ISSUE].copy()      # data-issue rows are not a review rule
    return {"bands": bands, "priority": review_priority(usable, "segment"), "combined": combined}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _rate_bars(ax, table, label, overall_rate_pct, title):
    x = np.arange(len(table))
    errors = [table["fraud_rate_pct"] - table["ci_low_pct"], table["ci_high_pct"] - table["fraud_rate_pct"]]
    ax.bar(x, table["fraud_rate_pct"], color=PALETTE["main"], yerr=errors, capsize=4,
           error_kw={"elinewidth": 1, "ecolor": "#333333"})
    ax.axhline(overall_rate_pct, color=PALETTE["accent"], linestyle="--", linewidth=1,
               label=f"Overall {overall_rate_pct:.1f}%")
    ax.set_xticks(x, table[label].astype(str), rotation=0)
    for xi, (rate, top, n) in enumerate(zip(table["fraud_rate_pct"], table["ci_high_pct"], table["orders"])):
        ax.annotate(f"{rate:.1f}%\nn={n:,}", (xi, top), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Fraud rate (% of orders)")
    ax.set_title(title)
    ax.set_ylim(0, max(table["ci_high_pct"].max(), overall_rate_pct) * 1.35)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def _share_bars(ax, table, label, title):
    long = table.melt(id_vars=label, value_vars=["share_of_orders_pct", "share_of_fraud_pct"],
                      var_name="measure", value_name="pct")
    long["measure"] = long["measure"].map({"share_of_orders_pct": "Share of all orders (review effort)",
                                           "share_of_fraud_pct": "Share of all fraud (fraud caught)"})
    long[label] = long[label].astype(str)
    sns.barplot(data=long, x=label, y="pct", hue="measure", ax=ax,
                palette=[PALETTE["muted"], PALETTE["accent"]])
    ax.set_xlabel("")
    ax.set_ylabel("% of total")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)


def plot_countries(q1, overall_rate_pct, path):
    countries = q1["countries"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _rate_bars(axes[0], countries, "country_code", overall_rate_pct, "Fraud rate by country (95% CI)")
    by_volume = countries.sort_values("frauds", ascending=False)
    axes[1].bar(by_volume["country_code"], by_volume["frauds"], color=PALETTE["accent"])
    axes[1].set_title("Fraud orders by country")
    axes[1].set_ylabel("Fraud orders")
    by_loss = countries.sort_values("fraud_order_value", ascending=False)
    axes[2].bar(by_loss["country_code"], by_loss["fraud_order_value"], color=PALETTE["muted"])
    axes[2].set_title("Value of fraud orders (SGD)")
    axes[2].set_ylabel("SGD")
    for ax, table, column in ((axes[1], by_volume, "share_of_fraud_pct"),
                              (axes[2], by_loss, "share_of_fraud_value_pct")):
        for xi, share in enumerate(table[column]):
            ax.annotate(f"{share:.0f}%", (xi, ax.patches[xi].get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("Q1  Geographic fraud overview", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_weekly_trend(q1, path):
    weekly = q1["weekly"][q1["weekly"]["orders"] >= MIN_WEEKLY_ORDERS]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=weekly, x="week_start", y="fraud_rate_pct", hue="country_code", marker="o", ax=ax)
    ax.axvspan(q1["recent_start"], weekly["week_start"].max() + pd.Timedelta(days=6),
               color=PALETTE["muted"], alpha=0.15, label=f"Latest {RECENT_WEEKS} weeks")
    ax.set_xlabel("Week starting")
    ax.set_ylabel("Fraud rate (% of orders)")
    ax.set_title(f"Q1  Weekly fraud rate by country (weeks with at least {MIN_WEEKLY_ORDERS} orders)")
    ax.legend(title="", fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_associated(q2, overall_rate_pct, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    _rate_bars(axes[0], q2["bands"], "segment", overall_rate_pct, "Fraud rate by number of associated accounts")
    axes[0].set_xlabel("Associated customer accounts")
    _share_bars(axes[1], q2["bands"], "segment", "Review effort vs fraud caught per band")
    fig.suptitle("Q2  Associated customer accounts", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cancellation(q3, overall_rate_pct, path):
    bands = q3["bands"][q3["bands"]["segment"] != DATA_ISSUE]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [3, 2]})
    _rate_bars(axes[0], bands, "segment", overall_rate_pct, "Fraud rate by cancellation rate (last 50 days)")
    axes[0].set_xlabel("Share of the customer's orders cancelled")
    axes[0].tick_params(axis="x", labelsize=8)
    combined = q3["combined"]
    _rate_bars(axes[1], combined, "signal", overall_rate_pct,
               f"Combined signal (high = {HIGH_CANCEL_RATE:.0%} or more)")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.suptitle("Q3  Cancellation behaviour", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Findings and reports
# ---------------------------------------------------------------------------
def focus_line(priority, what):
    """Effort vs fraud caught if analysts review only the segments with lift >= LIFT_FOCUS."""
    focus = priority[priority["lift"] >= LIFT_FOCUS]
    if focus.empty:
        return f"No {what} reach {LIFT_FOCUS:.0f}x the overall fraud rate."
    names = ", ".join(str(label) for label in focus.iloc[:, 1])
    last = focus.iloc[-1]
    rate = 100 * focus["frauds"].sum() / focus["orders"].sum()
    return (f"Reviewing only the {what} at {LIFT_FOCUS:.0f}x the overall rate or more ({names}) means "
            f"{last['cum_share_of_orders_pct']:.1f}% of orders and catches {last['cum_share_of_fraud_pct']:.1f}% "
            f"of all fraud, with {rate:.1f}% of reviewed orders being fraud.")


def findings(status, q1, q2, q3):
    """Plain-language findings recomputed from the latest data on every run."""
    overall = 100 * status["frauds"] / status["orders"]
    countries = q1["countries"]
    top_rate, top_volume = countries.iloc[0], countries.sort_values("frauds", ascending=False).iloc[0]
    top_loss = countries.sort_values("fraud_order_value", ascending=False).iloc[0]
    rising = countries.sort_values("change_pp", ascending=False).iloc[0]

    bands = q2["bands"]
    low_band, high_band = bands.iloc[0], bands.iloc[-1]
    riskiest = bands.sort_values("fraud_rate_pct", ascending=False).iloc[0]

    cancel = q3["bands"]
    cancel_rated = cancel[~cancel["segment"].isin([NO_RECENT, DATA_ISSUE])]
    zero = cancel_rated.iloc[0]
    top_cancel = cancel_rated.sort_values("fraud_rate_pct", ascending=False).iloc[0]
    no_recent_rows = cancel[cancel["segment"] == NO_RECENT]
    no_recent = no_recent_rows.iloc[0] if len(no_recent_rows) else None
    combined = q3["combined"].set_index("signal")
    issues = cancel.loc[cancel["segment"] == DATA_ISSUE, "orders"].sum()

    def rate(row):
        return f"{row['fraud_rate_pct']:.1f}% (95% CI {row['ci_low_pct']:.1f}-{row['ci_high_pct']:.1f}%)"

    lines = {
        "Q1": [
            f"Highest fraud rate: {top_rate['country']} at {rate(top_rate)}, "
            f"{top_rate['lift']:.2f}x the overall rate of {overall:.1f}%.",
            f"Most fraud orders: {top_volume['country']} with {top_volume['frauds']:,} "
            f"({top_volume['share_of_fraud_pct']:.1f}% of all fraud, from {top_volume['share_of_orders_pct']:.1f}% of orders).",
            f"Largest value of fraud orders: {top_loss['country']}, SGD {top_loss['fraud_order_value']:,.0f} "
            f"({top_loss['share_of_fraud_value_pct']:.1f}% of the SGD {countries['fraud_order_value'].sum():,.0f} total).",
            f"Largest change in the latest {RECENT_WEEKS} weeks: {rising['country']}, "
            f"{rising['earlier_fraud_rate_pct']:.1f}% -> {rising['recent_fraud_rate_pct']:.1f}% "
            f"({rising['change_pp']:+.1f} percentage points).",
        ],
        "Q2": [
            f"Customers with {low_band['segment']} associated accounts: {rate(low_band)}; "
            f"with {high_band['segment']}: {rate(high_band)}.",
            f"Riskiest band: {riskiest['segment']} associated accounts at {riskiest['fraud_rate_pct']:.1f}% "
            f"({riskiest['lift']:.2f}x overall), covering {riskiest['share_of_orders_pct']:.1f}% of orders "
            f"and {riskiest['share_of_fraud_pct']:.1f}% of fraud.",
            focus_line(q2["priority"], "associated-account bands"),
            f"{len(q2['review_list'])} confirmed-fraud accounts with at least {REVIEW_LIST_MIN_ASSOCIATED} "
            "associated accounts are listed for linked-account review (q2_review_list.csv).",
        ],
        "Q3": [
            f"Customers who cancelled none of their recent orders: {rate(zero)}; "
            f"riskiest cancellation band {top_cancel['segment']}: {rate(top_cancel)}, {top_cancel['lift']:.2f}x overall.",
            *([f"Customers with no orders in the last 50 days: {rate(no_recent)}, {no_recent['lift']:.2f}x overall, "
               f"{no_recent['share_of_orders_pct']:.1f}% of orders and {no_recent['share_of_fraud_pct']:.1f}% of fraud "
               "(their snapshot shows no recent activity even though they placed a labelled order)."]
              if no_recent is not None else []),
            focus_line(q3["priority"], "cancellation groups"),
            *(f"{name}: {rate(combined.loc[name])}, {combined.loc[name, 'share_of_orders_pct']:.1f}% of orders, "
              f"{combined.loc[name, 'share_of_fraud_pct']:.1f}% of fraud."
              for name in combined.index),
            f"Data check: {issues:,} orders belong to customers with more cancellations than orders.",
        ],
    }
    caveat = ("Customer columns (associated accounts, last-50-day counts) are each customer's latest "
              "snapshot and their timing relative to each order is unconfirmed, so Q2 and Q3 are "
              "associations for prioritising reviews, not proof that a behaviour causes fraud.")
    return lines, caveat


def _status_lines(status):
    batches = status["batches"]
    loaded = ", ".join(f"{b} ({s})" for b, s in zip(batches["batch_name"], batches["status"])) or "none recorded"
    return [f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            f"Labelled orders analysed: {status['orders']:,} ({status['frauds']:,} fraud, "
            f"{100 * status['frauds'] / status['orders']:.1f}%)",
            f"Order dates: {status['first_order_date']} to {status['last_order_date']}",
            f"ETL batches in RDS: {loaded}"]


def summary_markdown(status, lines, caveat):
    titles = {"Q1": "Q1 Geographic fraud patterns", "Q2": "Q2 Associated customer accounts",
              "Q3": "Q3 Cancellation behaviour"}
    out = ["# Geographic and account behaviour fraud analytics", ""]
    out += [f"- {line}" for line in _status_lines(status)]
    for key, title in titles.items():
        out += ["", f"## {title}", ""] + [f"- {line}" for line in lines[key]]
    out += ["", "## Caveat", "", caveat, ""]
    return "\n".join(out)


def _image_tag(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{Path(path).stem}">'


REPORT_COLUMNS = {   # column -> readable header, in display order (columns a table lacks are skipped)
    "country": "Country", "segment": "Group", "signal": "Signal", "customer_id": "Customer ID",
    "num_associated_customers": "Associated accounts", "orders": "Orders", "frauds": "Fraud orders",
    "fraud_rate_pct": "Fraud rate %", "ci_low_pct": "95% CI low %", "ci_high_pct": "95% CI high %",
    "lift": "Lift", "share_of_orders_pct": "% of orders", "share_of_fraud_pct": "% of fraud",
    "fraud_order_value": "Fraud value (SGD)", "share_of_fraud_value_pct": "% of fraud value",
    "earlier_fraud_rate_pct": "Earlier fraud rate %", "recent_fraud_rate_pct": "Latest 4 weeks %",
    "change_pp": "Change (pp)",
}


def _html_table(table):
    columns = [column for column in REPORT_COLUMNS if column in table]
    shown = table[columns].rename(columns=REPORT_COLUMNS)
    formats = {header: (lambda v: f"{v:,}") for header in ("Orders", "Fraud orders", "Associated accounts")}
    return shown.to_html(index=False, float_format=lambda v: f"{v:,.2f}", formatters=formats,
                         border=0, classes="data")


def report_html(status, lines, caveat, tables, charts):
    sections = [
        ("Q1 Which countries have the highest fraud rates and fraudulent transaction volumes?",
         "Q1", [charts["q1_country_overview"], charts["q1_weekly_trend"]], [tables["q1_countries"]]),
        ("Q2 Do customers with more associated customer accounts have higher fraud rates?",
         "Q2", [charts["q2_associated_accounts"]], [tables["q2_associated_bands"], tables["q2_review_list"]]),
        ("Q3 Does a higher cancellation rate over the last 50 days correspond to a higher fraud rate?",
         "Q3", [charts["q3_cancellation"]], [tables["q3_cancellation_bands"], tables["q3_combined_signal"]]),
    ]
    body = ["<h1>Geographic and account behaviour fraud analytics</h1>",
            "<ul>" + "".join(f"<li>{line}</li>" for line in _status_lines(status)) + "</ul>"]
    for title, key, images, frames in sections:
        body.append(f"<h2>{title}</h2><ul>" + "".join(f"<li>{line}</li>" for line in lines[key]) + "</ul>")
        body += [_image_tag(image) for image in images]
        body += [_html_table(frame) for frame in frames]
    body.append(f"<h2>Caveat</h2><p>{caveat}</p>")
    style = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1150px;margin:24px auto;color:#222}"
             "img{max-width:100%;margin:12px 0}table.data{border-collapse:collapse;font-size:13px;margin:12px 0}"
             "table.data th,table.data td{padding:4px 10px;border-bottom:1px solid #ddd;text-align:right}"
             "table.data th{background:#f2f4f7}h2{margin-top:36px;border-bottom:2px solid #3b6ea5}")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Fraud analytics report</title>"
            f"<style>{style}</style></head><body>{''.join(body)}</body></html>")


def save_outputs(status, q1, q2, q3, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_rate_pct = 100 * status["frauds"] / status["orders"]
    tables = {
        "q1_countries": q1["countries"],
        "q1_weekly": q1["weekly"],
        "q1_review_priority": q1["priority"],
        "q2_associated_bands": q2["bands"],
        "q2_review_priority": q2["priority"],
        "q2_review_list": q2["review_list"],
        "q3_cancellation_bands": q3["bands"],
        "q3_review_priority": q3["priority"],
        "q3_combined_signal": q3["combined"],
    }
    for name, table in tables.items():
        table.round(3).to_csv(output_dir / f"{name}.csv", index=False)

    charts = {name: output_dir / f"{name}.png" for name in
              ("q1_country_overview", "q1_weekly_trend", "q2_associated_accounts", "q3_cancellation")}
    sns.set_theme(style="whitegrid")
    plot_countries(q1, overall_rate_pct, charts["q1_country_overview"])
    plot_weekly_trend(q1, charts["q1_weekly_trend"])
    plot_associated(q2, overall_rate_pct, charts["q2_associated_accounts"])
    plot_cancellation(q3, overall_rate_pct, charts["q3_cancellation"])

    lines, caveat = findings(status, q1, q2, q3)
    summary = summary_markdown(status, lines, caveat)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    report_tables = {key: tables[key] for key in ("q1_countries", "q2_associated_bands", "q2_review_list",
                                                   "q3_cancellation_bands", "q3_combined_signal")}
    (output_dir / "report.html").write_text(report_html(status, lines, caveat, report_tables, charts),
                                            encoding="utf-8")
    return {"output_dir": output_dir, "summary": summary, "findings": lines, "caveat": caveat,
            "tables": tables, "charts": charts}


def country_weeks(counts):
    """Q1 input: orders, frauds and fraud value per country and order week."""
    return _sum_by(counts, ["country_code", "week_start"])


def associated_groups(counts):
    """Q2 input: orders and frauds per associated-account band."""
    return _sum_by(counts, "associated_segment", {"associated_segment": "segment"})


def cancellation_groups(counts):
    """Q3 input: orders and frauds per cancellation band and high-cancel/high-refund flags."""
    return _sum_by(counts, ["cancel_segment", "high_cancel", "high_refund"], {"cancel_segment": "segment"})


def run_analysis(output_dir=DEFAULT_OUTPUT_DIR, engine=None, log=print):
    """Run all three analyses on the latest RDS data and save the outputs."""
    engine = engine or db.get_engine()
    started = time.perf_counter()
    counts = count_orders_in_db(engine)
    log(f"Counted all labelled orders in RDS: {len(counts):,} groups ({time.perf_counter() - started:.1f} s)")
    review_list = query_associated_review_list(engine)
    status = data_status(counts, load_batches(engine))
    log(f"Loaded review list and batch log ({time.perf_counter() - started:.1f} s)")
    q1 = analyse_countries(country_weeks(counts))
    q2 = analyse_associated(associated_groups(counts), review_list)
    q3 = analyse_cancellation(cancellation_groups(counts))
    result = save_outputs(status, q1, q2, q3, output_dir)
    log(f"Saved tables, charts and reports ({time.perf_counter() - started:.1f} s)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Geographic and account behaviour fraud analytics from RDS")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="folder for tables, charts and reports")
    args = parser.parse_args()
    started = datetime.now()
    result = run_analysis(args.output_dir)
    print(result["summary"])
    print(f"Outputs saved to {result['output_dir']} in {(datetime.now() - started).total_seconds():.1f} s")


if __name__ == "__main__":
    main()
