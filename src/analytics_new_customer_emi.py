"""New customer risk analytics (PBI 7, Task #62, Emi).

Reads the latest labelled orders from the team's AWS RDS database and answers
three questions for Trust & Safety analysts:

    Q1  Do customers in their first 30 days have higher fraud rates than
        established customers?
    Q2  Among new customers, how does mobile verification relate to fraud?
    Q3  Are high-value orders from new customers riskier than similar orders
        from established customers?

Definitions (documented for the individual submission):

    account age   = order_date - date of the customer's first order (in days),
                    the same calculation as the team's feature engineering
    New           = account age 0-30 days on the order date
    Established   = account age above 30 days
    Unknown       = first order date missing, or later than the order date
                    (a data issue); kept out of the New/Established comparison
    High value    = order_value in the top 25% of orders from the same country
                    (prices differ a lot between countries, so one global
                    cut-off would mostly flag MY/PH orders)

How the data is read: MySQL does the counting. RDS sends back only the
high-value cut-off per country and the number of orders and frauds per
(account age band, mobile verification, value) group - a few dozen rows
instead of ~950,000 orders - so the run stays far below the 3-minute limit.
pandas then builds the tables, matplotlib/seaborn the charts.

Usage (from the repository root, with DB_HOST, DB_USER and DB_PASSWORD set):

    python src/analytics_new_customer_emi.py
    python src/analytics_new_customer_emi.py --output-dir outputs/analytics_emi

Tables (CSV), charts (PNG) and a short summary are written to the output folder.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # save charts without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sqlalchemy import text  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import database as db  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analytics_emi"   # git-ignored

NEW_CUSTOMER_DAYS = 30
HIGH_VALUE_QUANTILE = 0.75
NEW, ESTABLISHED, UNKNOWN = "New (0-30 days)", "Established (>30 days)", "Unknown age"
HIGH, NORMAL = "High value (top 25%)", "Normal value"
# (upper limit in days, label); a band covers ages above the previous limit up to its own
AGE_BANDS = [(0, "Same day"), (7, "1-7 days"), (30, "8-30 days"), (90, "31-90 days"),
             (180, "91-180 days"), (365, "181-365 days"), (None, "Over 1 year")]
BAND_ORDER = [label for _, label in AGE_BANDS]
NEW_BANDS = [label for limit, label in AGE_BANDS if limit is not None and limit <= NEW_CUSTOMER_DAYS]
UNKNOWN_BAND = "Unknown"
COUNT_COLUMNS = ["age_band", "mobile_verified", "value_segment", "orders", "frauds"]


# ---------------------------------------------------------------------------
# Reading the data (MySQL does the counting)
# ---------------------------------------------------------------------------
LABELLED_ORDERS = """
    FROM fraud_labels l
    JOIN orders o    ON o.country_code = l.country_code AND o.order_id = l.order_id
    JOIN customers c ON c.country_code = l.country_code AND c.customer_id = l.customer_id
"""

# 75th percentile of order_value per country, interpolated like pandas' quantile():
# position (n - 1) * 0.75 in the sorted values, between the values at its floor and ceiling.
CUTOFF_QUERY = f"""
    WITH ranked AS (
        SELECT o.country_code, o.order_value,
               ROW_NUMBER() OVER (PARTITION BY o.country_code ORDER BY o.order_value) - 1 AS pos,
               COUNT(*) OVER (PARTITION BY o.country_code) AS n
        {LABELLED_ORDERS}
    )
    SELECT country_code,
           MAX(CASE WHEN pos = FLOOR((n - 1) * {HIGH_VALUE_QUANTILE}) THEN order_value END) AS lower_value,
           MAX(CASE WHEN pos = CEIL((n - 1) * {HIGH_VALUE_QUANTILE}) THEN order_value END) AS upper_value,
           MAX((n - 1) * {HIGH_VALUE_QUANTILE} - FLOOR((n - 1) * {HIGH_VALUE_QUANTILE})) AS fraction
    FROM ranked
    WHERE pos IN (FLOOR((n - 1) * {HIGH_VALUE_QUANTILE}), CEIL((n - 1) * {HIGH_VALUE_QUANTILE}))
    GROUP BY country_code
    ORDER BY country_code
"""

def _age_band_sql():
    age = "DATEDIFF(o.order_date, DATE(c.first_order_datetime))"
    cases = [f"WHEN c.first_order_datetime IS NULL OR {age} < 0 THEN '{UNKNOWN_BAND}'"]
    cases += [f"WHEN {age} <= {limit} THEN '{label}'" for limit, label in AGE_BANDS if limit is not None]
    return "CASE " + " ".join(cases) + f" ELSE '{AGE_BANDS[-1][1]}' END"


def high_value_cutoffs(engine=None):
    """Per country: the 75th percentile of order_value, and the value an order needs to reach it."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        cutoffs = pd.read_sql(text(CUTOFF_QUERY), conn)
    cutoffs["fraction"] = cutoffs["fraction"].astype(float)
    cutoffs["high_value_cutoff"] = (cutoffs["lower_value"]
                                    + (cutoffs["upper_value"] - cutoffs["lower_value"]) * cutoffs["fraction"])
    # The cut-off lies strictly between two neighbouring values unless the fraction is 0,
    # so "order_value >= cutoff" is exactly "order_value >= this value" (no rounding issues)
    cutoffs["threshold_value"] = np.where(cutoffs["fraction"] == 0, cutoffs["lower_value"], cutoffs["upper_value"])
    return cutoffs


def count_orders_in_db(cutoffs, engine=None):
    """Orders and frauds per (age band, mobile verification, value) group, counted by MySQL."""
    engine = engine or db.get_engine()
    params, whens = {}, []
    for i, row in enumerate(cutoffs.itertuples()):
        params[f"c{i}"], params[f"t{i}"] = row.country_code, float(row.threshold_value)
        whens.append(f"WHEN :c{i} THEN :t{i}")
    threshold = f"CASE o.country_code {' '.join(whens)} END"
    query = f"""
        SELECT {_age_band_sql()} AS age_band,
               c.mobile_verified AS mobile_verified,
               CASE WHEN o.order_value >= {threshold} THEN '{HIGH}' ELSE '{NORMAL}' END AS value_segment,
               COUNT(*) AS orders, SUM(l.is_fraud) AS frauds,
               MIN(o.order_date) AS first_order_date, MAX(o.order_date) AS last_order_date
        {LABELLED_ORDERS}
        GROUP BY age_band, mobile_verified, value_segment
    """
    with engine.connect() as conn:
        counts = pd.read_sql(text(query), conn, params=params)
    return _tidy_counts(counts), _overview(counts)


def load_counts(engine=None):
    """Everything the analysis needs from RDS: (group counts, overview, high-value cut-offs)."""
    engine = engine or db.get_engine()
    cutoffs = high_value_cutoffs(engine)
    counts, overview = count_orders_in_db(cutoffs, engine)
    return counts, overview, cutoffs


def count_orders(orders):
    """The same group counts, computed in pandas from one row per labelled order.

    Used by the tests (and handy for small data); it applies exactly the
    definitions the SQL above applies inside MySQL.
    """
    df = orders.copy()
    age = (pd.to_datetime(df["order_date"]) - pd.to_datetime(df["first_order_datetime"]).dt.normalize()).dt.days
    band = pd.Series(UNKNOWN_BAND, index=df.index, dtype=object)
    known = age.notna() & (age >= 0)
    lower = -1
    for limit, label in AGE_BANDS:
        in_band = known & (age > lower) & ((age <= limit) if limit is not None else True)
        band[in_band] = label
        lower = limit
    cutoff = df.groupby("country_code")["order_value"].transform(lambda v: v.quantile(HIGH_VALUE_QUANTILE))
    df["age_band"] = band
    df["value_segment"] = np.where(df["order_value"] >= cutoff, HIGH, NORMAL)
    counts = (df.groupby(["age_band", "mobile_verified", "value_segment"])
                .agg(orders=("is_fraud", "size"), frauds=("is_fraud", "sum"),
                     first_order_date=("order_date", "min"), last_order_date=("order_date", "max"))
                .reset_index())
    cutoffs = (df.groupby("country_code")["order_value"].quantile(HIGH_VALUE_QUANTILE)
                 .rename("high_value_cutoff").reset_index())
    return _tidy_counts(counts), _overview(counts), cutoffs


def _tidy_counts(counts):
    counts = counts.copy()
    counts["mobile_verified"] = counts["mobile_verified"].astype(int).astype(bool)
    counts["orders"] = counts["orders"].astype(int)
    counts["frauds"] = counts["frauds"].astype(int)
    counts["customer_segment"] = np.select(
        [counts["age_band"].isin(NEW_BANDS), counts["age_band"] == UNKNOWN_BAND],
        [NEW, UNKNOWN], default=ESTABLISHED)
    counts["age_band"] = pd.Categorical(counts["age_band"], categories=BAND_ORDER + [UNKNOWN_BAND], ordered=True)
    return (counts.sort_values(["age_band", "mobile_verified", "value_segment"])
                  .reset_index(drop=True)[COUNT_COLUMNS + ["customer_segment"]])


def _overview(counts):
    """Totals and order-date range over all labelled orders, from the group counts."""
    return {"orders": int(counts["orders"].sum()), "frauds": int(counts["frauds"].sum()),
            "first_order_date": pd.Timestamp(pd.to_datetime(counts["first_order_date"]).min()),
            "last_order_date": pd.Timestamp(pd.to_datetime(counts["last_order_date"]).max())}


# ---------------------------------------------------------------------------
# The three questions (all work on the group counts)
# ---------------------------------------------------------------------------
def wilson_interval(frauds, orders, z=1.96):
    """95% confidence interval for a fraud rate (works for small groups too)."""
    frauds, orders = np.asarray(frauds, float), np.asarray(orders, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = frauds / orders
        centre = (rate + z**2 / (2 * orders)) / (1 + z**2 / orders)
        half = z * np.sqrt(rate * (1 - rate) / orders + z**2 / (4 * orders**2)) / (1 + z**2 / orders)
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def fraud_summary(counts, by):
    """Orders, frauds, fraud rate (%), its 95% interval and shares, per group."""
    table = counts.groupby(by, observed=True)[["orders", "frauds"]].sum().reset_index()
    low, high = wilson_interval(table["frauds"], table["orders"])
    table["fraud_rate_pct"] = 100 * table["frauds"] / table["orders"]
    table["ci_low_pct"], table["ci_high_pct"] = 100 * low, 100 * high
    table["share_of_orders_pct"] = 100 * table["orders"] / counts["orders"].sum()
    table["share_of_all_fraud_pct"] = 100 * table["frauds"] / counts["frauds"].sum()
    return table


def q1_account_age(counts):
    """Q1: fraud rate for new vs established customers, and by account age band."""
    by_segment = fraud_summary(counts, "customer_segment")
    known = counts[counts["customer_segment"] != UNKNOWN]
    by_band = fraud_summary(known, "age_band")
    by_band["age_band"] = by_band["age_band"].astype(str)
    seg = by_segment.set_index("customer_segment")
    findings = {
        "new_fraud_rate_pct": seg.loc[NEW, "fraud_rate_pct"] if NEW in seg.index else np.nan,
        "established_fraud_rate_pct": seg.loc[ESTABLISHED, "fraud_rate_pct"] if ESTABLISHED in seg.index else np.nan,
        "new_share_of_orders_pct": seg.loc[NEW, "share_of_orders_pct"] if NEW in seg.index else np.nan,
        "new_share_of_fraud_pct": seg.loc[NEW, "share_of_all_fraud_pct"] if NEW in seg.index else np.nan,
    }
    findings["times_higher"] = findings["new_fraud_rate_pct"] / findings["established_fraud_rate_pct"]
    return {"by_segment": by_segment, "by_age_band": by_band, "findings": findings}


def q2_mobile_verification(counts):
    """Q2: fraud rate by mobile verification, for new and established customers."""
    known = counts[counts["customer_segment"] != UNKNOWN]
    table = fraud_summary(known, ["customer_segment", "mobile_verified"])
    new = table[table["customer_segment"] == NEW].set_index("mobile_verified")
    new_orders = new["orders"].sum()
    findings = {
        "new_verified_pct": 100 * new.loc[True, "orders"] / new_orders if True in new.index else 0.0,
        "new_unverified_orders": int(new.loc[False, "orders"]) if False in new.index else 0,
        "new_unverified_fraud_rate_pct": new.loc[False, "fraud_rate_pct"] if False in new.index else np.nan,
        "new_verified_fraud_rate_pct": new.loc[True, "fraud_rate_pct"] if True in new.index else np.nan,
    }
    return {"table": table, "findings": findings}


def q3_high_value_orders(counts, cutoffs):
    """Q3: fraud rate of high-value vs normal orders, for new and established customers."""
    known = counts[counts["customer_segment"] != UNKNOWN]
    table = fraud_summary(known, ["customer_segment", "value_segment"])
    baseline = table[(table["customer_segment"] == ESTABLISHED) & (table["value_segment"] == NORMAL)]["fraud_rate_pct"]
    table["times_established_normal"] = table["fraud_rate_pct"] / (baseline.iloc[0] if len(baseline) else np.nan)
    pick = table.set_index(["customer_segment", "value_segment"])
    findings = {}
    if (NEW, HIGH) in pick.index:
        row = pick.loc[(NEW, HIGH)]
        findings = {
            "new_high_value_fraud_rate_pct": row["fraud_rate_pct"],
            "new_high_value_orders": int(row["orders"]),
            "new_high_value_share_of_orders_pct": 100 * row["orders"] / counts["orders"].sum(),
            "new_high_value_share_of_fraud_pct": 100 * row["frauds"] / counts["frauds"].sum(),
            "overall_fraud_rate_pct": 100 * counts["frauds"].sum() / counts["orders"].sum(),
        }
    return {"table": table, "cutoffs": cutoffs[["country_code", "high_value_cutoff"]], "findings": findings}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _grouped_bars(ax, table, x, hue, x_order, hue_order, colours):
    """Grouped bars with 95% interval whiskers and 'rate (n=orders)' labels."""
    width = 0.8 / len(hue_order)
    for j, level in enumerate(hue_order):
        for i, group in enumerate(x_order):
            row = table[(table[x] == group) & (table[hue] == level)]
            if row.empty:
                continue
            row = row.iloc[0]
            centre = i - 0.4 + width * (j + 0.5)
            rate = row["fraud_rate_pct"]
            ax.bar(centre, rate, width=width * 0.95, color=colours[level],
                   label=level if i == 0 else None)
            ax.errorbar(centre, rate, yerr=[[rate - row["ci_low_pct"]], [row["ci_high_pct"] - rate]],
                        color="black", capsize=3, linewidth=1)
            ax.annotate(f"{rate:.1f}%\n(n={row['orders']:,})", (centre, row["ci_high_pct"]),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(x_order)), x_order)
    ax.legend(frameon=False)


def plot_q1(result, overall_rate, path):
    table = result["by_age_band"]
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#c0392b" if band in NEW_BANDS else "#7f8c8d" for band in table["age_band"]]
    ax.bar(table["age_band"], table["fraud_rate_pct"], color=colours)
    for i, row in enumerate(table.itertuples()):
        ax.annotate(f"{row.fraud_rate_pct:.1f}%", (i, row.fraud_rate_pct), ha="center", va="bottom",
                    fontsize=9, xytext=(0, 2), textcoords="offset points")
    ax.axhline(overall_rate, color="black", linestyle="--", linewidth=1,
               label=f"Overall fraud rate {overall_rate:.1f}%")
    ax.legend(frameon=False, loc="upper right")
    ax.set(title="Q1. Fraud rate falls as customer accounts get older\n(red = new customers, first 30 days)",
           xlabel="Customer account age on the order date", ylabel="Fraud rate (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_q2(result, path):
    table = result["table"].copy()
    table["verification"] = table["mobile_verified"].map({True: "Verified", False: "Not verified"})
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, table, x="customer_segment", hue="verification",
                  x_order=[NEW, ESTABLISHED], hue_order=["Verified", "Not verified"],
                  colours={"Verified": "#2c7fb8", "Not verified": "#fdae61"})
    ax.set(title="Q2. Mobile verification does not separate risky new customers\n"
                 "(error bars = 95% confidence interval; few customers are unverified)",
           xlabel="", ylabel="Fraud rate (%)")
    ax.set_ylim(0, table["ci_high_pct"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_q3(result, path):
    table = result["table"]
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, table, x="customer_segment", hue="value_segment",
                  x_order=[NEW, ESTABLISHED], hue_order=[NORMAL, HIGH],
                  colours={HIGH: "#c0392b", NORMAL: "#95a5a6"})
    ax.set(title="Q3. High-value orders are riskier, most of all from new customers\n"
                 "(high value = top 25% of order values in the same country)",
           xlabel="", ylabel="Fraud rate (%)")
    ax.set_ylim(0, table["ci_high_pct"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary and full run
# ---------------------------------------------------------------------------
def summary_text(counts, overview, q1, q2, q3):
    """Plain-English findings for analysts (also shown in the Prefect flow)."""
    f1, f2, f3 = q1["findings"], q2["findings"], q3["findings"]
    unknown = int(counts.loc[counts["customer_segment"] == UNKNOWN, "orders"].sum())
    lines = [
        "# New customer risk analytics",
        "",
        f"Data: {overview['orders']:,} labelled orders from RDS, order dates "
        f"{overview['first_order_date']:%d %b %Y} to {overview['last_order_date']:%d %b %Y}; "
        f"overall fraud rate {100 * overview['frauds'] / overview['orders']:.1f}%. "
        f"{unknown:,} orders have a missing or invalid first-order date and are left out of the comparisons.",
        "",
        "## Q1. Are customers in their first 30 days riskier?",
        f"Yes. New customers have a {f1['new_fraud_rate_pct']:.1f}% fraud rate vs "
        f"{f1['established_fraud_rate_pct']:.1f}% for established customers ({f1['times_higher']:.1f}x higher). "
        f"They place {f1['new_share_of_orders_pct']:.1f}% of orders but account for "
        f"{f1['new_share_of_fraud_pct']:.1f}% of all fraud.",
        "Decision: orders from customers in their first 30 days deserve extra monitoring.",
        "",
        "## Q2. Does mobile verification help among new customers?",
        f"Not as a risk signal. {f2['new_verified_pct']:.1f}% of new customers' orders come from verified "
        f"accounts, and verified new customers still have a {f2['new_verified_fraud_rate_pct']:.1f}% fraud rate. "
        f"Only {f2['new_unverified_orders']:,} orders come from unverified new customers "
        f"({f2['new_unverified_fraud_rate_pct']:.1f}% fraud), too few to rely on.",
        "Decision: do not treat a verified phone number as evidence that a new customer is safe; "
        "verification alone should not lower the review priority.",
        "",
        "## Q3. Are high-value orders from new customers riskier?",
    ]
    if f3:
        lines += [
            f"Yes. High-value orders from new customers have a {f3['new_high_value_fraud_rate_pct']:.1f}% fraud "
            f"rate (overall {f3['overall_fraud_rate_pct']:.1f}%). Reviewing only these "
            f"{f3['new_high_value_orders']:,} orders ({f3['new_high_value_share_of_orders_pct']:.1f}% of all orders) "
            f"would cover {f3['new_high_value_share_of_fraud_pct']:.1f}% of all fraud.",
            "Decision: put high-value orders from new customers at the top of the review queue.",
        ]
    lines += ["", "Note: mobile_verified and first_order_datetime come from the customer snapshot; "
                  "their exact timing is unconfirmed (team Assumption 19)."]
    return "\n".join(lines)


def save_outputs(counts, overview, q1, q2, q3, output_dir):
    """Write tables (CSV), charts (PNG) and the summary (Markdown). Returns the file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "q1_by_segment": out / "q1_fraud_by_customer_segment.csv",
        "q1_by_age_band": out / "q1_fraud_by_account_age.csv",
        "q2_table": out / "q2_fraud_by_mobile_verification.csv",
        "q3_table": out / "q3_fraud_by_order_value.csv",
        "q3_cutoffs": out / "q3_high_value_cutoffs_by_country.csv",
        "q1_chart": out / "q1_fraud_by_account_age.png",
        "q2_chart": out / "q2_fraud_by_mobile_verification.png",
        "q3_chart": out / "q3_fraud_by_order_value.png",
        "summary": out / "summary.md",
    }
    q1["by_segment"].round(2).to_csv(files["q1_by_segment"], index=False)
    q1["by_age_band"].round(2).to_csv(files["q1_by_age_band"], index=False)
    q2["table"].round(2).to_csv(files["q2_table"], index=False)
    q3["table"].round(2).to_csv(files["q3_table"], index=False)
    q3["cutoffs"].round(2).to_csv(files["q3_cutoffs"], index=False)
    sns.set_theme(style="whitegrid")
    plot_q1(q1, 100 * overview["frauds"] / overview["orders"], files["q1_chart"])
    plot_q2(q2, files["q2_chart"])
    plot_q3(q3, files["q3_chart"])
    files["summary"].write_text(summary_text(counts, overview, q1, q2, q3), encoding="utf-8")
    return files


def analyse(counts, overview, cutoffs, output_dir=DEFAULT_OUTPUT_DIR):
    """Answer Q1-Q3 from the group counts and save everything."""
    q1, q2, q3 = q1_account_age(counts), q2_mobile_verification(counts), q3_high_value_orders(counts, cutoffs)
    files = save_outputs(counts, overview, q1, q2, q3, output_dir)
    return {"counts": counts, "overview": overview, "q1": q1, "q2": q2, "q3": q3, "files": files}


def run_analysis(output_dir=DEFAULT_OUTPUT_DIR, engine=None):
    """Read the latest counts from RDS, answer Q1-Q3 and save everything."""
    return analyse(*load_counts(engine), output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description="New customer risk analytics from RDS (Task #62)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to save tables and charts")
    args = parser.parse_args()
    result = run_analysis(args.output_dir)
    print(summary_text(result["counts"], result["overview"], result["q1"], result["q2"], result["q3"]))
    print(f"\nSaved tables and charts to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
