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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import database as db  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analytics_emi"   # git-ignored

NEW_CUSTOMER_DAYS = 30
HIGH_VALUE_QUANTILE = 0.75
NEW, ESTABLISHED, UNKNOWN = "New (0-30 days)", "Established (>30 days)", "Unknown age"
HIGH, NORMAL = "High value (top 25%)", "Normal value"
AGE_BANDS = [(-np.inf, 0, "Same day"), (0, 7, "1-7 days"), (7, 30, "8-30 days"),
             (30, 90, "31-90 days"), (90, 180, "91-180 days"), (180, 365, "181-365 days"),
             (365, np.inf, "Over 1 year")]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_orders(engine=None):
    """Latest labelled orders with order and customer details, straight from RDS."""
    return db.load_training_data(engine)


def add_customer_segments(orders):
    """Add account_age_days, customer_segment, age_band and value_segment."""
    df = orders.copy()
    age = (df["order_date"] - df["first_order_datetime"].dt.normalize()).dt.days
    df["account_age_days"] = age
    known = age.notna() & (age >= 0)
    df["customer_segment"] = np.select(
        [known & (age <= NEW_CUSTOMER_DAYS), known & (age > NEW_CUSTOMER_DAYS)],
        [NEW, ESTABLISHED], default=UNKNOWN)
    band = pd.Series(pd.NA, index=df.index, dtype="object")
    for low, high, label in AGE_BANDS:
        band[known & (age > low) & (age <= high)] = label
    df["age_band"] = pd.Categorical(band, categories=[b[2] for b in AGE_BANDS], ordered=True)
    cutoff = df.groupby("country_code")["order_value"].transform(
        lambda values: values.quantile(HIGH_VALUE_QUANTILE))
    df["value_segment"] = np.where(df["order_value"] >= cutoff, HIGH, NORMAL)
    return df


def wilson_interval(frauds, orders, z=1.96):
    """95% confidence interval for a fraud rate (works for small groups too)."""
    frauds, orders = np.asarray(frauds, float), np.asarray(orders, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = frauds / orders
        centre = (rate + z**2 / (2 * orders)) / (1 + z**2 / orders)
        half = z * np.sqrt(rate * (1 - rate) / orders + z**2 / (4 * orders**2)) / (1 + z**2 / orders)
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def fraud_summary(df, by):
    """Orders, frauds, fraud rate (%), its 95% interval and share of all fraud, per group."""
    table = (df.groupby(by, observed=True)["is_fraud"]
               .agg(orders="size", frauds="sum").reset_index())
    low, high = wilson_interval(table["frauds"], table["orders"])
    table["fraud_rate_pct"] = 100 * table["frauds"] / table["orders"]
    table["ci_low_pct"], table["ci_high_pct"] = 100 * low, 100 * high
    table["share_of_orders_pct"] = 100 * table["orders"] / len(df)
    table["share_of_all_fraud_pct"] = 100 * table["frauds"] / df["is_fraud"].sum()
    return table


# ---------------------------------------------------------------------------
# The three questions
# ---------------------------------------------------------------------------
def q1_account_age(df):
    """Q1: fraud rate for new vs established customers, and by account age band."""
    by_segment = fraud_summary(df, "customer_segment")
    by_band = fraud_summary(df[df["customer_segment"] != UNKNOWN], "age_band")
    rate = by_segment.set_index("customer_segment")["fraud_rate_pct"]
    share = by_segment.set_index("customer_segment")
    findings = {
        "new_fraud_rate_pct": rate.get(NEW, np.nan),
        "established_fraud_rate_pct": rate.get(ESTABLISHED, np.nan),
        "times_higher": rate.get(NEW, np.nan) / rate.get(ESTABLISHED, np.nan),
        "new_share_of_orders_pct": share.loc[NEW, "share_of_orders_pct"] if NEW in share.index else np.nan,
        "new_share_of_fraud_pct": share.loc[NEW, "share_of_all_fraud_pct"] if NEW in share.index else np.nan,
    }
    return {"by_segment": by_segment, "by_age_band": by_band, "findings": findings}


def q2_mobile_verification(df):
    """Q2: fraud rate by mobile verification, for new and established customers."""
    known = df[df["customer_segment"] != UNKNOWN]
    table = fraud_summary(known, ["customer_segment", "mobile_verified"])
    new = table[table["customer_segment"] == NEW].set_index("mobile_verified")
    new_customers = known["customer_segment"] == NEW
    findings = {
        "new_verified_pct": 100 * known.loc[new_customers, "mobile_verified"].mean(),
        "new_unverified_orders": int(new.loc[False, "orders"]) if False in new.index else 0,
        "new_unverified_fraud_rate_pct": new.loc[False, "fraud_rate_pct"] if False in new.index else np.nan,
        "new_verified_fraud_rate_pct": new.loc[True, "fraud_rate_pct"] if True in new.index else np.nan,
    }
    return {"table": table, "findings": findings}


def q3_high_value_orders(df):
    """Q3: fraud rate of high-value vs normal orders, for new and established customers."""
    known = df[df["customer_segment"] != UNKNOWN]
    table = fraud_summary(known, ["customer_segment", "value_segment"])
    baseline = table.query("customer_segment == @ESTABLISHED and value_segment == @NORMAL")["fraud_rate_pct"]
    table["times_established_normal"] = table["fraud_rate_pct"] / (baseline.iloc[0] if len(baseline) else np.nan)
    cutoffs = (df.groupby("country_code")["order_value"].quantile(HIGH_VALUE_QUANTILE)
                 .rename("high_value_cutoff").reset_index())
    pick = table.set_index(["customer_segment", "value_segment"])
    findings = {}
    if (NEW, HIGH) in pick.index:
        row = pick.loc[(NEW, HIGH)]
        findings = {
            "new_high_value_fraud_rate_pct": row["fraud_rate_pct"],
            "new_high_value_orders": int(row["orders"]),
            "new_high_value_share_of_orders_pct": 100 * row["orders"] / len(df),
            "new_high_value_share_of_fraud_pct": 100 * row["frauds"] / df["is_fraud"].sum(),
            "overall_fraud_rate_pct": 100 * df["is_fraud"].mean(),
        }
    return {"table": table, "cutoffs": cutoffs, "findings": findings}


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
    table = result["by_age_band"].sort_values("age_band")
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#c0392b" if band in ("Same day", "1-7 days", "8-30 days") else "#7f8c8d"
               for band in table["age_band"]]
    ax.bar(table["age_band"].astype(str), table["fraud_rate_pct"], color=colours)
    for i, row in enumerate(table.itertuples()):
        ax.annotate(f"{row.fraud_rate_pct:.1f}%", (i, row.fraud_rate_pct), ha="center", va="bottom",
                    fontsize=9, xytext=(0, 2), textcoords="offset points")
    ax.axhline(overall_rate, color="black", linestyle="--", linewidth=1)
    ax.text(2.6, overall_rate, f"overall {overall_rate:.1f}%", va="bottom", ha="left", fontsize=9)
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
def summary_text(df, q1, q2, q3):
    """Plain-English findings for analysts (also shown in the Prefect flow)."""
    f1, f2, f3 = q1["findings"], q2["findings"], q3["findings"]
    unknown = int((df["customer_segment"] == UNKNOWN).sum())
    lines = [
        "# New customer risk analytics",
        "",
        f"Data: {len(df):,} labelled orders from RDS, order dates {df['order_date'].min():%d %b %Y} "
        f"to {df['order_date'].max():%d %b %Y}; overall fraud rate {100 * df['is_fraud'].mean():.1f}%. "
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


def save_outputs(df, q1, q2, q3, output_dir):
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
    plot_q1(q1, 100 * df["is_fraud"].mean(), files["q1_chart"])
    plot_q2(q2, files["q2_chart"])
    plot_q3(q3, files["q3_chart"])
    files["summary"].write_text(summary_text(df, q1, q2, q3), encoding="utf-8")
    return files


def run_analysis(output_dir=DEFAULT_OUTPUT_DIR, engine=None):
    """Load the latest data from RDS, answer Q1-Q3 and save everything."""
    df = add_customer_segments(load_orders(engine))
    q1, q2, q3 = q1_account_age(df), q2_mobile_verification(df), q3_high_value_orders(df)
    files = save_outputs(df, q1, q2, q3, output_dir)
    return {"data": df, "q1": q1, "q2": q2, "q3": q3, "files": files}


def main():
    parser = argparse.ArgumentParser(description="New customer risk analytics from RDS (Task #62)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to save tables and charts")
    args = parser.parse_args()
    result = run_analysis(args.output_dir)
    print(summary_text(result["data"], result["q1"], result["q2"], result["q3"]))
    print(f"\nSaved tables and charts to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
