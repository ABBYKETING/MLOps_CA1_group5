"""Chu, Task #63: payment and order-pattern analytics from the latest RDS data.

Q1: payment-method fraud rates with transaction volumes and uncertainty.
Q2: fraud rates within payment methods across fixed SGD order-value ranges.
Q3: delivery/pickup comparisons within payment methods.

All queries use labelled orders as the denominator. SQL aggregates inside the
database; only anonymous group counts are returned. Run from the repository:
    python src/analytics_payment_order_chu.py
Connection settings come from the existing DB_* environment variables.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import database as db

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "analytics_chu"
VALUE_BANDS = ["SGD 0", "(0, 10]", "(10, 25]", "(25, 50]", "(50, 100]", "> SGD 100"]
MIN_HIGHLIGHT_ORDERS = 100  # reporting support rule; all groups remain in tables/plots
JOIN = "FROM fraud_labels l JOIN orders o ON o.country_code=l.country_code AND o.order_id=l.order_id"
VALUE_CASE = """CASE WHEN o.order_value=0 THEN 'SGD 0'
    WHEN o.order_value<=10 THEN '(0, 10]' WHEN o.order_value<=25 THEN '(10, 25]'
    WHEN o.order_value<=50 THEN '(25, 50]' WHEN o.order_value<=100 THEN '(50, 100]'
    ELSE '> SGD 100' END"""
GROUP_QUERY = f"""
    SELECT o.country_code, o.payment_method, o.collect_type, {VALUE_CASE} AS value_band,
           COUNT(*) AS orders, SUM(l.is_fraud) AS frauds,
           MIN(o.order_date) AS first_order_date, MAX(o.order_date) AS last_order_date
    {JOIN}
    GROUP BY o.country_code, o.payment_method, o.collect_type, value_band
"""
AUDIT_QUERY = """
    SELECT COUNT(*) AS joined_labels,
           SUM(CASE WHEN o.order_id IS NULL THEN 1 ELSE 0 END) AS missing_orders,
           SUM(CASE WHEN l.is_fraud IS NULL OR l.is_fraud NOT IN (0,1)
                     OR o.order_value IS NULL OR o.order_value<0
                     OR o.payment_method IS NULL OR TRIM(o.payment_method)=''
                     OR o.collect_type IS NULL OR o.collect_type NOT IN ('Delivery','Pickup')
                     OR o.order_date IS NULL THEN 1 ELSE 0 END) AS invalid_rows
    FROM fraud_labels l LEFT JOIN orders o
      ON o.country_code=l.country_code AND o.order_id=l.order_id
"""


def load_counts(engine=None):
    """Read fresh RDS aggregates. Caller-owned engines are left open."""
    owned = engine is None
    engine = db.get_engine() if owned else engine
    started = time.perf_counter()
    try:
        with engine.connect() as connection:
            labels = int(connection.execute(text("SELECT COUNT(*) FROM fraud_labels")).scalar())
            total_orders = int(connection.execute(text("SELECT COUNT(*) FROM orders")).scalar())
            audit = dict(connection.execute(text(AUDIT_QUERY)).mappings().one())
            if labels == 0:
                raise ValueError("RDS contains no labelled orders for analysis")
            if audit["joined_labels"] != labels or audit["missing_orders"] or audit["invalid_rows"]:
                raise ValueError("RDS labels/orders contain unmatched, duplicate or invalid records; check ETL")
            groups = pd.read_sql(text(GROUP_QUERY), connection)
    finally:
        if owned:
            engine.dispose()
    for column in ("orders", "frauds"):
        groups[column] = groups[column].astype(int)
    if int(groups["orders"].sum()) != labels:
        raise ValueError("Aggregate counts differ from the labelled population; retry if ETL is updating data")
    overview = {
        "source": "team RDS business database; fraud_labels joined to orders",
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "orders": labels,
        "frauds": int(groups["frauds"].sum()), "all_orders": total_orders,
        "unlabelled_orders": total_orders - labels,
        "first_order_date": str(pd.to_datetime(groups["first_order_date"]).min().date()),
        "last_order_date": str(pd.to_datetime(groups["last_order_date"]).max().date()),
        "countries": sorted(groups["country_code"].unique().tolist()),
        "read_seconds": time.perf_counter() - started,
    }
    return groups.drop(columns=["first_order_date", "last_order_date"]), overview


def wilson_interval(frauds, orders):
    """95% Wilson intervals; a zero-order cell has no estimable rate."""
    x, n = np.asarray(frauds, dtype=float), np.asarray(orders, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = x / n
        z = 1.959963984540054
        centre = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return np.clip(centre-half, 0, 1), np.clip(centre+half, 0, 1)


def summarise(groups, by):
    table = groups.groupby(by, observed=True, dropna=False)[["orders", "frauds"]].sum().reset_index()
    low, high = wilson_interval(table["frauds"], table["orders"])
    table["fraud_rate_pct"] = 100 * table["frauds"] / table["orders"]
    table["ci_low_pct"], table["ci_high_pct"] = 100*low, 100*high
    table["share_orders_pct"] = 100*table["orders"] / groups["orders"].sum()
    total_fraud = groups["frauds"].sum()
    table["share_fraud_pct"] = 100*table["frauds"] / total_fraud if total_fraud else np.nan
    baseline = total_fraud / groups["orders"].sum()
    table["risk_lift"] = table["fraud_rate_pct"]/(100*baseline) if baseline else np.nan
    table["highlight_support"] = table["orders"] >= MIN_HIGHLIGHT_ORDERS
    return table


def query_payment_methods(groups):
    """Q1: order-weighted rates, volume, fraud contribution, and country detail."""
    overall = summarise(groups, ["payment_method"])
    return overall.sort_values(["fraud_rate_pct", "orders", "payment_method"],
                               ascending=[False, False, True]).reset_index(drop=True)


def query_order_values(groups):
    """Q2: common SGD boundaries across methods, including valid zero-value orders."""
    table = summarise(groups, ["payment_method", "value_band"])
    baseline = summarise(groups, ["payment_method"])[["payment_method", "fraud_rate_pct"]]
    table = table.merge(baseline.rename(columns={"fraud_rate_pct": "method_baseline_pct"}),
                        on="payment_method", validate="many_to_one")
    table["within_method_lift"] = table["fraud_rate_pct"].div(table["method_baseline_pct"].replace(0, np.nan))
    table["value_band"] = pd.Categorical(table["value_band"], categories=VALUE_BANDS, ordered=True)
    return table.sort_values(["payment_method", "value_band"]).reset_index(drop=True)


def query_collection_types(groups):
    """Q3: compare delivery and pickup within each payment method."""
    return summarise(groups, ["payment_method", "collect_type"])


def collection_gaps(table):
    """Delivery minus pickup percentage points; unmatched pairs remain undefined."""
    keys = [c for c in ("country_code", "payment_method") if c in table]
    fields = ["orders", "frauds", "fraud_rate_pct", "ci_low_pct", "ci_high_pct"]
    delivery = table[table["collect_type"] == "Delivery"][keys + fields]
    pickup = table[table["collect_type"] == "Pickup"][keys + fields]
    pairs = delivery.merge(pickup, on=keys, how="outer", suffixes=("_delivery", "_pickup"), validate="one_to_one")
    for field in ("orders", "frauds"):
        for mode in ("delivery", "pickup"):
            pairs[f"{field}_{mode}"] = pairs[f"{field}_{mode}"].fillna(0).astype(int)
    pairs["delivery_minus_pickup_pp"] = pairs["fraud_rate_pct_delivery"] - pairs["fraud_rate_pct_pickup"]
    pairs["both_groups_supported"] = ((pairs["orders_delivery"] >= MIN_HIGHLIGHT_ORDERS)
                                      & (pairs["orders_pickup"] >= MIN_HIGHLIGHT_ORDERS))
    return pairs.sort_values(keys).reset_index(drop=True)


def analyse(groups):
    """Compute three questions from the same fresh aggregate snapshot."""
    q3 = query_collection_types(groups)
    country_q3 = summarise(groups, ["country_code", "payment_method", "collect_type"])
    return {
        "q1_payment_methods": query_payment_methods(groups),
        "q1_country_payment": summarise(groups, ["country_code", "payment_method"]),
        "q2_payment_value": query_order_values(groups),
        "q2_country_payment_value": summarise(groups, ["country_code", "payment_method", "value_band"]),
        "q3_payment_collection": q3, "q3_country_payment_collection": country_q3,
        "q3_delivery_pickup_gaps": collection_gaps(q3),
        "q3_country_delivery_pickup_gaps": collection_gaps(country_q3),
    }


def insights(tables, overview):
    """Findings are recomputed from results, including when patterns change."""
    rate = 100*overview["frauds"]/overview["orders"]
    lines = ["# Payment and order-pattern analytics - Chu", "",
             f"Data: {overview['orders']:,} labelled orders; {overview['frauds']:,} fraud labels ({rate:.2f}%).",
             f"Order dates: {overview['first_order_date']} to {overview['last_order_date']}. "
             f"Unlabelled orders excluded: {overview['unlabelled_orders']:,}.", "",
             "## Q1. Which payment methods merit closer monitoring?", ""]
    q1 = tables["q1_payment_methods"]
    supported = q1[q1["highlight_support"]]
    if not supported.empty:
        top = supported.sort_values(["fraud_rate_pct", "orders", "payment_method"], ascending=[False, False, True]).iloc[0]
        lines.append(f"Among methods with at least {MIN_HIGHLIGHT_ORDERS} labelled orders, **{top.payment_method}** "
                     f"has the highest observed fraud rate: {top.fraud_rate_pct:.2f}% "
                     f"({top.frauds:,}/{top.orders:,}; 95% interval {top.ci_low_pct:.2f}-{top.ci_high_pct:.2f}%).")
    volume = q1.sort_values(["frauds", "orders", "payment_method"], ascending=[False, False, True]).iloc[0]
    if overview["frauds"]:
        lines.append(f"**{volume.payment_method}** contributes the most fraud-labelled orders: {volume.frauds:,} "
                     f"({volume.share_fraud_pct:.2f}% of observed fraud), with a {volume.fraud_rate_pct:.2f}% rate.")
    lines += ["Decision: use both fraud concentration and transaction volumes to prioritise review; "
              "check the country breakdown before changing payment-method controls.", "",
              "## Q2. Which payment/value combinations merit review?", ""]
    q2 = tables["q2_payment_value"]
    eligible = q2[q2["highlight_support"] & q2["within_method_lift"].notna()]
    if not eligible.empty:
        top = eligible.sort_values(["within_method_lift", "orders", "payment_method", "value_band"],
                                   ascending=[False, False, True, True]).iloc[0]
        lines.append(f"The largest within-method lift among supported groups is **{top.payment_method}, {top.value_band}**: "
                     f"{top.fraud_rate_pct:.2f}% fraud ({top.frauds:,}/{top.orders:,}), "
                     f"{top.within_method_lift:.2f} times that method's {top.method_baseline_pct:.2f}% baseline.")
    lines += ["Decision: investigate the supported payment/value combinations with elevated rates and meaningful "
              "fraud counts; a larger order is not automatically riskier. Consult country-specific results and intervals.", "",
              "## Q3. Does collection type change the observed pattern?", ""]
    pairs = tables["q3_delivery_pickup_gaps"]
    comparable = pairs[pairs["both_groups_supported"]].copy()
    if not comparable.empty:
        comparable["absolute_gap"] = comparable["delivery_minus_pickup_pp"].abs()
        top = comparable.sort_values(["absolute_gap", "payment_method"], ascending=[False, True]).iloc[0]
        mode = "Delivery" if top.delivery_minus_pickup_pp > 0 else "Pickup" if top.delivery_minus_pickup_pp < 0 else "Neither mode"
        lines.append(f"The largest absolute gap with both groups supported is **{top.payment_method}**: "
                     f"Delivery {top.fraud_rate_pct_delivery:.2f}% (n={top.orders_delivery:,}) versus "
                     f"Pickup {top.fraud_rate_pct_pickup:.2f}% (n={top.orders_pickup:,}). "
                     f"{mode} has the higher point estimate; signed Delivery-minus-Pickup gap "
                     f"{top.delivery_minus_pickup_pp:+.2f} percentage points.")
    else:
        lines.append("No payment method has at least 100 labelled orders in both collection groups; "
                     "the available comparisons are descriptive and should not drive firm collection-type decisions.")
    lines += ["Decision: review payment/collection combinations rather than treating all delivery or pickup orders "
              "as equally risky. A missing comparison group means insufficient evidence, not zero risk.", "",
              "## Definitions and interpretation limits", "",
              "- Fraud rate = fraud-labelled orders / labelled orders in the same group; counts are order-weighted.",
              "- Intervals are 95% Wilson intervals for descriptive binomial uncertainty; repeated customers and "
              "non-random labels mean they are not a guarantee about future populations.",
              "- The 100-order rule controls highlighted examples only. All groups, including rare methods, remain in tables and charts.",
              "- Value ranges use SGD as specified in the brief. Zero-value orders are valid and retained separately.",
              "- `unknown` is an unrecorded-payment segment; `no payment` is retained as a valid recorded category.",
              "- Payment methods have different country mixes. Country tables support stratified checking; pooled gaps are not causal effects.",
              "- These are retrospective labelled-order associations, not fraud predictions, proof of causation or instructions to block transactions."]
    return "\n".join(lines)


def plot_outputs(tables, folder):
    sns.set_theme(style="whitegrid", context="notebook")
    q1 = tables["q1_payment_methods"].sort_values(["fraud_rate_pct", "payment_method"])
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(q1)*0.4)), sharey=True)
    y = np.arange(len(q1))
    axes[0].errorbar(q1["fraud_rate_pct"], y,
                     xerr=np.maximum(0, np.vstack([q1.fraud_rate_pct-q1.ci_low_pct, q1.ci_high_pct-q1.fraud_rate_pct])),
                     fmt="o", capsize=3, color="#236b8e")
    axes[0].set_yticks(y, q1.payment_method)
    axes[0].set(xlabel="Observed fraud rate (%) - 95% Wilson interval", title="Q1. Rates and uncertainty", xlim=(-1, 101))
    bars = axes[1].barh(y, q1.orders, color="#5eaaa8")
    axes[1].set(xlabel="Labelled orders (log scale)", xscale="log", title="Transaction volumes")
    axes[1].set_xlim(0.5, q1.orders.max()*4)
    axes[1].bar_label(bars, labels=[f"{n:,}" for n in q1.orders], padding=3, fontsize=8)
    fig.tight_layout(); fig.savefig(folder/"q1_payment_methods.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    methods = q1.payment_method.tolist()[::-1]
    q2 = tables["q2_payment_value"]
    rates = q2.pivot(index="payment_method", columns="value_band", values="fraud_rate_pct").reindex(index=methods, columns=VALUE_BANDS)
    counts = q2.pivot(index="payment_method", columns="value_band", values="orders").reindex(index=methods, columns=VALUE_BANDS)
    fig, ax = plt.subplots(figsize=(14, max(7, len(methods)*0.49)))
    sns.heatmap(rates, mask=rates.isna(), cmap="YlOrRd", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "Observed fraud rate (%)"}, linewidths=0.4)
    ax.grid(False)
    for i in range(len(methods)):
        for j in range(len(VALUE_BANDS)):
            value, n = rates.iloc[i,j], counts.iloc[i,j]
            label = f"{value:.1f}%\nn={int(n):,}" if pd.notna(value) else "-"
            ax.text(j+0.5, i+0.5, label, ha="center", va="center", fontsize=7,
                    color="white" if pd.notna(value) and value>55 else "black")
    ax.set(title="Q2. Payment method and order value\nCell = fraud rate and labelled-order count; - = no observations",
           xlabel="Order-value range (SGD)", ylabel="Payment method")
    fig.tight_layout(); fig.savefig(folder/"q2_payment_value.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    q3 = tables["q3_payment_collection"]
    fig, ax = plt.subplots(figsize=(13, max(7, len(methods)*0.45)))
    for mode, offset, colour, marker in [("Delivery", -0.15, "#236b8e", "o"), ("Pickup", 0.15, "#c67825", "s")]:
        frame = q3[q3.collect_type==mode].set_index("payment_method").reindex(methods)
        valid = frame.orders.notna()
        frame = frame[valid]
        ax.errorbar(frame.fraud_rate_pct, np.arange(len(methods))[valid]+offset,
                    xerr=np.maximum(0, np.vstack([frame.fraud_rate_pct-frame.ci_low_pct, frame.ci_high_pct-frame.fraud_rate_pct])),
                    fmt=marker, capsize=2, color=colour, label=mode)
    ax.set_yticks(np.arange(len(methods)), methods)
    ax.set(title="Q3. Delivery versus pickup within payment methods\n95% Wilson intervals; absent groups have no point",
           xlabel="Observed fraud rate (%)", ylabel="Payment method", xlim=(-1,101))
    ax.invert_yaxis()
    ax.legend(); fig.tight_layout(); fig.savefig(folder/"q3_payment_collection.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def save_outputs(tables, overview, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder/f"{name}.csv", index=False)
    plot_outputs(tables, folder)
    summary = insights(tables, overview)
    (folder/"summary.md").write_text(summary, encoding="utf-8")
    return summary


def run_analysis(output_dir=DEFAULT_OUTPUT_DIR, engine=None):
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    folder = Path(output_dir).resolve()/run_id
    folder.mkdir(parents=True, exist_ok=False)
    groups, overview = load_counts(engine)
    tables = analyse(groups)
    summary = save_outputs(tables, overview, folder)
    overview.update(run_id=run_id, duration_seconds=time.perf_counter()-started,
                    value_bands_sgd=VALUE_BANDS, highlight_min_orders=MIN_HIGHLIGHT_ORDERS)
    (folder/"metadata.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    return {"output_dir": str(folder), "overview": overview, "tables": tables, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    try:
        result = run_analysis(args.output_dir)
    except Exception as error:
        message = str(error)
        for key in ("DB_PASSWORD", "DB_HOST"):
            if os.environ.get(key):
                message = message.replace(os.environ[key], "[REDACTED]")
        print(f"Payment analytics failed: {type(error).__name__}: {message}", file=sys.stderr)
        return 1
    print(result["summary"])
    print(f"\nSaved results to {result['output_dir']}")
    print(f"Execution time: {result['overview']['duration_seconds']:.2f} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
