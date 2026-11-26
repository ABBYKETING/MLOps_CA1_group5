"""Generate visualisations for the model comparison results."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def model_labels(df):
    return (
        df["team_member"] + " - "
        + df["algorithm"].str.replace("_", " ")
    )


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_safe_full(df, output_dir):
    df = df.sort_values(
        "test_pr_auc_delta", ascending=True
    )

    labels = model_labels(df)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 6))

    safe_bars = ax.barh(
        y - 0.18,
        df["test_pr_auc_safe"],
        height=0.35,
        label="Safe",
        color="#4472C4",
    )

    full_bars = ax.barh(
        y + 0.18,
        df["test_pr_auc_full"],
        height=0.35,
        label="Full (exploratory)",
        color="#ED9B40",
    )

    ax.bar_label(safe_bars, fmt="%.3f", padding=4)
    ax.bar_label(full_bars, fmt="%.3f", padding=4)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Test Average Precision (AP)")
    ax.set_title("Safe vs Full Feature Sets")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(axis="x", alpha=0.2)

    save_plot(
        fig, output_dir / "01_safe_vs_full_ap.png"
    )


def plot_primary_comparison(df, output_dir):
    df = df.sort_values("val_recall", ascending=True)

    labels = model_labels(df)

    colors = ["#4472C4"] * len(df)
    colors[-2:] = ["#2E8B57", "#2E8B57"]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(
        labels,
        df["val_recall"] * 100,
        color=colors,
    )

    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set_xlim(0, 20)
    ax.set_xlabel("Validation Recall (%)")
    ax.set_title(
        "Safe Model Comparison at Minimum 70% Precision"
    )
    ax.grid(axis="x", alpha=0.2)

    save_plot(
        fig, output_dir / "02_primary_model_ranking.png"
    )


def plot_sensitivity(df, output_dir):
    fig, ax = plt.subplots(figsize=(11, 7))

    for (member, algorithm), group in df.groupby(
        ["team_member", "algorithm"]
    ):

        group = group.sort_values("precision_target")

        label = (
            f"{member} - "
            f"{algorithm.replace('_', ' ')}"
        )

        ax.plot(
            group["precision_target"] * 100,
            group["val_recall"] * 100,
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.axvline(
        70,
        color="grey",
        linestyle="--",
        label="Primary policy",
    )

    ax.set_xticks([60, 70, 80])
    ax.set_xlabel("Minimum Validation Precision (%)")
    ax.set_ylabel("Maximum Validation Recall (%)")
    ax.set_title("Precision Target Sensitivity Analysis")
    ax.set_ylim(0, 50)
    ax.grid(alpha=0.2)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_plot(
        fig, output_dir / "03_threshold_sensitivity.png"
    )


def plot_generalisation(df, output_dir):
    df = df.sort_values("rank")

    labels = model_labels(df)
    x = np.arange(len(df))

    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("pr_auc", "Average Precision (Threshold-independent)"),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5)
    )

    for ax, (key, title) in zip(axes, metrics):

        val_bars = ax.bar(
            x - 0.18,
            df[f"val_{key}"] * 100,
            width=0.35,
            label="Validation",
            color="#4472C4",
        )

        test_bars = ax.bar(
            x + 0.18,
            df[f"test_{key}"] * 100,
            width=0.35,
            label="Test",
            color="#ED9B40",
        )

        ax.bar_label(val_bars, fmt="%.1f%%", fontsize=8)
        ax.bar_label(test_bars, fmt="%.1f%%", fontsize=8)

        if key == "precision":
            ax.axhline(
                70,
                color="red",
                linestyle="--",
                label="70% target",
            )

        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Top Two Models: Validation vs Test "
        "at Locked Thresholds"
    )

    save_plot(
        fig, output_dir / "04_top_two_generalisation.png"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/model_comparison"),
    )

    args = parser.parse_args()
    source = args.output_dir

    safe_full = pd.read_csv(
        source / "safe_full_comparison.csv"
    )

    primary = pd.read_csv(
        source / "safe_model_comparison.csv"
    )

    sensitivity = pd.read_csv(
        source / "threshold_sensitivity.csv"
    )

    top_two = pd.read_csv(
        source / "top_two_test_results.csv"
    )

    if (
        len(safe_full) != 8
        or len(primary) != 8
        or len(sensitivity) != 24
        or len(top_two) != 2
    ):
        raise ValueError("Unexpected comparison row counts.")

    if not primary["target_achieved"].eq(True).all():
        raise ValueError(
            "This baseline plot expects eight feasible candidates."
        )

    plot_dir = source / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_safe_full(safe_full, plot_dir)
    plot_primary_comparison(primary, plot_dir)
    plot_sensitivity(sensitivity, plot_dir)
    plot_generalisation(top_two, plot_dir)

    print("\nAll four comparison plots generated successfully.")


if __name__ == "__main__":
    main()
