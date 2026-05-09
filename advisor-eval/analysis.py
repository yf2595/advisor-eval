"""Post-run analysis: aggregate metrics, summary tables, and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from logger import TaskLogger


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load all JSONL result files into a single DataFrame."""
    records = TaskLogger.read_results_dir(results_dir)
    if not records:
        print(f"No results found in {results_dir}")
        return pd.DataFrame()
    return pd.DataFrame(records)


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-method aggregate metrics."""
    if df.empty:
        return df

    agg = df.groupby("method").agg(
        accuracy=("correct", "mean"),
        cost_mean=("cost_total", "mean"),
        cost_total=("cost_total", "sum"),
        latency_mean=("latency_total_s", "mean"),
        latency_median=("latency_total_s", "median"),
        latency_p95=("latency_total_s", lambda x: np.percentile(x, 95)),
        advisor_calls_mean=("advisor_calls", "mean"),
        steps_mean=("steps", "mean"),
        n_tasks=("task_id", "count"),
    ).reset_index()

    agg["accuracy"] = (agg["accuracy"] * 100).round(2)
    agg["cost_mean"] = agg["cost_mean"].round(6)
    agg["latency_mean"] = agg["latency_mean"].round(3)
    agg["latency_median"] = agg["latency_median"].round(3)
    agg["latency_p95"] = agg["latency_p95"].round(3)
    agg["advisor_calls_mean"] = agg["advisor_calls_mean"].round(2)
    agg["steps_mean"] = agg["steps_mean"].round(2)

    if "tool_calls" in df.columns:
        agg_tool_calls = df.groupby("method")["tool_calls"].mean().rename("tool_calls_mean")
        agg = agg.merge(agg_tool_calls, on="method", how="left")
        agg["tool_calls_mean"] = agg["tool_calls_mean"].fillna(0.0).round(2)

    if "tool_errors" in df.columns:
        agg_tool_errors = df.groupby("method")["tool_errors"].mean().rename("tool_errors_mean")
        agg = agg.merge(agg_tool_errors, on="method", how="left")
        agg["tool_errors_mean"] = agg["tool_errors_mean"].fillna(0.0).round(2)

    if "dead_end_count" in df.columns:
        agg_dead_ends = df.groupby("method")["dead_end_count"].mean().rename("dead_ends_mean")
        agg = agg.merge(agg_dead_ends, on="method", how="left")
        agg["dead_ends_mean"] = agg["dead_ends_mean"].fillna(0.0).round(2)

    return agg


def compute_recovery_rate(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute advisor recovery rate by splitting into easy/hard tasks.

    Easy: cheap_only got it right.  Hard: cheap_only got it wrong.
    Recovery = fraction of hard tasks the advisor method got right.
    """
    cheap = df[df["method"] == "cheap_only"]
    if cheap.empty:
        return None

    cheap_correct_ids = set(cheap[cheap["correct"]]["task_id"])
    advisor_methods = [m for m in df["method"].unique() if m.startswith("advisor_")]

    rows = []
    for method in advisor_methods:
        adv = df[df["method"] == method]
        hard = adv[~adv["task_id"].isin(cheap_correct_ids)]
        easy = adv[adv["task_id"].isin(cheap_correct_ids)]

        if not hard.empty:
            recovery = hard["correct"].mean() * 100
            hard_latency = hard["latency_total_s"].mean()
        else:
            recovery = 0.0
            hard_latency = 0.0

        easy_latency = easy["latency_total_s"].mean() if not easy.empty else 0.0

        rows.append({
            "method": method,
            "total_hard": len(hard),
            "recovered": int(hard["correct"].sum()) if not hard.empty else 0,
            "recovery_rate_%": round(recovery, 2),
            "hard_latency_mean_s": round(hard_latency, 3),
            "easy_latency_mean_s": round(easy_latency, 3),
        })

    return pd.DataFrame(rows) if rows else None


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

PLOTS_DIR = Path("plots")


def _ensure_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_three_axis_summary(df: pd.DataFrame) -> None:
    """Grouped bar chart: accuracy, cost, latency for each method (Experiment 1)."""
    _ensure_plots_dir()
    agg = aggregate_metrics(df)
    if agg.empty:
        return

    methods = agg["method"].tolist()
    x = np.arange(len(methods))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars1 = ax1.bar(x - width, agg["accuracy"], width, label="Accuracy (%)", color="#2196F3")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Method")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, agg["cost_mean"] * 1000, width, label="Cost (m$)", color="#FF9800", alpha=0.8)
    bars3 = ax2.bar(x + width, agg["latency_mean"], width, label="Latency (s)", color="#4CAF50", alpha=0.8)
    ax2.set_ylabel("Cost (m$) / Latency (s)")

    lines = [bars1, bars2, bars3]
    labels = [b.get_label() for b in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Core Comparison: Accuracy, Cost, and Latency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "three_axis_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'three_axis_summary.png'}")


def plot_cost_vs_accuracy(df: pd.DataFrame) -> None:
    """Scatter plot: cost vs accuracy across methods/thresholds."""
    _ensure_plots_dir()
    agg = aggregate_metrics(df)
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(agg["cost_mean"] * 1000, agg["accuracy"], s=80, zorder=5)

    for _, row in agg.iterrows():
        ax.annotate(row["method"], (row["cost_mean"] * 1000, row["accuracy"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=8)

    ax.set_xlabel("Mean Cost per Task (m$)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost vs. Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cost_vs_accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'cost_vs_accuracy.png'}")


def plot_latency_vs_accuracy(df: pd.DataFrame) -> None:
    """Scatter plot: latency vs accuracy -- key plot for advisor value."""
    _ensure_plots_dir()
    agg = aggregate_metrics(df)
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(agg["latency_mean"], agg["accuracy"], s=80, zorder=5, c="#E91E63")

    for _, row in agg.iterrows():
        ax.annotate(row["method"], (row["latency_mean"], row["accuracy"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=8)

    ax.set_xlabel("Mean Latency per Task (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Latency vs. Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_vs_accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'latency_vs_accuracy.png'}")


def plot_cost_vs_latency(df: pd.DataFrame) -> None:
    """Scatter plot: cost vs latency Pareto frontier."""
    _ensure_plots_dir()
    agg = aggregate_metrics(df)
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(agg["cost_mean"] * 1000, agg["latency_mean"], s=80,
                    c=agg["accuracy"], cmap="RdYlGn", zorder=5)

    for _, row in agg.iterrows():
        ax.annotate(row["method"], (row["cost_mean"] * 1000, row["latency_mean"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=8)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Accuracy (%)")
    ax.set_xlabel("Mean Cost per Task (m$)")
    ax.set_ylabel("Mean Latency per Task (s)")
    ax.set_title("Cost vs. Latency (color = accuracy)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cost_vs_latency.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'cost_vs_latency.png'}")


def plot_latency_distribution(df: pd.DataFrame) -> None:
    """Box plots of latency distribution per method."""
    _ensure_plots_dir()
    if df.empty:
        return

    methods = sorted(df["method"].unique())
    data = [df[df["method"] == m]["latency_total_s"].values for m in methods]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=methods, patch_artist=True)

    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Latency per Task (s)")
    ax.set_title("Latency Distribution by Method")
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'latency_distribution.png'}")


def plot_policy_comparison(df: pd.DataFrame) -> None:
    """Grouped bar chart comparing policies on accuracy, cost, and latency."""
    _ensure_plots_dir()
    advisor_df = df[df["method"].str.startswith("advisor_")]
    if advisor_df.empty:
        return

    agg = aggregate_metrics(advisor_df)
    methods = agg["method"].tolist()
    x = np.arange(len(methods))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(14, 6))

    bars1 = ax1.bar(x - width, agg["accuracy"], width, label="Accuracy (%)", color="#2196F3")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Policy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, agg["cost_mean"] * 1000, width, label="Cost (m$)", color="#FF9800", alpha=0.8)
    bars3 = ax2.bar(x + width, agg["latency_mean"], width, label="Latency (s)", color="#4CAF50", alpha=0.8)
    ax2.set_ylabel("Cost (m$) / Latency (s)")

    lines = [bars1, bars2, bars3]
    labels = [b.get_label() for b in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Policy Comparison: Accuracy, Cost, and Latency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "policy_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'policy_comparison.png'}")


def plot_difficulty_split(df: pd.DataFrame) -> None:
    """Grouped bar chart for easy vs hard tasks."""
    _ensure_plots_dir()
    recovery_df = compute_recovery_rate(df)
    if recovery_df is None or recovery_df.empty:
        return

    methods = recovery_df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.3

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars1 = ax1.bar(x - width / 2, recovery_df["recovery_rate_%"], width,
                    label="Recovery Rate (%)", color="#E91E63")
    ax1.set_ylabel("Recovery Rate (%)")
    ax1.set_xlabel("Method")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, recovery_df["hard_latency_mean_s"], width,
                    label="Hard Task Latency (s)", color="#9C27B0", alpha=0.7)
    ax2.set_ylabel("Hard Task Latency (s)")

    lines = [bars1, bars2]
    labels = [b.get_label() for b in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Difficulty Split: Recovery Rate and Latency on Hard Tasks")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "difficulty_split.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'difficulty_split.png'}")


def compute_gaia_agentic_metrics(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute GAIA-specific tool-use and advisor rescue metrics."""
    if df.empty or "dataset" not in df.columns:
        return None
    gaia = df[df["dataset"] == "gaia"]
    if gaia.empty:
        return None

    has_tool_calls = "tool_calls" in gaia.columns
    has_tool_errors = "tool_errors" in gaia.columns
    has_recovery = "recovery_success" in gaia.columns
    has_adv_after_error = "advisor_calls_after_error" in gaia.columns

    rows = []
    for method, group in gaia.groupby("method"):
        total_calls = group["tool_calls"].sum() if has_tool_calls else 0
        total_errors = group["tool_errors"].sum() if has_tool_errors else 0
        tool_error_rate = (total_errors / total_calls) if total_calls > 0 else 0.0
        recovery_tasks = group[group["tool_errors"] > 0] if has_tool_errors else pd.DataFrame()
        recovery_rate = (
            recovery_tasks["recovery_success"].mean() * 100
            if not recovery_tasks.empty and has_recovery
            else 0.0
        )
        advisor_rescue_pool = (
            group[group["advisor_calls_after_error"] > 0]
            if has_adv_after_error
            else pd.DataFrame()
        )
        advisor_rescue_rate = (
            advisor_rescue_pool["correct"].mean() * 100 if not advisor_rescue_pool.empty else 0.0
        )
        advisor_calls_mean = group["advisor_calls"].mean() if "advisor_calls" in group.columns else 0.0
        success = group["correct"].mean() * 100
        advisor_intervention_efficiency = success / advisor_calls_mean if advisor_calls_mean > 0 else 0.0
        first_step_total = group["advisor_first_step_total"].sum() if "advisor_first_step_total" in group.columns else 0
        first_step_followed = (
            group["advisor_followed_first_step_count"].sum()
            if "advisor_followed_first_step_count" in group.columns
            else 0
        )
        first_step_follow_rate = (100.0 * first_step_followed / first_step_total) if first_step_total else 0.0
        repeated_query_rate = (
            group["repeated_query_violations"].sum() / total_calls
            if has_tool_calls and "repeated_query_violations" in group.columns and total_calls > 0
            else 0.0
        )
        blocked_host_rehit_rate = (
            group["blocked_host_rehits"].sum() / total_calls
            if has_tool_calls and "blocked_host_rehits" in group.columns and total_calls > 0
            else 0.0
        )

        rows.append({
            "method": method,
            "task_success_rate_%": round(success, 2),
            "tool_error_rate": round(tool_error_rate, 4),
            "error_recovery_rate_%": round(recovery_rate, 2),
            "advisor_rescue_rate_%": round(advisor_rescue_rate, 2),
            "advisor_intervention_efficiency": round(advisor_intervention_efficiency, 3),
            "advisor_followed_first_step_rate_%": round(first_step_follow_rate, 2),
            "repeated_query_violation_rate": round(repeated_query_rate, 4),
            "blocked_host_rehit_rate": round(blocked_host_rehit_rate, 4),
        })
    return pd.DataFrame(rows)


def check_gaia_smoke_gates(df: pd.DataFrame) -> dict[str, bool] | None:
    """Check smoke gates: strong > cheap and advisor >= cheap on GAIA."""
    if df.empty or "dataset" not in df.columns:
        return None
    gaia = df[df["dataset"] == "gaia"]
    if gaia.empty:
        return None
    by_method = gaia.groupby("method")["correct"].mean().to_dict()
    cheap = by_method.get("cheap_only")
    strong = by_method.get("strong_only_tool_agent", by_method.get("strong_only"))
    advisor = by_method.get("advisor_model_driven")
    if cheap is None or strong is None or advisor is None:
        return None
    return {
        "strong_beats_cheap": strong > cheap,
        "advisor_at_least_cheap": advisor >= cheap,
    }


def plot_gaia_success_vs_tool_error(df: pd.DataFrame) -> None:
    """Scatter: method success vs tool error rate for GAIA."""
    _ensure_plots_dir()
    gaia_metrics = compute_gaia_agentic_metrics(df)
    if gaia_metrics is None or gaia_metrics.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        gaia_metrics["tool_error_rate"] * 100,
        gaia_metrics["task_success_rate_%"],
        s=90,
        c="#3F51B5",
        zorder=5,
    )
    for _, row in gaia_metrics.iterrows():
        ax.annotate(
            row["method"],
            (row["tool_error_rate"] * 100, row["task_success_rate_%"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )
    ax.set_xlabel("Tool Error Rate (%)")
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("GAIA: Success vs Tool Error Rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gaia_success_vs_tool_error.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'gaia_success_vs_tool_error.png'}")


def plot_gaia_advisor_rescue(df: pd.DataFrame) -> None:
    """Bar chart: advisor rescue metrics by method for GAIA."""
    _ensure_plots_dir()
    gaia_metrics = compute_gaia_agentic_metrics(df)
    if gaia_metrics is None or gaia_metrics.empty:
        return

    methods = gaia_metrics["method"].tolist()
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        gaia_metrics["error_recovery_rate_%"],
        width,
        label="Error Recovery Rate (%)",
        color="#009688",
    )
    ax.bar(
        x + width / 2,
        gaia_metrics["advisor_rescue_rate_%"],
        width,
        label="Advisor Rescue Rate (%)",
        color="#FF5722",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("GAIA: Recovery and Advisor Rescue")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gaia_advisor_rescue.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'gaia_advisor_rescue.png'}")


def compute_hotpot_fullwiki_breakdown(df: pd.DataFrame) -> pd.DataFrame | None:
    """Per-question-type accuracy for HotpotQA fullwiki JSONL results."""
    if df.empty or "dataset" not in df.columns:
        return None
    hp = df[df["dataset"] == "hotpotqa_fullwiki"]
    if hp.empty:
        return None

    def _qtype(row: pd.Series) -> str:
        meta = row.get("metadata")
        if isinstance(meta, dict):
            return str(meta.get("type", "unknown"))
        return "unknown"

    hp = hp.copy()
    hp["_qtype"] = hp.apply(_qtype, axis=1)
    rows: list[dict] = []
    for (method, qtype), g in hp.groupby(["method", "_qtype"]):
        rows.append({
            "method": method,
            "question_type": qtype,
            "n": len(g),
            "accuracy_%": round(g["correct"].mean() * 100, 2),
        })
    if not rows:
        return None
    out = pd.DataFrame(rows)
    return out.sort_values(["question_type", "method"])


# ------------------------------------------------------------------
# Main analysis entry point
# ------------------------------------------------------------------

def analyse_results(results_dir: str | Path) -> None:
    """Run full analysis: print summary table, generate all plots."""
    df = load_results(results_dir)
    if df.empty:
        return

    # Summary table
    agg = aggregate_metrics(df)
    print("\n--- Aggregate Metrics ---")
    cols = ["method", "accuracy", "cost_mean", "latency_mean", "latency_median",
            "latency_p95", "advisor_calls_mean", "steps_mean", "n_tasks"]
    print(agg[cols].to_string(index=False))

    # Recovery rate
    recovery_df = compute_recovery_rate(df)
    if recovery_df is not None and not recovery_df.empty:
        print("\n--- Recovery Rate (Hard Tasks) ---")
        print(recovery_df.to_string(index=False))

    # Per-dataset breakdown
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        ds_agg = aggregate_metrics(ds_df)
        print(f"\n--- {ds.upper()} ---")
        print(ds_agg[cols].to_string(index=False))

    gaia_metrics = compute_gaia_agentic_metrics(df)
    if gaia_metrics is not None and not gaia_metrics.empty:
        print("\n--- GAIA Agentic Metrics ---")
        print(gaia_metrics.to_string(index=False))
    hp_break = compute_hotpot_fullwiki_breakdown(df)
    if hp_break is not None and not hp_break.empty:
        print("\n--- HotpotQA Fullwiki (by type) ---")
        print(hp_break.to_string(index=False))
    gates = check_gaia_smoke_gates(df)
    if gates is not None:
        print("\n--- GAIA Smoke Gates ---")
        print(f"strong > cheap: {gates['strong_beats_cheap']}")
        print(f"advisor >= cheap: {gates['advisor_at_least_cheap']}")

    # Plots
    print("\nGenerating plots...")
    plot_three_axis_summary(df)
    plot_cost_vs_accuracy(df)
    plot_latency_vs_accuracy(df)
    plot_cost_vs_latency(df)
    plot_latency_distribution(df)
    plot_policy_comparison(df)
    plot_difficulty_split(df)
    plot_gaia_success_vs_tool_error(df)
    plot_gaia_advisor_rescue(df)
    print("Done.")
