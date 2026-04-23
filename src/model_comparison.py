"""
src/model_comparison.py — Final Model Comparison & Summary
Generates the README table and the hero comparison plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PLOTS_DIR
from src.metrics import ModelMetrics, comparison_table

plt.style.use("dark_background")
COLORS = {"primary": "#00d4ff", "accent": "#a78bfa", "pos": "#34d399",
          "neg": "#f87171", "neutral": "#94a3b8", "warn": "#fb923c"}

MODEL_COLORS = {
    "SARIMA":              COLORS["neutral"],
    "XGBoost":             COLORS["primary"],
    "Bayesian Regression": COLORS["accent"],
}


def plot_model_comparison(all_metrics: list[ModelMetrics], ticker: str) -> None:
    """Hero comparison chart — goes at the top of your GitHub README."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#080c14")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    metric_defs = [
        ("RMSE",        "rmse",   True,  "Lower is better"),
        ("MAE",         "mae",    True,  "Lower is better"),
        ("MAPE %",      "mape",   True,  "Lower is better"),
        ("Dir.Acc %",   "da",     False, "Higher is better"),
        ("Sharpe",      "sharpe", False, "Higher is better"),
    ]

    names   = [m.model_name for m in all_metrics]
    m_colors = [MODEL_COLORS.get(n, COLORS["pos"]) for n in names]

    for idx, (label, attr, lower_better, subtitle) in enumerate(metric_defs):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#0a0f1a")

        values = [getattr(m, attr) for m in all_metrics]
        best   = min(values) if lower_better else max(values)

        bars = ax.bar(names, values, color=m_colors, alpha=0.75,
                      edgecolor="#1e293b", linewidth=0.8)

        # Highlight best model
        for bar, val in zip(bars, values):
            if val == best:
                bar.set_edgecolor("#ffffff")
                bar.set_linewidth(2.5)
                bar.set_alpha(1.0)
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.01,
                    f"{val:.4f}" if attr not in ("da", "sharpe") else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8,
                    color="#f1f5f9")

        ax.set_title(f"{label}\n{subtitle}", color="#f1f5f9", fontsize=10)
        ax.set_ylabel(label, color=COLORS["neutral"])
        ax.tick_params(colors=COLORS["neutral"], labelsize=8, axis="x", rotation=15)
        ax.spines[:].set_color("#1e293b")

    # 6th panel: Summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor("#0a0f1a")
    ax.axis("off")

    df = comparison_table(all_metrics)
    text = "MODEL COMPARISON\n" + "─"*32 + "\n"
    for model_name, row in df.iterrows():
        text += f"\n{model_name}\n"
        for col, val in row.items():
            text += f"  {col:<12}: {val}\n"
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=7.5, color="#e2e8f0",
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#0f172a",
                      edgecolor=COLORS["accent"], linewidth=1))

    plt.suptitle(f"Model Comparison — {ticker}\n"
                 f"SARIMA (statistical baseline) vs XGBoost (ML) vs Bayesian (uncertainty-aware)",
                 color="#f1f5f9", fontsize=13, y=1.01)

    path = PLOTS_DIR / f"{ticker.lower()}_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#080c14")
    plt.close()
    logger.success(f"Saved → {path}")


def print_final_report(
    all_metrics: list[ModelMetrics],
    ab_result: dict | None,
    ticker: str,
) -> None:
    df = comparison_table(all_metrics)
    print(f"\n{'='*65}")
    print(f"FINAL MODEL COMPARISON — {ticker}")
    print(f"{'='*65}")
    print(df.to_string())

    # Best model per metric
    print(f"\n{'─'*65}")
    print("BEST MODEL PER METRIC:")
    for col in df.columns:
        lower_better = col in ("RMSE", "MAE", "MAPE %")
        best_model = df[col].idxmin() if lower_better else df[col].idxmax()
        arrow = "↓" if lower_better else "↑"
        print(f"  {col:<14}: {best_model} ({arrow}{df.loc[best_model, col]})")

    if ab_result:
        print(f"\n{'─'*65}")
        print("BAYESIAN A/B TEST RESULT:")
        print(f"  P(XGBoost > SARIMA) = {ab_result['prob_b_better']:.1%}")
        verdict = "XGBoost is better" if ab_result["prob_b_better"] > 0.5 else "SARIMA is better"
        confidence = "with high confidence" if max(ab_result["prob_b_better"], ab_result["prob_a_better"]) > 0.90 else "but evidence is moderate"
        print(f"  Verdict: {verdict} {confidence}")

    print(f"\n{'─'*65}")
    print("RESUME BULLET (copy this):")
    best_mape_model = df["MAPE %"].idxmin()
    best_mape_val   = df.loc[best_mape_model, "MAPE %"]
    worst_mape_val  = df["MAPE %"].max()
    print(f'  "Achieved {best_mape_val:.2f}% MAPE ({best_mape_model}) vs {worst_mape_val:.2f}%')
    print(f'   baseline; Bayesian model provides 95% credible intervals')
    print(f'   quantifying forecast uncertainty for risk-aware trading."')
    print(f"{'='*65}\n")