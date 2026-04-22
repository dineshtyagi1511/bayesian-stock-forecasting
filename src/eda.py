"""
src/eda.py — Exploratory Data Analysis & Visualization
Week 1, Step 4: Statistical EDA with plots for GitHub README

Generates:
  1. Price & volume chart
  2. Return distribution (normality test)
  3. Rolling volatility
  4. Correlation heatmap of features
  5. Stationarity test (ADF)
  6. ACF / PACF plots (needed for SARIMA order selection)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PLOTS_DIR

plt.style.use("dark_background")
COLORS = {
    "primary":  "#00d4ff",
    "accent":   "#a78bfa",
    "positive": "#34d399",
    "negative": "#f87171",
    "neutral":  "#94a3b8",
}


# ── 1. Price & Volume ─────────────────────────────────────────────────────────

def plot_price_volume(df: pd.DataFrame, ticker: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.patch.set_facecolor("#0a0f1a")

    ax1.plot(df.index, df["Close"], color=COLORS["primary"], linewidth=1.2, label="Close")
    ax1.fill_between(df.index, df["Close"], alpha=0.08, color=COLORS["primary"])
    ax1.set_ylabel("Price", color=COLORS["neutral"])
    ax1.set_title(f"{ticker} — Price History ({df.index[0].year}–{df.index[-1].year})",
                  color="#f1f5f9", fontsize=14, pad=12)
    ax1.set_facecolor("#0a0f1a")
    ax1.tick_params(colors=COLORS["neutral"])
    ax1.spines[:].set_color("#1e293b")

    ax2.bar(df.index, df["Volume"], color=COLORS["accent"], alpha=0.6, width=1)
    ax2.set_ylabel("Volume", color=COLORS["neutral"])
    ax2.set_facecolor("#0a0f1a")
    ax2.tick_params(colors=COLORS["neutral"])
    ax2.spines[:].set_color("#1e293b")

    plt.tight_layout()
    path = PLOTS_DIR / f"{ticker.lower()}_price_volume.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close()
    logger.success(f"Saved → {path}")


# ── 2. Return Distribution & Normality Test ───────────────────────────────────

def plot_return_distribution(df: pd.DataFrame, ticker: str) -> None:
    """
    Plot log-return distribution and test for normality.
    Financial returns are NOT normally distributed (fat tails, negative skew).
    This is a key statistical insight to mention in interviews.
    """
    returns = df["log_return_1d"].dropna()

    # Statistical tests
    _, p_shapiro  = stats.shapiro(returns.sample(min(5000, len(returns)), random_state=42))
    _, p_jb       = stats.jarque_bera(returns)
    skewness      = returns.skew()
    excess_kurt   = returns.kurt()      # excess kurtosis (normal = 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0f1a")

    # Histogram + KDE
    ax = axes[0]
    ax.set_facecolor("#0a0f1a")
    ax.hist(returns, bins=100, density=True, color=COLORS["accent"],
            alpha=0.6, label="Empirical")

    # Overlay normal distribution
    x = np.linspace(returns.min(), returns.max(), 500)
    normal_pdf = stats.norm.pdf(x, returns.mean(), returns.std())
    ax.plot(x, normal_pdf, color=COLORS["negative"], linewidth=2, label="Normal fit")

    ax.set_title(f"{ticker} Log-Return Distribution\n"
                 f"Skew={skewness:.3f}  ExKurt={excess_kurt:.3f}  "
                 f"JB p={p_jb:.4f}",
                 color="#f1f5f9", fontsize=11)
    ax.set_xlabel("Log Return", color=COLORS["neutral"])
    ax.set_ylabel("Density",    color=COLORS["neutral"])
    ax.legend(fontsize=9)
    ax.tick_params(colors=COLORS["neutral"])
    ax.spines[:].set_color("#1e293b")

    # Q-Q Plot (fat tails will deviate from the diagonal)
    ax = axes[1]
    ax.set_facecolor("#0a0f1a")
    (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm")
    ax.scatter(osm, osr, color=COLORS["accent"], alpha=0.3, s=3)
    ax.plot(osm, slope * np.array(osm) + intercept,
            color=COLORS["negative"], linewidth=2, label=f"R²={r**2:.4f}")
    ax.set_title(f"{ticker} Q-Q Plot vs Normal\n(fat tails → S-curve deviation)",
                 color="#f1f5f9", fontsize=11)
    ax.set_xlabel("Theoretical Quantiles", color=COLORS["neutral"])
    ax.set_ylabel("Sample Quantiles",      color=COLORS["neutral"])
    ax.legend(fontsize=9)
    ax.tick_params(colors=COLORS["neutral"])
    ax.spines[:].set_color("#1e293b")

    plt.tight_layout()
    path = PLOTS_DIR / f"{ticker.lower()}_return_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close()
    logger.success(f"Saved → {path}")

    # Print statistical summary
    print(f"\n{ticker} Return Statistics:")
    print(f"  Mean daily return : {returns.mean()*100:.4f}%")
    print(f"  Daily volatility  : {returns.std()*100:.4f}%")
    print(f"  Ann. volatility   : {returns.std()*np.sqrt(252)*100:.2f}%")
    print(f"  Skewness          : {skewness:.4f}  {'← negatively skewed (crash risk)' if skewness < 0 else ''}")
    print(f"  Excess Kurtosis   : {excess_kurt:.4f}  ← fat tails (normal=0)")
    print(f"  Jarque-Bera p-val : {p_jb:.6f}  {'→ NOT normal' if p_jb < 0.05 else '→ normal'}")


# ── 3. Rolling Volatility ─────────────────────────────────────────────────────

def plot_rolling_volatility(df: pd.DataFrame, ticker: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0a0f1a")
    ax.set_facecolor("#0a0f1a")

    windows = {"21d": COLORS["primary"], "63d": COLORS["accent"], "252d": COLORS["positive"]}
    r = df["log_return_1d"]
    for w_label, color in windows.items():
        w = int(w_label.replace("d", ""))
        vol = r.rolling(w).std() * np.sqrt(252) * 100
        ax.plot(df.index, vol, color=color, linewidth=1.2, label=f"{w_label} Ann. Vol %", alpha=0.9)

    ax.set_title(f"{ticker} — Realized Volatility (Annualized %)", color="#f1f5f9", fontsize=13)
    ax.set_ylabel("Volatility %", color=COLORS["neutral"])
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS["neutral"])
    ax.spines[:].set_color("#1e293b")

    plt.tight_layout()
    path = PLOTS_DIR / f"{ticker.lower()}_volatility.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close()
    logger.success(f"Saved → {path}")


# ── 4. Stationarity Tests ─────────────────────────────────────────────────────

def test_stationarity(df: pd.DataFrame, ticker: str) -> None:
    """
    ADF and KPSS tests — critical before fitting SARIMA.

    ADF  H0: series has unit root (non-stationary)
    KPSS H0: series is stationary

    In interviews: "I always test stationarity before ARIMA because the model
    assumes a stationary series. Log-returns are typically stationary; prices are not."
    """
    print(f"\n{'='*55}")
    print(f"Stationarity Tests — {ticker}")
    print(f"{'='*55}")

    for series_name, series in [
        ("Close Price (levels)",  df["Close"]),
        ("Log Returns",           df["log_return_1d"].dropna()),
    ]:
        print(f"\n  Series: {series_name}")

        # ADF Test
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series.dropna(), autolag="AIC")
        print(f"  ADF stat={adf_stat:.4f}  p={adf_p:.6f}  "
              f"{'✅ Stationary' if adf_p < 0.05 else '❌ Non-stationary (unit root)'}")

        # KPSS Test
        try:
            kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression="c", nlags="auto")
            print(f"  KPSS stat={kpss_stat:.4f}  p={kpss_p:.6f}  "
                  f"{'✅ Stationary' if kpss_p > 0.05 else '❌ Non-stationary'}")
        except Exception:
            pass


# ── 5. ACF / PACF (for SARIMA order selection) ───────────────────────────────

def plot_acf_pacf(df: pd.DataFrame, ticker: str) -> None:
    """
    ACF and PACF plots on log returns.
    Used to visually select p, q orders for ARIMA before auto_arima.
    """
    returns = df["log_return_1d"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0f1a")

    plot_acf(returns,  lags=40, ax=axes[0], color=COLORS["primary"],
             title=f"{ticker} ACF — Log Returns")
    plot_pacf(returns, lags=40, ax=axes[1], color=COLORS["accent"],
              title=f"{ticker} PACF — Log Returns", method="ywm")

    for ax in axes:
        ax.set_facecolor("#0a0f1a")
        ax.tick_params(colors=COLORS["neutral"])
        ax.spines[:].set_color("#1e293b")

    plt.tight_layout()
    path = PLOTS_DIR / f"{ticker.lower()}_acf_pacf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close()
    logger.success(f"Saved → {path}")


# ── 6. Feature Correlation Heatmap ───────────────────────────────────────────

def plot_feature_correlation(df: pd.DataFrame, ticker: str, top_n: int = 20) -> None:
    """Correlation heatmap of top features vs target."""
    target = "target_5d"
    if target not in df.columns:
        logger.warning("target_5d not found — skipping correlation plot")
        return

    feature_cols = [c for c in df.columns
                    if c not in {"Open","High","Low","Close","Volume"} and c != target]

    corrs = df[feature_cols].corrwith(df[target]).abs().sort_values(ascending=False)
    top_features = corrs.head(top_n).index.tolist() + [target]

    corr_matrix = df[top_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#0a0f1a")
    ax.set_facecolor("#0a0f1a")

    sns.heatmap(
        corr_matrix, ax=ax,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, linecolor="#1e293b",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"{ticker} — Top {top_n} Feature Correlations with Target",
                 color="#f1f5f9", fontsize=12, pad=12)
    ax.tick_params(colors=COLORS["neutral"], labelsize=8)

    plt.tight_layout()
    path = PLOTS_DIR / f"{ticker.lower()}_feature_correlation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close()
    logger.success(f"Saved → {path}")


# ── Master EDA Runner ─────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame, ticker: str) -> None:
    logger.info(f"Running full EDA for {ticker}...")
    plot_price_volume(df, ticker)
    plot_return_distribution(df, ticker)
    plot_rolling_volatility(df, ticker)
    test_stationarity(df, ticker)
    plot_acf_pacf(df, ticker)
    plot_feature_correlation(df, ticker)
    logger.success(f"EDA complete. All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    from src.features import load_features
    logger.info("=== Week 1 · Step 4: EDA ===")
    df = load_features("NIFTY50")
    run_full_eda(df, "NIFTY50")
