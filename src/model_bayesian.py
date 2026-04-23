"""
src/model_bayesian.py — Bayesian Modeling with PyMC
Week 3: Bayesian Structural Time Series + Bayesian A/B Strategy Test

This is the crown jewel of the project — your M.Sc. Statistics differentiator.

Models:
  1. Bayesian Linear Regression with uncertainty quantification
  2. Bayesian A/B Test: Compare two trading strategies

Key concepts demonstrated:
  - Prior specification (encoding domain knowledge)
  - Posterior sampling via MCMC (NUTS sampler)
  - Credible intervals vs confidence intervals
  - Posterior predictive checks
  - Bayesian hypothesis testing P(A > B)

Interview talking points:
  - "Bayesian models give a *distribution* over predictions, not a point estimate.
     That's critical in finance — we care about downside risk, not just expected return."
  - "I encode prior knowledge that daily returns are small (prior on β ~ N(0,1)).
     This is regularisation with a statistical interpretation."
  - "A 95% credible interval means: given the data, there's 95% probability the
     true value lies here. A confidence interval does NOT mean this."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
import warnings, pickle
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PLOTS_DIR, MODELS_DIR, SEED
from src.metrics import evaluate, ModelMetrics
from src.splitter import DataSplit

warnings.filterwarnings("ignore")
plt.style.use("dark_background")
COLORS = {"primary": "#00d4ff", "accent": "#a78bfa", "pos": "#34d399",
          "neg": "#f87171", "neutral": "#94a3b8", "warn": "#fb923c"}


# ── 1. Bayesian Linear Regression ─────────────────────────────────────────────

def build_bayesian_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    n_draws: int = 1000,
    n_tune: int = 500,
) -> tuple:
    """
    Bayesian Linear Regression with weakly informative priors.

    Model:
        y ~ Normal(μ, σ)
        μ = α + X @ β
        α ~ Normal(0, 0.01)     # small intercept (returns ~0 on average)
        β ~ Normal(0, 0.1)      # small coefficients (weak signal in markets)
        σ ~ HalfNormal(0.05)    # noise level

    The priors encode our statistical belief that:
      - Stock return predictability is LOW (weak signal)
      - Feature effects are SMALL (markets are nearly efficient)
    """
    logger.info("Building Bayesian Linear Regression...")
    logger.info(f"  Features: {X_train.shape[1]}  |  Samples: {X_train.shape[0]}")
    logger.info(f"  MCMC: {n_draws} draws + {n_tune} tuning steps")

    with pm.Model() as bayes_model:
        # Data containers
        X_data = pm.Data("X", X_train)
        y_data = pm.Data("y", y_train)

        # ── Priors (encode domain knowledge) ────────────────────────
        alpha = pm.Normal("alpha", mu=0, sigma=0.01)
        beta  = pm.Normal("beta",  mu=0, sigma=0.1, shape=X_train.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=0.05)

        # ── Likelihood ───────────────────────────────────────────────
        mu  = alpha + pm.math.dot(X_data, beta)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_data)

        # ── MCMC Sampling (NUTS — No U-Turn Sampler) ─────────────────
        logger.info("  Sampling posterior (NUTS)...")
        trace = pm.sample(
            draws=n_draws,
            tune=n_tune,
            target_accept=0.9,    # higher = more careful = slower but better
            random_seed=SEED,
            progressbar=True,
            return_inferencedata=True,
        )

        # ── Posterior Predictive ──────────────────────────────────────
        logger.info("  Sampling posterior predictive...")
        ppc_train = pm.sample_posterior_predictive(trace, progressbar=False)

    logger.success("  MCMC sampling complete!")
    _log_convergence(trace)

    return bayes_model, trace, ppc_train


def _log_convergence(trace) -> None:
    """Check R-hat and ESS for convergence diagnostics."""
    summary = az.summary(trace, var_names=["alpha", "sigma"])
    rhat_max = summary["r_hat"].max()
    ess_min  = summary["ess_bulk"].min()
    logger.info(f"  Convergence: R-hat max={rhat_max:.4f} (want <1.01)  "
                f"ESS min={ess_min:.0f} (want >400)")
    if rhat_max > 1.01:
        logger.warning("  R-hat > 1.01 — chains may not have converged!")
    else:
        logger.success("  Convergence check passed ✅")


def predict_bayesian(
    model,
    trace,
    X_test: np.ndarray,
    credible_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate posterior predictive distribution on test data.
    Returns: (mean_pred, lower_CI, upper_CI)

    Uses posterior β samples directly (fast — no re-sampling needed).
    This is the KEY advantage over classical ML:
    We get a DISTRIBUTION of predictions, not just a point estimate.
    """
    alpha_samples = trace.posterior["alpha"].values.flatten()   # (chains*draws,)
    beta_samples  = trace.posterior["beta"].values.reshape(-1, X_test.shape[1])
    sigma_samples = trace.posterior["sigma"].values.flatten()

    # Vectorised: (n_samples, n_test)
    mu_samples = alpha_samples[:, None] + beta_samples @ X_test.T

    # Add observation noise
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, sigma_samples[:, None], size=mu_samples.shape)
    pred_samples = mu_samples + noise

    a = 1 - credible_level
    mean_pred = pred_samples.mean(axis=0)
    lower     = np.percentile(pred_samples, a / 2 * 100, axis=0)
    upper     = np.percentile(pred_samples, (1 - a / 2) * 100, axis=0)

    return mean_pred, lower, upper


# ── 2. Bayesian A/B Strategy Test ─────────────────────────────────────────────

def bayesian_ab_strategy_test(
    strategy_a_returns: np.ndarray,
    strategy_b_returns: np.ndarray,
    name_a: str = "SARIMA Strategy",
    name_b: str = "XGBoost Strategy",
    n_draws: int = 2000,
) -> dict:
    """
    Bayesian hypothesis test: P(Sharpe_B > Sharpe_A)

    Instead of p-values (frequentist), we compute the posterior probability
    that Strategy B is better than Strategy A.

    This is a powerful interview talking point:
    "I don't ask 'is this significant?' — I ask 'what's the probability
     Strategy B beats Strategy A?' That's directly actionable."
    """
    logger.info(f"\nBayesian A/B Test: {name_a} vs {name_b}")

    with pm.Model() as ab_model:
        # Priors: unknown mean return for each strategy
        mu_a    = pm.Normal("mu_a",    mu=0, sigma=0.02)
        mu_b    = pm.Normal("mu_b",    mu=0, sigma=0.02)
        sigma_a = pm.HalfNormal("sigma_a", sigma=0.02)
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.02)

        # Likelihoods
        pm.Normal("obs_a", mu=mu_a, sigma=sigma_a, observed=strategy_a_returns)
        pm.Normal("obs_b", mu=mu_b, sigma=sigma_b, observed=strategy_b_returns)

        # Derived quantities
        diff          = pm.Deterministic("mu_diff",    mu_b - mu_a)
        sharpe_a      = pm.Deterministic("sharpe_a",   mu_a / (sigma_a + 1e-9) * np.sqrt(252))
        sharpe_b      = pm.Deterministic("sharpe_b",   mu_b / (sigma_b + 1e-9) * np.sqrt(252))
        sharpe_diff   = pm.Deterministic("sharpe_diff", sharpe_b - sharpe_a)
        prob_b_better = pm.Deterministic("prob_b_better", pm.math.gt(mu_b, mu_a).astype(float))

        trace_ab = pm.sample(
            draws=n_draws, tune=500,
            random_seed=SEED, progressbar=False,
            return_inferencedata=True,
        )

    # Extract posterior results
    posterior = trace_ab.posterior
    prob_b_wins = float(posterior["prob_b_better"].values.mean())
    sharpe_a_post = posterior["sharpe_a"].values.flatten()
    sharpe_b_post = posterior["sharpe_b"].values.flatten()
    diff_post     = posterior["mu_diff"].values.flatten()

    result = {
        "prob_b_better":    prob_b_wins,
        "prob_a_better":    1 - prob_b_wins,
        "sharpe_a_mean":    sharpe_a_post.mean(),
        "sharpe_b_mean":    sharpe_b_post.mean(),
        "sharpe_a_ci":      np.percentile(sharpe_a_post, [2.5, 97.5]),
        "sharpe_b_ci":      np.percentile(sharpe_b_post, [2.5, 97.5]),
        "diff_mean":        diff_post.mean(),
        "diff_ci":          np.percentile(diff_post, [2.5, 97.5]),
        "trace":            trace_ab,
        "sharpe_a_samples": sharpe_a_post,
        "sharpe_b_samples": sharpe_b_post,
        "name_a": name_a,
        "name_b": name_b,
    }

    logger.success(f"  P({name_b} > {name_a}) = {prob_b_wins:.1%}")
    logger.info(f"  Sharpe {name_a}: {sharpe_a_post.mean():.3f} [{np.percentile(sharpe_a_post,2.5):.3f}, {np.percentile(sharpe_a_post,97.5):.3f}]")
    logger.info(f"  Sharpe {name_b}: {sharpe_b_post.mean():.3f} [{np.percentile(sharpe_b_post,2.5):.3f}, {np.percentile(sharpe_b_post,97.5):.3f}]")

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_bayesian_uncertainty(
    test_index: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    ticker: str,
    metrics: ModelMetrics,
) -> None:
    """The signature plot of Bayesian forecasting — predictions with credible intervals."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.patch.set_facecolor("#080c14")

    # 1. Main forecast with credible interval
    ax = axes[0]
    ax.set_facecolor("#0a0f1a")
    n = min(len(test_index), len(y_true), len(y_pred_mean))
    idx = test_index[:n]

    ax.fill_between(idx, y_pred_lower[:n]*100, y_pred_upper[:n]*100,
                    alpha=0.25, color=COLORS["accent"], label="95% Credible Interval")
    ax.plot(idx, y_pred_mean[:n]*100, color=COLORS["accent"],
            linewidth=1.5, label="Posterior Mean", alpha=0.9)
    ax.plot(idx, y_true[:n]*100,       color=COLORS["primary"],
            linewidth=1,   label="Actual Return",  alpha=0.8)

    ax.set_title(
        f"{ticker} — Bayesian Regression: Posterior Predictive Forecast\n"
        f"MAPE={metrics.mape:.2f}%  DA={metrics.da:.1f}%  Sharpe={metrics.sharpe:.2f}",
        color="#f1f5f9", fontsize=12
    )
    ax.set_ylabel("5-day Return %", color=COLORS["neutral"])
    ax.legend(fontsize=10); ax.tick_params(colors=COLORS["neutral"])
    ax.spines[:].set_color("#1e293b")

    # 2. Interval width (uncertainty) over time
    ax2 = axes[1]
    ax2.set_facecolor("#0a0f1a")
    width = (y_pred_upper[:n] - y_pred_lower[:n]) * 100
    ax2.fill_between(idx, 0, width, alpha=0.5, color=COLORS["warn"])
    ax2.plot(idx, width, color=COLORS["warn"], linewidth=1)
    ax2.set_title("Forecast Uncertainty Width (95% CI) — Wider = Less Confident",
                  color="#f1f5f9", fontsize=10)
    ax2.set_ylabel("CI Width %", color=COLORS["neutral"])
    ax2.tick_params(colors=COLORS["neutral"]); ax2.spines[:].set_color("#1e293b")

    plt.tight_layout()
    path = PLOTS_DIR / f"{ticker.lower()}_bayesian_forecast.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#080c14")
    plt.close()
    logger.success(f"Saved → {path}")


def plot_ab_test(ab_result: dict, ticker: str) -> None:
    """Visualise Bayesian A/B test posterior distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor("#080c14")

    name_a = ab_result["name_a"]
    name_b = ab_result["name_b"]
    sa = ab_result["sharpe_a_samples"]
    sb = ab_result["sharpe_b_samples"]

    # 1. Sharpe posteriors
    ax = axes[0]
    ax.set_facecolor("#0a0f1a")
    bins = np.linspace(min(sa.min(), sb.min()), max(sa.max(), sb.max()), 80)
    ax.hist(sa, bins=bins, alpha=0.6, color=COLORS["primary"],  label=name_a, density=True)
    ax.hist(sb, bins=bins, alpha=0.6, color=COLORS["accent"], label=name_b, density=True)
    ax.axvline(sa.mean(), color=COLORS["primary"],  linewidth=2, linestyle="--")
    ax.axvline(sb.mean(), color=COLORS["accent"], linewidth=2, linestyle="--")
    ax.set_title("Sharpe Ratio Posteriors", color="#f1f5f9", fontsize=11)
    ax.set_xlabel("Annualised Sharpe", color=COLORS["neutral"])
    ax.legend(fontsize=9); ax.tick_params(colors=COLORS["neutral"])
    ax.spines[:].set_color("#1e293b")

    # 2. Difference posterior
    ax = axes[1]
    ax.set_facecolor("#0a0f1a")
    diff = sb - sa
    ax.hist(diff, bins=60, color=COLORS["pos"], alpha=0.7, density=True, edgecolor="#1e293b")
    ax.axvline(0,          color=COLORS["neg"],     linewidth=2.5, linestyle="--", label="No difference")
    ax.axvline(diff.mean(), color=COLORS["primary"], linewidth=2,   linestyle="-",  label=f"Mean={diff.mean():.3f}")
    ci = np.percentile(diff, [2.5, 97.5])
    ax.axvspan(ci[0], ci[1], alpha=0.12, color=COLORS["pos"], label="95% CI")
    ax.set_title(f"Sharpe Difference ({name_b} − {name_a})\nP(B>A) = {ab_result['prob_b_better']:.1%}",
                 color="#f1f5f9", fontsize=11)
    ax.set_xlabel("Sharpe Difference", color=COLORS["neutral"])
    ax.legend(fontsize=8); ax.tick_params(colors=COLORS["neutral"])
    ax.spines[:].set_color("#1e293b")

    # 3. Decision summary
    ax = axes[2]
    ax.set_facecolor("#0a0f1a")
    ax.axis("off")
    prob_b = ab_result["prob_b_better"]
    winner = name_b if prob_b > 0.5 else name_a
    win_prob = prob_b if prob_b > 0.5 else 1 - prob_b

    verdict = "STRONG ✅" if win_prob > 0.90 else "MODERATE ⚠️" if win_prob > 0.75 else "WEAK ❓"
    text = (
        f"BAYESIAN A/B RESULT\n"
        f"{'─'*30}\n\n"
        f"P({name_b} > {name_a})\n"
        f"= {prob_b:.1%}\n\n"
        f"Winner: {winner}\n"
        f"Evidence: {verdict}\n\n"
        f"Sharpe {name_a}:\n"
        f"  {ab_result['sharpe_a_mean']:.3f} "
        f"[{ab_result['sharpe_a_ci'][0]:.3f}, {ab_result['sharpe_a_ci'][1]:.3f}]\n\n"
        f"Sharpe {name_b}:\n"
        f"  {ab_result['sharpe_b_mean']:.3f} "
        f"[{ab_result['sharpe_b_ci'][0]:.3f}, {ab_result['sharpe_b_ci'][1]:.3f}]"
    )
    ax.text(0.05, 0.95, text,
            transform=ax.transAxes, fontsize=10,
            color="#e2e8f0", verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a",
                      edgecolor=COLORS["accent"], linewidth=1.5))

    plt.suptitle(f"Bayesian A/B Strategy Test — {ticker}", color="#f1f5f9", fontsize=14, y=1.01)
    path = PLOTS_DIR / f"{ticker.lower()}_bayesian_ab_test.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#080c14")
    plt.close()
    logger.success(f"Saved → {path}")


def plot_posterior_coefficients(trace, feature_names: list, ticker: str, top_n: int = 15) -> None:
    """Plot posterior distributions of the top feature coefficients."""
    beta_samples = trace.posterior["beta"].values.reshape(-1, len(feature_names))
    means = beta_samples.mean(axis=0)
    top_n   = min(top_n, len(feature_names))
    top_idx = np.argsort(np.abs(means))[-top_n:]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0a0f1a")

    for j, i in enumerate(top_idx):
        samples = beta_samples[:, i]
        ci = np.percentile(samples, [2.5, 97.5])
        color = COLORS["pos"] if means[i] > 0 else COLORS["neg"]
        ax.barh(j, means[i], color=color, alpha=0.7, height=0.6)
        ax.errorbar(means[i], j, xerr=[[means[i]-ci[0]], [ci[1]-means[i]]],
                    fmt="none", color="#f1f5f9", capsize=4, linewidth=1.5)
        ax.axvline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
    ax.set_xlabel("Posterior β (with 95% Credible Interval)", color=COLORS["neutral"])
    ax.set_title(f"{ticker} — Bayesian Regression: Posterior Feature Coefficients\n"
                 f"(Error bars = 95% credible interval — not confidence interval!)",
                 color="#f1f5f9", fontsize=11)
    ax.tick_params(colors=COLORS["neutral"]); ax.spines[:].set_color("#1e293b")

    plt.tight_layout()
    path = PLOTS_DIR / f"{ticker.lower()}_bayesian_coefficients.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#080c14")
    plt.close()
    logger.success(f"Saved → {path}")


# ── Master Runner ─────────────────────────────────────────────────────────────

def run_bayesian(
    split: DataSplit,
    ticker: str,
    sarima_preds: np.ndarray = None,
    xgb_preds: np.ndarray = None,
) -> tuple[ModelMetrics, dict]:
    logger.info(f"\n{'='*50}\nBayesian Models — {ticker}\n{'='*50}")

    # Scale features (Bayesian models need scaled inputs for sensible priors)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(split.X_train.values)
    X_val   = scaler.transform(split.X_val.values)
    X_test  = scaler.transform(split.X_test.values)
    y_train = split.y_train.values
    y_test  = split.y_test.values

    # Use a subset of features (top 10 by variance) to keep MCMC fast
    var_idx = np.argsort(X_train.var(axis=0))[-10:]
    X_train_sub = X_train[:, var_idx]
    X_test_sub  = X_test[:, var_idx]
    feat_names_sub = [split.feature_cols[i] for i in var_idx]

    # ── 1. Bayesian Regression ───────────────────────────────────
    model, trace, ppc_train = build_bayesian_regression(
        X_train_sub, y_train,
        feature_names=feat_names_sub,
        n_draws=800, n_tune=400,
    )

    y_pred_mean, y_pred_lower, y_pred_upper = predict_bayesian(
        model, trace, X_test_sub
    )

    metrics = evaluate(y_test, y_pred_mean, "Bayesian Regression")

    plot_bayesian_uncertainty(
        split.X_test.index, y_test,
        y_pred_mean, y_pred_lower, y_pred_upper,
        ticker, metrics
    )
    plot_posterior_coefficients(trace, feat_names_sub, ticker)

    # ── 2. Bayesian A/B Test ─────────────────────────────────────
    ab_result = None
    if sarima_preds is not None and xgb_preds is not None:
        n = min(len(sarima_preds), len(xgb_preds), len(y_test))
        sa_ret = np.sign(sarima_preds[:n]) * y_test[:n]
        xb_ret = np.sign(xgb_preds[:n])   * y_test[:n]

        ab_result = bayesian_ab_strategy_test(
            sa_ret, xb_ret,
            name_a="SARIMA", name_b="XGBoost",
        )
        plot_ab_test(ab_result, ticker)

    logger.success(f"Bayesian done: {metrics}")
    return metrics, ab_result