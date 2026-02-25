"""
engine/risk_metrics.py — Comprehensive risk metrics engine.

All formulas are implemented from first principles using numpy/scipy.
Every function includes a docstring explaining the math, assumptions,
and how to interpret the result for a non-technical audience.

Mathematical references:
  - Jorion, P. (2006). Value at Risk, 3rd ed. McGraw-Hill.
  - Bali, T., Engle, R., Murray, S. (2016). Empirical Asset Pricing.
  - CFA Institute Curriculum, Level I–III.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.covariance import LedoitWolf

from engine.market_data import MarketData
from engine.utils import get_logger, safe_divide

log = get_logger("defender.risk_metrics")

TRADING_DAYS = 252   # conventional annualisation constant


# ═══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StockRiskMetrics:
    """Risk metrics for a single stock holding."""
    ticker:              str
    company_name:        str
    sector:              str
    annualized_vol:      float   # annualised std dev of log returns
    beta:                float   # vs benchmark (OLS)
    alpha:               float   # Jensen's alpha (annualised)
    sharpe:              float   # Sharpe ratio (annualised)
    sortino:             float   # Sortino ratio
    max_drawdown:        float   # peak-to-trough (negative)
    max_dd_duration:     int     # trading days from peak to recovery
    calmar:              float   # |annualised return / max drawdown|
    skewness:            float   # 3rd moment of return distribution
    kurtosis:            float   # excess kurtosis (4th moment − 3)
    roll_vol_30d:        float   # 30-day rolling vol (most recent)
    roll_vol_90d:        float   # 90-day rolling vol (most recent)
    annualized_return:   float   # geometric annualised return
    component_var_95:    float   # contribution to portfolio 95% VaR
    marginal_var_95:     float   # marginal VaR at 95%
    risk_contribution_pct: float # % of total portfolio risk
    component_var_note:  str     = ""  # CRIT-4: explanation when component VaR is negative (diversifier)


@dataclass
class VaRResult:
    """Value-at-Risk and CVaR at a specific confidence level."""
    confidence:     float   # e.g. 0.95
    parametric_var: float   # normal-distribution VaR (dollar, positive = loss)
    historical_var: float   # empirical percentile VaR (dollar, negative)
    mc_var:         float   # Monte Carlo VaR (dollar, negative, filled later)
    # Expected Shortfall (ES / CVaR) — all at the specified confidence level
    # FRTB IMA standard: ES at 97.5% confidence (≈ parametric VaR at 99% for normal dist)
    # Note: This class stores ES for any confidence; the FRTB-aligned ES uses
    # confidence=0.975 (configured via settings.es_confidence_level)
    parametric_cvar: float  # Expected Shortfall — parametric normal model (ES = CVaR)
    historical_cvar: float  # Expected Shortfall — historical/empirical
    mc_cvar:         float  = 0.0  # Expected Shortfall — Monte Carlo (filled by monte_carlo.py)
    cornish_fisher_var: float = 0.0  # Cornish-Fisher VaR (skewness/kurtosis-adjusted)
    # FRTB compliance note: all ES figures at 97.5% are FRTB IMA-standard.
    # At 97.5%, ES ≈ 99% VaR for normally-distributed returns.

    @property
    def worst_parametric(self) -> float:
        """Most conservative (most negative) parametric VaR."""
        return min(self.parametric_var, self.parametric_cvar)

    @property
    def parametric_es(self) -> float:
        """Expected Shortfall (ES) — parametric normal estimate. Also called CVaR."""
        return self.parametric_cvar

    @property
    def historical_es(self) -> float:
        """Expected Shortfall (ES) — historical/empirical estimate."""
        return self.historical_cvar

    @property
    def mc_es(self) -> float:
        """Expected Shortfall (ES) from Monte Carlo simulation."""
        return self.mc_cvar


@dataclass
class PortfolioRiskMetrics:
    """All portfolio-level risk metrics."""
    # Valuation
    total_value:             float
    # Return / volatility
    annualized_return:       float
    annualized_vol:          float
    # Ratios
    sharpe:                  float
    sortino:                 float
    calmar:                  float
    # Beta / alpha
    beta:                    float
    alpha:                   float
    # Drawdown
    max_drawdown:            float
    max_dd_duration:         int
    # VaR / CVaR
    var_95:                  VaRResult
    var_99:                  VaRResult
    # Concentration
    hhi:                     float   # Herfindahl-Hirschman Index (0–1)
    eff_num_bets:             float   # 1 / sum(w²)
    diversification_ratio:   float   # wtd avg vol / portfolio vol
    avg_pairwise_corr:       float   # average correlation of holdings
    # Distribution
    skewness:                float
    kurtosis:                float
    # HIGH-6: Normality test result
    normality_pvalue:        float = 1.0   # Jarque-Bera p-value (< 0.05 = non-normal)
    normality_warning:       str   = ""    # non-empty if normality rejected at 5% level
    # Per-stock detail
    stock_metrics:           list[StockRiskMetrics] = field(default_factory=list)
    # Correlation
    correlation_matrix:      Optional[pd.DataFrame] = None
    # ES at configured level
    var_es:                  Optional[VaRResult] = None  # ES at configured es_confidence_level (default 97.5%)
    # Factor decomposition
    n_pca_factors_90pct:     int = 0  # eigenvalue factors explaining 90% variance


# ═══════════════════════════════════════════════════════════════════════════════
# Core computation functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns: r_t = ln(P_t / P_{t-1}).

    Log returns are preferred for their time-additivity property —
    multi-period log returns equal the sum of single-period log returns.
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """Simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}."""
    return prices.pct_change().dropna()


def annualized_volatility(log_ret: pd.Series) -> float:
    """
    Annualise daily volatility using the square-root-of-time rule.

    σ_annual = σ_daily × √252

    Assumption: returns are i.i.d. (independent and identically distributed).
    In practice, equities exhibit volatility clustering (GARCH effects), so
    this is an approximation — the standard in academic and practitioner work.

    Interpretation: A 20% annualised vol means that in a typical year, returns
    will be within ±20% of the mean about two-thirds of the time (one std dev).
    """
    return float(log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))


def annualized_return(log_ret: pd.Series) -> float:
    """
    Geometric annualised return from log returns.

    r_annual = exp(mean(log_ret) × 252) − 1

    Uses the geometric method, which is correct for compounding.
    Arithmetic mean would overstate long-run performance.
    """
    if len(log_ret) == 0:
        return 0.0
    return float(np.exp(log_ret.mean() * TRADING_DAYS) - 1)


def compute_beta(stock_ret: pd.Series, bench_ret: pd.Series) -> tuple[float, float]:
    """
    Compute beta and alpha via OLS regression.

    Model: R_i = α + β × R_m + ε

    β = Cov(R_i, R_m) / Var(R_m)

    Interpretation:
      - β > 1: stock moves more than the market (e.g., tech stocks)
      - β < 1: stock moves less (e.g., utilities)
      - β < 0: stock moves opposite to market (rare for equities)

    Returns
    -------
    beta : float
    alpha : float  (annualised Jensen's alpha)
    """
    aligned = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 20:
        return 1.0, 0.0

    x = aligned.iloc[:, 1].values
    y = aligned.iloc[:, 0].values
    cov_matrix = np.cov(y, x, ddof=1)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # Annualised alpha from daily returns: α_annual = exp(α_daily × 252) − 1
    alpha_daily = float(np.mean(y) - beta * np.mean(x))
    alpha_annual = float(np.exp(alpha_daily * TRADING_DAYS) - 1)

    return float(beta), alpha_annual


def sharpe_ratio(ann_return: float, ann_vol: float, rfr: float) -> float:
    """
    Sharpe Ratio = (Annualised Return − Risk-Free Rate) / Annualised Volatility.

    Interpretation:
      - < 0:    Portfolio return is below the risk-free rate (very bad)
      - 0–0.5:  Mediocre risk-adjusted return
      - 0.5–1:  Acceptable
      - 1–2:    Good
      - > 2:    Excellent (rare in practice)

    Assumption: Returns are normally distributed. The ratio penalises both
    upside and downside volatility equally. The Sortino ratio addresses this.
    """
    return safe_divide(ann_return - rfr, ann_vol)


def sortino_ratio(log_ret: pd.Series, ann_return: float, rfr: float) -> float:
    """
    Sortino Ratio = (Annualised Return − Risk-Free Rate) / Downside Deviation.

    Unlike Sharpe, only penalises *negative* returns, making it more suitable
    for asymmetric return distributions. Downside deviation uses a target
    return of the risk-free rate (daily), converted to continuous (log) space
    for consistency with the log return input.

    Downside deviation = √(mean of squared negative excess returns × 252)

    Reference: Sortino & van der Meer (1991); CFA Level II.
    """
    # Convert annual RFR to continuous daily target: ln(1 + r_annual) / 252
    target_daily = np.log(1 + rfr) / TRADING_DAYS
    excess = log_ret - target_daily
    negative_excess = excess[excess < 0]
    if len(negative_excess) == 0:
        return np.inf  # No down days — avoid division by zero
    downside_variance = (negative_excess ** 2).mean() * TRADING_DAYS
    downside_dev = np.sqrt(downside_variance)
    return safe_divide(ann_return - rfr, float(downside_dev))


def max_drawdown_and_duration(prices_or_cumret: pd.Series) -> tuple[float, int]:
    """
    Compute the maximum drawdown and its duration in trading days.

    Maximum drawdown = (Trough Value − Peak Value) / Peak Value

    Duration = number of trading days from the peak to either:
      (a) full recovery to the prior peak, or
      (b) the end of the data if no recovery occurred.

    Returns
    -------
    max_dd : float
        Negative decimal, e.g. -0.35 means the portfolio fell 35% peak-to-trough.
    duration : int
        Trading days from peak to recovery (or end of sample).
    """
    cum = prices_or_cumret
    rolling_max = cum.cummax()
    drawdown    = (cum - rolling_max) / rolling_max

    max_dd = float(drawdown.min())
    if max_dd == 0:
        return 0.0, 0

    trough_idx = drawdown.idxmin()
    peak_idx   = rolling_max[:trough_idx].idxmax()

    # Find recovery (first time drawdown returns to 0 after trough)
    after_trough = drawdown[trough_idx:]
    recovery_dates = after_trough[after_trough >= -1e-6]
    if len(recovery_dates) > 0:
        recovery_idx = recovery_dates.index[0]
    else:
        recovery_idx = drawdown.index[-1]

    duration = len(drawdown[peak_idx:recovery_idx])
    return max_dd, int(duration)


def calmar_ratio(ann_return: float, max_dd: float) -> float:
    """
    Calmar Ratio = Annualised Return / |Maximum Drawdown|.

    Measures return per unit of drawdown risk. Useful for strategies where
    drawdown is the primary risk concern (e.g., endowments, retirees).

    A Calmar ratio > 0.5 is generally considered acceptable.
    """
    return safe_divide(ann_return, abs(max_dd)) if max_dd < 0 else 0.0


def rolling_volatility(log_ret: pd.Series, window: int) -> pd.Series:
    """
    Rolling annualised volatility over *window* trading days.

    Returns a Series of the same length, with NaNs for the first window-1 days.
    """
    return log_ret.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


# ─── VaR / CVaR ───────────────────────────────────────────────────────────────

def parametric_var(
    portfolio_value: float,
    mean_daily_ret:  float,
    daily_vol:       float,
    confidence:      float,
) -> tuple[float, float]:
    """
    Parametric (Normal) VaR and CVaR.

    VaR = −(μ − z × σ) × Portfolio Value

    where:
      μ = mean daily return
      σ = daily return std dev
      z = one-tailed normal quantile at (1 − confidence)

    Assumption: Returns are normally distributed. This understates tail risk
    for fat-tailed distributions (most equity returns have excess kurtosis > 0).

    CVaR (Expected Shortfall) = −(μ − σ × φ(z)/(1−confidence)) × Portfolio Value
    where φ(z) is the standard normal PDF at z.

    Returns
    -------
    var : float   (negative dollar amount, e.g. -2500)
    cvar : float  (negative dollar amount, always worse than VaR)
    """
    alpha = 1 - confidence
    z     = stats.norm.ppf(alpha)            # e.g. -1.645 for 95% confidence
    phi_z = stats.norm.pdf(z)                # PDF at z

    var  = -(mean_daily_ret + z * daily_vol) * portfolio_value
    cvar = -(mean_daily_ret - daily_vol * phi_z / alpha) * portfolio_value

    return float(var), float(cvar)


def cornish_fisher_var(
    portfolio_value: float,
    mean_daily_ret:  float,
    daily_vol:       float,
    confidence:      float,
    skewness:        float,
    excess_kurtosis: float,
) -> float:
    """
    Cornish-Fisher VaR — adjusts the normal quantile for skewness and kurtosis.

    The Cornish-Fisher expansion modifies the standard normal z-score to
    account for the actual shape of the return distribution:

      z_CF = z + (z² - 1) × S/6 + (z³ - 3z) × K/24 - (2z³ - 5z) × S²/36

    where:
      z = normal quantile (e.g., -1.645 for 95%)
      S = skewness of the return distribution
      K = excess kurtosis of the return distribution

    This is a standard adjustment taught in CFA Level II and widely used
    by risk managers to correct for fat-tailed distributions without
    abandoning the parametric framework entirely.

    For equity portfolios (typical S ≈ -0.3, K ≈ 1.5), the Cornish-Fisher
    VaR is typically 10-30% larger than the normal VaR.

    References
    ----------
    Cornish, E.A. & Fisher, R.A. (1938). "Moments and Cumulants in the
    Specification of Distributions."
    Maillard, D. (2012). "A User's Guide to the Cornish Fisher Expansion."
    CFA Institute Curriculum, Level II — Quantitative Methods.

    Returns
    -------
    var : float  (positive dollar loss amount)
    """
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)  # negative, e.g. -1.645
    S = skewness
    K = excess_kurtosis

    # MED-9 FIX: Cornish-Fisher expansion is unreliable for extreme skewness/kurtosis
    # The expansion is a 3rd-order approximation that breaks down at extreme moments
    _cf_valid = True
    if abs(S) > 2.0 or abs(K) > 6.0:
        _cf_valid = False
        log.warning(
            f"Cornish-Fisher expansion may be unreliable: "
            f"skewness={S:.2f} (|S|>2 threshold), excess_kurtosis={K:.2f} (|K|>6 threshold). "
            f"Result should be treated as an approximation only."
        )

    # Cornish-Fisher adjusted quantile
    z_cf = (z
            + (z**2 - 1) * S / 6
            + (z**3 - 3*z) * K / 24
            - (2*z**3 - 5*z) * S**2 / 36)

    var_cf = -(mean_daily_ret + z_cf * daily_vol) * portfolio_value
    return float(var_cf)


def historical_var(
    portfolio_returns: pd.Series,
    portfolio_value:   float,
    confidence:        float,
) -> tuple[float, float]:
    """
    Historical (Empirical) VaR and CVaR.

    VaR = portfolio_value × percentile(returns, 1 − confidence)

    No distributional assumption — uses actual return history.
    Limitation: past tail events may not represent future risk; sensitive to
    the look-back period chosen.

    CVaR = portfolio_value × mean of returns below VaR threshold.

    Returns
    -------
    var : float   (negative dollar amount)
    cvar : float  (negative dollar amount, worse than var)
    """
    alpha    = 1 - confidence
    var_pct  = float(np.percentile(portfolio_returns.dropna(), alpha * 100))
    tail     = portfolio_returns[portfolio_returns <= var_pct]
    cvar_pct = float(tail.mean()) if len(tail) > 0 else var_pct

    return var_pct * portfolio_value, cvar_pct * portfolio_value


# ─── Component / Marginal VaR ─────────────────────────────────────────────────

def component_var(
    weights:      np.ndarray,
    cov_matrix:   np.ndarray,
    portfolio_value: float,
    confidence:   float,
    mean_returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Component VaR for each holding using the parametric normal model.

    Full decomposition matching parametric_var (which includes the mean):

      VaR_i = w_i × [-μ_i + z × (Σw)_i / σ_p] × Portfolio Value

    The key property: Σ VaR_i = Total Portfolio VaR  (Euler decomposition).

    Where:
      μ_i        = daily mean return of stock i
      (Σw)_i/σ_p = marginal contribution of stock i to portfolio volatility
      z          = normal quantile at (1 − confidence)

    Parameters
    ----------
    mean_returns : np.ndarray, optional
        Daily mean returns per stock. If None, mean component is omitted.

    This tells us: "how much does each holding contribute to total portfolio VaR?"
    """
    alpha    = 1 - confidence
    z        = abs(stats.norm.ppf(alpha))

    port_var = float(weights @ cov_matrix @ weights)
    port_std = np.sqrt(port_var)
    if port_std < 1e-10:
        return np.zeros(len(weights))

    # Marginal contribution of each asset to portfolio std
    marginal_sigma = (cov_matrix @ weights) / port_std  # (Σw) / σ_p

    # Volatility component of Component VaR
    vol_component = z * marginal_sigma

    # Mean component (negative mean reduces VaR; positive mean increases it)
    if mean_returns is not None:
        mean_component = -mean_returns  # VaR = -(μ - z×σ) so mean term is -μ
    else:
        mean_component = np.zeros(len(weights))

    # Component VaR: positive = dollar loss contribution
    # Euler identity: Σ C_i = VaR_total = (z×σ_p - μ_p) × PV
    comp_var = weights * (vol_component + mean_component) * portfolio_value

    return comp_var


def marginal_var(
    weights:      np.ndarray,
    cov_matrix:   np.ndarray,
    portfolio_value: float,
    confidence:   float,
) -> np.ndarray:
    """
    Marginal VaR: change in portfolio VaR per 1% increase in each position.

    MVaR_i = z × (Σw)_i / σ_p × Portfolio Value × 0.01

    Use to identify positions that add the most risk per dollar.
    A negative marginal VaR means adding that position *reduces* risk
    (diversification effect).
    """
    alpha    = 1 - confidence
    z        = abs(stats.norm.ppf(alpha))
    port_var = float(weights @ cov_matrix @ weights)
    port_std = np.sqrt(port_var)
    if port_std < 1e-10:
        return np.zeros(len(weights))
    marginal_sigma = (cov_matrix @ weights) / port_std
    return z * marginal_sigma * portfolio_value * 0.01


# ─── Concentration ────────────────────────────────────────────────────────────

def herfindahl_hirschman_index(weights: np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index (HHI) — portfolio concentration measure.

    HHI = Σ w_i²

    Range: 1/N (perfectly diversified) → 1.0 (single holding).

    For regulatory purposes, HHI > 0.25 (25%) is considered concentrated.
    A well-diversified 35-stock portfolio should have HHI ≈ 0.03–0.07.
    """
    return float(np.sum(weights ** 2))


def effective_number_of_bets(weights: np.ndarray) -> float:
    """
    Effective Number of Bets = 1 / HHI.

    Interpretation: the number of equal-weight positions that would produce
    the same concentration as the actual portfolio. For 35 equal-weight
    positions, ENB = 35. Conviction-weighted portfolios will have lower ENB.
    """
    hhi = herfindahl_hirschman_index(weights)
    return safe_divide(1.0, hhi)


def diversification_ratio(
    weights:         np.ndarray,
    individual_vols: np.ndarray,
    portfolio_vol:   float,
) -> float:
    """
    Diversification Ratio = Weighted Average Individual Volatility / Portfolio Volatility.

    DR = (Σ w_i × σ_i) / σ_p

    Range: 1.0 (zero diversification, perfect correlation) → N (theoretical maximum).
    A DR significantly above 1.0 indicates meaningful risk reduction from diversification.
    For a 35-stock equity portfolio, DR ≈ 1.3–2.0 is typical.
    """
    wtd_avg_vol = float(np.dot(weights, individual_vols))
    return safe_divide(wtd_avg_vol, portfolio_vol)


# ─── Correlation / PCA ────────────────────────────────────────────────────────

def ledoit_wolf_covariance(log_rets: pd.DataFrame) -> np.ndarray:
    """
    Estimate the covariance matrix using the Ledoit-Wolf shrinkage estimator.

    The sample covariance matrix is statistically noisy when the number of
    assets (p) is non-trivial relative to the number of observations (T).
    For p=35 and T=504, the ratio p/T ≈ 0.07 — not extreme, but enough to
    benefit from shrinkage.

    Ledoit-Wolf (2004) shrinks the sample covariance towards a structured
    target (scaled identity), with the optimal shrinkage intensity determined
    analytically to minimise the Frobenius norm of the estimation error.

    This is the recommended covariance estimator in CFA Level III curriculum
    and is standard practice at institutional asset managers.

    References
    ----------
    Ledoit, O. & Wolf, M. (2004). "A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices." Journal of Multivariate Analysis.
    """
    lw = LedoitWolf().fit(log_rets.values)
    shrinkage_intensity = lw.shrinkage_
    log.info(f"Ledoit-Wolf shrinkage intensity: {shrinkage_intensity:.4f} "
             f"(0 = pure sample, 1 = pure target)")
    return lw.covariance_


def ewma_covariance(log_rets: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """
    Exponentially Weighted Moving Average (EWMA) covariance matrix.

    The RiskMetrics (1994) EWMA estimator:
      Σ_t = λ × Σ_{t-1} + (1-λ) × r_{t-1} × r_{t-1}ᵀ

    Advantages over static Ledoit-Wolf:
      - Gives more weight to recent observations
      - Adapts to regime changes and volatility clustering
      - Standard in short-horizon risk systems (RiskMetrics JP Morgan 1994)

    Disadvantages:
      - Noisier than Ledoit-Wolf for small portfolios
      - Entire history is implicitly used (just downweighted)
      - Parameter λ must be chosen; 0.94 is the RiskMetrics standard for daily data

    Parameters
    ----------
    log_rets : pd.DataFrame
        Daily log return matrix (rows=dates, cols=tickers).
    lam : float
        Decay factor. 0.94 for daily data (RiskMetrics default).
        Higher λ → more weight to distant past.

    Returns
    -------
    np.ndarray
        Positive-semidefinite EWMA covariance matrix (n_assets × n_assets).

    References
    ----------
    RiskMetrics (1994). "RiskMetrics Technical Document." JP Morgan.
    """
    data   = log_rets.values.astype(float)   # (T, N)
    T, N   = data.shape
    # HIGH-2 FIX: Use 252-day warmup (was 20). With λ=0.94, half-life ≈ 11 days
    # but covariance matrix stabilisation requires ~252 days (Basel/RiskMetrics standard).
    WARMUP = min(252, T)
    if T < 252:
        log.warning(
            f"EWMA warmup: only {T} days of data available (recommend ≥252). "
            f"Falling back to sample covariance for initialisation."
        )
    cov    = np.cov(data[:WARMUP].T, ddof=1) if WARMUP >= 2 else np.eye(N) * 0.01
    for t in range(1, T):
        r   = data[t - 1:t].T   # (N, 1)
        cov = lam * cov + (1 - lam) * (r @ r.T)
    # Ensure positive-definite via regularisation
    cov += np.eye(N) * 1e-8
    return cov


def pca_factors(corr_matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Eigenvalue decomposition of the correlation matrix.

    Returns the eigenvalues (sorted descending) and the number of factors
    required to explain 90% of the total variance.

    Interpretation: If 2–3 factors explain > 90%, the portfolio is heavily
    exposed to systematic (market) risk. More independent factors = better
    true diversification.
    """
    eigenvalues = np.linalg.eigvalsh(corr_matrix)[::-1]  # descending
    explained   = np.cumsum(eigenvalues) / eigenvalues.sum()
    n_factors   = int(np.searchsorted(explained, 0.90)) + 1
    return eigenvalues, n_factors


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level calculator
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(market_data: MarketData, settings=None) -> PortfolioRiskMetrics:
    """
    Master function: compute every risk metric for the portfolio.

    Parameters
    ----------
    market_data : MarketData
        Fully populated MarketData object from market_data.fetch_market_data().

    Returns
    -------
    PortfolioRiskMetrics
        Complete risk analytics, including per-stock and portfolio-level metrics.
    """
    from engine.data_loader import PortfolioSettings
    s = settings or PortfolioSettings()
    es_conf = s.es_confidence_level  # default 0.975
    
    md      = market_data
    rfr     = md.risk_free_rate
    p_value = md.total_portfolio_value
    tickers = md.portfolio_tickers

    # HIGH-3 FIX: Single-stock portfolio guard
    # Covariance, correlation, and diversification metrics require ≥2 assets
    _single_stock_mode = len(tickers) < 2
    if _single_stock_mode:
        log.warning(
            "Single-stock portfolio detected. Covariance/correlation/diversification "
            "metrics are not meaningful and will be set to neutral placeholder values."
        )

    log.info(f"Computing risk metrics for {len(tickers)} tickers, portfolio value ${p_value:,.0f}")

    # ── Return series ──────────────────────────────────────────────────────────
    prices = md.prices[tickers].dropna(how="all")
    log_rets = np.log(prices / prices.shift(1)).dropna()
    bench_log_rets = compute_log_returns(md.prices[md.benchmark_ticker])

    weights = md.weights.reindex(tickers).fillna(0)
    weights = weights / weights.sum()
    w_arr = weights.values.astype(float)

    # Align all series to common dates
    common_idx  = log_rets.index.intersection(bench_log_rets.index)
    log_rets    = log_rets.loc[common_idx]
    bench_ret   = bench_log_rets.loc[common_idx]
    port_ret    = (log_rets * weights).sum(axis=1)

    # CRIT-3 FIX: Compute simple returns for historical VaR
    # For historical VaR, we need portfolio simple return (not weighted sum of simple returns).
    # Correct formula: simple_port_return = exp(log_port_return) - 1
    # This matches the empirical return and avoids mixing log and simple returns.
    # Log returns are still correct for covariance, volatility, beta, parametric VaR.
    port_ret_simple = port_ret.apply(lambda x: np.exp(x) - 1)

    # ── Covariance matrix ─────────────────────────────────────────────────────
    # HIGH-3 FIX: Guard for single-stock portfolios (covariance needs ≥2 assets)
    if _single_stock_mode:
        # Single stock: variance only
        cov_daily = np.array([[float(log_rets.iloc[:, 0].var(ddof=1))]])
        daily_stds = np.sqrt(np.diag(cov_daily))
        corr_mat = pd.DataFrame([[1.0]], index=log_rets.columns, columns=log_rets.columns)
        corr_values = corr_mat.values
    else:
        covariance_mode = s.covariance_mode.lower().strip()
        if covariance_mode == "ewma":
            cov_daily = ewma_covariance(log_rets, lam=s.ewma_lambda).astype(float)
            log.info(f"Covariance mode: EWMA (λ={s.ewma_lambda:.2f})")
        else:
            cov_daily = ledoit_wolf_covariance(log_rets).astype(float)
        # Derive the correlation matrix from the shrunk covariance
        daily_stds = np.sqrt(np.diag(cov_daily))
        corr_values = cov_daily / np.outer(daily_stds, daily_stds)
        np.fill_diagonal(corr_values, 1.0)  # ensure exact 1.0 on diagonal
        corr_mat = pd.DataFrame(corr_values, index=log_rets.columns, columns=log_rets.columns)

    port_daily_var = float(w_arr @ cov_daily @ w_arr)
    port_daily_std = np.sqrt(port_daily_var)
    port_ann_vol   = float(port_daily_std * np.sqrt(TRADING_DAYS))
    port_ann_ret   = annualized_return(port_ret)

    # ── Portfolio drawdown on cumulative returns ───────────────────────────────
    cum_port = (1 + port_ret.apply(lambda x: np.exp(x) - 1)).cumprod()
    port_max_dd, port_dd_duration = max_drawdown_and_duration(cum_port)

    # ── Distribution stats (needed for Cornish-Fisher VaR) ──────────────────
    port_skew = float(stats.skew(port_ret.dropna()))
    port_kurt = float(stats.kurtosis(port_ret.dropna()))  # excess kurtosis

    # ── VaR calculations ───────────────────────────────────────────────────────
    var_results: dict[float, VaRResult] = {}
    for conf in sorted(set([0.95, 0.99, es_conf])):
        p_var, p_cvar = parametric_var(p_value, float(port_ret.mean()), port_daily_std, conf)
        # CRIT-3 FIX: Use simple portfolio returns for historical VaR (no log/simple mixing)
        h_var, h_cvar = historical_var(port_ret_simple, p_value, conf)
        cf_var = cornish_fisher_var(
            p_value, float(port_ret.mean()), port_daily_std,
            conf, port_skew, port_kurt,
        )
        var_results[conf] = VaRResult(
            confidence=conf,
            parametric_var=p_var,
            historical_var=h_var,
            mc_var=0.0,
            parametric_cvar=p_cvar,
            historical_cvar=h_cvar,
            cornish_fisher_var=cf_var,
        )

    # Verify: CVaR must be >= VaR in absolute terms
    for v in var_results.values():
        assert abs(v.parametric_cvar) >= abs(v.parametric_var) - 1e-6, \
            f"CVaR sanity check failed at {v.confidence}"

    # ── Portfolio beta / alpha ─────────────────────────────────────────────────
    p_beta, p_alpha = compute_beta(port_ret, bench_ret)

    # ── Concentration metrics ──────────────────────────────────────────────────
    hhi_val = herfindahl_hirschman_index(w_arr)
    enb_val = effective_number_of_bets(w_arr)
    indiv_vols = np.array([
        float(log_rets[t].std(ddof=1) * np.sqrt(TRADING_DAYS)) for t in tickers
    ])
    # HIGH-3 FIX: Diversification ratio and avg correlation are undefined for single stock
    if _single_stock_mode:
        div_ratio = 1.0  # no diversification by definition
        avg_corr  = 1.0  # single stock: correlation with itself is 1
    else:
        div_ratio = diversification_ratio(w_arr, indiv_vols, port_ann_vol)
        avg_corr  = float(corr_mat.values[np.triu_indices_from(corr_mat.values, k=1)].mean())

    # ── Component VaR ─────────────────────────────────────────────────────────
    mean_daily_rets = log_rets.mean().reindex(tickers).values.astype(float)
    comp_var_95 = component_var(w_arr, cov_daily, p_value, 0.95, mean_daily_rets)
    marg_var_95 = marginal_var(w_arr, cov_daily, p_value, 0.95)

    # Verify component VaR sum = total parametric VaR (Euler decomposition)
    total_comp_var = comp_var_95.sum()
    expected_var   = var_results[0.95].parametric_var
    diff           = abs(total_comp_var - expected_var)
    if diff > abs(expected_var) * 0.01:  # allow 1% tolerance for rounding
        log.warning(
            f"Component VaR sum ({total_comp_var:.2f}) differs from "
            f"total VaR ({expected_var:.2f}) by ${diff:.2f}"
        )
    else:
        log.info(f"Component VaR sum check: PASS (diff=${diff:.4f})")

    # ── PCA ───────────────────────────────────────────────────────────────────
    eigenvalues, n_pca = pca_factors(corr_mat.values)

    # ── Per-stock metrics ──────────────────────────────────────────────────────
    stock_metrics_list: list[StockRiskMetrics] = []
    total_abs_comp_var = comp_var_95.sum()  # should equal total VaR (Euler)

    for i, ticker in enumerate(tickers):
        s_log_ret = log_rets[ticker]
        s_ann_vol = float(s_log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
        s_ann_ret = annualized_return(s_log_ret)
        s_beta, s_alpha = compute_beta(s_log_ret, bench_ret)
        s_sharpe  = sharpe_ratio(s_ann_ret, s_ann_vol, rfr)
        s_sortino = sortino_ratio(s_log_ret, s_ann_ret, rfr)

        s_prices = prices[ticker].dropna()
        cum_s    = (1 + compute_simple_returns(s_prices)).cumprod()
        s_max_dd, s_dd_dur = max_drawdown_and_duration(cum_s)
        s_calmar  = calmar_ratio(s_ann_ret, s_max_dd)
        s_skew    = float(stats.skew(s_log_ret.dropna()))
        s_kurt    = float(stats.kurtosis(s_log_ret.dropna()))

        # Rolling vols
        rv_30 = float(rolling_volatility(s_log_ret, 30).iloc[-1]) if len(s_log_ret) >= 30 else s_ann_vol
        rv_90 = float(rolling_volatility(s_log_ret, 90).iloc[-1]) if len(s_log_ret) >= 90 else s_ann_vol

        # Risk contribution pct (component VaR / total portfolio VaR)
        risk_contrib_pct = safe_divide(comp_var_95[i], total_abs_comp_var) * 100

        # Find holding info
        holding = next((h for h in md.holdings if h.ticker == ticker), None)
        name    = holding.company_name if holding else ticker
        sector  = holding.sector       if holding else "Unknown"

        # CRIT-4 FIX: Generate sign convention note for negative Component VaR
        _cvvar = float(comp_var_95[i])
        _cvnote = ""
        if _cvvar < 0:
            _cvnote = (
                "Negative Component VaR: this position actively REDUCES portfolio risk. "
                "Its low or negative correlation with other holdings offsets portfolio "
                "volatility (Euler decomposition). This is a diversifier, not an error."
            )

        stock_metrics_list.append(StockRiskMetrics(
            ticker=ticker,
            company_name=name,
            sector=sector,
            annualized_vol=s_ann_vol,
            beta=s_beta,
            alpha=s_alpha,
            sharpe=s_sharpe,
            sortino=s_sortino,
            max_drawdown=s_max_dd,
            max_dd_duration=s_dd_dur,
            calmar=s_calmar,
            skewness=s_skew,
            kurtosis=s_kurt,
            roll_vol_30d=rv_30,
            roll_vol_90d=rv_90,
            annualized_return=s_ann_ret,
            component_var_95=_cvvar,
            marginal_var_95=float(marg_var_95[i]),
            risk_contribution_pct=risk_contrib_pct,
            component_var_note=_cvnote,
        ))

    log.info(
        f"Risk computation complete. Portfolio vol={port_ann_vol:.2%}, beta={p_beta:.2f}, "
        f"VaR(95%)=${var_results[0.95].parametric_var:,.0f}, "
        f"ES({es_conf:.1%})=${var_results[es_conf].parametric_cvar:,.0f}"
    )

    # ── HIGH-6 FIX: Jarque-Bera normality test ────────────────────────────────
    from scipy.stats import jarque_bera
    _jb_stat, _jb_pval = jarque_bera(port_ret.dropna().values)
    _jb_pval = float(_jb_pval)
    _normality_warning = ""
    if _jb_pval < 0.05:
        _normality_warning = (
            f"Return distribution is NON-NORMAL (Jarque-Bera p={_jb_pval:.4f} < 0.05). "
            f"Skewness={port_skew:.2f}, Excess Kurtosis={port_kurt:.2f}. "
            f"Parametric VaR (which assumes normality) may UNDERSTATE tail risk. "
            f"Prefer Historical or Cornish-Fisher VaR for this portfolio."
        )
        log.warning(_normality_warning)
    else:
        log.info(f"Normality test: PASS (Jarque-Bera p={_jb_pval:.4f})")

    return PortfolioRiskMetrics(
        total_value=p_value,
        annualized_return=port_ann_ret,
        annualized_vol=port_ann_vol,
        sharpe=sharpe_ratio(port_ann_ret, port_ann_vol, rfr),
        sortino=sortino_ratio(port_ret, port_ann_ret, rfr),
        calmar=calmar_ratio(port_ann_ret, port_max_dd),
        beta=p_beta,
        alpha=p_alpha,
        max_drawdown=port_max_dd,
        max_dd_duration=port_dd_duration,
        var_95=var_results[0.95],
        var_99=var_results[0.99],
        var_es=var_results[es_conf],
        hhi=hhi_val,
        eff_num_bets=enb_val,
        diversification_ratio=div_ratio,
        avg_pairwise_corr=avg_corr,
        skewness=port_skew,
        kurtosis=port_kurt,
        stock_metrics=stock_metrics_list,
        correlation_matrix=corr_mat,
        n_pca_factors_90pct=n_pca,
        normality_pvalue=_jb_pval,
        normality_warning=_normality_warning,
    )


def build_rolling_vol_df(market_data: MarketData, window: int = 30) -> pd.DataFrame:
    """
    Build a DataFrame of rolling annualised volatility for the portfolio
    and benchmark over the specified window.

    Returns a two-column DataFrame: ['Portfolio', benchmark_ticker].
    """
    port_ret  = market_data.portfolio_returns(log_returns=True)
    bench_ret = market_data.benchmark_returns(log_returns=True)
    combined  = pd.concat([port_ret, bench_ret], axis=1).dropna()
    return combined.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def build_drawdown_series(market_data: MarketData) -> pd.DataFrame:
    """
    Build daily drawdown series for the portfolio and benchmark.

    Returns a DataFrame with ['Portfolio', benchmark_ticker] columns,
    values are the percentage decline from the rolling peak (negative).
    """
    port_prices  = (1 + market_data.portfolio_returns()).cumprod()
    bench_prices = (1 + market_data.benchmark_returns()).cumprod()

    def _dd(series: pd.Series) -> pd.Series:
        peak = series.cummax()
        return (series - peak) / peak

    return pd.DataFrame({
        "Portfolio":                _dd(port_prices),
        market_data.benchmark_ticker: _dd(bench_prices),
    }).dropna()
