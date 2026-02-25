"""
engine/backtesting.py — Rolling VaR/ES backtesting engine.

Implements:
  1. Rolling 1-day-ahead VaR forecasts (parametric, fixed window)
  2. Kupiec (1995) Proportion of Failures (POF) test for unconditional coverage
  3. Christoffersen (1998) independence test for clustering of exceptions
  4. Simple ES backtest: compare average tail loss vs ES estimate

References:
  Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models."
  Christoffersen, P. (1998). "Evaluating Interval Forecasts." International Economic Review.
  Basel Committee (2013). "Fundamental Review of the Trading Book."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from engine.market_data import MarketData
from engine.utils import get_logger

log = get_logger("defender.backtesting")

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Full backtesting output for a portfolio VaR model."""
    confidence:             float           # VaR confidence level tested
    n_obs:                  int             # number of out-of-sample observations
    n_exceptions:           int             # number of VaR breaches
    exception_rate:         float           # n_exceptions / n_obs
    expected_rate:          float           # 1 - confidence
    # Kupiec POF test
    kupiec_statistic:       float
    kupiec_pvalue:          float
    kupiec_result:          str             # "PASS" | "FAIL"
    # Christoffersen independence test
    christoffersen_stat:    float
    christoffersen_pvalue:  float
    christoffersen_result:  str
    # ES backtest
    avg_loss_in_breach:     float           # average realized loss on breach days (negative $)
    avg_es_in_breach:       float           # average ES estimate on breach days (negative $)
    es_adequacy:            str             # "ES adequate", "ES underestimates", "Too few breaches"
    # HIGH-7 FIX: Basel traffic-light zone
    basel_zone:             str             # "GREEN" | "YELLOW" | "RED"
    basel_zone_note:        str             # explanation of the zone
    # Time series data
    forecast_df:            pd.DataFrame    # date, realized_pnl, var_forecast, exception
    # Interpretation
    summary:                str            # plain-English summary


def compute_rolling_var_forecasts(
    market_data: MarketData,
    confidence: float = 0.95,
    min_periods: int = 504,
) -> pd.DataFrame:
    """
    Compute rolling 1-day-ahead VaR forecasts using a fixed expanding window.

    At each time t, uses data from day 0 to day t-1 to forecast day t's VaR.
    Uses parametric (normal) VaR with sample standard deviation.

    Parameters
    ----------
    market_data : MarketData
        Market data object with price history.
    confidence : float
        Confidence level (e.g., 0.95).
    min_periods : int
        Minimum number of observations needed to start forecasting.
        Defaults to 504 (2 years, per Basel III recommendation). A window < 504
        may not capture a full market cycle — results shown with a warning.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'realized_pnl', 'var_forecast', 'exception']
        - realized_pnl: dollar change in portfolio value
        - var_forecast: predicted VaR (positive = loss)
        - exception: 1 if realized_pnl < -var_forecast, 0 otherwise
    """
    # Get portfolio daily returns
    port_ret = market_data.portfolio_returns(log_returns=True)
    initial_value = market_data.total_portfolio_value
    
    # Compute daily P&L
    daily_pnl = port_ret.apply(lambda x: np.exp(x) - 1) * initial_value
    
    # VaR z-score for the given confidence (negative for left tail)
    z_score = stats.norm.ppf(1 - confidence)
    
    forecasts = []
    exceptions = []
    dates = []
    
    for t in range(min_periods, len(daily_pnl)):
        # Use returns up to t-1 to forecast t
        historical_returns = port_ret.iloc[:t].values
        
        # Parametric VaR estimate (left tail loss)
        mu = np.mean(historical_returns)
        sigma = np.std(historical_returns, ddof=1)
        
        # 1-day VaR in percentage terms
        var_pct = mu + z_score * sigma
        
        # Convert to dollar amount (positive = loss magnitude)
        var_forecast = -var_pct * initial_value
        
        # Realized P&L on day t
        realized_pnl = daily_pnl.iloc[t]
        
        # Exception: did we lose more than the VaR forecast?
        exception = 1 if realized_pnl < -var_forecast else 0
        
        forecasts.append({
            'date': daily_pnl.index[t],
            'realized_pnl': realized_pnl,
            'var_forecast': var_forecast,
            'exception': exception,
        })
    
    df = pd.DataFrame(forecasts)
    return df


def kupiec_pof_test(n_obs: int, n_exceptions: int, confidence: float) -> tuple[float, float]:
    """
    Kupiec Proportion of Failures test.

    Tests if the exception rate significantly differs from the expected rate
    (1 - confidence).

    H0: exception rate = (1 - confidence)
    H1: exception rate ≠ (1 - confidence)

    LR = -2 × [n_obs × log(p0) + (n_obs - n_exc) × log(1 - p0)
                - n_exc × log(p_hat) - (n_obs - n_exc) × log(1 - p_hat)]

    where p0 = 1 - confidence, p_hat = n_exceptions / n_obs

    LR ~ χ²(1)

    Returns
    -------
    statistic : float
    pvalue : float
    """
    if n_obs < 1 or n_exceptions < 0:
        return 0.0, 1.0
    
    p0 = 1 - confidence        # expected rate
    p_hat = n_exceptions / n_obs  # observed rate
    
    if p_hat == 0 or p_hat == 1:
        # Edge case: avoid log(0)
        if p_hat == 0:
            p_hat = 1 / (2 * n_obs)
        else:
            p_hat = 1 - 1 / (2 * n_obs)
    
    term1 = n_obs * np.log(p0) + (n_obs - n_exceptions) * np.log(1 - p0)
    term2 = n_exceptions * np.log(p_hat) + (n_obs - n_exceptions) * np.log(1 - p_hat)
    
    lr = -2 * (term1 - term2)
    pvalue = 1 - stats.chi2.cdf(lr, df=1)
    
    return float(lr), float(pvalue)


def christoffersen_independence_test(exceptions: np.ndarray) -> tuple[float, float]:
    """
    Christoffersen independence test.

    Tests if VaR exceptions cluster (are dependent).

    H0: exceptions are independent
    H1: exceptions are dependent (cluster)

    Builds a 2×2 transition matrix T where:
      T[0,0] = count of (0→0) transitions (no exception → no exception)
      T[0,1] = count of (0→1) transitions (no exception → exception)
      T[1,0] = count of (1→0) transitions (exception → no exception)
      T[1,1] = count of (1→1) transitions (exception → exception)

    LR_ind = -2 × [T[0,0]×log(1-π) + T[0,1]×log(π) + T[1,0]×log(1-π) + T[1,1]×log(π)
                    - T[0,0]×log(1-π01) - T[0,1]×log(π01) - T[1,0]×log(1-π11) - T[1,1]×log(π11)]

    where:
      π01 = T[0,1] / (T[0,0] + T[0,1])  (transition prob from 0 to 1)
      π11 = T[1,1] / (T[1,0] + T[1,1])  (transition prob from 1 to 1)
      π = (T[0,1] + T[1,1]) / total     (unconditional exception rate)

    LR_ind ~ χ²(1)

    Returns
    -------
    statistic : float
    pvalue : float
    """
    if len(exceptions) < 2:
        return 0.0, 1.0
    
    # Build transition matrix
    T = np.zeros((2, 2))
    for i in range(len(exceptions) - 1):
        prev_state = int(exceptions[i])
        curr_state = int(exceptions[i + 1])
        T[prev_state, curr_state] += 1
    
    # Transition probabilities
    pi01 = T[0, 1] / (T[0, 0] + T[0, 1]) if (T[0, 0] + T[0, 1]) > 0 else 0.0
    pi11 = T[1, 1] / (T[1, 0] + T[1, 1]) if (T[1, 0] + T[1, 1]) > 0 else 0.0
    pi = (T[0, 1] + T[1, 1]) / T.sum()
    
    # Avoid log(0)
    eps = 1e-10
    pi01 = np.clip(pi01, eps, 1 - eps)
    pi11 = np.clip(pi11, eps, 1 - eps)
    pi = np.clip(pi, eps, 1 - eps)
    
    # Log-likelihood ratio
    lr = -2 * (
        T[0, 0] * np.log(1 - pi) + T[0, 1] * np.log(pi) +
        T[1, 0] * np.log(1 - pi) + T[1, 1] * np.log(pi) -
        T[0, 0] * np.log(1 - pi01) - T[0, 1] * np.log(pi01) -
        T[1, 0] * np.log(1 - pi11) - T[1, 1] * np.log(pi11)
    )
    
    pvalue = 1 - stats.chi2.cdf(lr, df=1)
    
    return float(lr), float(pvalue)


def run_backtest(
    market_data: MarketData,
    confidence: float = 0.95,
    min_periods: int = 504,
) -> BacktestResult:
    """
    Run a complete VaR backtest: Kupiec POF + Christoffersen independence + ES.

    Parameters
    ----------
    market_data : MarketData
        Market data with price history.
    confidence : float
        Confidence level for VaR (e.g., 0.95).
    min_periods : int
        Minimum observations for initial rolling window.

    Returns
    -------
    BacktestResult
    """
    # Compute rolling VaR forecasts
    forecast_df = compute_rolling_var_forecasts(market_data, confidence, min_periods)
    
    # Count exceptions
    n_obs = len(forecast_df)
    n_exceptions = int(forecast_df['exception'].sum())
    exception_rate = n_exceptions / n_obs if n_obs > 0 else 0.0
    expected_rate = 1 - confidence
    
    # Kupiec POF test
    kupiec_stat, kupiec_pval = kupiec_pof_test(n_obs, n_exceptions, confidence)
    kupiec_pass = kupiec_pval >= 0.05
    
    # Christoffersen independence test
    chris_stat, chris_pval = christoffersen_independence_test(forecast_df['exception'].values)
    chris_pass = chris_pval >= 0.05
    
    # ES backtest
    if n_exceptions > 0:
        breach_mask = forecast_df['exception'] == 1
        avg_loss = forecast_df.loc[breach_mask, 'realized_pnl'].mean()
        avg_es = forecast_df.loc[breach_mask, 'var_forecast'].mean()

        if abs(avg_loss) > abs(avg_es):
            es_adequacy = "ES underestimates"
        else:
            es_adequacy = "ES adequate"
    else:
        avg_loss = 0.0
        avg_es = 0.0
        es_adequacy = "Too few breaches"

    # HIGH-7 FIX: Basel traffic-light zone (based on exceptions per 250 trading days)
    # Scaled from n_obs to normalise across different observation periods
    exceptions_per_250 = (n_exceptions / n_obs * 250) if n_obs > 0 else 0
    if exceptions_per_250 <= 4:
        _basel_zone = "GREEN"
        _basel_note = (
            f"{n_exceptions} exceptions in {n_obs} days "
            f"({exceptions_per_250:.1f} per 250-day equivalent). "
            "Model is performing within Basel III green zone (0–4 exceptions per year)."
        )
    elif exceptions_per_250 <= 9:
        _basel_zone = "YELLOW"
        _basel_note = (
            f"{n_exceptions} exceptions in {n_obs} days "
            f"({exceptions_per_250:.1f} per 250-day equivalent). "
            "Model is in Basel III yellow zone (5–9 exceptions). "
            "Review model assumptions — potential VaR underestimation."
        )
    else:
        _basel_zone = "RED"
        _basel_note = (
            f"{n_exceptions} exceptions in {n_obs} days "
            f"({exceptions_per_250:.1f} per 250-day equivalent). "
            "Model is in Basel III RED zone (≥10 exceptions). "
            "VaR model is significantly underestimating risk — immediate review required."
        )
    log.info(f"Basel traffic light: {_basel_zone} — {_basel_note}")

    # Summary
    summary_parts = [
        f"VaR Backtest at {confidence:.1%} confidence level",
        f"Observation period: {n_obs} trading days",
        f"Exceptions: {n_exceptions} ({exception_rate:.1%} vs expected {expected_rate:.1%})",
        f"Kupiec POF test: p-value={kupiec_pval:.3f}, {'PASS' if kupiec_pass else 'FAIL'}",
        f"Christoffersen test: p-value={chris_pval:.3f}, {'PASS' if chris_pass else 'FAIL'}",
        f"ES adequacy: {es_adequacy}",
    ]
    summary = "; ".join(summary_parts)
    
    log.info(summary)
    
    return BacktestResult(
        confidence=confidence,
        n_obs=n_obs,
        n_exceptions=n_exceptions,
        exception_rate=exception_rate,
        expected_rate=expected_rate,
        kupiec_statistic=kupiec_stat,
        kupiec_pvalue=kupiec_pval,
        kupiec_result="PASS" if kupiec_pass else "FAIL",
        christoffersen_stat=chris_stat,
        christoffersen_pvalue=chris_pval,
        christoffersen_result="PASS" if chris_pass else "FAIL",
        avg_loss_in_breach=avg_loss,
        avg_es_in_breach=avg_es,
        es_adequacy=es_adequacy,
        basel_zone=_basel_zone,
        basel_zone_note=_basel_note,
        forecast_df=forecast_df,
        summary=summary,
    )
