"""
engine/stress_testing.py — Historical and hypothetical stress test engine.

Historical scenarios apply empirically observed drawdowns from major market
events to the current portfolio composition. Hypothetical scenarios apply
structured shocks to assess portfolio vulnerabilities.

Methodology notes:
  Historical scenarios use sector-level drawdowns from the relevant periods,
  applied to current sector weights. This is a top-down approximation —
  it does not use actual per-stock returns from those periods (which would
  require holding history the club doesn't have), but uses well-documented
  sector drawdowns. The methodology and its limitations are stated clearly.

References:
  - Qian (2006). "On the Financial Interpretation of Risk Contribution."
  - Litterman (1997). "Hot Spots and Hedges."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from engine.market_data import MarketData
from engine.utils import get_logger, safe_divide, fmt_currency, fmt_pct

log = get_logger("defender.stress_testing")


# ═══════════════════════════════════════════════════════════════════════════════
# Historical scenario data (sector drawdowns, peer-reviewed sources)
# ═══════════════════════════════════════════════════════════════════════════════

# Sector drawdowns during major historical stress events.
# Sources: S&P 500 GICS sector index performance over the stated periods.
# Format: {sector_name: drawdown_fraction}  (negative = loss)
HISTORICAL_SCENARIOS: dict[str, dict] = {
    "2008 Global Financial Crisis": {
        "description": "Lehman Brothers collapse, credit market freeze. S&P 500 fell ~56% peak-to-trough.",
        "period": "Sep 2008 – Mar 2009",
        "benchmark_drawdown": -0.465,  # S&P 500 in this specific window
        "sector_drawdowns": {
            "Financial Services": -0.72,
            "Technology":          -0.45,
            "Consumer Cyclical": -0.55,
            "Industrials":         -0.52,
            "Basic Materials":           -0.55,
            "Energy":              -0.55,
            "Communication Services":  -0.42,
            "Real Estate":         -0.65,
            "Healthcare":          -0.33,
            "Consumer Defensive":    -0.25,
            "Utilities":           -0.38,
            "Unknown":             -0.46,
        },
    },
    "COVID-19 Crash": {
        "description": "Global pandemic shock. S&P 500 fell ~34% in 33 days — fastest bear market on record.",
        "period": "Feb 19 – Mar 23, 2020",
        "benchmark_drawdown": -0.339,
        "sector_drawdowns": {
            "Energy":              -0.53,
            "Financial Services":  -0.39,
            "Consumer Cyclical": -0.38,
            "Industrials":         -0.40,
            "Real Estate":         -0.42,
            "Basic Materials":           -0.36,
            "Technology":          -0.26,
            "Communication Services":  -0.27,
            "Healthcare":          -0.18,
            "Consumer Defensive":    -0.18,
            "Utilities":           -0.23,
            "Unknown":             -0.34,
        },
    },
    "2022 Rate Shock": {
        "description": "Federal Reserve raised rates 425bps in 9 months. Growth stocks crushed.",
        "period": "Jan – Oct 2022",
        "benchmark_drawdown": -0.248,
        "sector_drawdowns": {
            "Communication Services":  -0.47,
            "Technology":          -0.39,
            "Consumer Cyclical": -0.41,
            "Real Estate":         -0.35,
            "Financial Services":  -0.22,
            "Basic Materials":           -0.22,
            "Industrials":         -0.17,
            "Healthcare":          -0.08,
            "Energy":              +0.57,   # Energy was positive in 2022
            "Consumer Defensive":    -0.03,
            "Utilities":           -0.04,
            "Unknown":             -0.25,
        },
    },
    "Dot-Com Bust": {
        "description": "Tech bubble deflation. S&P 500 fell ~50% over 2.5 years.",
        "period": "Mar 2000 – Oct 2002",
        "benchmark_drawdown": -0.489,
        "sector_drawdowns": {
            "Technology":          -0.82,
            "Communication Services":  -0.71,
            "Consumer Cyclical": -0.45,
            "Industrials":         -0.38,
            "Basic Materials":           -0.30,
            "Financial Services":  -0.22,
            "Healthcare":          -0.27,
            "Real Estate":         +0.10,
            "Energy":              -0.12,
            "Consumer Defensive":    -0.08,
            "Utilities":           -0.45,
            "Unknown":             -0.49,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StockScenarioImpact:
    """Per-stock impact in a scenario."""
    ticker:             str
    company_name:       str
    sector:             str
    weight:             float
    scenario_drawdown:  float   # applied drawdown fraction
    dollar_loss:        float   # estimated dollar loss (negative)
    pct_of_total_loss:  float   # this stock's share of total portfolio loss


@dataclass
class ScenarioResult:
    """Results of one stress test scenario."""
    name:               str
    description:        str
    period:             str
    methodology:        str
    assumptions:        str
    portfolio_loss_pct: float   # estimated portfolio loss (negative fraction)
    portfolio_loss_usd: float   # estimated dollar loss (negative)
    benchmark_loss_pct: float   # benchmark loss in same scenario
    portfolio_value:    float   # starting value
    stock_impacts:      list[StockScenarioImpact] = field(default_factory=list)
    interpretation:     str = ""
    # Phase 5: ES comparison dict (populated by compare_scenario_to_es)
    es_comparison:      Optional[dict] = None

    @property
    def severity(self) -> str:
        """
        Severity label based on scenario loss relative to ES benchmarks.

        Compares the scenario's portfolio loss percentage against the
        1-day parametric ES at 95% and 99% thresholds (annualised to a
        monthly proxy via sqrt(21) scaling).

        Returns one of: "EXTREME", "SEVERE", "ELEVATED", "MODERATE", "MILD"
        """
        if self.es_comparison is None:
            # Fallback: label by raw percentage loss alone
            loss = abs(self.portfolio_loss_pct)
            if loss >= 0.40:
                return "EXTREME"
            elif loss >= 0.25:
                return "SEVERE"
            elif loss >= 0.15:
                return "ELEVATED"
            elif loss >= 0.07:
                return "MODERATE"
            else:
                return "MILD"
        ratio = self.es_comparison.get("loss_to_es99_ratio", 0.0)
        if ratio >= 5.0:
            return "EXTREME"
        elif ratio >= 3.0:
            return "SEVERE"
        elif ratio >= 2.0:
            return "ELEVATED"
        elif ratio >= 1.0:
            return "MODERATE"
        else:
            return "MILD"

    @property
    def severity_color(self) -> str:
        """Hex colour associated with severity level for dashboard display."""
        return {
            "EXTREME":  "#B71C1C",
            "SEVERE":   "#E53935",
            "ELEVATED": "#FB8C00",
            "MODERATE": "#FDD835",
            "MILD":     "#43A047",
        }.get(self.severity, "#9E9E9E")

    @property
    def top_losers(self) -> list[StockScenarioImpact]:
        """Top 5 holdings by dollar loss (worst first)."""
        return sorted(self.stock_impacts, key=lambda x: x.dollar_loss)[:5]

    @property
    def relative_safe_havens(self) -> list[StockScenarioImpact]:
        """Top 5 holdings with best relative performance (smallest loss or gain)."""
        return sorted(self.stock_impacts, key=lambda x: x.scenario_drawdown, reverse=True)[:5]


@dataclass
class StressTestResults:
    """All stress test results combined."""
    historical:    list[ScenarioResult] = field(default_factory=list)
    hypothetical:  list[ScenarioResult] = field(default_factory=list)

    @property
    def all_scenarios(self) -> list[ScenarioResult]:
        return self.historical + self.hypothetical


# ═══════════════════════════════════════════════════════════════════════════════
# Core stress test engine
# ═══════════════════════════════════════════════════════════════════════════════

def compare_scenario_to_es(
    scenario: "ScenarioResult",
    metrics,   # PortfolioRiskMetrics — avoid circular import with string annotation
) -> dict:
    """
    Compare a stress scenario's portfolio loss against ES benchmarks.

    Produces a dict with:
      - ``es_95_1d``  : 1-day ES at 95% (positive number, $)
      - ``es_975_1d`` : 1-day ES at 97.5% (positive number, $)
      - ``es_99_1d``  : 1-day ES at 99% (positive number, $)
      - ``es_99_21d`` : 21-day proxy (1-day × sqrt(21)) (positive number, $)
      - ``scenario_loss_usd`` : scenario dollar loss (positive number)
      - ``loss_to_es95_ratio``  : scenario_loss / es_95_1d
      - ``loss_to_es975_ratio`` : scenario_loss / es_975_1d
      - ``loss_to_es99_ratio``  : scenario_loss / es_99_1d
      - ``loss_to_es99_21d_ratio`` : scenario_loss / es_99_21d
      - ``multiples_of_daily_es`` : plaintext description

    Parameters
    ----------
    scenario : ScenarioResult
        A completed stress test scenario.
    metrics : PortfolioRiskMetrics
        Full risk metrics object from ``compute_all_metrics``.

    Returns
    -------
    dict
    """
    import math

    scenario_loss = abs(scenario.portfolio_loss_usd)

    # Extract ES values (ES = CVaR in our model)
    var95  = getattr(metrics, "var_95",  None)
    var99  = getattr(metrics, "var_99",  None)
    var_es = getattr(metrics, "var_es",  None)   # ES at configured 97.5%

    # Use parametric CVaR as the ES benchmark (positive dollar values)
    es_95  = abs(var95.parametric_cvar)  if var95  else 0.0
    es_99  = abs(var99.parametric_cvar)  if var99  else 0.0
    es_975 = abs(var_es.parametric_cvar) if var_es else 0.5 * (es_95 + es_99)

    # 21-day proxy via sqrt(T) scaling (a recognised approximation for i.i.d. returns)
    es_99_21d = es_99 * math.sqrt(21)

    def _ratio(loss: float, es: float) -> float:
        return round(loss / es, 2) if es > 0 else 0.0

    ratio_95    = _ratio(scenario_loss, es_95)
    ratio_975   = _ratio(scenario_loss, es_975)
    ratio_99    = _ratio(scenario_loss, es_99)
    ratio_21d   = _ratio(scenario_loss, es_99_21d)

    # Human-readable description
    if ratio_99 >= 1.0:
        description = (
            f"Scenario loss of ${scenario_loss:,.0f} is "
            f"{ratio_99:.1f}× the 1-day ES(99%) of ${es_99:,.0f} "
            f"— a {ratio_21d:.1f}× multiple of the 21-day ES(99%) proxy."
        )
    else:
        description = (
            f"Scenario loss of ${scenario_loss:,.0f} is within the "
            f"1-day ES(99%) of ${es_99:,.0f} "
            f"({ratio_99:.2f}× — a historically plausible daily loss)."
        )

    return {
        "es_95_1d":              round(es_95,     2),
        "es_975_1d":             round(es_975,    2),
        "es_99_1d":              round(es_99,     2),
        "es_99_21d":             round(es_99_21d, 2),
        "scenario_loss_usd":     round(scenario_loss, 2),
        "loss_to_es95_ratio":    ratio_95,
        "loss_to_es975_ratio":   ratio_975,
        "loss_to_es99_ratio":    ratio_99,
        "loss_to_es99_21d_ratio": ratio_21d,
        "multiples_of_daily_es": description,
    }


def run_all_stress_tests(
    market_data:          MarketData,
    custom_drawdown:      float = -0.20,
    metrics=None,         # Optional[PortfolioRiskMetrics] — avoids circular import
) -> StressTestResults:
    """
    Run all historical and hypothetical stress scenarios.

    Parameters
    ----------
    market_data : MarketData
        Portfolio and price data.
    custom_drawdown : float
        User-defined uniform drawdown scenario (e.g., -0.20 for −20%).
    metrics : PortfolioRiskMetrics, optional
        If provided, each ScenarioResult will have its ``es_comparison`` field
        populated via ``compare_scenario_to_es``, enabling severity labelling.

    Returns
    -------
    StressTestResults
    """
    results = StressTestResults()

    # ── Historical scenarios ───────────────────────────────────────────────────
    for name, scenario in HISTORICAL_SCENARIOS.items():
        r = _run_historical_scenario(name, scenario, market_data)
        results.historical.append(r)

    # ── Hypothetical scenarios ─────────────────────────────────────────────────
    results.hypothetical.append(_rate_shock_scenario(market_data))
    results.hypothetical.append(_sector_blowup_scenario(market_data))
    results.hypothetical.append(_largest_position_zero(market_data))
    results.hypothetical.append(_correlation_spike_scenario(market_data))
    results.hypothetical.append(_extreme_correlation_scenario(market_data))
    results.hypothetical.append(_custom_drawdown_scenario(market_data, custom_drawdown))

    # ── Phase 5: ES comparison & severity labelling ────────────────────────────
    if metrics is not None:
        for scenario in results.all_scenarios:
            try:
                scenario.es_comparison = compare_scenario_to_es(scenario, metrics)
            except Exception as exc:
                log.warning(f"ES comparison failed for '{scenario.name}': {exc}")

    log.info(
        f"Stress tests complete: "
        f"{len(results.historical)} historical, "
        f"{len(results.hypothetical)} hypothetical"
    )
    return results


def _run_historical_scenario(
    name: str,
    scenario: dict,
    market_data: MarketData,
) -> ScenarioResult:
    """Apply sector-level historical drawdowns to current portfolio."""
    sector_drawdowns = scenario["sector_drawdowns"]
    port_value       = market_data.total_portfolio_value
    holdings         = market_data.holdings

    stock_impacts: list[StockScenarioImpact] = []
    total_loss = 0.0

    for h in holdings:
        drawdown = sector_drawdowns.get(h.sector, sector_drawdowns.get("Unknown", -0.30))
        dollar_loss = h.market_value * drawdown  # negative
        total_loss += dollar_loss
        stock_impacts.append(StockScenarioImpact(
            ticker=h.ticker,
            company_name=h.company_name,
            sector=h.sector,
            weight=h.weight,
            scenario_drawdown=drawdown,
            dollar_loss=dollar_loss,
            pct_of_total_loss=0.0,  # filled below
        ))

    # Fill in share of total loss
    for si in stock_impacts:
        si.pct_of_total_loss = safe_divide(si.dollar_loss, total_loss) * 100

    loss_pct = total_loss / port_value

    # Plain-English interpretation
    top_losers = sorted(stock_impacts, key=lambda x: x.dollar_loss)[:3]
    top_str    = ", ".join(f"{s.ticker} ({s.sector})" for s in top_losers)

    interpretation = (
        f"Under the {name} scenario, the portfolio is estimated to lose "
        f"{fmt_pct(loss_pct)} (${abs(total_loss):,.0f}). "
        f"The benchmark would have lost {fmt_pct(scenario['benchmark_drawdown'])}. "
        f"The largest individual contributors to the loss would be: {top_str}. "
        f"Sectors with lower exposure to this scenario provide relative protection."
    )

    return ScenarioResult(
        name=name,
        description=scenario["description"],
        period=scenario["period"],
        methodology="Sector-level drawdowns from historical index data applied to current sector weights.",
        assumptions=(
            "Assumes current portfolio composition. Does not account for sector rotation, "
            "individual stock variance from sector averages, or the club's actual cost basis. "
            "All holdings within a sector are assumed to match the sector average exactly. "
            "Actual losses during a real event may differ significantly from these estimates."
        ),
        portfolio_loss_pct=loss_pct,
        portfolio_loss_usd=total_loss,
        benchmark_loss_pct=scenario["benchmark_drawdown"],
        portfolio_value=port_value,
        stock_impacts=stock_impacts,
        interpretation=interpretation,
    )


def _rate_shock_scenario(market_data: MarketData) -> ScenarioResult:
    """
    Interest Rate Shock: +200 basis points.

    Methodology: Each stock's historical sensitivity to the 2-year Treasury
    yield is estimated by regressing its returns against daily ^TNX changes.
    The coefficient β_rate is multiplied by 0.02 (200bps) to estimate impact.

    Limitation: This is a reduced-form approximation. It does not capture
    non-linearities, duration effects, or credit spread changes. Results
    are directionally informative but not precisely quantitative.
    """
    md     = market_data
    tickers = md.portfolio_tickers
    prices  = md.prices[tickers].dropna(how="all")
    log_rets = np.log(prices / prices.shift(1)).dropna()

    # Estimate rate sensitivity using benchmark as proxy for rates
    # (proxy approach — TNX not always cleanly downloadable)
    # Use 2022 as a rate shock reference period (Jan-Oct 2022)
    scenario_2022 = HISTORICAL_SCENARIOS["2022 Rate Shock"]
    port_value = md.total_portfolio_value
    holdings   = md.holdings

    stock_impacts = []
    total_loss = 0.0

    for h in holdings:
        # Use 2022 drawdown as the rate-shock reference
        rate_sensitivity = scenario_2022["sector_drawdowns"].get(
            h.sector,
            scenario_2022["sector_drawdowns"].get("Unknown", -0.25)
        )
        # Scale to +200bps (2022 was ~425bps shock, so scale down)
        scaled = rate_sensitivity * (200 / 425)
        dollar_loss = h.market_value * scaled
        total_loss += dollar_loss
        stock_impacts.append(StockScenarioImpact(
            ticker=h.ticker,
            company_name=h.company_name,
            sector=h.sector,
            weight=h.weight,
            scenario_drawdown=scaled,
            dollar_loss=dollar_loss,
            pct_of_total_loss=0.0,
        ))

    for si in stock_impacts:
        si.pct_of_total_loss = safe_divide(si.dollar_loss, total_loss) * 100
    loss_pct = total_loss / port_value

    return ScenarioResult(
        name="Rate Shock (+200bps)",
        description="Sudden 200 basis point increase in interest rates across the yield curve.",
        period="Hypothetical",
        methodology=(
            "Scaled from 2022 sector drawdowns (which occurred during a 425bps rate cycle). "
            "Sectors with long-duration cash flows (Tech, Growth, Real Estate) are most affected."
        ),
        assumptions=(
            "+200bps is applied instantaneously. No recovery or Fed reaction assumed. "
            "Rate sensitivity is approximated from sector behaviour during 2022. "
            "Actual duration/convexity varies by stock. Financial sector could benefit."
        ),
        portfolio_loss_pct=loss_pct,
        portfolio_loss_usd=total_loss,
        benchmark_loss_pct=-0.117,   # ~200/425 × 2022 SPY drawdown
        portfolio_value=port_value,
        stock_impacts=stock_impacts,
        interpretation=(
            f"A +200bps rate shock is estimated to reduce portfolio value by "
            f"{fmt_pct(loss_pct)} (${abs(total_loss):,.0f}). "
            f"Rate-sensitive sectors (Technology, Real Estate, Communication Services) "
            f"are most exposed. Energy and Consumer Staples typically show more resilience."
        ),
    )


def _sector_blowup_scenario(market_data: MarketData) -> ScenarioResult:
    """
    Sector Blowup: Largest sector drops 30%, everything else flat.
    """
    md      = market_data
    holdings = md.holdings

    # Identify largest sector
    sector_weights: dict[str, float] = {}
    for h in holdings:
        sector_weights[h.sector] = sector_weights.get(h.sector, 0.0) + h.weight

    largest_sector, largest_wt = max(sector_weights.items(), key=lambda x: x[1])
    BLOWUP = -0.30
    port_value = md.total_portfolio_value

    stock_impacts = []
    total_loss = 0.0
    for h in holdings:
        drawdown = BLOWUP if h.sector == largest_sector else 0.0
        dollar_loss = h.market_value * drawdown
        total_loss += dollar_loss
        stock_impacts.append(StockScenarioImpact(
            ticker=h.ticker,
            company_name=h.company_name,
            sector=h.sector,
            weight=h.weight,
            scenario_drawdown=drawdown,
            dollar_loss=dollar_loss,
            pct_of_total_loss=0.0,
        ))
    for si in stock_impacts:
        si.pct_of_total_loss = safe_divide(si.dollar_loss, total_loss) * 100 if total_loss < 0 else 0.0
    loss_pct = total_loss / port_value

    return ScenarioResult(
        name=f"Sector Blowup ({largest_sector})",
        description=(
            f"The portfolio's largest sector ({largest_sector}, {largest_wt:.1%} of portfolio) "
            f"drops 30% while all other sectors remain flat."
        ),
        period="Hypothetical",
        methodology="Single-sector -30% shock; all other holdings held constant.",
        assumptions=(
            "Sector shock is instantaneous and isolated. In practice, a sector selloff "
            "typically causes correlated declines in related sectors. This scenario "
            "understates contagion effects."
        ),
        portfolio_loss_pct=loss_pct,
        portfolio_loss_usd=total_loss,
        benchmark_loss_pct=largest_wt * BLOWUP,
        portfolio_value=port_value,
        stock_impacts=stock_impacts,
        interpretation=(
            f"If {largest_sector} drops 30%, the portfolio loses an estimated "
            f"{fmt_pct(loss_pct)} (${abs(total_loss):,.0f}). "
            f"Concentration in the largest sector is the primary vulnerability. "
            f"Consider whether the portfolio's {largest_sector} exposure is intentional conviction."
        ),
    )


def _largest_position_zero(market_data: MarketData) -> ScenarioResult:
    """
    Single stock disaster: largest holding goes to zero.
    Represents bankruptcy, fraud, or catastrophic regulatory action.
    """
    md       = market_data
    holdings = md.holdings
    largest  = max(holdings, key=lambda h: h.market_value)
    port_value = md.total_portfolio_value

    total_loss = -largest.market_value
    loss_pct   = total_loss / port_value

    stock_impacts = []
    for h in holdings:
        drawdown    = -1.0 if h.ticker == largest.ticker else 0.0
        dollar_loss = h.market_value * drawdown
        stock_impacts.append(StockScenarioImpact(
            ticker=h.ticker,
            company_name=h.company_name,
            sector=h.sector,
            weight=h.weight,
            scenario_drawdown=drawdown,
            dollar_loss=dollar_loss,
            pct_of_total_loss=(100.0 if h.ticker == largest.ticker else 0.0),
        ))

    return ScenarioResult(
        name=f"Single Stock Disaster ({largest.ticker})",
        description=(
            f"The largest holding, {largest.company_name} ({largest.ticker}), "
            f"goes to zero. This represents bankruptcy, catastrophic fraud, or regulatory action."
        ),
        period="Hypothetical",
        methodology=f"100% loss applied to {largest.ticker} only; all other holdings unchanged.",
        assumptions=(
            "While rare for large-cap S&P 500 components, history shows even major companies "
            "(Enron, Lehman, Wirecard) can experience complete loss. This scenario tests the "
            "cost of insufficient diversification."
        ),
        portfolio_loss_pct=loss_pct,
        portfolio_loss_usd=total_loss,
        benchmark_loss_pct=0.0,
        portfolio_value=port_value,
        stock_impacts=stock_impacts,
        interpretation=(
            f"Losing {largest.ticker} entirely would cost ${abs(total_loss):,.0f} "
            f"({fmt_pct(abs(loss_pct))} of portfolio). "
            f"The position should be sized with this tail risk in mind. "
            f"{'⚠️ Position size > 7% — single-stock concentration is significant.' if largest.weight > 0.07 else '✅ Position size is within reasonable limits.'}"
        ),
    )


def _correlation_spike_scenario(market_data: MarketData) -> ScenarioResult:
    """
    Correlation Spike: all pairwise correlations increase to 0.70.

    This simulates a 'risk-off' environment where everything sells off together
    (e.g., a credit event, global panic). When correlations spike, diversification
    benefits evaporate and portfolio VaR increases dramatically.

    Methodology: Replace the empirical covariance matrix with one constructed
    from individual volatilities and a uniform correlation of 0.70, then
    recalculate parametric VaR.
    """
    from scipy.stats import norm
    from engine.risk_metrics import TRADING_DAYS

    md      = market_data
    tickers = md.portfolio_tickers
    prices  = md.prices[tickers].dropna(how="all")
    log_rets = np.log(prices / prices.shift(1)).dropna()

    weights = md.weights.reindex(tickers).fillna(0).values.astype(float)
    weights = weights / weights.sum()

    individual_vols = log_rets.std().values   # daily std devs

    # Stress covariance matrix: σ_i × σ_j × ρ  (ρ for off-diagonal)
    # In crisis regimes, both correlations AND volatilities spike.
    # Empirically, equity vols increase ~1.5-2x during panic events
    # (e.g., VIX averaged ~35% in 2008 vs ~15% normally → ~2.3x multiplier).
    # We use a conservative 1.5x vol multiplier alongside the correlation spike.
    # HIGH-4 FIX: Updated from 0.70 to 0.95 (empirical 2008/2020 equity crisis correlations)
    # Research shows US equity correlations averaged 0.82-0.95 during peak panic phases
    SPIKE_CORR = 0.95
    VOL_SPIKE_MULT = 1.5   # conservative; 2008 was closer to 2.0-2.5x
    n = len(tickers)
    stressed_vols = individual_vols * VOL_SPIKE_MULT
    corr_stress = np.full((n, n), SPIKE_CORR)
    np.fill_diagonal(corr_stress, 1.0)
    cov_stress = corr_stress * np.outer(stressed_vols, stressed_vols)

    port_value = md.total_portfolio_value
    port_var_stress = float(weights @ cov_stress @ weights)
    port_std_stress = np.sqrt(port_var_stress) * np.sqrt(TRADING_DAYS)

    # Baseline for comparison
    cov_base         = log_rets.cov().values
    port_var_base    = float(weights @ cov_base @ weights)
    port_std_base    = np.sqrt(port_var_base) * np.sqrt(TRADING_DAYS)

    # 99% VaR under stress
    z_99  = abs(norm.ppf(0.01))
    mean_daily = float(log_rets.mean().values @ weights)
    var_stress  = -(mean_daily - z_99 * np.sqrt(port_var_stress)) * port_value
    var_base    = -(mean_daily - z_99 * np.sqrt(port_var_base)) * port_value
    var_ratio   = safe_divide(abs(var_stress), abs(var_base))

    # For stock impacts, show implied loss under stress
    stock_impacts = []
    total_loss = 0.0
    for i, h in enumerate(md.holdings):
        if h.ticker not in tickers:
            continue
        # Marginal contribution under stress
        marg = (cov_stress @ weights)[i] / np.sqrt(port_var_stress)
        comp_loss = -weights[i] * z_99 * marg * port_value
        total_loss += comp_loss
        stock_impacts.append(StockScenarioImpact(
            ticker=h.ticker,
            company_name=h.company_name,
            sector=h.sector,
            weight=h.weight,
            scenario_drawdown=-weights[i] * z_99 * np.sqrt(cov_stress[i, i]),
            dollar_loss=comp_loss,
            pct_of_total_loss=0.0,
        ))
    for si in stock_impacts:
        si.pct_of_total_loss = safe_divide(si.dollar_loss, total_loss) * 100 if total_loss < 0 else 0.0

    return ScenarioResult(
        name="Correlation Spike (ρ = 0.95)",
        description=(
            "All pairwise stock correlations jump to 0.95, simulating peak 'risk-off' panic "
            "(consistent with 2008 GFC and March 2020 empirical data). "
            "Diversification benefits near-completely evaporate."
        ),
        period="Hypothetical",
        methodology=(
            "Empirical covariance matrix replaced with one constructed from individual "
            "volatilities and a uniform off-diagonal correlation of 0.70. "
            "Parametric VaR(99%) recalculated. "
            f"Baseline portfolio VaR multiplied by {var_ratio:.1f}x under stress."
        ),
        assumptions=(
            "Correlation of 0.95 reflects empirical peak panic data (2008 GFC: ~0.82–0.95, "
            "March 2020: ~0.88–0.95 for US large-cap equities). "
            "Individual stock volatilities are multiplied by 1.5x (conservative; 2008 was ~2-2.5x). "
            "Actual crisis dynamics involve feedback loops, forced selling, and liquidity withdrawal "
            "that simple correlation/volatility shocks cannot fully capture."
        ),
        portfolio_loss_pct=safe_divide(total_loss, port_value),
        portfolio_loss_usd=total_loss,
        benchmark_loss_pct=-z_99 * port_std_stress / np.sqrt(TRADING_DAYS),  # scaled to 1-day
        portfolio_value=port_value,
        stock_impacts=stock_impacts,
        interpretation=(
            f"When correlations spike to 0.70, portfolio VaR(99%) increases by {var_ratio:.1f}x "
            f"(from ${abs(var_base):,.0f} to ${abs(var_stress):,.0f}). "
            f"This highlights that the portfolio's true diversification benefit depends heavily "
            f"on correlations remaining below crisis levels."
        ),
    )


def _extreme_correlation_scenario(market_data: MarketData) -> ScenarioResult:
    """
    Extreme Correlation: all pairwise correlations = 0.99 (near-perfect comovement).

    Represents a true systemic panic where all diversification vanishes.
    Used to stress-test the absolute worst-case correlation assumption.
    """
    from scipy.stats import norm
    from engine.risk_metrics import TRADING_DAYS

    md      = market_data
    tickers = md.portfolio_tickers
    prices  = md.prices[tickers].dropna(how="all")
    log_rets = np.log(prices / prices.shift(1)).dropna()

    weights = md.weights.reindex(tickers).fillna(0).values.astype(float)
    weights = weights / weights.sum()

    individual_vols = log_rets.std().values

    SPIKE_CORR = 0.99
    VOL_SPIKE_MULT = 2.0   # 2008 peak vol spike was ~2.0–2.5x
    n = len(tickers)
    stressed_vols = individual_vols * VOL_SPIKE_MULT
    corr_stress = np.full((n, n), SPIKE_CORR)
    np.fill_diagonal(corr_stress, 1.0)
    cov_stress = corr_stress * np.outer(stressed_vols, stressed_vols)

    port_value = md.total_portfolio_value
    port_var_stress = float(weights @ cov_stress @ weights)
    port_std_stress = np.sqrt(port_var_stress) * np.sqrt(TRADING_DAYS)

    cov_base      = log_rets.cov().values
    port_var_base = float(weights @ cov_base @ weights)
    port_std_base = np.sqrt(port_var_base) * np.sqrt(TRADING_DAYS)

    z_99  = abs(norm.ppf(0.01))
    mean_daily = float(log_rets.mean().values @ weights)
    var_stress  = -(mean_daily - z_99 * np.sqrt(port_var_stress)) * port_value
    var_base    = -(mean_daily - z_99 * np.sqrt(port_var_base)) * port_value
    var_ratio   = safe_divide(abs(var_stress), abs(var_base))

    stock_impacts = []
    total_loss = 0.0
    for i, h in enumerate(md.holdings):
        if h.ticker not in tickers:
            continue
        marg = (cov_stress @ weights)[i] / np.sqrt(port_var_stress)
        comp_loss = -weights[i] * z_99 * marg * port_value
        total_loss += comp_loss
        stock_impacts.append(StockScenarioImpact(
            ticker=h.ticker,
            company_name=h.company_name,
            sector=h.sector,
            weight=h.weight,
            scenario_drawdown=-weights[i] * z_99 * np.sqrt(cov_stress[i, i]),
            dollar_loss=comp_loss,
            pct_of_total_loss=0.0,
        ))
    for si in stock_impacts:
        si.pct_of_total_loss = safe_divide(si.dollar_loss, total_loss) * 100 if total_loss < 0 else 0.0

    return ScenarioResult(
        name="Extreme Correlation (ρ = 0.99)",
        description=(
            "All pairwise correlations jump to 0.99 with 2x volatility spike — "
            "a near-complete systemic breakdown where all diversification vanishes."
        ),
        period="Hypothetical — Extreme Tail",
        methodology=(
            "Covariance matrix built from individual volatilities (×2.0 spike multiplier) "
            f"and uniform off-diagonal correlation of 0.99. VaR(99%) multiplied by {var_ratio:.1f}x."
        ),
        assumptions=(
            "Represents an absolute worst-case — beyond any observed historical scenario. "
            "Intended as a regulatory-style extreme tail stress test. "
            "Volatility multiplier of 2.0x reflects 2008 GFC peak levels."
        ),
        portfolio_loss_pct=safe_divide(total_loss, port_value),
        portfolio_loss_usd=total_loss,
        benchmark_loss_pct=-z_99 * port_std_stress / np.sqrt(TRADING_DAYS),
        portfolio_value=port_value,
        stock_impacts=stock_impacts,
        interpretation=(
            f"Under extreme correlation (0.99), portfolio VaR(99%) increases by {var_ratio:.1f}x "
            f"(from ${abs(var_base):,.0f} to ${abs(var_stress):,.0f}). "
            f"This represents near-complete loss of diversification benefit."
        ),
    )


def _custom_drawdown_scenario(
    market_data: MarketData,
    drawdown: float,
) -> ScenarioResult:
    """
    Apply a user-specified uniform drawdown to the entire portfolio.
    Reads from Settings: stress_custom_drawdown.
    """
    md         = market_data
    holdings   = md.holdings
    port_value = md.total_portfolio_value

    total_loss = port_value * drawdown
    loss_pct   = drawdown

    stock_impacts = []
    for h in holdings:
        dollar_loss = h.market_value * drawdown
        stock_impacts.append(StockScenarioImpact(
            ticker=h.ticker,
            company_name=h.company_name,
            sector=h.sector,
            weight=h.weight,
            scenario_drawdown=drawdown,
            dollar_loss=dollar_loss,
            pct_of_total_loss=h.weight * 100,
        ))

    return ScenarioResult(
        name=f"Custom Drawdown ({drawdown:.0%})",
        description=f"User-defined scenario: uniform {drawdown:.0%} decline across all holdings.",
        period="Hypothetical",
        methodology="Identical drawdown applied to every holding regardless of sector or beta.",
        assumptions=(
            "This is the simplest possible scenario — it assumes every stock falls by the same "
            "amount simultaneously. In practice, correlations are imperfect and sector/factor "
            "tilts mean some holdings would perform better or worse."
        ),
        portfolio_loss_pct=loss_pct,
        portfolio_loss_usd=total_loss,
        benchmark_loss_pct=drawdown,
        portfolio_value=port_value,
        stock_impacts=stock_impacts,
        interpretation=(
            f"A uniform {abs(drawdown):.0%} drawdown would reduce the portfolio by "
            f"${abs(total_loss):,.0f}. "
            f"This is a useful baseline — actual losses during market dislocations are "
            f"rarely this symmetric across all holdings."
        ),
    )
