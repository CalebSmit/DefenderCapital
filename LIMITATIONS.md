# Known Limitations & Model Assumptions

This document describes the assumptions, simplifications, and limitations of the Defender Capital Management Portfolio Risk Model. **Read this before presenting results to a professional audience.**

---

## 1. Distributional Assumptions (Parametric VaR)

**What we assume:** The parametric VaR assumes portfolio returns follow a normal (Gaussian) distribution.

**Reality:** Equity return distributions have:
- **Fat tails (excess kurtosis > 0)** — Extreme events occur more frequently than a normal distribution predicts. The 2008 GFC and COVID-19 crash were 5-10 standard deviation events under normality, but occurred in reality.
- **Negative skewness** — Large losses are more likely than large gains of equivalent magnitude.
- **Volatility clustering** — Periods of high volatility tend to cluster (GARCH effects). Our model uses constant volatility estimated over the full lookback period.

**Impact:** The standard parametric VaR likely **understates** true tail risk by 20-50% in stress periods. Historical VaR partially corrects for this if the look-back period includes a crisis.

**Mitigations in this model:**
1. **Cornish-Fisher VaR (CF-VaR):** We compute a modified VaR that adjusts the normal quantile for observed skewness and excess kurtosis using the Cornish-Fisher expansion (CFA Level II standard). For typical equity portfolios, CF-VaR is 10-30% higher than the normal parametric VaR.
2. **Four VaR methods:** We provide parametric (normal), Cornish-Fisher (fat-tail adjusted), historical (non-parametric), and Monte Carlo VaR. Always compare all four; if they diverge significantly, the normal parametric estimate is most suspect.
3. **CVaR (Expected Shortfall):** Reports the expected loss *given* that VaR is breached, providing a more complete picture of tail risk.

---

## 2. Covariance Estimation

**Ledoit-Wolf Shrinkage:** The model uses the Ledoit-Wolf (2004) shrinkage estimator rather than the sample covariance matrix. This is critical because:
- With p=35 assets and T=504 observations, the sample covariance has p(p+1)/2 = 630 free parameters, yielding a p/T ratio of ~0.07.
- The sample covariance matrix amplifies estimation noise, leading to unstable VaR estimates and portfolio optimisation artifacts.
- Ledoit-Wolf analytically determines the optimal shrinkage intensity towards a structured target (scaled identity), minimising the Frobenius-norm estimation error.

**Remaining limitations:**
- Shrinkage improves conditioning but still assumes a static covariance structure. True correlations are time-varying (DCC-GARCH would address this but is not implemented).
- The shrinkage target is a scaled identity matrix. Alternative targets (e.g., single-factor model, constant correlation) may be more appropriate depending on portfolio structure.

**Reference:** Ledoit, O. & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365-411.

---

## 3. Monte Carlo Simulation

**Constant parameters assumption:** The simulation uses mean returns and the Ledoit-Wolf covariance matrix estimated from the historical look-back period. In reality:
- Expected returns are highly uncertain — the Sharpe ratio of any individual stock is estimated with enormous error over 2 years of data.
- Correlations change dramatically during crises (correlation spikes — the diversification collapse scenario partially addresses this).
- Volatility is not constant (heteroscedasticity).

**Geometric Brownian Motion:** We simulate log-normal price paths (GBM). This is the foundation of the Black-Scholes framework but:
- Does not capture jump discontinuities (earnings surprises, regulatory announcements).
- Does not capture mean reversion in volatility.
- Assumes continuous trading with no liquidity premium.

**Reproducibility:** The simulation uses `numpy.random.Generator` with a fixed seed (default: 42) for deterministic reproducibility. The same inputs will always produce the same simulation results.

**Confidence in VaR vs. CVaR:** With 10,000 paths and a 1% tail (100 paths), the CVaR(99%) estimate has high standard error. CVaR(95%) is more reliable (500 paths in tail).

---

## 4. Historical Stress Tests

**Survivorship bias:** The stress test scenarios apply historical sector-level drawdowns to the *current* portfolio. Companies that went bankrupt during those crises (Lehman Brothers, Bear Stearns, Washington Mutual) are not in today's portfolios. This means our GFC and Dot-Com scenarios may **understate** losses for a 2008/2000 portfolio composition.

**Sector aggregation:** We apply a uniform drawdown to all companies within a sector. In reality, individual stocks within a sector have dramatically different performances (some Energy stocks gained +100% in 2022 while others fell).

**Point-in-time calculation:** Stress tests apply a single instantaneous shock. They do not model the multi-month deterioration of the actual crisis, feedback loops, liquidity crises, or forced selling.

**Correlation spike scenario:** The correlation spike scenario applies both a correlation increase (to 0.70) **and** a volatility multiplier (1.5x) to all holdings. In the 2008 GFC, actual equity correlations averaged ~0.82 and volatilities increased ~2-2.5x, so our scenario is conservative. The combined correlation + volatility shock is more realistic than a correlation-only shock.

**Historical scenarios are not forecasts:** The fact that we experienced a 48% drawdown in the GFC does not mean the *next* crisis will produce a 48% loss. It is illustrative, not predictive.

---

## 5. Market Data & Liquidity

**No liquidity adjustment:** VaR is computed assuming you can exit any position at the current mid-price. In reality:
- Bid-ask spreads widen dramatically during stress periods.
- Large positions in small-cap or thinly-traded stocks cannot be unwound at market prices.
- This model does not penalize concentration in illiquid names.

**Price source:** Yahoo Finance (or synthetic data in offline mode). Yahoo Finance prices are adjusted for splits and dividends but:
- May have occasional data errors or gaps.
- Do not reflect after-hours trading.
- Adjusted prices for delisted stocks are unavailable.

**Synthetic data fallback:** When Yahoo Finance is unreachable, the system generates synthetic prices calibrated to approximate historical parameters. This is adequate for demonstrating the analytical framework but should **not** be used for actual investment decisions. All production use requires live market data.

---

## 6. Return Window (Look-Back Period)

**Default: 2 years (504 trading days).** This means:
- The model "forgets" anything that happened before 2 years ago (e.g., COVID-19 drawdown).
- 2 years of daily data gives ~504 observations per asset. While the Ledoit-Wolf shrinkage estimator mitigates the worst effects of limited data, the underlying signal-to-noise ratio for expected returns is inherently low with 2 years of data.
- The mean return estimate has a standard error of approximately sigma/sqrt(504) per stock. For a 25% annualised vol stock, this is roughly 1.1% per year — making it difficult to distinguish genuine alpha from noise.

**Regime sensitivity:** If the look-back period falls entirely within a bull market, volatility and correlation estimates will be understated for stress periods.

---

## 7. Single-Asset & Concentration Risks

**No options or derivatives:** The model covers long equity positions only. Short positions, options, and leveraged instruments are not supported. If DCM holds warrants, convertibles, or other derivatives, their risk is not captured.

**No cross-asset correlation:** The model does not capture correlations with bonds, commodities, or currencies. A concentrated position in a stock with significant FX exposure (e.g., AAPL's 60%+ non-US revenue) has currency risk that is invisible to this model.

**Factor exposures:** The PCA decomposition identifies factor structure but does not label factors (growth, value, momentum, rates, etc.). Users must interpret factor loadings qualitatively.

---

## 8. Benchmark & Risk-Free Rate

**Benchmark = SPY:** Beta and Alpha are computed relative to the S&P 500 ETF. This is appropriate for a US large-cap equity portfolio. If DCM holds significant small-cap, international, or sector-specific positions, SPY is not the appropriate benchmark.

**Risk-free rate = 10-year Treasury yield:** In practice, short-duration risk-free rates (3-month T-bill, Fed Funds) are more commonly used for daily VaR and Sharpe ratio calculations. The 10-year yield is appropriate for longer-horizon metrics but introduces a slight inconsistency for 1-day VaR. The user can override this in Settings with a manual rate.

---

## 9. Regulatory & Professional Standards

This model is built for **educational and internal analysis purposes only.**

- It does **not** comply with Basel III/IV internal models approach (IMA) for regulatory capital.
- It does **not** implement FRTB (Fundamental Review of the Trading Book) requirements.
- VaR estimates from this model should **not** be used to set regulatory capital requirements.
- For SEC filings, fund prospectuses, or external investor reports, consult a qualified risk manager or financial advisor.

**This software is provided "as is" with no warranty, express or implied, as to its accuracy or fitness for any particular purpose.**

---

## 10. Model Validation

The model passes 4 formal audit phases covering 40+ automated checks, including:
- Parametric VaR formula verification (exact match to hand calculation)
- Cornish-Fisher VaR formula verification
- Historical VaR percentile verification
- CVaR >= VaR property
- Euler decomposition identity (Sum of Component VaR = Total VaR, to $0.00 precision)
- Monte Carlo mean return consistency with historical
- Ledoit-Wolf covariance positive-definiteness
- 86 pytest unit tests across all modules

However, audits verify mathematical correctness, not appropriateness of assumptions. **Garbage in, garbage out** -- the model is only as good as the underlying data and assumptions.

---

## 11. What This Model Does NOT Do

| Feature | Status |
|---|---|
| Options / derivatives pricing | Not supported |
| Liquidity-adjusted VaR (L-VaR) | Not implemented |
| Conditional VaR with regime switching | Not implemented |
| Dynamic correlation (DCC-GARCH) | Not implemented |
| Multi-period VaR scaling | Uses sqrt(T) scaling (approximate) |
| Transaction cost modeling | Not supported |
| Short positions | Not supported |
| Fixed income, FX, commodities | Equities only |
| Regulatory capital calculation | Not for regulatory use |
| Real-time intraday pricing | End-of-day prices only |
| Ledoit-Wolf covariance shrinkage | Implemented (analytical optimal) |
| Cornish-Fisher VaR adjustment | Implemented (skewness + kurtosis) |
| Euler VaR decomposition | Implemented (exact attribution) |

---

*Last updated: February 2026 -- Defender Capital Management*
