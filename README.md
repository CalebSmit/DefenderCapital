# Defender Capital Management — Portfolio Risk Model

A production-grade portfolio risk analytics system built for institutional-quality risk management. Live market data, Cholesky-correlated Monte Carlo simulation, Euler VaR decomposition, and 9 stress test scenarios — all in a 6-page interactive dashboard.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Folder Structure](#folder-structure)
5. [How to Use](#how-to-use)
6. [Risk Metrics Glossary](#risk-metrics-glossary)
7. [Stress Test Scenarios](#stress-test-scenarios)
8. [Configuration](#configuration)
9. [Deployment](#deployment)
10. [Development & Testing](#development--testing)

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Internet connection (for live market data via Yahoo Finance)

### Install & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard opens at **http://localhost:8501**.

### One-Click Setup (Optional)

```bash
# macOS / Linux
bash setup.sh && bash run_dashboard.sh

# Windows
setup.bat
run_dashboard.bat
```

---

## Features

**Risk Analytics Engine**
- 4 VaR methods: Parametric, Historical, Monte Carlo, Cornish-Fisher (skew/kurtosis adjusted)
- CVaR / Expected Shortfall at 95% and 99% confidence
- Euler VaR decomposition — per-holding risk contribution that sums exactly to total VaR
- Ledoit-Wolf shrinkage covariance estimation
- Sharpe, Sortino, Calmar, Beta, Jensen's Alpha, Diversification Ratio
- PCA factor analysis

**Monte Carlo Simulation**
- 10,000 Cholesky-correlated paths over 252 trading days
- Fan chart with P5/P25/P50/P75/P95 percentile bands
- Terminal value distribution with probability analysis

**Stress Testing**
- 4 historical scenarios (2008 GFC, COVID-19, 2022 Rate Shock, Dot-Com Bust)
- 5 hypothetical scenarios (Rate Shock +200bps, Tech Blowup, Single-Stock Disaster, Correlation Spike, Custom Drawdown)
- Stock-level impact breakdown for each scenario

**Dashboard**
- 6 interactive pages with Plotly charts
- Live market data from Yahoo Finance (no synthetic fallback)
- Portfolio upload via drag-and-drop Excel
- HTML report generation and Excel write-back
- Responsive design

---

## Architecture

```
portfolio_holdings.xlsx  <--  User edits holdings here (or uploads via dashboard)
        |
        v
engine/data_loader.py    <--  Validates & parses Excel input
        |
        v
engine/market_data.py    <--  Fetches live prices from Yahoo Finance
        |
        v
engine/risk_metrics.py   <--  VaR, CVaR, Sharpe, Beta, drawdown, Euler decomposition
engine/monte_carlo.py    <--  10,000-path Cholesky-correlated simulation
engine/stress_testing.py <--  4 historical + 5 hypothetical scenarios
        |
        |-->  engine/report_generator.py  <--  HTML report + Excel write-back
        |
        '-->  dashboard/app.py            <--  6-page Streamlit dashboard
```

**Data flow:** The Excel file is the single source of truth for holdings. The engine fetches live prices and populates calculated fields (current price, market value, weight) automatically. All outputs are written to `data/exports/`.

---

## Folder Structure

```
DefenderCapital/
|-- data/
|   |-- portfolio_holdings.xlsx    <-- YOUR PORTFOLIO (edit this)
|   '-- exports/                   <-- Generated reports & audit logs
|-- engine/
|   |-- data_loader.py             <-- Excel parser & validator
|   |-- market_data.py             <-- yfinance live data fetcher
|   |-- synthetic_data.py          <-- Synthetic data generator (testing only)
|   |-- risk_metrics.py            <-- Full risk analytics engine
|   |-- monte_carlo.py             <-- Monte Carlo simulation
|   |-- stress_testing.py          <-- Scenario analysis
|   |-- report_generator.py        <-- HTML/Excel output
|   '-- utils.py                   <-- Logging, paths, helpers
|-- dashboard/
|   '-- app.py                     <-- Streamlit 6-page dashboard
|-- tests/
|   |-- conftest.py                <-- Shared pytest fixtures
|   |-- test_data_loader.py        <-- 23 tests
|   |-- test_risk_metrics.py       <-- 26 tests
|   |-- test_monte_carlo.py        <-- 16 tests
|   '-- test_stress_testing.py     <-- 21 tests
|-- .streamlit/
|   '-- config.toml                <-- Theme configuration
|-- setup.sh / setup.bat           <-- One-time environment setup
|-- run_dashboard.sh / .bat        <-- Launch the dashboard
|-- update_model.sh / .bat         <-- Refresh data & generate report
|-- requirements.txt               <-- Python dependencies
'-- README.md                      <-- This file
```

---

## How to Use

### Option A — Upload via Dashboard

1. Launch the dashboard: `streamlit run dashboard/app.py`
2. In the sidebar, use the **Upload Portfolio** widget
3. Upload an Excel file (.xlsx) with a "Holdings" sheet containing:
   - **Ticker** — stock ticker symbol (e.g., AAPL)
   - **Shares Held** — number of shares
   - **Cost Basis** — average purchase price per share
4. The dashboard will automatically fetch live prices and run all analytics

### Option B — Edit the Excel File Directly

Open `data/portfolio_holdings.xlsx` and fill in the Holdings sheet:

| Column | Required | Description | Example |
|---|---|---|---|
| Ticker | Yes | Stock ticker (uppercase) | AAPL |
| Shares Held | Yes | Number of shares | 500 |
| Cost Basis | Yes | Average price per share ($) | 145.00 |
| Company Name | No | Auto-populated from yfinance | Apple Inc. |
| Sector | No | Auto-populated from yfinance | Technology |
| Industry | No | Auto-populated from yfinance | Consumer Electronics |

Only Ticker, Shares Held, and Cost Basis are required. All other fields are auto-populated.

### Configure Settings (Optional)

On the **Settings** sheet of the Excel file:

| Parameter | Default | Description |
|---|---|---|
| benchmark_ticker | SPY | Benchmark index for Beta & Alpha |
| risk_free_rate | auto | "auto" fetches 10Y Treasury yield; or enter a decimal (0.045) |
| lookback_years | 2 | Historical data window |
| simulation_paths | 10000 | Monte Carlo paths |
| confidence_level_1 | 0.95 | Primary VaR confidence |
| confidence_level_2 | 0.99 | Secondary VaR confidence |

### Dashboard Pages

- **Portfolio Overview** — Sector allocation, weight distribution, concentration metrics
- **Risk Dashboard** — VaR/CVaR, rolling volatility, drawdown, full risk table
- **Monte Carlo** — 1-year forward simulation fan chart, terminal value distribution
- **Stress Tests** — All 9 scenario results with stock-level breakdown
- **Stock Analysis** — Per-holding risk table, component VaR chart, correlation heatmap
- **Reports & Export** — Generate HTML report, export to Excel, view data quality

---

## Risk Metrics Glossary

### Value at Risk (VaR)

The maximum expected loss over a given time horizon at a specified confidence level.

**Parametric:** `VaR = -(u - z*s) * PV` where u = mean daily return, s = daily std dev, z = norm.ppf(1 - confidence).

**Four methods implemented:**
- *Parametric (Normal)* — assumes normal distribution
- *Historical* — empirical percentile of actual returns
- *Monte Carlo* — simulated from 10,000 correlated paths
- *Cornish-Fisher* — adjusts for skewness and kurtosis in the return distribution

### CVaR / Expected Shortfall

Average loss in the worst a% of scenarios (the tail beyond VaR). CVaR >= VaR always. CVaR is a coherent risk measure; VaR is not.

### Component VaR (Euler Decomposition)

Each holding's contribution to total portfolio VaR. The sum of all component VaRs equals total portfolio VaR exactly (Euler identity).

### Sharpe Ratio

`(R_p - R_f) / s_p` — return earned per unit of total risk.

### Sortino Ratio

Like Sharpe, but uses downside deviation instead of total volatility. Does not penalize upside moves.

### Beta

Sensitivity of the portfolio to benchmark movements. Beta = 1.0 means 1:1 with the market.

### Jensen's Alpha

Excess return above CAPM prediction: `a = R_p - [R_f + B*(R_m - R_f)]`. Positive alpha = genuine outperformance.

### Maximum Drawdown

Peak-to-trough decline in portfolio value over the measurement period.

### HHI / Effective Number of Bets

HHI = sum of squared weights. ENB = 1/HHI. Measures portfolio concentration.

### Diversification Ratio

Weighted average individual volatility divided by portfolio volatility. DR > 1 means diversification is working.

---

## Stress Test Scenarios

### Historical Scenarios

| Scenario | Period | Typical Loss |
|---|---|---|
| 2008 Global Financial Crisis | Sep 2008 - Mar 2009 | -30% to -55% |
| COVID-19 Crash | Feb 2020 - Mar 2020 | -20% to -35% |
| 2022 Rate Shock | Jan 2022 - Oct 2022 | -15% to -40% |
| Dot-Com Bust | Mar 2000 - Oct 2002 | -20% to -50% |

### Hypothetical Scenarios

| Scenario | Description |
|---|---|
| Rate Shock +200bps | +2% rise in rates — hurts bonds, REITs, growth stocks |
| Tech Sector Blowup | Technology falls 40%, rest unchanged |
| Single-Stock Disaster | Largest holding falls 80% |
| Correlation Spike | All pairwise correlations snap to 0.70, volatility spikes 1.5x |
| Custom Drawdown | Configurable uniform decline (default: -20%) |

---

## Configuration

All settings are in the **Settings** sheet of `data/portfolio_holdings.xlsx`. No code changes required.

### Key files

| File | Purpose |
|---|---|
| `data/portfolio_holdings.xlsx` | Holdings and settings |
| `.streamlit/config.toml` | Dashboard theme |
| `data/exports/` | Generated reports |

---

## Deployment

### Local (Default)

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

### Docker

```bash
docker build -t dcm-risk .
docker run -p 8501:8501 dcm-risk
```

### Streamlit Cloud

1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set main file path: `DefenderCapital/dashboard/app.py`

---

## Development & Testing

### Run all tests

```bash
python -m pytest tests/ -v
```

86 tests covering data loading, VaR formulas, Euler decomposition, Monte Carlo statistics, and stress test scenarios.

### Requirements

- Python >= 3.10
- openpyxl >= 3.1
- pandas >= 2.0
- numpy >= 1.26
- yfinance >= 0.2.36
- scipy >= 1.11
- scikit-learn >= 1.3
- plotly >= 5.17
- streamlit >= 1.41
- kaleido >= 0.2.1
- requests >= 2.31
- jinja2 >= 3.1
- pytest >= 7.4

---

*Defender Capital Management*
