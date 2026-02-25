"""
engine/synthetic_data.py — Synthetic market data generator for offline testing.

Generates realistic correlated price series based on known historical parameters.
Used when live yfinance data is unavailable (e.g., during testing or offline use).
The system automatically falls back to this when network requests fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from engine.data_loader import Holding, LoadResult, PortfolioSettings
from engine.market_data import MarketData, DataQualityReport
from engine.utils import get_logger

log = get_logger("defender.synthetic")

# ── Known approximate parameters for common US large-cap stocks ────────────────
# (annualized return, annualized volatility, beta vs SPY)
# Source: approximate historical values 2020-2024
STOCK_PARAMS: dict[str, tuple[float, float, float, float]] = {
    # ticker: (ann_ret, ann_vol, beta, price_approx)
    "AAPL":  (0.28,  0.30, 1.20, 182.0),
    "MSFT":  (0.32,  0.28, 1.15, 378.0),
    "NVDA":  (0.85,  0.65, 1.75, 505.0),
    "GOOGL": (0.25,  0.27, 1.10, 152.0),
    "CRM":   (0.18,  0.36, 1.30, 275.0),
    "JPM":   (0.22,  0.24, 1.05,  195.0),
    "BAC":   (0.15,  0.30, 1.20,   38.0),
    "GS":    (0.18,  0.26, 1.15,  465.0),
    "V":     (0.20,  0.20, 0.95,  268.0),
    "JNJ":   (0.05,  0.15, 0.60,  155.0),
    "UNH":   (0.22,  0.20, 0.75,  520.0),
    "PFE":   (-0.12, 0.25, 0.65,   26.0),
    "ABBV":  (0.15,  0.22, 0.70,  173.0),
    "PG":    (0.12,  0.15, 0.55,  161.0),
    "KO":    (0.08,  0.14, 0.55,   60.0),
    "WMT":   (0.28,  0.18, 0.55,  183.0),
    "AMZN":  (0.38,  0.35, 1.25,  185.0),
    "HD":    (0.15,  0.22, 1.00,  365.0),
    "NKE":   (-0.05, 0.28, 1.05,   74.0),
    "HON":   (0.10,  0.20, 0.90,  206.0),
    "CAT":   (0.25,  0.26, 1.10,  344.0),
    "UPS":   (-0.08, 0.22, 0.85,  133.0),
    "XOM":   (0.22,  0.25, 0.95,  105.0),
    "CVX":   (0.18,  0.24, 0.90,  155.0),
    "META":  (0.62,  0.42, 1.35,  503.0),
    "DIS":   (-0.05, 0.28, 1.10,   99.0),
    "LIN":   (0.18,  0.18, 0.80,  440.0),
    "APD":   (0.05,  0.20, 0.85,  264.0),
    "AMT":   (-0.05, 0.22, 0.70,  186.0),
    "PLD":   (0.10,  0.25, 0.85,  117.0),
    "NEE":   (-0.10, 0.20, 0.60,   68.0),
    "DUK":   (0.05,  0.15, 0.55,   92.0),
    "TSLA":  (0.05,  0.65, 1.80,  195.0),
    "COST":  (0.32,  0.20, 0.90,  893.0),
    "LMT":   (0.12,  0.18, 0.65,  467.0),
    "SPY":   (0.18,  0.17, 1.00,  510.0),  # benchmark
}

# ── Metadata for known tickers (sector / industry / company name) ─────────────
STOCK_META: dict[str, tuple[str, str, str]] = {
    # ticker: (company_name, sector, industry)
    "AAPL":  ("Apple Inc.",           "Technology",           "Consumer Electronics"),
    "MSFT":  ("Microsoft Corp.",      "Technology",           "Software-Infrastructure"),
    "NVDA":  ("NVIDIA Corp.",         "Technology",           "Semiconductors"),
    "GOOGL": ("Alphabet Inc. Cl A",   "Communication Services","Internet Content"),
    "GOOG":  ("Alphabet Inc. Cl C",   "Communication Services","Internet Content"),
    "META":  ("Meta Platforms",       "Communication Services","Internet Content"),
    "AMZN":  ("Amazon.com Inc.",      "Consumer Cyclical",    "Internet Retail"),
    "TSLA":  ("Tesla Inc.",           "Consumer Cyclical",    "Auto Manufacturers"),
    "CRM":   ("Salesforce Inc.",      "Technology",           "Software-Application"),
    "JPM":   ("JPMorgan Chase",       "Financial Services",   "Diversified Banks"),
    "BAC":   ("Bank of America",      "Financial Services",   "Diversified Banks"),
    "GS":    ("Goldman Sachs",        "Financial Services",   "Capital Markets"),
    "MS":    ("Morgan Stanley",       "Financial Services",   "Capital Markets"),
    "V":     ("Visa Inc.",            "Financial Services",   "Credit Services"),
    "MA":    ("Mastercard Inc.",      "Financial Services",   "Credit Services"),
    "AXP":   ("American Express",     "Financial Services",   "Credit Services"),
    "JNJ":   ("Johnson & Johnson",    "Healthcare",           "Drug Manufacturers"),
    "UNH":   ("UnitedHealth Group",   "Healthcare",           "Healthcare Plans"),
    "PFE":   ("Pfizer Inc.",          "Healthcare",           "Drug Manufacturers"),
    "ABBV":  ("AbbVie Inc.",          "Healthcare",           "Drug Manufacturers"),
    "LLY":   ("Eli Lilly",            "Healthcare",           "Drug Manufacturers"),
    "MRK":   ("Merck & Co.",          "Healthcare",           "Drug Manufacturers"),
    "PG":    ("Procter & Gamble",     "Consumer Defensive",   "Household Products"),
    "KO":    ("Coca-Cola Co.",        "Consumer Defensive",   "Beverages-Non-Alcoholic"),
    "WMT":   ("Walmart Inc.",         "Consumer Defensive",   "Discount Stores"),
    "COST":  ("Costco Wholesale",     "Consumer Defensive",   "Discount Stores"),
    "HD":    ("Home Depot",           "Consumer Cyclical",    "Home Improvement Retail"),
    "NKE":   ("Nike Inc.",            "Consumer Cyclical",    "Footwear & Accessories"),
    "HON":   ("Honeywell Intl.",      "Industrials",          "Diversified Industrials"),
    "CAT":   ("Caterpillar Inc.",     "Industrials",          "Farm & Heavy Construction"),
    "UPS":   ("United Parcel Service","Industrials",          "Integrated Freight"),
    "LMT":   ("Lockheed Martin",      "Industrials",          "Aerospace & Defense"),
    "BA":    ("Boeing Co.",           "Industrials",          "Aerospace & Defense"),
    "XOM":   ("ExxonMobil Corp.",     "Energy",               "Oil & Gas Integrated"),
    "CVX":   ("Chevron Corp.",        "Energy",               "Oil & Gas Integrated"),
    "LIN":   ("Linde plc",            "Basic Materials",      "Specialty Chemicals"),
    "APD":   ("Air Products",         "Basic Materials",      "Specialty Chemicals"),
    "AMT":   ("American Tower",       "Real Estate",          "REIT-Specialty"),
    "PLD":   ("Prologis Inc.",        "Real Estate",          "REIT-Industrial"),
    "DLR":   ("Digital Realty Trust", "Real Estate",          "REIT-Specialty"),
    "NEE":   ("NextEra Energy",       "Utilities",            "Utilities-Regulated Electric"),
    "DUK":   ("Duke Energy",          "Utilities",            "Utilities-Regulated Electric"),
    "DIS":   ("Walt Disney Co.",      "Communication Services","Entertainment"),
    "NFLX":  ("Netflix Inc.",         "Communication Services","Entertainment"),
    "INTC":  ("Intel Corp.",          "Technology",           "Semiconductors"),
    "AMD":   ("Advanced Micro Devices","Technology",          "Semiconductors"),
    "QCOM":  ("Qualcomm Inc.",        "Technology",           "Semiconductors"),
    "AVGO":  ("Broadcom Inc.",        "Technology",           "Semiconductors"),
    "ADBE":  ("Adobe Inc.",           "Technology",           "Software-Application"),
    "PYPL":  ("PayPal Holdings",      "Financial Services",   "Credit Services"),
    "SYK":   ("Stryker Corp.",        "Healthcare",           "Medical Devices"),
    "ELV":   ("Elevance Health",      "Healthcare",           "Healthcare Plans"),
    "LTH":   ("Life Time Group",      "Consumer Cyclical",    "Leisure"),
    "CARR":  ("Carrier Global",       "Industrials",          "Building Products & Equipment"),
    "CROX":  ("Crocs Inc.",           "Consumer Cyclical",    "Footwear & Accessories"),
    "SMR":   ("NuScale Power",        "Utilities",            "Utilities-Renewable"),
    "SPY":   ("SPDR S&P 500 ETF",     "ETF",                  "Broad Market ETF"),
}

# Sector correlation structure (approximate)
SECTOR_BASE_CORR = 0.55   # average intra-sector correlation
MARKET_CORR      = 0.45   # average cross-sector correlation


def generate_synthetic_market_data(
    load_result: LoadResult,
    n_days: int = 504,
    seed: int = 42,
) -> MarketData:
    """
    Generate synthetic correlated price history for all portfolio holdings.

    Uses Cholesky decomposition to create correlated returns matching
    approximate real-world correlations. Prices end near known market levels.

    Parameters
    ----------
    load_result : LoadResult
        Holdings and settings from data_loader.
    n_days : int
        Number of trading days to generate (default: 504 ≈ 2 years).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MarketData
        Fully populated MarketData with synthetic prices.
    """
    rng = np.random.default_rng(seed)
    log.info(f"Generating synthetic market data: {n_days} days for {len(load_result.holdings)} tickers")

    holdings = load_result.holdings
    settings = load_result.settings
    benchmark = settings.benchmark_ticker

    all_tickers = [h.ticker for h in holdings] + [benchmark]
    n_tickers   = len(all_tickers)

    # ── Build correlation matrix ───────────────────────────────────────────────
    corr = np.full((n_tickers, n_tickers), MARKET_CORR)
    np.fill_diagonal(corr, 1.0)

    # Slightly higher intra-sector correlations
    sector_map = {h.ticker: h.sector for h in holdings}
    for i, t1 in enumerate(all_tickers[:-1]):
        for j, t2 in enumerate(all_tickers[:-1]):
            if i != j and sector_map.get(t1) == sector_map.get(t2):
                corr[i, j] = SECTOR_BASE_CORR

    # Benchmark highly correlated with everything
    corr[-1, :-1] = 0.75
    corr[:-1, -1] = 0.75

    # Ensure PD
    min_eig = np.linalg.eigvalsh(corr).min()
    if min_eig < 1e-8:
        corr += np.eye(n_tickers) * (abs(min_eig) + 1e-6)

    # ── Build volatility vector and covariance matrix ──────────────────────────
    vols = np.array([
        STOCK_PARAMS.get(t, (0.15, 0.25, 1.0, 100.0))[1] / np.sqrt(252)  # daily vol
        for t in all_tickers
    ])

    cov = corr * np.outer(vols, vols)
    L   = np.linalg.cholesky(cov + np.eye(n_tickers) * 1e-10)

    # ── Generate return paths ─────────────────────────────────────────────────
    mus = np.array([
        STOCK_PARAMS.get(t, (0.15, 0.25, 1.0, 100.0))[0] / 252  # daily mean
        for t in all_tickers
    ])

    Z    = rng.standard_normal((n_tickers, n_days))
    rets = mus[:, np.newaxis] + L @ Z   # (n_tickers, n_days)

    # ── Build price series ────────────────────────────────────────────────────
    # Start prices such that the series ends near the known market price
    end_prices = np.array([
        STOCK_PARAMS.get(t, (0.15, 0.25, 1.0, 100.0))[3]
        for t in all_tickers
    ])

    # Generate cumulative log returns
    cum_rets = np.cumsum(rets, axis=1)   # (n_tickers, n_days)

    # Back out start price: start_price = end_price / exp(total_log_return)
    total_ret  = cum_rets[:, -1]
    start_prices = end_prices / np.exp(total_ret)

    # Build price matrix
    price_matrix = start_prices[:, np.newaxis] * np.exp(cum_rets)

    # Build date index (trading days ending today)
    end_date   = datetime.now().date()
    all_dates  = pd.bdate_range(end=end_date, periods=n_days)
    prices_df  = pd.DataFrame(price_matrix.T, index=all_dates, columns=all_tickers)

    # ── Update holdings with synthetic current prices & metadata ─────────────
    for h in holdings:
        h.current_price = float(prices_df[h.ticker].iloc[-1]) if h.ticker in prices_df else STOCK_PARAMS.get(h.ticker, (0,0,0,100))[3]
        h.market_value   = h.shares_held * h.current_price
        h.unrealized_pnl = (h.current_price - h.cost_basis) * h.shares_held
        h.unrealized_pct = (h.current_price - h.cost_basis) / h.cost_basis if h.cost_basis > 0 else 0.0
        # Populate metadata from STOCK_META if not already set in the Excel file
        meta = STOCK_META.get(h.ticker, ())
        if meta:
            if not h.company_name or h.company_name in ("", "nan"):
                h.company_name = meta[0]
            if not h.sector or h.sector in ("", "nan"):
                h.sector = meta[1]
            if not h.industry or h.industry in ("", "nan"):
                h.industry = meta[2]
        else:
            # Fallback: use ticker as name, "Unknown" for sector
            if not h.company_name or h.company_name in ("", "nan"):
                h.company_name = h.ticker
            if not h.sector or h.sector in ("", "nan"):
                h.sector = "Unknown"
            if not h.industry or h.industry in ("", "nan"):
                h.industry = "Unknown"

    total_value = sum(h.market_value for h in holdings)
    for h in holdings:
        h.weight = h.market_value / total_value if total_value > 0 else 0.0

    # ── Data quality report (synthetic) ────────────────────────────────────────
    quality = DataQualityReport(
        valid_tickers=[h.ticker for h in holdings],
        failed_tickers=[],
        date_range=(str(prices_df.index[0].date()), str(prices_df.index[-1].date())),
        total_rows=n_days,
        missing_data_points=0,
        missing_tickers=[],
        log_lines=[f"SYNTH {t}: {n_days} synthetic data points" for t in all_tickers],
    )

    # ── Risk-free rate ─────────────────────────────────────────────────────────
    rfr_override = settings.risk_free_rate_value
    risk_free_rate = rfr_override if rfr_override is not None else 0.045

    log.info(
        f"Synthetic data ready: ${total_value:,.0f} portfolio value, "
        f"{n_days} trading days"
    )

    return MarketData(
        prices=prices_df,
        holdings=holdings,
        benchmark_ticker=benchmark,
        risk_free_rate=risk_free_rate,
        quality=quality,
    )


def get_market_data(load_result: LoadResult) -> MarketData:
    """
    Smart market data fetcher: tries live data first, falls back to synthetic.

    In production (user's machine with internet access), this will use live
    yfinance data. In restricted environments (testing, CI), it generates
    synthetic data automatically.
    """
    from engine.market_data import fetch_market_data

    try:
        md = fetch_market_data(load_result)
        # Validate we got real data
        valid = len([h for h in md.holdings if h.current_price > 0])
        if valid < len(md.holdings) * 0.5:
            raise ValueError(f"Only {valid}/{len(md.holdings)} tickers resolved")
        log.info(f"Using live market data ({valid} tickers resolved)")
        return md
    except Exception as e:
        log.warning(
            f"Live market data fetch failed ({e}). "
            "Generating synthetic data for demonstration…"
        )
        return generate_synthetic_market_data(load_result)
