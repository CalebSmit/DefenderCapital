"""
Shared pytest fixtures for DCM Risk Model tests.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.data_loader import load_portfolio, Holding, LoadResult, PortfolioSettings
from engine.synthetic_data import generate_synthetic_market_data
from engine.risk_metrics import compute_all_metrics
from engine.monte_carlo import run_simulation
from engine.stress_testing import run_all_stress_tests


@pytest.fixture(scope="session")
def load_result():
    """Load the main portfolio."""
    return load_portfolio()


@pytest.fixture(scope="session")
def market_data(load_result):
    """Generate synthetic market data (deterministic, seed=42)."""
    return generate_synthetic_market_data(load_result, n_days=504, seed=42)


@pytest.fixture(scope="session")
def metrics(market_data):
    """Compute all risk metrics from synthetic market data."""
    return compute_all_metrics(market_data)


@pytest.fixture(scope="session")
def simulation(market_data):
    """Run a Monte Carlo simulation (5,000 paths for speed)."""
    return run_simulation(market_data, n_paths=5_000, n_days=252, seed=42)


@pytest.fixture(scope="session")
def stress_results(market_data):
    """Run all stress tests (without ES comparison)."""
    return run_all_stress_tests(market_data, custom_drawdown=-0.20)


@pytest.fixture(scope="session")
def stress_results_with_es(market_data, metrics):
    """Run all stress tests with ES comparison populated."""
    return run_all_stress_tests(market_data, custom_drawdown=-0.20, metrics=metrics)


@pytest.fixture(scope="session")
def single_stock_market_data():
    """Single-stock portfolio for edge-case testing."""
    h = Holding(
        ticker="AAPL", company_name="Apple Inc.", sector="Technology",
        industry="Consumer Electronics", shares_held=100, cost_basis=150.0,
        current_price=182.0, market_value=18_200.0, weight=1.0,
    )
    lr = LoadResult(holdings=[h], settings=PortfolioSettings())
    return generate_synthetic_market_data(lr, n_days=252, seed=99)


@pytest.fixture(scope="session")
def two_stock_market_data():
    """Two-stock portfolio for diversification testing."""
    holdings = [
        Holding(ticker="AAPL", company_name="Apple Inc.", sector="Technology",
                industry="Consumer Electronics", shares_held=100, cost_basis=150.0,
                current_price=182.0, market_value=18_200.0, weight=0.5),
        Holding(ticker="JPM", company_name="JPMorgan Chase", sector="Financial Services",
                industry="Diversified Banks", shares_held=50, cost_basis=140.0,
                current_price=175.0, market_value=8_750.0, weight=0.5),
    ]
    lr = LoadResult(holdings=holdings, settings=PortfolioSettings())
    return generate_synthetic_market_data(lr, n_days=252, seed=77)
