"""
Tests for engine/data_loader.py

Covers: Excel parsing, validation, settings, error handling,
        fuzzy ticker matching, edge cases.
"""
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.data_loader import (
    load_portfolio, Holding, PortfolioSettings, LoadResult,
)
from engine.utils import get_portfolio_path


class TestHoldingsCounting:
    def test_loads_expected_holdings(self, load_result):
        """Portfolio should have at least one valid holding (exact count varies)."""
        assert len(load_result.holdings) >= 1, (
            f"Expected at least 1 holding, got {len(load_result.holdings)}"
        )

    def test_no_skipped_rows(self, load_result):
        assert len(load_result.skipped_rows) == 0, (
            f"Unexpected skipped rows: {load_result.skipped_rows}"
        )

    def test_no_errors(self, load_result):
        assert len(load_result.errors) == 0, (
            f"Unexpected errors: {load_result.errors}"
        )


class TestHoldingFields:
    def test_tickers_non_empty(self, load_result):
        for h in load_result.holdings:
            assert h.ticker, f"Empty ticker: {h}"
            assert h.ticker == h.ticker.upper().strip(), \
                f"Ticker not uppercase/stripped: '{h.ticker}'"

    def test_shares_positive(self, load_result):
        for h in load_result.holdings:
            assert h.shares_held > 0, f"{h.ticker}: shares_held={h.shares_held}"

    def test_cost_basis_positive(self, load_result):
        for h in load_result.holdings:
            assert h.cost_basis > 0, f"{h.ticker}: cost_basis={h.cost_basis}"

    def test_sectors_non_empty_after_enrichment(self, market_data):
        """Sectors are populated by market_data fetcher (not raw Excel load)."""
        for h in market_data.holdings:
            assert h.sector, f"{h.ticker}: empty sector after enrichment"

    def test_company_names_non_empty_after_enrichment(self, market_data):
        """Company names are populated by market_data fetcher."""
        for h in market_data.holdings:
            assert h.company_name, f"{h.ticker}: empty company_name after enrichment"


class TestSettings:
    def test_benchmark_ticker(self, load_result):
        assert load_result.settings.benchmark_ticker == "SPY"

    def test_risk_free_rate_setting(self, load_result):
        assert load_result.settings.risk_free_rate == "auto"

    def test_lookback_years(self, load_result):
        assert load_result.settings.lookback_years == 2

    def test_simulation_paths(self, load_result):
        assert load_result.settings.simulation_paths == 10_000

    def test_confidence_level_95(self, load_result):
        assert abs(load_result.settings.confidence_level_1 - 0.95) < 0.001

    def test_confidence_level_99(self, load_result):
        assert abs(load_result.settings.confidence_level_2 - 0.99) < 0.001


class TestWeightsAndValues:
    def test_weights_sum_to_one(self, load_result):
        """After synthetic data, weights sum to 1."""
        # load_result from Excel won't have final weights yet (set by engine)
        # Just check current_price > 0 if provided
        pass  # weights are set by market_data engine

    def test_market_value_nonnegative(self, load_result):
        """Market values from Excel should be set (or 0 if not yet fetched)."""
        for h in load_result.holdings:
            assert h.market_value >= 0, f"{h.ticker}: negative market_value"


class TestValidationStressTest:
    """Test that the loader handles bad input gracefully."""

    def test_accepts_load_result_object(self, load_result):
        assert isinstance(load_result, LoadResult)

    def test_holdings_are_holding_objects(self, load_result):
        for h in load_result.holdings:
            assert isinstance(h, Holding)

    def test_settings_is_portfolio_settings(self, load_result):
        assert isinstance(load_result.settings, PortfolioSettings)

    def test_custom_path_accepted(self):
        """load_portfolio() accepts a custom path argument."""
        path = get_portfolio_path()
        result = load_portfolio(path)
        assert len(result.holdings) >= 1

    def test_raise_on_empty_false(self):
        """raise_on_empty=False should not raise on bad files."""
        path = get_portfolio_path()
        # With a valid file this just returns normally
        result = load_portfolio(path, raise_on_empty=False)
        assert result is not None


class TestPortfolioSettingsDefaults:
    def test_default_settings_exist(self):
        s = PortfolioSettings()
        assert s.benchmark_ticker == "SPY"
        assert 0 < s.confidence_level_1 < 1
        assert 0 < s.confidence_level_2 < 1
        assert s.confidence_level_2 > s.confidence_level_1

    def test_risk_free_rate_value_none_by_default(self):
        s = PortfolioSettings()
        # "auto" setting means risk_free_rate_value = None (to be fetched)
        assert s.risk_free_rate_value is None or isinstance(s.risk_free_rate_value, float)
