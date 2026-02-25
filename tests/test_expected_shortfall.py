"""
Tests for Expected Shortfall (ES) — Phase 1 enhancements.
"""
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.risk_metrics import compute_all_metrics

class TestExpectedShortfallPrimary:
    """Test ES at 97.5% confidence (primary FRTB standard)."""

    def test_es_975_computed(self, metrics):
        """var_es at 97.5% should be computed."""
        assert metrics.var_es is not None
        assert metrics.var_es.confidence == 0.975

    def test_es_975_geq_es_95(self, metrics):
        """ES(97.5%) >= ES(95%) in absolute value."""
        es_975 = abs(metrics.var_es.parametric_cvar)
        es_95  = abs(metrics.var_95.parametric_cvar)
        assert es_975 >= es_95 - 1e-6, f"ES(97.5%)={es_975:.2f} < ES(95%)={es_95:.2f}"

    def test_es_properties_alias_cvar(self, metrics):
        """ES property aliases must equal their CVaR equivalents."""
        v = metrics.var_95
        assert v.parametric_es == v.parametric_cvar
        assert v.historical_es == v.historical_cvar
        assert v.mc_es == v.mc_cvar

    def test_es_geq_var_at_same_level(self, metrics):
        """ES must be at least as large as VaR at the same confidence."""
        v = metrics.var_es
        assert abs(v.parametric_es) >= abs(v.parametric_var) - 1e-6

    def test_var_es_ordering(self, metrics):
        """ES(99%) >= ES(97.5%) >= ES(95%) — monotone in confidence."""
        es_95  = abs(metrics.var_95.parametric_es)
        es_975 = abs(metrics.var_es.parametric_es)
        es_99  = abs(metrics.var_99.parametric_es)
        assert es_99 >= es_975 - 1e-6
        assert es_975 >= es_95 - 1e-6


class TestESProperties:
    """Test that ES property aliases work correctly."""

    def test_parametric_es_alias(self, metrics):
        """parametric_es should equal parametric_cvar."""
        for v in [metrics.var_95, metrics.var_99, metrics.var_es]:
            assert v.parametric_es == v.parametric_cvar

    def test_historical_es_alias(self, metrics):
        """historical_es should equal historical_cvar."""
        for v in [metrics.var_95, metrics.var_99, metrics.var_es]:
            assert v.historical_es == v.historical_cvar

    def test_mc_es_alias(self, metrics):
        """mc_es should equal mc_cvar."""
        for v in [metrics.var_95, metrics.var_99, metrics.var_es]:
            assert v.mc_es == v.mc_cvar


class TestESWithSettings:
    """Test that ES respects PortfolioSettings."""

    def test_custom_es_confidence(self, market_data):
        """Custom ES confidence level should be used."""
        from engine.data_loader import PortfolioSettings
        
        s = PortfolioSettings()
        s.es_confidence_level = 0.99
        m = compute_all_metrics(market_data, settings=s)
        assert m.var_es.confidence == 0.99

    def test_default_es_confidence(self, market_data):
        """Default ES confidence should be 0.975."""
        m = compute_all_metrics(market_data)
        assert m.var_es.confidence == 0.975
