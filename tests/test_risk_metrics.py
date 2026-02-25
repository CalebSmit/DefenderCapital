"""
Tests for engine/risk_metrics.py

Covers: VaR formulas, CVaR relationships, Euler decomposition,
        Sharpe/Sortino/Calmar ratios, Beta/Alpha, drawdown,
        concentration metrics, single-stock edge case.
"""
import pytest
import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.risk_metrics import (
    compute_all_metrics, parametric_var, historical_var, component_var,
    StockRiskMetrics, PortfolioRiskMetrics, VaRResult,
)
from engine.synthetic_data import generate_synthetic_market_data
from engine.data_loader import Holding, LoadResult, PortfolioSettings


class TestVaRFormulas:
    def test_parametric_var_matches_manual(self, metrics, market_data):
        """
        Parametric VaR should be close to the hand-rolled formula.
        The implementation uses Ledoit-Wolf shrinkage for the covariance matrix
        rather than the raw sample std, so a small difference (<2%) is expected.
        """
        port_ret = market_data.portfolio_returns(log_returns=True)
        mu  = float(port_ret.mean())
        sig = float(port_ret.std())
        pv  = market_data.total_portfolio_value
        z   = norm.ppf(0.05)
        expected = -(mu + z * sig) * pv
        actual = metrics.var_95.parametric_var
        pct_diff = abs(expected - actual) / max(abs(expected), 1.0) * 100
        assert pct_diff < 2.0, (
            f"VaR mismatch exceeds 2%: expected={expected:.2f}, actual={actual:.2f}, "
            f"diff={pct_diff:.2f}% (Ledoit-Wolf shrinkage causes minor divergence)"
        )

    def test_historical_var_matches_percentile(self, metrics, market_data):
        """Historical VaR must match the 5th percentile of simple returns × PV."""
        log_ret = market_data.portfolio_returns(log_returns=True)
        simple_ret = log_ret.apply(lambda x: np.exp(x) - 1)
        p5 = float(np.percentile(simple_ret.dropna(), 5))
        pv = market_data.total_portfolio_value
        expected = p5 * pv
        actual = metrics.var_95.historical_var
        assert abs(expected - actual) < 1.0, (
            f"Hist VaR mismatch: expected={expected:.2f}, actual={actual:.2f}"
        )

    def test_cvar_geq_var_absolute(self, metrics):
        """CVaR must have a larger absolute value than VaR (at any confidence level)."""
        # 95% level
        assert abs(metrics.var_95.parametric_cvar) >= abs(metrics.var_95.parametric_var) - 1e-6, \
            f"CVaR95 not worse than VaR95: {metrics.var_95.parametric_cvar:.2f} vs {metrics.var_95.parametric_var:.2f}"
        # 99% level
        assert abs(metrics.var_99.parametric_cvar) >= abs(metrics.var_99.parametric_var) - 1e-6, \
            "CVaR99 not worse than VaR99"

    def test_var_99_geq_var_95(self, metrics):
        """99% VaR should be at least as large as 95% VaR (higher confidence = larger loss)."""
        var95 = abs(metrics.var_95.parametric_var)
        var99 = abs(metrics.var_99.parametric_var)
        assert var99 >= var95 - 1e-6, \
            f"99% VaR ({var99:.2f}) < 95% VaR ({var95:.2f})"

    def test_var_methods_in_ballpark(self, metrics):
        """Parametric and historical VaR should not differ by more than 3×."""
        p_var = abs(metrics.var_95.parametric_var)
        h_var = abs(metrics.var_95.historical_var)
        ratio = max(p_var, h_var) / max(min(p_var, h_var), 0.01)
        assert ratio < 3.0, (
            f"VaR methods diverge too much: parametric={p_var:.0f}, "
            f"historical={h_var:.0f}, ratio={ratio:.1f}x"
        )


class TestEulerDecomposition:
    def test_component_var_sum_equals_total(self, metrics):
        """Sum of component VaRs must equal total portfolio VaR (Euler identity)."""
        total_comp = sum(sm.component_var_95 for sm in metrics.stock_metrics)
        total_var  = metrics.var_95.parametric_var
        pct_diff   = abs(total_comp - total_var) / max(abs(total_var), 1.0) * 100
        assert pct_diff < 1.0, (
            f"Euler decomp failed: sum={total_comp:.4f}, total={total_var:.4f}, "
            f"diff={pct_diff:.4f}%"
        )

    def test_component_vars_correct_count(self, metrics, load_result):
        """One component VaR per holding."""
        assert len(metrics.stock_metrics) == len(load_result.holdings)


class TestReturnRatios:
    def test_sharpe_ratio_reasonable(self, metrics):
        """Sharpe ratio should be within a plausible range for equities."""
        assert -5.0 < metrics.sharpe < 10.0, \
            f"Sharpe={metrics.sharpe:.3f} outside plausible range"

    def test_sortino_geq_sharpe_when_positive(self, metrics):
        """When returns are positive, Sortino >= Sharpe (downside vol < total vol)."""
        if metrics.annualized_return > 0:
            assert metrics.sortino >= metrics.sharpe - 0.1, \
                f"Sortino ({metrics.sortino:.3f}) < Sharpe ({metrics.sharpe:.3f})"

    def test_calmar_positive_when_positive_return(self, metrics):
        """Calmar ratio should be positive when annualized return is positive."""
        if metrics.annualized_return > 0 and metrics.max_drawdown < 0:
            assert metrics.calmar > 0, f"Calmar={metrics.calmar:.3f} should be positive"

    def test_annualized_vol_positive(self, metrics):
        assert metrics.annualized_vol > 0, "Annualized vol must be positive"

    def test_annualized_vol_plausible(self, metrics):
        """Portfolio vol should be in a plausible range (0–100% annualized)."""
        assert 0.01 < metrics.annualized_vol < 1.0, \
            f"Annualized vol {metrics.annualized_vol:.1%} seems implausible"


class TestBetaAlpha:
    def test_beta_positive_for_equity_portfolio(self, metrics):
        """US equity portfolio should have positive beta vs SPY."""
        assert metrics.beta > 0, f"Beta={metrics.beta:.3f} should be positive"

    def test_beta_plausible_range(self, metrics):
        """Diversified US equity portfolio beta should be roughly 0.5–2.0."""
        assert 0.3 < metrics.beta < 3.0, \
            f"Beta={metrics.beta:.3f} outside expected range"

    def test_alpha_is_float(self, metrics):
        assert isinstance(metrics.alpha, float)


class TestDrawdown:
    def test_max_drawdown_negative(self, metrics):
        """Max drawdown must be negative (or zero)."""
        assert metrics.max_drawdown <= 0, \
            f"Max drawdown {metrics.max_drawdown:.1%} should be ≤ 0"

    def test_max_drawdown_geq_minus_100pct(self, metrics):
        """Portfolio can't lose more than 100%."""
        assert metrics.max_drawdown >= -1.0, \
            f"Max drawdown {metrics.max_drawdown:.1%} < -100%"

    def test_max_dd_duration_nonneg(self, metrics):
        assert metrics.max_dd_duration >= 0


class TestConcentration:
    def test_hhi_bounds(self, metrics, load_result):
        """HHI must be between 1/n (equal) and 1 (fully concentrated)."""
        n = len(load_result.holdings)
        assert metrics.hhi >= 1.0 / n - 1e-9, \
            f"HHI={metrics.hhi:.4f} below 1/n={1/n:.4f}"
        assert metrics.hhi <= 1.0 + 1e-9, f"HHI={metrics.hhi:.4f} > 1"

    def test_eff_num_bets_reasonable(self, metrics, load_result):
        n = len(load_result.holdings)
        assert 1.0 <= metrics.eff_num_bets <= n, \
            f"ENB={metrics.eff_num_bets:.1f} outside [1, {n}]"

    def test_diversification_ratio_geq_1(self, metrics):
        """Diversification ratio must be ≥ 1 (portfolio vol ≤ weighted avg vol)."""
        assert metrics.diversification_ratio >= 1.0 - 1e-6, \
            f"Diversification ratio {metrics.diversification_ratio:.3f} < 1"


class TestSingleStock:
    def test_single_stock_metrics_compute(self, single_stock_market_data):
        """Risk metrics should work on a single-stock portfolio."""
        m = compute_all_metrics(single_stock_market_data)
        assert m.var_95.parametric_var != 0
        assert m.annualized_vol > 0
        assert len(m.stock_metrics) == 1

    def test_single_stock_hhi_equals_one(self, single_stock_market_data):
        """A single-stock portfolio is fully concentrated: HHI = 1."""
        m = compute_all_metrics(single_stock_market_data)
        assert abs(m.hhi - 1.0) < 1e-6, f"Single stock HHI={m.hhi:.6f} ≠ 1"

    def test_single_stock_eff_num_bets_equals_one(self, single_stock_market_data):
        m = compute_all_metrics(single_stock_market_data)
        assert abs(m.eff_num_bets - 1.0) < 1e-6, \
            f"Single stock ENB={m.eff_num_bets:.6f} ≠ 1"


class TestStockMetrics:
    def test_all_stocks_have_metrics(self, metrics, load_result):
        tickers_with_metrics = {sm.ticker for sm in metrics.stock_metrics}
        expected_tickers     = {h.ticker for h in load_result.holdings}
        assert tickers_with_metrics == expected_tickers

    def test_stock_vol_positive(self, metrics):
        for sm in metrics.stock_metrics:
            assert sm.annualized_vol > 0, f"{sm.ticker}: vol={sm.annualized_vol}"

    def test_stock_beta_finite(self, metrics):
        for sm in metrics.stock_metrics:
            assert np.isfinite(sm.beta), f"{sm.ticker}: beta={sm.beta}"


class TestEWMACovariance:
    """Test EWMA covariance estimator from Phase 3."""

    def test_ewma_returns_ndarray(self, market_data):
        from engine.risk_metrics import ewma_covariance
        tickers = market_data.portfolio_tickers
        prices  = market_data.prices[tickers].dropna(how='all')
        log_rets = np.log(prices / prices.shift(1)).dropna()
        cov = ewma_covariance(log_rets, lam=0.94)
        assert isinstance(cov, np.ndarray)
        assert cov.shape == (len(tickers), len(tickers))

    def test_ewma_positive_definite(self, market_data):
        from engine.risk_metrics import ewma_covariance
        tickers = market_data.portfolio_tickers
        prices  = market_data.prices[tickers].dropna(how='all')
        log_rets = np.log(prices / prices.shift(1)).dropna()
        cov = ewma_covariance(log_rets, lam=0.94)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > -1e-6), f"EWMA cov not PD: min_eig={eigvals.min():.6f}"

    def test_ewma_metrics_compute(self, market_data):
        from engine.data_loader import PortfolioSettings
        s = PortfolioSettings()
        s.covariance_mode = "ewma"
        m = compute_all_metrics(market_data, settings=s)
        assert m.annualized_vol > 0
        assert m.var_95.parametric_var != 0

    def test_ledoit_wolf_metrics_unchanged(self, market_data, metrics):
        """Ledoit-Wolf (default) must produce same result as before."""
        from engine.data_loader import PortfolioSettings
        s = PortfolioSettings()
        s.covariance_mode = "ledoit_wolf"
        m_new = compute_all_metrics(market_data, settings=s)
        # Should match the session-cached metrics fixture
        assert abs(m_new.annualized_vol - metrics.annualized_vol) < 1e-6
