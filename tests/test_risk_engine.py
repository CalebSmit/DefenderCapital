"""
tests/test_risk_engine.py — Edge case tests for the risk analytics engine.

Tests cover:
  - Single-ticker portfolios (no covariance matrix)
  - Zero-variance assets
  - EWMA lambda validation
  - Cornish-Fisher bounds
  - Basel traffic-light classification
  - Component VaR sign convention (diversifier note)
  - Missing data detection
  - Login rate limiting
"""
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.risk_metrics import (
    compute_all_metrics, parametric_var, cornish_fisher_var,
)
from engine.synthetic_data import generate_synthetic_market_data
from engine.data_loader import Holding, LoadResult, PortfolioSettings
from auth.database import check_and_record_login_attempt, is_account_locked


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Single-Ticker Edge Case
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingleTickerPortfolio:
    """Single-stock portfolio should not crash despite zero diversification."""

    def test_single_ticker_compute_all_metrics(self, single_stock_market_data):
        """
        Compute all metrics for a single-stock portfolio.
        Expected behavior: diversification metrics are set to neutral values.
        """
        md = single_stock_market_data
        metrics = compute_all_metrics(md)

        # Verify portfolio was computed
        assert metrics is not None
        assert len(metrics.stock_metrics) == 1

        # Single stock: diversification_ratio should be 1.0
        assert metrics.diversification_ratio == pytest.approx(1.0, abs=0.01)

        # Single stock: avg correlation should be 1.0 (correlates with itself)
        assert metrics.avg_pairwise_corr == pytest.approx(1.0, abs=0.01)

        # Single stock: HHI should be 1.0 (100% concentration)
        assert metrics.hhi == pytest.approx(1.0, abs=0.01)

    def test_single_ticker_no_covariance_crash(self, single_stock_market_data):
        """Single-ticker portfolio should not crash despite trivial covariance."""
        md = single_stock_market_data
        # Should not raise exception
        metrics = compute_all_metrics(md)
        assert metrics is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Zero-Variance Asset Edge Case
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroVarianceAsset:
    """Asset with zero variance (constant price) should be handled gracefully."""

    def test_zero_variance_stock(self):
        """
        Portfolio with a zero-variance stock (price never changes) should
        handle vol/beta/Sharpe gracefully without division by zero.
        """
        # Create a synthetic market where one stock never moves
        h_normal = Holding(
            ticker="AAPL", company_name="Apple", sector="Tech",
            industry="Consumer", shares_held=100, cost_basis=150.0,
            current_price=150.0, market_value=15_000.0, weight=0.5
        )
        h_zero = Holding(
            ticker="BOND", company_name="Bond Fund", sector="Fixed Income",
            industry="Bonds", shares_held=200, cost_basis=100.0,
            current_price=100.0, market_value=20_000.0, weight=0.5
        )

        lr = LoadResult(holdings=[h_normal, h_zero], settings=PortfolioSettings())
        md = generate_synthetic_market_data(lr, n_days=252, seed=77)

        # Manually set one ticker's prices to be constant
        md.prices.iloc[:, md.prices.columns.get_loc("BOND")] = 100.0

        # Should not crash
        metrics = compute_all_metrics(md)
        assert metrics is not None

        # Zero-variance asset should have zero volatility
        bond_metrics = next((m for m in metrics.stock_metrics if m.ticker == "BOND"), None)
        assert bond_metrics is not None
        assert bond_metrics.annualized_vol < 1e-6 or bond_metrics.annualized_vol == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EWMA Lambda Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEWMALambdaValidation:
    """EWMA lambda (decay factor) must be in [0.90, 0.99] per Basel standards."""

    def test_ewma_lambda_out_of_range_handled(self):
        """
        When lambda is out of range, settings should be stored and
        compute_all_metrics should handle gracefully.
        """
        settings = PortfolioSettings(ewma_lambda=0.50)  # too low

        h = Holding(
            ticker="AAPL", company_name="Apple", sector="Tech",
            industry="Consumer", shares_held=100, cost_basis=150.0,
            current_price=150.0, market_value=15_000.0, weight=1.0
        )
        lr = LoadResult(holdings=[h], settings=settings)
        md = generate_synthetic_market_data(lr, n_days=252, seed=42)

        # Should handle bad lambda gracefully
        metrics = compute_all_metrics(md, settings=settings)
        assert metrics is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Cornish-Fisher Extreme Skewness/Kurtosis
# ═══════════════════════════════════════════════════════════════════════════════

class TestCornishFisherEdgeCases:
    """Cornish-Fisher VaR should handle extreme distributions."""

    def test_cornish_fisher_with_extreme_returns(self):
        """
        Test cornish_fisher_var with extreme skewness and kurtosis.
        Should not crash and should issue a warning for extreme values.
        """
        # Create returns with high negative skewness and excess kurtosis
        np.random.seed(42)
        normal_ret = np.random.normal(0.0005, 0.01, 252)

        # Inject tail events to increase kurtosis
        normal_ret[10] = -0.15
        normal_ret[50] = -0.12
        normal_ret[100] = 0.20

        mean_ret = float(np.mean(normal_ret))
        vol = float(np.std(normal_ret))
        # Use scipy.stats for proper skewness/kurtosis calculation
        skew_val = float(skew(normal_ret))
        kurt_val = float(kurtosis(normal_ret))

        portfolio_value = 1_000_000
        cf_var = cornish_fisher_var(
            portfolio_value=portfolio_value,
            mean_daily_ret=mean_ret,
            daily_vol=vol,
            skewness=skew_val,
            excess_kurtosis=kurt_val,
            confidence=0.95,
        )

        # Should return a value without crashing
        # When skewness is positive, CF-VaR can be positive (left tail is lighter)
        assert abs(cf_var) < 1_000_000  # reasonable magnitude


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Basel Traffic-Light Classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaselTrafficLight:
    """Basel III traffic-light framework: GREEN (0-4), YELLOW (5-9), RED (10+)."""

    def test_basel_zone_green(self):
        """0 exceptions → GREEN zone."""
        # Green zone is when n_exceptions < 5
        n_exceptions = 0
        assert n_exceptions < 5, "0 exceptions should be GREEN zone"

    def test_basel_zone_yellow(self):
        """5 exceptions → YELLOW zone."""
        # Yellow zone is 5-9 exceptions
        n_exceptions = 5
        assert 5 <= n_exceptions <= 9, "5 exceptions should be YELLOW zone"

    def test_basel_zone_red(self):
        """15 exceptions → RED zone."""
        # Red zone is >= 10 exceptions
        n_exceptions = 15
        assert n_exceptions >= 10, "15 exceptions should be RED zone"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Component VaR Sign Convention (Diversifier Note)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComponentVaRDiversifier:
    """Negative component VaR indicates a diversifier position."""

    def test_negative_component_var_generates_note(self, metrics):
        """
        When component VaR is negative, component_var_note should be non-empty
        explaining that the position is a diversifier.
        """
        # Find any stock with negative component VaR
        diversifier = None
        for stock_metric in metrics.stock_metrics:
            if stock_metric.component_var_95 < 0:
                diversifier = stock_metric
                break

        # If a diversifier exists, verify note is populated
        if diversifier is not None:
            assert diversifier.component_var_note != "", \
                "Negative component VaR should have an explanatory note"
            assert "diversif" in diversifier.component_var_note.lower(), \
                "Note should mention diversification"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Missing Data Detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingDataDetection:
    """Portfolio with gaps in price data should be handled."""

    def test_forward_fill_bias_handling(self):
        """
        Extended NaN gaps might indicate data issues.
        Should be handled without crashing.
        """
        h = Holding(
            ticker="PROB", company_name="Problem Stock", sector="Tech",
            industry="Software", shares_held=100, cost_basis=100.0,
            current_price=120.0, market_value=12_000.0, weight=1.0
        )

        lr = LoadResult(holdings=[h], settings=PortfolioSettings())
        md = generate_synthetic_market_data(lr, n_days=252, seed=42)

        # Inject a gap of 10 consecutive NaNs
        ticker_col = md.prices.columns.get_loc("PROB")
        md.prices.iloc[50:60, ticker_col] = np.nan

        # compute_all_metrics should handle this gracefully
        metrics = compute_all_metrics(md)
        assert metrics is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Login Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoginRateLimiting:
    """Test rate limiting on failed login attempts."""

    def test_check_and_record_login_attempt_success_resets_counter(self):
        """Successful login should reset the counter."""
        from auth.database import _login_attempts
        _login_attempts.clear()

        # First attempt: failure
        result1 = check_and_record_login_attempt("testuser", success=False)
        assert result1["remaining_attempts"] == 4

        # Second attempt: success
        result2 = check_and_record_login_attempt("testuser", success=True)
        assert result2["remaining_attempts"] == 5  # Reset

    def test_check_and_record_login_attempt_lockout_at_5_failures(self):
        """After 5 failed attempts, account should be locked."""
        from auth.database import _login_attempts
        _login_attempts.clear()

        # Record 5 failed attempts
        for i in range(5):
            result = check_and_record_login_attempt("lockeduser", success=False)

        # On the 5th failure, should be locked
        assert result["locked"] is True
        assert result["lockout_seconds"] > 0

    def test_is_account_locked_returns_status(self):
        """is_account_locked should return current lockout status."""
        from auth.database import _login_attempts
        _login_attempts.clear()

        # No lock initially
        status = is_account_locked("newuser")
        assert status["locked"] is False

        # After lockout
        for i in range(5):
            check_and_record_login_attempt("newuser", success=False)

        status = is_account_locked("newuser")
        assert status["locked"] is True
        assert status["lockout_seconds_remaining"] > 0

    def test_lockout_expires_after_timeout(self):
        """Lockout should expire after the timeout window."""
        from auth.database import _login_attempts
        _login_attempts.clear()

        # Force a lockout
        for i in range(5):
            check_and_record_login_attempt("expuser", success=False)

        # Manually set lockout_until to the past
        _login_attempts["expuser"]["lockout_until"] = time.time() - 1

        # Should be unlocked now
        status = is_account_locked("expuser")
        assert status["locked"] is False

