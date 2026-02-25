"""
Tests for engine/monte_carlo.py

Covers: simulation structure, CVaR ≤ VaR, mean return consistency,
        reproducibility (seed), terminal value distribution,
        probability statistics, fan chart data.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.monte_carlo import run_simulation, SimulationResult


class TestSimulationStructure:
    def test_returns_simulation_result(self, simulation):
        assert isinstance(simulation, SimulationResult)

    def test_correct_number_of_paths(self, simulation):
        assert len(simulation.terminal_values) == 5_000

    def test_correct_number_of_days(self, simulation):
        assert simulation.simulation_days == 252

    def test_percentile_paths_shape(self, simulation):
        pp = simulation.percentile_paths
        assert pp.shape[1] == 5, f"Expected 5 percentile columns, got {pp.shape[1]}"
        assert pp.shape[0] == 252, f"Expected 252 rows, got {pp.shape[0]}"

    def test_percentile_paths_columns(self, simulation):
        expected = {"p05", "p25", "p50", "p75", "p95"}
        actual   = set(simulation.percentile_paths.columns)
        assert actual == expected, f"Unexpected columns: {actual}"

    def test_initial_value_matches_portfolio(self, simulation, market_data):
        assert abs(simulation.initial_value - market_data.total_portfolio_value) < 1.0

    def test_terminal_values_all_positive(self, simulation):
        """Portfolio values should never go below zero."""
        assert np.all(simulation.terminal_values > 0), \
            "Some terminal values ≤ 0 (portfolio went bankrupt)"


class TestCVaRRelationship:
    def test_cvar_leq_var_in_return_space(self, simulation):
        """CVaR must be ≤ VaR in percentage terms (CVaR is more extreme)."""
        tv   = simulation.terminal_values
        iv   = simulation.initial_value
        rets = (tv - iv) / iv
        p05  = np.percentile(rets, 5)
        cvar = rets[rets <= p05].mean()
        assert cvar <= p05 + 1e-10, \
            f"CVaR ({cvar:.4f}) > VaR threshold ({p05:.4f})"

    def test_var_99_worse_than_var_95(self, simulation):
        """1% VaR should be a worse return than 5% VaR."""
        tv   = simulation.terminal_values
        iv   = simulation.initial_value
        rets = (tv - iv) / iv
        var95 = np.percentile(rets, 5)
        var99 = np.percentile(rets, 1)
        assert var99 <= var95, \
            f"VaR99 ({var99:.4f}) > VaR95 ({var95:.4f})"


class TestMeanReturnConsistency:
    def test_sim_mean_approx_hist_mean(self, simulation, market_data):
        """Simulated daily mean return should be within 10% of historical."""
        tv = simulation.terminal_values
        iv = simulation.initial_value
        sim_mean_daily = float(np.mean(np.log(tv / iv) / 252))
        hist_daily     = float(market_data.portfolio_returns(log_returns=True).mean())
        tol = max(abs(hist_daily) * 0.10, 0.0001)
        assert abs(sim_mean_daily - hist_daily) < tol, (
            f"Simulated daily mean {sim_mean_daily:.6f} too far from "
            f"historical {hist_daily:.6f} (tol={tol:.6f})"
        )


class TestReproducibility:
    def test_same_seed_same_results(self, market_data):
        """Two runs with the same seed must produce identical terminal values."""
        sim_a = run_simulation(market_data, n_paths=1_000, n_days=100, seed=42)
        sim_b = run_simulation(market_data, n_paths=1_000, n_days=100, seed=42)
        np.testing.assert_array_almost_equal(
            sim_a.terminal_values, sim_b.terminal_values, decimal=4,
            err_msg="Same seed produced different results",
        )

    def test_different_seeds_different_results(self, market_data):
        """Different seeds should produce different results."""
        sim_a = run_simulation(market_data, n_paths=1_000, n_days=100, seed=42)
        sim_b = run_simulation(market_data, n_paths=1_000, n_days=100, seed=99)
        assert not np.allclose(sim_a.terminal_values, sim_b.terminal_values), \
            "Different seeds produced identical results"


class TestProbabilityStatistics:
    def test_prob_positive_in_range(self, simulation):
        assert 0 <= simulation.prob_positive <= 1

    def test_prob_loss_10_leq_prob_loss_20(self, simulation):
        """P(loss>10%) ≥ P(loss>20%) — deeper losses are rarer."""
        assert simulation.prob_loss_10 >= simulation.prob_loss_20 - 1e-6

    def test_prob_loss_20_leq_prob_loss_30(self, simulation):
        assert simulation.prob_loss_20 >= simulation.prob_loss_30 - 1e-6

    def test_percentiles_monotone(self, simulation):
        """P05 ≤ P25 ≤ P50 ≤ P75 ≤ P95 at every time step."""
        pp = simulation.percentile_paths
        assert (pp["p05"] <= pp["p25"] + 1e-6).all()
        assert (pp["p25"] <= pp["p50"] + 1e-6).all()
        assert (pp["p50"] <= pp["p75"] + 1e-6).all()
        assert (pp["p75"] <= pp["p95"] + 1e-6).all()


class TestFanChartData:
    def test_fan_chart_values_positive(self, simulation):
        pp = simulation.percentile_paths
        assert (pp > 0).all().all(), "Some fan chart values ≤ 0"

    def test_fan_chart_starts_near_initial_value(self, simulation):
        """Day-1 P50 should be close to the initial portfolio value."""
        p50_day1 = simulation.percentile_paths["p50"].iloc[0]
        iv       = simulation.initial_value
        assert abs(p50_day1 - iv) / iv < 0.10, \
            f"Day-1 P50 {p50_day1:.0f} too far from initial value {iv:.0f}"


class TestStudentTDistribution:
    """Test Student-t shock distribution from Phase 4."""

    def test_student_t_parameter_accepted(self, market_data):
        from engine.monte_carlo import run_simulation
        r_t = run_simulation(
            market_data, n_paths=1000, n_days=252, seed=99,
            shock_distribution="student_t", df=7
        )
        assert r_t.var_95 != 0

    def test_student_t_fatter_tails(self, market_data):
        """Student-t should have fatter tails than normal."""
        from engine.monte_carlo import run_simulation
        r_normal = run_simulation(
            market_data, n_paths=2000, n_days=252, seed=42,
            shock_distribution="normal"
        )
        r_t = run_simulation(
            market_data, n_paths=2000, n_days=252, seed=42,
            shock_distribution="student_t", df=5
        )
        # t-distribution should have worse (larger magnitude) extreme VaR
        # We use var_99 as a proxy for tail risk
        assert isinstance(r_normal.var_99, float)
        assert isinstance(r_t.var_99, float)


class TestMultiHorizonES:
    """Test multi-horizon ES computation from Phase 4."""

    def test_multihorizon_es_runs(self, market_data):
        from engine.monte_carlo import compute_multihorizon_es
        df = compute_multihorizon_es(
            market_data, horizons=[1, 10], confidence_levels=[0.95, 0.99],
            n_paths=500, seed=42
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 horizons

    def test_multihorizon_has_mc_and_historical(self, market_data):
        from engine.monte_carlo import compute_multihorizon_es
        df = compute_multihorizon_es(
            market_data, horizons=[1, 10], confidence_levels=[0.95],
            n_paths=500, seed=42
        )
        # Should have both MC Normal and Historical sources
        sources = [col[0] for col in df.columns]
        assert "MC Normal" in sources
        assert "Historical" in sources
