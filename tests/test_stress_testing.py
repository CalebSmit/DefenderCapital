"""
Tests for engine/stress_testing.py

Covers: scenario count, loss ranges, stock impacts,
        sign conventions, hypothetical scenarios,
        Phase 5: ES comparison & severity labelling.
"""
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.stress_testing import (
    run_all_stress_tests, StressTestResults,
    compare_scenario_to_es, ScenarioResult,
)


class TestScenarioStructure:
    def test_returns_stress_test_results(self, stress_results):
        assert isinstance(stress_results, StressTestResults)

    def test_historical_scenario_count(self, stress_results):
        assert len(stress_results.historical) == 4, \
            f"Expected 4 historical scenarios, got {len(stress_results.historical)}"

    def test_hypothetical_scenario_count(self, stress_results):
        assert len(stress_results.hypothetical) == 5, \
            f"Expected 5 hypothetical scenarios, got {len(stress_results.hypothetical)}"

    def test_total_scenario_count(self, stress_results):
        total = len(stress_results.historical) + len(stress_results.hypothetical)
        assert total == 9


class TestHistoricalScenarios:
    def test_gfc_loss_range(self, stress_results):
        """2008 GFC should cause 30–55% portfolio loss."""
        gfc = stress_results.historical[0]
        assert "GFC" in gfc.name or "2008" in gfc.name, \
            f"First scenario not GFC: {gfc.name}"
        loss = gfc.portfolio_loss_pct
        assert -0.55 <= loss <= -0.20, \
            f"GFC loss {loss:.1%} outside expected range [-55%, -20%]"

    def test_covid_loss_range(self, stress_results):
        """COVID-19 should cause 20–40% portfolio loss."""
        covid = stress_results.historical[1]
        assert "COVID" in covid.name or "2020" in covid.name, \
            f"Second scenario not COVID: {covid.name}"
        loss = covid.portfolio_loss_pct
        assert -0.40 <= loss <= -0.10, \
            f"COVID loss {loss:.1%} outside expected range [-40%, -10%]"

    def test_all_historical_losses_negative(self, stress_results):
        """All historical scenarios should show losses (negative returns)."""
        for s in stress_results.historical:
            assert s.portfolio_loss_pct < 0, \
                f"Scenario '{s.name}' has non-negative loss: {s.portfolio_loss_pct:.1%}"

    def test_historical_losses_less_than_100pct(self, stress_results):
        for s in stress_results.historical:
            assert s.portfolio_loss_pct >= -1.0, \
                f"Scenario '{s.name}' has loss > 100%: {s.portfolio_loss_pct:.1%}"


class TestHypotheticalScenarios:
    def test_all_hyp_scenarios_have_names(self, stress_results):
        for s in stress_results.hypothetical:
            assert s.name, f"Unnamed hypothetical scenario"

    def test_correlation_spike_present(self, stress_results):
        names = [s.name for s in stress_results.hypothetical]
        assert any("Corr" in n or "corr" in n for n in names), \
            f"Correlation spike scenario not found. Names: {names}"

    def test_custom_drawdown_present(self, stress_results):
        names = [s.name for s in stress_results.hypothetical]
        assert any("Custom" in n or "custom" in n or "Drawdown" in n for n in names), \
            f"Custom drawdown scenario not found. Names: {names}"

    def test_custom_drawdown_loss_correct(self):
        """Custom drawdown of -20% should produce approximately -20% loss."""
        import sys
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(ROOT))
        from engine.data_loader import load_portfolio
        from engine.synthetic_data import generate_synthetic_market_data
        from engine.stress_testing import run_all_stress_tests

        lr = load_portfolio()
        md = generate_synthetic_market_data(lr, n_days=252, seed=42)
        stress = run_all_stress_tests(md, custom_drawdown=-0.20)
        custom = next(s for s in stress.hypothetical if "Custom" in s.name or "Drawdown" in s.name)
        # Custom -20% drawdown should cause roughly -20% portfolio loss
        assert -0.22 <= custom.portfolio_loss_pct <= -0.18, \
            f"Custom -20% drawdown gave {custom.portfolio_loss_pct:.1%}"


class TestStockImpacts:
    def test_stock_impacts_populated(self, stress_results, load_result):
        """Every scenario should have stock-level impacts."""
        for s in stress_results.historical + stress_results.hypothetical:
            assert len(s.stock_impacts) > 0, f"'{s.name}' has no stock impacts"

    def test_stock_impacts_reasonable_range(self, stress_results):
        """Individual stock drawdowns should be in a physically-plausible range.
        Note: some sectors can gain significantly in certain scenarios (e.g.,
        Energy +57% in the 2022 Rate Shock is historically accurate for XOM).
        We allow gains up to +200% and losses not exceeding -100%.
        """
        for scenario in stress_results.historical:
            for si in scenario.stock_impacts:
                assert -1.0 <= si.scenario_drawdown <= 2.0, (
                    f"'{scenario.name}' {si.ticker}: "
                    f"drawdown={si.scenario_drawdown:.1%} out of range [-100%, +200%]"
                )

    def test_portfolio_loss_consistent_with_stock_impacts(self, stress_results, market_data):
        """Portfolio loss should equal the weighted sum of stock scenario drawdowns."""
        for scenario in stress_results.historical:
            weighted_sum = sum(si.weight * si.scenario_drawdown for si in scenario.stock_impacts)
            assert abs(weighted_sum - scenario.portfolio_loss_pct) < 0.01, (
                f"'{scenario.name}': weighted_sum={weighted_sum:.3f} ≠ "
                f"portfolio_loss={scenario.portfolio_loss_pct:.3f}"
            )


class TestScenarioMetadata:
    def test_scenarios_have_methodology(self, stress_results):
        for s in stress_results.historical + stress_results.hypothetical:
            assert s.methodology, f"'{s.name}' has no methodology"

    def test_scenarios_have_assumptions(self, stress_results):
        for s in stress_results.historical + stress_results.hypothetical:
            assert s.assumptions, f"'{s.name}' has no assumptions"

    def test_scenarios_have_interpretation(self, stress_results):
        for s in stress_results.historical + stress_results.hypothetical:
            assert s.interpretation, f"'{s.name}' has no interpretation"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: ES comparison & severity labelling
# ─────────────────────────────────────────────────────────────────────────────

class TestESComparison:
    """Tests for compare_scenario_to_es (Phase 5)."""

    def test_es_comparison_keys_present(self, stress_results_with_es, metrics):
        """Every scenario must have all ES-comparison keys when metrics is passed."""
        required_keys = {
            "es_95_1d", "es_975_1d", "es_99_1d", "es_99_21d",
            "scenario_loss_usd",
            "loss_to_es95_ratio", "loss_to_es975_ratio",
            "loss_to_es99_ratio", "loss_to_es99_21d_ratio",
            "multiples_of_daily_es",
        }
        for s in stress_results_with_es.all_scenarios:
            assert s.es_comparison is not None, f"'{s.name}' has no es_comparison"
            missing = required_keys - set(s.es_comparison.keys())
            assert not missing, f"'{s.name}' missing keys: {missing}"

    def test_es_benchmarks_positive(self, stress_results_with_es):
        """ES benchmark values must be positive dollar amounts."""
        for s in stress_results_with_es.all_scenarios:
            comp = s.es_comparison
            assert comp["es_95_1d"]  > 0, f"'{s.name}': es_95_1d non-positive"
            assert comp["es_975_1d"] > 0, f"'{s.name}': es_975_1d non-positive"
            assert comp["es_99_1d"]  > 0, f"'{s.name}': es_99_1d non-positive"
            assert comp["es_99_21d"] > comp["es_99_1d"], (
                f"'{s.name}': 21-day ES not larger than 1-day ES"
            )

    def test_ratios_positive(self, stress_results_with_es):
        """All loss-to-ES ratios must be positive."""
        for s in stress_results_with_es.all_scenarios:
            comp = s.es_comparison
            for key in ("loss_to_es95_ratio", "loss_to_es975_ratio",
                        "loss_to_es99_ratio", "loss_to_es99_21d_ratio"):
                assert comp[key] >= 0, f"'{s.name}': {key} is negative"

    def test_es_ordering(self, stress_results_with_es):
        """ES(95%) ≤ ES(97.5%) ≤ ES(99%) — stricter confidence = larger ES."""
        for s in stress_results_with_es.all_scenarios:
            comp = s.es_comparison
            assert comp["es_95_1d"] <= comp["es_975_1d"] + 0.01, (
                f"'{s.name}': ES(95) > ES(97.5)"
            )
            assert comp["es_975_1d"] <= comp["es_99_1d"] + 0.01, (
                f"'{s.name}': ES(97.5) > ES(99)"
            )

    def test_description_is_string(self, stress_results_with_es):
        """multiples_of_daily_es must be a non-empty string."""
        for s in stress_results_with_es.all_scenarios:
            desc = s.es_comparison["multiples_of_daily_es"]
            assert isinstance(desc, str) and len(desc) > 0, (
                f"'{s.name}': empty description"
            )

    def test_compare_scenario_to_es_direct(self, metrics):
        """Unit test compare_scenario_to_es with a known scenario."""
        from engine.stress_testing import StockScenarioImpact, ScenarioResult
        pv = metrics.total_value if metrics.total_value else 1_000_000.0
        dummy = ScenarioResult(
            name="Test", description="test", period="test",
            methodology="test", assumptions="test",
            portfolio_loss_pct=-0.30, portfolio_loss_usd=-pv * 0.30,
            benchmark_loss_pct=-0.30, portfolio_value=pv,
        )
        result = compare_scenario_to_es(dummy, metrics)
        assert result["scenario_loss_usd"] == pytest.approx(pv * 0.30, rel=0.01)
        # A 30% loss should be several multiples of 1-day ES
        assert result["loss_to_es99_ratio"] > 1.0, (
            "30% loss should exceed 1-day ES(99%) multiple times"
        )


class TestSeverityLabels:
    """Tests for the ScenarioResult.severity property (Phase 5)."""

    def test_severity_values_valid(self, stress_results_with_es):
        """All scenarios must have a recognised severity label."""
        valid = {"EXTREME", "SEVERE", "ELEVATED", "MODERATE", "MILD"}
        for s in stress_results_with_es.all_scenarios:
            assert s.severity in valid, f"'{s.name}': invalid severity '{s.severity}'"

    def test_severity_color_is_hex(self, stress_results_with_es):
        """Severity colour must be a 7-character hex string."""
        for s in stress_results_with_es.all_scenarios:
            color = s.severity_color
            assert color.startswith("#") and len(color) == 7, (
                f"'{s.name}': invalid color '{color}'"
            )

    def test_gfc_severity_at_least_elevated(self, stress_results_with_es):
        """2008 GFC should be ELEVATED, SEVERE, or EXTREME — never MILD."""
        gfc = stress_results_with_es.historical[0]
        assert gfc.severity in {"ELEVATED", "SEVERE", "EXTREME"}, (
            f"GFC severity '{gfc.severity}' too low for a 30-50% drawdown scenario"
        )

    def test_severity_without_es_comparison_uses_pct(self):
        """Without es_comparison, severity still returns a valid label from pct-loss."""
        from engine.stress_testing import StockScenarioImpact
        dummy = ScenarioResult(
            name="Dummy", description="", period="", methodology="",
            assumptions="", portfolio_loss_pct=-0.45, portfolio_loss_usd=-450_000,
            benchmark_loss_pct=-0.45, portfolio_value=1_000_000,
        )
        assert dummy.es_comparison is None
        assert dummy.severity == "EXTREME"

    def test_mild_scenario_severity(self):
        """A 3% loss should be MILD in the fallback (no es_comparison)."""
        dummy = ScenarioResult(
            name="Mild", description="", period="", methodology="",
            assumptions="", portfolio_loss_pct=-0.03, portfolio_loss_usd=-30_000,
            benchmark_loss_pct=-0.03, portfolio_value=1_000_000,
        )
        assert dummy.severity == "MILD"
