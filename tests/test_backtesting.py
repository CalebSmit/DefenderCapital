"""Tests for engine/backtesting.py"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.backtesting import (
    run_backtest, kupiec_pof_test, christoffersen_independence_test, BacktestResult
)


class TestBacktestStructure:
    def test_returns_backtest_result(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        assert isinstance(result, BacktestResult)

    def test_forecast_df_has_required_columns(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        required = {'date', 'realized_pnl', 'var_forecast', 'exception'}
        assert required.issubset(set(result.forecast_df.columns))

    def test_exception_count_consistent(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        assert result.n_exceptions == int(result.forecast_df['exception'].sum())

    def test_exception_rate_plausible(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        # Should be in [0%, 30%] — not trivially 0 or too high
        assert 0.0 <= result.exception_rate <= 0.30


class TestKupiecTest:
    def test_kupiec_returns_float_pvalue(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        assert isinstance(result.kupiec_pvalue, float)
        assert 0.0 <= result.kupiec_pvalue <= 1.0

    def test_kupiec_statistic_nonneg(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        assert result.kupiec_statistic >= 0.0

    def test_kupiec_result_is_pass_or_fail(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        assert result.kupiec_result in ("PASS", "FAIL")


class TestChristoffersenTest:
    def test_christoffersen_pvalue_valid(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        assert 0.0 <= result.christoffersen_pvalue <= 1.0

    def test_christoffersen_result_valid(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        assert result.christoffersen_result in ("PASS", "FAIL")


class TestESBacktest:
    def test_es_adequacy_string(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        assert isinstance(result.es_adequacy, str)
        assert len(result.es_adequacy) > 0

    def test_backtest_summary_nonempty(self, market_data):
        result = run_backtest(market_data, confidence=0.95, min_periods=100)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 20
