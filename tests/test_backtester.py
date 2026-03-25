"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import BacktestEngine, BacktestResult


def _synthetic_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame long enough for all indicators."""
    rng = np.random.default_rng(seed)
    prices = 1.10 + np.cumsum(rng.normal(0, 0.001, n))
    spread = rng.uniform(0.0002, 0.001, n)
    return pd.DataFrame(
        {
            "open": prices - spread / 2,
            "high": prices + spread,
            "low": prices - spread,
            "close": prices,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )


class TestBacktestEngine:
    def setup_method(self):
        self.engine = BacktestEngine()

    def test_returns_result_object(self):
        df = _synthetic_df()
        result = self.engine.run(df, pair="EUR/USD")
        assert isinstance(result, BacktestResult)

    def test_result_pair(self):
        df = _synthetic_df()
        result = self.engine.run(df, pair="EUR/USD")
        assert result.pair == "EUR/USD"

    def test_initial_balance(self):
        df = _synthetic_df()
        result = self.engine.run(df)
        assert result.initial_balance == 10_000.0

    def test_trade_counts_consistent(self):
        df = _synthetic_df()
        result = self.engine.run(df)
        assert result.winning_trades + result.losing_trades == result.total_trades

    def test_win_rate_range(self):
        df = _synthetic_df()
        result = self.engine.run(df)
        if result.total_trades > 0:
            assert 0.0 <= result.win_rate <= 1.0

    def test_max_drawdown_non_negative(self):
        df = _synthetic_df()
        result = self.engine.run(df)
        assert result.max_drawdown_pct >= 0.0

    def test_no_trades_on_tiny_df(self):
        # Too few bars for indicators to produce valid values
        df = _synthetic_df(n=10)
        result = self.engine.run(df)
        assert result.total_trades == 0

    def test_trades_list_matches_count(self):
        df = _synthetic_df()
        result = self.engine.run(df)
        assert len(result.trades) == result.total_trades

    def test_str_representation(self):
        df = _synthetic_df()
        result = self.engine.run(df, pair="GBP/USD")
        text = str(result)
        assert "GBP/USD" in text
        assert "Balance" in text
