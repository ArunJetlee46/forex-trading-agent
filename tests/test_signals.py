"""Tests for signal generation logic."""

import numpy as np
import pandas as pd
import pytest

from src.agent.indicators import compute_all
from src.agent.signals import Signal, SignalGenerator, SignalType


def _make_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with indicator columns."""
    rng = np.random.default_rng(seed)
    prices = 1.10 + np.cumsum(rng.normal(0, 0.001, n))
    spread = rng.uniform(0.0002, 0.001, n)
    df = pd.DataFrame(
        {
            "open": prices - spread / 2,
            "high": prices + spread,
            "low": prices - spread,
            "close": prices,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )
    return compute_all(df)


class TestSignalGenerator:
    def setup_method(self):
        self.gen = SignalGenerator(min_confirmations=3)

    def test_returns_hold_when_insufficient_data(self):
        df = _make_df(3)
        sig = self.gen.generate(df)
        assert isinstance(sig, Signal)
        assert sig.signal == SignalType.HOLD

    def test_returns_signal_object(self):
        df = _make_df(60)
        sig = self.gen.generate(df)
        assert isinstance(sig, Signal)
        assert sig.signal in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)

    def test_confirmations_non_negative(self):
        df = _make_df(60)
        sig = self.gen.generate(df)
        assert sig.confirmations >= 0

    def test_generate_series_length(self):
        df = _make_df(60)
        result = self.gen.generate_series(df)
        assert len(result) == len(df)
        assert "signal" in result.columns

    def test_generate_series_valid_values(self):
        df = _make_df(60)
        result = self.gen.generate_series(df)
        valid_values = {st.value for st in SignalType}
        assert set(result["signal"].unique()).issubset(valid_values)

    def test_buy_signal_conditions(self):
        """Manufacture a scenario that clearly triggers a BUY."""
        # Create a DataFrame where all 4 BUY conditions are met on the last two rows:
        # 1. EMA crossover (fast crosses above slow)
        # 2. RSI < 70
        # 3. MACD hist crosses from negative to positive
        # 4. bb_pct_b < 0.5 (price in lower band zone)
        df = _make_df(60)
        df = df.copy()
        last_idx = df.index[-1]
        prev_idx = df.index[-2]

        # Force indicator values on the last two rows
        df.loc[prev_idx, "ema_fast"] = 1.09
        df.loc[prev_idx, "ema_slow"] = 1.10  # fast < slow (bearish)
        df.loc[last_idx, "ema_fast"] = 1.11
        df.loc[last_idx, "ema_slow"] = 1.10  # fast > slow (bullish crossover)

        df.loc[last_idx, "rsi"] = 45.0          # not overbought
        df.loc[prev_idx, "macd_hist"] = -0.001  # negative
        df.loc[last_idx, "macd_hist"] = 0.001   # positive crossover
        df.loc[last_idx, "bb_pct_b"] = 0.3      # lower zone

        sig = self.gen.generate(df)
        assert sig.signal == SignalType.BUY
        assert sig.confirmations >= 3

    def test_sell_signal_conditions(self):
        """Manufacture a scenario that clearly triggers a SELL."""
        df = _make_df(60)
        df = df.copy()
        last_idx = df.index[-1]
        prev_idx = df.index[-2]

        df.loc[prev_idx, "ema_fast"] = 1.11
        df.loc[prev_idx, "ema_slow"] = 1.10  # fast > slow (bullish)
        df.loc[last_idx, "ema_fast"] = 1.09
        df.loc[last_idx, "ema_slow"] = 1.10  # fast < slow (bearish crossover)

        df.loc[last_idx, "rsi"] = 55.0          # not oversold
        df.loc[prev_idx, "macd_hist"] = 0.001   # positive
        df.loc[last_idx, "macd_hist"] = -0.001  # negative crossover
        df.loc[last_idx, "bb_pct_b"] = 0.7      # upper zone

        sig = self.gen.generate(df)
        assert sig.signal == SignalType.SELL
        assert sig.confirmations >= 3

    def test_min_confirmations_respected(self):
        """With min_confirmations=5, a 4-confirmation scenario → HOLD."""
        gen = SignalGenerator(min_confirmations=5)
        df = _make_df(60)
        df = df.copy()
        last_idx = df.index[-1]
        prev_idx = df.index[-2]

        # Only 4 buy confirmations possible (max score)
        df.loc[prev_idx, "ema_fast"] = 1.09
        df.loc[prev_idx, "ema_slow"] = 1.10
        df.loc[last_idx, "ema_fast"] = 1.11
        df.loc[last_idx, "ema_slow"] = 1.10
        df.loc[last_idx, "rsi"] = 45.0
        df.loc[prev_idx, "macd_hist"] = -0.001
        df.loc[last_idx, "macd_hist"] = 0.001
        df.loc[last_idx, "bb_pct_b"] = 0.3

        sig = gen.generate(df)
        # 4 confirmations < 5 → HOLD
        assert sig.signal == SignalType.HOLD
