"""Tests for technical indicator calculations."""

import numpy as np
import pandas as pd
import pytest

from src.agent.indicators import (
    atr,
    bollinger_bands,
    compute_all,
    ema,
    macd,
    rsi,
    sma,
)


def _price_series(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 1.10 + np.cumsum(rng.normal(0, 0.001, n))
    return pd.Series(prices, name="close")


def _ohlcv_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    close = _price_series(n, seed).values
    rng = np.random.default_rng(seed + 1)
    spread = rng.uniform(0.0002, 0.001, n)
    df = pd.DataFrame(
        {
            "open": close - spread / 2,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )
    return df


class TestEMA:
    def test_length(self):
        s = _price_series(50)
        result = ema(s, 12)
        assert len(result) == len(s)

    def test_convergence(self):
        # EMA of constant series should equal that constant
        s = pd.Series([1.5] * 50)
        result = ema(s, 10)
        assert abs(result.iloc[-1] - 1.5) < 1e-6

    def test_no_nan_at_end(self):
        s = _price_series(60)
        result = ema(s, 12)
        assert not result.iloc[-1:].isna().any()


class TestSMA:
    def test_length(self):
        s = _price_series(50)
        result = sma(s, 20)
        assert len(result) == len(s)

    def test_first_valid(self):
        s = pd.Series(range(1, 11), dtype=float)
        result = sma(s, 5)
        # First 4 should be NaN, 5th should be (1+2+3+4+5)/5
        assert result.iloc[:4].isna().all()
        assert abs(result.iloc[4] - 3.0) < 1e-9


class TestRSI:
    def test_range(self):
        s = _price_series(100)
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_overbought_uptrend(self):
        # Monotonically rising prices → RSI should be high
        s = pd.Series(np.linspace(1.0, 1.5, 200))
        result = rsi(s, 14)
        assert result.iloc[-1] > 60

    def test_oversold_downtrend(self):
        # Monotonically falling prices → RSI should be low
        s = pd.Series(np.linspace(1.5, 1.0, 200))
        result = rsi(s, 14)
        assert result.iloc[-1] < 40


class TestMACD:
    def test_keys(self):
        s = _price_series(100)
        result = macd(s)
        assert set(result.keys()) == {"macd", "signal", "histogram"}

    def test_histogram_is_macd_minus_signal(self):
        s = _price_series(100)
        result = macd(s)
        diff = (result["macd"] - result["signal"] - result["histogram"]).abs()
        assert diff.max() < 1e-10

    def test_length(self):
        s = _price_series(100)
        result = macd(s)
        assert len(result["macd"]) == 100


class TestBollingerBands:
    def test_keys(self):
        s = _price_series(100)
        result = bollinger_bands(s)
        assert "upper" in result and "lower" in result and "middle" in result

    def test_upper_above_lower(self):
        s = _price_series(100)
        result = bollinger_bands(s)
        valid_idx = result["upper"].dropna().index
        assert (result["upper"][valid_idx] >= result["lower"][valid_idx]).all()

    def test_pct_b_range_mostly_within_bounds(self):
        s = _price_series(200)
        result = bollinger_bands(s, period=20, num_std=2.0)
        pct_b = result["pct_b"].dropna()
        # Most values should be within [0, 1] for a normal price series
        within_bounds = ((pct_b >= 0) & (pct_b <= 1)).mean()
        assert within_bounds > 0.7


class TestATR:
    def test_positive(self):
        df = _ohlcv_df(100)
        result = atr(df, 14)
        assert (result.dropna() > 0).all()


class TestComputeAll:
    def test_columns_added(self):
        df = _ohlcv_df(100)
        result = compute_all(df)
        expected_cols = [
            "ema_fast", "ema_slow", "rsi", "macd", "macd_signal",
            "macd_hist", "bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "atr",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self):
        df = _ohlcv_df(50)
        result = compute_all(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
