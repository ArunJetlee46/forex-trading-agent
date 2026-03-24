"""Technical indicator calculations.

All functions accept a pandas Series or DataFrame and return a Series
(or dict of Series) so they can be easily composed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # When avg_loss is 0 the RS is infinite → RSI = 100
    rsi_values = avg_loss.copy()
    rsi_values[:] = np.nan
    non_zero_loss = avg_loss != 0
    rs = avg_gain[non_zero_loss] / avg_loss[non_zero_loss]
    rsi_values[non_zero_loss] = 100 - (100 / (1 + rs))
    rsi_values[~non_zero_loss & avg_gain.notna() & (avg_gain > 0)] = 100.0
    rsi_values[~non_zero_loss & avg_gain.notna() & (avg_gain == 0)] = 50.0
    return rsi_values


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """MACD indicator.

    Returns a dict with keys: ``macd``, ``signal``, ``histogram``.
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> dict[str, pd.Series]:
    """Bollinger Bands.

    Returns a dict with keys: ``upper``, ``middle``, ``lower``, ``bandwidth``, ``pct_b``.
    """
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "pct_b": pct_b,
    }


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_all(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """Add all indicator columns to a copy of *df* and return it."""
    df = df.copy()
    close = df["close"]

    df["ema_fast"] = ema(close, ema_fast)
    df["ema_slow"] = ema(close, ema_slow)
    df["rsi"] = rsi(close, rsi_period)

    _macd = macd(close, macd_fast, macd_slow, macd_signal)
    df["macd"] = _macd["macd"]
    df["macd_signal"] = _macd["signal"]
    df["macd_hist"] = _macd["histogram"]

    _bb = bollinger_bands(close, bb_period, bb_std)
    df["bb_upper"] = _bb["upper"]
    df["bb_middle"] = _bb["middle"]
    df["bb_lower"] = _bb["lower"]
    df["bb_pct_b"] = _bb["pct_b"]

    df["atr"] = atr(df)

    return df
