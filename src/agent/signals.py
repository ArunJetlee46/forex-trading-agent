"""Buy / sell signal generation using a multi-indicator consensus approach."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    signal: SignalType
    confirmations: int
    reasons: List[str] = field(default_factory=list)
    price: float = 0.0
    rsi: float = 0.0
    macd_hist: float = 0.0
    bb_pct_b: float = 0.0

    def __str__(self) -> str:
        return (
            f"Signal({self.signal.value}, confirmations={self.confirmations}, "
            f"price={self.price:.5f}, reasons={self.reasons})"
        )


class SignalGenerator:
    """Generate BUY / SELL / HOLD signals from indicator-enriched DataFrames.

    A BUY signal is generated when at least *min_confirmations* of the
    following conditions are true simultaneously:

    1. Fast EMA crossed above slow EMA (bullish MA crossover)
    2. RSI < *rsi_overbought* (not in overbought territory)
    3. MACD histogram > 0 and crossed from negative to positive
    4. Price is in the lower Bollinger Band zone (pct_b < 0.5)

    A SELL signal mirrors those conditions in reverse.
    """

    def __init__(
        self,
        min_confirmations: int = 3,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
    ) -> None:
        self.min_confirmations = min_confirmations
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, df: pd.DataFrame) -> Signal:
        """Return the latest signal for an indicator-enriched *df*."""
        required = [
            "ema_fast", "ema_slow", "rsi", "macd_hist",
            "bb_pct_b", "close",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing or len(df) < 2:
            logger.debug("Insufficient data for signal generation: %s", missing)
            return Signal(SignalType.HOLD, 0)

        # Use the last two rows to detect crossovers
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        buy_score, buy_reasons = self._score_buy(curr, prev)
        sell_score, sell_reasons = self._score_sell(curr, prev)

        price = float(curr["close"])
        rsi_val = float(curr["rsi"]) if pd.notna(curr["rsi"]) else 50.0
        macd_hist = float(curr["macd_hist"]) if pd.notna(curr["macd_hist"]) else 0.0
        bb_pct_b = float(curr["bb_pct_b"]) if pd.notna(curr["bb_pct_b"]) else 0.5

        if buy_score >= self.min_confirmations and buy_score > sell_score:
            logger.info("BUY signal at %.5f  confirmations=%d", price, buy_score)
            return Signal(
                SignalType.BUY, buy_score, buy_reasons, price, rsi_val, macd_hist, bb_pct_b
            )

        if sell_score >= self.min_confirmations and sell_score > buy_score:
            logger.info("SELL signal at %.5f  confirmations=%d", price, sell_score)
            return Signal(
                SignalType.SELL, sell_score, sell_reasons, price, rsi_val, macd_hist, bb_pct_b
            )

        return Signal(SignalType.HOLD, max(buy_score, sell_score), [], price, rsi_val, macd_hist, bb_pct_b)

    def generate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with a ``signal`` column for every row."""
        signals = []
        for i in range(1, len(df)):
            window = df.iloc[: i + 1]
            sig = self.generate(window)
            signals.append(sig.signal.value)
        # Prepend HOLD for the first row (no previous row available)
        signals = [SignalType.HOLD.value] + signals
        result = df.copy()
        result["signal"] = signals
        return result

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _score_buy(self, curr: pd.Series, prev: pd.Series):
        score = 0
        reasons: List[str] = []

        # 1. Bullish EMA crossover
        if (
            pd.notna(curr["ema_fast"]) and pd.notna(curr["ema_slow"])
            and pd.notna(prev["ema_fast"]) and pd.notna(prev["ema_slow"])
            and float(prev["ema_fast"]) <= float(prev["ema_slow"])
            and float(curr["ema_fast"]) > float(curr["ema_slow"])
        ):
            score += 1
            reasons.append("EMA bullish crossover")

        # 2. RSI not overbought
        if pd.notna(curr["rsi"]) and float(curr["rsi"]) < self.rsi_overbought:
            score += 1
            reasons.append(f"RSI={curr['rsi']:.1f} below overbought ({self.rsi_overbought})")

        # 3. MACD histogram crossed from negative to positive
        if (
            pd.notna(curr["macd_hist"]) and pd.notna(prev["macd_hist"])
            and float(prev["macd_hist"]) < 0
            and float(curr["macd_hist"]) >= 0
        ):
            score += 1
            reasons.append("MACD histogram bullish crossover")

        # 4. Price in lower Bollinger Band zone (pct_b < 0.5)
        if pd.notna(curr["bb_pct_b"]) and float(curr["bb_pct_b"]) < 0.5:
            score += 1
            reasons.append(f"Price in lower BB zone (pct_b={curr['bb_pct_b']:.2f})")

        return score, reasons

    def _score_sell(self, curr: pd.Series, prev: pd.Series):
        score = 0
        reasons: List[str] = []

        # 1. Bearish EMA crossover
        if (
            pd.notna(curr["ema_fast"]) and pd.notna(curr["ema_slow"])
            and pd.notna(prev["ema_fast"]) and pd.notna(prev["ema_slow"])
            and float(prev["ema_fast"]) >= float(prev["ema_slow"])
            and float(curr["ema_fast"]) < float(curr["ema_slow"])
        ):
            score += 1
            reasons.append("EMA bearish crossover")

        # 2. RSI not oversold
        if pd.notna(curr["rsi"]) and float(curr["rsi"]) > self.rsi_oversold:
            score += 1
            reasons.append(f"RSI={curr['rsi']:.1f} above oversold ({self.rsi_oversold})")

        # 3. MACD histogram crossed from positive to negative
        if (
            pd.notna(curr["macd_hist"]) and pd.notna(prev["macd_hist"])
            and float(prev["macd_hist"]) > 0
            and float(curr["macd_hist"]) <= 0
        ):
            score += 1
            reasons.append("MACD histogram bearish crossover")

        # 4. Price in upper Bollinger Band zone (pct_b > 0.5)
        if pd.notna(curr["bb_pct_b"]) and float(curr["bb_pct_b"]) > 0.5:
            score += 1
            reasons.append(f"Price in upper BB zone (pct_b={curr['bb_pct_b']:.2f})")

        return score, reasons
