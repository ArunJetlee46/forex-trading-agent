"""Profit analysis: evaluate expected profit before order execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProfitAnalysis:
    """Result of profit analysis for a potential trade.

    Attributes
    ----------
    pair:
        Currency pair symbol.
    direction:
        ``"BUY"`` or ``"SELL"``.
    entry_price:
        Proposed order entry price.
    stop_loss:
        Stop-loss price level.
    take_profit:
        Take-profit price level.
    atr:
        Average True Range at signal time (0.0 when unavailable).
    risk_reward_ratio:
        Reward-to-risk ratio for this trade (e.g. 3.0 means 3:1).
    estimated_win_rate:
        Probability of the trade being a winner, estimated from signal
        strength (confirmations).
    expected_value:
        Dollar-denominated expected value:
        ``win_rate × expected_profit − loss_rate × expected_loss``.
    expected_profit:
        Dollar profit if the trade hits take-profit.
    expected_loss:
        Dollar loss (positive value) if the trade hits stop-loss.
    is_viable:
        ``True`` when all viability conditions pass.
    reasons:
        Human-readable list of analysis notes and any failing conditions.
    """

    pair: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    risk_reward_ratio: float
    estimated_win_rate: float
    expected_value: float
    expected_profit: float
    expected_loss: float
    is_viable: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "atr": self.atr,
            "risk_reward_ratio": self.risk_reward_ratio,
            "estimated_win_rate": self.estimated_win_rate,
            "expected_value": self.expected_value,
            "expected_profit": self.expected_profit,
            "expected_loss": self.expected_loss,
            "is_viable": self.is_viable,
            "reasons": self.reasons,
        }


class ProfitAnalyzer:
    """Estimate expected profit/loss before executing a trade signal.

    The analyser uses the signal *confirmations* count as a proxy for
    win-probability and the ATR to validate that stop-loss placement is
    meaningful relative to current market volatility.

    Parameters
    ----------
    min_risk_reward:
        Minimum acceptable reward-to-risk ratio (default 1.5).
    min_confirmations:
        Minimum signal confirmations treated as the baseline for win-rate
        estimation (should match ``SignalGenerator.min_confirmations``).
    max_confirmations:
        Maximum possible signal confirmations (number of indicators).
    base_win_rate:
        Estimated win probability at *min_confirmations* (default 50 %).
    max_win_rate:
        Estimated win probability at *max_confirmations* (default 65 %).
    min_sl_atr_ratio:
        Stop-loss distance must be at least this multiple of ATR so that
        the stop is not clipped by normal intra-bar noise (default 0.3).
        The check is skipped when ATR is unavailable (0.0).
    """

    def __init__(
        self,
        min_risk_reward: float = 1.5,
        min_confirmations: int = 3,
        max_confirmations: int = 4,
        base_win_rate: float = 0.50,
        max_win_rate: float = 0.65,
        min_sl_atr_ratio: float = 0.3,
    ) -> None:
        self.min_risk_reward = min_risk_reward
        self.min_confirmations = min_confirmations
        self.max_confirmations = max_confirmations
        self.base_win_rate = base_win_rate
        self.max_win_rate = max_win_rate
        self.min_sl_atr_ratio = min_sl_atr_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_amount: float,
        confirmations: int,
        df: Optional[pd.DataFrame] = None,
    ) -> ProfitAnalysis:
        """Compute profit analysis for a potential trade.

        Parameters
        ----------
        pair:
            Currency pair symbol.
        direction:
            ``"BUY"`` or ``"SELL"``.
        entry_price, stop_loss, take_profit:
            Price levels produced by the risk manager.
        risk_amount:
            Dollar amount at risk (balance × risk_per_trade).
        confirmations:
            Number of indicator confirmations for the signal.
        df:
            Indicator-enriched OHLCV DataFrame used to read ATR.
            Pass ``None`` when unavailable (ATR check is then skipped).

        Returns
        -------
        ProfitAnalysis
        """
        reasons: List[str] = []

        # ------------------------------------------------------------------
        # 1. Geometric distances
        # ------------------------------------------------------------------
        if direction == "BUY":
            profit_distance = take_profit - entry_price
            loss_distance = entry_price - stop_loss
        else:
            profit_distance = entry_price - take_profit
            loss_distance = stop_loss - entry_price

        profit_distance = max(profit_distance, 0.0)
        loss_distance = max(loss_distance, 1e-10)  # guard against division by zero

        # ------------------------------------------------------------------
        # 2. Risk/reward ratio
        # ------------------------------------------------------------------
        risk_reward_ratio = profit_distance / loss_distance
        reasons.append(
            f"Risk/reward ratio: {risk_reward_ratio:.2f} "
            f"(min required: {self.min_risk_reward:.2f})"
        )

        # ------------------------------------------------------------------
        # 3. Dollar-denominated P&L
        # ------------------------------------------------------------------
        units = risk_amount / loss_distance
        expected_profit = profit_distance * units
        expected_loss = loss_distance * units  # equals risk_amount

        # ------------------------------------------------------------------
        # 4. Win-rate estimate from signal confirmations
        # ------------------------------------------------------------------
        conf_range = max(1, self.max_confirmations - self.min_confirmations)
        conf_above_min = max(0, confirmations - self.min_confirmations)
        win_rate_range = self.max_win_rate - self.base_win_rate
        estimated_win_rate = min(
            self.max_win_rate,
            self.base_win_rate + win_rate_range * (conf_above_min / conf_range),
        )
        reasons.append(
            f"Estimated win rate: {estimated_win_rate*100:.1f}% "
            f"(from {confirmations} confirmations)"
        )

        # ------------------------------------------------------------------
        # 5. Expected value
        # ------------------------------------------------------------------
        expected_value = (
            estimated_win_rate * expected_profit
            - (1.0 - estimated_win_rate) * expected_loss
        )
        reasons.append(f"Expected value: ${expected_value:.2f}")

        # ------------------------------------------------------------------
        # 6. ATR-based stop placement check
        # ------------------------------------------------------------------
        atr_val = self._extract_atr(df)
        atr_check_passed = True
        if atr_val > 0:
            sl_atr_ratio = loss_distance / atr_val
            if sl_atr_ratio < self.min_sl_atr_ratio:
                atr_check_passed = False
                reasons.append(
                    f"Stop-loss distance ({loss_distance:.5f}) is too tight "
                    f"relative to ATR ({atr_val:.5f}): ratio={sl_atr_ratio:.2f} "
                    f"< min {self.min_sl_atr_ratio:.2f}"
                )
            else:
                reasons.append(
                    f"ATR check passed: SL/ATR ratio={sl_atr_ratio:.2f}"
                )
        else:
            reasons.append("ATR unavailable; volatility check skipped")

        # ------------------------------------------------------------------
        # 7. Viability decision
        # ------------------------------------------------------------------
        is_viable = (
            risk_reward_ratio >= self.min_risk_reward
            and expected_value > 0
            and atr_check_passed
        )

        if not is_viable:
            if risk_reward_ratio < self.min_risk_reward:
                reasons.append(
                    f"REJECTED: RR {risk_reward_ratio:.2f} below minimum {self.min_risk_reward:.2f}"
                )
            if expected_value <= 0:
                reasons.append(f"REJECTED: Non-positive expected value (${expected_value:.2f})")
        else:
            reasons.append("VIABLE: trade passes all profit analysis checks")

        logger.info(
            "%s %s  RR=%.2f  EV=$%.2f  win_rate=%.0f%%  viable=%s",
            direction, pair, risk_reward_ratio, expected_value,
            estimated_win_rate * 100, is_viable,
        )

        return ProfitAnalysis(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=round(atr_val, 5),
            risk_reward_ratio=round(risk_reward_ratio, 4),
            estimated_win_rate=round(estimated_win_rate, 4),
            expected_value=round(expected_value, 2),
            expected_profit=round(expected_profit, 2),
            expected_loss=round(expected_loss, 2),
            is_viable=is_viable,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_atr(df: Optional[pd.DataFrame]) -> float:
        """Return the last ATR value from *df*, or 0.0 when unavailable."""
        if df is None or "atr" not in df.columns:
            return 0.0
        last_atr = df["atr"].dropna()
        if last_atr.empty:
            return 0.0
        return float(last_atr.iloc[-1])
