"""Risk management: position sizing, stop-loss, take-profit, drawdown guard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.utils import get_logger, round_price

logger = get_logger(__name__)


@dataclass
class PositionSpec:
    """Fully defined trade parameters produced by the risk manager."""

    pair: str
    direction: str          # "BUY" | "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    units: float            # position size in base currency units
    risk_amount: float      # $ at risk for this trade


class RiskManager:
    """Compute position sizes and guard against excessive drawdown.

    Parameters
    ----------
    initial_balance:
        Starting account equity in account currency (default USD).
    risk_per_trade:
        Fraction of balance to risk on each trade (default 2 %).
    stop_loss_pct:
        Distance from entry to stop-loss expressed as a fraction of entry
        price (default 2 %).
    take_profit_ratio:
        Reward-to-risk ratio for take-profit placement (default 3:1).
    max_drawdown:
        Maximum allowed drawdown as a fraction of peak equity before
        trading is halted (default 10 %).
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.02,
        take_profit_ratio: float = 3.0,
        max_drawdown: float = 0.10,
    ) -> None:
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_ratio = take_profit_ratio
        self.max_drawdown = max_drawdown

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position(
        self,
        pair: str,
        direction: str,
        entry_price: float,
    ) -> Optional[PositionSpec]:
        """Return a :class:`PositionSpec` or *None* if trading should be halted."""

        if self.is_drawdown_breached():
            logger.warning(
                "Max drawdown breached (%.1f%%). Halting trade on %s.",
                self.current_drawdown() * 100,
                pair,
            )
            return None

        if entry_price <= 0:
            logger.error("Invalid entry price %.5f for %s", entry_price, pair)
            return None

        risk_amount = self.balance * self.risk_per_trade
        stop_distance = entry_price * self.stop_loss_pct
        units = risk_amount / stop_distance if stop_distance > 0 else 0.0

        if direction == "BUY":
            stop_loss = round_price(entry_price - stop_distance)
            take_profit = round_price(entry_price + stop_distance * self.take_profit_ratio)
        elif direction == "SELL":
            stop_loss = round_price(entry_price + stop_distance)
            take_profit = round_price(entry_price - stop_distance * self.take_profit_ratio)
        else:
            logger.error("Unknown direction: %s", direction)
            return None

        logger.info(
            "%s %s  entry=%.5f  SL=%.5f  TP=%.5f  units=%.2f  risk=$%.2f",
            direction, pair, entry_price, stop_loss, take_profit, units, risk_amount,
        )

        return PositionSpec(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            units=units,
            risk_amount=risk_amount,
        )

    def update_balance(self, pnl: float) -> None:
        """Update balance and peak after a trade closes."""
        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        logger.debug("Balance updated: $%.2f  (peak=$%.2f)", self.balance, self.peak_balance)

    def current_drawdown(self) -> float:
        """Return the current drawdown as a positive fraction."""
        if self.peak_balance == 0:
            return 0.0
        return max(0.0, (self.peak_balance - self.balance) / self.peak_balance)

    def is_drawdown_breached(self) -> bool:
        return self.current_drawdown() >= self.max_drawdown

    def reset(self) -> None:
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
