"""Trade tracking and portfolio P&L analytics."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.utils import get_logger, timestamp_now, trades_to_dataframe

logger = get_logger(__name__)


@dataclass
class Trade:
    pair: str
    direction: str              # "BUY" | "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    units: float
    risk_amount: float
    open_time: str = field(default_factory=timestamp_now)
    close_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"        # "OPEN" | "CLOSED" | "STOPPED"
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "pair": self.pair,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "units": self.units,
            "risk_amount": self.risk_amount,
            "pnl": self.pnl,
            "status": self.status,
            "open_time": self.open_time,
            "close_time": self.close_time,
        }


class Portfolio:
    """Maintain the list of open / closed trades and compute analytics."""

    def __init__(self) -> None:
        self._trades: List[Trade] = []

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def open_trade(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        units: float,
        risk_amount: float,
    ) -> Trade:
        trade = Trade(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            units=units,
            risk_amount=risk_amount,
        )
        self._trades.append(trade)
        logger.info("Trade opened: %s", trade.to_dict())
        return trade

    def close_trade(
        self,
        trade: Trade,
        exit_price: float,
        status: str = "CLOSED",
    ) -> float:
        """Close *trade* at *exit_price*, compute P&L and return it."""
        trade.exit_price = exit_price
        trade.close_time = timestamp_now()
        trade.status = status

        if trade.direction == "BUY":
            pnl = (exit_price - trade.entry_price) * trade.units
        else:
            pnl = (trade.entry_price - exit_price) * trade.units

        trade.pnl = round(pnl, 2)
        logger.info(
            "Trade closed [%s]: %s  exit=%.5f  pnl=$%.2f",
            status, trade.trade_id, exit_price, trade.pnl,
        )
        return trade.pnl

    def get_open_trades(self) -> List[Trade]:
        return [t for t in self._trades if t.status == "OPEN"]

    def get_closed_trades(self) -> List[Trade]:
        return [t for t in self._trades if t.status != "OPEN"]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        closed = self.get_closed_trades()
        if not closed:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_pnl": 0.0,
                "profit_factor": 0.0,
                "open_trades": len(self.get_open_trades()),
            }

        pnls = [t.pnl for t in closed if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        return {
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(closed), 4),
            "total_pnl": round(sum(pnls), 2),
            "average_pnl": round(sum(pnls) / len(pnls), 2),
            "profit_factor": round(profit_factor, 4),
            "open_trades": len(self.get_open_trades()),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return trades_to_dataframe([t.to_dict() for t in self._trades])

    def reset(self) -> None:
        self._trades.clear()
