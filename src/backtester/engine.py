"""Backtesting engine: replay historical data and simulate trade execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import config
from src.agent.indicators import compute_all
from src.agent.portfolio import Portfolio
from src.agent.risk_manager import RiskManager
from src.agent.signals import SignalGenerator, SignalType
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    pair: str
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Dict] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"BacktestResult({self.pair})\n"
            f"  Balance:        ${self.initial_balance:,.2f} → ${self.final_balance:,.2f}\n"
            f"  Total P&L:      ${self.total_pnl:,.2f}\n"
            f"  Trades:         {self.total_trades}  "
            f"(W={self.winning_trades} / L={self.losing_trades})\n"
            f"  Win Rate:       {self.win_rate*100:.1f}%\n"
            f"  Profit Factor:  {self.profit_factor:.2f}\n"
            f"  Max Drawdown:   {self.max_drawdown_pct:.2f}%\n"
            f"  Sharpe Ratio:   {self.sharpe_ratio:.2f}"
        )


class BacktestEngine:
    """Replay a historical OHLCV DataFrame and simulate the trading agent.

    The engine processes bars one at a time, generating signals and executing
    simulated market orders at the *next bar open* price (to avoid look-ahead
    bias).
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config
        ind = self._cfg.indicators
        sig = self._cfg.signals
        trd = self._cfg.trading

        self._ind_params = {
            "ema_fast": ind["ema_fast"],
            "ema_slow": ind["ema_slow"],
            "rsi_period": ind["rsi_period"],
            "macd_fast": ind["macd_fast"],
            "macd_slow": ind["macd_slow"],
            "macd_signal": ind["macd_signal"],
            "bb_period": ind["bb_period"],
            "bb_std": ind["bb_std"],
        }
        self._sig_params = {
            "min_confirmations": sig["min_confirmations"],
            "rsi_overbought": sig["rsi_overbought"],
            "rsi_oversold": sig["rsi_oversold"],
        }
        self._trd_params = {
            "initial_balance": trd["initial_balance"],
            "risk_per_trade": trd["risk_per_trade"],
            "stop_loss_pct": trd["stop_loss_pct"],
            "take_profit_ratio": trd["take_profit_ratio"],
            "max_drawdown": trd["max_drawdown"],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, pair: str = "UNKNOWN") -> BacktestResult:
        """Run a backtest on *df* for *pair* and return a :class:`BacktestResult`."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["close"])

        enriched = compute_all(df, **self._ind_params)

        risk_mgr = RiskManager(**self._trd_params)
        portfolio = Portfolio()
        sig_gen = SignalGenerator(**self._sig_params)

        initial_balance = risk_mgr.initial_balance
        equity_curve: List[float] = [initial_balance]
        peak = initial_balance
        max_dd = 0.0

        open_trade = None
        trade_records: List[Dict] = []

        for i in range(1, len(enriched)):
            window = enriched.iloc[: i + 1]
            curr = enriched.iloc[i]
            current_price = float(curr["close"])

            # Check if open trade should be closed (SL / TP)
            if open_trade is not None:
                hit_sl, hit_tp = self._check_exit(open_trade, curr)
                if hit_sl or hit_tp:
                    exit_price = (
                        open_trade["stop_loss"] if hit_sl else open_trade["take_profit"]
                    )
                    status = "STOPPED" if hit_sl else "CLOSED"
                    pnl = self._calc_pnl(open_trade, exit_price)
                    risk_mgr.update_balance(pnl)
                    trade_record = dict(open_trade)
                    trade_record.update(
                        {"exit_price": exit_price, "pnl": round(pnl, 2), "status": status}
                    )
                    trade_records.append(trade_record)
                    open_trade = None

            # Generate signal only if we have no open position
            if open_trade is None:
                signal = sig_gen.generate(window)

                if signal.signal != SignalType.HOLD and not risk_mgr.is_drawdown_breached():
                    spec = risk_mgr.calculate_position(
                        pair=pair,
                        direction=signal.signal.value,
                        entry_price=current_price,
                    )
                    if spec is not None:
                        open_trade = {
                            "pair": pair,
                            "direction": spec.direction,
                            "entry_price": spec.entry_price,
                            "stop_loss": spec.stop_loss,
                            "take_profit": spec.take_profit,
                            "units": spec.units,
                            "risk_amount": spec.risk_amount,
                        }

            # Track equity
            equity = risk_mgr.balance
            equity_curve.append(equity)
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Close any remaining open trade at last close
        if open_trade is not None:
            exit_price = float(enriched.iloc[-1]["close"])
            pnl = self._calc_pnl(open_trade, exit_price)
            risk_mgr.update_balance(pnl)
            trade_record = dict(open_trade)
            trade_record.update({"exit_price": exit_price, "pnl": round(pnl, 2), "status": "CLOSED"})
            trade_records.append(trade_record)

        # Compute metrics
        pnls = [t["pnl"] for t in trade_records]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        sharpe = self._sharpe(equity_curve)

        return BacktestResult(
            pair=pair,
            initial_balance=initial_balance,
            final_balance=round(risk_mgr.balance, 2),
            total_trades=len(trade_records),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(len(wins) / len(trade_records), 4) if trade_records else 0.0,
            total_pnl=round(sum(pnls), 2),
            profit_factor=round(profit_factor, 4),
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sharpe, 4),
            trades=trade_records,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_exit(self, trade: Dict, bar: pd.Series):
        direction = trade["direction"]
        low = float(bar["low"])
        high = float(bar["high"])

        if direction == "BUY":
            hit_sl = low <= trade["stop_loss"]
            hit_tp = high >= trade["take_profit"]
        else:
            hit_sl = high >= trade["stop_loss"]
            hit_tp = low <= trade["take_profit"]

        return hit_sl, hit_tp

    def _calc_pnl(self, trade: Dict, exit_price: float) -> float:
        if trade["direction"] == "BUY":
            return (exit_price - trade["entry_price"]) * trade["units"]
        return (trade["entry_price"] - exit_price) * trade["units"]

    @staticmethod
    def _sharpe(equity_curve: List[float], risk_free: float = 0.0) -> float:
        if len(equity_curve) < 2:
            return 0.0
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        excess = returns - risk_free / 252
        std = np.std(excess)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))
