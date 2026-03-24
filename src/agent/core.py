"""Main ForexTradingAgent orchestrator."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from config import config
from src.agent.indicators import compute_all
from src.agent.portfolio import Portfolio, Trade
from src.agent.risk_manager import RiskManager
from src.agent.signals import Signal, SignalGenerator, SignalType
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.utils import get_logger, timestamp_now

logger = get_logger(__name__)


class ForexTradingAgent:
    """High-level agent that orchestrates data → indicators → signals → trades.

    Parameters
    ----------
    pairs:
        List of currency pairs to trade (e.g. ``["EUR/USD", "GBP/USD"]``).
    balance:
        Initial account balance.
    cfg:
        Optional config object; falls back to the global singleton.
    """

    DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]

    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        balance: Optional[float] = None,
        cfg=None,
    ) -> None:
        self._cfg = cfg or config
        self.pairs = pairs or self.DEFAULT_PAIRS

        ind = self._cfg.indicators
        sig = self._cfg.signals
        trd = self._cfg.trading

        self.signal_gen = SignalGenerator(
            min_confirmations=sig["min_confirmations"],
            rsi_overbought=sig["rsi_overbought"],
            rsi_oversold=sig["rsi_oversold"],
        )
        self.risk_manager = RiskManager(
            initial_balance=balance or trd["initial_balance"],
            risk_per_trade=trd["risk_per_trade"],
            stop_loss_pct=trd["stop_loss_pct"],
            take_profit_ratio=trd["take_profit_ratio"],
            max_drawdown=trd["max_drawdown"],
        )
        self.portfolio = Portfolio()
        self.fetcher = DataFetcher(cache_ttl=self._cfg.data["cache_ttl_seconds"])
        self.processor = DataProcessor()

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

        self._running = False
        self._last_signals: Dict[str, Signal] = {}

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------

    def analyse(self, pair: str, df: Optional[pd.DataFrame] = None) -> Signal:
        """Return the latest signal for *pair*, optionally using supplied *df*."""
        if df is None:
            raw = self.fetcher.fetch(
                pair,
                period=self._cfg.data["history_period"],
                interval=self._cfg.data["interval"],
            )
            df = self.processor.clean(raw)

        if df is None or df.empty:
            logger.warning("No data available for %s", pair)
            return Signal(SignalType.HOLD, 0, price=0.0)

        enriched = compute_all(df, **self._ind_params)
        signal = self.signal_gen.generate(enriched)
        self._last_signals[pair] = signal
        return signal

    def analyse_all(self) -> Dict[str, Signal]:
        """Analyse all pairs and return a dict of signals."""
        return {pair: self.analyse(pair) for pair in self.pairs}

    def execute_signal(self, pair: str, signal: Signal) -> Optional[Trade]:
        """Execute a BUY or SELL signal by creating a position."""
        if signal.signal == SignalType.HOLD:
            return None

        spec = self.risk_manager.calculate_position(
            pair=pair,
            direction=signal.signal.value,
            entry_price=signal.price,
        )
        if spec is None:
            return None

        trade = self.portfolio.open_trade(
            pair=pair,
            direction=spec.direction,
            entry_price=spec.entry_price,
            stop_loss=spec.stop_loss,
            take_profit=spec.take_profit,
            units=spec.units,
            risk_amount=spec.risk_amount,
        )
        return trade

    def close_trade(self, trade: Trade, exit_price: float, status: str = "CLOSED") -> float:
        """Close *trade* and update the risk manager balance."""
        pnl = self.portfolio.close_trade(trade, exit_price, status)
        self.risk_manager.update_balance(pnl)
        return pnl

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        return {
            "timestamp": timestamp_now(),
            "running": self._running,
            "pairs": self.pairs,
            "balance": round(self.risk_manager.balance, 2),
            "peak_balance": round(self.risk_manager.peak_balance, 2),
            "current_drawdown_pct": round(self.risk_manager.current_drawdown() * 100, 2),
            "drawdown_breached": self.risk_manager.is_drawdown_breached(),
            "portfolio": self.portfolio.summary(),
            "last_signals": {
                pair: {
                    "signal": s.signal.value,
                    "confirmations": s.confirmations,
                    "price": s.price,
                    "reasons": s.reasons,
                }
                for pair, s in self._last_signals.items()
            },
        }

    def reset(self) -> None:
        self.portfolio.reset()
        self.risk_manager.reset()
        self._last_signals.clear()
        self.fetcher.clear_cache()
        logger.info("Agent reset.")
