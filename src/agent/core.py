"""Main ForexTradingAgent orchestrator."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from config import config
from src.agent.indicators import compute_all
from src.agent.portfolio import Portfolio, Trade
from src.agent.profit_analyzer import ProfitAnalysis, ProfitAnalyzer
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

    DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "XAU/EUR"]

    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        balance: Optional[float] = None,
        cfg=None,
        broker=None,
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
        self.broker = broker  # optional MT5Broker (or None for paper trading)
        self.profit_analyzer = ProfitAnalyzer(
            min_risk_reward=trd.get("min_risk_reward", 1.5),
            min_confirmations=sig["min_confirmations"],
        )

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
        self._last_dfs: Dict[str, pd.DataFrame] = {}  # cached enriched DataFrames

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
        self._last_dfs[pair] = enriched  # cache for downstream profit analysis
        signal = self.signal_gen.generate(enriched)
        self._last_signals[pair] = signal
        return signal

    def analyse_all(self) -> Dict[str, Signal]:
        """Analyse all pairs and return a dict of signals."""
        return {pair: self.analyse(pair) for pair in self.pairs}

    def execute_signal(self, pair: str, signal: Signal) -> Optional[Trade]:
        """Execute a BUY or SELL signal by creating a position.

        When an :class:`~src.broker.mt5_broker.MT5Broker` is attached (via the
        *broker* constructor parameter), the order is also sent to MetaTrader 5.
        """
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

        if self.broker is not None:
            from src.data.mt5_fetcher import MT5_SYMBOL_MAP
            symbol = MT5_SYMBOL_MAP.get(pair, pair.replace("/", ""))
            ticket = self.broker.place_order(
                symbol=symbol,
                direction=spec.direction,
                volume=spec.units,
                price=spec.entry_price,
                stop_loss=spec.stop_loss,
                take_profit=spec.take_profit,
            )
            if ticket is not None:
                trade.broker_ticket = ticket
            else:
                logger.warning("MT5 order placement failed for %s; position tracked locally only.", pair)

        return trade

    def close_trade(self, trade: Trade, exit_price: float, status: str = "CLOSED") -> float:
        """Close *trade* and update the risk manager balance."""
        pnl = self.portfolio.close_trade(trade, exit_price, status)
        self.risk_manager.update_balance(pnl)
        return pnl

    # ------------------------------------------------------------------
    # Profit analysis & automated execution
    # ------------------------------------------------------------------

    def analyse_profit(self, pair: str, signal: Signal) -> Optional[ProfitAnalysis]:
        """Run profit analysis for an actionable *signal* on *pair*.

        Returns ``None`` when the signal is ``HOLD`` or the risk manager
        cannot compute a position (e.g. drawdown breached).
        """
        if signal.signal == SignalType.HOLD:
            return None

        spec = self.risk_manager.calculate_position(
            pair=pair,
            direction=signal.signal.value,
            entry_price=signal.price,
        )
        if spec is None:
            return None

        df = self._last_dfs.get(pair)
        return self.profit_analyzer.analyse(
            pair=pair,
            direction=spec.direction,
            entry_price=spec.entry_price,
            stop_loss=spec.stop_loss,
            take_profit=spec.take_profit,
            risk_amount=spec.risk_amount,
            confirmations=signal.confirmations,
            df=df,
        )

    def auto_execute_signals(self) -> List[Trade]:
        """Analyse all pairs, apply profit analysis, and execute viable trades.

        For each pair that does *not* already have an open position:

        1. Fetch market data and generate a signal.
        2. Run :meth:`analyse_profit` to estimate expected profit.
        3. Execute the trade only when the analysis is viable.

        Returns
        -------
        list[Trade]
            Trades that were opened during this cycle.
        """
        executed: List[Trade] = []
        open_pairs = {t.pair for t in self.portfolio.get_open_trades()}

        for pair in self.pairs:
            if pair in open_pairs:
                logger.debug("Skipping %s: already has an open position.", pair)
                continue

            signal = self.analyse(pair)

            if signal.signal == SignalType.HOLD:
                logger.debug("%s: HOLD – skipping.", pair)
                continue

            analysis = self.analyse_profit(pair, signal)
            if analysis is None:
                logger.info("%s: profit analysis unavailable – skipping.", pair)
                continue

            if not analysis.is_viable:
                logger.info(
                    "Skipping %s %s: not viable (RR=%.2f, EV=$%.2f).",
                    pair, signal.signal.value,
                    analysis.risk_reward_ratio, analysis.expected_value,
                )
                continue

            trade = self.execute_signal(pair, signal)
            if trade is not None:
                logger.info(
                    "Auto-executed %s %s  entry=%.5f  SL=%.5f  TP=%.5f  EV=$%.2f",
                    trade.direction, pair,
                    trade.entry_price, trade.stop_loss, trade.take_profit,
                    analysis.expected_value,
                )
                executed.append(trade)

        return executed

    def monitor_positions(self, current_prices: Dict[str, float]) -> List[Trade]:
        """Check open positions against *current_prices* and close SL/TP hits.

        Parameters
        ----------
        current_prices:
            Mapping of pair → current market price (e.g. ``{"EUR/USD": 1.09}``).

        Returns
        -------
        list[Trade]
            Trades that were closed during this call.
        """
        closed: List[Trade] = []
        for trade in self.portfolio.get_open_trades():
            price = current_prices.get(trade.pair)
            if price is None:
                continue

            hit_sl, hit_tp = False, False
            if trade.direction == "BUY":
                hit_sl = price <= trade.stop_loss
                hit_tp = price >= trade.take_profit
            else:  # SELL
                hit_sl = price >= trade.stop_loss
                hit_tp = price <= trade.take_profit

            if hit_sl or hit_tp:
                exit_price = trade.stop_loss if hit_sl else trade.take_profit
                status = "STOPPED" if hit_sl else "CLOSED"
                pnl = self.close_trade(trade, exit_price, status)
                logger.info(
                    "Auto-closed trade %s [%s]: %s  exit=%.5f  pnl=$%.2f",
                    trade.trade_id, status, trade.pair, exit_price, pnl,
                )
                closed.append(trade)

        return closed

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
        self._last_dfs.clear()
        self.fetcher.clear_cache()
        logger.info("Agent reset.")
