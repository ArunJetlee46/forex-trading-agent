"""Tests for the Portfolio module."""

import pytest

from src.agent.portfolio import Portfolio, Trade


class TestPortfolio:
    def setup_method(self):
        self.portfolio = Portfolio()

    def _open(self, direction="BUY", price=1.10):
        return self.portfolio.open_trade(
            pair="EUR/USD",
            direction=direction,
            entry_price=price,
            stop_loss=price * 0.98 if direction == "BUY" else price * 1.02,
            take_profit=price * 1.06 if direction == "BUY" else price * 0.94,
            units=1000,
            risk_amount=200,
        )

    def test_open_trade_returns_trade(self):
        trade = self._open()
        assert isinstance(trade, Trade)
        assert trade.status == "OPEN"

    def test_open_trade_in_open_list(self):
        trade = self._open()
        assert trade in self.portfolio.get_open_trades()

    def test_close_buy_trade_profit(self):
        trade = self._open("BUY", 1.10)
        pnl = self.portfolio.close_trade(trade, exit_price=1.15)
        assert pnl > 0
        assert trade.status == "CLOSED"
        assert trade.exit_price == 1.15

    def test_close_buy_trade_loss(self):
        trade = self._open("BUY", 1.10)
        pnl = self.portfolio.close_trade(trade, exit_price=1.05)
        assert pnl < 0

    def test_close_sell_trade_profit(self):
        trade = self._open("SELL", 1.10)
        pnl = self.portfolio.close_trade(trade, exit_price=1.05)
        assert pnl > 0

    def test_close_sell_trade_loss(self):
        trade = self._open("SELL", 1.10)
        pnl = self.portfolio.close_trade(trade, exit_price=1.15)
        assert pnl < 0

    def test_closed_trades_list(self):
        trade = self._open()
        self.portfolio.close_trade(trade, exit_price=1.12)
        assert trade in self.portfolio.get_closed_trades()
        assert trade not in self.portfolio.get_open_trades()

    def test_summary_empty(self):
        summary = self.portfolio.summary()
        assert summary["total_trades"] == 0
        assert summary["win_rate"] == 0.0

    def test_summary_with_trades(self):
        t1 = self._open("BUY", 1.10)
        self.portfolio.close_trade(t1, exit_price=1.15)  # win
        t2 = self._open("BUY", 1.10)
        self.portfolio.close_trade(t2, exit_price=1.05)  # loss

        summary = self.portfolio.summary()
        assert summary["total_trades"] == 2
        assert summary["winning_trades"] == 1
        assert summary["losing_trades"] == 1
        assert summary["win_rate"] == 0.5

    def test_to_dict(self):
        trade = self._open()
        d = trade.to_dict()
        assert "trade_id" in d
        assert "pair" in d
        assert "direction" in d

    def test_reset(self):
        self._open()
        self.portfolio.reset()
        assert len(self.portfolio.get_open_trades()) == 0
        assert len(self.portfolio.get_closed_trades()) == 0
