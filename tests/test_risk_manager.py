"""Tests for the risk management module."""

import pytest

from src.agent.risk_manager import PositionSpec, RiskManager


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(
            initial_balance=10_000,
            risk_per_trade=0.02,
            stop_loss_pct=0.02,
            take_profit_ratio=3.0,
            max_drawdown=0.10,
        )

    def test_buy_position_spec(self):
        spec = self.rm.calculate_position("EUR/USD", "BUY", entry_price=1.10000)
        assert spec is not None
        assert spec.direction == "BUY"
        assert spec.stop_loss < spec.entry_price
        assert spec.take_profit > spec.entry_price

    def test_sell_position_spec(self):
        spec = self.rm.calculate_position("EUR/USD", "SELL", entry_price=1.10000)
        assert spec is not None
        assert spec.direction == "SELL"
        assert spec.stop_loss > spec.entry_price
        assert spec.take_profit < spec.entry_price

    def test_risk_amount(self):
        spec = self.rm.calculate_position("EUR/USD", "BUY", entry_price=1.10000)
        expected_risk = 10_000 * 0.02
        assert abs(spec.risk_amount - expected_risk) < 1e-6

    def test_units_calculation(self):
        entry = 1.10000
        spec = self.rm.calculate_position("EUR/USD", "BUY", entry_price=entry)
        stop_distance = entry * 0.02
        expected_units = (10_000 * 0.02) / stop_distance
        assert abs(spec.units - expected_units) < 1e-6

    def test_take_profit_ratio(self):
        entry = 1.10000
        spec = self.rm.calculate_position("EUR/USD", "BUY", entry_price=entry)
        stop_dist = entry - spec.stop_loss
        tp_dist = spec.take_profit - entry
        assert abs(tp_dist / stop_dist - 3.0) < 0.001

    def test_invalid_entry_price(self):
        spec = self.rm.calculate_position("EUR/USD", "BUY", entry_price=0.0)
        assert spec is None

    def test_invalid_direction(self):
        spec = self.rm.calculate_position("EUR/USD", "HOLD", entry_price=1.10)
        assert spec is None

    def test_drawdown_halts_trading(self):
        # Simulate a 10%+ loss to breach max drawdown
        self.rm.update_balance(-1_500)  # 15% loss on 10k
        assert self.rm.is_drawdown_breached()
        spec = self.rm.calculate_position("EUR/USD", "BUY", entry_price=1.10)
        assert spec is None

    def test_update_balance(self):
        self.rm.update_balance(500)
        assert self.rm.balance == 10_500
        assert self.rm.peak_balance == 10_500

    def test_current_drawdown(self):
        # First a gain to set peak
        self.rm.update_balance(1_000)  # balance=11k, peak=11k
        # Then a loss
        self.rm.update_balance(-2_000)  # balance=9k
        expected_dd = (11_000 - 9_000) / 11_000
        assert abs(self.rm.current_drawdown() - expected_dd) < 1e-6

    def test_reset(self):
        self.rm.update_balance(-500)
        self.rm.reset()
        assert self.rm.balance == 10_000
        assert self.rm.peak_balance == 10_000
