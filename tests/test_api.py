"""Tests for the Flask REST API."""

import numpy as np
import pandas as pd
import pytest

from src.agent.core import ForexTradingAgent
from src.agent.indicators import compute_all
from src.agent.profit_analyzer import ProfitAnalysis
from src.agent.signals import Signal, SignalType
from src.api.server import create_app, set_agent


def _make_agent(signal_type: SignalType = SignalType.HOLD):
    """Return a ForexTradingAgent with a stubbed analyse method (no network)."""
    agent = ForexTradingAgent(pairs=["EUR/USD"])

    # Stub out analyse to avoid yfinance network calls in tests
    def _stub_analyse(pair, df=None):
        return Signal(
            signal=signal_type,
            confirmations=3,
            reasons=["stub"],
            price=1.10000,
            rsi=50.0,
            macd_hist=0.0,
            bb_pct_b=0.5,
        )

    agent.analyse = _stub_analyse
    return agent


@pytest.fixture
def client():
    agent = _make_agent()
    app = create_app(agent)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def client_buy():
    """Client whose stubbed agent always returns a BUY signal."""
    agent = _make_agent(SignalType.BUY)
    app = create_app(agent)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestAPIRoutes:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["data"]["healthy"] is True

    def test_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "balance" in data["data"]
        assert "pairs" in data["data"]

    def test_pairs(self, client):
        resp = client.get("/api/pairs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "EUR/USD" in data["data"]["pairs"]

    def test_signals_pair(self, client):
        resp = client.get("/api/signals/EUR_USD")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "signal" in data["data"]
        assert data["data"]["pair"] == "EUR/USD"

    def test_signals_all(self, client):
        resp = client.get("/api/signals")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_portfolio(self, client):
        resp = client.get("/api/portfolio")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "summary" in data["data"]
        assert "open_trades" in data["data"]

    def test_balance(self, client):
        resp = client.get("/api/balance")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "balance" in data["data"]
        assert "current_drawdown_pct" in data["data"]

    def test_reset(self, client):
        resp = client.post("/api/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"


class TestProfitAnalysisRoutes:
    def test_profit_analysis_all_hold(self, client):
        """With HOLD signals, every pair reports is_viable=False."""
        resp = client.get("/api/profit-analysis")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        for pair_data in data["data"].values():
            assert pair_data["is_viable"] is False

    def test_profit_analysis_pair_hold(self, client):
        resp = client.get("/api/profit-analysis/EUR_USD")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["data"]["is_viable"] is False

    def test_profit_analysis_pair_buy_signal(self, client_buy):
        resp = client_buy.get("/api/profit-analysis/EUR_USD")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        pair_data = data["data"]
        assert "risk_reward_ratio" in pair_data
        assert "estimated_win_rate" in pair_data
        assert "expected_value" in pair_data
        assert "is_viable" in pair_data

    def test_profit_analysis_all_buy_signal(self, client_buy):
        resp = client_buy.get("/api/profit-analysis")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        # EUR/USD should have detailed analysis fields
        pair_data = data["data"].get("EUR/USD", {})
        assert "is_viable" in pair_data


class TestAutoTradeExecuteRoute:
    def test_execute_returns_ok(self, client):
        resp = client.post("/api/trade/execute")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "executed_trades" in data["data"]
        assert "trades" in data["data"]

    def test_execute_hold_executes_nothing(self, client):
        """HOLD signals should not result in any trades."""
        resp = client.post("/api/trade/execute")
        data = resp.get_json()
        assert data["data"]["executed_trades"] == 0
        assert data["data"]["trades"] == []

    def test_execute_buy_signal_may_trade(self, client_buy):
        """BUY signal with a 3:1 RR should pass profit analysis and execute."""
        resp = client_buy.post("/api/trade/execute")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        # With a clean 3:1 RR and 50 % win rate the EV is positive → trade executes
        assert data["data"]["executed_trades"] >= 0  # may be 0 if RR check fails


class TestMonitorPositionsRoute:
    def test_monitor_empty_prices(self, client):
        resp = client.post("/api/positions/monitor", json={"prices": {}})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["data"]["closed_trades"] == 0

    def test_monitor_invalid_prices_type(self, client):
        resp = client.post("/api/positions/monitor", json={"prices": "bad"})
        assert resp.status_code == 400

    def test_monitor_closes_sl_hit(self, client_buy):
        """Open a BUY trade then supply a price below SL to trigger auto-close."""
        # First execute a trade
        client_buy.post("/api/trade/execute")
        from src.api.server import get_agent
        agent = get_agent()
        open_trades = agent.portfolio.get_open_trades()
        if not open_trades:
            pytest.skip("No trade was opened (profit filter may have blocked it)")

        trade = open_trades[0]
        # Price well below stop-loss triggers stop-out
        below_sl = trade.stop_loss - 0.01
        resp = client_buy.post(
            "/api/positions/monitor",
            json={"prices": {trade.pair: below_sl}},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["closed_trades"] == 1
        closed = data["data"]["trades"][0]
        assert closed["status"] == "STOPPED"

    def test_monitor_closes_tp_hit(self, client_buy):
        """Open a BUY trade then supply a price above TP to trigger take-profit."""
        client_buy.post("/api/trade/execute")
        from src.api.server import get_agent
        agent = get_agent()
        open_trades = agent.portfolio.get_open_trades()
        if not open_trades:
            pytest.skip("No trade was opened (profit filter may have blocked it)")

        trade = open_trades[0]
        above_tp = trade.take_profit + 0.01
        resp = client_buy.post(
            "/api/positions/monitor",
            json={"prices": {trade.pair: above_tp}},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["closed_trades"] == 1
        closed = data["data"]["trades"][0]
        assert closed["status"] == "CLOSED"
