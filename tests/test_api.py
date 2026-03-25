"""Tests for the Flask REST API."""

import numpy as np
import pandas as pd
import pytest

from src.agent.core import ForexTradingAgent
from src.agent.indicators import compute_all
from src.agent.signals import Signal, SignalType
from src.api.server import create_app, set_agent


def _make_agent():
    """Return a ForexTradingAgent with a stubbed analyse method (no network)."""
    agent = ForexTradingAgent(pairs=["EUR/USD"])

    # Stub out analyse to avoid yfinance network calls in tests
    def _stub_analyse(pair, df=None):
        return Signal(
            signal=SignalType.HOLD,
            confirmations=1,
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
