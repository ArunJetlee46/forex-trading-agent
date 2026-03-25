"""Flask REST API for monitoring and controlling the Forex Trading Agent."""

from __future__ import annotations

from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import config
from src.agent.core import ForexTradingAgent
from src.utils import get_logger, timestamp_now

logger = get_logger(__name__)

app = Flask(__name__)
CORS(app)

# Shared agent instance (initialised lazily so tests can override it)
_agent: ForexTradingAgent | None = None


def get_agent() -> ForexTradingAgent:
    global _agent
    if _agent is None:
        _agent = ForexTradingAgent()
    return _agent


def set_agent(agent: ForexTradingAgent) -> None:
    global _agent
    _agent = agent


def _ok(data: Any, status: int = 200):
    return jsonify({"status": "ok", "data": data}), status


def _error(message: str, status: int = 400):
    return jsonify({"status": "error", "message": message}), status


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@app.get("/api/status")
def api_status():
    """Return agent status and current signals."""
    return _ok(get_agent().status())


@app.get("/api/signals/<pair>")
def api_signals_pair(pair: str):
    """Return the latest signal for a given currency pair."""
    agent = get_agent()
    # Normalise URL-encoded pair (e.g. "EUR_USD" → "EUR/USD")
    normalised = pair.replace("_", "/").upper()
    signal = agent.analyse(normalised)
    return _ok(
        {
            "pair": normalised,
            "signal": signal.signal.value,
            "confirmations": signal.confirmations,
            "reasons": signal.reasons,
            "price": signal.price,
            "rsi": signal.rsi,
            "macd_hist": signal.macd_hist,
            "bb_pct_b": signal.bb_pct_b,
            "timestamp": timestamp_now(),
        }
    )


@app.get("/api/signals")
def api_signals_all():
    """Return signals for all configured pairs."""
    agent = get_agent()
    signals = agent.analyse_all()
    result = {}
    for pair, sig in signals.items():
        result[pair] = {
            "signal": sig.signal.value,
            "confirmations": sig.confirmations,
            "reasons": sig.reasons,
            "price": sig.price,
        }
    return _ok(result)


@app.get("/api/portfolio")
def api_portfolio():
    """Return portfolio summary and trade history."""
    agent = get_agent()
    return _ok(
        {
            "summary": agent.portfolio.summary(),
            "open_trades": [t.to_dict() for t in agent.portfolio.get_open_trades()],
            "closed_trades": [t.to_dict() for t in agent.portfolio.get_closed_trades()],
        }
    )


@app.get("/api/balance")
def api_balance():
    """Return current balance and risk metrics."""
    rm = get_agent().risk_manager
    return _ok(
        {
            "balance": round(rm.balance, 2),
            "peak_balance": round(rm.peak_balance, 2),
            "current_drawdown_pct": round(rm.current_drawdown() * 100, 2),
            "drawdown_breached": rm.is_drawdown_breached(),
        }
    )


@app.post("/api/reset")
def api_reset():
    """Reset the agent (portfolio, balance, signals)."""
    get_agent().reset()
    return _ok({"message": "Agent reset successfully."})


@app.get("/api/pairs")
def api_pairs():
    """Return the list of configured currency pairs."""
    return _ok({"pairs": get_agent().pairs})


@app.get("/api/health")
def api_health():
    return _ok({"healthy": True, "timestamp": timestamp_now()})


@app.post("/api/trade/execute")
def api_trade_execute():
    """Analyse all pairs and auto-execute signals that pass profit analysis."""
    agent = get_agent()
    try:
        trades = agent.auto_execute_signals()
        return _ok(
            {
                "executed_trades": len(trades),
                "trades": [t.to_dict() for t in trades],
            }
        )
    except Exception as exc:
        logger.exception("Error during auto-execute")
        return _error(str(exc), 500)


@app.get("/api/profit-analysis")
def api_profit_analysis_all():
    """Return profit analysis for every configured pair.

    Signals are freshly generated.  Pairs with a HOLD signal are included
    with ``is_viable=false``.
    """
    agent = get_agent()
    signals = agent.analyse_all()
    result = {}
    for pair, sig in signals.items():
        analysis = agent.analyse_profit(pair, sig)
        if analysis is not None:
            result[pair] = analysis.to_dict()
        else:
            result[pair] = {
                "pair": pair,
                "signal": sig.signal.value,
                "is_viable": False,
                "reasons": ["HOLD – no actionable signal"],
            }
    return _ok(result)


@app.get("/api/profit-analysis/<pair>")
def api_profit_analysis_pair(pair: str):
    """Return profit analysis for a specific currency pair."""
    agent = get_agent()
    normalised = pair.replace("_", "/").upper()
    sig = agent.analyse(normalised)
    analysis = agent.analyse_profit(normalised, sig)
    if analysis is None:
        return _ok(
            {
                "pair": normalised,
                "signal": sig.signal.value,
                "is_viable": False,
                "reasons": ["HOLD – no actionable signal"],
            }
        )
    data = analysis.to_dict()
    data["signal"] = sig.signal.value
    return _ok(data)


@app.post("/api/positions/monitor")
def api_monitor_positions():
    """Check open positions against supplied prices and auto-close SL/TP hits.

    Request body (JSON)::

        {"prices": {"EUR/USD": 1.0850, "GBP/USD": 1.2500}}
    """
    agent = get_agent()
    body = request.get_json(silent=True) or {}
    current_prices = body.get("prices", {})
    if not isinstance(current_prices, dict):
        return _error("'prices' must be a JSON object mapping pair to price.", 400)

    closed_trades = agent.monitor_positions(current_prices)
    return _ok(
        {
            "closed_trades": len(closed_trades),
            "trades": [t.to_dict() for t in closed_trades],
        }
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def create_app(agent: ForexTradingAgent | None = None) -> Flask:
    if agent is not None:
        set_agent(agent)
    return app


def run_server(
    host: str | None = None,
    port: int | None = None,
    debug: bool | None = None,
    broker=None,
    mt5_fetcher=None,
) -> None:
    api_cfg = config.api
    if broker is not None or mt5_fetcher is not None:
        agent = get_agent()
        if broker is not None:
            agent.broker = broker
        if mt5_fetcher is not None:
            agent.fetcher = mt5_fetcher
    app.run(
        host=host or api_cfg["host"],
        port=port or api_cfg["port"],
        debug=debug if debug is not None else api_cfg["debug"],
    )
