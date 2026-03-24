# Forex Trading Agent

A production-ready, modular forex trading agent with accurate buy and sell signals using a multi-indicator technical analysis consensus approach.

---

## Features

### Multi-Indicator Technical Analysis
| Indicator | Parameters |
|---|---|
| EMA Crossover | Fast 12 / Slow 26 periods |
| RSI | 14 period – overbought/oversold detection |
| MACD | 12 / 26 / 9 – momentum confirmation |
| Bollinger Bands | 20 period, 2 std dev – volatility context |

### Intelligent Signal Generation
A **consensus-based** approach requiring ≥ 3 indicator confirmations before generating a signal.

**BUY triggers:**
- Fast EMA crosses above Slow EMA
- RSI < 70 (not in overbought territory)
- MACD histogram crosses from negative → positive
- Price in lower Bollinger Band zone (pct_b < 0.5)

**SELL triggers:**
- Fast EMA crosses below Slow EMA
- RSI > 30 (not in oversold territory)
- MACD histogram crosses from positive → negative
- Price in upper Bollinger Band zone (pct_b > 0.5)

### Risk Management
- Dynamic position sizing (2% risk per trade by default)
- Stop-loss at 2% below/above entry
- Take-profit at 3:1 risk-reward ratio
- Maximum drawdown protection (10% by default)
- Full trade logging and P&L tracking

### Supported Currency Pairs
`EUR/USD` · `GBP/USD` · `USD/JPY` · `AUD/USD` · `USD/CAD`

---

## Project Structure

```
forex-trading-agent/
├── src/
│   ├── agent/
│   │   ├── core.py          # Main ForexTradingAgent class
│   │   ├── indicators.py    # EMA, RSI, MACD, Bollinger Bands, ATR
│   │   ├── signals.py       # Buy/sell signal generation
│   │   ├── risk_manager.py  # Position sizing & drawdown protection
│   │   └── portfolio.py     # Trade tracking and P&L analytics
│   ├── data/
│   │   ├── fetcher.py       # yfinance historical data retrieval
│   │   └── processor.py     # Data cleaning and validation
│   ├── api/
│   │   └── server.py        # Flask REST API
│   ├── backtester/
│   │   └── engine.py        # Historical backtesting framework
│   └── utils.py             # Shared utilities and helpers
├── tests/                   # Comprehensive pytest test suite
├── config/
│   └── default.json         # Default configuration
├── config.py                # Configuration management
├── main.py                  # CLI entry point
├── Dockerfile               # Container configuration
└── requirements.txt         # Python dependencies
```

---

## Quick Start

### Local Installation

```bash
pip install -r requirements.txt
```

### Analyse Currency Pairs

```bash
# Analyse all default pairs
python main.py analyse

# Analyse specific pairs
python main.py analyse EUR/USD GBP/USD
```

### Run Backtests

```bash
# Backtest all pairs over 1 year
python main.py backtest

# Backtest specific pairs
python main.py backtest EUR/USD --period 6mo
```

### Start the REST API

```bash
python main.py api
# or with custom host/port
python main.py api --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker build -t forex-agent .
docker run -p 5000:5000 forex-agent
```

---

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/status` | Agent status + current signals |
| GET | `/api/signals` | Signals for all pairs |
| GET | `/api/signals/<pair>` | Signal for a specific pair (e.g. `EUR_USD`) |
| GET | `/api/portfolio` | Trade history and P&L summary |
| GET | `/api/balance` | Balance and drawdown metrics |
| GET | `/api/pairs` | Configured currency pairs |
| POST | `/api/reset` | Reset agent state |

### Example Response – `/api/signals/EUR_USD`

```json
{
  "status": "ok",
  "data": {
    "pair": "EUR/USD",
    "signal": "BUY",
    "confirmations": 3,
    "reasons": [
      "EMA bullish crossover",
      "RSI=42.3 below overbought (70)",
      "MACD histogram bullish crossover"
    ],
    "price": 1.08432,
    "rsi": 42.3,
    "macd_hist": 0.00012,
    "bb_pct_b": 0.38,
    "timestamp": "2024-01-15T10:30:00+00:00"
  }
}
```

---

## Configuration

Edit `config/default.json` or set the `FOREX_CONFIG` environment variable to point to a custom JSON override file.

```json
{
  "trading": {
    "initial_balance": 10000.0,
    "risk_per_trade": 0.02,
    "max_drawdown": 0.10,
    "take_profit_ratio": 3.0,
    "stop_loss_pct": 0.02
  },
  "indicators": {
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0
  },
  "signals": {
    "min_confirmations": 3,
    "rsi_overbought": 70,
    "rsi_oversold": 30
  }
}
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice. Forex trading carries significant risk. Always use proper risk management and consult a qualified financial advisor before trading with real funds.
