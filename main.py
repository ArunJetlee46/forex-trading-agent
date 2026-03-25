#!/usr/bin/env python3
"""Entry point for the Forex Trading Agent."""

from __future__ import annotations

import argparse
import sys

from config import config
from src.utils import get_logger

logger = get_logger("main")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Forex Trading Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- api ----
    api_parser = subparsers.add_parser("api", help="Start the REST API server.")
    api_parser.add_argument("--host", default=None, help="Host to bind to.")
    api_parser.add_argument("--port", type=int, default=None, help="Port to listen on.")
    api_parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    api_parser.add_argument(
        "--broker",
        choices=["mt5"],
        default=None,
        help="Connect to a live broker (currently: mt5).",
    )
    api_parser.add_argument("--mt5-login", type=int, default=None, help="MT5 account login number.")
    api_parser.add_argument("--mt5-password", default=None, help="MT5 account password.")
    api_parser.add_argument("--mt5-server", default=None, help="MT5 broker server name.")

    # ---- analyse ----
    analyse_parser = subparsers.add_parser("analyse", help="Analyse currency pairs and print signals.")
    analyse_parser.add_argument(
        "pairs",
        nargs="*",
        default=None,
        help="Pairs to analyse (e.g. EUR/USD GBP/USD). Defaults to all configured pairs.",
    )
    analyse_parser.add_argument(
        "--broker",
        choices=["mt5"],
        default=None,
        help="Connect to a live broker for data and execution (currently: mt5).",
    )
    analyse_parser.add_argument("--mt5-login", type=int, default=None, help="MT5 account login number.")
    analyse_parser.add_argument("--mt5-password", default=None, help="MT5 account password.")
    analyse_parser.add_argument("--mt5-server", default=None, help="MT5 broker server name.")

    # ---- backtest ----
    bt_parser = subparsers.add_parser("backtest", help="Run backtests on historical data.")
    bt_parser.add_argument(
        "pairs",
        nargs="*",
        default=None,
        help="Pairs to backtest. Defaults to all configured pairs.",
    )
    bt_parser.add_argument("--period", default="1y", help="History period (e.g. 1y, 6mo).")

    return parser.parse_args(argv)


def _build_mt5_broker(args):
    """Initialise the MT5 fetcher and broker from CLI args.

    Returns a ``(MT5Broker, MT5DataFetcher)`` tuple when ``--broker mt5`` is
    requested, or ``None`` when it is not.
    Raises ``RuntimeError`` if the MT5 package is unavailable or the connection fails.
    """
    if getattr(args, "broker", None) != "mt5":
        return None

    from src.data.mt5_fetcher import MT5DataFetcher
    from src.broker.mt5_broker import MT5Broker

    fetcher = MT5DataFetcher(
        login=getattr(args, "mt5_login", None),
        password=getattr(args, "mt5_password", None),
        server=getattr(args, "mt5_server", None),
    )
    if not fetcher.connect():
        raise RuntimeError("Failed to connect to MetaTrader 5 terminal.")
    broker = MT5Broker()
    return broker, fetcher


def cmd_api(args) -> None:
    from src.api.server import run_server

    logger.info("Starting Forex Trading Agent REST API …")

    broker = None
    mt5_fetcher = None
    if getattr(args, "broker", None) == "mt5":
        result = _build_mt5_broker(args)
        if result is not None:
            broker, mt5_fetcher = result

    run_server(host=args.host, port=args.port, debug=args.debug, broker=broker, mt5_fetcher=mt5_fetcher)


def cmd_analyse(args) -> None:
    from src.agent.core import ForexTradingAgent

    broker = None
    mt5_fetcher = None
    if getattr(args, "broker", None) == "mt5":
        result = _build_mt5_broker(args)
        if result is not None:
            broker, mt5_fetcher = result

    pairs = args.pairs or None
    agent = ForexTradingAgent(pairs=pairs, broker=broker)

    if mt5_fetcher is not None:
        agent.fetcher = mt5_fetcher

    signals = agent.analyse_all()
    print("\n=== Forex Signal Summary ===")
    for pair, sig in signals.items():
        print(f"  {pair:10s}  {sig.signal.value:4s}  confirmations={sig.confirmations}  price={sig.price:.5f}")
        for reason in sig.reasons:
            print(f"             › {reason}")
    print()

    if mt5_fetcher is not None:
        mt5_fetcher.disconnect()


def cmd_backtest(args) -> None:
    from src.agent.core import ForexTradingAgent
    from src.backtester.engine import BacktestEngine
    from src.data.fetcher import DataFetcher
    from src.data.processor import DataProcessor

    pairs = args.pairs or ForexTradingAgent.DEFAULT_PAIRS
    fetcher = DataFetcher()
    processor = DataProcessor()
    engine = BacktestEngine()

    print(f"\n=== Backtest  period={args.period} ===\n")
    for pair in pairs:
        raw = fetcher.fetch(pair, period=args.period)
        df = processor.clean(raw)
        if df.empty:
            print(f"  {pair}: No data available.\n")
            continue
        result = engine.run(df, pair=pair)
        print(result)
        print()


def main(argv=None) -> int:
    args = parse_args(argv)
    dispatch = {
        "api": cmd_api,
        "analyse": cmd_analyse,
        "backtest": cmd_backtest,
    }
    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
