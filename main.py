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

    # ---- analyse ----
    analyse_parser = subparsers.add_parser("analyse", help="Analyse currency pairs and print signals.")
    analyse_parser.add_argument(
        "pairs",
        nargs="*",
        default=None,
        help="Pairs to analyse (e.g. EUR/USD GBP/USD). Defaults to all configured pairs.",
    )

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


def cmd_api(args) -> None:
    from src.api.server import run_server

    logger.info("Starting Forex Trading Agent REST API …")
    run_server(host=args.host, port=args.port, debug=args.debug)


def cmd_analyse(args) -> None:
    from src.agent.core import ForexTradingAgent

    pairs = args.pairs or None
    agent = ForexTradingAgent(pairs=pairs)
    signals = agent.analyse_all()
    print("\n=== Forex Signal Summary ===")
    for pair, sig in signals.items():
        print(f"  {pair:10s}  {sig.signal.value:4s}  confirmations={sig.confirmations}  price={sig.price:.5f}")
        for reason in sig.reasons:
            print(f"             › {reason}")
    print()


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
