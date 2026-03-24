"""Utility helpers shared across the Forex Trading Agent."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def pct_change(new_value: float, old_value: float) -> float:
    """Return percentage change from *old_value* to *new_value*."""
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / abs(old_value)


def format_currency(amount: float, symbol: str = "$") -> str:
    return f"{symbol}{amount:,.2f}"


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """Return True if *df* has the required OHLCV columns and no empty rows."""
    required = {"open", "high", "low", "close", "volume"}
    cols = {c.lower() for c in df.columns}
    if not required.issubset(cols):
        return False
    if df.empty:
        return False
    return True


def timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def round_price(price: float, decimals: int = 5) -> float:
    return round(price, decimals)


def trades_to_dataframe(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of trade dicts to a DataFrame."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)
