"""Data cleaning and validation utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils import get_logger, validate_ohlcv

logger = get_logger(__name__)


class DataProcessor:
    """Clean, validate and enrich raw OHLCV DataFrames."""

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of *df*."""
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("Missing columns: %s", missing)
            return pd.DataFrame()

        # Drop rows where any OHLC value is NaN or non-positive
        df = df.dropna(subset=required)
        df = df[(df[required] > 0).all(axis=1)]

        # Ensure volume column exists
        if "volume" not in df.columns:
            df["volume"] = 0.0

        # Sort chronologically
        df = df.sort_index()

        # Remove duplicate index entries, keeping last
        df = df[~df.index.duplicated(keep="last")]

        return df

    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``returns`` column with simple daily returns."""
        df = df.copy()
        df["returns"] = df["close"].pct_change()
        return df

    @staticmethod
    def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``log_returns`` column."""
        df = df.copy()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        return df

    @staticmethod
    def validate(df: pd.DataFrame) -> bool:
        """Return True if *df* passes basic sanity checks."""
        return validate_ohlcv(df)

    @staticmethod
    def normalize(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        """Add a ``normalized`` column = column / first non-NaN value."""
        df = df.copy()
        first_valid = df[column].dropna().iloc[0]
        df["normalized"] = df[column] / first_valid
        return df
