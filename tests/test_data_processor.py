"""Tests for data processing utilities."""

import numpy as np
import pandas as pd
import pytest

from src.data.processor import DataProcessor


def _sample_df(n: int = 20) -> pd.DataFrame:
    prices = np.linspace(1.1, 1.2, n)
    return pd.DataFrame(
        {
            "open": prices - 0.001,
            "high": prices + 0.002,
            "low": prices - 0.002,
            "close": prices,
            "volume": np.ones(n) * 100,
        }
    )


class TestDataProcessor:
    def test_clean_returns_non_empty(self):
        df = _sample_df()
        result = DataProcessor.clean(df)
        assert not result.empty

    def test_clean_lowercase_columns(self):
        df = _sample_df()
        df.columns = [c.upper() for c in df.columns]
        result = DataProcessor.clean(df)
        assert all(c == c.lower() for c in result.columns)

    def test_clean_removes_nan_rows(self):
        df = _sample_df()
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        result = DataProcessor.clean(df)
        assert result["close"].notna().all()

    def test_clean_removes_non_positive(self):
        df = _sample_df()
        df.iloc[3, df.columns.get_loc("close")] = 0.0
        result = DataProcessor.clean(df)
        assert (result["close"] > 0).all()

    def test_clean_empty_input(self):
        result = DataProcessor.clean(pd.DataFrame())
        assert result.empty

    def test_clean_missing_required_column(self):
        df = _sample_df().drop(columns=["close"])
        result = DataProcessor.clean(df)
        assert result.empty

    def test_add_returns(self):
        df = _sample_df()
        result = DataProcessor.add_returns(df)
        assert "returns" in result.columns
        # First element should be NaN
        assert pd.isna(result["returns"].iloc[0])

    def test_add_log_returns(self):
        df = _sample_df()
        result = DataProcessor.add_log_returns(df)
        assert "log_returns" in result.columns

    def test_validate_valid_df(self):
        df = _sample_df()
        assert DataProcessor.validate(df)

    def test_validate_missing_column(self):
        df = _sample_df().drop(columns=["volume"])
        assert not DataProcessor.validate(df)

    def test_normalize(self):
        df = _sample_df()
        result = DataProcessor.normalize(df, "close")
        assert "normalized" in result.columns
        assert abs(result["normalized"].iloc[0] - 1.0) < 1e-9
