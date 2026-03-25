"""Historical market data retrieval using yfinance."""

from __future__ import annotations

import time
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from src.utils import get_logger, validate_ohlcv

logger = get_logger(__name__)


class DataFetcher:
    """Download OHLCV data for forex pairs via yfinance with simple caching."""

    # Map human-friendly pair names to yfinance tickers
    PAIR_MAP: Dict[str, str] = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "XAU/EUR": "XAUEUR=X",
    }

    def __init__(self, cache_ttl: int = 300) -> None:
        self._cache: Dict[str, tuple] = {}  # ticker -> (timestamp, df)
        self._cache_ttl = cache_ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        pair: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return a OHLCV DataFrame for *pair*.

        *pair* can be a human-friendly name (``"EUR/USD"``) or a yfinance
        ticker symbol (``"EURUSD=X"``).
        """
        ticker = self.PAIR_MAP.get(pair, pair)
        cache_key = f"{ticker}|{period}|{interval}"

        if not force_refresh and self._is_cached(cache_key):
            logger.debug("Cache hit for %s", cache_key)
            return self._cache[cache_key][1]

        logger.info("Fetching %s  period=%s  interval=%s", ticker, period, interval)
        df = self._download(ticker, period, interval)

        if df.empty:
            logger.warning("No data returned for %s", ticker)
            return df

        self._cache[cache_key] = (time.time(), df)
        return df

    def fetch_multi(
        self,
        pairs: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Return a dict of ``{pair: DataFrame}`` for each pair in *pairs*."""
        return {pair: self.fetch(pair, period, interval) for pair in pairs}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        try:
            raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        except Exception as exc:
            logger.error("yfinance download failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        # Flatten MultiIndex columns produced by yfinance ≥ 0.2
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [c.lower() for c in raw.columns]
        raw.index.name = "date"
        raw = raw.dropna(subset=["close"])
        return raw

    def _is_cached(self, key: str) -> bool:
        if key not in self._cache:
            return False
        age = time.time() - self._cache[key][0]
        return age < self._cache_ttl

    def clear_cache(self) -> None:
        self._cache.clear()
