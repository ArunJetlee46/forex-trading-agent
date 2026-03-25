"""MetaTrader 5 data fetcher — retrieves OHLCV bars from a running MT5 terminal."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from src.utils import get_logger, validate_ohlcv

logger = get_logger(__name__)

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False


# Map human-friendly pair names to MT5 symbol names
MT5_SYMBOL_MAP: Dict[str, str] = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
    "AUD/USD": "AUDUSD",
    "USD/CAD": "USDCAD",
    "XAU/EUR": "XAUEUR",
}

# Map interval strings to MT5 timeframe constants (populated at runtime)
_TIMEFRAME_MAP: Dict[str, int] = {
    "1m":  1,    # TIMEFRAME_M1
    "5m":  5,    # TIMEFRAME_M5
    "15m": 15,   # TIMEFRAME_M15
    "30m": 30,   # TIMEFRAME_M30
    "1h":  16385,  # TIMEFRAME_H1
    "4h":  16388,  # TIMEFRAME_H4
    "1d":  16408,  # TIMEFRAME_D1
    "1wk": 32769,  # TIMEFRAME_W1
    "1mo": 49153,  # TIMEFRAME_MN1
}


class MT5DataFetcher:
    """Fetch OHLCV data from a locally running MetaTrader 5 terminal.

    Parameters
    ----------
    login:
        MT5 account login number.  Can be ``None`` when connecting to an
        already-authenticated terminal session.
    password:
        MT5 account password.
    server:
        Broker server name (e.g. ``"MetaQuotes-Demo"``).
    path:
        Optional path to the MetaTrader 5 terminal executable.
    cache_ttl:
        Cache time-to-live in seconds (default 300).
    """

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        cache_ttl: int = 300,
    ) -> None:
        if not _MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 Python package is not installed or not available on "
                "this platform. Install it on Windows with: pip install MetaTrader5"
            )
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._connected = False
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = cache_ttl

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise the connection to the MT5 terminal.

        Returns ``True`` on success, ``False`` otherwise.
        """
        kwargs: Dict = {}
        if self._path:
            kwargs["path"] = self._path
        if self._login:
            kwargs["login"] = self._login
        if self._password:
            kwargs["password"] = self._password
        if self._server:
            kwargs["server"] = self._server

        if not mt5.initialize(**kwargs):
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            return False

        info = mt5.account_info()
        if info is None:
            logger.error("MT5 account_info() returned None: %s", mt5.last_error())
            return False

        logger.info(
            "Connected to MT5: login=%s  server=%s  balance=%.2f",
            info.login,
            info.server,
            info.balance,
        )
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 connection."""
        if _MT5_AVAILABLE:
            mt5.shutdown()
        self._connected = False
        logger.info("MT5 connection closed.")

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
        """Return an OHLCV DataFrame for *pair* fetched from MT5.

        Parameters
        ----------
        pair:
            Human-friendly pair name (e.g. ``"EUR/USD"``) or an MT5 symbol
            (e.g. ``"EURUSD"``).
        period:
            Look-back period string matching yfinance conventions
            (``"1y"``, ``"6mo"``, ``"3mo"``, ``"1mo"``).
        interval:
            Bar interval (``"1d"``, ``"1h"``, ``"15m"`` …).
        force_refresh:
            Bypass the local cache when ``True``.
        """
        if not self._connected:
            raise RuntimeError("Not connected to MT5. Call connect() first.")

        symbol = MT5_SYMBOL_MAP.get(pair, pair)
        cache_key = f"{symbol}|{period}|{interval}"

        if not force_refresh and self._is_cached(cache_key):
            logger.debug("Cache hit for %s", cache_key)
            return self._cache[cache_key][1]

        logger.info("Fetching MT5 data  symbol=%s  period=%s  interval=%s", symbol, period, interval)
        df = self._download(symbol, period, interval)

        if df.empty:
            logger.warning("No MT5 data returned for %s", symbol)
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

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        timeframe = _TIMEFRAME_MAP.get(interval)
        if timeframe is None:
            logger.error("Unsupported MT5 interval: %s", interval)
            return pd.DataFrame()

        bars_needed = self._period_to_bars(period, interval)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars_needed)

        if rates is None or len(rates) == 0:
            logger.error("MT5 copy_rates_from_pos failed for %s: %s", symbol, mt5.last_error())
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("date")
        df = df.rename(columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]
        df = df.dropna(subset=["close"])

        if not validate_ohlcv(df):
            logger.warning("MT5 data for %s failed OHLCV validation", symbol)
            return pd.DataFrame()

        return df

    @staticmethod
    def _period_to_bars(period: str, interval: str) -> int:
        """Convert a period string to an approximate bar count."""
        _bars_per_day: Dict[str, float] = {
            "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
            "1h": 24, "4h": 6, "1d": 1, "1wk": 1 / 5, "1mo": 1 / 22,
        }
        _period_days: Dict[str, int] = {
            "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
        }
        days = _period_days.get(period, 365)
        bars_per_day = _bars_per_day.get(interval, 1)
        return max(1, int(days * bars_per_day))

    def _is_cached(self, key: str) -> bool:
        if key not in self._cache:
            return False
        age = time.time() - self._cache[key][0]
        return age < self._cache_ttl
