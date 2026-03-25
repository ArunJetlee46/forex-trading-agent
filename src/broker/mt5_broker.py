"""MetaTrader 5 broker — executes trades via a locally running MT5 terminal."""

from __future__ import annotations

from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False

# MT5 order type constants (mirror mt5.ORDER_TYPE_*)
_ORDER_BUY = 0
_ORDER_SELL = 1

# MT5 trade action constants
_TRADE_ACTION_DEAL = 1


class MT5Broker:
    """Send orders to MetaTrader 5 via the Python MT5 API.

    This class assumes that the MT5 terminal has already been initialised
    (e.g. by :class:`~src.data.mt5_fetcher.MT5DataFetcher`).  If you use
    *MT5DataFetcher* and *MT5Broker* together you only need to call
    ``connect()`` once on the fetcher.

    Parameters
    ----------
    deviation:
        Maximum price deviation from the requested price (in points).
    magic:
        Magic number to tag orders placed by this agent.
    """

    def __init__(self, deviation: int = 20, magic: int = 234000) -> None:
        if not _MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 Python package is not installed or not available on "
                "this platform. Install it on Windows with: pip install MetaTrader5"
            )
        self._deviation = deviation
        self._magic = magic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "forex-agent",
    ) -> Optional[int]:
        """Send a market order to MT5.

        Parameters
        ----------
        symbol:
            MT5 symbol name (e.g. ``"EURUSD"``).
        direction:
            ``"BUY"`` or ``"SELL"``.
        volume:
            Trade size in lots.
        price:
            Requested price.  When ``None`` the current ask/bid is used.
        stop_loss:
            Stop-loss price level.
        take_profit:
            Take-profit price level.
        comment:
            Order comment string shown in MT5 history.

        Returns
        -------
        int or None
            The MT5 order ticket number on success, ``None`` on failure.
        """
        order_type = _ORDER_BUY if direction.upper() == "BUY" else _ORDER_SELL

        # Resolve price from tick if not supplied
        if price is None:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error("Cannot resolve tick price for %s", symbol)
                return None
            price = tick.ask if order_type == _ORDER_BUY else tick.bid

        request = {
            "action": _TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": comment,
            "type_time": 0,   # ORDER_TIME_GTC
            "type_filling": 2,  # ORDER_FILLING_IOC
        }

        if stop_loss is not None:
            request["sl"] = stop_loss
        if take_profit is not None:
            request["tp"] = take_profit

        result = mt5.order_send(request)

        if result is None or result.retcode != 10009:  # TRADE_RETCODE_DONE
            err = mt5.last_error() if result is None else result.retcode
            logger.error(
                "MT5 order_send failed for %s %s %.2f lots: retcode=%s",
                direction, symbol, volume, err,
            )
            return None

        logger.info(
            "MT5 order placed: ticket=%s  %s %s  %.2f lots @ %.5f",
            result.order, direction, symbol, volume, price,
        )
        return result.order

    def close_position(self, ticket: int, volume: Optional[float] = None) -> bool:
        """Close an open MT5 position by ticket number.

        Parameters
        ----------
        ticket:
            MT5 position ticket.
        volume:
            Lots to close.  Closes the full position when ``None``.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.
        """
        position = None
        for pos in (mt5.positions_get(ticket=ticket) or []):
            position = pos
            break

        if position is None:
            logger.error("MT5 position with ticket %s not found.", ticket)
            return False

        close_volume = volume if volume is not None else position.volume
        close_type = _ORDER_SELL if position.type == _ORDER_BUY else _ORDER_BUY
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            logger.error("Cannot resolve tick for %s", position.symbol)
            return False

        close_price = tick.bid if close_type == _ORDER_SELL else tick.ask

        request = {
            "action": _TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": float(close_volume),
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": "forex-agent-close",
            "type_time": 0,
            "type_filling": 2,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != 10009:
            err = mt5.last_error() if result is None else result.retcode
            logger.error("MT5 close_position failed for ticket %s: retcode=%s", ticket, err)
            return False

        logger.info("MT5 position closed: ticket=%s", ticket)
        return True

    def account_info(self) -> Optional[dict]:
        """Return MT5 account information as a plain dict."""
        info = mt5.account_info()
        if info is None:
            return None
        return info._asdict()
