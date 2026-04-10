"""Market data handler using Alpaca API."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from alerts import get_logger
from config import AlpacaConfig

logger = get_logger("data_handler")


def _retry(fn, retries: int = 3, backoff: float = 1.0):
    """Call *fn* up to *retries* times with exponential back-off."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait = backoff * (2 ** attempt)
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs…",
                attempt + 1,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"All {retries} attempts failed") from last_exc


class DataHandler:
    """Fetches historical and real-time market data from Alpaca."""

    def __init__(self, config: AlpacaConfig) -> None:
        self.config = config
        self._trading_client = None
        self._data_client = None
        self._init_clients()

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _init_clients(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self._trading_client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.is_paper,
            )
            self._data_client = StockHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
            )
            logger.info("Alpaca clients initialised (paper=%s)", self.config.is_paper)
        except Exception as exc:
            logger.error("Failed to initialise Alpaca clients: %s", exc)
            raise

    def reinit_clients(self) -> None:
        """Re-create trading and data clients (called by the watchdog on connection failure)."""
        logger.info("Re-initialising Alpaca clients...")
        self._init_clients()

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def get_account(self):
        return _retry(self._trading_client.get_account)

    def get_portfolio_value(self) -> float:
        account = self.get_account()
        return float(account.portfolio_value)

    def get_cash(self) -> float:
        account = self.get_account()
        return float(account.cash)

    def get_positions(self) -> List:
        return _retry(self._trading_client.get_all_positions)

    def get_open_orders(self) -> List:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return _retry(lambda: self._trading_client.get_orders(req))

    # ------------------------------------------------------------------
    # Historical bars
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Return OHLCV DataFrames keyed by symbol."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        if end is None:
            end = datetime.now(tz=timezone.utc)
        if start is None:
            start = end - timedelta(days=365)

        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )

        def _fetch():
            return self._data_client.get_stock_bars(req)

        bars = _retry(_fetch)

        result: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = bars[symbol].df.copy()
                df.index = pd.to_datetime(df.index, utc=True)
                result[symbol] = df
            except (KeyError, AttributeError):
                logger.warning("No bar data for %s", symbol)
        return result

    def get_single_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        result = self.get_bars([symbol], timeframe, start, end, limit)
        return result.get(symbol, pd.DataFrame())

    # ------------------------------------------------------------------
    # Latest quotes / snapshots
    # ------------------------------------------------------------------

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return the latest trade price for *symbol*."""
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            req = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            trades = _retry(lambda: self._data_client.get_stock_latest_trade(req))
            return float(trades[symbol].price)
        except Exception as exc:
            logger.error("Could not fetch latest price for %s: %s", symbol, exc)
            return None

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Return latest prices for multiple symbols."""
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            req = StockLatestTradeRequest(symbol_or_symbols=symbols)
            trades = _retry(lambda: self._data_client.get_stock_latest_trade(req))
            return {sym: float(trades[sym].price) for sym in symbols if sym in trades}
        except Exception as exc:
            logger.error("Could not fetch latest prices: %s", exc)
            return {}

    def get_snapshots(self, symbols: List[str]) -> Dict[str, Dict]:
        """Return snapshot data (bid, ask, last price) for *symbols*.

        Each value is a dict with keys ``bid``, ``ask``, ``last_price``, and
        ``bid_ask_spread_pct``.  Symbols for which snapshot data is unavailable
        are omitted from the result.
        """
        try:
            from alpaca.data.requests import StockSnapshotRequest

            req = StockSnapshotRequest(symbol_or_symbols=symbols)
            raw = _retry(lambda: self._data_client.get_stock_snapshot(req))
            result: Dict[str, Dict] = {}
            for sym in symbols:
                snap = raw.get(sym)
                if snap is None:
                    continue
                bid = float(snap.latest_quote.bid_price) if snap.latest_quote else 0.0
                ask = float(snap.latest_quote.ask_price) if snap.latest_quote else 0.0
                last = float(snap.latest_trade.price) if snap.latest_trade else 0.0
                mid = (bid + ask) / 2 if bid and ask else last
                spread_pct = ((ask - bid) / mid * 100) if mid > 0 and ask > bid else 0.0
                result[sym] = {
                    "bid": bid,
                    "ask": ask,
                    "last_price": last,
                    "bid_ask_spread_pct": round(spread_pct, 4),
                }
            return result
        except Exception as exc:
            logger.error("Could not fetch snapshots: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ):
        """Submit a trading order.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares (positive).
            side: ``"buy"`` or ``"sell"``.
            order_type: ``"market"``, ``"limit"``, or ``"stop"``.
            limit_price: Required for limit orders.
            stop_price: Required for stop orders.
            time_in_force: ``"day"``, ``"gtc"``, etc.
        """
        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
            StopOrderRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif_enum = TimeInForce(time_in_force.lower()) if time_in_force else TimeInForce.DAY

        if order_type == "limit" and limit_price is not None:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif_enum,
                limit_price=round(limit_price, 2),
            )
        elif order_type == "stop" and stop_price is not None:
            req = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif_enum,
                stop_price=round(stop_price, 2),
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif_enum,
            )

        order = _retry(lambda: self._trading_client.submit_order(req))
        logger.info(
            "Order submitted: %s %s %s (qty=%.4f, type=%s)",
            side,
            symbol,
            order.id,
            qty,
            order_type,
        )
        return order

    def cancel_order(self, order_id: str) -> None:
        _retry(lambda: self._trading_client.cancel_order_by_id(order_id))
        logger.info("Order cancelled: %s", order_id)

    def cancel_all_orders(self) -> None:
        _retry(self._trading_client.cancel_orders)
        logger.info("All open orders cancelled")

    def close_position(self, symbol: str):
        result = _retry(lambda: self._trading_client.close_position(symbol))
        logger.info("Position closed: %s", symbol)
        return result

    def close_all_positions(self) -> None:
        _retry(lambda: self._trading_client.close_all_positions(cancel_orders=True))
        logger.info("All positions closed")
