"""Tests for new features: health endpoint, validate command, slippage guard, MTF."""
from __future__ import annotations

import os
import sys
import sqlite3
import math
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig, DashboardConfig, RiskConfig
from dashboard import Dashboard, push_signal


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def _make_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS trades "
                "(id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, side TEXT, "
                "qty REAL, price REAL, strategy TEXT, reason TEXT, pnl REAL, order_id TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS daily_pnl "
                "(trade_date TEXT PRIMARY KEY, realized_pnl REAL, starting_portfolio_value REAL)"
            )
        return db

    def test_health_returns_ok(self, tmp_path):
        db = self._make_db(tmp_path)
        cfg = DashboardConfig(enabled=True, host="127.0.0.1", port=0)
        dash = Dashboard(cfg, db_path=db)
        app = dash._build_app()
        client = app.test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data

    def test_health_uptime_is_non_negative(self, tmp_path):
        db = self._make_db(tmp_path)
        cfg = DashboardConfig(enabled=True, host="127.0.0.1", port=0)
        dash = Dashboard(cfg, db_path=db)
        import time
        dash._start_time = time.time() - 5
        app = dash._build_app()
        client = app.test_client()
        resp = client.get("/health")
        data = resp.get_json()
        assert data["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# Validate command
# ---------------------------------------------------------------------------

class TestValidateCommand:
    def _make_populated_db(self, tmp_path, trade_count=25, sharpe_good=True):
        """Create a SQLite DB with synthetic trade history."""
        db = str(tmp_path / "validate_test.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE trades "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT, side TEXT, "
                "qty REAL, price REAL, strategy TEXT, reason TEXT DEFAULT '', "
                "pnl REAL DEFAULT 0, order_id TEXT DEFAULT '')"
            )
            conn.execute(
                "CREATE TABLE daily_pnl "
                "(trade_date TEXT PRIMARY KEY, realized_pnl REAL DEFAULT 0, "
                "starting_portfolio_value REAL DEFAULT 0)"
            )
            # Insert sell trades (completed trades)
            now = datetime.now(tz=timezone.utc)
            for i in range(trade_count):
                ts = (now - timedelta(days=i % 25)).isoformat()
                conn.execute(
                    "INSERT INTO trades (timestamp, symbol, side, qty, price, strategy, pnl) "
                    "VALUES (?, ?, 'sell', 10, 100, 'test', ?)",
                    (ts, f"SYM{i}", 50.0 if sharpe_good else -50.0),
                )
            # Insert daily P&L
            for i in range(25):
                day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
                pnl = 200.0 if sharpe_good else -200.0
                conn.execute(
                    "INSERT OR IGNORE INTO daily_pnl (trade_date, realized_pnl, starting_portfolio_value) "
                    "VALUES (?, ?, 100000)",
                    (day, pnl),
                )
        return db

    def test_validate_pass(self, tmp_path):
        from main import cmd_validate
        db = self._make_populated_db(tmp_path, trade_count=25, sharpe_good=True)
        cfg = AppConfig()
        cfg.db_path = db
        result = cmd_validate(cfg, days=30, min_trades=20)
        assert result == 0

    def test_validate_fail_insufficient_trades(self, tmp_path):
        from main import cmd_validate
        db = self._make_populated_db(tmp_path, trade_count=5, sharpe_good=True)
        cfg = AppConfig()
        cfg.db_path = db
        result = cmd_validate(cfg, days=30, min_trades=20)
        assert result == 1

    def test_validate_fail_negative_pnl(self, tmp_path):
        from main import cmd_validate
        db = self._make_populated_db(tmp_path, trade_count=25, sharpe_good=False)
        cfg = AppConfig()
        cfg.db_path = db
        result = cmd_validate(cfg, days=30, min_trades=20)
        assert result == 1

    def test_validate_empty_db(self, tmp_path):
        from main import cmd_validate
        db = self._make_populated_db(tmp_path, trade_count=0)
        cfg = AppConfig()
        cfg.db_path = db
        result = cmd_validate(cfg, days=30, min_trades=20)
        assert result == 1


# ---------------------------------------------------------------------------
# Slippage guard (TradingEngine._apply_slippage_guard)
# ---------------------------------------------------------------------------

class TestSlippageGuard:
    def _engine_with_spread(self, spread_pct: float):
        """Return a (engine, snapshots) pair with the given spread."""
        from main import TradingEngine
        cfg = AppConfig()
        cfg.risk.max_bid_ask_spread_pct = 0.5  # 0.5 % max
        # Prevent actual Alpaca client init
        with patch("data_handler.DataHandler._init_clients"):
            engine = TradingEngine.__new__(TradingEngine)
            engine.config = cfg
            import logging
            engine.logger = logging.getLogger("test")
        return engine

    def test_buy_blocked_when_spread_too_wide(self):
        from main import TradingEngine
        from strategies.base_strategy import Signal, SignalType
        cfg = AppConfig()
        cfg.risk.max_bid_ask_spread_pct = 0.3
        with patch("data_handler.DataHandler._init_clients"):
            eng = TradingEngine.__new__(TradingEngine)
            eng.config = cfg
            import logging
            eng.logger = logging.getLogger("test")

        signals = [Signal("AAPL", SignalType.BUY, 150.0, 10.0)]
        snapshots = {"AAPL": {"bid": 149.0, "ask": 151.0, "last_price": 150.0, "bid_ask_spread_pct": 1.33}}
        result = eng._apply_slippage_guard(signals, snapshots)
        assert len(result) == 0

    def test_buy_passes_when_spread_ok(self):
        from main import TradingEngine
        from strategies.base_strategy import Signal, SignalType
        cfg = AppConfig()
        cfg.risk.max_bid_ask_spread_pct = 0.5
        with patch("data_handler.DataHandler._init_clients"):
            eng = TradingEngine.__new__(TradingEngine)
            eng.config = cfg
            import logging
            eng.logger = logging.getLogger("test")

        signals = [Signal("AAPL", SignalType.BUY, 150.0, 10.0)]
        snapshots = {"AAPL": {"bid": 149.9, "ask": 150.1, "last_price": 150.0, "bid_ask_spread_pct": 0.13}}
        result = eng._apply_slippage_guard(signals, snapshots)
        assert len(result) == 1

    def test_sell_always_passes(self):
        from main import TradingEngine
        from strategies.base_strategy import Signal, SignalType
        cfg = AppConfig()
        cfg.risk.max_bid_ask_spread_pct = 0.1
        with patch("data_handler.DataHandler._init_clients"):
            eng = TradingEngine.__new__(TradingEngine)
            eng.config = cfg
            import logging
            eng.logger = logging.getLogger("test")

        signals = [Signal("AAPL", SignalType.SELL, 150.0, 10.0)]
        snapshots = {"AAPL": {"bid_ask_spread_pct": 5.0}}  # very wide spread
        result = eng._apply_slippage_guard(signals, snapshots)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Multi-timeframe confirmation filter
# ---------------------------------------------------------------------------

class TestMultiTimeframeFilter:
    def _engine(self, mtf_enabled=True, lookback=5):
        from main import TradingEngine
        cfg = AppConfig()
        cfg.multi_timeframe.enabled = mtf_enabled
        cfg.multi_timeframe.trend_lookback_days = lookback
        with patch("data_handler.DataHandler._init_clients"):
            eng = TradingEngine.__new__(TradingEngine)
            eng.config = cfg
            import logging
            eng.logger = logging.getLogger("test")
        return eng

    def test_buy_suppressed_in_bearish_trend(self):
        import pandas as pd
        from strategies.base_strategy import Signal, SignalType
        eng = self._engine()
        # Downtrend: SMA > current price
        prices = [110.0, 108.0, 106.0, 104.0, 102.0, 100.0]
        dates = pd.date_range("2023-01-01", periods=len(prices), freq="B", tz="UTC")
        df = pd.DataFrame({"close": prices}, index=dates)

        signals = [Signal("AAPL", SignalType.BUY, 100.0, 5.0)]
        result = eng._apply_mtf_filter(signals, {"AAPL": df})
        assert len(result) == 0

    def test_buy_passes_in_bullish_trend(self):
        import pandas as pd
        from strategies.base_strategy import Signal, SignalType
        eng = self._engine()
        # Uptrend: current price > SMA
        prices = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
        dates = pd.date_range("2023-01-01", periods=len(prices), freq="B", tz="UTC")
        df = pd.DataFrame({"close": prices}, index=dates)

        signals = [Signal("AAPL", SignalType.BUY, 100.0, 5.0)]
        result = eng._apply_mtf_filter(signals, {"AAPL": df})
        assert len(result) == 1

    def test_sell_always_passes(self):
        import pandas as pd
        from strategies.base_strategy import Signal, SignalType
        eng = self._engine()
        prices = [110.0, 108.0, 106.0, 104.0, 102.0, 100.0]
        dates = pd.date_range("2023-01-01", periods=len(prices), freq="B", tz="UTC")
        df = pd.DataFrame({"close": prices}, index=dates)

        signals = [Signal("AAPL", SignalType.SELL, 100.0, 5.0)]
        result = eng._apply_mtf_filter(signals, {"AAPL": df})
        assert len(result) == 1
