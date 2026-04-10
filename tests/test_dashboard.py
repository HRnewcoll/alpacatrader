"""Tests for the web Dashboard."""
from __future__ import annotations

import os
import sys
import sqlite3
import tempfile
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DashboardConfig
from dashboard import Dashboard, push_signal, _signal_feed, _signal_lock


def _cfg(enabled: bool = True, port: int = 0) -> DashboardConfig:
    return DashboardConfig(enabled=enabled, host="127.0.0.1", port=port)


class TestSignalFeed:
    def test_push_signal_adds_entry(self):
        with _signal_lock:
            _signal_feed.clear()
        push_signal("AAPL", "BUY", 150.0, "test_strategy", "test reason")
        with _signal_lock:
            assert len(_signal_feed) == 1
            entry = _signal_feed[-1]
        assert entry["symbol"] == "AAPL"
        assert entry["action"] == "BUY"
        assert entry["price"] == 150.0
        assert entry["strategy"] == "test_strategy"
        assert "ts" in entry

    def test_feed_capped_at_100(self):
        with _signal_lock:
            _signal_feed.clear()
        for i in range(110):
            push_signal("X", "BUY", float(i), "s", "r")
        with _signal_lock:
            assert len(_signal_feed) == 100


class TestDashboardBuildApp:
    def _make_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                """CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, side TEXT, qty REAL,
                    price REAL, strategy TEXT, reason TEXT, pnl REAL,
                    order_id TEXT
                )"""
            )
            conn.execute(
                "INSERT INTO trades VALUES (NULL,'2024-01-02T15:00:00','AAPL','buy',10,150.0,'test','r',5.0,'')"
            )
            conn.commit()
        return db

    def test_index_returns_html(self, tmp_path):
        db = self._make_db(tmp_path)
        cfg = _cfg(enabled=True)
        d = Dashboard(cfg, db_path=db)
        app = d._build_app()
        with app.test_client() as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert b"AlpacaTrader" in resp.data

    def test_api_trades_returns_json(self, tmp_path):
        db = self._make_db(tmp_path)
        cfg = _cfg(enabled=True)
        d = Dashboard(cfg, db_path=db)
        app = d._build_app()
        with app.test_client() as client:
            resp = client.get("/api/trades")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "trades" in data
            assert len(data["trades"]) == 1
            assert data["trades"][0]["symbol"] == "AAPL"

    def test_api_positions_returns_json(self, tmp_path):
        db = self._make_db(tmp_path)
        cfg = _cfg(enabled=True)
        d = Dashboard(cfg, db_path=db, get_positions_fn=lambda: [])
        app = d._build_app()
        with app.test_client() as client:
            resp = client.get("/api/positions")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "positions" in data

    def test_api_signals_returns_json(self, tmp_path):
        db = self._make_db(tmp_path)
        with _signal_lock:
            _signal_feed.clear()
        push_signal("TSLA", "SELL", 200.0, "momentum", "rsi_overbought")
        cfg = _cfg(enabled=True)
        d = Dashboard(cfg, db_path=db)
        app = d._build_app()
        with app.test_client() as client:
            resp = client.get("/api/signals")
            assert resp.status_code == 200
            data = resp.get_json()
            assert any(s["symbol"] == "TSLA" for s in data.get("signals", []))

    def test_api_performance_returns_json(self, tmp_path):
        db = self._make_db(tmp_path)
        perf = {"portfolio_value": 100_000.0, "daily_pnl": 250.0}
        cfg = _cfg(enabled=True)
        d = Dashboard(cfg, db_path=db, get_performance_fn=lambda: perf)
        app = d._build_app()
        with app.test_client() as client:
            resp = client.get("/api/performance")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["portfolio_value"] == 100_000.0


class TestDashboardStartStop:
    def test_disabled_start_is_noop(self, tmp_path):
        db = str(tmp_path / "d.db")
        cfg = _cfg(enabled=False)
        d = Dashboard(cfg, db_path=db)
        d.start()
        assert d._thread is None
