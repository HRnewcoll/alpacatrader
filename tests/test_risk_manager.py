"""Tests for RiskManager."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RiskConfig
from risk_manager import RiskManager
from strategies.base_strategy import Signal, SignalType


RISK_CFG = RiskConfig(
    max_portfolio_risk_pct=2.0,
    max_daily_loss_pct=5.0,
    max_position_size_pct=10.0,
    max_open_positions=3,
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
)


@pytest.fixture
def risk_manager(tmp_path):
    db = str(tmp_path / "test_trading.db")
    return RiskManager(RISK_CFG, db_path=db)


class TestRiskManager:
    def test_init_creates_db(self, tmp_path):
        db = str(tmp_path / "init_test.db")
        rm = RiskManager(RISK_CFG, db_path=db)
        assert os.path.exists(db)

    def test_record_and_get_trade_history(self, risk_manager):
        risk_manager.record_trade("AAPL", "buy", 10, 150.0, "test_strategy", "test reason")
        history = risk_manager.get_trade_history()
        assert len(history) == 1
        assert history[0]["symbol"] == "AAPL"
        assert history[0]["side"] == "buy"
        assert history[0]["qty"] == 10
        assert history[0]["price"] == 150.0

    def test_calculate_position_size(self, risk_manager):
        # Default uses max_portfolio_risk_pct=2.0%: $100k * 2% = $2k / $100 = 20 shares
        qty = risk_manager.calculate_position_size(100_000.0, 100.0)
        assert qty == 20.0

    def test_position_size_respects_max(self, risk_manager):
        qty = risk_manager.calculate_position_size(100_000.0, 100.0, risk_pct=20.0)
        # Capped at max_position_size_pct=10%
        assert qty == 100.0

    def test_stop_loss_price(self, risk_manager):
        stop = risk_manager.compute_stop_loss_price(100.0)
        assert stop == pytest.approx(98.0, rel=1e-4)

    def test_take_profit_price(self, risk_manager):
        tp = risk_manager.compute_take_profit_price(100.0)
        assert tp == pytest.approx(104.0, rel=1e-4)

    def test_filter_signals_buy_approved(self, risk_manager):
        signals = [
            Signal("AAPL", SignalType.BUY, 150.0, 10.0, reason="test"),
        ]
        approved = risk_manager.filter_signals(signals, 100_000.0, [])
        assert len(approved) == 1

    def test_filter_signals_sell_approved_without_position_check(self, risk_manager):
        """SELL signals are passed through directly (position check is in strategy)."""
        signals = [Signal("AAPL", SignalType.SELL, 150.0, 10.0)]
        approved = risk_manager.filter_signals(signals, 100_000.0, ["AAPL"])
        assert len(approved) == 1

    def test_filter_signals_max_positions_reached(self, risk_manager):
        """New BUY should be blocked when max open positions is reached."""
        open_positions = ["AAPL", "MSFT", "GOOG"]  # equals max_open_positions=3
        signals = [Signal("TSLA", SignalType.BUY, 200.0, 5.0)]
        approved = risk_manager.filter_signals(signals, 100_000.0, open_positions)
        assert len(approved) == 0

    def test_filter_signals_add_to_existing_position_allowed(self, risk_manager):
        """BUY on already-held symbol is allowed even at max positions."""
        open_positions = ["AAPL", "MSFT", "GOOG"]
        signals = [Signal("AAPL", SignalType.BUY, 150.0, 5.0)]
        approved = risk_manager.filter_signals(signals, 100_000.0, open_positions)
        assert len(approved) == 1

    def test_filter_blocks_all_when_daily_loss_exceeded(self, risk_manager):
        """When daily loss limit is breached, no signals should pass."""
        # Force a large daily loss
        risk_manager.update_daily_pnl(-10_000.0, 100_000.0)
        signals = [Signal("AAPL", SignalType.BUY, 150.0, 10.0)]
        approved = risk_manager.filter_signals(signals, 100_000.0, [])
        assert len(approved) == 0

    def test_performance_summary(self, risk_manager):
        summary = risk_manager.performance_summary(100_000.0)
        assert "portfolio_value" in summary
        assert "daily_pnl" in summary
        assert "drawdown_pct" in summary
        assert summary["portfolio_value"] == 100_000.0

    def test_daily_pnl_accumulates(self, risk_manager):
        risk_manager.update_daily_pnl(500.0, 100_000.0)
        risk_manager.update_daily_pnl(300.0, 100_000.0)
        assert risk_manager.get_daily_pnl() == pytest.approx(800.0)
