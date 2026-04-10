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

    def test_generate_daily_report_returns_string(self, risk_manager):
        report = risk_manager.generate_daily_report(100_000.0)
        assert isinstance(report, str)
        assert "DAILY PERFORMANCE REPORT" in report

    def test_daily_report_includes_key_metrics(self, risk_manager):
        risk_manager.record_trade("AAPL", "buy", 10, 150.0, "mean_reversion", "test")
        risk_manager.update_daily_pnl(200.0, 100_000.0)
        report = risk_manager.generate_daily_report(100_000.0)
        assert "Portfolio Value" in report
        assert "Daily P&L" in report
        assert "Weekly P&L" in report
        assert "Win Rate" in report


class TestTrailingStop:
    def test_trailing_stop_disabled_when_pct_zero(self, risk_manager):
        """compute_trailing_stop_price returns None when trailing_stop_pct == 0."""
        assert risk_manager.config.trailing_stop_pct == 0.0
        result = risk_manager.compute_trailing_stop_price("AAPL", 100.0)
        assert result is None

    def test_trailing_stop_enabled(self, tmp_path):
        cfg = RiskConfig(
            max_portfolio_risk_pct=2.0,
            max_daily_loss_pct=5.0,
            max_position_size_pct=10.0,
            max_open_positions=3,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            trailing_stop_pct=5.0,
        )
        rm = RiskManager(cfg, db_path=str(tmp_path / "ts_test.db"))
        # No high watermark yet — falls back to entry price
        stop = rm.compute_trailing_stop_price("AAPL", 100.0)
        assert stop == pytest.approx(95.0, rel=1e-4)

    def test_trailing_high_updates(self, tmp_path):
        cfg = RiskConfig(
            max_portfolio_risk_pct=2.0,
            max_daily_loss_pct=5.0,
            max_position_size_pct=10.0,
            max_open_positions=3,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            trailing_stop_pct=10.0,
        )
        rm = RiskManager(cfg, db_path=str(tmp_path / "ts2_test.db"))
        rm.update_trailing_high("AAPL", 150.0)
        rm.update_trailing_high("AAPL", 140.0)  # should NOT lower the high
        stop = rm.compute_trailing_stop_price("AAPL", 100.0)
        assert stop == pytest.approx(135.0, rel=1e-4)  # 10% below 150

    def test_trailing_high_cleared_on_sell(self, tmp_path):
        cfg = RiskConfig(
            max_portfolio_risk_pct=2.0,
            max_daily_loss_pct=5.0,
            max_position_size_pct=10.0,
            max_open_positions=3,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            trailing_stop_pct=5.0,
        )
        rm = RiskManager(cfg, db_path=str(tmp_path / "ts3_test.db"))
        rm.update_trailing_high("AAPL", 200.0)
        assert "AAPL" in rm._trailing_highs
        # Recording a sell should clear the trailing high
        rm.record_trade("AAPL", "buy", 10, 150.0, "test")
        rm.record_trade("AAPL", "sell", 10, 160.0, "test")
        assert "AAPL" not in rm._trailing_highs


class TestRealizedPnL:
    def test_pnl_computed_on_sell(self, tmp_path):
        """When a sell is recorded, P&L should be auto-computed from prior buy."""
        cfg = RISK_CFG
        rm = RiskManager(cfg, db_path=str(tmp_path / "pnl_test.db"))
        rm.record_trade("AAPL", "buy", 10, 100.0, "test")
        rm.record_trade("AAPL", "sell", 10, 120.0, "test")
        history = rm.get_trade_history()
        sell_trade = next(t for t in history if t["side"] == "sell")
        assert sell_trade["pnl"] == pytest.approx(200.0, rel=1e-4)  # (120-100)*10

    def test_pnl_negative_on_loss(self, tmp_path):
        cfg = RISK_CFG
        rm = RiskManager(cfg, db_path=str(tmp_path / "pnl_loss_test.db"))
        rm.record_trade("MSFT", "buy", 5, 200.0, "test")
        rm.record_trade("MSFT", "sell", 5, 180.0, "test")
        history = rm.get_trade_history()
        sell_trade = next(t for t in history if t["side"] == "sell")
        assert sell_trade["pnl"] == pytest.approx(-100.0, rel=1e-4)  # (180-200)*5

    def test_pnl_zero_when_no_prior_buy(self, tmp_path):
        """Sell without a preceding buy should record zero P&L."""
        cfg = RISK_CFG
        rm = RiskManager(cfg, db_path=str(tmp_path / "pnl_nobuy_test.db"))
        rm.record_trade("GOOG", "sell", 3, 150.0, "test")
        history = rm.get_trade_history()
        assert history[0]["pnl"] == 0.0
