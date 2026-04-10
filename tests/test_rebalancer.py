"""Tests for the PortfolioRebalancer."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RebalancerConfig, RiskConfig
from rebalancer import PortfolioRebalancer
from strategies.base_strategy import SignalType


_RISK = RiskConfig(
    max_portfolio_risk_pct=2.0,
    max_daily_loss_pct=5.0,
    max_position_size_pct=10.0,  # 10% target weight
    max_open_positions=10,
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
)


def _make_position(symbol: str, qty: float, current_price: float):
    pos = SimpleNamespace(
        symbol=symbol,
        qty=qty,
        current_price=current_price,
    )
    return pos


def _rebalancer(enabled: bool = True, interval_hours: int = 1) -> PortfolioRebalancer:
    cfg = RebalancerConfig(enabled=enabled, check_interval_hours=interval_hours)
    return PortfolioRebalancer(cfg, _RISK)


class TestRebalancerIsDue:
    def test_initially_due(self):
        rb = _rebalancer()
        assert rb.is_due()

    def test_not_due_after_check(self):
        rb = _rebalancer(interval_hours=100)
        rb.mark_checked()
        assert not rb.is_due()

    def test_disabled_never_due(self):
        rb = _rebalancer(enabled=False)
        assert not rb.is_due()


class TestNeedsRebalancing:
    def test_overweight_position_detected(self):
        rb = _rebalancer()
        # $15k position in $100k portfolio = 15% > 10% target
        pos = _make_position("AAPL", 100.0, 150.0)
        result = rb.needs_rebalancing([pos], {"AAPL": 150.0}, 100_000.0)
        assert len(result) == 1
        sym, weight, excess_qty = result[0]
        assert sym == "AAPL"
        assert weight == pytest.approx(15.0)
        assert excess_qty > 0

    def test_within_target_not_overweight(self):
        rb = _rebalancer()
        # $8k position in $100k portfolio = 8% < 10% target
        pos = _make_position("AAPL", 80.0, 100.0)
        result = rb.needs_rebalancing([pos], {"AAPL": 100.0}, 100_000.0)
        assert result == []

    def test_exact_target_weight_not_overweight(self):
        rb = _rebalancer()
        # Exactly 10%: $10k in $100k
        pos = _make_position("AAPL", 100.0, 100.0)
        result = rb.needs_rebalancing([pos], {"AAPL": 100.0}, 100_000.0)
        assert result == []

    def test_zero_portfolio_value_returns_empty(self):
        rb = _rebalancer()
        pos = _make_position("AAPL", 100.0, 150.0)
        result = rb.needs_rebalancing([pos], {"AAPL": 150.0}, 0.0)
        assert result == []

    def test_uses_position_current_price_as_fallback(self):
        rb = _rebalancer()
        pos = _make_position("AAPL", 100.0, 150.0)
        # No price in current_prices dict → falls back to pos.current_price
        result = rb.needs_rebalancing([pos], {}, 100_000.0)
        assert len(result) == 1


class TestGenerateRebalanceSignals:
    def test_generates_sell_signal_for_overweight(self):
        rb = _rebalancer()
        pos = _make_position("AAPL", 100.0, 150.0)
        signals = rb.generate_rebalance_signals([pos], {"AAPL": 150.0}, 100_000.0)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.symbol == "AAPL"
        assert sig.signal_type == SignalType.SELL
        assert sig.quantity > 0
        assert "rebalance" in sig.reason

    def test_no_signals_for_balanced_portfolio(self):
        rb = _rebalancer()
        pos = _make_position("AAPL", 50.0, 100.0)  # 5% weight
        signals = rb.generate_rebalance_signals([pos], {"AAPL": 100.0}, 100_000.0)
        assert signals == []

    def test_excess_qty_calculation(self):
        rb = _rebalancer()
        # $15k position, target $10k → excess = $5k / $150 ≈ 33.33 shares
        pos = _make_position("AAPL", 100.0, 150.0)
        signals = rb.generate_rebalance_signals([pos], {"AAPL": 150.0}, 100_000.0)
        assert len(signals) == 1
        expected_excess = round((15_000 - 10_000) / 150.0, 4)
        assert signals[0].quantity == pytest.approx(expected_excess, rel=1e-3)
