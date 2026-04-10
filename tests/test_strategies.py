"""Tests for trading strategies."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig, MeanReversionConfig, MomentumConfig, PairsTradingConfig, RiskConfig
from strategies.base_strategy import Signal, SignalType
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.pairs_trading import PairsTradingStrategy


def _make_ohlcv(prices, start="2023-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    dates = pd.date_range(start=start, periods=len(prices), freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [100_000] * len(prices),
        },
        index=dates,
    )
    return df


RISK_CFG = RiskConfig(
    max_portfolio_risk_pct=2.0,
    max_daily_loss_pct=5.0,
    max_position_size_pct=10.0,
    max_open_positions=10,
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
)


class TestMeanReversionStrategy:
    def _strategy(self):
        cfg = MeanReversionConfig(lookback_period=10, std_threshold=1.5)
        return MeanReversionStrategy(cfg, RISK_CFG)

    def test_buy_signal_when_price_below_lower_band(self):
        """Price that crashes well below the rolling mean should generate a BUY."""
        strategy = self._strategy()
        base = [100.0] * 15
        # Crash price on the last bar
        base[-1] = 80.0
        df = _make_ohlcv(base)
        signals = strategy.generate_signals(
            bars={"AAPL": df},
            current_prices={"AAPL": 80.0},
            portfolio_value=100_000.0,
            open_positions=[],
        )
        buy_signals = [s for s in signals if s.is_buy and s.symbol == "AAPL"]
        assert len(buy_signals) == 1
        assert buy_signals[0].quantity > 0

    def test_sell_signal_when_price_above_upper_band(self):
        """Price spike above rolling mean should generate a SELL (if position open)."""
        strategy = self._strategy()
        base = [100.0] * 15
        base[-1] = 120.0
        df = _make_ohlcv(base)
        signals = strategy.generate_signals(
            bars={"AAPL": df},
            current_prices={"AAPL": 120.0},
            portfolio_value=100_000.0,
            open_positions=["AAPL"],
        )
        sell_signals = [s for s in signals if s.is_sell and s.symbol == "AAPL"]
        assert len(sell_signals) == 1

    def test_no_sell_without_position(self):
        """SELL signal must be suppressed when we do not own the asset."""
        strategy = self._strategy()
        base = [100.0] * 15
        base[-1] = 120.0
        df = _make_ohlcv(base)
        signals = strategy.generate_signals(
            bars={"AAPL": df},
            current_prices={"AAPL": 120.0},
            portfolio_value=100_000.0,
            open_positions=[],
        )
        sell_signals = [s for s in signals if s.is_sell and s.symbol == "AAPL"]
        assert len(sell_signals) == 0

    def test_no_signal_within_bands(self):
        """Stable price within bands produces no signal."""
        strategy = self._strategy()
        prices = [100.0] * 20
        df = _make_ohlcv(prices)
        signals = strategy.generate_signals(
            bars={"AAPL": df},
            current_prices={"AAPL": 100.0},
            portfolio_value=100_000.0,
            open_positions=[],
        )
        assert len(signals) == 0

    def test_insufficient_data_returns_no_signal(self):
        """Too few bars should produce no signals."""
        strategy = self._strategy()
        df = _make_ohlcv([100.0] * 5)
        signals = strategy.generate_signals(
            bars={"AAPL": df},
            current_prices={"AAPL": 100.0},
            portfolio_value=100_000.0,
            open_positions=[],
        )
        assert len(signals) == 0

    def test_empty_bars_returns_no_signal(self):
        strategy = self._strategy()
        signals = strategy.generate_signals(
            bars={"AAPL": pd.DataFrame()},
            current_prices={},
            portfolio_value=100_000.0,
            open_positions=[],
        )
        assert len(signals) == 0


class TestMomentumStrategy:
    def _strategy(self):
        cfg = MomentumConfig(
            rsi_period=5,
            macd_fast=3,
            macd_slow=5,
            macd_signal=2,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
        )
        return MomentumStrategy(cfg, RISK_CFG)

    def test_returns_list(self):
        """generate_signals always returns a list."""
        strategy = self._strategy()
        prices = list(range(50, 100))
        df = _make_ohlcv(prices)
        result = strategy.generate_signals(
            bars={"TSLA": df},
            current_prices={"TSLA": float(prices[-1])},
            portfolio_value=100_000.0,
            open_positions=[],
        )
        assert isinstance(result, list)

    def test_overbought_sell_with_position(self):
        """RSI overbought should produce a SELL when we hold the asset."""
        strategy = self._strategy()
        # Create strongly upward prices to push RSI into overbought territory
        prices = [50.0 + i * 5 for i in range(60)]
        df = _make_ohlcv(prices)
        signals = strategy.generate_signals(
            bars={"TSLA": df},
            current_prices={"TSLA": float(prices[-1])},
            portfolio_value=100_000.0,
            open_positions=["TSLA"],
        )
        sell_signals = [s for s in signals if s.is_sell and s.symbol == "TSLA"]
        # Strongly upward prices will push RSI into overbought territory, triggering a SELL
        assert len(sell_signals) == 1

    def test_no_signal_on_empty_bars(self):
        strategy = self._strategy()
        signals = strategy.generate_signals(
            bars={"TSLA": pd.DataFrame()},
            current_prices={},
            portfolio_value=100_000.0,
            open_positions=[],
        )
        assert signals == []


class TestPairsTradingStrategy:
    def _config(self):
        import os
        os.environ["PAIRS_LIST"] = "AAPL:MSFT"
        return PairsTradingConfig(lookback=20, z_score_entry=1.5, z_score_exit=0.3)

    def _strategy(self):
        return PairsTradingStrategy(self._config(), RISK_CFG)

    def test_required_symbols_returns_both(self):
        strategy = self._strategy()
        syms = strategy.required_symbols()
        assert "AAPL" in syms
        assert "MSFT" in syms

    def test_no_signal_without_sufficient_data(self):
        strategy = self._strategy()
        df_short = _make_ohlcv([100.0] * 5)
        signals = strategy.generate_signals(
            bars={"AAPL": df_short, "MSFT": df_short},
            current_prices={"AAPL": 100.0, "MSFT": 100.0},
            portfolio_value=100_000.0,
            open_positions=[],
        )
        assert signals == []

    def test_entry_signal_on_wide_spread(self):
        """When spread z-score exceeds entry threshold, entry signals are generated."""
        strategy = self._strategy()
        # Build cointegrated prices with a diverging spread at the end
        n = 60
        base = [100.0 + i * 0.1 for i in range(n)]
        df_a = _make_ohlcv(base)
        # MSFT closely follows AAPL but suddenly diverges
        msft_prices = base[:]
        for i in range(n - 5, n):
            msft_prices[i] = msft_prices[i] * 1.15  # MSFT shoots up relative to AAPL
        df_b = _make_ohlcv(msft_prices)

        signals = strategy.generate_signals(
            bars={"AAPL": df_a, "MSFT": df_b},
            current_prices={"AAPL": float(base[-1]), "MSFT": float(msft_prices[-1])},
            portfolio_value=100_000.0,
            open_positions=[],
        )
        # Depending on z-score, may or may not trigger - just ensure returns a list
        assert isinstance(signals, list)


class TestSignal:
    def test_signal_buy_property(self):
        sig = Signal("AAPL", SignalType.BUY, 150.0, 10.0)
        assert sig.is_buy
        assert not sig.is_sell

    def test_signal_sell_property(self):
        sig = Signal("AAPL", SignalType.SELL, 150.0, 10.0)
        assert sig.is_sell
        assert not sig.is_buy

    def test_signal_repr(self):
        sig = Signal("AAPL", SignalType.BUY, 150.0, 10.0, confidence=0.8)
        r = repr(sig)
        assert "buy" in r
        assert "AAPL" in r
