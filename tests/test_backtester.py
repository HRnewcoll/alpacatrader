"""Tests for the backtesting engine."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine import Backtester, BacktestResult, Trade
from config import AppConfig, MeanReversionConfig, MomentumConfig, RiskConfig
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy


def _make_ohlcv(prices, start="2022-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": [500_000] * len(prices),
        },
        index=dates,
    )


def _default_config() -> AppConfig:
    cfg = AppConfig()
    cfg.risk = RiskConfig(
        max_portfolio_risk_pct=2.0,
        max_daily_loss_pct=5.0,
        max_position_size_pct=10.0,
        max_open_positions=10,
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
    )
    return cfg


class TestBacktestResult:
    def _make_result(self):
        prices = [100.0 + i for i in range(100)]
        equity = pd.Series(
            [100_000 + i * 100 for i in range(100)],
            index=pd.date_range("2022-01-01", periods=100, freq="B", tz="UTC"),
        )
        trades = [
            Trade("AAPL", datetime(2022, 1, 10, tzinfo=timezone.utc),
                  datetime(2022, 2, 1, tzinfo=timezone.utc),
                  100.0, 110.0, 10.0, "long", "test", pnl=100.0, pnl_pct=10.0),
            Trade("AAPL", datetime(2022, 2, 5, tzinfo=timezone.utc),
                  datetime(2022, 2, 20, tzinfo=timezone.utc),
                  110.0, 105.0, 10.0, "long", "test", pnl=-50.0, pnl_pct=-4.5),
        ]
        return BacktestResult(
            strategy_name="test",
            start_date=datetime(2022, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2022, 12, 31, tzinfo=timezone.utc),
            initial_capital=100_000.0,
            final_capital=105_000.0,
            trades=trades,
            equity_curve=equity,
        )

    def test_total_return(self):
        result = self._make_result()
        assert result.total_return_pct == pytest.approx(5.0)

    def test_win_rate(self):
        result = self._make_result()
        assert result.win_rate == pytest.approx(50.0)

    def test_num_trades(self):
        result = self._make_result()
        assert result.num_trades == 2

    def test_avg_pnl(self):
        result = self._make_result()
        assert result.avg_pnl == pytest.approx(25.0)

    def test_max_drawdown_non_positive(self):
        result = self._make_result()
        assert result.max_drawdown_pct <= 0

    def test_sharpe_ratio_calculated(self):
        result = self._make_result()
        sharpe = result.sharpe_ratio
        assert isinstance(sharpe, float)

    def test_profit_factor(self):
        result = self._make_result()
        assert result.profit_factor == pytest.approx(2.0)

    def test_summary_keys(self):
        result = self._make_result()
        s = result.summary()
        required_keys = [
            "strategy", "start_date", "end_date", "initial_capital",
            "final_capital", "total_return_pct", "num_trades", "win_rate_pct",
            "sharpe_ratio", "max_drawdown_pct", "profit_factor",
        ]
        for k in required_keys:
            assert k in s


class TestBacktester:
    def _mean_reversion_strategy(self):
        cfg = MeanReversionConfig(lookback_period=10, std_threshold=1.5)
        risk = RiskConfig(
            max_portfolio_risk_pct=2.0,
            max_daily_loss_pct=5.0,
            max_position_size_pct=10.0,
            max_open_positions=10,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        return MeanReversionStrategy(cfg, risk)

    def test_backtester_runs_and_returns_result(self):
        strategy = self._mean_reversion_strategy()
        config = _default_config()
        bt = Backtester(strategy, config, initial_capital=100_000.0)

        # Create sinusoidal prices to generate buy/sell signals
        import math
        prices = [100.0 + 10.0 * math.sin(i * 0.3) for i in range(80)]
        bars = {"AAPL": _make_ohlcv(prices)}

        result = bt.run(bars)
        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 100_000.0
        assert result.final_capital > 0

    def test_backtester_raises_on_empty_bars(self):
        strategy = self._mean_reversion_strategy()
        config = _default_config()
        bt = Backtester(strategy, config)
        with pytest.raises(ValueError, match="No bar data"):
            bt.run({})

    def test_backtester_raises_when_insufficient_data(self):
        strategy = self._mean_reversion_strategy()
        config = _default_config()
        bt = Backtester(strategy, config)
        bars = {"AAPL": _make_ohlcv([100.0] * 5)}
        with pytest.raises(ValueError):
            bt.run(bars)

    def test_backtester_capital_never_negative(self):
        strategy = self._mean_reversion_strategy()
        config = _default_config()
        bt = Backtester(strategy, config, initial_capital=10_000.0)

        import math
        prices = [100.0 + 20.0 * math.sin(i * 0.5) for i in range(100)]
        bars = {"AAPL": _make_ohlcv(prices)}

        result = bt.run(bars)
        assert result.final_capital >= 0
