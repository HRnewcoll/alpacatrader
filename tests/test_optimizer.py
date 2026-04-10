"""Tests for WalkForwardOptimizer."""
from __future__ import annotations

import os
import sys
import math
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig, MeanReversionConfig, MomentumConfig, OptimizerConfig, RiskConfig
from optimizer import WalkForwardOptimizer, MR_GRID, MOM_GRID, OptimizationResult


def _make_ohlcv(prices, start="2022-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
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


class TestParamCombinations:
    def test_grid_count(self):
        grid = {"a": [1, 2], "b": [3, 4, 5]}
        combos = list(WalkForwardOptimizer._param_combinations(grid))
        assert len(combos) == 6

    def test_single_value_grid(self):
        grid = {"x": [42]}
        combos = list(WalkForwardOptimizer._param_combinations(grid))
        assert len(combos) == 1
        assert combos[0]["x"] == 42

    def test_combo_keys_match_grid_keys(self):
        grid = MR_GRID
        combos = list(WalkForwardOptimizer._param_combinations(grid))
        assert set(combos[0].keys()) == set(grid.keys())


class TestMRGridStructure:
    def test_mr_grid_has_required_keys(self):
        assert "lookback_period" in MR_GRID
        assert "std_threshold" in MR_GRID

    def test_mr_grid_has_multiple_values(self):
        assert len(MR_GRID["lookback_period"]) > 1
        assert len(MR_GRID["std_threshold"]) > 1


class TestMOMGridStructure:
    def test_mom_grid_has_required_keys(self):
        for key in ("rsi_period", "macd_fast", "macd_slow", "macd_signal"):
            assert key in MOM_GRID


class TestOptimizeMeanReversion:
    def test_optimize_returns_result(self):
        config = _default_config()
        opt = WalkForwardOptimizer(OptimizerConfig(lookback_months=1))
        prices = [100.0 + 10.0 * math.sin(i * 0.3) for i in range(80)]
        bars = {"AAPL": _make_ohlcv(prices)}
        # Use a tiny grid for speed
        tiny_grid = {"lookback_period": [10, 20], "std_threshold": [1.5, 2.0]}
        result = opt.optimize_strategy("mean_reversion", bars, config, tiny_grid)
        assert isinstance(result, OptimizationResult)
        assert result.strategy_name == "mean_reversion"
        assert result.total_combinations == 4
        assert isinstance(result.best_sharpe, float)
        assert set(result.best_params.keys()) == {"lookback_period", "std_threshold"}

    def test_unknown_strategy_raises(self):
        config = _default_config()
        opt = WalkForwardOptimizer()
        bars = {"AAPL": _make_ohlcv([100.0] * 80)}
        with pytest.raises(ValueError, match="Unknown strategy"):
            opt.optimize_strategy("unknown_strat", bars, config, {"x": [1]})


class TestWriteEnv:
    def test_writes_new_file(self, tmp_path):
        env_file = str(tmp_path / ".env")
        params = {"lookback_period": 15, "std_threshold": 1.8}
        WalkForwardOptimizer.write_env(params, env_file)
        content = open(env_file).read()
        assert "MR_LOOKBACK_PERIOD=15" in content
        assert "MR_STD_THRESHOLD=1.8" in content

    def test_updates_existing_key(self, tmp_path):
        env_file = str(tmp_path / ".env")
        with open(env_file, "w") as f:
            f.write("MR_LOOKBACK_PERIOD=20\nMR_STD_THRESHOLD=2.0\n")
        params = {"lookback_period": 10, "std_threshold": 1.5}
        WalkForwardOptimizer.write_env(params, env_file)
        lines = open(env_file).readlines()
        assert any("MR_LOOKBACK_PERIOD=10" in l for l in lines)
        assert not any("MR_LOOKBACK_PERIOD=20" in l for l in lines)

    def test_preserves_other_keys(self, tmp_path):
        env_file = str(tmp_path / ".env")
        with open(env_file, "w") as f:
            f.write("ALPACA_API_KEY=mykey\nMR_LOOKBACK_PERIOD=20\n")
        WalkForwardOptimizer.write_env({"lookback_period": 10}, env_file)
        content = open(env_file).read()
        assert "ALPACA_API_KEY=mykey" in content

    def test_empty_params_no_change(self, tmp_path):
        env_file = str(tmp_path / ".env")
        original = "ALPACA_API_KEY=test\n"
        with open(env_file, "w") as f:
            f.write(original)
        WalkForwardOptimizer.write_env({}, env_file)
        assert open(env_file).read() == original


class TestOptimizationResult:
    def test_print_summary_runs(self, capsys):
        result = OptimizationResult(
            strategy_name="test",
            best_params={"lookback_period": 20},
            best_sharpe=1.23,
            total_combinations=4,
        )
        result.print_summary()
        out = capsys.readouterr().out
        assert "test" in out
        assert "1.2300" in out
