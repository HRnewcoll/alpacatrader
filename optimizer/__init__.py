"""Walk-forward optimizer.

Grid-searches strategy parameters over a rolling historical window and
returns (and optionally writes to ``.env``) the configuration that
maximises the Sharpe ratio.

Usage (CLI):
    python main.py optimize [--months 6] [--write-env]
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from alerts import get_logger
from backtest_engine import Backtester
from config import (
    AppConfig,
    MeanReversionConfig,
    MomentumConfig,
    OptimizerConfig,
)
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy

logger = get_logger("optimizer")

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

MR_GRID: Dict[str, List[Any]] = {
    "lookback_period": [10, 15, 20, 30],
    "std_threshold": [1.5, 2.0, 2.5],
}

MOM_GRID: Dict[str, List[Any]] = {
    "rsi_period": [10, 14, 21],
    "macd_fast": [8, 12],
    "macd_slow": [21, 26],
    "macd_signal": [7, 9],
    "rsi_overbought": [65.0, 70.0, 75.0],
    "rsi_oversold": [25.0, 30.0, 35.0],
}


@dataclass
class OptimizationResult:
    strategy_name: str
    best_params: Dict[str, Any]
    best_sharpe: float
    total_combinations: int
    all_results: List[Dict[str, Any]] = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print(f"  Optimization: {self.strategy_name}")
        print(f"  Combinations tested : {self.total_combinations}")
        print(f"  Best Sharpe         : {self.best_sharpe:.4f}")
        print(f"  Best Parameters:")
        for k, v in self.best_params.items():
            print(f"    {k:25s}: {v}")
        print(f"{'='*55}\n")


class WalkForwardOptimizer:
    """Grid-searches strategy parameters to maximise Sharpe ratio."""

    def __init__(self, config: Optional[OptimizerConfig] = None) -> None:
        self.config = config or OptimizerConfig()

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    @staticmethod
    def _param_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Return the Cartesian product of all parameter values."""
        keys = list(grid.keys())
        for values in itertools.product(*[grid[k] for k in keys]):
            yield dict(zip(keys, values))

    # ------------------------------------------------------------------
    # Per-strategy helpers
    # ------------------------------------------------------------------

    def _build_mr_strategy(self, params: Dict[str, Any], app_config: AppConfig) -> BaseStrategy:
        cfg = MeanReversionConfig(
            lookback_period=params["lookback_period"],
            std_threshold=params["std_threshold"],
        )
        return MeanReversionStrategy(cfg, app_config.risk)

    def _build_mom_strategy(self, params: Dict[str, Any], app_config: AppConfig) -> BaseStrategy:
        cfg = MomentumConfig(
            rsi_period=params["rsi_period"],
            macd_fast=params["macd_fast"],
            macd_slow=params["macd_slow"],
            macd_signal=params["macd_signal"],
            rsi_overbought=params["rsi_overbought"],
            rsi_oversold=params["rsi_oversold"],
        )
        return MomentumStrategy(cfg, app_config.risk)

    # ------------------------------------------------------------------
    # Core optimization loop
    # ------------------------------------------------------------------

    def optimize_strategy(
        self,
        strategy_name: str,
        bars: Dict,
        app_config: AppConfig,
        grid: Dict[str, List[Any]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> OptimizationResult:
        """Run grid search and return the best parameter set.

        Args:
            strategy_name: ``"mean_reversion"`` or ``"momentum"``.
            bars: Historical OHLCV data keyed by symbol.
            app_config: Current :class:`AppConfig`.
            grid: Parameter grid dict.
            start_date / end_date: Date range for backtesting.
        """
        if strategy_name not in ("mean_reversion", "momentum"):
            raise ValueError(f"Unknown strategy: {strategy_name!r}. Choose 'mean_reversion' or 'momentum'.")

        best_sharpe = float("-inf")
        best_params: Dict[str, Any] = {}
        all_results: List[Dict[str, Any]] = []
        combos = list(self._param_combinations(grid))
        logger.info(
            "Optimizing %s over %d parameter combinations…",
            strategy_name,
            len(combos),
        )

        for params in combos:
            try:
                if strategy_name == "mean_reversion":
                    strategy = self._build_mr_strategy(params, app_config)
                elif strategy_name == "momentum":
                    strategy = self._build_mom_strategy(params, app_config)
                else:
                    raise ValueError(f"Unknown strategy: {strategy_name}")

                bt = Backtester(strategy, app_config, initial_capital=100_000.0)
                strategy_bars = {
                    sym: bars[sym]
                    for sym in (strategy.required_symbols() or list(bars.keys()))
                    if sym in bars
                }
                if not strategy_bars:
                    continue

                result = bt.run(strategy_bars, start_date=start_date, end_date=end_date)
                sharpe = result.sharpe_ratio
                record = {**params, "sharpe": sharpe, "total_return_pct": result.total_return_pct}
                all_results.append(record)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = dict(params)

            except Exception as exc:
                logger.debug("Params %s failed: %s", params, exc)

        logger.info("Best Sharpe for %s: %.4f — %s", strategy_name, best_sharpe, best_params)
        return OptimizationResult(
            strategy_name=strategy_name,
            best_params=best_params,
            best_sharpe=best_sharpe,
            total_combinations=len(combos),
            all_results=all_results,
        )

    # ------------------------------------------------------------------
    # .env writer
    # ------------------------------------------------------------------

    @staticmethod
    def write_env(params: Dict[str, Any], env_path: str = ".env") -> None:
        """Update (or create) the ``.env`` file with optimised parameters.

        Only the keys present in *params* are written / updated; all
        other lines in the file are preserved.
        """
        key_map = {
            # Mean reversion
            "lookback_period": "MR_LOOKBACK_PERIOD",
            "std_threshold": "MR_STD_THRESHOLD",
            # Momentum
            "rsi_period": "MOM_RSI_PERIOD",
            "macd_fast": "MOM_MACD_FAST",
            "macd_slow": "MOM_MACD_SLOW",
            "macd_signal": "MOM_MACD_SIGNAL",
            "rsi_overbought": "MOM_RSI_OVERBOUGHT",
            "rsi_oversold": "MOM_RSI_OVERSOLD",
        }
        env_updates = {key_map[k]: str(v) for k, v in params.items() if k in key_map}
        if not env_updates:
            return

        lines: List[str] = []
        updated_keys: set = set()

        if os.path.exists(env_path):
            with open(env_path, "r") as fh:
                for line in fh:
                    stripped = line.strip()
                    if "=" in stripped and not stripped.startswith("#"):
                        env_key = stripped.split("=", 1)[0].strip()
                        if env_key in env_updates:
                            lines.append(f"{env_key}={env_updates[env_key]}\n")
                            updated_keys.add(env_key)
                            continue
                    lines.append(line)

        # Append keys that were not already in the file
        for k, v in env_updates.items():
            if k not in updated_keys:
                lines.append(f"{k}={v}\n")

        with open(env_path, "w") as fh:
            fh.writelines(lines)

        logger.info("Wrote optimised parameters to %s: %s", env_path, env_updates)
