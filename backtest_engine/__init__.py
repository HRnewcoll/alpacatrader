"""Backtesting engine.

Runs strategies over historical data and computes performance metrics
including Sharpe ratio, max drawdown, win rate, and trade-by-trade analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alerts import get_logger
from config import AppConfig
from strategies.base_strategy import BaseStrategy, Signal, SignalType

logger = get_logger("backtester")


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: float
    qty: float
    side: str  # "long" or "short"
    strategy: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason_entry: str = ""
    reason_exit: str = ""


@dataclass
class BacktestResult:
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.pnl > 0)
        return winners / len(self.trades) * 100

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades) / len(self.trades)

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        return float(drawdown.min() * 100)

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        returns = self.equity_curve.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        # Annualise assuming daily data (252 trading days)
        return float((returns.mean() / returns.std()) * math.sqrt(252))

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def summary(self) -> Dict:
        return {
            "strategy": self.strategy_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "num_trades": self.num_trades,
            "win_rate_pct": round(self.win_rate, 2),
            "avg_pnl": round(self.avg_pnl, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'='*55}")
        print(f"  Backtest: {s['strategy']}")
        print(f"  Period:   {s['start_date'][:10]} → {s['end_date'][:10]}")
        print(f"{'='*55}")
        print(f"  Initial Capital : ${s['initial_capital']:>12,.2f}")
        print(f"  Final Capital   : ${s['final_capital']:>12,.2f}")
        print(f"  Total Return    : {s['total_return_pct']:>10.2f}%")
        print(f"  Num Trades      : {s['num_trades']:>12}")
        print(f"  Win Rate        : {s['win_rate_pct']:>10.2f}%")
        print(f"  Avg PnL/Trade   : ${s['avg_pnl']:>11,.2f}")
        print(f"  Max Drawdown    : {s['max_drawdown_pct']:>10.2f}%")
        print(f"  Sharpe Ratio    : {s['sharpe_ratio']:>12.4f}")
        print(f"  Profit Factor   : {s['profit_factor']:>12.4f}")
        print(f"{'='*55}\n")


class Backtester:
    """Simple event-driven backtester for single strategies."""

    def __init__(
        self,
        strategy: BaseStrategy,
        config: AppConfig,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.strategy = strategy
        self.config = config
        self.initial_capital = initial_capital

    def run(
        self,
        bars: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """Execute a backtest over *bars*.

        Args:
            bars: Dict mapping symbol → OHLCV DataFrame with a UTC DatetimeIndex.
            start_date: Optional start cutoff.
            end_date: Optional end cutoff.

        Returns:
            A :class:`BacktestResult` instance.
        """
        if not bars:
            raise ValueError("No bar data provided for backtesting")

        # Determine common dates across all symbols
        all_dates = sorted(
            set.intersection(*[set(df.index) for df in bars.values()])
        )
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date, tz="UTC")]

        if len(all_dates) < self.strategy.required_bars():
            raise ValueError(
                f"Not enough data: have {len(all_dates)} bars, need {self.strategy.required_bars()}"
            )

        capital = self.initial_capital
        positions: Dict[str, Tuple[float, float, datetime, str]] = {}
        # positions[symbol] = (qty, entry_price, entry_date, reason)
        completed_trades: List[Trade] = []
        equity_values: Dict[pd.Timestamp, float] = {}

        required = self.strategy.required_bars()

        for i, current_date in enumerate(all_dates):
            if i < required:
                equity_values[current_date] = capital
                continue

            # Slice bars up to and including current_date
            window_bars: Dict[str, pd.DataFrame] = {}
            for sym, df in bars.items():
                slice_df = df[df.index <= current_date].tail(required + 5)
                if not slice_df.empty:
                    window_bars[sym] = slice_df

            current_prices: Dict[str, float] = {
                sym: float(df["close"].iloc[-1])
                for sym, df in window_bars.items()
                if not df.empty
            }
            open_positions = list(positions.keys())

            # Mark current equity
            unrealised = sum(
                positions[sym][0] * current_prices.get(sym, positions[sym][1])
                for sym in positions
            )
            equity_values[current_date] = capital + unrealised

            # Generate signals
            try:
                signals = self.strategy.generate_signals(
                    window_bars,
                    current_prices,
                    equity_values[current_date],
                    open_positions,
                )
            except Exception as exc:
                logger.error("Signal generation error at %s: %s", current_date, exc)
                continue

            # Execute signals
            for sig in signals:
                price = current_prices.get(sig.symbol)
                if price is None:
                    continue

                if sig.is_buy and sig.symbol not in positions:
                    # Determine position size
                    max_pos = equity_values[current_date] * (
                        self.config.risk.max_position_size_pct / 100
                    )
                    qty = min(sig.quantity, max_pos / price) if sig.quantity > 0 else round(max_pos / price, 4)
                    cost = qty * price
                    if cost > capital:
                        qty = round(capital / price, 4)
                        cost = qty * price
                    if qty <= 0:
                        continue
                    capital -= cost
                    positions[sig.symbol] = (qty, price, current_date, sig.reason)

                elif sig.is_sell and sig.symbol in positions:
                    qty, entry_price, entry_date, entry_reason = positions.pop(sig.symbol)
                    proceeds = qty * price
                    capital += proceeds
                    pnl = proceeds - qty * entry_price
                    pnl_pct = pnl / (qty * entry_price) * 100 if entry_price > 0 else 0
                    completed_trades.append(Trade(
                        symbol=sig.symbol,
                        entry_date=entry_date,
                        exit_date=current_date,
                        entry_price=entry_price,
                        exit_price=price,
                        qty=qty,
                        side="long",
                        strategy=self.strategy.name,
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        reason_entry=entry_reason,
                        reason_exit=sig.reason,
                    ))

        # Close any remaining open positions at last price
        last_date = all_dates[-1] if all_dates else None
        for sym, (qty, entry_price, entry_date, entry_reason) in list(positions.items()):
            price = current_prices.get(sym, entry_price)
            proceeds = qty * price
            capital += proceeds
            pnl = proceeds - qty * entry_price
            pnl_pct = pnl / (qty * entry_price) * 100 if entry_price > 0 else 0
            completed_trades.append(Trade(
                symbol=sym,
                entry_date=entry_date,
                exit_date=last_date,
                entry_price=entry_price,
                exit_price=price,
                qty=qty,
                side="long",
                strategy=self.strategy.name,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                reason_entry=entry_reason,
                reason_exit="End of backtest",
            ))

        equity_curve = pd.Series(equity_values).sort_index()

        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=all_dates[0].to_pydatetime() if all_dates else datetime.now(tz=timezone.utc),
            end_date=all_dates[-1].to_pydatetime() if all_dates else datetime.now(tz=timezone.utc),
            initial_capital=self.initial_capital,
            final_capital=round(capital, 2),
            trades=completed_trades,
            equity_curve=equity_curve,
        )
