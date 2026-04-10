"""Advanced Alpaca Trading System - Main entry point.

Supports three operating modes:
  trade   - Run the live/paper trading loop
  backtest - Backtest strategies against historical data
  status  - Print current account and position status
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from alerts import Alerter, setup_logging
from backtest_engine import Backtester
from config import AppConfig
from data_handler import DataHandler
from risk_manager import RiskManager
from strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    PairsTradingStrategy,
    Signal,
    SignalType,
)
from strategies.base_strategy import BaseStrategy


class TradingEngine:
    """Orchestrates all components for live/paper trading."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = setup_logging(config.log_level, config.log_dir)

        self.data = DataHandler(config.alpaca)
        self.risk = RiskManager(config.risk, db_path=config.db_path)
        self.alerter = Alerter(config.alerts)

        self.strategies: List[BaseStrategy] = [
            MeanReversionStrategy(config.mean_reversion, config.risk),
            MomentumStrategy(config.momentum, config.risk),
            PairsTradingStrategy(config.pairs_trading, config.risk),
        ]

        self._running = False

    # ------------------------------------------------------------------
    # Symbol collection
    # ------------------------------------------------------------------

    def _get_symbols(self) -> List[str]:
        """Collect all symbols needed across all strategies."""
        symbols = set(self.config.watchlist)
        for strategy in self.strategies:
            required = strategy.required_symbols()
            if required:
                symbols.update(required)
        return sorted(symbols)

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    def _run_cycle(self) -> None:
        """Execute one complete trading cycle."""
        symbols = self._get_symbols()
        self.logger.info("Running trading cycle for %d symbols", len(symbols))

        # Fetch data
        try:
            portfolio_value = self.data.get_portfolio_value()
            positions_list = self.data.get_positions()
            open_symbols = [p.symbol for p in positions_list]
        except Exception as exc:
            self.logger.error("Failed to fetch account data: %s", exc)
            self.alerter.error_alert(str(exc), "Account data fetch")
            return

        try:
            bars = self.data.get_bars(symbols, timeframe="1Day")
        except Exception as exc:
            self.logger.error("Failed to fetch bar data: %s", exc)
            self.alerter.error_alert(str(exc), "Bar data fetch")
            return

        try:
            current_prices = self.data.get_latest_prices(symbols)
        except Exception as exc:
            self.logger.warning("Could not fetch latest prices: %s. Using close prices.", exc)
            current_prices = {
                sym: float(bars[sym]["close"].iloc[-1])
                for sym in bars
                if not bars[sym].empty
            }

        # Check stop-loss and take-profit for open positions
        self._check_exits(positions_list, current_prices)

        # Generate signals from all strategies
        all_signals: List[Signal] = []
        for strategy in self.strategies:
            try:
                strategy_bars = {
                    sym: bars[sym]
                    for sym in bars
                    if sym in (strategy.required_symbols() or symbols)
                }
                signals = strategy.generate_signals(
                    strategy_bars,
                    current_prices,
                    portfolio_value,
                    open_symbols,
                )
                all_signals.extend(signals)
                self.logger.info(
                    "Strategy %s generated %d signals", strategy.name, len(signals)
                )
            except Exception as exc:
                self.logger.error("Strategy %s failed: %s", strategy.name, exc)

        # Filter by risk rules
        approved = self.risk.filter_signals(all_signals, portfolio_value, open_symbols)
        self.logger.info("%d/%d signals approved by risk manager", len(approved), len(all_signals))

        # Execute approved signals
        for sig in approved:
            self._execute_signal(sig, portfolio_value)

        # Log performance
        perf = self.risk.performance_summary(portfolio_value)
        self.logger.info(
            "Performance: portfolio=%.2f daily_pnl=%.2f (%.2f%%) drawdown=%.2f%%",
            perf["portfolio_value"],
            perf["daily_pnl"],
            perf["daily_pnl_pct"],
            perf["drawdown_pct"],
        )

    def _check_exits(self, positions_list, current_prices: Dict[str, float]) -> None:
        """Check if any open positions hit stop-loss or take-profit."""
        for pos in positions_list:
            sym = pos.symbol
            current_price = current_prices.get(sym)
            if current_price is None:
                continue

            avg_entry = float(pos.avg_entry_price)
            stop = self.risk.compute_stop_loss_price(avg_entry)
            target = self.risk.compute_take_profit_price(avg_entry)

            if current_price <= stop:
                self.logger.info(
                    "Stop-loss triggered for %s: price=%.2f stop=%.2f", sym, current_price, stop
                )
                self._close_position_market(sym, current_price, "stop_loss")
            elif current_price >= target:
                self.logger.info(
                    "Take-profit triggered for %s: price=%.2f target=%.2f",
                    sym,
                    current_price,
                    target,
                )
                self._close_position_market(sym, current_price, "take_profit")

    def _close_position_market(self, symbol: str, price: float, reason: str) -> None:
        try:
            self.data.close_position(symbol)
            self.alerter.trade_alert(symbol, "CLOSE", 0, price, "risk_manager", reason)
        except Exception as exc:
            self.logger.error("Failed to close position %s: %s", symbol, exc)

    def _execute_signal(self, sig: Signal, portfolio_value: float) -> None:
        """Execute a trade signal."""
        try:
            if sig.is_buy:
                order = self.data.submit_order(
                    symbol=sig.symbol,
                    qty=sig.quantity,
                    side="buy",
                )
                self.risk.record_trade(
                    symbol=sig.symbol,
                    side="buy",
                    qty=sig.quantity,
                    price=sig.price,
                    strategy=sig.metadata.get("strategy", "unknown"),
                    reason=sig.reason,
                    order_id=str(order.id) if order else "",
                )
                self.alerter.trade_alert(
                    sig.symbol, "BUY", sig.quantity, sig.price,
                    sig.metadata.get("strategy", "unknown"), sig.reason
                )
            elif sig.is_sell:
                order = self.data.submit_order(
                    symbol=sig.symbol,
                    qty=sig.quantity if sig.quantity > 0 else None,
                    side="sell",
                )
                self.risk.record_trade(
                    symbol=sig.symbol,
                    side="sell",
                    qty=sig.quantity,
                    price=sig.price,
                    strategy=sig.metadata.get("strategy", "unknown"),
                    reason=sig.reason,
                    order_id=str(order.id) if order else "",
                )
                self.alerter.trade_alert(
                    sig.symbol, "SELL", sig.quantity, sig.price,
                    sig.metadata.get("strategy", "unknown"), sig.reason
                )
        except Exception as exc:
            self.logger.error("Order execution failed for %s: %s", sig.symbol, exc)
            self.alerter.error_alert(str(exc), f"Order execution {sig.symbol}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _handle_signal(self, signum, frame) -> None:
        self.logger.info("Shutdown signal received, stopping…")
        self._running = False

    def run(self, interval_seconds: int = 300) -> None:
        """Start the continuous trading loop."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.logger.info("Trading engine started (mode=%s)", self.config.alpaca.trading_mode)
        self._running = True

        while self._running:
            try:
                self._run_cycle()
            except Exception as exc:
                self.logger.error("Unhandled error in trading cycle: %s", exc)
                self.alerter.error_alert(str(exc), "Trading cycle")
            finally:
                if self._running:
                    self.logger.info("Sleeping %ds until next cycle…", interval_seconds)
                    time.sleep(interval_seconds)

        self.logger.info("Trading engine stopped cleanly.")


# ------------------------------------------------------------------
# CLI commands
# ------------------------------------------------------------------

def cmd_status(config: AppConfig) -> None:
    """Print account status and open positions."""
    setup_logging(config.log_level, config.log_dir)
    data = DataHandler(config.alpaca)

    account = data.get_account()
    print(f"\n{'='*50}")
    print(f"  Account: {account.id}")
    print(f"  Portfolio Value : ${float(account.portfolio_value):>12,.2f}")
    print(f"  Cash            : ${float(account.cash):>12,.2f}")
    print(f"  Buying Power    : ${float(account.buying_power):>12,.2f}")
    print(f"{'='*50}")

    positions = data.get_positions()
    if positions:
        print(f"\n  Open Positions ({len(positions)}):")
        for p in positions:
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            print(
                f"    {p.symbol:8s} qty={float(p.qty):.4f}  "
                f"entry=${float(p.avg_entry_price):.2f}  "
                f"current=${float(p.current_price):.2f}  "
                f"PnL=${pnl:.2f} ({pnl_pct:.2f}%)"
            )
    else:
        print("\n  No open positions.")
    print()


def cmd_backtest(config: AppConfig, days: int = 365) -> None:
    """Run backtests for all strategies and print results."""
    setup_logging(config.log_level, config.log_dir)
    data = DataHandler(config.alpaca)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)

    symbols: set = set(config.watchlist)
    # Add pairs symbols
    for a, b in config.pairs_trading.pairs:
        symbols.update([a, b])
    symbols = sorted(symbols)

    print(f"\nFetching {days} days of data for {len(symbols)} symbols…")
    bars = data.get_bars(symbols, timeframe="1Day", start=start, end=end)
    print(f"Got data for: {list(bars.keys())}")

    strategies: List[BaseStrategy] = [
        MeanReversionStrategy(config.mean_reversion, config.risk),
        MomentumStrategy(config.momentum, config.risk),
        PairsTradingStrategy(config.pairs_trading, config.risk),
    ]

    for strategy in strategies:
        bt = Backtester(strategy, config, initial_capital=100_000.0)
        required_syms = strategy.required_symbols() or symbols
        strategy_bars = {sym: bars[sym] for sym in required_syms if sym in bars}
        if not strategy_bars:
            print(f"\n[{strategy.name}] No data available, skipping.")
            continue
        try:
            result = bt.run(strategy_bars, start_date=start, end_date=end)
            result.print_summary()
        except Exception as exc:
            print(f"\n[{strategy.name}] Backtest failed: {exc}")


def cmd_trade(config: AppConfig, interval: int = 300) -> None:
    """Start the live/paper trading engine."""
    engine = TradingEngine(config)
    engine.run(interval_seconds=interval)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Advanced Alpaca Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # trade
    trade_parser = subparsers.add_parser("trade", help="Run the trading engine")
    trade_parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Cycle interval in seconds (default: 300)",
    )

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="Backtest strategies")
    bt_parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of historical days to test (default: 365)",
    )

    # status
    subparsers.add_parser("status", help="Print account and position status")

    args = parser.parse_args(argv)

    config = AppConfig()

    if not config.alpaca.api_key or not config.alpaca.secret_key:
        print(
            "ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment or .env file."
        )
        return 1

    if args.command == "trade":
        cmd_trade(config, interval=args.interval)
    elif args.command == "backtest":
        cmd_backtest(config, days=args.days)
    elif args.command == "status":
        cmd_status(config)
    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
