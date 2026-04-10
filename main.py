"""Advanced Alpaca Trading System - Main entry point.

Supports five operating modes:
  trade    - Run the live/paper trading loop (market-hours aware)
  backtest - Backtest strategies against historical data
  status  - Print current account and position status

Advanced Features:
  - Market regime detection for adaptive strategy selection
  - Volatility breakout strategy
  - Multi-timeframe analysis
  - Dynamic parameter adjustment based on market conditions
  - Enhanced performance analytics
  status   - Print current account and position status
  optimize - Grid-search strategy parameters and optionally write to .env
  validate - Check paper-trading metrics before going live
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from alerts import Alerter, setup_logging
from backtest_engine import Backtester
from config import AppConfig
from dashboard import Dashboard, push_signal
from data_handler import DataHandler
from market_regime import MarketRegimeDetector, MarketRegime
from news_handler import NewsHandler
from optimizer import WalkForwardOptimizer, MR_GRID, MOM_GRID
from rebalancer import PortfolioRebalancer
from risk_manager import RiskManager
from scheduler import MarketScheduler
from screener import MarketScreener
from strategies import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    PairsTradingStrategy,
    VWAPReversionStrategy,
    Signal,
    SignalType,
)
from strategies.base_strategy import BaseStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy, VolatilityBreakoutConfig


class TradingEngine:
    """Orchestrates all components for live/paper trading."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = setup_logging(config.log_level, config.log_dir)

        self.data = DataHandler(config.alpaca)
        self.risk = RiskManager(config.risk, db_path=config.db_path)
        self.alerter = Alerter(config.alerts)
        
        # Market regime detector for adaptive strategy selection
        self.regime_detector = MarketRegimeDetector()
        self._current_regime: Optional[MarketRegime] = None
        self._regime_recommendations: Dict = {}
        self.scheduler = MarketScheduler(config.scheduler)
        self.news = NewsHandler(config.alpaca, config.news)
        self.rebalancer = PortfolioRebalancer(config.rebalancer, config.risk)
        self.screener = MarketScreener(config.alpaca, config.screener)
        self.dashboard = Dashboard(
            config.dashboard,
            db_path=config.db_path,
            get_positions_fn=self.data.get_positions,
            get_performance_fn=lambda: self.risk.performance_summary(
                self._last_portfolio_value
            ),
        )

        self.strategies: List[BaseStrategy] = [
            MeanReversionStrategy(config.mean_reversion, config.risk),
            MomentumStrategy(config.momentum, config.risk),
            PairsTradingStrategy(config.pairs_trading, config.risk),
            VolatilityBreakoutStrategy(
                VolatilityBreakoutConfig(),
                config.risk
            ),
            BreakoutStrategy(config.breakout, config.risk),
            VWAPReversionStrategy(config.vwap_reversion, config.risk),
        ]

        self._running = False
        self._last_portfolio_value: float = 0.0
        self._dynamic_watchlist: List[str] = list(config.watchlist)
        self._last_report_date: Optional[str] = None
        # Watchdog: timestamp of last successful data fetch
        self._last_successful_fetch: float = time.time()
        self._watchdog_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Symbol collection
    # ------------------------------------------------------------------

    def _get_symbols(self) -> List[str]:
        """Collect all symbols needed across all strategies."""
        symbols = set(self._dynamic_watchlist)
        for strategy in self.strategies:
            required = strategy.required_symbols()
            if required:
                symbols.update(required)
        return sorted(symbols)

    # ------------------------------------------------------------------
    # Pre-market screener
    # ------------------------------------------------------------------

    def _run_premarket_screener(self) -> None:
        """Fetch bars for the screener universe and update the watchlist."""
        if not self.config.screener.enabled:
            return
        try:
            universe = self.config.screener.universe or self.config.watchlist
            end = datetime.now(tz=timezone.utc)
            start = end - timedelta(days=self.config.screener.lookback_days + 5)
            bars = self.data.get_bars(universe, timeframe="1Day", start=start, end=end)
            selected = self.screener.get_screened_symbols(bars, top_n=self.config.screener.top_n)
            if selected:
                # Always keep pairs-trading symbols in the list
                pairs_syms = set()
                for a, b in self.config.pairs_trading.pairs:
                    pairs_syms.update([a, b])
                merged = list(dict.fromkeys(selected + sorted(pairs_syms)))
                self._dynamic_watchlist = merged
                self.logger.info(
                    "Screener updated watchlist (%d symbols): %s...",
                    len(merged),
                    merged[:8],
                )
        except Exception as exc:
            self.logger.warning("Pre-market screener failed: %s", exc)

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    def _maybe_send_daily_report(self) -> None:
        """Send the daily performance report once after market close."""
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if self._last_report_date == today:
            return
        try:
            report = self.risk.generate_daily_report(self._last_portfolio_value)
            self.logger.info("\n%s", report)
            self.alerter.alert("Daily Performance Report", report, send_email=True)
            self._last_report_date = today
        except Exception as exc:
            self.logger.warning("Daily report generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    def _run_cycle(self, closeout: bool = False) -> None:
        """Execute one complete trading cycle.

        Args:
            closeout: When True, skip new BUY signals (end-of-day mode).
        """
        symbols = self._get_symbols()
        self.logger.info(
            "Running %s cycle for %d symbols",
            "close-out" if closeout else "trading",
            len(symbols),
        )

        # Fetch data
        try:
            portfolio_value = self.data.get_portfolio_value()
            self._last_portfolio_value = portfolio_value
            positions_list = self.data.get_positions()
            open_symbols = [p.symbol for p in positions_list]
            self._last_successful_fetch = time.time()
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

        # Detect market regime and adjust strategy weights
        self._update_regime_analysis(bars, symbols)

        # Check stop-loss and take-profit for open positions
        self._check_exits(positions_list, current_prices)

        # Generate signals from all strategies (with regime-based filtering)
        # Portfolio rebalancer
        if self.rebalancer.is_due():
            rebalance_signals = self.rebalancer.generate_rebalance_signals(
                positions_list, current_prices, portfolio_value
            )
            self.rebalancer.mark_checked()
            for sig in rebalance_signals:
                self._execute_signal(sig, portfolio_value)

        # In close-out mode, skip generating new BUY signals
        if closeout:
            self.logger.info("Close-out mode: skipping new signal generation.")
            return

        # Fetch news sentiment
        sentiment: Dict[str, float] = {}
        if self.config.news.enabled:
            try:
                sentiment = self.news.get_sentiment_scores(symbols)
            except Exception as exc:
                self.logger.warning("News sentiment fetch failed: %s", exc)

        # Fetch bid-ask snapshots for slippage guard (optional)
        snapshots: Dict[str, Dict] = {}
        if self.config.risk.max_bid_ask_spread_pct > 0:
            try:
                snapshots = self.data.get_snapshots(symbols)
            except Exception as exc:
                self.logger.warning("Snapshot fetch failed (slippage guard disabled): %s", exc)

        # Generate signals from all strategies
        all_signals: List[Signal] = []
        for strategy in self.strategies:
            # Skip strategies that are not recommended for current regime
            if self._should_skip_strategy(strategy.name):
                self.logger.debug("Skipping strategy %s based on regime", strategy.name)
                continue
                
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
                
                # Apply regime-based confidence adjustment
                signals = self._adjust_signal_confidence(signals)
                
                all_signals.extend(signals)
                self.logger.info(
                    "Strategy %s generated %d signals", strategy.name, len(signals)
                )
            except Exception as exc:
                self.logger.error("Strategy %s failed: %s", strategy.name, exc)

        # Apply multi-timeframe confirmation filter (suppress BUY in bearish daily trend)
        if self.config.multi_timeframe.enabled:
            all_signals = self._apply_mtf_filter(all_signals, bars)

        # Apply news sentiment filter
        if sentiment:
            all_signals = self.news.filter_signals(all_signals, sentiment)

        # Apply bid-ask slippage guard: drop BUYs where spread is too wide
        if snapshots and self.config.risk.max_bid_ask_spread_pct > 0:
            all_signals = self._apply_slippage_guard(all_signals, snapshots)

        # Filter by risk rules
        approved = self.risk.filter_signals(all_signals, portfolio_value, open_symbols)
        self.logger.info(
            "%d/%d signals approved by risk manager", len(approved), len(all_signals)
        )

        # Execute approved signals
        for sig in approved:
            self._execute_signal(sig, portfolio_value)

        # Log performance and regime
        # Push to dashboard signal feed
        for sig in approved:
            push_signal(
                symbol=sig.symbol,
                action=sig.signal_type.value.upper(),
                price=sig.price,
                strategy=sig.metadata.get("strategy", "unknown"),
                reason=sig.reason,
            )

        # Log performance
        perf = self.risk.performance_summary(portfolio_value)
        regime_str = self._current_regime.value if self._current_regime else "unknown"
        self.logger.info(
            "Performance: portfolio=%.2f daily_pnl=%.2f (%.2f%%) drawdown=%.2f%% regime=%s",
            perf["portfolio_value"],
            perf["daily_pnl"],
            perf["daily_pnl_pct"],
            perf["drawdown_pct"],
            regime_str,
        )

    def _check_exits(self, positions_list, current_prices: Dict[str, float]) -> None:
        """Check if any open positions hit stop-loss, trailing stop, or take-profit."""
        for pos in positions_list:
            sym = pos.symbol
            current_price = current_prices.get(sym)
            if current_price is None:
                continue

            avg_entry = float(pos.avg_entry_price)

            # Update trailing high-water mark
            self.risk.update_trailing_high(sym, current_price)

            # Determine effective stop: trailing stop takes priority when enabled
            trailing_stop = self.risk.compute_trailing_stop_price(sym, avg_entry)
            stop = trailing_stop if trailing_stop is not None else self.risk.compute_stop_loss_price(avg_entry)
            target = self.risk.compute_take_profit_price(avg_entry)

            if current_price <= stop:
                stop_type = "trailing_stop" if trailing_stop is not None else "stop_loss"
                self.logger.info(
                    "%s triggered for %s: price=%.2f stop=%.2f",
                    stop_type,
                    sym,
                    current_price,
                    stop,
                )
                self._close_position_market(sym, current_price, stop_type)
            elif current_price >= target:
                self.logger.info(
                    "Take-profit triggered for %s: price=%.2f target=%.2f",
                    sym,
                    current_price,
                    target,
                )
                self._close_position_market(sym, current_price, "take_profit")

    def _update_regime_analysis(self, bars: Dict[str, pd.DataFrame], symbols: List[str]) -> None:
        """Update market regime analysis and strategy recommendations."""
        try:
            # Use first symbol with sufficient data as regime indicator
            for symbol in symbols[:5]:  # Check up to 5 symbols
                if symbol in bars and len(bars[symbol]) > 100:
                    regime_metrics = self.regime_detector.detect_regime(symbol, bars[symbol])
                    if regime_metrics and regime_metrics.confidence > 0.6:
                        self._current_regime = regime_metrics.regime
                        self._regime_recommendations = self.regime_detector.get_regime_recommendation(regime_metrics)
                        self.logger.info(
                            "Market regime detected: %s (confidence=%.2f) based on %s",
                            self._current_regime.value,
                            regime_metrics.confidence,
                            symbol
                        )
                        return
            
            # Default regime if no clear signal
            self._current_regime = MarketRegime.NEUTRAL
            self._regime_recommendations = {}
        except Exception as exc:
            self.logger.warning("Failed to detect market regime: %s", exc)
            self._current_regime = MarketRegime.NEUTRAL
            self._regime_recommendations = {}

    def _should_skip_strategy(self, strategy_name: str) -> bool:
        """Determine if a strategy should be skipped based on regime."""
        if not self._regime_recommendations:
            return False
        avoid = self._regime_recommendations.get("avoid_strategies", [])
        return strategy_name in avoid

    def _adjust_signal_confidence(self, signals: List[Signal]) -> List[Signal]:
        """Adjust signal confidence based on regime recommendations."""
        if not self._regime_recommendations:
            return signals
        
        risk_adj = self._regime_recommendations.get("risk_adjustment", 1.0)
        
        adjusted = []
        for sig in signals:
            # Create a copy with adjusted confidence
            new_sig = Signal(
                symbol=sig.symbol,
                signal_type=sig.signal_type,
                price=sig.price,
                quantity=sig.quantity,
                reason=sig.reason,
                confidence=min(sig.confidence * risk_adj, 1.0),
                metadata={**sig.metadata, "regime": self._current_regime.value if self._current_regime else "unknown"},
            )
            adjusted.append(new_sig)
        
        return adjusted

    def _close_position_market(self, symbol: str, price: float, reason: str) -> None:
        try:
            self.data.close_position(symbol)
            self.alerter.trade_alert(symbol, "CLOSE", 0, price, "risk_manager", reason)
        except Exception as exc:
            self.logger.error("Failed to close position %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Multi-timeframe confirmation filter
    # ------------------------------------------------------------------

    def _apply_mtf_filter(
        self, signals: List[Signal], bars: Dict[str, "pd.DataFrame"]
    ) -> List[Signal]:
        """Suppress BUY signals when the daily trend is bearish.

        The daily trend is considered bearish when the current close is below
        the *MTF_TREND_LOOKBACK_DAYS*-day simple moving average.
        """
        import pandas as pd

        lookback = self.config.multi_timeframe.trend_lookback_days
        filtered: List[Signal] = []
        for sig in signals:
            if not sig.is_buy:
                filtered.append(sig)
                continue
            df = bars.get(sig.symbol)
            if df is None or df.empty or len(df) < lookback:
                filtered.append(sig)
                continue
            sma = float(df["close"].rolling(lookback).mean().iloc[-1])
            current = float(df["close"].iloc[-1])
            if current < sma:
                self.logger.debug(
                    "MTF filter: suppressing BUY %s (price %.2f < SMA %.2f)",
                    sig.symbol,
                    current,
                    sma,
                )
            else:
                filtered.append(sig)
        return filtered

    # ------------------------------------------------------------------
    # Bid-ask slippage guard
    # ------------------------------------------------------------------

    def _apply_slippage_guard(
        self, signals: List[Signal], snapshots: Dict[str, Dict]
    ) -> List[Signal]:
        """Drop BUY signals where the bid-ask spread exceeds the configured maximum."""
        max_spread = self.config.risk.max_bid_ask_spread_pct
        filtered: List[Signal] = []
        for sig in signals:
            if not sig.is_buy:
                filtered.append(sig)
                continue
            snap = snapshots.get(sig.symbol)
            if snap is None:
                filtered.append(sig)
                continue
            spread_pct = snap.get("bid_ask_spread_pct", 0.0)
            if spread_pct > max_spread:
                self.logger.info(
                    "Slippage guard: dropping BUY %s (spread=%.3f%% > max=%.3f%%)",
                    sig.symbol,
                    spread_pct,
                    max_spread,
                )
            else:
                filtered.append(sig)
        return filtered

    # ------------------------------------------------------------------
    # Reconnect watchdog
    # ------------------------------------------------------------------

    def _start_watchdog(self, stale_seconds: int = 600) -> None:
        """Start a background thread that reinitialises the data client if it goes stale."""

        def _watchdog():
            while self._running:
                time.sleep(60)
                gap = time.time() - self._last_successful_fetch
                if gap > stale_seconds:
                    self.logger.warning(
                        "Watchdog: no successful data fetch for %.0fs — reinitialising clients",
                        gap,
                    )
                    try:
                        self.data.reinit_clients()
                        self._last_successful_fetch = time.time()
                        self.logger.info("Watchdog: clients reinitialised successfully")
                    except Exception as exc:
                        self.logger.error("Watchdog: reinit failed: %s", exc)

        self._watchdog_thread = threading.Thread(
            target=_watchdog, name="watchdog", daemon=True
        )
        self._watchdog_thread.start()
        self.logger.info("Reconnect watchdog started (stale threshold=%ds)", stale_seconds)

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
                    sig.symbol,
                    "BUY",
                    sig.quantity,
                    sig.price,
                    sig.metadata.get("strategy", "unknown"),
                    sig.reason,
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
                    sig.symbol,
                    "SELL",
                    sig.quantity,
                    sig.price,
                    sig.metadata.get("strategy", "unknown"),
                    sig.reason,
                )
        except Exception as exc:
            self.logger.error("Order execution failed for %s: %s", sig.symbol, exc)
            self.alerter.error_alert(str(exc), f"Order execution {sig.symbol}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _handle_signal(self, signum, frame) -> None:
        self.logger.info("Shutdown signal received, stopping...")
        self._running = False

    def run(self, interval_seconds: int = 300) -> None:
        """Start the continuous trading loop.

        When ``USE_MARKET_HOURS=true`` (the default) the loop is
        market-hours aware:
          - Sleeps until the NYSE opens (minus the warmup window).
          - Runs a pre-market screener on wake-up.
          - Runs a close-out cycle near market close then sleeps.
          - Sends the daily performance report after close.
        """
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.logger.info("Trading engine started (mode=%s)", self.config.alpaca.trading_mode)

        # Start the web dashboard in the background if enabled
        self.dashboard.start()

        self._running = True
        # Start reconnect watchdog
        self._start_watchdog()

        _warmup_done_today: Optional[str] = None

        while self._running:
            now = datetime.now(tz=timezone.utc)
            today_str = now.strftime("%Y-%m-%d")

            # Market-hours gate
            if self.config.scheduler.use_market_hours:
                if not self.scheduler.is_market_open(now):
                    # Send daily report once, right after close
                    self._maybe_send_daily_report()
                    self.scheduler.sleep_until_open(now)
                    # Run pre-market screener on wake-up (once per day)
                    if _warmup_done_today != today_str:
                        self._run_premarket_screener()
                        _warmup_done_today = today_str
                    continue

                # Near close: run close-out cycle then sleep past close
                if self.scheduler.is_near_close(now):
                    self.logger.info("Approaching market close - running close-out cycle.")
                    try:
                        self._run_cycle(closeout=True)
                    except Exception as exc:
                        self.logger.error("Close-out cycle error: %s", exc)
                    if self._running:
                        time.sleep(15 * 60)
                    continue

            # Normal cycle
            try:
                self._run_cycle()
            except Exception as exc:
                self.logger.error("Unhandled error in trading cycle: %s", exc)
                self.alerter.error_alert(str(exc), "Trading cycle")
            finally:
                if self._running:
                    self.logger.info("Sleeping %ds until next cycle...", interval_seconds)
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
    for a, b in config.pairs_trading.pairs:
        symbols.update([a, b])
    symbols = sorted(symbols)

    print(f"\nFetching {days} days of data for {len(symbols)} symbols...")
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


def cmd_optimize(config: AppConfig, months: int = 6, write_env: bool = False) -> None:
    """Grid-search strategy parameters over historical data."""
    setup_logging(config.log_level, config.log_dir)
    data = DataHandler(config.alpaca)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=months * 30)

    symbols: set = set(config.watchlist)
    for a, b in config.pairs_trading.pairs:
        symbols.update([a, b])
    symbols_list = sorted(symbols)

    print(f"\nFetching {months} months of data for {len(symbols_list)} symbols...")
    bars = data.get_bars(symbols_list, timeframe="1Day", start=start, end=end)
    print(f"Got data for {len(bars)} symbols.\n")

    optimizer = WalkForwardOptimizer(config.optimizer)

    mr_result = optimizer.optimize_strategy(
        "mean_reversion", bars, config, MR_GRID, start_date=start, end_date=end
    )
    mr_result.print_summary()

    mom_result = optimizer.optimize_strategy(
        "momentum", bars, config, MOM_GRID, start_date=start, end_date=end
    )
    mom_result.print_summary()

    if write_env or config.optimizer.write_env:
        env_path = ".env"
        optimizer.write_env(mr_result.best_params, env_path)
        optimizer.write_env(mom_result.best_params, env_path)
        print(f"Optimised parameters written to {env_path}")


def cmd_trade(config: AppConfig, interval: int = 300) -> None:
    """Start the live/paper trading engine."""
    engine = TradingEngine(config)
    engine.run(interval_seconds=interval)


def cmd_validate(config: AppConfig, days: int = 30, min_trades: int = 20) -> int:
    """Gate paper-to-live promotion by validating SQLite trade history.

    Checks:
    - At least *min_trades* completed sell trades in the last *days* days.
    - Annualised Sharpe ratio ≥ 1.0 (from daily P&L).
    - Max drawdown ≤ 15 %.

    Returns 0 (pass) or 1 (fail).
    """
    import sqlite3
    import math

    setup_logging(config.log_level, config.log_dir)
    sep = "=" * 55

    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%d"
    )

    with sqlite3.connect(config.db_path) as conn:
        # Count completed sell trades
        trade_count = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE side='sell' AND timestamp >= ?",
            (cutoff + "T00:00:00",),
        ).fetchone()[0]

        # Daily P&L rows
        rows = conn.execute(
            "SELECT trade_date, realized_pnl, starting_portfolio_value "
            "FROM daily_pnl WHERE trade_date >= ? ORDER BY trade_date",
            (cutoff,),
        ).fetchall()

    print(f"\n{sep}")
    print(f"  PAPER-TO-LIVE VALIDATION — last {days} days")
    print(sep)

    # ---- Trade count check ----
    trade_ok = trade_count >= min_trades
    print(f"  Completed trades  : {trade_count:>5}  (min {min_trades})  {'✓' if trade_ok else '✗'}")

    # ---- Sharpe ratio (annualised) ----
    sharpe_ok = False
    sharpe = 0.0
    if rows:
        import statistics

        pnl_vals = [r[1] for r in rows]
        start_vals = [r[2] for r in rows if r[2] > 0]
        avg_start = statistics.mean(start_vals) if start_vals else 100_000.0
        daily_returns = [p / avg_start for p in pnl_vals]
        mean_ret = statistics.mean(daily_returns)
        std_ret = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0
        if std_ret == 0:
            # All returns identical: infinite Sharpe when positive, 0 when non-positive
            sharpe = float("inf") if mean_ret > 0 else 0.0
        else:
            sharpe = mean_ret / std_ret * math.sqrt(252)
        sharpe_ok = sharpe >= 1.0

    print(f"  Sharpe ratio      : {sharpe:>7.3f}  (min 1.0)  {'✓' if sharpe_ok else '✗'}")

    # ---- Max drawdown ----
    max_dd_ok = False
    max_dd_pct = 0.0
    if rows:
        peak = 0.0
        max_dd = 0.0
        equity = 0.0
        for _, pnl, start_val in rows:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / abs(peak) if peak != 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = max_dd * 100
        max_dd_ok = max_dd_pct <= 15.0

    print(f"  Max drawdown      : {max_dd_pct:>7.2f}%  (max 15.0%)  {'✓' if max_dd_ok else '✗'}")

    passed = trade_ok and sharpe_ok and max_dd_ok
    print(sep)
    if passed:
        print("  RESULT: PASS — safe to switch TRADING_MODE=live")
    else:
        print("  RESULT: FAIL — continue paper trading before going live")
    print(sep + "\n")

    return 0 if passed else 1


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
    trade_parser.add_argument(
        "--no-market-hours",
        action="store_true",
        default=False,
        help="Disable market-hours gating (run cycles 24/7)",
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

    # optimize
    opt_parser = subparsers.add_parser(
        "optimize", help="Grid-search strategy parameters"
    )
    opt_parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Months of history for optimization (default: 6)",
    )
    opt_parser.add_argument(
        "--write-env",
        action="store_true",
        default=False,
        help="Write best params back to .env",
    )

    # validate
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate paper-trading metrics before switching to live mode",
    )
    val_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of historical days to evaluate (default: 30)",
    )
    val_parser.add_argument(
        "--min-trades",
        type=int,
        default=20,
        help="Minimum number of completed trades required (default: 20)",
    )

    args = parser.parse_args(argv)

    config = AppConfig()

    if not config.alpaca.api_key or not config.alpaca.secret_key:
        print(
            "ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set "
            "in the environment or .env file."
        )
        return 1

    if args.command == "trade":
        if args.no_market_hours:
            config.scheduler.use_market_hours = False
        cmd_trade(config, interval=args.interval)
    elif args.command == "backtest":
        cmd_backtest(config, days=args.days)
    elif args.command == "status":
        cmd_status(config)
    elif args.command == "optimize":
        cmd_optimize(config, months=args.months, write_env=args.write_env)
    elif args.command == "validate":
        return cmd_validate(config, days=args.days, min_trades=args.min_trades)
    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
