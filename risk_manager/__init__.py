"""Risk management module.

Handles:
- Position sizing
- Daily loss limits (circuit breaker)
- Drawdown protection
- Stop-loss and take-profit automation
- Trailing stop-loss
- Maximum open position limits
- Realized P&L journal enrichment
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

from alerts import get_logger
from config import RiskConfig
from strategies.base_strategy import Signal, SignalType

logger = get_logger("risk_manager")


class RiskManager:
    """Evaluates and filters trade signals based on risk parameters."""

    def __init__(self, config: RiskConfig, db_path: str = "trading.db") -> None:
        self.config = config
        self.db_path = db_path
        self._daily_pnl: float = 0.0
        self._peak_portfolio_value: float = 0.0
        # Trailing stop: symbol → highest price seen since entry
        self._trailing_highs: Dict[str, float] = {}
        self._init_db()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    reason TEXT,
                    pnl REAL DEFAULT 0,
                    order_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    trade_date TEXT PRIMARY KEY,
                    realized_pnl REAL DEFAULT 0,
                    starting_portfolio_value REAL DEFAULT 0
                )
                """
            )
            conn.commit()

    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        strategy: str,
        reason: str = "",
        order_id: str = "",
        pnl: float = 0.0,
    ) -> None:
        # Auto-compute realized P&L when selling by matching the most recent open BUY
        if side.lower() == "sell" and pnl == 0.0:
            pnl = self._compute_realized_pnl(symbol, price, qty)
            if pnl != 0.0:
                self.update_daily_pnl(pnl, 0.0)
                # Clear trailing high on close
                self._trailing_highs.pop(symbol, None)

        ts = datetime.now(tz=timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trades (timestamp, symbol, side, qty, price, strategy, reason, pnl, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, symbol, side, qty, price, strategy, reason, pnl, order_id),
            )
            conn.commit()
        logger.info(
            "Trade recorded: %s %s %.4f @ $%.2f (pnl=%.2f)", side, symbol, qty, price, pnl
        )

    def _compute_realized_pnl(self, symbol: str, sell_price: float, sell_qty: float) -> float:
        """Compute realized P&L for a sell by finding the matching buy cost basis."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT price, qty FROM trades
                WHERE symbol = ? AND side = 'buy'
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return 0.0
        avg_entry = float(row[0])
        qty = float(row[1]) if sell_qty <= 0 else sell_qty
        return round((sell_price - avg_entry) * qty, 4)

    def get_daily_pnl(self) -> float:
        today = date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT realized_pnl FROM daily_pnl WHERE trade_date = ?", (today,)
            ).fetchone()
        return row[0] if row else 0.0

    def update_daily_pnl(self, pnl_delta: float, portfolio_value: float) -> None:
        today = date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT realized_pnl FROM daily_pnl WHERE trade_date = ?", (today,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE daily_pnl SET realized_pnl = realized_pnl + ? WHERE trade_date = ?",
                    (pnl_delta, today),
                )
            else:
                conn.execute(
                    "INSERT INTO daily_pnl (trade_date, realized_pnl, starting_portfolio_value) VALUES (?, ?, ?)",
                    (today, pnl_delta, portfolio_value),
                )
            conn.commit()
        self._daily_pnl = self.get_daily_pnl()

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _check_daily_loss_limit(self, portfolio_value: float) -> bool:
        """Return True if trading is allowed (daily loss limit not breached)."""
        daily_pnl = self.get_daily_pnl()
        max_daily_loss = portfolio_value * (self.config.max_daily_loss_pct / 100)
        if daily_pnl < -max_daily_loss:
            logger.warning(
                "Daily loss limit breached: pnl=%.2f, limit=-%.2f", daily_pnl, max_daily_loss
            )
            return False
        return True

    def _check_drawdown(self, portfolio_value: float) -> bool:
        """Return True if drawdown is within acceptable range."""
        if self._peak_portfolio_value == 0:
            self._peak_portfolio_value = portfolio_value
        self._peak_portfolio_value = max(self._peak_portfolio_value, portfolio_value)
        drawdown = (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value
        max_drawdown = self.config.max_daily_loss_pct / 100 * 3  # 3× daily loss = max drawdown
        if drawdown > max_drawdown:
            logger.warning(
                "Drawdown limit breached: drawdown=%.2f%%, limit=%.2f%%",
                drawdown * 100,
                max_drawdown * 100,
            )
            return False
        return True

    def _check_position_limit(self, open_positions: List[str], new_symbol: str) -> bool:
        """Return True if we can open another position."""
        if new_symbol in open_positions:
            return True  # Already have it, can add/reduce
        if len(open_positions) >= self.config.max_open_positions:
            logger.debug(
                "Max open positions reached (%d). Cannot add %s.",
                self.config.max_open_positions,
                new_symbol,
            )
            return False
        return True

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        risk_pct: Optional[float] = None,
    ) -> float:
        """Calculate the number of shares to buy based on risk parameters."""
        if price <= 0:
            return 0.0
        risk_pct = risk_pct or self.config.max_portfolio_risk_pct
        max_pct = min(risk_pct, self.config.max_position_size_pct)
        position_value = portfolio_value * (max_pct / 100)
        return round(position_value / price, 4)

    def compute_stop_loss_price(self, entry_price: float) -> float:
        return round(entry_price * (1 - self.config.stop_loss_pct / 100), 4)

    def compute_take_profit_price(self, entry_price: float) -> float:
        return round(entry_price * (1 + self.config.take_profit_pct / 100), 4)

    def update_trailing_high(self, symbol: str, current_price: float) -> None:
        """Update the trailing high-water mark for *symbol*."""
        prev = self._trailing_highs.get(symbol, 0.0)
        self._trailing_highs[symbol] = max(prev, current_price)

    def compute_trailing_stop_price(self, symbol: str, entry_price: float) -> Optional[float]:
        """Return the trailing stop price for *symbol*, or None if trailing stops are disabled.

        Uses ``TRAILING_STOP_PCT`` percentage below the highest price seen since
        entry.  Falls back to ``entry_price`` if no high-water mark has been
        recorded yet.
        """
        if self.config.trailing_stop_pct <= 0:
            return None
        high = self._trailing_highs.get(symbol, entry_price)
        return round(high * (1 - self.config.trailing_stop_pct / 100), 4)

    # ------------------------------------------------------------------
    # Signal filtering (main public interface)
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signals: List[Signal],
        portfolio_value: float,
        open_positions: List[str],
    ) -> List[Signal]:
        """Filter and adjust signals according to risk rules.

        Returns a list of approved signals (potentially with adjusted quantities).
        """
        if not self._check_daily_loss_limit(portfolio_value):
            logger.warning("All signals blocked: daily loss limit active")
            return []

        if not self._check_drawdown(portfolio_value):
            logger.warning("All signals blocked: drawdown limit active")
            return []

        approved: List[Signal] = []
        for sig in signals:
            if sig.is_buy:
                if not self._check_position_limit(open_positions, sig.symbol):
                    continue
                # Recalculate quantity using risk-based sizing
                max_qty = self.calculate_position_size(portfolio_value, sig.price)
                sig.quantity = min(sig.quantity, max_qty) if sig.quantity > 0 else max_qty
                if sig.quantity <= 0:
                    continue
            approved.append(sig)

        return approved

    def performance_summary(self, portfolio_value: float) -> Dict:
        """Return a dictionary of current risk metrics."""
        daily_pnl = self.get_daily_pnl()
        drawdown = 0.0
        if self._peak_portfolio_value > 0:
            drawdown = (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value

        return {
            "portfolio_value": portfolio_value,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": (daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0,
            "drawdown_pct": drawdown * 100,
            "peak_portfolio_value": self._peak_portfolio_value,
            "max_daily_loss_limit": portfolio_value * (self.config.max_daily_loss_pct / 100),
        }

    def generate_daily_report(self, portfolio_value: float) -> str:
        """Generate a formatted daily performance digest from the trade journal.

        Returns a multi-line string suitable for logging or emailing.
        """
        today = date.today()
        today_str = today.isoformat()
        week_start = (today - timedelta(days=7)).isoformat()
        month_start = (today - timedelta(days=30)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            today_trades = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp",
                    (f"{today_str}T00:00:00",),
                ).fetchall()
            ]
            weekly_pnl = (
                conn.execute(
                    "SELECT COALESCE(SUM(realized_pnl), 0) FROM daily_pnl WHERE trade_date >= ?",
                    (week_start,),
                ).fetchone()[0]
                or 0.0
            )
            monthly_pnl = (
                conn.execute(
                    "SELECT COALESCE(SUM(realized_pnl), 0) FROM daily_pnl WHERE trade_date >= ?",
                    (month_start,),
                ).fetchone()[0]
                or 0.0
            )

        daily_pnl = self.get_daily_pnl()
        perf = self.performance_summary(portfolio_value)

        def _win_rate(trades: List[Dict]) -> float:
            if not trades:
                return 0.0
            return sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades) * 100

        best = max(today_trades, key=lambda t: t.get("pnl", 0), default=None)
        worst = min(today_trades, key=lambda t: t.get("pnl", 0), default=None)

        strategy_pnl: Dict[str, float] = {}
        for t in today_trades:
            strat = t.get("strategy", "unknown")
            strategy_pnl[strat] = strategy_pnl.get(strat, 0.0) + t.get("pnl", 0.0)

        daily_pnl_pct = daily_pnl / portfolio_value * 100 if portfolio_value else 0.0
        sep = "=" * 55
        lines = [
            sep,
            f"  DAILY PERFORMANCE REPORT — {today_str}",
            sep,
            f"  Portfolio Value    : ${portfolio_value:>12,.2f}",
            f"  Daily P&L          : ${daily_pnl:>+12,.2f}  ({daily_pnl_pct:+.2f}%)",
            f"  Weekly P&L         : ${weekly_pnl:>+12,.2f}",
            f"  Monthly P&L        : ${monthly_pnl:>+12,.2f}",
            f"  Drawdown           : {perf['drawdown_pct']:>+10.2f}%",
            "",
            f"  Today's Trades     : {len(today_trades)}",
            f"  Win Rate           : {_win_rate(today_trades):>10.2f}%",
        ]
        if best:
            lines.append(
                f"  Best Trade         : {best['symbol']} {best['side']} +${best.get('pnl', 0):.2f}"
            )
        if worst and worst is not best:
            lines.append(
                f"  Worst Trade        : {worst['symbol']} {worst['side']} ${worst.get('pnl', 0):.2f}"
            )
        if strategy_pnl:
            lines.append("")
            lines.append("  Strategy Breakdown:")
            for strat, pnl in sorted(strategy_pnl.items(), key=lambda x: -x[1]):
                lines.append(f"    {strat:20s}: ${pnl:>+10,.2f}")
        lines.append(sep)
        return "\n".join(lines)
