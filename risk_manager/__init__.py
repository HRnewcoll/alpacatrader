"""Risk management module.

Handles:
- Position sizing
- Daily loss limits (circuit breaker)
- Drawdown protection
- Stop-loss and take-profit automation
- Maximum open position limits
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
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
