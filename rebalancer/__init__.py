"""Portfolio rebalancer.

Checks whether any open position has grown to exceed the configured
``MAX_POSITION_SIZE_PCT`` target weight and generates sell signals to
bring it back in line.  A full rebalance cycle is run at most once per
``REBALANCER_CHECK_INTERVAL_HOURS``.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from alerts import get_logger
from config import RebalancerConfig, RiskConfig
from strategies.base_strategy import Signal, SignalType

logger = get_logger("rebalancer")


class PortfolioRebalancer:
    """Identifies overweight positions and generates trim signals."""

    def __init__(
        self, rebalancer_config: RebalancerConfig, risk_config: RiskConfig
    ) -> None:
        self.config = rebalancer_config
        self.risk_config = risk_config
        self._last_check_ts: float = 0.0

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def is_due(self, now: Optional[datetime] = None) -> bool:
        """Return True if the rebalance interval has elapsed."""
        if not self.config.enabled:
            return False
        elapsed = time.time() - self._last_check_ts
        return elapsed >= self.config.check_interval_hours * 3600

    def mark_checked(self) -> None:
        """Record the current time as the last rebalance check."""
        self._last_check_ts = time.time()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def needs_rebalancing(
        self,
        positions: List,
        current_prices: Dict[str, float],
        portfolio_value: float,
    ) -> List[Tuple[str, float, float]]:
        """Return a list of (symbol, current_weight_pct, excess_qty) tuples
        for positions that exceed the target weight.

        *positions* is the list returned by ``DataHandler.get_positions()``.
        Each item must expose ``.symbol``, ``.qty``, and optionally
        ``.current_price``.
        """
        if portfolio_value <= 0:
            return []

        target_pct = self.risk_config.max_position_size_pct
        overweight: List[Tuple[str, float, float]] = []

        for pos in positions:
            sym = pos.symbol
            price = current_prices.get(sym)
            if price is None:
                try:
                    price = float(pos.current_price)
                except (AttributeError, TypeError, ValueError):
                    continue

            qty = float(pos.qty)
            market_value = qty * price
            weight_pct = market_value / portfolio_value * 100

            if weight_pct > target_pct:
                target_value = portfolio_value * (target_pct / 100)
                excess_value = market_value - target_value
                excess_qty = round(excess_value / price, 4)
                overweight.append((sym, weight_pct, excess_qty))
                logger.info(
                    "Rebalance: %s is %.1f%% of portfolio (target ≤ %.1f%%). "
                    "Trim %.4f shares.",
                    sym,
                    weight_pct,
                    target_pct,
                    excess_qty,
                )

        return overweight

    def generate_rebalance_signals(
        self,
        positions: List,
        current_prices: Dict[str, float],
        portfolio_value: float,
    ) -> List[Signal]:
        """Return SELL signals to rebalance overweight positions."""
        overweight = self.needs_rebalancing(positions, current_prices, portfolio_value)
        signals: List[Signal] = []
        for sym, weight_pct, excess_qty in overweight:
            price = current_prices.get(sym, 0.0)
            if excess_qty <= 0:
                continue
            sig = Signal(
                symbol=sym,
                signal_type=SignalType.SELL,
                price=price,
                quantity=excess_qty,
                reason=f"rebalance: {weight_pct:.1f}% > {self.risk_config.max_position_size_pct:.1f}% target",
                metadata={"strategy": "rebalancer"},
            )
            signals.append(sig)
        return signals
