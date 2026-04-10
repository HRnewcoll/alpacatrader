"""Breakout / Gap strategy.

Buys when the price closes above the N-day rolling high on above-average
volume (classic momentum breakout).  Exits when the price falls back below
the rolling high (momentum exhaustion) or when the RiskManager stop-loss /
take-profit fires.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from alerts import get_logger
from config import BreakoutConfig, RiskConfig
from strategies.base_strategy import BaseStrategy, Signal, SignalType

logger = get_logger("strategy.breakout")


class BreakoutStrategy(BaseStrategy):
    name = "breakout"

    def __init__(self, config: BreakoutConfig, risk_config: RiskConfig) -> None:
        self.config = config
        self.risk_config = risk_config

    def required_bars(self) -> int:
        return self.config.lookback_days + 5

    def _compute_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        portfolio_value: float,
        has_position: bool,
    ) -> List[Signal]:
        if len(df) < self.config.lookback_days + 1:
            return []

        close = df["close"]
        volume = df["volume"]

        # Rolling high over *lookback_days* bars (excluding the current bar)
        rolling_high = close.iloc[:-1].rolling(self.config.lookback_days).max()
        prev_high = float(rolling_high.iloc[-1]) if not rolling_high.empty else float("nan")

        avg_volume = float(volume.iloc[:-1].rolling(self.config.lookback_days).mean().iloc[-1])
        current_price = float(close.iloc[-1])
        current_volume = float(volume.iloc[-1])

        if pd.isna(prev_high) or prev_high <= 0 or avg_volume <= 0:
            return []

        signals: List[Signal] = []

        if not has_position:
            # Entry: price breaks above prior N-day high on elevated volume
            if current_price > prev_high and current_volume >= self.config.volume_factor * avg_volume:
                max_position_value = portfolio_value * (
                    self.risk_config.max_position_size_pct / 100
                )
                qty = round(max_position_value / current_price, 4)
                if qty > 0:
                    vol_ratio = current_volume / avg_volume
                    confidence = min(vol_ratio / (self.config.volume_factor * 2), 1.0)
                    signals.append(
                        Signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            price=current_price,
                            quantity=qty,
                            reason=(
                                f"Breakout above {prev_high:.2f} "
                                f"(vol_ratio={vol_ratio:.2f}x)"
                            ),
                            confidence=confidence,
                            metadata={
                                "prev_high": prev_high,
                                "vol_ratio": vol_ratio,
                                "avg_volume": avg_volume,
                            },
                        )
                    )
        else:
            # Exit: price falls back below the rolling high → momentum exhaustion
            if current_price < prev_high:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        quantity=0.0,
                        reason=f"Breakout failed: price {current_price:.2f} < high {prev_high:.2f}",
                        confidence=0.9,
                        metadata={"prev_high": prev_high},
                    )
                )

        return signals

    def generate_signals(
        self,
        bars: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        portfolio_value: float,
        open_positions: List[str],
    ) -> List[Signal]:
        signals: List[Signal] = []
        for symbol, df in bars.items():
            if df.empty:
                continue
            try:
                has_position = symbol in open_positions
                sym_signals = self._compute_signals(
                    symbol, df, portfolio_value, has_position
                )
                for sig in sym_signals:
                    signals.append(sig)
                    logger.debug("Signal: %s", sig)
            except Exception as exc:
                logger.error("Error generating breakout signal for %s: %s", symbol, exc)
        return signals
