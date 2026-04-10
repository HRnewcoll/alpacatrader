"""Volume-Weighted Average Price (VWAP) Mean Reversion strategy.

Uses the rolling VWAP as the anchor price (instead of a simple rolling mean).
Buys when the price drops a configurable number of standard deviations below
the VWAP; sells when the price reverts to or above the VWAP.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from alerts import get_logger
from config import RiskConfig, VWAPReversionConfig
from strategies.base_strategy import BaseStrategy, Signal, SignalType

logger = get_logger("strategy.vwap_reversion")


def _rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute a rolling VWAP over *window* bars.

    VWAP = sum(typical_price × volume) / sum(volume)
    where typical_price = (high + low + close) / 3.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    vwap = tp_vol.rolling(window).sum() / df["volume"].rolling(window).sum()
    return vwap


class VWAPReversionStrategy(BaseStrategy):
    name = "vwap_reversion"

    def __init__(
        self, config: VWAPReversionConfig, risk_config: RiskConfig
    ) -> None:
        self.config = config
        self.risk_config = risk_config

    def required_bars(self) -> int:
        # spread_std requires 2 × lookback_days to be valid (vwap window + std window)
        return self.config.lookback_days * 2 + 2

    def _compute_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        portfolio_value: float,
        has_position: bool,
    ) -> List[Signal]:
        if len(df) < self.config.lookback_days:
            return []

        vwap = _rolling_vwap(df, self.config.lookback_days)
        close = df["close"]

        # Spread between close and VWAP, normalized by rolling std of close
        spread = close - vwap
        spread_std = spread.rolling(self.config.lookback_days).std()

        current_price = float(close.iloc[-1])
        current_vwap = float(vwap.iloc[-1])
        current_std = float(spread_std.iloc[-1])

        if pd.isna(current_vwap) or pd.isna(current_std) or current_std == 0:
            return []

        z_score = float(spread.iloc[-1]) / current_std

        signals: List[Signal] = []
        threshold = self.config.std_threshold

        if not has_position and z_score <= -threshold:
            # Price significantly below VWAP → buy (expect reversion upward)
            max_position_value = portfolio_value * (
                self.risk_config.max_position_size_pct / 100
            )
            qty = round(max_position_value / current_price, 4)
            if qty > 0:
                confidence = min(abs(z_score) / (threshold * 2), 1.0)
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        quantity=qty,
                        reason=(
                            f"VWAP reversion: price {current_price:.2f} is "
                            f"{abs(z_score):.2f}σ below VWAP {current_vwap:.2f}"
                        ),
                        confidence=confidence,
                        metadata={
                            "vwap": current_vwap,
                            "z_score": z_score,
                            "spread_std": current_std,
                        },
                    )
                )

        elif has_position and z_score >= 0:
            # Price has reverted to or above VWAP → take profit
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    quantity=0.0,
                    reason=(
                        f"VWAP reversion complete: price {current_price:.2f} "
                        f"at/above VWAP {current_vwap:.2f}"
                    ),
                    confidence=min(z_score / threshold + 0.5, 1.0),
                    metadata={
                        "vwap": current_vwap,
                        "z_score": z_score,
                    },
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
                logger.error(
                    "Error generating VWAP reversion signal for %s: %s", symbol, exc
                )
        return signals
