"""Pairs Trading strategy.

Identifies cointegrated asset pairs and exploits the mean-reverting spread
between them. When the z-score of the spread exceeds the entry threshold,
we go long the underperformer and short the outperformer.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from alerts import get_logger
from config import PairsTradingConfig, RiskConfig
from strategies.base_strategy import BaseStrategy, Signal, SignalType

logger = get_logger("strategy.pairs_trading")


def _ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Compute OLS hedge ratio (beta) of y ~ x."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, _, _, _, _ = stats.linregress(x, y)
    return float(slope)


def _spread_z_score(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
    lookback: int,
) -> pd.Series:
    """Return the rolling z-score of the spread price_a - hedge_ratio * price_b."""
    spread = price_a - hedge_ratio * price_b
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()
    return (spread - rolling_mean) / rolling_std.replace(0, np.nan)


class PairsTradingStrategy(BaseStrategy):
    name = "pairs_trading"

    def __init__(
        self,
        config: PairsTradingConfig,
        risk_config: RiskConfig,
    ) -> None:
        self.config = config
        self.risk_config = risk_config
        # Track open pair positions: key = "A:B", value = ("long"|"short", hedge_ratio)
        self._open_pairs: Dict[str, Tuple[str, float]] = {}

    def required_bars(self) -> int:
        return self.config.lookback + 10

    def required_symbols(self) -> Optional[List[str]]:
        symbols = set()
        for a, b in self.config.pairs:
            symbols.add(a)
            symbols.add(b)
        return list(symbols)

    def _analyse_pair(
        self,
        sym_a: str,
        sym_b: str,
        bars: Dict[str, pd.DataFrame],
        portfolio_value: float,
    ) -> List[Signal]:
        """Generate signals for a single pair."""
        if sym_a not in bars or sym_b not in bars:
            return []

        df_a = bars[sym_a]
        df_b = bars[sym_b]

        if df_a.empty or df_b.empty:
            return []

        # Align on index
        price_a = df_a["close"].rename(sym_a)
        price_b = df_b["close"].rename(sym_b)
        combined = pd.concat([price_a, price_b], axis=1).dropna()

        if len(combined) < self.config.lookback:
            return []

        pa = combined[sym_a]
        pb = combined[sym_b]

        hedge_ratio = _ols_hedge_ratio(pa, pb)
        z_scores = _spread_z_score(pa, pb, hedge_ratio, self.config.lookback)
        z_now = float(z_scores.iloc[-1])

        if pd.isna(z_now):
            return []

        current_a = float(pa.iloc[-1])
        current_b = float(pb.iloc[-1])

        max_pos_value = portfolio_value * (self.risk_config.max_position_size_pct / 100) / 2
        qty_a = round(max_pos_value / current_a, 4) if current_a > 0 else 0
        qty_b = round(max_pos_value / current_b, 4) if current_b > 0 else 0

        pair_key = f"{sym_a}:{sym_b}"
        signals: List[Signal] = []

        pair_meta = {"z_score": z_now, "hedge_ratio": hedge_ratio}

        if pair_key not in self._open_pairs:
            # Entry signals
            if z_now > self.config.z_score_entry:
                # Spread too high: short A, long B
                if qty_a > 0:
                    signals.append(Signal(
                        symbol=sym_a,
                        signal_type=SignalType.SELL,
                        price=current_a,
                        quantity=qty_a,
                        reason=f"Pairs: z-score={z_now:.2f} > {self.config.z_score_entry} → short {sym_a}",
                        confidence=min(abs(z_now) / (self.config.z_score_entry * 2), 1.0),
                        metadata=pair_meta,
                    ))
                if qty_b > 0:
                    signals.append(Signal(
                        symbol=sym_b,
                        signal_type=SignalType.BUY,
                        price=current_b,
                        quantity=qty_b,
                        reason=f"Pairs: z-score={z_now:.2f} > {self.config.z_score_entry} → long {sym_b}",
                        confidence=min(abs(z_now) / (self.config.z_score_entry * 2), 1.0),
                        metadata=pair_meta,
                    ))
                if signals:
                    self._open_pairs[pair_key] = ("short_a", hedge_ratio)

            elif z_now < -self.config.z_score_entry:
                # Spread too low: long A, short B
                if qty_a > 0:
                    signals.append(Signal(
                        symbol=sym_a,
                        signal_type=SignalType.BUY,
                        price=current_a,
                        quantity=qty_a,
                        reason=f"Pairs: z-score={z_now:.2f} < -{self.config.z_score_entry} → long {sym_a}",
                        confidence=min(abs(z_now) / (self.config.z_score_entry * 2), 1.0),
                        metadata=pair_meta,
                    ))
                if qty_b > 0:
                    signals.append(Signal(
                        symbol=sym_b,
                        signal_type=SignalType.SELL,
                        price=current_b,
                        quantity=qty_b,
                        reason=f"Pairs: z-score={z_now:.2f} < -{self.config.z_score_entry} → short {sym_b}",
                        confidence=min(abs(z_now) / (self.config.z_score_entry * 2), 1.0),
                        metadata=pair_meta,
                    ))
                if signals:
                    self._open_pairs[pair_key] = ("long_a", hedge_ratio)
        else:
            # Check exit condition
            if abs(z_now) <= self.config.z_score_exit:
                direction, _ = self._open_pairs[pair_key]
                # Close both legs
                signals.append(Signal(
                    symbol=sym_a,
                    signal_type=SignalType.SELL if direction == "long_a" else SignalType.BUY,
                    price=current_a,
                    quantity=0.0,
                    reason=f"Pairs exit: z-score={z_now:.2f} reverted",
                    confidence=1.0,
                    metadata=pair_meta,
                ))
                signals.append(Signal(
                    symbol=sym_b,
                    signal_type=SignalType.BUY if direction == "long_a" else SignalType.SELL,
                    price=current_b,
                    quantity=0.0,
                    reason=f"Pairs exit: z-score={z_now:.2f} reverted",
                    confidence=1.0,
                    metadata=pair_meta,
                ))
                del self._open_pairs[pair_key]

        return signals

    def generate_signals(
        self,
        bars: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        portfolio_value: float,
        open_positions: List[str],
    ) -> List[Signal]:
        signals: List[Signal] = []
        for sym_a, sym_b in self.config.pairs:
            try:
                pair_signals = self._analyse_pair(sym_a, sym_b, bars, portfolio_value)
                for sig in pair_signals:
                    signals.append(sig)
                    logger.debug("Signal: %s", sig)
            except Exception as exc:
                logger.error(
                    "Error generating pairs signal for %s:%s: %s", sym_a, sym_b, exc
                )
        return signals
