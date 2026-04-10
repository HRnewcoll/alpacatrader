"""Momentum strategy using RSI and MACD indicators.

- Buys when RSI crosses up from oversold AND MACD line crosses above signal.
- Sells when RSI crosses into overbought territory OR MACD crosses below signal.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import ta

from alerts import get_logger
from config import MomentumConfig, RiskConfig
from strategies.base_strategy import BaseStrategy, Signal, SignalType

logger = get_logger("strategy.momentum")


def _safe_float(series: pd.Series, idx: int = -1) -> float:
    try:
        val = series.iloc[idx]
        return float(val) if not pd.isna(val) else 0.0
    except (IndexError, TypeError):
        return 0.0


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def __init__(self, config: MomentumConfig, risk_config: RiskConfig) -> None:
        self.config = config
        self.risk_config = risk_config

    def required_bars(self) -> int:
        return self.config.macd_slow + self.config.macd_signal + 10

    def _compute_indicators(
        self, df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        close = df["close"]
        rsi = ta.momentum.RSIIndicator(
            close=close, window=self.config.rsi_period
        ).rsi()
        macd_obj = ta.trend.MACD(
            close=close,
            window_slow=self.config.macd_slow,
            window_fast=self.config.macd_fast,
            window_sign=self.config.macd_signal,
        )
        return {
            "rsi": rsi,
            "macd": macd_obj.macd(),
            "macd_signal": macd_obj.macd_signal(),
            "macd_diff": macd_obj.macd_diff(),
        }

    def _compute_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        portfolio_value: float,
        has_position: bool,
    ) -> List[Signal]:
        if len(df) < self.required_bars():
            return []

        try:
            indicators = self._compute_indicators(df)
        except Exception as exc:
            logger.error("Indicator computation failed for %s: %s", symbol, exc)
            return []

        current_price = _safe_float(df["close"])
        rsi_now = _safe_float(indicators["rsi"])
        rsi_prev = _safe_float(indicators["rsi"], -2)
        macd_now = _safe_float(indicators["macd"])
        macd_signal_now = _safe_float(indicators["macd_signal"])
        macd_prev = _safe_float(indicators["macd"], -2)
        macd_signal_prev = _safe_float(indicators["macd_signal"], -2)

        if current_price <= 0:
            return []

        signals: List[Signal] = []

        # BUY: RSI rising from oversold AND MACD bullish crossover
        rsi_oversold_cross = (
            rsi_prev <= self.config.rsi_oversold
            and rsi_now > self.config.rsi_oversold
        )
        macd_bullish_cross = macd_prev < macd_signal_prev and macd_now >= macd_signal_now

        if rsi_oversold_cross and macd_bullish_cross and not has_position:
            max_position_value = portfolio_value * (
                self.risk_config.max_position_size_pct / 100
            )
            qty = round(max_position_value / current_price, 4)
            if qty > 0:
                confidence = min((self.config.rsi_oversold - rsi_prev + 1) / 20, 1.0)
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        quantity=qty,
                        reason=(
                            f"RSI crossed up from oversold ({rsi_prev:.1f}→{rsi_now:.1f}) "
                            f"and MACD bullish crossover"
                        ),
                        confidence=confidence,
                        metadata={
                            "rsi": rsi_now,
                            "macd": macd_now,
                            "macd_signal": macd_signal_now,
                        },
                    )
                )

        # SELL: RSI overbought OR MACD bearish crossover while holding
        rsi_overbought = rsi_now >= self.config.rsi_overbought
        macd_bearish_cross = macd_prev >= macd_signal_prev and macd_now < macd_signal_now

        if has_position and (rsi_overbought or macd_bearish_cross):
            reason_parts = []
            if rsi_overbought:
                reason_parts.append(f"RSI overbought ({rsi_now:.1f})")
            if macd_bearish_cross:
                reason_parts.append("MACD bearish crossover")
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    quantity=0.0,
                    reason=" + ".join(reason_parts),
                    confidence=0.9,
                    metadata={
                        "rsi": rsi_now,
                        "macd": macd_now,
                        "macd_signal": macd_signal_now,
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
                logger.error("Error generating momentum signal for %s: %s", symbol, exc)
        return signals
