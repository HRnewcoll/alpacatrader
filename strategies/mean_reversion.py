"""Mean Reversion strategy.

Buys when price drops significantly below its rolling mean and sells when it
reverts. Uses Bollinger Bands (price vs. rolling mean ± N standard deviations).
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from alerts import get_logger
from config import MeanReversionConfig, RiskConfig
from strategies.base_strategy import BaseStrategy, Signal, SignalType

logger = get_logger("strategy.mean_reversion")


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def __init__(
        self,
        config: MeanReversionConfig,
        risk_config: RiskConfig,
    ) -> None:
        self.config = config
        self.risk_config = risk_config

    def required_bars(self) -> int:
        return self.config.lookback_period + 5

    def _compute_signals(
        self, symbol: str, df: pd.DataFrame, portfolio_value: float
    ) -> List[Signal]:
        """Return BUY/SELL/HOLD signals for a single symbol."""
        if len(df) < self.config.lookback_period:
            return []

        close = df["close"]
        rolling_mean = close.rolling(self.config.lookback_period).mean()
        rolling_std = close.rolling(self.config.lookback_period).std()

        current_price = float(close.iloc[-1])
        mean = float(rolling_mean.iloc[-1])
        std = float(rolling_std.iloc[-1])

        if std == 0 or pd.isna(mean) or pd.isna(std):
            return []

        z_score = (current_price - mean) / std
        upper_band = mean + self.config.std_threshold * std
        lower_band = mean - self.config.std_threshold * std

        signals: List[Signal] = []

        if current_price <= lower_band:
            # Price significantly below mean → buy signal
            max_position_value = portfolio_value * (
                self.risk_config.max_position_size_pct / 100
            )
            qty = round(max_position_value / current_price, 4)
            if qty > 0:
                confidence = min(abs(z_score) / (self.config.std_threshold * 2), 1.0)
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        quantity=qty,
                        reason=(
                            f"Price {current_price:.2f} below lower band {lower_band:.2f} "
                            f"(z-score={z_score:.2f})"
                        ),
                        confidence=confidence,
                        metadata={
                            "z_score": z_score,
                            "mean": mean,
                            "std": std,
                            "upper_band": upper_band,
                            "lower_band": lower_band,
                        },
                    )
                )
        elif current_price >= upper_band:
            # Price significantly above mean → sell signal
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    quantity=0.0,
                    reason=(
                        f"Price {current_price:.2f} above upper band {upper_band:.2f} "
                        f"(z-score={z_score:.2f})"
                    ),
                    confidence=min(abs(z_score) / (self.config.std_threshold * 2), 1.0),
                    metadata={
                        "z_score": z_score,
                        "mean": mean,
                        "upper_band": upper_band,
                        "lower_band": lower_band,
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
                sym_signals = self._compute_signals(symbol, df, portfolio_value)
                # Only emit SELL if we actually hold the position
                for sig in sym_signals:
                    if sig.is_sell and symbol not in open_positions:
                        continue
                    signals.append(sig)
                    logger.debug("Signal: %s", sig)
            except Exception as exc:
                logger.error("Error generating mean reversion signal for %s: %s", symbol, exc)
        return signals
