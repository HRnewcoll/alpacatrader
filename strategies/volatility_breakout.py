"""Volatility Breakout Strategy.

Identifies periods of low volatility (compression) and trades breakouts
from consolidation ranges using ATR-based dynamic thresholds.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import ta

from alerts import get_logger
from config import RiskConfig
from strategies.base_strategy import BaseStrategy, Signal, SignalType

logger = get_logger("strategy.volatility_breakout")


class VolatilityBreakoutConfig:
    """Configuration for volatility breakout strategy."""
    
    def __init__(
        self,
        atr_period: int = 14,
        lookback_period: int = 20,
        breakout_multiplier: float = 1.5,
        atr_stop_multiplier: float = 2.0,
        volume_confirmation: bool = True,
        volume_ma_period: int = 20,
    ):
        self.atr_period = atr_period
        self.lookback_period = lookback_period
        self.breakout_multiplier = breakout_multiplier
        self.atr_stop_multiplier = atr_stop_multiplier
        self.volume_confirmation = volume_confirmation
        self.volume_ma_period = volume_ma_period


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy using ATR and range compression."""
    
    name = "volatility_breakout"

    def __init__(
        self,
        config: VolatilityBreakoutConfig,
        risk_config: RiskConfig,
    ) -> None:
        self.config = config
        self.risk_config = risk_config
        self._positions: Dict[str, dict] = {}

    def required_bars(self) -> int:
        return max(self.config.lookback_period, self.config.atr_period) + self.config.volume_ma_period + 10

    def _compute_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute ATR, rolling range, and volume MA."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df.get("volume", pd.Series([0] * len(df), index=df.index))
        
        # ATR
        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=self.config.atr_period
        ).average_true_range()
        
        # Rolling high/low for range detection
        rolling_high = high.rolling(self.config.lookback_period).max()
        rolling_low = low.rolling(self.config.lookback_period).min()
        range_size = rolling_high - rolling_low
        
        # Range compression: current range vs average range
        avg_range = range_size.rolling(self.config.lookback_period).mean()
        range_ratio = range_size / avg_range.replace(0, float('nan'))
        
        # Volume MA
        volume_ma = volume.rolling(self.config.volume_ma_period).mean()
        volume_ratio = volume / volume_ma.replace(0, float('nan'))
        
        # Bollinger Bands for additional confirmation
        bb = ta.volatility.BollingerBands(close=close, window=self.config.lookback_period)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_width = (bb_upper - bb_lower) / close
        
        return {
            "atr": atr,
            "rolling_high": rolling_high,
            "rolling_low": rolling_low,
            "range_ratio": range_ratio,
            "volume_ratio": volume_ratio,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_width": bb_width,
        }

    def _detect_compression(self, indicators: Dict[str, pd.Series], threshold: float = 0.8) -> bool:
        """Detect if market is in compression (low volatility)."""
        range_ratio = float(indicators["range_ratio"].iloc[-1])
        bb_width = float(indicators["bb_width"].iloc[-1])
        bb_width_avg = float(indicators["bb_width"].rolling(self.config.lookback_period).mean().iloc[-1])
        
        # Compression when range is below average and BB width is narrowing
        is_compressed = range_ratio < threshold and bb_width < bb_width_avg * 0.9
        return bool(is_compressed)

    def _generate_signals_for_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        portfolio_value: float,
        has_position: bool,
    ) -> List[Signal]:
        """Generate buy/sell signals for a single symbol."""
        if len(df) < self.required_bars():
            return []

        try:
            indicators = self._compute_indicators(df)
        except Exception as exc:
            logger.error("Indicator computation failed for %s: %s", symbol, exc)
            return []

        current_price = float(df["close"].iloc[-1])
        current_high = float(df["high"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        
        if current_price <= 0:
            return []

        atr_now = float(indicators["atr"].iloc[-1])
        rolling_high = float(indicators["rolling_high"].iloc[-1])
        rolling_low = float(indicators["rolling_low"].iloc[-1])
        volume_ratio = float(indicators["volume_ratio"].iloc[-1])
        
        if pd.isna(atr_now) or pd.isna(rolling_high) or pd.isna(rolling_low):
            return []

        signals: List[Signal] = []
        
        # Breakout levels
        upper_breakout = rolling_high + self.config.breakout_multiplier * atr_now
        lower_breakout = rolling_low - self.config.breakout_multiplier * atr_now
        
        # Check for compression before breakout
        is_compressed = self._detect_compression(indicators)
        
        # Volume confirmation
        volume_confirmed = not self.config.volume_confirmation or volume_ratio > 1.2

        # BUY signal: price breaks above upper level with volume
        if current_high >= upper_breakout and not has_position:
            if volume_confirmed or not self.config.volume_confirmation:
                confidence = min((current_high - rolling_high) / atr_now, 2.0) / 2.0
                if is_compressed:
                    confidence = min(confidence + 0.2, 1.0)
                
                max_position_value = portfolio_value * (self.risk_config.max_position_size_pct / 100)
                qty = round(max_position_value / current_price, 4)
                
                if qty > 0:
                    stop_loss = current_price - self.config.atr_stop_multiplier * atr_now
                    signals.append(
                        Signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            price=current_price,
                            quantity=qty,
                            reason=(
                                f"Volatility breakout: price ${current_price:.2f} > upper ${upper_breakout:.2f}, "
                                f"ATR={atr_now:.2f}, compressed={is_compressed}, vol_ratio={volume_ratio:.2f}"
                            ),
                            confidence=confidence,
                            metadata={
                                "atr": atr_now,
                                "stop_loss": stop_loss,
                                "breakout_level": upper_breakout,
                                "is_compression": is_compressed,
                                "volume_ratio": volume_ratio,
                            },
                        )
                    )

        # SELL signal: price breaks below lower level OR stop loss hit
        if has_position and symbol in self._positions:
            entry_info = self._positions[symbol]
            entry_price = entry_info.get("entry_price", current_price)
            stop_loss = entry_info.get("stop_loss", entry_price * 0.95)
            
            # Breakout failure or breakdown
            if current_low <= lower_breakout or current_price <= stop_loss:
                reason = "Breakdown" if current_low <= lower_breakout else "Stop loss"
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        quantity=0.0,
                        reason=f"{reason}: price ${current_price:.2f}, ATR={atr_now:.2f}",
                        confidence=0.9,
                        metadata={"atr": atr_now, "breakdown_level": lower_breakout},
                    )
                )
                if symbol in self._positions:
                    del self._positions[symbol]

        # Track new positions
        if signals and signals[0].is_buy and symbol not in self._positions:
            self._positions[symbol] = {
                "entry_price": current_price,
                "stop_loss": current_price - self.config.atr_stop_multiplier * atr_now,
            }

        return signals

    def generate_signals(
        self,
        bars: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        portfolio_value: float,
        open_positions: List[str],
    ) -> List[Signal]:
        """Generate signals for all symbols."""
        signals: List[Signal] = []
        for symbol, df in bars.items():
            if df.empty:
                continue
            try:
                has_position = symbol in open_positions
                sym_signals = self._generate_signals_for_symbol(
                    symbol, df, portfolio_value, has_position
                )
                for sig in sym_signals:
                    signals.append(sig)
                    logger.debug("Signal: %s", sig)
            except Exception as exc:
                logger.error("Error generating volatility breakout signal for %s: %s", symbol, exc)
        return signals
