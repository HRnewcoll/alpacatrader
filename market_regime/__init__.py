"""Market Regime Detection and Adaptive Strategy Selection.

Uses statistical methods to detect market regimes (trending, mean-reverting, 
high volatility, low volatility) and adjusts strategy parameters accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

from alerts import get_logger

logger = get_logger("market_regime")


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    NEUTRAL = "neutral"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeMetrics:
    """Metrics describing the current market regime."""
    trend_strength: float  # -1 to 1, negative=downtrend, positive=uptrend
    volatility_level: float  # normalized volatility
    mean_reversion_score: float  # 0-1, higher=more mean-reverting
    momentum_score: float  # -1 to 1
    regime: MarketRegime
    confidence: float  # 0-1 confidence in regime classification


class MarketRegimeDetector:
    """Detects market regime using multiple technical indicators."""
    
    def __init__(
        self,
        lookback_period: int = 60,
        adx_threshold: float = 25.0,
        volatility_lookback: int = 20,
    ):
        self.lookback_period = lookback_period
        self.adx_threshold = adx_threshold
        self.volatility_lookback = volatility_lookback
        self._history: Dict[str, pd.DataFrame] = {}

    def _compute_trend_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute trend strength indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # ADX for trend strength
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
        
        # MACD for trend direction
        macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd = macd_obj.macd()
        macd_signal = macd_obj.macd_signal()
        macd_diff = macd_obj.macd_diff()
        
        # Moving averages
        sma_20 = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        sma_50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        sma_200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
        
        # Linear regression slope
        returns = close.pct_change()
        trend_slope = returns.rolling(self.lookback_period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=False
        )
        
        return {
            "adx": adx,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_diff": macd_diff,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "trend_slope": trend_slope,
        }

    def _compute_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute volatility indicators."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # ATR
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        
        # Bollinger Band width
        bb = ta.volatility.BollingerBands(close=close, window=20)
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / close
        
        # Historical volatility (rolling std of returns)
        returns = close.pct_change()
        hist_vol = returns.rolling(self.volatility_lookback).std() * np.sqrt(252)
        
        # Normalize ATR by price
        atr_normalized = atr / close
        
        return {
            "atr": atr,
            "bb_width": bb_width,
            "hist_vol": hist_vol,
            "atr_normalized": atr_normalized,
        }

    def _compute_mean_reversion_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute mean reversion indicators."""
        close = df["close"]
        
        # RSI
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        
        # Stochastic oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=close, window=14
        ).stoch()
        
        # Distance from moving average
        sma_20 = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        distance_from_ma = (close - sma_20) / sma_20
        
        # Hurst exponent approximation (simplified)
        hurst = close.rolling(self.lookback_period).apply(
            lambda x: self._compute_hurst(x) if len(x) > 10 else 0.5,
            raw=False
        )
        
        return {
            "rsi": rsi,
            "stoch": stoch,
            "distance_from_ma": distance_from_ma,
            "hurst": hurst,
        }

    @staticmethod
    def _compute_hurst(prices: pd.Series) -> float:
        """Compute simplified Hurst exponent."""
        if len(prices) < 10:
            return 0.5
        
        log_returns = np.log(prices / prices.shift(1)).dropna()
        if len(log_returns) < 10:
            return 0.5
        
        # Simplified R/S analysis
        n = len(log_returns)
        tau = max(2, n // 4)
        
        rs_values = []
        for lag in range(2, min(tau, n), max(1, tau // 10)):
            subset = log_returns.iloc[:lag]
            mean = subset.mean()
            std = subset.std()
            if std == 0:
                continue
            cumulated = subset - mean
            z = cumulated.cumsum()
            range_val = z.max() - z.min()
            rs = range_val / std
            rs_values.append((lag, rs))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Linear regression on log-log scale
        log_lags = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        try:
            slope, _ = np.polyfit(log_lags, log_rs, 1)
            return float(slope)
        except Exception:
            return 0.5

    def detect_regime(self, symbol: str, df: pd.DataFrame) -> Optional[RegimeMetrics]:
        """Detect the current market regime for a symbol."""
        if len(df) < self.lookback_period + 50:
            return None

        try:
            trend_ind = self._compute_trend_indicators(df)
            vol_ind = self._compute_volatility_indicators(df)
            mr_ind = self._compute_mean_reversion_indicators(df)
        except Exception as exc:
            logger.error("Error computing indicators for %s: %s", symbol, exc)
            return None

        # Get latest values
        adx_now = float(trend_ind["adx"].iloc[-1])
        macd_diff_now = float(trend_ind["macd_diff"].iloc[-1])
        trend_slope_now = float(trend_ind["trend_slope"].iloc[-1])
        
        close_now = float(df["close"].iloc[-1])
        sma_20_now = float(trend_ind["sma_20"].iloc[-1])
        sma_50_now = float(trend_ind["sma_50"].iloc[-1])
        sma_200_now = float(trend_ind["sma_200"].iloc[-1])
        
        vol_now = float(vol_ind["hist_vol"].iloc[-1])
        vol_avg = float(vol_ind["hist_vol"].rolling(self.volatility_lookback).mean().iloc[-1])
        atr_norm_now = float(vol_ind["atr_normalized"].iloc[-1])
        
        rsi_now = float(mr_ind["rsi"].iloc[-1])
        hurst_now = float(mr_ind["hurst"].iloc[-1])
        
        if any(pd.isna([adx_now, macd_diff_now, vol_now, rsi_now])):
            return None

        # Compute scores
        # Trend strength: based on ADX
        trend_strength_raw = adx_now / 100.0
        
        # Trend direction: based on price vs MAs and MACD
        trend_direction = 0.0
        if close_now > sma_200_now:
            trend_direction += 0.3
        if close_now > sma_50_now:
            trend_direction += 0.3
        if close_now > sma_20_now:
            trend_direction += 0.2
        if macd_diff_now > 0:
            trend_direction += 0.2
        
        trend_strength = trend_strength_raw * (2 * trend_direction - 1)
        
        # Volatility level: normalized
        volatility_level = vol_now / 0.5  # Assume 50% annualized vol is very high
        volatility_level = min(max(volatility_level, 0), 1)
        
        # Mean reversion score: based on Hurst exponent
        # H < 0.5 suggests mean reversion, H > 0.5 suggests trending
        mean_reversion_score = max(0, 1 - hurst_now) if hurst_now > 0 else 0.5
        
        # Momentum score: based on RSI and trend
        momentum_score = (rsi_now - 50) / 50  # -1 to 1
        
        # Determine primary regime
        if volatility_level > 0.7:
            regime = MarketRegime.HIGH_VOLATILITY
        elif volatility_level < 0.3:
            regime = MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.4:
            regime = MarketRegime.STRONG_UPTREND if trend_direction > 0.5 else MarketRegime.STRONG_DOWNTREND
        elif trend_strength > 0.2:
            regime = MarketRegime.WEAK_UPTREND if trend_direction > 0.5 else MarketRegime.WEAK_DOWNTREND
        else:
            regime = MarketRegime.NEUTRAL
        
        # Confidence based on indicator agreement
        confidence = 0.5
        if adx_now > self.adx_threshold:
            confidence += 0.2
        if abs(trend_direction - 0.5) > 0.3:
            confidence += 0.15
        if volatility_level > 0.7 or volatility_level < 0.3:
            confidence += 0.15
        
        confidence = min(confidence, 1.0)
        
        return RegimeMetrics(
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            mean_reversion_score=mean_reversion_score,
            momentum_score=momentum_score,
            regime=regime,
            confidence=confidence,
        )

    def get_regime_recommendation(self, regime: RegimeMetrics) -> Dict[str, any]:
        """Get strategy recommendations based on detected regime."""
        recommendations = {
            "preferred_strategies": [],
            "avoid_strategies": [],
            "parameter_adjustments": {},
            "risk_adjustment": 1.0,
        }
        
        if regime.regime == MarketRegime.STRONG_UPTREND:
            recommendations["preferred_strategies"] = ["momentum", "volatility_breakout"]
            recommendations["avoid_strategies"] = ["mean_reversion"]
            recommendations["parameter_adjustments"] = {
                "take_profit_pct": 6.0,  # Let winners run
                "stop_loss_pct": 3.0,
            }
            recommendations["risk_adjustment"] = 1.2
            
        elif regime.regime == MarketRegime.STRONG_DOWNTREND:
            recommendations["preferred_strategies"] = ["momentum"]
            recommendations["avoid_strategies"] = ["mean_reversion", "pairs_trading"]
            recommendations["parameter_adjustments"] = {
                "max_position_size_pct": 5.0,  # Reduce exposure
                "stop_loss_pct": 2.0,
            }
            recommendations["risk_adjustment"] = 0.5
            
        elif regime.regime == MarketRegime.NEUTRAL or regime.regime == MarketRegime.LOW_VOLATILITY:
            recommendations["preferred_strategies"] = ["mean_reversion", "pairs_trading"]
            recommendations["avoid_strategies"] = ["volatility_breakout"]
            recommendations["parameter_adjustments"] = {
                "mr_std_threshold": 1.5,  # More sensitive
                "take_profit_pct": 3.0,
            }
            recommendations["risk_adjustment"] = 1.0
            
        elif regime.regime == MarketRegime.HIGH_VOLATILITY:
            recommendations["preferred_strategies"] = ["volatility_breakout"]
            recommendations["avoid_strategies"] = ["mean_reversion"]
            recommendations["parameter_adjustments"] = {
                "max_position_size_pct": 5.0,
                "stop_loss_pct": 3.0,
                "vb_breakout_multiplier": 2.0,  # Require stronger breakout
            }
            recommendations["risk_adjustment"] = 0.7
            
        elif regime.regime in [MarketRegime.WEAK_UPTREND, MarketRegime.WEAK_DOWNTREND]:
            recommendations["preferred_strategies"] = ["momentum", "mean_reversion"]
            recommendations["avoid_strategies"] = []
            recommendations["parameter_adjustments"] = {
                "take_profit_pct": 4.0,
                "stop_loss_pct": 2.0,
            }
            recommendations["risk_adjustment"] = 1.0
        
        return recommendations
