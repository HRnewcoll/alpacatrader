"""
Candle Data Structure - Jesse/Backtrader Inspired
==================================================
Features:
- Candle-based data structure (OHLCV)
- Multi-timeframe support
- Efficient indexing and slicing
- Technical indicator integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """Single candlestick representation"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades_count: int = 0
    vwap: float = 0.0
    
    def __post_init__(self):
        if self.vwap == 0.0 and self.volume > 0:
            # Calculate VWAP if not provided
            typical_price = (self.high + self.low + self.close) / 3
            self.vwap = typical_price
    
    @property
    def range(self) -> float:
        """Candle range (high - low)"""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Candle body size (abs(close - open))"""
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish"""
        return self.close < self.open
    
    @property
    def upper_shadow(self) -> float:
        """Upper shadow size"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Lower shadow size"""
        return min(self.open, self.close) - self.low
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'trades_count': self.trades_count,
            'vwap': self.vwap
        }


class CandleSeries:
    """
    Time series of candles with efficient operations
    Inspired by Jesse's candle data structure
    """
    
    def __init__(self, candles: Optional[List[Candle]] = None):
        self.candles: List[Candle] = candles or []
        self._df_cache: Optional[pd.DataFrame] = None
        
    def add_candle(self, candle: Candle):
        """Add a new candle"""
        if self.candles and candle.timestamp <= self.candles[-1].timestamp:
            logger.warning("Candle timestamp not sequential")
        self.candles.append(candle)
        self._df_cache = None  # Invalidate cache
    
    def add_candles(self, candles: List[Candle]):
        """Add multiple candles"""
        self.candles.extend(candles)
        self._df_cache = None
    
    def __len__(self) -> int:
        return len(self.candles)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Candle, List[Candle]]:
        return self.candles[idx]
    
    def __iter__(self):
        return iter(self.candles)
    
    @property
    def df(self) -> pd.DataFrame:
        """Convert to DataFrame (cached)"""
        if self._df_cache is None:
            if not self.candles:
                return pd.DataFrame()
            
            data = [c.to_dict() for c in self.candles]
            self._df_cache = pd.DataFrame(data)
            self._df_cache.set_index('timestamp', inplace=True)
        
        return self._df_cache
    
    def get_last(self, n: int = 1) -> Union[Candle, List[Candle]]:
        """Get last n candles"""
        if n == 1:
            return self.candles[-1] if self.candles else None
        return self.candles[-n:]
    
    def calculate_sma(self, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return self.df['close'].rolling(window=period).mean()
    
    def calculate_ema(self, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return self.df['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def calculate_macd(
        self, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = self.calculate_ema(fast_period)
        ema_slow = self.calculate_ema(slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(
        self, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(period)
        std = self.df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def pattern_doji(self, idx: int = -1) -> bool:
        """Detect Doji pattern"""
        candle = self.candles[idx]
        body_ratio = candle.body / candle.range if candle.range > 0 else 0
        return body_ratio < 0.1
    
    def pattern_hammer(self, idx: int = -1) -> bool:
        """Detect Hammer pattern"""
        candle = self.candles[idx]
        if not candle.is_bullish:
            return False
        
        lower_shadow_ratio = candle.lower_shadow / candle.body
        upper_shadow_ratio = candle.upper_shadow / candle.body
        
        return lower_shadow_ratio > 2 and upper_shadow_ratio < 0.5
    
    def pattern_engulfing(self, idx: int = -1) -> Optional[str]:
        """Detect Engulfing pattern"""
        if len(self.candles) < 2:
            return None
        
        current = self.candles[idx]
        previous = self.candles[idx - 1] if idx != -1 else self.candles[-2]
        
        # Bullish engulfing
        if (previous.is_bearish and current.is_bullish and
            current.open < previous.close and
            current.close > previous.open):
            return 'bullish'
        
        # Bearish engulfing
        if (previous.is_bullish and current.is_bearish and
            current.open > previous.close and
            current.close < previous.open):
            return 'bearish'
        
        return None


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe analysis inspired by Jesse
    Analyzes same asset across different timeframes
    """
    
    def __init__(self):
        self.timeframes: Dict[str, CandleSeries] = {}
        
    def add_timeframe(self, name: str, candles: CandleSeries):
        """Add a timeframe"""
        self.timeframes[name] = candles
    
    def get_alignment(
        self, 
        direction: str = 'bullish'
    ) -> Dict[str, bool]:
        """
        Check if all timeframes align in same direction
        
        Args:
            direction: 'bullish' or 'bearish'
            
        Returns:
            Dict of timeframe -> alignment status
        """
        alignment = {}
        
        for tf_name, candles in self.timeframes.items():
            if len(candles) < 2:
                alignment[tf_name] = False
                continue
            
            last_close = candles.get_last(2)[1].close
            prev_close = candles.get_last(2)[0].close
            
            if direction == 'bullish':
                alignment[tf_name] = last_close > prev_close
            else:
                alignment[tf_name] = last_close < prev_close
        
        return alignment
    
    def get_strongest_signal(self) -> Tuple[str, float]:
        """
        Get strongest directional signal across timeframes
        
        Returns:
            Tuple of (direction, confidence)
        """
        if not self.timeframes:
            return ('neutral', 0.0)
        
        bullish_count = 0
        total = len(self.timeframes)
        
        for candles in self.timeframes.values():
            if len(candles) < 2:
                continue
            
            last_close = candles.get_last(2)[1].close
            prev_close = candles.get_last(2)[0].close
            
            if last_close > prev_close:
                bullish_count += 1
        
        bullish_ratio = bullish_count / total
        
        if bullish_ratio > 0.7:
            return ('bullish', bullish_ratio)
        elif bullish_ratio < 0.3:
            return ('bearish', 1 - bullish_ratio)
        else:
            return ('neutral', 0.5)


class CandleDataManager:
    """
    Manage candle data from multiple sources
    Supports CSV, database, live feeds
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_store: Dict[str, CandleSeries] = {}
        
    def load_from_dataframe(
        self, 
        symbol: str, 
        df: pd.DataFrame,
        timeframe: str = '1m'
    ) -> CandleSeries:
        """
        Load candles from DataFrame
        
        Expected columns: timestamp, open, high, low, close, volume
        """
        candles = []
        
        for _, row in df.iterrows():
            candle = Candle(
                timestamp=pd.to_datetime(row['timestamp']),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row.get('volume', 0),
                trades_count=row.get('trades_count', 0)
            )
            candles.append(candle)
        
        series = CandleSeries(candles)
        self.data_store[f"{symbol}_{timeframe}"] = series
        
        logger.info(f"Loaded {len(candles)} candles for {symbol} ({timeframe})")
        return series
    
    def load_from_csv(
        self, 
        symbol: str, 
        filepath: str,
        timeframe: str = '1m'
    ) -> CandleSeries:
        """Load candles from CSV file"""
        df = pd.read_csv(filepath)
        return self.load_from_dataframe(symbol, df, timeframe)
    
    def get_series(self, key: str) -> Optional[CandleSeries]:
        """Get candle series by key"""
        return self.data_store.get(key)
    
    def update_live(self, symbol: str, candle: Candle, timeframe: str = '1m'):
        """Update with live candle data"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.data_store:
            self.data_store[key] = CandleSeries([candle])
        else:
            self.data_store[key].add_candle(candle)


# Example usage
if __name__ == "__main__":
    print("Candle Data Module Loaded Successfully")
    print("Features:")
    print("  - Candle-based OHLCV structure")
    print("  - Multi-timeframe analysis")
    print("  - Pattern recognition (Doji, Hammer, Engulfing)")
    print("  - Technical indicators (SMA, EMA, RSI, MACD, ATR, BB)")
    print("\nUsage:")
    print("  from candle_data import Candle, CandleSeries, MultiTimeframeAnalyzer")
    print("  candle = Candle(timestamp, open, high, low, close, volume)")
    print("  series = CandleSeries([candle1, candle2, ...])")
    print("  rsi = series.calculate_rsi(14)")
