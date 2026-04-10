"""Strategies package."""
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.pairs_trading import PairsTradingStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalType",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "PairsTradingStrategy",
]
