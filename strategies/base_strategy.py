"""Base strategy interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import pandas as pd


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    symbol: str
    signal_type: SignalType
    price: float
    quantity: float = 0.0
    reason: str = ""
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.signal_type == SignalType.BUY

    @property
    def is_sell(self) -> bool:
        return self.signal_type == SignalType.SELL

    def __repr__(self) -> str:
        return (
            f"Signal({self.signal_type.value} {self.symbol} @ ${self.price:.2f} "
            f"qty={self.quantity:.4f} conf={self.confidence:.2f})"
        )


class BaseStrategy(ABC):
    """Abstract base class that all strategies must implement."""

    name: str = "base"

    @abstractmethod
    def generate_signals(
        self,
        bars: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
        portfolio_value: float,
        open_positions: List[str],
    ) -> List[Signal]:
        """Analyse market data and return trade signals."""

    def required_bars(self) -> int:
        """Minimum number of historical bars needed to generate signals."""
        return 30

    def required_symbols(self) -> Optional[List[str]]:
        """Return the symbols this strategy requires, or None to use watchlist."""
        return None
