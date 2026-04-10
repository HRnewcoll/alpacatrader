"""Tests for MarketScreener."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AlpacaConfig, ScreenerConfig
from screener import MarketScreener


def _alpaca_cfg() -> AlpacaConfig:
    return AlpacaConfig(api_key="k", secret_key="s")


def _screener_cfg(**kwargs) -> ScreenerConfig:
    defaults = dict(
        enabled=True,
        top_n=5,
        min_avg_volume=0,  # No minimum so test data always qualifies
        lookback_days=5,
    )
    defaults.update(kwargs)
    return ScreenerConfig(**defaults)


def _make_bars(prices, volumes=None, start="2023-01-01") -> pd.DataFrame:
    n = len(prices)
    dates = pd.date_range(start=start, periods=n, freq="B", tz="UTC")
    vols = volumes if volumes is not None else [1_000_000] * n
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": vols,
        },
        index=dates,
    )


class TestAtrComputation:
    def test_atr_positive(self):
        prices = [100.0 + i for i in range(30)]
        df = _make_bars(prices)
        atr = MarketScreener._compute_atr(df)
        assert atr > 0

    def test_atr_insufficient_data(self):
        df = _make_bars([100.0] * 5)
        atr = MarketScreener._compute_atr(df, period=14)
        assert atr == 0.0


class TestRelativeVolume:
    def test_high_relative_volume(self):
        # Last bar has 10x average volume
        vols = [1_000_000] * 10 + [10_000_000]
        df = _make_bars([100.0] * 11, volumes=vols)
        rv = MarketScreener._compute_relative_volume(df, lookback=10)
        assert rv == pytest.approx(10.0, rel=0.1)

    def test_normal_relative_volume(self):
        vols = [1_000_000] * 11
        df = _make_bars([100.0] * 11, volumes=vols)
        rv = MarketScreener._compute_relative_volume(df, lookback=10)
        assert rv == pytest.approx(1.0, rel=0.01)


class TestMomentum:
    def test_positive_momentum(self):
        prices = [100.0] * 5 + [110.0] * 5 + [120.0]
        df = _make_bars(prices)
        mom = MarketScreener._compute_momentum(df, lookback=10)
        assert mom > 0

    def test_negative_momentum(self):
        prices = [120.0] * 5 + [110.0] * 5 + [100.0]
        df = _make_bars(prices)
        mom = MarketScreener._compute_momentum(df, lookback=10)
        assert mom < 0

    def test_insufficient_data_returns_zero(self):
        df = _make_bars([100.0] * 3)
        mom = MarketScreener._compute_momentum(df, lookback=10)
        assert mom == 0.0


class TestScoreSymbols:
    def _screener(self, **kwargs):
        return MarketScreener(_alpaca_cfg(), _screener_cfg(**kwargs))

    def test_returns_dict(self):
        screener = self._screener()
        bars = {
            "AAPL": _make_bars([100.0 + i for i in range(20)]),
            "MSFT": _make_bars([200.0 + i * 0.5 for i in range(20)]),
        }
        scores = screener.score_symbols(bars)
        assert isinstance(scores, dict)

    def test_higher_momentum_scores_higher(self):
        screener = self._screener()
        # AAPL strong upward momentum, MSFT flat
        aapl_prices = [100.0 + i * 2 for i in range(20)]
        msft_prices = [200.0] * 20
        bars = {
            "AAPL": _make_bars(aapl_prices),
            "MSFT": _make_bars(msft_prices),
        }
        scores = screener.score_symbols(bars)
        assert scores.get("AAPL", 0) >= scores.get("MSFT", 0)

    def test_empty_bars_returns_empty(self):
        screener = self._screener()
        assert screener.score_symbols({}) == {}

    def test_min_volume_filter(self):
        screener = self._screener(min_avg_volume=10_000_000)
        bars = {
            "AAPL": _make_bars(
                [100.0 + i for i in range(20)],
                volumes=[500_000] * 20,  # Below threshold
            )
        }
        scores = screener.score_symbols(bars)
        assert "AAPL" not in scores


class TestGetScreenedSymbols:
    def _screener(self, **kwargs):
        return MarketScreener(_alpaca_cfg(), _screener_cfg(**kwargs))

    def test_returns_list(self):
        screener = self._screener()
        bars = {
            "AAPL": _make_bars([100.0 + i for i in range(20)]),
            "MSFT": _make_bars([200.0 + i for i in range(20)]),
        }
        # Disable most-actives API call
        with patch.object(screener, "get_most_actives", return_value=[]):
            result = screener.get_screened_symbols(bars, use_most_actives=False)
        assert isinstance(result, list)

    def test_top_n_respected(self):
        screener = self._screener(top_n=2)
        bars = {sym: _make_bars([100.0 + i for i in range(20)]) for sym in "ABCDE"}
        with patch.object(screener, "get_most_actives", return_value=[]):
            result = screener.get_screened_symbols(bars, top_n=2, use_most_actives=False)
        assert len(result) <= 2

    def test_disabled_returns_all(self):
        screener = MarketScreener(_alpaca_cfg(), _screener_cfg(enabled=False))
        bars = {sym: _make_bars([100.0 + i for i in range(20)]) for sym in ["AAPL", "MSFT"]}
        result = screener.get_screened_symbols(bars)
        assert set(result) == {"AAPL", "MSFT"}
