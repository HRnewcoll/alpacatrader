"""Tests for the NewsHandler sentiment module."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AlpacaConfig, NewsConfig
from news_handler import NewsHandler, _score_text
from strategies.base_strategy import Signal, SignalType


def _alpaca_cfg() -> AlpacaConfig:
    return AlpacaConfig(api_key="k", secret_key="s")


def _news_cfg(**kwargs) -> NewsConfig:
    defaults = dict(
        enabled=True,
        lookback_hours=24,
        negative_threshold=-0.3,
        positive_threshold=0.2,
    )
    defaults.update(kwargs)
    return NewsConfig(**defaults)


class TestScoreText:
    def test_positive_headline(self):
        score = _score_text("Company beats earnings record with strong revenue growth")
        assert score > 0

    def test_negative_headline(self):
        score = _score_text("Company faces lawsuit and fraud investigation penalty")
        assert score < 0

    def test_neutral_headline(self):
        score = _score_text("Company releases quarterly update")
        assert score == 0.0

    def test_empty_string(self):
        assert _score_text("") == 0.0

    def test_mixed_headline(self):
        # Equal positive and negative words → score of 0
        score = _score_text("beat loss")
        assert score == 0.0

    def test_score_bounded(self):
        score = _score_text("beat beats record growth expand profit")
        assert -1.0 <= score <= 1.0


class TestNewsHandlerSentiment:
    def _handler(self, **cfg_kwargs):
        return NewsHandler(_alpaca_cfg(), _news_cfg(**cfg_kwargs))

    def test_disabled_returns_empty(self):
        handler = self._handler(enabled=False)
        scores = handler.get_sentiment_scores(["AAPL", "MSFT"])
        assert scores == {}

    def test_scores_neutral_when_no_news(self):
        handler = self._handler()
        # Mock fetch_news to return empty lists
        with patch.object(handler, "fetch_news", return_value={"AAPL": [], "MSFT": []}):
            scores = handler.get_sentiment_scores(["AAPL", "MSFT"])
        assert scores["AAPL"] == 0.0
        assert scores["MSFT"] == 0.0

    def test_scores_positive_text(self):
        handler = self._handler()
        with patch.object(
            handler,
            "fetch_news",
            return_value={"AAPL": ["Company beats record with strong profit growth"]},
        ):
            scores = handler.get_sentiment_scores(["AAPL"])
        assert scores["AAPL"] > 0

    def test_scores_negative_text(self):
        handler = self._handler()
        with patch.object(
            handler,
            "fetch_news",
            return_value={"AAPL": ["Fraud lawsuit investigation penalty scandal"]},
        ):
            scores = handler.get_sentiment_scores(["AAPL"])
        assert scores["AAPL"] < 0

    def test_api_failure_returns_empty(self):
        handler = self._handler()
        with patch.object(handler, "_get_client", side_effect=RuntimeError("API down")):
            result = handler.fetch_news(["AAPL"])
        assert result == {}


class TestNewsHandlerFilterSignals:
    def _handler(self):
        return NewsHandler(_alpaca_cfg(), _news_cfg())

    def test_blocks_buy_on_negative_sentiment(self):
        handler = self._handler()
        signals = [Signal("AAPL", SignalType.BUY, 150.0, 10.0)]
        scores = {"AAPL": -0.8}
        result = handler.filter_signals(signals, scores)
        assert len(result) == 0

    def test_passes_buy_on_neutral_sentiment(self):
        handler = self._handler()
        signals = [Signal("AAPL", SignalType.BUY, 150.0, 10.0)]
        scores = {"AAPL": 0.0}
        result = handler.filter_signals(signals, scores)
        assert len(result) == 1

    def test_passes_buy_on_positive_sentiment(self):
        handler = self._handler()
        signals = [Signal("AAPL", SignalType.BUY, 150.0, 10.0)]
        scores = {"AAPL": 0.5}
        result = handler.filter_signals(signals, scores)
        assert len(result) == 1
        # Confidence should be boosted for positive news
        assert result[0].confidence > 1.0

    def test_always_passes_sell_signals(self):
        handler = self._handler()
        signals = [Signal("AAPL", SignalType.SELL, 150.0, 10.0)]
        scores = {"AAPL": -1.0}  # Very negative — but SELL should still pass
        result = handler.filter_signals(signals, scores)
        assert len(result) == 1

    def test_disabled_passes_all(self):
        handler = NewsHandler(_alpaca_cfg(), _news_cfg(enabled=False))
        signals = [
            Signal("AAPL", SignalType.BUY, 150.0, 10.0),
            Signal("MSFT", SignalType.BUY, 300.0, 5.0),
        ]
        result = handler.filter_signals(signals, {"AAPL": -1.0, "MSFT": -1.0})
        assert len(result) == 2

    def test_missing_symbol_sentiment_treated_as_neutral(self):
        handler = self._handler()
        signals = [Signal("AAPL", SignalType.BUY, 150.0, 10.0)]
        # No sentiment for AAPL → 0.0 → should pass
        result = handler.filter_signals(signals, {})
        assert len(result) == 1
