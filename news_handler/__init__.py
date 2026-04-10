"""News-driven sentiment filter.

Fetches recent headlines for watchlist symbols via Alpaca's News API,
scores each symbol with a simple keyword-based sentiment model, and
exposes helpers to block or down-weight BUY signals on strongly negative
news.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from alerts import get_logger
from config import AlpacaConfig, NewsConfig
from strategies.base_strategy import Signal, SignalType

logger = get_logger("news_handler")

# ---------------------------------------------------------------------------
# Sentiment keyword lexicons
# ---------------------------------------------------------------------------

_NEGATIVE = frozenset(
    [
        "lawsuit", "fraud", "recall", "bankrupt", "bankruptcy", "loss", "losses",
        "decline", "fall", "drop", "drops", "downgrade", "miss", "misses", "missed",
        "warning", "investigation", "fine", "penalty", "layoff", "layoffs", "fired",
        "resign", "resignation", "scandal", "debt", "default", "cut", "cuts",
        "reduce", "weak", "poor", "disappoint", "disappoints", "disappointed",
        "concern", "risk", "uncertain", "uncertainty", "volatile", "volatility",
        "crash", "plunge", "slump", "tumble", "warn", "warns", "warned", "short",
        "bearish", "negative", "halt", "halted", "suspend", "suspended", "delist",
        "delisted", "breach", "hack", "cyber", "attack", "trouble", "troubled",
        "downside", "headwinds", "pressure", "pressured", "challenging",
    ]
)

_POSITIVE = frozenset(
    [
        "beat", "beats", "record", "records", "growth", "raise", "upgrade",
        "upgrades", "profit", "profits", "surge", "surges", "gain", "gains",
        "rally", "bullish", "positive", "strong", "stronger", "better", "exceed",
        "exceeds", "exceeded", "boost", "boosts", "improve", "improves", "improved",
        "recovery", "rebound", "expand", "expands", "increase", "increases",
        "dividend", "buyback", "acquire", "acquires", "acquired", "partner",
        "partners", "deal", "contract", "award", "launch", "launches", "innovate",
        "breakthrough", "outperform", "outperforms", "momentum", "opportunity",
        "guidance", "raised", "upside", "tailwinds", "robust", "accelerate",
        "accelerating",
    ]
)


def _score_text(text: str) -> float:
    """Return a sentiment score in [-1, +1] based on keyword matching."""
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


class NewsHandler:
    """Fetches news and produces per-symbol sentiment scores."""

    def __init__(self, alpaca_config: AlpacaConfig, news_config: NewsConfig) -> None:
        self.alpaca_config = alpaca_config
        self.config = news_config
        self._client = None

    # ------------------------------------------------------------------
    # Client initialisation (lazy)
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from alpaca.data import NewsClient

            self._client = NewsClient(
                api_key=self.alpaca_config.api_key,
                secret_key=self.alpaca_config.secret_key,
            )
        return self._client

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch_news(
        self,
        symbols: List[str],
        lookback_hours: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Return a mapping of symbol → list of headline+summary strings.

        Falls back gracefully to an empty dict on any API error so that
        news unavailability never stops the trading engine.
        """
        if not self.config.enabled or not symbols:
            return {}

        hours = lookback_hours or self.config.lookback_hours
        start = datetime.now(tz=timezone.utc) - timedelta(hours=hours)

        try:
            from alpaca.data import NewsRequest

            client = self._get_client()
            # Alpaca news API accepts comma-joined symbols as a single string
            req = NewsRequest(
                symbols=",".join(symbols),
                start=start,
                limit=50,
                include_content=False,
            )
            news_set = client.get_news(req)
        except Exception as exc:
            logger.warning("News API unavailable: %s. Skipping sentiment.", exc)
            return {}

        result: Dict[str, List[str]] = {sym: [] for sym in symbols}
        try:
            items = news_set.data.get("news", [])
        except AttributeError:
            items = []

        for item in items:
            text = " ".join(filter(None, [item.headline, item.summary]))
            for sym in (item.symbols or []):
                if sym in result:
                    result[sym].append(text)

        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_sentiment_scores(
        self,
        symbols: List[str],
        lookback_hours: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return a dict of symbol → sentiment score in [-1, +1].

        Symbols with no news get a neutral score of 0.0.
        Returns an empty dict when the news feature is disabled.
        """
        if not self.config.enabled:
            return {}

        news_by_symbol = self.fetch_news(symbols, lookback_hours)
        scores: Dict[str, float] = {}
        for sym in symbols:
            texts = news_by_symbol.get(sym, [])
            if not texts:
                scores[sym] = 0.0
            else:
                scores[sym] = sum(_score_text(t) for t in texts) / len(texts)
            if scores[sym] != 0.0:
                logger.debug("Sentiment %s=%.3f (%d articles)", sym, scores[sym], len(texts))
        return scores

    # ------------------------------------------------------------------
    # Signal filtering
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signals: List[Signal],
        sentiment_scores: Dict[str, float],
    ) -> List[Signal]:
        """Drop BUY signals for symbols with strongly negative news.

        SELL signals are always passed through unchanged.
        """
        if not self.config.enabled:
            return signals

        approved: List[Signal] = []
        for sig in signals:
            if sig.is_buy:
                score = sentiment_scores.get(sig.symbol, 0.0)
                if score < self.config.negative_threshold:
                    logger.info(
                        "BUY %s blocked by negative news (sentiment=%.3f < %.3f)",
                        sig.symbol,
                        score,
                        self.config.negative_threshold,
                    )
                    continue
                # Boost confidence for positive news (values above 1.0 are
                # treated as a relative multiplier by downstream callers)
                if score >= self.config.positive_threshold:
                    sig.confidence = sig.confidence * 1.1
            approved.append(sig)

        return approved
