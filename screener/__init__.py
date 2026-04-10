"""Dynamic watchlist screener.

Each morning, before the session begins, the screener ranks symbols by a
composite score derived from relative volume, ATR-based volatility, and
recent price momentum.  The top-N symbols replace the static watchlist
for that day's trading session.

If the Alpaca ``ScreenerClient`` (most-actives endpoint) is available,
it is used as an additional signal source.  Otherwise the screener works
purely from bar data already fetched by the engine.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alerts import get_logger
from config import AlpacaConfig, ScreenerConfig

logger = get_logger("screener")


class MarketScreener:
    """Ranks symbols by liquidity, volatility, and momentum."""

    def __init__(
        self, alpaca_config: AlpacaConfig, screener_config: ScreenerConfig
    ) -> None:
        self.alpaca_config = alpaca_config
        self.config = screener_config
        self._screener_client = None

    # ------------------------------------------------------------------
    # Optional Alpaca screener client
    # ------------------------------------------------------------------

    def _get_screener_client(self):
        if self._screener_client is None:
            try:
                from alpaca.data import ScreenerClient

                self._screener_client = ScreenerClient(
                    api_key=self.alpaca_config.api_key,
                    secret_key=self.alpaca_config.secret_key,
                )
            except Exception as exc:
                logger.warning("Could not initialise ScreenerClient: %s", exc)
        return self._screener_client

    def get_most_actives(self, top_n: int = 20) -> List[str]:
        """Return the top-N most actively traded symbols via Alpaca API."""
        client = self._get_screener_client()
        if client is None:
            return []
        try:
            from alpaca.data import MostActivesRequest, MostActivesBy

            req = MostActivesRequest(top=top_n, by=MostActivesBy.VOLUME)
            result = client.get_most_actives(req)
            symbols = [item.symbol for item in (result.most_actives or [])]
            logger.info("Most-actives from Alpaca: %s", symbols[:10])
            return symbols
        except Exception as exc:
            logger.warning("Most-actives API call failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Bar-based scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Compute the 14-day Average True Range as a fraction of close price."""
        if len(df) < period + 1:
            return 0.0
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = float(np.mean(tr[-period:]))
        last_close = float(close[-1])
        return atr / last_close if last_close > 0 else 0.0

    @staticmethod
    def _compute_relative_volume(df: pd.DataFrame, lookback: int = 10) -> float:
        """Current-day volume relative to average of last *lookback* days."""
        if len(df) < 2:
            return 0.0
        volumes = df["volume"].values
        avg_vol = float(np.mean(volumes[-lookback - 1 : -1])) if len(volumes) > lookback else float(np.mean(volumes[:-1]))
        if avg_vol == 0:
            return 0.0
        return float(volumes[-1]) / avg_vol

    @staticmethod
    def _compute_momentum(df: pd.DataFrame, lookback: int = 10) -> float:
        """Price return over *lookback* days."""
        if len(df) <= lookback:
            return 0.0
        closes = df["close"].values
        start_price = float(closes[-lookback - 1])
        end_price = float(closes[-1])
        return (end_price - start_price) / start_price if start_price > 0 else 0.0

    def score_symbols(
        self,
        bars: Dict[str, pd.DataFrame],
        lookback: Optional[int] = None,
        min_avg_volume: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return a composite score for each symbol in *bars*.

        Score = 0.4 × normalised_rel_volume
              + 0.3 × normalised_atr
              + 0.3 × normalised_momentum

        Returns a dict sorted by score (highest first).
        """
        lb = lookback or self.config.lookback_days
        min_vol = min_avg_volume or self.config.min_avg_volume

        raw: Dict[str, Dict[str, float]] = {}
        for sym, df in bars.items():
            if df.empty or len(df) < lb + 1:
                continue
            avg_vol = float(df["volume"].iloc[-lb - 1 : -1].mean()) if len(df) > lb else 0.0
            if avg_vol < min_vol:
                continue
            raw[sym] = {
                "rel_vol": self._compute_relative_volume(df, lb),
                "atr": self._compute_atr(df),
                "momentum": self._compute_momentum(df, lb),
            }

        if not raw:
            return {}

        def _normalise(values: List[float]) -> List[float]:
            arr = np.array(values, dtype=float)
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return [0.5] * len(values)
            return list((arr - mn) / (mx - mn))

        syms = list(raw.keys())
        rv_norm = _normalise([raw[s]["rel_vol"] for s in syms])
        atr_norm = _normalise([raw[s]["atr"] for s in syms])
        mom_norm = _normalise([raw[s]["momentum"] for s in syms])

        scores = {
            sym: 0.4 * rv + 0.3 * at + 0.3 * mo
            for sym, rv, at, mo in zip(syms, rv_norm, atr_norm, mom_norm)
        }
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Main public interface
    # ------------------------------------------------------------------

    def get_screened_symbols(
        self,
        bars: Dict[str, pd.DataFrame],
        top_n: Optional[int] = None,
        use_most_actives: bool = True,
    ) -> List[str]:
        """Return up to *top_n* symbols ranked by composite score.

        If ``use_most_actives`` is True and the Alpaca screener API is
        reachable, most-actives symbols are merged in (they add to the
        candidate universe rather than replacing bar-scored symbols).
        """
        if not self.config.enabled:
            return list(bars.keys())

        n = top_n or self.config.top_n

        # Bar-based scoring of current universe
        scores = self.score_symbols(bars)

        # Optionally blend in API most-actives (boosted score = 1.0)
        if use_most_actives:
            most_active = self.get_most_actives(top_n=n)
            for sym in most_active:
                if sym not in scores:
                    scores[sym] = 1.0  # Will be fetched on next cycle
                else:
                    scores[sym] = min(1.0, scores[sym] + 0.2)  # Slight boost

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [sym for sym, _ in ranked[:n]]
        logger.info("Screener selected %d symbols: %s", len(selected), selected[:10])
        return selected
