"""Market-hours aware scheduler for the trading engine.

Uses pandas_market_calendars to determine NYSE trading hours so the bot
only runs cycles during live sessions, warms up before open, and runs a
close-out cycle near the end of the trading day.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas_market_calendars as mcal

from alerts import get_logger
from config import SchedulerConfig

logger = get_logger("scheduler")

_WARMUP_DEFAULT_MINUTES = 5
_CLOSEOUT_DEFAULT_MINUTES = 5

# Lazily created NYSE calendar (module-level singleton to avoid re-creating)
_NYSE_CAL = None


def _nyse() -> mcal.MarketCalendar:
    global _NYSE_CAL
    if _NYSE_CAL is None:
        _NYSE_CAL = mcal.get_calendar("NYSE")
    return _NYSE_CAL


class MarketScheduler:
    """Determines when NYSE is open and manages sleep intervals.

    All public methods accept an optional *now* parameter (UTC-aware
    datetime) so that they are straightforward to unit-test without
    patching ``datetime.now``.
    """

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        self.config = config or SchedulerConfig()

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def _schedule(self, date_str: str):
        """Return the NYSE schedule DataFrame for a single date."""
        return _nyse().schedule(start_date=date_str, end_date=date_str)

    def _open_close(self, now: datetime):
        """Return (market_open, market_close) as UTC datetimes or (None, None)."""
        date_str = now.strftime("%Y-%m-%d")
        schedule = self._schedule(date_str)
        if schedule.empty:
            return None, None
        market_open = schedule.iloc[0]["market_open"].to_pydatetime()
        market_close = schedule.iloc[0]["market_close"].to_pydatetime()
        return market_open, market_close

    # ------------------------------------------------------------------
    # Status checks
    # ------------------------------------------------------------------

    def is_market_open(self, now: Optional[datetime] = None) -> bool:
        """Return True if the NYSE regular session is currently active."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        market_open, market_close = self._open_close(now)
        if market_open is None:
            return False
        return market_open <= now < market_close

    def is_near_close(
        self,
        now: Optional[datetime] = None,
        buffer_minutes: Optional[int] = None,
    ) -> bool:
        """Return True if within *buffer_minutes* of market close."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        if buffer_minutes is None:
            buffer_minutes = self.config.closeout_buffer_minutes
        _, market_close = self._open_close(now)
        if market_close is None:
            return False
        return 0 <= (market_close - now).total_seconds() <= buffer_minutes * 60

    def is_warmup_window(
        self,
        now: Optional[datetime] = None,
        warmup_minutes: Optional[int] = None,
    ) -> bool:
        """Return True if within *warmup_minutes* before market open."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        if warmup_minutes is None:
            warmup_minutes = self.config.warmup_minutes
        market_open, _ = self._open_close(now)
        if market_open is None:
            return False
        seconds_to_open = (market_open - now).total_seconds()
        return 0 < seconds_to_open <= warmup_minutes * 60

    def next_market_open(self, now: Optional[datetime] = None) -> datetime:
        """Return the next NYSE market open as a UTC datetime."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        # Search up to 7 days ahead (covers long holiday weekends)
        for delta in range(1, 8):
            candidate = now + timedelta(days=delta)
            date_str = candidate.strftime("%Y-%m-%d")
            schedule = self._schedule(date_str)
            if not schedule.empty:
                return schedule.iloc[0]["market_open"].to_pydatetime()
        # Fallback: 24 h from now
        return now + timedelta(hours=24)

    def market_close_today(self, now: Optional[datetime] = None) -> Optional[datetime]:
        """Return today's NYSE close time, or None if today is not a trading day."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        _, market_close = self._open_close(now)
        return market_close

    # ------------------------------------------------------------------
    # Blocking sleep
    # ------------------------------------------------------------------

    def sleep_until_open(self, now: Optional[datetime] = None) -> None:
        """Block until the next market open (minus the warmup window)."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        next_open = self.next_market_open(now)
        wake_time = next_open - timedelta(minutes=self.config.warmup_minutes)
        sleep_seconds = max(0.0, (wake_time - now).total_seconds())
        logger.info(
            "Market closed. Next open at %s. Sleeping %.0f s…",
            next_open.strftime("%Y-%m-%d %H:%M UTC"),
            sleep_seconds,
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
