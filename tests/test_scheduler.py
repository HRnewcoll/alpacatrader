"""Tests for the MarketScheduler."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SchedulerConfig
from scheduler import MarketScheduler


def _utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# NYSE opens at 14:30 UTC (9:30 ET) and closes at 21:00 UTC (4:00 PM ET)
_OPEN_TIME = _utc(2024, 1, 2, 14, 30)   # NYSE open (Tuesday)
_CLOSE_TIME = _utc(2024, 1, 2, 21, 0)   # NYSE close


def _mock_schedule(market_open: datetime, market_close: datetime):
    """Build a mock schedule DataFrame row."""
    import pandas as pd

    row = MagicMock()
    row.__getitem__ = lambda self, key: (
        pd.Timestamp(market_open) if key == "market_open" else pd.Timestamp(market_close)
    )
    schedule = MagicMock()
    schedule.empty = False
    schedule.iloc = [row]
    return schedule


def _make_scheduler() -> MarketScheduler:
    cfg = SchedulerConfig(use_market_hours=True, warmup_minutes=5, closeout_buffer_minutes=5)
    return MarketScheduler(cfg)


class TestMarketSchedulerIsOpen:
    def test_during_session(self):
        scheduler = _make_scheduler()
        mid_session = _utc(2024, 1, 2, 18, 0)  # 1 PM ET
        with patch.object(scheduler, "_open_close", return_value=(_OPEN_TIME, _CLOSE_TIME)):
            assert scheduler.is_market_open(mid_session)

    def test_before_open(self):
        scheduler = _make_scheduler()
        before_open = _utc(2024, 1, 2, 13, 0)  # 8 AM ET
        with patch.object(scheduler, "_open_close", return_value=(_OPEN_TIME, _CLOSE_TIME)):
            assert not scheduler.is_market_open(before_open)

    def test_after_close(self):
        scheduler = _make_scheduler()
        after_close = _utc(2024, 1, 2, 22, 0)  # 5 PM ET
        with patch.object(scheduler, "_open_close", return_value=(_OPEN_TIME, _CLOSE_TIME)):
            assert not scheduler.is_market_open(after_close)

    def test_weekend_returns_false(self):
        scheduler = _make_scheduler()
        weekend = _utc(2024, 1, 6, 16, 0)  # Saturday
        with patch.object(scheduler, "_open_close", return_value=(None, None)):
            assert not scheduler.is_market_open(weekend)


class TestMarketSchedulerNearClose:
    def test_within_buffer(self):
        scheduler = _make_scheduler()
        # 4 minutes before close (buffer = 5 min)
        near = _CLOSE_TIME - timedelta(minutes=4)
        with patch.object(scheduler, "_open_close", return_value=(_OPEN_TIME, _CLOSE_TIME)):
            assert scheduler.is_near_close(near)

    def test_outside_buffer(self):
        scheduler = _make_scheduler()
        # 30 minutes before close
        far = _CLOSE_TIME - timedelta(minutes=30)
        with patch.object(scheduler, "_open_close", return_value=(_OPEN_TIME, _CLOSE_TIME)):
            assert not scheduler.is_near_close(far)

    def test_after_close(self):
        scheduler = _make_scheduler()
        after = _CLOSE_TIME + timedelta(minutes=10)
        with patch.object(scheduler, "_open_close", return_value=(_OPEN_TIME, _CLOSE_TIME)):
            assert not scheduler.is_near_close(after)


class TestMarketSchedulerWarmup:
    def test_within_warmup_window(self):
        scheduler = _make_scheduler()
        # 3 minutes before open (warmup = 5 min)
        warmup_time = _OPEN_TIME - timedelta(minutes=3)
        with patch.object(scheduler, "_open_close", return_value=(_OPEN_TIME, _CLOSE_TIME)):
            assert scheduler.is_warmup_window(warmup_time)

    def test_outside_warmup_window(self):
        scheduler = _make_scheduler()
        before = _OPEN_TIME - timedelta(minutes=60)
        with patch.object(scheduler, "_open_close", return_value=(_OPEN_TIME, _CLOSE_TIME)):
            assert not scheduler.is_warmup_window(before)

    def test_after_open_not_warmup(self):
        scheduler = _make_scheduler()
        after = _OPEN_TIME + timedelta(minutes=1)
        with patch.object(scheduler, "_open_close", return_value=(_OPEN_TIME, _CLOSE_TIME)):
            assert not scheduler.is_warmup_window(after)


class TestMarketSchedulerNextOpen:
    def test_next_open_is_in_future(self):
        scheduler = _make_scheduler()
        # The real pandas_market_calendars is queried here — just check it's a future UTC dt
        now = datetime.now(tz=timezone.utc)
        next_open = scheduler.next_market_open(now)
        assert next_open > now
        assert next_open.tzinfo is not None

    def test_next_open_skips_weekend(self):
        """If called on a Friday after close, next open should be Monday."""
        scheduler = _make_scheduler()
        # 2024-01-05 is Friday; check we get a Monday
        friday_evening = _utc(2024, 1, 5, 22, 0)
        next_open = scheduler.next_market_open(friday_evening)
        # Monday 2024-01-08
        assert next_open.weekday() == 0  # Monday
