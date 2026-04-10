"""Tests for Telegram/Slack alerting in Alerter."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AlertConfig
from alerts import Alerter


def _cfg(**kwargs) -> AlertConfig:
    defaults = {
        "email": "",
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "",
        "smtp_password": "",
        "telegram_token": "",
        "telegram_chat_id": "",
        "slack_webhook_url": "",
    }
    defaults.update(kwargs)
    return AlertConfig(**defaults)


class TestTelegramAlerter:
    def test_send_telegram_skips_when_not_configured(self):
        alerter = Alerter(_cfg())
        result = alerter.send_telegram("Subject", "Body")
        assert result is False

    def test_send_telegram_posts_when_configured(self):
        alerter = Alerter(_cfg(telegram_token="tok123", telegram_chat_id="chat456"))
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("alerts.requests.post", return_value=mock_resp) as mock_post:
            result = alerter.send_telegram("Trade alert", "AAPL BUY")
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "tok123" in call_kwargs[0][0]  # URL contains token
        payload = call_kwargs[1]["json"]
        assert payload["chat_id"] == "chat456"
        assert "Trade alert" in payload["text"]

    def test_send_telegram_returns_false_on_error(self):
        alerter = Alerter(_cfg(telegram_token="tok", telegram_chat_id="chat"))
        with patch("alerts.requests.post", side_effect=Exception("network error")):
            result = alerter.send_telegram("Subject", "Body")
        assert result is False


class TestSlackAlerter:
    def test_send_slack_skips_when_not_configured(self):
        alerter = Alerter(_cfg())
        result = alerter.send_slack("Subject", "Body")
        assert result is False

    def test_send_slack_posts_when_configured(self):
        webhook = "https://hooks.slack.com/services/T000/B000/xxx"
        alerter = Alerter(_cfg(slack_webhook_url=webhook))
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("alerts.requests.post", return_value=mock_resp) as mock_post:
            result = alerter.send_slack("Daily report", "P&L: +$500")
        assert result is True
        mock_post.assert_called_once_with(webhook, json={"text": mock_post.call_args[1]["json"]["text"]}, timeout=10)
        assert "Daily report" in mock_post.call_args[1]["json"]["text"]

    def test_send_slack_returns_false_on_error(self):
        alerter = Alerter(_cfg(slack_webhook_url="https://hooks.slack.com/x"))
        with patch("alerts.requests.post", side_effect=Exception("timeout")):
            result = alerter.send_slack("Subject", "Body")
        assert result is False


class TestAlertIntegration:
    def test_alert_calls_telegram_and_slack(self):
        alerter = Alerter(_cfg(telegram_token="tok", telegram_chat_id="chat",
                               slack_webhook_url="https://hooks.slack.com/x"))
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("alerts.requests.post", return_value=mock_resp) as mock_post:
            alerter.alert("test subject", "test body")
        # Should have called requests.post twice (Telegram + Slack)
        assert mock_post.call_count == 2
