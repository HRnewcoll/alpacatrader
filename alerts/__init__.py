"""Alerting and logging utilities."""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import requests

from config import AlertConfig


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Configure application-wide logging with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if not root.handlers:
        console = logging.StreamHandler()
        console.setLevel(numeric_level)
        console.setFormatter(formatter)
        root.addHandler(console)

        file_handler = logging.FileHandler(os.path.join(log_dir, "trading.log"))
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return logging.getLogger("alpacatrader")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"alpacatrader.{name}")


class Alerter:
    """Sends alerts via email and/or logs them."""

    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self.logger = get_logger("alerter")

    def send_email(self, subject: str, body: str) -> bool:
        """Send an email alert. Returns True on success."""
        if not self.config.email or not self.config.smtp_user:
            self.logger.debug("Email alerting not configured, skipping.")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.config.smtp_user
            msg["To"] = self.config.email
            msg["Subject"] = f"[AlpacaTrader] {subject}"
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.sendmail(
                    self.config.smtp_user,
                    self.config.email,
                    msg.as_string(),
                )
            self.logger.info("Email alert sent: %s", subject)
            return True
        except Exception as exc:
            self.logger.error("Failed to send email alert: %s", exc)
            return False

    def send_telegram(self, subject: str, body: str) -> bool:
        """Send a Telegram message via the Bot API. Returns True on success."""
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            self.logger.debug("Telegram alerting not configured, skipping.")
            return False
        try:
            text = f"*[AlpacaTrader]* {subject}\n{body}"
            url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
            resp = requests.post(
                url,
                json={
                    "chat_id": self.config.telegram_chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            resp.raise_for_status()
            self.logger.info("Telegram alert sent: %s", subject)
            return True
        except Exception as exc:
            self.logger.error("Failed to send Telegram alert: %s", exc)
            return False

    def send_slack(self, subject: str, body: str) -> bool:
        """Send a Slack message via an incoming webhook. Returns True on success."""
        if not self.config.slack_webhook_url:
            self.logger.debug("Slack alerting not configured, skipping.")
            return False
        try:
            text = f"*[AlpacaTrader]* {subject}\n{body}"
            resp = requests.post(
                self.config.slack_webhook_url,
                json={"text": text},
                timeout=10,
            )
            resp.raise_for_status()
            self.logger.info("Slack alert sent: %s", subject)
            return True
        except Exception as exc:
            self.logger.error("Failed to send Slack alert: %s", exc)
            return False

    def alert(
        self,
        subject: str,
        body: str,
        level: str = "INFO",
        send_email: bool = False,
    ) -> None:
        """Log an alert and optionally send an email, Telegram message, and Slack message."""
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn("[ALERT] %s: %s", subject, body)
        if send_email:
            self.send_email(subject, body)
        self.send_telegram(subject, body)
        self.send_slack(subject, body)

    def trade_alert(
        self,
        symbol: str,
        action: str,
        qty: float,
        price: float,
        strategy: str,
        reason: str = "",
    ) -> None:
        subject = f"Trade {action} {qty} {symbol} @ ${price:.2f}"
        body = (
            f"Strategy: {strategy}\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Quantity: {qty}\n"
            f"Price: ${price:.2f}\n"
            f"Reason: {reason}"
        )
        self.alert(subject, body, level="info", send_email=bool(self.config.email))

    def error_alert(self, error: str, context: Optional[str] = None) -> None:
        subject = "Trading System Error"
        body = f"Error: {error}"
        if context:
            body += f"\nContext: {context}"
        self.alert(subject, body, level="error", send_email=bool(self.config.email))
