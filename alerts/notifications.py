"""
Advanced Alerting System with Multiple Notification Channels
Supports Telegram, Discord, Slack, Email, and Webhooks
"""
import os
import json
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


class Alert:
    """Represents a trading alert"""
    
    def __init__(self, message: str, level: AlertLevel = AlertLevel.INFO,
                 symbol: str = None, strategy: str = None,
                 metadata: Dict = None):
        self.message = message
        self.level = level
        self.symbol = symbol
        self.strategy = strategy
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.id = f"{self.timestamp.timestamp()}_{id(self)}"
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "message": self.message,
            "level": self.level.value,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __repr__(self):
        return f"Alert({self.level.value}: {self.message})"


class AlertManager:
    """
    Centralized alert management with multiple notification channels
    """
    
    def __init__(self):
        self.channels = {}
        self.alert_history = []
        self.max_history = 1000
        
        # Configuration from environment
        self.config = {
            "telegram": {
                "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID")
            },
            "discord": {
                "webhook_url": os.getenv("DISCORD_WEBHOOK_URL")
            },
            "slack": {
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL")
            },
            "email": {
                "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("EMAIL_USERNAME"),
                "password": os.getenv("EMAIL_PASSWORD"),
                "recipients": os.getenv("EMAIL_RECIPIENTS", "").split(",")
            },
            "webhook": {
                "url": os.getenv("CUSTOM_WEBHOOK_URL")
            }
        }
        
        # Alert filters
        self.min_level = AlertLevel.INFO
        self.symbol_filters = []  # Empty = all symbols
        self.strategy_filters = []  # Empty = all strategies
    
    def add_channel(self, channel: NotificationChannel, config: Dict = None):
        """Add a notification channel"""
        if config:
            self.config[channel.value] = config
        print(f"Added notification channel: {channel.value}")
    
    def set_alert_filter(self, min_level: AlertLevel = None,
                        symbols: List[str] = None,
                        strategies: List[str] = None):
        """Configure alert filters"""
        if min_level:
            self.min_level = min_level
        if symbols is not None:
            self.symbol_filters = symbols
        if strategies is not None:
            self.strategy_filters = strategies
    
    def should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent based on filters"""
        if alert.level.value < self.min_level.value:
            return False
        
        if self.symbol_filters and alert.symbol not in self.symbol_filters:
            return False
        
        if self.strategy_filters and alert.strategy not in self.strategy_filters:
            return False
        
        return True
    
    async def send_alert(self, alert: Alert, channels: List[NotificationChannel] = None):
        """Send alert to specified channels"""
        if not self.should_send(alert):
            return
        
        # Add to history
        self.alert_history.append(alert.to_dict())
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Default to all configured channels
        if channels is None:
            channels = [ch for ch in NotificationChannel if self.config.get(ch.value)]
        
        # Send to each channel
        tasks = []
        for channel in channels:
            if channel == NotificationChannel.CONSOLE:
                self._send_console(alert)
            elif channel == NotificationChannel.TELEGRAM:
                tasks.append(self._send_telegram(alert))
            elif channel == NotificationChannel.DISCORD:
                tasks.append(self._send_discord(alert))
            elif channel == NotificationChannel.SLACK:
                tasks.append(self._send_slack(alert))
            elif channel == NotificationChannel.EMAIL:
                tasks.append(self._send_email(alert))
            elif channel == NotificationChannel.WEBHOOK:
                tasks.append(self._send_webhook(alert))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _send_console(self, alert: Alert):
        """Send alert to console"""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }.get(alert.level, "📢")
        
        timestamp = alert.timestamp.strftime("%H:%M:%S")
        print(f"{emoji} [{timestamp}] {alert.message}")
        if alert.symbol:
            print(f"   Symbol: {alert.symbol}")
        if alert.strategy:
            print(f"   Strategy: {alert.strategy}")
    
    async def _send_telegram(self, alert: Alert):
        """Send alert via Telegram"""
        config = self.config.get("telegram", {})
        if not config.get("bot_token") or not config.get("chat_id"):
            return
        
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        
        # Format message
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }.get(alert.level, "📢")
        
        text = f"{emoji} *{alert.message}*\n\n"
        if alert.symbol:
            text += f"📈 Symbol: `{alert.symbol}`\n"
        if alert.strategy:
            text += f"📊 Strategy: `{alert.strategy}`\n"
        text += f"⏰ Time: `{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`"
        
        payload = {
            "chat_id": config["chat_id"],
            "text": text,
            "parse_mode": "Markdown"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        print(f"Telegram error: {await response.text()}")
        except Exception as e:
            print(f"Telegram send failed: {e}")
    
    async def _send_discord(self, alert: Alert):
        """Send alert via Discord webhook"""
        config = self.config.get("discord", {})
        if not config.get("webhook_url"):
            return
        
        # Color based on level
        colors = {
            AlertLevel.INFO: 3447003,  # Blue
            AlertLevel.WARNING: 15158332,  # Orange
            AlertLevel.ERROR: 15158332,  # Orange
            AlertLevel.CRITICAL: 15548997  # Red
        }
        
        embed = {
            "title": f"{alert.message}",
            "color": colors.get(alert.level, 3447003),
            "timestamp": alert.timestamp.isoformat(),
            "fields": []
        }
        
        if alert.symbol:
            embed["fields"].append({"name": "Symbol", "value": alert.symbol, "inline": True})
        if alert.strategy:
            embed["fields"].append({"name": "Strategy", "value": alert.strategy, "inline": True})
        if alert.metadata:
            for key, value in alert.metadata.items():
                embed["fields"].append({"name": key, "value": str(value), "inline": True})
        
        payload = {"embeds": [embed]}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(config["webhook_url"], json=payload) as response:
                    if response.status != 204:
                        print(f"Discord error: {await response.text()}")
        except Exception as e:
            print(f"Discord send failed: {e}")
    
    async def _send_slack(self, alert: Alert):
        """Send alert via Slack webhook"""
        config = self.config.get("slack", {})
        if not config.get("webhook_url"):
            return
        
        # Color based on level
        colors = {
            AlertLevel.INFO: "#36a64f",  # Green
            AlertLevel.WARNING: "#ff9800",  # Orange
            AlertLevel.ERROR: "#ff9800",  # Orange
            AlertLevel.CRITICAL: "#ff0000"  # Red
        }
        
        attachment = {
            "text": alert.message,
            "color": colors.get(alert.level, "#36a64f"),
            "ts": int(alert.timestamp.timestamp()),
            "fields": []
        }
        
        if alert.symbol:
            attachment["fields"].append({"title": "Symbol", "value": alert.symbol, "short": True})
        if alert.strategy:
            attachment["fields"].append({"title": "Strategy", "value": alert.strategy, "short": True})
        
        payload = {"attachments": [attachment]}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(config["webhook_url"], json=payload) as response:
                    if response.status != 200:
                        print(f"Slack error: {await response.text()}")
        except Exception as e:
            print(f"Slack send failed: {e}")
    
    async def _send_email(self, alert: Alert):
        """Send alert via email"""
        config = self.config.get("email", {})
        if not config.get("username") or not config.get("password"):
            return
        
        recipients = [r.strip() for r in config.get("recipients", []) if r.strip()]
        if not recipients:
            return
        
        # Create message
        msg = MIMEMultipart()
        msg["From"] = config["username"]
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"[{alert.level.value.upper()}] {alert.message}"
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2>{alert.message}</h2>
            <table>
                <tr><td><strong>Level:</strong></td><td>{alert.level.value}</td></tr>
                <tr><td><strong>Time:</strong></td><td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
        """
        
        if alert.symbol:
            body += f"<tr><td><strong>Symbol:</strong></td><td>{alert.symbol}</td></tr>"
        if alert.strategy:
            body += f"<tr><td><strong>Strategy:</strong></td><td>{alert.strategy}</td></tr>"
        
        if alert.metadata:
            for key, value in alert.metadata.items():
                body += f"<tr><td><strong>{key}:</strong></td><td>{value}</td></tr>"
        
        body += """
            </table>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, "html"))
        
        try:
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"Email send failed: {e}")
    
    async def _send_webhook(self, alert: Alert):
        """Send alert to custom webhook"""
        config = self.config.get("webhook", {})
        if not config.get("url"):
            return
        
        payload = alert.to_dict()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(config["url"], json=payload) as response:
                    if response.status != 200:
                        print(f"Webhook error: {await response.text()}")
        except Exception as e:
            print(f"Webhook send failed: {e}")
    
    def get_history(self, limit: int = 100, level: AlertLevel = None) -> List[Dict]:
        """Get alert history"""
        history = self.alert_history[-limit:]
        if level:
            history = [a for a in history if a["level"] == level.value]
        return history


# Pre-built alert templates for common trading scenarios
class TradingAlerts:
    """Helper class for common trading alerts"""
    
    @staticmethod
    def trade_executed(symbol: str, side: str, quantity: float, price: float,
                      strategy: str, pnl_estimate: float = None) -> Alert:
        message = f"{'BUY' if side == 'buy' else 'SELL'} {quantity} {symbol} @ ${price:.2f}"
        metadata = {"side": side, "quantity": quantity, "price": price}
        if pnl_estimate:
            metadata["pnl_estimate"] = pnl_estimate
            message += f" (Est. P&L: ${pnl_estimate:.2f})"
        
        return Alert(message, AlertLevel.INFO, symbol, strategy, metadata)
    
    @staticmethod
    def stop_loss_triggered(symbol: str, entry_price: float, exit_price: float,
                           loss_percent: float, strategy: str) -> Alert:
        message = f"🛑 STOP LOSS: {symbol} lost {loss_percent:.2f}% (${entry_price:.2f} → ${exit_price:.2f})"
        return Alert(message, AlertLevel.WARNING, symbol, strategy,
                    {"entry": entry_price, "exit": exit_price, "loss_percent": loss_percent})
    
    @staticmethod
    def take_profit_hit(symbol: str, entry_price: float, exit_price: float,
                       profit_percent: float, strategy: str) -> Alert:
        message = f"✅ TAKE PROFIT: {symbol} gained {profit_percent:.2f}% (${entry_price:.2f} → ${exit_price:.2f})"
        return Alert(message, AlertLevel.INFO, symbol, strategy,
                    {"entry": entry_price, "exit": exit_price, "profit_percent": profit_percent})
    
    @staticmethod
    def regime_change(old_regime: str, new_regime: str, confidence: float) -> Alert:
        message = f"🔄 MARKET REGIME CHANGE: {old_regime} → {new_regime} (confidence: {confidence:.0%})"
        return Alert(message, AlertLevel.WARNING, metadata={
            "old_regime": old_regime,
            "new_regime": new_regime,
            "confidence": confidence
        })
    
    @staticmethod
    def risk_limit_warning(current_exposure: float, max_exposure: float,
                          daily_loss: float, daily_limit: float) -> Alert:
        message = f"⚠️ RISK WARNING: Exposure {current_exposure:.0%}/{max_exposure:.0%}, Daily Loss ${daily_loss:.2f}/${daily_limit:.2f}"
        return Alert(message, AlertLevel.WARNING, metadata={
            "current_exposure": current_exposure,
            "max_exposure": max_exposure,
            "daily_loss": daily_loss,
            "daily_limit": daily_limit
        })
    
    @staticmethod
    def system_error(error: str, component: str = None) -> Alert:
        message = f"🚨 SYSTEM ERROR: {error}"
        return Alert(message, AlertLevel.ERROR, metadata={"component": component})
