"""Configuration management for the Alpaca trading system."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, str(default)).lower()
    return v in ("1", "true", "yes")


@dataclass
class AlpacaConfig:
    api_key: str = field(default_factory=lambda: _env_str("ALPACA_API_KEY"))
    secret_key: str = field(default_factory=lambda: _env_str("ALPACA_SECRET_KEY"))
    base_url: str = field(
        default_factory=lambda: _env_str(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
    )
    trading_mode: str = field(
        default_factory=lambda: _env_str("TRADING_MODE", "paper")
    )

    @property
    def is_paper(self) -> bool:
        return self.trading_mode.lower() == "paper"


@dataclass
class RiskConfig:
    max_portfolio_risk_pct: float = field(
        default_factory=lambda: _env_float("MAX_PORTFOLIO_RISK_PCT", 2.0)
    )
    max_daily_loss_pct: float = field(
        default_factory=lambda: _env_float("MAX_DAILY_LOSS_PCT", 5.0)
    )
    max_position_size_pct: float = field(
        default_factory=lambda: _env_float("MAX_POSITION_SIZE_PCT", 10.0)
    )
    max_open_positions: int = field(
        default_factory=lambda: _env_int("MAX_OPEN_POSITIONS", 10)
    )
    stop_loss_pct: float = field(
        default_factory=lambda: _env_float("STOP_LOSS_PCT", 2.0)
    )
    take_profit_pct: float = field(
        default_factory=lambda: _env_float("TAKE_PROFIT_PCT", 4.0)
    )
    trailing_stop_pct: float = field(
        default_factory=lambda: _env_float("TRAILING_STOP_PCT", 0.0)
    )
    max_bid_ask_spread_pct: float = field(
        default_factory=lambda: _env_float("MAX_BID_ASK_SPREAD_PCT", 0.0)
    )


@dataclass
class MeanReversionConfig:
    lookback_period: int = field(
        default_factory=lambda: _env_int("MR_LOOKBACK_PERIOD", 20)
    )
    std_threshold: float = field(
        default_factory=lambda: _env_float("MR_STD_THRESHOLD", 2.0)
    )


@dataclass
class MomentumConfig:
    rsi_period: int = field(
        default_factory=lambda: _env_int("MOM_RSI_PERIOD", 14)
    )
    macd_fast: int = field(default_factory=lambda: _env_int("MOM_MACD_FAST", 12))
    macd_slow: int = field(default_factory=lambda: _env_int("MOM_MACD_SLOW", 26))
    macd_signal: int = field(
        default_factory=lambda: _env_int("MOM_MACD_SIGNAL", 9)
    )
    rsi_overbought: float = field(
        default_factory=lambda: _env_float("MOM_RSI_OVERBOUGHT", 70.0)
    )
    rsi_oversold: float = field(
        default_factory=lambda: _env_float("MOM_RSI_OVERSOLD", 30.0)
    )


@dataclass
class PairsTradingConfig:
    lookback: int = field(
        default_factory=lambda: _env_int("PAIRS_LOOKBACK", 60)
    )
    z_score_entry: float = field(
        default_factory=lambda: _env_float("PAIRS_Z_SCORE_ENTRY", 2.0)
    )
    z_score_exit: float = field(
        default_factory=lambda: _env_float("PAIRS_Z_SCORE_EXIT", 0.5)
    )

    @property
    def pairs(self) -> List[Tuple[str, str]]:
        raw = _env_str("PAIRS_LIST", "AAPL:MSFT,JPM:BAC")
        result = []
        for pair in raw.split(","):
            parts = pair.strip().split(":")
            if len(parts) == 2:
                result.append((parts[0].strip(), parts[1].strip()))
        return result


@dataclass
class AlertConfig:
    email: str = field(default_factory=lambda: _env_str("ALERT_EMAIL", ""))
    smtp_host: str = field(
        default_factory=lambda: _env_str("SMTP_HOST", "smtp.gmail.com")
    )
    smtp_port: int = field(default_factory=lambda: _env_int("SMTP_PORT", 587))
    smtp_user: str = field(default_factory=lambda: _env_str("SMTP_USER", ""))
    smtp_password: str = field(
        default_factory=lambda: _env_str("SMTP_PASSWORD", "")
    )
    telegram_token: str = field(
        default_factory=lambda: _env_str("TELEGRAM_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: _env_str("TELEGRAM_CHAT_ID", "")
    )
    slack_webhook_url: str = field(
        default_factory=lambda: _env_str("SLACK_WEBHOOK_URL", "")
    )


@dataclass
class SchedulerConfig:
    use_market_hours: bool = field(
        default_factory=lambda: _env_bool("USE_MARKET_HOURS", True)
    )
    warmup_minutes: int = field(
        default_factory=lambda: _env_int("SCHEDULER_WARMUP_MINUTES", 5)
    )
    closeout_buffer_minutes: int = field(
        default_factory=lambda: _env_int("SCHEDULER_CLOSEOUT_MINUTES", 5)
    )


@dataclass
class NewsConfig:
    enabled: bool = field(
        default_factory=lambda: _env_bool("NEWS_ENABLED", True)
    )
    lookback_hours: int = field(
        default_factory=lambda: _env_int("NEWS_LOOKBACK_HOURS", 24)
    )
    negative_threshold: float = field(
        default_factory=lambda: _env_float("NEWS_NEGATIVE_THRESHOLD", -0.3)
    )
    positive_threshold: float = field(
        default_factory=lambda: _env_float("NEWS_POSITIVE_THRESHOLD", 0.2)
    )


@dataclass
class ScreenerConfig:
    enabled: bool = field(
        default_factory=lambda: _env_bool("SCREENER_ENABLED", True)
    )
    top_n: int = field(default_factory=lambda: _env_int("SCREENER_TOP_N", 20))
    min_avg_volume: int = field(
        default_factory=lambda: _env_int("SCREENER_MIN_AVG_VOLUME", 500_000)
    )
    lookback_days: int = field(
        default_factory=lambda: _env_int("SCREENER_LOOKBACK_DAYS", 10)
    )

    @property
    def universe(self) -> List[str]:
        raw = _env_str("SCREENER_UNIVERSE", "")
        return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass
class RebalancerConfig:
    enabled: bool = field(
        default_factory=lambda: _env_bool("REBALANCER_ENABLED", True)
    )
    check_interval_hours: int = field(
        default_factory=lambda: _env_int("REBALANCER_CHECK_INTERVAL_HOURS", 168)
    )


@dataclass
class OptimizerConfig:
    lookback_months: int = field(
        default_factory=lambda: _env_int("OPTIMIZER_LOOKBACK_MONTHS", 6)
    )
    write_env: bool = field(
        default_factory=lambda: _env_bool("OPTIMIZER_WRITE_ENV", False)
    )


@dataclass
class DashboardConfig:
    enabled: bool = field(
        default_factory=lambda: _env_bool("DASHBOARD_ENABLED", False)
    )
    host: str = field(
        default_factory=lambda: _env_str("DASHBOARD_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: _env_int("DASHBOARD_PORT", 8080)
    )


@dataclass
class BreakoutConfig:
    lookback_days: int = field(
        default_factory=lambda: _env_int("BREAKOUT_LOOKBACK_DAYS", 20)
    )
    volume_factor: float = field(
        default_factory=lambda: _env_float("BREAKOUT_VOLUME_FACTOR", 1.5)
    )


@dataclass
class VWAPReversionConfig:
    std_threshold: float = field(
        default_factory=lambda: _env_float("VWAP_STD_THRESHOLD", 1.5)
    )
    lookback_days: int = field(
        default_factory=lambda: _env_int("VWAP_LOOKBACK_DAYS", 20)
    )


@dataclass
class MultiTimeframeConfig:
    enabled: bool = field(
        default_factory=lambda: _env_bool("MTF_ENABLED", False)
    )
    trend_lookback_days: int = field(
        default_factory=lambda: _env_int("MTF_TREND_LOOKBACK_DAYS", 20)
    )


@dataclass
class AppConfig:
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    mean_reversion: MeanReversionConfig = field(
        default_factory=MeanReversionConfig
    )
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    pairs_trading: PairsTradingConfig = field(
        default_factory=PairsTradingConfig
    )
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)
    vwap_reversion: VWAPReversionConfig = field(
        default_factory=VWAPReversionConfig
    )
    multi_timeframe: MultiTimeframeConfig = field(
        default_factory=MultiTimeframeConfig
    )
    alerts: AlertConfig = field(default_factory=AlertConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    rebalancer: RebalancerConfig = field(default_factory=RebalancerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    db_path: str = field(
        default_factory=lambda: _env_str("DB_PATH", "trading.db")
    )
    log_level: str = field(
        default_factory=lambda: _env_str("LOG_LEVEL", "INFO")
    )
    log_dir: str = field(
        default_factory=lambda: _env_str("LOG_DIR", "logs")
    )

    @property
    def watchlist(self) -> List[str]:
        raw = _env_str(
            "WATCHLIST",
            "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,JPM,BAC,WMT",
        )
        return [s.strip() for s in raw.split(",") if s.strip()]
