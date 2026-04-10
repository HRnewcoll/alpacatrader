# AlpacaTrader – Advanced Algorithmic Trading System

A production-ready algorithmic trading system built on the [Alpaca](https://alpaca.markets) paper/live trading API. Supports multiple concurrent strategies, comprehensive risk management, backtesting, and detailed logging.

---

## Features

| Feature | Details |
|---|---|
| **3 strategies** | Mean Reversion (Bollinger Bands), Momentum (RSI + MACD), Pairs Trading (z-score spread) |
| **Risk management** | Position sizing, stop-loss/take-profit, daily-loss circuit breaker, drawdown protection, max-position cap |
| **Backtesting** | Walk-forward testing with Sharpe ratio, max drawdown, win rate, profit factor |
| **Paper trading** | Safe sandbox mode via Alpaca paper endpoint |
| **Logging** | Structured file + console logging for every trade decision |
| **Email alerts** | Optional SMTP alerts for trades and errors |
| **Persistence** | Trade journal stored in SQLite |
| **Graceful shutdown** | SIGINT/SIGTERM handling, retry with exponential back-off |

---

## Project Structure

```
alpacatrader/
├── strategies/
│   ├── base_strategy.py      # Abstract base + Signal dataclass
│   ├── mean_reversion.py     # Bollinger Band mean-reversion
│   ├── momentum.py           # RSI + MACD momentum
│   └── pairs_trading.py      # Statistical pairs trading
├── risk_manager/
│   └── __init__.py           # RiskManager: sizing, limits, journal
├── data_handler/
│   └── __init__.py           # Alpaca API wrapper (bars, orders, account)
├── backtest_engine/
│   └── __init__.py           # Event-driven backtester + metrics
├── alerts/
│   └── __init__.py           # Logging setup + email Alerter
├── config/
│   └── __init__.py           # Typed configuration (env / .env file)
├── tests/                    # pytest test suite (40 tests)
├── main.py                   # CLI entry point
├── requirements.txt
└── .env.example              # Configuration template
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/HRnewcoll/alpacatrader.git
cd alpacatrader
pip install -r requirements.txt
```

### 2. Configure credentials

Copy `.env.example` to `.env` and fill in your Alpaca credentials:

```bash
cp .env.example .env
```

```
# .env
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
TRADING_MODE=paper
```

> ⚠️ **Never commit your `.env` file.** It is already in `.gitignore`.

### 3. Check account status

```bash
python main.py status
```

### 4. Run a backtest

```bash
python main.py backtest --days 365
```

Sample output:

```
=======================================================
  Backtest: mean_reversion
  Period:   2024-04-10 → 2025-04-09
=======================================================
  Initial Capital :    $100,000.00
  Final Capital   :    $107,234.56
  Total Return    :       7.23%
  Num Trades      :           42
  Win Rate        :       57.14%
  Avg PnL/Trade   :       $172.23
  Max Drawdown    :      -4.12%
  Sharpe Ratio    :      1.3421
  Profit Factor   :      1.8900
=======================================================
```

### 5. Start paper trading

```bash
python main.py trade --interval 300
```

This runs a full strategy cycle every 5 minutes (300 seconds).

---

## Configuration Reference

All settings can be set via environment variables or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | — | Alpaca API key |
| `ALPACA_SECRET_KEY` | — | Alpaca secret key |
| `ALPACA_BASE_URL` | paper endpoint | API base URL |
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `MAX_PORTFOLIO_RISK_PCT` | `2.0` | Max % of portfolio risked per trade |
| `MAX_DAILY_LOSS_PCT` | `5.0` | Daily loss circuit-breaker (% of portfolio) |
| `MAX_POSITION_SIZE_PCT` | `10.0` | Max single-position size (% of portfolio) |
| `MAX_OPEN_POSITIONS` | `10` | Maximum concurrent open positions |
| `STOP_LOSS_PCT` | `2.0` | Stop-loss distance from entry (%) |
| `TAKE_PROFIT_PCT` | `4.0` | Take-profit distance from entry (%) |
| `WATCHLIST` | 10 large-caps | Comma-separated symbol list |
| `PAIRS_LIST` | `AAPL:MSFT,JPM:BAC` | Pairs for pairs trading |
| `ALERT_EMAIL` | — | Send email alerts to this address |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DB_PATH` | `trading.db` | SQLite database path |

---

## Strategies

### Mean Reversion
Uses a rolling Bollinger Band (mean ± N standard deviations).
- **BUY** when price drops below the lower band (z-score < −threshold).
- **SELL** when price rises above the upper band (z-score > +threshold).

Configure with `MR_LOOKBACK_PERIOD` and `MR_STD_THRESHOLD`.

### Momentum (RSI + MACD)
- **BUY** when RSI crosses up from oversold *and* MACD produces a bullish crossover.
- **SELL** when RSI enters overbought territory *or* MACD crosses bearishly.

Configure with `MOM_RSI_PERIOD`, `MOM_MACD_*`, `MOM_RSI_OVERBOUGHT`, `MOM_RSI_OVERSOLD`.

### Pairs Trading
Exploits mean-reverting spreads between cointegrated pairs.
- Computes OLS hedge ratio and rolling z-score of the spread.
- **ENTRY** when |z-score| > `PAIRS_Z_SCORE_ENTRY`.
- **EXIT** when |z-score| < `PAIRS_Z_SCORE_EXIT`.

Configure with `PAIRS_LIST`, `PAIRS_LOOKBACK`, `PAIRS_Z_SCORE_ENTRY`, `PAIRS_Z_SCORE_EXIT`.

---

## Risk Management

All signals pass through `RiskManager.filter_signals()` which enforces:
1. **Daily loss circuit breaker** – halts trading if daily P&L < −`MAX_DAILY_LOSS_PCT`%.
2. **Drawdown protection** – halts trading if drawdown from peak > 3× daily loss limit.
3. **Position cap** – blocks new BUY signals when `MAX_OPEN_POSITIONS` is reached.
4. **Position sizing** – adjusts quantity to never exceed `MAX_POSITION_SIZE_PCT` of portfolio.
5. **Stop-loss / take-profit** – automatically closes positions at predefined price levels each cycle.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers strategies, risk manager, and backtest engine (40 tests, no Alpaca API required).

---

## Security Notes

- API credentials are loaded from environment variables / `.env` file only.
- The `.env` file is `.gitignore`d and must never be committed.
- Paper trading mode is the default — set `TRADING_MODE=live` explicitly to trade real money.
- Always backtest and paper trade before going live.

---

## License

MIT
