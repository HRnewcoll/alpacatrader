# 🚀 COMPLETE PAPER TRADING SETUP GUIDE

## ✅ YES! Your System is Ready for LIVE Paper Trading with Alpaca

Your trading system has **full production-ready support** for Alpaca paper trading. Everything is configured and ready to go!

---

## 🎯 QUICK START (3 Steps)

### Step 1: Get Your Alpaca Keys (2 minutes)
1. Visit: https://app.alpaca.markets/paper/dashboard/overview
2. Sign up / Log in
3. Click **"API Keys"** tab
4. Copy your **API Key** (starts with `PK...`) and **Secret Key**

### Step 2: Configure `.env` File (1 minute)
```bash
# Open the file
nano .env

# Replace these two lines with YOUR keys:
ALPACA_API_KEY=PK_YOUR_ACTUAL_KEY_HERE
ALPACA_SECRET_KEY=your_actual_secret_here
```

### Step 3: Test Connection (30 seconds)
```bash
python test_alpaca_connection.py
```

If you see **"🎉 ALL TESTS PASSED"**, you're ready to trade!

---

## 📋 DETAILED SETUP INSTRUCTIONS

### 1. Install All Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- ✅ Alpaca SDK (`alpaca-py`)
- ✅ Data analysis (`pandas`, `numpy`, `scipy`)
- ✅ Technical indicators (`ta`)
- ✅ Machine Learning (`scikit-learn`, `lightgbm`, `xgboost`, `torch`)
- ✅ Backtesting (`quantstats`)
- ✅ Dashboard (`fastapi`, `plotly`)
- ✅ And 40+ other professional libraries

### 2. Configure Your Environment

The `.env` file is already created with sensible defaults. You only need to change:

```bash
# REQUIRED - Your Alpaca credentials
ALPACA_API_KEY=PK_YOUR_PAPER_API_KEY_HERE        # ← Change this
ALPACA_SECRET_KEY=YOUR_SECRET_KEY_HERE            # ← Change this

# OPTIONAL - Customize these
TRADING_MODE=paper                                # Keep as 'paper' for testing
WATCHLIST=AAPL,MSFT,NVDA,TSLA,SPY,QQQ            # Your stocks
MAX_POSITION_SIZE_PCT=10.0                        # Max 10% per position
STOP_LOSS_PCT=2.0                                 # Auto-sell if down 2%
```

### 3. Verify Configuration
```bash
# Test connection to Alpaca
python test_alpaca_connection.py

# Expected output:
# ✅ CONNECTION SUCCESSFUL!
# 💼 ACCOUNT DETAILS
#    Portfolio Value: $100,000.00
#    Cash: $100,000.00
# 🎉 ALL TESTS PASSED - SYSTEM READY FOR PAPER TRADING!
```

---

## ▶️ START TRADING

### Option A: Check Account Status First
```bash
python main.py status
```

Shows:
- Portfolio value
- Available cash
- Buying power
- Open positions with P&L

### Option B: Run Backtest (Recommended Before Live Trading)
```bash
python main.py backtest --days 90
```

Tests your strategies on the last 90 days of historical data.

### Option C: Start Paper Trading (Live)
```bash
# Runs every 5 minutes (300 seconds)
python main.py trade --interval 300

# Or run every minute for faster iteration
python main.py trade --interval 60
```

---

## 🤖 WHAT HAPPENS DURING TRADING

Every trading cycle (default: 5 minutes), the system:

1. **Fetches Real-Time Data** from Alpaca
   - Latest prices for all watchlist symbols
   - Current portfolio value
   - Open positions

2. **Detects Market Regime**
   - Trending (bull/bear)
   - Mean-reverting
   - High/low volatility
   - Adapts strategy selection accordingly

3. **Runs All Strategies**
   - **Mean Reversion**: Buys oversold, sells overbought
   - **Momentum**: RSI + MACD signals
   - **Pairs Trading**: Statistical arbitrage
   - **Volatility Breakout**: ATR-based breakouts

4. **Applies Risk Management**
   - Position sizing limits
   - Stop-loss / Take-profit levels
   - Daily loss circuit breakers
   - Maximum concurrent positions

5. **Executes Trades** via Alpaca API
   - Market orders (default)
   - Or TWAP/VWAP/Iceberg algorithms
   - Logs all trades to database

6. **Sends Alerts** (if configured)
   - Email notifications
   - Telegram messages
   - Discord webhooks

7. **Logs Everything**
   - SQLite database (`alpaca_trader.db`)
   - Text logs (`logs/trading.log`)
   - Performance metrics

---

## 🛑 STOP TRADING

Press **`Ctrl+C`** to gracefully stop the trading loop.

The system will:
- Complete any pending orders
- Save all data to database
- Log shutdown message
- Exit cleanly

---

## 📊 MONITOR YOUR TRADING

### View Account Status Anytime
```bash
python main.py status
```

### Check Trade History in Database
```bash
sqlite3 alpaca_trader.db "SELECT symbol, side, qty, price, timestamp FROM trades ORDER BY timestamp DESC LIMIT 20;"
```

### View Live Logs
```bash
tail -f logs/trading.log
```

### Start Web Dashboard (Optional)
```bash
cd dashboard
python app.py
# Visit http://localhost:8000
```

Shows:
- Real-time P&L
- Open positions
- Recent trades
- Performance charts

---

## ⚙️ CUSTOMIZATION GUIDE

### Adjust Risk Parameters
Edit `.env`:
```bash
MAX_PORTFOLIO_RISK_PCT=2.0      # Max 2% risk per trade
MAX_DAILY_LOSS_PCT=5.0          # Stop if down 5% today
MAX_POSITION_SIZE_PCT=10.0      # Max 10% in one stock
MAX_OPEN_POSITIONS=10           # Max 10 concurrent positions
STOP_LOSS_PCT=2.0               # Auto-sell if down 2%
TAKE_PROFIT_PCT=4.0             # Auto-sell if up 4%
```

### Change Watchlist
```bash
WATCHLIST=AAPL,MSFT,NVDA,TSLA,META,GOOGL,AMZN,SPY,QQQ
```

### Modify Strategy Parameters
```bash
# Mean Reversion
MR_LOOKBACK_PERIOD=20
MR_STD_THRESHOLD=2.0

# Momentum
MOM_RSI_PERIOD=14
MOM_RSI_OVERBOUGHT=70
MOM_RSI_OVERSOLD=30

# Pairs Trading
PAIRS_Z_SCORE_ENTRY=2.0
PAIRS_Z_SCORE_EXIT=0.5
```

### Enable/Disable Strategies
Edit `main.py`:
```python
self.strategies = [
    MeanReversionStrategy(...),
    # MomentumStrategy(...),     # Commented out
    PairsTradingStrategy(...),
    VolatilityBreakoutStrategy(...),
]
```

---

## 🆘 TROUBLESHOOTING

### "Invalid API Key" Error
```bash
# Check your .env file:
cat .env | grep ALPACA

# Should show:
# ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxx
# ALPACA_SECRET_KEY=yyyyyyyyyyyyyyyyyyyyyyyyyy

# Make sure:
# - No spaces around '=' sign
# - Using paper keys (start with 'PK')
# - Keys are copied correctly (no extra characters)
```

### "Insufficient Buying Power"
```bash
# Reduce position size in .env:
MAX_POSITION_SIZE_PCT=5.0

# Or increase account balance in Alpaca dashboard
```

### "No Symbols" or "Empty Watchlist"
```bash
# Check WATCHLIST in .env:
WATCHLIST=AAPL,MSFT,SPY

# Make sure:
# - No spaces between symbols
# - Symbols are valid US stocks/ETFs
# - Comma-separated (not semicolon)
```

### Connection Timeout
```bash
# Test internet connection:
ping api.alpaca.markets

# Check Alpaca status:
# https://status.alpaca.markets/

# Retry with longer timeout:
python test_alpaca_connection.py
```

### Module Import Errors
```bash
# Reinstall dependencies:
pip install -r requirements.txt --upgrade

# Or install specific package:
pip install alpaca-py --upgrade
```

---

## 📈 PAPER TRADING vs LIVE TRADING

| Feature | Paper Trading | Live Trading |
|---------|--------------|--------------|
| **Money** | Virtual ($100k fake) | Real money 💰 |
| **API URL** | `paper-api.alpaca.markets` | `api.alpaca.markets` |
| **Risk** | Zero financial risk | Real financial risk ⚠️ |
| **Latency** | Simulated (~100ms) | Real market latency |
| **Slippage** | Minimal | Real slippage occurs |
| **Order Fills** | Ideal fills | Real market fills |
| **Emotions** | No stress | Psychological pressure |

### To Switch to LIVE Trading (REAL MONEY)

⚠️ **WARNING: Only do this when you're consistently profitable in paper trading!**

```bash
# In .env file, change TWO lines:
ALPACA_BASE_URL=https://api.alpaca.markets    # ← Changed from paper-api
TRADING_MODE=live                              # ← Changed from paper
```

Then restart:
```bash
python main.py trade --interval 300
```

---

## 🎓 BEST PRACTICES

### Before Going Live
1. ✅ Paper trade for at least 2-4 weeks
2. ✅ Achieve consistent profitability
3. ✅ Understand max drawdown
4. ✅ Test in different market conditions
5. ✅ Review all trades in database
6. ✅ Start with small position sizes

### Risk Management Rules
- Never risk more than 2% per trade
- Set daily loss limits (5% max)
- Use stop-losses on every position
- Diversify across sectors
- Don't overtrade (max 10 positions)

### Monitoring
- Check performance daily
- Review weekly reports
- Adjust parameters monthly
- Keep trading journal
- Monitor market regime changes

---

## 📞 RESOURCES

### Alpaca Documentation
- [Paper Trading Guide](https://alpaca.markets/docs/trading/paper-trading/)
- [API Reference](https://alpaca.markets/docs/api-references/)
- [Trading Client](https://alpaca.markets/docs/python-sdk/)

### Community Support
- [Alpaca Forum](https://forum.alpaca.markets/)
- [Discord Community](https://discord.gg/alpaca)
- [GitHub Issues](https://github.com/alpacahq/alpaca-py/issues)

### Educational Resources
- [QuantConnect Education](https://www.quantconnect.com/learn)
- [Investopedia Trading](https://www.investopedia.com/trading/)
- [BabyPips (for concepts)](https://www.babypips.com/)

---

## 🎉 YOU'RE READY!

### Final Checklist
- [ ] ✅ Got Alpaca paper trading account
- [ ] ✅ Set `ALPACA_API_KEY` in `.env`
- [ ] ✅ Set `ALPACA_SECRET_KEY` in `.env`
- [ ] ✅ Verified `TRADING_MODE=paper`
- [ ] ✅ Installed `requirements.txt`
- [ ] ✅ Ran `test_alpaca_connection.py` successfully
- [ ] ✅ Reviewed risk parameters
- [ ] ✅ Understood stop-loss/take-profit settings

### Start Trading Now!
```bash
# Run this command:
python main.py trade --interval 300
```

**Good luck and happy trading! 🚀📈**

Remember: Paper trading is for learning. Always use proper risk management!
