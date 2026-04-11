# 🚀 ALPACA PAPER TRADING - QUICK START GUIDE

## ✅ YES! Your System Supports LIVE Paper Trading with Alpaca

Your trading system is **fully configured** for live paper trading using your Alpaca account. Here's how to get started:

---

## 📋 STEP 1: Get Your Alpaca Paper Trading Keys

1. Go to: https://app.alpaca.markets/paper/dashboard/overview
2. Sign up / Log in to your Alpaca account
3. Click on **"API Keys"** tab
4. Copy your **API Key** (starts with `PK...`) and **Secret Key**

---

## 🔧 STEP 2: Configure Your `.env` File

Open the `.env` file in the root directory and replace the placeholder keys:

```bash
# Edit this file:
nano .env

# Replace these lines with YOUR actual keys:
ALPACA_API_KEY=PK_YOUR_ACTUAL_PAPER_API_KEY_HERE
ALPACA_SECRET_KEY=your_actual_secret_key_here
```

**Important:** 
- Keep `ALPACA_BASE_URL=https://paper-api.alpaca.markets` for paper trading
- Keep `TRADING_MODE=paper` for simulated trading (no real money)

---

## 🎯 STEP 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ STEP 4: Start Paper Trading

### Option A: Run Continuous Trading Loop
```bash
# Runs every 5 minutes (300 seconds) by default
python main.py trade --interval 300
```

### Option B: Check Account Status First
```bash
# View your portfolio value, cash, and open positions
python main.py status
```

### Option C: Run Backtest First (Recommended)
```bash
# Test strategies on historical data before going live
python main.py backtest --days 90
```

---

## 📊 WHAT HAPPENS DURING PAPER TRADING

The system will:

1. **Fetch Real-Time Data** from Alpaca's paper trading API
2. **Detect Market Regime** (trending, mean-reverting, volatile, etc.)
3. **Run All Strategies**:
   - Mean Reversion
   - Momentum (RSI + MACD)
   - Pairs Trading
   - Volatility Breakout
4. **Apply Risk Management**:
   - Position sizing limits
   - Stop-loss / Take-profit
   - Daily loss limits
5. **Execute Trades** in your Alpaca paper account
6. **Send Alerts** (if configured) via email/Telegram/Discord
7. **Log Everything** to database and files

---

## 🛑 STOP TRADING

Press `Ctrl+C` to gracefully stop the trading loop.

---

## 📈 MONITOR YOUR TRADING

### View Live Dashboard (if enabled):
```bash
cd dashboard
python app.py
# Visit http://localhost:8000
```

### Check Database:
```bash
sqlite3 alpaca_trader.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;"
```

### View Logs:
```bash
tail -f logs/trading.log
```

---

## ⚠️ IMPORTANT NOTES

### Paper Trading vs Live Trading

| Feature | Paper Trading | Live Trading |
|---------|--------------|--------------|
| **Money** | Virtual ($100k fake) | Real money |
| **URL** | `paper-api.alpaca.markets` | `api.alpaca.markets` |
| **Risk** | Zero financial risk | Real financial risk |
| **Latency** | Simulated | Real market latency |
| **Slippage** | Minimal | Real slippage occurs |

### To Switch to LIVE Trading (REAL MONEY):

⚠️ **WARNING: Only do this when you're ready to risk real capital!**

```bash
# In .env file, change:
ALPACA_BASE_URL=https://api.alpaca.markets
TRADING_MODE=live
```

---

## 🎛️ CUSTOMIZE YOUR TRADING

### Adjust Risk Parameters:
```bash
# In .env file:
MAX_PORTFOLIO_RISK_PCT=2.0      # Max 2% risk per trade
MAX_DAILY_LOSS_PCT=5.0          # Stop if down 5% today
MAX_POSITION_SIZE_PCT=10.0      # Max 10% in one stock
STOP_LOSS_PCT=2.0               # Auto-sell if down 2%
TAKE_PROFIT_PCT=4.0             # Auto-sell if up 4%
```

### Change Watchlist:
```bash
WATCHLIST=AAPL,MSFT,NVDA,TSLA,SPY,QQQ
```

### Enable/Disable Strategies:
Edit `main.py` to comment out strategies you don't want:
```python
self.strategies: List[BaseStrategy] = [
    MeanReversionStrategy(...),
    # MomentumStrategy(...),  # Commented out
    PairsTradingStrategy(...),
]
```

---

## 🤖 ADVANCED FEATURES ACTIVE

Your system includes:

✅ **Market Regime Detection** - Adapts to market conditions  
✅ **ML Signal Enhancement** - Boosts signal confidence with AI  
✅ **Smart Order Execution** - TWAP, VWAP, Iceberg algorithms  
✅ **Multi-Agent Consensus** - 4 AI agents vote on trades  
✅ **Real-time Alerts** - Email, Telegram, Discord notifications  
✅ **Database Logging** - All trades persisted to SQLite  
✅ **Performance Analytics** - 50+ metrics (Sharpe, Sortino, etc.)  
✅ **Stop-Loss/Take-Profit** - Automatic risk management  

---

## 🆘 TROUBLESHOOTING

### "Invalid API Key" Error:
- Double-check your keys in `.env`
- Make sure there are no spaces around the `=` sign
- Verify you're using paper trading keys (start with `PK`)

### "No Symbols" Error:
- Check your `WATCHLIST` in `.env`
- Ensure symbols are valid US stocks/ETFs

### "Insufficient Buying Power":
- Reduce `MAX_POSITION_SIZE_PCT`
- Increase your paper account balance in Alpaca dashboard

### Connection Issues:
```bash
# Test connection:
python -c "from config import AppConfig; from data_handler import DataHandler; c = AppConfig(); d = DataHandler(c.alpaca); print(d.get_account())"
```

---

## 📞 SUPPORT

- Alpaca Docs: https://alpaca.markets/docs/
- Paper Trading Dashboard: https://app.alpaca.markets/paper/dashboard/overview
- Community Forum: https://forum.alpaca.markets/

---

## 🎉 YOU'RE READY TO TRADE!

```bash
# Final checklist:
# 1. ✅ Set ALPACA_API_KEY in .env
# 2. ✅ Set ALPACA_SECRET_KEY in .env
# 3. ✅ Verified TRADING_MODE=paper
# 4. ✅ Installed requirements.txt
# 5. ✅ Tested with: python main.py status

# Then run:
python main.py trade --interval 300
```

**Happy Trading! 🚀📈**
