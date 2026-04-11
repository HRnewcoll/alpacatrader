# 🚀 QUICK START - Alpaca Paper Trading

## 3 Steps to Start Trading

### 1️⃣ Get Keys (2 min)
Visit: https://app.alpaca.markets/paper/dashboard/overview
- Copy API Key (starts with PK...)
- Copy Secret Key

### 2️⃣ Configure (1 min)
```bash
nano .env
```
Replace:
```
ALPACA_API_KEY=PK_YOUR_ACTUAL_KEY_HERE
ALPACA_SECRET_KEY=your_actual_secret
```

### 3️⃣ Test & Trade (30 sec)
```bash
# Test connection
python test_alpaca_connection.py

# Start trading
python main.py trade --interval 300
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `python test_alpaca_connection.py` | Test Alpaca connection |
| `python main.py status` | View account & positions |
| `python main.py backtest --days 90` | Backtest strategies |
| `python main.py trade --interval 300` | Start paper trading |

---

## What You Get

✅ **4 Trading Strategies**: Momentum, Mean Reversion, Pairs Trading, Volatility Breakout  
✅ **Market Regime Detection**: Adapts to market conditions  
✅ **Risk Management**: Stop-loss, take-profit, position limits  
✅ **Real-time Alerts**: Email, Telegram, Discord  
✅ **Database Logging**: All trades saved  
✅ **Performance Analytics**: 50+ metrics  

---

**Read full guide**: `COMPLETE_SETUP_GUIDE.md`

Happy Trading! 📈
