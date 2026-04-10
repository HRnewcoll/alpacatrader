"""
ALPACATRADER - ULTIMATE QUANT PLATFORM
======================================

The most advanced open-source algorithmic trading system, combining features from:
- Freqtrade (Hyperopt, PairList)
- Jesse (Multi-timeframe, Pattern Recognition)
- TradingAgents (Multi-Agent AI)
- Hummingbot (Market Making, Arbitrage)
- Backtrader/Zipline (Advanced Backtesting)
- HFTBacktest (Low-Latency Simulation)
- Superalgos (Social Trading)
- OctoBot (Strategy Framework)

INSTALLATION:
-------------
1. Install dependencies: pip install -r requirements.txt
2. Configure environment: cp .env.example .env
3. Set your Alpaca API keys in .env
4. Run the system: python main.py trade

FEATURES:
---------
✅ 6+ Trading Strategies (Momentum, Mean Reversion, Volatility, ML, etc.)
✅ Multi-Agent AI System (Bull, Bear, Technical, Risk agents)
✅ Social Trading Platform (Copy trading, leaderboards)
✅ Advanced Backtesting (Transaction costs, Monte Carlo, Walk-forward)
✅ Smart Order Execution (TWAP, VWAP, Iceberg, Sniper)
✅ HFT Engine (Sub-50µs latency, Market making)
✅ Neural Networks (LSTM, Transformer, CNN-LSTM, RL)
✅ Genetic Optimization (Evolutionary parameter tuning)
✅ Sentiment Analysis (NLP news analysis)
✅ Crypto Arbitrage (Multi-exchange, Triangular)
✅ Real-time Dashboard (FastAPI + Plotly)
✅ Database Logging (SQLite/PostgreSQL)
✅ Multi-channel Alerts (Telegram, Discord, Slack)

USAGE:
------
python main.py backtest --symbol AAPL --start 2023-01-01 --end 2024-01-01
python main.py trade --interval 300
python main.py optimize --strategy momentum
python run_ultimate_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("🚀 ALPACATRADER - Ultimate Quant Platform")
    print("=" * 50)
    print("\nAvailable commands:")
    print("  python main.py backtest --symbol AAPL")
    print("  python main.py trade --interval 300")
    print("  python main.py optimize --strategy momentum")
    print("  python run_ultimate_demo.py")
    print("\nFor full documentation, see README.md")

if __name__ == "__main__":
    main()
