#!/usr/bin/env python3
"""Test script to verify Alpaca paper trading connection."""

from config import AppConfig
from data_handler import DataHandler

def test_connection():
    """Test Alpaca paper trading connection."""
    print("=" * 70)
    print("🧪 ALPACA PAPER TRADING CONNECTION TEST")
    print("=" * 70)
    
    # Load configuration
    config = AppConfig()
    
    print(f"\n📋 Configuration:")
    print(f"   Trading Mode: {config.alpaca.trading_mode}")
    print(f"   Base URL: {config.alpaca.base_url}")
    print(f"   API Key: {config.alpaca.api_key[:10]}...")
    print(f"   Is Paper: {config.alpaca.is_paper}")
    
    # Check if credentials are set
    if config.alpaca.api_key == "PK_YOUR_PAPER_API_KEY_HERE" or not config.alpaca.api_key:
        print("\n❌ ERROR: API credentials not configured!")
        print("\n📝 ACTION REQUIRED:")
        print("   1. Open .env file")
        print("   2. Replace ALPACA_API_KEY with your actual key")
        print("   3. Replace ALPACA_SECRET_KEY with your actual secret")
        print("   4. Run this test again")
        print("\n   Get your keys from: https://app.alpaca.markets/paper/dashboard/overview")
        return False
    
    try:
        # Initialize data handler
        print("\n🔌 Connecting to Alpaca...")
        data = DataHandler(config.alpaca)
        
        # Get account info
        print("📊 Fetching account information...")
        account = data.get_account()
        
        print("\n✅ CONNECTION SUCCESSFUL!")
        print("\n" + "=" * 70)
        print("💼 ACCOUNT DETAILS")
        print("=" * 70)
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Day Trading Buying Power: ${float(account.daytrading_buying_power):,.2f}")
        
        # Get positions
        print("\n📈 OPEN POSITIONS")
        print("-" * 70)
        positions = data.get_positions()
        
        if positions:
            print(f"   Found {len(positions)} open position(s):\n")
            for pos in positions:
                pnl_pct = float(pos.unrealized_plpc) * 100
                print(f"   • {pos.symbol}")
                print(f"     Qty: {float(pos.qty):.2f}")
                print(f"     Entry: ${float(pos.avg_entry_price):.2f}")
                print(f"     Current: ${float(pos.current_price):.2f}")
                print(f"     P&L: ${float(pos.unrealized_pl):,.2f} ({pnl_pct:+.2f}%)")
                print()
        else:
            print("   No open positions")
        
        # Test market data
        print("\n📊 TESTING MARKET DATA FETCH")
        print("-" * 70)
        symbols = ["AAPL", "MSFT", "SPY"]
        print(f"   Fetching data for: {', '.join(symbols)}")
        
        bars = data.get_bars(symbols, timeframe="1Day")
        
        for symbol in symbols:
            if symbol in bars and not bars[symbol].empty:
                latest = bars[symbol].iloc[-1]
                print(f"   ✓ {symbol}: Close=${latest['close']:.2f}, Volume={latest['volume']:,}")
            else:
                print(f"   ✗ {symbol}: No data available")
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PAPER TRADING!")
        print("=" * 70)
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Review your account details above")
        print("   2. Run backtest: python main.py backtest --days 90")
        print("   3. Start paper trading: python main.py trade --interval 300")
        print("\n   Press Ctrl+C to stop trading at any time")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CONNECTION FAILED!")
        print(f"\n   Error: {str(e)}")
        print("\n🔍 TROUBLESHOOTING:")
        print("   1. Verify your API keys in .env file")
        print("   2. Make sure there are no spaces around '=' in .env")
        print("   3. Confirm you're using paper trading keys (start with 'PK')")
        print("   4. Check your internet connection")
        print("   5. Visit: https://alpaca.markets/docs/trading/paper-trading/")
        return False

if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)
