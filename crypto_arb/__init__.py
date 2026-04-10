"""
Crypto Arbitrage Detector
Multi-exchange price monitoring and arbitrage opportunity detection
Inspired by Hummingbot and crypto market making firms
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class ExchangePrice:
    """Price data from a single exchange"""
    exchange: str
    symbol: str
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        if self.mid_price == 0:
            return 0.0
        return (self.spread / self.mid_price) * 100


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    pair: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    max_volume: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self):
        return (f"Arb: {self.pair} | Buy @{self.buy_exchange} ${self.buy_price:.2f} | "
                f"Sell @{self.sell_exchange} ${self.sell_price:.2f} | "
                f"Profit: {self.profit_pct:.2f}%")


class ExchangeConnector:
    """
    Simulates exchange API connections
    In production, would connect to real exchange APIs (Binance, Coinbase, Kraken, etc.)
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.prices: Dict[str, ExchangePrice] = {}
        self.last_update: Optional[datetime] = None
        
    def update_prices(self, prices: Dict[str, Tuple[float, float]]):
        """
        Update prices for multiple symbols
        prices: {symbol: (bid, ask)}
        """
        for symbol, (bid, ask) in prices.items():
            self.prices[symbol] = ExchangePrice(
                exchange=self.exchange_name,
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp=datetime.now(),
            )
        self.last_update = datetime.now()
    
    def get_price(self, symbol: str) -> Optional[ExchangePrice]:
        """Get current price for a symbol"""
        return self.prices.get(symbol)


class CryptoArbitrageDetector:
    """
    Detects arbitrage opportunities across multiple exchanges
    Supports: spatial arbitrage, triangular arbitrage, funding rate arb
    """
    
    def __init__(self):
        self.exchanges: Dict[str, ExchangeConnector] = {}
        self.price_history: Dict[str, Dict[str, List[Tuple[datetime, float]]]] = defaultdict(lambda: defaultdict(list))
        self.opportunities_found: List[ArbitrageOpportunity] = []
        
        # Configuration
        self.min_profit_threshold = 0.1  # Minimum profit % to consider
        self.max_age_seconds = 5  # Maximum age of price data
        
    def add_exchange(self, exchange_name: str) -> ExchangeConnector:
        """Add exchange connector"""
        connector = ExchangeConnector(exchange_name)
        self.exchanges[exchange_name] = connector
        return connector
    
    def update_prices(self, exchange: str, prices: Dict[str, float]):
        """
        Update prices for an exchange
        prices: {symbol: mid_price}
        """
        if exchange not in self.exchanges:
            self.add_exchange(exchange)
        
        # Convert mid prices to bid/ask (simulate spread)
        bid_ask_prices = {}
        for symbol, mid in prices.items():
            spread_pct = 0.001  # 0.1% spread
            bid = mid * (1 - spread_pct / 2)
            ask = mid * (1 + spread_pct / 2)
            bid_ask_prices[symbol] = (bid, ask)
            
            # Record history
            self.price_history[exchange][symbol].append((datetime.now(), mid))
            # Keep only last 1000 records
            if len(self.price_history[exchange][symbol]) > 1000:
                self.price_history[exchange][symbol] = self.price_history[exchange][symbol][-1000:]
        
        self.exchanges[exchange].update_prices(bid_ask_prices)
    
    def find_arbitrage_opportunities(
        self,
        min_profit_pct: float = 0.05,
        symbols: Optional[List[str]] = None,
    ) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities across exchanges
        Returns list of opportunities sorted by profit %
        """
        opportunities = []
        
        # Get all symbols across exchanges
        all_symbols = set()
        for exchange in self.exchanges.values():
            all_symbols.update(exchange.prices.keys())
        
        if symbols:
            all_symbols = all_symbols.intersection(set(symbols))
        
        # Check each symbol
        for symbol in all_symbols:
            # Get prices from all exchanges
            exchange_prices = []
            for exchange_name, connector in self.exchanges.items():
                price = connector.get_price(symbol)
                if price and (datetime.now() - price.timestamp).total_seconds() < self.max_age_seconds:
                    exchange_prices.append(price)
            
            # Need at least 2 exchanges
            if len(exchange_prices) < 2:
                continue
            
            # Find best bid and ask across exchanges
            best_bid_exchange = max(exchange_prices, key=lambda p: p.bid)
            best_ask_exchange = min(exchange_prices, key=lambda p: p.ask)
            
            # Check if profitable (buy at ask, sell at bid)
            if best_bid_exchange.bid > best_ask_exchange.ask:
                profit_pct = ((best_bid_exchange.bid - best_ask_exchange.ask) / best_ask_exchange.ask) * 100
                
                if profit_pct >= min_profit_pct:
                    # Calculate max volume (limited by order book size)
                    max_volume = min(best_bid_exchange.bid_size or float('inf'), 
                                    best_ask_exchange.ask_size or float('inf'))
                    
                    opp = ArbitrageOpportunity(
                        pair=symbol,
                        buy_exchange=best_ask_exchange.exchange,
                        sell_exchange=best_bid_exchange.exchange,
                        buy_price=best_ask_exchange.ask,
                        sell_price=best_bid_exchange.bid,
                        profit_pct=profit_pct,
                        max_volume=max_volume if max_volume != float('inf') else 1000000,
                    )
                    opportunities.append(opp)
                    self.opportunities_found.append(opp)
        
        # Sort by profit descending
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        
        return opportunities
    
    def detect_triangular_arbitrage(
        self,
        exchange: str,
        pairs: List[Tuple[str, str, str]],
        min_profit_pct: float = 0.1,
    ) -> List[Dict]:
        """
        Detect triangular arbitrage on a single exchange
        Example pairs: [("BTC", "USDT"), ("ETH", "BTC"), ("ETH", "USDT")]
        """
        opportunities = []
        
        if exchange not in self.exchanges:
            return opportunities
        
        connector = self.exchanges[exchange]
        
        for base, quote, cross in pairs:
            # Get prices
            base_quote = connector.get_price(f"{base}/{quote}")
            cross_base = connector.get_price(f"{cross}/{base}")
            cross_quote = connector.get_price(f"{cross}/{quote}")
            
            if not all([base_quote, cross_base, cross_quote]):
                continue
            
            # Triangular arbitrage: start with quote currency
            # 1. Buy base with quote
            # 2. Buy cross with base
            # 3. Sell cross for quote
            
            initial_amount = 1000  # Start with 1000 units
            
            # Step 1: Buy base
            base_amount = initial_amount / base_quote.ask
            
            # Step 2: Buy cross with base
            cross_amount = base_amount / cross_base.ask
            
            # Step 3: Sell cross for quote
            final_amount = cross_amount * cross_quote.bid
            
            profit_pct = ((final_amount - initial_amount) / initial_amount) * 100
            
            if abs(profit_pct) >= min_profit_pct:
                opportunities.append({
                    'type': 'triangular',
                    'exchange': exchange,
                    'path': f"{quote} -> {base} -> {cross} -> {quote}",
                    'profit_pct': profit_pct,
                    'direction': 'positive' if profit_pct > 0 else 'negative',
                })
        
        return opportunities
    
    def get_exchange_statistics(self) -> Dict:
        """Get statistics about monitored exchanges"""
        stats = {
            'exchanges': {},
            'total_opportunities': len(self.opportunities_found),
        }
        
        for name, connector in self.exchanges.items():
            stats['exchanges'][name] = {
                'symbols': len(connector.prices),
                'last_update': connector.last_update.isoformat() if connector.last_update else None,
            }
        
        return stats


def demo_crypto_arbitrage():
    """Demonstrate crypto arbitrage detection"""
    print("=" * 70)
    print("₿ CRYPTO ARBITRAGE DETECTOR DEMO")
    print("=" * 70)
    
    detector = CryptoArbitrageDetector()
    
    # Add exchanges
    exchanges_config = {
        'binance': {'BTC/USDT': 45100.0, 'ETH/USDT': 2920.0, 'SOL/USDT': 98.5},
        'coinbase': {'BTC/USDT': 45150.0, 'ETH/USDT': 2915.0, 'SOL/USDT': 98.8},
        'kraken': {'BTC/USDT': 45080.0, 'ETH/USDT': 2925.0, 'SOL/USDT': 98.2},
        'ftx': {'BTC/USDT': 45120.0, 'ETH/USDT': 2918.0, 'SOL/USDT': 98.6},
    }
    
    print(f"\n🔍 Monitoring {len(exchanges_config)} exchanges:")
    for exchange in exchanges_config.keys():
        print(f"   • {exchange.capitalize()}")
    
    # Update prices
    for exchange, prices in exchanges_config.items():
        detector.update_prices(exchange, prices)
    
    print("\n📊 Current Prices:")
    for symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
        prices = [(ex, detector.exchanges[ex].get_price(symbol).mid_price if detector.exchanges[ex].get_price(symbol) else None) 
                  for ex in exchanges_config.keys()]
        print(f"   {symbol}:")
        for ex, price in prices:
            if price:
                print(f"      {ex.capitalize()}: ${price:.2f}")
    
    # Find arbitrage opportunities
    print("\n" + "-" * 70)
    print("\n💰 Scanning for arbitrage opportunities...\n")
    
    opportunities = detector.find_arbitrage_opportunities(min_profit_pct=0.02)
    
    if opportunities:
        print(f"✅ Found {len(opportunities)} arbitrage opportunities:\n")
        for i, opp in enumerate(opportunities[:5], 1):
            print(f"{i}. {opp.pair}")
            print(f"   Buy:  @{opp.buy_exchange.capitalize()} ${opp.buy_price:.2f}")
            print(f"   Sell: @{opp.sell_exchange.capitalize()} ${opp.sell_price:.2f}")
            print(f"   Profit: {opp.profit_pct:.2f}% | Max Volume: ${opp.max_volume:.2f}")
            print()
    else:
        print("   No significant arbitrage opportunities detected")
        print("   (Try increasing price differences or lowering threshold)")
    
    # Get statistics
    stats = detector.get_exchange_statistics()
    print("-" * 70)
    print(f"\n📈 System Statistics:")
    print(f"   Exchanges Monitored: {len(stats['exchanges'])}")
    print(f"   Total Opportunities Found: {stats['total_opportunities']}")
    
    print("\n" + "=" * 70)
    print("✅ Crypto Arbitrage Demo Complete!")
    print("=" * 70)
    
    return detector


if __name__ == "__main__":
    demo_crypto_arbitrage()
