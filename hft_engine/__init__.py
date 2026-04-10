"""
HFT (High-Frequency Trading) Engine
Ultra-low latency order processing, market making, and arbitrage detection
Inspired by Jump Trading, Citadel Securities, and Virtu Financial
"""

import time
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderBookLevel:
    price: float
    size: float
    order_count: int = 1


@dataclass
class OrderBook:
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return float('inf')
        return self.asks[0].price - self.bids[0].price
    
    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0].price + self.asks[0].price) / 2
    
    @property
    def spread_bps(self) -> float:
        if self.mid_price == 0:
            return 0.0
        return (self.spread / self.mid_price) * 10000


@dataclass
class HFTOrder:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "IOC"  # Immediate or Cancel for HFT
    order_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ns: int = 0  # Nanoseconds latency tracking
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    latency_ns: int
    commission: float = 0.0


class LatencyMonitor:
    """Track nanosecond-level latencies for HFT performance"""
    
    def __init__(self, window_size: int = 1000):
        self.latencies = deque(maxlen=window_size)
        self.order_timestamps: Dict[str, int] = {}
        
    def record_order_sent(self, order_id: str):
        self.order_timestamps[order_id] = time.time_ns()
    
    def record_fill(self, fill: Fill):
        if fill.order_id in self.order_timestamps:
            latency = time.time_ns() - self.order_timestamps[fill.order_id]
            self.latencies.append(latency)
            del self.order_timestamps[fill.order_id]
            
    @property
    def avg_latency_us(self) -> float:
        if not self.latencies:
            return 0.0
        return np.mean(self.latencies) / 1000  # Convert to microseconds
    
    @property
    def p99_latency_us(self) -> float:
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, 99) / 1000
    
    @property
    def max_latency_us(self) -> float:
        if not self.latencies:
            return 0.0
        return max(self.latencies) / 1000


class MarketMaker:
    """
    High-frequency market making strategy
    Continuously quotes bid/ask prices to capture spread
    Implements inventory management and adverse selection protection
    """
    
    def __init__(
        self,
        symbol: str,
        spread_bps: float = 10.0,
        max_inventory: float = 1000.0,
        inventory_skew_factor: float = 0.5,
        min_quote_size: float = 10.0,
        max_quote_size: float = 100.0,
        adverse_selection_threshold: float = 0.02,
    ):
        self.symbol = symbol
        self.spread_bps = spread_bps
        self.max_inventory = max_inventory
        self.inventory_skew_factor = inventory_skew_factor
        self.min_quote_size = min_quote_size
        self.max_quote_size = max_quote_size
        self.adverse_selection_threshold = adverse_selection_threshold
        
        self.current_inventory = 0.0
        self.last_trade_price = 0.0
        self.price_trend = 0.0  # -1 to 1
        self.quote_updates = 0
        self.fills_received = 0
        
    def update_market_data(self, mid_price: float, recent_trades: List[Tuple[float, float]]):
        """Update with latest market data"""
        self.last_trade_price = mid_price
        
        # Calculate short-term price trend
        if len(recent_trades) > 1:
            prices = [t[0] for t in recent_trades[-10:]]
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                self.price_trend = np.clip(np.mean(returns) * 1000, -1, 1)
    
    def calculate_quotes(self, mid_price: float) -> Tuple[float, float, float, float]:
        """
        Calculate bid/ask prices and sizes
        Returns: (bid_price, ask_price, bid_size, ask_size)
        """
        # Base spread
        half_spread = (self.spread_bps / 10000) * mid_price / 2
        
        # Inventory skew: widen spread on side we're long, narrow on short side
        inventory_ratio = self.current_inventory / self.max_inventory
        skew = inventory_ratio * self.inventory_skew_factor * half_spread
        
        # Adverse selection: widen spread if we detect informed trading
        adverse_adjustment = abs(self.price_trend) * half_spread * 0.5
        
        total_adjustment = skew + adverse_adjustment
        
        bid_price = mid_price - half_spread - total_adjustment
        ask_price = mid_price + half_spread - total_adjustment
        
        # Size adjustment based on inventory
        base_size = (self.min_quote_size + self.max_quote_size) / 2
        if self.current_inventory > 0:
            # Long inventory: offer more on bid (to sell), less on ask
            bid_size = np.clip(base_size * (1 + inventory_ratio), self.min_quote_size, self.max_quote_size)
            ask_size = np.clip(base_size * (1 - inventory_ratio), self.min_quote_size, self.max_quote_size)
        else:
            # Short inventory: offer more on ask (to buy back), less on bid
            bid_size = np.clip(base_size * (1 - abs(inventory_ratio)), self.min_quote_size, self.max_quote_size)
            ask_size = np.clip(base_size * (1 + abs(inventory_ratio)), self.min_quote_size, self.max_quote_size)
        
        return bid_price, ask_price, bid_size, ask_size
    
    def update_inventory(self, side: OrderSide, quantity: float):
        """Update inventory after fill"""
        if side == OrderSide.BUY:
            self.current_inventory += quantity
        else:
            self.current_inventory -= quantity
        
        self.fills_received += 1
        
        # Check inventory limits
        if abs(self.current_inventory) > self.max_inventory * 0.9:
            return True  # Need to reduce inventory
        return False


class ArbitrageDetector:
    """
    Detect arbitrage opportunities across multiple venues/symbols
    Supports: triangular arbitrage, statistical arbitrage, cross-exchange arb
    """
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.price_history: Dict[str, deque] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.cointegration_pairs: List[Tuple[str, str]] = []
        
    def add_price(self, symbol: str, price: float, timestamp: datetime):
        """Add price observation"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback_window)
        self.price_history[symbol].append((timestamp, price))
    
    def detect_triangular_arb(self, pair1: str, pair2: str, pair3: str) -> Optional[Dict]:
        """
        Detect triangular arbitrage opportunity
        Example: BTC/USD, ETH/BTC, ETH/USD
        """
        if not all(p in self.price_history and len(self.price_history[p]) > 0 
                   for p in [pair1, pair2, pair3]):
            return None
        
        p1 = self.price_history[pair1][-1][1]
        p2 = self.price_history[pair2][-1][1]
        p3 = self.price_history[pair3][-1][1]
        
        # Check implied vs actual price
        implied_price = p1 * p2
        discrepancy = abs(implied_price - p3) / p3
        
        if discrepancy > 0.001:  # 0.1% threshold
            direction = "buy" if implied_price > p3 else "sell"
            return {
                "type": "triangular",
                "pairs": [pair1, pair2, pair3],
                "direction": direction,
                "discrepancy_pct": discrepancy * 100,
                "implied_price": implied_price,
                "actual_price": p3,
            }
        return None
    
    def detect_statistical_arb(self) -> List[Dict]:
        """
        Detect statistical arbitrage opportunities using cointegration
        """
        opportunities = []
        
        if len(self.price_history) < 2:
            return opportunities
        
        # Build price matrix
        symbols = list(self.price_history.keys())
        min_len = min(len(self.price_history[s]) for s in symbols)
        if min_len < 30:  # Need sufficient history
            return opportunities
        
        prices = pd.DataFrame({
            s: [p[1] for p in list(self.price_history[s])[-min_len:]]
            for s in symbols
        })
        
        # Simple correlation-based pairs (full cointegration test would use engle_granger)
        corr_matrix = prices.pct_change().corr()
        
        # Find highly correlated pairs (>0.8)
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                corr = corr_matrix.loc[s1, s2]
                if abs(corr) > 0.8:
                    # Calculate z-score of spread
                    spread = prices[s1] - prices[s2] * (prices[s1].std() / prices[s2].std())
                    z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
                    
                    if abs(z_score) > 2.0:  # 2 sigma deviation
                        opportunities.append({
                            "type": "statistical",
                            "pair": (s1, s2),
                            "correlation": corr,
                            "z_score": z_score,
                            "signal": "long_short" if z_score < 0 else "short_long",
                        })
        
        return opportunities


class OrderMatchingEngine:
    """
    Ultra-fast order matching engine for HFT
    Simulates exchange matching logic with price-time priority
    """
    
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}
        self.pending_orders: Dict[str, List[HFTOrder]] = {}
        self.fills: List[Fill] = []
        self.latency_monitor = LatencyMonitor()
        self._lock = threading.Lock()
        
    def update_order_book(self, order_book: OrderBook):
        """Update order book for a symbol"""
        with self._lock:
            self.order_books[order_book.symbol] = order_book
    
    def submit_order(self, order: HFTOrder) -> Optional[Fill]:
        """Submit order and attempt immediate match"""
        self.latency_monitor.record_order_sent(order.order_id or str(time.time_ns()))
        
        with self._lock:
            if order.symbol not in self.order_books:
                return None
            
            book = self.order_books[order.symbol]
            
            # Try to match immediately
            fill = self._match_order(order, book)
            
            if fill:
                self.fills.append(fill)
                self.latency_monitor.record_fill(fill)
                return fill
            
            # Add to pending if not filled
            if order.time_in_force != "IOC":
                if order.symbol not in self.pending_orders:
                    self.pending_orders[order.symbol] = []
                heapq.heappush(self.pending_orders[order.symbol], order)
            
            return None
    
    def _match_order(self, order: HFTOrder, book: OrderBook) -> Optional[Fill]:
        """Match order against order book"""
        if order.side == OrderSide.BUY:
            # Match against asks
            while book.asks and order.quantity > 0:
                best_ask = book.asks[0]
                
                if order.order_type == OrderType.LIMIT and order.limit_price < best_ask.price:
                    break
                
                fill_qty = min(order.quantity, best_ask.size)
                order.quantity -= fill_qty
                best_ask.size -= fill_qty
                
                if best_ask.size <= 0:
                    book.asks.pop(0)
                
                fill = Fill(
                    order_id=order.order_id or str(time.time_ns()),
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=best_ask.price,
                    timestamp=datetime.now(),
                    latency_ns=time.time_ns() - int(order.timestamp.timestamp() * 1e9),
                    commission=fill_qty * best_ask.price * 0.0001  # 1 bps
                )
                return fill
        else:
            # Match against bids
            while book.bids and order.quantity > 0:
                best_bid = book.bids[0]
                
                if order.order_type == OrderType.LIMIT and order.limit_price > best_bid.price:
                    break
                
                fill_qty = min(order.quantity, best_bid.size)
                order.quantity -= fill_qty
                best_bid.size -= fill_qty
                
                if best_bid.size <= 0:
                    book.bids.pop(0)
                
                fill = Fill(
                    order_id=order.order_id or str(time.time_ns()),
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=best_bid.price,
                    timestamp=datetime.now(),
                    latency_ns=time.time_ns() - int(order.timestamp.timestamp() * 1e9),
                    commission=fill_qty * best_bid.price * 0.0001,
                )
                return fill
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get HFT engine statistics"""
        return {
            "total_fills": len(self.fills),
            "avg_latency_us": self.latency_monitor.avg_latency_us,
            "p99_latency_us": self.latency_monitor.p99_latency_us,
            "max_latency_us": self.latency_monitor.max_latency_us,
            "active_symbols": list(self.order_books.keys()),
        }


class HFTEngine:
    """
    Main HFT Engine coordinating all components
    Designed for sub-microsecond decision making
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.matching_engine = OrderMatchingEngine()
        self.market_makers: Dict[str, MarketMaker] = {
            symbol: MarketMaker(symbol) for symbol in symbols
        }
        self.arb_detector = ArbitrageDetector()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Performance metrics
        self.total_quotes = 0
        self.total_fills = 0
        self.total_pnl = 0.0
        
    async def start(self):
        """Start the HFT engine"""
        self.is_running = True
        print(f"🚀 HFT Engine started for {len(self.symbols)} symbols")
        print(f"   Symbols: {', '.join(self.symbols)}")
        
        # Run main loop
        await self._main_loop()
    
    async def _main_loop(self):
        """Main HFT loop - runs at microsecond frequency"""
        loop_count = 0
        
        while self.is_running:
            start_time = time.time_ns()
            
            # Update market data (simulated)
            for symbol in self.symbols:
                await self._process_market_data(symbol)
            
            # Generate quotes
            for symbol, mm in self.market_makers.items():
                if symbol in self.matching_engine.order_books:
                    book = self.matching_engine.order_books[symbol]
                    bid, ask, bid_size, ask_size = mm.calculate_quotes(book.mid_price)
                    self.total_quotes += 2
            
            # Check for arbitrage
            arb_opps = self.arb_detector.detect_statistical_arb()
            if arb_opps:
                print(f"📊 Found {len(arb_opps)} stat arb opportunities")
            
            # Performance tracking
            loop_time = time.time_ns() - start_time
            loop_count += 1
            
            if loop_count % 1000 == 0:
                stats = self.matching_engine.get_statistics()
                print(f"⚡ Loop {loop_count}: {loop_time/1000:.2f}µs | "
                      f"Fills: {stats['total_fills']} | "
                      f"Avg Latency: {stats['avg_latency_us']:.2f}µs")
            
            # Yield control (in real HFT, this would be busy-wait or hardware-timed)
            await asyncio.sleep(0.0001)  # 100µs cycle time
    
    async def _process_market_data(self, symbol: str):
        """Process incoming market data for a symbol"""
        # Simulate market data updates
        # In production, this would connect to exchange websocket
        pass
    
    def submit_order(self, order: HFTOrder) -> Optional[Fill]:
        """Submit order to matching engine"""
        return self.matching_engine.submit_order(order)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive HFT statistics"""
        stats = self.matching_engine.get_statistics()
        stats.update({
            "total_quotes": self.total_quotes,
            "total_fills": self.total_fills,
            "total_pnl": self.total_pnl,
            "symbols": self.symbols,
        })
        return stats
    
    def stop(self):
        """Stop the HFT engine"""
        self.is_running = False
        self.executor.shutdown(wait=False)
        print("🛑 HFT Engine stopped")


# Demo function
def demo_hft_engine():
    """Demonstrate HFT engine capabilities"""
    print("=" * 70)
    print("🚀 HFT ENGINE DEMO")
    print("=" * 70)
    
    # Initialize engine
    symbols = ["AAPL", "MSFT", "GOOGL"]
    engine = HFTEngine(symbols)
    
    # Create sample order book
    book = OrderBook(
        symbol="AAPL",
        bids=[
            OrderBookLevel(price=175.50, size=1000, order_count=5),
            OrderBookLevel(price=175.49, size=500, order_count=3),
        ],
        asks=[
            OrderBookLevel(price=175.52, size=800, order_count=4),
            OrderBookLevel(price=175.53, size=600, order_count=2),
        ],
    )
    
    engine.matching_engine.update_order_book(book)
    
    # Test market maker
    mm = engine.market_makers["AAPL"]
    mm.update_market_data(book.mid_price, [(book.mid_price, time.time())])
    
    bid, ask, bid_size, ask_size = mm.calculate_quotes(book.mid_price)
    print(f"\n📊 Market Maker Quotes:")
    print(f"   Bid: ${bid:.2f} x {bid_size:.0f}")
    print(f"   Ask: ${ask:.2f} x {ask_size:.0f}")
    print(f"   Spread: {(ask-bid):.4f} ({(ask-bid)/book.mid_price*10000:.1f} bps)")
    
    # Submit test orders
    print(f"\n⚡ Testing Order Matching:")
    
    order1 = HFTOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=175.52,
        order_id="TEST001",
    )
    
    fill1 = engine.submit_order(order1)
    if fill1:
        print(f"   ✅ Fill: {fill1.quantity} @ ${fill1.price:.2f}")
        print(f"   Latency: {fill1.latency_ns/1000:.2f}µs")
    
    # Get statistics
    stats = engine.get_statistics()
    print(f"\n📈 HFT Statistics:")
    print(f"   Total Fills: {stats['total_fills']}")
    print(f"   Avg Latency: {stats['avg_latency_us']:.2f}µs")
    print(f"   P99 Latency: {stats['p99_latency_us']:.2f}µs")
    print(f"   Max Latency: {stats['max_latency_us']:.2f}µs")
    
    # Test arbitrage detection
    print(f"\n🔍 Testing Arbitrage Detection:")
    engine.arb_detector.add_price("BTC/USD", 45000.0, datetime.now())
    engine.arb_detector.add_price("ETH/BTC", 0.065, datetime.now())
    engine.arb_detector.add_price("ETH/USD", 2925.0, datetime.now())
    
    tri_arb = engine.arb_detector.detect_triangular_arb("BTC/USD", "ETH/BTC", "ETH/USD")
    if tri_arb:
        print(f"   🎯 Triangular Arb: {tri_arb['discrepancy_pct']:.3f}% discrepancy")
    else:
        print(f"   No triangular arb detected")
    
    print("\n" + "=" * 70)
    print("✅ HFT Engine Demo Complete!")
    print("=" * 70)
    
    return engine


if __name__ == "__main__":
    demo_hft_engine()
