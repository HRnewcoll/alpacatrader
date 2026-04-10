"""
Order Execution Module with Smart Order Routing and Algorithms
Implements TWAP, VWAP, and adaptive execution strategies
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List
import time


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class ExecutionAlgorithm(Enum):
    MARKET = "market"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"  # Hide true order size
    SNIPER = "sniper"  # Execute on liquidity spikes


class Order:
    """Represents a trading order"""
    
    def __init__(self, symbol: str, side: OrderSide, quantity: float,
                 order_type: OrderType = OrderType.MARKET,
                 limit_price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET,
                 time_in_force: str = "DAY",
                 client_order_id: Optional[str] = None):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.algorithm = algorithm
        self.time_in_force = time_in_force
        self.client_order_id = client_order_id or f"{symbol}_{datetime.now().timestamp()}"
        
        # Execution tracking
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0
        self.status = "PENDING"
        self.created_at = datetime.now()
        self.completed_at = None
        self.fills = []
        
        # Algorithm-specific parameters
        self.twap_duration = timedelta(minutes=30)  # Default TWAP duration
        self.iceberg_display_qty = quantity * 0.1  # Show 10% for iceberg
        self.vwap_participation_rate = 0.1  # 10% of market volume
    
    def __repr__(self):
        return f"Order({self.side.value} {self.quantity} {self.symbol} @ {self.order_type.value})"


class SmartOrderRouter:
    """
    Smart Order Router that selects optimal execution strategy
    based on market conditions, order size, and urgency
    """
    
    def __init__(self, alpaca_api=None):
        self.alpaca_api = alpaca_api
        self.order_history = []
        self.market_data_cache = {}
        
        # Configuration
        self.large_order_threshold = 0.05  # 5% of ADV
        self.urgent_time_threshold = timedelta(minutes=5)
        self.slippage_tolerance = 0.002  # 0.2% max slippage
    
    def analyze_market_conditions(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Analyze current market conditions for optimal execution"""
        if df.empty:
            return {'volatility': 'normal', 'liquidity': 'normal', 'spread': 'normal'}
        
        # Calculate volatility
        returns = df['close'].pct_change()
        current_vol = returns.tail(20).std()
        avg_vol = returns.std()
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        # Estimate liquidity (using volume as proxy)
        current_vol_volume = df['volume'].tail(5).mean() if 'volume' in df.columns else 0
        avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
        liquidity_ratio = current_vol_volume / (avg_volume + 1e-8)
        
        # Determine conditions
        volatility = 'high' if vol_ratio > 1.5 else 'low' if vol_ratio < 0.7 else 'normal'
        liquidity = 'high' if liquidity_ratio > 1.3 else 'low' if liquidity_ratio < 0.7 else 'normal'
        
        return {
            'volatility': volatility,
            'liquidity': liquidity,
            'vol_ratio': vol_ratio,
            'liquidity_ratio': liquidity_ratio,
            'current_price': df['close'].iloc[-1],
            'spread_estimate': df['close'].iloc[-1] * 0.001  # Estimated spread
        }
    
    def select_algorithm(self, order: Order, market_conditions: Dict, 
                        adv: float = None) -> ExecutionAlgorithm:
        """Select optimal execution algorithm"""
        
        # Check order urgency
        is_urgent = order.time_in_force == "IOC" or order.time_in_force == "FOK"
        
        # Check order size relative to volume
        is_large = False
        if adv and order.quantity > adv * self.large_order_threshold:
            is_large = True
        
        # Decision logic
        if is_urgent:
            return ExecutionAlgorithm.MARKET
        
        if market_conditions['volatility'] == 'high':
            # In high volatility, use TWAP to average out price swings
            return ExecutionAlgorithm.TWAP
        
        if is_large:
            if market_conditions['liquidity'] == 'low':
                # Large order in low liquidity: use iceberg
                return ExecutionAlgorithm.ICEBERG
            else:
                # Large order with good liquidity: use VWAP
                return ExecutionAlgorithm.VWAP
        
        if market_conditions['liquidity'] == 'low':
            # Small order but low liquidity: wait for sniper opportunity
            return ExecutionAlgorithm.SNIPER
        
        # Default: simple market order for small orders in normal conditions
        return ExecutionAlgorithm.MARKET
    
    def create_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "market", limit_price: float = None,
                    stop_price: float = None, algorithm: str = None,
                    **kwargs) -> Order:
        """Create and configure an order with smart routing"""
        
        order = Order(
            symbol=symbol,
            side=OrderSide(side.upper()),
            quantity=quantity,
            order_type=OrderType(order_type.lower()),
            limit_price=limit_price,
            stop_price=stop_price,
            **kwargs
        )
        
        # Auto-select algorithm if not specified
        if algorithm is None:
            # Would need market data here - using default for now
            order.algorithm = ExecutionAlgorithm.MARKET
        else:
            order.algorithm = ExecutionAlgorithm(algorithm.lower())
        
        self.order_history.append(order)
        return order


class TWAPExecutor:
    """Time-Weighted Average Price execution"""
    
    def __init__(self, order: Order, duration_minutes: int = 30):
        self.order = order
        self.duration = timedelta(minutes=duration_minutes)
        self.num_slices = max(10, duration_minutes // 2)  # At least 10 slices
        self.slice_quantity = order.quantity / self.num_slices
        self.interval = self.duration / self.num_slices
        
        self.executed_quantity = 0.0
        self.slices_executed = 0
        self.start_time = None
        self.prices = []
    
    def start(self):
        """Start TWAP execution"""
        self.start_time = datetime.now()
        return self.next_slice()
    
    def next_slice(self) -> Optional[Order]:
        """Get next slice order"""
        if self.executed_quantity >= self.order.quantity:
            return None
        
        remaining = self.order.quantity - self.executed_quantity
        slice_qty = min(self.slice_quantity, remaining)
        
        slice_order = Order(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=slice_qty,
            order_type=OrderType.MARKET
        )
        
        return slice_order
    
    def update(self, fill_price: float, fill_qty: float):
        """Update execution progress"""
        self.executed_quantity += fill_qty
        self.prices.append(fill_price)
        self.slices_executed += 1
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        if not self.prices:
            return {}
        
        return {
            'executed_quantity': self.executed_quantity,
            'remaining_quantity': self.order.quantity - self.executed_quantity,
            'slices_completed': self.slices_executed,
            'total_slices': self.num_slices,
            'avg_price': np.mean(self.prices),
            'completion_percent': (self.executed_quantity / self.order.quantity) * 100
        }


class VWAPExecutor:
    """Volume-Weighted Average Price execution"""
    
    def __init__(self, order: Order, historical_volume_profile: pd.Series = None,
                 participation_rate: float = 0.1):
        self.order = order
        self.participation_rate = participation_rate
        self.volume_profile = historical_volume_profile
        
        self.executed_quantity = 0.0
        self.prices = []
        self.volumes = []
        self.target_quantities = []
    
    def calculate_target_quantity(self, current_market_volume: float) -> float:
        """Calculate target quantity based on market volume"""
        target = current_market_volume * self.participation_rate
        
        # Ensure we don't exceed remaining order quantity
        remaining = self.order.quantity - self.executed_quantity
        target = min(target, remaining)
        
        # Minimum execution threshold
        target = max(target, self.order.quantity * 0.01)
        
        return target
    
    def update(self, fill_price: float, fill_qty: float, market_volume: float):
        """Update execution progress"""
        self.executed_quantity += fill_qty
        self.prices.append(fill_price)
        self.volumes.append(market_volume)
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        if not self.prices:
            return {}
        
        total_volume = sum(self.volumes)
        vwap = sum(p * v for p, v in zip(self.prices, self.volumes)) / (total_volume + 1e-8)
        
        return {
            'executed_quantity': self.executed_quantity,
            'remaining_quantity': self.order.quantity - self.executed_quantity,
            'avg_price': np.mean(self.prices),
            'vwap': vwap,
            'completion_percent': (self.executed_quantity / self.order.quantity) * 100
        }


class IcebergExecutor:
    """Iceberg order execution - hide true order size"""
    
    def __init__(self, order: Order, display_quantity: float = None):
        self.order = order
        self.display_quantity = display_quantity or order.quantity * 0.1
        self.remaining_quantity = order.quantity
        self.active_order_id = None
        
        self.executed_quantity = 0.0
        self.num_refreshes = 0
        self.prices = []
    
    def get_visible_order(self) -> Order:
        """Get the visible portion of the iceberg"""
        visible_qty = min(self.display_quantity, self.remaining_quantity)
        
        return Order(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=visible_qty,
            order_type=self.order.order_type,
            limit_price=self.order.limit_price
        )
    
    def update(self, fill_price: float, fill_qty: float):
        """Update after partial fill"""
        self.executed_quantity += fill_qty
        self.remaining_quantity -= fill_qty
        self.prices.append(fill_price)
        
        if fill_qty >= self.display_quantity * 0.9:  # If mostly filled
            self.num_refreshes += 1
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        return {
            'executed_quantity': self.executed_quantity,
            'remaining_quantity': self.remaining_quantity,
            'display_quantity': self.display_quantity,
            'refreshes': self.num_refreshes,
            'avg_price': np.mean(self.prices) if self.prices else 0,
            'completion_percent': (self.executed_quantity / self.order.quantity) * 100
        }


class SniperExecutor:
    """Sniper execution - wait for liquidity spikes"""
    
    def __init__(self, order: Order, liquidity_threshold: float = 2.0):
        self.order = order
        self.liquidity_threshold = liquidity_threshold  # Multiply of average volume
        
        self.executed_quantity = 0.0
        self.waiting = True
        self.opportunities_missed = 0
        self.opportunities_taken = 0
        self.prices = []
        self.last_check_time = None
    
    def check_opportunity(self, current_volume: float, avg_volume: float, 
                         current_price: float) -> bool:
        """Check if conditions are right to execute"""
        volume_ratio = current_volume / (avg_volume + 1e-8)
        
        if volume_ratio >= self.liquidity_threshold:
            # Liquidity spike detected!
            self.waiting = False
            self.opportunities_taken += 1
            return True
        else:
            self.opportunities_missed += 1
            return False
    
    def execute(self, price: float, quantity: float = None) -> Order:
        """Execute when opportunity arises"""
        qty = quantity or self.order.quantity
        
        self.executed_quantity += qty
        self.prices.append(price)
        
        return Order(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=qty,
            order_type=OrderType.MARKET
        )
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        return {
            'executed_quantity': self.executed_quantity,
            'waiting': self.waiting,
            'opportunities_taken': self.opportunities_taken,
            'opportunities_missed': self.opportunities_missed,
            'avg_price': np.mean(self.prices) if self.prices else 0,
            'completion_percent': (self.executed_quantity / self.order.quantity) * 100
        }


class ExecutionMonitor:
    """Monitor and report on order execution quality"""
    
    def __init__(self):
        self.executions = []
    
    def record_execution(self, order: Order, executor, benchmark_price: float):
        """Record execution for analysis"""
        stats = executor.get_stats() if hasattr(executor, 'get_stats') else {}
        
        execution_record = {
            'order_id': order.client_order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'algorithm': order.algorithm.value,
            'requested_quantity': order.quantity,
            'executed_quantity': stats.get('executed_quantity', order.quantity),
            'avg_price': stats.get('avg_price', 0),
            'benchmark_price': benchmark_price,
            'slippage': (stats.get('avg_price', benchmark_price) - benchmark_price) / benchmark_price,
            'completion_percent': stats.get('completion_percent', 100),
            'execution_time': datetime.now() - order.created_at,
            'timestamp': datetime.now()
        }
        
        self.executions.append(execution_record)
        return execution_record
    
    def get_performance_summary(self) -> Dict:
        """Get summary of execution performance"""
        if not self.executions:
            return {}
        
        df = pd.DataFrame(self.executions)
        
        return {
            'total_executions': len(df),
            'avg_slippage_bps': df['slippage'].mean() * 10000,
            'max_slippage_bps': df['slippage'].abs().max() * 10000,
            'avg_completion_percent': df['completion_percent'].mean(),
            'by_algorithm': df.groupby('algorithm')['slippage'].mean().to_dict()
        }
