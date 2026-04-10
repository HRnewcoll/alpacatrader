"""
Advanced Backtester with Transaction Cost Modeling
Features:
- Slippage modeling (fixed, percentage, volume-based)
- Commission modeling (per-share, per-trade, percentage)
- Market impact modeling
- Partial fill simulation
- Realistic order execution logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeConfig:
    """Configuration for trade execution modeling"""
    commission_per_share: float = 0.0  # Commission per share
    commission_per_trade: float = 0.0  # Fixed commission per trade
    commission_pct: float = 0.001  # Commission as % of trade value (0.1% default)
    slippage_fixed: float = 0.01  # Fixed slippage in dollars
    slippage_pct: float = 0.001  # Slippage as % of price (0.1%)
    slippage_volume_impact: float = 0.0001  # Volume impact coefficient
    daily_volume_limit: float = 0.05  # Max % of daily volume we can trade
    partial_fill_prob: float = 0.0  # Probability of partial fill
    partial_fill_ratio: float = 0.8  # Ratio filled on partial fill


class AdvancedBacktester:
    """
    Event-driven backtester with realistic transaction cost modeling
    
    Inspired by: Backtrader, QuantConnect, Hudson & Thames mlfinlab
    """
    
    def __init__(self, initial_capital: float = 100000.0, config: Optional[TradeConfig] = None):
        self.initial_capital = initial_capital
        self.config = config or TradeConfig()
        self.reset()
        
    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.position_prices: Dict[str, float] = {}  # symbol -> avg entry price
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def _calculate_transaction_costs(self, symbol: str, quantity: float, price: float, 
                                     daily_volume: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Calculate total transaction costs including commission, slippage, and market impact
        
        Returns: (total_cost, slippage_cost, commission_cost)
        """
        abs_qty = abs(quantity)
        
        # Commission calculation (use max of three methods)
        commission_share = abs_qty * self.config.commission_per_share
        commission_trade = self.config.commission_per_trade if quantity != 0 else 0
        commission_pct = abs_qty * price * self.config.commission_pct
        commission = max(commission_share, commission_trade, commission_pct)
        
        # Slippage calculation
        slippage_fixed = abs_qty * self.config.slippage_fixed
        slippage_pct = abs_qty * price * self.config.slippage_pct
        
        # Volume-based market impact (simplified Kyle model)
        slippage_volume = 0.0
        if daily_volume and daily_volume > 0:
            participation_rate = abs_qty / daily_volume
            if participation_rate > self.config.daily_volume_limit:
                # Exponential impact for large trades
                impact_multiplier = np.exp(10 * (participation_rate - self.config.daily_volume_limit))
                slippage_volume = abs_qty * price * self.config.slippage_volume_impact * impact_multiplier
            else:
                slippage_volume = abs_qty * price * self.config.slippage_volume_impact * participation_rate
        
        slippage = slippage_fixed + slippage_pct + slippage_volume
        
        # Partial fill simulation
        if self.config.partial_fill_prob > 0 and np.random.random() < self.config.partial_fill_prob:
            fill_ratio = self.config.partial_fill_ratio
            quantity = quantity * fill_ratio
            logger.debug(f"Partial fill for {symbol}: {fill_ratio:.1%} of order")
        
        total_cost = commission + slippage
        return total_cost, slippage, commission
    
    def _execute_signal(self, signal: Dict, bar: pd.Series, daily_volume: Optional[float] = None) -> Optional[Dict]:
        """
        Execute a trading signal with realistic fills
        
        Args:
            signal: Dict with keys: symbol, side (buy/sell), quantity, price (optional)
            bar: OHLCV data for the bar
            daily_volume: Daily trading volume for market impact calc
            
        Returns: Executed trade dict or None if rejected
        """
        symbol = signal['symbol']
        side = signal['side']
        target_qty = signal.get('quantity', 0)
        
        if target_qty == 0:
            return None
            
        # Use bar close price as execution price (can be enhanced to use VWAP/TWAP)
        base_price = bar.get('close', bar.get('name', 0))
        if isinstance(base_price, pd.Series):
            base_price = base_price.iloc[-1] if len(base_price) > 0 else 0
        
        # Determine execution direction and adjust for slippage
        if side == 'buy':
            execution_price = base_price * (1 + self.config.slippage_pct) + self.config.slippage_fixed
            actual_qty = min(target_qty, self.capital / execution_price)  # Can't exceed capital
        else:  # sell
            execution_price = base_price * (1 - self.config.slippage_pct) - self.config.slippage_fixed
            current_pos = self.positions.get(symbol, 0)
            actual_qty = min(target_qty, current_pos)  # Can't sell more than we have
            
        if actual_qty <= 0:
            logger.debug(f"Signal rejected: insufficient {'capital' if side == 'buy' else 'position'}")
            return None
        
        # Calculate costs
        total_cost, slippage_cost, commission_cost = self._calculate_transaction_costs(
            symbol, actual_qty, execution_price, daily_volume
        )
        
        # Update positions and capital
        trade_value = actual_qty * execution_price
        
        if side == 'buy':
            # Update average entry price
            current_pos = self.positions.get(symbol, 0)
            current_value = current_pos * self.position_prices.get(symbol, 0)
            new_value = current_value + trade_value + total_cost
            new_pos = current_pos + actual_qty
            self.position_prices[symbol] = new_value / new_pos if new_pos > 0 else 0
            self.positions[symbol] = new_pos
            self.capital -= (trade_value + total_cost)
        else:
            # Calculate P&L
            entry_price = self.position_prices.get(symbol, 0)
            pnl = (execution_price - entry_price) * actual_qty - total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) - actual_qty
            if self.positions[symbol] <= 0:
                del self.positions[symbol]
                del self.position_prices[symbol]
            self.capital += (trade_value - total_cost)
        
        trade_record = {
            'symbol': symbol,
            'side': side,
            'quantity': actual_qty,
            'price': execution_price,
            'commission': commission_cost,
            'slippage': slippage_cost,
            'total_cost': total_cost,
            'pnl': pnl if side == 'sell' else 0,
            'timestamp': bar.name if hasattr(bar, 'name') else pd.Timestamp.now()
        }
        
        self.trades.append(trade_record)
        return trade_record
    
    def run(self, signals_df: pd.DataFrame, bars_df: pd.DataFrame, 
            strategy_name: str = "strategy") -> Dict:
        """
        Run backtest on historical data
        
        Args:
            signals_df: DataFrame with columns [timestamp, symbol, side, quantity]
            bars_df: MultiIndex DataFrame with OHLCV data indexed by [timestamp, symbol]
            strategy_name: Name of the strategy for reporting
            
        Returns: Dictionary with backtest results
        """
        self.reset()
        logger.info(f"Starting advanced backtest for {strategy_name}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"Transaction costs: commission={self.config.commission_pct:.2%}, "
                   f"slippage={self.config.slippage_pct:.2%}")
        
        timestamps = sorted(signals_df['timestamp'].unique())
        equity_history = [(timestamps[0], self.initial_capital)]
        
        for ts in timestamps:
            # Get bars for this timestamp
            if ts in bars_df.index.get_level_values(0).unique():
                bars_ts = bars_df.xs(ts, level=0)
            else:
                continue
                
            # Get signals for this timestamp
            signals_ts = signals_df[signals_df['timestamp'] == ts]
            
            # Execute each signal
            for _, signal in signals_ts.iterrows():
                symbol = signal['symbol']
                if symbol in bars_ts.index:
                    bar = bars_ts.loc[symbol]
                    daily_vol = bar.get('volume', None)
                    self._execute_signal(signal.to_dict(), bar, daily_vol)
            
            # Calculate portfolio equity
            total_equity = self.capital
            for symbol, qty in self.positions.items():
                if symbol in bars_ts.index:
                    price = bars_ts.loc[symbol, 'close']
                    total_equity += qty * price
                    
            equity_history.append((ts, total_equity))
            
        # Build results
        equity_series = pd.Series([x[1] for x in equity_history], 
                                  index=[x[0] for x in equity_history])
        returns = equity_series.pct_change().dropna()
        
        results = {
            'strategy_name': strategy_name,
            'initial_capital': self.initial_capital,
            'final_capital': equity_series.iloc[-1],
            'total_return': (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital,
            'equity_curve': equity_series,
            'daily_returns': returns,
            'trades': pd.DataFrame(self.trades),
            'positions_closed': len([t for t in self.trades if t['side'] == 'sell']),
            'total_commission': sum(t['commission'] for t in self.trades),
            'total_slippage': sum(t['slippage'] for t in self.trades),
            'total_transaction_costs': sum(t['total_cost'] for t in self.trades)
        }
        
        # Add risk metrics
        if len(returns) > 0:
            results['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            results['max_drawdown'] = self._calculate_max_drawdown(equity_series)
            results['win_rate'] = len([t for t in self.trades if t.get('pnl', 0) > 0]) / max(1, results['positions_closed'])
            
        logger.info(f"Backtest complete: {results['total_return']:.2%} return, "
                   f"Sharpe={results.get('sharpe_ratio', 0):.2f}, "
                   f"MaxDD={results.get('max_drawdown', 0):.2%}")
        
        return results
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        return abs(drawdown.min())
