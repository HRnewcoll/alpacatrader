"""
Freqtrade-Inspired Hyperopt Engine
===================================
Features:
- Strategy parameter optimization with Scikit-optimize
- Dry-run/live mode switching
- Pairlist generators (VolumePairList, StaticPairList, PerformancePairList)
- Stoploss on loss/profit/ROI
- Performance tracking per pair
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not installed. Install with: pip install scikit-optimize")

# Use absolute imports instead of relative to avoid import errors
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_backtest.backtester import AdvancedBacktester as Backtester
from strategies.base_strategy import BaseStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HyperoptResult:
    """Stores hyperopt optimization results"""
    params: Dict[str, Any]
    score: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    trades_count: int
    evaluation_count: int
    
    def to_dict(self) -> Dict:
        return {
            'params': self.params,
            'metrics': {
                'score': self.score,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'total_return': self.total_return,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'trades_count': self.trades_count
            },
            'evaluations': self.evaluation_count
        }


class PairListGenerator:
    """Generate trading pair lists inspired by Freqtrade"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.exchange = config.get('exchange', 'alpaca')
        
    def generate_volume_pairlist(
        self, 
        quote_currency: str = 'USD',
        top_n: int = 20,
        min_volume_24h: float = 100000,
        sort_key: str = 'quoteVolume'
    ) -> List[str]:
        """
        Generate pair list sorted by 24h volume
        
        Args:
            quote_currency: Quote currency (USD, BTC, etc.)
            top_n: Number of top pairs to return
            min_volume_24h: Minimum 24h volume filter
            sort_key: Volume metric to sort by
            
        Returns:
            List of symbol strings
        """
        # Simulated - in production would fetch from exchange API
        # For Alpaca, we'd use alpaca.get_assets() and market data
        base_pairs = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 
                      'META', 'AMD', 'SPY', 'QQQ', 'IWM', 'DIA',
                      'COIN', 'PLTR', 'SOFI', 'NIO', 'BABA', 'DIS',
                      'NFLX', 'INTC', 'PYPL', 'SQ', 'SHOP', 'UBER']
        
        # Filter and sort (simulated volumes)
        np.random.seed(42)
        volumes = {pair: np.random.uniform(1e6, 1e9) for pair in base_pairs}
        
        filtered = [
            pair for pair in base_pairs 
            if volumes[pair] >= min_volume_24h
        ]
        
        sorted_pairs = sorted(
            filtered, 
            key=lambda x: volumes[x], 
            reverse=True
        )[:top_n]
        
        logger.info(f"Generated volume pairlist: {len(sorted_pairs)} pairs")
        return sorted_pairs
    
    def generate_static_pairlist(self, pairs: List[str]) -> List[str]:
        """Return static list of pairs"""
        logger.info(f"Using static pairlist: {len(pairs)} pairs")
        return pairs
    
    def generate_performance_pairlist(
        self,
        lookback_days: int = 7,
        top_n: int = 10,
        performance_metric: str = 'return'
    ) -> List[str]:
        """
        Generate pair list based on recent performance
        
        Args:
            lookback_days: Days to look back for performance
            top_n: Top performers to select
            performance_metric: Metric to rank by ('return', 'sharpe', 'momentum')
        """
        # Would fetch historical performance in production
        base_pairs = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN',
                      'META', 'AMD', 'SPY', 'QQQ', 'COIN', 'PLTR']
        
        np.random.seed(datetime.now().timestamp())
        performances = {
            pair: np.random.uniform(-0.15, 0.25) 
            for pair in base_pairs
        }
        
        sorted_pairs = sorted(
            performances.keys(),
            key=lambda x: performances[x],
            reverse=True
        )[:top_n]
        
        logger.info(f"Generated performance pairlist: {sorted_pairs}")
        return sorted_pairs
    
    def generate_spread_pairlist(
        self,
        min_spread: float = 0.001,
        max_spread: float = 0.05
    ) -> List[Tuple[str, str]]:
        """Generate pairs for spread/arbitrage trading"""
        sectors = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'ev': ['TSLA', 'NIO', 'RIVN', 'LCID'],
            'semiconductor': ['NVDA', 'AMD', 'INTC', 'TSM'],
            'etf': ['SPY', 'QQQ', 'IWM', 'DIA']
        }
        
        pairs = []
        for sector, stocks in sectors.items():
            for i, stock1 in enumerate(stocks):
                for stock2 in stocks[i+1:]:
                    pairs.append((stock1, stock2))
        
        logger.info(f"Generated {len(pairs)} spread pairs")
        return pairs


class StoplossManager:
    """Advanced stoploss management inspired by Freqtrade"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def calculate_stoploss(
        self,
        entry_price: float,
        current_price: float,
        is_long: bool,
        stoploss_type: str = 'fixed',
        stoploss_value: float = 0.05,
        roi_table: Optional[Dict[float, float]] = None,
        trailing_stop: bool = False,
        trailing_stop_positive: float = 0.02,
        trailing_stop_positive_offset: float = 0.03
    ) -> float:
        """
        Calculate dynamic stoploss price
        
        Args:
            entry_price: Entry price of position
            current_price: Current market price
            is_long: True for long positions
            stoploss_type: 'fixed', 'roi', 'trailing'
            stoploss_value: Base stoploss percentage
            roi_table: ROI table {profit_pct: stoploss_pct}
            trailing_stop: Enable trailing stoploss
            trailing_stop_positive: Trailing stop offset
            trailing_stop_positive_offset: Offset to activate trailing
            
        Returns:
            Stoploss price level
        """
        if is_long:
            if stoploss_type == 'fixed':
                stop_price = entry_price * (1 - stoploss_value)
                
            elif stoploss_type == 'trailing' and trailing_stop:
                profit_pct = (current_price - entry_price) / entry_price
                
                if profit_pct > trailing_stop_positive_offset:
                    # Activate trailing stop
                    stop_price = current_price * (1 - trailing_stop_positive)
                else:
                    stop_price = entry_price * (1 - stoploss_value)
                    
            elif stoploss_type == 'roi' and roi_table:
                profit_pct = (current_price - entry_price) / entry_price
                
                # Find applicable ROI level
                applicable_stop = stoploss_value
                for roi_level, sl_level in sorted(roi_table.items()):
                    if profit_pct >= roi_level:
                        applicable_stop = sl_level
                        
                stop_price = entry_price * (1 - applicable_stop)
            else:
                stop_price = entry_price * (1 - stoploss_value)
                
        else:  # Short position
            if stoploss_type == 'fixed':
                stop_price = entry_price * (1 + stoploss_value)
                
            elif stoploss_type == 'trailing' and trailing_stop:
                profit_pct = (entry_price - current_price) / entry_price
                
                if profit_pct > trailing_stop_positive_offset:
                    stop_price = current_price * (1 + trailing_stop_positive)
                else:
                    stop_price = entry_price * (1 + stoploss_value)
            else:
                stop_price = entry_price * (1 + stoploss_value)
        
        return round(stop_price, 2)


class HyperoptEngine:
    """
    Strategy parameter optimization engine
    Inspired by Freqtrade's hyperopt module
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        config: Dict,
        spaces: Optional[List[str]] = None
    ):
        """
        Initialize hyperopt engine
        
        Args:
            strategy: Strategy instance to optimize
            config: Configuration dictionary
            spaces: Optimization spaces ['buy', 'sell', 'roi', 'stoploss', 'all']
        """
        self.strategy = strategy
        self.config = config
        self.spaces = spaces or ['all']
        self.results: List[HyperoptResult] = []
        self.best_result: Optional[HyperoptResult] = None
        
        # Define default search spaces
        self.default_spaces = {
            'buy': [
                Integer(5, 50, name='buy_rsi_period'),
                Real(0.2, 0.5, name='buy_rsi_threshold'),
                Integer(10, 100, name='buy_ma_period'),
                Real(0.01, 0.1, name='buy_volume_multiplier'),
            ],
            'sell': [
                Integer(5, 50, name='sell_rsi_period'),
                Real(0.5, 0.8, name='sell_rsi_threshold'),
                Integer(10, 100, name='sell_ma_period'),
            ],
            'roi': [
                Real(0.01, 0.1, name='roi_1_time'),
                Real(0.02, 0.2, name='roi_1_profit'),
                Real(0.1, 0.5, name='roi_2_time'),
                Real(0.05, 0.3, name='roi_2_profit'),
            ],
            'stoploss': [
                Real(0.02, 0.15, name='stoploss'),
            ],
            'trailing': [
                Categorical([True, False], name='trailing_stop'),
                Real(0.01, 0.05, name='trailing_stop_positive'),
                Real(0.02, 0.1, name='trailing_stop_positive_offset'),
            ]
        }
        
    def _get_search_space(self) -> List:
        """Build combined search space based on selected spaces"""
        space = []
        
        if 'all' in self.spaces:
            for category_params in self.default_spaces.values():
                space.extend(category_params)
        else:
            for space_name in self.spaces:
                if space_name in self.default_spaces:
                    space.extend(self.default_spaces[space_name])
                    
        return space
    
    def _objective_function(
        self,
        params: List,
        data: pd.DataFrame,
        initial_capital: float = 10000
    ) -> float:
        """
        Objective function for optimization
        
        Minimizes negative Sharpe ratio (maximizes Sharpe)
        """
        # Convert params to dict
        param_names = [p.name for p in self._get_search_space()]
        param_dict = dict(zip(param_names, params))
        
        try:
            # Update strategy with new parameters
            self._apply_params_to_strategy(param_dict)
            
            # Run backtest
            backtester = Backtester(
                strategy=self.strategy,
                initial_capital=initial_capital,
                commission_rate=self.config.get('commission', 0.001),
                slippage=self.config.get('slippage', 0.001)
            )
            
            result = backtester.run(data)
            
            # Calculate objective score
            # Negative Sharpe (we minimize)
            sharpe = result.get('sharpe_ratio', 0)
            max_dd = result.get('max_drawdown', 1)
            total_return = result.get('total_return', 0)
            trades = result.get('total_trades', 0)
            
            # Penalize low trade count
            if trades < 10:
                penalty = 10
            else:
                penalty = 0
            
            # Score: maximize Sharpe, minimize drawdown, ensure sufficient trades
            score = -(sharpe * (1 - max_dd) * np.log1p(total_return)) + penalty
            
            # Store result
            hyperopt_result = HyperoptResult(
                params=param_dict,
                score=-score,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                total_return=total_return,
                win_rate=result.get('win_rate', 0),
                profit_factor=result.get('profit_factor', 0),
                trades_count=trades,
                evaluation_count=len(self.results) + 1
            )
            
            self.results.append(hyperopt_result)
            
            logger.info(
                f"Iteration {hyperopt_result.evaluation_count}: "
                f"Sharpe={sharpe:.3f}, DD={max_dd:.3f}, Return={total_return:.3f}, "
                f"Trades={trades}, Score={-score:.3f}"
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1e6  # Return large penalty on error
    
    def _apply_params_to_strategy(self, params: Dict):
        """Apply optimized parameters to strategy"""
        # Update strategy attributes
        for key, value in params.items():
            if hasattr(self.strategy, key):
                setattr(self.strategy, key, value)
    
    def optimize(
        self,
        data: pd.DataFrame,
        n_calls: int = 50,
        n_random_starts: int = 10,
        optimizer: str = 'gp',
        random_state: int = 42,
        save_results: bool = True,
        results_file: str = 'hyperopt_results.json'
    ) -> HyperoptResult:
        """
        Run hyperopt optimization
        
        Args:
            data: Historical price data
            n_calls: Number of optimization iterations
            n_random_starts: Random initial points
            optimizer: 'gp' (Gaussian Process), 'rf', 'gbrt'
            random_state: Random seed
            save_results: Save results to file
            results_file: Output file path
            
        Returns:
            Best HyperoptResult
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize required. Install with: pip install scikit-optimize"
            )
        
        logger.info(f"Starting hyperopt with {n_calls} iterations...")
        logger.info(f"Search space: {[p.name for p in self._get_search_space()]}")
        
        search_space = self._get_search_space()
        
        @use_named_args(search_space)
        def objective(**kwargs):
            params_list = list(kwargs.values())
            return self._objective_function(params_list, data)
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            random_state=random_state,
            verbose=1
        )
        
        # Extract best parameters
        best_params = dict(zip([p.name for p in search_space], result.x))
        
        # Get metrics from last evaluation with these params
        best_result = max(self.results, key=lambda x: x.score)
        
        self.best_result = best_result
        
        logger.info("\n" + "="*60)
        logger.info("HYPEROPT COMPLETE")
        logger.info("="*60)
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best Sharpe Ratio: {best_result.sharpe_ratio:.3f}")
        logger.info(f"Best Max Drawdown: {best_result.max_drawdown:.3f}")
        logger.info(f"Best Total Return: {best_result.total_return:.3f}")
        logger.info(f"Total Trades: {best_result.trades_count}")
        logger.info(f"Win Rate: {best_result.win_rate:.3f}")
        logger.info("="*60)
        
        # Save results
        if save_results:
            self.save_results(results_file)
        
        return best_result
    
    def save_results(self, filepath: str):
        """Save optimization results to JSON"""
        results_data = {
            'best_result': self.best_result.to_dict() if self.best_result else None,
            'all_results': [r.to_dict() for r in self.results],
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict:
        """Load previous optimization results"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot optimization convergence"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.results:
                logger.warning("No results to plot")
                return
            
            scores = [r.score for r in self.results]
            iterations = range(1, len(scores) + 1)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Score convergence
            axes[0, 0].plot(iterations, scores, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Optimization Convergence')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Sharpe ratio progression
            sharpes = [r.sharpe_ratio for r in self.results]
            axes[0, 1].plot(iterations, sharpes, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].set_title('Sharpe Ratio Progression')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Drawdown distribution
            drawdowns = [r.max_drawdown for r in self.results]
            axes[1, 0].hist(drawdowns, bins=20, color='red', alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Max Drawdown')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Drawdown Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Returns vs Sharpe scatter
            returns = [r.total_return for r in self.results]
            axes[1, 1].scatter(sharpes, returns, c=drawdowns, cmap='RdYlGn', alpha=0.6)
            axes[1, 1].set_xlabel('Sharpe Ratio')
            axes[1, 1].set_ylabel('Total Return')
            axes[1, 1].set_title('Returns vs Risk-Adjusted Performance')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not installed for plotting")


# Example usage
if __name__ == "__main__":
    print("Hyperopt Engine Module Loaded Successfully")
    print("Features:")
    print("  - Scikit-optimize integration")
    print("  - Multiple pairlist generators")
    print("  - Advanced stoploss management")
    print("  - Parameter space optimization")
    print("\nUsage:")
    print("  from hyperopt import HyperoptEngine, PairListGenerator")
    print("  hyperopt = HyperoptEngine(strategy, config)")
    print("  best_params = hyperopt.optimize(data, n_calls=50)")
