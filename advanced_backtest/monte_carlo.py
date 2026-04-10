"""
Monte Carlo Simulation Engine for Strategy Robustness Testing
Features:
- Path generation via bootstrapping
- Parameter perturbation analysis
- Randomized trade sequencing
- Probability of ruin calculation
- Confidence interval estimation
Inspired by: QuantConnect, Hudson & Thames, Early Warning project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for testing strategy robustness
    
    Methodologies implemented:
    1. Trade Bootstrapping: Random resampling of historical trades
    2. Return Bootstrapping: Random resampling of daily returns
    3. Parameter Perturbation: Add noise to strategy parameters
    4. Price Path Simulation: Geometric Brownian Motion with jumps
    
    Inspired by: Michael Harris' "Early Warning" methodology
    """
    
    def __init__(self, n_simulations: int = 1000, confidence_levels: List[float] = [0.90, 0.95, 0.99],
                 seed: Optional[int] = None, n_jobs: int = -1):
        """
        Args:
            n_simulations: Number of Monte Carlo paths to simulate
            confidence_levels: Confidence levels for VaR/CVaR calculations
            seed: Random seed for reproducibility
            n_jobs: Number of parallel workers (-1 for all CPUs)
        """
        self.n_simulations = n_simulations
        self.confidence_levels = sorted(confidence_levels)
        self.seed = seed
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
        if seed is not None:
            np.random.seed(seed)
    
    def bootstrap_trades(self, trades_df: pd.DataFrame, n_paths: Optional[int] = None) -> pd.DataFrame:
        """
        Bootstrap trade-level results by random sampling with replacement
        
        Args:
            trades_df: DataFrame with columns [pnl, timestamp]
            n_paths: Number of bootstrap paths (default: self.n_simulations)
            
        Returns: DataFrame with bootstrap equity curves
        """
        n_paths = n_paths or self.n_simulations
        
        if len(trades_df) == 0:
            raise ValueError("No trades provided for bootstrapping")
        
        pnls = trades_df['pnl'].values
        initial_capital = 100000.0
        
        # Generate bootstrap samples
        bootstrap_results = []
        
        for i in range(n_paths):
            # Sample trades with replacement (same length as original)
            sampled_pnls = np.random.choice(pnls, size=len(pnls), replace=True)
            
            # Calculate cumulative equity curve
            cumulative_pnl = np.cumsum(sampled_pnls)
            equity_curve = initial_capital + cumulative_pnl
            
            bootstrap_results.append({
                'path_id': i,
                'equity_curve': equity_curve,
                'final_equity': equity_curve[-1],
                'total_return': (equity_curve[-1] - initial_capital) / initial_capital,
                'max_drawdown': self._calculate_max_drawdown(equity_curve),
                'n_wins': sum(sampled_pnls > 0),
                'n_losses': sum(sampled_pnls <= 0)
            })
        
        return pd.DataFrame(bootstrap_results)
    
    def bootstrap_returns(self, returns_series: pd.Series, n_paths: Optional[int] = None) -> pd.DataFrame:
        """
        Bootstrap daily returns to generate alternative equity paths
        
        Args:
            returns_series: Series of daily returns
            n_paths: Number of bootstrap paths
            
        Returns: DataFrame with bootstrap statistics
        """
        n_paths = n_paths or self.n_simulations
        initial_capital = 100000.0
        
        returns = returns_series.dropna().values
        n_days = len(returns)
        
        bootstrap_results = []
        
        for i in range(n_paths):
            # Sample returns with replacement
            sampled_returns = np.random.choice(returns, size=n_days, replace=True)
            
            # Calculate equity curve from returns
            cumulative_returns = np.cumprod(1 + sampled_returns)
            equity_curve = initial_capital * cumulative_returns
            
            bootstrap_results.append({
                'path_id': i,
                'equity_curve': equity_curve,
                'final_equity': equity_curve[-1],
                'total_return': (equity_curve[-1] - initial_capital) / initial_capital,
                'max_drawdown': self._calculate_max_drawdown(equity_curve),
                'annualized_return': np.mean(sampled_returns) * 252,
                'annualized_vol': np.std(sampled_returns) * np.sqrt(252)
            })
        
        return pd.DataFrame(bootstrap_results)
    
    def simulate_price_paths(self, initial_price: float, mu: float, sigma: float,
                            n_days: int, n_paths: Optional[int] = None,
                            jump_prob: float = 0.0, jump_mean: float = 0.0,
                            jump_std: float = 0.0) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion with optional jumps
        
        Args:
            initial_price: Starting price
            mu: Expected annual return (drift)
            sigma: Annual volatility
            n_days: Number of days to simulate
            n_paths: Number of paths
            jump_prob: Probability of jump on any given day (Merton jump-diffusion)
            jump_mean: Mean jump size (as % of price)
            jump_std: Standard deviation of jump size
            
        Returns: Array of shape (n_paths, n_days) with simulated prices
        """
        n_paths = n_paths or self.n_simulations
        dt = 1.0 / 252.0  # Daily time step
        
        # Generate GBM paths
        Z = np.random.normal(0, 1, (n_paths, n_days))
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        log_returns = drift + diffusion
        
        # Add jumps if specified
        if jump_prob > 0:
            jump_mask = np.random.random((n_paths, n_days)) < jump_prob
            jump_sizes = np.random.normal(jump_mean, jump_std, (n_paths, n_days))
            log_returns[jump_mask] += np.log(1 + jump_sizes[jump_mask])
        
        # Convert to prices
        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        price_paths = initial_price * np.exp(cumulative_log_returns)
        
        # Prepend initial price
        price_paths = np.column_stack([np.full(n_paths, initial_price), price_paths])
        
        return price_paths
    
    def perturb_parameters(self, base_params: Dict[str, float], 
                          perturbation_pct: float = 0.1,
                          n_samples: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Generate parameter sets by adding random noise to base parameters
        
        Args:
            base_params: Dictionary of base parameter values
            perturbation_pct: Maximum perturbation as % of parameter value
            n_samples: Number of parameter sets to generate
            
        Returns: List of perturbed parameter dictionaries
        """
        n_samples = n_samples or self.n_simulations
        
        perturbed_params = []
        
        for _ in range(n_samples):
            new_params = {}
            for param_name, base_value in base_params.items():
                # Add uniform noise within ±perturbation_pct
                noise = np.random.uniform(-perturbation_pct, perturbation_pct)
                new_value = base_value * (1 + noise)
                
                # Ensure positive values for parameters that must be positive
                if base_value > 0 and new_value <= 0:
                    new_value = base_value * 0.01  # Minimum 1% of original
                    
                new_params[param_name] = new_value
                
            perturbed_params.append(new_params)
        
        return perturbed_params
    
    def calculate_probability_of_ruin(self, bootstrap_results: pd.DataFrame,
                                     ruin_threshold: float = 0.5) -> float:
        """
        Calculate probability of ruin (equity falling below threshold)
        
        Args:
            bootstrap_results: DataFrame from bootstrap methods
            ruin_threshold: Threshold as % of initial capital (0.5 = 50% loss)
            
        Returns: Probability of ruin
        """
        min_equities = []
        
        for _, row in bootstrap_results.iterrows():
            equity_curve = row['equity_curve']
            min_equity = np.min(equity_curve)
            min_equities.append(min_equity)
        
        # Initial capital assumed to be 100000
        initial_capital = 100000.0
        ruin_level = initial_capital * ruin_threshold
        
        prob_ruin = sum(1 for me in min_equities if me < ruin_level) / len(min_equities)
        
        return prob_ruin
    
    def calculate_var_cvar(self, bootstrap_results: pd.DataFrame,
                          confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR
        
        Args:
            bootstrap_results: DataFrame with final_equity column
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns: (VaR, CVaR) as percentages
        """
        returns = bootstrap_results['total_return'].values
        
        # VaR: percentile of loss distribution
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        # CVaR (Expected Shortfall): average of losses beyond VaR
        tail_losses = returns[returns <= var]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        return var, cvar
    
    def run_full_analysis(self, trades_df: pd.DataFrame = None,
                         returns_series: pd.Series = None,
                         base_params: Dict = None,
                         strategy_func=None, data=None) -> Dict:
        """
        Run comprehensive Monte Carlo analysis
        
        Args:
            trades_df: Historical trades for bootstrapping
            returns_series: Historical returns for bootstrapping
            base_params: Base parameters for perturbation analysis
            strategy_func: Strategy function for parameter perturbation testing
            data: Data for strategy evaluation
            
        Returns: Comprehensive analysis results
        """
        logger.info(f"Running Monte Carlo analysis with {self.n_simulations} simulations")
        
        results = {
            'n_simulations': self.n_simulations,
            'confidence_levels': self.confidence_levels
        }
        
        # Trade bootstrapping
        if trades_df is not None and len(trades_df) > 0:
            logger.info("Bootstrapping trades...")
            trade_bootstrap = self.bootstrap_trades(trades_df)
            
            results['trade_bootstrap'] = trade_bootstrap
            results['prob_ruin_50pct'] = self.calculate_probability_of_ruin(trade_bootstrap, 0.5)
            results['prob_ruin_30pct'] = self.calculate_probability_of_ruin(trade_bootstrap, 0.3)
            
            # VaR/CVaR for each confidence level
            var_cvar_results = {}
            for cl in self.confidence_levels:
                var, cvar = self.calculate_var_cvar(trade_bootstrap, cl)
                var_cvar_results[cl] = {'var': var, 'cvar': cvar}
            results['var_cvar_from_trades'] = var_cvar_results
            
            # Summary statistics
            results['trade_bootstrap_summary'] = {
                'mean_return': trade_bootstrap['total_return'].mean(),
                'std_return': trade_bootstrap['total_return'].std(),
                'median_return': trade_bootstrap['total_return'].median(),
                'mean_max_dd': trade_bootstrap['max_drawdown'].mean(),
                'worst_case_return': trade_bootstrap['total_return'].min(),
                'best_case_return': trade_bootstrap['total_return'].max()
            }
        
        # Return bootstrapping
        if returns_series is not None and len(returns_series) > 0:
            logger.info("Bootstrapping returns...")
            return_bootstrap = self.bootstrap_returns(returns_series)
            
            results['return_bootstrap'] = return_bootstrap
            
            var_cvar_results = {}
            for cl in self.confidence_levels:
                var, cvar = self.calculate_var_cvar(return_bootstrap, cl)
                var_cvar_results[cl] = {'var': var, 'cvar': cvar}
            results['var_cvar_from_returns'] = var_cvar_results
            
            results['return_bootstrap_summary'] = {
                'mean_return': return_bootstrap['total_return'].mean(),
                'std_return': return_bootstrap['total_return'].std(),
                'median_return': return_bootstrap['total_return'].median(),
                'mean_annualized_return': return_bootstrap['annualized_return'].mean(),
                'mean_annualized_vol': return_bootstrap['annualized_vol'].mean()
            }
        
        # Parameter perturbation
        if base_params is not None and strategy_func is not None and data is not None:
            logger.info("Running parameter perturbation analysis...")
            perturbed_params = self.perturb_parameters(base_params, perturbation_pct=0.1)
            
            perturbation_results = []
            for params in perturbed_params:
                try:
                    metrics = strategy_func(data, params)
                    perturbation_results.append({
                        'params': params,
                        'sharpe': metrics.get('sharpe_ratio', 0),
                        'return': metrics.get('total_return', 0)
                    })
                except:
                    continue
            
            if len(perturbation_results) > 0:
                results['parameter_perturbation'] = pd.DataFrame(perturbation_results)
                results['parameter_sensitivity'] = {
                    'sharpe_mean': results['parameter_perturbation']['sharpe'].mean(),
                    'sharpe_std': results['parameter_perturbation']['sharpe'].std(),
                    'return_mean': results['parameter_perturbation']['return'].mean(),
                    'return_std': results['parameter_perturbation']['return'].std(),
                    'robustness_score': 1.0 / (1.0 + results['parameter_perturbation']['sharpe'].std())
                }
        
        logger.info("Monte Carlo analysis complete")
        return results
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(np.min(drawdown))
