"""
Walk-Forward Optimization Engine
Features:
- Rolling window optimization
- Out-of-sample testing
- Parameter stability analysis
- Avoids overfitting through cross-validation
Inspired by: QuantConnect, Hudson & Thames mlfinlab
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from itertools import product
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """
    Walk-forward optimization to prevent overfitting and test parameter robustness
    
    Methodology:
    1. Split data into multiple training/testing periods (rolling windows)
    2. Optimize parameters on training period
    3. Test optimized parameters on out-of-sample period
    4. Aggregate results across all periods
    
    Inspired by: Pardo's Walk-Forward Analysis methodology
    """
    
    def __init__(self, train_window_days: int = 90, test_window_days: int = 30, 
                 step_days: int = 30, n_jobs: int = -1):
        """
        Args:
            train_window_days: Length of in-sample training period
            test_window_days: Length of out-of-sample testing period
            step_days: Step size between consecutive windows
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
    def generate_windows(self, dates: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generate walk-forward windows
        
        Returns: List of tuples (train_start, train_end, test_start, test_end)
        """
        windows = []
        min_date = dates.min()
        max_date = dates.max()
        
        current_train_start = min_date
        while True:
            train_end = current_train_start + pd.Timedelta(days=self.train_window_days)
            test_end = train_end + pd.Timedelta(days=self.test_window_days)
            
            if test_end > max_date:
                break
                
            test_start = train_end
            windows.append((current_train_start, train_end, test_start, test_end))
            
            current_train_start += pd.Timedelta(days=self.step_days)
            
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def optimize_window(self, window: Tuple, data: pd.DataFrame, strategy_func: Callable,
                       param_grid: Dict[str, List], metric: str = 'sharpe_ratio') -> Dict:
        """
        Optimize parameters for a single window
        
        Args:
            window: (train_start, train_end, test_start, test_end)
            data: Full dataset
            strategy_func: Function that takes (data, params) and returns metrics dict
            param_grid: Dictionary of parameter names to lists of values
            metric: Metric to optimize (default: sharpe_ratio)
            
        Returns: Dict with best params and test results
        """
        train_start, train_end, test_start, test_end = window
        
        # Split data
        train_data = data[(data.index >= train_start) & (data.index < train_end)]
        test_data = data[(data.index >= test_start) & (data.index < test_end)]
        
        if len(train_data) == 0 or len(test_data) == 0:
            return None
            
        # Grid search over parameter space
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        best_metric = -np.inf
        best_params = None
        all_results = []
        
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            try:
                # Train period metrics
                train_metrics = strategy_func(train_data, params)
                train_score = train_metrics.get(metric, 0)
                
                # Test period metrics
                test_metrics = strategy_func(test_data, params)
                test_score = test_metrics.get(metric, 0)
                
                result = {
                    'params': params,
                    'train_score': train_score,
                    'test_score': test_score,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics
                }
                all_results.append(result)
                
                if test_score > best_metric:
                    best_metric = test_score
                    best_params = params
                    
            except Exception as e:
                logger.debug(f"Error with params {params}: {e}")
                continue
        
        if best_params is None:
            return None
            
        # Calculate parameter stability metrics
        if len(all_results) > 1:
            test_scores = [r['test_score'] for r in all_results]
            score_std = np.std(test_scores)
            score_mean = np.mean(test_scores)
            stability_ratio = score_std / abs(score_mean) if score_mean != 0 else np.inf
        else:
            stability_ratio = 0
        
        return {
            'window': window,
            'best_params': best_params,
            'best_test_score': best_metric,
            'stability_ratio': stability_ratio,
            'all_results': all_results,
            'n_combinations_tested': len(all_results)
        }
    
    def run(self, data: pd.DataFrame, strategy_func: Callable, 
            param_grid: Dict[str, List], metric: str = 'sharpe_ratio') -> Dict:
        """
        Run full walk-forward optimization
        
        Args:
            data: Dataset indexed by datetime
            strategy_func: Strategy function
            param_grid: Parameter grid to search
            metric: Optimization metric
            
        Returns: Comprehensive results dictionary
        """
        logger.info("Starting walk-forward optimization")
        logger.info(f"Train window: {self.train_window_days} days, "
                   f"Test window: {self.test_window_days} days, "
                   f"Step: {self.step_days} days")
        
        windows = self.generate_windows(data.index)
        if len(windows) == 0:
            raise ValueError("No valid windows generated. Check data length vs window sizes.")
        
        # Run optimization for each window
        results = []
        
        if self.n_jobs > 1 and len(windows) > 1:
            # Parallel execution
            logger.info(f"Running {len(windows)} optimizations in parallel with {self.n_jobs} workers")
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self.optimize_window, window, data, strategy_func, 
                                          param_grid, metric) for window in windows]
                for future in futures:
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            # Sequential execution
            for window in windows:
                result = self.optimize_window(window, data, strategy_func, param_grid, metric)
                if result:
                    results.append(result)
        
        if len(results) == 0:
            raise ValueError("Walk-forward optimization produced no valid results")
        
        # Aggregate results
        aggregated = self._aggregate_results(results, metric)
        
        logger.info(f"Walk-forward optimization complete")
        logger.info(f"Average OOS {metric}: {aggregated['avg_test_score']:.3f}")
        logger.info(f"Parameter stability ratio: {aggregated['avg_stability_ratio']:.3f}")
        
        return aggregated
    
    def _aggregate_results(self, results: List[Dict], metric: str) -> Dict:
        """Aggregate results across all windows"""
        
        # Extract test scores
        test_scores = [r['best_test_score'] for r in results]
        stability_ratios = [r['stability_ratio'] for r in results]
        
        # Collect all best parameters
        all_best_params = [r['best_params'] for r in results]
        
        # Find most common best parameters (mode)
        from collections import Counter
        param_strings = [str(sorted(p.items())) for p in all_best_params]
        param_counts = Counter(param_strings)
        most_common_param_str = param_counts.most_common(1)[0][0]
        most_common_params = dict(eval(most_common_param_str))
        
        # Calculate statistics
        aggregated = {
            'num_windows': len(results),
            'avg_test_score': np.mean(test_scores),
            'std_test_score': np.std(test_scores),
            'min_test_score': np.min(test_scores),
            'max_test_score': np.max(test_scores),
            'avg_stability_ratio': np.mean(stability_ratios),
            'most_common_params': most_common_params,
            'parameter_frequency': dict(param_counts),
            'window_results': results,
            'is_robust': np.mean(stability_ratios) < 0.5  # Stability ratio < 0.5 considered robust
        }
        
        # Add confidence intervals
        if len(test_scores) > 1:
            from scipy import stats
            ci = stats.t.interval(0.95, len(test_scores)-1, 
                                 loc=np.mean(test_scores), 
                                 scale=stats.sem(test_scores))
            aggregated['test_score_95_ci'] = ci
            
        return aggregated
    
    def analyze_parameter_sensitivity(self, results: Dict, param_name: str) -> pd.DataFrame:
        """
        Analyze how sensitive results are to a specific parameter
        
        Returns: DataFrame with parameter value vs average test score
        """
        sensitivity_data = []
        
        for window_result in results['window_results']:
            for combo_result in window_result['all_results']:
                param_value = combo_result['params'].get(param_name)
                test_score = combo_result['test_score']
                sensitivity_data.append({
                    param_name: param_value,
                    'test_score': test_score,
                    'window': window_result['window']
                })
        
        df = pd.DataFrame(sensitivity_data)
        
        # Group by parameter value and calculate statistics
        grouped = df.groupby(param_name)['test_score'].agg(['mean', 'std', 'count', 'min', 'max'])
        grouped.columns = ['avg_score', 'std_score', 'n_trials', 'min_score', 'max_score']
        
        return grouped.sort_values('avg_score', ascending=False)
