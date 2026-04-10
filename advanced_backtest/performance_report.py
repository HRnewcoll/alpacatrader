"""
Professional Performance Report Generator
Features:
- 50+ performance metrics (Sharpe, Sortino, Calmar, Omega, etc.)
- Risk metrics (VaR, CVaR, Ulcer Index, Beta, Alpha)
- Tear sheets similar to QuantConnect and QuantStats
- HTML report generation with interactive charts
- Benchmark comparison
Inspired by: quantstats, tear-sheet, Hudson & Thames
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceReportGenerator:
    """
    Generate comprehensive performance reports with institutional-grade metrics
    
    Metrics categories:
    - Returns: Total, Annualized, CAGR, Monthly/Weekly/Daily stats
    - Risk: Volatility, VaR, CVaR, Max Drawdown, Ulcer Index
    - Risk-adjusted: Sharpe, Sortino, Calmar, Omega, Information Ratio
    - Trading: Win rate, Profit Factor, Avg Win/Loss, Expectancy
    - Drawdown: Max DD, Avg DD, DD Duration, Recovery Factor
    
    Inspired by: quantstats, FactSet, Bloomberg PORT
    """
    
    def __init__(self, risk_free_rate: float = 0.02, benchmark_name: str = "SPY"):
        """
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
            benchmark_name: Name of benchmark for comparison
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_name = benchmark_name
        
    def calculate_all_metrics(self, returns: pd.Series, equity_curve: pd.Series = None,
                             trades: pd.DataFrame = None, 
                             benchmark_returns: pd.Series = None) -> Dict:
        """
        Calculate comprehensive set of performance metrics
        
        Args:
            returns: Series of period returns (daily recommended)
            equity_curve: Series of portfolio values over time
            trades: DataFrame of individual trades with pnl column
            benchmark_returns: Series of benchmark returns for comparison
            
        Returns: Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns, equity_curve))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns, equity_curve))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns, benchmark_returns))
        
        # Trading statistics (if trades provided)
        if trades is not None and len(trades) > 0:
            metrics.update(self._calculate_trading_stats(trades))
        
        # Drawdown analysis
        metrics.update(self._calculate_drawdown_metrics(equity_curve if equity_curve is not None else returns))
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_comparison(returns, benchmark_returns))
        
        # Add metadata
        metrics['report_generated_at'] = datetime.now().isoformat()
        metrics['n_observations'] = len(returns)
        metrics['start_date'] = str(returns.index[0]) if len(returns) > 0 else None
        metrics['end_date'] = str(returns.index[-1]) if len(returns) > 0 else None
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series, equity_curve: pd.Series = None) -> Dict:
        """Calculate return-related metrics"""
        metrics = {}
        
        # Total return
        if equity_curve is not None:
            total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        else:
            total_return = (1 + returns).prod() - 1
        metrics['total_return'] = total_return
        
        # Annualized return (CAGR)
        n_years = len(returns) / 252.0
        if n_years > 0:
            cagr = (1 + total_return) ** (1 / n_years) - 1
        else:
            cagr = 0
        metrics['cagr'] = cagr
        metrics['annualized_return'] = cagr
        
        # Period returns
        metrics['daily_mean_return'] = returns.mean()
        metrics['daily_median_return'] = returns.median()
        metrics['daily_std'] = returns.std()
        
        # Monthly aggregation
        if len(returns) > 0:
            monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            metrics['monthly_mean_return'] = monthly_returns.mean()
            metrics['monthly_std'] = monthly_returns.std()
            metrics['best_month'] = monthly_returns.max()
            metrics['worst_month'] = monthly_returns.min()
            
            # Positive months percentage
            metrics['pct_positive_months'] = (monthly_returns > 0).mean()
        
        # Skewness and Kurtosis
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        return metrics
    
    def _calculate_risk_metrics(self, returns: pd.Series, equity_curve: pd.Series = None) -> Dict:
        """Calculate risk metrics"""
        metrics = {}
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        metrics['daily_volatility'] = daily_vol
        metrics['annual_volatility'] = annual_vol
        
        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        metrics['var_95'] = var_95
        metrics['var_99'] = var_99
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        metrics['cvar_95'] = cvar_95
        metrics['cvar_99'] = cvar_99
        
        # Downside deviation (for Sortino)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        metrics['downside_deviation'] = downside_deviation
        metrics['downside_volatility_annual'] = downside_deviation * np.sqrt(252)
        
        # Ulcer Index (measure of drawdown severity)
        if equity_curve is not None:
            ulcer_index = self._calculate_ulcer_index(equity_curve)
        else:
            ulcer_index = self._calculate_ulcer_index_from_returns(returns)
        metrics['ulcer_index'] = ulcer_index
        
        # Tail ratio
        tail_ratio = abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else np.inf
        metrics['tail_ratio'] = tail_ratio
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series, 
                                         benchmark_returns: pd.Series = None) -> Dict:
        """Calculate risk-adjusted return metrics"""
        metrics = {}
        
        rf_daily = self.risk_free_rate / 252
        
        # Sharpe Ratio
        excess_returns = returns - rf_daily
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        metrics['sharpe_ratio'] = sharpe
        
        # Sortino Ratio (uses downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        metrics['sortino_ratio'] = sortino
        
        # Calmar Ratio (return / max_drawdown)
        if len(returns) > 20:  # Need sufficient data for drawdown
            cumulative = (1 + returns).cumprod()
            max_dd = self._calculate_max_drawdown(cumulative)
            calmar = abs(returns.mean() * 252 / max_dd) if max_dd != 0 else 0
            metrics['calmar_ratio'] = calmar
        
        # Omega Ratio (probability-weighted ratio of gains/losses)
        threshold = rf_daily
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf
        metrics['omega_ratio'] = omega
        
        # Information Ratio (vs benchmark)
        if benchmark_returns is not None:
            aligned_returns = returns.align(benchmark_returns, join='inner')[0]
            aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
            active_returns = aligned_returns - aligned_benchmark
            tracking_error = active_returns.std() * np.sqrt(252)
            info_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            metrics['information_ratio'] = info_ratio
            
            # Treynor Ratio (requires beta)
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_var = np.var(aligned_benchmark)
            beta = covariance / benchmark_var if benchmark_var > 0 else 1
            treynor = (returns.mean() * 252 - self.risk_free_rate) / beta if beta != 0 else 0
            metrics['treynor_ratio'] = treynor
            metrics['beta'] = beta
            
            # Jensen's Alpha
            expected_return = self.risk_free_rate + beta * (benchmark_returns.mean() * 252 - self.risk_free_rate)
            alpha = returns.mean() * 252 - expected_return
            metrics['alpha'] = alpha
        
        return metrics
    
    def _calculate_trading_stats(self, trades: pd.DataFrame) -> Dict:
        """Calculate trading-specific statistics"""
        metrics = {}
        
        pnls = trades['pnl'].values
        
        # Win/Loss statistics
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        
        metrics['total_trades'] = len(pnls)
        metrics['winning_trades'] = len(wins)
        metrics['losing_trades'] = len(losses)
        metrics['win_rate'] = len(wins) / len(pnls) if len(pnls) > 0 else 0
        
        # Average win/loss
        metrics['avg_win'] = wins.mean() if len(wins) > 0 else 0
        metrics['avg_loss'] = abs(losses.mean()) if len(losses) > 0 else 0
        metrics['win_loss_ratio'] = metrics['avg_win'] / metrics['avg_loss'] if metrics['avg_loss'] > 0 else np.inf
        
        # Profit Factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        metrics['profit_factor'] = profit_factor
        
        # Expectancy
        expectancy = (metrics['win_rate'] * metrics['avg_win']) - ((1 - metrics['win_rate']) * metrics['avg_loss'])
        metrics['expectancy_per_trade'] = expectancy
        
        # Largest win/loss
        metrics['largest_win'] = pnls.max()
        metrics['largest_loss'] = pnls.min()
        
        # Consecutive wins/losses
        metrics['max_consecutive_wins'] = self._max_consecutive(pnls > 0)
        metrics['max_consecutive_losses'] = self._max_consecutive(pnls <= 0)
        
        return metrics
    
    def _calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict:
        """Calculate drawdown-related metrics"""
        metrics = {}
        
        if isinstance(equity_curve, pd.Series) and equity_curve.dtype in ['float64', 'int64']:
            # It's already an equity curve
            pass
        else:
            # Convert returns to equity curve
            equity_curve = (1 + equity_curve).cumprod()
        
        # Maximum Drawdown
        max_dd = self._calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = max_dd
        
        # Average Drawdown
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        metrics['avg_drawdown'] = abs(drawdown_series.mean())
        metrics['drawdown_std'] = drawdown_series.std()
        
        # Drawdown duration
        dd_durations = self._calculate_drawdown_durations(equity_curve)
        if len(dd_durations) > 0:
            metrics['avg_drawdown_duration_days'] = np.mean(dd_durations)
            metrics['max_drawdown_duration_days'] = np.max(dd_durations)
            metrics['current_drawdown_duration_days'] = dd_durations[-1] if len(dd_durations) > 0 else 0
        
        # Recovery Factor
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        recovery_factor = abs(total_return / max_dd) if max_dd != 0 else np.inf
        metrics['recovery_factor'] = recovery_factor
        
        return metrics
    
    def _calculate_benchmark_comparison(self, returns: pd.Series, 
                                       benchmark_returns: pd.Series) -> Dict:
        """Calculate benchmark comparison metrics"""
        metrics = {}
        
        # Align series
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        
        # Total returns comparison
        strategy_total = (1 + aligned_returns).prod() - 1
        benchmark_total = (1 + aligned_benchmark).prod() - 1
        metrics['benchmark_total_return'] = benchmark_total
        metrics['excess_return_vs_benchmark'] = strategy_total - benchmark_total
        
        # Correlation
        correlation = aligned_returns.corr(aligned_benchmark)
        metrics['correlation_to_benchmark'] = correlation
        
        # R-squared
        metrics['r_squared'] = correlation ** 2 if correlation is not None else 0
        
        # Capture ratios
        up_capture = self._calculate_capture_ratio(aligned_returns, aligned_benchmark, 1)
        down_capture = self._calculate_capture_ratio(aligned_returns, aligned_benchmark, -1)
        metrics['up_capture_ratio'] = up_capture
        metrics['down_capture_ratio'] = down_capture
        metrics['capture_ratio'] = up_capture / down_capture if down_capture != 0 else np.inf
        
        return metrics
    
    def generate_tear_sheet(self, metrics: Dict) -> str:
        """Generate a formatted text tear sheet"""
        lines = []
        lines.append("=" * 70)
        lines.append("PERFORMANCE TEAR SHEET")
        lines.append("=" * 70)
        lines.append(f"Generated: {metrics.get('report_generated_at', 'N/A')}")
        lines.append(f"Period: {metrics.get('start_date', 'N/A')} to {metrics.get('end_date', 'N/A')}")
        lines.append(f"Observations: {metrics.get('n_observations', 0)}")
        lines.append("")
        
        # Returns section
        lines.append("-" * 70)
        lines.append("RETURNS")
        lines.append("-" * 70)
        lines.append(f"Total Return:          {metrics.get('total_return', 0):>12.2%}")
        lines.append(f"CAGR:                  {metrics.get('cagr', 0):>12.2%}")
        lines.append(f"Daily Mean Return:     {metrics.get('daily_mean_return', 0):>12.4%}")
        lines.append(f"Monthly Mean Return:   {metrics.get('monthly_mean_return', 0):>12.2%}")
        lines.append(f"Best Month:            {metrics.get('best_month', 0):>12.2%}")
        lines.append(f"Worst Month:           {metrics.get('worst_month', 0):>12.2%}")
        lines.append("")
        
        # Risk section
        lines.append("-" * 70)
        lines.append("RISK METRICS")
        lines.append("-" * 70)
        lines.append(f"Annual Volatility:     {metrics.get('annual_volatility', 0):>12.2%}")
        lines.append(f"VaR (95%):             {metrics.get('var_95', 0):>12.2%}")
        lines.append(f"CVaR (95%):            {metrics.get('cvar_95', 0):>12.2%}")
        lines.append(f"Ulcer Index:           {metrics.get('ulcer_index', 0):>12.4f}")
        lines.append(f"Skewness:              {metrics.get('skewness', 0):>12.3f}")
        lines.append(f"Kurtosis:              {metrics.get('kurtosis', 0):>12.3f}")
        lines.append("")
        
        # Risk-adjusted section
        lines.append("-" * 70)
        lines.append("RISK-ADJUSTED RETURNS")
        lines.append("-" * 70)
        lines.append(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>12.3f}")
        lines.append(f"Sortino Ratio:         {metrics.get('sortino_ratio', 0):>12.3f}")
        lines.append(f"Calmar Ratio:          {metrics.get('calmar_ratio', 0):>12.3f}")
        lines.append(f"Omega Ratio:           {metrics.get('omega_ratio', 0):>12.3f}")
        if 'information_ratio' in metrics:
            lines.append(f"Information Ratio:     {metrics.get('information_ratio', 0):>12.3f}")
            lines.append(f"Beta:                  {metrics.get('beta', 0):>12.3f}")
            lines.append(f"Alpha:                 {metrics.get('alpha', 0):>12.2%}")
        lines.append("")
        
        # Drawdown section
        lines.append("-" * 70)
        lines.append("DRAWDOWN ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"Maximum Drawdown:      {metrics.get('max_drawdown', 0):>12.2%}")
        lines.append(f"Average Drawdown:      {metrics.get('avg_drawdown', 0):>12.2%}")
        lines.append(f"Recovery Factor:       {metrics.get('recovery_factor', 0):>12.3f}")
        if 'avg_drawdown_duration_days' in metrics:
            lines.append(f"Avg DD Duration:       {metrics.get('avg_drawdown_duration_days', 0):>12.1f} days")
            lines.append(f"Max DD Duration:       {metrics.get('max_drawdown_duration_days', 0):>12.1f} days")
        lines.append("")
        
        # Trading stats section
        if 'total_trades' in metrics:
            lines.append("-" * 70)
            lines.append("TRADING STATISTICS")
            lines.append("-" * 70)
            lines.append(f"Total Trades:          {metrics.get('total_trades', 0):>12d}")
            lines.append(f"Win Rate:              {metrics.get('win_rate', 0):>12.1%}")
            lines.append(f"Profit Factor:         {metrics.get('profit_factor', 0):>12.2f}")
            lines.append(f"Expectancy/Trade:      ${metrics.get('expectancy_per_trade', 0):>11.2f}")
            lines.append(f"Avg Win:               ${metrics.get('avg_win', 0):>11.2f}")
            lines.append(f"Avg Loss:              ${metrics.get('avg_loss', 0):>11.2f}")
            lines.append(f"Win/Loss Ratio:        {metrics.get('win_loss_ratio', 0):>12.2f}")
            lines.append(f"Largest Win:           ${metrics.get('largest_win', 0):>11.2f}")
            lines.append(f"Largest Loss:          ${metrics.get('largest_loss', 0):>11.2f}")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        peak = equity_curve.expanding(min_periods=1).max()
        return (equity_curve - peak) / peak
    
    def _calculate_drawdown_durations(self, equity_curve: pd.Series) -> List[int]:
        """Calculate durations of drawdown periods"""
        drawdown = self._calculate_drawdown_series(equity_curve)
        in_drawdown = drawdown < 0
        
        durations = []
        current_duration = 0
        
        for in_dd in in_drawdown:
            if in_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        # Add current ongoing drawdown
        if current_duration > 0:
            durations.append(current_duration)
        
        return durations if len(durations) > 0 else [0]
    
    def _calculate_ulcer_index(self, equity_curve: pd.Series) -> float:
        """Calculate Ulcer Index"""
        drawdown = self._calculate_drawdown_series(equity_curve)
        squared_dd = drawdown ** 2
        return np.sqrt(squared_dd.mean())
    
    def _calculate_ulcer_index_from_returns(self, returns: pd.Series) -> float:
        """Calculate Ulcer Index from returns"""
        equity_curve = (1 + returns).cumprod()
        return self._calculate_ulcer_index(equity_curve)
    
    def _max_consecutive(self, boolean_array: np.ndarray) -> int:
        """Calculate maximum consecutive True values"""
        max_count = 0
        current_count = 0
        
        for val in boolean_array:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _calculate_capture_ratio(self, returns: pd.Series, benchmark: pd.Series, 
                                 direction: int) -> float:
        """Calculate up/down capture ratio"""
        if direction == 1:
            # Up markets: benchmark > 0
            mask = benchmark > 0
        else:
            # Down markets: benchmark < 0
            mask = benchmark < 0
            
        if mask.sum() == 0:
            return 1.0
            
        strategy_return = (1 + returns[mask]).prod() - 1
        benchmark_return = (1 + benchmark[mask]).prod() - 1
        
        if benchmark_return == 0:
            return 1.0
            
        return strategy_return / benchmark_return
