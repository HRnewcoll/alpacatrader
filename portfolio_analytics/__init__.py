"""
Portfolio Analytics Module
Advanced performance metrics, risk analysis, and reporting
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class PortfolioAnalytics:
    """
    Comprehensive portfolio performance and risk analytics
    """
    
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate periodic returns"""
        return prices.pct_change().dropna()
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns"""
        return (1 + returns).cumprod() - 1
    
    def cagr(self, returns: pd.Series, periods_per_year=252) -> float:
        """Calculate Compound Annual Growth Rate"""
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / periods_per_year
        if n_years <= 0:
            return 0.0
        return (1 + total_return) ** (1 / n_years) - 1
    
    def sharpe_ratio(self, returns: pd.Series, periods_per_year=252) -> float:
        """Calculate Sharpe Ratio"""
        if returns.std() == 0:
            return 0.0
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    
    def sortino_ratio(self, returns: pd.Series, periods_per_year=252) -> float:
        """Calculate Sortino Ratio (downside deviation)"""
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_std = np.sqrt((downside_returns ** 2).mean())
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    def calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar Ratio (CAGR / Max Drawdown)"""
        cagr_value = self.cagr(returns)
        max_dd = self.max_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        return cagr_value / abs(max_dd)
    
    def max_drawdown(self, returns: pd.Series) -> float:
        """Calculate Maximum Drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def var(self, returns: pd.Series, confidence_level=0.95, method='historical') -> float:
        """
        Calculate Value at Risk
        Methods: historical, parametric, cornish_fisher
        """
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return mean + z_score * std
        
        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion for non-normal distributions
            mean = returns.mean()
            std = returns.std()
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            z = stats.norm.ppf(1 - confidence_level)
            z_cf = (z + (z**2 - 1) * skew / 6 + 
                   (z**3 - 3*z) * (kurt - 3) / 24 - 
                   (2*z**3 - 5*z) * skew**2 / 36)
            
            return mean + z_cf * std
        
        return 0.0
    
    def cvar(self, returns: pd.Series, confidence_level=0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var_value = self.var(returns, confidence_level)
        return returns[returns <= var_value].mean()
    
    def beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Beta relative to benchmark"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 1.0
        
        covariance = returns.cov(benchmark_returns)
        variance = benchmark_returns.var()
        
        if variance == 0:
            return 1.0
        
        return covariance / variance
    
    def alpha(self, returns: pd.Series, benchmark_returns: pd.Series,
              periods_per_year=252) -> float:
        """Calculate Jensen's Alpha"""
        beta_value = self.beta(returns, benchmark_returns)
        market_return = benchmark_returns.mean() * periods_per_year
        portfolio_return = returns.mean() * periods_per_year
        
        return portfolio_return - (self.risk_free_rate + beta_value * (market_return - self.risk_free_rate))
    
    def information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series,
                         periods_per_year=252) -> float:
        """Calculate Information Ratio"""
        active_returns = returns - benchmark_returns
        
        if active_returns.std() == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * active_returns.mean() / active_returns.std()
    
    def tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series,
                      periods_per_year=252) -> float:
        """Calculate Tracking Error"""
        active_returns = returns - benchmark_returns
        return np.sqrt(periods_per_year) * active_returns.std()
    
    def win_rate(self, returns: pd.Series) -> float:
        """Calculate Win Rate"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)
    
    def profit_factor(self, returns: pd.Series) -> float:
        """Calculate Profit Factor (Gross Profit / Gross Loss)"""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def avg_win(self, returns: pd.Series) -> float:
        """Calculate Average Winning Trade"""
        wins = returns[returns > 0]
        return wins.mean() if len(wins) > 0 else 0.0
    
    def avg_loss(self, returns: pd.Series) -> float:
        """Calculate Average Losing Trade"""
        losses = returns[returns < 0]
        return losses.mean() if len(losses) > 0 else 0.0
    
    def max_consecutive_wins(self, returns: pd.Series) -> int:
        """Calculate Maximum Consecutive Wins"""
        is_win = returns > 0
        max_consecutive = 0
        current_consecutive = 0
        
        for win in is_win:
            if win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate Maximum Consecutive Losses"""
        is_loss = returns < 0
        max_consecutive = 0
        current_consecutive = 0
        
        for loss in is_loss:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer Index (measure of downside volatility)"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown_percent = (cumulative - running_max) / running_max * 100
        
        return np.sqrt((drawdown_percent ** 2).mean())
    
    def serenity_ratio(self, returns: pd.Series) -> float:
        """Calculate Serenity Ratio (Risk-adjusted return with drawdown penalty)"""
        cagr_value = self.cagr(returns)
        ulcer = self.ulcer_index(returns)
        
        if ulcer == 0:
            return 0.0
        
        return cagr_value / ulcer
    
    def kelly_criterion(self, returns: pd.Series) -> float:
        """Calculate Kelly Criterion optimal position size"""
        win_rate = self.win_rate(returns)
        avg_win = self.avg_win(returns)
        avg_loss = abs(self.avg_loss(returns))
        
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        return max(0, kelly)
    
    def correlation_matrix(self, returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets"""
        df = pd.DataFrame(returns_dict)
        return df.corr()
    
    def rolling_metrics(self, prices: pd.Series, window=252) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        returns = self.calculate_returns(prices)
        
        rolling_data = pd.DataFrame(index=prices.index)
        rolling_data['rolling_sharpe'] = returns.rolling(window).apply(
            lambda x: self.sharpe_ratio(x), raw=False
        )
        rolling_data['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        rolling_data['rolling_max_dd'] = returns.rolling(window).apply(
            lambda x: self.max_drawdown(x), raw=False
        )
        rolling_data['rolling_cagr'] = returns.rolling(window).apply(
            lambda x: self.cagr(x), raw=False
        )
        
        return rolling_data
    
    def generate_report(self, returns: pd.Series, benchmark_returns: pd.Series = None,
                       initial_capital=100000) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_return": f"{((1 + returns).prod() - 1) * 100:.2f}%",
                "cagr": f"{self.cagr(returns) * 100:.2f}%",
                "sharpe_ratio": f"{self.sharpe_ratio(returns):.3f}",
                "sortino_ratio": f"{self.sortino_ratio(returns):.3f}",
                "calmar_ratio": f"{self.calmar_ratio(returns):.3f}",
                "max_drawdown": f"{self.max_drawdown(returns) * 100:.2f}%",
                "volatility_annual": f"{returns.std() * np.sqrt(252) * 100:.2f}%"
            },
            "risk_metrics": {
                "var_95": f"{self.var(returns, 0.95) * 100:.2f}%",
                "cvar_95": f"{self.cvar(returns, 0.95) * 100:.2f}%",
                "ulcer_index": f"{self.ulcer_index(returns):.3f}",
                "kelly_fraction": f"{self.kelly_criterion(returns) * 100:.2f}%"
            },
            "trade_statistics": {
                "total_trades": len(returns),
                "win_rate": f"{self.win_rate(returns) * 100:.2f}%",
                "profit_factor": f"{self.profit_factor(returns):.2f}",
                "avg_win": f"{self.avg_win(returns) * 100:.2f}%",
                "avg_loss": f"{self.avg_loss(returns) * 100:.2f}%",
                "max_consecutive_wins": self.max_consecutive_wins(returns),
                "max_consecutive_losses": self.max_consecutive_losses(returns)
            }
        }
        
        if benchmark_returns is not None:
            report["relative_performance"] = {
                "beta": f"{self.beta(returns, benchmark_returns):.3f}",
                "alpha": f"{self.alpha(returns, benchmark_returns) * 100:.2f}%",
                "information_ratio": f"{self.information_ratio(returns, benchmark_returns):.3f}",
                "tracking_error": f"{self.tracking_error(returns, benchmark_returns) * 100:.2f}%"
            }
        
        # Final capital
        final_capital = initial_capital * (1 + returns).prod()
        report["capital"] = {
            "initial": initial_capital,
            "final": final_capital,
            "profit": final_capital - initial_capital
        }
        
        return report


def create_tearsheet(returns: pd.Series, title="Performance Tearsheet"):
    """
    Create a visual tearsheet (requires matplotlib/plotly)
    Returns data structure for plotting
    """
    analytics = PortfolioAnalytics()
    
    tearsheet = {
        "title": title,
        "cumulative_returns": analytics.calculate_cumulative_returns(returns),
        "daily_returns": returns,
        "rolling_sharpe": analytics.rolling_metrics(returns.to_frame()['close'] if hasattr(returns, 'to_frame') else returns)['rolling_sharpe'],
        "drawdown": (1 + returns).cumprod().cummax() - (1 + returns).cumprod(),
        "monthly_returns": returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
        "yearly_returns": returns.resample('Y').apply(lambda x: (1 + x).prod() - 1),
        "report": analytics.generate_report(returns)
    }
    
    return tearsheet
