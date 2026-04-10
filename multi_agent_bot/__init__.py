"""
Multi-Agent Trading Bot System
Inspired by: TradingAgents, Freqtrade, Hummingbot

A collaborative multi-agent system where specialized AI agents work together
to make trading decisions through debate, consensus, and ensemble methods.
"""

import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime
import json


class AgentRole(Enum):
    """Specialized trading agent roles"""
    BULL = "bull"  # Optimistic agent, looks for long opportunities
    BEAR = "bear"  # Pessimistic agent, looks for short opportunities  
    TECHNICAL = "technical"  # Pure technical analysis
    FUNDAMENTAL = "fundamental"  # Fundamental analysis focus
    SENTIMENT = "sentiment"  # News/social sentiment analysis
    RISK = "risk"  # Risk management specialist
    QUANT = "quant"  # Quantitative/statistical arbitrage
    MACRO = "macro"  # Macro economic trends
    HFT = "hft"  # High-frequency trading signals
    CONTRARIAN = "contrarian"  # Goes against crowd sentiment


class ConsensusMethod(Enum):
    """Methods for reaching agent consensus"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    DEBATE = "debate"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADVERSARIAL = "adversarial"


@dataclass
class AgentSignal:
    """Signal from an individual agent"""
    agent_id: str
    role: AgentRole
    direction: int  # 1=long, -1=short, 0=neutral
    confidence: float  # 0.0 to 1.0
    reasoning: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon: str = "medium"  # short/medium/long
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'direction': self.direction,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'time_horizon': self.time_horizon,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ConsensusDecision:
    """Final consensus decision from all agents"""
    symbol: str
    direction: int
    aggregate_confidence: float
    participating_agents: List[str]
    individual_signals: List[AgentSignal]
    consensus_method: ConsensusMethod
    reasoning_summary: str
    dissenting_opinions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'aggregate_confidence': self.aggregate_confidence,
            'participating_agents': self.participating_agents,
            'individual_signals': [s.to_dict() for s in self.individual_signals],
            'consensus_method': self.consensus_method.value,
            'reasoning_summary': self.reasoning_summary,
            'dissenting_opinions': self.dissenting_opinions,
            'timestamp': self.timestamp.isoformat()
        }


class BaseTradingAgent(ABC):
    """Abstract base class for all trading agents"""
    
    def __init__(self, agent_id: str, role: AgentRole, initial_capital: float = 100000):
        self.agent_id = agent_id
        self.role = role
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.performance_history = []
        self.confidence_score = 0.5  # Dynamic confidence based on performance
        self.logger = logging.getLogger(f"Agent.{agent_id}")
        
    @abstractmethod
    async def analyze(self, market_data: pd.DataFrame, 
                     news_sentiment: Optional[dict] = None,
                     order_book: Optional[dict] = None) -> AgentSignal:
        """Generate trading signal based on analysis"""
        pass
    
    def update_performance(self, pnl: float):
        """Update agent performance metrics"""
        self.current_capital += pnl
        self.performance_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'cumulative_pnl': self.current_capital - self.initial_capital,
            'roi': (self.current_capital - self.initial_capital) / self.initial_capital
        })
        
        # Adjust confidence based on recent performance (last 10 trades)
        if len(self.performance_history) >= 10:
            recent_returns = [p['pnl'] for p in self.performance_history[-10:]]
            win_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns)
            self.confidence_score = min(0.95, max(0.1, win_rate))
    
    def get_statistics(self) -> dict:
        """Get agent performance statistics"""
        if not self.performance_history:
            return {}
        
        returns = [p['pnl'] for p in self.performance_history]
        cumulative = [p['cumulative_pnl'] for p in self.performance_history]
        
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'total_trades': len(returns),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
            'total_pnl': sum(returns),
            'avg_pnl': np.mean(returns) if returns else 0,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0,
            'max_drawdown': min(cumulative) if cumulative else 0,
            'current_capital': self.current_capital,
            'confidence_score': self.confidence_score
        }


class BullAgent(BaseTradingAgent):
    """Optimistic agent that looks for long opportunities"""
    
    def __init__(self):
        super().__init__("bull_001", AgentRole.BULL)
        
    async def analyze(self, market_data: pd.DataFrame, 
                     news_sentiment: Optional[dict] = None,
                     order_book: Optional[dict] = None) -> AgentSignal:
        """Look for bullish patterns and positive momentum"""
        if market_data.empty or len(market_data) < 20:
            return AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=0,
                confidence=0.0,
                reasoning="Insufficient data for analysis"
            )
        
        # Calculate momentum indicators
        close_prices = market_data['close'].values
        volumes = market_data.get('volume', pd.Series([1]*len(close_prices))).values
        
        # Simple momentum strategy
        short_ma = np.mean(close_prices[-5:])
        long_ma = np.mean(close_prices[-20:])
        momentum = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
        volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
        
        # Sentiment boost
        sentiment_boost = 0.0
        if news_sentiment and news_sentiment.get('score', 0) > 0.3:
            sentiment_boost = news_sentiment['score'] * 0.2
        
        # Generate signal
        if short_ma > long_ma and momentum > 0.02:
            confidence = min(0.95, 0.6 + abs(momentum) * 2 + sentiment_boost + (0.1 if volume_trend > 1.2 else 0))
            return AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=1,
                confidence=confidence,
                reasoning=f"Bullish crossover detected. Momentum: {momentum:.2%}, Volume trend: {volume_trend:.2f}x",
                target_price=close_prices[-1] * 1.05,
                stop_loss=close_prices[-1] * 0.97,
                time_horizon="medium"
            )
        elif momentum > 0.05:
            return AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=1,
                confidence=0.7,
                reasoning=f"Strong momentum: {momentum:.2%}",
                target_price=close_prices[-1] * 1.03,
                time_horizon="short"
            )
        
        return AgentSignal(
            agent_id=self.agent_id,
            role=self.role,
            direction=0,
            confidence=0.3,
            reasoning="No strong bullish signals detected"
        )


class BearAgent(BaseTradingAgent):
    """Pessimistic agent that looks for short opportunities"""
    
    def __init__(self):
        super().__init__("bear_001", AgentRole.BEAR)
        
    async def analyze(self, market_data: pd.DataFrame, 
                     news_sentiment: Optional[dict] = None,
                     order_book: Optional[dict] = None) -> AgentSignal:
        """Look for bearish patterns and negative momentum"""
        if market_data.empty or len(market_data) < 20:
            return AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=0,
                confidence=0.0,
                reasoning="Insufficient data for analysis"
            )
        
        close_prices = market_data['close'].values
        volumes = market_data.get('volume', pd.Series([1]*len(close_prices))).values
        
        # Simple momentum strategy (inverse of bull)
        short_ma = np.mean(close_prices[-5:])
        long_ma = np.mean(close_prices[-20:])
        momentum = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
        volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
        
        # Sentiment penalty
        sentiment_penalty = 0.0
        if news_sentiment and news_sentiment.get('score', 0) < -0.3:
            sentiment_penalty = abs(news_sentiment['score']) * 0.2
        
        # Generate signal
        if short_ma < long_ma and momentum < -0.02:
            confidence = min(0.95, 0.6 + abs(momentum) * 2 + sentiment_penalty + (0.1 if volume_trend > 1.2 else 0))
            return AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=-1,
                confidence=confidence,
                reasoning=f"Bearish crossover detected. Momentum: {momentum:.2%}, Volume trend: {volume_trend:.2f}x",
                target_price=close_prices[-1] * 0.95,
                stop_loss=close_prices[-1] * 1.03,
                time_horizon="medium"
            )
        elif momentum < -0.05:
            return AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=-1,
                confidence=0.7,
                reasoning=f"Strong negative momentum: {momentum:.2%}",
                target_price=close_prices[-1] * 0.97,
                time_horizon="short"
            )
        
        return AgentSignal(
            agent_id=self.agent_id,
            role=self.role,
            direction=0,
            confidence=0.3,
            reasoning="No strong bearish signals detected"
        )


class TechnicalAgent(BaseTradingAgent):
    """Pure technical analysis specialist"""
    
    def __init__(self):
        super().__init__("tech_001", AgentRole.TECHNICAL)
        
    async def analyze(self, market_data: pd.DataFrame, 
                     news_sentiment: Optional[dict] = None,
                     order_book: Optional[dict] = None) -> AgentSignal:
        """Analyze using technical indicators only"""
        if market_data.empty or len(market_data) < 50:
            return AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=0,
                confidence=0.0,
                reasoning="Insufficient data for technical analysis"
            )
        
        close = market_data['close'].values
        high = market_data['high'].values if 'high' in market_data.columns else close
        low = market_data['low'].values if 'low' in market_data.columns else close
        
        # RSI calculation
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
        avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9).mean().values
        
        # Bollinger Bands
        sma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
        
        signals = []
        confidences = []
        reasonings = []
        
        # RSI signals
        if rsi < 30:
            signals.append(1)
            confidences.append(0.7)
            reasonings.append(f"RSI oversold at {rsi:.1f}")
        elif rsi > 70:
            signals.append(-1)
            confidences.append(0.7)
            reasonings.append(f"RSI overbought at {rsi:.1f}")
        
        # MACD signals
        if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
            signals.append(1)
            confidences.append(0.6)
            reasonings.append("MACD bullish crossover")
        elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
            signals.append(-1)
            confidences.append(0.6)
            reasonings.append("MACD bearish crossover")
        
        # Bollinger Band signals
        if close[-1] < lower_band:
            signals.append(1)
            confidences.append(0.65)
            reasonings.append("Price below lower Bollinger Band")
        elif close[-1] > upper_band:
            signals.append(-1)
            confidences.append(0.65)
            reasonings.append("Price above upper Bollinger Band")
        
        if not signals:
            return AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=0,
                confidence=0.3,
                reasoning="No clear technical signals"
            )
        
        # Aggregate signals
        weighted_signal = sum(s * c for s, c in zip(signals, confidences))
        total_confidence = sum(confidences)
        final_direction = 1 if weighted_signal > 0 else -1 if weighted_signal < 0 else 0
        final_confidence = min(0.95, abs(weighted_signal) / total_confidence)
        
        return AgentSignal(
            agent_id=self.agent_id,
            role=self.role,
            direction=final_direction,
            confidence=final_confidence,
            reasoning="; ".join(reasonings),
            target_price=close[-1] * (1.03 if final_direction > 0 else 0.97),
            stop_loss=close[-1] * (0.97 if final_direction > 0 else 1.03)
        )


class RiskAgent(BaseTradingAgent):
    """Risk management specialist - can veto trades"""
    
    def __init__(self):
        super().__init__("risk_001", AgentRole.RISK)
        self.max_position_size = 0.1  # 10% of portfolio per trade
        self.max_daily_loss = 0.05  # 5% daily loss limit
        self.correlation_threshold = 0.8
        
    async def analyze(self, market_data: pd.DataFrame, 
                     news_sentiment: Optional[dict] = None,
                     order_book: Optional[dict] = None) -> AgentSignal:
        """Assess risk levels and provide risk-adjusted signals"""
        if market_data.empty or len(market_data) < 20:
            return AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=0,
                confidence=1.0,
                reasoning="Insufficient data for risk assessment"
            )
        
        close = market_data['close'].values
        returns = np.diff(close) / close[:-1]
        
        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate VaR (Value at Risk)
        var_95 = np.percentile(returns, 5)
        
        # Calculate max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        
        # Risk assessment
        risk_level = "LOW"
        confidence = 1.0
        reasoning_parts = []
        
        if volatility > 0.5:
            risk_level = "HIGH"
            confidence = 0.3
            reasoning_parts.append(f"High volatility: {volatility:.1%}")
        elif volatility > 0.3:
            risk_level = "MEDIUM"
            confidence = 0.6
            reasoning_parts.append(f"Elevated volatility: {volatility:.1%}")
        else:
            reasoning_parts.append(f"Normal volatility: {volatility:.1%}")
        
        if max_dd < -0.15:
            risk_level = "HIGH"
            confidence = min(confidence, 0.4)
            reasoning_parts.append(f"Large drawdown: {max_dd:.1%}")
        
        if var_95 < -0.05:
            reasoning_parts.append(f"High VaR: {var_95:.1%}")
            confidence = min(confidence, 0.5)
        
        # Risk agent doesn't take directional bets, but adjusts confidence
        return AgentSignal(
            agent_id=self.agent_id,
            role=self.role,
            direction=0,  # Neutral - just assesses risk
            confidence=confidence,
            reasoning=f"Risk Level: {risk_level}. " + "; ".join(reasoning_parts),
            time_horizon="continuous"
        )


class MultiAgentTradingBot:
    """
    Orchestrates multiple trading agents to reach consensus decisions.
    Inspired by TradingAgents research and ensemble methods.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.agents: Dict[str, BaseTradingAgent] = {}
        self.consensus_history: List[ConsensusDecision] = []
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = logging.getLogger("MultiAgentBot")
        
        # Initialize default agents
        self._initialize_default_agents()
        
    def _initialize_default_agents(self):
        """Create the default set of specialized agents"""
        self.add_agent(BullAgent())
        self.add_agent(BearAgent())
        self.add_agent(TechnicalAgent())
        self.add_agent(RiskAgent())
        
    def add_agent(self, agent: BaseTradingAgent):
        """Add a new agent to the collective"""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Added agent: {agent.agent_id} ({agent.role.value})")
        
    def remove_agent(self, agent_id: str):
        """Remove an agent from the collective"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Removed agent: {agent_id}")
            
    async def analyze_market(self, symbol: str, market_data: pd.DataFrame,
                            news_sentiment: Optional[dict] = None,
                            order_book: Optional[dict] = None,
                            consensus_method: ConsensusMethod = ConsensusMethod.CONFIDENCE_WEIGHTED
                            ) -> ConsensusDecision:
        """
        Collect signals from all agents and reach consensus.
        """
        if not self.agents:
            raise ValueError("No agents available for analysis")
        
        # Gather signals from all agents
        tasks = [
            agent.analyze(market_data, news_sentiment, order_book)
            for agent in self.agents.values()
        ]
        signals = await asyncio.gather(*tasks)
        
        # Filter out neutral signals with low confidence
        active_signals = [s for s in signals if s.direction != 0 or s.confidence > 0.5]
        
        if not active_signals:
            return ConsensusDecision(
                symbol=symbol,
                direction=0,
                aggregate_confidence=0.0,
                participating_agents=list(self.agents.keys()),
                individual_signals=signals,
                consensus_method=consensus_method,
                reasoning_summary="No actionable signals from any agent",
                dissenting_opinions=[]
            )
        
        # Apply consensus method
        if consensus_method == ConsensusMethod.MAJORITY_VOTE:
            direction, confidence, reasoning = self._majority_vote(active_signals)
        elif consensus_method == ConsensusMethod.WEIGHTED_AVERAGE:
            direction, confidence, reasoning = self._weighted_average(active_signals)
        elif consensus_method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            direction, confidence, reasoning = self._confidence_weighted(active_signals)
        elif consensus_method == ConsensusMethod.ADVERSARIAL:
            direction, confidence, reasoning = self._adversarial_debate(active_signals)
        else:
            direction, confidence, reasoning = self._confidence_weighted(active_signals)
        
        # Identify dissenting opinions
        dissenting = [
            f"{s.agent_id} ({s.role.value}): {s.reasoning}"
            for s in active_signals
            if s.direction != direction and s.direction != 0
        ]
        
        consensus = ConsensusDecision(
            symbol=symbol,
            direction=direction,
            aggregate_confidence=confidence,
            participating_agents=[s.agent_id for s in active_signals],
            individual_signals=active_signals,
            consensus_method=consensus_method,
            reasoning_summary=reasoning,
            dissenting_opinions=dissenting
        )
        
        self.consensus_history.append(consensus)
        return consensus
    
    def _majority_vote(self, signals: List[AgentSignal]) -> Tuple[int, float, str]:
        """Simple majority voting"""
        long_votes = sum(1 for s in signals if s.direction == 1)
        short_votes = sum(1 for s in signals if s.direction == -1)
        
        if long_votes > short_votes:
            direction = 1
            confidence = long_votes / len(signals)
            reasoning = f"Majority bullish: {long_votes}/{len(signals)} agents"
        elif short_votes > long_votes:
            direction = -1
            confidence = short_votes / len(signals)
            reasoning = f"Majority bearish: {short_votes}/{len(signals)} agents"
        else:
            direction = 0
            confidence = 0.5
            reasoning = "Split decision - no clear majority"
        
        return direction, confidence, reasoning
    
    def _weighted_average(self, signals: List[AgentSignal]) -> Tuple[int, float, str]:
        """Weight by agent performance history"""
        weighted_sum = 0
        total_weight = 0
        
        for signal in signals:
            weight = self.agents[signal.agent_id].confidence_score
            weighted_sum += signal.direction * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0, 0.5, "No weights available"
        
        normalized = weighted_sum / total_weight
        direction = 1 if normalized > 0.1 else -1 if normalized < -0.1 else 0
        confidence = min(0.95, abs(normalized))
        
        reasoning = f"Performance-weighted score: {normalized:.2f}"
        return direction, confidence, reasoning
    
    def _confidence_weighted(self, signals: List[AgentSignal]) -> Tuple[int, float, str]:
        """Weight by signal confidence levels"""
        weighted_sum = 0
        total_confidence = 0
        
        for signal in signals:
            weighted_sum += signal.direction * signal.confidence
            total_confidence += signal.confidence
        
        if total_confidence == 0:
            return 0, 0.5, "No confidence scores available"
        
        normalized = weighted_sum / total_confidence
        direction = 1 if normalized > 0.15 else -1 if normalized < -0.15 else 0
        confidence = min(0.95, abs(normalized) * (1 + total_confidence / len(signals) / 2))
        
        reasoning = f"Confidence-weighted score: {normalized:.2f} from {len(signals)} signals"
        return direction, confidence, reasoning
    
    def _adversarial_debate(self, signals: List[AgentSignal]) -> Tuple[int, float, str]:
        """Simulate debate between bull and bear agents"""
        bulls = [s for s in signals if s.direction == 1]
        bears = [s for s in signals if s.direction == -1]
        
        if not bulls or not bears:
            # No debate possible
            if bulls:
                avg_conf = np.mean([s.confidence for s in bulls])
                return 1, avg_conf, "Unanimous bullish - no opposition"
            elif bears:
                avg_conf = np.mean([s.confidence for s in bears])
                return -1, avg_conf, "Unanimous bearish - no opposition"
            else:
                return 0, 0.5, "No directional signals"
        
        # Calculate team strengths
        bull_strength = sum(s.confidence * self.agents[s.agent_id].confidence_score for s in bulls)
        bear_strength = sum(s.confidence * self.agents[s.agent_id].confidence_score for s in bears)
        
        total_strength = bull_strength + bear_strength
        if total_strength == 0:
            return 0, 0.5, "Equal strength - stalemate"
        
        # Winner takes majority, but confidence is reduced by opposition
        if bull_strength > bear_strength:
            direction = 1
            margin = (bull_strength - bear_strength) / total_strength
            confidence = 0.5 + margin * 0.4  # Max 0.9
            reasoning = f"Bulls won debate: {bull_strength:.2f} vs {bear_strength:.2f}"
        else:
            direction = -1
            margin = (bear_strength - bull_strength) / total_strength
            confidence = 0.5 + margin * 0.4
            reasoning = f"Bears won debate: {bear_strength:.2f} vs {bull_strength:.2f}"
        
        return direction, confidence, reasoning
    
    def get_collective_performance(self) -> dict:
        """Get aggregated performance metrics across all agents"""
        if not self.agents:
            return {}
        
        stats = [agent.get_statistics() for agent in self.agents.values()]
        valid_stats = [s for s in stats if s and 'agent_id' in s]
        
        if not valid_stats:
            return {
                'total_agents': len(self.agents),
                'avg_win_rate': 0,
                'total_pnl': 0,
                'avg_sharpe': 0,
                'best_performer': None,
                'worst_performer': None,
                'agent_details': stats
            }
        
        return {
            'total_agents': len(self.agents),
            'avg_win_rate': np.mean([s['win_rate'] for s in valid_stats]),
            'total_pnl': sum(s.get('total_pnl', 0) for s in valid_stats),
            'avg_sharpe': np.mean([s['sharpe_ratio'] for s in valid_stats]),
            'best_performer': max(valid_stats, key=lambda x: x.get('total_pnl', 0))['agent_id'],
            'worst_performer': min(valid_stats, key=lambda x: x.get('total_pnl', 0))['agent_id'],
            'agent_details': stats
        }


# Example usage and testing
async def demo_multi_agent_bot():
    """Demonstrate the multi-agent trading bot"""
    print("=" * 80)
    print("MULTI-AGENT TRADING BOT DEMO")
    print("=" * 80)
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    volumes = np.random.randint(1000000, 5000000, 100)
    
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(100),
        'high': prices + np.abs(np.random.randn(100) * 2),
        'low': prices - np.abs(np.random.randn(100) * 2),
        'close': prices,
        'volume': volumes
    }).set_index('date')
    
    # Initialize bot
    bot = MultiAgentTradingBot(initial_capital=100000)
    
    print(f"\nInitialized {len(bot.agents)} agents:")
    for agent_id, agent in bot.agents.items():
        print(f"  - {agent_id} ({agent.role.value})")
    
    # Run analysis
    print("\nRunning market analysis...")
    consensus = await bot.analyze_market(
        symbol="AAPL",
        market_data=market_data,
        news_sentiment={'score': 0.4, 'articles': 5},
        consensus_method=ConsensusMethod.CONFIDENCE_WEIGHTED
    )
    
    print(f"\n{'='*60}")
    print(f"CONSENSUS DECISION for {consensus.symbol}")
    print(f"{'='*60}")
    print(f"Direction: {'LONG' if consensus.direction > 0 else 'SHORT' if consensus.direction < 0 else 'NEUTRAL'}")
    print(f"Confidence: {consensus.aggregate_confidence:.1%}")
    print(f"Method: {consensus.consensus_method.value}")
    print(f"\nReasoning: {consensus.reasoning_summary}")
    
    if consensus.dissenting_opinions:
        print(f"\nDissenting opinions:")
        for opinion in consensus.dissenting_opinions:
            print(f"  • {opinion}")
    
    print(f"\nIndividual signals:")
    for signal in consensus.individual_signals:
        dir_str = 'LONG' if signal.direction > 0 else 'SHORT' if signal.direction < 0 else 'NEUTRAL'
        print(f"  {signal.agent_id:15} | {dir_str:6} | Conf: {signal.confidence:.1%} | {signal.reasoning[:50]}")
    
    # Show agent performance
    print(f"\n{'='*60}")
    print("AGENT PERFORMANCE STATISTICS")
    print(f"{'='*60}")
    perf = bot.get_collective_performance()
    print(f"Total Agents: {perf['total_agents']}")
    print(f"Average Win Rate: {perf['avg_win_rate']:.1%}")
    print(f"Best Performer: {perf['best_performer']}")
    print(f"Worst Performer: {perf['worst_performer']}")
    
    print("\n✅ Multi-Agent Bot Demo Complete!")
    return True


if __name__ == "__main__":
    asyncio.run(demo_multi_agent_bot())
