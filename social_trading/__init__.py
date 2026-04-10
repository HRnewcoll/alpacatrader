"""
Social Trading Platform
Inspired by: eToro, ZuluTrade, Naga

Features:
- Trader profiles with performance metrics
- Social feed for trade sharing
- Follow/copy functionality
- Leaderboards and rankings
- Community sentiment analysis
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import json
from collections import defaultdict


class TraderTier(Enum):
    """Trader reputation tiers"""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    ELITE = "elite"
    LEGENDARY = "legendary"


@dataclass
class Trade:
    """Represents a single trade"""
    trader_id: str
    symbol: str
    direction: int  # 1=long, -1=short
    entry_price: float
    quantity: float
    timestamp: datetime
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    status: str = "open"  # open/closed
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            'trader_id': self.trader_id,
            'symbol': self.symbol,
            'direction': 'LONG' if self.direction > 0 else 'SHORT',
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'timestamp': self.timestamp.isoformat(),
            'exit_price': self.exit_price,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'status': self.status,
            'notes': self.notes
        }


@dataclass
class TraderProfile:
    """Trader profile with performance metrics"""
    trader_id: str
    username: str
    bio: str = ""
    join_date: datetime = field(default_factory=datetime.now)
    tier: TraderTier = TraderTier.NOVICE
    followers: Set[str] = field(default_factory=set)
    following: Set[str] = field(default_factory=set)
    total_pnl: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    risk_score: float = 5.0  # 1-10 scale
    verified: bool = False
    avatar_url: str = ""
    
    def to_dict(self) -> dict:
        return {
            'trader_id': self.trader_id,
            'username': self.username,
            'bio': self.bio,
            'join_date': self.join_date.isoformat(),
            'tier': self.tier.value,
            'followers_count': len(self.followers),
            'following_count': len(self.following),
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'total_return': self.total_return,
            'risk_score': self.risk_score,
            'verified': self.verified
        }


@dataclass
class SocialPost:
    """Social media post about trades or market insights"""
    post_id: str
    trader_id: str
    content: str
    timestamp: datetime
    likes: int = 0
    comments: List[dict] = field(default_factory=list)
    shares: int = 0
    related_symbols: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1 to 1
    
    def to_dict(self) -> dict:
        return {
            'post_id': self.post_id,
            'trader_id': self.trader_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'likes': self.likes,
            'comments_count': len(self.comments),
            'shares': self.shares,
            'related_symbols': self.related_symbols,
            'sentiment_score': self.sentiment_score
        }


class SocialTradingPlatform:
    """
    Social trading platform enabling traders to share, follow, and copy trades.
    Inspired by eToro, ZuluTrade, and modern social trading platforms.
    """
    
    def __init__(self):
        self.traders: Dict[str, TraderProfile] = {}
        self.trades: Dict[str, List[Trade]] = defaultdict(list)  # trader_id -> trades
        self.posts: List[SocialPost] = []
        self.copied_trades: Dict[str, List[dict]] = defaultdict(list)  # follower_id -> copied trades
        self.logger = logging.getLogger("SocialTrading")
        
    def register_trader(self, trader_id: str, username: str, bio: str = "") -> TraderProfile:
        """Register a new trader on the platform"""
        if trader_id in self.traders:
            raise ValueError(f"Trader {trader_id} already exists")
        
        profile = TraderProfile(
            trader_id=trader_id,
            username=username,
            bio=bio
        )
        self.traders[trader_id] = profile
        self.logger.info(f"Registered trader: {username} ({trader_id})")
        return profile
    
    def follow_trader(self, follower_id: str, trader_id: str):
        """Follow a trader"""
        if trader_id not in self.traders:
            raise ValueError(f"Trader {trader_id} not found")
        if follower_id not in self.traders:
            # Auto-register follower if not exists
            self.register_trader(follower_id, f"user_{follower_id}")
        
        self.traders[trader_id].followers.add(follower_id)
        self.traders[follower_id].following.add(trader_id)
        self.logger.info(f"{follower_id} is now following {trader_id}")
    
    def unfollow_trader(self, follower_id: str, trader_id: str):
        """Unfollow a trader"""
        if trader_id in self.traders and follower_id in self.traders:
            self.traders[trader_id].followers.discard(follower_id)
            self.traders[follower_id].following.discard(trader_id)
    
    def publish_trade(self, trade: Trade):
        """Publish a trade to the social feed"""
        self.trades[trade.trader_id].append(trade)
        
        # Update trader statistics
        self._update_trader_stats(trade.trader_id)
        
        # Create social post automatically
        self._auto_create_trade_post(trade)
        
        # Notify followers (in real implementation)
        self._notify_followers(trade)
    
    def _auto_create_trade_post(self, trade: Trade):
        """Automatically create a social post for a trade"""
        trader = self.traders.get(trade.trader_id)
        if not trader:
            return
        
        action = "LONG" if trade.direction > 0 else "SHORT"
        content = f"📈 {action} {trade.symbol} @ ${trade.entry_price:.2f}"
        
        if trader.verified:
            content += f" ✓ (Verified Trader)"
        
        post = SocialPost(
            post_id=f"post_{len(self.posts)}",
            trader_id=trade.trader_id,
            content=content,
            timestamp=trade.timestamp,
            related_symbols=[trade.symbol],
            sentiment_score=0.3 if trade.direction > 0 else -0.3
        )
        
        self.posts.append(post)
    
    def _update_trader_stats(self, trader_id: str):
        """Update trader performance statistics"""
        trader = self.traders.get(trader_id)
        if not trader:
            return
        
        trader_trades = self.trades[trader_id]
        closed_trades = [t for t in trader_trades if t.status == "closed"]
        
        if not closed_trades:
            return
        
        # Calculate metrics
        pnls = [t.pnl for t in closed_trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        trader.total_trades = len(closed_trades)
        trader.total_pnl = sum(pnls)
        trader.win_rate = len(wins) / len(pnls) if pnls else 0
        trader.avg_win = np.mean(wins) if wins else 0
        trader.avg_loss = np.mean(losses) if losses else 0
        trader.total_return = trader.total_pnl / 10000  # Assuming 10k initial
        
        # Calculate max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / np.maximum(running_max, 1)
        trader.max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate Sharpe ratio
        if len(pnls) > 1:
            trader.sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        
        # Update tier based on performance
        self._update_trader_tier(trader)
    
    def _update_trader_tier(self, trader: TraderProfile):
        """Update trader tier based on performance"""
        total_trades = trader.total_trades
        win_rate = trader.win_rate
        total_pnl = trader.total_pnl
        sharpe = trader.sharpe_ratio
        
        if total_trades >= 500 and win_rate >= 0.6 and total_pnl >= 50000 and sharpe >= 2.0:
            trader.tier = TraderTier.LEGENDARY
        elif total_trades >= 200 and win_rate >= 0.55 and total_pnl >= 20000 and sharpe >= 1.5:
            trader.tier = TraderTier.ELITE
        elif total_trades >= 100 and win_rate >= 0.5 and total_pnl >= 5000 and sharpe >= 1.0:
            trader.tier = TraderTier.EXPERT
        elif total_trades >= 20 and win_rate >= 0.45:
            trader.tier = TraderTier.INTERMEDIATE
        else:
            trader.tier = TraderTier.NOVICE
    
    def _notify_followers(self, trade: Trade):
        """Notify followers about a new trade (placeholder)"""
        trader = self.traders.get(trade.trader_id)
        if not trader:
            return
        
        # In real implementation, send push notifications, emails, etc.
        follower_count = len(trader.followers)
        if follower_count > 0:
            self.logger.debug(f"Notified {follower_count} followers about {trade.symbol} trade")
    
    def copy_trade(self, follower_id: str, original_trade: Trade, 
                   copy_ratio: float = 1.0) -> Trade:
        """Copy another trader's trade"""
        if follower_id not in self.traders:
            raise ValueError(f"Follower {follower_id} not registered")
        
        # Scale the trade size
        copied_quantity = original_trade.quantity * copy_ratio
        
        copied_trade = Trade(
            trader_id=follower_id,
            symbol=original_trade.symbol,
            direction=original_trade.direction,
            entry_price=original_trade.entry_price,
            quantity=copied_quantity,
            timestamp=datetime.now(),
            notes=f"Copied from {original_trade.trader_id}"
        )
        
        self.trades[follower_id].append(copied_trade)
        self.copied_trades[follower_id].append({
            'original_trade': original_trade.to_dict(),
            'copied_trade': copied_trade.to_dict(),
            'copy_ratio': copy_ratio,
            'timestamp': datetime.now()
        })
        
        self.logger.info(f"{follower_id} copied trade from {original_trade.trader_id}")
        return copied_trade
    
    def get_leaderboard(self, period: str = "all_time", 
                       metric: str = "total_pnl",
                       top_n: int = 10) -> List[Dict]:
        """Get leaderboard of top traders"""
        if not self.traders:
            return []
        
        # Sort traders by metric
        if metric == "total_pnl":
            sorted_traders = sorted(
                self.traders.values(),
                key=lambda t: t.total_pnl,
                reverse=True
            )
        elif metric == "win_rate":
            sorted_traders = sorted(
                self.traders.values(),
                key=lambda t: t.win_rate,
                reverse=True
            )
        elif metric == "sharpe_ratio":
            sorted_traders = sorted(
                self.traders.values(),
                key=lambda t: t.sharpe_ratio,
                reverse=True
            )
        elif metric == "followers":
            sorted_traders = sorted(
                self.traders.values(),
                key=lambda t: len(t.followers),
                reverse=True
            )
        else:
            sorted_traders = list(self.traders.values())
        
        # Return top N
        return [t.to_dict() for t in sorted_traders[:top_n]]
    
    def get_social_feed(self, limit: int = 20) -> List[Dict]:
        """Get recent social posts"""
        sorted_posts = sorted(self.posts, key=lambda p: p.timestamp, reverse=True)
        return [p.to_dict() for p in sorted_posts[:limit]]
    
    def get_trader_profile(self, trader_id: str) -> Optional[Dict]:
        """Get detailed trader profile"""
        trader = self.traders.get(trader_id)
        if not trader:
            return None
        
        profile = trader.to_dict()
        profile['recent_trades'] = [
            t.to_dict() for t in self.trades[trader_id][-10:]
        ]
        profile['recent_posts'] = [
            p.to_dict() for p in self.posts 
            if p.trader_id == trader_id
        ][-5:]
        
        return profile
    
    def get_community_sentiment(self, symbol: str) -> Dict:
        """Analyze community sentiment for a symbol"""
        relevant_posts = [
            p for p in self.posts 
            if symbol in p.related_symbols
        ]
        
        if not relevant_posts:
            return {'symbol': symbol, 'sentiment': 'neutral', 'score': 0.0}
        
        avg_sentiment = np.mean([p.sentiment_score for p in relevant_posts])
        
        # Count long vs short positions
        recent_trades = [
            t for trader_trades in self.trades.values()
            for t in trader_trades
            if t.symbol == symbol and t.status == "open"
        ]
        
        long_count = sum(1 for t in recent_trades if t.direction > 0)
        short_count = sum(1 for t in recent_trades if t.direction < 0)
        
        if avg_sentiment > 0.2:
            sentiment = "bullish"
        elif avg_sentiment < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        return {
            'symbol': symbol,
            'sentiment': sentiment,
            'score': avg_sentiment,
            'long_positions': long_count,
            'short_positions': short_count,
            'posts_count': len(relevant_posts),
            'bullish_percentage': long_count / (long_count + short_count) if (long_count + short_count) > 0 else 0.5
        }


# Demo function
async def demo_social_trading():
    """Demonstrate social trading platform"""
    print("=" * 80)
    print("SOCIAL TRADING PLATFORM DEMO")
    print("=" * 80)
    
    platform = SocialTradingPlatform()
    
    # Register traders
    print("\n📝 Registering traders...")
    traders_data = [
        ("trader_001", "AlphaWolf", "Professional day trader specializing in tech stocks"),
        ("trader_002", "CryptoKing", "Crypto enthusiast since 2017"),
        ("trader_003", "ValueInvestor", "Long-term value investing strategy"),
        ("trader_004", "SwingMaster", "Swing trading expert"),
        ("trader_005", "QuantGenius", "PhD in Mathematics, quantitative strategies")
    ]
    
    for trader_id, username, bio in traders_data:
        platform.register_trader(trader_id, username, bio)
        print(f"  ✓ Registered: {username}")
    
    # Simulate some trades
    print("\n💼 Publishing trades...")
    base_time = datetime.now() - timedelta(hours=5)
    
    trades_data = [
        ("trader_001", "AAPL", 1, 175.50, 100, 0.8),
        ("trader_001", "MSFT", 1, 380.20, 50, 0.9),
        ("trader_002", "BTC", 1, 45000.0, 0.5, 0.7),
        ("trader_003", "GOOGL", -1, 140.0, 75, 0.6),
        ("trader_004", "TSLA", 1, 250.0, 200, 0.75),
        ("trader_005", "SPY", 1, 480.0, 300, 0.85),
    ]
    
    for i, (trader_id, symbol, direction, price, qty, win_prob) in enumerate(trades_data):
        trade = Trade(
            trader_id=trader_id,
            symbol=symbol,
            direction=direction,
            entry_price=price,
            quantity=qty,
            timestamp=base_time + timedelta(minutes=i*30)
        )
        
        # Simulate some closed trades with PnL
        if i % 2 == 0:  # Close every other trade
            pnl_multiplier = 1.0 if np.random.random() < win_prob else -1.0
            trade.exit_price = price * (1 + 0.05 * pnl_multiplier)
            trade.exit_timestamp = trade.timestamp + timedelta(hours=2)
            trade.pnl = (trade.exit_price - price) * qty * direction
            trade.pnl_percent = (trade.exit_price - price) / price * 100
            trade.status = "closed"
        
        platform.publish_trade(trade)
        print(f"  {'✓' if trade.status == 'closed' else '○'} {symbol}: ${price:.2f} ({trade.status})")
    
    # Simulate following
    print("\n👥 Building social network...")
    platform.follow_trader("follower_001", "trader_001")
    platform.follow_trader("follower_001", "trader_005")
    platform.follow_trader("follower_002", "trader_002")
    platform.follow_trader("follower_002", "trader_003")
    platform.follow_trader("follower_003", "trader_001")
    print("  ✓ Following relationships established")
    
    # Copy trading
    print("\n📋 Copy trading...")
    original_trade = platform.trades["trader_001"][0]
    copied = platform.copy_trade("follower_001", original_trade, copy_ratio=0.5)
    print(f"  ✓ follower_001 copied {original_trade.symbol} trade (50% size)")
    
    # Get leaderboards
    print("\n🏆 LEADERBOARDS")
    print("-" * 60)
    
    print("\nTop by Total P&L:")
    leaderboard = platform.get_leaderboard(metric="total_pnl", top_n=3)
    for i, trader in enumerate(leaderboard, 1):
        print(f"  {i}. {trader['username']:15} | P&L: ${trader['total_pnl']:>10,.2f} | Win Rate: {trader['win_rate']:.1%}")
    
    print("\nTop by Win Rate:")
    leaderboard = platform.get_leaderboard(metric="win_rate", top_n=3)
    for i, trader in enumerate(leaderboard, 1):
        print(f"  {i}. {trader['username']:15} | Win Rate: {trader['win_rate']:.1%} | Trades: {trader['total_trades']}")
    
    print("\nTop by Followers:")
    leaderboard = platform.get_leaderboard(metric="followers", top_n=3)
    for i, trader in enumerate(leaderboard, 1):
        print(f"  {i}. {trader['username']:15} | Followers: {trader['followers_count']}")
    
    # Social feed
    print("\n📱 SOCIAL FEED")
    print("-" * 60)
    feed = platform.get_social_feed(limit=5)
    for post in feed:
        print(f"  @{post['trader_id']}: {post['content']}")
    
    # Community sentiment
    print("\n💭 COMMUNITY SENTIMENT")
    print("-" * 60)
    for symbol in ["AAPL", "MSFT", "TSLA"]:
        sentiment = platform.get_community_sentiment(symbol)
        emoji = "🐂" if sentiment['sentiment'] == 'bullish' else "🐻" if sentiment['sentiment'] == 'bearish' else "➖"
        print(f"  {emoji} {symbol}: {sentiment['sentiment'].upper()} (Score: {sentiment['score']:.2f})")
    
    # Trader profile
    print("\n👤 TRADER PROFILE EXAMPLE")
    print("-" * 60)
    profile = platform.get_trader_profile("trader_001")
    if profile:
        print(f"  Username: {profile['username']}")
        print(f"  Tier: {profile['tier'].upper()}")
        print(f"  Followers: {profile['followers_count']}")
        print(f"  Total P&L: ${profile['total_pnl']:,.2f}")
        print(f"  Win Rate: {profile['win_rate']:.1%}")
        print(f"  Sharpe Ratio: {profile['sharpe_ratio']:.2f}")
        print(f"  Verified: {'✓' if profile['verified'] else '✗'}")
    
    print("\n✅ Social Trading Platform Demo Complete!")
    return True


if __name__ == "__main__":
    asyncio.run(demo_social_trading())
