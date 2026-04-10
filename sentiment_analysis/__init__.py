"""
Sentiment Analysis Engine for Trading
NLP-powered news analysis, social media monitoring, and sentiment scoring
Inspired by Bloomberg Terminal sentiment analytics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import re


@dataclass
class NewsArticle:
    """Represents a news article or social media post"""
    title: str
    content: str
    source: str
    timestamp: datetime
    symbols: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    
    def __post_init__(self):
        if not self.symbols:
            # Try to extract stock symbols from title
            self.symbols = self._extract_symbols()
    
    def _extract_symbols(self) -> List[str]:
        """Extract stock ticker symbols from text"""
        # Simple pattern: uppercase letters, 1-5 chars
        pattern = r'\b[A-Z]{1,5}\b'
        matches = re.findall(pattern, self.title)
        
        # Filter common words
        common_words = {'THE', 'AND', 'FOR', 'NOT', 'BUT', 'YOU', 'ALL', 'CAN', 'HAS'}
        return [m for m in matches if m not in common_words]


class SentimentAnalyzer:
    """
    Rule-based sentiment analyzer (no external dependencies)
    Uses lexicon-based approach with financial context
    """
    
    def __init__(self):
        # Positive words with financial context
        self.positive_words = {
            'beat': 0.8, 'outperform': 0.9, 'surge': 0.7, 'soar': 0.8, 'gain': 0.6,
            'profit': 0.7, 'growth': 0.6, 'bullish': 0.8, 'upgrade': 0.7, 'buy': 0.6,
            'record': 0.7, 'strong': 0.5, 'positive': 0.6, 'optimistic': 0.7,
            'breakthrough': 0.8, 'success': 0.7, 'innovative': 0.5, 'leading': 0.5,
            'exceed': 0.7, 'rally': 0.7, 'jump': 0.6, 'rise': 0.5, 'up': 0.4,
            'approve': 0.6, 'deal': 0.5, 'partnership': 0.6, 'launch': 0.5,
        }
        
        # Negative words with financial context
        self.negative_words = {
            'miss': -0.7, 'underperform': -0.8, 'plunge': -0.8, 'crash': -0.9, 'loss': -0.7,
            'decline': -0.6, 'bearish': -0.8, 'downgrade': -0.7, 'sell': -0.6,
            'weak': -0.5, 'negative': -0.6, 'pessimistic': -0.7, 'fail': -0.7,
            'lawsuit': -0.7, 'investigation': -0.6, 'warning': -0.6, 'risk': -0.4,
            'drop': -0.6, 'fall': -0.5, 'cut': -0.5, 'layoff': -0.8, 'bankruptcy': -0.9,
            'fraud': -0.9, 'scandal': -0.8, 'concern': -0.5, 'challenge': -0.4,
        }
        
        # Intensifiers
        self.intensifiers = {
            'very': 1.3, 'extremely': 1.5, 'highly': 1.4, 'significantly': 1.3,
            'substantially': 1.4, 'dramatically': 1.5, 'sharply': 1.3, 'strongly': 1.3,
        }
        
        # Negators
        self.negators = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere'}
        
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        Returns dict with score, label, and breakdown
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        score = 0.0
        positive_count = 0
        negative_count = 0
        intensifier_multiplier = 1.0
        
        for i, word in enumerate(words):
            # Check for intensifier before word
            if i > 0 and words[i-1] in self.intensifiers:
                intensifier_multiplier = self.intensifiers[words[i-1]]
            else:
                intensifier_multiplier = 1.0
            
            # Check for negation (within 3 words)
            negated = False
            for j in range(max(0, i-3), i):
                if words[j] in self.negators:
                    negated = True
                    break
            
            # Score positive words
            if word in self.positive_words:
                word_score = self.positive_words[word] * intensifier_multiplier
                if negated:
                    word_score *= -0.5  # Partial negation
                score += word_score
                positive_count += 1
            
            # Score negative words
            elif word in self.negative_words:
                word_score = self.negative_words[word] * intensifier_multiplier
                if negated:
                    word_score *= -0.5  # Partial negation
                score += word_score
                negative_count += 1
        
        # Normalize score to [-1, 1]
        total_words = positive_count + negative_count
        if total_words > 0:
            normalized_score = np.tanh(score / total_words)
        else:
            normalized_score = 0.0
        
        # Determine label
        if normalized_score > 0.1:
            label = "bullish"
        elif normalized_score < -0.1:
            label = "bearish"
        else:
            label = "neutral"
        
        return {
            'score': normalized_score,
            'compound': normalized_score,
            'label': label,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_words': len(words),
        }
    
    def aggregate_sentiment(self, sentiments: List[Dict]) -> Dict:
        """Aggregate multiple sentiment scores"""
        if not sentiments:
            return {
                'overall_score': 0.0,
                'overall_sentiment': 'neutral',
                'bullish_pct': 0.0,
                'bearish_pct': 0.0,
                'neutral_pct': 0.0,
            }
        
        scores = [s['score'] for s in sentiments]
        avg_score = np.mean(scores)
        
        bullish = sum(1 for s in sentiments if s['label'] == 'bullish')
        bearish = sum(1 for s in sentiments if s['label'] == 'bearish')
        neutral = sum(1 for s in sentiments if s['label'] == 'neutral')
        total = len(sentiments)
        
        if avg_score > 0.1:
            overall = 'bullish'
        elif avg_score < -0.1:
            overall = 'bearish'
        else:
            overall = 'neutral'
        
        return {
            'overall_score': avg_score,
            'overall_sentiment': overall,
            'bullish_pct': (bullish / total) * 100,
            'bearish_pct': (bearish / total) * 100,
            'neutral_pct': (neutral / total) * 100,
            'count': total,
        }


class NewsAggregator:
    """
    Aggregates news from multiple sources
    Simulates RSS feeds and API integrations
    """
    
    def __init__(self):
        self.articles: deque = deque(maxlen=1000)
        self.symbol_news: Dict[str, List[NewsArticle]] = {}
        
    def add_article(self, article: NewsArticle):
        """Add news article"""
        self.articles.append(article)
        
        # Index by symbol
        for symbol in article.symbols:
            if symbol not in self.symbol_news:
                self.symbol_news[symbol] = []
            self.symbol_news[symbol].append(article)
    
    def get_recent_news(self, symbol: str, hours: int = 24) -> List[NewsArticle]:
        """Get recent news for a symbol"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        if symbol not in self.symbol_news:
            return []
        
        return [
            article for article in self.symbol_news[symbol]
            if article.timestamp > cutoff
        ]
    
    def get_sentiment_summary(self, symbol: str, hours: int = 24) -> Dict:
        """Get sentiment summary for a symbol"""
        articles = self.get_recent_news(symbol, hours)
        
        if not articles:
            return {'sentiment': 'neutral', 'score': 0.0, 'article_count': 0}
        
        analyzer = SentimentAnalyzer()
        sentiments = [analyzer.analyze_sentiment(a.title + " " + a.content) for a in articles]
        aggregated = analyzer.aggregate_sentiment(sentiments)
        
        return {
            'sentiment': aggregated['overall_sentiment'],
            'score': aggregated['overall_score'],
            'article_count': len(articles),
            'bullish_pct': aggregated['bullish_pct'],
            'bearish_pct': aggregated['bearish_pct'],
        }


def demo_sentiment_analysis():
    """Demonstrate sentiment analysis capabilities"""
    print("=" * 70)
    print("📰 SENTIMENT ANALYSIS DEMO")
    print("=" * 70)
    
    analyzer = SentimentAnalyzer()
    
    # Test headlines
    headlines = [
        "Apple beats earnings expectations with record iPhone sales",
        "Tesla stock plunges after production warning",
        "Microsoft announces major AI breakthrough in cloud computing",
        "Fed signals potential rate cuts amid inflation concerns",
        "Amazon reports strong growth but warns of challenges ahead",
        "Goldman Sachs upgrades tech sector to overweight",
        "Netflix faces lawsuit over accounting practices",
        "NVIDIA surges on datacenter demand surge",
    ]
    
    print(f"\n📊 Analyzing {len(headlines)} headlines:\n")
    
    sentiments = []
    for headline in headlines:
        sentiment = analyzer.analyze_sentiment(headline)
        sentiments.append(sentiment)
        
        emoji = "🟢" if sentiment['label'] == 'bullish' else "🔴" if sentiment['label'] == 'bearish' else "🟡"
        print(f"{emoji} {sentiment['label']:8s} ({sentiment['score']:+.3f}): {headline[:60]}")
    
    # Aggregate
    print("\n" + "-" * 70)
    aggregated = analyzer.aggregate_sentiment(sentiments)
    
    print(f"\n📈 Overall Market Sentiment:")
    print(f"   Sentiment: {aggregated['overall_sentiment'].upper()}")
    print(f"   Score: {aggregated['overall_score']:+.3f}")
    print(f"   Bullish: {aggregated['bullish_pct']:.1f}%")
    print(f"   Bearish: {aggregated['bearish_pct']:.1f}%")
    print(f"   Neutral: {aggregated['neutral_pct']:.1f}%")
    
    # News aggregator demo
    print("\n" + "-" * 70)
    print("\n🗞️  News Aggregator Demo:\n")
    
    aggregator = NewsAggregator()
    
    # Add sample articles
    now = datetime.now()
    articles = [
        NewsArticle(
            title="Apple reports record quarterly earnings",
            content="Apple Inc. exceeded analyst expectations...",
            source="Reuters",
            timestamp=now - timedelta(hours=1),
            symbols=["AAPL"],
        ),
        NewsArticle(
            title="Tech stocks rally on positive economic data",
            content="Major technology stocks surged today...",
            source="Bloomberg",
            timestamp=now - timedelta(hours=2),
            symbols=["AAPL", "MSFT", "GOOGL"],
        ),
        NewsArticle(
            title="Microsoft announces new AI partnership",
            content="Microsoft Corporation revealed a strategic partnership...",
            source="CNBC",
            timestamp=now - timedelta(hours=3),
            symbols=["MSFT"],
        ),
    ]
    
    for article in articles:
        aggregator.add_article(article)
    
    # Get sentiment for AAPL
    aapl_sentiment = aggregator.get_sentiment_summary("AAPL")
    print(f"AAPL Sentiment: {aapl_sentiment['sentiment'].upper()}")
    print(f"   Score: {aapl_sentiment['score']:+.3f}")
    print(f"   Articles: {aapl_sentiment['article_count']}")
    
    print("\n" + "=" * 70)
    print("✅ Sentiment Analysis Demo Complete!")
    print("=" * 70)
    
    return analyzer, aggregator


if __name__ == "__main__":
    demo_sentiment_analysis()
