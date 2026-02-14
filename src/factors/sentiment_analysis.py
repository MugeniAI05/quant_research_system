"""
Sentiment Analysis Module
Production-grade sentiment scoring for financial news
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
import logging

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    ticker: str
    aggregate_score: float
    sentiment_label: str
    individual_scores: List[float]
    headlines: List[str]
    n_positive: int
    n_negative: int
    n_neutral: int
    score_std: float
    score_range: float
    
    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'aggregate_score': round(self.aggregate_score, 3),
            'sentiment_label': self.sentiment_label,
            'n_positive': self.n_positive,
            'n_negative': self.n_negative,
            'n_neutral': self.n_neutral,
            'score_std': round(self.score_std, 3),
            'score_range': round(self.score_range, 3),
            'n_headlines': len(self.headlines)
        }


class SentimentLexicon:
    """Financial sentiment lexicon based on Loughran-McDonald dictionary"""
    
    def __init__(self):
        # Positive words in financial context
        self.positive_words = {
            # Performance
            'surge', 'surges', 'surged', 'rally', 'rallies', 'rallied',
            'gain', 'gains', 'gained', 'beat', 'beats', 'beating',
            'bullish', 'strength', 'strong', 'stronger', 'strongest',
            'outperform', 'outperforms', 'outperformed', 'outperforming',
            'upgrade', 'upgrades', 'upgraded', 'exceed', 'exceeds', 'exceeded',
            'soar', 'soars', 'soared', 'jump', 'jumps', 'jumped',
            'rise', 'rises', 'rose', 'risen', 'growth', 'growing', 'grew',
            
            # Profitability
            'profitable', 'profit', 'profits', 'success', 'successful',
            'win', 'wins', 'won', 'positive', 'upbeat', 'optimistic',
            'confidence', 'confident', 'robust', 'solid', 'healthy',
            
            # Innovation & expansion
            'innovative', 'innovation', 'expand', 'expansion', 'expanding',
            'breakthrough', 'partnership', 'collaboration', 'award',
            
            # Market position
            'leader', 'leading', 'dominant', 'competitive', 'advantage'
        }
        
        # Negative words in financial context
        self.negative_words = {
            # Performance
            'fall', 'falls', 'fell', 'fallen', 'drop', 'drops', 'dropped',
            'decline', 'declines', 'declined', 'slump', 'slumps', 'slumped',
            'bearish', 'weakness', 'weak', 'weaker', 'weakest',
            'underperform', 'underperforms', 'underperformed', 'underperforming',
            'downgrade', 'downgrades', 'downgraded', 'miss', 'misses', 'missed',
            'plunge', 'plunges', 'plunged', 'tumble', 'tumbles', 'tumbled',
            
            # Financial distress
            'loss', 'losses', 'losing', 'lost', 'debt', 'default',
            'bankruptcy', 'bankrupt', 'insolvent', 'restructure', 'restructuring',
            
            # Risk & uncertainty
            'concern', 'concerns', 'concerned', 'worry', 'worries', 'worried',
            'risk', 'risks', 'risky', 'threat', 'threats', 'threaten',
            'negative', 'pessimistic', 'volatile', 'volatility',
            'uncertain', 'uncertainty', 'doubt', 'doubtful',
            
            # Legal & regulatory
            'lawsuit', 'litigation', 'probe', 'investigation', 'investigating',
            'fraud', 'fraudulent', 'scandal', 'fine', 'penalty', 'violation',
            
            # Operations
            'layoff', 'layoffs', 'cut', 'cuts', 'cutting', 'closure', 'shutdown'
        }
        
        # Negation words
        self.negations = {
            'not', 'no', 'never', 'none', 'nothing', 'neither', 
            'nowhere', 'nobody', 'barely', 'hardly', 'scarcely'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'highly': 1.5, 'extremely': 2.0,
            'significantly': 1.5, 'substantially': 1.5,
            'sharply': 1.8, 'dramatically': 2.0, 'massively': 2.0,
            'strongly': 1.6, 'considerably': 1.4
        }
        
        self.diminishers = {
            'slightly': 0.5, 'barely': 0.5, 'somewhat': 0.7,
            'relatively': 0.7, 'moderately': 0.8, 'fairly': 0.8
        }


class SentimentScorer:
    """Score financial news headlines for sentiment"""
    
    def __init__(self):
        self.lexicon = SentimentLexicon()
    
    def score_headline(self, headline: str) -> float:
        """
        Score a single headline from -1 (very negative) to +1 (very positive)
        
        Args:
            headline: News headline text
            
        Returns:
            Sentiment score
        """
        if not headline:
            return 0.0
        
        # Tokenize
        words = re.findall(r'\b\w+\b', headline.lower())
        
        if not words:
            return 0.0
        
        sentiment_score = 0.0
        negation_active = False
        intensity_multiplier = 1.0
        
        for i, word in enumerate(words):
            # Check for negation
            if word in self.lexicon.negations:
                negation_active = True
                continue
            
            # Check for intensity modifiers
            if word in self.lexicon.intensifiers:
                intensity_multiplier = self.lexicon.intensifiers[word]
                continue
            
            if word in self.lexicon.diminishers:
                intensity_multiplier = self.lexicon.diminishers[word]
                continue
            
            # Score the word
            word_score = 0.0
            
            if word in self.lexicon.positive_words:
                word_score = 1.0
            elif word in self.lexicon.negative_words:
                word_score = -1.0
            
            # Apply modifiers
            if word_score != 0:
                word_score *= intensity_multiplier
                
                if negation_active:
                    word_score *= -1
                    negation_active = False
                
                sentiment_score += word_score
                intensity_multiplier = 1.0
        
        # Normalize by sqrt(length) to reduce bias towards longer headlines
        normalized_score = sentiment_score / np.sqrt(len(words))
        
        # Clip to [-1, 1]
        return np.clip(normalized_score, -1.0, 1.0)
    
    def score_batch(self, headlines: List[str]) -> List[float]:
        """Score multiple headlines"""
        return [self.score_headline(h) for h in headlines]
    
    def aggregate_sentiment(
        self,
        headlines: List[str],
        ticker: str = "",
        method: str = None
    ) -> SentimentResult:
        """
        Aggregate sentiment across multiple headlines
        
        Args:
            headlines: List of news headlines
            ticker: Stock ticker
            method: Aggregation method ('mean', 'median', 'weighted')
            
        Returns:
            SentimentResult object
        """
        if method is None:
            method = config.sentiment.aggregation_method
        
        if not headlines:
            return SentimentResult(
                ticker=ticker,
                aggregate_score=0.0,
                sentiment_label='NEUTRAL',
                individual_scores=[],
                headlines=[],
                n_positive=0,
                n_negative=0,
                n_neutral=0,
                score_std=0.0,
                score_range=0.0
            )
        
        # Score all headlines
        scores = self.score_batch(headlines)
        
        # Calculate aggregate
        if method == 'mean':
            agg_score = np.mean(scores)
        elif method == 'median':
            agg_score = np.median(scores)
        else:  # weighted - more recent news weighted higher
            weights = np.exp(np.linspace(0, 1, len(scores)))
            agg_score = np.average(scores, weights=weights)
        
        # Categorize headlines
        n_positive = sum(1 for s in scores if s > 0.1)
        n_negative = sum(1 for s in scores if s < -0.1)
        n_neutral = len(scores) - n_positive - n_negative
        
        # Label aggregate sentiment
        if agg_score > config.sentiment.positive_threshold:
            label = 'POSITIVE'
        elif agg_score < config.sentiment.negative_threshold:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        return SentimentResult(
            ticker=ticker,
            aggregate_score=agg_score,
            sentiment_label=label,
            individual_scores=scores,
            headlines=headlines,
            n_positive=n_positive,
            n_negative=n_negative,
            n_neutral=n_neutral,
            score_std=float(np.std(scores)),
            score_range=float(max(scores) - min(scores)) if scores else 0.0
        )


class SentimentFactorBuilder:
    """Build tradeable factors from sentiment data"""
    
    @staticmethod
    def create_sentiment_series(
        sentiment_score: float,
        prices: pd.Series
    ) -> pd.Series:
        """
        Create a sentiment time series
        
        Note: In production, you'd have time-varying sentiment
        For single snapshot, we create a constant series
        """
        return pd.Series(sentiment_score, index=prices.index)
    
    @staticmethod
    def sentiment_price_divergence(
        sentiment_series: pd.Series,
        price_returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Detect divergence between sentiment and price action
        
        Positive divergence: Sentiment improving while price falling (bullish)
        Negative divergence: Sentiment declining while price rising (bearish)
        
        Args:
            sentiment_series: Sentiment scores over time
            price_returns: Price returns
            window: Window for z-score normalization
            
        Returns:
            Divergence score
        """
        # Normalize both to z-scores
        sent_mean = sentiment_series.rolling(window).mean()
        sent_std = sentiment_series.rolling(window).std()
        sent_zscore = (sentiment_series - sent_mean) / sent_std
        
        price_mean = price_returns.rolling(window).mean()
        price_std = price_returns.rolling(window).std()
        price_zscore = (price_returns - price_mean) / price_std
        
        # Divergence = sentiment moving opposite to price
        divergence = sent_zscore - price_zscore
        
        return divergence
    
    @staticmethod
    def sentiment_momentum(
        sentiment_series: pd.Series,
        window: int = 5
    ) -> pd.Series:
        """Calculate change in sentiment over time"""
        return sentiment_series.diff(window)
    
    @staticmethod
    def sentiment_surprise(
        sentiment_series: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        Calculate how much current sentiment deviates from baseline
        
        Large positive surprise = unexpectedly positive news
        Large negative surprise = unexpectedly negative news
        """
        rolling_mean = sentiment_series.rolling(lookback).mean()
        rolling_std = sentiment_series.rolling(lookback).std()
        
        surprise = (sentiment_series - rolling_mean) / rolling_std
        return surprise
    
    @classmethod
    def build_all_sentiment_factors(
        cls,
        sentiment_score: float,
        prices: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Build all sentiment-based factors
        
        Args:
            sentiment_score: Single sentiment score
            prices: Price series
            
        Returns:
            Dictionary of sentiment factors
        """
        factors = {}
        
        # Create sentiment series
        sentiment_series = cls.create_sentiment_series(sentiment_score, prices)
        
        # Price returns for divergence
        price_returns = prices.pct_change(5)
        
        # Build factors
        factors['sentiment_price_div_20'] = cls.sentiment_price_divergence(
            sentiment_series, price_returns, window=20
        )
        
        factors['sentiment_momentum_5'] = cls.sentiment_momentum(
            sentiment_series, window=5
        )
        
        factors['sentiment_surprise_20'] = cls.sentiment_surprise(
            sentiment_series, lookback=20
        )
        
        # Simple sentiment factor (constant in this case)
        factors['sentiment_raw'] = sentiment_series
        
        return factors


# Convenience function
def analyze_sentiment(headlines: List[str], ticker: str = "") -> SentimentResult:
    """Quick sentiment analysis"""
    scorer = SentimentScorer()
    return scorer.aggregate_sentiment(headlines, ticker)
