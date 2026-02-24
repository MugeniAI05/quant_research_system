"""
Data Fetching Module
Handles market data and news acquisition with robust error handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data"""
    ticker: str
    prices: pd.Series
    volumes: Optional[pd.Series] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.start_date is None and len(self.prices) > 0:
            self.start_date = self.prices.index[0]
        if self.end_date is None and len(self.prices) > 0:
            self.end_date = self.prices.index[-1]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'ticker': self.ticker,
            'prices': self.prices.tolist(),
            'dates': self.prices.index.strftime('%Y-%m-%d').tolist(),
            'volumes': self.volumes.tolist() if self.volumes is not None else None,
            'n_days': len(self.prices),
            'start_date': str(self.start_date.date()) if self.start_date else None,
            'end_date': str(self.end_date.date()) if self.end_date else None
        }


@dataclass
class NewsData:
    """Container for news data"""
    ticker: str
    headlines: List[str]
    fetch_time: datetime
    source: str = "Live Search"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'ticker': self.ticker,
            'headlines': self.headlines,
            'n_headlines': len(self.headlines),
            'fetch_time': str(self.fetch_time),
            'source': self.source
        }


class MarketDataFetcher:
    """Fetches and validates market data from Yahoo Finance"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
    
    def fetch(
        self, 
        ticker: str, 
        period: str = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[MarketData]:
        """
        Fetch market data with retries and validation
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., '1y', '2y', '5y')
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            MarketData object or None if failed
        """
        logger.info(f"Fetching market data for {ticker} (period={period})")
        
        for attempt in range(self.max_retries):
            try:
                # Download data
                if start_date and end_date:
                    df = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date, 
                        progress=False,
                        auto_adjust=True  # Adjust for splits/dividends
                    )
                else:
                    df = yf.download(
                        ticker, 
                        period=period, 
                        progress=False,
                        auto_adjust=True
                    )
                
                # Validate data
                if df.empty:
                    logger.warning(f"No data returned for {ticker} (attempt {attempt + 1})")
                    continue
                
                # Handle multi-index columns (if multiple tickers)
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(ticker, axis=1, level=1)
                
                # Extract close prices and volumes
                if 'Close' not in df.columns:
                    logger.error(f"No 'Close' column in data for {ticker}")
                    continue
                
                prices = df['Close'].dropna()
                volumes = df['Volume'].dropna() if 'Volume' in df.columns else None
                
                # Validate minimum data
                if len(prices) < 50:
                    logger.warning(f"Insufficient data for {ticker}: {len(prices)} days")
                    return None
                
                # Check for data quality issues
                if (prices <= 0).any():
                    logger.warning(f"Invalid prices (<=0) detected for {ticker}")
                    prices = prices[prices > 0]
                
                # Check for excessive gaps
                returns = prices.pct_change().dropna()
                if (returns.abs() > 0.5).any():
                    logger.warning(f"Extreme returns detected for {ticker} (possible data error)")
                
                market_data = MarketData(
                    ticker=ticker.upper(),
                    prices=prices,
                    volumes=volumes
                )
                
                logger.info(f"Successfully fetched {len(prices)} days for {ticker}")
                return market_data
                
            except Exception as e:
                logger.error(f"Error fetching {ticker} (attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return None
        
        return None
    
    def fetch_multiple(
        self, 
        tickers: List[str], 
        period: str = "1y"
    ) -> Dict[str, MarketData]:
        """
        Fetch data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            period: Time period
            
        Returns:
            Dictionary mapping ticker to MarketData
        """
        results = {}
        
        for ticker in tickers:
            data = self.fetch(ticker, period)
            if data is not None:
                results[ticker.upper()] = data
        
        logger.info(f"Fetched data for {len(results)}/{len(tickers)} tickers")
        return results


class NewsDataFetcher:
    """Fetches news data with fallback mechanisms"""
    
    def __init__(self, use_fallback: bool = True):
        self.use_fallback = use_fallback
        
        # Fallback news database (for demo/testing)
        self.fallback_news = {
            'NVDA': [
                "Nvidia stock surges as AI demand shows no signs of slowing",
                "Analysts raise price targets on NVDA ahead of earnings",
                "Nvidia faces new competition from custom silicon chips",
                "Tech sector rally driven by semiconductor strength",
                "Nvidia announces new partnership for healthcare AI"
            ],
            'AAPL': [
                "Apple reveals new iPhone features in surprise announcement",
                "iPhone sales exceed expectations in China market",
                "Apple services revenue hits all-time high",
                "Concerns about smartphone market saturation persist",
                "Apple AI features draw mixed reviews from users"
            ],
            'TSLA': [
                "Tesla deliveries beat analyst estimates",
                "Concerns over EV price cuts impact margins",
                "Tesla energy storage business shows strong growth",
                "Regulatory scrutiny increases for autonomous driving",
                "Tesla expands manufacturing capacity in Texas"
            ],
            'DEFAULT': [
                "Market volatility affects trading volumes",
                "Investors monitor Federal Reserve policy decisions",
                "Sector rotation continues as rates remain elevated",
                "Earnings season brings mixed results",
                "Technical indicators suggest continued uncertainty"
            ]
        }
    
    def fetch(self, ticker: str) -> NewsData:
        """
        Fetch news for a ticker with DuckDuckGo and fallback
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            NewsData object
        """
        logger.info(f"Fetching news for {ticker}")
        
        headlines = []
        source = "Live Search"
        
        # Try live search first
        try:
            from duckduckgo_search import DDGS
            
            results = DDGS().news(keywords=f"{ticker} stock news", max_results=5)
            headlines = [item['title'] for item in results]
            
            logger.info(f"Fetched {len(headlines)} headlines from live search")
            
        except Exception as e:
            logger.warning(f"Live search failed: {str(e)}")
        
        # Use fallback if needed
        if not headlines and self.use_fallback:
            source = "Fallback Data"
            ticker_upper = ticker.upper()
            
            if ticker_upper in self.fallback_news:
                headlines = self.fallback_news[ticker_upper]
            else:
                headlines = self.fallback_news['DEFAULT']
            
            logger.info(f"Using fallback news ({len(headlines)} headlines)")
        
        return NewsData(
            ticker=ticker.upper(),
            headlines=headlines,
            fetch_time=datetime.now(),
            source=source
        )
    
    def fetch_multiple(self, tickers: List[str]) -> Dict[str, NewsData]:
        """Fetch news for multiple tickers"""
        results = {}
        
        for ticker in tickers:
            results[ticker.upper()] = self.fetch(ticker)
        
        return results


class DataValidator:
    """Validates data quality and integrity"""
    
    @staticmethod
    def validate_market_data(data: MarketData, min_days: int = 100) -> Tuple[bool, List[str]]:
        """
        Validate market data quality
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check minimum length
        if len(data.prices) < min_days:
            issues.append(f"Insufficient data: {len(data.prices)} < {min_days} days")
        
        # Check for NaN values
        if data.prices.isna().any():
            n_missing = data.prices.isna().sum()
            issues.append(f"Missing values: {n_missing} NaN entries")
        
        # Check for non-positive prices
        if (data.prices <= 0).any():
            issues.append("Invalid prices: negative or zero values detected")
        
        # Check for extreme returns (possible errors)
        returns = data.prices.pct_change().dropna()
        extreme_returns = returns[returns.abs() > 0.5]
        if len(extreme_returns) > 0:
            issues.append(f"Extreme returns: {len(extreme_returns)} days with >50% moves")
        
        # Check for flatlines (data feed issues)
        if (data.prices.diff() == 0).sum() > len(data.prices) * 0.1:
            issues.append("Suspicious flatlines: >10% of days have no price change")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def validate_news_data(data: NewsData, min_headlines: int = 1) -> Tuple[bool, List[str]]:
        """Validate news data quality"""
        issues = []
        
        if len(data.headlines) < min_headlines:
            issues.append(f"Insufficient headlines: {len(data.headlines)} < {min_headlines}")
        
        # Check for duplicate headlines
        if len(data.headlines) != len(set(data.headlines)):
            issues.append("Duplicate headlines detected")
        
        # Check for empty headlines
        empty_count = sum(1 for h in data.headlines if not h.strip())
        if empty_count > 0:
            issues.append(f"{empty_count} empty headlines")
        
        is_valid = len(issues) == 0
        return is_valid, issues


# Convenience functions for easy import
def fetch_market_data(ticker: str, period: str = "1y") -> Optional[MarketData]:
    """Quick fetch market data"""
    fetcher = MarketDataFetcher()
    return fetcher.fetch(ticker, period)


def fetch_news_data(ticker: str) -> NewsData:
    """Quick fetch news data"""
    fetcher = NewsDataFetcher()
    return fetcher.fetch(ticker)
