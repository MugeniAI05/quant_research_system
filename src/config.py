"""
Configuration and Constants for Quantitative Research System
Production-ready configuration management
"""

from dataclasses import dataclass
from typing import List
import os


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    # Trading costs
    transaction_cost_bps: float = 10.0  # 10 basis points per trade
    slippage_bps: float = 5.0           # 5 bps market impact
    
    # Holding periods
    holding_period_days: int = 5        # How long to hold positions
    rebalance_frequency: int = 1        # Daily rebalancing
    
    # Risk limits
    max_position_size: float = 0.10     # Max 10% of portfolio per position
    max_leverage: float = 1.0           # No leverage
    
    # Validation thresholds
    min_observations: int = 200         # Minimum data points for backtest
    min_sharpe_ratio: float = 0.5       # Minimum acceptable Sharpe
    min_ic: float = 0.02                # Minimum Information Coefficient
    max_turnover: float = 0.5           # Maximum daily turnover rate


@dataclass
class FactorConfig:
    """Configuration for factor computation"""
    # Momentum windows
    momentum_windows: List[int] = None
    
    # Volatility windows
    volatility_windows: List[int] = None
    
    # Reversal parameters
    rsi_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    def __post_init__(self):
        if self.momentum_windows is None:
            self.momentum_windows = [5, 10, 20, 60]
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20, 60]


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    aggregation_method: str = 'weighted'  # 'mean', 'median', 'weighted'
    min_headlines: int = 3                 # Minimum headlines for valid signal
    positive_threshold: float = 0.2        # Threshold for positive label
    negative_threshold: float = -0.2       # Threshold for negative label


@dataclass
class RiskConfig:
    """Risk management configuration"""
    target_volatility: float = 0.15        # 15% annualized vol target
    max_drawdown: float = 0.20             # 20% maximum drawdown
    stop_loss: float = 0.05                # 5% stop loss per position
    trailing_stop: float = 0.03            # 3% trailing stop
    
    # Position sizing
    use_kelly_criterion: bool = False      # Use Kelly for sizing
    kelly_fraction: float = 0.5            # Half Kelly (more conservative)


@dataclass
class APIConfig:
    """API configuration for data sources"""
    google_api_key: str = None
    
    # Rate limiting
    max_requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        # Load from environment
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")


class Config:
    """Master configuration object"""
    
    def __init__(self):
        self.backtest = BacktestConfig()
        self.factor = FactorConfig()
        self.sentiment = SentimentConfig()
        self.risk = RiskConfig()
        self.api = APIConfig()
    
    def to_dict(self):
        """Serialize config for logging"""
        return {
            'backtest': vars(self.backtest),
            'factor': vars(self.factor),
            'sentiment': vars(self.sentiment),
            'risk': vars(self.risk)
        }


# Singleton instance
config = Config()
