"""
config.py
=========
Central configuration for equity-factor-lab.
All modules import from here. Edit this file to change research parameters.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BacktestConfig:
    # Position sizing
    holding_period_days:    int   = 5
    max_position_size:      float = 1.0

    # Transaction costs (bps)
    transaction_cost_bps:   float = 5.0
    slippage_bps:           float = 0.0

    # Viability thresholds (used by FactorValidator)
    min_ic:                 float = 0.02
    min_sharpe_ratio:       float = 0.30
    max_turnover:           float = 2.0

    # Minimum observations for valid backtest
    min_observations:       int   = 252


@dataclass
class FactorConfig:
    momentum_windows:   List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    rsi_period:         int       = 14
    bollinger_period:   int       = 20
    bollinger_std:      float     = 2.0


@dataclass
class ResearchConfig:
    # Signal decomposition
    z_window:           int   = 60

    # Fama-MacBeth
    fmb_horizon:        int   = 1
    fmb_hac_lags:       int   = 5

    # Long/short portfolio
    long_quantile:      float = 0.80
    short_quantile:     float = 0.20
    max_weight:         float = 0.10

    # Walk-forward
    train_days:         int   = 252
    test_days:          int   = 63
    wf_z_windows:       List[int] = field(default_factory=lambda: [20, 40, 60, 90])


@dataclass
class SentimentConfig:
    aggregation_method:  str   = "mean"
    positive_threshold:  float =  0.1
    negative_threshold:  float = -0.1


@dataclass
class Config:
    backtest:   BacktestConfig  = field(default_factory=BacktestConfig)
    factor:     FactorConfig    = field(default_factory=FactorConfig)
    research:   ResearchConfig  = field(default_factory=ResearchConfig)
    sentiment:  SentimentConfig = field(default_factory=SentimentConfig)


# Singleton — all modules import this object
config = Config()
