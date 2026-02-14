"""
Factor Engineering Module
Implements production-grade factor computation and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging
from dataclasses import dataclass

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Factor:
    """Container for a single factor"""
    name: str
    values: pd.Series
    family: str  # 'momentum', 'volatility', 'reversal', 'sentiment'
    description: str
    
    def to_dict(self) -> Dict:
        """Serialize factor"""
        return {
            'name': self.name,
            'values': self.values.tolist(),
            'dates': self.values.index.strftime('%Y-%m-%d').tolist(),
            'family': self.family,
            'description': self.description,
            'n_valid': int((~self.values.isna()).sum()),
            'mean': float(self.values.mean()),
            'std': float(self.values.std())
        }


class MomentumFactors:
    """Momentum-based factor calculations"""
    
    @staticmethod
    def simple_momentum(prices: pd.Series, window: int) -> pd.Series:
        """Simple price momentum: (P_t / P_{t-w}) - 1"""
        return prices.pct_change(window)
    
    @staticmethod
    def acceleration(prices: pd.Series, short_window: int = 20, long_window: int = 60) -> pd.Series:
        """Momentum acceleration: short_mom - long_mom"""
        short_mom = prices.pct_change(short_window)
        long_mom = prices.pct_change(long_window)
        return short_mom - long_mom
    
    @staticmethod
    def time_series_momentum(prices: pd.Series, window: int = 20) -> pd.Series:
        """Fraction of positive days in rolling window"""
        returns = prices.pct_change()
        return returns.rolling(window).apply(lambda x: (x > 0).sum() / len(x))
    
    @staticmethod
    def normalized_momentum(prices: pd.Series, signal_window: int = 20, norm_window: int = 60) -> pd.Series:
        """Z-score normalized momentum"""
        mom = prices.pct_change(signal_window)
        rolling_mean = mom.rolling(norm_window).mean()
        rolling_std = mom.rolling(norm_window).std()
        return (mom - rolling_mean) / rolling_std
    
    @staticmethod
    def exponential_momentum(prices: pd.Series, span: int = 20) -> pd.Series:
        """Exponentially weighted momentum"""
        ema = prices.ewm(span=span).mean()
        return (prices - ema) / ema
    
    @classmethod
    def compute_all(cls, prices: pd.Series) -> Dict[str, pd.Series]:
        """Compute all momentum factors"""
        factors = {}
        
        for window in config.factor.momentum_windows:
            factors[f'mom_{window}d'] = cls.simple_momentum(prices, window)
        
        factors['mom_accel'] = cls.acceleration(prices)
        factors['ts_momentum'] = cls.time_series_momentum(prices)
        
        for window in [20, 60]:
            factors[f'mom_{window}d_zscore'] = cls.normalized_momentum(prices, window)
        
        factors['exp_momentum'] = cls.exponential_momentum(prices)
        
        return factors


class VolatilityFactors:
    """Volatility-based factor calculations"""
    
    @staticmethod
    def realized_volatility(prices: pd.Series, window: int) -> pd.Series:
        """Annualized realized volatility"""
        returns = np.log(prices / prices.shift(1))
        return returns.rolling(window).std() * np.sqrt(252)
    
    @staticmethod
    def volatility_ratio(prices: pd.Series, short_window: int, long_window: int) -> pd.Series:
        """Ratio of short-term to long-term volatility"""
        short_vol = VolatilityFactors.realized_volatility(prices, short_window)
        long_vol = VolatilityFactors.realized_volatility(prices, long_window)
        return short_vol / long_vol
    
    @staticmethod
    def downside_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """Volatility of negative returns only"""
        returns = np.log(prices / prices.shift(1))
        
        def downside_std(x):
            neg_returns = x[x < 0]
            return neg_returns.std() if len(neg_returns) > 0 else 0
        
        return returns.rolling(window).apply(downside_std) * np.sqrt(252)
    
    @staticmethod
    def volatility_of_volatility(prices: pd.Series, vol_window: int = 20, volvol_window: int = 60) -> pd.Series:
        """Volatility of volatility (second-order risk)"""
        vol = VolatilityFactors.realized_volatility(prices, vol_window)
        return vol.rolling(volvol_window).std()
    
    @staticmethod
    def garman_klass_volatility(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Garman-Klass volatility estimator (more efficient than close-to-close)
        Requires OHLC data
        """
        hl = np.log(high / low) ** 2
        co = np.log(close / close.shift(1)) ** 2
        
        gk = 0.5 * hl - (2 * np.log(2) - 1) * co
        return np.sqrt(gk.rolling(window).mean() * 252)
    
    @classmethod
    def compute_all(cls, prices: pd.Series) -> Dict[str, pd.Series]:
        """Compute all volatility factors"""
        factors = {}
        
        for window in config.factor.volatility_windows:
            factors[f'vol_{window}d'] = cls.realized_volatility(prices, window)
        
        factors['vol_ratio_5_20'] = cls.volatility_ratio(prices, 5, 20)
        factors['vol_ratio_20_60'] = cls.volatility_ratio(prices, 20, 60)
        factors['downside_vol_20d'] = cls.downside_volatility(prices, 20)
        factors['volvol_20_60'] = cls.volatility_of_volatility(prices, 20, 60)
        
        return factors


class ReversalFactors:
    """Mean reversion factor calculations"""
    
    @staticmethod
    def distance_from_ma(prices: pd.Series, window: int) -> pd.Series:
        """Distance from moving average"""
        ma = prices.rolling(window).mean()
        return (prices - ma) / ma
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_position(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
        """Position within Bollinger Bands (-1 to +1)"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = ma + num_std * std
        lower = ma - num_std * std
        
        # Normalize to [-1, 1]
        position = (prices - ma) / (num_std * std)
        return position.clip(-1, 1)
    
    @staticmethod
    def mean_reversion_z_score(prices: pd.Series, window: int = 20) -> pd.Series:
        """Z-score for mean reversion"""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        return (prices - ma) / std
    
    @staticmethod
    def rate_of_change_oscillator(prices: pd.Series, window: int = 14) -> pd.Series:
        """Rate of change oscillator"""
        roc = prices.pct_change(window) * 100
        return roc
    
    @classmethod
    def compute_all(cls, prices: pd.Series) -> Dict[str, pd.Series]:
        """Compute all reversal factors"""
        factors = {}
        
        for window in [10, 20, 50]:
            factors[f'dist_from_ma{window}'] = cls.distance_from_ma(prices, window)
        
        factors['rsi_14'] = cls.rsi(prices, config.factor.rsi_period)
        factors['bb_position'] = cls.bollinger_position(
            prices, 
            config.factor.bollinger_period,
            config.factor.bollinger_std
        )
        factors['mr_zscore_20'] = cls.mean_reversion_z_score(prices, 20)
        factors['roc_14'] = cls.rate_of_change_oscillator(prices, 14)
        
        return factors


class VolumeFactors:
    """Volume-based factor calculations"""
    
    @staticmethod
    def volume_weighted_return(prices: pd.Series, volumes: pd.Series, window: int = 20) -> pd.Series:
        """Volume-weighted returns"""
        returns = prices.pct_change()
        avg_volume = volumes.rolling(window).mean()
        return returns * (volumes / avg_volume)
    
    @staticmethod
    def volume_trend(volumes: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
        """Volume trend indicator"""
        short_avg = volumes.rolling(short_window).mean()
        long_avg = volumes.rolling(long_window).mean()
        return (short_avg / long_avg) - 1
    
    @staticmethod
    def on_balance_volume(prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """On-Balance Volume indicator"""
        direction = np.sign(prices.diff())
        obv = (direction * volumes).cumsum()
        return obv
    
    @classmethod
    def compute_all(cls, prices: pd.Series, volumes: pd.Series) -> Dict[str, pd.Series]:
        """Compute all volume factors"""
        if volumes is None:
            return {}
        
        factors = {}
        factors['vol_weighted_ret'] = cls.volume_weighted_return(prices, volumes)
        factors['volume_trend'] = cls.volume_trend(volumes)
        factors['obv'] = cls.on_balance_volume(prices, volumes)
        
        return factors


class FactorEngine:
    """Main factor computation engine"""
    
    def __init__(self):
        self.factors = {}
    
    def compute_all_factors(
        self, 
        prices: pd.Series, 
        volumes: Optional[pd.Series] = None
    ) -> Dict[str, Factor]:
        """
        Compute all available factors
        
        Args:
            prices: Price series
            volumes: Optional volume series
            
        Returns:
            Dictionary of Factor objects
        """
        logger.info("Computing all factors...")
        
        all_factors = {}
        
        # Momentum factors
        mom_factors = MomentumFactors.compute_all(prices)
        for name, values in mom_factors.items():
            all_factors[name] = Factor(
                name=name,
                values=values,
                family='momentum',
                description=f'Momentum factor: {name}'
            )
        
        # Volatility factors
        vol_factors = VolatilityFactors.compute_all(prices)
        for name, values in vol_factors.items():
            all_factors[name] = Factor(
                name=name,
                values=values,
                family='volatility',
                description=f'Volatility factor: {name}'
            )
        
        # Reversal factors
        rev_factors = ReversalFactors.compute_all(prices)
        for name, values in rev_factors.items():
            all_factors[name] = Factor(
                name=name,
                values=values,
                family='reversal',
                description=f'Reversal factor: {name}'
            )
        
        # Volume factors (if available)
        if volumes is not None:
            vol_based_factors = VolumeFactors.compute_all(prices, volumes)
            for name, values in vol_based_factors.items():
                all_factors[name] = Factor(
                    name=name,
                    values=values,
                    family='volume',
                    description=f'Volume factor: {name}'
                )
        
        logger.info(f"Computed {len(all_factors)} factors across {len(set(f.family for f in all_factors.values()))} families")
        
        self.factors = all_factors
        return all_factors
    
    def get_factor(self, name: str) -> Optional[Factor]:
        """Get a specific factor by name"""
        return self.factors.get(name)
    
    def get_factors_by_family(self, family: str) -> Dict[str, Factor]:
        """Get all factors from a specific family"""
        return {
            name: factor 
            for name, factor in self.factors.items() 
            if factor.family == family
        }
    
    def list_factors(self) -> List[str]:
        """List all available factor names"""
        return list(self.factors.keys())
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all factors to a DataFrame"""
        if not self.factors:
            return pd.DataFrame()
        
        factor_dict = {name: factor.values for name, factor in self.factors.items()}
        return pd.DataFrame(factor_dict)
