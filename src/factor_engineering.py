"""
factor_engineering.py
=====================
Factor computation library: momentum, volatility, reversal, and volume factors.

Used by run_analysis.py for single-stock exploratory research.
The main research pipeline (run_v3_research.py) uses signal_decomposition.py
instead, which focuses specifically on OBV sub-components.

This module is kept for completeness and as a general-purpose factor library.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class Factor:
    """Container for a single computed factor."""
    name:        str
    values:      pd.Series
    family:      str        # 'momentum' | 'volatility' | 'reversal' | 'volume'
    description: str

    def to_dict(self) -> Dict:
        return {
            "name":        self.name,
            "values":      self.values.tolist(),
            "dates":       self.values.index.strftime("%Y-%m-%d").tolist(),
            "family":      self.family,
            "description": self.description,
            "n_valid":     int((~self.values.isna()).sum()),
            "mean":        float(self.values.mean()),
            "std":         float(self.values.std()),
        }


# ════════════════════════════════════════════════════════════════════════════
# Momentum Factors
# ════════════════════════════════════════════════════════════════════════════

class MomentumFactors:

    @staticmethod
    def simple_momentum(prices: pd.Series, window: int) -> pd.Series:
        """(P_t / P_{t-w}) - 1"""
        return prices.pct_change(window)

    @staticmethod
    def acceleration(prices: pd.Series, short: int = 20, long: int = 60) -> pd.Series:
        """Short-term momentum minus long-term momentum."""
        return prices.pct_change(short) - prices.pct_change(long)

    @staticmethod
    def time_series_momentum(prices: pd.Series, window: int = 20) -> pd.Series:
        """Fraction of positive days in rolling window."""
        return prices.pct_change().rolling(window).apply(lambda x: (x > 0).mean())

    @staticmethod
    def normalized_momentum(prices: pd.Series, signal: int = 20, norm: int = 60) -> pd.Series:
        """Rolling z-score of momentum."""
        mom = prices.pct_change(signal)
        return (mom - mom.rolling(norm).mean()) / mom.rolling(norm).std()

    @staticmethod
    def exponential_momentum(prices: pd.Series, span: int = 20) -> pd.Series:
        """(Price - EMA) / EMA"""
        ema = prices.ewm(span=span).mean()
        return (prices - ema) / ema

    @classmethod
    def compute_all(cls, prices: pd.Series) -> Dict[str, pd.Series]:
        factors = {}
        for w in config.factor.momentum_windows:
            factors[f"mom_{w}d"] = cls.simple_momentum(prices, w)
        factors["mom_accel"]     = cls.acceleration(prices)
        factors["ts_momentum"]   = cls.time_series_momentum(prices)
        for w in [20, 60]:
            factors[f"mom_{w}d_z"] = cls.normalized_momentum(prices, w)
        factors["exp_momentum"]  = cls.exponential_momentum(prices)
        return factors


# ════════════════════════════════════════════════════════════════════════════
# Volatility Factors
# ════════════════════════════════════════════════════════════════════════════

class VolatilityFactors:

    @staticmethod
    def realized_volatility(prices: pd.Series, window: int) -> pd.Series:
        """Annualised close-to-close realized volatility."""
        return np.log(prices / prices.shift(1)).rolling(window).std() * np.sqrt(252)

    @staticmethod
    def volatility_ratio(prices: pd.Series, short: int, long: int) -> pd.Series:
        """Short-term vol / long-term vol."""
        return VolatilityFactors.realized_volatility(prices, short) / \
               VolatilityFactors.realized_volatility(prices, long)

    @staticmethod
    def downside_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """Annualised volatility of negative returns only."""
        rets = np.log(prices / prices.shift(1))
        return rets.rolling(window).apply(
            lambda x: x[x < 0].std() if len(x[x < 0]) > 0 else 0.0
        ) * np.sqrt(252)

    @staticmethod
    def volatility_of_volatility(prices: pd.Series, vol_w: int = 20, volvol_w: int = 60) -> pd.Series:
        """Volatility of rolling volatility."""
        return VolatilityFactors.realized_volatility(prices, vol_w).rolling(volvol_w).std()

    @classmethod
    def compute_all(cls, prices: pd.Series) -> Dict[str, pd.Series]:
        factors = {}
        for w in config.factor.volatility_windows:
            factors[f"vol_{w}d"] = cls.realized_volatility(prices, w)
        factors["vol_ratio_5_20"]  = cls.volatility_ratio(prices, 5, 20)
        factors["vol_ratio_20_60"] = cls.volatility_ratio(prices, 20, 60)
        factors["downside_vol"]    = cls.downside_volatility(prices, 20)
        factors["volvol"]          = cls.volatility_of_volatility(prices, 20, 60)
        return factors


# ════════════════════════════════════════════════════════════════════════════
# Reversal Factors
# ════════════════════════════════════════════════════════════════════════════

class ReversalFactors:

    @staticmethod
    def distance_from_ma(prices: pd.Series, window: int) -> pd.Series:
        """(Price - MA) / MA"""
        ma = prices.rolling(window).mean()
        return (prices - ma) / ma

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (0-100)."""
        delta = prices.diff()
        avg_gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        avg_loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_position(prices: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
        """Position within Bollinger Bands, clipped to [-1, 1]."""
        ma  = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return ((prices - ma) / (n_std * std)).clip(-1, 1)

    @staticmethod
    def mean_reversion_z(prices: pd.Series, window: int = 20) -> pd.Series:
        """Rolling z-score of price level."""
        ma  = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        return (prices - ma) / std

    @classmethod
    def compute_all(cls, prices: pd.Series) -> Dict[str, pd.Series]:
        factors = {}
        for w in [10, 20, 50]:
            factors[f"dist_ma{w}"] = cls.distance_from_ma(prices, w)
        factors["rsi_14"]      = cls.rsi(prices, config.factor.rsi_period)
        factors["bb_position"] = cls.bollinger_position(
            prices, config.factor.bollinger_period, config.factor.bollinger_std)
        factors["mr_z_20"]     = cls.mean_reversion_z(prices, 20)
        return factors


# ════════════════════════════════════════════════════════════════════════════
# Volume Factors (including stationary OBV transforms)
# ════════════════════════════════════════════════════════════════════════════

class VolumeFactors:

    @staticmethod
    def on_balance_volume(prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """
        Raw cumulative OBV.
        NOTE: Non-stationary — use the stationary transforms below for signals.
        """
        direction = np.sign(prices.pct_change()).fillna(0.0)
        return (direction * volumes.fillna(0.0)).cumsum()

    @staticmethod
    def obv_change(prices: pd.Series, volumes: pd.Series, n: int = 1) -> pd.Series:
        """N-day change in OBV — stationary, captures short-term flow."""
        return VolumeFactors.on_balance_volume(prices, volumes).diff(n)

    @staticmethod
    def obv_zscore(prices: pd.Series, volumes: pd.Series, n: int = 1, z_w: int = 60) -> pd.Series:
        """Rolling z-score of n-day OBV change — normalised for cross-asset comparison."""
        chg = VolumeFactors.obv_change(prices, volumes, n)
        return (chg - chg.rolling(z_w).mean()) / chg.rolling(z_w).std()

    @staticmethod
    def volume_trend(volumes: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
        """Short-window avg volume / long-window avg volume - 1."""
        return (volumes.rolling(short).mean() / volumes.rolling(long).mean()) - 1

    @staticmethod
    def volume_weighted_return(prices: pd.Series, volumes: pd.Series, window: int = 20) -> pd.Series:
        """Return scaled by volume relative to recent average."""
        avg_vol = volumes.rolling(window).mean()
        return prices.pct_change() * (volumes / avg_vol)

    @classmethod
    def compute_all(cls, prices: pd.Series, volumes: pd.Series) -> Dict[str, pd.Series]:
        if volumes is None:
            return {}
        obv = cls.on_balance_volume(prices, volumes)
        return {
            "obv_raw":          obv,
            "obv_1d_change":    cls.obv_change(prices, volumes, 1),
            "obv_5d_change":    cls.obv_change(prices, volumes, 5),
            "obv_20d_change":   cls.obv_change(prices, volumes, 20),
            "obv_1d_z60":       cls.obv_zscore(prices, volumes, 1,  60),
            "obv_5d_z60":       cls.obv_zscore(prices, volumes, 5,  60),
            "obv_20d_z60":      cls.obv_zscore(prices, volumes, 20, 60),
            "volume_trend":     cls.volume_trend(volumes),
            "vol_wt_ret":       cls.volume_weighted_return(prices, volumes),
        }


# ════════════════════════════════════════════════════════════════════════════
# Main Factor Engine
# ════════════════════════════════════════════════════════════════════════════

class FactorEngine:
    """Compute and store all factors for a single stock."""

    def __init__(self):
        self.factors: Dict[str, Factor] = {}

    def compute_all_factors(
        self,
        prices:  pd.Series,
        volumes: Optional[pd.Series] = None,
    ) -> Dict[str, Factor]:
        """Compute all factors. Returns dict of Factor objects."""
        all_raw: Dict[str, pd.Series] = {}
        all_raw.update(MomentumFactors.compute_all(prices))
        all_raw.update(VolatilityFactors.compute_all(prices))
        all_raw.update(ReversalFactors.compute_all(prices))
        if volumes is not None:
            all_raw.update(VolumeFactors.compute_all(prices, volumes))

        family_map = {
            "mom":      "momentum",
            "vol":      "volatility",
            "dist":     "reversal",
            "rsi":      "reversal",
            "bb":       "reversal",
            "mr":       "reversal",
            "ts":       "momentum",
            "exp":      "momentum",
            "obv":      "volume",
            "volume":   "volume",
            "volvol":   "volatility",
            "downside": "volatility",
        }

        self.factors = {}
        for name, values in all_raw.items():
            prefix = name.split("_")[0]
            family = family_map.get(prefix, "other")
            self.factors[name] = Factor(
                name=name,
                values=values,
                family=family,
                description=f"{family} factor: {name}",
            )

        return self.factors

    def get_factor(self, name: str) -> Optional[Factor]:
        return self.factors.get(name)

    def get_by_family(self, family: str) -> Dict[str, Factor]:
        return {k: v for k, v in self.factors.items() if v.family == family}

    def list_factors(self) -> List[str]:
        return list(self.factors.keys())

    def to_dataframe(self) -> pd.DataFrame:
        if not self.factors:
            return pd.DataFrame()
        return pd.DataFrame({name: f.values for name, f in self.factors.items()})
