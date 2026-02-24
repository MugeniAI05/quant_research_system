"""
Factor Validation Module
Statistical testing and validation of factor predictive power
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

# Optional: HAC/Newey-West inference for overlapping horizons
try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
    _HAS_STATSMODELS = True
except Exception:
    sm = None
    cov_hac = None
    _HAS_STATSMODELS = False

from dataclasses import dataclass
import logging

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ICAnalysis:
    """Information Coefficient analysis results"""
    ic: float
    t_stat: float
    p_value: float
    n_obs: int
    is_significant: bool
    
    def to_dict(self) -> Dict:
        return {
            'ic': round(self.ic, 4),
            't_stat': round(self.t_stat, 3),
            'p_value': round(self.p_value, 4),
            'n_obs': self.n_obs,
            'is_significant': self.is_significant
        }


@dataclass
class DecayAnalysis:
    """Factor decay analysis results"""
    horizons: List[int]
    ics: List[float]
    t_stats: List[float]
    p_values: List[float]
    optimal_horizon: int
    
    def to_dict(self) -> Dict:
        return {
            'horizons': self.horizons,
            'ics': [round(ic, 4) for ic in self.ics],
            't_stats': [round(t, 3) for t in self.t_stats],
            'p_values': [round(p, 4) for p in self.p_values],
            'optimal_horizon': self.optimal_horizon
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy viewing"""
        return pd.DataFrame({
            'horizon': self.horizons,
            'ic': self.ics,
            't_stat': self.t_stats,
            'p_value': self.p_values
        })


@dataclass
class QuintileAnalysis:
    """Quintile analysis results"""
    quintiles: List[int]
    avg_returns: List[float]
    std_returns: List[float]
    n_obs: List[int]
    is_monotonic: bool
    spread: float  # Q5 - Q1 return spread
    
    def to_dict(self) -> Dict:
        return {
            'quintiles': self.quintiles,
            'avg_returns': [round(r * 100, 3) for r in self.avg_returns],
            'std_returns': [round(s * 100, 3) for s in self.std_returns],
            'n_obs': self.n_obs,
            'is_monotonic': self.is_monotonic,
            'spread': round(self.spread * 100, 3)
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame({
            'quintile': self.quintiles,
            'avg_return': [r * 100 for r in self.avg_returns],
            'std_return': [s * 100 for s in self.std_returns],
            'n_obs': self.n_obs
        })


@dataclass
class TurnoverAnalysis:
    """Position turnover analysis"""
    total_changes: int
    turnover_rate: float
    avg_holding_periods: float
    
    def to_dict(self) -> Dict:
        return {
            'total_changes': self.total_changes,
            'turnover_rate': round(self.turnover_rate, 4),
            'avg_holding_periods': round(self.avg_holding_periods, 1)
        }


@dataclass
class FactorValidationReport:
    """Complete validation report for a factor"""
    factor_name: str
    ic_analysis: ICAnalysis
    decay_analysis: Optional[DecayAnalysis]
    quintile_analysis: Optional[QuintileAnalysis]
    turnover_analysis: TurnoverAnalysis
    is_viable: bool
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            'factor_name': self.factor_name,
            'ic_analysis': self.ic_analysis.to_dict(),
            'decay_analysis': self.decay_analysis.to_dict() if self.decay_analysis else None,
            'quintile_analysis': self.quintile_analysis.to_dict() if self.quintile_analysis else None,
            'turnover_analysis': self.turnover_analysis.to_dict(),
            'is_viable': self.is_viable,
            'recommendation': self.recommendation
        }


class FactorValidator:
    """Statistical validation of factors"""
    
    @staticmethod
    def calculate_ic(
        factor_values: pd.Series,
        forward_returns: pd.Series,
        method: str = 'spearman',
        hac_lags: Optional[int] = None,
    ) -> ICAnalysis:
        """
        Calculate Information Coefficient
        
        IC measures the correlation between factor values at time t
        and forward returns from t to t+h
        
        Args:
            factor_values: Factor values
            forward_returns: Forward-looking returns
            method: 'spearman' (rank-based, more robust) or 'pearson'
            
        Returns:
            ICAnalysis object
        """
        # Align data
        common_idx = factor_values.index.intersection(forward_returns.index)
        factor_aligned = factor_values.loc[common_idx].dropna()
        returns_aligned = forward_returns.loc[common_idx].dropna()
        
        # Further align after dropna
        common_idx = factor_aligned.index.intersection(returns_aligned.index)
        
        if len(common_idx) < 20:
            logger.warning(f"Insufficient observations for IC: {len(common_idx)}")
            return ICAnalysis(
                ic=0.0, t_stat=0.0, p_value=1.0, 
                n_obs=len(common_idx), is_significant=False
            )
        
        factor_final = factor_aligned.loc[common_idx]
        returns_final = returns_aligned.loc[common_idx]
        
        # Calculate IC
        if method == 'spearman':
            ic, p_value = stats.spearmanr(factor_final, returns_final)
        else:
            ic, p_value = stats.pearsonr(factor_final, returns_final)
        
        # -----------------------------------------------------------------
        # Inference: HAC/Newey-West t-stat for the predictive relationship
        #
        # Why: forward returns with horizon > 1 overlap across dates, creating
        # autocorrelation. Naive t-stats (and Spearman p-values) are too optimistic.
        #
        # Approach:
        # - Keep IC as Spearman/Pearson correlation for interpretability.
        # - Compute a regression y ~ 1 + x (x is ranked if Spearman) and use
        #   HAC/Newey-West standard errors with nlags = hac_lags.
        # -----------------------------------------------------------------
        n = len(common_idx)
        t_stat = 0.0
        p_value_hac = p_value

        if _HAS_STATSMODELS and n >= 20:
            try:
                y = returns_final.values.astype(float)
                if method == "spearman":
                    x = pd.Series(factor_final.values).rank().values.astype(float)
                else:
                    x = factor_final.values.astype(float)

                X = sm.add_constant(x, has_constant="add")
                model = sm.OLS(y, X, missing="drop").fit()

                nlags = int(hac_lags) if hac_lags is not None else 0
                if nlags < 0:
                    nlags = 0

                V = cov_hac(model, nlags=nlags)
                se_beta = float(np.sqrt(V[1, 1]))
                beta = float(model.params[1])

                if se_beta > 0:
                    t_stat = beta / se_beta
                    # two-sided p-value (t distribution approximation)
                    df = max(n - 2, 1)
                    p_value_hac = float(2.0 * stats.t.sf(abs(t_stat), df=df))
            except Exception as e:
                logger.warning(f"HAC inference failed; falling back to naive t-stat. Error: {e}")

        if (not _HAS_STATSMODELS) or (p_value_hac is None):
            # Fallback: naive correlation-based t-stat (assumes i.i.d. observations)
            if abs(ic) < 1:
                t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)
            else:
                t_stat = 0.0
            p_value_hac = p_value

        p_value = p_value_hac

        is_significant = p_value < 0.05
        
        return ICAnalysis(
            ic=ic,
            t_stat=t_stat,
            p_value=p_value,
            n_obs=n,
            is_significant=is_significant
        )
    
    @staticmethod
    def calculate_decay(
        factor_values: pd.Series,
        prices: pd.Series,
        max_horizon: int = 20
    ) -> DecayAnalysis:
        """
        Calculate how IC decays over different forward horizons
        
        This tells you the optimal holding period for the factor
        
        Args:
            factor_values: Factor values
            prices: Price series
            max_horizon: Maximum horizon to test
            
        Returns:
            DecayAnalysis object
        """
        horizons = []
        ics = []
        t_stats = []
        p_values = []
        
        for horizon in range(1, max_horizon + 1):
            forward_returns = prices.pct_change(horizon).shift(-horizon)
            
            ic_analysis = FactorValidator.calculate_ic(factor_values, forward_returns, hac_lags=max(horizon - 1, 0))
            
            horizons.append(horizon)
            ics.append(ic_analysis.ic)
            t_stats.append(ic_analysis.t_stat)
            p_values.append(ic_analysis.p_value)
        
        # Find optimal horizon (highest absolute IC)
        optimal_idx = np.argmax(np.abs(ics))
        optimal_horizon = horizons[optimal_idx]
        
        return DecayAnalysis(
            horizons=horizons,
            ics=ics,
            t_stats=t_stats,
            p_values=p_values,
            optimal_horizon=optimal_horizon
        )
    
    @staticmethod
    def quintile_analysis(
        factor_values: pd.Series,
        forward_returns: pd.Series,
        n_quintiles: int = 5
    ) -> QuintileAnalysis:
        """
        Divide factor into quintiles and analyze return monotonicity
        
        A good factor should show monotonic returns across quintiles
        
        Args:
            factor_values: Factor values
            forward_returns: Forward returns
            n_quintiles: Number of quintiles (typically 5)
            
        Returns:
            QuintileAnalysis object
        """
        # Align data
        data = pd.DataFrame({
            'factor': factor_values,
            'forward_return': forward_returns
        }).dropna()
        
        if len(data) < n_quintiles * 10:
            logger.warning(f"Insufficient data for quintile analysis: {len(data)}")
            return QuintileAnalysis(
                quintiles=[], avg_returns=[], std_returns=[], 
                n_obs=[], is_monotonic=False, spread=0.0
            )
        
        # Assign quintiles
        try:
            data['quintile'] = pd.qcut(
                data['factor'], 
                q=n_quintiles, 
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            # If not enough unique values
            logger.warning("Not enough unique values for quintile split")
            return QuintileAnalysis(
                quintiles=[], avg_returns=[], std_returns=[], 
                n_obs=[], is_monotonic=False, spread=0.0
            )
        
        # Calculate statistics per quintile
        quintile_stats = data.groupby('quintile')['forward_return'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        quintiles = (quintile_stats['quintile'] + 1).tolist()
        avg_returns = quintile_stats['mean'].tolist()
        std_returns = quintile_stats['std'].tolist()
        n_obs = quintile_stats['count'].tolist()
        
        # Check monotonicity
        # Returns should either consistently increase or decrease
        diffs = np.diff(avg_returns)
        is_monotonic_increasing = all(d > 0 for d in diffs)
        is_monotonic_decreasing = all(d < 0 for d in diffs)
        is_monotonic = is_monotonic_increasing or is_monotonic_decreasing
        
        # Calculate spread (Q5 - Q1)
        spread = avg_returns[-1] - avg_returns[0]
        
        return QuintileAnalysis(
            quintiles=quintiles,
            avg_returns=avg_returns,
            std_returns=std_returns,
            n_obs=n_obs,
            is_monotonic=is_monotonic,
            spread=spread
        )
    
    @staticmethod
    def turnover_analysis(
        factor_values: pd.Series,
        threshold: float = 0.0
    ) -> TurnoverAnalysis:
        """
        Calculate position turnover
        
        High turnover = high transaction costs
        
        Args:
            factor_values: Factor values
            threshold: Signal threshold for position
            
        Returns:
            TurnoverAnalysis object
        """
        # Generate positions
        positions = (factor_values > threshold).astype(int)
        position_changes = positions.diff().abs()
        
        total_changes = int(position_changes.sum())
        n_periods = len(positions)
        
        turnover_rate = total_changes / n_periods if n_periods > 0 else 0
        avg_holding = 1 / turnover_rate if turnover_rate > 0 else np.inf
        
        return TurnoverAnalysis(
            total_changes=total_changes,
            turnover_rate=turnover_rate,
            avg_holding_periods=avg_holding
        )
    
    @staticmethod
    def validate_factor(
        factor_name: str,
        factor_values: pd.Series,
        prices: pd.Series,
        horizon: int = 5,
        run_decay: bool = True,
        run_quintile: bool = True
    ) -> FactorValidationReport:
        """
        Complete validation of a factor
        
        Args:
            factor_name: Name of the factor
            factor_values: Factor values
            prices: Price series
            horizon: Forward return horizon
            run_decay: Whether to run decay analysis (slow)
            run_quintile: Whether to run quintile analysis
            
        Returns:
            FactorValidationReport
        """
        logger.info(f"Validating factor: {factor_name}")
        
        # Calculate forward returns
        forward_returns = prices.pct_change(horizon).shift(-horizon)
        
        # 1. IC Analysis
        ic_analysis = FactorValidator.calculate_ic(factor_values, forward_returns, hac_lags=max(horizon - 1, 0))
        
        # 2. Decay Analysis (optional, slow)
        decay_analysis = None
        if run_decay:
            decay_analysis = FactorValidator.calculate_decay(
                factor_values, prices, max_horizon=20
            )
        
        # 3. Quintile Analysis (optional)
        quintile_analysis = None
        if run_quintile:
            quintile_analysis = FactorValidator.quintile_analysis(
                factor_values, forward_returns
            )
        
        # 4. Turnover Analysis
        turnover_analysis = FactorValidator.turnover_analysis(factor_values)
        
        # 5. Determine viability
        is_viable = (
            abs(ic_analysis.ic) >= config.backtest.min_ic and
            ic_analysis.is_significant and
            turnover_analysis.turnover_rate <= config.backtest.max_turnover
        )
        
        # 6. Generate recommendation
        if not ic_analysis.is_significant:
            recommendation = f"REJECT: IC not statistically significant (p={ic_analysis.p_value:.3f})"
        elif abs(ic_analysis.ic) < config.backtest.min_ic:
            recommendation = f"REJECT: IC too low ({ic_analysis.ic:.3f} < {config.backtest.min_ic})"
        elif turnover_analysis.turnover_rate > config.backtest.max_turnover:
            recommendation = f"CAUTION: High turnover ({turnover_analysis.turnover_rate:.2%}) will erode returns"
        else:
            recommendation = f"ACCEPT: Factor shows predictive power (IC={ic_analysis.ic:.3f}, p={ic_analysis.p_value:.3f})"
        
        return FactorValidationReport(
            factor_name=factor_name,
            ic_analysis=ic_analysis,
            decay_analysis=decay_analysis,
            quintile_analysis=quintile_analysis,
            turnover_analysis=turnover_analysis,
            is_viable=is_viable,
            recommendation=recommendation
        )


class MultiFactorValidator:
    """Validate multiple factors and rank them"""
    
    def __init__(self):
        self.reports: Dict[str, FactorValidationReport] = {}
    
    def validate_all(
        self,
        factors: Dict[str, pd.Series],
        prices: pd.Series,
        horizon: int = 5,
        run_decay: bool = False  # Slow, so default to False
    ) -> Dict[str, FactorValidationReport]:
        """Validate all factors"""
        logger.info(f"Validating {len(factors)} factors...")
        
        for name, values in factors.items():
            report = FactorValidator.validate_factor(
                factor_name=name,
                factor_values=values,
                prices=prices,
                horizon=horizon,
                run_decay=run_decay,
                run_quintile=True
            )
            self.reports[name] = report
        
        logger.info(f"Validation complete. {sum(1 for r in self.reports.values() if r.is_viable)} viable factors found.")
        return self.reports
    
    def get_viable_factors(self) -> List[str]:
        """Get list of viable factor names"""
        return [name for name, report in self.reports.items() if report.is_viable]
    
    def rank_factors(self, by: str = 'ic') -> pd.DataFrame:
        """
        Rank factors by quality metric
        
        Args:
            by: 'ic', 't_stat', or 'spread'
            
        Returns:
            DataFrame with ranked factors
        """
        if not self.reports:
            return pd.DataFrame()
        
        data = []
        for name, report in self.reports.items():
            ic_abs = abs(report.ic_analysis.ic)
            spread = report.quintile_analysis.spread if report.quintile_analysis else 0
            
            data.append({
                'factor': name,
                'ic': report.ic_analysis.ic,
                'ic_abs': ic_abs,
                't_stat': report.ic_analysis.t_stat,
                'p_value': report.ic_analysis.p_value,
                'spread': spread,
                'turnover': report.turnover_analysis.turnover_rate,
                'is_viable': report.is_viable
            })
        
        df = pd.DataFrame(data)
        
        # Sort by chosen metric
        if by == 'ic':
            df = df.sort_values('ic_abs', ascending=False)
        elif by == 't_stat':
            df = df.sort_values('t_stat', key=abs, ascending=False)
        elif by == 'spread':
            df = df.sort_values('spread', key=abs, ascending=False)
        
        return df
    
    def summary_report(self) -> str:
        """Generate text summary of validation results"""
        if not self.reports:
            return "No validation results available."
        
        viable = self.get_viable_factors()
        total = len(self.reports)
        
        report = f"=== FACTOR VALIDATION SUMMARY ===\n\n"
        report += f"Total factors tested: {total}\n"
        report += f"Viable factors: {len(viable)} ({len(viable)/total*100:.1f}%)\n\n"
        
        if viable:
            report += "Top 5 factors by IC:\n"
            ranked = self.rank_factors(by='ic')
            for i, row in ranked.head(5).iterrows():
                report += f"  {row['factor']:20s} IC={row['ic']:6.3f}  t={row['t_stat']:5.2f}  p={row['p_value']:.3f}\n"
        
        return report
