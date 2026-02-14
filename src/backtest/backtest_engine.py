"""
Backtest Engine
Production-grade backtesting with realistic assumptions
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest"""
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    win_rate: float
    information_coefficient: float
    calmar_ratio: float
    avg_return_per_trade: float
    volatility: float
    total_return: float
    total_trades: int
    avg_holding_periods: float
    
    def to_dict(self) -> Dict:
        return {
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'cagr_pct': round(self.cagr * 100, 2),
            'max_drawdown_pct': round(self.max_drawdown * 100, 2),
            'win_rate_pct': round(self.win_rate * 100, 2),
            'information_coefficient': round(self.information_coefficient, 3),
            'calmar_ratio': round(self.calmar_ratio, 3),
            'avg_return_per_trade_bps': round(self.avg_return_per_trade * 10000, 2),
            'volatility_pct': round(self.volatility * 100, 2),
            'total_return_pct': round(self.total_return * 100, 2),
            'total_trades': self.total_trades,
            'avg_holding_periods': round(self.avg_holding_periods, 1)
        }
    
    def is_acceptable(self) -> bool:
        """Check if metrics meet minimum thresholds"""
        return (
            self.sharpe_ratio >= config.backtest.min_sharpe_ratio and
            abs(self.information_coefficient) >= config.backtest.min_ic
        )


@dataclass
class BacktestResult:
    """Complete backtest results"""
    metrics: BacktestMetrics
    equity_curve: pd.Series
    positions: pd.Series
    returns: pd.Series
    drawdowns: pd.Series
    
    def to_dict(self) -> Dict:
        return {
            'metrics': self.metrics.to_dict(),
            'equity_curve': self.equity_curve.tolist(),
            'dates': self.equity_curve.index.strftime('%Y-%m-%d').tolist(),
            'final_equity': float(self.equity_curve.iloc[-1]),
            'n_observations': len(self.equity_curve)
        }


class VectorBacktester:
    """
    Event-driven backtest engine with realistic trading costs
    
    Key features:
    - Prevents look-ahead bias
    - Includes transaction costs and slippage
    - Proper position sizing
    - Risk management
    """
    
    def __init__(self):
        self.config = config.backtest
    
    def backtest_signal(
        self,
        prices: pd.Series,
        signal: pd.Series,
        signal_threshold: float = 0.0,
        position_size: float = 1.0
    ) -> BacktestResult:
        """
        Backtest a signal with proper forward-looking logic
        
        Args:
            prices: Price series
            signal: Signal values (factor scores)
            signal_threshold: Minimum signal value to enter position
            position_size: Position sizing (0-1, default 1.0 = fully invested)
            
        Returns:
            BacktestResult object
        """
        logger.info(f"Running backtest with {len(prices)} observations")
        
        # Validate inputs
        if len(prices) < self.config.min_observations:
            logger.error(f"Insufficient data: {len(prices)} < {self.config.min_observations}")
            return self._empty_result(prices.index)
        
        # Align data
        data = pd.DataFrame({
            'price': prices,
            'signal': signal
        }).dropna()
        
        if len(data) < self.config.min_observations:
            logger.error(f"Insufficient valid data after dropna: {len(data)}")
            return self._empty_result(prices.index)
        
        # Calculate forward returns (what we'd actually earn)
        holding_period = self.config.holding_period_days
        data['forward_return'] = data['price'].pct_change(holding_period).shift(-holding_period)
        
        # Generate positions based on PAST signal values only
        # This prevents look-ahead bias
        data['raw_position'] = np.where(
            data['signal'] > signal_threshold,
            position_size,  # Long
            0.0  # Flat
        )
        
        # Apply position size limits
        data['position'] = data['raw_position'].clip(0, self.config.max_position_size)
        
        # Calculate position changes (when we trade)
        data['position_change'] = data['position'].diff().abs()
        
        # Calculate gross returns
        data['gross_return'] = data['position'].shift(1) * data['forward_return']
        
        # Apply transaction costs
        total_cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        data['transaction_costs'] = data['position_change'] * (total_cost_bps / 10000.0)
        
        # Net returns after costs
        data['net_return'] = data['gross_return'] - data['transaction_costs']
        
        # Calculate cumulative returns (equity curve)
        data['equity'] = (1 + data['net_return'].fillna(0)).cumprod()
        
        # Calculate drawdowns
        data['running_max'] = data['equity'].expanding().max()
        data['drawdown'] = (data['equity'] - data['running_max']) / data['running_max']
        
        # Calculate metrics
        metrics = self._calculate_metrics(data)
        
        return BacktestResult(
            metrics=metrics,
            equity_curve=data['equity'],
            positions=data['position'],
            returns=data['net_return'],
            drawdowns=data['drawdown']
        )
    
    def _calculate_metrics(self, data: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        
        returns = data['net_return'].dropna()
        
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Annualization factor
        annual_factor = 252 / self.config.holding_period_days
        
        # Mean and std of returns
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Sharpe Ratio
        sharpe = (mean_return / std_return * np.sqrt(annual_factor)) if std_return > 0 else 0.0
        
        # CAGR
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / annual_factor
        cagr = ((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        
        # Maximum Drawdown
        max_drawdown = data['drawdown'].min()
        
        # Win Rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        
        # Information Coefficient
        signal_values = data['signal'].shift(1).dropna()
        forward_returns = data['forward_return'].dropna()
        common_idx = signal_values.index.intersection(forward_returns.index)
        
        if len(common_idx) > 10:
            ic = np.corrcoef(
                signal_values.loc[common_idx],
                forward_returns.loc[common_idx]
            )[0, 1]
        else:
            ic = 0.0
        
        # Calmar Ratio (CAGR / abs(MaxDD))
        calmar = abs(cagr / max_drawdown) if max_drawdown < 0 else 0.0
        
        # Volatility (annualized)
        volatility = std_return * np.sqrt(annual_factor)
        
        # Trading statistics
        position_changes = data['position_change'].sum()
        total_trades = int(position_changes / 2)  # Divide by 2 for round trips
        avg_holding = self.config.holding_period_days
        
        return BacktestMetrics(
            sharpe_ratio=sharpe,
            cagr=cagr,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            information_coefficient=ic,
            calmar_ratio=calmar,
            avg_return_per_trade=mean_return,
            volatility=volatility,
            total_return=total_return,
            total_trades=total_trades,
            avg_holding_periods=avg_holding
        )
    
    def _empty_result(self, index: pd.DatetimeIndex) -> BacktestResult:
        """Return empty result on error"""
        empty_series = pd.Series(0.0, index=index)
        
        return BacktestResult(
            metrics=self._empty_metrics(),
            equity_curve=empty_series + 1.0,
            positions=empty_series,
            returns=empty_series,
            drawdowns=empty_series
        )
    
    def _empty_metrics(self) -> BacktestMetrics:
        """Return zero metrics"""
        return BacktestMetrics(
            sharpe_ratio=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            information_coefficient=0.0,
            calmar_ratio=0.0,
            avg_return_per_trade=0.0,
            volatility=0.0,
            total_return=0.0,
            total_trades=0,
            avg_holding_periods=0.0
        )


class WalkForwardOptimizer:
    """
    Walk-forward optimization for parameter tuning
    
    Prevents overfitting by using rolling train/test splits
    """
    
    def __init__(self, train_window: int = 252, test_window: int = 63):
        """
        Args:
            train_window: Training period in days (default 1 year)
            test_window: Test period in days (default 3 months)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.backtester = VectorBacktester()
    
    def walk_forward(
        self,
        prices: pd.Series,
        signal: pd.Series,
        thresholds: list = None
    ) -> Dict:
        """
        Perform walk-forward analysis
        
        Args:
            prices: Price series
            signal: Signal series
            thresholds: List of signal thresholds to test
            
        Returns:
            Results for each window
        """
        if thresholds is None:
            thresholds = [-0.1, 0.0, 0.1]
        
        total_length = len(prices)
        n_windows = (total_length - self.train_window) // self.test_window
        
        logger.info(f"Walk-forward with {n_windows} windows")
        
        results = []
        
        for i in range(n_windows):
            # Define windows
            train_start = i * self.test_window
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, total_length)
            
            # Split data
            train_prices = prices.iloc[train_start:train_end]
            train_signal = signal.iloc[train_start:train_end]
            test_prices = prices.iloc[test_start:test_end]
            test_signal = signal.iloc[test_start:test_end]
            
            # Find best threshold on training data
            best_sharpe = -np.inf
            best_threshold = 0.0
            
            for threshold in thresholds:
                train_result = self.backtester.backtest_signal(
                    train_prices, train_signal, threshold
                )
                
                if train_result.metrics.sharpe_ratio > best_sharpe:
                    best_sharpe = train_result.metrics.sharpe_ratio
                    best_threshold = threshold
            
            # Test on out-of-sample data
            test_result = self.backtester.backtest_signal(
                test_prices, test_signal, best_threshold
            )
            
            results.append({
                'window': i + 1,
                'train_sharpe': best_sharpe,
                'test_sharpe': test_result.metrics.sharpe_ratio,
                'best_threshold': best_threshold,
                'test_return': test_result.metrics.total_return
            })
        
        return {
            'results': results,
            'avg_test_sharpe': np.mean([r['test_sharpe'] for r in results]),
            'avg_test_return': np.mean([r['test_return'] for r in results])
        }


# Convenience function
def quick_backtest(
    prices: pd.Series,
    signal: pd.Series,
    threshold: float = 0.0
) -> BacktestResult:
    """Quick backtest with default settings"""
    backtester = VectorBacktester()
    return backtester.backtest_signal(prices, signal, threshold)
