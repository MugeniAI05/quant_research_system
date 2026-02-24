"""
Backtest Engine
A research-grade backtesting module focused on correct timing, position logic, and cost-aware PnL.

Key upgrades vs the prior version:
- Removes overlapping-forward-return compounding (no more daily compounding of 5-day forward returns).
- Adds long/short support (threshold- or quantile-based).
- Adds a cross-sectional long/short ranking backtest (dollar-neutral by default).
- Makes the claims in docstrings match the math being done.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Config handling (keeps compatibility with your repo's config module,
# while allowing this file to run standalone in a pinch).
# ---------------------------------------------------------------------
try:
    from config import config  # type: ignore
except Exception:  # pragma: no cover
    @dataclass
    class _BacktestCfg:
        min_observations: int = 252
        holding_period_days: int = 1
        max_position_size: float = 1.0
        transaction_cost_bps: float = 15.0
        slippage_bps: float = 0.0
        min_sharpe_ratio: float = 0.0
        min_ic: float = 0.0

    @dataclass
    class _Config:
        backtest: _BacktestCfg = _BacktestCfg()

    config = _Config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------
@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest."""
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    win_rate: float
    information_coefficient: float
    calmar_ratio: float
    volatility: float
    total_return: float
    total_trades: int
    avg_holding_periods: float
    turnover: float  # annualized turnover (approx)

    def to_dict(self) -> Dict:
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "cagr_pct": round(self.cagr * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "information_coefficient": round(self.information_coefficient, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "volatility_pct": round(self.volatility * 100, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "total_trades": int(self.total_trades),
            "avg_holding_periods": round(self.avg_holding_periods, 1),
            "turnover_annualized": round(self.turnover, 3),
        }

    def is_acceptable(self) -> bool:
        """Check if metrics meet minimum thresholds."""
        return (
            self.sharpe_ratio >= config.backtest.min_sharpe_ratio
            and abs(self.information_coefficient) >= config.backtest.min_ic
        )


@dataclass
class BacktestResult:
    """Complete backtest results."""
    metrics: BacktestMetrics
    equity_curve: pd.Series
    positions: pd.Series | pd.DataFrame  # Series for single asset, DataFrame for cross-sectional
    returns: pd.Series
    drawdowns: pd.Series

    def to_dict(self) -> Dict:
        return {
            "metrics": self.metrics.to_dict(),
            "equity_curve": self.equity_curve.tolist(),
            "dates": self.equity_curve.index.strftime("%Y-%m-%d").tolist(),
            "final_equity": float(self.equity_curve.iloc[-1]),
            "n_observations": int(len(self.equity_curve)),
        }


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _compute_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def _sharpe(returns: pd.Series, ann_factor: float = 252.0) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(mu / sd * np.sqrt(ann_factor))


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_years = (len(equity) - 1) / 252.0
    if n_years <= 0:
        return 0.0
    return float((1.0 + total) ** (1.0 / n_years) - 1.0)


def _turnover_annualized(weights: pd.DataFrame, rebalance_period_days: int = 1) -> float:
    """Approx annualized turnover from weight changes.

    Turnover per rebalance is ~0.5 * sum_i |w_i(t) - w_i(t-1)|.
    Annualize by multiplying by number of rebalances per year.
    """
    if weights.shape[0] < 2:
        return 0.0
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    per_reb = 0.5 * dw
    rebs_per_year = 252.0 / float(max(rebalance_period_days, 1))
    return float(per_reb.mean() * rebs_per_year)


# ---------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------
class VectorBacktester:
    """Backtest signals with correct timing and cost accounting.

    Modes:
    1) Single-asset: (prices: Series, signal: Series)
       - Uses next-day realized returns.
       - Positions formed from lagged signal (prevents look-ahead).
       - Supports long-only and long/short.

    2) Cross-sectional: (prices: DataFrame, signal: DataFrame)
       - Rank signals cross-sectionally; long top quantile, short bottom quantile.
       - Turnover-based transaction costs.
       - Dollar-neutral by default.
    """

    def __init__(self):
        self.config = config.backtest

    # -----------------------------------------------------------------
    # Backwards-compatible API (so your existing run_analysis.py works)
    # -----------------------------------------------------------------
    def backtest_signal(
        self,
        prices: pd.Series,
        signal,
        signal_threshold: float = 0.0,
        mode: Literal["long_only", "long_short"] = "long_short",
    ) -> BacktestResult:
        """Compatibility wrapper.

        Your existing pipeline calls backtest_signal(prices=Series, signal=array, signal_threshold=...).
        Internally we run the corrected single-asset daily-return backtest.
        """
        sig = pd.Series(signal, index=prices.index)
        return self.backtest_single_asset(
            prices=prices,
            signal=sig,
            mode=mode,
            threshold=float(signal_threshold),
        )


    # -----------------------------
    # Single-asset backtest
    # -----------------------------
    def backtest_single_asset(
        self,
        prices: pd.Series,
        signal: pd.Series,
        *,
        mode: Literal["long_only", "long_short"] = "long_short",
        threshold: float = 0.0,
        position_size: float = 1.0,
        use_quantile_threshold: bool = False,
        quantile: float = 0.6,
        ic_horizon_days: int = 5,
    ) -> BacktestResult:
        """Backtest a single-asset signal using realized next-day returns.

        Args:
            prices: Price series
            signal: Signal series (factor values)
            mode: 'long_only' or 'long_short'
            threshold: absolute threshold for long/short decisions (when use_quantile_threshold=False)
            position_size: max absolute exposure
            use_quantile_threshold: if True, derive thresholds from in-sample signal quantiles (diagnostic)
            quantile: quantile used for long/short cutoffs when use_quantile_threshold=True
            ic_horizon_days: horizon used for IC diagnostic (default 5d forward returns)

        Returns:
            BacktestResult
        """
        logger.info(f"Running single-asset backtest with {len(prices)} observations")

        data = pd.DataFrame({"price": prices, "signal": signal}).dropna()
        if len(data) < self.config.min_observations:
            logger.error(f"Insufficient data: {len(data)} < {self.config.min_observations}")
            return self._empty_result(prices.index)

        # Realized next-day return (what you can actually earn with daily positions)
        data["ret_1d_fwd"] = data["price"].pct_change().shift(-1).fillna(0.0)

        # Forward return for IC diagnostic
        h = int(max(ic_horizon_days, 1))
        data["ret_hd_fwd"] = data["price"].pct_change(h).shift(-h)

        # Thresholds
        if use_quantile_threshold:
            hi = float(data["signal"].quantile(quantile))
            lo = float(data["signal"].quantile(1.0 - quantile))
        else:
            hi = float(threshold)
            lo = -float(threshold)

        # Lag signal to prevent look-ahead
        sig_lag = data["signal"].shift(1)

        if mode == "long_only":
            raw_pos = np.where(sig_lag > hi, position_size, 0.0)
        else:
            raw_pos = np.where(sig_lag > hi, position_size, np.where(sig_lag < lo, -position_size, 0.0))

        max_pos = float(min(self.config.max_position_size, abs(position_size)))
        data["position"] = pd.Series(raw_pos, index=data.index).clip(-max_pos, max_pos).fillna(0.0)

        # Position changes drive costs
        data["pos_change"] = data["position"].diff().abs().fillna(0.0)

        # Gross and net returns
        data["gross"] = data["position"] * data["ret_1d_fwd"]

        total_cost_bps = float(self.config.transaction_cost_bps + self.config.slippage_bps)
        data["cost"] = data["pos_change"] * (total_cost_bps / 10000.0)

        data["net"] = (data["gross"] - data["cost"]).fillna(0.0)

        equity = (1.0 + data["net"]).cumprod()
        dd = _compute_drawdown(equity)

        # Trades: count non-zero position changes
        trades = int((data["pos_change"] > 0).sum())

        # IC (Spearman rank correlation between lagged signal and h-day forward returns)
        ic = 0.0
        sv = data["signal"].shift(1)
        fr = data["ret_hd_fwd"]
        common = sv.dropna().index.intersection(fr.dropna().index)
        if len(common) > 10:
            ic = float(pd.Series(sv.loc[common]).rank().corr(pd.Series(fr.loc[common]).rank()))

        turnover = float(data["pos_change"].mean() * 252.0)

        metrics = self._metrics_from_daily_series(
            returns=data["net"],
            equity=equity,
            trades=trades,
            turnover=turnover,
            avg_holding=1.0,
            ic=ic,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity,
            positions=data["position"],
            returns=data["net"],
            drawdowns=dd,
        )

    # -----------------------------
    # Cross-sectional backtest
    # -----------------------------
    def backtest_cross_section(
        self,
        prices: pd.DataFrame,
        signal: pd.DataFrame,
        *,
        rebalance_every: int = 1,
        long_quantile: float = 0.9,
        short_quantile: float = 0.1,
        gross_exposure: float = 1.0,
        dollar_neutral: bool = True,
        max_weight: float = 0.05,
    ) -> BacktestResult:
        """Cross-sectional long/short ranking backtest.

        - Uses next-day returns.
        - Forms weights from lagged signals at rebalance dates.
        - Equal-weight in long and short baskets.
        - Costs proportional to turnover.
        """
        if not isinstance(prices, pd.DataFrame) or not isinstance(signal, pd.DataFrame):
            raise TypeError("prices and signal must be DataFrames for cross-sectional backtests")

        # Strip timezone and normalize to midnight — yfinance Ticker.history()
        # returns tz-aware timestamps that cause KeyError on .loc[dt] lookups.
        def _norm(df):
            idx = pd.to_datetime(df.index)
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_localize(None)
            df = df.copy()
            df.index = idx.normalize()
            return df[~df.index.duplicated(keep="first")]

        prices = _norm(prices)
        signal = _norm(signal)

        common_cols = prices.columns.intersection(signal.columns)
        # Use shared index so px.index == sig.index exactly
        shared = prices.index.intersection(signal.index)
        px  = prices.loc[shared, common_cols].copy()
        sig = signal.loc[shared, common_cols].copy()
        # Forward-fill NaNs rather than dropping rows (keeps index aligned)
        px  = px.ffill().dropna(how="all")
        sig = sig.ffill().reindex(px.index)

        if len(px) < self.config.min_observations:
            logger.error(f"Insufficient data: {len(px)} < {self.config.min_observations}")
            return self._empty_result(px.index)

        rebalance_every = int(max(rebalance_every, 1))

        # Next-day returns
        rets = px.pct_change().shift(-1).fillna(0.0)

        # Weight matrix
        w = pd.DataFrame(0.0, index=px.index, columns=px.columns)

        # Lagged signal for portfolio formation
        sig_lag = sig.shift(1)

        for t, dt in enumerate(px.index):
            if t == 0:
                continue

            if (t % rebalance_every) != 0:
                w.iloc[t] = w.iloc[t - 1].values
                continue

            # iloc avoids KeyError from any residual timestamp mismatches
            s = sig_lag.iloc[t].dropna()
            if len(s) < 10:
                w.iloc[t] = 0.0
                continue

            hi = float(s.quantile(long_quantile))
            lo = float(s.quantile(short_quantile))

            long_names = s[s >= hi].index
            short_names = s[s <= lo].index

            weights = pd.Series(0.0, index=px.columns)

            if len(long_names) > 0:
                weights.loc[long_names] = 1.0 / float(len(long_names))
            if len(short_names) > 0:
                weights.loc[short_names] = -1.0 / float(len(short_names))

            if dollar_neutral:
                lg = float(weights[weights > 0].sum())
                sg = float(-weights[weights < 0].sum())
                if lg > 0 and sg > 0:
                    weights[weights > 0] *= (0.5 / lg)
                    weights[weights < 0] *= (0.5 / sg)

            gross = float(weights.abs().sum())
            if gross > 0:
                weights *= (gross_exposure / gross)

            weights = weights.clip(-max_weight, max_weight)
            gross2 = float(weights.abs().sum())
            if gross2 > 0:
                weights *= (gross_exposure / gross2)

            w.iloc[t] = weights.values

        # Turnover + costs
        dw = w.diff().abs().fillna(0.0)
        turnover_per_day = 0.5 * dw.sum(axis=1)

        total_cost_bps = float(self.config.transaction_cost_bps + self.config.slippage_bps)
        costs = turnover_per_day * (total_cost_bps / 10000.0)

        port_gross = (w * rets).sum(axis=1)
        port_net = (port_gross - costs).fillna(0.0)

        equity = (1.0 + port_net).cumprod()
        dd = _compute_drawdown(equity)

        # Daily cross-sectional Spearman IC (diagnostic)
        daily_ic = []
        for dt in sig_lag.index.intersection(rets.index):
            try:
                s = sig_lag.loc[dt]
                r = rets.loc[dt]
            except KeyError:
                continue
            common = s.index.intersection(r.index)
            if len(common) < 10:
                continue
            ic = pd.Series(s.loc[common]).rank().corr(pd.Series(r.loc[common]).rank())
            if pd.notna(ic):
                daily_ic.append(float(ic))
        ic_mean = float(np.mean(daily_ic)) if len(daily_ic) else 0.0

        trades = int((turnover_per_day > 0).sum())
        turnover_ann = _turnover_annualized(w, rebalance_period_days=rebalance_every)

        metrics = self._metrics_from_daily_series(
            returns=port_net,
            equity=equity,
            trades=trades,
            turnover=turnover_ann,
            avg_holding=float(rebalance_every),
            ic=ic_mean,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity,
            positions=w,
            returns=port_net,
            drawdowns=dd,
        )

    # -----------------------------
    # Metrics helper
    # -----------------------------
    def _metrics_from_daily_series(
        self,
        *,
        returns: pd.Series,
        equity: pd.Series,
        trades: int,
        turnover: float,
        avg_holding: float,
        ic: float,
    ) -> BacktestMetrics:
        r = returns.dropna()

        sharpe = _sharpe(r, 252.0)
        cagr = _cagr(equity)
        max_dd = float(_compute_drawdown(equity).min()) if len(equity) else 0.0
        win_rate = float((r > 0).mean()) if len(r) else 0.0
        vol = float(r.std(ddof=1) * np.sqrt(252.0)) if len(r) > 1 else 0.0
        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0
        calmar = float(abs(cagr / max_dd)) if max_dd < 0 else 0.0

        return BacktestMetrics(
            sharpe_ratio=float(sharpe),
            cagr=float(cagr),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            information_coefficient=float(ic),
            calmar_ratio=float(calmar),
            volatility=float(vol),
            total_return=float(total_ret),
            total_trades=int(trades),
            avg_holding_periods=float(avg_holding),
            turnover=float(turnover),
        )

    # -----------------------------
    # Empty fallback
    # -----------------------------
    def _empty_result(self, index: pd.DatetimeIndex) -> BacktestResult:
        empty = pd.Series(0.0, index=index)
        equity = empty + 1.0
        dd = _compute_drawdown(equity)
        metrics = BacktestMetrics(
            sharpe_ratio=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            information_coefficient=0.0,
            calmar_ratio=0.0,
            volatility=0.0,
            total_return=0.0,
            total_trades=0,
            avg_holding_periods=0.0,
            turnover=0.0,
        )
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity,
            positions=empty,
            returns=empty,
            drawdowns=dd,
        )


def quick_backtest(prices: pd.Series, signal: pd.Series, threshold: float = 0.0) -> BacktestResult:
    """Quick single-asset long/short backtest (next-day returns)."""
    bt = VectorBacktester()
    return bt.backtest_single_asset(prices, signal, mode="long_short", threshold=threshold)


# Notes:
# - This module fixes the earlier bug where overlapping forward returns were compounded daily.
# - Statistical inference for overlapping horizons should use HAC/Newey-West and belongs in factor_validation.py.
