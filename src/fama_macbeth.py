"""
fama_macbeth.py
===============
Improvement 4 + 7: Fama-MacBeth cross-sectional regression with factor controls.

The professional framework for panel factor research:

  Step 1: Each day t, run a cross-sectional OLS regression:
              r_{i,t+1} = alpha_t + beta_t * signal_{i,t} + gamma_t * controls_{i,t} + e_{i,t}

  Step 2: Collect the daily time series of beta_t coefficients.

  Step 3: Test whether mean(beta_t) is significantly different from zero
          using Newey-West HAC standard errors on the time series of betas.

Why this is better than per-stock IC averaging:
  - Correctly handles cross-sectional correlation (all stocks move together
    on market days, which inflates naive t-stats)
  - Controls for known factors simultaneously so you isolate INCREMENTAL
    predictive power of your signal
  - Standard in academic finance since Fama-MacBeth (1973)

Improvements over naive IC:
  - Raw coefficient: does OBV predict returns at all?
  - Controlled coefficient: does OBV predict returns AFTER controlling for
    momentum, reversal, volatility, and size?
  - If controlled coefficient drops to zero → OBV is a proxy for known effects
  - If controlled coefficient survives → OBV contains incremental information
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@dataclass
class FMBResult:
    """Results from a single Fama-MacBeth regression."""
    signal_name:        str
    n_days:             int
    n_stocks_avg:       float

    # Raw regression (signal only)
    raw_coef:           float
    raw_tstat:          float
    raw_pval:           float
    raw_r2:             float

    # Controlled regression (signal + controls)
    controlled_coef:    float
    controlled_tstat:   float
    controlled_pval:    float
    controlled_r2:      float

    # Control coefficients (for interpretation)
    control_coefs:      Dict[str, float]

    # Attenuation ratio: controlled/raw
    # < 0.5 → signal is mostly proxying known factors
    # > 0.5 → signal has genuine incremental content
    attenuation_ratio:  float


def _hac_tstat(coef_series: pd.Series, nlags: int = 5) -> Tuple[float, float, float]:
    """
    Compute mean, HAC t-stat and p-value for a time series of coefficients.
    Uses Newey-West to correct for autocorrelation in daily betas.
    """
    coef_series = coef_series.dropna()
    n    = len(coef_series)
    mean = float(coef_series.mean())

    if n < 20:
        return mean, np.nan, np.nan

    if HAS_STATSMODELS:
        try:
            y   = coef_series.values
            X   = sm.add_constant(np.ones(n))
            mod = sm.OLS(y, X).fit()
            V   = cov_hac(mod, nlags=nlags)
            se  = float(np.sqrt(V[0, 0]))
            t   = mean / se if se > 0 else 0.0
            p   = float(2 * stats.t.sf(abs(t), df=max(n - 1, 1)))
            return mean, t, p
        except Exception:
            pass

    # Fallback: naive t-stat
    se = float(coef_series.std() / np.sqrt(n))
    t  = mean / se if se > 0 else 0.0
    p  = float(2 * stats.t.sf(abs(t), df=max(n - 1, 1)))
    return mean, t, p


def run_fama_macbeth(
    signal_df:   pd.DataFrame,
    prices_df:   pd.DataFrame,
    control_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    horizon:     int = 1,
    min_stocks:  int = 20,
    hac_lags:    int = 5,
    signal_name: str = "signal",
) -> FMBResult:
    """
    Run Fama-MacBeth regression for a single signal.

    Args:
        signal_df   : (dates x tickers) signal values
        prices_df   : (dates x tickers) prices (to compute forward returns)
        control_dfs : dict of control_name -> (dates x tickers) DataFrames
        horizon     : forward return horizon in days
        min_stocks  : minimum stocks required for a valid cross-section
        hac_lags    : Newey-West lags for time-series HAC
        signal_name : label for output

    Returns:
        FMBResult dataclass
    """
    if control_dfs is None:
        control_dfs = {}

    # Forward returns
    fwd_rets = prices_df.pct_change(horizon).shift(-horizon)

    # Align all panels to common dates/tickers
    common_tickers = signal_df.columns
    for ctrl in control_dfs.values():
        common_tickers = common_tickers.intersection(ctrl.columns)
    common_tickers = common_tickers.intersection(prices_df.columns)

    common_dates = signal_df.index
    for ctrl in control_dfs.values():
        common_dates = common_dates.intersection(ctrl.index)
    common_dates = common_dates.intersection(fwd_rets.index)

    signal_df = signal_df.loc[common_dates, common_tickers]
    fwd_rets  = fwd_rets.loc[common_dates, common_tickers]
    ctrl_aligned = {
        k: v.loc[common_dates, common_tickers]
        for k, v in control_dfs.items()
    }

    # ── Daily cross-sectional regressions ───────────────────────────────
    raw_betas        = []
    ctrl_betas       = []
    ctrl_coef_dict   = {k: [] for k in control_dfs.keys()}
    raw_r2s          = []
    ctrl_r2s         = []
    n_stocks_per_day = []
    valid_dates      = []

    ctrl_names = list(control_dfs.keys())

    for dt in common_dates[:-horizon]:  # skip last horizon days (no forward return)
        sig_row = signal_df.loc[dt]
        ret_row = fwd_rets.loc[dt]

        # Build cross-section for this day
        df_day = pd.DataFrame({"signal": sig_row, "ret": ret_row})
        for k in ctrl_names:
            df_day[k] = ctrl_aligned[k].loc[dt]

        df_day = df_day.dropna()
        n = len(df_day)

        if n < min_stocks:
            continue

        # Winsorize at 1/99th percentile to reduce outlier influence
        for col in df_day.columns:
            lo = df_day[col].quantile(0.01)
            hi = df_day[col].quantile(0.99)
            df_day[col] = df_day[col].clip(lo, hi)

        # Cross-sectional standardize signal and controls (not returns)
        for col in ["signal"] + ctrl_names:
            mu    = df_day[col].mean()
            sigma = df_day[col].std()
            if sigma > 0:
                df_day[col] = (df_day[col] - mu) / sigma

        y = df_day["ret"].values

        # ── Raw regression (signal only) ────────────────────────────────
        try:
            X_raw = sm.add_constant(df_day[["signal"]].values) if HAS_STATSMODELS \
                    else np.column_stack([np.ones(n), df_day["signal"].values])
            if HAS_STATSMODELS:
                res_raw = sm.OLS(y, X_raw).fit()
                raw_betas.append(res_raw.params[1])
                raw_r2s.append(res_raw.rsquared)
            else:
                # numpy fallback
                coeffs = np.linalg.lstsq(X_raw, y, rcond=None)[0]
                raw_betas.append(float(coeffs[1]))
                ss_tot = np.sum((y - y.mean())**2)
                ss_res = np.sum((y - X_raw @ coeffs)**2)
                raw_r2s.append(1 - ss_res / max(ss_tot, 1e-12))
        except Exception:
            continue

        # ── Controlled regression (signal + all controls) ───────────────
        if ctrl_names:
            try:
                X_ctrl_cols = df_day[["signal"] + ctrl_names].values
                X_ctrl = sm.add_constant(X_ctrl_cols) if HAS_STATSMODELS \
                         else np.column_stack([np.ones(n), X_ctrl_cols])

                if HAS_STATSMODELS:
                    res_ctrl = sm.OLS(y, X_ctrl).fit()
                    ctrl_betas.append(res_ctrl.params[1])  # signal coef
                    ctrl_r2s.append(res_ctrl.rsquared)
                    for i, k in enumerate(ctrl_names):
                        ctrl_coef_dict[k].append(res_ctrl.params[i + 2])
                else:
                    coeffs_ctrl = np.linalg.lstsq(X_ctrl, y, rcond=None)[0]
                    ctrl_betas.append(float(coeffs_ctrl[1]))
                    ss_tot = np.sum((y - y.mean())**2)
                    ss_res = np.sum((y - X_ctrl @ coeffs_ctrl)**2)
                    ctrl_r2s.append(1 - ss_res / max(ss_tot, 1e-12))
                    for i, k in enumerate(ctrl_names):
                        ctrl_coef_dict[k].append(float(coeffs_ctrl[i + 2]))
            except Exception:
                ctrl_betas.append(np.nan)
                ctrl_r2s.append(np.nan)
        else:
            ctrl_betas.append(raw_betas[-1])
            ctrl_r2s.append(raw_r2s[-1])

        n_stocks_per_day.append(n)
        valid_dates.append(dt)

    if len(raw_betas) < 20:
        print(f"  WARNING: Only {len(raw_betas)} valid cross-sections for {signal_name}")
        return FMBResult(
            signal_name=signal_name, n_days=0, n_stocks_avg=0,
            raw_coef=np.nan, raw_tstat=np.nan, raw_pval=np.nan, raw_r2=np.nan,
            controlled_coef=np.nan, controlled_tstat=np.nan,
            controlled_pval=np.nan, controlled_r2=np.nan,
            control_coefs={}, attenuation_ratio=np.nan,
        )

    raw_series  = pd.Series(raw_betas,  index=valid_dates)
    ctrl_series = pd.Series(ctrl_betas, index=valid_dates)

    raw_mean,  raw_t,  raw_p  = _hac_tstat(raw_series,  nlags=hac_lags)
    ctrl_mean, ctrl_t, ctrl_p = _hac_tstat(ctrl_series, nlags=hac_lags)

    # Average control coefficients
    avg_ctrl_coefs = {
        k: float(np.nanmean(v)) for k, v in ctrl_coef_dict.items()
    }

    # Attenuation ratio
    atten = abs(ctrl_mean) / max(abs(raw_mean), 1e-9) if not np.isnan(ctrl_mean) else np.nan

    return FMBResult(
        signal_name       = signal_name,
        n_days            = len(valid_dates),
        n_stocks_avg      = float(np.mean(n_stocks_per_day)),
        raw_coef          = round(raw_mean, 6),
        raw_tstat         = round(raw_t, 3),
        raw_pval          = round(raw_p, 4),
        raw_r2            = round(float(np.nanmean(raw_r2s)), 5),
        controlled_coef   = round(ctrl_mean, 6),
        controlled_tstat  = round(ctrl_t, 3),
        controlled_pval   = round(ctrl_p, 4),
        controlled_r2     = round(float(np.nanmean(ctrl_r2s)), 5),
        control_coefs     = avg_ctrl_coefs,
        attenuation_ratio = round(atten, 3),
    )


def run_fmb_all_signals(
    signal_panels:  Dict[str, pd.DataFrame],
    prices_df:      pd.DataFrame,
    control_panels: Optional[Dict[str, pd.DataFrame]] = None,
    horizon:        int = 1,
) -> pd.DataFrame:
    """
    Run Fama-MacBeth for all signal decompositions and return summary table.
    """
    rows = []
    n = len(signal_panels)

    for i, (name, sig_df) in enumerate(signal_panels.items()):
        print(f"  [{i+1}/{n}] FMB: {name}...")
        result = run_fama_macbeth(
            signal_df   = sig_df,
            prices_df   = prices_df,
            control_dfs = control_panels,
            horizon     = horizon,
            signal_name = name,
        )
        rows.append({
            "Signal":           result.signal_name,
            "N days":           result.n_days,
            "Raw coef":         result.raw_coef,
            "Raw t-stat":       result.raw_tstat,
            "Raw p-val":        result.raw_pval,
            "Ctrl coef":        result.controlled_coef,
            "Ctrl t-stat":      result.controlled_tstat,
            "Ctrl p-val":       result.controlled_pval,
            "Attenuation":      result.attenuation_ratio,
            "Avg R²":           result.controlled_r2,
            "Sig (raw)":        "YES" if (not np.isnan(result.raw_pval) and result.raw_pval < 0.05) else "no",
            "Sig (ctrl)":       "YES" if (not np.isnan(result.controlled_pval) and result.controlled_pval < 0.05) else "no",
        })

    return pd.DataFrame(rows)
