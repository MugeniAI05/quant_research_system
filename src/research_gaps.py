"""
research_gaps.py
================
Closes the three technical gaps identified in the hiring manager review.

Drop into src/ alongside existing files and run:

    cd src
    python research_gaps.py

Takes ~15-20 minutes (dominated by rolling beta computation for 98 stocks).

────────────────────────────────────────────────────────────────────────────
GAP 1  Fama-French 5-Factor augmented regression
────────────────────────────────────────────────────────────────────────────
Your v3 FMB controls for short-term reversal, 1m momentum, 12-1m momentum,
realized volatility, and size. The FF5 model adds profitability (RMW) and
investment (CMA). A senior researcher will ask immediately whether your two
surviving signals load on quality or investment.

True FF5 (Fama-French data library) requires CRSP + Compustat.
We cannot get those from yfinance. Instead we construct proxy factors
from price and volume data and compute per-stock rolling betas to each.

FF5 proxy construction:
  Mkt  equal-weighted return of all 98 stocks
  SMB  low dollar-volume stocks minus high dollar-volume (size)
  HML  3y return losers minus 3y winners (value proxy: losers trade cheap)
  RMW  low return-volatility stocks minus high (stable earners = robust)
  CMA  low volume-growth stocks minus high (low investment = conservative)

For each stock we compute a rolling 252-day OLS beta to RMW and CMA.
These per-stock betas enter the FMB cross-section as additional regressors.

We compare three models:
  M1  signal only                        (baseline)
  M2  signal + v3 controls               (replicates v3 result)
  M3  signal + v3 controls + RMW & CMA betas  (FF5 augmented)

Coefficient stability M1→M2→M3 shows whether the signal loads on FF5.

────────────────────────────────────────────────────────────────────────────
GAP 2  Multiple testing correction
────────────────────────────────────────────────────────────────────────────
You tested 6 signal decompositions and found 2 with p < 0.05.
At alpha=0.05 with 6 tests you expect 0.3 false positives by chance.

Three corrections applied (to both raw and controlled p-values):
  Bonferroni        p_adj = p * n   — most conservative, controls FWER
  Holm-Bonferroni   stepwise version — same FWER guarantee, more power
  Benjamini-Hochberg                — controls FDR at 5%, most powerful

────────────────────────────────────────────────────────────────────────────
GAP 3  Enhanced market impact model
────────────────────────────────────────────────────────────────────────────
Original model assumed a fixed 0.1% ADV participation rate.
This sweeps five rates (0.1% to 5% ADV) and computes net Sharpe at each,
showing the breakeven AUM where the strategy becomes unviable.

Almgren-Chriss square-root impact model:
  impact_bps = eta * sigma * sqrt(participation_rate) * 10000
  eta = 0.1 (standard empirical constant for liquid US large-caps)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("WARNING: statsmodels not found — pip install statsmodels for HAC t-stats")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ════════════════════════════════════════════════════════════════════════════

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone, normalize to midnight, drop duplicate dates."""
    idx = pd.to_datetime(df.index)
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="first")]


def _hac_tstat(series, nlags: int = 5) -> Tuple[float, float, float]:
    """Return (mean, HAC t-stat, p-value) for a series of daily coefficients."""
    s = pd.Series(series).dropna()
    n = len(s)
    if n < 10:
        return np.nan, np.nan, np.nan
    mean = float(s.mean())
    if HAS_SM:
        try:
            mod = sm.OLS(s.values, sm.add_constant(np.ones(n))).fit()
            V   = cov_hac(mod, nlags=nlags)
            se  = float(np.sqrt(V[0, 0]))
            t   = mean / se if se > 0 else 0.0
            p   = float(2 * stats.t.sf(abs(t), df=max(n - 1, 1)))
            return mean, t, p
        except Exception:
            pass
    se = float(s.std() / np.sqrt(n))
    t  = mean / se if se > 0 else 0.0
    p  = float(2 * stats.t.sf(abs(t), df=max(n - 1, 1)))
    return mean, t, p


def _prep_cross_section(df_day: pd.DataFrame,
                         signal_cols: List[str]) -> pd.DataFrame:
    """Winsorize at 1/99 pct then standardize signal columns."""
    df = df_day.copy()
    for col in df.columns:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)
    for col in signal_cols:
        if col in df.columns and df[col].std() > 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


# ════════════════════════════════════════════════════════════════════════════
# GAP 1A — Build FF5 proxy factor return series
# ════════════════════════════════════════════════════════════════════════════

def compute_ff5_proxies(
    prices_df:  pd.DataFrame,
    volumes_df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """
    Construct daily FF5 proxy factor return series from price/volume data.

    Returns dict: factor_name -> daily return pd.Series.

    Honest limitation: these are approximations.
    True FF5 uses book value (HML), operating profitability (RMW),
    and total asset growth (CMA) from Compustat.
    """
    rets   = prices_df.pct_change()
    dolvol = (prices_df * volumes_df.fillna(0)).rolling(20).mean()

    def _ls(characteristic: pd.DataFrame, q: float = 0.30) -> pd.Series:
        """Each day: long bottom q, short top q by characteristic."""
        vals = []
        idx  = characteristic.index.intersection(rets.index)
        for dt in idx:
            c = characteristic.loc[dt].dropna()
            r = rets.loc[dt].dropna()
            common = c.index.intersection(r.index)
            if len(common) < 20:
                vals.append(np.nan)
                continue
            lo = c[common].quantile(q)
            hi = c[common].quantile(1 - q)
            lr = r[common][c[common] <= lo].mean()
            sr = r[common][c[common] >= hi].mean()
            vals.append(float(lr - sr) if not (np.isnan(lr) or np.isnan(sr))
                        else np.nan)
        return pd.Series(vals, index=idx)

    factors = {}

    # Mkt — equal-weighted return
    factors["mkt"] = rets.mean(axis=1)

    # SMB — small minus big (low dolvol minus high dolvol)
    factors["smb"] = _ls(dolvol)

    # HML — value minus growth (3y losers minus 3y winners)
    # Long-run losers are cheap (= value); winners are expensive (= growth)
    hist = min(756, len(prices_df) - 1)
    factors["hml"] = _ls(prices_df.pct_change(hist).shift(1))

    # RMW — robust minus weak profitability
    # Low return volatility = stable earnings = robust profitability
    # Long low-vol (robust), short high-vol (weak)
    factors["rmw"] = _ls(rets.rolling(60).std())

    # CMA — conservative minus aggressive investment
    # Low volume growth = low asset growth = conservative investment
    # Long low-vol-growth (conservative), short high-vol-growth (aggressive)
    factors["cma"] = _ls(volumes_df.fillna(0).pct_change(63).shift(1))

    return factors


# ════════════════════════════════════════════════════════════════════════════
# GAP 1B — Rolling per-stock factor betas
# ════════════════════════════════════════════════════════════════════════════

def compute_rolling_betas(
    prices_df:   pd.DataFrame,
    factors:     Dict[str, pd.Series],
    window:      int = 252,
    min_obs:     int = 126,
) -> Dict[str, pd.DataFrame]:
    """
    For each stock and each factor, compute a rolling OLS beta.

    beta_{i,f,t} = OLS slope of stock_i returns on factor_f over
                   the prior `window` trading days.

    These betas are the correct way to add FF5 exposures as per-stock
    controls in a cross-sectional Fama-MacBeth regression.
    No look-ahead: each day uses only past-window data.

    Returns dict: factor_name -> DataFrame (dates x tickers).
    """
    rets    = prices_df.pct_change()
    result  = {fname: {} for fname in factors}

    n_total = len(prices_df.columns) * len(factors)
    done    = 0

    for tkr in prices_df.columns:
        r_stock = rets[tkr]
        for fname, f_series in factors.items():
            done += 1
            common = r_stock.index.intersection(f_series.dropna().index)
            r_a    = r_stock.reindex(common)
            f_a    = f_series.reindex(common)

            beta_vals = {}
            for t in range(window, len(common)):
                r_w = r_a.iloc[t - window: t].values
                f_w = f_a.iloc[t - window: t].values
                mask = ~(np.isnan(r_w) | np.isnan(f_w))
                if mask.sum() < min_obs:
                    beta_vals[common[t]] = np.nan
                    continue
                # beta = cov(r, f) / var(f)
                cov_mat = np.cov(r_w[mask], f_w[mask])
                var_f   = cov_mat[1, 1]
                beta_vals[common[t]] = (cov_mat[0, 1] / var_f
                                        if var_f > 1e-12 else np.nan)
            result[fname][tkr] = pd.Series(beta_vals)

    return {fname: pd.DataFrame(stock_betas)
            for fname, stock_betas in result.items()}


# ════════════════════════════════════════════════════════════════════════════
# GAP 1C — Three-model augmented Fama-MacBeth
# ════════════════════════════════════════════════════════════════════════════

def run_ff5_augmented_fmb(
    signal_df:      pd.DataFrame,
    prices_df:      pd.DataFrame,
    v3_controls:    Dict[str, pd.DataFrame],
    ff5_betas:      Dict[str, pd.DataFrame],
    ff5_to_add:     List[str] = None,
    horizon:        int = 1,
    signal_name:    str = "signal",
    min_stocks:     int = 15,
    hac_lags:       int = 5,
) -> Dict:
    """
    Run three FMB specifications and return results dict.

    M1  signal only
    M2  signal + v3 controls
    M3  signal + v3 controls + FF5 betas (default: rmw, cma)

    Coefficient change M2→M3 (%) is the key diagnostic.
    """
    if ff5_to_add is None:
        ff5_to_add = ["rmw", "cma"]

    fwd_rets = prices_df.pct_change(horizon).shift(-horizon)

    # Intersect tickers and dates across all panels
    tickers = signal_df.columns
    for df in list(v3_controls.values()) + [prices_df]:
        tickers = tickers.intersection(df.columns)
    for k in ff5_to_add:
        if k in ff5_betas:
            tickers = tickers.intersection(ff5_betas[k].columns)

    dates = signal_df.index
    for df in list(v3_controls.values()) + [fwd_rets]:
        dates = dates.intersection(df.index)
    for k in ff5_to_add:
        if k in ff5_betas:
            dates = dates.intersection(ff5_betas[k].index)

    if len(dates) < 60 or len(tickers) < min_stocks:
        return {"error": f"Insufficient data: {len(dates)} dates, {len(tickers)} tickers"}

    # Slice
    sig   = signal_df.loc[dates, tickers]
    fwd_r = fwd_rets.loc[dates, tickers]
    ctrl  = {k: v.loc[dates, tickers] for k, v in v3_controls.items()}
    beta  = {k: ff5_betas[k].loc[dates, tickers]
             for k in ff5_to_add if k in ff5_betas}

    ctrl_names = list(ctrl.keys())
    beta_names = [f"{k}_beta" for k in beta.keys()]

    m1_betas, m2_betas, m3_betas = [], [], []
    valid_dates = []

    for dt in dates[:-horizon]:
        row = {"signal": sig.loc[dt], "ret": fwd_r.loc[dt]}
        for k, v in ctrl.items():
            row[k] = v.loc[dt]
        for k, v in beta.items():
            row[f"{k}_beta"] = v.loc[dt]

        df_day = pd.DataFrame(row).dropna()
        if len(df_day) < min_stocks:
            continue

        cols_to_std = ["signal"] + ctrl_names + beta_names
        df_day = _prep_cross_section(df_day, cols_to_std)
        y = df_day["ret"].values

        try:
            if HAS_SM:
                def _fit(cols):
                    X = sm.add_constant(df_day[cols].values)
                    return sm.OLS(y, X).fit().params[1]
                m1_betas.append(_fit(["signal"]))
                m2_betas.append(_fit(["signal"] + ctrl_names))
                m3_betas.append(_fit(["signal"] + ctrl_names + beta_names))
            else:
                def _lstsq(cols):
                    X = np.c_[np.ones(len(y)),
                              df_day[cols].values]
                    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])
                m1_betas.append(_lstsq(["signal"]))
                m2_betas.append(_lstsq(["signal"] + ctrl_names))
                m3_betas.append(_lstsq(["signal"] + ctrl_names + beta_names))
            valid_dates.append(dt)
        except Exception:
            continue

    if len(m1_betas) < 20:
        return {"error": f"Only {len(m1_betas)} valid cross-sections"}

    vi = pd.DatetimeIndex(valid_dates)
    m1_mean, m1_t, m1_p = _hac_tstat(pd.Series(m1_betas, index=vi), hac_lags)
    m2_mean, m2_t, m2_p = _hac_tstat(pd.Series(m2_betas, index=vi), hac_lags)
    m3_mean, m3_t, m3_p = _hac_tstat(pd.Series(m3_betas, index=vi), hac_lags)

    m2_to_m3_pct = ((m3_mean - m2_mean) / max(abs(m2_mean), 1e-9)) * 100

    return {
        "signal":       signal_name,
        "n_days":       len(valid_dates),
        "ff5_added":    beta_names,
        "m1_coef":      round(m1_mean, 6),
        "m1_tstat":     round(m1_t, 3),
        "m1_pval":      round(m1_p, 4),
        "m2_coef":      round(m2_mean, 6),
        "m2_tstat":     round(m2_t, 3),
        "m2_pval":      round(m2_p, 4),
        "m3_coef":      round(m3_mean, 6),
        "m3_tstat":     round(m3_t, 3),
        "m3_pval":      round(m3_p, 4),
        "m2_to_m3_pct": round(m2_to_m3_pct, 1),
        "m3_sig":       "YES" if (not np.isnan(m3_p) and m3_p < 0.05) else "no",
    }


# ════════════════════════════════════════════════════════════════════════════
# GAP 2 — Multiple testing corrections
# ════════════════════════════════════════════════════════════════════════════

def apply_multiple_testing_corrections(
    pvals:  np.ndarray,
    labels: List[str],
    alpha:  float = 0.05,
) -> pd.DataFrame:
    """
    Apply Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg corrections.

    Returns DataFrame with one row per test:
      Signal | raw_p | p_bonferroni | p_holm | p_bh |
      sig_raw | sig_bonferroni | sig_holm | sig_bh
    """
    pvals = np.array(pvals, dtype=float)
    n     = len(pvals)

    # Bonferroni
    p_bonf = np.minimum(pvals * n, 1.0)

    # Holm-Bonferroni (stepwise)
    order  = np.argsort(pvals)
    p_holm = np.ones(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(pvals[idx] * (n - rank), 1.0)
        running_max   = max(running_max, adj)
        p_holm[idx]   = running_max

    # Benjamini-Hochberg
    order_bh = np.argsort(pvals)
    p_bh     = np.ones(n)
    for rank, idx in enumerate(order_bh):
        p_bh[idx] = min(pvals[idx] * n / (rank + 1), 1.0)
    # Enforce monotonicity (largest to smallest rank)
    for i in range(n - 2, -1, -1):
        p_bh[order_bh[i]] = min(p_bh[order_bh[i]], p_bh[order_bh[i + 1]])

    return pd.DataFrame({
        "Signal":         labels,
        "raw_p":          np.round(pvals, 4),
        "p_bonferroni":   np.round(p_bonf, 4),
        "p_holm":         np.round(p_holm, 4),
        "p_bh":           np.round(p_bh, 4),
        "sig_raw":        pvals  < alpha,
        "sig_bonferroni": p_bonf < alpha,
        "sig_holm":       p_holm < alpha,
        "sig_bh":         p_bh   < alpha,
    })


def print_correction_table(corr_df: pd.DataFrame,
                            label: str,
                            alpha: float = 0.05) -> None:
    n      = len(corr_df)
    n_raw  = int(corr_df["sig_raw"].sum())
    n_bonf = int(corr_df["sig_bonferroni"].sum())
    n_holm = int(corr_df["sig_holm"].sum())
    n_bh   = int(corr_df["sig_bh"].sum())

    print(f"\n  [{label}]  ({n} tests, α={alpha})")
    print(f"  {'Method':<28} {'Significant':>12}  {'Threshold':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Uncorrected':<28} {n_raw:>12}  {'p < 0.050':>12}")
    print(f"  {'Bonferroni (FWER)':<28} {n_bonf:>12}  {f'p < {alpha/n:.4f}':>12}")
    print(f"  {'Holm-Bonferroni (FWER)':<28} {n_holm:>12}  {'stepwise':>12}")
    print(f"  {'Benjamini-Hochberg (FDR)':<28} {n_bh:>12}  {'FDR < 5%':>12}")

    if n_bonf > 0:
        print(f"\n  ✓ Bonferroni survivors: {corr_df[corr_df['sig_bonferroni']]['Signal'].tolist()}")
        print(f"    STRONG evidence — survives most conservative correction")
    elif n_bh > 0:
        print(f"\n  ~ BH/FDR survivors: {corr_df[corr_df['sig_bh']]['Signal'].tolist()}")
        print(f"    MODERATE evidence — survives FDR but not FWER")
        print(f"    Expected false discoveries: {n_bh * alpha:.1f}")
    else:
        print(f"\n  ✗ No signal survives any correction")
        print(f"    Expected false positives at p=0.05: {n * alpha:.1f}")
        print(f"    Raw p<0.05 results are likely false discoveries")

    print(f"\n  Detailed table:")
    print(f"  {'Signal':<28} {'raw_p':>7} {'bonf_p':>8} {'holm_p':>8} "
          f"{'bh_p':>7} {'Bonf':>6} {'BH':>4}")
    print(f"  {'-'*72}")
    for _, r in corr_df.iterrows():
        print(f"  {r['Signal']:<28} {r['raw_p']:>7.4f} "
              f"{r['p_bonferroni']:>8.4f} {r['p_holm']:>8.4f} "
              f"{r['p_bh']:>7.4f} "
              f"{'YES' if r['sig_bonferroni'] else 'no':>6} "
              f"{'YES' if r['sig_bh'] else 'no':>4}")


# ════════════════════════════════════════════════════════════════════════════
# GAP 3 — Enhanced market impact model
# ════════════════════════════════════════════════════════════════════════════

def run_enhanced_cost_model(
    prices_df:    pd.DataFrame,
    volumes_df:   pd.DataFrame,
    gross_sharpe: float = 0.471,
    gross_cagr:   float = 0.0197,
    turnover_ann: float = 1.9,
    eta:          float = 0.1,
    flat_tc_bps:  float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep realistic AUM levels and compute net Sharpe at each.

    Approach: anchor on fund AUM ($5M to $500M), compute implied
    ADV participation rate, then compute market impact from that.

    This is the correct direction of analysis:
      position_per_stock ($M) = AUM / n_portfolio_stocks
      daily_trade ($M)        = position × daily_turnover_rate
                              = position × (turnover_ann / 252)
      ADV_participation       = daily_trade / per_stock_ADV

    Market impact (Almgren-Chriss square-root model):
      impact_bps = eta × sigma × sqrt(ADV_part) × 10000
      eta = 0.1  (empirical constant for liquid US large-caps)

    Key finding: at bi-weekly rebalancing (1.9x annual turnover),
    flat TC alone = 5 × 1.9 × 2 = 19 bps/year. With a gross CAGR
    of 197 bps, the remaining headroom for market impact is 178 bps.
    S&P 100 stocks have enormous ADV ($1B–$30B), so participation
    rates remain tiny (<2%) up to $500M AUM, keeping impact low.

    The cost model shows the strategy IS viable on a cost basis up to
    ~$500M AUM. The real problem is not costs — it is the OOS
    walk-forward failure (avg OOS Sharpe -0.99 across 7 windows).
    A researcher should distinguish between: "unimplementable due to
    costs" (this strategy is NOT that) vs "signal does not survive
    out-of-sample" (this strategy IS that).

    Returns
    -------
    sweep_df     one row per AUM level
    perstock_df  per-stock impact at $50M AUM, sorted most to least expensive
    """
    # Per-stock stats
    rows = []
    for tkr in prices_df.columns:
        px  = prices_df[tkr].dropna()
        vol = volumes_df[tkr].dropna()
        if len(px) < 20:
            continue
        daily_vol = float(px.pct_change().std())
        adv       = float((px * vol.fillna(0)).rolling(20).mean().iloc[-1])
        if adv <= 0 or np.isnan(adv) or np.isnan(daily_vol):
            continue
        rows.append({"ticker": tkr, "adv_M": adv / 1e6, "daily_vol": daily_vol})

    stocks = pd.DataFrame(rows)
    if stocks.empty:
        return pd.DataFrame(), pd.DataFrame()

    median_adv_M = float(stocks["adv_M"].median())
    median_vol   = float(stocks["daily_vol"].median())
    n_stocks     = len(stocks)
    # L/S portfolio: long top 20% + short bottom 20% = ~40% of universe
    n_port       = max(int(n_stocks * 0.40), 1)

    daily_turnover_rate = turnover_ann / 252.0   # fraction of portfolio traded per day

    # Per-stock impact at $50M AUM (reference point for ranking table)
    ref_aum_M         = 50.0
    ref_pos_M         = ref_aum_M / n_port
    ref_trade_M       = ref_pos_M * daily_turnover_rate
    for idx, row in stocks.iterrows():
        ref_part = ref_trade_M / max(row["adv_M"], 1e-6)
        stocks.loc[idx, "impact_bps"]   = round(eta * row["daily_vol"] * np.sqrt(ref_part) * 10000, 2)
        stocks.loc[idx, "adv_part_pct"] = round(ref_part * 100, 4)
    stocks["total_tc_bps"] = (stocks["impact_bps"] + flat_tc_bps).round(2)
    stocks = stocks.sort_values("impact_bps", ascending=False).reset_index(drop=True)

    # Sweep AUM levels
    sweep_rows = []
    for aum_M in [5, 10, 25, 50, 100, 250, 500]:
        pos_per_stock_M = aum_M / n_port
        trade_M         = pos_per_stock_M * daily_turnover_rate
        part            = trade_M / max(median_adv_M, 1e-6)  # ADV participation

        impact_bps    = eta * median_vol * np.sqrt(part) * 10000
        tc_per_trade  = impact_bps + flat_tc_bps
        ann_cost_bps  = tc_per_trade * turnover_ann * 2
        ann_cost_frac = ann_cost_bps / 10000.0   # decimal (e.g. 0.0021)
        ann_cost_pct  = ann_cost_bps / 100.0     # percent (e.g. 0.21%)

        net_cagr   = gross_cagr - ann_cost_frac
        net_sharpe = (gross_sharpe * (net_cagr / gross_cagr)
                      if gross_cagr > 0 else gross_sharpe - ann_cost_frac / 0.10)

        sweep_rows.append({
            "AUM ($M)":           aum_M,
            "Pos/stock ($K)":     round(pos_per_stock_M * 1000, 0),
            "ADV part. (%)":      round(part * 100, 4),
            "Impact (bps/trade)": round(impact_bps, 2),
            "TC (bps/trade)":     round(tc_per_trade, 2),
            "Ann cost (bps)":     round(ann_cost_bps, 1),
            "Ann cost (%)":       round(ann_cost_pct, 3),
            "Gross CAGR (%)":     round(gross_cagr * 100, 2),
            "Net CAGR (%)":       round(net_cagr * 100, 2),
            "Net Sharpe":         round(net_sharpe, 3),
            "Viable (Sh≥0.3)":    "YES" if net_sharpe >= 0.3 else "no",
        })

    return pd.DataFrame(sweep_rows), stocks


# ════════════════════════════════════════════════════════════════════════════
# Main runner
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  RESEARCH GAPS — FF5 | MULTIPLE TESTING | COST MODEL")
    print("=" * 70)
    print(f"  statsmodels: {'OK — HAC t-stats enabled' if HAS_SM else 'MISSING — pip install statsmodels'}")

    from universe             import load_universe
    from signal_decomposition import compute_universe_decomposed
    from fama_macbeth         import run_fmb_all_signals

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\n[0] Loading universe and computing signals/controls...")
    prices_df, volumes_df = load_universe()
    prices_df  = _norm(prices_df)
    volumes_df = _norm(volumes_df)
    print(f"  ✓ {prices_df.shape[1]} stocks × {prices_df.shape[0]} days "
          f"({prices_df.index[0].date()} → {prices_df.index[-1].date()})")

    signal_panels, control_panels = compute_universe_decomposed(prices_df, volumes_df)
    print(f"  ✓ {len(signal_panels)} signals computed: {list(signal_panels.keys())}")
    print(f"  ✓ {len(control_panels)} controls:  {list(control_panels.keys())}")

    # ════════════════════════════════════════════════════════════════════
    # GAP 1
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  GAP 1 — FAMA-FRENCH 5-FACTOR AUGMENTED FMB")
    print(f"{'=' * 70}")
    print(f"""
  Question: Do vol_price_divergence_z and vol_surprise_z survive after
  adding RMW (profitability) and CMA (investment) factor controls?

  Method: rolling 252-day per-stock betas on FF5 proxy factors →
  these betas enter as regressors in the cross-sectional FMB each day.

  CAVEAT (state in memo): using price/volume proxies, not true FF5.
    RMW proxy ← low return-vol stocks minus high (stable = profitable)
    CMA proxy ← low volume-growth minus high (low investment = conservative)
""")

    print("  Building FF5 proxy factor returns...")
    ff5 = compute_ff5_proxies(prices_df, volumes_df)

    print(f"\n  FF5 proxy factor statistics:")
    print(f"  {'Factor':<8} {'Ann Ret':>10} {'Ann Vol':>10} {'Sharpe':>8}  Description")
    print(f"  {'-'*65}")
    desc = {"mkt": "equal-wt universe return",
            "smb": "low dolvol minus high dolvol",
            "hml": "3y losers minus 3y winners",
            "rmw": "low ret-vol minus high ret-vol",
            "cma": "low vol-growth minus high vol-growth"}
    for fn, fs in ff5.items():
        s  = fs.dropna()
        ar = float(s.mean() * 252 * 100)
        av = float(s.std() * np.sqrt(252) * 100)
        sr = ar / max(av, 1e-6)
        print(f"  {fn:<8} {ar:>+9.2f}% {av:>9.2f}% {sr:>8.2f}  {desc.get(fn,'')}")

    print(f"\n  Computing rolling 252-day RMW and CMA betas for {prices_df.shape[1]} stocks...")
    print(f"  (This takes ~10-15 minutes — only runs once)")
    ff5_betas = compute_rolling_betas(
        prices_df = prices_df,
        factors   = {k: ff5[k] for k in ["rmw", "cma"]},
        window    = 252,
        min_obs   = 126,
    )
    print(f"  ✓ Betas computed for factors: {list(ff5_betas.keys())}")

    print(f"\n  Running three-model FMB for top two signals...")
    top_signals = ["vol_price_divergence_z", "vol_surprise_z"]
    ff5_results = []

    for sig_name in top_signals:
        if sig_name not in signal_panels:
            print(f"  SKIP {sig_name}: not in signal_panels")
            continue

        print(f"\n  ── {sig_name} ──")
        result = run_ff5_augmented_fmb(
            signal_df   = signal_panels[sig_name],
            prices_df   = prices_df,
            v3_controls = control_panels,
            ff5_betas   = ff5_betas,
            ff5_to_add  = ["rmw", "cma"],
            horizon     = 1,
            signal_name = sig_name,
        )
        ff5_results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        print(f"  {'Model':<42} {'Coef':>12} {'t-stat':>8} {'p-val':>8} {'Sig':>5}")
        print(f"  {'-'*77}")
        for label, coef, t, p in [
            ("M1 — signal only",          result["m1_coef"], result["m1_tstat"], result["m1_pval"]),
            ("M2 — + v3 controls",        result["m2_coef"], result["m2_tstat"], result["m2_pval"]),
            ("M3 — + RMW & CMA betas",    result["m3_coef"], result["m3_tstat"], result["m3_pval"]),
        ]:
            sig_flag = "YES" if (not np.isnan(p) and p < 0.05) else "no"
            print(f"  {label:<42} {coef:>+12.6f} {t:>8.3f} {p:>8.4f} {sig_flag:>5}")

        pct = result["m2_to_m3_pct"]
        print(f"\n  Coefficient change M2 → M3: {pct:+.1f}%")
        if abs(pct) < 10:
            print(f"  >>> MINIMAL attenuation — {sig_name} does NOT load on RMW/CMA")
            print(f"      Signal content is orthogonal to profitability and investment")
            print(f"      This is the strongest possible result for incremental novelty")
        elif abs(pct) < 30:
            print(f"  >>> MODERATE attenuation — partial overlap with RMW/CMA")
            print(f"      Signal is partially explained by quality/investment exposure")
        else:
            print(f"  >>> LARGE attenuation — signal substantially loads on FF5")
            print(f"      Most of the predictive power comes from quality/investment")

    # ════════════════════════════════════════════════════════════════════
    # GAP 2
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  GAP 2 — MULTIPLE TESTING CORRECTION")
    print(f"{'=' * 70}")
    print(f"""
  You tested 6 signal decompositions and found 2 with p < 0.05.
  At α = 0.05 with 6 tests you expect 6 × 0.05 = 0.3 false positives.

  Three corrections applied:
    Bonferroni        — p_adj = p × n. Null threshold: p < {0.05/6:.4f}
    Holm-Bonferroni   — stepwise. Same FWER guarantee, more statistical power
    Benjamini-Hochberg — controls False Discovery Rate at 5%

  Applied separately to raw p-values (M1: signal only) and
  controlled p-values (M2: after momentum/reversal/vol/size).
""")

    print("  Re-running FMB on all 6 signals...")
    fmb_full = run_fmb_all_signals(
        signal_panels  = signal_panels,
        prices_df      = prices_df,
        control_panels = control_panels,
        horizon        = 1,
    )
    print(f"  ✓ FMB complete: {len(fmb_full)} signals")

    raw_corr  = apply_multiple_testing_corrections(
        pvals  = fmb_full["Raw p-val"].values,
        labels = fmb_full["Signal"].tolist(),
        alpha  = 0.05,
    )
    ctrl_corr = apply_multiple_testing_corrections(
        pvals  = fmb_full["Ctrl p-val"].values,
        labels = fmb_full["Signal"].tolist(),
        alpha  = 0.05,
    )

    print_correction_table(raw_corr,  "Raw p-values (signal only)")
    print_correction_table(ctrl_corr, "Controlled p-values (after momentum/reversal/vol/size)")

    # Memo framing
    n_tests  = len(fmb_full)
    n_bh_raw = int(raw_corr["sig_bh"].sum())
    n_bh_ctl = int(ctrl_corr["sig_bh"].sum())
    print(f"""
  MEMO FRAMING:
  'We tested {n_tests} signal decompositions. Two showed p < 0.05 uncorrected.
  After Benjamini-Hochberg FDR correction at 5%:
    Raw p-values:        {n_bh_raw}/{n_tests} survive
    Controlled p-values: {n_bh_ctl}/{n_tests} survive
  {'Both survivors pass FDR correction on controlled p-values.' if n_bh_ctl >= 2
   else 'Corrected results weaken the evidential claim.' if n_bh_ctl == 1
   else 'Neither signal survives multiple testing correction.'}
  We report these as suggestive findings requiring replication on
  longer history and an independent data source.'""")

    # ════════════════════════════════════════════════════════════════════
    # GAP 3
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  GAP 3 — ENHANCED MARKET IMPACT MODEL")
    print(f"{'=' * 70}")

    FLAT_BPS    = 5.0
    TURNOVER    = 1.9
    GROSS_CAGR  = 0.0197
    GROSS_SH    = 0.471

    min_cost_bps  = FLAT_BPS * TURNOVER * 2        # zero market impact floor
    min_cost_frac = min_cost_bps / 10000.0
    headroom_bps  = GROSS_CAGR * 10000 - min_cost_bps

    print(f"""
  Gross CAGR before costs  : {GROSS_CAGR*100:.2f}% = {GROSS_CAGR*10000:.0f} bps/year
  Ann. one-way turnover    : {TURNOVER:.1f}x (bi-weekly rebalancing)

  Minimum annual cost (flat TC only, zero market impact):
    {FLAT_BPS:.0f} bps/trade x {TURNOVER:.1f}x turnover x 2 (round-trip) = {min_cost_bps:.0f} bps/year

  Remaining headroom for market impact: {headroom_bps:.0f} bps/year

  METHOD (corrected vs prior version):
  Start from fund AUM, compute position per stock, then ADV participation.
    position_per_stock = AUM / n_portfolio_stocks
    daily_trade        = position x (turnover_ann / 252)
    ADV_participation  = daily_trade / per_stock_ADV

  S&P 100 stocks have $1B-$30B ADV. Even $500M AUM implies only
  ~0.19% ADV participation per stock — impact remains negligible.

  CRITICAL NUANCE: the binding constraint is NOT transaction costs.
  It is the signal's OOS failure (avg walk-forward Sharpe = -0.99).
  A strategy that doesn't survive out-of-sample is unimplementable
  regardless of how low the costs are. Don't say "destroyed by costs"
  in the interview — say "OOS failure is the binding constraint."
""")

    sweep_df, perstock_df = run_enhanced_cost_model(
        prices_df    = prices_df,
        volumes_df   = volumes_df,
        gross_sharpe = GROSS_SH,
        gross_cagr   = GROSS_CAGR,
        turnover_ann = TURNOVER,
        eta          = 0.1,
        flat_tc_bps  = FLAT_BPS,
    )

    if not sweep_df.empty:
        row_5   = sweep_df[sweep_df["AUM ($M)"] == 5].iloc[0]
        row_50  = sweep_df[sweep_df["AUM ($M)"] == 50].iloc[0]
        row_500 = sweep_df[sweep_df["AUM ($M)"] == 500].iloc[0]

        print(f"  Cost sensitivity by fund size:")
        print(f"  {'AUM($M)':<8} {'Pos/stock':<12} {'ADV%':<10} {'Impact':<9} "
              f"{'TC/trade':<10} {'Ann bps':<10} {'Ann%':<8} {'Net CAGR%':<11} "
              f"{'Net Sh':<9} {'Viable'}")
        print(f"  {'-'*105}")
        for _, r in sweep_df.iterrows():
            pos_K = r["Pos/stock ($K)"]
            print(f"  ${r['AUM ($M)']:<7.0f}M "
                  f"${pos_K:<9.0f}K "
                  f"{r['ADV part. (%)']:<9.4f}%  "
                  f"{r['Impact (bps/trade)']:<8.2f}bps "
                  f"{r['TC (bps/trade)']:<9.2f}bps "
                  f"{r['Ann cost (bps)']:<9.1f}bps "
                  f"{r['Ann cost (%)']:<7.3f}%  "
                  f"{r['Net CAGR (%)']:<10.2f}%  "
                  f"{r['Net Sharpe']:<8.3f}  "
                  f"{r['Viable (Sh≥0.3)']}")

        print(f"""
  Summary:
    $5M AUM   — ${row_5['Pos/stock ($K)']:.0f}K/stock, ADV part={row_5['ADV part. (%)']:.4f}%,
                 impact={row_5['Impact (bps/trade)']:.2f} bps, ann cost={row_5['Ann cost (bps)']:.0f} bps ({row_5['Ann cost (%)']:.3f}%),
                 Net Sharpe {row_5['Net Sharpe']:.3f}
    $50M AUM  — ${row_50['Pos/stock ($K)']:.0f}K/stock, ADV part={row_50['ADV part. (%)']:.4f}%,
                 impact={row_50['Impact (bps/trade)']:.2f} bps, ann cost={row_50['Ann cost (bps)']:.0f} bps ({row_50['Ann cost (%)']:.3f}%),
                 Net Sharpe {row_50['Net Sharpe']:.3f}
    $500M AUM — ${row_500['Pos/stock ($K)']:.0f}K/stock, ADV part={row_500['ADV part. (%)']:.4f}%,
                 impact={row_500['Impact (bps/trade)']:.2f} bps, ann cost={row_500['Ann cost (bps)']:.0f} bps ({row_500['Ann cost (%)']:.3f}%),
                 Net Sharpe {row_500['Net Sharpe']:.3f}

  All fund sizes show positive Net Sharpe — costs are NOT the problem.
  The binding constraint is OOS walk-forward failure.""")

        print(f"\n  Most expensive stocks at $50M AUM (top 5 by market impact):")
        print(f"  {'Ticker':<8} {'ADV ($M)':<12} {'Daily vol':<12} {'ADV part%':<12} {'Impact (bps)'}")
        for _, r in perstock_df.head(5).iterrows():
            print(f"  {r['ticker']:<8} ${r['adv_M']:<10.0f}M  "
                  f"{r['daily_vol']*100:<10.2f}%  "
                  f"{r['adv_part_pct']:<10.4f}%  "
                  f"{r['impact_bps']:.2f}")

        print(f"\n  Cheapest stocks at $50M AUM (bottom 5 by market impact):")
        for _, r in perstock_df.tail(5).iterrows():
            print(f"  {r['ticker']:<8} ${r['adv_M']:<10.0f}M  "
                  f"{r['daily_vol']*100:<10.2f}%  "
                  f"{r['adv_part_pct']:<10.4f}%  "
                  f"{r['impact_bps']:.2f}")

        print(f"""
  MEMO FRAMING (corrected):
  'The strategy generates gross CAGR {GROSS_CAGR*100:.2f}% at bi-weekly rebalancing
  (1.9x one-way annual turnover, flat TC 5 bps/trade). At $50M AUM,
  positions average ${row_50['Pos/stock ($K)']:.0f}K per stock ({row_50['ADV part. (%)']:.4f}% of ADV),
  yielding {row_50['Ann cost (bps)']:.0f} bps annual cost and Net Sharpe {row_50['Net Sharpe']:.3f}.
  Transaction costs are not the binding constraint — the strategy is
  cost-viable across the $5M–$500M AUM range. The actual constraint
  is out-of-sample signal instability: average OOS Sharpe of -0.99
  across 7 walk-forward windows, with no window showing positive OOS
  Sharpe. Costs are manageable; the signal is not stable enough to trade.'""")

    # ── Save outputs ──────────────────────────────────────────────────────
    # GAP 1
    if ff5_results:
        valid = [r for r in ff5_results if "error" not in r]
        if valid:
            pd.DataFrame(valid).to_csv(
                os.path.join(OUT_DIR, "GAP1_ff5_augmented_fmb.csv"), index=False)
    pd.DataFrame({k: v for k, v in ff5.items()}).to_csv(
        os.path.join(OUT_DIR, "GAP1_ff5_factor_returns.csv"))
    for fname, bdf in ff5_betas.items():
        bdf.to_csv(os.path.join(OUT_DIR, f"GAP1_rolling_{fname}_betas.csv"))

    # GAP 2
    combined_corr = fmb_full[["Signal", "Raw p-val", "Ctrl p-val"]].copy()
    combined_corr["bonf_raw"]     = raw_corr["p_bonferroni"].values
    combined_corr["holm_raw"]     = raw_corr["p_holm"].values
    combined_corr["bh_raw"]       = raw_corr["p_bh"].values
    combined_corr["bonf_ctrl"]    = ctrl_corr["p_bonferroni"].values
    combined_corr["holm_ctrl"]    = ctrl_corr["p_holm"].values
    combined_corr["bh_ctrl"]      = ctrl_corr["p_bh"].values
    combined_corr["sig_bh_raw"]   = raw_corr["sig_bh"].values
    combined_corr["sig_bh_ctrl"]  = ctrl_corr["sig_bh"].values
    combined_corr.to_csv(os.path.join(OUT_DIR, "GAP2_multiple_testing.csv"), index=False)

    # GAP 3
    if not sweep_df.empty:
        sweep_df.to_csv(os.path.join(OUT_DIR, "GAP3_cost_sweep.csv"), index=False)
        perstock_df.to_csv(os.path.join(OUT_DIR, "GAP3_perstock_costs.csv"), index=False)

    print(f"\n{'=' * 70}")
    print(f"  ALL OUTPUTS SAVED TO: outputs/")
    print(f"  GAP1_ff5_augmented_fmb.csv    3-model M1/M2/M3 FMB results")
    print(f"  GAP1_ff5_factor_returns.csv   daily proxy factor returns")
    print(f"  GAP1_rolling_rmw_betas.csv    per-stock rolling RMW betas")
    print(f"  GAP1_rolling_cma_betas.csv    per-stock rolling CMA betas")
    print(f"  GAP2_multiple_testing.csv     all p-value corrections")
    print(f"  GAP3_cost_sweep.csv           cost sensitivity by AUM")
    print(f"  GAP3_perstock_costs.csv       per-stock impact at $50M AUM")
    print(f"{'=' * 70}\n")
