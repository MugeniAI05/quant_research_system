"""
alt_data.py
===========
Alternative data module: Google Trends as a retail attention proxy.

Academic basis:
  Da, Engelberg & Gao (2011) "In Search of Attention" — Journal of Finance
  Shows Google Search Volume Index (SVI) predicts short-term return reversals
  and abnormal trading volume. High search interest = retail attention =
  attention-driven buying that subsequently mean-reverts.

Hypothesis tested here:
  Your vol_surprise_z signal captures abnormal volume. But abnormal volume
  has two possible causes with opposite return implications:

    (A) Retail attention spike → price overshoots → MEAN REVERSION
        Predicted by: high Google Trends + high volume surprise together

    (B) Informed institutional flow → price underreacts → CONTINUATION
        Predicted by: high volume surprise WITHOUT elevated Google Trends

  If (A): vol_surprise_z should be MORE negative when Google Trends is elevated
  If (B): vol_surprise_z should be MORE positive when Google Trends is LOW

  The interaction coefficient in Fama-MacBeth tells us which mechanism dominates.

Data:
  Google Trends via pytrends (free, no API key needed)
  Weekly frequency → aligned to Friday closes for weekly research

Install:
  pip install pytrends

Usage:
  from alt_data import load_trends_universe, compute_attention_signals
  trends_df = load_trends_universe(tickers, company_names)
  signals   = compute_attention_signals(trends_df, prices_df, volumes_df)
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
CACHE_PATH = os.path.join(OUT_DIR, "trends_cache.parquet")

# ── Company name map for Google search terms ──────────────────────────────
# Using company names rather than tickers gives cleaner search signal
# (people search "Apple stock" not "AAPL stock")
COMPANY_NAMES = {
    "AAPL": "Apple stock",     "MSFT": "Microsoft stock",
    "NVDA": "Nvidia stock",    "AMZN": "Amazon stock",
    "META": "Meta stock",      "GOOGL": "Google stock",
    "JPM":  "JPMorgan stock",  "TSLA": "Tesla stock",
    "V":    "Visa stock",      "XOM":  "Exxon stock",
    "AVGO": "Broadcom stock",  "MA":   "Mastercard stock",
    "JNJ":  "Johnson Johnson stock", "PG": "Procter Gamble stock",
    "COST": "Costco stock",    "HD":   "Home Depot stock",
    "MRK":  "Merck stock",     "ABBV": "AbbVie stock",
    "CVX":  "Chevron stock",   "KO":   "Coca Cola stock",
    "WMT":  "Walmart stock",   "PEP":  "Pepsi stock",
    "ADBE": "Adobe stock",     "CRM":  "Salesforce stock",
    "AMD":  "AMD stock",       "NFLX": "Netflix stock",
    "BAC":  "Bank America stock", "TMO": "Thermo Fisher stock",
    "MCD":  "McDonald stock",  "CSCO": "Cisco stock",
    "ABT":  "Abbott stock",    "WFC":  "Wells Fargo stock",
    "TXN":  "Texas Instruments stock", "ORCL": "Oracle stock",
    "DIS":  "Disney stock",    "QCOM": "Qualcomm stock",
    "AMGN": "Amgen stock",     "HON":  "Honeywell stock",
    "UNP":  "Union Pacific stock", "IBM": "IBM stock",
    "COP":  "ConocoPhillips stock", "CAT": "Caterpillar stock",
    "INTU": "Intuit stock",    "GE":   "GE stock",
    "LOW":  "Lowes stock",     "MS":   "Morgan Stanley stock",
    "GS":   "Goldman Sachs stock", "AXP": "American Express stock",
    "SBUX": "Starbucks stock", "AMAT": "Applied Materials stock",
    "GILD": "Gilead stock",    "REGN": "Regeneron stock",
    "C":    "Citigroup stock", "TJX":  "TJX stock",
    "GE":   "GE Aerospace stock", "FDX": "FedEx stock",
    "TGT":  "Target stock",    "TSLA": "Tesla stock",
}


# ════════════════════════════════════════════════════════════════════════════
# Google Trends fetching
# ════════════════════════════════════════════════════════════════════════════

def _fetch_trends_single(
    keyword: str,
    timeframe: str = "today 5-y",
    retries: int = 3,
    sleep: float = 1.5,
) -> Optional[pd.Series]:
    """
    Fetch weekly Google Trends for a single keyword.
    Returns normalized 0-100 interest series or None on failure.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise ImportError(
            "pytrends not installed. Run: pip install pytrends"
        )

    for attempt in range(retries):
        try:
            pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2)
            pt.build_payload([keyword], timeframe=timeframe, geo="US")
            df = pt.interest_over_time()

            if df is None or df.empty:
                return None

            series = df[keyword].astype(float)
            # Remove "isPartial" column artifact
            series = series[series.index.dayofweek == 6]  # Sundays (Google week end)
            series.index = pd.to_datetime(series.index)
            return series

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(sleep * (attempt + 1))
            else:
                logger.warning(f"Trends fetch failed for '{keyword}': {e}")
                return None

    return None


def fetch_trends_batch(
    tickers: List[str],
    company_names: Dict[str, str] = None,
    timeframe: str = "today 5-y",
    max_tickers: int = 50,
    sleep_between: float = 1.5,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch Google Trends for a list of tickers.

    Rate limits: Google Trends allows ~1 request/second.
    Caches results to parquet to avoid re-fetching.

    Returns DataFrame (weekly dates x tickers) with normalized 0-100 values.
    Missing = could not fetch (treat as NaN, exclude from that week's analysis).
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    if company_names is None:
        company_names = COMPANY_NAMES

    # Load cache
    if not force_refresh and os.path.exists(CACHE_PATH):
        print("  Loading trends from cache...")
        cached = pd.read_parquet(CACHE_PATH)
        # Check if cached tickers cover what we need
        cached_tickers = set(cached.columns)
        needed = set(tickers[:max_tickers]) - cached_tickers
        if not needed:
            print(f"  ✓ {len(cached.columns)} tickers in cache")
            return cached
        print(f"  Cache has {len(cached_tickers)} tickers, fetching {len(needed)} more...")
        tickers_to_fetch = list(needed)
    else:
        tickers_to_fetch = tickers[:max_tickers]
        cached = pd.DataFrame()

    print(f"  Fetching Google Trends for {len(tickers_to_fetch)} tickers...")
    print(f"  (Rate limited to ~1.5s between requests — takes ~{len(tickers_to_fetch)*2//60+1} min)")

    results = {}
    failed  = []

    for i, tkr in enumerate(tickers_to_fetch):
        keyword = company_names.get(tkr, f"{tkr} stock")
        series  = _fetch_trends_single(keyword, timeframe=timeframe)

        if series is not None and len(series) > 52:
            results[tkr] = series
        else:
            failed.append(tkr)

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(tickers_to_fetch)} ({len(results)} OK)")

        time.sleep(sleep_between)

    if not results:
        raise RuntimeError(
            "No Google Trends data fetched. Check internet connection "
            "or try: pip install --upgrade pytrends"
        )

    new_df = pd.DataFrame(results)
    new_df.index = pd.to_datetime(new_df.index)

    # Merge with cache
    if not cached.empty:
        trends_df = pd.concat([cached, new_df], axis=1)
        trends_df = trends_df.loc[:, ~trends_df.columns.duplicated()]
    else:
        trends_df = new_df

    trends_df.to_parquet(CACHE_PATH)
    print(f"  ✓ {len(results)} tickers fetched, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed[:10]}")

    return trends_df


# ════════════════════════════════════════════════════════════════════════════
# Signal construction from Trends data
# ════════════════════════════════════════════════════════════════════════════

def compute_attention_signals(
    trends_df:  pd.DataFrame,
    prices_df:  pd.DataFrame,
    volumes_df: pd.DataFrame,
    z_window_weeks: int = 8,
) -> Dict[str, pd.DataFrame]:
    """
    Construct attention-based signals from Google Trends data.

    All signals are at WEEKLY frequency (Friday close aligned).
    Forward returns computed at weekly horizon.

    Signals:
      attention_surprise  : this week's SVI vs 8-week rolling mean (z-scored)
                           Positive = elevated retail attention
      attention_momentum  : 4-week change in SVI (is attention growing?)
      vol_attn_interaction: vol_surprise * attention_surprise
                           The KEY signal: high vol + high attention = retail driven
                           High vol + LOW attention = potentially institutional

    Returns dict of signal_name -> weekly (dates x tickers) DataFrame
    """
    # ── Align trends to Friday prices ──────────────────────────────────
    # Resample prices to weekly (Friday close)
    weekly_px  = prices_df.resample("W-FRI").last()
    weekly_vol = volumes_df.resample("W-FRI").sum()

    # Align trends index to Friday (Google uses Sunday week-end, shift to Friday)
    trends_aligned = trends_df.copy()
    trends_aligned.index = trends_aligned.index - pd.Timedelta(days=2)
    trends_aligned = trends_aligned.resample("W-FRI").last()

    # Common tickers and dates
    common_tickers = weekly_px.columns.intersection(trends_aligned.columns)
    common_dates   = weekly_px.index.intersection(trends_aligned.index)

    if len(common_tickers) == 0:
        raise ValueError("No common tickers between prices and trends data")
    if len(common_dates) < 52:
        raise ValueError(f"Only {len(common_dates)} common weekly dates — need at least 52")

    print(f"  ✓ Alt data aligned: {len(common_tickers)} tickers x {len(common_dates)} weeks")
    print(f"    Date range: {common_dates[0].date()} → {common_dates[-1].date()}")

    wp  = weekly_px.loc[common_dates, common_tickers]
    wv  = weekly_vol.loc[common_dates, common_tickers]
    tr  = trends_aligned.loc[common_dates, common_tickers].astype(float)

    signals = {}

    # ── Signal 1: Attention surprise ─────────────────────────────────────
    attn_roll_mean = tr.rolling(z_window_weeks).mean()
    attn_roll_std  = tr.rolling(z_window_weeks).std()
    attn_surprise  = (tr - attn_roll_mean) / attn_roll_std.replace(0, np.nan)
    signals["attention_surprise"] = attn_surprise

    # ── Signal 2: Attention momentum (4-week change) ─────────────────────
    attn_mom = tr.diff(4) / tr.shift(4).replace(0, np.nan)
    attn_mom_z = (attn_mom - attn_mom.rolling(z_window_weeks).mean()) / \
                  attn_mom.rolling(z_window_weeks).std().replace(0, np.nan)
    signals["attention_momentum"] = attn_mom_z

    # ── Signal 3: Volume surprise (weekly) ───────────────────────────────
    avg_wv   = wv.rolling(z_window_weeks).mean()
    vol_surp = (wv - avg_wv) / avg_wv.replace(0, np.nan)
    vol_surp_z = (vol_surp - vol_surp.rolling(z_window_weeks).mean()) / \
                  vol_surp.rolling(z_window_weeks).std().replace(0, np.nan)
    signals["weekly_vol_surprise"] = vol_surp_z

    # ── Signal 4: THE KEY INTERACTION ────────────────────────────────────
    # vol_surprise * attention_surprise
    # High value = both vol and attention elevated = likely retail-driven
    # Hypothesis: this predicts REVERSAL (negative return next week)
    # Low/negative value = vol elevated but attention NOT = institutional
    # Hypothesis: this predicts CONTINUATION (positive return next week)
    interaction = vol_surp_z * attn_surprise
    signals["vol_attn_interaction"] = interaction

    # ── Signal 5: Residual volume (vol surprise unexplained by attention) ─
    # Regress vol_surp on attn_surprise cross-sectionally each week,
    # take residuals = volume that is NOT explained by retail attention
    # This isolates the institutional component of abnormal volume
    resid_dict = {}
    for dt in common_dates:
        vs = vol_surp_z.loc[dt].dropna()
        at = attn_surprise.loc[dt].dropna()
        common_t = vs.index.intersection(at.index)
        if len(common_t) < 10:
            resid_dict[dt] = pd.Series(np.nan, index=common_tickers)
            continue
        try:
            import statsmodels.api as sm
            X = sm.add_constant(at[common_t].values)
            y = vs[common_t].values
            fit = sm.OLS(y, X).fit()
            resids = pd.Series(fit.resid, index=common_t)
        except Exception:
            # Fallback: simple demeaned residual
            b = np.cov(vs[common_t], at[common_t])[0,1] / max(np.var(at[common_t]), 1e-9)
            resids = vs[common_t] - b * at[common_t]
        resid_dict[dt] = resids.reindex(common_tickers)

    signals["institutional_vol_proxy"] = pd.DataFrame(resid_dict).T.reindex(common_dates)

    return signals, common_dates, common_tickers


# ════════════════════════════════════════════════════════════════════════════
# Fama-MacBeth at weekly frequency with alt data
# ════════════════════════════════════════════════════════════════════════════

def run_weekly_fmb(
    signals:       Dict[str, pd.DataFrame],
    prices_df:     pd.DataFrame,
    common_dates:  pd.DatetimeIndex,
    common_tickers: pd.Index,
    horizon_weeks:  int = 1,
) -> pd.DataFrame:
    """
    Run weekly Fama-MacBeth regression for all attention signals.

    Tests three questions:
      1. Does attention_surprise predict returns? (Da et al. 2011 replication)
      2. Does vol_attn_interaction predict reversals?
      3. Does institutional_vol_proxy predict continuation?
    """
    from scipy import stats as scipy_stats

    weekly_px  = prices_df.resample("W-FRI").last()
    fwd_rets   = weekly_px.pct_change(horizon_weeks).shift(-horizon_weeks)
    fwd_rets   = fwd_rets.loc[common_dates, common_tickers]

    rows = []

    for sig_name, sig_df in signals.items():
        sig_df = sig_df.reindex(index=common_dates, columns=common_tickers)
        daily_betas = []
        valid_dates = []
        n_stocks_per_week = []

        for dt in common_dates[:-horizon_weeks]:
            s = sig_df.loc[dt].dropna()
            r = fwd_rets.loc[dt].dropna()
            common_t = s.index.intersection(r.index)

            if len(common_t) < 10:
                continue

            s_c = s[common_t]
            r_c = r[common_t]

            # Winsorize
            for ser in [s_c, r_c]:
                lo, hi = ser.quantile(0.05), ser.quantile(0.95)
                ser.clip(lo, hi, inplace=True)

            # Standardize signal
            if s_c.std() > 0:
                s_c = (s_c - s_c.mean()) / s_c.std()

            try:
                slope, intercept, r_val, p_val, se = scipy_stats.linregress(s_c, r_c)
                daily_betas.append(slope)
                valid_dates.append(dt)
                n_stocks_per_week.append(len(common_t))
            except Exception:
                continue

        if len(daily_betas) < 20:
            continue

        beta_series = pd.Series(daily_betas, index=valid_dates)
        mean_beta   = float(beta_series.mean())
        n           = len(beta_series)

        # HAC t-stat
        try:
            import statsmodels.api as sm
            from statsmodels.stats.sandwich_covariance import cov_hac
            y   = beta_series.values
            X   = sm.add_constant(np.ones(n))
            mod = sm.OLS(y, X).fit()
            V   = cov_hac(mod, nlags=4)  # 4 weeks lag for weekly data
            se  = float(np.sqrt(V[0, 0]))
            t   = mean_beta / se if se > 0 else 0.0
            p   = float(2 * scipy_stats.t.sf(abs(t), df=max(n-1, 1)))
        except Exception:
            se = float(beta_series.std() / np.sqrt(n))
            t  = mean_beta / se if se > 0 else 0.0
            p  = float(2 * scipy_stats.t.sf(abs(t), df=max(n-1, 1)))

        rows.append({
            "Signal":        sig_name,
            "N weeks":       n,
            "Avg N stocks":  round(float(np.mean(n_stocks_per_week)), 1),
            "Mean beta":     round(mean_beta, 6),
            "HAC t-stat":    round(t, 3),
            "p-value":       round(p, 4),
            "Significant":   "YES" if p < 0.05 else "no",
            "Direction":     "reversal" if mean_beta < 0 else "continuation",
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Mechanism test: does attention moderate the vol_surprise effect?
# ════════════════════════════════════════════════════════════════════════════

def test_attention_mechanism(
    signals:        Dict[str, pd.DataFrame],
    prices_df:      pd.DataFrame,
    common_dates:   pd.DatetimeIndex,
    common_tickers: pd.Index,
) -> Dict:
    """
    The core mechanism test.

    Split weeks into HIGH vs LOW attention regimes (median attention_surprise).
    For each regime, compute IC of weekly_vol_surprise on next-week returns.

    Prediction:
      HIGH attention IC < LOW attention IC
      (vol surprise in high-attention weeks = retail-driven = mean-reverts more)

    This directly tests the Da et al. (2011) mechanism.
    """
    from scipy import stats as scipy_stats

    weekly_px = prices_df.resample("W-FRI").last()
    fwd_rets  = weekly_px.pct_change(1).shift(-1)
    fwd_rets  = fwd_rets.reindex(index=common_dates, columns=common_tickers)

    attn = signals["attention_surprise"].reindex(index=common_dates, columns=common_tickers)
    vs   = signals["weekly_vol_surprise"].reindex(index=common_dates, columns=common_tickers)

    # Median attention per week (cross-sectional)
    median_attn = attn.median(axis=1)
    high_attn_dates = median_attn[median_attn >= median_attn.median()].index
    low_attn_dates  = median_attn[median_attn <  median_attn.median()].index

    def compute_ic(dates):
        ics = []
        for dt in dates:
            if dt not in fwd_rets.index:
                continue
            s = vs.loc[dt].dropna()
            r = fwd_rets.loc[dt].dropna()
            common_t = s.index.intersection(r.index)
            if len(common_t) < 10:
                continue
            ic, _ = scipy_stats.spearmanr(s[common_t], r[common_t])
            if not np.isnan(ic):
                ics.append(ic)
        return np.array(ics)

    high_ics = compute_ic(high_attn_dates)
    low_ics  = compute_ic(low_attn_dates)

    if len(high_ics) < 5 or len(low_ics) < 5:
        return {"error": "Insufficient data for mechanism test"}

    t_stat, p_val = scipy_stats.ttest_ind(high_ics, low_ics)

    return {
        "high_attn_mean_ic":   round(float(np.mean(high_ics)), 4),
        "high_attn_n_weeks":   len(high_ics),
        "low_attn_mean_ic":    round(float(np.mean(low_ics)), 4),
        "low_attn_n_weeks":    len(low_ics),
        "t_stat":              round(float(t_stat), 3),
        "p_value":             round(float(p_val), 4),
        "hypothesis_supported": (
            p_val < 0.10 and
            float(np.mean(high_ics)) < float(np.mean(low_ics))
        ),
    }


if __name__ == "__main__":
    print("\nalt_data.py — run via run_v3_altdata.py, not directly")
    print("Install pytrends first: pip install pytrends")
