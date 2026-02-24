"""
hypothesis_testing.py
=====================
Tests economic hypotheses about WHY OBV predicts short-term returns.

The central hypothesis: OBV signals in low-volatility large-cap stocks
reflect institutional accumulation/distribution patterns that mean-revert
over 1-5 days as liquidity absorbs the flow.

Two testable sub-hypotheses:
  H1 (Institutional Flow): OBV signal should be stronger in stocks with
     higher institutional ownership (proxied by market cap — larger stocks
     have more institutional activity). If true: IC should be more negative
     for large-cap stocks than small-cap stocks.

  H2 (Liquidity Mean-Reversion): Signal should work better in liquid stocks
     (high average volume) where institutions can accumulate quietly without
     moving price, leading to cleaner mean-reversion. If true: IC should
     correlate negatively with bid-ask spread proxy (volume-to-price ratio).

  H3 (Regime Dependence): If the signal captures institutional flow,
     it should be stronger in low-VIX (calm) regimes where institutions
     are more active, and weaker in high-VIX periods where flow is dominated
     by retail panic. Tests this against the realized-vol regime split.

Run:
    cd src
    python hypothesis_testing.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats
import logging
logging.basicConfig(level=logging.WARNING)

from universe import load_universe

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON = 1   # 1-day forward return (optimal from holding period sweep)


# ── Helpers ───────────────────────────────────────────────────────────────────

def obv_1d_z60(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    """Stationary OBV: 1-day change, z-scored over 60-day rolling window."""
    direction = np.sign(prices.pct_change()).fillna(0.0)
    obv_raw   = (direction * volumes.fillna(0)).cumsum()
    chg       = obv_raw.diff(1)
    mu        = chg.rolling(60).mean()
    sigma     = chg.rolling(60).std()
    return (chg - mu) / sigma.replace(0, np.nan)


def spearman_ic(factor: pd.Series, prices: pd.Series, horizon: int = 1):
    """Spearman IC with HAC t-stat (Newey-West, nlags=horizon-1)."""
    fwd = prices.pct_change(horizon).shift(-horizon)
    df  = pd.DataFrame({"f": factor, "r": fwd}).dropna()
    if len(df) < 60:
        return np.nan, np.nan, np.nan, len(df)

    ic, _ = stats.spearmanr(df["f"], df["r"])
    n      = len(df)

    # HAC t-stat via Newey-West if statsmodels available
    try:
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import cov_hac

        y    = df["r"].values
        x    = df["f"].rank().values
        X    = sm.add_constant(x)
        mod  = sm.OLS(y, X).fit()
        nlags = max(horizon - 1, 1)
        V    = cov_hac(mod, nlags=nlags)
        se   = float(np.sqrt(V[1, 1]))
        t    = float(mod.params[1]) / se if se > 0 else 0.0
        pval = float(2 * stats.t.sf(abs(t), df=max(n - 2, 1)))
    except Exception:
        # Fallback to naive t-stat with Newey-West approximation
        t    = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic**2, 1e-9))
        pval = float(2 * stats.t.sf(abs(t), df=max(n - 2, 1)))

    return round(ic, 4), round(t, 3), round(pval, 4), n


def stock_characteristics(prices_df: pd.DataFrame, volumes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute characteristics used to test hypotheses.
    For each stock:
      - avg_price:      proxy for stock price level
      - avg_volume:     proxy for liquidity
      - market_cap_proxy: price * volume (rough size proxy)
      - avg_vol_20d:    average realized volatility
      - vol_rank:       percentile rank by avg_volume (liquidity rank)
      - size_rank:      percentile rank by market_cap_proxy
    """
    chars = {}
    for tkr in prices_df.columns:
        px  = prices_df[tkr].dropna()
        vol = volumes_df[tkr].dropna()
        rets = px.pct_change().dropna()

        chars[tkr] = {
            "avg_price":         float(px.mean()),
            "avg_volume":        float(vol.mean()),
            "market_cap_proxy":  float((px * vol).mean()),
            "avg_vol_20d":       float(rets.rolling(20).std().mean() * np.sqrt(252)),
            "price_last":        float(px.iloc[-1]),
        }

    df = pd.DataFrame(chars).T
    df["vol_rank"]  = df["avg_volume"].rank(pct=True)
    df["size_rank"] = df["market_cap_proxy"].rank(pct=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
# Main hypothesis tests
# ════════════════════════════════════════════════════════════════════════════

def test_h1_size_effect(
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    chars: pd.DataFrame,
) -> pd.DataFrame:
    """
    H1: OBV IC should be stronger (more negative) in larger stocks.
    Mechanism: Larger stocks have more institutional activity; institutional
    accumulation/distribution creates cleaner OBV patterns that mean-revert.

    Test: Compute per-stock IC, then regress IC on size_rank.
    Prediction: Negative slope (larger stocks → more negative IC).
    """
    rows = []
    for tkr in prices_df.columns:
        px  = prices_df[tkr]
        vol = volumes_df[tkr]
        sig = obv_1d_z60(px, vol)
        ic, t, p, n = spearman_ic(sig, px, HORIZON)
        if not np.isnan(ic):
            rows.append({"ticker": tkr, "ic": ic, "t_stat": t, "p_val": p, "n_obs": n})

    ic_df = pd.DataFrame(rows).set_index("ticker")
    ic_df = ic_df.join(chars[["size_rank", "vol_rank", "avg_vol_20d"]])

    # Test: does IC correlate with size?
    clean = ic_df[["ic", "size_rank"]].dropna()
    slope, intercept, r, p_slope, _ = stats.linregress(clean["size_rank"], clean["ic"])

    print(f"\n  H1 — Size Effect:")
    print(f"    Slope of IC on size_rank : {slope:+.4f}")
    print(f"    R-squared                : {r**2:.4f}")
    print(f"    p-value                  : {p_slope:.4f}")
    if p_slope < 0.10 and slope < 0:
        print(f"    Result: SUPPORTED — larger stocks show more negative IC (institutional flow)")
    elif p_slope < 0.10 and slope > 0:
        print(f"    Result: REJECTED — larger stocks show LESS negative IC")
    else:
        print(f"    Result: NOT SIGNIFICANT — no clear size effect (p={p_slope:.3f})")

    return ic_df.reset_index()


def test_h2_liquidity_effect(ic_df: pd.DataFrame) -> None:
    """
    H2: Signal should be stronger in liquid stocks (high avg_volume).
    Mechanism: Liquid stocks absorb institutional flow more quietly,
    creating cleaner price/volume divergences that mean-revert.

    Test: Regress IC on vol_rank.
    """
    clean = ic_df[["ic", "vol_rank"]].dropna()
    slope, intercept, r, p_slope, _ = stats.linregress(clean["vol_rank"], clean["ic"])

    print(f"\n  H2 — Liquidity Effect:")
    print(f"    Slope of IC on vol_rank  : {slope:+.4f}")
    print(f"    R-squared                : {r**2:.4f}")
    print(f"    p-value                  : {p_slope:.4f}")
    if p_slope < 0.10 and slope < 0:
        print(f"    Result: SUPPORTED — more liquid stocks show stronger mean-reversion")
    elif p_slope < 0.10 and slope > 0:
        print(f"    Result: REJECTED — less liquid stocks show stronger signal")
    else:
        print(f"    Result: NOT SIGNIFICANT (p={p_slope:.3f})")


def test_h3_regime_dependence(
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    ic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    H3: Signal should be stronger in low-volatility regimes.
    Mechanism: In calm markets, institutional flow is more systematic
    and detectable via OBV. In high-vol periods, flow is dominated
    by reactive trading that does not mean-revert cleanly.

    Test: For each stock, compute IC in low-vol vs high-vol days
    and test whether the difference is systematic across the universe.
    """
    low_ics  = []
    high_ics = []

    for tkr in prices_df.columns:
        px  = prices_df[tkr]
        vol = volumes_df[tkr]
        sig = obv_1d_z60(px, vol)

        rv      = px.pct_change().rolling(20).std() * np.sqrt(252)
        median  = float(rv.median())
        low_idx  = rv[rv <= median].index
        high_idx = rv[rv >  median].index

        for idx, store in [(low_idx, low_ics), (high_idx, high_ics)]:
            common = px.index.intersection(idx)
            if len(common) < 60:
                store.append(np.nan)
                continue
            ic, _, _, _ = spearman_ic(sig.reindex(common), px.reindex(common), HORIZON)
            store.append(ic)

    low_arr  = np.array(low_ics,  dtype=float)
    high_arr = np.array(high_ics, dtype=float)

    mask = ~(np.isnan(low_arr) | np.isnan(high_arr))
    low_clean  = low_arr[mask]
    high_clean = high_arr[mask]

    t_stat, p_val = stats.ttest_rel(low_clean, high_clean)

    print(f"\n  H3 — Regime Dependence (Low Vol vs High Vol):")
    print(f"    Avg IC in low-vol  regime : {np.nanmean(low_arr):+.4f}")
    print(f"    Avg IC in high-vol regime : {np.nanmean(high_arr):+.4f}")
    print(f"    Paired t-test t-stat      : {t_stat:+.3f}")
    print(f"    p-value                   : {p_val:.4f}")

    if p_val < 0.10 and np.nanmean(low_arr) < np.nanmean(high_arr):
        print(f"    Result: SUPPORTED — IC is significantly more negative in low-vol regimes")
    elif p_val < 0.10:
        print(f"    Result: SIGNIFICANT but in unexpected direction")
    else:
        print(f"    Result: NOT SIGNIFICANT — regime effect is not consistent across universe (p={p_val:.3f})")

    regime_df = pd.DataFrame({
        "ticker":   prices_df.columns,
        "low_vol_ic":  low_ics,
        "high_vol_ic": high_ics,
    })
    return regime_df


# ── Runner ────────────────────────────────────────────────────────────────────

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone and normalize to midnight DatetimeIndex."""
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="first")]
    return df


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  HYPOTHESIS TESTING — ECONOMIC RATIONALE FOR OBV SIGNAL")
    print("="*65)

    print("\n  Central hypothesis:")
    print("  OBV predicts short-term returns because it detects institutional")
    print("  accumulation/distribution in liquid large-cap stocks, which")
    print("  mean-reverts as market liquidity absorbs the directional flow.\n")

    print("  Loading universe...")
    prices_df, volumes_df = load_universe()
    prices_df  = _normalize_index(prices_df)
    volumes_df = _normalize_index(volumes_df)
    chars = stock_characteristics(prices_df, volumes_df)
    print(f"  ✓ {prices_df.shape[1]} stocks, {prices_df.shape[0]} days")

    print("\n" + "-"*65)
    print("  Running hypothesis tests...")
    print("-"*65)

    ic_df    = test_h1_size_effect(prices_df, volumes_df, chars)
    test_h2_liquidity_effect(ic_df)
    regime_df = test_h3_regime_dependence(prices_df, volumes_df, ic_df)

    print("\n" + "="*65)
    print("  CROSS-SECTIONAL IC DISTRIBUTION (all stocks)")
    print("="*65)
    ic_vals = ic_df["ic"].dropna()
    print(f"  N stocks with IC computed : {len(ic_vals)}")
    print(f"  Mean IC across universe   : {ic_vals.mean():+.4f}")
    print(f"  Median IC                 : {ic_vals.median():+.4f}")
    print(f"  Std IC                    : {ic_vals.std():.4f}")
    print(f"  % stocks with negative IC : {(ic_vals < 0).mean()*100:.1f}%")
    print(f"  % stocks significant      : {(ic_df['p_val'].dropna() < 0.05).mean()*100:.1f}%")

    t_vs_zero, p_vs_zero = stats.ttest_1samp(ic_vals.dropna(), 0)
    print(f"\n  t-test: is mean IC different from zero?")
    print(f"    t = {t_vs_zero:+.3f},  p = {p_vs_zero:.4f}")
    if p_vs_zero < 0.05:
        print(f"    YES — OBV signal has systematic predictive content across the universe")
    else:
        print(f"    NO — OBV signal is not systematically different from zero across stocks")

    # Save
    ic_df.to_csv(os.path.join(OUT_DIR, "hypothesis_ic_by_stock.csv"), index=False)
    regime_df.to_csv(os.path.join(OUT_DIR, "hypothesis_regime.csv"), index=False)
    print(f"\n  Results saved to outputs/")
    print("="*65 + "\n")
