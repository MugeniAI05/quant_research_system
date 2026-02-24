"""
run_altdata_research.py
=======================
Alternative data research suite — Google Trends as retail attention proxy.

Integrates with the existing v3 research to test:

  CORE QUESTION:
  Your vol_surprise_z signal (abnormal volume) predicts returns in
  Fama-MacBeth. But WHY? Two competing mechanisms:

    (A) Retail attention → overreaction → mean REVERSION
        Evidence: Da, Engelberg & Gao (2011) Journal of Finance
        Test: vol_surprise stronger predictor in HIGH Google Trends weeks

    (B) Institutional flow → underreaction → CONTINUATION
        Evidence: Llorente et al. (2002), Gervais et al. (2001)
        Test: vol_surprise stronger predictor in LOW Google Trends weeks

  The interaction between volume surprise and attention surprise is the
  key signal. If attention moderates the vol_surprise effect, the retail
  attention mechanism is supported.

  Additionally tests:
    - Attention surprise alone (Da et al. 2011 replication)
    - Institutional vol proxy (volume unexplained by attention)
    - Attention momentum (is search interest growing?)

Run:
    cd src
    pip install pytrends
    python run_altdata_research.py

Note: Google Trends is weekly, price data is daily.
All alt-data analysis is at WEEKLY frequency.
This is a genuine methodological limitation documented in the memo.

Outputs saved to: ../outputs/ALT_*.csv
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from universe  import load_universe
from alt_data  import (
    fetch_trends_batch,
    compute_attention_signals,
    run_weekly_fmb,
    test_attention_mechanism,
    COMPANY_NAMES,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────
MAX_TICKERS   = 40    # keep to 40 to avoid Google rate limits
TRENDS_PERIOD = "today 5-y"
Z_WINDOW_WKS  = 8     # 8-week rolling z-score for attention surprise


def _normalize_index(df):
    idx = pd.to_datetime(df.index)
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="first")]


def print_mechanism_result(mech: dict) -> None:
    """Pretty print the mechanism test result."""
    if "error" in mech:
        print(f"  Error: {mech['error']}")
        return

    print(f"  High-attention weeks: mean IC = {mech['high_attn_mean_ic']:+.4f} "
          f"(n={mech['high_attn_n_weeks']} weeks)")
    print(f"  Low-attention weeks:  mean IC = {mech['low_attn_mean_ic']:+.4f} "
          f"(n={mech['low_attn_n_weeks']} weeks)")
    print(f"  t-test (difference):  t = {mech['t_stat']:+.3f}, p = {mech['p_value']:.4f}")

    if mech["hypothesis_supported"]:
        print(f"\n  >>> MECHANISM SUPPORTED (p={mech['p_value']:.3f})")
        print(f"  >>> vol_surprise is weaker when retail attention is high")
        print(f"  >>> Consistent with retail-driven mean reversion story (Da et al. 2011)")
    elif mech["p_value"] < 0.10 and mech["high_attn_mean_ic"] > mech["low_attn_mean_ic"]:
        print(f"\n  >>> OPPOSITE DIRECTION (p={mech['p_value']:.3f})")
        print(f"  >>> vol_surprise is STRONGER in high-attention weeks")
        print(f"  >>> Inconsistent with retail story — may support institutional flow")
    else:
        print(f"\n  >>> NOT SIGNIFICANT (p={mech['p_value']:.3f})")
        print(f"  >>> Cannot distinguish retail from institutional mechanism")
        print(f"  >>> Attention does not moderate the volume-return relationship")


if __name__ == "__main__":

    print("\n" + "="*70)
    print("  ALTERNATIVE DATA RESEARCH — GOOGLE TRENDS ATTENTION PROXY")
    print("="*70)
    print("  Academic basis: Da, Engelberg & Gao (2011) 'In Search of Attention'")
    print("  Question: Is abnormal volume driven by retail attention or institutional flow?")
    print("  Frequency: WEEKLY (Google Trends limitation — noted as honest caveat)")

    # ── Load universe ────────────────────────────────────────────────────
    print(f"\n[0] Loading universe...")
    prices_df, volumes_df = load_universe()
    prices_df  = _normalize_index(prices_df)
    volumes_df = _normalize_index(volumes_df)

    # Limit to tickers with Google Trends mappings
    available  = [t for t in prices_df.columns if t in COMPANY_NAMES][:MAX_TICKERS]
    prices_df  = prices_df[available]
    volumes_df = volumes_df[available]
    print(f"  ✓ Using {len(available)} tickers with Google Trends coverage")

    # ── Fetch Google Trends ──────────────────────────────────────────────
    print(f"\n[1/4] Fetching Google Trends (retail attention proxy)...")
    print(f"  Fetching {len(available)} tickers — takes ~{len(available)*2//60+1} minutes")
    print(f"  Rate limited to avoid Google blocking. Cache saved after first run.\n")

    try:
        trends_df = fetch_trends_batch(
            tickers       = available,
            company_names = COMPANY_NAMES,
            timeframe     = TRENDS_PERIOD,
            max_tickers   = MAX_TICKERS,
            force_refresh = False,
        )
        print(f"  ✓ Trends data: {trends_df.shape[1]} tickers x {trends_df.shape[0]} weeks")

    except ImportError:
        print("\n  ERROR: pytrends not installed.")
        print("  Run: pip install pytrends")
        print("  Then re-run this script.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR fetching trends: {e}")
        print("  Google may be rate-limiting. Wait 5 minutes and retry.")
        sys.exit(1)

    # ── Compute attention signals ────────────────────────────────────────
    print(f"\n[2/4] Computing attention signals...")
    try:
        signals, common_dates, common_tickers = compute_attention_signals(
            trends_df  = trends_df,
            prices_df  = prices_df,
            volumes_df = volumes_df,
            z_window_weeks = Z_WINDOW_WKS,
        )
    except Exception as e:
        print(f"  ERROR computing signals: {e}")
        sys.exit(1)

    print(f"  ✓ Signals computed: {list(signals.keys())}")
    print(f"\n  Signal descriptions:")
    print(f"  attention_surprise    — SVI this week vs 8-week mean (z-scored)")
    print(f"  attention_momentum    — 4-week change in SVI")
    print(f"  weekly_vol_surprise   — abnormal volume vs 8-week avg (weekly)")
    print(f"  vol_attn_interaction  — vol_surprise × attention_surprise")
    print(f"                          High = retail-driven, Low = institutional")
    print(f"  institutional_vol_proxy — vol surprise unexplained by attention")

    # ── Weekly Fama-MacBeth on all attention signals ─────────────────────
    print(f"\n[3/4] Weekly Fama-MacBeth Regression...")
    print(f"  Tests whether each attention signal predicts next-week returns")
    print(f"  HAC standard errors with 4-week Newey-West lag\n")

    fmb_df = run_weekly_fmb(
        signals        = signals,
        prices_df      = prices_df,
        common_dates   = common_dates,
        common_tickers = common_tickers,
        horizon_weeks  = 1,
    )

    if not fmb_df.empty:
        print(f"\n{'='*70}")
        print(f"  WEEKLY FAMA-MACBETH RESULTS")
        print(f"{'='*70}")
        print(fmb_df.to_string(index=False))

        sig_signals = fmb_df[fmb_df["Significant"] == "YES"]
        print(f"\n  Significant signals: {len(sig_signals)}/{len(fmb_df)}")

        if len(sig_signals) > 0:
            for _, row in sig_signals.iterrows():
                direction_str = (
                    "HIGH attention → return REVERSAL (retail overreaction)"
                    if row["Direction"] == "reversal" and "attn" in row["Signal"]
                    else "HIGH volume surprise → positive returns"
                    if row["Direction"] == "continuation"
                    else f"{row['Direction']}"
                )
                print(f"  {row['Signal']}: t={row['HAC t-stat']:.3f}, p={row['p-value']:.4f}")
                print(f"    Interpretation: {direction_str}")

        # Key results interpretation
        if "vol_attn_interaction" in fmb_df["Signal"].values:
            row = fmb_df[fmb_df["Signal"] == "vol_attn_interaction"].iloc[0]
            print(f"\n  KEY INTERACTION RESULT:")
            print(f"  vol_attn_interaction: beta={row['Mean beta']:.6f}, "
                  f"t={row['HAC t-stat']:.3f}, p={row['p-value']:.4f}")
            if row["Significant"] == "YES" and row["Direction"] == "reversal":
                print(f"  >>> SUPPORTED: High vol + high attention → reversal")
                print(f"  >>> Retail attention drives abnormal volume → mean reversion")
                print(f"  >>> Consistent with Da, Engelberg & Gao (2011)")
            elif row["Significant"] == "YES" and row["Direction"] == "continuation":
                print(f"  >>> OPPOSITE: High vol + high attention → continuation")
                print(f"  >>> May reflect attention-chasing momentum (not Da et al.)")
            else:
                print(f"  >>> NOT SIGNIFICANT: Cannot distinguish mechanisms from data")

        if "institutional_vol_proxy" in fmb_df["Signal"].values:
            row = fmb_df[fmb_df["Signal"] == "institutional_vol_proxy"].iloc[0]
            print(f"\n  INSTITUTIONAL PROXY RESULT:")
            print(f"  institutional_vol_proxy: t={row['HAC t-stat']:.3f}, p={row['p-value']:.4f}")
            if row["Significant"] == "YES" and row["Direction"] == "continuation":
                print(f"  >>> Volume unexplained by retail attention predicts continuation")
                print(f"  >>> Supports institutional flow / informed trading story")
            else:
                print(f"  >>> Institutional proxy not significant at weekly horizon")

    # ── Mechanism test ───────────────────────────────────────────────────
    print(f"\n[4/4] Attention Mechanism Test...")
    print(f"  Does retail attention moderate the volume-return relationship?")
    print(f"  Splits weeks into HIGH vs LOW attention regimes, tests vol_surprise IC in each\n")

    mech = test_attention_mechanism(
        signals        = signals,
        prices_df      = prices_df,
        common_dates   = common_dates,
        common_tickers = common_tickers,
    )
    print_mechanism_result(mech)

    # ── Summary and memo framing ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  WHAT THIS ADDS TO YOUR RESEARCH")
    print(f"{'='*70}")
    print(f"""
  Your v3 research found that vol_surprise_z (abnormal volume) has
  statistically significant predictive power in Fama-MacBeth (t=2.18).
  The mechanism was unknown — retail attention or institutional flow?

  This analysis uses Google Trends search volume as a free alternative
  data source to test the mechanism directly, following Da et al. (2011).

  The interaction signal (vol_attn_interaction) isolates weeks where
  abnormal volume is accompanied by elevated retail search interest.
  The institutional_vol_proxy isolates the component of volume that
  cannot be explained by retail attention.

  HONEST LIMITATIONS (include these in the memo):
  1. Weekly frequency only — Google Trends does not provide daily data
  2. Google Trends is normalized 0-100 within each query, not comparable
     across tickers without the z-scoring transformation applied here
  3. Search terms are imperfect proxies — "Apple stock" captures retail
     but may also include institutional research departments
  4. 3-year history = ~150 weekly observations — limited statistical power
  5. Survivorship bias in the universe (yfinance limitation, not alt data)

  WHAT THIS DEMONSTRATES TO DE SHAW:
  You understand the difference between finding a signal and understanding
  WHY it works. You used publicly available alternative data to test a
  specific economic mechanism, cited the relevant academic literature,
  and documented the limitations honestly. This is the right approach.
""")

    # ── Save results ─────────────────────────────────────────────────────
    if not fmb_df.empty:
        fmb_df.to_csv(os.path.join(OUT_DIR, "ALT_A_weekly_fmb.csv"), index=False)
    pd.DataFrame([mech]).to_csv(os.path.join(OUT_DIR, "ALT_B_mechanism_test.csv"), index=False)

    # Save weekly IC time series for plotting
    weekly_px = prices_df.resample("W-FRI").last()
    fwd_rets  = weekly_px.pct_change(1).shift(-1)
    vs_df     = signals["weekly_vol_surprise"].reindex(
                    index=common_dates, columns=common_tickers)
    ic_weekly = []
    for dt in common_dates:
        if dt not in fwd_rets.index:
            continue
        s = vs_df.loc[dt].dropna()
        r = fwd_rets.loc[dt].reindex(s.index).dropna()
        common_t = s.index.intersection(r.index)
        if len(common_t) < 10:
            continue
        ic, _ = stats.spearmanr(s[common_t], r[common_t])
        ic_weekly.append({"date": dt, "weekly_vol_surprise_IC": round(ic, 4)})
    pd.DataFrame(ic_weekly).to_csv(
        os.path.join(OUT_DIR, "ALT_C_weekly_ic_timeseries.csv"), index=False)

    print(f"  Results saved to outputs/ALT_*.csv")
    print(f"{'='*70}\n")
