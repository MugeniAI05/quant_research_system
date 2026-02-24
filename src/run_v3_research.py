"""
run_v3_research.py
==================
Complete research suite implementing all improvements possible with yfinance data.

Improvements implemented:
  1. Universe expansion         — 98 S&P large-cap stocks
  2. HAC statistical inference  — Newey-West corrected t-stats
  3. Economic hypothesis tests  — see hypothesis_testing.py
  4. Factor controls            — Fama-MacBeth with momentum/reversal/vol/size controls
  5. Long/short portfolio       — dollar-neutral, rebalance frequency sweep
  6. Signal decomposition       — 5 OBV sub-components tested separately
  7. Fama-MacBeth regression    — correct panel statistical framework
  8. Market impact cost model   — turnover-weighted + volume-adjusted costs
  9. Clean walk-forward         — parameters re-estimated per fold

Improvements NOT possible with yfinance (noted for memo):
  - Survivorship-bias-free data (need CRSP point-in-time constituents)
  - Intraday volume decomposition (need TAQ/DTAQ)
  - Short-selling costs by stock (need borrow rate data)
  - Longer history > 3 years (yfinance 3y limit for daily data)

Run:
    cd src
    python universe.py         # once only
    python run_v3_research.py  # full suite (~10-15 min)
    python hypothesis_testing.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from config               import config
from backtest_engine      import VectorBacktester
from universe             import load_universe
from signal_decomposition import compute_universe_decomposed, compute_decomposed_signals
from fama_macbeth         import run_fmb_all_signals, run_fama_macbeth

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("  NOTE: install statsmodels for HAC: pip install statsmodels")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
HORIZON       = 1
Z_WINDOW      = 60
TRAIN_DAYS    = 252
TEST_DAYS     = 63
LONG_QUANTILE = 0.80
SHORT_QUANTILE= 0.20


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="first")]


# ════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 8: Market impact cost model
# ════════════════════════════════════════════════════════════════════════════

def estimate_market_impact(
    prices_df:  pd.DataFrame,
    volumes_df: pd.DataFrame,
    trade_size_pct: float = 0.001,  # trade 0.1% of daily volume per stock
) -> pd.DataFrame:
    """
    Estimate per-stock market impact cost using the square-root rule.

    Market impact ≈ sigma * sqrt(trade_size / adv)
    where adv = average daily volume in dollars.

    This is a simplified version of the Almgren-Chriss model used at
    professional funds. It shows which stocks are cheapest to trade
    and whether high-IC stocks are also high-cost stocks.

    Returns DataFrame with per-stock cost estimates in bps.
    """
    rows = []
    for tkr in prices_df.columns:
        px  = prices_df[tkr].dropna()
        vol = volumes_df[tkr].dropna()

        # Realized volatility (daily)
        daily_vol = px.pct_change().std()

        # Average daily dollar volume
        adv = (px * vol).rolling(20).mean().iloc[-1]
        if adv <= 0 or np.isnan(adv):
            continue

        # Trade size in dollars (assuming $1M position, 0.1% of ADV)
        trade_usd = adv * trade_size_pct

        # Square-root market impact in bps
        # impact = sigma * sqrt(Q / ADV) * 10000
        impact_bps = daily_vol * np.sqrt(trade_size_pct) * 10000

        rows.append({
            "ticker":       tkr,
            "adv_usd_M":    round(adv / 1e6, 1),
            "daily_vol":    round(daily_vol * 100, 3),
            "impact_bps":   round(impact_bps, 2),
            "flat_tc_bps":  15,
            "total_tc_bps": round(impact_bps + 15, 2),
        })

    return pd.DataFrame(rows).sort_values("impact_bps")


# ════════════════════════════════════════════════════════════════════════════
# Long/Short portfolio with rebalance sweep
# ════════════════════════════════════════════════════════════════════════════

def longshort_sweep(
    prices_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    label:     str = "signal",
) -> pd.DataFrame:
    """Test L/S portfolio across rebalance frequencies."""
    bt = VectorBacktester()
    rebalance_map = {
        "Daily (1d)":       1,
        "Weekly (5d)":      5,
        "Bi-weekly (10d)": 10,
        "Monthly (21d)":   21,
    }
    rows = []
    for name, reb in rebalance_map.items():
        res = bt.backtest_cross_section(
            prices         = prices_df,
            signal         = signal_df,
            rebalance_every= reb,
            long_quantile  = LONG_QUANTILE,
            short_quantile = SHORT_QUANTILE,
            gross_exposure = 1.0,
            dollar_neutral = True,
            max_weight     = 0.10,
        )
        m = res.metrics
        rows.append({
            "Signal":    label,
            "Rebalance": name,
            "Sharpe":    round(m.sharpe_ratio, 3),
            "CAGR %":    round(m.cagr * 100, 2),
            "Max DD %":  round(m.max_drawdown * 100, 2),
            "Win %":     round(m.win_rate * 100, 1),
            "Turnover":  round(m.turnover, 1),
            "IC":        round(m.information_coefficient, 4),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Clean walk-forward with parameter re-estimation
# ════════════════════════════════════════════════════════════════════════════

def walkforward_v3(
    prices_df:  pd.DataFrame,
    volumes_df: pd.DataFrame,
    train_days: int = TRAIN_DAYS,
    test_days:  int = TEST_DAYS,
) -> pd.DataFrame:
    """
    Walk-forward validation selecting BOTH best signal decomposition
    AND best z_window on training data, then evaluating on test data.

    This is cleaner than the v2 version which only searched z_window.
    """
    from signal_decomposition import compute_decomposed_signals

    n          = len(prices_df)
    n_windows  = (n - train_days) // test_days
    bt         = VectorBacktester()
    z_windows  = [20, 40, 60, 90]
    rows       = []

    print(f"  Running {n_windows} walk-forward windows...")

    for i in range(n_windows):
        t0 = i * test_days
        t1 = t0 + train_days
        t2 = min(t1 + test_days, n)

        if t2 - t1 < 20:
            continue

        train_px  = prices_df.iloc[t0:t1]
        train_vol = volumes_df.iloc[t0:t1]
        test_px   = prices_df.iloc[t1:t2]
        test_vol  = volumes_df.iloc[t1:t2]

        # ── Select best signal + z_window on TRAINING data only ─────────
        best_sharpe = -np.inf
        best_sig    = "obv_composite_z"
        best_zw     = Z_WINDOW
        best_tIC    = np.nan

        for zw in z_windows:
            # Compute all decomposed signals on training data
            train_sigs = {}
            for tkr in train_px.columns:
                sigs = compute_decomposed_signals(train_px[tkr], train_vol[tkr], zw)
                for sname, sval in sigs.items():
                    if sname not in train_sigs:
                        train_sigs[sname] = {}
                    train_sigs[sname][tkr] = sval

            for sname, sdict in train_sigs.items():
                sig_df = pd.DataFrame(sdict)
                try:
                    res = bt.backtest_cross_section(
                        prices         = train_px,
                        signal         = sig_df,
                        rebalance_every= 5,   # weekly in walk-forward
                        long_quantile  = LONG_QUANTILE,
                        short_quantile = SHORT_QUANTILE,
                        gross_exposure = 1.0,
                        dollar_neutral = True,
                        max_weight     = 0.10,
                    )
                    if res.metrics.sharpe_ratio > best_sharpe:
                        best_sharpe = res.metrics.sharpe_ratio
                        best_sig    = sname
                        best_zw     = zw
                        best_tIC    = res.metrics.information_coefficient
                except Exception:
                    continue

        # ── Re-compute best signal on TEST data ─────────────────────────
        test_sigs = {}
        for tkr in test_px.columns:
            sigs = compute_decomposed_signals(test_px[tkr], test_vol[tkr], best_zw)
            for sname, sval in sigs.items():
                if sname not in test_sigs:
                    test_sigs[sname] = {}
                test_sigs[sname][tkr] = sval

        test_sig_df = pd.DataFrame(test_sigs.get(best_sig, {}))

        if test_sig_df.empty:
            continue

        try:
            test_res = bt.backtest_cross_section(
                prices         = test_px,
                signal         = test_sig_df,
                rebalance_every= 5,
                long_quantile  = LONG_QUANTILE,
                short_quantile = SHORT_QUANTILE,
                gross_exposure = 1.0,
                dollar_neutral = True,
                max_weight     = 0.10,
            )
            tm = test_res.metrics
        except Exception:
            continue

        rows.append({
            "Window":         i + 1,
            "Train Period":   f"{train_px.index[0].date()} → {train_px.index[-1].date()}",
            "Test Period":    f"{test_px.index[0].date()} → {test_px.index[-1].date()}",
            "Best Signal":    best_sig,
            "Best z_window":  best_zw,
            "Train Sharpe":   round(best_sharpe, 3),
            "Train IC":       round(best_tIC, 4),
            "Test Sharpe":    round(tm.sharpe_ratio, 3),
            "Test CAGR %":    round(tm.cagr * 100, 2),
            "Test Max DD %":  round(tm.max_drawdown * 100, 2),
            "Test IC":        round(tm.information_coefficient, 4),
        })

        print(f"    W{i+1}: best={best_sig}(z={best_zw}) | "
              f"Train Sharpe={best_sharpe:.3f} | Test Sharpe={tm.sharpe_ratio:.3f}")

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Main runner
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*70)
    print("  EQUITY FACTOR LAB — v3 FULL RESEARCH SUITE")
    print("="*70)
    print("  Improvements: HAC inference, Fama-MacBeth, signal decomposition,")
    print("  factor controls, market impact model, clean walk-forward")
    print(f"  Statistical framework: {'HAC/Newey-West via statsmodels' if HAS_STATSMODELS else 'naive (install statsmodels)'}")

    # ── Load universe ────────────────────────────────────────────────────
    print(f"\n[0] Loading universe...")
    prices_df, volumes_df = load_universe()
    prices_df  = _normalize_index(prices_df)
    volumes_df = _normalize_index(volumes_df)
    n_stocks, n_days = prices_df.shape[1], prices_df.shape[0]
    print(f"  ✓ {n_stocks} stocks, {n_days} days "
          f"({prices_df.index[0].date()} → {prices_df.index[-1].date()})")

    # ── Compute all signals and controls ────────────────────────────────
    print(f"\n[1/5] Computing signal decompositions and factor controls...")
    print(f"  5 OBV sub-components + 5 known factor controls across {n_stocks} stocks")
    signal_panels, control_panels = compute_universe_decomposed(
        prices_df, volumes_df, z_window=Z_WINDOW
    )
    print(f"  ✓ Signals: {list(signal_panels.keys())}")
    print(f"  ✓ Controls: {list(control_panels.keys())}")

    # ── IMPROVEMENT 7: Fama-MacBeth regression ───────────────────────────
    print(f"\n[2/5] Fama-MacBeth Regression (signal decompositions, with + without controls)...")
    print(f"  Panel: {n_stocks} stocks x {n_days} days")
    print(f"  Controls: strev, mom_1m, mom_12_1, rvol_20d, log_dolvol")
    print(f"  Question: does OBV have incremental predictive power beyond known factors?\n")

    fmb_df = run_fmb_all_signals(
        signal_panels  = signal_panels,
        prices_df      = prices_df,
        control_panels = control_panels,
        horizon        = HORIZON,
    )

    print(f"\n{'='*70}")
    print(f"  FAMA-MACBETH RESULTS")
    print(f"  Raw: signal only | Ctrl: after controlling for momentum/reversal/vol/size")
    print(f"  Attenuation: Ctrl coef / Raw coef  (< 0.5 = mostly proxying known factors)")
    print(f"{'='*70}")

    display_cols = ["Signal", "Raw coef", "Raw t-stat", "Raw p-val",
                    "Sig (raw)", "Ctrl coef", "Ctrl t-stat", "Ctrl p-val",
                    "Sig (ctrl)", "Attenuation"]
    print(fmb_df[display_cols].to_string(index=False))

    n_raw_sig  = (fmb_df["Sig (raw)"]  == "YES").sum()
    n_ctrl_sig = (fmb_df["Sig (ctrl)"] == "YES").sum()
    print(f"\n  Signals significant raw      : {n_raw_sig}/{len(fmb_df)}")
    print(f"  Signals significant controlled: {n_ctrl_sig}/{len(fmb_df)}")

    if n_ctrl_sig > 0:
        survivors = fmb_df[fmb_df["Sig (ctrl)"] == "YES"]["Signal"].tolist()
        print(f"  Survivors after factor control: {survivors}")
        print(f"  >>> These signals have INCREMENTAL predictive power beyond known factors")
    else:
        print(f"  >>> No signal survives factor control — OBV is proxying known effects")

    # Average attenuation
    avg_atten = fmb_df["Attenuation"].replace([np.inf, -np.inf], np.nan).dropna().mean()
    print(f"  Avg attenuation ratio: {avg_atten:.3f}")
    if avg_atten < 0.3:
        print(f"  >>> Heavy attenuation — signal mostly captures momentum/reversal")
    elif avg_atten < 0.7:
        print(f"  >>> Moderate attenuation — partial overlap with known factors")
    else:
        print(f"  >>> Low attenuation — signal is largely orthogonal to known factors")

    # ── IMPROVEMENT 8: Market impact cost model ──────────────────────────
    print(f"\n[3/5] Market Impact Cost Model...")
    print(f"  Square-root impact model: cost ≈ sigma * sqrt(trade_size / ADV)")
    impact_df = estimate_market_impact(prices_df, volumes_df, trade_size_pct=0.001)

    print(f"\n  Cost summary across {len(impact_df)} stocks:")
    print(f"  Median flat TC          : 15 bps/trade")
    print(f"  Median market impact    : {impact_df['impact_bps'].median():.1f} bps/trade")
    print(f"  Median total TC         : {impact_df['total_tc_bps'].median():.1f} bps/trade")
    print(f"  Cheapest 10 stocks      : {list(impact_df.head(10)['ticker'])}")
    print(f"  Most expensive 10       : {list(impact_df.tail(10)['ticker'])}")
    print(f"\n  IMPLICATION: At daily rebalance (161x turnover), total cost ~")
    total_cost_daily = impact_df['total_tc_bps'].median() * 161 * 2
    print(f"  {total_cost_daily:.0f} bps/year ({total_cost_daily/100:.1f}% per year)")
    print(f"  This exceeds any realistic gross return for a 1-day signal.")

    # ── IMPROVEMENT 5+6: L/S backtest on best signal decomposition ───────
    print(f"\n[4/5] Long/Short Portfolio — Best Signal Decomposition...")

    # Find best signal by raw FMB t-stat
    best_fmb_row = fmb_df.loc[fmb_df["Raw t-stat"].abs().idxmax()]
    best_sig_name = best_fmb_row["Signal"]
    print(f"  Using best signal by FMB t-stat: {best_sig_name}")
    print(f"  (Raw t-stat: {best_fmb_row['Raw t-stat']:.3f}, "
          f"Ctrl t-stat: {best_fmb_row['Ctrl t-stat']:.3f})\n")

    best_sig_df = _normalize_index(signal_panels[best_sig_name])
    ls_sweep_df = longshort_sweep(prices_df, best_sig_df, label=best_sig_name)

    print(f"  {'Rebalance':<18} {'Sharpe':>8} {'CAGR %':>8} {'Max DD %':>9} "
          f"{'Win %':>7} {'Turnover':>10}")
    print(f"  {'-'*65}")
    for _, row in ls_sweep_df.iterrows():
        flag = " ◄" if row["Sharpe"] == ls_sweep_df["Sharpe"].max() else ""
        print(f"  {row['Rebalance']:<18} {row['Sharpe']:>8.3f} {row['CAGR %']:>8.2f} "
              f"{row['Max DD %']:>9.2f} {row['Win %']:>7.1f} "
              f"{row['Turnover']:>10.1f}{flag}")

    best_ls = ls_sweep_df.loc[ls_sweep_df["Sharpe"].idxmax()]
    if best_ls["Sharpe"] >= 0.5:
        print(f"\n  >>> Best decomposed signal achieves Sharpe {best_ls['Sharpe']:.3f} "
              f"at {best_ls['Rebalance']}")
    elif best_ls["Sharpe"] >= 0.0:
        print(f"\n  >>> Marginal — best Sharpe {best_ls['Sharpe']:.3f} at {best_ls['Rebalance']}")
    else:
        print(f"\n  >>> No decomposed signal achieves positive Sharpe at any frequency")

    # Also run original composite for comparison
    print(f"\n  Composite OBV signal for comparison:")
    composite_df = _normalize_index(signal_panels["obv_composite_z"])
    ls_composite = longshort_sweep(prices_df, composite_df, label="obv_composite_z")
    best_comp = ls_composite.loc[ls_composite["Sharpe"].idxmax()]
    print(f"  Best Sharpe: {best_comp['Sharpe']:.3f} at {best_comp['Rebalance']}")

    # ── IMPROVEMENT 9: Clean walk-forward ────────────────────────────────
    print(f"\n[5/5] Clean Walk-Forward Validation (signal + parameter selection per fold)...")
    print(f"  Selects best signal decomposition AND z_window on training data only.")
    print(f"  Weekly rebalance in all folds.\n")

    wf_df = walkforward_v3(prices_df, volumes_df)

    if not wf_df.empty:
        display_wf = ["Window", "Test Period", "Best Signal", "Best z_window",
                      "Train Sharpe", "Test Sharpe", "Test CAGR %", "Test IC"]
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD RESULTS (clean, signal + params re-selected per fold)")
        print(f"{'='*70}")
        print(wf_df[display_wf].to_string(index=False))

        avg_ts   = wf_df["Train Sharpe"].mean()
        avg_oos  = wf_df["Test Sharpe"].mean()
        pct_pos  = (wf_df["Test Sharpe"] > 0).mean() * 100
        avg_tIC  = wf_df["Train IC"].mean()
        avg_oIC  = wf_df["Test IC"].mean()
        decay    = abs(avg_oIC) / max(abs(avg_tIC), 1e-9)

        print(f"\n  Summary:")
        print(f"  Avg train Sharpe  : {avg_ts:+.3f}")
        print(f"  Avg OOS Sharpe    : {avg_oos:+.3f}  ← honest number")
        print(f"  Avg train IC      : {avg_tIC:+.4f}")
        print(f"  Avg OOS IC        : {avg_oIC:+.4f}  ← honest number")
        print(f"  IC decay ratio    : {decay:.3f}  (1.0 = no decay)")
        print(f"  % positive OOS    : {pct_pos:.0f}%")

        # Signal selection frequency
        sig_counts = wf_df["Best Signal"].value_counts()
        print(f"\n  Signal selected most often: {sig_counts.index[0]} ({sig_counts.iloc[0]}/{len(wf_df)} windows)")
        print(f"  (Consistent selection = stable signal, varied selection = unstable)")

        if avg_oos >= 0.5 and pct_pos >= 60:
            print(f"\n  >>> OOS Sharpe {avg_oos:.2f} — ROBUST after clean selection")
        elif avg_oos >= 0.0:
            print(f"\n  >>> Marginally positive OOS — signal survives weakly")
        else:
            print(f"\n  >>> Negative OOS Sharpe — signal does not survive clean selection")

        print(f"\n  NOTE: {len(wf_df)} windows. Directional evidence only, not statistically conclusive.")

    # ── Save all results ─────────────────────────────────────────────────
    fmb_df.to_csv(os.path.join(OUT_DIR, "V3_A_fama_macbeth.csv"), index=False)
    impact_df.to_csv(os.path.join(OUT_DIR, "V3_B_market_impact.csv"), index=False)
    ls_sweep_df.to_csv(os.path.join(OUT_DIR, "V3_C_longshort_sweep.csv"), index=False)
    if not wf_df.empty:
        wf_df.to_csv(os.path.join(OUT_DIR, "V3_D_walkforward.csv"), index=False)

    # Save per-stock FMB detail
    ic_rows = []
    for sig_name, sig_df in signal_panels.items():
        for tkr in sig_df.columns:
            s = sig_df[tkr].dropna()
            fwd = prices_df[tkr].pct_change(HORIZON).shift(-HORIZON)
            common = s.index.intersection(fwd.dropna().index)
            if len(common) < 60:
                continue
            ic, _ = stats.spearmanr(s[common], fwd[common])
            ic_rows.append({"ticker": tkr, "signal": sig_name, "IC": round(ic, 4)})
    pd.DataFrame(ic_rows).to_csv(os.path.join(OUT_DIR, "V3_E_ic_by_stock_signal.csv"), index=False)

    print(f"\n{'='*70}")
    print(f"  ALL RESULTS SAVED TO: outputs/")
    print(f"  V3_A_fama_macbeth.csv       — FMB results with/without factor controls")
    print(f"  V3_B_market_impact.csv      — per-stock cost model")
    print(f"  V3_C_longshort_sweep.csv    — L/S results by rebalance frequency")
    print(f"  V3_D_walkforward.csv        — clean walk-forward")
    print(f"  V3_E_ic_by_stock_signal.csv — IC by stock and signal decomposition")
    print(f"\n  Next: python hypothesis_testing.py")
    print(f"{'='*70}\n")
