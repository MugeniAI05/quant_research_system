"""
signal_decomposition.py
========================
Improvement 6: Decompose OBV into testable sub-components.

OBV combines price direction and volume into one number. This module
breaks it apart to identify WHICH component drives predictive power:

  Component A — Up-day volume:   volume on days price rises
  Component B — Down-day volume: volume on days price falls
  Component C — Volume/price divergence: volume trend vs price trend
  Component D — Volume surprise: today's volume vs recent average
  Component E — Signed volume imbalance: (up_vol - down_vol) / total_vol

Each is a separate testable hypothesis about what information volume
actually contains.

Also computes the known factor controls needed for Fama-MacBeth:
  - 1-day return (short-term reversal)
  - 1-month momentum (21d)
  - 12-1 month momentum (standard Jegadeesh-Titman)
  - Realized volatility (20d)
  - Dollar volume (size proxy)
"""

import numpy as np
import pandas as pd
from typing import Dict


def _zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score."""
    mu    = s.rolling(window).mean()
    sigma = s.rolling(window).std()
    return (s - mu) / sigma.replace(0, np.nan)


# ════════════════════════════════════════════════════════════════════════════
# OBV Signal Decompositions
# ════════════════════════════════════════════════════════════════════════════

def compute_decomposed_signals(
    prices:  pd.Series,
    volumes: pd.Series,
    z_window: int = 60,
) -> Dict[str, pd.Series]:
    """
    Decompose OBV into 5 testable sub-components.

    Returns dict of signal name -> z-scored daily signal.
    All signals are z-scored for comparability.
    """
    ret       = prices.pct_change()
    direction = np.sign(ret).fillna(0.0)
    vol       = volumes.fillna(0.0)

    signals = {}

    # ── A: Up-day volume (buying pressure proxy) ────────────────────────
    # Volume that flows on days price rises — proxy for demand
    up_vol   = vol * (ret > 0).astype(float)
    up_chg   = up_vol.diff(1)
    signals["obv_upvol_z"] = _zscore(up_chg, z_window)

    # ── B: Down-day volume (selling pressure proxy) ─────────────────────
    # Volume on days price falls — proxy for supply
    down_vol = vol * (ret < 0).astype(float)
    down_chg = down_vol.diff(1)
    signals["obv_downvol_z"] = _zscore(down_chg, z_window)

    # ── C: Volume/price divergence ───────────────────────────────────────
    # 5-day volume trend MINUS 5-day price trend (normalized)
    # Positive = volume rising faster than price = potential reversal
    vol_trend   = vol.pct_change(5)
    price_trend = prices.pct_change(5)
    divergence  = vol_trend - price_trend
    signals["vol_price_divergence_z"] = _zscore(divergence, z_window)

    # ── D: Volume surprise ───────────────────────────────────────────────
    # Today's volume vs 20-day average — abnormal volume signal
    avg_vol  = vol.rolling(20).mean()
    vol_surp = (vol - avg_vol) / avg_vol.replace(0, np.nan)
    signals["vol_surprise_z"] = _zscore(vol_surp, z_window)

    # ── E: Signed volume imbalance ───────────────────────────────────────
    # (up_vol - down_vol) / total_vol — net buying/selling pressure
    # This is the cleanest proxy for order flow imbalance
    total_vol = up_vol + down_vol
    imbalance = (up_vol - down_vol) / total_vol.replace(0, np.nan)
    imbalance_chg = imbalance.diff(1)
    signals["signed_vol_imbalance_z"] = _zscore(imbalance_chg, z_window)

    # ── Original composite (for comparison) ─────────────────────────────
    obv_raw = (direction * vol).cumsum()
    obv_chg = obv_raw.diff(1)
    signals["obv_composite_z"] = _zscore(obv_chg, z_window)

    return signals


# ════════════════════════════════════════════════════════════════════════════
# Known Factor Controls (for Fama-MacBeth)
# ════════════════════════════════════════════════════════════════════════════

def compute_control_factors(
    prices:  pd.Series,
    volumes: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Compute standard known factors used to control for in Fama-MacBeth.

    If OBV signal's coefficient drops to zero after adding these controls,
    it means OBV is just a noisy proxy for known effects.

    Controls:
      strev     — 1-day return (short-term reversal, Jegadeesh 1990)
      mom_1m    — 21-day return (1-month momentum)
      mom_12_1  — 252-21 day return (12-1 month momentum, Jegadeesh-Titman 1993)
      vol_20d   — 20-day realized volatility (risk factor)
      dollar_vol — log(price * volume), size proxy
    """
    ret = prices.pct_change()

    controls = {}

    # Short-term reversal: yesterday's return (sign-flipped for reversal)
    controls["strev"] = -ret.shift(1)

    # 1-month momentum
    controls["mom_1m"] = prices.pct_change(21).shift(1)

    # 12-1 month momentum (skip last month to avoid reversal contamination)
    controls["mom_12_1"] = (
        prices.pct_change(252).shift(1) - prices.pct_change(21).shift(1)
    )

    # Realized volatility (annualized 20-day)
    controls["rvol_20d"] = ret.rolling(20).std() * np.sqrt(252)

    # Size proxy: log dollar volume
    dollar_vol = prices * volumes.fillna(0)
    controls["log_dolvol"] = np.log(dollar_vol.rolling(20).mean().replace(0, np.nan))

    return controls


# ════════════════════════════════════════════════════════════════════════════
# Cross-sectional universe computation
# ════════════════════════════════════════════════════════════════════════════

def compute_universe_decomposed(
    prices_df:  pd.DataFrame,
    volumes_df: pd.DataFrame,
    z_window:   int = 60,
) -> Dict[str, pd.DataFrame]:
    """
    Compute all decomposed signals AND controls for every stock.

    Returns:
        signal_panels : dict of signal_name -> (dates x tickers) DataFrame
        control_panels: dict of control_name -> (dates x tickers) DataFrame
    """
    signal_names  = None
    control_names = None
    sig_dicts     = {}
    ctrl_dicts    = {}

    for tkr in prices_df.columns:
        px  = prices_df[tkr]
        vol = volumes_df[tkr]

        sigs  = compute_decomposed_signals(px, vol, z_window)
        ctrls = compute_control_factors(px, vol)

        if signal_names is None:
            signal_names  = list(sigs.keys())
            control_names = list(ctrls.keys())
            for k in signal_names:
                sig_dicts[k] = {}
            for k in control_names:
                ctrl_dicts[k] = {}

        for k, v in sigs.items():
            sig_dicts[k][tkr] = v
        for k, v in ctrls.items():
            ctrl_dicts[k][tkr] = v

    signal_panels  = {k: pd.DataFrame(v) for k, v in sig_dicts.items()}
    control_panels = {k: pd.DataFrame(v) for k, v in ctrl_dicts.items()}

    return signal_panels, control_panels
