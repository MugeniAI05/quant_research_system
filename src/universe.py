"""
universe.py — compatible with yfinance 1.2.0
Replace your existing src/universe.py with this file.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
import time

SP100_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "LLY", "JPM",
    "TSLA", "UNH", "V", "XOM", "AVGO", "MA", "JNJ", "PG", "COST", "HD",
    "MRK", "ABBV", "CVX", "KO", "WMT", "PEP", "ADBE", "CRM", "AMD",
    "NFLX", "BAC", "TMO", "ACN", "MCD", "CSCO", "ABT", "WFC", "LIN",
    "DHR", "TXN", "NEE", "PM", "ORCL", "DIS", "QCOM", "RTX", "AMGN",
    "BMY", "HON", "UNP", "IBM", "COP", "SPGI", "CAT", "INTU", "GE",
    "LOW", "MS", "GS", "AXP", "SBUX", "AMAT", "ADI", "GILD", "MDLZ",
    "REGN", "C", "MMC", "TJX", "CI", "DE", "BLK", "ISRG", "SYK",
    "VRTX", "ZTS", "CB", "MO", "SO", "EOG", "BSX", "NOC", "PGR",
    "USB", "ITW", "CME", "DUK", "APD", "MCO", "FDX", "TGT", "HUM",
    "CL", "EQIX", "AON", "NSC", "EMR", "ETN", "MMM", "PLD",
]

PERIOD    = "3y"
MIN_DAYS  = 400
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
CACHE_PX  = os.path.join(OUT_DIR, "universe_prices.parquet")
CACHE_VOL = os.path.join(OUT_DIR, "universe_volumes.parquet")


def _clean_series(s):
    """Flatten any MultiIndex columns, squeeze to Series."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.squeeze().dropna()


def fetch_universe(
    tickers: list       = None,
    period:  str        = PERIOD,
    min_days: int       = MIN_DAYS,
    force_refresh: bool = False,
) -> tuple:

    os.makedirs(OUT_DIR, exist_ok=True)

    if tickers is None:
        tickers = SP100_TICKERS

    # ── Load cache ───────────────────────────────────────────────────────────
    if not force_refresh and os.path.exists(CACHE_PX) and os.path.exists(CACHE_VOL):
        print("  Loading from cache (pass force_refresh=True to re-download)...")
        prices_df  = pd.read_parquet(CACHE_PX)
        volumes_df = pd.read_parquet(CACHE_VOL)
        print(f"  ✓ {prices_df.shape[1]} tickers, {prices_df.shape[0]} days (cached)")
        return prices_df, volumes_df

    print(f"  yfinance {yf.__version__} — fetching {len(tickers)} tickers one-by-one")
    print(f"  (Using Ticker.history() — the method that works on your machine)\n")

    px_dict  = {}
    vol_dict = {}
    failed   = []

    for i, tkr in enumerate(tickers):
        try:
            df = yf.Ticker(tkr).history(period=period, auto_adjust=True)

            if df is None or df.empty:
                failed.append(tkr)
                continue

            close  = _clean_series(df["Close"])
            volume = _clean_series(df["Volume"])

            # Remove timezone for consistent indexing
            if hasattr(close.index, "tz") and close.index.tz is not None:
                close.index  = close.index.tz_localize(None)
                volume.index = volume.index.tz_localize(None)

            if len(close) < min_days or (close <= 0).any():
                failed.append(tkr)
                continue

            px_dict[tkr]  = close
            vol_dict[tkr] = volume

        except Exception as e:
            failed.append(tkr)

        # Progress every 10 tickers
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(tickers)}  ({len(px_dict)} OK, {len(failed)} failed)")

        time.sleep(0.1)   # gentle rate limiting

    if len(px_dict) == 0:
        raise RuntimeError(
            "0 tickers fetched. Test manually: "
            "yf.Ticker('AAPL').history(period='5d')"
        )

    print(f"\n  ✓ {len(px_dict)} tickers fetched, {len(failed)} skipped")
    if failed:
        print(f"    Skipped: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    # ── Align to common dates ────────────────────────────────────────────────
    prices_df  = pd.DataFrame(px_dict).sort_index()
    volumes_df = pd.DataFrame(vol_dict).sort_index().reindex(prices_df.index)

    # Drop tickers with > 5% missing after alignment
    keep = prices_df.columns[prices_df.isna().mean() < 0.05]
    prices_df  = prices_df[keep].ffill().dropna(how="any")
    volumes_df = volumes_df[keep].ffill().fillna(0)

    n_stocks = prices_df.shape[1]
    n_days   = prices_df.shape[0]

    print(f"  ✓ Final universe: {n_stocks} tickers x {n_days} days")
    print(f"    Date range: {prices_df.index[0].date()} → {prices_df.index[-1].date()}")

    # ── Save parquet cache ───────────────────────────────────────────────────
    prices_df.to_parquet(CACHE_PX)
    volumes_df.to_parquet(CACHE_VOL)
    print(f"  ✓ Cached to outputs/")

    return prices_df, volumes_df


def load_universe(force_refresh: bool = False) -> tuple:
    return fetch_universe(force_refresh=force_refresh)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BUILDING EQUITY UNIVERSE")
    print("="*60 + "\n")

    px, vol = fetch_universe(force_refresh=True)

    print(f"\n  Done.")
    print(f"  Universe : {px.shape[1]} stocks x {px.shape[0]} days")
    print(f"  Tickers  : {list(px.columns[:8])} ...")
    print("\n" + "="*60 + "\n")
