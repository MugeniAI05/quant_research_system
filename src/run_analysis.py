"""
run_analysis.py
===============
Main entry point for the quant research pipeline.

Run with:
    cd src
    python run_analysis.py

Change TICKER / PERIOD at the top to analyse a different stock.
"""

import sys
import os

# ── Ensure src/ is on the path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from data_fetcher        import MarketDataFetcher, NewsDataFetcher
from factor_engineering  import FactorEngine
from sentiment_analysis  import SentimentScorer
from factor_validation   import FactorValidator, MultiFactorValidator
from backtest_engine     import VectorBacktester

# ── Settings ─────────────────────────────────────────────────────────────────
TICKER = "NVDA"   # change to AAPL, TSLA, MSFT, GOOGL, etc.
PERIOD = "2y"
TOP_N  = 5        # number of top factors to display

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"  QUANT RESEARCH PIPELINE  —  {TICKER}  ({PERIOD})")
print("="*70)

# ── Step 1: Fetch data ───────────────────────────────────────────────────────
print("\n[1/5] Fetching market data & news...")
market = MarketDataFetcher().fetch(TICKER, period=PERIOD)
news   = NewsDataFetcher().fetch(TICKER)

if market is None:
    print("  ERROR: Could not fetch market data. Check your internet connection.")
    sys.exit(1)

print(f"  ✓ {len(market.prices)} trading days  "
      f"({market.start_date.date()} → {market.end_date.date()})")
print(f"  ✓ {len(news.headlines)} headlines  (source: {news.source})")

# ── Step 2: Compute factors ──────────────────────────────────────────────────
print("\n[2/5] Computing factors...")
engine  = FactorEngine()
factors = engine.compute_all_factors(market.prices, market.volumes)
print(f"  ✓ {len(factors)} factors computed across "
      f"{len(set(f.family for f in factors.values()))} families")

# ── Step 3: Sentiment ────────────────────────────────────────────────────────
print("\n[3/5] Scoring sentiment...")
sentiment = SentimentScorer().aggregate_sentiment(news.headlines, TICKER)
print(f"  ✓ Aggregate score : {sentiment.aggregate_score:+.3f}  →  {sentiment.sentiment_label}")
print(f"     Positive: {sentiment.n_positive}  |  Negative: {sentiment.n_negative}  |  Neutral: {sentiment.n_neutral}")
for headline, score in zip(sentiment.headlines, sentiment.individual_scores):
    bar = "+" if score > 0.1 else "-" if score < -0.1 else "="
    print(f"    [{bar}] ({score:+.2f})  {headline[:80]}")

# ── Step 4: Validate all factors ─────────────────────────────────────────────
print("\n[4/5] Validating factors (IC analysis)...")
mv = MultiFactorValidator()
mv.validate_all(
    {name: f.values for name, f in factors.items()},
    market.prices,
    horizon=5,
    run_decay=False,
)
print(mv.summary_report())

ranked = mv.rank_factors(by="ic")
print("Top factors by |IC|:")
print(f"  {'Factor':<25} {'IC':>7}  {'t-stat':>7}  {'p-val':>7}  {'Viable':>7}")
print("  " + "-"*58)
for _, row in ranked.head(TOP_N).iterrows():
    viable_str = "YES" if row["is_viable"] else "no"
    print(f"  {row['factor']:<25} {row['ic']:>+7.4f}  {row['t_stat']:>7.3f}  "
          f"{row['p_value']:>7.4f}  {viable_str:>7}")

# ── Step 5: Backtest best factor ─────────────────────────────────────────────
print("\n[5/5] Backtesting best viable factor...")
viable = mv.get_viable_factors()

if not viable:
    print("  ✗ No viable factors found — try a longer period or different ticker.")
    sys.exit(0)

best_name   = ranked[ranked["is_viable"]].iloc[0]["factor"]
best_signal = factors[best_name].values

result = VectorBacktester().backtest_signal(
    prices=market.prices,
    signal=best_signal,
    signal_threshold=0.0,
)
m = result.metrics

rating = ("EXCELLENT"  if m.sharpe_ratio >= 1.5 else
          "STRONG"     if m.sharpe_ratio >= 1.0 else
          "ACCEPTABLE" if m.sharpe_ratio >= 0.5 else "WEAK")

print(f"\n  Best factor    : {best_name}")
print(f"  Sharpe ratio   : {m.sharpe_ratio:+.3f}  [{rating}]")
print(f"  CAGR           : {m.cagr*100:+.2f}%")
print(f"  Max drawdown   : {m.max_drawdown*100:.2f}%")
print(f"  Win rate       : {m.win_rate*100:.1f}%")
print(f"  Total trades   : {m.total_trades}")
print(f"  Calmar ratio   : {m.calmar_ratio:.3f}")

print("\n" + "="*70)
print("  DONE — run run_improvements.py for the full research suite")
print("="*70 + "\n")
