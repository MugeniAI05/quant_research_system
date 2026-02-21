# Stock Factor Backtester

A pipeline for testing whether technical indicators predict short-term stock returns. Given a ticker, it downloads price history, computes ~30 indicators, measures each one's predictive power using Information Coefficient analysis, and backtests the strongest signal with transaction costs included.

## What It Does

1. **Downloads data** — pulls daily OHLCV from Yahoo Finance via `yfinance`, and fetches recent news headlines via DuckDuckGo
2. **Computes indicators** — calculates ~30 technical indicators across four families: momentum, volatility, mean-reversion, and volume
3. **Validates predictive power** — for each indicator, computes the Spearman correlation between today's signal value and the stock's return over the next N days (Information Coefficient). Filters to indicators with IC > 0.02 and p-value < 0.05
4. **Backtests the best indicator** — simulates a long/flat strategy using the top indicator, applying 15 bps round-trip transaction costs (10 bps commissions + 5 bps slippage)
5. **Scores news sentiment** — scores each headline −1 to +1 using a word-count lexicon, then builds sentiment-derived indicators
6. **Generates a report** — outputs a formatted research note with IC stats, backtest metrics, and a signal recommendation

## Limitations to Know About

- **Single stock only** — no portfolio construction or cross-sectional ranking
- **Sentiment is a constant, not time-varying** — the sentiment score from today's headlines gets broadcast across the full price history, so sentiment-derived factors are not meaningful for backtesting
- **No walk-forward validation** — the backtest uses the full in-sample period; there is no out-of-sample test
- **Simple signal** — long when signal > threshold, otherwise flat; no short selling, no position sizing beyond a fixed cap
- **Daily data only** — no intraday, no alternative data

## Quickstart

```bash
pip install numpy pandas scipy yfinance duckduckgo-search
python main_pipeline.py NVDA
```

Or from Python:

```python
from main_pipeline import analyze_ticker

report, results = analyze_ticker("NVDA", period="2y")
print(report)
```

## Indicators Computed

**Momentum** — simple returns over 5/10/20/60-day windows, momentum acceleration (short minus long window), time-series momentum (fraction of up days), z-score normalized momentum, exponential moving average deviation

**Volatility** — realized volatility over 5/10/20/60-day windows, short/long vol ratios, downside-only volatility, volatility of volatility

**Mean-reversion** — distance from 10/20/50-day moving average, RSI(14), Bollinger Band position, z-score from rolling mean, rate-of-change oscillator

**Volume** — volume-weighted returns, volume trend (short vs. long average), on-balance volume

**Sentiment** — raw headline score, sentiment/price divergence, sentiment momentum, sentiment surprise vs. rolling baseline

## Key Concepts

**Information Coefficient (IC)** — Spearman rank correlation between a factor's value on day t and the stock's return from t to t+5. An IC of 0.04 is typical for a single technical factor; anything above 0.02 with p < 0.05 clears the viability bar used here.

**Forward returns** — calculated as `prices.pct_change(5).shift(-5)`, meaning the return earned *after* the signal date. Positions are lagged by one day (`signal.shift(1)`) to prevent look-ahead bias.

**Transaction costs** — 15 bps deducted per round trip (entry + exit), applied whenever position changes. High-turnover strategies are penalized accordingly.

## Configuration

All parameters are in `config.py`:

```python
transaction_cost_bps = 10.0   # Commission cost per trade
slippage_bps = 5.0            # Market impact per trade
holding_period_days = 5        # Forward return horizon
min_ic = 0.02                  # Minimum IC to proceed to backtest
min_sharpe_ratio = 0.5         # Minimum acceptable Sharpe
momentum_windows = [5, 10, 20, 60]
```

## Output Metrics

| Metric | Description |
|--------|-------------|
| IC | Spearman correlation between signal and 5-day forward returns |
| T-stat / p-value | Statistical significance of the IC |
| Sharpe ratio | Annualized risk-adjusted return after costs |
| CAGR | Compound annual growth rate |
| Max drawdown | Largest peak-to-trough decline |
| Win rate | Fraction of holding periods with positive return |
| Calmar ratio | CAGR divided by absolute max drawdown |

## File Structure

```
config.py              # All parameters
data_fetcher.py        # yfinance + DuckDuckGo data acquisition
factor_engineering.py  # Indicator computation
sentiment_analysis.py  # Headline scoring and sentiment factors
factor_validation.py   # IC analysis and viability filtering
backtest_engine.py     # Signal backtesting with costs
reporting.py           # Research note generation
main_pipeline.py       # Orchestrates the full workflow
```

## Dependencies

```
numpy >= 1.24
pandas >= 2.0
scipy >= 1.10
yfinance >= 0.2.28
duckduckgo-search >= 3.8
```

## What to Add to Make This More Rigorous

- **Walk-forward validation** — fit on a rolling training window, evaluate on held-out periods
- **Cross-sectional testing** — rank factors across a universe of stocks rather than testing on one at a time
- **Time-varying sentiment** — ingest a news feed with timestamps so sentiment factors update daily
- **Factor combination** — weight multiple factors using regression or a simple ensemble instead of picking the single best
- **Regime conditioning** — track whether the market is trending or mean-reverting and adjust which factor family to use

## License

For educational and research purposes. Not financial advice.
