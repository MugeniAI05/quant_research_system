# equity-factor-lab

A systematic framework for testing whether a quantitative equity signal has predictive content, applied to volume-based signals in S&P 100 stocks.

Built as a research writing sample demonstrating production-grade quantitative methods: signal decomposition, Fama-MacBeth regression with factor controls, multiple testing correction, Fama-French 5-factor augmentation, alternative data integration, and honest out-of-sample validation.

---

## Research Question

Does On-Balance Volume (OBV) contain incremental predictive information beyond known equity risk factors?

OBV is a price-volume technical indicator created by Joseph Granville (1963). The academic hypothesis tested here is whether abnormal volume relative to price trend captures institutional accumulation — informed large traders spreading orders over time to minimise market impact (Kyle 1985), creating detectable volume/price divergence before full price adjustment.

---

## Key Findings

| Test | Result | Interpretation |
|------|--------|----------------|
| Cross-sectional mean IC (98 stocks) | +0.003, p=0.54 | OBV does not generalise beyond NVDA |
| vol_price_divergence_z (FMB, controlled) | t=1.90, p=0.058 | Suggestive but not significant |
| vol_surprise_z (FMB, controlled) | t=2.25, p=0.025 | Significant uncorrected |
| Both signals post-FF5 augmentation | <10% attenuation | Orthogonal to profitability/investment factors |
| Both signals post-multiple-testing (BH) | 0/6 survive | Consistent with false discovery |
| Best L/S Sharpe (bi-weekly, gross) | 0.471 | Marginal, pre-cost |
| Walk-forward OOS Sharpe (7 windows) | −0.99 | Signal does not survive OOS |
| Google Trends: attention_surprise | t=3.69, p=0.0003 | Continuation, not reversal |
| Market impact at $50M AUM | 19 bps/year | Costs are NOT the binding constraint |

**Conclusion:** The binding constraint is OOS signal instability, not transaction costs. The signal earns ~2% CAGR gross, which survives costs at any realistic fund size (S&P 100 stocks are too liquid for impact to matter at <$500M AUM), but does not survive walk-forward validation.

---

## Economic Mechanisms

**Why volume/price divergence might work:** Kyle (1985) shows informed institutional traders spread large orders over time to minimise market impact. During accumulation, volume rises while price rises slowly because the buyer uses limit orders and manages footprint. This creates positive volume/price divergence — a proxy for the accumulation phase — which should predict returns as the position completes and the catalyst becomes public.

**Why it fails OOS:** The signal requires the divergence to be caused by informed accumulation rather than retail disagreement. With only 3 years of data across 98 stocks, the sample is insufficient to distinguish these regimes reliably.

**Why Google Trends shows continuation, not reversal:** Da, Engelberg & Gao (2011) found reversal using smaller stocks where retail investors dominate flow. In S&P 100 stocks (75-90% institutional ownership), elevated search interest more plausibly reflects information arrival — earnings catalyst research, regulatory analysis — rather than retail attention-chasing. Institutional arbitrageurs can trade against retail mispricings in large-caps far more easily than in small-caps.

---

## Repository Structure

```
equity-factor-lab/
├── src/
│   ├── config.py                # Central configuration (edit parameters here)
│   ├── universe.py              # S&P 100 universe construction with caching
│   ├── signal_decomposition.py  # 5 OBV sub-components + factor controls
│   ├── fama_macbeth.py          # FMB panel regression with HAC inference
│   ├── hypothesis_testing.py    # H1 (size), H2 (liquidity), H3 (regime) tests
│   ├── backtest_engine.py       # Vector backtester: single-asset + cross-sectional
│   ├── factor_validation.py     # IC analysis, decay, quintiles, HAC t-stats
│   ├── factor_engineering.py    # General factor library: momentum/vol/reversal/volume
│   ├── data_fetcher.py          # yfinance wrapper with validation
│   ├── sentiment_analysis.py    # Loughran-McDonald lexicon sentiment scorer
│   ├── alt_data.py              # Google Trends fetching and attention signals
│   ├── reporting.py             # Research report generator
│   │
│   ├── run_analysis.py          # [ENTRY] Single-stock exploratory pipeline
│   ├── run_v3_research.py       # [ENTRY] Full cross-sectional research suite
│   ├── run_altdata_research.py  # [ENTRY] Google Trends mechanism test
│   └── research_gaps.py         # [ENTRY] FF5 augmentation, multiple testing, cost model
│
└── outputs/                     # Generated CSVs (git-ignored)
```

---

## Running the Research

**Step 1 — Install dependencies**
```bash
pip install yfinance pandas numpy scipy statsmodels pyarrow pytrends
```

**Step 2 — Build the universe (once, ~5 minutes)**
```bash
cd src
python universe.py
```
Downloads 3 years of daily OHLCV for 98 S&P 100 stocks and caches to `outputs/`.

**Step 3 — Run the full research suite (~15 minutes)**
```bash
python run_v3_research.py
```
Produces: FMB results, market impact model, L/S backtest sweep, walk-forward validation.

**Step 4 — Economic hypothesis tests**
```bash
python hypothesis_testing.py
```
Tests H1 (size effect), H2 (liquidity effect), H3 (regime dependence).

**Step 5 — Alternative data: Google Trends mechanism test**
```bash
python run_altdata_research.py
```
Tests whether abnormal volume is driven by retail attention (Da et al. 2011) or institutional flow. Requires `pytrends`.

**Step 6 — Research gaps: FF5, multiple testing, cost model**
```bash
python research_gaps.py
```
Runs ~25 minutes (rolling 252-day FF5 betas for 98 stocks). Produces the three gap analyses.

**Single-stock exploration**
```bash
python run_analysis.py   # TICKER = "NVDA" by default, change at top of file
```

---

## Methodology Notes

### What this does correctly
- **HAC/Newey-West inference** throughout — naive t-stats overstate significance for overlapping forward returns
- **Fama-MacBeth with factor controls** — tests incremental predictive power, not just raw correlation
- **Clean walk-forward** — signal AND z-window selected on training data only, never on test data
- **Multiple testing correction** — Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg applied to all 6 signals
- **FF5 augmentation** — rolling per-stock betas on price/volume proxies for RMW and CMA
- **Cost model anchored to AUM** — correct direction (AUM → position → ADV participation), not reverse

### Limitations
- **Survivorship bias** — yfinance returns current index constituents; stocks that were delisted are excluded. True point-in-time data requires CRSP.
- **3-year history** — limits statistical power; 7 walk-forward windows is directional evidence, not conclusive.
- **FF5 proxies** — RMW and CMA require Compustat (book value, operating profitability, total assets). Price/volume proxies are approximations.
- **Google Trends** — weekly frequency only; normalized 0-100 within each query, not comparable across tickers without z-scoring.
- **Short-selling costs** — assumed zero; actual borrow rates vary and would widen the gap between gross and net returns.

---

## Academic References

- Fama & MacBeth (1973) — Panel regression framework for cross-sectional predictability
- Kyle (1985) — Optimal order splitting by informed traders under price impact
- Admati & Pfleiderer (1988) — Concentration of informed trading in liquid periods
- Glosten & Milgrom (1985) — Bid-ask spread as function of informed order flow
- Jegadeesh & Titman (1993) — Momentum factor (12-1 month)
- Jegadeesh (1990) — Short-term reversal factor
- Fama & French (2015) — Five-factor model (Mkt, SMB, HML, RMW, CMA)
- Da, Engelberg & Gao (2011) — Google SVI as retail attention proxy; reversal in small-caps
- Barber & Odean (2008) — Attention-driven buying concentrated in retail-dominated stocks
- Llorente et al. (2002) — Volume and return autocorrelation: informed vs uninformed trading
- Ross (1989) — Information flow and price volatility relationship
- Almgren & Chriss (2000) — Optimal execution and the square-root market impact model
- Benjamini & Hochberg (1995) — False discovery rate correction for multiple comparisons

---

## Data Sources

- **Price and volume data:** Yahoo Finance via `yfinance` (free, 3-year daily history)
- **Attention proxy:** Google Trends via `pytrends` (free, weekly, US region)
- **Factor data:** Constructed from price/volume only (no external data vendors required)

---

*Research conducted Jan–Mar 2026. Not investment advice.*
