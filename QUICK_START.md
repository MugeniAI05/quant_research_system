# Quick Start Guide

## Installation (5 minutes)

### Step 1: Download Files
All files are in `/mnt/user-data/outputs/`. Download these files to your local machine:

```
Core System Files (Required):
✓ config.py
✓ data_fetcher.py
✓ factor_engineering.py
✓ sentiment_analysis.py
✓ factor_validation.py
✓ backtest_engine.py
✓ reporting.py
✓ main_pipeline.py

Examples & Documentation:
✓ example_usage.py
✓ requirements.txt
✓ README.md
✓ SYSTEM_SUMMARY.md
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas scipy yfinance duckduckgo-search
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3: Run Your First Analysis

```python
from main_pipeline import analyze_ticker

# Analyze NVIDIA
report, results = analyze_ticker("NVDA", period="2y")

# Print the research report
print(report)

# Access results
print(f"\nBest Factor: {results['best_factor']}")
print(f"Sharpe Ratio: {results['backtest']['metrics']['sharpe_ratio']:.2f}")
```

Or from command line:
```bash
python main_pipeline.py NVDA
```

## File Descriptions

| File | Size | Purpose |
|------|------|---------|
| **config.py** | 3.4K | Central configuration - transaction costs, thresholds, etc. |
| **data_fetcher.py** | 12K | Fetch market data (yfinance) and news (DuckDuckGo) |
| **factor_engineering.py** | 13K | Compute 30+ technical factors (momentum, vol, reversal) |
| **sentiment_analysis.py** | 13K | Quantitative sentiment scoring with financial lexicon |
| **factor_validation.py** | 17K | Statistical testing (IC, quintiles, turnover) |
| **backtest_engine.py** | 12K | Realistic backtest with transaction costs |
| **reporting.py** | 11K | Generate professional research notes |
| **main_pipeline.py** | 13K | Main orchestrator - ties everything together |
| **example_usage.py** | 6.4K | Usage examples and demos |
| **README.md** | 11K | Complete documentation |
| **SYSTEM_SUMMARY.md** | 12K | What was built and why |

## 30-Second Test

```python
# Test that everything works
import numpy as np
import pandas as pd
from factor_validation import FactorValidator

# Create test data
dates = pd.date_range('2020-01-01', periods=300)
prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)), index=dates)
signal = pd.Series(np.random.randn(300), index=dates)
forward_returns = prices.pct_change(5).shift(-5)

# Validate
validator = FactorValidator()
ic_result = validator.calculate_ic(signal, forward_returns)

print(f"✓ System working! IC: {ic_result.ic:.3f}, p-value: {ic_result.p_value:.3f}")
```

## Common Use Cases

### Use Case 1: Analyze Single Stock
```python
from main_pipeline import QuantResearchPipeline

pipeline = QuantResearchPipeline()
report, results = pipeline.run_complete_analysis("AAPL", period="1y")
print(report)
```

### Use Case 2: Custom Configuration
```python
from config import config

# Modify settings
config.backtest.holding_period_days = 10
config.backtest.transaction_cost_bps = 5.0

# Then run analysis
from main_pipeline import analyze_ticker
report, results = analyze_ticker("MSFT")
```

### Use Case 3: Test Individual Components
```python
from data_fetcher import fetch_market_data
from factor_engineering import FactorEngine

# Fetch data
data = fetch_market_data("GOOGL", period="2y")

# Compute factors
engine = FactorEngine()
factors = engine.compute_all_factors(data.prices, data.volumes)

# List available factors
print(f"Available factors: {engine.list_factors()}")
```

### Use Case 4: Validate Custom Factor
```python
from factor_validation import FactorValidator
import pandas as pd

# Your custom factor
my_factor = pd.Series([...])  # Your values
prices = pd.Series([...])     # Price data

# Validate
validator = FactorValidator()
report = validator.validate_factor(
    factor_name='my_custom_factor',
    factor_values=my_factor,
    prices=prices,
    horizon=5
)

print(report.recommendation)
```

## Troubleshooting

### Issue: "No module named 'yfinance'"
**Solution:** Install dependencies
```bash
pip install yfinance duckduckgo-search
```

### Issue: "No data found for ticker"
**Solution:** Check ticker symbol is correct
```python
# Wrong
analyze_ticker("NVIDA")  # Typo

# Right
analyze_ticker("NVDA")
```

### Issue: DuckDuckGo search blocked
**Solution:** The system has fallback news data built-in. It will automatically use cached data if live search fails.

### Issue: "Insufficient data"
**Solution:** Use longer time period
```python
# Too short
analyze_ticker("AAPL", period="1mo")  # Only 20 days

# Better
analyze_ticker("AAPL", period="1y")   # 252 days
```

## Expected Output

When you run the analysis, you'll see:

```
================================================================================
QUANTITATIVE RESEARCH NOTE
================================================================================

Ticker:    NVDA
Generated: 2024-02-14 12:00

1. DATA SUMMARY
--------------------------------------------------------------------------------
Market Data:
  - Observations:  504 trading days
  - Price Range:   $45.23 - $892.45

2. SENTIMENT ANALYSIS
--------------------------------------------------------------------------------
Aggregate Sentiment:
  - Score:         +0.347
  - Classification: POSITIVE

3. FACTOR VALIDATION (Rank #1)
--------------------------------------------------------------------------------
Selected Factor: mom_20d_zscore
  - IC (Spearman):  +0.0421
  - T-Statistic:    +3.156
  - P-Value:        0.0018
  - Significance:   ✓ YES

4. BACKTEST RESULTS
--------------------------------------------------------------------------------
  - Sharpe Ratio:   1.234
  - CAGR:          +18.45%
  - Max Drawdown:   -12.3%

5. INVESTMENT THESIS & RECOMMENDATIONS
--------------------------------------------------------------------------------
Signal Strength: BUY
Confidence Level: MODERATE
```


## Support

If you have questions:
1. Check README.md first
2. Review SYSTEM_SUMMARY.md for design decisions
3. Look at example_usage.py for patterns
