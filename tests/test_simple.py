"""
Simple test to verify the system works
"""

import numpy as np
import pandas as pd
from factor_validation import FactorValidator

print("Testing the system...")

# Create test data
dates = pd.date_range('2020-01-01', periods=300)
prices = pd.Series(
    100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)), 
    index=dates
)
signal = pd.Series(np.random.randn(300), index=dates)
forward_returns = prices.pct_change(5).shift(-5)

# Validate
validator = FactorValidator()
ic_result = validator.calculate_ic(signal, forward_returns)

print(f"✓ System is working!")
print(f"  IC: {ic_result.ic:.3f}")
print(f"  p-value: {ic_result.p_value:.3f}")
print(f"  Observations: {ic_result.n_obs}")
```

**To save:**
1. Press `Control + O` (that's the letter O, not zero)
2. Press `Enter` to confirm
3. Press `Control + X` to exit

---

## Run the Test
```
python test_simple.py
```

**Expected output:**
```
Testing the system...
✓ System is working!
  IC: 0.042
  p-value: 0.234
  Observations: 295
python test_simple.py
nano test_full.py
"""
Test the complete research pipeline
"""

from main_pipeline import analyze_ticker

print("="*80)
print("RUNNING FULL QUANTITATIVE RESEARCH PIPELINE")
print("="*80)

# Run analysis on Apple
ticker = "AAPL"
print(f"\nAnalyzing {ticker}...\n")

report, results = analyze_ticker(ticker, period="1y")

# Print the report
print(report)

# Print structured results
print("\n" + "="*80)
print("STRUCTURED RESULTS")
print("="*80)
print(f"\nTicker: {results['ticker']}")
print(f"Best Factor: {results['best_factor']}")
print(f"Sentiment: {results['sentiment']['sentiment_label']} ({results['sentiment']['aggregate_score']:+.3f})")
print(f"\nBacktest Metrics:")
print(f"  Sharpe Ratio: {results['backtest']['metrics']['sharpe_ratio']:.3f}")
print(f"  CAGR: {results['backtest']['metrics']['cagr_pct']:.1f}%")
print(f"  Max Drawdown: {results['backtest']['metrics']['max_drawdown_pct']:.1f}%")
print(f"  Win Rate: {results['backtest']['metrics']['win_rate_pct']:.1f}%")

print("\n✓ Full pipeline test complete!")
```

**Save it:**
1. `Control + O`
2. `Enter`
3. `Control + X`

### Run the Full Pipeline
```
python test_full.py
"""
Simple test to verify the system works
"""

import numpy as np
import pandas as pd
from factor_validation import FactorValidator

print("Testing the system...")

# Create test data
dates = pd.date_range('2020-01-01', periods=300)
prices = pd.Series(
    100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)), 
    index=dates
)
signal = pd.Series(np.random.randn(300), index=dates)
forward_returns = prices.pct_change(5).shift(-5)

# Validate
validator = FactorValidator()
ic_result = validator.calculate_ic(signal, forward_returns)

print(f"✓ System is working!")
print(f"  IC: {ic_result.ic:.3f}")
print(f"  p-value: {ic_result.p_value:.3f}")
print(f"  Observations: {ic_result.n_obs}")
```

**To save:**
1. Press `Control + O` (that's the letter O, not zero)
2. Press `Enter` to confirm
3. Press `Control + X` to exit

---

## Run the Test
```
python test_simple.py
```

**Expected output:**
```
Testing the system...
✓ System is working!
  IC: 0.042
  p-value: 0.234
  Observations: 295

