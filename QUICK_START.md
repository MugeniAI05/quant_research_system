# Quick Start Guide

This guide walks you through setting up and running the Production Quantitative Research System.

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/quant_research_system.git
cd quant_research_system
```
## 2. Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

## 3. Install Dependencies
pip install -r requirements.txt

## 4. Run Your First Analysis

**Option A: Run from Python**

```bash
from src.main_pipeline import analyze_ticker

# Analyze NVIDIA
report, results = analyze_ticker("NVDA", period="2y")

print(report)
print("Best factor:", results["best_factor"])
print("Sharpe ratio:", results["backtest"]["metrics"]["sharpe_ratio"])
```
**Option B: Run from Command Line**

```bash
python -m src.main_pipeline NVDA
```

## 5. Verify Installation (30-Second Test)
If this runs without errors, the system is configured correctly.

```bash
import numpy as np
import pandas as pd
from src.factors.factor_validation import FactorValidator

# Create synthetic test data
dates = pd.date_range("2020-01-01", periods=300)
prices = pd.Series(
    100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)),
    index=dates
)
signal = pd.Series(np.random.randn(300), index=dates)

forward_returns = prices.pct_change(5).shift(-5)

validator = FactorValidator()
ic_result = validator.calculate_ic(signal, forward_returns)

print("IC:", round(ic_result.ic, 3))
print("p-value:", round(ic_result.p_value, 3))
```
## Common Issues
1. Module Not Found

Make sure you are running commands from the project root directory:
```bash
cd quant_research_system
```
2. No Data Returned

Confirm the ticker symbol is valid.

Ensure you have an active internet connection.

3. Insufficient Data Error

Use a longer historical period:

```bash
analyze_ticker("AAPL", period="2y")
```
## Next Steps

Review README.md for full methodology and system design.

Read SYSTEM_SUMMARY.md for architectural decisions.

Explore the src/ directory to understand the modular pipeline structure.

Extend the system with new factors or validation criteria.
