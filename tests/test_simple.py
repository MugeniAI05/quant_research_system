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
