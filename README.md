# Production Quantitative Research System

A production-ready quantitative research pipeline that implements industry-standard practices for systematic trading research. Built to demonstrate the technical rigor expected at firms like Two Sigma and SIG.

## 🎯 Key Features

### Statistical Rigor
- **Information Coefficient (IC) Analysis**: Measures correlation between signals and forward returns
- **Statistical Significance Testing**: T-tests and p-values to validate predictive power  
- **Quintile Analysis**: Verifies monotonicity of returns across factor buckets
- **Turnover Analysis**: Quantifies trading costs and holding periods

### Realistic Backtesting
- **No Look-Ahead Bias**: Uses only data available at decision time
- **Transaction Costs**: 10 bps commissions + 5 bps slippage per trade
- **Proper Position Sizing**: Risk-based allocation with maximum limits
- **Comprehensive Metrics**: Sharpe, Calmar, IC, Win Rate, Max Drawdown

### Multi-Modal Analysis
- **Technical Factors**: 30+ momentum, volatility, and reversal indicators
- **Sentiment Analysis**: Quantitative news scoring using financial lexicons
- **Factor Validation**: Pre-backtest filtering to avoid overfitting
- **Automated Reporting**: Professional research notes with recommendations

## 📁 Project Structure

```
quant_research_system/
├── config.py                  # Configuration and constants
├── data_fetcher.py           # Market data and news acquisition
├── factor_engineering.py     # Technical factor computation
├── sentiment_analysis.py     # News sentiment scoring
├── factor_validation.py      # Statistical testing
├── backtest_engine.py        # Realistic backtest simulation
├── reporting.py              # Research report generation
├── main_pipeline.py          # Main orchestration
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy pandas scipy yfinance duckduckgo-search
```

### Basic Usage

```python
from main_pipeline import analyze_ticker

# Run complete analysis
report, results = analyze_ticker("NVDA", period="2y")

# Print report
print(report)

# Access structured results
print(f"Best Factor: {results['best_factor']}")
print(f"Sharpe Ratio: {results['backtest']['metrics']['sharpe_ratio']}")
```

### Command Line Usage

```bash
python main_pipeline.py AAPL
```

## 📊 Sample Output

```
================================================================================
QUANTITATIVE RESEARCH NOTE
================================================================================

Ticker:    NVDA
Generated: 2024-02-14 12:00

================================================================================

1. DATA SUMMARY
--------------------------------------------------------------------------------
Market Data:
  - Observations:  504 trading days
  - Price Range:   $45.23 - $892.45
  
News Data:
  - Headlines:     5
  - Source:        Live Search

2. SENTIMENT ANALYSIS
--------------------------------------------------------------------------------
Aggregate Sentiment:
  - Score:         +0.347 (range: -1 to +1)
  - Classification: POSITIVE
  
3. FACTOR VALIDATION (Rank #1)
--------------------------------------------------------------------------------
Selected Factor: mom_20d_zscore

Information Coefficient Analysis:
  - IC (Spearman):  +0.0421
  - T-Statistic:    +3.156
  - P-Value:        0.0018
  - Significance:   ✓ YES (p < 0.05)

Quintile Analysis:
  - Monotonicity:  ✓ Returns are monotonic
  - Q1-Q5 Spread:  4.32%

4. BACKTEST RESULTS
--------------------------------------------------------------------------------
Overall Rating: STRONG

Risk-Adjusted Performance:
  - Sharpe Ratio:        1.234
  - CAGR:               +18.45%
  - Maximum Drawdown:    -12.3%

5. INVESTMENT THESIS & RECOMMENDATIONS
--------------------------------------------------------------------------------
Signal Strength: BUY
Confidence Level: MODERATE

Key Findings:
  ✓ Factor shows positive predictive power (IC = 0.042)
  ✓ Relationship is statistically significant (p = 0.002)
  ✓ Risk-adjusted returns are acceptable (Sharpe = 1.23)
```

## 🔬 Technical Details

### Factor Families

**Momentum Factors** (9 factors)
- Simple momentum (5d, 10d, 20d, 60d windows)
- Momentum acceleration
- Time-series momentum
- Z-score normalized momentum
- Exponential momentum

**Volatility Factors** (7 factors)  
- Realized volatility (multiple windows)
- Volatility ratios
- Downside volatility
- Volatility of volatility

**Reversal Factors** (7 factors)
- Distance from moving averages
- RSI (Relative Strength Index)
- Bollinger Band position
- Mean reversion z-scores
- Rate of change oscillator

**Sentiment Factors** (4 factors)
- Raw sentiment score
- Sentiment-price divergence
- Sentiment momentum
- Sentiment surprise

### Validation Pipeline

1. **IC Calculation**
   - Spearman rank correlation (robust to outliers)
   - T-statistic for significance testing
   - Minimum IC threshold: 0.02

2. **Quintile Analysis**
   - Divides factor into 5 buckets
   - Checks for monotonic returns
   - Calculates Q1-Q5 spread

3. **Turnover Analysis**
   - Counts position flips
   - Estimates transaction costs
   - Maximum turnover: 50% per period

4. **Backtest Validation**
   - 5-day holding period
   - 15 bps round-trip costs
   - Minimum Sharpe: 0.5

## 🎓 Interview Talking Points

### On Backtesting
> "I prevent look-ahead bias by only using forward returns calculated from time t to t+h, ensuring no future information leaks into past decisions. Transaction costs are modeled at 10 bps for commissions plus 5 bps for slippage, totaling 15 bps per round trip."

### On Factor Validation
> "I validate factors using Information Coefficient analysis, which measures the Spearman correlation between factor values and forward returns. A factor must have an IC above 0.02 with statistical significance (p < 0.05) to be considered viable."

### On Overfitting Prevention
> "I use multiple safeguards: statistical significance testing with p-values, minimum sample size requirements (200+ observations), quintile analysis to verify monotonicity, and accept/reject thresholds before backtesting to avoid data mining."

### On Sentiment
> "Rather than qualitative analysis, I use a Loughran-McDonald financial lexicon to score each headline numerically from -1 to +1. The scores are then aggregated using exponentially weighted averaging to give more weight to recent news."

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Backtest settings
transaction_cost_bps = 10.0  # Per-trade cost
slippage_bps = 5.0           # Market impact
holding_period_days = 5       # Position hold time

# Validation thresholds
min_ic = 0.02                # Minimum IC
max_turnover = 0.5           # Maximum turnover rate
min_sharpe_ratio = 0.5       # Minimum acceptable Sharpe

# Factor parameters
momentum_windows = [5, 10, 20, 60]
volatility_windows = [5, 10, 20, 60]
```

## 📈 Advanced Usage

### Custom Factor Development

```python
from factor_engineering import FactorEngine
from data_fetcher import fetch_market_data

# Fetch data
data = fetch_market_data("AAPL", period="2y")

# Compute factors
engine = FactorEngine()
factors = engine.compute_all_factors(data.prices, data.volumes)

# Access specific factor
momentum_5d = engine.get_factor('mom_5d')
```

### Factor Validation

```python
from factor_validation import FactorValidator

# Validate a single factor
validator = FactorValidator()
report = validator.validate_factor(
    factor_name='mom_20d',
    factor_values=factor_series,
    prices=price_series,
    horizon=5
)

print(report.recommendation)
print(f"IC: {report.ic_analysis.ic:.3f}")
print(f"Viable: {report.is_viable}")
```

### Custom Backtesting

```python
from backtest_engine import VectorBacktester

backtester = VectorBacktester()
result = backtester.backtest_signal(
    prices=price_series,
    signal=factor_series,
    signal_threshold=0.0,
    position_size=1.0
)

print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
print(f"Max DD: {result.metrics.max_drawdown*100:.1f}%")
```

## 🧪 Testing

```python
# Run validation tests
from factor_validation import FactorValidator
import numpy as np
import pandas as pd

# Create synthetic data
dates = pd.date_range('2020-01-01', periods=500)
prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(500) * 0.01)), index=dates)
signal = pd.Series(np.random.randn(500), index=dates)

# Validate
validator = FactorValidator()
ic_result = validator.calculate_ic(signal, prices.pct_change(5).shift(-5))

print(f"IC: {ic_result.ic:.3f}, p-value: {ic_result.p_value:.3f}")
```

## 📝 Best Practices

1. **Always validate before backtesting**
   - Run IC analysis first
   - Check statistical significance  
   - Verify turnover is acceptable

2. **Use realistic assumptions**
   - Include transaction costs
   - Model slippage
   - Account for market impact

3. **Prevent overfitting**
   - Use out-of-sample testing
   - Require statistical significance
   - Limit parameter optimization

4. **Document everything**
   - Log all assumptions
   - Record validation results
   - Maintain audit trail

## 🚫 Common Pitfalls to Avoid

❌ **Using future data in the past** (look-ahead bias)
✅ Only use forward returns calculated from time t onward

❌ **Ignoring transaction costs**
✅ Include 10-15 bps per trade minimum

❌ **Over-optimizing on historical data**
✅ Validate on out-of-sample periods

❌ **Trading signals with IC < 0.02**
✅ Set minimum quality thresholds

❌ **High turnover strategies without cost analysis**
✅ Calculate turnover impact explicitly

## 📚 References

- **Backtesting**: "Advances in Financial Machine Learning" - Marcos López de Prado
- **Factor Validation**: "Active Portfolio Management" - Grinold & Kahn  
- **Sentiment**: Loughran-McDonald Financial Sentiment Dictionary
- **Risk Management**: "Quantitative Trading" - Ernest Chan

## 📄 License

This project is intended for educational and research purposes. 
Not financial advice. Trade at your own risk.

## 🤝 Contributing

Suggestions for improvements:
1. Add machine learning factor combination
2. Implement walk-forward optimization
3. Add multi-asset support
4. Include regime detection
5. Build portfolio construction module

## ✉️ Contact

For questions about implementation or methodology, please open an issue on GitHub.

---

**Note**: This system is designed to demonstrate quantitative research capabilities 
for interviews and educational purposes. Always validate strategies on paper trading 
before risking real capital.
