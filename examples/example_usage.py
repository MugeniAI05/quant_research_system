"""
Example Usage Script
Demonstrates how to use the quantitative research system
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from main_pipeline import QuantResearchPipeline
from config import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def example_1_basic_analysis():
    """Example 1: Basic single-ticker analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Analysis")
    print("="*80 + "\n")
    
    # Create pipeline
    pipeline = QuantResearchPipeline()
    
    # Run analysis
    report, results = pipeline.run_complete_analysis(
        ticker="NVDA",
        period="2y",
        top_n_factors=3
    )
    
    # Print report
    print(report)
    
    # Save results
    pipeline.save_results("NVDA", report, results)
    
    return results


def example_2_custom_configuration():
    """Example 2: Analysis with custom configuration"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Configuration")
    print("="*80 + "\n")
    
    # Modify configuration
    config.backtest.holding_period_days = 10  # Longer holding period
    config.backtest.transaction_cost_bps = 5.0  # Lower costs (institutional rates)
    config.backtest.min_ic = 0.03  # Higher IC threshold
    
    print("Custom Configuration:")
    print(f"  Holding Period: {config.backtest.holding_period_days} days")
    print(f"  Transaction Cost: {config.backtest.transaction_cost_bps} bps")
    print(f"  Min IC: {config.backtest.min_ic}")
    
    # Run analysis
    pipeline = QuantResearchPipeline()
    report, results = pipeline.run_complete_analysis(
        ticker="AAPL",
        period="1y",
        top_n_factors=5
    )
    
    print(f"\nResults:")
    print(f"  Best Factor: {results['best_factor']}")
    print(f"  Sentiment: {results['sentiment']['sentiment_label']}")
    print(f"  Sharpe: {results['backtest']['metrics']['sharpe_ratio']:.3f}")
    
    return results


def example_3_component_usage():
    """Example 3: Using individual components"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Component-Level Usage")
    print("="*80 + "\n")
    
    from data_fetcher import MarketDataFetcher, NewsDataFetcher
    from factor_engineering import FactorEngine
    from sentiment_analysis import SentimentScorer
    from factor_validation import FactorValidator
    from backtest_engine import VectorBacktester
    
    # 1. Fetch data
    print("1. Fetching data...")
    market_fetcher = MarketDataFetcher()
    news_fetcher = NewsDataFetcher()
    
    market_data = market_fetcher.fetch("TSLA", period="1y")
    news_data = news_fetcher.fetch("TSLA")
    
    print(f"   Got {len(market_data.prices)} days of price data")
    print(f"   Got {len(news_data.headlines)} news headlines")
    
    # 2. Compute factors
    print("\n2. Computing factors...")
    engine = FactorEngine()
    factors = engine.compute_all_factors(market_data.prices, market_data.volumes)
    
    print(f"   Computed {len(factors)} factors")
    print(f"   Factor families: {list(set(f.family for f in factors.values()))}")
    
    # 3. Analyze sentiment
    print("\n3. Analyzing sentiment...")
    scorer = SentimentScorer()
    sentiment = scorer.aggregate_sentiment(news_data.headlines, "TSLA")
    
    print(f"   Sentiment: {sentiment.sentiment_label} ({sentiment.aggregate_score:+.3f})")
    
    # 4. Validate a factor
    print("\n4. Validating factor...")
    validator = FactorValidator()
    
    mom_factor = factors['mom_20d']
    validation = validator.validate_factor(
        factor_name='mom_20d',
        factor_values=mom_factor.values,
        prices=market_data.prices,
        horizon=5
    )
    
    print(f"   IC: {validation.ic_analysis.ic:+.4f}")
    print(f"   p-value: {validation.ic_analysis.p_value:.4f}")
    print(f"   Viable: {validation.is_viable}")
    
    # 5. Backtest
    print("\n5. Running backtest...")
    backtester = VectorBacktester()
    
    result = backtester.backtest_signal(
        prices=market_data.prices,
        signal=mom_factor.values,
        signal_threshold=0.0
    )
    
    print(f"   Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
    print(f"   CAGR: {result.metrics.cagr*100:.2f}%")
    print(f"   Max Drawdown: {result.metrics.max_drawdown*100:.2f}%")
    print(f"   Total Trades: {result.metrics.total_trades}")
    
    return result


def example_4_multi_ticker_comparison():
    """Example 4: Compare multiple tickers"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-Ticker Comparison")
    print("="*80 + "\n")
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    results_summary = []
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        
        pipeline = QuantResearchPipeline()
        report, results = pipeline.run_complete_analysis(
            ticker=ticker,
            period="1y",
            top_n_factors=1
        )
        
        results_summary.append({
            'ticker': ticker,
            'factor': results['best_factor'],
            'sharpe': results['backtest']['metrics']['sharpe_ratio'],
            'cagr': results['backtest']['metrics']['cagr_pct'],
            'sentiment': results['sentiment']['sentiment_label']
        })
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Ticker':<10} {'Best Factor':<20} {'Sharpe':<10} {'CAGR':<10} {'Sentiment':<10}")
    print("-"*80)
    
    for r in results_summary:
        print(f"{r['ticker']:<10} {r['factor']:<20} {r['sharpe']:<10.3f} {r['cagr']:<10.1f} {r['sentiment']:<10}")
    
    return results_summary


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("QUANTITATIVE RESEARCH SYSTEM - EXAMPLES")
    print("="*80)
    
    # Run examples
    try:
        # Example 1: Basic
        example_1_basic_analysis()
        
        # Example 2: Custom config
        example_2_custom_configuration()
        
        # Example 3: Component usage
        example_3_component_usage()
        
        # Example 4: Multi-ticker (comment out if too slow)
        # example_4_multi_ticker_comparison()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
