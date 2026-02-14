"""
Main Orchestration System
Production-ready quantitative research pipeline
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Import all modules
from config import config
from data_fetcher import MarketDataFetcher, NewsDataFetcher, DataValidator, MarketData, NewsData
from factor_engineering import FactorEngine
from sentiment_analysis import SentimentScorer, SentimentFactorBuilder, SentimentResult
from factor_validation import FactorValidator, MultiFactorValidator, FactorValidationReport
from backtest_engine import VectorBacktester, BacktestResult
from reporting import ResearchReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantResearchPipeline:
    """
    Complete quantitative research pipeline
    
    Workflow:
    1. Data Acquisition (prices + news)
    2. Factor Engineering (compute all factors)
    3. Sentiment Analysis (score news sentiment)
    4. Factor Validation (statistical testing)
    5. Backtest (validate top factors)
    6. Report Generation (professional output)
    """
    
    def __init__(self):
        self.market_fetcher = MarketDataFetcher()
        self.news_fetcher = NewsDataFetcher()
        self.factor_engine = FactorEngine()
        self.sentiment_scorer = SentimentScorer()
        self.factor_validator = MultiFactorValidator()
        self.backtester = VectorBacktester()
        
        # Storage
        self.market_data: Optional[MarketData] = None
        self.news_data: Optional[NewsData] = None
        self.sentiment_result: Optional[SentimentResult] = None
        self.validation_reports: Dict[str, FactorValidationReport] = {}
        self.backtest_results: Dict[str, BacktestResult] = {}
    
    def run_complete_analysis(
        self,
        ticker: str,
        period: str = "2y",
        top_n_factors: int = 3
    ) -> Tuple[str, Dict]:
        """
        Run complete analysis pipeline
        
        Args:
            ticker: Stock ticker
            period: Time period for analysis
            top_n_factors: Number of top factors to backtest
            
        Returns:
            (research_report_text, results_dict)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING QUANT RESEARCH PIPELINE FOR {ticker}")
        logger.info(f"{'='*80}\n")
        
        # Step 1: Data Acquisition
        logger.info("STEP 1: Data Acquisition")
        success = self._acquire_data(ticker, period)
        if not success:
            return self._error_report("Data acquisition failed"), {}
        
        # Step 2: Factor Engineering
        logger.info("\nSTEP 2: Factor Engineering")
        factors = self._compute_factors()
        logger.info(f"Computed {len(factors)} factors")
        
        # Step 3: Sentiment Analysis
        logger.info("\nSTEP 3: Sentiment Analysis")
        self._analyze_sentiment(ticker)
        
        # Step 4: Add Sentiment Factors
        logger.info("\nSTEP 4: Building Sentiment Factors")
        sentiment_factors = self._build_sentiment_factors()
        factors.update(sentiment_factors)
        logger.info(f"Total factors including sentiment: {len(factors)}")
        
        # Step 5: Factor Validation
        logger.info("\nSTEP 5: Statistical Validation")
        self.validation_reports = self._validate_factors(factors)
        
        # Step 6: Rank and Select Top Factors
        logger.info("\nSTEP 6: Ranking Factors")
        top_factors = self._select_top_factors(top_n_factors)
        
        if not top_factors:
            logger.warning("No viable factors found")
            return self._error_report("No viable factors passed validation"), {}
        
        # Step 7: Backtest Top Factors
        logger.info(f"\nSTEP 7: Backtesting Top {len(top_factors)} Factors")
        self._backtest_factors(top_factors, factors)
        
        # Step 8: Select Best Factor
        best_factor_name = self._select_best_factor()
        
        if not best_factor_name:
            return self._error_report("All backtests failed"), {}
        
        logger.info(f"\nBest factor: {best_factor_name}")
        
        # Step 9: Generate Report
        logger.info("\nSTEP 9: Generating Research Report")
        report = self._generate_report(ticker, best_factor_name)
        
        # Compile results
        results = {
            'ticker': ticker,
            'best_factor': best_factor_name,
            'sentiment': self.sentiment_result.to_dict(),
            'validation': self.validation_reports[best_factor_name].to_dict(),
            'backtest': self.backtest_results[best_factor_name].to_dict(),
            'all_factors_tested': len(factors),
            'viable_factors': len(self.factor_validator.get_viable_factors())
        }
        
        logger.info(f"\n{'='*80}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'='*80}\n")
        
        return report, results
    
    def _acquire_data(self, ticker: str, period: str) -> bool:
        """Acquire market and news data"""
        # Fetch market data
        self.market_data = self.market_fetcher.fetch(ticker, period)
        
        if self.market_data is None:
            logger.error(f"Failed to fetch market data for {ticker}")
            return False
        
        # Validate market data
        validator = DataValidator()
        is_valid, issues = validator.validate_market_data(self.market_data)
        
        if not is_valid:
            logger.warning(f"Data quality issues: {issues}")
            # Continue anyway if we have enough data
            if len(self.market_data.prices) < config.backtest.min_observations:
                return False
        
        # Fetch news data
        self.news_data = self.news_fetcher.fetch(ticker)
        
        logger.info(f"Data acquired: {len(self.market_data.prices)} days, {len(self.news_data.headlines)} headlines")
        return True
    
    def _compute_factors(self) -> Dict[str, pd.Series]:
        """Compute all technical factors"""
        all_factors = self.factor_engine.compute_all_factors(
            self.market_data.prices,
            self.market_data.volumes
        )
        
        # Convert to Series dict
        factor_series = {name: factor.values for name, factor in all_factors.items()}
        return factor_series
    
    def _analyze_sentiment(self, ticker: str):
        """Analyze sentiment from news"""
        self.sentiment_result = self.sentiment_scorer.aggregate_sentiment(
            self.news_data.headlines,
            ticker
        )
        
        logger.info(f"Sentiment: {self.sentiment_result.sentiment_label} ({self.sentiment_result.aggregate_score:+.3f})")
    
    def _build_sentiment_factors(self) -> Dict[str, pd.Series]:
        """Build factors combining sentiment and price"""
        sentiment_factors = SentimentFactorBuilder.build_all_sentiment_factors(
            self.sentiment_result.aggregate_score,
            self.market_data.prices
        )
        
        return sentiment_factors
    
    def _validate_factors(self, factors: Dict[str, pd.Series]) -> Dict[str, FactorValidationReport]:
        """Validate all factors statistically"""
        validation_reports = self.factor_validator.validate_all(
            factors,
            self.market_data.prices,
            horizon=config.backtest.holding_period_days,
            run_decay=False  # Skip decay analysis for speed
        )
        
        # Print summary
        print(self.factor_validator.summary_report())
        
        return validation_reports
    
    def _select_top_factors(self, top_n: int) -> List[str]:
        """Select top N viable factors"""
        viable = self.factor_validator.get_viable_factors()
        
        if not viable:
            return []
        
        # Rank by IC
        ranked_df = self.factor_validator.rank_factors(by='ic')
        
        # Filter to viable only
        ranked_viable = ranked_df[ranked_df['is_viable']].head(top_n)
        
        top_factors = ranked_viable['factor'].tolist()
        
        logger.info(f"Top factors selected: {top_factors}")
        return top_factors
    
    def _backtest_factors(self, factor_names: List[str], factors: Dict[str, pd.Series]):
        """Backtest selected factors"""
        for name in factor_names:
            logger.info(f"Backtesting: {name}")
            
            result = self.backtester.backtest_signal(
                self.market_data.prices,
                factors[name],
                signal_threshold=0.0
            )
            
            self.backtest_results[name] = result
            
            logger.info(f"  Sharpe: {result.metrics.sharpe_ratio:.3f}, "
                       f"CAGR: {result.metrics.cagr*100:.1f}%, "
                       f"MaxDD: {result.metrics.max_drawdown*100:.1f}%")
    
    def _select_best_factor(self) -> Optional[str]:
        """Select factor with best Sharpe ratio"""
        if not self.backtest_results:
            return None
        
        best_name = None
        best_sharpe = -np.inf
        
        for name, result in self.backtest_results.items():
            if result.metrics.sharpe_ratio > best_sharpe:
                best_sharpe = result.metrics.sharpe_ratio
                best_name = name
        
        return best_name
    
    def _generate_report(self, ticker: str, factor_name: str) -> str:
        """Generate professional research report"""
        
        # Find rank of this factor
        ranked = self.factor_validator.rank_factors(by='ic')
        rank = ranked[ranked['factor'] == factor_name].index[0] + 1 if len(ranked) > 0 else None
        
        report = ResearchReportGenerator.generate_full_report(
            ticker=ticker,
            market_data=self.market_data,
            news_data=self.news_data,
            sentiment_result=self.sentiment_result,
            validation_report=self.validation_reports[factor_name],
            backtest_result=self.backtest_results[factor_name],
            factor_rank=rank
        )
        
        return report
    
    def _error_report(self, reason: str) -> str:
        """Generate error report"""
        return f"""
{'='*80}
ERROR IN QUANT RESEARCH PIPELINE
{'='*80}

Reason: {reason}

The analysis could not be completed. Please check:
1. Ticker symbol is valid
2. Sufficient historical data is available
3. Network connection is stable

{'='*80}
"""
    
    def save_results(self, ticker: str, report: str, results: Dict):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save report
        report_filename = f"/mnt/user-data/outputs/research_report_{ticker}_{timestamp}.txt"
        ResearchReportGenerator.save_report(report, report_filename)
        
        # Save equity curve if available
        if self.backtest_results:
            best_factor = list(self.backtest_results.keys())[0]
            equity_df = pd.DataFrame({
                'date': self.backtest_results[best_factor].equity_curve.index,
                'equity': self.backtest_results[best_factor].equity_curve.values
            })
            
            equity_filename = f"/mnt/user-data/outputs/equity_curve_{ticker}_{timestamp}.csv"
            equity_df.to_csv(equity_filename, index=False)
            logger.info(f"Equity curve saved to {equity_filename}")


# Convenience function for quick analysis
def analyze_ticker(ticker: str, period: str = "2y") -> Tuple[str, Dict]:
    """
    Quick analysis of a ticker
    
    Args:
        ticker: Stock ticker symbol
        period: Historical period (e.g., '1y', '2y', '5y')
        
    Returns:
        (research_report, results_dict)
    """
    pipeline = QuantResearchPipeline()
    return pipeline.run_complete_analysis(ticker, period)


if __name__ == "__main__":
    # Example usage
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    
    print(f"Analyzing {ticker}...")
    report, results = analyze_ticker(ticker, period="2y")
    
    print("\n" + report)
    
    print(f"\nResults summary:")
    print(f"  Best factor: {results.get('best_factor')}")
    print(f"  Sentiment: {results.get('sentiment', {}).get('sentiment_label')}")
    print(f"  Sharpe: {results.get('backtest', {}).get('metrics', {}).get('sharpe_ratio')}")
