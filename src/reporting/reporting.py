"""
Reporting Module
Generates professional research notes and analysis reports
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

from data_fetcher import MarketData, NewsData
from factor_validation import FactorValidationReport
from backtest_engine import BacktestResult
from sentiment_analysis import SentimentResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchReportGenerator:
    """Generate professional research reports"""
    
    @staticmethod
    def generate_header(ticker: str) -> str:
        """Generate report header"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        header = f"""
{'=' * 80}
QUANTITATIVE RESEARCH NOTE
{'=' * 80}

Ticker:    {ticker}
Generated: {now}
System:    Production Quant Research Lab v2.0

{'=' * 80}
"""
        return header
    
    @staticmethod
    def format_data_summary(market_data: MarketData, news_data: NewsData) -> str:
        """Format data acquisition summary"""
        section = f"""
1. DATA SUMMARY
{'-' * 80}

Market Data:
  - Ticker:        {market_data.ticker}
  - Period:        {market_data.start_date} to {market_data.end_date}
  - Observations:  {len(market_data.prices)} trading days
  - Price Range:   ${market_data.prices.min():.2f} - ${market_data.prices.max():.2f}
  - Latest Price:  ${market_data.prices.iloc[-1]:.2f}

News Data:
  - Headlines:     {len(news_data.headlines)}
  - Source:        {news_data.source}
  - Fetch Time:    {news_data.fetch_time.strftime("%Y-%m-%d %H:%M")}

"""
        return section
    
    @staticmethod
    def format_sentiment_analysis(sentiment_result: SentimentResult) -> str:
        """Format sentiment analysis section"""
        section = f"""
2. SENTIMENT ANALYSIS
{'-' * 80}

Aggregate Sentiment:
  - Score:         {sentiment_result.aggregate_score:+.3f} (range: -1 to +1)
  - Classification: {sentiment_result.sentiment_label}
  - Confidence:    {abs(sentiment_result.aggregate_score):.1%}

Headline Breakdown:
  - Positive:      {sentiment_result.n_positive} ({sentiment_result.n_positive/len(sentiment_result.headlines)*100:.1f}%)
  - Negative:      {sentiment_result.n_negative} ({sentiment_result.n_negative/len(sentiment_result.headlines)*100:.1f}%)
  - Neutral:       {sentiment_result.n_neutral} ({sentiment_result.n_neutral/len(sentiment_result.headlines)*100:.1f}%)

Statistical Properties:
  - Std Deviation: {sentiment_result.score_std:.3f}
  - Score Range:   {sentiment_result.score_range:.3f}

Sample Headlines:
"""
        # Add sample headlines with scores
        for i, (headline, score) in enumerate(zip(
            sentiment_result.headlines[:5], 
            sentiment_result.individual_scores[:5]
        )):
            sentiment_label = "+" if score > 0 else "-" if score < 0 else "="
            section += f"  [{sentiment_label}] ({score:+.2f}) {headline}\n"
        
        if len(sentiment_result.headlines) > 5:
            section += f"  ... and {len(sentiment_result.headlines) - 5} more\n"
        
        section += "\n"
        return section
    
    @staticmethod
    def format_factor_validation(
        validation_report: FactorValidationReport,
        rank: Optional[int] = None
    ) -> str:
        """Format factor validation section"""
        
        rank_str = f" (Rank #{rank})" if rank is not None else ""
        
        section = f"""
3. FACTOR VALIDATION{rank_str}
{'-' * 80}

Selected Factor: {validation_report.factor_name}

Information Coefficient Analysis:
  - IC (Spearman):  {validation_report.ic_analysis.ic:+.4f}
  - T-Statistic:    {validation_report.ic_analysis.t_stat:+.3f}
  - P-Value:        {validation_report.ic_analysis.p_value:.4f}
  - Significance:   {'✓ YES (p < 0.05)' if validation_report.ic_analysis.is_significant else '✗ NO (p >= 0.05)'}
  - Observations:   {validation_report.ic_analysis.n_obs}

Interpretation:
  The IC of {validation_report.ic_analysis.ic:.3f} indicates that this factor has 
  {'a positive' if validation_report.ic_analysis.ic > 0 else 'a negative'} correlation with forward returns.
  {'This is statistically significant.' if validation_report.ic_analysis.is_significant else 'This is NOT statistically significant.'}

Turnover Analysis:
  - Position Changes:    {validation_report.turnover_analysis.total_changes}
  - Turnover Rate:       {validation_report.turnover_analysis.turnover_rate:.2%} per period
  - Avg Hold Periods:    {validation_report.turnover_analysis.avg_holding_periods:.1f} days
  - Transaction Impact:  {'LOW - Suitable for trading' if validation_report.turnover_analysis.turnover_rate < 0.3 else 'HIGH - May erode returns'}

"""
        
        # Add quintile analysis if available
        if validation_report.quintile_analysis:
            qa = validation_report.quintile_analysis
            section += f"""Quintile Analysis:
  - Monotonicity:  {'✓ Returns are monotonic' if qa.is_monotonic else '✗ Returns are NOT monotonic'}
  - Q1-Q5 Spread:  {qa.spread*100:.2f}%
  
  Quintile Returns:
"""
            for q, ret, std in zip(qa.quintiles, qa.avg_returns, qa.std_returns):
                section += f"    Q{q}: {ret*100:+6.2f}% ± {std*100:5.2f}%\n"
        
        section += f"\nRecommendation: {validation_report.recommendation}\n\n"
        
        return section
    
    @staticmethod
    def format_backtest_results(backtest_result: BacktestResult) -> str:
        """Format backtest results section"""
        
        metrics = backtest_result.metrics
        
        # Determine rating
        if metrics.sharpe_ratio >= 1.5:
            rating = "EXCELLENT"
        elif metrics.sharpe_ratio >= 1.0:
            rating = "STRONG"
        elif metrics.sharpe_ratio >= 0.5:
            rating = "ACCEPTABLE"
        else:
            rating = "WEAK"
        
        section = f"""
4. BACKTEST RESULTS
{'-' * 80}

Overall Rating: {rating}

Risk-Adjusted Performance:
  - Sharpe Ratio:        {metrics.sharpe_ratio:.3f}
  - Calmar Ratio:        {metrics.calmar_ratio:.3f}
  - Information Coef:    {metrics.information_coefficient:+.3f}

Returns:
  - Total Return:        {metrics.total_return*100:+.2f}%
  - CAGR:               {metrics.cagr*100:+.2f}%
  - Avg Return/Trade:    {metrics.avg_return_per_trade*10000:+.2f} bps
  - Volatility (Annual): {metrics.volatility*100:.2f}%

Risk Metrics:
  - Maximum Drawdown:    {metrics.max_drawdown*100:.2f}%
  - Win Rate:           {metrics.win_rate*100:.1f}%

Trading Statistics:
  - Total Round Trips:   {metrics.total_trades}
  - Avg Holding Period:  {metrics.avg_holding_periods:.1f} days

Transaction Costs Included:
  - Per-Trade Cost:      10 bps (commissions)
  - Slippage:           5 bps (market impact)
  - Total Cost:         15 bps per round trip

"""
        return section
    
    @staticmethod
    def format_recommendations(
        backtest_result: BacktestResult,
        validation_report: FactorValidationReport
    ) -> str:
        """Format investment recommendations"""
        
        metrics = backtest_result.metrics
        
        # Determine signal strength
        if metrics.sharpe_ratio >= 1.0 and validation_report.is_viable:
            signal_strength = "STRONG BUY"
            confidence = "HIGH"
        elif metrics.sharpe_ratio >= 0.5 and validation_report.is_viable:
            signal_strength = "BUY"
            confidence = "MODERATE"
        elif metrics.sharpe_ratio >= 0.0:
            signal_strength = "HOLD"
            confidence = "LOW"
        else:
            signal_strength = "AVOID"
            confidence = "N/A"
        
        section = f"""
5. INVESTMENT THESIS & RECOMMENDATIONS
{'-' * 80}

Signal Strength: {signal_strength}
Confidence Level: {confidence}

Key Findings:
"""
        
        # Add specific findings
        if validation_report.ic_analysis.ic > 0:
            section += f"  ✓ Factor shows positive predictive power (IC = {validation_report.ic_analysis.ic:.3f})\n"
        else:
            section += f"  ✗ Factor shows negative correlation with returns\n"
        
        if validation_report.ic_analysis.is_significant:
            section += f"  ✓ Relationship is statistically significant (p = {validation_report.ic_analysis.p_value:.3f})\n"
        else:
            section += f"  ✗ Relationship is NOT statistically significant\n"
        
        if metrics.sharpe_ratio >= 0.5:
            section += f"  ✓ Risk-adjusted returns are acceptable (Sharpe = {metrics.sharpe_ratio:.2f})\n"
        else:
            section += f"  ✗ Risk-adjusted returns are weak (Sharpe = {metrics.sharpe_ratio:.2f})\n"
        
        if abs(metrics.max_drawdown) <= 0.20:
            section += f"  ✓ Drawdown is manageable ({metrics.max_drawdown*100:.1f}%)\n"
        else:
            section += f"  ⚠ Significant drawdown risk ({metrics.max_drawdown*100:.1f}%)\n"
        
        if validation_report.turnover_analysis.turnover_rate <= 0.3:
            section += f"  ✓ Low turnover strategy - transaction costs manageable\n"
        else:
            section += f"  ⚠ High turnover - transaction costs may be significant\n"
        
        section += f"""
Risk Warnings:
  - This is a backtest using historical data
  - Past performance does not guarantee future results
  - Model assumes normal market conditions
  - Actual execution may differ from simulated results
  - Maximum loss tolerance: {abs(metrics.max_drawdown)*100:.1f}%

Next Steps:
  1. {'Proceed to paper trading' if signal_strength in ['STRONG BUY', 'BUY'] else 'Further research required'}
  2. {'Monitor position with 5% stop loss' if signal_strength in ['STRONG BUY', 'BUY'] else 'Validate on different time periods'}
  3. {'Size position at 10% of portfolio' if signal_strength == 'STRONG BUY' else 'Size position at 5% of portfolio' if signal_strength == 'BUY' else 'Do not trade'}

"""
        return section
    
    @staticmethod
    def generate_full_report(
        ticker: str,
        market_data: MarketData,
        news_data: NewsData,
        sentiment_result: SentimentResult,
        validation_report: FactorValidationReport,
        backtest_result: BacktestResult,
        factor_rank: Optional[int] = None
    ) -> str:
        """Generate complete research report"""
        
        report = ""
        report += ResearchReportGenerator.generate_header(ticker)
        report += ResearchReportGenerator.format_data_summary(market_data, news_data)
        report += ResearchReportGenerator.format_sentiment_analysis(sentiment_result)
        report += ResearchReportGenerator.format_factor_validation(validation_report, factor_rank)
        report += ResearchReportGenerator.format_backtest_results(backtest_result)
        report += ResearchReportGenerator.format_recommendations(backtest_result, validation_report)
        
        report += f"\n{'=' * 80}\n"
        report += "END OF REPORT\n"
        report += f"{'=' * 80}\n"
        
        return report
    
    @staticmethod
    def save_report(report: str, filename: str):
        """Save report to file"""
        with open(filename, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {filename}")
