#!/usr/bin/env python3
"""
Demo script for the Algorithmic Trading System

This script demonstrates the main features of the trading system including
data fetching, strategy backtesting, and performance analysis.
"""

import sys
import os

# Add the project root to Python path
sys.path.append("/home/atul/projects/algo_trading")

from src.main import TradingEngine


def demo_data_fetching():
    """Demonstrate data fetching capabilities."""
    print("=== Data Fetching Demo ===")

    engine = TradingEngine()

    # Fetch data for Apple
    print("Fetching AAPL data...")
    data = engine.fetch_data("AAPL", period="6mo")

    print(f"Retrieved {len(data)} records")
    print("Sample data:")
    print(data[["close", "sma_20", "sma_50", "rsi"]].head(10))
    print()


def demo_single_strategy():
    """Demonstrate single strategy backtesting."""
    print("=== Single Strategy Backtest Demo ===")

    engine = TradingEngine()

    # Test Moving Average strategy
    print("Testing Moving Average Crossover Strategy on AAPL...")
    result = engine.backtest(
        strategy="moving_average",
        symbol="AAPL",
        short_window=20,
        long_window=50,
        initial_capital=100000,
    )

    # Print summary
    engine.print_summary()
    print()


def demo_strategy_comparison():
    """Demonstrate strategy comparison."""
    print("=== Strategy Comparison Demo ===")

    engine = TradingEngine()

    # Compare multiple strategies
    strategies = [
        {"name": "Buy & Hold", "strategy": "buy_and_hold"},
        {
            "name": "MA 20/50",
            "strategy": "moving_average",
            "parameters": {"short_window": 20, "long_window": 50},
        },
        {"name": "RSI 14", "strategy": "rsi", "parameters": {"rsi_period": 14}},
    ]

    print("Comparing strategies on AAPL...")
    comparison = engine.compare_strategies_on_symbol(strategies, "AAPL")

    print("Comparison Results:")
    print(comparison.round(2))
    print()


def demo_multiple_symbols():
    """Demonstrate testing on multiple symbols."""
    print("=== Multiple Symbols Demo ===")

    engine = TradingEngine()
    symbols = ["AAPL", "GOOGL", "MSFT"]

    results = []

    for symbol in symbols:
        print(f"Testing Moving Average strategy on {symbol}...")
        result = engine.backtest(
            "moving_average", symbol, short_window=10, long_window=30
        )

        results.append(
            {
                "Symbol": symbol,
                "Total Return (%)": result["total_return"] * 100,
                "Sharpe Ratio": result["sharpe_ratio"],
                "Max Drawdown (%)": result["max_drawdown"] * 100,
                "Win Rate (%)": result["win_rate"] * 100,
            }
        )

    import pandas as pd

    results_df = pd.DataFrame(results)
    print("\nResults across symbols:")
    print(results_df.round(2))
    print()


def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis."""
    print("=== Comprehensive Analysis Demo ===")

    engine = TradingEngine()

    print("Running comprehensive analysis on AAPL...")
    results = engine.run_analysis("AAPL")

    print("Analysis Results:")
    print(results.round(2))
    print()


def demo_quantstats_integration():
    """Demonstrate QuantStats integration with enhanced analytics."""
    print("=== QuantStats Integration Demo ===")
    
    engine = TradingEngine()
    
    print("Running backtest with enhanced QuantStats metrics...")
    
    # Run a backtest
    result = engine.backtest(
        'moving_average', 
        'AAPL', 
        short_window=10, 
        long_window=30,
        start_date='2023-01-01'
    )
    
    print("\n📊 Enhanced Performance Summary (powered by QuantStats):")
    engine.print_summary()
    
    print("\n📋 Generating comprehensive QuantStats HTML report...")
    
    # Create reports directory
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate QuantStats report
    report_path = engine.generate_quantstats_report(
        output_path=os.path.join(reports_dir, "aapl_moving_average_report.html"),
        benchmark_symbol="SPY",
        title="AAPL Moving Average Strategy - Professional Analysis"
    )
    
    if report_path:
        print(f"✅ Comprehensive report generated: {report_path}")
        print("   This report includes:")
        print("   • Interactive performance charts")
        print("   • Risk-adjusted metrics (Sharpe, Sortino, Calmar)")
        print("   • Drawdown analysis")
        print("   • Monthly/yearly performance breakdown")
        print("   • Benchmark comparison with SPY")
        print("   • Distribution analysis")
        print("   • Rolling performance metrics")
    else:
        print("❌ Report generation failed")
    
    print(f"\n💡 Key enhancements with QuantStats:")
    print(f"   • Sortino Ratio: {result['sortino_ratio']:.3f}")
    print(f"   • Calmar Ratio: {result['calmar_ratio']:.3f}")
    print(f"   • VaR (95%): {result['var_95']*100:.2f}%")
    print(f"   • CVaR (95%): {result['cvar_95']*100:.2f}%")
    print(f"   • Skewness: {result['skewness']:.3f}")
    print(f"   • Kurtosis: {result['kurtosis']:.3f}")
    print()


def main():
    """Run all demo functions."""
    print("🚀 Algorithmic Trading System - Demo")
    print("=" * 50)

    try:
        # Demo 1: Basic usage
        demo_data_fetching()

        # Demo 2: Simple backtest
        demo_single_strategy()

        # Demo 3: Strategy comparison
        demo_strategy_comparison()

        # Demo 4: Multiple Symbols
        demo_multiple_symbols()

        # Demo 5: Comprehensive Analysis
        demo_comprehensive_analysis()

        # Demo 6: QuantStats Integration (NEW!)
        demo_quantstats_integration()
        print("✅ All demos completed successfully!")
        print("\nNext steps:")
        print("1. Modify strategy parameters in the demos above")
        print("2. Try different symbols and time periods")
        print("3. Create your own custom strategies")
        print("4. Add risk management rules")
        print("5. Implement live trading capabilities")
        print("6. Open the generated HTML reports for detailed analysis")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
