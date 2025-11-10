#!/usr/bin/env python3
"""
Demo script for the Algorithmic Trading System

This script demonstrates the main features of the trading system including
data fetching, strategy backtesting, and performance analysis.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

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
        strategy_name="moving_average",
        symbol="AAPL",
        short_window=20,
        long_window=50,
        initial_capital=100000,
    )

    if result:
        # Print summary using the backtest engine if it exists
        if hasattr(engine, 'backtest_engine') and engine.backtest_engine:
            engine.backtest_engine.print_summary()
        else:
            print(f"Backtest completed successfully!")
            print(f"Total Return: {result.get('total_return', 0)*100:.2f}%")
            print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
        print()
    else:
        print("Backtest failed. Please check your data connection or symbol.")
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

        if result:
            results.append(
                {
                    "Symbol": symbol,
                    "Total Return (%)": result.get("total_return", 0) * 100,
                    "Sharpe Ratio": result.get("sharpe_ratio", 0),
                    "Max Drawdown (%)": result.get("max_drawdown", 0) * 100,
                    "Win Rate (%)": result.get("win_rate", 0) * 100,
                }
            )
        else:
            print(f"  Failed to backtest {symbol}")
            results.append(
                {
                    "Symbol": symbol,
                    "Total Return (%)": 0,
                    "Sharpe Ratio": 0,
                    "Max Drawdown (%)": 0,
                    "Win Rate (%)": 0,
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


def main():
    """Run all demos."""
    print("🚀 Algorithmic Trading System Demo")
    print("=" * 50)

    try:
        # Demo 1: Data Fetching
        demo_data_fetching()

        # Demo 2: Single Strategy
        demo_single_strategy()

        # Demo 3: Strategy Comparison
        demo_strategy_comparison()

        # Demo 4: Multiple Symbols
        demo_multiple_symbols()

        # Demo 5: Comprehensive Analysis
        demo_comprehensive_analysis()

        print("✅ All demos completed successfully!")
        print("\nNext steps:")
        print("1. Modify strategy parameters in the demos above")
        print("2. Try different symbols and time periods")
        print("3. Create your own custom strategies")
        print("4. Add risk management rules")
        print("5. Implement live trading capabilities")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
