"""
Main Trading Engine

This module orchestrates all components of the algorithmic trading system.
It provides a unified interface for running backtests, live trading, and analysis.
"""

import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging
import os

# Import our modules
from src.data.data_fetcher import DataFetcher
from src.strategies.base_strategy import BaseStrategy
from src.strategies.sample_strategies import (
    MovingAverageCrossoverStrategy,
    SimpleRSIStrategy,
    BuyAndHoldStrategy,
)
from src.backtesting.backtest_engine import BacktestEngine, compare_strategies
from src.utils.config import setup_logging, get_config


class TradingEngine:
    """
    Main trading engine that coordinates all system components.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the trading engine.

        Args:
            config_file: Optional path to configuration file
        """
        # Load configuration
        self.config = get_config(config_file)

        # Set up logging
        self.logger = setup_logging(self.config)
        self.logger.info("Trading Engine initialized")

        # Initialize components
        self.data_fetcher = DataFetcher()
        self.backtest_engine = None

        # Storage for results
        self.current_data = pd.DataFrame()
        self.current_strategy = None
        self.backtest_results = {}

    def fetch_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        source: str = "yahoo",
    ) -> pd.DataFrame:
        """
        Fetch market data for analysis.

        Args:
            symbol: Stock symbol
            period: Time period for data
            interval: Data interval
            source: Data source (yahoo, alpha_vantage)

        Returns:
            DataFrame with market data
        """
        self.logger.info(f"Fetching data for {symbol} - {period} - {interval}")

        if source.lower() == "yahoo":
            data = self.data_fetcher.get_yahoo_data(symbol, period, interval)
        elif source.lower() == "alpha_vantage":
            data = self.data_fetcher.get_alpha_vantage_data(symbol)
        else:
            raise ValueError(f"Unsupported data source: {source}")

        if data.empty:
            raise ValueError(f"No data retrieved for {symbol}")

        # Add technical indicators
        data = self.data_fetcher.add_technical_indicators(data)

        # Store for reuse
        self.current_data = data

        self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
        return data

    def create_strategy(self, strategy_name: str, **parameters) -> BaseStrategy:
        """
        Create a trading strategy instance.

        Args:
            strategy_name: Name of the strategy
            **parameters: Strategy parameters

        Returns:
            Strategy instance
        """
        strategy_map = {
            "moving_average": MovingAverageCrossoverStrategy,
            "ma_crossover": MovingAverageCrossoverStrategy,
            "rsi": SimpleRSIStrategy,
            "simple_rsi": SimpleRSIStrategy,
            "buy_and_hold": BuyAndHoldStrategy,
            "benchmark": BuyAndHoldStrategy,
        }

        strategy_class = strategy_map.get(strategy_name.lower())
        if strategy_class is None:
            available = ", ".join(strategy_map.keys())
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Available: {available}"
            )

        strategy = strategy_class(**parameters)
        self.current_strategy = strategy

        self.logger.info(
            f"Created strategy: {strategy.name} with parameters: {parameters}"
        )
        return strategy

    def backtest(
        self,
        strategy: Union[str, BaseStrategy],
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: Optional[float] = None,
        **strategy_params,
    ) -> Dict:
        """
        Run a backtest on the given strategy and symbol.

        Args:
            strategy: Strategy name or instance
            symbol: Stock symbol to test
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Starting capital
            **strategy_params: Strategy parameters

        Returns:
            Backtest results dictionary
        """
        # Get initial capital from config if not provided
        if initial_capital is None:
            initial_capital = self.config.get("trading.initial_capital", 100000)
        
        # Ensure we have a valid float value
        assert initial_capital is not None, "initial_capital cannot be None"
        initial_capital = float(initial_capital)

        # Create strategy if string provided
        if isinstance(strategy, str):
            strategy_obj = self.create_strategy(strategy, **strategy_params)
        else:
            strategy_obj = strategy

        # Fetch data if not already loaded or if symbol is different
        if (
            self.current_data.empty
            or symbol not in self.current_data.get("symbol", pd.Series()).values
        ):
            self.fetch_data(symbol)

        # Initialize backtest engine
        commission = self.config.get("trading.commission", 0.001)
        self.backtest_engine = BacktestEngine(
            initial_capital=initial_capital, commission=commission
        )

        # Run backtest
        results = self.backtest_engine.run_backtest(
            strategy_obj, self.current_data, start_date, end_date
        )

        # Store results
        self.backtest_results[f"{strategy_obj.name}_{symbol}"] = results

        return results

    def compare_strategies_on_symbol(
        self, strategies: List[Dict[str, Any]], symbol: str, **kwargs
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same symbol.

        Args:
            strategies: List of strategy configurations
            symbol: Symbol to test on
            **kwargs: Additional parameters for backtesting

        Returns:
            Comparison DataFrame
        """
        self.logger.info(f"Comparing {len(strategies)} strategies on {symbol}")

        # Fetch data
        if (
            self.current_data.empty
            or symbol not in self.current_data.get("symbol", pd.Series()).values
        ):
            self.fetch_data(symbol)

        # Prepare strategies for comparison
        strategy_list = []
        for strat_config in strategies:
            name = strat_config.get("name", "Unknown")
            strategy_name = strat_config.get("strategy", "")
            params = strat_config.get("parameters", {})

            strategy_obj = self.create_strategy(strategy_name, **params)
            strategy_list.append({"name": name, "strategy": strategy_obj})

        # Run comparison
        initial_capital = kwargs.get("initial_capital") or self.config.get(
            "trading.initial_capital", 100000
        )
        comparison_results = compare_strategies(
            strategy_list, self.current_data, initial_capital
        )

        return comparison_results

    def plot_results(
        self, strategy_name: Optional[str] = None, save_path: Optional[str] = None
    ):
        """
        Plot backtest results.

        Args:
            strategy_name: Specific strategy to plot (if None, plots most recent)
            save_path: Optional path to save plot
        """
        if not self.backtest_results:
            raise ValueError("No backtest results to plot. Run a backtest first.")

        if strategy_name:
            key = strategy_name
        else:
            key = list(self.backtest_results.keys())[-1]  # Most recent

        if key not in self.backtest_results:
            available = ", ".join(self.backtest_results.keys())
            raise ValueError(f"Strategy '{key}' not found. Available: {available}")

        # Use the stored backtest engine to plot
        if self.backtest_engine:
            self.backtest_engine.results = self.backtest_results[key]
            self.backtest_engine.plot_results(save_path)

    def print_summary(self, strategy_name: Optional[str] = None):
        """
        Print backtest summary.

        Args:
            strategy_name: Specific strategy to summarize (if None, summarizes most recent)
        """
        if not self.backtest_results:
            raise ValueError("No backtest results to summarize. Run a backtest first.")

        if strategy_name:
            key = strategy_name
        else:
            key = list(self.backtest_results.keys())[-1]  # Most recent

        if key not in self.backtest_results:
            available = ", ".join(self.backtest_results.keys())
            raise ValueError(f"Strategy '{key}' not found. Available: {available}")

        # Use the stored backtest engine to print summary
        if self.backtest_engine:
            self.backtest_engine.results = self.backtest_results[key]
            self.backtest_engine.print_summary()

    def export_results(self, filepath: str, strategy_name: Optional[str] = None):
        """
        Export backtest results to CSV.

        Args:
            filepath: Path to save results
            strategy_name: Specific strategy to export (if None, exports most recent)
        """
        if not self.backtest_results:
            raise ValueError("No backtest results to export. Run a backtest first.")

        if strategy_name:
            key = strategy_name
        else:
            key = list(self.backtest_results.keys())[-1]  # Most recent

        if key not in self.backtest_results:
            available = ", ".join(self.backtest_results.keys())
            raise ValueError(f"Strategy '{key}' not found. Available: {available}")

        # Use the stored backtest engine to export
        if self.backtest_engine:
            self.backtest_engine.results = self.backtest_results[key]
            self.backtest_engine.export_results(filepath)

    def generate_quantstats_report(
        self, 
        strategy_name: Optional[str] = None,
        output_path: Optional[str] = None,
        benchmark_symbol: str = "SPY",
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a comprehensive QuantStats HTML report.
        
        Args:
            strategy_name: Specific strategy to generate report for (if None, uses most recent)
            output_path: Path to save the HTML report
            benchmark_symbol: Benchmark symbol for comparison (default: 'SPY')
            title: Custom title for the report
            
        Returns:
            Path to generated report or None if failed
        """
        if not self.backtest_results:
            raise ValueError("No backtest results to analyze. Run a backtest first.")

        if strategy_name:
            key = strategy_name
        else:
            key = list(self.backtest_results.keys())[-1]  # Most recent

        if key not in self.backtest_results:
            available = ", ".join(self.backtest_results.keys())
            raise ValueError(f"Strategy '{key}' not found. Available: {available}")

        # Use the stored backtest engine to generate report
        if self.backtest_engine:
            self.backtest_engine.results = self.backtest_results[key]
            return self.backtest_engine.generate_quantstats_report(
                output_path=output_path,
                benchmark_symbol=benchmark_symbol,
                title=title
            )
        else:
            raise ValueError("No backtest engine available")

    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategy names.

        Returns:
            List of strategy names
        """
        return [
            "moving_average",
            "ma_crossover",
            "rsi",
            "simple_rsi",
            "buy_and_hold",
            "benchmark",
        ]

    def run_analysis(
        self, symbol: str, strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run a comprehensive analysis with multiple strategies.

        Args:
            symbol: Symbol to analyze
            strategies: List of strategy names to test (if None, uses default set)

        Returns:
            Comparison results DataFrame
        """
        if strategies is None:
            strategies = ["buy_and_hold", "moving_average", "rsi"]

        self.logger.info(f"Running comprehensive analysis on {symbol}")

        # Prepare strategy configurations
        strategy_configs = []

        for strat_name in strategies:
            if strat_name == "moving_average":
                strategy_configs.extend(
                    [
                        {
                            "name": "MA_10_30",
                            "strategy": "moving_average",
                            "parameters": {"short_window": 10, "long_window": 30},
                        },
                        {
                            "name": "MA_20_50",
                            "strategy": "moving_average",
                            "parameters": {"short_window": 20, "long_window": 50},
                        },
                    ]
                )
            elif strat_name == "rsi":
                strategy_configs.extend(
                    [
                        {
                            "name": "RSI_14",
                            "strategy": "rsi",
                            "parameters": {"rsi_period": 14},
                        },
                        {
                            "name": "RSI_21",
                            "strategy": "rsi",
                            "parameters": {"rsi_period": 21},
                        },
                    ]
                )
            elif strat_name == "buy_and_hold":
                strategy_configs.append(
                    {"name": "Buy_Hold", "strategy": "buy_and_hold", "parameters": {}}
                )

        # Run comparison
        results = self.compare_strategies_on_symbol(strategy_configs, symbol)

        return results


def main():
    """Main function for running the trading engine."""
    # Initialize trading engine
    engine = TradingEngine()

    print("=== Algorithmic Trading System ===")
    print("Available strategies:", engine.get_available_strategies())

    # Example usage
    symbol = "AAPL"

    try:
        # Run analysis with multiple strategies
        print(f"\nRunning analysis on {symbol}...")
        results = engine.run_analysis(symbol)
        print("\nStrategy Comparison Results:")
        print(results.to_string(index=False))

        # Run individual backtest
        print(f"\nRunning detailed backtest on Moving Average strategy...")
        backtest_result = engine.backtest(
            "moving_average", symbol, short_window=20, long_window=50
        )

        # Print summary
        engine.print_summary()

        # Create plots directory if it doesn't exist
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Plot results
        plot_path = os.path.join(plots_dir, f"{symbol}_backtest_results.png")
        engine.plot_results(save_path=plot_path)

        print(f"\nResults plotted and saved to: {plot_path}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
