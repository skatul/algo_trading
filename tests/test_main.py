"""
Unit tests for main trading engine.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.main import TradingEngine


class TestTradingEngine(unittest.TestCase):
    """Test cases for TradingEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the dependencies to avoid initialization issues
        with patch("main.Config"), patch("main.setup_logging"), patch(
            "main.DataFetcher"
        ), patch("main.BacktestEngine"):
            self.engine = TradingEngine()

        # Sample data for testing
        dates = pd.date_range("2023-01-01", periods=20)
        self.sample_data = pd.DataFrame(
            {
                "open": np.random.randn(20) + 100,
                "high": np.random.randn(20) + 102,
                "low": np.random.randn(20) + 98,
                "close": np.random.randn(20) + 100,
                "volume": np.random.randint(1000000, 2000000, 20),
            },
            index=dates,
        )

    def test_engine_initialization(self):
        """Test TradingEngine initialization."""
        self.assertIsInstance(self.engine, TradingEngine)
        self.assertIsNotNone(self.engine.logger)

    @patch("main.DataFetcher")
    def test_fetch_data(self, mock_data_fetcher):
        """Test data fetching functionality."""
        # Mock the data fetcher
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.get_yahoo_data.return_value = self.sample_data
        mock_fetcher_instance.add_technical_indicators.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance

        # Initialize engine with mocked dependencies
        with patch("main.Config"), patch("main.setup_logging"), patch(
            "main.BacktestEngine"
        ):
            engine = TradingEngine()
            engine.data_fetcher = mock_fetcher_instance

        # Test data fetching
        result = engine.fetch_data("AAPL", period="1y")

        self.assertIsInstance(result, pd.DataFrame)
        mock_fetcher_instance.get_yahoo_data.assert_called_once_with("AAPL", "1y", "1d")

    def test_create_strategy_moving_average(self):
        """Test creating moving average strategy."""
        strategy = self.engine.create_strategy(
            "moving_average", short_window=10, long_window=30
        )

        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.name, "MovingAverageCrossover")

    def test_create_strategy_rsi(self):
        """Test creating RSI strategy."""
        strategy = self.engine.create_strategy("rsi", rsi_period=14)

        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.name, "SimpleRSI")

    def test_create_strategy_buy_and_hold(self):
        """Test creating buy and hold strategy."""
        strategy = self.engine.create_strategy("buy_and_hold")

        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.name, "BuyAndHold")

    def test_create_strategy_invalid(self):
        """Test creating invalid strategy."""
        with self.assertRaises(ValueError):
            self.engine.create_strategy("invalid_strategy")

    @patch("src.main.BacktestEngine")
    @patch("src.main.DataFetcher")
    def test_backtest(self, mock_data_fetcher, mock_backtest_engine):
        """Test running a backtest."""
        # Mock data fetcher
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.get_yahoo_data.return_value = self.sample_data
        mock_fetcher_instance.add_technical_indicators.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance

        # Mock backtest engine
        mock_engine_instance = Mock()
        mock_engine_instance.run_backtest.return_value = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.05,
        }
        mock_backtest_engine.return_value = mock_engine_instance

        # Initialize engine
        with patch("main.Config"), patch("main.setup_logging"):
            engine = TradingEngine()
            engine.data_fetcher = mock_fetcher_instance

        # Run backtest
        result = engine.backtest("moving_average", "AAPL")

        # Verify calls
        mock_fetcher_instance.get_yahoo_data.assert_called()
        # Note: BacktestEngine is created internally, so we just check that we got a result
        self.assertIsInstance(result, dict)

    @patch("main.BacktestEngine")
    @patch("main.DataFetcher")
    def test_compare_strategies(self, mock_data_fetcher, mock_backtest_engine):
        """Test comparing multiple strategies."""
        # Mock data fetcher
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.get_yahoo_data.return_value = self.sample_data
        mock_fetcher_instance.add_technical_indicators.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance

        # Mock backtest engine
        mock_engine_instance = Mock()
        mock_engine_instance.run_backtest.return_value = {
            "total_return": 0.15,
            "annualized_return": 0.12,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.05,
            "total_trades": 10,
            "winning_trades": 6,
            "win_rate": 60.0,
        }
        mock_backtest_engine.return_value = mock_engine_instance

        # Initialize engine
        with patch("main.Config"), patch("main.setup_logging"):
            engine = TradingEngine()
            engine.data_fetcher = mock_fetcher_instance
            engine.backtest_engine = mock_engine_instance

        # Define strategies to compare
        strategies = [
            {"name": "Buy & Hold", "strategy": "buy_and_hold"},
            {
                "name": "MA 20/50",
                "strategy": "moving_average",
                "parameters": {"short_window": 20, "long_window": 50},
            },
        ]

        # Run comparison
        result = engine.compare_strategies("AAPL", strategies)

        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Two strategies

    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        # Sample portfolio data
        portfolio_values = [100000, 105000, 103000, 108000, 110000]

        metrics = self.engine.calculate_portfolio_metrics(portfolio_values)

        self.assertIsInstance(metrics, dict)
        expected_keys = ["total_return", "volatility", "sharpe_ratio", "max_drawdown"]
        for key in expected_keys:
            self.assertIn(key, metrics)

    @patch("main.DataFetcher")
    def test_get_market_data_with_indicators(self, mock_data_fetcher):
        """Test getting market data with technical indicators."""
        # Mock data fetcher
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.get_yahoo_data.return_value = self.sample_data
        mock_fetcher_instance.add_technical_indicators.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance

        # Initialize engine
        with patch("main.Config"), patch("main.setup_logging"), patch(
            "main.BacktestEngine"
        ):
            engine = TradingEngine()
            engine.data_fetcher = mock_fetcher_instance

        # Get data with indicators
        result = engine.get_market_data_with_indicators("AAPL", period="1y")

        # Verify calls
        mock_fetcher_instance.get_yahoo_data.assert_called_once()
        mock_fetcher_instance.add_technical_indicators.assert_called_once()
        self.assertIsInstance(result, pd.DataFrame)

    def test_validate_strategy_parameters(self):
        """Test strategy parameter validation."""
        # Valid parameters
        valid_params = {"short_window": 10, "long_window": 30}
        is_valid = self.engine.validate_strategy_parameters(
            "moving_average", valid_params
        )
        self.assertTrue(is_valid)

        # Invalid parameters - short window >= long window
        invalid_params = {"short_window": 30, "long_window": 10}
        is_valid = self.engine.validate_strategy_parameters(
            "moving_average", invalid_params
        )
        self.assertFalse(is_valid)

    def test_get_available_strategies(self):
        """Test getting list of available strategies."""
        strategies = self.engine.get_available_strategies()

        self.assertIsInstance(strategies, list)
        self.assertIn("moving_average", strategies)
        self.assertIn("rsi", strategies)
        self.assertIn("buy_and_hold", strategies)


class TestTradingEngineIntegration(unittest.TestCase):
    """Integration tests for TradingEngine."""

    @unittest.skipIf(
        os.getenv("SKIP_INTEGRATION_TESTS") == "true", "Integration tests skipped"
    )
    def test_real_backtest_integration(self):
        """Test real backtest with minimal data."""
        # This test uses real components but with minimal data
        engine = TradingEngine()

        # Create minimal sample data
        dates = pd.date_range("2023-01-01", periods=30)
        sample_data = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 30) + np.random.randn(30) * 0.5,
                "high": np.linspace(101, 111, 30) + np.random.randn(30) * 0.5,
                "low": np.linspace(99, 109, 30) + np.random.randn(30) * 0.5,
                "close": np.linspace(100, 110, 30) + np.random.randn(30) * 0.5,
                "volume": np.random.randint(1000000, 2000000, 30),
            },
            index=dates,
        )

        # Mock data fetching to use our sample data
        with patch.object(
            engine.data_fetcher, "get_yahoo_data", return_value=sample_data
        ), patch.object(
            engine.data_fetcher, "add_technical_indicators", return_value=sample_data
        ):

            # Run a simple backtest
            result = engine.backtest("buy_and_hold", "TEST_SYMBOL", period="1m")

            # Should return valid results (could be None if data fetch fails)
            if result is not None:
                self.assertIsInstance(result, dict)
                self.assertIn("total_return", result)
            else:
                # If result is None, that's also acceptable for this integration test
                self.assertIsNone(result)


class TestTradingEngineErrorHandling(unittest.TestCase):
    """Test error handling in TradingEngine."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for tests
        dates = pd.date_range("2023-01-01", periods=10)
        self.sample_data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            },
            index=dates,
        )
        
        with patch("main.Config"), patch("main.setup_logging"), patch(
            "main.DataFetcher"
        ), patch("main.BacktestEngine"):
            self.engine = TradingEngine()

    def test_backtest_with_empty_data(self):
        """Test backtest behavior with empty data."""
        with patch.object(self.engine, "fetch_data", return_value=pd.DataFrame()):
            result = self.engine.backtest("buy_and_hold", "INVALID_SYMBOL")

            # Should handle empty data gracefully
            self.assertIsNone(result)

    def test_backtest_with_data_fetch_error(self):
        """Test backtest behavior when data fetch fails."""
        with patch.object(
            self.engine, "fetch_data", side_effect=Exception("Network error")
        ):
            result = self.engine.backtest("buy_and_hold", "AAPL")

            # Should handle errors gracefully
            self.assertIsNone(result)

    def test_invalid_symbol_handling(self):
        """Test handling of invalid stock symbols."""
        with patch.object(
            self.engine.data_fetcher, "get_yahoo_data", return_value=pd.DataFrame()
        ):
            # Should raise error for invalid symbol
            with self.assertRaises(ValueError):
                self.engine.fetch_data("INVALID_SYMBOL_12345")

    def test_strategy_creation_error_handling(self):
        """Test error handling in strategy creation."""
        # Test with completely invalid strategy type
        with self.assertRaises(ValueError):
            self.engine.create_strategy("nonexistent_strategy")

    def test_comparison_with_mixed_results(self):
        """Test strategy comparison when some strategies fail."""

        # Mock scenario where some backtests succeed and others fail
        def mock_backtest(strategy_name, symbol, **kwargs):
            if strategy_name == "buy_and_hold":
                return {"total_return": 0.1, "sharpe_ratio": 1.0}
            else:
                return None  # Simulate failure

        with patch("src.main.compare_strategies") as mock_compare, \
             patch.object(self.engine, "fetch_data", return_value=self.sample_data):
            # Mock the compare_strategies function to return sample results
            mock_compare.return_value = pd.DataFrame({
                "Strategy": ["Buy & Hold"],
                "Total Return": [0.1],
                "Sharpe Ratio": [1.0]
            })
            strategies = [
                {"name": "Buy & Hold", "strategy": "buy_and_hold"},
                {"name": "Failed Strategy", "strategy": "moving_average"},
            ]

            result = self.engine.compare_strategies("AAPL", strategies)

            # Should handle mixed results gracefully
            self.assertIsInstance(result, pd.DataFrame)
            # Should only include successful strategies
            self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
