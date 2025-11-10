"""
Unit tests for backtesting engine.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtesting.backtest_engine import BacktestEngine
from strategies.base_strategy import BaseStrategy, PositionType


class TestBacktestEngine(unittest.TestCase):
    """Test cases for BacktestEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine(initial_capital=100000, commission=0.001)

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=10)
        self.sample_data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
                "signal": [0, 1, 0, 0, -1, 0, 1, 0, 0, -1],
                "signal_reason": ["", "BUY", "", "", "SELL", "", "BUY", "", "", "SELL"],
            },
            index=dates,
        )

        # Create a mock strategy
        self.mock_strategy = Mock(spec=BaseStrategy)
        self.mock_strategy.name = "TestStrategy"
        self.mock_strategy.run_strategy.return_value = self.sample_data

    def test_engine_initialization(self):
        """Test BacktestEngine initialization."""
        self.assertEqual(self.engine.initial_capital, 100000)
        self.assertEqual(self.engine.commission, 0.001)
        self.assertEqual(self.engine.cash, 100000)
        self.assertEqual(self.engine.shares, 0)
        self.assertEqual(len(self.engine.transactions), 0)

    def test_calculate_transaction_cost(self):
        """Test transaction cost calculation."""
        cost = self.engine.calculate_transaction_cost(10000)  # $10,000 transaction
        expected_cost = 10000 * 0.001  # 0.1% commission
        self.assertEqual(cost, expected_cost)

    def test_execute_buy_order(self):
        """Test executing a buy order."""
        initial_cash = self.engine.cash
        price = 100

        self.engine.execute_buy_order(price, datetime(2023, 1, 1))

        expected_shares = (initial_cash * (1 - self.engine.commission)) // price
        expected_cash = initial_cash - (
            expected_shares * price * (1 + self.engine.commission)
        )

        self.assertEqual(self.engine.shares, expected_shares)
        self.assertAlmostEqual(self.engine.cash, expected_cash, places=2)
        self.assertEqual(len(self.engine.transactions), 1)

    def test_execute_sell_order(self):
        """Test executing a sell order."""
        # First buy some shares
        self.engine.execute_buy_order(100, datetime(2023, 1, 1))
        initial_shares = self.engine.shares
        initial_cash = self.engine.cash

        # Then sell them
        sell_price = 110
        self.engine.execute_sell_order(sell_price, datetime(2023, 1, 2))

        expected_cash_increase = (
            initial_shares * sell_price * (1 - self.engine.commission)
        )
        expected_final_cash = initial_cash + expected_cash_increase

        self.assertEqual(self.engine.shares, 0)
        self.assertAlmostEqual(self.engine.cash, expected_final_cash, places=2)
        self.assertEqual(len(self.engine.transactions), 2)

    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation."""
        # Buy some shares
        self.engine.execute_buy_order(100, datetime(2023, 1, 1))

        # Calculate portfolio value at different price
        current_price = 110
        portfolio_value = self.engine.calculate_portfolio_value(current_price)

        expected_value = self.engine.cash + (self.engine.shares * current_price)
        self.assertEqual(portfolio_value, expected_value)

    def test_run_backtest(self):
        """Test running a complete backtest."""
        result = self.engine.run_backtest(self.mock_strategy, self.sample_data)

        # Check that mock strategy was called
        self.mock_strategy.run.assert_called_once_with(self.sample_data)

        # Check result structure
        self.assertIsInstance(result, dict)
        expected_keys = [
            "initial_capital",
            "final_capital",
            "total_return",
            "total_trades",
            "winning_trades",
            "losing_trades",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_calculate_metrics(self):
        """Test performance metrics calculation."""
        # Create sample portfolio values and returns
        portfolio_values = [100000, 101000, 99000, 102000, 105000]
        returns = pd.Series(
            [0.01, -0.0198, 0.0303, 0.0294]
        )  # Returns from portfolio values

        metrics = self.engine.calculate_metrics(portfolio_values, returns)

        expected_keys = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)

        # Check total return calculation
        expected_total_return = (105000 - 100000) / 100000
        self.assertAlmostEqual(metrics["total_return"], expected_total_return, places=4)

        # Check max drawdown (should be negative)
        self.assertLessEqual(metrics["max_drawdown"], 0)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        risk_free_rate = 0.02

        sharpe = self.engine.calculate_sharpe_ratio(returns, risk_free_rate)

        # Should be a float
        self.assertIsInstance(sharpe, (float, int))

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        portfolio_values = [100000, 110000, 95000, 105000, 90000, 120000]

        max_dd = self.engine.calculate_max_drawdown(portfolio_values)

        # Max drawdown should be negative and between -1 and 0
        self.assertLessEqual(max_dd, 0)
        self.assertGreaterEqual(max_dd, -1)

    def test_export_results(self):
        """Test results export functionality."""
        # Run a backtest first
        result = self.engine.run_backtest(self.mock_strategy, self.sample_data)

        # Test export (mock file operations)
        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            self.engine.export_results("test_results.csv")
            mock_to_csv.assert_called()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_results(self, mock_show, mock_savefig):
        """Test results plotting functionality."""
        # Run a backtest first
        result = self.engine.run_backtest(self.mock_strategy, self.sample_data)

        # Test plotting
        self.engine.plot_results()

        # Should not raise an error
        self.assertTrue(True)

    def test_print_summary(self):
        """Test summary printing."""
        # Run a backtest first
        result = self.engine.run_backtest(self.mock_strategy, self.sample_data)

        # Test that print_summary doesn't raise an error
        try:
            self.engine.print_summary()
            success = True
        except Exception:
            success = False

        self.assertTrue(success)

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()

        with self.assertRaises((ValueError, KeyError)):
            self.engine.run_backtest(self.mock_strategy, empty_data)

    def test_invalid_signals_handling(self):
        """Test handling of invalid signals."""
        # Data with invalid signals
        invalid_data = self.sample_data.copy()
        invalid_data["signal"] = [
            2,
            -2,
            5,
            0,
            1,
            -1,
            3,
            0,
            0,
            -1,
        ]  # Invalid signal values

        self.mock_strategy.run.return_value = invalid_data

        # Should still run without crashing (engine should handle invalid signals gracefully)
        result = self.engine.run_backtest(self.mock_strategy, invalid_data)
        self.assertIsInstance(result, dict)

    def test_insufficient_cash_handling(self):
        """Test handling when there's insufficient cash for a trade."""
        # Create scenario with very expensive stock
        expensive_data = self.sample_data.copy()
        expensive_data["close"] = (
            expensive_data["close"] * 10000
        )  # Very expensive stock

        self.mock_strategy.run.return_value = expensive_data

        # Should handle insufficient cash gracefully
        result = self.engine.run_backtest(self.mock_strategy, expensive_data)
        self.assertIsInstance(result, dict)


class TestBacktestEngineIntegration(unittest.TestCase):
    """Integration tests for BacktestEngine with real strategies."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine(initial_capital=100000)

        # Create realistic sample data
        dates = pd.date_range("2023-01-01", periods=50)
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 50)  # Daily returns
        prices = [100]  # Starting price
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        self.sample_data = pd.DataFrame(
            {
                "open": [p * 0.999 for p in prices],
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": np.random.randint(100000, 200000, 50),
            },
            index=dates,
        )

    def test_buy_and_hold_strategy_integration(self):
        """Test integration with buy and hold strategy."""
        from strategies.sample_strategies import BuyAndHoldStrategy

        strategy = BuyAndHoldStrategy()
        result = self.engine.run_backtest(strategy, self.sample_data)

        # Should have results
        self.assertIsInstance(result, dict)
        self.assertIn("total_return", result)
        self.assertIn("final_capital", result)

        # Should have at least one trade (buy and potentially sell)
        self.assertGreaterEqual(result["total_trades"], 1)

    def test_moving_average_strategy_integration(self):
        """Test integration with moving average strategy."""
        from strategies.sample_strategies import MovingAverageCrossoverStrategy

        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=10)
        result = self.engine.run_backtest(strategy, self.sample_data)

        # Should have results
        self.assertIsInstance(result, dict)
        self.assertIn("total_return", result)
        self.assertIn("final_capital", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
