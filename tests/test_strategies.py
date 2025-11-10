"""
Unit tests for strategy modules.
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

from src.strategies.base_strategy import BaseStrategy, PositionType
from src.strategies.sample_strategies import (
    MovingAverageCrossoverStrategy,
    SimpleRSIStrategy,
    BuyAndHoldStrategy,
)


class TestBaseStrategy(unittest.TestCase):
    """Test cases for BaseStrategy abstract class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a concrete implementation for testing
        class TestStrategy(BaseStrategy):
            def calculate_indicators(self, data):
                df = data.copy()
                df["test_indicator"] = df["close"].rolling(5).mean()
                return df

            def generate_signals(self, data):
                df = data.copy()
                df["signal"] = 0
                df["signal_reason"] = ""
                # Simple buy signal when price > moving average
                buy_condition = df["close"] > df["test_indicator"]
                df.loc[buy_condition, "signal"] = 1
                df.loc[buy_condition, "signal_reason"] = "BUY_SIGNAL"
                return df

        self.strategy = TestStrategy(name="TestStrategy", parameters={"test_param": 10})

        # Sample data
        self.sample_data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [102, 103, 104, 105, 106],
                "low": [99, 100, 101, 102, 103],
                "close": [101, 102, 103, 104, 105],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "TestStrategy")
        self.assertEqual(self.strategy.parameters["test_param"], 10)
        self.assertEqual(self.strategy.current_position, PositionType.NONE)
        self.assertEqual(len(self.strategy.trades), 0)

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        is_valid = self.strategy.validate_data(self.sample_data)
        self.assertTrue(is_valid)

    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        invalid_data = pd.DataFrame({"price": [100, 101, 102]})
        is_valid = self.strategy.validate_data(invalid_data)
        self.assertFalse(is_valid)

    def test_validate_data_empty(self):
        """Test data validation with empty data."""
        empty_data = pd.DataFrame()
        is_valid = self.strategy.validate_data(empty_data)
        self.assertFalse(is_valid)

    def test_run_strategy(self):
        """Test running the complete strategy."""
        result = self.strategy.run_strategy(self.sample_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("test_indicator", result.columns)
        self.assertIn("signal", result.columns)
        self.assertIn("signal_reason", result.columns)

    def test_enter_position_long(self):
        """Test entering a long position."""
        date = datetime(2023, 1, 1)
        price = 100.0

        self.strategy.enter_position(PositionType.LONG, price, date)

        self.assertEqual(self.strategy.current_position, PositionType.LONG)
        self.assertEqual(self.strategy.entry_price, price)
        self.assertEqual(self.strategy.entry_date, date)

    def test_exit_position(self):
        """Test exiting a position."""
        # First enter a position
        entry_date = datetime(2023, 1, 1)
        entry_price = 100.0
        self.strategy.enter_position(PositionType.LONG, entry_price, entry_date)

        # Then exit it
        exit_date = datetime(2023, 1, 2)
        exit_price = 110.0
        self.strategy.exit_position(exit_price, exit_date)

        self.assertEqual(self.strategy.current_position, PositionType.NONE)
        self.assertEqual(len(self.strategy.trades), 1)

        trade = self.strategy.trades[0]
        self.assertEqual(trade["pnl"], 10.0)
        self.assertEqual(trade["pnl_pct"], 10.0)

    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Add some sample trades
        self.strategy.trades = [
            {
                "pnl": 10,
                "pnl_pct": 10,
                "entry_date": datetime(2023, 1, 1),
                "exit_date": datetime(2023, 1, 2),
                "duration": 1,
            },
            {
                "pnl": -5,
                "pnl_pct": -5,
                "entry_date": datetime(2023, 1, 3),
                "exit_date": datetime(2023, 1, 4),
                "duration": 1,
            },
            {
                "pnl": 15,
                "pnl_pct": 15,
                "entry_date": datetime(2023, 1, 5),
                "exit_date": datetime(2023, 1, 6),
                "duration": 1,
            },
        ]

        metrics = self.strategy.calculate_performance_metrics()

        self.assertIn("total_trades", metrics)
        self.assertIn("winning_trades", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("total_pnl", metrics)
        self.assertIn("avg_win", metrics)
        self.assertIn("avg_loss", metrics)

        self.assertEqual(metrics["total_trades"], 3)
        self.assertEqual(metrics["winning_trades"], 2)
        self.assertAlmostEqual(metrics["win_rate"], 0.6667, places=4)
        self.assertEqual(metrics["total_pnl"], 20)


class TestMovingAverageCrossoverStrategy(unittest.TestCase):
    """Test cases for Moving Average Crossover Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=10)

        # Create sample data with trend
        dates = pd.date_range("2023-01-01", periods=20)
        # Create upward trend that should trigger MA crossover
        prices = np.linspace(100, 120, 20) + np.random.randn(20) * 0.5

        self.sample_data = pd.DataFrame(
            {
                "open": prices - 1,
                "high": prices + 1,
                "low": prices - 2,
                "close": prices,
                "volume": np.random.randint(1000, 2000, 20),
            },
            index=dates,
        )

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.short_window, 5)
        self.assertEqual(self.strategy.long_window, 10)
        self.assertEqual(self.strategy.name, "MovingAverageCrossover")

    def test_calculate_indicators(self):
        """Test moving average calculation."""
        result = self.strategy.calculate_indicators(self.sample_data)

        self.assertIn("sma_5", result.columns)
        self.assertIn("sma_10", result.columns)

        # Check that moving averages are calculated correctly
        expected_short_ma = self.sample_data["close"].rolling(5).mean()
        pd.testing.assert_series_equal(
            result["sma_short"], expected_short_ma, check_names=False
        )

    def test_generate_signals(self):
        """Test signal generation."""
        data_with_indicators = self.strategy.calculate_indicators(self.sample_data)
        result = self.strategy.generate_signals(data_with_indicators)

        self.assertIn("signal", result.columns)
        self.assertIn("signal_reason", result.columns)

        # Check that signals are only 0 or 1
        signals = result["signal"].dropna()
        self.assertTrue(all(signal in [0, 1] for signal in signals))

    def test_run_complete_strategy(self):
        """Test running the complete strategy."""
        result = self.strategy.run_strategy(self.sample_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("sma_5", result.columns)
        self.assertIn("sma_10", result.columns)
        self.assertIn("signal", result.columns)


class TestSimpleRSIStrategy(unittest.TestCase):
    """Test cases for Simple RSI Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = SimpleRSIStrategy(rsi_period=14, oversold=30, overbought=70)

        # Create sample data with volatility for RSI calculation
        dates = pd.date_range("2023-01-01", periods=30)
        # Create data that will generate various RSI levels
        base_price = 100
        changes = np.random.randn(30) * 2  # Random price changes
        prices = [base_price]
        for change in changes[1:]:
            prices.append(prices[-1] + change)

        self.sample_data = pd.DataFrame(
            {
                "open": [p - 0.5 for p in prices],
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": np.random.randint(1000, 2000, 30),
            },
            index=dates,
        )

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.rsi_period, 14)
        self.assertEqual(self.strategy.oversold, 30)
        self.assertEqual(self.strategy.overbought, 70)
        self.assertEqual(self.strategy.name, "SimpleRSI")

    def test_calculate_indicators(self):
        """Test RSI calculation."""
        result = self.strategy.calculate_indicators(self.sample_data)

        self.assertIn("rsi", result.columns)

        # Check RSI bounds
        rsi_values = result["rsi"].dropna()
        if len(rsi_values) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in rsi_values))

    def test_generate_signals(self):
        """Test signal generation."""
        data_with_indicators = self.strategy.calculate_indicators(self.sample_data)
        result = self.strategy.generate_signals(data_with_indicators)

        self.assertIn("signal", result.columns)
        self.assertIn("signal_reason", result.columns)

        # Check that signals are only 0 or 1
        signals = result["signal"].dropna()
        self.assertTrue(all(signal in [0, 1] for signal in signals))


class TestBuyAndHoldStrategy(unittest.TestCase):
    """Test cases for Buy and Hold Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = BuyAndHoldStrategy()

        self.sample_data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [102, 103, 104, 105, 106],
                "low": [99, 100, 101, 102, 103],
                "close": [101, 102, 103, 104, 105],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "BuyAndHold")

    def test_calculate_indicators(self):
        """Test that no additional indicators are calculated."""
        result = self.strategy.calculate_indicators(self.sample_data)

        # Should return the same data
        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_generate_signals(self):
        """Test signal generation - should buy on first day only."""
        result = self.strategy.generate_signals(self.sample_data)

        self.assertIn("signal", result.columns)
        self.assertIn("signal_reason", result.columns)

        # Should have exactly one buy signal on first day
        signals = result["signal"]
        self.assertEqual(signals.iloc[0], 1)  # First day: buy
        self.assertTrue(all(signals.iloc[1:] == 0))  # Rest: hold


class TestPositionType(unittest.TestCase):
    """Test cases for PositionType enum."""

    def test_position_types(self):
        """Test that position types are properly defined."""
        self.assertEqual(PositionType.NONE.value, 0)
        self.assertEqual(PositionType.LONG.value, 1)
        self.assertEqual(PositionType.SHORT.value, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
