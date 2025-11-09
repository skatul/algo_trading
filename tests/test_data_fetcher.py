"""
Unit tests for data_fetcher module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.data_fetcher import DataFetcher


class TestDataFetcher(unittest.TestCase):
    """Test cases for DataFetcher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = DataFetcher()

        # Sample data for testing
        self.sample_data = pd.DataFrame(
            {
                "open": [150.0, 151.0, 149.0, 152.0, 148.0],
                "high": [152.0, 153.0, 151.0, 154.0, 150.0],
                "low": [149.0, 150.0, 147.0, 151.0, 146.0],
                "close": [151.0, 150.0, 150.5, 153.0, 149.0],
                "volume": [1000000, 1100000, 900000, 1200000, 800000],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

    def test_init(self):
        """Test DataFetcher initialization."""
        fetcher = DataFetcher()
        self.assertIsInstance(fetcher, DataFetcher)
        self.assertIsNotNone(fetcher.logger)
        self.assertIsInstance(fetcher.data_cache, dict)

    @patch("data.data_fetcher.yf.Ticker")
    def test_get_yahoo_data_success(self, mock_ticker):
        """Test successful Yahoo Finance data retrieval."""
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.sample_data
        mock_ticker.return_value = mock_ticker_instance

        # Test data fetching
        result = self.fetcher.get_yahoo_data("AAPL", "1y")

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn("symbol", result.columns)
        self.assertEqual(result["symbol"].iloc[0], "AAPL")
        mock_ticker.assert_called_once_with("AAPL")

    @patch("data.data_fetcher.yf.Ticker")
    def test_get_yahoo_data_empty(self, mock_ticker):
        """Test handling of empty Yahoo Finance response."""
        # Mock empty response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        result = self.fetcher.get_yahoo_data("INVALID")

        self.assertTrue(result.empty)

    @patch("data.data_fetcher.yf.Ticker")
    def test_get_yahoo_data_exception(self, mock_ticker):
        """Test exception handling in Yahoo Finance data retrieval."""
        # Mock exception
        mock_ticker.side_effect = Exception("Network error")

        result = self.fetcher.get_yahoo_data("AAPL")

        self.assertTrue(result.empty)

    def test_get_multiple_symbols(self):
        """Test fetching data for multiple symbols."""
        with patch.object(self.fetcher, "get_yahoo_data") as mock_get_data:
            mock_get_data.return_value = self.sample_data

            symbols = ["AAPL", "GOOGL", "MSFT"]
            result = self.fetcher.get_multiple_symbols(symbols)

            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 3)
            for symbol in symbols:
                self.assertIn(symbol, result)

    @patch("data.data_fetcher.requests.get")
    def test_get_alpha_vantage_data_success(self, mock_get):
        """Test successful Alpha Vantage data retrieval."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2023-01-01": {
                    "1. open": "150",
                    "2. high": "152",
                    "3. low": "149",
                    "4. close": "151",
                    "5. volume": "1000000",
                },
                "2023-01-02": {
                    "1. open": "151",
                    "2. high": "153",
                    "3. low": "150",
                    "4. close": "150",
                    "5. volume": "1100000",
                },
            }
        }
        mock_get.return_value = mock_response

        # Set API key for test
        self.fetcher.alpha_vantage_key = "test_key"

        result = self.fetcher.get_alpha_vantage_data("AAPL")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn("symbol", result.columns)
        self.assertEqual(result["symbol"].iloc[0], "AAPL")

    def test_get_alpha_vantage_data_no_key(self):
        """Test Alpha Vantage data retrieval without API key."""
        self.fetcher.alpha_vantage_key = None

        result = self.fetcher.get_alpha_vantage_data("AAPL")

        self.assertTrue(result.empty)

    @patch("data.data_fetcher.yf.Ticker")
    def test_get_stock_info(self, mock_ticker):
        """Test stock information retrieval."""
        # Mock stock info
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "marketCap": 2000000000000,
            "trailingPE": 25.5,
        }
        mock_ticker.return_value = mock_ticker_instance

        result = self.fetcher.get_stock_info("AAPL")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["company_name"], "Apple Inc.")
        self.assertEqual(result["sector"], "Technology")

    def test_calculate_returns(self):
        """Test return calculations."""
        result = self.fetcher.calculate_returns(self.sample_data)

        self.assertIn("returns", result.columns)
        self.assertIn("log_returns", result.columns)
        self.assertIn("cumulative_returns", result.columns)

        # Test that returns are calculated correctly
        expected_return = (150.0 - 151.0) / 151.0
        actual_return = result["returns"].iloc[1]
        self.assertAlmostEqual(actual_return, expected_return, places=6)

    def test_add_technical_indicators(self):
        """Test technical indicator calculations."""
        # Create larger dataset for moving averages
        data = pd.DataFrame(
            {
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 102,
                "low": np.random.randn(100) + 98,
                "close": np.random.randn(100) + 100,
                "volume": np.random.randint(1000000, 2000000, 100),
            },
            index=pd.date_range("2023-01-01", periods=100),
        )

        result = self.fetcher.add_technical_indicators(data)

        # Check that indicators are added
        expected_indicators = [
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "macd_histogram",
            "rsi",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "volume_sma",
            "volume_ratio",
        ]

        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)

        # Test RSI bounds
        rsi_values = result["rsi"].dropna()
        self.assertTrue(all(0 <= val <= 100 for val in rsi_values))

    @patch("data.data_fetcher.pd.DataFrame.to_csv")
    @patch("data.data_fetcher.os.makedirs")
    def test_save_data_csv(self, mock_makedirs, mock_to_csv):
        """Test saving data to CSV format."""
        self.fetcher.save_data(self.sample_data, "test_data", "csv")

        mock_makedirs.assert_called_once()
        mock_to_csv.assert_called_once()

    @patch("data.data_fetcher.pd.read_csv")
    def test_load_data_csv(self, mock_read_csv):
        """Test loading data from CSV format."""
        mock_read_csv.return_value = self.sample_data

        result = self.fetcher.load_data("test_data", "csv")

        mock_read_csv.assert_called_once()
        self.assertIsInstance(result, pd.DataFrame)

    def test_save_data_invalid_format(self):
        """Test saving data with invalid format."""
        # This should not raise an exception, just log an error
        self.fetcher.save_data(self.sample_data, "test_data", "invalid_format")

    def test_load_data_file_not_found(self):
        """Test loading data from non-existent file."""
        result = self.fetcher.load_data("non_existent_file", "csv")

        self.assertTrue(result.empty)


class TestDataFetcherIntegration(unittest.TestCase):
    """Integration tests for DataFetcher (require internet connection)."""

    @unittest.skipIf(
        os.getenv("SKIP_INTEGRATION_TESTS") == "true", "Integration tests skipped"
    )
    def test_real_yahoo_data_fetch(self):
        """Test fetching real data from Yahoo Finance."""
        fetcher = DataFetcher()

        # Fetch a small amount of recent data
        data = fetcher.get_yahoo_data("AAPL", period="5d")

        if not data.empty:  # Only test if data was successfully fetched
            self.assertIn("close", data.columns)
            self.assertIn("volume", data.columns)
            self.assertIn("symbol", data.columns)
            self.assertEqual(data["symbol"].iloc[0], "AAPL")


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2)
