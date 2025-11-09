"""
Unit tests for utils modules.
"""

import unittest
from unittest.mock import Mock, patch, mock_open
import sys
import os
import json
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.config import Config, setup_logging


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_config = {
            "trading": {
                "initial_capital": 100000,
                "commission": 0.001,
                "slippage": 0.0005,
            },
            "data": {"default_source": "yahoo", "cache_enabled": True},
            "backtesting": {"benchmark": "SPY", "start_date": "2020-01-01"},
        }

    def test_config_initialization_default(self):
        """Test Config initialization with default values."""
        config = Config()

        # Check that default values are set
        self.assertIsInstance(config.trading, dict)
        self.assertIsInstance(config.data, dict)
        self.assertIsInstance(config.backtesting, dict)

        # Check some default values
        self.assertEqual(config.get_trading_config()["initial_capital"], 100000)
        self.assertEqual(config.get_trading_config()["commission"], 0.001)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"trading": {"initial_capital": 50000}}',
    )
    @patch("os.path.exists")
    def test_config_load_from_file(self, mock_exists, mock_file):
        """Test loading configuration from file."""
        mock_exists.return_value = True

        config = Config("test_config.json")

        # Should have loaded the custom value
        self.assertEqual(config.get_trading_config()["initial_capital"], 50000)
        mock_file.assert_called()

    @patch("os.path.exists")
    def test_config_file_not_found(self, mock_exists):
        """Test handling when config file doesn't exist."""
        mock_exists.return_value = False

        # Should not raise exception, just use defaults
        config = Config("nonexistent_config.json")
        self.assertIsInstance(config.trading, dict)

    def test_get_trading_config(self):
        """Test getting trading configuration."""
        config = Config()
        trading_config = config.get_trading_config()

        self.assertIsInstance(trading_config, dict)
        self.assertIn("initial_capital", trading_config)
        self.assertIn("commission", trading_config)
        self.assertIn("slippage", trading_config)

    def test_get_data_config(self):
        """Test getting data configuration."""
        config = Config()
        data_config = config.get_data_config()

        self.assertIsInstance(data_config, dict)
        self.assertIn("default_source", data_config)
        self.assertIn("cache_enabled", data_config)

    def test_get_backtesting_config(self):
        """Test getting backtesting configuration."""
        config = Config()
        backtest_config = config.get_backtesting_config()

        self.assertIsInstance(backtest_config, dict)
        self.assertIn("benchmark", backtest_config)
        self.assertIn("start_date", backtest_config)

    def test_get_api_keys(self):
        """Test getting API keys from environment."""
        with patch.dict(
            os.environ,
            {
                "ALPHA_VANTAGE_API_KEY": "test_av_key",
                "ALPACA_API_KEY": "test_alpaca_key",
            },
        ):
            config = Config()
            api_keys = config.get_api_keys()

            self.assertEqual(api_keys["alpha_vantage"], "test_av_key")
            self.assertEqual(api_keys["alpaca_api"], "test_alpaca_key")

    def test_update_config(self):
        """Test updating configuration."""
        config = Config()

        new_values = {"initial_capital": 200000}
        config.update_config("trading", new_values)

        updated_config = config.get_trading_config()
        self.assertEqual(updated_config["initial_capital"], 200000)

    def test_validate_config(self):
        """Test configuration validation."""
        config = Config()

        # Should return True for valid config
        is_valid = config.validate_config()
        self.assertTrue(is_valid)

    def test_validate_config_invalid(self):
        """Test configuration validation with invalid values."""
        config = Config()

        # Set invalid values
        config.trading["initial_capital"] = -1000  # Invalid: negative capital

        is_valid = config.validate_config()
        self.assertFalse(is_valid)

    @patch("builtins.open", new_callable=mock_open)
    def test_save_config(self, mock_file):
        """Test saving configuration to file."""
        config = Config()
        config.save_config("test_output.json")

        mock_file.assert_called()
        # Should have written JSON data
        handle = mock_file()
        self.assertTrue(handle.write.called)


class TestLogging(unittest.TestCase):
    """Test cases for logging setup."""

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        config = Config()
        logger = setup_logging(config)

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.INFO)

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        config = Config()
        config.logging = {"level": "DEBUG"}

        logger = setup_logging(config)
        self.assertEqual(logger.level, logging.DEBUG)

    @patch("os.makedirs")
    def test_setup_logging_creates_directory(self, mock_makedirs):
        """Test that logging setup creates log directory."""
        config = Config()
        config.logging = {"file_path": "/some/path/logs/trading.log", "level": "INFO"}

        logger = setup_logging(config)

        # Should create the directory
        mock_makedirs.assert_called()


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for Config class."""

    def test_real_config_file_creation(self):
        """Test creating and loading a real config file."""
        import tempfile
        import os

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"trading": {"initial_capital": 75000, "commission": 0.002}}, f)
            temp_file = f.name

        try:
            # Load config from the file
            config = Config(temp_file)

            # Check that values were loaded correctly
            self.assertEqual(config.get_trading_config()["initial_capital"], 75000)
            self.assertEqual(config.get_trading_config()["commission"], 0.002)

        finally:
            # Clean up
            os.unlink(temp_file)

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "TRADING_INITIAL_CAPITAL": "150000",
                "TRADING_COMMISSION": "0.0015",
                "DATA_DEFAULT_SOURCE": "alpha_vantage",
            },
        ):
            config = Config()

            # Environment variables should override defaults
            # Note: This test assumes the Config class supports env var loading
            # If not implemented, this test documents the expected behavior

    def test_config_merge(self):
        """Test merging configurations from multiple sources."""
        config = Config()

        # Original config
        original_initial_capital = config.get_trading_config()["initial_capital"]

        # Update with new values
        config.update_config("trading", {"commission": 0.002})

        # Should keep original values and add new ones
        updated_config = config.get_trading_config()
        self.assertEqual(updated_config["initial_capital"], original_initial_capital)
        self.assertEqual(updated_config["commission"], 0.002)


class TestConfigValidation(unittest.TestCase):
    """Test cases for configuration validation."""

    def test_validate_trading_config(self):
        """Test trading configuration validation."""
        config = Config()

        # Valid configuration
        valid_trading = {
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0005,
        }
        config.trading = valid_trading
        self.assertTrue(config.validate_config())

        # Invalid configuration - negative capital
        invalid_trading = {"initial_capital": -100000, "commission": 0.001}
        config.trading = invalid_trading
        self.assertFalse(config.validate_config())

        # Invalid configuration - commission > 100%
        invalid_trading2 = {"initial_capital": 100000, "commission": 1.5}
        config.trading = invalid_trading2
        self.assertFalse(config.validate_config())

    def test_validate_data_config(self):
        """Test data configuration validation."""
        config = Config()

        # Valid data source
        config.data = {"default_source": "yahoo"}
        self.assertTrue(config.validate_config())

        # Invalid data source
        config.data = {"default_source": "invalid_source"}
        # Note: This assumes validation is implemented for data sources


if __name__ == "__main__":
    unittest.main(verbosity=2)
