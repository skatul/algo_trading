"""
Configuration and Logging Setup for Algorithmic Trading System

This module handles configuration management and logging setup for the trading system.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for the trading system."""

    # Default configuration
    DEFAULT_CONFIG = {
        "trading": {
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0005,
            "max_position_size": 0.1,  # 10% of portfolio
            "risk_free_rate": 0.02,
        },
        "data": {
            "default_source": "yahoo",
            "cache_enabled": True,
            "cache_duration_hours": 24,
        },
        "backtesting": {
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": None,  # Will use current date
            "plot_results": True,
            "save_results": True,
        },
        "logging": {
            "level": "INFO",
            "log_to_file": True,
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Optional path to custom configuration file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_file = config_file

        # Load custom configuration if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

        # Override with environment variables
        self._load_env_variables()

    def load_config(self, config_file: str):
        """
        Load configuration from JSON file.

        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, "r") as f:
                custom_config = json.load(f)

            # Deep merge with default config
            self._deep_merge(self.config, custom_config)

        except Exception as e:
            logging.warning(f"Could not load config file {config_file}: {e}")

    def _deep_merge(self, default_dict: Dict, custom_dict: Dict):
        """
        Deep merge custom configuration with default configuration.

        Args:
            default_dict: Default configuration dictionary
            custom_dict: Custom configuration dictionary
        """
        for key, value in custom_dict.items():
            if (
                isinstance(value, dict)
                and key in default_dict
                and isinstance(default_dict[key], dict)
            ):
                self._deep_merge(default_dict[key], value)
            else:
                default_dict[key] = value

    def _load_env_variables(self):
        """Load configuration from environment variables."""
        # Trading configuration
        initial_capital = os.getenv("INITIAL_CAPITAL")
        if initial_capital:
            self.config["trading"]["initial_capital"] = float(initial_capital)

        commission_rate = os.getenv("COMMISSION_RATE")
        if commission_rate:
            self.config["trading"]["commission"] = float(commission_rate)

        # Data configuration
        if os.getenv("DEFAULT_DATA_SOURCE"):
            self.config["data"]["default_source"] = os.getenv("DEFAULT_DATA_SOURCE")

        # Logging configuration
        if os.getenv("LOG_LEVEL"):
            self.config["logging"]["level"] = os.getenv("LOG_LEVEL")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated key path (e.g., 'trading.initial_capital')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split(".")
        current = self.config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split(".")
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def save_config(self, filepath: str):
        """
        Save current configuration to file.

        Args:
            filepath: Path to save configuration
        """
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if there is one
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def get_trading_config(self) -> Dict:
        """Get trading configuration section."""
        return self.config.get("trading", {})
    
    def get_data_config(self) -> Dict:
        """Get data configuration section."""
        return self.config.get("data", {})
    
    def get_backtesting_config(self) -> Dict:
        """Get backtesting configuration section."""
        return self.config.get("backtesting", {})
    
    def get_api_keys(self) -> Dict:
        """Get API keys from configuration and environment."""
        api_keys = {}
        
        # Get from environment variables
        alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if alpha_vantage_key:
            api_keys["alpha_vantage"] = alpha_vantage_key
            
        alpaca_key = os.getenv("ALPACA_API_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
        if alpaca_key and alpaca_secret:
            api_keys["alpaca"] = {
                "key": alpaca_key,
                "secret": alpaca_secret
            }
            
        return api_keys
    
    def update_config(self, section: str, values: Dict):
        """Update configuration section with new values."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(values)
    
    def validate_config(self) -> bool:
        """Validate configuration for required fields and valid ranges."""
        try:
            # Validate trading config
            trading = self.get_trading_config()
            if trading.get("initial_capital", 0) <= 0:
                return False
            if not (0 <= trading.get("commission", 0) < 1):
                return False
            if not (0 <= trading.get("max_position_size", 0) <= 1):
                return False
                
            # Validate data config
            data = self.get_data_config()
            valid_sources = ["yahoo", "alpha_vantage", "alpaca"]
            if data.get("default_source") not in valid_sources:
                return False
                
            return True
        except Exception:
            return False
    
    @property
    def trading(self) -> Dict:
        """Get trading configuration (property for backward compatibility)."""
        return self.get_trading_config()
    
    @property
    def data(self) -> Dict:
        """Get data configuration (property for backward compatibility)."""
        return self.get_data_config()
        
    @property
    def backtesting(self) -> Dict:
        """Get backtesting configuration (property for backward compatibility)."""
        return self.get_backtesting_config()
        
    @property
    def logging(self) -> Dict:
        """Get logging configuration (property for backward compatibility)."""
        return self.config.get("logging", {})

    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self.config.copy()


def setup_logging(config: Config) -> logging.Logger:
    """
    Set up logging for the trading system.

    Args:
        config: Configuration object

    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Get logging configuration
    log_level = config.get("logging.level", "INFO")
    log_format = config.get("logging.log_format")
    log_to_file = config.get("logging.log_to_file", True)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if enabled)
    if log_to_file:
        log_filename = f"trading_system_{datetime.now().strftime('%Y%m%d')}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Create application logger
    app_logger = logging.getLogger("algo_trading")
    app_logger.info("Logging system initialized")

    return app_logger


def get_api_keys() -> Dict[str, Optional[str]]:
    """
    Get API keys from environment variables.

    Returns:
        Dictionary with API keys (values can be None)
    """
    return {
        "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "alpaca_key": os.getenv("ALPACA_API_KEY"),
        "alpaca_secret": os.getenv("ALPACA_SECRET_KEY"),
        "alpaca_base_url": os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        ),
    }


def create_default_config_file():
    """Create a default configuration file template."""
    config = Config()

    # Add comments to the configuration
    config_with_comments = {
        "_comment_trading": "Trading configuration",
        "trading": config.config["trading"],
        "_comment_data": "Data source configuration",
        "data": config.config["data"],
        "_comment_backtesting": "Backtesting configuration",
        "backtesting": config.config["backtesting"],
        "_comment_logging": "Logging configuration",
        "logging": config.config["logging"],
    }

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json"
    )

    with open(config_path, "w") as f:
        json.dump(config_with_comments, f, indent=2, default=str)

    print(f"Default configuration file created at: {config_path}")


# Global configuration instance
_config_instance = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get global configuration instance.

    Args:
        config_file: Optional path to configuration file

    Returns:
        Configuration instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_file)

    return _config_instance


if __name__ == "__main__":
    # Create default configuration file when run directly
    create_default_config_file()

    # Test configuration and logging
    config = get_config()
    logger = setup_logging(config)

    logger.info("Configuration and logging test successful")
    print("Configuration test completed. Check logs directory for log file.")
