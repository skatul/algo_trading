"""
Base strategy class for algorithmic trading system.
All trading strategies should inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from enum import Enum


class SignalType(Enum):
    """Enum for different signal types."""

    BUY = 1
    SELL = -1
    HOLD = 0


class PositionType(Enum):
    """Enum for position types."""

    NONE = 0
    LONG = 1
    SHORT = -1


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    This class defines the interface that all trading strategies must implement,
    including methods for signal generation, position management, and performance tracking.
    """

    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the base strategy.

        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Strategy state
        self.current_position = PositionType.NONE
        self.entry_price = 0.0
        self.entry_date = None
        self.signals = []
        self.trades = []
        self.performance_metrics = {}

        # Data storage
        self.data = pd.DataFrame()
        self.processed_data = pd.DataFrame()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the input data.

        Args:
            data: DataFrame with OHLCV data and technical indicators

        Returns:
            DataFrame with signals added
        """
        pass

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy-specific technical indicators.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with indicators added
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has the required columns and format.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ["open", "high", "low", "close", "volume"]

        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False

        if data.empty:
            self.logger.error("Data is empty")
            return False

        if data.isnull().any().any():
            self.logger.warning("Data contains NaN values")
            # Fill NaN values with forward fill
            data.ffill(inplace=True)

        return True

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data before signal generation.

        Args:
            data: Raw OHLCV data

        Returns:
            Preprocessed data with indicators
        """
        if not self.validate_data(data):
            raise ValueError("Invalid input data")

        # Calculate indicators
        processed_data = self.calculate_indicators(data.copy())

        # Store processed data
        self.processed_data = processed_data

        return processed_data

    def run_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete strategy pipeline.

        Args:
            data: Input OHLCV data

        Returns:
            DataFrame with signals and all calculations
        """
        self.logger.info(f"Running strategy: {self.name}")

        # Preprocess data
        processed_data = self.preprocess_data(data)

        # Generate signals
        signal_data = self.generate_signals(processed_data)

        # Store results
        self.data = signal_data

        return signal_data

    def get_current_signal(self, data: pd.DataFrame, index: int) -> SignalType:
        """
        Get the signal for a specific data point.

        Args:
            data: DataFrame with signals
            index: Row index

        Returns:
            Signal type
        """
        if "signal" not in data.columns or index >= len(data):
            return SignalType.HOLD

        signal_value = data.iloc[index]["signal"]

        if signal_value > 0:
            return SignalType.BUY
        elif signal_value < 0:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    def should_enter_long(self, data: pd.DataFrame, index: int) -> bool:
        """
        Determine if strategy should enter a long position.

        Args:
            data: DataFrame with signals and indicators
            index: Current data point index

        Returns:
            True if should enter long position
        """
        signal = self.get_current_signal(data, index)
        return signal == SignalType.BUY and self.current_position != PositionType.LONG

    def should_enter_short(self, data: pd.DataFrame, index: int) -> bool:
        """
        Determine if strategy should enter a short position.

        Args:
            data: DataFrame with signals and indicators
            index: Current data point index

        Returns:
            True if should enter short position
        """
        signal = self.get_current_signal(data, index)
        return signal == SignalType.SELL and self.current_position != PositionType.SHORT

    def should_exit_position(self, data: pd.DataFrame, index: int) -> bool:
        """
        Determine if strategy should exit current position.

        Args:
            data: DataFrame with signals and indicators
            index: Current data point index

        Returns:
            True if should exit position
        """
        if self.current_position == PositionType.NONE:
            return False

        signal = self.get_current_signal(data, index)

        # Exit long position on sell signal
        if self.current_position == PositionType.LONG and signal == SignalType.SELL:
            return True

        # Exit short position on buy signal
        if self.current_position == PositionType.SHORT and signal == SignalType.BUY:
            return True

        return False

    def enter_position(self, position_type: PositionType, price: float, date: datetime):
        """
        Enter a new position.

        Args:
            position_type: Type of position (LONG/SHORT)
            price: Entry price
            date: Entry date
        """
        self.current_position = position_type
        self.entry_price = price
        self.entry_date = date

        self.logger.info(f"Entered {position_type.name} position at {price} on {date}")

    def exit_position(self, price: float, date: datetime) -> Dict:
        """
        Exit current position and record trade.

        Args:
            price: Exit price
            date: Exit date

        Returns:
            Dictionary with trade information
        """
        if self.current_position == PositionType.NONE:
            return {}

        # Calculate trade performance
        if self.current_position == PositionType.LONG:
            pnl = price - self.entry_price
            pnl_pct = (price - self.entry_price) / self.entry_price * 100
        else:  # SHORT
            pnl = self.entry_price - price
            pnl_pct = (self.entry_price - price) / self.entry_price * 100

        trade = {
            "entry_date": self.entry_date,
            "exit_date": date,
            "position_type": self.current_position.name,
            "entry_price": self.entry_price,
            "exit_price": price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "duration": (date - self.entry_date).days if self.entry_date else 0,
        }

        self.trades.append(trade)

        self.logger.info(
            f"Exited {self.current_position.name} position at {price} on {date}, P&L: {pnl:.2f} ({pnl_pct:.2f}%)"
        )

        # Reset position
        self.current_position = PositionType.NONE
        self.entry_price = 0.0
        self.entry_date = None

        return trade

    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate strategy performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        losing_trades = len(trades_df[trades_df["pnl"] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = trades_df["pnl"].sum()
        avg_pnl = trades_df["pnl"].mean()

        avg_winning_trade = (
            trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        )
        avg_losing_trade = (
            trades_df[trades_df["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0
        )

        # Risk metrics
        profit_factor = (
            abs(avg_winning_trade * winning_trades / (avg_losing_trade * losing_trades))
            if losing_trades > 0 and avg_losing_trade != 0
            else float("inf")
        )

        max_winning_streak = self._calculate_max_streak(trades_df["pnl"] > 0)
        max_losing_streak = self._calculate_max_streak(trades_df["pnl"] < 0)

        # Drawdown
        cumulative_pnl = trades_df["pnl"].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()

        self.performance_metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_winning_trade": avg_winning_trade,
            "avg_losing_trade": avg_losing_trade,
            # Legacy aliases for backward compatibility
            "avg_win": avg_winning_trade,
            "avg_loss": avg_losing_trade,
            "profit_factor": profit_factor,
            "max_winning_streak": max_winning_streak,
            "max_losing_streak": max_losing_streak,
            "max_drawdown": max_drawdown,
            "avg_trade_duration": trades_df["duration"].mean(),
        }

        return self.performance_metrics

    def _calculate_max_streak(self, condition_series: pd.Series) -> int:
        """
        Calculate maximum consecutive streak of True values.

        Args:
            condition_series: Boolean series

        Returns:
            Maximum streak length
        """
        streaks = []
        current_streak = 0

        for value in condition_series:
            if value:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0

        # Don't forget the last streak
        if current_streak > 0:
            streaks.append(current_streak)

        return max(streaks) if streaks else 0

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get a strategy parameter.

        Args:
            key: Parameter key
            default: Default value if key not found

        Returns:
            Parameter value
        """
        return self.parameters.get(key, default)

    def set_parameter(self, key: str, value: Any):
        """
        Set a strategy parameter.

        Args:
            key: Parameter key
            value: Parameter value
        """
        self.parameters[key] = value

    def reset(self):
        """Reset strategy state for new run."""
        self.current_position = PositionType.NONE
        self.entry_price = 0.0
        self.entry_date = None
        self.signals = []
        self.trades = []
        self.performance_metrics = {}
        self.data = pd.DataFrame()
        self.processed_data = pd.DataFrame()

    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name} Strategy (Parameters: {self.parameters})"

    def __repr__(self) -> str:
        """Detailed representation of the strategy."""
        return f"BaseStrategy(name='{self.name}', parameters={self.parameters})"
