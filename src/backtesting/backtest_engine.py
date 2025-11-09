"""
Backtesting Engine for Algorithmic Trading System

This module provides comprehensive backtesting capabilities for trading strategies,
including performance metrics, visualization, and trade analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import os


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategy performance.
    """

    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Initialize the backtesting engine.

        Args:
            initial_capital: Starting capital for backtesting
            commission: Commission rate per trade (as decimal, e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        self.results = {}

    def run_backtest(
        self,
        strategy,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict:
        """
        Run a complete backtest on the given strategy and data.

        Args:
            strategy: Trading strategy instance (must inherit from BaseStrategy)
            data: Historical price data
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)

        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest for strategy: {strategy.name}")

        # Filter data by date range if provided
        test_data = self._prepare_data(data, start_date, end_date)

        if test_data.empty:
            raise ValueError("No data available for the specified date range")

        # Run strategy to generate signals
        signal_data = strategy.run_strategy(test_data)

        # Execute trades based on signals
        portfolio_data = self._simulate_trading(signal_data)

        # Calculate performance metrics
        performance = self._calculate_performance_metrics(portfolio_data)

        # Store results
        self.results = {
            "strategy_name": strategy.name,
            "parameters": strategy.parameters,
            "start_date": test_data.index[0],
            "end_date": test_data.index[-1],
            "initial_capital": self.initial_capital,
            "final_capital": portfolio_data["portfolio_value"].iloc[-1],
            "total_return": performance["total_return"],
            "annualized_return": performance["annualized_return"],
            "volatility": performance["volatility"],
            "sharpe_ratio": performance["sharpe_ratio"],
            "max_drawdown": performance["max_drawdown"],
            "total_trades": performance["total_trades"],
            "winning_trades": performance["winning_trades"],
            "win_rate": performance["win_rate"],
            "profit_factor": performance["profit_factor"],
            "data": portfolio_data,
            "trades": self.trades.copy(),
        }

        self.logger.info(
            f"Backtest completed. Final portfolio value: ${self.results['final_capital']:.2f}"
        )

        return self.results

    def _prepare_data(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prepare and filter data for backtesting.

        Args:
            data: Input data
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Filtered DataFrame
        """
        df = data.copy()

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by date
        df = df.sort_index()

        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def _simulate_trading(self, signal_data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate trading based on generated signals.

        Args:
            signal_data: DataFrame with signals and price data

        Returns:
            DataFrame with portfolio performance data
        """
        df = signal_data.copy()

        # Initialize portfolio tracking
        df["position"] = 0.0  # Number of shares held
        df["cash"] = float(self.initial_capital)  # Cash available
        df["portfolio_value"] = float(self.initial_capital)  # Total portfolio value
        df["returns"] = 0.0  # Period returns

        current_position = 0.0
        current_cash = float(self.initial_capital)

        for i in range(len(df)):
            signal = df.iloc[i]["signal"]
            price = df.iloc[i]["close"]

            if pd.isna(signal) or pd.isna(price):
                continue

            # Execute trades based on signals
            if signal == 1 and current_position == 0:  # Buy signal
                # Calculate shares to buy (use all available cash)
                commission_cost = current_cash * self.commission
                available_cash = current_cash - commission_cost
                shares_to_buy = available_cash / price

                if shares_to_buy > 0:
                    current_position = shares_to_buy
                    current_cash = 0

                    # Record trade
                    trade = {
                        "date": df.index[i],
                        "action": "BUY",
                        "shares": shares_to_buy,
                        "price": price,
                        "value": shares_to_buy * price,
                        "commission": commission_cost,
                    }
                    self.trades.append(trade)

            elif signal == -1 and current_position > 0:  # Sell signal
                # Sell all shares
                sell_value = current_position * price
                commission_cost = sell_value * self.commission
                current_cash = sell_value - commission_cost

                # Record trade
                trade = {
                    "date": df.index[i],
                    "action": "SELL",
                    "shares": current_position,
                    "price": price,
                    "value": sell_value,
                    "commission": commission_cost,
                }
                self.trades.append(trade)

                current_position = 0

            # Update portfolio values
            portfolio_value = current_cash + (current_position * price)

            current_date = df.index[i]
            df.loc[current_date, "position"] = current_position
            df.loc[current_date, "cash"] = current_cash
            df.loc[current_date, "portfolio_value"] = portfolio_value

            # Calculate returns
            if i > 0:
                prev_value = df.iloc[i - 1]["portfolio_value"]
                if prev_value > 0:
                    df.loc[current_date, "returns"] = (
                        portfolio_value - prev_value
                    ) / prev_value

        return df

    def _calculate_performance_metrics(self, portfolio_data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            portfolio_data: DataFrame with portfolio performance data

        Returns:
            Dictionary with performance metrics
        """
        # Basic return calculations
        initial_value = portfolio_data["portfolio_value"].iloc[0]
        final_value = portfolio_data["portfolio_value"].iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        # Annualized return
        days = len(portfolio_data)
        years = days / 252  # Assume 252 trading days per year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility (annualized)
        daily_returns = portfolio_data["returns"].dropna()
        volatility = daily_returns.std() * (252**0.5) if len(daily_returns) > 1 else 0

        # Sharpe ratio (assume 2% risk-free rate)
        risk_free_rate = 0.02
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        portfolio_values = portfolio_data["portfolio_value"]
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()

        # Trade statistics
        total_trades = len(self.trades)

        # Calculate trade P&L
        buy_trades = [t for t in self.trades if t["action"] == "BUY"]
        sell_trades = [t for t in self.trades if t["action"] == "SELL"]

        winning_trades = 0
        trade_returns = []

        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                buy = buy_trades[i]
                trade_return = (sell["price"] - buy["price"]) / buy["price"]
                trade_returns.append(trade_return)
                if trade_return > 0:
                    winning_trades += 1

        win_rate = winning_trades / len(trade_returns) if trade_returns else 0

        # Profit factor
        winning_returns = [r for r in trade_returns if r > 0]
        losing_returns = [r for r in trade_returns if r < 0]

        total_wins = sum(winning_returns) if winning_returns else 0
        total_losses = abs(sum(losing_returns)) if losing_returns else 0

        profit_factor = (
            total_wins / total_losses
            if total_losses > 0
            else float("inf") if total_wins > 0 else 0
        )

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results including portfolio value and drawdown.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            raise ValueError("No backtest results to plot. Run backtest first.")

        data = self.results["data"]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Portfolio value
        axes[0].plot(
            data.index, data["portfolio_value"], label="Portfolio Value", color="blue"
        )
        axes[0].axhline(
            y=self.initial_capital, color="red", linestyle="--", label="Initial Capital"
        )
        axes[0].set_title(f'{self.results["strategy_name"]} - Portfolio Performance')
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Returns
        axes[1].plot(
            data.index, data["returns"], label="Daily Returns", color="green", alpha=0.7
        )
        axes[1].set_ylabel("Daily Returns")
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Drawdown
        portfolio_values = data["portfolio_value"]
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak * 100

        axes[2].fill_between(
            data.index, drawdown, 0, color="red", alpha=0.3, label="Drawdown"
        )
        axes[2].set_ylabel("Drawdown (%)")
        axes[2].set_xlabel("Date")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Plot saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print a summary of backtest results."""
        if not self.results:
            raise ValueError("No backtest results to display. Run backtest first.")

        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY: {self.results['strategy_name']}")
        print(f"{'='*60}")
        print(
            f"Period: {self.results['start_date'].date()} to {self.results['end_date'].date()}"
        )
        print(f"Initial Capital: ${self.results['initial_capital']:,.2f}")
        print(f"Final Capital: ${self.results['final_capital']:,.2f}")
        print(f"\nPERFORMANCE METRICS:")
        print(f"Total Return: {self.results['total_return']*100:.2f}%")
        print(f"Annualized Return: {self.results['annualized_return']*100:.2f}%")
        print(f"Volatility: {self.results['volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {self.results['max_drawdown']*100:.2f}%")
        print(f"\nTRADE STATISTICS:")
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Winning Trades: {self.results['winning_trades']}")
        print(f"Win Rate: {self.results['win_rate']*100:.2f}%")
        print(f"Profit Factor: {self.results['profit_factor']:.3f}")
        print(f"{'='*60}")

    def export_results(self, filepath: str):
        """
        Export backtest results to CSV.

        Args:
            filepath: Path to save the CSV file
        """
        if not self.results:
            raise ValueError("No backtest results to export. Run backtest first.")

        # Export portfolio data
        data_path = filepath.replace(".csv", "_data.csv")
        self.results["data"].to_csv(data_path)

        # Export trades
        trades_path = filepath.replace(".csv", "_trades.csv")
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(trades_path, index=False)

        # Export summary metrics
        summary_data = {
            "Metric": [
                "Initial Capital",
                "Final Capital",
                "Total Return (%)",
                "Annualized Return (%)",
                "Volatility (%)",
                "Sharpe Ratio",
                "Max Drawdown (%)",
                "Total Trades",
                "Winning Trades",
                "Win Rate (%)",
                "Profit Factor",
            ],
            "Value": [
                self.results["initial_capital"],
                self.results["final_capital"],
                self.results["total_return"] * 100,
                self.results["annualized_return"] * 100,
                self.results["volatility"] * 100,
                self.results["sharpe_ratio"],
                self.results["max_drawdown"] * 100,
                self.results["total_trades"],
                self.results["winning_trades"],
                self.results["win_rate"] * 100,
                self.results["profit_factor"],
            ],
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filepath, index=False)

        self.logger.info(f"Results exported to {filepath}")


def compare_strategies(
    strategies: List[Dict], data: pd.DataFrame, initial_capital: float = 100000
) -> pd.DataFrame:
    """
    Compare multiple strategies on the same dataset.

    Args:
        strategies: List of dictionaries with 'name', 'strategy' keys
        data: Historical price data
        initial_capital: Starting capital for each strategy

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    for strat_info in strategies:
        name = strat_info["name"]
        strategy = strat_info["strategy"]

        # Run backtest
        backtest = BacktestEngine(initial_capital=initial_capital)
        result = backtest.run_backtest(strategy, data)

        # Extract key metrics
        results.append(
            {
                "Strategy": name,
                "Total Return (%)": result["total_return"] * 100,
                "Annualized Return (%)": result["annualized_return"] * 100,
                "Volatility (%)": result["volatility"] * 100,
                "Sharpe Ratio": result["sharpe_ratio"],
                "Max Drawdown (%)": result["max_drawdown"] * 100,
                "Win Rate (%)": result["win_rate"] * 100,
                "Total Trades": result["total_trades"],
            }
        )

    return pd.DataFrame(results)
