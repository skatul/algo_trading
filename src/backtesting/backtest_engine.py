"""
Backtesting Engine for Algorithmic Trading System

This module provides comprehensive backtesting capabilities for trading strategies,
including performance metrics, visualization, and trade analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs
import math
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
        
        # Legacy interface for backward compatibility
        self.cash = initial_capital
        self.shares = 0
        self.transactions = []

    @staticmethod
    def _safe_float_conversion(value, default: float = 0.0) -> float:
        """
        Safely convert a value to float, handling NaN and infinity cases.
        
        For array-like inputs (pandas Series, numpy arrays):
        - Series: Extracts first element if non-empty, else returns default
        - numpy arrays: Extracts single element if size==1, else returns default
        - Multi-element arrays return default (intended for scalar conversion only)
        
        Args:
            value: Value to convert (scalar, Series, array from QuantStats, etc.)
            default: Default value to return if conversion fails or value is NaN/inf
            
        Returns:
            Converted float value or default if value is NaN/infinity/unconvertible
            
        Note:
            This method is designed for converting QuantStats metrics to float.
            Non-scalar arrays are intentionally converted to default for safety.
        """
        # Handle pandas Series by extracting first value
        if hasattr(value, 'iloc'):
            if len(value) == 0:
                logging.warning("Empty pandas Series provided to _safe_float_conversion, using default")
                return default
            logging.debug(f"Extracting first element from pandas Series of length {len(value)}")
            return BacktestEngine._safe_float_conversion(value.iloc[0], default)
            
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                logging.warning("Empty numpy array provided to _safe_float_conversion, using default")
                return default
            elif value.size == 1:
                logging.debug("Extracting single element from numpy array")
                return BacktestEngine._safe_float_conversion(value.item(), default)
            else:
                logging.warning(f"Multi-element numpy array (size {value.size}) provided to _safe_float_conversion, using default")
                return default
        
        # Check for NaN using pandas (for scalars)
        try:
            if pd.isna(value):
                return default
        except (ValueError, TypeError):
            # If pd.isna fails, continue with other checks
            pass
        
        # Convert to float
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            return default
        
        # Check for infinity
        if math.isinf(float_value):
            return default
            
        return float_value

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
            "losing_trades": performance["losing_trades"],
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
        Calculate comprehensive performance metrics using QuantStats and custom calculations.

        Args:
            portfolio_data: DataFrame with portfolio performance data

        Returns:
            Dictionary with performance metrics
        """
        # Prepare returns data for QuantStats
        returns = portfolio_data["returns"].dropna()
        returns.name = "Strategy Returns"
        
        # Handle edge case where there are no returns
        if len(returns) == 0:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "best_day": 0.0,
                "worst_day": 0.0,
            }
        
        try:
            # Use QuantStats for advanced metrics
            total_return = qs.stats.comp(returns)
            annualized_return = qs.stats.cagr(returns)
            volatility = qs.stats.volatility(returns)
            sharpe_ratio = qs.stats.sharpe(returns)
            sortino_ratio = qs.stats.sortino(returns)
            calmar_ratio = qs.stats.calmar(returns)
            max_drawdown = qs.stats.max_drawdown(returns)
            
            # Risk metrics (VaR/CVaR are reported as raw negative values for losses)
            # VaR: Value at Risk at 95% confidence level (5th percentile of returns)
            # CVaR: Conditional Value at Risk (expected return in worst 5% of cases)
            # Note: Negative values indicate losses, consistent with risk convention
            var_95 = qs.stats.var(returns)
            cvar_95 = qs.stats.cvar(returns)
            
            # Distribution metrics
            skewness = qs.stats.skew(returns)
            kurtosis = qs.stats.kurtosis(returns)
            best_day = qs.stats.best(returns)
            worst_day = qs.stats.worst(returns)
            
        except Exception as e:
            self.logger.warning(f"QuantStats calculation failed: {e}. Using fallback calculations.")
            # Fallback to basic calculations if QuantStats fails
            initial_value = portfolio_data["portfolio_value"].iloc[0]
            final_value = portfolio_data["portfolio_value"].iloc[-1]
            total_return = (final_value - initial_value) / initial_value
            
            days = len(portfolio_data)
            years = days / 252
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            volatility = returns.std() * (252**0.5) if len(returns) > 1 else 0
            risk_free_rate = 0.02
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            portfolio_values = portfolio_data["portfolio_value"]
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = drawdown.min()
            
            # Set defaults for enhanced metrics
            sortino_ratio = sharpe_ratio  # Approximation
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Fallback VaR/CVaR calculations (maintaining negative values for losses)
            var_95 = returns.quantile(0.05) if len(returns) > 0 else 0  # 5th percentile
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            skewness = returns.skew() if len(returns) > 2 else 0
            kurtosis = returns.kurtosis() if len(returns) > 2 else 0
            best_day = returns.max() if len(returns) > 0 else 0
            worst_day = returns.min() if len(returns) > 0 else 0

        # Trade statistics
        total_trades = len(self.trades)
        buy_trades = [t for t in self.trades if t["action"] == "BUY"]
        sell_trades = [t for t in self.trades if t["action"] == "SELL"]

        winning_trades = 0
        losing_trades = 0
        trade_returns = []

        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                buy = buy_trades[i]
                trade_return = (sell["price"] - buy["price"]) / buy["price"]
                trade_returns.append(trade_return)
                if trade_return > 0:
                    winning_trades += 1
                elif trade_return < 0:
                    losing_trades += 1

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
            "total_return": self._safe_float_conversion(total_return),
            "annualized_return": self._safe_float_conversion(annualized_return),
            "volatility": self._safe_float_conversion(volatility),
            "sharpe_ratio": self._safe_float_conversion(sharpe_ratio),
            "sortino_ratio": self._safe_float_conversion(sortino_ratio),
            "calmar_ratio": self._safe_float_conversion(calmar_ratio),
            "max_drawdown": self._safe_float_conversion(max_drawdown),
            "var_95": self._safe_float_conversion(var_95),
            "cvar_95": self._safe_float_conversion(cvar_95),
            "skewness": self._safe_float_conversion(skewness),
            "kurtosis": self._safe_float_conversion(kurtosis),
            "best_day": self._safe_float_conversion(best_day),
            "worst_day": self._safe_float_conversion(worst_day),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
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
        """Print a comprehensive summary of backtest results."""
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
        print(f"\n📊 RETURN METRICS:")
        print(f"Total Return: {self.results['total_return']*100:.2f}%")
        print(f"Annualized Return: {self.results['annualized_return']*100:.2f}%")
        print(f"Volatility: {self.results['volatility']*100:.2f}%")
        print(f"\n📈 RISK-ADJUSTED METRICS:")
        print(f"Sharpe Ratio: {self.results.get('sharpe_ratio', 'N/A')}")
        if 'sortino_ratio' in self.results:
            print(f"Sortino Ratio: {self.results['sortino_ratio']:.3f}")
        if 'calmar_ratio' in self.results:
            print(f"Calmar Ratio: {self.results['calmar_ratio']:.3f}")
        print(f"\n⚠️  RISK METRICS:")
        print(f"Max Drawdown: {self.results['max_drawdown']*100:.2f}%")
        if 'var_95' in self.results:
            print(f"VaR (95%): {self.results['var_95']*100:.2f}% (worst loss at 95% confidence)")
        if 'cvar_95' in self.results:
            print(f"CVaR (95%): {self.results['cvar_95']*100:.2f}% (expected loss in worst 5%)")
        if any(key in self.results for key in ['skewness', 'kurtosis', 'best_day', 'worst_day']):
            print(f"\n📊 DISTRIBUTION METRICS:")
            if 'skewness' in self.results:
                print(f"Skewness: {self.results['skewness']:.3f}")
            if 'kurtosis' in self.results:
                print(f"Kurtosis: {self.results['kurtosis']:.3f}")
            if 'best_day' in self.results:
                print(f"Best Day: {self.results['best_day']*100:.2f}%")
            if 'worst_day' in self.results:
                print(f"Worst Day: {self.results['worst_day']*100:.2f}%")
        print(f"\n🔄 TRADE STATISTICS:")
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Winning Trades: {self.results['winning_trades']}")
        print(f"Win Rate: {self.results['win_rate']*100:.2f}%")
        print(f"Profit Factor: {'∞ (no losses)' if self.results['profit_factor'] == float('inf') else f'{self.results['profit_factor']:.3f}'}")
        print(f"{'='*60}")
        print("📈 Enhanced with QuantStats - Professional Portfolio Analytics")

    def generate_quantstats_report(
        self, 
        output_path: Optional[str] = None, 
        benchmark_symbol: Optional[str] = "SPY",
        title: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive QuantStats HTML report.
        
        Args:
            output_path: Path for the HTML report (default: auto-generated)
            benchmark_symbol: Benchmark symbol for comparison (default: SPY)
            title: Custom title for the report
            
        Returns:
            Path to the generated report
        """
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Extract returns data
        data = self.results["data"]
        if "returns" in data.columns:
            returns = data["returns"].dropna()
        elif "portfolio_value" in data.columns:
            returns = data["portfolio_value"].pct_change().dropna()
        else:
            raise ValueError("No returns data available in results.")
        
        returns.name = "Strategy Returns"
        
        # Set default output path
        if output_path is None:
            strategy_name = self.results["strategy_name"].replace(" ", "_")
            output_path = f"{strategy_name}_quantstats_report.html"
        
        # Set default title
        if title is None:
            title = f'{self.results["strategy_name"]} Performance Report'
        
        try:
            # Fetch benchmark data if requested
            benchmark_returns = None
            if benchmark_symbol:
                try:
                    import yfinance as yf
                    
                    # Get date range from results
                    start_date = self.results["start_date"].strftime("%Y-%m-%d")
                    end_date = self.results["end_date"].strftime("%Y-%m-%d")
                    
                    benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date)
                    if benchmark_data is not None and not benchmark_data.empty:
                        benchmark_returns = benchmark_data["Close"].pct_change().dropna()
                        benchmark_returns.name = f"{benchmark_symbol} Benchmark"
                        
                        # Align dates with strategy returns
                        benchmark_returns = benchmark_returns.reindex(returns.index).ffill()
                        
                        self.logger.info(f"Using {benchmark_symbol} as benchmark for comparison")
                    else:
                        self.logger.warning(f"Could not fetch benchmark data for {benchmark_symbol}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to fetch benchmark data: {e}")
            
            # Generate the report
            if benchmark_returns is not None:
                qs.reports.html(
                    returns, 
                    benchmark=benchmark_returns,
                    output=output_path,
                    title=title
                )
                self.logger.info(f"QuantStats report with benchmark generated: {output_path}")
            else:
                qs.reports.html(
                    returns,
                    output=output_path, 
                    title=title
                )
                self.logger.info(f"QuantStats report generated: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate QuantStats report: {e}")
            # Fallback to basic report
            try:
                qs.reports.html(returns, output=output_path, title="Basic Performance Report")
                return output_path
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to generate any report: {fallback_error}")

    def calculate_transaction_cost(self, amount: float) -> float:
        """Calculate transaction cost for a given amount."""
        return amount * self.commission

    def execute_buy_order(self, price: float, date: datetime) -> None:
        """Execute a buy order (legacy interface for backward compatibility)."""
        available_cash = self.cash * (1 - self.commission)
        shares_to_buy = available_cash // price
        
        if shares_to_buy > 0:
            cost = shares_to_buy * price * (1 + self.commission)
            self.cash -= cost
            self.shares += shares_to_buy
            
            transaction = {
                "date": date,
                "action": "BUY",
                "shares": shares_to_buy,
                "price": price,
                "cost": cost
            }
            self.transactions.append(transaction)

    def execute_sell_order(self, price: float, date: datetime) -> None:
        """Execute a sell order (legacy interface for backward compatibility)."""
        if self.shares > 0:
            proceeds = self.shares * price * (1 - self.commission)
            shares_sold = self.shares
            
            self.cash += proceeds
            self.shares = 0
            
            transaction = {
                "date": date,
                "action": "SELL",
                "shares": shares_sold,
                "price": price,
                "proceeds": proceeds
            }
            self.transactions.append(transaction)

    def calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        return self.cash + (self.shares * current_price)

    def calculate_max_drawdown(self, portfolio_values) -> float:
        """Calculate maximum drawdown from portfolio values."""
        # Convert to pandas Series if it's a list
        if not isinstance(portfolio_values, pd.Series):
            portfolio_values = pd.Series(portfolio_values)
            
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = returns.mean() * 252  # Annualized
        volatility = returns.std() * (252 ** 0.5)  # Annualized
        
        if volatility == 0:
            return 0.0
            
        return (mean_return - risk_free_rate) / volatility

    def calculate_metrics(self, portfolio_values=None, returns=None) -> Dict:
        """Calculate metrics (legacy interface)."""
        if portfolio_values is not None and returns is not None:
            # Legacy interface with separate portfolio values and returns
            # Ensure we handle different array lengths properly
            if isinstance(returns, pd.Series):
                returns_data = returns
            else:
                returns_data = pd.Series(returns)
                
            if isinstance(portfolio_values, (list, np.ndarray)):
                # Create aligned portfolio data - returns should be one less than values
                if len(portfolio_values) == len(returns_data) + 1:
                    # Use returns as-is and align portfolio values
                    portfolio_data = pd.DataFrame({
                        'returns': returns_data,
                        'portfolio_value': portfolio_values[1:]  # Skip first value
                    })
                elif len(portfolio_values) == len(returns_data):
                    portfolio_data = pd.DataFrame({
                        'returns': returns_data,
                        'portfolio_value': portfolio_values
                    })
                else:
                    # Fallback - just use returns
                    portfolio_data = pd.DataFrame({'returns': returns_data})
            else:
                portfolio_data = pd.DataFrame({'returns': returns_data})
        elif isinstance(portfolio_values, pd.DataFrame):
            # Modern interface with portfolio_data DataFrame
            portfolio_data = portfolio_values
        else:
            # Fallback to empty DataFrame
            portfolio_data = pd.DataFrame({'returns': []})
            
        return self._calculate_performance_metrics(portfolio_data)
    

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
