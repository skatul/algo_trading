"""
Advanced risk management utilities for the trading system.

This module provides comprehensive risk analysis, position sizing,
and portfolio risk management capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging


@dataclass
class RiskConfig:
    """Configuration for risk management parameters."""
    max_kelly_fraction: float = 0.25
    max_position_size: float = 0.10
    max_portfolio_risk: float = 0.15
    max_sector_exposure: float = 0.30
    confidence_levels: Tuple[float, ...] = (0.95, 0.99)
    min_confidence_score: float = 0.5


class ValidationUtils:
    """Common validation utilities for risk management."""
    
    @staticmethod
    def validate_numeric_inputs(*args) -> bool:
        """Validate that all inputs are numeric."""
        return all(isinstance(arg, (int, float)) for arg in args)
    
    @staticmethod
    def validate_probability(value: float, name: str = "probability") -> None:
        """Validate that a value is a valid probability (0-1)."""
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
    @staticmethod
    def validate_positive(value: float, name: str = "value") -> None:
        """Validate that a value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Container for risk analysis results."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    risk_level: RiskLevel
    confidence_score: float


class PositionSizer:
    """Advanced position sizing based on risk parameters."""
    
    def __init__(self, max_portfolio_risk: float = 0.02):
        """
        Initialize position sizer.
        
        Args:
            max_portfolio_risk: Maximum portfolio risk per trade (default 2%)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.logger = logging.getLogger(__name__)
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive)
            
        Returns:
            Optimal position size fraction (0-1)
            
        Raises:
            ValueError: If input parameters are invalid
        """
        # Enhanced input validation using utilities
        if not ValidationUtils.validate_numeric_inputs(win_rate, avg_win, avg_loss):
            raise ValueError("All parameters must be numeric")
        
        ValidationUtils.validate_probability(win_rate, "Win rate")
        ValidationUtils.validate_positive(avg_win, "Average win amount")
        
        if avg_loss <= 0:
            self.logger.warning("Average loss is non-positive, returning 0 position size")
            return 0.0
        
        if win_rate <= 0 or win_rate >= 1:
            self.logger.warning(f"Win rate {win_rate} is at extremes, returning 0 position size")
            return 0.0
        
        try:
            b = avg_win / avg_loss  # Odds received on the wager
            p = win_rate  # Probability of winning
            q = 1 - p  # Probability of losing
            
            kelly_fraction = (b * p - q) / b
            
            # Cap at reasonable limits
            result = max(0.0, min(kelly_fraction, 0.25))  # Never risk more than 25%
            
            self.logger.debug(f"Kelly Criterion: win_rate={win_rate}, avg_win={avg_win}, avg_loss={avg_loss}, result={result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly Criterion: {e}")
            return 0.0
    
    def position_size_atr(
        self, 
        portfolio_value: float, 
        atr: float, 
        entry_price: float,
        stop_loss_distance: float = 2.0
    ) -> int:
        """
        Calculate position size based on Average True Range (ATR).
        
        Args:
            portfolio_value: Current portfolio value
            atr: Average True Range
            entry_price: Entry price for the position
            stop_loss_distance: Stop loss in ATR multiples
            
        Returns:
            Number of shares to buy
        """
        risk_amount = portfolio_value * self.max_portfolio_risk
        stop_loss_amount = atr * stop_loss_distance
        
        if stop_loss_amount <= 0:
            return 0
        
        shares = int(risk_amount / stop_loss_amount)
        max_affordable = int((portfolio_value * 0.95) / entry_price)  # 95% max allocation
        
        return min(shares, max_affordable)
    
    def position_size_volatility(
        self,
        portfolio_value: float,
        price_volatility: float,
        target_volatility: float = 0.15
    ) -> float:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            portfolio_value: Current portfolio value
            price_volatility: Asset price volatility (annualized)
            target_volatility: Target portfolio volatility
            
        Returns:
            Position size as fraction of portfolio
        """
        if price_volatility <= 0:
            return 0.0
        
        position_fraction = target_volatility / price_volatility
        return min(position_fraction, 1.0)


class RiskAnalyzer:
    """Comprehensive risk analysis for trading strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (0.95 for 95% VaR)
            
        Returns:
            VaR value (negative number indicating potential loss)
        """
        if len(returns) == 0:
            return 0.0
        
        return float(np.percentile(returns, (1 - confidence_level) * 100))
    
    def calculate_cvar(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value (expected loss beyond VaR)
        """
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_maximum_drawdown(self, returns: pd.Series) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and its duration.
        
        Args:
            returns: Series of returns
            
        Returns:
            Tuple of (max_drawdown, start_index, end_index)
        """
        if len(returns) == 0:
            return 0.0, 0, 0
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()
        
        # Find start of drawdown
        max_dd_start = rolling_max.loc[:max_dd_end].idxmax()
        
        return max_dd, max_dd_start, max_dd_end
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino ratio (return/downside deviation).
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        
        if downside_deviation == 0:
            return float('inf')
        
        return (returns.mean() * 252 - risk_free_rate) / downside_deviation
    
    def analyze_portfolio_risk(
        self, 
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Comprehensive risk analysis of a portfolio.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            RiskMetrics object with comprehensive analysis
        """
        if len(returns) == 0:
            return RiskMetrics(
                var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
                max_drawdown=0.0, volatility=0.0, sharpe_ratio=0.0,
                sortino_ratio=0.0, calmar_ratio=0.0,
                risk_level=RiskLevel.LOW, confidence_score=0.0
            )
        
        # Calculate risk metrics
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        
        max_dd, _, _ = self.calculate_maximum_drawdown(returns)
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Ratios
        mean_return = returns.mean() * 252  # Annualized
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = mean_return / abs(max_dd) if max_dd < 0 else float('inf')
        
        # Risk level assessment
        risk_level = self._assess_risk_level(volatility, max_dd, var_95)
        confidence_score = self._calculate_confidence_score(returns, benchmark_returns)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            risk_level=risk_level,
            confidence_score=confidence_score
        )
    
    def _assess_risk_level(
        self, 
        volatility: float, 
        max_drawdown: float, 
        var_95: float
    ) -> RiskLevel:
        """Assess overall risk level based on key metrics."""
        risk_score = 0
        
        # Volatility scoring
        if volatility > 0.30:  # 30%+ volatility
            risk_score += 3
        elif volatility > 0.20:  # 20-30%
            risk_score += 2
        elif volatility > 0.15:  # 15-20%
            risk_score += 1
        
        # Drawdown scoring
        if max_drawdown < -0.30:  # >30% drawdown
            risk_score += 3
        elif max_drawdown < -0.20:  # 20-30%
            risk_score += 2
        elif max_drawdown < -0.10:  # 10-20%
            risk_score += 1
        
        # VaR scoring
        if var_95 < -0.05:  # >5% daily VaR
            risk_score += 3
        elif var_95 < -0.03:  # 3-5%
            risk_score += 2
        elif var_95 < -0.02:  # 2-3%
            risk_score += 1
        
        # Risk level classification
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _calculate_confidence_score(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> float:
        """Calculate confidence score based on statistical significance."""
        if len(returns) < 30:  # Insufficient data
            return 0.0
        
        # Base confidence on sample size and consistency
        base_score = min(len(returns) / 252, 1.0)  # Up to 1 year of data = 100%
        
        # Adjust for consistency (lower volatility = higher confidence)
        volatility_penalty = min(returns.std() * 4, 0.5)  # Max 50% penalty
        consistency_score = max(0.0, base_score - volatility_penalty)
        
        # Bonus for outperforming benchmark
        benchmark_bonus = 0.0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            try:
                excess_returns = returns - benchmark_returns
                if excess_returns.mean() > 0:
                    benchmark_bonus = 0.2  # 20% bonus for outperformance
            except Exception:
                pass  # Skip if benchmark comparison fails
        
        return min(consistency_score + benchmark_bonus, 1.0)


class PortfolioRiskManager:
    """Portfolio-level risk management and monitoring."""
    
    def __init__(
        self,
        max_portfolio_risk: float = 0.15,
        max_position_size: float = 0.10,
        max_sector_exposure: float = 0.30
    ):
        """
        Initialize portfolio risk manager.
        
        Args:
            max_portfolio_risk: Maximum portfolio volatility target
            max_position_size: Maximum single position size
            max_sector_exposure: Maximum sector concentration
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.logger = logging.getLogger(__name__)
        self.risk_analyzer = RiskAnalyzer()
        self.position_sizer = PositionSizer(max_portfolio_risk * 0.5)  # 50% of portfolio risk per trade
    
    def check_position_limits(
        self,
        current_positions: Dict[str, float],
        new_position: Tuple[str, float]
    ) -> Tuple[bool, str]:
        """
        Check if new position violates risk limits.
        
        Args:
            current_positions: Dictionary of {symbol: position_size}
            new_position: Tuple of (symbol, proposed_size)
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        symbol, size = new_position
        
        # Check individual position size
        if abs(size) > self.max_position_size:
            return False, f"Position size {size:.2%} exceeds limit {self.max_position_size:.2%}"
        
        # Check total portfolio exposure
        total_exposure = sum(abs(pos) for pos in current_positions.values()) + abs(size)
        if total_exposure > 1.0:  # 100% exposure
            return False, f"Total exposure {total_exposure:.2%} would exceed 100%"
        
        return True, "Position approved"
    
    def generate_risk_report(
        self,
        portfolio_returns: pd.Series,
        positions: Dict[str, float],
        benchmark_returns: Optional[pd.Series] = None
    ) -> str:
        """Generate comprehensive risk report."""
        risk_metrics = self.risk_analyzer.analyze_portfolio_risk(
            portfolio_returns, benchmark_returns
        )
        
        report = [
            "=" * 60,
            "PORTFOLIO RISK ANALYSIS REPORT",
            "=" * 60,
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Period: {len(portfolio_returns)} observations",
            "",
            "RISK METRICS:",
            f"  Overall Risk Level: {risk_metrics.risk_level.value.upper()}",
            f"  Confidence Score: {risk_metrics.confidence_score:.1%}",
            f"  Volatility (Annual): {risk_metrics.volatility:.2%}",
            f"  Maximum Drawdown: {risk_metrics.max_drawdown:.2%}",
            "",
            "VALUE AT RISK:",
            f"  VaR (95%): {risk_metrics.var_95:.2%}",
            f"  VaR (99%): {risk_metrics.var_99:.2%}",
            f"  CVaR (95%): {risk_metrics.cvar_95:.2%}",
            f"  CVaR (99%): {risk_metrics.cvar_99:.2%}",
            "",
            "RISK-ADJUSTED RETURNS:",
            f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.3f}",
            f"  Sortino Ratio: {risk_metrics.sortino_ratio:.3f}",
            f"  Calmar Ratio: {risk_metrics.calmar_ratio:.3f}",
            "",
            "CURRENT POSITIONS:",
        ]
        
        if positions:
            total_exposure = sum(abs(pos) for pos in positions.values())
            report.append(f"  Total Exposure: {total_exposure:.2%}")
            report.append(f"  Number of Positions: {len(positions)}")
            report.append("  Position Breakdown:")
            for symbol, size in sorted(positions.items(), key=lambda x: abs(x[1]), reverse=True):
                report.append(f"    {symbol}: {size:+.2%}")
        else:
            report.append("  No active positions")
        
        report.extend([
            "",
            "RISK LIMITS:",
            f"  Max Portfolio Risk: {self.max_portfolio_risk:.2%}",
            f"  Max Position Size: {self.max_position_size:.2%}",
            f"  Max Sector Exposure: {self.max_sector_exposure:.2%}",
            "=" * 60
        ])
        
        return "\n".join(report)


# Global risk manager instance
risk_manager = PortfolioRiskManager()