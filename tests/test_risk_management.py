"""
Tests for risk management utilities.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.utils.risk_management import (
    RiskLevel, RiskMetrics, PositionSizer, RiskAnalyzer,
    PortfolioRiskManager, risk_manager
)


class TestPositionSizer:
    """Test position sizing calculations."""
    
    def test_kelly_criterion_basic(self):
        """Test Kelly criterion calculation."""
        sizer = PositionSizer()
        
        # Favorable scenario
        result = sizer.kelly_criterion(0.6, 100, 50)
        assert 0 < result <= 0.25  # Should be positive but capped
        
        # Unfavorable scenario
        result = sizer.kelly_criterion(0.4, 50, 100)
        assert result == 0.0  # Should be zero
        
        # Edge cases
        assert sizer.kelly_criterion(0, 100, 50) == 0.0
        assert sizer.kelly_criterion(1, 100, 50) == 0.0
        assert sizer.kelly_criterion(0.5, 100, 0) == 0.0
    
    def test_position_size_atr(self):
        """Test ATR-based position sizing."""
        sizer = PositionSizer(max_portfolio_risk=0.02)
        
        # Normal scenario
        shares = sizer.position_size_atr(
            portfolio_value=100000,
            atr=2.0,
            entry_price=100.0,
            stop_loss_distance=2.0
        )
        
        assert shares > 0
        assert shares <= 950  # Max 95% of portfolio
        
        # Edge cases
        assert sizer.position_size_atr(100000, 0, 100, 2.0) == 0
        assert sizer.position_size_atr(100000, 2.0, 100, 0) == 0
    
    def test_position_size_volatility(self):
        """Test volatility-based position sizing."""
        sizer = PositionSizer()
        
        # Normal scenario
        fraction = sizer.position_size_volatility(
            portfolio_value=100000,
            price_volatility=0.20,
            target_volatility=0.15
        )
        
        assert 0 < fraction < 1
        
        # High volatility scenario
        fraction = sizer.position_size_volatility(100000, 0.50, 0.15)
        assert fraction < 1
        
        # Edge case
        assert sizer.position_size_volatility(100000, 0, 0.15) == 0.0


class TestRiskAnalyzer:
    """Test risk analysis calculations."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        return pd.Series(returns)
    
    @pytest.fixture
    def volatile_returns(self):
        """Generate volatile return data."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.05, 252)  # More volatile
        return pd.Series(returns)
    
    def test_calculate_var(self, sample_returns):
        """Test VaR calculation."""
        analyzer = RiskAnalyzer()
        
        var_95 = analyzer.calculate_var(sample_returns, 0.95)
        var_99 = analyzer.calculate_var(sample_returns, 0.99)
        
        assert var_95 < 0  # VaR should be negative (loss)
        assert var_99 < var_95  # 99% VaR should be more extreme
        
        # Empty series
        assert analyzer.calculate_var(pd.Series([]), 0.95) == 0.0
    
    def test_calculate_cvar(self, sample_returns):
        """Test CVaR calculation."""
        analyzer = RiskAnalyzer()
        
        cvar_95 = analyzer.calculate_cvar(sample_returns, 0.95)
        var_95 = analyzer.calculate_var(sample_returns, 0.95)
        
        assert cvar_95 <= var_95  # CVaR should be more extreme than VaR
    
    def test_calculate_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        analyzer = RiskAnalyzer()
        
        # Create returns with known drawdown
        returns = pd.Series([0.1, -0.05, -0.1, -0.05, 0.2, 0.1])
        max_dd, start_idx, end_idx = analyzer.calculate_maximum_drawdown(returns)
        
        assert max_dd < 0  # Drawdown should be negative
        assert isinstance(start_idx, (int, np.integer))
        assert isinstance(end_idx, (int, np.integer))
        
        # Empty series
        max_dd, _, _ = analyzer.calculate_maximum_drawdown(pd.Series([]))
        assert max_dd == 0.0
    
    def test_calculate_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        analyzer = RiskAnalyzer()
        
        sortino = analyzer.calculate_sortino_ratio(sample_returns)
        assert isinstance(sortino, (float, int))
        
        # All positive returns should give high Sortino
        positive_returns = pd.Series([0.01] * 100)
        sortino_positive = analyzer.calculate_sortino_ratio(positive_returns)
        assert sortino_positive > 0
        
        # Empty series
        assert analyzer.calculate_sortino_ratio(pd.Series([])) == 0.0
    
    def test_analyze_portfolio_risk(self, sample_returns, volatile_returns):
        """Test comprehensive risk analysis."""
        analyzer = RiskAnalyzer()
        
        # Normal risk analysis
        metrics = analyzer.analyze_portfolio_risk(sample_returns)
        assert isinstance(metrics, RiskMetrics)
        assert metrics.var_95 < 0
        assert metrics.volatility > 0
        assert isinstance(metrics.risk_level, RiskLevel)
        assert 0 <= metrics.confidence_score <= 1
        
        # Volatile returns should have higher risk level
        volatile_metrics = analyzer.analyze_portfolio_risk(volatile_returns)
        assert volatile_metrics.volatility > metrics.volatility
        
        # Empty series
        empty_metrics = analyzer.analyze_portfolio_risk(pd.Series([]))
        assert empty_metrics.var_95 == 0.0
        assert empty_metrics.risk_level == RiskLevel.LOW
    
    def test_assess_risk_level(self):
        """Test risk level assessment."""
        analyzer = RiskAnalyzer()
        
        # Test different scenarios
        assert analyzer._assess_risk_level(0.10, -0.05, -0.01) == RiskLevel.LOW
        assert analyzer._assess_risk_level(0.25, -0.15, -0.03) == RiskLevel.MODERATE
        assert analyzer._assess_risk_level(0.40, -0.35, -0.06) == RiskLevel.CRITICAL
    
    def test_calculate_confidence_score(self, sample_returns):
        """Test confidence score calculation."""
        analyzer = RiskAnalyzer()
        
        # Sufficient data
        score = analyzer._calculate_confidence_score(sample_returns)
        assert 0 <= score <= 1
        
        # Insufficient data
        short_returns = pd.Series([0.01] * 10)
        short_score = analyzer._calculate_confidence_score(short_returns)
        assert short_score == 0.0
        
        # With benchmark
        benchmark = pd.Series(np.random.normal(0.0005, 0.015, len(sample_returns)))
        score_with_benchmark = analyzer._calculate_confidence_score(
            sample_returns, benchmark
        )
        assert 0 <= score_with_benchmark <= 1


class TestPortfolioRiskManager:
    """Test portfolio risk management."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance."""
        return PortfolioRiskManager(
            max_portfolio_risk=0.15,
            max_position_size=0.10,
            max_sector_exposure=0.30
        )
    
    @pytest.fixture
    def sample_positions(self):
        """Sample position data."""
        return {
            'AAPL': 0.05,
            'GOOGL': 0.08,
            'MSFT': 0.06,
            'TSLA': -0.03  # Short position
        }
    
    def test_check_position_limits(self, risk_manager, sample_positions):
        """Test position limit checking."""
        # Valid position
        is_allowed, reason = risk_manager.check_position_limits(
            sample_positions, ('NVDA', 0.05)
        )
        assert is_allowed
        assert "approved" in reason.lower()
        
        # Position too large
        is_allowed, reason = risk_manager.check_position_limits(
            sample_positions, ('NVDA', 0.15)
        )
        assert not is_allowed
        assert "exceeds limit" in reason.lower()
        
        # Total exposure too high
        is_allowed, reason = risk_manager.check_position_limits(
            sample_positions, ('NVDA', 0.80)
        )
        assert not is_allowed
        assert "exceeds limit" in reason.lower()
    
    def test_generate_risk_report(self, risk_manager):
        """Test risk report generation."""
        # Generate sample data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        positions = {'AAPL': 0.05, 'GOOGL': 0.08}
        
        report = risk_manager.generate_risk_report(returns, positions)
        
        assert "PORTFOLIO RISK ANALYSIS REPORT" in report
        assert "RISK METRICS" in report
        assert "VALUE AT RISK" in report
        assert "CURRENT POSITIONS" in report
        assert "AAPL" in report
        assert "GOOGL" in report
        
        # Test with empty positions
        report_empty = risk_manager.generate_risk_report(returns, {})
        assert "No active positions" in report_empty
    
    def test_initialization(self):
        """Test risk manager initialization."""
        manager = PortfolioRiskManager(
            max_portfolio_risk=0.20,
            max_position_size=0.15,
            max_sector_exposure=0.40
        )
        
        assert manager.max_portfolio_risk == 0.20
        assert manager.max_position_size == 0.15
        assert manager.max_sector_exposure == 0.40
        assert hasattr(manager, 'risk_analyzer')
        assert hasattr(manager, 'position_sizer')


class TestIntegration:
    """Integration tests for risk management components."""
    
    def test_global_risk_manager(self):
        """Test global risk manager instance."""
        assert risk_manager is not None
        assert isinstance(risk_manager, PortfolioRiskManager)
        assert hasattr(risk_manager, 'check_position_limits')
        assert hasattr(risk_manager, 'generate_risk_report')
    
    def test_end_to_end_risk_analysis(self):
        """Test complete risk analysis workflow."""
        # Generate sample portfolio data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = pd.Series(
            np.random.normal(0.0008, 0.018, 252),
            index=dates
        )
        
        positions = {
            'AAPL': 0.08,
            'MSFT': 0.06,
            'GOOGL': 0.05,
            'TSLA': -0.02
        }
        
        # Analyze risk
        manager = PortfolioRiskManager()
        report = manager.generate_risk_report(returns, positions)
        
        # Verify report contains expected elements
        assert len(report) > 100  # Should be substantial report
        assert "Risk Level:" in report
        assert "Volatility" in report
        assert "Sharpe Ratio" in report
        
        # Test position limit checking
        is_allowed, _ = manager.check_position_limits(positions, ('NVDA', 0.04))
        assert is_allowed
        
        is_allowed, _ = manager.check_position_limits(positions, ('NVDA', 0.20))
        assert not is_allowed
    
    def test_risk_metrics_consistency(self):
        """Test consistency of risk metrics across components."""
        analyzer = RiskAnalyzer()
        
        # Generate test data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.025, 252))
        
        metrics = analyzer.analyze_portfolio_risk(returns)
        
        # Verify metric relationships
        assert metrics.var_99 <= metrics.var_95  # 99% VaR more extreme
        assert metrics.cvar_99 <= metrics.cvar_95  # Same for CVaR
        assert metrics.volatility > 0
        assert metrics.max_drawdown <= 0  # Drawdown should be negative
        
        # Risk level should match volatility
        if metrics.volatility > 0.30:
            assert metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        elif metrics.volatility < 0.10:
            assert metrics.risk_level in [RiskLevel.LOW, RiskLevel.MODERATE]


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        sizer = PositionSizer()
        analyzer = RiskAnalyzer()
        
        # Test with invalid inputs (negative values)
        assert sizer.kelly_criterion(-0.1, 100, 50) == 0.0
        assert analyzer.calculate_var(pd.Series([]), 0.95) == 0.0
        
        # Test with edge case confidence levels
        returns = pd.Series([0.01, -0.02, 0.015])
        var = analyzer.calculate_var(returns, 0.99)  # Valid high confidence
        assert isinstance(var, float)
        
        var = analyzer.calculate_var(returns, 0.01)  # Valid low confidence
        assert isinstance(var, float)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        analyzer = RiskAnalyzer()
        manager = PortfolioRiskManager()
        
        empty_returns = pd.Series([])
        
        # Risk analysis with empty data
        metrics = analyzer.analyze_portfolio_risk(empty_returns)
        assert metrics.risk_level == RiskLevel.LOW
        assert metrics.confidence_score == 0.0
        
        # Risk report with empty data
        report = manager.generate_risk_report(empty_returns, {})
        assert "No active positions" in report
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        analyzer = RiskAnalyzer()
        
        # Extreme positive returns (with variation to create volatility)
        np.random.seed(42)
        extreme_positive = pd.Series(np.random.normal(0.1, 0.05, 100))  # High mean with volatility
        metrics = analyzer.analyze_portfolio_risk(extreme_positive)
        assert metrics.volatility > 0.20  # Should have high volatility
        
        # Extreme negative returns
        extreme_negative = pd.Series([-0.1] * 100)  # -10% daily returns
        metrics = analyzer.analyze_portfolio_risk(extreme_negative)
        assert metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]  # Accept both high and critical
        assert metrics.max_drawdown < -0.5  # Significant drawdown


if __name__ == '__main__':
    pytest.main([__file__])