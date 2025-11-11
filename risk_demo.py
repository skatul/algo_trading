#!/usr/bin/env python3
"""
Risk Management System Demo

Demonstrates the advanced risk management capabilities of the trading system.
This script showcases:
1. Position sizing using different methods
2. Comprehensive risk analysis 
3. Portfolio risk monitoring
4. Real-time risk reporting
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.risk_management import (
    PositionSizer, RiskAnalyzer, PortfolioRiskManager,
    RiskLevel, risk_manager
)


def generate_sample_returns(mean_return=0.001, volatility=0.02, periods=252):
    """Generate sample return data for demonstration."""
    np.random.seed(42)
    returns = np.random.normal(mean_return, volatility, periods)
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    return pd.Series(returns, index=dates)


def demo_position_sizing():
    """Demonstrate different position sizing methods."""
    print("=" * 80)
    print("POSITION SIZING DEMONSTRATION")
    print("=" * 80)
    
    sizer = PositionSizer(max_portfolio_risk=0.02)
    portfolio_value = 100000
    
    # Kelly Criterion demonstration
    print("\n1. KELLY CRITERION POSITION SIZING")
    print("-" * 40)
    
    scenarios = [
        ("Conservative Strategy", 0.55, 50, 45),
        ("Aggressive Strategy", 0.65, 80, 60),
        ("High Risk Strategy", 0.70, 120, 100),
        ("Poor Strategy", 0.45, 40, 60)
    ]
    
    for name, win_rate, avg_win, avg_loss in scenarios:
        kelly_fraction = sizer.kelly_criterion(win_rate, avg_win, avg_loss)
        suggested_amount = portfolio_value * kelly_fraction
        print(f"{name:20s}: {kelly_fraction:6.2%} -> ${suggested_amount:8,.0f}")
    
    # ATR-based position sizing
    print("\n2. ATR-BASED POSITION SIZING")
    print("-" * 40)
    
    atr_scenarios = [
        ("Low Volatility Stock", 1.5, 100),
        ("Medium Volatility Stock", 3.0, 75),
        ("High Volatility Stock", 5.0, 50),
        ("Crypto Asset", 8.0, 25)
    ]
    
    for name, atr, price in atr_scenarios:
        shares = sizer.position_size_atr(portfolio_value, atr, price)
        position_value = shares * price
        print(f"{name:22s}: {shares:4d} shares -> ${position_value:8,.0f} ({position_value/portfolio_value:5.1%})")
    
    # Volatility-based position sizing
    print("\n3. VOLATILITY TARGETING")
    print("-" * 40)
    
    vol_scenarios = [
        ("Conservative Bond Fund", 0.05),
        ("Blue Chip Stock", 0.15),
        ("Growth Stock", 0.25),
        ("Penny Stock", 0.45)
    ]
    
    for name, volatility in vol_scenarios:
        fraction = sizer.position_size_volatility(portfolio_value, volatility, 0.15)
        position_value = portfolio_value * fraction
        print(f"{name:22s}: {fraction:6.2%} -> ${position_value:8,.0f}")


def demo_risk_analysis():
    """Demonstrate comprehensive risk analysis."""
    print("\n" + "=" * 80)
    print("RISK ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    analyzer = RiskAnalyzer()
    
    # Generate different risk profiles
    portfolios = {
        "Conservative Portfolio": generate_sample_returns(0.0005, 0.008, 252),
        "Moderate Portfolio": generate_sample_returns(0.001, 0.015, 252),
        "Aggressive Portfolio": generate_sample_returns(0.002, 0.025, 252),
        "High-Risk Portfolio": generate_sample_returns(0.003, 0.035, 252)
    }
    
    benchmark = generate_sample_returns(0.0008, 0.012, 252)  # Market benchmark
    
    print(f"\n{'Portfolio':<20} {'Risk Level':<10} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<10} {'VaR 95%':<10}")
    print("-" * 80)
    
    risk_results = {}
    for name, returns in portfolios.items():
        metrics = analyzer.analyze_portfolio_risk(returns, benchmark)
        risk_results[name] = metrics
        
        print(f"{name:<20} {metrics.risk_level.value:<10} "
              f"{metrics.volatility:>10.1%} {metrics.sharpe_ratio:>6.2f} "
              f"{metrics.max_drawdown:>8.1%} {metrics.var_95:>8.1%}")
    
    # Detailed analysis for one portfolio
    print(f"\nDETAILED ANALYSIS: Aggressive Portfolio")
    print("-" * 50)
    aggressive_metrics = risk_results["Aggressive Portfolio"]
    
    print(f"Risk Classification: {aggressive_metrics.risk_level.value.upper()}")
    print(f"Confidence Score: {aggressive_metrics.confidence_score:.1%}")
    print(f"Annual Volatility: {aggressive_metrics.volatility:.2%}")
    print(f"Maximum Drawdown: {aggressive_metrics.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {aggressive_metrics.sharpe_ratio:.3f}")
    print(f"Sortino Ratio: {aggressive_metrics.sortino_ratio:.3f}")
    print(f"Calmar Ratio: {aggressive_metrics.calmar_ratio:.3f}")
    print(f"VaR (95%): {aggressive_metrics.var_95:.2%}")
    print(f"VaR (99%): {aggressive_metrics.var_99:.2%}")
    print(f"CVaR (95%): {aggressive_metrics.cvar_95:.2%}")
    print(f"CVaR (99%): {aggressive_metrics.cvar_99:.2%}")


def demo_portfolio_risk_management():
    """Demonstrate portfolio-level risk management."""
    print("\n" + "=" * 80)
    print("PORTFOLIO RISK MANAGEMENT DEMONSTRATION")
    print("=" * 80)
    
    # Create a sample portfolio
    positions = {
        'AAPL': 0.12,   # 12% position
        'MSFT': 0.08,   # 8% position
        'GOOGL': 0.06,  # 6% position
        'TSLA': -0.03,  # 3% short position
        'AMZN': 0.05,   # 5% position
    }
    
    # Generate portfolio returns
    portfolio_returns = generate_sample_returns(0.0012, 0.018, 252)
    
    # Test position limit checking
    print("\nPOSITION LIMIT CHECKING:")
    print("-" * 30)
    
    test_positions = [
        ('NVDA', 0.05),   # Valid position
        ('META', 0.15),   # Too large
        ('NFLX', 0.08),   # Valid but check total exposure
    ]
    
    manager = PortfolioRiskManager(
        max_portfolio_risk=0.20,
        max_position_size=0.10,
        max_sector_exposure=0.30
    )
    
    for symbol, size in test_positions:
        is_allowed, reason = manager.check_position_limits(positions, (symbol, size))
        status = "✓ APPROVED" if is_allowed else "✗ REJECTED"
        print(f"{symbol} ({size:5.1%}): {status} - {reason}")
    
    # Generate comprehensive risk report
    print("\nCOMPREHENSIVE RISK REPORT:")
    print("-" * 30)
    report = manager.generate_risk_report(portfolio_returns, positions)
    print(report)


def demo_real_time_monitoring():
    """Demonstrate real-time risk monitoring simulation."""
    print("\n" + "=" * 80)
    print("REAL-TIME RISK MONITORING SIMULATION")
    print("=" * 80)
    
    # Simulate a trading day with position changes
    initial_positions = {'AAPL': 0.05, 'MSFT': 0.03}
    manager = PortfolioRiskManager()
    
    # Simulate position additions throughout the day
    trading_actions = [
        ("09:30", "BUY", "GOOGL", 0.04),
        ("10:15", "BUY", "TSLA", 0.06),
        ("11:00", "SELL", "AMZN", -0.02),  # Short position
        ("13:30", "BUY", "NVDA", 0.08),
        ("14:45", "BUY", "META", 0.12),    # This should be rejected
    ]
    
    current_positions = initial_positions.copy()
    
    print(f"Starting positions: {current_positions}")
    print(f"Initial exposure: {sum(abs(p) for p in current_positions.values()):.1%}")
    print("\nTrading Activity:")
    print("-" * 50)
    
    for time_str, action, symbol, size in trading_actions:
        is_allowed, reason = manager.check_position_limits(current_positions, (symbol, size))
        
        if is_allowed:
            current_positions[symbol] = current_positions.get(symbol, 0) + size
            status = "✓ EXECUTED"
        else:
            status = "✗ BLOCKED"
        
        total_exposure = sum(abs(p) for p in current_positions.values())
        
        print(f"{time_str} {action:4s} {symbol:5s} {size:+6.1%}: {status}")
        print(f"         Reason: {reason}")
        print(f"         Total Exposure: {total_exposure:.1%}")
        print()


def create_risk_visualization():
    """Create visualizations of risk metrics."""
    print("\n" + "=" * 80)
    print("RISK VISUALIZATION")
    print("=" * 80)
    
    # Generate sample data for multiple strategies
    strategies = {
        'Conservative': generate_sample_returns(0.0005, 0.01, 252),
        'Balanced': generate_sample_returns(0.001, 0.015, 252),
        'Growth': generate_sample_returns(0.0015, 0.02, 252),
        'Aggressive': generate_sample_returns(0.002, 0.03, 252),
    }
    
    analyzer = RiskAnalyzer()
    
    # Collect metrics
    risk_data = []
    for name, returns in strategies.items():
        metrics = analyzer.analyze_portfolio_risk(returns)
        risk_data.append({
            'Strategy': name,
            'Volatility': metrics.volatility,
            'Sharpe_Ratio': metrics.sharpe_ratio,
            'Max_Drawdown': abs(metrics.max_drawdown),
            'VaR_95': abs(metrics.var_95),
            'Risk_Level': metrics.risk_level.value
        })
    
    df = pd.DataFrame(risk_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Risk Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Risk vs Return scatter
    axes[0, 0].scatter(df['Volatility'], df['Sharpe_Ratio'], 
                      s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
    for i, strategy in enumerate(df['Strategy']):
        axes[0, 0].annotate(strategy, (df['Volatility'][i], df['Sharpe_Ratio'][i]),
                           xytext=(5, 5), textcoords='offset points')
    axes[0, 0].set_xlabel('Volatility (Annual)')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].set_title('Risk-Return Profile')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Volatility comparison
    axes[0, 1].bar(df['Strategy'], df['Volatility'], color='skyblue', alpha=0.7)
    axes[0, 1].set_ylabel('Annual Volatility')
    axes[0, 1].set_title('Volatility Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Maximum Drawdown
    axes[1, 0].bar(df['Strategy'], df['Max_Drawdown'], color='coral', alpha=0.7)
    axes[1, 0].set_ylabel('Maximum Drawdown')
    axes[1, 0].set_title('Maximum Drawdown Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # VaR comparison
    axes[1, 1].bar(df['Strategy'], df['VaR_95'], color='lightgreen', alpha=0.7)
    axes[1, 1].set_ylabel('VaR (95%)')
    axes[1, 1].set_title('Value at Risk (95%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_path = os.path.join(output_dir, 'risk_analysis_dashboard.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Risk visualization saved to: {plot_path}")
    
    # Display the plot if running interactively
    try:
        plt.show()
    except:
        print("Note: Plot display not available in this environment")


def main():
    """Main demonstration function."""
    print("ADVANCED RISK MANAGEMENT SYSTEM DEMO")
    print("=" * 80)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Run all demonstrations
        demo_position_sizing()
        demo_risk_analysis() 
        demo_portfolio_risk_management()
        demo_real_time_monitoring()
        create_risk_visualization()
        
        print("\n" + "=" * 80)
        print("DEMO SUMMARY")
        print("=" * 80)
        print("✓ Position sizing methods demonstrated")
        print("✓ Risk analysis capabilities shown")
        print("✓ Portfolio risk management tested")
        print("✓ Real-time monitoring simulated")
        print("✓ Risk visualizations created")
        print()
        print("The risk management system is ready for production use!")
        print("Key features:")
        print("  - Kelly Criterion position sizing")
        print("  - ATR and volatility-based sizing")
        print("  - Comprehensive risk metrics (VaR, CVaR, drawdowns)")
        print("  - Real-time position limit monitoring")
        print("  - Risk level classification")
        print("  - Professional reporting and visualization")
        
    except Exception as e:
        print(f"Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)