# Algorithmic Trading System

A comprehensive Python-based algorithmic trading system with backtesting, risk management, and portfolio optimization capabilities.

## Features

- **Data Management**: Real-time and historical data fetching from multiple sources
- **Strategy Framework**: Modular strategy implementation with common technical indicators
- **Backtesting Engine**: Comprehensive backtesting with performance metrics and QuantStats integration
- **Advanced Risk Management**: 
  - Position sizing using Kelly Criterion, ATR, and volatility targeting
  - Value at Risk (VaR) and Conditional VaR analysis
  - Maximum drawdown monitoring and risk level classification
  - Real-time portfolio risk monitoring and limit enforcement
- **Performance Analytics**: Professional-grade metrics including Sharpe, Sortino, and Calmar ratios
- **Portfolio Management**: Position tracking, P&L calculation, and capital allocation
- **Live Trading**: Paper trading and live execution capabilities

## Project Structure

```
algo_trading/
├── src/
│   ├── data/              # Data fetching and management
│   ├── strategies/        # Trading strategies
│   ├── backtesting/       # Backtesting engine
│   └── utils/             # Utility functions (config, risk management, performance)
├── logs/                  # System logs
├── output/                # Generated reports and visualizations
├── tests/                 # Unit tests
├── demo.py               # Basic system demonstration
├── risk_demo.py          # Risk management system demo
└── requirements.txt      # Python dependencies
```

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demonstration scripts:

```bash
# Basic system demonstration
python demo.py

# Advanced risk management demonstration
python risk_demo.py
```

Or use programmatically:

```python
from src.main import TradingEngine
from src.utils.risk_management import PortfolioRiskManager

# Initialize trading engine
engine = TradingEngine()

# Run backtest with risk management
results = engine.backtest(strategy='moving_average', symbol='AAPL', period='2y')

# Analyze risk metrics
risk_manager = PortfolioRiskManager()
risk_report = risk_manager.generate_risk_report(results['returns'], results['positions'])
print(risk_report)
```

## Configuration

Create a `.env` file in the root directory with your API keys:

```
ALPHA_VANTAGE_API_KEY=your_key_here
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```
