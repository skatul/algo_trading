# Algorithmic Trading System

A comprehensive Python-based algorithmic trading system with backtesting, risk management, and portfolio optimization capabilities.

## Features

- **Data Management**: Real-time and historical data fetching from multiple sources
- **Strategy Framework**: Modular strategy implementation with common technical indicators
- **Backtesting Engine**: Comprehensive backtesting with performance metrics
- **Risk Management**: Position sizing, stop-loss, and portfolio-level risk controls
- **Portfolio Management**: Position tracking, P&L calculation, and capital allocation
- **Live Trading**: Paper trading and live execution capabilities

## Project Structure

```
algo_trading/
├── src/
│   ├── data/              # Data fetching and management
│   ├── strategies/        # Trading strategies
│   ├── backtesting/       # Backtesting engine
│   ├── risk_management/   # Risk control modules
│   ├── portfolio/         # Portfolio management
│   └── utils/             # Utility functions
├── data/                  # Stored market data
├── logs/                  # System logs
└── tests/                 # Unit tests
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

```python
from src.main import TradingEngine

# Initialize trading engine
engine = TradingEngine()

# Run backtest
results = engine.backtest(strategy='moving_average', symbol='AAPL', start='2023-01-01')

# View results
engine.plot_results(results)
```

## Configuration

Create a `.env` file in the root directory with your API keys:

```
ALPHA_VANTAGE_API_KEY=your_key_here
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```
