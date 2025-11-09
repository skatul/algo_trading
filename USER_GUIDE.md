# Algorithmic Trading System - User Guide

A comprehensive Python-based algorithmic trading system with backtesting, strategy development, and performance analysis capabilities.

## 🚀 Quick Start

### 1. Setup

```bash
# Clone/navigate to project directory
cd /home/atul/projects/algo_trading

# Activate virtual environment (already set up)
source .venv/bin/activate

# Run demo to verify everything works
python demo.py
```

### 2. Basic Usage

```python
from src.main import TradingEngine

# Initialize the trading engine
engine = TradingEngine()

# Run a simple backtest
results = engine.backtest('moving_average', 'AAPL', short_window=20, long_window=50)

# Print results summary
engine.print_summary()
```

## 📊 Features

### ✅ Data Management

- **Yahoo Finance Integration**: Real-time and historical data
- **Alpha Vantage Support**: Alternative data source
- **Technical Indicators**: Automatic calculation of SMA, EMA, RSI, MACD, Bollinger Bands
- **Data Caching**: Efficient data storage and retrieval

### ✅ Trading Strategies

- **Moving Average Crossover**: Buy/sell on MA crossovers
- **RSI Strategy**: Overbought/oversold signals
- **Buy and Hold**: Benchmark strategy
- **Extensible Framework**: Easy to add custom strategies

### ✅ Backtesting Engine

- **Performance Metrics**: Return, Sharpe ratio, drawdown, win rate
- **Transaction Costs**: Configurable commission and slippage
- **Detailed Analytics**: Trade-by-trade analysis
- **Visualization**: Portfolio performance charts

### ✅ Configuration & Logging

- **Flexible Configuration**: JSON config files and environment variables
- **Comprehensive Logging**: File and console logging
- **Environment Management**: Secure API key handling

## 🛠️ System Architecture

```
algo_trading/
├── src/
│   ├── data/              # Data fetching and management
│   │   ├── __init__.py
│   │   └── data_fetcher.py
│   ├── strategies/        # Trading strategies
│   │   ├── __init__.py
│   │   ├── base_strategy.py
│   │   └── sample_strategies.py
│   ├── backtesting/       # Backtesting engine
│   │   ├── __init__.py
│   │   └── backtest_engine.py
│   ├── utils/             # Utilities and configuration
│   │   ├── __init__.py
│   │   └── config.py
│   ├── __init__.py
│   └── main.py           # Main trading engine
├── data/                 # Stored market data
├── logs/                 # System logs
├── tests/                # Unit tests
├── demo.py               # Demonstration script
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 📈 Usage Examples

### Example 1: Single Strategy Backtest

```python
from src.main import TradingEngine

engine = TradingEngine()

# Test Moving Average strategy on Apple stock
result = engine.backtest(
    strategy='moving_average',
    symbol='AAPL',
    short_window=20,
    long_window=50,
    start_date='2023-01-01',
    initial_capital=100000
)

# Display results
engine.print_summary()

# Export results
engine.export_results('results/aapl_ma_backtest.csv')

# Plot performance
engine.plot_results(save_path='plots/aapl_performance.png')
```

### Example 2: Strategy Comparison

```python
from src.main import TradingEngine

engine = TradingEngine()

strategies = [
    {'name': 'Buy & Hold', 'strategy': 'buy_and_hold'},
    {'name': 'MA 10/30', 'strategy': 'moving_average', 'parameters': {'short_window': 10, 'long_window': 30}},
    {'name': 'MA 20/50', 'strategy': 'moving_average', 'parameters': {'short_window': 20, 'long_window': 50}},
    {'name': 'RSI 14', 'strategy': 'rsi', 'parameters': {'rsi_period': 14}}
]

# Compare strategies on Apple
results = engine.compare_strategies_on_symbol(strategies, 'AAPL')
print(results)
```

### Example 3: Multi-Symbol Analysis

```python
from src.main import TradingEngine

engine = TradingEngine()
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

for symbol in symbols:
    print(f"\nAnalyzing {symbol}...")
    result = engine.backtest('moving_average', symbol)
    print(f"Total Return: {result['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
```

### Example 4: Custom Strategy Development

```python
from src.strategies.base_strategy import BaseStrategy
import pandas as pd

class CustomStrategy(BaseStrategy):
    def __init__(self, my_parameter=10, **kwargs):
        super().__init__("CustomStrategy", {'my_parameter': my_parameter, **kwargs})
        self.my_parameter = my_parameter

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Add your custom indicators here
        df['custom_indicator'] = df['close'].rolling(self.my_parameter).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['signal'] = 0
        df['signal_reason'] = ''

        # Add your signal logic here
        # Example: Buy when price > custom_indicator
        buy_signal = df['close'] > df['custom_indicator']
        df.loc[buy_signal, 'signal'] = 1
        df.loc[buy_signal, 'signal_reason'] = 'CUSTOM_BUY'

        return df

# Use your custom strategy
engine = TradingEngine()
custom_strategy = CustomStrategy(my_parameter=20)
result = engine.backtest(custom_strategy, 'AAPL')
```

## 🔧 Configuration

### Environment Variables (.env file)

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Parameters
INITIAL_CAPITAL=100000
COMMISSION_RATE=0.001

# System Configuration
LOG_LEVEL=INFO
DEFAULT_DATA_SOURCE=yahoo
```

### Configuration File (config.json)

```json
{
  "trading": {
    "initial_capital": 100000,
    "commission": 0.001,
    "slippage": 0.0005,
    "max_position_size": 0.1,
    "risk_free_rate": 0.02
  },
  "data": {
    "default_source": "yahoo",
    "cache_enabled": true,
    "cache_duration_hours": 24
  },
  "backtesting": {
    "benchmark": "SPY",
    "start_date": "2020-01-01",
    "plot_results": true,
    "save_results": true
  }
}
```

## 📊 Performance Metrics

The system calculates comprehensive performance metrics:

- **Total Return**: Overall portfolio return
- **Annualized Return**: Yearly return rate
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Total Trades**: Number of completed trades

## 🚨 Risk Management (Future Enhancement)

Current system includes basic risk controls:

- Position sizing based on available capital
- Commission costs in backtesting
- Maximum drawdown tracking

Future enhancements will include:

- Stop-loss orders
- Position sizing algorithms
- Portfolio-level risk controls
- Dynamic risk adjustment

## 🔮 Future Enhancements

### Planned Features

1. **Live Trading Integration**

   - Alpaca API integration
   - Paper trading mode
   - Real-time signal execution

2. **Advanced Strategies**

   - Machine learning models
   - Sentiment analysis
   - Options strategies

3. **Portfolio Management**

   - Multi-asset portfolios
   - Asset allocation models
   - Rebalancing algorithms

4. **Enhanced Analytics**
   - Monte Carlo simulation
   - Stress testing
   - Performance attribution

## 🐛 Troubleshooting

### Common Issues

1. **Module Import Errors**

   ```bash
   # Ensure you're in the right directory and virtual environment is activated
   cd /home/atul/projects/algo_trading
   source .venv/bin/activate
   ```

2. **Data Fetching Errors**

   ```python
   # Check internet connection and symbol validity
   # Try different time periods if data is sparse
   data = engine.fetch_data('AAPL', period='1y')  # Instead of '1d'
   ```

3. **Strategy Errors**
   ```python
   # Ensure sufficient data for strategy parameters
   # For MA strategy, ensure long_window < data length
   strategy = engine.create_strategy('moving_average', short_window=5, long_window=20)
   ```

### Getting Help

1. Check the logs in the `logs/` directory
2. Run the demo script to verify system functionality
3. Review the example usage patterns above
4. Check that all required packages are installed

## 📝 License & Disclaimer

This is an educational algorithmic trading system.

**Important Disclaimer**:

- This system is for educational and research purposes only
- Past performance does not guarantee future results
- Always test strategies thoroughly before using real money
- Consider transaction costs, slippage, and market conditions
- Consult with financial advisors before making investment decisions

## 🎯 Getting Started Checklist

- [ ] ✅ System installed and virtual environment activated
- [ ] ✅ Run `python demo.py` successfully
- [ ] ✅ Understand basic TradingEngine usage
- [ ] ✅ Try strategy comparison example
- [ ] ✅ Create your first custom strategy
- [ ] ✅ Export and analyze results
- [ ] ✅ Set up configuration for your needs

**Happy Trading! 🚀📈**
