"""
Data fetching module for algorithmic trading system.
Handles real-time and historical market data from multiple sources.
"""

import pandas as pd
import yfinance as yf
import requests
import os
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataFetcher:
    """
    A comprehensive data fetcher class that retrieves market data from various sources.
    Supports Yahoo Finance, Alpha Vantage, and other data providers.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.data_cache = {}
        
    def get_yahoo_data(self, 
                      symbol: str, 
                      period: str = "1y", 
                      interval: str = "1d",
                      start: Optional[str] = None,
                      end: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if start and end:
                data = ticker.history(start=start, end=end, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning("No data retrieved for %s", symbol)
                return pd.DataFrame()
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data.index.name = 'date'
            
            # Add symbol column
            data['symbol'] = symbol
            
            self.logger.info("Successfully fetched %d records for %s", len(data), symbol)
            return data
            
        except (ValueError, KeyError) as e:
            self.logger.error("Error fetching Yahoo data for %s: %s", symbol, str(e))
            return pd.DataFrame()
        except Exception as e:
            self.logger.error("Unexpected error fetching Yahoo data for %s: %s", symbol, str(e))
            return pd.DataFrame()
    
    def get_multiple_symbols(self, 
                           symbols: List[str], 
                           period: str = "1y",
                           interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            period: Time period
            interval: Data interval
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data_dict = {}
        
        for symbol in symbols:
            self.logger.info("Fetching data for %s", symbol)
            data = self.get_yahoo_data(symbol, period, interval)
            if not data.empty:
                data_dict[symbol] = data
        
        return data_dict
    
    def get_alpha_vantage_data(self, 
                              symbol: str, 
                              function: str = "TIME_SERIES_DAILY",
                              outputsize: str = "compact") -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage API.
        
        Args:
            symbol: Stock symbol
            function: API function (TIME_SERIES_DAILY, TIME_SERIES_INTRADAY, etc.)
            outputsize: 'compact' (100 data points) or 'full' (up to 20 years)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpha_vantage_key:
            self.logger.error("Alpha Vantage API key not found in environment variables")
            return pd.DataFrame()
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.alpha_vantage_key,
                "outputsize": outputsize
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Handle different response formats
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
            elif "Time Series (5min)" in data:
                time_series = data["Time Series (5min)"]
            elif "Time Series (1min)" in data:
                time_series = data["Time Series (1min)"]
            else:
                self.logger.error("Unexpected Alpha Vantage response format for %s", symbol)
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            # Clean column names
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            df['symbol'] = symbol
            
            self.logger.info("Successfully fetched %d records for %s from Alpha Vantage", len(df), symbol)
            return df
            
        except (ValueError, KeyError, requests.RequestException) as e:
            self.logger.error("Error fetching Alpha Vantage data for %s: %s", symbol, str(e))
            return pd.DataFrame()
        except Exception as e:
            self.logger.error("Unexpected error fetching Alpha Vantage data for %s: %s", symbol, str(e))
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get comprehensive stock information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            stock_info = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'current_price': info.get('currentPrice', 0)
            }
            
            return stock_info
            
        except (ValueError, KeyError) as e:
            self.logger.error("Error fetching stock info for %s: %s", symbol, str(e))
            return {}
        except Exception as e:
            self.logger.error("Unexpected error fetching stock info for %s: %s", symbol, str(e))
            return {}
    
    def calculate_returns(self, data: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Calculate various types of returns.
        
        Args:
            data: DataFrame with price data
            column: Column to calculate returns on
            
        Returns:
            DataFrame with return calculations
        """
        df = data.copy()
        
        # Simple returns
        df['returns'] = df[column].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df[column] / df[column].shift(1))
        
        # Cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Average
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        delta = pd.to_numeric(delta, errors='coerce')
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def save_data(self, data: pd.DataFrame, filename: str, file_format: str = 'csv'):
        """
        Save data to file.
        
        Args:
            data: DataFrame to save
            filename: Filename (without extension)
            file_format: File format ('csv', 'parquet', 'pickle')
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        filepath = os.path.join(data_dir, f"{filename}.{file_format}")
        
        try:
            if file_format == 'csv':
                data.to_csv(filepath)
            elif file_format == 'parquet':
                data.to_parquet(filepath)
            elif file_format == 'pickle':
                data.to_pickle(filepath)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            self.logger.info("Data saved to %s", filepath)
            
        except (IOError, ValueError) as e:
            self.logger.error("Error saving data to %s: %s", filepath, str(e))
        except Exception as e:
            self.logger.error("Unexpected error saving data to %s: %s", filepath, str(e))
    
    def load_data(self, filename: str, file_format: str = 'csv') -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            filename: Filename (without extension)
            file_format: File format ('csv', 'parquet', 'pickle')
            
        Returns:
            DataFrame with loaded data
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        filepath = os.path.join(data_dir, f"{filename}.{file_format}")
        
        try:
            if file_format == 'csv':
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif file_format == 'parquet':
                data = pd.read_parquet(filepath)
            elif file_format == 'pickle':
                data = pd.read_pickle(filepath)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            self.logger.info("Data loaded from %s", filepath)
            return data
            
        except (IOError, ValueError, FileNotFoundError) as e:
            self.logger.error("Error loading data from %s: %s", filepath, str(e))
            return pd.DataFrame()
        except Exception as e:
            self.logger.error("Unexpected error loading data from %s: %s", filepath, str(e))
            return pd.DataFrame()

# Import numpy for log returns calculation
import numpy as np

if __name__ == "__main__":
    # Example usage
    fetcher = DataFetcher()
    
    # Fetch Apple stock data
    aapl_data = fetcher.get_yahoo_data('AAPL', period='1y')
    
    if not aapl_data.empty:
        # Add technical indicators
        aapl_data = fetcher.add_technical_indicators(aapl_data)
        
        # Calculate returns
        aapl_data = fetcher.calculate_returns(aapl_data)
        
        # Save data
        fetcher.save_data(aapl_data, 'AAPL_1y_with_indicators')
        
        print("Sample data:")
        print(aapl_data.head())
        print(f"\nData shape: {aapl_data.shape}")
        print(f"Columns: {list(aapl_data.columns)}")