"""
Unit tests for the safe_float helper function.
Tests various edge cases including empty Series, numpy arrays, NaN, inf, complex numbers.
"""

import unittest
import pandas as pd
import numpy as np
import math
from src.backtesting.backtest_engine import BacktestEngine


class TestSafeFloat(unittest.TestCase):
    """Test cases for the safe_float helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine(initial_capital=100000, commission=0.001)
        
    def _test_safe_float_via_method(self, test_value):
        """Test safe_float by patching a metric value and seeing if it handles it."""
        # Create minimal portfolio data
        portfolio_data = pd.DataFrame({
            'portfolio_value': [100000, 101000, 102000, 103000, 104000],
            'returns': [0.0, 0.01, 0.01, 0.01, 0.01]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        # Patch quantstats to return our test value
        import quantstats as qs
        original_stats_func = qs.stats.sharpe
        
        def mock_stats(*args, **kwargs):
            return test_value
        
        try:
            qs.stats.sharpe = mock_stats
            result = self.engine._calculate_performance_metrics(portfolio_data)
            return result['sharpe_ratio']
        finally:
            qs.stats.sharpe = original_stats_func

    def test_none_value(self):
        """Test that None returns default value."""
        result = self._test_safe_float_via_method(None)
        self.assertEqual(result, 0.0)

    def test_empty_pandas_series(self):
        """Test that empty pandas Series returns default value."""
        empty_series = pd.Series(dtype=float)
        result = self._test_safe_float_via_method(empty_series)
        self.assertEqual(result, 0.0)

    def test_pandas_series_with_values(self):
        """Test that pandas Series with values extracts first value."""
        series = pd.Series([1.5, 2.5, 3.5])
        result = self._test_safe_float_via_method(series)
        self.assertEqual(result, 1.5)

    def test_numpy_array(self):
        """Test that numpy arrays are handled safely."""
        array = np.array([2.5, 3.5, 4.5])
        result = self._test_safe_float_via_method(array)
        # Should either extract first value or return default safely
        self.assertIsInstance(result, float)
        
    def test_empty_numpy_array(self):
        """Test that empty numpy arrays return default."""
        empty_array = np.array([])
        result = self._test_safe_float_via_method(empty_array)
        self.assertEqual(result, 0.0)

    def test_nan_values(self):
        """Test that NaN values return default."""
        result = self._test_safe_float_via_method(float('nan'))
        self.assertEqual(result, 0.0)

    def test_infinity_values(self):
        """Test that infinity values return default."""
        result = self._test_safe_float_via_method(float('inf'))
        self.assertEqual(result, 0.0)
        
        result = self._test_safe_float_via_method(float('-inf'))
        self.assertEqual(result, 0.0)

    def test_complex_numbers(self):
        """Test that complex numbers return real part."""
        result = self._test_safe_float_via_method(3.5 + 2.0j)
        self.assertEqual(result, 3.5)

    def test_regular_numbers(self):
        """Test that regular int/float values work correctly."""
        result = self._test_safe_float_via_method(42)
        self.assertEqual(result, 42.0)
        
        result = self._test_safe_float_via_method(3.14159)
        self.assertEqual(result, 3.14159)

    def test_string_numbers(self):
        """Test that string representations of numbers are converted."""
        result = self._test_safe_float_via_method("123.45")
        self.assertEqual(result, 123.45)

    def test_invalid_strings(self):
        """Test that invalid strings return default."""
        result = self._test_safe_float_via_method("invalid")
        self.assertEqual(result, 0.0)

    def test_pandas_series_with_nan(self):
        """Test pandas Series containing NaN values."""
        series = pd.Series([float('nan'), 2.5, 3.5])
        result = self._test_safe_float_via_method(series)
        self.assertEqual(result, 0.0)  # NaN should return default

    def test_pandas_series_with_inf(self):
        """Test pandas Series containing infinity values."""
        series = pd.Series([float('inf'), 2.5, 3.5])
        result = self._test_safe_float_via_method(series)
        self.assertEqual(result, 0.0)  # Inf should return default


class TestSafeFloatDirect(unittest.TestCase):
    """Direct tests using a standalone safe_float implementation."""
    
    def safe_float(self, value) -> float:
        """
        Standalone implementation of safe_float for testing.
        Copy of the robust implementation.
        """
        import math
        
        # Default for invalid values
        default = 0.0
        
        if value is None:
            return default
            
        # Handle pandas Series/Index or other array-like with iloc
        if hasattr(value, 'iloc'):
            try:
                if len(value) == 0:
                    return default
                scalar = value.iloc[0]
                return self.safe_float(scalar)  # Recursive call
            except Exception:
                return default
                
        # Handle numeric scalars explicitly
        if isinstance(value, (int, float)):
            try:
                f = float(value)
                if math.isnan(f) or math.isinf(f):
                    return default
                return f
            except Exception:
                return default
        
        # Handle complex numbers
        if isinstance(value, complex):
            try:
                f = float(value.real)
                if math.isnan(f) or math.isinf(f):
                    return default
                return f
            except Exception:
                return default
                
        # Fallback: try direct conversion and validate
        try:
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return default
            return f
        except Exception:
            return default

    def test_all_edge_cases(self):
        """Test comprehensive edge cases."""
        
        # Test None
        self.assertEqual(self.safe_float(None), 0.0)
        
        # Test empty pandas Series
        empty_series = pd.Series(dtype=float)
        self.assertEqual(self.safe_float(empty_series), 0.0)
        
        # Test pandas Series with valid values
        series = pd.Series([1.5, 2.5, 3.5])
        self.assertEqual(self.safe_float(series), 1.5)
        
        # Test pandas Series with NaN first
        nan_series = pd.Series([float('nan'), 2.5])
        self.assertEqual(self.safe_float(nan_series), 0.0)
        
        # Test numpy arrays (should fall through to fallback)
        array = np.array([2.5])
        result = self.safe_float(array)
        self.assertIsInstance(result, float)
        
        # Test NaN and Inf
        self.assertEqual(self.safe_float(float('nan')), 0.0)
        self.assertEqual(self.safe_float(float('inf')), 0.0)
        self.assertEqual(self.safe_float(float('-inf')), 0.0)
        
        # Test complex numbers
        self.assertEqual(self.safe_float(3.5 + 2j), 3.5)
        self.assertEqual(self.safe_float(complex('nan+0j')), 0.0)
        
        # Test regular numbers
        self.assertEqual(self.safe_float(42), 42.0)
        self.assertEqual(self.safe_float(3.14), 3.14)
        
        # Test string conversions
        self.assertEqual(self.safe_float("123.45"), 123.45)
        self.assertEqual(self.safe_float("invalid"), 0.0)


if __name__ == '__main__':
    unittest.main()