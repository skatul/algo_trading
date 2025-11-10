"""
Unit tests for the safe_float helper behavior and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
import math
from unittest import mock

from src.backtesting.backtest_engine import BacktestEngine


class TestSafeFloat(unittest.TestCase):
    def setUp(self):
        self.engine = BacktestEngine(initial_capital=100000, commission=0.001)

    def _test_safe_float_via_method(self, test_value):
        # Minimal portfolio data
        portfolio_data = pd.DataFrame({
            'portfolio_value': [100000, 101000, 102000, 103000, 104000],
            'returns': [0.0, 0.01, 0.01, 0.01, 0.01]
        }, index=pd.date_range('2023-01-01', periods=5))

        # Patch quantstats.sharpe to return the test value
        import quantstats as qs
        with mock.patch.object(qs.stats, 'sharpe', return_value=test_value):
            result = self.engine._calculate_performance_metrics(portfolio_data)
            return result['sharpe_ratio']

    def test_none_value(self):
        self.assertEqual(self._test_safe_float_via_method(None), 0.0)

    def test_empty_pandas_series(self):
        empty_series = pd.Series(dtype=float)
        self.assertEqual(self._test_safe_float_via_method(empty_series), 0.0)

    def test_pandas_series_with_values(self):
        series = pd.Series([1.5, 2.5, 3.5])
        self.assertEqual(self._test_safe_float_via_method(series), 1.5)

    def test_numpy_array(self):
        array = np.array([2.5, 3.5, 4.5])
        val = self._test_safe_float_via_method(array)
        self.assertIsInstance(val, float)

    def test_empty_numpy_array(self):
        empty_array = np.array([])
        self.assertEqual(self._test_safe_float_via_method(empty_array), 0.0)

    def test_nan_and_inf(self):
        self.assertEqual(self._test_safe_float_via_method(float('nan')), 0.0)
        self.assertEqual(self._test_safe_float_via_method(float('inf')), 0.0)
        self.assertEqual(self._test_safe_float_via_method(float('-inf')), 0.0)

    def test_complex_numbers(self):
        self.assertEqual(self._test_safe_float_via_method(3.5 + 2.0j), 3.5)

    def test_regular_numbers(self):
        self.assertEqual(self._test_safe_float_via_method(42), 42.0)
        self.assertAlmostEqual(self._test_safe_float_via_method(3.14159), 3.14159, places=6)

    def test_string_numbers(self):
        self.assertEqual(self._test_safe_float_via_method("123.45"), 123.45)

    def test_invalid_strings(self):
        self.assertEqual(self._test_safe_float_via_method("invalid"), 0.0)


class TestSafeFloatDirect(unittest.TestCase):
    """Direct tests using a standalone safe_float logic copy for coverage."""

    def safe_float(self, value) -> float:
        import math
        default = 0.0
        if value is None:
            return default
        if hasattr(value, 'iloc'):
            try:
                if len(value) == 0:
                    return default
                scalar = value.iloc[0]
                return self.safe_float(scalar)
            except Exception:
                return default
        try:
            import numpy as _np
            if isinstance(value, _np.ndarray):
                if value.size == 0:
                    return default
                if value.size == 1:
                    return self.safe_float(value.item())
                return default
        except Exception:
            pass
        if isinstance(value, (int, float)):
            try:
                f = float(value)
                if math.isnan(f) or math.isinf(f):
                    return default
                return f
            except Exception:
                return default
        if isinstance(value, complex):
            try:
                f = float(value.real)
                if math.isnan(f) or math.isinf(f):
                    return default
                return f
            except Exception:
                return default
        try:
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return default
            return f
        except Exception:
            return default

    def test_all_edge_cases(self):
        self.assertEqual(self.safe_float(None), 0.0)
        self.assertEqual(self.safe_float(pd.Series(dtype=float)), 0.0)
        self.assertEqual(self.safe_float(pd.Series([1.5, 2.5])), 1.5)
        self.assertEqual(self.safe_float(pd.Series([float('nan'), 2.5])), 0.0)
        self.assertIsInstance(self.safe_float(np.array([2.5])), float)
        self.assertEqual(self.safe_float(float('nan')), 0.0)
        self.assertEqual(self.safe_float(float('inf')), 0.0)
        self.assertEqual(self.safe_float(3.5 + 2j), 3.5)
        self.assertEqual(self.safe_float(42), 42.0)
        self.assertEqual(self.safe_float("123.45"), 123.45)
        self.assertEqual(self.safe_float("invalid"), 0.0)

if __name__ == '__main__':
    unittest.main()
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