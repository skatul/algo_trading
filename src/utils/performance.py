"""
Performance benchmarking and profiling utilities for the trading system.

This module provides comprehensive performance analysis tools including
execution time profiling, memory usage tracking, and system metrics.
"""

import time
import functools
import logging
import psutil
import tracemalloc
from typing import Dict, Any, Callable, Optional
from datetime import datetime
from functools import lru_cache


class PerformanceProfiler:
    """Advanced performance profiler for trading system operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.active_profiles = {}
        
    def profile_execution(self, operation_name: Optional[str] = None):
        """
        Decorator for profiling function execution time and memory usage.
        
        Args:
            operation_name: Custom name for the operation
            
        Returns:
            Decorated function with performance profiling
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # Start profiling
                start_time = time.perf_counter()
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                success = False
                error = None
                result = None
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    # Record metrics
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = end_memory - start_memory
                    
                    # Store metrics
                    self._record_metrics(name, {
                        'execution_time': execution_time,
                        'memory_delta': memory_delta,
                        'peak_memory_trace': peak / 1024 / 1024,  # MB
                        'success': success,
                        'error': error,
                        'timestamp': datetime.now()
                    })
                    
                    self.logger.debug(
                        f"Performance: {name} - {execution_time:.4f}s, "
                        f"Memory: {memory_delta:+.2f}MB, Success: {success}"
                    )
                
                return result
            return wrapper
        return decorator
    
    def start_operation_profile(self, operation_name: str):
        """Start profiling a long-running operation."""
        self.active_profiles[operation_name] = {
            'start_time': time.perf_counter(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024
        }
        tracemalloc.start()
    
    def end_operation_profile(self, operation_name: str) -> Dict[str, Any]:
        """End profiling and return metrics for a long-running operation."""
        if operation_name not in self.active_profiles:
            raise ValueError(f"No active profile found for: {operation_name}")
        
        profile = self.active_profiles.pop(operation_name)
        end_time = time.perf_counter()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        metrics = {
            'execution_time': end_time - profile['start_time'],
            'memory_delta': end_memory - profile['start_memory'],
            'peak_memory_trace': peak / 1024 / 1024,
            'timestamp': datetime.now()
        }
        
        self._record_metrics(operation_name, metrics)
        return metrics
    
    def _record_metrics(self, name: str, metrics: Dict[str, Any]):
        """Record performance metrics internally."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metrics)
        
        # Keep only last 100 records per operation
        if len(self.metrics[name]) > 100:
            self.metrics[name] = self.metrics[name][-100:]
    
    def get_performance_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance summary for specific operation or all operations.
        
        Args:
            operation_name: Specific operation to analyze, None for all
            
        Returns:
            Dictionary with performance statistics
        """
        if operation_name:
            if operation_name not in self.metrics:
                return {}
            data = self.metrics[operation_name]
        else:
            # Aggregate all operations
            data = []
            for records in self.metrics.values():
                data.extend(records)
        
        if not data:
            return {}
        
        # Calculate statistics
        successful_ops = [m for m in data if m.get('success', True)]
        execution_times = [m['execution_time'] for m in successful_ops]
        memory_deltas = [m['memory_delta'] for m in successful_ops]
        
        if not execution_times:
            return {'total_operations': len(data), 'successful_operations': 0}
        
        summary = {
            'total_operations': len(data),
            'successful_operations': len(successful_ops),
            'success_rate': len(successful_ops) / len(data) * 100,
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
            'total_memory_used': sum(m for m in memory_deltas if m > 0),
            'operations_per_second': len(successful_ops) / sum(execution_times) if execution_times else 0
        }
        
        return summary
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        report = ["=" * 60, "PERFORMANCE ANALYSIS REPORT", "=" * 60]
        
        overall_summary = self.get_performance_summary()
        if not overall_summary:
            return "No performance data available."
        
        report.extend([
            f"Total Operations: {overall_summary['total_operations']}",
            f"Success Rate: {overall_summary['success_rate']:.2f}%",
            f"Average Execution Time: {overall_summary['avg_execution_time']:.4f}s",
            f"Operations per Second: {overall_summary['operations_per_second']:.2f}",
            f"Average Memory Impact: {overall_summary['avg_memory_delta']:+.2f}MB",
            ""
        ])
        
        # Individual operation analysis
        report.append("OPERATION BREAKDOWN:")
        report.append("-" * 40)
        
        for operation_name in sorted(self.metrics.keys()):
            op_summary = self.get_performance_summary(operation_name)
            if op_summary:
                report.extend([
                    f"Operation: {operation_name}",
                    f"  Calls: {op_summary['total_operations']}",
                    f"  Success Rate: {op_summary['success_rate']:.1f}%",
                    f"  Avg Time: {op_summary['avg_execution_time']:.4f}s",
                    f"  Memory: {op_summary['avg_memory_delta']:+.2f}MB",
                    ""
                ])
        
        return "\n".join(report)


# Global profiler instance
profiler = PerformanceProfiler()


def benchmark_backtest(func: Callable) -> Callable:
    """Specific decorator for benchmarking backtest operations."""
    return profiler.profile_execution("backtest_operation")(func)


def benchmark_data_fetch(func: Callable) -> Callable:
    """Specific decorator for benchmarking data fetching operations."""
    return profiler.profile_execution("data_fetch_operation")(func)


def benchmark_strategy_execution(func: Callable) -> Callable:
    """Specific decorator for benchmarking strategy execution."""
    return profiler.profile_execution("strategy_execution")(func)


class SystemMonitor:
    """System resource monitoring for trading operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics."""
        process = psutil.Process()
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available / 1024 / 1024,  # MB
            'process_memory': process.memory_info().rss / 1024 / 1024,  # MB
            'process_cpu': process.cpu_percent(),
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
            'timestamp': datetime.now()
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check system health and return warnings if needed."""
        metrics = self.get_system_metrics()
        health_status = {'status': 'healthy', 'warnings': [], 'metrics': metrics}
        
        # Check for resource constraints
        if metrics['cpu_percent'] > 90:
            health_status['warnings'].append("High CPU usage detected")
            health_status['status'] = 'warning'
            
        if metrics['memory_percent'] > 90:
            health_status['warnings'].append("High memory usage detected")
            health_status['status'] = 'warning'
            
        if metrics['disk_usage'] > 90:
            health_status['warnings'].append("High disk usage detected")
            health_status['status'] = 'warning'
            
        if metrics['process_memory'] > 1000:  # 1GB
            health_status['warnings'].append("Process using excessive memory")
            health_status['status'] = 'warning'
        
        return health_status


# Global system monitor instance
system_monitor = SystemMonitor()