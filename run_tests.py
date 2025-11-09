#!/usr/bin/env python3
"""
Test runner for the algorithmic trading system.
Runs all unit tests and generates coverage reports.
"""

import unittest
import sys
import os
from io import StringIO

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def run_tests():
    """Run all tests and return results."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")

    # Create test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, buffer=True)

    # Run tests
    result = runner.run(suite)

    # Print results
    print("=" * 70)
    print("ALGORITHMIC TRADING SYSTEM - TEST RESULTS")
    print("=" * 70)
    print(stream.getvalue())

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    # Print failures and errors if any
    if result.failures:
        print("\nFAILURES:")
        print("-" * 50)
        for test, traceback in result.failures:
            print(f"FAIL: {test}")
            print(traceback)
            print("-" * 50)

    if result.errors:
        print("\nERRORS:")
        print("-" * 50)
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
            print(traceback)
            print("-" * 50)

    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0

    if success:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")

    return success


def run_integration_tests():
    """Run integration tests (requiring internet connection)."""
    print("\n" + "=" * 70)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 70)
    print("Note: Integration tests require internet connection and may take longer.")

    # Set environment to run integration tests
    os.environ.pop("SKIP_INTEGRATION_TESTS", None)

    # Run only integration test methods
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add integration test classes
    from tests.test_data_fetcher import TestDataFetcherIntegration
    from tests.test_main import TestTradingEngineIntegration

    suite.addTest(loader.loadTestsFromTestCase(TestDataFetcherIntegration))
    suite.addTest(loader.loadTestsFromTestCase(TestTradingEngineIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """Main test runner function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run tests for the algorithmic trading system"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests (requires internet)",
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only unit tests (skip integration)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report (requires coverage package)",
    )

    args = parser.parse_args()

    # Skip integration tests by default
    if args.unit_only:
        os.environ["SKIP_INTEGRATION_TESTS"] = "true"

    success = True

    # Run unit tests
    if args.coverage:
        try:
            import coverage

            cov = coverage.Coverage()
            cov.start()

            success = run_tests()

            cov.stop()
            cov.save()

            print("\n" + "=" * 70)
            print("COVERAGE REPORT")
            print("=" * 70)
            cov.report()

            # Generate HTML report
            try:
                cov.html_report(directory="htmlcov")
                print("\nHTML coverage report generated in 'htmlcov' directory")
            except Exception as e:
                print(f"Could not generate HTML report: {e}")

        except ImportError:
            print("Coverage package not installed. Running tests without coverage.")
            success = run_tests()
    else:
        success = run_tests()

    # Run integration tests if requested
    if args.integration:
        integration_success = run_integration_tests()
        success = success and integration_success

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
