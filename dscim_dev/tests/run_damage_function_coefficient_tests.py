#!/usr/bin/env python3
"""
Damage Function Coefficient Test Runner
=======================================

Comprehensive test runner for damage function coefficient fitting validation.

This module provides a unified interface for running all damage function
coefficient tests, including both implementation validation and consistency
testing against the original DSCIM. It orchestrates the complete test suite
and provides summary reporting of results.

Test Categories:
    - Implementation Tests: Validate the refactored fitting implementation
    - Consistency Tests: Compare results with original DSCIM output
    - Integration Tests: Verify complete workflow functionality

Features:
    - Automated test discovery and execution
    - Comprehensive error handling and reporting
    - Summary statistics and result aggregation
    - Professional logging without decorative elements
    - Configurable test execution options

Design Principles:
    - Centralized test orchestration
    - Clear separation of test categories
    - Comprehensive result reporting
    - Professional output formatting
    - Robust error handling and recovery

Usage:
    This runner can be used to execute the complete damage function test suite
    or individual test categories as needed for validation and debugging.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pytest_suite(test_file: str, description: str) -> bool:
    """Run a specific pytest suite and return success status."""
    logger.info(f"=== Running {description} ===")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            test_file, 
            '-v', 
            '--tb=short',
            '--color=yes'
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        if success:
            logger.info(f"PASSED: {description}")
        else:
            logger.error(f"FAILED: {description}")
        
        return success
        
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT: {description}")
        return False
    except Exception as e:
        logger.error(f"ERROR: {description} - {e}")
        return False


def run_all_damage_function_coefficient_tests():
    """Run all damage function coefficient tests."""
    logger.info("Starting Damage Function Coefficient Test Suite")
    
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    
    # Define test suites
    test_suites = [
        {
            'file': test_dir / 'test_damage_function_coefficient_fitting.py',
            'description': 'Damage Function Coefficient Fitting Workflow Tests'
        },
        {
            'file': test_dir / 'test_damage_function_coefficient_consistency.py', 
            'description': 'Damage Function Coefficient Consistency Tests'
        }
    ]
    
    # Track results
    results = {}
    
    # Run each test suite
    for suite in test_suites:
        test_file = suite['file']
        description = suite['description']
        
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            results[description] = False
            continue
        
        success = run_pytest_suite(str(test_file), description)
        results[description] = success
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUITE SUMMARY")
    logger.info("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for description, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{status}: {description}")
    
    logger.info("-"*60)
    logger.info(f"Total: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        logger.info("ALL DAMAGE FUNCTION COEFFICIENT TESTS PASSED!")
        return True
    else:
        logger.error(f"{total_tests - passed_tests} test suite(s) failed")
        return False


def run_quick_validation():
    """Run a quick validation to check if the basic setup works."""
    logger.info("Running quick validation...")
    
    try:
        # Test imports
        from dscim_dev.io.data_loaders import load_reduced_damages, load_dummy_climate_data
        from dscim_dev.processing.damage_workflows import generate_damage_function_points
        logger.info("Imports successful")
        
        # Test basic data loading
        reduced_damages = load_reduced_damages(
            config_path='configs/dummy_config.yaml',
            discounting_type='adding_up'
        )
        logger.info(f"Loaded reduced damages: {reduced_damages.shape}")
        
        climate_data = load_dummy_climate_data('configs/dummy_config.yaml')
        logger.info(f"Loaded climate data: {climate_data.dims}")
        
        # Test basic point generation
        points = generate_damage_function_points(
            reduced_damages={'adding_up': reduced_damages},
            climate_data=climate_data,
            formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            ssps_to_use=["ssp2", "ssp3", "ssp4"]
        )
        logger.info(f"Generated damage function points: {len(points)} points")
        
        if len(points) == 264:
            logger.info("Point count matches expected (264)")
        else:
            logger.warning(f"Point count mismatch: expected 264, got {len(points)}")
        
        logger.info("Quick validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return False


def main():
    """Main function to run all tests."""
    logger.info("Damage Function Coefficient Test Runner")
    logger.info("="*60)
    
    # Check if we're in the right directory
    if not Path('configs/dummy_config.yaml').exists():
        logger.error("Not in the correct directory. Please run from the project root.")
        return False
    
    # Run quick validation first
    if not run_quick_validation():
        logger.error("Quick validation failed. Aborting test run.")
        return False
    
    # Run all test suites
    success = run_all_damage_function_coefficient_tests()
    
    if success:
        logger.info("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("The damage function coefficient fitting implementation is working correctly.")
    else:
        logger.error("\n💥 SOME TESTS FAILED!")
        logger.error("Please check the test output above for details.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 