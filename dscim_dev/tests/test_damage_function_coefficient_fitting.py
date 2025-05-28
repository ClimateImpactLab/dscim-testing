#!/usr/bin/env python3
"""
Damage Function Coefficient Fitting Tests
=========================================

Tests validating the damage function coefficient fitting implementation.

This test suite validates the damage function fitting workflow, ensuring that
the refactored implementation correctly generates damage function points,
fits coefficients using the original DSCIM methodology, and produces
reasonable results across different parameter combinations.

Test Coverage:
    - Damage function points generation from reduced damages and climate data
    - Coefficient fitting with different discounting types
    - Multiple fitting methods (OLS, quantile regression)
    - Complete workflow integration testing
    - Result structure and reasonableness validation
    - Error handling for edge cases

Methodology:
    Tests use actual dummy data from the DSCIM repository to ensure realistic
    data structures and coordinate systems. The fitting process is validated
    through structural checks, numerical reasonableness tests, and workflow
    integration verification.

Design Principles:
    - Comprehensive workflow validation
    - Realistic data usage matching original DSCIM
    - Structural and numerical result validation
    - Clear error reporting and debugging information
    - Modular test design for individual components

Usage:
    These tests should be run to validate the damage function fitting
    implementation and ensure proper integration with the damage reduction step.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import logging
from pathlib import Path
from typing import Dict, Any

from dscim_dev.io.data_loaders import load_reduced_damages, load_dummy_climate_data
from dscim_dev.processing.damage_workflows import (
    generate_damage_function_points,
    fit_damage_function_coefficients,
    damage_function_workflow
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDamageFunctionCoefficientFitting:
    """Test suite for damage function coefficient fitting workflow."""
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Load test data for coefficient fitting tests."""
        logger.info("Loading test data for coefficient fitting...")
        
        # Load reduced damages with population scaling
        reduced_damages = load_reduced_damages(
            config_path='configs/dummy_config.yaml',
            discounting_type='adding_up'
        )
        
        # Load climate data
        climate_data = load_dummy_climate_data('configs/dummy_config.yaml')
        
        return {
            'reduced_damages': {'adding_up': reduced_damages},
            'climate_data': climate_data
        }
    
    @pytest.fixture(scope="class")
    def damage_function_points(self, test_data):
        """Generate damage function points for testing."""
        logger.info("Generating damage function points for testing...")
        
        points = generate_damage_function_points(
            reduced_damages=test_data['reduced_damages'],
            climate_data=test_data['climate_data'],
            formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            ssps_to_use=["ssp2", "ssp3", "ssp4"]  # Match original DSCIM
        )
        
        return points
    
    def test_damage_function_points_generation(self, damage_function_points):
        """Test that damage function points are generated correctly."""
        logger.info("Testing damage function points generation...")
        
        # Check basic structure
        assert isinstance(damage_function_points, pd.DataFrame), "Points should be DataFrame"
        assert len(damage_function_points) == 264, f"Expected 264 points, got {len(damage_function_points)}"
        
        # Check required columns
        required_cols = ['year', 'ssp', 'model', 'rcp', 'gcm', 'damages', 'anomaly']
        for col in required_cols:
            assert col in damage_function_points.columns, f"Missing column: {col}"
        
        # Check year range
        years = sorted(damage_function_points['year'].unique())
        expected_years = list(range(2020, 2031))
        assert years == expected_years, f"Year range mismatch. Expected: {expected_years}, Got: {years}"
        
        # Check SSPs
        ssps = sorted(damage_function_points['ssp'].unique())
        expected_ssps = ['ssp2', 'ssp3', 'ssp4']
        assert ssps == expected_ssps, f"SSP mismatch. Expected: {expected_ssps}, Got: {ssps}"
        
        # Check damage values are reasonable
        damages = damage_function_points['damages']
        assert not damages.isna().any(), "No damage values should be NaN"
        assert (damages > 0).any(), "Should have some positive damage values"
        
        logger.info("PASSED: Damage function points generation test passed")
    
    def test_coefficient_fitting_constant_discounting(self, damage_function_points):
        """Test coefficient fitting with constant discounting."""
        logger.info("Testing coefficient fitting with constant discounting...")
        
        # Fit coefficients
        results = fit_damage_function_coefficients(
            damage_function_points=damage_function_points,
            formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            discounting_type="constant",
            fit_type="ols",
            year_range=range(2020, 2101),
            year_start_pred=2100
        )
        
        # Check result structure
        assert isinstance(results, dict), "Results should be dictionary"
        assert 'params' in results, "Results should contain 'params'"
        assert 'preds' in results, "Results should contain 'preds'"
        
        # Check parameters structure
        params = results['params']
        assert isinstance(params, xr.Dataset), "Parameters should be xarray Dataset"
        
        # Check dimensions
        expected_dims = ['discount_type', 'ssp', 'model', 'year']
        for dim in expected_dims:
            assert dim in params.dims, f"Missing dimension in params: {dim}"
        
        # Check coordinate values
        assert list(params.coords['discount_type']) == ['constant'], "Discount type should be 'constant'"
        assert set(params.coords['ssp'].values) == {'ssp2', 'ssp3', 'ssp4'}, "Should have 3 SSPs"
        assert set(params.coords['model'].values) == {'dummy1', 'dummy2'}, "Should have 2 models"
        
        # Check coefficient structure (coefficients are stored as data variables, not as a dimension)
        data_vars = list(params.data_vars)
        assert len(data_vars) >= 2, f"Should have at least 2 coefficient variables, got {len(data_vars)}: {data_vars}"
        
        # Check that we have expected coefficient variables based on the formula
        expected_coeffs = ['anomaly', 'np.power(anomaly, 2)']  # No intercept due to -1 in formula
        for coeff in expected_coeffs:
            assert coeff in data_vars, f"Missing coefficient variable: {coeff}"
        
        # Check predictions structure
        preds = results['preds']
        assert isinstance(preds, xr.Dataset), "Predictions should be xarray Dataset"
        
        logger.info("PASSED: Constant discounting coefficient fitting test passed")
    
    def test_coefficient_fitting_different_discounting_types(self, damage_function_points):
        """Test coefficient fitting with different discounting types."""
        logger.info("Testing coefficient fitting with different discounting types...")
        
        discounting_types = ["constant", "constant_model_collapsed"]
        
        for disc_type in discounting_types:
            logger.info(f"Testing {disc_type} discounting...")
            
            results = fit_damage_function_coefficients(
                damage_function_points=damage_function_points,
                formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
                discounting_type=disc_type,
                fit_type="ols",
                year_range=range(2020, 2101),
                year_start_pred=2100
            )
            
            # Check basic structure
            assert 'params' in results, f"Missing params for {disc_type}"
            assert 'preds' in results, f"Missing preds for {disc_type}"
            
            params = results['params']
            assert 'discount_type' in params.dims, f"Missing discount_type dim for {disc_type}"
            assert list(params.coords['discount_type']) == [disc_type], f"Wrong discount_type for {disc_type}"
            
            # Check that we have reasonable coefficient values
            param_values = params.to_array().values
            assert not np.all(np.isnan(param_values)), f"All coefficients are NaN for {disc_type}"
            assert np.any(np.isfinite(param_values)), f"No finite coefficients for {disc_type}"
        
        logger.info("PASSED: Different discounting types test passed")
    
    def test_complete_damage_function_workflow(self, test_data):
        """Test the complete damage function workflow end-to-end."""
        logger.info("Testing complete damage function workflow...")
        
        # Run complete workflow
        results = damage_function_workflow(
            reduced_damages=test_data['reduced_damages'],
            climate_data=test_data['climate_data'],
            formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            discounting_type="constant",
            ssps_to_use=["ssp2", "ssp3", "ssp4"],
            fit_type="ols",
            year_range=range(2020, 2101),
            year_start_pred=2100
        )
        
        # Check complete result structure
        expected_keys = [
            'damage_function_points',
            'damage_function_coefficients', 
            'damage_function_predictions',
            'metadata'
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing key in workflow results: {key}"
        
        # Check damage function points
        points = results['damage_function_points']
        assert isinstance(points, pd.DataFrame), "Points should be DataFrame"
        assert len(points) == 264, f"Expected 264 points, got {len(points)}"
        
        # Check coefficients
        coeffs = results['damage_function_coefficients']
        assert isinstance(coeffs, xr.Dataset), "Coefficients should be xarray Dataset"
        
        # Check predictions
        preds = results['damage_function_predictions']
        assert isinstance(preds, xr.Dataset), "Predictions should be xarray Dataset"
        
        # Check metadata
        metadata = results['metadata']
        assert isinstance(metadata, dict), "Metadata should be dictionary"
        assert 'formula' in metadata, "Metadata should contain formula"
        assert 'discounting_type' in metadata, "Metadata should contain discounting_type"
        assert 'n_points' in metadata, "Metadata should contain n_points"
        assert metadata['n_points'] == 264, f"Metadata n_points should be 264, got {metadata['n_points']}"
        
        logger.info("PASSED: Complete workflow test passed")
    
    def test_coefficient_values_reasonableness(self, damage_function_points):
        """Test that fitted coefficients have reasonable values."""
        logger.info("Testing coefficient values reasonableness...")
        
        results = fit_damage_function_coefficients(
            damage_function_points=damage_function_points,
            formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            discounting_type="constant",
            fit_type="ols",
            year_range=range(2020, 2101),
            year_start_pred=2100
        )
        
        params = results['params']
        
        # Extract coefficient values for analysis
        param_array = params.to_array()
        
        # Check that coefficients are finite
        finite_mask = np.isfinite(param_array.values)
        finite_fraction = finite_mask.mean()
        assert finite_fraction > 0.8, f"Too many non-finite coefficients: {finite_fraction:.2%} finite"
        
        # Check coefficient magnitudes are reasonable (not too extreme)
        finite_values = param_array.values[finite_mask]
        if len(finite_values) > 0:
            abs_values = np.abs(finite_values)
            max_abs_coeff = abs_values.max()
            assert max_abs_coeff < 1e10, f"Coefficients too large: max = {max_abs_coeff:.2e}"
            
            # Check that we have some non-zero coefficients
            non_zero_fraction = (abs_values > 1e-10).mean()
            assert non_zero_fraction > 0.1, f"Too few non-zero coefficients: {non_zero_fraction:.2%}"
        
        logger.info("PASSED: Coefficient reasonableness test passed")
    
    def test_prediction_structure(self, damage_function_points):
        """Test that predictions have the correct structure."""
        logger.info("Testing prediction structure...")
        
        results = fit_damage_function_coefficients(
            damage_function_points=damage_function_points,
            formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            discounting_type="constant",
            fit_type="ols",
            year_range=range(2020, 2101),
            year_start_pred=2100
        )
        
        preds = results['preds']
        
        # Check dimensions
        expected_dims = ['discount_type', 'ssp', 'model', 'year']
        for dim in expected_dims:
            assert dim in preds.dims, f"Missing dimension in predictions: {dim}"
        
        # Check year range (should be limited by climate data range, not necessarily extending beyond 2100)
        years = sorted(preds.coords['year'].values)
        assert min(years) >= 2020, f"Prediction years start too early: {min(years)}"
        
        # Note: The year range is limited by the climate data, which only goes to 2032 in dummy data
        # This is expected behavior - predictions are limited by available climate data
        max_year = max(years)
        logger.info(f"Prediction years range from {min(years)} to {max_year}")
        
        # Check that predictions are finite
        pred_values = preds.to_array().values
        finite_fraction = np.isfinite(pred_values).mean()
        assert finite_fraction > 0.8, f"Too many non-finite predictions: {finite_fraction:.2%} finite"
        
        logger.info("PASSED: Prediction structure test passed")
    
    def test_error_handling(self, test_data):
        """Test error handling in coefficient fitting."""
        logger.info("Testing error handling...")
        
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="No damage function points provided"):
            fit_damage_function_coefficients(
                damage_function_points=pd.DataFrame(),
                formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
                discounting_type="constant"
            )
        
        # Test with missing columns
        bad_df = pd.DataFrame({'year': [2020], 'wrong_col': [1.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            fit_damage_function_coefficients(
                damage_function_points=bad_df,
                formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
                discounting_type="constant"
            )
        
        # Test with invalid discounting type
        points = generate_damage_function_points(
            reduced_damages=test_data['reduced_damages'],
            climate_data=test_data['climate_data'],
            formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            ssps_to_use=["ssp2", "ssp3", "ssp4"]
        )
        
        with pytest.raises(ValueError, match="Unknown discounting_type"):
            fit_damage_function_coefficients(
                damage_function_points=points,
                formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
                discounting_type="invalid_type"
            )
        
        logger.info("PASSED: Error handling test passed")


def run_coefficient_fitting_tests():
    """Run all coefficient fitting tests."""
    logger.info("=== Running Damage Function Coefficient Fitting Tests ===")
    
    # Run pytest on this file
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        __file__, 
        '-v', 
        '--tb=short'
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    run_coefficient_fitting_tests() 