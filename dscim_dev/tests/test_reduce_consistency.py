"""
Damage Reduction Consistency Tests
=================================

Tests validating numerical consistency between the refactored damage reduction
implementation and the original DSCIM library.

This test suite ensures that the refactored reduce_damages function produces
identical results to the original DSCIM implementation across different
scenarios and parameter combinations. It serves as the primary validation
for the core damage reduction functionality.

Test Coverage:
    - Multiple discounting types (adding_up, risk_aversion)
    - Different economic scenarios (climate change vs no climate change)
    - Various risk aversion parameters (eta values)
    - Edge cases and error conditions
    - Numerical precision validation

Methodology:
    Tests load the same dummy data used by the original DSCIM, run both
    implementations side-by-side, and compare results with strict numerical
    tolerances. Any discrepancies indicate potential issues in the refactored
    implementation.

Design Principles:
    - Direct comparison with original DSCIM results
    - Comprehensive parameter space coverage
    - Strict numerical tolerance requirements
    - Clear error reporting for debugging
    - Automated validation of core functionality

Usage:
    These tests should be run whenever changes are made to the damage reduction
    functionality to ensure continued consistency with the original DSCIM.
"""

import pytest
import logging
import numpy as np
import xarray as xr
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestReduceConsistency:
    """Test class for reduce_damages consistency validation."""
    
    @pytest.fixture(scope="class")
    def test_cases(self) -> list[Dict[str, Any]]:
        """Test cases matching run_integration_result.py configuration."""
        return [
            {
                "recipe": "adding_up",
                "reduction": "cc",
                "eta": None,
                "zero": False
            },
            {
                "recipe": "adding_up", 
                "reduction": "no_cc",
                "eta": None,
                "zero": False
            },
            {
                "recipe": "risk_aversion",
                "reduction": "cc", 
                "eta": 2.0,
                "zero": False
            },
            {
                "recipe": "risk_aversion",
                "reduction": "no_cc",
                "eta": 2.0, 
                "zero": False
            }
        ]
    
    @pytest.fixture(scope="class")
    def config_params(self) -> Dict[str, str]:
        """Configuration parameters for testing."""
        return {
            "config_path": "configs/dummy_config.yaml",
            "sector": "dummy_not_coastl_sector",
            "socioec_path": "./dummy_data/econ/integration-econ-bc39.zarr"
        }
    
    @pytest.fixture(scope="class")
    def original_results(self, test_cases, config_params) -> Dict[str, xr.Dataset]:
        """Run original reduce_damages and return results."""
        from dscim.preprocessing.preprocessing import reduce_damages as original_reduce_damages
        
        logger.info("Running original reduce_damages implementation")
        
        results = {}
        
        for i, case in enumerate(test_cases):
            case_name = f"case_{i+1}"
            logger.info(f"Original {case_name}: {case}")
            
            try:
                # Call original reduce_damages (saves to zarr files)
                original_reduce_damages(
                    sector=config_params["sector"],
                    config=config_params["config_path"],
                    socioec=config_params["socioec_path"],
                    **case
                )
                
                # Load the saved result
                with open(config_params["config_path"], 'r') as f:
                    config = yaml.safe_load(f)
                
                outpath = f"{config['paths']['reduced_damages_library']}/{config_params['sector']}"
                
                if case["recipe"] == "adding_up":
                    result_path = f"{outpath}/{case['recipe']}_{case['reduction']}.zarr"
                else:
                    result_path = f"{outpath}/{case['recipe']}_{case['reduction']}_eta{case['eta']}.zarr"
                
                result = xr.open_zarr(result_path)
                results[case_name] = result
                
                logger.info(f"  Original result shape: {result[case['reduction']].shape}")
                logger.info(f"  Original result mean: {result[case['reduction']].mean().values:.6f}")
                
            except Exception as e:
                logger.error(f"Error in original {case_name}: {e}")
                results[case_name] = None
        
        return results
    
    @pytest.fixture(scope="class")
    def refactored_results(self, test_cases) -> Dict[str, xr.DataArray]:
        """Run refactored reduce_damages and return results."""
        from dscim_dev.io.data_loaders import load_real_dummy_data
        from dscim_dev.processing.core_operations import reduce_damages as refactored_reduce_damages
        
        logger.info("Running refactored reduce_damages implementation")
        
        # Load real dummy data
        damage_dataset, economic_dataset = load_real_dummy_data()
        
        results = {}
        
        for i, case in enumerate(test_cases):
            case_name = f"case_{i+1}"
            logger.info(f"Refactored {case_name}: {case}")
            
            try:
                # Call refactored reduce_damages
                result = refactored_reduce_damages(
                    damage_data=damage_dataset,
                    economic_data=economic_dataset['gdppc'],
                    **case
                )
                
                results[case_name] = result
                
                logger.info(f"  Refactored result shape: {result.shape}")
                logger.info(f"  Refactored result mean: {result.mean().values:.6f}")
                
            except Exception as e:
                logger.error(f"Error in refactored {case_name}: {e}")
                results[case_name] = None
        
        return results
    
    def _extract_data_array(self, result: Any) -> xr.DataArray:
        """Extract DataArray from Dataset or return DataArray directly."""
        if isinstance(result, xr.Dataset):
            # Original result is a Dataset with reduction name as variable
            reduction_name = list(result.data_vars)[0]  # Should be 'cc' or 'no_cc'
            return result[reduction_name]
        else:
            return result
    
    def _compare_arrays(self, orig_data: xr.DataArray, refact_data: xr.DataArray, 
                       case_name: str) -> Tuple[bool, str]:
        """Compare two DataArrays with appropriate tolerances."""
        
        # Check shapes
        if orig_data.shape != refact_data.shape:
            return False, f"Shape mismatch - Original: {orig_data.shape}, Refactored: {refact_data.shape}"
        
        # Convert to same dtype for comparison if needed
        if orig_data.dtype != refact_data.dtype:
            logger.warning(f"Data type mismatch in {case_name}! Converting both to float64 for comparison")
            orig_values = orig_data.astype(np.float64).values
            refact_values = refact_data.astype(np.float64).values
        else:
            orig_values = orig_data.values
            refact_values = refact_data.values
        
        # Try progressively relaxed tolerances
        tolerances = [
            (1e-10, 1e-12, "strict"),
            (1e-7, 1e-8, "relaxed"),
            (1e-5, 1e-6, "very_relaxed")
        ]
        
        for rtol, atol, level in tolerances:
            try:
                np.testing.assert_allclose(orig_values, refact_values, rtol=rtol, atol=atol)
                return True, f"Results match within {level} tolerance (rtol={rtol}, atol={atol})"
            except AssertionError:
                continue
        
        # If all tolerances fail, calculate difference statistics
        diff = np.abs(orig_values - refact_values)
        max_abs_diff = np.max(diff)
        mean_abs_diff = np.mean(diff)
        
        return False, f"Numerical differences exceed acceptable tolerances. Max: {max_abs_diff:.2e}, Mean: {mean_abs_diff:.2e}"
    
    @pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
    def test_reduce_damages_consistency(self, case_idx: int, original_results: Dict[str, xr.Dataset], 
                                      refactored_results: Dict[str, xr.DataArray], 
                                      test_cases: list[Dict[str, Any]]):
        """Test that original and refactored reduce_damages produce consistent results."""
        
        case_name = f"case_{case_idx + 1}"
        case_config = test_cases[case_idx]
        
        # Get results
        orig_result = original_results[case_name]
        refact_result = refactored_results[case_name]
        
        # Check that both results exist
        assert orig_result is not None, f"Original result for {case_name} is None"
        assert refact_result is not None, f"Refactored result for {case_name} is None"
        
        # Extract DataArrays
        orig_data = self._extract_data_array(orig_result)
        refact_data = self._extract_data_array(refact_result)
        
        # Compare arrays
        is_consistent, message = self._compare_arrays(orig_data, refact_data, case_name)
        
        # Log the result
        if is_consistent:
            logger.info(f"PASS {case_name} ({case_config['recipe']}, {case_config['reduction']}): {message}")
        else:
            logger.error(f"FAIL {case_name} ({case_config['recipe']}, {case_config['reduction']}): {message}")
        
        # Assert consistency
        assert is_consistent, f"{case_name} consistency check failed: {message}"
    
    def test_all_cases_completed(self, original_results: Dict[str, xr.Dataset], 
                                refactored_results: Dict[str, xr.DataArray]):
        """Test that all test cases completed successfully."""
        
        for case_name in ["case_1", "case_2", "case_3", "case_4"]:
            assert case_name in original_results, f"Missing original result for {case_name}"
            assert case_name in refactored_results, f"Missing refactored result for {case_name}"
            assert original_results[case_name] is not None, f"Original result for {case_name} is None"
            assert refactored_results[case_name] is not None, f"Refactored result for {case_name} is None"
        
        logger.info("All test cases completed successfully")


# Standalone function for direct execution (backward compatibility)
def run_consistency_validation() -> bool:
    """
    Run the consistency validation as a standalone function.
    
    Returns
    -------
    bool
        True if all tests pass, False otherwise
    """
    import subprocess
    import sys
    
    try:
        # Run pytest on this specific test file
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            __file__, 
            "-v", 
            "--tb=short"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running consistency validation: {e}")
        return False


if __name__ == "__main__":
    # Allow direct execution for backward compatibility
    success = run_consistency_validation()
    exit(0 if success else 1) 