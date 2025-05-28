# DSCIM Development Library

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![Coverage](https://codecov.io/gh/ClimateImpactLab/dscim/branch/dscim-modular/graph/badge.svg)](https://codecov.io/gh/ClimateImpactLab/dscim)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/ClimateImpactLab/dscim/actions/workflows/ci.yml)

A modular refactor of the DSCIM (Data-driven Spatial Climate Impact Model) library for economic damage calculations from climate change.

## Purpose

This project provides a clean, modular architecture for climate damage modeling while maintaining numerical consistency with the original DSCIM implementation. The refactored library enables flexible workflow composition and comprehensive testing of climate economic models.

## Current Capabilities

### Phase 1: Damage Reduction
- Economic damage reduction and aggregation with multiple discounting methods
- Support for both "adding_up" and "risk_aversion" aggregation recipes
- Comprehensive validation against original DSCIM implementation
- Flexible handling of climate change and no-climate-change scenarios

### Phase 2: Damage Function Fitting
- Damage function coefficient fitting using temperature and sea level data
- Rolling window estimation with 5-year windows (matching original DSCIM)
- Support for both OLS and quantile regression fitting methods
- Extrapolation using global consumption ratios
- Complete replication of original DSCIM damage function methodology

### Architecture
- **math/**: Pure mathematical functions for damage calculations and statistical operations
- **processing/**: Data processing workflows and orchestration functions
- **io/**: Input/output operations for data loading and result persistence
- **tests/**: Comprehensive test suite validating consistency with original DSCIM

## Installation

### Prerequisites
- Python 3.8+

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd dev-DSCIM

# Create conda environment
conda env create -f environment.yaml
conda activate dscim-testing

# Install in development mode
pip install -e .
```

## Testing

### Run All Tests
```bash
# Run complete test suite
python -m pytest dscim_dev/tests/ -v

# Run specific test categories
python -m pytest dscim_dev/tests/test_reduce_consistency.py -v
python -m pytest dscim_dev/tests/test_damage_function_coefficient_fitting.py -v
python -m pytest dscim_dev/tests/test_damage_function_coefficient_consistency.py -v
```

### Test Runner
```bash
# Use the comprehensive test runner
python dscim_dev/tests/run_damage_function_coefficient_tests.py
```

## Usage Example

```python
from dscim_dev import reduce_damages, damage_function_workflow, load_real_dummy_data

# Load data
damage_data, economic_data = load_real_dummy_data()

# Reduce damages with uncertainty quantification
reduced_damages = reduce_damages(
    damage_data, 
    economic_data, 
    recipe='adding_up',
    reduction='cc'
)

# Fit damage functions
results = damage_function_workflow(
    reduced_damages, 
    climate_data,
    formula="damages ~ -1 + anomaly + np.power(anomaly, 2)"
)
```

## Validation

The library includes comprehensive consistency tests that validate numerical equivalence with the original DSCIM implementation:

- **Damage Reduction**: Exact numerical consistency across all scenarios and parameters
- **Damage Function Fitting**: Structural and methodological consistency with known limitations due to data availability

## Next Steps

### Phase 3: Social Cost of Carbon Calculation
- Implement SCC calculation using fitted damage function coefficients
- Support for multiple pulse years and emission scenarios
- Integration with uncertainty quantification methods
- Validation against original DSCIM SCC results

### Sector Expansion
- Extend implementation to all DSCIM sectors beyond the current dummy sector
- Support for sector-specific damage functions and aggregation methods
- Comprehensive multi-sector SCC calculations

### Performance Optimization
- Parallel processing for large-scale calculations
- Memory optimization for high-resolution climate data
- Distributed computing support for uncertainty quantification

