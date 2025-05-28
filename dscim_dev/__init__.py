"""
DSCIM Development Library
========================

A modular refactor of the DSCIM library 
for economic damage calculations from climate change.

This package provides a modular architecture for climate damage modeling with:

Architecture:
    math/: Pure mathematical functions for damage calculations and statistical operations
    processing/: Data processing workflows and orchestration functions
    io/: Input/output operations for data loading and result persistence
    tests/: Comprehensive test suite validating consistency with original DSCIM

Key Features:
    - Modular design enabling flexible workflow composition
    - Testing ensuring numerical consistency with original DSCIM

Usage Example:
    ```python
    from dscim_dev import reduce_damages, damage_function_workflow, load_real_dummy_data
    
    # Load data
    damage_data, economic_data = load_real_dummy_data()
    
    # Reduce damages with uncertainty quantification
    reduced_damages = reduce_damages(damage_data, economic_data, 
                                   discounting_type='constant')
    
    # Fit damage functions
    results = damage_function_workflow(reduced_damages, climate_data)
    ```

Version: 0.2.0
Authors: Sebastian Cadavid-Sanchez
"""

# Core mathematical functions (Phase 1)
from dscim_dev.math.core_functions import (
    discounted_damages,
    weitzman_min, 
    calculate_scc,
    ce_func,
    mean_func,
    power
)

# Damage function fitting (Phase 2)
from dscim_dev.math.damage_fitting import (
    fit_damage_function,
    model_outputs_dscim,
    modeler_dscim
)

# Core data processing operations (Phase 1)
from dscim_dev.processing.core_operations import (
    reduce_damages,
    aggregate_by_region,
    collapse_uncertainty,
    validate_processing_inputs,
    reduce_damages_workflow,
    batch_reduce_damages
)

# Damage function workflows (Phase 2)
from dscim_dev.processing.damage_workflows import (
    generate_damage_function_points,
    fit_damage_function_coefficients,
    damage_function_workflow,
    fit_damage_functions_batch,
    validate_fitting_inputs,
    summarize_fitting_results
)

# I/O operations
from dscim_dev.io.data_loaders import (
    load_real_dummy_data,
    load_zarr_with_fallback,
    create_synthetic_dummy_data,
    load_dummy_climate_data
)

from dscim_dev.io.result_exporters import (
    save_reduced_damages,
    save_fitting_results,
    save_multiple_results,
    create_summary_report
)

# Package metadata
__version__ = "0.2.0"
__author__ = "DSCIM Development Team"
__description__ = "Modular DSCIM library for economic damage calculations"

# Define public API
__all__ = [
    # Phase 1: Core functions
    "discounted_damages",
    "weitzman_min", 
    "calculate_scc",
    "ce_func",
    "mean_func",
    "power",
    "reduce_damages",
    "aggregate_by_region",
    "collapse_uncertainty",
    "validate_processing_inputs",
    "reduce_damages_workflow",
    "batch_reduce_damages",
    
    # Phase 2: Damage function fitting
    "fit_damage_function",
    "model_outputs_dscim",
    "modeler_dscim",
    "generate_damage_function_points",
    "fit_damage_function_coefficients", 
    "damage_function_workflow",
    "fit_damage_functions_batch",
    "validate_fitting_inputs",
    "summarize_fitting_results",
    
    # I/O operations
    "load_real_dummy_data",
    "load_zarr_with_fallback",
    "create_synthetic_dummy_data",
    "load_dummy_climate_data",
    "save_reduced_damages",
    "save_fitting_results",
    "save_multiple_results",
    "create_summary_report"
] 