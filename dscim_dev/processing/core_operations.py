"""
Core Data Processing Operations
==============================

Data processing workflows for economic damage reduction and aggregation.

This module implements the core data processing operations from the original DSCIM,
focusing on the damage reduction step that applies economic aggregation methods
to raw damage data. These operations bridge the gap between raw climate impact
data and the damage functions used for Social Cost of Carbon calculations.

Key Functions:
    - reduce_damages: Core damage reduction using economic aggregation methods
    - aggregate_by_region: Spatial aggregation with optional weighting
    - collapse_uncertainty: Uncertainty aggregation across Monte Carlo simulations
    - reduce_damages_workflow: Orchestrated workflow for multiple scenarios
    - batch_reduce_damages: Batch processing for large-scale calculations

DSCIM Methodology:
    The damage reduction follows the original DSCIM approach:
    1. Separate handling of historical climate (histclim) and climate change (delta) components
    2. Economic bottom-coding using GDP per capita thresholds
    3. Two aggregation recipes: "adding_up" (simple mean) and "risk_aversion" (CRRA)
    4. Support for both climate change (cc) and no climate change (no_cc) scenarios
    5. Comprehensive validation ensuring numerical consistency with original implementation

Design Principles:
    - Pure functions with comprehensive input validation
    - Exact replication of original DSCIM calculation logic
    - Support for both Dataset (separate variables) and DataArray (variable dimension) inputs
    - Flexible workflow orchestration for different output strategies
    - Comprehensive error handling and logging

Usage:
    These functions form the foundation of the damage calculation pipeline,
    typically called before damage function fitting in the complete DSCIM workflow.
"""

import numpy as np
import xarray as xr
import logging
from typing import Union, Optional, Dict, Any, List

from dscim_dev.math.core_functions import ce_func, mean_func

logger = logging.getLogger(__name__)


def reduce_damages(
    damage_data: Union[xr.DataArray, xr.Dataset],
    economic_data: xr.DataArray,
    recipe: str,
    reduction: str,
    eta: Optional[float] = None,
    bottom_coding_gdppc: float = 39.39265060424805,
    zero: bool = False,
    histclim_var: str = "histclim_dummy",
    delta_var: str = "delta_dummy"
) -> xr.DataArray:
    """
    Reduce damages using economic aggregation methods.
    
    Pure function extracted from preprocessing.py reduce_damages.
    Phase 1: Core data processing with no file I/O.
    
    This function replicates the exact logic from the original ce_from_chunk function
    to ensure numerical equivalence.
    
    Parameters
    ----------
    damage_data : Union[xr.DataArray, xr.Dataset]
        Damage data with histclim and delta variables (Dataset) or 
        DataArray with variable dimension containing histclim/delta
    economic_data : xr.DataArray
        Economic data (GDP per capita) for bottom coding
    recipe : str
        Aggregation method ("adding_up" or "risk_aversion")
    reduction : str
        Type of reduction ("cc" for climate change, "no_cc" for no climate change)
    eta : float, optional
        Risk aversion parameter (required for "risk_aversion" recipe)
    bottom_coding_gdppc : float
        Bottom coding threshold for GDP per capita
    zero : bool
        Whether to set histclim to zero for no_cc case
    histclim_var : str
        Name of historical climate variable
    delta_var : str
        Name of delta (change) variable
        
    Returns
    -------
    xr.DataArray
        Reduced damages
    """
    # Input validation
    if recipe == "adding_up" and eta is not None:
        raise ValueError("Adding up does not take an eta argument. Please set to None.")
    if recipe == "risk_aversion" and eta is None:
        raise ValueError("Risk aversion recipe requires eta parameter.")
    
    if reduction not in ["cc", "no_cc"]:
        raise ValueError("reduction must be 'cc' or 'no_cc'")
    
    # Handle both Dataset and DataArray inputs
    if isinstance(damage_data, xr.Dataset):
        # Dataset with separate variables
        if histclim_var not in damage_data.data_vars:
            # Try to find histclim variable by pattern matching
            histclim_candidates = [var for var in damage_data.data_vars if 'histclim' in var.lower()]
            if histclim_candidates:
                histclim_var = histclim_candidates[0]
                logger.info(f"Using histclim variable: {histclim_var}")
            else:
                raise ValueError(f"Variable '{histclim_var}' not found in damage_data. Available variables: {list(damage_data.data_vars)}")
        
        if delta_var not in damage_data.data_vars:
            # Try to find delta variable by pattern matching
            delta_candidates = [var for var in damage_data.data_vars if 'delta' in var.lower()]
            if delta_candidates:
                delta_var = delta_candidates[0]
                logger.info(f"Using delta variable: {delta_var}")
            else:
                raise ValueError(f"Variable '{delta_var}' not found in damage_data. Available variables: {list(damage_data.data_vars)}")
        
        # Extract damage components from Dataset
        histclim = damage_data[histclim_var]
        delta = damage_data[delta_var]
        
    else:
        # DataArray with variable dimension (legacy format)
        if 'variable' not in damage_data.dims:
            raise ValueError("DataArray input must have 'variable' dimension with histclim/delta")
        
        # Check required variables exist
        if histclim_var not in damage_data.variable.values:
            raise ValueError(f"Variable '{histclim_var}' not found in damage_data variable dimension")
        if delta_var not in damage_data.variable.values:
            raise ValueError(f"Variable '{delta_var}' not found in damage_data variable dimension")
        
        # Extract damage components from DataArray
        histclim = damage_data.sel(variable=histclim_var)
        delta = damage_data.sel(variable=delta_var)
    
    # Replicate the exact coordinate selection logic from ce_from_chunk
    # The original function extracts coordinates from the chunk and selects matching economic data
    
    # Get coordinates that exist in both damage and economic data
    damage_coords = {}
    for coord in ['year', 'ssp', 'model', 'region']:
        if coord in histclim.dims and coord in economic_data.dims:
            damage_coords[coord] = histclim[coord].values
    
    # Select economic data for matching coordinates, dropping unused dimensions
    gdppc = economic_data.sel(damage_coords, drop=True)
    
    # Apply the exact calculation logic from ce_from_chunk
    if reduction == "no_cc":
        if zero:
            # Set histclim to zero (using xr.where as in original)
            histclim_adjusted = xr.where(histclim == 0, 0, 0)
        else:
            histclim_adjusted = histclim
        
        # Original formula: gdppc + histclim.mean("batch") - histclim
        calculation = gdppc + histclim_adjusted.mean("batch") - histclim_adjusted
        
    elif reduction == "cc":
        # Original formula: gdppc - delta
        calculation = gdppc - delta
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")
    
    # Apply aggregation recipe using the exact same structure as original
    # The np.maximum is applied INSIDE the function calls, not before
    if recipe == "adding_up":
        result = mean_func(
            np.maximum(
                calculation,
                bottom_coding_gdppc,
            ),
            "batch",
        )
    elif recipe == "risk_aversion":
        result = ce_func(
            np.maximum(
                calculation,
                bottom_coding_gdppc,
            ),
            "batch",
            eta=eta,
        )
    else:
        raise ValueError(f"Unknown recipe: {recipe}")
    
    # Add metadata to match original output
    result.attrs.update({
        'recipe': recipe,
        'reduction': reduction,
        'bottom_coding_gdppc': bottom_coding_gdppc,
        'zero_histclim': int(zero)  # Convert boolean to int for NetCDF compatibility
    })
    
    if eta is not None:
        result.attrs['eta'] = eta
    
    return result


def aggregate_by_region(
    data: xr.DataArray,
    weights: Optional[xr.DataArray] = None,
    region_dim: str = "region"
) -> xr.DataArray:
    """
    Aggregate data across regions using optional weights.
    
    Pure function for spatial aggregation.
    
    Parameters
    ----------
    data : xr.DataArray
        Data to aggregate
    weights : xr.DataArray, optional
        Weights for aggregation (e.g., population, GDP)
    region_dim : str
        Name of region dimension
        
    Returns
    -------
    xr.DataArray
        Aggregated data
    """
    if region_dim not in data.dims:
        raise ValueError(f"Region dimension '{region_dim}' not found in data")
    
    if weights is None:
        # Simple mean
        return data.mean(dim=region_dim)
    else:
        # Weighted average
        if region_dim not in weights.dims:
            raise ValueError(f"Region dimension '{region_dim}' not found in weights")
        
        # Normalize weights
        weight_sum = weights.sum(dim=region_dim)
        normalized_weights = weights / weight_sum
        
        # Calculate weighted average
        weighted_data = data * normalized_weights
        return weighted_data.sum(dim=region_dim)


def collapse_uncertainty(
    data: xr.DataArray,
    method: str = "mean",
    uncertainty_dim: str = "simulation",
    quantiles: Optional[List[float]] = None
) -> xr.DataArray:
    """
    Collapse uncertainty dimension using specified method.
    
    Pure function for uncertainty reduction.
    
    Parameters
    ----------
    data : xr.DataArray
        Data with uncertainty dimension
    method : str
        Aggregation method ("mean", "median", "quantile")
    uncertainty_dim : str
        Name of uncertainty dimension
    quantiles : List[float], optional
        Quantiles to compute (for "quantile" method)
        
    Returns
    -------
    xr.DataArray
        Data with uncertainty collapsed
    """
    if uncertainty_dim not in data.dims:
        raise ValueError(f"Uncertainty dimension '{uncertainty_dim}' not found in data")
    
    if method == "mean":
        return data.mean(dim=uncertainty_dim)
    elif method == "median":
        return data.median(dim=uncertainty_dim)
    elif method == "quantile":
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]  # Default quantiles
        return data.quantile(quantiles, dim=uncertainty_dim)
    else:
        raise ValueError(f"Unknown method: {method}")


def validate_processing_inputs(
    damage_data: xr.DataArray,
    economic_data: xr.DataArray,
    required_vars: List[str]
) -> tuple[bool, List[str]]:
    """
    Validate inputs for data processing operations.
    
    Pure function for input validation.
    
    Parameters
    ----------
    damage_data : xr.DataArray
        Damage data to validate
    economic_data : xr.DataArray
        Economic data to validate
    required_vars : List[str]
        Required variables in damage_data
        
    Returns
    -------
    tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check damage data
    if not isinstance(damage_data, xr.DataArray):
        issues.append("damage_data must be xarray.DataArray")
    else:
        # Check for required variables
        for var in required_vars:
            if var not in damage_data:
                issues.append(f"Required variable '{var}' not found in damage_data")
        
        # Check for batch dimension
        if "batch" not in damage_data.dims:
            issues.append("'batch' dimension required in damage_data")
    
    # Check economic data
    if not isinstance(economic_data, xr.DataArray):
        issues.append("economic_data must be xarray.DataArray")
    else:
        # Check for required economic variables
        if economic_data.size == 0:
            issues.append("economic_data is empty")
    
    # Check coordinate alignment
    if isinstance(damage_data, xr.DataArray) and isinstance(economic_data, xr.DataArray):
        common_dims = set(damage_data.dims) & set(economic_data.dims)
        if not common_dims:
            issues.append("No common dimensions between damage_data and economic_data")
    
    is_valid = len(issues) == 0
    return is_valid, issues 


def reduce_damages_workflow(
    damage_data: Union[xr.DataArray, xr.Dataset],
    economic_data: xr.DataArray,
    scenarios: List[Dict[str, Any]],
    output_strategy: str = "memory",
    output_dir: Optional[str] = None,
    save_format: str = "zarr",
    **reduce_kwargs
) -> Union[Dict[str, xr.DataArray], Dict[str, str]]:
    """
    Flexible workflow for running multiple damage reduction scenarios.
    
    This function provides flexibility in how results are handled:
    - Keep in memory for further processing
    - Save to disk automatically
    - Both (save and return in-memory copies)
    
    Parameters
    ----------
    damage_data : Union[xr.DataArray, xr.Dataset]
        Damage data
    economic_data : xr.DataArray
        Economic data (GDP per capita)
    scenarios : List[Dict[str, Any]]
        List of scenario configurations, each containing:
        - recipe: str ("adding_up" or "risk_aversion")
        - reduction: str ("cc" or "no_cc")
        - eta: Optional[float] (for risk_aversion)
        - name: Optional[str] (scenario name for output)
    output_strategy : str
        How to handle results:
        - "memory": Return results in memory (default)
        - "disk": Save to disk, return file paths
        - "both": Save to disk AND return in memory
    output_dir : str, optional
        Directory to save results (required if output_strategy includes "disk")
    save_format : str
        Format for saving ("zarr", "netcdf", "csv")
    **reduce_kwargs
        Additional arguments passed to reduce_damages()
        
    Returns
    -------
    Union[Dict[str, xr.DataArray], Dict[str, str]]
        If output_strategy is "memory" or "both": Dict mapping scenario names to DataArrays
        If output_strategy is "disk": Dict mapping scenario names to file paths
        
    Examples
    --------
    # Keep results in memory for further processing
    results = reduce_damages_workflow(
        damage_data, economic_data, scenarios,
        output_strategy="memory"
    )
    
    # Save to disk automatically
    file_paths = reduce_damages_workflow(
        damage_data, economic_data, scenarios,
        output_strategy="disk",
        output_dir="output/scenarios"
    )
    
    # Both: save to disk AND keep in memory
    results = reduce_damages_workflow(
        damage_data, economic_data, scenarios,
        output_strategy="both",
        output_dir="output/scenarios"
    )
    """
    from pathlib import Path
    from dscim_dev.io.result_exporters import save_reduced_damages
    
    # Validate inputs
    if output_strategy not in ["memory", "disk", "both"]:
        raise ValueError("output_strategy must be 'memory', 'disk', or 'both'")
    
    if output_strategy in ["disk", "both"] and output_dir is None:
        raise ValueError("output_dir is required when output_strategy includes 'disk'")
    
    # Create output directory if needed
    if output_strategy in ["disk", "both"]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process scenarios
    memory_results = {}
    file_paths = {}
    
    for i, scenario in enumerate(scenarios):
        # Generate scenario name
        scenario_name = scenario.get('name', f"scenario_{i+1}")
        
        # Extract scenario parameters
        scenario_params = {k: v for k, v in scenario.items() if k != 'name'}
        
        # Merge with additional kwargs
        params = {**reduce_kwargs, **scenario_params}
        
        # Run computation
        result = reduce_damages(
            damage_data=damage_data,
            economic_data=economic_data,
            **params
        )
        
        # Handle output based on strategy
        if output_strategy in ["memory", "both"]:
            memory_results[scenario_name] = result
        
        if output_strategy in ["disk", "both"]:
            # Generate filename
            filename = scenario_name.lower().replace(" ", "_").replace("-", "_")
            
            if save_format == "zarr":
                file_path = output_path / f"{filename}.zarr"
            elif save_format == "netcdf":
                file_path = output_path / f"{filename}.nc"
            elif save_format == "csv":
                file_path = output_path / f"{filename}.csv"
            else:
                raise ValueError(f"Unsupported save_format: {save_format}")
            
            # Save result
            save_reduced_damages(result, str(file_path), format=save_format)
            file_paths[scenario_name] = str(file_path)
    
    # Return appropriate results
    if output_strategy == "memory":
        return memory_results
    elif output_strategy == "disk":
        return file_paths
    else:  # "both"
        return memory_results


def batch_reduce_damages(
    scenarios: List[Dict[str, Any]],
    data_loader_func: callable = None,
    config_path: Optional[str] = None,
    output_strategy: str = "memory",
    output_dir: Optional[str] = None,
    **kwargs
) -> Union[Dict[str, xr.DataArray], Dict[str, str]]:
    """
    Convenience function for batch processing with automatic data loading.
    
    Parameters
    ----------
    scenarios : List[Dict[str, Any]]
        List of scenario configurations
    data_loader_func : callable, optional
        Function to load data. If None, uses load_real_dummy_data()
    config_path : str, optional
        Path to configuration file. If None, uses default for data_loader_func
    output_strategy : str
        How to handle results ("memory", "disk", "both")
    output_dir : str, optional
        Directory to save results
    **kwargs
        Additional arguments passed to reduce_damages_workflow()
        
    Returns
    -------
    Union[Dict[str, xr.DataArray], Dict[str, str]]
        Results based on output_strategy
        
    Examples
    --------
    # Simple batch processing with default data loader
    scenarios = [
        {"recipe": "adding_up", "reduction": "cc", "name": "baseline"},
        {"recipe": "risk_aversion", "reduction": "cc", "eta": 2.0, "name": "risk_averse"}
    ]
    
    results = batch_reduce_damages(scenarios, output_strategy="memory")
    
    # From examples/ folder, specify config path
    results = batch_reduce_damages(
        scenarios, 
        config_path="../configs/dummy_config.yaml",
        output_strategy="memory"
    )
    """
    # Load data
    if data_loader_func is None:
        from dscim_dev.io.data_loaders import load_real_dummy_data
        data_loader_func = load_real_dummy_data
    
    # Call data loader with config_path if provided
    if config_path is not None:
        damage_data, economic_data = data_loader_func(config_path=config_path)
    else:
        damage_data, economic_data = data_loader_func()
    
    gdppc = economic_data['gdppc']
    
    # Run workflow
    return reduce_damages_workflow(
        damage_data=damage_data,
        economic_data=gdppc,
        scenarios=scenarios,
        output_strategy=output_strategy,
        output_dir=output_dir,
        **kwargs
    ) 