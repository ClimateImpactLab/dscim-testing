"""
Damage Function Workflows
=========================

High-level workflows for damage function calculation and coefficient fitting.

This module orchestrates the complete damage function calculation pipeline,
combining reduced damages with climate data to generate damage function points,
fit coefficients using the original DSCIM methodology, and produce predictions
for Social Cost of Carbon calculations.

Key Functions:
    - generate_damage_function_points: Combine reduced damages with climate data
    - fit_damage_function_coefficients: Fit damage functions using DSCIM methodology
    - damage_function_workflow: Complete end-to-end damage function pipeline
    - fit_damage_functions_batch: Batch processing for multiple scenarios
    - validate_fitting_inputs: Input validation for damage function fitting

DSCIM Methodology:
    The damage function workflow follows the original DSCIM approach:
    1. Merge reduced damages with climate variables (temperature, sea level)
    2. Apply SSP filtering to match original DSCIM scenarios
    3. Fit damage functions using rolling window estimation (5-year windows)
    4. Support multiple discounting types (constant, ramsey, growth-weighted)
    5. Generate predictions over specified climate variable ranges
    6. Handle extrapolation using global consumption ratios

Workflow Integration:
    This module bridges the gap between the damage reduction step (core_operations)
    and the final SCC calculation, implementing the second major phase of the
    DSCIM pipeline. It ensures numerical consistency with the original implementation
    while providing a modular, testable interface.

Design Principles:
    - Comprehensive workflow orchestration with flexible configuration
    - Full replication of original DSCIM damage function methodology
    - Support for multiple discounting approaches and fitting methods
    - Robust error handling and validation at each step
    - Clear separation between data processing and mathematical operations

Usage:
    These workflows are typically called after damage reduction to complete
    the damage function calculation pipeline before SCC computation.
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from itertools import product

from .core_operations import reduce_damages
from ..math.damage_fitting import model_outputs_dscim
from ..io.data_loaders import load_real_dummy_data, load_dummy_climate_data

logger = logging.getLogger(__name__)


def _extract_climate_variables_from_formula(formula: str) -> List[str]:
    """Extract climate variable names from R-style formula."""
    climate_vars = []
    if 'anomaly' in formula:
        climate_vars.append('anomaly')
    if 'gmsl' in formula:
        climate_vars.append('gmsl')
    return climate_vars


def generate_damage_function_points(
    reduced_damages: Dict[str, xr.DataArray],
    climate_data: xr.Dataset,
    formula: str,
    ssps_to_use: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate damage function points by combining reduced damages with climate data.
    
    Parameters
    ----------
    reduced_damages : Dict[str, xr.DataArray]
        Dictionary of reduced damage arrays for different scenarios
        (should already be population-scaled and regionally aggregated for adding_up)
    climate_data : xr.Dataset
        Climate data containing temperature anomaly and sea level data
    formula : str
        R-style formula for damage function (e.g., "damages ~ -1 + anomaly + np.power(anomaly, 2)")
    ssps_to_use : Optional[List[str]]
        List of SSPs to include. If None, uses all available SSPs.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with damage function points ready for fitting
    """
    logger.info("Generating damage function points from reduced damages and climate data")
    
    # Extract climate variables based on formula
    climate_vars = _extract_climate_variables_from_formula(formula)
    
    all_points = []
    
    for scenario_name, damages in reduced_damages.items():
        logger.info(f"Processing scenario: {scenario_name}")
        
        # Filter SSPs if specified
        if ssps_to_use is not None:
            # Filter damages to only include specified SSPs
            available_ssps = damages.coords['ssp'].values
            ssps_to_keep = [ssp for ssp in ssps_to_use if ssp in available_ssps]
            if ssps_to_keep:
                damages = damages.sel(ssp=ssps_to_keep)
            else:
                logger.warning(f"No matching SSPs found for scenario {scenario_name}. Available: {available_ssps}, Requested: {ssps_to_use}")
                continue
        
        # Note: Regional aggregation and population scaling should already be done
        # in the data loading step for adding_up discounting (matching original DSCIM)
        if 'region' in damages.dims:
            logger.warning(f"Region dimension still present in damages data. This should have been aggregated in data loading for adding_up discounting.")
            # For backward compatibility, still aggregate if region dimension exists
            logger.info(f"Aggregating damages across {len(damages.coords['region'])} regions")
            damages = damages.sum(dim='region')
        
        # Convert damages to DataFrame
        damages_df = damages.to_dataframe(name='damages').reset_index()
        
        # Find common dimensions between damages and climate data
        damage_dims = set(damages.dims)
        climate_dims = set(climate_data.dims)
        common_dims = damage_dims & climate_dims
        
        logger.info(f"Damage dimensions: {damage_dims}")
        logger.info(f"Climate dimensions: {climate_dims}")
        logger.info(f"Common dimensions: {common_dims}")
        
        if not common_dims:
            logger.error(f"No common dimensions between damages and climate data")
            continue
        
        # Create selection dictionary for climate data using only common dimensions
        climate_selection = {}
        for dim in common_dims:
            if dim in damages.coords:
                climate_selection[dim] = damages.coords[dim]
        
        # Filter climate data to match damage coordinates
        climate_subset = climate_data.sel(climate_selection)
        
        # Convert climate data to DataFrame
        climate_df = climate_subset.to_dataframe().reset_index()
        
        # Merge damages with climate data on common columns that exist in both DataFrames
        damages_cols = set(damages_df.columns)
        climate_cols = set(climate_df.columns)
        merge_cols = list(damages_cols & climate_cols & common_dims)
        
        logger.info(f"Available columns in damages_df: {list(damages_cols)}")
        logger.info(f"Available columns in climate_df: {list(climate_cols)}")
        logger.info(f"Merging on columns: {merge_cols}")
        
        if not merge_cols:
            logger.error(f"No common columns found for merging")
            continue
        
        points_df = pd.merge(damages_df, climate_df, on=merge_cols, how='inner')
        
        # Add scenario identifier
        points_df['scenario'] = scenario_name
        
        logger.info(f"Generated {len(points_df)} points for scenario {scenario_name}")
        all_points.append(points_df)
    
    if not all_points:
        raise ValueError("No damage function points generated. Check SSP filtering and data compatibility.")
    
    # Combine all scenarios
    combined_points = pd.concat(all_points, ignore_index=True)
    
    logger.info(f"Generated {len(combined_points)} damage function points")
    
    return combined_points


def fit_damage_function_coefficients(
    damage_function_points: pd.DataFrame,
    formula: str,
    discounting_type: str = "constant",
    global_consumption: Optional[xr.DataArray] = None,
    fit_type: str = "ols",
    year_range: Optional[range] = None,
    extrapolation_method: str = "global_c_ratio",
    year_start_pred: int = 2100,
    quantiles: Optional[List[float]] = None
) -> Dict[str, xr.Dataset]:
    """
    Fit damage function coefficients using the original DSCIM methodology.
    
    This replicates the damage_function_calculation method from MainRecipe.
    
    Parameters
    ----------
    damage_function_points : pd.DataFrame
        Damage function points from generate_damage_function_points
    formula : str
        R-style formula for fitting (e.g., "damages ~ -1 + anomaly + np.power(anomaly, 2)")
    discounting_type : str
        Type of discounting ("constant", "ramsey", "gwr")
    global_consumption : xr.DataArray, optional
        Global consumption data for extrapolation
    fit_type : str
        Type of fitting ("ols", "quantreg")
    year_range : range, optional
        Years to fit over (default: 2020-2099)
    extrapolation_method : str
        Method for extrapolating beyond year_start_pred
    year_start_pred : int
        Year to start extrapolation
    quantiles : List[float], optional
        Quantiles for quantile regression
        
    Returns
    -------
    Dict[str, xr.Dataset]
        Dictionary with keys:
        - 'params': Fitted coefficients by year
        - 'preds': Predictions from fitted model
    """
    logger.info(f"Fitting damage functions with {discounting_type} discounting")
    
    # Validate inputs first before accessing any columns
    if len(damage_function_points) == 0:
        raise ValueError("No damage function points provided")
    
    required_cols = ['damages', 'year']
    missing_cols = [col for col in required_cols if col not in damage_function_points.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if year_range is None:
        # Determine year range from available data instead of defaulting to 2020-2100
        available_years = sorted(damage_function_points['year'].unique())
        year_range = range(min(available_years), max(available_years) + 1)
        logger.info(f"Using year range from data: {min(available_years)}-{max(available_years)}")
    
    params_list, preds_list = [], []
    
    # Handle different discounting types (following original DSCIM logic)
    if discounting_type == "constant_model_collapsed":
        # Fit by SSP
        for ssp in damage_function_points["ssp"].unique():
            fit_subset = damage_function_points[damage_function_points["ssp"] == ssp]
            
            global_c_subset = None
            if global_consumption is not None:
                global_c_subset = global_consumption.sel({"ssp": ssp})
            
            # Fit damage function
            damage_function = model_outputs_dscim(
                damage_function=fit_subset,
                formula=formula,
                type_estimation=fit_type,
                global_c=global_c_subset,
                extrapolation_type=extrapolation_method,
                quantiles=quantiles,
                year_range=year_range,
                year_start_pred=year_start_pred,
            )
            
            # Add dimensions
            params = damage_function["parameters"].expand_dims({
                "discount_type": [discounting_type],
                "ssp": [ssp],
                "model": ["combined"]
            })
            
            preds = damage_function["preds"].expand_dims({
                "discount_type": [discounting_type],
                "ssp": [ssp],
                "model": ["combined"]
            })
            
            params_list.append(params)
            preds_list.append(preds)
    
    elif discounting_type == "constant" or "ramsey" in discounting_type:
        # Fit by SSP-model combination
        ssp_model_combinations = list(product(
            damage_function_points.ssp.unique(),
            damage_function_points.model.unique()
        ))
        
        for ssp, model in ssp_model_combinations:
            fit_subset = damage_function_points[
                (damage_function_points["ssp"] == ssp) &
                (damage_function_points["model"] == model)
            ]
            
            if len(fit_subset) == 0:
                logger.warning(f"No data for SSP={ssp}, model={model}")
                continue
            
            global_c_subset = None
            if global_consumption is not None:
                global_c_subset = global_consumption.sel({"ssp": ssp, "model": model})
            
            # Fit damage function
            damage_function = model_outputs_dscim(
                damage_function=fit_subset,
                formula=formula,
                type_estimation=fit_type,
                global_c=global_c_subset,
                extrapolation_type=extrapolation_method,
                quantiles=quantiles,
                year_range=year_range,
                year_start_pred=year_start_pred,
            )
            
            # Add dimensions
            params = damage_function["parameters"].expand_dims({
                "discount_type": [discounting_type],
                "ssp": [ssp],
                "model": [model]
            })
            
            preds = damage_function["preds"].expand_dims({
                "discount_type": [discounting_type],
                "ssp": [ssp],
                "model": [model]
            })
            
            params_list.append(params)
            preds_list.append(preds)
    
    elif "gwr" in discounting_type:
        # Fit across all SSP-model combinations
        fit_subset = damage_function_points
        
        # Fit damage function
        damage_function = model_outputs_dscim(
            damage_function=fit_subset,
            formula=formula,
            type_estimation=fit_type,
            global_c=global_consumption,
            extrapolation_type=extrapolation_method,
            quantiles=quantiles,
            year_range=year_range,
            year_start_pred=year_start_pred,
        )
        
        # Add dimensions
        params = damage_function["parameters"].expand_dims({
            "discount_type": [discounting_type],
            "ssp": ["combined"],
            "model": ["combined"]
        })
        
        preds = damage_function["preds"].expand_dims({
            "discount_type": [discounting_type],
            "ssp": ["combined"],
            "model": ["combined"]
        })
        
        params_list.append(params)
        preds_list.append(preds)
    
    else:
        raise ValueError(f"Unknown discounting_type: {discounting_type}")
    
    # Combine results
    if not params_list:
        raise ValueError("No damage functions could be fitted")
    
    result = {
        'params': xr.combine_by_coords(params_list),
        'preds': xr.combine_by_coords(preds_list)
    }
    
    logger.info(f"Successfully fitted {len(params_list)} damage function(s)")
    return result


def damage_function_workflow(
    reduced_damages: Dict[str, xr.DataArray],
    climate_data: xr.Dataset,
    formula: str = "damages ~ -1 + anomaly + np.power(anomaly, 2)",
    discounting_type: str = "constant",
    global_consumption: Optional[xr.DataArray] = None,
    ssps_to_use: Optional[List[str]] = None,
    **fit_kwargs
) -> Dict[str, Any]:
    """
    Complete workflow for damage function generation.
    
    This orchestrates the full damage function calculation process,
    replicating the original DSCIM workflow.
    
    Parameters
    ----------
    reduced_damages : Dict[str, xr.DataArray]
        Reduced damages from reduce_damages step
    climate_data : xr.Dataset
        Climate data (temperature, sea level)
    formula : str
        Damage function formula
    discounting_type : str
        Discounting approach
    global_consumption : xr.DataArray, optional
        Global consumption for extrapolation
    ssps_to_use : Optional[List[str]]
        List of SSPs to include. If None, uses all available SSPs.
    **fit_kwargs
        Additional arguments for fitting
        
    Returns
    -------
    Dict[str, Any]
        Complete damage function results with:
        - 'damage_function_points': DataFrame of input points
        - 'damage_function_coefficients': Fitted coefficients
        - 'damage_function_predictions': Model predictions
        - 'metadata': Fitting metadata
    """
    logger.info("Starting complete damage function workflow")
    
    # Step 1: Generate damage function points
    damage_function_points = generate_damage_function_points(
        reduced_damages=reduced_damages,
        climate_data=climate_data,
        formula=formula,
        ssps_to_use=ssps_to_use
    )
    
    # Step 2: Fit damage function coefficients
    damage_function_results = fit_damage_function_coefficients(
        damage_function_points=damage_function_points,
        formula=formula,
        discounting_type=discounting_type,
        global_consumption=global_consumption,
        **fit_kwargs
    )
    
    # Step 3: Compile results
    results = {
        'damage_function_points': damage_function_points,
        'damage_function_coefficients': damage_function_results['params'],
        'damage_function_predictions': damage_function_results['preds'],
        'metadata': {
            'formula': formula,
            'discounting_type': discounting_type,
            'n_points': len(damage_function_points),
            'n_scenarios': len(reduced_damages),
            'fit_kwargs': fit_kwargs
        }
    }
    
    logger.info("Damage function workflow completed successfully")
    return results


# Keep existing functions for backward compatibility
def fit_damage_functions_batch(
    damage_data: xr.DataArray,
    regions: List[str],
    gcms: List[str],
    scenarios: List[str],
    model_type: str = "polynomial",
    degree: int = 2,
    time_dim: str = "year",
    **fit_kwargs
) -> pd.DataFrame:
    """
    Driver that loops over every region, GCM, and scenario combination.
    
    Phase 2 requirement: Orchestrates pure fit_damage_function calls
    across all dimensional combinations. No I/O operations.
    
    Parameters
    ----------
    damage_data : xr.DataArray
        Multi-dimensional damage data with coordinates for regions, GCMs, scenarios
    regions : List[str]
        List of region identifiers
    gcms : List[str]
        List of GCM (Global Climate Model) identifiers
    scenarios : List[str]
        List of scenario identifiers
    model_type : str
        Type of damage function model to fit
    degree : int
        Polynomial degree (if model_type="polynomial")
    time_dim : str
        Name of time dimension in damage_data
    **fit_kwargs
        Additional arguments passed to fit_damage_function
        
    Returns
    -------
    pd.DataFrame
        Structured results with fitted parameters for each combination.
        Columns include: region, gcm, scenario, coefficients, r_squared, etc.
    """
    logger.info(f"Starting batch damage function fitting for {len(regions)} regions, "
                f"{len(gcms)} GCMs, {len(scenarios)} scenarios")
    
    # Validate inputs
    _validate_batch_inputs(damage_data, regions, gcms, scenarios, time_dim)
    
    # Create all combinations of dimensions
    combinations = list(product(regions, gcms, scenarios))
    total_combinations = len(combinations)
    
    logger.info(f"Processing {total_combinations} total combinations")
    
    # Process each combination
    results = []
    for i, (region, gcm, scenario) in enumerate(combinations):
        
        if i % 100 == 0:
            logger.info(f"Processing combination {i+1}/{total_combinations}")
        
        try:
            # Extract time series for this specific combination
            selection = {
                'region': region,
                'gcm': gcm, 
                'scenario': scenario
            }
            
            # Select data subset (this is pure array operation, no I/O)
            damage_vector = damage_data.sel(selection).values
            
            # Call pure fitting function (Phase 2 requirement)
            fit_result = fit_damage_function(
                damage_vector,
                model_type=model_type,
                degree=degree,
                **fit_kwargs
            )
            
            # Structure the result
            result_record = {
                'region': region,
                'gcm': gcm,
                'scenario': scenario,
                'model_type': fit_result['model_type'],
                'r_squared': fit_result['r_squared'],
                'convergence_success': fit_result['convergence_info']['success'],
                'convergence_message': fit_result['convergence_info']['message'],
                'n_points': fit_result['metadata'].get('n_points', 0)
            }
            
            # Add individual coefficients as separate columns
            coefficients = fit_result['coefficients']
            for j, coeff in enumerate(coefficients):
                result_record[f'coeff_{j}'] = coeff
            
            # Add any additional metadata
            if 'parameter_errors' in fit_result['metadata']:
                errors = fit_result['metadata']['parameter_errors']
                for j, error in enumerate(errors):
                    result_record[f'coeff_{j}_error'] = error
            
            results.append(result_record)
            
        except Exception as e:
            logger.warning(f"Error processing {region}/{gcm}/{scenario}: {str(e)}")
            
            # Create error record
            error_record = {
                'region': region,
                'gcm': gcm,
                'scenario': scenario,
                'model_type': model_type,
                'r_squared': np.nan,
                'convergence_success': False,
                'convergence_message': f'Processing error: {str(e)}',
                'n_points': 0
            }
            
            # Add NaN coefficients
            n_coeffs = degree + 1 if model_type == "polynomial" else 3
            for j in range(n_coeffs):
                error_record[f'coeff_{j}'] = np.nan
            
            results.append(error_record)
    
    # Convert to DataFrame (Phase 2 requirement: structured result)
    df_results = pd.DataFrame(results)
    
    logger.info(f"Completed batch fitting. "
                f"Success rate: {df_results['convergence_success'].mean():.2%}")
    
    return df_results


def _validate_batch_inputs(damage_data, regions, gcms, scenarios, time_dim):
    """Validate inputs for batch fitting."""
    if not isinstance(damage_data, xr.DataArray):
        raise ValueError("damage_data must be xarray.DataArray")
    
    required_dims = ['region', 'gcm', 'scenario', time_dim]
    missing_dims = [dim for dim in required_dims if dim not in damage_data.dims]
    if missing_dims:
        raise ValueError(f"Missing required dimensions: {missing_dims}")
    
    # Check that provided lists match available coordinates
    for dim_name, dim_list in [('region', regions), ('gcm', gcms), ('scenario', scenarios)]:
        available = list(damage_data[dim_name].values)
        missing = [item for item in dim_list if item not in available]
        if missing:
            logger.warning(f"Requested {dim_name} values not in data: {missing}")


def validate_fitting_inputs(
    damage_data: xr.DataArray,
    metadata: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate inputs for damage function fitting.
    
    Pure function that checks data quality and structure.
    
    Parameters
    ----------
    damage_data : xr.DataArray
        Damage data to validate
    metadata : Dict[str, Any]
        Metadata with validation requirements
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check data structure
    if not isinstance(damage_data, xr.DataArray):
        issues.append("damage_data must be xarray.DataArray")
        return False, issues
    
    # Check for required dimensions
    required_dims = metadata.get('required_dimensions', ['region', 'gcm', 'scenario', 'year'])
    missing_dims = [dim for dim in required_dims if dim not in damage_data.dims]
    if missing_dims:
        issues.append(f"Missing required dimensions: {missing_dims}")
    
    # Check data quality
    if damage_data.size == 0:
        issues.append("damage_data is empty")
    
    total_nan_fraction = np.isnan(damage_data.values).mean()
    if total_nan_fraction > 0.9:
        issues.append(f"Data is {total_nan_fraction:.1%} NaN values")
    
    # Check coordinate consistency
    for dim in damage_data.dims:
        if len(damage_data[dim]) == 0:
            issues.append(f"Dimension '{dim}' has no coordinates")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def collect_fitting_results(
    results_list: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Collect and structure fitting results from multiple runs.
    
    Pure function for result aggregation.
    
    Parameters
    ----------
    results_list : List[Dict[str, Any]]
        List of fitting results
        
    Returns
    -------
    pd.DataFrame
        Structured results
    """
    if not results_list:
        return pd.DataFrame()
    
    # Flatten nested results
    flattened_results = []
    for result in results_list:
        if isinstance(result, dict):
            flattened_results.append(result)
        elif isinstance(result, list):
            flattened_results.extend(result)
    
    return pd.DataFrame(flattened_results)


def summarize_fitting_results(
    results: pd.DataFrame
) -> Dict[str, Any]:
    """
    Summarize batch fitting results.
    
    Pure function for result analysis.
    
    Parameters
    ----------
    results : pd.DataFrame
        Fitting results from fit_damage_functions_batch
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics
    """
    if len(results) == 0:
        return {'total_fits': 0, 'success_rate': 0.0}
    
    summary = {
        'total_fits': len(results),
        'success_rate': results['convergence_success'].mean(),
        'mean_r_squared': results.loc[results['convergence_success'], 'r_squared'].mean(),
        'median_r_squared': results.loc[results['convergence_success'], 'r_squared'].median(),
        'failed_fits': (~results['convergence_success']).sum(),
        'model_types': results['model_type'].value_counts().to_dict()
    }
    
    # Add coefficient statistics if available
    coeff_cols = [col for col in results.columns if col.startswith('coeff_')]
    if coeff_cols:
        coeff_stats = {}
        for col in coeff_cols:
            valid_coeffs = results.loc[results['convergence_success'], col]
            if len(valid_coeffs) > 0:
                coeff_stats[col] = {
                    'mean': valid_coeffs.mean(),
                    'std': valid_coeffs.std(),
                    'min': valid_coeffs.min(),
                    'max': valid_coeffs.max()
                }
        summary['coefficient_statistics'] = coeff_stats
    
    return summary 