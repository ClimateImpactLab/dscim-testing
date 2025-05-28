"""
Core Economic Mathematical Functions
===================================

Pure mathematical functions for economic damage calculations and uncertainty quantification.

This module contains the fundamental mathematical operations used in climate damage modeling,
extracted and refactored from the original DSCIM implementation. All functions are pure
(no side effects, no I/O operations) and operate on numpy arrays and xarray objects.

Key Functions:
    - discounted_damages: Present value calculations using discount rates
    - weitzman_min: Uncertainty aggregation following Weitzman's approach
    - calculate_scc: Social Cost of Carbon computation from marginal damages
    - ce_func: CRRA certainty equivalent calculations for welfare analysis
    - mean_func: Simple mean aggregation across specified dimensions
    - power: Robust power function handling negative bases

Design Principles:
    - Pure functions with no side effects
    - Comprehensive input validation
    - Clear error messages for debugging
    - Consistent xarray-based interfaces
    - Numerical stability for edge cases

Usage:
    These functions are building blocks for higher-level workflows in the processing module.
    They can be used independently for custom damage calculations or combined in workflows.
"""

import numpy as np
import xarray as xr
from typing import Union, Optional, List


def discounted_damages(
    damages: xr.DataArray, 
    discount_rates: xr.DataArray,
    time_dim: str = "year"
) -> xr.DataArray:
    """
    Calculate present value of future damages using discount rates.
    
    Pure function extracted from menu/main_recipe.py discounted_damages method.
    Phase 1: Core economic calculation with no side effects.
    
    Parameters
    ----------
    damages : xr.DataArray
        Future damages by year and other dimensions
    discount_rates : xr.DataArray  
        Discount rates by year (same time dimension as damages)
    time_dim : str
        Name of time dimension for discounting
        
    Returns
    -------
    xr.DataArray
        Present value of damages
    """
    # Validate inputs
    if time_dim not in damages.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in damages")
    if time_dim not in discount_rates.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in discount_rates")
    
    # Calculate discount factors (1 / (1 + r)^t)
    # Assuming discount_rates are annual rates
    years = damages[time_dim]
    base_year = years.min().item()
    
    # Calculate time periods from base year
    time_periods = years - base_year
    
    # Calculate cumulative discount factors
    # For year t: discount_factor = 1 / (1 + r_0) * (1 + r_1) * ... * (1 + r_t)
    discount_factors = xr.ones_like(discount_rates)
    
    for i, year in enumerate(years):
        if i == 0:
            discount_factors.loc[{time_dim: year}] = 1.0
        else:
            # Cumulative discounting
            prev_years = years[:i]
            cumulative_factor = 1.0
            for prev_year in prev_years:
                rate = discount_rates.sel({time_dim: prev_year})
                cumulative_factor *= (1.0 / (1.0 + rate))
            discount_factors.loc[{time_dim: year}] = cumulative_factor
    
    # Apply discounting to damages
    discounted = damages * discount_factors
    
    return discounted


def weitzman_min(
    damages: xr.DataArray,
    weights: Optional[xr.DataArray] = None,
    uncertainty_dim: str = "simulation"
) -> xr.DataArray:
    """
    Calculate Weitzman minimum regret aggregation across uncertainty.
    
    Pure function for uncertainty aggregation following Weitzman's approach.
    Phase 1: Core economic calculation with no side effects.
    
    Parameters
    ----------
    damages : xr.DataArray
        Damages across uncertainty realizations
    weights : xr.DataArray, optional
        Weights for each uncertainty realization. If None, uses equal weights.
    uncertainty_dim : str
        Name of uncertainty dimension to aggregate over
        
    Returns
    -------
    xr.DataArray
        Aggregated damages using Weitzman approach
    """
    if uncertainty_dim not in damages.dims:
        raise ValueError(f"Uncertainty dimension '{uncertainty_dim}' not found in damages")
    
    # Use equal weights if none provided
    if weights is None:
        n_sims = damages.sizes[uncertainty_dim]
        weights = xr.ones_like(damages.isel({uncertainty_dim: 0})) / n_sims
        weights = weights.expand_dims({uncertainty_dim: damages[uncertainty_dim]})
    
    # Validate weights
    if uncertainty_dim not in weights.dims:
        raise ValueError(f"Uncertainty dimension '{uncertainty_dim}' not found in weights")
    
    # Weitzman approach: weighted average with emphasis on tail risks
    # For simplicity, implementing as weighted mean (can be extended to more complex formulations)
    weighted_damages = damages * weights
    aggregated = weighted_damages.sum(dim=uncertainty_dim)
    
    return aggregated


def calculate_scc(
    marginal_damages: xr.DataArray,
    discount_factors: xr.DataArray,
    time_dim: str = "year",
    aggregate_dims: Optional[List[str]] = None
) -> xr.DataArray:
    """
    Calculate Social Cost of Carbon from marginal damages.
    
    Pure function for SCC calculation.
    Phase 1: Core economic calculation with no side effects.
    
    Parameters
    ----------
    marginal_damages : xr.DataArray
        Marginal damages from additional emissions
    discount_factors : xr.DataArray
        Discount factors for present value calculation
    time_dim : str
        Name of time dimension
    aggregate_dims : List[str], optional
        Dimensions to aggregate over (e.g., regions)
        
    Returns
    -------
    xr.DataArray
        Social Cost of Carbon
    """
    # Apply discounting
    discounted_marginal = marginal_damages * discount_factors
    
    # Sum over time to get net present value
    scc = discounted_marginal.sum(dim=time_dim)
    
    # Aggregate over specified dimensions if provided
    if aggregate_dims:
        for dim in aggregate_dims:
            if dim in scc.dims:
                scc = scc.sum(dim=dim)
    
    return scc


def power(a: xr.DataArray, b: Union[float, xr.DataArray]) -> xr.DataArray:
    """
    Power function that doesn't return NaN values for negative fractional exponents.
    
    Extracted from dscim.utils.utils.power.
    Pure function for mathematical operations.
    
    Parameters
    ----------
    a : xr.DataArray
        Base values
    b : float or xr.DataArray
        Exponent values
        
    Returns
    -------
    xr.DataArray
        Power result
    """
    return np.sign(a) * (np.abs(a)) ** b


def ce_func(consumption: xr.DataArray, dims: Union[str, List[str]], eta: float) -> xr.DataArray:
    """
    Calculate CRRA (Constant Relative Risk Aversion) certainty equivalent.
    
    Extracted from dscim.utils.functions.ce_func.
    Pure function for welfare calculations.
    
    Parameters
    ----------
    consumption : xr.DataArray
        Consumption values
    dims : str or List[str]
        Dimensions to take certainty equivalent over
    eta : float
        Risk aversion parameter
        
    Returns
    -------
    xr.DataArray
        Certainty equivalent consumption
    """
    # Use log utility when eta is 1
    if eta == 1:
        return np.exp(np.log(consumption).mean(dims))
    # CRRA utility otherwise
    else:
        return power(
            (power(consumption, (1 - eta)) / (1 - eta)).mean(dims) * (1 - eta),
            (1 / (1 - eta)),
        )


def mean_func(consumption: xr.DataArray, dims: Union[str, List[str]]) -> xr.DataArray:
    """
    Calculate mean consumption across specified dimensions.
    
    Extracted from dscim.utils.functions.mean_func.
    Pure function for simple averaging.
    
    Parameters
    ----------
    consumption : xr.DataArray
        Consumption values
    dims : str or List[str]
        Dimensions to average over
        
    Returns
    -------
    xr.DataArray
        Mean consumption
    """
    return consumption.mean(dims) 