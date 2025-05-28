"""
Damage Function Fitting Algorithms
==================================

Mathematical functions for fitting damage functions to climate and economic data.

This module implements the core damage function fitting algorithms from the original DSCIM,
providing both simple polynomial fitting and the complete DSCIM methodology for relating
climate variables (temperature, sea level) to economic damages.

Key Functions:
    - fit_damage_function: Simple polynomial/exponential fitting for individual damage vectors
    - model_outputs_dscim: Complete DSCIM damage function estimation with rolling windows
    - modeler_dscim: Core regression fitting (OLS and quantile regression)

DSCIM Methodology:
    The damage function fitting follows the original DSCIM approach:
    1. Rolling window estimation (5-year windows by default)
    2. Regression of damages against climate variables using R-style formulas
    3. Support for both OLS and quantile regression
    4. Extrapolation using global consumption ratios
    5. Prediction over specified climate variable ranges

Design Principles:
    - Pure functions with no side effects or I/O operations
    - Comprehensive input validation and error handling
    - Numerical stability for edge cases (NaN values, insufficient data)
    - Consistent interfaces using pandas DataFrames and xarray Datasets
    - Full replication of original DSCIM fitting methodology

Usage:
    These functions are used by the damage_workflows module to implement the complete
    damage function calculation pipeline, from reduced damages to fitted coefficients.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Optional, Union, List
import warnings


def fit_damage_function(
    vector: np.ndarray, 
    model_type: str = "polynomial",
    degree: int = 2,
    bounds: Optional[Dict[str, tuple]] = None
) -> dict:
    """
    Fit a damage function to a 1D array of damage/mortality data.
    
    Pure function implementing Phase 2 requirement.
    No I/O, no side effects, fully testable in isolation.
    
    Parameters
    ----------
    vector : np.ndarray
        1D array of damage/mortality values
    model_type : str
        Type of model to fit ("polynomial", "exponential")
    degree : int
        Degree for polynomial models
    bounds : dict, optional
        Parameter bounds for optimization
        
    Returns
    -------
    dict
        Fitted parameters with keys:
        - 'coefficients': np.ndarray of fitted coefficients
        - 'r_squared': float, goodness of fit
        - 'model_type': str, type of model fitted
        - 'convergence_info': dict with success status and messages
        - 'metadata': dict with fitting diagnostics
    """
    # Input validation
    if not isinstance(vector, np.ndarray) or vector.ndim != 1:
        raise ValueError("Input must be 1D numpy array")
    if len(vector) == 0:
        raise ValueError("Input vector cannot be empty")
    
    # Handle all-NaN case
    if np.all(np.isnan(vector)):
        return _create_nan_result(model_type, degree)
    
    # Clean data by removing NaN values
    clean_vector = vector[~np.isnan(vector)]
    if len(clean_vector) < 2:
        return _create_insufficient_data_result(model_type, degree)
    
    # Dispatch to specific fitting function
    if model_type == "polynomial":
        return fit_polynomial_model(clean_vector, degree)
    elif model_type == "exponential":
        return fit_exponential_model(clean_vector, bounds)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def model_outputs_dscim(
    damage_function: pd.DataFrame,
    extrapolation_type: str,
    formula: str,
    year_range: range,
    year_start_pred: int,
    quantiles: Optional[List[float]],
    global_c: Optional[xr.DataArray] = None,
    min_anomaly: float = 0,
    max_anomaly: float = 20,
    step_anomaly: float = 0.2,
    min_gmsl: float = 0,
    max_gmsl: float = 300,
    step_gmsl: float = 3,
    type_estimation: str = "ols",
) -> Dict[str, xr.Dataset]:
    """
    Replicate the original DSCIM model_outputs function.
    
    Estimate damage function coefficients and predictions using passed formula.
    This model is estimated for each year in our timeframe (2010 to 2099) and
    using a rolling window across time (5-yr by default).
    
    Parameters
    ----------
    damage_function : pd.DataFrame
        A global damage function with columns: damages, year, and climate variables
    extrapolation_type : str
        Type of extrapolation: `global_c_ratio`
    formula : str
        R-style formula for fitting (e.g., "damages ~ -1 + anomaly + np.power(anomaly, 2)")
    year_range : range
        Range of years to estimate over
    year_start_pred : int
        Start of extrapolation
    quantiles : List[float], optional
        Quantiles for quantile regression
    global_c : xr.DataArray, optional
        Array with global consumption extrapolated to 2300
    min_anomaly : float
        Minimum temperature anomaly for predictions
    max_anomaly : float
        Maximum temperature anomaly for predictions
    step_anomaly : float
        Step size for temperature anomaly predictions
    min_gmsl : float
        Minimum sea level for predictions
    max_gmsl : float
        Maximum sea level for predictions
    step_gmsl : float
        Step size for sea level predictions
    type_estimation : str
        Type of model use for damage function fitting: `ols`, `quantreg`
        
    Returns
    -------
    Dict[str, xr.Dataset]
        Dictionary with keys:
        - 'parameters': Fitted coefficients by year
        - 'preds': Predictions from fitted model
    """
    # Set year of prediction for global C extrapolation
    fix_global_c = year_start_pred - 1
    
    # Set exogenous variables for predictions
    gmsl = np.arange(min_gmsl, max_gmsl, step_gmsl)
    temps = np.arange(min_anomaly, max_anomaly, step_anomaly)
    
    # Determine which variables are needed based on damage_function columns
    has_anomaly = "anomaly" in damage_function.columns
    has_gmsl = "gmsl" in damage_function.columns
    
    if has_anomaly and has_gmsl:
        from itertools import product
        exog_X = pd.DataFrame(product(temps, gmsl))
        exog = dict(anomaly=exog_X.values[:, 0], gmsl=exog_X.values[:, 1])
    elif has_anomaly:
        exog = dict(anomaly=temps)
    elif has_gmsl:
        exog = dict(gmsl=gmsl)
    else:
        raise ValueError("Independent variables (anomaly or gmsl) not found in damage_function")
    
    # Rolling window estimation (5-yr)
    list_params, list_y_hats = [], []
    
    for year in year_range:
        time_window = range(year - 2, year + 3)
        df = damage_function[damage_function.year.isin(time_window)]
        
        if len(df) == 0:
            # No data for this time window, skip
            continue
        
        params, y_hat = modeler_dscim(
            df=df,
            formula=formula,
            type_estimation=type_estimation,
            exog=exog,
            quantiles=quantiles,
        )
        
        params, y_hat = params.assign(year=year), y_hat.assign(year=year)
        list_params.append(params)
        list_y_hats.append(y_hat)
    
    if not list_params:
        raise ValueError("No damage functions could be fitted for any year")
    
    # Concatenate results
    param_df = pd.concat(list_params)
    y_hat_df = pd.concat(list_y_hats)
    
    if extrapolation_type == "global_c_ratio":
        # Convert to xarray immediately
        index = ["year", "q"] if type_estimation == "quantreg" else ["year"]
        y_hat_df = y_hat_df.set_index(
            [i for i in y_hat_df.columns if "y_hat" not in i]
        ).to_xarray()
        param_df = param_df.set_index(index).to_xarray()
        
        if global_c is not None:
            # Calculate global consumption ratios to fixed year
            global_c_factors = (
                global_c.sel(year=slice(fix_global_c + 1, None))
                / global_c.sel(year=fix_global_c)
            ).squeeze()
            
            # Extrapolate by multiplying fixed params by ratios
            extrap_preds = y_hat_df.sel(year=fix_global_c) * global_c_factors
            extrap_params = param_df.sel(year=fix_global_c) * global_c_factors
            
            # Concatenate extrapolation and pre-2100
            preds = xr.concat([y_hat_df, extrap_preds], dim="year")
            parameters = xr.concat([param_df, extrap_params], dim="year")
        else:
            # No extrapolation, just use fitted values
            preds = y_hat_df
            parameters = param_df
    else:
        # Convert to xarray without extrapolation
        index = ["year", "q"] if type_estimation == "quantreg" else ["year"]
        preds = y_hat_df.set_index(
            [i for i in y_hat_df.columns if "y_hat" not in i]
        ).to_xarray()
        parameters = param_df.set_index(index).to_xarray()
    
    # Return all results
    res = {
        "parameters": parameters,
        "preds": preds,
    }
    
    return res


def modeler_dscim(
    df: pd.DataFrame, 
    formula: str, 
    type_estimation: str, 
    exog: Dict[str, np.ndarray], 
    quantiles: Optional[List[float]] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replicate the original DSCIM modeler function.
    
    Wrapper function for statsmodels functions (OLS and quantiles).
    
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of prepared damage function with standardized column names:
        - 'damages': damage values
        - 'anomaly': temperature anomalies (if used)
        - 'gmsl': sea level (if used)
    formula : str
        R-like formula to start fitting (e.g., "damages ~ anomaly + np.power(anomaly, 2)")
    type_estimation : str
        Model to use: 'ols' or 'quantreg'
    exog : Dict[str, np.ndarray]
        Predictors used to generate predicted damage function fit
    quantiles : List[float], optional
        List of quantile regressions to be run
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (coefficients_df, predictions_df)
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels is required for damage function fitting")
    
    if len(df) == 0:
        raise ValueError("Empty dataframe provided to modeler")
    
    # Validate required columns
    if 'damages' not in df.columns:
        raise ValueError("'damages' column required in dataframe")
    
    # Check for NaN values in damages
    if df['damages'].isna().all():
        raise ValueError("All damage values are NaN")
    
    if type_estimation == "ols":
        # Ordinary Least Squares
        try:
            model = smf.ols(formula=formula, data=df)
            fitted_model = model.fit()
            
            # Extract coefficients
            coeffs_df = pd.DataFrame({
                'coefficient': fitted_model.params.index,
                'value': fitted_model.params.values
            })
            
            # Pivot to have coefficients as columns
            coeffs_df = coeffs_df.set_index('coefficient').T
            
            # Generate predictions using exog
            exog_df = pd.DataFrame(exog)
            predictions = fitted_model.predict(exog=exog_df)
            
            # Create predictions dataframe
            preds_df = exog_df.copy()
            preds_df['y_hat'] = predictions
            
            return coeffs_df, preds_df
            
        except Exception as e:
            # Return NaN results if fitting fails
            coeffs_df = pd.DataFrame({col: [np.nan] for col in ['Intercept', 'anomaly', 'np.power(anomaly, 2)']})
            preds_df = pd.DataFrame(exog)
            preds_df['y_hat'] = np.nan
            return coeffs_df, preds_df
    
    elif type_estimation == "quantreg":
        # Quantile regression
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        try:
            import statsmodels.regression.quantile_regression as qr
            
            all_coeffs = []
            all_preds = []
            
            for q in quantiles:
                model = smf.quantreg(formula=formula, data=df)
                fitted_model = model.fit(q=q)
                
                # Extract coefficients
                coeffs_df = pd.DataFrame({
                    'coefficient': fitted_model.params.index,
                    'value': fitted_model.params.values,
                    'q': q
                })
                all_coeffs.append(coeffs_df)
                
                # Generate predictions
                exog_df = pd.DataFrame(exog)
                predictions = fitted_model.predict(exog=exog_df)
                
                preds_df = exog_df.copy()
                preds_df['y_hat'] = predictions
                preds_df['q'] = q
                all_preds.append(preds_df)
            
            # Combine all quantiles
            coeffs_combined = pd.concat(all_coeffs, ignore_index=True)
            preds_combined = pd.concat(all_preds, ignore_index=True)
            
            # Pivot coefficients to have them as columns
            coeffs_pivot = coeffs_combined.pivot(index='q', columns='coefficient', values='value')
            
            return coeffs_pivot, preds_combined
            
        except Exception as e:
            # Return NaN results if fitting fails
            coeffs_df = pd.DataFrame({
                'q': quantiles,
                'Intercept': np.nan,
                'anomaly': np.nan,
                'np.power(anomaly, 2)': np.nan
            }).set_index('q')
            
            preds_df = pd.DataFrame(exog)
            preds_df['y_hat'] = np.nan
            preds_df['q'] = quantiles[0]  # Just use first quantile
            
            return coeffs_df, preds_df
    
    else:
        raise ValueError(f"Unknown type_estimation: {type_estimation}")


def fit_polynomial_model(vector: np.ndarray, degree: int) -> dict:
    """
    Fit polynomial model using least squares.
    
    Pure function with no side effects.
    
    Parameters
    ----------
    vector : np.ndarray
        1D array of clean data (no NaNs)
    degree : int
        Polynomial degree
        
    Returns
    -------
    dict
        Fitting results
    """
    x = np.arange(len(vector))
    
    try:
        # Fit polynomial using numpy
        coeffs = np.polyfit(x, vector, degree)
        
        # Calculate goodness of fit
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((vector - y_pred) ** 2)
        ss_tot = np.sum((vector - np.mean(vector)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Calculate condition number for numerical stability assessment
        vander_matrix = np.vander(x, degree + 1)
        condition_number = np.linalg.cond(vander_matrix)
        
        return {
            'coefficients': coeffs,
            'r_squared': r_squared,
            'model_type': f'polynomial_degree_{degree}',
            'convergence_info': {
                'success': True,
                'message': 'Polynomial fit converged',
                'iterations': None
            },
            'metadata': {
                'n_points': len(vector),
                'condition_number': condition_number,
                'residual_std': np.sqrt(ss_res / (len(vector) - degree - 1)) if len(vector) > degree + 1 else np.nan
            }
        }
        
    except np.linalg.LinAlgError as e:
        return {
            'coefficients': np.full(degree + 1, np.nan),
            'r_squared': np.nan,
            'model_type': f'polynomial_degree_{degree}',
            'convergence_info': {
                'success': False,
                'message': f'Polynomial fit failed: {str(e)}',
                'iterations': None
            },
            'metadata': {'n_points': len(vector), 'error': str(e)}
        }


def fit_exponential_model(vector: np.ndarray, bounds: Optional[dict]) -> dict:
    """
    Fit exponential model using nonlinear optimization.
    
    Pure function with no side effects.
    Model: f(x) = a * exp(b * x) + c
    
    Parameters
    ----------
    vector : np.ndarray
        1D array of clean data (no NaNs)
    bounds : dict, optional
        Parameter bounds for optimization
        
    Returns
    -------
    dict
        Fitting results
    """
    try:
        from scipy import optimize
    except ImportError:
        return {
            'coefficients': np.full(3, np.nan),
            'r_squared': np.nan,
            'model_type': 'exponential',
            'convergence_info': {
                'success': False,
                'message': 'SciPy not available for exponential fitting',
                'iterations': None
            },
            'metadata': {'n_points': len(vector), 'error': 'SciPy import failed'}
        }
    
    x = np.arange(len(vector))
    
    def exp_func(x, a, b, c):
        """Exponential function: a * exp(b * x) + c"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return a * np.exp(b * x) + c
    
    try:
        # Initial parameter guess
        y_range = np.max(vector) - np.min(vector)
        p0 = [
            y_range / 2,     # a: scale parameter
            0.1,             # b: growth rate
            np.min(vector)   # c: offset
        ]
        
        # Set parameter bounds
        if bounds:
            bounds_tuple = (
                [bounds.get('a', (-np.inf,))[0], 
                 bounds.get('b', (-np.inf,))[0], 
                 bounds.get('c', (-np.inf,))[0]],
                [bounds.get('a', (np.inf,))[1], 
                 bounds.get('b', (np.inf,))[1], 
                 bounds.get('c', (np.inf,))[1]]
            )
        else:
            # Default reasonable bounds
            bounds_tuple = (
                [-10 * y_range, -2.0, -10 * y_range],
                [10 * y_range, 2.0, 10 * y_range]
            )
        
        # Fit exponential model
        popt, pcov = optimize.curve_fit(
            exp_func, x, vector, 
            p0=p0, bounds=bounds_tuple,
            maxfev=1000
        )
        
        # Calculate goodness of fit
        y_pred = exp_func(x, *popt)
        ss_res = np.sum((vector - y_pred) ** 2)
        ss_tot = np.sum((vector - np.mean(vector)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Calculate parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(3, np.nan)
        
        return {
            'coefficients': popt,
            'r_squared': r_squared,
            'model_type': 'exponential',
            'convergence_info': {
                'success': True,
                'message': 'Exponential fit converged',
                'iterations': None
            },
            'metadata': {
                'n_points': len(vector),
                'parameter_covariance': pcov,
                'parameter_errors': param_errors,
                'residual_std': np.sqrt(ss_res / (len(vector) - 3)) if len(vector) > 3 else np.nan
            }
        }
        
    except Exception as e:
        return {
            'coefficients': np.full(3, np.nan),
            'r_squared': np.nan,
            'model_type': 'exponential',
            'convergence_info': {
                'success': False,
                'message': f'Exponential fit failed: {str(e)}',
                'iterations': None
            },
            'metadata': {'n_points': len(vector), 'error': str(e)}
        }


def _create_nan_result(model_type: str, degree: int) -> dict:
    """Create result dictionary for all-NaN input."""
    n_coeffs = degree + 1 if model_type == "polynomial" else 3
    return {
        'coefficients': np.full(n_coeffs, np.nan),
        'r_squared': np.nan,
        'model_type': model_type,
        'convergence_info': {
            'success': False,
            'message': 'All input values are NaN',
            'iterations': None
        },
        'metadata': {'n_points': 0, 'error': 'All NaN input'}
    }


def _create_insufficient_data_result(model_type: str, degree: int) -> dict:
    """Create result dictionary for insufficient data."""
    n_coeffs = degree + 1 if model_type == "polynomial" else 3
    return {
        'coefficients': np.full(n_coeffs, np.nan),
        'r_squared': np.nan,
        'model_type': model_type,
        'convergence_info': {
            'success': False,
            'message': 'Insufficient data points for fitting',
            'iterations': None
        },
        'metadata': {'n_points': 0, 'error': 'Insufficient data'}
    }


def validate_damage_vector(vector: np.ndarray) -> tuple[bool, list[str]]:
    """
    Validate damage vector for fitting.
    
    Pure function that checks data quality.
    
    Parameters
    ----------
    vector : np.ndarray
        Input damage vector
        
    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check basic properties
    if not isinstance(vector, np.ndarray):
        issues.append("Input must be numpy array")
        return False, issues
    
    if vector.ndim != 1:
        issues.append("Input must be 1-dimensional")
    
    if len(vector) == 0:
        issues.append("Input vector is empty")
    
    # Check data quality
    if len(vector) > 0:
        nan_fraction = np.isnan(vector).mean()
        if nan_fraction == 1.0:
            issues.append("All values are NaN")
        elif nan_fraction > 0.8:
            issues.append(f"High fraction of NaN values: {nan_fraction:.1%}")
        
        # Check for infinite values
        inf_count = np.isinf(vector).sum()
        if inf_count > 0:
            issues.append(f"Contains {inf_count} infinite values")
        
        # Check for constant values (after removing NaNs)
        clean_vector = vector[~np.isnan(vector)]
        if len(clean_vector) > 1:
            if np.all(clean_vector == clean_vector[0]):
                issues.append("All non-NaN values are identical")
    
    is_valid = len(issues) == 0
    return is_valid, issues 