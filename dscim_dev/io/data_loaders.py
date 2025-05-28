"""
Data Loading Functions
=====================

Input/output operations for loading climate, economic, and damage data.

This module provides comprehensive data loading capabilities for the DSCIM pipeline,
handling various data formats (Zarr, NetCDF, CSV) and ensuring compatibility
between different data sources. It includes both real data loading from the
original DSCIM data files and synthetic data generation for testing.

Key Functions:
    - load_real_dummy_data: Load actual DSCIM dummy data files
    - load_dummy_climate_data: Load climate data (temperature, sea level)
    - load_reduced_damages: Load pre-computed reduced damages
    - load_zarr_with_fallback: Robust Zarr dataset loading with error handling
    - create_synthetic_dummy_data: Generate synthetic data for testing

Data Compatibility:
    The module handles multiple data formats and structures:
    - Dataset vs DataArray formats for damage data
    - Coordinate alignment between damage, economic, and climate data
    - Automatic adaptation of real data to expected processing formats
    - Comprehensive validation and error handling

DSCIM Integration:
    These functions load data in the exact format expected by the original DSCIM,
    ensuring numerical consistency while providing a clean, modular interface.
    They support both the damage reduction and damage function fitting phases
    of the DSCIM pipeline.

Design Principles:
    - Pure I/O functions with no computational logic
    - Comprehensive error handling and fallback mechanisms
    - Flexible data format support with automatic adaptation
    - Clear separation between data loading and data processing
    - Extensive logging for debugging and validation

Usage:
    These functions are typically called at the beginning of DSCIM workflows
    to load the necessary input data for damage calculations and function fitting.
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import logging
import yaml

logger = logging.getLogger(__name__)


def load_dummy_data(
    data_dir: str = "dummy_data",
    damage_source: str = "sectoral",
    economic_source: str = "econ"
) -> Tuple[Union[xr.DataArray, xr.Dataset], xr.DataArray]:
    """
    Load dummy data for testing the refactored pipeline.
    
    Pure I/O function - only loads data, no processing.
    
    Parameters
    ----------
    data_dir : str
        Directory containing dummy data files
    damage_source : str
        Subdirectory for damage data ("sectoral")
    economic_source : str
        Subdirectory for economic data ("econ")
        
    Returns
    -------
    Tuple[Union[xr.DataArray, xr.Dataset], xr.DataArray]
        (damage_data, economic_data) - damage_data can be Dataset with multiple variables
        or DataArray with variable dimension
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} not found, creating synthetic data")
        return create_synthetic_dummy_data()
    
    try:
        # Load damage data from sectoral directory
        damage_dir = data_path / damage_source
        if damage_dir.exists():
            # Look for Zarr directories in sectoral directory
            zarr_dirs = [d for d in damage_dir.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
            if zarr_dirs:
                # Load the first available damage dataset
                damage_path = zarr_dirs[0]
                logger.info(f"Found {len(zarr_dirs)} Zarr directories in {damage_dir}")
                logger.info(f"Loading damage data from {damage_path}")
                
                # Use the new loading function
                damage_data = load_zarr_with_fallback(damage_path)
                logger.info(f"Loaded damage data successfully")
            else:
                logger.warning(f"No Zarr directories found in {damage_dir}, creating synthetic")
                damage_data = create_synthetic_damage_dataset()
        else:
            logger.warning(f"Damage directory {damage_dir} not found, creating synthetic")
            damage_data = create_synthetic_damage_dataset()
        
        # Load economic data from econ directory
        econ_dir = data_path / economic_source
        if econ_dir.exists():
            # Look for Zarr directories in econ directory
            zarr_dirs = [d for d in econ_dir.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
            if zarr_dirs:
                # Load the first available economic dataset
                econ_path = zarr_dirs[0]
                logger.info(f"Found {len(zarr_dirs)} Zarr directories in {econ_dir}")
                logger.info(f"Loading economic data from {econ_path}")
                
                # Use the new loading function with preference for GDP variables
                economic_data = load_zarr_with_fallback(econ_path, preferred_var='gdppc')
                logger.info(f"Loaded economic data successfully")
            else:
                logger.warning(f"No Zarr directories found in {econ_dir}, creating synthetic")
                economic_data = create_synthetic_economic_data()
        else:
            logger.warning(f"Economic directory {econ_dir} not found, creating synthetic")
            economic_data = create_synthetic_economic_data()
        
        # Validate that we have compatible data structures
        logger.info(f"Damage data dimensions: {list(damage_data.dims)}")
        logger.info(f"Economic data dimensions: {list(economic_data.dims)}")
        
        # Adapt real data to expected structure for processing functions
        try:
            damage_data, economic_data = create_compatible_dummy_data_from_real(
                damage_data, economic_data
            )
            logger.info("Successfully adapted real data to compatible structure")
        except Exception as adapt_error:
            logger.warning(f"Error adapting real data structure: {adapt_error}")
            logger.info("Falling back to synthetic data")
            return create_synthetic_dummy_data()
        
        return damage_data, economic_data
        
    except Exception as e:
        logger.error(f"Error loading dummy data: {e}")
        logger.info("Falling back to synthetic data generation")
        return create_synthetic_dummy_data()


def load_netcdf_data(
    file_path: str,
    variables: Optional[list] = None
) -> xr.Dataset:
    """
    Load NetCDF data file.
    
    Pure I/O function for loading NetCDF files.
    
    Parameters
    ----------
    file_path : str
        Path to NetCDF file
    variables : list, optional
        Specific variables to load
        
    Returns
    -------
    xr.Dataset
        Loaded dataset
    """
    try:
        if variables:
            dataset = xr.open_dataset(file_path)[variables]
        else:
            dataset = xr.open_dataset(file_path)
        
        logger.info(f"Loaded NetCDF data from {file_path}")
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading NetCDF file {file_path}: {e}")
        raise


def create_synthetic_dummy_data() -> Tuple[xr.Dataset, xr.DataArray]:
    """
    Create synthetic dummy data for testing when real data is not available.
    
    Returns
    -------
    Tuple[xr.Dataset, xr.DataArray]
        (damage_dataset, economic_data) - damage_dataset has separate histclim_dummy and delta_dummy variables
    """
    logger.info("Creating synthetic dummy data for testing")
    
    # Create synthetic damage data as Dataset
    damage_dataset = create_synthetic_damage_dataset()
    
    # Create synthetic economic data
    economic_data = create_synthetic_economic_data()
    
    return damage_dataset, economic_data


def create_synthetic_damage_dataset() -> xr.Dataset:
    """Create synthetic damage dataset with separate histclim_dummy and delta_dummy variables."""
    
    # Define dimensions
    regions = ['USA', 'CHN', 'EUR', 'IND', 'BRA']
    gcms = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR']
    scenarios = ['ssp126', 'ssp245', 'ssp585']
    years = np.arange(2020, 2101)
    batches = np.arange(10)  # Monte Carlo batches
    
    # Create coordinate arrays
    coords = {
        'region': regions,
        'gcm': gcms,
        'scenario': scenarios,
        'year': years,
        'batch': batches
    }
    
    # Generate synthetic damage data with realistic patterns
    np.random.seed(42)  # For reproducibility
    
    # Base damage that increases with time and scenario severity
    base_damage = np.zeros((len(regions), len(gcms), len(scenarios), len(years), len(batches)))
    
    for i, region in enumerate(regions):
        for j, gcm in enumerate(gcms):
            for k, scenario in enumerate(scenarios):
                # Scenario severity factor
                scenario_factor = {'ssp126': 1.0, 'ssp245': 2.0, 'ssp585': 4.0}[scenario]
                
                # Regional vulnerability factor
                region_factor = {'USA': 1.0, 'CHN': 1.2, 'EUR': 0.8, 'IND': 1.5, 'BRA': 1.3}[region]
                
                for l, year in enumerate(years):
                    # Time trend (damages increase over time)
                    time_factor = 1 + 0.02 * (year - 2020)  # 2% annual increase
                    
                    for m in range(len(batches)):
                        # Add stochastic variation
                        noise = np.random.normal(0, 0.1)
                        
                        # Combine factors
                        damage_value = (
                            100 * scenario_factor * region_factor * time_factor * (1 + noise)
                        )
                        
                        base_damage[i, j, k, l, m] = max(0, damage_value)
    
    # Create histclim and delta components as separate variables
    histclim_data = base_damage * 0.3  # Historical climate component
    delta_data = base_damage * 0.7     # Climate change component
    
    # Create Dataset with separate variables
    damage_dataset = xr.Dataset(
        {
            'histclim_dummy': (
                ['region', 'gcm', 'scenario', 'year', 'batch'], 
                histclim_data
            ),
            'delta_dummy': (
                ['region', 'gcm', 'scenario', 'year', 'batch'], 
                delta_data
            )
        },
        coords=coords
    )
    
    # Add metadata
    damage_dataset.attrs.update({
        'title': 'Synthetic Damage Dataset for DSCIM Testing',
        'units': 'billion USD',
        'description': 'Synthetic economic damage data with separate histclim_dummy and delta_dummy variables',
        'created_by': 'dscim_dev.io.data_loaders.create_synthetic_damage_dataset'
    })
    
    return damage_dataset


def create_synthetic_economic_data() -> xr.DataArray:
    """Create synthetic economic data (GDP per capita)."""
    
    # Define dimensions (matching damage data)
    regions = ['USA', 'CHN', 'EUR', 'IND', 'BRA']
    years = np.arange(2020, 2101)
    
    coords = {
        'region': regions,
        'year': years
    }
    
    # Generate synthetic GDP per capita data
    np.random.seed(123)  # Different seed for economic data
    
    gdppc_data = np.zeros((len(regions), len(years)))
    
    # Base GDP per capita by region (in thousands USD)
    base_gdppc = {
        'USA': 65.0,
        'CHN': 12.0, 
        'EUR': 45.0,
        'IND': 2.5,
        'BRA': 9.0
    }
    
    # Growth rates by region (annual %)
    growth_rates = {
        'USA': 0.02,
        'CHN': 0.04,
        'EUR': 0.015,
        'IND': 0.05,
        'BRA': 0.025
    }
    
    for i, region in enumerate(regions):
        base_value = base_gdppc[region]
        growth_rate = growth_rates[region]
        
        for j, year in enumerate(years):
            # Compound growth with some noise
            years_elapsed = year - 2020
            noise = np.random.normal(0, 0.05)  # 5% noise
            
            gdppc_value = base_value * (1 + growth_rate) ** years_elapsed * (1 + noise)
            gdppc_data[i, j] = max(1.0, gdppc_value)  # Minimum 1k USD
    
    # Create DataArray
    economic_data = xr.DataArray(
        gdppc_data,
        coords=coords,
        dims=['region', 'year'],
        name='gdppc'
    )
    
    # Add metadata
    economic_data.attrs.update({
        'title': 'Synthetic Economic Data for DSCIM Testing',
        'units': 'thousand USD per capita',
        'description': 'Synthetic GDP per capita data with regional growth patterns',
        'created_by': 'dscim_dev.io.data_loaders.create_synthetic_economic_data'
    })
    
    return economic_data


def validate_data_structure(
    data: xr.DataArray,
    expected_dims: list,
    expected_vars: Optional[list] = None
) -> Tuple[bool, list]:
    """
    Validate loaded data structure.
    
    Pure validation function.
    
    Parameters
    ----------
    data : xr.DataArray
        Data to validate
    expected_dims : list
        Expected dimension names
    expected_vars : list, optional
        Expected variable names
        
    Returns
    -------
    Tuple[bool, list]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check dimensions
    missing_dims = set(expected_dims) - set(data.dims)
    if missing_dims:
        issues.append(f"Missing dimensions: {missing_dims}")
    
    # Check variables if specified
    if expected_vars:
        if hasattr(data, 'variable') and 'variable' in data.dims:
            missing_vars = set(expected_vars) - set(data.variable.values)
            if missing_vars:
                issues.append(f"Missing variables: {missing_vars}")
        else:
            issues.append("Data does not have variable dimension")
    
    # Check for empty data
    if data.size == 0:
        issues.append("Data is empty")
    
    # Check for all-NaN data
    if np.all(np.isnan(data.values)):
        issues.append("All data values are NaN")
    
    is_valid = len(issues) == 0
    return is_valid, issues 


def adapt_real_data_structure(
    damage_data: xr.DataArray,
    economic_data: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Adapt real data structure to match expected format for processing functions.
    
    The real dummy data might have different dimension names and structures
    than our synthetic data. This function standardizes the format.
    
    Parameters
    ----------
    damage_data : xr.DataArray
        Raw damage data from files
    economic_data : xr.DataArray
        Raw economic data from files
        
    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        (adapted_damage_data, adapted_economic_data)
    """
    logger.info("Adapting real data structure to expected format...")
    
    # Adapt damage data
    adapted_damage = damage_data.copy()
    
    # Rename common dimension variations to standard names
    dim_mapping = {
        # Common variations for region dimension
        'region': 'region',
        'regions': 'region', 
        'iso': 'region',
        'country': 'region',
        
        # Common variations for GCM dimension
        'gcm': 'gcm',
        'model': 'gcm',
        'climate_model': 'gcm',
        
        # Common variations for scenario dimension  
        'scenario': 'scenario',
        'ssp': 'scenario',
        'rcp': 'scenario',
        
        # Common variations for year dimension
        'year': 'year',
        'time': 'year',
        'years': 'year',
        
        # Common variations for batch/simulation dimension
        'batch': 'batch',
        'simulation': 'batch',
        'sim': 'batch',
        'monte_carlo': 'batch',
        'mc': 'batch'
    }
    
    # Rename dimensions in damage data
    for old_dim, new_dim in dim_mapping.items():
        if old_dim in adapted_damage.dims and old_dim != new_dim:
            adapted_damage = adapted_damage.rename({old_dim: new_dim})
            logger.info(f"Renamed dimension '{old_dim}' to '{new_dim}' in damage data")
    
    # If damage data doesn't have a variable dimension, create one
    if 'variable' not in adapted_damage.dims:
        # Assume this is delta (climate change damages) if no variable dimension
        adapted_damage = adapted_damage.expand_dims('variable')
        adapted_damage = adapted_damage.assign_coords(variable=['delta'])
        logger.info("Added 'variable' dimension with 'delta' to damage data")
    
    # Adapt economic data
    adapted_economic = economic_data.copy()
    
    # Rename dimensions in economic data
    for old_dim, new_dim in dim_mapping.items():
        if old_dim in adapted_economic.dims and old_dim != new_dim:
            adapted_economic = adapted_economic.rename({old_dim: new_dim})
            logger.info(f"Renamed dimension '{old_dim}' to '{new_dim}' in economic data")
    
    # Ensure economic data has the right name
    if adapted_economic.name != 'gdppc':
        adapted_economic.name = 'gdppc'
        logger.info("Set economic data name to 'gdppc'")
    
    # Log final structure
    logger.info(f"Adapted damage data shape: {adapted_damage.shape}")
    logger.info(f"Adapted damage data dims: {list(adapted_damage.dims)}")
    logger.info(f"Adapted economic data shape: {adapted_economic.shape}")
    logger.info(f"Adapted economic data dims: {list(adapted_economic.dims)}")
    
    return adapted_damage, adapted_economic


def create_compatible_dummy_data_from_real(
    damage_data: xr.DataArray,
    economic_data: xr.DataArray
) -> Tuple[xr.Dataset, xr.DataArray]:
    """
    Create a compatible dummy dataset that mimics the structure needed for reduce_damages.
    
    This function takes real data and creates the histclim/delta structure expected
    by the reduce_damages function. The real data has separate variables for histclim
    and delta, so we need to return a Dataset, not a DataArray with a variable dimension.
    
    Parameters
    ----------
    damage_data : xr.DataArray
        Real damage data (could be delta or histclim)
    economic_data : xr.DataArray
        Real economic data
        
    Returns
    -------
    Tuple[xr.Dataset, xr.DataArray]
        (compatible_damage_dataset, compatible_economic_data)
    """
    logger.info("Creating compatible dummy data structure from real data...")
    
    # Adapt the basic structure first
    adapted_damage, adapted_economic = adapt_real_data_structure(damage_data, economic_data)
    
    # Check if we have a Dataset (multiple variables) or DataArray (single variable)
    if isinstance(adapted_damage, xr.Dataset):
        # We have a Dataset with multiple variables - this is what we want
        compatible_damage = adapted_damage
        logger.info("Using existing Dataset with multiple damage variables")
        
        # Check if we have the expected variable names
        expected_vars = ['histclim_dummy', 'delta_dummy']
        available_vars = list(compatible_damage.data_vars)
        logger.info(f"Available variables: {available_vars}")
        
        # If we don't have the expected names, try to find similar ones
        if not any(var in available_vars for var in expected_vars):
            # Look for variables containing 'histclim' or 'delta'
            histclim_vars = [var for var in available_vars if 'histclim' in var.lower()]
            delta_vars = [var for var in available_vars if 'delta' in var.lower()]
            
            if histclim_vars and delta_vars:
                logger.info(f"Found histclim variable: {histclim_vars[0]}")
                logger.info(f"Found delta variable: {delta_vars[0]}")
            else:
                logger.warning("Could not find histclim/delta variables, creating synthetic structure")
                # Create synthetic histclim/delta from first available variable
                first_var = available_vars[0]
                base_data = compatible_damage[first_var]
                
                # Create new dataset with expected variable names
                compatible_damage = xr.Dataset({
                    'histclim_dummy': base_data * 0.3,  # 30% histclim
                    'delta_dummy': base_data * 0.7      # 70% delta
                })
                logger.info("Created synthetic histclim_dummy and delta_dummy variables")
        
    else:
        # We have a DataArray - need to create Dataset with separate variables
        logger.info("Converting DataArray to Dataset with separate histclim/delta variables")
        
        # Remove variable dimension if it exists
        if 'variable' in adapted_damage.dims:
            base_damage = adapted_damage.isel(variable=0).drop('variable')
        else:
            base_damage = adapted_damage
        
        # Create Dataset with separate histclim and delta variables
        compatible_damage = xr.Dataset({
            'histclim_dummy': base_damage * 0.3,  # 30% histclim component
            'delta_dummy': base_damage * 0.7      # 70% delta component
        })
        logger.info("Created Dataset with histclim_dummy and delta_dummy variables")
    
    # Ensure we have a batch dimension for reduce_damages
    for var_name in compatible_damage.data_vars:
        if 'batch' not in compatible_damage[var_name].dims:
            # If no batch dimension, create one with a single batch
            compatible_damage[var_name] = compatible_damage[var_name].expand_dims('batch')
            compatible_damage = compatible_damage.assign_coords(batch=[0])
            logger.info(f"Added 'batch' dimension to {var_name}")
    
    # Add metadata
    compatible_damage.attrs.update({
        'title': 'Real Dummy Data Adapted for DSCIM Testing',
        'description': 'Real damage data adapted to expected Dataset structure with separate histclim/delta variables',
        'adapted_by': 'dscim_dev.io.data_loaders.create_compatible_dummy_data_from_real'
    })
    
    adapted_economic.attrs.update({
        'title': 'Real Economic Data Adapted for DSCIM Testing', 
        'description': 'Real economic data adapted to expected structure',
        'adapted_by': 'dscim_dev.io.data_loaders.create_compatible_dummy_data_from_real'
    })
    
    return compatible_damage, adapted_economic


def inspect_zarr_dataset(zarr_path: Path) -> Dict[str, Any]:
    """
    Inspect a Zarr dataset and return its structure information.
    
    Parameters
    ----------
    zarr_path : Path
        Path to the Zarr dataset
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with dataset information
    """
    try:
        dataset = xr.open_zarr(zarr_path)
        
        info = {
            'path': str(zarr_path),
            'type': type(dataset).__name__,
            'dims': dict(dataset.dims) if hasattr(dataset, 'dims') else {},
            'coords': list(dataset.coords) if hasattr(dataset, 'coords') else [],
            'data_vars': list(dataset.data_vars) if hasattr(dataset, 'data_vars') else [],
            'attrs': dict(dataset.attrs) if hasattr(dataset, 'attrs') else {},
            'size_mb': dataset.nbytes / (1024 * 1024) if hasattr(dataset, 'nbytes') else 0
        }
        
        logger.info(f"Zarr dataset info for {zarr_path.name}:")
        logger.info(f"  Type: {info['type']}")
        logger.info(f"  Dimensions: {info['dims']}")
        logger.info(f"  Coordinates: {info['coords']}")
        logger.info(f"  Data variables: {info['data_vars']}")
        logger.info(f"  Size: {info['size_mb']:.2f} MB")
        
        return info
        
    except Exception as e:
        logger.error(f"Error inspecting Zarr dataset {zarr_path}: {e}")
        return {'error': str(e)}


def load_zarr_with_fallback(zarr_path: Path, preferred_var: Optional[str] = None) -> Union[xr.DataArray, xr.Dataset]:
    """
    Load a Zarr dataset and convert to DataArray or return Dataset if multiple variables.
    
    Parameters
    ----------
    zarr_path : Path
        Path to Zarr dataset
    preferred_var : str, optional
        Preferred variable name to extract (if None, returns full Dataset for multiple variables)
        
    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Loaded data as DataArray (single variable) or Dataset (multiple variables)
    """
    logger.info(f"Loading Zarr dataset: {zarr_path}")
    
    # First inspect the dataset
    info = inspect_zarr_dataset(zarr_path)
    if 'error' in info:
        raise ValueError(f"Cannot load Zarr dataset: {info['error']}")
    
    # Load the dataset
    dataset = xr.open_zarr(zarr_path)
    
    if isinstance(dataset, xr.DataArray):
        logger.info("Dataset is already a DataArray")
        return dataset
    
    # Handle Dataset -> check if we should return full Dataset or single variable
    data_vars = list(dataset.data_vars)
    if not data_vars:
        raise ValueError("No data variables found in dataset")
    
    # Check if we have damage-related variables (histclim/delta)
    damage_vars = [var for var in data_vars if any(keyword in var.lower() for keyword in ['histclim', 'delta', 'damage'])]
    
    if len(damage_vars) > 1:
        # Multiple damage variables - return full Dataset
        logger.info(f"Found multiple damage variables: {damage_vars}")
        logger.info("Returning full Dataset with multiple variables")
        return dataset
    
    # Single variable or preferred variable specified
    if preferred_var and preferred_var in data_vars:
        selected_var = preferred_var
        logger.info(f"Using preferred variable: {selected_var}")
    else:
        # Use first variable as fallback
        selected_var = data_vars[0]
        logger.info(f"Using first available variable: {selected_var}")
        
        if len(data_vars) > 1:
            logger.info(f"Other available variables: {data_vars[1:]}")
    
    return dataset[selected_var]


def load_real_dummy_data(
    config_path: str = "configs/dummy_config.yaml"
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Load the actual dummy data files created by file_creation.ipynb.
    
    This function loads the data exactly as the original DSCIM code does,
    without any synthetic fallbacks.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        (damage_dataset, economic_dataset)
    """
    # Convert config_path to Path object and resolve it
    config_path = Path(config_path)
    
    # If config_path is relative, resolve it relative to current working directory
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    
    # Get the directory containing the config file to resolve relative paths
    config_dir = config_path.parent
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loading real dummy data using config: {config_path}")
    
    # Load sectoral damage data (non-coastal)
    sector_config = config['sectors']['dummy_not_coastl_sector']
    damage_path = sector_config['sector_path']
    
    # Resolve damage path relative to config directory if it's relative
    # Check the original string before creating Path object (Path normalizes and removes './')
    damage_path_str = str(damage_path)
    damage_path = Path(damage_path)
    if not damage_path.is_absolute():
        # If original path starts with './', resolve relative to config directory's parent
        # (since config is in configs/ and data is in dummy_data/ at same level)
        if damage_path_str.startswith('./'):
            # Remove './' prefix and resolve relative to config directory's parent
            relative_path = damage_path_str[2:]  # Remove './'
            damage_path = config_dir.parent / relative_path
        else:
            damage_path = config_dir / damage_path
    
    damage_dataset = xr.open_zarr(damage_path)
    
    # Load economic data
    econ_path = config['econdata']['global_ssp']
    
    # Resolve economic path relative to config directory if it's relative
    # Check the original string before creating Path object
    econ_path_str = str(econ_path)
    econ_path = Path(econ_path)
    if not econ_path.is_absolute():
        # If original path starts with './', resolve relative to config directory's parent
        if econ_path_str.startswith('./'):
            # Remove './' prefix and resolve relative to config directory's parent
            relative_path = econ_path_str[2:]  # Remove './'
            econ_path = config_dir.parent / relative_path
        else:
            econ_path = config_dir / econ_path
    
    economic_dataset = xr.open_zarr(econ_path)
    
    logger.info("Successfully loaded real dummy data")
    logger.info(f"Damage variables: {list(damage_dataset.data_vars)}")
    logger.info(f"Economic variables: {list(economic_dataset.data_vars)}")
    
    return damage_dataset, economic_dataset


def load_dummy_climate_data(config_path: str = 'configs/dummy_config.yaml') -> xr.Dataset:
    """
    Load dummy climate data from the actual dummy data files.
    
    This loads the real dummy climate data that the original DSCIM uses,
    ensuring consistency with the original implementation.
    Only loads temperature anomaly data to match original DSCIM exactly.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    xr.Dataset
        Climate dataset with temperature anomalies only (matching original DSCIM)
    """
    import yaml
    import pandas as pd
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    climate_config = config['AR6_ssp_climate']
    
    # Load temperature data from CSV
    gmst_path = climate_config['gmst_path']
    temp_df = pd.read_csv(gmst_path)
    
    logger.info(f"Loaded temperature data from {gmst_path}")
    logger.info(f"Temperature data shape: {temp_df.shape}")
    logger.info(f"Temperature columns: {list(temp_df.columns)}")
    
    # The temperature data has columns: year, rcp, gcm, anomaly
    # We need to create a combined dataset that matches the damage data coordinates
    # From the damage data, we know we need: year, ssp, model, rcp, gcm
    
    # Create coordinate mappings to match damage data structure
    years = sorted(temp_df['year'].unique())
    rcps = sorted(temp_df['rcp'].unique())  # ['dummy1', 'dummy2']
    gcms = sorted(temp_df['gcm'].unique())  # ['dummy1', 'dummy2']
    
    # For SSPs and models, we need to match the damage data structure
    # From the reduced damages, we know: ssp=['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5'], model=['dummy1', 'dummy2']
    ssps = ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5']
    models = ['dummy1', 'dummy2']
    
    # Create a mapping strategy:
    # - Use actual temperature anomalies from the CSV file
    # - Replicate across SSPs and models to match damage data structure
    
    # Initialize arrays for the full coordinate space
    n_years = len(years)
    n_ssps = len(ssps)
    n_models = len(models)
    n_rcps = len(rcps)
    n_gcms = len(gcms)
    
    # Create temperature anomaly array
    temp_data = np.zeros((n_years, n_ssps, n_models, n_rcps, n_gcms))
    
    for i, year in enumerate(years):
        for j, ssp in enumerate(ssps):
            for k, model in enumerate(models):
                for l, rcp in enumerate(rcps):
                    for m, gcm in enumerate(gcms):
                        # Get the actual temperature anomaly from the CSV data
                        temp_row = temp_df[
                            (temp_df['year'] == year) & 
                            (temp_df['rcp'] == rcp) & 
                            (temp_df['gcm'] == gcm)
                        ]
                        
                        if len(temp_row) > 0:
                            temp_data[i, j, k, l, m] = temp_row['anomaly'].iloc[0]
                        else:
                            # Fallback if no exact match
                            temp_data[i, j, k, l, m] = 1.0
    
    # Create xarray Dataset with only temperature anomaly (matching original DSCIM)
    climate_ds = xr.Dataset({
        'anomaly': (['year', 'ssp', 'model', 'rcp', 'gcm'], temp_data)
    }, coords={
        'year': years,
        'ssp': ssps,
        'model': models,
        'rcp': rcps,
        'gcm': gcms
    })
    
    # Add attributes
    climate_ds['anomaly'].attrs = {
        'units': 'degrees_C',
        'long_name': 'Temperature anomaly relative to pre-industrial',
        'source': gmst_path
    }
    
    logger.info(f"Created climate dataset with dimensions: {list(climate_ds.dims)}")
    logger.info(f"Climate dataset coordinates: {list(climate_ds.coords)}")
    
    return climate_ds


def load_reduced_damages(
    config_path: str = 'configs/dummy_config.yaml',
    discounting_type: str = 'adding_up',
    climate_change: bool = True
) -> xr.DataArray:
    """
    Load reduced damages data from the dummy data directory.
    
    For adding_up discounting, this loads both CC and No-CC files and calculates
    the difference (No-CC - CC) following the original DSCIM methodology.
    This includes population scaling and regional aggregation.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    discounting_type : str
        Type of discounting ('adding_up' or 'risk_aversion')
    climate_change : bool
        Whether to load climate change scenario (True) or no climate change (False)
        For adding_up, this parameter is ignored as we calculate the difference
        
    Returns
    -------
    xr.DataArray
        Reduced damages data (population-scaled and regionally aggregated for adding_up)
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Construct path to reduced damages
    reduced_damages_dir = Path(config['paths']['reduced_damages_library'])
    sector_dir = reduced_damages_dir / 'dummy_not_coastl_sector'
    
    if discounting_type == 'adding_up':
        # For adding_up, load both CC and No-CC and calculate difference
        # Following original DSCIM: (No-CC - CC) * population, summed across regions
        cc_path = sector_dir / 'adding_up_cc.zarr'
        no_cc_path = sector_dir / 'adding_up_no_cc.zarr'
        
        if not cc_path.exists():
            raise FileNotFoundError(f"CC damages file not found: {cc_path}")
        if not no_cc_path.exists():
            raise FileNotFoundError(f"No-CC damages file not found: {no_cc_path}")
        
        logger.info(f"Loading CC damages from {cc_path}")
        cc_data = xr.open_zarr(cc_path)
        
        logger.info(f"Loading No-CC damages from {no_cc_path}")
        no_cc_data = xr.open_zarr(no_cc_path)
        
        # Extract the main variables
        if isinstance(cc_data, xr.Dataset):
            cc_var = list(cc_data.data_vars)[0]
            cc_damages = cc_data[cc_var]
        else:
            cc_damages = cc_data
            
        if isinstance(no_cc_data, xr.Dataset):
            no_cc_var = list(no_cc_data.data_vars)[0]
            no_cc_damages = no_cc_data[no_cc_var]
        else:
            no_cc_damages = no_cc_data
        
        # Calculate difference: No-CC - CC (following original DSCIM)
        damage_diff = no_cc_damages - cc_damages
        
        logger.info("Calculated damage difference (No-CC - CC)")
        logger.info(f"Sample values - CC: {float(cc_damages.isel(year=0, ssp=0, region=0, model=0, rcp=0, gcm=0).values):.1f}")
        logger.info(f"Sample values - No-CC: {float(no_cc_damages.isel(year=0, ssp=0, region=0, model=0, rcp=0, gcm=0).values):.1f}")
        logger.info(f"Sample values - Difference: {float(damage_diff.isel(year=0, ssp=0, region=0, model=0, rcp=0, gcm=0).values):.1f}")
        
        # Load population data for scaling (following original DSCIM)
        econ_path = Path(config['econdata']['global_ssp'])
        logger.info(f"Loading population data from {econ_path}")
        econ_data = xr.open_zarr(econ_path)
        
        if 'pop' not in econ_data:
            raise ValueError("Population data ('pop') not found in economic dataset")
        
        pop_data = econ_data['pop']
        logger.info(f"Population data dimensions: {list(pop_data.dims)}")
        
        # Multiply by population (following original DSCIM: damages * population)
        # Need to align dimensions for multiplication
        # damage_diff dims: ['year', 'ssp', 'region', 'model', 'rcp', 'gcm']
        # pop_data dims: ['year', 'ssp', 'model', 'region']
        
        # Align population data to match damage dimensions
        pop_aligned = pop_data.reindex_like(damage_diff, method='nearest', fill_value=0)
        
        # Multiply damages by population
        scaled_damages = damage_diff * pop_aligned
        
        logger.info("Applied population scaling")
        logger.info(f"Sample scaled damage: {float(scaled_damages.isel(year=0, ssp=0, region=0, model=0, rcp=0, gcm=0).values):.1f}")
        
        # Sum across regions (following original DSCIM)
        data = scaled_damages.sum(dim='region')
        
        logger.info("Aggregated damages across regions")
        logger.info(f"Sample aggregated damage: {float(data.isel(year=0, ssp=0, model=0, rcp=0, gcm=0).values):.1f}")
        
    elif discounting_type == 'risk_aversion':
        # For risk_aversion, use single file as before
        cc_suffix = 'cc' if climate_change else 'no_cc'
        zarr_file = f'risk_aversion_{cc_suffix}_eta2.0.zarr'
        zarr_path = sector_dir / zarr_file
        
        if not zarr_path.exists():
            raise FileNotFoundError(f"Reduced damages file not found: {zarr_path}")
        
        logger.info(f"Loading reduced damages from {zarr_path}")
        data = xr.open_zarr(zarr_path)
        
        # If it's a Dataset, extract the main variable
        if isinstance(data, xr.Dataset):
            if 'damages' in data.data_vars:
                data = data['damages']
            elif len(data.data_vars) == 1:
                var_name = list(data.data_vars.keys())[0]
                data = data[var_name]
            else:
                raise ValueError(f"Multiple variables found in dataset: {list(data.data_vars.keys())}")
    else:
        raise ValueError(f"Unknown discounting_type: {discounting_type}")
    
    logger.info(f"Loaded reduced damages with dimensions: {list(data.dims)}")
    logger.info(f"Reduced damages coordinates: {list(data.coords.keys())}")
    
    return data 