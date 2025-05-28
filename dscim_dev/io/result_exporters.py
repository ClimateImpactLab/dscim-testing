"""
Result Export Functions
=======================

Pure I/O functions for saving results to various formats.
Completely decoupled from computational logic.
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def save_fitting_results(
    results: pd.DataFrame,
    output_path: str,
    format: str = "csv",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save damage function fitting results to file.
    
    Pure I/O function for result export.
    
    Parameters
    ----------
    results : pd.DataFrame
        Fitting results from fit_damage_functions_batch
    output_path : str
        Output file path
    format : str
        Output format ("csv", "parquet", "excel")
    metadata : dict, optional
        Additional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format.lower() == "csv":
            results.to_csv(output_path, index=False)
            
            # Save metadata separately if provided
            if metadata:
                metadata_path = output_path.with_suffix('.metadata.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                    
        elif format.lower() == "parquet":
            results.to_parquet(output_path, index=False)
            
        elif format.lower() == "excel":
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                results.to_excel(writer, sheet_name='fitting_results', index=False)
                
                # Add metadata sheet if provided
                if metadata:
                    metadata_df = pd.DataFrame([metadata]).T
                    metadata_df.columns = ['value']
                    metadata_df.to_excel(writer, sheet_name='metadata')
                    
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Saved fitting results to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        raise


def _sanitize_attrs_for_netcdf(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize attributes for NetCDF compatibility.
    
    NetCDF has restrictions on attribute data types.
    """
    sanitized = {}
    for key, value in attrs.items():
        if isinstance(value, bool):
            # Convert boolean to integer
            sanitized[key] = int(value)
        elif isinstance(value, np.bool_):
            # Convert numpy boolean to integer
            sanitized[key] = int(value)
        elif isinstance(value, (list, tuple)):
            # Convert sequences to strings if they contain unsupported types
            try:
                sanitized[key] = value
            except:
                sanitized[key] = str(value)
        else:
            sanitized[key] = value
    return sanitized


def save_reduced_damages(
    data: xr.DataArray, 
    output_path: str,
    format: str = "zarr",
    compression: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save reduced damages data to file with flexible format support.
    
    Pure I/O function for saving computation results.
    
    Parameters
    ----------
    data : xr.DataArray
        Reduced damages data to save
    output_path : str
        Output file path (extension will be added if not present)
    format : str
        Output format ("zarr", "netcdf", "csv", "parquet")
    compression : str, optional
        Compression method (format-dependent)
    metadata : Dict[str, Any], optional
        Additional metadata to include in output
        
    Raises
    ------
    ValueError
        If format is not supported
    """
    output_path = Path(output_path)
    
    # Add metadata if provided
    if metadata:
        data = data.copy()
        data.attrs.update(metadata)
    
    # Add timestamp
    import datetime
    data.attrs['created_at'] = datetime.datetime.now().isoformat()
    
    try:
        if format.lower() == "zarr":
            # Ensure .zarr extension
            if not output_path.suffix == '.zarr':
                output_path = output_path.with_suffix('.zarr')
            
            # Convert to Dataset for zarr compatibility
            dataset = data.to_dataset(name='reduced_damages')
            
            # Save with compression
            encoding = {}
            if compression:
                encoding['reduced_damages'] = {'compressor': compression}
            
            dataset.to_zarr(
                output_path, 
                mode='w', 
                consolidated=True,
                encoding=encoding if compression else None
            )
            
        elif format.lower() == "netcdf":
            # Ensure .nc extension
            if not output_path.suffix == '.nc':
                output_path = output_path.with_suffix('.nc')
            
            # Convert to Dataset for netcdf compatibility
            dataset = data.to_dataset(name='reduced_damages')
            
            # Sanitize attributes for NetCDF compatibility
            dataset.attrs = _sanitize_attrs_for_netcdf(dataset.attrs)
            dataset['reduced_damages'].attrs = _sanitize_attrs_for_netcdf(dataset['reduced_damages'].attrs)
            
            # Save with compression
            encoding = {}
            if compression:
                encoding['reduced_damages'] = {'zlib': True, 'complevel': 9}
            
            dataset.to_netcdf(
                output_path,
                encoding=encoding if compression else None
            )
            
        elif format.lower() == "csv":
            # Ensure .csv extension
            if not output_path.suffix == '.csv':
                output_path = output_path.with_suffix('.csv')
            
            # Convert to DataFrame and save
            df = data.to_dataframe(name='reduced_damages').reset_index()
            df.to_csv(output_path, index=False)
            
        elif format.lower() == "parquet":
            # Ensure .parquet extension
            if not output_path.suffix == '.parquet':
                output_path = output_path.with_suffix('.parquet')
            
            # Convert to DataFrame and save
            df = data.to_dataframe(name='reduced_damages').reset_index()
            df.to_parquet(output_path, compression=compression or 'snappy')
            
        else:
            raise ValueError(f"Unsupported format: {format}. Supported formats: zarr, netcdf, csv, parquet")
            
        logger.info(f"Successfully saved reduced damages to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save reduced damages to {output_path}: {e}")
        raise


def create_summary_report(
    fitting_results: pd.DataFrame,
    reduced_damages: xr.DataArray,
    output_dir: str
) -> None:
    """
    Create a comprehensive summary report.
    
    Pure I/O function for report generation.
    
    Parameters
    ----------
    fitting_results : pd.DataFrame
        Damage function fitting results
    reduced_damages : xr.DataArray
        Reduced damage results
    output_dir : str
        Output directory for report files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create summary statistics
        summary_stats = {
            'fitting_summary': {
                'total_fits': len(fitting_results),
                'success_rate': fitting_results['convergence_success'].mean(),
                'mean_r_squared': fitting_results.loc[
                    fitting_results['convergence_success'], 'r_squared'
                ].mean(),
                'unique_regions': fitting_results['region'].nunique(),
                'unique_gcms': fitting_results['gcm'].nunique(),
                'unique_scenarios': fitting_results['scenario'].nunique()
            },
            'damage_summary': {
                'data_shape': reduced_damages.shape,
                'dimensions': list(reduced_damages.dims),
                'mean_damage': float(reduced_damages.mean().values),
                'total_damage': float(reduced_damages.sum().values),
                'min_damage': float(reduced_damages.min().values),
                'max_damage': float(reduced_damages.max().values)
            }
        }
        
        # Save summary as JSON
        import json
        summary_path = output_dir / 'summary_report.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        # Save detailed results
        fitting_path = output_dir / 'fitting_results.csv'
        fitting_results.to_csv(fitting_path, index=False)
        
        damages_path = output_dir / 'reduced_damages.nc'
        reduced_damages.to_netcdf(damages_path)
        
        logger.info(f"Created summary report in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating summary report: {e}")
        raise


def save_multiple_results(
    results: Dict[str, xr.DataArray],
    output_dir: str,
    format: str = "zarr",
    prefix: str = "",
    suffix: str = "",
    **save_kwargs
) -> Dict[str, str]:
    """
    Save multiple results to files with consistent naming.
    
    Parameters
    ----------
    results : Dict[str, xr.DataArray]
        Dictionary mapping names to DataArrays
    output_dir : str
        Output directory
    format : str
        Output format for all files
    prefix : str
        Prefix to add to all filenames
    suffix : str
        Suffix to add to all filenames (before extension)
    **save_kwargs
        Additional arguments passed to save_reduced_damages()
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping result names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    for name, data in results.items():
        # Clean filename
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        filename = f"{prefix}{clean_name}{suffix}"
        
        file_path = output_path / filename
        
        save_reduced_damages(
            data, 
            str(file_path), 
            format=format,
            **save_kwargs
        )
        
        file_paths[name] = str(file_path)
    
    return file_paths 