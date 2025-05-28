"""
Input/Output Operations Module
=============================

This module contains all file I/O operations decoupled from
computational logic. Functions here handle data loading and saving only.

Phase 1: Basic data loading for testing
Phase 2: Result export functionality
"""

from .data_loaders import load_dummy_data, load_netcdf_data
from .result_exporters import save_fitting_results

__all__ = [
    'load_dummy_data',
    'load_netcdf_data', 
    'save_fitting_results'
] 