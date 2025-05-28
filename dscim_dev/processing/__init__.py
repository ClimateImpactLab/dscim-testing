"""
Data Processing and Orchestration Module
========================================

This module contains orchestration functions that coordinate pure mathematical
operations. Functions here operate on in-memory data and call pure functions
from the math module.

Phase 1: Core data processing (reduce_damages)
Phase 2: Damage function fitting workflows
"""

from .core_operations import reduce_damages
from .damage_workflows import fit_damage_functions_batch

__all__ = [
    'reduce_damages',
    'fit_damage_functions_batch'
] 