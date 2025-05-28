"""
Pure Mathematical Functions Module
=================================

This module contains all pure mathematical functions for DSCIM.
Functions here have no side effects and operate only on in-memory data.

Phase 1: Core economic functions (discounted_damages, weitzman_min, SCC)
Phase 2: Damage function fitting algorithms
"""

from .core_functions import discounted_damages, weitzman_min, calculate_scc
from .damage_fitting import fit_damage_function

__all__ = [
    'discounted_damages',
    'weitzman_min', 
    'calculate_scc',
    'fit_damage_function'
] 