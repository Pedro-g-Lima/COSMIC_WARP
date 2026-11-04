"""
COSMIC WARP - Spacecraft Simulator

A GPU-accelerated spacecraft dynamics simulator built on NVIDIA WARP.
"""

__version__ = "0.1.0"
__author__ = "Pedro Lima, Aman Arora"

# Import main simulator class for convenience
from cosmic_warp.core.pipeline import SpacecraftSimulator

__all__ = [
    'SpacecraftSimulator',
]
