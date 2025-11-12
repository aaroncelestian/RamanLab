"""
RamanLab Cluster Analysis Module

This module provides comprehensive clustering analysis capabilities for Raman spectroscopy data,
including hierarchical clustering, visualization tools, and advanced analysis methods.
"""

__version__ = "1.4.0"
__author__ = "Aaron J. Celestian Ph.D."

from .main import RamanClusterAnalysisQt6, launch_cluster_analysis

__all__ = ['RamanClusterAnalysisQt6', 'launch_cluster_analysis']
