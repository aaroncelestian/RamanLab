"""
RamanLab Cluster Analysis Module

This module provides comprehensive clustering analysis capabilities for Raman spectroscopy data,
including hierarchical clustering, visualization tools, and advanced analysis methods.
"""

__version__ = "1.4.0"
__author__ = "Aaron J. Celestian Ph.D."

print("="*60)
print("DEBUG: cluster_analysis module __init__.py is being imported!")
print("="*60)

try:
    from .main import RamanClusterAnalysisQt6, launch_cluster_analysis
    print("DEBUG: Successfully imported RamanClusterAnalysisQt6 and launch_cluster_analysis")
except Exception as e:
    print(f"DEBUG: ERROR importing from main: {e}")
    import traceback
    traceback.print_exc()
    raise

__all__ = ['RamanClusterAnalysisQt6', 'launch_cluster_analysis']
