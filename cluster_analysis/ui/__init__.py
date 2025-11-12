"""
UI Components for RamanLab Cluster Analysis

This module contains all UI components including tabs and dialogs.
"""

from .tabs.import_tab import ImportTab
from .tabs.clustering_tab import ClusteringTab
from .tabs.visualization_tabs import VisualizationTab, DendrogramTab, HeatmapTab, ScatterTab
from .tabs.analysis_tab import AnalysisTab
from .tabs.refinement_tab import RefinementTab
from .tabs.advanced_tabs import (TimeSeriesTab, KineticsTab, StructuralAnalysisTab, 
                                ValidationTab, AdvancedStatisticsTab)

__all__ = [
    'ImportTab',
    'ClusteringTab', 
    'VisualizationTab',
    'DendrogramTab',
    'HeatmapTab',
    'ScatterTab',
    'AnalysisTab',
    'RefinementTab',
    'TimeSeriesTab',
    'KineticsTab',
    'StructuralAnalysisTab',
    'ValidationTab',
    'AdvancedStatisticsTab'
]
