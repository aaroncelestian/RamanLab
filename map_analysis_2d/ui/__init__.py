"""
User Interface module for the map analysis application.

This module contains all UI components including the main window, plotting widgets,
control panels, and dialogs.
"""

from .main_window import MapAnalysisMainWindow
from .base_widgets import (
    SafeWidgetMixin, ParameterGroupBox, ButtonGroup, 
    ScrollableControlPanel, TitleLabel, StandardButton, ProgressStatusWidget
)
from .plotting_widgets import (
    BasePlotWidget, MapPlotWidget, SpectrumPlotWidget, 
    SplitMapSpectrumWidget, PCANMFPlotWidget
)
from .control_panels import (
    BaseControlPanel, MapViewControlPanel, PCAControlPanel,
    DimensionalityReductionControlPanel, TemplateControlPanel,
    NMFControlPanel, MLControlPanel, ResultsControlPanel
)

__all__ = [
    'MapAnalysisMainWindow',
    'SafeWidgetMixin', 
    'ParameterGroupBox',
    'ButtonGroup',
    'ScrollableControlPanel',
    'TitleLabel',
    'StandardButton',
    'ProgressStatusWidget',
    'BasePlotWidget',
    'MapPlotWidget', 
    'SpectrumPlotWidget',
    'SplitMapSpectrumWidget',
    'PCANMFPlotWidget',
    'BaseControlPanel',
    'MapViewControlPanel',
    'PCAControlPanel',
    'DimensionalityReductionControlPanel',
    'TemplateControlPanel',
    'NMFControlPanel',
    'MLControlPanel',
    'ResultsControlPanel'
] 