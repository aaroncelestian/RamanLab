"""
RamanLab UI Module Package

This package contains modular UI components for the RamanLab application.
All modules use PySide6 for consistency with the main application.
"""

# Global matplotlib configuration for compact toolbars
def configure_matplotlib_ui():
    """Configure matplotlib for compact UI elements."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    # Toolbar configuration (using valid rcParams only)
    mpl.rcParams['toolbar'] = 'toolbar2'
    
    # Font sizes for better UI integration
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    
    # Figure layout for better spacing
    mpl.rcParams['figure.autolayout'] = True
    
    # Set DPI for sharper icons (if supported)
    try:
        mpl.rcParams['figure.dpi'] = 100
    except KeyError:
        pass

# Apply configuration when module is imported
configure_matplotlib_ui()

import sys
import os

# Ensure the ui directory is in the path for relative imports
ui_dir = os.path.dirname(os.path.abspath(__file__))
if ui_dir not in sys.path:
    sys.path.insert(0, ui_dir)

# Import all UI modules
try:
    from .spectrum_analysis import SpectrumAnalysisWidget
    print("✓ Spectrum analysis module loaded")
except ImportError as e:
    print(f"⚠ Warning: Could not import spectrum_analysis: {e}")

try:
    from .peak_fitting import PeakFittingWidget
    print("✓ Peak fitting module loaded")
except ImportError as e:
    print(f"⚠ Warning: Could not import peak_fitting: {e}")

try:
    from .crystal_structure import CrystalStructureWidget, MineralSelectionDialog
    print("✓ Crystal structure module loaded")
except ImportError as e:
    print(f"⚠ Warning: Could not import crystal_structure: {e}")

try:
    from .polarization_dialogs import (
        PolarizedSpectraLoadingDialog, 
        DatabaseGenerationDialog, 
        PolarizedSpectraDialog,  # Backward compatibility
        FileLoadingWidget,
        DatabaseGenerationWidget
    )
    print("✓ Polarization dialogs module loaded")
except ImportError as e:
    print(f"⚠ Warning: Could not import polarization_dialogs: {e}")


def list_available_modules():
    """Return a list of available UI modules."""
    modules = []
    
    try:
        from . import spectrum_analysis
        modules.append('spectrum_analysis')
    except ImportError:
        pass
        
    try:
        from . import peak_fitting
        modules.append('peak_fitting')
    except ImportError:
        pass
        
    try:
        from . import crystal_structure
        modules.append('crystal_structure')
    except ImportError:
        pass
        
    try:
        from . import polarization_dialogs
        modules.append('polarization_dialogs')
    except ImportError:
        pass
        
    return modules


def get_module_info():
    """Get information about available UI modules."""
    info = {
        'total_modules': len(list_available_modules()),
        'available_modules': list_available_modules(),
        'description': 'Modular UI components for RamanLab'
    }
    return info


# Export main classes for easy import
__all__ = [
    'SpectrumAnalysisWidget',
    'PeakFittingWidget', 
    'CrystalStructureWidget',
    'MineralSelectionDialog',
    'PolarizedSpectraLoadingDialog',
    'DatabaseGenerationDialog',
    'PolarizedSpectraDialog',
    'FileLoadingWidget',
    'DatabaseGenerationWidget',
    'list_available_modules',
    'get_module_info',
    'configure_matplotlib_ui'
]

# Version information
__version__ = "1.0.0"
__author__ = "RamanLab Development Team"

# Module registry for dynamic loading
AVAILABLE_MODULES = {
    'spectrum_analysis': {
        'class': 'SpectrumAnalysisWidget',
        'description': 'Complete spectrum analysis interface with visualization',
        'features': [
            'Multi-spectrum loading and comparison',
            'Real-time analysis parameter adjustment', 
            'Professional plotting with multiple visualization modes',
            'Integration with database and core analysis modules'
        ]
    },
    'peak_fitting': {
        'class': 'PeakFittingWidget', 
        'description': 'Advanced peak fitting and deconvolution capabilities',
        'features': [
            'Interactive peak detection with parameter adjustment',
            'Multiple fitting models with real-time preview',
            'Manual peak addition and deletion',
            'Advanced deconvolution and component separation'
        ]
    },
    'crystal_structure': {
        'class': 'CrystalStructureWidget',
        'description': 'Crystal structure analysis and 3D visualization',
        'features': [
            'CIF file loading with pymatgen integration',
            'Database-driven mineral structure selection',
            'Interactive 3D visualization',
            'Bond length and angle analysis'
        ]
    },
    'polarization_dialogs': {
        'class': 'PolarizedSpectraLoadingDialog',
        'description': 'Polarized spectra loading and database generation',
        'features': [
            'Sophisticated polarized spectra loading dialog',
            'Two-tab interface for file loading and database generation',
            'Real-time preview with parameter adjustment',
            'Integration with core polarization modules'
        ]
    }
}

def get_module_info(module_name):
    """Get information about a specific UI module."""
    return AVAILABLE_MODULES.get(module_name, None)

def get_module_features(module_name):
    """Get features list for a specific module."""
    module_info = AVAILABLE_MODULES.get(module_name, {})
    return module_info.get('features', [])

# Import main widgets for easy access
try:
    from .crystal_structure_widget import CrystalStructureWidget
    __all__ = ['CrystalStructureWidget']
except ImportError:
    __all__ = [] 