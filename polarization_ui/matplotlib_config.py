"""
Matplotlib Configuration for RamanLab

This module provides centralized configuration for matplotlib in Qt applications,
specifically optimized for RamanLab's interface requirements.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT


def configure_compact_ui():
    """Configure matplotlib for compact, professional UI integration."""
    
    # Toolbar configuration
    mpl.rcParams['toolbar'] = 'toolbar2'
    
    # Font sizes optimized for embedded plots
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['axes.titlesize'] = 11
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['legend.title_fontsize'] = 9
    
    # Figure layout optimization
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 150
    
    # Axis and grid styling
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['axes.axisbelow'] = True
    
    # Line and marker settings
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 4
    
    # Color cycle optimized for scientific data
    mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])


class CompactNavigationToolbar(NavigationToolbar2QT):
    """
    Enhanced navigation toolbar with reduced size and improved styling.
    
    Features:
    - 30% smaller icons
    - Reduced height
    - Professional styling
    - Hover effects
    """
    
    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        self._setup_compact_style()
    
    def _setup_compact_style(self):
        """Apply compact styling to the toolbar."""
        
        # Reduce icon size by 30%
        original_size = self.iconSize()
        compact_size = original_size * 0.7
        self.setIconSize(compact_size)
        
        # Set maximum height
        self.setMaximumHeight(32)
        self.setMinimumHeight(28)
        
        # Apply professional stylesheet
        self.setStyleSheet("""
            QToolBar {
                spacing: 1px;
                padding: 2px 4px;
                border: none;
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }
            
            QToolButton {
                padding: 3px;
                margin: 1px;
                border: 1px solid transparent;
                border-radius: 4px;
                background-color: transparent;
                min-width: 20px;
                min-height: 20px;
            }
            
            QToolButton:hover {
                background-color: #e9ecef;
                border: 1px solid #ced4da;
            }
            
            QToolButton:pressed {
                background-color: #dee2e6;
                border: 1px solid #adb5bd;
            }
            
            QToolButton:checked {
                background-color: #0d6efd;
                border: 1px solid #0a58ca;
                color: white;
            }
            
            QToolButton:disabled {
                opacity: 0.5;
                background-color: transparent;
                border: 1px solid transparent;
            }
            
            /* Style the coordinates display */
            QLabel {
                color: #495057;
                font-size: 9px;
                padding: 2px 4px;
                background-color: transparent;
            }
        """)


class MiniNavigationToolbar(CompactNavigationToolbar):
    """
    Ultra-compact toolbar for small interfaces.
    
    Shows only essential tools: pan, zoom, home, back/forward.
    """
    
    # Override the default tools to show only essential ones
    toolitems = [
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),  # Separator
        ('Pan', 'Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),  # Separator
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    ]
    
    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        self.setMaximumHeight(28)


def apply_theme(theme='compact'):
    """
    Apply a predefined theme to matplotlib.
    
    Args:
        theme (str): Theme name - 'compact', 'minimal', or 'publication'
    """
    if theme == 'compact':
        configure_compact_ui()
    
    elif theme == 'minimal':
        # Ultra-minimal configuration
        mpl.rcParams['toolbar'] = 'toolbar2'
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['axes.titlesize'] = 9
        mpl.rcParams['axes.labelsize'] = 8
        mpl.rcParams['xtick.labelsize'] = 7
        mpl.rcParams['ytick.labelsize'] = 7
        mpl.rcParams['legend.fontsize'] = 7
        mpl.rcParams['figure.autolayout'] = True
        mpl.rcParams['axes.grid'] = False
        
    elif theme == 'publication':
        # High-quality publication theme
        mpl.rcParams['toolbar'] = 'toolbar2'
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.titlesize'] = 12
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['xtick.labelsize'] = 9
        mpl.rcParams['ytick.labelsize'] = 9
        mpl.rcParams['legend.fontsize'] = 9
        mpl.rcParams['figure.dpi'] = 150
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['figure.autolayout'] = True
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['grid.alpha'] = 0.2


def add_colorbar_no_shrink(figure, mappable, ax, **kwargs):
    """
    PERMANENT SOLUTION: Add colorbar without shrinking the main plot.
    
    This is the definitive fix for matplotlib colorbar shrinkage issues.
    Use this function instead of figure.colorbar() to prevent plot shrinkage.
    
    Args:
        figure: matplotlib figure object
        mappable: the plot object (e.g., from imshow, contour, etc.)
        ax: the axes object to add colorbar to
        **kwargs: additional arguments passed to colorbar()
    
    Returns:
        colorbar object or None if creation fails
    
    RULE: Always use this function for colorbars in RamanLab to prevent shrinkage!
    """
    try:
        # Method 1: Use make_axes_locatable (most robust)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return figure.colorbar(mappable, cax=cax, **kwargs)
        
    except ImportError:
        # Fallback if mpl_toolkits not available
        try:
            return figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, **kwargs)
        except Exception:
            return None
    except Exception:
        # Second fallback
        try:
            return figure.colorbar(mappable, ax=ax, shrink=0.8, **kwargs)
        except Exception:
            return None


def get_toolbar_class(size='compact'):
    """
    Get the appropriate toolbar class based on size preference.
    
    Args:
        size (str): 'standard', 'compact', or 'mini'
    
    Returns:
        Toolbar class to use
    """
    if size == 'mini':
        return MiniNavigationToolbar
    elif size == 'compact':
        return CompactNavigationToolbar
    else:
        return NavigationToolbar2QT


# Apply compact configuration by default when module is imported
configure_compact_ui()


# Export the main configuration functions and classes
__all__ = [
    'configure_compact_ui',
    'CompactNavigationToolbar', 
    'MiniNavigationToolbar',
    'apply_theme',
    'get_toolbar_class'
] 