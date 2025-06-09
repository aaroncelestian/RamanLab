#!/usr/bin/env python3
"""
Canvas and Colorbar Management Fix

This module provides improved functions for handling matplotlib canvas replotting
and colorbar management to prevent plot shrinking and layout issues.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging

# Import the matplotlib configuration
try:
    from ui.matplotlib_config import configure_compact_ui, apply_theme
    configure_compact_ui()
except ImportError:
    print("Warning: Could not import matplotlib_config, using default settings")

logger = logging.getLogger(__name__)

class CanvasManager:
    """
    Manages matplotlib canvas operations with proper layout and colorbar handling.
    """
    
    def __init__(self, figure, canvas, axes):
        """
        Initialize the canvas manager.
        
        Args:
            figure: matplotlib Figure object
            canvas: matplotlib canvas object  
            axes: matplotlib Axes object
        """
        self.figure = figure
        self.canvas = canvas
        self.axes = axes
        self.colorbar = None
        self._layout_params = None
        self._store_original_layout()
    
    def _store_original_layout(self):
        """Store the original layout parameters."""
        try:
            # Get current subplot parameters
            subplotpars = self.figure.subplotpars
            self._layout_params = {
                'left': subplotpars.left,
                'bottom': subplotpars.bottom,
                'right': subplotpars.right,
                'top': subplotpars.top,
                'wspace': subplotpars.wspace,
                'hspace': subplotpars.hspace
            }
        except:
            # Default layout if we can't get current parameters
            self._layout_params = {
                'left': 0.1,
                'bottom': 0.1,
                'right': 0.9,
                'top': 0.9,
                'wspace': 0.2,
                'hspace': 0.2
            }
    
    def safe_clear_and_replot(self, plot_function, *args, **kwargs):
        """
        Safely clear the plot and replot with proper layout management.
        
        Args:
            plot_function: Function that creates the new plot
            *args, **kwargs: Arguments to pass to the plot function
        """
        
        # Store current view limits if they exist
        x_lim = None
        y_lim = None
        try:
            if hasattr(self.axes, 'get_xlim'):
                x_lim = self.axes.get_xlim()
                y_lim = self.axes.get_ylim()
        except:
            pass
        
        # Remove existing colorbar if it exists
        self._remove_colorbar_safely()
        
        # Clear the axes but preserve figure structure
        self.axes.clear()
        
        # Call the plotting function
        result = plot_function(self.axes, *args, **kwargs)
        
        # Restore view limits if they were previously set
        if x_lim is not None and y_lim is not None:
            try:
                self.axes.set_xlim(x_lim)
                self.axes.set_ylim(y_lim)
            except:
                pass
        
        # Refresh canvas properly
        self._refresh_canvas()
        
        return result
    
    def add_colorbar_safely(self, mappable, **kwargs):
        """
        Add a colorbar with proper layout management.
        
        Args:
            mappable: The mappable object (e.g., result of imshow)
            **kwargs: Additional arguments for colorbar
        """
        
        # Remove existing colorbar first
        self._remove_colorbar_safely()
        
        try:
            # Default colorbar parameters
            cbar_kwargs = {
                'ax': self.axes,
                'shrink': 0.8,
                'aspect': 20,
                'pad': 0.02
            }
            cbar_kwargs.update(kwargs)
            
            # Create colorbar
            self.colorbar = self.figure.colorbar(mappable, **cbar_kwargs)
            
            # Adjust layout to accommodate colorbar
            self._adjust_layout_for_colorbar()
            
            logger.debug("Colorbar added successfully")
            
        except Exception as e:
            logger.warning(f"Error creating colorbar: {e}")
            self.colorbar = None
            # Restore layout without colorbar
            self._restore_original_layout()
    
    def _remove_colorbar_safely(self):
        """Remove the existing colorbar safely."""
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
                logger.debug("Colorbar removed successfully")
            except Exception as e:
                logger.debug(f"Error removing colorbar: {e}")
            finally:
                self.colorbar = None
                # Restore original layout
                self._restore_original_layout()
    
    def _adjust_layout_for_colorbar(self):
        """Adjust subplot parameters to make room for colorbar."""
        try:
            # Leave space for colorbar on the right
            self.figure.subplots_adjust(
                left=self._layout_params['left'],
                bottom=self._layout_params['bottom'], 
                right=0.85,  # Make room for colorbar
                top=self._layout_params['top']
            )
        except Exception as e:
            logger.debug(f"Error adjusting layout for colorbar: {e}")
    
    def _restore_original_layout(self):
        """Restore the original layout parameters."""
        try:
            self.figure.subplots_adjust(**self._layout_params)
        except Exception as e:
            logger.debug(f"Error restoring original layout: {e}")
    
    def _refresh_canvas(self):
        """Properly refresh the canvas with multiple steps to ensure complete redraw."""
        try:
            # Multiple refresh steps to ensure proper rendering
            self.canvas.flush_events()
            self.figure.canvas.draw_idle()
            self.canvas.draw()
            
            # Process any pending Qt events
            try:
                from PySide6.QtCore import QCoreApplication
                QCoreApplication.processEvents()
            except ImportError:
                try:
                    from PyQt6.QtCore import QCoreApplication
                    QCoreApplication.processEvents()
                except ImportError:
                    pass
                    
        except Exception as e:
            logger.debug(f"Error refreshing canvas: {e}")


def create_improved_map_plot(canvas_manager, map_data, title="Map", colormap='viridis', 
                           extent=None, interpolation='nearest', **kwargs):
    """
    Create an improved map plot with proper colorbar management.
    
    Args:
        canvas_manager: CanvasManager instance
        map_data: 2D numpy array of map data
        title: Plot title
        colormap: Colormap name
        extent: Extent for the plot [x_min, x_max, y_min, y_max]
        interpolation: Interpolation method
        **kwargs: Additional arguments for imshow
    """
    
    def plot_function(ax):
        """Internal plotting function."""
        
        # Clean the data
        map_data_clean = np.ma.masked_invalid(map_data)
        
        # Set up default imshow parameters
        imshow_kwargs = {
            'cmap': colormap,
            'aspect': 'auto',
            'origin': 'lower',
            'interpolation': interpolation
        }
        
        if extent is not None:
            imshow_kwargs['extent'] = extent
        
        # Update with any additional kwargs
        imshow_kwargs.update(kwargs)
        
        # Create the image
        im = ax.imshow(map_data_clean, **imshow_kwargs)
        
        # Set labels and title
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        
        return im
    
    # Use the canvas manager to safely replot
    im = canvas_manager.safe_clear_and_replot(plot_function)
    
    # Add colorbar safely
    canvas_manager.add_colorbar_safely(im)
    
    return im


def fix_existing_map_plotting(map_analysis_instance):
    """
    Fix the map plotting in an existing TwoDMapAnalysisQt6 instance.
    
    Args:
        map_analysis_instance: Instance of TwoDMapAnalysisQt6
    """
    
    # Create a canvas manager for the map plot
    if hasattr(map_analysis_instance, 'map_fig') and hasattr(map_analysis_instance, 'map_canvas'):
        map_analysis_instance.canvas_manager = CanvasManager(
            map_analysis_instance.map_fig,
            map_analysis_instance.map_canvas, 
            map_analysis_instance.map_ax
        )
        
        # Store original update_map method and replace with improved version
        original_update_map = map_analysis_instance.update_map
        
        def improved_update_map():
            """Improved update_map method with proper canvas management."""
            try:
                # Get current feature and data
                current_feature = getattr(map_analysis_instance, 'current_feature', 'Integrated Intensity')
                
                # Get map data based on selected feature (simplified version)
                if current_feature == "Integrated Intensity":
                    min_wn = 800.0
                    max_wn = 1000.0
                    if hasattr(map_analysis_instance, 'min_wavenumber_edit') and map_analysis_instance.min_wavenumber_edit:
                        try:
                            min_wn = float(map_analysis_instance.min_wavenumber_edit.text() or "800")
                        except:
                            pass
                    if hasattr(map_analysis_instance, 'max_wavenumber_edit') and map_analysis_instance.max_wavenumber_edit:
                        try:
                            max_wn = float(map_analysis_instance.max_wavenumber_edit.text() or "1000")
                        except:
                            pass
                    
                    map_data = map_analysis_instance.create_integrated_intensity_map(min_wn, max_wn)
                    title = f"Integrated Intensity ({min_wn}-{max_wn} cm⁻¹)"
                
                else:
                    # Fall back to original method for other features
                    return original_update_map()
                
                if map_data is not None and map_data.size > 0:
                    # Apply interpolation if needed
                    map_data, interpolation_method = map_analysis_instance.apply_map_interpolation(map_data)
                    
                    # Calculate extent
                    extent = None
                    if hasattr(map_analysis_instance.map_data, 'x_positions') and hasattr(map_analysis_instance.map_data, 'y_positions'):
                        x_min, x_max = min(map_analysis_instance.map_data.x_positions), max(map_analysis_instance.map_data.x_positions)
                        y_min, y_max = min(map_analysis_instance.map_data.y_positions), max(map_analysis_instance.map_data.y_positions)
                        extent = [x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5]
                    
                    # Get colormap
                    colormap = 'viridis'
                    if hasattr(map_analysis_instance, 'colormap_combo') and map_analysis_instance.colormap_combo:
                        try:
                            colormap = map_analysis_instance.colormap_combo.currentText()
                        except:
                            pass
                    
                    # Create improved plot
                    create_improved_map_plot(
                        map_analysis_instance.canvas_manager,
                        map_data,
                        title=title,
                        colormap=colormap,
                        extent=extent,
                        interpolation=interpolation_method
                    )
                
            except Exception as e:
                logger.error(f"Error in improved_update_map: {e}")
                # Fall back to original method
                return original_update_map()
        
        # Replace the method
        map_analysis_instance.update_map = improved_update_map
        
        logger.info("Map plotting fixed with improved canvas management")
        return True
    
    else:
        logger.error("Could not find map figure and canvas attributes")
        return False


# Example usage function
def apply_canvas_fix_to_instance(instance):
    """
    Apply the canvas and colorbar fix to a map analysis instance.
    
    Args:
        instance: TwoDMapAnalysisQt6 instance
    """
    return fix_existing_map_plotting(instance) 