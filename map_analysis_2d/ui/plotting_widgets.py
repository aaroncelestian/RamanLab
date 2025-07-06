"""
Plotting widgets for the map analysis application.

This module contains matplotlib-based plotting widgets for visualizing
spectroscopic data and analysis results.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# NavigationToolbar import moved below with the rest of polarization_ui imports

# Import compact styling from root ui directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from polarization_ui.matplotlib_config import CompactNavigationToolbar, configure_compact_ui, apply_theme
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import logging
from typing import Optional, Tuple
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PySide6.QtCore import Qt, Signal

logger = logging.getLogger(__name__)


class BasePlotWidget(QWidget):
    """Base widget for matplotlib plotting with toolbar."""
    
    def __init__(self, figsize=(12, 8), parent=None):
        super().__init__(parent)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=figsize, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        
        # Apply compact theme and create compact navigation toolbar
        apply_theme('compact')
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        
        # Add to layout
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
        
        # Clear and setup initial plot
        self.clear_plot()
    
    def clear_plot(self):
        """Clear the plot and set up axes."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def draw(self):
        """Refresh the plot."""
        self.figure.tight_layout()
        self.canvas.draw()


class MapPlotWidget(BasePlotWidget):
    """Widget for displaying 2D maps with click interaction."""
    
    # Signal emitted when map is clicked
    map_clicked = Signal(float, float)  # x, y coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, figsize=(14, 10))
        
        # Connect click events and cursor changes
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)
        
        # Current colorbar reference
        self.colorbar = None
        
    def _on_click(self, event):
        """Handle mouse clicks on the map."""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.map_clicked.emit(event.xdata, event.ydata)
            
    def _on_hover(self, event):
        """Handle mouse hover to change cursor."""
        from PySide6.QtCore import Qt
        if event.inaxes == self.ax:
            self.canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
    
    def plot_map(self, data: np.ndarray, extent: Optional[Tuple] = None, 
                 title: str = "2D Map", cmap: str = 'viridis', 
                 xlabel: str = "X Position", ylabel: str = "Y Position",
                 discrete_labels: Optional[list] = None, vmin: Optional[float] = None, vmax: Optional[float] = None):
        """Plot 2D map data."""
        # Validate input data
        if data is None:
            logger.warning("Cannot plot map: data is None")
            return
        
        if not isinstance(data, np.ndarray):
            logger.warning(f"Cannot plot map: data is not numpy array (type: {type(data)})")
            return
            
        if data.size == 0:
            logger.warning("Cannot plot map: data array is empty")
            return
        
        self.ax.clear()
        
        # Use provided vmin/vmax or calculate from data
        if vmin is None or np.isnan(vmin):
            vmin = np.nanmin(data)
        if vmax is None or np.isnan(vmax):
            vmax = np.nanmax(data)
        
        # Plot the map with proper intensity scaling
        im = self.ax.imshow(data, cmap=cmap, aspect='auto', 
                           interpolation='nearest', extent=extent, origin='lower',
                           vmin=vmin, vmax=vmax)
        
        # Remove old colorbar if it exists
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                # Colorbar might already be removed
                pass
            finally:
                self.colorbar = None
        
        # Add new colorbar using the permanent no-shrink solution
        try:
            # Import the permanent colorbar solution
            from polarization_ui.matplotlib_config import add_colorbar_no_shrink
            self.colorbar = add_colorbar_no_shrink(self.figure, im, self.ax)
            
            if self.colorbar is None:
                logger.warning("Permanent colorbar function failed, trying direct method")
                # Direct implementation as final fallback
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.colorbar = self.figure.colorbar(im, cax=cax)
        except Exception as e:
            logger.warning(f"Could not create colorbar with permanent solution: {e}")
            # Final fallback
            try:
                self.colorbar = self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            except Exception as e2:
                logger.warning(f"All colorbar methods failed: {e2}")
                self.colorbar = None
        
        # Handle discrete data labeling (for ML clusters/classifications)
        if self.colorbar is not None:
            try:
                if discrete_labels is not None and len(discrete_labels) > 0:
                    # Get unique values in the data
                    unique_values = np.unique(data[~np.isnan(data)])
                    unique_values = unique_values[unique_values >= 0]  # Remove negative values (e.g., -1 for no data)
                    
                    if len(unique_values) <= len(discrete_labels):
                        # Set colorbar ticks at the center of each discrete value
                        tick_positions = unique_values
                        tick_labels = [discrete_labels[int(val)] if int(val) < len(discrete_labels) else f"Class {int(val)}" 
                                     for val in unique_values]
                        
                        self.colorbar.set_ticks(tick_positions)
                        self.colorbar.set_ticklabels(tick_labels)
                        self.colorbar.set_label('Classification/Cluster')
                    else:
                        self.colorbar.set_label('Value')
                else:
                    # For continuous data, just add a generic label
                    self.colorbar.set_label('Intensity')
            except Exception as e:
                logger.warning(f"Could not set colorbar labels: {e}")
        
        # Set labels and title
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        
        self.draw()


class SpectrumPlotWidget(BasePlotWidget):
    """Widget for displaying individual spectra."""
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, figsize=(14, 4))
        
    def plot_spectrum(self, wavenumbers: np.ndarray, intensities: np.ndarray,
                      title: str = "Spectrum", label: str = None, 
                      color: str = 'blue', clear: bool = True):
        """Plot a spectrum."""
        if clear:
            self.ax.clear()
            
        self.ax.plot(wavenumbers, intensities, color=color, 
                    label=label, linewidth=1)
        
        self.ax.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax.set_ylabel('Intensity')
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        
        if label:
            self.ax.legend()
            
        self.draw()


class SplitMapSpectrumWidget(QWidget):
    """Widget that combines map and spectrum plotting with splitter."""
    
    # Signal emitted when map is clicked
    spectrum_requested = Signal(float, float)  # x, y coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for map and spectrum
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Create map widget
        self.map_widget = MapPlotWidget()
        self.map_widget.map_clicked.connect(self._on_map_clicked)
        
        # Create spectrum widget (initially hidden)
        self.spectrum_widget = SpectrumPlotWidget()
        
        # Add to splitter
        self.splitter.addWidget(self.map_widget)
        self.splitter.addWidget(self.spectrum_widget)
        
        # Initially hide spectrum panel
        self.spectrum_widget.hide()
        
        # Add splitter to main layout
        self.layout.addWidget(self.splitter)
    
    def _on_map_clicked(self, x: float, y: float):
        """Handle map click and emit spectrum request."""
        self.spectrum_requested.emit(x, y)
    
    def show_spectrum_panel(self, show: bool = True):
        """Show or hide the spectrum panel."""
        if show:
            self.spectrum_widget.show()
            # Set sizes: 70% map, 30% spectrum
            total_height = self.splitter.height()
            self.splitter.setSizes([int(total_height * 0.7), int(total_height * 0.3)])
        else:
            self.spectrum_widget.hide()
    
    def plot_map(self, *args, **kwargs):
        """Forward to map widget."""
        return self.map_widget.plot_map(*args, **kwargs)
    
    def plot_spectrum(self, *args, **kwargs):
        """Forward to spectrum widget."""
        return self.spectrum_widget.plot_spectrum(*args, **kwargs)


class PCANMFPlotWidget(BasePlotWidget):
    """
    Widget for displaying PCA and NMF analysis results in a 2x2 layout.
    
    Provides comprehensive visualization including:
    - PCA results (explained variance)
    - PCA clustering scatter plot  
    - NMF components
    - NMF clustering results
    """
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, figsize=(16, 12))
        self.pca_results = None
        self.nmf_results = None
        self.pca_clusters = None
        self.nmf_clusters = None
        self.setup_subplots()
    
    def setup_subplots(self):
        """Set up the 2x2 subplot layout."""
        self.figure.clear()
        
        # Create 2x2 subplot grid
        self.ax_pca_variance = self.figure.add_subplot(2, 2, 1)
        self.ax_pca_scatter = self.figure.add_subplot(2, 2, 2)
        self.ax_nmf_components = self.figure.add_subplot(2, 2, 3)
        self.ax_nmf_scatter = self.figure.add_subplot(2, 2, 4)
        
        # Set titles
        self.ax_pca_variance.set_title('PCA Explained Variance')
        self.ax_pca_scatter.set_title('PCA Clustering Results')
        self.ax_nmf_components.set_title('NMF Components')
        self.ax_nmf_scatter.set_title('NMF Clustering Results')
        
        # Add grid to all subplots
        for ax in [self.ax_pca_variance, self.ax_pca_scatter, 
                   self.ax_nmf_components, self.ax_nmf_scatter]:
            ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()
    
    def plot_pca_results(self, pca_results, pca_clusters=None):
        """
        Plot PCA analysis results.
        
        Args:
            pca_results: Dictionary with PCA analysis results
            pca_clusters: Optional clustering results for PCA components
        """
        self.pca_results = pca_results
        self.pca_clusters = pca_clusters
        
        # Clear PCA plots
        self.ax_pca_variance.clear()
        self.ax_pca_scatter.clear()
        
        if not pca_results or not pca_results.get('success'):
            self._plot_error_message(self.ax_pca_variance, "PCA analysis failed")
            self._plot_error_message(self.ax_pca_scatter, "No PCA clustering data")
            return
        
        # Plot 1: PCA Explained Variance
        explained_variance = pca_results['explained_variance_ratio']
        n_components = len(explained_variance)
        x = range(1, n_components + 1)
        
        self.ax_pca_variance.plot(x, explained_variance, 'bo-', label='Individual', linewidth=2, markersize=6)
        self.ax_pca_variance.plot(x, np.cumsum(explained_variance), 'ro-', label='Cumulative', linewidth=2, markersize=6)
        
        self.ax_pca_variance.set_xlabel('Principal Component')
        self.ax_pca_variance.set_ylabel('Explained Variance Ratio')
        self.ax_pca_variance.set_title('PCA Explained Variance')
        self.ax_pca_variance.legend()
        self.ax_pca_variance.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, (ind, cum) in enumerate(zip(explained_variance, np.cumsum(explained_variance))):
            self.ax_pca_variance.text(i+1, ind + 0.01, f'{ind:.1%}', 
                                    ha='center', va='bottom', fontsize=9)
        
        # Plot 2: PCA Clustering Scatter Plot
        components = pca_results.get('components')
        positions = pca_results.get('positions', None)
        scatter = None
        if components is not None and components.shape[1] >= 2:
            # Use first two principal components for scatter plot
            pc1 = components[:, 0]
            pc2 = components[:, 1]
            
            if pca_clusters is not None and 'labels' in pca_clusters:
                # Plot with cluster colors
                labels = pca_clusters['labels']
                unique_labels = np.unique(labels)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    if label == -1:  # Noise points (if using DBSCAN)
                        color = 'black'
                        marker = 'x'
                        alpha = 0.5
                        label_name = 'Noise'
                    else:
                        color = colors[i % len(colors)]
                        marker = 'o'
                        alpha = 0.7
                        label_name = f'Cluster {label}'
                        
                    mask = labels == label
                    scatter = self.ax_pca_scatter.scatter(pc1[mask], pc2[mask], 
                                              c=[color], marker=marker, alpha=alpha,
                                              label=label_name, s=50)
                
                self.ax_pca_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Plot without clustering information
                scatter = self.ax_pca_scatter.scatter(pc1, pc2, alpha=0.6, s=50)
            
            pc1_var = explained_variance[0] if len(explained_variance) > 0 else 0
            pc2_var = explained_variance[1] if len(explained_variance) > 1 else 0
            
            self.ax_pca_scatter.set_xlabel(f'PC1 ({pc1_var:.1%} variance)')
            self.ax_pca_scatter.set_ylabel(f'PC2 ({pc2_var:.1%} variance)')
            self.ax_pca_scatter.set_title('PCA Clustering Results')
            # Add hover tooltips for XY positions
            if positions is not None and scatter is not None:
                self._add_hover_tooltips(self.ax_pca_scatter, scatter, positions)
        else:
            self._plot_error_message(self.ax_pca_scatter, "Insufficient PCA components\nfor scatter plot")
        
        self.ax_pca_scatter.grid(True, alpha=0.3)
        self._refresh_canvas()
    
    def plot_nmf_results(self, nmf_results, nmf_clusters=None, wavenumbers=None):
        """
        Plot NMF analysis results.
        
        Args:
            nmf_results: Dictionary with NMF analysis results
            nmf_clusters: Optional clustering results for NMF components
            wavenumbers: Optional wavenumber array for component plotting
        """
        self.nmf_results = nmf_results
        self.nmf_clusters = nmf_clusters
        
        # Clear NMF plots
        self.ax_nmf_components.clear()
        self.ax_nmf_scatter.clear()
        
        if not nmf_results or not nmf_results.get('success'):
            self._plot_error_message(self.ax_nmf_components, "NMF analysis failed")
            self._plot_error_message(self.ax_nmf_scatter, "No NMF clustering data")
            return
        
        # Plot 3: NMF Components (Spectral Signatures)
        feature_components = nmf_results.get('feature_components')  # H matrix
        n_components = nmf_results.get('n_components', 0)
        
        if feature_components is not None:
            if wavenumbers is not None and len(wavenumbers) == feature_components.shape[1]:
                x_data = wavenumbers
                x_label = 'Wavenumber (cm⁻¹)'
            else:
                x_data = range(feature_components.shape[1])
                x_label = 'Feature Index'
            
            colors = plt.cm.Set1(np.linspace(0, 1, n_components))
            for i in range(n_components):
                self.ax_nmf_components.plot(x_data, feature_components[i, :], 
                                          color=colors[i], linewidth=1,
                                          label=f'Component {i+1}')
            
            self.ax_nmf_components.set_xlabel(x_label)
            self.ax_nmf_components.set_ylabel('Component Loading')
            self.ax_nmf_components.set_title('NMF Components (Spectral Signatures)')
            self.ax_nmf_components.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            self._plot_error_message(self.ax_nmf_components, "No NMF components\navailable")
        
        # Plot 4: NMF Clustering Scatter Plot
        components = nmf_results.get('components')  # W matrix
        positions = nmf_results.get('positions', None)
        scatter = None
        if components is not None and components.shape[1] >= 2:
            # Use first two NMF components for scatter plot
            comp1 = components[:, 0]
            comp2 = components[:, 1]
            
            if nmf_clusters is not None and 'labels' in nmf_clusters:
                # Plot with cluster colors
                labels = nmf_clusters['labels']
                unique_labels = np.unique(labels)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    if label == -1:  # Noise points (if using DBSCAN)
                        color = 'black'
                        marker = 'x'
                        alpha = 0.5
                        label_name = 'Noise'
                    else:
                        color = colors[i % len(colors)]
                        marker = 'o'
                        alpha = 0.7
                        label_name = f'Cluster {label}'
                        
                    mask = labels == label
                    scatter = self.ax_nmf_scatter.scatter(comp1[mask], comp2[mask], 
                                              c=[color], marker=marker, alpha=alpha,
                                              label=label_name, s=50)
                
                self.ax_nmf_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Plot without clustering information
                scatter = self.ax_nmf_scatter.scatter(comp1, comp2, alpha=0.6, s=50)
            
            self.ax_nmf_scatter.set_xlabel('NMF Component 1')
            self.ax_nmf_scatter.set_ylabel('NMF Component 2')
            self.ax_nmf_scatter.set_title('NMF Clustering Results')
            # Add hover tooltips for XY positions
            if positions is not None and scatter is not None:
                self._add_hover_tooltips(self.ax_nmf_scatter, scatter, positions)
        else:
            self._plot_error_message(self.ax_nmf_scatter, "Insufficient NMF components\nfor scatter plot")
        
        self.ax_nmf_scatter.grid(True, alpha=0.3)
        self._refresh_canvas()
    
    def _plot_error_message(self, ax, message):
        """Plot an error message on the given axis."""
        ax.clear()
        ax.text(0.5, 0.5, message, ha='center', va='center', 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax.set_title(ax.get_title())  # Keep the original title
        ax.axis('off')
    
    def _refresh_canvas(self):
        """Refresh the canvas with proper layout."""
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()
    
    def clear_plot(self):
        """Clear all plots and reset to initial state."""
        self.pca_results = None
        self.nmf_results = None
        self.pca_clusters = None
        self.nmf_clusters = None
        self.setup_subplots()

    def _add_hover_tooltips(self, ax, scatter, positions):
        """Add mouse hover tooltips to a scatter plot showing XY map positions."""
        from matplotlib.backend_bases import MouseEvent
        import matplotlib.pyplot as plt
        annot = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        fig = ax.figure
        coords = scatter.get_offsets()
        def update_annot(ind):
            idx = ind["ind"][0]
            pos = positions[idx]
            annot.xy = coords[idx]
            annot.set_text(f"XY: {pos}")
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                elif vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        fig.canvas.mpl_connect("motion_notify_event", hover)
