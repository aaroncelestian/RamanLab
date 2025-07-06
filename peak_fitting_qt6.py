#!/usr/bin/env python3
"""
Enhanced Peak Fitting and Spectral Deconvolution Module for RamanLab Qt6
Includes:
• Component separation
• Overlapping peak resolution  
• Principal component analysis
• Non-negative matrix factorization
• Direct file loading capability
"""

import numpy as np
import sys
from pathlib import Path

# Import Qt6
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, 
    QPushButton, QLineEdit, QTextEdit, QSlider, QCheckBox, QComboBox,
    QGroupBox, QSplitter, QMessageBox, QProgressBar, QSpinBox, 
    QDoubleSpinBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QFrame, QGridLayout, QListWidget, QListWidgetItem,
    QMenuBar, QMenu, QFileDialog, QStatusBar, QApplication, QStackedWidget,
    QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QAction

# Import matplotlib for Qt6
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from polarization_ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar, apply_theme

# Scientific computing
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import pandas as pd
import time

# File loading utilities
try:
    from utils.file_loaders import SpectrumLoader, load_spectrum_file
    FILE_LOADING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: File loading utilities not available: {e}")
    FILE_LOADING_AVAILABLE = False

# Centralized peak fitting imports
try:
    from core.peak_fitting import PeakFitter, PeakData, auto_find_peaks, baseline_correct_spectrum
    from core.peak_fitting_ui import BackgroundControlsWidget, PeakFittingControlsWidget
    CENTRALIZED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Centralized peak fitting not available: {e}")
    CENTRALIZED_AVAILABLE = False

# Machine learning imports for advanced features
try:
    from sklearn.decomposition import PCA, NMF
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# Enhanced PeakFitter with baseline_als support
class EnhancedPeakFitter(PeakFitter if CENTRALIZED_AVAILABLE else object):
    """Enhanced peak fitter with baseline correction capabilities."""
    
    def __init__(self):
        if CENTRALIZED_AVAILABLE:
            super().__init__()
    
    @staticmethod
    def baseline_als(y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares Smoothing for baseline correction.
        
        Parameters:
        -----------
        y : array-like
            Input spectrum.
        lam : float
            Smoothness parameter (default: 1e5).
        p : float
            Asymmetry parameter (default: 0.01).
        niter : int
            Number of iterations (default: 10).
            
        Returns:
        --------
        array-like
            Estimated baseline.
        """
        L = len(y)
        D = csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        
        for i in range(niter):
            W = csc_matrix((w, (np.arange(L), np.arange(L))))
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        
        return z
    
    # Ensure backward compatibility for standalone operation
    @staticmethod
    def gaussian(x, amp, cen, wid):
        """Gaussian peak function."""
        if CENTRALIZED_AVAILABLE:
            return PeakFitter.gaussian(x, amp, cen, wid)
        else:
            # Fallback implementation
            width = abs(wid) + 1e-10
            return amp * np.exp(-((x - cen) / width) ** 2)
    
    @staticmethod
    def lorentzian(x, amp, cen, wid):
        """Lorentzian peak function."""
        if CENTRALIZED_AVAILABLE:
            return PeakFitter.lorentzian(x, amp, cen, wid)
        else:
            # Fallback implementation
            width = abs(wid) + 1e-10
            return amp * (width**2) / ((x - cen)**2 + width**2)
    
    @staticmethod
    def pseudo_voigt(x, amp, cen, wid, eta=0.5):
        """Pseudo-Voigt peak function."""
        if CENTRALIZED_AVAILABLE:
            return PeakFitter.pseudo_voigt(x, amp, cen, wid, eta)
        else:
            # Fallback implementation
            eta = np.clip(eta, 0, 1)
            gaussian_part = EnhancedPeakFitter.gaussian(x, 1.0, cen, wid)
            lorentzian_part = EnhancedPeakFitter.lorentzian(x, 1.0, cen, wid)
            return amp * ((1 - eta) * gaussian_part + eta * lorentzian_part)


class BatchProcessingMonitor(QDialog):
    """Real-time monitor window for batch processing progress."""
    
    # Signals for communication
    processing_cancelled = Signal()
    processing_paused = Signal()
    processing_resumed = Signal()
    save_results_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Monitor")
        self.setGeometry(100, 100, 1200, 800)
        self.setModal(False)  # Allow interaction with main window
        
        # Processing state
        self.is_paused = False
        self.is_cancelled = False
        
        # Navigation state for reviewing results
        self.stored_results = []  # Store all processing results for navigation
        self.current_result_index = 0
        self.is_completed = False
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the monitoring UI."""
        layout = QVBoxLayout(self)
        
        # Header with progress info
        header_layout = QHBoxLayout()
        
        # Progress information
        self.progress_label = QLabel("Starting batch processing...")
        self.progress_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(self.progress_label)
        
        header_layout.addStretch()
        
        # Control buttons
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        header_layout.addWidget(self.pause_btn)
        
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        header_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(header_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Plot area
        self.setup_plot_area(layout)
        
        # Status information
        self.setup_status_area(layout)
        
    def setup_plot_area(self, layout):
        """Setup the real-time plotting area."""
        # Create matplotlib figure with extra height for proper spacing
        self.figure = Figure(figsize=(12, 7))
        self.canvas = FigureCanvas(self.figure)
        
        # Create optimized layout: plots fill most space, legend on the side
        # Left column: spectrum and residuals, Right column: legend spanning full height
        # Increased height ratio for main plot and reduced spacing
        gs = self.figure.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[5, 1], 
                                    hspace=0.05, wspace=0.2)
        self.ax_main = self.figure.add_subplot(gs[0, 0])      # Main spectrum plot (clean, no legend)
        self.ax_progress = self.figure.add_subplot(gs[1, 0], sharex=self.ax_main)  # Residuals plot
        self.ax_legend = self.figure.add_subplot(gs[:, 1])    # Legend area spanning full right column
        
        # Configure main plot - clean without legend and x-axis label
        self.ax_main.set_ylabel("Intensity (a.u.)")
        self.ax_main.set_title("Current Spectrum")
        self.ax_main.grid(True, alpha=0.3)
        # Remove x-axis labels for main plot since residuals plot has them
        self.ax_main.tick_params(axis='x', labelbottom=False)
        
        # Configure residuals plot - clean and minimal
        self.ax_progress.set_xlabel("Wavenumber (cm⁻¹)")
        # No y-label or title for cleaner appearance
        self.ax_progress.grid(True, alpha=0.3)
        
        # Configure dedicated legend area - spans full right column
        self.ax_legend.set_title("Legend", fontsize=11, pad=15)
        self.ax_legend.axis('off')
        
        # Connect zoom/pan events to keep plots synchronized
        self.ax_main.callbacks.connect('xlim_changed', self._on_xlim_changed)
        self.ax_progress.callbacks.connect('xlim_changed', self._on_xlim_changed)
        
        # Use constrained_layout for better automatic spacing
        self.figure.set_constrained_layout(True)
        
        # Add canvas to layout
        layout.addWidget(self.canvas)
        
        # Add matplotlib navigation toolbar for interactive zooming/panning
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                spacing: 3px;
            }
            QToolButton {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 2px;
                margin: 1px;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
            }
            QToolButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        layout.addWidget(self.toolbar)
        
        # Initialize data storage for progress tracking
        self.file_indices = []
        self.peaks_counts = []
        self.processing_times = []
        
    def _on_xlim_changed(self, ax):
        """Callback to keep spectrum and residuals plots synchronized during interactive zoom/pan."""
        if hasattr(self, 'ax_main') and hasattr(self, 'ax_progress'):
            # Prevent infinite recursion by temporarily disconnecting callbacks
            self.ax_main.callbacks.disconnect('xlim_changed', self._on_xlim_changed)
            self.ax_progress.callbacks.disconnect('xlim_changed', self._on_xlim_changed)
            
            # Get the current x-limits from the changed axis
            xlims = ax.get_xlim()
            
            # Apply the same limits to both axes
            self.ax_main.set_xlim(xlims)
            self.ax_progress.set_xlim(xlims)
            
            # Redraw the canvas
            self.canvas.draw_idle()
            
            # Reconnect callbacks
            self.ax_main.callbacks.connect('xlim_changed', self._on_xlim_changed)
            self.ax_progress.callbacks.connect('xlim_changed', self._on_xlim_changed)
        
    def setup_status_area(self, layout):
        """Setup the status information area."""
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        
        # Current file info
        file_info_layout = QVBoxLayout()
        file_info_layout.addWidget(QLabel("Current File:"))
        self.current_file_label = QLabel("None")
        self.current_file_label.setWordWrap(True)
        file_info_layout.addWidget(self.current_file_label)
        status_layout.addLayout(file_info_layout)
        
        # Current region info
        region_layout = QVBoxLayout()
        region_layout.addWidget(QLabel("Current Region:"))
        self.region_label = QLabel("None")
        region_layout.addWidget(self.region_label)
        status_layout.addLayout(region_layout)
        
        layout.addWidget(status_frame)
        
        # Navigation controls (initially hidden)
        self.setup_navigation_controls(layout)
        
    def setup_navigation_controls(self, layout):
        """Setup navigation controls for reviewing results after processing."""
        self.nav_frame = QFrame()
        self.nav_frame.setFrameStyle(QFrame.StyledPanel)
        self.nav_frame.setVisible(False)  # Initially hidden
        nav_layout = QVBoxLayout(self.nav_frame)
        
        # Navigation header
        nav_header = QLabel("📊 Browse Results")
        nav_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #4CAF50;")
        nav_layout.addWidget(nav_header)
        
        # Navigation controls
        nav_controls_layout = QHBoxLayout()
        
        # Previous button
        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.setToolTip("View previous spectrum")
        self.prev_btn.clicked.connect(self.show_previous_result)
        nav_controls_layout.addWidget(self.prev_btn)
        
        # Current result info
        self.result_info_label = QLabel("Result 1 of 1")
        self.result_info_label.setAlignment(Qt.AlignCenter)
        self.result_info_label.setStyleSheet("font-weight: bold;")
        nav_controls_layout.addWidget(self.result_info_label)
        
        # Next button
        self.next_btn = QPushButton("Next ➡️")
        self.next_btn.setToolTip("View next spectrum")
        self.next_btn.clicked.connect(self.show_next_result)
        nav_controls_layout.addWidget(self.next_btn)
        
        nav_layout.addLayout(nav_controls_layout)
        
        # File selector dropdown
        file_selector_layout = QHBoxLayout()
        file_selector_layout.addWidget(QLabel("Quick Jump:"))
        
        self.file_selector = QComboBox()
        self.file_selector.setToolTip("Select a file to view its results")
        self.file_selector.currentIndexChanged.connect(self.on_file_selected)
        file_selector_layout.addWidget(self.file_selector)
        
        nav_layout.addLayout(file_selector_layout)
        
        layout.addWidget(self.nav_frame)
        
    def toggle_pause(self):
        """Toggle pause/resume processing."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.setText("▶️ Resume")
            self.processing_paused.emit()
        else:
            self.pause_btn.setText("⏸️ Pause")
            self.processing_resumed.emit()
    
    def cancel_processing(self):
        """Cancel the batch processing."""
        self.is_cancelled = True
        self.processing_cancelled.emit()
        self.close()
        
    def update_progress(self, current_file, total_files, file_path):
        """Update the overall progress."""
        self.progress_bar.setMaximum(total_files)
        self.progress_bar.setValue(current_file + 1)  # Fix: Use count, not index
        
        progress_text = f"Processing file {current_file + 1} of {total_files}"
        self.progress_label.setText(progress_text)
        
        self.current_file_label.setText(Path(file_path).name)
        
    def update_region_info(self, region_start, region_end, region_index, total_regions):
        """Update current region information."""
        region_text = f"Region {region_index + 1}/{total_regions}: {region_start:.0f}-{region_end:.0f} cm⁻¹"
        self.region_label.setText(region_text)
        
    def update_spectrum_plot(self, wavenumbers, intensities, background=None, peaks=None, region_start=None, region_end=None, full_wavenumbers=None, full_intensities=None, fitted_peaks=None, residuals=None):
        """Update the real-time spectrum plot and residuals."""
        self.ax_main.clear()
        
        # Plot original spectrum
        self.ax_main.plot(wavenumbers, intensities, 'b-', label='Spectrum', linewidth=1)
        
        # Plot background if available
        corrected = intensities
        if background is not None:
            corrected = intensities - background
            self.ax_main.plot(wavenumbers, corrected, 'g-', label='Background Corrected', linewidth=1)
            self.ax_main.plot(wavenumbers, background, 'r--', label='Background', alpha=0.7)
        
        # Plot fitted peaks if available
        if fitted_peaks is not None:
            self.ax_main.plot(wavenumbers, fitted_peaks, 'm-', label='Fitted Peaks', linewidth=2, alpha=0.8)
        
        # Plot detected peaks
        if peaks is not None and len(peaks) > 0:
            peak_intensities = intensities[peaks] if background is None else (intensities - background)[peaks]
            self.ax_main.plot(wavenumbers[peaks], peak_intensities, 'ro', 
                            markersize=8, label=f'Peaks ({len(peaks)})')
            
            # Add peak labels
            for i, peak_idx in enumerate(peaks):
                self.ax_main.annotate(f'{wavenumbers[peak_idx]:.0f}', 
                                    (wavenumbers[peak_idx], peak_intensities[i]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, alpha=0.8)
        
        # Highlight region if specified
        if region_start is not None and region_end is not None:
            self.ax_main.axvspan(region_start, region_end, alpha=0.2, color='yellow', label='Analysis Region')
        
        self.ax_main.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_main.set_ylabel("Intensity (a.u.)")
        self.ax_main.set_title("Current Spectrum")
        # No legend on main plot - it's now in dedicated area
        self.ax_main.grid(True, alpha=0.3)
        
        # Update the external legend
        self._update_external_legend()
        
        # Update residuals plot
        self.update_residuals_plot(wavenumbers, corrected, peaks, region_start, region_end, full_wavenumbers, full_intensities, fitted_peaks, residuals)
    
    def _update_external_legend(self):
        """Update the external legend area with current plot elements."""
        self.ax_legend.clear()
        self.ax_legend.set_title("Legend", fontsize=11, pad=15)
        self.ax_legend.axis('off')
        
        # Get legend elements from main plot
        handles, labels = self.ax_main.get_legend_handles_labels()
        
        if handles and labels:
            # Create legend in the dedicated area with better spacing
            legend = self.ax_legend.legend(handles, labels, loc='upper center', 
                                         frameon=False, fontsize=10,
                                         bbox_to_anchor=(0.5, 0.9))
            
            # Style the legend text with better readability
            for text in legend.get_texts():
                text.set_fontsize(10)
                text.set_color('#333333')
    
    def update_residuals_plot(self, wavenumbers, corrected_intensities, peaks=None, region_start=None, region_end=None, full_wavenumbers=None, full_intensities=None, fitted_peaks=None, residuals=None):
        """Update the residuals plot showing fit quality."""
        self.ax_progress.clear()
        
        # Use actual residuals if available, otherwise show background-corrected signal
        if residuals is not None:
            # Plot the actual residuals (measured - fitted)
            self.ax_progress.plot(wavenumbers, residuals, 'purple', linewidth=1, alpha=0.8)
            
            # Add zero line for reference
            self.ax_progress.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
        else:
            # Fall back to showing background-corrected signal
            residual_wave = wavenumbers
            display_residuals = corrected_intensities
            
            # Plot residuals
            if len(display_residuals) > 0:
                self.ax_progress.plot(residual_wave, display_residuals, 'purple', linewidth=1, alpha=0.8)
                
                # Add zero line for reference
                self.ax_progress.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight fitting regions
        if region_start is not None and region_end is not None:
            self.ax_progress.axvspan(region_start, region_end, alpha=0.1, color='yellow')
            
        # Mark peak positions if available
        if peaks is not None and len(peaks) > 0:
            peak_wavenumbers = wavenumbers[peaks]
            for peak_wave in peak_wavenumbers:
                self.ax_progress.axvline(x=peak_wave, color='red', alpha=0.3, linestyle=':', linewidth=1)
        
        self.ax_progress.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_progress.set_ylabel("Intensity (a.u.)")
        self.ax_progress.grid(True, alpha=0.3)
        
        # X-axis alignment is automatically handled by sharex in setup
        # Just ensure limits are synchronized if needed
        if hasattr(self, 'ax_main'):
            main_xlims = self.ax_main.get_xlim()
            self.ax_progress.set_xlim(main_xlims)
        
        # Refresh the canvas
        self.canvas.draw()
        
    def update_statistics(self, file_index, peaks_found, processing_time, total_peaks, total_files):
        """Update processing statistics."""
        # Store data for progress plot
        self.file_indices.append(file_index)
        self.peaks_counts.append(peaks_found)
        self.processing_times.append(processing_time)
        
        # Statistics display has been removed to give more space to plots
        # Statistics are still tracked internally for export purposes
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        # Processing stats are now shown in the status area below plots
        # No need for separate stats plot area
        
        # Don't overwrite the residuals plot - it should always show residuals
        # The residuals plot is updated by update_residuals_plot() called from update_spectrum_plot()
        
        # Refresh the canvas
        self.canvas.draw()
        
    def wait_if_paused(self):
        """Wait while processing is paused."""
        while self.is_paused and not self.is_cancelled:
            QApplication.processEvents()
            time.sleep(0.1)
    
    def processing_completed(self):
        """Signal that processing is complete and update buttons."""
        self.progress_label.setText("✅ Batch processing completed successfully!")
        self.progress_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        # Ensure progress bar shows 100%
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        # Update buttons for completion state
        self.pause_btn.setText("💾 Save CSV")
        self.pause_btn.setToolTip("Save comprehensive CSV results")
        self.pause_btn.clicked.disconnect()  # Disconnect old handler
        self.pause_btn.clicked.connect(self.export_to_csv)
        
        self.cancel_btn.setText("✖️ Close")
        self.cancel_btn.setToolTip("Close the batch processing monitor")
        self.cancel_btn.clicked.disconnect()  # Disconnect old handler
        self.cancel_btn.clicked.connect(self.close)
        
        # Update progress bar to show completion
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        
        # Show navigation controls for reviewing results
        if self.stored_results:
            self.setup_navigation_display()
    
    def save_results(self):
        """Request to save results."""
        self.save_results_requested.emit()
    
    def export_to_csv(self):
        """Export comprehensive batch results to CSV files."""
        if not self.stored_results:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return
        
        # Get base filename and location from user
        base_file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Results - Choose Base Filename", 
            "batch_analysis_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not base_file_path:
            return
        
        try:
            self._export_comprehensive_csv_with_custom_names(base_file_path)
            save_dir = Path(base_file_path).parent
            QMessageBox.information(self, "Export Complete", 
                                  f"CSV files exported successfully to:\n{save_dir}\n\n"
                                  f"Files created:\n"
                                  f"• Summary with peak parameters\n"
                                  f"• Detailed peak parameters\n" 
                                  f"• Full spectral data")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {str(e)}")
    
    def _export_comprehensive_csv_with_custom_names(self, base_file_path):
        """Export comprehensive CSV files with user-chosen base filename."""
        base_path = Path(base_file_path)
        base_name = base_path.stem  # Get filename without extension
        save_dir = base_path.parent
        
        # Generate descriptive filenames based on user's choice
        summary_file = save_dir / f"{base_name}_summary.csv"
        peak_params_file = save_dir / f"{base_name}_peak_parameters.csv"
        spectral_data_file = save_dir / f"{base_name}_spectral_data.csv"
        
        if not self.stored_results:
            return
        
        # Create a unified wavenumber grid (use the first spectrum as reference)
        reference_wavenumbers = self.stored_results[0]['wavenumbers']
        
        # Initialize the main DataFrame with wavenumbers
        csv_data = {'Wavenumber_cm-1': reference_wavenumbers}
        
        # Track peak parameters for separate summary
        peak_summary = []
        
        for i, result in enumerate(self.stored_results):
            filename = result['filename']
            wavenumbers = result['wavenumbers']
            intensities = result['intensities']
            background = result.get('background')
            fitted_peaks = result.get('fitted_peaks')
            residuals = result.get('residuals')
            peaks = result.get('peaks')
            fit_params = result.get('fit_params')
            total_r2 = result.get('total_r2')
            
            # Create clean filename for column headers (remove extension and special chars)
            clean_name = Path(filename).stem.replace(' ', '_').replace('-', '_')
            
            # Interpolate all data to the reference wavenumber grid
            raw_interp = np.interp(reference_wavenumbers, wavenumbers, intensities)
            csv_data[f'{clean_name}_Raw'] = raw_interp
            
            # Fitted data (if available)
            if fitted_peaks is not None:
                fitted_interp = np.interp(reference_wavenumbers, wavenumbers, fitted_peaks)
                csv_data[f'{clean_name}_Fitted'] = fitted_interp
            else:
                csv_data[f'{clean_name}_Fitted'] = np.zeros_like(reference_wavenumbers)
            
            # Background (if available)
            if background is not None:
                bg_interp = np.interp(reference_wavenumbers, wavenumbers, background)
                csv_data[f'{clean_name}_Background'] = bg_interp
            else:
                csv_data[f'{clean_name}_Background'] = np.zeros_like(reference_wavenumbers)
            
            # Residuals/Difference (if available)
            if residuals is not None:
                resid_interp = np.interp(reference_wavenumbers, wavenumbers, residuals)
                csv_data[f'{clean_name}_Difference'] = resid_interp
            else:
                # Fallback: raw - background if no residuals available
                if background is not None:
                    bg_corrected = intensities - background
                    diff_interp = np.interp(reference_wavenumbers, wavenumbers, bg_corrected)
                    csv_data[f'{clean_name}_Difference'] = diff_interp
                else:
                    csv_data[f'{clean_name}_Difference'] = np.zeros_like(reference_wavenumbers)
            
            # Collect peak summary data
            if peaks is not None and fit_params is not None:
                n_peaks = len(peaks)
                for j in range(n_peaks):
                    if j * 3 + 2 < len(fit_params):
                        amplitude = fit_params[j * 3]
                        center = fit_params[j * 3 + 1]
                        width = fit_params[j * 3 + 2]
                        
                        # Calculate FWHM and area
                        fwhm = width * 2 * np.sqrt(2 * np.log(2))
                        area = amplitude * width * np.sqrt(2 * np.pi)
                        
                        peak_summary.append({
                            'Filename': filename,
                            'Peak_Number': j + 1,
                            'Center_cm-1': center,
                            'Amplitude': amplitude,
                            'Width_Sigma': width,
                            'FWHM': fwhm,
                            'Area': area,
                            'Total_R2': total_r2 if j == 0 else '',  # Only show R² for first peak
                            'Peak_Height_Raw': intensities[peaks[j]] if j < len(peaks) else np.nan
                        })
        
        # Save main CSV with 4 columns per spectrum
        main_df = pd.DataFrame(csv_data)
        main_df.to_csv(spectral_data_file, index=False)
        print(f"✅ Saved spectral data: {spectral_data_file}")
        
        # Save peak parameters summary
        if peak_summary:
            peak_df = pd.DataFrame(peak_summary)
            peak_df.to_csv(peak_params_file, index=False)
            print(f"✅ Saved peak parameters: {peak_params_file}")
        
        # Save enhanced processing summary with individual peak parameters
        summary_data = []
        max_peaks = 0
        
        # First pass: determine the maximum number of peaks for column creation
        for result in self.stored_results:
            peaks = result.get('peaks')
            n_peaks = len(peaks) if peaks is not None else 0
            max_peaks = max(max_peaks, n_peaks)
        
        # Second pass: create summary data with peak parameters as columns
        for result in self.stored_results:
            peaks = result.get('peaks')
            fit_params = result.get('fit_params')
            total_r2 = result.get('total_r2')
            n_peaks = len(peaks) if peaks is not None else 0
            
            # Base summary info
            summary_row = {
                'Filename': result['filename'],
                'Number_of_Peaks': n_peaks,
                'Total_R2': total_r2,
                'Fitting_Success': 'Yes' if total_r2 is not None and total_r2 > 0 else 'No',
                'Region_Start': result.get('region_start', 'Full'),
                'Region_End': result.get('region_end', 'Spectrum')
            }
            
            # Add individual peak parameters as columns
            if fit_params is not None and n_peaks > 0:
                for i in range(max_peaks):
                    if i < n_peaks and i * 3 + 2 < len(fit_params):
                        amplitude = fit_params[i * 3]
                        center = fit_params[i * 3 + 1]
                        width = fit_params[i * 3 + 2]
                        
                        # Calculate FWHM
                        fwhm = width * 2 * np.sqrt(2 * np.log(2))
                        
                        # Add peak parameters
                        summary_row[f'Peak{i+1}_Amplitude'] = amplitude
                        summary_row[f'Peak{i+1}_Position_cm-1'] = center
                        summary_row[f'Peak{i+1}_FWHM'] = fwhm
                        summary_row[f'Peak{i+1}_Width_Sigma'] = width
                    else:
                        # Fill with NaN for missing peaks
                        summary_row[f'Peak{i+1}_Amplitude'] = np.nan
                        summary_row[f'Peak{i+1}_Position_cm-1'] = np.nan
                        summary_row[f'Peak{i+1}_FWHM'] = np.nan
                        summary_row[f'Peak{i+1}_Width_Sigma'] = np.nan
            else:
                # No fit parameters available - fill with NaN
                for i in range(max_peaks):
                    summary_row[f'Peak{i+1}_Amplitude'] = np.nan
                    summary_row[f'Peak{i+1}_Position_cm-1'] = np.nan
                    summary_row[f'Peak{i+1}_FWHM'] = np.nan
                    summary_row[f'Peak{i+1}_Width_Sigma'] = np.nan
            
            summary_data.append(summary_row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Saved enhanced processing summary: {summary_file}")

    def _export_comprehensive_csv(self, save_dir):
        """Export comprehensive CSV with 4 columns per spectrum (Raw, Fitted, Background, Difference). LEGACY METHOD."""
        save_path = Path(save_dir)
        
        if not self.stored_results:
            return
        
        # Create a unified wavenumber grid (use the first spectrum as reference)
        reference_wavenumbers = self.stored_results[0]['wavenumbers']
        
        # Initialize the main DataFrame with wavenumbers
        csv_data = {'Wavenumber_cm-1': reference_wavenumbers}
        
        # Track peak parameters for separate summary
        peak_summary = []
        
        for i, result in enumerate(self.stored_results):
            filename = result['filename']
            wavenumbers = result['wavenumbers']
            intensities = result['intensities']
            background = result.get('background')
            fitted_peaks = result.get('fitted_peaks')
            residuals = result.get('residuals')
            peaks = result.get('peaks')
            fit_params = result.get('fit_params')
            total_r2 = result.get('total_r2')
            
            # Create clean filename for column headers (remove extension and special chars)
            clean_name = Path(filename).stem.replace(' ', '_').replace('-', '_')
            
            # Interpolate all data to the reference wavenumber grid
            raw_interp = np.interp(reference_wavenumbers, wavenumbers, intensities)
            csv_data[f'{clean_name}_Raw'] = raw_interp
            
            # Fitted data (if available)
            if fitted_peaks is not None:
                fitted_interp = np.interp(reference_wavenumbers, wavenumbers, fitted_peaks)
                csv_data[f'{clean_name}_Fitted'] = fitted_interp
            else:
                csv_data[f'{clean_name}_Fitted'] = np.zeros_like(reference_wavenumbers)
            
            # Background (if available)
            if background is not None:
                bg_interp = np.interp(reference_wavenumbers, wavenumbers, background)
                csv_data[f'{clean_name}_Background'] = bg_interp
            else:
                csv_data[f'{clean_name}_Background'] = np.zeros_like(reference_wavenumbers)
            
            # Residuals/Difference (if available)
            if residuals is not None:
                resid_interp = np.interp(reference_wavenumbers, wavenumbers, residuals)
                csv_data[f'{clean_name}_Difference'] = resid_interp
            else:
                # Fallback: raw - background if no residuals available
                if background is not None:
                    bg_corrected = intensities - background
                    diff_interp = np.interp(reference_wavenumbers, wavenumbers, bg_corrected)
                    csv_data[f'{clean_name}_Difference'] = diff_interp
                else:
                    csv_data[f'{clean_name}_Difference'] = np.zeros_like(reference_wavenumbers)
            
            # Collect peak summary data
            if peaks is not None and fit_params is not None:
                n_peaks = len(peaks)
                for j in range(n_peaks):
                    if j * 3 + 2 < len(fit_params):
                        amplitude = fit_params[j * 3]
                        center = fit_params[j * 3 + 1]
                        width = fit_params[j * 3 + 2]
                        
                        # Calculate FWHM and area
                        fwhm = width * 2 * np.sqrt(2 * np.log(2))
                        area = amplitude * width * np.sqrt(2 * np.pi)
                        
                        peak_summary.append({
                            'Filename': filename,
                            'Peak_Number': j + 1,
                            'Center_cm-1': center,
                            'Amplitude': amplitude,
                            'Width_Sigma': width,
                            'FWHM': fwhm,
                            'Area': area,
                            'Total_R2': total_r2 if j == 0 else '',  # Only show R² for first peak
                            'Peak_Height_Raw': intensities[peaks[j]] if j < len(peaks) else np.nan
                        })
        
        # Save main CSV with 4 columns per spectrum
        main_df = pd.DataFrame(csv_data)
        main_df.to_csv(save_path / 'batch_spectral_data.csv', index=False)
        print(f"✅ Saved spectral data: {len(self.stored_results)} spectra × 4 columns = {len(self.stored_results) * 4} data columns")
        
        # Save peak parameters summary
        if peak_summary:
            peak_df = pd.DataFrame(peak_summary)
            peak_df.to_csv(save_path / 'batch_peak_parameters.csv', index=False)
            print(f"✅ Saved peak parameters: {len(peak_summary)} peaks")
        
        # Save enhanced processing summary with individual peak parameters
        summary_data = []
        max_peaks = 0
        
        # First pass: determine the maximum number of peaks for column creation
        for result in self.stored_results:
            peaks = result.get('peaks')
            n_peaks = len(peaks) if peaks is not None else 0
            max_peaks = max(max_peaks, n_peaks)
        
        # Second pass: create summary data with peak parameters as columns
        for result in self.stored_results:
            peaks = result.get('peaks')
            fit_params = result.get('fit_params')
            total_r2 = result.get('total_r2')
            n_peaks = len(peaks) if peaks is not None else 0
            
            # Base summary info
            summary_row = {
                'Filename': result['filename'],
                'Number_of_Peaks': n_peaks,
                'Total_R2': total_r2,
                'Fitting_Success': 'Yes' if total_r2 is not None and total_r2 > 0 else 'No',
                'Region_Start': result.get('region_start', 'Full'),
                'Region_End': result.get('region_end', 'Spectrum')
            }
            
            # Add individual peak parameters as columns
            if fit_params is not None and n_peaks > 0:
                for i in range(max_peaks):
                    if i < n_peaks and i * 3 + 2 < len(fit_params):
                        amplitude = fit_params[i * 3]
                        center = fit_params[i * 3 + 1]
                        width = fit_params[i * 3 + 2]
                        
                        # Calculate FWHM
                        fwhm = width * 2 * np.sqrt(2 * np.log(2))
                        
                        # Add peak parameters
                        summary_row[f'Peak{i+1}_Amplitude'] = amplitude
                        summary_row[f'Peak{i+1}_Position_cm-1'] = center
                        summary_row[f'Peak{i+1}_FWHM'] = fwhm
                        summary_row[f'Peak{i+1}_Width_Sigma'] = width
                    else:
                        # Fill with NaN for missing peaks
                        summary_row[f'Peak{i+1}_Amplitude'] = np.nan
                        summary_row[f'Peak{i+1}_Position_cm-1'] = np.nan
                        summary_row[f'Peak{i+1}_FWHM'] = np.nan
                        summary_row[f'Peak{i+1}_Width_Sigma'] = np.nan
            else:
                # No fit parameters available - fill with NaN
                for i in range(max_peaks):
                    summary_row[f'Peak{i+1}_Amplitude'] = np.nan
                    summary_row[f'Peak{i+1}_Position_cm-1'] = np.nan
                    summary_row[f'Peak{i+1}_FWHM'] = np.nan
                    summary_row[f'Peak{i+1}_Width_Sigma'] = np.nan
            
            summary_data.append(summary_row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(save_path / 'batch_summary.csv', index=False)
            print(f"✅ Saved enhanced processing summary: {len(summary_data)} files with up to {max_peaks} peaks per file")
    
    def processing_failed(self, error_message):
        """Signal that processing failed."""
        self.progress_label.setText(f"❌ Batch processing failed: {error_message}")
        self.progress_label.setStyleSheet("color: #f44336; font-weight: bold;")
        
        # Update buttons for failure state
        self.pause_btn.setText("💾 Save Partial")
        self.pause_btn.setToolTip("Save partial results to pickle file")
        self.pause_btn.clicked.disconnect()  # Disconnect old handler
        self.pause_btn.clicked.connect(self.save_results)
        
        self.cancel_btn.setText("✖️ Close")
        self.cancel_btn.setToolTip("Close the batch processing monitor")
        self.cancel_btn.clicked.disconnect()  # Disconnect old handler
        self.cancel_btn.clicked.connect(self.close)
        
        # Update progress bar to show failure
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
        
        # Show navigation if we have results
        if self.stored_results:
            self.setup_navigation_display()
    
    def store_processing_result(self, wavenumbers, intensities, background, peaks, region_start, region_end, filename, full_wavenumbers=None, full_intensities=None, fitted_peaks=None, residuals=None, fit_params=None, total_r2=None):
        """Store a processing result for later navigation."""
        result = {
            'wavenumbers': wavenumbers,
            'intensities': intensities,
            'background': background,
            'peaks': peaks,
            'fitted_peaks': fitted_peaks,
            'residuals': residuals,
            'region_start': region_start,
            'region_end': region_end,
            'filename': filename,
            'full_wavenumbers': full_wavenumbers,
            'full_intensities': full_intensities,
            'fit_params': fit_params,
            'total_r2': total_r2
        }
        self.stored_results.append(result)
    
    def setup_navigation_display(self):
        """Setup navigation display when processing is complete."""
        if not self.stored_results:
            return
            
        self.is_completed = True
        self.nav_frame.setVisible(True)
        
        # Populate file selector
        self.file_selector.clear()
        for i, result in enumerate(self.stored_results):
            self.file_selector.addItem(f"{i+1}. {result['filename']}")
        
        # Update navigation state
        self.current_result_index = 0
        self.update_navigation_state()
        self.show_result(0)
        
        # Change plot title
        self.ax_main.set_title("Review Results - Navigate through processed spectra")
        
    def update_navigation_state(self):
        """Update navigation button states and labels."""
        if not self.stored_results:
            return
            
        total_results = len(self.stored_results)
        self.result_info_label.setText(f"Result {self.current_result_index + 1} of {total_results}")
        
        # Update button states
        self.prev_btn.setEnabled(self.current_result_index > 0)
        self.next_btn.setEnabled(self.current_result_index < total_results - 1)
        
        # Update file selector
        self.file_selector.setCurrentIndex(self.current_result_index)
    
    def show_previous_result(self):
        """Show the previous result."""
        if self.current_result_index > 0:
            self.current_result_index -= 1
            self.show_result(self.current_result_index)
            self.update_navigation_state()
    
    def show_next_result(self):
        """Show the next result."""
        if self.current_result_index < len(self.stored_results) - 1:
            self.current_result_index += 1
            self.show_result(self.current_result_index)
            self.update_navigation_state()
    
    def on_file_selected(self, index):
        """Handle file selection from dropdown."""
        if 0 <= index < len(self.stored_results):
            self.current_result_index = index
            self.show_result(index)
            self.update_navigation_state()
    
    def show_result(self, index):
        """Display a specific result."""
        if not (0 <= index < len(self.stored_results)):
            return
            
        result = self.stored_results[index]
        
        # Update the spectrum plot with stored result
        self.update_spectrum_plot(
            result['wavenumbers'],
            result['intensities'],
            result['background'],
            result['peaks'],
            result['region_start'],
            result['region_end'],
            result.get('full_wavenumbers'),
            result.get('full_intensities'),
            result.get('fitted_peaks'),
            result.get('residuals')
        )
        
        # Update file info
        self.current_file_label.setText(result['filename'])
        
        # Update region info
        if result['region_start'] is not None and result['region_end'] is not None:
            self.region_label.setText(f"Region: {result['region_start']:.0f}-{result['region_end']:.0f} cm⁻¹")
        else:
            self.region_label.setText("Full spectrum")
        
        # Refresh the plot
        self.canvas.draw()


class StackedTabWidget(QWidget):
    """Custom tab widget with stacked tab buttons."""
    
    def __init__(self):
        super().__init__()
        
        # Apply compact UI configuration for consistent toolbar sizing
        apply_theme('compact')
        
        self.tabs = []
        self.current_index = 0
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create button grid
        button_frame = QFrame()
        button_frame.setFrameStyle(QFrame.Box)
        button_frame.setMaximumHeight(80)
        self.button_layout = QGridLayout(button_frame)
        self.button_layout.setContentsMargins(2, 2, 2, 2)
        self.button_layout.setSpacing(2)
        
        layout.addWidget(button_frame)
        
        # Create stacked widget for tab contents
        self.stacked_widget = QWidget()
        self.stacked_layout = QVBoxLayout(self.stacked_widget)
        self.stacked_layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(self.stacked_widget)
        
        # Button group for exclusive selection
        self.buttons = []
    
    def add_tab(self, widget, text, row=0, col=0):
        """Add a tab to the stacked widget."""
        # Create button
        button = QPushButton(text)
        button.setCheckable(True)
        button.setMaximumHeight(35)
        button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 4px 8px;
                text-align: center;
                font-size: 10px;
            }
            QPushButton:checked {
                background-color: #4A90E2;
                color: white;
                border: 1px solid #357ABD;
            }
            QPushButton:hover:!checked {
                background-color: #e0e0e0;
            }
        """)
        
        # Connect button to show tab
        tab_index = len(self.tabs)
        button.clicked.connect(lambda: self.show_tab(tab_index))
        
        # Add to button layout
        self.button_layout.addWidget(button, row, col)
        self.buttons.append(button)
        
        # Add widget to tabs list
        self.tabs.append(widget)
        widget.hide()
        self.stacked_layout.addWidget(widget)
        
        # Show first tab by default
        if len(self.tabs) == 1:
            self.show_tab(0)
    
    def show_tab(self, index):
        """Show the specified tab."""
        if 0 <= index < len(self.tabs):
            # Hide current tab
            if self.current_index < len(self.tabs):
                self.tabs[self.current_index].hide()
                if self.current_index < len(self.buttons):
                    self.buttons[self.current_index].setChecked(False)
            
            # Show new tab
            self.current_index = index
            self.tabs[index].show()
            self.buttons[index].setChecked(True)

class SpectralDeconvolutionQt6(QDialog):
    """Enhanced spectral deconvolution window with advanced analysis capabilities."""
    
    def __init__(self, parent, wavenumbers=None, intensities=None):
        super().__init__(parent)
        
        # Apply compact UI configuration for consistent toolbar sizing
        apply_theme('compact')
        
        self.parent = parent
        
        # Initialize spectrum data - can be empty if loading from file
        if wavenumbers is not None and intensities is not None:
            self.wavenumbers = np.array(wavenumbers)
            self.original_intensities = np.array(intensities)
            self.current_file = None
        else:
            # Empty spectrum - will be loaded from file
            self.wavenumbers = np.array([])
            self.original_intensities = np.array([])
            self.current_file = None
        
        self.processed_intensities = self.original_intensities.copy()
        
        # File loading utilities
        if FILE_LOADING_AVAILABLE:
            self.spectrum_loader = SpectrumLoader()
        else:
            self.spectrum_loader = None
        
        # Analysis data
        self.peaks = np.array([])  # Initialize as empty numpy array
        self.manual_peaks = np.array([])  # Manually selected peaks
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.background_preview_active = False  # Track background preview state
        self.residuals = None
        self.components = []
        self.pca_result = None
        self.nmf_result = None
        
        # UI state
        self.show_individual_peaks = True
        self.show_components = True
        self.current_model = "Gaussian"
        
        # Interactive peak selection state
        self.interactive_mode = False
        self.click_connection = None
        
        # Background update timer for responsive live preview
        self.bg_update_timer = QTimer()
        self.bg_update_timer.setSingleShot(True)
        self.bg_update_timer.timeout.connect(self._update_background_calculation)
        self.bg_update_delay = 150  # milliseconds
        
        # Peak detection timer for responsive live preview
        self.peak_update_timer = QTimer()
        self.peak_update_timer.setSingleShot(True)
        self.peak_update_timer.timeout.connect(self._update_peak_detection)
        self.peak_update_delay = 100  # milliseconds
        
        # Plot line references for efficient updates
        self.spectrum_line = None
        self.background_line = None
        self.fitted_line = None
        self.auto_peaks_scatter = None
        self.manual_peaks_scatter = None
        self.individual_peak_lines = []
        self.filter_preview_line = None
        
        # Background auto-preview data
        self.background_options = []  # List of (background_data, description, params) tuples
        self.background_option_lines = []  # List of plot line references
        
        # Fourier analysis data storage
        self.fft_data = None
        self.fft_frequencies = None
        self.fft_magnitude = None
        self.fft_phase = None
        
        self.setup_ui()
        self.initial_plot()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.update_window_title()
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create menu bar
        self.setup_menu_bar()
        main_layout.addWidget(self.menu_bar)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.update_status_bar()
        
        # Content layout (horizontal splitter)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # Left panel - controls (splitter for resizing)
        splitter = QSplitter(Qt.Horizontal)
        content_layout.addWidget(splitter)
        
        # Control panel
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Visualization panel
        viz_panel = self.create_visualization_panel()
        splitter.addWidget(viz_panel)
        
        # Set splitter proportions (30% controls, 70% visualization)
        splitter.setSizes([400, 1200])
        
        # Add status bar at the bottom
        main_layout.addWidget(self.status_bar)
    
    def setup_menu_bar(self):
        """Create the menu bar with file operations."""
        self.menu_bar = QMenuBar()
        
        # File menu
        file_menu = self.menu_bar.addMenu("File")
        
        # Open file action
        open_action = QAction("Open Spectrum File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        if not FILE_LOADING_AVAILABLE:
            open_action.setEnabled(False)
            open_action.setToolTip("File loading not available - install required dependencies")
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Recent files submenu (placeholder for future enhancement)
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        self.recent_files_menu.addAction("No recent files").setEnabled(False)
        
        file_menu.addSeparator()
        
        # Save/Export actions
        save_action = QAction("Export Results...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.export_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Close action
        close_action = QAction("Close", self)
        close_action.setShortcut("Ctrl+W")
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        
        # Analysis menu
        analysis_menu = self.menu_bar.addMenu("Analysis")
        
        # Quick actions
        reset_action = QAction("Reset Analysis", self)
        reset_action.triggered.connect(self.reset_spectrum)
        analysis_menu.addAction(reset_action)
        
        clear_peaks_action = QAction("Clear All Peaks", self)
        clear_peaks_action.triggered.connect(self.clear_peaks)
        analysis_menu.addAction(clear_peaks_action)
        
        analysis_menu.addSeparator()
        
        # Background subtraction
        apply_bg_action = QAction("Apply Background Subtraction", self)
        apply_bg_action.triggered.connect(self.apply_background)
        analysis_menu.addAction(apply_bg_action)
        
        # Peak fitting
        fit_peaks_action = QAction("Fit Peaks", self)
        fit_peaks_action.setShortcut("Ctrl+F")
        fit_peaks_action.triggered.connect(self.fit_peaks)
        analysis_menu.addAction(fit_peaks_action)
    
    def update_window_title(self):
        """Update the window title with current file information."""
        base_title = "Spectral Deconvolution & Advanced Analysis"
        if self.current_file:
            file_name = Path(self.current_file).name
            self.setWindowTitle(f"{base_title} - {file_name}")
        else:
            if len(self.wavenumbers) > 0:
                self.setWindowTitle(f"{base_title} - {len(self.wavenumbers)} data points")
            else:
                self.setWindowTitle(f"{base_title} - No data loaded")
    
    def update_status_bar(self):
        """Update the status bar with current spectrum information."""
        if len(self.wavenumbers) > 0:
            wn_range = f"{self.wavenumbers[0]:.1f} - {self.wavenumbers[-1]:.1f} cm⁻¹"
            intensity_range = f"{np.min(self.processed_intensities):.1f} - {np.max(self.processed_intensities):.1f}"
            n_peaks = len(self.get_all_peaks_for_fitting())
            
            status_text = f"Data points: {len(self.wavenumbers)} | Range: {wn_range} | Intensity: {intensity_range} | Peaks: {n_peaks}"
            
            if self.current_file:
                file_size = Path(self.current_file).stat().st_size / 1024  # KB
                status_text += f" | File: {file_size:.1f} KB"
            
            self.status_bar.showMessage(status_text)
        else:
            self.status_bar.showMessage("No spectrum data loaded - use File → Open to load a spectrum file")
    
    def open_file(self):
        """Open a spectrum file and load the data."""
        if not FILE_LOADING_AVAILABLE:
            QMessageBox.warning(self, "Feature Unavailable", 
                              "File loading is not available. Please install required dependencies.")
            return
        
        # Get supported file types for the dialog
        extensions = self.spectrum_loader.get_supported_extensions()
        file_filter = "Spectrum files ({});;All files (*.*)".format(
            " ".join([f"*{ext}" for ext in extensions])
        )
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Spectrum File",
            "",
            file_filter
        )
        
        if file_path:
            self.load_spectrum_file(file_path)
    
    def load_spectrum_file(self, file_path):
        """Load spectrum data from the specified file."""
        if not FILE_LOADING_AVAILABLE:
            QMessageBox.critical(self, "Error", "File loading utilities not available.")
            return
        
        try:
            # Show loading status
            self.status_bar.showMessage("Loading spectrum file...")
            
            # Load the spectrum data
            wavenumbers, intensities, metadata = self.spectrum_loader.load_spectrum(file_path)
            
            if wavenumbers is None or intensities is None:
                error_msg = metadata.get("error", "Unknown error occurred")
                QMessageBox.critical(self, "Loading Error", 
                                   f"Failed to load spectrum file:\n{error_msg}")
                self.status_bar.showMessage("Ready")
                return
            
            # Update spectrum data
            self.wavenumbers = wavenumbers
            self.original_intensities = intensities
            self.processed_intensities = self.original_intensities.copy()
            self.current_file = file_path
            
            # Reset analysis state
            self.reset_analysis_state()
            
            # Update UI
            self.update_window_title()
            self.update_status_bar()
            self.update_plot()
            
            # Show success message
            n_points = len(wavenumbers)
            wn_range = f"{wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹"
            
            QMessageBox.information(self, "File Loaded Successfully", 
                                  f"Loaded spectrum from:\n{Path(file_path).name}\n\n"
                                  f"Data points: {n_points}\n"
                                  f"Wavenumber range: {wn_range}\n"
                                  f"File size: {Path(file_path).stat().st_size / 1024:.1f} KB")
            
        except Exception as e:
            QMessageBox.critical(self, "Loading Error", 
                               f"An error occurred while loading the file:\n{str(e)}")
            self.status_bar.showMessage("Ready")
    
    def reset_analysis_state(self):
        """Reset all analysis state when loading new data."""
        # Clear peaks
        self.peaks = np.array([])
        self.manual_peaks = np.array([])
        
        # Clear fitting results
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        
        # Clear background
        self.background = None
        self.background_preview_active = False
        
        # Clear components and analysis results
        self.components = []
        self.pca_result = None
        self.nmf_result = None
        
        # Clear background options
        self.clear_background_options()
        
        # Disable interactive mode
        if self.interactive_mode:
            self.interactive_mode = False
            if hasattr(self, 'interactive_btn'):
                self.interactive_btn.setChecked(False)
                self.interactive_btn.setText("🖱️ Enable Interactive Selection")
            if hasattr(self, 'interactive_status_label'):
                self.interactive_status_label.setText("Interactive mode: OFF")
                self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Disconnect mouse click event
            if self.click_connection is not None:
                self.canvas.mpl_disconnect(self.click_connection)
                self.click_connection = None
        
        # Update peak displays
        if hasattr(self, 'peak_count_label'):
            self.update_peak_count_display()
        if hasattr(self, 'peak_list_widget'):
            self.update_peak_list()
        
        # Clear results
        if hasattr(self, 'results_text'):
            self.results_text.clear()
        if hasattr(self, 'results_table'):
            self.results_table.setRowCount(0)
        
        # Clear Fourier analysis data
        self.fft_data = None
        self.fft_frequencies = None
        self.fft_magnitude = None
        self.fft_phase = None
        
        # Reset plot line references
        self._reset_line_references()
        
    def create_control_panel(self):
        """Create the control panel with tabs."""
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        
        # Create tab widget with stacked layout
        self.tab_widget = StackedTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs with 2-row layout:
        # Row 1: Background, Peak Detection, Peak Fitting
        # Row 2: Batch, Analysis Results, Advanced
        self.tab_widget.add_tab(self.create_background_tab(), "Background", row=0, col=0)
        self.tab_widget.add_tab(self.create_peak_detection_tab(), "Peak Detection", row=0, col=1)
        self.tab_widget.add_tab(self.create_fitting_tab(), "Peak Fitting", row=0, col=2)
        self.tab_widget.add_tab(self.create_batch_tab(), "Batch", row=1, col=0)
        self.tab_widget.add_tab(self.create_analysis_tab(), "Analysis Results", row=1, col=1)
        self.tab_widget.add_tab(self.create_advanced_tab(), "Advanced", row=1, col=2)
        
        return panel
        
    def create_visualization_panel(self):
        """Create the visualization panel with tabs for Current Spectrum and Plotting Analysis."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for switching between spectrum view and plotting analysis
        self.visualization_tabs = QTabWidget()
        
        # Tab 1: Current Spectrum (existing functionality)
        spectrum_tab = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_tab)
        
        # Create matplotlib figure for current spectrum
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, spectrum_tab)
        
        # Create subplots with custom height ratios - main plot gets 75%, residual gets 25%
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
        self.ax_main = self.figure.add_subplot(gs[0])      # Main spectrum (top, larger)
        self.ax_residual = self.figure.add_subplot(gs[1])  # Residuals (bottom, smaller)
        
        self.figure.tight_layout(pad=3.0)
        
        spectrum_layout.addWidget(self.toolbar)
        spectrum_layout.addWidget(self.canvas)
        
        # Tab 2: Plotting Analysis
        plotting_tab = QWidget()
        plotting_layout = QHBoxLayout(plotting_tab)
        
        # Left panel: Plot controls
        plot_controls_panel = QWidget()
        plot_controls_panel.setMaximumWidth(300)
        plot_controls_layout = QVBoxLayout(plot_controls_panel)
        
        # Plot type selection
        plot_type_group = QGroupBox("📊 Plot Type")
        plot_type_layout = QVBoxLayout(plot_type_group)
        
        # Create toggle buttons for plot types
        self.plot_type_buttons = QButtonGroup()
        
        self.grid_plot_btn = QPushButton("Peak Features Grid")
        self.grid_plot_btn.setCheckable(True)
        self.grid_plot_btn.setChecked(True)
        self.grid_plot_btn.setToolTip("2x2 grid showing amplitude, position, FWHM, and R² for each spectrum")
        self.plot_type_buttons.addButton(self.grid_plot_btn, 0)
        
        self.waterfall_plot_btn = QPushButton("Waterfall Plot")
        self.waterfall_plot_btn.setCheckable(True)
        self.waterfall_plot_btn.setToolTip("Stacked spectra with adjustable offset and colors")
        self.plot_type_buttons.addButton(self.waterfall_plot_btn, 1)
        
        self.heatmap_plot_btn = QPushButton("Heatmap Plot")
        self.heatmap_plot_btn.setCheckable(True)
        self.heatmap_plot_btn.setToolTip("2D intensity heatmap of all spectra")
        self.plot_type_buttons.addButton(self.heatmap_plot_btn, 2)
        
        # Style the buttons
        button_style = """
            QPushButton {
                text-align: left;
                padding: 8px 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                border-color: #45a049;
            }
            QPushButton:hover {
                background-color: #e9e9e9;
            }
            QPushButton:checked:hover {
                background-color: #45a049;
            }
        """
        
        for btn in [self.grid_plot_btn, self.waterfall_plot_btn, self.heatmap_plot_btn]:
            btn.setStyleSheet(button_style)
        
        plot_type_layout.addWidget(self.grid_plot_btn)
        plot_type_layout.addWidget(self.waterfall_plot_btn)
        plot_type_layout.addWidget(self.heatmap_plot_btn)
        
        plot_controls_layout.addWidget(plot_type_group)
        
        # Data source information
        data_source_group = QGroupBox("📁 Data Source")
        data_source_layout = QVBoxLayout(data_source_group)
        
        self.data_source_label = QLabel("No batch results available")
        self.data_source_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        self.data_source_label.setWordWrap(True)
        data_source_layout.addWidget(self.data_source_label)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.clicked.connect(self.refresh_analysis_plots)
        data_source_layout.addWidget(refresh_btn)
        
        plot_controls_layout.addWidget(data_source_group)
        
        # Plot-specific controls (will be populated dynamically)
        self.plot_controls_group = QGroupBox("⚙️ Plot Controls")
        self.plot_controls_layout = QVBoxLayout(self.plot_controls_group)
        plot_controls_layout.addWidget(self.plot_controls_group)
        
        plot_controls_layout.addStretch()
        
        # Right panel: Plotting area
        plotting_panel = QWidget()
        plotting_panel_layout = QVBoxLayout(plotting_panel)
        
        # Create stacked widget for different plot types
        self.analysis_plots_stack = QStackedWidget()
        
        # Create plot widgets
        self.grid_plot_widget = self.create_grid_plot_widget()
        self.waterfall_plot_widget = self.create_waterfall_plot_widget()
        self.heatmap_plot_widget = self.create_heatmap_plot_widget()
        
        # Add to stack
        self.analysis_plots_stack.addWidget(self.grid_plot_widget)
        self.analysis_plots_stack.addWidget(self.waterfall_plot_widget)
        self.analysis_plots_stack.addWidget(self.heatmap_plot_widget)
        
        plotting_panel_layout.addWidget(self.analysis_plots_stack)
        
        # Connect plot type buttons to stack switching
        self.plot_type_buttons.buttonClicked.connect(self.on_plot_type_changed)
        
        # Add panels to plotting tab layout
        plotting_layout.addWidget(plot_controls_panel)
        plotting_layout.addWidget(plotting_panel)
        
        # Add tabs to tab widget
        self.visualization_tabs.addTab(spectrum_tab, "Current Spectrum")
        self.visualization_tabs.addTab(plotting_tab, "Plotting Analysis")
        
        # Add tab widget to main layout
        layout.addWidget(self.visualization_tabs)
        
        # Initialize plot controls
        self.setup_plot_controls()
        self.update_data_source_info()
        
        return panel
        
    def create_background_tab(self):
        """Create background subtraction tab using unified components."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Always use enhanced fallback controls to ensure new Moving Average and Spline options are available
        print("🎛️ Creating enhanced background controls with Moving Average and Spline support")
        self._create_fallback_background_controls(layout)
        
        # Action buttons (tool-specific functionality)
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_background)
        button_layout.addWidget(apply_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_background_preview)
        button_layout.addWidget(clear_btn)
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_spectrum)
        button_layout.addWidget(reset_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return tab
    
    def _create_fallback_background_controls(self, layout):
        """Create fallback background controls when centralized UI is not available."""
        method_group = QGroupBox("Background Method")
        method_layout = QVBoxLayout(method_group)
        
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["ALS", "Linear", "Polynomial", "Moving Average", "Spline"])
        self.bg_method_combo.currentTextChanged.connect(self._on_bg_method_changed)
        method_layout.addWidget(self.bg_method_combo)
        
        layout.addWidget(method_group)
        
        # Create a stacked widget to hold different parameter groups
        self.bg_params_stack = QStackedWidget()
        
        # ALS parameters
        als_widget = QWidget()
        als_layout = QFormLayout(als_widget)
        
        # Lambda slider (log scale: 10^3 to 10^7)
        self.lambda_slider = QSlider(Qt.Horizontal)
        self.lambda_slider.setRange(30, 70)  # 3.0 to 7.0 in log10 scale
        self.lambda_slider.setValue(50)  # 10^5 = 100,000
        self.lambda_slider.setToolTip("Smoothness parameter (higher = smoother baseline)")
        self.lambda_slider.valueChanged.connect(self._update_lambda_label)
        self.lambda_slider.valueChanged.connect(self._trigger_background_update)
        
        self.lambda_label = QLabel("100,000")
        lambda_layout = QHBoxLayout()
        lambda_layout.addWidget(self.lambda_slider)
        lambda_layout.addWidget(self.lambda_label)
        als_layout.addRow("Lambda:", lambda_layout)
        
        # P slider (0.001 to 0.1)
        self.p_slider = QSlider(Qt.Horizontal)
        self.p_slider.setRange(1, 100)  # 0.001 to 0.1 scaled by 1000
        self.p_slider.setValue(10)  # 0.01
        self.p_slider.setToolTip("Asymmetry parameter (0.001-0.1, lower = more asymmetric)")
        self.p_slider.valueChanged.connect(self._update_p_label)
        self.p_slider.valueChanged.connect(self._trigger_background_update)
        
        self.p_label = QLabel("0.01")
        p_layout = QHBoxLayout()
        p_layout.addWidget(self.p_slider)
        p_layout.addWidget(self.p_label)
        als_layout.addRow("P:", p_layout)
        
        # Iterations slider (5 to 20)
        self.niter_slider = QSlider(Qt.Horizontal)
        self.niter_slider.setRange(5, 20)
        self.niter_slider.setValue(10)
        self.niter_slider.setToolTip("Number of iterations (5-20 typically)")
        self.niter_slider.valueChanged.connect(self._update_niter_label)
        self.niter_slider.valueChanged.connect(self._trigger_background_update)
        
        self.niter_label = QLabel("10")
        niter_layout = QHBoxLayout()
        niter_layout.addWidget(self.niter_slider)
        niter_layout.addWidget(self.niter_label)
        als_layout.addRow("Iterations:", niter_layout)
        
        self.bg_params_stack.addWidget(als_widget)
        
        # Linear parameters
        linear_widget = QWidget()
        linear_layout = QFormLayout(linear_widget)
        
        # Start Weight slider (0.1 to 2.0)
        self.start_weight_slider = QSlider(Qt.Horizontal)
        self.start_weight_slider.setRange(1, 20)  # 0.1 to 2.0 scaled by 10
        self.start_weight_slider.setValue(10)  # 1.0
        self.start_weight_slider.setToolTip("Weight for start point (0.1-2.0, higher = more emphasis)")
        self.start_weight_slider.valueChanged.connect(self._update_start_weight_label)
        self.start_weight_slider.valueChanged.connect(self._trigger_background_update)
        
        self.start_weight_label = QLabel("1.0")
        start_weight_layout = QHBoxLayout()
        start_weight_layout.addWidget(self.start_weight_slider)
        start_weight_layout.addWidget(self.start_weight_label)
        linear_layout.addRow("Start Weight:", start_weight_layout)
        
        # End Weight slider (0.1 to 2.0)
        self.end_weight_slider = QSlider(Qt.Horizontal)
        self.end_weight_slider.setRange(1, 20)  # 0.1 to 2.0 scaled by 10
        self.end_weight_slider.setValue(10)  # 1.0
        self.end_weight_slider.setToolTip("Weight for end point (0.1-2.0, higher = more emphasis)")
        self.end_weight_slider.valueChanged.connect(self._update_end_weight_label)
        self.end_weight_slider.valueChanged.connect(self._trigger_background_update)
        
        self.end_weight_label = QLabel("1.0")
        end_weight_layout = QHBoxLayout()
        end_weight_layout.addWidget(self.end_weight_slider)
        end_weight_layout.addWidget(self.end_weight_label)
        linear_layout.addRow("End Weight:", end_weight_layout)
        
        self.bg_params_stack.addWidget(linear_widget)
        
        # Polynomial parameters
        poly_widget = QWidget()
        poly_layout = QFormLayout(poly_widget)
        
        # Polynomial Order slider (1 to 6)
        self.poly_order_slider = QSlider(Qt.Horizontal)
        self.poly_order_slider.setRange(1, 6)
        self.poly_order_slider.setValue(2)  # quadratic
        self.poly_order_slider.setToolTip("Polynomial order (1=linear, 2=quadratic, 3=cubic, etc.)")
        self.poly_order_slider.valueChanged.connect(self._update_poly_order_label)
        self.poly_order_slider.valueChanged.connect(self._trigger_background_update)
        
        self.poly_order_label = QLabel("2 (Quadratic)")
        poly_order_layout = QHBoxLayout()
        poly_order_layout.addWidget(self.poly_order_slider)
        poly_order_layout.addWidget(self.poly_order_label)
        poly_layout.addRow("Order:", poly_order_layout)
        
        self.poly_method_combo = QComboBox()
        self.poly_method_combo.addItems(["Least Squares", "Robust"])
        self.poly_method_combo.setToolTip("Least Squares: standard fitting, Robust: reduces outlier influence")
        self.poly_method_combo.currentTextChanged.connect(self._trigger_background_update)
        poly_layout.addRow("Method:", self.poly_method_combo)
        
        self.bg_params_stack.addWidget(poly_widget)
        
        # Moving Average parameters
        mavg_widget = QWidget()
        mavg_layout = QFormLayout(mavg_widget)
        
        self.window_percent_slider = QSlider(Qt.Horizontal)
        self.window_percent_slider.setRange(5, 50)
        self.window_percent_slider.setValue(15)
        self.window_percent_slider.setToolTip("Window size as percentage of spectrum length (5-50%)")
        self.window_percent_slider.valueChanged.connect(self._update_window_percent_label)
        self.window_percent_slider.valueChanged.connect(self._trigger_background_update)
        
        self.window_percent_label = QLabel("15%")
        window_percent_layout = QHBoxLayout()
        window_percent_layout.addWidget(self.window_percent_slider)
        window_percent_layout.addWidget(self.window_percent_label)
        
        mavg_layout.addRow("Window Size:", window_percent_layout)
        
        self.window_type_combo = QComboBox()
        self.window_type_combo.addItems(["Uniform", "Gaussian", "Hann", "Hamming"])
        self.window_type_combo.setToolTip("Window type: Uniform=box, Gaussian=smooth, Hann/Hamming=tapered")
        self.window_type_combo.currentTextChanged.connect(self._trigger_background_update)
        mavg_layout.addRow("Window Type:", self.window_type_combo)
        
        self.bg_params_stack.addWidget(mavg_widget)
        
        # Spline parameters
        spline_widget = QWidget()
        spline_layout = QFormLayout(spline_widget)
        
        self.n_knots_slider = QSlider(Qt.Horizontal)
        self.n_knots_slider.setRange(3, 20)
        self.n_knots_slider.setValue(10)
        self.n_knots_slider.setToolTip("Number of spline knots (3-20, more = more flexible)")
        self.n_knots_slider.valueChanged.connect(self._update_n_knots_label)
        self.n_knots_slider.valueChanged.connect(self._trigger_background_update)
        
        self.n_knots_label = QLabel("10")
        knots_layout = QHBoxLayout()
        knots_layout.addWidget(self.n_knots_slider)
        knots_layout.addWidget(self.n_knots_label)
        
        spline_layout.addRow("Number of Knots:", knots_layout)
        
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(1, 1000)
        self.smoothing_slider.setValue(100)
        self.smoothing_slider.setToolTip("Smoothing factor (1-1000, higher = smoother spline)")
        self.smoothing_slider.valueChanged.connect(self._update_smoothing_label)
        self.smoothing_slider.valueChanged.connect(self._trigger_background_update)
        
        self.smoothing_label = QLabel("100")
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(self.smoothing_slider)
        smoothing_layout.addWidget(self.smoothing_label)
        
        spline_layout.addRow("Smoothing:", smoothing_layout)
        
        self.spline_degree_combo = QComboBox()
        self.spline_degree_combo.addItems(["1", "2", "3"])
        self.spline_degree_combo.setCurrentText("3")
        self.spline_degree_combo.setToolTip("Spline degree (1=linear, 2=quadratic, 3=cubic)")
        self.spline_degree_combo.currentTextChanged.connect(self._trigger_background_update)
        spline_layout.addRow("Degree:", self.spline_degree_combo)
        
        self.bg_params_stack.addWidget(spline_widget)
        
        # Add the stack to the layout
        bg_params_group = QGroupBox("Parameters")
        bg_params_group_layout = QVBoxLayout(bg_params_group)
        bg_params_group_layout.addWidget(self.bg_params_stack)
        
        layout.addWidget(bg_params_group)
        
        # Set initial stack index and log setup
        self.bg_params_stack.setCurrentIndex(0)
        
        # Initialize all slider labels with default values
        self._update_lambda_label()
        self._update_p_label()
        self._update_niter_label()
        self._update_start_weight_label()
        self._update_end_weight_label()
        self._update_poly_order_label()
        self._update_window_percent_label()
        self._update_n_knots_label()
        self._update_smoothing_label()
        
        print(f"✅ Enhanced background controls created successfully!")
        print(f"   📋 Available methods: {[self.bg_method_combo.itemText(i) for i in range(self.bg_method_combo.count())]}")
        print(f"   🔢 Parameter panels: {self.bg_params_stack.count()} (ALS, Linear, Polynomial, Moving Average, Spline)")
        print(f"   🎯 Currently showing: {self.bg_method_combo.currentText()} parameters")
        print(f"   🎛️ All controls now use sliders for consistent user experience!")
    
    def _on_bg_method_changed(self):
        """Handle background method change and update parameter display."""
        if not hasattr(self, 'bg_params_stack'):
            print("⚠️ bg_params_stack not found - background controls may not be initialized properly")
            return
        
        method = self.bg_method_combo.currentText()
        print(f"🔄 Background method changed to: {method}")
        
        # Update the stacked widget to show appropriate parameters
        if method == "ALS":
            self.bg_params_stack.setCurrentIndex(0)
            print("📊 Showing ALS parameters")
        elif method == "Linear":
            self.bg_params_stack.setCurrentIndex(1)
            print("📊 Showing Linear parameters")
        elif method == "Polynomial":
            self.bg_params_stack.setCurrentIndex(2)
            print("📊 Showing Polynomial parameters")
        elif method == "Moving Average":
            self.bg_params_stack.setCurrentIndex(3)
            print("📊 Showing Moving Average parameters (Window Size, Window Type)")
        elif method == "Spline":
            self.bg_params_stack.setCurrentIndex(4)
            print("📊 Showing Spline parameters (Knots, Smoothing, Degree)")
        else:
            print(f"⚠️ Unknown background method: {method}")
        
        # Trigger background update after method change
        self._trigger_background_update()
    
    def _update_window_percent_label(self):
        """Update the window percent label for moving average."""
        if hasattr(self, 'window_percent_slider') and hasattr(self, 'window_percent_label'):
            value = self.window_percent_slider.value()
            self.window_percent_label.setText(f"{value}%")
    
    def _update_n_knots_label(self):
        """Update the number of knots label for spline."""
        if hasattr(self, 'n_knots_slider') and hasattr(self, 'n_knots_label'):
            value = self.n_knots_slider.value()
            self.n_knots_label.setText(str(value))
    
    def _update_smoothing_label(self):
        """Update the smoothing label for spline."""
        if hasattr(self, 'smoothing_slider') and hasattr(self, 'smoothing_label'):
            value = self.smoothing_slider.value()
            self.smoothing_label.setText(str(value))
    
    def _update_lambda_label(self):
        """Update the lambda label for ALS (log scale)."""
        if hasattr(self, 'lambda_slider') and hasattr(self, 'lambda_label'):
            value = self.lambda_slider.value()
            # Convert from log scale (30-70) to actual value (10^3 to 10^7)
            lambda_val = 10 ** (value / 10.0)
            if lambda_val >= 1000000:
                self.lambda_label.setText(f"{lambda_val/1000000:.1f}M")
            elif lambda_val >= 1000:
                self.lambda_label.setText(f"{lambda_val/1000:.0f}K")
            else:
                self.lambda_label.setText(f"{lambda_val:.0f}")
    
    def _update_p_label(self):
        """Update the p label for ALS."""
        if hasattr(self, 'p_slider') and hasattr(self, 'p_label'):
            value = self.p_slider.value()
            # Convert from slider value (1-100) to actual p value (0.001-0.1)
            p_val = value / 1000.0
            self.p_label.setText(f"{p_val:.3f}")
    
    def _update_niter_label(self):
        """Update the iterations label for ALS."""
        if hasattr(self, 'niter_slider') and hasattr(self, 'niter_label'):
            value = self.niter_slider.value()
            self.niter_label.setText(str(value))
    
    def _update_start_weight_label(self):
        """Update the start weight label for Linear."""
        if hasattr(self, 'start_weight_slider') and hasattr(self, 'start_weight_label'):
            value = self.start_weight_slider.value()
            # Convert from slider value (1-20) to actual weight (0.1-2.0)
            weight_val = value / 10.0
            self.start_weight_label.setText(f"{weight_val:.1f}")
    
    def _update_end_weight_label(self):
        """Update the end weight label for Linear."""
        if hasattr(self, 'end_weight_slider') and hasattr(self, 'end_weight_label'):
            value = self.end_weight_slider.value()
            # Convert from slider value (1-20) to actual weight (0.1-2.0)
            weight_val = value / 10.0
            self.end_weight_label.setText(f"{weight_val:.1f}")
    
    def _update_poly_order_label(self):
        """Update the polynomial order label."""
        if hasattr(self, 'poly_order_slider') and hasattr(self, 'poly_order_label'):
            value = self.poly_order_slider.value()
            order_names = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic", 5: "Quintic", 6: "Sextic"}
            name = order_names.get(value, f"Order {value}")
            self.poly_order_label.setText(f"{value} ({name})")
    
    def get_fallback_background_parameters(self):
        """Get background parameters from fallback controls."""
        try:
            method = self.bg_method_combo.currentText() if hasattr(self, 'bg_method_combo') else "ALS"
            
            if method == "ALS":
                # Get values from sliders
                lambda_val = 10 ** (self.lambda_slider.value() / 10.0) if hasattr(self, 'lambda_slider') else 1e5
                p_val = self.p_slider.value() / 1000.0 if hasattr(self, 'p_slider') else 0.01
                niter_val = self.niter_slider.value() if hasattr(self, 'niter_slider') else 10
                return {
                    'method': 'ALS',
                    'lambda': lambda_val,
                    'p': p_val,
                    'niter': niter_val
                }
            elif method == "Linear":
                # Get values from sliders
                start_weight = self.start_weight_slider.value() / 10.0 if hasattr(self, 'start_weight_slider') else 1.0
                end_weight = self.end_weight_slider.value() / 10.0 if hasattr(self, 'end_weight_slider') else 1.0
                return {
                    'method': 'Linear',
                    'start_weight': start_weight,
                    'end_weight': end_weight
                }
            elif method == "Polynomial":
                # Get values from slider and combo
                order = self.poly_order_slider.value() if hasattr(self, 'poly_order_slider') else 2
                poly_method = self.poly_method_combo.currentText() if hasattr(self, 'poly_method_combo') else "Least Squares"
                return {
                    'method': 'Polynomial',
                    'order': order,
                    'poly_method': poly_method
                }
            elif method == "Moving Average":
                return {
                    'method': 'Moving Average',
                    'window_percent': self.window_percent_slider.value() if hasattr(self, 'window_percent_slider') else 15,
                    'window_type': self.window_type_combo.currentText() if hasattr(self, 'window_type_combo') else "Uniform"
                }
            elif method == "Spline":
                return {
                    'method': 'Spline',
                    'n_knots': self.n_knots_slider.value() if hasattr(self, 'n_knots_slider') else 10,
                    'smoothing': self.smoothing_slider.value() if hasattr(self, 'smoothing_slider') else 100,
                    'degree': int(self.spline_degree_combo.currentText()) if hasattr(self, 'spline_degree_combo') else 3
                }
            else:
                return {
                    'method': method,
                    'order': 2,
                    'start_weight': 1.0,
                    'end_weight': 1.0
                }
        except Exception:
            # Ultimate fallback
            return {'method': 'ALS', 'lambda': 1e5, 'p': 0.01, 'niter': 10}
    
    def get_fallback_peak_parameters(self):
        """Get peak parameters from fallback controls."""
        try:
            if hasattr(self, 'height_slider') and hasattr(self, 'distance_slider') and hasattr(self, 'prominence_slider'):
                return {
                    'model': self.current_model,
                    'height': self.height_slider.value() / 100.0,
                    'distance': self.distance_slider.value(),
                    'prominence': self.prominence_slider.value() / 100.0
                }
            else:
                # Ultimate fallback
                return {
                    'model': 'Gaussian',
                    'height': 0.1,
                    'distance': 10,
                    'prominence': 0.05
                }
        except Exception:
            # Ultimate fallback
            return {
                'model': 'Gaussian',
                'height': 0.1,
                'distance': 10,
                'prominence': 0.05
                         }
    
    def get_peak_parameters(self):
        """Get peak parameters from centralized or fallback controls."""
        if CENTRALIZED_AVAILABLE and hasattr(self, 'peak_controls'):
            return self.peak_controls.get_peak_parameters()
        else:
            return self.get_fallback_peak_parameters()
         
    def create_peak_detection_tab(self):
        """Create peak detection tab with interactive selection using slider controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Always use fallback controls (sliders) for better user experience
        # The centralized controls use spinboxes which are less intuitive than sliders
        self._create_fallback_peak_controls(layout)
        
        # Individual Peak Management group
        peak_management_group = QGroupBox("Individual Peak Management")
        peak_management_layout = QVBoxLayout(peak_management_group)
        
        # Peak list widget
        list_label = QLabel("Current Peaks (click to select, then remove):")
        list_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        peak_management_layout.addWidget(list_label)
        
        self.peak_list_widget = QListWidget()
        self.peak_list_widget.setMaximumHeight(120)
        self.peak_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #fafafa;
            }
            QListWidget::item {
                padding: 2px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e6f3ff;
            }
        """)
        peak_management_layout.addWidget(self.peak_list_widget)
        
        # Individual peak removal buttons
        individual_buttons_layout = QHBoxLayout()
        
        remove_selected_btn = QPushButton("Remove Selected Peak")
        remove_selected_btn.clicked.connect(self.remove_selected_peak)
        remove_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #800020;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #660018;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        refresh_list_btn = QPushButton("Refresh List")
        refresh_list_btn.clicked.connect(self.update_peak_list)
        refresh_list_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        
        individual_buttons_layout.addWidget(remove_selected_btn)
        individual_buttons_layout.addWidget(refresh_list_btn)
        peak_management_layout.addLayout(individual_buttons_layout)
        
        layout.addWidget(peak_management_group)
        
        # Interactive peak selection group
        interactive_group = QGroupBox("Interactive Peak Selection")
        interactive_layout = QVBoxLayout(interactive_group)
        
        # Instructions
        instructions = QLabel(
            "Enable interactive mode and click on the spectrum plot to manually select peaks.\n"
            "Click near existing peaks to remove them."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #555; font-size: 10px; margin: 5px;")
        interactive_layout.addWidget(instructions)
        
        # Interactive mode toggle button
        self.interactive_btn = QPushButton("🖱️ Enable Interactive Selection")
        self.interactive_btn.setCheckable(True)
        self.interactive_btn.clicked.connect(self.toggle_interactive_mode)
        self.interactive_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #FF5722;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:checked:hover {
                background-color: #E64A19;
            }
        """)
        interactive_layout.addWidget(self.interactive_btn)
        
        # Manual peak controls
        manual_controls_layout = QHBoxLayout()
        
        clear_manual_btn = QPushButton("Clear Manual Peaks")
        clear_manual_btn.clicked.connect(self.clear_manual_peaks)
        clear_manual_btn.setStyleSheet("""
            QPushButton {
                background-color: #800020;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #660018;
            }
        """)
        
        combine_btn = QPushButton("Combine Auto + Manual")
        combine_btn.clicked.connect(self.combine_peaks)
        combine_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        
        manual_controls_layout.addWidget(clear_manual_btn)
        manual_controls_layout.addWidget(combine_btn)
        interactive_layout.addLayout(manual_controls_layout)
        
        layout.addWidget(interactive_group)
        
        # Peak count and status
        status_group = QGroupBox("Peak Status")
        status_layout = QVBoxLayout(status_group)
        
        self.peak_count_label = QLabel("Auto peaks: 0 | Manual peaks: 0 | Total: 0")
        self.peak_count_label.setStyleSheet("font-weight: bold; color: #333;")
        status_layout.addWidget(self.peak_count_label)
        
        self.interactive_status_label = QLabel("Interactive mode: OFF")
        self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
        status_layout.addWidget(self.interactive_status_label)
        
        layout.addWidget(status_group)
        layout.addStretch()
        
        return tab
        
    def create_fitting_tab(self):
        """Create peak fitting tab with centralized model selection."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection - use centralized controls if available for consistency
        if CENTRALIZED_AVAILABLE and hasattr(self, 'peak_controls'):
            # Model selection is already handled by centralized controls
            model_info = QLabel("Peak model selection is managed in the Peak Detection tab.")
            model_info.setStyleSheet("color: #666; font-style: italic; margin: 10px;")
            layout.addWidget(model_info)
        else:
            # Fallback model selection for when centralized controls aren't available
            model_group = QGroupBox("Peak Model")
            model_layout = QVBoxLayout(model_group)
            
            self.model_combo = QComboBox()
            self.model_combo.addItems([
                "Gaussian", "Lorentzian", "Pseudo-Voigt", "Asymmetric Voigt"
            ])
            self.model_combo.currentTextChanged.connect(self.on_model_changed)
            model_layout.addWidget(self.model_combo)
            
            layout.addWidget(model_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_peaks_check = QCheckBox("Show Individual Peaks")
        self.show_peaks_check.setChecked(True)
        self.show_peaks_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_peaks_check)
        
        self.show_components_check = QCheckBox("Show Components")
        self.show_components_check.setChecked(True)
        self.show_components_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_components_check)
        
        self.show_legend_check = QCheckBox("Show Legend")
        self.show_legend_check.setChecked(True)
        self.show_legend_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_legend_check)
        
        self.show_grid_check = QCheckBox("Show Grid")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_grid_check)
        
        layout.addWidget(display_group)
        
        # Fitting
        fit_group = QGroupBox("Peak Fitting")
        fit_layout = QVBoxLayout(fit_group)
        
        fit_btn = QPushButton("Fit Peaks")
        fit_btn.clicked.connect(self.fit_peaks)
        fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        fit_layout.addWidget(fit_btn)
        
        # Results
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        fit_layout.addWidget(self.results_text)
        
        layout.addWidget(fit_group)
        
        # Analysis Results table (moved from Analysis tab)
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Peak", "Position", "Amplitude", "Width", "R²"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        # Export options (moved from Analysis tab)
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        export_layout.addWidget(export_btn)
        
        layout.addWidget(export_group)
        layout.addStretch()
        
        return tab
    
    def _create_fallback_peak_controls(self, layout):
        """Create enhanced peak detection controls with live-updating sliders."""
        # Peak detection group
        peak_group = QGroupBox("Peak Detection - Live Updates")
        peak_layout = QVBoxLayout(peak_group)
        
        # Parameters with enhanced sliders
        params_layout = QFormLayout()
        
        # Height parameter
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(1, 100)
        self.height_slider.setValue(10)
        self.height_slider.setTracking(True)  # Enable live tracking
        self.height_slider.setToolTip("Minimum peak height as percentage of maximum intensity")
        self.height_label = QLabel("10%")
        self.height_label.setMinimumWidth(40)
        height_layout = QHBoxLayout()
        height_layout.addWidget(self.height_slider)
        height_layout.addWidget(self.height_label)
        params_layout.addRow("Min Height:", height_layout)
        
        # Distance parameter
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(1, 50)
        self.distance_slider.setValue(10)
        self.distance_slider.setTracking(True)  # Enable live tracking
        self.distance_slider.setToolTip("Minimum distance between peaks (data points)")
        self.distance_label = QLabel("10")
        self.distance_label.setMinimumWidth(40)
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(self.distance_slider)
        distance_layout.addWidget(self.distance_label)
        params_layout.addRow("Min Distance:", distance_layout)
        
        # Prominence parameter
        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(1, 50)
        self.prominence_slider.setValue(5)
        self.prominence_slider.setTracking(True)  # Enable live tracking
        self.prominence_slider.setToolTip("Minimum peak prominence as percentage of maximum intensity")
        self.prominence_label = QLabel("5%")
        self.prominence_label.setMinimumWidth(40)
        prominence_layout = QHBoxLayout()
        prominence_layout.addWidget(self.prominence_slider)
        prominence_layout.addWidget(self.prominence_label)
        params_layout.addRow("Prominence:", prominence_layout)
        
        peak_layout.addLayout(params_layout)
        
        # Preset buttons for common peak detection settings
        presets_layout = QHBoxLayout()
        
        sensitive_btn = QPushButton("Sensitive")
        sensitive_btn.clicked.connect(lambda: self.apply_peak_preset(height=5, distance=5, prominence=2))
        sensitive_btn.setToolTip("Detect more peaks - good for noisy spectra")
        presets_layout.addWidget(sensitive_btn)
        
        balanced_btn = QPushButton("Balanced")
        balanced_btn.clicked.connect(lambda: self.apply_peak_preset(height=10, distance=10, prominence=5))
        balanced_btn.setToolTip("Balanced detection - good default settings")
        presets_layout.addWidget(balanced_btn)
        
        strict_btn = QPushButton("Strict")
        strict_btn.clicked.connect(lambda: self.apply_peak_preset(height=20, distance=20, prominence=10))
        strict_btn.setToolTip("Detect only prominent peaks")
        presets_layout.addWidget(strict_btn)
        
        peak_layout.addLayout(presets_layout)
        
        # Connect sliders to update labels
        self.height_slider.valueChanged.connect(self.update_height_label)
        self.distance_slider.valueChanged.connect(self.update_distance_label)
        self.prominence_slider.valueChanged.connect(self.update_prominence_label)
        
        # Connect sliders to live peak detection with debouncing
        self.height_slider.valueChanged.connect(self._trigger_peak_update)
        self.distance_slider.valueChanged.connect(self._trigger_peak_update)
        self.prominence_slider.valueChanged.connect(self._trigger_peak_update)
        
        # Automatic detection buttons
        auto_button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear All Peaks")
        clear_btn.clicked.connect(self.clear_peaks)
        auto_button_layout.addWidget(clear_btn)
        
        peak_layout.addLayout(auto_button_layout)
        
        layout.addWidget(peak_group)
        
    def create_deconvolution_tab(self):
        """Create spectral deconvolution tab with Fourier-based analysis."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Fourier Analysis
        fourier_group = QGroupBox("Fourier Transform Analysis")
        fourier_layout = QVBoxLayout(fourier_group)
        
        # Fourier transform display
        fft_btn = QPushButton("Show Frequency Spectrum")
        fft_btn.clicked.connect(self.show_frequency_spectrum)
        fft_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        fourier_layout.addWidget(fft_btn)
        
        # Power spectral density
        psd_btn = QPushButton("Power Spectral Density")
        psd_btn.clicked.connect(self.show_power_spectral_density)
        fourier_layout.addWidget(psd_btn)
        
        layout.addWidget(fourier_group)
        
        # Fourier Filtering
        filter_group = QGroupBox("Fourier Filtering")
        filter_layout = QVBoxLayout(filter_group)
        
        # Filter type selection
        filter_type_layout = QHBoxLayout()
        filter_type_layout.addWidget(QLabel("Filter Type:"))
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Low-pass", "High-pass", "Band-pass", "Band-stop", "Butterworth Low-pass", "Butterworth High-pass", "Butterworth Band-pass", "Butterworth Band-stop"])
        self.filter_type_combo.currentTextChanged.connect(self.on_filter_type_changed)
        filter_type_layout.addWidget(self.filter_type_combo)
        filter_layout.addLayout(filter_type_layout)
        
        # Cutoff frequency controls
        cutoff_layout = QFormLayout()
        
        # Low cutoff (for band filters)
        self.low_cutoff_slider = QSlider(Qt.Horizontal)
        self.low_cutoff_slider.setRange(1, 50)
        self.low_cutoff_slider.setValue(10)
        self.low_cutoff_label = QLabel("10%")
        low_cutoff_layout = QHBoxLayout()
        low_cutoff_layout.addWidget(self.low_cutoff_slider)
        low_cutoff_layout.addWidget(self.low_cutoff_label)
        cutoff_layout.addRow("Low Cutoff:", low_cutoff_layout)
        
        # High cutoff
        self.high_cutoff_slider = QSlider(Qt.Horizontal)
        self.high_cutoff_slider.setRange(10, 90)
        self.high_cutoff_slider.setValue(50)
        self.high_cutoff_label = QLabel("50%")
        high_cutoff_layout = QHBoxLayout()
        high_cutoff_layout.addWidget(self.high_cutoff_slider)
        high_cutoff_layout.addWidget(self.high_cutoff_label)
        cutoff_layout.addRow("High Cutoff:", high_cutoff_layout)
        
        # Butterworth filter order (initially hidden)
        self.butterworth_order_slider = QSlider(Qt.Horizontal)
        self.butterworth_order_slider.setRange(1, 10)
        self.butterworth_order_slider.setValue(4)
        self.butterworth_order_label = QLabel("4")
        butterworth_order_layout = QHBoxLayout()
        butterworth_order_layout.addWidget(self.butterworth_order_slider)
        butterworth_order_layout.addWidget(self.butterworth_order_label)
        self.butterworth_order_row = cutoff_layout.addRow("Butterworth Order:", butterworth_order_layout)
        
        # Initially hide Butterworth order control
        self.butterworth_order_slider.setVisible(False)
        self.butterworth_order_label.setVisible(False)
        
        # Connect sliders to update labels
        self.low_cutoff_slider.valueChanged.connect(self.update_low_cutoff_label)
        self.high_cutoff_slider.valueChanged.connect(self.update_high_cutoff_label)
        self.butterworth_order_slider.valueChanged.connect(self.update_butterworth_order_label)
        
        filter_layout.addLayout(cutoff_layout)
        
        # Filter buttons
        filter_buttons_layout = QHBoxLayout()
        
        preview_filter_btn = QPushButton("Preview Filter")
        preview_filter_btn.clicked.connect(self.preview_fourier_filter)
        filter_buttons_layout.addWidget(preview_filter_btn)
        
        apply_filter_btn = QPushButton("Apply Filter")
        apply_filter_btn.clicked.connect(self.apply_fourier_filter)
        filter_buttons_layout.addWidget(apply_filter_btn)
        
        filter_layout.addLayout(filter_buttons_layout)
        
        layout.addWidget(filter_group)
        
        # Fourier Enhancement
        enhance_group = QGroupBox("Fourier Enhancement")
        enhance_layout = QVBoxLayout(enhance_group)
        
        # Smoothing
        smooth_btn = QPushButton("Fourier Smoothing")
        smooth_btn.clicked.connect(self.apply_fourier_smoothing)
        enhance_layout.addWidget(smooth_btn)
        
        # Deconvolution
        deconv_btn = QPushButton("Richardson-Lucy Deconvolution")
        deconv_btn.clicked.connect(self.apply_richardson_lucy)
        enhance_layout.addWidget(deconv_btn)
        
        # Apodization
        apod_btn = QPushButton("Apply Apodization")
        apod_btn.clicked.connect(self.apply_apodization)
        enhance_layout.addWidget(apod_btn)
        
        layout.addWidget(enhance_group)
        
        layout.addStretch()
        return tab
        
    def create_analysis_tab(self):
        """Create analysis results tab with plotting information."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Information about plotting functionality
        info_group = QGroupBox("📊 Plotting Analysis")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel("""
        <b>Plotting Analysis Features:</b><br><br>
        
        Comprehensive plotting analysis is now available in the <b>Plotting Analysis</b> tab 
        in the right panel. Switch between "Current Spectrum" and "Plotting Analysis" tabs 
        to access advanced visualization tools.<br><br>
        
        <b>Available Plot Types:</b><br>
        • <b>Peak Features Grid:</b> 2×2 plots showing amplitude, position, FWHM, and R² trends<br>
        • <b>Waterfall Plot:</b> Stacked spectra with customizable colors and offsets<br>
        • <b>Heatmap Plot:</b> 2D intensity visualization with multiple colormaps<br><br>
        
        <b>How to Use:</b><br>
        1. Run batch processing to generate data<br>
        2. Click the "Plotting Analysis" tab in the right panel<br>
        3. Choose your plot type and customize settings<br>
        4. Analyze trends across your spectral dataset
        """)
        info_text.setStyleSheet("""
            QLabel {
                color: #333;
                padding: 15px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                line-height: 1.4;
            }
        """)
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # Quick access button
        access_group = QGroupBox("🚀 Quick Access")
        access_layout = QVBoxLayout(access_group)
        
        switch_btn = QPushButton("Switch to Plotting Analysis Tab")
        switch_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        switch_btn.clicked.connect(self.switch_to_plotting_tab)
        access_layout.addWidget(switch_btn)
        
        layout.addWidget(access_group)
        layout.addStretch()
        
        return tab
    
    def switch_to_plotting_tab(self):
        """Switch to the Plotting Analysis tab in the visualization panel."""
        if hasattr(self, 'visualization_tabs'):
            self.visualization_tabs.setCurrentIndex(1)  # Switch to Plotting Analysis tab

    def create_grid_plot_widget(self):
        """Create 2x2 grid plot widget for peak features analysis."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import sys
        import os
        
        # Import matplotlib config
        sys.path.append(os.path.join(os.path.dirname(__file__), 'polarization_ui'))
        try:
            from matplotlib_config import configure_compact_ui, CompactNavigationToolbar
            configure_compact_ui()
        except ImportError:
            pass  # Fallback if config not available
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create matplotlib figure with 2x2 subplots
        self.grid_figure = Figure(figsize=(12, 8))
        self.grid_canvas = FigureCanvas(self.grid_figure)
        
        # Create 2x2 subplots
        self.grid_axes = self.grid_figure.subplots(2, 2)
        self.grid_figure.suptitle('Peak Features Analysis', fontsize=14, fontweight='bold')
        
        # Add navigation toolbar
        try:
            toolbar = CompactNavigationToolbar(self.grid_canvas, widget)
        except:
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
            toolbar = NavigationToolbar2QT(self.grid_canvas, widget)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.grid_canvas)
        
        # Initialize empty plots
        self.setup_empty_grid_plots()
        
        return widget

    def create_waterfall_plot_widget(self):
        """Create waterfall plot widget for stacked spectra visualization."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import sys
        import os
        
        # Import matplotlib config
        sys.path.append(os.path.join(os.path.dirname(__file__), 'polarization_ui'))
        try:
            from matplotlib_config import configure_compact_ui, CompactNavigationToolbar
            configure_compact_ui()
        except ImportError:
            pass
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create matplotlib figure
        self.waterfall_figure = Figure(figsize=(12, 8))
        self.waterfall_canvas = FigureCanvas(self.waterfall_figure)
        self.waterfall_ax = self.waterfall_figure.add_subplot(111)
        
        # Add navigation toolbar
        try:
            toolbar = CompactNavigationToolbar(self.waterfall_canvas, widget)
        except:
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
            toolbar = NavigationToolbar2QT(self.waterfall_canvas, widget)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.waterfall_canvas)
        
        # Initialize empty plot
        self.setup_empty_waterfall_plot()
        
        return widget

    def create_heatmap_plot_widget(self):
        """Create heatmap plot widget for 2D intensity visualization."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import sys
        import os
        
        # Import matplotlib config
        sys.path.append(os.path.join(os.path.dirname(__file__), 'polarization_ui'))
        try:
            from matplotlib_config import configure_compact_ui, CompactNavigationToolbar, add_colorbar_no_shrink
            configure_compact_ui()
        except ImportError:
            pass
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create matplotlib figure
        self.heatmap_figure = Figure(figsize=(12, 8))
        self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
        self.heatmap_ax = self.heatmap_figure.add_subplot(111)
        
        # Add navigation toolbar
        try:
            toolbar = CompactNavigationToolbar(self.heatmap_canvas, widget)
        except:
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
            toolbar = NavigationToolbar2QT(self.heatmap_canvas, widget)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.heatmap_canvas)
        
        # Initialize empty plot
        self.setup_empty_heatmap_plot()
        
        return widget

    def setup_empty_grid_plots(self):
        """Setup empty 2x2 grid plots with proper labels."""
        # Clear all subplots
        for i in range(2):
            for j in range(2):
                self.grid_axes[i, j].clear()
        
        # Setup subplot titles and labels
        titles = ['Peak Amplitude', 'Peak Position', 'FWHM', 'R² Values']
        ylabels = ['Amplitude (a.u.)', 'Position (cm⁻¹)', 'FWHM (cm⁻¹)', 'R² Value']
        
        for idx, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            ax = self.grid_axes[i, j]
            ax.set_title(titles[idx], fontweight='bold')
            ax.set_xlabel('Spectrum Index')
            ax.set_ylabel(ylabels[idx])
            ax.grid(True, alpha=0.3)
            ax.text(0.5, 0.5, 'No data available\nRun batch processing first', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='gray', style='italic')
        
        self.grid_figure.tight_layout()
        self.grid_canvas.draw()

    def setup_empty_waterfall_plot(self):
        """Setup empty waterfall plot."""
        self.waterfall_ax.clear()
        self.waterfall_ax.set_title('Waterfall Plot - Stacked Spectra', fontweight='bold')
        self.waterfall_ax.set_xlabel('Wavenumber (cm⁻¹)')
        self.waterfall_ax.set_ylabel('Intensity + Offset')
        self.waterfall_ax.grid(True, alpha=0.3)
        self.waterfall_ax.text(0.5, 0.5, 'No data available\nRun batch processing first', 
                              transform=self.waterfall_ax.transAxes, ha='center', va='center',
                              fontsize=12, color='gray', style='italic')
        self.waterfall_figure.tight_layout()
        self.waterfall_canvas.draw()

    def setup_empty_heatmap_plot(self):
        """Setup empty heatmap plot."""
        self.heatmap_ax.clear()
        self.heatmap_ax.set_title('Heatmap - Spectral Intensity Map', fontweight='bold')
        self.heatmap_ax.set_xlabel('Wavenumber (cm⁻¹)')
        self.heatmap_ax.set_ylabel('Spectrum Index')
        self.heatmap_ax.text(0.5, 0.5, 'No data available\nRun batch processing first', 
                            transform=self.heatmap_ax.transAxes, ha='center', va='center',
                            fontsize=12, color='gray', style='italic')
        self.heatmap_figure.tight_layout()
        self.heatmap_canvas.draw()

    def on_plot_type_changed(self, button):
        """Handle plot type button changes."""
        plot_index = self.plot_type_buttons.id(button)
        self.analysis_plots_stack.setCurrentIndex(plot_index)
        self.setup_plot_controls()
        self.refresh_current_plot()

    def setup_plot_controls(self):
        """Setup plot-specific controls based on current plot type."""
        # Clear existing controls
        for i in reversed(range(self.plot_controls_layout.count())):
            child = self.plot_controls_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        current_plot = self.analysis_plots_stack.currentIndex()
        
        if current_plot == 0:  # Grid plot
            self.setup_grid_plot_controls()
        elif current_plot == 1:  # Waterfall plot
            self.setup_waterfall_plot_controls()
        elif current_plot == 2:  # Heatmap plot
            self.setup_heatmap_plot_controls()

    def setup_grid_plot_controls(self):
        """Setup controls for 2x2 grid plot."""
        # Peak selection for multi-peak spectra
        peak_selection_layout = QHBoxLayout()
        peak_selection_layout.addWidget(QLabel("Peak to analyze:"))
        
        self.peak_selection_combo = QComboBox()
        self.peak_selection_combo.addItems(["Peak 1", "Peak 2", "Peak 3", "All Peaks (Average)"])
        self.peak_selection_combo.setCurrentText("Peak 1")
        self.peak_selection_combo.currentTextChanged.connect(self.refresh_current_plot)
        peak_selection_layout.addWidget(self.peak_selection_combo)
        
        self.plot_controls_layout.addLayout(peak_selection_layout)
        
        # Error bars option
        self.show_error_bars_check = QCheckBox("Show error bars")
        self.show_error_bars_check.setChecked(False)
        self.show_error_bars_check.stateChanged.connect(self.refresh_current_plot)
        self.plot_controls_layout.addWidget(self.show_error_bars_check)

    def setup_waterfall_plot_controls(self):
        """Setup controls for waterfall plot."""
        # Y-offset control
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Y-offset:"))
        
        self.y_offset_spin = QDoubleSpinBox()
        self.y_offset_spin.setRange(0, 10000)
        self.y_offset_spin.setValue(100)
        self.y_offset_spin.setSuffix(" units")
        self.y_offset_spin.valueChanged.connect(self.refresh_current_plot)
        offset_layout.addWidget(self.y_offset_spin)
        
        self.plot_controls_layout.addLayout(offset_layout)
        
        # Interval skipping
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Show every:"))
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 20)
        self.interval_spin.setValue(1)
        self.interval_spin.setSuffix(" spectrum")
        self.interval_spin.valueChanged.connect(self.refresh_current_plot)
        interval_layout.addWidget(self.interval_spin)
        
        self.plot_controls_layout.addLayout(interval_layout)
        
        # Colormap selection
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        
        self.waterfall_colormap_combo = QComboBox()
        self.waterfall_colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "jet", "rainbow", "cool", "hot"])
        self.waterfall_colormap_combo.setCurrentText("viridis")
        self.waterfall_colormap_combo.currentTextChanged.connect(self.refresh_current_plot)
        colormap_layout.addWidget(self.waterfall_colormap_combo)
        
        self.plot_controls_layout.addLayout(colormap_layout)

    def setup_heatmap_plot_controls(self):
        """Setup controls for heatmap plot."""
        # Colormap selection
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        
        self.heatmap_colormap_combo = QComboBox()
        self.heatmap_colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "jet", "rainbow", "cool", "hot"])
        self.heatmap_colormap_combo.setCurrentText("viridis")
        self.heatmap_colormap_combo.currentTextChanged.connect(self.refresh_current_plot)
        colormap_layout.addWidget(self.heatmap_colormap_combo)
        
        self.plot_controls_layout.addLayout(colormap_layout)
        
        # Interpolation method
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("Interpolation:"))
        
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["nearest", "bilinear", "bicubic", "spline16", "spline36"])
        self.interpolation_combo.setCurrentText("nearest")
        self.interpolation_combo.currentTextChanged.connect(self.refresh_current_plot)
        interp_layout.addWidget(self.interpolation_combo)
        
        self.plot_controls_layout.addLayout(interp_layout)
        
        # Aspect ratio
        aspect_layout = QHBoxLayout()
        aspect_layout.addWidget(QLabel("Aspect ratio:"))
        
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["auto", "equal"])
        self.aspect_combo.setCurrentText("auto")
        self.aspect_combo.currentTextChanged.connect(self.refresh_current_plot)
        aspect_layout.addWidget(self.aspect_combo)
        
        self.plot_controls_layout.addLayout(aspect_layout)

    def update_data_source_info(self):
        """Update data source information display."""
        if hasattr(self, 'batch_results') and self.batch_results:
            total_files = len(self.batch_results)
            total_regions = sum(len(r.get('regions', [])) for r in self.batch_results)
            
            # Count successful fits
            successful_fits = 0
            for file_result in self.batch_results:
                for region_result in file_result.get('regions', []):
                    if region_result.get('total_r2') is not None:
                        successful_fits += 1
            
            info_text = f"✓ {total_files} files, {total_regions} regions, {successful_fits} successful fits"
            self.data_source_label.setText(info_text)
            self.data_source_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 5px;")
        else:
            self.data_source_label.setText("No batch results available")
            self.data_source_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")

    def refresh_analysis_plots(self):
        """Refresh all analysis plots with current data."""
        self.update_data_source_info()
        self.refresh_current_plot()

    def refresh_current_plot(self):
        """Refresh the currently visible plot."""
        if not hasattr(self, 'batch_results') or not self.batch_results:
            return
        
        current_plot = self.analysis_plots_stack.currentIndex()
        
        if current_plot == 0:  # Grid plot
            self.update_grid_plot()
        elif current_plot == 1:  # Waterfall plot
            self.update_waterfall_plot()
        elif current_plot == 2:  # Heatmap plot
            self.update_heatmap_plot()

    def extract_peak_features_data(self):
        """Extract peak features data from batch results."""
        import numpy as np
        
        if not hasattr(self, 'batch_results') or not self.batch_results:
            return None
        
        # Determine which peak to analyze
        peak_selection = self.peak_selection_combo.currentText() if hasattr(self, 'peak_selection_combo') else "Peak 1"
        peak_index = 0  # Default to first peak
        if "Peak 2" in peak_selection:
            peak_index = 1
        elif "Peak 3" in peak_selection:
            peak_index = 2
        
        data = {
            'filenames': [],
            'amplitudes': [],
            'positions': [],
            'fwhms': [],
            'r2_values': [],
            'spectrum_indices': []
        }
        
        spectrum_index = 0
        for file_result in self.batch_results:
            filename = file_result.get('filename', f'Spectrum_{spectrum_index}')
            
            for region_result in file_result.get('regions', []):
                peaks = region_result.get('peaks')
                fit_params = region_result.get('fit_params')
                total_r2 = region_result.get('total_r2')
                
                if peaks is not None and fit_params is not None and len(peaks) > peak_index:
                    # Extract parameters for the selected peak
                    if peak_index * 3 + 2 < len(fit_params):
                        amplitude = fit_params[peak_index * 3]
                        center = fit_params[peak_index * 3 + 1]
                        width = fit_params[peak_index * 3 + 2]
                        
                        # Calculate FWHM
                        fwhm = width * 2 * np.sqrt(2 * np.log(2))
                        
                        data['filenames'].append(filename)
                        data['amplitudes'].append(amplitude)
                        data['positions'].append(center)
                        data['fwhms'].append(fwhm)
                        data['r2_values'].append(total_r2 if total_r2 is not None else 0)
                        data['spectrum_indices'].append(spectrum_index)
                
                spectrum_index += 1
        
        return data if data['filenames'] else None

    def extract_spectral_data(self):
        """Extract spectral data from batch results for waterfall and heatmap plots."""
        if not hasattr(self, 'batch_results') or not self.batch_results:
            return None
        
        spectra_data = []
        filenames = []
        
        for file_result in self.batch_results:
            filename = file_result.get('filename', 'Unknown')
            
            for region_result in file_result.get('regions', []):
                wavenumbers = region_result.get('wavenumbers')
                intensities = region_result.get('intensities')
                
                if wavenumbers is not None and intensities is not None:
                    spectra_data.append({
                        'wavenumbers': wavenumbers,
                        'intensities': intensities,
                        'filename': filename
                    })
                    filenames.append(filename)
        
        return spectra_data if spectra_data else None

    def update_grid_plot(self):
        """Update the 2x2 grid plot with peak features data."""
        import numpy as np
        
        data = self.extract_peak_features_data()
        if data is None:
            self.setup_empty_grid_plots()
            return
        
        # Clear all subplots
        for i in range(2):
            for j in range(2):
                self.grid_axes[i, j].clear()
        
        # Setup subplot titles and labels
        titles = ['Peak Amplitude', 'Peak Position', 'FWHM', 'R² Values']
        ylabels = ['Amplitude (a.u.)', 'Position (cm⁻¹)', 'FWHM (cm⁻¹)', 'R² Value']
        plot_data = [data['amplitudes'], data['positions'], data['fwhms'], data['r2_values']]
        
        x_data = data['spectrum_indices']
        
        for idx, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            ax = self.grid_axes[i, j]
            y_data = plot_data[idx]
            
            if y_data:
                # Plot the data
                ax.plot(x_data, y_data, 'o-', linewidth=2, markersize=6, alpha=0.7)
                
                # Add error bars if requested
                if hasattr(self, 'show_error_bars_check') and self.show_error_bars_check.isChecked():
                    # Calculate simple standard error
                    if len(y_data) > 1:
                        std_err = np.std(y_data) / np.sqrt(len(y_data))
                        ax.errorbar(x_data, y_data, yerr=std_err, fmt='none', alpha=0.5)
                
                # Customize plot
                ax.set_title(titles[idx], fontweight='bold')
                ax.set_xlabel('Spectrum Index')
                ax.set_ylabel(ylabels[idx])
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = np.mean(y_data)
                std_val = np.std(y_data)
                stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, color='gray', style='italic')
        
        self.grid_figure.tight_layout()
        self.grid_canvas.draw()

    def update_waterfall_plot(self):
        """Update the waterfall plot with spectral data."""
        import numpy as np
        import matplotlib.pyplot as plt
        
        spectra_data = self.extract_spectral_data()
        if spectra_data is None:
            self.setup_empty_waterfall_plot()
            return
        
        self.waterfall_ax.clear()
        
        # Get control values
        y_offset = self.y_offset_spin.value() if hasattr(self, 'y_offset_spin') else 100
        interval = self.interval_spin.value() if hasattr(self, 'interval_spin') else 1
        colormap_name = self.waterfall_colormap_combo.currentText() if hasattr(self, 'waterfall_colormap_combo') else 'viridis'
        
        # Select spectra based on interval
        selected_spectra = spectra_data[::interval]
        
        # Get colormap
        colormap = plt.get_cmap(colormap_name)
        n_spectra = len(selected_spectra)
        
        for i, spectrum in enumerate(selected_spectra):
            wavenumbers = spectrum['wavenumbers']
            intensities = spectrum['intensities']
            filename = spectrum['filename']
            
            # Apply offset
            offset_intensities = intensities + (i * y_offset)
            
            # Get color from colormap
            color = colormap(i / max(1, n_spectra - 1))
            
            # Plot spectrum
            self.waterfall_ax.plot(wavenumbers, offset_intensities, 
                                  color=color, linewidth=1.5, alpha=0.8,
                                  label=f'{i}: {filename}')
        
        # Customize plot
        self.waterfall_ax.set_title('Waterfall Plot - Stacked Spectra', fontweight='bold')
        self.waterfall_ax.set_xlabel('Wavenumber (cm⁻¹)')
        self.waterfall_ax.set_ylabel('Intensity + Offset')
        self.waterfall_ax.grid(True, alpha=0.3)
        
        # Add legend if not too many spectra
        if n_spectra <= 10:
            self.waterfall_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Add info text
        info_text = f'Showing {n_spectra} spectra (every {interval})\nOffset: {y_offset} units'
        self.waterfall_ax.text(0.02, 0.98, info_text, transform=self.waterfall_ax.transAxes, 
                              verticalalignment='top', fontsize=9, 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.waterfall_figure.tight_layout()
        self.waterfall_canvas.draw()

    def update_heatmap_plot(self):
        """Update the heatmap plot with spectral data."""
        import numpy as np
        import matplotlib.pyplot as plt
        
        spectra_data = self.extract_spectral_data()
        if spectra_data is None:
            self.setup_empty_heatmap_plot()
            return
        
        self.heatmap_ax.clear()
        
        # Get control values
        colormap_name = self.heatmap_colormap_combo.currentText() if hasattr(self, 'heatmap_colormap_combo') else 'viridis'
        interpolation = self.interpolation_combo.currentText() if hasattr(self, 'interpolation_combo') else 'nearest'
        aspect = self.aspect_combo.currentText() if hasattr(self, 'aspect_combo') else 'auto'
        
        # Prepare data for heatmap
        # Find common wavenumber range
        all_wavenumbers = [spectrum['wavenumbers'] for spectrum in spectra_data]
        min_wn = max(np.min(wn) for wn in all_wavenumbers)
        max_wn = min(np.max(wn) for wn in all_wavenumbers)
        
        # Create common wavenumber grid
        n_points = min(500, min(len(wn) for wn in all_wavenumbers))  # Limit for performance
        common_wavenumbers = np.linspace(min_wn, max_wn, n_points)
        
        # Interpolate all spectra onto common grid
        intensity_matrix = []
        filenames = []
        
        for spectrum in spectra_data:
            wavenumbers = spectrum['wavenumbers']
            intensities = spectrum['intensities']
            filename = spectrum['filename']
            
            # Interpolate to common grid
            interpolated_intensities = np.interp(common_wavenumbers, wavenumbers, intensities)
            intensity_matrix.append(interpolated_intensities)
            filenames.append(filename)
        
        intensity_matrix = np.array(intensity_matrix)
        
        # Create heatmap
        im = self.heatmap_ax.imshow(intensity_matrix, 
                                   cmap=colormap_name,
                                   interpolation=interpolation,
                                   aspect=aspect,
                                   extent=[min_wn, max_wn, len(spectra_data), 0])
        
        # Add colorbar using the no-shrink function
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'polarization_ui'))
            from matplotlib_config import add_colorbar_no_shrink
            add_colorbar_no_shrink(self.heatmap_figure, im, self.heatmap_ax, label='Intensity')
        except ImportError:
            # Fallback to regular colorbar
            self.heatmap_figure.colorbar(im, ax=self.heatmap_ax, label='Intensity')
        
        # Customize plot
        self.heatmap_ax.set_title('Heatmap - Spectral Intensity Map', fontweight='bold')
        self.heatmap_ax.set_xlabel('Wavenumber (cm⁻¹)')
        self.heatmap_ax.set_ylabel('Spectrum Index')
        
        # Add y-axis labels for filenames (if not too many)
        if len(filenames) <= 20:
            y_positions = np.arange(len(filenames))
            self.heatmap_ax.set_yticks(y_positions)
            self.heatmap_ax.set_yticklabels([f'{i}: {name[:15]}...' if len(name) > 15 else f'{i}: {name}' 
                                           for i, name in enumerate(filenames)], fontsize=8)
        
        # Add info text
        info_text = f'Matrix: {intensity_matrix.shape[0]} × {intensity_matrix.shape[1]}\nColormap: {colormap_name}'
        self.heatmap_ax.text(0.02, 0.98, info_text, transform=self.heatmap_ax.transAxes, 
                            verticalalignment='top', fontsize=9, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.heatmap_figure.tight_layout()
        self.heatmap_canvas.draw()

    def create_batch_tab(self):
        """Create batch processing tab for processing multiple spectra."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File management section
        file_group = QGroupBox("📁 File Management")
        file_layout = QVBoxLayout(file_group)
        
        # File list
        self.batch_file_list = QListWidget()
        self.batch_file_list.setMaximumHeight(150)
        file_layout.addWidget(QLabel("Selected Files:"))
        file_layout.addWidget(self.batch_file_list)
        
        # File management buttons
        file_buttons_layout = QHBoxLayout()
        
        add_files_btn = QPushButton("Add Files")
        add_files_btn.clicked.connect(self.add_batch_files)
        file_buttons_layout.addWidget(add_files_btn)
        
        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.clicked.connect(self.add_batch_folder)
        file_buttons_layout.addWidget(add_folder_btn)
        
        clear_files_btn = QPushButton("Clear All")
        clear_files_btn.clicked.connect(self.clear_batch_files)
        file_buttons_layout.addWidget(clear_files_btn)
        
        file_layout.addLayout(file_buttons_layout)
        layout.addWidget(file_group)
        
        # Fitting regions section
        regions_group = QGroupBox("🎯 Fitting Regions")
        regions_layout = QVBoxLayout(regions_group)
        
        # Region controls
        region_controls_layout = QHBoxLayout()
        
        self.region_start_spin = QDoubleSpinBox()
        self.region_start_spin.setRange(0, 4000)  # Initial range, will be updated dynamically
        self.region_start_spin.setValue(400)
        self.region_start_spin.setSuffix(" cm⁻¹")
        region_controls_layout.addWidget(QLabel("Start:"))
        region_controls_layout.addWidget(self.region_start_spin)
        
        self.region_end_spin = QDoubleSpinBox()
        self.region_end_spin.setRange(0, 4000)  # Initial range, will be updated dynamically
        self.region_end_spin.setValue(1600)
        self.region_end_spin.setSuffix(" cm⁻¹")
        region_controls_layout.addWidget(QLabel("End:"))
        region_controls_layout.addWidget(self.region_end_spin)
        
        add_region_btn = QPushButton("Add Region")
        add_region_btn.clicked.connect(self.add_batch_region)
        region_controls_layout.addWidget(add_region_btn)
        
        regions_layout.addLayout(region_controls_layout)
        
        # Regions list
        self.batch_regions_list = QListWidget()
        self.batch_regions_list.setMaximumHeight(80)
        regions_layout.addWidget(self.batch_regions_list)
        
        # Clear regions button
        clear_regions_btn = QPushButton("Clear Regions")
        clear_regions_btn.clicked.connect(self.clear_batch_regions)
        regions_layout.addWidget(clear_regions_btn)
        
        layout.addWidget(regions_group)
        
        # Batch processing section
        processing_group = QGroupBox("⚙️ Batch Processing")
        processing_layout = QVBoxLayout(processing_group)
        
        # Processing settings
        settings_layout = QVBoxLayout()
        
        # Add explanatory text
        settings_info = QLabel("Background correction will be used during batch processing")
        settings_info.setStyleSheet("color: #666; font-style: italic; font-size: 11px; padding: 5px;")
        settings_layout.addWidget(settings_info)
        
        # Peak source selection (radio buttons for exclusive choice)
        peak_source_group = QGroupBox("Peak Source Selection")
        peak_source_layout = QVBoxLayout(peak_source_group)
        
        # Create radio button group
        self.peak_source_group = QButtonGroup()
        
        radio_layout = QHBoxLayout()
        
        self.batch_auto_peaks_radio = QRadioButton("Auto-detect")
        #self.batch_auto_peaks_radio.setText("Auto-detect") # Explicitly set text
        self.batch_auto_peaks_radio.setChecked(True)
        self.batch_auto_peaks_radio.setToolTip("Automatically detect peaks in each spectrum using peak detection parameters")
        self.peak_source_group.addButton(self.batch_auto_peaks_radio, 0)
        radio_layout.addWidget(self.batch_auto_peaks_radio)
        
        self.batch_found_peaks_radio = QRadioButton("Auto fit")
        #self.batch_found_peaks_radio.setText("Auto fit") # Explicitly set text
        self.batch_found_peaks_radio.setChecked(False)
        self.batch_found_peaks_radio.setToolTip("Use the auto-detected peaks in the Find Peaks tab (if available)")
        self.peak_source_group.addButton(self.batch_found_peaks_radio, 1)
        radio_layout.addWidget(self.batch_found_peaks_radio)
        
        self.batch_manual_peaks_radio = QRadioButton("User fit")
        #self.batch_manual_peaks_radio.setText("User fit") # Explicitly set text
        self.batch_manual_peaks_radio.setChecked(False)
        self.batch_manual_peaks_radio.setToolTip("Use manually selected peaks from the main interface (if available)")
        self.peak_source_group.addButton(self.batch_manual_peaks_radio, 2)
        radio_layout.addWidget(self.batch_manual_peaks_radio)
        
        peak_source_layout.addLayout(radio_layout)
        
        # Update manual peaks radio button state initially
        self.update_manual_peaks_radio_state()
        
        settings_layout.addWidget(peak_source_group)
        processing_layout.addLayout(settings_layout)
        
        # Progress bar
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        processing_layout.addWidget(self.batch_progress_bar)
        
        # Process button
        self.batch_process_btn = QPushButton("🚀 Start Batch Processing")
        self.batch_process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.batch_process_btn.clicked.connect(self.start_batch_processing)
        processing_layout.addWidget(self.batch_process_btn)
        
        layout.addWidget(processing_group)
        
        # Results section
        results_group = QGroupBox("📊 Results")
        results_layout = QVBoxLayout(results_group)
        
        self.batch_results_text = QTextEdit()
        self.batch_results_text.setReadOnly(True)
        self.batch_results_text.setMaximumHeight(100)
        results_layout.addWidget(self.batch_results_text)
        
        # Export buttons layout
        export_buttons_layout = QHBoxLayout()
        
        # Export to pickle button
        export_pickle_btn = QPushButton("Export to Pickle")
        export_pickle_btn.clicked.connect(self.export_batch_results)
        export_pickle_btn.setToolTip("Export results to pickle file for advanced analysis modules")
        export_buttons_layout.addWidget(export_pickle_btn)
        
        # Export to CSV button
        export_csv_btn = QPushButton("Export to CSV")
        export_csv_btn.clicked.connect(self.export_batch_to_csv)
        export_csv_btn.setToolTip("Export comprehensive CSV files with peak parameters, backgrounds, and residuals")
        export_buttons_layout.addWidget(export_csv_btn)
        
        results_layout.addLayout(export_buttons_layout)
        
        # Monitor control buttons
        monitor_buttons_layout = QHBoxLayout()
        
        # Reopen monitor button
        reopen_monitor_btn = QPushButton("🔍 View Results Monitor")
        reopen_monitor_btn.clicked.connect(self.reopen_batch_monitor)
        reopen_monitor_btn.setToolTip("Reopen the batch processing monitor to view results")
        monitor_buttons_layout.addWidget(reopen_monitor_btn)
        
        results_layout.addLayout(monitor_buttons_layout)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
        return tab
    
    def create_advanced_tab(self):
        """Create advanced analysis tab with module launchers."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header_label = QLabel("🔬 Advanced Analysis Modules")
        header_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel("Launch specialized analysis modules that work with pickle files from batch processing:")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(desc_label)
        
        # Module launchers
        modules_group = QGroupBox("Analysis Modules")
        modules_layout = QVBoxLayout(modules_group)
        
        # Deconvolution launcher
        deconv_btn = QPushButton("🔧 Spectral Deconvolution")
        deconv_btn.setToolTip("Launch advanced spectral deconvolution tools")
        deconv_btn.clicked.connect(self.launch_deconvolution_module)
        modules_layout.addWidget(deconv_btn)
        
        # Batch data analysis launcher
        batch_analysis_btn = QPushButton("📈 Batch Data Analysis")
        batch_analysis_btn.setToolTip("Analyze batch processing results with advanced statistics")
        batch_analysis_btn.clicked.connect(self.launch_batch_analysis_module)
        modules_layout.addWidget(batch_analysis_btn)
        
        # Geothermometry launcher
        geotherm_btn = QPushButton("🌡️ Geothermometry Analysis")
        geotherm_btn.setToolTip("Temperature analysis based on Raman spectra")
        geotherm_btn.clicked.connect(self.launch_geothermometry_module)
        modules_layout.addWidget(geotherm_btn)
        
        # Density analysis launcher
        density_btn = QPushButton("⚖️ Density Analysis")
        density_btn.setToolTip("Fluid density analysis from Raman spectra")
        density_btn.clicked.connect(self.launch_density_module)
        modules_layout.addWidget(density_btn)
        
        # Map analysis launcher
        map_analysis_btn = QPushButton("🗺️ 2D Map Analysis")
        map_analysis_btn.setToolTip("Analyze 2D Raman mapping data")
        map_analysis_btn.clicked.connect(self.launch_map_analysis_module)
        modules_layout.addWidget(map_analysis_btn)
        
        # Advanced Jupyter Console launcher
        jupyter_console_btn = QPushButton("🐍 Advanced Jupyter Console")
        jupyter_console_btn.setToolTip("Launch advanced Jupyter console with selected pickle file from Data Management")
        jupyter_console_btn.clicked.connect(self.launch_jupyter_console)
        modules_layout.addWidget(jupyter_console_btn)
        
        layout.addWidget(modules_group)
        
        # Data management section
        data_group = QGroupBox("📁 Data Management")
        data_layout = QVBoxLayout(data_group)
        
        # Data file selection
        data_file_layout = QHBoxLayout()
        
        self.data_file_label = QLabel("No pickle file selected")
        self.data_file_label.setStyleSheet("color: #666; font-style: italic;")
        data_file_layout.addWidget(self.data_file_label)
        
        select_data_btn = QPushButton("Select Pickle File")
        select_data_btn.clicked.connect(self.select_batch_data_file)
        data_file_layout.addWidget(select_data_btn)
        
        data_layout.addLayout(data_file_layout)
        
        # Data preview
        self.data_preview_text = QTextEdit()
        self.data_preview_text.setReadOnly(True)
        self.data_preview_text.setMaximumHeight(80)
        self.data_preview_text.setPlainText("Select a pickle file to preview its contents...")
        data_layout.addWidget(self.data_preview_text)
        
        layout.addWidget(data_group)
        
        layout.addStretch()
        return tab
    
    def update_manual_peaks_radio_state(self):
        """Update the manual peaks and found peaks radio button states based on available peaks."""
        # Update manual peaks radio button
        if hasattr(self, 'batch_manual_peaks_radio'):
            has_manual_peaks = (hasattr(self, 'manual_peaks') and 
                               self.manual_peaks is not None and 
                               len(self.manual_peaks) > 0)
            
            if has_manual_peaks:
                count = len(self.manual_peaks)
                self.batch_manual_peaks_radio.setEnabled(True)
                self.batch_manual_peaks_radio.setText(f"User fit")
                self.batch_manual_peaks_radio.setToolTip(f"Use {count} manually selected peaks from the main interface")
            else:
                self.batch_manual_peaks_radio.setEnabled(False)
                self.batch_manual_peaks_radio.setText("User fit")
                self.batch_manual_peaks_radio.setToolTip("No manual peaks available. Select peaks in the main interface first.")
                # If this radio button was selected but no manual peaks available, switch to auto-detect
                if self.batch_manual_peaks_radio.isChecked():
                    self.batch_auto_peaks_radio.setChecked(True)
        
        # Update found peaks radio button
        if hasattr(self, 'batch_found_peaks_radio'):
            has_found_peaks = (hasattr(self, 'peaks') and 
                              self.peaks is not None and 
                              len(self.peaks) > 0)
            
            if has_found_peaks:
                count = len(self.peaks)
                self.batch_found_peaks_radio.setEnabled(True)
                self.batch_found_peaks_radio.setText(f"Auto fit")
                self.batch_found_peaks_radio.setToolTip(f"Use {count} peaks found in the Find Peaks tab")
            else:
                self.batch_found_peaks_radio.setEnabled(False)
                self.batch_found_peaks_radio.setText("Auto fit")
                self.batch_found_peaks_radio.setToolTip("No peaks found yet. Use Find Peaks tab to detect peaks first.")
                # If this radio button was selected but no found peaks available, switch to auto-detect
                if self.batch_found_peaks_radio.isChecked():
                    self.batch_auto_peaks_radio.setChecked(True)

    # Implementation methods
    def initial_plot(self):
        """Create initial plot."""
        if len(self.wavenumbers) > 0:
            self.update_plot()
        else:
            # Show empty plot with instruction message
            self.setup_empty_plot()
    
    def setup_empty_plot(self):
        """Setup an empty plot with instruction message."""
        # Clear axes
        self.ax_main.clear()
        self.ax_residual.clear()
        
        # Add instruction text
        self.ax_main.text(0.5, 0.5, 
                         'No spectrum data loaded\n\n'
                         'Use File → Open Spectrum File (Ctrl+O)\n'
                         'to load a spectrum file\n\n'
                         'Supported formats: .txt, .csv, .dat, .asc, .spc',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=self.ax_main.transAxes,
                         fontsize=12,
                         bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Spectral Deconvolution & Advanced Analysis - No Data')
        self.ax_main.grid(True, alpha=0.3)
        
        # Empty residuals plot
        self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_residual.set_ylabel('Residuals')
        self.ax_residual.set_title('Residuals - No Data')
        self.ax_residual.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    # Debounced update methods for responsive live preview
    def _trigger_background_update(self):
        """Trigger debounced background update."""
        self.bg_update_timer.stop()
        self.bg_update_timer.start(self.bg_update_delay)
    
    def _trigger_peak_update(self):
        """Trigger debounced peak detection update."""
        self.peak_update_timer.stop()
        self.peak_update_timer.start(self.peak_update_delay)
    
    def _update_background_calculation(self):
        """Perform background calculation and update plot efficiently using unified widget."""
        try:
            if CENTRALIZED_AVAILABLE and hasattr(self, 'background_controls'):
                # Get parameters from unified background controls widget
                try:
                    params = self.background_controls.get_background_parameters()
                    method = params['method']
                except (AttributeError, Exception) as e:
                    print(f"Warning: Could not get background parameters: {e}")
                    # Fallback to default parameters
                    params = {'method': 'ALS', 'lambda': 1e5, 'p': 0.01, 'niter': 10}
                    method = 'ALS'
                
                if method == "ALS":
                    # Calculate background using ALS with unified parameters
                    self.background = self._get_baseline_fitter().baseline_als(
                        self.original_intensities, 
                        params['lambda'], 
                        params['p'], 
                        params['niter']
                    )
                else:
                    # Handle other methods
                    self.background = self._calculate_background_with_method(method, params)
            else:
                # Fallback method selection
                params = self.get_fallback_background_parameters()
                method = params['method']
                
                if method == "ALS":
                    self.background = self._get_baseline_fitter().baseline_als(
                        self.original_intensities, 
                        params['lambda'], 
                        params['p'], 
                        params['niter']
                    )
                else:
                    # Handle other methods
                    self.background = self._calculate_background_with_method(method, params)
            
            # Set preview flag
            self.background_preview_active = True
            
            # Efficient plot update - only update background line
            self._update_background_line()
            
        except Exception as e:
            print(f"Background preview error: {str(e)}")
    
    def _calculate_background_with_method(self, method, params):
        """Calculate background using specified method and parameters."""
        if method == "Linear":
            return self._calculate_linear_background(
                params.get('start_weight', 1.0), 
                params.get('end_weight', 1.0)
            )
        elif method == "Polynomial":
            return self._calculate_polynomial_background(
                params.get('order', 2), 
                params.get('poly_method', "Least Squares")
            )
        elif method == "Moving Average":
            return self._calculate_moving_average_background(
                params.get('window_percent', 15),
                params.get('window_type', "Uniform")
            )
        elif method == "Spline":
            return self._calculate_spline_background_for_subtraction(
                params.get('n_knots', 10),
                params.get('smoothing', 100),
                params.get('degree', 3)
            )
        else:
            # Default to ALS
            return self._get_baseline_fitter().baseline_als(self.original_intensities)
    
    def _calculate_background_for_batch(self, intensities, method, params):
        """Calculate background for batch processing using specified method and parameters.
        
        Args:
            intensities: Input intensities (numpy array)
            method: Background method name
            params: Parameter dictionary
            
        Returns:
            numpy array: Background values
        """
        if method == "Linear":
            return self._calculate_linear_background_for_batch(intensities, 
                params.get('start_weight', 1.0), 
                params.get('end_weight', 1.0)
            )
        elif method == "Polynomial":
            return self._calculate_polynomial_background_for_batch(intensities,
                params.get('order', 2), 
                params.get('poly_method', "Least Squares")
            )
        elif method == "Moving Average":
            return self._calculate_moving_average_background_for_batch(intensities,
                params.get('window_percent', 15),
                params.get('window_type', "Uniform")
            )
        elif method == "Spline":
            return self._calculate_spline_background_for_batch(intensities,
                params.get('n_knots', 10),
                params.get('smoothing', 100),
                params.get('degree', 3)
            )
        else:
            # Default to ALS
            return self._get_baseline_fitter().baseline_als(
                intensities,
                params.get('lambda', 1e5),
                params.get('p', 0.01),
                params.get('niter', 10)
            )
    
    def _calculate_linear_background(self, start_weight, end_weight):
        """Calculate linear background between weighted endpoints."""
        try:
            y = self.original_intensities
            start_val = y[0] * start_weight
            end_val = y[-1] * end_weight
            return np.linspace(start_val, end_val, len(y))
        except Exception as e:
            print(f"Linear background calculation error: {str(e)}")
            return np.linspace(y[0], y[-1], len(y))
    
    def _calculate_linear_background_for_batch(self, intensities, start_weight, end_weight):
        """Calculate linear background for batch processing."""
        try:
            start_val = intensities[0] * start_weight
            end_val = intensities[-1] * end_weight
            return np.linspace(start_val, end_val, len(intensities))
        except Exception as e:
            print(f"Linear background calculation error: {str(e)}")
            return np.linspace(intensities[0], intensities[-1], len(intensities))
    
    def _calculate_polynomial_background(self, order, method):
        """Calculate polynomial background fit using proper baseline estimation."""
        try:
            y = self.original_intensities
            x = np.arange(len(y))
            
            # Step 1: Apply minimum filtering to identify baseline regions
            from scipy import ndimage
            window_size = max(len(y) // 20, 5)
            y_min_filtered = ndimage.minimum_filter1d(y, size=window_size)
            
            # Step 2: Use iterative approach to fit polynomial to baseline
            if method == "Robust":
                # Robust baseline estimation
                current_background = y_min_filtered.copy()
                
                for iteration in range(3):  # Multiple iterations for refinement
                    # Fit polynomial to current baseline estimate
                    coeffs = np.polyfit(x, current_background, min(order, len(y)-1))
                    fitted_background = np.polyval(coeffs, x)
                    
                    # Identify points that are likely background (below fitted curve)
                    residuals = y - fitted_background
                    threshold = np.percentile(residuals, 25)  # Use lower quartile
                    baseline_mask = residuals <= threshold
                    
                    if np.sum(baseline_mask) < order + 1:  # Need enough points
                        break
                    
                    # Weighted fit with higher weights for baseline points
                    weights = np.ones_like(y)
                    weights[baseline_mask] = 2.0  # Higher weight for baseline
                    weights[~baseline_mask] = 0.1  # Lower weight for peaks
                    
                    # Robust reweighting
                    if iteration > 0:
                        abs_residuals = np.abs(y - fitted_background)
                        median_residual = np.median(abs_residuals)
                        robust_weights = 1.0 / (1.0 + abs_residuals / (median_residual + 1e-10))
                        weights *= robust_weights
                    
                    coeffs = np.polyfit(x, y, min(order, len(y)-1), w=weights)
                    current_background = np.polyval(coeffs, x)
                    
                    # Ensure background doesn't go above original data
                    current_background = np.minimum(current_background, y)
                
                background = current_background
                
            else:
                # Standard baseline estimation
                # Start with minimum filtered data
                baseline_points = y_min_filtered.copy()
                
                # Iteratively refine baseline
                for iteration in range(2):
                    # Fit polynomial to baseline points
                    coeffs = np.polyfit(x, baseline_points, min(order, len(y)-1))
                    fitted_background = np.polyval(coeffs, x)
                    
                    # Update baseline points - use minimum of fitted curve and original data
                    baseline_points = np.minimum(fitted_background, y)
                    
                    # Further constrain to lower envelope
                    residuals = y - baseline_points
                    threshold = np.percentile(residuals, 30)
                    mask = residuals <= threshold
                    
                    if np.sum(mask) >= order + 1:
                        # Fit only to points identified as baseline
                        coeffs = np.polyfit(x[mask], y[mask], min(order, len(y)-1))
                        baseline_points = np.polyval(coeffs, x)
                        baseline_points = np.minimum(baseline_points, y)
                
                background = baseline_points
            
            # Final constraint: ensure background is below original data
            background = np.minimum(background, y)
            
            return background
            
        except ImportError:
            print("scipy not available, using simple polynomial baseline estimation")
            # Simple fallback without scipy
            x = np.arange(len(y))
            
            # Simple approach: fit polynomial to lower envelope
            window_size = max(len(y) // 10, 3)
            y_smooth = np.array([np.min(y[max(0, i-window_size):min(len(y), i+window_size+1)]) 
                               for i in range(len(y))])
            
            # Fit polynomial to smoothed minimum
            coeffs = np.polyfit(x, y_smooth, min(order, len(y)-1))
            background = np.polyval(coeffs, x)
            
            return np.minimum(background, y)
            
        except Exception as e:
            print(f"Polynomial background calculation error: {str(e)}")
            # Fallback to linear
            return np.linspace(y[0], y[-1], len(y))
    
    def _calculate_polynomial_background_for_batch(self, intensities, order, method):
        """Calculate polynomial background for batch processing using proper baseline estimation."""
        try:
            x = np.arange(len(intensities))
            
            # Step 1: Apply minimum filtering to identify baseline regions
            from scipy import ndimage
            window_size = max(len(intensities) // 20, 5)
            y_min_filtered = ndimage.minimum_filter1d(intensities, size=window_size)
            
            # Step 2: Use iterative approach to fit polynomial to baseline
            if method == "Robust":
                # Robust baseline estimation
                current_background = y_min_filtered.copy()
                
                for iteration in range(3):  # Multiple iterations for refinement
                    # Fit polynomial to current baseline estimate
                    coeffs = np.polyfit(x, current_background, min(order, len(intensities)-1))
                    fitted_background = np.polyval(coeffs, x)
                    
                    # Identify points that are likely background (below fitted curve)
                    residuals = intensities - fitted_background
                    threshold = np.percentile(residuals, 25)  # Use lower quartile
                    baseline_mask = residuals <= threshold
                    
                    if np.sum(baseline_mask) < order + 1:  # Need enough points
                        break
                    
                    # Weighted fit with higher weights for baseline points
                    weights = np.ones_like(intensities)
                    weights[baseline_mask] = 2.0  # Higher weight for baseline
                    weights[~baseline_mask] = 0.1  # Lower weight for peaks
                    
                    # Robust reweighting
                    if iteration > 0:
                        abs_residuals = np.abs(intensities - fitted_background)
                        median_residual = np.median(abs_residuals)
                        robust_weights = 1.0 / (1.0 + abs_residuals / (median_residual + 1e-10))
                        weights *= robust_weights
                    
                    coeffs = np.polyfit(x, intensities, min(order, len(intensities)-1), w=weights)
                    current_background = np.polyval(coeffs, x)
                    
                    # Ensure background doesn't go above original data
                    current_background = np.minimum(current_background, intensities)
                
                background = current_background
                
            else:
                # Standard baseline estimation
                # Start with minimum filtered data
                baseline_points = y_min_filtered.copy()
                
                # Iteratively refine baseline
                for iteration in range(2):
                    # Fit polynomial to baseline points
                    coeffs = np.polyfit(x, baseline_points, min(order, len(intensities)-1))
                    fitted_background = np.polyval(coeffs, x)
                    
                    # Update baseline points - use minimum of fitted curve and original data
                    baseline_points = np.minimum(fitted_background, intensities)
                    
                    # Further constrain to lower envelope
                    residuals = intensities - baseline_points
                    threshold = np.percentile(residuals, 30)
                    mask = residuals <= threshold
                    
                    if np.sum(mask) >= order + 1:
                        # Fit only to points identified as baseline
                        coeffs = np.polyfit(x[mask], intensities[mask], min(order, len(intensities)-1))
                        baseline_points = np.polyval(coeffs, x)
                        baseline_points = np.minimum(baseline_points, intensities)
                
                background = baseline_points
            
            # Final constraint: ensure background is below original data
            background = np.minimum(background, intensities)
            
            return background
            
        except ImportError:
            print("scipy not available, using simple polynomial baseline estimation")
            # Simple fallback without scipy
            x = np.arange(len(intensities))
            
            # Simple approach: fit polynomial to lower envelope
            window_size = max(len(intensities) // 10, 3)
            y_smooth = np.array([np.min(intensities[max(0, i-window_size):min(len(intensities), i+window_size+1)]) 
                               for i in range(len(intensities))])
            
            # Fit polynomial to smoothed minimum
            coeffs = np.polyfit(x, y_smooth, min(order, len(intensities)-1))
            background = np.polyval(coeffs, x)
            
            return np.minimum(background, intensities)
            
        except Exception as e:
            print(f"Polynomial background calculation error: {str(e)}")
            # Fallback to linear
            return np.linspace(intensities[0], intensities[-1], len(intensities))
    
    def _update_peak_detection(self):
        """Perform live peak detection using centralized or fallback methods."""
        try:
            if CENTRALIZED_AVAILABLE and hasattr(self, 'peak_controls'):
                # Use centralized peak detection
                params = self.peak_controls.get_peak_parameters()
                
                # Use centralized auto_find_peaks function
                self.peaks = auto_find_peaks(
                    self.processed_intensities,
                    height=params['height'],
                    distance=params['distance'],
                    prominence=params['prominence']
                )
                
                # Update current model from centralized controls
                self.current_model = params['model']
                
            else:
                # Fallback to manual slider values
                height_percent = self.height_slider.value()
                distance = self.distance_slider.value()
                prominence_percent = self.prominence_slider.value()
                
                # Calculate thresholds - ensure they are scalars
                max_intensity = float(np.max(self.processed_intensities))
                height_threshold = (height_percent / 100.0) * max_intensity if height_percent > 0 else None
                prominence_threshold = (prominence_percent / 100.0) * max_intensity if prominence_percent > 0 else None
                
                # Ensure distance is an integer
                distance = int(distance) if distance > 0 else None
                
                # Find peaks with proper parameter handling using scipy directly
                peak_kwargs = {}
                if height_threshold is not None:
                    peak_kwargs['height'] = height_threshold
                if distance is not None:
                    peak_kwargs['distance'] = distance
                if prominence_threshold is not None:
                    peak_kwargs['prominence'] = prominence_threshold
                
                self.peaks, properties = find_peaks(self.processed_intensities, **peak_kwargs)
            
            self.update_peak_count_display()
            
            # Update batch radio buttons state
            self.update_manual_peaks_radio_state()
            
            # Efficient plot update - only update peak markers
            self._update_peak_markers()
            
        except Exception as e:
            print(f"Live peak detection error: {str(e)}")
    
    def _update_background_line(self):
        """Efficiently update only the background line in the plot."""
        if self.background is not None and self.ax_main is not None:
            # Remove existing background line if it exists
            if self.background_line is not None:
                try:
                    self.background_line.remove()
                except:
                    pass
            
            # Add new background line
            self.background_line, = self.ax_main.plot(
                self.wavenumbers, self.background, 'r--', 
                linewidth=1, alpha=0.7, label='Background'
            )
            
            # Update legend
            self.ax_main.legend()
            
            # Redraw canvas
            self.canvas.draw_idle()
    
    def _update_peak_markers(self):
        """Efficiently update only the peak markers in the plot."""
        # Remove existing peak markers
        if self.auto_peaks_scatter is not None:
            try:
                self.auto_peaks_scatter.remove()
            except:
                pass
            self.auto_peaks_scatter = None
        
        # Add new auto peak markers
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            valid_auto_peaks = self.validate_peak_indices(self.peaks)
            if len(valid_auto_peaks) > 0:
                auto_peak_positions = [self.wavenumbers[p] for p in valid_auto_peaks]
                auto_peak_intensities = [self.processed_intensities[p] for p in valid_auto_peaks]
                self.auto_peaks_scatter = self.ax_main.scatter(
                    auto_peak_positions, auto_peak_intensities, 
                    c='red', s=64, marker='o', label='Auto Peaks', alpha=0.8, zorder=5
                )
        
        # Update legend
        self.ax_main.legend()
        
        # Update peak list
        if hasattr(self, 'peak_list_widget'):
            self.update_peak_list()
        
        # Redraw canvas
        self.canvas.draw_idle()
    
    def _update_manual_peak_markers(self):
        """Efficiently update only the manual peak markers in the plot."""
        # Remove existing manual peak markers
        if self.manual_peaks_scatter is not None:
            try:
                self.manual_peaks_scatter.remove()
            except:
                pass
            self.manual_peaks_scatter = None
        
        # Add new manual peak markers
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            valid_manual_peaks = self.validate_peak_indices(self.manual_peaks)
            if len(valid_manual_peaks) > 0:
                manual_peak_positions = [self.wavenumbers[p] for p in valid_manual_peaks]
                manual_peak_intensities = [self.processed_intensities[p] for p in valid_manual_peaks]
                self.manual_peaks_scatter = self.ax_main.scatter(
                    manual_peak_positions, manual_peak_intensities, 
                    c='green', s=100, marker='s', label='Manual Peaks', 
                    alpha=0.8, edgecolor='darkgreen', zorder=5
                )
        
        # Update legend
        self.ax_main.legend()
        
        # Update peak list
        if hasattr(self, 'peak_list_widget'):
            self.update_peak_list()
        
        # Redraw canvas
        self.canvas.draw_idle()
        
    def update_plot(self):
        """Update all plots with line reference storage for efficient updates."""
        # Clear all axes and reset line references
        self.ax_main.clear()
        self.ax_residual.clear()
        self._reset_line_references()
        
        # Main spectrum plot
        self.spectrum_line, = self.ax_main.plot(self.wavenumbers, self.processed_intensities, 'b-', 
                                              linewidth=1.5, label='Spectrum')
        
        # Plot background if available
        if self.background is not None:
            self.background_line, = self.ax_main.plot(self.wavenumbers, self.background, 'r--', 
                                                    linewidth=1, alpha=0.7, label='Background')
        
        # Plot fitted peaks if available
        if self.fit_result is not None and self.show_individual_peaks:
            # Plot total fitted curve
            fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
            
            # Calculate total R² for the global fit
            total_r2 = self.calculate_total_r2()
            
            self.fitted_line, = self.ax_main.plot(self.wavenumbers, fitted_curve, 'g-', 
                                                linewidth=2, label=f'Total Fit (R²={total_r2:.4f})')
            
            # Plot individual peaks
            self.plot_individual_peaks()
        
        # Plot automatically detected peaks - Handle numpy array boolean check properly
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            # Validate peak indices before plotting
            valid_auto_peaks = self.validate_peak_indices(self.peaks)
            if len(valid_auto_peaks) > 0:
                auto_peak_positions = [self.wavenumbers[p] for p in valid_auto_peaks]
                auto_peak_intensities = [self.processed_intensities[p] for p in valid_auto_peaks]
                self.auto_peaks_scatter = self.ax_main.scatter(auto_peak_positions, auto_peak_intensities, 
                                                             c='red', s=64, marker='o', label='Auto Peaks', 
                                                             alpha=0.8, zorder=5)
        
        # Plot manually selected peaks - Validate indices to prevent IndexError
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            # Validate peak indices before plotting
            valid_manual_peaks = self.validate_peak_indices(self.manual_peaks)
            if len(valid_manual_peaks) > 0:
                manual_peak_positions = [self.wavenumbers[p] for p in valid_manual_peaks]
                manual_peak_intensities = [self.processed_intensities[p] for p in valid_manual_peaks]
                self.manual_peaks_scatter = self.ax_main.scatter(manual_peak_positions, manual_peak_intensities, 
                                                               c='green', s=100, marker='s', label='Manual Peaks', 
                                                               alpha=0.8, edgecolor='darkgreen', zorder=5)
        
        # Add interactive mode indicator
        if self.interactive_mode:
            self.ax_main.text(0.02, 0.98, '🖱️ Interactive Mode ON\nClick to select peaks', 
                             transform=self.ax_main.transAxes, 
                             verticalalignment='top', fontsize=10,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Spectrum and Peak Analysis')
        
        # Show/hide legend based on checkbox
        if hasattr(self, 'show_legend_check') and self.show_legend_check.isChecked():
            self.ax_main.legend()
        
        # Show/hide grid based on checkbox
        if hasattr(self, 'show_grid_check'):
            self.ax_main.grid(self.show_grid_check.isChecked(), alpha=0.3)
        else:
            self.ax_main.grid(True, alpha=0.3)  # Default behavior
        
        # Residuals plot
        if self.residuals is not None:
            self.ax_residual.plot(self.wavenumbers, self.residuals, 'k-', linewidth=1)
            self.ax_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_residual.set_ylabel('Residuals')
            self.ax_residual.set_title('Fit Residuals')
            
            # Show/hide grid based on checkbox for residuals plot too
            if hasattr(self, 'show_grid_check'):
                self.ax_residual.grid(self.show_grid_check.isChecked(), alpha=0.3)
            else:
                self.ax_residual.grid(True, alpha=0.3)  # Default behavior
        
        # Update peak list if it exists
        if hasattr(self, 'peak_list_widget'):
            self.update_peak_list()
        
        # Update status bar
        if hasattr(self, 'status_bar'):
            self.update_status_bar()
        
        self.canvas.draw()
    
    def _reset_line_references(self):
        """Reset all line references for clean plot updates."""
        self.spectrum_line = None
        self.background_line = None
        self.fitted_line = None
        self.auto_peaks_scatter = None
        self.manual_peaks_scatter = None
        self.individual_peak_lines = []
        self.filter_preview_line = None

    def validate_peak_indices(self, peak_indices):
        """Validate peak indices to ensure they're within bounds and are integers."""
        if peak_indices is None or len(peak_indices) == 0:
            return np.array([])
        
        valid_peaks = []
        max_index = len(self.wavenumbers) - 1
        
        for peak_idx in peak_indices:
            try:
                # Convert to integer if possible
                peak_idx = int(peak_idx)
                
                # Check if within bounds
                if 0 <= peak_idx <= max_index:
                    valid_peaks.append(peak_idx)
                    
            except (ValueError, TypeError):
                # Skip invalid indices
                continue
        
        return np.array(valid_peaks, dtype=int)

    def plot_individual_peaks(self):
        """Plot individual fitted peaks with different colors and R² values."""
        # Check if we have fit parameters and any peaks to plot
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0):
            return
        
        # Get all peaks that were used for fitting (auto + manual)
        all_fitted_peaks = self.get_all_peaks_for_fitting()
        if len(all_fitted_peaks) == 0:
            return
        
        # Validate peaks before using them
        validated_peaks = self.validate_peak_indices(np.array(all_fitted_peaks))
        if len(validated_peaks) == 0:
            return
            
        n_peaks = len(validated_peaks)
        model = self.current_model
        
        # Define a color palette for individual peaks
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        # Calculate individual R² values for each peak
        individual_r2_values = self.calculate_individual_r2_values()
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                
                # Generate individual peak curve
                if model == "Gaussian":
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                elif model == "Lorentzian":
                    peak_curve = self.lorentzian(self.wavenumbers, amp, cen, wid)
                elif model == "Pseudo-Voigt":
                    peak_curve = self.pseudo_voigt(self.wavenumbers, amp, cen, wid)
                else:
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)  # Default
                
                # Select color from palette (cycle if more peaks than colors)
                color = colors[i % len(colors)]
                
                # Get individual R² value
                r2_value = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                
                # Create label with peak info and R²
                label = f'Peak {i+1} ({cen:.1f} cm⁻¹, R²={r2_value:.3f})'
                
                # Plot individual peak
                line, = self.ax_main.plot(self.wavenumbers, peak_curve, '--', 
                                        linewidth=1.5, alpha=0.8, color=color,
                                        label=label)
                self.individual_peak_lines.append(line)
                
                # Add peak position label on the plot
                peak_max_idx = np.argmax(peak_curve)
                peak_max_intensity = peak_curve[peak_max_idx]
                
                # Offset label slightly above the peak
                label_y = peak_max_intensity + np.max(self.processed_intensities) * 0.05
                
                self.ax_main.annotate(f'{cen:.1f}', 
                                    xy=(cen, peak_max_intensity),
                                    xytext=(cen, label_y),
                                    ha='center', va='bottom',
                                    fontsize=9, fontweight='bold',
                                    color=color,
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='white', 
                                            edgecolor=color, 
                                            alpha=0.8),
                                    arrowprops=dict(arrowstyle='->', 
                                                  color=color, 
                                                  alpha=0.6, 
                                                  lw=1))
    
    def calculate_individual_r2_values(self):
        """Calculate R² values for individual peaks using a straightforward regional method."""
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0):
            return []
        
        all_fitted_peaks = self.get_all_peaks_for_fitting()
        if len(all_fitted_peaks) == 0:
            return []
        
        validated_peaks = self.validate_peak_indices(np.array(all_fitted_peaks))
        n_peaks = len(validated_peaks)
        model = self.current_model
        
        individual_r2_values = []
        
        # Generate total fit for reference
        total_fit = self.multi_peak_model(self.wavenumbers, *self.fit_params)
        
        # Calculate R² for each peak in its local region
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                
                # Generate individual peak curve
                if model == "Gaussian":
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                elif model == "Lorentzian":
                    peak_curve = self.lorentzian(self.wavenumbers, amp, cen, wid)
                elif model == "Pseudo-Voigt":
                    peak_curve = self.pseudo_voigt(self.wavenumbers, amp, cen, wid)
                else:
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                
                # Calculate R² for this peak in a focused region
                individual_r2 = self._calculate_simple_regional_r2(cen, wid, peak_curve, total_fit)
                individual_r2_values.append(individual_r2)
            else:
                individual_r2_values.append(0.0)
        
        return individual_r2_values
    
    def _calculate_simple_regional_r2(self, peak_center, peak_width, peak_curve, total_fit):
        """Calculate R² for a peak using a simple, focused regional approach."""
        try:
            # Define region around peak center (3 times the width, minimum 20 cm⁻¹)
            region_width = max(abs(peak_width) * 3, 20)
            region_start = peak_center - region_width
            region_end = peak_center + region_width
            
            # Find indices within this region
            region_mask = (self.wavenumbers >= region_start) & (self.wavenumbers <= region_end)
            
            if not np.any(region_mask) or np.sum(region_mask) < 5:
                return 0.0
            
            # Extract regional data
            region_data = self.processed_intensities[region_mask]
            region_total_fit = total_fit[region_mask]
            region_individual_peak = peak_curve[region_mask]
            
            # Method: Compare how much the individual peak contributes to the total fit quality in this region
            # Calculate what the fit would be WITHOUT this peak
            fit_without_this_peak = region_total_fit - region_individual_peak
            
            # Calculate residuals with and without this peak
            residuals_with_peak = region_data - region_total_fit
            residuals_without_peak = region_data - fit_without_this_peak
            
            # R² represents how much this peak improves the fit
            ss_res_with = np.sum(residuals_with_peak ** 2)
            ss_res_without = np.sum(residuals_without_peak ** 2)
            
            # The improvement ratio gives us the individual peak R²
            if ss_res_without > 0:
                improvement_ratio = (ss_res_without - ss_res_with) / ss_res_without
                r2 = max(0.0, min(1.0, improvement_ratio))
            else:
                # If no improvement possible, calculate direct correlation
                r2 = self._calculate_correlation_r2(region_data, region_individual_peak, region_total_fit)
            
            return r2
            
        except Exception as e:
            return 0.0
    
    def _calculate_correlation_r2(self, region_data, peak_curve, total_fit):
        """Calculate R² based on how well the peak correlates with the data in its region."""
        try:
            # Remove baseline trend
            baseline = np.linspace(region_data[0], region_data[-1], len(region_data))
            data_corrected = region_data - baseline
            
            # Scale peak to match the data magnitude in this region
            if np.max(peak_curve) > 0:
                peak_scaled = peak_curve * (np.max(data_corrected) / np.max(peak_curve))
            else:
                return 0.0
            
            # Calculate correlation-based R²
            mean_data = np.mean(data_corrected)
            ss_tot = np.sum((data_corrected - mean_data) ** 2)
            
            if ss_tot > 0:
                # Use the scaled peak as the model
                ss_res = np.sum((data_corrected - peak_scaled) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                return max(0.0, min(1.0, r2))
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_regional_r2(self, peak_index, peak_curve, all_individual_peaks):
        """Calculate R² for a peak in its local region, accounting for overlaps."""
        try:
            start_idx = peak_index * 3
            if start_idx + 2 >= len(self.fit_params):
                return 0.0
                
            amp, cen, wid = self.fit_params[start_idx:start_idx+3]
            
            # Define a region around this peak (2.5 sigma to reduce overlap issues)
            peak_width = abs(wid) * 2.5
            region_mask = (self.wavenumbers >= cen - peak_width) & \
                         (self.wavenumbers <= cen + peak_width)
            
            if not np.any(region_mask):
                return 0.0
            
            # Extract regional data
            region_data = self.processed_intensities[region_mask]
            region_peak = peak_curve[region_mask]
            
            # Calculate baseline for this region (linear interpolation)
            region_wavenumbers = self.wavenumbers[region_mask]
            if len(region_wavenumbers) < 3:
                return 0.0
                
            baseline = np.linspace(region_data[0], region_data[-1], len(region_data))
            region_data_corrected = region_data - baseline
            
            # Calculate R² comparing peak to baseline-corrected data
            mean_data = np.mean(region_data_corrected)
            ss_tot = np.sum((region_data_corrected - mean_data) ** 2)
            
            if ss_tot > 0:
                # For overlapping regions, subtract contributions from other significant peaks
                other_peaks_contribution = np.zeros_like(region_peak)
                for j, other_peak in enumerate(all_individual_peaks):
                    if j != peak_index:
                        other_region_peak = other_peak[region_mask]
                        # Only subtract if the other peak contributes significantly in this region
                        if np.max(other_region_peak) > np.max(region_peak) * 0.1:
                            other_peaks_contribution += other_region_peak
                
                # Adjusted data = original - other peaks
                adjusted_data = region_data_corrected - other_peaks_contribution
                
                ss_res = np.sum((adjusted_data - region_peak) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                return max(0.0, min(1.0, r2))
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def calculate_total_r2(self):
        """Calculate total R² for the global fit."""
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0):
            return 0.0
        
        # Generate total fitted curve
        fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
        
        # Calculate R² for global fit
        ss_res = np.sum((self.processed_intensities - fitted_curve) ** 2)
        ss_tot = np.sum((self.processed_intensities - np.mean(self.processed_intensities)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Ensure R² is reasonable (between 0 and 1)
        return max(0.0, min(1.0, r2))

    def plot_analysis_results(self):
        """Plot PCA/NMF analysis results on the main plot."""
        # Note: This method is kept for compatibility but analysis results
        # will now be displayed on the main plot when PCA/NMF methods are called
        pass

    # Background subtraction methods
    def on_bg_method_changed(self, method=None):
        """Handle change in background method."""
        # Clear any active background preview when method changes
        if hasattr(self, 'background_preview_active') and self.background_preview_active:
            self.clear_background_preview()

    # REFACTORED: Removed all background label update methods - handled by unified widget now!

    def clear_background_preview(self):
        """Clear the background preview."""
        self.background = None
        self.background_preview_active = False
        self.update_plot()

    def preview_background(self):
        """Preview background subtraction (legacy method)."""
        self._trigger_background_update()
    
    def apply_background(self):
        """Apply background subtraction."""
        if self.background is not None:
            self.processed_intensities = self.original_intensities - self.background
            # Clear preview flag
            self.background_preview_active = False
            self.update_plot()
        else:
            # If no preview, generate background first
            self._update_background_calculation()
            if self.background is not None:
                self.apply_background()
    
    def reset_spectrum(self):
        """Reset to original spectrum."""
        self.processed_intensities = self.original_intensities.copy()
        self.background = None
        self.background_preview_active = False  # Clear background preview state
        self.peaks = np.array([])  # Reset as empty numpy array
        self.manual_peaks = np.array([])  # Reset manual peaks
        self.fit_params = []
        self.fit_result = None
        self.residuals = None  # Clear residuals
        self.components = []
        
        # Clear background options
        self.clear_background_options()
        
        # Disable interactive mode
        if self.interactive_mode:
            self.interactive_mode = False
            self.interactive_btn.setChecked(False)
            self.interactive_btn.setText("🖱️ Enable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: OFF")
            self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Disconnect mouse click event
            if self.click_connection is not None:
                self.canvas.mpl_disconnect(self.click_connection)
                self.click_connection = None
        
        self.update_peak_count_display()
        self.update_peak_list()  # Update the peak list widget
        self.update_plot()
    
    def baseline_als(self, y, lam=1e5, p=0.01, niter=10):
        """Asymmetric Least Squares baseline correction using centralized implementation."""
        return self._get_baseline_fitter().baseline_als(y, lam, p, niter)
    
    # Peak detection methods (fallback controls only)
    def update_height_label(self):
        """Update height slider label (fallback controls only)."""
        if hasattr(self, 'height_slider') and hasattr(self, 'height_label'):
            value = self.height_slider.value()
            self.height_label.setText(f"{value}%")
    
    def update_distance_label(self):
        """Update distance slider label (fallback controls only)."""
        if hasattr(self, 'distance_slider') and hasattr(self, 'distance_label'):
            value = self.distance_slider.value()
            self.distance_label.setText(str(value))
    
    def update_prominence_label(self):
        """Update prominence slider label (fallback controls only)."""
        if hasattr(self, 'prominence_slider') and hasattr(self, 'prominence_label'):
            value = self.prominence_slider.value()
            self.prominence_label.setText(f"{value}%")
    
    def apply_peak_preset(self, height, distance, prominence):
        """Apply preset peak detection parameters."""
        # Set slider values
        if hasattr(self, 'height_slider'):
            self.height_slider.setValue(height)
        if hasattr(self, 'distance_slider'):
            self.distance_slider.setValue(distance)
        if hasattr(self, 'prominence_slider'):
            self.prominence_slider.setValue(prominence)
        
        # Update labels
        if hasattr(self, 'height_label'):
            self.height_label.setText(f"{height}%")
        if hasattr(self, 'distance_label'):
            self.distance_label.setText(str(distance))
        if hasattr(self, 'prominence_label'):
            self.prominence_label.setText(f"{prominence}%")
        
        # Trigger peak detection update
        self._trigger_peak_update()
        
        # Provide feedback in status bar
        self.update_status_bar()
        print(f"Applied preset: Height={height}%, Distance={distance}, Prominence={prominence}%")
    
    def detect_peaks(self):
        """Detect peaks in the spectrum (legacy method - now calls debounced detection)."""
        self._trigger_peak_update()
    
    def clear_peaks(self):
        """Clear all detected peaks (both automatic and manual)."""
        self.peaks = np.array([])  # Initialize as empty numpy array
        self.manual_peaks = np.array([])  # Clear manual peaks too
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        self.update_peak_count_display()
        self.update_peak_list()  # Update the peak list widget
        self.update_manual_peaks_radio_state()  # Update batch radio buttons state
        self.update_plot()
    
    def update_peak_count_display(self):
        """Update the peak count display with auto and manual counts."""
        auto_count = len(self.peaks) if hasattr(self, 'peaks') and self.peaks is not None else 0
        manual_count = len(self.manual_peaks) if hasattr(self, 'manual_peaks') and self.manual_peaks is not None else 0
        total_count = auto_count + manual_count
        
        self.peak_count_label.setText(f"Auto peaks: {auto_count} | Manual peaks: {manual_count} | Total: {total_count}")

    # Interactive peak selection methods
    def toggle_interactive_mode(self):
        """Toggle interactive peak selection mode."""
        self.interactive_mode = self.interactive_btn.isChecked()
        
        if self.interactive_mode:
            # Enable interactive mode
            self.interactive_btn.setText("🖱️ Disable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: ON - Click on spectrum to select peaks")
            self.interactive_status_label.setStyleSheet("color: #4CAF50; font-size: 10px; font-weight: bold;")
            
            # Connect mouse click event
            self.click_connection = self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
            
        else:
            # Disable interactive mode
            self.interactive_btn.setText("🖱️ Enable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: OFF")
            self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Disconnect mouse click event
            if self.click_connection is not None:
                self.canvas.mpl_disconnect(self.click_connection)
                self.click_connection = None
        
        self.update_plot()

    def on_canvas_click(self, event):
        """Handle mouse clicks on the canvas for peak selection."""
        # Only respond to clicks on the main spectrum plot
        if event.inaxes != self.ax_main:
            return
        
        # Only respond to left mouse button
        if event.button != 1:
            return
        
        if not self.interactive_mode:
            return
        
        # Get click coordinates
        click_x = event.xdata
        click_y = event.ydata
        
        if click_x is None or click_y is None:
            return
        
        try:
            # Find the closest data point to the click
            click_wavenumber = click_x
            
            # Find the closest wavenumber index
            closest_idx = np.argmin(np.abs(self.wavenumbers - click_wavenumber))
            
            # Validate the index
            if closest_idx < 0 or closest_idx >= len(self.wavenumbers):
                return
            
            # Check if we're clicking near an existing peak to remove it
            removal_threshold = 20  # wavenumber units
            
            # Check automatic peaks
            removed_auto = False
            if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                valid_auto_peaks = self.validate_peak_indices(self.peaks)
                for i, peak_idx in enumerate(valid_auto_peaks):
                    peak_wavenumber = self.wavenumbers[peak_idx]
                    if abs(peak_wavenumber - click_wavenumber) < removal_threshold:
                        # Remove this automatic peak
                        peak_list = list(self.peaks)
                        if peak_idx in peak_list:
                            peak_list.remove(peak_idx)
                            self.peaks = np.array(peak_list)
                            removed_auto = True
                            break
            
            # Check manual peaks
            removed_manual = False
            if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
                valid_manual_peaks = self.validate_peak_indices(self.manual_peaks)
                for i, peak_idx in enumerate(valid_manual_peaks):
                    peak_wavenumber = self.wavenumbers[peak_idx]
                    if abs(peak_wavenumber - click_wavenumber) < removal_threshold:
                        # Remove this manual peak
                        peak_list = list(self.manual_peaks)
                        if peak_idx in peak_list:
                            peak_list.remove(peak_idx)
                            self.manual_peaks = np.array(peak_list)
                            removed_manual = True
                            break
            
            # If we didn't remove any peak, add a new manual peak
            if not removed_auto and not removed_manual:
                # Add new manual peak
                if not hasattr(self, 'manual_peaks') or self.manual_peaks is None:
                    self.manual_peaks = np.array([closest_idx])
                else:
                    # Check if this peak is already in manual peaks
                    should_add_peak = True
                    if len(self.manual_peaks) > 0:
                        should_add_peak = closest_idx not in self.manual_peaks.tolist()
                    
                    if should_add_peak:
                        self.manual_peaks = np.append(self.manual_peaks, closest_idx)
            
            # Update display
            self.update_peak_count_display()
            # Use efficient updates for interactive changes
            self._update_peak_markers()
            self._update_manual_peak_markers()
            # Update batch radio buttons state
            self.update_manual_peaks_radio_state()
            
        except Exception as e:
            print(f"Error in interactive peak selection: {e}")

    def clear_manual_peaks(self):
        """Clear only manually selected peaks."""
        self.manual_peaks = np.array([])
        self.update_peak_count_display()
        self._update_manual_peak_markers()
        # Update batch radio buttons state
        self.update_manual_peaks_radio_state()

    def combine_peaks(self):
        """Combine automatic and manual peaks into the main peaks list."""
        # Combine automatic and manual peaks
        all_peaks = []
        
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            all_peaks.extend(self.peaks.tolist())
        
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            all_peaks.extend(self.manual_peaks.tolist())
        
        # Remove duplicates and sort
        if len(all_peaks) > 0:
            unique_peaks = sorted(list(set(all_peaks)))
            self.peaks = np.array(unique_peaks)
        else:
            self.peaks = np.array([])
        
        # Clear manual peaks since they're now in the main list
        self.manual_peaks = np.array([])
        
        # Update display
        self.update_peak_count_display()
        self._update_peak_markers()
        self._update_manual_peak_markers()
        
        # Show confirmation
        QMessageBox.information(self, "Peaks Combined", 
                              f"Combined peaks into main list.\nTotal peaks: {len(self.peaks)}")
        
        # Update batch radio buttons state
        self.update_manual_peaks_radio_state()

    # Peak fitting methods
    def on_model_changed(self):
        """Handle model selection change (fallback controls)."""
        if hasattr(self, 'model_combo'):
            self.current_model = self.model_combo.currentText()
            if self.fit_result is not None:
                self.update_plot()
    
    def on_centralized_model_changed(self, model_name):
        """Handle model selection change from centralized controls."""
        self.current_model = model_name
        if self.fit_result is not None:
            self.update_plot()
    
    def on_display_changed(self):
        """Handle display option changes."""
        self.show_individual_peaks = self.show_peaks_check.isChecked()
        self.show_components = self.show_components_check.isChecked()
        self.update_plot()
    
    # REFACTORED: Use centralized peak fitting functions instead of duplicating them
    def gaussian(self, x, amp, cen, wid):
        """Gaussian peak function using centralized implementation."""
        return self._get_baseline_fitter().gaussian(x, amp, cen, wid)
    
    def lorentzian(self, x, amp, cen, wid):
        """Lorentzian peak function using centralized implementation."""
        return self._get_baseline_fitter().lorentzian(x, amp, cen, wid)
    
    def pseudo_voigt(self, x, amp, cen, wid, eta=0.5):
        """Pseudo-Voigt peak function using centralized implementation."""
        return self._get_baseline_fitter().pseudo_voigt(x, amp, cen, wid, eta)
    
    def _get_baseline_fitter(self):
        """Get baseline correction fitter instance."""
        if not hasattr(self, '_baseline_fitter'):
            self._baseline_fitter = EnhancedPeakFitter()
        return self._baseline_fitter
    
    def multi_peak_model(self, x, *params):
        """Multi-peak model function."""
        
        # Use fitted peaks indices if available (during fitting), otherwise use original peaks
        peaks_to_use = None
        if hasattr(self, 'fitted_peaks_indices') and self.fitted_peaks_indices is not None and len(self.fitted_peaks_indices) > 0:
            peaks_to_use = self.fitted_peaks_indices
        elif hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            peaks_to_use = self.peaks
            
        if peaks_to_use is None or len(peaks_to_use) == 0:
            return np.zeros_like(x)
        
        # Validate peaks before using them
        validated_peaks = self.validate_peak_indices(peaks_to_use)
        
        if len(validated_peaks) == 0:
            return np.zeros_like(x)
        
        # Ensure we have the right number of parameters for the peaks
        n_peaks = len(validated_peaks)
        expected_params = n_peaks * 3
        if len(params) < expected_params:
            # Pad with default values if not enough parameters
            params = list(params) + [100, 1500, 50] * (expected_params - len(params) // 3)
        
        model = np.zeros_like(x)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(params):
                amp, cen, wid = params[start_idx:start_idx+3]
                
                # Ensure positive width
                wid = max(abs(wid), 1.0)
                
                if self.current_model == "Gaussian":
                    component = self.gaussian(x, amp, cen, wid)
                elif self.current_model == "Lorentzian":
                    component = self.lorentzian(x, amp, cen, wid)
                elif self.current_model == "Pseudo-Voigt":
                    component = self.pseudo_voigt(x, amp, cen, wid)
                else:
                    component = self.gaussian(x, amp, cen, wid)  # Default
                
                model += component
        
        return model
    
    def fit_peaks(self):
        """Fit peaks to the spectrum (uses combined automatic and manual peaks)."""
        try:
            # Get all peaks (automatic + manual) for fitting
            all_peaks = self.get_all_peaks_for_fitting()
            
            if len(all_peaks) == 0:
                QMessageBox.warning(self, "No Peaks", 
                                  "Detect peaks or select peaks manually first before fitting.\\n\\n"
                                  "Use 'Combine Auto + Manual' button to merge peak lists if needed.")
                return
            
            # Store peaks for the model function (don't overwrite original peak detection results)
            self.fitted_peaks_indices = np.array(all_peaks)
            
            # Create initial parameter guesses
            initial_params = []
            bounds_lower = []
            bounds_upper = []
            
            for i, peak_idx in enumerate(all_peaks):
                if 0 <= peak_idx < len(self.wavenumbers):
                    # Amplitude: Use actual intensity at peak
                    amp = self.processed_intensities[peak_idx]
                    
                    # Center: Use wavenumber at peak
                    cen = self.wavenumbers[peak_idx]
                    
                    # Width: Estimate from local curvature
                    wid = self.estimate_peak_width(peak_idx)
                    
                    initial_params.extend([amp, cen, wid])
                    
                    # Set reasonable bounds
                    bounds_lower.extend([amp * 0.1, cen - wid * 2, wid * 0.3])
                    bounds_upper.extend([amp * 10, cen + wid * 2, wid * 3])
            
            if not initial_params:
                QMessageBox.warning(self, "Invalid Peaks", "No valid peak parameters could be estimated.")
                return
            
            # Prepare bounds
            bounds = (bounds_lower, bounds_upper)
            
            # Get current model from centralized or fallback controls
            current_params = self.get_peak_parameters()
            self.current_model = current_params.get('model', self.current_model)
            
            # Define fitting strategies with different approaches
            strategies = [
                # Strategy 1: Standard fitting
                {
                    'p0': initial_params,
                    'bounds': bounds,
                    'method': 'trf',
                    'max_nfev': 2000
                },
                # Strategy 2: Relaxed bounds
                {
                    'p0': initial_params,
                    'bounds': ([b * 0.5 for b in bounds_lower], [b * 1.5 for b in bounds_upper]),
                    'method': 'lm',
                    'max_nfev': 1000
                },
                # Strategy 3: No bounds (if others fail)
                {
                    'p0': initial_params,
                    'method': 'lm',
                    'max_nfev': 3000
                }
            ]
            
            fit_success = False
            best_params = None
            best_r_squared = -1
            
            for i, strategy in enumerate(strategies):
                try:
                    # Apply strategy
                    if 'bounds' in strategy:
                        popt, pcov = curve_fit(
                            self.multi_peak_model, 
                            self.wavenumbers, 
                            self.processed_intensities,
                            p0=strategy['p0'],
                            bounds=strategy['bounds'],
                            method=strategy['method'],
                            max_nfev=strategy['max_nfev']
                        )
                    else:
                        popt, pcov = curve_fit(
                            self.multi_peak_model, 
                            self.wavenumbers, 
                            self.processed_intensities,
                            p0=strategy['p0'],
                            method=strategy['method'],
                            max_nfev=strategy['max_nfev']
                        )
                    
                    # Calculate R-squared
                    fitted_y = self.multi_peak_model(self.wavenumbers, *popt)
                    ss_res = np.sum((self.processed_intensities - fitted_y) ** 2)
                    ss_tot = np.sum((self.processed_intensities - np.mean(self.processed_intensities)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Check if this is the best fit so far
                    if r_squared > best_r_squared:
                        best_params = popt
                        best_r_squared = r_squared
                        fit_success = True
                    
                    break  # If we reach here, fitting was successful
                    
                except Exception as e:
                    continue  # Try next strategy
            
            if fit_success:
                # Store the best results
                self.fit_params = best_params
                self.fit_result = True
                
                # Calculate residuals properly
                fitted_curve = self.multi_peak_model(self.wavenumbers, *best_params)
                self.residuals = self.processed_intensities - fitted_curve
                
                # Update displays
                self.update_plot()
                
                # Use the more accurate total R² calculation for display
                total_r2 = self.calculate_total_r2()
                self.display_fit_results(total_r2, all_peaks)
                self.update_results_table()
                
                QMessageBox.information(self, "Success", 
                                      f"Peak fitting completed successfully!\\n"
                                      f"Total R² = {total_r2:.4f}\\n"
                                      f"Fitted {len(all_peaks)} peaks")
            else:
                QMessageBox.warning(self, "Fitting Failed", 
                                  "Peak fitting failed with all strategies.\\n\\n"
                                  "Try:\\n"
                                  "• Adjusting peak detection parameters\\n"
                                  "• Reducing the number of peaks\\n"
                                  "• Improving background subtraction\\n"
                                  "• Using a different fitting model")
                
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Peak fitting failed: {str(e)}")
        finally:
            # Always clean up temporary fitting peaks array
            if hasattr(self, 'fitted_peaks_indices'):
                delattr(self, 'fitted_peaks_indices')
    
    def estimate_peak_width(self, peak_idx):
        """Estimate peak width based on local data around peak."""
        try:
            # Look at points around the peak
            window = 20  # points on each side
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(self.processed_intensities), peak_idx + window + 1)
            
            local_intensities = self.processed_intensities[start_idx:end_idx]
            local_wavenumbers = self.wavenumbers[start_idx:end_idx]
            
            peak_intensity = self.processed_intensities[peak_idx]
            half_max = peak_intensity / 2
            
            # Find FWHM (Full Width at Half Maximum)
            above_half = local_intensities > half_max
            if np.any(above_half):
                indices = np.where(above_half)[0]
                if len(indices) > 1:
                    fwhm_indices = [indices[0], indices[-1]]
                    fwhm = abs(local_wavenumbers[fwhm_indices[1]] - local_wavenumbers[fwhm_indices[0]])
                    # Convert FWHM to Gaussian sigma (width parameter)
                    width = fwhm / (2 * np.sqrt(2 * np.log(2)))
                    return max(width, 5.0)  # minimum width
            
            # Fallback: estimate based on wavenumber spacing
            wavenumber_spacing = np.mean(np.diff(self.wavenumbers))
            return max(10 * wavenumber_spacing, 5.0)
            
        except Exception:
            # Default fallback
            return 10.0

    def get_all_peaks_for_fitting(self):
        """Get all peaks (automatic + manual) for fitting operations."""
        all_peaks = []
        
        # Get automatic peaks with validation
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            validated_auto = self.validate_peak_indices(self.peaks)
            all_peaks.extend(validated_auto.tolist())
        
        # Get manual peaks with validation
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            validated_manual = self.validate_peak_indices(self.manual_peaks)
            all_peaks.extend(validated_manual.tolist())
        
        # Remove duplicates and sort
        if len(all_peaks) > 0:
            unique_peaks = sorted(list(set(all_peaks)))
            return unique_peaks
        else:
            return []

    def display_fit_results(self, r_squared, fitted_peaks=None):
        """Display fitting results with individual and total R² values."""
        if fitted_peaks is None:
            fitted_peaks = self.get_all_peaks_for_fitting()
        
        # Ensure fitted_peaks are validated
        validated_fitted_peaks = []
        if isinstance(fitted_peaks, (list, np.ndarray)) and len(fitted_peaks) > 0:
            validated_fitted_peaks = self.validate_peak_indices(np.array(fitted_peaks))
        
        # Calculate individual R² values
        individual_r2_values = self.calculate_individual_r2_values()
        total_r2 = self.calculate_total_r2()
        
        results = f"Peak Fitting Results\n{'='*30}\n\n"
        results += f"Model: {self.current_model}\n"
        results += f"Number of peaks fitted: {len(validated_fitted_peaks)}\n"
        results += f"Total R² = {total_r2:.4f}\n\n"
        
        # Handle numpy array boolean check properly
        if (self.fit_params is not None and 
            hasattr(self.fit_params, '__len__') and 
            len(self.fit_params) > 0 and 
            len(validated_fitted_peaks) > 0):
            
            results += "Peak Parameters:\n"
            n_peaks = len(validated_fitted_peaks)
            for i in range(n_peaks):
                start_idx = i * 3
                if start_idx + 2 < len(self.fit_params):
                    amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                    
                    # Get individual R² value
                    individual_r2 = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                    
                    # Determine peak type safely
                    peak_type = "Auto"
                    if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                        validated_auto_peaks = self.validate_peak_indices(self.peaks)
                        if len(validated_auto_peaks) > 0 and validated_fitted_peaks[i] not in validated_auto_peaks.tolist():
                            peak_type = "Manual"
                    
                    results += (f"Peak {i+1} ({peak_type}): Center={cen:.1f} cm⁻¹, "
                              f"Amplitude={amp:.1f}, Width={wid:.1f}, R²={individual_r2:.3f}\n")
            
            # Add summary statistics
            if len(individual_r2_values) > 0:
                avg_individual_r2 = np.mean(individual_r2_values)
                results += f"\nAverage Individual R²: {avg_individual_r2:.3f}\n"
        
        self.results_text.setPlainText(results)

    def update_results_table(self):
        """Update the results table with R² values."""
        # Properly handle numpy array boolean check
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0):
            return
        
        fitted_peaks = self.get_all_peaks_for_fitting()
        if len(fitted_peaks) == 0:
            return
        
        # Validate the fitted peaks
        validated_fitted_peaks = self.validate_peak_indices(np.array(fitted_peaks))
        n_peaks = len(validated_fitted_peaks)
        self.results_table.setRowCount(n_peaks)
        
        # Calculate individual R² values
        individual_r2_values = self.calculate_individual_r2_values()
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                
                # Get individual R² value
                individual_r2 = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                
                # Determine peak type safely
                peak_type = "Auto"
                if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                    validated_auto_peaks = self.validate_peak_indices(self.peaks)
                    if len(validated_auto_peaks) > 0 and validated_fitted_peaks[i] not in validated_auto_peaks.tolist():
                        peak_type = "Manual"
                
                self.results_table.setItem(i, 0, QTableWidgetItem(f"Peak {i+1} ({peak_type})"))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{cen:.1f}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{amp:.1f}"))
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{wid:.1f}"))
                self.results_table.setItem(i, 4, QTableWidgetItem(f"{individual_r2:.3f}"))

         # Fourier Transform Analysis Methods
    def show_frequency_spectrum(self):
        """Show the frequency spectrum of the processed intensities."""
        try:
            # Calculate FFT
            fft_data = np.fft.fft(self.processed_intensities)
            fft_magnitude = np.abs(fft_data)
            fft_phase = np.angle(fft_data)
            
            # Create frequency array (normalized to Nyquist frequency)
            n_points = len(self.processed_intensities)
            frequencies = np.fft.fftfreq(n_points, d=1.0)[:n_points//2]  # Only positive frequencies
            magnitude_spectrum = fft_magnitude[:n_points//2]
            phase_spectrum = fft_phase[:n_points//2]
            
            # Store for other operations
            self.fft_data = fft_data
            self.fft_frequencies = frequencies
            self.fft_magnitude = magnitude_spectrum
            self.fft_phase = phase_spectrum
            
            # Create a new plot window or update residual plot
            self.ax_residual.clear()
            
            # Plot magnitude spectrum
            self.ax_residual.semilogy(frequencies, magnitude_spectrum, 'b-', linewidth=1.5, label='Magnitude')
            self.ax_residual.set_xlabel('Normalized Frequency')
            self.ax_residual.set_ylabel('Magnitude (log scale)')
            self.ax_residual.set_title('Frequency Spectrum (FFT Magnitude)')
            self.ax_residual.grid(True, alpha=0.3)
            self.ax_residual.legend()
            
            # Display results
            dominant_freq_idx = np.argmax(magnitude_spectrum[1:]) + 1  # Skip DC component
            dominant_freq = frequencies[dominant_freq_idx]
            max_magnitude = magnitude_spectrum[dominant_freq_idx]
            
            results = "Frequency Spectrum Analysis:\n"
            results += f"Total data points: {n_points}\n"
            results += f"Frequency resolution: {frequencies[1]:.6f}\n"
            results += f"DC component magnitude: {magnitude_spectrum[0]:.2f}\n"
            results += f"Dominant frequency: {dominant_freq:.4f}\n"
            results += f"Dominant magnitude: {max_magnitude:.2f}\n"
            results += f"Total spectral energy: {np.sum(magnitude_spectrum**2):.2e}\n"
            
            self.results_text.setPlainText(results)
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "FFT Error", f"Frequency spectrum calculation failed: {str(e)}")
    
    def show_power_spectral_density(self):
        """Show the power spectral density."""
        try:
            # Calculate power spectral density
            if not hasattr(self, 'fft_data'):
                self.show_frequency_spectrum()  # Calculate FFT first
            
            # PSD is the square of the magnitude spectrum
            psd = self.fft_magnitude ** 2
            
            # Normalize by frequency resolution and total power
            psd_normalized = psd / (len(self.processed_intensities) * np.sum(psd))
            
            # Plot PSD
            self.ax_residual.clear()
            self.ax_residual.semilogy(self.fft_frequencies, psd_normalized, 'r-', linewidth=1.5, label='Power Spectral Density')
            self.ax_residual.set_xlabel('Normalized Frequency')
            self.ax_residual.set_ylabel('Power Density (log scale)')
            self.ax_residual.set_title('Power Spectral Density')
            self.ax_residual.grid(True, alpha=0.3)
            self.ax_residual.legend()
            
            # Calculate spectral statistics
            total_power = np.sum(psd)
            mean_freq = np.sum(self.fft_frequencies * psd) / total_power
            spectral_centroid = np.sum(self.fft_frequencies * psd) / np.sum(psd)
            
            # Spectral bandwidth (standard deviation)
            spectral_bandwidth = np.sqrt(np.sum(((self.fft_frequencies - spectral_centroid) ** 2) * psd) / np.sum(psd))
            
            results = "Power Spectral Density Analysis:\n"
            results += f"Total power: {total_power:.2e}\n"
            results += f"Spectral centroid: {spectral_centroid:.4f}\n"
            results += f"Spectral bandwidth: {spectral_bandwidth:.4f}\n"
            results += f"Peak PSD frequency: {self.fft_frequencies[np.argmax(psd)]:.4f}\n"
            results += f"Peak PSD value: {np.max(psd):.2e}\n"
            
            self.results_text.setPlainText(results)
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "PSD Error", f"Power spectral density calculation failed: {str(e)}")
    
    # Fourier Filter Label Update Methods
    def update_low_cutoff_label(self):
        """Update low cutoff frequency label."""
        value = self.low_cutoff_slider.value()
        self.low_cutoff_label.setText(f"{value}%")
    
    def update_high_cutoff_label(self):
        """Update high cutoff frequency label."""
        value = self.high_cutoff_slider.value()
        self.high_cutoff_label.setText(f"{value}%")
    
    def update_butterworth_order_label(self):
        """Update Butterworth filter order label."""
        value = self.butterworth_order_slider.value()
        self.butterworth_order_label.setText(str(value))
    
    def on_filter_type_changed(self):
        """Handle filter type change to show/hide Butterworth order control."""
        filter_type = self.filter_type_combo.currentText()
        is_butterworth = "Butterworth" in filter_type
        
        # Show/hide Butterworth order control
        self.butterworth_order_slider.setVisible(is_butterworth)
        self.butterworth_order_label.setVisible(is_butterworth)
    
    def preview_fourier_filter(self):
        """Preview the effect of Fourier filtering."""
        try:
            filtered_spectrum = self._apply_fourier_filter_internal(preview_only=True)
            
            # Update main plot with preview
            if hasattr(self, 'filter_preview_line') and self.filter_preview_line is not None:
                try:
                    self.filter_preview_line.remove()
                except:
                    pass
            
            self.filter_preview_line, = self.ax_main.plot(
                self.wavenumbers, filtered_spectrum, 'orange', 
                linewidth=2, alpha=0.7, linestyle='--', label='Filter Preview'
            )
            self.ax_main.legend()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Filter Preview Error", f"Filter preview failed: {str(e)}")
    
    def apply_fourier_filter(self):
        """Apply the Fourier filter to the spectrum."""
        try:
            filtered_spectrum = self._apply_fourier_filter_internal(preview_only=False)
            
            # Apply to processed intensities
            self.processed_intensities = filtered_spectrum.copy()
            
            # Clear any existing filter preview
            if hasattr(self, 'filter_preview_line') and self.filter_preview_line is not None:
                try:
                    self.filter_preview_line.remove()
                    self.filter_preview_line = None
                except:
                    pass
            
            # Update plot
            self.update_plot()
            
            filter_type = self.filter_type_combo.currentText()
            QMessageBox.information(self, "Filter Applied", f"{filter_type} filter applied successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", f"Filter application failed: {str(e)}")
    
    def _apply_fourier_filter_internal(self, preview_only=False):
        """Internal method to apply Fourier filtering."""
        # Calculate FFT
        fft_data = np.fft.fft(self.processed_intensities)
        n_points = len(fft_data)
        frequencies = np.fft.fftfreq(n_points)
        
        # Get filter parameters
        filter_type = self.filter_type_combo.currentText()
        low_cutoff = self.low_cutoff_slider.value() / 100.0  # Convert to fraction
        high_cutoff = self.high_cutoff_slider.value() / 100.0
        
        # Check if it's a Butterworth filter
        is_butterworth = "Butterworth" in filter_type
        
        if is_butterworth:
            # Get Butterworth filter order
            order = self.butterworth_order_slider.value()
            
            # Create Butterworth filter response
            filter_response = self._create_butterworth_response(frequencies, filter_type, low_cutoff, high_cutoff, order)
            
            # Apply filter to FFT data (multiply by response, not mask)
            filtered_fft = fft_data * filter_response
        else:
            # Original binary mask filters
            filter_mask = np.ones_like(frequencies, dtype=bool)
            
            # Apply filter based on type
            freq_magnitude = np.abs(frequencies)
            
            if filter_type == "Low-pass":
                filter_mask = freq_magnitude <= high_cutoff
            elif filter_type == "High-pass":
                filter_mask = freq_magnitude >= low_cutoff
            elif filter_type == "Band-pass":
                filter_mask = (freq_magnitude >= low_cutoff) & (freq_magnitude <= high_cutoff)
            elif filter_type == "Band-stop":
                filter_mask = (freq_magnitude < low_cutoff) | (freq_magnitude > high_cutoff)
            
            # Apply filter to FFT data
            filtered_fft = fft_data.copy()
            filtered_fft[~filter_mask] = 0
        
        # Convert back to time domain
        filtered_spectrum = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_spectrum
    
    def _create_butterworth_response(self, frequencies, filter_type, low_cutoff, high_cutoff, order):
        """Create Butterworth filter frequency response."""
        freq_magnitude = np.abs(frequencies)
        
        # Initialize response array
        response = np.ones_like(frequencies, dtype=float)
        
        # Avoid division by zero
        epsilon = 1e-10
        freq_magnitude = np.maximum(freq_magnitude, epsilon)
        
        if filter_type == "Butterworth Low-pass":
            # |H(ω)|² = 1 / (1 + (ω/ωc)^(2n))
            response = 1.0 / (1.0 + (freq_magnitude / max(high_cutoff, epsilon)) ** (2 * order))
            
        elif filter_type == "Butterworth High-pass":
            # |H(ω)|² = (ω/ωc)^(2n) / (1 + (ω/ωc)^(2n))
            ratio = freq_magnitude / max(low_cutoff, epsilon)
            response = (ratio ** (2 * order)) / (1.0 + ratio ** (2 * order))
            
        elif filter_type == "Butterworth Band-pass":
            # Combination of high-pass and low-pass
            # High-pass component
            ratio_low = freq_magnitude / max(low_cutoff, epsilon)
            high_pass_response = (ratio_low ** (2 * order)) / (1.0 + ratio_low ** (2 * order))
            
            # Low-pass component
            ratio_high = freq_magnitude / max(high_cutoff, epsilon)
            low_pass_response = 1.0 / (1.0 + ratio_high ** (2 * order))
            
            # Combine
            response = high_pass_response * low_pass_response
            
        elif filter_type == "Butterworth Band-stop":
            # Inverse of band-pass: 1 - band_pass_response
            # High-pass component
            ratio_low = freq_magnitude / max(low_cutoff, epsilon)
            high_pass_response = (ratio_low ** (2 * order)) / (1.0 + ratio_low ** (2 * order))
            
            # Low-pass component  
            ratio_high = freq_magnitude / max(high_cutoff, epsilon)
            low_pass_response = 1.0 / (1.0 + ratio_high ** (2 * order))
            
            # Band-pass response
            band_pass_response = high_pass_response * low_pass_response
            
            # Band-stop is inverse
            response = 1.0 - band_pass_response
        
        # Take square root to get magnitude response (since we calculated |H(ω)|²)
        response = np.sqrt(np.maximum(response, 0))
        
        return response
    
    def apply_fourier_smoothing(self):
        """Apply Fourier-based smoothing to reduce noise."""
        try:
            # Calculate FFT
            fft_data = np.fft.fft(self.processed_intensities)
            n_points = len(fft_data)
            frequencies = np.fft.fftfreq(n_points)
            
            # Create Gaussian smoothing filter
            sigma = 0.1  # Smoothing parameter
            smoothing_filter = np.exp(-0.5 * (frequencies / sigma) ** 2)
            
            # Apply smoothing in frequency domain
            smoothed_fft = fft_data * smoothing_filter
            
            # Convert back to time domain
            smoothed_spectrum = np.real(np.fft.ifft(smoothed_fft))
            
            # Apply to processed intensities
            self.processed_intensities = smoothed_spectrum.copy()
            
            # Update plot
            self.update_plot()
            
            QMessageBox.information(self, "Smoothing Applied", "Fourier smoothing applied successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Smoothing Error", f"Fourier smoothing failed: {str(e)}")
    
    def apply_richardson_lucy(self):
        """Apply Richardson-Lucy deconvolution for resolution enhancement."""
        try:
            # Enhanced Richardson-Lucy implementation
            iterations = 20
            deconvolved = self.richardson_lucy_deconvolution(self.processed_intensities, iterations)
            
            # Apply to processed intensities
            self.processed_intensities = deconvolved.copy()
            
            # Update plot
            self.update_plot()
            
            QMessageBox.information(self, "Deconvolution Applied", 
                                  f"Richardson-Lucy deconvolution applied with {iterations} iterations!")
            
        except Exception as e:
            QMessageBox.critical(self, "Deconvolution Error", f"Richardson-Lucy deconvolution failed: {str(e)}")
    
    def richardson_lucy_deconvolution(self, data, iterations=10):
        """Apply Richardson-Lucy deconvolution algorithm."""
        # Simple implementation - could be enhanced with proper PSF estimation
        deconvolved = data.copy()
        
        for _ in range(iterations):
            # Estimate point spread function
            psf = self.estimate_psf(len(data))
            
            # Convolve current estimate
            convolved = np.convolve(deconvolved, psf, mode='same')
            
            # Avoid division by zero
            ratio = np.divide(data, convolved + 1e-10)
            
            # Update estimate
            deconvolved *= np.convolve(ratio, psf[::-1], mode='same')
            
            # Ensure non-negativity
            deconvolved = np.maximum(deconvolved, 0)
        
        return deconvolved
    
    def estimate_psf(self, size, sigma=2.0):
        """Estimate point spread function for deconvolution."""
        x = np.arange(size) - size // 2
        psf = np.exp(-0.5 * (x / sigma)**2)
        return psf / np.sum(psf)
    
    def apply_apodization(self):
        """Apply apodization (windowing) to the spectrum."""
        try:
            # Create apodization options dialog
            from PySide6.QtWidgets import QInputDialog
            
            window_types = ["Hann", "Hamming", "Blackman", "Gaussian", "Tukey"]
            window_type, ok = QInputDialog.getItem(
                self, "Select Window Type", "Choose apodization window:", 
                window_types, 0, False
            )
            
            if not ok:
                return
            
            n_points = len(self.processed_intensities)
            
            # Create window function
            if window_type == "Hann":
                window = np.hanning(n_points)
            elif window_type == "Hamming":
                window = np.hamming(n_points)
            elif window_type == "Blackman":
                window = np.blackman(n_points)
            elif window_type == "Gaussian":
                sigma = n_points / 8
                x = np.arange(n_points) - n_points // 2
                window = np.exp(-0.5 * (x / sigma) ** 2)
            elif window_type == "Tukey":
                # Simple Tukey window implementation
                alpha = 0.5
                window = np.ones(n_points)
                n_taper = int(alpha * n_points / 2)
                
                # Left taper
                for i in range(n_taper):
                    window[i] = 0.5 * (1 + np.cos(np.pi * (2 * i / (alpha * n_points) - 1)))
                
                # Right taper
                for i in range(n_points - n_taper, n_points):
                    window[i] = 0.5 * (1 + np.cos(np.pi * (2 * (i - n_points + n_taper) / (alpha * n_points) - 1)))
            
            # Apply window to spectrum
            windowed_spectrum = self.processed_intensities * window
            
            # Normalize to preserve total intensity
            windowed_spectrum *= np.sum(self.processed_intensities) / np.sum(windowed_spectrum)
            
            # Apply to processed intensities
            self.processed_intensities = windowed_spectrum.copy()
            
            # Update plot
            self.update_plot()
            
            QMessageBox.information(self, "Apodization Applied", f"{window_type} window applied successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Apodization Error", f"Apodization failed: {str(e)}")
    
    def export_results(self):
        """Export analysis results."""
        try:
            from PySide6.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "", 
                "CSV files (*.csv);;Text files (*.txt);;All files (*.*)"
            )
            
            if file_path:
                self.export_to_file(file_path)
                
                # Count additional files created
                base_name = file_path.replace('.csv', '').replace('.txt', '')
                additional_files = []
                
                if (self.fit_result is not None and self.fit_params is not None and 
                    hasattr(self, 'peaks') and len(self.get_all_peaks_for_fitting()) > 0):
                    
                    n_peaks = len(self.get_all_peaks_for_fitting())
                    additional_files.append(f"• Peak parameters: {base_name}_peak_parameters.csv")
                    additional_files.append(f"• {n_peaks} individual peak region files")
                
                message = f"Export completed successfully!\n\nMain file: {file_path}\n"
                if additional_files:
                    message += "\nAdditional files created:\n" + "\n".join(additional_files)
                    message += f"\n\nTotal files exported: {1 + len(additional_files)}"
                
                QMessageBox.information(self, "Export Complete", message)
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")
    
    # Background Auto-Preview Methods
    def generate_background_previews(self):
        """Generate multiple background subtraction options with different parameters."""
        try:
            # Clear previous options
            self.clear_background_options()
            
            # Define ALS parameter sets - top 6 most useful combinations
            parameter_sets = [
                ("ALS (Conservative)", "ALS", {"lambda": 1e6, "p": 0.001, "niter": 10}),
                ("ALS (Moderate)", "ALS", {"lambda": 1e5, "p": 0.01, "niter": 10}),
                ("ALS (Aggressive)", "ALS", {"lambda": 1e4, "p": 0.05, "niter": 15}),
                ("ALS (Ultra Smooth)", "ALS", {"lambda": 1e7, "p": 0.002, "niter": 20}),
                ("ALS (Balanced)", "ALS", {"lambda": 5e5, "p": 0.02, "niter": 12}),
                ("ALS (Fast)", "ALS", {"lambda": 1e5, "p": 0.01, "niter": 5}),
            ]
            
            # Generate backgrounds for each parameter set
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                     '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                     '#F1948A', '#73C6B6', '#AED6F1', '#A9DFBF', '#F9E79F']
            
            for i, (description, method, params) in enumerate(parameter_sets):
                try:
                    background = self._calculate_background_with_params(method, params)
                    if background is not None:
                        # Store background data
                        self.background_options.append((background, description, method, params))
                        
                        # Plot preview line
                        color = colors[i % len(colors)]
                        line, = self.ax_main.plot(
                            self.wavenumbers, background, 
                            color=color, linewidth=1.5, alpha=0.7, 
                            linestyle='--', label=description
                        )
                        self.background_option_lines.append(line)
                        
                except Exception as e:
                    print(f"Failed to generate {description}: {str(e)}")
                    continue
            
            # Update dropdown with options
            self.update_background_options_dropdown()
            
            # Update legend and redraw
            self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.canvas.draw()
            
            # Show info message
            QMessageBox.information(self, "Options Generated", 
                                  f"Generated {len(self.background_options)} background options.\n"
                                  f"Select one from the dropdown and preview it by clicking on the option.")
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"Failed to generate background options: {str(e)}")
    
    def _calculate_background_with_params(self, method, params):
        """Calculate background using specified method and parameters."""
        try:
            if method == "ALS":
                lambda_val = params.get("lambda", 1e5)
                p_val = params.get("p", 0.01)
                niter_val = params.get("niter", 10)
                return self.baseline_als(self.original_intensities, lambda_val, p_val, niter_val)
                
            elif method == "Linear":
                start_weight = params.get("start_weight", 1.0)
                end_weight = params.get("end_weight", 1.0)
                start_val = self.original_intensities[0] * start_weight
                end_val = self.original_intensities[-1] * end_weight
                return np.linspace(start_val, end_val, len(self.original_intensities))
                
            elif method == "Polynomial":
                order = params.get("order", 2)
                method_type = params.get("method", "Least Squares")
                x = np.arange(len(self.original_intensities))
                
                if method_type == "Robust":
                    # Robust fitting with iterative reweighting
                    coeffs = np.polyfit(x, self.original_intensities, order)
                    background = np.polyval(coeffs, x)
                    
                    # Apply robust reweighting
                    for _ in range(3):
                        residuals = np.abs(self.original_intensities - background)
                        weights = 1.0 / (1.0 + residuals / np.median(residuals))
                        coeffs = np.polyfit(x, self.original_intensities, order, w=weights)
                        background = np.polyval(coeffs, x)
                    
                    return background
                else:
                    coeffs = np.polyfit(x, self.original_intensities, order)
                    return np.polyval(coeffs, x)
                    
            elif method == "Moving Average":
                window_percent = params.get("window_percent", 10)
                window_type = params.get("window_type", "Gaussian")
                
                window_size = max(int(len(self.original_intensities) * window_percent / 100.0), 3)
                
                if window_type == "Gaussian":
                    from scipy import ndimage
                    sigma = window_size / 4.0
                    return ndimage.gaussian_filter1d(self.original_intensities, sigma=sigma)
                elif window_type == "Uniform":
                    from scipy import ndimage
                    return ndimage.uniform_filter1d(self.original_intensities, size=window_size)
                else:
                    # Default to Gaussian
                    from scipy import ndimage
                    sigma = window_size / 4.0
                    return ndimage.gaussian_filter1d(self.original_intensities, sigma=sigma)
                     
            elif method == "Spline":
                n_knots = params.get("n_knots", 20)
                smoothing = params.get("smoothing", 500)
                return self._calculate_spline_background(n_knots, smoothing)
            
            return None
            
        except Exception as e:
            print(f"Background calculation error for {method}: {str(e)}")
            return None
    
    def _calculate_spline_background_for_subtraction(self, n_knots, smoothing, degree):
        """Calculate spline-based background that fits below the peaks for background subtraction."""
        try:
            from scipy.interpolate import UnivariateSpline
            
            # Create x values (indices or wavenumbers)
            x = np.arange(len(self.original_intensities))
            y = self.original_intensities
            
            # For proper background subtraction, we need to fit the baseline, not the peaks
            # Method: Use minimum filtering and iterative approach to fit below the data
            
            # Step 1: Apply a minimum filter to identify baseline regions
            from scipy import ndimage
            window_size = max(len(y) // 20, 5)  # Adaptive window size
            y_min_filtered = ndimage.minimum_filter1d(y, size=window_size)
            
            # Step 2: Create initial baseline estimate using the minimum filtered data
            if n_knots <= 2:
                n_knots = 3  # Minimum for spline
            
            # For background subtraction, use higher smoothing to avoid fitting peaks
            background_smoothing = max(smoothing, len(y) / 10)  # Ensure minimum smoothing
            
            try:
                # Fit spline to minimum filtered data for initial background estimate
                spline = UnivariateSpline(x, y_min_filtered, s=background_smoothing, k=min(degree, 3))
                initial_background = spline(x)
                
                # Step 3: Iterative refinement - only use points below or near the background
                current_background = initial_background.copy()
                
                for iteration in range(3):  # Limited iterations
                    # Identify points that are likely background (below or close to current estimate)
                    threshold = np.percentile(y - current_background, 20)  # Use 20th percentile
                    mask = (y - current_background) <= threshold
                    
                    if np.sum(mask) < n_knots:  # Need enough points
                        break
                    
                    # Fit spline only to identified background points
                    spline = UnivariateSpline(x[mask], y[mask], s=background_smoothing, k=min(degree, 3))
                    current_background = spline(x)
                    
                    # Ensure background doesn't go above data unrealistically
                    current_background = np.minimum(current_background, y)
                
                # Final constraint: background should be below the data
                background = np.minimum(current_background, y)
                
                return background
                
            except Exception:
                # Fallback: use simple percentile-based baseline
                from scipy.signal import savgol_filter
                
                # Use Savitzky-Golay filter on minimum filtered data
                window_length = min(len(y) // 5, 51)
                if window_length % 2 == 0:
                    window_length += 1  # Must be odd
                
                background = savgol_filter(y_min_filtered, window_length, polyorder=min(degree, 3))
                return np.minimum(background, y)
                
        except ImportError:
            # If scipy is not available, fallback to simple polynomial on minimum values
            print("scipy not available, using polynomial fallback for spline background")
            
            # Simple approach: fit polynomial to lower envelope
            window_size = max(len(y) // 10, 3)
            y_smooth = np.array([np.min(y[max(0, i-window_size):min(len(y), i+window_size+1)]) 
                               for i in range(len(y))])
            
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y_smooth, min(degree, 3))
            background = np.polyval(coeffs, x)
            
            return np.minimum(background, y)
            
        except Exception as e:
            print(f"Spline background calculation error: {str(e)}")
            # Final fallback: simple linear baseline
            background = np.linspace(y[0], y[-1], len(y))
            return np.minimum(background, y)
    
    def _calculate_spline_background(self, n_knots, smoothing):
        """Legacy method - calculate spline-based background using UnivariateSpline."""
        # This method kept for compatibility with auto-preview (though spline removed from auto-preview)
        return self._calculate_spline_background_for_subtraction(n_knots, smoothing, 3)
    
    def _calculate_linear_background(self, start_weight, end_weight):
        """Calculate linear background that fits the baseline, not the data."""
        try:
            from scipy import ndimage
            
            y = self.original_intensities
            
            # Method 1: Use minimum filtering to identify baseline regions
            window_size = max(len(y) // 15, 5)  # Adaptive window size for minimum filtering
            y_min_filtered = ndimage.minimum_filter1d(y, size=window_size)
            
            # Method 2: Identify baseline points using percentile approach
            # Take points in the lower percentile as likely baseline
            percentile_threshold = 30  # Use bottom 30% as baseline candidates
            threshold = np.percentile(y, percentile_threshold)
            baseline_mask = y <= threshold
            
            # Combine minimum filtered data with baseline mask
            baseline_indices = np.where(baseline_mask)[0]
            
            if len(baseline_indices) < 2:  # Need at least 2 points for linear fit
                # Fallback: use endpoint-weighted linear fit to minimum filtered data
                start_val = y_min_filtered[0] * start_weight
                end_val = y_min_filtered[-1] * end_weight
            else:
                # Fit line to identified baseline points
                baseline_y = y[baseline_indices]
                
                # Apply weights to endpoints
                if len(baseline_indices) > 0:
                    # Find closest baseline points to start and end
                    start_idx = baseline_indices[0]
                    end_idx = baseline_indices[-1]
                    
                    start_val = y[start_idx] * start_weight
                    end_val = y[end_idx] * end_weight
                else:
                    start_val = y[0] * start_weight
                    end_val = y[-1] * end_weight
            
            # Create linear background
            background = np.linspace(start_val, end_val, len(y))
            
            # Ensure background doesn't exceed data unrealistically
            background = np.minimum(background, y)
            
            return background
            
        except ImportError:
            # Fallback without scipy
            y = self.original_intensities
            
            # Simple approach: weight the endpoints and create linear baseline
            start_val = y[0] * start_weight
            end_val = y[-1] * end_weight
            background = np.linspace(start_val, end_val, len(y))
            
            return np.minimum(background, y)
        
        except Exception as e:
            print(f"Linear background calculation error: {str(e)}")
            # Final fallback
            y = self.original_intensities
            background = np.linspace(y[0], y[-1], len(y))
            return np.minimum(background, y)
    
    def _calculate_polynomial_background(self, poly_order, poly_method):
        """Calculate polynomial background that fits the baseline, not the data."""
        try:
            from scipy import ndimage
            
            y = self.original_intensities
            x = np.arange(len(y))
            
            # Step 1: Use minimum filtering to identify baseline regions
            window_size = max(len(y) // 20, 5)
            y_min_filtered = ndimage.minimum_filter1d(y, size=window_size)
            
            # Step 2: Iterative baseline fitting approach
            current_background = y_min_filtered.copy()
            
            for iteration in range(3):  # Limited iterations
                # Identify points likely to be baseline (below current estimate)
                threshold = np.percentile(y - current_background, 25)  # Use 25th percentile
                mask = (y - current_background) <= threshold
                
                if np.sum(mask) < poly_order + 2:  # Need enough points for polynomial
                    break
                
                # Fit polynomial to identified baseline points
                if poly_method == "Robust":
                    # Robust polynomial fitting with iterative reweighting
                    try:
                        coeffs = np.polyfit(x[mask], y[mask], poly_order)
                        poly_fit = np.polyval(coeffs, x)
                        
                        # Apply robust reweighting for 2 iterations
                        for _ in range(2):
                            residuals = np.abs(y[mask] - np.polyval(coeffs, x[mask]))
                            weights = 1.0 / (1.0 + residuals / (np.median(residuals) + 1e-10))
                            coeffs = np.polyfit(x[mask], y[mask], poly_order, w=weights)
                        
                        current_background = np.polyval(coeffs, x)
                    except:
                        # Fallback to regular fitting
                        coeffs = np.polyfit(x[mask], y[mask], poly_order)
                        current_background = np.polyval(coeffs, x)
                else:
                    # Regular least squares fitting to baseline points
                    coeffs = np.polyfit(x[mask], y[mask], poly_order)
                    current_background = np.polyval(coeffs, x)
                
                # Ensure background doesn't go above data
                current_background = np.minimum(current_background, y)
            
            return current_background
            
        except ImportError:
            # Fallback without scipy
            y = self.original_intensities
            x = np.arange(len(y))
            
            # Simple approach: use lower envelope points
            window_size = max(len(y) // 10, 3)
            y_envelope = np.array([np.min(y[max(0, i-window_size):min(len(y), i+window_size+1)]) 
                                 for i in range(len(y))])
            
            # Fit polynomial to envelope
            coeffs = np.polyfit(x, y_envelope, min(poly_order, len(y)-1))
            background = np.polyval(coeffs, x)
            
            return np.minimum(background, y)
            
        except Exception as e:
            print(f"Polynomial background calculation error: {str(e)}")
            # Final fallback
            y = self.original_intensities
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, min(2, len(y)-1))  # Simple quadratic fallback
            background = np.polyval(coeffs, x)
            return np.minimum(background, y)
    
    def _calculate_moving_average_background(self, window_percent, window_type):
        """Calculate moving average background that fits the baseline, not the data."""
        try:
            from scipy import ndimage
            
            y = self.original_intensities
            
            # Calculate window size as percentage of spectrum length
            window_size = max(int(len(y) * window_percent / 100.0), 3)
            
            # Step 1: Apply minimum filtering to get baseline candidate
            min_window = max(window_size // 2, 3)
            y_min_filtered = ndimage.minimum_filter1d(y, size=min_window)
            
            # Step 2: Apply the specified moving average filter to the minimum filtered data
            if window_type == "Uniform":
                background = ndimage.uniform_filter1d(y_min_filtered, size=window_size)
            elif window_type == "Gaussian":
                sigma = window_size / 4.0  # Standard deviation
                background = ndimage.gaussian_filter1d(y_min_filtered, sigma=sigma)
            elif window_type in ["Hann", "Hamming"]:
                # Apply windowed convolution to minimum filtered data
                if window_type == "Hann":
                    window = np.hanning(window_size)
                else:  # Hamming
                    window = np.hamming(window_size)
                
                window = window / np.sum(window)  # Normalize
                background = np.convolve(y_min_filtered, window, mode='same')
            else:
                # Default to Gaussian
                sigma = window_size / 4.0
                background = ndimage.gaussian_filter1d(y_min_filtered, sigma=sigma)
            
            # Step 3: Additional constraint - ensure it stays below original data
            background = np.minimum(background, y)
            
            # Step 4: Optional second pass for better baseline fitting
            # Create mask for points close to the current background estimate
            tolerance = np.std(y - background) * 0.5
            baseline_mask = (y - background) <= tolerance
            
            if np.sum(baseline_mask) > window_size:
                # Apply the filter again, but only to baseline regions
                baseline_points = y.copy()
                baseline_points[~baseline_mask] = background[~baseline_mask]  # Replace peaks with current estimate
                
                if window_type == "Uniform":
                    refined_background = ndimage.uniform_filter1d(baseline_points, size=window_size)
                elif window_type == "Gaussian":
                    sigma = window_size / 4.0
                    refined_background = ndimage.gaussian_filter1d(baseline_points, sigma=sigma)
                else:
                    refined_background = background  # Keep original if windowed
                
                background = np.minimum(refined_background, y)
            
            return background
            
        except ImportError:
            # Fallback without scipy
            y = self.original_intensities
            window_size = max(int(len(y) * window_percent / 100.0), 3)
            
            # Simple moving minimum approach
            background = np.array([np.min(y[max(0, i-window_size//2):min(len(y), i+window_size//2+1)]) 
                                 for i in range(len(y))])
            
            # Simple smoothing
            for _ in range(2):  # 2 passes of smoothing
                smoothed = background.copy()
                for i in range(1, len(background)-1):
                    smoothed[i] = (background[i-1] + background[i] + background[i+1]) / 3
                background = smoothed
            
            return np.minimum(background, y)
            
        except Exception as e:
            print(f"Moving average background calculation error: {str(e)}")
            # Final fallback
            y = self.original_intensities
            return np.full_like(y, np.min(y))
    
    def _calculate_moving_average_background_for_batch(self, intensities, window_percent, window_type):
        """Calculate moving average background for batch processing."""
        try:
            from scipy import ndimage
            
            # Calculate window size as percentage of spectrum length
            window_size = max(int(len(intensities) * window_percent / 100.0), 3)
            
            # Step 1: Apply minimum filtering to get baseline candidate
            min_window = max(window_size // 2, 3)
            y_min_filtered = ndimage.minimum_filter1d(intensities, size=min_window)
            
            # Step 2: Apply the specified moving average filter to the minimum filtered data
            if window_type == "Uniform":
                background = ndimage.uniform_filter1d(y_min_filtered, size=window_size)
            elif window_type == "Gaussian":
                sigma = window_size / 4.0  # Standard deviation
                background = ndimage.gaussian_filter1d(y_min_filtered, sigma=sigma)
            elif window_type in ["Hann", "Hamming"]:
                # Apply windowed convolution to minimum filtered data
                if window_type == "Hann":
                    window = np.hanning(window_size)
                else:  # Hamming
                    window = np.hamming(window_size)
                
                window = window / np.sum(window)  # Normalize
                background = np.convolve(y_min_filtered, window, mode='same')
            else:
                # Default to Gaussian
                sigma = window_size / 4.0
                background = ndimage.gaussian_filter1d(y_min_filtered, sigma=sigma)
            
            # Step 3: Additional constraint - ensure it stays below original data
            background = np.minimum(background, intensities)
            
            # Step 4: Optional second pass for better baseline fitting
            tolerance = np.std(intensities - background) * 0.5
            baseline_mask = (intensities - background) <= tolerance
            
            if np.sum(baseline_mask) > window_size:
                # Apply the filter again, but only to baseline regions
                baseline_points = intensities.copy()
                baseline_points[~baseline_mask] = background[~baseline_mask]
                
                if window_type == "Uniform":
                    refined_background = ndimage.uniform_filter1d(baseline_points, size=window_size)
                elif window_type == "Gaussian":
                    sigma = window_size / 4.0
                    refined_background = ndimage.gaussian_filter1d(baseline_points, sigma=sigma)
                else:
                    refined_background = background
                
                background = np.minimum(refined_background, intensities)
            
            return background
            
        except ImportError:
            # Fallback without scipy
            window_size = max(int(len(intensities) * window_percent / 100.0), 3)
            
            # Simple moving minimum approach
            background = np.array([np.min(intensities[max(0, i-window_size//2):min(len(intensities), i+window_size//2+1)]) 
                                 for i in range(len(intensities))])
            
            # Simple smoothing
            for _ in range(2):
                smoothed = background.copy()
                for i in range(1, len(background)-1):
                    smoothed[i] = (background[i-1] + background[i] + background[i+1]) / 3
                background = smoothed
            
            return np.minimum(background, intensities)
            
        except Exception as e:
            print(f"Moving average background calculation error: {str(e)}")
            return np.full_like(intensities, np.min(intensities))
    
    def _calculate_spline_background_for_batch(self, intensities, n_knots, smoothing, degree):
        """Calculate spline background for batch processing."""
        try:
            from scipy.interpolate import UnivariateSpline
            
            # Create x values (indices)
            x = np.arange(len(intensities))
            
            # Step 1: Apply minimum filtering to identify baseline regions
            from scipy import ndimage
            window_size = max(len(intensities) // 20, 5)
            y_min_filtered = ndimage.minimum_filter1d(intensities, size=window_size)
            
            # Step 2: Create initial baseline estimate
            if n_knots <= 2:
                n_knots = 3  # Minimum for spline
            
            # For background subtraction, use higher smoothing to avoid fitting peaks
            background_smoothing = max(smoothing, len(intensities) / 10)
            
            try:
                # Fit spline to minimum filtered data for initial background estimate
                spline = UnivariateSpline(x, y_min_filtered, s=background_smoothing, k=min(degree, 3))
                initial_background = spline(x)
                
                # Step 3: Iterative refinement - only use points below or near the background
                current_background = initial_background.copy()
                
                for iteration in range(3):  # Limited iterations
                    # Identify points that are likely background
                    threshold = np.percentile(intensities - current_background, 20)
                    mask = (intensities - current_background) <= threshold
                    
                    if np.sum(mask) < n_knots:  # Need enough points
                        break
                    
                    # Fit spline only to identified background points
                    spline = UnivariateSpline(x[mask], intensities[mask], s=background_smoothing, k=min(degree, 3))
                    current_background = spline(x)
                    
                    # Ensure background doesn't go above data unrealistically
                    current_background = np.minimum(current_background, intensities)
                
                # Final constraint: background should be below the data
                background = np.minimum(current_background, intensities)
                
                return background
                
            except Exception:
                # Fallback: use simple Savitzky-Golay filter
                from scipy.signal import savgol_filter
                
                # Use Savitzky-Golay filter on minimum filtered data
                window_length = min(len(intensities) // 5, 51)
                if window_length % 2 == 0:
                    window_length += 1  # Must be odd
                
                background = savgol_filter(y_min_filtered, window_length, polyorder=min(degree, 3))
                return np.minimum(background, intensities)
                
        except ImportError:
            # If scipy is not available, fallback to simple polynomial
            print("scipy not available, using polynomial fallback for spline background")
            
            # Simple approach: fit polynomial to lower envelope
            window_size = max(len(intensities) // 10, 3)
            y_smooth = np.array([np.min(intensities[max(0, i-window_size):min(len(intensities), i+window_size+1)]) 
                               for i in range(len(intensities))])
            
            x = np.arange(len(intensities))
            coeffs = np.polyfit(x, y_smooth, min(degree, 3))
            background = np.polyval(coeffs, x)
            
            return np.minimum(background, intensities)
            
        except Exception as e:
            print(f"Spline background calculation error: {str(e)}")
            # Final fallback: simple linear baseline
            background = np.linspace(intensities[0], intensities[-1], len(intensities))
            return np.minimum(background, intensities)

    def update_background_options_dropdown(self):
        """Update the background options dropdown."""
        self.bg_options_combo.clear()
        self.bg_options_combo.addItem("None - Select an option")
        
        for i, (_, description, _, _) in enumerate(self.background_options):
            self.bg_options_combo.addItem(f"{i+1}. {description}")
    
    def on_bg_option_selected(self):
        """Handle selection of a background option."""
        selected_text = self.bg_options_combo.currentText()
        
        if selected_text.startswith("None") or not self.background_options:
            return
        
        try:
            # Extract option index
            option_index = int(selected_text.split('.')[0]) - 1
            
            if 0 <= option_index < len(self.background_options):
                # Highlight the selected option
                self._highlight_selected_background_option(option_index)
                
        except (ValueError, IndexError):
            pass
    
    def _highlight_selected_background_option(self, option_index):
        """Highlight the selected background option on the plot."""
        # Reset all line styles to normal
        for line in self.background_option_lines:
            try:
                line.set_linewidth(1.5)
                line.set_alpha(0.7)
            except:
                pass
        
        # Highlight selected option
        if 0 <= option_index < len(self.background_option_lines):
            try:
                selected_line = self.background_option_lines[option_index]
                selected_line.set_linewidth(3.0)
                selected_line.set_alpha(1.0)
                
                # Update canvas
                self.canvas.draw_idle()
                
                # Show option details
                _, description, method, params = self.background_options[option_index]
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                
                self.results_text.setPlainText(
                    f"Selected Background Option:\n"
                    f"Description: {description}\n"
                    f"Method: {method}\n"
                    f"Parameters: {param_str}\n\n"
                    f"Click 'Apply Selected' to use this background subtraction."
                )
                
            except Exception as e:
                print(f"Error highlighting option: {str(e)}")
    
    def apply_selected_background_option(self):
        """Apply the selected background option."""
        selected_text = self.bg_options_combo.currentText()
        
        if selected_text.startswith("None") or not self.background_options:
            QMessageBox.warning(self, "No Selection", "Please select a background option first.")
            return
        
        try:
            # Extract option index
            option_index = int(selected_text.split('.')[0]) - 1
            
            if 0 <= option_index < len(self.background_options):
                background_data, description, method, params = self.background_options[option_index]
                
                # Apply the background subtraction
                self.background = background_data.copy()
                self.processed_intensities = self.original_intensities - self.background
                self.background_preview_active = False
                
                # Update the manual parameter controls to match this selection
                self._update_manual_controls_from_params(method, params)
                
                # Clear the options and update plot
                self.clear_background_options()
                self.update_plot()
                
                # Show confirmation
                QMessageBox.information(self, "Background Applied", 
                                      f"Applied background subtraction:\n{description}\n\n"
                                      f"Method: {method}\n"
                                      f"Manual controls have been updated to match.")
                
        except (ValueError, IndexError) as e:
            QMessageBox.warning(self, "Selection Error", f"Invalid selection: {str(e)}")
    
    def _update_manual_controls_from_params(self, method, params):
        """Update manual parameter controls to match the selected option."""
        try:
            # Update method combo box - first check if using fallback controls
            if hasattr(self, 'bg_method_combo'):
                # Find and set the method
                index = self.bg_method_combo.findText(method)
                if index >= 0:
                    self.bg_method_combo.setCurrentIndex(index)
                    self._on_bg_method_changed()  # Update visibility of parameter widgets
                
                # Update method-specific parameters for fallback controls
                if method == "ALS":
                    if hasattr(self, 'lambda_slider') and "lambda" in params:
                        lambda_val = params["lambda"]
                        # Convert to log scale (3.0 to 7.0)
                        log_val = np.log10(lambda_val) * 10
                        slider_val = max(30, min(70, int(log_val)))
                        self.lambda_slider.setValue(slider_val)
                        self._update_lambda_label()
                    
                    if hasattr(self, 'p_slider') and "p" in params:
                        p_val = params["p"]
                        # Convert to slider scale (1-100)
                        slider_val = max(1, min(100, int(p_val * 1000)))
                        self.p_slider.setValue(slider_val)
                        self._update_p_label()
                    
                    if hasattr(self, 'niter_slider') and "niter" in params:
                        niter_val = params["niter"]
                        slider_val = max(5, min(20, int(niter_val)))
                        self.niter_slider.setValue(slider_val)
                        self._update_niter_label()
                        
                elif method == "Linear":
                    if hasattr(self, 'start_weight_slider') and "start_weight" in params:
                        weight_val = params["start_weight"]
                        # Convert to slider scale (1-20)
                        slider_val = max(1, min(20, int(weight_val * 10)))
                        self.start_weight_slider.setValue(slider_val)
                        self._update_start_weight_label()
                    
                    if hasattr(self, 'end_weight_slider') and "end_weight" in params:
                        weight_val = params["end_weight"]
                        # Convert to slider scale (1-20)
                        slider_val = max(1, min(20, int(weight_val * 10)))
                        self.end_weight_slider.setValue(slider_val)
                        self._update_end_weight_label()
                        
                elif method == "Polynomial":
                    if hasattr(self, 'poly_order_slider') and "order" in params:
                        order_val = params["order"]
                        slider_val = max(1, min(6, int(order_val)))
                        self.poly_order_slider.setValue(slider_val)
                        self._update_poly_order_label()
                    
                    if hasattr(self, 'poly_method_combo') and "poly_method" in params:
                        method_type = params["poly_method"]
                        index = self.poly_method_combo.findText(method_type)
                        if index >= 0:
                            self.poly_method_combo.setCurrentIndex(index)
                            
                elif method == "Moving Average":
                    if hasattr(self, 'window_percent_slider') and "window_percent" in params:
                        window_val = params["window_percent"]
                        self.window_percent_slider.setValue(max(5, min(50, window_val)))
                        self._update_window_percent_label()
                    
                    if hasattr(self, 'window_type_combo') and "window_type" in params:
                        window_type = params["window_type"]
                        index = self.window_type_combo.findText(window_type)
                        if index >= 0:
                            self.window_type_combo.setCurrentIndex(index)
                            
                elif method == "Spline":
                    if hasattr(self, 'n_knots_slider') and "n_knots" in params:
                        knots_val = params["n_knots"]
                        self.n_knots_slider.setValue(max(3, min(20, knots_val)))
                        self._update_n_knots_label()
                    
                    if hasattr(self, 'smoothing_slider') and "smoothing" in params:
                        smoothing_val = params["smoothing"]
                        self.smoothing_slider.setValue(max(1, min(1000, smoothing_val)))
                        self._update_smoothing_label()
                    
                    if hasattr(self, 'spline_degree_combo') and "degree" in params:
                        degree_val = str(params["degree"])
                        index = self.spline_degree_combo.findText(degree_val)
                        if index >= 0:
                            self.spline_degree_combo.setCurrentIndex(index)
            
        except Exception as e:
            print(f"Error updating manual controls: {str(e)}")
    
    def clear_background_options(self):
        """Clear all background options and their preview lines."""
        try:
            # Clear plot lines
            for line in self.background_option_lines:
                try:
                    line.remove()
                except:
                    pass
            
            # Clear data
            self.background_options.clear()
            self.background_option_lines.clear()
            
            # Reset dropdown
            self.bg_options_combo.clear()
            self.bg_options_combo.addItem("None - Generate options first")
            
            # Clear results text if it was showing option info
            if hasattr(self, 'results_text') and "Selected Background Option:" in self.results_text.toPlainText():
                self.results_text.clear()
            
            # Update plot
            if hasattr(self, 'ax_main'):
                self.ax_main.legend()
                self.canvas.draw_idle()
                
        except Exception as e:
            print(f"Error clearing background options: {str(e)}")

    def export_to_file(self, file_path):
        """Export results to specified file with enhanced peak data."""
        data = {
            'Wavenumber': self.wavenumbers,
            'Original_Intensity': self.original_intensities,
            'Processed_Intensity': self.processed_intensities
        }
        
        # Add background if available
        if self.background is not None:
            data['Background'] = self.background
        
        # Add residuals if available
        if self.residuals is not None:
            data['Residuals'] = self.residuals
        
        # Add total fitted curve if available
        if self.fit_result is not None and self.fit_params is not None:
            fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
            data['Total_Fitted_Curve'] = fitted_curve
        
        # Add individual peak curves (full spectrum)
        if (self.fit_result is not None and self.fit_params is not None and 
            hasattr(self, 'peaks') and len(self.get_all_peaks_for_fitting()) > 0):
            
            individual_r2_values = self.calculate_individual_r2_values()
            all_fitted_peaks = self.get_all_peaks_for_fitting()
            validated_peaks = self.validate_peak_indices(np.array(all_fitted_peaks))
            n_peaks = len(validated_peaks)
            
            for i in range(n_peaks):
                start_idx = i * 3
                if start_idx + 2 < len(self.fit_params):
                    amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                    r2_value = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                    
                    # Generate individual peak curve
                    if self.current_model == "Gaussian":
                        peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                    elif self.current_model == "Lorentzian":
                        peak_curve = self.lorentzian(self.wavenumbers, amp, cen, wid)
                    elif self.current_model == "Pseudo-Voigt":
                        peak_curve = self.pseudo_voigt(self.wavenumbers, amp, cen, wid)
                    else:
                        peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                    
                    data[f'Peak_{i+1}_Full_Curve'] = peak_curve
        
        # Add components if available
        for i, component in enumerate(self.components):
            data[f'Component_{i+1}'] = component
        
        # Create main DataFrame
        df = pd.DataFrame(data)
        
        # Save main spectral data
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
            base_path = file_path.replace('.csv', '')
        else:
            df.to_csv(file_path, sep='\t', index=False)
            base_path = file_path.replace('.txt', '') if file_path.endswith('.txt') else file_path
        
        # Export peak parameters and regional curves
        self._export_peak_details(base_path)
    
    def _export_peak_details(self, base_path):
        """Export detailed peak parameters and regional curves."""
        if (self.fit_result is None or self.fit_params is None or 
            not hasattr(self, 'peaks') or len(self.get_all_peaks_for_fitting()) == 0):
            return
        
        # Get peak data
        individual_r2_values = self.calculate_individual_r2_values()
        total_r2 = self.calculate_total_r2()
        all_fitted_peaks = self.get_all_peaks_for_fitting()
        validated_peaks = self.validate_peak_indices(np.array(all_fitted_peaks))
        n_peaks = len(validated_peaks)
        
        # Export 1: Peak Parameters Summary
        self._export_peak_parameters(base_path, n_peaks, individual_r2_values, total_r2)
        
        # Export 2: Regional Peak Curves (+/- 75 cm^-1 around each peak)
        self._export_regional_peak_curves(base_path, n_peaks, individual_r2_values)
    
    def _export_peak_parameters(self, base_path, n_peaks, individual_r2_values, total_r2):
        """Export peak parameters to a separate file."""
        peak_params = []
        
        # Add header information
        peak_params.append({
            'Parameter': 'Analysis Summary',
            'Value': '',
            'Units': '',
            'Notes': f'Model: {self.current_model}, Total R²: {total_r2:.4f}'
        })
        peak_params.append({
            'Parameter': 'Number of Peaks',
            'Value': n_peaks,
            'Units': '',
            'Notes': f'Average Individual R²: {np.mean(individual_r2_values):.3f}' if individual_r2_values else ''
        })
        peak_params.append({
            'Parameter': '',
            'Value': '',
            'Units': '',
            'Notes': ''
        })  # Empty row
        
        # Add individual peak parameters
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                r2_value = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                
                # Determine peak type
                peak_type = "Auto"
                if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                    validated_auto_peaks = self.validate_peak_indices(self.peaks)
                    all_fitted_peaks = self.get_all_peaks_for_fitting()
                    if (len(validated_auto_peaks) > 0 and i < len(all_fitted_peaks) and 
                        all_fitted_peaks[i] not in validated_auto_peaks.tolist()):
                        peak_type = "Manual"
                
                # Calculate additional peak properties
                fwhm = self._calculate_fwhm(wid)
                area = self._calculate_peak_area(amp, wid)
                
                peak_params.extend([
                    {
                        'Parameter': f'Peak {i+1} Type',
                        'Value': peak_type,
                        'Units': '',
                        'Notes': f'R² = {r2_value:.3f}'
                    },
                    {
                        'Parameter': f'Peak {i+1} Center',
                        'Value': f'{cen:.2f}',
                        'Units': 'cm⁻¹',
                        'Notes': 'Peak centroid position'
                    },
                    {
                        'Parameter': f'Peak {i+1} Amplitude',
                        'Value': f'{amp:.2f}',
                        'Units': 'intensity',
                        'Notes': 'Peak height'
                    },
                    {
                        'Parameter': f'Peak {i+1} Width',
                        'Value': f'{wid:.2f}',
                        'Units': 'cm⁻¹',
                        'Notes': f'Model parameter (FWHM ≈ {fwhm:.2f})'
                    },
                    {
                        'Parameter': f'Peak {i+1} Area',
                        'Value': f'{area:.2f}',
                        'Units': 'intensity·cm⁻¹',
                        'Notes': 'Integrated peak area'
                    },
                    {
                        'Parameter': '',
                        'Value': '',
                        'Units': '',
                        'Notes': ''
                    }  # Empty row
                ])
        
        # Save peak parameters
        peak_df = pd.DataFrame(peak_params)
        param_file = f"{base_path}_peak_parameters.csv"
        peak_df.to_csv(param_file, index=False)
    
    def _export_regional_peak_curves(self, base_path, n_peaks, individual_r2_values):
        """Export individual peak curves in +/- 75 cm^-1 regions around centroids."""
        regional_data = {}
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                r2_value = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                
                # Define region: +/- 75 cm^-1 around centroid
                region_start = cen - 75
                region_end = cen + 75
                
                # Find indices within this region
                region_mask = (self.wavenumbers >= region_start) & (self.wavenumbers <= region_end)
                
                if np.any(region_mask):
                    # Extract regional data
                    region_wavenumbers = self.wavenumbers[region_mask]
                    region_original = self.original_intensities[region_mask]
                    region_processed = self.processed_intensities[region_mask]
                    
                    # Generate individual peak curve for this region
                    if self.current_model == "Gaussian":
                        region_peak = self.gaussian(region_wavenumbers, amp, cen, wid)
                    elif self.current_model == "Lorentzian":
                        region_peak = self.lorentzian(region_wavenumbers, amp, cen, wid)
                    elif self.current_model == "Pseudo-Voigt":
                        region_peak = self.pseudo_voigt(region_wavenumbers, amp, cen, wid)
                    else:
                        region_peak = self.gaussian(region_wavenumbers, amp, cen, wid)
                    
                    # Generate total fit for this region
                    region_total_fit = self.multi_peak_model(region_wavenumbers, *self.fit_params)
                    
                    # Add background if available
                    region_background = None
                    if self.background is not None:
                        region_background = self.background[region_mask]
                    
                    # Store data with consistent length
                    max_length = max(len(regional_data.get('Wavenumber', [])), len(region_wavenumbers))
                    
                    # Extend existing columns if needed
                    for key in regional_data:
                        while len(regional_data[key]) < max_length:
                            regional_data[key].append(np.nan)
                    
                    # Add new data (pad if shorter than existing data)
                    if 'Wavenumber' not in regional_data:
                        regional_data['Wavenumber'] = []
                    
                    regional_data['Wavenumber'].extend(region_wavenumbers.tolist())
                    
                    # Pad wavenumber if needed
                    while len(regional_data['Wavenumber']) < max_length:
                        regional_data['Wavenumber'].append(np.nan)
                    
                    # Add peak-specific columns
                    col_prefix = f'Peak_{i+1}_{cen:.1f}cm'
                    
                    regional_data[f'{col_prefix}_Original'] = [np.nan] * max_length
                    regional_data[f'{col_prefix}_Processed'] = [np.nan] * max_length
                    regional_data[f'{col_prefix}_Individual'] = [np.nan] * max_length
                    regional_data[f'{col_prefix}_Total_Fit'] = [np.nan] * max_length
                    
                    if region_background is not None:
                        regional_data[f'{col_prefix}_Background'] = [np.nan] * max_length
                    
                    # Fill in the actual data for this peak's region
                    start_fill = max_length - len(region_wavenumbers)
                    for j, val in enumerate(region_original):
                        regional_data[f'{col_prefix}_Original'][start_fill + j] = val
                    for j, val in enumerate(region_processed):
                        regional_data[f'{col_prefix}_Processed'][start_fill + j] = val
                    for j, val in enumerate(region_peak):
                        regional_data[f'{col_prefix}_Individual'][start_fill + j] = val
                    for j, val in enumerate(region_total_fit):
                        regional_data[f'{col_prefix}_Total_Fit'][start_fill + j] = val
                    
                    if region_background is not None:
                        for j, val in enumerate(region_background):
                            regional_data[f'{col_prefix}_Background'][start_fill + j] = val
        
        # Create a simpler approach: separate file for each peak region
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                self._export_single_peak_region(base_path, i+1, amp, cen, wid, individual_r2_values)
    
    def _export_single_peak_region(self, base_path, peak_num, amp, cen, wid, individual_r2_values):
        """Export a single peak's regional data to its own file."""
        # Define region: +/- 75 cm^-1 around centroid
        region_start = cen - 75
        region_end = cen + 75
        
        # Find indices within this region
        region_mask = (self.wavenumbers >= region_start) & (self.wavenumbers <= region_end)
        
        if not np.any(region_mask):
            return
        
        # Extract regional data
        region_wavenumbers = self.wavenumbers[region_mask]
        region_original = self.original_intensities[region_mask]
        region_processed = self.processed_intensities[region_mask]
        
        # Generate curves for this region
        if self.current_model == "Gaussian":
            region_peak = self.gaussian(region_wavenumbers, amp, cen, wid)
        elif self.current_model == "Lorentzian":
            region_peak = self.lorentzian(region_wavenumbers, amp, cen, wid)
        elif self.current_model == "Pseudo-Voigt":
            region_peak = self.pseudo_voigt(region_wavenumbers, amp, cen, wid)
        else:
            region_peak = self.gaussian(region_wavenumbers, amp, cen, wid)
        
        # Generate total fit for this region
        region_total_fit = self.multi_peak_model(region_wavenumbers, *self.fit_params)
        
        # Create data dictionary
        peak_data = {
            'Wavenumber': region_wavenumbers,
            'Original_Intensity': region_original,
            'Processed_Intensity': region_processed,
            'Individual_Peak_Fit': region_peak,
            'Total_Fit': region_total_fit,
            'Residual': region_processed - region_total_fit
        }
        
        # Add background if available
        if self.background is not None:
            region_background = self.background[region_mask]
            peak_data['Background'] = region_background
        
        # Create DataFrame and save
        peak_df = pd.DataFrame(peak_data)
        r2_value = individual_r2_values[peak_num-1] if (peak_num-1) < len(individual_r2_values) else 0.0
        region_file = f"{base_path}_peak_{peak_num:02d}_{cen:.1f}cm_R2_{r2_value:.3f}.csv"
        peak_df.to_csv(region_file, index=False)
    
    def _calculate_fwhm(self, width_param):
        """Calculate FWHM from model width parameter."""
        if self.current_model == "Gaussian":
            # For Gaussian: FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
            return 2.355 * abs(width_param)
        elif self.current_model == "Lorentzian":
            # For Lorentzian: FWHM = 2 * gamma
            return 2 * abs(width_param)
        else:
            # Default to Gaussian approximation
            return 2.355 * abs(width_param)
    
    def _calculate_peak_area(self, amplitude, width_param):
        """Calculate integrated peak area."""
        if self.current_model == "Gaussian":
            # For Gaussian: Area = amplitude * width * sqrt(2*pi)
            return abs(amplitude * width_param * np.sqrt(2 * np.pi))
        elif self.current_model == "Lorentzian":
            # For Lorentzian: Area = amplitude * width * pi
            return abs(amplitude * width_param * np.pi)
        else:
            # Default to Gaussian approximation
            return abs(amplitude * width_param * np.sqrt(2 * np.pi))

    def update_peak_list(self):
        """Update the peak list widget with current peaks."""
        if not hasattr(self, 'peak_list_widget'):
            return
        
        self.peak_list_widget.clear()
        
        # Add automatic peaks
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            valid_auto_peaks = self.validate_peak_indices(self.peaks)
            for i, peak_idx in enumerate(valid_auto_peaks):
                wavenumber = self.wavenumbers[peak_idx]
                intensity = self.processed_intensities[peak_idx]
                item_text = f"🔴 Auto Peak {i+1}: {wavenumber:.1f} cm⁻¹ (I: {intensity:.1f})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, ('auto', peak_idx))
                self.peak_list_widget.addItem(item)
        
        # Add manual peaks
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            valid_manual_peaks = self.validate_peak_indices(self.manual_peaks)
            for i, peak_idx in enumerate(valid_manual_peaks):
                wavenumber = self.wavenumbers[peak_idx]
                intensity = self.processed_intensities[peak_idx]
                item_text = f"🟢 Manual Peak {i+1}: {wavenumber:.1f} cm⁻¹ (I: {intensity:.1f})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, ('manual', peak_idx))
                self.peak_list_widget.addItem(item)

    def remove_selected_peak(self):
        """Remove the selected peak from the list."""
        if not hasattr(self, 'peak_list_widget'):
            return
        
        current_item = self.peak_list_widget.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "No Selection", "Please select a peak to remove from the list.")
            return
        
        # Get peak data
        peak_data = current_item.data(Qt.UserRole)
        if peak_data is None:
            return
        
        peak_type, peak_idx = peak_data
        
        try:
            if peak_type == 'auto':
                # Remove from automatic peaks
                if hasattr(self, 'peaks') and self.peaks is not None:
                    peak_indices = list(self.peaks)
                    if peak_idx in peak_indices:
                        peak_indices.remove(peak_idx)
                        self.peaks = np.array(peak_indices)
                        
            elif peak_type == 'manual':
                # Remove from manual peaks
                if hasattr(self, 'manual_peaks') and self.manual_peaks is not None:
                    peak_indices = list(self.manual_peaks)
                    if peak_idx in peak_indices:
                        peak_indices.remove(peak_idx)
                        self.manual_peaks = np.array(peak_indices)
            
            # Update displays
            self.update_peak_count_display()
            self.update_peak_list()
            self.update_plot()
            
            # Show confirmation
            wavenumber = self.wavenumbers[peak_idx]
            QMessageBox.information(self, "Peak Removed", 
                                  f"Removed {peak_type} peak at {wavenumber:.1f} cm⁻¹")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to remove peak: {str(e)}")
    
    # ==================== BATCH PROCESSING METHODS ====================
    
    def add_batch_files(self):
        """Add files to the batch processing list."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Spectrum Files", "", 
            "Spectrum Files (*.txt *.csv *.dat *.asc *.spc *.xy *.tsv);;All Files (*)"
        )
        
        for file_path in files:
            if file_path not in [self.batch_file_list.item(i).data(Qt.UserRole) 
                                for i in range(self.batch_file_list.count())]:
                item = QListWidgetItem(Path(file_path).name)
                item.setData(Qt.UserRole, file_path)
                self.batch_file_list.addItem(item)
    
    def add_batch_folder(self):
        """Add all spectrum files from a folder to the batch processing list."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Spectrum Files")
        if not folder:
            return
        
        # Find supported spectrum files
        supported_extensions = ['.txt', '.csv', '.dat', '.asc', '.spc', '.xy', '.tsv']
        files = []
        for ext in supported_extensions:
            files.extend(Path(folder).glob(f"*{ext}"))
        
        added_count = 0
        for file_path in files:
            file_str = str(file_path)
            if file_str not in [self.batch_file_list.item(i).data(Qt.UserRole) 
                               for i in range(self.batch_file_list.count())]:
                item = QListWidgetItem(file_path.name)
                item.setData(Qt.UserRole, file_str)
                self.batch_file_list.addItem(item)
                added_count += 1
        
        QMessageBox.information(self, "Files Added", f"Added {added_count} files from {folder}")
    
    def clear_batch_files(self):
        """Clear the batch file list."""
        self.batch_file_list.clear()
    
    def add_batch_region(self):
        """Add a region to the batch processing list."""
        start = self.region_start_spin.value()
        end = self.region_end_spin.value()
        
        if start >= end:
            QMessageBox.warning(self, "Invalid Region", "Start wavenumber must be less than end wavenumber.")
            return
        
        region_text = f"{start:.1f} - {end:.1f} cm⁻¹"
        item = QListWidgetItem(region_text)
        item.setData(Qt.UserRole, (start, end))
        self.batch_regions_list.addItem(item)
    
    def clear_batch_regions(self):
        """Clear the batch regions list."""
        self.batch_regions_list.clear()
    
    def update_region_ranges_from_data(self, wavenumbers):
        """Update region spinbox ranges based on actual data range."""
        if len(wavenumbers) == 0:
            return None, None
            
        min_wave = float(np.min(wavenumbers))
        max_wave = float(np.max(wavenumbers))
        
        # Add small buffer
        range_buffer = (max_wave - min_wave) * 0.05
        min_range = max(0, min_wave - range_buffer)
        max_range = max_wave + range_buffer
        
        # Update spinbox ranges
        self.region_start_spin.setRange(min_range, max_range)
        self.region_end_spin.setRange(min_range, max_range)
        
        # Update default values to reasonable ranges within the data
        range_span = max_wave - min_wave
        default_start = min_wave + range_span * 0.1  # Start at 10% into the range
        default_end = min_wave + range_span * 0.9    # End at 90% into the range
        
        self.region_start_spin.setValue(default_start)
        self.region_end_spin.setValue(default_end)
        
        return min_wave, max_wave
    
    def start_batch_processing(self):
        """Start batch processing with current parameters and real-time monitoring."""
        # Check if files are selected
        if self.batch_file_list.count() == 0:
            QMessageBox.warning(self, "No Files", "Please add files to process.")
            return
        
        # Get current fitting parameters
        try:
            bg_params = self.get_fallback_background_parameters()
            peak_params = self.get_peak_parameters()
            
            # Get regions
            regions = []
            for i in range(self.batch_regions_list.count()):
                item = self.batch_regions_list.item(i)
                start, end = item.data(Qt.UserRole)
                regions.append((start, end))
            
            # If no regions specified, determine full spectrum range from first file
            default_region_calculated = False
            if not regions:
                regions = None  # Will be set after loading first file
            
            # Create and show the real-time monitor
            self.batch_monitor = BatchProcessingMonitor(self)
            self.batch_monitor.show()
            
            # Store reference for reopening
            self.last_batch_monitor = self.batch_monitor
            
            # Connect monitor signals
            self.batch_monitor.processing_cancelled.connect(self._on_batch_cancelled)
            self.batch_monitor.save_results_requested.connect(self.export_batch_results)
            
            # Show progress
            self.batch_progress_bar.setVisible(True)
            self.batch_progress_bar.setMaximum(self.batch_file_list.count())
            self.batch_progress_bar.setValue(0)
            
            # Process files with real-time monitoring
            results = []
            total_peaks_found = 0
            
            for i in range(self.batch_file_list.count()):
                # Check if processing was cancelled
                if hasattr(self, 'batch_monitor') and self.batch_monitor.is_cancelled:
                    break
                    
                item = self.batch_file_list.item(i)
                file_path = item.data(Qt.UserRole)
                
                # Update monitor progress
                self.batch_monitor.update_progress(i, self.batch_file_list.count(), file_path)
                
                try:
                    start_time = time.time()
                    
                    # Load spectrum
                    if FILE_LOADING_AVAILABLE:
                        wavenumbers, intensities, metadata = load_spectrum_file(file_path)
                        # Check if loading was successful
                        if wavenumbers is None or intensities is None:
                            error_msg = metadata.get("error", "Unknown error occurred")
                            raise Exception(f"Failed to load spectrum: {error_msg}")
                    else:
                        # Fallback loading
                        data = np.loadtxt(file_path)
                        if data.ndim == 1:
                            raise Exception("File must contain at least 2 columns (wavenumbers and intensities)")
                        elif data.shape[1] < 2:
                            raise Exception("File must contain at least 2 columns (wavenumbers and intensities)")
                        wavenumbers = data[:, 0]
                        intensities = data[:, 1]
                    
                    # Set default region based on first file if no regions specified
                    if regions is None and not default_region_calculated:
                        min_wave, max_wave = self.update_region_ranges_from_data(wavenumbers)
                        if min_wave is not None and max_wave is not None:
                            regions = [(min_wave, max_wave)]
                            default_region_calculated = True
                            print(f"Auto-detected spectrum range: {min_wave:.1f} - {max_wave:.1f} cm⁻¹")
                        else:
                            regions = [(0, 4000)]  # Fallback if range detection fails
                    
                    # Process each region
                    file_results = {
                        'filename': Path(file_path).name,
                        'filepath': file_path,
                        'regions': []
                    }
                    
                    file_peaks_count = 0
                    
                    for region_idx, (start, end) in enumerate(regions):
                        # Check pause/cancel state
                        if hasattr(self, 'batch_monitor'):
                            self.batch_monitor.wait_if_paused()
                            if self.batch_monitor.is_cancelled:
                                break
                        
                        # Update region info
                        self.batch_monitor.update_region_info(start, end, region_idx, len(regions))
                        
                        # Filter data to region
                        mask = (wavenumbers >= start) & (wavenumbers <= end)
                        region_wave = wavenumbers[mask]
                        region_int = intensities[mask]
                        
                        if len(region_wave) < 10:  # Skip very small regions
                            continue
                        
                        # Store original intensities for plotting
                        original_region_int = region_int.copy()
                        background = None
                        
                        # Apply background correction (always enabled)
                        # Use the same background method selected in the background tab
                        method = bg_params.get('method', 'ALS')
                        try:
                            background = self._calculate_background_for_batch(region_int, method, bg_params)
                            region_int = region_int - background
                            print(f"✅ Applied {method} background correction to {Path(file_path).name}")
                        except Exception as e:
                            print(f"⚠️ Background correction failed for {Path(file_path).name}: {str(e)}")
                            # Fall back to ALS if the selected method fails
                            baseline_fitter = self._get_baseline_fitter()
                            background = baseline_fitter.baseline_als(
                                region_int, 
                                bg_params.get('lambda', 1e5),
                                bg_params.get('p', 0.01), 
                                bg_params.get('niter', 10)
                            )
                            region_int = region_int - background
                            print(f"⚠️ Fell back to ALS background correction")
                        
                        # Determine peaks to use
                        peaks = None
                        fitted_peaks = None
                        residuals = None
                        fit_params = None
                        total_r2 = None
                        
                        # Check for manual peaks first
                        use_manual_peaks = False
                        if (self.batch_manual_peaks_radio.isChecked() and 
                            hasattr(self, 'manual_peaks') and 
                            self.manual_peaks is not None and 
                            len(self.manual_peaks) > 0):
                            
                            try:
                                # Convert manual peaks (wavenumber indices) to region indices
                                manual_peaks_in_region = []
                                print(f"🔍 Converting {len(self.manual_peaks)} manual peaks for region {start}-{end} cm⁻¹")
                                
                                for i, peak_idx in enumerate(self.manual_peaks):
                                    # Ensure peak_idx is an integer for numpy indexing
                                    peak_idx = int(peak_idx)
                                    if 0 <= peak_idx < len(self.wavenumbers):
                                        peak_wavenumber = self.wavenumbers[peak_idx]
                                        # Find closest point in region
                                        if start <= peak_wavenumber <= end:
                                            region_idx = np.argmin(np.abs(region_wave - peak_wavenumber))
                                            manual_peaks_in_region.append(int(region_idx))
                                            print(f"  ✓ Manual peak {i+1}: {peak_wavenumber:.1f} cm⁻¹ → region index {int(region_idx)}")
                                        else:
                                            print(f"  ⚠ Manual peak {i+1}: {peak_wavenumber:.1f} cm⁻¹ outside region")
                                    else:
                                        print(f"  ⚠ Manual peak {i+1}: index {peak_idx} out of bounds")
                                
                                if manual_peaks_in_region:
                                    peaks = np.array(manual_peaks_in_region, dtype=int)
                                    use_manual_peaks = True
                                    print(f"📍 Using {len(peaks)} manual peaks in {Path(file_path).name}")
                                else:
                                    print(f"⚠️ No manual peaks found in region {start}-{end} cm⁻¹")
                                    
                            except Exception as e:
                                print(f"❌ Error processing manual peaks: {str(e)}")
                                print(f"   Manual peaks array: {self.manual_peaks}")
                                print(f"   Manual peaks type: {type(self.manual_peaks)}")
                                # Fall back to auto-detection on error
                        
                        # Check for found peaks if not using manual peaks
                        use_found_peaks = False
                        if (not use_manual_peaks and 
                            self.batch_found_peaks_radio.isChecked() and 
                            hasattr(self, 'peaks') and 
                            self.peaks is not None and 
                            len(self.peaks) > 0):
                            
                            try:
                                # Convert found peaks (wavenumber indices) to region indices
                                found_peaks_in_region = []
                                print(f"🔍 Converting {len(self.peaks)} found peaks for region {start}-{end} cm⁻¹")
                                
                                for i, peak_idx in enumerate(self.peaks):
                                    # Ensure peak_idx is an integer for numpy indexing
                                    peak_idx = int(peak_idx)
                                    if 0 <= peak_idx < len(self.wavenumbers):
                                        peak_wavenumber = self.wavenumbers[peak_idx]
                                        # Find closest point in region
                                        if start <= peak_wavenumber <= end:
                                            region_idx = np.argmin(np.abs(region_wave - peak_wavenumber))
                                            found_peaks_in_region.append(int(region_idx))
                                            print(f"  ✓ Found peak {i+1}: {peak_wavenumber:.1f} cm⁻¹ → region index {int(region_idx)}")
                                        else:
                                            print(f"  ⚠ Found peak {i+1}: {peak_wavenumber:.1f} cm⁻¹ outside region")
                                    else:
                                        print(f"  ⚠ Found peak {i+1}: index {peak_idx} out of bounds")
                                
                                if found_peaks_in_region:
                                    peaks = np.array(found_peaks_in_region, dtype=int)
                                    use_found_peaks = True
                                    print(f"📍 Using {len(peaks)} found peaks in {Path(file_path).name}")
                                else:
                                    print(f"⚠️ No found peaks in region {start}-{end} cm⁻¹")
                                    
                            except Exception as e:
                                print(f"❌ Error processing found peaks: {str(e)}")
                                print(f"   Found peaks array: {self.peaks}")
                                print(f"   Found peaks type: {type(self.peaks)}")
                                # Fall back to auto-detection on error
                        
                        # Auto-detect peaks if requested and no manual/found peaks used
                        if (not use_manual_peaks and 
                            not use_found_peaks and 
                            self.batch_auto_peaks_radio.isChecked()):
                            peaks, _ = find_peaks(
                                region_int, 
                                height=peak_params.get('height', 0.1) * np.max(region_int),
                                distance=peak_params.get('distance', 10),
                                prominence=peak_params.get('prominence', 0.05) * np.max(region_int)
                            )
                            if peaks is not None and len(peaks) > 0:
                                print(f"🔍 Auto-detected {len(peaks)} peaks in {Path(file_path).name}")
                        
                        # Perform peak fitting if we have peaks
                        if peaks is not None and len(peaks) > 0:
                            file_peaks_count += len(peaks)
                            
                            try:
                                fitted_peaks, fit_params, total_r2, residuals = self._fit_peaks_for_batch(
                                    region_wave, region_int, peaks, peak_params
                                )
                                print(f"✅ Fitted {len(peaks)} peaks in {Path(file_path).name}, R² = {total_r2:.3f}")
                            except Exception as e:
                                print(f"⚠️ Peak fitting failed for {Path(file_path).name}: {str(e)}")
                                fitted_peaks = None
                                residuals = region_int.copy()  # No fitting, show original
                        
                        # Update real-time plot with fitted peaks and residuals
                        self.batch_monitor.update_spectrum_plot(
                            region_wave, original_region_int, 
                            background=background, peaks=peaks,
                            region_start=start, region_end=end,
                            full_wavenumbers=wavenumbers, full_intensities=intensities,
                            fitted_peaks=fitted_peaks, residuals=residuals
                        )
                        
                        # Store result for navigation after completion
                        self.batch_monitor.store_processing_result(
                            region_wave, original_region_int, background, peaks,
                            start, end, Path(file_path).name,
                            full_wavenumbers=wavenumbers, full_intensities=intensities,
                            fitted_peaks=fitted_peaks, residuals=residuals,
                            fit_params=fit_params, total_r2=total_r2
                        )
                        
                        # Allow GUI to update
                        QApplication.processEvents()
                        
                        # Small delay to see the visualization
                        time.sleep(0.1)
                        
                        region_result = {
                            'start': start,
                            'end': end,
                            'wavenumbers': region_wave,
                            'intensities': region_int,  # Background-corrected intensities
                            'original_intensities': original_region_int,  # Original intensities
                            'background': background,  # Background profile
                            'full_wavenumbers': wavenumbers,  # Complete spectrum wavenumbers
                            'full_intensities': intensities,  # Complete spectrum intensities
                            'peaks': peaks,
                            'fitted_peaks': fitted_peaks,
                            'fit_params': fit_params,
                            'residuals': residuals,
                            'total_r2': total_r2,
                            'background_params': bg_params,
                            'peak_params': peak_params
                        }
                        
                        file_results['regions'].append(region_result)
                    
                    # Check if cancelled during region processing
                    if hasattr(self, 'batch_monitor') and self.batch_monitor.is_cancelled:
                        break
                    
                    results.append(file_results)
                    total_peaks_found += file_peaks_count
                    
                    # Update statistics
                    processing_time = time.time() - start_time
                    self.batch_monitor.update_statistics(
                        i, file_peaks_count, processing_time, 
                        total_peaks_found, self.batch_file_list.count()
                    )
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    QMessageBox.warning(self, "Processing Error", error_msg)
                    print(f"Batch processing error: {error_msg}")
                
                # Update progress
                self.batch_progress_bar.setValue(i + 1)
            
            # Store results for export
            self.batch_results = results
            
            # Update results display
            total_files = len(results)
            total_regions = sum(len(r['regions']) for r in results)
            
            results_text = f"Batch Processing Complete!\n"
            results_text += f"Files processed: {total_files}\n"
            results_text += f"Regions analyzed: {total_regions}\n"
            results_text += f"Peaks detected: {total_peaks_found}\n"
            
            self.batch_results_text.setPlainText(results_text)
            self.batch_progress_bar.setVisible(False)
            
            # Refresh analysis plots with new data
            if hasattr(self, 'refresh_analysis_plots'):
                self.refresh_analysis_plots()
            
            # Show completion message if not cancelled
            if not (hasattr(self, 'batch_monitor') and self.batch_monitor.is_cancelled):
                # Signal completion to monitor
                self.batch_monitor.processing_completed()
                
                QMessageBox.information(self, "Batch Complete", 
                                      f"Batch processing completed successfully!\n\n"
                                      f"Processed {total_files} files\n"
                                      f"Found {total_peaks_found} peaks total\n"
                                      f"Analyzed {total_regions} regions")
            
        except Exception as e:
            error_msg = str(e)
            QMessageBox.critical(self, "Batch Error", f"Batch processing failed: {error_msg}")
            self.batch_progress_bar.setVisible(False)
            if hasattr(self, 'batch_monitor'):
                self.batch_monitor.processing_failed(error_msg)
    
    def _fit_peaks_for_batch(self, wavenumbers, intensities, peaks, peak_params):
        """
        Fit peaks for batch processing.
        
        Args:
            wavenumbers: Region wavenumbers
            intensities: Region intensities (background-corrected)
            peaks: Peak indices
            peak_params: Peak detection parameters
            
        Returns:
            tuple: (fitted_peaks, fit_params, total_r2, residuals)
        """
        if len(peaks) == 0:
            return None, None, None, intensities.copy()
            
        # Create initial parameter guesses
        initial_params = []
        bounds_lower = []
        bounds_upper = []
        
        for peak_idx in peaks:
            # Ensure peak_idx is an integer for numpy indexing
            peak_idx = int(peak_idx)
            if 0 <= peak_idx < len(intensities):
                # Amplitude: Use actual intensity at peak
                amp = intensities[peak_idx]
                
                # Center: Use wavenumber at peak
                cen = wavenumbers[peak_idx]
                
                # Width: Estimate from local curvature
                wid = self._estimate_peak_width_for_batch(wavenumbers, intensities, peak_idx)
                
                initial_params.extend([amp, cen, wid])
                
                # Set reasonable bounds
                bounds_lower.extend([amp * 0.1, cen - wid * 2, wid * 0.3])
                bounds_upper.extend([amp * 10, cen + wid * 2, wid * 3])
        
        if not initial_params:
            return None, None, None, intensities.copy()
        
        # Get current model
        current_model = peak_params.get('model', 'gaussian')
        
        # Define fitting strategies
        strategies = [
            {
                'p0': initial_params,
                'bounds': (bounds_lower, bounds_upper),
                'method': 'trf',
                'max_nfev': 1000
            },
            {
                'p0': initial_params,
                'bounds': ([b * 0.5 for b in bounds_lower], [b * 1.5 for b in bounds_upper]),
                'method': 'lm',
                'max_nfev': 500
            },
            {
                'p0': initial_params,
                'method': 'lm',
                'max_nfev': 1000
            }
        ]
        
        # Try fitting with different strategies
        for strategy in strategies:
            try:
                if 'bounds' in strategy:
                    popt, pcov = curve_fit(
                        lambda x, *params: self._multi_peak_model_for_batch(x, params, current_model),
                        wavenumbers, intensities,
                        p0=strategy['p0'],
                        bounds=strategy['bounds'],
                        method=strategy['method'],
                        max_nfev=strategy['max_nfev']
                    )
                else:
                    popt, pcov = curve_fit(
                        lambda x, *params: self._multi_peak_model_for_batch(x, params, current_model),
                        wavenumbers, intensities,
                        p0=strategy['p0'],
                        method=strategy['method'],
                        max_nfev=strategy['max_nfev']
                    )
                
                # Calculate fitted curve and residuals
                fitted_curve = self._multi_peak_model_for_batch(wavenumbers, popt, current_model)
                residuals = intensities - fitted_curve
                
                # Calculate R-squared
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
                total_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return fitted_curve, popt, total_r2, residuals
                
            except Exception as e:
                continue
        
        # If all strategies failed, return None
        return None, None, None, intensities.copy()
    
    def _multi_peak_model_for_batch(self, x, params, model='gaussian'):
        """Multi-peak model for batch processing."""
        if len(params) % 3 != 0:
            raise ValueError("Parameters must be multiples of 3 (amp, cen, wid)")
        
        n_peaks = len(params) // 3
        y = np.zeros_like(x)
        
        for i in range(n_peaks):
            amp = params[i * 3]
            cen = params[i * 3 + 1]
            wid = params[i * 3 + 2]
            
            if model == 'gaussian':
                y += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
            elif model == 'lorentzian':
                y += amp * (wid**2 / ((x - cen)**2 + wid**2))
            elif model == 'pseudo_voigt':
                eta = 0.5  # Default mixing parameter
                gaussian_part = np.exp(-(x - cen)**2 / (2 * wid**2))
                lorentzian_part = wid**2 / ((x - cen)**2 + wid**2)
                y += amp * ((1 - eta) * gaussian_part + eta * lorentzian_part)
            else:
                # Default to gaussian
                y += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
        
        return y
    
    def _estimate_peak_width_for_batch(self, wavenumbers, intensities, peak_idx):
        """Estimate peak width for batch processing."""
        try:
            # Look at points around the peak
            window = min(20, len(intensities) // 4)  # Adaptive window
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(intensities), peak_idx + window + 1)
            
            local_intensities = intensities[start_idx:end_idx]
            local_wavenumbers = wavenumbers[start_idx:end_idx]
            
            peak_intensity = intensities[peak_idx]
            half_max = peak_intensity / 2
            
            # Find FWHM
            above_half = local_intensities > half_max
            if np.any(above_half):
                indices = np.where(above_half)[0]
                if len(indices) > 1:
                    fwhm_indices = [indices[0], indices[-1]]
                    fwhm = abs(local_wavenumbers[fwhm_indices[1]] - local_wavenumbers[fwhm_indices[0]])
                    # Convert FWHM to Gaussian sigma
                    width = fwhm / (2 * np.sqrt(2 * np.log(2)))
                    return max(width, 5.0)
            
            # Fallback: estimate based on wavenumber spacing
            if len(wavenumbers) > 1:
                wavenumber_spacing = np.mean(np.diff(wavenumbers))
                return max(10 * wavenumber_spacing, 5.0)
            
            return 10.0
            
        except Exception:
            return 10.0

    def _on_batch_cancelled(self):
        """Handle batch processing cancellation."""
        self.batch_progress_bar.setVisible(False)
        QMessageBox.information(self, "Batch Cancelled", "Batch processing was cancelled by user.")
    
    def export_batch_results(self):
        """Export batch results to a pickle file."""
        if not hasattr(self, 'batch_results') or not self.batch_results:
            QMessageBox.warning(self, "No Results", "No batch results to export. Please run batch processing first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Results", "batch_results.pkl", 
            "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            try:
                import pickle
                import pandas as pd
                
                # Convert batch results to pandas format expected by jupyter console
                converted_data = self._convert_batch_results_to_pandas_format()
                
                with open(file_path, 'wb') as f:
                    pickle.dump(converted_data, f)
                
                QMessageBox.information(self, "Export Complete", f"Batch results exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def _convert_batch_results_to_pandas_format(self):
        """Convert batch results to pandas format expected by jupyter console."""
        import pandas as pd
        import numpy as np
        
        peak_data = []
        summary_data = []
        spectra_dict = {}
        
        # Process each file result
        for file_result in self.batch_results:
            filename = file_result['filename']
            filepath = file_result.get('filepath', '')
            
            # Add to spectra dict for each region
            for region_idx, region_result in enumerate(file_result['regions']):
                region_start = region_result['start']
                region_end = region_result['end']
                peaks = region_result.get('peaks')
                fit_params = region_result.get('fit_params')
                total_r2 = region_result.get('total_r2')
                wavenumbers = region_result.get('wavenumbers')
                intensities = region_result.get('intensities')
                
                # Create unique key for spectra dict
                if len(file_result['regions']) > 1:
                    spectrum_key = f"{filename}_region_{region_idx+1}"
                else:
                    spectrum_key = filename
                
                # Store comprehensive spectrum data
                if wavenumbers is not None and intensities is not None:
                    spectrum_data = {
                        'wavenumbers': wavenumbers,
                        'intensities': intensities,  # Background-corrected intensities
                        'filename': filename,
                        'region_start': region_start,
                        'region_end': region_end
                    }
                    
                    # Add original intensities (before background correction)
                    original_intensities = region_result.get('original_intensities')
                    if original_intensities is not None:
                        spectrum_data['original_intensities'] = original_intensities
                    
                    # Add background profile
                    background = region_result.get('background')
                    if background is not None:
                        spectrum_data['background'] = background
                    
                    # Add fitted peaks (calculated profile) if available
                    fitted_peaks = region_result.get('fitted_peaks')
                    if fitted_peaks is not None:
                        spectrum_data['fitted_peaks'] = fitted_peaks
                    
                    # Add residuals if available
                    residuals = region_result.get('residuals')
                    if residuals is not None:
                        spectrum_data['residuals'] = residuals
                    
                    # Add full spectrum data if available (beyond just the region)
                    # This is useful for plotting complete spectra
                    full_wavenumbers = region_result.get('full_wavenumbers')
                    full_intensities = region_result.get('full_intensities')
                    if full_wavenumbers is not None and full_intensities is not None:
                        spectrum_data['full_wavenumbers'] = full_wavenumbers
                        spectrum_data['full_intensities'] = full_intensities
                    
                    spectra_dict[spectrum_key] = spectrum_data
                
                n_peaks = len(peaks) if peaks is not None else 0
                
                # Create summary row
                summary_row = {
                    'file_index': len(summary_data),
                    'filename': filename,
                    'filepath': filepath,
                    'region_start': region_start,
                    'region_end': region_end,
                    'n_peaks': n_peaks,
                    'total_r2': total_r2,
                    'processing_time': None,  # Not available from batch_results
                    'fitting_success': 'Yes' if total_r2 is not None and total_r2 > 0 else 'No'
                }
                
                summary_data.append(summary_row)
                
                # Add peak parameters if available
                if peaks is not None and fit_params is not None and n_peaks > 0:
                    for i in range(n_peaks):
                        if i * 3 + 2 < len(fit_params):
                            amplitude = fit_params[i * 3]
                            center = fit_params[i * 3 + 1]
                            width = fit_params[i * 3 + 2]
                            
                            # Calculate derived parameters
                            fwhm = width * 2 * np.sqrt(2 * np.log(2))
                            area = amplitude * width * np.sqrt(2 * np.pi)
                            
                            # Create peak center in wavenumbers
                            peak_center_wavenumber = center
                            
                            peak_row = {
                                'file_index': len(summary_data) - 1,
                                'filename': filename,
                                'region_start': region_start,
                                'region_end': region_end,
                                'peak_number': i + 1,
                                'peak_center': peak_center_wavenumber,
                                'amplitude': amplitude,
                                'width': width,
                                'fwhm': fwhm,
                                'area': area,
                                'total_r2': total_r2 if i == 0 else None  # Only show R2 for first peak
                            }
                            
                            peak_data.append(peak_row)
        
        # Create pandas DataFrames
        summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
        peaks_df = pd.DataFrame(peak_data) if peak_data else pd.DataFrame()
        
        # Create the expected data structure
        converted_data = {
            'summary_df': summary_df,
            'peaks_df': peaks_df,
            'spectra_dict': spectra_dict,
            'metadata': {
                'export_method': 'batch_processing',
                'total_files': len(self.batch_results),
                'total_regions': sum(len(r['regions']) for r in self.batch_results),
                'total_peaks': len(peak_data)
            }
        }
        
        return converted_data
    
    def export_batch_to_csv(self):
        """Export batch results to comprehensive CSV files from the batch tab."""
        if not hasattr(self, 'batch_results') or not self.batch_results:
            QMessageBox.warning(self, "No Results", "No batch results to export. Please run batch processing first.")
            return
        
        # Check if the monitor has results (prefer monitor's enhanced export)
        if hasattr(self, 'last_batch_monitor') and self.last_batch_monitor:
            if self.last_batch_monitor.stored_results:
                # Use the monitor's CSV export functionality with custom naming
                self.last_batch_monitor.export_to_csv()
                return
        
        # Fallback: create export from batch_results with custom naming
        base_file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Results - Choose Base Filename", 
            "batch_analysis_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not base_file_path:
            return
        
        try:
            self._convert_batch_results_to_csv_with_custom_names(base_file_path)
            save_dir = Path(base_file_path).parent
            QMessageBox.information(self, "Export Complete", 
                                  f"CSV files exported successfully to:\n{save_dir}\n\n"
                                  f"Files created:\n"
                                  f"• Summary with peak parameters\n"
                                  f"• Detailed peak parameters\n"
                                  f"• Full spectral data")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {str(e)}")
    
    def _convert_batch_results_to_csv_with_custom_names(self, base_file_path):
        """Convert batch_results to CSV format with custom filenames."""
        base_path = Path(base_file_path)
        base_name = base_path.stem  # Get filename without extension
        save_dir = base_path.parent
        
        # Generate descriptive filenames based on user's choice
        summary_file = save_dir / f"{base_name}_summary.csv"
        peak_params_file = save_dir / f"{base_name}_peak_parameters.csv"
        
        peak_data = []
        summary_data = []
        max_peaks = 0
        
        # First pass: determine maximum number of peaks for enhanced summary
        for file_result in self.batch_results:
            for region_result in file_result['regions']:
                peaks = region_result.get('peaks')
                n_peaks = len(peaks) if peaks is not None else 0
                max_peaks = max(max_peaks, n_peaks)
        
        # Second pass: process data
        for file_result in self.batch_results:
            filename = file_result['filename']
            
            for region_result in file_result['regions']:
                region_start = region_result['start']
                region_end = region_result['end']
                peaks = region_result.get('peaks')
                fit_params = region_result.get('fit_params')
                total_r2 = region_result.get('total_r2')
                
                n_peaks = len(peaks) if peaks is not None else 0
                
                # Enhanced summary with individual peak parameters as columns
                summary_row = {
                    'Filename': filename,
                    'Region_Start': region_start,
                    'Region_End': region_end,
                    'Number_of_Peaks': n_peaks,
                    'Total_R2': total_r2,
                    'Fitting_Success': 'Yes' if total_r2 is not None and total_r2 > 0 else 'No'
                }
                
                # Add individual peak parameters as columns
                if fit_params is not None and n_peaks > 0:
                    for i in range(max_peaks):
                        if i < n_peaks and i * 3 + 2 < len(fit_params):
                            amplitude = fit_params[i * 3]
                            center = fit_params[i * 3 + 1]
                            width = fit_params[i * 3 + 2]
                            
                            # Calculate FWHM
                            fwhm = width * 2 * np.sqrt(2 * np.log(2))
                            
                            # Add peak parameters
                            summary_row[f'Peak{i+1}_Amplitude'] = amplitude
                            summary_row[f'Peak{i+1}_Position_cm-1'] = center
                            summary_row[f'Peak{i+1}_FWHM'] = fwhm
                            summary_row[f'Peak{i+1}_Width_Sigma'] = width
                        else:
                            # Fill with NaN for missing peaks
                            summary_row[f'Peak{i+1}_Amplitude'] = np.nan
                            summary_row[f'Peak{i+1}_Position_cm-1'] = np.nan
                            summary_row[f'Peak{i+1}_FWHM'] = np.nan
                            summary_row[f'Peak{i+1}_Width_Sigma'] = np.nan
                else:
                    # No fit parameters available - fill with NaN
                    for i in range(max_peaks):
                        summary_row[f'Peak{i+1}_Amplitude'] = np.nan
                        summary_row[f'Peak{i+1}_Position_cm-1'] = np.nan
                        summary_row[f'Peak{i+1}_FWHM'] = np.nan
                        summary_row[f'Peak{i+1}_Width_Sigma'] = np.nan
                
                summary_data.append(summary_row)
                
                # Add peak parameters for detailed file (unchanged for backward compatibility)
                if peaks is not None and fit_params is not None:
                    for i in range(n_peaks):
                        if i * 3 + 2 < len(fit_params):
                            amplitude = fit_params[i * 3]
                            center = fit_params[i * 3 + 1]
                            width = fit_params[i * 3 + 2]
                            
                            # Calculate FWHM and area
                            fwhm = width * 2 * np.sqrt(2 * np.log(2))
                            area = amplitude * width * np.sqrt(2 * np.pi)
                            
                            peak_data.append({
                                'Filename': filename,
                                'Region_Start': region_start,
                                'Region_End': region_end,
                                'Peak_Number': i + 1,
                                'Center_cm-1': center,
                                'Amplitude': amplitude,
                                'Width_Sigma': width,
                                'FWHM': fwhm,
                                'Area': area,
                                'Total_R2': total_r2 if i == 0 else ''
                            })
        
        # Save CSV files with custom names
        if peak_data:
            peak_df = pd.DataFrame(peak_data)
            peak_df.to_csv(peak_params_file, index=False)
            print(f"✅ Saved detailed peak parameters: {peak_params_file}")
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Saved enhanced summary: {summary_file}")

    def _convert_batch_results_to_csv(self, save_dir):
        """Convert batch_results to CSV format (legacy fallback method)."""
        save_path = Path(save_dir)
        
        peak_data = []
        summary_data = []
        max_peaks = 0
        
        # First pass: determine maximum number of peaks for enhanced summary
        for file_result in self.batch_results:
            for region_result in file_result['regions']:
                peaks = region_result.get('peaks')
                n_peaks = len(peaks) if peaks is not None else 0
                max_peaks = max(max_peaks, n_peaks)
        
        # Second pass: process data
        for file_result in self.batch_results:
            filename = file_result['filename']
            
            for region_result in file_result['regions']:
                region_start = region_result['start']
                region_end = region_result['end']
                peaks = region_result.get('peaks')
                fit_params = region_result.get('fit_params')
                total_r2 = region_result.get('total_r2')
                
                n_peaks = len(peaks) if peaks is not None else 0
                
                # Enhanced summary with individual peak parameters as columns
                summary_row = {
                    'Filename': filename,
                    'Region_Start': region_start,
                    'Region_End': region_end,
                    'Number_of_Peaks': n_peaks,
                    'Total_R2': total_r2,
                    'Fitting_Success': 'Yes' if total_r2 is not None and total_r2 > 0 else 'No'
                }
                
                # Add individual peak parameters as columns
                if fit_params is not None and n_peaks > 0:
                    for i in range(max_peaks):
                        if i < n_peaks and i * 3 + 2 < len(fit_params):
                            amplitude = fit_params[i * 3]
                            center = fit_params[i * 3 + 1]
                            width = fit_params[i * 3 + 2]
                            
                            # Calculate FWHM
                            fwhm = width * 2 * np.sqrt(2 * np.log(2))
                            
                            # Add peak parameters
                            summary_row[f'Peak{i+1}_Amplitude'] = amplitude
                            summary_row[f'Peak{i+1}_Position_cm-1'] = center
                            summary_row[f'Peak{i+1}_FWHM'] = fwhm
                            summary_row[f'Peak{i+1}_Width_Sigma'] = width
                        else:
                            # Fill with NaN for missing peaks
                            summary_row[f'Peak{i+1}_Amplitude'] = np.nan
                            summary_row[f'Peak{i+1}_Position_cm-1'] = np.nan
                            summary_row[f'Peak{i+1}_FWHM'] = np.nan
                            summary_row[f'Peak{i+1}_Width_Sigma'] = np.nan
                else:
                    # No fit parameters available - fill with NaN
                    for i in range(max_peaks):
                        summary_row[f'Peak{i+1}_Amplitude'] = np.nan
                        summary_row[f'Peak{i+1}_Position_cm-1'] = np.nan
                        summary_row[f'Peak{i+1}_FWHM'] = np.nan
                        summary_row[f'Peak{i+1}_Width_Sigma'] = np.nan
                
                summary_data.append(summary_row)
                
                # Add peak parameters for detailed file (unchanged for backward compatibility)
                if peaks is not None and fit_params is not None:
                    for i in range(n_peaks):
                        if i * 3 + 2 < len(fit_params):
                            amplitude = fit_params[i * 3]
                            center = fit_params[i * 3 + 1]
                            width = fit_params[i * 3 + 2]
                            
                            # Calculate FWHM and area
                            fwhm = width * 2 * np.sqrt(2 * np.log(2))
                            area = amplitude * width * np.sqrt(2 * np.pi)
                            
                            peak_data.append({
                                'Filename': filename,
                                'Region_Start': region_start,
                                'Region_End': region_end,
                                'Peak_Number': i + 1,
                                'Center_cm-1': center,
                                'Amplitude': amplitude,
                                'Width_Sigma': width,
                                'FWHM': fwhm,
                                'Area': area,
                                'Total_R2': total_r2 if i == 0 else ''
                            })
        
        # Save CSV files
        if peak_data:
            peak_df = pd.DataFrame(peak_data)
            peak_df.to_csv(save_path / 'batch_peak_parameters.csv', index=False)
            print(f"✅ Saved peak parameters: {len(peak_data)} peaks")
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(save_path / 'batch_summary.csv', index=False)
            print(f"✅ Saved enhanced summary: {len(summary_data)} regions processed with up to {max_peaks} peaks per region")
    
    def reopen_batch_monitor(self):
        """Reopen the batch processing monitor if available."""
        if hasattr(self, 'last_batch_monitor') and self.last_batch_monitor:
            if self.last_batch_monitor.stored_results:
                # Show the existing monitor
                self.last_batch_monitor.show()
                self.last_batch_monitor.raise_()
                self.last_batch_monitor.activateWindow()
            else:
                QMessageBox.information(self, "No Results", 
                                      "No batch processing results available to view.\n\n"
                                      "Run batch processing first to view results in the monitor.")
        else:
            QMessageBox.information(self, "No Monitor", 
                                  "No batch processing monitor available.\n\n"
                                  "Run batch processing first to create a monitor.")
    
    # ==================== ADVANCED MODULE LAUNCHERS ====================
    
    def select_batch_data_file(self):
        """Select a pickle file containing batch processing results."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Batch Results File", "", 
            "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            try:
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Update label
                self.data_file_label.setText(Path(file_path).name)
                self.selected_data_file = file_path
                
                # Update preview
                preview_text = f"Loaded: {Path(file_path).name}\n"
                preview_text += f"Files: {len(data)}\n"
                preview_text += f"Total regions: {sum(len(r['regions']) for r in data)}\n"
                
                self.data_preview_text.setPlainText(preview_text)
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load pickle file: {str(e)}")
    
    def launch_deconvolution_module(self):
        """Launch the deconvolution module."""
        try:
            # Create a new instance of the deconvolution tab content
            deconv_dialog = QDialog(self)
            deconv_dialog.setWindowTitle("Spectral Deconvolution")
            deconv_dialog.setGeometry(100, 100, 800, 600)
            
            layout = QVBoxLayout(deconv_dialog)
            deconv_content = self.create_deconvolution_tab()
            layout.addWidget(deconv_content)
            
            deconv_dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch deconvolution module: {str(e)}")
    
    def launch_batch_analysis_module(self):
        """Launch the batch data analysis module."""
        try:
            # Launch the integrated batch processing (already available in this application)
            QMessageBox.information(self, "Batch Processing", 
                                  "Batch processing is available in the 'Batch' tab of this application.\n\n"
                                  "Use the Batch tab to:\n"
                                  "• Add files or folders for batch processing\n"
                                  "• Define analysis regions\n"
                                  "• Configure background correction and peak detection\n"
                                  "• Export results to pickle files for advanced analysis")
            
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch batch analysis module: {str(e)}")
    
    def launch_geothermometry_module(self):
        """Launch the geothermometry analysis module."""
        try:
            # Import and launch geothermometry
            from raman_geothermometry import launch_geothermometry_analysis
            
            # Pass selected data file if available
            data_file = getattr(self, 'selected_data_file', None)
            launch_geothermometry_analysis(data_file)
            
        except ImportError:
            QMessageBox.warning(self, "Module Not Available", 
                              "Geothermometry module is not available.\n"
                              "Please ensure raman_geothermometry.py is accessible.")
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch geothermometry module: {str(e)}")
    
    def launch_density_module(self):
        """Launch the density analysis module."""
        try:
            # Import and launch density analysis
            from advanced_analysis.density_analysis import launch_density_analysis
            
            # Pass selected data file if available
            data_file = getattr(self, 'selected_data_file', None)
            launch_density_analysis(data_file)
            
        except ImportError:
            QMessageBox.warning(self, "Module Not Available", 
                              "Density analysis module is not available.\n"
                              "Please ensure Density module is accessible.")
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch density module: {str(e)}")
    
    def launch_map_analysis_module(self):
        """Launch the 2D map analysis module."""
        try:
            # Import and launch map analysis
            from map_analysis_2d.main import launch_map_analysis
            launch_map_analysis()
            
        except ImportError:
            QMessageBox.warning(self, "Module Not Available", 
                              "2D Map Analysis module is not available.\n"
                              "Please ensure map_analysis_2d is accessible.")
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch map analysis module: {str(e)}")
    
    def launch_jupyter_console(self):
        """Launch the advanced Jupyter console with selected pickle file from data management."""
        try:
            import subprocess
            import sys
            import os
            from pathlib import Path
            
            # Check if a pickle file is selected in the data management groupbox
            selected_pickle_file = getattr(self, 'selected_data_file', None)
            
            # Try the advanced console first, then fall back to simplified version
            console_files = ['advanced_jupyter_console.py', 'simple_jupyter_console.py']
            console_launched = False
            
            for console_file in console_files:
                try:
                    if selected_pickle_file and os.path.exists(selected_pickle_file):
                        # Launch with the selected pickle file
                        subprocess.Popen([sys.executable, console_file, selected_pickle_file])
                        
                        # Get file info for the message
                        file_name = Path(selected_pickle_file).name
                        try:
                            import pickle
                            with open(selected_pickle_file, 'rb') as f:
                                data = pickle.load(f)
                            n_spectra = len(data) if isinstance(data, list) else "Unknown"
                        except:
                            n_spectra = "Unknown"
                        
                        console_type = "Advanced" if console_file.startswith('advanced') else "Simplified"
                        QMessageBox.information(self, "Console Launched", 
                                              f"{console_type} Jupyter Console launched with selected data file.\n\n"
                                              f"• File: {file_name}\n"
                                              f"• Spectra: {n_spectra}\n"
                                              f"• Data available as pandas DataFrames\n"
                                              f"• Console type: {console_type}\n\n"
                                              f"The console runs independently - you can continue using RamanLab.")
                    else:
                        # Launch without data
                        subprocess.Popen([sys.executable, console_file])
                        
                        console_type = "Advanced" if console_file.startswith('advanced') else "Simplified"
                        QMessageBox.information(self, "Console Launched", 
                                              f"{console_type} Jupyter Console launched.\n\n"
                                              "No pickle file selected in Data Management.\n"
                                              "Use 'Select Pickle File' in the Data Management section\n"
                                              "or load a file directly in the console.")
                    
                    console_launched = True
                    break
                    
                except Exception as e:
                    print(f"Failed to launch {console_file}: {e}")
                    continue
            
            if not console_launched:
                QMessageBox.critical(self, "Launch Error", 
                                   "Failed to launch any Jupyter console.\n\n"
                                   "Make sure the console files are available:\n"
                                   "• advanced_jupyter_console.py\n"
                                   "• simple_jupyter_console.py\n\n"
                                   "For full functionality, install: pip install qtconsole jupyter-client ipykernel")
                
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", 
                               f"Failed to launch Jupyter console:\n{str(e)}\n\n"
                               "Make sure the console files are available.\n"
                               "For full functionality, install: pip install qtconsole jupyter-client ipykernel")
            print(f"Jupyter console launch error: {e}")


# Launch function for integration with main app
def launch_spectral_deconvolution(parent, wavenumbers=None, intensities=None):
    """Launch the spectral deconvolution window."""
    dialog = SpectralDeconvolutionQt6(parent, wavenumbers, intensities)
    dialog.exec()

# Standalone launch function
def launch_standalone_spectral_deconvolution():
    """Launch the spectral deconvolution window as a standalone application."""
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
    
    # Create a minimal parent widget
    from PySide6.QtWidgets import QWidget
    parent = QWidget()
    parent.hide()
    
    dialog = SpectralDeconvolutionQt6(parent)
    dialog.show()
    
    if app:
        sys.exit(app.exec())

# Allow running as standalone script
if __name__ == "__main__":
    launch_standalone_spectral_deconvolution() 