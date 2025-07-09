#!/usr/bin/env python3
"""
RamanLab Qt6 - Basic Spectrum Viewer
Demonstrates cross-platform file operations and matplotlib integration
"""

import sys
import numpy as np

# Fix matplotlib backend for Qt6/PySide6
import matplotlib
matplotlib.use("QtAgg")  # Use QtAgg backend which works with PySide6
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import Qt6-compatible matplotlib backends and UI toolbar
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
except ImportError:
    # Fallback for older matplotlib versions
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar

from core.matplotlib_config import configure_compact_ui, apply_theme

from scipy.signal import find_peaks

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QMessageBox, QFileDialog,
    QTextEdit, QGroupBox, QSplitter
)
from PySide6.QtCore import QStandardPaths, QUrl
from PySide6.QtGui import QDesktopServices
from pathlib import Path


class SpectrumViewerQt6(QMainWindow):
    """Basic Raman spectrum viewer using Qt6."""
    
    def __init__(self):
        super().__init__()
        
        # Apply compact UI configuration for consistent toolbar sizing
        apply_theme('compact')
        
        self.setWindowTitle("RamanLab Qt6 - Spectrum Viewer")
        self.resize(1200, 700)
        
        # Data storage
        self.wavenumbers = None
        self.intensities = None
        self.peaks = None
        
        self.setup_ui()
        self.show_welcome_message()
    
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter()
        main_layout.addWidget(splitter)
        
        # Left panel - matplotlib
        self.setup_plot_panel(splitter)
        
        # Right panel - controls
        self.setup_control_panel(splitter)
        
        # Set proportions (75% plot, 25% controls)
        splitter.setSizes([900, 300])
    
    def setup_plot_panel(self, parent):
        """Set up the matplotlib plotting panel."""
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # Create matplotlib components
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)
        
        # Create the plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Raman Spectrum")
        self.ax.grid(True, alpha=0.3)
        
        # Add to layout
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        parent.addWidget(plot_widget)
    
    def setup_control_panel(self, parent):
        """Set up the control panel."""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # File operations group
        file_group = QGroupBox("File Operations (Cross-Platform)")
        file_layout = QVBoxLayout(file_group)
        
        # Import button
        import_btn = QPushButton("📁 Import Spectrum")
        import_btn.clicked.connect(self.import_spectrum)
        file_layout.addWidget(import_btn)
        
        # Save button
        save_btn = QPushButton("💾 Save Spectrum")
        save_btn.clicked.connect(self.save_spectrum)
        file_layout.addWidget(save_btn)
        
        # Open results folder button
        open_folder_btn = QPushButton("📂 Open Documents Folder")
        open_folder_btn.clicked.connect(self.open_documents_folder)
        file_layout.addWidget(open_folder_btn)
        
        control_layout.addWidget(file_group)
        
        # Analysis group
        analysis_group = QGroupBox("Basic Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Find peaks button
        peaks_btn = QPushButton("🔍 Find Peaks")
        peaks_btn.clicked.connect(self.find_peaks)
        analysis_layout.addWidget(peaks_btn)
        
        # Clear button
        clear_btn = QPushButton("🧹 Clear Plot")
        clear_btn.clicked.connect(self.clear_plot)
        analysis_layout.addWidget(clear_btn)
        
        control_layout.addWidget(analysis_group)
        
        # Info display
        info_group = QGroupBox("Spectrum Info")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setPlainText("No spectrum loaded")
        info_layout.addWidget(self.info_text)
        
        control_layout.addWidget(info_group)
        
        # Status display
        status_group = QGroupBox("Cross-Platform Status")
        status_layout = QVBoxLayout(status_group)
        
        status_label = QLabel("✅ No platform checks needed!\n✅ Works on macOS, Windows, Linux\n✅ Native file dialogs\n✅ One codebase")
        status_label.setWordWrap(True)
        status_layout.addWidget(status_label)
        
        control_layout.addWidget(status_group)
        
        # Add stretch to push everything to top
        control_layout.addStretch()
        
        parent.addWidget(control_widget)
    
    def show_welcome_message(self):
        """Show welcome message on the plot."""
        self.ax.text(0.5, 0.5, 
                     'RamanLab Qt6 Spectrum Viewer\n\n'
                     '🎯 Cross-platform file operations\n'
                     '🚀 No more platform-specific code\n'
                     '📊 Matplotlib integration\n\n'
                     'Click "Import Spectrum" to get started!',
                     ha='center', va='center', fontsize=12,
                     transform=self.ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        self.canvas.draw()
    
    def import_spectrum(self):
        """Import a spectrum using cross-platform file dialog."""
        # This is the cross-platform magic - no platform checks needed!
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Raman Spectrum",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt *.csv *.dat);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Try to load the data
                data = np.loadtxt(file_path)
                
                if data.ndim == 1:
                    # Single column - assume it's intensities with index as wavenumber
                    self.wavenumbers = np.arange(len(data))
                    self.intensities = data
                elif data.ndim == 2 and data.shape[1] >= 2:
                    # Two columns - wavenumber and intensity
                    self.wavenumbers = data[:, 0]
                    self.intensities = data[:, 1]
                else:
                    raise ValueError("Invalid data format")
                
                self.plot_spectrum()
                self.update_info(file_path)
                
                # Show success message highlighting cross-platform nature
                QMessageBox.information(
                    self, 
                    "Success!", 
                    f"Spectrum loaded from:\n{file_path}\n\n"
                    "🎯 This file dialog works identically on:\n"
                    "• macOS (your current platform)\n"
                    "• Windows\n"
                    "• Linux\n\n"
                    "No platform-specific code needed!"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Import Error", 
                    f"Failed to load spectrum:\n{str(e)}\n\n"
                    "Try a text file with:\n"
                    "• Two columns: wavenumber intensity\n"
                    "• Tab or space separated"
                )
    
    def save_spectrum(self):
        """Save the current spectrum using cross-platform dialog."""
        if self.wavenumbers is None or self.intensities is None:
            QMessageBox.warning(self, "No Data", "No spectrum to save.")
            return
        
        # Cross-platform save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spectrum",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Prepare data
                data = np.column_stack([self.wavenumbers, self.intensities])
                np.savetxt(file_path, data, delimiter='\t', 
                          header='Wavenumber\tIntensity', comments='')
                
                QMessageBox.information(
                    self, 
                    "Success!", 
                    f"Spectrum saved to:\n{file_path}\n\n"
                    "🎯 Cross-platform file saving complete!"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save spectrum:\n{str(e)}")
    
    def open_documents_folder(self):
        """Open the documents folder - demonstrates cross-platform folder opening."""
        docs_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        success = QDesktopServices.openUrl(QUrl.fromLocalFile(docs_dir))
        
        if success:
            QMessageBox.information(
                self, 
                "Folder Opened!", 
                f"Opened: {docs_dir}\n\n"
                "🔥 This ONE LINE replaces ALL of this code:\n\n"
                "OLD WAY (your current pain):\n"
                "if platform.system() == 'Windows':\n"
                "    os.startfile(docs_dir)\n"
                "elif platform.system() == 'Darwin':\n"
                "    subprocess.run(['open', docs_dir])\n"
                "else:\n"
                "    subprocess.run(['xdg-open', docs_dir])\n\n"
                "NEW WAY (Qt6):\n"
                "QDesktopServices.openUrl(QUrl.fromLocalFile(docs_dir))\n\n"
                "🚀 Works everywhere!"
            )
        else:
            QMessageBox.warning(self, "Error", "Could not open folder.")
    
    def plot_spectrum(self):
        """Plot the current spectrum."""
        self.ax.clear()
        
        if self.wavenumbers is not None and self.intensities is not None:
            # Plot spectrum
            self.ax.plot(self.wavenumbers, self.intensities, 'b-', linewidth=1)
            
            # Plot peaks if found
            if self.peaks is not None:
                peak_wavenumbers = self.wavenumbers[self.peaks]
                peak_intensities = self.intensities[self.peaks]
                self.ax.plot(peak_wavenumbers, peak_intensities, 'ro', markersize=6)
            
            self.ax.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax.set_ylabel("Intensity (a.u.)")
            self.ax.set_title("Raman Spectrum")
            self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def find_peaks(self):
        """Find peaks in the spectrum."""
        if self.intensities is None:
            QMessageBox.warning(self, "No Data", "Load a spectrum first.")
            return
        
        # Simple peak finding
        self.peaks, _ = find_peaks(self.intensities, height=0.1*np.max(self.intensities), distance=10)
        
        self.plot_spectrum()
        
        QMessageBox.information(
            self, 
            "Peaks Found", 
            f"Found {len(self.peaks)} peaks\n\n"
            "Red dots show peak positions."
        )
    
    def clear_plot(self):
        """Clear the plot."""
        self.wavenumbers = None
        self.intensities = None
        self.peaks = None
        self.ax.clear()
        self.show_welcome_message()
        self.info_text.setPlainText("No spectrum loaded")
    
    def update_info(self, file_path):
        """Update the info display."""
        info = f"File: {Path(file_path).name}\n"
        info += f"Data points: {len(self.wavenumbers)}\n"
        info += f"Wavenumber range: {self.wavenumbers.min():.1f} - {self.wavenumbers.max():.1f} cm⁻¹\n"
        info += f"Intensity range: {self.intensities.min():.2e} - {self.intensities.max():.2e}"
        
        if self.peaks is not None:
            info += f"\nPeaks found: {len(self.peaks)}"
        
        self.info_text.setPlainText(info)


def main():
    """Run the spectrum viewer."""
    app = QApplication(sys.argv)
    
    viewer = SpectrumViewerQt6()
    viewer.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())