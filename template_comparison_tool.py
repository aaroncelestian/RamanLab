#!/usr/bin/env python3
"""
Tool to compare extracted templates with original NMF spectra.
This helps identify preprocessing issues that might be destroying template distinguishability.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QWidget, QTextEdit, QLabel,
                              QComboBox, QCheckBox, QApplication)

class TemplateComparisonTool(QMainWindow):
    """Tool to debug template extraction and compare with NMF."""
    
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Template Extraction Debug Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Explicitly set window flags to ensure minimize/maximize/close buttons on Windows
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        self.setup_ui()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.template_combo = QComboBox()
        self.template_combo.currentTextChanged.connect(self.on_template_selected)
        controls_layout.addWidget(QLabel("Template:"))
        controls_layout.addWidget(self.template_combo)
        
        self.nmf_combo = QComboBox()
        self.nmf_combo.currentTextChanged.connect(self.on_nmf_selected)
        controls_layout.addWidget(QLabel("NMF Component:"))
        controls_layout.addWidget(self.nmf_combo)
        
        self.normalize_check = QCheckBox("Normalize for Comparison")
        self.normalize_check.setChecked(True)
        self.normalize_check.toggled.connect(self.update_plot)
        controls_layout.addWidget(self.normalize_check)
        
        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.clicked.connect(self.refresh_data)
        controls_layout.addWidget(refresh_btn)
        
        layout.addLayout(controls_layout)
        
        # Plot area
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Info area
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        layout.addWidget(self.info_text)
        
        self.refresh_data()
        
    def refresh_data(self):
        """Refresh data from main window."""
        if not self.main_window:
            return
            
        # Get templates
        self.template_combo.clear()
        if hasattr(self.main_window, 'template_manager'):
            template_names = self.main_window.template_manager.get_template_names()
            self.template_combo.addItems(template_names)
            
        # Get NMF components
        self.nmf_combo.clear()
        if hasattr(self.main_window, 'nmf_results'):
            nmf_count = len(self.main_window.nmf_results['components'])
            self.nmf_combo.addItems([f"Component {i}" for i in range(nmf_count)])
            
        self.update_info()
        self.update_plot()
        
    def on_template_selected(self):
        self.update_plot()
        
    def on_nmf_selected(self):
        self.update_plot()
        
    def update_plot(self):
        """Update the comparison plot."""
        self.figure.clear()
        
        if not self.main_window:
            return
            
        template_name = self.template_combo.currentText()
        nmf_text = self.nmf_combo.currentText()
        
        if not template_name or not nmf_text:
            return
            
        try:
            # Get template data
            template_data = None
            for template in self.main_window.template_manager.templates:
                if template.name == template_name:
                    template_data = template
                    break
                    
            if template_data is None:
                return
                
            # Get NMF component
            nmf_idx = int(nmf_text.split()[-1])
            nmf_components = self.main_window.nmf_results['components']
            
            # Create subplots
            gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Plot 1: Raw comparison
            ax1 = self.figure.add_subplot(gs[0, 0])
            self.plot_spectra_comparison(ax1, template_data, nmf_components[nmf_idx], 
                                       "Raw Comparison", normalize=False)
            
            # Plot 2: Normalized comparison
            ax2 = self.figure.add_subplot(gs[0, 1])
            self.plot_spectra_comparison(ax2, template_data, nmf_components[nmf_idx], 
                                       "Normalized Comparison", normalize=True)
            
            # Plot 3: Difference analysis
            ax3 = self.figure.add_subplot(gs[1, :])
            self.plot_difference_analysis(ax3, template_data, nmf_components[nmf_idx])
            
            self.canvas.draw()
            
        except Exception as e:
            self.info_text.append(f"Error updating plot: {e}")
            
    def plot_spectra_comparison(self, ax, template_data, nmf_component, title, normalize=False):
        """Plot template vs NMF component comparison."""
        # Get wavenumbers (assume same for both)
        if hasattr(template_data, 'wavenumbers'):
            wavenumbers = template_data.wavenumbers
        else:
            # Fallback to main window wavenumbers
            first_spectrum = next(iter(self.main_window.map_data.spectra.values()))
            wavenumbers = first_spectrum.wavenumbers
            
        # Get template intensities (try processed first)
        if hasattr(template_data, 'processed_intensities') and template_data.processed_intensities is not None:
            template_intensities = template_data.processed_intensities
            template_label = f"{template_data.name} (Processed)"
        else:
            template_intensities = template_data.intensities
            template_label = f"{template_data.name} (Raw)"
            
        nmf_intensities = nmf_component
        
        # Normalize if requested
        if normalize:
            template_intensities = template_intensities / np.max(template_intensities)
            nmf_intensities = nmf_intensities / np.max(nmf_intensities)
            
        # Plot
        ax.plot(wavenumbers, template_intensities, 'b-', label=template_label, alpha=0.8)
        ax.plot(wavenumbers, nmf_intensities, 'r-', label=f"NMF Component", alpha=0.8)
        
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Intensity' + (' (Normalized)' if normalize else ''))
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_difference_analysis(self, ax, template_data, nmf_component):
        """Plot difference between template and NMF component."""
        # Get wavenumbers
        if hasattr(template_data, 'wavenumbers'):
            wavenumbers = template_data.wavenumbers
        else:
            first_spectrum = next(iter(self.main_window.map_data.spectra.values()))
            wavenumbers = first_spectrum.wavenumbers
            
        # Get intensities
        if hasattr(template_data, 'processed_intensities') and template_data.processed_intensities is not None:
            template_intensities = template_data.processed_intensities
        else:
            template_intensities = template_data.intensities
            
        # Normalize both for fair comparison
        template_norm = template_intensities / np.max(template_intensities)
        nmf_norm = nmf_component / np.max(nmf_component)
        
        # Calculate difference
        difference = template_norm - nmf_norm
        
        ax.plot(wavenumbers, difference, 'g-', alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Difference (Template - NMF)')
        ax.set_title('Normalized Difference Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        rms_diff = np.sqrt(np.mean(difference**2))
        max_diff = np.max(np.abs(difference))
        correlation = np.corrcoef(template_norm, nmf_norm)[0, 1]
        
        stats_text = f"RMS Diff: {rms_diff:.4f}\nMax Diff: {max_diff:.4f}\nCorrelation: {correlation:.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def update_info(self):
        """Update the info text area."""
        info = []
        
        if self.main_window and hasattr(self.main_window, 'template_manager'):
            info.append(f"Templates available: {self.main_window.template_manager.get_template_count()}")
            
            # Template info
            for template in self.main_window.template_manager.templates:
                if hasattr(template, 'metadata') and 'source' in template.metadata:
                    source = template.metadata['source']
                    info.append(f"  • {template.name}: {source}")
                else:
                    info.append(f"  • {template.name}: Unknown source")
                    
        if self.main_window and hasattr(self.main_window, 'nmf_results'):
            nmf_count = len(self.main_window.nmf_results['components'])
            info.append(f"NMF components available: {nmf_count}")
            
        self.info_text.clear()
        self.info_text.append("\n".join(info))

def show_template_comparison_tool(main_window):
    """Show the template comparison tool."""
    tool = TemplateComparisonTool(main_window)
    tool.show()
    return tool

if __name__ == "__main__":
    app = QApplication([])
    tool = TemplateComparisonTool()
    tool.show()
    app.exec() 