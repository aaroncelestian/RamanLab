#!/usr/bin/env python3
"""
GUI Launcher for Raman Density Analysis
Provides a simple interface for running density analysis on Raman spectra.
"""

import os
import sys
import numpy as np

# Fix Qt platform plugin issues on macOS
if sys.platform == "darwin":  # macOS
    os.environ.setdefault('QT_MAC_WANTS_LAYER', '1')
    # Clear any problematic Qt plugin paths
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
        del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Import RamanLab matplotlib configuration
import sys
from pathlib import Path
# Add parent directory to path to access polarization_ui
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar, apply_theme

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
    QPushButton, QTextEdit, QFileDialog, QMessageBox, QGroupBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QTabWidget, QSplitter, QDialog,
    QDialogButtonBox, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# Import the density analyzer
from raman_density_analysis import RamanDensityAnalyzer, MaterialConfigs

class CustomMaterialDialog(QDialog):
    """Dialog for configuring custom material properties."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Material Configuration")
        self.setMinimumSize(500, 600)
        self.custom_config = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Scroll area for the form
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Material name
        name_group = QGroupBox("Material Information")
        name_layout = QFormLayout(name_group)
        
        self.material_name_edit = QTextEdit()
        self.material_name_edit.setMaximumHeight(30)
        self.material_name_edit.setPlainText("My Custom Material")
        name_layout.addRow("Material Name:", self.material_name_edit)
        
        scroll_layout.addWidget(name_group)
        
        # Characteristic peaks
        peaks_group = QGroupBox("Characteristic Peaks (cm⁻¹)")
        peaks_layout = QFormLayout(peaks_group)
        
        self.main_peak_spin = QDoubleSpinBox()
        self.main_peak_spin.setRange(100, 4000)
        self.main_peak_spin.setValue(1000)
        self.main_peak_spin.setSuffix(" cm⁻¹")

        peaks_layout.addRow("Main Peak:", self.main_peak_spin)
        
        self.secondary_peak_spin = QDoubleSpinBox()
        self.secondary_peak_spin.setRange(100, 4000)
        self.secondary_peak_spin.setValue(500)
        self.secondary_peak_spin.setSuffix(" cm⁻¹")
        peaks_layout.addRow("Secondary Peak:", self.secondary_peak_spin)
        
        self.tertiary_peak_spin = QDoubleSpinBox()
        self.tertiary_peak_spin.setRange(100, 4000)
        self.tertiary_peak_spin.setValue(1500)
        self.tertiary_peak_spin.setSuffix(" cm⁻¹")
        peaks_layout.addRow("Tertiary Peak:", self.tertiary_peak_spin)
        
        scroll_layout.addWidget(peaks_group)
        
        # Reference regions
        regions_group = QGroupBox("Reference Regions (cm⁻¹)")
        regions_layout = QFormLayout(regions_group)
        
        # Baseline region
        baseline_layout = QHBoxLayout()
        self.baseline_start = QDoubleSpinBox()
        self.baseline_start.setRange(100, 4000)
        self.baseline_start.setValue(200)
        self.baseline_start.setSuffix(" cm⁻¹")
        self.baseline_end = QDoubleSpinBox()
        self.baseline_end.setRange(100, 4000)
        self.baseline_end.setValue(400)
        self.baseline_end.setSuffix(" cm⁻¹")
        baseline_layout.addWidget(self.baseline_start)
        baseline_layout.addWidget(QLabel("to"))
        baseline_layout.addWidget(self.baseline_end)
        regions_layout.addRow("Baseline Region:", baseline_layout)
        
        # Fingerprint region
        fingerprint_layout = QHBoxLayout()
        self.fingerprint_start = QDoubleSpinBox()
        self.fingerprint_start.setRange(100, 4000)
        self.fingerprint_start.setValue(400)
        self.fingerprint_start.setSuffix(" cm⁻¹")
        self.fingerprint_end = QDoubleSpinBox()
        self.fingerprint_end.setRange(100, 4000)
        self.fingerprint_end.setValue(800)
        self.fingerprint_end.setSuffix(" cm⁻¹")
        fingerprint_layout.addWidget(self.fingerprint_start)
        fingerprint_layout.addWidget(QLabel("to"))
        fingerprint_layout.addWidget(self.fingerprint_end)
        regions_layout.addRow("Fingerprint Region:", fingerprint_layout)
        
        # High frequency region
        high_freq_layout = QHBoxLayout()
        self.high_freq_start = QDoubleSpinBox()
        self.high_freq_start.setRange(100, 4000)
        self.high_freq_start.setValue(800)
        self.high_freq_start.setSuffix(" cm⁻¹")
        self.high_freq_end = QDoubleSpinBox()
        self.high_freq_end.setRange(100, 4000)
        self.high_freq_end.setValue(1200)
        self.high_freq_end.setSuffix(" cm⁻¹")
        high_freq_layout.addWidget(self.high_freq_start)
        high_freq_layout.addWidget(QLabel("to"))
        high_freq_layout.addWidget(self.high_freq_end)
        regions_layout.addRow("High Freq Region:", high_freq_layout)
        
        scroll_layout.addWidget(regions_group)
        
        # Densities
        densities_group = QGroupBox("Material Densities (g/cm³)")
        densities_layout = QFormLayout(densities_group)
        
        self.crystalline_density = QDoubleSpinBox()
        self.crystalline_density.setRange(0.5, 10.0)
        self.crystalline_density.setValue(2.5)
        self.crystalline_density.setDecimals(2)
        self.crystalline_density.setSuffix(" g/cm³")
        densities_layout.addRow("Crystalline Density:", self.crystalline_density)
        
        self.matrix_density = QDoubleSpinBox()
        self.matrix_density.setRange(0.5, 10.0)
        self.matrix_density.setValue(2.0)
        self.matrix_density.setDecimals(2)
        self.matrix_density.setSuffix(" g/cm³")
        densities_layout.addRow("Matrix Density:", self.matrix_density)
        
        self.low_density = QDoubleSpinBox()
        self.low_density.setRange(0.5, 10.0)
        self.low_density.setValue(1.5)
        self.low_density.setDecimals(2)
        self.low_density.setSuffix(" g/cm³")
        densities_layout.addRow("Low Density:", self.low_density)
        
        scroll_layout.addWidget(densities_group)
        
        # Classification thresholds
        thresholds_group = QGroupBox("Classification Thresholds (0-1)")
        thresholds_layout = QFormLayout(thresholds_group)
        
        self.low_threshold = QDoubleSpinBox()
        self.low_threshold.setRange(0.0, 1.0)
        self.low_threshold.setValue(0.3)
        self.low_threshold.setDecimals(2)
        self.low_threshold.setSingleStep(0.05)
        thresholds_layout.addRow("Low Threshold:", self.low_threshold)
        
        self.medium_threshold = QDoubleSpinBox()
        self.medium_threshold.setRange(0.0, 1.0)
        self.medium_threshold.setValue(0.6)
        self.medium_threshold.setDecimals(2)
        self.medium_threshold.setSingleStep(0.05)
        thresholds_layout.addRow("Medium Threshold:", self.medium_threshold)
        
        self.high_threshold = QDoubleSpinBox()
        self.high_threshold.setRange(0.0, 1.0)
        self.high_threshold.setValue(0.85)
        self.high_threshold.setDecimals(2)
        self.high_threshold.setSingleStep(0.05)
        thresholds_layout.addRow("High Threshold:", self.high_threshold)
        
        scroll_layout.addWidget(thresholds_group)
        
        # Reference intensity
        ref_group = QGroupBox("Analysis Parameters")
        ref_layout = QFormLayout(ref_group)
        
        ref_intensity_layout = QHBoxLayout()
        self.reference_intensity = QDoubleSpinBox()
        self.reference_intensity.setRange(100, 10000)
        self.reference_intensity.setValue(800)
        self.reference_intensity.setDecimals(0)
        ref_intensity_layout.addWidget(self.reference_intensity)
        
        # Auto-update button from batch fitting results
        auto_ref_btn = QPushButton("From Batch")
        auto_ref_btn.clicked.connect(self.update_reference_from_batch)
        auto_ref_btn.setMaximumWidth(80)
        auto_ref_btn.setToolTip("Auto-calculate from batch peak fitting results")
        auto_ref_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 4px;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        ref_intensity_layout.addWidget(auto_ref_btn)
        
        ref_layout.addRow("Reference Intensity:", ref_intensity_layout)
        
        scroll_layout.addWidget(ref_group)
        
        # Set up scroll area
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Dialog buttons with help
        button_layout = QHBoxLayout()
        
        # Help button
        help_btn = QPushButton("? Help")
        help_btn.clicked.connect(self.show_help)
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        button_layout.addWidget(help_btn)
        
        button_layout.addStretch()
        
        # Standard dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        button_layout.addWidget(buttons)
        
        layout.addWidget(QWidget())  # Add some spacing
        layout.addLayout(button_layout)
        
    def get_config(self):
        """Get the custom configuration from the dialog."""
        if not self.custom_config:
            self.create_config()
        return self.custom_config
        
    def create_config(self):
        """Create configuration from dialog inputs."""
        self.custom_config = {
            'name': self.material_name_edit.toPlainText().strip() or "Custom Material",
            'characteristic_peaks': {
                'main': self.main_peak_spin.value(),
                'secondary': self.secondary_peak_spin.value(),
                'tertiary': self.tertiary_peak_spin.value()
            },
            'reference_regions': {
                'baseline': (self.baseline_start.value(), self.baseline_end.value()),
                'fingerprint': (self.fingerprint_start.value(), self.fingerprint_end.value()),
                'high_freq': (self.high_freq_start.value(), self.high_freq_end.value())
            },
            'densities': {
                'crystalline': self.crystalline_density.value(),
                'matrix': self.matrix_density.value(),
                'low_density': self.low_density.value()
            },
            'density_ranges': {
                'low_range': (self.low_density.value() - 0.1, self.low_density.value() + 0.1),
                'medium_range': (self.matrix_density.value() - 0.2, self.matrix_density.value() + 0.2),
                'mixed_range': (self.matrix_density.value(), self.crystalline_density.value() - 0.2),
                'crystalline_range': (self.crystalline_density.value() - 0.1, self.crystalline_density.value() + 0.1)
            },
            'reference_intensity': self.reference_intensity.value(),
            'classification_thresholds': {
                'low': self.low_threshold.value(),
                'medium': self.medium_threshold.value(),
                'high': self.high_threshold.value()
            }
        }
        
    def show_help(self):
        """Show help dialog for custom material configuration."""
        # Create custom dialog for better control over size and scrolling
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Custom Material Configuration Help")
        help_dialog.setFixedSize(900, 400)  # 50% wider (600 -> 900), much shorter (600 -> 400)
        
        layout = QVBoxLayout(help_dialog)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        help_text = """
<h3>🔬 Custom Material Configuration Guide</h3>

<p><b>📋 Parameter Guide:</b></p>
<ul>
<li><b>Material Name:</b> Give your material a descriptive name (e.g., "Olivine basalt", "Sandstone")</li>
<li><b>Characteristic Peaks:</b> Main Raman peak positions for your material (cm⁻¹)
  <ul>
    <li><i>Main Peak:</i> Your strongest, most characteristic peak</li>
    <li><i>Secondary/Tertiary:</i> Additional identifying peaks</li>
  </ul>
</li>
<li><b>Reference Regions:</b> Spectral regions used for baseline and analysis
  <ul>
    <li><i>Baseline:</i> Low-intensity region for background subtraction</li>
    <li><i>Fingerprint:</i> Region containing characteristic features</li>
    <li><i>High Frequency:</i> Higher wavenumber analysis region</li>
  </ul>
</li>
<li><b>Material Densities:</b> Known density values from literature or measurements (g/cm³)
  <ul>
    <li><i>Crystalline:</i> Pure crystalline phase density</li>
    <li><i>Matrix:</i> Mixed/polycrystalline density</li>
    <li><i>Low Density:</i> Amorphous or low-density phase</li>
  </ul>
</li>
<li><b>Classification Thresholds:</b> CDI values that separate crystallinity levels (0-1 scale)
  <ul>
    <li><i>Low:</i> Below this = low crystallinity</li>
    <li><i>Medium:</i> Between low and medium = mixed regions</li>
    <li><i>High:</i> Above this = pure crystalline</li>
  </ul>
</li>
<li><b>Reference Intensity:</b> Typical peak intensity for CDI normalization
  <ul>
    <li><i>Manual:</i> Enter a value from literature or experience</li>
    <li><i>From Batch:</i> Auto-calculate from batch peak fitting results (recommended)</li>
  </ul>
</li>
</ul>

<p><b>💡 Tips for Success:</b></p>
<ul>
<li><b>Find density values:</b> Check mineral databases (mindat.org, webmineral.com)</li>
<li><b>Peak selection:</b> Use your strongest, most reproducible Raman peak as main peak</li>
<li><b>Start simple:</b> Begin with a similar material's settings and adjust</li>
<li><b>Reference regions:</b> Choose regions with minimal overlapping peaks</li>
<li><b>Thresholds:</b> Start with default values and refine based on your data</li>
</ul>

<p><b>🎯 Example Values (Quartz-like mineral):</b></p>
<ul>
<li>Main Peak: ~464 cm⁻¹</li>
<li>Crystalline Density: ~2.65 g/cm³</li>
<li>Matrix Density: ~2.2 g/cm³</li>
<li>Low Density: ~1.8 g/cm³</li>
<li>Reference Intensity: ~800</li>
</ul>

<p><b>🔧 Common Applications:</b></p>
<ul>
<li>Novel mineral phases or synthetic materials</li>
<li>Fine-tuning analysis for specific sample types</li>
<li>Research on crystalline-amorphous transitions</li>
<li>Custom calibrations for specialized studies</li>
</ul>
"""
        
        # Create label with help text
        help_label = QLabel(help_text)
        help_label.setTextFormat(Qt.RichText)
        help_label.setWordWrap(True)
        help_label.setAlignment(Qt.AlignTop)
        help_label.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
        """)
        
        content_layout.addWidget(help_label)
        content_layout.addStretch()
        
        # Set the content widget to the scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        layout.addWidget(scroll_area)
        
        # Add close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(help_dialog.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        help_dialog.exec()
        
    def update_reference_from_batch(self):
        """Update reference intensity from batch peak fitting results."""
        try:
            main_peak_pos = self.main_peak_spin.value()
            
            # Try to find batch fitting results - check common locations
            batch_results = self._find_batch_results()
            
            if not batch_results:
                QMessageBox.information(self, "No Batch Data", 
                    "No batch peak fitting results found.\n\n"
                    "To use this feature:\n"
                    "1. Run batch peak fitting first\n"
                    "2. Ensure peaks are fitted near your main peak position\n"
                    "3. Return here to auto-populate reference intensity")
                return
            
            # Extract intensities for peaks near the main peak position
            reference_intensities = self._extract_peak_intensities(batch_results, main_peak_pos)
            
            if not reference_intensities:
                QMessageBox.warning(self, "No Matching Peaks", 
                    f"No fitted peaks found near {main_peak_pos:.0f} cm⁻¹ in batch results.\n\n"
                    f"Try:\n"
                    f"• Adjusting the main peak position\n"
                    f"• Running batch fitting with peak detection around {main_peak_pos:.0f} cm⁻¹")
                return
            
            # Calculate representative reference intensity
            import numpy as np
            median_intensity = np.median(reference_intensities)
            mean_intensity = np.mean(reference_intensities)
            
            # Use median as it's more robust to outliers
            recommended_ref = int(median_intensity)
            
            # Update the reference intensity
            self.reference_intensity.setValue(recommended_ref)
            
            # Show success message with statistics
            QMessageBox.information(self, "Reference Intensity Updated", 
                f"Reference intensity updated from batch fitting results:\n\n"
                f"• Peaks analyzed: {len(reference_intensities)}\n"
                f"• Peak position: {main_peak_pos:.0f} cm⁻¹ (±20 cm⁻¹)\n"
                f"• Median intensity: {recommended_ref}\n"
                f"• Mean intensity: {mean_intensity:.0f}\n"
                f"• Range: {min(reference_intensities):.0f} - {max(reference_intensities):.0f}\n\n"
                f"Using median value ({recommended_ref}) as reference intensity.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                f"Failed to update reference intensity from batch results:\n{str(e)}")
    
    def _find_batch_results(self):
        """Find available batch peak fitting results."""
        # Check if parent GUI has batch results
        parent_gui = self.parent()
        while parent_gui and not isinstance(parent_gui, DensityAnalysisGUI):
            parent_gui = parent_gui.parent()
            
        if hasattr(parent_gui, 'batch_fitting_results'):
            return parent_gui.batch_fitting_results
            
        # Check for result files in common locations
        import os
        import glob
        
        possible_paths = [
            "../batch_fitting_results.pkl",
            "../results/*.csv",
            "batch_fitting_results.pkl",
            "results/*.csv"
        ]
        
        for pattern in possible_paths:
            files = glob.glob(pattern)
            if files:
                return self._load_results_from_file(files[0])
                
        return None
    
    def _load_results_from_file(self, file_path):
        """Load batch fitting results from file."""
        try:
            import pandas as pd
            import pickle
            
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path)
        except Exception:
            pass
        return None
    
    def _extract_peak_intensities(self, batch_results, target_peak, tolerance=20):
        """Extract peak intensities near the target peak position."""
        intensities = []
        
        # Handle different result formats
        if isinstance(batch_results, dict):
            # If results are in dictionary format
            for file_result in batch_results.values():
                if 'fitted_peaks' in file_result:
                    peaks = file_result['fitted_peaks']
                    for peak in peaks:
                        if abs(peak.get('center', 0) - target_peak) <= tolerance:
                            intensities.append(peak.get('amplitude', 0))
                            
        elif hasattr(batch_results, 'iterrows'):
            # If results are in pandas DataFrame format
            for _, row in batch_results.iterrows():
                if 'peak_center' in row and 'peak_amplitude' in row:
                    if abs(row['peak_center'] - target_peak) <= tolerance:
                        intensities.append(row['peak_amplitude'])
                        
        elif isinstance(batch_results, list):
            # If results are in list format
            for result in batch_results:
                if isinstance(result, dict) and 'peaks' in result:
                    for peak in result['peaks']:
                        if abs(peak.get('position', 0) - target_peak) <= tolerance:
                            intensities.append(peak.get('intensity', 0))
        
        # Filter out zero or negative intensities
        intensities = [i for i in intensities if i > 0]
        
        return intensities
        
    def accept(self):
        """Handle dialog acceptance."""
        self.create_config()
        super().accept()

class DensityAnalysisGUI(QMainWindow):
    """GUI for Raman Density Analysis"""
    
    def __init__(self):
        super().__init__()
        
        # Apply RamanLab matplotlib configuration
        apply_theme('compact')
        
        self.analyzer = RamanDensityAnalyzer()
        self.current_spectrum = None
        self.wavenumbers = None
        self.intensities = None
        self.processed_data = None
        self.custom_config = None
        self.batch_fitting_results = None  # Store batch fitting results for reference intensity calculation
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Raman Density Analysis Tool")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - plots
        right_panel = self.create_plot_panel()
        splitter.addWidget(right_panel)
        
        # Set proportions
        splitter.setSizes([300, 900])
        
    def create_control_panel(self):
        """Create the control panel with tabs."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different analysis modes
        self.analysis_tabs = QTabWidget()
        
        # Standard Analysis Tab
        standard_tab = self.create_standard_analysis_tab()
        self.analysis_tabs.addTab(standard_tab, "Standard Analysis")
        
        # Trend Analysis Tab
        trend_tab = self.create_trend_analysis_tab()
        self.analysis_tabs.addTab(trend_tab, "Trend Analysis")
        
        layout.addWidget(self.analysis_tabs)
        
        return panel
    
    def create_standard_analysis_tab(self):
        """Create the standard analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Batch data section (initially hidden)
        self.batch_group = QGroupBox("Batch Data from Peak Fitting")
        batch_layout = QVBoxLayout(self.batch_group)
        
        self.batch_status_label = QLabel("No batch data loaded")
        self.batch_status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        batch_layout.addWidget(self.batch_status_label)
        
        # Spectrum selector for batch data
        spectrum_selector_layout = QHBoxLayout()
        spectrum_selector_layout.addWidget(QLabel("Spectrum:"))
        self.batch_spectrum_combo = QComboBox()
        self.batch_spectrum_combo.currentTextChanged.connect(self.on_batch_spectrum_selected)
        spectrum_selector_layout.addWidget(self.batch_spectrum_combo)
        batch_layout.addLayout(spectrum_selector_layout)
        
        # Batch analysis buttons
        batch_buttons_layout = QHBoxLayout()
        self.load_batch_spectrum_btn = QPushButton("Load Selected")
        self.load_batch_spectrum_btn.clicked.connect(self.load_selected_batch_spectrum)
        self.load_batch_spectrum_btn.setEnabled(False)
        batch_buttons_layout.addWidget(self.load_batch_spectrum_btn)
        
        self.analyze_all_batch_btn = QPushButton("Analyze All")
        self.analyze_all_batch_btn.clicked.connect(self.analyze_all_batch_spectra)
        self.analyze_all_batch_btn.setEnabled(False)
        batch_buttons_layout.addWidget(self.analyze_all_batch_btn)
        batch_layout.addLayout(batch_buttons_layout)
        
        # Export buttons
        export_layout = QHBoxLayout()
        self.export_batch_btn = QPushButton("Export Results")
        self.export_batch_btn.clicked.connect(self.export_batch_density_results)
        self.export_batch_btn.setEnabled(False)
        export_layout.addWidget(self.export_batch_btn)
        
        self.save_plots_btn = QPushButton("Save Plots")
        self.save_plots_btn.clicked.connect(self.save_density_plots)
        self.save_plots_btn.setEnabled(False)
        export_layout.addWidget(self.save_plots_btn)
        batch_layout.addLayout(export_layout)
        
        self.batch_group.setVisible(False)  # Initially hidden
        layout.addWidget(self.batch_group)
        
        # File loading
        file_group = QGroupBox("Data Input")
        file_layout = QVBoxLayout(file_group)
        
        load_btn = QPushButton("Load Spectrum File")
        load_btn.clicked.connect(self.load_spectrum_file)
        file_layout.addWidget(load_btn)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        layout.addWidget(file_group)
        
        # Analysis parameters
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QFormLayout(params_group)
        
        # Material type selection
        material_layout = QHBoxLayout()
        self.material_combo = QComboBox()
        self.material_combo.addItems(MaterialConfigs.get_available_materials())
        self.material_combo.currentTextChanged.connect(self.on_material_changed)
        material_layout.addWidget(self.material_combo)
        
        # Custom material configuration button
        self.config_custom_btn = QPushButton("Configure")
        self.config_custom_btn.clicked.connect(self.configure_custom_material)
        self.config_custom_btn.setEnabled(False)
        self.config_custom_btn.setMaximumWidth(80)
        self.config_custom_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        material_layout.addWidget(self.config_custom_btn)
        
        params_layout.addRow("Material Type:", material_layout)
        
        # Density type selection (formerly biofilm type)
        self.density_type_combo = QComboBox()
        self.update_density_type_options()
        params_layout.addRow("Density Type:", self.density_type_combo)
        
        layout.addWidget(params_group)
        
        # Analysis buttons
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        analyze_btn = QPushButton("Analyze Spectrum")
        analyze_btn.clicked.connect(self.analyze_spectrum)
        analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        analysis_layout.addWidget(analyze_btn)
        
        layout.addWidget(analysis_group)
        
        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
        return tab
    
    def create_trend_analysis_tab(self):
        """Create the trend analysis tab for sequential analysis."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Trend analysis parameters
        trend_params_group = QGroupBox("Trend Analysis Parameters")
        trend_params_layout = QFormLayout(trend_params_group)
        
        # Sequence ordering
        self.sequence_order_combo = QComboBox()
        self.sequence_order_combo.addItems([
            "Filename (alphabetical)",
            "Filename (numerical)",
            "Timestamp (if available)",
            "Manual ordering"
        ])
        trend_params_layout.addRow("Sequence Order:", self.sequence_order_combo)
        
        # Moving average window
        self.moving_avg_window = QSpinBox()
        self.moving_avg_window.setRange(1, 20)
        self.moving_avg_window.setValue(3)
        self.moving_avg_window.setSuffix(" points")
        trend_params_layout.addRow("Smoothing Window:", self.moving_avg_window)
        
        # Confidence interval level
        self.confidence_level = QDoubleSpinBox()
        self.confidence_level.setRange(0.50, 0.99)
        self.confidence_level.setValue(0.95)
        self.confidence_level.setDecimals(2)
        self.confidence_level.setSingleStep(0.05)
        trend_params_layout.addRow("Confidence Level:", self.confidence_level)
        
        # Trend detection sensitivity
        self.trend_sensitivity = QDoubleSpinBox()
        self.trend_sensitivity.setRange(0.01, 0.50)
        self.trend_sensitivity.setValue(0.05)
        self.trend_sensitivity.setDecimals(3)
        self.trend_sensitivity.setSuffix(" (slope threshold)")
        trend_params_layout.addRow("Trend Sensitivity:", self.trend_sensitivity)
        
        layout.addWidget(trend_params_group)
        
        # Trend analysis options
        trend_options_group = QGroupBox("Analysis Options")
        trend_options_layout = QVBoxLayout(trend_options_group)
        
        # Parameter selection checkboxes
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Analyze trends in:"))
        trend_options_layout.addLayout(param_layout)
        
        checkbox_layout = QVBoxLayout()
        self.analyze_cdi_checkbox = QPushButton("✓ CDI (Crystalline Density Index)")
        self.analyze_cdi_checkbox.setCheckable(True)
        self.analyze_cdi_checkbox.setChecked(True)
        self.analyze_cdi_checkbox.setStyleSheet(self.get_checkbox_style())
        checkbox_layout.addWidget(self.analyze_cdi_checkbox)
        
        self.analyze_density_checkbox = QPushButton("✓ Specialized Density")
        self.analyze_density_checkbox.setCheckable(True)
        self.analyze_density_checkbox.setChecked(True)
        self.analyze_density_checkbox.setStyleSheet(self.get_checkbox_style())
        checkbox_layout.addWidget(self.analyze_density_checkbox)
        
        self.analyze_peak_height_checkbox = QPushButton("✓ Peak Height")
        self.analyze_peak_height_checkbox.setCheckable(True)
        self.analyze_peak_height_checkbox.setChecked(False)
        self.analyze_peak_height_checkbox.setStyleSheet(self.get_checkbox_style())
        checkbox_layout.addWidget(self.analyze_peak_height_checkbox)
        
        self.analyze_peak_width_checkbox = QPushButton("✓ Peak Width")
        self.analyze_peak_width_checkbox.setCheckable(True)
        self.analyze_peak_width_checkbox.setChecked(False)
        self.analyze_peak_width_checkbox.setStyleSheet(self.get_checkbox_style())
        checkbox_layout.addWidget(self.analyze_peak_width_checkbox)
        
        trend_options_layout.addLayout(checkbox_layout)
        
        layout.addWidget(trend_options_group)
        
        # Trend analysis controls
        trend_controls_group = QGroupBox("Trend Analysis")
        trend_controls_layout = QVBoxLayout(trend_controls_group)
        
        self.analyze_trends_btn = QPushButton("Analyze Trends")
        self.analyze_trends_btn.clicked.connect(self.analyze_trends)
        self.analyze_trends_btn.setEnabled(False)
        self.analyze_trends_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        trend_controls_layout.addWidget(self.analyze_trends_btn)
        
        # Export trend results
        export_trend_layout = QHBoxLayout()
        self.export_trends_btn = QPushButton("Export Trends")
        self.export_trends_btn.clicked.connect(self.export_trend_analysis)
        self.export_trends_btn.setEnabled(False)
        export_trend_layout.addWidget(self.export_trends_btn)
        
        self.save_trend_plots_btn = QPushButton("Save Plots")
        self.save_trend_plots_btn.clicked.connect(self.save_trend_plots)
        self.save_trend_plots_btn.setEnabled(False)
        export_trend_layout.addWidget(self.save_trend_plots_btn)
        
        trend_controls_layout.addLayout(export_trend_layout)
        
        layout.addWidget(trend_controls_group)
        
        # Trend results display
        trend_results_group = QGroupBox("Trend Analysis Results")
        trend_results_layout = QVBoxLayout(trend_results_group)
        
        self.trend_results_text = QTextEdit()
        self.trend_results_text.setMaximumHeight(200)
        self.trend_results_text.setReadOnly(True)
        self.trend_results_text.setText("📈 TREND ANALYSIS\n" + "="*30 + "\n\n"
                                       "Welcome to the Sequential Trend Analysis module!\n\n"
                                       "This tool analyzes how density and crystallinity change across your scan sequence with statistical confidence intervals.\n\n"
                                       "📋 Getting Started:\n"
                                       "1. Load batch data from the 'Standard Analysis' tab\n"
                                       "2. Run batch density analysis first\n"
                                       "3. Return here to analyze trends\n\n"
                                       "🔍 Features:\n"
                                       "• Linear regression with confidence intervals\n"
                                       "• Multiple sequence ordering options\n"
                                       "• Smoothing and trend detection\n"
                                       "• Statistical significance testing\n"
                                       "• Comprehensive visualization\n\n"
                                       "Ready to discover patterns in your data!")
        trend_results_layout.addWidget(self.trend_results_text)
        
        layout.addWidget(trend_results_group)
        
        layout.addStretch()
        return tab
    
    def get_checkbox_style(self):
        """Get consistent checkbox button styling."""
        return """
            QPushButton {
                background-color: #E3F2FD;
                color: #1976D2;
                border: 2px solid #1976D2;
                padding: 5px;
                border-radius: 3px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #1976D2;
                color: white;
            }
            QPushButton:hover {
                background-color: #BBDEFB;
            }
            QPushButton:checked:hover {
                background-color: #1565C0;
            }
        """
    
    def on_material_changed(self):
        """Handle material type change."""
        material_type = self.material_combo.currentText()
        
        # Enable/disable custom configuration button
        self.config_custom_btn.setEnabled(material_type == 'Other (Custom)')
        
        # Create analyzer with custom config if needed
        if material_type == 'Other (Custom)' and self.custom_config:
            self.analyzer = RamanDensityAnalyzer(material_type, self.custom_config)
        elif material_type == 'Other (Custom)':
            # Use default template if no custom config yet
            self.custom_config = MaterialConfigs.get_custom_template()
            self.analyzer = RamanDensityAnalyzer(material_type, self.custom_config)
        else:
            self.analyzer = RamanDensityAnalyzer(material_type)
            self.custom_config = None
            
        self.update_density_type_options()
        self.results_text.clear()
    
    def configure_custom_material(self):
        """Open dialog to configure custom material properties."""
        dialog = CustomMaterialDialog(self)
        
        # Load existing custom config if available
        if self.custom_config:
            # Pre-populate dialog with existing config
            config = self.custom_config
            dialog.material_name_edit.setPlainText(config['name'])
            dialog.main_peak_spin.setValue(config['characteristic_peaks']['main'])
            dialog.secondary_peak_spin.setValue(config['characteristic_peaks']['secondary'])
            dialog.tertiary_peak_spin.setValue(config['characteristic_peaks']['tertiary'])
            
            dialog.baseline_start.setValue(config['reference_regions']['baseline'][0])
            dialog.baseline_end.setValue(config['reference_regions']['baseline'][1])
            dialog.fingerprint_start.setValue(config['reference_regions']['fingerprint'][0])
            dialog.fingerprint_end.setValue(config['reference_regions']['fingerprint'][1])
            dialog.high_freq_start.setValue(config['reference_regions']['high_freq'][0])
            dialog.high_freq_end.setValue(config['reference_regions']['high_freq'][1])
            
            dialog.crystalline_density.setValue(config['densities']['crystalline'])
            dialog.matrix_density.setValue(config['densities']['matrix'])
            dialog.low_density.setValue(config['densities']['low_density'])
            
            dialog.low_threshold.setValue(config['classification_thresholds']['low'])
            dialog.medium_threshold.setValue(config['classification_thresholds']['medium'])
            dialog.high_threshold.setValue(config['classification_thresholds']['high'])
            
            dialog.reference_intensity.setValue(config['reference_intensity'])
        
        if dialog.exec() == QDialog.Accepted:
            # Get the new configuration
            self.custom_config = dialog.get_config()
            
            # Update analyzer with new config
            self.analyzer = RamanDensityAnalyzer('Other (Custom)', self.custom_config)
            
            # Update UI
            self.update_density_type_options()
            self.results_text.append(f"Custom material configured: {self.custom_config['name']}")
            
            # Store the custom material for future use
            material_name = self.custom_config['name']
            MaterialConfigs.add_custom_material(material_name, self.custom_config)
            
            # Update the combo box to include the new custom material
            current_text = self.material_combo.currentText()
            self.material_combo.clear()
            self.material_combo.addItems(MaterialConfigs.get_available_materials())
            
            # Select the appropriate item
            if material_name in MaterialConfigs.get_available_materials():
                self.material_combo.setCurrentText(material_name)
            else:
                self.material_combo.setCurrentText(current_text)
    

    def update_density_type_options(self):
        """Update density type options based on selected material."""
        current_material = self.material_combo.currentText()
        self.density_type_combo.clear()
        
        if current_material == 'Kidney Stones (COM)':
            # Biofilm-specific options for kidney stones
            self.density_type_combo.addItems(["mixed", "pure", "bacteria_rich"])
        else:
            # Generic options for other materials
            self.density_type_combo.addItems(["mixed", "low", "medium", "crystalline"])
        
    def create_plot_panel(self):
        """Create the plot panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        return panel
        
    def load_spectrum_file(self):
        """Load a spectrum file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Spectrum File", "",
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # Try to load the file
            data = np.loadtxt(file_path)
            
            if data.shape[1] >= 2:
                self.wavenumbers = data[:, 0]
                self.intensities = data[:, 1]
                
                self.file_label.setText(f"Loaded: {Path(file_path).name}")
                self.plot_spectrum()
                
            else:
                QMessageBox.warning(self, "Format Error", 
                                  "File must contain at least 2 columns (wavenumber, intensity)")
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")
            
    def plot_spectrum(self):
        """Plot the current spectrum."""
        if self.wavenumbers is None or self.intensities is None:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        
        ax.plot(self.wavenumbers, self.intensities, 'b-', linewidth=1, label='Raw spectrum')
        
        if self.processed_data is not None:
            ax.plot(self.processed_data['wavenumbers'], 
                   self.processed_data['corrected_intensity'], 
                   'r-', linewidth=1, label='Processed spectrum')
                   
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Intensity')
        ax.set_title('Raman Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def analyze_spectrum(self):
        """Analyze the current spectrum."""
        if self.wavenumbers is None or self.intensities is None:
            QMessageBox.warning(self, "No Data", "Please load a spectrum file first.")
            return
            
        try:
            # Preprocess spectrum
            wn, corrected_int = self.analyzer.preprocess_spectrum(
                self.wavenumbers, self.intensities)
            
            self.processed_data = {
                'wavenumbers': wn,
                'corrected_intensity': corrected_int
            }
            
            # Calculate CDI and metrics
            cdi, metrics = self.analyzer.calculate_crystalline_density_index(wn, corrected_int)
            
            # Calculate densities
            apparent_density = self.analyzer.calculate_apparent_density(cdi)
            density_type = self.density_type_combo.currentText()
            
            # Use appropriate density calculation based on material type
            if self.analyzer.material_type == 'Kidney Stones (COM)':
                specialized_density = self.analyzer.calculate_biofilm_density(cdi, density_type)
                density_label = f"Biofilm-Calibrated Density ({density_type})"
            else:
                specialized_density = self.analyzer.calculate_density_by_type(cdi, density_type)
                density_label = f"Specialized Density ({density_type})"
            
            # Get material-specific information
            material_config = self.analyzer.config
            thresholds = self.analyzer.classification_thresholds
            
            # Display results
            result_text = f"""Density Analysis Results ({self.analyzer.material_type}):

Crystalline Density Index (CDI): {cdi:.4f}
Standard Apparent Density: {apparent_density:.3f} g/cm³
{density_label}: {specialized_density:.3f} g/cm³

Detailed Metrics:
• Main Peak Height: {metrics['main_peak_height']:.1f}
• Main Peak Position: {metrics['main_peak_position']} cm⁻¹
• Baseline Intensity: {metrics['baseline_intensity']:.1f}
• Peak Width (FWHM): {metrics['peak_width']:.1f} cm⁻¹
• Spectral Contrast: {metrics['spectral_contrast']:.4f}

Classification Guidelines for {self.analyzer.material_type}:
• CDI < {thresholds['low']:.2f}: Low crystallinity regions
• CDI {thresholds['low']:.2f}-{thresholds['medium']:.2f}: Mixed regions
• CDI {thresholds['medium']:.2f}-{thresholds['high']:.2f}: Mixed crystalline
• CDI > {thresholds['high']:.2f}: Pure crystalline material

Material-Specific Density Ranges:
• Low density: {material_config['density_ranges']['low_range'][0]:.2f}-{material_config['density_ranges']['low_range'][1]:.2f} g/cm³
• Mixed: {material_config['density_ranges']['mixed_range'][0]:.2f}-{material_config['density_ranges']['mixed_range'][1]:.2f} g/cm³
• Crystalline: {material_config['density_ranges']['crystalline_range'][0]:.2f}-{material_config['density_ranges']['crystalline_range'][1]:.2f} g/cm³
"""
            
            self.results_text.setText(result_text)
            
            # Create comprehensive density plots
            self.create_density_plots(cdi, apparent_density, specialized_density, metrics, wn, corrected_int)
            
            # Enable save plots button for single spectrum analysis
            self.save_plots_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", 
                               f"Failed to analyze spectrum:\n{str(e)}")

    def create_density_plots(self, cdi, apparent_density, specialized_density, metrics, wavenumbers, intensities):
        """Create comprehensive density analysis plots."""
        try:
            # Clear the figure and create subplots
            self.figure.clear()
            
            # Create a 2x2 subplot layout for comprehensive visualization
            gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # 1. Main spectrum with density annotations (top-left)
            ax1 = self.figure.add_subplot(gs[0, 0])
            ax1.plot(wavenumbers, intensities, 'b-', linewidth=1.5, label='Processed spectrum')
            
            # Add main peak annotation
            main_peak_pos = metrics['main_peak_position']
            main_peak_height = metrics['main_peak_height']
            ax1.axvline(main_peak_pos, color='red', linestyle='--', alpha=0.7, label=f'Main peak ({main_peak_pos} cm⁻¹)')
            ax1.annotate(f'CDI: {cdi:.4f}\nDensity: {specialized_density:.3f} g/cm³', 
                        xy=(main_peak_pos, main_peak_height), 
                        xytext=(main_peak_pos + 100, main_peak_height * 0.8),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
            ax1.set_xlabel('Wavenumber (cm⁻¹)')
            ax1.set_ylabel('Intensity')
            ax1.set_title('Spectrum with Density Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. CDI and Density Visualization (top-right)
            ax2 = self.figure.add_subplot(gs[0, 1])
            
            # Create a bar chart showing different density values
            density_types = ['Apparent\nDensity', 'Specialized\nDensity']
            density_values = [apparent_density, specialized_density]
            colors = ['lightblue', 'orange']
            
            bars = ax2.bar(density_types, density_values, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Density (g/cm³)')
            ax2.set_title(f'Density Analysis Results\nCDI: {cdi:.4f}')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, density_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Add CDI classification color coding
            thresholds = self.analyzer.classification_thresholds
            if cdi < thresholds['low']:
                cdi_color = 'red'
                cdi_class = 'Low Crystallinity'
            elif cdi < thresholds['medium']:
                cdi_color = 'orange'
                cdi_class = 'Mixed'
            elif cdi < thresholds['high']:
                cdi_color = 'yellow'
                cdi_class = 'Mixed Crystalline'
            else:
                cdi_color = 'green'
                cdi_class = 'Pure Crystalline'
            
            ax2.text(0.5, 0.95, f'Classification: {cdi_class}', 
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=cdi_color, alpha=0.7),
                    fontweight='bold')
            
            # 3. Spectral Metrics Visualization (bottom-left)
            ax3 = self.figure.add_subplot(gs[1, 0])
            
            metric_names = ['Peak Height', 'Baseline', 'Peak Width', 'Contrast×1000']
            metric_values = [
                metrics['main_peak_height'],
                metrics['baseline_intensity'],
                metrics['peak_width'],
                metrics['spectral_contrast'] * 1000  # Scale for visibility
            ]
            
            bars3 = ax3.bar(metric_names, metric_values, color=['green', 'blue', 'purple', 'red'], alpha=0.7)
            ax3.set_ylabel('Value')
            ax3.set_title('Spectral Quality Metrics')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars3, metric_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(metric_values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=9)
            
            # 4. Density Classification Guide (bottom-right)
            ax4 = self.figure.add_subplot(gs[1, 1])
            
            # Create a visual guide showing density ranges
            material_config = self.analyzer.config
            density_ranges = material_config['density_ranges']
            
            range_names = ['Low', 'Mixed', 'Crystalline']
            range_values = [
                (density_ranges['low_range'][0] + density_ranges['low_range'][1]) / 2,
                (density_ranges['mixed_range'][0] + density_ranges['mixed_range'][1]) / 2,
                (density_ranges['crystalline_range'][0] + density_ranges['crystalline_range'][1]) / 2
            ]
            range_colors = ['red', 'orange', 'green']
            
            bars4 = ax4.bar(range_names, range_values, color=range_colors, alpha=0.7, edgecolor='black')
            ax4.set_ylabel('Density (g/cm³)')
            ax4.set_title(f'Material: {self.analyzer.material_type}')
            ax4.grid(True, alpha=0.3)
            
            # Highlight current density
            current_density_type = self.density_type_combo.currentText()
            ax4.axhline(specialized_density, color='black', linestyle='--', linewidth=2, 
                       label=f'Current: {specialized_density:.3f} g/cm³')
            ax4.legend()
            
            # Add range labels
            for bar, value in zip(bars4, range_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(range_values)*0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            self.figure.suptitle(f'Comprehensive Density Analysis - {os.path.basename(self.file_label.text())}', 
                               fontsize=14, fontweight='bold')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error creating density plots: {e}")
            # Fallback to simple spectrum plot
            self.plot_spectrum()
    
    def set_batch_fitting_results(self, results):
        """Set batch fitting results for reference intensity calculation."""
        self.batch_fitting_results = results
        print(f"Batch fitting results loaded: {len(results) if results else 0} entries")
        
        if results and len(results) > 0:
            # Update UI to show batch data is available
            self.batch_group.setVisible(True)
            self.batch_status_label.setText(f"✓ {len(results)} spectra loaded from batch processing")
            self.batch_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
            
            # Populate the spectrum selector
            self.batch_spectrum_combo.clear()
            for i, result in enumerate(results):
                file_path = result.get('file', f'Spectrum {i+1}')
                filename = os.path.basename(file_path) if isinstance(file_path, str) else f'Spectrum {i+1}'
                self.batch_spectrum_combo.addItem(f"{i+1}: {filename}")
            
            # Enable buttons
            self.load_batch_spectrum_btn.setEnabled(True)
            self.analyze_all_batch_btn.setEnabled(True)
            
            # Auto-load the first spectrum
            if self.batch_spectrum_combo.count() > 0:
                self.batch_spectrum_combo.setCurrentIndex(0)
                self.load_selected_batch_spectrum()
                
            self.results_text.setText(f"📊 Batch Data Loaded!\n\n"
                                    f"• {len(results)} spectra from batch peak fitting\n"
                                    f"• First spectrum automatically loaded\n"
                                    f"• Use the dropdown above to select different spectra\n"
                                    f"• Click 'Analyze All' for batch density analysis\n\n"
                                    f"Ready for density analysis!")
        else:
            # Hide batch section if no data
            self.batch_group.setVisible(False)

    def on_batch_spectrum_selected(self):
        """Handle batch spectrum selection change."""
        if hasattr(self, 'batch_fitting_results') and self.batch_fitting_results:
            current_index = self.batch_spectrum_combo.currentIndex()
            if 0 <= current_index < len(self.batch_fitting_results):
                result = self.batch_fitting_results[current_index]
                file_path = result.get('file', f'Spectrum {current_index+1}')
                filename = os.path.basename(file_path) if isinstance(file_path, str) else f'Spectrum {current_index+1}'
                
                # Update status to show which spectrum is selected
                self.results_text.append(f"\n📌 Selected: {filename}")

    def load_selected_batch_spectrum(self):
        """Load the currently selected batch spectrum."""
        if not hasattr(self, 'batch_fitting_results') or not self.batch_fitting_results:
            QMessageBox.warning(self, "No Data", "No batch data available.")
            return
            
        current_index = self.batch_spectrum_combo.currentIndex()
        if current_index < 0 or current_index >= len(self.batch_fitting_results):
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid spectrum.")
            return
            
        try:
            result = self.batch_fitting_results[current_index]
            
            # Extract spectrum data
            self.wavenumbers = result.get('wavenumbers', np.array([]))
            self.intensities = result.get('intensities', np.array([]))
            
            if len(self.wavenumbers) == 0 or len(self.intensities) == 0:
                QMessageBox.warning(self, "No Data", "Selected spectrum contains no data.")
                return
            
            # Update file label
            file_path = result.get('file', f'Spectrum {current_index+1}')
            filename = os.path.basename(file_path) if isinstance(file_path, str) else f'Spectrum {current_index+1}'
            self.file_label.setText(f"Loaded from batch: {filename}")
            
            # Plot the spectrum
            self.plot_spectrum()
            
            # Update results text
            self.results_text.setText(f"📊 Loaded Spectrum from Batch Data!\n\n"
                                    f"• File: {filename}\n"
                                    f"• Data points: {len(self.wavenumbers)}\n"
                                    f"• Wavenumber range: {self.wavenumbers.min():.1f} - {self.wavenumbers.max():.1f} cm⁻¹\n"
                                    f"• Original R²: {result.get('r_squared', 'N/A')}\n"
                                    f"• Fitted peaks: {result.get('n_peaks_fitted', 'N/A')}\n\n"
                                    f"Ready to analyze density!")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load batch spectrum:\n{str(e)}")

    def analyze_all_batch_spectra(self):
        """Analyze all spectra in the batch for density."""
        if not hasattr(self, 'batch_fitting_results') or not self.batch_fitting_results:
            QMessageBox.warning(self, "No Data", "No batch data available.")
            return
            
        try:
            from PySide6.QtWidgets import QProgressDialog
            from PySide6.QtCore import Qt
            
            # Create progress dialog
            progress = QProgressDialog("Analyzing batch spectra for density...", "Cancel", 0, len(self.batch_fitting_results), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            
            batch_density_results = []
            
            for i, result in enumerate(self.batch_fitting_results):
                if progress.wasCanceled():
                    break
                    
                progress.setValue(i)
                progress.setLabelText(f"Analyzing spectrum {i+1} of {len(self.batch_fitting_results)}...")
                QApplication.processEvents()
                
                try:
                    # Extract spectrum data
                    wavenumbers = result.get('wavenumbers', np.array([]))
                    intensities = result.get('intensities', np.array([]))
                    
                    if len(wavenumbers) == 0 or len(intensities) == 0:
                        continue
                    
                    # Preprocess spectrum
                    wn, corrected_int = self.analyzer.preprocess_spectrum(wavenumbers, intensities)
                    
                    # Calculate CDI and metrics
                    cdi, metrics = self.analyzer.calculate_crystalline_density_index(wn, corrected_int)
                    
                    # Calculate densities
                    apparent_density = self.analyzer.calculate_apparent_density(cdi)
                    density_type = self.density_type_combo.currentText()
                    
                    if self.analyzer.material_type == 'Kidney Stones (COM)':
                        specialized_density = self.analyzer.calculate_biofilm_density(cdi, density_type)
                    else:
                        specialized_density = self.analyzer.calculate_density_by_type(cdi, density_type)
                    
                    # Store results
                    file_path = result.get('file', f'Spectrum {i+1}')
                    filename = os.path.basename(file_path) if isinstance(file_path, str) else f'Spectrum {i+1}'
                    
                    batch_density_results.append({
                        'filename': filename,
                        'file_path': file_path,
                        'cdi': cdi,
                        'apparent_density': apparent_density,
                        'specialized_density': specialized_density,
                        'metrics': metrics,
                        'original_r_squared': result.get('r_squared', 'N/A')
                    })
                    
                except Exception as e:
                    print(f"Error processing spectrum {i}: {e}")
                    continue
            
            progress.setValue(len(self.batch_fitting_results))
            
            if batch_density_results:
                # Display batch results
                results_text = f"🎯 Batch Density Analysis Results ({len(batch_density_results)} spectra):\n\n"
                
                for i, res in enumerate(batch_density_results):
                    results_text += f"{i+1}. {res['filename']}\n"
                    results_text += f"   CDI: {res['cdi']:.4f} | Apparent: {res['apparent_density']:.3f} g/cm³ | Specialized: {res['specialized_density']:.3f} g/cm³\n"
                    results_text += f"   Original R²: {res['original_r_squared']}\n\n"
                
                results_text += f"\nMaterial: {self.analyzer.material_type}\n"
                results_text += f"Density Type: {density_type}\n"
                results_text += f"Analysis completed successfully!"
                
                self.results_text.setText(results_text)
                
                # Store results for potential export
                self.batch_density_results = batch_density_results
                
                # Enable export buttons
                self.export_batch_btn.setEnabled(True)
                self.save_plots_btn.setEnabled(True)
                
                # Enable trend analysis after batch processing
                self.analyze_trends_btn.setEnabled(True)
                
                # Create batch density visualization
                self.create_batch_density_plots(batch_density_results)
                
            else:
                QMessageBox.warning(self, "No Results", "No valid density analysis results obtained.")
                
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze batch spectra:\n{str(e)}")

    def create_batch_density_plots(self, batch_results):
        """Create comprehensive batch density analysis plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Clear the figure and create subplots for batch analysis
            self.figure.clear()
            
            # Create a 2x2 subplot layout for batch visualization
            gs = self.figure.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
            
            # Extract data for plotting
            filenames = [res['filename'] for res in batch_results]
            cdis = [res['cdi'] for res in batch_results]
            apparent_densities = [res['apparent_density'] for res in batch_results]
            specialized_densities = [res['specialized_density'] for res in batch_results]
            original_r2s = [float(res['original_r_squared']) if res['original_r_squared'] != 'N/A' else 0 for res in batch_results]
            
            # 1. CDI Distribution (top-left)
            ax1 = self.figure.add_subplot(gs[0, 0])
            
            # Create histogram of CDI values
            ax1.hist(cdis, bins=min(10, len(cdis)), alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(np.mean(cdis), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cdis):.4f}')
            ax1.set_xlabel('Crystalline Density Index (CDI)')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'CDI Distribution ({len(batch_results)} spectra)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Density Comparison (top-right)
            ax2 = self.figure.add_subplot(gs[0, 1])
            
            # Scatter plot comparing apparent vs specialized density
            scatter = ax2.scatter(apparent_densities, specialized_densities, 
                                c=cdis, cmap='viridis', alpha=0.7, s=60, edgecolors='black')
            
            # Add diagonal line for reference
            min_density = min(min(apparent_densities), min(specialized_densities))
            max_density = max(max(apparent_densities), max(specialized_densities))
            ax2.plot([min_density, max_density], [min_density, max_density], 
                    'r--', alpha=0.5, label='Equal densities')
            
            ax2.set_xlabel('Apparent Density (g/cm³)')
            ax2.set_ylabel('Specialized Density (g/cm³)')
            ax2.set_title('Density Correlation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar for CDI
            cbar = self.figure.colorbar(scatter, ax=ax2)
            cbar.set_label('CDI', rotation=270, labelpad=15)
            
            # 3. Spectrum Quality vs Density (bottom-left)
            ax3 = self.figure.add_subplot(gs[1, 0])
            
            # Plot R² vs specialized density
            ax3.scatter(specialized_densities, original_r2s, 
                       c=cdis, cmap='plasma', alpha=0.7, s=60, edgecolors='black')
            ax3.set_xlabel('Specialized Density (g/cm³)')
            ax3.set_ylabel('Original Fit R²')
            ax3.set_title('Fit Quality vs Density')
            ax3.grid(True, alpha=0.3)
            
            # 4. Batch Overview Bar Chart (bottom-right)
            ax4 = self.figure.add_subplot(gs[1, 1])
            
            # Show density values for each spectrum (limited to first 10 for readability)
            display_count = min(10, len(batch_results))
            x_indices = range(display_count)
            
            # Truncate filenames for display
            display_names = [name[:8] + '...' if len(name) > 8 else name 
                           for name in filenames[:display_count]]
            
            bars = ax4.bar(x_indices, specialized_densities[:display_count], 
                          color=plt.cm.viridis(np.array(cdis[:display_count])/max(cdis)), 
                          alpha=0.7, edgecolor='black')
            
            ax4.set_xlabel('Spectrum')
            ax4.set_ylabel('Specialized Density (g/cm³)')
            ax4.set_title(f'Density by Spectrum (first {display_count})')
            ax4.set_xticks(x_indices)
            ax4.set_xticklabels(display_names, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, density, cdi in zip(bars, specialized_densities[:display_count], cdis[:display_count]):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(specialized_densities)*0.01,
                        f'{density:.3f}\n({cdi:.3f})', ha='center', va='bottom', fontsize=8)
            
            # Add summary statistics
            summary_text = f"""Batch Statistics:
• Spectra analyzed: {len(batch_results)}
• CDI range: {min(cdis):.4f} - {max(cdis):.4f}
• Mean CDI: {np.mean(cdis):.4f} ± {np.std(cdis):.4f}
• Density range: {min(specialized_densities):.3f} - {max(specialized_densities):.3f} g/cm³
• Mean density: {np.mean(specialized_densities):.3f} ± {np.std(specialized_densities):.3f} g/cm³
• Material: {self.analyzer.material_type}"""
            
            # Add text box with summary
            self.figure.text(0.02, 0.02, summary_text, fontsize=9, 
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
                           verticalalignment='bottom')
            
            self.figure.suptitle(f'Batch Density Analysis Results - {len(batch_results)} Spectra', 
                               fontsize=14, fontweight='bold')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error creating batch density plots: {e}")
            # Fallback to text results only
            pass

    def export_batch_density_results(self):
        """Export batch density results to CSV file."""
        if not hasattr(self, 'batch_density_results') or not self.batch_density_results:
            QMessageBox.warning(self, "No Data", "No batch density results to export.")
            return
            
        try:
            from PySide6.QtWidgets import QFileDialog
            import csv
            
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Batch Density Results", 
                f"batch_density_results_{self.analyzer.material_type.replace(' ', '_')}.csv",
                "CSV files (*.csv);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            # Write results to CSV
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = ['filename', 'file_path', 'cdi', 'apparent_density', 'specialized_density', 
                            'main_peak_height', 'main_peak_position', 'baseline_intensity', 
                            'peak_width', 'spectral_contrast', 'original_r_squared', 'material_type', 'density_type']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.batch_density_results:
                    row = {
                        'filename': result['filename'],
                        'file_path': result['file_path'],
                        'cdi': result['cdi'],
                        'apparent_density': result['apparent_density'],
                        'specialized_density': result['specialized_density'],
                        'main_peak_height': result['metrics']['main_peak_height'],
                        'main_peak_position': result['metrics']['main_peak_position'],
                        'baseline_intensity': result['metrics']['baseline_intensity'],
                        'peak_width': result['metrics']['peak_width'],
                        'spectral_contrast': result['metrics']['spectral_contrast'],
                        'original_r_squared': result['original_r_squared'],
                        'material_type': self.analyzer.material_type,
                        'density_type': self.density_type_combo.currentText()
                    }
                    writer.writerow(row)
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Batch density results exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")

    def save_density_plots(self):
        """Save the current density plots to file."""
        try:
            from PySide6.QtWidgets import QFileDialog
            
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Density Plots", 
                f"density_analysis_plots_{self.analyzer.material_type.replace(' ', '_')}.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            # Save the current figure
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
            
            QMessageBox.information(self, "Save Complete", 
                                  f"Density plots saved to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save plots:\n{str(e)}")

    def analyze_trends(self):
        """Analyze trends in batch density data with confidence intervals."""
        if not hasattr(self, 'batch_density_results') or not self.batch_density_results:
            QMessageBox.warning(self, "No Data", "No batch density results available for trend analysis.\n\nPlease run batch analysis first.")
            return
        
        try:
            import scipy.stats as stats
            from scipy import signal
            
            # Get the batch results
            results = self.batch_density_results.copy()
            
            # Sort results according to sequence order
            results = self.sort_sequence(results)
            
            # Extract data for trend analysis
            sequence_indices = np.arange(len(results))
            
            # Prepare data arrays
            data_arrays = {}
            if self.analyze_cdi_checkbox.isChecked():
                data_arrays['CDI'] = np.array([r['cdi'] for r in results])
            if self.analyze_density_checkbox.isChecked():
                data_arrays['Specialized Density'] = np.array([r['specialized_density'] for r in results])
            if self.analyze_peak_height_checkbox.isChecked():
                data_arrays['Peak Height'] = np.array([r['metrics']['main_peak_height'] for r in results])
            if self.analyze_peak_width_checkbox.isChecked():
                data_arrays['Peak Width'] = np.array([r['metrics']['peak_width'] for r in results])
            
            if not data_arrays:
                QMessageBox.warning(self, "No Parameters", "Please select at least one parameter to analyze.")
                return
            
            # Perform trend analysis for each parameter
            trend_results = {}
            confidence_level = self.confidence_level.value()
            window_size = self.moving_avg_window.value()
            sensitivity = self.trend_sensitivity.value()
            
            for param_name, data in data_arrays.items():
                trend_result = self.calculate_trend_statistics(
                    sequence_indices, data, confidence_level, window_size, sensitivity
                )
                trend_results[param_name] = trend_result
            
            # Store results for plotting and export
            self.trend_analysis_results = {
                'results': trend_results,
                'sequence_indices': sequence_indices,
                'filenames': [r['filename'] for r in results],
                'parameters': {
                    'confidence_level': confidence_level,
                    'window_size': window_size,
                    'sensitivity': sensitivity,
                    'sequence_order': self.sequence_order_combo.currentText()
                }
            }
            
            # Display results
            self.display_trend_results(trend_results)
            
            # Create trend plots
            self.create_trend_plots(trend_results, sequence_indices, [r['filename'] for r in results])
            
            # Enable export buttons
            self.export_trends_btn.setEnabled(True)
            self.save_trend_plots_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze trends:\n{str(e)}")
    
    def sort_sequence(self, results):
        """Sort the sequence according to the selected ordering method."""
        order_method = self.sequence_order_combo.currentText()
        
        if order_method == "Filename (alphabetical)":
            return sorted(results, key=lambda x: x['filename'])
        elif order_method == "Filename (numerical)":
            import re
            def extract_number(filename):
                numbers = re.findall(r'\d+', filename)
                return int(numbers[0]) if numbers else 0
            return sorted(results, key=lambda x: extract_number(x['filename']))
        elif order_method == "Timestamp (if available)":
            # Try to extract timestamp from filename or use modification time
            import os
            import time
            def get_timestamp(result):
                try:
                    if os.path.exists(result['file_path']):
                        return os.path.getmtime(result['file_path'])
                    else:
                        # Extract timestamp from filename if possible
                        import re
                        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}|\d{8})', result['filename'])
                        if timestamp_match:
                            timestamp_str = timestamp_match.group(1)
                            if '-' in timestamp_str:
                                return time.mktime(time.strptime(timestamp_str, '%Y-%m-%d'))
                            else:
                                return time.mktime(time.strptime(timestamp_str, '%Y%m%d'))
                        return 0
                except:
                    return 0
            return sorted(results, key=get_timestamp)
        else:  # Manual ordering - keep original order
            return results
    
    def calculate_trend_statistics(self, x, y, confidence_level, window_size, sensitivity):
        """Calculate comprehensive trend statistics with confidence intervals."""
        import scipy.stats as stats
        from scipy import signal
        
        n = len(y)
        if n < 3:
            return {
                'trend': 'insufficient_data',
                'slope': 0,
                'r_squared': 0,
                'p_value': 1.0,
                'confidence_interval': (0, 0),
                'smoothed_data': y,
                'residuals': np.zeros_like(y)
            }
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Calculate confidence interval for the slope
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 2)
        slope_ci = (slope - t_critical * std_err, slope + t_critical * std_err)
        
        # Calculate smoothed data using moving average
        if window_size > 1 and n >= window_size:
            smoothed = signal.savgol_filter(y, min(window_size, n), min(2, window_size-1))
        else:
            smoothed = y.copy()
        
        # Calculate residuals
        fitted_line = slope * x + intercept
        residuals = y - fitted_line
        
        # Determine trend significance
        if abs(slope) < sensitivity:
            trend = 'stable'
        elif slope > sensitivity and p_value < 0.05:
            trend = 'increasing'
        elif slope < -sensitivity and p_value < 0.05:
            trend = 'decreasing'
        else:
            trend = 'uncertain'
        
        # Calculate prediction intervals
        pred_intervals = self.calculate_prediction_intervals(x, y, confidence_level)
        
        return {
            'trend': trend,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err,
            'confidence_interval': slope_ci,
            'smoothed_data': smoothed,
            'residuals': residuals,
            'fitted_line': fitted_line,
            'prediction_intervals': pred_intervals
        }
    
    def calculate_prediction_intervals(self, x, y, confidence_level):
        """Calculate prediction intervals for the regression."""
        import scipy.stats as stats
        
        n = len(y)
        if n < 3:
            return {'lower': y, 'upper': y}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate residual standard error
        fitted = slope * x + intercept
        residuals = y - fitted
        mse = np.sum(residuals**2) / (n - 2)
        
        # Calculate standard error for prediction
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean)**2)
        
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 2)
        
        pred_se = np.sqrt(mse * (1 + 1/n + (x - x_mean)**2 / sxx))
        margin = t_critical * pred_se
        
        return {
            'lower': fitted - margin,
            'upper': fitted + margin
        }
    
    def display_trend_results(self, trend_results):
        """Display trend analysis results in the text widget."""
        results_text = "📈 TREND ANALYSIS RESULTS\n" + "="*50 + "\n\n"
        
        for param_name, result in trend_results.items():
            results_text += f"🔍 {param_name}:\n"
            results_text += f"   • Trend: {result['trend'].upper()}\n"
            results_text += f"   • Slope: {result['slope']:.6f} ± {result['std_err']:.6f}\n"
            results_text += f"   • R²: {result['r_squared']:.4f}\n"
            results_text += f"   • P-value: {result['p_value']:.4f}\n"
            
            # Confidence interval for slope
            ci_lower, ci_upper = result['confidence_interval']
            results_text += f"   • {self.confidence_level.value()*100:.0f}% CI: [{ci_lower:.6f}, {ci_upper:.6f}]\n"
            
            # Trend interpretation
            if result['trend'] == 'increasing':
                results_text += f"   • ↗️ Significant INCREASING trend detected\n"
            elif result['trend'] == 'decreasing':
                results_text += f"   • ↘️ Significant DECREASING trend detected\n"
            elif result['trend'] == 'stable':
                results_text += f"   • ➡️ STABLE - no significant trend\n"
            else:
                results_text += f"   • ❓ UNCERTAIN trend (low significance)\n"
            
            results_text += "\n"
        
        # Analysis parameters
        results_text += f"📊 Analysis Parameters:\n"
        results_text += f"   • Confidence Level: {self.confidence_level.value()*100:.0f}%\n"
        results_text += f"   • Smoothing Window: {self.moving_avg_window.value()} points\n"
        results_text += f"   • Trend Sensitivity: {self.trend_sensitivity.value():.3f}\n"
        results_text += f"   • Sequence Order: {self.sequence_order_combo.currentText()}\n"
        results_text += f"   • Number of Spectra: {len(self.trend_analysis_results['sequence_indices'])}\n"
        
        self.trend_results_text.setText(results_text)
    
    def create_trend_plots(self, trend_results, sequence_indices, filenames):
        """Create comprehensive trend analysis plots with confidence intervals."""
        try:
            # Clear the figure and create subplots
            self.figure.clear()
            
            # Determine subplot layout based on number of parameters
            n_params = len(trend_results)
            if n_params == 1:
                rows, cols = 1, 1
            elif n_params == 2:
                rows, cols = 1, 2
            elif n_params <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 3, 2
            
            # Create subplots with shared x-axis
            fig_height = max(8, rows * 3)
            gs = self.figure.add_gridspec(rows, cols, hspace=0.4, wspace=0.3)
            
            plot_idx = 0
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for param_name, result in trend_results.items():
                if plot_idx >= rows * cols:
                    break
                    
                row = plot_idx // cols
                col = plot_idx % cols
                ax = self.figure.add_subplot(gs[row, col])
                
                color = colors[plot_idx % len(colors)]
                
                # Plot raw data points
                ax.scatter(sequence_indices, result['smoothed_data'], 
                          color=color, alpha=0.6, s=40, label='Data points', zorder=3)
                
                # Plot fitted line
                ax.plot(sequence_indices, result['fitted_line'], 
                       color='red', linewidth=2, label='Linear fit', zorder=4)
                
                # Plot confidence intervals (prediction intervals)
                pred_intervals = result['prediction_intervals']
                ax.fill_between(sequence_indices, pred_intervals['lower'], pred_intervals['upper'],
                               color='gray', alpha=0.2, label=f'{self.confidence_level.value()*100:.0f}% Prediction Interval')
                
                # Plot smoothed trend line
                if len(result['smoothed_data']) > 1:
                    ax.plot(sequence_indices, result['smoothed_data'], 
                           color=color, linewidth=1.5, alpha=0.8, linestyle='--', 
                           label='Smoothed trend', zorder=2)
                
                # Formatting
                ax.set_xlabel('Sequence Index')
                ax.set_ylabel(param_name)
                ax.set_title(f"{param_name} - {result['trend'].title()} Trend")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                # Add trend information as text
                trend_info = f"Slope: {result['slope']:.4f}\nR²: {result['r_squared']:.3f}\nP: {result['p_value']:.3f}"
                ax.text(0.02, 0.98, trend_info, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor='white', alpha=0.8), fontsize=8)
                
                # Color-code the title based on trend significance
                if result['trend'] == 'increasing':
                    title_color = 'green'
                elif result['trend'] == 'decreasing':
                    title_color = 'red'
                elif result['trend'] == 'stable':
                    title_color = 'blue'
                else:
                    title_color = 'orange'
                
                ax.set_title(f"{param_name} - {result['trend'].title()} Trend", color=title_color, fontweight='bold')
                
                plot_idx += 1
            
            # Add overall title with analysis summary
            n_spectra = len(sequence_indices)
            sequence_order = self.sequence_order_combo.currentText()
            
            summary_title = f"Trend Analysis: {n_spectra} Spectra - {sequence_order}"
            self.figure.suptitle(summary_title, fontsize=14, fontweight='bold')
            
            # Add analysis parameters as figure text
            params_text = (f"Confidence: {self.confidence_level.value()*100:.0f}%, "
                          f"Window: {self.moving_avg_window.value()}, "
                          f"Sensitivity: {self.trend_sensitivity.value():.3f}")
            
            self.figure.text(0.02, 0.02, params_text, fontsize=9, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error creating trend plots: {e}")
            # Fallback to simple plot
            self.figure.clear()
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error creating trend plots:\n{str(e)}", 
                   transform=ax.transAxes, ha='center', va='center')
            self.canvas.draw()
    
    def export_trend_analysis(self):
        """Export trend analysis results to CSV file."""
        if not hasattr(self, 'trend_analysis_results'):
            QMessageBox.warning(self, "No Data", "No trend analysis results to export.")
            return
            
        try:
            from PySide6.QtWidgets import QFileDialog
            import csv
            import json
            
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Trend Analysis Results", 
                f"trend_analysis_{self.analyzer.material_type.replace(' ', '_')}.csv",
                "CSV files (*.csv);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            # Write results to CSV
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = ['parameter', 'trend', 'slope', 'intercept', 'r_squared', 'p_value', 
                            'slope_std_err', 'ci_lower', 'ci_upper', 'material_type', 
                            'confidence_level', 'window_size', 'sensitivity', 'sequence_order', 'n_spectra']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                results = self.trend_analysis_results['results']
                params = self.trend_analysis_results['parameters']
                
                for param_name, result in results.items():
                    ci_lower, ci_upper = result['confidence_interval']
                    row = {
                        'parameter': param_name,
                        'trend': result['trend'],
                        'slope': result['slope'],
                        'intercept': result['intercept'],
                        'r_squared': result['r_squared'],
                        'p_value': result['p_value'],
                        'slope_std_err': result['std_err'],
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'material_type': self.analyzer.material_type,
                        'confidence_level': params['confidence_level'],
                        'window_size': params['window_size'],
                        'sensitivity': params['sensitivity'],
                        'sequence_order': params['sequence_order'],
                        'n_spectra': len(self.trend_analysis_results['sequence_indices'])
                    }
                    writer.writerow(row)
            
            # Also save detailed data file
            detail_file = file_path.replace('.csv', '_detailed_data.csv')
            with open(detail_file, 'w', newline='') as csvfile:
                # Create header with all parameters
                fieldnames = ['sequence_index', 'filename']
                for param_name in results.keys():
                    fieldnames.extend([
                        f'{param_name}_raw',
                        f'{param_name}_smoothed',
                        f'{param_name}_fitted',
                        f'{param_name}_residual',
                        f'{param_name}_pred_lower',
                        f'{param_name}_pred_upper'
                    ])
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write data for each spectrum
                sequence_indices = self.trend_analysis_results['sequence_indices']
                filenames = self.trend_analysis_results['filenames']
                
                for i, (seq_idx, filename) in enumerate(zip(sequence_indices, filenames)):
                    row = {'sequence_index': seq_idx, 'filename': filename}
                    
                    for param_name, result in results.items():
                        if i < len(result['smoothed_data']):
                            row[f'{param_name}_raw'] = result['smoothed_data'][i]  # Actually the data points
                            row[f'{param_name}_smoothed'] = result['smoothed_data'][i]
                            row[f'{param_name}_fitted'] = result['fitted_line'][i]
                            row[f'{param_name}_residual'] = result['residuals'][i]
                            row[f'{param_name}_pred_lower'] = result['prediction_intervals']['lower'][i]
                            row[f'{param_name}_pred_upper'] = result['prediction_intervals']['upper'][i]
                    
                    writer.writerow(row)
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Trend analysis results exported to:\n{file_path}\n\n"
                                  f"Detailed data saved to:\n{detail_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export trend results:\n{str(e)}")
    
    def save_trend_plots(self):
        """Save the current trend plots to file."""
        try:
            from PySide6.QtWidgets import QFileDialog
            
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Trend Analysis Plots", 
                f"trend_analysis_plots_{self.analyzer.material_type.replace(' ', '_')}.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            # Save the current figure
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
            
            QMessageBox.information(self, "Save Complete", 
                                  f"Trend analysis plots saved to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save trend plots:\n{str(e)}")
    
    def set_batch_fitting_results(self, results):
        """Set batch fitting results and enable trend analysis."""
        # Call the original method to handle batch data
        super_method = getattr(super(), 'set_batch_fitting_results', None)
        if super_method:
            super_method(results)
        else:
            # Handle the batch data setup manually
            self.batch_fitting_results = results
            print(f"Batch fitting results loaded: {len(results) if results else 0} entries")
            
            if results and len(results) > 0:
                # Update UI to show batch data is available
                self.batch_group.setVisible(True)
                self.batch_status_label.setText(f"✓ {len(results)} spectra loaded from batch processing")
                self.batch_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
                
                # Populate the spectrum selector
                self.batch_spectrum_combo.clear()
                for i, result in enumerate(results):
                    file_path = result.get('file', f'Spectrum {i+1}')
                    filename = os.path.basename(file_path) if isinstance(file_path, str) else f'Spectrum {i+1}'
                    self.batch_spectrum_combo.addItem(f"{i+1}: {filename}")
                
                # Enable buttons
                self.load_batch_spectrum_btn.setEnabled(True)
                self.analyze_all_batch_btn.setEnabled(True)
                
                # Enable trend analysis once batch data is available
                self.analyze_trends_btn.setEnabled(True)
                
                # Auto-load the first spectrum
                if self.batch_spectrum_combo.count() > 0:
                    self.batch_spectrum_combo.setCurrentIndex(0)
                    self.load_selected_batch_spectrum()
                    
                self.results_text.setText(f"📊 Batch Data Loaded!\n\n"
                                        f"• {len(results)} spectra from batch peak fitting\n"
                                        f"• First spectrum automatically loaded\n"
                                        f"• Use the dropdown above to select different spectra\n"
                                        f"• Click 'Analyze All' for batch density analysis\n"
                                        f"• Switch to 'Trend Analysis' tab for sequential analysis\n\n"
                                        f"Ready for density analysis!")
            else:
                # Hide batch section if no data
                self.batch_group.setVisible(False)
                self.analyze_trends_btn.setEnabled(False)


def main():
    """Main function to run the density analysis GUI."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Apply RamanLab matplotlib theme (in case it wasn't applied in __init__)
    apply_theme('compact')
    
    # Create and show the main window
    window = DensityAnalysisGUI()
    
    # Check for command line arguments for automatic file loading
    if len(sys.argv) > 1:
        spectrum_file = sys.argv[1]
        try:
            # Try to load the spectrum file automatically
            import numpy as np
            data = np.loadtxt(spectrum_file)
            if data.shape[1] >= 2:
                window.wavenumbers = data[:, 0]
                window.intensities = data[:, 1]
                window.file_label.setText(f"Loaded: {os.path.basename(spectrum_file)}")
                window.plot_spectrum()
                print(f"✓ Auto-loaded spectrum file: {spectrum_file}")
            else:
                print(f"⚠️ Invalid spectrum file format: {spectrum_file}")
        except Exception as e:
            print(f"⚠️ Failed to auto-load spectrum file {spectrum_file}: {e}")
    
    window.show()
    
    # If this script is run directly (not from another app), start the event loop
    if __name__ == "__main__":
        sys.exit(app.exec())
    
    return window


if __name__ == "__main__":
    main() 