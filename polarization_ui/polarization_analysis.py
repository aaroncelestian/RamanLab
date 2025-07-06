"""
Polarization Analysis UI Module

This module provides comprehensive polarization analysis functionality for Qt6,
including Raman tensor calculations, depolarization ratio analysis, and
symmetry classification based on the legacy tkinter implementation.

Features:
- Polarized spectra loading and management
- Raman tensor calculation and display
- Depolarization ratio analysis
- Symmetry classification
- Angular dependence analysis
- Professional Qt6 implementation
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QTabWidget,
    QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem, QTableWidget,
    QTableWidgetItem, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QTextEdit,
    QFileDialog, QMessageBox, QFormLayout, QSplitter, QFrame, QProgressBar,
    QHeaderView, QDialog, QDialogButtonBox, QTreeWidget, QTreeWidgetItem
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPalette, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from .matplotlib_config import CompactNavigationToolbar as NavigationToolbar
from scipy.signal import find_peaks

# Try to import core modules
try:
    from core.polarization import PolarizationAnalyzer, PolarizationData
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    
    # Mock implementations
    class PolarizationData:
        def __init__(self, wavenumbers=None, intensities=None, config="xx"):
            self.wavenumbers = wavenumbers if wavenumbers is not None else np.linspace(200, 2000, 1000)
            self.intensities = intensities if intensities is not None else np.random.normal(100, 10, 1000)
            self.config = config

# Utility functions for Raman tensor generation
def generate_symmetry_adapted_tensors(crystal_system: str) -> Dict[str, np.ndarray]:
    """
    Generate symmetry-adapted Raman tensors for a given crystal system.
    
    Args:
        crystal_system (str): The crystal system (e.g., 'Cubic', 'Tetragonal', etc.)
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of symmetry-adapted tensors
    """
    # Initialize empty 3x3 tensor
    tensor = np.zeros((3, 3))
    
    # Dictionary to store symmetry-adapted tensors
    symmetry_tensors = {}
    
    if crystal_system == "Cubic":
        # A1g (totally symmetric)
        tensor = np.eye(3)
        symmetry_tensors['A1g'] = tensor
        
        # Eg (doubly degenerate)
        tensor1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        tensor2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3)
        symmetry_tensors['Eg1'] = tensor1
        symmetry_tensors['Eg2'] = tensor2
        
        # T2g (triply degenerate)
        tensor1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        tensor2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        tensor3 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        symmetry_tensors['T2g1'] = tensor1
        symmetry_tensors['T2g2'] = tensor2
        symmetry_tensors['T2g3'] = tensor3
        
    elif crystal_system == "Tetragonal":
        # A1g (totally symmetric)
        tensor = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        symmetry_tensors['A1g'] = tensor
        
        # B1g
        tensor = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        symmetry_tensors['B1g'] = tensor
        
        # B2g
        tensor = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        symmetry_tensors['B2g'] = tensor
        
        # Eg (doubly degenerate)
        tensor1 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        tensor2 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        symmetry_tensors['Eg1'] = tensor1
        symmetry_tensors['Eg2'] = tensor2
        
    elif crystal_system == "Orthorhombic":
        # Ag (totally symmetric)
        tensor = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        symmetry_tensors['Ag'] = tensor
        
        # B1g
        tensor = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        symmetry_tensors['B1g'] = tensor
        
        # B2g
        tensor = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        symmetry_tensors['B2g'] = tensor
        
        # B3g
        tensor = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        symmetry_tensors['B3g'] = tensor
        
    elif crystal_system == "Hexagonal":
        # A1g (totally symmetric)
        tensor = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        symmetry_tensors['A1g'] = tensor
        
        # E1g (doubly degenerate)
        tensor1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        tensor2 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        symmetry_tensors['E1g1'] = tensor1
        symmetry_tensors['E1g2'] = tensor2
        
        # E2g (doubly degenerate)
        tensor1 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        tensor2 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        symmetry_tensors['E2g1'] = tensor1
        symmetry_tensors['E2g2'] = tensor2
        
    elif crystal_system == "Trigonal":
        # A1g (totally symmetric)
        tensor = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        symmetry_tensors['A1g'] = tensor
        
        # Eg (doubly degenerate)
        tensor1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        tensor2 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        symmetry_tensors['Eg1'] = tensor1
        symmetry_tensors['Eg2'] = tensor2
        
    elif crystal_system == "Monoclinic":
        # Ag (totally symmetric)
        tensor = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        symmetry_tensors['Ag'] = tensor
        
        # Bg
        tensor = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        symmetry_tensors['Bg'] = tensor
        
    elif crystal_system == "Triclinic":
        # Ag (totally symmetric)
        tensor = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        symmetry_tensors['Ag'] = tensor
        
    return symmetry_tensors

def calculate_tensor_intensities(tensor: np.ndarray, scattering_geometry: str) -> Dict[str, float]:
    """
    Calculate Raman intensities for different polarization configurations.
    
    Args:
        tensor (np.ndarray): 3x3 Raman tensor
        scattering_geometry (str): Scattering geometry ('Backscattering', 'Right-angle', 'Forward')
        
    Returns:
        Dict[str, float]: Dictionary of intensities for different polarization configurations
    """
    # Define polarization vectors for different configurations
    if scattering_geometry == "Backscattering":
        # k_i = -k_s along z-axis
        e_i = np.array([1, 0, 0])  # Incident polarization
        e_s = np.array([0, 1, 0])  # Scattered polarization
    elif scattering_geometry == "Right-angle":
        # k_i along x-axis, k_s along y-axis
        e_i = np.array([0, 1, 0])
        e_s = np.array([1, 0, 0])
    else:  # Forward
        # k_i = k_s along z-axis
        e_i = np.array([1, 0, 0])
        e_s = np.array([1, 0, 0])
    
    # Calculate intensities for different configurations
    intensities = {}
    
    # xx configuration
    e_i_xx = np.array([1, 0, 0])
    e_s_xx = np.array([1, 0, 0])
    I_xx = np.abs(np.dot(e_s_xx, np.dot(tensor, e_i_xx)))**2
    intensities['xx'] = I_xx
    
    # xy configuration
    e_i_xy = np.array([1, 0, 0])
    e_s_xy = np.array([0, 1, 0])
    I_xy = np.abs(np.dot(e_s_xy, np.dot(tensor, e_i_xy)))**2
    intensities['xy'] = I_xy
    
    # xz configuration
    e_i_xz = np.array([1, 0, 0])
    e_s_xz = np.array([0, 0, 1])
    I_xz = np.abs(np.dot(e_s_xz, np.dot(tensor, e_i_xz)))**2
    intensities['xz'] = I_xz
    
    # yx configuration
    e_i_yx = np.array([0, 1, 0])
    e_s_yx = np.array([1, 0, 0])
    I_yx = np.abs(np.dot(e_s_yx, np.dot(tensor, e_i_yx)))**2
    intensities['yx'] = I_yx
    
    # yy configuration
    e_i_yy = np.array([0, 1, 0])
    e_s_yy = np.array([0, 1, 0])
    I_yy = np.abs(np.dot(e_s_yy, np.dot(tensor, e_i_yy)))**2
    intensities['yy'] = I_yy
    
    # yz configuration
    e_i_yz = np.array([0, 1, 0])
    e_s_yz = np.array([0, 0, 1])
    I_yz = np.abs(np.dot(e_s_yz, np.dot(tensor, e_i_yz)))**2
    intensities['yz'] = I_yz
    
    # zx configuration
    e_i_zx = np.array([0, 0, 1])
    e_s_zx = np.array([1, 0, 0])
    I_zx = np.abs(np.dot(e_s_zx, np.dot(tensor, e_i_zx)))**2
    intensities['zx'] = I_zx
    
    # zy configuration
    e_i_zy = np.array([0, 0, 1])
    e_s_zy = np.array([0, 1, 0])
    I_zy = np.abs(np.dot(e_s_zy, np.dot(tensor, e_i_zy)))**2
    intensities['zy'] = I_zy
    
    # zz configuration
    e_i_zz = np.array([0, 0, 1])
    e_s_zz = np.array([0, 0, 1])
    I_zz = np.abs(np.dot(e_s_zz, np.dot(tensor, e_i_zz)))**2
    intensities['zz'] = I_zz
    
    return intensities

def determine_raman_tensor_from_data(polarization_data: Dict[str, PolarizationData], 
                                   crystal_system: str,
                                   scattering_geometry: str) -> Dict[str, Any]:
    """
    Determine Raman tensor from experimental polarization data.
    
    Args:
        polarization_data (Dict[str, PolarizationData]): Dictionary of polarization data
        crystal_system (str): Crystal system
        scattering_geometry (str): Scattering geometry
        
    Returns:
        Dict[str, Any]: Dictionary containing tensor information and analysis results
    """
    # Get symmetry-adapted tensors
    symmetry_tensors = generate_symmetry_adapted_tensors(crystal_system)
    
    # Initialize results dictionary
    results = {
        'symmetry_tensors': symmetry_tensors,
        'experimental_intensities': {},
        'theoretical_intensities': {},
        'best_fit_symmetry': None,
        'best_fit_tensor': None,
        'fit_error': float('inf')
    }
    
    # Calculate experimental intensities at peak positions
    for config, data in polarization_data.items():
        # Find peaks
        peaks, _ = find_peaks(data.intensities, prominence=np.max(data.intensities) * 0.1, distance=20)
        
        # Store peak intensities
        results['experimental_intensities'][config] = {
            'wavenumbers': data.wavenumbers[peaks],
            'intensities': data.intensities[peaks]
        }
    
    # Compare with theoretical predictions
    for symmetry, tensor in symmetry_tensors.items():
        # Calculate theoretical intensities
        theoretical_intensities = calculate_tensor_intensities(tensor, scattering_geometry)
        
        # Calculate error between experimental and theoretical intensities
        error = 0
        for config in polarization_data.keys():
            if config in theoretical_intensities:
                exp_int = np.mean(results['experimental_intensities'][config]['intensities'])
                theo_int = theoretical_intensities[config]
                error += (exp_int - theo_int)**2
        
        # Update best fit if error is lower
        if error < results['fit_error']:
            results['fit_error'] = error
            results['best_fit_symmetry'] = symmetry
            results['best_fit_tensor'] = tensor
            results['theoretical_intensities'] = theoretical_intensities
    
    return results

class PolarizationAnalysisWidget(QWidget):
    """
    Main widget for polarization analysis functionality.
    
    This widget provides comprehensive polarization analysis tools including:
    - Polarized spectra management
    - Raman tensor calculations
    - Depolarization ratio analysis
    - Symmetry classification
    """
    
    # Signals
    analysis_complete = Signal(dict)
    tensor_calculated = Signal(dict)
    
    def __init__(self, parent=None):
        """Initialize the polarization analysis widget."""
        super().__init__(parent)
        
        # Data storage
        self.polarization_data = {}  # Dict[config, PolarizationData]
        self.raman_tensors = {}
        self.depolarization_ratios = {}
        self.tensor_analysis_results = {}
        self.symmetry_classifications = {}
        
        # UI state
        self.current_crystal_system = "Unknown"
        self.current_scattering_geometry = "Backscattering"
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - controls
        self.control_panel = self.create_control_panel()
        splitter.addWidget(self.control_panel)
        
        # Right panel - visualization
        self.viz_panel = self.create_visualization_panel()
        splitter.addWidget(self.viz_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 600])
    
    def create_control_panel(self):
        """Create the control panel with analysis options."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(350)
        
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Polarization Analysis")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Data management section
        data_group = QGroupBox("Polarized Data")
        data_layout = QVBoxLayout(data_group)
        
        load_btn = QPushButton("Load Polarized Spectra")
        load_btn.clicked.connect(self.load_polarized_spectra)
        # Apply styling directly with a test background color to verify it works
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #e6f3ff;
                border: 2px solid #4dabf7;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
                color: #1971c2;
            }
            QPushButton:hover {
                background-color: #d0ebff;
                border-color: #339af0;
            }
            QPushButton:pressed {
                background-color: #a5d8ff;
                border-color: #228be6;
            }
        """)
        data_layout.addWidget(load_btn)
        
        generate_btn = QPushButton("Generate from Database")
        generate_btn.clicked.connect(self.generate_polarized_spectra)
        # Apply a different test style to verify styling works
        generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #fff0e6;
                border: 2px solid #fd7e14;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
                color: #d63384;
            }
            QPushButton:hover {
                background-color: #ffe8cc;
                border-color: #e8590c;
            }
            QPushButton:pressed {
                background-color: #ffd8a8;
                border-color: #d63384;
            }
        """)
        data_layout.addWidget(generate_btn)
        
        clear_btn = QPushButton("Clear Data")
        clear_btn.clicked.connect(self.clear_polarization_data)
        self.apply_flat_rounded_style(clear_btn)
        data_layout.addWidget(clear_btn)
        
        # Status label
        self.data_status_label = QLabel("No data loaded")
        self.data_status_label.setStyleSheet("color: gray; font-style: italic;")
        data_layout.addWidget(self.data_status_label)
        
        layout.addWidget(data_group)
        
        # Configuration section
        config_group = QGroupBox("Measurement Configuration")
        config_layout = QFormLayout(config_group)
        
        # Scattering geometry
        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems(["Backscattering", "Right-angle", "Forward"])
        self.geometry_combo.currentTextChanged.connect(self.on_geometry_changed)
        config_layout.addRow("Scattering Geometry:", self.geometry_combo)
        
        # Crystal system
        self.crystal_system_combo = QComboBox()
        self.crystal_system_combo.addItems([
            "Cubic", "Tetragonal", "Orthorhombic", "Hexagonal", 
            "Trigonal", "Monoclinic", "Triclinic", "Unknown"
        ])
        self.crystal_system_combo.currentTextChanged.connect(self.on_crystal_system_changed)
        config_layout.addRow("Crystal System:", self.crystal_system_combo)
        
        layout.addWidget(config_group)
        
        # Analysis section
        analysis_group = QGroupBox("Tensor Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        depol_btn = QPushButton("Calculate Depolarization Ratios")
        depol_btn.clicked.connect(self.calculate_depolarization_ratios)
        self.apply_flat_rounded_style(depol_btn)
        analysis_layout.addWidget(depol_btn)
        
        tensor_btn = QPushButton("Determine Raman Tensors")
        tensor_btn.clicked.connect(self.determine_raman_tensors)
        self.apply_flat_rounded_style(tensor_btn)
        analysis_layout.addWidget(tensor_btn)
        
        symmetry_btn = QPushButton("Classify Symmetries")
        symmetry_btn.clicked.connect(self.classify_symmetries)
        self.apply_flat_rounded_style(symmetry_btn)
        analysis_layout.addWidget(symmetry_btn)
        
        angular_btn = QPushButton("Angular Dependence")
        angular_btn.clicked.connect(self.analyze_angular_dependence)
        self.apply_flat_rounded_style(angular_btn)
        analysis_layout.addWidget(angular_btn)
        
        layout.addWidget(analysis_group)
        
        # Tensor display section
        tensor_group = QGroupBox("Tensor Elements")
        tensor_layout = QVBoxLayout(tensor_group)
        
        self.tensor_table = QTableWidget()
        self.tensor_table.setMaximumHeight(200)
        tensor_layout.addWidget(self.tensor_table)
        
        show_tensors_btn = QPushButton("Show Detailed Tensors")
        show_tensors_btn.clicked.connect(self.show_tensor_details)
        self.apply_flat_rounded_style(show_tensors_btn)
        tensor_layout.addWidget(show_tensors_btn)
        
        layout.addWidget(tensor_group)
        
        # Export section
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        self.apply_flat_rounded_style(export_btn)
        export_layout.addWidget(export_btn)
        
        layout.addWidget(export_group)
        
        # Add stretch
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Create the visualization panel with plots."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Plot type selection
        plot_control_layout = QHBoxLayout()
        
        plot_control_layout.addWidget(QLabel("Plot Type:"))
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Polarized Spectra",
            "Depolarization Ratios", 
            "Tensor Elements",
            "Angular Dependence",
            "Polar Plot"
        ])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)
        plot_control_layout.addWidget(self.plot_type_combo)
        
        plot_control_layout.addStretch()
        
        update_btn = QPushButton("Update Plot")
        update_btn.clicked.connect(self.update_plot)
        self.apply_flat_rounded_style(update_btn)
        plot_control_layout.addWidget(update_btn)
        
        layout.addLayout(plot_control_layout)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Initialize empty plot
        self.ax = self.figure.add_subplot(111)
        self.ax.text(0.5, 0.5, 'Load polarized spectra to begin analysis', 
                    transform=self.ax.transAxes, ha='center', va='center',
                    fontsize=12, alpha=0.6)
        self.canvas.draw()
        
        return panel
    
    def setup_connections(self):
        """Setup signal connections."""
        pass
    
    def apply_flat_rounded_style(self, button):
        """Apply flat rounded styling to a button."""
        # Set a unique object name to make styling more specific
        if not button.objectName():
            button.setObjectName(f"styled_button_{id(button)}")
        
        # Apply the stylesheet directly - this should override most defaults
        style = """
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 500;
                color: #333333;
                font-size: 12px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border-color: #bbb;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
                border-color: #999;
            }
            QPushButton:disabled {
                background-color: #f8f8f8;
                color: #999999;
                border-color: #e0e0e0;
            }
        """
        button.setStyleSheet(style)
        
        # Force the button to use the new styling
        button.setFlat(False)
        button.update()  # Force a repaint
        
        # Debug: Print to confirm styling is being applied
        print(f"Applied flat rounded styling to button: {button.text()}")
        print(f"Button stylesheet: {button.styleSheet()[:100]}...")
    
    # === Data Loading Methods ===
    
    def load_polarized_spectra(self):
        """Load polarized spectra from files."""
        try:
            # Import the dialog from polarization_dialogs
            from polarization_ui.polarization_dialogs import PolarizedSpectraLoadingDialog
            
            dialog = PolarizedSpectraLoadingDialog(parent=self)
            dialog.spectra_loaded.connect(self.on_spectra_loaded)
            dialog.exec()
            
        except ImportError:
            # Fallback implementation
            self.load_spectra_fallback()
    
    def load_spectra_fallback(self):
        """Fallback method for loading spectra when dialogs aren't available."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Polarized Spectrum", "", 
            "Text files (*.txt *.dat *.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                data = np.loadtxt(file_path)
                if data.shape[1] >= 2:
                    wavenumbers = data[:, 0]
                    intensities = data[:, 1]
                    
                    # Ask for polarization configuration
                    config, ok = self.get_config_from_user()
                    if ok:
                        pol_data = PolarizationData(wavenumbers, intensities, config)
                        self.polarization_data[config] = pol_data
                        self.update_data_status()
                        self.update_plot()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def get_config_from_user(self):
        """Get polarization configuration from user."""
        from PySide6.QtWidgets import QInputDialog
        
        configs = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
        config, ok = QInputDialog.getItem(
            self, "Polarization Configuration", 
            "Select polarization configuration:", configs, 0, False
        )
        return config, ok
    
    def generate_polarized_spectra(self):
        """Generate synthetic polarized spectra from database."""
        QMessageBox.information(self, "Generate Spectra", 
                              "Spectrum generation from database will be implemented.")
    
    def on_spectra_loaded(self, polarization_data):
        """Handle loaded polarized spectra."""
        self.polarization_data = polarization_data
        self.update_data_status()
        self.update_plot()
    
    def clear_polarization_data(self):
        """Clear all polarization data."""
        self.polarization_data.clear()
        self.raman_tensors.clear()
        self.depolarization_ratios.clear()
        self.tensor_analysis_results.clear()
        self.symmetry_classifications.clear()
        
        self.update_data_status()
        self.update_tensor_table()
        self.update_plot()
    
    def update_data_status(self):
        """Update the data status label."""
        if self.polarization_data:
            configs = list(self.polarization_data.keys())
            status = f"Loaded {len(configs)} configurations: {', '.join(configs)}"
            self.data_status_label.setText(status)
            self.data_status_label.setStyleSheet("color: green;")
        else:
            self.data_status_label.setText("No data loaded")
            self.data_status_label.setStyleSheet("color: gray; font-style: italic;")
    
    # === Configuration Methods ===
    
    def on_geometry_changed(self, geometry):
        """Handle scattering geometry change."""
        self.current_scattering_geometry = geometry
    
    def on_crystal_system_changed(self, system):
        """Handle crystal system change."""
        self.current_crystal_system = system
    
    # === Analysis Methods ===
    
    def calculate_depolarization_ratios(self):
        """Calculate depolarization ratios from polarized data."""
        if not self.polarization_data:
            QMessageBox.warning(self, "No Data", "Please load polarized spectra first.")
            return
        
        try:
            # Need at least two orthogonal configurations
            configs = list(self.polarization_data.keys())
            
            # Try to find parallel and perpendicular configurations
            parallel_config = None
            perpendicular_config = None
            
            # Look for standard configurations
            if 'xx' in configs:
                parallel_config = 'xx'
            elif 'yy' in configs:
                parallel_config = 'yy'
            elif 'zz' in configs:
                parallel_config = 'zz'
            
            if 'xy' in configs:
                perpendicular_config = 'xy'
            elif 'yx' in configs:
                perpendicular_config = 'yx'
            elif 'xz' in configs:
                perpendicular_config = 'xz'
            
            if not parallel_config or not perpendicular_config:
                QMessageBox.warning(self, "Insufficient Data", 
                                  "Need both parallel and perpendicular polarization configurations.")
                return
            
            # Get data
            parallel_data = self.polarization_data[parallel_config]
            perpendicular_data = self.polarization_data[perpendicular_config]
            
            # Interpolate to common wavenumber grid
            wavenumbers = parallel_data.wavenumbers
            I_parallel = parallel_data.intensities
            I_perpendicular = np.interp(wavenumbers, perpendicular_data.wavenumbers, 
                                      perpendicular_data.intensities)
            
            # Calculate depolarization ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = I_perpendicular / I_parallel
                ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Store results
            self.depolarization_ratios = {
                'wavenumbers': wavenumbers,
                'I_parallel': I_parallel,
                'I_perpendicular': I_perpendicular,
                'ratio': ratio,
                'parallel_config': parallel_config,
                'perpendicular_config': perpendicular_config
            }
            
            QMessageBox.information(self, "Success", "Depolarization ratios calculated successfully.")
            self.update_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error calculating depolarization ratios: {str(e)}")
    
    def determine_raman_tensors(self):
        """Determine Raman tensor elements from polarized measurements."""
        if not self.polarization_data:
            QMessageBox.warning(self, "No Data", "Please load polarized spectra first.")
            return
        
        try:
            # Get tensor analysis results
            results = determine_raman_tensor_from_data(
                self.polarization_data,
                self.current_crystal_system,
                self.current_scattering_geometry
            )
            
            # Store results
            self.raman_tensors = {
                'wavenumbers': next(iter(self.polarization_data.values())).wavenumbers,
                'tensor_elements': results['theoretical_intensities'],
                'configurations': list(self.polarization_data.keys()),
                'crystal_system': self.current_crystal_system,
                'best_fit_symmetry': results['best_fit_symmetry'],
                'best_fit_tensor': results['best_fit_tensor'],
                'fit_error': results['fit_error'],
                'timestamp': datetime.now()
            }
            
            # Update tensor table
            self.update_tensor_table()
            
            # Show results
            QMessageBox.information(
                self, 
                "Success", 
                f"Raman tensor analysis completed.\nBest fit symmetry: {results['best_fit_symmetry']}\nFit error: {results['fit_error']:.2f}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error determining Raman tensors: {str(e)}")
    
    def classify_symmetries(self):
        """Classify vibrational modes based on polarization behavior."""
        if not self.depolarization_ratios:
            QMessageBox.warning(self, "No Data", "Please calculate depolarization ratios first.")
            return
        
        try:
            # Show symmetry classification dialog
            dialog = SymmetryClassificationDialog(
                self.depolarization_ratios, 
                self.current_crystal_system,
                self.current_scattering_geometry,
                parent=self
            )
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in symmetry classification: {str(e)}")
    
    def analyze_angular_dependence(self):
        """Analyze angular dependence of Raman intensities."""
        QMessageBox.information(self, "Angular Analysis", 
                              "Angular dependence analysis will be implemented.")
    
    # === Visualization Methods ===
    
    def update_plot(self):
        """Update the main plot based on selected type."""
        if not hasattr(self, 'ax'):
            return
        
        self.ax.clear()
        
        plot_type = self.plot_type_combo.currentText()
        
        if plot_type == "Polarized Spectra":
            self.plot_polarized_spectra()
        elif plot_type == "Depolarization Ratios":
            self.plot_depolarization_ratios()
        elif plot_type == "Tensor Elements":
            self.plot_tensor_elements()
        elif plot_type == "Angular Dependence":
            self.plot_angular_dependence()
        elif plot_type == "Polar Plot":
            self.plot_polar_diagram()
        else:
            self.ax.text(0.5, 0.5, f'{plot_type} not implemented yet', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.6)
        
        self.canvas.draw()
    
    def plot_polarized_spectra(self):
        """Plot all loaded polarized spectra."""
        if not self.polarization_data:
            self.ax.text(0.5, 0.5, 'Load polarized spectra to display', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.6)
            return
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (config, data) in enumerate(self.polarization_data.items()):
            color = colors[i % len(colors)]
            self.ax.plot(data.wavenumbers, data.intensities, 
                        color=color, linewidth=2, label=f"{config.upper()}", alpha=0.8)
        
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Polarized Raman Spectra")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
    
    def plot_depolarization_ratios(self):
        """Plot depolarization ratios."""
        if not self.depolarization_ratios:
            self.ax.text(0.5, 0.5, 'Calculate depolarization ratios first', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.6)
            return
        
        wavenumbers = self.depolarization_ratios['wavenumbers']
        ratios = self.depolarization_ratios['ratio']
        
        self.ax.plot(wavenumbers, ratios, 'b-', linewidth=2, label='ρ = I_⊥ / I_∥')
        
        # Add reference lines
        self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Fully polarized')
        self.ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Fully depolarized (theory)')
        self.ax.axhline(y=0.1, color='green', linestyle=':', alpha=0.5, label='Polarized threshold')
        
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Depolarization Ratio")
        self.ax.set_title("Depolarization Ratios")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(0, 1)
    
    def plot_tensor_elements(self):
        """Plot Raman tensor elements."""
        if not self.raman_tensors:
            self.ax.text(0.5, 0.5, 'Calculate Raman tensors first', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.6)
            return
        
        wavenumbers = self.raman_tensors['wavenumbers']
        tensor_elements = self.raman_tensors['tensor_elements']
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (element, intensities) in enumerate(tensor_elements.items()):
            color = colors[i % len(colors)]
            self.ax.plot(wavenumbers, intensities, 
                        color=color, linewidth=2, label=element, alpha=0.8)
        
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Tensor Element Intensity")
        self.ax.set_title("Raman Tensor Elements")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
    
    def plot_angular_dependence(self):
        """Plot angular dependence."""
        self.ax.text(0.5, 0.5, 'Angular dependence analysis not yet implemented', 
                    transform=self.ax.transAxes, ha='center', va='center',
                    fontsize=12, alpha=0.6)
    
    def plot_polar_diagram(self):
        """Plot polar diagram of tensor elements."""
        self.ax.text(0.5, 0.5, 'Polar diagram not yet implemented', 
                    transform=self.ax.transAxes, ha='center', va='center',
                    fontsize=12, alpha=0.6)
    
    # === Tensor Display Methods ===
    
    def update_tensor_table(self):
        """Update the tensor elements table."""
        if not self.raman_tensors:
            self.tensor_table.setRowCount(0)
            self.tensor_table.setColumnCount(0)
            return
        
        tensor_elements = self.raman_tensors['tensor_elements']
        configs = list(tensor_elements.keys())
        
        self.tensor_table.setRowCount(len(configs))
        self.tensor_table.setColumnCount(3)
        self.tensor_table.setHorizontalHeaderLabels(["Configuration", "Max Intensity", "Type"])
        
        for i, config in enumerate(configs):
            intensities = tensor_elements[config]
            
            # Configuration
            self.tensor_table.setItem(i, 0, QTableWidgetItem(config))
            
            # Max intensity
            max_intensity = np.max(intensities)
            self.tensor_table.setItem(i, 1, QTableWidgetItem(f"{max_intensity:.1f}"))
            
            # Type classification
            if config in ['xx', 'yy', 'zz']:
                tensor_type = "Diagonal"
            elif config in ['xy', 'xz', 'yx', 'yz', 'zx', 'zy']:
                tensor_type = "Off-diagonal"
            elif config == 'isotropic':
                tensor_type = "Isotropic"
            elif config == 'anisotropic':
                tensor_type = "Anisotropic"
            else:
                tensor_type = "Other"
            
            self.tensor_table.setItem(i, 2, QTableWidgetItem(tensor_type))
        
        self.tensor_table.resizeColumnsToContents()
    
    def show_tensor_details(self):
        """Show detailed tensor information in a dialog."""
        if not self.raman_tensors:
            QMessageBox.warning(self, "No Data", "Please calculate Raman tensors first.")
            return
        
        dialog = TensorDetailsDialog(self.raman_tensors, parent=self)
        dialog.exec()
    
    # === Export Methods ===
    
    def export_results(self):
        """Export analysis results to file."""
        if not self.polarization_data and not self.raman_tensors:
            QMessageBox.warning(self, "No Data", "No data to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Polarization Results", "", 
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.save_results_to_file(file_path)
                QMessageBox.information(self, "Success", "Results exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")
    
    def save_results_to_file(self, file_path):
        """Save results to file."""
        with open(file_path, 'w') as f:
            f.write("POLARIZATION ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Crystal System: {self.current_crystal_system}\n")
            f.write(f"Scattering Geometry: {self.current_scattering_geometry}\n\n")
            
            # Polarization data summary
            if self.polarization_data:
                f.write("LOADED POLARIZATION CONFIGURATIONS:\n")
                f.write("-" * 35 + "\n")
                for config, data in self.polarization_data.items():
                    f.write(f"{config}: {len(data.wavenumbers)} points, ")
                    f.write(f"range {np.min(data.wavenumbers):.1f}-{np.max(data.wavenumbers):.1f} cm⁻¹\n")
                f.write("\n")
            
            # Tensor analysis results
            if self.raman_tensors:
                f.write("RAMAN TENSOR ELEMENTS:\n")
                f.write("-" * 22 + "\n")
                tensor_elements = self.raman_tensors['tensor_elements']
                for config, intensities in tensor_elements.items():
                    max_int = np.max(intensities)
                    f.write(f"{config}: Max intensity = {max_int:.1f}\n")
                f.write("\n")
            
            # Depolarization ratios
            if self.depolarization_ratios:
                f.write("DEPOLARIZATION RATIO ANALYSIS:\n")
                f.write("-" * 31 + "\n")
                ratios = self.depolarization_ratios['ratio']
                f.write(f"Average depolarization ratio: {np.mean(ratios):.3f}\n")
                f.write(f"Range: {np.min(ratios):.3f} - {np.max(ratios):.3f}\n\n")


class SymmetryClassificationDialog(QDialog):
    """Dialog for displaying symmetry classification results."""
    
    def __init__(self, depolarization_data, crystal_system, scattering_geometry, parent=None):
        super().__init__(parent)
        
        self.depol_data = depolarization_data
        self.crystal_system = crystal_system
        self.scattering_geometry = scattering_geometry
        
        self.init_ui()
        self.generate_classification()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Symmetry Classification")
        self.setModal(True)
        self.resize(700, 600)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Symmetry Classification Analysis")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Text widget for results
        self.text_widget = QTextEdit()
        self.text_widget.setReadOnly(True)
        self.text_widget.setFont(QFont("Courier", 10))
        layout.addWidget(self.text_widget)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 500;
                color: #333333;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border-color: #bbb;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
                border-color: #999;
            }
            QPushButton:disabled {
                background-color: #f8f8f8;
                color: #999999;
                border-color: #e0e0e0;
            }
        """)
        layout.addWidget(close_btn)
    
    def generate_classification(self):
        """Generate symmetry classification report."""
        report = "SYMMETRY CLASSIFICATION ANALYSIS\n"
        report += "=" * 50 + "\n\n"
        report += f"Crystal System: {self.crystal_system}\n"
        report += f"Scattering Geometry: {self.scattering_geometry}\n\n"
        
        # Analyze depolarization ratios
        wavenumbers = self.depol_data['wavenumbers']
        ratios = self.depol_data['ratio']
        I_parallel = self.depol_data['I_parallel']
        
        # Find peaks for analysis
        peaks, _ = find_peaks(I_parallel, prominence=np.max(I_parallel) * 0.1, distance=20)
        
        report += f"Found {len(peaks)} significant peaks for analysis:\n\n"
        
        for i, peak_idx in enumerate(peaks):
            wavenumber = wavenumbers[peak_idx]
            ratio = ratios[peak_idx]
            
            # Classify based on depolarization ratio
            if ratio < 0.1:
                symmetry = "Totally symmetric (A1, A1g)"
                polarized = "Polarized"
            elif 0.1 <= ratio < 0.5:
                symmetry = "Partially polarized"
                polarized = "Partially polarized"
            elif 0.5 <= ratio < 0.75:
                symmetry = "Depolarized (E, T2, etc.)"
                polarized = "Depolarized"
            else:
                symmetry = "Highly depolarized"
                polarized = "Highly depolarized"
            
            report += f"Peak {i+1}: {wavenumber:.1f} cm⁻¹\n"
            report += f"  Depolarization ratio: {ratio:.3f}\n"
            report += f"  Classification: {symmetry}\n"
            report += f"  Polarization: {polarized}\n\n"
        
        # Add theoretical background
        report += "\nTHEORETICAL BACKGROUND:\n"
        report += "-" * 30 + "\n"
        report += "Depolarization ratio ρ = I_⊥ / I_∥\n\n"
        report += "For different symmetries:\n"
        report += "• A1 (totally symmetric): ρ ≈ 0 (polarized)\n"
        report += "• E (doubly degenerate): ρ = 3/4 (depolarized)\n"
        report += "• T2 (triply degenerate): ρ = 3/4 (depolarized)\n\n"
        
        if self.crystal_system != "Unknown":
            report += f"Expected symmetries for {self.crystal_system} system:\n"
            symmetries = self.get_expected_symmetries()
            for sym in symmetries:
                report += f"• {sym}\n"
        
        self.text_widget.setPlainText(report)
    
    def get_expected_symmetries(self):
        """Get expected symmetries for crystal system."""
        symmetries = {
            "Cubic": ["A1g", "Eg", "T1g", "T2g", "A1u", "Eu", "T1u", "T2u"],
            "Tetragonal": ["A1g", "A2g", "B1g", "B2g", "Eg", "A1u", "A2u", "B1u", "B2u", "Eu"],
            "Orthorhombic": ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
            "Hexagonal": ["A1g", "A2g", "B1g", "B2g", "E1g", "E2g", "A1u", "A2u", "B1u", "B2u", "E1u", "E2u"],
            "Trigonal": ["A1g", "A2g", "Eg", "A1u", "A2u", "Eu"],
            "Monoclinic": ["Ag", "Bg", "Au", "Bu"],
            "Triclinic": ["Ag", "Au"]
        }
        return symmetries.get(self.crystal_system, ["Unknown"])


class TensorDetailsDialog(QDialog):
    """Dialog for displaying detailed tensor information."""
    
    def __init__(self, tensor_data, parent=None):
        super().__init__(parent)
        
        self.tensor_data = tensor_data
        
        self.init_ui()
        self.populate_data()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Detailed Tensor Information")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Raman Tensor Elements - Detailed View")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create tab widget for different views
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Matrix view tab
        matrix_tab = self.create_matrix_tab()
        tab_widget.addTab(matrix_tab, "Tensor Matrix")
        
        # Elements tab
        elements_tab = self.create_elements_tab()
        tab_widget.addTab(elements_tab, "Individual Elements")
        
        # Analysis tab
        analysis_tab = self.create_analysis_tab()
        tab_widget.addTab(analysis_tab, "Analysis")
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 500;
                color: #333333;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border-color: #bbb;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
                border-color: #999;
            }
            QPushButton:disabled {
                background-color: #f8f8f8;
                color: #999999;
                border-color: #e0e0e0;
            }
        """)
        layout.addWidget(close_btn)
    
    def create_matrix_tab(self):
        """Create tensor matrix view tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        info_label = QLabel("Raman Tensor Matrix Elements (at representative frequency)")
        info_label.setFont(QFont("Arial", 12))
        layout.addWidget(info_label)
        
        # Matrix table
        self.matrix_table = QTableWidget(3, 3)
        self.matrix_table.setHorizontalHeaderLabels(['x', 'y', 'z'])
        self.matrix_table.setVerticalHeaderLabels(['x', 'y', 'z'])
        
        # Set fixed size
        self.matrix_table.setFixedSize(300, 200)
        
        layout.addWidget(self.matrix_table)
        layout.addStretch()
        
        return widget
    
    def create_elements_tab(self):
        """Create individual elements tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Table for all elements
        self.elements_table = QTableWidget()
        layout.addWidget(self.elements_table)
        
        return widget
    
    def create_analysis_tab(self):
        """Create analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Analysis text
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.analysis_text)
        
        return widget
    
    def populate_data(self):
        """Populate the dialog with tensor data."""
        tensor_elements = self.tensor_data['tensor_elements']
        wavenumbers = self.tensor_data['wavenumbers']
        
        # Populate matrix table (using middle frequency)
        mid_idx = len(wavenumbers) // 2
        
        matrix_elements = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        for i in range(3):
            for j in range(3):
                element = matrix_elements[i*3 + j]
                if element in tensor_elements:
                    value = tensor_elements[element][mid_idx]
                    item = QTableWidgetItem(f"{value:.2f}")
                else:
                    item = QTableWidgetItem("0.00")
                self.matrix_table.setItem(i, j, item)
        
        # Populate elements table
        self.elements_table.setRowCount(len(tensor_elements))
        self.elements_table.setColumnCount(4)
        self.elements_table.setHorizontalHeaderLabels([
            "Element", "Max Intensity", "Min Intensity", "Type"
        ])
        
        for i, (element, intensities) in enumerate(tensor_elements.items()):
            self.elements_table.setItem(i, 0, QTableWidgetItem(element))
            self.elements_table.setItem(i, 1, QTableWidgetItem(f"{np.max(intensities):.1f}"))
            self.elements_table.setItem(i, 2, QTableWidgetItem(f"{np.min(intensities):.1f}"))
            
            # Classify element type
            if element in ['xx', 'yy', 'zz']:
                elem_type = "Diagonal"
            elif element in ['xy', 'xz', 'yx', 'yz', 'zx', 'zy']:
                elem_type = "Off-diagonal"
            else:
                elem_type = "Derived"
            
            self.elements_table.setItem(i, 3, QTableWidgetItem(elem_type))
        
        self.elements_table.resizeColumnsToContents()
        
        # Populate analysis
        analysis_text = "TENSOR ANALYSIS SUMMARY\n"
        analysis_text += "=" * 30 + "\n\n"
        analysis_text += f"Crystal System: {self.tensor_data.get('crystal_system', 'Unknown')}\n"
        analysis_text += f"Number of measured elements: {len(tensor_elements)}\n"
        analysis_text += f"Wavenumber range: {np.min(wavenumbers):.1f} - {np.max(wavenumbers):.1f} cm⁻¹\n\n"
        
        analysis_text += "ELEMENT SUMMARY:\n"
        analysis_text += "-" * 16 + "\n"
        for element, intensities in tensor_elements.items():
            max_int = np.max(intensities)
            analysis_text += f"{element:12s}: Max = {max_int:.1f}\n"
        
        self.analysis_text.setPlainText(analysis_text)


# Export the main widget
__all__ = ['PolarizationAnalysisWidget'] 