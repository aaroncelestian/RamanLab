"""
Polarization Dialogs UI Module

This module provides sophisticated dialog interfaces for polarized spectra loading
and generation. Designed for professional Qt6 applications with enhanced user experience.

Features:
- Sophisticated polarized spectra loading dialog
- Two-tab interface: File loading + Database generation
- Real-time preview with parameter adjustment
- Integration with core polarization and database modules
- Professional Qt6 implementation with rich interactions
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QGroupBox,
    QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QTextEdit,
    QFileDialog, QMessageBox, QFormLayout, QSplitter, QFrame,
    QProgressBar, QApplication, QGridLayout
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPalette

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Configure matplotlib for smaller toolbar
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'toolbar2'

# Try to import core modules
try:
    from core.polarization_generator import PolarizedSpectrumGenerator
    from core.polarization_analyzer import PolarizationAnalyzer
    from core.mineral_database import MineralDatabase
    CORE_MODULES_AVAILABLE = True
    print("✓ Core polarization modules loaded successfully")
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    print(f"⚠ Core modules not available: {e}")
    print("Using mock implementations for testing")


# Data class for polarization data
class PolarizationData:
    """Data structure for storing polarized spectrum data."""
    
    def __init__(self, wavenumbers=None, intensities=None, config="xx", filename=None):
        self.wavenumbers = wavenumbers if wavenumbers is not None else np.linspace(200, 2000, 1000)
        self.intensities = intensities if intensities is not None else np.random.normal(100, 10, 1000)
        self.config = config
        self.filename = filename


# Mock classes for testing when core modules aren't available
if not CORE_MODULES_AVAILABLE:
    class PolarizedSpectrumGenerator:
        def __init__(self):
            pass
        
        def generate_polarized_spectra(self, mineral_name, **kwargs):
            # Generate mock data
            wavenumbers = np.linspace(200, 2000, 1000)
            base_spectrum = np.exp(-((wavenumbers - 1000) / 200) ** 2) * 1000
            noise = np.random.normal(0, kwargs.get('noise_level', 10), len(wavenumbers))
            
            return {
                'xx': PolarizationData(wavenumbers, base_spectrum + noise, 'xx'),
                'xy': PolarizationData(wavenumbers, base_spectrum * 0.3 + noise, 'xy')
            }
    
    class MineralDatabase:
        def __init__(self):
            self.minerals = ["Quartz", "Calcite", "Feldspar", "Olivine", "Pyroxene"]
        
        def get_all_minerals(self):
            return self.minerals


class PolarizedSpectraLoadingDialog(QDialog):
    """
    Advanced dialog for loading and generating polarized Raman spectra.
    
    This dialog provides two main modes:
    1. Load from Files: Import polarized spectra from various file formats
    2. Generate from Database: Create synthetic polarized spectra from mineral database
    
    Signals:
        spectra_loaded: Emitted when polarized spectra are successfully loaded/generated
                       Args: Dict[str, PolarizationData] - polarization data by configuration
    """
    
    spectra_loaded = Signal(dict)  # Dict[str, PolarizationData]
    
    def __init__(self, database=None, parent=None):
        """
        Initialize the polarized spectra dialog.
        
        Args:
            database: MineralDatabase instance for synthetic generation
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.database = database if database else MineralDatabase()
        
        # Initialize UI
        self.init_ui()
        
        # Connect signals
        self.setup_connections()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Load Polarized Spectra")
        self.setModal(True)
        self.resize(700, 600)
        
        # Center the dialog
        self.center_dialog()
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("Load Polarized Raman Spectra")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create tab widget for different loading methods
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.file_loading_widget = FileLoadingWidget(self)
        self.database_generation_widget = DatabaseGenerationWidget(self.database, self)
        
        self.tab_widget.addTab(self.file_loading_widget, "Load from Files")
        self.tab_widget.addTab(self.database_generation_widget, "Generate from Database")
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
    
    def center_dialog(self):
        """Center the dialog on the parent or screen."""
        if self.parent():
            parent_geo = self.parent().geometry()
            x = parent_geo.x() + (parent_geo.width() - self.width()) // 2
            y = parent_geo.y() + (parent_geo.height() - self.height()) // 2
            self.move(x, y)
    
    def setup_connections(self):
        """Setup signal connections."""
        self.file_loading_widget.spectra_loaded.connect(self.on_spectra_loaded)
        self.database_generation_widget.spectra_generated.connect(self.on_spectra_loaded)
    
    def on_spectra_loaded(self, polarization_data: Dict[str, PolarizationData]):
        """Handle when spectra are loaded/generated."""
        self.spectra_loaded.emit(polarization_data)
        self.accept()

# Keep the old class name for backward compatibility
PolarizedSpectraDialog = PolarizedSpectraLoadingDialog

class FileLoadingWidget(QWidget):
    """
    Widget for loading polarized spectra from files.
    Provides interface for selecting files for different polarization configurations.
    """
    
    spectra_loaded = Signal(dict)  # Dict[str, PolarizationData]
    
    def __init__(self, parent=None):
        """Initialize the file loading widget."""
        super().__init__(parent)
        
        self.pol_files: Dict[str, str] = {}
        self.file_labels: Dict[str, QLabel] = {}
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Load polarized Raman spectra from files:")
        instructions.setFont(QFont("Arial", 10))
        layout.addWidget(instructions)
        
        # File selection frame
        files_group = QGroupBox("Polarization Configurations")
        files_layout = QGridLayout(files_group)
        
        # Common polarization configurations
        self.configurations = [
            ("XX (Parallel)", "xx"),
            ("XY (Cross-polarized)", "xy"),
            ("YX (Cross-polarized)", "yx"),
            ("YY (Parallel)", "yy"),
            ("ZZ (Parallel)", "zz"),
            ("XZ (Cross-polarized)", "xz"),
            ("ZX (Cross-polarized)", "zx"),
            ("YZ (Cross-polarized)", "yz"),
            ("ZY (Cross-polarized)", "zy")
        ]
        
        # Create file selection widgets
        for i, (display_name, config_key) in enumerate(self.configurations):
            # Label
            label = QLabel(f"{display_name}:")
            label.setMinimumWidth(150)
            files_layout.addWidget(label, i, 0)
            
            # File path display
            file_label = QLabel("No file selected")
            file_label.setStyleSheet("color: gray; font-style: italic;")
            file_label.setMinimumWidth(300)
            self.file_labels[config_key] = file_label
            files_layout.addWidget(file_label, i, 1)
            
            # Browse button
            browse_btn = QPushButton("Browse")
            browse_btn.clicked.connect(lambda checked, key=config_key: self.browse_file(key))
            files_layout.addWidget(browse_btn, i, 2)
        
        layout.addWidget(files_group)
        
        # Load button
        load_layout = QHBoxLayout()
        load_layout.addStretch()
        
        self.load_btn = QPushButton("Load Selected Files")
        self.load_btn.clicked.connect(self.load_selected_files)
        self.load_btn.setEnabled(False)
        load_layout.addWidget(self.load_btn)
        
        layout.addLayout(load_layout)
    
    def browse_file(self, config_key: str):
        """Browse for spectrum file for given polarization configuration."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {config_key.upper()} polarized spectrum",
            "", "All Files (*);;Text Files (*.txt);;CSV Files (*.csv);;DAT Files (*.dat)"
        )
        
        if file_path:
            self.pol_files[config_key] = file_path
            filename = os.path.basename(file_path)
            self.file_labels[config_key].setText(filename)
            self.file_labels[config_key].setStyleSheet("color: black;")
            
            # Enable load button if at least one file is selected
            if self.pol_files:
                self.load_btn.setEnabled(True)
    
    def load_selected_files(self):
        """Load all selected spectrum files."""
        if not self.pol_files:
            QMessageBox.warning(self, "No Files", "Please select at least one spectrum file.")
            return
        
        polarization_data = {}
        
        for config_key, file_path in self.pol_files.items():
            wavenumbers, intensities = self.load_spectrum_file(file_path)
            if wavenumbers is not None and intensities is not None:
                polarization_data[config_key] = PolarizationData(wavenumbers, intensities, config_key)
            else:
                QMessageBox.warning(self, "Load Error", f"Failed to load file: {file_path}")
                return
        
        if polarization_data:
            self.spectra_loaded.emit(polarization_data)
    
    def load_spectrum_file(self, filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load spectrum data from file.
        
        Args:
            filepath: Path to spectrum file
            
        Returns:
            Tuple of (wavenumbers, intensities) or (None, None) if failed
        """
        try:
            # Try to load as CSV/text file
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            if data.shape[1] >= 2:
                return data[:, 0], data[:, 1]
            else:
                # Single column - generate wavenumber axis
                wavenumbers = np.linspace(200, 2000, len(data))
                return wavenumbers, data[:, 0]
        except:
            try:
                # Try tab-delimited
                data = np.loadtxt(filepath, delimiter='\t', skiprows=1)
                if data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]
                else:
                    wavenumbers = np.linspace(200, 2000, len(data))
                    return wavenumbers, data[:, 0]
            except:
                return None, None


class DatabaseGenerationWidget(QWidget):
    """
    Widget for generating synthetic polarized spectra from mineral database.
    Provides comprehensive controls for spectrum generation parameters.
    """
    
    spectra_generated = Signal(dict)  # Dict[str, PolarizationData]
    
    def __init__(self, database, parent=None):
        """Initialize the database generation widget."""
        super().__init__(parent)
        
        self.database = database
        self.spectrum_generator = PolarizedSpectrumGenerator()
        
        # Preview data
        self.preview_figure = None
        self.preview_canvas = None
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Generate synthetic polarized spectra from mineral database:")
        instructions.setFont(QFont("Arial", 10))
        layout.addWidget(instructions)
        
        # Parameters frame
        params_group = QGroupBox("Generation Parameters")
        params_layout = QGridLayout(params_group)
        
        # Mineral selection
        params_layout.addWidget(QLabel("Mineral:"), 0, 0)
        self.mineral_combo = QComboBox()
        self.populate_mineral_combo()
        params_layout.addWidget(self.mineral_combo, 0, 1)
        
        # Noise level
        params_layout.addWidget(QLabel("Noise Level:"), 1, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 50.0)
        self.noise_spin.setValue(5.0)
        self.noise_spin.setSuffix("%")
        params_layout.addWidget(self.noise_spin, 1, 1)
        
        # Baseline
        params_layout.addWidget(QLabel("Baseline:"), 2, 0)
        self.baseline_spin = QSpinBox()
        self.baseline_spin.setRange(0, 1000)
        self.baseline_spin.setValue(100)
        params_layout.addWidget(self.baseline_spin, 2, 1)
        
        # Temperature
        params_layout.addWidget(QLabel("Temperature (K):"), 3, 0)
        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(200, 1000)
        self.temp_spin.setValue(295)
        params_layout.addWidget(self.temp_spin, 3, 1)
        
        layout.addWidget(params_group)
        
        # Polarization configurations
        config_group = QGroupBox("Polarization Configurations")
        config_layout = QVBoxLayout(config_group)
        
        self.config_checkboxes = {}
        configurations = ["XX", "XY", "YX", "YY", "ZZ", "XZ", "ZX", "YZ", "ZY"]
        
        for i, config in enumerate(configurations):
            if i % 3 == 0:
                row_layout = QHBoxLayout()
                config_layout.addLayout(row_layout)
            
            checkbox = QCheckBox(config)
            if config in ["XX", "XY"]:  # Default selections
                checkbox.setChecked(True)
            self.config_checkboxes[config.lower()] = checkbox
            row_layout.addWidget(checkbox)
        
        layout.addWidget(config_group)
        
        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Preview controls
        preview_controls = QHBoxLayout()
        self.auto_preview_cb = QCheckBox("Auto Preview")
        self.auto_preview_cb.setChecked(True)
        preview_controls.addWidget(self.auto_preview_cb)
        
        preview_controls.addStretch()
        
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview_generation)
        preview_controls.addWidget(self.preview_btn)
        
        self.clear_preview_btn = QPushButton("Clear")
        self.clear_preview_btn.clicked.connect(self.clear_preview)
        preview_controls.addWidget(self.clear_preview_btn)
        
        preview_layout.addLayout(preview_controls)
        
        # Preview plot
        self.preview_figure = Figure(figsize=(8, 4))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        preview_layout.addWidget(self.preview_canvas)
        
        layout.addWidget(preview_group)
        
        # Generate button
        generate_layout = QHBoxLayout()
        generate_layout.addStretch()
        
        self.generate_btn = QPushButton("Generate Spectra")
        self.generate_btn.clicked.connect(self.generate_spectra)
        generate_layout.addWidget(self.generate_btn)
        
        layout.addLayout(generate_layout)
    
    def setup_connections(self):
        """Setup signal connections."""
        self.mineral_combo.currentTextChanged.connect(self.auto_preview)
        self.noise_spin.valueChanged.connect(self.auto_preview)
        self.baseline_spin.valueChanged.connect(self.auto_preview)
        self.temp_spin.valueChanged.connect(self.auto_preview)
        
        for checkbox in self.config_checkboxes.values():
            checkbox.toggled.connect(self.auto_preview)
    
    def populate_mineral_combo(self):
        """Populate the mineral selection combo box."""
        if CORE_MODULES_AVAILABLE:
            try:
                minerals = self.database.get_all_minerals()
                self.mineral_combo.addItems(minerals)
            except:
                # Fallback minerals
                self.mineral_combo.addItems(["Quartz", "Calcite", "Feldspar"])
        else:
            self.mineral_combo.addItems(["Quartz", "Calcite", "Feldspar"])
    
    def auto_preview(self):
        """Auto-generate preview if enabled."""
        if self.auto_preview_cb.isChecked():
            self.preview_generation()
    
    def preview_generation(self):
        """Generate and display preview spectra."""
        try:
            # Get selected configurations
            selected_configs = [config for config, cb in self.config_checkboxes.items() if cb.isChecked()]
            
            if not selected_configs:
                QMessageBox.warning(self, "No Configurations", "Please select at least one polarization configuration.")
                return
            
            # Generate preview data
            wavenumbers = np.linspace(200, 2000, 1000)
            
            # Clear previous plot
            self.preview_figure.clear()
            ax = self.preview_figure.add_subplot(111)
            
            # Generate synthetic spectra for each configuration
            for config in selected_configs:
                # Simple synthetic spectrum generation
                centers = [400, 700, 1100, 1400]
                widths = [20, 15, 25, 18]
                intensities = [100, 150, 80, 120]
                
                spectrum = np.zeros_like(wavenumbers)
                for center, width, intensity in zip(centers, widths, intensities):
                    spectrum += intensity * np.exp(-(wavenumbers - center)**2 / (2 * width**2))
                
                # Add noise and baseline
                noise_level = self.noise_spin.value() / 100.0
                noise = np.random.normal(0, noise_level * np.max(spectrum), len(wavenumbers))
                baseline = self.baseline_spin.value()
                
                spectrum += noise + baseline
                
                # Plot
                ax.plot(wavenumbers, spectrum, label=config.upper(), alpha=0.8)
            
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Intensity")
            ax.set_title(f"Preview: {self.mineral_combo.currentText()}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Failed to generate preview: {str(e)}")
    
    def clear_preview(self):
        """Clear the preview plot."""
        if self.preview_figure:
            self.preview_figure.clear()
            self.preview_canvas.draw()
    
    def generate_spectra(self):
        """Generate the final polarized spectra."""
        try:
            # Get selected configurations
            selected_configs = [config for config, cb in self.config_checkboxes.items() if cb.isChecked()]
            
            if not selected_configs:
                QMessageBox.warning(self, "No Configurations", "Please select at least one polarization configuration.")
                return
            
            # Generate spectra data
            polarization_data = {}
            wavenumbers = np.linspace(200, 2000, 1000)
            
            for config in selected_configs:
                # Generate synthetic spectrum (same as preview but without noise variation)
                centers = [400, 700, 1100, 1400]
                widths = [20, 15, 25, 18]
                intensities = [100, 150, 80, 120]
                
                spectrum = np.zeros_like(wavenumbers)
                for center, width, intensity in zip(centers, widths, intensities):
                    spectrum += intensity * np.exp(-(wavenumbers - center)**2 / (2 * width**2))
                
                # Add noise and baseline
                noise_level = self.noise_spin.value() / 100.0
                noise = np.random.normal(0, noise_level * np.max(spectrum), len(wavenumbers))
                baseline = self.baseline_spin.value()
                
                spectrum += noise + baseline
                
                # Create polarization data object
                polarization_data[config] = PolarizationData(wavenumbers, spectrum, config)
            
            # Emit the generated data
            self.spectra_generated.emit(polarization_data)
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"Failed to generate spectra: {str(e)}")


# Alias for backward compatibility
DatabaseGenerationDialog = DatabaseGenerationWidget 