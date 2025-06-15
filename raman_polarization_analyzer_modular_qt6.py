#!/usr/bin/env python3
"""
Raman Polarization Analyzer - Qt6 Modular Version
Demonstrates how to use the modular structure with separated core functionality.
"""

import sys
import os
import numpy as np
from typing import Dict

# Qt6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QFrame, QLabel, QPushButton, QLineEdit, QComboBox, QListWidget,
    QGroupBox, QSplitter, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QStandardPaths
from PySide6.QtGui import QFont

# Matplotlib Qt6 backend
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from polarization_ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
from matplotlib.figure import Figure

# Import core modules
from core.database import MineralDatabase, infer_crystal_system_from_name
from core.peak_fitting import PeakFitter, PeakData, auto_find_peaks
from core.polarization import (
    PolarizationAnalyzer, 
    PolarizedSpectrumGenerator,
    PolarizationData,
    DepolarizationResult
)

# Import UI modules
from polarization_ui.polarization_dialogs import PolarizedSpectraDialog


class RamanPolarizationAnalyzerModular(QMainWindow):
    """
    Modular Qt6 version of the Raman Polarization Analyzer.
    
    This version demonstrates the benefits of separating core functionality
    into dedicated modules while keeping UI components in the main class.
    """
    
    def __init__(self):
        """Initialize the modular analyzer."""
        super().__init__()
        
        self.setWindowTitle("Raman Polarization Analyzer - Qt6 Modular")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize core modules
        self.database = MineralDatabase()
        self.peak_fitter = PeakFitter()
        self.polarization_analyzer = PolarizationAnalyzer()
        self.spectrum_generator = PolarizedSpectrumGenerator()
        
        # Initialize variables
        self.current_spectrum = None
        self.imported_spectrum = None
        self.selected_peaks = []
        self.fitted_peaks = []
        self.peak_selection_mode = False
        self.selected_reference_mineral = None
        
        # Store plot components
        self.plot_components = {}
        
        # Load database and create UI
        self.init_core_modules()
        self.init_ui()
    
    def init_core_modules(self):
        """Initialize core modules with data."""
        # Load mineral database
        success = self.database.load_database()
        if success:
            stats = self.database.get_statistics()
            print(f"Database loaded: {stats}")
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs demonstrating the modular approach
        self.create_spectrum_analysis_tab()
        self.create_peak_fitting_tab()
        self.create_polarization_analysis_tab()
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def create_spectrum_analysis_tab(self):
        """Create the spectrum analysis tab using modular approach."""
        # Create tab widget
        tab_widget = QWidget()
        self.tab_widget.addTab(tab_widget, "Spectrum Analysis")
        
        # Create layout with splitter
        splitter = QSplitter(Qt.Horizontal)
        tab_layout = QHBoxLayout(tab_widget)
        tab_layout.addWidget(splitter)
        
        # Side panel
        side_panel = QFrame()
        side_panel.setFrameStyle(QFrame.StyledPanel)
        side_panel.setFixedWidth(300)
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Spectrum Analysis")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        load_btn = QPushButton("Load Spectrum")
        load_btn.clicked.connect(self.load_spectrum)
        file_layout.addWidget(load_btn)
        
        side_layout.addWidget(file_group)
        
        # Database search using modular database
        db_group = QGroupBox("Mineral Database Search")
        db_layout = QVBoxLayout(db_group)
        
        search_label = QLabel("Search:")
        db_layout.addWidget(search_label)
        
        self.search_entry = QLineEdit()
        self.search_entry.textChanged.connect(self.on_database_search)
        db_layout.addWidget(self.search_entry)
        
        self.search_results = QListWidget()
        self.search_results.setMaximumHeight(120)
        self.search_results.itemDoubleClicked.connect(self.on_mineral_selected)
        db_layout.addWidget(self.search_results)
        
        import_btn = QPushButton("Import Selected Mineral")
        import_btn.clicked.connect(self.import_selected_mineral)
        db_layout.addWidget(import_btn)
        
        side_layout.addWidget(db_group)
        side_layout.addStretch()
        
        # Content area with plot
        content_area = QFrame()
        content_area.setFrameStyle(QFrame.StyledPanel)
        content_layout = QVBoxLayout(content_area)
        
        # Create matplotlib components
        self.spectrum_fig = Figure(figsize=(8, 6))
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)
        self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, content_area)
        
        content_layout.addWidget(self.spectrum_toolbar)
        content_layout.addWidget(self.spectrum_canvas)
        
        # Add to splitter
        splitter.addWidget(side_panel)
        splitter.addWidget(content_area)
        splitter.setSizes([300, 900])
        
        # Store components
        self.plot_components["Spectrum Analysis"] = {
            'fig': self.spectrum_fig,
            'canvas': self.spectrum_canvas,
            'ax': self.spectrum_ax
        }
        
        # Initialize plot
        self.update_spectrum_plot()
    
    def create_peak_fitting_tab(self):
        """Create the peak fitting tab using modular peak fitting."""
        # Create tab widget
        tab_widget = QWidget()
        self.tab_widget.addTab(tab_widget, "Peak Fitting")
        
        # Create layout with splitter
        splitter = QSplitter(Qt.Horizontal)
        tab_layout = QHBoxLayout(tab_widget)
        tab_layout.addWidget(splitter)
        
        # Side panel
        side_panel = QFrame()
        side_panel.setFrameStyle(QFrame.StyledPanel)
        side_panel.setFixedWidth(300)
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Peak Fitting")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Peak selection controls
        peak_group = QGroupBox("Peak Selection")
        peak_layout = QVBoxLayout(peak_group)
        
        self.toggle_peak_btn = QPushButton("Toggle Peak Selection")
        self.toggle_peak_btn.clicked.connect(self.toggle_peak_selection)
        peak_layout.addWidget(self.toggle_peak_btn)
        
        auto_peak_btn = QPushButton("Auto Find Peaks")
        auto_peak_btn.clicked.connect(self.auto_find_peaks)
        peak_layout.addWidget(auto_peak_btn)
        
        clear_peaks_btn = QPushButton("Clear Peaks")
        clear_peaks_btn.clicked.connect(self.clear_peaks)
        peak_layout.addWidget(clear_peaks_btn)
        
        side_layout.addWidget(peak_group)
        
        # Fitting controls
        fitting_group = QGroupBox("Peak Fitting")
        fitting_layout = QVBoxLayout(fitting_group)
        
        fit_btn = QPushButton("Fit Selected Peaks")
        fit_btn.clicked.connect(self.fit_peaks)
        fitting_layout.addWidget(fit_btn)
        
        side_layout.addWidget(fitting_group)
        side_layout.addStretch()
        
        # Content area with plot
        content_area = QFrame()
        content_area.setFrameStyle(QFrame.StyledPanel)
        content_layout = QVBoxLayout(content_area)
        
        # Create matplotlib components
        self.peak_fig = Figure(figsize=(8, 6))
        self.peak_canvas = FigureCanvas(self.peak_fig)
        self.peak_ax = self.peak_fig.add_subplot(111)
        self.peak_toolbar = NavigationToolbar(self.peak_canvas, content_area)
        
        # Connect click events for peak selection
        self.peak_canvas.mpl_connect('button_press_event', self.on_peak_click)
        
        content_layout.addWidget(self.peak_toolbar)
        content_layout.addWidget(self.peak_canvas)
        
        # Add to splitter
        splitter.addWidget(side_panel)
        splitter.addWidget(content_area)
        splitter.setSizes([300, 900])
        
        # Store components
        self.plot_components["Peak Fitting"] = {
            'fig': self.peak_fig,
            'canvas': self.peak_canvas,
            'ax': self.peak_ax
        }
        
        # Initialize plot
        self.update_peak_plot()

    def create_polarization_analysis_tab(self):
        """Create the polarization analysis tab using the new modular polarization module."""
        # Create tab widget
        tab_widget = QWidget()
        self.tab_widget.addTab(tab_widget, "Polarization Analysis")
        
        # Create layout with splitter
        splitter = QSplitter(Qt.Horizontal)
        tab_layout = QHBoxLayout(tab_widget)
        tab_layout.addWidget(splitter)
        
        # Side panel
        side_panel = QFrame()
        side_panel.setFrameStyle(QFrame.StyledPanel)
        side_panel.setFixedWidth(300)
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Polarization Analysis")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Generate synthetic polarized spectra
        generation_group = QGroupBox("Generate Polarized Spectra")
        generation_layout = QVBoxLayout(generation_group)
        
        # Add sophisticated dialog button
        load_polarized_btn = QPushButton("Load Polarized Spectra")
        load_polarized_btn.setFont(QFont("Arial", 10, QFont.Bold))
        load_polarized_btn.clicked.connect(self.open_polarized_spectra_dialog)
        generation_layout.addWidget(load_polarized_btn)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        generation_layout.addWidget(separator)
        
        # Quick generation section
        quick_label = QLabel("Quick Generation:")
        quick_label.setFont(QFont("Arial", 9))
        generation_layout.addWidget(quick_label)
        
        # Mineral selection
        mineral_label = QLabel("Select Mineral:")
        generation_layout.addWidget(mineral_label)
        
        self.mineral_combo = QComboBox()
        generation_layout.addWidget(self.mineral_combo)
        
        # Populate mineral combo with database minerals
        self.populate_mineral_combo()
        
        # Configuration selection
        config_label = QLabel("Polarization Configurations:")
        generation_layout.addWidget(config_label)
        
        self.config_buttons = {}
        for config in ['xx', 'xy', 'yy', 'zz']:
            btn = QPushButton(f"Generate {config.upper()}")
            btn.clicked.connect(lambda checked, c=config: self.generate_polarized_spectrum(c))
            generation_layout.addWidget(btn)
            self.config_buttons[config] = btn
        
        side_layout.addWidget(generation_group)
        
        # Analysis controls
        analysis_group = QGroupBox("Polarization Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        depol_btn = QPushButton("Calculate Depolarization Ratios")
        depol_btn.clicked.connect(self.calculate_depolarization_ratios)
        analysis_layout.addWidget(depol_btn)
        
        tensor_btn = QPushButton("Analyze Raman Tensors")
        tensor_btn.clicked.connect(self.analyze_raman_tensors)
        analysis_layout.addWidget(tensor_btn)
        
        symmetry_btn = QPushButton("Classify Symmetries")
        symmetry_btn.clicked.connect(self.classify_symmetries)
        analysis_layout.addWidget(symmetry_btn)
        
        clear_pol_btn = QPushButton("Clear Polarization Data")
        clear_pol_btn.clicked.connect(self.clear_polarization_data)
        analysis_layout.addWidget(clear_pol_btn)
        
        side_layout.addWidget(analysis_group)
        side_layout.addStretch()
        
        # Content area with plot
        content_area = QFrame()
        content_area.setFrameStyle(QFrame.StyledPanel)
        content_layout = QVBoxLayout(content_area)
        
        # Create matplotlib components
        self.polarization_fig = Figure(figsize=(8, 6))
        self.polarization_canvas = FigureCanvas(self.polarization_fig)
        self.polarization_ax = self.polarization_fig.add_subplot(111)
        self.polarization_toolbar = NavigationToolbar(self.polarization_canvas, content_area)
        
        content_layout.addWidget(self.polarization_toolbar)
        content_layout.addWidget(self.polarization_canvas)
        
        # Add to splitter
        splitter.addWidget(side_panel)
        splitter.addWidget(content_area)
        splitter.setSizes([300, 900])
        
        # Store components
        self.plot_components["Polarization Analysis"] = {
            'fig': self.polarization_fig,
            'canvas': self.polarization_canvas,
            'ax': self.polarization_ax
        }
        
        # Initialize plot
        self.update_polarization_plot()

    def populate_mineral_combo(self):
        """Populate the mineral selection combo box."""
        self.mineral_combo.clear()
        self.mineral_combo.addItems(self.database.mineral_list[:50])  # Limit to first 50 for demo
    
    def generate_polarized_spectrum(self, config):
        """Generate a polarized spectrum using the modular spectrum generator."""
        try:
            mineral_name = self.mineral_combo.currentText()
            if not mineral_name:
                QMessageBox.warning(self, "No Mineral", "Please select a mineral first.")
                return
            
            # Get mineral data from database
            mineral_data = self.database.get_mineral_data(mineral_name)
            if not mineral_data:
                QMessageBox.warning(self, "Error", f"Could not find data for {mineral_name}")
                return
            
            # Generate polarized spectrum
            wavenumbers, intensities = self.spectrum_generator.generate_polarized_spectrum(
                mineral_data, config, peak_width=5.0, intensity_scale=1.0
            )
            
            # Add to polarization analyzer
            self.polarization_analyzer.add_polarized_spectrum(
                config, wavenumbers, intensities, f"{mineral_name}_{config.upper()}"
            )
            
            # Update plot
            self.update_polarization_plot()
            
            QMessageBox.information(self, "Success", 
                                  f"Generated {config.upper()} spectrum for {mineral_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating spectrum: {str(e)}")
    
    def calculate_depolarization_ratios(self):
        """Calculate depolarization ratios using the modular polarization analyzer."""
        try:
            result = self.polarization_analyzer.calculate_depolarization_ratios()
            self.update_polarization_plot()
            QMessageBox.information(self, "Success", "Depolarization ratios calculated successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error calculating depolarization ratios: {str(e)}")
    
    def analyze_raman_tensors(self):
        """Analyze Raman tensors using the modular polarization analyzer."""
        try:
            result = self.polarization_analyzer.determine_raman_tensors()
            QMessageBox.information(self, "Success", 
                                  f"Tensor analysis completed for {len(result.configurations)} configurations")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing tensors: {str(e)}")
    
    def classify_symmetries(self):
        """Classify symmetries based on polarization behavior."""
        try:
            crystal_system = "Unknown"  # Could be determined from mineral data
            results = self.polarization_analyzer.classify_symmetries(crystal_system)
            
            # Create results dialog
            self.show_symmetry_results(results)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error classifying symmetries: {str(e)}")
    
    def show_symmetry_results(self, results):
        """Display symmetry classification results."""
        result_text = f"SYMMETRY CLASSIFICATION RESULTS\n"
        result_text += "=" * 40 + "\n\n"
        result_text += f"Crystal System: {results['crystal_system']}\n\n"
        
        result_text += f"Found {len(results['peaks'])} peaks:\n\n"
        for peak in results['peaks']:
            result_text += f"Peak {peak['peak_number']}: {peak['wavenumber']:.1f} cm⁻¹\n"
            result_text += f"  Depolarization ratio: {peak['depolarization_ratio']:.3f}\n"
            result_text += f"  Symmetry: {peak['symmetry']}\n"
            result_text += f"  Type: {peak['polarization_type']}\n\n"
        
        result_text += "THEORETICAL BACKGROUND:\n"
        result_text += "-" * 25 + "\n"
        for key, value in results['theoretical_background'].items():
            result_text += f"{key}: {value}\n"
        
        QMessageBox.information(self, "Symmetry Classification Results", result_text)
    
    def clear_polarization_data(self):
        """Clear all polarization data."""
        self.polarization_analyzer.clear_data()
        self.update_polarization_plot()
        QMessageBox.information(self, "Success", "Polarization data cleared.")
    
    def open_polarized_spectra_dialog(self):
        """Open the sophisticated polarized spectra dialog."""
        dialog = PolarizedSpectraDialog(self.database, self)
        dialog.spectra_loaded.connect(self.handle_loaded_polarized_spectra)
        dialog.exec()
    
    def handle_loaded_polarized_spectra(self, polarization_data: Dict[str, PolarizationData]):
        """Handle polarized spectra loaded from the dialog."""
        try:
            # Clear existing data
            self.polarization_analyzer.clear_data()
            
            # Add all loaded spectra to the analyzer
            for config, pol_data in polarization_data.items():
                self.polarization_analyzer.add_polarized_spectrum(
                    config, 
                    pol_data.wavenumbers, 
                    pol_data.intensities, 
                    pol_data.filename
                )
            
            # Update the plot
            self.update_polarization_plot()
            
            # Show success message
            configs = list(polarization_data.keys())
            config_str = ', '.join([c.upper() for c in configs])
            QMessageBox.information(
                self, 
                "Polarized Spectra Loaded", 
                f"Successfully loaded {len(polarization_data)} polarized spectra:\n{config_str}\n\n"
                f"You can now calculate depolarization ratios and analyze symmetries."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error handling loaded spectra: {str(e)}")
    
    def update_polarization_plot(self):
        """Update the polarization analysis plot."""
        if "Polarization Analysis" not in self.plot_components:
            return
        
        components = self.plot_components["Polarization Analysis"]
        ax = components['ax']
        canvas = components['canvas']
        
        # Clear the plot
        ax.clear()
        
        # Check if we have polarization data
        if not self.polarization_analyzer.polarization_data:
            ax.text(0.5, 0.5, 'Generate polarized spectra to begin analysis', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=12, alpha=0.6)
            ax.set_title("Polarization Analysis")
        elif self.polarization_analyzer.depolarization_results:
            # Plot depolarization ratios
            result = self.polarization_analyzer.depolarization_results
            ax.plot(result.wavenumbers, result.ratio, 'b-', linewidth=2, label='ρ = I_⊥ / I_∥')
            
            # Add reference lines
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Fully polarized')
            ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Fully depolarized (theory)')
            ax.axhline(y=0.1, color='green', linestyle=':', alpha=0.5, label='Polarized threshold')
            
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Depolarization Ratio")
            ax.set_title("Depolarization Ratio Analysis")
            ax.set_ylim(0, 1.2)
            ax.legend()
        else:
            # Plot polarized spectra
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (config, data) in enumerate(self.polarization_analyzer.polarization_data.items()):
                color = colors[i % len(colors)]
                ax.plot(data.wavenumbers, data.intensities, 
                       color=color, linewidth=2, label=f"{config.upper()}", alpha=0.8)
            
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title("Polarized Raman Spectra")
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        canvas.draw()

    # === Event Handlers ===
    
    def on_tab_changed(self, index):
        """Handle tab changes."""
        tab_name = self.tab_widget.tabText(index)
        if tab_name == "Spectrum Analysis":
            self.update_spectrum_plot()
        elif tab_name == "Peak Fitting":
            self.update_peak_plot()
        elif tab_name == "Polarization Analysis":
            self.update_polarization_plot()
    
    def on_database_search(self, text):
        """Handle database search using modular database."""
        self.search_results.clear()
        
        if not text:
            return
        
        # Use modular database search
        matches = self.database.search_minerals(text, max_results=15)
        
        for mineral in matches:
            self.search_results.addItem(mineral)
    
    def on_mineral_selected(self, item):
        """Handle mineral selection."""
        if item:
            mineral_name = item.text()
            self.import_mineral_spectrum(mineral_name)
    
    def on_peak_click(self, event):
        """Handle peak selection clicks."""
        if not self.peak_selection_mode or not event.inaxes or not self.current_spectrum:
            return
        
        if event.xdata is not None:
            self.selected_peaks.append(event.xdata)
            self.update_peak_plot()
    
    # === Core Operations Using Modular Components ===
    
    def load_spectrum(self):
        """Load spectrum from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Spectrum",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt *.csv *.dat);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            data = np.loadtxt(file_path, delimiter=None)
            if data.shape[1] < 2:
                QMessageBox.warning(self, "Error", "File must have 2 columns")
                return
            
            filename = os.path.basename(file_path)
            self.current_spectrum = {
                'name': filename,
                'wavenumbers': data[:, 0],
                'intensities': data[:, 1],
                'source': 'file'
            }
            
            self.update_spectrum_plot()
            QMessageBox.information(self, "Success", f"Loaded: {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def import_selected_mineral(self):
        """Import selected mineral using modular database."""
        current_item = self.search_results.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a mineral")
            return
        
        mineral_name = current_item.text()
        self.import_mineral_spectrum(mineral_name)
    
    def import_mineral_spectrum(self, mineral_name):
        """Import mineral spectrum using modular database."""
        try:
            # Use modular database to generate spectrum
            wavenumbers, intensities = self.database.generate_spectrum(
                mineral_name,
                wavenumber_range=(100, 1200),
                num_points=2200,
                add_noise=True
            )
            
            if wavenumbers is not None and intensities is not None:
                self.imported_spectrum = {
                    'name': mineral_name,
                    'wavenumbers': wavenumbers,
                    'intensities': intensities,
                    'source': 'database'
                }
                
                self.selected_reference_mineral = mineral_name
                self.update_spectrum_plot()
                
                # Show crystal system info
                crystal_info = self.database.get_crystal_system_info(mineral_name)
                if crystal_info:
                    info_text = f"Crystal System: {crystal_info['crystal_system']}\n"
                    info_text += f"Space Group: {crystal_info['space_group']}"
                    QMessageBox.information(self, f"Imported {mineral_name}", info_text)
                
            else:
                QMessageBox.warning(self, "Error", f"Could not generate spectrum for {mineral_name}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error importing mineral: {str(e)}")
    
    def toggle_peak_selection(self):
        """Toggle peak selection mode."""
        self.peak_selection_mode = not self.peak_selection_mode
        
        if self.peak_selection_mode:
            self.toggle_peak_btn.setText("Exit Peak Selection")
            self.toggle_peak_btn.setStyleSheet("background-color: #ffcccc;")
        else:
            self.toggle_peak_btn.setText("Toggle Peak Selection")
            self.toggle_peak_btn.setStyleSheet("")
    
    def auto_find_peaks(self):
        """Auto-find peaks using modular peak fitting."""
        if not self.current_spectrum:
            QMessageBox.warning(self, "Warning", "Load a spectrum first")
            return
        
        wavenumbers = self.current_spectrum['wavenumbers']
        intensities = self.current_spectrum['intensities']
        
        # Use modular auto peak finding
        peaks = auto_find_peaks(wavenumbers, intensities, height_threshold=0.1, distance=15.0)
        
        self.selected_peaks = peaks
        self.update_peak_plot()
        
        QMessageBox.information(self, "Success", f"Found {len(peaks)} peaks automatically")
    
    def clear_peaks(self):
        """Clear all selected and fitted peaks."""
        self.selected_peaks.clear()
        self.fitted_peaks.clear()
        self.update_peak_plot()
    
    def fit_peaks(self):
        """Fit peaks using modular peak fitting."""
        if not self.selected_peaks or not self.current_spectrum:
            QMessageBox.warning(self, "Warning", "Select peaks first")
            return
        
        try:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            shape = self.shape_combo.currentText()
            
            # Use modular peak fitter
            fitted_peaks = self.peak_fitter.fit_multiple_peaks(
                wavenumbers, intensities, self.selected_peaks, shape
            )
            
            self.fitted_peaks = fitted_peaks
            self.update_peak_plot()
            
            # Show fitting results
            if fitted_peaks:
                results = []
                for peak in fitted_peaks:
                    properties = self.peak_fitter.calculate_peak_properties(peak)
                    results.append(f"Peak at {peak.position:.1f} cm⁻¹:")
                    results.append(f"  Quality: {peak.fit_quality}")
                    results.append(f"  R²: {peak.r_squared:.3f}")
                    if 'fwhm' in properties:
                        results.append(f"  FWHM: {properties['fwhm']:.2f} cm⁻¹")
                    results.append("")
                
                QMessageBox.information(self, "Fitting Results", "\n".join(results))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error fitting peaks: {str(e)}")
    
    # === Plot Updates ===
    
    def update_spectrum_plot(self):
        """Update spectrum plot."""
        if "Spectrum Analysis" not in self.plot_components:
            return
        
        ax = self.plot_components["Spectrum Analysis"]['ax']
        canvas = self.plot_components["Spectrum Analysis"]['canvas']
        
        ax.clear()
        
        has_data = False
        
        # Plot current spectrum
        if self.current_spectrum:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            ax.plot(wavenumbers, intensities, 'b-', linewidth=2, 
                   label=f"Current: {self.current_spectrum['name']}")
            has_data = True
        
        # Plot imported spectrum
        if self.imported_spectrum:
            wavenumbers = self.imported_spectrum['wavenumbers']
            intensities = self.imported_spectrum['intensities']
            
            # Normalize if current exists
            if self.current_spectrum:
                current_max = np.max(self.current_spectrum['intensities'])
                imported_max = np.max(intensities)
                if imported_max > 0:
                    intensities = intensities * (current_max / imported_max)
            
            ax.plot(wavenumbers, intensities, 'r-', linewidth=2, alpha=0.7,
                   label=f"Database: {self.imported_spectrum['name']}")
            has_data = True
        
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Raman Spectrum Analysis")
        ax.grid(True, alpha=0.3)
        
        if has_data:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Load a spectrum or search database',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.6)
        
        canvas.draw()
    
    def update_peak_plot(self):
        """Update peak fitting plot."""
        if "Peak Fitting" not in self.plot_components:
            return
        
        ax = self.plot_components["Peak Fitting"]['ax']
        canvas = self.plot_components["Peak Fitting"]['canvas']
        
        ax.clear()
        
        if self.current_spectrum:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            
            ax.plot(wavenumbers, intensities, 'b-', linewidth=1, alpha=0.7, label='Spectrum')
            
            # Plot selected peaks
            for peak_pos in self.selected_peaks:
                ax.axvline(x=peak_pos, color='red', linestyle='--', alpha=0.7)
            
            # Plot fitted peaks
            if self.fitted_peaks:
                x_plot = np.linspace(np.min(wavenumbers), np.max(wavenumbers), 1000)
                
                for peak in self.fitted_peaks:
                    if peak.shape == "Lorentzian":
                        y_plot = self.peak_fitter.lorentzian(x_plot, *peak.parameters)
                    elif peak.shape == "Gaussian":
                        y_plot = self.peak_fitter.gaussian(x_plot, *peak.parameters)
                    elif peak.shape == "Voigt":
                        y_plot = self.peak_fitter.voigt(x_plot, *peak.parameters)
                    else:
                        continue
                    
                    ax.plot(x_plot, y_plot, 'g-', linewidth=2, alpha=0.8)
                    ax.axvline(x=peak.position, color='green', linestyle='-', alpha=0.5)
            
            ax.legend()
        
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Peak Fitting Analysis")
        ax.grid(True, alpha=0.3)
        
        canvas.draw()


def main():
    """Main function for modular Qt6 application."""
    app = QApplication(sys.argv)
    
    app.setApplicationName("Raman Polarization Analyzer - Modular")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("RamanLab")
    
    window = RamanPolarizationAnalyzerModular()
    window.show()
    
    # Center window
    screen = app.primaryScreen().geometry()
    window_size = window.geometry()
    x = (screen.width() - window_size.width()) // 2
    y = (screen.height() - window_size.height()) // 2
    window.move(x, y)
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 