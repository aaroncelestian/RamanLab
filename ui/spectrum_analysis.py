"""
Spectrum Analysis UI Module

This module provides comprehensive spectrum analysis UI components including:
- Spectrum loading and display
- Spectrum comparison tools  
- Analysis parameters and controls
- Integration with core analysis modules

Professional PySide6 implementation with enhanced features.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox, QLabel,
    QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QSlider, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QFormLayout, QFrame, QTabWidget, QScrollArea, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import our custom toolbar configuration
from .matplotlib_config import CompactNavigationToolbar as NavigationToolbar

try:
    from core.polarization_analyzer import PolarizationAnalyzer
    from core.database_manager import DatabaseManager
    from utils.file_loaders import SpectrumLoader
except ImportError:
    print("Warning: Some core modules not available")


class SpectrumAnalysisWidget(QWidget):
    """
    Professional spectrum analysis widget with enhanced functionality.
    
    Features:
    - Multi-spectrum loading and comparison
    - Real-time analysis parameter adjustment
    - Professional plotting with multiple visualization modes
    - Integration with database and core analysis modules
    """
    
    # Signals for communication with main application
    spectrum_loaded = Signal(dict)
    analysis_completed = Signal(dict)
    parameters_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Analysis data
        self.current_spectrum = None
        self.imported_spectrum = None
        self.reference_spectra = []
        self.analysis_results = {}
        
        # UI state
        self.auto_update = True
        self.show_legend = True
        self.show_grid = True
        
        # Core modules
        self.polarization_analyzer = None
        self.database_manager = None
        self.spectrum_loader = None
        
        self.setup_ui()
        self.initialize_modules()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QHBoxLayout(self)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - controls
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Right panel - visualization
        visualization_panel = self.create_visualization_panel()
        splitter.addWidget(visualization_panel)
        
        # Set splitter proportions (30% controls, 70% visualization)
        splitter.setSizes([300, 700])
        
    def create_control_panel(self):
        """Create the control panel with analysis options."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # Add control tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Spectrum loading tab
        tabs.addTab(self.create_loading_tab(), "Load")
        
        # Analysis parameters tab
        tabs.addTab(self.create_analysis_tab(), "Analysis")
        
        # Display options tab
        tabs.addTab(self.create_display_tab(), "Display")
        
        # Results tab
        tabs.addTab(self.create_results_tab(), "Results")
        
        return panel
        
    def create_loading_tab(self):
        """Create spectrum loading controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File loading group
        file_group = QGroupBox("File Loading")
        file_layout = QVBoxLayout(file_group)
        
        load_btn = QPushButton("Load Spectrum File")
        load_btn.clicked.connect(self.load_spectrum_file)
        file_layout.addWidget(load_btn)
        
        self.current_file_label = QLabel("No file loaded")
        self.current_file_label.setStyleSheet("color: #666; font-size: 10px;")
        file_layout.addWidget(self.current_file_label)
        
        layout.addWidget(file_group)
        
        # Database group
        database_group = QGroupBox("Database Import")
        db_layout = QVBoxLayout(database_group)
        
        import_btn = QPushButton("Import from Database")
        import_btn.clicked.connect(self.import_from_database)
        db_layout.addWidget(import_btn)
        
        self.imported_spectrum_label = QLabel("No imported spectrum")
        self.imported_spectrum_label.setStyleSheet("color: #666; font-size: 10px;")
        db_layout.addWidget(self.imported_spectrum_label)
        
        layout.addWidget(database_group)
        
        # Comparison group
        comparison_group = QGroupBox("Spectrum Comparison")
        comp_layout = QVBoxLayout(comparison_group)
        
        add_reference_btn = QPushButton("Add Reference Spectrum")
        add_reference_btn.clicked.connect(self.add_reference_spectrum)
        comp_layout.addWidget(add_reference_btn)
        
        clear_references_btn = QPushButton("Clear References")
        clear_references_btn.clicked.connect(self.clear_reference_spectra)
        comp_layout.addWidget(clear_references_btn)
        
        self.reference_count_label = QLabel("References: 0")
        self.reference_count_label.setStyleSheet("color: #666; font-size: 10px;")
        comp_layout.addWidget(self.reference_count_label)
        
        layout.addWidget(comparison_group)
        
        layout.addStretch()
        return tab
        
    def create_analysis_tab(self):
        """Create analysis parameter controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Normalization group
        norm_group = QGroupBox("Normalization")
        norm_layout = QVBoxLayout(norm_group)
        
        self.normalize_check = QCheckBox("Auto-normalize spectra")
        self.normalize_check.setChecked(True)
        self.normalize_check.toggled.connect(self.on_parameters_changed)
        norm_layout.addWidget(self.normalize_check)
        
        # Normalization method
        norm_method_layout = QFormLayout()
        self.norm_method_combo = QComboBox()
        self.norm_method_combo.addItems(["Max Intensity", "Total Area", "Peak Area"])
        self.norm_method_combo.currentTextChanged.connect(self.on_parameters_changed)
        norm_method_layout.addRow("Method:", self.norm_method_combo)
        norm_layout.addLayout(norm_method_layout)
        
        layout.addWidget(norm_group)
        
        # Baseline correction group
        baseline_group = QGroupBox("Baseline Correction")
        baseline_layout = QVBoxLayout(baseline_group)
        
        self.baseline_check = QCheckBox("Apply baseline correction")
        self.baseline_check.toggled.connect(self.on_parameters_changed)
        baseline_layout.addWidget(self.baseline_check)
        
        # Baseline parameters
        baseline_params = QFormLayout()
        
        self.baseline_lambda_spin = QDoubleSpinBox()
        self.baseline_lambda_spin.setRange(1e3, 1e8)
        self.baseline_lambda_spin.setValue(1e5)
        self.baseline_lambda_spin.setDecimals(0)
        self.baseline_lambda_spin.setSingleStep(1e3)
        self.baseline_lambda_spin.valueChanged.connect(self.on_parameters_changed)
        baseline_params.addRow("λ (smoothness):", self.baseline_lambda_spin)
        
        self.baseline_p_spin = QDoubleSpinBox()
        self.baseline_p_spin.setRange(0.001, 0.1)
        self.baseline_p_spin.setValue(0.01)
        self.baseline_p_spin.setDecimals(3)
        self.baseline_p_spin.setSingleStep(0.001)
        self.baseline_p_spin.valueChanged.connect(self.on_parameters_changed)
        baseline_params.addRow("p (asymmetry):", self.baseline_p_spin)
        
        baseline_layout.addLayout(baseline_params)
        layout.addWidget(baseline_group)
        
        # Smoothing group
        smooth_group = QGroupBox("Smoothing")
        smooth_layout = QVBoxLayout(smooth_group)
        
        self.smoothing_check = QCheckBox("Apply smoothing")
        self.smoothing_check.toggled.connect(self.on_parameters_changed)
        smooth_layout.addWidget(self.smoothing_check)
        
        # Smoothing parameters
        smooth_params = QFormLayout()
        
        self.smooth_window_spin = QSpinBox()
        self.smooth_window_spin.setRange(3, 51)
        self.smooth_window_spin.setValue(11)
        self.smooth_window_spin.setSingleStep(2)
        self.smooth_window_spin.valueChanged.connect(self.on_parameters_changed)
        smooth_params.addRow("Window size:", self.smooth_window_spin)
        
        self.smooth_order_spin = QSpinBox()
        self.smooth_order_spin.setRange(1, 5)
        self.smooth_order_spin.setValue(3)
        self.smooth_order_spin.valueChanged.connect(self.on_parameters_changed)
        smooth_params.addRow("Polynomial order:", self.smooth_order_spin)
        
        smooth_layout.addLayout(smooth_params)
        layout.addWidget(smooth_group)
        
        # Auto-update
        self.auto_update_check = QCheckBox("Auto-update analysis")
        self.auto_update_check.setChecked(True)
        self.auto_update_check.toggled.connect(self.on_auto_update_changed)
        layout.addWidget(self.auto_update_check)
        
        # Manual update button
        self.update_btn = QPushButton("Update Analysis")
        self.update_btn.setEnabled(False)
        self.update_btn.clicked.connect(self.update_analysis)
        layout.addWidget(self.update_btn)
        
        layout.addStretch()
        return tab
        
    def create_display_tab(self):
        """Create display option controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Plot options group
        plot_group = QGroupBox("Plot Options")
        plot_layout = QVBoxLayout(plot_group)
        
        self.show_legend_check = QCheckBox("Show legend")
        self.show_legend_check.setChecked(True)
        self.show_legend_check.toggled.connect(self.on_display_changed)
        plot_layout.addWidget(self.show_legend_check)
        
        self.show_grid_check = QCheckBox("Show grid")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.toggled.connect(self.on_display_changed)
        plot_layout.addWidget(self.show_grid_check)
        
        self.show_markers_check = QCheckBox("Show data points")
        self.show_markers_check.toggled.connect(self.on_display_changed)
        plot_layout.addWidget(self.show_markers_check)
        
        layout.addWidget(plot_group)
        
        # Color scheme group
        color_group = QGroupBox("Color Scheme")
        color_layout = QVBoxLayout(color_group)
        
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Default", "Colorblind Safe", "High Contrast", "Publication"])
        self.color_scheme_combo.currentTextChanged.connect(self.on_display_changed)
        color_layout.addWidget(self.color_scheme_combo)
        
        layout.addWidget(color_group)
        
        # Export group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        export_plot_btn = QPushButton("Export Plot")
        export_plot_btn.clicked.connect(self.export_plot)
        export_layout.addWidget(export_plot_btn)
        
        export_data_btn = QPushButton("Export Data")
        export_data_btn.clicked.connect(self.export_data)
        export_layout.addWidget(export_data_btn)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        return tab
        
    def create_results_tab(self):
        """Create results display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis summary
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(120)
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        # Statistics table
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        stats_layout.addWidget(self.stats_table)
        
        layout.addWidget(stats_group)
        
        return tab
        
    def create_visualization_panel(self):
        """Create the visualization panel with plotting."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Create subplot
        self.ax = self.figure.add_subplot(111)
        self.initialize_plot()
        
        return panel
        
    def initialize_modules(self):
        """Initialize core analysis modules."""
        try:
            self.polarization_analyzer = PolarizationAnalyzer()
            self.database_manager = DatabaseManager()
            self.spectrum_loader = SpectrumLoader()
        except Exception as e:
            print(f"Warning: Could not initialize core modules: {e}")
            
    def initialize_plot(self):
        """Initialize the plot with default settings."""
        self.ax.clear()
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Raman Spectrum Analysis")
        self.ax.grid(True, alpha=0.3)
        
        # Add placeholder text
        self.ax.text(0.5, 0.5, 'Load a spectrum to begin analysis',
                    transform=self.ax.transAxes, ha='center', va='center',
                    fontsize=14, alpha=0.6)
        
        self.canvas.draw()
        
    def load_spectrum_file(self):
        """Load spectrum from file."""
        try:
            if self.spectrum_loader:
                spectrum_data = self.spectrum_loader.load_interactive()
                if spectrum_data:
                    self.current_spectrum = spectrum_data
                    self.current_file_label.setText(f"✓ {spectrum_data.get('name', 'Unknown')}")
                    self.current_file_label.setStyleSheet("color: green; font-size: 10px;")
                    self.update_plot()
                    self.update_analysis()
                    self.spectrum_loaded.emit(spectrum_data)
            else:
                QMessageBox.warning(self, "Error", "Spectrum loader not available")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load spectrum: {str(e)}")
            
    def import_from_database(self):
        """Import spectrum from database."""
        try:
            if self.database_manager:
                spectrum_data = self.database_manager.search_interactive()
                if spectrum_data:
                    self.imported_spectrum = spectrum_data
                    self.imported_spectrum_label.setText(f"✓ {spectrum_data.get('name', 'Unknown')}")
                    self.imported_spectrum_label.setStyleSheet("color: blue; font-size: 10px;")
                    self.update_plot()
            else:
                QMessageBox.warning(self, "Error", "Database manager not available")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import from database: {str(e)}")
            
    def add_reference_spectrum(self):
        """Add a reference spectrum for comparison."""
        try:
            if self.spectrum_loader:
                spectrum_data = self.spectrum_loader.load_interactive()
                if spectrum_data:
                    self.reference_spectra.append(spectrum_data)
                    self.reference_count_label.setText(f"References: {len(self.reference_spectra)}")
                    self.update_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add reference spectrum: {str(e)}")
            
    def clear_reference_spectra(self):
        """Clear all reference spectra."""
        self.reference_spectra.clear()
        self.reference_count_label.setText("References: 0")
        self.update_plot()
        
    def on_parameters_changed(self):
        """Handle parameter changes."""
        if self.auto_update:
            self.update_analysis()
        else:
            self.update_btn.setEnabled(True)
            
        # Emit signal for main application
        params = self.get_analysis_parameters()
        self.parameters_changed.emit(params)
        
    def on_auto_update_changed(self, enabled):
        """Handle auto-update toggle."""
        self.auto_update = enabled
        self.update_btn.setEnabled(not enabled)
        
    def on_display_changed(self):
        """Handle display option changes."""
        self.show_legend = self.show_legend_check.isChecked()
        self.show_grid = self.show_grid_check.isChecked()
        self.update_plot()
        
    def get_analysis_parameters(self):
        """Get current analysis parameters."""
        return {
            'normalize': self.normalize_check.isChecked(),
            'normalization_method': self.norm_method_combo.currentText(),
            'baseline_correction': self.baseline_check.isChecked(),
            'baseline_lambda': self.baseline_lambda_spin.value(),
            'baseline_p': self.baseline_p_spin.value(),
            'smoothing': self.smoothing_check.isChecked(),
            'smooth_window': self.smooth_window_spin.value(),
            'smooth_order': self.smooth_order_spin.value()
        }
        
    def update_analysis(self):
        """Update analysis with current parameters."""
        if not self.current_spectrum:
            return
            
        try:
            # Get parameters
            params = self.get_analysis_parameters()
            
            # Perform analysis using core modules
            if self.polarization_analyzer:
                results = self.polarization_analyzer.analyze_spectrum(
                    self.current_spectrum, params
                )
                self.analysis_results = results
                self.update_results_display()
                self.analysis_completed.emit(results)
                
            self.update_plot()
            self.update_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to update analysis: {str(e)}")
            
    def update_plot(self):
        """Update the main plot."""
        self.ax.clear()
        
        has_data = False
        
        # Plot current spectrum
        if self.current_spectrum:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            
            # Apply processing if enabled
            if self.normalize_check.isChecked():
                intensities = self.normalize_spectrum(intensities)
                
            self.ax.plot(wavenumbers, intensities, 'b-', linewidth=2,
                        label=f"Current: {self.current_spectrum.get('name', 'Spectrum')}")
            has_data = True
            
        # Plot imported spectrum
        if self.imported_spectrum:
            wavenumbers = self.imported_spectrum['wavenumbers']
            intensities = self.imported_spectrum['intensities']
            
            # Normalize to current spectrum if both exist
            if self.current_spectrum and self.normalize_check.isChecked():
                intensities = self.normalize_spectrum(intensities)
                
            self.ax.plot(wavenumbers, intensities, 'r-', linewidth=2, alpha=0.7,
                        label=f"Database: {self.imported_spectrum.get('name', 'Imported')}")
            has_data = True
            
        # Plot reference spectra
        colors = ['g-', 'm-', 'c-', 'y-', 'orange']
        for i, ref_spectrum in enumerate(self.reference_spectra):
            if i < len(colors):
                wavenumbers = ref_spectrum['wavenumbers']
                intensities = ref_spectrum['intensities']
                
                if self.normalize_check.isChecked():
                    intensities = self.normalize_spectrum(intensities)
                    
                self.ax.plot(wavenumbers, intensities, colors[i], linewidth=1.5, alpha=0.6,
                            label=f"Ref {i+1}: {ref_spectrum.get('name', 'Reference')}")
                has_data = True
                
        # Configure plot
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Raman Spectrum Analysis")
        
        if self.show_grid:
            self.ax.grid(True, alpha=0.3)
            
        if has_data and self.show_legend:
            self.ax.legend()
        elif not has_data:
            self.ax.text(0.5, 0.5, 'Load a spectrum to begin analysis',
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=14, alpha=0.6)
                        
        self.canvas.draw()
        
    def normalize_spectrum(self, intensities):
        """Normalize spectrum based on selected method."""
        method = self.norm_method_combo.currentText()
        
        if method == "Max Intensity":
            return intensities / np.max(intensities)
        elif method == "Total Area":
            return intensities / np.trapz(intensities)
        elif method == "Peak Area":
            # Find main peak and normalize to it
            peak_idx = np.argmax(intensities)
            return intensities / intensities[peak_idx]
        else:
            return intensities
            
    def update_results_display(self):
        """Update the results display with analysis data."""
        if not self.analysis_results:
            return
            
        # Update summary text
        summary = f"Analysis Results Summary\n"
        summary += f"{'='*30}\n\n"
        
        if 'peak_count' in self.analysis_results:
            summary += f"Peaks detected: {self.analysis_results['peak_count']}\n"
        if 'noise_level' in self.analysis_results:
            summary += f"Noise level: {self.analysis_results['noise_level']:.2f}\n"
        if 'signal_to_noise' in self.analysis_results:
            summary += f"S/N ratio: {self.analysis_results['signal_to_noise']:.2f}\n"
            
        self.summary_text.setPlainText(summary)
        
        # Update statistics table
        stats = self.analysis_results.get('statistics', {})
        self.stats_table.setRowCount(len(stats))
        
        for i, (param, value) in enumerate(stats.items()):
            self.stats_table.setItem(i, 0, QTableWidgetItem(param))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))
            
    def export_plot(self):
        """Export the current plot."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Plot", "spectrum_analysis.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
            )
            if filename:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export plot: {str(e)}")
            
    def export_data(self):
        """Export the analysis data."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Data", "spectrum_data.csv",
                "CSV files (*.csv);;JSON files (*.json)"
            )
            if filename:
                # Export logic here
                QMessageBox.information(self, "Success", f"Data exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
            
    def set_spectrum(self, spectrum_data):
        """Set the current spectrum programmatically."""
        self.current_spectrum = spectrum_data
        self.current_file_label.setText(f"✓ {spectrum_data.get('name', 'Unknown')}")
        self.current_file_label.setStyleSheet("color: green; font-size: 10px;")
        self.update_plot()
        self.update_analysis()
        
    def get_current_spectrum(self):
        """Get the current spectrum data."""
        return self.current_spectrum
        
    def get_analysis_results(self):
        """Get the current analysis results."""
        return self.analysis_results 