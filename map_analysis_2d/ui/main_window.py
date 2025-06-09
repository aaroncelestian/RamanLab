"""
Main window for the map analysis application.

This module contains the main application window that integrates all UI components
and connects them to the core functionality.
"""

import logging
import numpy as np
from typing import Optional
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, 
    QTabWidget, QMessageBox, QFileDialog, QDialog, QTextEdit, QPushButton, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from datetime import datetime

# Import core functionality
from ..core import RamanMapData
from ..core.cosmic_ray_detection import CosmicRayConfig, SimpleCosmicRayManager
from ..analysis import PCAAnalyzer, NMFAnalyzer
from ..workers import MapAnalysisWorker

# Import UI components
from .base_widgets import ScrollableControlPanel, ProgressStatusWidget
from .plotting_widgets import SplitMapSpectrumWidget, BasePlotWidget, PCANMFPlotWidget
from .control_panels import (
    MapViewControlPanel, PCAControlPanel, TemplateControlPanel,
    NMFControlPanel, MLControlPanel, ResultsControlPanel,
    DimensionalityReductionControlPanel
)

# Import additional modules
from ..core.template_management import TemplateSpectraManager

logger = logging.getLogger(__name__)


class MapAnalysisMainWindow(QMainWindow):
    """Main window for 2D Raman map analysis."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Window setup
        self.setWindowTitle("RamanLab - 2D Map Analysis")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Initialize data and analysis objects
        self.map_data: Optional[RamanMapData] = None
        self.pca_analyzer = PCAAnalyzer()
        self.nmf_analyzer = NMFAnalyzer()
        
        # Initialize ML analyzers and data manager
        from ..analysis.ml_classification import SupervisedMLAnalyzer, UnsupervisedAnalyzer, MLTrainingDataManager
        self.supervised_analyzer = SupervisedMLAnalyzer()
        self.unsupervised_analyzer = UnsupervisedAnalyzer()
        self.ml_data_manager = MLTrainingDataManager()
        
        # Initialize model manager for multiple trained models
        from map_analysis_2d.analysis.model_manager import ModelManager
        self.model_manager = ModelManager()
        
        # Setup automatic model persistence
        self._setup_model_persistence()
        
        # Initialize template manager
        self.template_manager = TemplateSpectraManager()
        
        # Initialize cosmic ray detection
        self.cosmic_ray_config = CosmicRayConfig()
        self.cosmic_ray_manager = SimpleCosmicRayManager(self.cosmic_ray_config)
        
        # Worker thread management
        self.worker: Optional[MapAnalysisWorker] = None
        
        # Current analysis state
        self.current_feature = "Integrated Intensity"
        self.use_processed = True
        self.intensity_vmin = None
        self.intensity_vmax = None
        self.integration_center = None
        self.integration_width = None
        
        # Initialize spectrum selection tracking
        self.current_marker_position = None
        self.current_selected_spectrum = None
        
        # Setup UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        
        logger.info("Main window initialized")
        
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for cleaner look
        
        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Configure splitter for better user interaction
        self.splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely
        self.splitter.setHandleWidth(8)  # Make the splitter handle more visible/grabbable
        
        # Set splitter style for better visibility
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #d0d0d0;
                border: 1px solid #a0a0a0;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #b0b0b0;
            }
            QSplitter::handle:pressed {
                background-color: #909090;
            }
        """)
        
        main_layout.addWidget(self.splitter)
        
        # Left panel for controls (remove max_width constraint to allow resizing)
        self.controls_panel = ScrollableControlPanel(max_width=500)  # Increased max width
        self.controls_panel.setMinimumWidth(200)  # Set minimum width to prevent too small
        self.create_permanent_controls()
        self.splitter.addWidget(self.controls_panel)
        
        # Right panel for visualization
        self.create_visualization_panel(self.splitter)
        
        # Set initial splitter proportions (left panel ~20%, right panel ~80%)
        self.splitter.setSizes([300, 1200])
        
        # Make the splitter handle more responsive
        self.splitter.setOpaqueResize(True)  # Show content while dragging
        
    def create_permanent_controls(self):
        """Create controls that are always visible."""
        # Data loading functions have been moved to the File menu
        # This creates a cleaner interface with less button duplication
        pass
        
    def create_visualization_panel(self, parent):
        """Create the right visualization panel."""
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        viz_layout.addWidget(self.tab_widget)
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Create all tabs
        self.create_map_tab()
        self.create_template_tab()
        self.create_dimensionality_reduction_tab()
        self.create_ml_tab()
        self.create_results_tab()
        
        parent.addWidget(viz_widget)
        
    def create_map_tab(self):
        """Create the main map visualization tab."""
        # Just use the plotting widget directly - controls are in the left panel
        self.map_plot_widget = SplitMapSpectrumWidget()
        
        # Connect spectrum request signal
        self.map_plot_widget.spectrum_requested.connect(self.on_spectrum_requested)
        
        self.tab_widget.addTab(self.map_plot_widget, "Map View")
        
    def create_template_tab(self):
        """Create the template analysis tab."""
        self.template_plot_widget = BasePlotWidget(figsize=(12, 8))
        self.tab_widget.addTab(self.template_plot_widget, "Template Analysis")
        
    def create_dimensionality_reduction_tab(self):
        """Create the combined PCA/NMF analysis tab."""
        # Create the new 2x2 plotting widget for comprehensive PCA and NMF visualization
        self.dimensionality_plot_widget = PCANMFPlotWidget()
        # Keep references for backward compatibility
        self.pca_plot_widget = self.dimensionality_plot_widget
        self.nmf_plot_widget = self.dimensionality_plot_widget
        self.tab_widget.addTab(self.dimensionality_plot_widget, "PCA & NMF Analysis")
        
    def create_ml_tab(self):
        """Create the ML classification tab."""
        self.ml_plot_widget = BasePlotWidget(figsize=(12, 8))
        self.tab_widget.addTab(self.ml_plot_widget, "ML Classification")
        
    def create_results_tab(self):
        """Create the results summary tab."""
        from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QTextEdit
        from PySide6.QtCore import Qt
        
        # Create the main widget for the results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        # Add title and export button
        header_layout = QHBoxLayout()
        title_label = QLabel("Comprehensive Analysis Results")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Export button
        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.clicked.connect(self.export_comprehensive_results)
        self.export_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        header_layout.addWidget(self.export_results_btn)
        
        results_layout.addLayout(header_layout)
        
        # Create the comprehensive plot widget
        self.results_plot_widget = BasePlotWidget(figsize=(14, 10))
        results_layout.addWidget(self.results_plot_widget)
        
        # Add statistics text area
        self.results_statistics = QTextEdit()
        self.results_statistics.setMaximumHeight(120)
        self.results_statistics.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        results_layout.addWidget(self.results_statistics)
        
        self.tab_widget.addTab(results_widget, "Results")
        
    def setup_menu_bar(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Data Loading section
        load_action = QAction('&Load Map Data...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.setStatusTip('Load Raman map data from directory')
        load_action.triggered.connect(self.load_map_data)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        # PKL File Management section
        load_pkl_action = QAction('Load &PKL Map...', self)
        load_pkl_action.setShortcut('Ctrl+Shift+O')
        load_pkl_action.setStatusTip('Load previously saved PKL map file (preserves processed data)')
        load_pkl_action.triggered.connect(self.load_map_from_pkl)
        file_menu.addAction(load_pkl_action)
        
        save_pkl_action = QAction('&Save Map to PKL...', self)
        save_pkl_action.setShortcut('Ctrl+S')
        save_pkl_action.setStatusTip('Save current map data to PKL file (preserves all processing)')
        save_pkl_action.triggered.connect(self.save_map_to_pkl)
        file_menu.addAction(save_pkl_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('&Analysis')
        
        # PCA submenu
        pca_action = QAction('Run &PCA Analysis', self)
        pca_action.triggered.connect(lambda: (self.tab_widget.setCurrentIndex(2), self.run_pca()))
        analysis_menu.addAction(pca_action)
        
        # NMF submenu
        nmf_action = QAction('Run &NMF Analysis', self)
        nmf_action.triggered.connect(lambda: (self.tab_widget.setCurrentIndex(2), self.run_nmf()))
        analysis_menu.addAction(nmf_action)
        
        analysis_menu.addSeparator()
        
        # Save/Load results
        save_nmf_action = QAction('&Save NMF Results...', self)
        save_nmf_action.triggered.connect(self.save_nmf_results)
        analysis_menu.addAction(save_nmf_action)
        
        load_nmf_action = QAction('&Load NMF Results...', self)
        load_nmf_action.triggered.connect(self.load_nmf_results)
        analysis_menu.addAction(load_nmf_action)
        
        analysis_menu.addSeparator()
        
        # Template analysis
        template_action = QAction('&Template Analysis', self)
        template_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        analysis_menu.addAction(template_action)
        
        analysis_menu.addSeparator()
        
        # Machine Learning submenu
        ml_menu = analysis_menu.addMenu('&Machine Learning')
        
        ml_load_training_action = ml_menu.addAction('Load &Training Data...')
        ml_load_training_action.triggered.connect(self.load_training_data)
        
        ml_menu.addSeparator()
        
        ml_train_supervised_action = ml_menu.addAction('Train &Supervised Model')
        ml_train_supervised_action.triggered.connect(self.train_supervised_model)
        
        ml_train_unsupervised_action = ml_menu.addAction('Train &Clustering Model')
        ml_train_unsupervised_action.triggered.connect(self.train_unsupervised_model)
        
        ml_classify_action = ml_menu.addAction('&Apply to Map')
        ml_classify_action.triggered.connect(self.classify_map)
        
        ml_menu.addSeparator()
        
        ml_save_model_action = ml_menu.addAction('&Save ML Model...')
        ml_save_model_action.triggered.connect(self.save_named_model)
        
        ml_load_model_action = ml_menu.addAction('&Load ML Model...')
        ml_load_model_action.triggered.connect(self.load_ml_model)
        
        ml_view_action = ml_menu.addAction('&ML Analysis View')
        ml_view_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(3))
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Panel layout controls
        reset_layout_action = QAction('&Reset Panel Layout', self)
        reset_layout_action.setShortcut('Ctrl+R')
        reset_layout_action.setStatusTip('Reset panel sizes to default proportions')
        reset_layout_action.triggered.connect(self.reset_panel_layout)
        view_menu.addAction(reset_layout_action)
        
        view_menu.addSeparator()
        
        # Tab navigation
        map_view_action = QAction('&Map View', self)
        map_view_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        view_menu.addAction(map_view_action)
        
        template_view_action = QAction('&Template Analysis', self)
        template_view_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        view_menu.addAction(template_view_action)
        
        dimensionality_view_action = QAction('&PCA && NMF Analysis', self)
        dimensionality_view_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))
        view_menu.addAction(dimensionality_view_action)
        
        ml_view_action = QAction('&ML Analysis', self)
        ml_view_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(3))
        view_menu.addAction(ml_view_action)
        
        results_view_action = QAction('&Results Summary', self)
        results_view_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(4))
        view_menu.addAction(results_view_action)
        
    def setup_status_bar(self):
        """Set up the status bar."""
        self.progress_status = ProgressStatusWidget()
        self.statusBar().addWidget(self.progress_status, 1)
        
    def on_tab_changed(self, index: int):
        """Handle tab changes to update control panel."""
        self.controls_panel.clear_dynamic_sections()
        
        if index == 0:  # Map View
            control_panel = MapViewControlPanel()
            control_panel.feature_changed.connect(self.on_feature_changed)
            control_panel.use_processed_changed.connect(self.on_use_processed_changed)
            control_panel.show_spectrum_toggled.connect(self.on_show_spectrum_toggled)
            control_panel.intensity_scaling_changed.connect(self.on_intensity_scaling_changed)
            control_panel.wavenumber_range_changed.connect(self.on_wavenumber_range_changed)
            
            # Connect cosmic ray signals
            control_panel.cosmic_ray_enabled_changed.connect(self.on_cosmic_ray_enabled_changed)
            control_panel.cosmic_ray_params_changed.connect(self.on_cosmic_ray_params_changed)
            control_panel.reprocess_cosmic_rays_requested.connect(self.reprocess_cosmic_rays)
            control_panel.apply_cre_to_all_files_requested.connect(self.apply_cre_to_all_files)
            control_panel.show_cosmic_ray_stats.connect(self.show_cosmic_ray_statistics)
            control_panel.show_cosmic_ray_diagnostics.connect(self.show_cosmic_ray_diagnostics)
            control_panel.test_shape_analysis_requested.connect(self.test_shape_analysis)
            
            # Connect template fitting signal
            control_panel.fit_templates_to_map_requested.connect(self.fit_templates)
            
            self.controls_panel.add_section("map_controls", control_panel)
            
            # Update template status in map control panel
            self.update_map_template_status()
            
            # Always check for classification results and add them to the new control panel
            if hasattr(self, 'classification_results'):
                logger.info("Found classification results, adding to new map control panel...")
                self.update_map_features_with_classification()
            
            # Also check for clustering results
            if hasattr(self, 'ml_results') and self.ml_results.get('type') == 'unsupervised':
                logger.info("Found clustering results, adding to new map control panel...")
                self.update_map_features_with_clustering()
            
            # Also check for NMF results
            if hasattr(self, 'nmf_analyzer') and hasattr(self.nmf_analyzer, 'nmf') and self.nmf_analyzer.nmf is not None:
                logger.info("Found NMF results, adding to new map control panel...")
                self.update_map_features_with_nmf()
            
        elif index == 1:  # Template Analysis
            control_panel = TemplateControlPanel()
            control_panel.load_template_file_requested.connect(self.load_template_file)
            control_panel.load_template_folder_requested.connect(self.load_template_folder)
            control_panel.remove_template_requested.connect(self.remove_template)
            control_panel.clear_templates_requested.connect(self.clear_templates)
            control_panel.plot_templates_requested.connect(self.plot_templates)
            control_panel.fit_templates_requested.connect(self.fit_templates)
            control_panel.normalize_templates_requested.connect(self.normalize_templates)
            self.controls_panel.add_section("template_controls", control_panel)
            
            # Update template list in the control panel
            self.update_template_control_panel()
            
        elif index == 2:  # Combined PCA & NMF
            control_panel = DimensionalityReductionControlPanel()
            control_panel.run_pca_requested.connect(self.run_pca)
            control_panel.run_nmf_requested.connect(self.run_nmf)
            control_panel.rerun_pca_requested.connect(self.rerun_pca_analysis)
            control_panel.rerun_nmf_requested.connect(self.rerun_nmf_analysis)
            control_panel.save_nmf_requested.connect(self.save_nmf_results)
            control_panel.load_nmf_requested.connect(self.load_nmf_results)
            self.controls_panel.add_section("dimensionality_controls", control_panel)
            
        elif index == 3:  # ML Analysis
            control_panel = MLControlPanel()
            control_panel.train_supervised_requested.connect(self.train_supervised_model)
            control_panel.train_unsupervised_requested.connect(self.train_unsupervised_model)
            control_panel.classify_map_requested.connect(self.classify_map)
            control_panel.load_training_data_requested.connect(self.load_training_data)
            control_panel.save_named_model_requested.connect(self.save_named_model)
            control_panel.load_model_requested.connect(self.load_ml_model)
            control_panel.remove_model_requested.connect(self.remove_selected_model)
            control_panel.apply_selected_model_requested.connect(self.apply_selected_model)
            control_panel.model_selection_changed.connect(self.on_model_selection_changed)
            
            # Check if we have a trained model and enable/disable classify button accordingly
            if hasattr(self, 'supervised_analyzer') and self.supervised_analyzer.model is not None:
                control_panel.enable_classify_button()
            else:
                control_panel.disable_classify_button()
                
            self.controls_panel.add_section("ml_controls", control_panel)
            
            # Populate model list from model manager
            self._populate_ml_control_panel_models(control_panel)
            
        elif index == 4:  # Results
            control_panel = ResultsControlPanel()
            control_panel.generate_report_requested.connect(self.generate_report)
            control_panel.export_results_requested.connect(self.export_results)
            self.controls_panel.add_section("results_controls", control_panel)
            
            # Automatically refresh comprehensive results when switching to results tab
            self.plot_comprehensive_results()
            
        # Add stretch to push all controls to the top
        self.controls_panel.add_stretch()
    
    def reset_panel_layout(self):
        """Reset the splitter panel layout to default proportions."""
        # Reset to default sizes (control panel ~20%, visualization ~80%)
        total_width = self.width()
        control_width = 300
        viz_width = total_width - control_width
        self.splitter.setSizes([control_width, viz_width])
        self.statusBar().showMessage("Panel layout reset to default", 2000)
    
    def load_map_data(self):
        """Load map data from a directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Map Data Directory")
        
        if directory:
            self.progress_status.show_progress("Loading map data...")
            
            try:
                # Load map data with cosmic ray detection
                self.map_data = RamanMapData(directory, cosmic_ray_config=self.cosmic_ray_config)
                
                self.progress_status.hide_progress()
                self.statusBar().showMessage(f"Loaded {len(self.map_data.spectra)} spectra")
                logger.info(f"Loaded map data with {len(self.map_data.spectra)} spectra")
                
                # Initialize integration slider with spectrum midpoint
                self._initialize_integration_slider()
                
                self.update_map()
                
                # Suggest saving to PKL for future quick access
                self.suggest_pkl_save_after_load()
                
            except Exception as e:
                self.progress_status.hide_progress()
                QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")
                logger.error(f"Error loading map data: {e}")
    
    def _initialize_integration_slider(self):
        """Initialize the integration slider with spectrum midpoint."""
        if self.map_data is None or not self.map_data.spectra:
            return
            
        try:
            # Get the first spectrum to determine wavenumber range
            first_spectrum = next(iter(self.map_data.spectra.values()))
            wavenumbers = first_spectrum.wavenumbers
            
            # Get the map control panel and initialize slider
            control_panel = self.get_current_map_control_panel()
            if control_panel:
                # Update slider range based on actual spectrum data
                control_panel.update_slider_range(wavenumbers)
                # Set slider to spectrum midpoint
                control_panel.set_spectrum_midpoint(wavenumbers)
                
                logger.info(f"Initialized integration slider: range {wavenumbers.min():.0f}-{wavenumbers.max():.0f} cm⁻¹, "
                           f"midpoint {(wavenumbers.min() + wavenumbers.max()) / 2:.0f} cm⁻¹")
                
        except Exception as e:
            logger.error(f"Error initializing integration slider: {e}")
                
    def on_feature_changed(self, feature: str):
        """Handle feature selection change."""
        self.current_feature = feature
        self.update_map()
        
    def on_use_processed_changed(self, use_processed: bool):
        """Handle use processed data change."""
        self.use_processed = use_processed
        self.update_map()
        
    def on_show_spectrum_toggled(self, show: bool):
        """Handle show spectrum plot toggle."""
        self.map_plot_widget.show_spectrum_panel(show)
        
    def on_intensity_scaling_changed(self, vmin: float, vmax: float):
        """Handle intensity scaling changes."""
        # Store the intensity scaling values
        self.intensity_vmin = vmin if not np.isnan(vmin) else None
        self.intensity_vmax = vmax if not np.isnan(vmax) else None
        
        # Refresh the current map with new scaling
        self.update_map()
    
    def on_wavenumber_range_changed(self, center: float, width: float):
        """Handle wavenumber range changes for custom integration."""
        # Store the integration range
        if center == 0.0 and width == 0.0:
            # Special case: use full spectrum
            self.integration_center = None
            self.integration_width = None
        else:
            self.integration_center = center
            self.integration_width = width
        
        # Refresh the map if we're showing integrated intensity
        if self.current_feature == "Integrated Intensity":
            self.update_map()
        
    def on_spectrum_requested(self, x: float, y: float):
        """Handle spectrum request from map click."""
        if self.map_data is None:
            return
            
        try:
            # Find the closest spectrum to the clicked position
            closest_spectrum = self.find_closest_spectrum(x, y)
            
            if closest_spectrum is not None:
                # Show the spectrum panel if it's not already shown
                self.map_plot_widget.show_spectrum_panel(True)
                
                # Get wavenumbers and intensities
                wavenumbers = closest_spectrum.wavenumbers
                intensities = (closest_spectrum.processed_intensities 
                             if self.use_processed and closest_spectrum.processed_intensities is not None
                             else closest_spectrum.intensities)
                
                # Validate dimensions before plotting
                if len(wavenumbers) != len(intensities):
                    logger.warning(f"Dimension mismatch: wavenumbers({len(wavenumbers)}) != intensities({len(intensities)})")
                    # Trim to shorter length to avoid plotting error
                    min_len = min(len(wavenumbers), len(intensities))
                    wavenumbers = wavenumbers[:min_len]
                    intensities = intensities[:min_len]
                    logger.info(f"Trimmed both arrays to length {min_len}")
                
                # Create title with position info
                title = f"Spectrum at ({closest_spectrum.x_pos:.1f}, {closest_spectrum.y_pos:.1f})"
                if closest_spectrum.processed_intensities is not None and self.use_processed:
                    title += " [Processed]"
                else:
                    title += " [Raw]"
                
                # Add template fitting info to title if available
                pos_key = (closest_spectrum.x_pos, closest_spectrum.y_pos)
                if hasattr(self, 'template_fitting_results') and pos_key in self.template_fitting_results['r_squared']:
                    r_squared = self.template_fitting_results['r_squared'][pos_key]
                    title += f" | Template Fit R²: {r_squared:.3f}"
                
                # Clear previous plot
                self.map_plot_widget.spectrum_widget.ax.clear()
                
                # Plot the original spectrum
                self.map_plot_widget.spectrum_widget.ax.plot(
                    wavenumbers, intensities,
                    color='blue' if self.use_processed else 'red',
                    linewidth=1.5,
                    label='Measured Spectrum'
                )
                
                # Plot template fitting results if available
                if hasattr(self, 'template_fitting_results') and pos_key in self.template_fitting_results['coefficients']:
                    self.plot_template_fit_overlay(closest_spectrum, wavenumbers, intensities)
                
                # Set labels and title
                self.map_plot_widget.spectrum_widget.ax.set_xlabel('Wavenumber (cm⁻¹)')
                self.map_plot_widget.spectrum_widget.ax.set_ylabel('Intensity')
                self.map_plot_widget.spectrum_widget.ax.set_title(title)
                self.map_plot_widget.spectrum_widget.ax.grid(True, alpha=0.3)
                self.map_plot_widget.spectrum_widget.ax.legend()
                
                self.map_plot_widget.spectrum_widget.draw()
                
                # Add a marker on the map to show the selected position
                self.add_position_marker(closest_spectrum.x_pos, closest_spectrum.y_pos)
                
                # Track the current marker position for CRE test
                self.current_marker_position = (closest_spectrum.x_pos, closest_spectrum.y_pos)
                self.current_selected_spectrum = closest_spectrum
                
                # Update status bar with template info
                status_msg = f"Showing spectrum at position ({closest_spectrum.x_pos:.1f}, {closest_spectrum.y_pos:.1f})"
                if hasattr(self, 'template_fitting_results') and pos_key in self.template_fitting_results['coefficients']:
                    coeffs = self.template_fitting_results['coefficients'][pos_key]
                    r_squared = self.template_fitting_results['r_squared'][pos_key]
                    # Show contribution of dominant template
                    if len(coeffs) > 0:
                        max_coeff_idx = max(range(len(coeffs)), key=lambda i: coeffs[i])
                        if max_coeff_idx < len(self.template_fitting_results['template_names']):
                            dominant_template = self.template_fitting_results['template_names'][max_coeff_idx]
                            status_msg += f" | Dominant: {dominant_template} ({coeffs[max_coeff_idx]:.2f})"
                
                self.statusBar().showMessage(status_msg)
                
        except Exception as e:
            logger.error(f"Error displaying spectrum: {e}")
            QMessageBox.warning(self, "Error", f"Failed to display spectrum:\n{str(e)}")
            
    def find_closest_spectrum(self, x: float, y: float):
        """Find the spectrum closest to the given coordinates."""
        if self.map_data is None:
            return None
            
        min_distance = float('inf')
        closest_spectrum = None
        
        for spectrum in self.map_data.spectra.values():
            distance = ((spectrum.x_pos - x) ** 2 + (spectrum.y_pos - y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_spectrum = spectrum
                
        return closest_spectrum
        
    def add_position_marker(self, x: float, y: float):
        """Add a marker to show the selected position on the map."""
        try:
            # Get the map axes
            map_ax = self.map_plot_widget.map_widget.ax
            
            # Remove previous markers
            for artist in map_ax.get_children():
                if hasattr(artist, '_spectrum_marker'):
                    artist.remove()
            
            # Add new marker (red cross)
            marker = map_ax.plot(x, y, 'r+', markersize=15, markeredgewidth=3, 
                               label=f'Selected ({x:.1f}, {y:.1f})')[0]
            marker._spectrum_marker = True  # Flag for easy removal
            
            # Refresh the canvas
            self.map_plot_widget.map_widget.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error adding position marker: {e}")
        
    def update_map(self):
        """Update the map display."""
        if self.map_data is None:
            return
            
        try:
            import numpy as np
            # Create map based on selected feature
            if self.current_feature == "Cosmic Ray Map":
                map_data = self.create_cosmic_ray_map()
            elif self.current_feature.startswith("Template: "):
                template_name = self.current_feature[10:]  # Remove "Template: " prefix
                map_data = self.create_template_contribution_map(template_name)
            elif self.current_feature == "Template Fit Quality (R²)":
                map_data = self.create_template_fit_quality_map()
            elif self.current_feature.startswith("NMF Component "):
                # Extract component number
                try:
                    component_num = int(self.current_feature.split()[-1])
                    component_index = component_num - 1  # Convert to 0-based index
                    map_data = self.create_nmf_component_map(component_index)
                except (ValueError, IndexError):
                    logger.error(f"Invalid NMF component feature: {self.current_feature}")
                    map_data = self.create_integrated_intensity_map()
            elif self.current_feature.startswith("ML Clusters"):
                map_data = self.create_ml_clustering_map()
            elif self.current_feature.startswith("ML Classification"):
                map_data = self.create_ml_classification_map()
            elif self.current_feature.startswith("Class Probability: "):
                class_name = self.current_feature[18:]  # Remove "Class Probability: " prefix
                map_data = self.create_class_probability_map(class_name)
            elif self.current_feature.startswith("Model: "):
                map_data = self.create_model_result_map()
            else:
                map_data = self.create_integrated_intensity_map()
            
            if map_data is not None:
                x_positions = [s.x_pos for s in self.map_data.spectra.values()]
                y_positions = [s.y_pos for s in self.map_data.spectra.values()]
                
                extent = [min(x_positions), max(x_positions), 
                         min(y_positions), max(y_positions)]
                
                # Choose appropriate colormap and get discrete labels if applicable
                cmap = 'viridis'
                discrete_labels = None
                
                if self.current_feature.startswith("Template: "):
                    cmap = 'plasma'
                elif self.current_feature == "Template Fit Quality (R²)":
                    cmap = 'RdYlGn'
                elif self.current_feature == "Cosmic Ray Map":
                    cmap = 'Reds'
                elif self.current_feature.startswith("NMF Component "):
                    cmap = 'inferno'
                elif self.current_feature.startswith("ML Clusters"):
                    cmap = 'tab10'  # Discrete colormap for clusters
                    # Get cluster method name for labeling
                    if hasattr(self, 'ml_results') and 'method' in self.ml_results:
                        method = self.ml_results['method']
                        n_clusters = self.ml_results.get('n_clusters', 0)
                        if method == 'DBSCAN':
                            discrete_labels = ['Noise'] + [f'Cluster {i}' for i in range(n_clusters)]
                        else:
                            discrete_labels = [f'Cluster {i}' for i in range(n_clusters)]
                elif self.current_feature.startswith("ML Classification"):
                    cmap = 'Set1'  # Discrete colormap for classification
                    # Get class names for labeling
                    if hasattr(self, 'classification_results') and 'class_names' in self.classification_results:
                        discrete_labels = self.classification_results['class_names']
                    elif hasattr(self, 'supervised_analyzer') and hasattr(self.supervised_analyzer, 'class_names'):
                        discrete_labels = self.supervised_analyzer.class_names
                elif self.current_feature.startswith("Class Probability: "):
                    cmap = 'plasma'  # Continuous colormap for probabilities
                elif self.current_feature.startswith("Model: "):
                    # Get discrete labels for model results
                    if hasattr(self, 'model_results') and self.current_feature in self.model_results:
                        result_data = self.model_results[self.current_feature]
                        if result_data['type'] == 'classification':
                            cmap = 'Set1'
                            # Try to get class names from model info
                            model_name = self.current_feature[7:]  # Remove "Model: " prefix
                            model_info = self.model_manager.get_model_info(model_name)
                            if model_info and 'class_names' in model_info:
                                discrete_labels = model_info['class_names']
                            else:
                                # Fallback to generic class labels
                                unique_preds = np.unique(result_data['predictions'])
                                discrete_labels = [f'Class {int(p)}' for p in unique_preds if p >= 0]
                        else:  # clustering
                            cmap = 'tab10'
                            # Generate cluster labels
                            unique_labels = np.unique(result_data['labels'])
                            discrete_labels = []
                            for label in unique_labels:
                                if label == -1:
                                    discrete_labels.append('Noise')
                                else:
                                    discrete_labels.append(f'Cluster {int(label)}')
                    else:
                        cmap = 'viridis'
                
                # Update control panel with current data range for auto-scaling
                if hasattr(self, 'map_control_panel') and map_data is not None:
                    data_min, data_max = np.nanmin(map_data), np.nanmax(map_data)
                    self.map_control_panel.update_intensity_range(data_min, data_max)
                
                # Create title with integration info
                title = f"{self.current_feature} Map"
                if (self.current_feature == "Integrated Intensity" and 
                    self.integration_center is not None and self.integration_width is not None):
                    min_wn = self.integration_center - self.integration_width / 2
                    max_wn = self.integration_center + self.integration_width / 2
                    title += f" ({min_wn:.0f}-{max_wn:.0f} cm⁻¹)"
                
                self.map_plot_widget.plot_map(
                    map_data, extent=extent,
                    title=title,
                    cmap=cmap,
                    discrete_labels=discrete_labels,
                    vmin=self.intensity_vmin,
                    vmax=self.intensity_vmax
                )
                
        except Exception as e:
            import traceback
            logger.error(f"Error updating map: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Try to plot with basic parameters as fallback
            try:
                if map_data is not None:
                    # Create fallback title
                    title = f"{self.current_feature} Map"
                    if (self.current_feature == "Integrated Intensity" and 
                        self.integration_center is not None and self.integration_width is not None):
                        min_wn = self.integration_center - self.integration_width / 2
                        max_wn = self.integration_center + self.integration_width / 2
                        title += f" ({min_wn:.0f}-{max_wn:.0f} cm⁻¹)"
                    
                    self.map_plot_widget.plot_map(
                        map_data, extent=extent,
                        title=title,
                        cmap='viridis',  # Use basic colormap as fallback
                        vmin=self.intensity_vmin,
                        vmax=self.intensity_vmax
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback plotting also failed: {fallback_error}")
            
    def create_integrated_intensity_map(self):
        """Create integrated intensity map with optional wavenumber range."""
        import numpy as np
        
        positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
        
        if not positions:
            return None
            
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        map_array = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
        
        for spectrum in self.map_data.spectra.values():
            intensities = (spectrum.processed_intensities 
                         if self.use_processed and spectrum.processed_intensities is not None
                         else spectrum.intensities)
            
            # Apply wavenumber range integration if specified
            if self.integration_center is not None and self.integration_width is not None:
                wavenumbers = spectrum.wavenumbers
                
                # Calculate range bounds
                min_wavenumber = self.integration_center - self.integration_width / 2
                max_wavenumber = self.integration_center + self.integration_width / 2
                
                # Find indices within the range
                mask = (wavenumbers >= min_wavenumber) & (wavenumbers <= max_wavenumber)
                
                if np.any(mask):
                    # Integrate only within the specified range
                    integrated_intensity = np.sum(intensities[mask])
                else:
                    # No data in range, set to zero
                    integrated_intensity = 0.0
            else:
                # Full spectrum integration (default behavior)
                integrated_intensity = np.sum(intensities)
            
            map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = integrated_intensity
            
        return map_array
        
    def create_template_contribution_map(self, template_name: str):
        """Create a map showing the contribution of a specific template."""
        if not hasattr(self, 'template_fitting_results'):
            return None
            
        try:
            import numpy as np
            
            # Find template index
            template_names = self.template_fitting_results['template_names']
            if template_name not in template_names:
                logger.error(f"Template {template_name} not found in fitting results")
                return None
                
            template_index = template_names.index(template_name)
            
            # Get map dimensions
            positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            map_array = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            # Fill map with template contribution coefficients
            for spectrum in self.map_data.spectra.values():
                pos_key = (spectrum.x_pos, spectrum.y_pos)
                if pos_key in self.template_fitting_results['coefficients']:
                    coeffs = self.template_fitting_results['coefficients'][pos_key]
                    if template_index < len(coeffs):
                        contribution = coeffs[template_index]
                        map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = contribution
            
            return map_array
            
        except Exception as e:
            logger.error(f"Error creating template contribution map: {e}")
            return None
    
    def create_template_fit_quality_map(self):
        """Create a map showing template fitting quality (R-squared values)."""
        if not hasattr(self, 'template_fitting_results'):
            return None
            
        try:
            import numpy as np
            
            # Get map dimensions
            positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            map_array = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            # Fill map with R-squared values
            for spectrum in self.map_data.spectra.values():
                pos_key = (spectrum.x_pos, spectrum.y_pos)
                if pos_key in self.template_fitting_results['r_squared']:
                    r_squared = self.template_fitting_results['r_squared'][pos_key]
                    map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = r_squared
            
            return map_array
            
        except Exception as e:
            logger.error(f"Error creating template fit quality map: {e}")
            return None
        
    def run_pca(self):
        """Run PCA analysis with parameters from control panel."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
            
        try:
            # Get parameters from combined control panel
            control_panel = self.get_current_dimensionality_control_panel()
            n_components = control_panel.get_pca_n_components() if control_panel else 5
            
            self.progress_status.show_progress("Running PCA...")
            
            # Prepare data
            spectra_list = list(self.map_data.spectra.values())
            data_matrix = []
            
            for spectrum in spectra_list:
                intensities = (spectrum.processed_intensities 
                             if self.use_processed and spectrum.processed_intensities is not None
                             else spectrum.intensities)
                data_matrix.append(intensities)
                
            import numpy as np
            X = np.array(data_matrix)
            
            # Run PCA
            results = self.pca_analyzer.run_analysis(X, n_components=n_components)
            
            if results['success']:
                self.tab_widget.setCurrentIndex(2)  # Switch to combined tab
                self.plot_pca_results(results)
                self.statusBar().showMessage("PCA analysis completed")
            else:
                QMessageBox.warning(self, "PCA Error", 
                                  f"PCA failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            QMessageBox.critical(self, "PCA Error", f"Error in PCA:\n{str(e)}")
        finally:
            self.progress_status.hide_progress()
            
    def perform_pca_clustering(self, pca_results, n_clusters=3):
        """Perform clustering on PCA components."""
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            
            components = pca_results.get('components')
            if components is None or components.shape[1] < 2:
                return None
            
            # Use the first few components for clustering
            n_comp_for_clustering = min(components.shape[1], 5)
            clustering_data = components[:, :n_comp_for_clustering]
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(clustering_data)
            
            return {
                'method': 'K-Means',
                'labels': labels,
                'n_clusters': n_clusters,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            }
            
        except Exception as e:
            logger.error(f"Error in PCA clustering: {e}")
            return None
    
    def plot_pca_results(self, results):
        """Plot comprehensive PCA analysis results with clustering."""
        # Store results for potential re-runs
        self.pca_results = results
        
        # Perform clustering on PCA components
        pca_clusters = self.perform_pca_clustering(results)
        
        # Use the new plotting widget to display results
        self.dimensionality_plot_widget.plot_pca_results(results, pca_clusters)
        
        # Store results for potential re-runs
        self.last_pca_results = results
        self.last_pca_clusters = pca_clusters
        
    # Template Analysis Methods
    def load_template_file(self):
        """Load a single template spectrum file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Template Spectrum", "", 
            "Text files (*.txt *.csv *.dat);;All files (*)")
        
        if file_path:
            try:
                success = self.template_manager.load_template(file_path)
                if success:
                    self.statusBar().showMessage(f"Template loaded: {file_path}")
                    self.update_template_control_panel()
                    self.update_map_template_status()  # Update map control panel
                else:
                    QMessageBox.warning(self, "Error", f"Failed to load template: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading template:\n{str(e)}")
                logger.error(f"Error loading template {file_path}: {e}")
            
    def load_template_folder(self):
        """Load all template spectra from a folder."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Template Folder")
        
        if directory:
            try:
                import os
                loaded_count = 0
                failed_files = []
                
                # Get all text files in the directory
                for filename in os.listdir(directory):
                    if filename.lower().endswith(('.txt', '.csv', '.dat')):
                        filepath = os.path.join(directory, filename)
                        success = self.template_manager.load_template(filepath)
                        if success:
                            loaded_count += 1
                        else:
                            failed_files.append(filename)
                
                # Update UI and show results
                self.update_template_control_panel()
                self.update_map_template_status()  # Update map control panel
                
                message = f"Loaded {loaded_count} template(s) from folder"
                if failed_files:
                    message += f"\nFailed to load: {', '.join(failed_files)}"
                
                if loaded_count > 0:
                    QMessageBox.information(self, "Templates Loaded", message)
                    self.statusBar().showMessage(f"{loaded_count} templates loaded from folder")
                else:
                    QMessageBox.warning(self, "No Templates", "No valid template files found in folder")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading templates from folder:\n{str(e)}")
                logger.error(f"Error loading templates from folder {directory}: {e}")
             
    def remove_template(self, index: int):
        """Remove a template spectrum by index."""
        try:
            success = self.template_manager.remove_template(index)
            if success:
                self.update_template_control_panel()
                self.statusBar().showMessage(f"Template removed")
                
                # Update plot if templates are currently displayed
                if self.tab_widget.currentIndex() == 1:  # Template tab
                    self.plot_templates()
            else:
                QMessageBox.warning(self, "Error", "Failed to remove template")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing template:\n{str(e)}")
            logger.error(f"Error removing template at index {index}: {e}")
         
    def clear_templates(self):
        """Clear all templates."""
        try:
            self.template_manager.clear_templates()
            self.update_template_control_panel()
            self.update_map_template_status()  # Update map control panel
            self.statusBar().showMessage("All templates cleared")
            
            # Clear the plot
            if self.tab_widget.currentIndex() == 1:  # Template tab
                self.template_plot_widget.clear_plot()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error clearing templates:\n{str(e)}")
            logger.error(f"Error clearing templates: {e}")
          
    def plot_templates(self):
        """Plot all loaded templates."""
        try:
            self.tab_widget.setCurrentIndex(1)  # Switch to template tab
            self.template_plot_widget.clear_plot()
            
            if self.template_manager.get_template_count() == 0:
                self.template_plot_widget.ax.text(0.5, 0.5, 'No templates loaded\nUse "Load Single File" or "Load Folder" to add templates', 
                                                  ha='center', va='center', transform=self.template_plot_widget.ax.transAxes,
                                                  fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                self.template_plot_widget.draw()
                return
            
            # Get control panel to check display options
            control_panel = self.get_current_template_control_panel()
            show_raw = control_panel.show_raw_data() if control_panel else True
            show_processed = control_panel.show_processed_data() if control_panel else True
            
            # Plot templates
            for i, template in enumerate(self.template_manager.templates):
                # Plot raw data if requested
                if show_raw and template.intensities is not None:
                    self.template_plot_widget.ax.plot(
                        template.wavenumbers, template.intensities,
                        color=template.color, alpha=0.3, linestyle='--',
                        label=f'{template.name} (raw)' if len(self.template_manager.templates) <= 10 else None
                    )
                
                # Plot processed data if requested
                if show_processed and template.processed_intensities is not None:
                    self.template_plot_widget.ax.plot(
                        self.template_manager.target_wavenumbers, template.processed_intensities,
                        color=template.color, linewidth=2,
                        label=f'{template.name}' if len(self.template_manager.templates) <= 10 else None
                    )
            
            # Set labels and title
            self.template_plot_widget.ax.set_xlabel('Wavenumber (cm⁻¹)')
            self.template_plot_widget.ax.set_ylabel('Intensity')
            self.template_plot_widget.ax.set_title(f'Template Spectra ({self.template_manager.get_template_count()} loaded)')
            self.template_plot_widget.ax.grid(True, alpha=0.3)
            
            # Add legend if not too many templates
            if len(self.template_manager.templates) <= 10:
                self.template_plot_widget.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            self.template_plot_widget.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting templates:\n{str(e)}")
            logger.error(f"Error plotting templates: {e}")
        
    def fit_templates(self):
        """Fit templates to map data."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
            
        if self.template_manager.get_template_count() == 0:
            QMessageBox.warning(self, "No Templates", "Load template spectra first.")
            return
        
        try:
            # Get fitting parameters from control panel
            control_panel = self.get_current_template_control_panel()
            method = control_panel.get_fitting_method() if control_panel else 'nnls'
            use_baseline = control_panel.use_baseline_fitting() if control_panel else True
            
            # Confirm fitting operation
            reply = QMessageBox.question(
                self, "Template Fitting",
                f"Fit {self.template_manager.get_template_count()} templates to {len(self.map_data.spectra)} spectra?\n\n"
                f"Method: {method.upper()}\n"
                f"Use baseline: {use_baseline}\n\n"
                f"This may take a few minutes for large maps.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            self.progress_status.show_progress("Fitting templates to map data...")
            
            import numpy as np
            from scipy.optimize import nnls
            
            # Prepare template matrix - ensure dimensions match map data
            first_spectrum = next(iter(self.map_data.spectra.values()))
            target_intensities = (first_spectrum.processed_intensities 
                                if self.use_processed and first_spectrum.processed_intensities is not None
                                else first_spectrum.intensities)
            
            # Update template manager target wavenumbers to match map data
            self.template_manager.target_wavenumbers = first_spectrum.wavenumbers
            self.template_manager.validate_and_fix_template_dimensions(len(target_intensities))
            
            # Get template matrix
            template_matrix = self.template_manager.get_template_matrix()
            if template_matrix.size == 0:
                raise ValueError("No valid template data available")
            
            # Add baseline if requested
            if use_baseline:
                baseline = np.ones(len(target_intensities))
                template_matrix = np.column_stack([template_matrix, baseline])
            
            # Initialize storage for fitting results
            self.template_fitting_results = {
                'coefficients': {},  # {(x, y): [coeff1, coeff2, ...]}
                'r_squared': {},     # {(x, y): r_squared_value}
                'template_names': self.template_manager.get_template_names(),
                'use_baseline': use_baseline,
                'method': method
            }
            
            # Fit templates to each spectrum
            total_spectra = len(self.map_data.spectra)
            processed_count = 0
            failed_count = 0
            
            for spectrum in self.map_data.spectra.values():
                try:
                    # Get spectrum intensities
                    intensities = (spectrum.processed_intensities 
                                 if self.use_processed and spectrum.processed_intensities is not None
                                 else spectrum.intensities)
                    
                    # Ensure dimensions match
                    if len(intensities) != template_matrix.shape[0]:
                        failed_count += 1
                        continue
                    
                    # Perform fitting
                    if method == 'nnls':
                        coeffs, residual = nnls(template_matrix, intensities)
                    else:  # lsqr/lstsq
                        coeffs, residuals, rank, s = np.linalg.lstsq(
                            template_matrix, intensities, rcond=None
                        )
                        residual = residuals[0] if len(residuals) > 0 else 0
                    
                    # Calculate R-squared
                    y_mean = np.mean(intensities)
                    ss_tot = np.sum((intensities - y_mean) ** 2)
                    ss_res = residual if residual > 0 else np.sum((intensities - template_matrix @ coeffs) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Store results
                    pos_key = (spectrum.x_pos, spectrum.y_pos)
                    self.template_fitting_results['coefficients'][pos_key] = coeffs
                    self.template_fitting_results['r_squared'][pos_key] = max(0, min(1, r_squared))  # Clamp to [0,1]
                    
                    processed_count += 1
                    
                    # Update progress
                    if processed_count % 50 == 0:
                        progress = int((processed_count / total_spectra) * 100)
                        self.progress_status.update_progress(f"Fitting templates... {progress}%")
                
                except Exception as e:
                    logger.warning(f"Failed to fit spectrum at ({spectrum.x_pos}, {spectrum.y_pos}): {e}")
                    failed_count += 1
                    continue
            
            self.progress_status.hide_progress()
            
            # Show results
            success_rate = (processed_count / total_spectra) * 100
            message = f"Template fitting completed!\n\n"
            message += f"Successfully fitted: {processed_count}/{total_spectra} spectra ({success_rate:.1f}%)\n"
            if failed_count > 0:
                message += f"Failed: {failed_count} spectra\n"
            message += f"\nNew map features available:\n"
            for i, name in enumerate(self.template_fitting_results['template_names']):
                message += f"• {name} Contribution\n"
            message += f"• Template Fit Quality (R²)\n"
            
            QMessageBox.information(self, "Template Fitting Complete", message)
            
            # Update map features in control panel
            self.update_map_features_with_templates()
            self.update_map_template_status()  # Update template status
            
            self.statusBar().showMessage(f"Template fitting complete: {processed_count} spectra fitted")
            
        except Exception as e:
            self.progress_status.hide_progress()
            QMessageBox.critical(self, "Error", f"Error in template fitting:\n{str(e)}")
            logger.error(f"Error in template fitting: {e}")
            
    def update_map_features_with_templates(self):
        """Update map features dropdown to include template fitting results."""
        try:
            # Get current map control panel
            sections = self.controls_panel.sections
            if "map_controls" in sections:
                control_panel = sections["map_controls"]["widget"]
                if hasattr(control_panel, 'feature_combo') and hasattr(self, 'template_fitting_results'):
                    # Store current selection
                    current_feature = control_panel.feature_combo.currentText()
                    
                    # Add template features
                    control_panel.feature_combo.clear()
                    control_panel.feature_combo.addItems([
                        "Integrated Intensity",
                        "Peak Height", 
                        "Cosmic Ray Map"
                    ])
                    
                    # Add template contribution maps
                    for template_name in self.template_fitting_results['template_names']:
                        control_panel.feature_combo.addItem(f"Template: {template_name}")
                    
                    # Add fit quality map
                    control_panel.feature_combo.addItem("Template Fit Quality (R²)")
                    
                    # Restore selection if possible
                    index = control_panel.feature_combo.findText(current_feature)
                    if index >= 0:
                        control_panel.feature_combo.setCurrentIndex(index)
                        
        except Exception as e:
            logger.error(f"Error updating map features: {e}")
    
    def normalize_templates(self):
        """Normalize all template spectra."""
        if self.template_manager.get_template_count() == 0:
            QMessageBox.warning(self, "No Templates", "Load template spectra first.")
            return
        
        try:
            control_panel = self.get_current_template_control_panel()
            if not control_panel:
                return
                
            method = control_panel.get_normalization_method()
            
            import numpy as np
            
            for template in self.template_manager.templates:
                if template.processed_intensities is not None:
                    data = template.processed_intensities.copy()
                    
                    if method == "Min-Max (0-1)":
                        # Min-max normalization to [0,1]
                        data_min, data_max = np.min(data), np.max(data)
                        if data_max > data_min:
                            data = (data - data_min) / (data_max - data_min)
                            
                    elif method == "Max Normalization":
                        # Normalize by maximum value
                        data_max = np.max(data)
                        if data_max > 0:
                            data = data / data_max
                            
                    elif method == "Area Normalization":
                        # Normalize by area under curve
                        area = np.trapz(data, self.template_manager.target_wavenumbers)
                        if area > 0:
                            data = data / area
                            
                    elif method == "Standard Normalization":
                        # Z-score normalization
                        data_mean, data_std = np.mean(data), np.std(data)
                        if data_std > 0:
                            data = (data - data_mean) / data_std
                    
                    template.processed_intensities = data
            
            # Update plot if currently displayed
            if self.tab_widget.currentIndex() == 1:  # Template tab
                self.plot_templates()
                
            self.statusBar().showMessage(f"Templates normalized using {method}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error normalizing templates:\n{str(e)}")
            logger.error(f"Error normalizing templates: {e}")
     
    def update_template_control_panel(self):
        """Update the template control panel with current template list."""
        try:
            control_panel = self.get_current_template_control_panel()
            if control_panel:
                template_names = self.template_manager.get_template_names()
                control_panel.update_template_list(template_names)
        except Exception as e:
            logger.error(f"Error updating template control panel: {e}")
    
    def get_current_template_control_panel(self):
        """Get the current template control panel if it exists."""
        try:
            sections = self.controls_panel.sections
            if "template_controls" in sections:
                return sections["template_controls"]["widget"]
        except Exception:
            pass
        return None
        
    # NMF Analysis Methods
    def run_nmf(self):
        """Run NMF analysis with parameters from control panel."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
            
        try:
            # Get parameters from combined control panel
            control_panel = self.get_current_dimensionality_control_panel()
            if control_panel is None:
                QMessageBox.warning(self, "Error", "Dimensionality reduction control panel not found.")
                return
            
            n_components = control_panel.get_nmf_n_components()
            max_iter = control_panel.get_nmf_max_iter()
            random_state = control_panel.get_nmf_random_state()
            batch_size = control_panel.get_nmf_batch_size()
            solver = control_panel.get_nmf_solver()
            
            self.progress_status.show_progress(f"Running NMF with {n_components} components...")
            
            # Prepare data
            spectra_list = list(self.map_data.spectra.values())
            data_matrix = []
            valid_positions = []
            
            for spectrum in spectra_list:
                if spectrum.wavenumbers is not None:
                    intensities = (spectrum.processed_intensities 
                                 if self.use_processed and spectrum.processed_intensities is not None
                                 else spectrum.intensities)
                    if intensities is not None:
                        data_matrix.append(intensities)
                        valid_positions.append((spectrum.x_pos, spectrum.y_pos))
                
            if not data_matrix:
                QMessageBox.warning(self, "No Data", "No valid spectra found for NMF analysis.")
                return
                
            import numpy as np
            X = np.array(data_matrix)
            
            logger.info(f"Running NMF on {X.shape[0]} spectra with {X.shape[1]} features")
            logger.info(f"NMF parameters: components={n_components}, max_iter={max_iter}, "
                       f"batch_size={batch_size}, solver={solver}")
            
            # Update NMF analyzer with random state
            self.nmf_analyzer = NMFAnalyzer()
            
            # Run NMF with user parameters
            results = self.nmf_analyzer.run_analysis(
                X, 
                n_components=n_components,
                batch_size=batch_size,
                max_iter=max_iter
            )
            
            if results['success']:
                # Store results for map integration
                self.nmf_results = results
                self.nmf_valid_positions = valid_positions
                
                # Switch to combined tab and plot results
                self.tab_widget.setCurrentIndex(2)
                self.plot_nmf_results(results)
                
                # Update map features to include NMF components
                self.update_map_features_with_nmf()
                
                # Update control panel with results
                if control_panel:
                    results_info = (
                        f"NMF Analysis Complete!\n\n"
                        f"Components: {results['n_components']}\n"
                        f"Samples: {results['n_samples']}\n"
                        f"Features: {results['n_features']}\n"
                        f"Reconstruction Error: {results['reconstruction_error']:.4f}\n"
                        f"Iterations: {results.get('n_iterations', 'N/A')}\n\n"
                        f"Switch to Map View to visualize component distributions."
                    )
                    control_panel.update_nmf_info(results_info)
                
                self.statusBar().showMessage(
                    f"NMF analysis completed: {results['n_components']} components, "
                    f"reconstruction error: {results['reconstruction_error']:.4f}"
                )
                
                logger.info(f"NMF completed successfully with {results['n_components']} components")
                
            else:
                QMessageBox.warning(self, "NMF Error", 
                                  f"NMF failed: {results.get('error', 'Unknown error')}")
                logger.error(f"NMF analysis failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            QMessageBox.critical(self, "NMF Error", f"Error in NMF analysis:\n{str(e)}")
            logger.error(f"NMF analysis error: {e}")
        finally:
            self.progress_status.hide_progress()
            
    def perform_nmf_clustering(self, nmf_results, n_clusters=3):
        """Perform clustering on NMF components."""
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            
            components = nmf_results.get('components')  # W matrix
            if components is None or components.shape[1] < 2:
                return None
            
            # Use all NMF components for clustering (they're typically meaningful)
            clustering_data = components
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(clustering_data)
            
            return {
                'method': 'K-Means',
                'labels': labels,
                'n_clusters': n_clusters,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            }
            
        except Exception as e:
            logger.error(f"Error in NMF clustering: {e}")
            return None
    
    def plot_nmf_results(self, results):
        """Plot comprehensive NMF analysis results with clustering."""
        # Store results for potential re-runs
        self.nmf_results = results
        
        # Get wavenumbers for plotting
        wavenumbers = None
        if self.map_data and self.map_data.spectra:
            for spectrum in self.map_data.spectra.values():
                if spectrum.wavenumbers is not None:
                    wavenumbers = spectrum.wavenumbers
                    break
        
        # Perform clustering on NMF components
        nmf_clusters = self.perform_nmf_clustering(results)
        
        # Use the new plotting widget to display results
        self.dimensionality_plot_widget.plot_nmf_results(results, nmf_clusters, wavenumbers)
        
        # Store results for potential re-runs
        self.last_nmf_results = results
        self.last_nmf_clusters = nmf_clusters
    
    def rerun_pca_analysis(self):
        """Re-run PCA analysis and update plots."""
        if hasattr(self, 'last_pca_results') and self.last_pca_results:
            # Re-perform clustering (potentially with different parameters)
            pca_clusters = self.perform_pca_clustering(self.last_pca_results)
            # Update the plot
            self.dimensionality_plot_widget.plot_pca_results(self.last_pca_results, pca_clusters)
            self.last_pca_clusters = pca_clusters
        else:
            # If no previous results, run new analysis
            self.run_pca()
    
    def rerun_nmf_analysis(self):
        """Re-run NMF analysis and update plots."""
        if hasattr(self, 'last_nmf_results') and self.last_nmf_results:
            # Get wavenumbers
            wavenumbers = None
            if self.map_data and self.map_data.spectra:
                for spectrum in self.map_data.spectra.values():
                    if spectrum.wavenumbers is not None:
                        wavenumbers = spectrum.wavenumbers
                        break
            
            # Re-perform clustering (potentially with different parameters)
            nmf_clusters = self.perform_nmf_clustering(self.last_nmf_results)
            # Update the plot
            self.dimensionality_plot_widget.plot_nmf_results(self.last_nmf_results, nmf_clusters, wavenumbers)
            self.last_nmf_clusters = nmf_clusters
        else:
            # If no previous results, run new analysis
            self.run_nmf()
    
    def update_clustering_parameters(self, pca_clusters=3, nmf_clusters=3):
        """Update clustering parameters and refresh plots if results exist."""
        # Update PCA clustering if results exist
        if hasattr(self, 'last_pca_results') and self.last_pca_results:
            pca_cluster_results = self.perform_pca_clustering(self.last_pca_results, pca_clusters)
            
            # Get current NMF results and wavenumbers for consistent plotting
            wavenumbers = None
            if self.map_data and self.map_data.spectra:
                for spectrum in self.map_data.spectra.values():
                    if spectrum.wavenumbers is not None:
                        wavenumbers = spectrum.wavenumbers
                        break
            
            # If we have both PCA and NMF results, update both; otherwise just update PCA
            if hasattr(self, 'last_nmf_results') and self.last_nmf_results:
                nmf_cluster_results = self.perform_nmf_clustering(self.last_nmf_results, nmf_clusters)
                # Update both plots
                self.dimensionality_plot_widget.plot_pca_results(self.last_pca_results, pca_cluster_results)
                self.dimensionality_plot_widget.plot_nmf_results(self.last_nmf_results, nmf_cluster_results, wavenumbers)
                self.last_nmf_clusters = nmf_cluster_results
            else:
                # Update just PCA
                self.dimensionality_plot_widget.plot_pca_results(self.last_pca_results, pca_cluster_results)
            
            self.last_pca_clusters = pca_cluster_results
        
        # Update NMF clustering if results exist (and we haven't already updated it above)
        elif hasattr(self, 'last_nmf_results') and self.last_nmf_results:
            wavenumbers = None
            if self.map_data and self.map_data.spectra:
                for spectrum in self.map_data.spectra.values():
                    if spectrum.wavenumbers is not None:
                        wavenumbers = spectrum.wavenumbers
                        break
                        
            nmf_cluster_results = self.perform_nmf_clustering(self.last_nmf_results, nmf_clusters)
            self.dimensionality_plot_widget.plot_nmf_results(self.last_nmf_results, nmf_cluster_results, wavenumbers)
            self.last_nmf_clusters = nmf_cluster_results
    
    def get_current_nmf_control_panel(self):
        """Get the current NMF control panel (legacy method for compatibility)."""
        return self.get_current_dimensionality_control_panel()
    
    def get_current_dimensionality_control_panel(self):
        """Get the current combined dimensionality reduction control panel."""
        for name, section in self.controls_panel.sections.items():
            if name == "dimensionality_controls":
                return section['widget']
        return None
    
    def update_map_features_with_nmf(self):
        """Update map view features to include NMF components."""
        if not hasattr(self, 'nmf_results') or not hasattr(self, 'nmf_valid_positions'):
            return
            
        try:
            # Get the map control panel
            control_panel = None
            for name, section in self.controls_panel.sections.items():
                if name == "map_controls" and hasattr(section['widget'], 'feature_combo'):
                    control_panel = section['widget']
                    break
            
            if control_panel is None:
                return
            
            # Add NMF component options to feature combo
            current_features = [control_panel.feature_combo.itemText(i) 
                              for i in range(control_panel.feature_combo.count())]
            
            # Add NMF components if not already present
            for i in range(self.nmf_results['n_components']):
                component_name = f"NMF Component {i+1}"
                if component_name not in current_features:
                    control_panel.feature_combo.addItem(component_name)
            
            logger.info(f"Added {self.nmf_results['n_components']} NMF components to map features")
            
        except Exception as e:
            logger.error(f"Error updating map features with NMF: {e}")
    
    def create_nmf_component_map(self, component_index: int):
        """Create a map showing the contribution of a specific NMF component."""
        if not hasattr(self, 'nmf_results') or not hasattr(self, 'nmf_valid_positions'):
            return None
            
        try:
            import numpy as np
            
            components = self.nmf_results['components']
            positions = self.nmf_valid_positions
            
            if component_index >= components.shape[1]:
                logger.error(f"Component index {component_index} out of range")
                return None
            
            # Create position to component value mapping
            pos_to_value = {}
            for i, (x, y) in enumerate(positions):
                pos_to_value[(x, y)] = components[i, component_index]
            
            # Create map array
            all_positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            map_array = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            for spectrum in self.map_data.spectra.values():
                pos_key = (spectrum.x_pos, spectrum.y_pos)
                if pos_key in pos_to_value:
                    map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = pos_to_value[pos_key]
            
            return map_array
            
        except Exception as e:
            logger.error(f"Error creating NMF component map: {e}")
            return None
    
    def create_ml_clustering_map(self):
        """Create a map showing ML clustering results."""
        if not hasattr(self, 'ml_results') or self.ml_results.get('type') != 'unsupervised':
            return None
            
        try:
            import numpy as np
            
            labels = self.ml_results['labels']
            positions = self.ml_valid_positions
            
            # Create position to label mapping
            pos_to_label = {}
            for i, (x, y) in enumerate(positions):
                pos_to_label[(x, y)] = labels[i]
            
            # Create map array
            all_positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            map_array = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=float)
            
            for spectrum in self.map_data.spectra.values():
                pos_key = (spectrum.x_pos, spectrum.y_pos)
                if pos_key in pos_to_label:
                    map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = pos_to_label[pos_key]
            
            return map_array
            
        except Exception as e:
            logger.error(f"Error creating ML clustering map: {e}")
            return None
        
    # ML Analysis Methods
    def load_training_data(self):
        """Load training data from multiple class directories."""
        try:
            from PySide6.QtWidgets import QInputDialog
            
            # Get number of classes
            n_classes, ok = QInputDialog.getInt(
                self, "Number of Classes", 
                "How many classes do you want to train on?", 
                value=2, minValue=2, maxValue=10
            )
            
            if not ok:
                return
            
            # Get directories for each class
            class_directories = {}
            for i in range(n_classes):
                class_name, ok = QInputDialog.getText(
                    self, f"Class {i+1} Name", 
                    f"Enter name for class {i+1}:"
                )
                
                if not ok or not class_name:
                    return
                
                directory = QFileDialog.getExistingDirectory(
                    self, f"Select Directory for Class '{class_name}'")
                
                if not directory:
                    return
                
                class_directories[class_name] = directory
            
            # Load the data
            self.progress_status.show_progress("Loading training data...")
            
            results = self.ml_data_manager.load_class_data(
                class_directories, 
                cosmic_ray_manager=self.cosmic_ray_manager
            )
            
            self.progress_status.hide_progress()
            
            if results['success']:
                # Update control panel with data info
                control_panel = self.get_current_ml_control_panel()
                if control_panel:
                    info_text = f"Training data loaded:\n"
                    info_text += f"Classes: {results['n_classes']}\n"
                    info_text += f"Total spectra: {results['total_spectra']}\n"
                    for class_name, count in results['class_counts'].items():
                        info_text += f"  {class_name}: {count} spectra\n"
                    
                    control_panel.update_training_data_info(
                        f"Loaded {results['n_classes']} classes, {results['total_spectra']} spectra total"
                    )
                
                QMessageBox.information(
                    self, "Data Loaded", 
                    f"Successfully loaded training data:\n"
                    f"{results['n_classes']} classes, {results['total_spectra']} total spectra"
                )
                
                self.statusBar().showMessage(f"Training data loaded: {results['total_spectra']} spectra")
                logger.info(f"Training data loaded: {results}")
                
            else:
                QMessageBox.critical(
                    self, "Load Failed", 
                    f"Failed to load training data:\n{results.get('error', 'Unknown error')}"
                )
                
        except Exception as e:
            self.progress_status.hide_progress()
            QMessageBox.critical(self, "Error", f"Error loading training data:\n{str(e)}")
            logger.error(f"Training data loading error: {e}")
    
    def train_supervised_model(self):
        """Train supervised ML classification model."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
        
        # Check if training data is loaded
        if not self.ml_data_manager.class_data:
            QMessageBox.warning(self, "No Training Data", 
                              "Load training data first using 'Load Training Data Folders'.")
            return
        
        try:
            # Get parameters from control panel
            control_panel = self.get_current_ml_control_panel()
            if control_panel is None:
                QMessageBox.warning(self, "Error", "ML control panel not found.")
                return
            
            model_type = control_panel.get_supervised_model()
            n_estimators = control_panel.get_n_estimators()
            max_depth = control_panel.get_max_depth()
            test_size = control_panel.get_test_size()
            feature_options = control_panel.get_feature_options()
            
            self.progress_status.show_progress(f"Training {model_type} model...")
            
            # Get training data with error handling for inconsistent spectra
            try:
                X, y, class_names, common_wavenumbers = self.ml_data_manager.get_training_data(self.preprocessor)
            except ValueError as e:
                if "No overlapping wavenumber range" in str(e):
                    QMessageBox.critical(
                        self, "Incompatible Spectra", 
                        "Your training spectra have no overlapping wavenumber range.\n\n"
                        "Please ensure all spectra cover a common wavenumber range.\n\n"
                        "Tips:\n"
                        "• Check that all spectra are from similar instruments\n"
                        "• Verify wavenumber ranges are compatible\n"
                        "• Remove any truncated or incomplete spectra"
                    )
                    return
                elif "No valid spectra could be processed" in str(e):
                    QMessageBox.critical(
                        self, "Processing Error", 
                        "All training spectra failed to process.\n\n"
                        "Please check:\n"
                        "• File format is correct (CSV with wavenumber, intensity columns)\n"
                        "• Files are not corrupted\n"
                        "• Wavenumber ranges are reasonable"
                    )
                    return
                else:
                    QMessageBox.critical(self, "Training Data Error", f"Error processing training data:\n{str(e)}")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Unexpected Error", f"Unexpected error loading training data:\n{str(e)}")
                return
            
            # Apply feature transformation if requested
            feature_transformer = None
            feature_used = 'full'
            
            # Check and apply PCA features if requested
            if feature_options['use_pca']:
                if not hasattr(self, 'pca_analyzer') or self.pca_analyzer.pca is None:
                    QMessageBox.warning(
                        self, "PCA Not Available", 
                        "PCA features requested but no PCA model found.\n\n"
                        "Please run PCA analysis first:\n"
                        "1. Go to 'PCA & NMF Analysis' tab\n"
                        "2. Click 'Run PCA Analysis'\n"
                        "3. Return to ML tab and try training again"
                    )
                    return
                else:
                    feature_transformer = self.pca_analyzer
                    feature_used = 'pca'
            
            # Check and apply NMF features if requested  
            elif feature_options['use_nmf']:
                if not hasattr(self, 'nmf_analyzer') or self.nmf_analyzer.nmf is None:
                    QMessageBox.warning(
                        self, "NMF Not Available", 
                        "NMF features requested but no NMF model found.\n\n"
                        "Please run NMF analysis first:\n"
                        "1. Go to 'PCA & NMF Analysis' tab\n"
                        "2. Click 'Run NMF Analysis'\n"
                        "3. Return to ML tab and try training again"
                    )
                    return
                else:
                    feature_transformer = self.nmf_analyzer
                    feature_used = 'nmf'
            
            # Train the actual model using the training data manager
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Create and train the model
            if model_type == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                raise ValueError(f"Model type {model_type} not yet supported for multi-class")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store the trained model in the analyzer
            self.supervised_analyzer.model = model
            self.supervised_analyzer.model_type = model_type
            self.supervised_analyzer.X_test = X_test
            self.supervised_analyzer.y_test = y_test
            self.supervised_analyzer.y_pred = y_pred
            self.supervised_analyzer.class_names = class_names
            
            # Store the common wavenumber grid used for training (this is the interpolated grid)
            self.supervised_analyzer.training_wavenumbers = common_wavenumbers
            logger.info(f"Stored training wavenumbers: {len(common_wavenumbers)} points")
            
            # Store results for visualization
            self.ml_results = {
                'success': True,
                'type': 'supervised',
                'model_type': model_type,
                'class_names': class_names,
                'n_classes': len(class_names),
                'accuracy': accuracy,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'n_features': X.shape[1]
            }
            
            # Switch to ML tab and plot results
            self.tab_widget.setCurrentIndex(3)  # ML tab is now at index 3, not 4
            self.plot_ml_results(self.ml_results)
            
            # Update control panel with results
            if control_panel:
                results_info = (
                    f"Supervised Training Complete!\n\n"
                    f"Model: {model_type}\n"
                    f"Classes: {len(class_names)} ({', '.join(class_names)})\n"
                    f"Training samples: {len(X_train)}\n"
                    f"Test samples: {len(X_test)}\n"
                    f"Accuracy: {accuracy:.3f}\n"
                    f"Features: {X.shape[1]} ({feature_used.upper()})\n\n"
                    "✅ Model ready for map classification.\n"
                    "Click 'Apply Model to Map' below!"
                )
                control_panel.update_info(results_info)
                
                # Enable the classify button
                control_panel.enable_classify_button()
            
            # Also create classification results immediately for feature dropdown
            # This allows the classification features to appear in the map dropdown right after training
            self.classification_results = {
                'predictions': [],  # Empty until map is classified
                'probabilities': None,
                'positions': [],
                'type': 'supervised',
                'class_names': class_names
            }
            
            # Automatically apply the model to the current map to populate classification results
            if self.map_data is not None:
                logger.info("Automatically applying supervised model to map after training...")
                self.classify_map()  # This will populate the actual classification results
                logger.info("Auto-classification completed")
            else:
                # No map data, just create placeholder for feature dropdown
                logger.info("No map data available, creating placeholder classification results")
                self.update_map_features_with_classification()
            
            self.statusBar().showMessage(f"Supervised model trained: {model_type}")
            logger.info(f"Supervised training completed: {model_type}")
            
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"Error training supervised model:\n{str(e)}")
            logger.error(f"Supervised training error: {e}")
        finally:
            self.progress_status.hide_progress()
    
    def train_unsupervised_model(self):
        """Train unsupervised clustering model."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
        
        try:
            # Get parameters from control panel
            control_panel = self.get_current_ml_control_panel()
            if control_panel is None:
                QMessageBox.warning(self, "Error", "ML control panel not found.")
                return
            
            method = control_panel.get_clustering_method()
            n_clusters = control_panel.get_n_clusters()
            eps = control_panel.get_eps()
            min_samples = control_panel.get_min_samples()
            feature_options = control_panel.get_feature_options()
            
            self.progress_status.show_progress(f"Training {method} clustering...")
            
            # Prepare data from current map
            spectra_list = list(self.map_data.spectra.values())
            data_matrix = []
            valid_positions = []
            
            for spectrum in spectra_list:
                if spectrum.wavenumbers is not None:
                    intensities = (spectrum.processed_intensities 
                                 if self.use_processed and spectrum.processed_intensities is not None
                                 else spectrum.intensities)
                    if intensities is not None:
                        data_matrix.append(intensities)
                        valid_positions.append((spectrum.x_pos, spectrum.y_pos))
            
            if not data_matrix:
                QMessageBox.warning(self, "No Data", "No valid spectra found for clustering.")
                return
            
            import numpy as np
            X = np.array(data_matrix)
            
            # Apply feature transformation if requested
            feature_transformer = None
            feature_used = 'full'
            
            # Check and apply PCA features if requested
            if feature_options['use_pca']:
                if not hasattr(self, 'pca_analyzer') or self.pca_analyzer.pca is None:
                    QMessageBox.warning(
                        self, "PCA Not Available", 
                        "PCA features requested but no PCA model found.\n\n"
                        "Please run PCA analysis first:\n"
                        "1. Go to 'PCA & NMF Analysis' tab\n"
                        "2. Click 'Run PCA Analysis'\n"
                        "3. Return to ML tab and try clustering again"
                    )
                    return
                else:
                    feature_transformer = self.pca_analyzer
                    feature_used = 'pca'
            
            # Check and apply NMF features if requested  
            elif feature_options['use_nmf']:
                if not hasattr(self, 'nmf_analyzer') or self.nmf_analyzer.nmf is None:
                    QMessageBox.warning(
                        self, "NMF Not Available", 
                        "NMF features requested but no NMF model found.\n\n"
                        "Please run NMF analysis first:\n"
                        "1. Go to 'PCA & NMF Analysis' tab\n"
                        "2. Click 'Run NMF Analysis'\n"
                        "3. Return to ML tab and try clustering again"
                    )
                    return
                else:
                    feature_transformer = self.nmf_analyzer
                    feature_used = 'nmf'
            
            logger.info(f"Running {method} clustering on {X.shape[0]} spectra with {X.shape[1]} features using {feature_used} features")
            
            # Run clustering
            results = self.unsupervised_analyzer.train_clustering(
                X, 
                method=method,
                n_clusters=n_clusters,
                eps=eps,
                min_samples=min_samples,
                feature_transformer=feature_transformer
            )
            
            if results['success']:
                # Store results for map integration
                self.ml_results = results
                self.ml_results['type'] = 'unsupervised'
                self.ml_valid_positions = valid_positions
                
                # Switch to ML tab and plot results
                self.tab_widget.setCurrentIndex(3)  # ML tab is now at index 3, not 4
                self.plot_ml_results(results)
                
                # Update map features to include clustering results
                self.update_map_features_with_clustering()
                
                # Update control panel with results
                if control_panel:
                    results_info = (
                        f"Clustering Complete!\n\n"
                        f"Method: {method}\n"
                        f"Clusters found: {results['n_clusters']}\n"
                        f"Samples: {results['n_samples']}\n"
                        f"Silhouette score: {results['silhouette_score']:.3f}\n"
                    )
                    if method == 'DBSCAN':
                        results_info += f"Noise points: {results['n_noise']}\n"
                    results_info += "\nSwitch to Map View to visualize clusters."
                    
                    control_panel.update_info(results_info)
                
                self.statusBar().showMessage(
                    f"Clustering completed: {method}, {results['n_clusters']} clusters, "
                    f"silhouette: {results['silhouette_score']:.3f}"
                )
                
                logger.info(f"Clustering completed: {method}, {results['n_clusters']} clusters")
                
            else:
                QMessageBox.warning(self, "Clustering Error", 
                                  f"Clustering failed: {results.get('error', 'Unknown error')}")
                logger.error(f"Clustering failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            QMessageBox.critical(self, "Clustering Error", f"Error in clustering:\n{str(e)}")
            logger.error(f"Clustering error: {e}")
        finally:
            self.progress_status.hide_progress()
    
    def classify_map(self):
        """Apply trained model to classify the map."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
        
        # Check if we have trained models
        has_supervised = (self.supervised_analyzer.model is not None)
        has_unsupervised = (self.unsupervised_analyzer.model is not None)
        
        if not has_supervised and not has_unsupervised:
            QMessageBox.warning(self, "No Model", 
                              "Train a model first (supervised or unsupervised).")
            return
        
        try:
            control_panel = self.get_current_ml_control_panel()
            analysis_type = control_panel.get_analysis_type() if control_panel else "Supervised Classification"
            
            self.progress_status.show_progress("Classifying map...")
            
            # Prepare map data
            spectra_list = list(self.map_data.spectra.values())
            data_matrix = []
            valid_positions = []
            
            for spectrum in spectra_list:
                if spectrum.wavenumbers is not None:
                    intensities = (spectrum.processed_intensities 
                                 if self.use_processed and spectrum.processed_intensities is not None
                                 else spectrum.intensities)
                    if intensities is not None:
                        data_matrix.append(intensities)
                        valid_positions.append((spectrum.x_pos, spectrum.y_pos))
            
            if not data_matrix:
                QMessageBox.warning(self, "No Data", "No valid spectra found for classification.")
                return
            
            import numpy as np
            X = np.array(data_matrix)
            
            # Apply appropriate model
            if analysis_type == "Supervised Classification" and has_supervised:
                # Get feature options used during training
                control_panel = self.get_current_ml_control_panel()
                feature_options = control_panel.get_feature_options() if control_panel else {'use_pca': False, 'use_nmf': False}
                
                # Step 1: Align features to training wavenumbers if needed
                if hasattr(self.supervised_analyzer, 'training_wavenumbers'):
                    # Get the first spectrum's wavenumbers as reference
                    map_wavenumbers = None
                    for spectrum in self.map_data.spectra.values():
                        if spectrum.wavenumbers is not None:
                            map_wavenumbers = spectrum.wavenumbers
                            break
                    
                    if map_wavenumbers is not None and len(map_wavenumbers) != len(self.supervised_analyzer.training_wavenumbers):
                        # Automatically align features using interpolation
                        try:
                            self.statusBar().showMessage("Aligning map features to training data...")
                            X_aligned = self.align_features_to_training(X, self.supervised_analyzer.training_wavenumbers)
                            X = X_aligned
                            logger.info(f"Features aligned: {len(map_wavenumbers)} -> {len(self.supervised_analyzer.training_wavenumbers)} wavenumber points")
                        except Exception as e:
                            QMessageBox.critical(
                                self, "Feature Alignment Failed", 
                                f"Failed to align map features to training data:\n\n{str(e)}\n\n"
                                f"Training data: {len(self.supervised_analyzer.training_wavenumbers)} wavenumber points\n"
                                f"Map data: {len(map_wavenumbers)} wavenumber points\n\n"
                                "This usually happens when the wavenumber ranges don't overlap sufficiently.\n"
                                "Please ensure training and map data have compatible wavenumber ranges."
                            )
                            return
                
                # Step 2: Apply the same feature transformation as used during training
                feature_transformer = None
                feature_used = 'full'
                
                if feature_options['use_pca'] and hasattr(self, 'pca_analyzer') and self.pca_analyzer.pca is not None:
                    feature_transformer = self.pca_analyzer
                    feature_used = 'pca'
                elif feature_options['use_nmf'] and hasattr(self, 'nmf_analyzer') and self.nmf_analyzer.nmf is not None:
                    feature_transformer = self.nmf_analyzer
                    feature_used = 'nmf'
                
                if feature_transformer is not None:
                    try:
                        self.statusBar().showMessage(f"Applying {feature_used.upper()} transformation to map data...")
                        X_transformed, actual_type = feature_transformer.transform_data(X, fallback_to_full=True)
                        if X_transformed is not None:
                            X = X_transformed
                            logger.info(f"Applied {actual_type} transformation to map data: {X.shape}")
                        else:
                            logger.warning(f"Failed to apply {feature_used} transformation, using full spectrum")
                    except Exception as e:
                        logger.warning(f"Feature transformation failed: {str(e)}, using full spectrum")
                
                # Step 3: Use the trained model for prediction
                try:
                    predictions = self.supervised_analyzer.model.predict(X)
                    probabilities = None
                    if hasattr(self.supervised_analyzer.model, 'predict_proba'):
                        try:
                            probabilities = self.supervised_analyzer.model.predict_proba(X)
                        except:
                            probabilities = None
                except ValueError as e:
                    if "features" in str(e):
                        QMessageBox.critical(
                            self, "Feature Dimension Mismatch", 
                            f"Cannot apply model to map data:\n\n{str(e)}\n\n"
                            f"Map data shape after processing: {X.shape}\n"
                            f"Expected features: {getattr(self.supervised_analyzer.model, 'n_features_in_', 'unknown')}\n\n"
                            "Feature alignment failed. This usually happens when:\n"
                            "• Training data and map data have different wavenumber ranges\n"
                            "• Different spectrometer settings were used\n"
                            "• Different preprocessing was applied\n"
                            "• PCA/NMF transformation parameters differ\n\n"
                            "Solution: Ensure training and map data have compatible wavenumber ranges."
                        )
                        return
                    else:
                        raise
                        
                self.classification_results = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'positions': valid_positions,
                    'type': 'supervised',
                    'class_names': getattr(self.supervised_analyzer, 'class_names', None)
                }
            else:  # Unsupervised clustering
                if hasattr(self, 'ml_results') and self.ml_results.get('type') == 'unsupervised':
                    # Use existing clustering results
                    self.classification_results = {
                        'predictions': self.ml_results['labels'],
                        'positions': valid_positions,
                        'type': 'unsupervised'
                    }
                else:
                    QMessageBox.warning(self, "No Clustering", "Run unsupervised training first.")
                    return
            
            # Update map features with classification results
            self.update_map_features_with_classification()
            
            # Debug and fix map features if needed
            self.debug_and_fix_map_features()
            
            # Force a manual refresh of the map tab controls as a workaround
            try:
                # Try to force refresh the map controls
                if hasattr(self, 'tab_widget'):
                    # Switch away and back to trigger refresh
                    current_tab = self.tab_widget.currentIndex()
                    if current_tab != 0:  # If not already on map tab
                        self.tab_widget.setCurrentIndex(0)  # Switch to map tab
                    else:
                        # If already on map tab, briefly switch to another tab and back
                        self.tab_widget.setCurrentIndex(1)  # Switch away
                        self.tab_widget.setCurrentIndex(0)  # Switch back
                        
                logger.info("Forced refresh of map tab controls")
            except Exception as refresh_error:
                logger.warning(f"Could not force refresh map controls: {refresh_error}")
            
            # Generate comprehensive results automatically
            self.plot_comprehensive_results()
            
            # Switch to map view to show results
            self.tab_widget.setCurrentIndex(0)
            
            QMessageBox.information(
                self, "Classification Complete", 
                "Map classification completed. Check the Map View and Results tab for comprehensive analysis."
            )
            
            self.statusBar().showMessage("Map classification completed - Comprehensive results generated")
            logger.info("Map classification completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Classification Error", f"Error classifying map:\n{str(e)}")
            logger.error(f"Map classification error: {e}")
        finally:
            self.progress_status.hide_progress()
    
    def get_current_ml_control_panel(self):
        """Get the current ML control panel."""
        for name, section in self.controls_panel.sections.items():
            if name == "ml_controls":
                return section['widget']
        return None
    
    def get_current_map_control_panel(self):
        """Get the current map control panel."""
        for name, section in self.controls_panel.sections.items():
            if name == "map_controls":
                return section['widget']
        return None
    
    def plot_ml_results(self, results):
        """Plot ML analysis results."""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Clear the existing plot
            self.ml_plot_widget.figure.clear()
            
            if results.get('type') == 'unsupervised':
                # Plot clustering results
                self._plot_clustering_results(results)
            else:
                # Plot supervised learning results
                self._plot_supervised_results(results)
            
            plt.tight_layout()
            self.ml_plot_widget.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error plotting ML results: {e}")
            # Fallback to simple plot
            self.ml_plot_widget.figure.clear()
            ax = self.ml_plot_widget.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'ML Analysis Complete\n{results.get("method", "Model")} ready for use', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('ML Analysis Results')
            ax.axis('off')
            self.ml_plot_widget.canvas.draw()
    
    def _plot_clustering_results(self, results):
        """Plot clustering analysis results."""
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        labels = results['labels']
        n_clusters = results['n_clusters']
        method = results['method']
        silhouette = results['silhouette_score']
        
        gs = GridSpec(2, 2, figure=self.ml_plot_widget.figure)
        
        # Plot 1: Cluster distribution
        ax1 = self.ml_plot_widget.figure.add_subplot(gs[0, 0])
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Handle DBSCAN noise points
        if method == 'DBSCAN' and -1 in unique_labels:
            noise_idx = np.where(unique_labels == -1)[0][0]
            colors = ['red' if i == noise_idx else f'C{i}' for i in range(len(unique_labels))]
            labels_for_plot = ['Noise' if l == -1 else f'Cluster {l}' for l in unique_labels]
        else:
            colors = [f'C{i}' for i in range(len(unique_labels))]
            labels_for_plot = [f'Cluster {l}' for l in unique_labels]
        
        bars = ax1.bar(range(len(unique_labels)), counts, color=colors)
        ax1.set_title(f'{method} - Cluster Distribution')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(range(len(unique_labels)))
        ax1.set_xticklabels(labels_for_plot, rotation=45)
        
        # Plot 2: Clustering metrics
        ax2 = self.ml_plot_widget.figure.add_subplot(gs[0, 1])
        ax2.text(0.1, 0.8, f'Method: {method}', transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.7, f'Clusters: {n_clusters}', transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.6, f'Samples: {results["n_samples"]}', transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.5, f'Silhouette Score: {silhouette:.3f}', transform=ax2.transAxes, fontsize=12)
        
        if method == 'DBSCAN':
            ax2.text(0.1, 0.4, f'Noise Points: {results["n_noise"]}', transform=ax2.transAxes, fontsize=12)
        
        ax2.set_title('Clustering Metrics')
        ax2.axis('off')
        
        # Plot 3: Cluster size pie chart
        ax3 = self.ml_plot_widget.figure.add_subplot(gs[1, :])
        if len(unique_labels) <= 10:  # Only show pie chart if not too many clusters
            # Create pie chart without labels on the chart itself
            wedges, texts, autotexts = ax3.pie(counts, autopct='%1.1f%%', startangle=90, colors=colors)
            ax3.set_title('Cluster Size Distribution')
            
            # Create legend with cluster labels
            ax3.legend(wedges, labels_for_plot, 
                      title="Clusters",
                      loc="center left", 
                      bbox_to_anchor=(1, 0, 0.5, 1))
        else:
            ax3.text(0.5, 0.5, f'Too many clusters ({len(unique_labels)}) for pie chart', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.axis('off')
    
    def _plot_supervised_results(self, results):
        """Plot supervised learning results."""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create subplots for different metrics
        gs = GridSpec(2, 2, figure=self.ml_plot_widget.figure)
        
        # Plot 1: Training Summary
        ax1 = self.ml_plot_widget.figure.add_subplot(gs[0, 0])
        ax1.text(0.1, 0.9, f'Model: {results.get("model_type", "Unknown")}', transform=ax1.transAxes, fontsize=12, weight='bold')
        ax1.text(0.1, 0.8, f'Classes: {results.get("n_classes", "Unknown")}', transform=ax1.transAxes, fontsize=11)
        ax1.text(0.1, 0.7, f'Accuracy: {results.get("accuracy", 0):.3f}', transform=ax1.transAxes, fontsize=11)
        ax1.text(0.1, 0.6, f'Training samples: {results.get("n_train_samples", 0)}', transform=ax1.transAxes, fontsize=11)
        ax1.text(0.1, 0.5, f'Test samples: {results.get("n_test_samples", 0)}', transform=ax1.transAxes, fontsize=11)
        ax1.text(0.1, 0.4, f'Features: {results.get("n_features", 0)}', transform=ax1.transAxes, fontsize=11)
        ax1.text(0.1, 0.2, '✅ Ready for map classification', transform=ax1.transAxes, fontsize=11, color='green', weight='bold')
        ax1.set_title('Training Summary')
        ax1.axis('off')
        
        # Plot 2: Class Information
        ax2 = self.ml_plot_widget.figure.add_subplot(gs[0, 1])
        class_names = results.get('class_names', [])
        if class_names:
            y_pos = np.arange(len(class_names))
            ax2.barh(y_pos, [1]*len(class_names), color=[f'C{i}' for i in range(len(class_names))])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(class_names)
            ax2.set_xlabel('Class')
            ax2.set_title('Class Overview')
        else:
            ax2.text(0.5, 0.5, 'No class information available', ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        # Plot 3: Performance indicator
        ax3 = self.ml_plot_widget.figure.add_subplot(gs[1, :])
        accuracy = results.get('accuracy', 0)
        
        # Create a simple accuracy bar
        colors = ['red' if accuracy < 0.6 else 'orange' if accuracy < 0.8 else 'green']
        bars = ax3.bar(['Model Accuracy'], [accuracy], color=colors[0])
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Model Performance')
        
        # Add accuracy text on the bar
        if accuracy > 0:
            ax3.text(0, accuracy + 0.02, f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add reference lines
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
        ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good performance')
        ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent performance')
        ax3.legend(loc='upper right', fontsize=9)
    
    def update_map_features_with_clustering(self):
        """Update map view features to include clustering results."""
        if not hasattr(self, 'ml_results') or self.ml_results.get('type') != 'unsupervised':
            return
            
        try:
            # Get the map control panel
            control_panel = None
            for name, section in self.controls_panel.sections.items():
                if name == "map_controls" and hasattr(section['widget'], 'feature_combo'):
                    control_panel = section['widget']
                    break
            
            if control_panel is None:
                return
            
            # Add clustering options to feature combo
            current_features = [control_panel.feature_combo.itemText(i) 
                              for i in range(control_panel.feature_combo.count())]
            
            cluster_feature_name = f"ML Clusters ({self.ml_results['method']})"
            if cluster_feature_name not in current_features:
                control_panel.feature_combo.addItem(cluster_feature_name)
            
            logger.info(f"Added clustering results to map features: {cluster_feature_name}")
            
        except Exception as e:
            logger.error(f"Error updating map features with clustering: {e}")
    
    def update_map_features_with_classification(self):
        """Update map view features to include classification results."""
        if not hasattr(self, 'classification_results'):
            return
            
        try:
            # Get the map control panel (should be available since we're called from map tab creation)
            control_panel = None
            for name, section in self.controls_panel.sections.items():
                if name == "map_controls" and hasattr(section.get('widget'), 'feature_combo'):
                    control_panel = section['widget']
                    break
            
            if control_panel is None:
                logger.warning("Could not find map control panel for classification features")
                return
            
            # Add classification options to feature combo
            current_features = [control_panel.feature_combo.itemText(i) 
                              for i in range(control_panel.feature_combo.count())]
            
            class_type = self.classification_results['type']
            feature_name = f"ML Classification ({class_type.title()})"
            
            if feature_name not in current_features:
                control_panel.feature_combo.addItem(feature_name)
                logger.info(f"Added classification feature: {feature_name}")
            
            # Also add individual class probability features if available
            if ('class_names' in self.classification_results and 
                self.classification_results['class_names'] is not None):
                
                class_names = self.classification_results['class_names']
                for class_name in class_names:
                    prob_feature_name = f"Class Probability: {class_name}"
                    if prob_feature_name not in current_features:
                        control_panel.feature_combo.addItem(prob_feature_name)
                        logger.info(f"Added class probability feature: {prob_feature_name}")
            
            logger.info(f"Classification features added to map dropdown. Total features: {control_panel.feature_combo.count()}")
            
        except Exception as e:
            logger.error(f"Error updating map features with classification: {e}")
    
    def create_ml_classification_map(self):
        """Create a map showing ML classification results."""
        if not hasattr(self, 'classification_results'):
            return None
            
        try:
            import numpy as np
            
            predictions = self.classification_results['predictions']
            positions = self.classification_results['positions']
            
            # Create position to prediction mapping
            pos_to_prediction = {}
            for i, (x, y) in enumerate(positions):
                pos_to_prediction[(x, y)] = predictions[i]
            
            # Create map array
            all_positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            map_array = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            for spectrum in self.map_data.spectra.values():
                pos_key = (spectrum.x_pos, spectrum.y_pos)
                if pos_key in pos_to_prediction:
                    map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = pos_to_prediction[pos_key]
            
            return map_array
            
        except Exception as e:
            logger.error(f"Error creating ML classification map: {e}")
            return None
    
    def create_class_probability_map(self, class_name: str):
        """Create a map showing the probability of a specific class."""
        if not hasattr(self, 'classification_results'):
            return None
            
        try:
            import numpy as np
            
            probabilities = self.classification_results.get('probabilities')
            positions = self.classification_results['positions']
            class_names = self.classification_results.get('class_names')
            
            if probabilities is None or class_names is None:
                logger.warning("No probabilities or class names available for probability map")
                return None
            
            # Find the class index
            try:
                class_index = list(class_names).index(class_name)
            except ValueError:
                logger.error(f"Class '{class_name}' not found in class names: {class_names}")
                return None
            
            # Create position to probability mapping
            pos_to_probability = {}
            for i, (x, y) in enumerate(positions):
                if i < len(probabilities):
                    pos_to_probability[(x, y)] = probabilities[i][class_index]
            
            # Create map array
            all_positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            map_array = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            for spectrum in self.map_data.spectra.values():
                pos_key = (spectrum.x_pos, spectrum.y_pos)
                if pos_key in pos_to_probability:
                    map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = pos_to_probability[pos_key]
            
            return map_array
            
        except Exception as e:
            logger.error(f"Error creating class probability map for '{class_name}': {e}")
            return None
    
    def save_named_model(self):
        """Save trained ML model with a custom name."""
        control_panel = self.get_current_ml_control_panel()
        if control_panel is None:
            return
            
        # Get model name from user input
        model_name = control_panel.get_model_name()
        if not model_name:
            QMessageBox.warning(self, "Model Name Required", 
                              "Please enter a name for the model.")
            return
            
        analysis_type = control_panel.get_analysis_type()
        
        try:
            if analysis_type == "Supervised Classification":
                if self.supervised_analyzer.model is None:
                    QMessageBox.warning(self, "No Model", "No supervised model to save.")
                    return
                analyzer = self.supervised_analyzer
                model_type = "supervised"
            else:
                if self.unsupervised_analyzer.model is None:
                    QMessageBox.warning(self, "No Model", "No unsupervised model to save.")
                    return
                analyzer = self.unsupervised_analyzer
                model_type = "unsupervised"
            
            # Get training information
            training_info = {
                'analysis_type': analysis_type,
                'feature_options': control_panel.get_feature_options()
            }
            
            # Add model to the manager
            if self.model_manager.add_model(model_name, analyzer, model_type, training_info):
                # Update UI
                control_panel.add_model_to_list(model_name)
                control_panel.clear_model_name()
                
                # Save to file automatically
                self._save_models_to_file()
                
                # Update info display
                info_text = f"Model '{model_name}' saved successfully!\n"
                info_text += f"Type: {analysis_type}\n"
                info_text += f"Saved to: {self.models_file}\n"
                if hasattr(analyzer, 'training_results') and analyzer.training_results:
                    accuracy = analyzer.training_results.get('accuracy', 'N/A')
                    info_text += f"Accuracy: {accuracy:.3f}" if isinstance(accuracy, float) else f"Accuracy: {accuracy}"
                
                control_panel.update_info(info_text)
                
                QMessageBox.information(self, "Save Complete", 
                                      f"Model '{model_name}' saved successfully to {self.models_file}")
                self.statusBar().showMessage(f"Model '{model_name}' added to collection and saved to file")
            else:
                QMessageBox.warning(self, "Save Failed", 
                                  f"Failed to save model '{model_name}'.")
                    
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving model '{model_name}':\n{str(e)}")
            logger.error(f"Save named model error: {e}")
    
    def remove_selected_model(self):
        """Remove the currently selected model."""
        control_panel = self.get_current_ml_control_panel()
        if control_panel is None:
            return
            
        selected_model = control_panel.get_selected_model()
        if not selected_model:
            QMessageBox.warning(self, "No Model Selected", 
                              "Please select a model to remove.")
            return
            
        # Confirm removal
        reply = QMessageBox.question(
            self, "Confirm Removal", 
            f"Are you sure you want to remove model '{selected_model}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if self.model_manager.remove_model(selected_model):
                control_panel.remove_selected_model()
                
                # Save to file after removal
                self._save_models_to_file()
                
                QMessageBox.information(self, "Model Removed", 
                                      f"Model '{selected_model}' removed successfully.")
                self.statusBar().showMessage(f"Model '{selected_model}' removed and changes saved")
            else:
                QMessageBox.warning(self, "Remove Failed", 
                                  f"Failed to remove model '{selected_model}'.")
    
    def apply_selected_model(self):
        """Apply the selected model to the current map."""
        control_panel = self.get_current_ml_control_panel()
        if control_panel is None:
            return
            
        selected_model = control_panel.get_selected_model()
        if not selected_model:
            QMessageBox.warning(self, "No Model Selected", 
                              "Please select a model to apply.")
            return
            
        if self.map_data is None:
            QMessageBox.warning(self, "No Map Data", 
                              "Please load map data first.")
            return
            
        try:
            # Get model info
            model_info = self.model_manager.get_model_info(selected_model)
            if not model_info:
                QMessageBox.warning(self, "Model Error", 
                                  f"Could not get information for model '{selected_model}'.")
                return
                
            model_type = model_info['type']
            
            self.progress_status.show_progress(f"Applying model '{selected_model}' to map...")
            
            # Load model into appropriate analyzer
            if model_type == 'supervised':
                analyzer = self.supervised_analyzer
            else:
                analyzer = self.unsupervised_analyzer
                
            if not self.model_manager.load_model_into_analyzer(selected_model, analyzer):
                QMessageBox.warning(self, "Load Error", 
                                  f"Failed to load model '{selected_model}'.")
                return
            
            # Apply to map data
            if model_type == 'supervised':
                self._apply_supervised_model_to_map(selected_model, analyzer)
            else:
                self._apply_unsupervised_model_to_map(selected_model, analyzer)
                
            self.progress_status.hide_progress()
            self.statusBar().showMessage(f"Applied model '{selected_model}' to map")
            
            # Try to refresh map features to make the model available in the dropdown
            self.refresh_map_features_with_models()
            
            # Update info display
            info_text = f"Applied model: {selected_model}\n"
            info_text += f"Type: {model_info['type']}\n"
            info_text += f"Algorithm: {model_info['algorithm']}"
            control_panel.update_info(info_text)
            
        except Exception as e:
            self.progress_status.hide_progress()
            QMessageBox.critical(self, "Apply Error", 
                               f"Error applying model '{selected_model}':\n{str(e)}")
            logger.error(f"Apply model error: {e}")
    
    def on_model_selection_changed(self, model_name: str):
        """Handle model selection change."""
        if not model_name or model_name == "No models loaded":
            return
            
        # Update info display with model details
        control_panel = self.get_current_ml_control_panel()
        if control_panel is None:
            return
            
        model_info = self.model_manager.get_model_info(model_name)
        if model_info:
            info_text = f"Selected: {model_name}\n"
            info_text += f"Type: {model_info['type']}\n"
            info_text += f"Algorithm: {model_info['algorithm']}\n"
            info_text += f"Created: {model_info['created_at'][:10]}"  # Show date only
            control_panel.update_info(info_text)
    
    def _apply_supervised_model_to_map(self, model_name: str, analyzer):
        """Apply supervised model to map data."""
        try:
            import numpy as np
            
            # Prepare map data for classification
            map_intensities = []
            positions = []
            
            for spectrum in self.map_data.spectra.values():
                intensities = (spectrum.processed_intensities 
                             if self.use_processed and spectrum.processed_intensities is not None
                             else spectrum.intensities)
                map_intensities.append(intensities)
                positions.append((spectrum.x_pos, spectrum.y_pos))
            
            map_intensities = np.array(map_intensities)
            
            # Apply preprocessing if available
            if hasattr(self, 'preprocessor'):
                processed_intensities = []
                for i, spectrum in enumerate(self.map_data.spectra.values()):
                    processed = self.preprocessor(spectrum.wavenumbers, map_intensities[i])
                    processed_intensities.append(processed)
                map_intensities = np.array(processed_intensities)
            
            # Classify the map data
            results = analyzer.classify_data(map_intensities)
            
            if results['success']:
                # Create classification map
                self._create_model_classification_map(model_name, results, positions)
                
                # Update map features
                map_control_panel = self.get_current_map_control_panel()
                if map_control_panel is not None:
                    map_features = map_control_panel.get_available_features()
                    classification_feature = f"Model: {model_name}"
                    if classification_feature not in map_features:
                        map_features.append(classification_feature)
                        map_control_panel.update_feature_list(map_features)
                
            else:
                QMessageBox.warning(self, "Classification Failed", 
                                  f"Failed to classify map with model '{model_name}':\n{results.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error applying supervised model to map: {e}")
            raise
    
    def _apply_unsupervised_model_to_map(self, model_name: str, analyzer):
        """Apply unsupervised model to map data."""
        try:
            import numpy as np
            
            # Prepare map data for clustering prediction
            map_intensities = []
            positions = []
            
            for spectrum in self.map_data.spectra.values():
                intensities = (spectrum.processed_intensities 
                             if self.use_processed and spectrum.processed_intensities is not None
                             else spectrum.intensities)
                map_intensities.append(intensities)
                positions.append((spectrum.x_pos, spectrum.y_pos))
            
            map_intensities = np.array(map_intensities)
            
            # Apply preprocessing if available
            if hasattr(self, 'preprocessor'):
                processed_intensities = []
                for i, spectrum in enumerate(self.map_data.spectra.values()):
                    processed = self.preprocessor(spectrum.wavenumbers, map_intensities[i])
                    processed_intensities.append(processed)
                map_intensities = np.array(processed_intensities)
            
            # Predict clusters for the map data
            results = analyzer.predict_clusters(map_intensities)
            
            if results['success']:
                # Create clustering map
                self._create_model_clustering_map(model_name, results, positions)
                
                # Update map features
                map_control_panel = self.get_current_map_control_panel()
                if map_control_panel is not None:
                    map_features = map_control_panel.get_available_features()
                    clustering_feature = f"Model: {model_name}"
                    if clustering_feature not in map_features:
                        map_features.append(clustering_feature)
                        map_control_panel.update_feature_list(map_features)
                        logger.info(f"Added '{clustering_feature}' to map features dropdown")
                    else:
                        logger.info(f"'{clustering_feature}' already exists in map features")
                else:
                    logger.warning("Map control panel is None - cannot update feature dropdown")
                    logger.info(f"Model results stored as 'Model: {model_name}' - you can access it programmatically")
                
            else:
                QMessageBox.warning(self, "Clustering Failed", 
                                  f"Failed to cluster map with model '{model_name}':\n{results.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error applying unsupervised model to map: {e}")
            raise
    
    def _create_model_classification_map(self, model_name: str, results, positions):
        """Create a map showing classification results from a named model."""
        try:
            import numpy as np
            
            # Create a grid for the classification map
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Create map grid
            map_data = np.full((len(set(y_coords)), len(set(x_coords))), np.nan)
            
            # Populate the grid with classification results
            predictions = results['predictions']
            probabilities = results['probabilities']
            
            # Store results for map features
            if not hasattr(self, 'model_results'):
                self.model_results = {}
            
            self.model_results[f"Model: {model_name}"] = {
                'type': 'classification',
                'predictions': predictions,
                'probabilities': probabilities,
                'positions': positions,
                'map_data': map_data
            }
            
        except Exception as e:
            logger.error(f"Error creating model classification map: {e}")
            raise
    
    def _create_model_clustering_map(self, model_name: str, results, positions):
        """Create a map showing clustering results from a named model."""
        try:
            import numpy as np
            
            # Create a grid for the clustering map
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Create map grid
            map_data = np.full((len(set(y_coords)), len(set(x_coords))), np.nan)
            
            # Populate the grid with clustering results
            cluster_labels = results['labels']
            
            # Store results for map features
            if not hasattr(self, 'model_results'):
                self.model_results = {}
            
            self.model_results[f"Model: {model_name}"] = {
                'type': 'clustering',
                'labels': cluster_labels,
                'positions': positions,
                'map_data': map_data
            }
            
        except Exception as e:
            logger.error(f"Error creating model clustering map: {e}")
            raise
    
    def create_model_result_map(self):
        """Create a map showing results from a named model."""
        if not hasattr(self, 'model_results'):
            logger.warning("No model_results attribute found")
            return None
            
        try:
            import numpy as np
            
            model_feature = self.current_feature
            logger.info(f"Creating model result map for feature: {model_feature}")
            logger.info(f"Available model results: {list(self.model_results.keys())}")
            
            if model_feature not in self.model_results:
                logger.warning(f"Model results not found for: {model_feature}")
                logger.warning(f"Available keys: {list(self.model_results.keys())}")
                return None
            
            result_data = self.model_results[model_feature]
            
            # Get positions and create coordinate arrays
            positions = result_data['positions']
            x_coords = sorted(set(pos[0] for pos in positions))
            y_coords = sorted(set(pos[1] for pos in positions))
            
            # Create map grid
            map_data = np.full((len(y_coords), len(x_coords)), np.nan)
            
            # Create mapping from coordinates to grid indices
            x_to_idx = {x: i for i, x in enumerate(x_coords)}
            y_to_idx = {y: i for i, y in enumerate(y_coords)}
            
            # Fill the map with results
            if result_data['type'] == 'classification':
                predictions = result_data['predictions']
                for i, (x, y) in enumerate(positions):
                    grid_y = y_to_idx[y]
                    grid_x = x_to_idx[x]
                    map_data[grid_y, grid_x] = predictions[i]
            else:  # clustering
                labels = result_data['labels']
                for i, (x, y) in enumerate(positions):
                    grid_y = y_to_idx[y]
                    grid_x = x_to_idx[x]
                    map_data[grid_y, grid_x] = labels[i]
            
            return map_data
            
        except Exception as e:
            logger.error(f"Error creating model result map: {e}")
            return None
    
    def load_ml_model(self):
        """Load trained ML model from file and add to model manager."""
        try:
            control_panel = self.get_current_ml_control_panel()
            if control_panel is None:
                return
            
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load ML Model", 
                str(self.models_dir),  # Start in the models directory
                "Pickle Files (*.pkl)")
            
            if filepath:
                # Try to load the model into the model manager
                model_name = self.model_manager.load_single_model(filepath)
                
                if model_name:
                    # Update the control panel to show the new model
                    control_panel.add_model_to_list(model_name)
                    
                    # Save the updated model collection
                    self._save_models_to_file()
                    
                    # Get model info for display
                    model_info = self.model_manager.get_model_info(model_name)
                    
                    info_text = f"Model loaded: {model_name}\n"
                    if model_info:
                        info_text += f"Type: {model_info['type']}\n"
                        info_text += f"Algorithm: {model_info['algorithm']}\n"
                    info_text += f"Added to collection and ready for use."
                    
                    control_panel.update_info(info_text)
                    
                    QMessageBox.information(self, "Load Complete", 
                                          f"Model '{model_name}' loaded and added to collection.")
                    self.statusBar().showMessage(f"Model '{model_name}' loaded from {filepath}")
                else:
                    QMessageBox.warning(self, "Load Failed", 
                                      "Failed to load ML model from file.")
                    
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading ML model:\n{str(e)}")
            logger.error(f"Load ML model error: {e}")
    
    def align_features_to_training(self, map_data, training_wavenumbers):
        """Align map data features to match training data wavenumbers."""
        import numpy as np
        from scipy.interpolate import interp1d
        
        aligned_data = []
        
        # Get map wavenumbers (assuming all spectra have same wavenumbers)
        map_wavenumbers = None
        if hasattr(self, 'map_data') and self.map_data.spectra:
            for spectrum in self.map_data.spectra.values():
                if spectrum.wavenumbers is not None:
                    map_wavenumbers = spectrum.wavenumbers
                    break
        
        if map_wavenumbers is None:
            raise ValueError("Cannot find wavenumbers in map data")
        
        # Check if wavenumber ranges overlap sufficiently
        map_min, map_max = map_wavenumbers.min(), map_wavenumbers.max()
        train_min, train_max = training_wavenumbers.min(), training_wavenumbers.max()
        
        overlap_min = max(map_min, train_min)
        overlap_max = min(map_max, train_max)
        
        if overlap_min >= overlap_max:
            raise ValueError(f"No wavenumber overlap between map ({map_min:.1f}-{map_max:.1f}) and training ({train_min:.1f}-{train_max:.1f}) data")
        
        overlap_fraction = (overlap_max - overlap_min) / (train_max - train_min)
        if overlap_fraction < 0.5:
            logger.warning(f"Limited wavenumber overlap ({overlap_fraction:.1%}). Results may be unreliable.")
        
        for spectrum_data in map_data:
            # Apply preprocessing first (same as during training)
            if hasattr(self, 'preprocessor'):
                try:
                    processed_spectrum = self.preprocessor(map_wavenumbers, spectrum_data)
                except:
                    processed_spectrum = spectrum_data
            else:
                processed_spectrum = spectrum_data
            
            # Interpolate spectrum to match training wavenumbers
            try:
                interp_func = interp1d(
                    map_wavenumbers, processed_spectrum, 
                    kind='linear', bounds_error=False, fill_value=0.0
                )
                aligned_spectrum = interp_func(training_wavenumbers)
                
                # Remove any NaN or infinite values that might result from interpolation
                aligned_spectrum = np.nan_to_num(aligned_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
                aligned_data.append(aligned_spectrum)
                
            except Exception as e:
                logger.warning(f"Interpolation failed for a spectrum: {str(e)}")
                # Fallback: pad or truncate to match training length
                if len(processed_spectrum) < len(training_wavenumbers):
                    # Pad with zeros
                    padded = np.pad(processed_spectrum, (0, len(training_wavenumbers) - len(processed_spectrum)))
                    aligned_data.append(padded)
                else:
                    # Truncate
                    aligned_data.append(processed_spectrum[:len(training_wavenumbers)])
        
        aligned_array = np.array(aligned_data)
        logger.info(f"Feature alignment complete: {map_data.shape} -> {aligned_array.shape}")
        return aligned_array

    def preprocessor(self, wavenumbers, intensities):
        """Preprocessor function for ML training data."""
        # This is a placeholder - you can add specific preprocessing here
        return intensities
        
    # Results Methods
    def generate_report(self):
        """Generate comprehensive analysis report."""
        # Plot comprehensive results first
        self.plot_comprehensive_results()
        # Switch to results tab
        self.tab_widget.setCurrentIndex(4)  # Updated index for results tab
        self.statusBar().showMessage("Comprehensive analysis report generated")
        
    def plot_comprehensive_results(self):
        """Plot comprehensive analysis results with 2x2 layout."""
        try:
            # Import necessary modules including matplotlib config
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from ui.matplotlib_config import configure_compact_ui
            configure_compact_ui()
            
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Clear the figure
            self.results_plot_widget.figure.clear()
            
            # Create 2x2 grid
            gs = GridSpec(2, 2, figure=self.results_plot_widget.figure, hspace=0.3, wspace=0.3)
            
            # Plot 1: PCA scatter with positive groups
            ax1 = self.results_plot_widget.figure.add_subplot(gs[0, 0])
            self._plot_pca_scatter_with_positive_groups(ax1)
            
            # Plot 2: NMF scatter with positive groups  
            ax2 = self.results_plot_widget.figure.add_subplot(gs[0, 1])
            self._plot_nmf_scatter_with_positive_groups(ax2)
            
            # Plot 3: Top 5 spectral matches
            ax3 = self.results_plot_widget.figure.add_subplot(gs[1, 0])
            self._plot_top_spectral_matches(ax3)
            
            # Plot 4: Component analysis and statistics
            ax4 = self.results_plot_widget.figure.add_subplot(gs[1, 1])
            self._plot_component_statistics(ax4)
            
            # Update statistics text
            self._update_statistics_text()
            
            # Draw the canvas
            self.results_plot_widget.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error plotting comprehensive results: {e}")
            # Show error message
            self.results_plot_widget.figure.clear()
            ax = self.results_plot_widget.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error generating comprehensive results:\n{str(e)}\n\nPlease ensure all analysis steps are complete.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange"))
            ax.set_title('Comprehensive Results - Error')
            ax.axis('off')
            self.results_plot_widget.canvas.draw()
            
    def _plot_pca_scatter_with_positive_groups(self, ax):
        """Plot PCA scatter plot with positive groups color-coded."""
        if not hasattr(self, 'pca_results') or self.pca_results is None:
            ax.text(0.5, 0.5, 'No PCA results available.\nRun PCA analysis first.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('PCA Analysis - Not Available')
            ax.axis('off')
            return
            
        try:
            # PCA results use 'components' key, not 'transformed_data'
            pca_transformed = self.pca_results.get('components', self.pca_results.get('transformed_data'))
            if pca_transformed is None:
                ax.text(0.5, 0.5, 'PCA transformed data not available.\nCheck PCA results structure.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('PCA Analysis - Data Not Available')
                ax.axis('off')
                return
            
            # Get positive groups if available
            positive_mask = self._get_positive_groups_mask()
            
            if positive_mask is not None:
                # Plot negative groups first (background)
                negative_indices = ~positive_mask
                if np.any(negative_indices):
                    ax.scatter(pca_transformed[negative_indices, 0], pca_transformed[negative_indices, 1], 
                             c='lightgray', alpha=0.3, s=20, label='Background/Negative')
                
                # Plot positive groups prominently
                if np.any(positive_mask):
                    ax.scatter(pca_transformed[positive_mask, 0], pca_transformed[positive_mask, 1], 
                             c='red', alpha=0.8, s=40, label='Positive Groups', edgecolors='darkred', linewidths=0.5)
            else:
                # No classification available, use clustering if available
                if hasattr(self, 'pca_results') and 'cluster_labels' in self.pca_results:
                    cluster_labels = self.pca_results['cluster_labels']
                    unique_labels = np.unique(cluster_labels)
                    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                    
                    for i, label in enumerate(unique_labels):
                        mask = cluster_labels == label
                        ax.scatter(pca_transformed[mask, 0], pca_transformed[mask, 1], 
                                 c=[colors[i]], alpha=0.7, s=30, label=f'Cluster {label}')
                else:
                    # Basic PCA plot
                    ax.scatter(pca_transformed[:, 0], pca_transformed[:, 1], 
                             c='blue', alpha=0.6, s=20)
            
            ax.set_xlabel(f'PC1 ({self.pca_results["explained_variance_ratio"][0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({self.pca_results["explained_variance_ratio"][1]:.1%} variance)')
            ax.set_title('PCA Analysis with Positive Groups')
            ax.grid(True, alpha=0.3)
            if positive_mask is not None or (hasattr(self, 'pca_results') and 'cluster_labels' in self.pca_results):
                ax.legend(fontsize=8)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting PCA results:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('PCA Analysis - Error')
            ax.axis('off')
            
    def _plot_nmf_scatter_with_positive_groups(self, ax):
        """Plot NMF scatter plot with positive groups color-coded."""
        if not hasattr(self, 'nmf_results') or self.nmf_results is None:
            ax.text(0.5, 0.5, 'No NMF results available.\nRun NMF analysis first.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('NMF Analysis - Not Available')
            ax.axis('off')
            return
            
        try:
            # NMF results use 'components' key, not 'transformed_data'
            nmf_transformed = self.nmf_results.get('components', self.nmf_results.get('transformed_data'))
            if nmf_transformed is None:
                ax.text(0.5, 0.5, 'NMF transformed data not available.\nCheck NMF results structure.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('NMF Analysis - Data Not Available')
                ax.axis('off')
                return
            
            # Use first two components for scatter plot
            if nmf_transformed.shape[1] < 2:
                ax.text(0.5, 0.5, 'NMF results have less than 2 components.\nCannot create scatter plot.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('NMF Analysis - Insufficient Components')
                ax.axis('off')
                return
            
            # Get positive groups if available
            positive_mask = self._get_positive_groups_mask()
            
            if positive_mask is not None:
                # Plot negative groups first (background)
                negative_indices = ~positive_mask
                if np.any(negative_indices):
                    ax.scatter(nmf_transformed[negative_indices, 0], nmf_transformed[negative_indices, 1], 
                             c='lightgray', alpha=0.3, s=20, label='Background/Negative')
                
                # Plot positive groups prominently
                if np.any(positive_mask):
                    ax.scatter(nmf_transformed[positive_mask, 0], nmf_transformed[positive_mask, 1], 
                             c='orange', alpha=0.8, s=40, label='Positive Groups', edgecolors='darkorange', linewidths=0.5)
            else:
                # No classification available, use clustering if available
                if hasattr(self, 'nmf_results') and 'cluster_labels' in self.nmf_results:
                    cluster_labels = self.nmf_results['cluster_labels']
                    unique_labels = np.unique(cluster_labels)
                    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
                    
                    for i, label in enumerate(unique_labels):
                        mask = cluster_labels == label
                        ax.scatter(nmf_transformed[mask, 0], nmf_transformed[mask, 1], 
                                 c=[colors[i]], alpha=0.7, s=30, label=f'Cluster {label}')
                else:
                    # Basic NMF plot
                    ax.scatter(nmf_transformed[:, 0], nmf_transformed[:, 1], 
                             c='green', alpha=0.6, s=20)
            
            ax.set_xlabel('NMF Component 1')
            ax.set_ylabel('NMF Component 2')
            ax.set_title('NMF Analysis with Positive Groups')
            ax.grid(True, alpha=0.3)
            if positive_mask is not None or (hasattr(self, 'nmf_results') and 'cluster_labels' in self.nmf_results):
                ax.legend(fontsize=8)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting NMF results:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('NMF Analysis - Error')
            ax.axis('off')
            
    def _plot_top_spectral_matches(self, ax):
        """Plot top 5 spectral matches from classification."""
        try:
            # Check if we have spectral data to analyze
            if not hasattr(self, 'map_data') or self.map_data is None:
                ax.text(0.5, 0.5, 'No map data available.\nLoad map data first.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('Top Spectral Matches - No Data')
                ax.axis('off')
                return
            
            # Get positive groups if available
            positive_mask = self._get_positive_groups_mask()
            
            if positive_mask is None or not np.any(positive_mask):
                ax.text(0.5, 0.5, 'No positive groups identified.\nRun ML classification first.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('Top Spectral Matches - No Positive Groups')
                ax.axis('off')
                return
            
            # Get spectra from positive groups
            positive_indices = np.where(positive_mask)[0]
            
            # Limit to top 5 positive spectra (by intensity or classification confidence if available)
            if len(positive_indices) > 5:
                # Try to get classification probabilities for ranking
                if hasattr(self, 'classification_results') and 'probabilities' in self.classification_results:
                    probs = self.classification_results['probabilities']
                    if probs is not None and len(probs) == len(positive_mask):
                        # Get probabilities for positive class
                        positive_probs = probs[positive_mask]
                        if len(positive_probs.shape) > 1:
                            positive_probs = positive_probs[:, 1]  # Assume positive class is index 1
                        
                        # Sort by probability and take top 5
                        sorted_indices = np.argsort(positive_probs)[-5:]
                        top_positive_indices = positive_indices[sorted_indices]
                    else:
                        top_positive_indices = positive_indices[:5]
                else:
                    top_positive_indices = positive_indices[:5]
            else:
                top_positive_indices = positive_indices
            
            # Plot the spectra
            colors = plt.cm.Reds(np.linspace(0.4, 1.0, len(top_positive_indices)))
            
            # Get all spectra as a list for indexing
            all_spectra = list(self.map_data.spectra.values())
            
            plotted_count = 0
            for i, idx in enumerate(top_positive_indices):
                if idx >= len(all_spectra):
                    continue  # Skip invalid indices
                    
                spectrum = all_spectra[idx]
                wavenumbers = spectrum.wavenumbers
                intensities = (spectrum.processed_intensities 
                             if self.use_processed and spectrum.processed_intensities is not None
                             else spectrum.intensities)
                
                if wavenumbers is None or intensities is None:
                    continue  # Skip spectra without data
                
                # Normalize for display
                intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min()) + plotted_count * 0.2
                
                ax.plot(wavenumbers, intensities_norm, color=colors[plotted_count], 
                       linewidth=1.5, alpha=0.8, label=f'Positive #{plotted_count+1}')
                plotted_count += 1
            
            if plotted_count == 0:
                ax.text(0.5, 0.5, 'No valid positive spectra found for plotting.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('Top Spectral Matches - No Valid Data')
                ax.axis('off')
                return
            
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Normalized Intensity (offset)')
            ax.set_title('Top Positive Group Spectra')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting spectral matches:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Top Spectral Matches - Error')
            ax.axis('off')
            
    def _plot_component_statistics(self, ax):
        """Plot component analysis and statistics."""
        try:
            # Collect statistics from all analyses
            stats_text = []
            
            # PCA Statistics
            if hasattr(self, 'pca_results') and self.pca_results is not None:
                pca_variance = self.pca_results['explained_variance_ratio'][:3]  # Top 3 components
                stats_text.append("=== PCA ANALYSIS ===")
                for i, var in enumerate(pca_variance):
                    stats_text.append(f"PC{i+1}: {var:.1%} variance")
                total_var = np.sum(pca_variance)
                stats_text.append(f"Total (3 comp): {total_var:.1%}")
                stats_text.append("")
            
            # NMF Statistics
            if hasattr(self, 'nmf_results') and self.nmf_results is not None:
                n_components = self.nmf_results.get('n_components', 0)
                reconstruction_error = self.nmf_results.get('reconstruction_error', 0)
                stats_text.append("=== NMF ANALYSIS ===")
                stats_text.append(f"Components: {n_components}")
                stats_text.append(f"Reconstruction error: {reconstruction_error:.3f}")
                stats_text.append("")
            
            # ML Classification Statistics
            positive_mask = self._get_positive_groups_mask()
            if positive_mask is not None:
                n_positive = np.sum(positive_mask)
                n_total = len(positive_mask)
                positive_rate = n_positive / n_total if n_total > 0 else 0
                
                stats_text.append("=== ML CLASSIFICATION ===")
                stats_text.append(f"Total spectra: {n_total}")
                stats_text.append(f"Positive groups: {n_positive}")
                stats_text.append(f"Positive rate: {positive_rate:.1%}")
                
                if hasattr(self, 'classification_results'):
                    if 'type' in self.classification_results:
                        stats_text.append(f"Type: {self.classification_results['type']}")
                    if 'class_names' in self.classification_results and self.classification_results['class_names']:
                        stats_text.append(f"Classes: {', '.join(self.classification_results['class_names'])}")
                
                stats_text.append("")
            
            # Template Fitting Statistics
            if hasattr(self, 'template_manager') and self.template_manager.templates:
                stats_text.append("=== TEMPLATE ANALYSIS ===")
                stats_text.append(f"Templates loaded: {len(self.template_manager.templates)}")
                if hasattr(self, 'template_fit_results'):
                    avg_r2 = np.mean([r['r_squared'] for r in self.template_fit_results.values()])
                    stats_text.append(f"Avg. R²: {avg_r2:.3f}")
                stats_text.append("")
            
            # Display statistics
            if stats_text:
                ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No analysis results available.\nComplete analysis steps first.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
            
            ax.set_title('Analysis Statistics')
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error displaying statistics:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Analysis Statistics - Error')
            ax.axis('off')
            
    def _get_positive_groups_mask(self):
        """Get boolean mask for positive groups from ML classification."""
        try:
            if not hasattr(self, 'classification_results') or not self.classification_results:
                return None
            
            predictions = self.classification_results.get('predictions')
            if predictions is None:
                return None
            
            # For supervised classification, positive class is typically 1
            if self.classification_results.get('type') == 'supervised':
                return predictions == 1
            
            # For unsupervised clustering, we need to identify which cluster(s) are "interesting"
            # This is heuristic - typically the smallest clusters are more likely to be positive
            elif self.classification_results.get('type') == 'unsupervised':
                unique_labels, counts = np.unique(predictions, return_counts=True)
                
                # Consider clusters with less than 20% of total data as potentially positive
                total_count = len(predictions)
                small_clusters = unique_labels[counts < 0.2 * total_count]
                
                if len(small_clusters) > 0:
                    mask = np.isin(predictions, small_clusters)
                    return mask
                else:
                    # If no small clusters, consider the two smallest clusters
                    sorted_indices = np.argsort(counts)
                    smallest_clusters = unique_labels[sorted_indices[:min(2, len(unique_labels))]]
                    mask = np.isin(predictions, smallest_clusters)
                    return mask
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting positive groups mask: {e}")
            return None
            
    def _update_statistics_text(self):
        """Update the statistics text area with comprehensive analysis summary."""
        try:
            stats_lines = []
            
            # Header
            stats_lines.append("=" * 80)
            stats_lines.append("COMPREHENSIVE RAMAN MAP ANALYSIS SUMMARY")
            stats_lines.append("=" * 80)
            
            # Data overview
            if hasattr(self, 'map_data') and self.map_data is not None:
                n_spectra = len(self.map_data.spectra)
                stats_lines.append(f"Total spectra analyzed: {n_spectra}")
                if hasattr(self.map_data, 'x_positions'):
                    x_range = f"{self.map_data.x_positions.min():.1f} - {self.map_data.x_positions.max():.1f}"
                    y_range = f"{self.map_data.y_positions.min():.1f} - {self.map_data.y_positions.max():.1f}"
                    stats_lines.append(f"Map dimensions: X: {x_range}, Y: {y_range}")
            
            stats_lines.append("")
            
            # Positive groups analysis
            positive_mask = self._get_positive_groups_mask()
            if positive_mask is not None:
                n_positive = np.sum(positive_mask)
                n_total = len(positive_mask)
                positive_rate = n_positive / n_total * 100 if n_total > 0 else 0
                
                stats_lines.append(f"POSITIVE GROUPS IDENTIFICATION:")
                stats_lines.append(f"  • Total positive groups found: {n_positive} ({positive_rate:.1f}%)")
                stats_lines.append(f"  • Background/negative spectra: {n_total - n_positive} ({100-positive_rate:.1f}%)")
                
                if positive_rate < 5:
                    stats_lines.append(f"  • Analysis: Very few positive groups detected - likely outliers as expected for plastic analysis")
                elif positive_rate < 20:
                    stats_lines.append(f"  • Analysis: Small number of positive groups - good separation achieved")
                else:
                    stats_lines.append(f"  • Analysis: Significant positive groups detected - review classification parameters")
            
            stats_lines.append("")
            
            # Component analysis summary
            if hasattr(self, 'pca_results') and self.pca_results is not None:
                pca_var = self.pca_results['explained_variance_ratio']
                stats_lines.append(f"PCA COMPONENT ANALYSIS:")
                stats_lines.append(f"  • PC1 explains {pca_var[0]:.1%} of variance")
                stats_lines.append(f"  • PC2 explains {pca_var[1]:.1%} of variance")
                if len(pca_var) > 2:
                    stats_lines.append(f"  • PC3 explains {pca_var[2]:.1%} of variance")
                    stats_lines.append(f"  • First 3 components: {np.sum(pca_var[:3]):.1%} total variance")
            
            if hasattr(self, 'nmf_results') and self.nmf_results is not None:
                stats_lines.append(f"NMF COMPONENT ANALYSIS:")
                stats_lines.append(f"  • {self.nmf_results.get('n_components', 0)} components extracted")
                stats_lines.append(f"  • Reconstruction error: {self.nmf_results.get('reconstruction_error', 0):.4f}")
            
            # Set the text
            self.results_statistics.setPlainText('\n'.join(stats_lines))
            
        except Exception as e:
            self.results_statistics.setPlainText(f"Error generating statistics summary: {str(e)}")
            
    def export_comprehensive_results(self):
        """Export comprehensive results including plots and statistics."""
        try:
            from PySide6.QtWidgets import QFileDialog
            import os
            from datetime import datetime
            
            # Get save directory
            save_dir = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Comprehensive Results", 
                os.path.expanduser("~/Desktop")
            )
            
            if not save_dir:
                return
            
            # Create timestamped folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_folder = os.path.join(save_dir, f"RamanLab_Comprehensive_Results_{timestamp}")
            os.makedirs(results_folder, exist_ok=True)
            
            # Save the main figure
            figure_path = os.path.join(results_folder, "comprehensive_analysis.png")
            self.results_plot_widget.figure.savefig(figure_path, dpi=300, bbox_inches='tight')
            
            # Save statistics as text file
            stats_path = os.path.join(results_folder, "analysis_statistics.txt")
            with open(stats_path, 'w') as f:
                f.write(self.results_statistics.toPlainText())
            
            # Save individual component data if available
            data_folder = os.path.join(results_folder, "data_exports")
            os.makedirs(data_folder, exist_ok=True)
            
            # Export PCA data
            if hasattr(self, 'pca_results') and self.pca_results is not None:
                pca_path = os.path.join(data_folder, "pca_results.csv")
                import pandas as pd
                pca_data = self.pca_results.get('components', self.pca_results.get('transformed_data'))
                if pca_data is not None:
                    pca_df = pd.DataFrame(pca_data, 
                                        columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
                    pca_df.to_csv(pca_path, index=False)
            
            # Export NMF data
            if hasattr(self, 'nmf_results') and self.nmf_results is not None:
                nmf_path = os.path.join(data_folder, "nmf_results.csv")
                import pandas as pd
                nmf_data = self.nmf_results.get('components', self.nmf_results.get('transformed_data'))
                if nmf_data is not None:
                    nmf_df = pd.DataFrame(nmf_data, 
                                        columns=[f'NMF{i+1}' for i in range(nmf_data.shape[1])])
                    nmf_df.to_csv(nmf_path, index=False)
            
            # Export positive groups data
            positive_mask = self._get_positive_groups_mask()
            if positive_mask is not None:
                groups_path = os.path.join(data_folder, "positive_groups.csv")
                import pandas as pd
                groups_df = pd.DataFrame({
                    'spectrum_index': range(len(positive_mask)),
                    'is_positive': positive_mask
                })
                if hasattr(self.map_data, 'x_positions'):
                    groups_df['x_position'] = self.map_data.x_positions
                    groups_df['y_position'] = self.map_data.y_positions
                groups_df.to_csv(groups_path, index=False)
            
            QMessageBox.information(
                self, "Export Complete", 
                f"Comprehensive results exported to:\n{results_folder}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting results:\n{str(e)}")
        
    def export_results(self):
        """Export analysis results."""
        if not hasattr(self, 'nmf_results') and not hasattr(self, 'pca_results'):
            QMessageBox.warning(self, "No Results", "No analysis results to export.")
            return
            
        try:
            # Get export directory
            directory = QFileDialog.getExistingDirectory(
                self, "Select Export Directory")
            
            if not directory:
                return
                
            import os
            import numpy as np
            
            # Export NMF results if available
            if hasattr(self, 'nmf_results'):
                nmf_dir = os.path.join(directory, "nmf_results")
                os.makedirs(nmf_dir, exist_ok=True)
                
                # Save component maps
                for i in range(self.nmf_results['n_components']):
                    component_map = self.create_nmf_component_map(i)
                    if component_map is not None:
                        np.savetxt(os.path.join(nmf_dir, f"component_{i+1}_map.csv"), 
                                  component_map, delimiter=',')
                
                # Save component spectra (H matrix)
                np.savetxt(os.path.join(nmf_dir, "component_spectra.csv"), 
                          self.nmf_results['feature_components'], delimiter=',')
                
                # Save component contributions (W matrix)
                np.savetxt(os.path.join(nmf_dir, "component_contributions.csv"), 
                          self.nmf_results['components'], delimiter=',')
                
                # Save metadata
                with open(os.path.join(nmf_dir, "nmf_info.txt"), 'w') as f:
                    f.write(f"NMF Analysis Results\n")
                    f.write(f"==================\n\n")
                    f.write(f"Number of components: {self.nmf_results['n_components']}\n")
                    f.write(f"Number of samples: {self.nmf_results['n_samples']}\n")
                    f.write(f"Number of features: {self.nmf_results['n_features']}\n")
                    f.write(f"Reconstruction error: {self.nmf_results['reconstruction_error']:.6f}\n")
                    f.write(f"Iterations: {self.nmf_results.get('n_iterations', 'N/A')}\n")
                
                logger.info(f"NMF results exported to {nmf_dir}")
            
            # Export current plots as images
            current_tab = self.tab_widget.currentIndex()
            if current_tab == 3 and hasattr(self, 'nmf_results'):  # NMF tab
                plot_path = os.path.join(directory, "nmf_analysis_plot.png")
                self.nmf_plot_widget.figure.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"NMF plot saved to {plot_path}")
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Results exported successfully to:\n{directory}")
            self.statusBar().showMessage(f"Results exported to {directory}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
            logger.error(f"Export error: {e}")
    
    def save_nmf_results(self):
        """Save NMF results to file."""
        if not hasattr(self, 'nmf_results'):
            QMessageBox.warning(self, "No Results", "No NMF results to save.")
            return
            
        try:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save NMF Results", 
                "nmf_results.pkl", 
                "Pickle Files (*.pkl)")
            
            if filepath:
                if self.nmf_analyzer.save_results(filepath):
                    QMessageBox.information(self, "Save Complete", 
                                          "NMF results saved successfully.")
                    self.statusBar().showMessage(f"NMF results saved to {filepath}")
                else:
                    QMessageBox.warning(self, "Save Failed", 
                                      "Failed to save NMF results.")
                    
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving NMF results:\n{str(e)}")
            logger.error(f"Save NMF results error: {e}")
    
    def load_nmf_results(self):
        """Load NMF results from file."""
        try:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load NMF Results", 
                "", 
                "Pickle Files (*.pkl)")
            
            if filepath:
                if self.nmf_analyzer.load_results(filepath):
                    # Reconstruct results dict for compatibility
                    self.nmf_results = {
                        'success': True,
                        'components': self.nmf_analyzer.components,
                        'feature_components': self.nmf_analyzer.feature_components,
                        'reconstruction_error': self.nmf_analyzer.reconstruction_error,
                        'n_components': self.nmf_analyzer.components.shape[1] if self.nmf_analyzer.components is not None else 0,
                        'n_samples': self.nmf_analyzer.components.shape[0] if self.nmf_analyzer.components is not None else 0,
                        'n_features': self.nmf_analyzer.feature_components.shape[1] if self.nmf_analyzer.feature_components is not None else 0
                    }
                    
                    # Switch to NMF tab and plot results
                    self.tab_widget.setCurrentIndex(3)
                    self.plot_nmf_results(self.nmf_results)
                    
                    # Update map features
                    self.update_map_features_with_nmf()
                    
                    QMessageBox.information(self, "Load Complete", 
                                          "NMF results loaded successfully.")
                    self.statusBar().showMessage(f"NMF results loaded from {filepath}")
                else:
                    QMessageBox.warning(self, "Load Failed", 
                                      "Failed to load NMF results.")
                    
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading NMF results:\n{str(e)}")
            logger.error(f"Load NMF results error: {e}")
        
    # Cosmic Ray Detection Methods
    def on_cosmic_ray_enabled_changed(self, enabled: bool):
        """Handle cosmic ray detection enable/disable."""
        self.cosmic_ray_config.enabled = enabled
        self.cosmic_ray_manager = SimpleCosmicRayManager(self.cosmic_ray_config)
        self.statusBar().showMessage(f"Cosmic ray detection {'enabled' if enabled else 'disabled'}")
        
    def on_cosmic_ray_params_changed(self):
        """Handle cosmic ray parameter changes."""
        try:
            # Get the current control panel
            control_panel = None
            for name, section in self.controls_panel.sections.items():
                if name == "map_controls" and hasattr(section['widget'], 'threshold_spin'):
                    control_panel = section['widget']
                    break
            
            if control_panel:
                # Update basic detection parameters
                self.cosmic_ray_config.absolute_threshold = control_panel.threshold_spin.value()
                self.cosmic_ray_config.neighbor_ratio = control_panel.neighbor_ratio_spin.value()
                
                # Update shape analysis parameters
                if hasattr(control_panel, 'enable_shape_cb'):
                    self.cosmic_ray_config.enable_shape_analysis = control_panel.enable_shape_cb.isChecked()
                    self.cosmic_ray_config.max_cosmic_fwhm = control_panel.max_fwhm_spin.value()
                    self.cosmic_ray_config.min_sharpness_ratio = control_panel.min_sharpness_spin.value()
                    self.cosmic_ray_config.max_asymmetry_factor = control_panel.max_asymmetry_spin.value()
                    self.cosmic_ray_config.gradient_threshold = control_panel.gradient_thresh_spin.value()
                
                # Update removal range parameters
                if hasattr(control_panel, 'removal_range_spin'):
                    self.cosmic_ray_config.removal_range = control_panel.removal_range_spin.value()
                    self.cosmic_ray_config.adaptive_range = control_panel.adaptive_range_cb.isChecked()
                    self.cosmic_ray_config.max_removal_range = control_panel.max_removal_range_spin.value()
                
                # Create new manager with updated config
                self.cosmic_ray_manager = SimpleCosmicRayManager(self.cosmic_ray_config)
                self.statusBar().showMessage("Cosmic ray parameters updated")
            
        except Exception as e:
            print(f"ERROR in on_cosmic_ray_params_changed: {e}")
            import traceback
            traceback.print_exc()
        
        # Update CRE test if it's currently open
        if hasattr(self, 'cre_test_dialog') and self.cre_test_dialog is not None and self.cre_test_dialog.isVisible():
            # Get the current test spectrum from the dialog, fallback to selected spectrum
            test_spectrum = getattr(self, 'current_test_spectrum', None)
            if test_spectrum is None and hasattr(self, 'current_selected_spectrum'):
                test_spectrum = self.current_selected_spectrum
            
            if test_spectrum:
                self.update_cre_test(test_spectrum)
            
    def reprocess_cosmic_rays(self):
        """Reprocess all spectra with current cosmic ray settings."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
            
        try:
            self.progress_status.show_progress("Reprocessing cosmic rays...")
            
            processed_count = 0
            cosmic_rays_found = 0
            
            for spectrum in self.map_data.spectra.values():
                if spectrum.wavenumbers is not None and spectrum.intensities is not None:
                    # Apply cosmic ray detection
                    had_cosmic_rays, cleaned_intensities, detection_info = \
                        self.cosmic_ray_manager.detect_and_remove_cosmic_rays(
                            spectrum.wavenumbers, spectrum.intensities)
                    
                    # Update processed intensities
                    spectrum.processed_intensities = cleaned_intensities
                    processed_count += 1
                    
                    # Get the correct cosmic ray count from detection_info
                    cosmic_rays_in_spectrum = detection_info.get('cosmic_ray_count', 0)
                    if cosmic_rays_in_spectrum > 0:
                        cosmic_rays_found += cosmic_rays_in_spectrum
            
            self.progress_status.hide_progress()
            
            # Update the map display
            self.update_map()
            
            # Show results
            QMessageBox.information(
                self, "Cosmic Ray Processing Complete",
                f"Processed {processed_count} spectra.\n"
                f"Found and removed {cosmic_rays_found} cosmic rays."
            )
            
            self.statusBar().showMessage(f"Cosmic ray processing complete: {cosmic_rays_found} cosmic rays removed")
            
        except Exception as e:
            self.progress_status.hide_progress()
            QMessageBox.critical(self, "Error", f"Error processing cosmic rays:\n{str(e)}")
            
    def create_cosmic_ray_map(self):
        """Create a map showing cosmic ray detection intensity."""
        import numpy as np
        
        if self.map_data is None:
            return None
            
        positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
        
        if not positions:
            return None
            
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        map_array = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
        
        for spectrum in self.map_data.spectra.values():
            if spectrum.wavenumbers is not None and spectrum.intensities is not None:
                # Run cosmic ray detection to get count
                had_cosmic_rays, _, detection_info = \
                    self.cosmic_ray_manager.detect_and_remove_cosmic_rays(
                        spectrum.wavenumbers, spectrum.intensities)
                
                cosmic_ray_count = detection_info.get('cosmic_ray_count', 0)
                map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = cosmic_ray_count
                
        return map_array
        
    def show_cosmic_ray_statistics(self):
        """Show cosmic ray detection statistics."""
        stats = self.cosmic_ray_manager.get_statistics()
        
        stats_text = f"""Cosmic Ray Detection Statistics:
        
Total Spectra Processed: {stats['total_spectra']}
Spectra with Cosmic Rays: {stats['spectra_with_cosmic_rays']}
Total Cosmic Rays Removed: {stats['total_cosmic_rays_removed']}
Shape Analysis Rejections: {stats['shape_analysis_rejections']}
False Positive Prevention: {stats['false_positive_prevention']}

Detection Rate: {stats['spectra_with_cosmic_rays'] / max(1, stats['total_spectra']) * 100:.1f}%
Average CRs per Affected Spectrum: {stats['total_cosmic_rays_removed'] / max(1, stats['spectra_with_cosmic_rays']):.1f}
"""
        
        QMessageBox.information(self, "Cosmic Ray Statistics", stats_text)
        
    def show_cosmic_ray_diagnostics(self):
        """Show detailed cosmic ray diagnostics."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
            
        # Get a representative spectrum for diagnostics
        if not self.map_data.spectra:
            QMessageBox.warning(self, "No Spectra", "No spectra available for diagnostics.")
            return
            
        # Use the first spectrum for demonstration
        spectrum = next(iter(self.map_data.spectra.values()))
        
        if spectrum.wavenumbers is None or spectrum.intensities is None:
            QMessageBox.warning(self, "Invalid Spectrum", "Selected spectrum has no data.")
            return
            
        try:
            # Run diagnostics
            diagnostics = self.cosmic_ray_manager.diagnose_peak_shape(
                spectrum.wavenumbers, spectrum.intensities, display_details=True)
            
            # Create a simple dialog to show results
            dialog = QDialog(self)
            dialog.setWindowTitle("Cosmic Ray Diagnostics")
            dialog.setMinimumSize(600, 400)
            
            layout = QVBoxLayout(dialog)
            
            # Create text widget to show diagnostics
            text_widget = QTextEdit()
            text_widget.setReadOnly(True)
            
            # Format diagnostics text
            diag_text = f"""Cosmic Ray Shape Analysis Diagnostics:

Spectrum: Position ({spectrum.x_pos}, {spectrum.y_pos})
Total Data Points: {len(spectrum.intensities)}

Diagnostics Results:
{diagnostics.get('summary', 'No summary available')}

Parameters Used:
- Threshold: {self.cosmic_ray_config.absolute_threshold}
- Neighbor Ratio: {self.cosmic_ray_config.neighbor_ratio}
- Max FWHM: {self.cosmic_ray_config.max_cosmic_fwhm}
- Min Sharpness Ratio: {self.cosmic_ray_config.min_sharpness_ratio}
"""
            
            text_widget.setPlainText(diag_text)
            layout.addWidget(text_widget)
            
            # Add close button
            from .base_widgets import StandardButton
            close_btn = StandardButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.exec()
            
        except Exception as e:
            self.progress_status.hide_progress()
            QMessageBox.critical(self, "Diagnostics Error", f"Error running diagnostics:\n{str(e)}")
             
    def apply_cre_to_all_files(self):
        """Apply cosmic ray elimination to all files in the current map directory."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
            
        # Confirm the operation
        reply = QMessageBox.question(
            self, "Apply CRE to All Files",
            f"This will apply cosmic ray elimination to all {len(self.map_data.spectra)} spectra "
            f"in the current map with the current settings.\n\n"
            f"Current Settings:\n"
            f"• Threshold: {self.cosmic_ray_config.absolute_threshold}\n"
            f"• Neighbor Ratio: {self.cosmic_ray_config.neighbor_ratio}\n"
            f"• Shape Analysis: {'Enabled' if self.cosmic_ray_config.enable_shape_analysis else 'Disabled'}\n"
            f"• Removal Range: {self.cosmic_ray_config.removal_range}\n\n"
            f"Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        try:
            self.progress_status.show_progress("Applying CRE to all files...")
            
            total_spectra = len(self.map_data.spectra)
            processed_count = 0
            total_cosmic_rays_removed = 0
            spectra_with_cosmic_rays = 0
            
            # Reset statistics
            self.cosmic_ray_manager.reset_statistics()
            
            for i, spectrum in enumerate(self.map_data.spectra.values()):
                if spectrum.wavenumbers is not None and spectrum.intensities is not None:
                    # Apply cosmic ray detection with full shape analysis
                    had_cosmic_rays, cleaned_intensities, detection_info = \
                        self.cosmic_ray_manager.detect_and_remove_cosmic_rays(
                            spectrum.wavenumbers, spectrum.intensities, 
                            spectrum_id=f"({spectrum.x_pos},{spectrum.y_pos})")
                    
                    # Update processed intensities
                    spectrum.processed_intensities = cleaned_intensities
                    processed_count += 1
                    
                    # Get the correct cosmic ray count from detection_info
                    cosmic_rays_in_spectrum = detection_info.get('cosmic_ray_count', 0)
                    
                    if cosmic_rays_in_spectrum > 0:
                        spectra_with_cosmic_rays += 1
                        total_cosmic_rays_removed += cosmic_rays_in_spectrum
                
                # Update progress
                progress = int((i + 1) / total_spectra * 100)
                self.progress_status.update_progress(progress, f"Processing spectrum {i+1}/{total_spectra}")
            
            self.progress_status.hide_progress()
            
            # Update the map display
            self.update_map()
            
            # Show comprehensive results
            stats = self.cosmic_ray_manager.get_statistics()
            results_text = f"""Cosmic Ray Elimination Complete!

Processing Summary:
• Total Spectra Processed: {processed_count}
• Spectra with Cosmic Rays: {spectra_with_cosmic_rays}
• Total Cosmic Rays Removed: {total_cosmic_rays_removed}
• Detection Rate: {spectra_with_cosmic_rays / max(1, processed_count) * 100:.1f}%

Advanced Analysis:
• Shape Analysis Rejections: {stats['shape_analysis_rejections']}
• False Positive Prevention: {stats['false_positive_prevention']}
• Average CRs per Affected Spectrum: {total_cosmic_rays_removed / max(1, spectra_with_cosmic_rays):.2f}

All spectra have been processed and cleaned data is now available for analysis."""
            
            QMessageBox.information(self, "CRE Processing Complete", results_text)
            
            self.statusBar().showMessage(f"CRE applied to all files: {total_cosmic_rays_removed} cosmic rays removed")
            
        except Exception as e:
            self.progress_status.hide_progress()
            QMessageBox.critical(self, "Error", f"Error applying CRE to all files:\n{str(e)}")
            
    def test_shape_analysis(self):
        """Test shape analysis on current spectrum with real-time visual feedback."""
        from .base_widgets import StandardButton
        import matplotlib.pyplot as plt
        from matplotlib.backends.qt_compat import QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
        
        # If dialog is already open, just bring it to front
        if hasattr(self, 'cre_test_dialog') and self.cre_test_dialog is not None:
            self.cre_test_dialog.raise_()
            self.cre_test_dialog.activateWindow()
            return
            
        # Get the currently selected spectrum (if any) or use highest intensity spectrum
        test_spectrum = None
        
        # First try to get the spectrum from the current selected spectrum
        if hasattr(self, 'current_selected_spectrum') and self.current_selected_spectrum is not None:
            test_spectrum = self.current_selected_spectrum
        # Fallback to using marker position
        elif hasattr(self, 'current_marker_position'):
            x_marker, y_marker = self.current_marker_position
            test_spectrum = self.find_closest_spectrum(x_marker, y_marker)
        
        # Last fallback to highest intensity spectrum
        if test_spectrum is None:
            if not self.map_data.spectra:
                QMessageBox.warning(self, "No Spectra", "No spectra available for testing.")
                return
                
            max_intensity = 0
            for spectrum in self.map_data.spectra.values():
                if spectrum.wavenumbers is not None and spectrum.intensities is not None:
                    total_intensity = sum(spectrum.intensities)
                    if total_intensity > max_intensity:
                        max_intensity = total_intensity
                        test_spectrum = spectrum
        
        if test_spectrum is None:
            QMessageBox.warning(self, "No Valid Spectrum", "No valid spectrum found for testing.")
            return
        
        # Create the CRE test dialog
        self.cre_test_dialog = QDialog(self)
        dialog_title = f"Real-Time CRE Test - Spectrum at ({test_spectrum.x_pos}, {test_spectrum.y_pos})"
        self.cre_test_dialog.setWindowTitle(dialog_title)
        self.cre_test_dialog.setMinimumSize(1000, 700)
        
        layout = QVBoxLayout(self.cre_test_dialog)
        
        # Create matplotlib figure for before/after comparison with configured styling
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from ui.matplotlib_config import apply_theme, CompactNavigationToolbar
        apply_theme('compact')  # Apply compact theme for consistent styling
        
        self.cre_test_figure = Figure(figsize=(12, 8))
        self.cre_test_canvas = FigureCanvas(self.cre_test_figure)
        
        # Add matplotlib navigation toolbar using configured style
        self.cre_test_toolbar = CompactNavigationToolbar(self.cre_test_canvas, self.cre_test_dialog)
        
        layout.addWidget(self.cre_test_toolbar)
        layout.addWidget(self.cre_test_canvas)
        
        # Results text area
        self.cre_results_text = QTextEdit()
        self.cre_results_text.setMaximumHeight(150)
        self.cre_results_text.setReadOnly(True)
        layout.addWidget(self.cre_results_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        refresh_btn = StandardButton("🔄 Refresh Test")
        refresh_btn.clicked.connect(lambda: self.update_cre_test(test_spectrum))
        refresh_btn.setToolTip("Manually refresh the test (updates automatically when parameters change)")
        button_layout.addWidget(refresh_btn)
        
        # Add a force update button that reads current params first
        force_update_btn = StandardButton("🔃 Force Update")
        force_update_btn.clicked.connect(lambda: (self.on_cosmic_ray_params_changed(), self.update_cre_test(test_spectrum)))
        force_update_btn.setToolTip("Force read parameters from left panel and update test")
        button_layout.addWidget(force_update_btn)
        
        apply_all_btn = StandardButton("✅ Apply to All Files")
        apply_all_btn.clicked.connect(lambda: (self.cre_test_dialog.accept(), self.apply_cre_to_all_files()))
        button_layout.addWidget(apply_all_btn)
        
        close_btn = StandardButton("❌ Close")
        close_btn.clicked.connect(self.cre_test_dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Store current test spectrum for parameter updates
        self.current_test_spectrum = test_spectrum
        
        # Force read current parameters to ensure they're up to date
        self.on_cosmic_ray_params_changed()
        
        # Initial test run
        self.update_cre_test(test_spectrum)
        
        # Show dialog (non-modal)
        self.cre_test_dialog.show()
        
        # Connect close event to clean up
        self.cre_test_dialog.finished.connect(lambda: self._cleanup_cre_test())
    
    def _cleanup_cre_test(self):
        """Clean up when CRE test dialog is closed."""
        self.current_test_spectrum = None
        if hasattr(self, 'cre_test_dialog'):
            self.cre_test_dialog = None
    
    def update_cre_test(self, spectrum):
        """Update the CRE test visualization with current parameters."""
        try:
            # Get original spectrum data
            wavenumbers = spectrum.wavenumbers
            original_intensities = spectrum.intensities
            
            if wavenumbers is None or original_intensities is None:
                return
            
            # Apply current CRE settings
            had_cosmic_rays, cleaned_intensities, detection_info = \
                self.cosmic_ray_manager.detect_and_remove_cosmic_rays(
                    wavenumbers, original_intensities,
                    spectrum_id=f"Test_({spectrum.x_pos},{spectrum.y_pos})")
            
            # Get detected cosmic ray indices
            cosmic_ray_indices = detection_info.get('cosmic_ray_indices', [])
            
            # Clear and set up the plot
            self.cre_test_figure.clear()
            
            # Create subplots: original, processed, and difference
            ax1 = self.cre_test_figure.add_subplot(3, 1, 1)
            ax2 = self.cre_test_figure.add_subplot(3, 1, 2)
            ax3 = self.cre_test_figure.add_subplot(3, 1, 3)
            
            # Plot 1: Original spectrum with cosmic rays highlighted
            ax1.plot(wavenumbers, original_intensities, 'b-', linewidth=1, label='Original Spectrum')
            
            # Highlight detected cosmic rays
            if cosmic_ray_indices:
                cosmic_wavenumbers = [wavenumbers[i] for i in cosmic_ray_indices if i < len(wavenumbers)]
                cosmic_intensities = [original_intensities[i] for i in cosmic_ray_indices if i < len(original_intensities)]
                ax1.scatter(cosmic_wavenumbers, cosmic_intensities, color='red', s=50, zorder=5, 
                           label=f'{len(cosmic_ray_indices)} Cosmic Rays Detected')
            
            ax1.set_title(f'Original Spectrum - Position ({spectrum.x_pos}, {spectrum.y_pos})')
            ax1.set_ylabel('Intensity')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Processed spectrum
            ax2.plot(wavenumbers, cleaned_intensities, 'g-', linewidth=1, label='After CRE')
            ax2.plot(wavenumbers, original_intensities, 'b-', linewidth=0.5, alpha=0.3, label='Original (faded)')
            
            # Highlight replacement regions
            if cosmic_ray_indices:
                for idx in cosmic_ray_indices:
                    if idx < len(wavenumbers):
                        # Show replacement range
                        range_start = max(0, idx - self.cosmic_ray_config.removal_range)
                        range_end = min(len(wavenumbers), idx + self.cosmic_ray_config.removal_range + 1)
                        ax2.axvspan(wavenumbers[range_start], wavenumbers[range_end-1], 
                                   alpha=0.2, color='green', label='Replacement Region' if idx == cosmic_ray_indices[0] else "")
            
            ax2.set_title('After Cosmic Ray Elimination')
            ax2.set_ylabel('Intensity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Difference (what was removed)
            difference = original_intensities - cleaned_intensities
            ax3.plot(wavenumbers, difference, 'r-', linewidth=1, label='Removed (Original - Cleaned)')
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_title('Difference: What Was Removed')
            ax3.set_xlabel('Wavenumber (cm⁻¹)')
            ax3.set_ylabel('Intensity Difference')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Adjust layout and refresh
            self.cre_test_figure.tight_layout()
            self.cre_test_canvas.draw()
            
            # Update results text
            results_text = f"""CRE Test Results (Position {spectrum.x_pos}, {spectrum.y_pos}) - Updates automatically with parameter changes:

🔍 Detection Results:
• Cosmic Rays Found: {'Yes' if had_cosmic_rays else 'No'}
• Cosmic Rays Removed: {len(cosmic_ray_indices)}
• Cosmic Ray Indices: {cosmic_ray_indices[:10]}{'...' if len(cosmic_ray_indices) > 10 else ''}

⚙️ Current Parameters (from left panel):
• Threshold: {self.cosmic_ray_config.absolute_threshold}
• Neighbor Ratio: {self.cosmic_ray_config.neighbor_ratio}
• Shape Analysis: {'Enabled' if self.cosmic_ray_config.enable_shape_analysis else 'Disabled'}
• Max FWHM: {self.cosmic_ray_config.max_cosmic_fwhm}
• Removal Range: {self.cosmic_ray_config.removal_range}
• Adaptive Range: {'Yes' if self.cosmic_ray_config.adaptive_range else 'No'}

💡 Visual Guide:
• RED points: Detected cosmic rays
• GREEN areas: Replacement regions  
• BLUE line: Original spectrum
• GREEN line: After CRE processing
• RED line (bottom): What was removed

⚠️ Safety Check: Look for legitimate Raman peaks in the "What Was Removed" plot - they should NOT appear there!

📝 Instructions: Adjust parameters in the left panel to see real-time updates here!
"""
            
            self.cre_results_text.setPlainText(results_text)
            
        except Exception as e:
            import traceback
            error_text = f"Error updating CRE test: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.cre_results_text.setPlainText(error_text)
            
    # PKL File Management Methods
    def save_map_to_pkl(self):
        """Save current map data to a PKL file."""
        if self.map_data is None:
            QMessageBox.warning(self, "No Data", "Load map data first.")
            return
            
        # Get save file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Map Data to PKL", "", 
            "Pickle files (*.pkl);;All files (*)")
        
        if not file_path:
            return
            
        # Ensure .pkl extension
        if not file_path.endswith('.pkl'):
            file_path += '.pkl'
            
        try:
            self.progress_status.show_progress("Saving map data to PKL...")
            
            # Prepare map data for saving
            save_data = {
                'map_data': self.map_data,
                'cosmic_ray_config': self.cosmic_ray_config,
                'metadata': {
                    'creation_time': datetime.now().isoformat(),
                    'total_spectra': len(self.map_data.spectra),
                    'has_processed_data': any(
                        s.processed_intensities is not None 
                        for s in self.map_data.spectra.values()
                    ),
                    'cosmic_ray_stats': self.cosmic_ray_manager.get_statistics(),
                    'version': '2.0.0'
                }
            }
            
            # Save to pickle file
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.progress_status.hide_progress()
            
            # Show success message
            spectra_count = len(self.map_data.spectra)
            processed_count = sum(1 for s in self.map_data.spectra.values() 
                                if s.processed_intensities is not None)
            
            QMessageBox.information(
                self, "Save Successful",
                f"Map data successfully saved to PKL file!\n\n"
                f"File: {file_path}\n"
                f"Total Spectra: {spectra_count}\n"
                f"Processed Spectra: {processed_count}\n"
                f"Cosmic Ray Config: Included\n"
                f"Creation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            self.statusBar().showMessage(f"Map data saved to PKL: {file_path}")
            
        except Exception as e:
            self.progress_status.hide_progress()
            QMessageBox.critical(self, "Save Error", f"Error saving map data to PKL:\n{str(e)}")
            
    def load_map_from_pkl(self):
        """Load map data from a PKL file."""
        # Get file path
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Map Data from PKL", "", 
            "Pickle files (*.pkl);;All files (*)")
        
        if not file_path:
            return
            
        try:
            self.progress_status.show_progress("Loading map data from PKL...")
            
            # Load from pickle file with module compatibility handling
            import pickle
            import sys
            import types
            
            # Create module compatibility mapping
            class ModuleCompatibilityUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Handle old module names that have been renamed
                    if module.startswith('map_analysis_2d_qt6'):
                        # Replace old module name with new one
                        new_module = module.replace('map_analysis_2d_qt6', 'map_analysis_2d')
                        logger.info(f"PKL Compatibility: Redirecting {module}.{name} -> {new_module}.{name}")
                        try:
                            # Try to import the new module
                            mod = __import__(new_module, fromlist=[name])
                            result = getattr(mod, name)
                            logger.info(f"PKL Compatibility: Successfully found {name} in {new_module}")
                            return result
                        except (ImportError, AttributeError) as e:
                            # If new module doesn't have the class, try to find it elsewhere
                            logger.warning(f"Module compatibility: {module}.{name} -> {new_module}.{name} failed: {e}")
                            pass
                    
                    # Handle other legacy module names
                    elif module in ['raman_analysis_qt6', 'raman_map_analysis_qt6']:
                        logger.info(f"PKL Compatibility: Redirecting legacy module {module}.{name}")
                        # Try to find the class in the new module structure
                        try:
                            if name == 'RamanMapData':
                                from map_analysis_2d.core.file_io import RamanMapData
                                logger.info(f"PKL Compatibility: Found {name} in file_io module")
                                return RamanMapData
                            elif name in ['CosmicRayConfig', 'SimpleCosmicRayManager']:
                                from map_analysis_2d.core.cosmic_ray_detection import CosmicRayConfig, SimpleCosmicRayManager
                                result = CosmicRayConfig if name == 'CosmicRayConfig' else SimpleCosmicRayManager
                                logger.info(f"PKL Compatibility: Found {name} in cosmic_ray_detection module")
                                return result
                            elif name == 'SpectrumData':
                                from map_analysis_2d.core.spectrum_data import SpectrumData
                                logger.info(f"PKL Compatibility: Found {name} in spectrum_data module")
                                return SpectrumData
                        except ImportError as e:
                            logger.warning(f"Could not find {name} in new module structure: {e}")
                            pass
                    
                    # Handle other known compatibility issues
                    elif module == 'raman_map_data' and name == 'RamanMapData':
                        logger.info(f"PKL Compatibility: Redirecting {module}.{name} to file_io module")
                        # Import from the correct location
                        from map_analysis_2d.core.file_io import RamanMapData
                        return RamanMapData
                    elif module == 'cosmic_ray_detection' and name in ['CosmicRayConfig', 'SimpleCosmicRayManager']:
                        logger.info(f"PKL Compatibility: Redirecting {module}.{name} to cosmic_ray_detection module")
                        # Import from the correct location
                        from map_analysis_2d.core.cosmic_ray_detection import CosmicRayConfig, SimpleCosmicRayManager
                        if name == 'CosmicRayConfig':
                            return CosmicRayConfig
                        else:
                            return SimpleCosmicRayManager
                    
                    # Fall back to default behavior
                    return super().find_class(module, name)
            
            with open(file_path, 'rb') as f:
                unpickler = ModuleCompatibilityUnpickler(f)
                save_data = unpickler.load()
            
            # Extract data based on file format
            if isinstance(save_data, dict):
                # New format with metadata
                self.map_data = save_data['map_data']
                
                # Restore cosmic ray config if available
                if 'cosmic_ray_config' in save_data:
                    self.cosmic_ray_config = save_data['cosmic_ray_config']
                    self.cosmic_ray_manager = SimpleCosmicRayManager(self.cosmic_ray_config)
                
                # Get metadata
                metadata = save_data.get('metadata', {})
                creation_time = metadata.get('creation_time', 'Unknown')
                version = metadata.get('version', 'Unknown')
                
            else:
                # Legacy format (direct RamanMapData object)
                self.map_data = save_data
                creation_time = 'Unknown (Legacy Format)'
                version = 'Legacy'
                metadata = {}
            
            self.progress_status.hide_progress()
            
            # Initialize integration slider with spectrum midpoint
            self._initialize_integration_slider()
            
            # Update the display
            self.update_map()
            
            # Show loading summary
            spectra_count = len(self.map_data.spectra)
            processed_count = sum(1 for s in self.map_data.spectra.values() 
                                if s.processed_intensities is not None)
            
            summary_text = f"""Map Data Loaded Successfully!

File: {file_path}
Version: {version}
Creation Time: {creation_time}

Data Summary:
• Total Spectra: {spectra_count}
• Processed Spectra: {processed_count}
• Has Processed Data: {'Yes' if processed_count > 0 else 'No'}
• Cosmic Ray Config: {'Restored' if 'cosmic_ray_config' in (save_data if isinstance(save_data, dict) else {}) else 'Default Used'}

The map is now ready for analysis!"""
            
            QMessageBox.information(self, "Load Successful", summary_text)
            
            self.statusBar().showMessage(f"PKL map loaded: {spectra_count} spectra")
            
        except Exception as e:
            self.progress_status.hide_progress()
            
            # Provide more helpful error messages for common issues
            error_msg = str(e)
            if "No module named" in error_msg:
                if "map_analysis_2d_qt6" in error_msg:
                    error_msg = ("This PKL file was created with an older version of the software.\n\n"
                               "The module compatibility system should have handled this automatically. "
                               "Please try the following:\n\n"
                               "1. Ensure all software components are properly installed\n"
                               "2. Try loading the original map data files instead\n"
                               "3. Contact support if the issue persists\n\n"
                               f"Technical details: {error_msg}")
                else:
                    error_msg = ("PKL file contains references to modules that are not available.\n\n"
                               "This may happen when:\n"
                               "- The PKL file was created with a different version of the software\n"
                               "- Required modules are not installed\n"
                               "- The file is corrupted\n\n"
                               "Try loading the original map data files instead.\n\n"
                               f"Technical details: {error_msg}")
            elif "pickle" in error_msg.lower():
                error_msg = ("PKL file appears to be corrupted or incompatible.\n\n"
                           "Try loading the original map data files instead.\n\n"
                           f"Technical details: {error_msg}")
            else:
                error_msg = f"Error loading map data from PKL:\n\n{error_msg}"
            
            QMessageBox.critical(self, "PKL Load Error", error_msg)
            
    def get_map_save_info(self):
        """Get information about current map data for saving."""
        if self.map_data is None:
            return None
            
        info = {
            'total_spectra': len(self.map_data.spectra),
            'processed_spectra': sum(1 for s in self.map_data.spectra.values() 
                                   if s.processed_intensities is not None),
                        'cosmic_ray_stats': self.cosmic_ray_manager.get_statistics(),
            'has_processed_data': any(s.processed_intensities is not None 
                                    for s in self.map_data.spectra.values())
        }
        return info
        
    def suggest_pkl_save_after_load(self):
        """Suggest saving to PKL after loading individual map files."""
        if self.map_data is None:
            return
            
        spectra_count = len(self.map_data.spectra)
        
        # Only suggest for reasonably sized maps (to avoid annoying users with small test datasets)
        if spectra_count < 10:
            return
            
        # Create a non-blocking suggestion dialog
        reply = QMessageBox.question(
            self, "Save to PKL?",
            f"You've loaded {spectra_count} spectra from individual files.\n\n"
            f"Would you like to save this map data to a PKL file?\n\n"
            f"Benefits of PKL files:\n"
            f"• Much faster loading for future sessions\n"
            f"• Preserves all processing (cosmic ray removal, etc.)\n"
            f"• Smaller file size than individual spectrum files\n"
            f"• Retains all metadata and settings\n\n"
            f"Save to PKL now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.save_map_to_pkl()

    def plot_template_fit_overlay(self, spectrum, wavenumbers, intensities):
        """Plot template fitting overlay on the spectrum."""
        try:
            import numpy as np
            
            pos_key = (spectrum.x_pos, spectrum.y_pos)
            coeffs = self.template_fitting_results['coefficients'][pos_key]
            template_names = self.template_fitting_results['template_names']
            
            # Get template matrix for reconstruction
            template_matrix = self.template_manager.get_template_matrix()
            if template_matrix.size == 0:
                return
            
            # Add baseline if it was used in fitting
            if self.template_fitting_results['use_baseline']:
                baseline = np.ones(len(intensities))
                template_matrix = np.column_stack([template_matrix, baseline])
            
            # Reconstruct fitted spectrum
            if len(coeffs) == template_matrix.shape[1]:
                fitted_spectrum = template_matrix @ coeffs
                
                # Plot fitted spectrum
                self.map_plot_widget.spectrum_widget.ax.plot(
                    wavenumbers, fitted_spectrum,
                    color='green', linewidth=1.5, linestyle='--',
                    label='Template Fit'
                )
                
                # Plot individual template contributions if there aren't too many
                if len(template_names) <= 5:  # Only show individual contributions for a few templates
                    for i, (template, coeff) in enumerate(zip(self.template_manager.templates, coeffs[:len(template_names)])):
                        if coeff > 0.01:  # Only show significant contributions
                            contribution = template.processed_intensities * coeff
                            if len(contribution) == len(wavenumbers):
                                self.map_plot_widget.spectrum_widget.ax.plot(
                                    wavenumbers, contribution,
                                    color=template.color, alpha=0.7, linewidth=1,
                                    label=f'{template.name} ({coeff:.2f})'
                                )
                
                # Plot residual
                residual = intensities - fitted_spectrum
                self.map_plot_widget.spectrum_widget.ax.plot(
                    wavenumbers, residual,
                    color='orange', alpha=0.6, linewidth=1,
                    label='Residual'
                )
                
        except Exception as e:
            logger.error(f"Error plotting template fit overlay: {e}")

    def update_map_template_status(self):
        """Update the template status in the map control panel."""
        try:
            # Get current map control panel
            sections = self.controls_panel.sections
            if "map_controls" in sections:
                control_panel = sections["map_controls"]["widget"]
                if hasattr(control_panel, 'template_fitting_group'):
                    # Check if templates are loaded
                    template_count = self.template_manager.get_template_count()
                    
                    if template_count > 0:
                        # Show template fitting section
                        control_panel.template_fitting_group.setVisible(True)
                        control_panel.template_status_label.setText(f"{template_count} templates loaded")
                        control_panel.fit_templates_btn.setEnabled(True)
                        
                        # Update fitting results status
                        if hasattr(self, 'template_fitting_results'):
                            fitted_count = len(self.template_fitting_results['coefficients'])
                            avg_r_squared = sum(self.template_fitting_results['r_squared'].values()) / len(self.template_fitting_results['r_squared']) if self.template_fitting_results['r_squared'] else 0
                            control_panel.fitting_results_label.setText(
                                f"Fitted {fitted_count} spectra (avg R²: {avg_r_squared:.3f})"
                            )
                        else:
                            control_panel.fitting_results_label.setText("No fitting results available")
                    else:
                        # Hide template fitting section
                        control_panel.template_fitting_group.setVisible(False)
                        
        except Exception as e:
            logger.error(f"Error updating map template status: {e}")

    def show_model_result_map(self, model_name: str):
        """Manually show a model result map by setting the current feature."""
        try:
            feature_name = f"Model: {model_name}"
            
            # Check if model results exist
            if not hasattr(self, 'model_results') or feature_name not in self.model_results:
                logger.error(f"No model results found for '{feature_name}'")
                return False
                
            # Set the current feature
            self.current_feature = feature_name
            logger.info(f"Setting current feature to: {feature_name}")
            
            # Update the map
            self.update_map()
            
            # Switch to map tab
            self.tab_widget.setCurrentIndex(0)
            
            logger.info(f"Map updated to show results for model '{model_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error showing model result map: {e}")
            return False

    def _populate_ml_control_panel_models(self, control_panel):
        """Populate ML control panel with models from model manager."""
        try:
            model_names = self.model_manager.get_model_names()
            if model_names:
                control_panel.update_model_list(model_names)
                logger.info(f"Populated ML control panel with {len(model_names)} models: {model_names}")
            else:
                logger.info("No models found in model manager")
        except Exception as e:
            logger.error(f"Error populating ML control panel models: {e}")

    def _setup_model_persistence(self):
        """Setup automatic model persistence to file."""
        import os
        from pathlib import Path
        
        # Create models directory if it doesn't exist
        self.models_dir = Path("saved_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Define the models file path
        self.models_file = self.models_dir / "ml_models.pkl"
        
        # Load existing models on startup
        self._load_models_from_file()
        
    def _load_models_from_file(self):
        """Load saved models from file on startup."""
        try:
            if self.models_file.exists():
                if self.model_manager.load_models_from_file(str(self.models_file)):
                    model_count = self.model_manager.count()
                    if model_count > 0:
                        logger.info(f"Loaded {model_count} saved models from {self.models_file}")
                        return True
                else:
                    logger.warning(f"Failed to load models from {self.models_file}")
            else:
                logger.info("No saved models file found - starting with empty model collection")
                
        except Exception as e:
            logger.error(f"Error loading models from file: {e}")
            
        return False
    
    def _save_models_to_file(self):
        """Save all models to file."""
        try:
            if self.model_manager.count() > 0:
                if self.model_manager.save_models_to_file(str(self.models_file)):
                    logger.info(f"Saved {self.model_manager.count()} models to {self.models_file}")
                    return True
                else:
                    logger.error(f"Failed to save models to {self.models_file}")
            else:
                logger.info("No models to save")
                
        except Exception as e:
            logger.error(f"Error saving models to file: {e}")
            
        return False

    def refresh_map_features_with_models(self):
        """Manually refresh map features to include model results."""
        try:
            # Get the map control panel
            map_control_panel = self.get_current_map_control_panel()
            if map_control_panel is None:
                logger.warning("Cannot refresh map features - map control panel is None")
                # Try to ensure we're on the map tab and the control panel exists
                self.tab_widget.setCurrentIndex(0)  # Switch to Map View tab
                map_control_panel = self.get_current_map_control_panel()
                if map_control_panel is None:
                    logger.error("Map control panel still None after switching to Map View tab")
                    return False
            
            # Get current features
            current_features = map_control_panel.get_available_features()
            logger.info(f"Current map features: {current_features}")
            
            # Check if we have model results to add
            if hasattr(self, 'model_results') and self.model_results:
                logger.info(f"Available model results: {list(self.model_results.keys())}")
                
                # Add model features that aren't already in the list
                updated = False
                for model_feature in self.model_results.keys():
                    if model_feature not in current_features:
                        current_features.append(model_feature)
                        updated = True
                        logger.info(f"Added model feature: {model_feature}")
                
                if updated:
                    # Update the feature list
                    map_control_panel.update_feature_list(current_features)
                    logger.info(f"Updated map features to: {current_features}")
                    return True
                else:
                    logger.info("All model features already in the feature list")
            else:
                logger.warning("No model results available to add")
            
            return False
            
        except Exception as e:
            logger.error(f"Error refreshing map features: {e}")
            return False

    def debug_and_fix_map_features(self):
        """Debug method to check and manually add classification features to map dropdown."""
        try:
            # Get the map control panel
            control_panel = self.get_current_map_control_panel()
            if control_panel is None:
                logger.error("Could not find map control panel for debugging")
                return
            
            # Check current features
            current_features = [control_panel.feature_combo.itemText(i) 
                              for i in range(control_panel.feature_combo.count())]
            logger.info(f"Current features in dropdown: {current_features}")
            
            # Check if we have classification results
            if not hasattr(self, 'classification_results'):
                logger.info("No classification_results attribute found")
                return
                
            logger.info(f"Classification results keys: {list(self.classification_results.keys())}")
            
            # Manually add classification features if missing
            classification_features = [
                "ML Classification (Supervised)"
            ]
            
            # Add class probability features if available
            if 'class_names' in self.classification_results:
                for class_name in self.classification_results['class_names']:
                    prob_feature = f"Class Probability: {class_name}"
                    classification_features.append(prob_feature)
            
            # Add missing features
            for feature in classification_features:
                if feature not in current_features:
                    control_panel.feature_combo.addItem(feature)
                    logger.info(f"Manually added missing feature: {feature}")
            
            # Final check
            final_features = [control_panel.feature_combo.itemText(i) 
                            for i in range(control_panel.feature_combo.count())]
            logger.info(f"Final features in dropdown: {final_features}")
            
        except Exception as e:
            logger.error(f"Error in debug_and_fix_map_features: {e}")