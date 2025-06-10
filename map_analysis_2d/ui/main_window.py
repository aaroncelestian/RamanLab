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
from PySide6.QtGui import QAction, QFont
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
        
        # Export button - use PrimaryButton style
        from .base_widgets import PrimaryButton
        self.export_results_btn = PrimaryButton("Export Results")
        self.export_results_btn.clicked.connect(self.export_comprehensive_results)
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
            
            # Also check for template results
            if hasattr(self, 'template_fitting_results'):
                logger.info("Found template fitting results, adding to new map control panel...")
                self.update_map_features_with_templates()
            
        elif index == 1:  # Template Analysis
            control_panel = TemplateControlPanel()
            control_panel.load_template_file_requested.connect(self.load_template_file)
            control_panel.load_template_folder_requested.connect(self.load_template_folder)
            control_panel.extract_from_map_requested.connect(self.start_template_extraction_mode)
            control_panel.debug_templates_requested.connect(self.show_template_debug_tool)
            control_panel.remove_template_requested.connect(self.remove_template)
            control_panel.clear_templates_requested.connect(self.clear_templates)
            control_panel.plot_templates_requested.connect(self.plot_templates)
            control_panel.fit_templates_requested.connect(self.fit_templates)
            control_panel.normalize_templates_requested.connect(self.normalize_templates)
            control_panel.show_detailed_stats.connect(self.show_detailed_template_statistics)
            control_panel.export_statistics.connect(lambda: self.export_template_statistics(self.calculate_template_statistics()))
            control_panel.show_chemical_analysis.connect(self.show_chemical_validity_analysis)
            control_panel.show_hybrid_analysis.connect(self.show_hybrid_analysis_dialog)
            control_panel.show_pp_analysis.connect(self.show_template_only_polypropylene_results)
            self.controls_panel.add_section("template_controls", control_panel)
            
            # Update template list in the control panel
            self.update_template_control_panel()
            
            # Update statistics display if available
            self.update_template_statistics_display()
            
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
            control_panel.show_feature_info_requested.connect(self.show_ml_feature_info)
            
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
        """Handle spectrum request from map click - enhanced for template extraction."""
        if self.map_data is None:
            return
            
        # Handle template extraction mode first
        if hasattr(self, 'template_extraction_mode') and self.template_extraction_mode:
            self._extract_template_from_position(x, y)
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
        """Update the map display based on current settings."""
        if self.map_data is None:
            return

        try:
            # Get map data based on current feature
            map_data = None
            discrete_labels = None
            cmap = 'viridis'  # Default colormap

            if self.current_feature == "Integrated Intensity":
                map_data = self.create_integrated_intensity_map()
            elif self.current_feature == "Peak Height":
                # Could implement peak height mapping
                pass
            elif self.current_feature == "Cosmic Ray Map":
                map_data = self.create_cosmic_ray_map()
                cmap = 'Reds'
            elif self.current_feature.startswith("Template: "):
                template_name = self.current_feature[10:]  # Remove "Template: " prefix
                map_data = self.create_template_contribution_map(template_name)
            elif self.current_feature == "Template Fit Quality":
                map_data = self.create_template_fit_quality_map()
            elif self.current_feature == "Template Dominance Map":
                map_data = self.create_template_dominance_map()
                cmap = 'Set1'
                # Get template names for discrete labels
                if hasattr(self, 'template_fitting_results'):
                    template_names = self.template_fitting_results['template_names']
                    discrete_labels = template_names + ['Mixed/Unclear']
            elif self.current_feature.startswith("PCA Component "):
                component_num = int(self.current_feature.split()[-1]) - 1  # Convert to 0-based index
                map_data = self.create_pca_component_map(component_num)
            elif self.current_feature.startswith("NMF Component "):
                component_num = int(self.current_feature.split()[-1]) - 1  # Convert to 0-based index
                map_data = self.create_nmf_component_map(component_num)
            elif self.current_feature.startswith("ML Clusters"):
                map_data = self.create_ml_clustering_map()
                cmap = 'tab10'
                # Get cluster labels
                if hasattr(self, 'ml_results') and 'labels' in self.ml_results:
                    unique_labels = np.unique(self.ml_results['labels'])
                    discrete_labels = []
                    for label in unique_labels:
                        if label == -1:
                            discrete_labels.append('Noise')
                        else:
                            discrete_labels.append(f'Cluster {int(label)}')
            elif self.current_feature.startswith("ML Classification"):
                map_data = self.create_ml_classification_map()
                if map_data is not None:
                    cmap = 'Set1'  # Discrete colormap for classification
                    # Get class names for labeling
                    if hasattr(self, 'classification_results') and 'class_names' in self.classification_results:
                        discrete_labels = self.classification_results['class_names']
                    elif hasattr(self, 'supervised_analyzer') and hasattr(self.supervised_analyzer, 'class_names'):
                        discrete_labels = self.supervised_analyzer.class_names
            elif self.current_feature.startswith("Class Probability: "):
                # Fix: Strip whitespace from class name to handle spacing issues
                class_name = self.current_feature[19:].strip()  # Remove "Class Probability: " prefix and strip whitespace
                map_data = self.create_class_probability_map(class_name)
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
                    map_data = self.create_model_result_map()
                else:
                    cmap = 'viridis'
            elif "Hybrid: Confidence Map" in self.current_feature:
                map_data = self.create_hybrid_confidence_map()
                cmap = 'viridis'
            elif "Enhanced: NMF Component (Log Scale)" in self.current_feature:
                map_data = self.create_enhanced_nmf_component_map(log_scale=True)
                cmap = 'viridis'
            elif "Enhanced: NMF Component (High Contrast)" in self.current_feature:
                map_data = self.create_enhanced_nmf_component_map(high_contrast=True)
                cmap = 'plasma'
            elif "Hybrid: NMF Candidates" in self.current_feature:
                map_data = self.create_nmf_candidate_regions_map()
                cmap = 'Reds'
            elif "Hybrid: High Confidence Regions" in self.current_feature:
                map_data = self.create_high_confidence_regions_map()
                cmap = 'Oranges'
                
                # Update control panel with current data range for auto-scaling
                if hasattr(self, 'map_control_panel') and map_data is not None:
                    data_min, data_max = np.nanmin(map_data), np.nanmax(map_data)
                    self.map_control_panel.update_intensity_range(data_min, data_max)

            # Update visualization
            if map_data is not None:
                # Fix: Use the map plot widget instead of missing visualizer
                x_positions = [s.x_pos for s in self.map_data.spectra.values()]
                y_positions = [s.y_pos for s in self.map_data.spectra.values()]
                
                extent = [min(x_positions), max(x_positions), 
                         min(y_positions), max(y_positions)]
                
                # Create title with integration info
                title = f"{self.current_feature} Map"
                if (self.current_feature == "Integrated Intensity" and 
                    hasattr(self, 'integration_center') and self.integration_center is not None and 
                    hasattr(self, 'integration_width') and self.integration_width is not None):
                    min_wn = self.integration_center - self.integration_width / 2
                    max_wn = self.integration_center + self.integration_width / 2
                    title += f" ({min_wn:.0f}-{max_wn:.0f} cm⁻¹)"
                
                # Use the map plot widget
                self.map_plot_widget.plot_map(
                    map_data, extent=extent,
                    title=title,
                    cmap=cmap,
                    discrete_labels=discrete_labels,
                    vmin=getattr(self, 'intensity_vmin', None),
                    vmax=getattr(self, 'intensity_vmax', None)
                )
            else:
                logger.warning(f"No data available for feature: {self.current_feature}")

        except Exception as e:
            logger.error(f"Error updating map for feature '{self.current_feature}': {e}")
            import traceback
            traceback.print_exc()

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

    def create_template_dominance_map(self):
        """Create a map showing which template dominates at each position."""
        if not hasattr(self, 'template_fitting_results'):
            return None
            
        try:
            import numpy as np
            
            coefficients = self.template_fitting_results['coefficients']
            template_names = self.template_fitting_results['template_names']
            n_templates = len(template_names)
            
            # Get map dimensions
            positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            map_array = np.full((y_max - y_min + 1, x_max - x_min + 1), n_templates, dtype=int)  # Default to "Mixed/Unclear"
            
            # Thresholds for dominance
            confidence_threshold = 0.30  # Must contribute at least 30%
            significance_margin = 0.10   # Must be 10% higher than second best
            
            # Fill map with dominant template indices
            for spectrum in self.map_data.spectra.values():
                pos_key = (spectrum.x_pos, spectrum.y_pos)
                if pos_key in coefficients:
                    coeffs = np.array(coefficients[pos_key][:n_templates])  # Only template coefficients, exclude baseline
                    
                    # Calculate relative contributions
                    total_contrib = np.sum(coeffs)
                    if total_contrib > 1e-10:  # Avoid division by zero
                        relative_contribs = coeffs / total_contrib
                        
                        # Find dominant template
                        dominant_idx = np.argmax(relative_contribs)
                        dominant_strength = relative_contribs[dominant_idx]
                        
                        # Check if it meets dominance criteria
                        if dominant_strength > confidence_threshold:
                            # Check significance margin
                            sorted_contribs = np.sort(relative_contribs)[::-1]
                            if len(sorted_contribs) > 1:
                                margin = sorted_contribs[0] - sorted_contribs[1]
                                if margin > significance_margin:
                                    map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = dominant_idx
            
            return map_array
            
        except Exception as e:
            logger.error(f"Error creating template dominance map: {e}")
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
            
            # DEBUG: Print template fitting setup info
            print(f"\n=== TEMPLATE FITTING DEBUG ===")
            print(f"Map data: {len(self.map_data.spectra)} spectra")
            print(f"Map wavenumbers: {len(first_spectrum.wavenumbers)} points ({first_spectrum.wavenumbers[0]:.1f} to {first_spectrum.wavenumbers[-1]:.1f})")
            print(f"Target intensities length: {len(target_intensities)}")
            print(f"Using {'processed' if self.use_processed else 'raw'} data")
            print(f"Number of templates: {self.template_manager.get_template_count()}")
            
            # Get template matrix
            template_matrix = self.template_manager.get_template_matrix()
            if template_matrix.size == 0:
                raise ValueError("No valid template data available")
            
            print(f"Template matrix shape: {template_matrix.shape}")
            print(f"Template matrix min/max: {np.min(template_matrix):.6f} to {np.max(template_matrix):.6f}")
            
            # Print template info
            for i, name in enumerate(self.template_manager.get_template_names()):
                template_col = template_matrix[:, i]
                print(f"Template '{name}': min={np.min(template_col):.6f}, max={np.max(template_col):.6f}, std={np.std(template_col):.6f}")
            print("==============================\n")
            
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
                        self.progress_status.update_progress(progress, f"Fitting templates... {progress}%")
                
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
            
            # Calculate and display basic statistics
            self.update_template_statistics_display()
            
            self.statusBar().showMessage(f"Template fitting complete: {processed_count} spectra fitted")
            
        except Exception as e:
            self.progress_status.hide_progress()
            QMessageBox.critical(self, "Error", f"Error in template fitting:\n{str(e)}")
            logger.error(f"Error in template fitting: {e}")
            
    def update_map_features_with_templates(self):
        """Update map features dropdown to include template fitting results."""
        try:
            logger.info("Updating map features with template results...")
            
            # Check if we're currently on the map tab - if so, update directly
            if self.tab_widget.currentIndex() == 0:  # Map View tab
                sections = self.controls_panel.sections
                logger.info(f"Available control panel sections: {list(sections.keys())}")
                
                if "map_controls" in sections:
                    control_panel = sections["map_controls"]["widget"]
                    logger.info(f"Found map control panel: {type(control_panel)}")
                    
                    if hasattr(control_panel, 'feature_combo'):
                        self._update_control_panel_features(control_panel)
                    else:
                        logger.warning("Map control panel does not have feature_combo attribute")
                else:
                    logger.warning("Map controls section not found in control panel")
            else:
                # Not on map tab - the features will be updated when we switch to the map tab
                # via the on_tab_changed method, but we should also check for any existing
                # map control panels that might need updating
                logger.info("Not currently on map tab - features will update when switching to map view")
                
            # Also update any cached/stored feature lists
            logger.info("Template features update initiated")
                        
        except Exception as e:
            logger.error(f"Error updating map features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_control_panel_features(self, control_panel):
        """Helper method to update a control panel's feature dropdown with template results."""
        try:
            if not hasattr(control_panel, 'feature_combo'):
                return
                
            logger.info(f"Found feature combo with {control_panel.feature_combo.count()} items")
            
            if hasattr(self, 'template_fitting_results'):
                logger.info(f"Template fitting results available with {len(self.template_fitting_results['template_names'])} templates")
                
                # Store current selection
                current_feature = control_panel.feature_combo.currentText()
                logger.info(f"Current feature selection: {current_feature}")
                
                # Get current features and remove old template features to avoid duplicates
                current_features = [control_panel.feature_combo.itemText(i) 
                                  for i in range(control_panel.feature_combo.count())]
                
                # Remove old template features to avoid duplicates
                filtered_features = []
                for feature in current_features:
                    if not (feature.startswith("Template: ") or 
                           feature == "Template Fit Quality (R²)" or
                           feature == "Template Dominance Map"):
                        filtered_features.append(feature)
                
                # Rebuild the dropdown with all features
                control_panel.feature_combo.clear()
                control_panel.feature_combo.addItems(filtered_features)
                
                # Add template contribution maps
                for template_name in self.template_fitting_results['template_names']:
                    feature_name = f"Template: {template_name}"
                    control_panel.feature_combo.addItem(feature_name)
                    logger.info(f"Added template feature: {feature_name}")
                
                # Add fit quality map
                control_panel.feature_combo.addItem("Template Fit Quality (R²)")
                logger.info("Added template fit quality feature")
                
                # Add template dominance map
                control_panel.feature_combo.addItem("Template Dominance Map")
                logger.info("Added template dominance map feature")
                
                # Restore selection if possible
                index = control_panel.feature_combo.findText(current_feature)
                if index >= 0:
                    control_panel.feature_combo.setCurrentIndex(index)
                    logger.info(f"Restored selection to: {current_feature}")
                else:
                    logger.info(f"Could not restore selection '{current_feature}', keeping default")
                
                logger.info(f"Template features update completed. Total features: {control_panel.feature_combo.count()}")
            else:
                logger.warning("No template fitting results available")
                
        except Exception as e:
            logger.error(f"Error updating control panel features: {e}")
    
    def get_base_map_features(self):
        """Get the base set of map features that should always be available."""
        return [
            "Integrated Intensity",
            "Peak Height", 
            "Cosmic Ray Map"
        ]
    
    def get_all_available_map_features(self):
        """Get all currently available map features from all analysis methods."""
        features = self.get_base_map_features().copy()
        
        # Add template features if available
        if hasattr(self, 'template_fitting_results'):
            for template_name in self.template_fitting_results['template_names']:
                features.append(f"Template: {template_name}")
            features.append("Template Fit Quality (R²)")
            features.append("Template Dominance Map")
        
        # Add PCA components if available
        if hasattr(self, 'pca_results') and self.pca_results is not None:
            n_components = self.pca_results.get('n_components', 0)
            for i in range(n_components):
                features.append(f"PCA Component {i+1}")
        
        # Add NMF components if available
        if hasattr(self, 'nmf_results') and self.nmf_results is not None:
            n_components = self.nmf_results.get('n_components', 0)
            for i in range(n_components):
                features.append(f"NMF Component {i+1}")
        
        # Add ML clustering results if available
        if hasattr(self, 'ml_results') and self.ml_results.get('type') == 'unsupervised':
            features.append("ML Clusters")
        
        # Add ML classification results if available
        if hasattr(self, 'classification_results') and self.classification_results:
            if self.classification_results.get('type') == 'supervised':
                features.append("ML Classification")
                # Add class probability maps
                if 'class_names' in self.classification_results:
                    for class_name in self.classification_results['class_names']:
                        features.append(f"Class Probability: {class_name}")
        
        # Add model-based features if available
        if hasattr(self, 'model_results') and self.model_results:
            for model_name in self.model_results.keys():
                features.append(f"Model: {model_name}")
        
        return features
    
    def refresh_all_map_features(self):
        """Refresh the map features dropdown with all available features."""
        try:
            # Get current map control panel
            control_panel = self.get_current_map_control_panel()
            if control_panel is None or not hasattr(control_panel, 'feature_combo'):
                logger.warning("Map control panel or feature combo not found")
                return
            
            # Store current selection
            current_feature = control_panel.feature_combo.currentText()
            
            # Get all available features
            all_features = self.get_all_available_map_features()
            
            # Update the dropdown
            control_panel.feature_combo.clear()
            control_panel.feature_combo.addItems(all_features)
            
            # Restore selection if possible
            index = control_panel.feature_combo.findText(current_feature)
            if index >= 0:
                control_panel.feature_combo.setCurrentIndex(index)
                logger.info(f"Restored selection to: {current_feature}")
            else:
                logger.info(f"Could not restore selection '{current_feature}', keeping default")
            
            logger.info(f"Refreshed map features. Total features: {len(all_features)}")
            
        except Exception as e:
            logger.error(f"Error refreshing map features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
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
            # Use the comprehensive feature refresh system
            self.refresh_all_map_features()
            logger.info(f"Added {self.nmf_results['n_components']} NMF components to map features")
            
        except Exception as e:
            logger.error(f"Error updating map features with NMF: {e}")
    
    def create_pca_component_map(self, component_index: int):
        """Create a map showing PCA component contribution."""
        if not hasattr(self, 'pca_results') or self.pca_results is None:
            return None
            
        try:
            import numpy as np
            
            # Get PCA transformed data (components matrix W)
            pca_transformed = self.pca_results.get('transformed_data', self.pca_results.get('components'))
            if pca_transformed is None or component_index >= pca_transformed.shape[1]:
                logger.warning(f"PCA component {component_index + 1} not available")
                return None
            
            # Get component contributions for all spectra
            component_contributions = pca_transformed[:, component_index]
            
            # Create position mapping
            if hasattr(self, 'pca_valid_positions'):
                positions = self.pca_valid_positions
            else:
                # Fallback: use all map positions in order
                positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            
            # Create position to contribution mapping
            pos_to_contribution = {}
            for i, (x, y) in enumerate(positions):
                if i < len(component_contributions):
                    pos_to_contribution[(x, y)] = component_contributions[i]
            
            # Create map array
            all_positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            map_array = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            for spectrum in self.map_data.spectra.values():
                pos_key = (spectrum.x_pos, spectrum.y_pos)
                if pos_key in pos_to_contribution:
                    map_array[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = pos_to_contribution[pos_key]
            
            return map_array
            
        except Exception as e:
            logger.error(f"Error creating PCA component map: {e}")
            return None

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
            
            # Apply feature transformation to training data if requested
            if feature_transformer is not None:
                logger.info(f"Applying {feature_used.upper()} transformation to training data...")
                
                # For NMF/PCA, we need to align training data to map data dimensions first
                if feature_used in ['nmf', 'pca'] and self.map_data is not None:
                    # Get map wavenumber grid
                    map_spectrum = next(iter(self.map_data.spectra.values()))
                    if map_spectrum.wavenumbers is not None:
                        map_wavenumbers = map_spectrum.wavenumbers
                        
                        # Align training data to map wavenumber grid
                        if len(common_wavenumbers) != len(map_wavenumbers):
                            logger.info(f"Aligning training data to map wavenumber grid: {len(common_wavenumbers)} -> {len(map_wavenumbers)}")
                            try:
                                from scipy.interpolate import interp1d
                                X_aligned = []
                                for spectrum_data in X:
                                    # Interpolate to map wavenumber grid
                                    interp_func = interp1d(common_wavenumbers, spectrum_data, 
                                                         kind='linear', bounds_error=False, fill_value=0)
                                    aligned_spectrum = interp_func(map_wavenumbers)
                                    X_aligned.append(aligned_spectrum)
                                X = np.array(X_aligned)
                                logger.info(f"Training data aligned to map grid: {X.shape}")
                            except Exception as e:
                                logger.warning(f"Failed to align training data to map grid: {e}")
                
                X_transformed, actual_type = feature_transformer.transform_data(X, fallback_to_full=True)
                if X_transformed is not None:
                    X = X_transformed
                    logger.info(f"Training data transformed: {X.shape[0]} samples, {X.shape[1]} features ({actual_type})")
                    
                    # ENHANCED DISCRIMINATIVE FEATURE SELECTION FOR NMF
                    if actual_type == 'nmf' and len(class_names) == 2:
                        logger.info("Applying discriminative feature enhancement for NMF...")
                        
                        # Find positive and negative class indices
                        positive_class_idx = None
                        negative_class_idx = None
                        
                        # Look for common positive class indicators
                        positive_keywords = ['positive', 'pos', 'target', 'signal', 'hit', '1', 'true']
                        negative_keywords = ['negative', 'neg', 'background', 'noise', 'miss', '0', 'false']
                        
                        for i, class_name in enumerate(class_names):
                            class_lower = class_name.lower()
                            if any(keyword in class_lower for keyword in positive_keywords):
                                positive_class_idx = i
                            elif any(keyword in class_lower for keyword in negative_keywords):
                                negative_class_idx = i
                        
                        # If we can't find by keywords, assume first class is negative, second is positive
                        if positive_class_idx is None or negative_class_idx is None:
                            positive_class_idx = 1 if len(class_names) > 1 else 0
                            negative_class_idx = 0 if positive_class_idx == 1 else 1
                            logger.info(f"Using default assignment: positive={class_names[positive_class_idx]}, negative={class_names[negative_class_idx]}")
                        else:
                            logger.info(f"Detected classes: positive={class_names[positive_class_idx]}, negative={class_names[negative_class_idx]}")
                        
                        positive_mask = y == positive_class_idx
                        negative_mask = y == negative_class_idx
                        
                        logger.info(f"Positive samples: {np.sum(positive_mask)}, Negative samples: {np.sum(negative_mask)}")
                        
                        component_discriminability = []
                        enhanced_features = []
                        enhancement_thresholds = []
                        threshold_directions = []
                        
                        for comp_idx in range(X.shape[1]):
                            pos_values = X[positive_mask, comp_idx]
                            neg_values = X[negative_mask, comp_idx]
                            
                            if len(pos_values) == 0 or len(neg_values) == 0:
                                logger.warning(f"Component {comp_idx}: Empty class found, skipping enhancement")
                                enhanced_features.append(X[:, comp_idx])
                                component_discriminability.append(0.0)
                                enhancement_thresholds.append(0.0)
                                threshold_directions.append('none')
                                continue
                            
                            pos_mean = np.mean(pos_values)
                            neg_mean = np.mean(neg_values)
                            pos_std = np.std(pos_values) + 1e-10  # Add small epsilon to avoid division by zero
                            neg_std = np.std(neg_values) + 1e-10
                            
                            # Calculate separability (Cohen's d effect size)
                            pooled_std = np.sqrt(((len(pos_values) - 1) * pos_std**2 + (len(neg_values) - 1) * neg_std**2) / 
                                               (len(pos_values) + len(neg_values) - 2))
                            cohen_d = abs(pos_mean - neg_mean) / (pooled_std + 1e-10)
                            
                            component_discriminability.append(cohen_d)
                            
                            logger.info(f"Component {comp_idx}: pos_mean={pos_mean:.3f}, neg_mean={neg_mean:.3f}, Cohen's d={cohen_d:.3f}")
                            
                            # Create enhanced feature based on discriminability
                            if cohen_d > 0.3:  # Lower threshold for better sensitivity
                                if pos_mean > neg_mean:
                                    # Positive class has higher values - use moderate threshold
                                    threshold = neg_mean + 1.0 * neg_std  # Less conservative threshold
                                    enhanced_feature = np.maximum(0, X[:, comp_idx] - threshold)
                                    direction = 'positive'
                                else:
                                    # Negative class has higher values - invert and threshold
                                    threshold = pos_mean + 1.0 * pos_std
                                    enhanced_feature = np.maximum(0, threshold - X[:, comp_idx])
                                    direction = 'negative'
                                
                                enhancement_thresholds.append(threshold)
                                threshold_directions.append(direction)
                                logger.info(f"Component {comp_idx}: Enhanced with threshold={threshold:.3f}, direction={direction}")
                            else:
                                # Low discriminability - keep original but slightly downweight
                                enhanced_feature = X[:, comp_idx] * 0.5  # Less aggressive downweighting
                                enhancement_thresholds.append(0.0)
                                threshold_directions.append('none')
                                logger.info(f"Component {comp_idx}: Low discriminability, slightly downweighted")
                            
                            enhanced_features.append(enhanced_feature)
                        
                        # Replace original features with enhanced features
                        X_enhanced = np.column_stack(enhanced_features)
                        
                        # Add multiple selectivity scores for robustness
                        discriminability_array = np.array(component_discriminability)
                        
                        # Top components selectivity
                        n_top = min(3, len(discriminability_array))  # Use top 3 or all if fewer
                        top_components = np.argsort(discriminability_array)[-n_top:]
                        
                        selectivity_score = np.zeros(X.shape[0])
                        if np.sum(discriminability_array[top_components]) > 0:
                            for comp_idx in top_components:
                                weight = discriminability_array[comp_idx] / np.sum(discriminability_array[top_components])
                                selectivity_score += weight * enhanced_features[comp_idx]
                        
                        # Ratio-based feature (positive vs negative signature)
                        pos_signature = np.mean([enhanced_features[i] for i in range(len(enhanced_features)) 
                                               if threshold_directions[i] == 'positive'], axis=0) if any(d == 'positive' for d in threshold_directions) else np.zeros(X.shape[0])
                        neg_signature = np.mean([enhanced_features[i] for i in range(len(enhanced_features)) 
                                               if threshold_directions[i] == 'negative'], axis=0) if any(d == 'negative' for d in threshold_directions) else np.zeros(X.shape[0])
                        
                        ratio_feature = pos_signature / (neg_signature + 1e-6)  # Avoid division by zero
                        
                        # Combine enhanced features with selectivity scores
                        X = np.column_stack([X_enhanced, selectivity_score, ratio_feature])
                        
                        logger.info(f"Enhanced NMF features: {X_enhanced.shape[1]} components + 2 selectivity scores = {X.shape[1]} total features")
                        logger.info(f"Most discriminative components: {top_components} (Cohen's d: {discriminability_array[top_components]})")
                        
                        # Store enhancement parameters for consistent application during classification
                        self.supervised_analyzer.component_discriminability = component_discriminability
                        self.supervised_analyzer.enhancement_thresholds = enhancement_thresholds
                        self.supervised_analyzer.threshold_directions = threshold_directions
                        self.supervised_analyzer.top_components = top_components
                        self.supervised_analyzer.positive_class_idx = positive_class_idx
                        self.supervised_analyzer.negative_class_idx = negative_class_idx
                    
                else:
                    logger.warning(f"Failed to apply {feature_used} transformation to training data, using full spectrum")
                    feature_used = 'full'
                    feature_transformer = None
            
            # Store the feature transformer in the analyzer for consistent application
            self.supervised_analyzer.feature_transformer = feature_transformer
            self.supervised_analyzer.feature_type = feature_used
            
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
            
            # Store the effective wavenumber grid used for training
            if feature_used in ['nmf', 'pca'] and self.map_data is not None:
                # For transformed features, store the map wavenumbers (what was actually used)
                map_spectrum = next(iter(self.map_data.spectra.values()))
                if map_spectrum.wavenumbers is not None:
                    self.supervised_analyzer.training_wavenumbers = map_spectrum.wavenumbers
                    logger.info(f"Stored map wavenumbers for {feature_used} training: {len(map_spectrum.wavenumbers)} points")
                else:
                    self.supervised_analyzer.training_wavenumbers = common_wavenumbers
                    logger.info(f"Stored interpolated wavenumbers for {feature_used} training: {len(common_wavenumbers)} points")
            else:
                # For full spectrum, store the interpolated common grid
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
            # Clean class names to prevent spacing issues
            clean_class_names = [name.strip() for name in class_names] if class_names else []
            self.classification_results = {
                'predictions': [],  # Empty until map is classified
                'probabilities': None,
                'positions': [],
                'type': 'supervised',
                'class_names': clean_class_names
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
                feature_transformer = getattr(self.supervised_analyzer, 'feature_transformer', None)
                feature_used = getattr(self.supervised_analyzer, 'feature_type', 'full')
                
                if feature_transformer is not None:
                    try:
                        self.statusBar().showMessage(f"Applying {feature_used.upper()} transformation to map data...")
                        X_transformed, actual_type = feature_transformer.transform_data(X, fallback_to_full=True)
                        if X_transformed is not None:
                            X = X_transformed
                            logger.info(f"Applied {actual_type} transformation to map data: {X.shape}")
                            
                            # Apply the same enhanced discriminative features if available
                            if (actual_type == 'nmf' and 
                                hasattr(self.supervised_analyzer, 'component_discriminability') and 
                                hasattr(self.supervised_analyzer, 'enhancement_thresholds')):
                                
                                logger.info("Applying enhanced discriminative features to map data...")
                                
                                component_discriminability = self.supervised_analyzer.component_discriminability
                                enhancement_thresholds = self.supervised_analyzer.enhancement_thresholds
                                threshold_directions = self.supervised_analyzer.threshold_directions
                                
                                enhanced_features = []
                                
                                for comp_idx in range(X.shape[1]):
                                    if comp_idx < len(component_discriminability):
                                        cohen_d = component_discriminability[comp_idx]
                                        
                                        if cohen_d > 0.3:  # Match training threshold (0.3, not 0.5)
                                            threshold = enhancement_thresholds[comp_idx]
                                            direction = threshold_directions[comp_idx]
                                            
                                            if direction == 'positive':
                                                enhanced_feature = np.maximum(0, X[:, comp_idx] - threshold)
                                            elif direction == 'negative':
                                                enhanced_feature = np.maximum(0, threshold - X[:, comp_idx])
                                            else:
                                                # 'none' direction - slightly downweight like in training
                                                enhanced_feature = X[:, comp_idx] * 0.5
                                            
                                            logger.info(f"Map Component {comp_idx}: Enhanced with threshold={threshold:.3f}, direction={direction}")
                                        else:
                                            # Low discriminability - match training downweighting (0.5, not 0.1)
                                            enhanced_feature = X[:, comp_idx] * 0.5
                                            logger.info(f"Map Component {comp_idx}: Low discriminability, slightly downweighted")
                                    else:
                                        # Fallback for components beyond training range
                                        enhanced_feature = X[:, comp_idx]
                                        logger.warning(f"Map Component {comp_idx}: No training data, using original")
                                    
                                    enhanced_features.append(enhanced_feature)
                                
                                # Replace original features with enhanced features
                                X_enhanced = np.column_stack(enhanced_features)
                                
                                # Recreate the selectivity score (match training logic)
                                discriminability_array = np.array(component_discriminability)
                                n_top = min(3, len(discriminability_array))  # Use top 3 or all if fewer
                                top_components = np.argsort(discriminability_array)[-n_top:]
                                
                                selectivity_score = np.zeros(X.shape[0])
                                if np.sum(discriminability_array[top_components]) > 0:
                                    for comp_idx in top_components:
                                        weight = discriminability_array[comp_idx] / np.sum(discriminability_array[top_components])
                                        selectivity_score += weight * enhanced_features[comp_idx]
                                
                                # Recreate the ratio-based feature (match training logic)
                                pos_signature = np.mean([enhanced_features[i] for i in range(len(enhanced_features)) 
                                                       if i < len(threshold_directions) and threshold_directions[i] == 'positive'], axis=0) if any(d == 'positive' for d in threshold_directions) else np.zeros(X.shape[0])
                                neg_signature = np.mean([enhanced_features[i] for i in range(len(enhanced_features)) 
                                                       if i < len(threshold_directions) and threshold_directions[i] == 'negative'], axis=0) if any(d == 'negative' for d in threshold_directions) else np.zeros(X.shape[0])
                                
                                ratio_feature = pos_signature / (neg_signature + 1e-6)  # Avoid division by zero
                                
                                # Combine enhanced features with both selectivity scores (match training)
                                X = np.column_stack([X_enhanced, selectivity_score, ratio_feature])
                                
                                logger.info(f"Applied enhanced NMF features to map: {len(enhanced_features)} components + 2 selectivity scores = {X.shape[1]} total features")
                        else:
                            logger.warning(f"Failed to apply {feature_used} transformation, using full spectrum")
                            feature_used = 'full'
                    except Exception as e:
                        logger.warning(f"Feature transformation failed: {str(e)}, using full spectrum")
                        feature_used = 'full'
                else:
                    logger.info(f"No feature transformation stored in analyzer, using {feature_used} features")
                
                # Step 3: Use the trained model for prediction
                try:
                    predictions = self.supervised_analyzer.model.predict(X)
                    probabilities = None
                    if hasattr(self.supervised_analyzer.model, 'predict_proba'):
                        try:
                            probabilities = self.supervised_analyzer.model.predict_proba(X)
                        except:
                            probabilities = None
                    
                    # Add detailed logging about predictions
                    unique_predictions, prediction_counts = np.unique(predictions, return_counts=True)
                    logger.info(f"Classification results:")
                    logger.info(f"  Total spectra classified: {len(predictions)}")
                    logger.info(f"  Unique predictions: {unique_predictions}")
                    logger.info(f"  Prediction counts: {prediction_counts}")
                    
                    if hasattr(self.supervised_analyzer, 'class_names') and self.supervised_analyzer.class_names:
                        for pred_val, count in zip(unique_predictions, prediction_counts):
                            if pred_val < len(self.supervised_analyzer.class_names):
                                class_name = self.supervised_analyzer.class_names[pred_val]
                                percentage = (count / len(predictions)) * 100
                                logger.info(f"  Class '{class_name}' (label {pred_val}): {count} spectra ({percentage:.1f}%)")
                    
                    if probabilities is not None:
                        logger.info(f"  Probability matrix shape: {probabilities.shape}")
                        logger.info(f"  Average probabilities per class: {np.mean(probabilities, axis=0)}")
                        
                        # Find high-confidence positive predictions
                        if hasattr(self.supervised_analyzer, 'positive_class_idx'):
                            pos_idx = self.supervised_analyzer.positive_class_idx
                            high_conf_positive = np.sum(probabilities[:, pos_idx] > 0.7)
                            medium_conf_positive = np.sum(probabilities[:, pos_idx] > 0.5)
                            logger.info(f"  High confidence positive (>70%): {high_conf_positive} spectra")
                            logger.info(f"  Medium confidence positive (>50%): {medium_conf_positive} spectra")
                    
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
            # Use the comprehensive feature refresh system
            self.refresh_all_map_features()
            logger.info("Added clustering results to map features")
            
        except Exception as e:
            logger.error(f"Error updating map features with clustering: {e}")
    
    def update_map_features_with_classification(self):
        """Update map view features to include classification results."""
        if not hasattr(self, 'classification_results'):
            return
            
        try:
            # Use the comprehensive feature refresh system
            self.refresh_all_map_features()
            logger.info("Added classification results to map features")
            
        except Exception as e:
            logger.error(f"Error updating map features with classification: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            # Fix: Clean up class names and make matching more robust
            clean_class_name = class_name.strip().lower()
            clean_class_names = [name.strip().lower() for name in class_names]
            
            # Find the class index with robust matching
            try:
                class_index = clean_class_names.index(clean_class_name)
            except ValueError:
                # Try partial matching if exact match fails
                for i, name in enumerate(clean_class_names):
                    if clean_class_name in name or name in clean_class_name:
                        class_index = i
                        logger.info(f"Found partial match for '{class_name}' with '{class_names[i]}'")
                        break
                else:
                    logger.error(f"Class '{class_name}' not found in class names: {class_names}")
                    logger.error(f"Available class names: {[name.strip() for name in class_names]}")
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
        """Plot comprehensive analysis results with optimized 2x2 layout focusing on key analyses."""
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
            
            # Create 2x2 grid with improved spacing for clarity
            gs = GridSpec(2, 2, figure=self.results_plot_widget.figure, 
                         hspace=0.4, wspace=0.4,
                         left=0.12, right=0.88, top=0.88, bottom=0.12)
            
            # Plot 1: PCA scatter with positive groups (keep as is - user likes it)
            ax1 = self.results_plot_widget.figure.add_subplot(gs[0, 0])
            self._plot_pca_scatter_with_positive_groups(ax1)
            
            # Plot 2: NMF scatter with positive groups (keep as is - user likes it)
            ax2 = self.results_plot_widget.figure.add_subplot(gs[0, 1])
            self._plot_nmf_scatter_with_positive_groups(ax2)
            
            # Plot 3: Top 5 spectral matches (improved and fixed)
            ax3 = self.results_plot_widget.figure.add_subplot(gs[1, 0])
            self._plot_top_spectral_matches(ax3)
            
            # Plot 4: Analysis statistics (focused on key metrics)
            ax4 = self.results_plot_widget.figure.add_subplot(gs[1, 1])
            self._plot_component_statistics(ax4)
            
            # Add overall title
            self.results_plot_widget.figure.suptitle('Comprehensive Raman Map Analysis Results', 
                                                    fontsize=14, fontweight='bold', y=0.98)
            
            # Update statistics text
            self._update_statistics_text()
            
            # Draw the canvas
            self.results_plot_widget.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error plotting comprehensive results: {e}")
            # Show error message
            self.results_plot_widget.figure.clear()
            ax = self.results_plot_widget.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error generating comprehensive results:\n{str(e)}\n\nPlease ensure all analysis steps are complete.\n\nSteps to complete:\n• Load map data\n• Run PCA analysis\n• Run NMF analysis\n• Train and apply ML model (optional)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"))
            ax.set_title('Comprehensive Results - Error')
            ax.axis('off')
            self.results_plot_widget.canvas.draw()
            
    def _plot_pca_scatter_with_positive_groups(self, ax):
        """Plot PCA scatter plot with positive groups color-coded."""
        try:
            if not hasattr(self, 'pca_results') or self.pca_results is None:
                ax.text(0.5, 0.5, 'PCA results not available.\nRun PCA analysis first.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('PCA Analysis - Not Available')
                ax.axis('off')
                return
            
            # PCA results use 'transformed_data' key
            pca_transformed = self.pca_results.get('transformed_data', self.pca_results.get('components'))
            if pca_transformed is None:
                ax.text(0.5, 0.5, 'PCA transformed data not available.\nCheck PCA results structure.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('PCA Analysis - Data Not Available')
                ax.axis('off')
                return
            
            # Use first two components for scatter plot
            if pca_transformed.shape[1] < 2:
                ax.text(0.5, 0.5, 'PCA results have less than 2 components.\nCannot create scatter plot.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('PCA Analysis - Insufficient Components')
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
                             c='orange', alpha=0.8, s=40, label='Positive Groups', edgecolors='darkorange', linewidths=0.5)
            else:
                # No classification available, use clustering if available
                if hasattr(self, 'pca_results') and 'cluster_labels' in self.pca_results:
                    cluster_labels = self.pca_results['cluster_labels']
                    unique_labels = np.unique(cluster_labels)
                    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
                    
                    for i, label in enumerate(unique_labels):
                        mask = cluster_labels == label
                        ax.scatter(pca_transformed[mask, 0], pca_transformed[mask, 1], 
                                 c=[colors[i]], alpha=0.7, s=30, label=f'Cluster {label}')
                else:
                    # Basic PCA plot
                    ax.scatter(pca_transformed[:, 0], pca_transformed[:, 1], 
                             c='blue', alpha=0.6, s=20)
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('PCA Analysis with Positive Groups')
            ax.grid(True, alpha=0.3)
            if positive_mask is not None or (hasattr(self, 'pca_results') and 'cluster_labels' in self.pca_results):
                # Use bbox_to_anchor to prevent tight_layout issues
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                
        except Exception as e:
            logger.error(f"Error plotting PCA scatter: {e}")
            ax.text(0.5, 0.5, f'Error plotting PCA scatter:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('PCA Analysis - Error')
            ax.axis('off')

    def _plot_nmf_scatter_with_positive_groups(self, ax):
        """Plot NMF scatter plot with positive groups color-coded."""
        try:
            if not hasattr(self, 'nmf_results') or self.nmf_results is None:
                ax.text(0.5, 0.5, 'NMF results not available.\nRun NMF analysis first.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('NMF Analysis - Not Available')
                ax.axis('off')
                return
            
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
                # Use bbox_to_anchor to prevent tight_layout issues
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                
        except Exception as e:
            logger.error(f"Error plotting NMF scatter: {e}")
            ax.text(0.5, 0.5, f'Error plotting NMF scatter:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('NMF Analysis - Error')
            ax.axis('off')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting NMF results:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('NMF Analysis - Error')
            ax.axis('off')
            
    def _plot_top_spectral_matches(self, ax):
        """Plot top 5 spectral matches from classification."""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Check if we have spectral data to analyze
            if not hasattr(self, 'map_data') or self.map_data is None:
                ax.text(0.5, 0.5, 'No map data available.\nLoad map data first.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('Top Spectral Matches - No Data')
                ax.axis('off')
                return
            
            # First try to get positive groups from ML classification
            positive_mask = self._get_positive_groups_mask()
            
            # If no positive groups from ML, try to find interesting spectra from other analyses
            if positive_mask is None or not np.any(positive_mask):
                # Try to identify interesting spectra from PCA/NMF results
                positive_mask = self._find_interesting_spectra_fallback()
            
            if positive_mask is None or not np.any(positive_mask):
                ax.text(0.5, 0.5, 'No interesting spectra identified.\n\nTry one of these steps:\n• Run ML classification\n• Run PCA/NMF analysis\n• Check that analysis completed successfully', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('Top Spectral Matches - No Interesting Groups')
                ax.axis('off')
                return
            
            # Get spectra from positive/interesting groups
            positive_indices = np.where(positive_mask)[0]
            
            # Limit to top 5 interesting spectra
            if len(positive_indices) > 5:
                # Try to rank by classification confidence or other metrics
                top_positive_indices = self._rank_interesting_spectra(positive_indices, positive_mask)[:5]
            else:
                top_positive_indices = positive_indices
            
            # Plot the spectra
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_positive_indices)))
            
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
                intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min()) + plotted_count * 0.3
                
                ax.plot(wavenumbers, intensities_norm, color=colors[plotted_count], 
                       linewidth=1.5, alpha=0.8, label=f'Match #{plotted_count+1} (pos: {spectrum.x_pos}, {spectrum.y_pos})')
                plotted_count += 1
            
            if plotted_count == 0:
                ax.text(0.5, 0.5, 'No valid spectra found for plotting.\nCheck that map data contains valid spectral information.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title('Top Spectral Matches - No Valid Data')
                ax.axis('off')
                return
            
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Normalized Intensity (offset)')
            ax.set_title(f'Top {plotted_count} Interesting Spectra')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting spectral matches:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Top Spectral Matches - Error')
            ax.axis('off')
    
    def _find_interesting_spectra_fallback(self):
        """Find interesting spectra when ML classification is not available."""
        try:
            import numpy as np
            
            # Try to use PCA outliers
            if hasattr(self, 'pca_results') and self.pca_results is not None:
                pca_transformed = self.pca_results.get('components', self.pca_results.get('transformed_data'))
                if pca_transformed is not None and pca_transformed.shape[0] > 0:
                    # Find outliers in PCA space (furthest from center)
                    center = np.mean(pca_transformed[:, :2], axis=0)
                    distances = np.linalg.norm(pca_transformed[:, :2] - center, axis=1)
                    # Consider top 10% as outliers/interesting
                    threshold = np.percentile(distances, 90)
                    return distances >= threshold
            
            # Try to use NMF high-contribution spectra
            if hasattr(self, 'nmf_results') and self.nmf_results is not None:
                nmf_transformed = self.nmf_results.get('components', self.nmf_results.get('transformed_data'))
                if nmf_transformed is not None and nmf_transformed.shape[0] > 0:
                    # Find spectra with high contributions to any component
                    max_contributions = np.max(nmf_transformed, axis=1)
                    threshold = np.percentile(max_contributions, 85)
                    return max_contributions >= threshold
            
            # Fallback: use intensity-based selection
            if hasattr(self, 'map_data') and self.map_data is not None:
                spectra_intensities = []
                for spectrum in self.map_data.spectra.values():
                    intensities = (spectrum.processed_intensities 
                                 if self.use_processed and spectrum.processed_intensities is not None
                                 else spectrum.intensities)
                    if intensities is not None:
                        # Use total intensity or maximum peak as criterion
                        spectra_intensities.append(np.max(intensities))
                    else:
                        spectra_intensities.append(0)
                
                if spectra_intensities:
                    intensities_array = np.array(spectra_intensities)
                    # Select top 10% by intensity
                    threshold = np.percentile(intensities_array, 90)
                    return intensities_array >= threshold
            
            return None
            
        except Exception as e:
            logger.warning(f"Error finding interesting spectra fallback: {e}")
            return None
    
    def _rank_interesting_spectra(self, positive_indices, positive_mask):
        """Rank interesting spectra by various criteria."""
        try:
            import numpy as np
            
            # Try to get classification probabilities for ranking
            if hasattr(self, 'classification_results') and 'probabilities' in self.classification_results:
                probs = self.classification_results['probabilities']
                if probs is not None and len(probs) == len(positive_mask):
                    # Get probabilities for positive class
                    positive_probs = probs[positive_mask]
                    if len(positive_probs.shape) > 1:
                        positive_probs = positive_probs[:, 1]  # Assume positive class is index 1
                    
                    # Sort by probability and take highest confidence
                    sorted_indices = np.argsort(positive_probs)[-len(positive_indices):][::-1]
                    return positive_indices[sorted_indices]
            
            # Try ranking by PCA distance from center
            if hasattr(self, 'pca_results') and self.pca_results is not None:
                pca_transformed = self.pca_results.get('components', self.pca_results.get('transformed_data'))
                if pca_transformed is not None:
                    center = np.mean(pca_transformed[:, :2], axis=0)
                    distances = np.linalg.norm(pca_transformed[positive_indices, :2] - center, axis=1)
                    sorted_indices = np.argsort(distances)[::-1]  # Highest distances first
                    return positive_indices[sorted_indices]
            
            # Try ranking by NMF contribution
            if hasattr(self, 'nmf_results') and self.nmf_results is not None:
                nmf_transformed = self.nmf_results.get('components', self.nmf_results.get('transformed_data'))
                if nmf_transformed is not None:
                    max_contributions = np.max(nmf_transformed[positive_indices], axis=1)
                    sorted_indices = np.argsort(max_contributions)[::-1]  # Highest contributions first
                    return positive_indices[sorted_indices]
            
            # Default: return as-is
            return positive_indices
            
        except Exception as e:
            logger.warning(f"Error ranking interesting spectra: {e}")
            return positive_indices
    
    def _plot_component_statistics(self, ax):
        """Plot focused analysis statistics with key metrics."""
        try:
            # Collect key statistics from all analyses
            stats_text = []
            
            # Data Overview
            if hasattr(self, 'map_data') and self.map_data is not None:
                n_spectra = len(self.map_data.spectra)
                stats_text.append("=== DATA OVERVIEW ===")
                stats_text.append(f"   Total spectra: {n_spectra}")
                stats_text.append("")
            
            # PCA Statistics (focused on key info)
            if hasattr(self, 'pca_results') and self.pca_results is not None:
                pca_variance = self.pca_results['explained_variance_ratio'][:3]  # Top 3 components
                stats_text.append("=== PCA ANALYSIS ===")
                stats_text.append(f"   PC1: {pca_variance[0]:.1%}")
                stats_text.append(f"   PC2: {pca_variance[1]:.1%}")
                if len(pca_variance) > 2:
                    stats_text.append(f"   PC3: {pca_variance[2]:.1%}")
                total_var = np.sum(pca_variance)
                stats_text.append(f"   Total (3 PC): {total_var:.1%}")
                stats_text.append("")
            
            # NMF Statistics
            if hasattr(self, 'nmf_results') and self.nmf_results is not None:
                n_components = self.nmf_results.get('n_components', 0)
                reconstruction_error = self.nmf_results.get('reconstruction_error', 0)
                stats_text.append("=== NMF DECOMPOSITION ===")
                stats_text.append(f"   Components: {n_components}")
                stats_text.append(f"   Reconstruction: {reconstruction_error:.3f}")
                stats_text.append("")
            
            # ML Classification Results (key finding)
            positive_mask = self._get_positive_groups_mask()
            if positive_mask is not None:
                n_positive = np.sum(positive_mask)
                n_total = len(positive_mask)
                positive_rate = n_positive / n_total * 100 if n_total > 0 else 0
                
                stats_text.append("=== INTERESTING SPECTRA ===")
                stats_text.append(f"   Found: {n_positive}/{n_total}")
                stats_text.append(f"   Rate: {positive_rate:.1f}%")
                
                # Add interpretation
                if positive_rate < 5:
                    stats_text.append(f"   -> Few outliers detected")
                elif positive_rate < 20:
                    stats_text.append(f"   -> Good separation achieved")
                else:
                    stats_text.append(f"   -> Many groups detected")
                
                stats_text.append("")
            
            # Template Fitting (if available)
            if hasattr(self, 'template_manager') and self.template_manager and self.template_manager.templates:
                stats_text.append("=== TEMPLATE FITTING ===")
                stats_text.append(f"   Templates: {len(self.template_manager.templates)}")
                if hasattr(self, 'template_fit_results') and self.template_fit_results:
                    avg_r2 = np.mean([r.get('r_squared', 0) for r in self.template_fit_results.values()])
                    stats_text.append(f"   Avg. fit R²: {avg_r2:.3f}")
                stats_text.append("")
            
            # Display statistics with improved formatting
            if stats_text:
                text_content = '\n'.join(stats_text)
                ax.text(0.05, 0.95, text_content, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", 
                               edgecolor="#dee2e6", alpha=0.9))
            else:
                ax.text(0.5, 0.5, 'No analysis results available\n\nComplete these steps:\n• Run PCA analysis\n• Run NMF analysis\n• Apply ML classification (optional)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            
            ax.set_title('Key Analysis Summary', fontsize=11, fontweight='bold')
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
            
            # For supervised classification, identify the minority class as "positive"
            if self.classification_results.get('type') == 'supervised':
                import numpy as np
                
                # Get class names if available to determine which is the minority class
                class_names = self.classification_results.get('class_names', [])
                
                # Count occurrences of each class
                unique_classes, counts = np.unique(predictions, return_counts=True)
                
                # If we have class names, try to identify "plastic" as the positive class
                if class_names:
                    try:
                        plastic_idx = class_names.index('plastic')
                        # Return mask for plastic class
                        return predictions == plastic_idx
                    except (ValueError, IndexError):
                        # If 'plastic' not found, fall back to minority class detection
                        pass
                
                # Fall back to minority class detection (smallest count)
                minority_class_idx = unique_classes[np.argmin(counts)]
                logger.info(f"Identifying minority class {minority_class_idx} as positive class. Counts: {dict(zip(unique_classes, counts))}")
                return predictions == minority_class_idx
            
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
        elif hasattr(self, 'current_marker_position') and self.current_marker_position is not None:
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
            control_panel = self.get_current_map_control_panel()
            if control_panel and hasattr(self, 'template_fitting_results'):
                # Template fitting has been performed
                total_templates = len(self.template_fitting_results['template_names'])
                total_spectra = len(self.template_fitting_results['coefficients'])
                
                status_text = f"Templates fitted: {total_templates} templates, {total_spectra} spectra"
                logger.info(f"Template status: {status_text}")
            else:
                logger.info("No template fitting results available")
        except Exception as e:
            logger.error(f"Error updating template status: {e}")

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

    def debug_ml_training_features(self):
        """Debug ML training to understand what features the model is learning from."""
        if not hasattr(self, 'ml_data_manager') or not self.ml_data_manager.class_data:
            print("No training data loaded")
            return
        
        if not hasattr(self, 'nmf_analyzer') or self.nmf_analyzer.nmf is None:
            print("No NMF model available")
            return
        
        print("\n=== ML Training Feature Analysis ===")
        
        try:
            # Get training data
            X, y, class_names, common_wavenumbers = self.ml_data_manager.get_training_data(self.preprocessor)
            print(f"Training data loaded: {X.shape[0]} spectra, {X.shape[1]} features")
            print(f"Classes: {class_names}")
            print(f"Class distribution: {np.unique(y, return_counts=True)}")
            
            # Transform with NMF
            X_nmf, feature_type = self.nmf_analyzer.transform_data(X, fallback_to_full=True)
            print(f"\nNMF transformation: {X.shape} -> {X_nmf.shape}")
            print(f"Feature type used: {feature_type}")
            
            if feature_type == 'nmf':
                # Analyze NMF components for each class
                print(f"\n=== NMF Component Analysis by Class ===")
                
                for class_idx, class_name in enumerate(class_names):
                    class_mask = y == class_idx
                    class_spectra_nmf = X_nmf[class_mask]
                    
                    print(f"\nClass '{class_name}' ({np.sum(class_mask)} spectra):")
                    mean_components = np.mean(class_spectra_nmf, axis=0)
                    std_components = np.std(class_spectra_nmf, axis=0)
                    
                    for comp_idx in range(len(mean_components)):
                        print(f"  Component {comp_idx}: mean={mean_components[comp_idx]:.3f}, std={std_components[comp_idx]:.3f}")
                    
                    # Find most discriminative components
                    print(f"  Dominant components: {np.argsort(mean_components)[::-1][:3]}")
                
                # Compare with map NMF
                print(f"\n=== Map NMF Component Analysis ===")
                if hasattr(self, 'nmf_analyzer') and self.nmf_analyzer.components is not None:
                    map_components = self.nmf_analyzer.components
                    print(f"Map has {map_components.shape[0]} spectra, {map_components.shape[1]} NMF components")
                    
                    for comp_idx in range(map_components.shape[1]):
                        comp_values = map_components[:, comp_idx]
                        print(f"  Map Component {comp_idx}: mean={np.mean(comp_values):.3f}, std={np.std(comp_values):.3f}, max={np.max(comp_values):.3f}")
                        
                        # Find pixels with high values for this component
                        high_threshold = np.percentile(comp_values, 95)
                        high_pixels = np.sum(comp_values > high_threshold)
                        print(f"    High-value pixels (>95th percentile): {high_pixels}/{len(comp_values)} ({100*high_pixels/len(comp_values):.1f}%)")
                
                # Check if training and map NMF ranges are compatible
                print(f"\n=== Training vs Map NMF Range Comparison ===")
                training_ranges = [(np.min(X_nmf[:, i]), np.max(X_nmf[:, i])) for i in range(X_nmf.shape[1])]
                map_ranges = [(np.min(map_components[:, i]), np.max(map_components[:, i])) for i in range(map_components.shape[1])]
                
                for comp_idx in range(len(training_ranges)):
                    train_min, train_max = training_ranges[comp_idx]
                    map_min, map_max = map_ranges[comp_idx]
                    overlap = max(0, min(train_max, map_max) - max(train_min, map_min))
                    train_range = train_max - train_min
                    map_range = map_max - map_min
                    
                    print(f"  Component {comp_idx}:")
                    print(f"    Training range: [{train_min:.3f}, {train_max:.3f}] (width: {train_range:.3f})")
                    print(f"    Map range: [{map_min:.3f}, {map_max:.3f}] (width: {map_range:.3f})")
                    print(f"    Overlap: {overlap:.3f} ({100*overlap/max(train_range,map_range):.1f}% of larger range)")
                    
                    if overlap < 0.1 * max(train_range, map_range):
                        print(f"    ⚠️  WARNING: Poor overlap for component {comp_idx}!")
                
            # Train a simple model and examine feature importance
            if feature_type == 'nmf' and X_nmf.shape[1] > 1:
                print(f"\n=== Feature Importance Analysis ===")
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                
                X_train, X_test, y_train, y_test = train_test_split(X_nmf, y, test_size=0.3, random_state=42, stratify=y)
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                
                importance = rf.feature_importances_
                print(f"Feature importance (NMF components):")
                for comp_idx, imp in enumerate(importance):
                    print(f"  Component {comp_idx}: {imp:.3f}")
                
                # Test predictions
                y_pred = rf.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                print(f"Training accuracy on test set: {accuracy:.3f}")
                
                # Show some prediction examples
                print(f"\nPrediction examples (first 10 test samples):")
                for i in range(min(10, len(y_test))):
                    actual_class = class_names[y_test[i]]
                    pred_class = class_names[y_pred[i]]
                    correct = "✓" if y_test[i] == y_pred[i] else "✗"
                    print(f"  {correct} Actual: {actual_class}, Predicted: {pred_class}")
                    print(f"    NMF features: {X_test[i]}")
            
            print(f"\n=== Recommendations ===")
            if feature_type != 'nmf':
                print("• NMF transformation failed - check if NMF was run successfully")
            else:
                # Check for potential issues
                issues = []
                
                # Check for very small NMF values
                if np.max(X_nmf) < 0.1:
                    issues.append("NMF component values are very small - may need different scaling")
                
                # Check for component dominance
                component_variances = np.var(X_nmf, axis=0)
                if np.max(component_variances) > 10 * np.min(component_variances):
                    dominant_comp = np.argmax(component_variances)
                    issues.append(f"Component {dominant_comp} dominates variance - may need different n_components")
                
                # Check class separability in NMF space
                if len(class_names) == 2:
                    class0_mean = np.mean(X_nmf[y == 0], axis=0)
                    class1_mean = np.mean(X_nmf[y == 1], axis=0)
                    separation = np.linalg.norm(class0_mean - class1_mean)
                    combined_std = np.mean([np.std(X_nmf[y == 0], axis=0), np.std(X_nmf[y == 1], axis=0)])
                    separability = separation / np.mean(combined_std)
                    
                    if separability < 1.0:
                        issues.append(f"Poor class separation in NMF space (separability: {separability:.2f})")
                
                if issues:
                    print("• Issues detected:")
                    for issue in issues:
                        print(f"  - {issue}")
                    print("• Suggestions:")
                    print("  - Try different number of NMF components")
                    print("  - Check if your training spectra have the expected spectral features")
                    print("  - Verify that NMF is capturing meaningful chemical information")
                    print("  - Consider using full spectrum features instead of NMF")
                else:
                    print("• No obvious issues detected with NMF features")
                    print("• The ML model should be learning meaningful patterns")
                    print("• Check if the background spectra have unexpected similarity to training features")
                
        except Exception as e:
            print(f"Error during feature analysis: {e}")
            import traceback
            traceback.print_exc()

    def show_ml_feature_info(self):
        """Show detailed information about ML features and training data."""
        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
            
            dialog = QDialog(self)
            dialog.setWindowTitle("ML Feature Information")
            dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Create text area for information
            info_text = QTextEdit()
            info_text.setReadOnly(True)
            info_text.setFont(QFont("Courier", 10))  # Monospace font for better formatting
            
            # Gather comprehensive ML information
            info_content = "=== ML TRAINING DIAGNOSTIC INFORMATION ===\n\n"
            
            # 1. Training Data Information
            info_content += "1. TRAINING DATA STATUS:\n"
            if hasattr(self, 'ml_data_manager') and self.ml_data_manager.class_data:
                class_info = self.ml_data_manager.get_class_info()
                info_content += f"   ✓ Training data loaded\n"
                info_content += f"   Classes: {class_info['n_classes']}\n"
                for class_name, count in class_info['class_counts'].items():
                    info_content += f"     - {class_name}: {count} spectra\n"
            else:
                info_content += "   ❌ No training data loaded\n"
            
            # 2. Model Information
            info_content += "\n2. MODEL STATUS:\n"
            if hasattr(self, 'supervised_analyzer') and self.supervised_analyzer.model is not None:
                info_content += f"   ✓ Supervised model trained\n"
                info_content += f"   Model type: {getattr(self.supervised_analyzer, 'model_type', 'Unknown')}\n"
                
                # Training performance
                if hasattr(self, 'ml_results') and self.ml_results.get('type') == 'supervised':
                    info_content += f"   Training accuracy: {self.ml_results.get('accuracy', 'Unknown'):.3f}\n"
            else:
                info_content += "   ❌ No supervised model trained\n"
            
            # 3. Classification Results
            info_content += "\n3. CLASSIFICATION RESULTS:\n"
            if hasattr(self, 'classification_results') and self.classification_results:
                predictions = self.classification_results.get('predictions', [])
                if len(predictions) > 0:
                    unique_preds, counts = np.unique(predictions, return_counts=True)
                    info_content += f"   ✓ Map classified - {len(predictions)} spectra\n"
                    
                    class_names = self.classification_results.get('class_names', [])
                    for pred_val, count in zip(unique_preds, counts):
                        if pred_val < len(class_names):
                            class_name = class_names[pred_val]
                            percentage = (count / len(predictions)) * 100
                            info_content += f"     - {class_name}: {count} spectra ({percentage:.1f}%)\n"
                else:
                    info_content += "   ❌ No classification results\n"
            else:
                info_content += "   ❌ No classification results\n"
            
            # 4. Recommendations
            info_content += "\n4. RECOMMENDATIONS:\n"
            if not hasattr(self, 'ml_data_manager') or not self.ml_data_manager.class_data:
                info_content += "   🔧 Load training data first using 'Load Training Data Folders'\n"
            elif not hasattr(self, 'supervised_analyzer') or self.supervised_analyzer.model is None:
                info_content += "   🔧 Train a supervised model using the ML tab\n"
            elif not hasattr(self, 'classification_results') or not self.classification_results:
                info_content += "   🔧 Apply the trained model to the map using 'Apply Model to Map'\n"
            else:
                info_content += "   ✓ ML pipeline appears to be working\n"
            
            info_text.setPlainText(info_content)
            layout.addWidget(info_text)
            
            # Add close button
            close_btn = StandardButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error showing ML feature info:\n{str(e)}")
            logger.error(f"Error in show_ml_feature_info: {e}")

    def calculate_template_statistics(self):
        """Calculate comprehensive template fitting statistics."""
        if not hasattr(self, 'template_fitting_results') or not self.template_fitting_results:
            return None
            
        import numpy as np
        from collections import defaultdict
        
        try:
            coefficients = self.template_fitting_results['coefficients']
            r_squared_values = self.template_fitting_results['r_squared']
            template_names = self.template_fitting_results['template_names']
            use_baseline = self.template_fitting_results.get('use_baseline', False)
            
            # Adjust for baseline if present
            n_templates = len(template_names)
            
            # Initialize statistics containers
            stats = {
                'template_names': template_names,
                'total_spectra': len(coefficients),
                'mean_r_squared': np.mean(list(r_squared_values.values())),
                'relative_contributions': {},  # Mean relative contribution per template
                'dominance_count': defaultdict(int),  # How often each template dominates
                'coverage_stats': {},  # Coverage analysis
                'template_correlations': {},  # Template interaction analysis
                'quality_by_template': {},  # R² when template dominates
                'spatial_distribution': {}  # Spatial characteristics
            }
            
            # Extract coefficient arrays
            coeff_arrays = []
            positions = []
            r_squared_list = []
            
            for pos, coeffs in coefficients.items():
                # Only consider template coefficients (exclude baseline if present)
                template_coeffs = coeffs[:n_templates]
                coeff_arrays.append(template_coeffs)
                positions.append(pos)
                r_squared_list.append(r_squared_values[pos])
            
            coeff_matrix = np.array(coeff_arrays)  # Shape: (n_spectra, n_templates)
            r_squared_array = np.array(r_squared_list)
            
            # 1. Relative Contributions (normalize each spectrum to sum to 100%)
            spectrum_totals = np.sum(coeff_matrix, axis=1)
            
            # Handle zero totals (avoid division by zero)
            nonzero_mask = spectrum_totals > 1e-10
            relative_contributions = np.zeros_like(coeff_matrix)
            relative_contributions[nonzero_mask] = (
                coeff_matrix[nonzero_mask] / spectrum_totals[nonzero_mask, np.newaxis]
            )
            
            # Mean relative contribution per template
            for i, name in enumerate(template_names):
                stats['relative_contributions'][name] = {
                    'mean_percent': np.mean(relative_contributions[:, i]) * 100,
                    'std_percent': np.std(relative_contributions[:, i]) * 100,
                    'max_percent': np.max(relative_contributions[:, i]) * 100
                }
            
            # 2. Dominance Analysis
            # Find dominant template for each spectrum (with confidence threshold)
            dominant_indices = np.argmax(relative_contributions, axis=1)
            dominant_strengths = np.max(relative_contributions, axis=1)
            
            # Only count as "dominant" if contribution > 30% and significantly higher than others
            confidence_threshold = 0.30
            significance_margin = 0.10  # Must be 10% higher than second best
            
            for i, (dom_idx, dom_strength) in enumerate(zip(dominant_indices, dominant_strengths)):
                if dom_strength > confidence_threshold:
                    # Check if significantly higher than second best
                    sorted_contribs = np.sort(relative_contributions[i])[::-1]
                    if len(sorted_contribs) > 1:
                        margin = sorted_contribs[0] - sorted_contribs[1]
                        if margin > significance_margin:
                            stats['dominance_count'][template_names[dom_idx]] += 1
            
            # 3. Coverage Statistics (where each template significantly contributes)
            significance_threshold = 0.15  # 15% contribution threshold
            
            for i, name in enumerate(template_names):
                significant_mask = relative_contributions[:, i] > significance_threshold
                stats['coverage_stats'][name] = {
                    'coverage_percent': np.sum(significant_mask) / len(coefficients) * 100,
                    'mean_contribution_when_present': np.mean(relative_contributions[significant_mask, i]) * 100 if np.any(significant_mask) else 0,
                    'max_contribution': np.max(relative_contributions[:, i]) * 100
                }
            
            # 4. Quality Metrics by Template
            for i, name in enumerate(template_names):
                # R² values when this template is dominant
                dominant_mask = (dominant_indices == i) & (dominant_strengths > confidence_threshold)
                if np.any(dominant_mask):
                    stats['quality_by_template'][name] = {
                        'mean_r_squared_when_dominant': np.mean(r_squared_array[dominant_mask]),
                        'count_dominant': np.sum(dominant_mask)
                    }
                else:
                    stats['quality_by_template'][name] = {
                        'mean_r_squared_when_dominant': 0,
                        'count_dominant': 0
                    }
            
            # 5. Template Correlation Analysis
            template_corr_matrix = np.corrcoef(coeff_matrix.T)
            for i, name1 in enumerate(template_names):
                correlations = {}
                for j, name2 in enumerate(template_names):
                    if i != j:
                        correlations[name2] = template_corr_matrix[i, j]
                stats['template_correlations'][name1] = correlations
            
            # 6. Spatial Distribution Analysis (if we have position info)
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                
                for i, name in enumerate(template_names):
                    # Calculate spatial clustering of high-contribution areas
                    high_contrib_mask = relative_contributions[:, i] > 0.20  # 20% threshold
                    if np.any(high_contrib_mask):
                        high_contrib_x = np.array([x_coords[j] for j in range(len(positions)) if high_contrib_mask[j]])
                        high_contrib_y = np.array([y_coords[j] for j in range(len(positions)) if high_contrib_mask[j]])
                        
                        stats['spatial_distribution'][name] = {
                            'spatial_extent_x': np.max(high_contrib_x) - np.min(high_contrib_x) if len(high_contrib_x) > 1 else 0,
                            'spatial_extent_y': np.max(high_contrib_y) - np.min(high_contrib_y) if len(high_contrib_y) > 1 else 0,
                            'centroid_x': np.mean(high_contrib_x),
                            'centroid_y': np.mean(high_contrib_y),
                            'high_contrib_count': len(high_contrib_x)
                        }
                    else:
                        stats['spatial_distribution'][name] = {
                            'spatial_extent_x': 0, 'spatial_extent_y': 0,
                            'centroid_x': 0, 'centroid_y': 0,
                            'high_contrib_count': 0
                        }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating template statistics: {e}")
            return None

    def format_template_statistics(self, stats):
        """Format template statistics for display."""
        if not stats:
            return "No template fitting statistics available."
        
        output = []
        output.append("=" * 60)
        output.append("TEMPLATE FITTING STATISTICS")
        output.append("=" * 60)
        output.append(f"Total Spectra Analyzed: {stats['total_spectra']}")
        output.append(f"Overall Fit Quality (R²): {stats['mean_r_squared']:.3f}")
        output.append("")
        
        # 1. Relative Contributions Summary
        output.append("1. RELATIVE CONTRIBUTION ANALYSIS")
        output.append("-" * 40)
        for name, contrib in stats['relative_contributions'].items():
            output.append(f"{name}:")
            output.append(f"  Mean: {contrib['mean_percent']:.1f}% ± {contrib['std_percent']:.1f}%")
            output.append(f"  Max:  {contrib['max_percent']:.1f}%")
        output.append("")
        
        # 2. Dominance Analysis
        output.append("2. DOMINANCE ANALYSIS")
        output.append("-" * 40)
        total_dominant = sum(stats['dominance_count'].values())
        for name in stats['template_names']:
            count = stats['dominance_count'][name]
            percentage = (count / stats['total_spectra']) * 100 if stats['total_spectra'] > 0 else 0
            output.append(f"{name}: {count} spectra ({percentage:.1f}% of map)")
        
        if total_dominant < stats['total_spectra']:
            mixed_count = stats['total_spectra'] - total_dominant
            mixed_percent = (mixed_count / stats['total_spectra']) * 100
            output.append(f"Mixed/Unclear: {mixed_count} spectra ({mixed_percent:.1f}% of map)")
        output.append("")
        
        # 3. Coverage Statistics
        output.append("3. COVERAGE ANALYSIS (>15% contribution)")
        output.append("-" * 40)
        for name, coverage in stats['coverage_stats'].items():
            output.append(f"{name}:")
            output.append(f"  Coverage: {coverage['coverage_percent']:.1f}% of map")
            output.append(f"  Avg when present: {coverage['mean_contribution_when_present']:.1f}%")
        output.append("")
        
        # 4. Quality by Template
        output.append("4. FIT QUALITY BY TEMPLATE")
        output.append("-" * 40)
        for name, quality in stats['quality_by_template'].items():
            if quality['count_dominant'] > 0:
                output.append(f"{name}: R² = {quality['mean_r_squared_when_dominant']:.3f} "
                             f"(from {quality['count_dominant']} dominant regions)")
            else:
                output.append(f"{name}: No dominant regions")
        output.append("")
        
        return "\n".join(output)

    def show_detailed_template_statistics(self):
        """Show detailed template statistics in a dialog."""
        stats = self.calculate_template_statistics()
        if not stats:
            QMessageBox.warning(self, "No Statistics", "No template fitting results available.")
            return
        
        # Create detailed statistics dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Detailed Template Statistics")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Text area for statistics
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setFont(QFont("Courier New", 10))
        
        # Generate comprehensive report
        detailed_text = self.format_detailed_template_statistics(stats)
        text_area.setPlainText(detailed_text)
        
        layout.addWidget(text_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export to File")
        export_btn.clicked.connect(lambda: self.export_template_statistics(stats))
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()

    def format_detailed_template_statistics(self, stats):
        """Format detailed template statistics for the dialog."""
        if not stats:
            return "No template fitting statistics available."
        
        output = []
        output.append("=" * 80)
        output.append("COMPREHENSIVE TEMPLATE FITTING ANALYSIS")
        output.append("=" * 80)
        output.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Total Spectra: {stats['total_spectra']}")
        output.append(f"Number of Templates: {len(stats['template_names'])}")
        output.append(f"Overall Fit Quality (R²): {stats['mean_r_squared']:.3f}")
        output.append("")
        
        # Detailed breakdown for each template
        for i, name in enumerate(stats['template_names']):
            output.append("=" * 80)
            output.append(f"TEMPLATE: {name}")
            output.append("=" * 80)
            
            # Contribution statistics
            contrib = stats['relative_contributions'][name]
            output.append("Contribution Statistics:")
            output.append(f"  Mean contribution: {contrib['mean_percent']:.2f}%")
            output.append(f"  Standard deviation: {contrib['std_percent']:.2f}%")
            output.append(f"  Maximum contribution: {contrib['max_percent']:.2f}%")
            output.append("")
            
            # Dominance and coverage
            dom_count = stats['dominance_count'][name]
            dom_percent = (dom_count / stats['total_spectra']) * 100 if stats['total_spectra'] > 0 else 0
            output.append("Dominance Analysis:")
            output.append(f"  Dominant regions: {dom_count} ({dom_percent:.1f}% of map)")
            output.append("")
            
            coverage = stats['coverage_stats'][name]
            output.append("Coverage Analysis:")
            output.append(f"  Significant presence: {coverage['coverage_percent']:.1f}% of map")
            output.append(f"  Average when present: {coverage['mean_contribution_when_present']:.1f}%")
            output.append("")
            
            # Quality metrics
            quality = stats['quality_by_template'][name]
            if quality['count_dominant'] > 0:
                output.append("Quality Metrics:")
                output.append(f"  R² when dominant: {quality['mean_r_squared_when_dominant']:.3f}")
                output.append(f"  Based on {quality['count_dominant']} dominant regions")
            else:
                output.append("Quality Metrics: No dominant regions for analysis")
            output.append("")
            
            # Spatial distribution
            if name in stats['spatial_distribution']:
                spatial = stats['spatial_distribution'][name]
                output.append("Spatial Distribution:")
                output.append(f"  High-contribution regions: {spatial['high_contrib_count']}")
                if spatial['high_contrib_count'] > 0:
                    output.append(f"  Spatial extent (X): {spatial['spatial_extent_x']:.2f}")
                    output.append(f"  Spatial extent (Y): {spatial['spatial_extent_y']:.2f}")
                    output.append(f"  Centroid: ({spatial['centroid_x']:.2f}, {spatial['centroid_y']:.2f})")
                output.append("")
            
            # Template correlations
            if name in stats['template_correlations']:
                correlations = stats['template_correlations'][name]
                output.append("Template Interactions:")
                for other_name, corr in correlations.items():
                    if abs(corr) > 0.3:  # Only show significant correlations
                        corr_type = "positive" if corr > 0 else "negative"
                        output.append(f"  {corr_type} correlation with {other_name}: {corr:.3f}")
                output.append("")
            
            output.append("")
        
        # Summary recommendations
        output.append("=" * 80)
        output.append("ANALYSIS SUMMARY & RECOMMENDATIONS")
        output.append("=" * 80)
        
        # Identify most important templates
        sorted_by_dominance = sorted(stats['dominance_count'].items(), key=lambda x: x[1], reverse=True)
        output.append("Template Importance Ranking (by dominance):")
        for i, (name, count) in enumerate(sorted_by_dominance[:5], 1):
            percent = (count / stats['total_spectra']) * 100 if stats['total_spectra'] > 0 else 0
            output.append(f"  {i}. {name}: {count} regions ({percent:.1f}%)")
        output.append("")
        
        # Coverage analysis
        sorted_by_coverage = sorted(
            [(name, data['coverage_percent']) for name, data in stats['coverage_stats'].items()],
            key=lambda x: x[1], reverse=True
        )
        output.append("Template Coverage Ranking:")
        for i, (name, coverage) in enumerate(sorted_by_coverage[:5], 1):
            output.append(f"  {i}. {name}: {coverage:.1f}% coverage")
        output.append("")
        
        return "\n".join(output)

    def export_template_statistics(self, stats):
        """Export template statistics to a file."""
        if not stats:
            QMessageBox.warning(self, "No Statistics", "No template fitting results available.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Template Statistics", 
            f"template_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                detailed_text = self.format_detailed_template_statistics(stats)
                with open(filename, 'w') as f:
                    f.write(detailed_text)
                QMessageBox.information(self, "Export Complete", f"Statistics exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export statistics:\n{str(e)}")

    def update_template_statistics_display(self):
        """Update the template statistics display in the control panel."""
        try:
            control_panel = self.get_current_template_control_panel()
            if control_panel:
                stats = self.calculate_template_statistics()
                if stats:
                    # Format basic statistics for display
                    basic_stats = self.format_template_statistics(stats)
                    control_panel.update_statistics_display(basic_stats)
                else:
                    control_panel.hide_statistics()
        except Exception as e:
            logger.error(f"Error updating template statistics display: {e}")

    def analyze_template_chemical_validity(self):
        """
        Analyze template assignments using spectroscopic features to distinguish
        between mathematical fitting and true chemical presence.
        """
        if not hasattr(self, 'template_fitting_results') or not self.template_fitting_results:
            return None
            
        import numpy as np
        from collections import defaultdict
        
        try:
            coefficients = self.template_fitting_results['coefficients']
            template_names = self.template_fitting_results['template_names']
            n_templates = len(template_names)
            
            # Initialize analysis containers
            analysis = {
                'template_names': template_names,
                'peak_based_analysis': {},
                'residual_analysis': {},
                'spectral_quality': {},
                'chemical_likelihood': {},
                'recommended_thresholds': {}
            }
            
            # Get template spectra for reference
            template_spectra = {}
            if hasattr(self, 'template_manager'):
                for i, name in enumerate(template_names):
                    if i < len(self.template_manager.templates):
                        template = self.template_manager.templates[i]
                        template_spectra[name] = {
                            'wavenumbers': template.wavenumbers,
                            'intensities': (template.processed_intensities 
                                          if template.processed_intensities is not None 
                                          else template.intensities)
                        }
            
            # Analyze each template
            for template_name in template_names:
                analysis['peak_based_analysis'][template_name] = self._analyze_template_peaks(
                    template_name, template_spectra.get(template_name), coefficients
                )
                analysis['residual_analysis'][template_name] = self._analyze_template_residuals(
                    template_name, coefficients
                )
                analysis['spectral_quality'][template_name] = self._analyze_spectral_quality(
                    template_name, coefficients
                )
            
            # Calculate chemical likelihood scores
            for template_name in template_names:
                peak_score = analysis['peak_based_analysis'][template_name].get('peak_presence_score', 0)
                residual_score = analysis['residual_analysis'][template_name].get('improvement_score', 0)
                quality_score = analysis['spectral_quality'][template_name].get('snr_score', 0)
                
                # Weighted combination (adjust weights based on your priorities)
                chemical_likelihood = (0.5 * peak_score + 0.3 * residual_score + 0.2 * quality_score)
                analysis['chemical_likelihood'][template_name] = chemical_likelihood
                
                # Recommend thresholds based on analysis
                if peak_score > 0.7 and quality_score > 0.6:
                    threshold = 0.10  # Lower threshold for high-confidence peaks
                elif peak_score > 0.4:
                    threshold = 0.20  # Medium threshold for moderate peaks
                else:
                    threshold = 0.40  # Higher threshold for background-like templates
                    
                analysis['recommended_thresholds'][template_name] = threshold
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in chemical validity analysis: {e}")
            return None

    def _analyze_template_peaks(self, template_name, template_spectrum, coefficients):
        """Analyze peak-based features for a template."""
        import numpy as np
        from scipy.signal import find_peaks
        
        try:
            peak_analysis = {
                'has_peaks': False,
                'peak_count': 0,
                'peak_prominence': 0,
                'peak_presence_score': 0,
                'characteristic_peaks': []
            }
            
            if template_spectrum is None:
                return peak_analysis
                
            wavenumbers = template_spectrum['wavenumbers']
            intensities = template_spectrum['intensities']
            
            # Find peaks in template spectrum
            peaks, properties = find_peaks(
                intensities, 
                prominence=np.std(intensities) * 0.5,  # Adaptive prominence threshold
                width=2
            )
            
            if len(peaks) > 0:
                peak_analysis['has_peaks'] = True
                peak_analysis['peak_count'] = len(peaks)
                peak_analysis['peak_prominence'] = np.mean(properties['prominences'])
                
                # Store characteristic peak positions
                for peak_idx in peaks:
                    peak_analysis['characteristic_peaks'].append({
                        'wavenumber': wavenumbers[peak_idx],
                        'intensity': intensities[peak_idx],
                        'prominence': properties['prominences'][np.where(peaks == peak_idx)[0][0]]
                    })
                
                # Calculate peak presence score (0-1)
                # Higher score for more/sharper peaks
                prominence_score = min(1.0, peak_analysis['peak_prominence'] / (np.max(intensities) * 0.1))
                count_score = min(1.0, len(peaks) / 10.0)  # Max score at 10 peaks
                peak_analysis['peak_presence_score'] = 0.7 * prominence_score + 0.3 * count_score
            
            return peak_analysis
            
        except Exception as e:
            logger.error(f"Error in peak analysis for {template_name}: {e}")
            return {'has_peaks': False, 'peak_count': 0, 'peak_prominence': 0, 'peak_presence_score': 0}

    def _analyze_template_residuals(self, template_name, coefficients):
        """Analyze fitting residuals to assess template necessity."""
        import numpy as np
        
        try:
            residual_analysis = {
                'avg_improvement': 0,
                'improvement_score': 0,
                'regions_with_improvement': 0
            }
            
            # This would require refitting without the template to compare residuals
            # For now, we'll use a proxy based on template contribution patterns
            template_idx = self.template_fitting_results['template_names'].index(template_name)
            
            contributions = []
            for pos, coeffs in coefficients.items():
                if template_idx < len(coeffs):
                    contributions.append(coeffs[template_idx])
            
            if contributions:
                contributions = np.array(contributions)
                # Score based on distribution - good templates should have some high contributions
                # and clear low contributions, not uniform medium contributions
                contrib_std = np.std(contributions)
                contrib_max = np.max(contributions)
                
                if contrib_max > 0:
                    # Higher score for templates with clear high-contribution regions
                    improvement_score = min(1.0, (contrib_std / contrib_max) * 2.0)
                    residual_analysis['improvement_score'] = improvement_score
                    residual_analysis['regions_with_improvement'] = np.sum(contributions > np.mean(contributions))
            
            return residual_analysis
            
        except Exception as e:
            logger.error(f"Error in residual analysis for {template_name}: {e}")
            return {'avg_improvement': 0, 'improvement_score': 0, 'regions_with_improvement': 0}

    def _analyze_spectral_quality(self, template_name, coefficients):
        """Analyze spectral quality metrics for template assignments."""
        import numpy as np
        
        try:
            quality_analysis = {
                'avg_snr': 0,
                'snr_score': 0,
                'baseline_stability': 0
            }
            
            # Get regions where this template contributes significantly
            template_idx = self.template_fitting_results['template_names'].index(template_name)
            
            snr_values = []
            for spectrum in self.map_data.spectra.values():
                pos_key = (spectrum.x_pos, spectrum.y_pos)
                if pos_key in coefficients:
                    coeffs = coefficients[pos_key]
                    if template_idx < len(coeffs) and coeffs[template_idx] > 0.1:  # Significant contribution
                        # Calculate SNR for this spectrum
                        intensities = (spectrum.processed_intensities 
                                     if self.use_processed and spectrum.processed_intensities is not None
                                     else spectrum.intensities)
                        
                        if intensities is not None and len(intensities) > 10:
                            signal = np.max(intensities)
                            noise = np.std(intensities[:10])  # Estimate noise from first 10 points
                            if noise > 0:
                                snr = signal / noise
                                snr_values.append(snr)
            
            if snr_values:
                quality_analysis['avg_snr'] = np.mean(snr_values)
                # Normalize SNR score (assuming good SNR > 10)
                quality_analysis['snr_score'] = min(1.0, np.mean(snr_values) / 20.0)
            
            return quality_analysis
            
        except Exception as e:
            logger.error(f"Error in quality analysis for {template_name}: {e}")
            return {'avg_snr': 0, 'snr_score': 0, 'baseline_stability': 0}

    def show_chemical_validity_analysis(self):
        """Show detailed chemical validity analysis in a dialog."""
        analysis = self.analyze_template_chemical_validity()
        if not analysis:
            QMessageBox.warning(self, "No Analysis", "No template fitting results available for chemical analysis.")
            return
        
        # Create detailed analysis dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Chemical Validity Analysis")
        dialog.setModal(True)
        dialog.resize(900, 700)
        
        layout = QVBoxLayout(dialog)
        
        # Text area for analysis
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setFont(QFont("Courier New", 10))
        
        # Generate comprehensive report
        detailed_text = self.format_chemical_validity_analysis(analysis)
        text_area.setPlainText(detailed_text)
        
        layout.addWidget(text_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Analysis")
        export_btn.clicked.connect(lambda: self.export_chemical_analysis(analysis))
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()

    def format_chemical_validity_analysis(self, analysis):
        """Format chemical validity analysis for display."""
        if not analysis:
            return "No chemical validity analysis available."
        
        output = []
        output.append("=" * 80)
        output.append("CHEMICAL VALIDITY ANALYSIS")
        output.append("=" * 80)
        output.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        
        output.append("SUMMARY - Chemical Likelihood Scores (0.0 = background-like, 1.0 = peak-rich)")
        output.append("-" * 70)
        
        # Sort templates by chemical likelihood
        likelihood_sorted = sorted(
            [(name, analysis['chemical_likelihood'][name]) for name in analysis['template_names']],
            key=lambda x: x[1], reverse=True
        )
        
        for name, likelihood in likelihood_sorted:
            recommended_threshold = analysis['recommended_thresholds'][name]
            output.append(f"{name[:50]:<50} {likelihood:.3f} (threshold: {recommended_threshold:.2f})")
        output.append("")
        
        # Detailed analysis for each template
        for name in analysis['template_names']:
            output.append("=" * 80)
            output.append(f"TEMPLATE: {name}")
            output.append("=" * 80)
            
            # Peak analysis
            peak_data = analysis['peak_based_analysis'][name]
            output.append("Peak Analysis:")
            output.append(f"  Has distinct peaks: {'Yes' if peak_data['has_peaks'] else 'No'}")
            output.append(f"  Peak count: {peak_data['peak_count']}")
            output.append(f"  Average prominence: {peak_data['peak_prominence']:.3f}")
            output.append(f"  Peak presence score: {peak_data['peak_presence_score']:.3f}")
            
            if peak_data['characteristic_peaks']:
                output.append("  Characteristic peaks:")
                for peak in peak_data['characteristic_peaks'][:5]:  # Show top 5
                    output.append(f"    {peak['wavenumber']:.1f} cm⁻¹ (prominence: {peak['prominence']:.3f})")
            output.append("")
            
            # Residual analysis
            residual_data = analysis['residual_analysis'][name]
            output.append("Fitting Quality:")
            output.append(f"  Improvement score: {residual_data['improvement_score']:.3f}")
            output.append(f"  Regions with improvement: {residual_data['regions_with_improvement']}")
            output.append("")
            
            # Spectral quality
            quality_data = analysis['spectral_quality'][name]
            output.append("Spectral Quality:")
            output.append(f"  Average SNR: {quality_data['avg_snr']:.1f}")
            output.append(f"  SNR score: {quality_data['snr_score']:.3f}")
            output.append("")
            
            # Overall assessment
            likelihood = analysis['chemical_likelihood'][name]
            threshold = analysis['recommended_thresholds'][name]
            
            output.append("Assessment:")
            if likelihood > 0.7:
                assessment = "HIGH confidence - likely represents real chemical component"
            elif likelihood > 0.4:
                assessment = "MEDIUM confidence - may represent chemical component"
            else:
                assessment = "LOW confidence - likely background or noise fitting"
            
            output.append(f"  Chemical likelihood: {likelihood:.3f}")
            output.append(f"  Recommended threshold: {threshold:.2f}")
            output.append(f"  Assessment: {assessment}")
            output.append("")
        
        # Recommendations
        output.append("=" * 80)
        output.append("RECOMMENDATIONS")
        output.append("=" * 80)
        
        # Find potential over-assignment issues
        for name in analysis['template_names']:
            likelihood = analysis['chemical_likelihood'][name]
            peak_data = analysis['peak_based_analysis'][name]
            
            if likelihood < 0.3 and not any(bg_word in name.lower() for bg_word in ['background', 'baseline', 'steno']):
                output.append(f"⚠️  {name}:")
                output.append(f"   Low chemical likelihood ({likelihood:.3f}) suggests potential over-assignment")
                if not peak_data['has_peaks']:
                    output.append(f"   Consider treating as background template")
                output.append("")
            
            elif likelihood > 0.7 and peak_data['peak_count'] > 3:
                output.append(f"✅ {name}:")
                output.append(f"   High confidence template with {peak_data['peak_count']} characteristic peaks")
                output.append(f"   Current assignment may be accurate")
                output.append("")
        
        output.append("Suggested next steps:")
        output.append("1. For low-likelihood templates with high dominance: add more background templates")
        output.append("2. For peak-rich templates: verify characteristic peaks are present in assigned regions")
        output.append("3. Consider using chemical-informed thresholds for more accurate assignments")
        
        return "\n".join(output)

    def export_chemical_analysis(self, analysis):
        """Export chemical validity analysis to a file."""
        if not analysis:
            QMessageBox.warning(self, "No Analysis", "No chemical validity analysis available.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Chemical Analysis", 
            f"chemical_validity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                detailed_text = self.format_chemical_validity_analysis(analysis)
                with open(filename, 'w') as f:
                    f.write(detailed_text)
                QMessageBox.information(self, "Export Complete", f"Chemical analysis exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export analysis:\n{str(e)}")

    def perform_nmf_guided_template_analysis(self, nmf_component_index=None, nmf_threshold=2.0):
        """
        Perform hybrid analysis using NMF to guide template fitting.
        Only applies template analysis to regions where NMF identifies strong signals.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting perform_nmf_guided_template_analysis with component_index={nmf_component_index}, threshold={nmf_threshold}")
        
        try:
            # Check template results
            logger.info("Checking template fitting results...")
            if not hasattr(self, 'template_fitting_results'):
                logger.error("No template_fitting_results attribute")
                QMessageBox.warning(self, "No Template Results", "Fit templates first before running hybrid analysis.")
                return None
            
            if not self.template_fitting_results:
                logger.error("template_fitting_results is empty/None")
                QMessageBox.warning(self, "No Template Results", "Fit templates first before running hybrid analysis.")
                return None
            
            logger.info("Template results OK")
            
            # Check NMF results  
            logger.info("Checking NMF results...")
            if not hasattr(self, 'nmf_results'):
                logger.error("No nmf_results attribute")
                QMessageBox.warning(self, "No NMF Results", "Run NMF analysis first before running hybrid analysis.")
                return None
                
            if not self.nmf_results:
                logger.error("nmf_results is empty/None")
                QMessageBox.warning(self, "No NMF Results", "Run NMF analysis first before running hybrid analysis.")
                return None
            
            logger.info("NMF results OK")
            
        except Exception as e:
            logger.error(f"Error in initial validation: {e}")
            QMessageBox.critical(self, "Validation Error", f"Error checking prerequisites:\n{str(e)}")
            return None
        
        try:
            import numpy as np
            from collections import defaultdict
            
            logger.info("Starting hybrid analysis...")
            
            # Get NMF components
            nmf_components = self.nmf_results.get('components', [])
            
            # Validate NMF components structure
            if nmf_components is None:
                QMessageBox.warning(self, "No NMF Components", "No NMF components available for analysis.")
                return None
                
            # Handle different component structures
            if hasattr(nmf_components, 'shape'):
                if len(nmf_components.shape) == 0 or (len(nmf_components.shape) == 1 and nmf_components.shape[0] == 0):
                    QMessageBox.warning(self, "No NMF Components", "No NMF components available for analysis.")
                    return None
                if len(nmf_components.shape) == 2:
                    n_components = nmf_components.shape[0]
                    logger.info(f"Found 2D NMF components array with shape {nmf_components.shape}")
                elif len(nmf_components.shape) == 1:
                    n_components = 1  # Single flattened component
                    logger.info(f"Found 1D NMF components array with {len(nmf_components)} elements")
                else:
                    logger.error(f"Unexpected NMF components shape: {nmf_components.shape}")
                    QMessageBox.warning(self, "Invalid Data Structure", "NMF components have unexpected structure.")
                    return None
            elif hasattr(nmf_components, '__len__'):
                if len(nmf_components) == 0:
                    QMessageBox.warning(self, "No NMF Components", "No NMF components available for analysis.")
                    return None
                n_components = len(nmf_components)
                logger.info(f"Found list/array with {n_components} NMF components")
            else:
                QMessageBox.warning(self, "Invalid Data Structure", "NMF components data structure is not recognized.")
                return None
            
            logger.info(f"Using {n_components} NMF components for analysis")
            
            # If no specific component specified, try to auto-detect polypropylene
            logger.info(f"Component index validation: input={nmf_component_index}")
            if nmf_component_index is None:
                logger.info("Auto-detecting polypropylene component...")
                nmf_component_index = self._auto_detect_polypropylene_component()
                logger.info(f"Auto-detection result: {nmf_component_index}")
                if nmf_component_index is None:
                    logger.warning("No component auto-detected")
                    QMessageBox.warning(self, "Component Selection", "Please specify which NMF component contains polypropylene.")
                    return None
            
            # Validate component index
            logger.info(f"Validating component index {nmf_component_index} against {n_components} available components")
            if not isinstance(nmf_component_index, (int, np.integer)):
                logger.error(f"Component index is not an integer: {type(nmf_component_index)}")
                QMessageBox.warning(self, "Invalid Component", "Component index must be an integer.")
                return None
                
            if nmf_component_index < 0 or nmf_component_index >= n_components:
                logger.error(f"Component index {nmf_component_index} out of range [0, {n_components-1}]")
                QMessageBox.warning(self, "Invalid Component", f"Component {nmf_component_index} not available. Only {n_components} components found.")
                return None
            
            logger.info(f"Component index {nmf_component_index} is valid")
            
            # Get NMF component data based on structure
            if hasattr(nmf_components, 'shape') and len(nmf_components.shape) == 2:
                # 2D array: extract the specific component (row)
                nmf_component_data = nmf_components[nmf_component_index]
            else:
                # List or other structure
                nmf_component_data = nmf_components[nmf_component_index]
            
            # Debug information
            logger.info(f"NMF component data type: {type(nmf_component_data)}")
            logger.info(f"NMF component data shape: {np.array(nmf_component_data).shape if hasattr(nmf_component_data, '__len__') else 'scalar'}")
            if hasattr(nmf_component_data, '__len__') and len(nmf_component_data) > 0:
                logger.info(f"First few values: {nmf_component_data[:5] if len(nmf_component_data) > 5 else nmf_component_data}")
                if hasattr(nmf_component_data, '__getitem__'):
                    try:
                        sample_value = nmf_component_data[0]
                        logger.info(f"Sample value type: {type(sample_value)}, shape: {np.array(sample_value).shape if hasattr(sample_value, '__len__') else 'scalar'}")
                    except Exception as e:
                        logger.error(f"Error accessing first value: {e}")
            
            # Validate component data structure
            if not hasattr(nmf_component_data, '__len__'):
                logger.error(f"NMF component data is not iterable: {type(nmf_component_data)}")
                QMessageBox.critical(self, "Data Error", "NMF component data structure is incompatible.")
                return None
                
            if len(nmf_component_data) == 0:
                logger.error("NMF component data is empty")
                QMessageBox.critical(self, "Data Error", "NMF component data is empty.")
                return None
            
            # Initialize hybrid analysis results
            hybrid_results = {
                'nmf_component_index': nmf_component_index,
                'nmf_threshold': nmf_threshold,
                'nmf_guided_assignments': {},
                'cross_validation': {},
                'refined_statistics': {},
                'confidence_regions': {}
            }
            
            # Get map dimensions and positions
            positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Create NMF intensity map for threshold detection
            nmf_map = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            # Fill NMF map with component intensities
            logger.info("Filling NMF map with component intensities...")
            for i, spectrum in enumerate(self.map_data.spectra.values()):
                try:
                    if i < len(nmf_component_data):
                        nmf_intensity = nmf_component_data[i]
                        
                        # Ensure we store scalar values in the map
                        if np.isscalar(nmf_intensity):
                            intensity_value = float(nmf_intensity)
                        elif hasattr(nmf_intensity, '__len__'):
                            # If it's an array, take the mean value
                            intensity_value = float(np.mean(nmf_intensity))
                        else:
                            # Fallback - try to convert to float
                            intensity_value = float(nmf_intensity)
                        
                        # Check bounds before assignment
                        map_y = spectrum.y_pos - y_min
                        map_x = spectrum.x_pos - x_min
                        if 0 <= map_y < nmf_map.shape[0] and 0 <= map_x < nmf_map.shape[1]:
                            nmf_map[map_y, map_x] = intensity_value
                        else:
                            logger.warning(f"Spectrum {i} at position ({spectrum.x_pos}, {spectrum.y_pos}) maps to out-of-bounds indices ({map_x}, {map_y})")
                    else:
                        logger.warning(f"Spectrum {i} has no corresponding NMF component data (only {len(nmf_component_data)} components available)")
                except Exception as e:
                    logger.error(f"Error processing spectrum {i} at position ({spectrum.x_pos}, {spectrum.y_pos}): {e}")
                    continue
            
            # Find regions where NMF component exceeds threshold
            nmf_candidates = nmf_map > nmf_threshold
            candidate_positions = []
            
            logger.info("Finding candidate positions...")
            for spectrum in self.map_data.spectra.values():
                try:
                    map_y = spectrum.y_pos - y_min
                    map_x = spectrum.x_pos - x_min
                    
                    # Safely check bounds first
                    if 0 <= map_y < nmf_candidates.shape[0] and 0 <= map_x < nmf_candidates.shape[1]:
                        # Safely check the boolean value
                        candidate_flag = nmf_candidates[map_y, map_x]
                        
                        # Handle different types of candidate_flag
                        is_candidate = False
                        if np.isscalar(candidate_flag):
                            is_candidate = bool(candidate_flag)
                        elif hasattr(candidate_flag, '__len__'):
                            # Handle array case - use any() for arrays
                            is_candidate = bool(np.any(candidate_flag))
                        else:
                            # Fallback - try to convert to bool
                            is_candidate = bool(candidate_flag)
                        
                        if is_candidate:
                            candidate_positions.append((spectrum.x_pos, spectrum.y_pos))
                    else:
                        logger.warning(f"Spectrum position ({spectrum.x_pos}, {spectrum.y_pos}) -> map indices ({map_x}, {map_y}) out of bounds for map shape {nmf_candidates.shape}")
                        
                except Exception as e:
                    logger.error(f"Error processing spectrum at ({spectrum.x_pos}, {spectrum.y_pos}): {e}")
                    continue
            
            logger.info(f"NMF-guided analysis: Found {len(candidate_positions)} candidate positions above threshold {nmf_threshold}")
            
            # Cross-validate with template results
            logger.info("Getting template results data...")
            try:
                template_coefficients = self.template_fitting_results['coefficients']
                template_names = self.template_fitting_results['template_names']
                logger.info(f"Template coefficients type: {type(template_coefficients)}")
                logger.info(f"Template names: {template_names}")
                logger.info(f"Number of coefficient entries: {len(template_coefficients) if hasattr(template_coefficients, '__len__') else 'N/A'}")
            except Exception as e:
                logger.error(f"Error accessing template results: {e}")
                QMessageBox.critical(self, "Data Error", f"Error accessing template results:\n{str(e)}")
                return None
            
            # Find polypropylene template index using comprehensive detection
            logger.info(f"Looking for polypropylene template in: {template_names}")
            
            # First try simple name matching
            polyprop_template_idx = None
            for i, name in enumerate(template_names):
                logger.info(f"Checking template {i}: '{name}'")
                try:
                    name_lower = str(name).lower()
                    if any(hint in name_lower for hint in ['polyprop', 'pp', 'plastic']):
                        polyprop_template_idx = i
                        logger.info(f"Found polypropylene template by name at index {i}: '{name}'")
                        break
                except Exception as e:
                    logger.error(f"Error processing template name '{name}': {e}")
                    continue
            
            # If name matching fails, use spectral analysis
            if polyprop_template_idx is None:
                logger.info("Name-based detection failed, trying spectral analysis...")
                polyprop_template_idx = self._auto_detect_polypropylene_component()
                
                if polyprop_template_idx is not None:
                    template_name = template_names[polyprop_template_idx]
                    logger.info(f"✅ Auto-detected polypropylene template: '{template_name}' at index {polyprop_template_idx}")
                else:
                    logger.warning("Spectral auto-detection also failed, asking user to select manually...")
                    
                    # Show dialog to let user manually select polypropylene template
                    from PySide6.QtWidgets import QInputDialog
                    template_choices = [f"{i}: {name}" for i, name in enumerate(template_names)]
                    choice, ok = QInputDialog.getItem(
                        self, 
                        "Select Polypropylene Template",
                        "Could not automatically identify polypropylene template.\nPlease select which template represents polypropylene:",
                        template_choices,
                        0,
                        False
                    )
                    
                    if ok and choice:
                        polyprop_template_idx = int(choice.split(':')[0])
                        logger.info(f"User selected polypropylene template: {template_names[polyprop_template_idx]} at index {polyprop_template_idx}")
                    else:
                        logger.error("User cancelled template selection")
                        QMessageBox.warning(self, "Template Selection Cancelled", "Hybrid analysis cancelled - no polypropylene template selected.")
                        return None
            
            # Analyze candidate regions
            logger.info("Analyzing candidate regions...")
            nmf_template_agreement = 0
            nmf_only_detections = 0
            template_only_detections = 0
            hybrid_confident_detections = []
            
            for i, pos in enumerate(candidate_positions):
                try:
                    if i % 100 == 0:  # Log progress every 100 positions
                        logger.info(f"Processing position {i+1}/{len(candidate_positions)}")
                    
                    pos_key = (pos[0], pos[1])
                    
                    # Debug the pos_key type and template_coefficients structure
                    if i == 0:  # Only log for first position to avoid spam
                        logger.info(f"pos_key type: {type(pos_key)}, value: {pos_key}")
                        logger.info(f"template_coefficients keys type: {type(list(template_coefficients.keys())[0]) if template_coefficients else 'empty'}")
                        if template_coefficients:
                            first_key = list(template_coefficients.keys())[0]
                            logger.info(f"First template key: {first_key}, type: {type(first_key)}")
                    
                    # Safe check for key presence
                    try:
                        key_exists = pos_key in template_coefficients
                    except Exception as key_error:
                        logger.error(f"Error checking if {pos_key} in template_coefficients: {key_error}")
                        continue
                    
                    if key_exists:
                        coeffs = template_coefficients[pos_key]
                        
                        # Safe array access
                        if polyprop_template_idx < len(coeffs):
                            # Calculate relative polypropylene contribution
                            coeffs_array = np.array(coeffs[:len(template_names)])
                            total_contrib = np.sum(coeffs_array)
                            
                            # Safe division check
                            if np.isscalar(total_contrib) and total_contrib > 1e-10:
                                pp_relative = coeffs[polyprop_template_idx] / total_contrib
                                
                                # Get NMF intensity at this position
                                spectrum_idx = list(self.map_data.spectra.keys()).index(pos_key) if pos_key in self.map_data.spectra else -1
                                if 0 <= spectrum_idx < len(nmf_component_data):
                                    nmf_intensity = nmf_component_data[spectrum_idx]
                                    # Ensure scalar value
                                    if not np.isscalar(nmf_intensity):
                                        nmf_intensity = float(np.mean(nmf_intensity))
                                else:
                                    nmf_intensity = 0.0
                                
                                # Store hybrid analysis data
                                hybrid_results['nmf_guided_assignments'][pos_key] = {
                                    'nmf_intensity': float(nmf_intensity),
                                    'template_contribution': float(coeffs[polyprop_template_idx]),
                                    'template_relative': float(pp_relative),
                                    'agreement_score': self._calculate_agreement_score(nmf_intensity, pp_relative)
                                }
                                
                                # Check for agreement (both methods detect significant signal)
                                if np.isscalar(pp_relative) and pp_relative > 0.3:  # Template says polypropylene dominant
                                    nmf_template_agreement += 1
                                    hybrid_confident_detections.append(pos_key)
                except Exception as e:
                    logger.error(f"Error analyzing position {pos}: {e}")
                    continue
            
            # Calculate refined statistics
            logger.info("Calculating refined statistics...")
            total_spectra = len(self.map_data.spectra)
            hybrid_results['refined_statistics'] = {
                'total_spectra': total_spectra,
                'nmf_candidates': len(candidate_positions),
                'nmf_candidate_percentage': (len(candidate_positions) / total_spectra) * 100 if total_spectra > 0 else 0,
                'agreement_count': nmf_template_agreement,
                'agreement_percentage': (nmf_template_agreement / total_spectra) * 100 if total_spectra > 0 else 0,
                'confident_detections': len(hybrid_confident_detections),
                'confident_percentage': (len(hybrid_confident_detections) / total_spectra) * 100 if total_spectra > 0 else 0
            }
            
            # Generate confidence regions based on agreement strength
            high_confidence = []
            medium_confidence = []
            low_confidence = []
            
            for pos_key, data in hybrid_results['nmf_guided_assignments'].items():
                try:
                    agreement = data['agreement_score']
                    # Ensure scalar comparison
                    if np.isscalar(agreement):
                        if agreement > 0.8:
                            high_confidence.append(pos_key)
                        elif agreement > 0.5:
                            medium_confidence.append(pos_key)
                        else:
                            low_confidence.append(pos_key)
                    else:
                        # Handle array case
                        agreement_val = float(np.mean(agreement))
                        if agreement_val > 0.8:
                            high_confidence.append(pos_key)
                        elif agreement_val > 0.5:
                            medium_confidence.append(pos_key)
                        else:
                            low_confidence.append(pos_key)
                except Exception as e:
                    logger.error(f"Error processing confidence for position {pos_key}: {e}")
                    low_confidence.append(pos_key)  # Default to low confidence
            
            hybrid_results['confidence_regions'] = {
                'high_confidence': high_confidence,
                'medium_confidence': medium_confidence,
                'low_confidence': low_confidence
            }
            
            # Store results
            self.hybrid_analysis_results = hybrid_results
            
            logger.info(f"Hybrid analysis complete: {nmf_template_agreement} confident detections ({(nmf_template_agreement/total_spectra)*100:.3f}%)")
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Error in NMF-guided template analysis: {e}")
            QMessageBox.critical(self, "Analysis Error", f"Error in hybrid analysis:\n{str(e)}")
            return None

    def _auto_detect_polypropylene_component(self):
        """Try to automatically detect which template contains polypropylene."""
        try:
            if not hasattr(self, 'template_manager') or not self.template_manager.templates:
                logger.info("No templates available for polypropylene detection")
                return None
                
            logger.info("Starting automatic polypropylene template detection...")
            
            # Look for polypropylene characteristic peaks around 1080, 1460, and 2900 cm⁻¹
            pp_peaks = [1080, 1460, 2900]  # Key polypropylene peaks
            best_template_idx = None
            best_score = 0
            
            for i, template in enumerate(self.template_manager.templates):
                logger.info(f"Analyzing template {i}: '{template.name}'")
                
                # Get template data
                if hasattr(template, 'processed_intensities') and template.processed_intensities is not None:
                    intensities = template.processed_intensities
                    wavenumbers = self.template_manager.target_wavenumbers
                    data_type = "processed"
                else:
                    intensities = template.intensities
                    wavenumbers = template.wavenumbers
                    data_type = "raw"
                    
                logger.info(f"Using {data_type} data for template '{template.name}'")
                logger.info(f"Wavenumber range: {np.min(wavenumbers):.1f} to {np.max(wavenumbers):.1f}")
                logger.info(f"Intensity stats: min={np.min(intensities):.6f}, max={np.max(intensities):.6f}, std={np.std(intensities):.6f}")
                
                # Calculate polypropylene score
                score = self._calculate_polypropylene_score(wavenumbers, intensities, template.name)
                logger.info(f"Template '{template.name}' polypropylene score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_template_idx = i
                    
            # Threshold for detection (lowered to be more permissive)
            min_score = 0.3
            if best_score > min_score:
                best_template = self.template_manager.templates[best_template_idx]
                logger.info(f"✅ Detected polypropylene template: '{best_template.name}' (score: {best_score:.3f})")
                return best_template_idx
            else:
                logger.info(f"❌ No template meets polypropylene threshold (best score: {best_score:.3f} < {min_score})")
                return None
                
        except Exception as e:
            logger.error(f"Error in polypropylene auto-detection: {e}")
            return None

    def _calculate_polypropylene_score(self, wavenumbers, intensities, template_name):
        """Calculate how likely a template is to be polypropylene."""
        import numpy as np
        score = 0.0
        
        try:
            # Key polypropylene peaks and their relative importance
            pp_peaks = {
                1080: 1.0,   # Most characteristic peak
                1460: 0.7,   # Secondary peak  
                2900: 0.5,   # Tertiary peak (C-H stretch)
                841: 0.6,    # Another PP peak
                1376: 0.4    # Additional peak
            }
            
            # Look for name hints first
            name_lower = template_name.lower()
            if any(hint in name_lower for hint in ['plastic', 'poly', 'pp', 'propylene']):
                score += 0.2
                logger.info(f"Template name '{template_name}' suggests plastic material (+0.2)")
            
            # Check for characteristic peaks
            total_weight = 0
            found_peaks = 0
            
            for peak_wn, weight in pp_peaks.items():
                # Find closest wavenumber
                closest_idx = np.argmin(np.abs(wavenumbers - peak_wn))
                closest_wn = wavenumbers[closest_idx]
                
                # Must be within reasonable range
                if abs(closest_wn - peak_wn) < 20:  # 20 cm⁻¹ tolerance
                    # Get intensity at this position and nearby region
                    window = 3  # Look at ±3 points around peak
                    start_idx = max(0, closest_idx - window)
                    end_idx = min(len(intensities), closest_idx + window + 1)
                    
                    peak_region = intensities[start_idx:end_idx]
                    peak_intensity = np.max(peak_region)  # Take max in region
                    
                    # Normalize by overall spectrum intensity
                    spectrum_max = np.max(intensities)
                    if spectrum_max > 0:
                        relative_intensity = peak_intensity / spectrum_max
                        
                        # Score based on relative intensity and weight
                        peak_score = relative_intensity * weight
                        score += peak_score
                        found_peaks += 1
                        
                        logger.info(f"  Peak at {peak_wn} cm⁻¹: found at {closest_wn:.1f}, "
                                  f"rel_intensity={relative_intensity:.3f}, score_contrib={peak_score:.3f}")
                    
                total_weight += weight
            
            # Normalize score by total possible weight
            if total_weight > 0:
                score = score / total_weight
                
            # Bonus for finding multiple peaks
            if found_peaks >= 2:
                bonus = min(0.2, found_peaks * 0.05)
                score += bonus
                logger.info(f"Multi-peak bonus for {found_peaks} peaks: +{bonus:.3f}")
                
            # Penalty for very noisy spectra (might indicate poor extraction)
            noise_level = np.std(intensities) / np.mean(intensities) if np.mean(intensities) > 0 else 0
            if noise_level > 0.5:  # High noise
                penalty = min(0.2, noise_level * 0.1)
                score -= penalty
                logger.info(f"High noise penalty (noise={noise_level:.3f}): -{penalty:.3f}")
                
            logger.info(f"Final polypropylene score for '{template_name}': {score:.3f}")
            return max(0.0, min(1.0, score))  # Clamp to [0,1]
            
        except Exception as e:
            logger.error(f"Error calculating polypropylene score for '{template_name}': {e}")
            return 0.0

    def _calculate_agreement_score(self, nmf_intensity, template_relative):
        """Calculate agreement score between NMF and template methods."""
        import numpy as np
        
        try:
            # Ensure scalar values
            if not np.isscalar(nmf_intensity):
                nmf_intensity = float(np.mean(nmf_intensity))
            if not np.isscalar(template_relative):
                template_relative = float(np.mean(template_relative))
            
            # Normalize both values to 0-1 scale
            nmf_norm = min(1.0, float(nmf_intensity) / 10.0)  # Assuming NMF values 0-10
            template_norm = float(template_relative)  # Already 0-1
            
            # Calculate agreement as inverse of difference
            diff = abs(nmf_norm - template_norm)
            agreement = 1.0 - diff
            
            return max(0.0, agreement)
        except Exception as e:
            logger.error(f"Error calculating agreement score: {e}")
            return 0.0

    def show_hybrid_analysis_dialog(self):
        """Show dialog for configuring and running hybrid NMF-Template analysis."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Add debug logging at the start
        logger.info("Starting show_hybrid_analysis_dialog()")
        
        # Check prerequisites
        if not hasattr(self, 'template_fitting_results') or not self.template_fitting_results:
            QMessageBox.warning(self, "No Template Results", "Fit templates first before running hybrid analysis.")
            return
            
        if not hasattr(self, 'nmf_results') or not self.nmf_results:
            QMessageBox.warning(self, "No NMF Results", "Run NMF analysis first before running hybrid analysis.")
            return
        
        # Debug NMF data structure
        logger.info(f"NMF results keys: {list(self.nmf_results.keys())}")
        nmf_components = self.nmf_results.get('components', [])
        logger.info(f"NMF components type: {type(nmf_components)}")
        logger.info(f"Number of NMF components: {len(nmf_components) if hasattr(nmf_components, '__len__') else 'Not iterable'}")
        
        # Handle case where components is a numpy array vs list of arrays
        if hasattr(nmf_components, '__len__') and len(nmf_components) > 0:
            logger.info(f"NMF components shape: {getattr(nmf_components, 'shape', 'No shape')}")
            
            # Check if it's a 2D array that needs to be transposed or split
            if hasattr(nmf_components, 'shape'):
                if len(nmf_components.shape) == 2:
                    logger.info(f"2D array detected with shape {nmf_components.shape}")
                    # This might be components x features, we want features x components for individual component access
                    if nmf_components.shape[0] > nmf_components.shape[1]:
                        logger.info("Transposing components array")
                        nmf_components = nmf_components.T
                elif len(nmf_components.shape) == 1:
                    logger.info(f"1D array detected with {len(nmf_components)} elements")
                    # This might be a flattened array - we need to know the structure better
                    logger.warning("1D array structure needs clarification")
            
            # Try to access first element safely
            try:
                if hasattr(nmf_components, 'shape') and len(nmf_components.shape) >= 2:
                    first_component = nmf_components[0]
                    logger.info(f"First component type: {type(first_component)}")
                    logger.info(f"First component shape: {getattr(first_component, 'shape', 'No shape')}")
                elif hasattr(nmf_components, '__getitem__'):
                    first_item = nmf_components[0]
                    logger.info(f"First item type: {type(first_item)}")
                    logger.info(f"First item value: {first_item}")
            except Exception as e:
                logger.error(f"Error accessing first component: {e}")
        
        # Create hybrid analysis dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Hybrid NMF-Template Analysis")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Header
        header_label = QLabel("🔬 Hybrid Analysis: NMF-Guided Template Refinement")
        header_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; padding: 10px;")
        layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel(
            "This analysis uses NMF component maps to identify candidate regions,\n"
            "then applies template fitting only to those regions for refined quantification.\n"
            "Perfect for detecting trace amounts of materials with weak signals."
        )
        desc_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        layout.addWidget(desc_label)
        
        # NMF Component Selection
        from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout
        
        config_group = QGroupBox("Analysis Configuration")
        config_layout = QFormLayout(config_group)
        
        # Component selection
        self.component_spin = QSpinBox()
        self.component_spin.setMinimum(1)
        
        # Handle different NMF component structures
        if hasattr(nmf_components, 'shape') and len(nmf_components.shape) == 2:
            # 2D array: number of components is the first dimension
            n_components = nmf_components.shape[0]
        elif hasattr(nmf_components, '__len__'):
            # List or 1D array: each element is a component
            n_components = len(nmf_components)
        else:
            n_components = 1
            
        logger.info(f"Setting component spin to {n_components} components")
        self.component_spin.setMaximum(n_components)
        self.component_spin.setValue(min(3, n_components))  # Default to component 3 or max available
        config_layout.addRow("NMF Component (Polypropylene):", self.component_spin)
        
        # Threshold selection
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(0.1)
        self.threshold_spin.setMaximum(20.0)
        self.threshold_spin.setValue(2.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setSuffix(" (NMF intensity)")
        config_layout.addRow("Detection Threshold:", self.threshold_spin)
        
        layout.addWidget(config_group)
        
        # Preview section
        preview_group = QGroupBox("Current Analysis Summary")
        preview_layout = QVBoxLayout(preview_group)
        
        # Get current stats for preview with robust error handling
        try:
            # Check if we have enough components for preview
            has_enough_components = False
            if hasattr(nmf_components, 'shape') and len(nmf_components.shape) == 2:
                has_enough_components = nmf_components.shape[0] >= 3
            elif hasattr(nmf_components, '__len__'):
                has_enough_components = len(nmf_components) >= 3
            
            if has_enough_components:
                import numpy as np
                
                # Extract component 3 data based on structure
                if hasattr(nmf_components, 'shape') and len(nmf_components.shape) == 2:
                    component_3_data = nmf_components[2]  # Get row 2 (component 3)
                else:
                    component_3_data = nmf_components[2]  # Get element 2
                
                # Debug the component data structure
                logger.info(f"Component 3 data type: {type(component_3_data)}")
                logger.info(f"Component 3 data shape: {getattr(component_3_data, 'shape', 'No shape attribute')}")
                
                # Convert to numpy array and handle different structures
                component_array = np.asarray(component_3_data).flatten()
                logger.info(f"Flattened component array shape: {component_array.shape}")
                
                # Safely calculate statistics
                mask = component_array > 2.0
                above_2 = np.sum(mask)
                total_spectra = len(component_array)
                percentage = (above_2 / total_spectra) * 100 if total_spectra > 0 else 0
                
                preview_text = f"Current NMF Component 3 analysis:\n"
                preview_text += f"• Total spectra: {total_spectra:,}\n"
                preview_text += f"• Above threshold (2.0): {above_2} spectra\n" 
                preview_text += f"• Percentage: {percentage:.3f}%\n\n"
                preview_text += f"Template analysis shows: {self.template_fitting_results['template_names']}\n"
                preview_text += f"Hybrid analysis will cross-validate these methods."
            else:
                # Determine the actual number of components
                if hasattr(nmf_components, 'shape') and len(nmf_components.shape) == 2:
                    n_comps = nmf_components.shape[0]
                elif hasattr(nmf_components, '__len__'):
                    n_comps = len(nmf_components)
                else:
                    n_comps = 0
                preview_text = f"NMF analysis has {n_comps} components. Need at least 3 for preview."
        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            preview_text = f"Preview generation failed: {str(e)}\nNMF data structure may be incompatible."
        
        preview_label = QLabel(preview_text)
        preview_label.setStyleSheet("font-family: 'Courier New'; font-size: 10px; background-color: #f8f9fa; padding: 10px; border: 1px solid #dee2e6;")
        preview_layout.addWidget(preview_label)
        
        layout.addWidget(preview_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        run_btn = QPushButton("🚀 Run Hybrid Analysis")
        run_btn.clicked.connect(lambda: self.run_hybrid_analysis_from_dialog(dialog))
        button_layout.addWidget(run_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()

    def run_hybrid_analysis_from_dialog(self, dialog):
        """Run hybrid analysis with parameters from dialog."""
        try:
            # Get parameters from dialog
            component_index = self.component_spin.value() - 1  # Convert to 0-based
            threshold = self.threshold_spin.value()
            
            dialog.accept()  # Close dialog
            
            # Show progress
            self.progress_status.show_progress("Running hybrid NMF-Template analysis...")
            
            # Run the analysis
            results = self.perform_nmf_guided_template_analysis(component_index, threshold)
            
            self.progress_status.hide_progress()
            
            if results:
                # Show results dialog
                self.show_hybrid_analysis_results(results)
                
                # Add hybrid maps to features dropdown
                self.update_map_features_with_hybrid_results()
            else:
                QMessageBox.warning(self, "Analysis Failed", "Hybrid analysis could not be completed.")
                
        except Exception as e:
            self.progress_status.hide_progress()
            QMessageBox.critical(self, "Error", f"Error running hybrid analysis:\n{str(e)}")
            logger.error(f"Error in hybrid analysis dialog: {e}")

    def show_hybrid_analysis_results(self, results):
        """Show detailed results of hybrid analysis."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Hybrid Analysis Results")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Results text area
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setFont(QFont("Courier New", 10))
        
        # Format results
        results_text = self.format_hybrid_analysis_results(results)
        text_area.setPlainText(results_text)
        
        layout.addWidget(text_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(lambda: self.export_hybrid_analysis_results(results))
        button_layout.addWidget(export_btn)
        
        show_maps_btn = QPushButton("Show Hybrid Maps")
        show_maps_btn.clicked.connect(lambda: (dialog.accept(), self.switch_to_hybrid_maps()))
        button_layout.addWidget(show_maps_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()

    def format_hybrid_analysis_results(self, results):
        """Format hybrid analysis results for display."""
        output = []
        output.append("=" * 80)
        output.append("HYBRID NMF-TEMPLATE ANALYSIS RESULTS")
        output.append("=" * 80)
        output.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"NMF Component: {results['nmf_component_index'] + 1}")
        output.append(f"Detection Threshold: {results['nmf_threshold']}")
        output.append("")
        
        stats = results['refined_statistics']
        output.append("REFINED POLYPROPYLENE STATISTICS")
        output.append("-" * 50)
        output.append(f"Total Spectra Analyzed: {stats['total_spectra']:,}")
        output.append(f"NMF Candidate Regions: {stats['nmf_candidates']} ({stats['nmf_candidate_percentage']:.3f}%)")
        output.append(f"NMF-Template Agreement: {stats['agreement_count']} ({stats['agreement_percentage']:.3f}%)")
        output.append(f"High-Confidence Detections: {stats['confident_detections']} ({stats['confident_percentage']:.3f}%)")
        output.append("")
        
        # Confidence breakdown
        confidence = results['confidence_regions']
        output.append("CONFIDENCE LEVEL BREAKDOWN")
        output.append("-" * 50)
        total_hybrid = len(results['nmf_guided_assignments'])
        if total_hybrid > 0:
            high_pct = (len(confidence['high_confidence']) / total_hybrid) * 100
            med_pct = (len(confidence['medium_confidence']) / total_hybrid) * 100
            low_pct = (len(confidence['low_confidence']) / total_hybrid) * 100
            
            output.append(f"High Confidence (>80% agreement): {len(confidence['high_confidence'])} ({high_pct:.1f}%)")
            output.append(f"Medium Confidence (50-80%): {len(confidence['medium_confidence'])} ({med_pct:.1f}%)")
            output.append(f"Low Confidence (<50%): {len(confidence['low_confidence'])} ({low_pct:.1f}%)")
        output.append("")
        
        # Comparison with original template analysis
        output.append("COMPARISON WITH ORIGINAL TEMPLATE ANALYSIS")
        output.append("-" * 50)
        if hasattr(self, 'template_fitting_results'):
            # Calculate original template dominance for polypropylene
            original_dominance = 0
            template_names = self.template_fitting_results['template_names']
            polyprop_idx = None
            for i, name in enumerate(template_names):
                if 'polyprop' in name.lower():
                    polyprop_idx = i
                    break
            
            if polyprop_idx is not None:
                template_coeffs = self.template_fitting_results['coefficients']
                for pos_key, coeffs in template_coeffs.items():
                    if polyprop_idx < len(coeffs):
                        total_contrib = np.sum(coeffs[:len(template_names)])
                        if total_contrib > 1e-10:
                            pp_relative = coeffs[polyprop_idx] / total_contrib
                            if pp_relative > 0.30:  # Using same threshold as hybrid
                                original_dominance += 1
                
                original_pct = (original_dominance / stats['total_spectra']) * 100
                output.append(f"Original Template Dominance: {original_dominance} ({original_pct:.3f}%)")
                output.append(f"Hybrid Refined Estimate: {stats['agreement_count']} ({stats['agreement_percentage']:.3f}%)")
                
                improvement = original_pct - stats['agreement_percentage']
                output.append(f"Refinement: {improvement:+.3f}% (negative = more conservative)")
        
        output.append("")
        output.append("INTERPRETATION")
        output.append("-" * 50)
        refined_pct = stats['agreement_percentage']
        if refined_pct < 0.5:
            output.append("✅ EXCELLENT: Very low contamination level detected")
            output.append("   Both NMF and template methods agree on minimal presence")
        elif refined_pct < 1.0:
            output.append("✅ GOOD: Low contamination level detected")
            output.append("   Consistent with trace material detection")
        else:
            output.append("⚠️  MODERATE: Higher contamination detected")
            output.append("   Consider verifying against independent methods")
        
        output.append("")
        output.append("RECOMMENDED ACTIONS")
        output.append("-" * 50)
        output.append("1. Examine high-confidence regions visually")
        output.append("2. Compare spatial patterns with expected contamination sources")
        output.append("3. Use hybrid confidence maps for publication-quality figures")
        
        return "\n".join(output)

    def export_hybrid_analysis_results(self, results):
        """Export hybrid analysis results to file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Hybrid Analysis Results", 
                f"hybrid_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Text files (*.txt);;All files (*.*)"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.format_hybrid_analysis_results(results))
                
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")

    def switch_to_hybrid_maps(self):
        """Switch to Map tab and show hybrid analysis features."""
        # Switch to Map tab
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Map View":
                self.tab_widget.setCurrentIndex(i)
                break
        
        # Update map features to include hybrid results
        self.update_map_features_with_hybrid_results()
        
        # Set map feature to hybrid confidence map
        control_panel = self.get_current_map_control_panel()
        if control_panel and hasattr(control_panel, 'feature_combo'):
            for i in range(control_panel.feature_combo.count()):
                if "Hybrid Confidence" in control_panel.feature_combo.itemText(i):
                    control_panel.feature_combo.setCurrentIndex(i)
                    break

    def update_map_features_with_hybrid_results(self):
        """Add hybrid analysis results to map features dropdown."""
        if not hasattr(self, 'hybrid_analysis_results'):
            return
        
        # Get current map control panel
        control_panel = self.get_current_map_control_panel()
        if not control_panel or not hasattr(control_panel, 'feature_combo'):
            return
        
        # Add hybrid features to dropdown
        hybrid_features = [
            "Hybrid: Confidence Map",
            "Hybrid: NMF-Template Agreement", 
            "Hybrid: High Confidence Regions",
            "Hybrid: NMF Candidates",
            "Enhanced: NMF Component (Log Scale)",
            "Enhanced: NMF Component (High Contrast)"
        ]
        
        current_items = [control_panel.feature_combo.itemText(i) for i in range(control_panel.feature_combo.count())]
        
        for feature in hybrid_features:
            if feature not in current_items:
                control_panel.feature_combo.addItem(feature)

    def create_hybrid_confidence_map(self):
        """Create a map showing hybrid analysis confidence levels."""
        if not hasattr(self, 'hybrid_analysis_results'):
            QMessageBox.warning(self, "No Hybrid Results", "Run hybrid analysis first.")
            return None
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            results = self.hybrid_analysis_results
            confidence_regions = results['confidence_regions']
            
            # Create base map
            positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            confidence_map = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            # Fill confidence map
            # 0 = no data, 1 = low confidence, 2 = medium confidence, 3 = high confidence
            for pos_key in confidence_regions['low_confidence']:
                x, y = pos_key
                confidence_map[y - y_min, x - x_min] = 1
                
            for pos_key in confidence_regions['medium_confidence']:
                x, y = pos_key
                confidence_map[y - y_min, x - x_min] = 2
                
            for pos_key in confidence_regions['high_confidence']:
                x, y = pos_key
                confidence_map[y - y_min, x - x_min] = 3
            
            # Plot with custom colormap
            plt.figure(figsize=(10, 8))
            
            # Create custom colormap: gray -> yellow -> orange -> red
            from matplotlib.colors import ListedColormap
            colors = ['#f0f0f0', '#fff3cd', '#fd7e14', '#dc3545']  # light gray, light yellow, orange, red
            cmap = ListedColormap(colors)
            
            im = plt.imshow(confidence_map, cmap=cmap, vmin=0, vmax=3, 
                           extent=[x_min, x_max, y_min, y_max], origin='lower')
            
            # Add colorbar with labels
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_ticks([0.375, 1.125, 1.875, 2.625])  # Center of each color band
            cbar.set_ticklabels(['No Detection', 'Low Confidence', 'Medium Confidence', 'High Confidence'])
            
            plt.title(f'Hybrid Analysis Confidence Map\n'
                     f'NMF Component {results["nmf_component_index"] + 1}, Threshold: {results["nmf_threshold"]}', 
                     fontweight='bold')
            plt.xlabel('X Position (μm)')
            plt.ylabel('Y Position (μm)')
            
            # Add statistics text
            stats = results['refined_statistics']
            stats_text = f'High Conf: {len(confidence_regions["high_confidence"])} ' \
                        f'({len(confidence_regions["high_confidence"])/stats["total_spectra"]*100:.2f}%)\n' \
                        f'Total NMF Candidates: {stats["nmf_candidates"]} ' \
                        f'({stats["nmf_candidate_percentage"]:.2f}%)'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            return confidence_map
            
        except Exception as e:
            logger.error(f"Error creating hybrid confidence map: {e}")
            QMessageBox.critical(self, "Map Error", f"Error creating confidence map:\n{str(e)}")
            return None

    def create_enhanced_nmf_component_map(self, log_scale=False, high_contrast=False):
        """Create enhanced NMF component map with improved visualization."""
        if not hasattr(self, 'nmf_results') or not self.nmf_results:
            QMessageBox.warning(self, "No NMF Results", "Run NMF analysis first.")
            return None
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get NMF component data (assume component 3 for polypropylene)
            nmf_components = self.nmf_results.get('components', [])
            if len(nmf_components) < 3:
                QMessageBox.warning(self, "Component Missing", "NMF Component 3 not available.")
                return None
            
            component_data = nmf_components[2]  # Component 3 (0-based index)
            
            # Create map
            positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            nmf_map = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            # Fill map with component intensities
            for i, spectrum in enumerate(self.map_data.spectra.values()):
                if i < len(component_data):
                    intensity = component_data[i]
                    
                    # Apply transformations
                    if log_scale:
                        # Handle potential arrays and ensure positive values
                        if np.isscalar(intensity):
                            if intensity > 0:
                                intensity = np.log10(intensity + 1e-6)
                            else:
                                intensity = np.log10(1e-6)  # Small value for zero/negative
                        else:
                            # Handle array case
                            intensity = np.where(intensity > 0, np.log10(intensity + 1e-6), np.log10(1e-6))
                    
                    nmf_map[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = intensity
            
            # Plot with enhanced visualization
            plt.figure(figsize=(12, 8))
            
            if high_contrast:
                # Use percentile-based scaling for high contrast
                positive_values = nmf_map[nmf_map > 0]
                if len(positive_values) > 0:
                    p1, p99 = np.percentile(positive_values, [1, 99])
                    vmin, vmax = p1, p99
                else:
                    vmin, vmax = np.min(nmf_map), np.max(nmf_map)
                cmap = 'plasma'  # High contrast colormap
            else:
                # Standard scaling
                vmin, vmax = np.min(nmf_map), np.max(nmf_map)
                cmap = 'viridis'
            
            im = plt.imshow(nmf_map, cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=[x_min, x_max, y_min, y_max], origin='lower')
            
            plt.colorbar(im, shrink=0.8, label='NMF Component Intensity')
            
            # Add threshold line if hybrid analysis available
            if hasattr(self, 'hybrid_analysis_results'):
                threshold = self.hybrid_analysis_results['nmf_threshold']
                threshold_color = nmf_map.copy()
                threshold_color[threshold_color < threshold] = np.nan
                
                plt.contour(threshold_color, levels=[threshold], colors='red', linewidths=2,
                           extent=[x_min, x_max, y_min, y_max], origin='lower')
                plt.text(0.02, 0.02, f'Red line: Detection threshold ({threshold})', 
                        transform=plt.gca().transAxes, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Title based on enhancement type
            title_suffix = ""
            if log_scale:
                title_suffix += " (Log Scale)"
            if high_contrast:
                title_suffix += " (High Contrast)"
            
            plt.title(f'Enhanced NMF Component 3 Map{title_suffix}', fontweight='bold')
            plt.xlabel('X Position (μm)')
            plt.ylabel('Y Position (μm)')
            
            # Add detection statistics
            if hasattr(self, 'hybrid_analysis_results'):
                stats = self.hybrid_analysis_results['refined_statistics']
                stats_text = f'Candidates: {stats["nmf_candidates"]} ({stats["nmf_candidate_percentage"]:.3f}%)\n' \
                            f'High Confidence: {stats["confident_detections"]} ({stats["confident_percentage"]:.3f}%)'
                plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                        horizontalalignment='right', verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            return nmf_map
            
        except Exception as e:
            logger.error(f"Error creating enhanced NMF map: {e}")
            QMessageBox.critical(self, "Map Error", f"Error creating enhanced NMF map:\n{str(e)}")
            return None

    def create_nmf_candidate_regions_map(self):
        """Create a map highlighting NMF candidate regions above threshold."""
        if not hasattr(self, 'hybrid_analysis_results'):
            QMessageBox.warning(self, "No Hybrid Results", "Run hybrid analysis first.")
            return None
            
        try:
            import numpy as np
            
            results = self.hybrid_analysis_results
            nmf_component_index = results['nmf_component_index']
            nmf_threshold = results['nmf_threshold']
            
            # Get NMF component data
            nmf_components = self.nmf_results.get('components', [])
            component_data = nmf_components[nmf_component_index]
            
            # Create base map
            positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            candidate_map = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            # Fill with candidate regions
            for i, spectrum in enumerate(self.map_data.spectra.values()):
                if i < len(component_data):
                    nmf_intensity = component_data[i]
                    # Handle both scalar and array values
                    if np.isscalar(nmf_intensity):
                        candidate_value = 1.0 if nmf_intensity > nmf_threshold else 0.0
                    else:
                        # For arrays, use mean value
                        candidate_value = 1.0 if np.mean(nmf_intensity) > nmf_threshold else 0.0
                    candidate_map[spectrum.y_pos - y_min, spectrum.x_pos - x_min] = candidate_value
            
            return candidate_map
            
        except Exception as e:
            logger.error(f"Error creating NMF candidate map: {e}")
            return None

    def create_high_confidence_regions_map(self):
        """Create a map highlighting only high confidence hybrid detections."""
        if not hasattr(self, 'hybrid_analysis_results'):
            QMessageBox.warning(self, "No Hybrid Results", "Run hybrid analysis first.")
            return None
            
        try:
            import numpy as np
            
            results = self.hybrid_analysis_results
            high_confidence_positions = results['confidence_regions']['high_confidence']
            
            # Create base map
            positions = [(s.x_pos, s.y_pos) for s in self.map_data.spectra.values()]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            confidence_map = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
            
            # Fill with high confidence regions
            for pos_key in high_confidence_positions:
                x, y = pos_key
                confidence_map[y - y_min, x - x_min] = 1.0
            
            return confidence_map
            
        except Exception as e:
            logger.error(f"Error creating high confidence map: {e}")
            return None

    # Template Extraction from Map Functionality
    def start_template_extraction_mode(self):
        """Start interactive template extraction mode."""
        if not hasattr(self, 'map_data') or self.map_data is None:
            QMessageBox.warning(self, "No Map Data", "Load map data first before extracting templates.")
            return
        
        # Show instruction dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Extract Templates from Map")
        dialog.setModal(True)
        dialog.resize(500, 350)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instruction_text = QLabel(
            "🔬 Template Extraction Mode\n\n"
            "Instructions:\n"
            "1. Click on map positions to select good example spectra\n"
            "2. Each click will show the spectrum for confirmation\n"
            "3. Give each template a descriptive name\n"
            "4. Templates will be added to your template list\n\n"
            "Tips:\n"
            "• Choose representative spectra from your map\n"
            "• Include both target material and background examples\n"
            "• Select positions with good signal-to-noise ratio"
        )
        instruction_text.setStyleSheet("padding: 10px; background-color: #f0f8ff; border: 1px solid #cce7ff; border-radius: 5px;")
        instruction_text.setWordWrap(True)
        layout.addWidget(instruction_text)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        start_btn = QPushButton("🎯 Start Extraction")
        start_btn.clicked.connect(lambda: self._activate_extraction_mode(dialog))
        button_layout.addWidget(start_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def _activate_extraction_mode(self, dialog):
        """Activate the extraction mode and close dialog."""
        dialog.accept()
        
        # Set extraction mode flag
        self.template_extraction_mode = True
        self.extraction_count = 0
        
        # Show status message
        self.statusBar().showMessage("🎯 Template Extraction Mode: Click on map positions to extract spectra as templates", 0)
        
        # Change cursor to indicate extraction mode
        from PySide6.QtCore import Qt
        self.setCursor(Qt.CursorShape.CrossCursor)
        
        # Show instruction in a temporary message box
        QMessageBox.information(
            self, 
            "Extraction Mode Active", 
            "Template extraction mode is now active!\n\n"
            "Click on any position in the map to extract that spectrum as a template.\n"
            "Press ESC or click 'Exit Extraction Mode' to stop."
        )
    
    def _extract_template_from_position(self, x: float, y: float):
        """Extract a template from the clicked map position."""
        try:
            # Find closest spectrum
            spectrum = self.find_closest_spectrum(x, y)
            if spectrum is None:
                QMessageBox.warning(self, "No Spectrum", "No spectrum found at this position.")
                return
            
            actual_x, actual_y = spectrum.x_pos, spectrum.y_pos
            wavenumbers = spectrum.wavenumbers
            
            # Choose intensities based on processing preference
            if self.use_processed and hasattr(spectrum, 'processed_intensities') and spectrum.processed_intensities is not None:
                intensities = spectrum.processed_intensities
                data_type = "Processed"
            else:
                intensities = spectrum.intensities
                data_type = "Raw"
            
            # Show confirmation dialog with spectrum preview
            self._show_template_extraction_dialog(wavenumbers, intensities, actual_x, actual_y, data_type)
            
        except Exception as e:
            QMessageBox.critical(self, "Extraction Error", f"Error extracting template:\n{str(e)}")
    
    def _show_template_extraction_dialog(self, wavenumbers, intensities, x, y, data_type):
        """Show dialog to confirm template extraction with spectrum preview."""
        from PySide6.QtWidgets import QGroupBox, QLineEdit, QFormLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Extract Template from Position ({x}, {y})")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Info header
        info_label = QLabel(f"📍 Extracting {data_type} spectrum from position ({x}, {y})")
        info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e8f4fd; border-radius: 3px;")
        layout.addWidget(info_label)
        
        # Spectrum preview
        preview_group = QGroupBox("Spectrum Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Create matplotlib widget for preview
        from matplotlib.backends.qt_compat import QtWidgets
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(10, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Plot the spectrum
        ax.plot(wavenumbers, intensities, 'b-', linewidth=1)
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'{data_type} Spectrum at ({x}, {y})')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        preview_layout.addWidget(canvas)
        layout.addWidget(preview_group)
        
        # Template naming
        name_group = QGroupBox("Template Information")
        name_layout = QFormLayout(name_group)
        
        template_name_input = QLineEdit()
        # Auto-suggest name based on position and extraction count
        suggested_name = f"Map_Extract_{self.extraction_count + 1}_({x}_{y})"
        template_name_input.setText(suggested_name)
        template_name_input.selectAll()  # Select all for easy replacement
        name_layout.addRow("Template Name:", template_name_input)
        
        description_input = QLineEdit()
        description_input.setPlaceholderText("Optional: Brief description of this spectrum")
        name_layout.addRow("Description:", description_input)
        
        layout.addWidget(name_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("✅ Add as Template")
        add_btn.clicked.connect(lambda: self._confirm_template_extraction(
            dialog, template_name_input.text(), description_input.text(), 
            wavenumbers, intensities, x, y, data_type))
        button_layout.addWidget(add_btn)
        
        skip_btn = QPushButton("⏭️ Skip This Position")
        skip_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(skip_btn)
        
        button_layout.addStretch()
        
        exit_btn = QPushButton("🚪 Exit Extraction Mode")
        exit_btn.clicked.connect(lambda: self._exit_extraction_mode(dialog))
        button_layout.addWidget(exit_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def _confirm_template_extraction(self, dialog, name, description, wavenumbers, intensities, x, y, data_type):
        """Confirm and add the extracted template."""
        if not name.strip():
            QMessageBox.warning(dialog, "Invalid Name", "Please provide a name for the template.")
            return
        
        try:
            # Add template to template manager
            from ..core.template_management import TemplateSpectrum
            import numpy as np
            
            # Create arrays from input data
            wn_array = np.array(wavenumbers)
            int_array = np.array(intensities)
            
            # DEBUG: Print original template info
            print(f"\n=== TEMPLATE EXTRACTION DEBUG ===")
            print(f"Template name: {name}")
            print(f"Original template - Min: {np.min(int_array):.6f}, Max: {np.max(int_array):.6f}, Mean: {np.mean(int_array):.6f}, Std: {np.std(int_array):.6f}")
            print(f"Original wavenumbers - Min: {np.min(wn_array):.1f}, Max: {np.max(wn_array):.1f}, Length: {len(wn_array)}")
            
            # Check if wavenumbers match map data
            if hasattr(self, 'map_data') and self.map_data:
                first_spectrum = next(iter(self.map_data.spectra.values()))
                map_wn = first_spectrum.wavenumbers
                print(f"Map wavenumbers - Min: {np.min(map_wn):.1f}, Max: {np.max(map_wn):.1f}, Length: {len(map_wn)}")
                print(f"Wavenumber arrays equal: {np.array_equal(wn_array, map_wn)}")
                
                # Check target wavenumbers
                if hasattr(self.template_manager, 'target_wavenumbers') and self.template_manager.target_wavenumbers is not None:
                    print(f"Target wavenumbers - Min: {np.min(self.template_manager.target_wavenumbers):.1f}, Max: {np.max(self.template_manager.target_wavenumbers):.1f}, Length: {len(self.template_manager.target_wavenumbers)}")
                    print(f"Target equals map: {np.array_equal(self.template_manager.target_wavenumbers, map_wn)}")
                else:
                    print("WARNING: Template manager has no target wavenumbers!")
            
            # Preprocess the spectrum using the template manager's method
            processed_intensities = self.template_manager._preprocess_spectrum(wn_array, int_array)
            
            # DEBUG: Print processed template info
            print(f"Processed template - Min: {np.min(processed_intensities):.6f}, Max: {np.max(processed_intensities):.6f}, Mean: {np.mean(processed_intensities):.6f}, Std: {np.std(processed_intensities):.6f}")
            
            # Check if processing destroyed the template
            orig_std = np.std(int_array)
            proc_std = np.std(processed_intensities)
            ratio = proc_std/orig_std if orig_std > 0 else 0
            print(f"Std deviation ratio (processed/original): {ratio:.3f}")
            if ratio < 0.5:
                print("⚠️  WARNING: Processing significantly reduced spectral variation!")
            print(f"================================\n")
            
            template = TemplateSpectrum(
                name=name.strip(),
                wavenumbers=wn_array,
                intensities=int_array,
                processed_intensities=processed_intensities
            )
            
            # Add metadata (if the TemplateSpectrum supports it)
            if hasattr(template, 'metadata'):
                template.metadata = {
                    'description': description.strip(),
                    'source': f'Map position ({x}, {y})',
                    'data_type': data_type,
                    'extraction_date': datetime.now().isoformat()
                }
            
            # Add directly to templates list
            self.template_manager.templates.append(template)
            self.extraction_count += 1
            
            # Update template control panel
            self.update_template_control_panel()
            
            # Show success message
            self.statusBar().showMessage(f"✅ Template '{name}' extracted successfully! ({self.extraction_count} total)", 3000)
            
            dialog.accept()
            
            # Ask if user wants to continue extracting
            reply = QMessageBox.question(
                self, 
                "Continue Extraction?", 
                f"Template '{name}' added successfully!\n\nDo you want to continue extracting more templates?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                self._exit_extraction_mode()
                
        except Exception as e:
            QMessageBox.critical(dialog, "Error", f"Error adding template:\n{str(e)}")
    
    def _exit_extraction_mode(self, dialog=None):
        """Exit template extraction mode."""
        self.template_extraction_mode = False
        
        # Reset cursor
        from PySide6.QtCore import Qt
        self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Update status
        self.statusBar().showMessage(f"🎯 Template extraction completed. {getattr(self, 'extraction_count', 0)} templates extracted.", 5000)
        
        if dialog:
            dialog.reject()
        
        # Show summary if templates were extracted
        if hasattr(self, 'extraction_count') and self.extraction_count > 0:
            QMessageBox.information(
                self,
                "Extraction Complete", 
                f"Successfully extracted {self.extraction_count} templates from map positions!\n\n"
                "You can now:\n"
                "• View templates in the Template Analysis tab\n"
                "• Fit these templates to your map\n"
                "• Run hybrid NMF-Template analysis"
            )

    def show_template_debug_tool(self):
        """Show the template comparison debug tool."""
        try:
            from template_comparison_tool import show_template_comparison_tool
            self.debug_tool = show_template_comparison_tool(self)
        except Exception as e:
            QMessageBox.warning(self, "Debug Tool Error", f"Could not open debug tool:\n{str(e)}")

    def calculate_template_only_material_stats(self):
        """Calculate material statistics from template fitting results only (no NMF interference)."""
        try:
            if not hasattr(self, 'template_fitting_results') or not self.template_fitting_results:
                QMessageBox.warning(self, "No Template Results", "Run template fitting first.")
                return None
                
            logger.info("Calculating template-only material statistics...")
            
            # Get template data
            template_names = self.template_fitting_results['template_names']
            template_coefficients = self.template_fitting_results['coefficients']
            r_squared_values = self.template_fitting_results['r_squared']
            
            # Find target material template (flexible detection + manual selection)
            target_template_idx = None
            
            # Try automatic detection for common materials first
            for i, name in enumerate(template_names):
                name_lower = str(name).lower()
                if any(hint in name_lower for hint in ['pp1', 'pp', 'polyprop', 'plastic', 'propyl', 'pe', 'polyethylene', 'ps', 'polystyrene']):
                    target_template_idx = i
                    logger.info(f"Auto-detected material template: '{name}' at index {i}")
                    break
                    
            # If auto-detection fails, let user select manually
            if target_template_idx is None:
                logger.info("Auto-detection failed, asking user to select target material template...")
                
                from PySide6.QtWidgets import QInputDialog
                template_choices = [f"{i}: {name}" for i, name in enumerate(template_names)]
                
                choice, ok = QInputDialog.getItem(
                    self, 
                    "Select Target Material Template",
                    "Which template represents the material you want to analyze?\n\nSelect the template for quantitative analysis:",
                    template_choices,
                    0,
                    False
                )
                
                if ok and choice:
                    target_template_idx = int(choice.split(':')[0])
                    logger.info(f"User selected target material template: '{template_names[target_template_idx]}' at index {target_template_idx}")
                else:
                    logger.info("User cancelled template selection")
                    QMessageBox.information(self, "Analysis Cancelled", "Material analysis cancelled - no template selected.")
                    return None
                
            target_template_name = template_names[target_template_idx]
            
            # Calculate statistics
            total_spectra = len(template_coefficients)
            material_contributions = []
            material_relative_contributions = []
            high_confidence_positions = []
            medium_confidence_positions = []
            low_confidence_positions = []
            
            # Quality thresholds
            min_r_squared = 0.5  # Minimum fit quality
            high_material_threshold = 0.5  # >50% target material contribution
            medium_material_threshold = 0.2  # 20-50% target material contribution
            
            for pos_key, coeffs in template_coefficients.items():
                try:
                    r_squared = r_squared_values.get(pos_key, 0)
                    
                    # Skip poor fits
                    if r_squared < min_r_squared:
                        continue
                        
                    # Get target material contribution
                    material_contrib = coeffs[target_template_idx] if target_template_idx < len(coeffs) else 0
                    
                    # Calculate relative contribution (target material vs all templates)
                    total_contrib = sum(coeffs[:len(template_names)])
                    material_relative = (material_contrib / total_contrib) if total_contrib > 1e-10 else 0
                    
                    material_contributions.append(material_contrib)
                    material_relative_contributions.append(material_relative)
                    
                    # Categorize confidence levels
                    if material_relative >= high_material_threshold:
                        high_confidence_positions.append(pos_key)
                    elif material_relative >= medium_material_threshold:
                        medium_confidence_positions.append(pos_key)
                    else:
                        low_confidence_positions.append(pos_key)
                        
                except Exception as e:
                    logger.error(f"Error processing position {pos_key}: {e}")
                    continue
            
            # Calculate overall statistics
            import numpy as np
            material_contributions = np.array(material_contributions)
            material_relative_contributions = np.array(material_relative_contributions)
            
            stats = {
                'template_name': target_template_name,
                'total_spectra_analyzed': total_spectra,
                'good_fit_spectra': len(material_contributions),
                'good_fit_percentage': (len(material_contributions) / total_spectra) * 100,
                
                # Material detection statistics
                'high_confidence_count': len(high_confidence_positions),
                'medium_confidence_count': len(medium_confidence_positions), 
                'low_confidence_count': len(low_confidence_positions),
                'detection_count': len(high_confidence_positions) + len(medium_confidence_positions),
                
                # Percentages of total map
                'high_confidence_percentage': (len(high_confidence_positions) / total_spectra) * 100,
                'medium_confidence_percentage': (len(medium_confidence_positions) / total_spectra) * 100,
                'total_detection_percentage': ((len(high_confidence_positions) + len(medium_confidence_positions)) / total_spectra) * 100,
                
                # Material contribution statistics
                'mean_material_contribution': np.mean(material_contributions),
                'std_material_contribution': np.std(material_contributions),
                'max_material_contribution': np.max(material_contributions),
                'mean_material_relative': np.mean(material_relative_contributions),
                'std_material_relative': np.std(material_relative_contributions),
                'max_material_relative': np.max(material_relative_contributions),
                
                # Position lists for mapping
                'high_confidence_positions': high_confidence_positions,
                'medium_confidence_positions': medium_confidence_positions,
                'low_confidence_positions': low_confidence_positions
            }
            
            logger.info(f"Template-only analysis complete:")
            logger.info(f"  Total detection: {stats['total_detection_percentage']:.2f}% of map")
            logger.info(f"  High confidence: {stats['high_confidence_percentage']:.2f}% of map")
            logger.info(f"  Medium confidence: {stats['medium_confidence_percentage']:.2f}% of map")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in template-only polypropylene analysis: {e}")
            QMessageBox.critical(self, "Analysis Error", f"Error calculating polypropylene statistics:\n{str(e)}")
            return None

    def show_template_only_polypropylene_results(self):
        """Show material statistics from template fitting only."""
        stats = self.calculate_template_only_material_stats()
        if stats is None:
            return
            
        # Create results dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Material Detection - Template Analysis Only")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Header
        header = QLabel(f"🧬 Material Analysis Results - Template: {stats['template_name']}")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; padding: 10px;")
        layout.addWidget(header)
        
        # Results text area
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        results_text.setFont(QFont("Courier", 10))
        
        # Format results
        results_content = self.format_template_only_material_results(stats)
        results_text.setPlainText(results_content)
        
        layout.addWidget(results_text)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Export button
        export_btn = QPushButton("📊 Export Results")
        export_btn.clicked.connect(lambda: self.export_template_only_results(stats))
        button_layout.addWidget(export_btn)
        
        # Create map button
        map_btn = QPushButton("🗺️ Show Detection Map")
        map_btn.clicked.connect(lambda: self.create_template_only_detection_map(stats))
        button_layout.addWidget(map_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()

    def format_template_only_material_results(self, stats):
        """Format template-only material results for display."""
        material_name = stats['template_name']
        content = []
        content.append("=" * 80)
        content.append(f"{material_name.upper()} DETECTION - TEMPLATE ANALYSIS ONLY")
        content.append("=" * 80)
        content.append("")
        
        content.append(f"Template Used: {stats['template_name']}")
        content.append(f"Total Spectra: {stats['total_spectra_analyzed']:,}")
        content.append(f"Good Fit Spectra: {stats['good_fit_spectra']:,} ({stats['good_fit_percentage']:.1f}%)")
        content.append("")
        
        content.append(f"{material_name.upper()} DETECTION SUMMARY:")
        content.append("-" * 40)
        content.append(f"🔴 High Confidence (>50%):     {stats['high_confidence_count']:,} spectra ({stats['high_confidence_percentage']:.3f}%)")
        content.append(f"🟡 Medium Confidence (20-50%): {stats['medium_confidence_count']:,} spectra ({stats['medium_confidence_percentage']:.3f}%)")
        content.append(f"🔵 Low Confidence (<20%):      {stats['low_confidence_count']:,} spectra")
        content.append("")
        content.append(f"📊 TOTAL {material_name.upper()} DETECTION: {stats['detection_count']:,} spectra ({stats['total_detection_percentage']:.3f}% of map)")
        content.append("")
        
        content.append(f"{material_name.upper()} CONTRIBUTION STATISTICS:")
        content.append("-" * 40)
        content.append(f"Mean Absolute Contribution: {stats['mean_material_contribution']:.6f} ± {stats['std_material_contribution']:.6f}")
        content.append(f"Maximum Absolute Contribution: {stats['max_material_contribution']:.6f}")
        content.append(f"Mean Relative Contribution: {stats['mean_material_relative']:.3f} ± {stats['std_material_relative']:.3f}")
        content.append(f"Maximum Relative Contribution: {stats['max_material_relative']:.3f}")
        content.append("")
        
        content.append("INTERPRETATION:")
        content.append("-" * 40)
        if stats['total_detection_percentage'] > 1.0:
            content.append(f"⚠️  Significant {material_name} contamination detected ({stats['total_detection_percentage']:.3f}%)")
        elif stats['total_detection_percentage'] > 0.1:
            content.append(f"✅ Moderate {material_name} presence detected ({stats['total_detection_percentage']:.3f}%)")
        else:
            content.append(f"✅ Low level {material_name} traces detected ({stats['total_detection_percentage']:.3f}%)")
            
        if stats['high_confidence_percentage'] > 0.01:
            content.append(f"🎯 High confidence detections: {stats['high_confidence_percentage']:.3f}% - these are very reliable")
            
        content.append("")
        content.append("Note: This analysis uses template fitting only, without NMF interference.")
        content.append(f"Results are based on spectral matching to the {material_name} template reference.")
        
        return "\n".join(content)

    def create_template_only_detection_map(self, stats):
        """Create a map showing template-only polypropylene detections."""
        try:
            if not self.map_data:
                return
                
            # Create detection intensity map
            detection_map = np.zeros((self.map_data.height, self.map_data.width))
            
            # Fill in detection levels
            for pos in stats['high_confidence_positions']:
                if pos in self.map_data.spectra:
                    spectrum = self.map_data.spectra[pos]
                    x_idx = int(spectrum.x_pos)
                    y_idx = int(spectrum.y_pos)
                    if 0 <= y_idx < self.map_data.height and 0 <= x_idx < self.map_data.width:
                        detection_map[y_idx, x_idx] = 3  # High confidence
                        
            for pos in stats['medium_confidence_positions']:
                if pos in self.map_data.spectra:
                    spectrum = self.map_data.spectra[pos]
                    x_idx = int(spectrum.x_pos)
                    y_idx = int(spectrum.y_pos)
                    if 0 <= y_idx < self.map_data.height and 0 <= x_idx < self.map_data.width:
                        if detection_map[y_idx, x_idx] == 0:  # Don't overwrite high confidence
                            detection_map[y_idx, x_idx] = 2  # Medium confidence
                            
            for pos in stats['low_confidence_positions']:
                if pos in self.map_data.spectra:
                    spectrum = self.map_data.spectra[pos]
                    x_idx = int(spectrum.x_pos)
                    y_idx = int(spectrum.y_pos)
                    if 0 <= y_idx < self.map_data.height and 0 <= x_idx < self.map_data.width:
                        if detection_map[y_idx, x_idx] == 0:  # Don't overwrite higher confidence
                            detection_map[y_idx, x_idx] = 1  # Low confidence
            
            # Create and display the map
            self.current_map_data = detection_map
            material_name = stats['template_name']
            self.current_map_title = f"{material_name} Detection - Template Only ({stats['total_detection_percentage']:.3f}%)"
            
            # Update the map display
            self.update_map()
            
            # Switch to map view
            self.tab_widget.setCurrentIndex(0)
            
            # Add custom feature to dropdown
            control_panel = self.get_current_map_control_panel()
            if control_panel and hasattr(control_panel, 'feature_combo'):
                current_features = [control_panel.feature_combo.itemText(i) 
                                  for i in range(control_panel.feature_combo.count())]
                
                feature_name = f"{material_name} Detection (Template Only)"
                if feature_name not in current_features:
                    control_panel.feature_combo.addItem(feature_name)
                    control_panel.feature_combo.setCurrentText(feature_name)
            
            self.statusBar().showMessage(f"Showing {material_name} detection map: {stats['total_detection_percentage']:.3f}% detection rate", 5000)
            
        except Exception as e:
            logger.error(f"Error creating template-only detection map: {e}")
            QMessageBox.critical(self, "Map Error", f"Error creating detection map:\n{str(e)}")

    def export_template_only_results(self, stats):
        """Export template-only polypropylene results."""
        try:
            from PySide6.QtWidgets import QFileDialog
            import os
            from datetime import datetime
            
            # Get save location
            material_name = stats['template_name'].replace(' ', '_').lower()
            default_name = f"{material_name}_template_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Template Analysis Results", default_name,
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                content = self.format_template_only_material_results(stats)
                
                with open(file_path, 'w') as f:
                    f.write(content)
                    f.write("\n\n")
                    f.write("DETAILED POSITION DATA:\n")
                    f.write("=" * 40 + "\n")
                    f.write("High Confidence Positions:\n")
                    for pos in stats['high_confidence_positions']:
                        f.write(f"  {pos}\n")
                    f.write("\nMedium Confidence Positions:\n")
                    for pos in stats['medium_confidence_positions']:
                        f.write(f"  {pos}\n")
                
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting results:\n{str(e)}")

    def keyPressEvent(self, event):
        """Handle key press events - including ESC to exit extraction mode."""
        from PySide6.QtCore import Qt
        
        if event.key() == Qt.Key.Key_Escape:
            if hasattr(self, 'template_extraction_mode') and self.template_extraction_mode:
                self._exit_extraction_mode()
                return
        
        super().keyPressEvent(event)