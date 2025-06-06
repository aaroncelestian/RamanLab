#!/usr/bin/env python3
"""
Raman Polarization Analyzer - Qt6 Version
Qt6 conversion of the comprehensive Raman polarization analysis tool.

Features:
- Compact matplotlib toolbar styling for professional appearance
- Consistent UI theme matching main RamanLab application  
- Optimized icon sizes and spacing for Qt6 integration
- Advanced peak matching and labeling system:
  * Matches experimental peaks with calculated reference peaks
  * Independent peak movement and alignment
  * Automatic vibrational character assignment (A1g, Eg, etc.)
  * Visual peak labeling and shift visualization
  * Quality assessment of peak matches
"""

import sys
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit, minimize
import matplotlib.patches as patches

# Qt6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QFrame, QLabel, QPushButton, QLineEdit, QComboBox, QListWidget,
    QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QProgressBar, QSlider,
    QScrollArea, QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QInputDialog, QDialog, QDialogButtonBox, QFormLayout,
    QListWidgetItem, QProgressDialog
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QStandardPaths
from PySide6.QtGui import QFont, QPixmap, QIcon, QAction, QColor

# Matplotlib Qt6 backend with compact styling
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend which works with PySide6

# Import compact UI configuration and toolbar
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
    from ui.matplotlib_config import apply_theme, configure_compact_ui
    COMPACT_UI_AVAILABLE = True
except ImportError:
    # Fallback for environments without ui.matplotlib_config
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    COMPACT_UI_AVAILABLE = False
    print("⚠ Compact UI configuration not available - using standard matplotlib toolbar")

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Try to import pymatgen for professional CIF parsing
try:
    from pymatgen.io.cif import CifParser
    from pymatgen.core import Structure
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PYMATGEN_AVAILABLE = True
    print("✓ pymatgen available - using professional CIF parser")
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("⚠ pymatgen not available - using simplified CIF parser")
    print("  Install with: pip install pymatgen")


class RamanPolarizationAnalyzerQt6(QMainWindow):
    """Qt6 version of the Raman Polarization Analyzer."""
    
    def __init__(self):
        """Initialize the Raman Polarization Analyzer Qt6 version."""
        super().__init__()
        
        # Apply compact UI configuration for consistent toolbar sizing
        if COMPACT_UI_AVAILABLE:
            apply_theme('compact')
        
        self.setWindowTitle("Raman Polarization Analyzer - Qt6")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables (same as original)
        self.mineral_database = None
        self.current_spectrum = None
        self.original_spectrum = None
        self.imported_spectrum = None
        self.mineral_list = []
        
        # Peak fitting variables
        self.selected_peaks = []
        self.fitted_peaks = []
        self.peak_assignments = {}
        self.frequency_shifts = {}
        self.peak_selection_mode = False
        
        # Peak matching and labeling variables
        self.matched_peaks = []  # Store matched experimental-calculated peak pairs
        self.peak_labels = {}    # Store character labels for each peak
        self.calculated_peaks = []  # Store calculated peaks from reference mineral
        self.peak_matching_tolerance = 50  # cm⁻¹ tolerance for peak matching
        
        # Polarization analysis variables
        self.polarization_data = {}
        self.raman_tensors = {}
        self.depolarization_ratios = {}
        self.angular_data = {}
        self.current_polarization_config = None
        
        # Cross-tab integration variables
        self.selected_reference_mineral = None
        
        # Crystal structure variables
        self.current_crystal_structure = None
        self.current_crystal_bonds = []
        
        # Raman tensor variables
        self.calculated_raman_tensors = {}
        self.tensor_analysis_results = {}
        
        # Orientation optimization variables
        self.orientation_results = {}
        self.optimization_parameters = {}
        self.stage_results = {'stage1': None, 'stage2': None, 'stage3': None}
        self.optimized_orientation = None
        
        # 3D visualization variables
        self.crystal_shape_data = None
        self.tensor_data_3d = None
        
        # Stress/strain analysis variables
        self.stress_strain_data = {}
        self.stress_coefficients = {}
        self.strain_analysis_results = {}
        
        # Store references to plot components for each tab
        self.plot_components = {}
        
        # Load mineral database
        self.load_mineral_database()
        
        # Create the main GUI
        self.init_ui()
        
        # Update mineral lists in UI after GUI is created
        self.update_mineral_lists()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_tabs()
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def create_tabs(self):
        """Create all tabs with their respective layouts."""
        tab_names = [
            "Spectrum Analysis",
            "Peak Fitting", 
            "Crystal Structure",
            "Tensor Analysis & Visualization",
            "Orientation Optimization",
            "Stress/Strain Analysis",
            "3D Visualization"
        ]
        
        for tab_name in tab_names:
            self.create_tab(tab_name)
    
    def create_tab(self, tab_name):
        """Create a tab with side panel and main content area."""
        # Create tab widget
        tab_widget = QWidget()
        self.tab_widget.addTab(tab_widget, tab_name)
        
        # Create horizontal splitter for side panel and main content
        splitter = QSplitter(Qt.Horizontal)
        tab_layout = QHBoxLayout(tab_widget)
        tab_layout.addWidget(splitter)
        
        # Create side panel
        side_panel = QFrame()
        side_panel.setFrameStyle(QFrame.StyledPanel)
        side_panel.setFixedWidth(300)
        splitter.addWidget(side_panel)
        
        # Create main content area
        content_area = QFrame()
        content_area.setFrameStyle(QFrame.StyledPanel)
        splitter.addWidget(content_area)
        
        # Set splitter proportions
        splitter.setSizes([300, 900])
        
        # Store references
        self.plot_components[tab_name] = {
            'tab_widget': tab_widget,
            'side_panel': side_panel,
            'content_area': content_area,
            'splitter': splitter
        }
        
        # Setup content based on tab
        self.setup_tab_content(tab_name)
    
    def setup_tab_content(self, tab_name):
        """Setup the specific content for each tab."""
        components = self.plot_components[tab_name]
        side_panel = components['side_panel']
        content_area = components['content_area']
        
        if tab_name == "Spectrum Analysis":
            self.setup_spectrum_analysis_tab(side_panel, content_area)
        elif tab_name == "Peak Fitting":
            self.setup_peak_fitting_tab(side_panel, content_area)
        elif tab_name == "Crystal Structure":
            self.setup_crystal_structure_tab(side_panel, content_area)
        elif tab_name == "Orientation Optimization":
            self.setup_orientation_optimization_tab(side_panel, content_area)
        elif tab_name == "Tensor Analysis & Visualization":
            self.setup_raman_tensors_tab(side_panel, content_area)
        elif tab_name == "Stress/Strain Analysis":
            self.setup_stress_strain_tab(side_panel, content_area)
        elif tab_name == "3D Visualization":
            self.setup_3d_visualization_tab(side_panel, content_area)
    
    def setup_spectrum_analysis_tab(self, side_panel, content_area):
        """Setup the Spectrum Analysis tab."""
        # Side panel layout
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Spectrum Analysis")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        load_btn = QPushButton("Load Spectrum")
        load_btn.clicked.connect(self.load_spectrum)
        file_layout.addWidget(load_btn)
        
        save_btn = QPushButton("Save Spectrum")
        save_btn.clicked.connect(self.save_spectrum)
        file_layout.addWidget(save_btn)
        
        export_btn = QPushButton("Export Plot")
        export_btn.clicked.connect(self.export_plot)
        file_layout.addWidget(export_btn)
        
        side_layout.addWidget(file_group)
        
        # Import spectrum group
        import_group = QGroupBox("Import Spectrum")
        import_layout = QVBoxLayout(import_group)
        
        search_label = QLabel("Search Mineral Database:")
        import_layout.addWidget(search_label)
        
        self.search_entry = QLineEdit()
        self.search_entry.textChanged.connect(self.on_search_change_with_filter)
        import_layout.addWidget(self.search_entry)
        
        self.search_listbox = QListWidget()
        self.search_listbox.setMaximumHeight(120)
        self.search_listbox.itemDoubleClicked.connect(self.on_mineral_select)
        import_layout.addWidget(self.search_listbox)
        
        import_btn = QPushButton("Import Selected")
        import_btn.clicked.connect(self.import_selected_mineral)
        import_layout.addWidget(import_btn)
        
        # Crystal system filter
        crystal_filter_layout = QHBoxLayout()
        crystal_filter_layout.addWidget(QLabel("Crystal System:"))
        
        self.crystal_filter_combo = QComboBox()
        crystal_systems = ["All", "Cubic", "Tetragonal", "Hexagonal", "Orthorhombic", "Monoclinic", "Triclinic", "Trigonal"]
        self.crystal_filter_combo.addItems(crystal_systems)
        self.crystal_filter_combo.currentTextChanged.connect(self.on_search_change_with_filter)
        crystal_filter_layout.addWidget(self.crystal_filter_combo)
        
        import_layout.addLayout(crystal_filter_layout)
        
        # Refresh database button
        refresh_btn = QPushButton("Refresh Database")
        refresh_btn.clicked.connect(self.refresh_database)
        refresh_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        import_layout.addWidget(refresh_btn)
        
        side_layout.addWidget(import_group)
        
        # Analysis options group - removed non-functional buttons
        # (Spectrum analysis is handled through dedicated tabs)
        
        # Add stretch to push everything to top
        side_layout.addStretch()
        
        # Content area - matplotlib plot
        content_layout = QVBoxLayout(content_area)
        
        # Create matplotlib figure and canvas
        self.spectrum_fig = Figure(figsize=(8, 6))
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)
        
        # Create navigation toolbar
        self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, content_area)
        self.apply_toolbar_styling(self.spectrum_toolbar)
        
        content_layout.addWidget(self.spectrum_toolbar)
        content_layout.addWidget(self.spectrum_canvas)
        
        # Store plot components
        self.plot_components["Spectrum Analysis"].update({
            'fig': self.spectrum_fig,
            'canvas': self.spectrum_canvas,
            'ax': self.spectrum_ax,
            'toolbar': self.spectrum_toolbar
        })
        
        # Initialize empty plot
        self.update_spectrum_plot()
    
    def setup_peak_fitting_tab(self, side_panel, content_area):
        """Setup the Peak Fitting tab with sub-tabs for Parameters and Assignment."""
        # Create a tabbed interface within the Peak Fitting tab
        peak_fitting_tabs = QTabWidget()
        
        # Create the two sub-tabs
        parameters_tab = QWidget()
        assignment_tab = QWidget()
        
        peak_fitting_tabs.addTab(parameters_tab, "Parameters")
        peak_fitting_tabs.addTab(assignment_tab, "Assignment")
        
        # Set up the Parameters sub-tab (original Peak Fitting functionality)
        self.setup_parameters_subtab(parameters_tab)
        
        # Set up the Assignment sub-tab (original Peak Analysis functionality)
        self.setup_assignment_subtab(assignment_tab)
        
        # Connect subtab change signal to update reference selections when switching to Assignment tab
        peak_fitting_tabs.currentChanged.connect(self.on_peak_fitting_subtab_changed)
        
        # Store reference to the peak fitting tabs widget for later access
        self.peak_fitting_tabs = peak_fitting_tabs
        
        # Add the tabbed widget to the side panel
        side_layout = QVBoxLayout(side_panel)
        
        # Title for the main tab
        title_label = QLabel("Peak Fitting")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Add the sub-tabs
        side_layout.addWidget(peak_fitting_tabs)
        
        # Content area - matplotlib plot (shared by both sub-tabs)
        content_layout = QVBoxLayout(content_area)
        
        # Create matplotlib figure and canvas
        self.peak_fig = Figure(figsize=(8, 6))
        self.peak_canvas = FigureCanvas(self.peak_fig)
        self.peak_ax = self.peak_fig.add_subplot(111)
        
        # Create navigation toolbar
        self.peak_toolbar = NavigationToolbar(self.peak_canvas, content_area)
        self.apply_toolbar_styling(self.peak_toolbar)
        
        content_layout.addWidget(self.peak_toolbar)
        content_layout.addWidget(self.peak_canvas)
        
        # Store plot components
        self.plot_components["Peak Fitting"].update({
            'fig': self.peak_fig,
            'canvas': self.peak_canvas,
            'ax': self.peak_ax,
            'toolbar': self.peak_toolbar
        })
        
        # Connect mouse events for peak selection
        self.peak_canvas.mpl_connect('button_press_event', self.on_peak_click)
        
        # Initialize preprocessing variables
        self.processed_spectrum = None
        self.baseline_corrected = False
        self.noise_filtered = False
        
        # Initialize empty plot
        self.update_peak_fitting_plot()
    
    def setup_parameters_subtab(self, parameters_tab):
        """Setup the Parameters sub-tab (original Peak Fitting functionality)."""
        layout = QVBoxLayout(parameters_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Peak selection group
        peak_group = QGroupBox("Peak Selection")
        peak_layout = QVBoxLayout(peak_group)
        
        self.peak_selection_btn = QPushButton("🎯 Enable Peak Selection")
        self.peak_selection_btn.clicked.connect(self.toggle_peak_selection)
        self.peak_selection_btn.setToolTip("Enable peak selection mode:\n• Left-click peaks to select them\n• Ctrl+Click OR Right-click near peaks to remove them (within 10 cm⁻¹)")
        peak_layout.addWidget(self.peak_selection_btn)
        
        # Add instruction label
        instruction_label = QLabel("💡 After enabling selection:\n• Left-click peaks to select them\n• Ctrl+Click OR Right-click near peaks to remove them")
        instruction_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        instruction_label.setWordWrap(True)
        peak_layout.addWidget(instruction_label)
        
        clear_peaks_btn = QPushButton("Clear Selected Peaks")
        clear_peaks_btn.clicked.connect(self.clear_selected_peaks)
        peak_layout.addWidget(clear_peaks_btn)
        
        # Add peak count status
        self.peak_count_label = QLabel("Selected peaks: 0")
        self.peak_count_label.setStyleSheet("color: #333; font-size: 11px; font-weight: bold;")
        peak_layout.addWidget(self.peak_count_label)
        
        # Auto peak detection button
        auto_detect_btn = QPushButton("Auto-Detect Peaks")
        auto_detect_btn.clicked.connect(self.auto_detect_peaks)
        auto_detect_btn.setStyleSheet("QPushButton { background-color: #e6f3ff; }")
        peak_layout.addWidget(auto_detect_btn)
        
        layout.addWidget(peak_group)
        
        # Data preprocessing group
        preprocess_group = QGroupBox("Data Preprocessing")
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        # Noise reduction
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Smoothing:"))
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 20)
        self.smoothing_spin.setValue(3)
        self.smoothing_spin.setSuffix(" pts")
        noise_layout.addWidget(self.smoothing_spin)
        preprocess_layout.addLayout(noise_layout)
        
        # Baseline correction
        baseline_btn = QPushButton("Correct Baseline")
        baseline_btn.clicked.connect(self.correct_baseline)
        preprocess_layout.addWidget(baseline_btn)
        
        # Noise filtering
        noise_filter_btn = QPushButton("Apply Noise Filter")
        noise_filter_btn.clicked.connect(self.apply_noise_filter)
        preprocess_layout.addWidget(noise_filter_btn)
        
        layout.addWidget(preprocess_group)
        
        # Fitting options group
        fitting_group = QGroupBox("Fitting Options")
        fitting_layout = QVBoxLayout(fitting_group)
        
        # Peak shape selection
        shape_label = QLabel("Peak Shape:")
        fitting_layout.addWidget(shape_label)
        
        self.peak_shape_combo = QComboBox()
        self.peak_shape_combo.addItems(["Lorentzian", "Gaussian", "Voigt", "Pseudo-Voigt"])
        fitting_layout.addWidget(self.peak_shape_combo)
        
        # Fitting range control
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Fit Range:"))
        self.fit_range_spin = QSpinBox()
        self.fit_range_spin.setRange(10, 200)
        self.fit_range_spin.setValue(50)
        self.fit_range_spin.setSuffix(" cm⁻¹")
        range_layout.addWidget(self.fit_range_spin)
        fitting_layout.addLayout(range_layout)
        
        # Multipeak fitting checkbox
        self.multipeak_check = QCheckBox("Multipeak Fitting")
        self.multipeak_check.setToolTip("Fit overlapping peaks simultaneously")
        fitting_layout.addWidget(self.multipeak_check)
        
        # Weak peak detection threshold
        weak_layout = QHBoxLayout()
        weak_layout.addWidget(QLabel("Weak Peak Threshold:"))
        self.weak_threshold_spin = QDoubleSpinBox()
        self.weak_threshold_spin.setRange(0.01, 1.0)
        self.weak_threshold_spin.setValue(0.1)
        self.weak_threshold_spin.setSingleStep(0.01)
        self.weak_threshold_spin.setDecimals(2)
        weak_layout.addWidget(self.weak_threshold_spin)
        fitting_layout.addLayout(weak_layout)
        
        # Fit button
        fit_btn = QPushButton("Fit Selected Peaks")
        fit_btn.clicked.connect(self.fit_peaks)
        fit_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        fitting_layout.addWidget(fit_btn)
        
        # Advanced multipeak fit button
        multipeak_btn = QPushButton("Multipeak + Weak Peak")
        multipeak_btn.clicked.connect(self.fit_overlapping_peaks)
        multipeak_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        multipeak_btn.setToolTip("Fit selected peaks and automatically detect/fit additional weak peaks in the region")
        fitting_layout.addWidget(multipeak_btn)
        
        layout.addWidget(fitting_group)
        
        # Peak analysis group (moved from original Peak Analysis)
        analysis_group = QGroupBox("Peak Quality & Export")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Peak quality assessment
        quality_btn = QPushButton("Assess Peak Quality")
        quality_btn.clicked.connect(self.assess_peak_quality)
        quality_btn.setToolTip("Evaluate S/N ratio, width, and fit quality for all peaks")
        analysis_layout.addWidget(quality_btn)
        
        # Peak deconvolution for overlaps
        deconv_btn = QPushButton("Deconvolve Overlapping Peaks")
        deconv_btn.clicked.connect(self.deconvolve_peaks)
        deconv_btn.setToolTip("Identify and analyze overlapping peak pairs")
        analysis_layout.addWidget(deconv_btn)
        
        # Export fitted parameters
        export_params_btn = QPushButton("Export Fit Parameters")
        export_params_btn.clicked.connect(self.export_fit_parameters)
        export_params_btn.setToolTip("Export all fitted parameters to CSV/TXT file")
        analysis_layout.addWidget(export_params_btn)
        
        layout.addWidget(analysis_group)
        
        # Add stretch
        layout.addStretch()
    
    def setup_assignment_subtab(self, assignment_tab):
        """Setup the Assignment sub-tab (original Peak Analysis functionality)."""
        layout = QVBoxLayout(assignment_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Quick access to fitted peaks
        status_group = QGroupBox("Fitting Status")
        status_layout = QVBoxLayout(status_group)
        
        self.fitting_status_label = QLabel("No peaks fitted")
        self.fitting_status_label.setStyleSheet("color: gray; font-style: italic;")
        status_layout.addWidget(self.fitting_status_label)
        
        # Switch to Parameters tab button
        params_btn = QPushButton("→ Go to Parameters Tab")
        params_btn.clicked.connect(lambda: self.switch_to_parameters_tab())
        params_btn.setStyleSheet("QPushButton { background-color: #f0f8ff; }")
        status_layout.addWidget(params_btn)
        
        layout.addWidget(status_group)
        
        # Peak matching group
        matching_group = QGroupBox("Peak Matching & Labeling")
        matching_layout = QVBoxLayout(matching_group)
        
        # Reference mineral selection
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference:"))
        self.analysis_reference_combo = QComboBox()
        self.analysis_reference_combo.currentTextChanged.connect(self.on_reference_mineral_changed)
        ref_layout.addWidget(self.analysis_reference_combo)
        matching_layout.addLayout(ref_layout)
        
        self.analysis_reference_status_label = QLabel("No reference selected")
        self.analysis_reference_status_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        matching_layout.addWidget(self.analysis_reference_status_label)
        
        # Tolerance setting
        tolerance_layout = QHBoxLayout()
        tolerance_layout.addWidget(QLabel("Tolerance:"))
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(5, 200)
        self.tolerance_spin.setValue(50)
        self.tolerance_spin.setSuffix(" cm⁻¹")
        self.tolerance_spin.valueChanged.connect(self.update_matching_tolerance)
        tolerance_layout.addWidget(self.tolerance_spin)
        matching_layout.addLayout(tolerance_layout)
        
        # Match peaks button
        match_peaks_btn = QPushButton("Match with Reference Peaks")
        match_peaks_btn.clicked.connect(self.match_peaks_with_reference)
        match_peaks_btn.setToolTip("Match fitted peaks with calculated reference peaks")
        match_peaks_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        matching_layout.addWidget(match_peaks_btn)
        
        # Auto-assign labels button
        assign_labels_btn = QPushButton("Auto-Assign Peak Labels")
        assign_labels_btn.clicked.connect(self.auto_assign_peak_labels)
        assign_labels_btn.setToolTip("Automatically assign vibrational mode labels")
        matching_layout.addWidget(assign_labels_btn)
        
        # Show assignments button
        show_assignments_btn = QPushButton("Show Peak Assignments")
        show_assignments_btn.clicked.connect(self.show_peak_assignments)
        show_assignments_btn.setToolTip("Display detailed peak assignment results")
        matching_layout.addWidget(show_assignments_btn)
        
        layout.addWidget(matching_group)
        
        # Advanced analysis group
        advanced_group = QGroupBox("Advanced Analysis")
        advanced_layout = QVBoxLayout(advanced_group)
        
        # Show Raman activity info button
        activity_info_btn = QPushButton("Raman Activity Filter")
        activity_info_btn.clicked.connect(self.show_raman_activity_info)
        activity_info_btn.setStyleSheet("QPushButton { background-color: #e7f3ff; }")
        activity_info_btn.setToolTip("Information about Raman active modes")
        advanced_layout.addWidget(activity_info_btn)
        
        # Debug calculated modes button
        debug_modes_btn = QPushButton("Debug Calculated Modes")
        debug_modes_btn.clicked.connect(self.debug_calculated_modes)
        debug_modes_btn.setStyleSheet("QPushButton { background-color: #fff3cd; }")
        debug_modes_btn.setToolTip("Debug information for calculated vibrational modes")
        advanced_layout.addWidget(debug_modes_btn)
        
        # Debug database content button
        debug_db_btn = QPushButton("Debug Database Content")
        debug_db_btn.clicked.connect(self.debug_database_content)
        debug_db_btn.setStyleSheet("QPushButton { background-color: #d4edda; }")
        debug_db_btn.setToolTip("Debug information for mineral database content")
        advanced_layout.addWidget(debug_db_btn)
        
        layout.addWidget(advanced_group)
        
        # Add stretch
        layout.addStretch()
    
    def switch_to_parameters_tab(self):
        """Switch to the Parameters sub-tab within Peak Fitting."""
        # Find the Peak Fitting tab and switch to Parameters sub-tab
        peak_fitting_tab_index = 1  # Peak Fitting is typically the second tab (index 1)
        self.tab_widget.setCurrentIndex(peak_fitting_tab_index)
        
        # Switch to the Parameters sub-tab (index 0)
        # We need to find the QTabWidget within the Peak Fitting tab
        peak_fitting_tab = self.tab_widget.widget(peak_fitting_tab_index)
        if hasattr(peak_fitting_tab, 'findChild'):
            tab_widget = peak_fitting_tab.findChild(QTabWidget)
            if tab_widget:
                tab_widget.setCurrentIndex(0)  # Parameters tab
    

    
    def setup_crystal_structure_tab(self, side_panel, content_area):
        """Setup the Crystal Structure tab using the new comprehensive module."""
        try:
            from ui.crystal_structure_widget import CrystalStructureWidget
            
            # Create the main crystal structure widget
            self.crystal_structure_widget = CrystalStructureWidget()
            
            # Replace content area with the new widget
            content_layout = QVBoxLayout(content_area)
            content_layout.addWidget(self.crystal_structure_widget)
            
            # Hide the side panel since the new widget has its own controls
            side_panel.hide()
            
            # Connect signals if needed
            self.crystal_structure_widget.structure_loaded.connect(self.on_structure_loaded)
            self.crystal_structure_widget.bond_calculated.connect(self.on_bonds_calculated)
            
        except ImportError as e:
            print(f"⚠ Could not load crystal structure module: {e}")
            # Fallback to the old implementation
            self.setup_crystal_structure_tab_fallback(side_panel, content_area)
    
    def setup_crystal_structure_tab_fallback(self, side_panel, content_area):
        """Fallback crystal structure tab setup."""
        # Side panel layout
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Crystal Structure")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Structure input group
        struct_group = QGroupBox("Structure Input")
        struct_layout = QVBoxLayout(struct_group)
        
        load_cif_btn = QPushButton("Load CIF File")
        load_cif_btn.clicked.connect(self.load_cif_file)
        struct_layout.addWidget(load_cif_btn)
        
        manual_input_btn = QPushButton("Manual Input")
        struct_layout.addWidget(manual_input_btn)
        
        side_layout.addWidget(struct_group)
        
        # Add stretch
        side_layout.addStretch()
        
        # Content area - placeholder
        content_layout = QVBoxLayout(content_area)
        
        struct_info_label = QLabel("Crystal structure visualization will be implemented here")
        struct_info_label.setAlignment(Qt.AlignCenter)
        struct_info_label.setStyleSheet("color: gray; font-style: italic;")
        content_layout.addWidget(struct_info_label)
    
    def setup_raman_tensors_tab(self, side_panel, content_area):
        """Setup the Tensor Analysis tab."""
        # Side panel layout
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Tensor Analysis & Visualization")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Crystal system group
        crystal_group = QGroupBox("Crystal System")
        crystal_layout = QVBoxLayout(crystal_group)
        
        crystal_label = QLabel("Select Crystal System:")
        crystal_layout.addWidget(crystal_label)
        
        self.tensor_crystal_system_combo = QComboBox()
        crystal_systems = ["Cubic", "Tetragonal", "Hexagonal", "Orthorhombic", "Monoclinic", "Triclinic", "Trigonal"]
        self.tensor_crystal_system_combo.addItems(crystal_systems)
        self.tensor_crystal_system_combo.currentTextChanged.connect(self.on_tensor_crystal_system_changed)
        crystal_layout.addWidget(self.tensor_crystal_system_combo)
        
        # Auto-detect button
        auto_detect_btn = QPushButton("Auto-Detect from Structure")
        auto_detect_btn.clicked.connect(self.auto_detect_crystal_system)
        crystal_layout.addWidget(auto_detect_btn)
        
        side_layout.addWidget(crystal_group)
        
        # Tensor calculation group
        tensor_group = QGroupBox("Tensor Calculation")
        tensor_layout = QVBoxLayout(tensor_group)
        
        # Calculation method
        method_label = QLabel("Calculation Method:")
        tensor_layout.addWidget(method_label)
        
        self.tensor_method_combo = QComboBox()
        self.tensor_method_combo.addItems([
            "Symmetry-Based",
            "Peak Intensity Analysis", 
            "Polarization Data",
            "Combined Analysis"
        ])
        tensor_layout.addWidget(self.tensor_method_combo)
        
        calc_tensor_btn = QPushButton("Calculate Raman Tensors")
        calc_tensor_btn.clicked.connect(self.calculate_raman_tensors)
        calc_tensor_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        tensor_layout.addWidget(calc_tensor_btn)
        
        # Tensor visualization options (radio buttons)
        viz_label = QLabel("Visualization Mode:")
        tensor_layout.addWidget(viz_label)
        
        self.viz_button_group = QButtonGroup()
        
        self.show_tensor_matrix_rb = QRadioButton("Tensor Matrices")
        self.show_tensor_matrix_rb.setChecked(True)
        self.viz_button_group.addButton(self.show_tensor_matrix_rb, 0)
        tensor_layout.addWidget(self.show_tensor_matrix_rb)
        
        self.show_eigenvalues_rb = QRadioButton("Eigenvalues & Analysis")
        self.viz_button_group.addButton(self.show_eigenvalues_rb, 1)
        tensor_layout.addWidget(self.show_eigenvalues_rb)
        
        self.show_ellipsoids_rb = QRadioButton("3D Tensor Shapes")
        self.viz_button_group.addButton(self.show_ellipsoids_rb, 2)
        tensor_layout.addWidget(self.show_ellipsoids_rb)
        
        self.show_overview_rb = QRadioButton("Overview Summary")
        self.viz_button_group.addButton(self.show_overview_rb, 3)
        tensor_layout.addWidget(self.show_overview_rb)
        
        # Update visualization button
        update_viz_btn = QPushButton("Update Visualization")
        update_viz_btn.clicked.connect(self.update_tensor_visualization)
        tensor_layout.addWidget(update_viz_btn)
        
        # Individual tensor windows button
        individual_btn = QPushButton("Open Individual Tensor Windows")
        individual_btn.clicked.connect(self.open_individual_tensor_windows)
        individual_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        tensor_layout.addWidget(individual_btn)
        
        side_layout.addWidget(tensor_group)
        
        # Analysis results group
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        show_results_btn = QPushButton("Show Detailed Results")
        show_results_btn.clicked.connect(self.show_tensor_results)
        results_layout.addWidget(show_results_btn)
        
        export_tensor_btn = QPushButton("Export Tensor Data")
        export_tensor_btn.clicked.connect(self.export_tensor_data)
        results_layout.addWidget(export_tensor_btn)
        
        export_viz_btn = QPushButton("Export Visualization")
        export_viz_btn.clicked.connect(self.export_tensor_visualization)
        results_layout.addWidget(export_viz_btn)
        
        side_layout.addWidget(results_group)
        
        # Peak mode assignment group
        assignment_group = QGroupBox("Mode Assignment")
        assignment_layout = QVBoxLayout(assignment_group)
        
        assign_modes_btn = QPushButton("Assign Vibrational Modes")
        assign_modes_btn.clicked.connect(self.assign_vibrational_modes)
        assignment_layout.addWidget(assign_modes_btn)
        
        show_assignments_btn = QPushButton("Show Mode Assignments")
        show_assignments_btn.clicked.connect(self.show_mode_assignments)
        assignment_layout.addWidget(show_assignments_btn)
        
        side_layout.addWidget(assignment_group)
        
        # Add stretch
        side_layout.addStretch()
        
        # Content area - matplotlib visualization
        content_layout = QVBoxLayout(content_area)
        
        # Create matplotlib figure for tensor visualization
        self.tensor_fig = Figure(figsize=(10, 8))
        self.tensor_canvas = FigureCanvas(self.tensor_fig)
        self.tensor_toolbar = NavigationToolbar(self.tensor_canvas, content_area)
        self.apply_toolbar_styling(self.tensor_toolbar)
        
        content_layout.addWidget(self.tensor_toolbar)
        content_layout.addWidget(self.tensor_canvas)
        
        # Store plot components
        self.plot_components["Tensor Analysis & Visualization"].update({
            'fig': self.tensor_fig,
            'canvas': self.tensor_canvas,
            'toolbar': self.tensor_toolbar
        })
        
        # Initialize empty plot
        self.initialize_tensor_plot()
    
    def setup_orientation_optimization_tab(self, side_panel, content_area):
        """Setup the Orientation Optimization tab."""
        # Side panel layout
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Orientation Optimization")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Optimization group
        opt_group = QGroupBox("Optimization Parameters")
        opt_layout = QVBoxLayout(opt_group)
        
        run_opt_btn = QPushButton("Run Optimization")
        run_opt_btn.clicked.connect(self.run_orientation_optimization)
        opt_layout.addWidget(run_opt_btn)
        
        side_layout.addWidget(opt_group)
        
        # Add stretch
        side_layout.addStretch()
        
        # Content area - placeholder
        content_layout = QVBoxLayout(content_area)
        
        opt_info_label = QLabel("Orientation optimization will be implemented here")
        opt_info_label.setAlignment(Qt.AlignCenter)
        opt_info_label.setStyleSheet("color: gray; font-style: italic;")
        content_layout.addWidget(opt_info_label)
    
    def setup_stress_strain_tab(self, side_panel, content_area):
        """Setup the Stress/Strain Analysis tab."""
        # Side panel layout
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Stress/Strain Analysis")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Analysis group
        stress_group = QGroupBox("Stress Analysis")
        stress_layout = QVBoxLayout(stress_group)
        
        calc_stress_btn = QPushButton("Calculate Stress")
        stress_layout.addWidget(calc_stress_btn)
        
        side_layout.addWidget(stress_group)
        
        # Add stretch
        side_layout.addStretch()
        
        # Content area - placeholder
        content_layout = QVBoxLayout(content_area)
        
        stress_info_label = QLabel("Stress/strain analysis will be implemented here")
        stress_info_label.setAlignment(Qt.AlignCenter)
        stress_info_label.setStyleSheet("color: gray; font-style: italic;")
        content_layout.addWidget(stress_info_label)
    
    def setup_3d_visualization_tab(self, side_panel, content_area):
        """Setup the 3D Visualization tab."""
        # Side panel layout
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("3D Visualization")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Visualization group
        vis_group = QGroupBox("3D Controls")
        vis_layout = QVBoxLayout(vis_group)
        
        render_3d_btn = QPushButton("Render 3D Structure")
        vis_layout.addWidget(render_3d_btn)
        
        side_layout.addWidget(vis_group)
        
        # Add stretch
        side_layout.addStretch()
        
        # Content area - placeholder
        content_layout = QVBoxLayout(content_area)
        
        vis_info_label = QLabel("3D visualization will be implemented here")
        vis_info_label.setAlignment(Qt.AlignCenter)
        vis_info_label.setStyleSheet("color: gray; font-style: italic;")
        content_layout.addWidget(vis_info_label)
    
    # === Core Functionality Methods ===
    
    def apply_toolbar_styling(self, toolbar):
        """Apply consistent styling to matplotlib toolbars."""
        if not COMPACT_UI_AVAILABLE:
            # Apply manual styling when compact UI is not available
            toolbar.setMaximumHeight(32)
            toolbar.setMinimumHeight(28)
            
            # Reduce icon size
            original_size = toolbar.iconSize()
            compact_size = original_size * 0.7
            toolbar.setIconSize(compact_size)
            
            # Apply basic styling
            toolbar.setStyleSheet("""
                QToolBar {
                    spacing: 1px;
                    padding: 2px 4px;
                    border: none;
                    background-color: #f8f9fa;
                    border-bottom: 1px solid #dee2e6;
                }
                
                QToolButton {
                    padding: 3px;
                    margin: 1px;
                    border: 1px solid transparent;
                    border-radius: 4px;
                    background-color: transparent;
                    min-width: 20px;
                    min-height: 20px;
                }
                
                QToolButton:hover {
                    background-color: #e9ecef;
                    border: 1px solid #ced4da;
                }
                
                QToolButton:pressed {
                    background-color: #dee2e6;
                    border: 1px solid #adb5bd;
                }
            """)
    
    def load_mineral_database(self):
        """Load the mineral database from available sources."""
        try:
            # Priority order for database loading
            db_sources = [
                ('mineral_modes.pkl', 'mineral_modes'),
                ('raman_database.pkl', 'raman_spectra'),
                ('mineral_database.pkl', 'simple'),
                ('mineral_database.py', 'python_module')
            ]
            
            # Check for database files in current directory
            for db_filename, db_type in db_sources:
                db_path = os.path.join(os.path.dirname(__file__), db_filename)
                if os.path.exists(db_path):
                    success = False
                    
                    if db_type == 'mineral_modes':
                        success = self.load_mineral_modes_database(db_path)
                    elif db_type == 'raman_spectra':
                        success = self.load_raman_spectra_database(db_path)
                    elif db_type == 'simple' and db_path.endswith('.pkl'):
                        success = self.load_simple_pickle_database(db_path)
                    elif db_type == 'python_module':
                        success = self.load_python_module_database(db_path)
                    
                    if success:
                        print(f"✓ Loaded {db_type} database with {len(self.mineral_list)} minerals from {db_filename}")
                        return
            
            # If no database found, create minimal one
            print("⚠ No mineral database found, creating minimal database")
            self.create_minimal_database()
            
        except Exception as e:
            print(f"Error loading mineral database: {e}")
            self.create_minimal_database()
    
    def load_mineral_modes_database(self, db_path):
        """Load the mineral modes database (calculated Raman modes)."""
        try:
            with open(db_path, 'rb') as f:
                raw_database = pickle.load(f)
            
            # Convert mineral modes database to our format
            self.mineral_database = {}
            
            for mineral_name, mineral_data in raw_database.items():
                # Skip metadata entries
                if mineral_name.startswith('__') or not isinstance(mineral_data, dict):
                    continue
                
                # Extract crystal system
                crystal_system = mineral_data.get('crystal_system', 'Unknown')
                space_group = mineral_data.get('space_group', 'Unknown')
                point_group = mineral_data.get('point_group', 'Unknown')
                
                # Get modes using the same method as the mineral modes browser
                modes = self.get_modes_from_mineral_data(mineral_name, mineral_data)
                
                # Convert modes to our raman_modes format
                raman_modes = []
                for mode in modes:
                    if len(mode) >= 3:
                        frequency, symmetry, intensity = mode[0], mode[1], mode[2]
                        
                        # Determine intensity category
                        if intensity > 0.8:
                            intensity_label = 'very_strong'
                        elif intensity > 0.6:
                            intensity_label = 'strong'
                        elif intensity > 0.3:
                            intensity_label = 'medium'
                        elif intensity > 0.1:
                            intensity_label = 'weak'
                        else:
                            intensity_label = 'very_weak'
                        
                        raman_modes.append({
                            'frequency': frequency,
                            'character': symmetry,
                            'intensity': intensity_label,
                            'numerical_intensity': intensity
                        })
                
                # Store in our format
                self.mineral_database[mineral_name] = {
                    'name': mineral_name,
                    'formula': mineral_data.get('formula', 'Unknown'),
                    'crystal_system': crystal_system,
                    'space_group': space_group,
                    'point_group': point_group,
                    'raman_modes': raman_modes,
                    'source': 'mineral_modes_database'
                }
            
            self.mineral_list = list(self.mineral_database.keys())
            return len(self.mineral_database) > 0
            
        except Exception as e:
            print(f"Error loading mineral modes database: {e}")
            return False
    
    def get_modes_from_mineral_data(self, mineral_name, mineral_data):
        """Extract modes from mineral data, handling both modes and phonon_modes."""
        # First try to get existing modes
        if 'modes' in mineral_data and mineral_data['modes']:
            return mineral_data['modes']
        
        # If no modes, try to convert from phonon_modes (same as mineral modes browser)
        if 'phonon_modes' in mineral_data and mineral_data['phonon_modes'] is not None:
            phonon_modes = mineral_data['phonon_modes']
            return self.convert_phonon_modes_to_modes(phonon_modes)
        
        return []
    
    def convert_phonon_modes_to_modes(self, phonon_modes):
        """Convert phonon_modes DataFrame to modes list (same logic as mineral modes browser)."""
        try:
            import pandas as pd
            
            if not isinstance(phonon_modes, pd.DataFrame):
                return []
            
            modes = []
            for _, row in phonon_modes.iterrows():
                # Extract frequency
                frequency = None
                for freq_col in ['Frequency', 'frequency', 'freq', 'Energy', 'energy']:
                    if freq_col in row and pd.notna(row[freq_col]) and str(row[freq_col]).strip():
                        try:
                            frequency = float(row[freq_col])
                            break
                        except (ValueError, TypeError):
                            continue
                
                if frequency is None:
                    continue
                
                # Extract symmetry/activity
                symmetry = str(row.get('Activity', '')).strip()
                if not symmetry:
                    symmetry = str(row.get('Symmetry', '')).strip()
                if not symmetry:
                    symmetry = 'unknown'
                
                # Skip acoustic modes
                if symmetry.lower() in ['ac', 'acoustic', '']:
                    continue
                
                # Extract intensity
                intensity = 1.0  # Default intensity
                for int_col in ['I_Total', 'Intensity', 'intensity', 'I_Parallel', 'I_Perpendicular']:
                    if int_col in row and pd.notna(row[int_col]) and str(row[int_col]).strip():
                        try:
                            intensity_val = float(row[int_col])
                            if intensity_val > 0:
                                intensity = intensity_val
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Normalize very large intensities
                if intensity > 1e10:
                    intensity = intensity / 1e38
                
                # Create mode tuple
                mode = (frequency, symmetry, intensity)
                modes.append(mode)
            
            return modes
            
        except Exception as e:
            print(f"Error converting phonon_modes: {e}")
            return []
    
    def load_raman_spectra_database(self, db_path):
        """Load the main Raman spectra database."""
        try:
            with open(db_path, 'rb') as f:
                raw_database = pickle.load(f)
            
            # Convert to our format
            self.mineral_database = {}
            
            # Handle different database structures
            for key, value in raw_database.items():
                if isinstance(value, dict) and 'wavenumbers' in value and 'intensities' in value:
                    # This is a spectrum entry
                    mineral_name = key
                    wavenumbers = value.get('wavenumbers', [])
                    intensities = value.get('intensities', [])
                    
                    # Generate synthetic modes from the spectrum
                    raman_modes = self.extract_modes_from_spectrum(wavenumbers, intensities)
                    
                    self.mineral_database[mineral_name] = {
                        'name': mineral_name,
                        'formula': value.get('formula', 'Unknown'),
                        'crystal_system': value.get('crystal_system', 'Unknown'),
                        'space_group': value.get('space_group', 'Unknown'),
                        'raman_modes': raman_modes,
                        'source': 'raman_spectra_database',
                        'spectrum_data': {
                            'wavenumbers': wavenumbers,
                            'intensities': intensities
                        }
                    }
            
            self.mineral_list = list(self.mineral_database.keys())
            return len(self.mineral_database) > 0
            
        except Exception as e:
            print(f"Error loading Raman spectra database: {e}")
            return False
    
    def extract_modes_from_spectrum(self, wavenumbers, intensities):
        """Extract peak modes from spectrum data."""
        try:
            if not wavenumbers or not intensities:
                return []
            
            wavenumbers = np.array(wavenumbers)
            intensities = np.array(intensities)
            
            # Find peaks in the spectrum
            peaks, properties = find_peaks(intensities, height=np.max(intensities) * 0.1, distance=10)
            
            raman_modes = []
            for peak_idx in peaks:
                frequency = wavenumbers[peak_idx]
                intensity = intensities[peak_idx]
                
                # Normalize intensity
                norm_intensity = intensity / np.max(intensities)
                
                # Determine intensity label
                if norm_intensity > 0.8:
                    intensity_label = 'very_strong'
                elif norm_intensity > 0.6:
                    intensity_label = 'strong'
                elif norm_intensity > 0.3:
                    intensity_label = 'medium'
                elif norm_intensity > 0.1:
                    intensity_label = 'weak'
                else:
                    intensity_label = 'very_weak'
                
                raman_modes.append({
                    'frequency': frequency,
                    'character': 'A1g',  # Default character
                    'intensity': intensity_label,
                    'numerical_intensity': norm_intensity
                })
            
            return raman_modes
            
        except Exception as e:
            print(f"Error extracting modes from spectrum: {e}")
            return []
    
    def load_simple_pickle_database(self, db_path):
        """Load a simple pickle database."""
        try:
            with open(db_path, 'rb') as f:
                self.mineral_database = pickle.load(f)
            
            self.mineral_list = list(self.mineral_database.keys())
            return len(self.mineral_database) > 0
            
        except Exception as e:
            print(f"Error loading simple pickle database: {e}")
            return False
    
    def load_python_module_database(self, db_path):
        """Load database from Python module."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("mineral_database", db_path)
            mineral_db_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mineral_db_module)
            
            if hasattr(mineral_db_module, 'get_mineral_database'):
                self.mineral_database = mineral_db_module.get_mineral_database()
                self.mineral_list = list(self.mineral_database.keys())
                return len(self.mineral_database) > 0
            
            return False
            
        except Exception as e:
            print(f"Error loading Python module database: {e}")
            return False
    
    def create_minimal_database(self):
        """Create a minimal mineral database if none is found."""
        self.mineral_database = {
            'QUARTZ': {
                'name': 'Quartz',
                'formula': 'SiO2',
                'crystal_system': 'Hexagonal',
                'space_group': 'P3121',
                'raman_modes': [
                    {'frequency': 128, 'character': 'E', 'intensity': 'medium'},
                    {'frequency': 206, 'character': 'A1', 'intensity': 'weak'},
                    {'frequency': 464, 'character': 'A1', 'intensity': 'very_strong'},
                    {'frequency': 696, 'character': 'E', 'intensity': 'weak'},
                    {'frequency': 808, 'character': 'E', 'intensity': 'weak'},
                    {'frequency': 1085, 'character': 'E', 'intensity': 'weak'}
                ]
            },
            'CALCITE': {
                'name': 'Calcite',
                'formula': 'CaCO3',
                'crystal_system': 'Hexagonal',
                'space_group': 'R3c',
                'raman_modes': [
                    {'frequency': 155, 'character': 'Eg', 'intensity': 'medium'},
                    {'frequency': 282, 'character': 'Eg', 'intensity': 'medium'},
                    {'frequency': 714, 'character': 'Eg', 'intensity': 'medium'},
                    {'frequency': 1086, 'character': 'A1g', 'intensity': 'very_strong'}
                ]
            }
        }
        self.mineral_list = list(self.mineral_database.keys())
        print("⚠ Using minimal mineral database")
    
    def update_mineral_lists(self):
        """Update mineral lists in UI components."""
        # Use the comprehensive function that updates all reference mineral selections
        self.update_reference_mineral_selections()
    
    def on_tab_changed(self, index):
        """Handle tab change events."""
        current_tab = self.tab_widget.tabText(index)
        
        # Update plots when switching tabs
        if current_tab == "Spectrum Analysis" and self.current_spectrum is not None:
            self.update_spectrum_plot()
        elif current_tab == "Peak Fitting":
            if self.current_spectrum is not None:
                self.update_peak_fitting_plot()
            # Update reference mineral selections across all subtabs when switching to Peak Fitting
            # This ensures combo boxes are always populated with available minerals
            self.update_reference_mineral_selections()
    
    def on_peak_fitting_subtab_changed(self, index):
        """Handle Peak Fitting subtab change events."""
        if index == 1:  # Assignment tab (index 1)
            # Update reference mineral selections when switching to Assignment subtab
            self.update_reference_mineral_selections()
    
    def on_search_change(self, text):
        """Handle mineral search text changes (legacy method for compatibility)."""
        # Just call the new method with filter
        self.on_search_change_with_filter()
    
    def on_search_change_with_filter(self):
        """Handle search changes with crystal system filter applied."""
        search_text = self.search_entry.text()
        crystal_filter = self.crystal_filter_combo.currentText() if hasattr(self, 'crystal_filter_combo') else "All"
        
        self.search_listbox.clear()
        
        if not search_text or not self.mineral_list:
            return
        
        # Filter minerals based on search text and crystal system
        search_text = search_text.lower()
        matching_minerals = []
        
        for mineral in self.mineral_list:
            mineral_data = self.mineral_database.get(mineral, {})
            
            # Apply crystal system filter first
            if crystal_filter != "All":
                mineral_crystal_system = str(mineral_data.get('crystal_system', 'Unknown'))
                if mineral_crystal_system != crystal_filter:
                    continue
            
            # Search in multiple fields (safely convert to strings)
            search_fields = [
                mineral.lower(),
                str(mineral_data.get('formula', '')).lower(),
                str(mineral_data.get('crystal_system', '')).lower(),
                str(mineral_data.get('space_group', '')).lower()
            ]
            
            # Check if search text matches any field
            if any(search_text in field for field in search_fields):
                # Add mineral info for display (safely convert to strings)
                formula = str(mineral_data.get('formula', 'Unknown'))
                crystal_system = str(mineral_data.get('crystal_system', 'Unknown'))
                num_modes = len(mineral_data.get('raman_modes', []))
                
                # Create clean display text - only show formula if known
                if formula != 'Unknown' and formula.strip():
                    display_text = f"{mineral} ({formula}) - {crystal_system} - {num_modes} modes"
                else:
                    display_text = f"{mineral} - {crystal_system} - {num_modes} modes"
                
                matching_minerals.append((mineral, display_text))
        
        # Sort by mineral name and add to listbox
        matching_minerals.sort(key=lambda x: x[0])
        for mineral, display_text in matching_minerals[:20]:  # Limit to 20 results
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, mineral)  # Store actual mineral name
            self.search_listbox.addItem(item)
    
    def refresh_database(self):
        """Refresh the mineral database."""
        try:
            # Show progress
            progress = QMessageBox(self)
            progress.setWindowTitle("Refreshing Database")
            progress.setText("Reloading mineral database...")
            progress.setStandardButtons(QMessageBox.NoButton)
            progress.show()
            QApplication.processEvents()
            
            # Reload database
            self.load_mineral_database()
            self.update_mineral_lists()
            
            progress.close()
            
            # Show success message
            QMessageBox.information(self, "Success", 
                                  f"Database refreshed successfully!\n"
                                  f"Loaded {len(self.mineral_list)} minerals.")
            
            # Clear search to show all minerals are available
            self.search_entry.clear()
            self.crystal_filter_combo.setCurrentText("All")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error refreshing database: {str(e)}")
    
    def on_mineral_select(self, item):
        """Handle mineral selection from search results."""
        if item:
            # Get actual mineral name from stored data
            selected_mineral = item.data(Qt.UserRole)
            if selected_mineral:
                self.import_selected_mineral(selected_mineral)
    
    def import_selected_mineral(self, mineral_name=None):
        """Import selected mineral spectrum from database."""
        if not mineral_name:
            current_item = self.search_listbox.currentItem()
            if not current_item:
                QMessageBox.warning(self, "Warning", "Please select a mineral from the list.")
                return
            # Get the actual mineral name from stored data (not display text)
            mineral_name = current_item.data(Qt.UserRole)
            if not mineral_name:
                # Fallback: extract mineral name from display text
                display_text = current_item.text()
                mineral_name = display_text.split(' (')[0]  # Extract name before first parenthesis
        
        # Import mineral spectrum from database
        
        try:
            # Generate spectrum from mineral data
            wavenumbers, intensities = self.generate_spectrum_from_mineral(mineral_name)
            
            if wavenumbers is not None and intensities is not None:
                self.imported_spectrum = {
                    'name': mineral_name,
                    'wavenumbers': wavenumbers,
                    'intensities': intensities,
                    'source': 'database'
                }
                
                # Set as reference mineral
                self.selected_reference_mineral = mineral_name
                
                # Update spectrum plot
                self.update_spectrum_plot()
                
                # Update UI elements
                self.update_reference_mineral_selections()
                
                QMessageBox.information(self, "Success", 
                                      f"Imported spectrum for {mineral_name}")
            else:
                QMessageBox.warning(self, "Error", 
                                  f"Could not generate spectrum for {mineral_name}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error importing mineral: {str(e)}")
    
    def generate_spectrum_from_mineral(self, mineral_name):
        """Generate a synthetic Raman spectrum from mineral database."""
        if not self.mineral_database or mineral_name not in self.mineral_database:
            return None, None
        
        try:
            mineral_data = self.mineral_database[mineral_name]
            
            # Check if we have stored spectrum data first
            if 'spectrum_data' in mineral_data:
                spectrum_data = mineral_data['spectrum_data']
                if 'wavenumbers' in spectrum_data and 'intensities' in spectrum_data:
                    return np.array(spectrum_data['wavenumbers']), np.array(spectrum_data['intensities'])
            
            # Otherwise generate from Raman modes
            if 'raman_modes' not in mineral_data or not mineral_data['raman_modes']:
                # Try to get modes from the raw data and convert them
                raw_modes = self.get_modes_from_mineral_data(mineral_name, mineral_data)
                if raw_modes:
                    # Convert to raman_modes format
                    raman_modes = []
                    for mode in raw_modes:
                        if len(mode) >= 3:
                            frequency, symmetry, intensity = mode[0], mode[1], mode[2]
                            
                            # Determine intensity category
                            if intensity > 0.8:
                                intensity_label = 'very_strong'
                            elif intensity > 0.6:
                                intensity_label = 'strong'
                            elif intensity > 0.3:
                                intensity_label = 'medium'
                            elif intensity > 0.1:
                                intensity_label = 'weak'
                            else:
                                intensity_label = 'very_weak'
                            
                            raman_modes.append({
                                'frequency': frequency,
                                'character': symmetry,
                                'intensity': intensity_label,
                                'numerical_intensity': intensity
                            })
                    
                    # Update the mineral data for future use
                    mineral_data['raman_modes'] = raman_modes
                    modes = raman_modes
                else:
                    return None, None
            else:
                modes = mineral_data['raman_modes']
            
            # Validate modes data
            valid_modes = []
            for i, mode in enumerate(modes):
                try:
                    if isinstance(mode, dict):
                        frequency = float(mode.get('frequency', 0))
                        if frequency > 0:  # Skip invalid frequencies
                            valid_modes.append(mode)
                    elif isinstance(mode, (list, tuple)) and len(mode) >= 2:
                        frequency = float(mode[0])
                        if frequency > 0:  # Skip invalid frequencies
                            # Convert tuple/list format to dict
                            intensity = float(mode[2]) if len(mode) > 2 else 1.0
                            valid_mode = {
                                'frequency': frequency,
                                'character': str(mode[1]) if len(mode) > 1 else 'A1g',
                                'numerical_intensity': intensity
                            }
                            valid_modes.append(valid_mode)
                except (ValueError, TypeError, IndexError) as e:
                    continue
            
            if not valid_modes:
                return None, None
            
            # Determine wavenumber range based on the modes
            frequencies = []
            for mode in valid_modes:
                if isinstance(mode, dict):
                    freq = mode.get('frequency', 0)
                else:
                    freq = mode[0] if len(mode) > 0 else 0
                if freq > 0:
                    frequencies.append(freq)
            
            if not frequencies:
                return None, None
            
            min_freq = min(frequencies)
            max_freq = max(frequencies)
            
            # Extend range by 20% on each side
            freq_range = max(max_freq - min_freq, 100)  # Ensure minimum range
            range_start = max(50, min_freq - 0.2 * freq_range)
            range_end = min(4000, max_freq + 0.2 * freq_range)
            
            # Create wavenumber array
            wavenumbers = np.linspace(range_start, range_end, int((range_end - range_start) * 2))
            intensities = np.zeros_like(wavenumbers)
            
            # First, collect all intensities to normalize properly while preserving ratios
            # Filter out ungerade modes for Raman spectra
            mode_data = []
            for mode in valid_modes:
                try:
                    if isinstance(mode, dict):
                        frequency = float(mode.get('frequency', 0))
                        character = mode.get('character', '')
                        
                        # Skip ungerade modes (contain 'u') for Raman spectra
                        if 'u' in str(character).lower():
                            continue
                        
                        # Get intensity - prefer numerical if available
                        if 'numerical_intensity' in mode:
                            intensity = float(mode['numerical_intensity'])
                        else:
                            # Convert intensity label to numerical value
                            intensity_label = mode.get('intensity', 'medium')
                            intensity_map = {
                                'very_weak': 0.1,
                                'weak': 0.3,
                                'medium': 0.6,
                                'strong': 0.8,
                                'very_strong': 1.0
                            }
                            intensity = intensity_map.get(intensity_label, 0.5)
                    else:
                        frequency = float(mode[0])
                        character = str(mode[1]) if len(mode) > 1 else ''
                        
                        # Skip ungerade modes (contain 'u') for Raman spectra
                        if 'u' in character.lower():
                            continue
                            
                        intensity = float(mode[2]) if len(mode) > 2 else 1.0
                    
                    if frequency <= 0 or intensity <= 0:
                        continue
                        
                    mode_data.append((frequency, intensity))
                    
                except (ValueError, TypeError, IndexError) as e:
                    continue
            
            if not mode_data:
                return None, None
            
            # Normalize to the most intense peak (like wurm.info reference)
            intensities_only = [data[1] for data in mode_data]
            max_intensity = max(intensities_only)
            
            # Handle very large intensities by scaling all together, preserving ratios
            if max_intensity > 1e10:
                # Use square root scaling to compress very large ranges while preserving ratios
                intensities_only = [np.sqrt(intensity) for intensity in intensities_only]
                max_intensity = max(intensities_only)
            elif max_intensity > 1000:
                # Use log scaling for moderately large intensities
                intensities_only = [np.log10(intensity + 1) for intensity in intensities_only]
                max_intensity = max(intensities_only)
            
            # Normalize to most intense peak = 1.0, all others relative to it
            if max_intensity > 0:
                intensities_only = [intensity / max_intensity for intensity in intensities_only]
            
            # Add peaks for each Raman mode with preserved ratios
            for i, (frequency, _) in enumerate(mode_data):
                intensity = intensities_only[i]
                
                # Use very narrow peak widths for calculated spectra (like wurm.info)
                width = max(1, 2 + frequency * 0.002)  # Much narrower peaks for calculated data
                
                # Add sharp Lorentzian peak
                peak_intensities = intensity / (1 + ((wavenumbers - frequency) / width) ** 2)
                intensities += peak_intensities
            
            # Check if we generated any intensity
            if np.max(intensities) == 0:
                return None, None
            
            # For calculated spectra, keep clean sharp peaks (minimal baseline/noise)
            # Add very minimal baseline and noise for calculated data
            baseline = 0.01 * np.exp(-wavenumbers / 2000)  # Very small baseline
            noise = np.random.normal(0, 0.002, len(intensities))  # Minimal noise
            intensities = intensities + baseline + noise
            
            # Ensure non-negative
            intensities = np.maximum(intensities, 0)
            
            return wavenumbers, intensities
            
        except Exception as e:
            print(f"Error generating spectrum for {mineral_name}: {e}")
            return None, None
    
    def on_reference_mineral_changed(self, mineral_name):
        """Handle reference mineral selection changes."""
        if mineral_name and mineral_name != self.selected_reference_mineral:
            self.selected_reference_mineral = mineral_name
            
            # Update status
            if hasattr(self, 'reference_status_label'):
                self.reference_status_label.setText("Manually selected")
                self.reference_status_label.setStyleSheet("color: blue; font-style: italic;")
    
    def update_reference_mineral_selections(self):
        """Update reference mineral selections across tabs."""
        # Populate combo boxes with mineral database
        mineral_names = []
        if self.mineral_database:
            mineral_names = sorted(self.mineral_database.keys())
        
        # Update Peak Fitting tab combo box (if it exists)
        if hasattr(self, 'reference_combo'):
            current_selection = self.reference_combo.currentText()
            self.reference_combo.clear()
            self.reference_combo.addItems(mineral_names)
            if self.selected_reference_mineral and self.selected_reference_mineral in mineral_names:
                self.reference_combo.setCurrentText(self.selected_reference_mineral)
            elif current_selection in mineral_names:
                self.reference_combo.setCurrentText(current_selection)
        
        # Update Peak Analysis tab combo box (if it exists)
        if hasattr(self, 'analysis_reference_combo'):
            current_selection = self.analysis_reference_combo.currentText()
            
            # Temporarily disconnect signal to prevent it from overwriting our selected_reference_mineral
            target_mineral = self.selected_reference_mineral  # Store the value we want
            self.analysis_reference_combo.currentTextChanged.disconnect()
            
            self.analysis_reference_combo.clear()
            self.analysis_reference_combo.addItems(mineral_names)
            
            # Reconnect the signal
            self.analysis_reference_combo.currentTextChanged.connect(self.on_reference_mineral_changed)
            
            # Try exact match first
            if target_mineral and target_mineral in mineral_names:
                self.analysis_reference_combo.setCurrentText(target_mineral)
            else:
                # Try fuzzy match for cases like 'CALCITE_4' vs 'CALCITE'
                if target_mineral:
                    # Remove suffixes like '_4', '_R3C', etc.
                    base_name = target_mineral.split('_')[0]
                    
                    # Look for the base name in mineral_names
                    for mineral in mineral_names:
                        if mineral.upper() == base_name.upper():
                            self.analysis_reference_combo.setCurrentText(mineral)
                            break
                    else:
                        # No match found, keep current selection if valid
                        if current_selection in mineral_names:
                            self.analysis_reference_combo.setCurrentText(current_selection)
                elif current_selection in mineral_names:
                    self.analysis_reference_combo.setCurrentText(current_selection)
        
        # Update status labels
        if self.selected_reference_mineral:
            if hasattr(self, 'reference_status_label'):
                self.reference_status_label.setText("Auto-selected from Spectrum Analysis tab")
                self.reference_status_label.setStyleSheet("color: green; font-style: italic;")
            if hasattr(self, 'analysis_reference_status_label'):
                self.analysis_reference_status_label.setText("Auto-selected from Spectrum Analysis tab")
                self.analysis_reference_status_label.setStyleSheet("color: green; font-style: italic;")
    
    # === File Operations ===
    
    def load_spectrum(self):
        """Load spectrum from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Spectrum",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt *.csv *.dat);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Try to load the file
            data = np.loadtxt(file_path, delimiter=None)
            
            if data.shape[1] < 2:
                QMessageBox.warning(self, "Error", "File must contain at least 2 columns (wavenumber, intensity)")
                return
            
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
            
            # Store spectrum
            filename = os.path.basename(file_path)
            self.current_spectrum = {
                'name': filename,
                'wavenumbers': wavenumbers,
                'intensities': intensities,
                'source': 'file'
            }
            self.original_spectrum = self.current_spectrum.copy()
            
            # Update plot
            self.update_spectrum_plot()
            
            QMessageBox.information(self, "Success", f"Loaded spectrum: {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def save_spectrum(self):
        """Save current spectrum to file."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "No spectrum to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spectrum",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            
            # Combine data
            data = np.column_stack((wavenumbers, intensities))
            
            # Save file
            np.savetxt(file_path, data, delimiter='\t', 
                      header='Wavenumber\tIntensity', comments='')
            
            QMessageBox.information(self, "Success", "Spectrum saved successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving file: {str(e)}")
    
    def export_plot(self):
        """Export current plot to image file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Get current tab's figure
            current_tab = self.tab_widget.tabText(self.tab_widget.currentIndex())
            
            if current_tab in self.plot_components and 'fig' in self.plot_components[current_tab]:
                fig = self.plot_components[current_tab]['fig']
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", "Plot exported successfully.")
            else:
                QMessageBox.warning(self, "Warning", "No plot to export.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting plot: {str(e)}")
    
    # === Peak Fitting Methods ===
    
    def toggle_peak_selection(self):
        """Toggle peak selection mode."""
        self.peak_selection_mode = not self.peak_selection_mode
        
        if self.peak_selection_mode:
            self.peak_selection_btn.setText("🚫 Exit Peak Selection")
            self.peak_selection_btn.setStyleSheet("background-color: #ffcccc; font-weight: bold;")
        else:
            self.peak_selection_btn.setText("🎯 Enable Peak Selection")
            self.peak_selection_btn.setStyleSheet("")
    
    def clear_selected_peaks(self):
        """Clear all selected peaks and associated data."""
        self.selected_peaks.clear()
        self.fitted_peaks.clear()
        self.matched_peaks.clear()
        self.peak_labels.clear()
        self.calculated_peaks.clear()
        if hasattr(self, 'calculated_peaks_shifted'):
            delattr(self, 'calculated_peaks_shifted')
        self.update_peak_fitting_plot()
        self.update_fitting_status()
        self.update_peak_count_display()
    
    def update_peak_count_display(self):
        """Update the display showing number of selected peaks."""
        if hasattr(self, 'peak_count_label'):
            count = len(self.selected_peaks)
            if count == 0:
                self.peak_count_label.setText("Selected peaks: 0")
                self.peak_count_label.setStyleSheet("color: #666; font-size: 11px;")
            else:
                self.peak_count_label.setText(f"Selected peaks: {count}")
                self.peak_count_label.setStyleSheet("color: #2E7D32; font-size: 11px; font-weight: bold;")
    
    def on_peak_click(self, event):
        """Handle mouse clicks for peak selection and removal."""
        if not self.peak_selection_mode or not event.inaxes:
            return
        
        if self.current_spectrum is None:
            return
        
        # Get click position
        x_click = event.xdata
        
        if x_click is None:
            return
        
        # Debug: Print event information
        print(f"Click event - key: '{event.key}', button: {event.button}")
        
        # Check for peak removal conditions:
        # 1. Ctrl key is pressed OR
        # 2. Right mouse button (button 3)
        is_ctrl_pressed = (event.key is not None and 
                          ('ctrl' in str(event.key).lower() or 'control' in str(event.key).lower()))
        is_right_click = (event.button == 3)
        
        print(f"Debug - is_ctrl_pressed: {is_ctrl_pressed}, is_right_click: {is_right_click}")
        
        if is_ctrl_pressed or is_right_click:
            # Remove nearest peak within 10 cm⁻¹
            if self.selected_peaks:
                # Find the nearest selected peak
                distances = [abs(peak - x_click) for peak in self.selected_peaks]
                min_distance = min(distances)
                
                # Only remove if within 10 cm⁻¹ tolerance
                if min_distance <= 10:
                    nearest_peak_idx = distances.index(min_distance)
                    removed_peak = self.selected_peaks.pop(nearest_peak_idx)
                    print(f"🗑️ Removed peak at {removed_peak:.1f} cm⁻¹ (clicked at {x_click:.1f} cm⁻¹)")
                else:
                    print(f"❌ No peak within 10 cm⁻¹ of click position {x_click:.1f} cm⁻¹")
            else:
                print("❌ No peaks selected to remove")
        else:
            # Normal click - add peak
            self.selected_peaks.append(x_click)
            print(f"➕ Added peak at {x_click:.1f} cm⁻¹")
        
        # Update plot and peak count
        self.update_peak_fitting_plot()
        self.update_peak_count_display()
    
    def fit_peaks(self):
        """Fit selected peaks with chosen shape."""
        if not self.selected_peaks or self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "Please select peaks first.")
            return
        
        try:
            shape = self.peak_shape_combo.currentText()
            
            # Use processed spectrum if available, otherwise use original
            if self.processed_spectrum is not None:
                wavenumbers = self.processed_spectrum['wavenumbers']
                intensities = self.processed_spectrum['intensities']
            else:
                wavenumbers = self.current_spectrum['wavenumbers']
                intensities = self.current_spectrum['intensities']
            
            self.fitted_peaks.clear()
            range_width = self.fit_range_spin.value()  # Use adjustable range
            
            # Check if multipeak fitting is requested
            if self.multipeak_check.isChecked() and len(self.selected_peaks) > 1:
                self.fit_overlapping_peaks()
                return
            
            for peak_pos in self.selected_peaks:
                # Find peak index
                peak_idx = np.argmin(np.abs(wavenumbers - peak_pos))
                
                # Define fitting range around peak
                mask = np.abs(wavenumbers - peak_pos) <= range_width
                
                if np.sum(mask) < 5:  # Need at least 5 points
                    continue
                
                x_fit = wavenumbers[mask]
                y_fit = intensities[mask]
                
                # Improved initial guess
                peak_idx_local = np.argmin(np.abs(x_fit - peak_pos))
                amplitude = y_fit[peak_idx_local]
                center = peak_pos
                
                # Estimate width from data
                half_max = amplitude / 2
                indices_above_half = np.where(y_fit >= half_max)[0]
                if len(indices_above_half) > 1:
                    width_estimate = (x_fit[indices_above_half[-1]] - x_fit[indices_above_half[0]]) / 2
                    width = max(2, min(width_estimate, 20))  # Reasonable bounds
                else:
                    width = 10
                
                try:
                    if shape == "Lorentzian":
                        popt, pcov = curve_fit(self.lorentzian, x_fit, y_fit, 
                                             p0=[amplitude, center, width])
                    elif shape == "Gaussian":
                        popt, pcov = curve_fit(self.gaussian, x_fit, y_fit, 
                                             p0=[amplitude, center, width])
                    elif shape == "Voigt":
                        popt, pcov = curve_fit(self.voigt, x_fit, y_fit, 
                                             p0=[amplitude, center, width, width])
                    elif shape == "Pseudo-Voigt":
                        popt, pcov = curve_fit(self.pseudo_voigt, x_fit, y_fit, 
                                             p0=[amplitude, center, width, 0.5])
                    
                    # Calculate R-squared for this fit
                    if shape == "Pseudo-Voigt":
                        y_pred = self.pseudo_voigt(x_fit, *popt)
                    elif shape == "Voigt":
                        y_pred = self.voigt(x_fit, *popt)
                    elif shape == "Gaussian":
                        y_pred = self.gaussian(x_fit, *popt)
                    else:
                        y_pred = self.lorentzian(x_fit, *popt)
                    
                    ss_res = np.sum((y_fit - y_pred) ** 2)
                    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Store fitted peak
                    fitted_peak = {
                        'position': popt[1],  # center
                        'amplitude': popt[0],
                        'width': popt[2],
                        'shape': shape,
                        'parameters': popt,
                        'r_squared': r_squared,
                        'covariance': pcov
                    }
                    
                    # Store additional parameters for complex shapes
                    if shape == "Pseudo-Voigt":
                        fitted_peak['fraction'] = popt[3]
                    elif shape == "Voigt":
                        fitted_peak['width_g'] = popt[3]
                    
                    self.fitted_peaks.append(fitted_peak)
                    
                except Exception as fit_error:
                    print(f"Error fitting peak at {peak_pos}: {fit_error}")
                    continue
            
            # Update plot
            self.update_peak_fitting_plot()
            
            avg_r_squared = np.mean([p['r_squared'] for p in self.fitted_peaks]) if self.fitted_peaks else 0
            
            # Update fitting status
            self.update_fitting_status()
            
            QMessageBox.information(self, "Success", 
                                  f"Fitted {len(self.fitted_peaks)} peaks successfully.\n"
                                  f"Average R² = {avg_r_squared:.4f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error fitting peaks: {str(e)}")
    
    @staticmethod
    def lorentzian(x, amplitude, center, width):
        """Lorentzian peak function."""
        return amplitude / (1 + ((x - center) / width) ** 2)
    
    @staticmethod
    def gaussian(x, amplitude, center, width):
        """Gaussian peak function."""
        return amplitude * np.exp(-((x - center) / width) ** 2)
    
    @staticmethod
    def voigt(x, amplitude, center, width_l, width_g):
        """Simplified Voigt profile (convolution of Lorentzian and Gaussian)."""
        # This is a simplified approximation
        lorentz = 1 / (1 + ((x - center) / width_l) ** 2)
        gauss = np.exp(-((x - center) / width_g) ** 2)
        return amplitude * lorentz * gauss
    
    @staticmethod
    def pseudo_voigt(x, amplitude, center, width, fraction):
        """Pseudo-Voigt profile (linear combination of Lorentzian and Gaussian)."""
        lorentz = 1 / (1 + ((x - center) / width) ** 2)
        gauss = np.exp(-((x - center) / width) ** 2)
        return amplitude * (fraction * lorentz + (1 - fraction) * gauss)
    
    def auto_detect_peaks(self):
        """Automatically detect peaks in the current spectrum."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "Please load a spectrum first.")
            return
        
        try:
            # Get the spectrum to analyze (processed if available, otherwise original)
            if self.processed_spectrum is not None:
                wavenumbers = self.processed_spectrum['wavenumbers']
                intensities = self.processed_spectrum['intensities']
            else:
                wavenumbers = self.current_spectrum['wavenumbers']
                intensities = self.current_spectrum['intensities']
            
            # Parameters for peak detection
            height_threshold = np.max(intensities) * self.weak_threshold_spin.value()
            prominence = height_threshold * 0.5
            distance = 10  # Minimum distance between peaks in data points
            
            # Find peaks
            peaks, properties = find_peaks(intensities, 
                                         height=height_threshold,
                                         prominence=prominence,
                                         distance=distance)
            
            # Convert peak indices to wavenumber positions
            self.selected_peaks = [wavenumbers[peak] for peak in peaks]
            
            # Update the plot and peak count
            self.update_peak_fitting_plot()
            self.update_peak_count_display()
            
            QMessageBox.information(self, "Peak Detection", 
                                  f"Found {len(self.selected_peaks)} peaks automatically.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in peak detection: {str(e)}")
    
    def correct_baseline(self):
        """Apply baseline correction to the spectrum."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "Please load a spectrum first.")
            return
        
        try:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            
            # Simple baseline correction using asymmetric least squares
            # This is a simplified version - could be enhanced with more sophisticated methods
            baseline = self.asymmetric_least_squares(intensities)
            corrected_intensities = intensities - baseline
            
            # Ensure no negative values
            corrected_intensities = np.maximum(corrected_intensities, 0)
            
            # Store processed spectrum
            self.processed_spectrum = {
                'wavenumbers': wavenumbers,
                'intensities': corrected_intensities,
                'original_intensities': intensities,
                'baseline': baseline
            }
            
            self.baseline_corrected = True
            self.update_peak_fitting_plot()
            
            QMessageBox.information(self, "Baseline Correction", "Baseline correction applied successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in baseline correction: {str(e)}")
    
    def asymmetric_least_squares(self, y, lam=1e4, p=0.001, niter=10):
        """Asymmetric least squares smoothing for baseline correction."""
        L = len(y)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        for i in range(niter):
            W = diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
    
    def apply_noise_filter(self):
        """Apply noise filtering to the spectrum."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "Please load a spectrum first.")
            return
        
        try:
            # Get spectrum to filter
            if self.processed_spectrum is not None:
                wavenumbers = self.processed_spectrum['wavenumbers']
                intensities = self.processed_spectrum['intensities']
            else:
                wavenumbers = self.current_spectrum['wavenumbers']
                intensities = self.current_spectrum['intensities']
            
            # Apply Savitzky-Golay filter
            window_length = max(5, 2 * self.smoothing_spin.value() + 1)  # Must be odd
            if window_length >= len(intensities):
                window_length = len(intensities) - 1 if len(intensities) % 2 == 0 else len(intensities) - 2
            
            smoothed_intensities = savgol_filter(intensities, window_length, 3)
            
            # Update processed spectrum
            if self.processed_spectrum is not None:
                self.processed_spectrum['intensities'] = smoothed_intensities
            else:
                self.processed_spectrum = {
                    'wavenumbers': wavenumbers,
                    'intensities': smoothed_intensities,
                    'original_intensities': intensities
                }
            
            self.noise_filtered = True
            self.update_peak_fitting_plot()
            
            QMessageBox.information(self, "Noise Filtering", "Noise filtering applied successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in noise filtering: {str(e)}")
    
    def fit_overlapping_peaks(self):
        """Advanced fitting for overlapping peaks with automatic weak peak detection."""
        if not self.selected_peaks or self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "Please select peaks first.")
            return
        
        try:
            # Get spectrum to fit
            if self.processed_spectrum is not None:
                wavenumbers = self.processed_spectrum['wavenumbers']
                intensities = self.processed_spectrum['intensities']
            else:
                wavenumbers = self.current_spectrum['wavenumbers']
                intensities = self.current_spectrum['intensities']
            
            # Find the range that encompasses all selected peaks
            min_peak = min(self.selected_peaks)
            max_peak = max(self.selected_peaks)
            range_width = self.fit_range_spin.value()
            
            # Extend range to include fitting margins
            fit_min = min_peak - range_width
            fit_max = max_peak + range_width
            
            # Create mask for fitting range
            mask = (wavenumbers >= fit_min) & (wavenumbers <= fit_max)
            
            if np.sum(mask) < 10:  # Need sufficient points
                QMessageBox.warning(self, "Warning", "Insufficient data points in fitting range.")
                return
            
            x_fit = wavenumbers[mask]
            y_fit = intensities[mask]
            
            # Auto-detect additional weak peaks in the fitting region
            height_threshold = np.max(y_fit) * self.weak_threshold_spin.value()
            prominence = height_threshold * 0.3  # Lower prominence for weak peaks
            distance = 5  # Minimum distance between peaks in data points
            
            # Find peaks in the fitting region
            peaks_indices, properties = find_peaks(y_fit, 
                                                 height=height_threshold,
                                                 prominence=prominence,
                                                 distance=distance)
            
            # Convert peak indices to wavenumber positions within fitting range
            detected_peaks = [x_fit[peak] for peak in peaks_indices]
            
            # Combine manually selected peaks with automatically detected ones
            all_peaks = list(self.selected_peaks)
            
            # Add detected peaks that are not too close to manually selected ones
            min_separation = 10  # cm⁻¹ minimum separation
            for detected_peak in detected_peaks:
                if not any(abs(detected_peak - selected_peak) < min_separation for selected_peak in self.selected_peaks):
                    all_peaks.append(detected_peak)
            
            # Sort peaks by position
            all_peaks.sort()
            
            # Update selected peaks to include newly detected ones
            original_selected_count = len(self.selected_peaks)
            self.selected_peaks = all_peaks
            self.update_peak_count_display()
            
            print(f"Advanced Multipeak Fit: Found {len(all_peaks) - original_selected_count} additional weak peaks")
            
            # Build multipeak function
            shape = self.peak_shape_combo.currentText()
            n_peaks = len(all_peaks)
            
            if n_peaks < 1:
                QMessageBox.warning(self, "Warning", "No peaks found for fitting.")
                return
            
            if shape == "Lorentzian":
                peak_func = self.lorentzian
                n_params_per_peak = 3
            elif shape == "Gaussian":
                peak_func = self.gaussian
                n_params_per_peak = 3
            elif shape == "Voigt":
                peak_func = self.voigt
                n_params_per_peak = 4
            elif shape == "Pseudo-Voigt":
                peak_func = self.pseudo_voigt
                n_params_per_peak = 4
            
            def multipeak_func(x, *params):
                """Multi-peak function for simultaneous fitting."""
                result = np.zeros_like(x)
                for i in range(n_peaks):
                    start_idx = i * n_params_per_peak
                    if shape == "Pseudo-Voigt":
                        amplitude, center, width, fraction = params[start_idx:start_idx+4]
                        result += self.pseudo_voigt(x, amplitude, center, width, fraction)
                    else:
                        peak_params = params[start_idx:start_idx+n_params_per_peak]
                        result += peak_func(x, *peak_params)
                return result
            
            # Initial parameter guess
            initial_params = []
            for peak_pos in all_peaks:
                # Find approximate amplitude at this position
                peak_idx = np.argmin(np.abs(x_fit - peak_pos))
                amplitude = y_fit[peak_idx]
                
                if shape == "Pseudo-Voigt":
                    initial_params.extend([amplitude, peak_pos, 10, 0.5])  # amplitude, center, width, fraction
                else:
                    if shape == "Voigt":
                        initial_params.extend([amplitude, peak_pos, 10, 10])  # amplitude, center, width_l, width_g
                    else:
                        initial_params.extend([amplitude, peak_pos, 10])  # amplitude, center, width
            
            # Perform fitting with bounds to prevent unrealistic parameters
            # Set reasonable bounds for peak parameters
            lower_bounds = []
            upper_bounds = []
            
            for peak_pos in all_peaks:
                if shape == "Pseudo-Voigt":
                    lower_bounds.extend([0, peak_pos - 50, 1, 0])    # amplitude, center, width, fraction
                    upper_bounds.extend([np.max(y_fit) * 2, peak_pos + 50, 100, 1])
                elif shape == "Voigt":
                    lower_bounds.extend([0, peak_pos - 50, 1, 1])    # amplitude, center, width_l, width_g
                    upper_bounds.extend([np.max(y_fit) * 2, peak_pos + 50, 100, 100])
                else:
                    lower_bounds.extend([0, peak_pos - 50, 1])       # amplitude, center, width
                    upper_bounds.extend([np.max(y_fit) * 2, peak_pos + 50, 100])
            
            # Perform fitting with bounds
            popt, pcov = curve_fit(multipeak_func, x_fit, y_fit, p0=initial_params, 
                                 bounds=(lower_bounds, upper_bounds), maxfev=10000)
            
            # Extract individual peak parameters
            self.fitted_peaks.clear()
            for i, peak_pos in enumerate(all_peaks):
                start_idx = i * n_params_per_peak
                peak_params = popt[start_idx:start_idx+n_params_per_peak]
                
                # Mark if this was a manually selected peak or auto-detected weak peak
                is_manual = i < original_selected_count
                
                fitted_peak = {
                    'position': peak_params[1],  # center
                    'amplitude': peak_params[0],
                    'width': peak_params[2],
                    'shape': shape,
                    'parameters': peak_params,
                    'multipeak_fit': True,
                    'manually_selected': is_manual,
                    'weak_peak_detected': not is_manual
                }
                
                if shape == "Pseudo-Voigt":
                    fitted_peak['fraction'] = peak_params[3]
                elif shape == "Voigt":
                    fitted_peak['width_g'] = peak_params[3]
                
                self.fitted_peaks.append(fitted_peak)
            
            # Calculate R-squared for quality assessment
            y_pred = multipeak_func(x_fit, *popt)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Update plot and status
            self.update_peak_fitting_plot()
            self.update_fitting_status()
            
            # Show fitting results with information about additional peaks found
            additional_peaks = n_peaks - original_selected_count
            if additional_peaks > 0:
                message = (f"Fitted {n_peaks} peaks successfully:\n"
                          f"• {original_selected_count} manually selected peaks\n"
                          f"• {additional_peaks} additional weak peaks detected\n"
                          f"R² = {r_squared:.4f}")
            else:
                message = f"Fitted {n_peaks} overlapping peaks successfully.\nR² = {r_squared:.4f}"
            
            QMessageBox.information(self, "Advanced Multipeak Fitting", message)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in multipeak fitting: {str(e)}")
    
    def assess_peak_quality(self):
        """Assess the quality of fitted peaks."""
        if not self.fitted_peaks:
            QMessageBox.warning(self, "Warning", "Please fit peaks first.")
            return
        
        try:
            quality_report = "Peak Quality Assessment\n" + "="*50 + "\n\n"
            
            for i, peak in enumerate(self.fitted_peaks):
                quality_report += f"Peak {i+1} at {peak['position']:.1f} cm⁻¹:\n"
                quality_report += f"  Amplitude: {peak['amplitude']:.2f}\n"
                quality_report += f"  Width: {peak['width']:.1f} cm⁻¹\n"
                
                # Assess peak quality based on various criteria
                quality_score = 0
                quality_issues = []
                
                # Width assessment
                if peak['width'] < 2:
                    quality_issues.append("Very narrow - may be noise")
                elif peak['width'] > 100:
                    quality_issues.append("Very broad - may be background")
                else:
                    quality_score += 1
                
                # Amplitude assessment
                if peak['amplitude'] < 0:
                    quality_issues.append("Negative amplitude - fit issue")
                else:
                    quality_score += 1
                
                # Signal-to-noise ratio estimation
                if self.current_spectrum is not None:
                    noise_level = np.std(self.current_spectrum['intensities'][:100])  # Estimate from first 100 points
                    snr = peak['amplitude'] / noise_level if noise_level > 0 else float('inf')
                    
                    if snr < 3:
                        quality_issues.append(f"Low S/N ratio ({snr:.1f})")
                    elif snr > 10:
                        quality_score += 1
                        quality_report += f"  S/N Ratio: {snr:.1f} (Good)\n"
                    else:
                        quality_report += f"  S/N Ratio: {snr:.1f}\n"
                
                # Overall quality
                if quality_score >= 2 and not quality_issues:
                    quality_report += f"  Quality: Excellent\n"
                elif quality_score >= 1 and len(quality_issues) <= 1:
                    quality_report += f"  Quality: Good\n"
                else:
                    quality_report += f"  Quality: Poor\n"
                
                if quality_issues:
                    quality_report += f"  Issues: {', '.join(quality_issues)}\n"
                
                quality_report += "\n"
            
            # Show quality report in a dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Peak Quality Assessment")
            dialog.setGeometry(200, 200, 500, 400)
            
            layout = QVBoxLayout(dialog)
            
            text_edit = QTextEdit()
            text_edit.setPlainText(quality_report)
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(dialog.accept)
            layout.addWidget(button_box)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in quality assessment: {str(e)}")
    
    def deconvolve_peaks(self):
        """Deconvolve overlapping peaks using advanced algorithms."""
        if not self.fitted_peaks:
            QMessageBox.warning(self, "Warning", "Please fit peaks first.")
            return
        
        try:
            # This is a placeholder for advanced deconvolution methods
            # In a full implementation, this could include:
            # - Non-negative matrix factorization
            # - Richardson-Lucy deconvolution
            # - Maximum entropy methods
            
            overlapping_pairs = []
            
            # Find potentially overlapping peaks (within 2x width of each other)
            for i, peak1 in enumerate(self.fitted_peaks):
                for j, peak2 in enumerate(self.fitted_peaks[i+1:], i+1):
                    distance = abs(peak1['position'] - peak2['position'])
                    combined_width = peak1['width'] + peak2['width']
                    
                    if distance < combined_width:
                        overlap_factor = 1 - (distance / combined_width)
                        overlapping_pairs.append({
                            'peak1_idx': i,
                            'peak2_idx': j,
                            'overlap_factor': overlap_factor,
                            'peak1_pos': peak1['position'],
                            'peak2_pos': peak2['position']
                        })
            
            if not overlapping_pairs:
                QMessageBox.information(self, "Deconvolution", "No overlapping peaks detected.")
                return
            
            # Report overlapping peaks
            report = "Overlapping Peaks Detected:\n" + "="*40 + "\n\n"
            
            for pair in overlapping_pairs:
                report += f"Peak at {pair['peak1_pos']:.1f} cm⁻¹ overlaps with peak at {pair['peak2_pos']:.1f} cm⁻¹\n"
                report += f"Overlap factor: {pair['overlap_factor']:.2f}\n\n"
            
            report += "Recommendation: Use 'Advanced Multipeak Fit' for better separation of overlapping peaks."
            
            QMessageBox.information(self, "Peak Deconvolution", report)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in deconvolution: {str(e)}")
    
    def export_fit_parameters(self):
        """Export fitted peak parameters to a file."""
        if not self.fitted_peaks:
            QMessageBox.warning(self, "Warning", "No fitted peaks to export.")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Peak Parameters",
                QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
                "CSV files (*.csv);;Text files (*.txt);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            # Prepare data for export
            with open(file_path, 'w') as f:
                # Write header
                f.write("Peak,Position(cm-1),Amplitude,Width(cm-1),Shape,R-squared,Additional_Parameters\n")
                
                # Write peak data
                for i, peak in enumerate(self.fitted_peaks):
                    additional_params = ""
                    if 'fraction' in peak:
                        additional_params = f"Fraction={peak['fraction']:.3f}"
                    elif 'width_g' in peak:
                        additional_params = f"Width_G={peak['width_g']:.2f}"
                    
                    # Calculate R-squared for this peak (simplified)
                    r_squared = "N/A"  # Would need individual fit data for accurate calculation
                    
                    f.write(f"{i+1},{peak['position']:.2f},{peak['amplitude']:.3f},"
                           f"{peak['width']:.2f},{peak['shape']},{r_squared},{additional_params}\n")
            
            QMessageBox.information(self, "Export Complete", f"Peak parameters exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting parameters: {str(e)}")
    
    def update_fitting_status(self):
        """Update the fitting status label in the Peak Analysis tab."""
        if hasattr(self, 'fitting_status_label'):
            if self.fitted_peaks:
                n_peaks = len(self.fitted_peaks)
                avg_r_squared = np.mean([p.get('r_squared', 0) for p in self.fitted_peaks])
                
                # Color code based on quality
                if avg_r_squared > 0.95:
                    color = "green"
                    quality = "Excellent"
                elif avg_r_squared > 0.9:
                    color = "blue"
                    quality = "Good"
                elif avg_r_squared > 0.8:
                    color = "orange"
                    quality = "Fair"
                else:
                    color = "red"
                    quality = "Poor"
                
                status_text = f"{n_peaks} peaks fitted | Avg R² = {avg_r_squared:.3f} ({quality})"
                self.fitting_status_label.setText(status_text)
                self.fitting_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            else:
                self.fitting_status_label.setText("No peaks fitted")
                self.fitting_status_label.setStyleSheet("color: gray; font-style: italic;")
    
    def update_matching_tolerance(self, value):
        """Update the peak matching tolerance."""
        self.peak_matching_tolerance = value
        
        # Re-match peaks if we have data
        if self.fitted_peaks and self.calculated_peaks:
            self.match_peaks_with_reference()
    
    def match_peaks_with_reference(self):
        """Match fitted peaks with reference mineral peaks."""
        if not self.fitted_peaks:
            QMessageBox.warning(self, "Warning", "Please fit peaks first.")
            return
        
        if not self.selected_reference_mineral:
            QMessageBox.warning(self, "Warning", "Please select a reference mineral first.")
            return
        
        try:
            # Get reference mineral data
            mineral_data = self.mineral_database.get(self.selected_reference_mineral)
            if not mineral_data or 'raman_modes' not in mineral_data:
                QMessageBox.warning(self, "Warning", "No Raman modes found for selected reference mineral.")
                return
            
            # Extract calculated peaks from reference mineral
            self.calculated_peaks = []
            
            # First try from raman_modes
            modes_from_list = mineral_data['raman_modes']
            for mode in modes_from_list:
                freq = mode.get('frequency', 0)
                character = mode.get('character', 'Unknown')
                intensity = mode.get('numerical_intensity', mode.get('intensity', 'medium'))
                
                # Skip modes with 'u' (ungerade) - these are not Raman active
                if 'u' in character.lower():
                    continue
                
                # Skip invalid frequencies
                if freq <= 0:
                    continue
                
                # Convert intensity label to numerical if needed
                if isinstance(intensity, str):
                    intensity_map = {
                        'very_weak': 0.1, 'weak': 0.3, 'medium': 0.6, 
                        'strong': 0.8, 'very_strong': 1.0
                    }
                    intensity = intensity_map.get(intensity, 0.5)
                
                # Ensure intensity is reasonable
                if intensity <= 0:
                    intensity = 0.1
                elif intensity > 10:  # Cap very large intensities
                    intensity = 1.0
                
                self.calculated_peaks.append({
                    'frequency': freq,
                    'character': character,
                    'intensity': intensity
                })
            
            # If we have fewer than expected peaks, also try to extract from raw data
            if len(self.calculated_peaks) < 5:  # Threshold to trigger raw data extraction
                try:
                    raw_modes = self.get_modes_from_mineral_data(self.selected_reference_mineral, mineral_data)
                    if raw_modes:
                        print(f"Found {len(raw_modes)} raw modes for {self.selected_reference_mineral}")
                        for mode in raw_modes:
                            if len(mode) >= 3:
                                frequency, symmetry, intensity = mode[0], mode[1], mode[2]
                                
                                # Skip modes with 'u' (ungerade) - these are not Raman active  
                                if 'u' in str(symmetry).lower():
                                    continue
                                
                                # Skip invalid frequencies
                                if frequency <= 0:
                                    continue
                                
                                # Check if we already have this frequency
                                existing_freqs = [p['frequency'] for p in self.calculated_peaks]
                                if any(abs(frequency - ef) < 5 for ef in existing_freqs):
                                    continue  # Skip duplicates
                                
                                # Normalize very large intensities
                                if intensity > 1e10:
                                    intensity = intensity / 1e38
                                elif intensity > 1000:
                                    intensity = np.log10(intensity + 1) / 10
                                
                                self.calculated_peaks.append({
                                    'frequency': frequency,
                                    'character': str(symmetry),
                                    'intensity': float(intensity),
                                    'source': 'raw_modes'
                                })
                except Exception as e:
                    print(f"Error extracting raw modes: {e}")
            
            # Perform peak matching
            self.matched_peaks = []
            self.peak_labels = {}
            
            tolerance = self.peak_matching_tolerance
            
            # For each fitted peak, find the closest calculated peak
            for fitted_peak in self.fitted_peaks:
                fitted_freq = fitted_peak['position']
                best_match = None
                best_distance = float('inf')
                
                for calc_peak in self.calculated_peaks:
                    calc_freq = calc_peak['frequency']
                    distance = abs(fitted_freq - calc_freq)
                    
                    if distance < tolerance and distance < best_distance:
                        best_distance = distance
                        best_match = calc_peak
                
                if best_match:
                    # Create a matched peak pair
                    matched_pair = {
                        'experimental_peak': fitted_peak,
                        'calculated_peak': best_match,
                        'frequency_shift': fitted_freq - best_match['frequency'],
                        'match_quality': 1.0 / (1.0 + best_distance)  # Higher is better
                    }
                    self.matched_peaks.append(matched_pair)
                    
                    # Store peak label
                    self.peak_labels[fitted_freq] = {
                        'character': best_match['character'],
                        'calculated_freq': best_match['frequency'],
                        'shift': fitted_freq - best_match['frequency']
                    }
            
            # Update plot to show matches
            self.update_peak_fitting_plot()
            
            # Show results with detailed diagnostics
            matched_count = len(self.matched_peaks)
            total_fitted = len(self.fitted_peaks)
            total_calc_modes = len(mineral_data.get('raman_modes', []))
            raman_active_modes = len(self.calculated_peaks)
            filtered_modes = total_calc_modes - raman_active_modes
            
            # Find unmatched experimental peaks
            unmatched_peaks = []
            for fitted_peak in self.fitted_peaks:
                fitted_freq = fitted_peak['position']
                is_matched = any(match['experimental_peak']['position'] == fitted_freq 
                               for match in self.matched_peaks)
                if not is_matched:
                    unmatched_peaks.append(fitted_freq)
            
            message = f"Peak Matching Results:\n"
            message += f"• Matched {matched_count} out of {total_fitted} fitted peaks\n"
            message += f"• Using tolerance: ±{tolerance} cm⁻¹\n"
            message += f"• Total calculated modes: {total_calc_modes}\n"
            message += f"• Raman-active modes: {raman_active_modes}\n"
            if filtered_modes > 0:
                message += f"• Filtered out {filtered_modes} ungerade ('u') modes (not Raman active)\n"
            
            if unmatched_peaks:
                message += f"\n⚠ Unmatched experimental peaks:\n"
                for freq in unmatched_peaks:
                    # Find closest calculated mode
                    closest_calc = None
                    min_distance = float('inf')
                    for calc_peak in self.calculated_peaks:
                        distance = abs(freq - calc_peak['frequency'])
                        if distance < min_distance:
                            min_distance = distance
                            closest_calc = calc_peak
                    
                    if closest_calc:
                        message += f"  • {freq:.1f} cm⁻¹ (closest calc: {closest_calc['frequency']:.1f} cm⁻¹, "
                        message += f"distance: {min_distance:.1f} cm⁻¹, character: {closest_calc['character']})\n"
                    else:
                        message += f"  • {freq:.1f} cm⁻¹ (no calculated modes found)\n"
            
            QMessageBox.information(self, "Peak Matching Complete", message)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error matching peaks: {str(e)}")
    
    def auto_assign_peak_labels(self):
        """Automatically assign labels to peaks and shift calculated peaks to match experimental ones."""
        if not self.matched_peaks:
            QMessageBox.warning(self, "Warning", "Please match peaks with reference first.")
            return
        
        try:
            # Apply frequency shifts to calculated peaks to match experimental ones
            shifted_calculated_peaks = []
            
            for match in self.matched_peaks:
                exp_peak = match['experimental_peak']
                calc_peak = match['calculated_peak']
                shift = match['frequency_shift']
                
                # Create shifted calculated peak
                shifted_peak = calc_peak.copy()
                shifted_peak['original_frequency'] = calc_peak['frequency']
                shifted_peak['frequency'] = exp_peak['position']  # Move to experimental position
                shifted_peak['frequency_shift'] = shift
                
                shifted_calculated_peaks.append(shifted_peak)
            
            # Store shifted peaks
            self.calculated_peaks_shifted = shifted_calculated_peaks
            
            # Update plot with labels and shifted peaks
            self.update_peak_fitting_plot()
            
            QMessageBox.information(self, "Auto-Assignment Complete", 
                                  f"Successfully assigned labels to {len(shifted_calculated_peaks)} peaks.\n"
                                  f"Calculated peaks have been shifted to match experimental positions.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in auto-assignment: {str(e)}")
    
    def show_peak_assignments(self):
        """Show a dialog with detailed peak assignments."""
        if not self.matched_peaks:
            QMessageBox.warning(self, "Warning", "No peak assignments available. Please match peaks first.")
            return
        
        # Create assignment dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Peak Assignments")
        dialog.setModal(True)
        dialog.resize(700, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title_label = QLabel("Peak Assignments and Character Labels")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create table for assignments
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Experimental (cm⁻¹)", 
            "Calculated (cm⁻¹)", 
            "Shift (cm⁻¹)", 
            "Character", 
            "Match Quality",
            "Intensity"
        ])
        
        # Populate table
        table.setRowCount(len(self.matched_peaks))
        for row, match in enumerate(self.matched_peaks):
            exp_peak = match['experimental_peak']
            calc_peak = match['calculated_peak']
            
            # Experimental frequency
            table.setItem(row, 0, QTableWidgetItem(f"{exp_peak['position']:.1f}"))
            
            # Calculated frequency
            table.setItem(row, 1, QTableWidgetItem(f"{calc_peak['frequency']:.1f}"))
            
            # Frequency shift
            shift = match['frequency_shift']
            shift_item = QTableWidgetItem(f"{shift:+.1f}")
            if abs(shift) > 20:  # Highlight large shifts
                shift_item.setBackground(QColor("#ffcccc"))
            table.setItem(row, 2, shift_item)
            
            # Character
            character_item = QTableWidgetItem(calc_peak['character'])
            character_item.setFont(QFont("Arial", 10, QFont.Bold))
            table.setItem(row, 3, character_item)
            
            # Match quality
            quality = match['match_quality']
            quality_item = QTableWidgetItem(f"{quality:.3f}")
            if quality > 0.8:
                quality_item.setBackground(QColor("#ccffcc"))
            elif quality < 0.5:
                quality_item.setBackground(QColor("#ffcccc"))
            table.setItem(row, 4, quality_item)
            
            # Intensity
            intensity = calc_peak.get('intensity', 0)
            table.setItem(row, 5, QTableWidgetItem(f"{intensity:.3f}"))
        
        # Resize columns to content
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # Summary info
        summary_text = f"Reference Mineral: {self.selected_reference_mineral}\n"
        summary_text += f"Total Matches: {len(self.matched_peaks)}\n"
        summary_text += f"Matching Tolerance: ±{self.peak_matching_tolerance} cm⁻¹"
        
        summary_label = QLabel(summary_text)
        summary_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(summary_label)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def show_raman_activity_info(self):
        """Show information about Raman activity filtering."""
        if not self.selected_reference_mineral:
            QMessageBox.warning(self, "Warning", "Please select a reference mineral first.")
            return
        
        mineral_data = self.mineral_database.get(self.selected_reference_mineral)
        if not mineral_data or 'raman_modes' not in mineral_data:
            return
        
        # Analyze all modes
        all_modes = mineral_data['raman_modes']
        raman_active = []
        filtered_out = []
        
        for mode in all_modes:
            character = mode.get('character', 'Unknown')
            freq = mode.get('frequency', 0)
            
            if 'u' in character.lower():
                filtered_out.append((freq, character, "Contains 'u' (ungerade)"))
            elif freq <= 0:
                filtered_out.append((freq, character, "Invalid frequency"))
            else:
                raman_active.append((freq, character))
        
        # Create info dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Raman Activity Analysis")
        dialog.setModal(True)
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Summary
        summary_text = f"Reference Mineral: {self.selected_reference_mineral}\n"
        summary_text += f"Total modes: {len(all_modes)}\n"
        summary_text += f"Raman-active modes: {len(raman_active)}\n"
        summary_text += f"Filtered out: {len(filtered_out)}"
        
        summary_label = QLabel(summary_text)
        summary_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(summary_label)
        
        # Filtered modes table
        if filtered_out:
            layout.addWidget(QLabel("Filtered Out Modes (Not Raman Active):"))
            filter_text = QTextEdit()
            filter_text.setMaximumHeight(100)
            filter_content = ""
            for freq, char, reason in filtered_out:
                filter_content += f"{freq:.1f} cm⁻¹ - {char} ({reason})\n"
            filter_text.setPlainText(filter_content)
            layout.addWidget(filter_text)
        
        # Raman active modes
        layout.addWidget(QLabel("Raman-Active Modes:"))
        active_text = QTextEdit()
        active_content = ""
        for freq, char in sorted(raman_active):
            active_content += f"{freq:.1f} cm⁻¹ - {char}\n"
        active_text.setPlainText(active_content)
        layout.addWidget(active_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def debug_calculated_modes(self):
        """Debug calculated modes to understand matching issues."""
        if not self.selected_reference_mineral:
            QMessageBox.warning(self, "Warning", "Please select a reference mineral first.")
            return
        
        mineral_data = self.mineral_database.get(self.selected_reference_mineral)
        if not mineral_data or 'raman_modes' not in mineral_data:
            return
        
        # Create debug dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Debug Calculated Modes")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Title and summary
        title_label = QLabel(f"Debug: {self.selected_reference_mineral}")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)
        
        # Create table for all modes
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Frequency (cm⁻¹)", 
            "Character", 
            "Intensity",
            "Status",
            "Distance to 1085",
            "Notes"
        ])
        
        # Analyze all modes
        all_modes = mineral_data['raman_modes']
        table.setRowCount(len(all_modes))
        
        target_freq = 1085.0  # The missing peak
        
        for row, mode in enumerate(all_modes):
            freq = mode.get('frequency', 0)
            character = mode.get('character', 'Unknown')
            intensity = mode.get('numerical_intensity', mode.get('intensity', 'medium'))
            
            # Convert intensity if needed
            if isinstance(intensity, str):
                intensity_map = {
                    'very_weak': 0.1, 'weak': 0.3, 'medium': 0.6, 
                    'strong': 0.8, 'very_strong': 1.0
                }
                intensity = intensity_map.get(intensity, 0.5)
            
            # Calculate distance to target
            distance = abs(freq - target_freq)
            
            # Determine status
            status = "Active"
            notes = ""
            if 'u' in character.lower():
                status = "Filtered (ungerade)"
                notes = "Contains 'u'"
            elif freq <= 0:
                status = "Filtered (invalid freq)"
                notes = "freq <= 0"
            elif intensity <= 0:
                notes = "Zero intensity"
            
            # Fill table
            table.setItem(row, 0, QTableWidgetItem(f"{freq:.1f}"))
            table.setItem(row, 1, QTableWidgetItem(character))
            table.setItem(row, 2, QTableWidgetItem(f"{intensity:.3f}"))
            
            status_item = QTableWidgetItem(status)
            if status != "Active":
                status_item.setBackground(QColor("#ffcccc"))
            table.setItem(row, 3, status_item)
            
            distance_item = QTableWidgetItem(f"{distance:.1f}")
            if distance < 50:  # Within tolerance
                distance_item.setBackground(QColor("#ffffcc"))
            table.setItem(row, 4, distance_item)
            
            table.setItem(row, 5, QTableWidgetItem(notes))
        
        # Sort by frequency
        table.sortItems(0, Qt.AscendingOrder)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # Summary info
        raman_active_count = sum(1 for mode in all_modes 
                                if 'u' not in mode.get('character', '').lower() 
                                and mode.get('frequency', 0) > 0)
        
        summary_text = f"Total modes: {len(all_modes)}\n"
        summary_text += f"Raman-active modes: {raman_active_count}\n"
        summary_text += f"Looking for modes near 1085 cm⁻¹ (±50 cm⁻¹ tolerance)\n"
        
        # Find modes in range
        modes_in_range = [mode for mode in all_modes 
                         if abs(mode.get('frequency', 0) - target_freq) < 50 
                         and 'u' not in mode.get('character', '').lower()
                         and mode.get('frequency', 0) > 0]
        
        if modes_in_range:
            summary_text += f"Found {len(modes_in_range)} Raman-active modes in range:\n"
            for mode in modes_in_range:
                summary_text += f"  • {mode.get('frequency', 0):.1f} cm⁻¹ ({mode.get('character', 'Unknown')})\n"
        else:
            summary_text += "❌ No Raman-active modes found in ±50 cm⁻¹ range!"
        
        summary_label = QLabel(summary_text)
        summary_label.setStyleSheet("background-color: #f8f9fa; padding: 10px; border-radius: 5px;")
        layout.addWidget(summary_label)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    # === Plot Update Methods ===
    
    def update_spectrum_plot(self):
        """Update the spectrum analysis plot."""
        if "Spectrum Analysis" not in self.plot_components:
            return
        
        components = self.plot_components["Spectrum Analysis"]
        ax = components['ax']
        canvas = components['canvas']
        
        # Clear plot
        ax.clear()
        
        has_data = False
        
        # Plot current spectrum
        if self.current_spectrum is not None:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            
            ax.plot(wavenumbers, intensities, 'b-', linewidth=2, 
                   label=f"Current: {self.current_spectrum['name']}")
            has_data = True
        
        # Plot imported spectrum
        if self.imported_spectrum is not None:
            wavenumbers = self.imported_spectrum['wavenumbers']
            intensities = self.imported_spectrum['intensities'].copy()  # Make a copy to avoid modifying original
            
            # Smart scaling to prevent wild normalization
            if self.current_spectrum is not None:
                current_max = np.max(self.current_spectrum['intensities'])
                imported_max = np.max(intensities)
                if imported_max > 0 and current_max > 0:
                    ratio = current_max / imported_max
                    # Only scale if there's a significant difference (more than 10x but less than 1000x)
                    if 10 < ratio < 1000:
                        intensities = intensities * ratio
                    elif ratio >= 1000:
                        # Cap very large scaling to prevent crazy normalization
                        intensities = intensities * (current_max * 0.5 / imported_max)
                    elif ratio <= 0.1:
                        # Cap very small scaling 
                        intensities = intensities * (current_max * 0.5 / imported_max)
            
            ax.plot(wavenumbers, intensities, 'r-', linewidth=2, alpha=0.7,
                   label=f"Imported: {self.imported_spectrum['name']}")
            has_data = True
        
        # Configure plot
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Raman Spectrum Analysis")
        ax.grid(True, alpha=0.3)
        
        if has_data:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Load a spectrum or import from database',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.6)
        
        canvas.draw()
    
    def update_peak_fitting_plot(self):
        """Update the peak fitting plot with peak matching and labeling."""
        if "Peak Fitting" not in self.plot_components:
            return
        
        components = self.plot_components["Peak Fitting"]
        ax = components['ax']
        canvas = components['canvas']
        
        # Clear plot
        ax.clear()
        
        has_data = False
        
        # Plot spectrum
        if self.current_spectrum is not None:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            
            # Plot original spectrum
            ax.plot(wavenumbers, intensities, 'b-', linewidth=1, alpha=0.5, label='Original Spectrum')
            has_data = True
            
            # Plot processed spectrum if available
            if self.processed_spectrum is not None:
                proc_wavenumbers = self.processed_spectrum['wavenumbers']
                proc_intensities = self.processed_spectrum['intensities']
                ax.plot(proc_wavenumbers, proc_intensities, 'g-', linewidth=1.5, alpha=0.8, label='Processed Spectrum')
                
                # Plot baseline if available
                if 'baseline' in self.processed_spectrum:
                    baseline = self.processed_spectrum['baseline']
                    ax.plot(wavenumbers, baseline, 'r--', linewidth=1, alpha=0.6, label='Baseline')
                
                # Use processed spectrum for peak operations
                wavenumbers = proc_wavenumbers
                intensities = proc_intensities
            
            # Plot selected peaks
            for peak_pos in self.selected_peaks:
                peak_idx = np.argmin(np.abs(wavenumbers - peak_pos))
                ax.axvline(x=peak_pos, color='red', linestyle='--', alpha=0.7)
                ax.plot(peak_pos, intensities[peak_idx], 'ro', markersize=8)
            
            # Plot fitted peaks
            if self.fitted_peaks:
                x_fit = np.linspace(np.min(wavenumbers), np.max(wavenumbers), 1000)
                
                for i, peak in enumerate(self.fitted_peaks):
                    if peak['shape'] == "Lorentzian":
                        y_fit = self.lorentzian(x_fit, *peak['parameters'])
                    elif peak['shape'] == "Gaussian":
                        y_fit = self.gaussian(x_fit, *peak['parameters'])
                    elif peak['shape'] == "Voigt":
                        y_fit = self.voigt(x_fit, *peak['parameters'])
                    elif peak['shape'] == "Pseudo-Voigt":
                        y_fit = self.pseudo_voigt(x_fit, *peak['parameters'])
                    
                    # Use different colors for multipeak fits
                    if peak.get('multipeak_fit', False):
                        color = plt.cm.Set1(i % 9)  # Cycle through colors
                        alpha = 0.9
                    else:
                        color = 'green'
                        alpha = 0.8
                    
                    ax.plot(x_fit, y_fit, color=color, linewidth=2, alpha=alpha)
                    ax.axvline(x=peak['position'], color=color, linestyle='-', alpha=0.5)
                    
                    # Add R-squared annotation for high-quality fits
                    if peak.get('r_squared', 0) > 0.95:
                        ax.annotate(f"R²={peak['r_squared']:.3f}", 
                                   xy=(peak['position'], peak['amplitude']), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.7)
            
            # Plot calculated peaks (original positions) if we have them
            if hasattr(self, 'calculated_peaks') and self.calculated_peaks:
                max_intensity = np.max(intensities)
                # Normalize calculated intensities to a reasonable scale
                calc_intensities = [peak['intensity'] for peak in self.calculated_peaks]
                if calc_intensities:
                    max_calc_intensity = max(calc_intensities)
                    scale_factor = (max_intensity * 0.3) / max_calc_intensity  # Scale to 30% of plot height
                    
                    for calc_peak in self.calculated_peaks:
                        freq = calc_peak['frequency']
                        # Only show if within plot range
                        if np.min(wavenumbers) <= freq <= np.max(wavenumbers):
                            height = calc_peak['intensity'] * scale_factor
                            ax.axvline(x=freq, color='orange', linestyle=':', alpha=0.6, linewidth=1)
                            ax.plot(freq, height, 'o', color='orange', markersize=4, alpha=0.7)
            
            # Plot matched peaks with labels and connections
            if self.matched_peaks:
                max_intensity = np.max(intensities)
                
                # Calculate consistent scale factor for calculated peaks
                calc_intensities = [match['calculated_peak']['intensity'] for match in self.matched_peaks]
                if calc_intensities:
                    max_calc_intensity = max(calc_intensities)
                    calc_scale_factor = (max_intensity * 0.3) / max_calc_intensity
                
                for match in self.matched_peaks:
                    exp_peak = match['experimental_peak']
                    calc_peak = match['calculated_peak']
                    exp_freq = exp_peak['position']
                    calc_freq = calc_peak['frequency']
                    character = calc_peak['character']
                    
                    # Find experimental intensity at this position
                    exp_idx = np.argmin(np.abs(wavenumbers - exp_freq))
                    exp_intensity = intensities[exp_idx]
                    
                    # Draw connection line between calculated and experimental positions
                    calc_height = calc_peak['intensity'] * calc_scale_factor
                    ax.plot([calc_freq, exp_freq], [calc_height, exp_intensity], 
                           'r--', alpha=0.5, linewidth=1)
                    
                    # Add character label above the experimental peak
                    label_height = exp_intensity + max_intensity * 0.05
                    ax.text(exp_freq, label_height, character, 
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                    
                    # Mark matched experimental peak with special marker
                    ax.plot(exp_freq, exp_intensity, 's', color='red', markersize=8, 
                           markerfacecolor='yellow', markeredgecolor='red', markeredgewidth=2)
            
            # Plot shifted calculated peaks if available
            if hasattr(self, 'calculated_peaks_shifted') and self.calculated_peaks_shifted:
                max_intensity = np.max(intensities)
                
                # Calculate consistent scale factor for shifted peaks
                shifted_intensities = [peak['intensity'] for peak in self.calculated_peaks_shifted]
                if shifted_intensities:
                    max_shifted_intensity = max(shifted_intensities)
                    shifted_scale_factor = (max_intensity * 0.25) / max_shifted_intensity
                
                for shifted_peak in self.calculated_peaks_shifted:
                    freq = shifted_peak['frequency']  # This is the shifted position
                    original_freq = shifted_peak['original_frequency']
                    character = shifted_peak['character']
                    
                    # Mark the shifted position
                    height = shifted_peak['intensity'] * shifted_scale_factor
                    ax.axvline(x=freq, color='purple', linestyle='-', alpha=0.8, linewidth=2)
                    ax.plot(freq, height, 'D', color='purple', markersize=6)
                    
                    # Show shift information
                    shift = freq - original_freq
                    ax.annotate(f'Δ={shift:+.1f}', xy=(freq, height), 
                               xytext=(5, 10), textcoords='offset points',
                               fontsize=8, color='purple',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='lavender', alpha=0.8))
        
        # Configure plot
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Peak Fitting Analysis with Character Assignment")
        ax.grid(True, alpha=0.3)
        
        # Create custom legend
        legend_elements = []
        if has_data:
            legend_elements.append(Line2D([0], [0], color='blue', alpha=0.5, label='Original Spectrum'))
            
            if self.processed_spectrum is not None:
                legend_elements.append(Line2D([0], [0], color='green', linewidth=1.5, alpha=0.8, label='Processed Spectrum'))
                
                if 'baseline' in self.processed_spectrum:
                    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', alpha=0.6, label='Baseline'))
            
            if self.fitted_peaks:
                legend_elements.append(Line2D([0], [0], color='green', linewidth=2, label='Fitted Peaks'))
            
            if hasattr(self, 'calculated_peaks') and self.calculated_peaks:
                legend_elements.append(Line2D([0], [0], color='orange', linestyle=':', label='Calculated Peaks (Original)'))
            
            if self.matched_peaks:
                legend_elements.append(Line2D([0], [0], marker='s', color='red', 
                                                markersize=8, markerfacecolor='yellow', 
                                                markeredgecolor='red', linestyle='None',
                                                label='Matched & Labeled Peaks'))
            
            if hasattr(self, 'calculated_peaks_shifted') and self.calculated_peaks_shifted:
                legend_elements.append(Line2D([0], [0], color='purple', linewidth=2, label='Shifted Calculated Peaks'))
            
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Load a spectrum to begin peak fitting',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.6)
        
        canvas.draw()
    
    # === Placeholder methods for complex functionality ===
    
    def load_cif_file(self):
        """Load crystal structure from CIF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load CIF File",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "CIF files (*.cif);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            if PYMATGEN_AVAILABLE:
                # Use pymatgen for professional CIF parsing
                self.load_cif_with_pymatgen(file_path)
            else:
                # Use simplified CIF parser
                self.load_cif_simple(file_path)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading CIF file: {str(e)}")
    
    def load_cif_with_pymatgen(self, file_path):
        """Load CIF using pymatgen library."""
        try:
            # Parse CIF file
            parser = CifParser(file_path)
            structures = parser.parse_structures(primitive=False)  # Use conventional cell, not primitive
            
            if not structures:
                QMessageBox.warning(self, "Warning", "No structures found in CIF file.")
                return
            
            # Take the first structure
            structure = structures[0]
            
            # Get crystal system using SpacegroupAnalyzer
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(structure)
                crystal_system = sga.get_crystal_system()
                space_group = sga.get_space_group_symbol()
            except Exception as e:
                print(f"Warning: Could not determine crystal system: {e}")
                # Fallback to lattice-based determination
                crystal_system = self.determine_crystal_system({
                    'a': structure.lattice.a,
                    'b': structure.lattice.b,
                    'c': structure.lattice.c,
                    'alpha': structure.lattice.alpha,
                    'beta': structure.lattice.beta,
                    'gamma': structure.lattice.gamma
                })
                try:
                    space_group = structure.get_space_group_info()[1]
                except:
                    space_group = "Unknown"
            
            # Extract crystal information
            crystal_info = {
                'formula': structure.formula,
                'space_group': space_group,
                'crystal_system': crystal_system,
                'lattice_params': {
                    'a': structure.lattice.a,
                    'b': structure.lattice.b,
                    'c': structure.lattice.c,
                    'alpha': structure.lattice.alpha,
                    'beta': structure.lattice.beta,
                    'gamma': structure.lattice.gamma,
                    'volume': structure.lattice.volume
                },
                'atoms': []
            }
            
            # Extract atomic information
            for site in structure:
                crystal_info['atoms'].append({
                    'element': site.specie.symbol,
                    'position': site.frac_coords.tolist(),
                    'cartesian': site.coords.tolist()
                })
            
            # Store structure data
            self.crystal_structure = crystal_info
            self.current_crystal_structure = crystal_info  # Also store as current_crystal_structure for consistency
            
            # Update crystal system combo box if it matches
            if hasattr(self, 'crystal_system_combo'):
                crystal_systems = [self.crystal_system_combo.itemText(i) for i in range(self.crystal_system_combo.count())]
                if crystal_info['crystal_system'] in crystal_systems:
                    self.crystal_system_combo.setCurrentText(crystal_info['crystal_system'])
            
            # Show structure information
            self.show_crystal_structure_info()
            
        except Exception as e:
            raise Exception(f"Pymatgen parsing error: {str(e)}")
    
    def load_cif_simple(self, file_path):
        """Load CIF using simplified parser."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            crystal_info = {
                'formula': 'Unknown',
                'space_group': 'Unknown',
                'crystal_system': 'Unknown',
                'lattice_params': {},
                'atoms': []
            }
            
            # Simple parsing for basic parameters
            for line in lines:
                line = line.strip()
                if line.startswith('_chemical_formula_sum'):
                    crystal_info['formula'] = line.split()[-1].strip("'\"")
                elif line.startswith('_space_group_name_H-M_alt'):
                    crystal_info['space_group'] = line.split()[-1].strip("'\"")
                elif line.startswith('_cell_length_a'):
                    crystal_info['lattice_params']['a'] = float(line.split()[1])
                elif line.startswith('_cell_length_b'):
                    crystal_info['lattice_params']['b'] = float(line.split()[1])
                elif line.startswith('_cell_length_c'):
                    crystal_info['lattice_params']['c'] = float(line.split()[1])
                elif line.startswith('_cell_angle_alpha'):
                    crystal_info['lattice_params']['alpha'] = float(line.split()[1])
                elif line.startswith('_cell_angle_beta'):
                    crystal_info['lattice_params']['beta'] = float(line.split()[1])
                elif line.startswith('_cell_angle_gamma'):
                    crystal_info['lattice_params']['gamma'] = float(line.split()[1])
            
            # Determine crystal system from lattice parameters
            crystal_info['crystal_system'] = self.determine_crystal_system(crystal_info['lattice_params'])
            
            # Store structure data
            self.crystal_structure = crystal_info
            self.current_crystal_structure = crystal_info  # Also store as current_crystal_structure for consistency
            
            # Update crystal system combo box
            if hasattr(self, 'crystal_system_combo'):
                crystal_systems = [self.crystal_system_combo.itemText(i) for i in range(self.crystal_system_combo.count())]
                if crystal_info['crystal_system'] in crystal_systems:
                    self.crystal_system_combo.setCurrentText(crystal_info['crystal_system'])
            
            # Show structure information
            self.show_crystal_structure_info()
            
        except Exception as e:
            raise Exception(f"Simple CIF parsing error: {str(e)}")
    
    def determine_crystal_system(self, lattice_params):
        """Determine crystal system from lattice parameters."""
        if not all(key in lattice_params for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
            return 'Unknown'
        
        a, b, c = lattice_params['a'], lattice_params['b'], lattice_params['c']
        alpha, beta, gamma = lattice_params['alpha'], lattice_params['beta'], lattice_params['gamma']
        
        # Tolerance for equality checks
        tol = 1.0  # Increased tolerance for real crystal data
        angle_tol = 2.0  # Tolerance for angles in degrees
        
        # Check for cubic: a = b = c, α = β = γ = 90°
        if (abs(a - b) < tol and abs(b - c) < tol and 
            abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol and abs(gamma - 90) < angle_tol):
            return 'Cubic'
        
        # Check for tetragonal: a = b ≠ c, α = β = γ = 90°
        elif (abs(a - b) < tol and abs(c - a) > tol and
              abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol and abs(gamma - 90) < angle_tol):
            return 'Tetragonal'
        
        # Check for orthorhombic: a ≠ b ≠ c, α = β = γ = 90°
        elif (abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol and abs(gamma - 90) < angle_tol):
            return 'Orthorhombic'
        
        # Check for hexagonal: a = b ≠ c, α = β = 90°, γ = 120°
        elif (abs(a - b) < tol and abs(c - a) > tol and
              abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol and abs(gamma - 120) < angle_tol):
            return 'Hexagonal'
        
        # Check for trigonal/rhombohedral: Two cases
        # Case 1: Rhombohedral setting: a = b = c, α = β = γ ≠ 90°
        elif (abs(a - b) < tol and abs(b - c) < tol and
              abs(alpha - beta) < angle_tol and abs(beta - gamma) < angle_tol and
              abs(alpha - 90) > angle_tol):
            return 'Trigonal'
        
        # Case 2: Hexagonal setting for trigonal: a = b ≠ c, α = β = 90°, γ = 120° 
        # (This overlaps with hexagonal, so we need space group info to distinguish)
        
        # Check for monoclinic: α = γ = 90°, β ≠ 90°
        elif (abs(alpha - 90) < angle_tol and abs(gamma - 90) < angle_tol and 
              abs(beta - 90) > angle_tol):
            return 'Monoclinic'
        
        # Default to triclinic
        else:
            return 'Triclinic'
    
    def show_crystal_structure_info(self):
        """Show crystal structure information in a dialog."""
        if not hasattr(self, 'crystal_structure'):
            return
        
        # Create info dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Crystal Structure Information")
        dialog.setModal(True)
        dialog.resize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title_label = QLabel("Crystal Structure Information")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create text area for structure info
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        # Format structure information
        struct = self.crystal_structure
        info_content = f"Chemical Formula: {struct['formula']}\n"
        info_content += f"Space Group: {struct['space_group']}\n"
        info_content += f"Crystal System: {struct['crystal_system']}\n\n"
        
        info_content += "Lattice Parameters:\n"
        for param, value in struct['lattice_params'].items():
            if isinstance(value, float):
                info_content += f"  {param}: {value:.4f}\n"
            else:
                info_content += f"  {param}: {value}\n"
        
        if struct['atoms']:
            info_content += f"\nNumber of atoms: {len(struct['atoms'])}\n"
            info_content += "Atomic positions (first 10):\n"
            for i, atom in enumerate(struct['atoms'][:10]):
                info_content += f"  {atom['element']}: {atom['position']}\n"
        
        info_text.setPlainText(info_content)
        layout.addWidget(info_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
        
        QMessageBox.information(self, "Success", "Crystal structure loaded successfully!")
    
    def calculate_polarization(self):
        """Calculate polarization analysis."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "Please load a spectrum first.")
            return
        
        try:
            # Get polarization configuration
            crystal_system = self.crystal_system_combo.currentText()
            incident_angle = self.incident_angle_spin.value()
            scattered_angle = self.scattered_angle_spin.value()
            
            # Perform basic polarization analysis
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            
            # Calculate depolarization ratios for each peak
            depolarization_ratios = {}
            
            if self.fitted_peaks:
                for i, peak in enumerate(self.fitted_peaks):
                    freq = peak['position']
                    intensity = peak['amplitude']
                    
                    # Simple depolarization ratio calculation based on crystal system
                    if crystal_system == "Cubic":
                        # For cubic crystals, only A1g modes are allowed
                        depol_ratio = 0.0  # Fully polarized
                    elif crystal_system == "Hexagonal":
                        # For hexagonal crystals, A1g and E2g modes
                        if freq < 500:  # Assume low freq are E2g modes
                            depol_ratio = 0.75  # Depolarized
                        else:
                            depol_ratio = 0.0  # Polarized A1g
                    else:
                        # General case - calculate based on intensity and frequency
                        depol_ratio = min(0.75, 0.5 * np.exp(-(freq - 1000)**2 / (2 * 200**2)))
                    
                    depolarization_ratios[freq] = {
                        'ratio': depol_ratio,
                        'intensity': intensity,
                        'classification': 'Polarized' if depol_ratio < 0.1 else 'Depolarized'
                    }
            
            # Store results
            self.depolarization_ratios = depolarization_ratios
            self.current_polarization_config = {
                'crystal_system': crystal_system,
                'incident_angle': incident_angle,
                'scattered_angle': scattered_angle,
                'timestamp': datetime.now()
            }
            
            # Show results
            self.show_polarization_results()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error calculating polarization: {str(e)}")
    
    def show_polarization_results(self):
        """Show polarization analysis results in a dialog."""
        if not self.depolarization_ratios:
            return
        
        # Create results dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Polarization Analysis Results")
        dialog.setModal(True)
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Configuration info
        config_label = QLabel(f"Configuration: {self.current_polarization_config['crystal_system']} crystal system")
        config_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(config_label)
        
        # Results table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Frequency (cm⁻¹)", "Intensity", "Depolarization Ratio", "Classification"])
        
        # Populate table
        table.setRowCount(len(self.depolarization_ratios))
        row = 0
        for freq, data in sorted(self.depolarization_ratios.items()):
            table.setItem(row, 0, QTableWidgetItem(f"{freq:.1f}"))
            table.setItem(row, 1, QTableWidgetItem(f"{data['intensity']:.3f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{data['ratio']:.3f}"))
            table.setItem(row, 3, QTableWidgetItem(data['classification']))
            row += 1
        
        # Resize columns to content
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def calculate_raman_tensors(self):
        """Calculate Raman tensors."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "Please load a spectrum first.")
            return
        
        if not self.fitted_peaks:
            QMessageBox.warning(self, "Warning", "Please fit peaks first in the Peak Fitting tab.")
            return
        
        try:
            # Get crystal system and calculation method
            crystal_system = self.tensor_crystal_system_combo.currentText()
            calc_method = self.tensor_method_combo.currentText()
            
            # Show progress
            progress = QProgressDialog("Calculating Raman tensors...", "Cancel", 0, len(self.fitted_peaks), self)
            progress.setWindowTitle("Tensor Calculation")
            progress.setModal(True)
            progress.show()
            
            # Calculate Raman tensors for each fitted peak
            raman_tensors = {}
            
            for i, peak in enumerate(self.fitted_peaks):
                progress.setValue(i)
                QApplication.processEvents()
                
                if progress.wasCanceled():
                    return
                
                freq = peak['position']
                intensity = peak['amplitude']
                width = peak['width']
                
                # Get vibrational character if available from peak matching
                character = 'Unknown'
                if hasattr(self, 'peak_labels') and freq in self.peak_labels:
                    character = self.peak_labels[freq]['character']
                elif hasattr(self, 'matched_peaks'):
                    # Find character from matched peaks
                    for match in self.matched_peaks:
                        if abs(match['experimental_peak']['position'] - freq) < 1.0:
                            character = match['calculated_peak']['character']
                            break
                
                # Generate appropriate tensor based on crystal system, character, and frequency
                tensor = self.generate_raman_tensor_advanced(crystal_system, freq, intensity, character, calc_method)
                
                # Calculate tensor properties
                tensor_props = self.analyze_tensor_properties(tensor)
                
                # Calculate additional properties
                additional_props = self.calculate_additional_tensor_properties(tensor, crystal_system, character)
                
                raman_tensors[freq] = {
                    'tensor': tensor,
                    'properties': tensor_props,
                    'additional_properties': additional_props,
                    'intensity': intensity,
                    'width': width,
                    'character': character,
                    'crystal_system': crystal_system,
                    'calculation_method': calc_method
                }
            
            progress.setValue(len(self.fitted_peaks))
            progress.close()
            
            # Store results
            self.calculated_raman_tensors = raman_tensors
            self.tensor_analysis_results = {
                'crystal_system': crystal_system,
                'calculation_method': calc_method,
                'peak_count': len(raman_tensors),
                'timestamp': datetime.now()
            }
            
            # Auto-assign vibrational mode characters
            self.auto_assign_mode_characters()
            
            # Update visualization
            self.update_tensor_visualization()
            
            # Show success message
            QMessageBox.information(self, "Success", 
                                  f"Calculated Raman tensors for {len(raman_tensors)} peaks using {calc_method} method.\nMode characters automatically assigned.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error calculating Raman tensors: {str(e)}")
    
    def generate_raman_tensor(self, crystal_system, frequency, intensity, character=None):
        """Generate a Raman tensor based on crystal system, frequency, intensity, and vibrational mode character."""
        # Normalize intensity for tensor scaling
        scale = intensity / 1000.0 if intensity > 0 else 0.001
        
        # Use character if provided, otherwise fallback to frequency-based assignment
        if character:
            character = character.upper()
        
        if crystal_system == "Cubic":
            # For cubic crystals, use character to determine tensor form
            if character and any(x in character for x in ['T2G', 'F2G']):
                # T2g/F2g modes have pure off-diagonal components
                # Use the three standard T2g tensor orientations
                orientations = [
                    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),      # xy coupling
                    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),      # xz coupling
                    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])       # yz coupling
                ]
                # Choose orientation based on frequency for variety
                tensor = orientations[int(frequency / 300) % 3] * scale
            elif character and 'A1G' in character:
                # A1g modes have isotropic diagonal components
                tensor = np.array([
                    [scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, scale]
                ])
            else:
                # Fallback: frequency-based assignment for cubic
                if frequency < 500:  # Assume T2g/F2g modes at lower frequencies
                    orientations = [
                        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
                        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
                    ]
                    tensor = orientations[int(frequency / 200) % 3] * scale
                else:  # A1g modes at higher frequencies
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, scale]
                    ])
        
        elif crystal_system == "Tetragonal":
            if character:
                if 'A1G' in character or 'A1' in character:
                    # A1g: diagonal with c-axis enhancement
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, scale * 1.2]
                    ])
                elif 'B1G' in character or 'B1' in character:
                    # B1g: xy coupling dominant
                    tensor = np.array([
                        [scale * 0.1, scale * 0.9, 0],
                        [scale * 0.9, scale * 0.1, 0],
                        [0, 0, scale * 0.05]
                    ])
                elif 'B2G' in character or 'B2' in character:
                    # B2g: x²-y² character
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, -scale, 0],
                        [0, 0, scale * 0.1]
                    ])
                elif 'EG' in character or 'E' in character:
                    # Eg: xz and yz coupling
                    tensor = np.array([
                        [scale * 0.2, 0, scale * 0.8],
                        [0, scale * 0.2, scale * 0.8],
                        [scale * 0.8, scale * 0.8, scale * 0.1]
                    ])
                else:
                    # Default tetragonal
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, scale * 1.2]
                    ])
            else:
                # Frequency-based fallback for tetragonal
                if frequency > 800:
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, scale * 1.5]
                    ])
                else:
                    tensor = np.array([
                        [scale * 0.8, 0, 0],
                        [0, scale * 0.8, 0],
                        [0, 0, scale]
                    ])
        
        elif crystal_system == "Hexagonal":
            if character:
                if 'A1G' in character or 'A1' in character:
                    # A1g: c-axis enhancement
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, scale * 1.5]
                    ])
                elif 'E2G' in character or 'E1G' in character or 'EG' in character:
                    # Eg modes: xy plane anisotropy
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, -scale, 0],
                        [0, 0, scale * 0.2]
                    ])
                else:
                    # Default hexagonal
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, scale * 1.2]
                    ])
            else:
                # Frequency-based fallback for hexagonal
                if frequency < 500:
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, -scale, 0],
                        [0, 0, 0]
                    ])
                else:
                    tensor = np.array([
                        [scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, scale * 1.2]
                    ])
        
        else:
            # General case - slightly anisotropic
            tensor = np.array([
                [scale, scale * 0.1, 0],
                [scale * 0.1, scale * 0.9, 0],
                [0, 0, scale * 1.1]
            ])
        
        return tensor
    
    def analyze_tensor_properties(self, tensor):
        """Analyze properties of a Raman tensor."""
        try:
            # Calculate eigenvalues and eigenvectors
            eigenvals, eigenvects = np.linalg.eig(tensor)
            
            # Calculate tensor invariants
            trace = np.trace(tensor)
            determinant = np.linalg.det(tensor)
            
            # Calculate anisotropy
            max_eigenval = np.max(eigenvals)
            min_eigenval = np.min(eigenvals)
            anisotropy = (max_eigenval - min_eigenval) / (max_eigenval + min_eigenval) if (max_eigenval + min_eigenval) != 0 else 0
            
            # Calculate spherical and deviatoric parts
            spherical_part = trace / 3.0
            deviatoric_tensor = tensor - spherical_part * np.eye(3)
            deviatoric_norm = np.linalg.norm(deviatoric_tensor)
            
            return {
                'eigenvalues': eigenvals,
                'eigenvectors': eigenvects,
                'trace': trace,
                'determinant': determinant,
                'anisotropy': anisotropy,
                'spherical_part': spherical_part,
                'deviatoric_norm': deviatoric_norm,
                'tensor_norm': np.linalg.norm(tensor)
            }
            
        except Exception as e:
            print(f"Error analyzing tensor properties: {e}")
            return {'error': str(e)}
    
    def show_tensor_results(self):
        """Show Raman tensor analysis results in a dialog."""
        if not self.calculated_raman_tensors:
            return
        
        # Create results dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Raman Tensor Analysis Results")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title_label = QLabel("Raman Tensor Analysis Results")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create scroll area for results
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Add results for each peak
        for freq, data in sorted(self.calculated_raman_tensors.items()):
            # Peak group box
            peak_group = QGroupBox(f"Peak at {freq:.1f} cm⁻¹")
            peak_layout = QVBoxLayout(peak_group)
            
            # Tensor matrix display
            tensor = data['tensor']
            tensor_text = f"Raman Tensor Matrix:\n"
            for i in range(3):
                row_text = "  [" + "  ".join([f"{tensor[i,j]:8.4f}" for j in range(3)]) + "]"
                tensor_text += row_text + "\n"
            
            tensor_label = QLabel(tensor_text)
            tensor_label.setFont(QFont("Courier", 10))
            tensor_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
            peak_layout.addWidget(tensor_label)
            
            # Properties
            props = data['properties']
            if 'error' not in props:
                props_text = f"Properties:\n"
                props_text += f"  Trace: {props['trace']:.4f}\n"
                props_text += f"  Anisotropy: {props['anisotropy']:.4f}\n"
                props_text += f"  Tensor Norm: {props['tensor_norm']:.4f}\n"
                props_text += f"  Eigenvalues: [{props['eigenvalues'][0]:.4f}, {props['eigenvalues'][1]:.4f}, {props['eigenvalues'][2]:.4f}]"
                
                props_label = QLabel(props_text)
                props_label.setFont(QFont("Arial", 10))
                peak_layout.addWidget(props_label)
            
            scroll_layout.addWidget(peak_group)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def run_orientation_optimization(self):
        """Run orientation optimization."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "Please load a spectrum first.")
            return
        
        if not self.fitted_peaks:
            QMessageBox.warning(self, "Warning", "Please fit peaks first in the Peak Fitting tab.")
            return
        
        if not hasattr(self, 'crystal_structure'):
            QMessageBox.warning(self, "Warning", "Please load crystal structure first in the Crystal Structure tab.")
            return
        
        try:
            # Show progress dialog
            progress = QProgressDialog("Running orientation optimization...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Optimization Progress")
            progress.setModal(True)
            progress.show()
            
            # Run optimization in stages
            results = {}
            
            # Stage 1: Peak intensity analysis
            progress.setValue(25)
            progress.setLabelText("Stage 1: Analyzing peak intensities...")
            QApplication.processEvents()
            
            stage1_results = self.optimize_stage1_intensity_analysis()
            results['stage1'] = stage1_results
            
            if progress.wasCanceled():
                return
            
            # Stage 2: Angular dependence fitting
            progress.setValue(50)
            progress.setLabelText("Stage 2: Fitting angular dependence...")
            QApplication.processEvents()
            
            stage2_results = self.optimize_stage2_angular_fitting(stage1_results)
            results['stage2'] = stage2_results
            
            if progress.wasCanceled():
                return
            
            # Stage 3: Final orientation refinement
            progress.setValue(75)
            progress.setLabelText("Stage 3: Refining crystal orientation...")
            QApplication.processEvents()
            
            stage3_results = self.optimize_stage3_refinement(stage1_results, stage2_results)
            results['stage3'] = stage3_results
            
            progress.setValue(100)
            progress.close()
            
            # Store results
            self.orientation_results = results
            self.optimized_orientation = stage3_results.get('best_orientation', {})
            
            # Show results
            self.show_optimization_results()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during orientation optimization: {str(e)}")
    
    def optimize_stage1_intensity_analysis(self):
        """Stage 1: Analyze peak intensities for initial orientation estimates."""
        results = {
            'peak_analysis': {},
            'intensity_ratios': {},
            'preliminary_orientation': {}
        }
        
        # Analyze each fitted peak
        for peak in self.fitted_peaks:
            freq = peak['position']
            intensity = peak['amplitude']
            width = peak['width']
            
            # Calculate theoretical intensity for different orientations
            orientation_scores = {}
            
            # Test various orientations (Euler angles)
            for phi in [0, 30, 45, 60, 90]:  # Simplified grid
                for theta in [0, 30, 45, 60, 90]:
                    for psi in [0, 45, 90]:
                        orientation = (phi, theta, psi)
                        
                        # Calculate theoretical Raman intensity for this orientation
                        theoretical_intensity = self.calculate_theoretical_intensity(freq, orientation)
                        
                        # Score based on agreement with experimental intensity
                        score = 1.0 / (1.0 + abs(theoretical_intensity - intensity))
                        orientation_scores[orientation] = score
            
            # Find best orientation for this peak
            best_orientation = max(orientation_scores, key=orientation_scores.get)
            best_score = orientation_scores[best_orientation]
            
            results['peak_analysis'][freq] = {
                'experimental_intensity': intensity,
                'best_orientation': best_orientation,
                'orientation_score': best_score,
                'all_scores': orientation_scores
            }
        
        # Calculate overall preliminary orientation
        if results['peak_analysis']:
            # Weight by peak intensity and score
            weighted_orientations = []
            total_weight = 0
            
            for freq, analysis in results['peak_analysis'].items():
                weight = analysis['experimental_intensity'] * analysis['orientation_score']
                orientation = analysis['best_orientation']
                weighted_orientations.append((orientation, weight))
                total_weight += weight
            
            # Calculate weighted average orientation
            if total_weight > 0:
                avg_phi = sum(orient[0] * weight for orient, weight in weighted_orientations) / total_weight
                avg_theta = sum(orient[1] * weight for orient, weight in weighted_orientations) / total_weight
                avg_psi = sum(orient[2] * weight for orient, weight in weighted_orientations) / total_weight
                
                results['preliminary_orientation'] = {
                    'phi': avg_phi,
                    'theta': avg_theta, 
                    'psi': avg_psi,
                    'confidence': total_weight / len(weighted_orientations)
                }
        
        return results
    
    def optimize_stage2_angular_fitting(self, stage1_results):
        """Stage 2: Fit angular dependence around preliminary orientation."""
        results = {
            'angular_fit': {},
            'refined_orientation': {},
            'fit_quality': {}
        }
        
        if not stage1_results.get('preliminary_orientation'):
            return results
        
        prelim = stage1_results['preliminary_orientation']
        center_phi, center_theta, center_psi = prelim['phi'], prelim['theta'], prelim['psi']
        
        # Fine grid search around preliminary orientation
        best_score = -1
        best_orientation = None
        
        for dphi in [-10, -5, 0, 5, 10]:
            for dtheta in [-10, -5, 0, 5, 10]:
                for dpsi in [-10, -5, 0, 5, 10]:
                    test_orientation = (
                        center_phi + dphi,
                        center_theta + dtheta,
                        center_psi + dpsi
                    )
                    
                    # Calculate overall agreement score
                    total_score = 0
                    for peak in self.fitted_peaks:
                        freq = peak['position']
                        exp_intensity = peak['amplitude']
                        theo_intensity = self.calculate_theoretical_intensity(freq, test_orientation)
                        
                        # Score based on relative agreement
                        if exp_intensity > 0:
                            score = min(theo_intensity, exp_intensity) / max(theo_intensity, exp_intensity)
                        else:
                            score = 0
                        total_score += score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_orientation = test_orientation
        
        if best_orientation:
            results['refined_orientation'] = {
                'phi': best_orientation[0],
                'theta': best_orientation[1],
                'psi': best_orientation[2],
                'total_score': best_score
            }
            
            results['fit_quality'] = {
                'average_agreement': best_score / len(self.fitted_peaks) if self.fitted_peaks else 0,
                'refinement_improvement': best_score - prelim.get('confidence', 0)
            }
        
        return results
    
    def optimize_stage3_refinement(self, stage1_results, stage2_results):
        """Stage 3: Final refinement using local optimization."""
        results = {
            'final_orientation': {},
            'optimization_metrics': {},
            'convergence_info': {}
        }
        
        if not stage2_results.get('refined_orientation'):
            return results
        
        refined = stage2_results['refined_orientation']
        start_orientation = [refined['phi'], refined['theta'], refined['psi']]
        
        # Define objective function for scipy optimization
        def objective_function(orientation):
            phi, theta, psi = orientation
            total_error = 0
            
            for peak in self.fitted_peaks:
                freq = peak['position']
                exp_intensity = peak['amplitude']
                theo_intensity = self.calculate_theoretical_intensity(freq, (phi, theta, psi))
                
                # Squared relative error
                if exp_intensity > 0:
                    error = ((theo_intensity - exp_intensity) / exp_intensity) ** 2
                else:
                    error = theo_intensity ** 2
                total_error += error
            
            return total_error
        
        # Run local optimization
        try:
            result = minimize(objective_function, start_orientation, method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-4})
            
            if result.success:
                final_phi, final_theta, final_psi = result.x
                
                results['final_orientation'] = {
                    'phi': final_phi,
                    'theta': final_theta,
                    'psi': final_psi,
                    'final_error': result.fun
                }
                
                results['optimization_metrics'] = {
                    'convergence': result.success,
                    'iterations': result.nit,
                    'function_evaluations': result.nfev,
                    'final_error': result.fun
                }
                
                results['convergence_info'] = {
                    'message': result.message,
                    'initial_error': objective_function(start_orientation),
                    'improvement': objective_function(start_orientation) - result.fun
                }
                
                # This becomes the best orientation
                results['best_orientation'] = results['final_orientation']
        
        except Exception as e:
            print(f"Optimization failed: {e}")
            # Fall back to stage 2 results
            results['best_orientation'] = refined
            results['optimization_metrics'] = {'convergence': False, 'error': str(e)}
        
        return results
    
    def calculate_theoretical_intensity(self, frequency, orientation):
        """Calculate theoretical Raman intensity for given frequency and orientation."""
        phi, theta, psi = orientation
        
        # Convert to radians
        phi_rad = np.radians(phi)
        theta_rad = np.radians(theta)
        psi_rad = np.radians(psi)
        
        # Simple model: intensity depends on orientation and frequency
        # This is a simplified calculation - real implementation would use
        # proper crystallographic orientation matrices and Raman tensor components
        
        # Base intensity from frequency (simple model)
        base_intensity = 1000 * np.exp(-(frequency - 500)**2 / (2 * 300**2))
        
        # Orientation-dependent modulation
        orientation_factor = (
            np.cos(phi_rad)**2 * np.sin(theta_rad)**2 +
            np.sin(phi_rad)**2 * np.cos(psi_rad)**2 +
            np.cos(theta_rad)**2
        ) / 3.0
        
        return base_intensity * orientation_factor
    
    def show_optimization_results(self):
        """Show orientation optimization results in a dialog."""
        if not self.orientation_results:
            return
        
        # Create results dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Orientation Optimization Results")
        dialog.setModal(True)
        dialog.resize(700, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title_label = QLabel("Crystal Orientation Optimization Results")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create text area for results
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        
        # Format results
        content = "ORIENTATION OPTIMIZATION SUMMARY\n"
        content += "=" * 50 + "\n\n"
        
        # Stage 1 results
        if 'stage1' in self.orientation_results:
            stage1 = self.orientation_results['stage1']
            content += "STAGE 1: Peak Intensity Analysis\n"
            content += "-" * 35 + "\n"
            
            if 'preliminary_orientation' in stage1:
                prelim = stage1['preliminary_orientation']
                content += f"Preliminary Orientation:\n"
                content += f"  φ (phi): {prelim.get('phi', 0):.2f}°\n"
                content += f"  θ (theta): {prelim.get('theta', 0):.2f}°\n"
                content += f"  ψ (psi): {prelim.get('psi', 0):.2f}°\n"
                content += f"  Confidence: {prelim.get('confidence', 0):.3f}\n\n"
        
        # Stage 2 results
        if 'stage2' in self.orientation_results:
            stage2 = self.orientation_results['stage2']
            content += "STAGE 2: Angular Dependence Fitting\n"
            content += "-" * 37 + "\n"
            
            if 'refined_orientation' in stage2:
                refined = stage2['refined_orientation']
                content += f"Refined Orientation:\n"
                content += f"  φ (phi): {refined.get('phi', 0):.2f}°\n"
                content += f"  θ (theta): {refined.get('theta', 0):.2f}°\n"
                content += f"  ψ (psi): {refined.get('psi', 0):.2f}°\n"
                content += f"  Total Score: {refined.get('total_score', 0):.3f}\n\n"
        
        # Stage 3 results
        if 'stage3' in self.orientation_results:
            stage3 = self.orientation_results['stage3']
            content += "STAGE 3: Final Optimization\n"
            content += "-" * 28 + "\n"
            
            if 'best_orientation' in stage3:
                best = stage3['best_orientation']
                content += f"Final Orientation:\n"
                content += f"  φ (phi): {best.get('phi', 0):.2f}°\n"
                content += f"  θ (theta): {best.get('theta', 0):.2f}°\n"
                content += f"  ψ (psi): {best.get('psi', 0):.2f}°\n"
                
                if 'final_error' in best:
                    content += f"  Final Error: {best['final_error']:.6f}\n"
            
            if 'optimization_metrics' in stage3:
                metrics = stage3['optimization_metrics']
                content += f"\nOptimization Metrics:\n"
                content += f"  Converged: {metrics.get('convergence', False)}\n"
                if 'iterations' in metrics:
                    content += f"  Iterations: {metrics['iterations']}\n"
                if 'improvement' in stage3.get('convergence_info', {}):
                    improvement = stage3['convergence_info']['improvement']
                    content += f"  Improvement: {improvement:.6f}\n"
        
        results_text.setPlainText(content)
        layout.addWidget(results_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def debug_database_content(self):
        """Debug database content to inspect the actual content."""
        if not self.selected_reference_mineral:
            QMessageBox.warning(self, "Warning", "Please select a reference mineral first.")
            return
        
        mineral_data = self.mineral_database.get(self.selected_reference_mineral)
        if not mineral_data or 'raman_modes' not in mineral_data:
            return
        
        # Create debug dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Debug Database Content")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Title and summary
        title_label = QLabel(f"Debug: {self.selected_reference_mineral}")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)
        
                # Create text area for debug content
        debug_text = QTextEdit()
        debug_text.setReadOnly(True)
        
        # Format debug content
        debug_content = f"Reference Mineral: {self.selected_reference_mineral}\n"
        debug_content += f"Total modes in raman_modes: {len(mineral_data['raman_modes'])}\n"
        
        # Check for spectrum data
        if 'spectrum_data' in mineral_data:
            debug_content += f"Has stored spectrum_data: YES\n"
            spectrum_data = mineral_data['spectrum_data']
            if 'wavenumbers' in spectrum_data and 'intensities' in spectrum_data:
                wavenumbers = spectrum_data['wavenumbers']
                intensities = spectrum_data['intensities']
                debug_content += f"Spectrum range: {min(wavenumbers):.1f} - {max(wavenumbers):.1f} cm⁻¹\n"
                debug_content += f"Number of points: {len(wavenumbers)}\n"
                
                # Check for peak near 1085
                target_indices = [i for i, w in enumerate(wavenumbers) if abs(w - 1085) < 20]
                if target_indices:
                    debug_content += f"Peak near 1085 cm⁻¹ in spectrum: YES\n"
                    for idx in target_indices[:3]:  # Show first 3 matches
                        debug_content += f"  {wavenumbers[idx]:.1f} cm⁻¹: intensity {intensities[idx]:.3f}\n"
                else:
                    debug_content += f"Peak near 1085 cm⁻¹ in spectrum: NO\n"
        else:
            debug_content += f"Has stored spectrum_data: NO\n"
        
        # Check for raw modes
        try:
            raw_modes = self.get_modes_from_mineral_data(self.selected_reference_mineral, mineral_data)
            if raw_modes:
                debug_content += f"Raw modes available: {len(raw_modes)}\n"
                # Check for mode near 1085 in raw modes
                near_1085_raw = [mode for mode in raw_modes if len(mode) >= 1 and abs(mode[0] - 1085) < 50]
                if near_1085_raw:
                    debug_content += f"Raw modes near 1085 cm⁻¹: {len(near_1085_raw)}\n"
                    for mode in near_1085_raw:
                        debug_content += f"  {mode[0]:.1f} cm⁻¹ - {mode[1] if len(mode) > 1 else 'Unknown'} - {mode[2] if len(mode) > 2 else 'Unknown'}\n"
            else:
                debug_content += f"Raw modes available: NO\n"
        except Exception as e:
            debug_content += f"Error checking raw modes: {e}\n"
        
        debug_content += "\n" + "="*50 + "\n"
        debug_content += "RAMAN_MODES LIST:\n"
        debug_content += "="*50 + "\n\n"
        
        for i, mode in enumerate(mineral_data['raman_modes']):
            debug_content += f"Mode {i+1}:\n"
            debug_content += f"  Frequency: {mode.get('frequency', 'Unknown')} cm⁻¹\n"
            debug_content += f"  Character: {mode.get('character', 'Unknown')}\n"
            debug_content += f"  Intensity: {mode.get('numerical_intensity', mode.get('intensity', 'medium'))}\n"
            
            # Check if near 1085
            freq = mode.get('frequency', 0)
            if isinstance(freq, (int, float)) and abs(freq - 1085) < 50:
                debug_content += f"  *** NEAR 1085 cm⁻¹ ***\n"
            
            debug_content += "\n"
        
        debug_text.setPlainText(debug_content)
        layout.addWidget(debug_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    # === Tensor Analysis Methods ===
    
    def on_tensor_crystal_system_changed(self, crystal_system):
        """Handle crystal system changes in tensor analysis tab."""
        print(f"Crystal system changed to: {crystal_system}")
        # If we have calculated tensors, recalculate with new crystal system
        if hasattr(self, 'calculated_raman_tensors') and self.calculated_raman_tensors:
            reply = QMessageBox.question(self, "Recalculate Tensors", 
                                       f"Crystal system changed to {crystal_system}. Recalculate tensors?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.calculate_raman_tensors()
    
    def auto_detect_crystal_system(self):
        """Auto-detect crystal system from loaded crystal structure."""
        # Check for crystal structure in multiple possible locations
        crystal_structure_data = None
        
        if hasattr(self, 'current_crystal_structure') and self.current_crystal_structure:
            crystal_structure_data = self.current_crystal_structure
        elif hasattr(self, 'crystal_structure') and self.crystal_structure:
            crystal_structure_data = self.crystal_structure
        elif hasattr(self, 'crystal_structure_widget') and self.crystal_structure_widget:
            # Try to get from the crystal structure widget
            if hasattr(self.crystal_structure_widget, 'current_structure') and self.crystal_structure_widget.current_structure:
                crystal_structure_data = self.crystal_structure_widget.current_structure
        
        if crystal_structure_data:
            # First try to use space group information (most reliable)
            space_group = crystal_structure_data.get('space_group', 'Unknown')
            crystal_system = self.determine_crystal_system_from_space_group(space_group)
            
            # If space group doesn't give us the answer, use stored crystal system
            if crystal_system == 'Unknown':
                crystal_system = crystal_structure_data.get('crystal_system', 'Unknown')
            
            # If still unknown, try lattice parameters
            if crystal_system == 'Unknown':
                lattice_params = crystal_structure_data.get('lattice_params', {})
                if lattice_params:
                    crystal_system = self.determine_crystal_system(lattice_params)
            
            if crystal_system != 'Unknown':
                # Find matching item in combo box
                for i in range(self.tensor_crystal_system_combo.count()):
                    if self.tensor_crystal_system_combo.itemText(i).lower() == crystal_system.lower():
                        self.tensor_crystal_system_combo.setCurrentIndex(i)
                        
                        # Show detailed information
                        info_msg = f"Crystal system detected: {crystal_system}"
                        if space_group != 'Unknown':
                            info_msg += f"\nSpace group: {space_group}"
                        
                        QMessageBox.information(self, "Auto-Detection", info_msg)
                        return
                        
                QMessageBox.warning(self, "Auto-Detection", 
                                  f"Crystal system '{crystal_system}' not found in available options.")
            else:
                QMessageBox.warning(self, "Auto-Detection", 
                                  "Crystal system could not be determined from structure data.")
        else:
            QMessageBox.warning(self, "Auto-Detection", 
                              "No crystal structure loaded. Please load a CIF file first.")
    
    def determine_crystal_system_from_space_group(self, space_group):
        """Determine crystal system from space group symbol or number."""
        if not space_group or space_group == 'Unknown':
            return 'Unknown'
        
        # Clean up space group input
        sg_str = str(space_group).strip()
        
        # First, try to parse as space group number
        try:
            sg_number = int(sg_str)
            return self.crystal_system_from_space_group_number(sg_number)
        except ValueError:
            # Not a number, try symbol matching
            pass
        
        # Clean up space group symbol for text matching
        sg = sg_str.upper()
        
        # Space group symbol to crystal system mapping
        # Cubic: 195-230
        cubic_symbols = ['P23', 'F23', 'I23', 'P213', 'I213', 'PM3', 'PN3', 'FM3', 'FD3', 'IM3', 'PA3', 'IA3',
                        'P432', 'P4232', 'F432', 'F4132', 'I432', 'P4332', 'P4132', 'I4132',
                        'P-43M', 'F-43M', 'I-43M', 'P-43N', 'F-43C', 'I-43D',
                        'PM-3M', 'PN-3N', 'PM-3N', 'PN-3M', 'FM-3M', 'FM-3C', 'FD-3M', 'FD-3C',
                        'IM-3M', 'IA-3D']
        
        # Hexagonal: 168-194
        hexagonal_symbols = ['P6', 'P61', 'P65', 'P62', 'P64', 'P63', 'P6-', 'P6/M', 'P63/M',
                           'P622', 'P6122', 'P6522', 'P6222', 'P6422', 'P6322',
                           'P6MM', 'P6CC', 'P63CM', 'P63MC', 'P6M2', 'P6C2', 'P62M', 'P62C',
                           'P6/MMM', 'P6/MCC', 'P63/MCM', 'P63/MMC']
        
        # Trigonal: 143-167 (includes rhombohedral R space groups)
        trigonal_symbols = ['P3', 'P31', 'P32', 'R3', 'P-3', 'R-3',
                          'P312', 'P321', 'P3112', 'P3121', 'P3212', 'P3221', 'R32',
                          'P3M1', 'P31M', 'P3C1', 'P31C', 'R3M', 'R3C',
                          'P-31M', 'P-31C', 'P-3M1', 'P-3C1', 'R-3M', 'R-3C']
        
        # Tetragonal: 75-142
        tetragonal_symbols = ['P4', 'P41', 'P42', 'P43', 'I4', 'I41', 'P-4', 'I-4', 'P4/M', 'P42/M', 'P4/N', 'P42/N', 'I4/M', 'I41/A',
                            'P422', 'P4212', 'P4122', 'P41212', 'P4222', 'P42212', 'P4322', 'P43212', 'I422', 'I4122',
                            'P4MM', 'P4BM', 'P42CM', 'P42NM', 'P4CC', 'P4NC', 'P42MC', 'P42BC', 'I4MM', 'I4CM', 'I41MD', 'I41CD',
                            'P-42M', 'P-42C', 'P-421M', 'P-421C', 'P-4M2', 'P-4C2', 'P-4B2', 'P-4N2', 'I-4M2', 'I-4C2', 'I-42M', 'I-42D',
                            'P4/MMM', 'P4/MCC', 'P4/NBM', 'P4/NNC', 'P4/MBM', 'P4/MNC', 'P4/NMM', 'P4/NCC', 'P42/MMC', 'P42/MCM', 'P42/NBC', 'P42/NNM', 'P42/MBC', 'P42/MNM', 'P42/NMC', 'P42/NCM', 'I4/MMM', 'I4/MCM', 'I41/AMD', 'I41/ACD']
        
        # Orthorhombic: 16-74
        orthorhombic_symbols = ['P222', 'P2221', 'P21212', 'P212121', 'C2221', 'C222', 'F222', 'I222', 'I212121',
                              'PMM2', 'PMC21', 'PCC2', 'PMA2', 'PCA21', 'PNC2', 'PMN21', 'PBA2', 'PNA21', 'PNN2', 'CMM2', 'CMC21', 'CCC2', 'AMM2', 'ABM2', 'AMA2', 'ABA2', 'FMM2', 'FDD2', 'IMM2', 'IBA2', 'IMA2',
                              'PMMM', 'PNNN', 'PCCM', 'PBAN', 'PMMA', 'PNNA', 'PMNA', 'PCCA', 'PBAM', 'PCCN', 'PBCM', 'PNNM', 'PMMN', 'PBCN', 'PBCA', 'PNMA', 'CMMM', 'CCCM', 'CMMA', 'CCCA', 'FMMM', 'FDDD', 'IMMM', 'IBAM', 'IBCA', 'IMMA']
        
        # Monoclinic: 3-15
        monoclinic_symbols = ['P2', 'P21', 'C2', 'PM', 'PC', 'CM', 'CC', 'P2/M', 'P21/M', 'C2/M', 'P2/C', 'P21/C', 'C2/C']
        
        # Check each crystal system
        for symbol in cubic_symbols:
            if symbol in sg or sg.startswith(symbol):
                return 'Cubic'
        
        for symbol in hexagonal_symbols:
            if symbol in sg or sg.startswith(symbol):
                return 'Hexagonal'
        
        for symbol in trigonal_symbols:
            if symbol in sg or sg.startswith(symbol):
                return 'Trigonal'
        
        for symbol in tetragonal_symbols:
            if symbol in sg or sg.startswith(symbol):
                return 'Tetragonal'
        
        for symbol in orthorhombic_symbols:
            if symbol in sg or sg.startswith(symbol):
                return 'Orthorhombic'
        
        for symbol in monoclinic_symbols:
            if symbol in sg or sg.startswith(symbol):
                return 'Monoclinic'
        
        # Special handling for R space groups (rhombohedral/trigonal)
        if sg.startswith('R'):
            return 'Trigonal'
        
        # If no match found, return unknown
        return 'Unknown'
    
    def crystal_system_from_space_group_number(self, sg_number):
        """Determine crystal system from space group number (1-230)."""
        try:
            sg_num = int(sg_number)
            
            # Crystal system ranges based on International Tables
            if 1 <= sg_num <= 2:
                return 'Triclinic'
            elif 3 <= sg_num <= 15:
                return 'Monoclinic'
            elif 16 <= sg_num <= 74:
                return 'Orthorhombic'
            elif 75 <= sg_num <= 142:
                return 'Tetragonal'
            elif 143 <= sg_num <= 167:
                return 'Trigonal'  # This includes 167 = R-3c!
            elif 168 <= sg_num <= 194:
                return 'Hexagonal'
            elif 195 <= sg_num <= 230:
                return 'Cubic'
            else:
                return 'Unknown'
                
        except (ValueError, TypeError):
            return 'Unknown'
    
    def generate_raman_tensor_advanced(self, crystal_system, frequency, intensity, character, calc_method):
        """Generate advanced Raman tensor based on multiple parameters."""
        # Base tensor generation - now using character parameter
        base_tensor = self.generate_raman_tensor(crystal_system, frequency, intensity, character)
        
        # Modify based on calculation method
        if calc_method == "Symmetry-Based":
            return self.apply_symmetry_constraints(base_tensor, crystal_system, character)
        elif calc_method == "Peak Intensity Analysis":
            return self.apply_intensity_scaling(base_tensor, intensity, frequency)
        elif calc_method == "Polarization Data":
            return self.apply_polarization_data(base_tensor)
        elif calc_method == "Combined Analysis":
            tensor = self.apply_symmetry_constraints(base_tensor, crystal_system, character)
            tensor = self.apply_intensity_scaling(tensor, intensity, frequency)
            return self.apply_polarization_data(tensor)
        else:
            return base_tensor
    
    def apply_symmetry_constraints(self, tensor, crystal_system, character):
        """Apply symmetry constraints based on crystal system and vibrational character."""
        if character == 'Unknown':
            return tensor
        
        # Generate character-specific tensors with proper symmetry
        base_intensity = np.max(np.abs(tensor))
        
        if crystal_system == "Tetragonal":
            if 'A1g' in character or character == 'Ag':
                # A1g: isotropic in xy plane, enhanced along z
                constrained_tensor = np.array([
                    [base_intensity, 0, 0],
                    [0, base_intensity, 0],
                    [0, 0, base_intensity * 1.5]
                ])
                
            elif 'B1g' in character:
                # B1g: (x²-y²) character - creates 4-lobed dumbbell pattern
                constrained_tensor = np.array([
                    [base_intensity * 3.0, 0, 0],        # Strong xx component
                    [0, -base_intensity * 3.0, 0],       # Opposite yy component (creates nodes)
                    [0, 0, 0]                            # No zz component
                ])
                
            elif 'B2g' in character:
                # B2g: xy character - creates rotated 4-lobed pattern
                constrained_tensor = np.array([
                    [0, base_intensity * 2.0, 0],        # Strong xy component
                    [base_intensity * 2.0, 0, 0],        # xy component
                    [0, 0, base_intensity * 0.2]         # Small zz component
                ])
                
            elif 'Eg' in character:
                # Eg: doubly degenerate modes
                constrained_tensor = np.array([
                    [base_intensity, 0, base_intensity * 0.7],
                    [0, base_intensity * 0.8, 0],
                    [base_intensity * 0.7, 0, base_intensity * 0.3]
                ])
            else:
                constrained_tensor = tensor.copy()
                
        elif crystal_system == "Hexagonal":
            if 'A1g' in character or character == 'Ag':
                # A1g: breathing mode along c-axis
                constrained_tensor = np.array([
                    [base_intensity, 0, 0],
                    [0, base_intensity, 0],
                    [0, 0, base_intensity * 2.0]
                ])
                
            elif 'E1g' in character:
                # E1g: mixed xy-z character
                constrained_tensor = np.array([
                    [base_intensity, 0, base_intensity * 0.8],
                    [0, base_intensity, 0],
                    [base_intensity * 0.8, 0, base_intensity * 0.5]
                ])
                
            elif 'E2g' in character:
                # E2g: (x²-y²) and xy characters
                constrained_tensor = np.array([
                    [base_intensity * 1.5, base_intensity * 0.5, 0],
                    [base_intensity * 0.5, -base_intensity * 1.5, 0],
                    [0, 0, 0]
                ])
            else:
                constrained_tensor = tensor.copy()
                
        elif crystal_system == "Trigonal":
            if 'A1g' in character or character == 'Ag':
                constrained_tensor = np.array([
                    [base_intensity, 0, 0],
                    [0, base_intensity, 0],
                    [0, 0, base_intensity * 1.6]
                ])
            elif 'Eg' in character:
                constrained_tensor = np.array([
                    [base_intensity * 1.3, 0, base_intensity * 0.9],
                    [0, base_intensity * 0.7, 0],
                    [base_intensity * 0.9, 0, base_intensity * 0.4]
                ])
            else:
                constrained_tensor = tensor.copy()
                
        elif crystal_system == "Cubic":
            if 'A1g' in character or character == 'Ag':
                # Totally symmetric
                constrained_tensor = np.diag([base_intensity, base_intensity, base_intensity])
            elif 'T2g' in character:
                # Triply degenerate
                constrained_tensor = np.array([
                    [base_intensity * 0.5, base_intensity * 1.2, base_intensity * 1.2],
                    [base_intensity * 1.2, base_intensity * 0.5, base_intensity * 1.2],
                    [base_intensity * 1.2, base_intensity * 1.2, base_intensity * 0.5]
                ])
            else:
                constrained_tensor = tensor.copy()
        else:
            constrained_tensor = tensor.copy()
        
        return constrained_tensor
    
    def apply_intensity_scaling(self, tensor, intensity, frequency):
        """Apply intensity-based scaling to tensor."""
        # Scale tensor based on experimental intensity
        intensity_factor = intensity / 1000.0 if intensity > 0 else 0.001
        
        # Frequency-dependent scaling
        freq_factor = 1.0 + 0.001 * (frequency - 1000)  # Higher frequencies slightly enhanced
        
        return tensor * intensity_factor * freq_factor
    
    def apply_polarization_data(self, tensor):
        """Apply polarization data constraints if available."""
        if hasattr(self, 'depolarization_ratios') and self.depolarization_ratios:
            # Find matching depolarization data and modify tensor accordingly
            # This is a simplified implementation
            avg_depol = np.mean([data['ratio'] for data in self.depolarization_ratios.values()])
            
            # Adjust off-diagonal elements based on depolarization
            tensor[0, 1] *= avg_depol
            tensor[1, 0] *= avg_depol
            tensor[0, 2] *= avg_depol
            tensor[2, 0] *= avg_depol
            tensor[1, 2] *= avg_depol
            tensor[2, 1] *= avg_depol
        
        return tensor
    
    def calculate_additional_tensor_properties(self, tensor, crystal_system, character):
        """Calculate additional tensor properties beyond basic analysis."""
        try:
            # Raman scattering cross-section
            cross_section = np.sum(tensor**2)
            
            # Polarizability derivatives
            alpha_derivatives = np.diag(tensor)
            
            # Symmetry analysis
            symmetry_score = self.analyze_tensor_symmetry(tensor, crystal_system)
            
            # Mode classification
            mode_type = self.classify_vibrational_mode(tensor, character, crystal_system)
            
            # Intensity predictions for different polarization configurations
            intensity_predictions = self.predict_polarized_intensities(tensor)
            
            return {
                'cross_section': cross_section,
                'alpha_derivatives': alpha_derivatives,
                'symmetry_score': symmetry_score,
                'mode_type': mode_type,
                'intensity_predictions': intensity_predictions,
                'tensor_volume': np.linalg.det(tensor) if np.linalg.det(tensor) != 0 else 0
            }
            
        except Exception as e:
            return {'error': f"Error calculating additional properties: {str(e)}"}
    
    def analyze_tensor_symmetry(self, tensor, crystal_system):
        """Analyze how well the tensor matches expected crystal symmetry."""
        try:
            # Calculate symmetry measures
            if crystal_system == "Cubic":
                # Check isotropy
                diag_vals = np.diag(tensor)
                isotropy = 1.0 / (1.0 + np.std(diag_vals))
                off_diag_penalty = np.sum(np.abs(tensor - np.diag(np.diag(tensor))))
                return max(0, isotropy - off_diag_penalty * 0.1)
            
            elif crystal_system == "Hexagonal":
                # Check for a = b ≠ c pattern
                a_equals_b = 1.0 / (1.0 + abs(tensor[0,0] - tensor[1,1]))
                c_different = abs(tensor[2,2] - tensor[0,0]) / (abs(tensor[2,2]) + abs(tensor[0,0]) + 1e-6)
                return (a_equals_b + c_different) / 2.0
            
            else:
                # General symmetry measure based on crystal system
                return 0.5  # Default moderate symmetry
                
        except Exception:
            return 0.0
    
    def classify_vibrational_mode(self, tensor, character, crystal_system):
        """Classify the vibrational mode based on tensor properties."""
        if character != 'Unknown':
            return f"Assigned: {character}"
        
        # Classify based on tensor properties
        eigenvals = np.linalg.eigvals(tensor)
        max_eigenval = np.max(np.abs(eigenvals))
        min_eigenval = np.min(np.abs(eigenvals))
        
        if np.allclose(eigenvals, eigenvals[0], rtol=1e-2):
            return "Totally Symmetric (A1g-like)"
        elif np.sum(np.abs(eigenvals) < 1e-3) >= 1:
            return "Partially Active (Eg-like)"
        elif max_eigenval / min_eigenval > 5:
            return "Highly Anisotropic"
        else:
            return "Mixed Character"
    
    def predict_polarized_intensities(self, tensor):
        """Predict intensities for different polarization configurations."""
        try:
            # Common polarization configurations
            configs = {
                'parallel_xx': tensor[0, 0]**2,
                'parallel_yy': tensor[1, 1]**2,
                'parallel_zz': tensor[2, 2]**2,
                'cross_xy': tensor[0, 1]**2,
                'cross_xz': tensor[0, 2]**2,
                'cross_yz': tensor[1, 2]**2
            }
            
            # Calculate depolarization ratio
            parallel_avg = (configs['parallel_xx'] + configs['parallel_yy'] + configs['parallel_zz']) / 3
            cross_avg = (configs['cross_xy'] + configs['cross_xz'] + configs['cross_yz']) / 3
            
            depol_ratio = cross_avg / parallel_avg if parallel_avg > 0 else 0
            
            configs['depolarization_ratio'] = min(depol_ratio, 0.75)  # Cap at theoretical max
            
            return configs
            
        except Exception as e:
            return {'error': f"Error predicting intensities: {str(e)}"}
    
    def initialize_tensor_plot(self):
        """Initialize the tensor visualization plot."""
        self.tensor_fig.clear()
        
        # Create welcome text
        ax = self.tensor_fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Calculate Raman tensors to see visualization\n\n'
                           '1. Load spectrum and fit peaks\n'
                           '2. Select crystal system\n'
                           '3. Choose calculation method\n'
                           '4. Click "Calculate Raman Tensors"',
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        self.tensor_fig.suptitle("Raman Tensor Analysis & Visualization", fontsize=14, fontweight='bold')
        self.tensor_canvas.draw()
    
    def update_tensor_visualization(self):
        """Update the tensor visualization based on current settings."""
        if not hasattr(self, 'calculated_raman_tensors') or not self.calculated_raman_tensors:
            self.initialize_tensor_plot()
            return
        
        # Clear the figure
        self.tensor_fig.clear()
        
        # Get selected visualization mode from radio buttons
        selected_mode = self.viz_button_group.checkedId()
        
        # Determine subplot layout
        n_peaks = len(self.calculated_raman_tensors)
        if n_peaks == 0:
            self.initialize_tensor_plot()
            return
        
        # Create visualization based on selected mode
        if selected_mode == 0:  # Tensor Matrices
            self.plot_tensor_matrices()
        elif selected_mode == 1:  # Eigenvalues & Analysis
            self.plot_eigenvalue_analysis()
        elif selected_mode == 2:  # 3D Tensor Shapes
            self.plot_tensor_ellipsoids()
        elif selected_mode == 3:  # Overview Summary
            self.plot_tensor_overview()
        else:
            # Default fallback
            self.plot_tensor_matrices()
        
        self.tensor_fig.suptitle(f"Raman Tensor Analysis - {len(self.calculated_raman_tensors)} Peaks", 
                                fontsize=14, fontweight='bold')
        self.tensor_canvas.draw()
    
    def plot_tensor_matrices(self):
        """Plot tensor matrices as heatmaps."""
        n_tensors = len(self.calculated_raman_tensors)
        cols = min(3, n_tensors)
        rows = (n_tensors + cols - 1) // cols
        
        for i, (freq, data) in enumerate(sorted(self.calculated_raman_tensors.items())):
            ax = self.tensor_fig.add_subplot(rows, cols, i + 1)
            
            tensor = data['tensor']
            character = data.get('character', 'Unknown')
            
            # Create heatmap
            im = ax.imshow(tensor, cmap='RdBu_r', aspect='equal')
            
            # Add values as text
            for row in range(3):
                for col in range(3):
                    ax.text(col, row, f'{tensor[row, col]:.3f}', 
                           ha='center', va='center', fontsize=8)
            
            ax.set_title(f'{freq:.1f} cm⁻¹\n{character}', fontsize=10)
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(['x', 'y', 'z'])
            ax.set_yticklabels(['x', 'y', 'z'])
            
            # Add colorbar for first subplot
            if i == 0:
                self.tensor_fig.colorbar(im, ax=ax, shrink=0.6)
    
    def plot_eigenvalue_analysis(self):
        """Plot eigenvalue analysis."""
        frequencies = []
        eigenval_data = []
        anisotropies = []
        
        for freq, data in sorted(self.calculated_raman_tensors.items()):
            frequencies.append(freq)
            eigenvals = data['properties']['eigenvalues']
            eigenval_data.append(eigenvals)
            anisotropies.append(data['properties']['anisotropy'])
        
        ax1 = self.tensor_fig.add_subplot(2, 2, 1)
        ax2 = self.tensor_fig.add_subplot(2, 2, 2)
        
        # Eigenvalue plot
        eigenval_array = np.array(eigenval_data)
        for i in range(3):
            ax1.scatter(frequencies, eigenval_array[:, i], label=f'λ{i+1}', alpha=0.7)
        
        ax1.set_xlabel('Frequency (cm⁻¹)')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Tensor Eigenvalues')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Anisotropy plot
        ax2.scatter(frequencies, anisotropies, color='red', alpha=0.7)
        ax2.set_xlabel('Frequency (cm⁻¹)')
        ax2.set_ylabel('Anisotropy')
        ax2.set_title('Tensor Anisotropy')
        ax2.grid(True, alpha=0.3)
    
    def plot_tensor_ellipsoids(self):
        """Plot 3D Raman tensor shapes showing true angular dependence."""
        # Clear the figure
        self.tensor_fig.clear()
        
        n_tensors = len(self.calculated_raman_tensors)
        if n_tensors == 0:
            return
        
        # Create subplot grid - each tensor gets its own 3D plot
        n_cols = min(3, n_tensors)
        n_rows = (n_tensors + n_cols - 1) // n_cols
        
        for i, (freq, data) in enumerate(sorted(self.calculated_raman_tensors.items())):
            ax = self.tensor_fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
            
            tensor = data['tensor']
            character = data.get('character', 'Unknown')
            
            # Create angular dependence surface for Raman scattering
            # This shows the intensity |e_incident · R · e_scattered|² as a function of direction
            
            # Create spherical coordinate grid
            theta = np.linspace(0, np.pi, 40)  # polar angle
            phi = np.linspace(0, 2*np.pi, 60)  # azimuthal angle
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            
            # Convert to Cartesian unit vectors
            x_dir = np.sin(theta_grid) * np.cos(phi_grid)
            y_dir = np.sin(theta_grid) * np.sin(phi_grid)
            z_dir = np.cos(theta_grid)
            
            # Calculate Raman intensity for each direction
            # For backscattering geometry: I ∝ |e · R · e|² where e is the unit vector
            intensity = np.zeros_like(x_dir)
            
            for j in range(x_dir.shape[0]):
                for k in range(x_dir.shape[1]):
                    # Unit vector in this direction
                    e_vec = np.array([x_dir[j,k], y_dir[j,k], z_dir[j,k]])
                    
                    # Raman amplitude: e^T · R · e (keep sign for proper visualization)
                    raman_amplitude = np.dot(e_vec, np.dot(tensor, e_vec))
                    
                    # For B1g and other modes with nodes, preserve sign structure
                    # by using the raw amplitude rather than squaring immediately
                    if 'B1g' in character or 'B2g' in character:
                        # Keep the sign to show node structure (4-lobed patterns)
                        intensity[j,k] = raman_amplitude
                    else:
                        # For other modes, use squared intensity as normal
                        intensity[j,k] = raman_amplitude**2
            
            # Handle negative or complex values
            intensity = np.real(intensity)
            
            # For B1g and B2g modes, we want to show the node structure
            if 'B1g' in character or 'B2g' in character:
                # Keep both positive and negative values to show nodes
                max_abs_intensity = np.max(np.abs(intensity))
                if max_abs_intensity > 0:
                    intensity_norm = intensity / max_abs_intensity
                else:
                    intensity_norm = np.ones_like(intensity) * 0.1
                
                # Scale radius: positive values extend outward, negative values contract inward
                # Use absolute value for radius but preserve sign information for coloring
                radius_scale = 0.3 + 0.7 * np.abs(intensity_norm)  # Smaller base to show nodes clearly
            else:
                # For other modes, handle as before
                min_intensity = np.min(intensity)
                if min_intensity < 0:
                    # Shift to make all positive
                    intensity = intensity - min_intensity
                
                # Normalize intensity for visualization
                max_intensity = np.max(intensity)
                if max_intensity > 0:
                    intensity_norm = intensity / max_intensity
                else:
                    intensity_norm = np.ones_like(intensity) * 0.1
                
                # Scale the radius by intensity to create the 3D shape
                radius_scale = 0.5 + 0.5 * intensity_norm  # Base radius + intensity modulation
            
            # Calculate surface coordinates
            x_surface = radius_scale * x_dir
            y_surface = radius_scale * y_dir
            z_surface = radius_scale * z_dir
            
            # Color mapping based on intensity
            if 'B1g' in character or 'B2g' in character:
                # Use a diverging colormap for B1g/B2g to show positive (red) and negative (blue) lobes
                colors = plt.cm.RdBu_r(0.5 + 0.5 * intensity_norm)  # Red for positive, blue for negative
            else:
                # Regular color mapping for other modes
                colors = plt.cm.RdYlBu_r(0.5 + 0.5 * np.abs(intensity_norm))
            
            # Plot the 3D surface
            surf = ax.plot_surface(x_surface, y_surface, z_surface, 
                                 facecolors=colors, alpha=0.8, 
                                 linewidth=0, antialiased=True)
            
            # Add wireframe for structure
            ax.plot_wireframe(x_surface, y_surface, z_surface, 
                            color='gray', alpha=0.3, linewidth=0.5)
            
            # Add coordinate system arrows
            arrow_length = 1.2
            ax.quiver(0, 0, 0, arrow_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.8)
            ax.quiver(0, 0, 0, 0, arrow_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.8)
            ax.quiver(0, 0, 0, 0, 0, arrow_length, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.8)
            
            # Add tensor information
            eigenvals = np.linalg.eigvals(tensor)
            eigenvals_real = np.real(eigenvals)
            tensor_trace = np.trace(tensor)
            tensor_det = np.linalg.det(tensor)
            
            # Classify tensor shape
            shape_type = self.classify_tensor_shape(tensor, eigenvals_real)
            
            # Set equal aspect ratio
            max_extent = 1.5
            ax.set_xlim(-max_extent, max_extent)
            ax.set_ylim(-max_extent, max_extent)
            ax.set_zlim(-max_extent, max_extent)
            
            # Labels and title
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            
            title = f'{freq:.1f} cm⁻¹\n{character}\n{shape_type}'
            ax.set_title(title, fontsize=9, fontweight='bold')
            
            # Add tensor properties as text
            info_text = f'Tr={tensor_trace:.3f}\nDet={tensor_det:.3f}'
            ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                     fontsize=7, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add overall title  
        title_text = '3D Raman Tensor Angular Dependence\n'
        if any('B1g' in data.get('character', '') or 'B2g' in data.get('character', '') 
               for data in self.calculated_raman_tensors.values()):
            title_text += '(B1g/B2g: Red=Positive lobes, Blue=Negative lobes; Others: Blue=Low, Red=High intensity)'
        else:
            title_text += '(Color: Blue=Low, Red=High intensity)'
        
        self.tensor_fig.suptitle(title_text, fontsize=12, fontweight='bold')
        
        # Adjust layout
        self.tensor_fig.tight_layout()
    
    def classify_tensor_shape(self, tensor, eigenvals):
        """Classify the 3D shape of the Raman tensor."""
        # Sort eigenvalues by absolute value to handle signs properly
        abs_eigenvals = np.abs(eigenvals)
        sorted_indices = np.argsort(abs_eigenvals)[::-1]  # Descending order
        sorted_eigenvals = eigenvals[sorted_indices]
        
        # Count positive and negative eigenvalues
        n_positive = np.sum(eigenvals > 1e-6)
        n_negative = np.sum(eigenvals < -1e-6)
        n_zero = np.sum(np.abs(eigenvals) <= 1e-6)
        
        # Tolerance for comparing eigenvalues
        tol = 0.1
        
        # Classify shape based on eigenvalue structure
        if n_zero >= 2:
            return "Planar/Linear"
        elif n_zero == 1:
            if n_positive == 2:
                return "Elliptic Cylinder"
            elif n_negative == 2:
                return "Hyperbolic Cylinder"
            else:
                return "Parabolic Cylinder"
        elif n_positive == 3:
            # All positive eigenvalues - various ellipsoid types
            eigenval_0 = abs(sorted_eigenvals[0])
            eigenval_1 = abs(sorted_eigenvals[1])
            eigenval_2 = abs(sorted_eigenvals[2])
            
            # Check if all eigenvalues are approximately equal (sphere)
            if (abs(eigenval_0 - eigenval_1) < tol * eigenval_0 and 
                abs(eigenval_1 - eigenval_2) < tol * eigenval_1):
                return "Sphere"
            # Check if first two are equal (oblate - flattened)
            elif abs(eigenval_0 - eigenval_1) < tol * eigenval_0:
                return "Oblate Ellipsoid"
            # Check if last two are equal (prolate - elongated)
            elif abs(eigenval_1 - eigenval_2) < tol * eigenval_1:
                return "Prolate Ellipsoid"
            else:
                return "Triaxial Ellipsoid"
        elif n_negative == 3:
            return "Inverted Ellipsoid"
        elif n_positive == 2 and n_negative == 1:
            return "Hyperboloid (1 sheet)"
        elif n_positive == 1 and n_negative == 2:
            return "Hyperboloid (2 sheet)"
        elif n_positive == 1 and n_negative == 1:
            return "Hyperbolic Paraboloid"
        else:
            return "Complex Shape"
    
    def plot_tensor_overview(self):
        """Plot tensor overview with key properties."""
        ax = self.tensor_fig.add_subplot(1, 1, 1)
        
        frequencies = []
        traces = []
        anisotropies = []
        cross_sections = []
        
        for freq, data in sorted(self.calculated_raman_tensors.items()):
            frequencies.append(freq)
            traces.append(data['properties']['trace'])
            anisotropies.append(data['properties']['anisotropy'])
            
            if 'additional_properties' in data:
                cross_sections.append(data['additional_properties']['cross_section'])
            else:
                cross_sections.append(0)
        
        # Normalize data for comparison
        if cross_sections and max(cross_sections) > 0:
            cross_sections_norm = [x / max(cross_sections) for x in cross_sections]
        else:
            cross_sections_norm = [0] * len(cross_sections)
        
        # Plot multiple properties
        ax.scatter(frequencies, traces, alpha=0.7, s=50, label='Trace', color='blue')
        ax.scatter(frequencies, anisotropies, alpha=0.7, s=50, label='Anisotropy', color='red')
        ax.scatter(frequencies, cross_sections_norm, alpha=0.7, s=50, label='Cross Section (norm)', color='green')
        
        ax.set_xlabel('Frequency (cm⁻¹)')
        ax.set_ylabel('Property Value')
        ax.set_title('Tensor Properties Overview')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def auto_assign_mode_characters(self):
        """Automatically assign vibrational mode characters after tensor calculation."""
        if not hasattr(self, 'calculated_raman_tensors') or not self.calculated_raman_tensors:
            return
        
        try:
            # Get crystal system
            crystal_system = self.tensor_crystal_system_combo.currentText()
            
            # Auto-assign modes based on tensor properties and crystal system
            for freq, data in self.calculated_raman_tensors.items():
                tensor = data['tensor']
                current_character = data.get('character', 'Unknown')
                
                if current_character == 'Unknown':
                    # Assign based on tensor analysis
                    assigned_character = self.predict_mode_character(tensor, crystal_system, freq)
                    
                    # Update the stored data
                    data['character'] = assigned_character
                    
                    # Update the tensor with proper symmetry constraints for the assigned character
                    updated_tensor = self.apply_symmetry_constraints(tensor, crystal_system, assigned_character)
                    data['tensor'] = updated_tensor
                    
        except Exception as e:
            print(f"Error in auto-assign mode characters: {str(e)}")

    def assign_vibrational_modes(self):
        """Assign vibrational modes to calculated tensors."""
        if not hasattr(self, 'calculated_raman_tensors') or not self.calculated_raman_tensors:
            QMessageBox.warning(self, "Warning", "Please calculate Raman tensors first.")
            return
        
        try:
            # Get crystal system
            crystal_system = self.tensor_crystal_system_combo.currentText()
            
            # Auto-assign modes based on tensor properties and crystal system
            mode_assignments = {}
            
            for freq, data in self.calculated_raman_tensors.items():
                tensor = data['tensor']
                current_character = data.get('character', 'Unknown')
                
                if current_character == 'Unknown':
                    # Assign based on tensor analysis
                    assigned_character = self.predict_mode_character(tensor, crystal_system, freq)
                    mode_assignments[freq] = assigned_character
                    
                    # Update the stored data
                    data['character'] = assigned_character
                else:
                    mode_assignments[freq] = current_character
            
            # Show assignment results
            self.show_mode_assignment_results(mode_assignments)
            
            # Update visualization
            self.update_tensor_visualization()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error assigning vibrational modes: {str(e)}")
    
    def predict_mode_character(self, tensor, crystal_system, frequency):
        """Predict vibrational mode character from tensor properties."""
        eigenvals = np.linalg.eigvals(tensor)
        eigenvals_real = np.real(eigenvals)
        eigenvals_sorted = np.sort(np.abs(eigenvals_real))[::-1]  # Sort by magnitude, descending
        
        # For Anatase (Tetragonal), use known frequency ranges and tensor patterns
        if crystal_system == "Tetragonal":
            # Anatase-specific frequency ranges and patterns
            if 100 <= frequency <= 150:
                # B1g mode around 115 cm⁻¹
                return "B1g"
            elif 150 <= frequency <= 200:
                # Ag mode around 158 cm⁻¹  
                return "Ag"
            elif 270 <= frequency <= 300:
                # B1g mode around 283 cm⁻¹
                return "B1g"
            elif 800 <= frequency <= 850:
                # Ag mode around 818 cm⁻¹
                return "Ag"
            elif frequency > 500:
                # High frequency modes are typically A1g or Ag
                if abs(tensor[2, 2]) > max(abs(tensor[0, 0]), abs(tensor[1, 1])):
                    return "Ag"
                else:
                    return "B1g"
            else:
                # Use tensor signature to classify
                # B1g: strong (xx - yy) character, weak zz
                xx_yy_diff = abs(tensor[0, 0] - tensor[1, 1])
                zz_component = abs(tensor[2, 2])
                xy_component = abs(tensor[0, 1])
                
                if xx_yy_diff > 0.5 * max(abs(tensor[0, 0]), abs(tensor[1, 1])) and zz_component < 0.3 * xx_yy_diff:
                    return "B1g"
                elif zz_component > max(abs(tensor[0, 0]), abs(tensor[1, 1])):
                    return "Ag"
                elif xy_component > 0.5 * max(abs(tensor[0, 0]), abs(tensor[1, 1])):
                    return "B2g"
                else:
                    return "Eg"
        
        elif crystal_system == "Hexagonal":
            if abs(tensor[2, 2]) > abs(tensor[0, 0]):
                return "A1g"
            elif abs(tensor[0, 0] - (-tensor[0, 0])) < 0.2 * abs(tensor[0, 0]):
                return "E2g"
            else:
                return "E1g"
                
        elif crystal_system == "Trigonal":
            # Similar to Hexagonal but with trigonal symmetry
            if abs(tensor[2, 2]) > abs(tensor[0, 0]):
                return "Ag"
            else:
                return "Eg"
        
        elif crystal_system == "Cubic":
            if np.allclose(eigenvals_sorted, [eigenvals_sorted[0]] * len(eigenvals_sorted), rtol=0.1):
                return "A1g"
            else:
                return "T2g"
        
        else:
            # General classification based on frequency and tensor
            if frequency < 300:
                return "Low-freq mode"
            elif frequency > 1000:
                return "High-freq mode"
            else:
                return "Mid-freq mode"
    
    def show_mode_assignment_results(self, mode_assignments):
        """Show mode assignment results in a dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Vibrational Mode Assignments")
        dialog.setModal(True)
        dialog.resize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title_label = QLabel("Vibrational Mode Character Assignments")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create table
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Frequency (cm⁻¹)", "Assigned Character", "Confidence"])
        
        table.setRowCount(len(mode_assignments))
        for row, (freq, character) in enumerate(sorted(mode_assignments.items())):
            table.setItem(row, 0, QTableWidgetItem(f"{freq:.1f}"))
            table.setItem(row, 1, QTableWidgetItem(character))
            table.setItem(row, 2, QTableWidgetItem("Auto-assigned"))
        
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def show_mode_assignments(self):
        """Show current mode assignments."""
        if not hasattr(self, 'calculated_raman_tensors') or not self.calculated_raman_tensors:
            QMessageBox.warning(self, "Warning", "No tensor data available.")
            return
        
        mode_data = {}
        for freq, data in self.calculated_raman_tensors.items():
            mode_data[freq] = data.get('character', 'Unknown')
        
        self.show_mode_assignment_results(mode_data)
    
    def export_tensor_data(self):
        """Export tensor data to file."""
        if not hasattr(self, 'calculated_raman_tensors') or not self.calculated_raman_tensors:
            QMessageBox.warning(self, "Warning", "No tensor data to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Tensor Data", 
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt);;JSON files (*.json);;CSV files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.json'):
                self.export_tensor_json(file_path)
            elif file_path.endswith('.csv'):
                self.export_tensor_csv(file_path)
            else:
                self.export_tensor_txt(file_path)
            
            QMessageBox.information(self, "Success", "Tensor data exported successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting tensor data: {str(e)}")
    
    def export_tensor_txt(self, file_path):
        """Export tensor data as formatted text file."""
        with open(file_path, 'w') as f:
            f.write("RAMAN TENSOR ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            if hasattr(self, 'tensor_analysis_results'):
                results = self.tensor_analysis_results
                f.write(f"Crystal System: {results.get('crystal_system', 'Unknown')}\n")
                f.write(f"Calculation Method: {results.get('calculation_method', 'Unknown')}\n")
                f.write(f"Number of Peaks: {results.get('peak_count', 0)}\n")
                f.write(f"Analysis Date: {results.get('timestamp', 'Unknown')}\n\n")
            
            for freq, data in sorted(self.calculated_raman_tensors.items()):
                f.write(f"Peak at {freq:.1f} cm⁻¹\n")
                f.write("-" * 30 + "\n")
                f.write(f"Character: {data.get('character', 'Unknown')}\n")
                f.write(f"Intensity: {data.get('intensity', 0):.3f}\n")
                f.write(f"Width: {data.get('width', 0):.3f}\n\n")
                
                f.write("Raman Tensor Matrix:\n")
                tensor = data['tensor']
                for i in range(3):
                    f.write("  [")
                    for j in range(3):
                        f.write(f"{tensor[i,j]:8.4f}")
                    f.write("]\n")
                f.write("\n")
                
                props = data['properties']
                f.write(f"Eigenvalues: {props['eigenvalues']}\n")
                f.write(f"Trace: {props['trace']:.4f}\n")
                f.write(f"Anisotropy: {props['anisotropy']:.4f}\n")
                f.write(f"Tensor Norm: {props['tensor_norm']:.4f}\n\n")
    
    def export_tensor_json(self, file_path):
        """Export tensor data as JSON file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        for freq, data in self.calculated_raman_tensors.items():
            export_data[str(freq)] = {
                'tensor': data['tensor'].tolist(),
                'properties': {
                    'eigenvalues': data['properties']['eigenvalues'].tolist(),
                    'trace': float(data['properties']['trace']),
                    'anisotropy': float(data['properties']['anisotropy']),
                    'tensor_norm': float(data['properties']['tensor_norm'])
                },
                'character': data.get('character', 'Unknown'),
                'intensity': float(data.get('intensity', 0)),
                'width': float(data.get('width', 0)),
                'crystal_system': data.get('crystal_system', 'Unknown')
            }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_tensor_csv(self, file_path):
        """Export tensor data as CSV file."""
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Frequency', 'Character', 'Intensity', 'Width',
                'T_xx', 'T_xy', 'T_xz', 'T_yx', 'T_yy', 'T_yz', 'T_zx', 'T_zy', 'T_zz',
                'Eigenval_1', 'Eigenval_2', 'Eigenval_3', 'Trace', 'Anisotropy', 'Tensor_Norm'
            ])
            
            # Data rows
            for freq, data in sorted(self.calculated_raman_tensors.items()):
                tensor = data['tensor']
                props = data['properties']
                
                row = [
                    freq, data.get('character', 'Unknown'), data.get('intensity', 0), data.get('width', 0)
                ]
                
                # Tensor elements
                for i in range(3):
                    for j in range(3):
                        row.append(tensor[i, j])
                
                # Properties
                eigenvals = props['eigenvalues']
                row.extend([eigenvals[0], eigenvals[1], eigenvals[2]])
                row.extend([props['trace'], props['anisotropy'], props['tensor_norm']])
                
                writer.writerow(row)
    
    def export_tensor_visualization(self):
        """Export current tensor visualization."""
        if not hasattr(self, 'calculated_raman_tensors') or not self.calculated_raman_tensors:
            QMessageBox.warning(self, "Warning", "No tensor visualization to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Tensor Visualization",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )
        
        if not file_path:
            return
        
        try:
            self.tensor_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Success", "Tensor visualization exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting visualization: {str(e)}")

    def open_individual_tensor_windows(self):
        """Open each tensor in its own dedicated window."""
        if not hasattr(self, 'calculated_raman_tensors') or not self.calculated_raman_tensors:
            QMessageBox.warning(self, "Warning", "Please calculate Raman tensors first.")
            return
        
        # Create individual windows for each tensor
        for freq, data in sorted(self.calculated_raman_tensors.items()):
            self.create_individual_tensor_window(freq, data)

    def create_individual_tensor_window(self, frequency, tensor_data):
        """Create a dedicated window for a single tensor."""
        # Create new window
        window = QMainWindow()
        window.setWindowTitle(f"Raman Tensor - {frequency:.1f} cm⁻¹ ({tensor_data.get('character', 'Unknown')})")
        window.setGeometry(100, 100, 900, 700)
        
        # Create central widget
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tabs for different visualizations
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Tab 1: Tensor Matrix
        matrix_widget = QWidget()
        tab_widget.addTab(matrix_widget, "Tensor Matrix")
        matrix_layout = QVBoxLayout(matrix_widget)
        
        matrix_fig = Figure(figsize=(6, 5))
        matrix_canvas = FigureCanvas(matrix_fig)
        matrix_toolbar = NavigationToolbar(matrix_canvas, matrix_widget)
        self.apply_toolbar_styling(matrix_toolbar)
        
        matrix_layout.addWidget(matrix_toolbar)
        matrix_layout.addWidget(matrix_canvas)
        
        # Plot tensor matrix
        self.plot_single_tensor_matrix(matrix_fig, frequency, tensor_data)
        matrix_canvas.draw()
        
        # Tab 2: 3D Shape (if B1g or other interesting symmetry)
        shape_widget = QWidget()
        tab_widget.addTab(shape_widget, "3D Shape")
        shape_layout = QVBoxLayout(shape_widget)
        
        shape_fig = Figure(figsize=(8, 6))
        shape_canvas = FigureCanvas(shape_fig)
        shape_toolbar = NavigationToolbar(shape_canvas, shape_widget)
        self.apply_toolbar_styling(shape_toolbar)
        
        shape_layout.addWidget(shape_toolbar)
        shape_layout.addWidget(shape_canvas)
        
        # Plot 3D tensor shape
        self.plot_single_tensor_shape(shape_fig, frequency, tensor_data)
        shape_canvas.draw()
        
        # Tab 3: Properties
        props_widget = QWidget()
        tab_widget.addTab(props_widget, "Properties")
        props_layout = QVBoxLayout(props_widget)
        
        # Create properties text
        props_text = QTextEdit()
        props_text.setReadOnly(True)
        props_text.setFont(QFont("Courier", 10))
        props_layout.addWidget(props_text)
        
        # Generate properties text
        properties_text = self.generate_tensor_properties_text(frequency, tensor_data)
        props_text.setPlainText(properties_text)
        
        # Add buttons for export
        button_layout = QHBoxLayout()
        export_matrix_btn = QPushButton("Export Matrix Plot")
        export_shape_btn = QPushButton("Export 3D Shape")
        export_props_btn = QPushButton("Export Properties")
        close_btn = QPushButton("Close")
        
        export_matrix_btn.clicked.connect(lambda: self.export_single_plot(matrix_fig, f"tensor_matrix_{frequency:.1f}cm"))
        export_shape_btn.clicked.connect(lambda: self.export_single_plot(shape_fig, f"tensor_shape_{frequency:.1f}cm"))
        export_props_btn.clicked.connect(lambda: self.export_single_properties(properties_text, f"tensor_props_{frequency:.1f}cm"))
        close_btn.clicked.connect(window.close)
        
        button_layout.addWidget(export_matrix_btn)
        button_layout.addWidget(export_shape_btn)
        button_layout.addWidget(export_props_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        # Show window
        window.show()
        
        # Store reference to prevent garbage collection
        if not hasattr(self, 'individual_tensor_windows'):
            self.individual_tensor_windows = []
        self.individual_tensor_windows.append(window)

    def plot_single_tensor_matrix(self, fig, frequency, tensor_data):
        """Plot a single tensor matrix with detailed information."""
        ax = fig.add_subplot(1, 1, 1)
        
        tensor = tensor_data['tensor']
        character = tensor_data.get('character', 'Unknown')
        
        # Create enhanced heatmap
        im = ax.imshow(tensor, cmap='RdBu_r', aspect='equal', vmin=-np.max(np.abs(tensor)), vmax=np.max(np.abs(tensor)))
        
        # Add values as text
        for row in range(3):
            for col in range(3):
                color = 'white' if abs(tensor[row, col]) > 0.5 * np.max(np.abs(tensor)) else 'black'
                ax.text(col, row, f'{tensor[row, col]:.4f}', 
                       ha='center', va='center', fontsize=12, fontweight='bold', color=color)
        
        ax.set_title(f'Raman Tensor Matrix\n{frequency:.1f} cm⁻¹ - {character}', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['x', 'y', 'z'], fontsize=12)
        ax.set_yticklabels(['x', 'y', 'z'], fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Tensor Element Value', fontsize=12)
        
        # Add eigenvalue information as text
        eigenvals = tensor_data['properties']['eigenvalues']
        trace = tensor_data['properties']['trace']
        anisotropy = tensor_data['properties']['anisotropy']
        
        info_text = f'Eigenvalues: {eigenvals[0]:.3f}, {eigenvals[1]:.3f}, {eigenvals[2]:.3f}\n'
        info_text += f'Trace: {trace:.3f}\n'
        info_text += f'Anisotropy: {anisotropy:.3f}'
        
        ax.text(1.05, 0.5, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        fig.tight_layout()

    def plot_single_tensor_shape(self, fig, frequency, tensor_data):
        """Plot a single tensor 3D shape with enhanced details."""
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        tensor = tensor_data['tensor']
        character = tensor_data.get('character', 'Unknown')
        
        # Create high-resolution spherical coordinate grid for detailed visualization
        theta = np.linspace(0, np.pi, 60)  # polar angle
        phi = np.linspace(0, 2*np.pi, 80)  # azimuthal angle
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        # Convert to Cartesian unit vectors
        x_dir = np.sin(theta_grid) * np.cos(phi_grid)
        y_dir = np.sin(theta_grid) * np.sin(phi_grid)
        z_dir = np.cos(theta_grid)
        
        # Calculate Raman intensity for each direction
        intensity = np.zeros_like(x_dir)
        
        for j in range(x_dir.shape[0]):
            for k in range(x_dir.shape[1]):
                # Unit vector in this direction
                e_vec = np.array([x_dir[j,k], y_dir[j,k], z_dir[j,k]])
                
                # Raman amplitude: e^T · R · e (keep sign for proper visualization)
                raman_amplitude = np.dot(e_vec, np.dot(tensor, e_vec))
                
                # For B1g and other modes with nodes, preserve sign structure
                if 'B1g' in character or 'B2g' in character:
                    intensity[j,k] = raman_amplitude
                else:
                    intensity[j,k] = raman_amplitude**2
        
        # Handle the signed intensity for B1g (preserve nodes)
        intensity = np.real(intensity)
        
        if 'B1g' in character or 'B2g' in character:
            max_abs_intensity = np.max(np.abs(intensity))
            if max_abs_intensity > 0:
                intensity_norm = intensity / max_abs_intensity
            else:
                intensity_norm = np.ones_like(intensity) * 0.1
            radius_scale = 0.3 + 0.7 * np.abs(intensity_norm)
            colors = plt.cm.RdBu_r(0.5 + 0.5 * intensity_norm)
        else:
            min_intensity = np.min(intensity)
            if min_intensity < 0:
                intensity = intensity - min_intensity
            max_intensity = np.max(intensity)
            if max_intensity > 0:
                intensity_norm = intensity / max_intensity
            else:
                intensity_norm = np.ones_like(intensity) * 0.1
            radius_scale = 0.5 + 0.5 * intensity_norm
            colors = plt.cm.RdYlBu_r(0.5 + 0.5 * intensity_norm)
        
        # Calculate surface coordinates
        x_surface = radius_scale * x_dir
        y_surface = radius_scale * y_dir
        z_surface = radius_scale * z_dir
        
        # Plot the 3D surface
        surf = ax.plot_surface(x_surface, y_surface, z_surface, 
                             facecolors=colors, alpha=0.9, 
                             linewidth=0, antialiased=True)
        
        # Add coordinate system arrows
        arrow_length = 1.5
        ax.quiver(0, 0, 0, arrow_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=3, alpha=0.9)
        ax.quiver(0, 0, 0, 0, arrow_length, 0, color='green', arrow_length_ratio=0.1, linewidth=3, alpha=0.9)
        ax.quiver(0, 0, 0, 0, 0, arrow_length, color='blue', arrow_length_ratio=0.1, linewidth=3, alpha=0.9)
        
        # Add axis labels
        ax.text(arrow_length*1.1, 0, 0, 'X', fontsize=12, fontweight='bold', color='red')
        ax.text(0, arrow_length*1.1, 0, 'Y', fontsize=12, fontweight='bold', color='green')
        ax.text(0, 0, arrow_length*1.1, 'Z', fontsize=12, fontweight='bold', color='blue')
        
        # Set equal aspect ratio
        max_extent = 1.8
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.set_zlim(-max_extent, max_extent)
        
        # Enhanced title with character information
        title = f'3D Raman Tensor Shape\n{frequency:.1f} cm⁻¹ - {character}'
        if 'B1g' in character:
            title += '\n(Red=Positive lobes, Blue=Negative lobes)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        fig.tight_layout()

    def generate_tensor_properties_text(self, frequency, tensor_data):
        """Generate detailed text description of tensor properties."""
        tensor = tensor_data['tensor']
        properties = tensor_data['properties']
        character = tensor_data.get('character', 'Unknown')
        crystal_system = tensor_data.get('crystal_system', 'Unknown')
        
        text = f"RAMAN TENSOR PROPERTIES\n"
        text += f"=" * 50 + "\n\n"
        text += f"Frequency: {frequency:.1f} cm⁻¹\n"
        text += f"Character: {character}\n"
        text += f"Crystal System: {crystal_system}\n"
        text += f"Calculation Method: {tensor_data.get('calculation_method', 'Unknown')}\n\n"
        
        text += f"TENSOR MATRIX:\n"
        text += f"-" * 20 + "\n"
        for i in range(3):
            row_text = "  ".join([f"{tensor[i,j]:8.4f}" for j in range(3)])
            text += f"[{row_text}]\n"
        text += "\n"
        
        text += f"EIGENVALUE ANALYSIS:\n"
        text += f"-" * 20 + "\n"
        eigenvals = properties['eigenvalues']
        for i, eval in enumerate(eigenvals):
            text += f"λ{i+1}: {eval:10.4f}\n"
        text += "\n"
        
        text += f"TENSOR INVARIANTS:\n"
        text += f"-" * 20 + "\n"
        text += f"Trace (I₁):      {properties['trace']:10.4f}\n"
        text += f"Determinant:     {properties['determinant']:10.4f}\n"
        text += f"Anisotropy:      {properties['anisotropy']:10.4f}\n"
        text += f"Tensor Norm:     {properties['tensor_norm']:10.4f}\n"
        text += f"Spherical Part:  {properties['spherical_part']:10.4f}\n"
        text += f"Deviatoric Norm: {properties['deviatoric_norm']:10.4f}\n\n"
        
        if 'additional_properties' in tensor_data:
            add_props = tensor_data['additional_properties']
            text += f"ADDITIONAL PROPERTIES:\n"
            text += f"-" * 20 + "\n"
            text += f"Cross Section:   {add_props.get('cross_section', 0):10.4f}\n"
            text += f"Symmetry Score:  {add_props.get('symmetry_score', 0):10.4f}\n"
            text += f"Mode Type:       {add_props.get('mode_type', 'Unknown')}\n\n"
        
        # Add symmetry-specific information
        if character == 'B1g':
            text += f"B1g MODE CHARACTERISTICS:\n"
            text += f"-" * 25 + "\n"
            text += f"• (x²-y²) symmetry character\n"
            text += f"• Creates 4-lobed angular dependence pattern\n"
            text += f"• Positive lobes along x-direction\n"
            text += f"• Negative lobes along y-direction\n"
            text += f"• Nodes at ±45° in xy-plane\n"
            text += f"• Typical of tetragonal crystals\n\n"
        
        return text

    def export_single_plot(self, fig, filename_base):
        """Export a single plot to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Export {filename_base}", 
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) + f"/{filename_base}.png",
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )
        
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting plot: {str(e)}")

    def export_single_properties(self, properties_text, filename_base):
        """Export tensor properties to text file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Export {filename_base} Properties", 
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) + f"/{filename_base}.txt",
            "Text files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(properties_text)
                QMessageBox.information(self, "Success", f"Properties exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting properties: {str(e)}")

    def on_structure_loaded(self, structure_data):
        """Handle crystal structure loaded event from the crystal structure widget."""
        print(f"Structure loaded: {structure_data.get('formula', 'Unknown')}")
        
        # Store the structure data so it can be accessed by other components
        self.current_crystal_structure = structure_data
        self.crystal_structure = structure_data  # Also store as crystal_structure for backward compatibility
        
        # Automatically update the tensor crystal system combo if available
        if hasattr(self, 'tensor_crystal_system_combo') and structure_data:
            crystal_system = structure_data.get('crystal_system', 'Unknown')
            if crystal_system != 'Unknown':
                # Find matching item in combo box
                for i in range(self.tensor_crystal_system_combo.count()):
                    if self.tensor_crystal_system_combo.itemText(i).lower() == crystal_system.lower():
                        self.tensor_crystal_system_combo.setCurrentIndex(i)
                        print(f"✓ Auto-updated tensor crystal system to: {crystal_system}")
                        break

    def on_bonds_calculated(self, bond_data):
        """Handle bond calculation event from the crystal structure widget."""
        print(f"Bonds calculated: {bond_data.get('count', 0)}")


def main():
    """Main function to run the Qt6 application."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Raman Polarization Analyzer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("RamanLab")
    
    # Create and show main window
    window = RamanPolarizationAnalyzerQt6()
    window.show()
    
    # Center window on screen
    screen = app.primaryScreen().geometry()
    window_size = window.geometry()
    x = (screen.width() - window_size.width()) // 2
    y = (screen.height() - window_size.height()) // 2
    window.move(x, y)
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 