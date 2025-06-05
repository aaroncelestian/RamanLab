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
            "Polarization",
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
        elif tab_name == "Polarization":
            self.setup_polarization_tab(side_panel, content_area)
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
        """Setup the Peak Fitting tab."""
        # Side panel layout
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Peak Fitting")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Peak selection group
        peak_group = QGroupBox("Peak Selection")
        peak_layout = QVBoxLayout(peak_group)
        
        self.peak_selection_btn = QPushButton("Toggle Peak Selection Mode")
        self.peak_selection_btn.clicked.connect(self.toggle_peak_selection)
        peak_layout.addWidget(self.peak_selection_btn)
        
        clear_peaks_btn = QPushButton("Clear Selected Peaks")
        clear_peaks_btn.clicked.connect(self.clear_selected_peaks)
        peak_layout.addWidget(clear_peaks_btn)
        
        side_layout.addWidget(peak_group)
        
        # Fitting options group
        fitting_group = QGroupBox("Fitting Options")
        fitting_layout = QVBoxLayout(fitting_group)
        
        # Peak shape selection
        shape_label = QLabel("Peak Shape:")
        fitting_layout.addWidget(shape_label)
        
        self.peak_shape_combo = QComboBox()
        self.peak_shape_combo.addItems(["Lorentzian", "Gaussian", "Voigt"])
        fitting_layout.addWidget(self.peak_shape_combo)
        
        # Fit button
        fit_btn = QPushButton("Fit Selected Peaks")
        fit_btn.clicked.connect(self.fit_peaks)
        fitting_layout.addWidget(fit_btn)
        
        side_layout.addWidget(fitting_group)
        
        # Peak matching group
        matching_group = QGroupBox("Peak Matching & Labeling")
        matching_layout = QVBoxLayout(matching_group)
        
        # Match peaks button
        match_peaks_btn = QPushButton("Match with Reference Peaks")
        match_peaks_btn.clicked.connect(self.match_peaks_with_reference)
        matching_layout.addWidget(match_peaks_btn)
        
        # Tolerance setting
        tolerance_layout = QHBoxLayout()
        tolerance_layout.addWidget(QLabel("Tolerance (cm⁻¹):"))
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(5, 200)
        self.tolerance_spin.setValue(50)
        self.tolerance_spin.valueChanged.connect(self.update_matching_tolerance)
        tolerance_layout.addWidget(self.tolerance_spin)
        matching_layout.addLayout(tolerance_layout)
        
        # Auto-assign labels button
        assign_labels_btn = QPushButton("Auto-Assign Peak Labels")
        assign_labels_btn.clicked.connect(self.auto_assign_peak_labels)
        matching_layout.addWidget(assign_labels_btn)
        
        # Show assignments button
        show_assignments_btn = QPushButton("Show Peak Assignments")
        show_assignments_btn.clicked.connect(self.show_peak_assignments)
        matching_layout.addWidget(show_assignments_btn)
        
        # Show Raman activity info button
        activity_info_btn = QPushButton("Show Raman Activity Filter")
        activity_info_btn.clicked.connect(self.show_raman_activity_info)
        activity_info_btn.setStyleSheet("QPushButton { background-color: #e7f3ff; }")
        matching_layout.addWidget(activity_info_btn)
        
        # Debug calculated modes button
        debug_modes_btn = QPushButton("Debug Calculated Modes")
        debug_modes_btn.clicked.connect(self.debug_calculated_modes)
        debug_modes_btn.setStyleSheet("QPushButton { background-color: #fff3cd; }")
        matching_layout.addWidget(debug_modes_btn)
        
        # NEW: Debug database content button
        debug_db_btn = QPushButton("Debug Database Content")
        debug_db_btn.clicked.connect(self.debug_database_content)
        debug_db_btn.setStyleSheet("QPushButton { background-color: #d4edda; }")
        matching_layout.addWidget(debug_db_btn)
        
        side_layout.addWidget(matching_group)
        
        # Reference mineral group
        ref_group = QGroupBox("Reference Mineral")
        ref_layout = QVBoxLayout(ref_group)
        
        self.reference_combo = QComboBox()
        self.reference_combo.currentTextChanged.connect(self.on_reference_mineral_changed)
        ref_layout.addWidget(self.reference_combo)
        
        self.reference_status_label = QLabel("No reference selected")
        self.reference_status_label.setStyleSheet("color: gray; font-style: italic;")
        ref_layout.addWidget(self.reference_status_label)
        
        side_layout.addWidget(ref_group)
        
        # Add stretch
        side_layout.addStretch()
        
        # Content area - matplotlib plot
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
        
        # Initialize empty plot
        self.update_peak_fitting_plot()
    
    def setup_polarization_tab(self, side_panel, content_area):
        """Setup the Polarization tab using the new comprehensive module."""
        try:
            from ui.polarization_analysis import PolarizationAnalysisWidget
            
            # Create the main polarization analysis widget
            self.polarization_widget = PolarizationAnalysisWidget()
            
            # Replace content area with the new widget
            content_layout = QVBoxLayout(content_area)
            content_layout.addWidget(self.polarization_widget)
            
            # Hide the side panel since the new widget has its own controls
            side_panel.hide()
            
            # Connect signals if needed
            self.polarization_widget.analysis_complete.connect(self.on_polarization_analysis_complete)
            self.polarization_widget.tensor_calculated.connect(self.on_tensor_calculated)
            
        except ImportError as e:
            print(f"⚠ Could not load new polarization module: {e}")
            # Fallback to the old implementation
            self.setup_polarization_tab_fallback(side_panel, content_area)
    
    def setup_polarization_tab_fallback(self, side_panel, content_area):
        """Fallback polarization tab setup."""
        # Side panel layout
        side_layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Polarization Analysis")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Polarization configuration group
        pol_group = QGroupBox("Polarization Configuration")
        pol_layout = QVBoxLayout(pol_group)
        
        # Crystal system selection
        system_label = QLabel("Crystal System:")
        pol_layout.addWidget(system_label)
        
        self.crystal_system_combo = QComboBox()
        crystal_systems = ["Cubic", "Tetragonal", "Hexagonal", "Orthorhombic", "Monoclinic", "Triclinic"]
        self.crystal_system_combo.addItems(crystal_systems)
        pol_layout.addWidget(self.crystal_system_combo)
        
        # Polarization angles
        angle_label = QLabel("Polarization Angles (degrees):")
        pol_layout.addWidget(angle_label)
        
        angle_layout = QHBoxLayout()
        self.incident_angle_spin = QDoubleSpinBox()
        self.incident_angle_spin.setRange(0, 360)
        self.incident_angle_spin.setSuffix("°")
        self.scattered_angle_spin = QDoubleSpinBox()
        self.scattered_angle_spin.setRange(0, 360)
        self.scattered_angle_spin.setSuffix("°")
        
        angle_layout.addWidget(QLabel("Incident:"))
        angle_layout.addWidget(self.incident_angle_spin)
        angle_layout.addWidget(QLabel("Scattered:"))
        angle_layout.addWidget(self.scattered_angle_spin)
        
        pol_layout.addLayout(angle_layout)
        
        # Calculate button
        calc_pol_btn = QPushButton("Calculate Polarization")
        calc_pol_btn.clicked.connect(self.calculate_polarization)
        pol_layout.addWidget(calc_pol_btn)
        
        side_layout.addWidget(pol_group)
        
        # Add stretch
        side_layout.addStretch()
        
        # Content area - placeholder for now
        content_layout = QVBoxLayout(content_area)
        
        pol_info_label = QLabel("Polarization analysis visualization will be implemented here")
        pol_info_label.setAlignment(Qt.AlignCenter)
        pol_info_label.setStyleSheet("color: gray; font-style: italic;")
        content_layout.addWidget(pol_info_label)
    
    def on_polarization_analysis_complete(self, results):
        """Handle completion of polarization analysis."""
        print(f"✓ Polarization analysis completed with {len(results)} results")
    
    def on_tensor_calculated(self, tensor_data):
        """Handle completion of tensor calculations."""
        print(f"✓ Raman tensors calculated for {len(tensor_data)} configurations")
    
    def on_structure_loaded(self, structure_data):
        """Handle crystal structure loading completion."""
        print(f"✓ Crystal structure loaded: {structure_data.get('formula', 'Unknown')} with {structure_data.get('num_atoms', 0)} atoms")
        
        # Store structure data for use in other tabs
        self.current_crystal_structure = structure_data
        
        # Update crystal system combo if available
        if hasattr(self, 'crystal_system_combo') and 'crystal_system' in structure_data:
            crystal_system = structure_data['crystal_system']
            # Find matching item in combo box
            for i in range(self.crystal_system_combo.count()):
                if self.crystal_system_combo.itemText(i).lower() == crystal_system.lower():
                    self.crystal_system_combo.setCurrentIndex(i)
                    break
    
    def on_bonds_calculated(self, bond_data):
        """Handle bond calculation completion."""
        bond_count = bond_data.get('count', 0)
        print(f"✓ Calculated {bond_count} bonds in crystal structure")
        
        # Store bond data
        self.current_crystal_bonds = bond_data.get('bonds', [])
    
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
        title_label = QLabel("Tensor Analysis")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title_label)
        
        # Tensor calculation group
        tensor_group = QGroupBox("Tensor Calculation")
        tensor_layout = QVBoxLayout(tensor_group)
        
        calc_tensor_btn = QPushButton("Calculate Raman Tensors")
        calc_tensor_btn.clicked.connect(self.calculate_raman_tensors)
        tensor_layout.addWidget(calc_tensor_btn)
        
        export_tensor_btn = QPushButton("Export Tensor Data")
        tensor_layout.addWidget(export_tensor_btn)
        
        side_layout.addWidget(tensor_group)
        
        # Add stretch
        side_layout.addStretch()
        
        # Content area - placeholder
        content_layout = QVBoxLayout(content_area)
        
        tensor_info_label = QLabel("Tensor analysis and visualization will be implemented here")
        tensor_info_label.setAlignment(Qt.AlignCenter)
        tensor_info_label.setStyleSheet("color: gray; font-style: italic;")
        content_layout.addWidget(tensor_info_label)
    
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
        if hasattr(self, 'reference_combo') and self.mineral_list:
            self.reference_combo.clear()
            self.reference_combo.addItems(self.mineral_list)
    
    def on_tab_changed(self, index):
        """Handle tab change events."""
        current_tab = self.tab_widget.tabText(index)
        
        # Update plots when switching tabs
        if current_tab == "Spectrum Analysis" and self.current_spectrum is not None:
            self.update_spectrum_plot()
        elif current_tab == "Peak Fitting" and self.current_spectrum is not None:
            self.update_peak_fitting_plot()
            if self.selected_reference_mineral and hasattr(self, 'reference_combo'):
                self.reference_combo.setCurrentText(self.selected_reference_mineral)
    
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
        if self.selected_reference_mineral:
            # Update Peak Fitting tab reference mineral
            if hasattr(self, 'reference_combo'):
                self.reference_combo.setCurrentText(self.selected_reference_mineral)
                
            # Update status label
            if hasattr(self, 'reference_status_label'):
                self.reference_status_label.setText("Auto-selected from Spectrum Analysis tab")
                self.reference_status_label.setStyleSheet("color: green; font-style: italic;")
    
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
            self.peak_selection_btn.setText("Exit Peak Selection Mode")
            self.peak_selection_btn.setStyleSheet("background-color: #ffcccc;")
        else:
            self.peak_selection_btn.setText("Toggle Peak Selection Mode")
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
    
    def on_peak_click(self, event):
        """Handle mouse clicks for peak selection."""
        if not self.peak_selection_mode or not event.inaxes:
            return
        
        if self.current_spectrum is None:
            return
        
        # Get click position
        x_click = event.xdata
        
        if x_click is None:
            return
        
        # Add to selected peaks
        self.selected_peaks.append(x_click)
        
        # Update plot
        self.update_peak_fitting_plot()
    
    def fit_peaks(self):
        """Fit selected peaks with chosen shape."""
        if not self.selected_peaks or self.current_spectrum is None:
            QMessageBox.warning(self, "Warning", "Please select peaks first.")
            return
        
        try:
            shape = self.peak_shape_combo.currentText()
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            
            self.fitted_peaks.clear()
            
            for peak_pos in self.selected_peaks:
                # Find peak index
                peak_idx = np.argmin(np.abs(wavenumbers - peak_pos))
                
                # Define fitting range around peak
                range_width = 50  # wavenumbers
                mask = np.abs(wavenumbers - peak_pos) <= range_width
                
                if np.sum(mask) < 5:  # Need at least 5 points
                    continue
                
                x_fit = wavenumbers[mask]
                y_fit = intensities[mask]
                
                # Initial guess
                amplitude = np.max(y_fit)
                center = peak_pos
                width = 10
                
                try:
                    if shape == "Lorentzian":
                        popt, _ = curve_fit(self.lorentzian, x_fit, y_fit, 
                                          p0=[amplitude, center, width])
                    elif shape == "Gaussian":
                        popt, _ = curve_fit(self.gaussian, x_fit, y_fit, 
                                          p0=[amplitude, center, width])
                    elif shape == "Voigt":
                        popt, _ = curve_fit(self.voigt, x_fit, y_fit, 
                                          p0=[amplitude, center, width, width])
                    
                    # Store fitted peak
                    fitted_peak = {
                        'position': popt[1],  # center
                        'amplitude': popt[0],
                        'width': popt[2],
                        'shape': shape,
                        'parameters': popt
                    }
                    self.fitted_peaks.append(fitted_peak)
                    
                except Exception as fit_error:
                    print(f"Error fitting peak at {peak_pos}: {fit_error}")
                    continue
            
            # Update plot
            self.update_peak_fitting_plot()
            
            QMessageBox.information(self, "Success", 
                                  f"Fitted {len(self.fitted_peaks)} peaks successfully.")
            
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
            
            ax.plot(wavenumbers, intensities, 'b-', linewidth=1, alpha=0.7, label='Experimental Spectrum')
            has_data = True
            
            # Plot selected peaks
            for peak_pos in self.selected_peaks:
                peak_idx = np.argmin(np.abs(wavenumbers - peak_pos))
                ax.axvline(x=peak_pos, color='red', linestyle='--', alpha=0.7)
                ax.plot(peak_pos, intensities[peak_idx], 'ro', markersize=8)
            
            # Plot fitted peaks
            if self.fitted_peaks:
                x_fit = np.linspace(np.min(wavenumbers), np.max(wavenumbers), 1000)
                
                for peak in self.fitted_peaks:
                    if peak['shape'] == "Lorentzian":
                        y_fit = self.lorentzian(x_fit, *peak['parameters'])
                    elif peak['shape'] == "Gaussian":
                        y_fit = self.gaussian(x_fit, *peak['parameters'])
                    elif peak['shape'] == "Voigt":
                        y_fit = self.voigt(x_fit, *peak['parameters'])
                    
                    ax.plot(x_fit, y_fit, 'g-', linewidth=2, alpha=0.8)
                    ax.axvline(x=peak['position'], color='green', linestyle='-', alpha=0.5)
            
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
            legend_elements.append(Line2D([0], [0], color='blue', alpha=0.7, label='Experimental Spectrum'))
            
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
            structures = parser.get_structures()
            
            if not structures:
                QMessageBox.warning(self, "Warning", "No structures found in CIF file.")
                return
            
            # Take the first structure
            structure = structures[0]
            
            # Extract crystal information
            crystal_info = {
                'formula': structure.formula,
                'space_group': structure.get_space_group_info()[1],
                'crystal_system': structure.crystal_system.name,
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
        tol = 0.01
        
        # Check for cubic
        if abs(a - b) < tol and abs(b - c) < tol and abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol:
            return 'Cubic'
        # Check for hexagonal
        elif abs(a - b) < tol and abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 120) < tol:
            return 'Hexagonal'
        # Check for tetragonal
        elif abs(a - b) < tol and abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol:
            return 'Tetragonal'
        # Check for orthorhombic
        elif abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol:
            return 'Orthorhombic'
        # Check for monoclinic
        elif abs(alpha - 90) < tol and abs(gamma - 90) < tol:
            return 'Monoclinic'
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
            # Get crystal system for tensor symmetry
            crystal_system = self.crystal_system_combo.currentText()
            
            # Calculate Raman tensors for each fitted peak
            raman_tensors = {}
            
            for peak in self.fitted_peaks:
                freq = peak['position']
                intensity = peak['amplitude']
                width = peak['width']
                
                # Generate appropriate tensor based on crystal system and frequency
                tensor = self.generate_raman_tensor(crystal_system, freq, intensity)
                
                # Calculate tensor properties
                tensor_props = self.analyze_tensor_properties(tensor)
                
                raman_tensors[freq] = {
                    'tensor': tensor,
                    'properties': tensor_props,
                    'intensity': intensity,
                    'width': width,
                    'crystal_system': crystal_system
                }
            
            # Store results
            self.calculated_raman_tensors = raman_tensors
            
            # Show results
            self.show_tensor_results()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error calculating Raman tensors: {str(e)}")
    
    def generate_raman_tensor(self, crystal_system, frequency, intensity):
        """Generate a Raman tensor based on crystal system and vibrational mode."""
        # Normalize intensity for tensor scaling
        scale = intensity / 1000.0 if intensity > 0 else 0.001
        
        if crystal_system == "Cubic":
            # Cubic system: only A1g modes (isotropic tensor)
            tensor = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, scale]
            ])
        
        elif crystal_system == "Hexagonal":
            # Hexagonal system: A1g and E2g modes
            if frequency < 500:  # Assume E2g modes at low frequency
                tensor = np.array([
                    [scale, 0, 0],
                    [0, -scale, 0],
                    [0, 0, 0]
                ])
            else:  # A1g modes
                tensor = np.array([
                    [scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, scale * 1.2]  # c-axis enhancement
                ])
        
        elif crystal_system == "Tetragonal":
            # Tetragonal system: A1g, B1g, B2g, Eg modes
            if frequency > 800:  # High frequency A1g
                tensor = np.array([
                    [scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, scale * 1.5]
                ])
            else:  # Lower frequency modes
                tensor = np.array([
                    [scale * 0.8, 0, 0],
                    [0, scale * 0.8, 0],
                    [0, 0, scale]
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