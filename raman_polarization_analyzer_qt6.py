#!/usr/bin/env python3
"""
Raman Polarization Analyzer - Qt6 Version
Qt6 conversion of the comprehensive Raman polarization analysis tool.
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
    QHeaderView, QInputDialog, QDialog, QDialogButtonBox, QFormLayout
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QStandardPaths
from PySide6.QtGui import QFont, QPixmap, QIcon, QAction

# Matplotlib Qt6 backend
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend which works with PySide6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
        
        # Polarization analysis variables
        self.polarization_data = {}
        self.raman_tensors = {}
        self.depolarization_ratios = {}
        self.angular_data = {}
        self.current_polarization_config = None
        
        # Cross-tab integration variables
        self.selected_reference_mineral = None
        
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
        self.search_entry.textChanged.connect(self.on_search_change)
        import_layout.addWidget(self.search_entry)
        
        self.search_listbox = QListWidget()
        self.search_listbox.setMaximumHeight(120)
        self.search_listbox.itemDoubleClicked.connect(self.on_mineral_select)
        import_layout.addWidget(self.search_listbox)
        
        import_btn = QPushButton("Import Selected")
        import_btn.clicked.connect(self.import_selected_mineral)
        import_layout.addWidget(import_btn)
        
        side_layout.addWidget(import_group)
        
        # Analysis options group
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Add analysis options here as needed
        normalize_btn = QPushButton("Normalize Spectrum")
        analyze_btn = QPushButton("Auto-Analyze")
        
        analysis_layout.addWidget(normalize_btn)
        analysis_layout.addWidget(analyze_btn)
        
        side_layout.addWidget(analysis_group)
        
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
        """Setup the Polarization tab."""
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
    
    def setup_crystal_structure_tab(self, side_panel, content_area):
        """Setup the Crystal Structure tab."""
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
    
    def load_mineral_database(self):
        """Load the mineral database from file."""
        try:
            # Check for database file in current directory
            db_paths = [
                'mineral_database.pkl',
                'mineral_database.py',
                os.path.join(os.path.dirname(__file__), 'mineral_database.pkl'),
                os.path.join(os.path.dirname(__file__), 'mineral_database.py')
            ]
            
            for db_path in db_paths:
                if os.path.exists(db_path):
                    if db_path.endswith('.pkl'):
                        with open(db_path, 'rb') as f:
                            self.mineral_database = pickle.load(f)
                    elif db_path.endswith('.py'):
                        # Try to import the database module
                        try:
                            import importlib.util
                            spec = importlib.util.spec_from_file_location("mineral_database", db_path)
                            mineral_db_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mineral_db_module)
                            if hasattr(mineral_db_module, 'get_mineral_database'):
                                self.mineral_database = mineral_db_module.get_mineral_database()
                        except Exception as e:
                            print(f"Error importing mineral database module: {e}")
                            continue
                    
                    if self.mineral_database:
                        self.mineral_list = list(self.mineral_database.keys())
                        print(f"✓ Loaded mineral database with {len(self.mineral_list)} minerals")
                        return
            
            # If no database found, create minimal one
            self.create_minimal_database()
            
        except Exception as e:
            print(f"Error loading mineral database: {e}")
            self.create_minimal_database()
    
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
        """Handle mineral search text changes."""
        self.search_listbox.clear()
        
        if not text or not self.mineral_list:
            return
        
        # Filter minerals based on search text
        search_text = text.lower()
        matching_minerals = [
            mineral for mineral in self.mineral_list 
            if search_text in mineral.lower()
        ]
        
        # Add to listbox
        for mineral in matching_minerals[:20]:  # Limit to 20 results
            self.search_listbox.addItem(mineral)
    
    def on_mineral_select(self, item):
        """Handle mineral selection from search results."""
        if item:
            selected_mineral = item.text()
            self.import_selected_mineral(selected_mineral)
    
    def import_selected_mineral(self, mineral_name=None):
        """Import selected mineral spectrum from database."""
        if not mineral_name:
            current_item = self.search_listbox.currentItem()
            if not current_item:
                QMessageBox.warning(self, "Warning", "Please select a mineral from the list.")
                return
            mineral_name = current_item.text()
        
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
            
            if 'raman_modes' not in mineral_data:
                return None, None
            
            # Create wavenumber range
            wavenumbers = np.linspace(100, 1200, 2200)
            intensities = np.zeros_like(wavenumbers)
            
            # Add peaks for each Raman mode
            for mode in mineral_data['raman_modes']:
                frequency = mode['frequency']
                intensity_label = mode.get('intensity', 'medium')
                
                # Convert intensity label to numerical value
                intensity_map = {
                    'very_weak': 0.1,
                    'weak': 0.3,
                    'medium': 0.6,
                    'strong': 0.8,
                    'very_strong': 1.0
                }
                intensity = intensity_map.get(intensity_label, 0.5)
                
                # Add Lorentzian peak
                width = 10  # Default width
                peak_intensities = intensity / (1 + ((wavenumbers - frequency) / width) ** 2)
                intensities += peak_intensities
            
            # Add small amount of noise
            noise = np.random.normal(0, 0.02, len(intensities))
            intensities += noise
            
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
        """Clear all selected peaks."""
        self.selected_peaks.clear()
        self.fitted_peaks.clear()
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
            intensities = self.imported_spectrum['intensities']
            
            # Normalize if current spectrum exists
            if self.current_spectrum is not None:
                current_max = np.max(self.current_spectrum['intensities'])
                imported_max = np.max(intensities)
                if imported_max > 0:
                    intensities = intensities * (current_max / imported_max)
            
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
        """Update the peak fitting plot."""
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
            
            ax.plot(wavenumbers, intensities, 'b-', linewidth=1, alpha=0.7, label='Spectrum')
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
        
        # Configure plot
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Peak Fitting Analysis")
        ax.grid(True, alpha=0.3)
        
        if has_data:
            ax.legend()
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
        
        if file_path:
            QMessageBox.information(self, "Info", "CIF loading functionality will be implemented")
    
    def calculate_polarization(self):
        """Calculate polarization analysis."""
        QMessageBox.information(self, "Info", "Polarization calculation will be implemented")
    
    def calculate_raman_tensors(self):
        """Calculate Raman tensors."""
        QMessageBox.information(self, "Info", "Raman tensor calculation will be implemented")
    
    def run_orientation_optimization(self):
        """Run orientation optimization."""
        QMessageBox.information(self, "Info", "Orientation optimization will be implemented")


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