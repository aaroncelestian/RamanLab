#!/usr/bin/env python3
"""
Crystal Structure Visualization Widget
Advanced 3D crystal structure visualization with CIF import and interactive manipulation.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Qt6 imports
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, 
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
    QFileDialog, QMessageBox, QTextEdit, QScrollArea, QTableWidget, 
    QTableWidgetItem, QHeaderView, QSplitter, QFrame, QProgressBar,
    QTabWidget, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QColor

# Matplotlib for 3D visualization
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Try to import pymatgen for professional CIF parsing
try:
    from pymatgen.io.cif import CifParser
    from pymatgen.core import Structure, Element
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.vis.structure_vtk import StructureVis
    PYMATGEN_AVAILABLE = True
    print("✓ pymatgen available for crystal structure visualization")
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("⚠ pymatgen not available - install with: pip install pymatgen")


class CrystalStructureWidget(QWidget):
    """Advanced crystal structure visualization widget with CIF import and 3D interaction."""
    
    # Signals
    structure_loaded = Signal(dict)  # Emitted when structure is loaded
    bond_calculated = Signal(dict)   # Emitted when bonds are calculated
    
    def __init__(self, parent=None):
        """Initialize the Crystal Structure widget."""
        super().__init__(parent)
        
        # Initialize variables
        self.current_structure = None
        self.pymatgen_structure = None
        self.bonds = []
        self.supercell_size = [1, 1, 1]
        self.visualization_settings = {
            'show_bonds': True,
            'show_unit_cell': True,
            'show_axes': True,
            'label_atoms': False,
            'label_bonds': False,
            'atom_scale': 1.0,
            'bond_thickness': 1.0,
            'transparency': 0.8,
            'color_scheme': 'element',
            'render_quality': 'High',
            'preserve_geometry': True
        }
        
        # 3D view settings
        self.view_elevation = 20
        self.view_azimuth = 45
        self.zoom_level = 1.0
        
        # Bond control variables
        self.element_bond_controls = {}
        
        # Initialize UI
        self.init_ui()
        
        # Setup periodic update for smooth rotation
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.update_rotation)
        self.auto_rotate = False
    
    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for side panel and 3D view
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Side panel with tabs
        side_panel = self.create_side_panel_with_tabs()
        splitter.addWidget(side_panel)
        
        # 3D visualization area
        viz_area = self.create_visualization_area()
        splitter.addWidget(viz_area)
        
        # Set splitter proportions
        splitter.setSizes([300, 700])
    
    def create_side_panel_with_tabs(self):
        """Create the side control panel with tabbed interface."""
        side_panel = QFrame()
        side_panel.setFrameStyle(QFrame.StyledPanel)
        side_panel.setFixedWidth(320)
        
        layout = QVBoxLayout(side_panel)
        
        # Title
        title_label = QLabel("Crystal Structure Analysis")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create tab widget for controls
        self.control_tabs = QTabWidget()
        layout.addWidget(self.control_tabs)
        
        # Create individual tabs
        self.create_main_controls_tab()
        self.create_bond_controls_tab()
        self.create_wyckoff_tab()
        self.create_visualization_tab()
        self.create_advanced_tab()
        
        return side_panel
    
    def create_main_controls_tab(self):
        """Create main controls tab."""
        main_tab = QWidget()
        layout = QVBoxLayout(main_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # File operations group
        file_group = self.create_file_operations_group()
        layout.addWidget(file_group)
        
        # Structure info group
        info_group = self.create_structure_info_group()
        layout.addWidget(info_group)
        
        # Supercell controls
        supercell_group = self.create_supercell_controls_group()
        layout.addWidget(supercell_group)
        
        layout.addStretch()
        
        self.control_tabs.addTab(main_tab, "📁 Main")
    
    def create_bond_controls_tab(self):
        """Create advanced bond controls tab."""
        bond_tab = QWidget()
        layout = QVBoxLayout(bond_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Bond calculation group
        calc_group = QGroupBox("Bond Calculation")
        calc_layout = QVBoxLayout(calc_group)
        
        # Calculate bonds button
        calc_bonds_btn = QPushButton("🔗 Calculate Bonds")
        calc_bonds_btn.clicked.connect(self.calculate_bonds)
        calc_bonds_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        calc_layout.addWidget(calc_bonds_btn)
        
        # Clear bonds button
        clear_bonds_btn = QPushButton("Clear All Bonds")
        clear_bonds_btn.clicked.connect(self.clear_bonds)
        clear_bonds_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        calc_layout.addWidget(clear_bonds_btn)
        
        layout.addWidget(calc_group)
        
        # Global bond settings
        global_group = QGroupBox("Global Bond Settings")
        global_layout = QVBoxLayout(global_group)
        
        # Global cutoff distance
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("Global Max Distance (Å):"))
        self.global_bond_cutoff_spin = QDoubleSpinBox()
        self.global_bond_cutoff_spin.setRange(0.5, 8.0)
        self.global_bond_cutoff_spin.setValue(3.0)
        self.global_bond_cutoff_spin.setSingleStep(0.1)
        self.global_bond_cutoff_spin.valueChanged.connect(self.update_global_bond_cutoff)
        cutoff_layout.addWidget(self.global_bond_cutoff_spin)
        global_layout.addLayout(cutoff_layout)
        
        # Global minimum distance
        min_cutoff_layout = QHBoxLayout()
        min_cutoff_layout.addWidget(QLabel("Global Min Distance (Å):"))
        self.global_min_cutoff_spin = QDoubleSpinBox()
        self.global_min_cutoff_spin.setRange(0.1, 3.0)
        self.global_min_cutoff_spin.setValue(0.5)
        self.global_min_cutoff_spin.setSingleStep(0.1)
        self.global_min_cutoff_spin.valueChanged.connect(self.update_global_bond_cutoff)
        min_cutoff_layout.addWidget(self.global_min_cutoff_spin)
        global_layout.addLayout(min_cutoff_layout)
        
        layout.addWidget(global_group)
        
        # Element-specific bond controls
        element_group = QGroupBox("Element-Specific Bond Controls")
        element_layout = QVBoxLayout(element_group)
        
        # Scroll area for element pairs
        scroll_area = QScrollArea()
        scroll_area.setMaximumHeight(200)
        self.bond_controls_widget = QWidget()
        self.bond_controls_layout = QVBoxLayout(self.bond_controls_widget)
        scroll_area.setWidget(self.bond_controls_widget)
        scroll_area.setWidgetResizable(True)
        element_layout.addWidget(scroll_area)
        
        # Auto-generate button
        auto_gen_btn = QPushButton("Auto-Generate Element Pairs")
        auto_gen_btn.clicked.connect(self.auto_generate_bond_controls)
        auto_gen_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        element_layout.addWidget(auto_gen_btn)
        
        layout.addWidget(element_group)
        
        # Bond statistics
        stats_group = QGroupBox("Bond Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.bond_stats_detailed = QLabel("No bonds calculated")
        self.bond_stats_detailed.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        self.bond_stats_detailed.setWordWrap(True)
        stats_layout.addWidget(self.bond_stats_detailed)
        
        layout.addWidget(stats_group)
        
        # Bond visualization debugging
        debug_group = QGroupBox("Visualization Debug")
        debug_layout = QVBoxLayout(debug_group)
        
        debug_bonds_btn = QPushButton("Analyze Drawn Bonds")
        debug_bonds_btn.clicked.connect(self.analyze_drawn_bonds)
        debug_bonds_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        debug_layout.addWidget(debug_bonds_btn)
        
        self.bond_debug_info = QLabel("No analysis available")
        self.bond_debug_info.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        self.bond_debug_info.setWordWrap(True)
        debug_layout.addWidget(self.bond_debug_info)
        
        layout.addWidget(debug_group)
        
        # CrystalNN debugging
        crystalnn_debug_group = QGroupBox("CrystalNN Debug")
        crystalnn_debug_layout = QVBoxLayout(crystalnn_debug_group)
        
        self.debug_crystalnn_cb = QCheckBox("Enable CrystalNN Debug Output")
        self.debug_crystalnn_cb.setChecked(False)
        self.debug_crystalnn_cb.setVisible(False)  # Hide debug option
        crystalnn_debug_layout.addWidget(self.debug_crystalnn_cb)
        
        layout.addWidget(crystalnn_debug_group)
        
        layout.addStretch()
        
        self.control_tabs.addTab(bond_tab, "🔗 Bonds")
        
        # Initialize bond controls storage
        self.element_bond_controls = {}
    
    def create_wyckoff_tab(self):
        """Create Wyckoff positions tab."""
        wyckoff_tab = QWidget()
        layout = QVBoxLayout(wyckoff_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Wyckoff positions table
        wyckoff_group = QGroupBox("Wyckoff Positions")
        wyckoff_layout = QVBoxLayout(wyckoff_group)
        
        # Table widget for Wyckoff positions
        self.wyckoff_table = QTableWidget()
        self.wyckoff_table.setColumnCount(5)
        self.wyckoff_table.setHorizontalHeaderLabels([
            "Element", "Wyckoff", "Multiplicity", "Site Symmetry", "Coordinates"
        ])
        
        # Set table properties
        header = self.wyckoff_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        self.wyckoff_table.setAlternatingRowColors(True)
        self.wyckoff_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.wyckoff_table.setMaximumHeight(200)
        
        wyckoff_layout.addWidget(self.wyckoff_table)
        
        # Refresh button
        refresh_wyckoff_btn = QPushButton("Refresh Wyckoff Analysis")
        refresh_wyckoff_btn.clicked.connect(self.refresh_wyckoff_analysis)
        refresh_wyckoff_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        wyckoff_layout.addWidget(refresh_wyckoff_btn)
        
        layout.addWidget(wyckoff_group)
        
        # Symmetry operations summary
        symm_group = QGroupBox("Symmetry Operations")
        symm_layout = QVBoxLayout(symm_group)
        
        self.symmetry_info = QLabel("No structure loaded")
        self.symmetry_info.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        self.symmetry_info.setWordWrap(True)
        symm_layout.addWidget(self.symmetry_info)
        
        layout.addWidget(symm_group)
        
        # Equivalent sites controls
        equiv_group = QGroupBox("Symmetrically Equivalent Sites")
        equiv_layout = QVBoxLayout(equiv_group)
        
        # Show equivalent sites checkbox
        self.show_equiv_sites_cb = QCheckBox("Show All Equivalent Sites")
        self.show_equiv_sites_cb.toggled.connect(self.toggle_equivalent_sites_display)
        equiv_layout.addWidget(self.show_equiv_sites_cb)
        
        # Info about equivalent sites
        self.equiv_sites_info = QLabel("No equivalent sites generated")
        self.equiv_sites_info.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        self.equiv_sites_info.setWordWrap(True)
        equiv_layout.addWidget(self.equiv_sites_info)
        
        layout.addWidget(equiv_group)
        
        layout.addStretch()
        
        self.control_tabs.addTab(wyckoff_tab, "🔬 Wyckoff")
    
    def create_visualization_tab(self):
        """Create visualization controls tab."""
        viz_tab = QWidget()
        layout = QVBoxLayout(viz_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Visualization controls group
        viz_group = self.create_visualization_controls_group()
        layout.addWidget(viz_group)
        
        # View controls group  
        view_group = self.create_view_controls_group()
        layout.addWidget(view_group)
        
        layout.addStretch()
        
        self.control_tabs.addTab(viz_tab, "👁️ View")
    
    def create_advanced_tab(self):
        """Create advanced controls tab."""
        advanced_tab = QWidget()
        layout = QVBoxLayout(advanced_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Export controls
        export_group = QGroupBox("Export & Analysis")
        export_layout = QVBoxLayout(export_group)
        
        export_btn = QPushButton("Export Structure Data")
        export_btn.clicked.connect(self.export_structure)
        export_layout.addWidget(export_btn)
        
        save_view_btn = QPushButton("Save View as Image")
        save_view_btn.clicked.connect(self.save_view_image)
        export_layout.addWidget(save_view_btn)
        
        # Analysis button
        analyze_btn = QPushButton("Detailed Structure Analysis")
        analyze_btn.clicked.connect(self.show_detailed_analysis)
        analyze_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        export_layout.addWidget(analyze_btn)
        
        layout.addWidget(export_group)
        
        # Performance settings
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QVBoxLayout(perf_group)
        
        # Bond calculation method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Bond Method:"))
        self.bond_method_combo = QComboBox()
        self.bond_method_combo.addItems(["CrystalNN", "Distance Only", "Voronoi"])
        self.bond_method_combo.currentTextChanged.connect(self.update_bond_method)
        method_layout.addWidget(self.bond_method_combo)
        perf_layout.addLayout(method_layout)
        
        # Rendering quality
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Render Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["High", "Low"])
        self.quality_combo.setCurrentText("High")
        self.quality_combo.currentTextChanged.connect(self.update_render_quality)
        quality_layout.addWidget(self.quality_combo)
        perf_layout.addLayout(quality_layout)
        
        layout.addWidget(perf_group)
        
        layout.addStretch()
        
        self.control_tabs.addTab(advanced_tab, "⚙️ Advanced")
    
    def create_file_operations_group(self):
        """Create file operations control group."""
        group = QGroupBox("File Operations")
        layout = QVBoxLayout(group)
        
        # Load CIF button
        load_btn = QPushButton("Load CIF File")
        load_btn.clicked.connect(self.load_cif_file)
        load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        layout.addWidget(load_btn)
        
        # Export structure button
        export_btn = QPushButton("Export Structure")
        export_btn.clicked.connect(self.export_structure)
        layout.addWidget(export_btn)
        
        # Save view button
        save_view_btn = QPushButton("Save View as Image")
        save_view_btn.clicked.connect(self.save_view_image)
        layout.addWidget(save_view_btn)
        
        return group
    
    def create_structure_info_group(self):
        """Create structure information display group."""
        group = QGroupBox("Structure Information")
        layout = QVBoxLayout(group)
        
        # Structure info text area
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setPlainText("No structure loaded")
        layout.addWidget(self.info_text)
        
        # Quick stats
        self.stats_label = QLabel("Statistics: -")
        self.stats_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.stats_label)
        
        return group
    
    def create_visualization_controls_group(self):
        """Create visualization controls group."""
        group = QGroupBox("Visualization Settings")
        layout = QVBoxLayout(group)
        
        # Show/hide options
        self.show_bonds_cb = QCheckBox("Show Bonds")
        self.show_bonds_cb.setChecked(True)
        self.show_bonds_cb.toggled.connect(self.update_visualization_settings)
        layout.addWidget(self.show_bonds_cb)
        
        self.show_unit_cell_cb = QCheckBox("Show Unit Cell")
        self.show_unit_cell_cb.setChecked(True)
        self.show_unit_cell_cb.toggled.connect(self.update_visualization_settings)
        layout.addWidget(self.show_unit_cell_cb)
        
        self.show_axes_cb = QCheckBox("Show Crystal Axes")
        self.show_axes_cb.setChecked(True)
        self.show_axes_cb.toggled.connect(self.update_visualization_settings)
        layout.addWidget(self.show_axes_cb)
        
        # Labeling options
        self.label_atoms_cb = QCheckBox("Label Atoms")
        self.label_atoms_cb.setChecked(False)
        self.label_atoms_cb.toggled.connect(self.update_visualization_settings)
        layout.addWidget(self.label_atoms_cb)
        
        self.label_bonds_cb = QCheckBox("Label Bonds")
        self.label_bonds_cb.setChecked(False)
        self.label_bonds_cb.toggled.connect(self.update_visualization_settings)
        layout.addWidget(self.label_bonds_cb)
        
        # Atom scale slider
        layout.addWidget(QLabel("Atom Size:"))
        self.atom_scale_slider = QSlider(Qt.Horizontal)
        self.atom_scale_slider.setRange(10, 200)
        self.atom_scale_slider.setValue(100)
        self.atom_scale_slider.valueChanged.connect(self.update_atom_scale)
        layout.addWidget(self.atom_scale_slider)
        
        # Bond thickness slider
        layout.addWidget(QLabel("Bond Thickness:"))
        self.bond_thickness_slider = QSlider(Qt.Horizontal)
        self.bond_thickness_slider.setRange(10, 300)
        self.bond_thickness_slider.setValue(100)
        self.bond_thickness_slider.valueChanged.connect(self.update_bond_thickness)
        layout.addWidget(self.bond_thickness_slider)
        
        # Color scheme
        layout.addWidget(QLabel("Color Scheme:"))
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Element Colors", "Structure", "Coordination"])
        self.color_scheme_combo.currentTextChanged.connect(self.update_color_scheme)
        layout.addWidget(self.color_scheme_combo)
        
        # Geometry preservation option
        self.preserve_geometry_cb = QCheckBox("Preserve Crystal Geometry")
        self.preserve_geometry_cb.setChecked(True)
        self.preserve_geometry_cb.setToolTip("Preserve true crystal geometry vs. equal aspect ratios")
        self.preserve_geometry_cb.toggled.connect(self.update_visualization_settings)
        layout.addWidget(self.preserve_geometry_cb)
        
        return group
    
    def create_view_controls_group(self):
        """Create view control group."""
        group = QGroupBox("View Controls")
        layout = QVBoxLayout(group)
        
        # View presets
        layout.addWidget(QLabel("View Presets:"))
        preset_layout = QHBoxLayout()
        
        preset_a_btn = QPushButton("a-axis")
        preset_a_btn.clicked.connect(lambda: self.set_view_preset('a'))
        preset_layout.addWidget(preset_a_btn)
        
        preset_b_btn = QPushButton("b-axis")
        preset_b_btn.clicked.connect(lambda: self.set_view_preset('b'))
        preset_layout.addWidget(preset_b_btn)
        
        preset_c_btn = QPushButton("c-axis")
        preset_c_btn.clicked.connect(lambda: self.set_view_preset('c'))
        preset_layout.addWidget(preset_c_btn)
        
        layout.addLayout(preset_layout)
        
        # Auto rotation
        self.auto_rotate_cb = QCheckBox("Auto Rotate")
        self.auto_rotate_cb.toggled.connect(self.toggle_auto_rotation)
        layout.addWidget(self.auto_rotate_cb)
        
        # Manual rotation controls
        layout.addWidget(QLabel("Manual Rotation:"))
        
        # Elevation control
        elev_layout = QHBoxLayout()
        elev_layout.addWidget(QLabel("Elevation:"))
        self.elevation_slider = QSlider(Qt.Horizontal)
        self.elevation_slider.setRange(-90, 90)
        self.elevation_slider.setValue(20)
        self.elevation_slider.valueChanged.connect(self.update_elevation)
        elev_layout.addWidget(self.elevation_slider)
        layout.addLayout(elev_layout)
        
        # Azimuth control
        azim_layout = QHBoxLayout()
        azim_layout.addWidget(QLabel("Azimuth:"))
        self.azimuth_slider = QSlider(Qt.Horizontal)
        self.azimuth_slider.setRange(0, 360)
        self.azimuth_slider.setValue(45)
        self.azimuth_slider.valueChanged.connect(self.update_azimuth)
        azim_layout.addWidget(self.azimuth_slider)
        layout.addLayout(azim_layout)
        
        # Reset view button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        layout.addWidget(reset_btn)
        
        return group
    
    def create_supercell_controls_group(self):
        """Create supercell controls group."""
        group = QGroupBox("Supercell Generation")
        layout = QVBoxLayout(group)
        
        # Supercell controls
        layout.addWidget(QLabel("Supercell Size:"))
        supercell_layout = QGridLayout()
        
        for i, axis in enumerate(['a', 'b', 'c']):
            supercell_layout.addWidget(QLabel(f"{axis}:"), 0, i)
            spin = QSpinBox()
            spin.setRange(1, 5)
            spin.setValue(1)
            spin.valueChanged.connect(self.update_supercell)
            setattr(self, f'supercell_{axis}_spin', spin)
            supercell_layout.addWidget(spin, 1, i)
        
        layout.addLayout(supercell_layout)
        
        # Quick presets
        preset_layout = QHBoxLayout()
        preset_2x2_btn = QPushButton("2×2×2")
        preset_2x2_btn.clicked.connect(lambda: self.set_supercell_preset(2, 2, 2))
        preset_layout.addWidget(preset_2x2_btn)
        
        preset_3x3_btn = QPushButton("3×3×1")
        preset_3x3_btn.clicked.connect(lambda: self.set_supercell_preset(3, 3, 1))
        preset_layout.addWidget(preset_3x3_btn)
        
        layout.addLayout(preset_layout)
        
        return group
    
    def create_visualization_area(self):
        """Create the 3D visualization area."""
        viz_frame = QFrame()
        viz_frame.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(viz_frame)
        
        # Create matplotlib figure and 3D axis with larger size
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        
        # Adjust subplot parameters for better space utilization
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
        
        # Create 3D axis
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('Crystal a-axis')
        self.ax.set_ylabel('Crystal b-axis')
        self.ax.set_zlabel('Crystal c-axis')
        self.ax.set_title('Crystal Structure Visualization')
        
        layout.addWidget(self.canvas)
        
        # Initialize empty plot
        self.update_3d_plot()
        
        return viz_frame
    
    def load_cif_file(self):
        """Load crystal structure from CIF file."""
        if not PYMATGEN_AVAILABLE:
            QMessageBox.critical(self, "Error", 
                               "pymatgen is not available. Please install it with:\npip install pymatgen")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CIF File", "", "CIF files (*.cif);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Parse CIF file using pymatgen
            parser = CifParser(file_path)
            structures = parser.parse_structures(primitive=True)
            
            if not structures:
                QMessageBox.warning(self, "Warning", "No structures found in CIF file.")
                return
            
            # Take the first structure
            self.pymatgen_structure = structures[0]
            
            # Extract structure information
            self.extract_structure_info()
            
            # Update displays
            self.update_structure_info_display()
            self.update_wyckoff_table()
            self.update_symmetry_info()
            self.update_equivalent_sites_info()
            self.update_3d_plot()
            
            # Calculate bonds automatically
            self.calculate_bonds()
            
            # Emit signal
            self.structure_loaded.emit(self.current_structure)
            
            QMessageBox.information(self, "Success", 
                                  f"Successfully loaded crystal structure from {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading CIF file:\n{str(e)}")
    
    def extract_structure_info(self):
        """Extract structure information from pymatgen structure."""
        if not self.pymatgen_structure:
            return
        
        structure = self.pymatgen_structure
        
        # Get crystal system using SpacegroupAnalyzer
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure)
            crystal_system = sga.get_crystal_system()
            space_group = sga.get_space_group_symbol()
            space_group_number = sga.get_space_group_number()
            
            # Get symmetry operations
            symm_ops = sga.get_symmetry_operations()
            
            # Get Wyckoff positions
            wyckoff_symbols = sga.get_wyckoff_symbols()
            
        except Exception as e:
            print(f"Warning: Could not determine crystal system: {e}")
            # Fallback to lattice-based determination
            crystal_system = self.determine_crystal_system_from_lattice(structure.lattice)
            try:
                space_group = structure.get_space_group_info()[1]
                space_group_number = structure.get_space_group_info()[0]
            except:
                space_group = "Unknown"
                space_group_number = 1
            symm_ops = []
            wyckoff_symbols = ['a'] * len(structure)  # Default fallback
        
        # Basic structure information
        self.current_structure = {
            'formula': str(structure.composition.reduced_formula),
            'space_group': space_group,
            'space_group_number': space_group_number,
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
            'num_atoms': len(structure),
            'elements': list(set([site.specie.symbol for site in structure])),
            'symmetry_operations': symm_ops,
            'sites': [],
            'wyckoff_positions': [],
            'equivalent_sites': {}
        }
        
        # Extract atomic sites with Wyckoff positions and symmetry
        for i, site in enumerate(structure):
            # Get atomic radius safely
            try:
                atomic_radius = site.specie.atomic_radius
                if atomic_radius is None:
                    atomic_radius = 1.0
            except:
                atomic_radius = 1.0
            
            # Get Wyckoff symbol for this site
            wyckoff_symbol = wyckoff_symbols[i] if i < len(wyckoff_symbols) else 'a'
            
            site_info = {
                'index': i,
                'element': site.specie.symbol,
                'frac_coords': site.frac_coords.tolist(),
                'cart_coords': site.coords.tolist(),
                'atomic_number': site.specie.Z,
                'radius': atomic_radius,
                'wyckoff_symbol': wyckoff_symbol,
                'site_symmetry': self.get_site_symmetry(site, symm_ops),
                'multiplicity': self.get_wyckoff_multiplicity(wyckoff_symbol, space_group_number)
            }
            self.current_structure['sites'].append(site_info)
        
        # Generate symmetrically equivalent sites
        self.generate_equivalent_sites()
        
        # Group sites by Wyckoff positions
        self.group_wyckoff_positions()
    
    def get_site_symmetry(self, site, symm_ops):
        """Get point group symmetry for a site."""
        if not symm_ops:
            return "1"  # Default C1 symmetry
        
        try:
            # Find symmetry operations that leave the site invariant
            invariant_ops = []
            tolerance = 1e-3
            
            for op in symm_ops:
                # Apply symmetry operation to site
                transformed_coords = op.operate(site.frac_coords)
                
                # Check if transformed coordinates are equivalent to original
                # (considering periodic boundary conditions)
                diff = np.array(transformed_coords) - np.array(site.frac_coords)
                diff = diff - np.round(diff)  # Handle periodic boundaries
                
                if np.allclose(diff, 0, atol=tolerance):
                    invariant_ops.append(op)
            
            # Return the order of the site symmetry group
            return str(len(invariant_ops))
            
        except Exception as e:
            print(f"Warning: Could not determine site symmetry: {e}")
            return "1"
    
    def get_wyckoff_multiplicity(self, wyckoff_symbol, space_group_number):
        """Get multiplicity for a Wyckoff position."""
        # This is a simplified lookup - in practice, you'd use a full table
        # For now, we'll use some common multiplicities
        common_multiplicities = {
            'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 6, 'f': 8, 'g': 12, 'h': 16,
            'i': 24, 'j': 32, 'k': 48, 'l': 96
        }
        
        return common_multiplicities.get(wyckoff_symbol.lower(), 1)
    
    def generate_equivalent_sites(self):
        """Generate all symmetrically equivalent sites."""
        if not self.current_structure or not self.current_structure.get('symmetry_operations'):
            return
        
        symm_ops = self.current_structure['symmetry_operations']
        original_sites = self.current_structure['sites'].copy()
        equivalent_sites = {}
        
        for site_idx, site in enumerate(original_sites):
            equivalent_positions = []
            
            # Apply all symmetry operations to this site
            for op_idx, op in enumerate(symm_ops):
                try:
                    # Apply symmetry operation
                    transformed_frac = op.operate(np.array(site['frac_coords']))
                    
                    # Convert to Cartesian coordinates
                    if self.pymatgen_structure:
                        transformed_cart = self.pymatgen_structure.lattice.get_cartesian_coords(transformed_frac)
                    else:
                        transformed_cart = transformed_frac  # Fallback
                    
                    # Normalize to unit cell (handle periodic boundaries)
                    transformed_frac = transformed_frac - np.floor(transformed_frac)
                    
                    equiv_site = {
                        'frac_coords': transformed_frac.tolist(),
                        'cart_coords': transformed_cart.tolist(),
                        'symmetry_op_index': op_idx,
                        'parent_site_index': site_idx
                    }
                    equivalent_positions.append(equiv_site)
                    
                except Exception as e:
                    print(f"Warning: Could not apply symmetry operation {op_idx} to site {site_idx}: {e}")
                    continue
            
            equivalent_sites[site_idx] = equivalent_positions
        
        self.current_structure['equivalent_sites'] = equivalent_sites
        print(f"✓ Generated equivalent sites for {len(original_sites)} atoms using {len(symm_ops)} symmetry operations")
    
    def group_wyckoff_positions(self):
        """Group sites by their Wyckoff positions."""
        if not self.current_structure:
            return
        
        wyckoff_groups = {}
        
        for site in self.current_structure['sites']:
            wyckoff_symbol = site.get('wyckoff_symbol', 'a')
            element = site['element']
            
            # Create a key combining element and Wyckoff symbol
            key = f"{element}_{wyckoff_symbol}"
            
            if key not in wyckoff_groups:
                wyckoff_groups[key] = {
                    'element': element,
                    'wyckoff_symbol': wyckoff_symbol,
                    'multiplicity': site.get('multiplicity', 1),
                    'site_symmetry': site.get('site_symmetry', '1'),
                    'sites': [],
                    'representative_coords': site['frac_coords']
                }
            
            wyckoff_groups[key]['sites'].append(site['index'])
        
        self.current_structure['wyckoff_positions'] = wyckoff_groups
        print(f"✓ Identified {len(wyckoff_groups)} unique Wyckoff positions")
    
    def determine_crystal_system_from_lattice(self, lattice):
        """Determine crystal system from lattice parameters as fallback."""
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        
        # Tolerance for equality checks
        tol = 0.01
        
        # Check for cubic
        if (abs(a - b) < tol and abs(b - c) < tol and 
            abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
            return 'cubic'
        
        # Check for hexagonal
        elif (abs(a - b) < tol and abs(alpha - 90) < tol and 
              abs(beta - 90) < tol and abs(gamma - 120) < tol):
            return 'hexagonal'
        
        # Check for trigonal (rhombohedral)
        elif (abs(a - b) < tol and abs(b - c) < tol and 
              abs(alpha - beta) < tol and abs(beta - gamma) < tol and abs(alpha - 90) > tol):
            return 'trigonal'
        
        # Check for tetragonal
        elif (abs(a - b) < tol and abs(alpha - 90) < tol and 
              abs(beta - 90) < tol and abs(gamma - 90) < tol):
            return 'tetragonal'
        
        # Check for orthorhombic
        elif (abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
            return 'orthorhombic'
        
        # Check for monoclinic
        elif (abs(alpha - 90) < tol and abs(gamma - 90) < tol):
            return 'monoclinic'
        
        else:
            return 'triclinic'
    
    def update_structure_info_display(self):
        """Update the structure information display."""
        if not self.current_structure:
            self.info_text.setPlainText("No structure loaded")
            self.stats_label.setText("Statistics: -")
            return
        
        # Format structure information
        info_text = f"Formula: {self.current_structure['formula']}\n"
        info_text += f"Space Group: {self.current_structure['space_group']}"
        if 'space_group_number' in self.current_structure:
            info_text += f" (#{self.current_structure['space_group_number']})"
        info_text += f"\nCrystal System: {self.current_structure['crystal_system']}\n"
        info_text += f"Atoms: {self.current_structure['num_atoms']}\n"
        info_text += f"Elements: {', '.join(self.current_structure['elements'])}\n"
        
        # Wyckoff positions summary
        if 'wyckoff_positions' in self.current_structure and self.current_structure['wyckoff_positions']:
            wyckoff_count = len(self.current_structure['wyckoff_positions'])
            info_text += f"Wyckoff Positions: {wyckoff_count}\n"
            
            # Show first few Wyckoff positions
            wyckoff_summary = []
            for key, wyckoff_data in list(self.current_structure['wyckoff_positions'].items())[:3]:
                element = wyckoff_data['element']
                symbol = wyckoff_data['wyckoff_symbol']
                mult = wyckoff_data['multiplicity']
                wyckoff_summary.append(f"{element}({symbol}×{mult})")
            
            if len(self.current_structure['wyckoff_positions']) > 3:
                wyckoff_summary.append("...")
            
            info_text += f"Wyckoff: {', '.join(wyckoff_summary)}\n"
        
        info_text += "\n"
        
        # Lattice parameters
        params = self.current_structure['lattice_params']
        info_text += f"Lattice Parameters:\n"
        info_text += f"a = {params['a']:.4f} Å\n"
        info_text += f"b = {params['b']:.4f} Å\n"
        info_text += f"c = {params['c']:.4f} Å\n"
        info_text += f"α = {params['alpha']:.2f}°\n"
        info_text += f"β = {params['beta']:.2f}°\n"
        info_text += f"γ = {params['gamma']:.2f}°\n"
        info_text += f"V = {params['volume']:.2f} Å³"
        
        self.info_text.setPlainText(info_text)
        
        # Statistics
        stats_text = f"Atoms: {self.current_structure['num_atoms']}, "
        stats_text += f"Elements: {len(self.current_structure['elements'])}, "
        if 'wyckoff_positions' in self.current_structure:
            stats_text += f"Wyckoff: {len(self.current_structure['wyckoff_positions'])}, "
        stats_text += f"Bonds: {len(self.bonds)}"
        self.stats_label.setText(f"Statistics: {stats_text}")
    
    def calculate_bonds(self):
        """Calculate bonds between atoms considering symmetrically equivalent sites."""
        if not self.pymatgen_structure:
            return
        
        try:
            # Use enhanced bond calculation considering symmetry
            self.bonds = []
            bond_distances = []
            
            # Get bond distance constraints
            max_distance = getattr(self, 'global_bond_cutoff_spin', None)
            min_distance = getattr(self, 'global_min_cutoff_spin', None)
            
            if max_distance is None:
                max_distance = 3.0  # Default fallback
            else:
                max_distance = max_distance.value()
                
            if min_distance is None:
                min_distance = 0.5  # Default fallback
            else:
                min_distance = min_distance.value()
            
            # Calculate bonds using both CrystalNN and symmetry-enhanced distance calculation
            method = getattr(self, 'bond_method_combo', None)
            if method and hasattr(method, 'currentText'):
                bond_method = method.currentText()
            else:
                bond_method = "CrystalNN"
            
            if bond_method == "CrystalNN" and PYMATGEN_AVAILABLE:
                self.calculate_bonds_crystalnn(min_distance, max_distance, bond_distances)
            else:
                self.calculate_bonds_distance_only(min_distance, max_distance, bond_distances)
            
            # Update bond statistics
            self.update_bond_statistics(bond_distances)
            
            # Update visualization
            self.update_3d_plot()
            
            # Emit signal
            self.bond_calculated.emit({'bonds': self.bonds, 'count': len(self.bonds)})
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error calculating bonds:\n{str(e)}")
    
    def calculate_bonds_crystalnn(self, min_distance, max_distance, bond_distances):
        """Calculate bonds using CrystalNN method with debugging."""
        try:
            from pymatgen.analysis.local_env import CrystalNN
            nn = CrystalNN()
            
            # Check if debug is enabled
            debug_enabled = getattr(self, 'debug_crystalnn_cb', None)
            debug_mode = debug_enabled.isChecked() if debug_enabled else False
            
            if debug_mode:
                print("\n🔍 CrystalNN Debug Information:")
            
            # Calculate bonds for each site
            for i, site in enumerate(self.pymatgen_structure):
                try:
                    # Get nearest neighbors
                    neighbors = nn.get_nn_info(self.pymatgen_structure, i)
                    
                    element_i = self.current_structure['sites'][i]['element']
                    if debug_mode:
                        print(f"\nSite {i} ({element_i}):")
                    
                    for neighbor in neighbors:
                        j = neighbor['site_index']
                        
                        # Debug: Print what CrystalNN returns
                        if debug_mode:
                            print(f"  Neighbor data: {neighbor}")
                        
                        # CrystalNN 'weight' might not be distance - let's calculate actual distance
                        actual_distance = self.calculate_actual_distance_between_sites(i, j)
                        
                        # Check if 'weight' matches actual distance
                        crystalnn_weight = neighbor.get('weight', actual_distance)
                        
                        if debug_mode:
                            print(f"  -> Site {j}: CrystalNN weight={crystalnn_weight:.3f}, Actual distance={actual_distance:.3f}")
                        
                        element_j = self.current_structure['sites'][j]['element']
                        
                        # Use actual distance instead of CrystalNN weight
                        if self.validate_and_add_bond(i, j, actual_distance, min_distance, max_distance, bond_distances):
                            if debug_mode:
                                print(f"  ✓ Added bond {element_i}{i}-{element_j}{j}: {actual_distance:.3f}Å")
                
                except Exception as e:
                    # If CrystalNN fails for this site, continue
                    if debug_mode:
                        print(f"  ❌ CrystalNN failed for site {i}: {e}")
                    continue
            
            if debug_mode:
                print(f"\n🔗 CrystalNN found {len(self.bonds)} bonds total")
                    
        except ImportError:
            print("CrystalNN not available, falling back to distance-only method")
            self.calculate_bonds_distance_only(min_distance, max_distance, bond_distances)
    
    def calculate_bonds_distance_only(self, min_distance, max_distance, bond_distances):
        """Calculate bonds using distance-only method with symmetry consideration."""
        # Get all possible atomic positions including equivalent sites
        all_positions = self.get_all_atomic_positions_for_bonding()
        
        for i, pos1 in enumerate(all_positions):
            for j, pos2 in enumerate(all_positions):
                if i >= j:  # Avoid duplicates and self-bonds
                    continue
                
                # Skip if same parent atom (symmetrically equivalent)
                if pos1['parent_index'] == pos2['parent_index']:
                    continue
                
                # Calculate distance considering periodic boundary conditions
                distance = self.calculate_periodic_distance(pos1['frac_coords'], pos2['frac_coords'])
                
                if min_distance <= distance <= max_distance:
                    # Validate and add bond
                    self.validate_and_add_bond(
                        pos1['parent_index'], pos2['parent_index'], 
                        distance, min_distance, max_distance, bond_distances,
                        pos1_coords=pos1['frac_coords'], pos2_coords=pos2['frac_coords']
                    )
    
    def get_all_atomic_positions_for_bonding(self):
        """Get all atomic positions including symmetrically equivalent sites for bond calculation."""
        all_positions = []
        
        # Add original unit cell positions
        for site in self.current_structure['sites']:
            all_positions.append({
                'parent_index': site['index'],
                'element': site['element'],
                'frac_coords': np.array(site['frac_coords']),
                'wyckoff_symbol': site.get('wyckoff_symbol', 'a'),
                'is_equivalent': False
            })
        
        # Add symmetrically equivalent positions within a reasonable range
        if self.current_structure.get('equivalent_sites'):
            for parent_idx, equiv_sites in self.current_structure['equivalent_sites'].items():
                parent_element = self.current_structure['sites'][parent_idx]['element']
                wyckoff_symbol = self.current_structure['sites'][parent_idx].get('wyckoff_symbol', 'a')
                
                for equiv_site in equiv_sites:
                    # Skip if it's the identity operation
                    if equiv_site['symmetry_op_index'] == 0:
                        continue
                    
                    all_positions.append({
                        'parent_index': parent_idx,
                        'element': parent_element,
                        'frac_coords': np.array(equiv_site['frac_coords']),
                        'wyckoff_symbol': wyckoff_symbol,
                        'is_equivalent': True,
                        'symmetry_op_index': equiv_site['symmetry_op_index']
                    })
        
        return all_positions
    
    def calculate_periodic_distance(self, frac_coords1, frac_coords2):
        """Calculate distance between two points considering periodic boundary conditions."""
        if not self.pymatgen_structure:
            # Fallback to simple Euclidean distance
            return np.linalg.norm(np.array(frac_coords1) - np.array(frac_coords2))
        
        # Convert to Cartesian coordinates
        cart1 = self.pymatgen_structure.lattice.get_cartesian_coords(frac_coords1)
        cart2 = self.pymatgen_structure.lattice.get_cartesian_coords(frac_coords2)
        
        # Consider periodic images within a reasonable range
        min_distance = float('inf')
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    # Add lattice translation
                    translated_frac2 = frac_coords2 + np.array([dx, dy, dz])
                    translated_cart2 = self.pymatgen_structure.lattice.get_cartesian_coords(translated_frac2)
                    
                    # Calculate distance
                    distance = np.linalg.norm(cart1 - translated_cart2)
                    min_distance = min(min_distance, distance)
        
        return min_distance
    
    def validate_and_add_bond(self, i, j, distance, min_distance, max_distance, bond_distances, pos1_coords=None, pos2_coords=None):
        """Validate and add a bond if it meets all criteria."""
        # Get element types
        element1 = self.current_structure['sites'][i]['element']
        element2 = self.current_structure['sites'][j]['element']
        
        # Check element-specific distance constraints
        pair_key = f"{min(element1, element2)}-{max(element1, element2)}"
        
        # Get specific constraints for this element pair
        if hasattr(self, 'element_bond_controls') and pair_key in self.element_bond_controls:
            element_min = self.element_bond_controls[pair_key]['min_distance']
            element_max = self.element_bond_controls[pair_key]['max_distance']
            element_enabled = self.element_bond_controls[pair_key]['enabled']
            
            if not element_enabled:
                return False  # Skip this element pair if disabled
        else:
            # Use global constraints
            element_min = min_distance
            element_max = max_distance
        
        # Check distance constraints
        if element_min <= distance <= element_max:
            # Check if this bond already exists (avoid duplicates)
            for existing_bond in self.bonds:
                if ((existing_bond['atom1_idx'] == i and existing_bond['atom2_idx'] == j) or
                    (existing_bond['atom1_idx'] == j and existing_bond['atom2_idx'] == i)):
                    # Bond already exists, check if this one is shorter
                    if distance < existing_bond['distance']:
                        # Update with shorter distance
                        existing_bond['distance'] = distance
                        if pos1_coords is not None and pos2_coords is not None:
                            existing_bond['pos1_coords'] = pos1_coords.tolist()
                            existing_bond['pos2_coords'] = pos2_coords.tolist()
                    return False
            
            # Add new bond
            bond_info = {
                'atom1_idx': i,
                'atom2_idx': j,
                'distance': distance,
                'atom1_element': element1,
                'atom2_element': element2,
                'bond_type': pair_key,
                'constraints_used': f"min:{element_min:.2f}, max:{element_max:.2f}",
                'wyckoff_info': f"{self.current_structure['sites'][i].get('wyckoff_symbol', 'a')}-{self.current_structure['sites'][j].get('wyckoff_symbol', 'a')}"
            }
            
            # Add position information if available
            if pos1_coords is not None and pos2_coords is not None:
                bond_info['pos1_coords'] = pos1_coords.tolist()
                bond_info['pos2_coords'] = pos2_coords.tolist()
            
            self.bonds.append(bond_info)
            bond_distances.append(distance)
            return True
        
        return False
    
    def analyze_drawn_bonds(self):
        """Analyze and display information about bonds actually drawn in the visualization."""
        if not hasattr(self, 'drawn_bonds') or not self.drawn_bonds:
            self.bond_debug_info.setText("No drawn bonds to analyze. Update visualization first.")
            return
        
        # Analyze drawn bonds
        total_drawn = len(self.drawn_bonds)
        calculated_bonds = len(self.bonds)
        
        # Group by bond type and analyze distances
        drawn_by_type = {}
        distance_errors = []
        
        for drawn_bond in self.drawn_bonds:
            atom1_element = self.current_structure['sites'][drawn_bond['atom1_idx']]['element']
            atom2_element = self.current_structure['sites'][drawn_bond['atom2_idx']]['element']
            bond_type = f"{min(atom1_element, atom2_element)}-{max(atom1_element, atom2_element)}"
            
            if bond_type not in drawn_by_type:
                drawn_by_type[bond_type] = {
                    'count': 0,
                    'distances': [],
                    'errors': []
                }
            
            drawn_by_type[bond_type]['count'] += 1
            drawn_by_type[bond_type]['distances'].append(drawn_bond['drawn_distance'])
            
            # Calculate error between drawn and calculated distance
            error = abs(drawn_bond['drawn_distance'] - drawn_bond['original_distance'])
            drawn_by_type[bond_type]['errors'].append(error)
            distance_errors.append(error)
        
        # Create analysis text
        analysis_text = f"Drawn Bonds Analysis:\n"
        analysis_text += f"Total drawn: {total_drawn}\n"
        analysis_text += f"Calculated bonds: {calculated_bonds}\n"
        analysis_text += f"Ratio: {total_drawn/calculated_bonds:.1f}x\n\n"
        
        if distance_errors:
            max_error = max(distance_errors)
            avg_error = np.mean(distance_errors)
            analysis_text += f"Distance Errors:\n"
            analysis_text += f"Max: {max_error:.3f} Å\n"
            analysis_text += f"Avg: {avg_error:.3f} Å\n\n"
        
        # Show breakdown by bond type
        analysis_text += "By bond type:\n"
        for bond_type, data in drawn_by_type.items():
            avg_distance = np.mean(data['distances'])
            avg_error = np.mean(data['errors'])
            analysis_text += f"{bond_type}: {data['count']} bonds\n"
            analysis_text += f"  Avg distance: {avg_distance:.3f} Å\n"
            analysis_text += f"  Avg error: {avg_error:.3f} Å\n"
        
        self.bond_debug_info.setText(analysis_text)
        
        # Also show detailed analysis in console
        print("\n" + "="*50)
        print("BOND VISUALIZATION ANALYSIS")
        print("="*50)
        print(analysis_text)
        
        # Show problematic bonds (large errors)
        problem_threshold = 0.2  # Angstrom
        problem_bonds = [bond for bond in self.drawn_bonds 
                        if abs(bond['drawn_distance'] - bond['original_distance']) > problem_threshold]
        
        if problem_bonds:
            print(f"\nPROBLEMATIC BONDS (error > {problem_threshold} Å):")
            for bond in problem_bonds:
                atom1_element = self.current_structure['sites'][bond['atom1_idx']]['element']
                atom2_element = self.current_structure['sites'][bond['atom2_idx']]['element']
                print(f"{atom1_element}{bond['atom1_idx']}-{atom2_element}{bond['atom2_idx']}: "
                      f"drawn={bond['drawn_distance']:.3f} Å, "
                      f"calculated={bond['original_distance']:.3f} Å, "
                      f"error={abs(bond['drawn_distance'] - bond['original_distance']):.3f} Å")
                print(f"  Unit cell: {bond['unit_cell']}")
                print(f"  Positions: {bond['pos1']} -> {bond['pos2']}")
        
        print("="*50)
    
    def update_3d_plot(self):
        """Update the 3D structure visualization."""
        if not self.current_structure:
            # Clear plot and show placeholder
            self.ax.clear()
            
            # Turn off all grid elements and axes for empty plot too
            self.ax.grid(False)
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_alpha(0)
            self.ax.yaxis.pane.set_alpha(0) 
            self.ax.zaxis.pane.set_alpha(0)
            
            # Turn off ticks and axis lines for empty plot
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_zticks([])
            self.ax.set_xlabel('')
            self.ax.set_ylabel('')
            self.ax.set_zlabel('')
            self.ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            
            self.ax.text(0.5, 0.5, 0.5, 'Load a CIF file to view structure', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.6)
            self.canvas.draw()
            return
        
        # Clear previous plot
        self.ax.clear()
        
        # Get supercell dimensions
        na, nb, nc = self.supercell_size
        
        # Plot atoms in crystal coordinates
        self.plot_atoms_crystal_coords(na, nb, nc)
        
        # Plot bonds if enabled
        if self.visualization_settings['show_bonds'] and self.bonds:
            self.plot_bonds_crystal_coords(na, nb, nc)
        
        # Plot unit cell if enabled
        if self.visualization_settings['show_unit_cell']:
            self.plot_unit_cell()
        
        # Plot crystal axes if enabled
        if self.visualization_settings['show_axes']:
            self.plot_crystal_axes()
        
        # Set view
        self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)
        
        # Turn off all grid elements and axes
        self.ax.grid(False)
        
        # Turn off axis panes and grid
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Make panes transparent
        self.ax.xaxis.pane.set_alpha(0)
        self.ax.yaxis.pane.set_alpha(0) 
        self.ax.zaxis.pane.set_alpha(0)
        
        # Turn off grid lines
        self.ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
        self.ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
        self.ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
        
        # Turn off axis lines and ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        
        # Remove axis labels
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_zlabel('')
        
        # Hide axis lines
        self.ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        formula = self.current_structure['formula']
        space_group = self.current_structure['space_group']
        self.ax.set_title(f'{formula} ({space_group})')
        
        # Set equal aspect ratio for crystal coordinates
        self.set_equal_aspect_crystal()
        
        # Optimize view for current structure
        self.optimize_view_bounds()
        
        self.canvas.draw()
    
    def plot_atoms_crystal_coords(self, na, nb, nc):
        """Plot atoms using proper Cartesian coordinates from crystal lattice."""
        # Element colors (simplified)
        element_colors = {
            'H': '#FFFFFF', 'C': '#000000', 'N': '#0000FF', 'O': '#FF0000',
            'F': '#00FF00', 'P': '#FFA500', 'S': '#FFFF00', 'Cl': '#00FF00',
            'Ti': '#BFC2C7', 'Si': '#F0C814', 'Ca': '#3DFF00', 'Fe': '#E06633',
            'Mg': '#8AFF00', 'Al': '#BFA6A6', 'K': '#8F40D4', 'Na': '#AB5CF2'
        }
        
        # Atomic radii scaling
        scale = self.visualization_settings['atom_scale']
        
        # Get render quality setting
        render_quality = self.visualization_settings.get('render_quality', 'High')
        
        for site_idx, site in enumerate(self.current_structure['sites']):
            element = site['element']
            frac_coords = np.array(site['frac_coords'])
            
            # Get color for element
            color = element_colors.get(element, '#CCCCCC')
            
            # Base radius from element or default
            base_radius = site.get('radius', 1.0)
            if base_radius is None or base_radius <= 0:
                base_radius = 0.5  # Default radius
            
            radius = base_radius * scale * 0.3  # Scale for visualization
            
            # Plot atoms in supercell
            for i in range(na):
                for j in range(nb):
                    for k in range(nc):
                        # Fractional coordinates including supercell translation
                        frac_pos = frac_coords + np.array([i, j, k])
                        
                        # Convert to proper Cartesian coordinates using lattice matrix
                        if self.pymatgen_structure:
                            cart_pos = self.pymatgen_structure.lattice.get_cartesian_coords(frac_pos)
                        else:
                            cart_pos = frac_pos  # Fallback
                        
                        # Plot atom based on quality setting
                        if render_quality == 'High':
                            self.plot_atom_sphere(cart_pos, radius, color)
                        else:  # Low quality
                            self.plot_atom_circle(cart_pos, radius, color)
                        
                        # Add atom labels if enabled
                        if self.visualization_settings['label_atoms']:
                            self.add_atom_label(cart_pos, site_idx, element, i, j, k)
    
    def plot_atom_sphere(self, position, radius, color):
        """Plot a high-quality 3D sphere for an atom."""
        try:
            # Ensure position and radius are scalars
            pos_x = float(position[0])
            pos_y = float(position[1]) 
            pos_z = float(position[2])
            r = float(radius)
            
            # Create sphere geometry
            u = np.linspace(0, 2 * np.pi, 12)  # Reduced resolution for performance
            v = np.linspace(0, np.pi, 8)
            
            # Create meshgrid for sphere coordinates
            u_mesh, v_mesh = np.meshgrid(u, v)
            
            # Sphere coordinates
            x = r * np.cos(u_mesh) * np.sin(v_mesh) + pos_x
            y = r * np.sin(u_mesh) * np.sin(v_mesh) + pos_y
            z = r * np.cos(v_mesh) + pos_z
            
            # Plot the 3D sphere surface
            self.ax.plot_surface(x, y, z, color=color, alpha=0.8, linewidth=0, 
                               antialiased=True, shade=True)
        except Exception as e:
            # Fallback to simple scatter plot if sphere plotting fails
            print(f"Warning: Could not plot 3D sphere, falling back to circle: {e}")
            self.plot_atom_circle(position, radius, color)
    
    def plot_atom_circle(self, position, radius, color):
        """Plot a simple circle for an atom (low quality mode)."""
        # Simple scatter plot with larger markers for low quality mode
        self.ax.scatter(position[0], position[1], position[2],
                      s=radius*200, c=color, alpha=0.8, 
                      edgecolors='black', linewidths=0.5, marker='o')
    
    def plot_bonds_crystal_coords(self, na, nb, nc):
        """Plot bonds using proper Cartesian coordinates with proper supercell handling."""
        thickness = self.visualization_settings['bond_thickness']
        
        # Store drawn bonds for debugging
        self.drawn_bonds = []
        
        for bond in self.bonds:
            i, j = bond['atom1_idx'], bond['atom2_idx']
            
            # Get fractional coordinates
            pos1 = np.array(self.current_structure['sites'][i]['frac_coords'])
            pos2 = np.array(self.current_structure['sites'][j]['frac_coords'])
            
            # Use the original bond distance for validation
            original_bond_distance = bond['distance']
            
            # For supercell visualization, we need to draw bonds intelligently
            # Strategy: For each unit cell, find the shortest bond between atoms i and j
            for ia in range(na):
                for ja in range(nb):
                    for ka in range(nc):
                        # Fractional position of atom i in this unit cell
                        frac_pos1 = pos1 + np.array([ia, ja, ka])
                        
                        # Find the closest image of atom j to this atom i
                        best_frac_pos2, best_distance = self.find_closest_periodic_image(
                            frac_pos1, pos2, na, nb, nc
                        )
                        
                        # Convert to Cartesian coordinates
                        if self.pymatgen_structure:
                            cart_pos1 = self.pymatgen_structure.lattice.get_cartesian_coords(frac_pos1)
                            cart_pos2 = self.pymatgen_structure.lattice.get_cartesian_coords(best_frac_pos2)
                        else:
                            cart_pos1 = frac_pos1  # Fallback
                            cart_pos2 = best_frac_pos2
                        
                        # Calculate actual Cartesian distance
                        actual_distance = np.linalg.norm(cart_pos1 - cart_pos2)
                        
                        # Only draw bond if the distance matches our calculated bond (within tolerance)
                        distance_tolerance = 0.1  # Angstrom tolerance
                        
                        if abs(actual_distance - original_bond_distance) < distance_tolerance:
                            # Draw the bond in Cartesian space
                            self.ax.plot([cart_pos1[0], cart_pos2[0]],
                                       [cart_pos1[1], cart_pos2[1]],
                                       [cart_pos1[2], cart_pos2[2]],
                                       'k-', linewidth=thickness, alpha=0.6)
                            
                            # Add bond labels if enabled
                            if self.visualization_settings['label_bonds']:
                                self.add_bond_label(cart_pos1, cart_pos2, bond, actual_distance)
                            
                            # Store for debugging
                            self.drawn_bonds.append({
                                'atom1_idx': i,
                                'atom2_idx': j,
                                'pos1': cart_pos1.copy(),
                                'pos2': cart_pos2.copy(),
                                'drawn_distance': actual_distance,
                                'original_distance': original_bond_distance,
                                'unit_cell': [ia, ja, ka]
                            })
    
    def find_closest_periodic_image(self, frac_pos1, base_frac_pos2, na, nb, nc):
        """Find the closest periodic image of atom 2 to atom 1 in fractional coordinates."""
        best_frac_pos2 = None
        best_distance = float('inf')
        
        # Check all possible images of atom 2 within the supercell and neighboring cells
        for ib in range(-1, na + 1):
            for jb in range(-1, nb + 1):
                for kb in range(-1, nc + 1):
                    # Fractional position of atom 2 in this image
                    frac_pos2 = base_frac_pos2 + np.array([ib, jb, kb])
                    
                    # Calculate actual Cartesian distance
                    if self.pymatgen_structure:
                        cart_pos1 = self.pymatgen_structure.lattice.get_cartesian_coords(frac_pos1)
                        cart_pos2 = self.pymatgen_structure.lattice.get_cartesian_coords(frac_pos2)
                        distance = np.linalg.norm(cart_pos1 - cart_pos2)
                    else:
                        distance = np.linalg.norm(frac_pos1 - frac_pos2)  # Fallback
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_frac_pos2 = frac_pos2.copy()
        
        return best_frac_pos2, best_distance
    
    def calculate_actual_distance_crystal_coords(self, crystal_pos1, crystal_pos2):
        """Calculate actual distance between two points in crystal coordinates."""
        if not self.pymatgen_structure:
            # Fallback to simple Euclidean distance (not accurate)
            return np.linalg.norm(crystal_pos1 - crystal_pos2)
        
        # Convert crystal coordinates to Cartesian
        cart1 = self.pymatgen_structure.lattice.get_cartesian_coords(crystal_pos1)
        cart2 = self.pymatgen_structure.lattice.get_cartesian_coords(crystal_pos2)
        
        # Calculate Cartesian distance
        return np.linalg.norm(cart1 - cart2)
    
    def calculate_actual_distance_between_sites(self, site_i, site_j):
        """Calculate the actual shortest distance between two atomic sites considering periodicity."""
        if not self.pymatgen_structure:
            return 0.0
        
        site1 = self.pymatgen_structure[site_i]
        site2 = self.pymatgen_structure[site_j]
        
        # Use pymatgen's built-in distance calculation which handles periodicity
        return site1.distance(site2)
    
    def add_atom_label(self, crystal_pos, site_idx, element, unit_cell_i, unit_cell_j, unit_cell_k):
        """Add label to an atom."""
        # Create label text
        label_text = f"{element}{site_idx+1}"
        
        # Add unit cell information if in supercell
        if unit_cell_i != 0 or unit_cell_j != 0 or unit_cell_k != 0:
            label_text += f"({unit_cell_i},{unit_cell_j},{unit_cell_k})"
        
        # Position label slightly offset from atom center
        offset = 0.1
        label_pos = [crystal_pos[0] + offset, crystal_pos[1] + offset, crystal_pos[2] + offset]
        
        # Add text label
        self.ax.text(label_pos[0], label_pos[1], label_pos[2], label_text,
                    fontsize=8, color='black', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))
    
    def add_bond_label(self, pos1, pos2, bond, actual_distance):
        """Add label to a bond."""
        # Calculate midpoint of bond
        midpoint = [(pos1[0] + pos2[0]) / 2, 
                   (pos1[1] + pos2[1]) / 2, 
                   (pos1[2] + pos2[2]) / 2]
        
        # Create label text with distance and bond type
        bond_type = bond.get('bond_type', 'unknown')
        label_text = f"{bond_type}\n{actual_distance:.2f}Å"
        
        # Add Wyckoff information if available
        if 'wyckoff_info' in bond:
            label_text += f"\n({bond['wyckoff_info']})"
        
        # Add text label at midpoint
        self.ax.text(midpoint[0], midpoint[1], midpoint[2], label_text,
                    fontsize=7, color='darkblue', weight='bold', ha='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

    def plot_unit_cell(self):
        """Plot unit cell outline using actual crystal lattice vectors."""
        if not self.pymatgen_structure:
            return
            
        # Unit cell corners in fractional coordinates
        frac_corners = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
        ]
        
        # Convert to Cartesian coordinates using actual lattice
        cart_corners = []
        for frac_corner in frac_corners:
            cart_corner = self.pymatgen_structure.lattice.get_cartesian_coords(frac_corner)
            cart_corners.append(cart_corner)
        
        # Unit cell edges (same connectivity)
        edges = [
            # Bottom face
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face  
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        # Draw edges with actual lattice geometry
        for edge in edges:
            p1, p2 = cart_corners[edge[0]], cart_corners[edge[1]]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       color='#333333', linewidth=2.5, alpha=0.8, solid_capstyle='round')
    
    def plot_crystal_axes(self):
        """Plot actual crystal lattice vectors as axes."""
        if not self.pymatgen_structure:
            return
            
        # Origin in Cartesian coordinates
        origin = np.array([0.0, 0.0, 0.0])
        
        # Get actual lattice vectors
        lattice = self.pymatgen_structure.lattice
        
        # Scale factor for axis display (make them shorter for clarity)
        scale_factor = 0.3
        
        # Actual lattice vectors scaled down
        a_vector = lattice.matrix[0] * scale_factor
        b_vector = lattice.matrix[1] * scale_factor  
        c_vector = lattice.matrix[2] * scale_factor
        
        # Axis definitions with actual lattice directions
        axes = {
            'a': (origin, origin + a_vector, '#FF0000'),  # Red
            'b': (origin, origin + b_vector, '#00AA00'),  # Green
            'c': (origin, origin + c_vector, '#0000FF')   # Blue
        }
        
        for axis_name, (start, end, color) in axes.items():
            # Draw axis line representing actual lattice vector
            self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                       color=color, linewidth=4, alpha=0.9, solid_capstyle='round')
            
            # Add axis labels slightly beyond the end
            direction = end - start
            label_pos = end + direction * 0.2  # 20% beyond the end
            self.ax.text(label_pos[0], label_pos[1], label_pos[2], 
                        axis_name, fontsize=14, fontweight='bold', color=color,
                        ha='center', va='center')
    
    def set_equal_aspect_crystal(self):
        """Set proper aspect ratio for crystal coordinate system to preserve true geometry."""
        # Get current data limits
        x_limits = self.ax.get_xlim()
        y_limits = self.ax.get_ylim()
        z_limits = self.ax.get_zlim()
        
        # Find the actual data range
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        
        # Calculate centers
        x_center = (x_limits[0] + x_limits[1]) / 2
        y_center = (y_limits[0] + y_limits[1]) / 2
        z_center = (z_limits[0] + z_limits[1]) / 2
        
        # For crystal structures, we want to preserve the true geometric proportions
        # rather than forcing equal aspect ratios, as this can distort the crystal geometry
        
        # Add minimal padding (2% of each range)
        padding_factor = 0.02
        x_pad = max(x_range * padding_factor, 0.1)  # Minimum 0.1 Å padding
        y_pad = max(y_range * padding_factor, 0.1)
        z_pad = max(z_range * padding_factor, 0.1)
        
        # Set limits that preserve the actual crystal geometry
        self.ax.set_xlim([x_center - x_range/2 - x_pad, x_center + x_range/2 + x_pad])
        self.ax.set_ylim([y_center - y_range/2 - y_pad, y_center + y_range/2 + y_pad])
        self.ax.set_zlim([z_center - z_range/2 - z_pad, z_center + z_range/2 + z_pad])
        
        # Set aspect ratio based on user preference
        try:
            preserve_geometry = self.visualization_settings.get('preserve_geometry', True)
            
            if preserve_geometry:
                # Preserve crystal geometry by using actual data ranges
                ranges = [x_range, y_range, z_range]
                max_range = max(ranges)
                
                # Calculate normalized aspect ratios
                if max_range > 0:
                    aspect_ratios = [r/max_range for r in ranges]
                    # Clamp aspect ratios to prevent extreme distortions
                    min_aspect = 0.3  # Minimum aspect ratio to prevent too much squashing
                    aspect_ratios = [max(ar, min_aspect) for ar in aspect_ratios]
                    self.ax.set_box_aspect(aspect_ratios)
                else:
                    self.ax.set_box_aspect([1, 1, 1])  # Fallback to cubic
            else:
                # Force equal aspect ratios (may distort crystal geometry)
                self.ax.set_box_aspect([1, 1, 1])
                
        except AttributeError:
            # Fallback for older matplotlib versions
            pass
    
    def optimize_view_bounds(self):
        """Optimize the view bounds to better fill the available space."""
        if not self.current_structure or not self.pymatgen_structure:
            return
        
        # Get supercell dimensions
        na, nb, nc = self.supercell_size
        
        # Calculate the actual extent of the structure in Cartesian coordinates
        all_positions = []
        
        for site in self.current_structure['sites']:
            frac_coords = np.array(site['frac_coords'])
            
            # Add all supercell positions
            for i in range(na):
                for j in range(nb):
                    for k in range(nc):
                        frac_pos = frac_coords + np.array([i, j, k])
                        cart_pos = self.pymatgen_structure.lattice.get_cartesian_coords(frac_pos)
                        all_positions.append(cart_pos)
        
        if all_positions:
            all_positions = np.array(all_positions)
            
            # Find actual bounds with some padding
            x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
            y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
            z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])
            
            # Add small padding (5% of the range)
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            padding = 0.05  # 5% padding
            x_pad = max(x_range * padding, 0.5)  # Minimum 0.5 Å padding
            y_pad = max(y_range * padding, 0.5)
            z_pad = max(z_range * padding, 0.5)
            
            # Set optimized bounds
            self.ax.set_xlim([x_min - x_pad, x_max + x_pad])
            self.ax.set_ylim([y_min - y_pad, y_max + y_pad])
            self.ax.set_zlim([z_min - z_pad, z_max + z_pad])
    
    # === Control Methods ===
    
    def update_visualization_settings(self):
        """Update visualization settings from controls."""
        self.visualization_settings['show_bonds'] = self.show_bonds_cb.isChecked()
        self.visualization_settings['show_unit_cell'] = self.show_unit_cell_cb.isChecked()
        self.visualization_settings['show_axes'] = self.show_axes_cb.isChecked()
        self.visualization_settings['label_atoms'] = self.label_atoms_cb.isChecked()
        self.visualization_settings['label_bonds'] = self.label_bonds_cb.isChecked()
        self.visualization_settings['preserve_geometry'] = self.preserve_geometry_cb.isChecked()
        self.update_3d_plot()
    
    def update_atom_scale(self, value):
        """Update atom scale factor."""
        self.visualization_settings['atom_scale'] = value / 100.0
        self.update_3d_plot()
    
    def update_bond_thickness(self, value):
        """Update bond thickness."""
        self.visualization_settings['bond_thickness'] = value / 100.0
        self.update_3d_plot()
    
    def update_color_scheme(self, scheme):
        """Update color scheme."""
        self.visualization_settings['color_scheme'] = scheme.lower().replace(' ', '_')
        self.update_3d_plot()
    
    def update_render_quality(self, quality):
        """Update render quality setting."""
        self.visualization_settings['render_quality'] = quality
        self.update_3d_plot()
        print(f"🎨 Render quality changed to: {quality}")
    
    def set_view_preset(self, axis):
        """Set view to look along specified crystal axis."""
        if axis == 'a':
            self.view_elevation = 0
            self.view_azimuth = 0
        elif axis == 'b':
            self.view_elevation = 0
            self.view_azimuth = 90
        elif axis == 'c':
            self.view_elevation = 90
            self.view_azimuth = 0
        
        # Update sliders
        self.elevation_slider.setValue(self.view_elevation)
        self.azimuth_slider.setValue(self.view_azimuth)
        
        self.update_3d_plot()
    
    def toggle_auto_rotation(self, enabled):
        """Toggle automatic rotation."""
        self.auto_rotate = enabled
        if enabled:
            self.rotation_timer.start(50)  # 50ms interval for smooth rotation
        else:
            self.rotation_timer.stop()
    
    def update_rotation(self):
        """Update rotation for auto-rotate mode."""
        if self.auto_rotate:
            self.view_azimuth = (self.view_azimuth + 2) % 360
            self.azimuth_slider.setValue(self.view_azimuth)
            self.update_3d_plot()
    
    def update_elevation(self, value):
        """Update view elevation."""
        self.view_elevation = value
        self.update_3d_plot()
    
    def update_azimuth(self, value):
        """Update view azimuth."""
        self.view_azimuth = value
        self.update_3d_plot()
    
    def reset_view(self):
        """Reset view to default."""
        self.view_elevation = 20
        self.view_azimuth = 45
        self.elevation_slider.setValue(self.view_elevation)
        self.azimuth_slider.setValue(self.view_azimuth)
        self.update_3d_plot()
    
    def update_supercell(self):
        """Update supercell size."""
        self.supercell_size = [
            self.supercell_a_spin.value(),
            self.supercell_b_spin.value(), 
            self.supercell_c_spin.value()
        ]
        self.update_3d_plot()
    
    def set_supercell_preset(self, a, b, c):
        """Set supercell to preset values."""
        self.supercell_a_spin.setValue(a)
        self.supercell_b_spin.setValue(b)
        self.supercell_c_spin.setValue(c)
        self.update_supercell()
    
    def update_global_bond_cutoff(self):
        """Update global bond distance constraints."""
        if self.bonds:  # Recalculate bonds if they exist
            self.calculate_bonds()
    
    def auto_generate_bond_controls(self):
        """Auto-generate element pair controls based on current structure."""
        if not self.current_structure:
            QMessageBox.warning(self, "Warning", "Please load a structure first.")
            return
        
        # Clear existing controls
        for i in reversed(range(self.bond_controls_layout.count())):
            child = self.bond_controls_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Get all unique element pairs
        elements = self.current_structure['elements']
        element_pairs = []
        
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements):
                if i <= j:  # Avoid duplicates
                    pair_key = f"{min(elem1, elem2)}-{max(elem1, elem2)}"
                    if pair_key not in [pair[0] for pair in element_pairs]:
                        element_pairs.append((pair_key, elem1, elem2))
        
        # Create controls for each pair
        for pair_key, elem1, elem2 in element_pairs:
            self.create_element_pair_control(pair_key, elem1, elem2)
        
        print(f"✓ Generated controls for {len(element_pairs)} element pairs")
    
    def create_element_pair_control(self, pair_key, elem1, elem2):
        """Create control widget for an element pair."""
        # Main widget for this pair
        pair_widget = QWidget()
        pair_layout = QVBoxLayout(pair_widget)
        pair_layout.setContentsMargins(5, 5, 5, 5)
        
        # Header with enable/disable
        header_layout = QHBoxLayout()
        
        # Enable checkbox
        enable_cb = QCheckBox(f"{pair_key}")
        enable_cb.setChecked(True)
        enable_cb.setFont(QFont("Arial", 10, QFont.Bold))
        header_layout.addWidget(enable_cb)
        header_layout.addStretch()
        
        pair_layout.addLayout(header_layout)
        
        # Distance controls
        distance_layout = QGridLayout()
        
        # Minimum distance
        distance_layout.addWidget(QLabel("Min (Å):"), 0, 0)
        min_spin = QDoubleSpinBox()
        min_spin.setRange(0.1, 5.0)
        min_spin.setValue(0.5)
        min_spin.setSingleStep(0.1)
        distance_layout.addWidget(min_spin, 0, 1)
        
        # Maximum distance  
        distance_layout.addWidget(QLabel("Max (Å):"), 0, 2)
        max_spin = QDoubleSpinBox()
        max_spin.setRange(0.5, 8.0)
        max_spin.setValue(self.get_default_bond_distance(elem1, elem2))
        max_spin.setSingleStep(0.1)
        distance_layout.addWidget(max_spin, 0, 3)
        
        pair_layout.addLayout(distance_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #cccccc;")
        pair_layout.addWidget(separator)
        
        # Store references and connect signals
        control_data = {
            'widget': pair_widget,
            'enabled_cb': enable_cb,
            'min_spin': min_spin,
            'max_spin': max_spin,
            'enabled': True,
            'min_distance': min_spin.value(),
            'max_distance': max_spin.value()
        }
        
        # Connect signals to update bond calculations
        enable_cb.toggled.connect(lambda checked, pk=pair_key: self.update_element_pair_enabled(pk, checked))
        min_spin.valueChanged.connect(lambda value, pk=pair_key: self.update_element_pair_min(pk, value))
        max_spin.valueChanged.connect(lambda value, pk=pair_key: self.update_element_pair_max(pk, value))
        
        self.element_bond_controls[pair_key] = control_data
        self.bond_controls_layout.addWidget(pair_widget)
    
    def get_default_bond_distance(self, elem1, elem2):
        """Get reasonable default bond distance for element pair."""
        # Common bond distances (approximate)
        bond_defaults = {
            'Ti-O': 2.0,
            'Si-O': 1.6,
            'Al-O': 1.8,
            'C-C': 1.5,
            'C-O': 1.4,
            'C-N': 1.5,
            'N-O': 1.4,
            'H-O': 1.0,
            'H-N': 1.0,
            'H-C': 1.1,
            'Ca-O': 2.4,
            'Mg-O': 2.1,
            'Fe-O': 2.1,
            'Na-O': 2.4,
            'K-O': 2.8
        }
        
        pair_key = f"{min(elem1, elem2)}-{max(elem1, elem2)}"
        return bond_defaults.get(pair_key, 3.0)  # Default to 3.0 Å
    
    def update_element_pair_enabled(self, pair_key, enabled):
        """Update enabled status for element pair."""
        if pair_key in self.element_bond_controls:
            self.element_bond_controls[pair_key]['enabled'] = enabled
            if self.bonds:  # Recalculate if bonds exist
                self.calculate_bonds()
    
    def update_element_pair_min(self, pair_key, min_distance):
        """Update minimum distance for element pair."""
        if pair_key in self.element_bond_controls:
            self.element_bond_controls[pair_key]['min_distance'] = min_distance
            if self.bonds:  # Recalculate if bonds exist
                self.calculate_bonds()
    
    def update_element_pair_max(self, pair_key, max_distance):
        """Update maximum distance for element pair."""
        if pair_key in self.element_bond_controls:
            self.element_bond_controls[pair_key]['max_distance'] = max_distance
            if self.bonds:  # Recalculate if bonds exist
                self.calculate_bonds()
    
    def clear_bonds(self):
        """Clear all calculated bonds."""
        self.bonds = []
        self.update_3d_plot()
        self.update_bond_statistics([])
        print("🧹 All bonds cleared")
    
    def update_bond_statistics(self, bond_distances):
        """Update detailed bond statistics display."""
        if not bond_distances:
            # Update both statistics labels
            if hasattr(self, 'bond_stats_label'):
                self.bond_stats_label.setText("No bonds calculated")
            if hasattr(self, 'bond_stats_detailed'):
                self.bond_stats_detailed.setText("No bonds calculated")
            return
        
        # Basic statistics
        avg_distance = np.mean(bond_distances)
        min_distance_val = np.min(bond_distances)
        max_distance_val = np.max(bond_distances)
        
        # Count by bond type
        bond_type_counts = {}
        bond_type_distances = {}
        
        for bond in self.bonds:
            bond_type = bond['bond_type']
            distance = bond['distance']
            
            if bond_type not in bond_type_counts:
                bond_type_counts[bond_type] = 0
                bond_type_distances[bond_type] = []
            
            bond_type_counts[bond_type] += 1
            bond_type_distances[bond_type].append(distance)
        
        # Update basic statistics (for old label if it exists)
        if hasattr(self, 'bond_stats_label'):
            stats_text = f"Bonds: {len(self.bonds)}, "
            stats_text += f"Avg: {avg_distance:.2f} Å, "
            stats_text += f"Range: {min_distance_val:.2f}-{max_distance_val:.2f} Å"
            self.bond_stats_label.setText(stats_text)
        
        # Update detailed statistics
        if hasattr(self, 'bond_stats_detailed'):
            detailed_text = f"Total bonds: {len(self.bonds)}\n"
            detailed_text += f"Overall range: {min_distance_val:.2f}-{max_distance_val:.2f} Å\n"
            detailed_text += f"Average: {avg_distance:.2f} Å\n\n"
            
            detailed_text += "By bond type:\n"
            for bond_type, count in sorted(bond_type_counts.items()):
                avg_type = np.mean(bond_type_distances[bond_type])
                min_type = np.min(bond_type_distances[bond_type])
                max_type = np.max(bond_type_distances[bond_type])
                detailed_text += f"{bond_type}: {count} bonds\n"
                detailed_text += f"  {min_type:.2f}-{max_type:.2f} Å (avg {avg_type:.2f})\n"
            
            self.bond_stats_detailed.setText(detailed_text)
    
    def update_bond_method(self, method):
        """Update bond calculation method."""
        print(f"🔧 Bond calculation method changed to: {method}")
        if self.bonds:  # Recalculate if bonds exist
            self.calculate_bonds()
    
    def show_detailed_analysis(self):
        """Show detailed structure analysis dialog."""
        if not self.current_structure:
            QMessageBox.warning(self, "Warning", "Please load a structure first.")
            return
        
        # Create analysis dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Detailed Structure Analysis")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Analysis content
        analysis_text = QTextEdit()
        analysis_text.setReadOnly(True)
        
        # Generate detailed analysis
        analysis_content = self.generate_structure_analysis()
        analysis_text.setPlainText(analysis_content)
        
        layout.addWidget(analysis_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def generate_structure_analysis(self):
        """Generate detailed structure analysis text."""
        if not self.current_structure:
            return "No structure loaded"
        
        analysis = "DETAILED CRYSTAL STRUCTURE ANALYSIS\n"
        analysis += "=" * 50 + "\n\n"
        
        # Basic information
        analysis += f"Formula: {self.current_structure['formula']}\n"
        analysis += f"Space Group: {self.current_structure['space_group']}\n"
        analysis += f"Crystal System: {self.current_structure['crystal_system']}\n"
        analysis += f"Number of atoms: {self.current_structure['num_atoms']}\n\n"
        
        # Lattice parameters
        lattice = self.current_structure['lattice_params']
        analysis += "LATTICE PARAMETERS:\n"
        analysis += f"a = {lattice['a']:.4f} Å\n"
        analysis += f"b = {lattice['b']:.4f} Å\n"
        analysis += f"c = {lattice['c']:.4f} Å\n"
        analysis += f"α = {lattice['alpha']:.2f}°\n"
        analysis += f"β = {lattice['beta']:.2f}°\n"
        analysis += f"γ = {lattice['gamma']:.2f}°\n"
        analysis += f"Volume = {lattice['volume']:.2f} Å³\n\n"
        
        # Wyckoff positions
        if 'wyckoff_positions' in self.current_structure and self.current_structure['wyckoff_positions']:
            analysis += "WYCKOFF POSITIONS:\n"
            for key, wyckoff_data in self.current_structure['wyckoff_positions'].items():
                analysis += f"{key}:\n"
                analysis += f"  Element: {wyckoff_data['element']}\n"
                analysis += f"  Wyckoff symbol: {wyckoff_data['wyckoff_symbol']}\n"
                analysis += f"  Multiplicity: {wyckoff_data['multiplicity']}\n"
                analysis += f"  Site symmetry: {wyckoff_data['site_symmetry']}\n"
                analysis += f"  Representative coordinates: ({wyckoff_data['representative_coords'][0]:.4f}, {wyckoff_data['representative_coords'][1]:.4f}, {wyckoff_data['representative_coords'][2]:.4f})\n"
                analysis += f"  Sites: {len(wyckoff_data['sites'])}\n"
            analysis += "\n"
        
        # Atomic sites
        analysis += "ATOMIC SITES:\n"
        for i, site in enumerate(self.current_structure['sites']):
            analysis += f"Site {i+1}: {site['element']}\n"
            analysis += f"  Fractional coordinates: ({site['frac_coords'][0]:.4f}, {site['frac_coords'][1]:.4f}, {site['frac_coords'][2]:.4f})\n"
            analysis += f"  Cartesian coordinates: ({site['cart_coords'][0]:.4f}, {site['cart_coords'][1]:.4f}, {site['cart_coords'][2]:.4f})\n"
            
            # Add Wyckoff information
            if 'wyckoff_symbol' in site:
                analysis += f"  Wyckoff position: {site['wyckoff_symbol']}\n"
            if 'site_symmetry' in site:
                analysis += f"  Site symmetry: {site['site_symmetry']}\n"
            if 'multiplicity' in site:
                analysis += f"  Multiplicity: {site['multiplicity']}\n"
        
        analysis += "\n"
        
        # Bond analysis
        if self.bonds:
            analysis += f"BOND ANALYSIS:\n"
            analysis += f"Total bonds calculated: {len(self.bonds)}\n\n"
            
            # Group by bond type
            bond_groups = {}
            for bond in self.bonds:
                bond_type = bond['bond_type']
                if bond_type not in bond_groups:
                    bond_groups[bond_type] = []
                bond_groups[bond_type].append(bond)
            
            for bond_type, bonds in bond_groups.items():
                distances = [b['distance'] for b in bonds]
                analysis += f"{bond_type} bonds ({len(bonds)} total):\n"
                analysis += f"  Distance range: {min(distances):.3f} - {max(distances):.3f} Å\n"
                analysis += f"  Average distance: {np.mean(distances):.3f} Å\n"
                
                # Show individual bonds if not too many
                if len(bonds) <= 5:
                    for bond in bonds:
                        wyckoff_info = bond.get('wyckoff_info', 'a-a')
                        analysis += f"    {bond['atom1_element']}{bond['atom1_idx']+1}-{bond['atom2_element']}{bond['atom2_idx']+1}: {bond['distance']:.3f} Å"
                        analysis += f" (Wyckoff: {wyckoff_info})\n"
                elif len(bonds) <= 10:
                    analysis += f"    (showing first 3 of {len(bonds)})\n"
                    for bond in bonds[:3]:
                        wyckoff_info = bond.get('wyckoff_info', 'a-a')
                        analysis += f"    {bond['atom1_element']}{bond['atom1_idx']+1}-{bond['atom2_element']}{bond['atom2_idx']+1}: {bond['distance']:.3f} Å"
                        analysis += f" (Wyckoff: {wyckoff_info})\n"
                analysis += "\n"
        else:
            analysis += "BOND ANALYSIS:\nNo bonds calculated. Use the Bonds tab to calculate bonds.\n\n"
        
        # Supercell information
        analysis += "VISUALIZATION SETTINGS:\n"
        analysis += f"Current supercell: {self.supercell_size[0]}×{self.supercell_size[1]}×{self.supercell_size[2]}\n"
        analysis += f"Total visualized atoms: {self.current_structure['num_atoms'] * np.prod(self.supercell_size)}\n"
        
        return analysis
    
    def refresh_wyckoff_analysis(self):
        """Refresh Wyckoff position analysis and update display."""
        if not self.current_structure:
            QMessageBox.warning(self, "Warning", "Please load a structure first.")
            return
        
        # Re-run symmetry analysis
        self.generate_equivalent_sites()
        self.group_wyckoff_positions()
        
        # Update all displays
        self.update_wyckoff_table()
        self.update_symmetry_info()
        self.update_equivalent_sites_info()
        self.update_structure_info_display()
        
        print("🔬 Wyckoff position analysis refreshed")
    
    def update_wyckoff_table(self):
        """Update the Wyckoff positions table."""
        if not hasattr(self, 'wyckoff_table'):
            return
        
        if not self.current_structure or 'wyckoff_positions' not in self.current_structure:
            self.wyckoff_table.setRowCount(0)
            return
        
        wyckoff_positions = self.current_structure['wyckoff_positions']
        self.wyckoff_table.setRowCount(len(wyckoff_positions))
        
        for row, (key, wyckoff_data) in enumerate(wyckoff_positions.items()):
            # Element
            self.wyckoff_table.setItem(row, 0, QTableWidgetItem(wyckoff_data['element']))
            
            # Wyckoff symbol
            self.wyckoff_table.setItem(row, 1, QTableWidgetItem(wyckoff_data['wyckoff_symbol']))
            
            # Multiplicity
            self.wyckoff_table.setItem(row, 2, QTableWidgetItem(str(wyckoff_data['multiplicity'])))
            
            # Site symmetry
            self.wyckoff_table.setItem(row, 3, QTableWidgetItem(wyckoff_data['site_symmetry']))
            
            # Representative coordinates
            coords = wyckoff_data['representative_coords']
            coord_str = f"({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})"
            self.wyckoff_table.setItem(row, 4, QTableWidgetItem(coord_str))
            
            # Set row color based on element
            element_color = self.get_element_color(wyckoff_data['element'])
            for col in range(5):
                item = self.wyckoff_table.item(row, col)
                if item:
                    item.setBackground(QColor(element_color))
    
    def get_element_color(self, element):
        """Get a light color for element background."""
        element_colors = {
            'H': '#FFEEEE', 'C': '#EEEEEE', 'N': '#EEEEFF', 'O': '#FFEEEE',
            'F': '#EEFFEE', 'P': '#FFEEDD', 'S': '#FFFFEE', 'Cl': '#EEFFEE',
            'Ti': '#EEEEFF', 'Si': '#FFFFDD', 'Ca': '#EEFFDD', 'Fe': '#FFDDDD',
            'Mg': '#DDFFDD', 'Al': '#EEDDDD', 'K': '#DDDDFF', 'Na': '#DDDDFF'
        }
        return element_colors.get(element, '#F0F0F0')
    
    def update_symmetry_info(self):
        """Update symmetry operations information."""
        if not hasattr(self, 'symmetry_info'):
            return
        
        if not self.current_structure:
            self.symmetry_info.setText("No structure loaded")
            return
        
        info_text = f"Space Group: {self.current_structure['space_group']}"
        if 'space_group_number' in self.current_structure:
            info_text += f" (#{self.current_structure['space_group_number']})"
        
        info_text += f"\nCrystal System: {self.current_structure['crystal_system']}"
        
        if 'symmetry_operations' in self.current_structure:
            num_ops = len(self.current_structure['symmetry_operations'])
            info_text += f"\nSymmetry Operations: {num_ops}"
            
            # Show first few operations
            if num_ops > 0:
                info_text += f"\nOperations: Identity"
                if num_ops > 1:
                    info_text += f", +{num_ops-1} others"
        
        self.symmetry_info.setText(info_text)
    
    def update_equivalent_sites_info(self):
        """Update equivalent sites information."""
        if not hasattr(self, 'equiv_sites_info'):
            return
        
        if not self.current_structure or 'equivalent_sites' not in self.current_structure:
            self.equiv_sites_info.setText("No equivalent sites generated")
            return
        
        equivalent_sites = self.current_structure['equivalent_sites']
        total_equiv_positions = sum(len(sites) for sites in equivalent_sites.values())
        
        info_text = f"Generated {total_equiv_positions} equivalent positions\n"
        info_text += f"for {len(equivalent_sites)} unique atomic sites\n"
        
        # Show breakdown by site
        breakdown = []
        for site_idx, equiv_positions in equivalent_sites.items():
            if site_idx < len(self.current_structure['sites']):
                element = self.current_structure['sites'][site_idx]['element']
                breakdown.append(f"{element}({len(equiv_positions)})")
        
        if len(breakdown) <= 5:
            info_text += f"Sites: {', '.join(breakdown)}"
        else:
            info_text += f"Sites: {', '.join(breakdown[:3])}, +{len(breakdown)-3} more"
        
        self.equiv_sites_info.setText(info_text)
    
    def toggle_equivalent_sites_display(self, enabled):
        """Toggle display of equivalent sites in visualization."""
        # This would be implemented to show/hide equivalent sites in the 3D plot
        # For now, just update the plot
        self.update_3d_plot()
        
        if enabled:
            print("🔬 Showing all symmetrically equivalent sites")
        else:
            print("🔬 Showing only unit cell sites")
    
    def export_structure(self):
        """Export structure data."""
        if not self.current_structure:
            QMessageBox.warning(self, "Warning", "No structure to export.")
            return
        
        QMessageBox.information(self, "Export", "Structure export functionality to be implemented.")
    
    def save_view_image(self):
        """Save current view as image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save View", "", "PNG files (*.png);;PDF files (*.pdf);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", "View saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving view:\n{str(e)}")


# Test function
def main():
    """Test the crystal structure widget."""
    from PySide6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    widget = CrystalStructureWidget()
    widget.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 