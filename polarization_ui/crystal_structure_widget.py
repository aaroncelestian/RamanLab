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
    QTabWidget, QListWidget, QListWidgetItem, QDialog
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QColor

# PyVista for 3D visualization
import pyvista as pv
from pyvistaqt import QtInteractor

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
        self.drawn_bonds = []  # Track bonds that are actually drawn
        self.supercell_size = [1, 1, 1]
        # Default visualization settings
        self.visualization_settings = {
            'show_atoms': True,
            'show_bonds': True,
            'show_unit_cell': True,
            'show_axes': True,
            'atom_scale': 0.3,  # Reduced from 0.5 to make atoms smaller
            'bond_radius': 0.05,  # Reduced bond thickness
            'transparency': 0.3,  # Reduced from 0.8 to make less transparent
            'color_scheme': 'element',
            'label_atoms': False,
            'supercell': [1, 1, 1],
            'bond_tolerance': 0.3  # Added tolerance for bond detection
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
    
    def apply_flat_rounded_style(self, button):
        """Apply flat rounded styling to a button."""
        button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 4px 16px;
                font-weight: 500;
                color: #333333;
                font-size: 12px;
                min-height: 12px;
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
        button.setFlat(False)
    
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
        self.apply_flat_rounded_style(calc_bonds_btn)
        calc_layout.addWidget(calc_bonds_btn)
        
        # Clear bonds button
        clear_bonds_btn = QPushButton("Clear All Bonds")
        clear_bonds_btn.clicked.connect(self.clear_bonds)
        self.apply_flat_rounded_style(clear_bonds_btn)
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
        self.apply_flat_rounded_style(auto_gen_btn)
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
        self.apply_flat_rounded_style(debug_bonds_btn)
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
        self.apply_flat_rounded_style(refresh_wyckoff_btn)
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
        self.apply_flat_rounded_style(analyze_btn)
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
        self.apply_flat_rounded_style(load_btn)
        layout.addWidget(load_btn)
        
        # Export structure button
        export_btn = QPushButton("Export Structure")
        export_btn.clicked.connect(self.export_structure)
        self.apply_flat_rounded_style(export_btn)
        layout.addWidget(export_btn)
        
        # Save view button
        save_view_btn = QPushButton("Save View as Image")
        save_view_btn.clicked.connect(self.save_view_image)
        self.apply_flat_rounded_style(save_view_btn)
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
        
        # Transparency slider
        layout.addWidget(QLabel("Transparency:"))
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(30)  # Default to 0.3 opacity (1 - 0.7)
        self.transparency_slider.valueChanged.connect(
            lambda value: self.update_visualization_settings('transparency', 1 - value/100.0)
        )
        layout.addWidget(self.transparency_slider)
        
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
        self.preserve_geometry_cb.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #2E7D32;
                background-color: #E8F5E8;
                padding: 4px;
                border-radius: 3px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #2E7D32;
            }
        """)
        self.preserve_geometry_cb.toggled.connect(self.update_visualization_settings)
        layout.addWidget(self.preserve_geometry_cb)
        
        # Unit cell corner visualization (for debugging)
        self.show_unit_cell_corners_cb = QCheckBox("Show Unit Cell Corners")
        self.show_unit_cell_corners_cb.setChecked(False)
        self.show_unit_cell_corners_cb.setToolTip("Show corner markers for unit cell debugging")
        self.show_unit_cell_corners_cb.toggled.connect(self.update_visualization_settings)
        layout.addWidget(self.show_unit_cell_corners_cb)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Debug info button
        debug_btn = QPushButton("Show Unit Cell Info")
        debug_btn.clicked.connect(self.show_unit_cell_debug_info)
        self.apply_flat_rounded_style(debug_btn)
        layout.addWidget(debug_btn)
        
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
        self.apply_flat_rounded_style(preset_a_btn)
        preset_layout.addWidget(preset_a_btn)
        
        preset_b_btn = QPushButton("b-axis")
        preset_b_btn.clicked.connect(lambda: self.set_view_preset('b'))
        self.apply_flat_rounded_style(preset_b_btn)
        preset_layout.addWidget(preset_b_btn)
        
        preset_c_btn = QPushButton("c-axis")
        preset_c_btn.clicked.connect(lambda: self.set_view_preset('c'))
        self.apply_flat_rounded_style(preset_c_btn)
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
        self.apply_flat_rounded_style(reset_btn)
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
        
        # Create PyVista QtInteractor for 3D visualization
        self.pv_widget = QtInteractor(viz_frame)
        layout.addWidget(self.pv_widget)
        
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
            structures = parser.parse_structures(primitive=False)  # Use conventional cell to match main app
            
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
            
            # Get crystal system info for the message
            crystal_system = self.current_structure.get('crystal_system', 'Unknown')
            lattice_params = self.current_structure.get('lattice_params', {})
            
            success_msg = (f"Successfully loaded crystal structure from {os.path.basename(file_path)}\n\n"
                          f"Crystal System: {crystal_system}\n"
                          f"Unit Cell: a={lattice_params.get('a', 0):.3f}Å, "
                          f"b={lattice_params.get('b', 0):.3f}Å, "
                          f"c={lattice_params.get('c', 0):.3f}Å\n"
                          f"Angles: α={lattice_params.get('alpha', 0):.1f}°, "
                          f"β={lattice_params.get('beta', 0):.1f}°, "
                          f"γ={lattice_params.get('gamma', 0):.1f}°\n\n"
                          f"✓ Crystal geometry is being preserved based on actual lattice parameters.\n"
                          f"Check 'View' tab if unit cell appears distorted.")
            
            QMessageBox.information(self, "Crystal Structure Loaded", success_msg)
            
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
        if not self.current_structure:
            print("No current structure to generate equivalent sites")
            return
            
        if not self.current_structure.get('symmetry_operations'):
            print("No symmetry operations found in structure")
            return
            
        print(f"Generating equivalent sites for {len(self.current_structure['sites'])} unique sites")
        
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
        total_sites = sum(len(sites) for sites in equivalent_sites.values())
        print(f"✓ Generated {total_sites} equivalent sites from {len(original_sites)} unique sites using {len(symm_ops)} symmetry operations")
    
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
        """Calculate bonds between atoms using pymatgen's neighbor list with nearest neighbors only."""
        if not self.pymatgen_structure:
            return []
            
        try:
            from pymatgen.analysis.local_env import CrystalNN
            
            # Use CrystalNN with more strict cutoff parameters
            cnn = CrystalNN(
                search_cutoff=3.0,  # Maximum distance to search for neighbors
                distance_cutoffs=(0.1, 0.5),  # Min and max bond distance as fraction of sum of radii
                x_diff_weight=-1,  # Don't weight by x-difference
                porous_adjustment=False  # Don't adjust for porous structures
            )
            
            bonds = []
            bonded_pairs = set()  # Track bonded pairs to avoid duplicates
            
            for i, site in enumerate(self.pymatgen_structure):
                try:
                    # Get only the nearest neighbors
                    neighbors = cnn.get_nn_info(self.pymatgen_structure, i)
                    
                    for neighbor in neighbors:
                        j = neighbor['site_index']
                        # Add bond only if not already added (i < j ensures no duplicates)
                        if i < j:
                            bond_key = tuple(sorted((i, j)))
                            if bond_key not in bonded_pairs:
                                bonds.append((i, j, neighbor['weight']))
                                bonded_pairs.add(bond_key)
                                
                except Exception as e:
                    print(f"Error finding neighbors for site {i}: {str(e)}")
                    continue
            
            # Sort bonds by distance (weight in the CrystalNN result)
            bonds.sort(key=lambda x: x[2])
            
            # Only keep the shortest bonds (nearest neighbors)
            # First, find the minimum distance for each atom
            min_distances = {}
            for i, j, dist in bonds:
                if i not in min_distances or dist < min_distances[i]:
                    min_distances[i] = dist
                if j not in min_distances or dist < min_distances[j]:
                    min_distances[j] = dist
            
            # Filter bonds to keep only those within tolerance of the minimum distance
            tolerance = self.visualization_settings.get('bond_tolerance', 0.3)
            filtered_bonds = []
            for i, j, dist in bonds:
                min_dist = min(min_distances[i], min_distances[j])
                if dist <= min_dist * (1 + tolerance):
                    filtered_bonds.append((i, j))
            
            print(f"Bonds calculated: {len(filtered_bonds)} (filtered from {len(bonds)} initial bonds)")
            return filtered_bonds
            
        except ImportError:
            print("pymatgen not available for bond calculation")
            return []           
    
    def update_visualization_settings(self, setting_name=None, value=None):
        """Update visualization settings and refresh the 3D plot."""
        if setting_name is not None and value is not None:
            self.visualization_settings[setting_name] = value
        
        # Update UI elements to match current settings
        if hasattr(self, 'show_bonds_cb') and 'show_bonds' in self.visualization_settings:
            self.show_bonds_cb.setChecked(self.visualization_settings['show_bonds'])
            
        if hasattr(self, 'show_unit_cell_cb') and 'show_unit_cell' in self.visualization_settings:
            self.show_unit_cell_cb.setChecked(self.visualization_settings['show_unit_cell'])
            
        if hasattr(self, 'show_axes_cb') and 'show_axes' in self.visualization_settings:
            self.show_axes_cb.setChecked(self.visualization_settings['show_axes'])
            
        if hasattr(self, 'atom_scale_slider') and 'atom_scale' in self.visualization_settings:
            self.atom_scale_slider.setValue(int(self.visualization_settings['atom_scale'] * 100))
            
        if hasattr(self, 'transparency_slider') and 'transparency' in self.visualization_settings:
            self.transparency_slider.setValue(int((1 - self.visualization_settings['transparency']) * 100))
        
        # Update the 3D plot with new settings
        self.update_3d_plot()

    def debug_structure_info(self):
        """Print debug information about the current structure."""
        if not self.current_structure:
            print("No structure loaded for debugging")
            return
            
        print("\n=== Structure Debug Info ===")
        print(f"Formula: {self.current_structure.get('formula', 'N/A')}")
        print(f"Space Group: {self.current_structure.get('space_group', 'N/A')}")
        print(f"Number of sites: {len(self.current_structure.get('sites', []))}")
        
        if 'symmetry_operations' in self.current_structure:
            print(f"Symmetry operations: {len(self.current_structure['symmetry_operations'])}")
        else:
            print("No symmetry operations found in structure")
            
        if 'equivalent_sites' in self.current_structure:
            total_equiv = sum(len(sites) for sites in self.current_structure['equivalent_sites'].values())
            print(f"Total equivalent sites: {total_equiv}")
        else:
            print("No equivalent sites generated")
            
        print("=" * 30 + "\n")

    def update_3d_plot(self):
        """Update the 3D structure visualization using PyVista."""
        if not self.current_structure:
            self.pv_widget.add_text('Load a CIF file to view structure', position='upper_left', font_size=14, color='gray')
            self.pv_widget.reset_camera()
            return
            
        print("\n=== Starting 3D Plot Update ===")
        # Force clear any existing meshes
        self.pv_widget.clear()
        
        # Ensure all equivalent sites are generated
        print("Generating equivalent sites...")
        self.generate_equivalent_sites()
        
        # If no equivalent sites were generated, ensure we have the original sites
        if 'equivalent_sites' not in self.current_structure or not self.current_structure['equivalent_sites']:
            print("No equivalent sites generated, using original sites")
            self.current_structure['equivalent_sites'] = {
                i: [{'frac_coords': site['frac_coords']}]
                for i, site in enumerate(self.current_structure['sites'])
            }
        
        na, nb, nc = self.supercell_size
        self.plot_atoms_pyvista(na, nb, nc)
        if self.visualization_settings.get('show_bonds', True) and self.bonds:
            self.plot_bonds_pyvista(na, nb, nc)
        if self.visualization_settings.get('show_unit_cell', True):
            self.plot_unit_cell_pyvista(na, nb, nc)
        if self.visualization_settings.get('show_axes', True):
            self.plot_crystal_axes_pyvista()
        self.pv_widget.reset_camera()

    def plot_atoms_pyvista(self, na, nb, nc):
        """Plot atoms as spheres in the PyVista scene."""
        if not self.current_structure or not self.pymatgen_structure:
            print("No structure or pymatgen structure to plot")
            return
            
        print("\n=== Plotting Atoms ===")
        # Print debug info
        self.debug_structure_info()
        print(f"Plotting atoms with supercell {na}x{nb}x{nc}")
        atom_scale = self.visualization_settings.get('atom_scale', 0.3)
        transparency = self.visualization_settings.get('transparency', 0.3)
        color_scheme = self.visualization_settings.get('color_scheme', 'element')
        label_atoms = self.visualization_settings.get('label_atoms', False)
        
        lattice = self.pymatgen_structure.lattice
        
        # Get the sites to plot - use equivalent_sites if available, otherwise use original sites
        if 'equivalent_sites' in self.current_structure and self.current_structure['equivalent_sites']:
            print(f"Using {sum(len(sites) for sites in self.current_structure['equivalent_sites'].values())} equivalent sites for plotting")
            sites_to_plot = self.current_structure['equivalent_sites']
        else:
            print("No equivalent sites found, using original sites")
            sites_to_plot = {
                i: [{'frac_coords': site['frac_coords']}]
                for i, site in enumerate(self.current_structure['sites'])
            }
        
        # Track unique positions to avoid duplicates
        unique_positions = set()
        
        for site_idx, equiv_sites in sites_to_plot.items():
            element = self.current_structure['sites'][site_idx % len(self.current_structure['sites'])]['element']
            atomic_radius = self.current_structure['sites'][site_idx % len(self.current_structure['sites'])].get('radius', 0.5) * atom_scale
            color = self.get_element_color(element) if color_scheme == 'element' else 'white'
            
            for site in equiv_sites:
                base_frac = np.array(site.get('frac_coords', site.get('cart_coords')))
                
                # Apply supercell translations
                for i in range(na):
                    for j in range(nb):
                        for k in range(nc):
                            supercell_frac = base_frac + np.array([i, j, k])
                            # Wrap fractional coordinates back into [0,1)
                            supercell_frac = supercell_frac - np.floor(supercell_frac)
                            cart_coords = lattice.get_cartesian_coords(supercell_frac)
                            
                            # Use position as a key to avoid duplicates
                            pos_key = tuple(round(x, 6) for x in cart_coords)
                            if pos_key in unique_positions:
                                continue
                            unique_positions.add(pos_key)
                            
                            # Add the atom sphere
                            sphere = pv.Sphere(radius=atomic_radius, 
                                             center=cart_coords, 
                                             theta_resolution=32, 
                                             phi_resolution=32)
                            self.pv_widget.add_mesh(sphere, 
                                                  color=color, 
                                                  opacity=transparency, 
                                                  name=f"atom_{element}_{len(unique_positions)}")
                            
                            # Add label if enabled
                            if label_atoms:
                                self.pv_widget.add_point_labels(
                                    [cart_coords], 
                                    [element], 
                                    font_size=10, 
                                    point_color=color, 
                                    point_size=0, 
                                    render_points_as_spheres=False
                                )
        
        print(f"Plotted {len(unique_positions)} unique atom positions")

    def analyze_drawn_bonds(self):
        """Analyze and return information about the bonds currently drawn in the visualization."""
        if not hasattr(self, 'drawn_bonds') or not self.drawn_bonds:
            return {
                'total_bonds': 0,
                'bond_types': {},
                'avg_bond_length': 0.0,
                'min_bond_length': 0.0,
                'max_bond_length': 0.0
            }
        
        bond_lengths = []
        bond_types = {}
        
        for bond in self.drawn_bonds:
            # Calculate bond length
            site1 = self.current_structure['sites'][bond['atom1_idx']]
            site2 = self.current_structure['sites'][bond['atom2_idx']]
            cart1 = np.array(site1['cart_coords'])
            cart2 = np.array(site2['cart_coords'])
            length = np.linalg.norm(cart2 - cart1)
            bond_lengths.append(length)
            
            # Track bond types
            elem1 = site1['element']
            elem2 = site2['element']
            bond_type = f"{min(elem1, elem2)}-{max(elem1, elem2)}"
            if bond_type not in bond_types:
                bond_types[bond_type] = []
            bond_types[bond_type].append(length)
        
        # Calculate statistics
        if bond_lengths:
            return {
                'total_bonds': len(bond_lengths),
                'bond_types': {k: {
                    'count': len(v),
                    'avg_length': np.mean(v),
                    'min_length': min(v),
                    'max_length': max(v)
                } for k, v in bond_types.items()},
                'avg_bond_length': np.mean(bond_lengths),
                'min_bond_length': min(bond_lengths) if bond_lengths else 0.0,
                'max_bond_length': max(bond_lengths) if bond_lengths else 0.0
            }
        return {
            'total_bonds': 0,
            'bond_types': {},
            'avg_bond_length': 0.0,
            'min_bond_length': 0.0,
            'max_bond_length': 0.0
        }

    def plot_bonds_pyvista(self, na, nb, nc):
        """Plot bonds as cylinders in the PyVista scene."""
        if not self.current_structure or not self.pymatgen_structure or not self.bonds:
            return
            
        bond_radius = self.visualization_settings.get('bond_radius', 0.05)
        transparency = self.visualization_settings.get('transparency', 0.3)
        label_bonds = self.visualization_settings.get('label_bonds', False)
        
        # Track drawn bonds for analysis
        self.drawn_bonds = []
        
        for bond in self.bonds:
            if len(bond) < 2:  # Skip invalid bonds
                continue
                
            i, j = bond[0], bond[1]
            site1 = self.current_structure['sites'][i]
            site2 = self.current_structure['sites'][j]
            
            # Get cartesian coordinates
            cart1 = np.array(site1['cart_coords'])
            cart2 = np.array(site2['cart_coords'])
            
            # Calculate bond properties
            center = (cart1 + cart2) / 2
            direction = cart2 - cart1
            length = np.linalg.norm(direction)
            
            # Skip zero-length bonds
            if length < 1e-3:
                continue
                
            # Normalize direction vector
            direction = direction / length
            
            # Only draw bonds within reasonable length
            if 0.1 < length < 3.0:  # Reasonable bond length range in Angstroms
                # Create cylinder for the bond
                cyl = pv.Cylinder(
                    center=center,
                    direction=direction,
                    radius=bond_radius,
                    height=length,
                    resolution=12  # Reduced resolution for better performance
                )
                
                # Add to scene with light gray color and slight transparency
                self.pv_widget.add_mesh(
                    cyl, 
                    color='lightgray', 
                    opacity=min(0.7, transparency + 0.2),  # Slightly more opaque than atoms
                    name=f"bond_{i}_{j}",
                    smooth_shading=True
                )
                
                # Record bond information for analysis
                self.drawn_bonds.append({
                    'atom1_idx': i,
                    'atom2_idx': j,
                    'drawn_distance': length,
                    'original_distance': length,  # In this implementation, we don't have a separate original distance
                    'unit_cell': [0, 0, 0]  # Simplified for basic implementation
                })
                
                # Add bond label if enabled
                if label_bonds:
                    mid = (cart1 + cart2) / 2
                    self.pv_widget.add_point_labels(
                        [mid], 
                        [f"{site1['element']}-{site2['element']}: {length:.2f}Å"],
                        font_size=8,
                        shape_opacity=0.5
                    )

    def plot_unit_cell_pyvista(self, na, nb, nc):
        """Plot the unit cell edges as lines in the PyVista scene."""
        if not self.pymatgen_structure:
            return
            
        lattice = self.pymatgen_structure.lattice
        # Get the 8 corners of the unit cell
        corners = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ])
        cart_corners = np.array([lattice.get_cartesian_coords(c) for c in corners])
        # Define the 12 edges of the unit cell
        edges = [
            (0,1),(1,2),(2,3),(3,0), # bottom
            (4,5),(5,6),(6,7),(7,4), # top
            (0,4),(1,5),(2,6),(3,7)  # verticals
        ]
        for start, end in edges:
            pts = np.vstack([cart_corners[start], cart_corners[end]])
            self.pv_widget.add_lines(pts, color='black', width=2, name=f"cell_{start}_{end}")

    def plot_crystal_axes_pyvista(self):
        """Plot crystal axes as colored arrows in the PyVista scene."""
        if not self.pymatgen_structure:
            return
        lattice = self.pymatgen_structure.lattice
        origin = lattice.get_cartesian_coords([0,0,0])
        a_vec = lattice.get_cartesian_coords([1,0,0]) - origin
        b_vec = lattice.get_cartesian_coords([0,1,0]) - origin
        c_vec = lattice.get_cartesian_coords([0,0,1]) - origin
        arrow_len = min(np.linalg.norm(a_vec), np.linalg.norm(b_vec), np.linalg.norm(c_vec)) * 0.8
        self.pv_widget.add_arrows(origin, a_vec/np.linalg.norm(a_vec)*arrow_len, color='red', name='a_axis')
        self.pv_widget.add_arrows(origin, b_vec/np.linalg.norm(b_vec)*arrow_len, color='green', name='b_axis')
        self.pv_widget.add_arrows(origin, c_vec/np.linalg.norm(c_vec)*arrow_len, color='blue', name='c_axis')

    # ... (rest of the code remains the same)
    
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
        """Get a darker, more vibrant color for elements using CPK color scheme."""
        element_colors = {
            # CPK colors with darker shades
            'H': '#FFFFFF',  # White
            'C': '#909090',  # Darker gray
            'N': '#4141FF',  # Darker blue
            'O': '#FF0D0D',  # Darker red
            'F': '#90E050',  # Darker green
            'P': '#FF8000',  # Darker orange
            'S': '#FFFF30',  # Darker yellow
            'Cl': '#1FF01F', # Darker green
            'Br': '#A52A2A', # Brown
            'I': '#940094',  # Dark purple
            'He': '#D9FFFF', # Cyan
            'Ne': '#B3E3F5', # Neon blue
            'Ar': '#80D1E3', # Darker cyan
            'Xe': '#429EB0', # Dark teal
            'Na': '#AB5CF2', # Darker blue-violet
            'K': '#8F40D3',  # Darker violet
            'Mg': '#8AFF00', # Darker green
            'Ca': '#3DFF00', # Darker green
            'Fe': '#E06633', # Darker rust
            'Ti': '#BFC2C7', # Darker silver
            'Al': '#AFAFAF', # Darker gray
            'Si': '#F0C8A0', # Darker tan
            'Cu': '#C88033', # Darker copper
            'Ag': '#C0C0C0', # Silver
            'Au': '#FFD700', # Gold
            'Zn': '#7D80B0', # Darker bluish-gray
            'Pb': '#575961', # Darker gray
            'Hg': '#B8B8D0', # Darker silver-blue
            'As': '#BD80E3', # Darker purple
            'Se': '#FFA100', # Darker orange
            'Li': '#CC80FF', # Darker violet
            'Be': '#C2FF00', # Darker green
            'B': '#FFB5B5',  # Darker pink
        }
        return element_colors.get(element, '#A0A0A0')  # Default to medium gray
    
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
    
    def show_unit_cell_debug_info(self):
        """Show debug information about the unit cell."""
        if not self.current_structure:
            QMessageBox.warning(self, "Warning", "No structure loaded.")
            return
        
        # Create a dialog to display unit cell information
        dialog = QDialog(self)
        dialog.setWindowTitle("Unit Cell Debug Information")
        dialog.setModal(True)
        dialog.resize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        # Add text to display unit cell information
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setPlainText(f"Formula: {self.current_structure['formula']}\n"
                               f"Space Group: {self.current_structure['space_group']}\n"
                               f"Crystal System: {self.current_structure['crystal_system']}\n"
                               f"Atoms: {self.current_structure['num_atoms']}\n"
                               f"Elements: {', '.join(self.current_structure['elements'])}\n"
                               f"Wyckoff Positions: {len(self.current_structure['wyckoff_positions'])}\n"
                               f"Symmetry Operations: {len(self.current_structure['symmetry_operations'])}\n"
                               f"Equivalent Sites: {len(self.current_structure['equivalent_sites'])}\n"
                               f"Lattice Parameters:\na = {self.current_structure['lattice_params']['a']:.4f} Å\n"
                               f"b = {self.current_structure['lattice_params']['b']:.4f} Å\n"
                               f"c = {self.current_structure['lattice_params']['c']:.4f} Å\n"
                               f"α = {self.current_structure['lattice_params']['alpha']:.2f}°\n"
                               f"β = {self.current_structure['lattice_params']['beta']:.2f}°\n"
                               f"γ = {self.current_structure['lattice_params']['gamma']:.2f}°\n"
                               f"Volume = {self.current_structure['lattice_params']['volume']:.3f} Å³\n")
        layout.addWidget(info_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()


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