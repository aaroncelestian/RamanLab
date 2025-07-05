"""
Crystal Structure UI Module

This module provides comprehensive crystal structure analysis UI components including:
- CIF file loading and parsing
- Database-driven structure selection
- 3D visualization with interactive controls
- Bond analysis and coordination geometry
- Symmetry analysis and unit cell generation
- Integration with Raman tensor calculations

Professional PySide6 implementation with enhanced features.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSlider, QTextEdit, QTableWidget, QTableWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QFormLayout, QFileDialog,
    QDialog, QDialogButtonBox, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Configure matplotlib for smaller toolbar
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'toolbar2'

# Make toolbar more compact
import matplotlib.pyplot as plt

try:
    from core.crystal_analyzer import CrystalAnalyzer
    from core.database_manager import DatabaseManager
    from parsers.cif_parser import CIFParser
    # Try importing pymatgen for advanced CIF parsing
    try:
        from pymatgen.core import Structure
        PYMATGEN_AVAILABLE = True
        print("✓ pymatgen available for crystal structure visualization")
    except ImportError:
        PYMATGEN_AVAILABLE = False
        print("ℹ️  Pymatgen not available - using simplified CIF parsing")
except ImportError:
    print("ℹ️  Enhanced crystal structure modules not available - using basic functionality")


class MineralSelectionDialog(QDialog):
    """Dialog for selecting minerals from database."""
    
    def __init__(self, parent=None, mineral_list=None):
        super().__init__(parent)
        self.mineral_list = mineral_list or []
        self.selected_mineral = None
        
        self.setWindowTitle("Select Mineral Structure")
        self.setMinimumSize(400, 500)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        
        # Instructions
        instruction_label = QLabel("Select mineral for structure analysis:")
        instruction_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(instruction_label)
        
        # Search box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_entry = QComboBox()
        self.search_entry.setEditable(True)
        self.search_entry.addItems(self.mineral_list)
        self.search_entry.currentTextChanged.connect(self.filter_minerals)
        search_layout.addWidget(self.search_entry)
        layout.addLayout(search_layout)
        
        # Mineral list
        self.mineral_listbox = QListWidget()
        self.mineral_listbox.addItems(self.mineral_list)
        self.mineral_listbox.itemDoubleClicked.connect(self.on_mineral_selected)
        layout.addWidget(self.mineral_listbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        load_btn = QPushButton("Load Structure")
        load_btn.clicked.connect(self.on_mineral_selected)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(load_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
    def filter_minerals(self, search_term):
        """Filter mineral list based on search term."""
        self.mineral_listbox.clear()
        search_term = search_term.lower()
        
        for mineral in self.mineral_list:
            if search_term in mineral.lower():
                self.mineral_listbox.addItem(mineral)
                
    def on_mineral_selected(self):
        """Handle mineral selection."""
        current_item = self.mineral_listbox.currentItem()
        if current_item:
            self.selected_mineral = current_item.text()
            self.accept()
            
    def get_selected_mineral(self):
        """Get the selected mineral name."""
        return self.selected_mineral


class CrystalStructureWidget(QWidget):
    """
    Professional crystal structure analysis widget.
    
    Features:
    - CIF file loading with pymatgen integration
    - Database-driven mineral structure selection
    - Interactive 3D visualization
    - Bond length and angle analysis
    - Symmetry operations and space group analysis
    - Unit cell parameter display and editing
    - Integration with Raman tensor calculations
    """
    
    # Signals
    structure_loaded = Signal(dict)
    analysis_completed = Signal(dict)
    structure_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Structure data
        self.crystal_structure = None
        self.atomic_positions = []
        self.bond_analysis = {}
        self.symmetry_operations = []
        
        # Visualization settings
        self.unit_cell_range = 2.0
        self.atom_size = 0.3
        self.show_bonds = True
        self.show_unit_cell = True
        self.show_labels = False
        
        # Core modules
        self.crystal_analyzer = None
        self.database_manager = None
        self.cif_parser = None
        
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
        """Create the control panel."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # Add control tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Loading tab
        tabs.addTab(self.create_loading_tab(), "Load")
        
        # Structure info tab
        tabs.addTab(self.create_structure_tab(), "Structure")
        
        # Analysis tab
        tabs.addTab(self.create_analysis_tab(), "Analysis")
        
        # Visualization tab
        tabs.addTab(self.create_visualization_tab(), "Display")
        
        return panel
        
    def create_loading_tab(self):
        """Create structure loading controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File loading group
        file_group = QGroupBox("File Loading")
        file_layout = QVBoxLayout(file_group)
        
        load_cif_btn = QPushButton("Load CIF File")
        load_cif_btn.clicked.connect(self.load_cif_file)
        file_layout.addWidget(load_cif_btn)
        
        self.cif_status_label = QLabel("No CIF file loaded")
        self.cif_status_label.setStyleSheet("color: #666; font-size: 10px;")
        file_layout.addWidget(self.cif_status_label)
        
        layout.addWidget(file_group)
        
        # Database group
        database_group = QGroupBox("Database Selection")
        db_layout = QVBoxLayout(database_group)
        
        load_from_db_btn = QPushButton("Load from Database")
        load_from_db_btn.clicked.connect(self.load_from_database)
        db_layout.addWidget(load_from_db_btn)
        
        self.db_status_label = QLabel("No database structure loaded")
        self.db_status_label.setStyleSheet("color: #666; font-size: 10px;")
        db_layout.addWidget(self.db_status_label)
        
        layout.addWidget(database_group)
        
        # Current structure status
        status_group = QGroupBox("Structure Status")
        status_layout = QVBoxLayout(status_group)
        
        self.structure_status_label = QLabel("No structure loaded")
        self.structure_status_label.setStyleSheet("font-weight: bold; color: #333;")
        status_layout.addWidget(self.structure_status_label)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        return tab
        
    def create_structure_tab(self):
        """Create structure information display."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Structure info display
        info_group = QGroupBox("Structure Information")
        info_layout = QVBoxLayout(info_group)
        
        self.structure_info_text = QTextEdit()
        self.structure_info_text.setReadOnly(True)
        self.structure_info_text.setMaximumHeight(200)
        info_layout.addWidget(self.structure_info_text)
        
        layout.addWidget(info_group)
        
        # Lattice parameters
        lattice_group = QGroupBox("Lattice Parameters")
        lattice_layout = QFormLayout(lattice_group)
        
        self.lattice_a_label = QLabel("--")
        self.lattice_b_label = QLabel("--")
        self.lattice_c_label = QLabel("--")
        self.lattice_alpha_label = QLabel("--")
        self.lattice_beta_label = QLabel("--")
        self.lattice_gamma_label = QLabel("--")
        self.lattice_volume_label = QLabel("--")
        
        lattice_layout.addRow("a (Å):", self.lattice_a_label)
        lattice_layout.addRow("b (Å):", self.lattice_b_label)
        lattice_layout.addRow("c (Å):", self.lattice_c_label)
        lattice_layout.addRow("α (°):", self.lattice_alpha_label)
        lattice_layout.addRow("β (°):", self.lattice_beta_label)
        lattice_layout.addRow("γ (°):", self.lattice_gamma_label)
        lattice_layout.addRow("Volume (Å³):", self.lattice_volume_label)
        
        layout.addWidget(lattice_group)
        
        # Space group info
        sg_group = QGroupBox("Space Group")
        sg_layout = QFormLayout(sg_group)
        
        self.space_group_label = QLabel("--")
        self.crystal_system_label = QLabel("--")
        self.point_group_label = QLabel("--")
        
        sg_layout.addRow("Space Group:", self.space_group_label)
        sg_layout.addRow("Crystal System:", self.crystal_system_label)
        sg_layout.addRow("Point Group:", self.point_group_label)
        
        layout.addWidget(sg_group)
        
        layout.addStretch()
        return tab
        
    def create_analysis_tab(self):
        """Create analysis controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Unit cell generation
        unit_cell_group = QGroupBox("Unit Cell Generation")
        unit_cell_layout = QVBoxLayout(unit_cell_group)
        
        generate_btn = QPushButton("Generate Unit Cell")
        generate_btn.clicked.connect(self.generate_unit_cell)
        unit_cell_layout.addWidget(generate_btn)
        
        self.atom_count_label = QLabel("Atoms: --")
        unit_cell_layout.addWidget(self.atom_count_label)
        
        layout.addWidget(unit_cell_group)
        
        # Bond analysis
        bond_group = QGroupBox("Bond Analysis")
        bond_layout = QVBoxLayout(bond_group)
        
        calc_bonds_btn = QPushButton("Calculate Bond Lengths")
        calc_bonds_btn.clicked.connect(self.calculate_bond_lengths)
        bond_layout.addWidget(calc_bonds_btn)
        
        coord_analysis_btn = QPushButton("Coordination Analysis")
        coord_analysis_btn.clicked.connect(self.analyze_coordination)
        bond_layout.addWidget(coord_analysis_btn)
        
        self.bond_count_label = QLabel("Bonds: --")
        bond_layout.addWidget(self.bond_count_label)
        
        layout.addWidget(bond_group)
        
        # Advanced analysis
        advanced_group = QGroupBox("Advanced Analysis")
        advanced_layout = QVBoxLayout(advanced_group)
        
        phonon_btn = QPushButton("Calculate Phonon Modes")
        phonon_btn.clicked.connect(self.calculate_phonon_modes)
        advanced_layout.addWidget(phonon_btn)
        
        raman_tensor_btn = QPushButton("Calculate Raman Tensors")
        raman_tensor_btn.clicked.connect(self.calculate_raman_tensors)
        advanced_layout.addWidget(raman_tensor_btn)
        
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        return tab
        
    def create_visualization_tab(self):
        """Create visualization controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_bonds_check = QCheckBox("Show bonds")
        self.show_bonds_check.setChecked(True)
        self.show_bonds_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_bonds_check)
        
        self.show_unit_cell_check = QCheckBox("Show unit cell edges")
        self.show_unit_cell_check.setChecked(True)
        self.show_unit_cell_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_unit_cell_check)
        
        self.show_labels_check = QCheckBox("Show atom labels")
        self.show_labels_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_labels_check)
        
        layout.addWidget(display_group)
        
        # Size controls
        size_group = QGroupBox("Size Controls")
        size_layout = QVBoxLayout(size_group)
        
        # Unit cell range
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Cell range:"))
        self.cell_range_slider = QSlider(Qt.Orientation.Horizontal)
        self.cell_range_slider.setRange(1, 5)
        self.cell_range_slider.setValue(2)
        self.cell_range_slider.valueChanged.connect(self.on_range_changed)
        range_layout.addWidget(self.cell_range_slider)
        self.cell_range_label = QLabel("2")
        range_layout.addWidget(self.cell_range_label)
        size_layout.addLayout(range_layout)
        
        # Atom size
        atom_layout = QHBoxLayout()
        atom_layout.addWidget(QLabel("Atom size:"))
        self.atom_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.atom_size_slider.setRange(1, 10)
        self.atom_size_slider.setValue(3)
        self.atom_size_slider.valueChanged.connect(self.on_atom_size_changed)
        atom_layout.addWidget(self.atom_size_slider)
        self.atom_size_label = QLabel("0.3")
        atom_layout.addWidget(self.atom_size_label)
        size_layout.addLayout(atom_layout)
        
        layout.addWidget(size_group)
        
        # Color scheme
        color_group = QGroupBox("Color Scheme")
        color_layout = QVBoxLayout(color_group)
        
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["CPK", "Element", "Custom"])
        self.color_scheme_combo.currentTextChanged.connect(self.on_display_changed)
        color_layout.addWidget(self.color_scheme_combo)
        
        layout.addWidget(color_group)
        
        # Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        export_structure_btn = QPushButton("Export Structure")
        export_structure_btn.clicked.connect(self.export_structure)
        export_layout.addWidget(export_structure_btn)
        
        export_image_btn = QPushButton("Export Image")
        export_image_btn.clicked.connect(self.export_image)
        export_layout.addWidget(export_image_btn)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        return tab
        
    def create_visualization_panel(self):
        """Create the 3D visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create matplotlib figure for 3D plotting
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Create 3D subplot
        self.ax_3d = self.figure.add_subplot(111, projection='3d')
        self.initialize_3d_plot()
        
        return panel
        
    def initialize_modules(self):
        """Initialize core modules."""
        try:
            self.crystal_analyzer = CrystalAnalyzer()
            self.database_manager = DatabaseManager()
            self.cif_parser = CIFParser()
        except Exception as e:
            print(f"Warning: Could not initialize crystal structure modules: {e}")
            
    def initialize_3d_plot(self):
        """Initialize the 3D plot."""
        self.ax_3d.clear()
        self.ax_3d.set_xlabel('X (Å)')
        self.ax_3d.set_ylabel('Y (Å)')
        self.ax_3d.set_zlabel('Z (Å)')
        self.ax_3d.set_title('Crystal Structure Visualization')
        
        # Add placeholder text
        self.ax_3d.text(0.5, 0.5, 0.5, 'Load a crystal structure to begin visualization',
                       transform=self.ax_3d.transAxes, ha='center', va='center',
                       fontsize=12, alpha=0.6)
        
        self.canvas.draw()
        
    def load_cif_file(self):
        """Load crystal structure from CIF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CIF File", "",
            "CIF files (*.cif);;All files (*.*)"
        )
        
        if file_path:
            try:
                if PYMATGEN_AVAILABLE:
                    # Use pymatgen for professional CIF parsing
                    self.crystal_structure = self.parse_cif_with_pymatgen(file_path)
                else:
                    # Fallback to simplified parser
                    self.crystal_structure = self.parse_cif_file(file_path)
                
                if self.crystal_structure:
                    self.cif_status_label.setText(f"✓ {self.crystal_structure.get('name', 'CIF File')}")
                    self.cif_status_label.setStyleSheet("color: green; font-size: 10px;")
                    
                    self.update_structure_displays()
                    self.structure_loaded.emit(self.crystal_structure)
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CIF file: {str(e)}")
                
    def parse_cif_with_pymatgen(self, file_path):
        """Parse CIF file using pymatgen."""
        try:
            structure = Structure.from_file(file_path)
            
            # Convert to our format
            crystal_data = {
                'name': file_path.split('/')[-1].replace('.cif', ''),
                'lattice_parameters': {
                    'a': structure.lattice.a,
                    'b': structure.lattice.b, 
                    'c': structure.lattice.c,
                    'alpha': structure.lattice.alpha,
                    'beta': structure.lattice.beta,
                    'gamma': structure.lattice.gamma,
                    'volume': structure.lattice.volume
                },
                'space_group': structure.get_space_group_info()[0],
                'space_group_number': structure.get_space_group_info()[1],
                'crystal_system': structure.get_crystal_system(),
                'atoms': []
            }
            
            # Extract atomic positions
            for site in structure:
                atom_data = {
                    'element': site.specie.symbol,
                    'position': site.frac_coords.tolist(),
                    'cartesian_position': site.coords.tolist()
                }
                crystal_data['atoms'].append(atom_data)
                
            return crystal_data
            
        except Exception as e:
            raise Exception(f"Pymatgen parsing failed: {str(e)}")
            
    def parse_cif_file(self, file_path):
        """Parse CIF file using simplified parser."""
        if self.cif_parser:
            return self.cif_parser.parse_file(file_path)
        else:
            raise Exception("CIF parser not available")
            
    def load_from_database(self):
        """Load crystal structure from mineral database."""
        try:
            if self.database_manager:
                # Get mineral list
                mineral_list = self.database_manager.get_mineral_list()
                
                if mineral_list:
                    dialog = MineralSelectionDialog(self, mineral_list)
                    if dialog.exec() == QDialog.DialogCode.Accepted:
                        mineral_name = dialog.get_selected_mineral()
                        if mineral_name:
                            self.load_structure_from_database(mineral_name)
                else:
                    QMessageBox.warning(self, "No Data", "No mineral database available")
            else:
                QMessageBox.warning(self, "Error", "Database manager not available")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load from database: {str(e)}")
            
    def load_structure_from_database(self, mineral_name):
        """Load specific structure from database."""
        try:
            if self.database_manager:
                structure_data = self.database_manager.get_mineral_structure(mineral_name)
                
                if structure_data:
                    self.crystal_structure = structure_data
                    self.db_status_label.setText(f"✓ {mineral_name}")
                    self.db_status_label.setStyleSheet("color: blue; font-size: 10px;")
                    
                    self.update_structure_displays()
                    self.structure_loaded.emit(structure_data)
                else:
                    QMessageBox.warning(self, "Error", f"Structure data not found for {mineral_name}")
            else:
                QMessageBox.warning(self, "Error", "Database manager not available")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load structure: {str(e)}")
            
    def update_structure_displays(self):
        """Update all structure information displays."""
        if not self.crystal_structure:
            return
            
        # Update status
        name = self.crystal_structure.get('name', 'Unknown')
        atom_count = len(self.crystal_structure.get('atoms', []))
        self.structure_status_label.setText(f"✓ {name} ({atom_count} atoms)")
        self.structure_status_label.setStyleSheet("color: green; font-weight: bold;")
        
        # Update lattice parameters
        lattice = self.crystal_structure.get('lattice_parameters', {})
        self.lattice_a_label.setText(f"{lattice.get('a', 0):.4f}")
        self.lattice_b_label.setText(f"{lattice.get('b', 0):.4f}")
        self.lattice_c_label.setText(f"{lattice.get('c', 0):.4f}")
        self.lattice_alpha_label.setText(f"{lattice.get('alpha', 0):.2f}")
        self.lattice_beta_label.setText(f"{lattice.get('beta', 0):.2f}")
        self.lattice_gamma_label.setText(f"{lattice.get('gamma', 0):.2f}")
        self.lattice_volume_label.setText(f"{lattice.get('volume', 0):.2f}")
        
        # Update space group info
        self.space_group_label.setText(str(self.crystal_structure.get('space_group', 'Unknown')))
        self.crystal_system_label.setText(str(self.crystal_structure.get('crystal_system', 'Unknown')))
        self.point_group_label.setText(str(self.crystal_structure.get('point_group', 'Unknown')))
        
        # Update structure info text
        self.update_structure_info_text()
        
        # Update 3D visualization
        self.update_3d_plot()
        
    def update_structure_info_text(self):
        """Update the structure information text display."""
        if not self.crystal_structure:
            return
            
        info_text = "CRYSTAL STRUCTURE INFORMATION\n"
        info_text += "=" * 35 + "\n\n"
        
        info_text += f"Name: {self.crystal_structure.get('name', 'Unknown')}\n"
        info_text += f"Space Group: {self.crystal_structure.get('space_group', 'Unknown')}\n"
        
        if 'space_group_number' in self.crystal_structure:
            info_text += f"Space Group Number: {self.crystal_structure['space_group_number']}\n"
        if 'crystal_system' in self.crystal_structure:
            info_text += f"Crystal System: {self.crystal_structure['crystal_system']}\n"
        if 'point_group' in self.crystal_structure:
            info_text += f"Point Group: {self.crystal_structure['point_group']}\n"
            
        info_text += "\nAtomic Composition:\n"
        info_text += "-" * 20 + "\n"
        
        # Count elements
        element_counts = {}
        for atom in self.crystal_structure.get('atoms', []):
            element = atom.get('element', 'Unknown')
            element_counts[element] = element_counts.get(element, 0) + 1
            
        for element, count in sorted(element_counts.items()):
            info_text += f"{element}: {count}\n"
            
        self.structure_info_text.setPlainText(info_text)
        
    def generate_unit_cell(self):
        """Generate all atomic positions in the unit cell."""
        if not self.crystal_structure:
            QMessageBox.warning(self, "No Structure", "Please load a crystal structure first")
            return
            
        try:
            if self.crystal_analyzer:
                self.atomic_positions = self.crystal_analyzer.generate_unit_cell(self.crystal_structure)
                self.atom_count_label.setText(f"Atoms: {len(self.atomic_positions)}")
                self.update_3d_plot()
            else:
                QMessageBox.warning(self, "Error", "Crystal analyzer not available")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate unit cell: {str(e)}")
            
    def calculate_bond_lengths(self):
        """Calculate bond lengths in the structure."""
        if not self.crystal_structure:
            QMessageBox.warning(self, "No Structure", "Please load a crystal structure first")
            return
            
        try:
            if self.crystal_analyzer:
                self.bond_analysis = self.crystal_analyzer.analyze_bonds(self.crystal_structure)
                bond_count = len(self.bond_analysis.get('bonds', []))
                self.bond_count_label.setText(f"Bonds: {bond_count}")
                self.update_3d_plot()
            else:
                QMessageBox.warning(self, "Error", "Crystal analyzer not available")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate bonds: {str(e)}")
            
    def analyze_coordination(self):
        """Analyze coordination geometry."""
        if not self.crystal_structure:
            QMessageBox.warning(self, "No Structure", "Please load a crystal structure first")
            return
            
        QMessageBox.information(self, "Info", "Coordination analysis feature coming soon")
        
    def calculate_phonon_modes(self):
        """Calculate phonon modes (placeholder)."""
        QMessageBox.information(self, "Phonon Modes", 
                              "Phonon mode calculation requires quantum mechanical methods.\n"
                              "This feature will be implemented in future versions.")
                              
    def calculate_raman_tensors(self):
        """Calculate Raman tensors (placeholder)."""
        QMessageBox.information(self, "Raman Tensors",
                              "Raman tensor calculation requires:\n"
                              "1. Phonon modes\n"
                              "2. Polarizability derivatives\n"
                              "3. Symmetry analysis\n\n"
                              "This feature will be implemented in future versions.")
                              
    def on_display_changed(self):
        """Handle display option changes."""
        self.show_bonds = self.show_bonds_check.isChecked()
        self.show_unit_cell = self.show_unit_cell_check.isChecked()
        self.show_labels = self.show_labels_check.isChecked()
        self.update_3d_plot()
        
    def on_range_changed(self, value):
        """Handle unit cell range change."""
        self.unit_cell_range = float(value)
        self.cell_range_label.setText(str(value))
        self.update_3d_plot()
        
    def on_atom_size_changed(self, value):
        """Handle atom size change."""
        self.atom_size = value / 10.0
        self.atom_size_label.setText(f"{self.atom_size:.1f}")
        self.update_3d_plot()
        
    def update_3d_plot(self):
        """Update the 3D structure visualization."""
        self.ax_3d.clear()
        
        if not self.crystal_structure:
            self.initialize_3d_plot()
            return
            
        try:
            # Plot atoms
            atoms = self.crystal_structure.get('atoms', [])
            if atoms:
                # Get element colors
                element_colors = self.get_element_colors()
                
                for atom in atoms:
                    element = atom.get('element', 'C')
                    pos = atom.get('cartesian_position', [0, 0, 0])
                    
                    color = element_colors.get(element, 'gray')
                    self.ax_3d.scatter(pos[0], pos[1], pos[2], 
                                     c=color, s=self.atom_size*1000, alpha=0.8)
                    
                    if self.show_labels:
                        self.ax_3d.text(pos[0], pos[1], pos[2], element,
                                       fontsize=8, ha='center', va='center')
                        
            # Plot bonds if calculated
            if self.show_bonds and self.bond_analysis:
                bonds = self.bond_analysis.get('bonds', [])
                for bond in bonds:
                    atom1_pos = bond.get('atom1_position', [0, 0, 0])
                    atom2_pos = bond.get('atom2_position', [0, 0, 0])
                    
                    self.ax_3d.plot([atom1_pos[0], atom2_pos[0]],
                                   [atom1_pos[1], atom2_pos[1]], 
                                   [atom1_pos[2], atom2_pos[2]],
                                   'k-', alpha=0.5, linewidth=1)
                                   
            # Plot unit cell edges if requested
            if self.show_unit_cell:
                self.plot_unit_cell_edges()
                
            self.ax_3d.set_xlabel('X (Å)')
            self.ax_3d.set_ylabel('Y (Å)')
            self.ax_3d.set_zlabel('Z (Å)')
            self.ax_3d.set_title(f"Crystal Structure: {self.crystal_structure.get('name', 'Unknown')}")
            
        except Exception as e:
            print(f"Error updating 3D plot: {e}")
            
        self.canvas.draw()
        
    def get_element_colors(self):
        """Get color mapping for elements."""
        # CPK colors for common elements
        colors = {
            'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red',
            'F': 'lightgreen', 'Ne': 'cyan', 'Na': 'blue', 'Mg': 'darkgreen',
            'Al': 'gray', 'Si': 'tan', 'P': 'orange', 'S': 'yellow',
            'Cl': 'green', 'Ar': 'cyan', 'K': 'purple', 'Ca': 'darkgreen',
            'Fe': 'orange', 'Cu': 'brown', 'Zn': 'gray'
        }
        return colors
        
    def plot_unit_cell_edges(self):
        """Plot unit cell edges."""
        # This would require lattice vectors to be properly calculated
        # Placeholder implementation
        pass
        
    def export_structure(self):
        """Export structure data."""
        if not self.crystal_structure:
            QMessageBox.warning(self, "No Data", "No structure to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Structure", "structure.json",
            "JSON files (*.json);;CIF files (*.cif)"
        )
        
        if file_path:
            try:
                import json
                with open(file_path, 'w') as f:
                    json.dump(self.crystal_structure, f, indent=2)
                QMessageBox.information(self, "Success", f"Structure exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
                
    def export_image(self):
        """Export 3D visualization image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "structure.png",
            "PNG files (*.png);;PDF files (*.pdf)"
        )
        
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Image exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export image: {str(e)}")
                
    def set_crystal_structure(self, structure_data):
        """Set crystal structure programmatically."""
        self.crystal_structure = structure_data
        self.update_structure_displays()
        
    def get_crystal_structure(self):
        """Get current crystal structure."""
        return self.crystal_structure
        
    def get_bond_analysis(self):
        """Get bond analysis results."""
        return self.bond_analysis 