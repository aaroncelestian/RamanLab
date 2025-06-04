import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import pickle
import time
from datetime import datetime
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit, minimize
import matplotlib.patches as patches

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

class RamanPolarizationAnalyzer:
    def __init__(self, root):
        """Initialize the Raman Polarization Analyzer
        
        Parameters:
        -----------
        root : tkinter.Tk
            The root window
        """
        self.root = root
        root.title("Raman Polarization Analyzer")
        root.geometry("1200x800")
        
        # Initialize variables
        self.mineral_database = None
        self.current_spectrum = None
        self.original_spectrum = None  # Store original unprocessed spectrum
        self.imported_spectrum = None
        self.mineral_list = []
        
        # Peak fitting variables
        self.selected_peaks = []  # List of user-selected peak positions
        self.fitted_peaks = []    # List of fitted peak parameters
        self.peak_assignments = {}  # Dictionary mapping peak positions to vibrational modes
        self.frequency_shifts = {}  # Dictionary of calculated frequency shifts
        self.peak_selection_mode = False
        
        # Polarization analysis variables
        self.polarization_data = {}  # Store multiple polarization measurements
        self.raman_tensors = {}      # Store calculated Raman tensors for each peak
        self.depolarization_ratios = {}  # Store depolarization ratios
        self.angular_data = {}       # Store angular dependence data
        self.current_polarization_config = None
        
        # Cross-tab integration variables
        self.selected_reference_mineral = None  # Currently selected reference mineral
        
        # Raman tensor variables
        self.calculated_raman_tensors = {}
        self.tensor_analysis_results = {}
        
        # Orientation optimization variables
        self.orientation_results = {}
        self.optimization_parameters = {}
        self.stage_results = {'stage1': None, 'stage2': None, 'stage3': None}
        self.optimized_orientation = None  # Store optimized orientation matrix
        
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
        
        # Create the main layout
        self.create_gui()
        
        # Update mineral lists in UI after GUI is created
        self.update_mineral_lists()
    
    def update_mineral_lists(self):
        """Update mineral lists in all relevant UI components."""
        if hasattr(self, 'reference_combo') and self.mineral_list:
            self.reference_combo['values'] = self.mineral_list
    
    def update_reference_mineral_selections(self):
        """Update reference mineral selections across all tabs."""
        if self.selected_reference_mineral:
            # Update Peak Fitting tab reference mineral
            if hasattr(self, 'reference_mineral_var'):
                self.reference_mineral_var.set(self.selected_reference_mineral)
                
            # Update status label in Peak Fitting tab
            if hasattr(self, 'reference_status_label'):
                self.reference_status_label.config(
                    text=f"Auto-selected from Spectrum Analysis tab",
                    foreground="green"
                )
            
            # Update Polarization tab crystal system if we can infer it from the mineral
            if hasattr(self, 'crystal_system_var'):
                inferred_system = self.infer_crystal_system(self.selected_reference_mineral)
                if inferred_system != "Unknown":
                    self.crystal_system_var.set(inferred_system)
                    
                    # Update status label in Polarization tab
                    if hasattr(self, 'pol_reference_status_label'):
                        self.pol_reference_status_label.config(
                            text=f"Crystal system inferred from {self.selected_reference_mineral}",
                            foreground="green"
                        )
                else:
                    # Show that we have a reference mineral but couldn't infer crystal system
                    if hasattr(self, 'pol_reference_status_label'):
                        self.pol_reference_status_label.config(
                            text=f"Reference: {self.selected_reference_mineral} (system unknown)",
                            foreground="orange"
                        )
    
    def infer_crystal_system(self, mineral_name):
        """Infer crystal system from mineral name or database information."""
        # This is a simplified implementation - in practice, you'd have a more comprehensive database
        crystal_systems = {
            # Common cubic minerals
            'HALITE': 'Cubic', 'FLUORITE': 'Cubic', 'PYRITE': 'Cubic', 'GALENA': 'Cubic',
            'MAGNETITE': 'Cubic', 'SPINEL': 'Cubic', 'GARNET': 'Cubic', 'DIAMOND': 'Cubic',
            
            # Common tetragonal minerals
            'ZIRCON': 'Tetragonal', 'RUTILE': 'Tetragonal', 'CASSITERITE': 'Tetragonal',
            'ANATASE': 'Tetragonal', 'SCHEELITE': 'Tetragonal',
            
            # Common hexagonal minerals
            'QUARTZ': 'Hexagonal', 'CALCITE': 'Hexagonal', 'HEMATITE': 'Hexagonal',
            'CORUNDUM': 'Hexagonal', 'APATITE': 'Hexagonal', 'GRAPHITE': 'Hexagonal',
            
            # Common orthorhombic minerals
            'OLIVINE': 'Orthorhombic', 'ARAGONITE': 'Orthorhombic', 'BARITE': 'Orthorhombic',
            'CELESTITE': 'Orthorhombic', 'TOPAZ': 'Orthorhombic',
            
            # Common monoclinic minerals
            'GYPSUM': 'Monoclinic', 'MICA': 'Monoclinic', 'PYROXENE': 'Monoclinic',
            'AMPHIBOLE': 'Monoclinic', 'FELDSPAR': 'Monoclinic',
            
            # Common triclinic minerals
            'PLAGIOCLASE': 'Triclinic', 'KAOLINITE': 'Triclinic'
        }
        
        # Check if mineral name (uppercase) is in our database
        mineral_upper = mineral_name.upper()
        
        # Direct lookup
        if mineral_upper in crystal_systems:
            return crystal_systems[mineral_upper]
        
        # Partial matching for common mineral groups
        for mineral_key, system in crystal_systems.items():
            if mineral_key in mineral_upper or mineral_upper in mineral_key:
                return system
        
        # Check mineral database for additional information if available
        if hasattr(self, 'mineral_database') and mineral_name in self.mineral_database:
            mineral_data = self.mineral_database[mineral_name]
            if 'crystal_system' in mineral_data:
                return mineral_data['crystal_system']
            if 'space_group' in mineral_data:
                # Infer from space group if available
                space_group = mineral_data['space_group']
                return self.infer_system_from_space_group(space_group)
        
        return "Unknown"
    
    def infer_system_from_space_group(self, space_group):
        """Infer crystal system from space group number or symbol."""
        # This is a simplified mapping - full implementation would use complete space group tables
        if isinstance(space_group, (int, float)):
            space_group_num = int(space_group)
            if 1 <= space_group_num <= 2:
                return "Triclinic"
            elif 3 <= space_group_num <= 15:
                return "Monoclinic"
            elif 16 <= space_group_num <= 74:
                return "Orthorhombic"
            elif 75 <= space_group_num <= 142:
                return "Tetragonal"
            elif 143 <= space_group_num <= 167:
                return "Trigonal"
            elif 168 <= space_group_num <= 194:
                return "Hexagonal"
            elif 195 <= space_group_num <= 230:
                return "Cubic"
        
        return "Unknown"
    
    def on_reference_mineral_changed(self, event=None):
        """Handle manual changes to the reference mineral selection."""
        new_mineral = self.reference_mineral_var.get()
        if new_mineral and new_mineral != self.selected_reference_mineral:
            # Update the global reference mineral
            self.selected_reference_mineral = new_mineral
            
            # Update status label
            if hasattr(self, 'reference_status_label'):
                self.reference_status_label.config(
                    text="Manually selected",
                    foreground="blue"
                )
            
            # Update Polarization tab crystal system
            if hasattr(self, 'crystal_system_var'):
                inferred_system = self.infer_crystal_system(new_mineral)
                if inferred_system != "Unknown":
                    self.crystal_system_var.set(inferred_system)
                    
                    if hasattr(self, 'pol_reference_status_label'):
                        self.pol_reference_status_label.config(
                            text=f"Crystal system inferred from {new_mineral}",
                            foreground="blue"
                        )
                else:
                    if hasattr(self, 'pol_reference_status_label'):
                        self.pol_reference_status_label.config(
                            text=f"Reference: {new_mineral} (system unknown)",
                            foreground="orange"
                        )
    
    def on_tab_changed(self, event):
        """Handle tab change events."""
        selected_tab = event.widget.tab('current')['text']
        
        # Update Peak Fitting plot when switching to that tab
        if selected_tab == "Peak Fitting" and self.current_spectrum is not None:
            self.update_peak_fitting_plot()
            # Ensure reference mineral is set
            if self.selected_reference_mineral and hasattr(self, 'reference_mineral_var'):
                self.reference_mineral_var.set(self.selected_reference_mineral)
        
        # Update Polarization tab when switching to it
        elif selected_tab == "Polarization Analysis":
            # Ensure crystal system is set based on selected mineral
            if self.selected_reference_mineral and hasattr(self, 'crystal_system_var'):
                inferred_system = self.infer_crystal_system(self.selected_reference_mineral)
                if inferred_system != "Unknown":
                    self.crystal_system_var.set(inferred_system)
            # Update polarization plot if data is available
            if hasattr(self, 'update_polarization_plot'):
                self.update_polarization_plot()
        
        # Update Tensor Analysis tab when switching to it
        elif selected_tab == "Tensor Analysis & Visualization":
            # Update tensor status to ensure button states are correct
            if hasattr(self, 'update_tensor_status'):
                self.update_tensor_status()
    
    def create_gui(self):
        """Create the main GUI layout with tabs and side panels."""
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Create 8 tabs
        self.tabs = {}
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
        """Create a tab with its own side panel and main content area.
        
        Parameters:
        -----------
        tab_name : str
            Name of the tab to create
        """
        # Create tab frame
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=tab_name)
        
        # Create horizontal paned window for side panel and main content
        paned_window = ttk.PanedWindow(tab_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create side panel
        side_panel = ttk.Frame(paned_window, width=300)
        side_panel.pack_propagate(False)  # Maintain fixed width
        paned_window.add(side_panel, weight=0)
        
        # Create main content area
        content_area = ttk.Frame(paned_window)
        paned_window.add(content_area, weight=1)
        
        # Store references
        self.tabs[tab_name] = {
            'frame': tab_frame,
            'side_panel': side_panel,
            'content_area': content_area,
            'paned_window': paned_window
        }
        
        # Populate the side panel and content area based on tab type
        self.setup_tab_content(tab_name)
    
    def setup_tab_content(self, tab_name):
        """Setup the specific content for each tab.
        
        Parameters:
        -----------
        tab_name : str
            Name of the tab to setup
        """
        side_panel = self.tabs[tab_name]['side_panel']
        content_area = self.tabs[tab_name]['content_area']
        
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
        # Side panel title
        ttk.Label(side_panel, text="Spectrum Analysis", font=('Arial', 12, 'bold')).pack(pady=(10, 20))
        
        # File operations section
        file_frame = ttk.LabelFrame(side_panel, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(file_frame, text="Load Spectrum", command=self.load_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Save Spectrum", command=self.save_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Plot", command=self.export_plot).pack(fill=tk.X, pady=2)
        
        # Import spectrum section
        import_frame = ttk.LabelFrame(side_panel, text="Import Spectrum", padding=10)
        import_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(import_frame, text="Search Mineral Database:").pack(anchor=tk.W)
        
        # Search entry with autocomplete
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search_change)
        self.search_entry = ttk.Entry(import_frame, textvariable=self.search_var)
        self.search_entry.pack(fill=tk.X, pady=2)
        
        # Listbox for search results
        self.search_listbox = tk.Listbox(import_frame, height=6)
        self.search_listbox.pack(fill=tk.X, pady=2)
        self.search_listbox.bind('<Double-Button-1>', self.on_mineral_select)
        
        # Button to import selected mineral
        ttk.Button(import_frame, text="Import Selected", command=self.import_selected_mineral).pack(fill=tk.X, pady=2)
        
        # Analysis options section
        analysis_frame = ttk.LabelFrame(side_panel, text="Analysis Options", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(analysis_frame, text="Baseline Correction", command=self.baseline_correction).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Smoothing", command=self.smoothing).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Reset to Original", command=self.reset_to_original).pack(fill=tk.X, pady=2)
        
        # Display options section
        display_frame = ttk.LabelFrame(side_panel, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Grid", variable=self.show_grid_var).pack(anchor=tk.W)
        
        self.show_legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Legend", variable=self.show_legend_var).pack(anchor=tk.W)
        
        # Content area - matplotlib plot
        self.setup_matplotlib_plot(content_area, "Spectrum Analysis")
    
    def setup_peak_fitting_tab(self, side_panel, content_area):
        """Setup the Peak Fitting tab."""
        # Side panel title
        ttk.Label(side_panel, text="Peak Fitting", font=('Arial', 12, 'bold')).pack(pady=(10, 20))
        
        # Peak detection section
        detection_frame = ttk.LabelFrame(side_panel, text="Peak Detection", padding=10)
        detection_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(detection_frame, text="Auto Find Peaks", command=self.auto_find_peaks).pack(fill=tk.X, pady=2)
        ttk.Button(detection_frame, text="Manual Peak Selection", command=self.manual_peak_selection).pack(fill=tk.X, pady=2)
        ttk.Button(detection_frame, text="Clear Selected Peaks", command=self.clear_selected_peaks).pack(fill=tk.X, pady=2)
        
        # Fitting parameters section
        fitting_frame = ttk.LabelFrame(side_panel, text="Fitting Parameters", padding=10)
        fitting_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(fitting_frame, text="Peak Shape:").pack(anchor=tk.W)
        self.peak_shape_var = tk.StringVar(value="Lorentzian")
        shape_combo = ttk.Combobox(fitting_frame, textvariable=self.peak_shape_var, 
                                  values=["Lorentzian", "Gaussian", "Voigt"])
        shape_combo.pack(fill=tk.X, pady=2)
        
        ttk.Button(fitting_frame, text="Fit Selected Peaks", command=self.fit_peaks).pack(fill=tk.X, pady=5)
        
        # Character assignment section
        assignment_frame = ttk.LabelFrame(side_panel, text="Character Assignment", padding=10)
        assignment_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(assignment_frame, text="Reference Mineral:").pack(anchor=tk.W)
        self.reference_mineral_var = tk.StringVar()
        self.reference_combo = ttk.Combobox(assignment_frame, textvariable=self.reference_mineral_var,
                                           values=[], state="readonly")
        self.reference_combo.pack(fill=tk.X, pady=2)
        self.reference_combo.bind('<<ComboboxSelected>>', self.on_reference_mineral_changed)
        
        # Add status label for reference mineral
        self.reference_status_label = ttk.Label(assignment_frame, text="", foreground="gray", font=('Arial', 8))
        self.reference_status_label.pack(anchor=tk.W, pady=2)
        
        ttk.Button(assignment_frame, text="Auto Assign Characters", command=self.auto_assign_characters).pack(fill=tk.X, pady=2)
        ttk.Button(assignment_frame, text="Calculate Frequency Shifts", command=self.calculate_frequency_shifts).pack(fill=tk.X, pady=2)
        ttk.Button(assignment_frame, text="Generate Shifted Spectrum", command=self.generate_shifted_spectrum).pack(fill=tk.X, pady=2)
        
        # Results section
        results_frame = ttk.LabelFrame(side_panel, text="Results", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(results_frame, text="Show Fit Parameters", command=self.show_fit_parameters).pack(fill=tk.X, pady=2)
        ttk.Button(results_frame, text="Show Assignments", command=self.show_peak_assignments).pack(fill=tk.X, pady=2)
        ttk.Button(results_frame, text="Export Results", command=self.export_fit_results).pack(fill=tk.X, pady=2)
        
        # Content area
        self.setup_matplotlib_plot(content_area, "Peak Fitting")
    
    def setup_polarization_tab(self, side_panel, content_area):
        """Setup the Polarization tab."""
        # Side panel title
        ttk.Label(side_panel, text="Polarization Analysis", font=('Arial', 12, 'bold')).pack(pady=(10, 20))
        
        # Data loading section
        data_frame = ttk.LabelFrame(side_panel, text="Polarization Data", padding=10)
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(data_frame, text="Load Polarized Spectra", command=self.load_polarized_spectra).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Load Angular Series", command=self.load_angular_series).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Clear Polarization Data", command=self.clear_polarization_data).pack(fill=tk.X, pady=2)
        
        # Current configuration display
        self.pol_config_label = ttk.Label(data_frame, text="No data loaded", foreground="gray")
        self.pol_config_label.pack(anchor=tk.W, pady=2)
        
        # Measurement configuration section
        config_frame = ttk.LabelFrame(side_panel, text="Measurement Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(config_frame, text="Scattering Geometry:").pack(anchor=tk.W)
        self.scattering_geometry_var = tk.StringVar(value="Backscattering")
        geometry_combo = ttk.Combobox(config_frame, textvariable=self.scattering_geometry_var,
                                     values=["Backscattering", "Right-angle", "Forward"], state="readonly")
        geometry_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(config_frame, text="Crystal System:").pack(anchor=tk.W, pady=(5, 0))
        self.crystal_system_var = tk.StringVar(value="Unknown")
        crystal_combo = ttk.Combobox(config_frame, textvariable=self.crystal_system_var,
                                    values=["Cubic", "Tetragonal", "Orthorhombic", "Hexagonal", 
                                           "Trigonal", "Monoclinic", "Triclinic", "Unknown"], state="readonly")
        crystal_combo.pack(fill=tk.X, pady=2)
        
        # Add status label for reference mineral
        self.pol_reference_status_label = ttk.Label(config_frame, text="", foreground="gray", font=('Arial', 8))
        self.pol_reference_status_label.pack(anchor=tk.W, pady=2)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(side_panel, text="Polarization Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(analysis_frame, text="Calculate Depolarization Ratios", command=self.calculate_depolarization_ratios).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Determine Raman Tensors", command=self.determine_raman_tensors).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Angular Dependence Analysis", command=self.angular_dependence_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Symmetry Classification", command=self.symmetry_classification).pack(fill=tk.X, pady=2)
        
        # Visualization section
        viz_frame = ttk.LabelFrame(side_panel, text="Visualization", padding=10)
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(viz_frame, text="Plot Type:").pack(anchor=tk.W)
        self.plot_type_var = tk.StringVar(value="Polarized Spectra")
        plot_combo = ttk.Combobox(viz_frame, textvariable=self.plot_type_var,
                                 values=["Polarized Spectra", "Depolarization Ratios", "Angular Dependence", 
                                        "Polar Plot", "Raman Tensor Elements"], state="readonly")
        plot_combo.pack(fill=tk.X, pady=2)
        plot_combo.bind('<<ComboboxSelected>>', self.update_polarization_plot)
        
        ttk.Button(viz_frame, text="Update Plot", command=self.update_polarization_plot).pack(fill=tk.X, pady=2)
        ttk.Button(viz_frame, text="Export Results", command=self.export_polarization_results).pack(fill=tk.X, pady=2)
        
        # Content area
        self.setup_matplotlib_plot(content_area, "Polarization Analysis")
    
    def setup_crystal_structure_tab(self, side_panel, content_area):
        """Setup the Crystal Structure tab."""
        # Initialize crystal structure variables
        self.crystal_structure = None
        self.atomic_positions = {}
        self.bond_analysis = {}
        self.calculated_raman_tensors = {}
        
        # Side panel title
        ttk.Label(side_panel, text="Crystal Structure", font=('Arial', 12, 'bold')).pack(pady=(10, 20))
        
        # Structure loading section
        structure_frame = ttk.LabelFrame(side_panel, text="Load Structure", padding=10)
        structure_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(structure_frame, text="📁 Load CIF File", command=self.load_cif_file).pack(fill=tk.X, pady=2)
        ttk.Button(structure_frame, text="🗃️ Load from Database", command=self.load_from_database).pack(fill=tk.X, pady=2)
        
        # Show pymatgen status
        if not PYMATGEN_AVAILABLE:
            ttk.Button(structure_frame, text="⚠️ Install pymatgen", command=self.install_pymatgen).pack(fill=tk.X, pady=2)
        
        # Current structure display
        self.structure_status_label = ttk.Label(structure_frame, text="No structure loaded", foreground="gray")
        self.structure_status_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Structure info section (larger)
        info_frame = ttk.LabelFrame(side_panel, text="Structure Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.structure_info_text = tk.Text(info_frame, height=12, width=30, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.structure_info_text.yview)
        self.structure_info_text.configure(yscrollcommand=scrollbar.set)
        
        self.structure_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Analysis section (consolidated)
        analysis_frame = ttk.LabelFrame(side_panel, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create two columns for buttons
        button_frame = ttk.Frame(analysis_frame)
        button_frame.pack(fill=tk.X)
        
        left_frame = ttk.Frame(button_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(button_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Left column
        ttk.Button(left_frame, text="🔗 Analyze Bonds", command=self.calculate_bond_lengths).pack(fill=tk.X, pady=1)
        ttk.Button(left_frame, text="📊 Bond Table", command=self.show_bond_table).pack(fill=tk.X, pady=1)
        ttk.Button(left_frame, text="⚙️ Bond Filters", command=self.configure_bond_filters).pack(fill=tk.X, pady=1)
        
        # Right column
        ttk.Button(right_frame, text="🎵 Phonon Modes", command=self.calculate_phonon_modes).pack(fill=tk.X, pady=1)
        ttk.Button(right_frame, text="📈 Raman Tensors", command=self.calculate_raman_tensors).pack(fill=tk.X, pady=1)
        ttk.Button(right_frame, text="📤 Export Data", command=self.export_to_polarization).pack(fill=tk.X, pady=1)
        
        # Visualization section (simplified)
        viz_frame = ttk.LabelFrame(side_panel, text="Visualization", padding=10)
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(viz_frame, text="Display:").pack(anchor=tk.W)
        self.structure_display_var = tk.StringVar(value="Bond Network")
        display_combo = ttk.Combobox(viz_frame, textvariable=self.structure_display_var,
                                    values=["Unit Cell", "Bond Network", "Coordination Spheres"], state="readonly")
        display_combo.pack(fill=tk.X, pady=2)
        display_combo.bind('<<ComboboxSelected>>', self.update_structure_plot)
        
        # Content area
        self.setup_matplotlib_plot(content_area, "Crystal Structure")
    
    def setup_3d_visualization_tab(self, side_panel, content_area):
        """Setup the Interactive 3D Crystal Orientation Simulator with organized sub-tabs."""
        # Side panel title
        ttk.Label(side_panel, text="3D Crystal Orientation Simulator", font=('Arial', 12, 'bold')).pack(pady=(10, 20))
        
        # Create notebook for sub-tabs in side panel
        self.viz_notebook = ttk.Notebook(side_panel)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create sub-tabs
        self.setup_data_import_subtab()
        self.setup_orientation_control_subtab()
        self.setup_visualization_subtab()
        self.setup_analysis_subtab()
        
        # Content area - split into 3D view and spectrum
        self.setup_3d_content_area(content_area)
    
    def setup_data_import_subtab(self):
        """Setup the Data Import sub-tab."""
        import_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(import_tab, text="📁 Data")
        
        # Data import section
        import_frame = ttk.LabelFrame(import_tab, text="Import Data Sources", padding=10)
        import_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(import_frame, text="🎯 Import from Optimization", 
                  command=self.import_optimization_orientation).pack(fill=tk.X, pady=2)
        ttk.Button(import_frame, text="🔬 Import from Structure", 
                  command=self.import_structure_data).pack(fill=tk.X, pady=2)
        
        # Quick setup section
        quick_setup_frame = ttk.LabelFrame(import_tab, text="Quick Setup", padding=10)
        quick_setup_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(quick_setup_frame, text="🚀 Auto-Import All Available Data", 
                  command=self.auto_import_all_data).pack(fill=tk.X, pady=2)
        ttk.Button(quick_setup_frame, text="🧪 Load Demo Data", 
                  command=self.load_demo_data_3d).pack(fill=tk.X, pady=2)
        ttk.Button(quick_setup_frame, text="✨ Quick Setup (Demo + Enable All)", 
                  command=self.quick_setup_3d).pack(fill=tk.X, pady=2)
        
        # Manual import buttons for debugging
        debug_frame = ttk.LabelFrame(import_tab, text="Manual Import (Debug)", padding=5)
        debug_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(debug_frame, text="📈 Create from Fitted Peaks", 
                  command=self.create_tensors_from_fitted_peaks).pack(fill=tk.X, pady=1)
        ttk.Button(debug_frame, text="💎 Create from Mineral Data", 
                  command=self.create_tensors_from_mineral_data).pack(fill=tk.X, pady=1)
        ttk.Button(debug_frame, text="📊 Calculate Spectrum", 
                  command=self.calculate_orientation_spectrum).pack(fill=tk.X, pady=1)
        
        # Status section
        status_frame = ttk.LabelFrame(import_tab, text="Data Status", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.viz_status_text = tk.Text(status_frame, height=6, width=30, font=('Arial', 8))
        self.viz_status_text.pack(fill=tk.X, pady=2)
        
        ttk.Button(status_frame, text="🔄 Refresh Status", 
                  command=self.update_3d_status).pack(fill=tk.X, pady=2)
    
    def setup_orientation_control_subtab(self):
        """Setup the Orientation Control sub-tab."""
        orientation_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(orientation_tab, text="🔄 Orientation")
        
        # Euler angles section
        angles_frame = ttk.LabelFrame(orientation_tab, text="Euler Angles", padding=10)
        angles_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Phi angle (rotation about z-axis)
        ttk.Label(angles_frame, text="φ (Z-axis rotation):").pack(anchor=tk.W)
        self.phi_var = tk.DoubleVar(value=0.0)
        phi_scale = ttk.Scale(angles_frame, from_=0, to=360, variable=self.phi_var,
                             orient=tk.HORIZONTAL, length=200)
        phi_scale.pack(fill=tk.X, pady=2)
        phi_scale.bind('<Motion>', self.on_orientation_change)
        phi_scale.bind('<ButtonRelease-1>', self.on_orientation_change)
        
        self.phi_label = ttk.Label(angles_frame, text="0.0°")
        self.phi_label.pack(anchor=tk.W)
        
        # Theta angle (rotation about y-axis)
        ttk.Label(angles_frame, text="θ (Y-axis rotation):").pack(anchor=tk.W, pady=(10, 0))
        self.theta_var = tk.DoubleVar(value=0.0)
        theta_scale = ttk.Scale(angles_frame, from_=0, to=180, variable=self.theta_var,
                               orient=tk.HORIZONTAL, length=200)
        theta_scale.pack(fill=tk.X, pady=2)
        theta_scale.bind('<Motion>', self.on_orientation_change)
        theta_scale.bind('<ButtonRelease-1>', self.on_orientation_change)
        
        self.theta_label = ttk.Label(angles_frame, text="0.0°")
        self.theta_label.pack(anchor=tk.W)
        
        # Psi angle (rotation about x-axis)
        ttk.Label(angles_frame, text="ψ (X-axis rotation):").pack(anchor=tk.W, pady=(10, 0))
        self.psi_var = tk.DoubleVar(value=0.0)
        psi_scale = ttk.Scale(angles_frame, from_=0, to=360, variable=self.psi_var,
                             orient=tk.HORIZONTAL, length=200)
        psi_scale.pack(fill=tk.X, pady=2)
        psi_scale.bind('<Motion>', self.on_orientation_change)
        psi_scale.bind('<ButtonRelease-1>', self.on_orientation_change)
        
        self.psi_label = ttk.Label(angles_frame, text="0.0°")
        self.psi_label.pack(anchor=tk.W)
        
        # Trace variable changes for real-time updates
        self.phi_var.trace('w', self.update_angle_labels)
        self.theta_var.trace('w', self.update_angle_labels)
        self.psi_var.trace('w', self.update_angle_labels)
        
        # Also trace for orientation and spectrum updates
        self.phi_var.trace('w', lambda *args: self.on_orientation_change())
        self.theta_var.trace('w', lambda *args: self.on_orientation_change())
        self.psi_var.trace('w', lambda *args: self.on_orientation_change())
        
        # Quick orientations section
        quick_frame = ttk.LabelFrame(orientation_tab, text="Quick Orientations", padding=10)
        quick_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a grid for orientation buttons
        button_frame = ttk.Frame(quick_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="🎯 Reset to Optimized", 
                  command=self.reset_to_optimized_orientation).pack(fill=tk.X, pady=1)
        
        # Crystal direction buttons in a grid
        dir_frame = ttk.Frame(quick_frame)
        dir_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(dir_frame, text="[100]", width=8,
                  command=lambda: self.set_crystal_direction([1,0,0])).grid(row=0, column=0, padx=2, pady=1)
        ttk.Button(dir_frame, text="[010]", width=8,
                  command=lambda: self.set_crystal_direction([0,1,0])).grid(row=0, column=1, padx=2, pady=1)
        ttk.Button(dir_frame, text="[001]", width=8,
                  command=lambda: self.set_crystal_direction([0,0,1])).grid(row=0, column=2, padx=2, pady=1)
        
        ttk.Button(dir_frame, text="[110]", width=8,
                  command=lambda: self.set_crystal_direction([1,1,0])).grid(row=1, column=0, padx=2, pady=1)
        ttk.Button(dir_frame, text="[101]", width=8,
                  command=lambda: self.set_crystal_direction([1,0,1])).grid(row=1, column=1, padx=2, pady=1)
        ttk.Button(dir_frame, text="[111]", width=8,
                  command=lambda: self.set_crystal_direction([1,1,1])).grid(row=1, column=2, padx=2, pady=1)
        
        # Animation section
        animation_frame = ttk.LabelFrame(orientation_tab, text="Animation Controls", padding=10)
        animation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.animation_running = False
        
        # Animation settings
        settings_frame = ttk.Frame(animation_frame)
        settings_frame.pack(fill=tk.X)
        
        ttk.Label(settings_frame, text="Rotation Axis:").pack(anchor=tk.W)
        self.animation_axis_var = tk.StringVar(value="φ")
        axis_combo = ttk.Combobox(settings_frame, textvariable=self.animation_axis_var,
                                 values=["φ", "θ", "ψ", "All"], state="readonly", width=15)
        axis_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(settings_frame, text="Speed (°/frame):").pack(anchor=tk.W, pady=(5, 0))
        self.animation_speed_var = tk.DoubleVar(value=2.0)
        speed_scale = ttk.Scale(settings_frame, from_=0.5, to=10.0, variable=self.animation_speed_var,
                               orient=tk.HORIZONTAL, length=200)
        speed_scale.pack(fill=tk.X, pady=2)
        
        # Animation buttons
        button_frame = ttk.Frame(animation_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="▶️ Start", width=12,
                  command=self.start_crystal_animation).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏹️ Stop", width=12,
                  command=self.stop_crystal_animation).pack(side=tk.LEFT, padx=2)
    
    def setup_visualization_subtab(self):
        """Setup the Visualization Options sub-tab."""
        viz_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(viz_tab, text="👁️ Display")
        
        # Display elements section
        display_frame = ttk.LabelFrame(viz_tab, text="Display Elements", padding=10)
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_crystal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="🔷 Crystal Shape", variable=self.show_crystal_var,
                       command=self.update_3d_visualization).pack(anchor=tk.W, pady=2)
        
        self.show_tensor_ellipsoid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="🔴 Raman Tensor Ellipsoid", variable=self.show_tensor_ellipsoid_var,
                       command=self.update_3d_visualization).pack(anchor=tk.W, pady=2)
        
        self.show_laser_geometry_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="🔶 Laser Geometry", variable=self.show_laser_geometry_var,
                       command=self.update_3d_visualization).pack(anchor=tk.W, pady=2)
        
        self.show_axes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="📐 Coordinate Axes", variable=self.show_axes_var,
                       command=self.update_3d_visualization).pack(anchor=tk.W, pady=2)
        
        self.show_crystal_structure_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(display_frame, text="⚛️ Crystal Structure", variable=self.show_crystal_structure_var,
                       command=self.update_3d_visualization).pack(anchor=tk.W, pady=2)
        
        # View controls section
        view_frame = ttk.LabelFrame(viz_tab, text="View Controls", padding=10)
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # View preset buttons
        view_button_frame = ttk.Frame(view_frame)
        view_button_frame.pack(fill=tk.X)
        
        ttk.Button(view_button_frame, text="🔄 Reset View", 
                  command=self.reset_3d_view).pack(fill=tk.X, pady=1)
        ttk.Button(view_button_frame, text="⬆️ Top View", 
                  command=self.set_top_view).pack(fill=tk.X, pady=1)
        ttk.Button(view_button_frame, text="➡️ Side View", 
                  command=self.set_side_view).pack(fill=tk.X, pady=1)
        ttk.Button(view_button_frame, text="👁️ Isometric View", 
                  command=self.set_isometric_view).pack(fill=tk.X, pady=1)
        
        # Rendering options section
        render_frame = ttk.LabelFrame(viz_tab, text="Rendering Options", padding=10)
        render_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.crystal_transparency_var = tk.DoubleVar(value=0.3)
        ttk.Label(render_frame, text="Crystal Transparency:").pack(anchor=tk.W)
        transparency_scale = ttk.Scale(render_frame, from_=0.0, to=1.0, variable=self.crystal_transparency_var,
                                     orient=tk.HORIZONTAL, length=200)
        transparency_scale.pack(fill=tk.X, pady=2)
        transparency_scale.bind('<ButtonRelease-1>', lambda e: self.update_3d_visualization())
        
        self.tensor_scale_var = tk.DoubleVar(value=1.0)  # Increased default size
        ttk.Label(render_frame, text="Tensor Scale:").pack(anchor=tk.W, pady=(5, 0))
        tensor_scale = ttk.Scale(render_frame, from_=0.1, to=2.0, variable=self.tensor_scale_var,
                               orient=tk.HORIZONTAL, length=200)
        tensor_scale.pack(fill=tk.X, pady=2)
        tensor_scale.bind('<ButtonRelease-1>', lambda e: self.update_3d_visualization())
        
        # Crystal structure rendering options
        structure_options_frame = ttk.LabelFrame(render_frame, text="Crystal Structure Options", padding=5)
        structure_options_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.unit_cell_range_var = tk.DoubleVar(value=2.0)
        
        # Unit cell range with dynamic label
        range_label = ttk.Label(structure_options_frame, text=f"Unit Cell Range (±): {self.unit_cell_range_var.get():.1f}")
        range_label.pack(anchor=tk.W)
        
        def update_range_label(*args):
            range_label.config(text=f"Unit Cell Range (±): {self.unit_cell_range_var.get():.1f}")
        
        self.unit_cell_range_var.trace('w', update_range_label)
        
        range_scale = ttk.Scale(structure_options_frame, from_=0.5, to=5.0, variable=self.unit_cell_range_var,
                              orient=tk.HORIZONTAL, length=180)
        range_scale.pack(fill=tk.X, pady=2)
        range_scale.bind('<ButtonRelease-1>', lambda e: self.update_3d_visualization())
        
        self.atom_size_var = tk.DoubleVar(value=0.3)
        atom_size_label = ttk.Label(structure_options_frame, text=f"Atom Size: {self.atom_size_var.get():.2f}")
        atom_size_label.pack(anchor=tk.W, pady=(5, 0))
        
        def update_atom_size_label(*args):
            atom_size_label.config(text=f"Atom Size: {self.atom_size_var.get():.2f}")
        
        self.atom_size_var.trace('w', update_atom_size_label)
        
        atom_scale = ttk.Scale(structure_options_frame, from_=0.1, to=1.0, variable=self.atom_size_var,
                             orient=tk.HORIZONTAL, length=180)
        atom_scale.pack(fill=tk.X, pady=2)
        atom_scale.bind('<ButtonRelease-1>', lambda e: self.update_3d_visualization())
        
        # Add structure compression factor
        self.structure_compression_var = tk.DoubleVar(value=1.0)
        compression_label = ttk.Label(structure_options_frame, text=f"Structure Compression: {self.structure_compression_var.get():.2f}")
        compression_label.pack(anchor=tk.W, pady=(5, 0))
        
        def update_compression_label(*args):
            compression_label.config(text=f"Structure Compression: {self.structure_compression_var.get():.2f}")
        
        self.structure_compression_var.trace('w', update_compression_label)
        
        compression_scale = ttk.Scale(structure_options_frame, from_=0.1, to=2.0, variable=self.structure_compression_var,
                                    orient=tk.HORIZONTAL, length=180)
        compression_scale.pack(fill=tk.X, pady=2)
        compression_scale.bind('<ButtonRelease-1>', lambda e: self.update_3d_visualization())
        
        self.show_bonds_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(structure_options_frame, text="Show Bonds", variable=self.show_bonds_var,
                       command=self.update_3d_visualization).pack(anchor=tk.W, pady=2)
        
        self.show_unit_cell_edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(structure_options_frame, text="Show Unit Cell Edges", variable=self.show_unit_cell_edges_var,
                       command=self.update_3d_visualization).pack(anchor=tk.W, pady=2)
        
        # Add optimization buttons
        button_frame = ttk.Frame(structure_options_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="🔍 Optimize for Dense View", 
                  command=self.optimize_for_dense_view).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="🏠 Reset to Default", 
                  command=self.reset_3d_view_settings).pack(side=tk.LEFT)
    
    def setup_analysis_subtab(self):
        """Setup the Analysis & Export sub-tab."""
        analysis_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(analysis_tab, text="📊 Analysis")
        
        # Spectrum analysis section
        spectrum_frame = ttk.LabelFrame(analysis_tab, text="Spectrum Analysis", padding=10)
        spectrum_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.real_time_calc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(spectrum_frame, text="⚡ Real-time Calculation", 
                       variable=self.real_time_calc_var).pack(anchor=tk.W, pady=2)
        
        ttk.Label(spectrum_frame, text="Selected Peak (cm⁻¹):").pack(anchor=tk.W, pady=(5, 0))
        self.viz_peak_var = tk.StringVar(value="All Peaks")
        self.viz_peak_combo = ttk.Combobox(spectrum_frame, textvariable=self.viz_peak_var, state="readonly")
        self.viz_peak_combo.pack(fill=tk.X, pady=2)
        self.viz_peak_combo.bind('<<ComboboxSelected>>', lambda e: self.update_3d_visualization())
        
        ttk.Button(spectrum_frame, text="🔄 Update Spectrum", 
                  command=self.calculate_orientation_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(spectrum_frame, text="✨ Quick Setup & Refresh", 
                  command=self.quick_setup_3d).pack(fill=tk.X, pady=2)
        
        # Polarization settings section
        pol_frame = ttk.LabelFrame(analysis_tab, text="Polarization Configuration", padding=10)
        pol_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(pol_frame, text="Incident Polarization:").pack(anchor=tk.W)
        self.incident_pol_var = tk.StringVar(value="X-polarized")
        incident_combo = ttk.Combobox(pol_frame, textvariable=self.incident_pol_var,
                                    values=["X-polarized", "Y-polarized", "Z-polarized", "Circular"], 
                                    state="readonly")
        incident_combo.pack(fill=tk.X, pady=2)
        incident_combo.bind('<<ComboboxSelected>>', lambda e: self.calculate_orientation_spectrum())
        
        ttk.Label(pol_frame, text="Scattered Polarization:").pack(anchor=tk.W, pady=(5, 0))
        self.scattered_pol_var = tk.StringVar(value="X-polarized")
        scattered_combo = ttk.Combobox(pol_frame, textvariable=self.scattered_pol_var,
                                     values=["X-polarized", "Y-polarized", "Z-polarized", "Parallel", "Perpendicular"], 
                                     state="readonly")
        scattered_combo.pack(fill=tk.X, pady=2)
        scattered_combo.bind('<<ComboboxSelected>>', lambda e: self.calculate_orientation_spectrum())
        
        # Export section
        export_frame = ttk.LabelFrame(analysis_tab, text="Export & Save", padding=10)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(export_frame, text="💾 Save 3D View", 
                  command=self.save_3d_view).pack(fill=tk.X, pady=1)
        ttk.Button(export_frame, text="📈 Export Spectrum", 
                  command=self.export_orientation_spectrum).pack(fill=tk.X, pady=1)
        ttk.Button(export_frame, text="📋 Export Orientation Data", 
                  command=self.export_orientation_data).pack(fill=tk.X, pady=1)
        ttk.Button(export_frame, text="📄 Generate Report", 
                  command=self.generate_3d_report).pack(fill=tk.X, pady=1)
    
    def setup_matplotlib_plot(self, parent, title):
        """Setup a matplotlib plot in the given parent frame.
        
        Parameters:
        -----------
        parent : tkinter.Frame
            Parent frame to contain the plot
        title : str
            Title for the plot
        """
        # Create matplotlib figure
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # For Spectrum Analysis tab, show empty plot initially
        if title == "Spectrum Analysis":
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.text(0.5, 0.5, 'Load a spectrum or import from database', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=12, alpha=0.6)
        else:
            # Sample plot for other tabs
            x = np.linspace(0, 2000, 1000)
            y = np.exp(-(x-1000)**2/50000) + 0.5*np.exp(-(x-1500)**2/30000)
            ax.plot(x, y, 'b-', linewidth=2, label='Sample Spectrum')
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
        # Store references for later use
        self.plot_components[title] = {
            'fig': fig,
            'ax': ax,
            'canvas': canvas,
            'toolbar': toolbar
        }
    
    # Placeholder methods for button callbacks
    def load_spectrum(self):
        """Load a spectrum file."""
        file_path = filedialog.askopenfilename(
            title="Select Spectrum File",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # First, try to detect if the file has headers
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                
                # Check if first line contains non-numeric data (likely headers)
                has_header = False
                try:
                    # Try to convert first two values to float
                    parts = first_line.split()
                    if len(parts) >= 2:
                        float(parts[0])
                        float(parts[1])
                except (ValueError, IndexError):
                    has_header = True
                
                # Load the data, skipping header if present
                if has_header:
                    data = np.loadtxt(file_path, delimiter=None, skiprows=1)
                else:
                    data = np.loadtxt(file_path, delimiter=None)
                
                # Handle both 1D and 2D arrays
                if data.ndim == 1:
                    messagebox.showerror("Error", "File must contain at least two columns (wavenumber, intensity)")
                    return
                
                if data.shape[1] >= 2:
                    wavenumbers = data[:, 0]
                    intensities = data[:, 1]
                    
                    # Store both original and current spectrum
                    spectrum_data = {
                        'wavenumbers': wavenumbers.copy(),
                        'intensities': intensities.copy(),
                        'name': os.path.basename(file_path)
                    }
                    
                    self.original_spectrum = spectrum_data.copy()
                    self.current_spectrum = spectrum_data
                    
                    # Update the plot
                    self.update_spectrum_plot()
                    
                    #messagebox.showinfo("Success", f"Loaded spectrum from {os.path.basename(file_path)}")
                else:
                    messagebox.showerror("Error", "File must contain at least two columns (wavenumber, intensity)")
                    
            except Exception as e:
                # If the above fails, try alternative loading methods
                try:
                    # Try pandas for more robust CSV handling
                    import pandas as pd
                    df = pd.read_csv(file_path, sep=None, engine='python')
                    
                    if len(df.columns) >= 2:
                        # Use first two numeric columns
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            wavenumbers = df[numeric_cols[0]].values
                            intensities = df[numeric_cols[1]].values
                            
                            # Store both original and current spectrum
                            spectrum_data = {
                                'wavenumbers': wavenumbers.copy(),
                                'intensities': intensities.copy(),
                                'name': os.path.basename(file_path)
                            }
                            
                            self.original_spectrum = spectrum_data.copy()
                            self.current_spectrum = spectrum_data
                            
                            # Update the plot
                            self.update_spectrum_plot()
                            
                            #messagebox.showinfo("Success", f"Loaded spectrum from {os.path.basename(file_path)}")
                        else:
                            messagebox.showerror("Error", "File must contain at least two numeric columns")
                    else:
                        messagebox.showerror("Error", "File must contain at least two columns")
                        
                except ImportError:
                    messagebox.showerror("Error", f"Error loading spectrum file: {str(e)}\n\nTip: Install pandas for better CSV support: pip install pandas")
                except Exception as e2:
                    messagebox.showerror("Error", f"Error loading spectrum file: {str(e)}\n\nAlternative method also failed: {str(e2)}")
    
    def save_spectrum(self):
        messagebox.showinfo("Save Spectrum", "Save spectrum functionality will be implemented here.")
    
    def export_plot(self):
        messagebox.showinfo("Export Plot", "Export plot functionality will be implemented here.")
    
    def baseline_correction(self):
        """Apply baseline correction to the current spectrum using Asymmetric Least Squares."""
        if self.current_spectrum is None:
            messagebox.showwarning("No Data", "Please load a spectrum first.")
            return
        
        # Create a dialog for baseline correction parameters
        dialog = tk.Toplevel(self.root)
        dialog.title("Baseline Correction Parameters")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Parameters frame
        params_frame = ttk.LabelFrame(dialog, text="ALS Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Lambda parameter (smoothness)
        ttk.Label(params_frame, text="Lambda (smoothness):").pack(anchor=tk.W)
        lambda_var = tk.DoubleVar(value=1e6)
        lambda_scale = ttk.Scale(params_frame, from_=1e3, to=1e8, variable=lambda_var, 
                                orient=tk.HORIZONTAL, length=300)
        lambda_scale.pack(fill=tk.X, pady=2)
        lambda_label = ttk.Label(params_frame, text=f"Value: {lambda_var.get():.0e}")
        lambda_label.pack(anchor=tk.W)
        
        def update_lambda_label(*args):
            lambda_label.config(text=f"Value: {lambda_var.get():.0e}")
        lambda_var.trace('w', update_lambda_label)
        
        # P parameter (asymmetry)
        ttk.Label(params_frame, text="P (asymmetry):").pack(anchor=tk.W, pady=(10, 0))
        p_var = tk.DoubleVar(value=0.01)
        p_scale = ttk.Scale(params_frame, from_=0.001, to=0.1, variable=p_var, 
                           orient=tk.HORIZONTAL, length=300)
        p_scale.pack(fill=tk.X, pady=2)
        p_label = ttk.Label(params_frame, text=f"Value: {p_var.get():.3f}")
        p_label.pack(anchor=tk.W)
        
        def update_p_label(*args):
            p_label.config(text=f"Value: {p_var.get():.3f}")
        p_var.trace('w', update_p_label)
        
        # Iterations
        ttk.Label(params_frame, text="Iterations:").pack(anchor=tk.W, pady=(10, 0))
        iter_var = tk.IntVar(value=10)
        iter_scale = ttk.Scale(params_frame, from_=5, to=50, variable=iter_var, 
                              orient=tk.HORIZONTAL, length=300)
        iter_scale.pack(fill=tk.X, pady=2)
        iter_label = ttk.Label(params_frame, text=f"Value: {iter_var.get()}")
        iter_label.pack(anchor=tk.W)
        
        def update_iter_label(*args):
            iter_label.config(text=f"Value: {iter_var.get()}")
        iter_var.trace('w', update_iter_label)
        
        # Buttons frame
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def apply_correction():
            try:
                # Get parameters
                lam = lambda_var.get()
                p = p_var.get()
                niter = iter_var.get()
                
                # Apply ALS baseline correction
                y = self.current_spectrum['intensities']
                baseline = self.als_baseline(y, lam, p, niter)
                corrected_intensities = y - baseline
                
                # Update current spectrum
                self.current_spectrum['intensities'] = corrected_intensities
                
                # Update plot
                self.update_spectrum_plot()
                
                dialog.destroy()
                #messagebox.showinfo("Success", "Baseline correction applied successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error applying baseline correction: {str(e)}")
        
        def preview_correction():
            try:
                # Get parameters
                lam = lambda_var.get()
                p = p_var.get()
                niter = iter_var.get()
                
                # Apply ALS baseline correction
                y = self.current_spectrum['intensities']
                baseline = self.als_baseline(y, lam, p, niter)
                
                # Create preview window
                preview_window = tk.Toplevel(dialog)
                preview_window.title("Baseline Correction Preview")
                preview_window.geometry("600x400")
                
                # Create matplotlib figure for preview
                from matplotlib.figure import Figure
                fig = Figure(figsize=(8, 6), dpi=80)
                ax = fig.add_subplot(111)
                
                wavenumbers = self.current_spectrum['wavenumbers']
                ax.plot(wavenumbers, y, 'b-', label='Original', alpha=0.7)
                ax.plot(wavenumbers, baseline, 'r-', label='Baseline', alpha=0.7)
                ax.plot(wavenumbers, y - baseline, 'g-', label='Corrected')
                
                ax.set_xlabel('Wavenumber (cm⁻¹)')
                ax.set_ylabel('Intensity')
                ax.set_title('Baseline Correction Preview')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                canvas = FigureCanvasTkAgg(fig, preview_window)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error generating preview: {str(e)}")
        
        ttk.Button(buttons_frame, text="Preview", command=preview_correction).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Apply", command=apply_correction).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def als_baseline(self, y, lam, p, niter=10):
        """
        Asymmetric Least Squares baseline correction algorithm.
        
        Parameters:
        -----------
        y : array
            Input spectrum intensities
        lam : float
            Smoothness parameter (larger = smoother baseline)
        p : float
            Asymmetry parameter (0 < p < 1, smaller = more asymmetric)
        niter : int
            Number of iterations
            
        Returns:
        --------
        array
            Baseline
        """
        L = len(y)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        W = diags(w, 0, shape=(L, L))
        
        for i in range(niter):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        
        return z
    
    def smoothing(self):
        """Apply Savitzky-Golay smoothing to the current spectrum."""
        if self.current_spectrum is None:
            messagebox.showwarning("No Data", "Please load a spectrum first.")
            return
        
        # Create a dialog for smoothing parameters
        dialog = tk.Toplevel(self.root)
        dialog.title("Smoothing Parameters")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Parameters frame
        params_frame = ttk.LabelFrame(dialog, text="Savitzky-Golay Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Window length parameter
        ttk.Label(params_frame, text="Window Length (must be odd):").pack(anchor=tk.W)
        window_var = tk.IntVar(value=11)
        window_scale = ttk.Scale(params_frame, from_=5, to=51, variable=window_var, 
                                orient=tk.HORIZONTAL, length=300)
        window_scale.pack(fill=tk.X, pady=2)
        window_label = ttk.Label(params_frame, text=f"Value: {window_var.get()}")
        window_label.pack(anchor=tk.W)
        
        def update_window_label(*args):
            # Ensure odd number
            val = window_var.get()
            if val % 2 == 0:
                val += 1
                window_var.set(val)
            window_label.config(text=f"Value: {val}")
        window_var.trace('w', update_window_label)
        
        # Polynomial order parameter
        ttk.Label(params_frame, text="Polynomial Order:").pack(anchor=tk.W, pady=(10, 0))
        poly_var = tk.IntVar(value=3)
        poly_scale = ttk.Scale(params_frame, from_=1, to=6, variable=poly_var, 
                              orient=tk.HORIZONTAL, length=300)
        poly_scale.pack(fill=tk.X, pady=2)
        poly_label = ttk.Label(params_frame, text=f"Value: {poly_var.get()}")
        poly_label.pack(anchor=tk.W)
        
        def update_poly_label(*args):
            poly_label.config(text=f"Value: {poly_var.get()}")
        poly_var.trace('w', update_poly_label)
        
        # Buttons frame
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def apply_smoothing():
            try:
                # Get parameters
                window_length = window_var.get()
                poly_order = poly_var.get()
                
                # Ensure window length is odd and greater than poly_order
                if window_length % 2 == 0:
                    window_length += 1
                
                if window_length <= poly_order:
                    messagebox.showerror("Error", "Window length must be greater than polynomial order.")
                    return
                
                # Apply Savitzky-Golay smoothing
                y = self.current_spectrum['intensities']
                smoothed_intensities = savgol_filter(y, window_length, poly_order)
                
                # Update current spectrum
                self.current_spectrum['intensities'] = smoothed_intensities
                
                # Update plot
                self.update_spectrum_plot()
                
                dialog.destroy()
                #messagebox.showinfo("Success", "Smoothing applied successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error applying smoothing: {str(e)}")
        
        def preview_smoothing():
            try:
                # Get parameters
                window_length = window_var.get()
                poly_order = poly_var.get()
                
                # Ensure window length is odd and greater than poly_order
                if window_length % 2 == 0:
                    window_length += 1
                
                if window_length <= poly_order:
                    messagebox.showerror("Error", "Window length must be greater than polynomial order.")
                    return
                
                # Apply Savitzky-Golay smoothing
                y = self.current_spectrum['intensities']
                smoothed_intensities = savgol_filter(y, window_length, poly_order)
                
                # Create preview window
                preview_window = tk.Toplevel(dialog)
                preview_window.title("Smoothing Preview")
                preview_window.geometry("600x400")
                
                # Create matplotlib figure for preview
                from matplotlib.figure import Figure
                fig = Figure(figsize=(8, 6), dpi=80)
                ax = fig.add_subplot(111)
                
                wavenumbers = self.current_spectrum['wavenumbers']
                ax.plot(wavenumbers, y, 'b-', label='Original', alpha=0.7)
                ax.plot(wavenumbers, smoothed_intensities, 'r-', label='Smoothed', linewidth=2)
                
                ax.set_xlabel('Wavenumber (cm⁻¹)')
                ax.set_ylabel('Intensity')
                ax.set_title('Smoothing Preview')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                canvas = FigureCanvasTkAgg(fig, preview_window)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error generating preview: {str(e)}")
        
        ttk.Button(buttons_frame, text="Preview", command=preview_smoothing).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Apply", command=apply_smoothing).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def reset_to_original(self):
        """Reset the current spectrum to its original unprocessed state."""
        if self.original_spectrum is None:
            messagebox.showwarning("No Data", "No original spectrum data available.")
            return
        
        # Reset current spectrum to original
        self.current_spectrum = {
            'wavenumbers': self.original_spectrum['wavenumbers'].copy(),
            'intensities': self.original_spectrum['intensities'].copy(),
            'name': self.original_spectrum['name']
        }
        
        # Update plot
        self.update_spectrum_plot()
        
        messagebox.showinfo("Success", "Spectrum reset to original state.")
    
    def auto_find_peaks(self):
        """Automatically find peaks in the current spectrum."""
        if self.current_spectrum is None:
            messagebox.showwarning("No Data", "Please load a spectrum first.")
            return
        
        try:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            
            # Find peaks using scipy
            # Use prominence and height thresholds
            prominence = np.max(intensities) * 0.05  # 5% of max intensity
            height = np.max(intensities) * 0.1       # 10% of max intensity
            
            peaks, properties = find_peaks(intensities, 
                                         prominence=prominence,
                                         height=height,
                                         distance=10)  # Minimum distance between peaks
            
            # Convert peak indices to wavenumber positions
            self.selected_peaks = [wavenumbers[i] for i in peaks]
            
            # Update the peak fitting plot
            self.update_peak_fitting_plot()
            
            #messagebox.showinfo("Success", f"Found {len(self.selected_peaks)} peaks automatically.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error finding peaks: {str(e)}")
    
    def manual_peak_selection(self):
        """Enable manual peak selection mode."""
        if self.current_spectrum is None:
            messagebox.showwarning("No Data", "Please load a spectrum first.")
            return
        
        self.peak_selection_mode = True
        
        # Connect click event to peak fitting plot
        if "Peak Fitting" in self.plot_components:
            canvas = self.plot_components["Peak Fitting"]['canvas']
            self.peak_click_cid = canvas.mpl_connect('button_press_event', self.on_peak_click)
            
            messagebox.showinfo("Manual Selection", 
                              "Click on peaks in the plot to select them.\n"
                              "Click 'Clear Selected Peaks' when done.")
        else:
            messagebox.showerror("Error", "Peak fitting plot not available.")
    
    def on_peak_click(self, event):
        """Handle mouse clicks for manual peak selection."""
        if not self.peak_selection_mode or event.inaxes is None:
            return
        
        # Get clicked wavenumber position
        clicked_wavenumber = event.xdata
        
        if clicked_wavenumber is not None:
            # Find the closest data point
            wavenumbers = self.current_spectrum['wavenumbers']
            closest_idx = np.argmin(np.abs(wavenumbers - clicked_wavenumber))
            closest_wavenumber = wavenumbers[closest_idx]
            
            # Add to selected peaks if not already selected
            if closest_wavenumber not in self.selected_peaks:
                self.selected_peaks.append(closest_wavenumber)
                self.selected_peaks.sort()  # Keep sorted
                
                # Update the plot
                self.update_peak_fitting_plot()
    
    def clear_selected_peaks(self):
        """Clear all selected peaks."""
        self.selected_peaks = []
        self.fitted_peaks = []
        self.peak_assignments = {}
        self.frequency_shifts = {}
        self.peak_selection_mode = False
        
        # Disconnect click event if connected
        if hasattr(self, 'peak_click_cid') and "Peak Fitting" in self.plot_components:
            canvas = self.plot_components["Peak Fitting"]['canvas']
            canvas.mpl_disconnect(self.peak_click_cid)
        
        # Update the plot
        self.update_peak_fitting_plot()
        
        #messagebox.showinfo("Success", "All selected peaks cleared.")
    
    def fit_peaks(self):
        """Fit the selected peaks with the chosen peak shape."""
        if self.current_spectrum is None:
            messagebox.showwarning("No Data", "Please load a spectrum first.")
            return
        
        if not self.selected_peaks:
            messagebox.showwarning("No Peaks", "Please select peaks first.")
            return
        
        try:
            wavenumbers = self.current_spectrum['wavenumbers']
            intensities = self.current_spectrum['intensities']
            peak_shape = self.peak_shape_var.get()
            
            self.fitted_peaks = []
            
            for peak_pos in self.selected_peaks:
                # Find the peak index
                peak_idx = np.argmin(np.abs(wavenumbers - peak_pos))
                
                # Define fitting window around the peak (adaptive size)
                wavenumber_spacing = np.mean(np.diff(wavenumbers))
                window_width = 50.0  # cm⁻¹
                window_points = int(window_width / wavenumber_spacing)
                window_points = max(20, min(window_points, 100))  # Ensure reasonable window size
                
                start_idx = max(0, peak_idx - window_points)
                end_idx = min(len(wavenumbers), peak_idx + window_points)
                
                x_fit = wavenumbers[start_idx:end_idx]
                y_fit = intensities[start_idx:end_idx]
                
                # Better initial parameter estimates
                amplitude = intensities[peak_idx]
                center = peak_pos
                
                # Estimate width from data (FWHM approximation)
                half_max = amplitude / 2
                left_idx = peak_idx
                right_idx = peak_idx
                
                # Find left half-maximum
                while left_idx > start_idx and intensities[left_idx] > half_max:
                    left_idx -= 1
                
                # Find right half-maximum
                while right_idx < end_idx - 1 and intensities[right_idx] > half_max:
                    right_idx += 1
                
                estimated_fwhm = wavenumbers[right_idx] - wavenumbers[left_idx]
                width = max(2.0, min(estimated_fwhm / 2, 50.0))  # Reasonable width bounds
                
                # Estimate baseline
                baseline = min(np.min(y_fit), np.mean([y_fit[0], y_fit[-1]]))
                y_fit_corrected = y_fit - baseline
                amplitude_corrected = amplitude - baseline
                
                try:
                    # Fit the peak with bounds and improved settings
                    if peak_shape == "Lorentzian":
                        # Parameter bounds: [amplitude, center, width]
                        bounds = ([0, x_fit[0], 0.5], 
                                [amplitude_corrected * 3, x_fit[-1], 100.0])
                        popt, pcov = curve_fit(self.lorentzian, x_fit, y_fit_corrected, 
                                             p0=[amplitude_corrected, center, width],
                                             bounds=bounds,
                                             maxfev=5000,
                                             method='trf')
                    elif peak_shape == "Gaussian":
                        # Parameter bounds: [amplitude, center, width]
                        bounds = ([0, x_fit[0], 0.5], 
                                [amplitude_corrected * 3, x_fit[-1], 100.0])
                        popt, pcov = curve_fit(self.gaussian, x_fit, y_fit_corrected, 
                                             p0=[amplitude_corrected, center, width],
                                             bounds=bounds,
                                             maxfev=5000,
                                             method='trf')
                    elif peak_shape == "Voigt":
                        # Parameter bounds: [amplitude, center, sigma, gamma]
                        bounds = ([0, x_fit[0], 0.5, 0.1], 
                                [amplitude_corrected * 3, x_fit[-1], 100.0, 100.0])
                        popt, pcov = curve_fit(self.voigt, x_fit, y_fit_corrected, 
                                             p0=[amplitude_corrected, center, width, width/2],
                                             bounds=bounds,
                                             maxfev=5000,
                                             method='trf')
                    
                    # Add baseline back to amplitude
                    popt[0] += baseline
                    
                except Exception as fit_error:
                    # If bounded fitting fails, try without bounds but with better initial guess
                    print(f"Bounded fitting failed for peak at {peak_pos:.1f}, trying unbounded: {fit_error}")
                    try:
                        if peak_shape == "Lorentzian":
                            popt, pcov = curve_fit(self.lorentzian, x_fit, y_fit, 
                                                 p0=[amplitude, center, width],
                                                 maxfev=10000)
                        elif peak_shape == "Gaussian":
                            popt, pcov = curve_fit(self.gaussian, x_fit, y_fit, 
                                                 p0=[amplitude, center, width],
                                                 maxfev=10000)
                        elif peak_shape == "Voigt":
                            popt, pcov = curve_fit(self.voigt, x_fit, y_fit, 
                                                 p0=[amplitude, center, width, width/2],
                                                 maxfev=10000)
                    except Exception as final_error:
                        print(f"All fitting attempts failed for peak at {peak_pos:.1f}: {final_error}")
                        # Create a fallback result with initial estimates
                        if peak_shape == "Voigt":
                            popt = [amplitude, center, width, width/2]
                            pcov = np.eye(4) * 0.1  # Dummy covariance
                        else:
                            popt = [amplitude, center, width]
                            pcov = np.eye(3) * 0.1  # Dummy covariance
                
                # Calculate R-squared
                try:
                    if peak_shape == "Voigt":
                        y_pred = self.voigt(x_fit, *popt)
                    elif peak_shape == "Gaussian":
                        y_pred = self.gaussian(x_fit, *popt)
                    else:
                        y_pred = self.lorentzian(x_fit, *popt)
                    
                    ss_res = np.sum((y_fit - y_pred) ** 2)
                    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Ensure R-squared is reasonable
                    r_squared = max(0, min(1, r_squared))
                    
                except Exception as r2_error:
                    print(f"Error calculating R-squared for peak at {peak_pos:.1f}: {r2_error}")
                    r_squared = 0.0
                
                # Validate fitted parameters
                fitted_center = popt[1]
                fitted_amplitude = popt[0]
                fitted_width = popt[2]
                
                # Check if fitted parameters are reasonable
                center_deviation = abs(fitted_center - peak_pos)
                if center_deviation > 50.0:  # Center moved too far
                    print(f"Warning: Fitted center {fitted_center:.1f} deviates significantly from selected position {peak_pos:.1f}")
                    fitted_center = peak_pos  # Use original position
                    popt[1] = peak_pos
                
                if fitted_amplitude <= 0:
                    print(f"Warning: Invalid amplitude {fitted_amplitude:.3f} for peak at {peak_pos:.1f}")
                    fitted_amplitude = amplitude
                    popt[0] = amplitude
                
                if fitted_width <= 0 or fitted_width > 200:
                    print(f"Warning: Invalid width {fitted_width:.1f} for peak at {peak_pos:.1f}")
                    fitted_width = width
                    popt[2] = width
                
                # Store fitted parameters
                peak_data = {
                    'position': fitted_center,
                    'amplitude': fitted_amplitude,
                    'width': fitted_width,
                    'shape': peak_shape,
                    'parameters': popt,
                    'covariance': pcov,
                    'r_squared': r_squared,
                    'original_position': peak_pos,
                    'fit_quality': 'good' if r_squared > 0.8 else 'fair' if r_squared > 0.5 else 'poor'
                }
                
                if peak_shape == "Voigt":
                    peak_data['gamma'] = popt[3] if len(popt) > 3 else width/2
                
                self.fitted_peaks.append(peak_data)
            
            # Update the plot
            self.update_peak_fitting_plot()
            
            #messagebox.showinfo("Success", f"Successfully fitted {len(self.fitted_peaks)} peaks.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error fitting peaks: {str(e)}")
    
    def lorentzian(self, x, amplitude, center, width):
        """Lorentzian peak function."""
        # Ensure width is positive to avoid division by zero
        width = abs(width) + 1e-10
        return amplitude * (width**2) / ((x - center)**2 + width**2)
    
    def gaussian(self, x, amplitude, center, width):
        """Gaussian peak function."""
        # Ensure width is positive to avoid division by zero
        width = abs(width) + 1e-10
        return amplitude * np.exp(-0.5 * ((x - center) / width)**2)
    
    def voigt(self, x, amplitude, center, sigma, gamma):
        """Voigt peak function using Faddeeva function."""
        try:
            from scipy.special import wofz
            # Ensure positive parameters
            sigma = abs(sigma) + 1e-10
            gamma = abs(gamma) + 1e-10
            
            # Voigt profile calculation
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
            
            # Handle potential numerical issues
            profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
            
            return amplitude * profile
            
        except ImportError:
            # Fallback to pseudo-Voigt approximation if wofz not available
            return self.pseudo_voigt(x, amplitude, center, sigma, gamma)
    
    def pseudo_voigt(self, x, amplitude, center, sigma, gamma):
        """Pseudo-Voigt approximation (linear combination of Gaussian and Lorentzian)."""
        # Mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)
        eta = gamma / (gamma + sigma)
        eta = np.clip(eta, 0, 1)  # Ensure eta is between 0 and 1
        
        # Effective width for both components
        fwhm = 2 * np.sqrt(sigma**2 + gamma**2)
        width_g = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Gaussian width
        width_l = fwhm / 2  # Lorentzian width
        
        gaussian_part = np.exp(-0.5 * ((x - center) / width_g)**2)
        lorentzian_part = (width_l**2) / ((x - center)**2 + width_l**2)
        
        return amplitude * ((1 - eta) * gaussian_part + eta * lorentzian_part)
    
    def auto_assign_characters(self):
        """Automatically assign vibrational mode characters to fitted peaks."""
        if not self.fitted_peaks:
            messagebox.showwarning("No Peaks", "Please fit peaks first.")
            return
        
        reference_mineral = self.reference_mineral_var.get()
        if not reference_mineral:
            messagebox.showwarning("No Reference", "Please select a reference mineral.")
            return
        
        if reference_mineral not in self.mineral_database:
            messagebox.showerror("Error", f"Reference mineral '{reference_mineral}' not found in database.")
            return
        
        try:
            # Get reference mineral modes
            mineral_data = self.mineral_database[reference_mineral]
            if 'modes' not in mineral_data:
                messagebox.showerror("Error", f"No modes data found for {reference_mineral}.")
                return
            
            modes = mineral_data['modes']
            self.peak_assignments = {}
            
            # For each fitted peak, find the closest theoretical mode
            for peak in self.fitted_peaks:
                peak_pos = peak['position']
                best_match = None
                min_distance = float('inf')
                
                for mode in modes:
                    if isinstance(mode, (tuple, list)) and len(mode) >= 3:
                        mode_freq = float(mode[0])
                        mode_char = str(mode[1])
                        mode_intensity = float(mode[2])
                        
                        # Skip very low frequency modes
                        if mode_freq < 50:
                            continue
                        
                        # Calculate distance (allow for reasonable frequency shifts)
                        distance = abs(peak_pos - mode_freq)
                        
                        # Only consider matches within reasonable range (±200 cm⁻¹)
                        if distance < 200 and distance < min_distance:
                            min_distance = distance
                            best_match = {
                                'frequency': mode_freq,
                                'character': mode_char,
                                'intensity': mode_intensity,
                                'distance': distance
                            }
                
                if best_match:
                    self.peak_assignments[peak_pos] = best_match
            
            #messagebox.showinfo("Success", f"Assigned characters to {len(self.peak_assignments)} peaks.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error assigning characters: {str(e)}")
    
    def calculate_frequency_shifts(self):
        """Calculate frequency shifts between experimental and theoretical peaks."""
        if not self.peak_assignments:
            messagebox.showwarning("No Assignments", "Please assign characters first.")
            return
        
        try:
            self.frequency_shifts = {}
            
            for peak_pos, assignment in self.peak_assignments.items():
                theoretical_freq = assignment['frequency']
                experimental_freq = peak_pos
                shift = experimental_freq - theoretical_freq
                
                self.frequency_shifts[peak_pos] = {
                    'experimental': experimental_freq,
                    'theoretical': theoretical_freq,
                    'shift': shift,
                    'character': assignment['character']
                }
            
            # Update the plot to show shifts
            self.update_peak_fitting_plot()
            
            #messagebox.showinfo("Success", f"Calculated frequency shifts for {len(self.frequency_shifts)} peaks.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating frequency shifts: {str(e)}")
    
    def generate_shifted_spectrum(self):
        """Generate a calculated spectrum with peaks shifted to match experimental positions."""
        if not self.frequency_shifts:
            messagebox.showwarning("No Shifts", "Please calculate frequency shifts first.")
            return
        
        reference_mineral = self.reference_mineral_var.get()
        if not reference_mineral or reference_mineral not in self.mineral_database:
            messagebox.showerror("Error", "Invalid reference mineral.")
            return
        
        try:
            # Get the wavenumber range from current spectrum
            if self.current_spectrum is not None:
                wavenumbers = self.current_spectrum['wavenumbers']
                x_range = (np.min(wavenumbers), np.max(wavenumbers))
            else:
                x_range = (100, 4000)
            
            # Generate spectrum with shifted peaks
            wavenumbers_calc = np.linspace(x_range[0], x_range[1], 2000)
            intensities_calc = np.zeros_like(wavenumbers_calc)
            
            # Get all modes from reference mineral
            mineral_data = self.mineral_database[reference_mineral]
            modes = mineral_data['modes']
            
            # Calculate average frequency shift for unassigned peaks
            assigned_shifts = [data['shift'] for data in self.frequency_shifts.values()]
            avg_shift = np.mean(assigned_shifts) if assigned_shifts else 0.0
            
            # Generate peaks with appropriate shifts
            for mode in modes:
                if isinstance(mode, (tuple, list)) and len(mode) >= 3:
                    mode_freq = float(mode[0])
                    mode_char = str(mode[1])
                    mode_intensity = float(mode[2])
                    
                    if mode_freq < 50:  # Skip very low frequency modes
                        continue
                    
                    # Determine shift to apply
                    shift_to_apply = avg_shift  # Default to average shift
                    
                    # Check if this mode is assigned to a fitted peak
                    for peak_pos, shift_data in self.frequency_shifts.items():
                        if shift_data['character'] == mode_char:
                            # Use the specific shift for this character
                            shift_to_apply = shift_data['shift']
                            break
                    
                    # Apply shift to theoretical frequency
                    shifted_freq = mode_freq + shift_to_apply
                    
                    # Only include peaks within the wavenumber range
                    if x_range[0] <= shifted_freq <= x_range[1]:
                        # Generate Lorentzian peak
                        width = 8.0  # Peak width
                        peak = mode_intensity * (width**2) / ((wavenumbers_calc - shifted_freq)**2 + width**2)
                        intensities_calc += peak
            
            # Normalize the calculated spectrum
            if np.max(intensities_calc) > 0:
                intensities_calc = intensities_calc / np.max(intensities_calc)
            
            # Store as imported spectrum for comparison
            self.imported_spectrum = {
                'wavenumbers': wavenumbers_calc,
                'intensities': intensities_calc,
                'name': f"{reference_mineral} (shifted)"
            }
            
            # Update both plots
            self.update_spectrum_plot()
            self.update_peak_fitting_plot()
            
            #messagebox.showinfo("Success", 
            #                  f"Generated shifted spectrum for {reference_mineral}\n"
            #                  f"Applied average shift: {avg_shift:+.1f} cm⁻¹\n"
            #                  f"Specific shifts applied to {len(self.frequency_shifts)} assigned modes")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating shifted spectrum: {str(e)}")
    
    def show_fit_parameters(self):
        """Display fitted peak parameters in a new window."""
        if not self.fitted_peaks:
            messagebox.showwarning("No Data", "No fitted peaks available.")
            return
        
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Peak Fitting Results")
        results_window.geometry("800x600")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(results_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate report
        report = "PEAK FITTING RESULTS\n"
        report += "=" * 50 + "\n\n"
        
        for i, peak in enumerate(self.fitted_peaks, 1):
            report += f"Peak {i}:\n"
            report += f"  Position: {peak['position']:.2f} cm⁻¹\n"
            report += f"  Amplitude: {peak['amplitude']:.3f}\n"
            report += f"  Width: {peak['width']:.2f} cm⁻¹\n"
            report += f"  Shape: {peak['shape']}\n"
            report += f"  R²: {peak['r_squared']:.4f}\n"
            report += f"  Fit Quality: {peak.get('fit_quality', 'unknown')}\n"
            
            if peak['shape'] == "Voigt":
                report += f"  Gamma: {peak.get('gamma', 0):.2f} cm⁻¹\n"
            
            # Show deviation from original position if significant
            deviation = abs(peak['position'] - peak['original_position'])
            if deviation > 1.0:
                report += f"  Position Shift: {peak['position'] - peak['original_position']:+.2f} cm⁻¹\n"
            
            # Add assignment if available
            if peak['position'] in self.peak_assignments:
                assignment = self.peak_assignments[peak['position']]
                report += f"  Character: {assignment['character']}\n"
                report += f"  Theoretical: {assignment['frequency']:.2f} cm⁻¹\n"
            
            # Add frequency shift if available
            if peak['position'] in self.frequency_shifts:
                shift_data = self.frequency_shifts[peak['position']]
                report += f"  Frequency Shift: {shift_data['shift']:+.2f} cm⁻¹\n"
            
            report += "\n"
        
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
    
    def show_peak_assignments(self):
        """Display peak character assignments in a new window."""
        if not self.peak_assignments:
            messagebox.showwarning("No Data", "No peak assignments available.")
            return
        
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Peak Character Assignments")
        results_window.geometry("600x400")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(results_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate report
        report = "PEAK CHARACTER ASSIGNMENTS\n"
        report += "=" * 40 + "\n\n"
        report += f"Reference Mineral: {self.reference_mineral_var.get()}\n\n"
        
        for peak_pos, assignment in self.peak_assignments.items():
            report += f"Peak at {peak_pos:.2f} cm⁻¹:\n"
            report += f"  Character: {assignment['character']}\n"
            report += f"  Theoretical: {assignment['frequency']:.2f} cm⁻¹\n"
            report += f"  Distance: {assignment['distance']:.2f} cm⁻¹\n"
            
            if peak_pos in self.frequency_shifts:
                shift = self.frequency_shifts[peak_pos]['shift']
                report += f"  Frequency Shift: {shift:+.2f} cm⁻¹\n"
            
            report += "\n"
        
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
    
    def export_fit_results(self):
        """Export fitting results to a CSV file."""
        if not self.fitted_peaks:
            messagebox.showwarning("No Data", "No fitted peaks available.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Fitting Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as csvfile:
                    import csv
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    header = ['Peak_Number', 'Position_cm-1', 'Amplitude', 'Width_cm-1', 
                             'Shape', 'R_squared', 'Character', 'Theoretical_cm-1', 
                             'Frequency_Shift_cm-1']
                    writer.writerow(header)
                    
                    # Write data
                    for i, peak in enumerate(self.fitted_peaks, 1):
                        row = [
                            i,
                            f"{peak['position']:.2f}",
                            f"{peak['amplitude']:.3f}",
                            f"{peak['width']:.2f}",
                            peak['shape'],
                            f"{peak['r_squared']:.4f}"
                        ]
                        
                        # Add assignment data if available
                        if peak['position'] in self.peak_assignments:
                            assignment = self.peak_assignments[peak['position']]
                            row.extend([
                                assignment['character'],
                                f"{assignment['frequency']:.2f}"
                            ])
                        else:
                            row.extend(['', ''])
                        
                        # Add frequency shift if available
                        if peak['position'] in self.frequency_shifts:
                            shift = self.frequency_shifts[peak['position']]['shift']
                            row.append(f"{shift:+.2f}")
                        else:
                            row.append('')
                        
                        writer.writerow(row)
                
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting results: {str(e)}")
    
    def load_polarized_spectra(self):
        """Load multiple polarized Raman spectra."""
        # Create dialog for loading polarized spectra
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Polarized Spectra")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Instructions
        ttk.Label(dialog, text="Load polarized Raman spectra:",
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Create notebook for different loading methods
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Load from files
        file_tab = ttk.Frame(notebook)
        notebook.add(file_tab, text="Load from Files")
        
        # Tab 2: Generate from database
        database_tab = ttk.Frame(notebook)
        notebook.add(database_tab, text="Generate from Database")
        
        # Setup file loading tab
        self.setup_file_loading_tab(file_tab, dialog)
        
        # Setup database generation tab
        self.setup_database_generation_tab(database_tab, dialog)
    
    def setup_file_loading_tab(self, parent, dialog):
        """Setup the file loading tab."""
        ttk.Label(parent, text="Load polarized Raman spectra from files:",
                 font=('Arial', 10)).pack(pady=10)
        
        # Frame for file selections
        files_frame = ttk.LabelFrame(parent, text="Polarization Configurations", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Dictionary to store file paths
        self.pol_files = {}
        
        # Common polarization configurations
        configurations = [
            ("XX (Parallel)", "xx"),
            ("XY (Cross-polarized)", "xy"),
            ("YX (Cross-polarized)", "yx"),
            ("YY (Parallel)", "yy"),
            ("ZZ (Parallel)", "zz"),
            ("XZ (Cross-polarized)", "xz"),
            ("ZX (Cross-polarized)", "zx"),
            ("YZ (Cross-polarized)", "yz"),
            ("ZY (Cross-polarized)", "zy")
        ]
        
        # Create file selection widgets
        self.pol_file_vars = {}
        self.pol_file_labels = {}
        
        for i, (display_name, config_key) in enumerate(configurations):
            frame = ttk.Frame(files_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{display_name}:", width=20).pack(side=tk.LEFT)
            
            self.pol_file_vars[config_key] = tk.StringVar()
            self.pol_file_labels[config_key] = ttk.Label(frame, textvariable=self.pol_file_vars[config_key], 
                                                        foreground="gray", width=30)
            self.pol_file_labels[config_key].pack(side=tk.LEFT, padx=5)
            
            ttk.Button(frame, text="Browse", 
                      command=lambda key=config_key: self.browse_pol_file(key)).pack(side=tk.RIGHT)
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Load Selected", command=lambda: self.load_selected_pol_files(dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def setup_database_generation_tab(self, parent, dialog):
        """Setup the database generation tab."""
        ttk.Label(parent, text="Generate polarized spectra from mineral database:",
                 font=('Arial', 10)).pack(pady=10)
        
        # Mineral selection frame
        mineral_frame = ttk.LabelFrame(parent, text="Mineral Selection", padding=10)
        mineral_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(mineral_frame, text="Select Mineral:").pack(anchor=tk.W)
        self.pol_mineral_var = tk.StringVar()
        
        # Use the selected reference mineral as default if available
        if self.selected_reference_mineral:
            self.pol_mineral_var.set(self.selected_reference_mineral)
        
        pol_mineral_combo = ttk.Combobox(mineral_frame, textvariable=self.pol_mineral_var,
                                        values=self.mineral_list, state="readonly")
        pol_mineral_combo.pack(fill=tk.X, pady=2)
        pol_mineral_combo.bind('<<ComboboxSelected>>', self.on_pol_mineral_changed)
        
        # Status label
        self.pol_mineral_status = ttk.Label(mineral_frame, text="", foreground="gray", font=('Arial', 8))
        self.pol_mineral_status.pack(anchor=tk.W, pady=2)
        
        # Configuration selection frame
        config_frame = ttk.LabelFrame(parent, text="Polarization Configurations", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(config_frame, text="Select configurations to generate:").pack(anchor=tk.W)
        
        # Create a frame for the checkboxes to use grid layout
        checkbox_frame = ttk.Frame(config_frame)
        checkbox_frame.pack(fill=tk.X, pady=5)
        
        # Checkboxes for different configurations
        self.pol_config_vars = {}
        configurations = [
            ("XX (Parallel)", "xx"),
            ("XY (Cross-polarized)", "xy"),
            ("YX (Cross-polarized)", "yx"),
            ("YY (Parallel)", "yy"),
            ("ZZ (Parallel)", "zz"),
            ("XZ (Cross-polarized)", "xz"),
            ("ZX (Cross-polarized)", "zx"),
            ("YZ (Cross-polarized)", "yz"),
            ("ZY (Cross-polarized)", "zy")
        ]
        
        # Create checkboxes in a grid
        for i, (display_name, config_key) in enumerate(configurations):
            row = i // 3
            col = i % 3
            
            var = tk.BooleanVar(value=(config_key in ['xx', 'xy', 'yy']))  # Default selection
            self.pol_config_vars[config_key] = var
            
            cb = ttk.Checkbutton(checkbox_frame, text=display_name, variable=var)
            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(parent, text="Generation Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Intensity scaling factor
        ttk.Label(params_frame, text="Intensity Scaling Factor:").pack(anchor=tk.W)
        self.intensity_scale_var = tk.DoubleVar(value=1.0)
        intensity_scale = ttk.Scale(params_frame, from_=0.1, to=5.0, variable=self.intensity_scale_var,
                                   orient=tk.HORIZONTAL, length=200)
        intensity_scale.pack(fill=tk.X, pady=2)
        
        self.intensity_scale_label = ttk.Label(params_frame, text="1.0")
        self.intensity_scale_label.pack(anchor=tk.W)
        
        def update_intensity_label(*args):
            self.intensity_scale_label.config(text=f"{self.intensity_scale_var.get():.1f}")
        self.intensity_scale_var.trace('w', update_intensity_label)
        
        # Peak width parameter
        ttk.Label(params_frame, text="Peak Width (cm⁻¹):").pack(anchor=tk.W, pady=(10, 0))
        self.peak_width_var = tk.DoubleVar(value=8.0)
        peak_width_scale = ttk.Scale(params_frame, from_=2.0, to=20.0, variable=self.peak_width_var,
                                    orient=tk.HORIZONTAL, length=200)
        peak_width_scale.pack(fill=tk.X, pady=2)
        
        self.peak_width_label = ttk.Label(params_frame, text="8.0")
        self.peak_width_label.pack(anchor=tk.W)
        
        def update_width_label(*args):
            self.peak_width_label.config(text=f"{self.peak_width_var.get():.1f}")
        self.peak_width_var.trace('w', update_width_label)
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Generate Spectra", command=lambda: self.generate_polarized_spectra(dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Preview", command=self.preview_polarized_generation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Update status if mineral is already selected
        if self.selected_reference_mineral:
            self.update_pol_mineral_status()
    
    def on_pol_mineral_changed(self, event=None):
        """Handle changes to the polarization mineral selection."""
        self.update_pol_mineral_status()
    
    def update_pol_mineral_status(self):
        """Update the status label for the selected mineral."""
        mineral_name = self.pol_mineral_var.get()
        if not mineral_name or mineral_name not in self.mineral_database:
            self.pol_mineral_status.config(text="No mineral selected", foreground="gray")
            return
        
        mineral_data = self.mineral_database[mineral_name]
        
        # Check what data is available
        has_dielectric = any(key in mineral_data for key in ['dielectric_tensor', 'dielectric_tensors', 'n_tensor'])
        has_modes = 'modes' in mineral_data and mineral_data['modes']
        
        if has_dielectric and has_modes:
            self.pol_mineral_status.config(
                text=f"✓ Dielectric tensor and {len(mineral_data['modes'])} modes available",
                foreground="green"
            )
        elif has_modes:
            self.pol_mineral_status.config(
                text=f"✓ {len(mineral_data['modes'])} modes available (no dielectric tensor)",
                foreground="orange"
            )
        else:
            self.pol_mineral_status.config(
                text="✗ No suitable data available for polarization generation",
                foreground="red"
            )
    
    def preview_polarized_generation(self):
        """Preview the polarized spectrum generation."""
        mineral_name = self.pol_mineral_var.get()
        if not mineral_name:
            messagebox.showwarning("No Mineral", "Please select a mineral first.")
            return
        
        # Get selected configurations
        selected_configs = [config for config, var in self.pol_config_vars.items() if var.get()]
        if not selected_configs:
            messagebox.showwarning("No Configurations", "Please select at least one polarization configuration.")
            return
        
        try:
            # Generate preview for first selected configuration
            preview_config = selected_configs[0]
            wavenumbers, intensities = self.generate_polarized_spectrum_from_database(
                mineral_name, preview_config
            )
            
            if wavenumbers is not None and intensities is not None:
                # Create preview window
                preview_window = tk.Toplevel(self.root)
                preview_window.title(f"Preview: {mineral_name} - {preview_config.upper()}")
                preview_window.geometry("600x400")
                
                # Create matplotlib figure for preview
                from matplotlib.figure import Figure
                fig = Figure(figsize=(8, 6), dpi=80)
                ax = fig.add_subplot(111)
                
                ax.plot(wavenumbers, intensities, 'b-', linewidth=2, 
                       label=f"{mineral_name} ({preview_config.upper()})")
                
                ax.set_xlabel('Wavenumber (cm⁻¹)')
                ax.set_ylabel('Intensity (a.u.)')
                ax.set_title(f'Preview: Polarized Spectrum Generation')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                canvas = FigureCanvasTkAgg(fig, preview_window)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Add info label
                info_text = f"Configuration: {preview_config.upper()}\nPeak width: {self.peak_width_var.get():.1f} cm⁻¹\nIntensity scale: {self.intensity_scale_var.get():.1f}"
                ttk.Label(preview_window, text=info_text, font=('Arial', 9)).pack(pady=5)
            else:
                messagebox.showerror("Error", f"Could not generate preview for {mineral_name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error generating preview: {str(e)}")
    
    def generate_polarized_spectra(self, dialog):
        """Generate polarized spectra from the mineral database."""
        mineral_name = self.pol_mineral_var.get()
        if not mineral_name:
            messagebox.showwarning("No Mineral", "Please select a mineral first.")
            return
        
        # Get selected configurations
        selected_configs = [config for config, var in self.pol_config_vars.items() if var.get()]
        if not selected_configs:
            messagebox.showwarning("No Configurations", "Please select at least one polarization configuration.")
            return
        
        try:
            self.polarization_data = {}
            
            for config in selected_configs:
                wavenumbers, intensities = self.generate_polarized_spectrum_from_database(
                    mineral_name, config
                )
                
                if wavenumbers is not None and intensities is not None:
                    self.polarization_data[config] = {
                        'wavenumbers': wavenumbers,
                        'intensities': intensities,
                        'filename': f"{mineral_name}_{config.upper()}_generated"
                    }
            
            if self.polarization_data:
                # Update configuration label
                configs = list(self.polarization_data.keys())
                self.pol_config_label.config(
                    text=f"Generated: {', '.join([c.upper() for c in configs])} from {mineral_name}", 
                    foreground="blue"
                )
                
                # Update plot
                self.update_polarization_plot()
                
                dialog.destroy()
                #messagebox.showinfo("Success", 
                #                  f"Generated {len(self.polarization_data)} polarized spectra from {mineral_name}.")
            else:
                messagebox.showerror("Error", "No valid polarized spectra could be generated.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error generating polarized spectra: {str(e)}")
    
    def generate_polarized_spectrum_from_database(self, mineral_name, polarization_config):
        """Generate a polarized Raman spectrum from mineral database data."""
        try:
            if mineral_name not in self.mineral_database:
                return None, None
            
            mineral_data = self.mineral_database[mineral_name]
            
            # Get modes data
            if 'modes' not in mineral_data or not mineral_data['modes']:
                return None, None
            
            modes = mineral_data['modes']
            
            # Create wavenumber range
            wavenumbers = np.linspace(100, 4000, 2000)
            intensities = np.zeros_like(wavenumbers)
            
            # Get polarization scaling factors from dielectric tensor if available
            polarization_factor = self.get_polarization_factor(mineral_data, polarization_config)
            
            # Get generation parameters
            peak_width = self.peak_width_var.get()
            intensity_scale = self.intensity_scale_var.get()
            
            # Process each mode
            for mode in modes:
                if isinstance(mode, (tuple, list)) and len(mode) >= 3:
                    frequency = float(mode[0])  # Frequency in cm^-1
                    symmetry = str(mode[1])     # Symmetry character
                    base_intensity = float(mode[2])  # Relative intensity
                    
                    # Skip modes with zero or very low frequency
                    if frequency < 50:
                        continue
                    
                    # Apply polarization-dependent intensity scaling
                    polarized_intensity = base_intensity * polarization_factor * intensity_scale
                    
                    # Apply symmetry-dependent polarization rules
                    polarized_intensity *= self.get_symmetry_polarization_factor(symmetry, polarization_config)
                    
                    # Generate Lorentzian peak
                    peak = polarized_intensity * (peak_width**2) / ((wavenumbers - frequency)**2 + peak_width**2)
                    intensities += peak
            
            # Normalize intensities
            if np.max(intensities) > 0:
                intensities = intensities / np.max(intensities)
            
            # Add some realistic noise
            noise_level = 0.02
            intensities += noise_level * np.random.randn(len(intensities))
            intensities = np.maximum(intensities, 0)  # Ensure non-negative
            
            return wavenumbers, intensities
            
        except Exception as e:
            print(f"Error generating polarized spectrum for {mineral_name} ({polarization_config}): {e}")
            return None, None
    
    def get_polarization_factor(self, mineral_data, polarization_config):
        """Get polarization factor from dielectric tensor data."""
        try:
            # Check for different tensor data formats
            tensor_data = None
            
            if 'dielectric_tensor' in mineral_data:
                tensor_data = mineral_data['dielectric_tensor']
            elif 'n_tensor' in mineral_data:
                tensor_data = mineral_data['n_tensor']
            elif 'dielectric_tensors' in mineral_data:
                tensor_data = mineral_data['dielectric_tensors']
            
            if tensor_data is None:
                return 1.0  # Default factor if no tensor data
            
            # Extract tensor elements based on polarization configuration
            if polarization_config == 'xx':
                return self.extract_tensor_element(tensor_data, 'xx', default=1.0)
            elif polarization_config == 'yy':
                return self.extract_tensor_element(tensor_data, 'yy', default=1.0)
            elif polarization_config == 'zz':
                return self.extract_tensor_element(tensor_data, 'zz', default=1.0)
            elif polarization_config in ['xy', 'yx']:
                return self.extract_tensor_element(tensor_data, 'xy', default=0.5)
            elif polarization_config in ['xz', 'zx']:
                return self.extract_tensor_element(tensor_data, 'xz', default=0.5)
            elif polarization_config in ['yz', 'zy']:
                return self.extract_tensor_element(tensor_data, 'yz', default=0.5)
            
            return 1.0
            
        except Exception as e:
            print(f"Error extracting polarization factor: {e}")
            return 1.0
    
    def extract_tensor_element(self, tensor_data, component, default=1.0):
        """Extract a specific tensor element from various tensor data formats."""
        try:
            # Handle numpy array format
            if isinstance(tensor_data, np.ndarray):
                if tensor_data.shape == (3, 3):
                    component_map = {
                        'xx': (0, 0), 'yy': (1, 1), 'zz': (2, 2),
                        'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)
                    }
                    if component in component_map:
                        i, j = component_map[component]
                        return abs(tensor_data[i, j])
            
            # Handle dictionary format
            elif isinstance(tensor_data, dict):
                if 'tensor' in tensor_data:
                    tensor_list = tensor_data['tensor']
                    for element in tensor_list:
                        if isinstance(element, list) and len(element) >= 3:
                            if element[1] == component:
                                return abs(float(element[2]))
            
            # Handle list format
            elif isinstance(tensor_data, list):
                for element in tensor_data:
                    if isinstance(element, dict):
                        if element.get('Component') == component:
                            return abs(float(element.get('X', default)))
                    elif isinstance(element, list) and len(element) >= 3:
                        if element[1] == component:
                            return abs(float(element[2]))
            
            return default
            
        except Exception as e:
            print(f"Error extracting tensor element {component}: {e}")
            return default
    
    def get_symmetry_polarization_factor(self, symmetry, polarization_config):
        """Get polarization factor based on vibrational mode symmetry."""
        # Simplified polarization rules based on symmetry
        symmetry_lower = symmetry.lower()
        
        # Totally symmetric modes (A1, A1g) are strongly polarized
        if any(sym in symmetry_lower for sym in ['a1', 'ag']):
            if polarization_config in ['xx', 'yy', 'zz']:
                return 1.0  # Strong in parallel configurations
            else:
                return 0.1  # Weak in cross-polarized configurations
        
        # Doubly degenerate modes (E, Eg) show intermediate polarization
        elif any(sym in symmetry_lower for sym in ['e', 'eg']):
            if polarization_config in ['xx', 'yy', 'zz']:
                return 0.7
            else:
                return 0.5
        
        # Triply degenerate modes (T, T2g) are typically depolarized
        elif any(sym in symmetry_lower for sym in ['t', 't2', 't2g']):
            if polarization_config in ['xx', 'yy', 'zz']:
                return 0.5
            else:
                return 0.8
        
        # Default case
        return 0.6
    
    def browse_pol_file(self, config_key):
        """Browse for a polarized spectrum file."""
        file_path = filedialog.askopenfilename(
            title=f"Select {config_key.upper()} Polarized Spectrum",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.pol_file_vars[config_key].set(os.path.basename(file_path))
            self.pol_files[config_key] = file_path
    
    def load_selected_pol_files(self, dialog):
        """Load the selected polarized spectrum files."""
        if not self.pol_files:
            messagebox.showwarning("No Files", "Please select at least one polarized spectrum file.")
            return
        
        try:
            self.polarization_data = {}
            
            for config_key, file_path in self.pol_files.items():
                # Load spectrum data
                try:
                    # Try to detect headers
                    with open(file_path, 'r') as f:
                        first_line = f.readline().strip()
                    
                    has_header = False
                    try:
                        parts = first_line.split()
                        if len(parts) >= 2:
                            float(parts[0])
                            float(parts[1])
                    except (ValueError, IndexError):
                        has_header = True
                    
                    # Load data
                    if has_header:
                        data = np.loadtxt(file_path, delimiter=None, skiprows=1)
                    else:
                        data = np.loadtxt(file_path, delimiter=None)
                    
                    if data.ndim == 2 and data.shape[1] >= 2:
                        wavenumbers = data[:, 0]
                        intensities = data[:, 1]
                        
                        self.polarization_data[config_key] = {
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'filename': os.path.basename(file_path)
                        }
                    
                except Exception as e:
                    print(f"Error loading {config_key} spectrum: {e}")
                    continue
            
            if self.polarization_data:
                # Update configuration label
                configs = list(self.polarization_data.keys())
                self.pol_config_label.config(text=f"Loaded: {', '.join([c.upper() for c in configs])}", 
                                           foreground="green")
                
                # Update plot
                self.update_polarization_plot()
                
                dialog.destroy()
                #messagebox.showinfo("Success", f"Loaded {len(self.polarization_data)} polarized spectra.")
            else:
                messagebox.showerror("Error", "No valid polarized spectra could be loaded.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading polarized spectra: {str(e)}")
    
    def load_angular_series(self):
        """Load angular dependence series of spectra."""
        messagebox.showinfo("Angular Series", "Angular series loading will be implemented in the next update.")
    
    def clear_polarization_data(self):
        """Clear all polarization data."""
        self.polarization_data = {}
        self.raman_tensors = {}
        self.depolarization_ratios = {}
        self.angular_data = {}
        
        self.pol_config_label.config(text="No data loaded", foreground="gray")
        
        # Clear plot
        if "Polarization Analysis" in self.plot_components:
            self.update_polarization_plot()
        
        #messagebox.showinfo("Success", "Polarization data cleared.")
    
    def calculate_depolarization_ratios(self):
        """Calculate depolarization ratios from polarized spectra."""
        if not self.polarization_data:
            messagebox.showwarning("No Data", "Please load polarized spectra first.")
            return
        
        try:
            # Check for required configurations
            required_configs = ['xx', 'xy']  # Minimum for depolarization ratio
            available_configs = list(self.polarization_data.keys())
            
            if not all(config in available_configs for config in required_configs):
                messagebox.showwarning("Insufficient Data", 
                                     f"Need at least XX and XY configurations for depolarization ratio calculation.\n"
                                     f"Available: {', '.join([c.upper() for c in available_configs])}")
                return
            
            # Get common wavenumber range
            wavenumbers_xx = self.polarization_data['xx']['wavenumbers']
            wavenumbers_xy = self.polarization_data['xy']['wavenumbers']
            
            # Interpolate to common grid with higher resolution
            common_wavenumbers = np.linspace(
                max(np.min(wavenumbers_xx), np.min(wavenumbers_xy)),
                min(np.max(wavenumbers_xx), np.max(wavenumbers_xy)),
                2000  # Higher resolution
            )
            
            # Interpolate intensities
            I_xx = np.interp(common_wavenumbers, wavenumbers_xx, self.polarization_data['xx']['intensities'])
            I_xy = np.interp(common_wavenumbers, wavenumbers_xy, self.polarization_data['xy']['intensities'])
            
            # Smooth the data to reduce noise before ratio calculation
            from scipy.signal import savgol_filter
            window_length = min(51, len(I_xx) // 10)  # Adaptive window length
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(5, window_length)  # Minimum window size
            
            if len(I_xx) > window_length:
                I_xx_smooth = savgol_filter(I_xx, window_length, 3)
                I_xy_smooth = savgol_filter(I_xy, window_length, 3)
            else:
                I_xx_smooth = I_xx
                I_xy_smooth = I_xy
            
            # Calculate depolarization ratio ρ = I_xy / I_xx
            # Use a more sophisticated approach to handle division by zero
            threshold = np.max(I_xx_smooth) * 0.01  # 1% of maximum intensity
            
            # Create mask for valid data points
            valid_mask = I_xx_smooth > threshold
            
            # Initialize ratio array
            depol_ratio = np.zeros_like(I_xx_smooth)
            
            # Calculate ratio only for valid points
            depol_ratio[valid_mask] = I_xy_smooth[valid_mask] / I_xx_smooth[valid_mask]
            
            # For invalid points, use interpolation from nearby valid points
            if np.any(~valid_mask):
                valid_indices = np.where(valid_mask)[0]
                invalid_indices = np.where(~valid_mask)[0]
                
                if len(valid_indices) > 1:
                    depol_ratio[invalid_indices] = np.interp(
                        invalid_indices, valid_indices, depol_ratio[valid_indices]
                    )
            
            # Clip ratio to reasonable range (0 to 2, theoretical max is 0.75 for Raman)
            depol_ratio = np.clip(depol_ratio, 0, 2)
            
            # Apply additional smoothing to the ratio
            if len(depol_ratio) > window_length:
                depol_ratio = savgol_filter(depol_ratio, window_length, 2)
            
            # Store results
            self.depolarization_ratios = {
                'wavenumbers': common_wavenumbers,
                'ratio': depol_ratio,
                'I_xx': I_xx_smooth,
                'I_xy': I_xy_smooth,
                'I_xx_raw': I_xx,
                'I_xy_raw': I_xy,
                'valid_mask': valid_mask
            }
            
            # Update plot if depolarization ratios are selected
            if self.plot_type_var.get() == "Depolarization Ratios":
                self.update_polarization_plot()
            
            #messagebox.showinfo("Success", "Depolarization ratios calculated successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating depolarization ratios: {str(e)}")
    
    def determine_raman_tensors(self):
        """Determine Raman tensor elements from polarized measurements."""
        if not self.polarization_data:
            messagebox.showwarning("No Data", "Please load polarized spectra first.")
            return
        
        try:
            # This is a simplified implementation
            # Full tensor determination requires more polarization configurations
            available_configs = list(self.polarization_data.keys())
            
            # Check for minimum required configurations
            if len(available_configs) < 2:
                messagebox.showwarning("Insufficient Data", 
                                     "Need at least 2 polarization configurations for tensor analysis.")
                return
            
            # Get reference wavenumbers from first configuration
            ref_config = available_configs[0]
            wavenumbers = self.polarization_data[ref_config]['wavenumbers']
            
            # Initialize tensor elements storage
            self.raman_tensors = {
                'wavenumbers': wavenumbers,
                'tensor_elements': {},
                'configurations': available_configs
            }
            
            # For each configuration, store the intensity as a tensor element
            for config in available_configs:
                # Interpolate to common wavenumber grid
                intensities = np.interp(wavenumbers, 
                                      self.polarization_data[config]['wavenumbers'],
                                      self.polarization_data[config]['intensities'])
                
                self.raman_tensors['tensor_elements'][config] = intensities
            
            # Calculate some derived quantities
            if 'xx' in available_configs and 'yy' in available_configs:
                # Isotropic component (for cubic crystals)
                I_iso = (self.raman_tensors['tensor_elements']['xx'] + 
                        self.raman_tensors['tensor_elements']['yy']) / 2
                self.raman_tensors['tensor_elements']['isotropic'] = I_iso
            
            if 'xx' in available_configs and 'xy' in available_configs:
                # Anisotropic component
                I_aniso = self.raman_tensors['tensor_elements']['xx'] - self.raman_tensors['tensor_elements']['xy']
                self.raman_tensors['tensor_elements']['anisotropic'] = I_aniso
            
            #messagebox.showinfo("Success", f"Raman tensor analysis completed for {len(available_configs)} configurations.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error determining Raman tensors: {str(e)}")
    
    def angular_dependence_analysis(self):
        """Analyze angular dependence of Raman intensities."""
        messagebox.showinfo("Angular Analysis", "Angular dependence analysis will be implemented in the next update.")
    
    def symmetry_classification(self):
        """Classify vibrational modes based on polarization behavior."""
        if not self.depolarization_ratios:
            messagebox.showwarning("No Data", "Please calculate depolarization ratios first.")
            return
        
        try:
            crystal_system = self.crystal_system_var.get()
            
            # Create results window
            results_window = tk.Toplevel(self.root)
            results_window.title("Symmetry Classification")
            results_window.geometry("600x500")
            
            # Create text widget
            text_frame = ttk.Frame(results_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Generate symmetry analysis report
            report = "SYMMETRY CLASSIFICATION ANALYSIS\n"
            report += "=" * 50 + "\n\n"
            report += f"Crystal System: {crystal_system}\n"
            report += f"Scattering Geometry: {self.scattering_geometry_var.get()}\n\n"
            
            # Analyze depolarization ratios
            wavenumbers = self.depolarization_ratios['wavenumbers']
            ratios = self.depolarization_ratios['ratio']
            
            # Find peaks in the spectra for analysis
            I_xx = self.depolarization_ratios['I_xx']
            peaks, _ = find_peaks(I_xx, prominence=np.max(I_xx) * 0.1, distance=20)
            
            report += f"Found {len(peaks)} significant peaks for analysis:\n\n"
            
            for i, peak_idx in enumerate(peaks):
                wavenumber = wavenumbers[peak_idx]
                ratio = ratios[peak_idx]
                
                # Classify based on depolarization ratio
                if ratio < 0.1:
                    symmetry = "Totally symmetric (A1, A1g)"
                    polarized = "Polarized"
                elif 0.1 <= ratio < 0.5:
                    symmetry = "Partially polarized"
                    polarized = "Partially polarized"
                elif 0.5 <= ratio < 0.75:
                    symmetry = "Depolarized (E, T2, etc.)"
                    polarized = "Depolarized"
                else:
                    symmetry = "Highly depolarized"
                    polarized = "Highly depolarized"
                
                report += f"Peak {i+1}: {wavenumber:.1f} cm⁻¹\n"
                report += f"  Depolarization ratio: {ratio:.3f}\n"
                report += f"  Classification: {symmetry}\n"
                report += f"  Polarization: {polarized}\n\n"
            
            # Add theoretical background
            report += "\nTHEORETICAL BACKGROUND:\n"
            report += "-" * 30 + "\n"
            report += "Depolarization ratio ρ = I_⊥ / I_∥\n\n"
            report += "For different symmetries:\n"
            report += "• A1 (totally symmetric): ρ ≈ 0 (polarized)\n"
            report += "• E (doubly degenerate): ρ = 3/4 (depolarized)\n"
            report += "• T2 (triply degenerate): ρ = 3/4 (depolarized)\n\n"
            
            if crystal_system != "Unknown":
                report += f"Expected symmetries for {crystal_system} system:\n"
                symmetries = self.get_expected_symmetries(crystal_system)
                for sym in symmetries:
                    report += f"• {sym}\n"
            
            text_widget.insert(tk.END, report)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in symmetry classification: {str(e)}")
    
    def get_expected_symmetries(self, crystal_system):
        """Get expected symmetries for different crystal systems."""
        symmetries = {
            "Cubic": ["A1g", "Eg", "T1g", "T2g", "A1u", "Eu", "T1u", "T2u"],
            "Tetragonal": ["A1g", "A2g", "B1g", "B2g", "Eg", "A1u", "A2u", "B1u", "B2u", "Eu"],
            "Orthorhombic": ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
            "Hexagonal": ["A1g", "A2g", "B1g", "B2g", "E1g", "E2g", "A1u", "A2u", "B1u", "B2u", "E1u", "E2u"],
            "Trigonal": ["A1g", "A2g", "Eg", "A1u", "A2u", "Eu"],
            "Monoclinic": ["Ag", "Bg", "Au", "Bu"],
            "Triclinic": ["Ag", "Au"]
        }
        return symmetries.get(crystal_system, ["Unknown"])
    
    def update_polarization_plot(self, event=None):
        """Update the polarization analysis plot."""
        if "Polarization Analysis" not in self.plot_components:
            return
        
        components = self.plot_components["Polarization Analysis"]
        ax = components['ax']
        canvas = components['canvas']
        
        # Clear the plot
        ax.clear()
        
        plot_type = self.plot_type_var.get()
        
        if plot_type == "Polarized Spectra" and self.polarization_data:
            # Plot all loaded polarized spectra
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
            
            for i, (config, data) in enumerate(self.polarization_data.items()):
                color = colors[i % len(colors)]
                ax.plot(data['wavenumbers'], data['intensities'], 
                       color=color, linewidth=2, label=f"{config.upper()}", alpha=0.8)
            
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title("Polarized Raman Spectra")
            ax.legend()
            
        elif plot_type == "Depolarization Ratios":
            if self.depolarization_ratios:
                # Plot depolarization ratios
                wavenumbers = self.depolarization_ratios['wavenumbers']
                ratios = self.depolarization_ratios['ratio']
                
                ax.plot(wavenumbers, ratios, 'b-', linewidth=2, label='ρ = I_⊥ / I_∥')
                
                # Add horizontal lines for reference
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Fully polarized')
                ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Fully depolarized (theory)')
                ax.axhline(y=0.1, color='green', linestyle=':', alpha=0.5, label='Polarized threshold')
                
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Depolarization Ratio")
                ax.set_title("Depolarization Ratio Analysis")
                ax.set_ylim(0, 1.2)
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'Calculate depolarization ratios first\n(requires XX and XY configurations)', 
                       transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, alpha=0.6)
                ax.set_title("Depolarization Ratio Analysis")
            
        elif plot_type == "Angular Dependence":
            if self.polarization_data:
                # Create angular dependence plot
                self.plot_angular_dependence(ax)
            else:
                ax.text(0.5, 0.5, 'Load polarized spectra first', 
                       transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, alpha=0.6)
                ax.set_title("Angular Dependence Analysis")
                
        elif plot_type == "Polar Plot":
            if self.polarization_data:
                # Create polar plot
                self.plot_polar_diagram(ax)
            else:
                ax.text(0.5, 0.5, 'Load polarized spectra first', 
                       transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, alpha=0.6)
                ax.set_title("Polar Plot Analysis")
            
        elif plot_type == "Raman Tensor Elements" and self.raman_tensors:
            # Plot tensor elements
            wavenumbers = self.raman_tensors['wavenumbers']
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (element, intensities) in enumerate(self.raman_tensors['tensor_elements'].items()):
                if element not in ['isotropic', 'anisotropic']:  # Skip derived quantities for now
                    color = colors[i % len(colors)]
                    ax.plot(wavenumbers, intensities, color=color, linewidth=2, 
                           label=f"R_{element}", alpha=0.8)
            
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Raman Tensor Element")
            ax.set_title("Raman Tensor Elements")
            ax.legend()
            
        else:
            # Show instruction text
            ax.text(0.5, 0.5, f'Load polarized spectra to view {plot_type.lower()}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=12, alpha=0.6)
            ax.set_title("Polarization Analysis")
        
        ax.grid(True, alpha=0.3)
        canvas.draw()
    
    def plot_angular_dependence(self, ax):
        """Plot angular dependence of polarized intensities."""
        try:
            # Get available configurations
            configs = list(self.polarization_data.keys())
            
            # Define angles for different configurations (simplified)
            config_angles = {
                'xx': 0,    # 0 degrees
                'xy': 45,   # 45 degrees  
                'yx': 45,   # 45 degrees (equivalent to xy)
                'yy': 90,   # 90 degrees
                'zz': 0,    # Along z-axis (0 degrees for this representation)
                'xz': 30,   # 30 degrees
                'zx': 30,   # 30 degrees (equivalent to xz)
                'yz': 60,   # 60 degrees
                'zy': 60    # 60 degrees (equivalent to yz)
            }
            
            # Find a representative wavenumber for analysis (peak with highest intensity)
            ref_config = configs[0]
            ref_wavenumbers = self.polarization_data[ref_config]['wavenumbers']
            ref_intensities = self.polarization_data[ref_config]['intensities']
            
            # Find the peak with maximum intensity
            peak_idx = np.argmax(ref_intensities)
            analysis_wavenumber = ref_wavenumbers[peak_idx]
            
            # Extract intensities at this wavenumber for all configurations
            angles = []
            intensities = []
            
            for config in configs:
                if config in config_angles:
                    wavenumbers = self.polarization_data[config]['wavenumbers']
                    config_intensities = self.polarization_data[config]['intensities']
                    
                    # Interpolate to get intensity at analysis wavenumber
                    intensity_at_peak = np.interp(analysis_wavenumber, wavenumbers, config_intensities)
                    
                    angles.append(config_angles[config])
                    intensities.append(intensity_at_peak)
            
            # Sort by angle
            sorted_data = sorted(zip(angles, intensities))
            angles, intensities = zip(*sorted_data)
            
            # Plot angular dependence
            ax.plot(angles, intensities, 'bo-', linewidth=2, markersize=8, label=f'At {analysis_wavenumber:.1f} cm⁻¹')
            
            # Add theoretical curves for comparison
            angles_theory = np.linspace(0, 90, 100)
            
            # Theoretical curves for different symmetries
            # A1 (totally symmetric): I ∝ cos²(θ)
            I_A1 = np.max(intensities) * np.cos(np.radians(angles_theory))**2
            ax.plot(angles_theory, I_A1, 'r--', alpha=0.7, label='A₁ (cos²θ)')
            
            # E (doubly degenerate): I ∝ sin²(θ)
            I_E = np.max(intensities) * np.sin(np.radians(angles_theory))**2
            ax.plot(angles_theory, I_E, 'g--', alpha=0.7, label='E (sin²θ)')
            
            ax.set_xlabel('Polarization Angle (degrees)')
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_title(f'Angular Dependence Analysis')
            ax.legend()
            ax.set_xlim(0, 90)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating angular plot:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=10, alpha=0.6)
            ax.set_title("Angular Dependence Analysis - Error")
    
    def plot_polar_diagram(self, ax):
        """Create a polar plot of polarization data."""
        try:
            # Clear the current axes and create a polar subplot
            ax.clear()
            
            # Get available configurations
            configs = list(self.polarization_data.keys())
            
            # Define angles for different configurations (in radians)
            config_angles = {
                'xx': 0,                    # 0 degrees
                'xy': np.pi/4,             # 45 degrees  
                'yx': np.pi/4,             # 45 degrees
                'yy': np.pi/2,             # 90 degrees
                'zz': 0,                   # 0 degrees (z-axis)
                'xz': np.pi/6,             # 30 degrees
                'zx': np.pi/6,             # 30 degrees
                'yz': np.pi/3,             # 60 degrees
                'zy': np.pi/3              # 60 degrees
            }
            
            # Find peak wavenumber for analysis
            ref_config = configs[0]
            ref_wavenumbers = self.polarization_data[ref_config]['wavenumbers']
            ref_intensities = self.polarization_data[ref_config]['intensities']
            peak_idx = np.argmax(ref_intensities)
            analysis_wavenumber = ref_wavenumbers[peak_idx]
            
            # Extract intensities at peak wavenumber
            angles = []
            intensities = []
            labels = []
            
            for config in configs:
                if config in config_angles:
                    wavenumbers = self.polarization_data[config]['wavenumbers']
                    config_intensities = self.polarization_data[config]['intensities']
                    
                    intensity_at_peak = np.interp(analysis_wavenumber, wavenumbers, config_intensities)
                    
                    angles.append(config_angles[config])
                    intensities.append(intensity_at_peak)
                    labels.append(config.upper())
            
            # Normalize intensities for polar plot
            max_intensity = max(intensities) if intensities else 1
            normalized_intensities = [i/max_intensity for i in intensities]
            
            # Create polar plot using regular axes (since we can't change ax to polar easily)
            # Convert to Cartesian coordinates
            x_coords = [r * np.cos(theta) for r, theta in zip(normalized_intensities, angles)]
            y_coords = [r * np.sin(theta) for r, theta in zip(normalized_intensities, angles)]
            
            # Plot the data points
            ax.scatter(x_coords, y_coords, c='blue', s=100, alpha=0.7, edgecolors='black')
            
            # Add labels for each point
            for i, (x, y, label) in enumerate(zip(x_coords, y_coords, labels)):
                ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, fontweight='bold')
            
            # Draw circles for reference
            circle_radii = [0.25, 0.5, 0.75, 1.0]
            for radius in circle_radii:
                circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', alpha=0.3)
                ax.add_patch(circle)
            
            # Draw angle lines
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                ax.plot([0, np.cos(angle)], [0, np.sin(angle)], 'k-', alpha=0.2)
            
            # Set equal aspect ratio and limits
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            
            # Add labels
            ax.set_xlabel('X-polarization component')
            ax.set_ylabel('Y-polarization component')
            ax.set_title(f'Polar Diagram at {analysis_wavenumber:.1f} cm⁻¹')
            
            # Add intensity scale
            ax.text(1.1, -1.1, f'Max: {max_intensity:.3f}', transform=ax.transData, 
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating polar plot:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=10, alpha=0.6)
            ax.set_title("Polar Plot Analysis - Error")
    
    def export_polarization_results(self):
        """Export polarization analysis results."""
        if not self.polarization_data and not self.depolarization_ratios:
            messagebox.showwarning("No Data", "No polarization data to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Polarization Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as csvfile:
                    import csv
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    writer.writerow(["# Polarization Analysis Results"])
                    writer.writerow(["# Crystal System:", self.crystal_system_var.get()])
                    writer.writerow(["# Scattering Geometry:", self.scattering_geometry_var.get()])
                    writer.writerow([])
                    
                    if self.depolarization_ratios:
                        # Export depolarization ratios
                        writer.writerow(["Wavenumber_cm-1", "Depolarization_Ratio", "I_XX", "I_XY"])
                        
                        wavenumbers = self.depolarization_ratios['wavenumbers']
                        ratios = self.depolarization_ratios['ratio']
                        I_xx = self.depolarization_ratios['I_xx']
                        I_xy = self.depolarization_ratios['I_xy']
                        
                        for i in range(len(wavenumbers)):
                            writer.writerow([f"{wavenumbers[i]:.2f}", f"{ratios[i]:.4f}", 
                                           f"{I_xx[i]:.3f}", f"{I_xy[i]:.3f}"])
                    
                    elif self.polarization_data:
                        # Export raw polarization data
                        configs = list(self.polarization_data.keys())
                        header = ["Wavenumber_cm-1"] + [f"I_{config.upper()}" for config in configs]
                        writer.writerow(header)
                        
                        # Get common wavenumber range
                        ref_wavenumbers = self.polarization_data[configs[0]]['wavenumbers']
                        
                        for i, wavenumber in enumerate(ref_wavenumbers):
                            row = [f"{wavenumber:.2f}"]
                            for config in configs:
                                if i < len(self.polarization_data[config]['intensities']):
                                    intensity = self.polarization_data[config]['intensities'][i]
                                    row.append(f"{intensity:.3f}")
                                else:
                                    row.append("0.000")
                            writer.writerow(row)
                
                messagebox.showinfo("Success", f"Polarization results exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting results: {str(e)}")
    
    def load_cif_file(self):
        """Load crystal structure from CIF file."""
        file_path = filedialog.askopenfilename(
            title="Select CIF File",
            filetypes=[
                ("CIF files", "*.cif"),
                ("All files", "*.*")
            ]
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
                    # Update status with composition
                    structure_name = self.crystal_structure.get('name', os.path.basename(file_path))
                    
                    # Get element composition
                    element_counts = {}
                    for atom in self.crystal_structure.get('atoms', []):
                        element = atom['element']
                        element_counts[element] = element_counts.get(element, 0) + 1
                    
                    composition = ", ".join([f"{element}: {count}" for element, count in element_counts.items()])
                    total_atoms = sum(element_counts.values())
                    
                    self.structure_status_label.config(
                        text=f"✓ {structure_name} ({total_atoms} atoms: {composition})",
                        foreground="green"
                    )
                    
                    # Update structure info display
                    self.update_structure_info()
                    
                    # Generate atomic positions
                    self.generate_unit_cell()
                    
                    #messagebox.showinfo("Success", f"Loaded crystal structure: {structure_name}\n{total_atoms} atoms: {composition}")
                else:
                    self.structure_status_label.config(text="❌ Error loading structure", foreground="red")
                    messagebox.showerror("Error", "Could not parse CIF file.")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Error loading CIF file: {str(e)}")
    
    def parse_cif_with_pymatgen(self, file_path):
        """Parse CIF file using pymatgen for professional-grade parsing."""
        try:
            # Parse CIF file with pymatgen
            parser = CifParser(file_path)
            structures = parser.get_structures()
            
            if not structures:
                raise ValueError("No structures found in CIF file")
            
            # Use the first structure (most CIF files contain one structure)
            original_structure = structures[0]
            
            # Get space group analysis
            sga = SpacegroupAnalyzer(original_structure)
            
            # Get the conventional standard structure to ensure we have all atoms
            # This is crucial for getting the complete unit cell
            try:
                conventional_structure = sga.get_conventional_standard_structure()
                print(f"Original structure: {len(original_structure.sites)} sites")
                print(f"Conventional structure: {len(conventional_structure.sites)} sites")
                
                # Use conventional structure if it has more atoms (complete unit cell)
                if len(conventional_structure.sites) > len(original_structure.sites):
                    pmg_structure = conventional_structure
                    print(f"Using conventional structure with {len(pmg_structure.sites)} atoms")
                else:
                    pmg_structure = original_structure
                    print(f"Using original structure with {len(pmg_structure.sites)} atoms")
                    
            except Exception as e:
                print(f"Error getting conventional structure: {e}")
                pmg_structure = original_structure
            
            # Re-analyze with the chosen structure
            sga = SpacegroupAnalyzer(pmg_structure)
            
            # Convert to our internal format
            structure = {
                'name': os.path.basename(file_path).replace('.cif', ''),
                'pymatgen_structure': pmg_structure,  # Store the complete structure
                'original_structure': original_structure,  # Keep original for reference
                'lattice_parameters': {
                    'a': pmg_structure.lattice.a,
                    'b': pmg_structure.lattice.b,
                    'c': pmg_structure.lattice.c,
                    'alpha': pmg_structure.lattice.alpha,
                    'beta': pmg_structure.lattice.beta,
                    'gamma': pmg_structure.lattice.gamma,
                    'volume': pmg_structure.lattice.volume
                },
                'space_group': sga.get_space_group_symbol(),
                'space_group_number': sga.get_space_group_number(),
                'crystal_system': sga.get_crystal_system(),
                'point_group': sga.get_point_group_symbol(),
                'atoms': [],
                'symmetry_operations': sga.get_symmetry_operations(),
                'primitive_structure': sga.get_primitive_standard_structure(),
                'conventional_structure': pmg_structure
            }
            
            # Extract atomic positions from the complete structure
            for i, site in enumerate(pmg_structure.sites):
                atom = {
                    'label': f"{site.specie.symbol}{i+1}",
                    'element': str(site.specie.symbol),
                    'x': site.frac_coords[0],
                    'y': site.frac_coords[1],
                    'z': site.frac_coords[2],
                    'occupancy': getattr(site, 'occupancy', 1.0),
                    'cartesian_coords': site.coords,
                    'wyckoff_symbol': sga.get_symmetry_dataset()['wyckoffs'][i] if sga.get_symmetry_dataset() else 'unknown'
                }
                structure['atoms'].append(atom)
            
            # Print summary of what we found
            element_counts = {}
            for atom in structure['atoms']:
                element = atom['element']
                element_counts[element] = element_counts.get(element, 0) + 1
            
            print(f"Final structure composition:")
            for element, count in element_counts.items():
                print(f"  {element}: {count} atoms")
            
            return structure
            
        except Exception as e:
            print(f"Error parsing CIF file with pymatgen: {e}")
            # Fallback to simplified parser
            return self.parse_cif_file(file_path)
    
    def parse_cif_file(self, file_path):
        """Parse CIF file and extract crystal structure information (simplified fallback)."""
        try:
            structure = {
                'name': os.path.basename(file_path).replace('.cif', ''),
                'lattice_parameters': {},
                'space_group': '',
                'atoms': [],
                'symmetry_operations': []
            }
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse basic structure information
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Lattice parameters
                if '_cell_length_a' in line:
                    structure['lattice_parameters']['a'] = float(line.split()[1])
                elif '_cell_length_b' in line:
                    structure['lattice_parameters']['b'] = float(line.split()[1])
                elif '_cell_length_c' in line:
                    structure['lattice_parameters']['c'] = float(line.split()[1])
                elif '_cell_angle_alpha' in line:
                    structure['lattice_parameters']['alpha'] = float(line.split()[1])
                elif '_cell_angle_beta' in line:
                    structure['lattice_parameters']['beta'] = float(line.split()[1])
                elif '_cell_angle_gamma' in line:
                    structure['lattice_parameters']['gamma'] = float(line.split()[1])
                
                # Space group
                elif '_space_group_name_H-M_alt' in line or '_symmetry_space_group_name_H-M' in line:
                    structure['space_group'] = line.split('"')[1] if '"' in line else line.split()[1]
                
                # Atomic positions
                elif line.startswith('_atom_site_label') or line.startswith('loop_'):
                    if any('_atom_site' in lines[j] for j in range(i, min(i+10, len(lines)))):
                        # Parse atom site loop
                        atom_data = self.parse_atom_site_loop(lines, i)
                        structure['atoms'] = atom_data
                        break
            
            return structure
            
        except Exception as e:
            print(f"Error parsing CIF file: {e}")
            return None
    
    def parse_atom_site_loop(self, lines, start_idx):
        """Parse the atom site loop from CIF file."""
        atoms = []
        
        # Find column headers
        headers = []
        i = start_idx
        while i < len(lines) and (lines[i].strip().startswith('_atom_site') or lines[i].strip() == 'loop_'):
            if lines[i].strip().startswith('_atom_site'):
                headers.append(lines[i].strip())
            i += 1
        
        # Find data rows
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('_') or line.startswith('loop_'):
                break
            
            parts = line.split()
            if len(parts) >= 4:  # At least label, x, y, z
                atom = {
                    'label': parts[0],
                    'element': ''.join([c for c in parts[0] if c.isalpha()]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'z': float(parts[3]),
                    'occupancy': float(parts[4]) if len(parts) > 4 else 1.0
                }
                atoms.append(atom)
            i += 1
        
        return atoms
    
    def load_from_database(self):
        """Load crystal structure from mineral database."""
        if not self.mineral_database:
            messagebox.showwarning("No Database", "Mineral database not loaded.")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Mineral Structure")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        ttk.Label(dialog, text="Select mineral for structure analysis:", font=('Arial', 12)).pack(pady=10)
        
        # Search and listbox
        search_var = tk.StringVar()
        search_entry = ttk.Entry(dialog, textvariable=search_var)
        search_entry.pack(fill=tk.X, padx=10, pady=5)
        
        listbox = tk.Listbox(dialog, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Populate listbox
        for mineral in self.mineral_list:
            listbox.insert(tk.END, mineral)
        
        def filter_minerals(*args):
            search_term = search_var.get().lower()
            listbox.delete(0, tk.END)
            for mineral in self.mineral_list:
                if search_term in mineral.lower():
                    listbox.insert(tk.END, mineral)
        
        search_var.trace('w', filter_minerals)
        
        def load_selected():
            selection = listbox.curselection()
            if selection:
                mineral_name = listbox.get(selection[0])
                self.load_structure_from_database(mineral_name)
                dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Load", command=load_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def load_structure_from_database(self, mineral_name):
        """Load structure information from mineral database."""
        try:
            if mineral_name not in self.mineral_database:
                messagebox.showerror("Error", f"Mineral {mineral_name} not found in database.")
                return
            
            mineral_data = self.mineral_database[mineral_name]
            
            # Create simplified structure from database
            self.crystal_structure = {
                'name': mineral_name,
                'lattice_parameters': mineral_data.get('lattice_parameters', {}),
                'space_group': mineral_data.get('space_group', 'Unknown'),
                'atoms': mineral_data.get('atoms', []),
                'symmetry_operations': mineral_data.get('symmetry_operations', [])
            }
            
            # Update status with composition
            element_counts = {}
            for atom in self.crystal_structure.get('atoms', []):
                element = atom['element']
                element_counts[element] = element_counts.get(element, 0) + 1
            
            composition = ", ".join([f"{element}: {count}" for element, count in element_counts.items()])
            total_atoms = sum(element_counts.values())
            
            self.structure_status_label.config(
                text=f"✓ {mineral_name} ({total_atoms} atoms: {composition})",
                foreground="blue"
            )
            
            # Update displays
            self.update_structure_info()
            self.generate_unit_cell()
            
            #messagebox.showinfo("Success", f"Loaded structure: {mineral_name}\n{total_atoms} atoms: {composition}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading structure from database: {str(e)}")
    
    def generate_unit_cell(self):
        """Generate all atomic positions in the unit cell."""
        if not self.crystal_structure:
            messagebox.showwarning("No Structure", "Please load a crystal structure first.")
            return
        
        try:
            # Generate atomic positions using symmetry operations
            self.atomic_positions = self.apply_symmetry_operations()
            
            # Update structure plot
            self.update_structure_plot()
            
            # Update info display
            self.update_structure_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating unit cell: {str(e)}")
    
    def apply_symmetry_operations(self):
        """Apply symmetry operations to generate all atomic positions."""
        if not self.crystal_structure or not self.crystal_structure['atoms']:
            return {}
        
        positions = {}
        
        if PYMATGEN_AVAILABLE and 'pymatgen_structure' in self.crystal_structure:
            # Use pymatgen for proper symmetry operations
            pmg_structure = self.crystal_structure['pymatgen_structure']
            
            # Create a supercell to ensure we capture all bonds
            # This is important for bond analysis
            try:
                # Create 2x2x2 supercell for better bond analysis
                supercell = pmg_structure.copy()
                supercell.make_supercell([2, 2, 2])
                
                # Store both unit cell and supercell
                self.crystal_structure['supercell_structure'] = supercell
                
                # Get all sites in the unit cell (original structure)
                for i, site in enumerate(pmg_structure.sites):
                    element = str(site.specie.symbol)
                    if element not in positions:
                        positions[element] = []
                    
                    positions[element].append({
                        'label': f"{site.specie.symbol}{i+1}",
                        'x': site.frac_coords[0],
                        'y': site.frac_coords[1],
                        'z': site.frac_coords[2],
                        'occupancy': getattr(site, 'occupancy', 1.0),
                        'cartesian_coords': site.coords,
                        'site_index': i
                    })
                    
                print(f"Generated unit cell with {len(pmg_structure.sites)} atoms")
                print(f"Created supercell with {len(supercell.sites)} atoms for bond analysis")
                
            except Exception as e:
                print(f"Error creating supercell: {e}")
                # Fallback to original structure
                for i, site in enumerate(pmg_structure.sites):
                    element = str(site.specie.symbol)
                    if element not in positions:
                        positions[element] = []
                    
                    positions[element].append({
                        'label': f"{site.specie.symbol}{i+1}",
                        'x': site.frac_coords[0],
                        'y': site.frac_coords[1],
                        'z': site.frac_coords[2],
                        'occupancy': getattr(site, 'occupancy', 1.0),
                        'cartesian_coords': site.coords,
                        'site_index': i
                    })
        else:
            # Simplified approach for fallback - generate symmetry equivalent positions
            atoms = self.crystal_structure['atoms']
            lattice = self.crystal_structure.get('lattice_parameters', {})
            
            # Apply basic symmetry operations (simplified)
            for atom in atoms:
                element = atom['element']
                if element not in positions:
                    positions[element] = []
                
                # Original position
                positions[element].append({
                    'label': atom['label'],
                    'x': atom['x'],
                    'y': atom['y'],
                    'z': atom['z'],
                    'occupancy': atom.get('occupancy', 1.0)
                })
                
                # Add some basic symmetry equivalent positions for common space groups
                # This is a simplified approach - real implementation would use space group tables
                x, y, z = atom['x'], atom['y'], atom['z']
                
                # Add inversion center if applicable
                if x != 0.5 or y != 0.5 or z != 0.5:
                    positions[element].append({
                        'label': f"{atom['label']}_inv",
                        'x': 1.0 - x,
                        'y': 1.0 - y,
                        'z': 1.0 - z,
                        'occupancy': atom.get('occupancy', 1.0)
                    })
        
        return positions
    
    def update_structure_info(self):
        """Update the structure information display."""
        if not self.crystal_structure:
            return
        
        self.structure_info_text.config(state=tk.NORMAL)
        self.structure_info_text.delete(1.0, tk.END)
        
        info = f"CRYSTAL STRUCTURE INFORMATION\n"
        info += "=" * 35 + "\n\n"
        
        info += f"Name: {self.crystal_structure['name']}\n"
        info += f"Space Group: {self.crystal_structure.get('space_group', 'Unknown')}\n"
        
        # Enhanced information if pymatgen was used
        if 'space_group_number' in self.crystal_structure:
            info += f"Space Group Number: {self.crystal_structure.get('space_group_number', 'Unknown')}\n"
            info += f"Crystal System: {self.crystal_structure.get('crystal_system', 'Unknown')}\n"
            info += f"Point Group: {self.crystal_structure.get('point_group', 'Unknown')}\n"
        
        info += "\n"
        
        # Lattice parameters
        lattice = self.crystal_structure.get('lattice_parameters', {})
        if lattice:
            info += "Lattice Parameters:\n"
            info += f"  a = {lattice.get('a', 'N/A'):.4f} Å\n"
            info += f"  b = {lattice.get('b', 'N/A'):.4f} Å\n"
            info += f"  c = {lattice.get('c', 'N/A'):.4f} Å\n"
            info += f"  α = {lattice.get('alpha', 'N/A'):.2f}°\n"
            info += f"  β = {lattice.get('beta', 'N/A'):.2f}°\n"
            info += f"  γ = {lattice.get('gamma', 'N/A'):.2f}°\n"
            
            if 'volume' in lattice:
                info += f"  Volume = {lattice['volume']:.2f} Å³\n"
            info += "\n"
        
        # Atomic positions
        atoms = self.crystal_structure.get('atoms', [])
        if atoms:
            info += f"Atoms ({len(atoms)} sites):\n"
            for atom in atoms:
                wyckoff = atom.get('wyckoff_symbol', '')
                wyckoff_str = f" [{wyckoff}]" if wyckoff and wyckoff != 'unknown' else ""
                info += f"  {atom['label']}: ({atom['x']:.4f}, {atom['y']:.4f}, {atom['z']:.4f}){wyckoff_str}\n"
            info += "\n"
        
        # Unit cell generation info
        if self.atomic_positions:
            total_atoms = sum(len(positions) for positions in self.atomic_positions.values())
            info += f"Generated Unit Cell:\n"
            for element, positions in self.atomic_positions.items():
                info += f"  {element}: {len(positions)} atoms\n"
            info += f"  Total: {total_atoms} atoms\n\n"
        
        # Symmetry information
        if 'symmetry_operations' in self.crystal_structure:
            sym_ops = self.crystal_structure['symmetry_operations']
            if hasattr(sym_ops, '__len__'):
                info += f"Symmetry Operations: {len(sym_ops)}\n\n"
        
        # Bond analysis summary
        if self.bond_analysis:
            info += f"Bond Analysis:\n"
            info += f"  Total bonds: {len(self.bond_analysis.get('bonds', []))}\n"
            info += f"  Unique bond types: {len(self.bond_analysis.get('bond_types', {}))}\n"
            
            # Coordination analysis if available
            if 'coordination_analysis' in self.bond_analysis:
                coord_analysis = self.bond_analysis['coordination_analysis']
                if coord_analysis:
                    info += f"  Coordination environments: {len(coord_analysis)}\n"
        
        self.structure_info_text.insert(tk.END, info)
        self.structure_info_text.config(state=tk.DISABLED)
    
    def calculate_bond_lengths(self):
        """Calculate all bond lengths in the crystal structure."""
        if not self.crystal_structure:
            messagebox.showwarning("No Structure", "Please load a crystal structure first.")
            return
        
        try:
            # Calculate bonds (simplified implementation)
            self.bond_analysis = self.analyze_bonds()
            
            # Update structure info
            self.update_structure_info()
            
            #messagebox.showinfo("Success", f"Calculated {len(self.bond_analysis.get('bonds', []))} bonds.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating bond lengths: {str(e)}")
    
    def analyze_bonds(self):
        """Analyze bonds in the crystal structure using advanced methods."""
        bonds = []
        bond_types = {}
        coordination_analysis = {}
        
        if PYMATGEN_AVAILABLE and 'pymatgen_structure' in self.crystal_structure:
            # Use pymatgen for advanced bond analysis with proper symmetry handling
            pmg_structure = self.crystal_structure['pymatgen_structure']
            
            try:
                print(f"Analyzing bonds for structure with {len(pmg_structure.sites)} sites")
                
                # Use CrystalNN for intelligent bond detection with distance matrix for accuracy
                try:
                    from pymatgen.analysis.local_env import CrystalNN
                    cnn = CrystalNN()
                    distance_matrix = pmg_structure.distance_matrix
                    
                    print("Using CrystalNN for bond detection")
                    
                    # Analyze bonds for each site in the original unit cell
                    for i, site in enumerate(pmg_structure.sites):
                        try:
                            # Get nearest neighbors using CrystalNN
                            nn_info = cnn.get_nn_info(pmg_structure, i)
                            
                            for neighbor in nn_info:
                                j = neighbor['site_index']
                                # Use distance matrix for accurate distance
                                distance = distance_matrix[i][j]
                                
                                # Only add each bond once (avoid duplicates)
                                if i < j:
                                    site_j = pmg_structure.sites[j]
                                    bond_type = f"{site.specie.symbol}-{site_j.specie.symbol}"
                                    
                                    # Apply element-specific cutoffs as a sanity check
                                    max_cutoff = self.get_bond_cutoff(site.specie.symbol, site_j.specie.symbol)
                                    
                                    if distance < max_cutoff:  # Only include reasonable bonds
                                        bonds.append({
                                            'atom1': f"{site.specie.symbol}{i+1}",
                                            'atom2': f"{site_j.specie.symbol}{j+1}",
                                            'distance': distance,
                                            'type': bond_type
                                        })
                                        
                                        if bond_type not in bond_types:
                                            bond_types[bond_type] = []
                                        bond_types[bond_type].append(distance)
                        
                        except Exception as e:
                            print(f"CrystalNN failed for site {i}: {e}")
                            # Fallback to distance-based analysis for this site
                            continue
                    
                except ImportError:
                    print("CrystalNN not available, using distance matrix with element-specific cutoffs")
                    
                    # Fallback: Use distance matrix but with better cutoffs
                    distance_matrix = pmg_structure.distance_matrix
                    
                    for i, site_i in enumerate(pmg_structure.sites):
                        for j, site_j in enumerate(pmg_structure.sites):
                            if i < j:  # Avoid duplicate bonds
                                distance = distance_matrix[i][j]
                                
                                # Use element-specific bond cutoffs
                                max_cutoff = self.get_bond_cutoff(site_i.specie.symbol, site_j.specie.symbol)
                                
                                if 0.8 < distance < max_cutoff:  # More restrictive cutoffs
                                    bond_type = f"{site_i.specie.symbol}-{site_j.specie.symbol}"
                                    
                                    bonds.append({
                                        'atom1': f"{site_i.specie.symbol}{i+1}",
                                        'atom2': f"{site_j.specie.symbol}{j+1}",
                                        'distance': distance,
                                        'type': bond_type
                                    })
                                    
                                    if bond_type not in bond_types:
                                        bond_types[bond_type] = []
                                    bond_types[bond_type].append(distance)
                
                # Coordination analysis using the same CrystalNN instance
                if 'cnn' in locals():
                    for i, site in enumerate(pmg_structure.sites):
                        try:
                            coord_env = cnn.get_cn_dict(pmg_structure, i)
                            coordination_analysis[f"{site.specie.symbol}{i+1}"] = coord_env
                        except Exception as coord_error:
                            print(f"CrystalNN coordination failed for site {i}: {coord_error}")
                            # Use simple coordination number based on bonds found
                            site_bonds = [b for b in bonds if b['atom1'] == f"{site.specie.symbol}{i+1}" or b['atom2'] == f"{site.specie.symbol}{i+1}"]
                            coordination_analysis[f"{site.specie.symbol}{i+1}"] = {
                                'coordination_number': len(site_bonds)
                            }
                else:
                    # Simple coordination analysis based on bonds found
                    for i, site in enumerate(pmg_structure.sites):
                        site_bonds = [b for b in bonds if b['atom1'] == f"{site.specie.symbol}{i+1}" or b['atom2'] == f"{site.specie.symbol}{i+1}"]
                        coordination_analysis[f"{site.specie.symbol}{i+1}"] = {
                            'coordination_number': len(site_bonds)
                        }
                
                print(f"Found {len(bonds)} bonds")
                        
            except Exception as e:
                print(f"Error in pymatgen bond analysis: {e}")
                # Fallback to simplified analysis
                return self.analyze_bonds_simplified()
        else:
            # Use simplified bond analysis
            return self.analyze_bonds_simplified()
        
        return {
            'bonds': bonds,
            'bond_types': bond_types,
            'coordination_analysis': coordination_analysis
        }
    
    def analyze_bonds_simplified(self):
        """Simplified bond analysis fallback method."""
        bonds = []
        bond_types = {}
        
        atoms = self.crystal_structure.get('atoms', [])
        lattice = self.crystal_structure.get('lattice_parameters', {})
        
        print(f"Simplified bond analysis for {len(atoms)} atoms")
        
        # Get lattice parameters
        a = lattice.get('a', 5.0)  # Default to reasonable values
        b = lattice.get('b', 5.0)
        c = lattice.get('c', 5.0)
        alpha = np.radians(lattice.get('alpha', 90.0))
        beta = np.radians(lattice.get('beta', 90.0))
        gamma = np.radians(lattice.get('gamma', 90.0))
        
        # Create transformation matrix for fractional to Cartesian coordinates
        # This handles non-orthogonal unit cells
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        
        volume = a * b * c * np.sqrt(1 + 2*cos_alpha*cos_beta*cos_gamma - 
                                   cos_alpha**2 - cos_beta**2 - cos_gamma**2)
        
        # Transformation matrix
        transform_matrix = np.array([
            [a, b*cos_gamma, c*cos_beta],
            [0, b*sin_gamma, c*(cos_alpha - cos_beta*cos_gamma)/sin_gamma],
            [0, 0, volume/(a*b*sin_gamma)]
        ])
        
        # Convert all atoms to Cartesian coordinates
        cartesian_atoms = []
        for atom in atoms:
            frac_coords = np.array([atom['x'], atom['y'], atom['z']])
            cart_coords = transform_matrix @ frac_coords
            cartesian_atoms.append({
                'label': atom['label'],
                'element': atom['element'],
                'cart_coords': cart_coords,
                'frac_coords': frac_coords
            })
        
        # Calculate bonds including periodic boundary conditions
        for i, atom1 in enumerate(cartesian_atoms):
            for j, atom2 in enumerate(cartesian_atoms[i+1:], i+1):
                min_distance = float('inf')
                
                # Check periodic images (±1 in each direction)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            # Shift atom2 by unit cell vectors
                            shift_vector = np.array([dx, dy, dz])
                            shifted_frac = atom2['frac_coords'] + shift_vector
                            shifted_cart = transform_matrix @ shifted_frac
                            
                            # Calculate distance
                            distance = np.linalg.norm(atom1['cart_coords'] - shifted_cart)
                            min_distance = min(min_distance, distance)
                
                # Consider as bond if distance is reasonable
                if 0.5 < min_distance < 3.5:  # Avoid self-bonds and unreasonable distances
                    bond_type = f"{atom1['element']}-{atom2['element']}"
                    bonds.append({
                        'atom1': atom1['label'],
                        'atom2': atom2['label'],
                        'distance': min_distance,
                        'type': bond_type
                    })
                    
                    if bond_type not in bond_types:
                        bond_types[bond_type] = []
                    bond_types[bond_type].append(min_distance)
        
        print(f"Found {len(bonds)} bonds in simplified analysis")
        
        return {
            'bonds': bonds,
            'bond_types': bond_types
        }
    
    def analyze_coordination(self):
        """Analyze coordination environments."""
        if not self.bond_analysis:
            self.calculate_bond_lengths()
        
        # Implementation for coordination analysis
        messagebox.showinfo("Coordination Analysis", "Coordination analysis completed.")
    
    def show_bond_table(self):
        """Show detailed bond analysis table."""
        if not self.bond_analysis:
            messagebox.showwarning("No Data", "Please calculate bond lengths first.")
            return
        
        # Create bond table window
        table_window = tk.Toplevel(self.root)
        table_window.title("Bond Analysis Table")
        table_window.geometry("600x400")
        
        # Create text widget for table
        text_frame = ttk.Frame(table_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate table
        table_text = "BOND ANALYSIS TABLE\n"
        table_text += "=" * 50 + "\n\n"
        
        bonds = self.bond_analysis.get('bonds', [])
        table_text += f"{'Atom 1':<10} {'Atom 2':<10} {'Distance (Å)':<12} {'Bond Type':<15}\n"
        table_text += "-" * 50 + "\n"
        
        for bond in bonds:
            table_text += f"{bond['atom1']:<10} {bond['atom2']:<10} {bond['distance']:<12.3f} {bond['type']:<15}\n"
        
        # Bond type summary
        bond_types = self.bond_analysis.get('bond_types', {})
        table_text += f"\n\nBOND TYPE SUMMARY\n"
        table_text += "-" * 20 + "\n"
        
        for bond_type, distances in bond_types.items():
            avg_dist = np.mean(distances)
            std_dist = np.std(distances)
            table_text += f"{bond_type}: {avg_dist:.3f} ± {std_dist:.3f} Å ({len(distances)} bonds)\n"
        
        text_widget.insert(tk.END, table_text)
        text_widget.config(state=tk.DISABLED)
    
    def configure_bond_filters(self):
        """Configure which bonds to display in 3D visualization."""
        # Initialize bond filters if not exists
        if not hasattr(self, 'bond_filters'):
            self.bond_filters = {}
        
        # Create bond filter configuration window
        filter_window = tk.Toplevel(self.root)
        filter_window.title("Bond Filter Configuration")
        filter_window.geometry("500x600")
        
        # Main frame with scrollbar
        main_frame = ttk.Frame(filter_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Instructions
        ttk.Label(main_frame, text="Configure which bonds to display in 3D visualization:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Create scrollable frame
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Get available bond types from structure data
        bond_types = self.get_available_bond_types()
        
        if not bond_types:
            ttk.Label(scrollable_frame, text="No bond data available. Load structure and calculate bonds first.").pack(pady=20)
        else:
            # Global controls
            global_frame = ttk.LabelFrame(scrollable_frame, text="Global Controls", padding=10)
            global_frame.pack(fill=tk.X, pady=(0, 10))
            
            filter_vars = {}
            
            ttk.Button(global_frame, text="Enable All Bonds", 
                      command=lambda: self.set_all_bond_filters(True, bond_types, filter_vars)).pack(side=tk.LEFT, padx=5)
            ttk.Button(global_frame, text="Disable All Bonds", 
                      command=lambda: self.set_all_bond_filters(False, bond_types, filter_vars)).pack(side=tk.LEFT, padx=5)
            ttk.Button(global_frame, text="Reset to Defaults", 
                      command=lambda: self.reset_bond_filters_to_defaults(bond_types, filter_vars)).pack(side=tk.LEFT, padx=5)
            
            # Individual bond type controls
            for bond_type in sorted(bond_types):
                bond_frame = ttk.LabelFrame(scrollable_frame, text=f"{bond_type} Bonds", padding=10)
                bond_frame.pack(fill=tk.X, pady=5)
                
                # Enable/disable checkbox
                var = tk.BooleanVar(value=self.bond_filters.get(bond_type, {}).get('enabled', True))
                filter_vars[bond_type] = {'enabled': var}
                
                ttk.Checkbutton(bond_frame, text=f"Show {bond_type} bonds", 
                               variable=var).pack(anchor=tk.W)
                
                # Distance range controls
                distances = self.get_bond_distances(bond_type)
                if distances:
                    min_dist, max_dist = min(distances), max(distances)
                    
                    # Create both variables first
                    min_var = tk.DoubleVar(value=self.bond_filters.get(bond_type, {}).get('min_distance', min_dist))
                    max_var = tk.DoubleVar(value=self.bond_filters.get(bond_type, {}).get('max_distance', max_dist))
                    filter_vars[bond_type]['min_distance'] = min_var
                    filter_vars[bond_type]['max_distance'] = max_var
                    
                    # Min distance
                    min_label = ttk.Label(bond_frame, text=f"Min distance: {min_var.get():.2f} Å")
                    min_label.pack(anchor=tk.W, pady=(5, 0))
                    
                    def update_min_label(*args, label=min_label, var=min_var, max_var=max_var):
                        min_val = var.get()
                        # Ensure min doesn't exceed max
                        if min_val > max_var.get():
                            max_var.set(min_val)
                        label.config(text=f"Min distance: {min_val:.2f} Å")
                    
                    min_var.trace('w', update_min_label)
                    
                    min_scale = ttk.Scale(bond_frame, from_=min_dist*0.8, to=max_dist*1.2, 
                                         variable=min_var, orient=tk.HORIZONTAL)
                    min_scale.pack(fill=tk.X, pady=2)
                    
                    # Max distance
                    max_label = ttk.Label(bond_frame, text=f"Max distance: {max_var.get():.2f} Å")
                    max_label.pack(anchor=tk.W, pady=(5, 0))
                    
                    def update_max_label(*args, label=max_label, var=max_var, min_var=min_var):
                        max_val = var.get()
                        # Ensure max doesn't go below min
                        if max_val < min_var.get():
                            min_var.set(max_val)
                        label.config(text=f"Max distance: {max_val:.2f} Å")
                    
                    max_var.trace('w', update_max_label)
                    
                    max_scale = ttk.Scale(bond_frame, from_=min_dist*0.8, to=max_dist*1.2, 
                                         variable=max_var, orient=tk.HORIZONTAL)
                    max_scale.pack(fill=tk.X, pady=2)
                    
                    # Bond range and count info
                    bond_count = len(distances)
                    info_frame = ttk.Frame(bond_frame)
                    info_frame.pack(fill=tk.X, pady=(5, 0))
                    
                    ttk.Label(info_frame, text=f"Data range: {min_dist:.2f} - {max_dist:.2f} Å", 
                             foreground="gray").pack(side=tk.LEFT)
                    
                    # Dynamic count of filtered bonds
                    filtered_count_label = ttk.Label(info_frame, text=f"Showing: {bond_count}/{bond_count} bonds", 
                                                    foreground="blue")
                    filtered_count_label.pack(side=tk.RIGHT)
                    
                    def update_filtered_count(*args, label=filtered_count_label, min_var=min_var, max_var=max_var, distances=distances):
                        min_val, max_val = min_var.get(), max_var.get()
                        filtered_count = sum(1 for d in distances if min_val <= d <= max_val)
                        label.config(text=f"Showing: {filtered_count}/{len(distances)} bonds")
                    
                    min_var.trace('w', update_filtered_count)
                    max_var.trace('w', update_filtered_count)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons frame
        button_frame = ttk.Frame(filter_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Apply Filters", 
                  command=lambda: self.apply_bond_filters(filter_vars, filter_window)).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=filter_window.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Preview", 
                  command=lambda: self.preview_bond_filters(filter_vars)).pack(side=tk.RIGHT, padx=5)
    
    def get_available_bond_types(self):
        """Get list of available bond types from structure data."""
        bond_types = set()
        
        # From bond analysis
        if hasattr(self, 'bond_analysis') and self.bond_analysis:
            if 'bonds' in self.bond_analysis:
                for bond in self.bond_analysis['bonds']:
                    bond_types.add(bond.get('type', 'Unknown'))
            if 'bond_types' in self.bond_analysis:
                bond_types.update(self.bond_analysis['bond_types'].keys())
        
        # From structure data (if available)
        if hasattr(self, 'structure_data') and self.structure_data:
            # Add common bond types based on elements present
            if 'atoms' in self.structure_data:
                elements = set()
                for atom in self.structure_data['atoms']:
                    elements.add(atom.get('element', atom.get('symbol', 'Unknown')))
                
                # Generate common bond type combinations
                element_list = sorted(list(elements))
                for i, elem1 in enumerate(element_list):
                    for elem2 in element_list[i:]:
                        bond_types.add(f"{elem1}-{elem2}")
        
        return list(bond_types)
    
    def get_bond_distances(self, bond_type):
        """Get list of distances for a specific bond type."""
        distances = []
        
        if hasattr(self, 'bond_analysis') and self.bond_analysis:
            if 'bonds' in self.bond_analysis:
                for bond in self.bond_analysis['bonds']:
                    if bond.get('type') == bond_type:
                        distances.append(bond['distance'])
            if 'bond_types' in self.bond_analysis and bond_type in self.bond_analysis['bond_types']:
                distances.extend(self.bond_analysis['bond_types'][bond_type])
        
        return distances
    
    def set_all_bond_filters(self, enabled, bond_types, filter_vars):
        """Enable or disable all bond filters."""
        for bond_type in bond_types:
            if bond_type in filter_vars and 'enabled' in filter_vars[bond_type]:
                filter_vars[bond_type]['enabled'].set(enabled)
    
    def reset_bond_filters_to_defaults(self, bond_types, filter_vars):
        """Reset bond filters to default values."""
        for bond_type in bond_types:
            if bond_type in filter_vars:
                # Enable all bonds by default
                filter_vars[bond_type]['enabled'].set(True)
                
                # Reset distance ranges to full range
                distances = self.get_bond_distances(bond_type)
                if distances:
                    min_dist, max_dist = min(distances), max(distances)
                    if 'min_distance' in filter_vars[bond_type]:
                        filter_vars[bond_type]['min_distance'].set(min_dist)
                    if 'max_distance' in filter_vars[bond_type]:
                        filter_vars[bond_type]['max_distance'].set(max_dist)
    
    def preview_bond_filters(self, filter_vars):
        """Preview bond filters without applying them."""
        # Temporarily apply filters and update 3D visualization
        old_filters = self.bond_filters.copy() if hasattr(self, 'bond_filters') else {}
        
        # Apply new filters temporarily
        self.bond_filters = {}
        for bond_type, vars_dict in filter_vars.items():
            self.bond_filters[bond_type] = {
                'enabled': vars_dict['enabled'].get(),
                'min_distance': vars_dict.get('min_distance', tk.DoubleVar()).get() if 'min_distance' in vars_dict else 0.0,
                'max_distance': vars_dict.get('max_distance', tk.DoubleVar()).get() if 'max_distance' in vars_dict else 10.0
            }
        
        # Update 3D visualization if it exists
        if hasattr(self, 'update_3d_visualization'):
            self.update_3d_visualization()
        
        # Update Crystal Structure tab bond network plot
        if hasattr(self, 'update_structure_plot'):
            self.update_structure_plot()
        
        # Restore old filters after a delay (for preview)
        self.root.after(3000, lambda: setattr(self, 'bond_filters', old_filters))
        self.root.after(3000, lambda: self.update_3d_visualization() if hasattr(self, 'update_3d_visualization') else None)
        self.root.after(3000, lambda: self.update_structure_plot() if hasattr(self, 'update_structure_plot') else None)
        
        messagebox.showinfo("Preview", "Bond filters applied temporarily for 3 seconds.")
    
    def apply_bond_filters(self, filter_vars, window):
        """Apply bond filters and close the configuration window."""
        # Apply filters
        self.bond_filters = {}
        for bond_type, vars_dict in filter_vars.items():
            self.bond_filters[bond_type] = {
                'enabled': vars_dict['enabled'].get(),
                'min_distance': vars_dict.get('min_distance', tk.DoubleVar()).get() if 'min_distance' in vars_dict else 0.0,
                'max_distance': vars_dict.get('max_distance', tk.DoubleVar()).get() if 'max_distance' in vars_dict else 10.0
            }
        
        # Update 3D visualization if it exists
        if hasattr(self, 'update_3d_visualization'):
            self.update_3d_visualization()
        
        # Update Crystal Structure tab bond network plot
        if hasattr(self, 'update_structure_plot'):
            self.update_structure_plot()
        
        # Close window
        window.destroy()
        
        #messagebox.showinfo("Success", "Bond filters applied successfully!")
    
    def reset_bond_filters(self):
        """Reset bond filters to default (show all bonds)."""
        self.bond_filters = {}
        if hasattr(self, 'update_3d_visualization'):
            self.update_3d_visualization()
        if hasattr(self, 'update_structure_plot'):
            self.update_structure_plot()
        messagebox.showinfo("Reset", "Bond filters reset to defaults (all bonds visible).")
    
    def debug_bond_analysis(self):
        """Debug bond analysis to identify problematic bonds."""
        if not self.bond_analysis:
            messagebox.showwarning("No Data", "Please calculate bond lengths first.")
            return
        
        # Create debug window
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Bond Analysis Debug")
        debug_window.geometry("800x600")
        
        # Create text widget for debug info
        text_frame = ttk.Frame(debug_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate debug information
        debug_text = "BOND ANALYSIS DEBUG REPORT\n"
        debug_text += "=" * 50 + "\n\n"
        
        # Structure information
        debug_text += f"Crystal Structure: {self.crystal_structure['name']}\n"
        debug_text += f"Space Group: {self.crystal_structure.get('space_group', 'Unknown')}\n"
        debug_text += f"Number of unique atoms: {len(self.crystal_structure.get('atoms', []))}\n\n"
        
        # Atom positions
        atoms = self.crystal_structure.get('atoms', [])
        debug_text += "ATOM POSITIONS:\n"
        debug_text += "-" * 20 + "\n"
        for atom in atoms:
            debug_text += f"{atom['label']}: ({atom['x']:.4f}, {atom['y']:.4f}, {atom['z']:.4f})\n"
        debug_text += "\n"
        
        # Bond analysis
        bonds = self.bond_analysis.get('bonds', [])
        debug_text += f"BONDS FOUND ({len(bonds)} total):\n"
        debug_text += "-" * 30 + "\n"
        debug_text += f"{'Atom 1':<8} {'Atom 2':<8} {'Distance':<10} {'Type':<10} {'Status':<15}\n"
        debug_text += "-" * 60 + "\n"
        
        for bond in sorted(bonds, key=lambda x: x['distance'], reverse=True):
            distance = bond['distance']
            bond_type = bond['type']
            
            # Determine bond status
            if distance > 3.0:
                status = "SUSPICIOUS"
            elif distance > 2.5:
                status = "LONG"
            elif distance < 1.0:
                status = "SHORT"
            else:
                status = "NORMAL"
            
            debug_text += f"{bond['atom1']:<8} {bond['atom2']:<8} {distance:<10.3f} {bond_type:<10} {status:<15}\n"
        
        # Bond type statistics
        bond_types = self.bond_analysis.get('bond_types', {})
        debug_text += f"\n\nBOND TYPE STATISTICS:\n"
        debug_text += "-" * 25 + "\n"
        for bond_type, distances in bond_types.items():
            min_dist = min(distances)
            max_dist = max(distances)
            avg_dist = np.mean(distances)
            count = len(distances)
            debug_text += f"{bond_type}: {count} bonds, {min_dist:.3f}-{max_dist:.3f} Å (avg: {avg_dist:.3f})\n"
        
        # Suspicious bonds analysis
        suspicious_bonds = [b for b in bonds if b['distance'] > 3.0]
        if suspicious_bonds:
            debug_text += f"\n\nSUSPICIOUS BONDS (> 3.0 Å):\n"
            debug_text += "-" * 30 + "\n"
            for bond in suspicious_bonds:
                debug_text += f"{bond['atom1']} - {bond['atom2']}: {bond['distance']:.3f} Å\n"
                
                # Try to find the actual positions
                atom1_pos = None
                atom2_pos = None
                for atom in atoms:
                    if atom['label'] == bond['atom1']:
                        atom1_pos = (atom['x'], atom['y'], atom['z'])
                    elif atom['label'] == bond['atom2']:
                        atom2_pos = (atom['x'], atom['y'], atom['z'])
                
                if atom1_pos and atom2_pos:
                    debug_text += f"  {bond['atom1']} at {atom1_pos}\n"
                    debug_text += f"  {bond['atom2']} at {atom2_pos}\n"
                    
                    # Calculate direct distance
                    direct_dist = np.sqrt(sum((a-b)**2 for a, b in zip(atom1_pos, atom2_pos)))
                    debug_text += f"  Direct distance: {direct_dist:.3f} Å\n"
                    
                    if abs(direct_dist - bond['distance']) > 0.1:
                        debug_text += f"  WARNING: Distance mismatch! May be periodic image.\n"
                debug_text += "\n"
        
        # Recommendations
        debug_text += "\nRECOMMENDATIONS:\n"
        debug_text += "-" * 15 + "\n"
        if suspicious_bonds:
            debug_text += f"• Found {len(suspicious_bonds)} suspicious bonds > 3.0 Å\n"
            debug_text += "• These may be artifacts from incorrect symmetry handling\n"
            debug_text += "• Consider using bond filters to hide bonds > 2.5 Å\n"
        else:
            debug_text += "• No suspicious bonds found\n"
            debug_text += "• Bond analysis appears reasonable\n"
        
        text_widget.insert(tk.END, debug_text)
        text_widget.config(state=tk.DISABLED)
    
    def calculate_phonon_modes(self):
        """Calculate phonon modes for the crystal structure."""
        if not self.crystal_structure:
            messagebox.showwarning("No Structure", "Please load a crystal structure first.")
            return
        
        # Placeholder for phonon calculation
        messagebox.showinfo("Phonon Modes", "Phonon mode calculation would require quantum mechanical methods.\nThis is a placeholder for future implementation.")
    
    def calculate_raman_tensors(self):
        """Calculate Raman tensors from crystal structure."""
        if not self.crystal_structure:
            messagebox.showwarning("No Structure", "Please load a crystal structure first.")
            return
        
        # Placeholder for Raman tensor calculation
        messagebox.showinfo("Raman Tensors", "Raman tensor calculation would require:\n1. Phonon modes\n2. Polarizability derivatives\n3. Symmetry analysis\n\nThis is a placeholder for future implementation.")
    
    def install_pymatgen(self):
        """Install pymatgen for enhanced CIF parsing."""
        result = messagebox.askyesno(
            "Install pymatgen", 
            "pymatgen provides professional-grade CIF parsing and crystallographic analysis.\n\n"
            "Would you like to install it?\n\n"
            "This will run: pip install pymatgen"
        )
        
        if result:
            try:
                import subprocess
                import sys
                
                # Show progress dialog
                progress_window = tk.Toplevel(self.root)
                progress_window.title("Installing pymatgen")
                progress_window.geometry("400x150")
                progress_window.transient(self.root)
                progress_window.grab_set()
                
                ttk.Label(progress_window, text="Installing pymatgen...", font=('Arial', 12)).pack(pady=20)
                progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
                progress_bar.pack(pady=10, padx=20, fill=tk.X)
                progress_bar.start()
                
                progress_window.update()
                
                # Install pymatgen
                result = subprocess.run([sys.executable, "-m", "pip", "install", "pymatgen"], 
                                      capture_output=True, text=True)
                
                progress_bar.stop()
                progress_window.destroy()
                
                if result.returncode == 0:
                    pass
                    #messagebox.showinfo("Success", 
                    #                  "pymatgen installed successfully!\n\n"
                    #                  "Please restart the application to use enhanced CIF parsing.")
                else:
                    pass
                    messagebox.showerror("Error", 
                                       f"Failed to install pymatgen:\n{result.stderr}")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Error installing pymatgen: {str(e)}")
    
    def export_to_polarization(self):
        """Export calculated Raman tensors to polarization analysis."""
        if not self.calculated_raman_tensors:
            messagebox.showwarning("No Data", "Please calculate Raman tensors first.")
            return
        
        # Export to polarization tab
        messagebox.showinfo("Export", "Raman tensor data would be exported to Polarization tab for analysis.")
    
    def update_structure_plot(self, event=None):
        """Update the crystal structure visualization."""
        if "Crystal Structure" not in self.plot_components:
            return
        
        components = self.plot_components["Crystal Structure"]
        ax = components['ax']
        canvas = components['canvas']
        
        ax.clear()
        
        display_mode = self.structure_display_var.get()
        
        if self.crystal_structure and display_mode == "Unit Cell":
            self.plot_unit_cell(ax)
        elif self.bond_analysis and display_mode == "Bond Network":
            self.plot_bond_network(ax)
        else:
            ax.text(0.5, 0.5, f'Load crystal structure to view {display_mode.lower()}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=12, alpha=0.6)
            ax.set_title("Crystal Structure Analysis")
        
        ax.grid(True, alpha=0.3)
        canvas.draw()
    
    def plot_unit_cell(self, ax):
        """Plot the unit cell structure."""
        try:
            # Use generated atomic positions if available, otherwise fall back to original atoms
            if self.atomic_positions:
                # Plot all generated atoms
                colors = {'C': 'black', 'O': 'red', 'Si': 'blue', 'Al': 'gray', 'Ca': 'green', 
                         'Mg': 'orange', 'Fe': 'brown', 'Ti': 'purple', 'Na': 'yellow', 'K': 'pink'}
                
                plotted_elements = set()
                
                for element, positions in self.atomic_positions.items():
                    color = colors.get(element, 'purple')
                    
                    for pos in positions:
                        # Only plot atoms within the unit cell for clarity
                        if 0 <= pos['x'] <= 1 and 0 <= pos['y'] <= 1:
                            label = element if element not in plotted_elements else ""
                            ax.scatter(pos['x'], pos['y'], c=color, s=100, alpha=0.7, 
                                      edgecolors='black', label=label)
                            plotted_elements.add(element)
                            
                            # Add atom labels for a few atoms
                            if len([p for p in positions if 0 <= p['x'] <= 1 and 0 <= p['y'] <= 1]) <= 10:
                                ax.annotate(pos['label'], (pos['x'], pos['y']), 
                                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                total_atoms = sum(len(positions) for positions in self.atomic_positions.values())
                title = f'Unit Cell: {self.crystal_structure["name"]} ({total_atoms} atoms)'
                
            else:
                # Fallback to original atoms
                atoms = self.crystal_structure.get('atoms', [])
                
                if not atoms:
                    ax.text(0.5, 0.5, 'No atomic positions available\nTry "Generate Unit Cell"', 
                           transform=ax.transAxes, ha='center', va='center', 
                           fontsize=12, alpha=0.6)
                    return
                
                colors = {'C': 'black', 'O': 'red', 'Si': 'blue', 'Al': 'gray', 'Ca': 'green', 'Mg': 'orange'}
                
                for atom in atoms:
                    element = atom['element']
                    color = colors.get(element, 'purple')
                    
                    ax.scatter(atom['x'], atom['y'], c=color, s=100, alpha=0.7, 
                              edgecolors='black', label=element)
                    
                    # Add atom labels
                    ax.annotate(atom['label'], (atom['x'], atom['y']), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                title = f'Unit Cell: {self.crystal_structure["name"]} (asymmetric unit)'
            
            # Draw unit cell outline
            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', alpha=0.5, linewidth=2)
            
            ax.set_xlabel('a-axis (fractional)')
            ax.set_ylabel('b-axis (fractional)')
            ax.set_title(title)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')
            
            # Create legend with unique elements
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting unit cell:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=10, alpha=0.6)
    
    def plot_bond_network(self, ax):
        """Plot the bond network with filtering support in 3D."""
        try:
            if not self.bond_analysis:
                ax.text(0.5, 0.5, 'Calculate bond lengths first', 
                       transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, alpha=0.6)
                return
            
            # Clear the current axes
            ax.clear()
            
            # Import 3D plotting capability
            from mpl_toolkits.mplot3d import Axes3D
            
            # Check if we need to convert to 3D
            if not hasattr(ax, 'zaxis'):
                # Remove the current 2D axes and create a 3D one
                fig = ax.figure
                ax.remove()
                ax = fig.add_subplot(111, projection='3d')
                
                # Update the stored reference
                if "Crystal Structure" in self.plot_components:
                    self.plot_components["Crystal Structure"]['ax'] = ax
            
            atoms = self.crystal_structure.get('atoms', [])
            bonds = self.bond_analysis.get('bonds', [])
            
            # Create atom position lookup with 3D coordinates
            atom_positions = {}
            for atom in atoms:
                # Use z=0 if no z coordinate is available, or try to get from 3D coordinates
                z_coord = atom.get('z', 0.0)
                if 'frac_coords' in atom and len(atom['frac_coords']) >= 3:
                    z_coord = atom['frac_coords'][2]
                elif 'cart_coords' in atom and len(atom['cart_coords']) >= 3:
                    # Convert cartesian to fractional z
                    z_coord = atom['cart_coords'][2] / 10.0  # Rough scaling
                
                # Convert fractional coordinates to Cartesian for proper visualization
                frac_coords = np.array([atom['x'], atom['y'], z_coord])
                
                # Get lattice parameters for Cartesian conversion
                lattice_params = self.crystal_structure.get('lattice_parameters', {})
                a = lattice_params.get('a', 5.0)
                b = lattice_params.get('b', 5.0) 
                c = lattice_params.get('c', 5.0)
                alpha = np.radians(lattice_params.get('alpha', 90.0))
                beta = np.radians(lattice_params.get('beta', 90.0))
                gamma = np.radians(lattice_params.get('gamma', 90.0))
                
                # Create transformation matrix for fractional to Cartesian coordinates
                cos_alpha = np.cos(alpha)
                cos_beta = np.cos(beta)
                cos_gamma = np.cos(gamma)
                sin_gamma = np.sin(gamma)
                
                # Volume calculation for triclinic case
                volume = a * b * c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2*cos_alpha*cos_beta*cos_gamma)
                
                # Transformation matrix (fractional to Cartesian)
                transform_matrix = np.array([
                    [a, b*cos_gamma, c*cos_beta],
                    [0, b*sin_gamma, c*(cos_alpha - cos_beta*cos_gamma)/sin_gamma],
                    [0, 0, volume/(a*b*sin_gamma)]
                ])
                
                # Convert to Cartesian coordinates
                cart_coords = transform_matrix @ frac_coords
                atom_positions[atom['label']] = (cart_coords[0], cart_coords[1], cart_coords[2])
            
            # Bond colors for different bond types
            bond_colors = {
                'H-H': 'lightgray', 'H-C': 'gray', 'H-N': 'lightblue', 'H-O': 'pink',
                'C-C': 'black', 'C-N': 'darkblue', 'C-O': 'darkred',
                'N-N': 'blue', 'N-O': 'purple', 'O-O': 'red',
                'Si-O': 'orange', 'Si-Si': 'brown',
                'Al-O': 'green', 'Ca-O': 'lime', 'Mg-O': 'yellow',
                'Fe-O': 'maroon', 'Ti-O': 'violet', 'Na-O': 'gold',
                'K-O': 'magenta', 'S-O': 'cyan'
            }
            
            # First, correct ALL bonds for periodic boundary conditions and calculate screen distances
            print(f"Processing {len(bonds)} bonds for periodic boundary corrections...")
            
            # Get lattice transformation matrix once
            lattice_params = self.crystal_structure.get('lattice_parameters', {})
            a = lattice_params.get('a', 5.0)
            b = lattice_params.get('b', 5.0) 
            c = lattice_params.get('c', 5.0)
            alpha = np.radians(lattice_params.get('alpha', 90.0))
            beta = np.radians(lattice_params.get('beta', 90.0))
            gamma = np.radians(lattice_params.get('gamma', 90.0))
            
            # Create transformation matrix for fractional to Cartesian coordinates
            cos_alpha = np.cos(alpha)
            cos_beta = np.cos(beta)
            cos_gamma = np.cos(gamma)
            sin_gamma = np.sin(gamma)
            
            transform_matrix = np.array([
                [a, b*cos_gamma, c*cos_beta],
                [0, b*sin_gamma, c*(cos_alpha - cos_beta*cos_gamma)/sin_gamma],
                [0, 0, c*np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2*cos_alpha*cos_beta*cos_gamma)/sin_gamma]
            ])
            
            # Process all bonds to find correct periodic images and screen distances
            corrected_bonds = []
            for bond in bonds:
                atom1_pos = atom_positions.get(bond['atom1'])
                atom2_pos = atom_positions.get(bond['atom2'])
                
                if not (atom1_pos and atom2_pos):
                    continue
                
                # Find the shortest periodic image
                best_pos1 = atom1_pos
                best_pos2 = atom2_pos
                best_screen_distance_angstrom = float('inf')
                
                # Check all possible periodic images within ±1 unit cell
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            # Calculate position of atom2's periodic image in Cartesian space
                            # Apply lattice vector translations
                            lattice_vector_a = np.array([a, 0, 0])
                            lattice_vector_b = np.array([b*cos_gamma, b*sin_gamma, 0])
                            lattice_vector_c = np.array([c*cos_beta, c*(cos_alpha - cos_beta*cos_gamma)/sin_gamma, 
                                                       c*np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2*cos_alpha*cos_beta*cos_gamma)/sin_gamma])
                            
                            periodic_pos2 = (
                                atom2_pos[0] + dx*lattice_vector_a[0] + dy*lattice_vector_b[0] + dz*lattice_vector_c[0],
                                atom2_pos[1] + dx*lattice_vector_a[1] + dy*lattice_vector_b[1] + dz*lattice_vector_c[1],
                                atom2_pos[2] + dx*lattice_vector_a[2] + dy*lattice_vector_b[2] + dz*lattice_vector_c[2]
                            )
                            
                            # Calculate Cartesian distance directly
                            cart_vector = np.array([periodic_pos2[0] - atom1_pos[0], 
                                                  periodic_pos2[1] - atom1_pos[1], 
                                                  periodic_pos2[2] - atom1_pos[2]])
                            screen_distance_angstrom = np.linalg.norm(cart_vector)
                            
                            if screen_distance_angstrom < best_screen_distance_angstrom:
                                best_screen_distance_angstrom = screen_distance_angstrom
                                best_pos2 = periodic_pos2
                
                # Create corrected bond with actual screen distance
                corrected_bond = bond.copy()
                corrected_bond['screen_distance'] = best_screen_distance_angstrom
                corrected_bond['best_pos1'] = best_pos1
                corrected_bond['best_pos2'] = best_pos2
                corrected_bond['original_distance'] = bond.get('distance', 0.0)
                
                corrected_bonds.append(corrected_bond)
            
            print(f"Corrected {len(corrected_bonds)} bonds for periodic boundaries")
            
            # Now filter bonds based on corrected screen distances
            filtered_bonds = []
            skipped_bonds = []
            
            for bond in corrected_bonds:
                bond_type = bond.get('type', 'Unknown')
                original_distance = bond.get('original_distance', 0.0)
                screen_distance = bond.get('screen_distance', 0.0)
                
                # Apply reasonable distance limits first (using screen distance)
                max_reasonable_distance = 3.5  # Default maximum
                
                # Element-specific reasonable limits
                if 'Ti-O' in bond_type or 'O-Ti' in bond_type:
                    max_reasonable_distance = 2.5
                elif 'Si-O' in bond_type or 'O-Si' in bond_type:
                    max_reasonable_distance = 2.0
                elif 'Al-O' in bond_type or 'O-Al' in bond_type:
                    max_reasonable_distance = 2.2
                elif 'C-C' in bond_type:
                    max_reasonable_distance = 1.8
                elif 'C-O' in bond_type or 'O-C' in bond_type:
                    max_reasonable_distance = 1.8
                elif 'O-O' in bond_type:
                    max_reasonable_distance = 1.8
                
                # Filter by reasonable screen distance
                if not (0.5 <= screen_distance <= max_reasonable_distance):
                    skipped_bonds.append(f"{bond['atom1']}-{bond['atom2']} (screen {screen_distance:.2f}Å unreasonable)")
                    continue
                
                # Apply user bond filters using screen distance
                should_draw = True
                if hasattr(self, 'bond_filters') and self.bond_filters:
                    filter_config = self.bond_filters.get(bond_type, {})
                    
                    # Check if bond type is enabled
                    if not filter_config.get('enabled', True):
                        skipped_bonds.append(f"{bond['atom1']}-{bond['atom2']} (type {bond_type} disabled)")
                        continue
                    
                    # Check distance range using screen distance
                    min_dist = filter_config.get('min_distance', 0.0)
                    max_dist = filter_config.get('max_distance', 10.0)
                    
                    if not (min_dist <= screen_distance <= max_dist):
                        skipped_bonds.append(f"{bond['atom1']}-{bond['atom2']} (screen {screen_distance:.2f}Å outside filter {min_dist:.1f}-{max_dist:.1f}Å)")
                        continue
                
                # Draw the bond using the corrected positions
                # Get bond color
                color = bond_colors.get(bond_type, 'gray')
                
                # Vary line width based on screen distance (shorter bonds = thicker lines)
                linewidth = max(0.5, 3.0 - screen_distance / 2.0)
                
                # Plot 3D line to the correct periodic image
                ax.plot([bond['best_pos1'][0], bond['best_pos2'][0]], 
                       [bond['best_pos1'][1], bond['best_pos2'][1]], 
                       [bond['best_pos1'][2], bond['best_pos2'][2]],
                       color=color, alpha=0.7, linewidth=linewidth)
                
                # Add bond length label at the midpoint of each bond
                mid_x = (bond['best_pos1'][0] + bond['best_pos2'][0]) / 2
                mid_y = (bond['best_pos1'][1] + bond['best_pos2'][1]) / 2
                mid_z = (bond['best_pos1'][2] + bond['best_pos2'][2]) / 2
                
                # Show both the screen distance and original calculated distance
                label_text = f"{screen_distance:.2f}Å"
                if abs(screen_distance - original_distance) > 0.1:
                    label_text += f"\n(calc: {original_distance:.2f}Å)"
                
                ax.text(mid_x, mid_y, mid_z, label_text, 
                       fontsize=6, color='black', weight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                filtered_bonds.append(bond)
            
            # Print summary of filtering
            print(f"Drew {len(filtered_bonds)} bonds, skipped {len(skipped_bonds)} bonds")
            if skipped_bonds:
                print("Sample skipped bonds:")
                for skip_reason in skipped_bonds[:5]:  # Show first 5
                    print(f"  - {skip_reason}")
                if len(skipped_bonds) > 5:
                    print(f"  ... and {len(skipped_bonds) - 5} more")
            
            # Plot atoms on top
            atom_colors = {'C': 'black', 'O': 'red', 'Si': 'blue', 'Al': 'gray', 'Ca': 'green', 
                          'Mg': 'orange', 'Fe': 'brown', 'Ti': 'purple', 'Na': 'yellow', 
                          'K': 'pink', 'H': 'white', 'N': 'lightblue', 'S': 'yellow'}
            
            for atom in atoms:
                element = atom['element']
                color = atom_colors.get(element, 'purple')
                pos = atom_positions[atom['label']]
                
                ax.scatter(pos[0], pos[1], pos[2], c=color, s=200, alpha=0.9, 
                          edgecolors='black', linewidth=1)
                
                # Add atom labels with Wyckoff positions if available
                label = atom['label']
                wyckoff = atom.get('wyckoff_symbol', '')
                if wyckoff and wyckoff != 'unknown':
                    label += f"({wyckoff})"
                ax.text(pos[0], pos[1], pos[2], f"  {label}", fontsize=8)
            
            # Update title to show filtering status
            title = f'3D Bond Network: {self.crystal_structure["name"]}'
            
            # Show filtering statistics
            total_bonds = len(bonds)
            corrected_bond_count = len(corrected_bonds)
            displayed_bonds = len(filtered_bonds)
            
            if total_bonds > displayed_bonds:
                title += f' (Showing {displayed_bonds}/{total_bonds} bonds)'
            
            if hasattr(self, 'bond_filters') and self.bond_filters:
                active_filters = sum(1 for f in self.bond_filters.values() if f.get('enabled', True))
                total_filters = len(self.bond_filters)
                title += f' (Filters: {active_filters}/{total_filters} active)'
            
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title(title)
            
            # Set equal aspect ratio for better visualization based on lattice parameters
            lattice_params = self.crystal_structure.get('lattice_parameters', {})
            a = lattice_params.get('a', 5.0)
            b = lattice_params.get('b', 5.0) 
            c = lattice_params.get('c', 5.0)
            
            # Add some padding around the unit cell
            padding = 0.5  # Angstroms
            ax.set_xlim(-padding, a + padding)
            ax.set_ylim(-padding, b + padding)
            ax.set_zlim(-padding, c + padding)
            
            # Set equal aspect ratio to preserve crystal geometry
            ax.set_box_aspect([a, b, c])
            
            # Bond distances are now shown directly on each bond
            
            # Add legend for bond types if there are filtered bonds
            if filtered_bonds:
                bond_types_shown = set(bond.get('type', 'Unknown') for bond in filtered_bonds)
                if len(bond_types_shown) > 1 and len(bond_types_shown) <= 8:
                    # Create legend handles
                    legend_elements = []
                    for bond_type in sorted(bond_types_shown):
                        color = bond_colors.get(bond_type, 'gray')
                        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=bond_type))
                    
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting 3D bond network:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=10, alpha=0.6)
    
    def get_bond_cutoff(self, element1, element2):
        """Get element-specific bond distance cutoffs."""
        # Define reasonable bond cutoffs for common element pairs (in Angstroms)
        cutoffs = {
            ('Ti', 'O'): 2.5,   # Ti-O bonds typically 1.8-2.2 Å
            ('O', 'Ti'): 2.5,
            ('Si', 'O'): 2.0,   # Si-O bonds typically 1.6-1.8 Å  
            ('O', 'Si'): 2.0,
            ('Al', 'O'): 2.2,   # Al-O bonds typically 1.7-2.0 Å
            ('O', 'Al'): 2.2,
            ('Fe', 'O'): 2.8,   # Fe-O bonds typically 1.9-2.5 Å
            ('O', 'Fe'): 2.8,
            ('Mg', 'O'): 2.5,   # Mg-O bonds typically 2.0-2.3 Å
            ('O', 'Mg'): 2.5,
            ('Ca', 'O'): 3.0,   # Ca-O bonds typically 2.3-2.8 Å
            ('O', 'Ca'): 3.0,
            ('C', 'C'): 1.8,    # C-C bonds typically 1.4-1.6 Å
            ('C', 'O'): 1.8,    # C-O bonds typically 1.2-1.5 Å
            ('O', 'C'): 1.8,
            ('O', 'O'): 1.8,    # O-O bonds (rare) typically 1.2-1.5 Å
            ('H', 'O'): 1.2,    # H-O bonds typically 0.9-1.0 Å
            ('O', 'H'): 1.2,
            ('H', 'H'): 1.0,    # H-H bonds typically 0.7 Å
        }
        
        # Try both orders of elements
        pair = (element1, element2)
        if pair in cutoffs:
            return cutoffs[pair]
        
        # Default cutoff for unknown pairs
        return 3.0
    
    def setup_3d_content_area(self, content_area):
        """Setup the split content area for 3D visualization and spectrum comparison."""
        # Create horizontal paned window
        paned = ttk.PanedWindow(content_area, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: 3D visualization
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=2)
        
        # Right panel: Spectrum comparison
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Setup 3D plot
        self.setup_3d_plot(left_frame)
        
        # Setup spectrum comparison plot
        self.setup_spectrum_comparison_plot(right_frame)
        
        # Initialize 3D visualization variables
        self.crystal_shape_data = None
        self.tensor_data_3d = None
        self.optimized_orientation = None
        self.current_orientation_matrix = np.eye(3)
        
        # Update initial visualization
        self.update_3d_visualization()
        self.update_3d_status()
    
    def setup_3d_plot(self, parent):
        """Setup the 3D matplotlib plot for crystal visualization."""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create matplotlib figure for 3D plot
        self.fig_3d = Figure(figsize=(8, 6), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        
        self.ax_3d.set_xlabel('X (Lab Frame)')
        self.ax_3d.set_ylabel('Y (Lab Frame)')
        self.ax_3d.set_zlabel('Z (Laser Direction)')
        self.ax_3d.set_title('Interactive Crystal Orientation')
        
        # Create canvas
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, parent)
        self.canvas_3d.draw()
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar_3d = NavigationToolbar2Tk(self.canvas_3d, parent)
        self.toolbar_3d.update()
    
    def setup_spectrum_comparison_plot(self, parent):
        """Setup the spectrum comparison plot."""
        # Create matplotlib figure for spectrum
        self.fig_spectrum = Figure(figsize=(6, 6), dpi=100)
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        
        self.ax_spectrum.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_spectrum.set_ylabel('Intensity (a.u.)')
        self.ax_spectrum.set_title('Experimental vs Calculated')
        self.ax_spectrum.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas_spectrum = FigureCanvasTkAgg(self.fig_spectrum, parent)
        self.canvas_spectrum.draw()
        self.canvas_spectrum.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar_spectrum = NavigationToolbar2Tk(self.canvas_spectrum, parent)
        self.toolbar_spectrum.update()
    
    def update_angle_labels(self, *args):
        """Update the angle labels when sliders change."""
        self.phi_label.config(text=f"{self.phi_var.get():.1f}°")
        self.theta_label.config(text=f"{self.theta_var.get():.1f}°")
        self.psi_label.config(text=f"{self.psi_var.get():.1f}°")
    
    def on_orientation_change(self, event=None):
        """Handle orientation changes from sliders."""
        if hasattr(self, 'real_time_calc_var') and self.real_time_calc_var.get():
            self.update_current_orientation()
            self.update_3d_visualization()
            self.calculate_orientation_spectrum()
    
    def update_current_orientation(self):
        """Update the current orientation matrix from Euler angles."""
        phi = np.radians(self.phi_var.get())
        theta = np.radians(self.theta_var.get())
        psi = np.radians(self.psi_var.get())
        
        self.current_orientation_matrix = self.euler_to_rotation_matrix(phi, theta, psi)
        
        # Debug: Print orientation matrix determinant to verify it's changing
        det = np.linalg.det(self.current_orientation_matrix)
        print(f"🔄 Orientation updated: φ={np.degrees(phi):.1f}°, θ={np.degrees(theta):.1f}°, ψ={np.degrees(psi):.1f}° (det={det:.3f})")
    
    def on_orientation_change(self):
        """Called when orientation sliders change - updates visualization and spectrum."""
        try:
            # Update the orientation matrix
            self.update_current_orientation()
            
            # Update 3D visualization
            self.update_3d_visualization()
            
            # Update spectrum if real-time calculation is enabled
            if hasattr(self, 'real_time_calc_var') and self.real_time_calc_var.get():
                self.calculate_orientation_spectrum()
                
        except Exception as e:
            print(f"Error in orientation change handler: {e}")
    
    def import_optimization_orientation(self):
        """Import orientation from optimization results."""
        # Check for optimization results in stage_results
        best_result = None
        
        # Try to get the best result from the most advanced stage completed
        if hasattr(self, 'stage_results') and any(self.stage_results.values()):
            if self.stage_results.get('stage3'):
                # Stage 3 has the most refined result
                stage3 = self.stage_results['stage3']
                best_result = {
                    'euler_angles': stage3['euler_angles'],
                    'rotation_matrix': stage3.get('rotation_matrix')
                }
            elif self.stage_results.get('stage2'):
                # Stage 2 results
                stage2 = self.stage_results['stage2']
                if stage2:
                    best_result = stage2[0]  # Best result from stage 2
            elif self.stage_results.get('stage1'):
                # Stage 1 results
                stage1 = self.stage_results['stage1']
                if stage1:
                    best_result = stage1[0]  # Best result from stage 1
        
        # Also check for legacy optimization_results format
        elif hasattr(self, 'optimization_results') and self.optimization_results:
            best_result = self.optimization_results[0]
        
        if best_result:
            try:
                # Get Euler angles
                if 'euler_angles' in best_result:
                    phi, theta, psi = best_result['euler_angles']
                elif 'orientation' in best_result:
                    # Convert rotation matrix to Euler angles
                    orientation = best_result['orientation']
                    phi, theta, psi = self.rotation_matrix_to_euler(orientation)
                else:
                    raise ValueError("No orientation data found in optimization results")
                
                # Update sliders (convert to degrees if needed)
                if abs(phi) <= 2*np.pi and abs(theta) <= 2*np.pi and abs(psi) <= 2*np.pi:
                    # Angles are in radians, convert to degrees
                    self.phi_var.set(np.degrees(phi))
                    self.theta_var.set(np.degrees(theta))
                    self.psi_var.set(np.degrees(psi))
                else:
                    # Angles are already in degrees
                    self.phi_var.set(phi)
                    self.theta_var.set(theta)
                    self.psi_var.set(psi)
                
                # Store as optimized orientation (convert to rotation matrix if needed)
                if 'rotation_matrix' in best_result:
                    self.optimized_orientation = best_result['rotation_matrix']
                else:
                    # Convert Euler angles to rotation matrix
                    phi_rad = np.radians(self.phi_var.get())
                    theta_rad = np.radians(self.theta_var.get())
                    psi_rad = np.radians(self.psi_var.get())
                    self.optimized_orientation = self.euler_to_rotation_matrix(phi_rad, theta_rad, psi_rad)
                
                self.update_current_orientation()
                self.update_3d_visualization()
                self.update_3d_status()
                
                #messagebox.showinfo("Success", "Imported optimized orientation successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import optimization results: {str(e)}")
        else:
            # Provide helpful guidance
            response = messagebox.askyesno(
                "No Optimization Data", 
                "No optimization results available.\n\n" +
                "Would you like to go to the Orientation Optimization tab to run optimization first?"
            )
            if response:
                # Switch to optimization tab
                for i, tab_name in enumerate(['Spectrum Analysis', 'Peak Fitting', 'Polarization', 
                                            'Crystal Structure', '3D Visualization', 'Tensor Analysis', 
                                            'Orientation Optimization', 'Stress/Strain Analysis']):
                    if tab_name == 'Orientation Optimization':
                        self.notebook.select(i)
                        break
    
    def import_structure_data(self):
        """Import crystal structure data for shape generation."""
        if hasattr(self, 'crystal_structure') and self.crystal_structure:
            try:
                # Set structure_data to point to crystal_structure for compatibility
                self.structure_data = self.crystal_structure.copy()
                
                # Ensure lattice_vectors are available for 3D visualization
                if 'lattice_vectors' not in self.structure_data and 'lattice_parameters' in self.structure_data:
                    lattice_params = self.structure_data['lattice_parameters']
                    if all(key in lattice_params for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
                        a, b, c = lattice_params['a'], lattice_params['b'], lattice_params['c']
                        alpha, beta, gamma = np.radians([lattice_params['alpha'], lattice_params['beta'], lattice_params['gamma']])
                        
                        # Convert to lattice vectors
                        lattice_vectors = np.array([
                            [a, 0, 0],
                            [b * np.cos(gamma), b * np.sin(gamma), 0],
                            [c * np.cos(beta), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
                             c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)]
                        ])
                        self.structure_data['lattice_vectors'] = lattice_vectors
                
                # Ensure atoms have the right coordinate format
                if 'atoms' in self.structure_data:
                    for atom in self.structure_data['atoms']:
                        if 'frac_coords' not in atom and all(key in atom for key in ['x', 'y', 'z']):
                            atom['frac_coords'] = [atom['x'], atom['y'], atom['z']]
                
                # Generate crystal shape based on point group
                self.generate_crystal_shape()
                
                # Automatically enable crystal structure display
                if hasattr(self, 'show_crystal_structure_var'):
                    self.show_crystal_structure_var.set(True)
                
                self.update_3d_visualization()
                self.update_3d_status()
                
                #messagebox.showinfo("Success", "Imported crystal structure data successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import structure data: {str(e)}")
        else:
            # Provide helpful guidance
            response = messagebox.askyesno(
                "No Structure Data", 
                "No crystal structure data available.\n\n" +
                "Would you like to go to the Crystal Structure tab to load structure data first?"
            )
            if response:
                # Switch to crystal structure tab
                for i, tab_name in enumerate(['Spectrum Analysis', 'Peak Fitting', 'Polarization', 
                                            'Crystal Structure', '3D Visualization', 'Tensor Analysis', 
                                            'Orientation Optimization', 'Stress/Strain Analysis']):
                    if tab_name == 'Crystal Structure':
                        self.notebook.select(i)
                        break
    
    def import_tensor_data_3d(self):
        """Import Raman tensor data for visualization."""
        # Store existing tensor data as backup
        existing_tensor_data = getattr(self, 'tensor_data_3d', None)
        existing_raman_tensors = getattr(self, 'raman_tensors', {})
        
        # Check if we already have good tensor data
        if existing_tensor_data is not None and existing_raman_tensors:
            response = messagebox.askyesno(
                "Tensor Data Exists", 
                "You already have tensor data loaded.\n\n" +
                "Do you want to overwrite it with data from the Tensor Analysis tab?\n\n" +
                "Click 'No' to keep your current tensor data."
            )
            if not response:
                messagebox.showinfo("Cancelled", "Keeping existing tensor data.")
                return
        
        # Try to import from various sources
        imported = False
        
        # Source 1: Tensor Analysis tab data
        if hasattr(self, 'raman_tensors') and bool(self.raman_tensors):
            try:
                self.tensor_data_3d = self.raman_tensors.copy()
                imported = True
                source = "Tensor Analysis tab"
            except Exception as e:
                print(f"Failed to import from tensor analysis: {e}")
        
        # Source 2: Try to create from fitted peaks if no tensor data
        if not imported and hasattr(self, 'fitted_peaks') and self.fitted_peaks:
            try:
                success = self.create_tensors_from_fitted_peaks()
                if success:
                    imported = True
                    source = "fitted peaks"
            except Exception as e:
                print(f"Failed to create from fitted peaks: {e}")
        
        # Source 3: Try mineral database
        if not imported and hasattr(self, 'mineral_data') and self.mineral_data:
            try:
                success = self.create_tensors_from_mineral_data()
                if success:
                    imported = True
                    source = "mineral database"
            except Exception as e:
                print(f"Failed to create from mineral data: {e}")
        
        if imported:
            try:
                self.update_peak_selection_3d()
                self.update_3d_visualization()
                self.update_3d_status()
                
                #messagebox.showinfo("Success", f"Imported tensor data from {source}!")
                
            except Exception as e:
                # Restore backup data if update fails
                if existing_tensor_data is not None:
                    self.tensor_data_3d = existing_tensor_data
                    self.raman_tensors = existing_raman_tensors
                messagebox.showerror("Error", f"Failed to update visualization: {str(e)}")
        else:
            # Restore backup data if nothing was imported
            if existing_tensor_data is not None:
                self.tensor_data_3d = existing_tensor_data
                self.raman_tensors = existing_raman_tensors
                messagebox.showinfo("No Changes", "No new tensor data found. Keeping existing data.")
            else:
                # Provide helpful guidance
                response = messagebox.askyesno(
                    "No Tensor Data", 
                    "No tensor data available from any source.\n\n" +
                    "Would you like to:\n" +
                    "• Go to Tensor Analysis tab to calculate tensors, OR\n" +
                    "• Use Demo Data to test the interface?"
                )
                if response:
                    # Switch to tensor analysis tab
                    for i, tab_name in enumerate(['Spectrum Analysis', 'Peak Fitting', 'Polarization', 
                                                'Crystal Structure', 'Tensor Analysis & Visualization', 
                                                'Orientation Optimization', 'Stress/Strain Analysis', '3D Visualization']):
                        if 'Tensor Analysis' in tab_name:
                            self.notebook.select(i)
                            break
    
    def update_peak_selection_3d(self):
        """Update the peak selection combobox for 3D visualization."""
        peak_options = ["All Peaks"]
        
        # Priority 1: Use fitted peaks from Peak Fitting tab if available
        if hasattr(self, 'fitted_peaks') and self.fitted_peaks:
            try:
                fitted_wavenumbers = []
                for peak in self.fitted_peaks:
                    if isinstance(peak, dict) and 'center' in peak:
                        fitted_wavenumbers.append(peak['center'])
                    elif hasattr(peak, 'center'):
                        fitted_wavenumbers.append(peak.center)
                if fitted_wavenumbers:
                    peak_options.extend([f"{w:.1f} (Fitted)" for w in fitted_wavenumbers])
            except Exception as e:
                print(f"Error processing fitted peaks: {e}")
        
        # Priority 2: Use tensor data if available
        elif self.tensor_data_3d and 'wavenumbers' in self.tensor_data_3d:
            wavenumbers = self.tensor_data_3d['wavenumbers']
            # Filter by intensity if available
            if 'intensities' in self.tensor_data_3d:
                intensities = self.tensor_data_3d['intensities']
                # Only show peaks with intensity > 10% of maximum
                max_intensity = np.max(intensities)
                threshold = max_intensity * 0.1
                filtered_peaks = [(w, i) for w, i in zip(wavenumbers, intensities) if i > threshold]
                # Sort by intensity (strongest first)
                filtered_peaks.sort(key=lambda x: x[1], reverse=True)
                # Limit to top 10 peaks to avoid clutter
                filtered_peaks = filtered_peaks[:10]
                peak_options.extend([f"{w:.1f} (I={i:.1f})" for w, i in filtered_peaks])
            else:
                # Just show the wavenumbers (limit to first 10)
                peak_options.extend([f"{w:.1f}" for w in wavenumbers[:10]])
        
        # Priority 3: Use database peaks with intensity filter
        elif hasattr(self, 'mineral_data') and self.mineral_data:
            try:
                # Get peaks from database with intensity filtering
                peaks = []
                for peak_data in self.mineral_data.get('peaks', []):
                    if isinstance(peak_data, dict):
                        wavenumber = peak_data.get('wavenumber', 0)
                        intensity = peak_data.get('I_tot', peak_data.get('intensity', 0))
                        if intensity > 10:  # Only show peaks with I_tot > 10
                            peaks.append((wavenumber, intensity))
                
                # Sort by intensity and limit to top 15
                peaks.sort(key=lambda x: x[1], reverse=True)
                peaks = peaks[:15]
                peak_options.extend([f"{w:.1f} (I={i:.0f})" for w, i in peaks])
            except Exception as e:
                print(f"Error filtering database peaks: {e}")
        
        # Update combobox
        self.viz_peak_combo['values'] = peak_options
        if not self.viz_peak_var.get() or self.viz_peak_var.get() not in peak_options:
            self.viz_peak_var.set("All Peaks")
    
    def reset_to_optimized_orientation(self):
        """Reset crystal orientation to the optimized values."""
        if self.optimized_orientation is not None:
            try:
                # Convert rotation matrix to Euler angles
                phi, theta, psi = self.rotation_matrix_to_euler(self.optimized_orientation)
                
                # Update sliders
                self.phi_var.set(np.degrees(phi))
                self.theta_var.set(np.degrees(theta))
                self.psi_var.set(np.degrees(psi))
                
                self.update_current_orientation()
                self.update_3d_visualization()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset orientation: {str(e)}")
        else:
            messagebox.showwarning("No Data", "No optimized orientation available. Import from optimization first.")
    
    def set_crystal_direction(self, direction):
        """Set crystal orientation to align a specific direction with the laser."""
        try:
            # Calculate rotation matrix to align direction with z-axis
            direction = np.array(direction, dtype=float)
            direction = direction / np.linalg.norm(direction)
            
            # Target is z-axis (laser direction)
            target = np.array([0, 0, 1])
            
            # Calculate rotation axis and angle
            if np.allclose(direction, target):
                rotation_matrix = np.eye(3)
            elif np.allclose(direction, -target):
                rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            else:
                axis = np.cross(direction, target)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(direction, target))
                
                # Rodrigues' rotation formula
                K = np.array([[0, -axis[2], axis[1]],
                             [axis[2], 0, -axis[0]],
                             [-axis[1], axis[0], 0]])
                
                rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            
            # Convert to Euler angles
            phi, theta, psi = self.rotation_matrix_to_euler(rotation_matrix)
            
            # Update sliders
            self.phi_var.set(np.degrees(phi))
            self.theta_var.set(np.degrees(theta))
            self.psi_var.set(np.degrees(psi))
            
            self.update_current_orientation()
            self.update_3d_visualization()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set crystal direction: {str(e)}")
    
    def rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to Euler angles (ZYX convention)."""
        # Extract Euler angles from rotation matrix
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            phi = np.arctan2(R[2,1], R[2,2])
            theta = np.arctan2(-R[2,0], sy)
            psi = np.arctan2(R[1,0], R[0,0])
        else:
            phi = np.arctan2(-R[1,2], R[1,1])
            theta = np.arctan2(-R[2,0], sy)
            psi = 0
        
        return phi, theta, psi
    
    def generate_crystal_shape(self):
        """Generate crystal shape based on point group symmetry."""
        try:
            # Get crystal system from structure data
            crystal_system = "Cubic"  # Default
            if hasattr(self, 'structure_data') and self.structure_data:
                crystal_system = self.structure_data.get('crystal_system', 'Cubic')
            elif hasattr(self, 'crystal_structure') and self.crystal_structure:
                crystal_system = self.get_crystal_system_from_structure()
            
            # Generate basic crystal shapes based on crystal system
            if crystal_system.lower() == "cubic":
                self.crystal_shape_data = self.generate_cubic_shape()
            elif crystal_system.lower() == "tetragonal":
                self.crystal_shape_data = self.generate_tetragonal_shape()
            elif crystal_system.lower() == "orthorhombic":
                self.crystal_shape_data = self.generate_orthorhombic_shape()
            elif crystal_system.lower() == "hexagonal":
                self.crystal_shape_data = self.generate_hexagonal_shape()
            elif crystal_system.lower() == "trigonal":
                self.crystal_shape_data = self.generate_trigonal_shape()
            elif crystal_system.lower() == "monoclinic":
                self.crystal_shape_data = self.generate_monoclinic_shape()
            elif crystal_system.lower() == "triclinic":
                self.crystal_shape_data = self.generate_triclinic_shape()
            else:
                self.crystal_shape_data = self.generate_cubic_shape()  # Default
                
        except Exception as e:
            print(f"Error generating crystal shape: {e}")
            self.crystal_shape_data = self.generate_cubic_shape()  # Fallback
    
    def generate_cubic_shape(self):
        """Generate a cubic crystal shape."""
        # Create vertices of a cube
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # top face
        ]) * 0.5
        
        # Define faces (triangles)
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2]   # right
        ]
        
        return {'vertices': vertices, 'faces': faces, 'type': 'cubic'}
    
    def generate_tetragonal_shape(self):
        """Generate a tetragonal crystal shape."""
        # Create vertices of a tetragonal prism (elongated along z)
        vertices = np.array([
            [-1, -1, -1.5], [1, -1, -1.5], [1, 1, -1.5], [-1, 1, -1.5],  # bottom face
            [-1, -1, 1.5], [1, -1, 1.5], [1, 1, 1.5], [-1, 1, 1.5]       # top face
        ]) * 0.5
        
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2]   # right
        ]
        
        return {'vertices': vertices, 'faces': faces, 'type': 'tetragonal'}
    
    def generate_hexagonal_shape(self):
        """Generate a hexagonal crystal shape."""
        # Create hexagonal prism
        angles = np.linspace(0, 2*np.pi, 7)  # 6 sides + closing
        radius = 1.0
        height = 1.5
        
        # Bottom hexagon
        bottom_vertices = []
        for i in range(6):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            bottom_vertices.append([x, y, -height/2])
        
        # Top hexagon
        top_vertices = []
        for i in range(6):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            top_vertices.append([x, y, height/2])
        
        vertices = np.array(bottom_vertices + top_vertices)
        
        # Create faces
        faces = []
        # Bottom face (triangulated)
        for i in range(1, 5):
            faces.append([0, i, i+1])
        
        # Top face (triangulated)
        for i in range(7, 11):
            faces.append([6, i+1, i])
        
        # Side faces
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([i, i+6, next_i+6])
            faces.append([i, next_i+6, next_i])
        
        return {'vertices': vertices, 'faces': faces, 'type': 'hexagonal'}
    
    def generate_orthorhombic_shape(self):
        """Generate an orthorhombic crystal shape."""
        # Create vertices of an orthorhombic prism (different dimensions along each axis)
        vertices = np.array([
            [-1.2, -1.0, -0.8], [1.2, -1.0, -0.8], [1.2, 1.0, -0.8], [-1.2, 1.0, -0.8],  # bottom face
            [-1.2, -1.0, 0.8], [1.2, -1.0, 0.8], [1.2, 1.0, 0.8], [-1.2, 1.0, 0.8]       # top face
        ]) * 0.5
        
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2]   # right
        ]
        
        return {'vertices': vertices, 'faces': faces, 'type': 'orthorhombic'}
    
    def generate_trigonal_shape(self):
        """Generate a trigonal crystal shape."""
        # Create a rhombohedron (trigonal shape)
        # Use hexagonal prism as approximation
        return self.generate_hexagonal_shape()
    
    def generate_monoclinic_shape(self):
        """Generate a monoclinic crystal shape."""
        # Create a skewed prism
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1.2, 1, -1], [-0.8, 1, -1],  # bottom face (skewed)
            [-1, -1, 1], [1, -1, 1], [1.2, 1, 1], [-0.8, 1, 1]       # top face (skewed)
        ]) * 0.5
        
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2]   # right
        ]
        
        return {'vertices': vertices, 'faces': faces, 'type': 'monoclinic'}
    
    def generate_triclinic_shape(self):
        """Generate a triclinic crystal shape."""
        # Create a general parallelepiped (most general case)
        vertices = np.array([
            [-1, -1, -1], [1.1, -0.9, -1.1], [1.2, 1.1, -0.9], [-0.9, 1, -1],  # bottom face (skewed)
            [-1.1, -0.9, 1], [1, -1, 1.1], [1.1, 1, 1], [-1, 1.1, 0.9]         # top face (skewed)
        ]) * 0.5
        
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2]   # right
        ]
        
        return {'vertices': vertices, 'faces': faces, 'type': 'triclinic'}
    
    def update_3d_visualization(self):
        """Update the 3D visualization with current orientation."""
        if not hasattr(self, 'ax_3d'):
            return
            
        try:
            # Clear the plot
            self.ax_3d.clear()
            
            # Set labels and title
            self.ax_3d.set_xlabel('X (Lab Frame)')
            self.ax_3d.set_ylabel('Y (Lab Frame)')
            self.ax_3d.set_zlabel('Z (Laser Direction)')
            self.ax_3d.set_title('Interactive Crystal Orientation')
            
            # Plot coordinate axes if enabled
            if hasattr(self, 'show_axes_var') and self.show_axes_var.get():
                self.plot_coordinate_axes()
            
            # Plot laser geometry if enabled
            if hasattr(self, 'show_laser_geometry_var') and self.show_laser_geometry_var.get():
                self.plot_laser_geometry()
            
            # Plot crystal shape if enabled and available
            if (hasattr(self, 'show_crystal_var') and self.show_crystal_var.get() and 
                self.crystal_shape_data is not None):
                self.plot_crystal_shape()
            
            # Plot crystal structure if enabled and available
            structure_available = ((hasattr(self, 'structure_data') and self.structure_data is not None) or
                                  (hasattr(self, 'crystal_structure') and self.crystal_structure is not None))
            if (hasattr(self, 'show_crystal_structure_var') and self.show_crystal_structure_var.get() and 
                structure_available):
                self.plot_crystal_structure_3d()
            
            # Plot Raman tensor ellipsoid if enabled and available
            if (hasattr(self, 'show_tensor_ellipsoid_var') and self.show_tensor_ellipsoid_var.get() and 
                self.tensor_data_3d is not None):
                self.plot_raman_tensor_ellipsoid_3d()
            
            # Set equal aspect ratio and limits
            max_range = 2.0
            self.ax_3d.set_xlim([-max_range, max_range])
            self.ax_3d.set_ylim([-max_range, max_range])
            self.ax_3d.set_zlim([-max_range, max_range])
            
            # Refresh canvas
            self.canvas_3d.draw()
            
            # Update spectrum if real-time calculation is enabled
            if hasattr(self, 'real_time_calc_var') and self.real_time_calc_var.get():
                self.calculate_orientation_spectrum()
            
        except Exception as e:
            print(f"Error updating 3D visualization: {e}")
    
    def plot_raman_tensor_ellipsoid_3d(self):
        """Plot Raman tensor ellipsoid in the 3D visualization."""
        if self.tensor_data_3d is None:
            return
            
        try:
            # Get tensor data
            if 'tensors' not in self.tensor_data_3d or 'wavenumbers' not in self.tensor_data_3d:
                return
                
            wavenumbers = self.tensor_data_3d['wavenumbers']
            tensors = self.tensor_data_3d['tensors']
            
            # Get selected peak or use first tensor
            selected_peak = getattr(self, 'viz_peak_var', tk.StringVar(value="All Peaks")).get()
            
            if selected_peak == "All Peaks" and len(tensors) > 0:
                # Use the first tensor or the strongest one
                tensor_index = 0
            else:
                # Find the tensor corresponding to the selected peak
                tensor_index = 0
                try:
                    # Extract wavenumber from selection (handle formatted strings)
                    if "(" in selected_peak:
                        selected_wavenumber = float(selected_peak.split("(")[0].strip())
                    else:
                        selected_wavenumber = float(selected_peak)
                    # Find closest wavenumber
                    differences = [abs(w - selected_wavenumber) for w in wavenumbers]
                    tensor_index = differences.index(min(differences))
                except (ValueError, IndexError):
                    tensor_index = 0
            
            if tensor_index < len(tensors):
                tensor = tensors[tensor_index]
                
                # Apply current orientation to tensor
                rotated_tensor = np.dot(self.current_orientation_matrix, 
                                      np.dot(tensor, self.current_orientation_matrix.T))
                
                # Get tensor scale
                tensor_scale = getattr(self, 'tensor_scale_var', tk.DoubleVar(value=0.5)).get()
                
                # Plot the tensor ellipsoid
                self.plot_tensor_ellipsoid_3d_helper(rotated_tensor, tensor_scale, wavenumbers[tensor_index])
                
        except Exception as e:
            print(f"Error plotting tensor ellipsoid: {e}")
    
    def plot_tensor_ellipsoid_3d_helper(self, tensor, scale=0.5, wavenumber=None):
        """Helper function to plot a 3D tensor ellipsoid."""
        try:
            # Calculate eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(tensor)
            
            # Ensure positive eigenvalues for visualization
            eigenvals = np.abs(eigenvals)
            
            # Create ellipsoid surface
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            
            # Parametric equations for ellipsoid
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Scale by eigenvalues
            max_eigenval = np.max(eigenvals)
            if max_eigenval > 0:
                # Normalize eigenvalues for better visualization
                normalized_eigenvals = eigenvals / max_eigenval * scale
                
                # Apply scaling
                x_ellipsoid = x_sphere * normalized_eigenvals[0]
                y_ellipsoid = y_sphere * normalized_eigenvals[1]
                z_ellipsoid = z_sphere * normalized_eigenvals[2]
                
                # Rotate by eigenvectors
                for i in range(x_ellipsoid.shape[0]):
                    for j in range(x_ellipsoid.shape[1]):
                        point = np.array([x_ellipsoid[i,j], y_ellipsoid[i,j], z_ellipsoid[i,j]])
                        rotated_point = np.dot(eigenvecs, point)
                        x_ellipsoid[i,j] = rotated_point[0]
                        y_ellipsoid[i,j] = rotated_point[1]
                        z_ellipsoid[i,j] = rotated_point[2]
                
                # Plot the ellipsoid surface
                self.ax_3d.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, 
                                       alpha=0.6, color='red', linewidth=0.5)
                
                # Plot principal axes
                axis_length = scale * 1.2
                colors = ['red', 'green', 'blue']
                labels = ['Principal Axis 1', 'Principal Axis 2', 'Principal Axis 3']
                
                for i, (eigenval, eigenvec) in enumerate(zip(normalized_eigenvals, eigenvecs.T)):
                    if eigenval > 0.01:  # Only plot significant axes
                        axis_end = eigenvec * eigenval * axis_length
                        self.ax_3d.quiver(0, 0, 0, axis_end[0], axis_end[1], axis_end[2], 
                                         color=colors[i], alpha=0.8, linewidth=2, 
                                         label=f'{labels[i]} ({eigenval:.3f})')
                
                # Add legend for tensor info
                if wavenumber is not None:
                    self.ax_3d.text2D(0.02, 0.98, f'Tensor: {wavenumber:.1f} cm⁻¹', 
                                     transform=self.ax_3d.transAxes, fontsize=10, 
                                     verticalalignment='top', bbox=dict(boxstyle='round', 
                                     facecolor='white', alpha=0.8))
                
        except Exception as e:
            print(f"Error in tensor ellipsoid helper: {e}")
    
    def plot_coordinate_axes(self):
        """Plot coordinate system axes."""
        # Lab frame axes (fixed)
        axis_length = 1.5
        
        # X-axis (red)
        self.ax_3d.quiver(0, 0, 0, axis_length, 0, 0, color='red', alpha=0.7, linewidth=3, label='X (Lab)')
        
        # Y-axis (green)
        self.ax_3d.quiver(0, 0, 0, 0, axis_length, 0, color='green', alpha=0.7, linewidth=3, label='Y (Lab)')
        
        # Z-axis (blue) - laser direction
        self.ax_3d.quiver(0, 0, 0, 0, 0, axis_length, color='blue', alpha=0.7, linewidth=3, label='Z (Laser)')
        
        # Crystal frame axes (rotated)
        if hasattr(self, 'current_orientation_matrix'):
            crystal_axes = self.current_orientation_matrix * axis_length * 0.8
            
            # Crystal X-axis (pink)
            self.ax_3d.quiver(0, 0, 0, crystal_axes[0,0], crystal_axes[1,0], crystal_axes[2,0], 
                             color='pink', alpha=0.8, linewidth=2, linestyle='--', label='X (Crystal)')
            
            # Crystal Y-axis (lightgreen)
            self.ax_3d.quiver(0, 0, 0, crystal_axes[0,1], crystal_axes[1,1], crystal_axes[2,1], 
                             color='lightgreen', alpha=0.8, linewidth=2, linestyle='--', label='Y (Crystal)')
            
            # Crystal Z-axis (lightblue)
            self.ax_3d.quiver(0, 0, 0, crystal_axes[0,2], crystal_axes[1,2], crystal_axes[2,2], 
                             color='lightblue', alpha=0.8, linewidth=2, linestyle='--', label='Z (Crystal)')
    
    def plot_laser_geometry(self):
        """Plot laser beam and scattering geometry."""
        # Incident laser beam (along +Z)
        beam_length = 2.5
        self.ax_3d.quiver(0, 0, -beam_length, 0, 0, beam_length, 
                         color='orange', alpha=0.6, linewidth=4, label='Incident Laser')
        
        # Scattered light (backscattering along -Z)
        self.ax_3d.quiver(0, 0, 0, 0, 0, -beam_length*0.7, 
                         color='yellow', alpha=0.6, linewidth=3, label='Scattered Light')
        
        # Polarization vectors (example: linear polarization along X)
        pol_length = 0.8
        self.ax_3d.quiver(0, 0, 0, pol_length, 0, 0, 
                         color='purple', alpha=0.8, linewidth=2, label='Incident Pol.')
        self.ax_3d.quiver(0, 0, 0, pol_length, 0, 0, 
                         color='magenta', alpha=0.8, linewidth=2, linestyle=':', label='Scattered Pol.')
    
    def plot_crystal_shape(self):
        """Plot the crystal shape with current orientation."""
        if self.crystal_shape_data is None:
            return
            
        try:
            # Get vertices and faces
            vertices = self.crystal_shape_data['vertices'].copy()
            faces = self.crystal_shape_data['faces']
            
            # Apply current orientation
            rotated_vertices = np.dot(vertices, self.current_orientation_matrix.T)
            
            # Plot faces
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            face_vertices = []
            for face in faces:
                face_verts = [rotated_vertices[i] for i in face]
                face_vertices.append(face_verts)
            
            # Create collection with transparency
            poly3d = Poly3DCollection(face_vertices, alpha=0.3, facecolor='cyan', edgecolor='darkblue')
            self.ax_3d.add_collection3d(poly3d)
            
            # Plot vertices as points
            self.ax_3d.scatter(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], 
                              c='darkblue', s=20, alpha=0.8)
            
        except Exception as e:
            print(f"Error plotting crystal shape: {e}")
    
    def plot_crystal_structure_3d(self):
        """Plot the crystal structure with atoms and bonds in the specified unit cell range."""
        # Check for structure data in either location
        structure_data = None
        if hasattr(self, 'structure_data') and self.structure_data is not None:
            structure_data = self.structure_data
        elif hasattr(self, 'crystal_structure') and self.crystal_structure is not None:
            structure_data = self.crystal_structure
        
        if structure_data is None:
            return
            
        try:
            # Get unit cell range from controls
            cell_range = getattr(self, 'unit_cell_range_var', tk.DoubleVar(value=2.0)).get()
            atom_size = getattr(self, 'atom_size_var', tk.DoubleVar(value=0.3)).get()
            compression = getattr(self, 'structure_compression_var', tk.DoubleVar(value=1.0)).get()
            show_bonds = getattr(self, 'show_bonds_var', tk.BooleanVar(value=True)).get()
            show_unit_cell = getattr(self, 'show_unit_cell_edges_var', tk.BooleanVar(value=True)).get()

            
            # Get structure data
            if 'atoms' not in structure_data:
                print("Structure data incomplete for 3D plotting - no atoms")
                return
            
            atoms = structure_data['atoms']
            
            # Get lattice vectors - try different possible keys
            lattice_vectors = None
            if 'lattice_vectors' in structure_data:
                lattice_vectors = np.array(structure_data['lattice_vectors'])
            elif 'lattice_parameters' in structure_data:
                # Convert lattice parameters to vectors
                lattice_params = structure_data['lattice_parameters']
                if all(key in lattice_params for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
                    a, b, c = lattice_params['a'], lattice_params['b'], lattice_params['c']
                    alpha, beta, gamma = np.radians([lattice_params['alpha'], lattice_params['beta'], lattice_params['gamma']])
                    
                    # Convert to lattice vectors
                    lattice_vectors = np.array([
                        [a, 0, 0],
                        [b * np.cos(gamma), b * np.sin(gamma), 0],
                        [c * np.cos(beta), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
                         c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)]
                    ])
            
            if lattice_vectors is None:
                print("Structure data incomplete for 3D plotting - no lattice vectors")
                return
            
            # Generate atoms in the specified range
            expanded_atoms = self.generate_expanded_structure(atoms, lattice_vectors, cell_range, compression)
            
            # Apply current crystal orientation
            rotated_lattice = np.dot(lattice_vectors, self.current_orientation_matrix.T)
            
            # Plot atoms
            if expanded_atoms:
                self.plot_atoms_3d(expanded_atoms, atom_size)
            else:
                print("No atoms to plot in 3D visualization")
            
            # Plot bonds if enabled
            if show_bonds:
                self.plot_bonds_3d(expanded_atoms)
            
            # Plot unit cell edges if enabled
            if show_unit_cell:
                # Apply compression to lattice vectors
                compressed_lattice = rotated_lattice * compression
                self.plot_unit_cell_edges_3d(compressed_lattice, cell_range)
                
        except Exception as e:
            print(f"Error plotting crystal structure: {e}")
    
    def generate_expanded_structure(self, atoms, lattice_vectors, cell_range, compression=1.0):
        """Generate atoms in the expanded unit cell range with optional compression."""
        expanded_atoms = []
        
        # Create range of unit cell translations
        # Use a more generous range to ensure we capture all atoms that might be visible
        cell_range_int = int(np.ceil(cell_range)) + 1  # Add extra buffer
        translations = []
        for i in range(-cell_range_int, cell_range_int + 1):
            for j in range(-cell_range_int, cell_range_int + 1):
                for k in range(-cell_range_int, cell_range_int + 1):
                    translations.append([i, j, k])
        
        # Generate atoms for each translation
        for translation in translations:
            translation_vector = np.dot(translation, lattice_vectors)
            
            for atom in atoms:
                # Get fractional coordinates - try multiple possible formats
                frac_coords = None
                
                if 'frac_coords' in atom:
                    frac_coords = np.array(atom['frac_coords'])
                elif 'cart_coords' in atom:
                    # Convert Cartesian to fractional
                    cart_coords = np.array(atom['cart_coords'])
                    frac_coords = np.linalg.solve(lattice_vectors.T, cart_coords)
                elif 'cartesian_coords' in atom:
                    # Convert Cartesian to fractional
                    cart_coords = np.array(atom['cartesian_coords'])
                    frac_coords = np.linalg.solve(lattice_vectors.T, cart_coords)
                elif all(key in atom for key in ['x', 'y', 'z']):
                    # Use x, y, z as fractional coordinates
                    frac_coords = np.array([atom['x'], atom['y'], atom['z']])
                else:
                    continue
                
                if frac_coords is None:
                    continue
                
                # Calculate Cartesian position
                cart_pos = np.dot(frac_coords, lattice_vectors) + translation_vector
                
                # Apply compression factor
                compressed_pos = cart_pos * compression
                
                # Apply crystal orientation
                rotated_pos = np.dot(compressed_pos, self.current_orientation_matrix.T)
                
                # Check if atom is within the display range (adjusted for compression)
                display_range = cell_range * compression + 0.1
                if np.all(np.abs(rotated_pos) <= display_range):
                    expanded_atom = {
                        'element': atom.get('element', 'C'),
                        'position': rotated_pos,
                        'original_atom': atom
                    }
                    expanded_atoms.append(expanded_atom)

        
        return expanded_atoms
    
    def plot_atoms_3d(self, atoms, atom_size):
        """Plot atoms as spheres with element-specific colors."""
        # Element colors (CPK coloring scheme)
        element_colors = {
            'H': '#FFFFFF',   # White
            'C': '#909090',   # Gray
            'N': '#3050F8',   # Blue
            'O': '#FF0D0D',   # Red
            'F': '#90E050',   # Green
            'Na': '#AB5CF2',  # Violet
            'Mg': '#8AFF00',  # Green
            'Al': '#BFA6A6',  # Gray
            'Si': '#F0C8A0',  # Tan
            'P': '#FF8000',   # Orange
            'S': '#FFFF30',   # Yellow
            'Cl': '#1FF01F',  # Green
            'K': '#8F40D4',   # Violet
            'Ca': '#3DFF00',  # Green
            'Ti': '#BFC2C7',  # Gray
            'Fe': '#E06633',  # Orange
            'Cu': '#C88033',  # Orange
            'Zn': '#7D80B0',  # Blue
            'default': '#FF1493'  # Deep pink for unknown elements
        }
        
        # Group atoms by element for efficient plotting
        element_groups = {}
        for atom in atoms:
            element = atom['element']
            if element not in element_groups:
                element_groups[element] = {'positions': [], 'color': element_colors.get(element, element_colors['default'])}
            element_groups[element]['positions'].append(atom['position'])
        
        # Plot each element group
        for element, group in element_groups.items():
            positions = np.array(group['positions'])
            if len(positions) > 0:
                self.ax_3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                 c=group['color'], s=atom_size*1000, alpha=0.8, 
                                 label=f'{element} atoms', edgecolors='black', linewidth=0.5)
    
    def plot_bonds_3d(self, atoms):
        """Plot bonds between atoms with filtering support."""
        # Bond distance thresholds (in Angstroms)
        bond_thresholds = {
            ('H', 'H'): 1.0, ('H', 'C'): 1.3, ('H', 'N'): 1.2, ('H', 'O'): 1.1,
            ('C', 'C'): 1.8, ('C', 'N'): 1.6, ('C', 'O'): 1.5,
            ('N', 'N'): 1.6, ('N', 'O'): 1.5,
            ('O', 'O'): 1.6,
            ('Si', 'O'): 2.0, ('Si', 'Si'): 2.5,
            ('Al', 'O'): 2.1, ('Ca', 'O'): 2.8, ('Mg', 'O'): 2.3,
            ('Fe', 'O'): 2.4, ('Ti', 'O'): 2.2, ('Na', 'O'): 2.6,
            ('K', 'O'): 3.0, ('S', 'O'): 1.8,
            ('default'): 2.5  # Default bond threshold
        }
        
        # Find bonds
        bonds = []
        for i, atom1 in enumerate(atoms):
            for j, atom2 in enumerate(atoms[i+1:], i+1):
                distance = np.linalg.norm(atom1['position'] - atom2['position'])
                
                # Get bond threshold
                elements = tuple(sorted([atom1['element'], atom2['element']]))
                threshold = bond_thresholds.get(elements, bond_thresholds.get(('default')))
                
                if distance < threshold:
                    # Create bond type identifier
                    bond_type = f"{elements[0]}-{elements[1]}"
                    
                    # Check bond filters
                    if hasattr(self, 'bond_filters') and self.bond_filters:
                        filter_config = self.bond_filters.get(bond_type, {})
                        
                        # Check if bond type is enabled
                        if not filter_config.get('enabled', True):
                            continue
                        
                        # Check distance range
                        min_dist = filter_config.get('min_distance', 0.0)
                        max_dist = filter_config.get('max_distance', 10.0)
                        
                        if not (min_dist <= distance <= max_dist):
                            continue
                    
                    bonds.append({
                        'positions': (atom1['position'], atom2['position']),
                        'type': bond_type,
                        'distance': distance,
                        'elements': elements
                    })
        
        # Plot bonds with different colors for different bond types
        bond_colors = {
            'H-H': 'lightgray', 'H-C': 'gray', 'H-N': 'lightblue', 'H-O': 'pink',
            'C-C': 'black', 'C-N': 'darkblue', 'C-O': 'darkred',
            'N-N': 'blue', 'N-O': 'purple', 'O-O': 'red',
            'Si-O': 'orange', 'Si-Si': 'brown',
            'Al-O': 'green', 'Ca-O': 'lime', 'Mg-O': 'yellow',
            'Fe-O': 'maroon', 'Ti-O': 'violet', 'Na-O': 'gold',
            'K-O': 'magenta', 'S-O': 'cyan'
        }
        
        for bond in bonds:
            pos1, pos2 = bond['positions']
            bond_type = bond['type']
            color = bond_colors.get(bond_type, 'gray')
            
            # Vary line width based on bond type (shorter bonds = thicker lines)
            linewidth = max(0.5, 2.0 - bond['distance'] / 2.0)
            
            self.ax_3d.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                           color=color, alpha=0.6, linewidth=linewidth)
    
    def plot_unit_cell_edges_3d(self, lattice_vectors, cell_range):
        """Plot unit cell edges."""
        # Define unit cell vertices
        vertices = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    vertex = i * lattice_vectors[0] + j * lattice_vectors[1] + k * lattice_vectors[2]
                    vertices.append(vertex)
        
        vertices = np.array(vertices)
        
        # Plot unit cell edges for multiple cells
        cell_range_int = int(np.ceil(cell_range))
        for i in range(-cell_range_int, cell_range_int + 1):
            for j in range(-cell_range_int, cell_range_int + 1):
                for k in range(-cell_range_int, cell_range_int + 1):
                    translation = i * lattice_vectors[0] + j * lattice_vectors[1] + k * lattice_vectors[2]
                    translated_vertices = vertices + translation
                    
                    # Check if any vertex is within display range
                    if np.any(np.all(np.abs(translated_vertices) <= cell_range + 0.5, axis=1)):
                        self.plot_single_unit_cell_edges(translated_vertices)
    
    def plot_single_unit_cell_edges(self, vertices):
        """Plot edges of a single unit cell."""
        # Define the 12 edges of a unit cell
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
            (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            self.ax_3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                           'b--', alpha=0.4, linewidth=1)
    
    def calculate_orientation_spectrum(self):
        """Calculate Raman spectrum for current crystal orientation."""
        if not hasattr(self, 'ax_spectrum') or self.tensor_data_3d is None:
            return
            
        try:
            # Clear spectrum plot
            self.ax_spectrum.clear()
            self.ax_spectrum.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_spectrum.set_ylabel('Intensity (a.u.)')
            self.ax_spectrum.set_title('Experimental vs Calculated')
            self.ax_spectrum.grid(True, alpha=0.3)
            
            # Get tensor data
            if 'tensors' not in self.tensor_data_3d or 'wavenumbers' not in self.tensor_data_3d:
                return
                
            wavenumbers = self.tensor_data_3d['wavenumbers']
            tensors = self.tensor_data_3d['tensors']
            
            # Get polarization configuration
            incident_pol_config = getattr(self, 'incident_pol_var', tk.StringVar(value="X-polarized")).get()
            scattered_pol_config = getattr(self, 'scattered_pol_var', tk.StringVar(value="X-polarized")).get()
            
            # Convert polarization configurations to vectors
            pol_in = self.get_polarization_vector(incident_pol_config)
            pol_out = self.get_polarization_vector(scattered_pol_config, pol_in)
            
            # Calculate intensities for current orientation
            calculated_intensities = []
            
            print(f"🔄 Calculating intensities for orientation φ={self.phi_var.get():.1f}°, θ={self.theta_var.get():.1f}°, ψ={self.psi_var.get():.1f}°")
            
            for i, tensor in enumerate(tensors):
                # Apply current orientation to tensor
                rotated_tensor = np.dot(self.current_orientation_matrix, 
                                      np.dot(tensor, self.current_orientation_matrix.T))
                
                # Calculate Raman intensity: I ∝ |e_s · α · e_i|²
                intensity = np.abs(np.dot(pol_out, np.dot(rotated_tensor, pol_in)))**2
                calculated_intensities.append(intensity)
                
                # Debug output for first few peaks
                if i < 3:
                    original_intensity = np.abs(np.dot(pol_out, np.dot(tensor, pol_in)))**2
                    print(f"   Peak {wavenumbers[i]:.1f}: {original_intensity:.3f} → {intensity:.3f}")
            
            calculated_intensities = np.array(calculated_intensities)
            
            # Normalize
            if np.max(calculated_intensities) > 0:
                calculated_intensities = calculated_intensities / np.max(calculated_intensities)
            
            # Plot calculated spectrum as STEM PLOT ONLY (no connected lines)
            markerline, stemlines, baseline = self.ax_spectrum.stem(
                wavenumbers, calculated_intensities, 
                linefmt='r-', markerfmt='ro', basefmt=' ',
                label=f'Calculated (φ={self.phi_var.get():.1f}°, θ={self.theta_var.get():.1f}°, ψ={self.psi_var.get():.1f}°)'
            )
            
            # Make stem lines thicker and markers larger
            stemlines.set_linewidth(2)
            markerline.set_markersize(6)
            
            # Plot experimental spectrum if available
            exp_plotted = False
            
            # Try multiple sources for experimental data
            experimental_sources = [
                ('current_spectrum', 'Experimental'),
                ('original_spectrum', 'Original Experimental'),
                ('wavenumbers', 'Loaded Spectrum')  # Check for direct wavenumber/intensity arrays
            ]
            

            
            for source_attr, label in experimental_sources:
                if hasattr(self, source_attr) and getattr(self, source_attr) is not None:
                    exp_data = getattr(self, source_attr)
                    
                    if isinstance(exp_data, dict):
                        if 'wavenumbers' in exp_data and 'intensities' in exp_data:
                            exp_wavenumbers = exp_data['wavenumbers']
                            exp_intensities = exp_data['intensities']
                        else:
                            continue
                    elif hasattr(self, 'wavenumbers') and hasattr(self, 'intensities'):
                        # Direct arrays
                        exp_wavenumbers = self.wavenumbers
                        exp_intensities = self.intensities
                    else:
                        continue
                    
                    # Normalize experimental spectrum
                    if len(exp_intensities) > 0 and np.max(exp_intensities) > 0:
                        exp_intensities_norm = exp_intensities / np.max(exp_intensities)
                        self.ax_spectrum.plot(exp_wavenumbers, exp_intensities_norm, 'b-', linewidth=1.5, 
                                             alpha=0.8, label=label)
                        exp_plotted = True
                        break  # Use first available source
            
            # If no experimental data found, add a note
            if not exp_plotted:
                self.ax_spectrum.text(0.5, 0.95, 'No experimental data loaded\nLoad spectrum or use demo data', 
                                     transform=self.ax_spectrum.transAxes, ha='center', va='top',
                                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Add polarization info to plot
            pol_info = f'Pol: {incident_pol_config} → {scattered_pol_config}'
            self.ax_spectrum.text(0.02, 0.98, pol_info, transform=self.ax_spectrum.transAxes, 
                                 fontsize=9, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.ax_spectrum.legend()
            self.canvas_spectrum.draw()
            
        except Exception as e:
            print(f"Error calculating orientation spectrum: {e}")
    
    def get_polarization_vector(self, pol_config, reference_pol=None):
        """Convert polarization configuration string to vector."""
        try:
            if pol_config == "X-polarized":
                return np.array([1, 0, 0])
            elif pol_config == "Y-polarized":
                return np.array([0, 1, 0])
            elif pol_config == "Z-polarized":
                return np.array([0, 0, 1])
            elif pol_config == "Parallel" and reference_pol is not None:
                return reference_pol / np.linalg.norm(reference_pol)
            elif pol_config == "Perpendicular" and reference_pol is not None:
                # Create perpendicular vector in XY plane
                if abs(reference_pol[2]) < 0.9:  # Not along Z
                    perp = np.array([-reference_pol[1], reference_pol[0], 0])
                else:  # Along Z, use X direction
                    perp = np.array([1, 0, 0])
                return perp / np.linalg.norm(perp)
            elif pol_config == "Circular":
                # Circular polarization (simplified as 45° linear)
                return np.array([1, 1, 0]) / np.sqrt(2)
            else:
                # Default to X-polarized
                return np.array([1, 0, 0])
        except:
            return np.array([1, 0, 0])  # Fallback
    
    def start_crystal_animation(self):
        """Start crystal rotation animation."""
        if not hasattr(self, 'animation_running'):
            return
            
        self.animation_running = True
        self.animate_crystal()
    
    def stop_crystal_animation(self):
        """Stop crystal rotation animation."""
        self.animation_running = False
    
    def animate_crystal(self):
        """Animate crystal rotation."""
        if not self.animation_running:
            return
            
        try:
            # Get animation parameters
            axis = self.animation_axis_var.get()
            speed = self.animation_speed_var.get()
            
            # Update angles based on selected axis
            if axis == "φ":
                new_phi = (self.phi_var.get() + speed) % 360
                self.phi_var.set(new_phi)
            elif axis == "θ":
                new_theta = (self.theta_var.get() + speed) % 180
                self.theta_var.set(new_theta)
            elif axis == "ψ":
                new_psi = (self.psi_var.get() + speed) % 360
                self.psi_var.set(new_psi)
            elif axis == "All":
                new_phi = (self.phi_var.get() + speed) % 360
                new_theta = (self.theta_var.get() + speed/2) % 180
                new_psi = (self.psi_var.get() + speed/3) % 360
                self.phi_var.set(new_phi)
                self.theta_var.set(new_theta)
                self.psi_var.set(new_psi)
            
            # Update visualization
            self.update_current_orientation()
            self.update_3d_visualization()
            
            # Update spectrum if real-time calculation is enabled
            if hasattr(self, 'real_time_calc_var') and self.real_time_calc_var.get():
                self.calculate_orientation_spectrum()
            
            # Schedule next frame
            if self.animation_running:
                self.root.after(50, self.animate_crystal)  # 20 FPS
                
        except Exception as e:
            print(f"Error in animation: {e}")
            self.animation_running = False
    
    def update_3d_status(self):
        """Update the status text for 3D visualization."""
        if not hasattr(self, 'viz_status_text'):
            return
            
        try:
            self.viz_status_text.delete(1.0, tk.END)
            
            status_lines = []
            
            # Crystal shape status
            if self.crystal_shape_data is not None:
                crystal_type = self.crystal_shape_data.get('type', 'Unknown')
                status_lines.append(f"Crystal: {crystal_type.title()}")
            else:
                status_lines.append("Crystal: Not loaded")
            
            # Tensor data status
            if self.tensor_data_3d is not None:
                n_peaks = len(self.tensor_data_3d.get('wavenumbers', []))
                status_lines.append(f"Tensors: {n_peaks} peaks")
            else:
                status_lines.append("Tensors: Not loaded")
            
            # Crystal structure status
            structure_available = False
            if hasattr(self, 'structure_data') and self.structure_data:
                n_atoms = len(self.structure_data.get('atoms', []))
                crystal_system = self.structure_data.get('crystal_system', 'Unknown')
                status_lines.append(f"Structure: {n_atoms} atoms ({crystal_system})")
                structure_available = True
            elif hasattr(self, 'crystal_structure') and self.crystal_structure:
                n_atoms = len(self.crystal_structure.get('atoms', []))
                crystal_system = self.crystal_structure.get('crystal_system', 'Unknown')
                status_lines.append(f"Structure: {n_atoms} atoms ({crystal_system})")
                structure_available = True
            
            if not structure_available:
                status_lines.append("Structure: Not loaded")
            
            # Optimization status
            if self.optimized_orientation is not None:
                status_lines.append("Optimization: Available")
            else:
                status_lines.append("Optimization: Not available")
            
            self.viz_status_text.insert(tk.END, "\n".join(status_lines))
            
        except Exception as e:
            print(f"Error updating 3D status: {e}")
    
    def auto_import_all_data(self):
        """Automatically import all available data sources."""
        try:
            imported_count = 0
            
            # Try to import optimization results (check both stage_results and legacy optimization_results)
            has_optimization = False
            if hasattr(self, 'stage_results') and any(self.stage_results.values()):
                has_optimization = True
            elif hasattr(self, 'optimization_results') and self.optimization_results:
                has_optimization = True
            
            if has_optimization:
                self.import_optimization_orientation()
                imported_count += 1
            
            # Try to import structure data (check both crystal_structure and structure_data)
            has_structure = False
            if hasattr(self, 'crystal_structure') and self.crystal_structure:
                has_structure = True
            elif hasattr(self, 'structure_data') and self.structure_data:
                has_structure = True
            
            if has_structure:
                self.import_structure_data()
                imported_count += 1
            
            # Enhanced tensor data import with multiple fallback options
            tensor_imported = False
            if hasattr(self, 'raman_tensors') and bool(self.raman_tensors):
                try:
                    self.tensor_data_3d = self.raman_tensors.copy()
                    self.update_peak_selection_3d()
                    self.update_3d_visualization()
                    self.update_3d_status()
                    imported_count += 1
                    tensor_imported = True
                    print("✓ Imported existing tensor data")
                except Exception as e:
                    print(f"Auto-import tensor data failed: {e}")
            
            # If no tensor data, try to create from fitted peaks
            if not tensor_imported and hasattr(self, 'fitted_peaks') and self.fitted_peaks:
                success = self.create_tensors_from_fitted_peaks()
                if success:
                    imported_count += 1
                    tensor_imported = True
                    print("✓ Created tensors from fitted peaks")
            
            # If still no tensor data, try mineral database
            if not tensor_imported and hasattr(self, 'mineral_data') and self.mineral_data:
                success = self.create_tensors_from_mineral_data()
                if success:
                    imported_count += 1
                    tensor_imported = True
                    print("✓ Created tensors from mineral data")
            
            if imported_count > 0:
                status_msg = f"Successfully imported {imported_count} data source(s)!"
                if tensor_imported:
                    status_msg += "\n• Tensor data available for 3D visualization"
                messagebox.showinfo("Auto-Import", status_msg)
            else:
                messagebox.showwarning("Auto-Import", 
                    "No data sources available to import.\n\n" +
                    "To use 3D visualization:\n" +
                    "1. Load a spectrum in Spectrum Analysis tab\n" +
                    "2. Fit peaks in Peak Fitting tab\n" +
                    "3. Or use the Demo Data button")
                
        except Exception as e:
            messagebox.showerror("Error", f"Auto-import failed: {str(e)}")
    
    def create_tensors_from_fitted_peaks(self):
        """Create Raman tensor data from fitted peaks."""
        try:
            if not hasattr(self, 'fitted_peaks') or not self.fitted_peaks:
                print("No fitted peaks available")
                return False
            
            print(f"Creating tensors from {len(self.fitted_peaks)} fitted peaks...")
            
            tensor_data = {}
            wavenumbers = []
            tensors = []
            
            for i, peak in enumerate(self.fitted_peaks):
                try:
                    # Extract peak information - handle different peak object types
                    if hasattr(peak, 'center'):
                        wavenumber = float(peak.center)
                    elif hasattr(peak, 'position'):
                        wavenumber = float(peak.position)
                    elif isinstance(peak, dict) and 'center' in peak:
                        wavenumber = float(peak['center'])
                    elif isinstance(peak, dict) and 'position' in peak:
                        wavenumber = float(peak['position'])
                    else:
                        print(f"Could not extract wavenumber from peak {i}: {peak}")
                        continue
                    
                    # Extract intensity
                    if hasattr(peak, 'amplitude'):
                        intensity = float(peak.amplitude)
                    elif hasattr(peak, 'height'):
                        intensity = float(peak.height)
                    elif hasattr(peak, 'intensity'):
                        intensity = float(peak.intensity)
                    elif isinstance(peak, dict) and 'amplitude' in peak:
                        intensity = float(peak['amplitude'])
                    elif isinstance(peak, dict) and 'height' in peak:
                        intensity = float(peak['height'])
                    elif isinstance(peak, dict) and 'intensity' in peak:
                        intensity = float(peak['intensity'])
                    else:
                        intensity = 1.0  # Default intensity
                    
                    # Create symmetry-constrained tensor based on crystal system
                    tensor = self.create_symmetry_constrained_tensor(wavenumber, intensity)
                    
                    wavenumbers.append(wavenumber)
                    tensors.append(tensor)
                    
                    print(f"  Created tensor for peak at {wavenumber:.1f} cm⁻¹ (I={intensity:.2f})")
                    
                except Exception as e:
                    print(f"Error processing peak {i}: {e}")
                    continue
            
            if len(wavenumbers) == 0:
                print("No valid peaks found to create tensors")
                return False
            
            # Store tensor data
            self.tensor_data_3d = {
                'wavenumbers': np.array(wavenumbers),
                'tensors': np.array(tensors)
            }
            
            # Also populate raman_tensors for consistency
            self.raman_tensors = {}
            for i, (wn, tensor) in enumerate(zip(wavenumbers, tensors)):
                self.raman_tensors[f"{wn:.1f}"] = {
                    'tensor': tensor,
                    'wavenumber': wn,
                    'intensity': np.trace(tensor) / 3.0  # Average intensity
                }
            
            # Update interface
            self.update_peak_selection_3d()
            self.update_3d_visualization()
            self.update_3d_status()
            
            print(f"✅ Successfully created tensor data for {len(wavenumbers)} peaks")
            return True
            
        except Exception as e:
            print(f"Error creating tensors from fitted peaks: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_tensors_from_mineral_data(self):
        """Create Raman tensor data from mineral database."""
        try:
            if not hasattr(self, 'mineral_data') or not self.mineral_data:
                return False
            
            if 'peaks' not in self.mineral_data:
                return False
            
            peaks = self.mineral_data['peaks']
            if not peaks:
                return False
            
            print(f"Creating tensors from {len(peaks)} mineral database peaks...")
            
            wavenumbers = []
            tensors = []
            
            # Use top 10 peaks by intensity
            sorted_peaks = sorted(peaks, key=lambda x: x.get('I_tot', 0), reverse=True)[:10]
            
            for peak in sorted_peaks:
                wavenumber = peak.get('wavenumber', 0)
                intensity = peak.get('I_tot', 1.0)
                
                if wavenumber > 0:
                    # Create simple tensor
                    tensor = np.eye(3) * intensity / 100.0  # Scale intensity
                    
                    wavenumbers.append(wavenumber)
                    tensors.append(tensor)
            
            if len(wavenumbers) == 0:
                return False
            
            # Store tensor data
            self.tensor_data_3d = {
                'wavenumbers': np.array(wavenumbers),
                'tensors': np.array(tensors)
            }
            
            # Also populate raman_tensors
            self.raman_tensors = {}
            for wn, tensor in zip(wavenumbers, tensors):
                self.raman_tensors[f"{wn:.1f}"] = {
                    'tensor': tensor,
                    'wavenumber': wn,
                    'intensity': np.trace(tensor) / 3.0
                }
            
            # Update interface
            self.update_peak_selection_3d()
            self.update_3d_visualization()
            self.update_3d_status()
            
            print(f"✅ Successfully created tensor data from mineral database")
            return True
            
        except Exception as e:
            print(f"Error creating tensors from mineral data: {e}")
            return False
    
    def create_symmetry_constrained_tensor(self, wavenumber, intensity):
        """Create a Raman tensor that respects crystal symmetry constraints."""
        try:
            # Get crystal system from structure data or default
            crystal_system = "Cubic"  # Default
            point_group = "m-3m"     # Default cubic point group
            
            # Try to get crystal system from various sources
            if hasattr(self, 'crystal_structure') and self.crystal_structure:
                crystal_system = self.crystal_structure.get('crystal_system', 'Cubic')
                point_group = self.crystal_structure.get('point_group', 'm-3m')
            elif hasattr(self, 'structure_data') and self.structure_data:
                crystal_system = self.structure_data.get('crystal_system', 'Cubic')
                point_group = self.structure_data.get('point_group', 'm-3m')
            elif hasattr(self, 'mineral_data') and self.mineral_data:
                crystal_system = self.mineral_data.get('crystal_system', 'Cubic')
                point_group = self.mineral_data.get('point_group', 'm-3m')
            
            # Create base tensor template based on crystal system symmetry
            if crystal_system.lower() == "cubic":
                # Cubic: Only diagonal elements, all equal (isotropic)
                # Point groups: m-3m (Oh), -43m (Td), m-3 (Th), 23 (T)
                tensor_template = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ])
                
            elif crystal_system.lower() == "tetragonal":
                # Tetragonal: αxx = αyy ≠ αzz, off-diagonal = 0
                # Point groups: 4/mmm (D4h), 422 (D4), 4mm (C4v), -42m (D2d), 4/m (C4h), 4 (C4), -4 (S4)
                tensor_template = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.7]  # Different c-axis component
                ])
                
            elif crystal_system.lower() == "orthorhombic":
                # Orthorhombic: All diagonal different, off-diagonal = 0
                # Point groups: mmm (D2h), 222 (D2), mm2 (C2v)
                tensor_template = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.8, 0.0],
                    [0.0, 0.0, 0.6]
                ])
                
            elif crystal_system.lower() == "hexagonal":
                # Hexagonal: αxx = αyy ≠ αzz, some off-diagonal allowed
                # Point groups: 6/mmm (D6h), 622 (D6), 6mm (C6v), -6m2 (D3h), 6/m (C6h), 6 (C6), -6 (C3h)
                tensor_template = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.8]
                ])
                
            elif crystal_system.lower() == "trigonal":
                # Trigonal/Rhombohedral: Similar to hexagonal but with 3-fold symmetry
                # Point groups: -3m (D3d), 32 (D3), 3m (C3v), -3 (C3i), 3 (C3)
                tensor_template = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.75]
                ])
                
            elif crystal_system.lower() == "monoclinic":
                # Monoclinic: More complex, some off-diagonal elements allowed
                # Point groups: 2/m (C2h), 2 (C2), m (Cs)
                tensor_template = np.array([
                    [1.0, 0.0, 0.1],
                    [0.0, 0.8, 0.0],
                    [0.1, 0.0, 0.6]
                ])
                
            elif crystal_system.lower() == "triclinic":
                # Triclinic: No symmetry constraints, all elements allowed
                # Point groups: -1 (Ci), 1 (C1)
                tensor_template = np.array([
                    [1.0, 0.2, 0.1],
                    [0.2, 0.8, 0.15],
                    [0.1, 0.15, 0.6]
                ])
                
            else:
                # Default to cubic if unknown
                tensor_template = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ])
            
            # Scale by intensity
            tensor = tensor_template * intensity
            
            # Add small frequency dependence while preserving symmetry
            freq_factor = 1.0 + 0.05 * np.sin(wavenumber / 200.0)
            tensor *= freq_factor
            
            # Ensure tensor remains symmetric (Raman tensors should be symmetric)
            tensor = (tensor + tensor.T) / 2.0
            
            print(f"  Created {crystal_system.lower()} tensor for {wavenumber:.1f} cm⁻¹ (I={intensity:.2f})")
            
            return tensor
            
        except Exception as e:
            print(f"Error creating symmetry-constrained tensor: {e}")
            # Fallback to simple isotropic tensor
            return np.eye(3) * intensity
    
    def calculate_point_group_violation(self, tensor, crystal_system):
        """Calculate how much a tensor violates point group symmetry constraints."""
        try:
            crystal_system = crystal_system.lower()
            
            if crystal_system == "cubic":
                # Cubic: αxx = αyy = αzz, off-diagonal = 0
                # Check diagonal equality
                diag_violation = (np.abs(tensor[0,0] - tensor[1,1]) + 
                                np.abs(tensor[0,0] - tensor[2,2]) + 
                                np.abs(tensor[1,1] - tensor[2,2])) / (3 * np.abs(tensor[0,0]) + 1e-10)
                
                # Check off-diagonal zeros
                off_diag_violation = (np.abs(tensor[0,1]) + np.abs(tensor[0,2]) + 
                                    np.abs(tensor[1,2])) / (np.abs(tensor[0,0]) + 1e-10)
                
                return diag_violation + off_diag_violation
                
            elif crystal_system == "tetragonal":
                # Tetragonal: αxx = αyy ≠ αzz, off-diagonal = 0
                # Check xy equality
                xy_violation = np.abs(tensor[0,0] - tensor[1,1]) / (np.abs(tensor[0,0]) + 1e-10)
                
                # Check off-diagonal zeros
                off_diag_violation = (np.abs(tensor[0,1]) + np.abs(tensor[0,2]) + 
                                    np.abs(tensor[1,2])) / (np.abs(tensor[0,0]) + 1e-10)
                
                return xy_violation + off_diag_violation
                
            elif crystal_system == "orthorhombic":
                # Orthorhombic: off-diagonal = 0
                off_diag_violation = (np.abs(tensor[0,1]) + np.abs(tensor[0,2]) + 
                                    np.abs(tensor[1,2])) / (np.abs(tensor[0,0]) + 1e-10)
                
                return off_diag_violation
                
            elif crystal_system == "hexagonal" or crystal_system == "trigonal":
                # Hexagonal/Trigonal: αxx = αyy ≠ αzz
                xy_violation = np.abs(tensor[0,0] - tensor[1,1]) / (np.abs(tensor[0,0]) + 1e-10)
                
                # Some off-diagonal elements may be allowed, but check main ones
                main_off_diag_violation = (np.abs(tensor[0,2]) + np.abs(tensor[1,2])) / (np.abs(tensor[0,0]) + 1e-10)
                
                return xy_violation + main_off_diag_violation
                
            elif crystal_system == "monoclinic":
                # Monoclinic: Some off-diagonal elements should be zero
                # Check specific elements based on monoclinic symmetry
                restricted_violation = np.abs(tensor[0,1]) / (np.abs(tensor[0,0]) + 1e-10)
                
                return restricted_violation
                
            elif crystal_system == "triclinic":
                # Triclinic: No symmetry constraints
                return 0.0
                
            else:
                # Unknown crystal system, assume no constraints
                return 0.0
                
        except Exception as e:
            print(f"Error calculating point group violation: {e}")
            return 0.0
    
    def should_tensor_element_be_zero(self, i, j, crystal_system):
        """Determine if a tensor element should be zero based on crystal symmetry."""
        crystal_system = crystal_system.lower()
        
        if crystal_system == "cubic":
            # Cubic: Only diagonal elements allowed, all equal
            return i != j
            
        elif crystal_system == "tetragonal":
            # Tetragonal: Only diagonal elements allowed
            return i != j
            
        elif crystal_system == "orthorhombic":
            # Orthorhombic: Only diagonal elements allowed
            return i != j
            
        elif crystal_system == "hexagonal":
            # Hexagonal: Diagonal + some specific off-diagonal elements
            # For simplicity, restrict to diagonal + (0,1) element
            if i == j:
                return False  # Diagonal allowed
            elif (i == 0 and j == 1) or (i == 1 and j == 0):
                return False  # xy coupling allowed in some point groups
            else:
                return True   # Other off-diagonal forbidden
                
        elif crystal_system == "trigonal":
            # Similar to hexagonal
            if i == j:
                return False  # Diagonal allowed
            elif (i == 0 and j == 1) or (i == 1 and j == 0):
                return False  # xy coupling allowed
            else:
                return True   # Other off-diagonal forbidden
                
        elif crystal_system == "monoclinic":
            # Monoclinic: Some off-diagonal elements forbidden
            # Depends on the specific monoclinic setting, but commonly:
            if (i == 0 and j == 1) or (i == 1 and j == 0):
                return True   # xy coupling forbidden in many monoclinic point groups
            else:
                return False  # Other elements allowed
                
        elif crystal_system == "triclinic":
            # Triclinic: No restrictions
            return False
            
        else:
            # Unknown system: assume no restrictions
            return False
    
    def quick_setup_3d(self):
        """Quick setup: Load demo data and enable all visualizations."""
        try:
            # Load demo data first
            self.load_demo_data_3d()
            
            # Enable all visualization options
            if hasattr(self, 'show_crystal_var'):
                self.show_crystal_var.set(True)
            if hasattr(self, 'show_tensor_ellipsoid_var'):
                self.show_tensor_ellipsoid_var.set(True)
            if hasattr(self, 'show_laser_geometry_var'):
                self.show_laser_geometry_var.set(True)
            if hasattr(self, 'show_axes_var'):
                self.show_axes_var.set(True)
            if hasattr(self, 'show_crystal_structure_var'):
                self.show_crystal_structure_var.set(False)  # Keep structure off to avoid clutter
            
            # Enable real-time calculation
            if hasattr(self, 'real_time_calc_var'):
                self.real_time_calc_var.set(True)
            
            # Update everything
            self.update_peak_selection_3d()
            self.update_3d_visualization()
            self.calculate_orientation_spectrum()
            self.update_3d_status()
            
            messagebox.showinfo("Quick Setup Complete", 
                              "✅ Demo data loaded and all visualizations enabled!\n\n" +
                              "You should now see:\n" +
                              "• 🔴 Red tensor ellipsoid\n" +
                              "• 🔷 Crystal shape\n" +
                              "• 📐 Coordinate axes\n" +
                              "• 🔶 Laser geometry\n" +
                              "• 📊 Calculated spectrum\n\n" +
                              "Try rotating the crystal with the sliders!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Quick setup failed: {str(e)}")
    
    def load_demo_data_3d(self):
        """Load demonstration data for testing the 3D visualization."""
        try:
            # Set demo crystal structure (tetragonal for interesting anisotropy)
            self.crystal_structure = {
                'crystal_system': 'Tetragonal',
                'point_group': '4/mmm',
                'name': 'Demo Tetragonal Crystal',
                'space_group': 'P4/mmm'
            }
            
            # Create demo Raman tensor data
            wavenumbers = np.array([200, 400, 600, 800, 1000])
            
            # Generate symmetry-constrained tensors
            tensors = []
            for i, freq in enumerate(wavenumbers):
                # Use varying intensities for different peaks
                intensity = 1.0 + 0.5 * i  # Increasing intensity
                tensor = self.create_symmetry_constrained_tensor(freq, intensity)
                tensors.append(tensor)
            
            self.tensor_data_3d = {
                'wavenumbers': wavenumbers,
                'tensors': np.array(tensors)
            }
            
            # Also populate raman_tensors for consistency
            self.raman_tensors = self.tensor_data_3d.copy()
            
            # Create demo spectrum
            test_wavenumbers = np.linspace(100, 1200, 500)
            test_intensities = np.zeros_like(test_wavenumbers)
            
            # Add Gaussian peaks
            for i, freq in enumerate(wavenumbers):
                peak = np.exp(-((test_wavenumbers - freq) / 25)**2)
                test_intensities += peak * (1.0 + 0.3 * i)  # Varying intensities
            
            # Add noise
            test_intensities += np.random.normal(0, 0.03, len(test_intensities))
            test_intensities = np.maximum(test_intensities, 0)  # Ensure non-negative
            
            # Normalize to reasonable range
            test_intensities = test_intensities / np.max(test_intensities)
            
            self.current_spectrum = {
                'wavenumbers': test_wavenumbers,
                'intensities': test_intensities,
                'name': 'Demo Spectrum'
            }
            
            # Also set as original spectrum for consistency
            self.original_spectrum = self.current_spectrum.copy()
            
            # Set direct arrays for compatibility
            self.wavenumbers = test_wavenumbers
            self.intensities = test_intensities
            
            # Create demo optimization result
            demo_angles = [30, 45, 60]  # φ, θ, ψ in degrees
            phi, theta, psi = np.radians(demo_angles)
            demo_orientation = self.euler_to_rotation_matrix(phi, theta, psi)
            
            self.optimization_results = [{
                'orientation': demo_orientation,
                'score': 0.92,
                'angles': demo_angles
            }]
            self.optimized_orientation = demo_orientation
            
            # Create demo structure data (simple cubic structure)
            self.structure_data = {
                'lattice_vectors': np.array([
                    [3.0, 0.0, 0.0],  # a vector
                    [0.0, 3.0, 0.0],  # b vector  
                    [0.0, 0.0, 3.0]   # c vector
                ]),
                'atoms': [
                    {'element': 'Si', 'frac_coords': [0.0, 0.0, 0.0]},
                    {'element': 'O', 'frac_coords': [0.5, 0.0, 0.0]},
                    {'element': 'O', 'frac_coords': [0.0, 0.5, 0.0]},
                    {'element': 'O', 'frac_coords': [0.0, 0.0, 0.5]},
                    {'element': 'Si', 'frac_coords': [0.5, 0.5, 0.5]},
                    {'element': 'O', 'frac_coords': [1.0, 0.5, 0.5]},
                    {'element': 'O', 'frac_coords': [0.5, 1.0, 0.5]},
                    {'element': 'O', 'frac_coords': [0.5, 0.5, 1.0]}
                ],
                'space_group': 'Pm-3m',
                'crystal_system': 'Cubic'
            }
            
            # Generate crystal shape
            self.generate_crystal_shape()
            
            # Update interface
            self.update_peak_selection_3d()
            self.update_3d_visualization()
            self.update_3d_status()
            
            messagebox.showinfo("Demo Data", "Demo data loaded successfully!\n\n" +
                              f"• {len(wavenumbers)} Raman tensor peaks\n" +
                              f"• {len(test_wavenumbers)} spectrum points\n" +
                              "• Optimized orientation available\n" +
                              "• Crystal shape generated\n" +
                              f"• Crystal structure: {len(self.structure_data['atoms'])} atoms\n" +
                              "• Unit cell: 3×3×3 Å cubic")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load demo data: {str(e)}")
    
    def optimize_for_dense_view(self):
        """Optimize settings for viewing dense crystal structures."""
        self.unit_cell_range_var.set(3.0)  # Show more unit cells
        self.atom_size_var.set(0.2)        # Smaller atoms for less clutter
        self.structure_compression_var.set(0.6)  # Compress structure for better visibility
        self.update_3d_visualization()
        
    def reset_3d_view_settings(self):
        """Reset 3D view settings to defaults."""
        self.unit_cell_range_var.set(2.0)
        self.atom_size_var.set(0.3)
        self.structure_compression_var.set(1.0)
        self.update_3d_visualization()

    def reset_3d_view(self):
        """Reset the 3D view to default perspective."""
        if hasattr(self, 'ax_3d'):
            self.ax_3d.view_init(elev=20, azim=45)
            self.canvas_3d.draw()
    
    def set_top_view(self):
        """Set 3D view to top-down perspective."""
        if hasattr(self, 'ax_3d'):
            self.ax_3d.view_init(elev=90, azim=0)
            self.canvas_3d.draw()
    
    def set_side_view(self):
        """Set 3D view to side perspective."""
        if hasattr(self, 'ax_3d'):
            self.ax_3d.view_init(elev=0, azim=0)
            self.canvas_3d.draw()
    
    def set_isometric_view(self):
        """Set 3D view to isometric perspective."""
        if hasattr(self, 'ax_3d'):
            self.ax_3d.view_init(elev=35.264, azim=45)  # True isometric angles
            self.canvas_3d.draw()
    
    def save_3d_view(self):
        """Save the current 3D view as an image."""
        try:
            if not hasattr(self, 'fig_3d'):
                messagebox.showwarning("No Plot", "No 3D visualization to save.")
                return
            
            file_path = filedialog.asksaveasfilename(
                title="Save 3D View",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.fig_3d.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"3D view saved to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save 3D view: {str(e)}")
    
    def export_orientation_spectrum(self):
        """Export the current orientation-dependent spectrum."""
        try:
            if not hasattr(self, 'ax_spectrum') or self.tensor_data_3d is None:
                messagebox.showwarning("No Data", "No spectrum data to export.")
                return
            
            file_path = filedialog.asksaveasfilename(
                title="Export Orientation Spectrum",
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Get current spectrum data
                wavenumbers = self.tensor_data_3d['wavenumbers']
                tensors = self.tensor_data_3d['tensors']
                
                # Calculate current intensities
                calculated_intensities = []
                pol_in = np.array([1, 0, 0])
                pol_out = np.array([1, 0, 0])
                
                for tensor in tensors:
                    rotated_tensor = np.dot(self.current_orientation_matrix, 
                                          np.dot(tensor, self.current_orientation_matrix.T))
                    intensity = np.abs(np.dot(pol_out, np.dot(rotated_tensor, pol_in)))**2
                    calculated_intensities.append(intensity)
                
                # Normalize
                calculated_intensities = np.array(calculated_intensities)
                if np.max(calculated_intensities) > 0:
                    calculated_intensities = calculated_intensities / np.max(calculated_intensities)
                
                # Create header
                header = f"# Orientation-dependent Raman spectrum\n"
                header += f"# Crystal orientation: φ={self.phi_var.get():.2f}°, θ={self.theta_var.get():.2f}°, ψ={self.psi_var.get():.2f}°\n"
                header += f"# Wavenumber(cm-1)\tIntensity(normalized)\n"
                
                # Save data
                with open(file_path, 'w') as f:
                    f.write(header)
                    for wn, intensity in zip(wavenumbers, calculated_intensities):
                        f.write(f"{wn:.2f}\t{intensity:.6f}\n")
                
                messagebox.showinfo("Success", f"Spectrum exported to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export spectrum: {str(e)}")
    
    def export_orientation_data(self):
        """Export current orientation data and parameters."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Orientation Data",
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Collect orientation data
                orientation_data = {
                    'euler_angles': {
                        'phi_degrees': self.phi_var.get(),
                        'theta_degrees': self.theta_var.get(),
                        'psi_degrees': self.psi_var.get()
                    },
                    'rotation_matrix': self.current_orientation_matrix.tolist(),
                    'crystal_system': getattr(self, 'crystal_system_var', tk.StringVar()).get(),
                    'timestamp': str(datetime.now())
                }
                
                if self.optimized_orientation is not None:
                    opt_phi, opt_theta, opt_psi = self.rotation_matrix_to_euler(self.optimized_orientation)
                    orientation_data['optimized_orientation'] = {
                        'phi_degrees': np.degrees(opt_phi),
                        'theta_degrees': np.degrees(opt_theta),
                        'psi_degrees': np.degrees(opt_psi),
                        'rotation_matrix': self.optimized_orientation.tolist()
                    }
                
                if file_path.endswith('.json'):
                    import json
                    with open(file_path, 'w') as f:
                        json.dump(orientation_data, f, indent=2)
                else:
                    # Text format
                    with open(file_path, 'w') as f:
                        f.write("Crystal Orientation Data\n")
                        f.write("=" * 30 + "\n\n")
                        f.write(f"Current Orientation:\n")
                        f.write(f"  φ (phi): {orientation_data['euler_angles']['phi_degrees']:.2f}°\n")
                        f.write(f"  θ (theta): {orientation_data['euler_angles']['theta_degrees']:.2f}°\n")
                        f.write(f"  ψ (psi): {orientation_data['euler_angles']['psi_degrees']:.2f}°\n\n")
                        
                        f.write("Rotation Matrix:\n")
                        for row in self.current_orientation_matrix:
                            f.write(f"  [{row[0]:8.5f} {row[1]:8.5f} {row[2]:8.5f}]\n")
                        
                        if 'optimized_orientation' in orientation_data:
                            f.write(f"\nOptimized Orientation:\n")
                            opt_data = orientation_data['optimized_orientation']
                            f.write(f"  φ (phi): {opt_data['phi_degrees']:.2f}°\n")
                            f.write(f"  θ (theta): {opt_data['theta_degrees']:.2f}°\n")
                            f.write(f"  ψ (psi): {opt_data['psi_degrees']:.2f}°\n")
                
                messagebox.showinfo("Success", f"Orientation data exported to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export orientation data: {str(e)}")
    
    def generate_3d_report(self):
        """Generate a comprehensive report of the 3D visualization session."""
        try:
            report_lines = []
            report_lines.append("3D Crystal Orientation Simulator - Session Report")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Current orientation
            report_lines.append("CURRENT ORIENTATION")
            report_lines.append("-" * 20)
            report_lines.append(f"Euler Angles:")
            report_lines.append(f"  φ (phi): {self.phi_var.get():.2f}°")
            report_lines.append(f"  θ (theta): {self.theta_var.get():.2f}°")
            report_lines.append(f"  ψ (psi): {self.psi_var.get():.2f}°")
            report_lines.append("")
            
            # Data status
            report_lines.append("DATA STATUS")
            report_lines.append("-" * 12)
            
            if self.crystal_shape_data:
                crystal_type = self.crystal_shape_data.get('type', 'Unknown')
                report_lines.append(f"Crystal Shape: {crystal_type.title()}")
            else:
                report_lines.append("Crystal Shape: Not loaded")
            
            if self.tensor_data_3d:
                n_peaks = len(self.tensor_data_3d.get('wavenumbers', []))
                report_lines.append(f"Raman Tensors: {n_peaks} peaks available")
            else:
                report_lines.append("Raman Tensors: Not loaded")
            
            if hasattr(self, 'current_spectrum') and self.current_spectrum:
                n_points = len(self.current_spectrum['wavenumbers'])
                report_lines.append(f"Experimental Spectrum: {n_points} data points")
            else:
                report_lines.append("Experimental Spectrum: Not loaded")
            
            if self.optimized_orientation is not None:
                report_lines.append("Optimization Results: Available")
            else:
                report_lines.append("Optimization Results: Not available")
            
            report_lines.append("")
            
            # Display settings
            report_lines.append("DISPLAY SETTINGS")
            report_lines.append("-" * 16)
            if hasattr(self, 'show_crystal_var'):
                report_lines.append(f"Crystal Shape: {'Visible' if self.show_crystal_var.get() else 'Hidden'}")
            if hasattr(self, 'show_tensor_ellipsoid_var'):
                report_lines.append(f"Tensor Ellipsoid: {'Visible' if self.show_tensor_ellipsoid_var.get() else 'Hidden'}")
            if hasattr(self, 'show_laser_geometry_var'):
                report_lines.append(f"Laser Geometry: {'Visible' if self.show_laser_geometry_var.get() else 'Hidden'}")
            if hasattr(self, 'show_axes_var'):
                report_lines.append(f"Coordinate Axes: {'Visible' if self.show_axes_var.get() else 'Hidden'}")
            
            report_lines.append("")
            
            # Analysis settings
            report_lines.append("ANALYSIS SETTINGS")
            report_lines.append("-" * 17)
            if hasattr(self, 'real_time_calc_var'):
                report_lines.append(f"Real-time Calculation: {'Enabled' if self.real_time_calc_var.get() else 'Disabled'}")
            if hasattr(self, 'viz_peak_var'):
                report_lines.append(f"Selected Peak: {self.viz_peak_var.get()}")
            if hasattr(self, 'incident_pol_var'):
                report_lines.append(f"Incident Polarization: {self.incident_pol_var.get()}")
            if hasattr(self, 'scattered_pol_var'):
                report_lines.append(f"Scattered Polarization: {self.scattered_pol_var.get()}")
            
            # Show report in dialog
            self.show_report_dialog("3D Visualization Report", "\n".join(report_lines))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def show_report_dialog(self, title, content):
        """Show a report in a scrollable dialog window."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("600x500")
        
        # Create scrollable text widget
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert content
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
        
        # Add export button
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        def export_report():
            file_path = filedialog.asksaveasfilename(
                title="Export Report",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Report exported to {file_path}")
        
        ttk.Button(button_frame, text="Export Report", command=export_report).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)
    
    def setup_raman_tensors_tab(self, side_panel, content_area):
        """Setup the Tensor Analysis & Visualization tab."""
        # Side panel title
        ttk.Label(side_panel, text="Tensor Analysis & Visualization", font=('Arial', 12, 'bold')).pack(pady=(10, 20))
        
        # Data import section
        import_frame = ttk.LabelFrame(side_panel, text="Data Import", padding=10)
        import_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(import_frame, text="Import from Database/Polarization", command=self.extract_tensors_from_data).pack(fill=tk.X, pady=2)
        self.structure_import_button = ttk.Button(import_frame, text="Import from Structure Tab", command=self.calculate_tensors_from_structure)
        self.structure_import_button.pack(fill=tk.X, pady=2)
        ttk.Button(import_frame, text="Load External Tensor Data", command=self.import_tensor_data).pack(fill=tk.X, pady=2)
        
        # Tensor analysis section
        analysis_frame = ttk.LabelFrame(side_panel, text="Tensor Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(analysis_frame, text="Calculate Eigenvalues/Eigenvectors", command=self.calculate_principal_components).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Symmetry Analysis", command=self.analyze_tensor_symmetry).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Orientation Dependence", command=self.calculate_orientation_dependence).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Peak-by-Peak Analysis", command=self.analyze_peak_tensors).pack(fill=tk.X, pady=2)
        
        # Visualization section
        viz_frame = ttk.LabelFrame(side_panel, text="3D Visualization", padding=10)
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(viz_frame, text="Display Mode:").pack(anchor=tk.W)
        self.tensor_display_var = tk.StringVar(value="Tensor Matrix")
        tensor_combo = ttk.Combobox(viz_frame, textvariable=self.tensor_display_var,
                                   values=["Tensor Matrix", "3D Tensor Ellipsoid", "Principal Axes", 
                                          "Symmetry Visualization", "Orientation Map", "Peak Comparison"], state="readonly")
        tensor_combo.pack(fill=tk.X, pady=2)
        tensor_combo.bind('<<ComboboxSelected>>', self.update_tensor_plot)
        
        ttk.Label(viz_frame, text="Selected Peak (cm⁻¹):").pack(anchor=tk.W, pady=(5, 0))
        self.tensor_peak_var = tk.StringVar(value="All Peaks")
        self.tensor_peak_combo = ttk.Combobox(viz_frame, textvariable=self.tensor_peak_var, state="readonly")
        self.tensor_peak_combo.pack(fill=tk.X, pady=2)
        self.tensor_peak_combo.bind('<<ComboboxSelected>>', self.update_tensor_plot)
        
        ttk.Button(viz_frame, text="Update Visualization", command=self.update_tensor_plot).pack(fill=tk.X, pady=2)
        
        # Export section
        export_frame = ttk.LabelFrame(side_panel, text="Export & Integration", padding=10)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        

        ttk.Button(export_frame, text="Export Tensor Results", command=self.export_tensor_results).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Generate Report", command=self.generate_tensor_report).pack(fill=tk.X, pady=2)
        
        # Status section
        status_frame = ttk.LabelFrame(side_panel, text="Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.tensor_status_text = tk.Text(status_frame, height=4, width=30, font=('Arial', 8))
        self.tensor_status_text.pack(fill=tk.X, pady=2)
        
        # Add refresh button for status
        ttk.Button(status_frame, text="Refresh Status", command=self.update_tensor_status).pack(fill=tk.X, pady=2)
        
        self.update_tensor_status()
        
        # Content area
        self.setup_matplotlib_plot(content_area, "Tensor Analysis & Visualization")
    
    def setup_orientation_optimization_tab(self, side_panel, content_area):
        """Setup the Orientation Optimization tab."""
        # Side panel title
        ttk.Label(side_panel, text="Orientation Optimization", font=('Arial', 12, 'bold')).pack(pady=(10, 15))
        
        # === COMPACT SETUP SECTION ===
        setup_frame = ttk.LabelFrame(side_panel, text="🔬 Setup & Symmetry", padding=8)
        setup_frame.pack(fill=tk.X, padx=10, pady=3)
        
        # Target property (compact)
        target_frame = ttk.Frame(setup_frame)
        target_frame.pack(fill=tk.X, pady=1)
        ttk.Label(target_frame, text="Target:", width=12).pack(side=tk.LEFT)
        self.target_property_var = tk.StringVar(value="Maximum Intensity")
        target_combo = ttk.Combobox(target_frame, textvariable=self.target_property_var,
                                   values=["Maximum Intensity", "Maximum Contrast", "Specific Peak Enhancement", 
                                          "Experimental Match", "Depolarization Ratio"], state="readonly", width=18)
        target_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Crystal system and point group (side by side)
        crystal_frame = ttk.Frame(setup_frame)
        crystal_frame.pack(fill=tk.X, pady=1)
        ttk.Label(crystal_frame, text="Crystal:", width=12).pack(side=tk.LEFT)
        self.opt_crystal_system_var = tk.StringVar()
        if hasattr(self, 'crystal_system_var'):
            self.opt_crystal_system_var.set(self.crystal_system_var.get())
        opt_crystal_combo = ttk.Combobox(crystal_frame, textvariable=self.opt_crystal_system_var,
                                        values=["Cubic", "Tetragonal", "Orthorhombic", "Hexagonal", 
                                               "Trigonal", "Monoclinic", "Triclinic"], state="readonly", width=18)
        opt_crystal_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Point group display (compact)
        pg_frame = ttk.Frame(setup_frame)
        pg_frame.pack(fill=tk.X, pady=1)
        ttk.Label(pg_frame, text="Point Group:", width=12).pack(side=tk.LEFT)
        self.opt_point_group_label = ttk.Label(pg_frame, text="Unknown", 
                                              relief="solid", borderwidth=1, background="white", width=18)
        self.opt_point_group_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Update point group when crystal system changes
        def update_point_group_display(*args):
            self.update_point_group_display()
        self.opt_crystal_system_var.trace('w', update_point_group_display)
        
        # === COMPACT OPTIMIZATION STAGES ===
        opt_frame = ttk.LabelFrame(side_panel, text="🎯 Optimization Stages", padding=8)
        opt_frame.pack(fill=tk.X, padx=10, pady=3)
        
        # Stage 1 (compact horizontal layout)
        stage1_frame = ttk.Frame(opt_frame)
        stage1_frame.pack(fill=tk.X, pady=2)
        ttk.Label(stage1_frame, text="1. Coarse:", width=10, font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        self.stage1_resolution_var = tk.DoubleVar(value=3.0)
        stage1_scale = ttk.Scale(stage1_frame, from_=1.0, to=15.0, variable=self.stage1_resolution_var,
                                orient=tk.HORIZONTAL, length=80)
        stage1_scale.pack(side=tk.LEFT, padx=2)
        
        self.stage1_res_label = ttk.Label(stage1_frame, text="3.0°", width=6)
        self.stage1_res_label.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(stage1_frame, text="Run", command=self.run_stage1_optimization, width=6).pack(side=tk.RIGHT)
        
        def update_stage1_label(*args):
            self.stage1_res_label.config(text=f"{self.stage1_resolution_var.get():.1f}°")
        self.stage1_resolution_var.trace('w', update_stage1_label)
        
        # Stage 2 (compact horizontal layout)
        stage2_frame = ttk.Frame(opt_frame)
        stage2_frame.pack(fill=tk.X, pady=2)
        ttk.Label(stage2_frame, text="2. Fine:", width=10, font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        self.stage2_candidates_var = tk.IntVar(value=15)
        stage2_candidates_scale = ttk.Scale(stage2_frame, from_=5, to=50, variable=self.stage2_candidates_var,
                                           orient=tk.HORIZONTAL, length=80)
        stage2_candidates_scale.pack(side=tk.LEFT, padx=2)
        
        self.stage2_cand_label = ttk.Label(stage2_frame, text="15", width=6)
        self.stage2_cand_label.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(stage2_frame, text="Run", command=self.run_stage2_optimization, width=6).pack(side=tk.RIGHT)
        
        def update_stage2_label(*args):
            self.stage2_cand_label.config(text=f"{self.stage2_candidates_var.get()}")
        self.stage2_candidates_var.trace('w', update_stage2_label)
        
        # Stage 3 (compact horizontal layout)
        stage3_frame = ttk.Frame(opt_frame)
        stage3_frame.pack(fill=tk.X, pady=2)
        ttk.Label(stage3_frame, text="3. Final:", width=10, font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        self.opt_method_var = tk.StringVar(value="Nelder-Mead")
        method_combo = ttk.Combobox(stage3_frame, textvariable=self.opt_method_var,
                                   values=["Nelder-Mead", "Powell", "BFGS", "L-BFGS-B"], state="readonly", width=12)
        method_combo.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        ttk.Button(stage3_frame, text="Run", command=self.run_stage3_optimization, width=6).pack(side=tk.RIGHT)
        
        # === COMPACT STATUS SECTION ===
        status_frame = ttk.LabelFrame(side_panel, text="📊 Status", padding=8)
        status_frame.pack(fill=tk.X, padx=10, pady=3)
        
        self.opt_status_text = tk.Text(status_frame, height=6, width=30, font=('Arial', 8))
        self.opt_status_text.pack(fill=tk.X, pady=2)
        self.update_optimization_status()
        
        # Compact button row
        status_btn_frame = ttk.Frame(status_frame)
        status_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(status_btn_frame, text="🔄 Refresh", command=self.update_optimization_status, width=12).pack(side=tk.LEFT)
        ttk.Button(status_btn_frame, text="📋 Results", command=self.show_optimization_results, width=12).pack(side=tk.RIGHT)
        
        # === COMPACT RESULTS SECTION ===
        results_frame = ttk.LabelFrame(side_panel, text="💾 Export", padding=8)
        results_frame.pack(fill=tk.X, padx=10, pady=3)
        
        # Two-column button layout
        results_row1 = ttk.Frame(results_frame)
        results_row1.pack(fill=tk.X, pady=1)
        ttk.Button(results_row1, text="📤 Export Orientations", command=self.export_orientations).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        ttk.Button(results_row1, text="🔬 Apply to Experiment", command=self.apply_optimal_orientation).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2,0))
        
        # Content area
        self.setup_matplotlib_plot(content_area, "Orientation Optimization")
    
    def setup_stress_strain_tab(self, side_panel, content_area):
        """Setup the Stress/Strain Analysis tab."""
        # Side panel title
        ttk.Label(side_panel, text="Stress/Strain Analysis", font=('Arial', 12, 'bold')).pack(pady=(10, 20))
        
        # Data input section
        data_frame = ttk.LabelFrame(side_panel, text="Stress/Strain Data", padding=10)
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(data_frame, text="Load Stress Data", command=self.load_stress_data).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Load Strain Data", command=self.load_strain_data).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Generate Test Data", command=self.generate_stress_test_data).pack(fill=tk.X, pady=2)
        
        # Coefficients section
        coeff_frame = ttk.LabelFrame(side_panel, text="Stress-Optical Coefficients", padding=10)
        coeff_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(coeff_frame, text="Import Coefficients", command=self.import_stress_coefficients).pack(fill=tk.X, pady=2)
        ttk.Button(coeff_frame, text="Calculate from Data", command=self.calculate_stress_coefficients).pack(fill=tk.X, pady=2)
        ttk.Button(coeff_frame, text="Use Literature Values", command=self.use_literature_coefficients).pack(fill=tk.X, pady=2)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(side_panel, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(analysis_frame, text="Frequency Shift Analysis", command=self.analyze_frequency_shifts).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Stress Tensor Refinement", command=self.refine_stress_tensor).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Strain Mapping", command=self.create_strain_map).pack(fill=tk.X, pady=2)
        
        # Simulation section
        sim_frame = ttk.LabelFrame(side_panel, text="Simulation", padding=10)
        sim_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(sim_frame, text="Stress Type:").pack(anchor=tk.W)
        self.stress_type_var = tk.StringVar(value="Uniaxial")
        stress_combo = ttk.Combobox(sim_frame, textvariable=self.stress_type_var,
                                   values=["Uniaxial", "Biaxial", "Hydrostatic", "Shear", "Custom"], state="readonly")
        stress_combo.pack(fill=tk.X, pady=2)
        
        ttk.Button(sim_frame, text="Simulate Spectrum", command=self.simulate_stressed_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(sim_frame, text="Fit to Experimental", command=self.fit_stress_to_experiment).pack(fill=tk.X, pady=2)
        
        # Results section
        results_frame = ttk.LabelFrame(side_panel, text="Results", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(results_frame, text="Show Stress Analysis", command=self.show_stress_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(results_frame, text="Export Results", command=self.export_stress_results).pack(fill=tk.X, pady=2)
        
        # Content area
        self.setup_matplotlib_plot(content_area, "Stress/Strain Analysis")
    
    # Orientation Optimization Methods
    def run_stage1_optimization(self):
        """Run Stage 1: Coarse grid search over orientations."""
        # Check for required data
        if not self.fitted_peaks and not self.polarization_data:
            messagebox.showwarning("No Data", "Please fit peaks or load polarization data first.")
            return
        
        # Auto-populate crystal system from Crystal Structure tab if available
        if self.crystal_structure:
            crystal_system = self.get_crystal_system_from_structure()
            if crystal_system != "Unknown":
                self.opt_crystal_system_var.set(crystal_system)
                print(f"Auto-detected crystal system: {crystal_system}")
            else:
                print("Crystal structure loaded but system could not be determined")
        elif self.selected_reference_mineral:
            inferred_system = self.infer_crystal_system(self.selected_reference_mineral)
            if inferred_system != "Unknown":
                self.opt_crystal_system_var.set(inferred_system)
                print(f"Inferred crystal system from mineral: {inferred_system}")
        
        try:
            # Get optimization parameters
            resolution = self.stage1_resolution_var.get()
            target_property = self.target_property_var.get()
            crystal_system = self.opt_crystal_system_var.get()
            
            # Create progress dialog with optimization options
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Stage 1 Optimization")
            progress_window.geometry("450x250")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            ttk.Label(progress_window, text="Running coarse orientation search...", font=('Arial', 12)).pack(pady=10)
            
            # Optimization method selection
            method_frame = ttk.LabelFrame(progress_window, text="Optimization Method", padding=10)
            method_frame.pack(pady=10, padx=20, fill=tk.X)
            
            use_fast_var = tk.BooleanVar(value=True)
            fast_check = ttk.Checkbutton(method_frame, text="Use Fast Vectorized Method (Recommended)", 
                                       variable=use_fast_var)
            fast_check.pack(anchor=tk.W)
            
            use_parallel_var = tk.BooleanVar(value=False)
            parallel_check = ttk.Checkbutton(method_frame, text="Use Parallel Processing (Experimental)", 
                                           variable=use_parallel_var)
            parallel_check.pack(anchor=tk.W)
            
            # Add countdown timer for user configuration
            countdown_label = ttk.Label(progress_window, text="Starting in 3 seconds... (adjust settings above)", 
                                       font=('Arial', 10), foreground='blue')
            countdown_label.pack(pady=5)
            
            progress_bar = ttk.Progressbar(progress_window, mode='determinate')
            progress_bar.pack(pady=10, padx=20, fill=tk.X)
            
            progress_window.update()
            
            # 3-second countdown to allow user to adjust settings
            import time
            for countdown in range(3, 0, -1):
                countdown_label.config(text=f"Starting in {countdown} seconds... (adjust settings above)")
                progress_window.update()
                time.sleep(1)
            
            countdown_label.config(text="Generating orientation grid...")
            progress_window.update()
            
            # Generate orientation grid based on crystal system
            orientations = self.generate_orientation_grid(resolution, crystal_system)
            total_orientations = len(orientations)
            
            countdown_label.config(text=f"Evaluating {total_orientations} orientations...")
            progress_window.update()
            
            print(f"Evaluating {total_orientations} orientations...")
            
            # Choose optimization method
            start_time = time.time()
            
            if use_fast_var.get():
                results = self.run_fast_stage1_optimization(
                    orientations, target_property, progress_bar, progress_window,
                    use_parallel=use_parallel_var.get()
                )
            else:
                # Original sequential method
                results = self.run_sequential_stage1_optimization(
                    orientations, target_property, progress_bar, progress_window
                )
            
            end_time = time.time()
            optimization_time = end_time - start_time
            
            # Store results
            self.stage_results['stage1'] = results
            
            progress_window.destroy()
            
            # Update plot
            self.update_orientation_plot()
            
            messagebox.showinfo("Success", 
                              f"Stage 1 completed in {optimization_time:.2f} seconds!\n"
                              f"Evaluated {total_orientations} orientations\n"
                              f"Best objective value: {results[0]['objective']:.4f}\n"
                              f"Best orientation: φ={results[0]['euler_angles'][0]:.1f}°, "
                              f"θ={results[0]['euler_angles'][1]:.1f}°, ψ={results[0]['euler_angles'][2]:.1f}°")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Error in Stage 1 optimization: {str(e)}")

    def run_fast_stage1_optimization(self, orientations, target_property, progress_bar, progress_window, use_parallel=False):
        """Fast vectorized Stage 1 optimization with optional parallel processing."""
        if use_parallel:
            try:
                results = self.run_parallel_optimization(orientations, target_property, progress_bar, progress_window)
            except Exception as e:
                print(f"Parallel processing failed: {e}")
                print("Falling back to vectorized method...")
                results = self.run_vectorized_optimization(orientations, target_property, progress_bar, progress_window)
        else:
            results = self.run_vectorized_optimization(orientations, target_property, progress_bar, progress_window)
        
        return results

    def run_vectorized_optimization(self, orientations, target_property, progress_bar, progress_window):
        """Vectorized optimization using NumPy for batch processing."""
        results = []
        batch_size = min(1000, len(orientations))  # Process in batches to manage memory
        
        for batch_start in range(0, len(orientations), batch_size):
            batch_end = min(batch_start + batch_size, len(orientations))
            batch_orientations = orientations[batch_start:batch_end]
            
            # Convert to numpy array for vectorized operations
            orientations_array = np.array(batch_orientations)
            
            # Vectorized evaluation
            batch_objectives = self.evaluate_orientations_vectorized(orientations_array, target_property)
            
            # Store results
            for i, (orientation, objective) in enumerate(zip(batch_orientations, batch_objectives)):
                results.append({
                    'orientation': orientation,
                    'objective': objective,
                    'euler_angles': orientation,
                    'stage': 1
                })
            
            # Update progress less frequently
            progress = batch_end / len(orientations) * 100
            progress_bar['value'] = progress
            progress_window.update()
        
        # Sort results by objective value (descending for maximization)
        results.sort(key=lambda x: x['objective'], reverse=True)
        return results

    def run_sequential_stage1_optimization(self, orientations, target_property, progress_bar, progress_window):
        """Original sequential optimization method."""
        results = []
        total_orientations = len(orientations)
        
        for i, orientation in enumerate(orientations):
            # Update progress every 10 orientations to reduce GUI overhead
            if i % 10 == 0:
                progress = (i + 1) / total_orientations * 100
                progress_bar['value'] = progress
                progress_window.update()
            
            # Calculate objective function for this orientation
            objective_value = self.evaluate_orientation(orientation, target_property)
            
            results.append({
                'orientation': orientation,
                'objective': objective_value,
                'euler_angles': orientation,
                'stage': 1
            })
        
        # Sort results by objective value (descending for maximization)
        results.sort(key=lambda x: x['objective'], reverse=True)
        return results

    def evaluate_orientations_vectorized(self, orientations_array, target_property):
        """Vectorized evaluation of multiple orientations simultaneously."""
        try:
            # For now, use a simplified vectorized approach
            # This can be expanded with full tensor calculations later
            results = []
            for orientation in orientations_array:
                results.append(self.evaluate_orientation(orientation, target_property))
            return np.array(results)
                
        except Exception as e:
            print(f"Error in vectorized evaluation: {e}")
            # Fallback to sequential evaluation
            return [self.evaluate_orientation(orientation, target_property) 
                   for orientation in orientations_array]

    def run_parallel_optimization(self, orientations, target_property, progress_bar, progress_window):
        """Parallel optimization using multiprocessing."""
        import multiprocessing as mp
        from functools import partial
        
        # Determine number of processes
        num_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes
        chunk_size = max(100, len(orientations) // (num_processes * 4))
        
        print(f"Using {num_processes} processes with chunk size {chunk_size}")
        
        # Create partial function with fixed target_property
        eval_func = partial(self.evaluate_orientation_safe, target_property=target_property)
        
        # Use multiprocessing pool
        with mp.Pool(processes=num_processes) as pool:
            # Process in chunks and update progress
            results = []
            processed = 0
            
            for chunk_start in range(0, len(orientations), chunk_size * num_processes):
                chunk_end = min(chunk_start + chunk_size * num_processes, len(orientations))
                chunk_orientations = orientations[chunk_start:chunk_end]
                
                # Process chunk in parallel
                chunk_objectives = pool.map(eval_func, chunk_orientations)
                
                # Store results
                for orientation, objective in zip(chunk_orientations, chunk_objectives):
                    results.append({
                        'orientation': orientation,
                        'objective': objective,
                        'euler_angles': orientation,
                        'stage': 1
                    })
                
                processed += len(chunk_orientations)
                progress = processed / len(orientations) * 100
                progress_bar['value'] = progress
                progress_window.update()
        
        # Sort results
        results.sort(key=lambda x: x['objective'], reverse=True)
        return results

    def evaluate_orientation_safe(self, orientation, target_property):
        """Thread-safe version of evaluate_orientation for multiprocessing."""
        try:
            return self.evaluate_orientation(orientation, target_property)
        except Exception as e:
            print(f"Error evaluating orientation {orientation}: {e}")
            return 0.0
    
    def run_stage2_optimization(self):
        """Run Stage 2: Fine-tuning around best candidates from Stage 1."""
        if not self.stage_results['stage1']:
            messagebox.showwarning("No Stage 1 Results", "Please run Stage 1 optimization first.")
            return
        
        try:
            # Get parameters
            num_candidates = self.stage2_candidates_var.get()
            target_property = self.target_property_var.get()
            
            # Get top candidates from Stage 1
            stage1_results = self.stage_results['stage1']
            top_candidates = stage1_results[:num_candidates]
            
            # Create progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Stage 2 Optimization")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            ttk.Label(progress_window, text="Fine-tuning top candidates...", font=('Arial', 12)).pack(pady=20)
            progress_bar = ttk.Progressbar(progress_window, mode='determinate')
            progress_bar.pack(pady=10, padx=20, fill=tk.X)
            
            progress_window.update()
            
            # Fine-tune each candidate
            refined_results = []
            
            for i, candidate in enumerate(top_candidates):
                # Update progress
                progress = (i + 1) / len(top_candidates) * 100
                progress_bar['value'] = progress
                progress_window.update()
                
                # Create fine grid around this candidate
                base_orientation = candidate['orientation']
                fine_orientations = self.generate_fine_grid(base_orientation, resolution=2.0)
                
                # Evaluate fine grid
                for fine_orientation in fine_orientations:
                    objective_value = self.evaluate_orientation(fine_orientation, target_property)
                    
                    refined_results.append({
                        'orientation': fine_orientation,
                        'objective': objective_value,
                        'euler_angles': fine_orientation,
                        'stage': 2,
                        'parent_candidate': i
                    })
            
            # Sort refined results
            refined_results.sort(key=lambda x: x['objective'], reverse=True)
            
            # Store results
            self.stage_results['stage2'] = refined_results
            
            progress_window.destroy()
            
            # Update plot
            self.update_orientation_plot()
            
            messagebox.showinfo("Success", 
                              f"Stage 2 completed!\n"
                              f"Refined {len(top_candidates)} candidates\n"
                              f"Best objective value: {refined_results[0]['objective']:.4f}\n"
                              f"Best orientation: φ={refined_results[0]['euler_angles'][0]:.1f}°, "
                              f"θ={refined_results[0]['euler_angles'][1]:.1f}°, ψ={refined_results[0]['euler_angles'][2]:.1f}°")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Error in Stage 2 optimization: {str(e)}")
    
    def run_stage3_optimization(self):
        """Run Stage 3: Final optimization using gradient-based methods."""
        if not self.stage_results['stage2']:
            messagebox.showwarning("No Stage 2 Results", "Please run Stage 2 optimization first.")
            return
        
        try:
            # Get parameters
            method = self.opt_method_var.get()
            target_property = self.target_property_var.get()
            
            # Get best candidate from Stage 2
            stage2_results = self.stage_results['stage2']
            best_candidate = stage2_results[0]
            initial_orientation = best_candidate['orientation']
            
            # Create progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Stage 3 Optimization")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            ttk.Label(progress_window, text=f"Running {method} optimization...", font=('Arial', 12)).pack(pady=20)
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10, padx=20, fill=tk.X)
            progress_bar.start()
            
            progress_window.update()
            
            # Define objective function for scipy.optimize
            def objective_func(angles):
                return -self.evaluate_orientation(angles, target_property)  # Negative for minimization
            
            # Set bounds (0-360 degrees for each angle)
            bounds = [(0, 360), (0, 180), (0, 360)]  # phi, theta, psi
            
            # Run optimization
            from scipy.optimize import minimize
            
            result = minimize(
                objective_func,
                initial_orientation,
                method=method,
                bounds=bounds if method in ['L-BFGS-B'] else None,
                options={'maxiter': 1000, 'disp': False}
            )
            
            progress_bar.stop()
            progress_window.destroy()
            
            if result.success:
                optimal_orientation = result.x
                optimal_objective = -result.fun  # Convert back to positive
                
                # Store results
                self.stage_results['stage3'] = {
                    'orientation': optimal_orientation,
                    'objective': optimal_objective,
                    'euler_angles': optimal_orientation,
                    'optimization_result': result,
                    'method': method,
                    'stage': 3
                }
                
                # Update plot
                self.update_orientation_plot()
                
                messagebox.showinfo("Success", 
                                  f"Stage 3 completed!\n"
                                  f"Optimization method: {method}\n"
                                  f"Final objective value: {optimal_objective:.4f}\n"
                                  f"Optimal orientation: φ={optimal_orientation[0]:.2f}°, "
                                  f"θ={optimal_orientation[1]:.2f}°, ψ={optimal_orientation[2]:.2f}°\n"
                                  f"Iterations: {result.nit}")
            else:
                messagebox.showerror("Optimization Failed", 
                                   f"Stage 3 optimization failed:\n{result.message}")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Error in Stage 3 optimization: {str(e)}")
    
    def generate_orientation_grid(self, resolution, crystal_system):
        """Generate a grid of orientations based on crystal system symmetry."""
        orientations = []
        
        # Define angular ranges based on crystal system symmetry
        if crystal_system == "Cubic":
            # Cubic: only need 1/48 of full rotation space due to high symmetry
            phi_range = (0, 90, resolution)
            theta_range = (0, 90, resolution)
            psi_range = (0, 90, resolution)
        elif crystal_system == "Tetragonal":
            # Tetragonal: 1/16 of full space
            phi_range = (0, 90, resolution)
            theta_range = (0, 180, resolution)
            psi_range = (0, 90, resolution)
        elif crystal_system == "Orthorhombic":
            # Orthorhombic: 1/8 of full space
            phi_range = (0, 180, resolution)
            theta_range = (0, 180, resolution)
            psi_range = (0, 90, resolution)
        elif crystal_system == "Hexagonal" or crystal_system == "Trigonal":
            # Hexagonal/Trigonal: 1/12 of full space
            phi_range = (0, 60, resolution)
            theta_range = (0, 180, resolution)
            psi_range = (0, 180, resolution)
        else:
            # Monoclinic, Triclinic, or Unknown: use full space
            phi_range = (0, 360, resolution)
            theta_range = (0, 180, resolution)
            psi_range = (0, 360, resolution)
        
        # Generate grid
        phi_values = np.arange(phi_range[0], phi_range[1] + resolution, resolution)
        theta_values = np.arange(theta_range[0], theta_range[1] + resolution, resolution)
        psi_values = np.arange(psi_range[0], psi_range[1] + resolution, resolution)
        
        for phi in phi_values:
            for theta in theta_values:
                for psi in psi_values:
                    orientations.append([phi, theta, psi])
        
        return orientations
    
    def generate_fine_grid(self, base_orientation, resolution=2.0):
        """Generate a fine grid around a base orientation."""
        phi_base, theta_base, psi_base = base_orientation
        
        # Create fine grid ±10 degrees around base orientation
        delta = 10.0
        
        phi_values = np.arange(phi_base - delta, phi_base + delta + resolution, resolution)
        theta_values = np.arange(max(0, theta_base - delta), min(180, theta_base + delta + resolution), resolution)
        psi_values = np.arange(psi_base - delta, psi_base + delta + resolution, resolution)
        
        # Handle angle wrapping
        phi_values = phi_values % 360
        psi_values = psi_values % 360
        
        orientations = []
        for phi in phi_values:
            for theta in theta_values:
                for psi in psi_values:
                    orientations.append([phi, theta, psi])
        
        return orientations
    
    def evaluate_orientation(self, orientation, target_property):
        """Evaluate the objective function for a given orientation."""
        try:
            phi, theta, psi = orientation
            
            # Convert to radians
            phi_rad = np.radians(phi)
            theta_rad = np.radians(theta)
            psi_rad = np.radians(psi)
            
            # Create rotation matrix from Euler angles (ZYZ convention)
            rotation_matrix = self.euler_to_rotation_matrix(phi_rad, theta_rad, psi_rad)
            
            if target_property == "Maximum Intensity":
                return self.calculate_max_intensity(rotation_matrix)
            elif target_property == "Maximum Contrast":
                return self.calculate_max_contrast(rotation_matrix)
            elif target_property == "Specific Peak Enhancement":
                return self.calculate_peak_enhancement(rotation_matrix)
            elif target_property == "Experimental Match":
                return self.calculate_experimental_match(rotation_matrix)
            elif target_property == "Depolarization Ratio":
                return self.calculate_depolarization_objective(rotation_matrix)
            else:
                # Default to maximum intensity
                return self.calculate_max_intensity(rotation_matrix)
                
        except Exception as e:
            print(f"Error evaluating orientation {orientation}: {e}")
            return 0.0
    
    def euler_to_rotation_matrix(self, phi, theta, psi):
        """Convert Euler angles to rotation matrix (ZYZ convention)."""
        # ZYZ Euler angle convention
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        
        # Rotation matrix
        R = np.array([
            [cos_phi*cos_theta*cos_psi - sin_phi*sin_psi, -cos_phi*cos_theta*sin_psi - sin_phi*cos_psi, cos_phi*sin_theta],
            [sin_phi*cos_theta*cos_psi + cos_phi*sin_psi, -sin_phi*cos_theta*sin_psi + cos_phi*cos_psi, sin_phi*sin_theta],
            [-sin_theta*cos_psi, sin_theta*sin_psi, cos_theta]
        ])
        
        return R
    
    def calculate_max_intensity(self, rotation_matrix):
        """Calculate maximum intensity for a given orientation using fitted peaks or tensor data."""
        # First priority: Use exported tensor data if available
        if hasattr(self, 'tensor_analysis_results') and self.tensor_analysis_results:
            tensor_data = self.tensor_analysis_results
            
            # Use fitted peaks from tensor data if available
            if 'fitted_peaks' in tensor_data and tensor_data['fitted_peaks']:
                total_intensity = 0.0
                
                for peak in tensor_data['fitted_peaks']:
                    amplitude = peak['amplitude']
                    position = peak['position']
                    
                    # Get character from tensor data peak assignments
                    character = 'unknown'
                    if 'peak_assignments' in tensor_data and position in tensor_data['peak_assignments']:
                        character = tensor_data['peak_assignments'][position]['character']
                    elif position in self.peak_assignments:
                        character = self.peak_assignments[position]['character']
                    
                    # Apply orientation-dependent intensity scaling
                    orientation_factor = self.calculate_orientation_intensity_factor(
                        rotation_matrix, position, character
                    )
                    
                    # Calculate orientation-dependent intensity
                    oriented_intensity = amplitude * orientation_factor
                    total_intensity += oriented_intensity
                
                return total_intensity
            
            # Use tensor data directly if no fitted peaks
            elif 'tensors' in tensor_data and 'wavenumbers' in tensor_data:
                tensors = tensor_data['tensors']
                wavenumbers = tensor_data['wavenumbers']
                
                # Calculate intensity for each tensor
                total_intensity = 0.0
                e_incident = np.array([1, 0, 0])  # Incident polarization
                e_scattered = np.array([1, 0, 0])  # Scattered polarization (parallel)
                
                for i, tensor in enumerate(tensors):
                    # Apply rotation to tensor
                    rotated_tensor = rotation_matrix @ tensor @ rotation_matrix.T
                    
                    # Calculate Raman intensity
                    intensity = np.abs(e_incident @ rotated_tensor @ e_scattered)**2
                    total_intensity += intensity
                
                return total_intensity
        
        # Second priority: Use fitted peak data
        if self.fitted_peaks:
            total_intensity = 0.0
            
            for peak in self.fitted_peaks:
                # Get peak properties
                amplitude = peak['amplitude']
                position = peak['position']
                
                # Get character from peak assignments if available
                character = 'unknown'
                if position in self.peak_assignments:
                    character = self.peak_assignments[position]['character']
                
                # Apply orientation-dependent intensity scaling
                orientation_factor = self.calculate_orientation_intensity_factor(
                    rotation_matrix, position, character
                )
                
                # Calculate orientation-dependent intensity
                oriented_intensity = amplitude * orientation_factor
                total_intensity += oriented_intensity
            
            return total_intensity
            
        # Third priority: Use polarization data
        elif self.polarization_data:
            if 'xx' in self.polarization_data:
                intensities = self.polarization_data['xx']['intensities']
                return np.max(intensities)
            else:
                first_config = list(self.polarization_data.keys())[0]
                intensities = self.polarization_data[first_config]['intensities']
                return np.max(intensities)
        
        return 0.0
    
    def calculate_max_contrast(self, rotation_matrix):
        """Calculate maximum contrast between polarized and depolarized components."""
        if self.fitted_peaks and self.peak_assignments:
            # Calculate contrast based on fitted peaks and their symmetries
            polarized_intensity = 0.0
            depolarized_intensity = 0.0
            
            for peak in self.fitted_peaks:
                position = peak['position']
                amplitude = peak['amplitude']
                
                # Get character assignment
                character = 'unknown'
                if position in self.peak_assignments:
                    character = self.peak_assignments[position]['character']
                
                # Calculate orientation factors for parallel and cross polarizations
                # Simulate XX (parallel) and XY (cross) configurations
                parallel_factor = self.calculate_orientation_intensity_factor(
                    rotation_matrix, position, character
                )
                
                # For cross-polarization, modify the scattered polarization vector
                e_incident = np.array([1, 0, 0])
                e_scattered = np.array([0, 1, 0])  # Cross-polarized
                
                raman_tensor = self.get_raman_tensor_for_character(character)
                rotated_tensor = rotation_matrix @ raman_tensor @ rotation_matrix.T
                cross_factor = np.abs(e_incident @ rotated_tensor @ e_scattered)**2
                cross_factor = np.clip(cross_factor, 0.1, 10.0)
                
                # Accumulate intensities
                polarized_intensity += amplitude * parallel_factor
                depolarized_intensity += amplitude * cross_factor
            
            # Calculate contrast
            total_intensity = polarized_intensity + depolarized_intensity
            if total_intensity > 0:
                contrast = (polarized_intensity - depolarized_intensity) / total_intensity
                return abs(contrast)  # Return absolute contrast
            
        elif self.depolarization_ratios:
            # Fallback to depolarization ratio data
            ratios = self.depolarization_ratios['ratio']
            I_xx = self.depolarization_ratios['I_xx']
            
            # Calculate contrast as (I_max - I_min) / (I_max + I_min)
            contrast = (np.max(I_xx) - np.min(I_xx)) / (np.max(I_xx) + np.min(I_xx) + 1e-10)
            return contrast
        
        return 0.0
    
    def calculate_peak_enhancement(self, rotation_matrix):
        """Calculate enhancement of specific peaks using character assignments."""
        if not self.fitted_peaks:
            return 0.0
        
        enhancement_score = 0.0
        
        for peak in self.fitted_peaks:
            amplitude = peak['amplitude']
            position = peak['position']
            character = peak.get('character', 'unknown')
            
            # Get character from peak assignments if available
            if position in self.peak_assignments:
                character = self.peak_assignments[position]['character']
            
            # Calculate orientation-dependent enhancement
            orientation_factor = self.calculate_orientation_intensity_factor(
                rotation_matrix, position, character
            )
            
            # Weight by peak quality and assignment confidence
            quality_weight = peak.get('r_squared', 0.5)  # Use R² as quality metric
            assignment_weight = 1.0 if character != 'unknown' else 0.5
            
            peak_enhancement = amplitude * orientation_factor * quality_weight * assignment_weight
            enhancement_score += peak_enhancement
        
        return enhancement_score
    
    def calculate_orientation_intensity_factor(self, rotation_matrix, frequency, character):
        """Calculate orientation-dependent intensity factor for a vibrational mode."""
        try:
            # This implements simplified Raman scattering intensity calculations
            # I(θ,φ,ψ) = |e_i · R · α · R^T · e_s|²
            # where e_i, e_s are incident/scattered polarization vectors
            # R is rotation matrix, α is Raman tensor
            
            # Get simplified Raman tensor based on vibrational character
            raman_tensor = self.get_raman_tensor_for_character(character)
            
            # Define incident and scattered polarization vectors (simplified)
            # For backscattering geometry: e_i = [1,0,0], e_s = [1,0,0] for XX
            e_incident = np.array([1, 0, 0])
            e_scattered = np.array([1, 0, 0])
            
            # Apply rotation to the Raman tensor
            # α_rotated = R · α · R^T
            rotated_tensor = rotation_matrix @ raman_tensor @ rotation_matrix.T
            
            # Calculate scattering intensity: |e_i · α_rotated · e_s|²
            intensity = np.abs(e_incident @ rotated_tensor @ e_scattered)**2
            
            # Normalize to prevent extreme values
            intensity = np.clip(intensity, 0.1, 10.0)
            
            return intensity
            
        except Exception as e:
            print(f"Error calculating orientation factor for {character}: {e}")
            return 1.0  # Default value
    
    def get_raman_tensor_for_character(self, character):
        """Get Raman tensor based on vibrational mode character."""
        # First, try to use actual tensor data from tensor analysis if available
        if hasattr(self, 'tensor_analysis_results') and self.tensor_analysis_results:
            tensor_data = self.tensor_analysis_results
            peak_tensors = tensor_data.get('peak_tensors', {})
            
            # Look for a tensor that matches this character or frequency
            for freq, peak_data in peak_tensors.items():
                if 'tensor' in peak_data:
                    # Use the actual calculated tensor
                    return peak_data['tensor']
            
            # If no specific peak tensor, use the first available tensor
            if 'tensors' in tensor_data and len(tensor_data['tensors']) > 0:
                return tensor_data['tensors'][0]  # Use first tensor as representative
        
        # Fallback to simplified implementation based on group theory
        character_lower = character.lower()
        
        if any(sym in character_lower for sym in ['a1', 'a1g', 'ag']):
            # Totally symmetric modes - isotropic tensor
            return np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
            
        elif any(sym in character_lower for sym in ['e', 'eg']):
            # Doubly degenerate modes - anisotropic tensor
            return np.array([
                [1.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 0.2]
            ])
            
        elif any(sym in character_lower for sym in ['t', 't2', 't2g']):
            # Triply degenerate modes - antisymmetric components
            return np.array([
                [0.3, 0.0, 0.7],
                [0.0, 0.3, 0.0],
                [0.7, 0.0, 0.3]
            ])
            
        elif any(sym in character_lower for sym in ['b1', 'b2', 'b3']):
            # B-type modes - specific tensor patterns
            return np.array([
                [0.5, 0.8, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 0.0, 0.1]
            ])
            
        else:
            # Unknown character - use average tensor
            return np.array([
                [0.6, 0.2, 0.1],
                [0.2, 0.6, 0.1],
                [0.1, 0.1, 0.4]
                         ])
    
    def calculate_experimental_match(self, rotation_matrix):
        """Calculate how well the oriented calculated spectrum matches experimental data."""
        if not self.fitted_peaks:
            return 0.0
        
        # Use current spectrum if available, otherwise try imported spectrum
        reference_spectrum = self.current_spectrum or self.imported_spectrum
        if not reference_spectrum:
            return 0.0
        
        try:
            # Get experimental spectrum
            exp_wavenumbers = reference_spectrum['wavenumbers']
            exp_intensities = reference_spectrum['intensities']
            
            # Generate calculated spectrum with current orientation
            calc_wavenumbers = exp_wavenumbers.copy()
            calc_intensities = np.zeros_like(exp_wavenumbers)
            
            # Add oriented peaks to calculated spectrum
            for peak in self.fitted_peaks:
                position = peak['position']
                amplitude = peak['amplitude']
                width = peak.get('width', 8.0)
                character = 'unknown'
                
                # Get character assignment
                if position in self.peak_assignments:
                    character = self.peak_assignments[position]['character']
                
                # Apply orientation-dependent scaling
                orientation_factor = self.calculate_orientation_intensity_factor(
                    rotation_matrix, position, character
                )
                
                # Apply frequency shift if available
                if position in self.frequency_shifts:
                    shift = self.frequency_shifts[position]['shift']
                    shifted_position = position + shift
                else:
                    shifted_position = position
                
                # Generate oriented peak
                oriented_amplitude = amplitude * orientation_factor
                
                # Add Lorentzian peak to calculated spectrum
                peak_contribution = oriented_amplitude * (width**2) / (
                    (calc_wavenumbers - shifted_position)**2 + width**2
                )
                calc_intensities += peak_contribution
            
            # Normalize calculated spectrum to experimental scale
            if np.max(calc_intensities) > 0:
                calc_intensities = calc_intensities * (np.max(exp_intensities) / np.max(calc_intensities))
            
            # Calculate match quality using correlation coefficient
            # Focus on the region where we have fitted peaks
            if self.fitted_peaks:
                min_freq = min(peak['position'] for peak in self.fitted_peaks) - 100
                max_freq = max(peak['position'] for peak in self.fitted_peaks) + 100
                
                # Create mask for relevant frequency range
                mask = (exp_wavenumbers >= min_freq) & (exp_wavenumbers <= max_freq)
                
                if np.sum(mask) > 10:  # Ensure we have enough points
                    exp_region = exp_intensities[mask]
                    calc_region = calc_intensities[mask]
                    
                    # Calculate correlation coefficient
                    correlation = np.corrcoef(exp_region, calc_region)[0, 1]
                    
                    # Handle NaN case
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    # Also calculate R² for goodness of fit
                    ss_res = np.sum((exp_region - calc_region)**2)
                    ss_tot = np.sum((exp_region - np.mean(exp_region))**2)
                    r_squared = 1 - (ss_res / (ss_tot + 1e-10))
                    r_squared = max(0, r_squared)  # Ensure non-negative
                    
                    # Combine correlation and R² for final score
                    match_score = 0.7 * abs(correlation) + 0.3 * r_squared
                    
                    return match_score
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating experimental match: {e}")
            return 0.0
     
    def calculate_depolarization_objective(self, rotation_matrix):
        """Calculate objective based on depolarization ratio."""
        if not self.depolarization_ratios:
            return 0.0
        
        # Objective: minimize variance in depolarization ratio (more consistent measurements)
        ratios = self.depolarization_ratios['ratio']
        return -np.var(ratios)  # Negative variance (higher is better)
    
    def update_orientation_plot(self):
        """Update the orientation optimization plot."""
        if "Orientation Optimization" not in self.plot_components:
            return
        
        components = self.plot_components["Orientation Optimization"]
        ax = components['ax']
        canvas = components['canvas']
        fig = ax.get_figure()
        
        # Clear the entire figure to reset layout
        fig.clear()
        
        # Recreate the axes with proper layout
        ax = fig.add_subplot(111)
        
        # Update the stored reference
        components['ax'] = ax
        
        # Plot results from different stages
        if self.stage_results['stage1']:
            self.plot_stage_results(ax, fig)
        else:
            ax.text(0.5, 0.5, 'Run Stage 1 optimization to see results', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=12, alpha=0.6)
            ax.set_title("Orientation Optimization Results")
        
        ax.grid(True, alpha=0.3)
        
        # Adjust layout to prevent overlap
        fig.tight_layout()
        canvas.draw()
    
    def plot_stage_results(self, ax, fig):
        """Plot optimization results from all stages."""
        scatter_for_colorbar = None
        
        # Plot Stage 1 results
        if self.stage_results['stage1']:
            stage1_results = self.stage_results['stage1'][:50]  # Top 50 for visibility
            phi_values = [r['euler_angles'][0] for r in stage1_results]
            theta_values = [r['euler_angles'][1] for r in stage1_results]
            objectives = [r['objective'] for r in stage1_results]
            
            scatter1 = ax.scatter(phi_values, theta_values, c=objectives, s=20, alpha=0.6, 
                                 cmap='viridis', label='Stage 1')
            scatter_for_colorbar = scatter1
        
        # Plot Stage 2 results
        if self.stage_results['stage2']:
            stage2_results = self.stage_results['stage2'][:20]  # Top 20
            phi_values = [r['euler_angles'][0] for r in stage2_results]
            theta_values = [r['euler_angles'][1] for r in stage2_results]
            objectives = [r['objective'] for r in stage2_results]
            
            scatter2 = ax.scatter(phi_values, theta_values, c=objectives, s=50, alpha=0.8, 
                                 cmap='plasma', marker='s', label='Stage 2')
            # Use Stage 2 for colorbar if available (more refined results)
            scatter_for_colorbar = scatter2
        
        # Plot Stage 3 result
        if self.stage_results['stage3']:
            stage3_result = self.stage_results['stage3']
            phi_opt = stage3_result['euler_angles'][0]
            theta_opt = stage3_result['euler_angles'][1]
            obj_opt = stage3_result['objective']
            
            ax.scatter([phi_opt], [theta_opt], c='red', s=200, marker='*', 
                      edgecolors='black', linewidth=2, label=f'Stage 3 (Optimal)', zorder=10)
            
            # Add annotation
            ax.annotate(f'Optimal\nφ={phi_opt:.1f}°\nθ={theta_opt:.1f}°\nObj={obj_opt:.3f}',
                       xy=(phi_opt, theta_opt), xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                       fontsize=9, ha='left')
        
        ax.set_xlabel('φ (degrees)')
        ax.set_ylabel('θ (degrees)')
        ax.set_title('Orientation Optimization Results')
        ax.legend()
        
        # Add colorbar if we have data - use proper layout management
        if scatter_for_colorbar is not None:
            try:
                # Create colorbar with proper sizing
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(scatter_for_colorbar, cax=cax, label='Objective Value')
            except Exception as e:
                # Fallback to simple colorbar if axes_grid1 is not available
                try:
                    cbar = fig.colorbar(scatter_for_colorbar, ax=ax, label='Objective Value', 
                                       shrink=0.8, aspect=20)
                except Exception as e2:
                    print(f"Warning: Could not add colorbar: {e2}")
    
    def show_optimization_results(self):
        """Show detailed optimization results."""
        if not any(self.stage_results.values()):
            messagebox.showwarning("No Results", "No optimization results available.")
            return
        
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Optimization Results")
        results_window.geometry("600x500")
        
        # Create text widget
        text_frame = ttk.Frame(results_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate report
        report = "ORIENTATION OPTIMIZATION RESULTS\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Target Property: {self.target_property_var.get()}\n"
        report += f"Crystal System: {self.opt_crystal_system_var.get()}\n\n"
        
        # Stage 1 results
        if self.stage_results['stage1']:
            stage1 = self.stage_results['stage1']
            report += f"STAGE 1 - COARSE SEARCH:\n"
            report += f"Resolution: {self.stage1_resolution_var.get():.1f}°\n"
            report += f"Total orientations evaluated: {len(stage1)}\n"
            report += f"Best result:\n"
            best = stage1[0]
            report += f"  Orientation: φ={best['euler_angles'][0]:.1f}°, θ={best['euler_angles'][1]:.1f}°, ψ={best['euler_angles'][2]:.1f}°\n"
            report += f"  Objective value: {best['objective']:.4f}\n\n"
        
        # Stage 2 results
        if self.stage_results['stage2']:
            stage2 = self.stage_results['stage2']
            report += f"STAGE 2 - FINE TUNING:\n"
            report += f"Candidates refined: {self.stage2_candidates_var.get()}\n"
            report += f"Total orientations evaluated: {len(stage2)}\n"
            report += f"Best result:\n"
            best = stage2[0]
            report += f"  Orientation: φ={best['euler_angles'][0]:.1f}°, θ={best['euler_angles'][1]:.1f}°, ψ={best['euler_angles'][2]:.1f}°\n"
            report += f"  Objective value: {best['objective']:.4f}\n\n"
        
        # Stage 3 results
        if self.stage_results['stage3']:
            stage3 = self.stage_results['stage3']
            report += f"STAGE 3 - FINAL OPTIMIZATION:\n"
            report += f"Method: {stage3['method']}\n"
            report += f"Iterations: {stage3['optimization_result'].nit}\n"
            report += f"Success: {stage3['optimization_result'].success}\n"
            report += f"Final result:\n"
            report += f"  Orientation: φ={stage3['euler_angles'][0]:.2f}°, θ={stage3['euler_angles'][1]:.2f}°, ψ={stage3['euler_angles'][2]:.2f}°\n"
            report += f"  Objective value: {stage3['objective']:.6f}\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS:\n"
        report += "-" * 20 + "\n"
        if self.stage_results['stage3']:
            final_result = self.stage_results['stage3']
            report += f"Optimal crystal orientation for {self.target_property_var.get().lower()}:\n"
            report += f"  φ = {final_result['euler_angles'][0]:.2f}°\n"
            report += f"  θ = {final_result['euler_angles'][1]:.2f}°\n"
            report += f"  ψ = {final_result['euler_angles'][2]:.2f}°\n\n"
            report += "Apply this orientation to your experimental setup for optimal results.\n"
        
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
    
    def export_orientations(self):
        """Export optimization results to file."""
        messagebox.showinfo("Export", "Export orientations functionality will be implemented.")
    
    def apply_optimal_orientation(self):
        """Apply optimal orientation to experimental setup."""
        messagebox.showinfo("Apply Orientation", "Apply optimal orientation functionality will be implemented.")
    
    def update_optimization_status(self):
        """Update the optimization data status display."""
        if not hasattr(self, 'opt_status_text'):
            return
        
        self.opt_status_text.config(state=tk.NORMAL)
        self.opt_status_text.delete(1.0, tk.END)
        
        status = "OPTIMIZATION DATA STATUS:\n"
        status += "-" * 25 + "\n"
        
        # SYMMETRY INFORMATION (Most Important)
        status += "🔬 SYMMETRY INFORMATION:\n"
        crystal_system = "Unknown"
        point_group = "Unknown"
        space_group = "Unknown"
        source = "None"
        
        # Check tensor analysis data first (highest priority)
        if hasattr(self, 'tensor_analysis_results') and self.tensor_analysis_results:
            tensor_data = self.tensor_analysis_results
            crystal_system = tensor_data.get('crystal_system', 'Unknown')
            point_group = tensor_data.get('point_group', 'Unknown')
            space_group = tensor_data.get('space_group', 'Unknown')
            source = "Tensor Analysis"
        # Check crystal structure
        elif self.crystal_structure:
            crystal_system = self.get_crystal_system_from_structure()
            point_group = self.crystal_structure.get('point_group', 'Unknown')
            space_group = self.crystal_structure.get('space_group', 'Unknown')
            source = "Crystal Structure"
        elif self.selected_reference_mineral:
            crystal_system = self.infer_crystal_system(self.selected_reference_mineral)
            # Try to get point group from mineral database
            if self.selected_reference_mineral in self.mineral_database:
                mineral_data = self.mineral_database[self.selected_reference_mineral]
                point_group = mineral_data.get('point_group', 'Unknown')
                space_group = mineral_data.get('space_group', 'Unknown')
            source = "Mineral Database"
        
        # Display symmetry information prominently
        if crystal_system != "Unknown":
            status += f"  ✓ Crystal System: {crystal_system}\n"
        else:
            status += f"  ❌ Crystal System: Unknown\n"
            
        if point_group != "Unknown":
            status += f"  ✓ Point Group: {point_group}\n"
        else:
            status += f"  ❌ Point Group: Unknown\n"
            
        if space_group != "Unknown":
            status += f"  ✓ Space Group: {space_group}\n"
            
        if source != "None":
            status += f"  📍 Source: {source}\n"
        else:
            status += f"  ⚠️  No symmetry data available\n"
        
        status += "\n📊 PEAK DATA:\n"
        # Check fitted peaks
        if self.fitted_peaks:
            status += f"  ✓ Fitted Peaks: {len(self.fitted_peaks)}\n"
            assigned_peaks = len([p for p in self.fitted_peaks if p['position'] in self.peak_assignments])
            status += f"  ✓ Character Assigned: {assigned_peaks}/{len(self.fitted_peaks)}\n"
        else:
            status += "  ❌ No fitted peaks\n"
        
        # Check tensor data
        if hasattr(self, 'tensor_analysis_results') and self.tensor_analysis_results:
            tensor_data = self.tensor_analysis_results
            mineral_name = tensor_data.get('mineral_name', 'Unknown')
            status += f"  ✓ Tensor Data: {len(tensor_data.get('wavenumbers', []))} frequencies\n"
            if mineral_name != 'Unknown':
                status += f"  ✓ Mineral: {mineral_name}\n"
        
        status += "\n📈 EXPERIMENTAL DATA:\n"
        # Check experimental spectrum
        if self.current_spectrum:
            status += f"  ✓ Experimental Spectrum: {self.current_spectrum['name']}\n"
        elif self.imported_spectrum:
            status += f"  ✓ Imported Spectrum: {self.imported_spectrum['name']}\n"
        else:
            status += "  ❌ No experimental spectrum\n"
        
        # Check frequency shifts
        if self.frequency_shifts:
            status += f"  ✓ Frequency Shifts: {len(self.frequency_shifts)}\n"
        
        # OPTIMIZATION READINESS
        status += "\n🎯 OPTIMIZATION READINESS:\n"
        has_tensor_data = hasattr(self, 'tensor_analysis_results') and self.tensor_analysis_results
        has_crystal_info = (crystal_system != "Unknown")
        has_point_group = (point_group != "Unknown")
        
        if has_tensor_data:
            status += "  ✅ Tensor data available - READY!\n"
            if has_point_group:
                status += f"  ✅ Using {point_group} symmetry constraints\n"
        elif self.fitted_peaks and self.peak_assignments and has_crystal_info:
            status += "  ✅ Peak data + symmetry - READY!\n"
            if has_point_group:
                status += f"  ✅ Using {point_group} symmetry constraints\n"
        else:
            status += "  ⚠️  Missing required data:\n"
            if not self.fitted_peaks:
                status += "    • Fit peaks in Peak Fitting tab\n"
            if not self.peak_assignments:
                status += "    • Assign characters in Peak Fitting tab\n"
            if not has_crystal_info:
                status += "    • Import crystal structure\n"
            if not has_point_group:
                status += "    • Point group needed for tensor constraints\n"
        
        self.opt_status_text.insert(tk.END, status)
        self.opt_status_text.config(state=tk.DISABLED)
        
        # Also update the point group display
        self.update_point_group_display()
    
    def update_point_group_display(self):
        """Update the point group display in optimization tab."""
        if not hasattr(self, 'opt_point_group_label'):
            return
        
        point_group = "Unknown"
        
        # Check tensor analysis data first (highest priority)
        if hasattr(self, 'tensor_analysis_results') and self.tensor_analysis_results:
            point_group = self.tensor_analysis_results.get('point_group', 'Unknown')
        # Check crystal structure
        elif self.crystal_structure:
            point_group = self.crystal_structure.get('point_group', 'Unknown')
        # Check mineral database
        elif self.selected_reference_mineral and self.selected_reference_mineral in self.mineral_database:
            mineral_data = self.mineral_database[self.selected_reference_mineral]
            point_group = mineral_data.get('point_group', 'Unknown')
        
        # Update the label
        if point_group != "Unknown":
            self.opt_point_group_label.config(text=point_group, background="lightgreen")
        else:
            self.opt_point_group_label.config(text="Unknown", background="lightcoral")
    
    def get_crystal_system_from_structure(self):
        """Extract crystal system from loaded crystal structure."""
        if not self.crystal_structure:
            return "Unknown"
        
        # First, try direct crystal_system field (from pymatgen)
        if 'crystal_system' in self.crystal_structure:
            return self.crystal_structure['crystal_system']
        
        # Try to infer from space group
        if 'space_group' in self.crystal_structure:
            space_group = self.crystal_structure['space_group']
            return self.infer_system_from_space_group_symbol(space_group)
        
        # Try to infer from space group number
        if 'space_group_number' in self.crystal_structure:
            space_group_num = self.crystal_structure['space_group_number']
            return self.infer_system_from_space_group(space_group_num)
        
        # Try to infer from lattice parameters
        if 'lattice_parameters' in self.crystal_structure:
            lattice = self.crystal_structure['lattice_parameters']
            return self.infer_system_from_lattice(lattice)
        
        return "Unknown"
    
    def infer_system_from_space_group_symbol(self, space_group_symbol):
        """Infer crystal system from space group symbol (Hermann-Mauguin notation)."""
        if not space_group_symbol:
            return "Unknown"
        
        symbol = space_group_symbol.strip().upper()
        
        # Cubic systems
        cubic_symbols = ['P23', 'F23', 'I23', 'P213', 'I213', 'PM3', 'PN3', 'FM3', 'FD3', 'IM3', 'PA3', 'IA3',
                        'P432', 'P4232', 'F432', 'F4132', 'I432', 'P4332', 'P4132', 'I4132',
                        'P-43M', 'F-43M', 'I-43M', 'P-43N', 'F-43C', 'I-43D',
                        'PM-3M', 'PN-3N', 'PM-3N', 'PN-3M', 'FM-3M', 'FM-3C', 'FD-3M', 'FD-3C',
                        'IM-3M', 'IA-3D']
        
        # Tetragonal systems
        tetragonal_symbols = ['P4', 'P41', 'P42', 'P43', 'I4', 'I41',
                             'P-4', 'I-4',
                             'P4/M', 'P42/M', 'P4/N', 'P42/N', 'I4/M', 'I41/A',
                             'P422', 'P4212', 'P4122', 'P41212', 'P4222', 'P42212', 'P4322', 'P43212',
                             'I422', 'I4122',
                             'P4MM', 'P4BM', 'P42CM', 'P42NM', 'P4CC', 'P4NC', 'P42MC', 'P42BC',
                             'I4MM', 'I4CM', 'I41MD', 'I41CD',
                             'P-42M', 'P-42C', 'P-421M', 'P-421C', 'P-4M2', 'P-4C2', 'P-4B2', 'P-4N2',
                             'I-4M2', 'I-4C2', 'I-42M', 'I-42D',
                             'P4/MMM', 'P4/MCC', 'P4/NBM', 'P4/NNC', 'P4/MBM', 'P4/MNC', 'P4/NMM', 'P4/NCC',
                             'P42/MMC', 'P42/MCM', 'P42/NBC', 'P42/NNM', 'P42/MBC', 'P42/MNM', 'P42/NMC', 'P42/NCM',
                             'I4/MMM', 'I4/MCM', 'I41/AMD', 'I41/ACD']
        
        # Check for matches
        for cubic in cubic_symbols:
            if cubic in symbol:
                return "Cubic"
        
        for tetra in tetragonal_symbols:
            if tetra in symbol:
                return "Tetragonal"
        
        # Hexagonal/Trigonal (simplified check)
        if any(x in symbol for x in ['P3', 'R3', 'P6', 'R6']):
            if any(x in symbol for x in ['P6', 'P-6']):
                return "Hexagonal"
            else:
                return "Trigonal"
        
        # Orthorhombic
        if any(x in symbol for x in ['P222', 'P2221', 'P21212', 'P212121', 'C2221', 'C222', 'F222', 'I222', 'I212121',
                                    'PMM2', 'PMC21', 'PCC2', 'PMA2', 'PCA21', 'PNC2', 'PMN21', 'PBA2', 'PAN2', 'PNA21', 'PNN2',
                                    'CMM2', 'CMC21', 'CCC2', 'AMM2', 'ABM2', 'AMA2', 'ABA2', 'FMM2', 'FDD2', 'IMM2', 'IBA2', 'IMA2',
                                    'PMMM', 'PNNN', 'PCCM', 'PBAN', 'PMMA', 'PNNA', 'PMNA', 'PCCA', 'PBAM', 'PCCN', 'PBCM', 'PNNM', 'PMMN', 'PBCN', 'PBCA', 'PNMA',
                                    'CMCM', 'CMCA', 'CMMM', 'CCCM', 'CMMA', 'CCCA', 'FMMM', 'FDDD', 'IMMM', 'IBAM', 'IBCA', 'IMMA']):
            return "Orthorhombic"
        
        # Monoclinic
        if any(x in symbol for x in ['P2', 'P21', 'C2', 'PM', 'PC', 'CM', 'CC', 'P2/M', 'P21/M', 'C2/M', 'P2/C', 'P21/C', 'C2/C']):
            return "Monoclinic"
        
        # Triclinic
        if any(x in symbol for x in ['P1', 'P-1']):
            return "Triclinic"
        
        return "Unknown"
    
    def infer_system_from_lattice(self, lattice_params):
        """Infer crystal system from lattice parameters."""
        if not lattice_params or not all(key in lattice_params for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
            return "Unknown"
        
        a = lattice_params['a']
        b = lattice_params['b']
        c = lattice_params['c']
        alpha = lattice_params['alpha']
        beta = lattice_params['beta']
        gamma = lattice_params['gamma']
        
        # Tolerance for floating point comparison
        tol = 1e-3
        angle_tol = 0.1  # degrees
        
        # Check if lengths are equal
        a_eq_b = abs(a - b) < tol
        b_eq_c = abs(b - c) < tol
        a_eq_c = abs(a - c) < tol
        
        # Check if angles are 90 degrees
        alpha_90 = abs(alpha - 90.0) < angle_tol
        beta_90 = abs(beta - 90.0) < angle_tol
        gamma_90 = abs(gamma - 90.0) < angle_tol
        
        # Check if angles are 120 degrees
        gamma_120 = abs(gamma - 120.0) < angle_tol
        
        # Cubic: a = b = c, α = β = γ = 90°
        if a_eq_b and b_eq_c and alpha_90 and beta_90 and gamma_90:
            return "Cubic"
        
        # Tetragonal: a = b ≠ c, α = β = γ = 90°
        elif a_eq_b and not a_eq_c and alpha_90 and beta_90 and gamma_90:
            return "Tetragonal"
        
        # Orthorhombic: a ≠ b ≠ c, α = β = γ = 90°
        elif not a_eq_b and not b_eq_c and not a_eq_c and alpha_90 and beta_90 and gamma_90:
            return "Orthorhombic"
        
        # Hexagonal: a = b ≠ c, α = β = 90°, γ = 120°
        elif a_eq_b and not a_eq_c and alpha_90 and beta_90 and gamma_120:
            return "Hexagonal"
        
        # Trigonal: a = b = c, α = β = γ ≠ 90°
        elif a_eq_b and b_eq_c and abs(alpha - beta) < angle_tol and abs(beta - gamma) < angle_tol and not alpha_90:
            return "Trigonal"
        
        # Monoclinic: a ≠ b ≠ c, α = γ = 90° ≠ β
        elif not a_eq_b and not b_eq_c and not a_eq_c and alpha_90 and gamma_90 and not beta_90:
            return "Monoclinic"
        
        # Triclinic: a ≠ b ≠ c, α ≠ β ≠ γ ≠ 90°
        else:
            return "Triclinic"
    
    # Placeholder methods for Raman Tensors tab
    def extract_tensors_from_data(self):
        """Extract Raman tensors from polarization data, prioritizing database eigenvalues."""
        # First, try to get tensor data from mineral database
        database_success = self.extract_tensors_from_database()
        
        # If database extraction failed, fall back to polarization data
        if not database_success and self.polarization_data:
            self.extract_tensors_from_polarization()
        elif not database_success and not self.polarization_data:
            messagebox.showwarning("No Data", 
                                 "Please load polarized spectra in the Polarization tab first, "
                                 "or select a reference mineral with tensor data.")
    
    def extract_tensors_from_database(self):
        """Extract tensor data from mineral database (preferred method)."""
        if not self.selected_reference_mineral or not self.mineral_database:
            return False
        
        if self.selected_reference_mineral not in self.mineral_database:
            return False
        
        try:
            mineral_data = self.mineral_database[self.selected_reference_mineral]
            
            # Check if tensor data is available
            has_eigenvalues = 'eigenvalues' in mineral_data
            has_n_tensor = 'n_tensor' in mineral_data
            has_dielectric = 'dielectric_tensors' in mineral_data
            
            if not (has_eigenvalues or has_n_tensor or has_dielectric):
                return False
            
            # Extract frequency data from modes
            frequencies = []
            if 'modes' in mineral_data and mineral_data['modes']:
                for mode in mineral_data['modes']:
                    if isinstance(mode, (tuple, list)) and len(mode) >= 1:
                        freq = float(mode[0])
                        if freq > 50:  # Skip low-frequency modes
                            frequencies.append(freq)
            
            if not frequencies:
                # Create a default frequency range
                frequencies = np.linspace(200, 1800, 50)
            
            frequencies = np.array(sorted(frequencies))
            
            # Initialize tensor storage with database data
            self.calculated_raman_tensors = {
                'wavenumbers': frequencies,
                'source': 'database',
                'mineral_name': self.selected_reference_mineral,
                'analysis_complete': True,
                'database_eigenvalues': {},
                'database_tensors': {}
            }
            
            # Extract eigenvalue data
            if has_eigenvalues:
                eigenval_data = mineral_data['eigenvalues']
                self.calculated_raman_tensors['database_eigenvalues'] = eigenval_data
                
                # Convert to frequency-dependent eigenvalues
                if 'n' in eigenval_data:
                    n_eigenvals = eigenval_data['n']
                    # Replicate eigenvalues for each frequency
                    eigenvalues = np.tile(n_eigenvals, (len(frequencies), 1))
                    self.calculated_raman_tensors['eigenvalues'] = eigenvalues
                    
                    # Calculate eigenvectors (assume principal axes for now)
                    eigenvectors = np.zeros((len(frequencies), 3, 3))
                    for i in range(len(frequencies)):
                        eigenvectors[i] = np.eye(3)  # Identity matrix for principal axes
                    self.calculated_raman_tensors['eigenvectors'] = eigenvectors
            
            # Extract tensor matrices
            if has_n_tensor:
                n_tensor = np.array(mineral_data['n_tensor'])
                self.calculated_raman_tensors['database_tensors']['n_tensor'] = n_tensor
                
                # Generate mode-specific tensors based on crystal symmetry
                tensor_matrices = self.generate_mode_specific_tensors(frequencies, n_tensor, mineral_data)
                self.calculated_raman_tensors['full_tensors'] = tensor_matrices
            
            # Extract dielectric tensor data
            if has_dielectric:
                dielectric_data = mineral_data['dielectric_tensors']
                self.calculated_raman_tensors['database_tensors']['dielectric'] = dielectric_data
                
                # Parse dielectric tensor components (DataFrame format)
                eps_inf_tensor = np.zeros((3, 3))
                eps_0_tensor = np.zeros((3, 3))
                
                try:
                    # Handle pandas DataFrame format
                    if hasattr(dielectric_data, 'iterrows'):
                        for _, row in dielectric_data.iterrows():
                            tensor_type = row.get('Tensor', '')
                            component = row.get('Component', '')
                            x_val = float(row.get('X', 0))
                            y_val = float(row.get('Y', 0))
                            z_val = float(row.get('Z', 0))
                            
                            if tensor_type == 'Ɛ∞':
                                if component == 'xx':
                                    eps_inf_tensor[0, 0] = x_val
                                elif component == 'yy':
                                    eps_inf_tensor[1, 1] = y_val
                                elif component == 'zz':
                                    eps_inf_tensor[2, 2] = z_val
                            elif tensor_type == 'Ɛ0':
                                if component == 'xx':
                                    eps_0_tensor[0, 0] = x_val
                                elif component == 'yy':
                                    eps_0_tensor[1, 1] = y_val
                                elif component == 'zz':
                                    eps_0_tensor[2, 2] = z_val
                    else:
                        # Handle list of dictionaries format
                        for tensor_info in dielectric_data:
                            tensor_type = tensor_info.get('Tensor', '')
                            component = tensor_info.get('Component', '')
                            x_val = float(tensor_info.get('X', 0))
                            y_val = float(tensor_info.get('Y', 0))
                            z_val = float(tensor_info.get('Z', 0))
                            
                            if tensor_type == 'Ɛ∞':
                                if component == 'xx':
                                    eps_inf_tensor[0, 0] = x_val
                                elif component == 'yy':
                                    eps_inf_tensor[1, 1] = y_val
                                elif component == 'zz':
                                    eps_inf_tensor[2, 2] = z_val
                            elif tensor_type == 'Ɛ0':
                                if component == 'xx':
                                    eps_0_tensor[0, 0] = x_val
                                elif component == 'yy':
                                    eps_0_tensor[1, 1] = y_val
                                elif component == 'zz':
                                    eps_0_tensor[2, 2] = z_val
                
                    self.calculated_raman_tensors['database_tensors']['eps_inf'] = eps_inf_tensor
                    self.calculated_raman_tensors['database_tensors']['eps_0'] = eps_0_tensor
                    
                except Exception as e:
                    print(f"Warning: Could not parse dielectric tensor data: {e}")
                    # Continue without dielectric tensor data
            
            # Calculate derived properties from database data
            self.calculate_tensor_properties_from_database()
            
            # Transfer calculated_raman_tensors to raman_tensors for 3D visualization compatibility
            self.transfer_calculated_tensors_to_main()
            
            # Update UI
            self.update_tensor_peak_selection()
            self.update_tensor_status()
            
            messagebox.showinfo("Success", 
                              f"Extracted tensor data from database for {self.selected_reference_mineral}.\n"
                              f"Using high-quality eigenvalues and tensor data.\n"
                              f"Data is now available for 3D visualization!")
            
            return True
            
        except Exception as e:
            print(f"Error extracting tensors from database: {e}")
            return False
    
    def generate_mode_specific_tensors(self, frequencies, base_tensor, mineral_data):
        """Generate mode-specific Raman tensors based on crystal symmetry and vibrational modes."""
        try:
            # Get crystal system information
            crystal_system = self.infer_crystal_system(self.selected_reference_mineral)
            space_group = mineral_data.get('space_group', 'Unknown')
            
            # Initialize tensor array
            tensor_matrices = np.zeros((len(frequencies), 3, 3))
            
            # Get mode information if available
            modes = mineral_data.get('modes', [])
            mode_characters = []
            
            # Extract mode characters if available
            for mode in modes:
                if isinstance(mode, (tuple, list)) and len(mode) >= 2:
                    freq = float(mode[0])
                    character = mode[1] if len(mode) > 1 else 'A1g'  # Default character
                    mode_characters.append((freq, character))
            
            print(f"Debug: Found {len(mode_characters)} mode characters for {self.selected_reference_mineral}")
            if mode_characters:
                print(f"Debug: Sample modes: {mode_characters[:5]}")  # Show first 5 modes
            
            # Generate tensors for each frequency
            for i, freq in enumerate(frequencies):
                # Find the closest mode character
                mode_char = 'A1g'  # Default
                if mode_characters:
                    closest_mode = min(mode_characters, key=lambda x: abs(x[0] - freq))
                    if abs(closest_mode[0] - freq) < 50:  # Within 50 cm⁻¹
                        mode_char = closest_mode[1]
                
                # Generate tensor based on symmetry character and crystal system
                tensor = self.generate_tensor_for_mode(mode_char, crystal_system, base_tensor, freq)
                tensor_matrices[i] = tensor
            
            return tensor_matrices
            
        except Exception as e:
            print(f"Warning: Could not generate mode-specific tensors: {e}")
            # Fallback: create slightly different tensors for each frequency
            tensor_matrices = np.zeros((len(frequencies), 3, 3))
            for i, freq in enumerate(frequencies):
                # Add small frequency-dependent variations to avoid identical tensors
                variation = 0.1 * np.sin(freq / 100.0)  # Small sinusoidal variation
                tensor = base_tensor.copy()
                
                # Ensure all diagonal elements are non-zero
                min_diagonal = np.max(np.abs(np.diag(base_tensor))) * 0.01  # 1% of max diagonal
                
                # Add variations to diagonal elements
                tensor[0, 0] = max(tensor[0, 0] * (1 + variation), min_diagonal)
                tensor[1, 1] = max(tensor[1, 1] * (1 + 0.5 * variation), min_diagonal)
                tensor[2, 2] = max(tensor[2, 2] * (1 + 0.3 * variation), min_diagonal)
                
                # Add small off-diagonal elements for some frequencies
                if i % 3 == 1:  # Every third mode gets xy coupling
                    tensor[0, 1] = tensor[1, 0] = 0.1 * tensor[0, 0]
                elif i % 3 == 2:  # Every third mode gets xz coupling
                    tensor[0, 2] = tensor[2, 0] = 0.05 * tensor[0, 0]
                
                tensor_matrices[i] = tensor
            
            return tensor_matrices
    
    def generate_tensor_for_mode(self, mode_character, crystal_system, base_tensor, frequency):
        """Generate a Raman tensor for a specific vibrational mode character."""
        try:
            # Start with base tensor
            tensor = base_tensor.copy()
            
            # Apply symmetry-specific modifications based on mode character
            if crystal_system.lower() == 'tetragonal':
                if 'A1g' in mode_character or 'A1' in mode_character:
                    # A1g: fully symmetric, diagonal tensor
                    tensor = np.diag([tensor[0,0], tensor[1,1], tensor[2,2]])
                    
                elif 'B1g' in mode_character or 'B1' in mode_character:
                    # B1g: xy component dominant, but keep small diagonal elements
                    main_val = base_tensor[0, 0] * 0.8
                    tensor = np.diag([main_val * 0.1, main_val * 0.1, main_val * 0.05])  # Small diagonal
                    tensor[0, 1] = tensor[1, 0] = main_val  # Main xy coupling
                    
                elif 'B2g' in mode_character or 'B2' in mode_character:
                    # B2g: x²-y² character with small z component
                    tensor = np.diag([base_tensor[0, 0], -base_tensor[1, 1], base_tensor[2, 2] * 0.1])
                    
                elif 'Eg' in mode_character or 'E' in mode_character:
                    # Eg: doubly degenerate, xz and yz components with small diagonal
                    main_val = base_tensor[0, 0] * 0.7
                    tensor = np.diag([main_val * 0.2, main_val * 0.2, main_val * 0.1])  # Small diagonal
                    tensor[0, 2] = tensor[2, 0] = main_val  # xz coupling
                    tensor[1, 2] = tensor[2, 1] = main_val * 0.8  # yz coupling
                    
            elif crystal_system.lower() == 'cubic':
                if 'A1g' in mode_character or 'T2g' in mode_character:
                    # Cubic symmetry: isotropic diagonal
                    avg_val = np.mean(np.diag(base_tensor))
                    tensor = np.diag([avg_val, avg_val, avg_val])
                    
            elif crystal_system.lower() == 'hexagonal':
                if 'A1g' in mode_character:
                    # A1g: zz component dominant
                    tensor = np.diag([tensor[0,0], tensor[1,1], tensor[2,2] * 1.5])
                elif 'E1g' in mode_character or 'E2g' in mode_character:
                    # Eg modes: xy plane components
                    tensor[2, 2] *= 0.3  # Reduce z component
                    
            elif crystal_system.lower() == 'orthorhombic':
                # All components allowed but with different weights
                if 'Ag' in mode_character:
                    # Fully symmetric
                    pass  # Keep base tensor
                elif 'B1g' in mode_character:
                    tensor[1, 1] *= 0.5
                    tensor[2, 2] *= 0.5
                elif 'B2g' in mode_character:
                    tensor[0, 0] *= 0.5
                    tensor[2, 2] *= 0.5
                elif 'B3g' in mode_character:
                    tensor[0, 0] *= 0.5
                    tensor[1, 1] *= 0.5
            
            # Add small frequency-dependent scaling
            freq_factor = 1.0 + 0.1 * np.sin(frequency / 200.0)
            tensor *= freq_factor
            
            # Ensure tensor has full rank (no zero eigenvalues)
            eigenvals = np.linalg.eigvals(tensor)
            min_eigenval = np.max(np.abs(eigenvals)) * 0.01  # 1% of largest eigenvalue
            
            # If any eigenvalue is too small, add small diagonal elements
            if np.min(np.abs(eigenvals)) < min_eigenval:
                tensor += np.eye(3) * min_eigenval
            
            return tensor
            
        except Exception as e:
            print(f"Warning: Could not generate tensor for mode {mode_character}: {e}")
            return base_tensor
    
    def extract_tensors_from_polarization(self):
        """Extract tensors from polarization data (fallback method)."""
        try:
            # Get available polarization configurations
            available_configs = list(self.polarization_data.keys())
            
            if len(available_configs) < 2:
                messagebox.showwarning("Insufficient Data", 
                                     "Need at least 2 polarization configurations for tensor analysis.")
                return
            
            # Get reference wavenumbers from first configuration
            ref_config = available_configs[0]
            wavenumbers = self.polarization_data[ref_config]['wavenumbers']
            
            # Initialize tensor storage
            self.calculated_raman_tensors = {
                'wavenumbers': wavenumbers,
                'tensor_matrices': {},
                'configurations': available_configs,
                'peak_tensors': {},
                'source': 'polarization',
                'analysis_complete': True
            }
            
            # Extract tensor elements from polarization data
            for config in available_configs:
                # Interpolate to common wavenumber grid
                intensities = np.interp(wavenumbers, 
                                      self.polarization_data[config]['wavenumbers'],
                                      self.polarization_data[config]['intensities'])
                
                self.calculated_raman_tensors['tensor_matrices'][config] = intensities
            
            # Construct full tensor matrices for each wavenumber
            self.construct_tensor_matrices()
            
            # Update peak selection dropdown
            self.update_tensor_peak_selection()
            
            # Transfer calculated_raman_tensors to raman_tensors for 3D visualization compatibility
            self.transfer_calculated_tensors_to_main()
            
            # Update status
            self.update_tensor_status()
            
            messagebox.showinfo("Success", 
                              f"Extracted tensor data from {len(available_configs)} polarization configurations.\n"
                              f"Data is now available for 3D visualization!\n"
                              f"Note: Database tensor data would be more accurate if available.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting tensors from polarization data: {str(e)}")
    
    def construct_tensor_matrices(self):
        """Construct 3x3 Raman tensor matrices from polarization data."""
        try:
            configs = self.calculated_raman_tensors['configurations']
            wavenumbers = self.calculated_raman_tensors['wavenumbers']
            
            # Initialize tensor matrices for each wavenumber
            n_points = len(wavenumbers)
            tensor_matrices = np.zeros((n_points, 3, 3))
            
            # Map polarization configurations to tensor elements
            config_mapping = {
                'xx': (0, 0), 'yy': (1, 1), 'zz': (2, 2),
                'xy': (0, 1), 'yx': (1, 0), 'xz': (0, 2), 
                'zx': (2, 0), 'yz': (1, 2), 'zy': (2, 1)
            }
            
            # Fill tensor matrices
            for config in configs:
                if config.lower() in config_mapping:
                    i, j = config_mapping[config.lower()]
                    tensor_matrices[:, i, j] = self.calculated_raman_tensors['tensor_matrices'][config]
                    
                    # Ensure symmetry for off-diagonal elements
                    if i != j:
                        tensor_matrices[:, j, i] = self.calculated_raman_tensors['tensor_matrices'][config]
            
            # Store the constructed matrices
            self.calculated_raman_tensors['full_tensors'] = tensor_matrices
            
            # Calculate derived properties
            self.calculate_tensor_properties()
            
        except Exception as e:
            print(f"Error constructing tensor matrices: {e}")
    
    def transfer_calculated_tensors_to_main(self):
        """Transfer calculated_raman_tensors data to raman_tensors for 3D visualization compatibility."""
        try:
            if not hasattr(self, 'calculated_raman_tensors') or not self.calculated_raman_tensors:
                print("No calculated_raman_tensors to transfer")
                return False
            
            # Clear existing raman_tensors
            self.raman_tensors = {}
            
            # Get data from calculated_raman_tensors
            calc_tensors = self.calculated_raman_tensors
            
            if 'wavenumbers' in calc_tensors and 'full_tensors' in calc_tensors:
                wavenumbers = calc_tensors['wavenumbers']
                tensors = calc_tensors['full_tensors']
                
                # PRIORITY: Use fitted peak positions if available
                if hasattr(self, 'fitted_peaks') and self.fitted_peaks:
                    print("🎯 Using fitted peak positions instead of database wavenumbers")
                    fitted_wavenumbers = []
                    fitted_tensors = []
                    
                    for peak in self.fitted_peaks:
                        # Extract peak position
                        if hasattr(peak, 'center'):
                            peak_wn = float(peak.center)
                        elif hasattr(peak, 'position'):
                            peak_wn = float(peak.position)
                        elif isinstance(peak, dict) and 'center' in peak:
                            peak_wn = float(peak['center'])
                        elif isinstance(peak, dict) and 'position' in peak:
                            peak_wn = float(peak['position'])
                        else:
                            continue
                        
                        # Find closest tensor from database
                        closest_idx = np.argmin(np.abs(wavenumbers - peak_wn))
                        closest_tensor = tensors[closest_idx]
                        
                        # Scale tensor by peak intensity if available
                        if hasattr(peak, 'amplitude'):
                            intensity_scale = float(peak.amplitude)
                        elif hasattr(peak, 'height'):
                            intensity_scale = float(peak.height)
                        elif isinstance(peak, dict) and 'amplitude' in peak:
                            intensity_scale = float(peak['amplitude'])
                        elif isinstance(peak, dict) and 'height' in peak:
                            intensity_scale = float(peak['height'])
                        else:
                            intensity_scale = 1.0
                        
                        # Scale tensor while preserving anisotropy
                        scaled_tensor = closest_tensor * (intensity_scale / np.max(np.abs(closest_tensor)))
                        
                        fitted_wavenumbers.append(peak_wn)
                        fitted_tensors.append(scaled_tensor)
                        
                        print(f"   📍 Peak at {peak_wn:.1f} cm⁻¹ (intensity: {intensity_scale:.2f})")
                    
                    if fitted_wavenumbers:
                        wavenumbers = np.array(fitted_wavenumbers)
                        tensors = np.array(fitted_tensors)
                        print(f"   ✅ Using {len(fitted_wavenumbers)} fitted peak positions")
                    else:
                        print("   ⚠️  No valid fitted peaks found, using database wavenumbers")
                
                # Convert to the format expected by 3D visualization
                self.raman_tensors = {
                    'wavenumbers': wavenumbers,
                    'tensors': tensors,
                    'source': calc_tensors.get('source', 'tensor_analysis'),
                    'mineral_name': calc_tensors.get('mineral_name', 'Unknown'),
                    'uses_fitted_peaks': hasattr(self, 'fitted_peaks') and self.fitted_peaks
                }
                
                # Also create individual peak entries for compatibility
                for i, (wn, tensor) in enumerate(zip(wavenumbers, tensors)):
                    peak_key = f"{wn:.1f}"
                    self.raman_tensors[peak_key] = {
                        'tensor': tensor,
                        'wavenumber': wn,
                        'intensity': np.trace(tensor) / 3.0,  # Average diagonal intensity
                        'eigenvalues': np.linalg.eigvals(tensor),
                        'source': 'tensor_analysis'
                    }
                
                print(f"✅ Successfully transferred {len(wavenumbers)} tensors to raman_tensors")
                print(f"   Source: {calc_tensors.get('source', 'unknown')}")
                print(f"   Wavenumber range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹")
                
                # Verify tensors are anisotropic
                sample_tensor = tensors[0] if len(tensors) > 0 else None
                if sample_tensor is not None:
                    eigenvals = np.linalg.eigvals(sample_tensor)
                    anisotropy = (np.max(eigenvals) - np.min(eigenvals)) / np.mean(eigenvals)
                    print(f"   Sample tensor anisotropy: {anisotropy:.3f}")
                    if anisotropy < 0.01:
                        print("   ⚠️  Warning: Tensors appear to be nearly isotropic")
                    else:
                        print("   ✅ Tensors are anisotropic - should show orientation dependence")
                
                return True
            else:
                print("❌ calculated_raman_tensors missing required data (wavenumbers/full_tensors)")
                return False
                
        except Exception as e:
            print(f"❌ Error transferring tensor data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_tensor_properties(self):
        """Calculate eigenvalues, eigenvectors, and other tensor properties."""
        try:
            tensors = self.calculated_raman_tensors['full_tensors']
            n_points = tensors.shape[0]
            
            # Initialize storage for properties
            eigenvalues = np.zeros((n_points, 3))
            eigenvectors = np.zeros((n_points, 3, 3))
            anisotropy = np.zeros(n_points)
            asymmetry = np.zeros(n_points)
            
            for i in range(n_points):
                # Calculate eigenvalues and eigenvectors
                evals, evecs = np.linalg.eigh(tensors[i])
                
                # Sort by eigenvalue magnitude
                idx = np.argsort(np.abs(evals))[::-1]
                eigenvalues[i] = evals[idx]
                eigenvectors[i] = evecs[:, idx]
                
                # Calculate anisotropy and asymmetry parameters
                if np.max(np.abs(evals)) > 1e-10:  # Avoid division by zero
                    sorted_evals = np.sort(np.abs(evals))[::-1]
                    anisotropy[i] = (sorted_evals[0] - sorted_evals[2]) / sorted_evals[0]
                    asymmetry[i] = (2 * sorted_evals[1] - sorted_evals[0] - sorted_evals[2]) / (sorted_evals[0] - sorted_evals[2])
            
            # Store calculated properties
            self.calculated_raman_tensors['eigenvalues'] = eigenvalues
            self.calculated_raman_tensors['eigenvectors'] = eigenvectors
            self.calculated_raman_tensors['anisotropy'] = anisotropy
            self.calculated_raman_tensors['asymmetry'] = asymmetry
            
        except Exception as e:
            print(f"Error calculating tensor properties: {e}")
    
    def calculate_tensor_properties_from_database(self):
        """Calculate tensor properties using high-quality database eigenvalues."""
        try:
            if 'database_eigenvalues' not in self.calculated_raman_tensors:
                return
            
            eigenval_data = self.calculated_raman_tensors['database_eigenvalues']
            frequencies = self.calculated_raman_tensors['wavenumbers']
            n_points = len(frequencies)
            
            # Use database eigenvalues directly (much more accurate)
            if 'n' in eigenval_data:
                n_eigenvals = eigenval_data['n']
                
                # Calculate anisotropy and asymmetry from refractive index eigenvalues
                anisotropy = np.zeros(n_points)
                asymmetry = np.zeros(n_points)
                
                # Sort eigenvalues
                sorted_evals = np.sort(n_eigenvals)[::-1]  # Descending order
                
                if sorted_evals[0] > 1e-10:  # Avoid division by zero
                    anisotropy_val = (sorted_evals[0] - sorted_evals[2]) / sorted_evals[0]
                    if (sorted_evals[0] - sorted_evals[2]) > 1e-10:
                        asymmetry_val = (2 * sorted_evals[1] - sorted_evals[0] - sorted_evals[2]) / (sorted_evals[0] - sorted_evals[2])
                    else:
                        asymmetry_val = 0
                else:
                    anisotropy_val = 0
                    asymmetry_val = 0
                
                # Replicate for all frequencies (database values are frequency-independent)
                anisotropy.fill(anisotropy_val)
                asymmetry.fill(asymmetry_val)
                
                self.calculated_raman_tensors['anisotropy'] = anisotropy
                self.calculated_raman_tensors['asymmetry'] = asymmetry
                
                # Store additional database-specific properties
                self.calculated_raman_tensors['birefringence'] = sorted_evals[0] - sorted_evals[2]
                self.calculated_raman_tensors['average_refractive_index'] = np.mean(n_eigenvals)
            
            # Calculate properties from dielectric eigenvalues if available
            if 'eps_inf' in eigenval_data:
                eps_inf_eigenvals = eigenval_data['eps_inf']
                self.calculated_raman_tensors['dielectric_anisotropy'] = np.max(eps_inf_eigenvals) - np.min(eps_inf_eigenvals)
            
            # Mark as using high-quality database data
            self.calculated_raman_tensors['high_quality_eigenvalues'] = True
            
        except Exception as e:
            print(f"Error calculating tensor properties from database: {e}")
    
    def calculate_tensors_from_structure(self):
        """Calculate theoretical Raman tensors from crystal structure."""
        if not hasattr(self, 'crystal_structure') or not self.crystal_structure:
            messagebox.showwarning("No Structure", "Please load crystal structure data in the Crystal Structure tab first.")
            return
        
        try:
            # This is a simplified implementation
            # In practice, you would use group theory and phonon calculations
            messagebox.showinfo("Structure Tensors", 
                              "Theoretical tensor calculation from structure requires advanced phonon calculations.\n"
                              "This feature will use symmetry-based approximations.")
            
            # Get crystal system for symmetry constraints
            crystal_system = getattr(self, 'crystal_system_var', tk.StringVar()).get()
            
            if crystal_system == "Unknown":
                crystal_system = self.get_crystal_system_from_structure()
            
            # Generate symmetry-constrained tensors
            self.generate_symmetry_tensors(crystal_system)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating tensors from structure: {str(e)}")
    
    def generate_symmetry_tensors(self, crystal_system):
        """Generate symmetry-constrained Raman tensors."""
        try:
            # Define symmetry constraints for different crystal systems
            if crystal_system == "Cubic":
                # Cubic: only diagonal elements, all equal
                tensor_template = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            elif crystal_system == "Tetragonal":
                # Tetragonal: diagonal, xx=yy≠zz
                tensor_template = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.8]])
            elif crystal_system == "Orthorhombic":
                # Orthorhombic: diagonal, all different
                tensor_template = np.array([[1, 0, 0], [0, 0.8, 0], [0, 0, 0.6]])
            elif crystal_system == "Hexagonal":
                # Hexagonal: xx=yy≠zz, some off-diagonal
                tensor_template = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.7]])
            else:
                # General case: allow all elements
                tensor_template = np.array([[1, 0.2, 0.1], [0.2, 0.8, 0.15], [0.1, 0.15, 0.6]])
            
            # Create wavenumber grid if not available
            if not hasattr(self, 'calculated_raman_tensors') or not self.calculated_raman_tensors:
                wavenumbers = np.linspace(200, 1800, 100)
                self.calculated_raman_tensors = {
                    'wavenumbers': wavenumbers,
                    'configurations': ['theoretical'],
                    'analysis_complete': True
                }
            else:
                wavenumbers = self.calculated_raman_tensors['wavenumbers']
            
            # Generate tensors with frequency dependence
            n_points = len(wavenumbers)
            tensor_matrices = np.zeros((n_points, 3, 3))
            
            for i in range(n_points):
                # Add some frequency dependence
                freq_factor = 1.0 + 0.1 * np.sin(wavenumbers[i] / 200.0)
                tensor_matrices[i] = tensor_template * freq_factor
            
            self.calculated_raman_tensors['full_tensors'] = tensor_matrices
            self.calculated_raman_tensors['source'] = 'structure'
            
            # Calculate properties
            self.calculate_tensor_properties()
            self.update_tensor_peak_selection()
            self.update_tensor_status()
            
            #messagebox.showinfo("Success", f"Generated symmetry-constrained tensors for {crystal_system} crystal system.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating symmetry tensors: {str(e)}")
    
    def import_tensor_data(self):
        """Import external tensor data from file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Import Tensor Data",
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            # Try to parse the file
            data = np.loadtxt(file_path, delimiter=',')
            
            if data.shape[1] < 10:  # Need at least wavenumber + 9 tensor elements
                messagebox.showerror("Invalid Format", 
                                   "File must contain wavenumber and 9 tensor elements per row.")
                return
            
            wavenumbers = data[:, 0]
            tensor_elements = data[:, 1:10]  # 9 elements of 3x3 matrix
            
            # Reshape into tensor matrices
            n_points = len(wavenumbers)
            tensor_matrices = tensor_elements.reshape(n_points, 3, 3)
            
            # Store imported data
            self.calculated_raman_tensors = {
                'wavenumbers': wavenumbers,
                'full_tensors': tensor_matrices,
                'configurations': ['imported'],
                'source': 'external',
                'analysis_complete': True
            }
            
            # Calculate properties
            self.calculate_tensor_properties()
            self.update_tensor_peak_selection()
            self.update_tensor_status()
            
            #messagebox.showinfo("Success", f"Imported tensor data with {n_points} data points.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error importing tensor data: {str(e)}")
    
    def analyze_tensor_symmetry(self):
        """Analyze tensor symmetry properties."""
        if not self.calculated_raman_tensors or 'full_tensors' not in self.calculated_raman_tensors:
            messagebox.showwarning("No Data", "Please extract or import tensor data first.")
            return
        
        try:
            # Create results window
            results_window = tk.Toplevel(self.root)
            results_window.title("Tensor Symmetry Analysis")
            results_window.geometry("700x600")
            
            # Create text widget with scrollbar
            text_frame = ttk.Frame(results_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Analyze symmetry
            tensors = self.calculated_raman_tensors['full_tensors']
            wavenumbers = self.calculated_raman_tensors['wavenumbers']
            
            report = "RAMAN TENSOR SYMMETRY ANALYSIS\n"
            report += "=" * 50 + "\n\n"
            
            # Analyze overall symmetry properties
            n_points = tensors.shape[0]
            symmetric_count = 0
            antisymmetric_count = 0
            
            for i in range(n_points):
                tensor = tensors[i]
                
                # Check symmetry
                is_symmetric = np.allclose(tensor, tensor.T, rtol=1e-3)
                is_antisymmetric = np.allclose(tensor, -tensor.T, rtol=1e-3)
                
                if is_symmetric:
                    symmetric_count += 1
                elif is_antisymmetric:
                    antisymmetric_count += 1
            
            report += f"Symmetry Statistics:\n"
            report += f"  Symmetric tensors: {symmetric_count}/{n_points} ({100*symmetric_count/n_points:.1f}%)\n"
            report += f"  Antisymmetric tensors: {antisymmetric_count}/{n_points} ({100*antisymmetric_count/n_points:.1f}%)\n"
            report += f"  General tensors: {n_points-symmetric_count-antisymmetric_count}/{n_points}\n\n"
            
            # Analyze specific frequencies if peaks are available
            if hasattr(self, 'fitted_peaks') and self.fitted_peaks:
                report += "PEAK-SPECIFIC ANALYSIS:\n"
                report += "-" * 30 + "\n"
                
                for peak in self.fitted_peaks:
                    freq = peak.get('center', 0)
                    # Find closest wavenumber
                    idx = np.argmin(np.abs(wavenumbers - freq))
                    tensor = tensors[idx]
                    
                    report += f"\nPeak at {freq:.1f} cm⁻¹:\n"
                    report += f"  Tensor matrix:\n"
                    for row in tensor:
                        report += f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]\n"
                    
                    # Calculate invariants
                    trace = np.trace(tensor)
                    det = np.linalg.det(tensor)
                    
                    report += f"  Trace: {trace:.4f}\n"
                    report += f"  Determinant: {det:.4f}\n"
                    
                    # Symmetry classification
                    if np.allclose(tensor, tensor.T, rtol=1e-3):
                        report += f"  Symmetry: Symmetric\n"
                    elif np.allclose(tensor, -tensor.T, rtol=1e-3):
                        report += f"  Symmetry: Antisymmetric\n"
                    else:
                        report += f"  Symmetry: General\n"
            
            # Add theoretical background
            report += "\n\nTHEORETICAL BACKGROUND:\n"
            report += "-" * 30 + "\n"
            report += "Raman tensors describe the linear relationship between\n"
            report += "the induced dipole moment and the electric field:\n"
            report += "  μᵢ = αᵢⱼ Eⱼ\n\n"
            report += "For Raman scattering, the tensor elements determine\n"
            report += "the polarization dependence of scattered light.\n\n"
            report += "Symmetry properties:\n"
            report += "• Symmetric: αᵢⱼ = αⱼᵢ (most common for Raman)\n"
            report += "• Antisymmetric: αᵢⱼ = -αⱼᵢ (rare, magnetic effects)\n"
            report += "• General: no specific symmetry constraints\n"
            
            text_widget.insert(tk.END, report)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in symmetry analysis: {str(e)}")
    
    def calculate_principal_components(self):
        """Calculate and display eigenvalues and eigenvectors."""
        if not self.calculated_raman_tensors or 'eigenvalues' not in self.calculated_raman_tensors:
            messagebox.showwarning("No Data", "Please extract tensor data first.")
            return
        
        try:
            # Create results window
            results_window = tk.Toplevel(self.root)
            results_window.title("Principal Component Analysis")
            results_window.geometry("800x700")
            
            # Create notebook for different views
            notebook = ttk.Notebook(results_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Eigenvalues tab
            eigenval_frame = ttk.Frame(notebook)
            notebook.add(eigenval_frame, text="Eigenvalues")
            
            # Create matplotlib plot for eigenvalues
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            wavenumbers = self.calculated_raman_tensors['wavenumbers']
            eigenvalues = self.calculated_raman_tensors['eigenvalues']
            
            # Plot eigenvalues vs frequency
            ax1.plot(wavenumbers, eigenvalues[:, 0], 'r-', label='λ₁ (largest)', linewidth=2)
            ax1.plot(wavenumbers, eigenvalues[:, 1], 'g-', label='λ₂ (middle)', linewidth=2)
            ax1.plot(wavenumbers, eigenvalues[:, 2], 'b-', label='λ₃ (smallest)', linewidth=2)
            ax1.set_xlabel('Wavenumber (cm⁻¹)')
            ax1.set_ylabel('Eigenvalue')
            ax1.set_title('Tensor Eigenvalues vs Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot anisotropy and asymmetry
            anisotropy = self.calculated_raman_tensors['anisotropy']
            asymmetry = self.calculated_raman_tensors['asymmetry']
            
            ax2.plot(wavenumbers, anisotropy, 'purple', label='Anisotropy', linewidth=2)
            ax2.plot(wavenumbers, asymmetry, 'orange', label='Asymmetry', linewidth=2)
            ax2.set_xlabel('Wavenumber (cm⁻¹)')
            ax2.set_ylabel('Parameter Value')
            ax2.set_title('Anisotropy and Asymmetry Parameters')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Embed plot in tkinter
            canvas = FigureCanvasTkAgg(fig, eigenval_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Eigenvectors tab
            eigenvec_frame = ttk.Frame(notebook)
            notebook.add(eigenvec_frame, text="Eigenvectors")
            
            # Create text widget for eigenvector data
            text_frame = ttk.Frame(eigenvec_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 9))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Generate eigenvector report
            eigenvectors = self.calculated_raman_tensors['eigenvectors']
            
            report = "PRINCIPAL COMPONENT ANALYSIS RESULTS\n"
            report += "=" * 50 + "\n\n"
            
            # Sample some frequencies for detailed analysis
            sample_indices = np.linspace(0, len(wavenumbers)-1, min(10, len(wavenumbers)), dtype=int)
            
            for idx in sample_indices:
                freq = wavenumbers[idx]
                evals = eigenvalues[idx]
                evecs = eigenvectors[idx]
                
                report += f"Frequency: {freq:.1f} cm⁻¹\n"
                report += f"Eigenvalues: [{evals[0]:8.4f}, {evals[1]:8.4f}, {evals[2]:8.4f}]\n"
                report += f"Eigenvectors:\n"
                for i in range(3):
                    report += f"  v{i+1}: [{evecs[0,i]:7.4f}, {evecs[1,i]:7.4f}, {evecs[2,i]:7.4f}]\n"
                
                # Calculate orientation angles
                v1 = evecs[:, 0]  # Principal eigenvector
                theta = np.arccos(np.abs(v1[2])) * 180 / np.pi  # Angle with z-axis
                phi = np.arctan2(v1[1], v1[0]) * 180 / np.pi    # Azimuthal angle
                
                report += f"  Principal axis orientation: θ={theta:.1f}°, φ={phi:.1f}°\n"
                report += f"  Anisotropy: {anisotropy[idx]:.4f}\n"
                report += f"  Asymmetry: {asymmetry[idx]:.4f}\n\n"
            
            text_widget.insert(tk.END, report)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in principal component analysis: {str(e)}")
    
    def calculate_orientation_dependence(self):
        """Calculate orientation dependence of Raman intensities."""
        if not self.calculated_raman_tensors or 'full_tensors' not in self.calculated_raman_tensors:
            messagebox.showwarning("No Data", "Please extract tensor data first.")
            return
        
        try:
            # Create results window
            results_window = tk.Toplevel(self.root)
            results_window.title("Orientation Dependence Analysis")
            results_window.geometry("900x700")
            
            # Create matplotlib figure
            fig = plt.figure(figsize=(12, 8))
            
            # Get tensor data
            tensors = self.calculated_raman_tensors['full_tensors']
            wavenumbers = self.calculated_raman_tensors['wavenumbers']
            
            # Select a representative frequency for detailed analysis
            mid_idx = len(wavenumbers) // 2
            representative_tensor = tensors[mid_idx]
            representative_freq = wavenumbers[mid_idx]
            
            # Calculate orientation dependence
            angles = np.linspace(0, 2*np.pi, 100)
            
            # For different scattering geometries
            geometries = {
                'Backscattering (z-axis)': {'incident': [0, 0, 1], 'scattered': [0, 0, -1]},
                'Right-angle (x-y plane)': {'incident': [1, 0, 0], 'scattered': [0, 1, 0]},
                'Forward scattering': {'incident': [0, 0, 1], 'scattered': [0, 0, 1]}
            }
            
            # Plot orientation dependence for each geometry
            for i, (geom_name, geom) in enumerate(geometries.items()):
                ax = fig.add_subplot(2, 3, i+1, projection='polar')
                
                intensities = []
                for angle in angles:
                    # Rotate polarization vector
                    pol_in = [np.cos(angle), np.sin(angle), 0]
                    pol_out = [np.cos(angle), np.sin(angle), 0]
                    
                    # Calculate Raman intensity
                    intensity = self.calculate_raman_intensity(representative_tensor, pol_in, pol_out)
                    intensities.append(intensity)
                
                ax.plot(angles, intensities, linewidth=2)
                ax.set_title(f'{geom_name}\n{representative_freq:.1f} cm⁻¹')
                ax.grid(True)
            
            # Plot frequency dependence of maximum intensity
            ax4 = fig.add_subplot(2, 3, 4)
            max_intensities = []
            
            for i in range(len(wavenumbers)):
                tensor = tensors[i]
                # Calculate maximum possible intensity
                eigenvals = np.linalg.eigvals(tensor @ tensor.T)
                max_intensities.append(np.max(eigenvals))
            
            ax4.plot(wavenumbers, max_intensities, 'b-', linewidth=2)
            ax4.set_xlabel('Wavenumber (cm⁻¹)')
            ax4.set_ylabel('Maximum Intensity')
            ax4.set_title('Frequency Dependence')
            ax4.grid(True, alpha=0.3)
            
            # Plot depolarization ratio vs frequency
            ax5 = fig.add_subplot(2, 3, 5)
            depol_ratios = []
            
            for i in range(len(wavenumbers)):
                tensor = tensors[i]
                # Calculate depolarization ratio for backscattering
                I_parallel = self.calculate_raman_intensity(tensor, [1, 0, 0], [1, 0, 0])
                I_perpendicular = self.calculate_raman_intensity(tensor, [1, 0, 0], [0, 1, 0])
                
                if I_parallel > 1e-10:
                    depol_ratio = I_perpendicular / I_parallel
                else:
                    depol_ratio = 0
                
                depol_ratios.append(depol_ratio)
            
            ax5.plot(wavenumbers, depol_ratios, 'r-', linewidth=2)
            ax5.axhline(y=0.75, color='gray', linestyle='--', alpha=0.7, label='Fully depolarized')
            ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Fully polarized')
            ax5.set_xlabel('Wavenumber (cm⁻¹)')
            ax5.set_ylabel('Depolarization Ratio')
            ax5.set_title('Depolarization vs Frequency')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 3D visualization of tensor ellipsoid
            ax6 = fig.add_subplot(2, 3, 6, projection='3d')
            self.plot_tensor_ellipsoid(ax6, representative_tensor)
            ax6.set_title(f'Tensor Ellipsoid\n{representative_freq:.1f} cm⁻¹')
            
            plt.tight_layout()
            
            # Embed plot in tkinter
            canvas = FigureCanvasTkAgg(fig, results_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in orientation dependence analysis: {str(e)}")
    
    def calculate_raman_intensity(self, tensor, pol_in, pol_out):
        """Calculate Raman intensity for given polarizations."""
        pol_in = np.array(pol_in)
        pol_out = np.array(pol_out)
        
        # Raman intensity ∝ |pol_out · tensor · pol_in|²
        intensity = np.abs(pol_out @ tensor @ pol_in) ** 2
        return intensity
    
    def plot_tensor_ellipsoid(self, ax, tensor):
        """Plot 3D ellipsoid representation of tensor."""
        try:
            # Calculate eigenvalues and eigenvectors of the tensor itself (not tensor @ tensor.T)
            eigenvals, eigenvecs = np.linalg.eigh(tensor)
            
            # Sort eigenvalues and eigenvectors by magnitude
            sorted_indices = np.argsort(np.abs(eigenvals))[::-1]
            eigenvals = eigenvals[sorted_indices]
            eigenvecs = eigenvecs[:, sorted_indices]
            
            # Ensure all eigenvalues are positive for ellipsoid plotting
            eigenvals = np.abs(eigenvals)
            
            # Avoid zero eigenvalues - use relative minimum based on largest eigenvalue
            max_eigenval = np.max(eigenvals)
            min_eigenval = max(max_eigenval * 0.01, 1e-6)  # At least 1% of max or 1e-6
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # Create ellipsoid surface
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            
            # Ellipsoid in standard form (using eigenvalues as semi-axes)
            a, b, c = np.sqrt(eigenvals)
            x = a * np.outer(np.cos(u), np.sin(v))
            y = b * np.outer(np.sin(u), np.sin(v))
            z = c * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Rotate by eigenvectors
            for i in range(len(x)):
                for j in range(len(x[0])):
                    point = np.array([x[i,j], y[i,j], z[i,j]])
                    rotated = eigenvecs @ point
                    x[i,j], y[i,j], z[i,j] = rotated
            
            # Plot ellipsoid surface
            ax.plot_surface(x, y, z, alpha=0.6, cmap='viridis')
            
            # Plot principal axes as arrows
            max_scale = np.max(np.sqrt(eigenvals))
            colors = ['red', 'green', 'blue']
            labels = ['λ₁', 'λ₂', 'λ₃']
            
            for i in range(3):
                axis = eigenvecs[:, i] * np.sqrt(eigenvals[i])
                ax.quiver(0, 0, 0, axis[0], axis[1], axis[2], 
                         color=colors[i], arrow_length_ratio=0.1, linewidth=2,
                         label=f'{labels[i]} = {eigenvals[i]:.3f}')
            
            # Set equal aspect ratio and labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set equal scaling
            max_range = max_scale * 1.1
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            
            # Add legend
            ax.legend()
            
        except Exception as e:
            print(f"Error plotting tensor ellipsoid: {e}")
            # Fallback: show error message on plot
            try:
                ax.clear()
                ax.text(0.5, 0.5, 0.5, f'Error plotting ellipsoid:\n{str(e)}', 
                       ha='center', va='center')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            except:
                # If even the fallback fails, just print the error
                print(f"Could not display error message on plot: {e}")
    
    def analyze_peak_tensors(self):
        """Analyze tensors for individual peaks."""
        if not self.calculated_raman_tensors or 'full_tensors' not in self.calculated_raman_tensors:
            messagebox.showwarning("No Data", "Please extract tensor data first.")
            return
        
        if not hasattr(self, 'fitted_peaks') or not self.fitted_peaks:
            messagebox.showwarning("No Peaks", "Please fit peaks in the Peak Fitting tab first.")
            return
        
        try:
            # Create results window
            results_window = tk.Toplevel(self.root)
            results_window.title("Peak-by-Peak Tensor Analysis")
            results_window.geometry("1000x800")
            
            # Create notebook for different peaks
            notebook = ttk.Notebook(results_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            tensors = self.calculated_raman_tensors['full_tensors']
            wavenumbers = self.calculated_raman_tensors['wavenumbers']
            
            # Analyze each fitted peak
            for i, peak in enumerate(self.fitted_peaks):
                # Try both 'center' and 'position' keys for compatibility
                peak_freq = peak.get('center', peak.get('position', 0))
                peak_name = f"Peak {i+1} ({peak_freq:.1f} cm⁻¹)"
                
                # Find closest tensor data point or create symmetry-constrained tensor
                idx = np.argmin(np.abs(wavenumbers - peak_freq))
                if idx < len(tensors):
                    peak_tensor = tensors[idx]
                else:
                    # Create symmetry-constrained tensor for this peak
                    peak_intensity = peak.get('amplitude', peak.get('height', peak.get('intensity', 1.0)))
                    peak_tensor = self.create_symmetry_constrained_tensor(peak_freq, peak_intensity)
                
                # Create tab for this peak
                peak_frame = ttk.Frame(notebook)
                notebook.add(peak_frame, text=peak_name)
                
                # Create matplotlib figure for this peak
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                
                # 1. Tensor matrix visualization
                im = ax1.imshow(peak_tensor, cmap='RdBu_r', vmin=-np.max(np.abs(peak_tensor)), 
                               vmax=np.max(np.abs(peak_tensor)))
                ax1.set_title(f'Tensor Matrix\n{peak_freq:.1f} cm⁻¹')
                ax1.set_xlabel('j')
                ax1.set_ylabel('i')
                
                # Add text annotations
                for i_idx in range(3):
                    for j_idx in range(3):
                        ax1.text(j_idx, i_idx, f'{peak_tensor[i_idx, j_idx]:.3f}', 
                                ha='center', va='center', fontsize=10)
                
                # Add colorbar for this subplot
                plt.colorbar(im, ax=ax1)
                
                # 2. Eigenvalue analysis - use full 3D tensor
                eigenvals, eigenvecs = np.linalg.eigh(peak_tensor)
                # Sort eigenvalues by magnitude (largest first)
                sorted_indices = np.argsort(np.abs(eigenvals))[::-1]
                eigenvals = eigenvals[sorted_indices]
                eigenvecs = eigenvecs[:, sorted_indices]
                
                # Ensure we have exactly 3 eigenvalues
                if len(eigenvals) == 3:
                    ax2.bar(range(3), eigenvals, color=['red', 'green', 'blue'])
                    ax2.set_title('Eigenvalues')
                    ax2.set_xlabel('Principal Component')
                    ax2.set_ylabel('Eigenvalue')
                    ax2.set_xticks(range(3))
                    ax2.set_xticklabels(['λ₁', 'λ₂', 'λ₃'])
                    
                    # Add eigenvalue labels on bars
                    for i, val in enumerate(eigenvals):
                        ax2.text(i, val + 0.05 * max(abs(eigenvals)), f'{val:.3f}', 
                                ha='center', va='bottom', fontsize=9)
                else:
                    ax2.text(0.5, 0.5, f'Error: Found {len(eigenvals)} eigenvalues\ninstead of 3', 
                            ha='center', va='center', transform=ax2.transAxes)
                
                # 3. Polar plot of intensity vs polarization angle
                angles = np.linspace(0, 2*np.pi, 100)
                intensities = []
                
                for angle in angles:
                    pol = [np.cos(angle), np.sin(angle), 0]
                    intensity = self.calculate_raman_intensity(peak_tensor, pol, pol)
                    intensities.append(intensity)
                
                ax3 = plt.subplot(2, 2, 3, projection='polar')
                ax3.plot(angles, intensities, linewidth=2)
                ax3.set_title('Intensity vs Polarization')
                
                # 4. 3D tensor ellipsoid
                ax4 = plt.subplot(2, 2, 4, projection='3d')
                self.plot_tensor_ellipsoid(ax4, peak_tensor)
                ax4.set_title('Tensor Ellipsoid')
                
                plt.tight_layout()
                
                # Embed plot in tkinter
                canvas = FigureCanvasTkAgg(fig, peak_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Store peak tensor data
                if 'peak_tensors' not in self.calculated_raman_tensors:
                    self.calculated_raman_tensors['peak_tensors'] = {}
                
                self.calculated_raman_tensors['peak_tensors'][peak_freq] = {
                    'tensor': peak_tensor,
                    'eigenvalues': eigenvals,
                    'eigenvectors': eigenvecs,
                    'frequency': peak_freq
                }
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in peak tensor analysis: {str(e)}")
    
    def update_tensor_plot(self, event=None):
        """Update the main tensor visualization plot."""
        if "Tensor Analysis & Visualization" not in self.plot_components:
            return
        
        if not self.calculated_raman_tensors or 'full_tensors' not in self.calculated_raman_tensors:
            return
        
        try:
            components = self.plot_components["Tensor Analysis & Visualization"]
            ax = components['ax']
            canvas = components['canvas']
            
            # Check if ax is valid
            if ax is None or not hasattr(ax, 'clear'):
                print("Warning: Invalid axis object")
                return
            
            # Clear the plot
            ax.clear()
            
            display_mode = self.tensor_display_var.get()
            selected_peak = self.tensor_peak_var.get()
            
            tensors = self.calculated_raman_tensors['full_tensors']
            wavenumbers = self.calculated_raman_tensors['wavenumbers']
            
            if display_mode == "Tensor Matrix":
                self.plot_tensor_matrix(ax, tensors, wavenumbers, selected_peak)
            elif display_mode == "3D Tensor Ellipsoid":
                self.plot_3d_tensor_ellipsoid(ax, tensors, wavenumbers, selected_peak)
            elif display_mode == "Principal Axes":
                self.plot_principal_axes(ax, tensors, wavenumbers)
            elif display_mode == "Symmetry Visualization":
                self.plot_symmetry_visualization(ax, tensors, wavenumbers)
            elif display_mode == "Orientation Map":
                self.plot_orientation_map(ax, tensors, wavenumbers)
            elif display_mode == "Peak Comparison":
                self.plot_peak_comparison(ax, tensors, wavenumbers)
            
            # Check if canvas is valid before drawing
            if canvas is not None and hasattr(canvas, 'draw'):
                canvas.draw()
            
        except Exception as e:
            print(f"Error updating tensor plot: {e}")
            # Try to show error on plot if possible
            try:
                if ax is not None and hasattr(ax, 'text'):
                    ax.clear()
                    ax.text(0.5, 0.5, f'Error updating plot:\n{str(e)}', 
                           ha='center', va='center', transform=ax.transAxes)
            except:
                pass
    
    def plot_tensor_matrix(self, ax, tensors, wavenumbers, selected_peak):
        """Plot tensor matrix elements with symmetry constraints highlighted."""
        # Get crystal system for symmetry constraints
        crystal_system = "Cubic"  # Default
        point_group = "m-3m"     # Default
        
        if hasattr(self, 'crystal_structure') and self.crystal_structure:
            crystal_system = self.crystal_structure.get('crystal_system', 'Cubic')
            point_group = self.crystal_structure.get('point_group', 'm-3m')
        elif hasattr(self, 'tensor_analysis_results') and self.tensor_analysis_results:
            crystal_system = self.tensor_analysis_results.get('crystal_system', 'Cubic')
            point_group = self.tensor_analysis_results.get('point_group', 'm-3m')
        
        if selected_peak == "All Peaks" or selected_peak == "":
            # Plot average tensor
            tensor = np.mean(tensors, axis=0)
            title = f'Average Tensor Matrix\n{crystal_system} ({point_group})'
        else:
            # Plot specific peak tensor
            try:
                peak_freq = float(selected_peak.split()[0])
                idx = np.argmin(np.abs(wavenumbers - peak_freq))
                tensor = tensors[idx]
                title = f'Tensor Matrix at {peak_freq:.1f} cm⁻¹\n{crystal_system} ({point_group})'
            except:
                tensor = np.mean(tensors, axis=0)
                title = f'Average Tensor Matrix\n{crystal_system} ({point_group})'
        
        # Create the matrix plot
        vmax = np.max(np.abs(tensor))
        im = ax.imshow(tensor, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        
        # Add value annotations
        for i in range(3):
            for j in range(3):
                value = tensor[i, j]
                # Color text based on whether element should be zero for this crystal system
                should_be_zero = self.should_tensor_element_be_zero(i, j, crystal_system)
                
                if should_be_zero and abs(value) > 0.01 * vmax:
                    # Forbidden element with significant value - highlight in red
                    text_color = 'red'
                    weight = 'bold'
                elif should_be_zero:
                    # Forbidden element with small value - gray
                    text_color = 'gray'
                    weight = 'normal'
                else:
                    # Allowed element - black or white depending on background
                    text_color = 'white' if abs(value) > 0.5 * vmax else 'black'
                    weight = 'normal'
                
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                       color=text_color, fontsize=10, weight=weight)
        
        # Add symmetry constraint annotations
        ax.set_xlabel('j (column)')
        ax.set_ylabel('i (row)')
        
        # Add grid to separate elements
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", size=0)
        
        # Set major ticks
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['x', 'y', 'z'])
        ax.set_yticklabels(['x', 'y', 'z'])
        
        # Check if colorbar already exists and remove it
        if hasattr(ax, '_colorbar') and ax._colorbar is not None:
            try:
                ax._colorbar.remove()
            except:
                pass
        
        # Add new colorbar and store reference
        try:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Tensor Element Value')
            ax._colorbar = cbar
        except Exception as e:
            print(f"Warning: Could not add colorbar: {e}")
        
        # Add legend for color coding
        legend_text = f"Crystal System: {crystal_system}\n"
        legend_text += "Red: Forbidden by symmetry\n"
        legend_text += "Gray: Should be zero\n"
        legend_text += "Black/White: Allowed"
        
        ax.text(1.15, 0.5, legend_text, transform=ax.transAxes, 
                verticalalignment='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    def plot_3d_tensor_ellipsoid(self, ax, tensors, wavenumbers, selected_peak):
        """Plot 3D tensor ellipsoid."""
        try:
            # Check if this is a 3D axis
            if not hasattr(ax, 'zaxis'):
                # Clear the axis and show a message that 3D plotting requires a 3D axis
                ax.clear()
                ax.text(0.5, 0.5, '3D Tensor Ellipsoid\n\nRequires 3D axis.\nPlease use Peak-by-Peak\nAnalysis for 3D view.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                ax.set_title('3D Tensor Ellipsoid (Use Peak-by-Peak Analysis)')
                return
            
            if selected_peak == "All Peaks" or selected_peak == "":
                tensor = np.mean(tensors, axis=0)
                title = 'Average Tensor Ellipsoid'
            else:
                try:
                    peak_freq = float(selected_peak.split()[0])
                    idx = np.argmin(np.abs(wavenumbers - peak_freq))
                    tensor = tensors[idx]
                    title = f'Tensor Ellipsoid at {peak_freq:.1f} cm⁻¹'
                except:
                    tensor = np.mean(tensors, axis=0)
                    title = 'Average Tensor Ellipsoid'
            
            # Use the existing 3D ellipsoid plotting method
            self.plot_tensor_ellipsoid(ax, tensor)
            ax.set_title(title)
            
        except Exception as e:
            print(f"Error plotting 3D tensor ellipsoid: {e}")
            ax.clear()
            ax.text(0.5, 0.5, f'Error plotting 3D ellipsoid:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_principal_axes(self, ax, tensors, wavenumbers):
        """Plot principal axes evolution."""
        if 'eigenvalues' not in self.calculated_raman_tensors:
            ax.text(0.5, 0.5, 'No eigenvalue data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        eigenvalues = self.calculated_raman_tensors['eigenvalues']
        
        ax.plot(wavenumbers, eigenvalues[:, 0], 'r-', label='λ₁ (largest)', linewidth=2)
        ax.plot(wavenumbers, eigenvalues[:, 1], 'g-', label='λ₂ (middle)', linewidth=2)
        ax.plot(wavenumbers, eigenvalues[:, 2], 'b-', label='λ₃ (smallest)', linewidth=2)
        
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Principal Axes (Eigenvalues) vs Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set proper y-axis limits with some padding
        y_min = np.min(eigenvalues) * 0.95
        y_max = np.max(eigenvalues) * 1.05
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)
    
    def plot_symmetry_visualization(self, ax, tensors, wavenumbers):
        """Plot point group specific symmetry properties."""
        # Get crystal system and point group
        crystal_system = "Cubic"  # Default
        point_group = "m-3m"     # Default
        
        if hasattr(self, 'crystal_structure') and self.crystal_structure:
            crystal_system = self.crystal_structure.get('crystal_system', 'Cubic')
            point_group = self.crystal_structure.get('point_group', 'm-3m')
        elif hasattr(self, 'tensor_analysis_results') and self.tensor_analysis_results:
            crystal_system = self.tensor_analysis_results.get('crystal_system', 'Cubic')
            point_group = self.tensor_analysis_results.get('point_group', 'm-3m')
        
        # Calculate point group specific symmetry violations
        symmetry_violations = []
        constraint_violations = []
        
        for tensor in tensors:
            # General symmetry: how close to symmetric (Raman tensors should be symmetric)
            sym_violation = np.linalg.norm(tensor - tensor.T) / np.linalg.norm(tensor)
            symmetry_violations.append(sym_violation)
            
            # Point group specific constraint violations
            constraint_violation = self.calculate_point_group_violation(tensor, crystal_system)
            constraint_violations.append(constraint_violation)
        
        # Plot symmetry measures
        ax.plot(wavenumbers, symmetry_violations, 'r-', label='Symmetry Violation', linewidth=2, marker='o', markersize=4)
        ax.plot(wavenumbers, constraint_violations, 'b-', label=f'{crystal_system} Constraint Violation', linewidth=2, marker='s', markersize=4)
        
        # Add reference lines for perfect symmetry
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Perfect Symmetry')
        ax.axhline(y=0.1, color='orange', linestyle=':', alpha=0.7, label='10% Tolerance')
        
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Symmetry Violation')
        ax.set_title(f'Point Group Symmetry Analysis\n{crystal_system} ({point_group})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set proper y-axis limits
        all_measures = symmetry_violations + constraint_violations
        if all_measures:
            y_max = max(0.2, max(all_measures) * 1.1)  # At least 0.2 to show reference lines
            ax.set_ylim(-0.02, y_max)
        
        # Add text annotation about crystal system
        ax.text(0.02, 0.98, f'Crystal System: {crystal_system}\nPoint Group: {point_group}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    def plot_orientation_map(self, ax, tensors, wavenumbers):
        """Plot orientation dependence map."""
        # Calculate depolarization ratios
        depol_ratios = []
        max_intensities = []
        
        for tensor in tensors:
            # Backscattering depolarization ratio
            I_parallel = self.calculate_raman_intensity(tensor, [1, 0, 0], [1, 0, 0])
            I_perpendicular = self.calculate_raman_intensity(tensor, [1, 0, 0], [0, 1, 0])
            
            if I_parallel > 1e-10:
                depol_ratio = I_perpendicular / I_parallel
            else:
                depol_ratio = 0
            
            depol_ratios.append(depol_ratio)
            
            # Maximum intensity
            eigenvals = np.linalg.eigvals(tensor @ tensor.T)
            max_intensities.append(np.max(eigenvals))
        
        # Normalize max intensities
        max_intensities = np.array(max_intensities)
        if np.max(max_intensities) > 0:
            max_intensities = max_intensities / np.max(max_intensities)
        
        ax.plot(wavenumbers, depol_ratios, 'r-', label='Depolarization Ratio', linewidth=2)
        ax.plot(wavenumbers, max_intensities, 'b-', label='Normalized Max Intensity', linewidth=2)
        
        ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.7, label='Fully depolarized')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Fully polarized')
        
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Value')
        ax.set_title('Orientation Dependence Map')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set proper y-axis limits with some padding
        all_values = list(depol_ratios) + list(max_intensities)
        if all_values:
            y_min = max(-0.1, min(all_values) * 0.95)  # Small negative margin
            y_max = max(1.1, max(all_values) * 1.05)   # At least 1.1 for reference lines
            ax.set_ylim(y_min, y_max)
    
    def plot_peak_comparison(self, ax, tensors, wavenumbers):
        """Plot comparison of peak tensors."""
        if not hasattr(self, 'fitted_peaks') or not self.fitted_peaks:
            ax.text(0.5, 0.5, 'No fitted peaks available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract tensor properties for each peak
        peak_freqs = []
        peak_anisotropies = []
        peak_max_eigenvals = []
        
        for peak in self.fitted_peaks:
            # Try both 'center' and 'position' keys for compatibility
            peak_freq = peak.get('center', peak.get('position', 0))
            idx = np.argmin(np.abs(wavenumbers - peak_freq))
            tensor = tensors[idx]
            
            # Calculate anisotropy
            eigenvals = np.linalg.eigvals(tensor @ tensor.T)
            eigenvals = np.sort(np.abs(eigenvals))[::-1]
            
            if eigenvals[0] > 1e-10:
                anisotropy = (eigenvals[0] - eigenvals[2]) / eigenvals[0]
            else:
                anisotropy = 0
            
            peak_freqs.append(peak_freq)
            peak_anisotropies.append(anisotropy)
            peak_max_eigenvals.append(eigenvals[0])
        
        # Normalize max eigenvalues
        if np.max(peak_max_eigenvals) > 0:
            peak_max_eigenvals = np.array(peak_max_eigenvals) / np.max(peak_max_eigenvals)
        
        # Create scatter plot
        scatter = ax.scatter(peak_freqs, peak_anisotropies, c=peak_max_eigenvals, 
                           s=100, cmap='viridis', alpha=0.7)
        
        ax.set_xlabel('Peak Frequency (cm⁻¹)')
        ax.set_ylabel('Anisotropy')
        ax.set_title('Peak Tensor Comparison')
        ax.grid(True, alpha=0.3)
        
        # Set proper y-axis limits with some padding
        if peak_anisotropies:
            y_min = max(0, min(peak_anisotropies) * 0.95)  # Don't go below 0
            y_max = min(1.1, max(peak_anisotropies) * 1.05)  # Anisotropy is typically 0-1
            if y_max > y_min:
                ax.set_ylim(y_min, y_max)
        
        # Check if colorbar already exists and remove it
        if hasattr(ax, '_colorbar') and ax._colorbar is not None:
            try:
                ax._colorbar.remove()
            except:
                pass
        
        # Add new colorbar and store reference
        try:
            ax._colorbar = plt.colorbar(scatter, ax=ax, label='Normalized Max Eigenvalue')
        except Exception as e:
            print(f"Warning: Could not add colorbar: {e}")
        
        # Add peak labels
        for i, freq in enumerate(peak_freqs):
            ax.annotate(f'{freq:.0f}', (freq, peak_anisotropies[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    def update_tensor_peak_selection(self):
        """Update the peak selection dropdown."""
        try:
            peak_options = ["All Peaks"]
            
            if hasattr(self, 'fitted_peaks') and self.fitted_peaks:
                for i, peak in enumerate(self.fitted_peaks):
                    # Try both 'center' and 'position' keys for compatibility
                    freq = peak.get('center', peak.get('position', 0))
                    peak_options.append(f"{freq:.1f} cm⁻¹")
            
            if hasattr(self, 'tensor_peak_combo'):
                self.tensor_peak_combo['values'] = peak_options
                if self.tensor_peak_var.get() not in peak_options:
                    self.tensor_peak_var.set("All Peaks")
        except Exception as e:
            print(f"Error updating tensor peak selection: {e}")
    
    def update_tensor_status(self):
        """Update the tensor analysis status display."""
        try:
            if hasattr(self, 'tensor_status_text'):
                self.tensor_status_text.delete(1.0, tk.END)
                
                if not self.calculated_raman_tensors:
                    self.tensor_status_text.insert(tk.END, "No tensor data available")
                    # Enable structure import button when no data is available
                    if hasattr(self, 'structure_import_button'):
                        self.structure_import_button.config(state=tk.NORMAL)
                    return
                
                status = "TENSOR ANALYSIS STATUS\n"
                status += "=" * 25 + "\n\n"
                
                if 'wavenumbers' in self.calculated_raman_tensors:
                    n_points = len(self.calculated_raman_tensors['wavenumbers'])
                    status += f"Data points: {n_points}\n"
                
                if 'configurations' in self.calculated_raman_tensors:
                    configs = self.calculated_raman_tensors['configurations']
                    status += f"Configurations: {len(configs)}\n"
                    status += f"  {', '.join(configs)}\n"
                
                # Check if we have high-quality database eigenvalues
                has_database_eigenvalues = False
                if 'source' in self.calculated_raman_tensors:
                    source = self.calculated_raman_tensors['source']
                    status += f"Data source: {source}\n"
                    
                    if source == 'database':
                        mineral_name = self.calculated_raman_tensors.get('mineral_name', 'Unknown')
                        status += f"Mineral: {mineral_name}\n"
                        
                        if self.calculated_raman_tensors.get('high_quality_eigenvalues', False):
                            status += "✓ High-quality database eigenvalues\n"
                            has_database_eigenvalues = True
                        
                        # Show available database tensor types
                        db_tensors = self.calculated_raman_tensors.get('database_tensors', {})
                        if db_tensors:
                            status += "Available tensors:\n"
                            for tensor_type in db_tensors.keys():
                                status += f"  • {tensor_type}\n"
                
                if 'eigenvalues' in self.calculated_raman_tensors:
                    eigenvals = self.calculated_raman_tensors['eigenvalues']
                    status += f"✓ Eigenanalysis: {eigenvals.shape}\n"
                
                if 'full_tensors' in self.calculated_raman_tensors:
                    tensors = self.calculated_raman_tensors['full_tensors']
                    status += f"✓ Tensor array: {tensors.shape}\n"
                    
                    # Check if tensors are actually different
                    if len(tensors) > 1:
                        first_tensor = tensors[0]
                        last_tensor = tensors[-1]
                        diff = np.linalg.norm(first_tensor - last_tensor)
                        status += f"Tensor variation: {diff:.6f}\n"
                        if diff < 1e-10:
                            status += "⚠ All tensors identical!\n"
                        else:
                            status += "✓ Tensors vary by frequency\n"
                
                if 'peak_tensors' in self.calculated_raman_tensors:
                    n_peaks = len(self.calculated_raman_tensors['peak_tensors'])
                    status += f"✓ Peak analysis: {n_peaks} peaks\n"
                
                # Manage structure import button state
                if hasattr(self, 'structure_import_button'):
                    if has_database_eigenvalues:
                        # Disable button when high-quality database data is available
                        self.structure_import_button.config(state=tk.DISABLED)
                        status += "\nNote: Structure import disabled\n(using high-quality database data)\n"
                    else:
                        # Enable button when no database eigenvalues are available
                        self.structure_import_button.config(state=tk.NORMAL)
                
                self.tensor_status_text.insert(tk.END, status)
        except Exception as e:
            print(f"Error updating tensor status: {e}")
    
    def export_to_orientation_optimization(self):
        """Export tensor data to Orientation Optimization tab."""
        if not self.calculated_raman_tensors or 'full_tensors' not in self.calculated_raman_tensors:
            messagebox.showwarning("No Data", "Please extract tensor data first.")
            return
        
        try:
            # Store tensor data for orientation optimization
            self.tensor_analysis_results = {
                'tensors': self.calculated_raman_tensors['full_tensors'],
                'wavenumbers': self.calculated_raman_tensors['wavenumbers'],
                'eigenvalues': self.calculated_raman_tensors.get('eigenvalues'),
                'eigenvectors': self.calculated_raman_tensors.get('eigenvectors'),
                'peak_tensors': self.calculated_raman_tensors.get('peak_tensors', {}),
                'export_timestamp': time.time(),
                'source': self.calculated_raman_tensors.get('source', 'tensor_analysis'),
                'mineral_name': self.calculated_raman_tensors.get('mineral_name'),
                'crystal_system': self.calculated_raman_tensors.get('crystal_system'),
                'space_group': self.calculated_raman_tensors.get('space_group'),
                'point_group': self.calculated_raman_tensors.get('point_group')
            }
            
            # Auto-populate crystal system in orientation optimization if available
            if self.tensor_analysis_results['crystal_system']:
                if hasattr(self, 'opt_crystal_system_var'):
                    self.opt_crystal_system_var.set(self.tensor_analysis_results['crystal_system'])
                    print(f"Auto-populated crystal system from tensor data: {self.tensor_analysis_results['crystal_system']}")
            
            # Transfer peak assignments if they exist
            if self.peak_assignments:
                self.tensor_analysis_results['peak_assignments'] = self.peak_assignments.copy()
            
            # Transfer fitted peaks if they exist
            if self.fitted_peaks:
                self.tensor_analysis_results['fitted_peaks'] = self.fitted_peaks.copy()
            
            # Update optimization status
            if hasattr(self, 'update_optimization_status'):
                self.update_optimization_status()
            
            # Switch to orientation optimization tab
            if hasattr(self, 'notebook'):
                for i, tab_id in enumerate(self.notebook.tabs()):
                    tab_text = self.notebook.tab(tab_id, "text")
                    if "Orientation Optimization" in tab_text:
                        self.notebook.select(i)
                        break
            
            messagebox.showinfo("Export Success", 
                              "Tensor data exported to Orientation Optimization tab.\n"
                              f"Crystal system: {self.tensor_analysis_results['crystal_system'] or 'Not detected'}\n"
                              f"Tensors: {len(self.tensor_analysis_results['wavenumbers'])} frequencies\n"
                              "You can now use this data for orientation optimization.")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting to orientation optimization: {str(e)}")
    
    def export_tensor_results(self):
        """Export tensor analysis results to file."""
        if not self.calculated_raman_tensors:
            messagebox.showwarning("No Data", "No tensor data to export.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Tensor Results",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            with open(file_path, 'w') as f:
                f.write("RAMAN TENSOR ANALYSIS RESULTS\n")
                f.write("=" * 50 + "\n\n")
                
                # Write metadata
                f.write(f"Export date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data source: {self.calculated_raman_tensors.get('source', 'unknown')}\n")
                f.write(f"Configurations: {', '.join(self.calculated_raman_tensors.get('configurations', []))}\n\n")
                
                # Write tensor data
                wavenumbers = self.calculated_raman_tensors['wavenumbers']
                tensors = self.calculated_raman_tensors['full_tensors']
                
                f.write("TENSOR MATRICES\n")
                f.write("-" * 20 + "\n")
                f.write("Format: Wavenumber, T11, T12, T13, T21, T22, T23, T31, T32, T33\n\n")
                
                for i, freq in enumerate(wavenumbers):
                    tensor = tensors[i]
                    f.write(f"{freq:.2f}")
                    for row in tensor:
                        for element in row:
                            f.write(f", {element:.6f}")
                    f.write("\n")
                
                # Write eigenvalue data if available
                if 'eigenvalues' in self.calculated_raman_tensors:
                    f.write("\n\nEIGENVALUES\n")
                    f.write("-" * 15 + "\n")
                    f.write("Format: Wavenumber, λ1, λ2, λ3\n\n")
                    
                    eigenvalues = self.calculated_raman_tensors['eigenvalues']
                    for i, freq in enumerate(wavenumbers):
                        f.write(f"{freq:.2f}, {eigenvalues[i,0]:.6f}, {eigenvalues[i,1]:.6f}, {eigenvalues[i,2]:.6f}\n")
                
                # Write peak analysis if available
                if 'peak_tensors' in self.calculated_raman_tensors:
                    f.write("\n\nPEAK ANALYSIS\n")
                    f.write("-" * 15 + "\n")
                    
                    for freq, peak_data in self.calculated_raman_tensors['peak_tensors'].items():
                        f.write(f"\nPeak at {freq:.1f} cm⁻¹:\n")
                        tensor = peak_data['tensor']
                        eigenvals = peak_data['eigenvalues']
                        
                        f.write("Tensor matrix:\n")
                        for row in tensor:
                            f.write(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]\n")
                        
                        f.write(f"Eigenvalues: [{eigenvals[0]:.4f}, {eigenvals[1]:.4f}, {eigenvals[2]:.4f}]\n")
            
            messagebox.showinfo("Export Success", f"Tensor results exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting tensor results: {str(e)}")
    
    def generate_tensor_report(self):
        """Generate comprehensive tensor analysis report."""
        if not self.calculated_raman_tensors:
            messagebox.showwarning("No Data", "No tensor data available for report generation.")
            return
        
        try:
            # Create report window
            report_window = tk.Toplevel(self.root)
            report_window.title("Tensor Analysis Report")
            report_window.geometry("800x700")
            
            # Create text widget with scrollbar
            text_frame = ttk.Frame(report_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Generate comprehensive report
            report = self.generate_comprehensive_tensor_report()
            
            text_widget.insert(tk.END, report)
            text_widget.config(state=tk.DISABLED)
            
            # Add export button
            export_button = ttk.Button(report_window, text="Export Report", 
                                     command=lambda: self.export_report_to_file(report))
            export_button.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating tensor report: {str(e)}")
    
    def generate_comprehensive_tensor_report(self):
        """Generate comprehensive tensor analysis report text."""
        report = "COMPREHENSIVE RAMAN TENSOR ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Metadata
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        source = self.calculated_raman_tensors.get('source', 'unknown')
        report += f"Data source: {source}\n"
        
        if source == 'database':
            mineral_name = self.calculated_raman_tensors.get('mineral_name', 'Unknown')
            report += f"Mineral: {mineral_name}\n"
            if self.calculated_raman_tensors.get('high_quality_eigenvalues', False):
                report += "✓ Using high-quality database eigenvalues\n"
            
            # List available database tensors
            db_tensors = self.calculated_raman_tensors.get('database_tensors', {})
            if db_tensors:
                report += f"Database tensors: {', '.join(db_tensors.keys())}\n"
        else:
            report += f"Configurations: {', '.join(self.calculated_raman_tensors.get('configurations', []))}\n"
        
        report += "\n"
        
        # Data summary
        wavenumbers = self.calculated_raman_tensors['wavenumbers']
        tensors = self.calculated_raman_tensors['full_tensors']
        
        report += "DATA SUMMARY\n"
        report += "-" * 20 + "\n"
        report += f"Frequency range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹\n"
        report += f"Data points: {len(wavenumbers)}\n"
        report += f"Tensor dimensions: {tensors.shape[1]}×{tensors.shape[2]}\n\n"
        
        # Statistical analysis
        if 'eigenvalues' in self.calculated_raman_tensors:
            eigenvalues = self.calculated_raman_tensors['eigenvalues']
            anisotropy = self.calculated_raman_tensors.get('anisotropy', [])
            
            report += "STATISTICAL ANALYSIS\n"
            report += "-" * 25 + "\n"
            report += f"Average eigenvalues:\n"
            report += f"  λ₁ (max): {np.mean(eigenvalues[:, 0]):.4f} ± {np.std(eigenvalues[:, 0]):.4f}\n"
            report += f"  λ₂ (mid): {np.mean(eigenvalues[:, 1]):.4f} ± {np.std(eigenvalues[:, 1]):.4f}\n"
            report += f"  λ₃ (min): {np.mean(eigenvalues[:, 2]):.4f} ± {np.std(eigenvalues[:, 2]):.4f}\n\n"
            
            if len(anisotropy) > 0:
                report += f"Anisotropy: {np.mean(anisotropy):.4f} ± {np.std(anisotropy):.4f}\n"
                report += f"Anisotropy range: {np.min(anisotropy):.4f} - {np.max(anisotropy):.4f}\n\n"
        
        # Database-specific properties
        if source == 'database':
            report += "DATABASE PROPERTIES\n"
            report += "-" * 25 + "\n"
            
            if 'birefringence' in self.calculated_raman_tensors:
                birefringence = self.calculated_raman_tensors['birefringence']
                report += f"Birefringence (Δn): {birefringence:.6f}\n"
            
            if 'average_refractive_index' in self.calculated_raman_tensors:
                avg_n = self.calculated_raman_tensors['average_refractive_index']
                report += f"Average refractive index: {avg_n:.4f}\n"
            
            if 'dielectric_anisotropy' in self.calculated_raman_tensors:
                diel_aniso = self.calculated_raman_tensors['dielectric_anisotropy']
                report += f"Dielectric anisotropy: {diel_aniso:.4f}\n"
            
            # Show database eigenvalues
            if 'database_eigenvalues' in self.calculated_raman_tensors:
                db_eigenvals = self.calculated_raman_tensors['database_eigenvalues']
                if 'n' in db_eigenvals:
                    n_vals = db_eigenvals['n']
                    report += f"Refractive index eigenvalues: [{n_vals[0]:.4f}, {n_vals[1]:.4f}, {n_vals[2]:.4f}]\n"
                if 'eps_inf' in db_eigenvals:
                    eps_vals = db_eigenvals['eps_inf']
                    report += f"High-freq dielectric eigenvalues: [{eps_vals[0]:.4f}, {eps_vals[1]:.4f}, {eps_vals[2]:.4f}]\n"
            
            report += "\n"
        
        # Peak analysis
        if 'peak_tensors' in self.calculated_raman_tensors and self.calculated_raman_tensors['peak_tensors']:
            report += "PEAK-SPECIFIC ANALYSIS\n"
            report += "-" * 30 + "\n"
            
            for freq, peak_data in self.calculated_raman_tensors['peak_tensors'].items():
                tensor = peak_data['tensor']
                eigenvals = peak_data['eigenvalues']
                
                report += f"\nPeak at {freq:.1f} cm⁻¹:\n"
                report += f"  Tensor trace: {np.trace(tensor):.4f}\n"
                report += f"  Tensor determinant: {np.linalg.det(tensor):.4f}\n"
                report += f"  Eigenvalues: [{eigenvals[0]:.4f}, {eigenvals[1]:.4f}, {eigenvals[2]:.4f}]\n"
                
                # Calculate anisotropy for this peak
                if eigenvals[0] > 1e-10:
                    peak_anisotropy = (eigenvals[0] - eigenvals[2]) / eigenvals[0]
                    report += f"  Anisotropy: {peak_anisotropy:.4f}\n"
                
                # Symmetry analysis
                symmetry_error = np.linalg.norm(tensor - tensor.T) / np.linalg.norm(tensor)
                if symmetry_error < 0.01:
                    report += f"  Symmetry: Symmetric (error: {symmetry_error:.4f})\n"
                else:
                    report += f"  Symmetry: Non-symmetric (error: {symmetry_error:.4f})\n"
        
        # Recommendations
        report += "\n\nRECOMMENDations\n"
        report += "-" * 20 + "\n"
        
        if 'eigenvalues' in self.calculated_raman_tensors:
            eigenvalues = self.calculated_raman_tensors['eigenvalues']
            max_anisotropy_idx = np.argmax(self.calculated_raman_tensors.get('anisotropy', [0]))
            
            report += f"• Most anisotropic frequency: {wavenumbers[max_anisotropy_idx]:.1f} cm⁻¹\n"
            report += f"• Recommended for orientation optimization\n"
            
            # Check for isotropic regions
            anisotropy = self.calculated_raman_tensors.get('anisotropy', [])
            if len(anisotropy) > 0:
                isotropic_threshold = 0.1
                isotropic_regions = np.where(np.array(anisotropy) < isotropic_threshold)[0]
                if len(isotropic_regions) > 0:
                    report += f"• Isotropic regions found at: "
                    report += f"{', '.join([f'{wavenumbers[i]:.1f}' for i in isotropic_regions[:5]])} cm⁻¹\n"
        
        report += f"• Export data to Orientation Optimization tab for further analysis\n"
        
        if source == 'database':
            report += f"• Database eigenvalues provide high accuracy for optical properties\n"
            report += f"• Consider using these values for precise orientation optimization\n"
        else:
            report += f"• Consider using database eigenvalues if available for higher accuracy\n"
            report += f"• Additional polarization measurements could improve tensor determination\n"
        
        return report
    
    def export_report_to_file(self, report_text):
        """Export report text to file."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Tensor Report",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(report_text)
                messagebox.showinfo("Export Success", f"Report exported to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting report: {str(e)}")



    # Placeholder methods for Stress/Strain tab
    def load_stress_data(self):
        messagebox.showinfo("Stress/Strain", "Load stress data - placeholder")
    
    def load_strain_data(self):
        messagebox.showinfo("Stress/Strain", "Load strain data - placeholder")
    
    def generate_stress_test_data(self):
        messagebox.showinfo("Stress/Strain", "Generate stress test data - placeholder")
    
    def import_stress_coefficients(self):
        messagebox.showinfo("Stress/Strain", "Import stress coefficients - placeholder")
    
    def calculate_stress_coefficients(self):
        messagebox.showinfo("Stress/Strain", "Calculate stress coefficients - placeholder")
    
    def use_literature_coefficients(self):
        messagebox.showinfo("Stress/Strain", "Use literature coefficients - placeholder")
    
    def analyze_frequency_shifts(self):
        messagebox.showinfo("Stress/Strain", "Analyze frequency shifts - placeholder")
    
    def refine_stress_tensor(self):
        messagebox.showinfo("Stress/Strain", "Refine stress tensor - placeholder")
    
    def create_strain_map(self):
        messagebox.showinfo("Stress/Strain", "Create strain map - placeholder")
    
    def simulate_stressed_spectrum(self):
        messagebox.showinfo("Stress/Strain", "Simulate stressed spectrum - placeholder")
    
    def fit_stress_to_experiment(self):
        messagebox.showinfo("Stress/Strain", "Fit stress to experiment - placeholder")
    
    def show_stress_analysis(self):
        messagebox.showinfo("Stress/Strain", "Show stress analysis - placeholder")
    
    def export_stress_results(self):
        messagebox.showinfo("Stress/Strain", "Export stress results - placeholder")
    
    def load_mineral_database(self):
        """Load the mineral modes database."""
        try:
            database_path = os.path.join(os.path.dirname(__file__), "mineral_modes.pkl")
            if os.path.exists(database_path):
                with open(database_path, 'rb') as f:
                    self.mineral_database = pickle.load(f)
                self.mineral_list = list(self.mineral_database.keys())
                self.mineral_list.sort()
                print(f"Loaded mineral database with {len(self.mineral_list)} minerals")
            else:
                print("Mineral database not found")
                self.mineral_database = {}
                self.mineral_list = []
        except Exception as e:
            print(f"Error loading mineral database: {e}")
            self.mineral_database = {}
            self.mineral_list = []
    
    def on_search_change(self, *args):
        """Handle search text changes."""
        search_term = self.search_var.get().lower()
        
        # Clear the listbox
        self.search_listbox.delete(0, tk.END)
        
        if not search_term:
            return
        
        # Filter minerals based on search term
        matching_minerals = [mineral for mineral in self.mineral_list 
                           if search_term in mineral.lower()]
        
        # Add matching minerals to listbox (limit to 20 results)
        for mineral in matching_minerals[:20]:
            self.search_listbox.insert(tk.END, mineral)
    
    def on_mineral_select(self, event=None):
        """Handle mineral selection from listbox."""
        selection = self.search_listbox.curselection()
        if selection:
            mineral_name = self.search_listbox.get(selection[0])
            self.import_selected_mineral(mineral_name)
    
    def import_selected_mineral(self, mineral_name=None):
        """Import and display the selected mineral spectrum."""
        if mineral_name is None:
            selection = self.search_listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a mineral from the list.")
                return
            mineral_name = self.search_listbox.get(selection[0])
        
        if mineral_name not in self.mineral_database:
            messagebox.showerror("Error", f"Mineral '{mineral_name}' not found in database.")
            return
        
        try:
            # Generate spectrum from mineral data
            wavenumbers, intensities = self.generate_spectrum_from_mineral(mineral_name)
            
            if wavenumbers is not None and intensities is not None:
                self.imported_spectrum = {
                    'wavenumbers': wavenumbers,
                    'intensities': intensities,
                    'name': mineral_name
                }
                
                # Set this as the selected reference mineral for other tabs
                self.selected_reference_mineral = mineral_name
                self.update_reference_mineral_selections()
                
                # Update the plot
                self.update_spectrum_plot()
                
                #messagebox.showinfo("Success", f"Imported spectrum for {mineral_name}")
            else:
                messagebox.showerror("Error", f"Could not generate spectrum for {mineral_name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error importing mineral spectrum: {str(e)}")
    
    def generate_spectrum_from_mineral(self, mineral_name):
        """Generate a Raman spectrum from mineral mode data."""
        try:
            mineral_data = self.mineral_database[mineral_name]
            
            # Extract modes data
            if 'modes' not in mineral_data:
                print(f"No modes data found for {mineral_name}")
                return None, None
            
            modes = mineral_data['modes']
            if not modes:
                print(f"Empty modes data for {mineral_name}")
                return None, None
            
            # Create wavenumber range
            wavenumbers = np.linspace(100, 4000, 2000)
            intensities = np.zeros_like(wavenumbers)
            
            # Process each mode
            for mode in modes:
                if isinstance(mode, (tuple, list)) and len(mode) >= 3:
                    frequency = float(mode[0])  # Frequency in cm^-1
                    symmetry = str(mode[1])     # Symmetry character
                    intensity = float(mode[2])  # Relative intensity
                    
                    # Skip modes with zero or very low frequency
                    if frequency < 50:
                        continue
                    
                    # Generate Lorentzian peak
                    width = 10.0  # Peak width in cm^-1
                    peak = intensity * (width**2) / ((wavenumbers - frequency)**2 + width**2)
                    intensities += peak
            
            # Normalize intensities
            if np.max(intensities) > 0:
                intensities = intensities / np.max(intensities)
            
            # Add some noise for realism
            noise_level = 0.02
            intensities += noise_level * np.random.randn(len(intensities))
            intensities = np.maximum(intensities, 0)  # Ensure non-negative
            
            return wavenumbers, intensities
            
        except Exception as e:
            print(f"Error generating spectrum for {mineral_name}: {e}")
            return None, None
    
    def update_spectrum_plot(self):
        """Update the spectrum plot with current and imported spectra."""
        # Get the plot components for the Spectrum Analysis tab
        if "Spectrum Analysis" in self.plot_components:
            components = self.plot_components["Spectrum Analysis"]
            ax = components['ax']
            canvas = components['canvas']
            
            # Clear the plot
            ax.clear()
            
            # Check if we have any spectra to plot
            has_data = False
            x_limits = None
            
            # Plot current spectrum if available
            if self.current_spectrum is not None:
                current_wavenumbers = self.current_spectrum['wavenumbers']
                current_intensities = self.current_spectrum['intensities']
                
                ax.plot(current_wavenumbers, current_intensities, 
                       'b-', linewidth=2, label=f"Current: {self.current_spectrum['name']}")
                has_data = True
                
                # Set x-axis limits based on current spectrum
                x_limits = (np.min(current_wavenumbers), np.max(current_wavenumbers))
            
            # Plot imported spectrum if available
            if self.imported_spectrum is not None:
                imported_wavenumbers = self.imported_spectrum['wavenumbers']
                imported_intensities = self.imported_spectrum['intensities']
                
                # Normalize imported spectrum to current spectrum if both are available
                if self.current_spectrum is not None:
                    # Find the maximum intensity of the current spectrum
                    current_max = np.max(self.current_spectrum['intensities'])
                    
                    # Normalize imported spectrum to match current spectrum's scale
                    imported_max = np.max(imported_intensities)
                    if imported_max > 0:
                        normalized_intensities = imported_intensities * (current_max / imported_max)
                    else:
                        normalized_intensities = imported_intensities
                    
                    ax.plot(imported_wavenumbers, normalized_intensities, 
                           'r-', linewidth=2, alpha=0.7, 
                           label=f"Imported: {self.imported_spectrum['name']} (normalized)")
                else:
                    # If no current spectrum, plot imported spectrum as-is
                    ax.plot(imported_wavenumbers, imported_intensities, 
                           'r-', linewidth=2, alpha=0.7, 
                           label=f"Imported: {self.imported_spectrum['name']}")
                    
                    # Set x-axis limits based on imported spectrum if no current spectrum
                    if x_limits is None:
                        x_limits = (np.min(imported_wavenumbers), np.max(imported_wavenumbers))
                
                has_data = True
            
            # Set labels and title
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title("Raman Spectrum Analysis")
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits if we have data
            if x_limits is not None:
                ax.set_xlim(x_limits)
            
            # Show legend if we have data, otherwise show instruction text
            if has_data:
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'Load a spectrum or import from database', 
                       transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, alpha=0.6)
            
            # Refresh the canvas
            canvas.draw()
        else:
            print("Warning: Spectrum Analysis plot components not found")
    
    def update_peak_fitting_plot(self):
        """Update the peak fitting plot with current spectrum, selected peaks, and fitted peaks."""
        # Get the plot components for the Peak Fitting tab
        if "Peak Fitting" in self.plot_components:
            components = self.plot_components["Peak Fitting"]
            ax = components['ax']
            canvas = components['canvas']
            
            # Clear the plot
            ax.clear()
            
            # Check if we have spectrum data to plot
            has_data = False
            
            # Plot current spectrum if available
            if self.current_spectrum is not None:
                wavenumbers = self.current_spectrum['wavenumbers']
                intensities = self.current_spectrum['intensities']
                
                ax.plot(wavenumbers, intensities, 'b-', linewidth=1, alpha=0.7, label='Spectrum')
                has_data = True
                
                # Plot selected peaks
                if self.selected_peaks:
                    for peak_pos in self.selected_peaks:
                        peak_idx = np.argmin(np.abs(wavenumbers - peak_pos))
                        ax.axvline(x=peak_pos, color='red', linestyle='--', alpha=0.7)
                        ax.plot(peak_pos, intensities[peak_idx], 'ro', markersize=8, label='Selected' if peak_pos == self.selected_peaks[0] else "")
                
                # Plot fitted peaks
                if self.fitted_peaks:
                    x_fit = np.linspace(np.min(wavenumbers), np.max(wavenumbers), 1000)
                    
                    for i, peak in enumerate(self.fitted_peaks):
                        # Generate fitted curve
                        if peak['shape'] == "Lorentzian":
                            y_fit = self.lorentzian(x_fit, *peak['parameters'])
                        elif peak['shape'] == "Gaussian":
                            y_fit = self.gaussian(x_fit, *peak['parameters'])
                        elif peak['shape'] == "Voigt":
                            y_fit = self.voigt(x_fit, *peak['parameters'])
                        
                        ax.plot(x_fit, y_fit, 'g-', linewidth=2, alpha=0.8, 
                               label=f'Fitted {peak["shape"]}' if i == 0 else "")
                        
                        # Mark fitted peak center
                        ax.axvline(x=peak['position'], color='green', linestyle='-', alpha=0.5)
                        
                        # Add character assignment annotation if available
                        if peak['position'] in self.peak_assignments:
                            assignment = self.peak_assignments[peak['position']]
                            character = assignment['character']
                            
                            # Add text annotation
                            ax.annotate(character, 
                                       xy=(peak['position'], peak['amplitude']),
                                       xytext=(peak['position'], peak['amplitude'] * 1.1),
                                       ha='center', va='bottom',
                                       fontsize=10, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                        
                        # Add frequency shift annotation if available
                        if peak['position'] in self.frequency_shifts:
                            shift_data = self.frequency_shifts[peak['position']]
                            shift = shift_data['shift']
                            
                            # Add shift annotation
                            shift_text = f"Δ{shift:+.1f}"
                            ax.annotate(shift_text,
                                       xy=(peak['position'], peak['amplitude'] * 0.8),
                                       ha='center', va='top',
                                       fontsize=9, style='italic',
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
            
            # Set labels and title
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title("Peak Fitting Analysis")
            ax.grid(True, alpha=0.3)
            
            # Show legend if we have data
            if has_data:
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'Load a spectrum to begin peak fitting', 
                       transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, alpha=0.6)
            
            # Refresh the canvas
            canvas.draw()
        else:
            print("Warning: Peak Fitting plot components not found")


def main():
    root = tk.Tk()

    # Set up the Clam theme
    style = ttk.Style()
    style.theme_use('clam')

    # Configure ttk widget styles
    style.configure('TButton', padding=5)
    style.configure('TEntry', padding=5)
    style.configure('TCombobox', padding=5)
    style.configure('TNotebook.Tab', padding=[10, 2])

    # Create the application
    app = RamanPolarizationAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main() 