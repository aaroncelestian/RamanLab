import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import pickle
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize, curve_fit
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import re
import io
import sys
from raman_tensor_3d_visualization import RamanTensor3DVisualizer


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
        root.geometry("1100x700")
        
        # Initialize variables
        self.spectrum_data = None  # Will hold the imported spectrum data
        self.mineral_database = None  # Will hold the mineral database
        self.current_mineral = None  # Currently selected mineral
        self.db_plot = None  # Reference to the database spectrum plot
        self.peak_annotations = []  # Store references to peak annotations
        
        # Initialize UI control variables
        self.show_components_var = tk.BooleanVar(value=True)
        
        # Alignment variables
        self.shift_value = 0.0  # Shift value for manual alignment
        self.scale_factor = 1.0  # Scale factor for manual alignment
        self.original_frequencies = []  # Original frequencies before alignment
        self.frequencies = []  # Current frequencies (might be shifted)
        self.intensities = []  # Current intensities
        self.activities = []  # Current activities
        
        # Peak fitting variables
        self.fitted_peaks = []  # List to store fitted peak parameters
        self.peak_fit_plot = None  # Reference to the fitted spectrum plot
        self.peak_components = []  # Individual peak component plots
        self.fitted_peak_markers = []  # Markers for fitted peak positions
        self.selected_range = None  # Selected range for peak fitting
        self.range_rect = None  # Rectangle for visualizing selected range
        self.selecting_range = False  # Flag for range selection mode
        
        # Multi-region fitting variables
        self.fitting_regions = []  # List of (start_x, end_x) tuples
        self.region_rectangles = []  # Visualization rectangles
        self.region_labels = []  # Text labels for regions
        self.fitted_regions = {}  # Dictionary of region: fitted parameters
        
        # Load the mineral database
        self.load_mineral_database()
        
        # Create the main layout
        self.create_gui()
    
    def create_gui(self):
        """Create the graphical user interface"""
        # Create main frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook with tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # First tab - Main analyzer
        self.analyzer_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analyzer_tab, text="Spectrum Analyzer")
        
        # Second tab - Peak fitting
        self.peak_fitting_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.peak_fitting_tab, text="Peak Fitting")
        
        # Setup each tab
        self.setup_analyzer_tab()
        self.setup_peak_fitting_tab()
        
        # Add menu bar
        self.create_menu_bar()
        
        # Add crystal orientation tab
        self.setup_crystal_orientation_tab()
        
        # Add 3D tensor visualization tab
        self.setup_3d_tensor_tab()
    
    def setup_analyzer_tab(self):
        """Set up the main analyzer tab"""
        # Create main frames for analyzer tab
        self.control_frame = ttk.Frame(self.analyzer_tab, padding="10", width=250)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.control_frame.pack_propagate(False)  # Prevent the frame from resizing
        
        self.plot_frame = ttk.Frame(self.analyzer_tab)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control panel
        ttk.Label(self.control_frame, text="Raman Polarization Analyzer", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Import button
        ttk.Button(self.control_frame, text="Open Spectrum File", command=self.import_spectrum).pack(fill=tk.X, pady=5)
        
        # File info label
        self.file_label = ttk.Label(self.control_frame, text="No file selected", wraplength=230)
        self.file_label.pack(fill=tk.X, pady=5)
        
        # Separator
        ttk.Separator(self.control_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        
        # Mineral database section
        ttk.Label(self.control_frame, text="Mineral Database", font=("Arial", 11, "bold")).pack(pady=5)
        
        # Search frame
        search_frame = ttk.Frame(self.control_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        # Search entry
        self.search_var = tk.StringVar()
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Search button
        ttk.Button(self.control_frame, text="Search Mineral", command=self.search_mineral).pack(fill=tk.X, pady=5)
        
        # Display current mineral
        self.mineral_label = ttk.Label(self.control_frame, text="No mineral selected", wraplength=230)
        self.mineral_label.pack(fill=tk.X, pady=5)
        
        # Clear mineral button
        ttk.Button(self.control_frame, text="Clear Mineral Overlay", command=self.clear_mineral_overlay).pack(fill=tk.X, pady=5)
        
        # Peak width control
        peak_frame = ttk.Frame(self.control_frame)
        peak_frame.pack(fill=tk.X, pady=10)
        ttk.Label(peak_frame, text="Peak Width:").pack(side=tk.LEFT)
        
        self.peak_width = tk.DoubleVar(value=10.0)
        peak_width_spinner = ttk.Spinbox(
            peak_frame, 
            from_=1.0, 
            to=30.0, 
            increment=0.5,
            textvariable=self.peak_width,
            width=5
        )
        peak_width_spinner.pack(side=tk.LEFT, padx=5)
        
        # Separator before alignment controls
        ttk.Separator(self.control_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        
        # Peak alignment section
        ttk.Label(self.control_frame, text="Peak Alignment", font=("Arial", 11, "bold")).pack(pady=5)
        
        # Manual shift controls
        shift_frame = ttk.Frame(self.control_frame)
        shift_frame.pack(fill=tk.X, pady=5)
        ttk.Label(shift_frame, text="Shift (cm⁻¹):").pack(side=tk.LEFT)
        
        self.shift_var = tk.DoubleVar(value=0.0)
        shift_spinner = ttk.Spinbox(
            shift_frame, 
            from_=-100.0, 
            to=100.0, 
            increment=1.0,
            textvariable=self.shift_var,
            width=6,
            command=self.on_manual_shift_change
        )
        shift_spinner.pack(side=tk.LEFT, padx=5)
        
        # Scale factor controls
        scale_frame = ttk.Frame(self.control_frame)
        scale_frame.pack(fill=tk.X, pady=5)
        ttk.Label(scale_frame, text="Scale Factor:").pack(side=tk.LEFT)
        
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_spinner = ttk.Spinbox(
            scale_frame, 
            from_=0.8, 
            to=1.2, 
            increment=0.01,
            textvariable=self.scale_var,
            width=6,
            command=self.on_manual_shift_change
        )
        scale_spinner.pack(side=tk.LEFT, padx=5)
        
        # Auto align button
        ttk.Button(self.control_frame, text="Auto-Align Peaks", command=self.auto_align_peaks).pack(fill=tk.X, pady=5)
        
        # Reset alignment button
        ttk.Button(self.control_frame, text="Reset Alignment", command=self.reset_alignment).pack(fill=tk.X, pady=5)
        
        # Update plot button
        ttk.Button(self.control_frame, text="Update Plot", command=self.update_plot).pack(fill=tk.X, pady=10)
        
        # Status label for database info
        self.status_label = ttk.Label(
            self.control_frame, 
            text=f"Database: {len(self.mineral_database or {})} minerals loaded",
            wraplength=230
        )
        self.status_label.pack(fill=tk.X, pady=10)
        
        # Create the plot area
        self.create_plot_area()
    
    def setup_peak_fitting_tab(self):
        """Set up the peak fitting tab"""
        # Create main frames for peak fitting tab
        self.fit_control_frame = ttk.Frame(self.peak_fitting_tab, padding="10", width=250)
        self.fit_control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.fit_control_frame.pack_propagate(False)  # Prevent the frame from resizing
        
        self.fit_plot_frame = ttk.Frame(self.peak_fitting_tab)
        self.fit_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control panel
        ttk.Label(self.fit_control_frame, text="Peak Fitting", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Range selection controls
        ttk.Button(self.fit_control_frame, text="Add Fitting Region", command=self.start_range_selection).pack(fill=tk.X, pady=5)
        ttk.Button(self.fit_control_frame, text="Clear All Regions", command=self.clear_all_regions).pack(fill=tk.X, pady=5)
        
        # Separator
        ttk.Separator(self.fit_control_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        
        # Peak fitting parameters
        ttk.Label(self.fit_control_frame, text="Fitting Parameters", font=("Arial", 11, "bold")).pack(pady=5)
        
        # Region selection
        region_frame = ttk.Frame(self.fit_control_frame)
        region_frame.pack(fill=tk.X, pady=5)
        ttk.Label(region_frame, text="Active Region:").pack(side=tk.LEFT)
        
        self.active_region_var = tk.StringVar(value="All")
        self.region_combo = ttk.Combobox(
            region_frame,
            textvariable=self.active_region_var,
            values=["All"],
            width=10,
            state="readonly"
        )
        self.region_combo.pack(side=tk.LEFT, padx=5)
        self.region_combo.bind("<<ComboboxSelected>>", self.on_region_selected)
        
        # Number of peaks
        num_peaks_frame = ttk.Frame(self.fit_control_frame)
        num_peaks_frame.pack(fill=tk.X, pady=5)
        ttk.Label(num_peaks_frame, text="Number of Peaks:").pack(side=tk.LEFT)
        
        self.num_peaks_var = tk.IntVar(value=1)
        num_peaks_spinner = ttk.Spinbox(
            num_peaks_frame, 
            from_=1, 
            to=10, 
            increment=1,
            textvariable=self.num_peaks_var,
            width=3
        )
        num_peaks_spinner.pack(side=tk.LEFT, padx=5)
        
        # Peak shape
        shape_frame = ttk.Frame(self.fit_control_frame)
        shape_frame.pack(fill=tk.X, pady=5)
        ttk.Label(shape_frame, text="Peak Shape:").pack(side=tk.LEFT)
        
        self.peak_shape_var = tk.StringVar(value="Lorentzian")
        peak_shape_combo = ttk.Combobox(
            shape_frame,
            textvariable=self.peak_shape_var,
            values=["Lorentzian", "Gaussian", "Voigt"],
            width=10,
            state="readonly"
        )
        peak_shape_combo.pack(side=tk.LEFT, padx=5)
        
        # Fitting buttons
        ttk.Button(self.fit_control_frame, text="Perform Peak Fitting", command=self.perform_peak_fitting).pack(fill=tk.X, pady=10)
        ttk.Button(self.fit_control_frame, text="Clear Fitted Peaks", command=self.clear_peak_fitting).pack(fill=tk.X, pady=5)
        
        # Separator
        ttk.Separator(self.fit_control_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        
        # Results section
        ttk.Label(self.fit_control_frame, text="Fitting Results", font=("Arial", 11, "bold")).pack(pady=5)
        
        # Results display - scrolled text
        fit_results_frame = ttk.Frame(self.fit_control_frame)
        fit_results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(fit_results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.fit_results_text = tk.Text(fit_results_frame, height=10, width=25, yscrollcommand=scrollbar.set)
        self.fit_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.fit_results_text.yview)
        
        # Create the plot area
        self.create_fit_plot_area()
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        
        # Add File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Spectrum", command=self.import_spectrum)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Add Database menu
        db_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Database", menu=db_menu)
        db_menu.add_command(label="Search Mineral", command=self.search_mineral)
        db_menu.add_command(label="Clear Mineral Overlay", command=self.clear_mineral_overlay)
        
        # Add Alignment menu
        align_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Alignment", menu=align_menu)
        align_menu.add_command(label="Auto-Align Peaks", command=self.auto_align_peaks)
        align_menu.add_command(label="Reset Alignment", command=self.reset_alignment)
        
        # Add Peak Fitting menu
        fit_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Peak Fitting", menu=fit_menu)
        fit_menu.add_command(label="Select Range", command=self.start_range_selection)
        fit_menu.add_command(label="Perform Fitting", command=self.perform_peak_fitting)
        fit_menu.add_command(label="Clear Fitting", command=self.clear_peak_fitting)
    
    def create_plot_area(self):
        """Create the matplotlib plot area for the analyzer tab"""
        # Create a figure and canvas for the plot
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add the toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create the initial empty axis
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Raman Spectrum")
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity")
        self.figure.tight_layout()
    
    def create_fit_plot_area(self):
        """Create the matplotlib plot area for the peak fitting tab"""
        # Create a figure and canvas for the plot
        self.fit_figure = Figure(figsize=(6, 4), dpi=100)
        self.fit_canvas = FigureCanvasTkAgg(self.fit_figure, self.fit_plot_frame)
        self.fit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add the toolbar
        fit_toolbar = NavigationToolbar2Tk(self.fit_canvas, self.fit_plot_frame)
        fit_toolbar.update()
        fit_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create the initial empty axis
        self.fit_ax = self.fit_figure.add_subplot(111)
        self.fit_ax.set_title("Peak Fitting")
        self.fit_ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.fit_ax.set_ylabel("Intensity")
        self.fit_figure.tight_layout()
        
        # Connect events for mouse interaction
        self.fit_canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fit_canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fit_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
    
    def import_spectrum(self):
        """Import a spectrum file (CSV or TXT)"""
        file_path = filedialog.askopenfilename(
            title="Select Spectrum File",
            filetypes=[
                ("Spectrum files", "*.csv;*.txt;*.dat"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("DAT files", "*.dat"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return  # User cancelled the dialog
        
        try:
            # Simple file reading assuming comma or tab separated x,y values
            x, y = [], []
            with open(file_path, 'r') as f:
                header_skipped = False
                for line in f:
                    # Skip comment lines and empty lines
                    if line.strip() and not line.startswith('#'):
                        # Skip the first line if it might be a header
                        if not header_skipped and any(keyword in line.lower() for keyword in ['wavenumber', 'raman', 'shift', 'cm']):
                            header_skipped = True
                            continue
                            
                        # Replace commas with spaces and split by whitespace
                        values = line.strip().replace(',', ' ').split()
                        if len(values) >= 2:
                            try:
                                x_val = float(values[0])
                                y_val = float(values[1])
                                x.append(x_val)
                                y.append(y_val)
                            except ValueError:
                                # Skip lines that don't contain valid numbers
                                continue
            
            if not x or not y:
                messagebox.showerror("Import Error", "No valid data found in the file.")
                return
            
            # Store the data as numpy arrays
            self.spectrum_data = {'x': np.array(x), 'y': np.array(y)}
            
            # Update the file label
            filename = os.path.basename(file_path)
            if len(filename) > 30:  # Truncate long filenames
                display_name = filename[:27] + "..."
            else:
                display_name = filename
                
            self.file_label.config(text=display_name)
            
            # Plot the spectrum on both tabs
            self.plot_spectrum()
            self.plot_fit_spectrum()
            
            # Clear any existing peak fitting
            self.clear_peak_fitting()
            self.clear_all_regions()
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Error importing spectrum: {e}")
    
    def plot_spectrum(self):
        """Plot the loaded spectrum data"""
        if self.spectrum_data is None:
            return
            
        # Clear the axis
        self.ax.clear()
        
        # Plot the data
        self.ax.plot(self.spectrum_data['x'], self.spectrum_data['y'], 'b-', label='Sample Spectrum')
        
        # Set labels and title
        self.ax.set_title("Raman Spectrum")
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity")
        
        # Set x-axis range to cover typical Raman shift range (adjust as needed)
        # Get the max x value from the data
        x_max = np.max(self.spectrum_data['x'])
        x_min = np.min(self.spectrum_data['x'])
        # Add some padding
        padding = (x_max - x_min) * 0.05
        self.ax.set_xlim(x_min - padding, x_max + padding)
        
        # Re-plot the mineral overlay if one is active
        if self.current_mineral is not None:
            self.plot_mineral_overlay()
        
        # Add legend if we have both spectra
        if self.current_mineral is not None:
            self.ax.legend()
        
        # Apply tight layout and redraw
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_fit_spectrum(self):
        """Plot the spectrum in the peak fitting tab"""
        if self.spectrum_data is None:
            return
        
        # Clear the axis
        self.fit_ax.clear()
        
        # Plot the data
        self.fit_ax.plot(self.spectrum_data['x'], self.spectrum_data['y'], 'b-', label='Sample Spectrum')
        
        # Set labels and title
        self.fit_ax.set_title("Peak Fitting")
        self.fit_ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.fit_ax.set_ylabel("Intensity")
        
        # Set x-axis range to cover the data range
        x_max = np.max(self.spectrum_data['x'])
        x_min = np.min(self.spectrum_data['x'])
        padding = (x_max - x_min) * 0.05
        self.fit_ax.set_xlim(x_min - padding, x_max + padding)
        
        # Apply tight layout and redraw
        self.fit_figure.tight_layout()
        self.fit_canvas.draw()
    
    def load_mineral_database(self):
        """Load the mineral database from the pickle file"""
        try:
            with open('mineral_modes.pkl', 'rb') as f:
                self.mineral_database = pickle.load(f)
            print(f"Loaded mineral database with {len(self.mineral_database)} entries")
            
            # Debug: Print structure of first mineral entry
            if self.mineral_database:
                first_mineral = next(iter(self.mineral_database))
                print(f"\nSample mineral entry ({first_mineral}):")
                print("Keys:", list(self.mineral_database[first_mineral].keys()))
                
                # Check Born charges structure
                if 'born_charges' in self.mineral_database[first_mineral]:
                    born_data = self.mineral_database[first_mineral]['born_charges']
                    print("\nBorn charges structure:")
                    if isinstance(born_data, dict):
                        print("Dictionary format with keys:", list(born_data.keys()))
                        for atom_type, data in born_data.items():
                            print(f"\n{atom_type}:")
                            print("Data type:", type(data))
                            if isinstance(data, dict):
                                print("Keys:", list(data.keys()))
                            elif isinstance(data, (list, tuple)):
                                print("Length:", len(data))
                                print("First few values:", data[:5] if len(data) > 5 else data)
                    else:
                        print("Data type:", type(born_data))
                        if isinstance(born_data, (list, tuple)):
                            print("Length:", len(born_data))
                            print("First few values:", born_data[:5] if len(born_data) > 5 else born_data)
                
                # Check eigenvectors structure
                if 'eigenvectors' in self.mineral_database[first_mineral]:
                    eigvec_data = self.mineral_database[first_mineral]['eigenvectors']
                    print("\nEigenvectors structure:")
                    print("Data type:", type(eigvec_data))
                    if isinstance(eigvec_data, dict):
                        print("Dictionary format with keys:", list(eigvec_data.keys()))
                        for atom_type, data in eigvec_data.items():
                            print(f"\n{atom_type}:")
                            print("Data type:", type(data))
                            if isinstance(data, (list, tuple)):
                                print("Length:", len(data))
                                print("First few values:", data[:5] if len(data) > 5 else data)
                    elif isinstance(eigvec_data, (list, tuple)):
                        print("Length:", len(eigvec_data))
                        print("First few values:", eigvec_data[:5] if len(eigvec_data) > 5 else eigvec_data)
        
        except Exception as e:
            messagebox.showerror("Database Error", f"Error loading mineral database: {e}")
            self.mineral_database = {}
    
    def search_mineral(self):
        """Search for a mineral in the database"""
        if self.mineral_database is None:
            messagebox.showerror("Database Error", "Mineral database not loaded.")
            return
        
        search_term = self.search_var.get().strip().lower()
        if not search_term:
            search_term = simpledialog.askstring("Search Mineral", "Enter mineral name to search:")
            if not search_term:
                return
            self.search_var.set(search_term)
        
        # Find matches in the database
        matches = []
        for mineral_name in self.mineral_database.keys():
            if search_term in mineral_name.lower():
                matches.append(mineral_name)
        
        if not matches:
            messagebox.showinfo("Search Results", "No matching minerals found.")
            return
        
        # If only one match, use it directly
        if len(matches) == 1:
            self.select_mineral(matches[0])
        else:
            # Create a selection dialog
            self.show_mineral_selection_dialog(matches)
    
    def show_mineral_selection_dialog(self, matches):
        """Show a dialog to select from multiple matching minerals"""
        # Create a toplevel window
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Mineral")
        select_window.geometry("300x400")
        select_window.transient(self.root)
        select_window.grab_set()
        
        # Add a label
        ttk.Label(select_window, text="Select a mineral:").pack(pady=10)
        
        # Create a frame with scrollbar for the list
        list_frame = ttk.Frame(select_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create listbox
        mineral_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        mineral_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=mineral_listbox.yview)
        
        # Populate the listbox
        for mineral in sorted(matches):
            mineral_listbox.insert(tk.END, mineral)
        
        # Add selection button
        def on_select():
            try:
                selected_idx = mineral_listbox.curselection()[0]
                mineral = mineral_listbox.get(selected_idx)
                select_window.destroy()
                self.select_mineral(mineral)
            except IndexError:
                messagebox.showerror("Selection Error", "Please select a mineral.")
        
        ttk.Button(select_window, text="Select", command=on_select).pack(pady=10)
        
        # Add cancel button
        ttk.Button(select_window, text="Cancel", command=select_window.destroy).pack(pady=5)
    
    def select_mineral(self, mineral_name):
        """Select a mineral from the database and plot its spectrum"""
        self.current_mineral = mineral_name
        self.mineral_label.config(text=f"Selected: {mineral_name}")
        
        # Reset alignment parameters
        self.shift_value = 0.0
        self.scale_factor = 1.0
        self.shift_var.set(0.0)
        self.scale_var.set(1.0)
        
        # Plot the spectrum
        self.extract_mineral_data()
        self.plot_mineral_overlay()
        
        # Update the main plot if needed
        if self.spectrum_data is not None:
            self.plot_spectrum()
    
    def extract_mineral_data(self):
        """Extract frequency, intensity, and activity data from the current mineral"""
        if self.current_mineral is None or self.current_mineral not in self.mineral_database:
            return False
            
        # Get the mineral data
        mineral_data = self.mineral_database[self.current_mineral]
        
        # Clear previous data
        self.original_frequencies = []
        self.frequencies = []
        self.intensities = []
        self.activities = []
        
        # Process the mineral data to extract frequencies and intensities
        if 'modes' in mineral_data and mineral_data['modes'] is not None:
            modes = mineral_data['modes']
            
            # Handle case where modes is a pandas DataFrame
            if isinstance(modes, pd.DataFrame):
                print(f"Found DataFrame for {self.current_mineral}, columns: {modes.columns}")
                if 'TO_Frequency' in modes.columns and 'I_Total' in modes.columns:
                    for _, row in modes.iterrows():
                        try:
                            freq = float(row['TO_Frequency'])
                            intensity = float(row['I_Total']) if pd.notna(row['I_Total']) else 0
                            activity = str(row['Activity']) if 'Activity' in row and pd.notna(row['Activity']) else ""
                            
                            if freq > 0 and intensity > 0:
                                self.original_frequencies.append(freq)
                                self.frequencies.append(freq)
                                self.intensities.append(intensity)
                                self.activities.append(activity)
                        except (ValueError, TypeError):
                            pass
            # Handle case where modes is a list of tuples
            elif isinstance(modes, list):
                for mode in modes:
                    # Check if this is a tuple with at least 2 elements (freq, activity, [intensity])
                    if isinstance(mode, tuple) and len(mode) >= 2:
                        try:
                            freq = float(mode[0])
                            activity = str(mode[1])
                            # If we have 3 elements, the third is intensity
                            intensity = float(mode[2]) if len(mode) > 2 else 1.0
                            
                            if freq > 0:
                                self.original_frequencies.append(freq)
                                self.frequencies.append(freq)
                                self.intensities.append(intensity)
                                self.activities.append(activity)
                        except (ValueError, TypeError):
                            pass
                    # Check if it's a dictionary-like object
                    elif hasattr(mode, 'items'):
                        try:
                            freq = float(mode['TO_Frequency']) if 'TO_Frequency' in mode else 0
                            intensity = float(mode['I_Total']) if 'I_Total' in mode else 0
                            activity = str(mode['Activity']) if 'Activity' in mode else ""
                            
                            if freq > 0 and intensity > 0:
                                self.original_frequencies.append(freq)
                                self.frequencies.append(freq)
                                self.intensities.append(intensity)
                                self.activities.append(activity)
                        except (ValueError, TypeError):
                            pass
        
        # If phonon_modes exists and we didn't find frequencies, try that
        if not self.frequencies and 'phonon_modes' in mineral_data:
            phonon_modes = mineral_data['phonon_modes']
            
            # Check if it's a pandas DataFrame
            if isinstance(phonon_modes, pd.DataFrame):
                print(f"Found phonon_modes DataFrame, columns: {phonon_modes.columns}")
                freq_col = None
                intensity_col = None
                activity_col = None
                
                # Look for appropriate columns
                for col in phonon_modes.columns:
                    col_lower = col.lower()
                    if 'freq' in col_lower:
                        freq_col = col
                    elif 'intens' in col_lower or 'i_tot' in col_lower:
                        intensity_col = col
                    elif 'activ' in col_lower or 'charac' in col_lower:
                        activity_col = col
                
                if freq_col:
                    for _, row in phonon_modes.iterrows():
                        try:
                            freq = float(row[freq_col])
                            intensity = float(row[intensity_col]) if intensity_col and pd.notna(row[intensity_col]) else 1.0
                            activity = str(row[activity_col]) if activity_col and pd.notna(row[activity_col]) else ""
                            
                            if freq > 0:
                                self.original_frequencies.append(freq)
                                self.frequencies.append(freq)
                                self.intensities.append(intensity)
                                self.activities.append(activity)
                        except (ValueError, TypeError):
                            pass
        
        if not self.frequencies:
            # Try one more approach - see if there are direct keys we can use
            print(f"Debug: Could not find frequency data for {self.current_mineral}")
            print(f"Available keys: {list(mineral_data.keys())}")
            
            # Let's check a few common patterns in the database
            if isinstance(mineral_data.get('modes'), list) and len(mineral_data['modes']) > 0:
                print(f"Sample modes data: {mineral_data['modes'][:3]}")
                
                # If modes are just a list of frequencies
                try:
                    all_numeric = True
                    for value in mineral_data['modes'][:5]:
                        if not isinstance(value, (int, float)) and not (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                            all_numeric = False
                            break
                    
                    if all_numeric:
                        # Just frequencies, no intensity or activity info
                        for value in mineral_data['modes']:
                            try:
                                freq = float(value)
                                if freq > 0:
                                    self.original_frequencies.append(freq)
                                    self.frequencies.append(freq)
                                    self.intensities.append(1.0)  # Default intensity
                                    self.activities.append("")   # No activity info
                            except (ValueError, TypeError):
                                pass
                except:
                    pass
        
        # Convert to numpy arrays for easier manipulation
        self.original_frequencies = np.array(self.original_frequencies)
        self.frequencies = np.array(self.frequencies)
        self.intensities = np.array(self.intensities)
        
        return len(self.frequencies) > 0
    
    def plot_mineral_overlay(self):
        """Plot the selected mineral's spectrum as an overlay"""
        if self.current_mineral is None or self.current_mineral not in self.mineral_database:
            return
        
        # Clear previous mineral plot if it exists
        if hasattr(self, 'db_plot') and self.db_plot is not None:
            try:
                self.db_plot.remove()
                self.db_plot = None
            except:
                pass
                
        # Clear previous peak annotations
        for annotation in self.peak_annotations:
            try:
                annotation.remove()
            except:
                pass
        self.peak_annotations = []
        
        # If we have no frequencies, try to extract them
        if len(self.frequencies) == 0:
            if not self.extract_mineral_data():
                messagebox.showinfo("Mineral Data", f"No valid frequency data found for {self.current_mineral}.")
                return
        
        # Normalize intensities to a reasonable scale for overlay
        intensities = np.array(self.intensities)
        max_intensity = np.max(intensities)
        if max_intensity > 0:
            normalized_intensities = intensities / max_intensity
            
            # If we have a user spectrum, scale to match its range
            if self.spectrum_data is not None:
                user_max = np.max(self.spectrum_data['y'])
                normalized_intensities *= user_max * 0.8  # Scale to 80% of user max
        else:
            normalized_intensities = intensities
        
        # Get the peak width from the UI
        peak_width = self.peak_width.get()
        
        # Create a synthetic spectrum using Lorentzian peaks
        # Determine the x range to use
        if self.spectrum_data is not None:
            # Use the same range as the user spectrum with extra padding
            x_min = np.min(self.spectrum_data['x'])
            x_max = np.max(self.spectrum_data['x'])
            padding = (x_max - x_min) * 0.1
            x_min = max(0, x_min - padding)
            x_max = x_max + padding
            x_range = np.linspace(x_min, x_max, 2000)
        else:
            # Default range covering typical Raman shift range
            x_range = np.linspace(0, 4000, 4000)
        
        y_simulated = np.zeros_like(x_range, dtype=float)
        
        # Add a peak for each mode (Lorentzian)
        for freq, intensity in zip(self.frequencies, normalized_intensities):
            y_simulated += intensity * peak_width**2 / ((x_range - freq)**2 + peak_width**2)
        
        # Plot the simulated spectrum
        alignment_text = ""
        if abs(self.shift_value) > 0.1 or abs(self.scale_factor - 1.0) > 0.01:
            alignment_text = f" (Aligned: Shift={self.shift_value:.1f}, Scale={self.scale_factor:.2f})"
            
        self.db_plot = self.ax.plot(x_range, y_simulated, 'r-', alpha=0.7, 
                                    label=f'Database: {self.current_mineral}{alignment_text}')[0]
        
        # Add peak annotations for the most significant peaks
        # Sort peaks by intensity
        peak_data = list(zip(self.frequencies, normalized_intensities, self.activities))
        peak_data.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top peaks (limited to prevent overcrowding)
        top_peaks = peak_data[:min(15, len(peak_data))]
        
        # Only annotate peaks that are above 5% of the maximum intensity
        threshold = 0.05 * np.max(normalized_intensities)
        for freq, intensity, activity in top_peaks:
            if intensity > threshold:
                # Find the corresponding y-value in the simulated spectrum
                idx = np.abs(x_range - freq).argmin()
                y_val = y_simulated[idx]
                
                # Add annotation
                activity_text = f"\n{activity}" if activity.strip() else ""
                annotation = self.ax.annotate(
                    f"{int(freq)} cm⁻¹{activity_text}",
                    xy=(freq, y_val),
                    xytext=(0, 10),  # Offset text 10 points above
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )
                self.peak_annotations.append(annotation)
        
        # Update legend
        self.ax.legend()
        
        # Redraw
        self.figure.tight_layout()
        self.canvas.draw()
    
    def on_manual_shift_change(self, *args):
        """Handle manual shift changes from the UI"""
        if not self.frequencies.size:
            return
            
        # Get values from UI
        self.shift_value = self.shift_var.get()
        self.scale_factor = self.scale_var.get()
        
        # Apply the shift and scale to original frequencies
        self.frequencies = self.original_frequencies * self.scale_factor + self.shift_value
        
        # Update the plot
        self.plot_mineral_overlay()
    
    def reset_alignment(self):
        """Reset all alignment parameters to defaults"""
        if not hasattr(self, 'original_frequencies') or not self.original_frequencies.size:
            return
            
        self.shift_value = 0.0
        self.scale_factor = 1.0
        self.shift_var.set(0.0)
        self.scale_var.set(1.0)
        
        # Reset frequencies to original values
        self.frequencies = np.array(self.original_frequencies)
        
        # Update the plot
        self.plot_mineral_overlay()
        messagebox.showinfo("Alignment Reset", "Peak alignment has been reset to original values.")
    
    def auto_align_peaks(self):
        """Automatically align the database peaks with the measured spectrum"""
        if self.spectrum_data is None:
            messagebox.showinfo("Auto-Align", "Please load a spectrum file first.")
            return
            
        if not hasattr(self, 'original_frequencies') or not self.original_frequencies.size:
            messagebox.showinfo("Auto-Align", "No mineral data to align.")
            return
        
        try:
            # Find peaks in the measured spectrum
            measured_x = self.spectrum_data['x']
            measured_y = self.spectrum_data['y']
            
            # Normalize the spectrum for better peak finding
            measured_y_norm = measured_y / np.max(measured_y)
            
            # Improved peak finding with more sensitive parameters
            # Try different prominence values until we find enough peaks
            prominences = [0.1, 0.05, 0.02, 0.01]
            peaks = []
            
            for prominence in prominences:
                peaks, _ = find_peaks(measured_y_norm, height=prominence, distance=5)
                if len(peaks) >= 3:  # We need at least 3 peaks for a good alignment
                    break
            
            if len(peaks) < 3:
                # Try smoothing the spectrum to find more peaks
                from scipy.signal import savgol_filter
                smoothed_y = savgol_filter(measured_y_norm, window_length=11, polyorder=3)
                for prominence in prominences:
                    peaks, _ = find_peaks(smoothed_y, height=prominence, distance=5)
                    if len(peaks) >= 3:
                        break
            
            if len(peaks) < 3:
                messagebox.showinfo("Auto-Align", 
                                   "Not enough peaks found in measured spectrum for alignment. Try manual alignment.")
                return
                
            measured_peak_positions = measured_x[peaks]
            print(f"Found {len(measured_peak_positions)} peaks in measured spectrum")
            
            # Get the most intense peaks from the database spectrum
            # Normalize intensities
            intensities_norm = self.intensities / np.max(self.intensities)
            
            # Sort by intensity and take the top peaks
            peak_indices = np.argsort(intensities_norm)[::-1]
            theoretical_peak_positions = self.original_frequencies[peak_indices[:min(len(peaks), len(self.original_frequencies))]]
            
            if len(theoretical_peak_positions) < 3:
                messagebox.showinfo("Auto-Align", "Not enough peaks in mineral data for alignment.")
                return
            
            # Objective function for optimization
            def alignment_error(params):
                shift, scale = params
                aligned_peaks = theoretical_peak_positions * scale + shift
                
                # Calculate distances between each theoretical peak and the nearest measured peak
                min_distances = []
                for tp in aligned_peaks:
                    distances = np.abs(measured_peak_positions - tp)
                    min_distances.append(np.min(distances))
                
                # Return the sum of squared minimum distances
                return np.sum(np.array(min_distances)**2)
            
            # Try multiple starting points to avoid local minima
            best_error = float('inf')
            best_params = [0.0, 1.0]  # Default [shift, scale]
            
            starting_shifts = [-50, -20, 0, 20, 50]
            starting_scales = [0.9, 0.95, 1.0, 1.05, 1.1]
            
            for shift in starting_shifts:
                for scale in starting_scales:
                    try:
                        result = minimize(alignment_error, [shift, scale], method='Nelder-Mead')
                        if result.success and result.fun < best_error:
                            best_error = result.fun
                            best_params = result.x
                    except:
                        continue
            
            # Apply the best parameters
            shift, scale = best_params
            
            # Apply the optimized parameters
            self.shift_value = shift
            self.scale_factor = scale
            self.shift_var.set(round(shift, 1))
            self.scale_var.set(round(scale, 2))
            
            # Update frequencies
            self.frequencies = self.original_frequencies * self.scale_factor + self.shift_value
            
            # Update the plot
            self.plot_mineral_overlay()
            
            messagebox.showinfo("Auto-Align", f"Peak alignment complete. Shift: {shift:.1f} cm⁻¹, Scale: {scale:.2f}")
        
        except Exception as e:
            messagebox.showerror("Auto-Align Error", f"Error during alignment: {str(e)}")
    
    def update_plot(self):
        """Update the plot with current settings"""
        if self.current_mineral is not None:
            self.plot_mineral_overlay()
        elif self.spectrum_data is not None:
            self.plot_spectrum()
    
    def clear_mineral_overlay(self):
        """Clear the mineral overlay from the plot"""
        if self.current_mineral is None:
            return
            
        self.current_mineral = None
        self.mineral_label.config(text="No mineral selected")
        
        # Clear frequencies and other data
        self.original_frequencies = np.array([])
        self.frequencies = np.array([])
        self.intensities = np.array([])
        self.activities = []
        
        # Clear annotations
        for annotation in self.peak_annotations:
            try:
                annotation.remove()
            except:
                pass
        self.peak_annotations = []
        
        # Clear the plot and redraw without the mineral data
        if self.spectrum_data is not None:
            self.plot_spectrum()

    # ===== Peak Fitting Methods =====
    
    def lorentzian(self, x, height, center, width):
        """Lorentzian peak function"""
        return height * width**2 / ((x - center)**2 + width**2)
    
    def gaussian(self, x, height, center, width):
        """Gaussian peak function"""
        return height * np.exp(-(x - center)**2 / (2 * width**2))
    
    def voigt(self, x, height, center, sigma, gamma):
        """Voigt peak function (combination of Gaussian and Lorentzian)"""
        from scipy.special import voigt_profile
        return height * voigt_profile(x - center, sigma, gamma)
    
    def multi_peak_model(self, x, *params):
        """Model for multiple peaks (Lorentzian or Gaussian)"""
        peak_shape = self.peak_shape_var.get()
        y = np.zeros_like(x)
        
        if peak_shape == "Voigt":
            # For Voigt profile, we have 4 parameters per peak: height, center, sigma, gamma
            num_peaks = len(params) // 4
            for i in range(num_peaks):
                idx = i * 4
                height, center, sigma, gamma = params[idx:idx+4]
                y += self.voigt(x, height, center, sigma, gamma)
        else:
            # For Lorentzian or Gaussian, we have 3 parameters per peak: height, center, width
            num_peaks = len(params) // 3
            for i in range(num_peaks):
                idx = i * 3
                height, center, width = params[idx:idx+3]
                
                if peak_shape == "Lorentzian":
                    y += self.lorentzian(x, height, center, width)
                else:  # "Gaussian"
                    y += self.gaussian(x, height, center, width)
        
        return y
    
    def on_mouse_press(self, event):
        """Handle mouse press events for range selection"""
        if self.selecting_range and event.inaxes == self.fit_ax:
            # Record the starting x-coordinate
            self.range_start_x = event.xdata
            
            # Clear any existing rectangle
            if self.range_rect:
                self.range_rect.remove()
                self.range_rect = None
            
            # Create a new rectangle starting at this point
            rect = Rectangle((event.xdata, 0), 0.1, 1, color='gray', alpha=0.3, transform=self.fit_ax.get_xaxis_transform())
            self.range_rect = self.fit_ax.add_patch(rect)
            self.fit_canvas.draw()
    
    def on_mouse_move(self, event):
        """Handle mouse movement events for range selection"""
        if self.selecting_range and self.range_rect and event.inaxes == self.fit_ax and hasattr(self, 'range_start_x'):
            # Update the width of the rectangle
            x_width = event.xdata - self.range_start_x
            self.range_rect.set_x(min(self.range_start_x, event.xdata))
            self.range_rect.set_width(abs(x_width))
            self.fit_canvas.draw()
    
    def on_mouse_release(self, event):
        """Handle mouse release events to finalize range selection"""
        if self.selecting_range and event.inaxes == self.fit_ax and hasattr(self, 'range_start_x'):
            # Store the selected range (ensuring start < end)
            start_x = min(self.range_start_x, event.xdata)
            end_x = max(self.range_start_x, event.xdata)
            
            # Ensure we have a minimum range width
            if end_x - start_x < 5:  # Minimum range of 5 cm-1
                end_x = start_x + 5
            
            # Add this region to our list
            self.fitting_regions.append((start_x, end_x))
            region_idx = len(self.fitting_regions) - 1
            
            # Choose a color for this region
            colors = list(mcolors.TABLEAU_COLORS)
            color = colors[region_idx % len(colors)]
            
            # Update the rectangle to show the final selection with the region's color
            if self.range_rect:
                self.range_rect.remove()
            
            # Create a new permanent rectangle for this region
            rect = Rectangle((start_x, 0), end_x - start_x, 1, 
                            color=color, alpha=0.2, 
                            transform=self.fit_ax.get_xaxis_transform())
            region_rect = self.fit_ax.add_patch(rect)
            self.region_rectangles.append(region_rect)
            
            # Add a label for the region
            region_label = self.fit_ax.text(
                (start_x + end_x) / 2, 0.95, f"Region {region_idx+1}", 
                transform=self.fit_ax.get_xaxis_transform(),
                horizontalalignment='center',
                color=color, fontweight='bold'
            )
            self.region_labels.append(region_label)
            
            # Update the region combobox
            region_values = ["All"] + [f"Region {i+1}" for i in range(len(self.fitting_regions))]
            self.region_combo['values'] = region_values
            
            # Select the newly added region
            self.active_region_var.set(f"Region {region_idx+1}")
            
            # Exit selection mode
            self.selecting_range = False
            self.root.config(cursor="")
            
            # Show info about selected range
            self.fit_results_text.delete(1.0, tk.END)
            self.fit_results_text.insert(tk.END, f"Added Region {region_idx+1}:\n{start_x:.1f} - {end_x:.1f} cm⁻¹\n\n")
            self.fit_results_text.insert(tk.END, f"Now you can adjust number of peaks\nand type, then click 'Perform Peak Fitting'")
            
            # Reset temporary selection rectangle
            self.range_rect = None
            
            # Redraw canvas
            self.fit_canvas.draw()
    
    def start_range_selection(self):
        """Start the range selection mode for adding a new fitting region"""
        if self.spectrum_data is None:
            messagebox.showinfo("Range Selection", "Please import a spectrum first.")
            return
        
        self.selecting_range = True
        self.root.config(cursor="crosshair")
        
        # Show instructions
        messagebox.showinfo("Range Selection", 
                           "Click and drag on the spectrum to select a region for peak fitting.")
    
    def clear_all_regions(self):
        """Clear all fitting regions"""
        # Remove all rectangles and labels
        for rect in self.region_rectangles:
            rect.remove()
        self.region_rectangles = []
        
        for label in self.region_labels:
            label.remove()
        self.region_labels = []
        
        # Clear temporary rectangle if it exists
        if self.range_rect:
            self.range_rect.remove()
            self.range_rect = None
        
        # Clear region data
        self.fitting_regions = []
        self.fitted_regions = {}
        
        # Reset combobox
        self.region_combo['values'] = ["All"]
        self.active_region_var.set("All")
        
        # Clear results
        self.fit_results_text.delete(1.0, tk.END)
        self.fit_results_text.insert(tk.END, "All regions cleared.")
        
        # Clear any existing peak fitting display
        self.clear_peak_fitting()
        
        # Redraw canvas
        self.fit_canvas.draw()
    
    def on_region_selected(self, event):
        """Handle region selection from dropdown"""
        region_label = self.active_region_var.get()
        
        # Update the UI to show parameters for the selected region
        if region_label == "All":
            # Show combined results
            self.show_all_regions_results()
        else:
            # Extract region index
            try:
                region_idx = int(region_label.split()[1]) - 1
                if region_idx < len(self.fitting_regions):
                    # Show specific region results
                    self.show_region_results(region_idx)
            except:
                pass
    
    def show_all_regions_results(self):
        """Show results for all regions"""
        self.fit_results_text.delete(1.0, tk.END)
        self.fit_results_text.insert(tk.END, "Combined Results for All Regions:\n\n")
        
        peak_count = 0
        for region_idx, region_data in self.fitted_regions.items():
            if 'peaks' in region_data:
                start_x, end_x = self.fitting_regions[region_idx]
                self.fit_results_text.insert(tk.END, f"Region {region_idx+1} ({start_x:.1f}-{end_x:.1f} cm⁻¹):\n")
                
                for i, peak in enumerate(region_data['peaks']):
                    peak_count += 1
                    self.fit_results_text.insert(tk.END, f"  Peak {i+1}: {peak['center']:.1f} cm⁻¹\n")
                
                self.fit_results_text.insert(tk.END, "\n")
    
    def show_region_results(self, region_idx):
        """Show results for a specific region"""
        if region_idx not in self.fitted_regions:
            self.fit_results_text.delete(1.0, tk.END)
            self.fit_results_text.insert(tk.END, f"No fitting results for Region {region_idx+1}")
            return
            
        region_data = self.fitted_regions[region_idx]
        start_x, end_x = self.fitting_regions[region_idx]
        
        self.fit_results_text.delete(1.0, tk.END)
        self.fit_results_text.insert(tk.END, f"Results for Region {region_idx+1}:\n")
        self.fit_results_text.insert(tk.END, f"Range: {start_x:.1f} - {end_x:.1f} cm⁻¹\n\n")
        
        if 'peaks' in region_data:
            peak_shape = region_data.get('shape', 'Lorentzian')
            self.fit_results_text.insert(tk.END, f"Peak Shape: {peak_shape}\n\n")
            
            for i, peak in enumerate(region_data['peaks']):
                self.fit_results_text.insert(tk.END, f"Peak {i+1}:\n")
                for param, value in peak.items():
                    self.fit_results_text.insert(tk.END, f"  {param}: {value:.2f}\n")
                self.fit_results_text.insert(tk.END, "\n")
    
    def perform_peak_fitting(self):
        """Perform peak fitting on the selected regions"""
        if self.spectrum_data is None:
            messagebox.showinfo("Peak Fitting", "Please import a spectrum first.")
            return
        
        if not self.fitting_regions:
            messagebox.showinfo("Peak Fitting", "Please select at least one region to fit.")
            return
        
        # Clear previous peak fitting displays
        self.clear_peak_fitting()
        
        # Get the active region selection
        region_selection = self.active_region_var.get()
        
        if region_selection == "All":
            # Fit all regions
            for region_idx in range(len(self.fitting_regions)):
                self.fit_single_region(region_idx)
        else:
            # Fit just the selected region
            try:
                region_idx = int(region_selection.split()[1]) - 1
                if region_idx < len(self.fitting_regions):
                    self.fit_single_region(region_idx)
            except:
                messagebox.showerror("Peak Fitting Error", "Invalid region selection.")
        
        # Update the results display
        self.on_region_selected(None)  # Refresh the results display
        
        # Compare with database peaks if available
        if self.current_mineral is not None:
            self.compare_fitted_peaks_with_database()
    
    def fit_single_region(self, region_idx):
        """Fit peaks in a single region"""
        try:
            # Get the region boundaries
            start_x, end_x = self.fitting_regions[region_idx]
            
            # Get the data within the selected range
            x_data = self.spectrum_data['x']
            y_data = self.spectrum_data['y']
            
            # Filter the data to the selected range
            mask = (x_data >= start_x) & (x_data <= end_x)
            x_range = x_data[mask]
            y_range = y_data[mask]
            
            if len(x_range) < 5:
                messagebox.showinfo("Peak Fitting", f"Not enough data points in Region {region_idx+1}.")
                return False
            
            # Get the number of peaks to fit
            num_peaks = self.num_peaks_var.get()
            peak_shape = self.peak_shape_var.get()
            
            # Find initial guesses for peak parameters
            # First, find local maxima in the range
            peaks, _ = find_peaks(y_range, height=0.1*np.max(y_range), distance=5)
            
            # If we found fewer peaks than requested, we'll estimate the rest
            if len(peaks) < num_peaks:
                # Evenly space the peaks across the range
                peak_indices = np.linspace(0, len(x_range)-1, num_peaks, dtype=int)
            else:
                # Sort peaks by height and take the top num_peaks
                peak_heights = y_range[peaks]
                sorted_indices = np.argsort(peak_heights)[::-1]  # Sort descending
                peak_indices = peaks[sorted_indices[:num_peaks]]
            
            # Prepare initial parameters for the fit
            initial_params = []
            
            if peak_shape == "Voigt":
                # For Voigt: height, center, sigma, gamma for each peak
                for idx in peak_indices:
                    height = y_range[idx]
                    center = x_range[idx]
                    # Initial guess for width parameters
                    sigma = 3.0  # Gaussian component
                    gamma = 5.0  # Lorentzian component
                    initial_params.extend([height, center, sigma, gamma])
            else:
                # For Lorentzian or Gaussian: height, center, width for each peak
                for idx in peak_indices:
                    height = y_range[idx]
                    center = x_range[idx]
                    width = 5.0  # Initial guess for width
                    initial_params.extend([height, center, width])
            
            # Perform the curve fitting
            params, covariance = curve_fit(self.multi_peak_model, x_range, y_range, 
                                         p0=initial_params, maxfev=10000)
            
            # Store fitted parameters for this region
            self.fitted_regions[region_idx] = {
                'shape': peak_shape,
                'params': params,
                'peaks': []
            }
            
            # Generate a dense x-array for smooth plotting
            x_dense = np.linspace(start_x, end_x, 1000)
            
            # Plot the overall fit
            y_fit = self.multi_peak_model(x_dense, *params)
            colors = list(mcolors.TABLEAU_COLORS)
            region_color = colors[region_idx % len(colors)]
            
            fit_plot = self.fit_ax.plot(
                x_dense, y_fit, '-', 
                color=region_color, linewidth=2, 
                label=f'Fit Region {region_idx+1}'
            )[0]
            self.peak_components.append(fit_plot)
            
            # Extract and store peak parameters
            if peak_shape == "Voigt":
                # 4 parameters per peak: height, center, sigma, gamma
                for i in range(num_peaks):
                    idx = i * 4
                    height, center, sigma, gamma = params[idx:idx+4]
                    
                    # Store peak info
                    self.fitted_regions[region_idx]['peaks'].append({
                        'height': height,
                        'center': center,
                        'sigma': sigma,
                        'gamma': gamma
                    })
                    
                    # Plot the individual peak
                    y_peak = self.voigt(x_dense, height, center, sigma, gamma)
                    peak_plot = self.fit_ax.plot(
                        x_dense, y_peak, '--', 
                        color=region_color, alpha=0.5
                    )[0]
                    self.peak_components.append(peak_plot)
                    
                    # Add a marker at the peak position
                    marker = self.fit_ax.axvline(
                        x=center, ymin=0, ymax=0.05, 
                        color=region_color, linewidth=1.5
                    )
                    self.fitted_peak_markers.append(marker)
                    
                    # Add a text label for the peak
                    text = self.fit_ax.text(
                        center, self.fit_ax.get_ylim()[1] * (0.1 + 0.05*i), 
                        f"{center:.1f}", 
                        color=region_color,
                        horizontalalignment='center',
                        fontsize=8
                    )
                    self.fitted_peak_markers.append(text)
            else:
                # 3 parameters per peak: height, center, width
                for i in range(num_peaks):
                    idx = i * 3
                    height, center, width = params[idx:idx+3]
                    
                    # Store peak info
                    self.fitted_regions[region_idx]['peaks'].append({
                        'height': height,
                        'center': center,
                        'width': width
                    })
                    
                    # Plot the individual peak
                    if peak_shape == "Lorentzian":
                        y_peak = self.lorentzian(x_dense, height, center, width)
                    else:  # "Gaussian"
                        y_peak = self.gaussian(x_dense, height, center, width)
                    
                    peak_plot = self.fit_ax.plot(
                        x_dense, y_peak, '--', 
                        color=region_color, alpha=0.5
                    )[0]
                    self.peak_components.append(peak_plot)
                    
                    # Add a marker at the peak position
                    marker = self.fit_ax.axvline(
                        x=center, ymin=0, ymax=0.05, 
                        color=region_color, linewidth=1.5
                    )
                    self.fitted_peak_markers.append(marker)
                    
                    # Add a text label for the peak
                    text = self.fit_ax.text(
                        center, self.fit_ax.get_ylim()[1] * (0.1 + 0.05*i), 
                        f"{center:.1f}", 
                        color=region_color,
                        horizontalalignment='center',
                        fontsize=8
                    )
                    self.fitted_peak_markers.append(text)
            
            # Redraw the canvas
            self.fit_figure.tight_layout()
            self.fit_canvas.draw()
            
            return True
            
        except Exception as e:
            messagebox.showerror("Peak Fitting Error", f"Error fitting Region {region_idx+1}: {str(e)}")
            return False
    
    def compare_fitted_peaks_with_database(self):
        """Compare all fitted peak positions with database entries and annotate plot/results"""
        if self.current_mineral is None or not self.fitted_regions:
            return  # No peaks to compare or no mineral selected

        try:
            # Remove previous fitted peak annotations (text labels)
            for marker in self.fitted_peak_markers:
                try:
                    marker.remove()
                except Exception:
                    pass
            self.fitted_peak_markers = []

            # Prepare to annotate on the plot
            ylim = self.fit_ax.get_ylim()

            # For each region, process all fitted peaks
            peak_counter = 1
            self.fit_results_text.insert(tk.END, "\nMatched Peaks to Database:\n")
            for region_idx, region_data in self.fitted_regions.items():
                if 'peaks' in region_data:
                    for i, peak in enumerate(region_data['peaks']):
                        if 'center' in peak:
                            fitted_pos = peak['center']
                            # Find the closest database peak
                            distances = np.abs(self.frequencies - fitted_pos)
                            closest_idx = np.argmin(distances)
                            db_pos = self.frequencies[closest_idx]
                            db_activity = self.activities[closest_idx] if closest_idx < len(self.activities) else ""

                            # Annotate on the plot with the Raman character
                            char_label = db_activity if db_activity else "?"
                            # Place annotation above the peak
                            y_annot = ylim[1] * (0.12 + 0.05*i)
                            text = self.fit_ax.text(
                                fitted_pos, y_annot,
                                f"{char_label}",
                                color="black",  # Use black for visibility
                                horizontalalignment='center',
                                fontsize=10,
                                fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7)
                            )
                            self.fitted_peak_markers.append(text)

                            # Add a marker at the peak position
                            marker = self.fit_ax.axvline(
                                x=fitted_pos, ymin=0, ymax=0.05,
                                color="black", linewidth=1.5, linestyle=':')
                            self.fitted_peak_markers.append(marker)

                            # Add to results text
                            self.fit_results_text.insert(
                                tk.END,
                                f"Peak {peak_counter}: Fitted {fitted_pos:.1f} cm⁻¹\n"
                                f"  DB: {db_pos:.1f} cm⁻¹ (Character: {char_label})\n"
                            )
                            peak_counter += 1
            # Redraw the canvas to show new annotations
            self.fit_figure.tight_layout()
            self.fit_canvas.draw()
        except Exception as e:
            print(f"Error comparing with database: {str(e)}")
    
    def clear_peak_fitting(self):
        """Clear all peak fitting results"""
        # Clear the fit plot
        if hasattr(self, 'peak_fit_plot') and self.peak_fit_plot:
            self.peak_fit_plot.remove()
            self.peak_fit_plot = None
        
        # Clear individual peak components
        for component in self.peak_components:
            component.remove()
        self.peak_components = []
        
        # Clear peak markers
        for marker in self.fitted_peak_markers:
            marker.remove()
        self.fitted_peak_markers = []
        
        # Keep the regions but clear the fitting results
        # self.fitted_regions = {}
        
        # Redraw the canvas
        if hasattr(self, 'fit_canvas'):
            self.fit_canvas.draw()

    def clear_selected_range(self):
        """DEPRECATED: Use clear_all_regions instead"""
        self.clear_all_regions()

    def setup_crystal_orientation_tab(self):
        """Set up the crystal orientation tab"""
        self.orientation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.orientation_tab, text="Crystal Orientation")
        
    def setup_3d_tensor_tab(self):
        """Setup the 3D tensor visualization tab"""
        # Create the tab
        self.tensor_3d_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tensor_3d_tab, text="3D Tensor")
        
        # Get optimized orientation if available, otherwise use current orientation
        if hasattr(self, 'optimized_angles'):
            optimized_orientation = self.optimized_angles
        elif hasattr(self, 'phi_var') and hasattr(self, 'theta_var') and hasattr(self, 'psi_var'):
            # Use current orientation from UI controls
            optimized_orientation = (self.phi_var.get(), self.theta_var.get(), self.psi_var.get())
        else:
            optimized_orientation = (0, 0, 0)
            
        # Create the visualizer
        self.tensor_3d_visualizer = RamanTensor3DVisualizer(
            self.tensor_3d_tab,
            crystal_structure=self.current_structure if hasattr(self, 'current_structure') else None,
            optimized_orientation=optimized_orientation
        )
        
        # Pack the visualizer frame
        self.tensor_3d_visualizer.frame.pack(fill=tk.BOTH, expand=True)
        
        # Override apply_to_main_view method to sync with main application
        def apply_to_main_view():
            """Apply the current orientation to the main application"""
            if not hasattr(self, 'phi_var') or not hasattr(self, 'theta_var') or not hasattr(self, 'psi_var'):
                messagebox.showinfo("Error", "Main application is not ready for orientation update.")
                return
                
            # Get current orientation from 3D visualizer
            phi = self.tensor_3d_visualizer.phi_var.get()
            theta = self.tensor_3d_visualizer.theta_var.get()
            psi = self.tensor_3d_visualizer.psi_var.get()
            
            # Update orientation in main application
            self.phi_var.set(phi)
            self.theta_var.set(theta)
            self.psi_var.set(psi)
            
            # Recalculate Raman spectrum with new orientation
            self.calculate_orientation_raman_spectrum()
            
            messagebox.showinfo("Orientation", f"Applied orientation (φ={phi}°, θ={theta}°, ψ={psi}°)")
        
        # Replace the method
        self.tensor_3d_visualizer.apply_to_main_view = apply_to_main_view
        
        # Add tab change listener to sync mineral selection
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Create a PanedWindow to allow resizing between panels
        main_paned = ttk.PanedWindow(self.orientation_tab, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for controls
        left_frame = ttk.Frame(main_paned, width=300)
        left_frame.pack_propagate(False)  # Prevent the frame from shrinking
        
        # Right frame for results and plots
        right_frame = ttk.Frame(main_paned)
        
        # Add frames to the paned window
        main_paned.add(left_frame, weight=0)  # Fixed weight for left panel
        main_paned.add(right_frame, weight=1)  # Expandable weight for right panel

        # === Left frame - Input controls ===
        # CIF file selection
        ttk.Label(left_frame, text="Select CIF File:").pack(anchor=tk.W)
        cif_frame = ttk.Frame(left_frame)
        cif_frame.pack(fill=tk.X, pady=5)
        
        self.cif_path_var = tk.StringVar()
        cif_entry = ttk.Entry(cif_frame, textvariable=self.cif_path_var, width=30)
        cif_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(cif_frame, text="Browse", command=self.browse_cif_file).pack(side=tk.LEFT)

        # Mineral selection
        ttk.Label(left_frame, text="\nSelect Mineral:").pack(anchor=tk.W, pady=(10,0))
        self.orientation_mineral_var = tk.StringVar()
        mineral_names = sorted(self.mineral_database.keys()) if self.mineral_database else []
        self.orientation_mineral_combo = ttk.Combobox(left_frame, textvariable=self.orientation_mineral_var, values=mineral_names, width=30, state="readonly")
        self.orientation_mineral_combo.pack(anchor=tk.W, padx=5, pady=5)
        if mineral_names:
            self.orientation_mineral_combo.current(0)
            
        # Sync mineral selection with analyzer tab if possible
        if hasattr(self, 'current_mineral') and self.current_mineral:
            if self.current_mineral in mineral_names:
                self.orientation_mineral_var.set(self.current_mineral)

        # Polarization selection - horizontal layout
        ttk.Label(left_frame, text="\nPolarization Configuration:").pack(anchor=tk.W, pady=(10,0))
        
        # Horizontal frame for polarization options
        polarization_frame = ttk.Frame(left_frame)
        polarization_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.polarization_var = tk.StringVar(value="VV")
        ttk.Radiobutton(polarization_frame, text="VV", variable=self.polarization_var, value="VV").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(polarization_frame, text="VH", variable=self.polarization_var, value="VH").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(polarization_frame, text="HV", variable=self.polarization_var, value="HV").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(polarization_frame, text="HH", variable=self.polarization_var, value="HH").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(polarization_frame, text="Unpolarized", variable=self.polarization_var, value="UNPOL").pack(side=tk.LEFT, padx=10)

        # Initial orientation controls
        ttk.Label(left_frame, text="\nCrystal Orientation (Euler angles):").pack(anchor=tk.W, pady=(10,0))
        
        angle_frame = ttk.Frame(left_frame)
        angle_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(angle_frame, text="φ:").pack(side=tk.LEFT, padx=(20,2))
        self.phi_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(angle_frame, from_=0, to=360, increment=5, textvariable=self.phi_var, width=5).pack(side=tk.LEFT)
        
        ttk.Label(angle_frame, text="θ:").pack(side=tk.LEFT, padx=(10,2))
        self.theta_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(angle_frame, from_=0, to=180, increment=5, textvariable=self.theta_var, width=5).pack(side=tk.LEFT)
        
        ttk.Label(angle_frame, text="ψ:").pack(side=tk.LEFT, padx=(10,2))
        self.psi_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(angle_frame, from_=0, to=360, increment=5, textvariable=self.psi_var, width=5).pack(side=tk.LEFT)

        # Main action buttons on the same line
        action_buttons_frame = ttk.Frame(left_frame)
        action_buttons_frame.pack(fill=tk.X, pady=5)
        ttk.Button(action_buttons_frame, text="Parse CIF", command=self.display_cif_info).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(action_buttons_frame, text="Calculate Raman Tensor", command=self.calculate_orientation_raman_tensor).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_buttons_frame, text="Calculate Raman Spectrum", command=self.calculate_orientation_raman_spectrum).pack(side=tk.LEFT, padx=5)
        
        # Add calibration controls before calibration buttons
        ttk.Label(left_frame, text="\nSpectrum Calibration:").pack(anchor=tk.W, pady=(10,0))
        
        calib_frame = ttk.Frame(left_frame)
        calib_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(calib_frame, text="Shift:").pack(side=tk.LEFT, padx=(20,2))
        self.orientation_shift_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(calib_frame, from_=-100, to=100, increment=1, textvariable=self.orientation_shift_var, width=6).pack(side=tk.LEFT)
        
        ttk.Label(calib_frame, text="Scale:").pack(side=tk.LEFT, padx=(10,2))
        self.orientation_scale_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(calib_frame, from_=0.8, to=1.2, increment=0.01, textvariable=self.orientation_scale_var, width=6).pack(side=tk.LEFT)
        
        # Calibration buttons after calibration controls
        calibration_buttons_frame = ttk.Frame(left_frame)
        calibration_buttons_frame.pack(fill=tk.X, pady=5)
        ttk.Button(calibration_buttons_frame, text="Auto-Align Spectrum", command=self.auto_align_orientation_spectrum).pack(side=tk.LEFT, padx=(5,5))
        ttk.Button(calibration_buttons_frame, text="Reset Alignment", command=self.reset_orientation_alignment).pack(side=tk.LEFT, padx=5)
        
        # Add optimization method selection
        ttk.Label(left_frame, text="\nOptimization Method:").pack(anchor=tk.W, pady=(10,0))
        
        # Frame for optimization options
        opt_method_frame = ttk.Frame(left_frame)
        opt_method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.opt_method_var = tk.StringVar(value="basin")
        ttk.Radiobutton(opt_method_frame, text="Basin-hopping", variable=self.opt_method_var, value="basin").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(opt_method_frame, text="Dual Annealing", variable=self.opt_method_var, value="annealing").pack(side=tk.LEFT, padx=5)
        
        # Optimization parameters frame
        opt_params_frame = ttk.Frame(left_frame)
        opt_params_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(opt_params_frame, text="Iterations:").pack(side=tk.LEFT, padx=(5,2))
        self.opt_iter_var = tk.IntVar(value=100)
        ttk.Spinbox(opt_params_frame, from_=20, to=1000, increment=10, textvariable=self.opt_iter_var, width=5).pack(side=tk.LEFT)
        
        ttk.Label(opt_params_frame, text="Seeds:").pack(side=tk.LEFT, padx=(10,2))
        self.opt_seeds_var = tk.IntVar(value=5)
        ttk.Spinbox(opt_params_frame, from_=1, to=20, increment=1, textvariable=self.opt_seeds_var, width=3).pack(side=tk.LEFT)
        
        # Optimization buttons
        opt_buttons_frame = ttk.Frame(left_frame)
        opt_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(opt_buttons_frame, text="Optimize Orientation", command=self.optimize_crystal_orientation).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(opt_buttons_frame, text="Refine Peak Fitting", command=self.refine_peak_fitting).pack(side=tk.LEFT, padx=5)

        # Info display
        ttk.Label(left_frame, text="\nCIF Lattice and Atoms:").pack(anchor=tk.W, pady=(10,0))
        self.cif_info_text = tk.Text(left_frame, height=6, width=40)
        self.cif_info_text.pack(fill=tk.X, padx=5, pady=5)

        # === Right frame - Results and plots ===
        # Output display
        ttk.Label(right_frame, text="Raman Tensor:").pack(anchor=tk.W)
        self.orientation_raman_text = tk.Text(right_frame, height=4, width=40)
        self.orientation_raman_text.pack(fill=tk.X, padx=5, pady=5)

        # Results display
        ttk.Label(right_frame, text="Optimization Results:").pack(anchor=tk.W, pady=(10,0))
        self.orientation_results_text = tk.Text(right_frame, height=8, width=40)
        self.orientation_results_text.pack(fill=tk.X, padx=5, pady=5)

        # Plot frame
        ttk.Label(right_frame, text="Raman Spectrum:").pack(anchor=tk.W, pady=(10,0))
        self.orientation_plot_frame = ttk.Frame(right_frame)
        self.orientation_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the orientation spectrum plot
        self.create_orientation_plot()
    
    def create_orientation_plot(self):
        """Create the plot for orientation-dependent Raman spectrum"""
        # Create figure and canvas
        self.orientation_figure = Figure(figsize=(5, 4), dpi=100)
        self.orientation_canvas = FigureCanvasTkAgg(self.orientation_figure, self.orientation_plot_frame)
        self.orientation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.orientation_canvas, self.orientation_plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create subplot layout: main spectrum and difference plot
        gs = self.orientation_figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        
        # Create main spectrum axis
        self.orientation_ax = self.orientation_figure.add_subplot(gs[0])
        self.orientation_ax.set_title("Raman Spectrum Comparison")
        self.orientation_ax.set_xlabel("")  # No xlabel for top plot
        self.orientation_ax.set_ylabel("Intensity")
        
        # Create difference plot axis
        self.difference_ax = self.orientation_figure.add_subplot(gs[1], sharex=self.orientation_ax)
        self.difference_ax.set_title("")
        self.difference_ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.difference_ax.set_ylabel("Diff")
        
        self.orientation_figure.tight_layout()
    
    def calculate_orientation_raman_spectrum(self):
        """Calculate and plot the Raman spectrum for the current orientation"""
        try:
            # Check if we have a crystal structure
            if not hasattr(self, 'current_structure') or self.current_structure is None:
                messagebox.showinfo("Raman Spectrum", "Please load and parse a CIF file first.")
                return
                
            # Get the current orientation parameters
            orientation = (self.phi_var.get(), self.theta_var.get(), self.psi_var.get())
            polarization = self.polarization_var.get()
            
            # Get the mineral data for frequencies
            mineral = self.orientation_mineral_var.get()
            if mineral in self.mineral_database:
                # Extract frequencies from the mineral database
                self.extract_mineral_data_for_orientation(mineral)
                
                # Pass frequencies to the crystal structure object
                self.current_structure.frequencies = self.frequencies
                self.current_structure.intensities = self.intensities
                self.current_structure.activities = self.activities
            
            # Calculate the spectrum
            spectrum = self.current_structure.calculate_raman_spectrum(orientation, polarization)
            
            # Clear the plot
            self.orientation_ax.clear()
            self.difference_ax.clear()
            
            # Extract data for plotting
            freqs, intensities, characters = zip(*spectrum) if spectrum else ([], [], [])
            
            # Get calibration parameters
            shift = self.orientation_shift_var.get()
            scale_factor = self.orientation_scale_var.get()
            
            # Apply calibration to frequencies for display
            displayed_freqs = [freq * scale_factor + shift for freq in freqs]
            
            # Create a high-resolution x-range for a smooth spectrum
            x_min = 50  # Minimum wavenumber to display
            x_max = 1500  # Maximum wavenumber to display
            
            # Adjust range if we have data
            if displayed_freqs:
                data_min = min(displayed_freqs) - 50
                data_max = max(displayed_freqs) + 50
                x_min = max(0, min(x_min, data_min))
                x_max = max(x_max, data_max)
            
            x_range = np.linspace(x_min, x_max, 1000)
            
            # Create a synthetic spectrum with Lorentzian peaks
            width = 10.0  # peak width
            y_calculated = np.zeros_like(x_range)
            for freq, intensity, _ in zip(displayed_freqs, intensities, characters):
                y_calculated += intensity * width**2 / ((x_range - freq)**2 + width**2)
            
            # Plot the synthetic spectrum
            self.orientation_ax.plot(x_range, y_calculated, 'r-', linewidth=2, label=f'Calculated ({polarization})')
            
            # Plot the individual peak positions
            for freq, intensity, character in zip(displayed_freqs, intensities, characters):
                relative_height = max(y_calculated) * 0.8 * intensity / max(intensities) if max(intensities) > 0 else 0
                self.orientation_ax.stem([freq], [relative_height], linefmt='r-', markerfmt='ro', basefmt=' ')
                
                # Add peak labels with both frequency and character
                y_pos = max(y_calculated) * 0.9 * intensity / max(intensities) if max(intensities) > 0 else 0
                self.orientation_ax.text(freq, y_pos, f"{int(freq)}\n{character}", ha='center', fontsize=8)
            
            # Generate fitted spectrum from peak fitting tab
            y_fitted = np.zeros_like(x_range)
            fitted_peaks_info = []
            
            if hasattr(self, 'fitted_regions') and self.fitted_regions:
                for region_idx, region_data in self.fitted_regions.items():
                    if 'peaks' in region_data:
                        shape = region_data.get('shape', 'Lorentzian')
                        
                        for peak in region_data['peaks']:
                            if 'center' in peak:
                                center = peak['center']
                                height = peak.get('height', 1.0)
                                width = peak.get('width', 10.0)
                                
                                # Find character if available
                                character = ""
                                for marker in self.fitted_peak_markers:
                                    if hasattr(marker, 'get_text') and callable(marker.get_text):
                                        text = marker.get_text()
                                        if hasattr(marker, 'get_position') and callable(marker.get_position):
                                            marker_pos = marker.get_position()
                                            if isinstance(marker_pos, tuple) and len(marker_pos) >= 1:
                                                if abs(marker_pos[0] - center) < 5:  # Within 5 cm-1
                                                    if len(text) <= 3:  # Character labels are typically short
                                                        character = text
                                
                                fitted_peaks_info.append((center, height, width, character, shape))
                
                # Now generate the fitted spectrum
                for center, height, width, character, shape in fitted_peaks_info:
                    if shape == 'Lorentzian':
                        y_fitted += height * width**2 / ((x_range - center)**2 + width**2)
                    elif shape == 'Gaussian':
                        y_fitted += height * np.exp(-(x_range - center)**2 / (2 * width**2))
                    else:  # Voigt or other - use Lorentzian as fallback
                        y_fitted += height * width**2 / ((x_range - center)**2 + width**2)
                
                # Normalize the fitted spectrum to a similar scale as calculated
                if np.max(y_fitted) > 0:
                    scaling_factor = max(y_calculated) / max(y_fitted) if max(y_fitted) > 0 else 1.0
                    y_fitted *= scaling_factor
                
                # Plot the fitted spectrum
                self.orientation_ax.plot(x_range, y_fitted, 'b-', linewidth=1.5, alpha=0.7, label='Fitted Spectrum')
                
                # Add vertical lines and labels for the fitted peaks
                for center, height, width, character, _ in fitted_peaks_info:
                    # Add a small vertical line at the peak position
                    self.orientation_ax.axvline(x=center, ymin=0, ymax=0.2, color='blue', linestyle='--', linewidth=1)
                    
                    # Add text label
                    if character:
                        y_pos = max(y_calculated) * 0.25
                        self.orientation_ax.text(center, y_pos, f"{int(center)}\n({character})", color='blue',
                                              ha='center', fontsize=8,
                                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Calculate and plot the difference
            y_diff = y_calculated - y_fitted
            self.difference_ax.plot(x_range, y_diff, 'k-', linewidth=1)
            self.difference_ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            
            # Label the difference plot
            self.difference_ax.set_ylabel("Diff")
            self.difference_ax.set_xlabel("Wavenumber (cm⁻¹)")
            
            # Fill areas for visual emphasis of differences
            self.difference_ax.fill_between(x_range, y_diff, 0, where=(y_diff > 0), 
                                          color='red', alpha=0.3, interpolate=True)
            self.difference_ax.fill_between(x_range, y_diff, 0, where=(y_diff < 0), 
                                          color='blue', alpha=0.3, interpolate=True)
            
            # Add title and labels to main plot
            self.orientation_ax.set_title(f"Raman Spectrum - {polarization} Polarization")
            self.orientation_ax.set_ylabel("Intensity")
            
            # Add legend to main plot
            self.orientation_ax.legend()
            
            # Set shared x-range for both plots
            self.orientation_ax.set_xlim(x_min, x_max)
            
            # Hide x-labels on top plot since they're shared
            plt.setp(self.orientation_ax.get_xticklabels(), visible=False)
            
            # Adjust layout
            self.orientation_figure.tight_layout()
            self.orientation_canvas.draw()
            
            # Return the calculated data
            return spectrum
            
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Error calculating Raman spectrum: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_mineral_data_for_orientation(self, mineral_name):
        """Extract frequency and activity data for the orientation calculation"""
        if mineral_name not in self.mineral_database:
            return False
            
        # This is similar to extract_mineral_data but specifically for orientation calculations
        mineral_data = self.mineral_database[mineral_name]
        
        # Clear previous data
        self.frequencies = []
        self.intensities = []
        self.activities = []
        
        # Process the mineral data to extract frequencies and activities
        if 'modes' in mineral_data and mineral_data['modes'] is not None:
            modes = mineral_data['modes']
            
            # Handle case where modes is a pandas DataFrame
            if isinstance(modes, pd.DataFrame):
                if 'TO_Frequency' in modes.columns:
                    for _, row in modes.iterrows():
                        try:
                            freq = float(row['TO_Frequency'])
                            intensity = float(row['I_Total']) if 'I_Total' in row and pd.notna(row['I_Total']) else 1.0
                            activity = str(row['Activity']) if 'Activity' in row and pd.notna(row['Activity']) else ""
                            
                            if freq > 0:
                                self.frequencies.append(freq)
                                self.intensities.append(intensity)
                                self.activities.append(activity)
                        except (ValueError, TypeError):
                            pass
            # Handle case where modes is a list of tuples
            elif isinstance(modes, list):
                for mode in modes:
                    # Check if this is a tuple with at least 2 elements (freq, activity, [intensity])
                    if isinstance(mode, tuple) and len(mode) >= 2:
                        try:
                            freq = float(mode[0])
                            activity = str(mode[1])
                            # If we have 3 elements, the third is intensity
                            intensity = float(mode[2]) if len(mode) > 2 else 1.0
                            
                            if freq > 0:
                                self.frequencies.append(freq)
                                self.intensities.append(intensity)
                                self.activities.append(activity)
                        except (ValueError, TypeError):
                            pass
                    # Check if it's a dictionary-like object
                    elif hasattr(mode, 'items'):
                        try:
                            freq = float(mode['TO_Frequency']) if 'TO_Frequency' in mode else 0
                            intensity = float(mode['I_Total']) if 'I_Total' in mode else 0
                            activity = str(mode['Activity']) if 'Activity' in mode else ""
                            
                            if freq > 0:
                                self.frequencies.append(freq)
                                self.intensities.append(intensity)
                                self.activities.append(activity)
                        except (ValueError, TypeError):
                            pass
        
        # If phonon_modes exists and we didn't find frequencies, try that
        if not self.frequencies and 'phonon_modes' in mineral_data:
            phonon_modes = mineral_data['phonon_modes']
            
            # Check if it's a pandas DataFrame
            if isinstance(phonon_modes, pd.DataFrame):
                freq_col = None
                intensity_col = None
                activity_col = None
                
                # Look for appropriate columns
                for col in phonon_modes.columns:
                    col_lower = col.lower()
                    if 'freq' in col_lower:
                        freq_col = col
                    elif 'intens' in col_lower or 'i_tot' in col_lower:
                        intensity_col = col
                    elif 'activ' in col_lower or 'charac' in col_lower:
                        activity_col = col
                
                if freq_col:
                    for _, row in phonon_modes.iterrows():
                        try:
                            freq = float(row[freq_col])
                            activity = str(row[activity_col]) if activity_col and pd.notna(row[activity_col]) else ""
                            
                            if freq > 0:
                                self.frequencies.append(freq)
                                self.intensities.append(1.0)  # Default intensity
                                self.activities.append(activity)
                        except (ValueError, TypeError):
                            pass
        
        # Try one more approach if we still have no frequencies
        if not self.frequencies:
            if isinstance(mineral_data.get('modes'), list) and len(mineral_data['modes']) > 0:
                # If modes are just a list of frequencies
                try:
                    all_numeric = True
                    for value in mineral_data['modes'][:5]:
                        if not isinstance(value, (int, float)) and not (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                            all_numeric = False
                            break
                    
                    if all_numeric:
                        # Just frequencies, no activity info
                        for value in mineral_data['modes']:
                            try:
                                freq = float(value)
                                if freq > 0:
                                    self.frequencies.append(freq)
                                    self.intensities.append(1.0)  # Default intensity
                                    self.activities.append("")   # No activity info
                            except (ValueError, TypeError):
                                pass
                except:
                    pass
        
        # Convert to numpy arrays for easier manipulation
        self.frequencies = np.array(self.frequencies)
        self.intensities = np.array(self.intensities)
        
        return len(self.frequencies) > 0

    def optimize_crystal_orientation(self):
        """Optimize crystal orientation to best match fitted peaks using grid search + NLLS."""
        try:
            # Check if we have fitted peaks
            if not hasattr(self, 'fitted_regions') or not self.fitted_regions:
                messagebox.showinfo("Optimization", "Please fit peaks in the Peak Fitting tab first.")
                return
                
            # Get theoretical spectrum data
            if not hasattr(self, 'current_structure') or not hasattr(self.current_structure, 'frequencies'):
                messagebox.showinfo("Optimization", "Please calculate a Raman spectrum first.")
                return
                
            # Extract theoretical frequencies and characters
            theoretical_freqs = self.current_structure.frequencies
            theoretical_chars = self.current_structure.activities if hasattr(self.current_structure, 'activities') else []
            
            # Flag to track if optimization is running
            self.optimization_running = True
            
            # Extract fitted peak positions and intensities
            fitted_peaks_info = []
            
            for region_idx, region_data in self.fitted_regions.items():
                if 'peaks' in region_data:
                    for peak in region_data['peaks']:
                        if 'center' in peak and 'height' in peak:
                            center = peak['center']
                            height = peak['height']
                            
                            # Find character if available
                            character = ""
                            for marker in self.fitted_peak_markers:
                                if hasattr(marker, 'get_position') and hasattr(marker, 'get_text'):
                                    marker_pos = marker.get_position()
                                    if abs(marker_pos[0] - center) < 5:  # Within 5 cm-1
                                        marker_text = marker.get_text()
                                        if len(marker_text) <= 3:  # Character labels are typically short
                                            character = marker_text
                            
                            fitted_peaks_info.append((center, height, center, character, True))
            
            # If no fitted peaks, use spectral peak positions
            if not fitted_peaks_info:
                messagebox.showinfo("Optimization", "No fitted peaks found. Please fit peaks in the Peak Fitting tab.")
                self.optimization_running = False
                return
                
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Optimizing Crystal Orientation")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Add progress bar
            ttk.Label(progress_window, text="Optimizing orientation...").pack(pady=10)
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            # Add status label
            status_label = ttk.Label(progress_window, text="Starting grid search...")
            status_label.pack(pady=5)
            
            # Add abort button
            abort_var = tk.BooleanVar(value=False)
            
            def abort_optimization():
                abort_var.set(True)
                status_label.config(text="Aborting optimization...")
            
            ttk.Button(progress_window, text="Abort", command=abort_optimization).pack(pady=10)
            
            # Update the window
            progress_window.update()
            
            # Get optimization method
            opt_method = "basin-hopping"
            if hasattr(self, 'opt_method_var'):
                if self.opt_method_var.get() == 2:
                    opt_method = "dual-annealing"
            
            # Get iteration count
            iterations = 100
            if hasattr(self, 'iterations_var'):
                try:
                    iterations = int(self.iterations_var.get())
                except:
                    pass
            
            # Get seed count
            seeds = 5
            if hasattr(self, 'seeds_var'):
                try:
                    seeds = int(self.seeds_var.get())
                except:
                    pass
            
            # Get current Euler angles as starting point
            phi_start = self.phi_var.get()
            theta_start = self.theta_var.get()
            psi_start = self.psi_var.get()
            
            # Get calibration parameters
            shift = self.orientation_shift_var.get()
            scale_factor = self.orientation_scale_var.get()
            
            # Create the objective function
            best_error = float('inf')
            best_result = None
            iteration_counter = [0]
            
            # Intermediate results for UI updates
            intermediate_results = []
            
            try:
                import numpy as np
                from scipy.optimize import minimize, basinhopping, dual_annealing
                
                # ===== GRID SEARCH PHASE =====
                # Define a coarse grid of Euler angles to search (9³ = 729 points)
                status_label.config(text="Performing grid search...")
                progress_window.update()
                
                # Set up grid search - stride depends on crystal symmetry
                if hasattr(self.current_structure, 'crystal_system'):
                    # For higher symmetry crystals, we can use larger strides
                    system = self.current_structure.crystal_system
                    if system in ['cubic']:
                        phi_stride, theta_stride, psi_stride = 45, 45, 45
                    elif system in ['hexagonal', 'tetragonal']:
                        phi_stride, theta_stride, psi_stride = 30, 30, 60
                    elif system in ['trigonal', 'orthorhombic']:
                        phi_stride, theta_stride, psi_stride = 30, 30, 30
                    else:  # monoclinic, triclinic
                        phi_stride, theta_stride, psi_stride = 20, 20, 20
                else:
                    # Default grid
                    phi_stride, theta_stride, psi_stride = 30, 30, 30
                
                # Create the grid
                phi_grid = np.arange(0, 360, phi_stride)
                theta_grid = np.arange(0, 180, theta_stride)
                psi_grid = np.arange(0, 360, psi_stride)
                
                # Add the current orientation as a grid point
                phi_grid = np.append(phi_grid, phi_start)
                theta_grid = np.append(theta_grid, theta_start)
                psi_grid = np.append(psi_grid, psi_start)
                
                # Use a subset of theoretical peaks for grid search (for speed)
                top_peaks = min(5, len(theoretical_freqs))
                
                # Define a simpler objective function for grid search
                def grid_objective(angles):
                    # Calculate spectrum using these angles
                    spectrum = self.current_structure.calculate_raman_spectrum(angles, 'VV')
                    
                    # Safety check for empty spectrum
                    if not spectrum:
                        return 1000  # Large error value
                        
                    # Apply calibration to theoretical positions
                    calibrated_freqs = [f * scale_factor + shift for f, _, _ in spectrum]
                    
                    # Safety check for empty frequencies
                    if not calibrated_freqs:
                        return 1000  # Large error value
                    
                    # Calculate error as weighted sum of squared differences
                    error = 0
                    matches = 0
                    
                    # Add small random jitter to avoid exactly zero matrices (helps optimization)
                    small_random = 1e-5 * np.random.rand()
                    
                    # For each experimental peak, find the closest theoretical peak
                    for exp_center, exp_height, _, exp_char, is_used in fitted_peaks_info:
                        if not is_used:
                            continue
                            
                        # If character information is available, use it for matching
                        if exp_char and len(theoretical_chars) == len(theoretical_freqs):
                            # Try to match by character
                            char_indices = [i for i, char in enumerate(theoretical_chars) if exp_char in char]
                            
                            if char_indices:
                                # Use distance to the closest matching character peak
                                distances = [abs(exp_center - calibrated_freqs[i]) for i in char_indices]
                                min_idx = char_indices[np.argmin(distances)]
                                min_dist = min(distances)
                                
                                # Weight by peak intensity 
                                theo_int = spectrum[min_idx][1]
                                error += min_dist**2 * exp_height
                                matches += 1
                            else:
                                # No matching character, use normal distance
                                min_dist = min(abs(exp_center - cf) for cf in calibrated_freqs)
                                error += min_dist**2 * exp_height
                        else:
                            # No character matching, find closest peak
                            min_dist = min(abs(exp_center - cf) for cf in calibrated_freqs)
                            error += min_dist**2 * exp_height
                            matches += 1
                    
                    if matches > 0:
                        # Add small jitter to avoid grid points having exactly the same error
                        return (error / matches) + small_random
                    return 1000  # Large error if no matches
                
                # Calculate objective function for each grid point
                grid_results = []
                total_points = len(phi_grid) * len(theta_grid) * len(psi_grid)
                point_counter = 0
                
                for phi in phi_grid:
                    for theta in theta_grid:
                        for psi in psi_grid:
                            # Check if aborted
                            if abort_var.get():
                                raise Exception("Optimization aborted by user")
                                
                            angles = (phi, theta, psi)
                            error = grid_objective(angles)
                            grid_results.append((angles, error))
                            
                            # Update progress
                            point_counter += 1
                            progress = (point_counter / total_points) * 40  # 40% of progress bar for grid search
                            progress_var.set(progress)
                            
                            # Update status every 100 points
                            if point_counter % 100 == 0 or point_counter == total_points:
                                status_label.config(text=f"Grid search: {point_counter}/{total_points} points")
                                progress_window.update()
                
                # Sort grid results by error (ascending)
                grid_results.sort(key=lambda x: x[1])
                
                # Take the top N points for NLLS optimization
                top_points = min(seeds, len(grid_results))
                starting_points = [grid_results[i][0] for i in range(top_points)]
                
                # ===== NLLS OPTIMIZATION PHASE =====
                status_label.config(text="Running NLLS optimization from top grid points...")
                progress_window.update()
                
                # Define the objective function for optimization
                def objective_function(angles):
                    # Update progress
                    iteration_counter[0] += 1
                    progress = 40 + (iteration_counter[0] / (iterations * top_points)) * 60  # Remaining 60% for NLLS
                    progress_var.set(min(99, progress))  # Cap at 99% until we're done
                    
                    # Periodic update of status
                    if iteration_counter[0] % 10 == 0:
                        status_label.config(text=f"NLLS optimization: iteration {iteration_counter[0]}")
                        progress_window.update()
                    
                    # Check if aborted
                    if abort_var.get():
                        raise Exception("Optimization aborted by user")
                    
                    # Normalize angles to respective ranges
                    phi, theta, psi = angles[0], angles[1], angles[2]
                    phi = phi % 360
                    theta = theta % 180
                    psi = psi % 360
                    
                    # Calculate spectrum using these angles
                    spectrum = self.current_structure.calculate_raman_spectrum((phi, theta, psi), 'VV')
                    
                    # Calculate error as weighted sum of squared differences
                    position_error = 0
                    intensity_error = 0
                    matches = 0
                    
                    # Apply calibration to theoretical positions
                    calibrated_freqs = [f * scale_factor + shift for f, _, _ in spectrum]
                    theo_intensities = [intensity for _, intensity, _ in spectrum]
                    
                    # Normalize experimental intensities
                    exp_heights = [height for _, height, _, _, is_used in fitted_peaks_info if is_used]
                    max_exp_height = max(exp_heights) if exp_heights else 1.0
                    
                    # Normalize theoretical intensities
                    max_theo_intensity = max(theo_intensities) if theo_intensities else 1.0
                    
                    # Matching constants
                    position_weight = 1.0  # Weight for position matching
                    intensity_weight = 2.0  # Weight for intensity matching (higher to prioritize intensity)
                    min_indices = {}  # Track which theoretical peak matches each experimental peak
                    
                    # For each experimental peak, find the closest theoretical peak
                    for exp_idx, (exp_center, exp_height, _, exp_char, is_used) in enumerate(fitted_peaks_info):
                        if not is_used:
                            continue
                            
                        # Normalize experimental height
                        norm_exp_height = exp_height / max_exp_height
                        
                        # Find matching theoretical peak
                        min_dist = float('inf')
                        min_idx = None
                        
                        # If character information is available, use it for matching
                        if exp_char and len(theoretical_chars) == len(theoretical_freqs):
                            # Try to match by character
                            char_indices = [i for i, char in enumerate(theoretical_chars) if exp_char in char]
                            
                            if char_indices:
                                # Use distance to the closest matching character peak
                                distances = [abs(exp_center - calibrated_freqs[i]) for i in char_indices]
                                min_idx = char_indices[np.argmin(distances)]
                                min_dist = min(distances)
                            else:
                                # No matching character, find closest peak by position
                                for i, cf in enumerate(calibrated_freqs):
                                    dist = abs(exp_center - cf)
                                    if dist < min_dist:
                                        min_dist = dist
                                        min_idx = i
                        else:
                            # No character info, find closest peak by position
                            for i, cf in enumerate(calibrated_freqs):
                                dist = abs(exp_center - cf)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = i
                        
                        # Store the matching index
                        if min_idx is not None:
                            min_indices[exp_idx] = min_idx
                            
                            # Add position error (weighted by peak height)
                            position_error += min_dist**2 * norm_exp_height
                            
                            # Add intensity error (normalized)
                            theo_intensity = theo_intensities[min_idx] / max_theo_intensity
                            intensity_diff = (theo_intensity - norm_exp_height)**2
                            intensity_error += intensity_diff
                            
                            matches += 1
                    
                    # Combine errors with weights
                    combined_error = position_weight * position_error + intensity_weight * intensity_error
                    
                    # Store intermediate results for UI updates
                    nonlocal best_error, best_result
                    current_error = combined_error / max(1, matches)
                    if current_error < best_error:
                        best_error = current_error
                        best_result = type('OptResult', (), {
                            'x': np.array([phi, theta, psi]), 
                            'fun': current_error,
                            'position_error': position_error / max(1, matches),
                            'intensity_error': intensity_error / max(1, matches)
                        })
                        intermediate_results.append((best_result.x, best_error))
                    
                    if matches > 0:
                        final_error = combined_error / matches
                        # Add a small penalty term to avoid numerical issues
                        regularization = 0.01 * (np.sum(np.array([phi, theta, psi])**2))  # L2 regularization
                        return final_error + regularization
                    return 1000  # Large error if no matches
                
                # Optimize from each starting point
                for i, start_point in enumerate(starting_points):
                    status_label.config(text=f"NLLS optimization: starting point {i+1}/{len(starting_points)}")
                    progress_window.update()
                    
                    # Set up bounds: full range for Euler angles
                    bounds = [(0, 360), (0, 180), (0, 360)]
                    
                    try:
                        # Choose optimization method
                        if opt_method == "basin-hopping":
                            # Basin-hopping with L-BFGS-B local minimizer
                            minimizer_kwargs = {
                                'method': 'L-BFGS-B',
                                'bounds': bounds
                            }
                            
                            result = basinhopping(
                                objective_function, 
                                np.array(start_point),
                                niter=iterations // top_points,
                                T=10.0,  # Increased temperature for more exploration (was 1.0)
                                stepsize=45.0,  # Increased step size (was 30.0)
                                minimizer_kwargs=minimizer_kwargs,
                                interval=10  # Check more frequently (was 20)
                            )
                        elif opt_method == "dual-annealing":
                            # Dual annealing (may work better for multi-modal surfaces)
                            result = dual_annealing(
                                objective_function,
                                bounds=bounds,
                                maxiter=iterations // top_points,
                                initial_temp=5000,
                                restart_temp_ratio=2e-5,
                                visit=2.62,
                                accept=-5.0,
                                maxfun=iterations * 2
                            )
                        else:
                            # Use SLSQP as fallback method
                            result = minimize(
                                objective_function,
                                np.array(start_point),
                                method='SLSQP',
                                bounds=bounds,
                                options={'maxiter': iterations // top_points, 'disp': False}
                            )
                            
                        # Update best result if this is better
                        if result.fun < best_error:
                            best_error = result.fun
                            best_result = result
                            
                    except Exception as e:
                        print(f"Error in optimization from starting point {start_point}: {str(e)}")
                        # Continue with next starting point
                        continue
                    
                    # Check if aborted
                    if abort_var.get():
                        status_label.config(text="Optimization aborted by user")
                        progress_window.update()
                        break
                
            except Exception as outer_error:
                print(f"Unexpected error in optimization process: {outer_error}")
                import traceback
                traceback.print_exc()
            
            finally:
                # Reset flag
                self.optimization_running = False
                
                # Close progress window if still open
                if progress_window and progress_window.winfo_exists():
                    progress_window.destroy()
                    progress_window = None
            
            # Process the results
            if best_result is not None:
                # Set the optimized angles
                opt_phi, opt_theta, opt_psi = best_result.x
                
                # Store the optimized angles for later use
                self.optimized_angles = (
                    round(opt_phi % 360, 1),
                    round(opt_theta % 180, 1), 
                    round(opt_psi % 360, 1)
                )
                
                # Update UI
                self.phi_var.set(self.optimized_angles[0])
                self.theta_var.set(self.optimized_angles[1])
                self.psi_var.set(self.optimized_angles[2])
                
                # Calculate spectrum with optimized orientation
                self.calculate_orientation_raman_spectrum()
                
                # Update the 3D tensor visualizer if it exists
                if hasattr(self, 'tensor_3d_visualizer'):
                    self.update_3d_visualizer()
                
                # Display results
                self.orientation_results_text.delete(1.0, tk.END)
                self.orientation_results_text.insert(tk.END, f"Optimization Results ({opt_method}):\n\n")
                self.orientation_results_text.insert(tk.END, f"Optimized orientation (Euler angles):\n")
                self.orientation_results_text.insert(tk.END, f"  φ: {self.phi_var.get()}°\n")
                self.orientation_results_text.insert(tk.END, f"  θ: {self.theta_var.get()}°\n")
                self.orientation_results_text.insert(tk.END, f"  ψ: {self.psi_var.get()}°\n\n")
                # Show detailed error breakdown
                position_error = getattr(best_result, 'position_error', best_error/2)
                intensity_error = getattr(best_result, 'intensity_error', best_error/2)
                
                self.orientation_results_text.insert(tk.END, f"Total error: {best_error:.2f}\n")
                self.orientation_results_text.insert(tk.END, f"Position error: {position_error:.2f}\n")
                self.orientation_results_text.insert(tk.END, f"Intensity error: {intensity_error:.2f}\n")
                self.orientation_results_text.insert(tk.END, f"Iterations: {iteration_counter[0]}\n")
                
                # Display fitted peaks that were used
                self.orientation_results_text.insert(tk.END, f"\nFitted peaks used:\n")
                for center, _, _, character, _ in fitted_peaks_info:
                    char_text = f" ({character})" if character else ""
                    self.orientation_results_text.insert(tk.END, f"  {center:.1f} cm⁻¹{char_text}\n")
            else:
                messagebox.showinfo("Optimization", "Optimization did not converge. Try different settings or starting angles.")
                
        except Exception as e:
            if progress_window and progress_window.winfo_exists():
                progress_window.destroy()
            messagebox.showerror("Optimization Error", f"Error during optimization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def auto_align_orientation_spectrum(self):
        """Automatically align the calculated spectrum with fitted peaks"""
        try:
            if not hasattr(self, 'fitted_regions') or not self.fitted_regions:
                messagebox.showinfo("Auto-Align", "Please fit peaks in the Peak Fitting tab first.")
                return
                
            if not hasattr(self, 'current_structure') or not hasattr(self.current_structure, 'frequencies'):
                messagebox.showinfo("Auto-Align", "Please calculate a Raman spectrum first.")
                return
            
            # Extract fitted peak positions
            fitted_peaks = []
            fitted_chars = []
            
            for region_idx, region_data in self.fitted_regions.items():
                if 'peaks' in region_data:
                    for peak in region_data['peaks']:
                        if 'center' in peak:
                            pos = peak['center']
                            fitted_peaks.append(pos)
                            
                            # Find character if available
                            char = ""
                            for marker in self.fitted_peak_markers:
                                if hasattr(marker, 'get_text') and callable(marker.get_text):
                                    text = marker.get_text()
                                    if hasattr(marker, 'get_position') and callable(marker.get_position):
                                        marker_pos = marker.get_position()
                                        if isinstance(marker_pos, tuple) and len(marker_pos) >= 1:
                                            if abs(marker_pos[0] - pos) < 5:  # Within 5 cm-1
                                                if len(text) <= 3:  # Character labels are typically short
                                                    char = text
                            fitted_chars.append(char)
            
            if not fitted_peaks:
                messagebox.showinfo("Auto-Align", "No fitted peaks found for alignment.")
                return
            
            # Get theoretical frequencies
            theoretical_freqs = self.current_structure.frequencies
            theoretical_chars = self.current_structure.activities if hasattr(self.current_structure, 'activities') else []
            
            # Define objective function for optimization
            def alignment_error(params):
                shift, scale = params
                aligned_freqs = theoretical_freqs * scale + shift
                
                # Calculate error based on matching characters if available
                error = 0
                for fit_pos, fit_char in zip(fitted_peaks, fitted_chars):
                    if fit_char and len(theoretical_chars) == len(theoretical_freqs):
                        # Try to match by character
                        matches = [(i, abs(fit_pos - (tf*scale+shift))) 
                                  for i, (tf, tc) in enumerate(zip(theoretical_freqs, theoretical_chars)) 
                                  if tc == fit_char]
                        
                        if matches:
                            # Use the closest match
                            matches.sort(key=lambda x: x[1])
                            _, dist = matches[0]
                            error += dist**2
                        else:
                            # No character match, use minimum distance
                            min_dist = min(abs(fit_pos - (tf*scale+shift)) for tf in theoretical_freqs)
                            error += min_dist**2 * 2  # Higher weight for no character match
                    else:
                        # No character information, use minimum distance
                        min_dist = min(abs(fit_pos - (tf*scale+shift)) for tf in theoretical_freqs)
                        error += min_dist**2
                
                return error
            
            # Run optimization to find best shift and scale
            from scipy.optimize import minimize
            
            # Try different starting points
            best_error = float('inf')
            best_params = (0.0, 1.0)
            
            starting_points = [
                (0.0, 1.0),    # No shift or scaling
                (-50.0, 1.0),  # Negative shift
                (50.0, 1.0),   # Positive shift
                (0.0, 0.95),   # Slight compression
                (0.0, 1.05),   # Slight expansion
            ]
            
            for start in starting_points:
                result = minimize(alignment_error, start, method='Powell')
                if result.fun < best_error:
                    best_error = result.fun
                    best_params = result.x
            
            # Apply the best parameters
            shift, scale = best_params
            
            # Update the UI
            self.orientation_shift_var.set(round(shift, 1))
            self.orientation_scale_var.set(round(scale, 3))
            
            # Update the plot
            self.calculate_orientation_raman_spectrum()
            
            messagebox.showinfo("Auto-Align", 
                               f"Spectrum aligned with fitted peaks.\nShift: {shift:.1f} cm⁻¹, Scale: {scale:.3f}")
            
        except Exception as e:
            messagebox.showerror("Auto-Align Error", f"Error during alignment: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def reset_orientation_alignment(self):
        """Reset spectrum alignment parameters"""
        self.orientation_shift_var.set(0.0)
        self.orientation_scale_var.set(1.0)
        self.calculate_orientation_raman_spectrum()

    def browse_cif_file(self):
        # Check if pymatgen is installed
        import importlib.util
        pymatgen_spec = importlib.util.find_spec("pymatgen")
        
        path = filedialog.askopenfilename(title="Select CIF File", filetypes=[("CIF files", "*.cif"), ("All files", "*.*")])
        
        if path:
            self.cif_path_var.set(path)
            
            # If pymatgen not installed, show a recommendation
            if pymatgen_spec is None:
                messagebox.showinfo(
                    "Pymatgen Recommended", 
                    "Pymatgen is not installed. For better crystallographic analysis, please install it using:\n\n" +
                    "pip install pymatgen\n\n" +
                    "The program will still work, but crystallographic analysis will be limited."
                )

    def display_cif_info(self):
        path = self.cif_path_var.get()
        if not path:
            messagebox.showinfo("CIF Info", "Please select a CIF file.")
            return
        try:
            # Create a StringIO to capture print output
            import io
            import sys
            old_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Create crystal structure and find bonds
            cs = CrystalStructure(path)
            
            # Basic crystal information
            info = []
            info.append(f"Lattice: a={cs.lattice['a']:.4f}, b={cs.lattice['b']:.4f}, c={cs.lattice['c']:.4f}, ")
            info.append(f"        alpha={cs.lattice['alpha']:.2f}, beta={cs.lattice['beta']:.2f}, gamma={cs.lattice['gamma']:.2f}")
            
            # Atom information
            info.append("\nAtoms:")
            for atom in cs.atoms[:10]:  # Show first 10 atoms
                info.append(f"  {atom['label']}: ({atom['cart_x']:.3f}, {atom['cart_y']:.3f}, {atom['cart_z']:.3f})")
            if len(cs.atoms) > 10:
                info.append(f"  ... and {len(cs.atoms)-10} more atoms")
            
            # Find and analyze bonds
            info.append("\nAnalyzing bonds...")
            cs.find_bonds(cutoff=3.0)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Get captured output
            bond_info = captured_output.getvalue()
            
            # Update the text box with all information
            self.cif_info_text.delete(1.0, tk.END)
            self.cif_info_text.insert(tk.END, "\n".join(info) + "\n" + bond_info)
            
            # Store the crystal structure for future use
            self.current_structure = cs
            
        except Exception as e:
            self.cif_info_text.delete(1.0, tk.END)
            self.cif_info_text.insert(tk.END, f"Error parsing CIF: {e}")
            import traceback
            traceback.print_exc()

    def calculate_orientation_raman_tensor(self):
        path = self.cif_path_var.get()
        mineral = self.orientation_mineral_var.get()
        if not path or not mineral:
            messagebox.showinfo("Raman Tensor", "Please select both a CIF file and a mineral.")
            return
        
        # Create a StringIO to capture print output
        import io
        import sys
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Get Born charges from the selected mineral
            if mineral not in self.mineral_database:
                raise ValueError(f"Mineral {mineral} not found in database")
                
            mineral_data = self.mineral_database[mineral]
            if 'born_charges' not in mineral_data:
                raise ValueError(f"No Born charges found for {mineral}")
                
            # Use the existing crystal structure if available
            if hasattr(self, 'current_structure') and self.current_structure is not None:
                cs = self.current_structure
                # Update Born charges
                cs.born_charges_db = mineral_data['born_charges']
                cs.map_born_charges()
            else:
                # Create a new crystal structure
                cs = CrystalStructure(path, mineral_data['born_charges'])
            
            # Make sure bonds are found
            if not cs.bonds:
                cs.find_bonds()
                
            # Calculate Raman tensors for all symmetry characters
            raman_tensors = cs.calculate_raman_tensor(character='all')
            
            # Set the first tensor as the primary one for compatibility
            if isinstance(raman_tensors, list) and len(raman_tensors) > 0:
                raman_tensor = raman_tensors[0]
            else:
                raman_tensor = raman_tensors
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Format output for display
            out = []
            out.append("Multiple Raman tensors calculated based on symmetry characters:")
            
            if hasattr(cs, 'activities') and len(cs.activities) > 0:
                for i, (tensor, activity) in enumerate(zip(raman_tensors, cs.activities)):
                    out.append(f"\n=== {activity} Tensor ===")
                    for row in tensor:
                        out.append("  ".join(f"{v:10.4f}" for v in row))
            else:
                # Just show the default tensor if no activities
                out.append("\n=== General Tensor ===")
                for row in raman_tensor:
                    out.append("  ".join(f"{v:10.4f}" for v in row))
            
            # Get captured output
            debug_info = captured_output.getvalue()
            
            # Display the results
            self.orientation_raman_text.delete(1.0, tk.END)
            self.orientation_raman_text.insert(tk.END, "\n".join(out))
            
            # Show additional debug info if the tensor is zero
            if np.all(np.abs(raman_tensor) < 1e-6):
                self.orientation_raman_text.insert(tk.END, "\n\nWARNING: Zero tensor calculated.\nDebug information:\n" + debug_info)
            
            # Update 3D visualization if available
            if hasattr(self, 'tensor_3d_visualizer'):
                self.current_structure = cs
                self.update_3d_visualizer()
            
        except Exception as e:
            sys.stdout = old_stdout  # Restore stdout before handling exception
            self.orientation_raman_text.delete(1.0, tk.END)
            self.orientation_raman_text.insert(tk.END, f"Error: {e}\n\n")
            self.orientation_raman_text.insert(tk.END, captured_output.getvalue())
            import traceback
            traceback.print_exc()

    def on_tab_changed(self, event):
        """Handle tab change events to sync mineral selection and update visualizations"""
        try:
            selected_tab = self.notebook.select()
            tab_id = self.notebook.index(selected_tab)
            
            # If switching to Crystal Orientation tab (index 2)
            if tab_id == 2 and hasattr(self, 'current_mineral') and self.current_mineral:
                # Get available minerals
                mineral_names = self.orientation_mineral_combo['values']
                
                # If current mineral is in the list, select it
                if self.current_mineral in mineral_names:
                    self.orientation_mineral_var.set(self.current_mineral)
            
            # If switching to 3D Tensor tab (index 3)
            elif tab_id == 3 and hasattr(self, 'tensor_3d_visualizer'):
                # Update 3D visualizer with current data
                self.update_3d_visualizer()
        except Exception as e:
            # Log errors in tab changing
            print(f"Error in tab change: {str(e)}")
            pass
    
    def update_3d_visualizer(self):
        """Update the 3D tensor visualizer with current data"""
        if not hasattr(self, 'tensor_3d_visualizer'):
            return
            
        # Update crystal structure if available
        if hasattr(self, 'current_structure') and self.current_structure is not None:
            # Only update if Raman tensor is calculated
            if hasattr(self.current_structure, 'raman_tensor'):
                self.tensor_3d_visualizer.set_crystal_structure(self.current_structure)
                
                # Get character information from activities if available
                if hasattr(self.current_structure, 'activities') and self.current_structure.activities:
                    # Extract unique characters from activities
                    characters = []
                    for activity in self.current_structure.activities:
                        if activity and isinstance(activity, str):
                            # Complete list of all possible character types for all point groups
                            all_possible_characters = [
                                # Generic types
                                "A", "B", "E", "T",
                                
                                # Numbered types
                                "A1", "A₁", "A2", "A₂", "B1", "B₁", "B2", "B₂", "B3", "B₃",
                                "E1", "E₁", "E2", "E₂", "T1", "T₁", "T2", "T₂",
                                
                                # g-type (gerade - with inversion)
                                "Ag", "Bg", "Eg", "Tg", "A1g", "A2g", "B1g", "B2g", "B3g", "E1g", "E2g", "T1g", "T2g",
                                
                                # u-type (ungerade - without inversion)
                                "Au", "Bu", "Eu", "Tu", "A1u", "A2u", "B1u", "B2u", "B3u", "E1u", "E2u", "T1u", "T2u"
                            ]
                            
                            for char in all_possible_characters:
                                if char in activity and char not in characters:
                                    characters.append(char)
                    
                    # Print found characters for debugging
                    print(f"Found characters for visualization: {characters}")
                    
                    # Update character options in visualizer
                    self.tensor_3d_visualizer.update_character_options(characters)
        
        # Update optimized orientation if available
        if hasattr(self, 'optimized_angles'):
            self.tensor_3d_visualizer.set_optimized_orientation(self.optimized_angles)
            
            # Also update the current orientation controls in the visualizer
            if hasattr(self.tensor_3d_visualizer, 'phi_var') and hasattr(self.tensor_3d_visualizer, 'theta_var') and hasattr(self.tensor_3d_visualizer, 'psi_var'):
                self.tensor_3d_visualizer.phi_var.set(self.optimized_angles[0])
                self.tensor_3d_visualizer.theta_var.set(self.optimized_angles[1])
                self.tensor_3d_visualizer.psi_var.set(self.optimized_angles[2])
                
            # Redraw the tensor
            if hasattr(self.tensor_3d_visualizer, 'redraw_tensor'):
                self.tensor_3d_visualizer.redraw_tensor()

    def refine_peak_fitting(self):
        """Refine the calculated peaks to better match the experimental data"""
        progress_window = None
        plot_window = None
        
        # Import needed libraries right away to avoid reference errors
        import numpy as np
        from scipy.optimize import least_squares
        from scipy.interpolate import interp1d
        
        try:
            # Check if we have experimental data and calculated spectrum
            if self.spectrum_data is None:
                messagebox.showinfo("Refinement", "Please import experimental data first.")
                return
                
            # Calculate the spectrum with current orientation
            spectrum = self.calculate_orientation_raman_spectrum()
            if not spectrum:
                messagebox.showinfo("Refinement", "Please calculate a spectrum first.")
                return
            
            # Create a progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Refining Peak Fitting")
            progress_window.geometry("300x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            ttk.Label(progress_window, text="Refining peak parameters...").pack(pady=10)
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            status_label = ttk.Label(progress_window, text="Starting refinement...")
            status_label.pack(pady=5)
            
            # Update the window
            progress_window.update()
            
            # Create convergence plot window
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Optimization Convergence")
            plot_window.geometry("500x400")
            plot_window.transient(self.root)
            
            # Create figure and canvas for convergence plot
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title("Optimization Convergence")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("R² (Coefficient of Determination)")
            ax.grid(True, linestyle='--', alpha=0.7)
            canvas = FigureCanvasTkAgg(fig, plot_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initialize data for convergence plot
            iterations = []
            r_squared_values = []
            rmse_values = []
            line_r2, = ax.plot(iterations, r_squared_values, 'b-', label='R²')
            ax.set_ylim(0, 1)  # R² typically ranges from 0 to 1
            ax.legend()
            fig.tight_layout()
            canvas.draw()
            
            # Extract experimental data
            exp_x = self.spectrum_data['x']
            exp_y = self.spectrum_data['y']
            
            # Normalize experimental data for comparison
            exp_y_norm = exp_y / np.max(exp_y) if np.max(exp_y) > 0 else exp_y
            exp_y_mean = np.mean(exp_y_norm)
            exp_y_var_sum = np.sum((exp_y_norm - exp_y_mean)**2)
            
            # Get calibration parameters
            shift = self.orientation_shift_var.get()
            scale_factor = self.orientation_scale_var.get()
            
            # Extract calculated peak data (positions, intensities)
            # Unpack spectrum data: [(freq, intensity, character), ...]
            calc_freqs, calc_intensities, calc_characters = zip(*spectrum) if spectrum else ([], [], [])
            
            # Apply calibration
            calc_freqs = [freq * scale_factor + shift for freq in calc_freqs]
            
            # Create initial guess for parameters (position adjustments, width adjustments)
            # For each peak: [position_offset, width, intensity_scale]
            initial_params = []
            for _ in range(len(calc_freqs)):
                initial_params.extend([0.0, 10.0, 1.0])  # No position offset, default width=10, no intensity scaling
            
            # Status reporting function for iterations
            iter_count = [0]
            last_plot_update = [0]  # Track when we last updated the plot
            
            # We can't use the callback in least_squares, so we'll implement a manual optimization
            # using a custom function that incrementally updates the UI
            
            # Define objective function for optimization
            def residual_function(params):
                # Reshape params into sets of [pos_offset, width, intensity_scale] for each peak
                param_sets = params.reshape(-1, 3)
                
                # Create a high-res x-range for generating smooth spectra
                x_min, x_max = min(exp_x), max(exp_x)
                padding = (x_max - x_min) * 0.05
                x_dense = np.linspace(x_min - padding, x_max + padding, 1000)
                
                # Generate synthetic spectrum
                y_model = np.zeros_like(x_dense)
                
                # Add each peak
                for i, (base_freq, base_intensity, _) in enumerate(zip(calc_freqs, calc_intensities, calc_characters)):
                    # Get refined parameters for this peak
                    pos_offset, width, intensity_scale = param_sets[i]
                    
                    # Calculate adjusted parameters
                    final_pos = base_freq + pos_offset
                    final_intensity = base_intensity * intensity_scale
                    
                    # Add Lorentzian peak
                    y_model += final_intensity * width**2 / ((x_dense - final_pos)**2 + width**2)
                
                # Interpolate model to match experimental x values
                from scipy.interpolate import interp1d
                model_interp = interp1d(x_dense, y_model, bounds_error=False, fill_value=0)
                y_model_at_exp_x = model_interp(exp_x)
                
                # Normalize both spectra for comparison
                exp_y_norm = exp_y / np.max(exp_y) if np.max(exp_y) > 0 else exp_y
                model_y_norm = y_model_at_exp_x / np.max(y_model_at_exp_x) if np.max(y_model_at_exp_x) > 0 else y_model_at_exp_x
                
                # Calculate residuals
                residuals = exp_y_norm - model_y_norm
                
                # Calculate R² value
                sse = np.sum(residuals**2)
                r_squared = 1 - sse / exp_y_var_sum if exp_y_var_sum > 0 else 0
                rmse = np.sqrt(np.mean(residuals**2))
                
                # Add constraints to limit large position shifts and unreasonable widths
                penalty = 0
                for pos_offset, width, _ in param_sets:
                    # Penalize large position shifts
                    if abs(pos_offset) > 20:
                        penalty += (abs(pos_offset) - 20)**2
                    
                    # Penalize too narrow or too wide peaks
                    if width < 2 or width > 30:
                        penalty += 10 * ((width - 2) if width < 2 else (width - 30))**2
                
                # Add penalty to residuals
                if penalty > 0:
                    penalty_factor = 0.1 * np.mean(residuals**2) * len(residuals) / penalty
                    residuals = np.append(residuals, np.sqrt(penalty * penalty_factor))
                
                return residuals, r_squared, rmse
            
            # Run optimization
            try:
                from scipy.optimize import least_squares
                import numpy as np
                
                # Set bounds to prevent unreasonable values
                # [pos_offset_min, width_min, intensity_min] * n_peaks
                lower_bounds = np.array([-20.0, 2.0, 0.1] * len(calc_freqs))
                # [pos_offset_max, width_max, intensity_max] * n_peaks
                upper_bounds = np.array([20.0, 30.0, 10.0] * len(calc_freqs))
                
                # Create custom optimization loop since least_squares doesn't support callbacks in older SciPy versions
                params = np.array(initial_params)
                
                # Use least_squares without callback, but manually update UI every few iterations
                total_iterations = 300  # Reduce max iterations for faster convergence
                batch_size = 10  # Smaller batches for more frequent UI updates
                
                # Initial evaluation
                iter_count[0] += 1
                residuals, r_squared, rmse = residual_function(params)
                
                # Add initial point to plot
                iterations.append(0)
                r_squared_values.append(r_squared)
                rmse_values.append(rmse)
                
                # Update plot with initial point and force redraw
                line_r2.set_xdata(iterations)
                line_r2.set_ydata(r_squared_values)
                ax.set_xlim(0, 20)
                for txt in ax.texts:
                    txt.remove()
                ax.text(0.98, 0.02, f"R²: {r_squared:.4f}, RMSE: {rmse:.4f}", 
                        transform=ax.transAxes, ha='right', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
                fig.tight_layout()
                canvas.draw()
                # Force GUI update to make plot visible
                plot_window.update_idletasks()
                plot_window.update()
                self.root.update()
                
                # Setup variables for progress monitoring
                last_r_squared = r_squared
                no_improvement_count = 0
                improvement_threshold = 0.0002  # Increase threshold for faster convergence
                
                # Run optimization in batches with UI updates between
                for batch_idx in range(0, total_iterations, batch_size):
                    # Run a batch of optimization iterations
                    try:
                        # Use a more robust method with faster convergence
                        result = least_squares(
                            lambda p: residual_function(p)[0],  # Just return residuals
                            params,
                            bounds=(lower_bounds, upper_bounds),
                            method='trf',  # Trust Region Reflective
                            ftol=1e-3,  # Less strict tolerance for faster convergence
                            xtol=1e-3,
                            gtol=1e-3,
                            max_nfev=batch_size,
                            verbose=0,
                            jac='2-point'  # Faster Jacobian approximation
                        )
                        
                        # Update params for next batch
                        params = result.x
                        
                        # Evaluate current result
                        iter_count[0] += batch_size
                        residuals, r_squared, rmse = residual_function(params)
                        
                        # Add to plot
                        iterations.append(iter_count[0])
                        r_squared_values.append(r_squared)
                        rmse_values.append(rmse)
                        
                        # Update progress bar and status
                        progress_pct = min(100, int(100 * iter_count[0] / total_iterations))
                        progress_var.set(progress_pct)
                        
                        # Show status including R²
                        param_sets = params.reshape(-1, 3)
                        peak_info = []
                        for i in range(min(2, len(param_sets))):
                            pos_offset, width, intensity = param_sets[i]
                            peak_info.append(f"P{i+1}: Δ={pos_offset:.1f}")
                        
                        status_text = f"Iter: {iter_count[0]}, R²: {r_squared:.3f}, " + ", ".join(peak_info)
                        if status_label.winfo_exists():
                            status_label.config(text=status_text)
                            progress_window.update()
                        
                        # Update convergence plot
                        line_r2.set_xdata(iterations)
                        line_r2.set_ydata(r_squared_values)
                        
                        # Adjust x-axis limits
                        ax.set_xlim(0, max(20, iter_count[0]))
                        
                        # Set y-axis limits
                        if r_squared_values:
                            y_min = max(0, min(r_squared_values) * 0.9)
                            y_max = min(1, max(r_squared_values) * 1.05 + 0.05)
                            ax.set_ylim(y_min, y_max)
                        
                        # Update annotation
                        for txt in ax.texts:
                            txt.remove()
                        ax.text(0.98, 0.02, f"R²: {r_squared:.4f}, RMSE: {rmse:.4f}", 
                                transform=ax.transAxes, ha='right', va='bottom',
                                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
                        
                        # Redraw the plot with forced update
                        ax.grid(True, linestyle='--', alpha=0.7)  # Make grid more visible
                        fig.tight_layout()
                        canvas.draw()
                        plot_window.update_idletasks()  # Force GUI update
                        plot_window.update()
                        
                        # Check for convergence - use more aggressive stopping criteria
                        improvement = r_squared - last_r_squared
                        if improvement < improvement_threshold:
                            no_improvement_count += 1
                            if no_improvement_count >= 2:  # Stop sooner with minimal improvement
                                break
                        else:
                            no_improvement_count = 0  # Reset counter if we're still improving
                            
                        last_r_squared = r_squared
                        
                    except Exception as batch_error:
                        print(f"Error in optimization batch: {batch_error}")
                        break
                
                # Consider the final params as our refined parameters
                refined_params = params.reshape(-1, 3)
                
                # Store the refined spectrum for display
                refined_spectrum = []
                for i, (base_freq, base_intensity, character) in enumerate(zip(calc_freqs, calc_intensities, calc_characters)):
                    pos_offset, width, intensity_scale = refined_params[i]
                    refined_pos = base_freq + pos_offset
                    refined_intensity = base_intensity * intensity_scale
                    refined_spectrum.append((refined_pos, refined_intensity, width, character))
                
                self.refined_spectrum = refined_spectrum
                
                # Close progress window
                if progress_window.winfo_exists():
                    progress_window.destroy()
                
                # Update the plot with final result label
                for txt in ax.texts:
                    txt.remove()
                ax.text(0.98, 0.02, f"Final: R²={r_squared:.4f}, RMSE={rmse:.4f}, Iterations={iter_count[0]}", 
                        transform=ax.transAxes, ha='right', va='bottom',
                        bbox=dict(facecolor='lightgreen', alpha=0.8, boxstyle='round,pad=0.3'))
                
                ax.set_title(f"Optimization Convergence (Complete)")
                fig.tight_layout()
                canvas.draw()
                
                # Add a button to close the window
                close_button = ttk.Button(plot_window, text="Close Plot", command=plot_window.destroy)
                close_button.pack(pady=10)
                
                # Generate and display the refined spectrum
                self.plot_refined_spectrum()
                
                # Show refinement statistics
                self.orientation_results_text.delete(1.0, tk.END)
                self.orientation_results_text.insert(tk.END, "Peak Refinement Results:\n\n")
                
                # Show details of refined peaks
                self.orientation_results_text.insert(tk.END, "Refined Peak Parameters:\n")
                for i, (pos, intensity, width, character) in enumerate(refined_spectrum):
                    orig_pos = calc_freqs[i]
                    pos_change = pos - orig_pos
                    char_str = f" ({character})" if character else ""
                    self.orientation_results_text.insert(
                        tk.END, 
                        f"Peak {i+1}{char_str}: {pos:.1f} cm⁻¹ (Δ{pos_change:+.1f}), "
                        f"Width: {width:.1f}, I: {intensity:.2f}\n"
                    )
                
                # Show goodness of fit
                residuals, _, _ = residual_function(params)
                fit_quality = 100 * (1 - np.sqrt(np.sum(residuals**2) / len(residuals)))
                self.orientation_results_text.insert(
                    tk.END, 
                    f"\nFit Quality: {fit_quality:.1f}% match\n"
                    f"R²: {r_squared:.4f}, RMSE: {rmse:.4f}\n"
                    f"Iterations: {iter_count[0]}\n"
                )
                
            except Exception as e:
                # Close progress window if open
                if progress_window and progress_window.winfo_exists():
                    progress_window.destroy()
                
                # Close plot window with error message
                if plot_window and plot_window.winfo_exists():
                    for widget in plot_window.winfo_children():
                        widget.destroy()
                    error_label = ttk.Label(plot_window, text=f"Error during optimization: {str(e)}")
                    error_label.pack(pady=20)
                    close_button = ttk.Button(plot_window, text="Close", command=plot_window.destroy)
                    close_button.pack(pady=10)
                else:
                    messagebox.showerror("Refinement Error", f"Error during peak refinement: {str(e)}")
                    
                import traceback
                traceback.print_exc()
        
        except Exception as e:
            # Make sure to close any open windows
            if progress_window and progress_window.winfo_exists():
                progress_window.destroy()
                
            if plot_window and plot_window.winfo_exists():
                plot_window.destroy()
                
            messagebox.showerror("Refinement Error", f"Error setting up refinement: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_refined_spectrum(self):
        """Plot the refined spectrum alongside the experimental data"""
        if not hasattr(self, 'refined_spectrum') or not self.refined_spectrum:
            return
            
        if self.spectrum_data is None:
            return
            
        # Clear previous plots
        self.orientation_ax.clear()
        self.difference_ax.clear()
        
        # Extract experimental data
        exp_x = self.spectrum_data['x']
        exp_y = self.spectrum_data['y']
        
        # Create a dense x-array for smooth plotting
        x_min, x_max = min(exp_x), max(exp_x)
        padding = (x_max - x_min) * 0.05
        x_dense = np.linspace(x_min - padding, x_max + padding, 1000)
        
        # Generate the refined spectrum
        y_refined = np.zeros_like(x_dense)
        
        # Track peaks by character type for coloring
        character_groups = {}
        for pos, intensity, width, character in self.refined_spectrum:
            char_key = character if character else "Unknown"
            if char_key not in character_groups:
                character_groups[char_key] = []
            character_groups[char_key].append((pos, intensity, width))
            
            # Add Lorentzian peak to full spectrum
            y_refined += intensity * width**2 / ((x_dense - pos)**2 + width**2)
        
        # Normalize both spectra for comparison
        exp_y_norm = exp_y / np.max(exp_y) if np.max(exp_y) > 0 else exp_y
        y_refined_norm = y_refined / np.max(y_refined) if np.max(y_refined) > 0 else y_refined
        
        # Plot the experimental data
        self.orientation_ax.plot(exp_x, exp_y_norm, 'k-', label='Experimental', linewidth=2)
        
        # Plot the refined spectrum
        self.orientation_ax.plot(x_dense, y_refined_norm, 'r-', label='Refined Fit', linewidth=1.5, alpha=0.7)
        
        # Interpolate refined spectrum to match experimental x values for residuals
        from scipy.interpolate import interp1d
        refined_interp = interp1d(x_dense, y_refined_norm, bounds_error=False, fill_value=0)
        y_refined_at_exp_x = refined_interp(exp_x)
        
        # Calculate residuals for fit quality assessment
        residuals = exp_y_norm - y_refined_at_exp_x
        
        # Identify regions of good/poor fit
        fit_quality_regions = []
        window_size = 10  # points to include in each region assessment
        for i in range(0, len(exp_x) - window_size, window_size):
            region_x = exp_x[i:i+window_size]
            region_residuals = residuals[i:i+window_size]
            region_rmse = np.sqrt(np.mean(region_residuals**2))
            
            # Classify region quality
            if region_rmse < 0.05:
                quality = "good"
            elif region_rmse < 0.1:
                quality = "medium"
            else:
                quality = "poor"
                
            # Store region info
            fit_quality_regions.append((region_x[0], region_x[-1], quality))
        
        # Color the background of regions based on fit quality
        for start_x, end_x, quality in fit_quality_regions:
            if quality == "poor":
                color = "mistyrose"  # Light red
                alpha = 0.2
            elif quality == "medium":
                color = "lightyellow"  # Light yellow
                alpha = 0.15
            else:
                continue  # Don't highlight good regions
                
            # Highlight the region
            self.orientation_ax.axvspan(start_x, end_x, alpha=alpha, color=color, zorder=0)
        
        # Sort peaks by intensity for selective labeling
        all_peaks = [(pos, intensity, width, character) for pos, intensity, width, character in self.refined_spectrum]
        all_peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Select top peaks to label (maximum 7)
        top_peaks = all_peaks[:min(7, len(all_peaks))]
        top_peak_positions = [p[0] for p in top_peaks]
        
        # Plot individual peak components by character group
        cmap = plt.cm.tab10
        colors = {char: cmap(i % 10) for i, char in enumerate(character_groups.keys())}
        
        # Plot all components if the toggle is on
        if self.show_components_var.get():
            for char_key, peaks in character_groups.items():
                color = colors[char_key]
                for pos, intensity, width in peaks:
                    # Scale intensity to match the normalized spectrum
                    scaled_intensity = intensity / np.max(y_refined) if np.max(y_refined) > 0 else intensity
                    
                    # Component curve
                    y_component = scaled_intensity * width**2 / ((x_dense - pos)**2 + width**2)
                    self.orientation_ax.plot(x_dense, y_component, '--', color=color, linewidth=0.8, alpha=0.3)
        
        # Always add labels for top peaks
        for char_key, peaks in character_groups.items():
            color = colors[char_key]
            for pos, intensity, width in peaks:
                if pos in top_peak_positions:
                    # Get y position at peak for label placement
                    idx = np.abs(x_dense - pos).argmin()
                    y_pos = y_refined_norm[idx] + 0.05  # Place label above peak
                    
                    # Add peak label with character
                    self.orientation_ax.annotate(
                        char_key, 
                        xy=(pos, y_pos),
                        xycoords='data',
                        color=color,
                        horizontalalignment='center',
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec=color)
                    )
        
        # Create a small table of top 5 peaks with their fit metrics
        top_5_peaks = all_peaks[:min(5, len(all_peaks))]
        peak_table = []
        
        # Calculate individual peak fit quality
        for pos, intensity, width, character in top_5_peaks:
            # Find experimental data points near this peak
            peak_range = 20  # cm-1 range around peak to assess fit
            mask = (exp_x >= pos - peak_range) & (exp_x <= pos + peak_range)
            
            if np.sum(mask) > 3:  # Need enough points for meaningful calculation
                local_exp = exp_y_norm[mask]
                local_x = exp_x[mask]
                local_fit = refined_interp(local_x)
                
                # Calculate local R²
                local_residuals = local_exp - local_fit
                local_r2 = 1 - np.sum(local_residuals**2) / np.sum((local_exp - np.mean(local_exp))**2)
                
                peak_table.append((pos, character, local_r2))
        
        # Add the peak quality table to the plot
        if peak_table:
            table_text = "Top Peaks:\n"
            for pos, char, r2 in peak_table:
                quality_marker = "★★★" if r2 > 0.95 else "★★" if r2 > 0.8 else "★"
                table_text += f"{int(pos)} cm⁻¹ ({char}): {r2:.2f} {quality_marker}\n"
                
            # Position the table in the upper left
            self.orientation_ax.text(
                0.02, 0.98, table_text,
                transform=self.orientation_ax.transAxes,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9)
            )
        
        # Plot difference
        self.difference_ax.plot(exp_x, residuals, 'k-', linewidth=1)
        self.difference_ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Fill areas for visual emphasis of differences
        self.difference_ax.fill_between(
            exp_x, residuals, 0, 
            where=(residuals > 0), 
            color='red', alpha=0.3, interpolate=True
        )
        self.difference_ax.fill_between(
            exp_x, residuals, 0, 
            where=(residuals < 0), 
            color='blue', alpha=0.3, interpolate=True
        )
        
        # Set limits and labels
        self.orientation_ax.set_xlim(x_min - padding, x_max + padding)
        self.orientation_ax.set_title("Refined Spectrum Fit")
        self.orientation_ax.set_ylabel("Normalized Intensity")
        self.orientation_ax.legend(loc='upper right')
        
        # Hide x-labels on top plot
        plt.setp(self.orientation_ax.get_xticklabels(), visible=False)
        
        # Label the difference plot
        self.difference_ax.set_ylabel("Residuals")
        self.difference_ax.set_xlabel("Wavenumber (cm⁻¹)")
        
        # Set y-limits for difference plot
        max_diff = max(0.2, np.max(np.abs(residuals)) * 1.2)
        self.difference_ax.set_ylim(-max_diff, max_diff)
        
        # Calculate overall goodness of fit metrics
        rmse = np.sqrt(np.mean(residuals**2))
        max_error = np.max(np.abs(residuals))
        r_squared = 1 - np.sum(residuals**2) / np.sum((exp_y_norm - np.mean(exp_y_norm))**2)
        
        # Add metrics to the plot
        metrics_text = f"RMSE: {rmse:.3f}   R²: {r_squared:.3f}   Max Error: {max_error:.3f}"
        self.difference_ax.text(
            0.98, 0.05, metrics_text,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=self.difference_ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8)
        )
        
        # Add toggle buttons for component visibility
        # First, let's clear any previous toggle frame
        for widget in self.orientation_plot_frame.winfo_children():
            if isinstance(widget, tk.Frame) and getattr(widget, 'is_toggle_frame', False):
                widget.destroy()
                
        # Create toggle frame
        toggle_frame = tk.Frame(self.orientation_plot_frame)
        toggle_frame.is_toggle_frame = True  # Mark for future reference
        toggle_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Add component toggle switch - use the existing variable from __init__
        component_checkbox = ttk.Checkbutton(
            toggle_frame, 
            text="Show Components", 
            variable=self.show_components_var, 
            command=self.toggle_components_visibility
        )
        component_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Add a label for character color legend
        ttk.Label(toggle_frame, text="Peak Types:").pack(side=tk.LEFT, padx=(10,5))
        
        # Add color swatches for each character
        for char_key, color in colors.items():
            # Create a small colored swatch
            swatch_frame = tk.Frame(toggle_frame, width=15, height=15, bg=mcolors.rgb2hex(color))
            swatch_frame.pack(side=tk.LEFT, padx=2)
            
            # Add character label
            ttk.Label(toggle_frame, text=char_key).pack(side=tk.LEFT, padx=(0,8))
        
        # Adjust layout and redraw
        self.orientation_figure.tight_layout()
        self.orientation_canvas.draw()
        
    def toggle_components_visibility(self):
        """Toggle visibility of individual peak components"""
        # The checkbox state is already updated by tkinter before this callback is called
        # Just re-plot the spectrum which will use the updated checkbox state
        self.plot_refined_spectrum()

    def correct_mode_assignment(self, peak_data):
        """Correct symmetry mode assignments, particularly for centrosymmetric crystals.
        
        In centrosymmetric crystals:
        - "g" (gerade) modes are Raman-active
        - "u" (ungerade) modes are IR-active
        
        This function corrects common mistaken assignments in databases.
        
        Parameters:
        -----------
        peak_data : dict
            Dictionary containing peak information
            
        Returns:
        --------
        dict
            Corrected peak data dictionary
        """
        # Don't modify the original dictionary
        peak_data = peak_data.copy()
        
        # Only proceed if this is a centrosymmetric crystal 
        # (detected by presence of inversion symmetry in point group)
        is_centrosymmetric = False
        
        # Check if we have a current structure with point group info
        if hasattr(self, 'current_structure') and hasattr(self.current_structure, 'point_group'):
            # Point groups with inversion center contain these symbols
            centro_indicators = ['i', 'h', 'd', 'ci', 'cih', 'th', 'oh', 'dih']
            
            # Check point group
            point_group = self.current_structure.point_group.lower()
            if any(indicator in point_group for indicator in centro_indicators):
                is_centrosymmetric = True
            
            # Handle specific point groups by name
            centrosymmetric_groups = [
                '-1', '-3', '-3m', '-4', '-42m', '-4m2', '-6', '-6m2', 
                'mmm', '2/m', '4/m', '4/mmm', '6/m', '6/mmm', 'm-3', 'm-3m'
            ]
            if any(pg == self.current_structure.point_group for pg in centrosymmetric_groups):
                is_centrosymmetric = True
        
        # Mode correction mapping for centrosymmetric crystals
        if is_centrosymmetric:
            mode_corrections = {
                'A1u': 'A1g',  # Convert incorrectly assigned modes
                'A2u': 'A2g',
                'Eu': 'Eg',
                'A3u': 'A3g',
                'Bu': 'Bg',
                'B1u': 'B1g',
                'B2u': 'B2g',
                'B3u': 'B3g'
            }
            
            # Correct the assignments
            if 'character' in peak_data:
                character = peak_data['character']
                if character in mode_corrections:
                    peak_data['character'] = mode_corrections[character]
                    print(f"Corrected mode assignment: {character} → {peak_data['character']}")
        
        return peak_data


class CrystalStructure:
    def __init__(self, cif_path, born_charges_db=None):
        self.cif_path = cif_path
        self.lattice = None  # a, b, c, alpha, beta, gamma
        self.asymmetric_atoms = []  # atoms in the asymmetric unit
        self.atoms = []      # list of dicts: {label, type, fract_x, fract_y, fract_z, cart_x, cart_y, cart_z}
        self.bonds = []      # list of (atom1_idx, atom2_idx, bond_vector, bond_length)
        self.born_charges = {}  # atom_type -> 3x3 tensor (from DB)
        self.born_charges_db = born_charges_db
        self.eigenvectors = {}  # atom_type -> list of eigenvectors
        self.eigenvalues = {}   # atom_type -> list of eigenvalues
        self.symops = []    # symmetry operations
        self.parse_cif()
        self.frac_to_cartesian()
        if born_charges_db:
            self.map_born_charges()

    def parse_cif_original(self):
        """Original method to parse lattice, atoms, and symmetry operations from a CIF file."""
        with open(self.cif_path, 'r') as f:
            lines = f.readlines()
        
        # Parse lattice parameters
        lattice = {}
        for line in lines:
            if line.startswith('_cell_length_a'):
                lattice['a'] = float(line.split()[1])
            elif line.startswith('_cell_length_b'):
                lattice['b'] = float(line.split()[1])
            elif line.startswith('_cell_length_c'):
                lattice['c'] = float(line.split()[1])
            elif line.startswith('_cell_angle_alpha'):
                lattice['alpha'] = float(line.split()[1])
            elif line.startswith('_cell_angle_beta'):
                lattice['beta'] = float(line.split()[1])
            elif line.startswith('_cell_angle_gamma'):
                lattice['gamma'] = float(line.split()[1])
        self.lattice = lattice
        
        # Parse symmetry operations
        self.symops = []
        symop_section = False
        for i, line in enumerate(lines):
            if '_symmetry_equiv_pos_as_xyz' in line or '_space_group_symop_operation_xyz' in line:
                symop_section = True
                continue
            if symop_section and line.strip() and not line.startswith('_'):
                # Extract the symmetry operation string
                op_string = line.strip().split("'")[-1] if "'" in line else line.strip()
                if op_string and ',' in op_string:  # Ensure it's a valid symop
                    self.symops.append(op_string)
            if symop_section and (line.startswith('_') or 'loop_' in line):
                if len(self.symops) > 0:  # End of symops section
                    symop_section = False
                    
        # If no symmetry operations found, add the identity
        if not self.symops:
            self.symops = ['x,y,z']
            
        print(f"Found {len(self.symops)} symmetry operations:")
        for op in self.symops[:5]:  # Show first few operations
            print(f"  {op}")
        if len(self.symops) > 5:
            print(f"  ... and {len(self.symops)-5} more")
        
        # Parse atom positions
        self.asymmetric_atoms = []  # Store only the asymmetric unit
        atom_loop_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('_atom_site_label'):
                atom_loop_start = i
                break
                
        if atom_loop_start is not None:
            # Find all loop header lines
            headers = []
            j = atom_loop_start
            while j < len(lines) and lines[j].strip().startswith('_'):
                headers.append(lines[j].strip())
                j += 1
                
            # Map header names to indices
            header_map = {h: idx for idx, h in enumerate(headers)}
            
            # Find required columns
            label_idx = None
            x_idx = y_idx = z_idx = None
            for idx, h in enumerate(headers):
                if h.lower() in ['_atom_site_label', '_atom_site_type_symbol']:
                    label_idx = idx
                elif '_fract_x' in h.lower():
                    x_idx = idx
                elif '_fract_y' in h.lower():
                    y_idx = idx
                elif '_fract_z' in h.lower():
                    z_idx = idx
                    
            # Read atom lines
            while j < len(lines):
                l = lines[j].strip()
                if not l or l.startswith('loop_') or l.startswith('_'):
                    break
                    
                parts = l.split()
                if len(parts) >= max(label_idx, x_idx, y_idx, z_idx) + 1:
                    label = parts[label_idx]
                    fract_x = float(parts[x_idx])
                    fract_y = float(parts[y_idx])
                    fract_z = float(parts[z_idx])
                    atom_type = re.sub(r'[^A-Za-z]', '', label)
                    
                    self.asymmetric_atoms.append({
                        'label': label,
                        'type': atom_type,
                        'fract_x': fract_x,
                        'fract_y': fract_y,
                        'fract_z': fract_z
                    })
                j += 1
        
        # Apply symmetry operations to generate full unit cell
        self.generate_unit_cell()

    def generate_unit_cell(self):
        """Apply symmetry operations to generate all atoms in the unit cell."""
        if not self.asymmetric_atoms:
            print("ERROR: No atoms found in asymmetric unit!")
            self.atoms = []
            return
        
        print(f"Generating unit cell from {len(self.asymmetric_atoms)} asymmetric atoms using {len(self.symops)} symmetry operations")
        
        # Storage for all atoms including symmetry-generated ones
        self.atoms = []
        unique_positions = set()  # To track unique positions
        
        for atom in self.asymmetric_atoms:
            x, y, z = atom['fract_x'], atom['fract_y'], atom['fract_z']
            atom_type = atom['type']
            label = atom['label']
            
            for i, symop in enumerate(self.symops):
                try:
                    # Apply symmetry operation
                    new_pos = self.apply_symop(symop, x, y, z)
                    new_x, new_y, new_z = new_pos
                    
                    # Ensure coordinates are within [0,1)
                    new_x = new_x % 1.0
                    new_y = new_y % 1.0
                    new_z = new_z % 1.0
                    
                    # Check if this position is unique (with tolerance)
                    pos_tuple = (round(new_x, 6), round(new_y, 6), round(new_z, 6))
                    if pos_tuple in unique_positions:
                        continue
                        
                    unique_positions.add(pos_tuple)
                    
                    # Add to atoms list
                    self.atoms.append({
                        'label': f"{label}_{i+1}",
                        'type': atom_type,
                        'fract_x': new_x,
                        'fract_y': new_y,
                        'fract_z': new_z
                    })
                except Exception as e:
                    print(f"Error applying symop {symop} to atom {label}: {e}")
        
        print(f"Generated {len(self.atoms)} atoms in the unit cell")
        
    def generate_expanded_cell(self, shell=1):
        """Generate an expanded supercell for better bond finding.
        
        Parameters:
        -----------
        shell : int
            Number of unit cell repetitions in each direction.
            shell=1 gives a 3x3x3 supercell.
            
        Returns:
        --------
        list
            List of expanded atoms (including original cell atoms).
        """
        if not self.atoms:
            print("ERROR: No atoms in unit cell to expand!")
            return []
            
        # First ensure we have cartesian coordinates
        if 'cart_x' not in self.atoms[0]:
            self.frac_to_cartesian()
            
        print(f"Generating {(2*shell+1)}x{(2*shell+1)}x{(2*shell+1)} supercell for bond finding...")
        
        # Get lattice vectors in cartesian coordinates
        from math import cos, sin, pi, sqrt
        a, b, c = self.lattice['a'], self.lattice['b'], self.lattice['c']
        alpha = self.lattice['alpha'] * pi / 180
        beta = self.lattice['beta'] * pi / 180
        gamma = self.lattice['gamma'] * pi / 180
        
        v_a = np.array([a, 0, 0])
        v_b = np.array([b * cos(gamma), b * sin(gamma), 0])
        cx = c * cos(beta)
        cy = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)
        cz = sqrt(c**2 - cx**2 - cy**2)
        v_c = np.array([cx, cy, cz])
        
        # Generate the supercell
        expanded_atoms = []
        
        # Loop through unit cell repetitions
        for i in range(-shell, shell+1):
            for j in range(-shell, shell+1):
                for k in range(-shell, shell+1):
                    # Skip if this is not a boundary cell and we're only looking at boundaries
                    if shell == 1 and i == 0 and j == 0 and k == 0:
                        # Always include the central cell
                        pass
                        
                    # Calculate the offset for this cell
                    offset = i * v_a + j * v_b + k * v_c
                    
                    # Add atoms from this cell
                    for atom in self.atoms:
                        # Create a copy of the atom
                        new_atom = atom.copy()
                        
                        # Add cell offset to cartesian position
                        if 'cart_x' in atom:
                            pos = np.array([atom['cart_x'], atom['cart_y'], atom['cart_z']])
                            new_pos = pos + offset
                            new_atom['cart_x'] = new_pos[0]
                            new_atom['cart_y'] = new_pos[1]
                            new_atom['cart_z'] = new_pos[2]
                            
                            # Update fractional coordinates
                            if i != 0 or j != 0 or k != 0:
                                new_atom['fract_x'] = atom['fract_x'] + i
                                new_atom['fract_y'] = atom['fract_y'] + j
                                new_atom['fract_z'] = atom['fract_z'] + k
                        else:
                            # If no cartesian coords, update fractional and convert
                            new_atom['fract_x'] = atom['fract_x'] + i
                            new_atom['fract_y'] = atom['fract_y'] + j
                            new_atom['fract_z'] = atom['fract_z'] + k
                            
                            # Convert to cartesian
                            frac = np.array([new_atom['fract_x'], new_atom['fract_y'], new_atom['fract_z']])
                            cart = frac[0] * v_a + frac[1] * v_b + frac[2] * v_c
                            new_atom['cart_x'] = cart[0]
                            new_atom['cart_y'] = cart[1]
                            new_atom['cart_z'] = cart[2]
                        
                        # Add to expanded atoms list
                        new_atom['cell_offset'] = (i, j, k)
                        new_atom['label'] = f"{atom['label']}_{i}_{j}_{k}"
                        expanded_atoms.append(new_atom)
        
        print(f"Generated {len(expanded_atoms)} atoms in expanded cell")
        return expanded_atoms

    def apply_symop(self, symop, x, y, z):
        """Apply a symmetry operation to fractional coordinates."""
        # Parse the symop string
        ops = symop.split(',')
        if len(ops) != 3:
            raise ValueError(f"Invalid symmetry operation: {symop}")
        
        # Replace symbols with their values
        for i, op in enumerate(ops):
            op = op.lower().replace('x', f'({x})').replace('y', f'({y})').replace('z', f'({z})')
            
            # Handle fractions like 1/2
            op = re.sub(r'(\d+)/(\d+)', lambda m: f"{float(m.group(1))/float(m.group(2))}", op)
            
            # Evaluate the expression
            ops[i] = eval(op)
        
        return ops

    def frac_to_cartesian(self):
        """Convert fractional coordinates to Cartesian using the lattice."""
        from math import cos, sin, pi, sqrt
        a, b, c = self.lattice['a'], self.lattice['b'], self.lattice['c']
        alpha = self.lattice['alpha'] * pi / 180
        beta = self.lattice['beta'] * pi / 180
        gamma = self.lattice['gamma'] * pi / 180
        # Lattice vectors
        v_a = np.array([a, 0, 0])
        v_b = np.array([b * cos(gamma), b * sin(gamma), 0])
        cx = c * cos(beta)
        cy = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)
        cz = sqrt(c**2 - cx**2 - cy**2)
        v_c = np.array([cx, cy, cz])
        # Convert
        for atom in self.atoms:
            f = np.array([atom['fract_x'], atom['fract_y'], atom['fract_z']])
            cart = f[0] * v_a + f[1] * v_b + f[2] * v_c
            atom['cart_x'], atom['cart_y'], atom['cart_z'] = cart.tolist()

    def find_bonds(self, cutoff=3.0):
        """Find bonds by distance, including across unit cell boundaries."""
        self.bonds = []
        n = len(self.atoms)
        
        # Dictionary of typical bond lengths for common pairs
        typical_bonds = {
            ('Si', 'O'): 1.6,  # Silicon-Oxygen bond ~1.6 Å
            ('O', 'Si'): 1.6,
            ('Al', 'O'): 1.9,  # Aluminum-Oxygen bond ~1.9 Å 
            ('O', 'Al'): 1.9,
            ('C', 'O'): 1.4,   # Carbon-Oxygen bond ~1.4 Å
            ('O', 'C'): 1.4,
            ('C', 'C'): 1.5,   # Carbon-Carbon bond ~1.5 Å
            ('Si', 'Si'): 2.3, # Silicon-Silicon bond ~2.3 Å
            ('Ca', 'O'): 2.4,  # Calcium-Oxygen bond
            ('O', 'Ca'): 2.4,
            ('Mg', 'O'): 2.1,  # Magnesium-Oxygen bond
            ('O', 'Mg'): 2.1,
            ('Fe', 'O'): 2.0,  # Iron-Oxygen bond
            ('O', 'Fe'): 2.0,
            ('K', 'O'): 2.7,   # Potassium-Oxygen bond
            ('O', 'K'): 2.7,
            ('Na', 'O'): 2.4,  # Sodium-Oxygen bond
            ('O', 'Na'): 2.4,
            ('Ti', 'O'): 1.9,  # Titanium-Oxygen bond
            ('O', 'Ti'): 1.9,
            ('Mn', 'O'): 2.0,  # Manganese-Oxygen bond
            ('O', 'Mn'): 2.0,
            ('Li', 'O'): 2.0,  # Lithium-Oxygen bond
            ('O', 'Li'): 2.0,
            ('Zn', 'O'): 2.0,  # Zinc-Oxygen bond
            ('O', 'Zn'): 2.0,
            ('S', 'O'): 1.5,   # Sulfur-Oxygen bond
            ('O', 'S'): 1.5
            # These are the most common bonds in minerals/crystals
        }
        
        # Atomic radii database (in Å) for calculating bonds when no specific pair is defined
        atomic_radii = {
            # Alkali metals
            'H': 0.38, 'Li': 0.76, 'Na': 1.02, 'K': 1.38, 'Rb': 1.52, 'Cs': 1.67, 'Fr': 1.80,
            
            # Alkaline earth metals
            'Be': 0.45, 'Mg': 0.72, 'Ca': 1.00, 'Sr': 1.18, 'Ba': 1.35, 'Ra': 1.48,
            
            # Transition metals (first row)
            'Sc': 0.75, 'Ti': 0.61, 'V': 0.58, 'Cr': 0.52, 'Mn': 0.46, 'Fe': 0.65, 
            'Co': 0.61, 'Ni': 0.69, 'Cu': 0.73, 'Zn': 0.74,
            
            # Transition metals (second row)
            'Y': 0.90, 'Zr': 0.72, 'Nb': 0.64, 'Mo': 0.59, 'Tc': 0.56, 'Ru': 0.68, 
            'Rh': 0.67, 'Pd': 0.86, 'Ag': 1.15, 'Cd': 0.95,
            
            # Transition metals (third row)
            'La': 1.06, 'Hf': 0.71, 'Ta': 0.64, 'W': 0.60, 'Re': 0.63, 'Os': 0.63, 
            'Ir': 0.68, 'Pt': 0.80, 'Au': 1.37, 'Hg': 1.02,
            
            # Post-transition metals
            'Al': 0.54, 'Ga': 0.62, 'In': 0.81, 'Tl': 0.95, 'Sn': 0.69, 'Pb': 1.19, 'Bi': 1.03,
            
            # Metalloids
            'B': 0.23, 'Si': 0.40, 'Ge': 0.53, 'As': 0.58, 'Sb': 0.76, 'Te': 0.97, 'Po': 1.08,
            
            # Non-metals
            'C': 0.16, 'N': 0.13, 'O': 1.40, 'F': 1.33, 'P': 0.44, 'S': 1.84, 'Cl': 1.81, 
            'Se': 0.50, 'Br': 1.96, 'I': 2.20, 'At': 2.23,
            
            # Lanthanides
            'Ce': 1.03, 'Pr': 1.01, 'Nd': 0.99, 'Pm': 0.97, 'Sm': 0.96, 'Eu': 0.95, 
            'Gd': 0.94, 'Tb': 0.92, 'Dy': 0.91, 'Ho': 0.90, 'Er': 0.89, 'Tm': 0.88, 
            'Yb': 0.87, 'Lu': 0.86,
            
            # Actinides
            'Ac': 1.12, 'Th': 1.05, 'Pa': 1.00, 'U': 0.97, 'Np': 0.95, 'Pu': 0.93, 
            'Am': 0.92, 'Cm': 0.91, 'Bk': 0.90, 'Cf': 0.89, 'Es': 0.88, 'Fm': 0.87, 
            'Md': 0.86, 'No': 0.85, 'Lr': 0.84
        }
        
        # Summary counters
        bond_counts = {}
        all_bond_lengths = {}
        
        print(f"\n=== Bond Analysis (cutoff = {cutoff}Å) ===")
        
        # Get lattice vectors
        from math import cos, sin, pi, sqrt
        a, b, c = self.lattice['a'], self.lattice['b'], self.lattice['c']
        alpha = self.lattice['alpha'] * pi / 180
        beta = self.lattice['beta'] * pi / 180
        gamma = self.lattice['gamma'] * pi / 180
        
        v_a = np.array([a, 0, 0])
        v_b = np.array([b * cos(gamma), b * sin(gamma), 0])
        cx = c * cos(beta)
        cy = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)
        cz = sqrt(c**2 - cx**2 - cy**2)
        v_c = np.array([cx, cy, cz])
        
        # Generate supercell offsets for periodic boundary conditions
        # Check neighboring cells (27 total including the original)
        supercell_offsets = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue  # Skip the original cell for now
                    supercell_offsets.append(i * v_a + j * v_b + k * v_c)
        
        # First find bonds within the original unit cell
        print("Finding bonds within original unit cell...")
        for i in range(n):
            for j in range(i+1, n):
                type1 = self.atoms[i]['type']
                type2 = self.atoms[j]['type']
                label1 = self.atoms[i]['label']
                label2 = self.atoms[j]['label']
                
                pos1 = np.array([self.atoms[i]['cart_x'], self.atoms[i]['cart_y'], self.atoms[i]['cart_z']])
                pos2 = np.array([self.atoms[j]['cart_x'], self.atoms[j]['cart_y'], self.atoms[j]['cart_z']])
                
                bond_vec = pos2 - pos1
                bond_len = np.linalg.norm(bond_vec)
                
                # Use pair-specific cutoff if available
                pair_key = (type1, type2)
                pair_cutoff = cutoff
                if pair_key in typical_bonds:
                    expected_len = typical_bonds[pair_key]
                    pair_cutoff = expected_len * 1.5  # 50% margin
                else:
                    # Use atomic radii sum when direct bond length not available
                    radius1 = atomic_radii.get(type1, 1.0)
                    radius2 = atomic_radii.get(type2, 1.0)
                    expected_len = radius1 + radius2
                    pair_cutoff = min(expected_len * 1.2, cutoff)  # 20% margin but not exceeding global cutoff
                    
                # Store all bond lengths for reporting, even if not a bond
                bond_type_key = f"{type1}-{type2}"
                if bond_type_key not in all_bond_lengths:
                    all_bond_lengths[bond_type_key] = []
                all_bond_lengths[bond_type_key].append((label1, label2, bond_len))
                
                # Check if this is a bond
                if bond_len < pair_cutoff:
                    self.bonds.append((i, j, bond_vec / bond_len, bond_len))
                    
                    # Update bond count
                    if bond_type_key not in bond_counts:
                        bond_counts[bond_type_key] = 0
                    bond_counts[bond_type_key] += 1
        
        # Now find bonds across periodic boundaries
        print("Finding bonds across periodic boundaries...")
        original_bond_count = len(self.bonds)
        
        for i in range(n):
            type1 = self.atoms[i]['type']
            label1 = self.atoms[i]['label']
            pos1 = np.array([self.atoms[i]['cart_x'], self.atoms[i]['cart_y'], self.atoms[i]['cart_z']])
            
            for j in range(n):
                # Skip if same atom
                if i == j:
                    continue
                    
                type2 = self.atoms[j]['type']
                label2 = self.atoms[j]['label']
                pos2 = np.array([self.atoms[j]['cart_x'], self.atoms[j]['cart_y'], self.atoms[j]['cart_z']])
                
                # Check all supercell offsets
                for offset in supercell_offsets:
                    pos2_shifted = pos2 + offset
                    bond_vec = pos2_shifted - pos1
                    bond_len = np.linalg.norm(bond_vec)
                    
                    # Use pair-specific cutoff
                    pair_key = (type1, type2)
                    pair_cutoff = cutoff
                    if pair_key in typical_bonds:
                        expected_len = typical_bonds[pair_key]
                        pair_cutoff = expected_len * 1.5  # 50% margin
                    else:
                        # Use atomic radii sum when direct bond length not available
                        radius1 = atomic_radii.get(type1, 1.0)
                        radius2 = atomic_radii.get(type2, 1.0)
                        expected_len = radius1 + radius2
                        pair_cutoff = min(expected_len * 1.2, cutoff)  # 20% margin but not exceeding global cutoff
                    
                    # Check if this is a bond and is not too close to another existing bond
                    if bond_len < pair_cutoff:
                        # Check if this bond is new
                        is_new_bond = True
                        for existing_i, existing_j, _, existing_len in self.bonds:
                            if (i == existing_i and j == existing_j) or (i == existing_j and j == existing_i):
                                # If we already have this atom pair, only keep the shorter bond
                                if bond_len < existing_len:
                                    is_new_bond = True
                                else:
                                    is_new_bond = False
                                break
                                
                        if is_new_bond:
                            self.bonds.append((i, j, bond_vec / bond_len, bond_len))
                            
                            # Update bond count
                            bond_type_key = f"{type1}-{type2}"
                            if bond_type_key not in bond_counts:
                                bond_counts[bond_type_key] = 0
                            bond_counts[bond_type_key] += 1
                            
                            # Store for reporting
                            if bond_type_key not in all_bond_lengths:
                                all_bond_lengths[bond_type_key] = []
                            all_bond_lengths[bond_type_key].append((f"{label1}-{label2}(PBC)", f"offset={offset}", bond_len))
        
        print(f"Found {len(self.bonds) - original_bond_count} additional bonds across periodic boundaries")
        
        # Print all bond length statistics by atom type pair
        print("\nBond Length Analysis by Atom Type:")
        for bond_type, lengths in sorted(all_bond_lengths.items()):
            # Sort lengths
            lengths.sort(key=lambda x: x[2])
            
            # Get statistics
            bond_lengths = [l[2] for l in lengths]
            min_len = min(bond_lengths)
            max_len = max(bond_lengths)
            avg_len = sum(bond_lengths) / len(bond_lengths)
            
            # Expected length from dictionary
            types = bond_type.split('-')
            expected = typical_bonds.get((types[0], types[1]), None)
            expected_str = f"(expected: {expected:.2f}Å)" if expected else ""
            
            # Print summary
            print(f"\n{bond_type} {expected_str}:")
            print(f"  Range: {min_len:.3f}Å - {max_len:.3f}Å, Average: {avg_len:.3f}Å")
            print(f"  Count: {len(lengths)} pairs, {bond_counts.get(bond_type, 0)} bonds within cutoff")
            
            # Print shortest bonds (up to 5)
            print("  Shortest bonds:")
            for idx, (label1, label2, length) in enumerate(lengths[:5]):
                print(f"    {idx+1}. {label1}-{label2}: {length:.3f}Å")
        
        # Print summary of bonds found
        print(f"\nTotal bonds found: {len(self.bonds)}")
        for bond_type, count in sorted(bond_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {bond_type}: {count}")
        
        return len(self.bonds) > 0

    def map_born_charges(self):
        """Map atom types to their Born charge tensors and eigenvectors from the database."""
        tensors = {}
        eigenvectors = {}
        eigenvalues = {}
        
        # Default tensors for common atoms
        default_tensors = {
            'Si': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            'O': np.array([[-0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]]),
        }
        
        print("\n=== Born Charges Mapping ===")
        
        atom_types_in_structure = set(atom['type'] for atom in self.atoms)
        print(f"Atom types in structure: {atom_types_in_structure}")
        
        try:
            # Check if born_charges_db is a dictionary (mineral database format)
            if isinstance(self.born_charges_db, dict):
                for atom_type, data in self.born_charges_db.items():
                    if not isinstance(data, dict):
                        print(f"Warning: Invalid data format for {atom_type}")
                        continue
                        
                    # Initialize tensors for this atom type
                    tensors[atom_type] = np.zeros((3,3))
                    
                    # Extract Born charge tensor components
                    if 'born_charges' in data:
                        born_data = data['born_charges']
                        if isinstance(born_data, dict):
                            # Handle dictionary format
                            if 'xx' in born_data:
                                tensors[atom_type][0,0] = float(born_data['xx'])
                            if 'xy' in born_data:
                                tensors[atom_type][0,1] = float(born_data['xy'])
                            if 'xz' in born_data:
                                tensors[atom_type][0,2] = float(born_data['xz'])
                            if 'yy' in born_data:
                                tensors[atom_type][1,1] = float(born_data['yy'])
                            if 'yz' in born_data:
                                tensors[atom_type][1,2] = float(born_data['yz'])
                            if 'zz' in born_data:
                                tensors[atom_type][2,2] = float(born_data['zz'])
                        elif isinstance(born_data, (list, tuple)):
                            # Handle list/tuple format
                            if len(born_data) >= 9:  # Full 3x3 tensor
                                tensors[atom_type] = np.array(born_data[:9]).reshape(3,3)
                    
                    # Extract eigenvectors and eigenvalues
                    if 'eigenvectors' in data:
                        eigvec_data = data['eigenvectors']
                        if isinstance(eigvec_data, (list, tuple)) and len(eigvec_data) >= 9:
                            eigenvectors[atom_type] = np.array(eigvec_data[:9]).reshape(3,3).tolist()
                        else:
                            # Use identity matrix as default
                            eigenvectors[atom_type] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    
                    if 'eigenvalues' in data:
                        eigval_data = data['eigenvalues']
                        if isinstance(eigval_data, (list, tuple)) and len(eigval_data) >= 3:
                            eigenvalues[atom_type] = [float(x) for x in eigval_data[:3]]
                        else:
                            # Use default eigenvalues
                            eigenvalues[atom_type] = [1.0, 1.0, 1.0]
            
            # Fill in defaults for missing atom types
            for atom_type in atom_types_in_structure:
                if atom_type not in tensors or np.all(np.abs(tensors[atom_type]) < 1e-10):
                    print(f"\nUsing default tensor for {atom_type}")
                    tensors[atom_type] = default_tensors.get(atom_type, np.eye(3))
                    print(tensors[atom_type])
                
                if atom_type not in eigenvectors:
                    eigenvectors[atom_type] = np.eye(3)
                
                if atom_type not in eigenvalues:
                    eigenvalues[atom_type] = [1.0, 1.0, 1.0]
            
            # Print debug information
            print(f"\nLoaded Born charges for {len(tensors)} atom types")
            for atom_type in tensors:
                if atom_type in atom_types_in_structure:
                    print(f"\nAtom type: {atom_type}")
                    print("Born charge tensor:")
                    print(tensors[atom_type])
                    print("Eigenvalues:", eigenvalues.get(atom_type, [1.0, 1.0, 1.0]))
        
        except Exception as e:
            print(f"Error in map_born_charges: {str(e)}")
            # Set default values if mapping fails
            import traceback
            traceback.print_exc()
        
        # Add fallback values for any missing Born charges
        for atom in self.atoms:
            atom_type = atom['type']
            if atom_type not in tensors or np.all(np.abs(tensors[atom_type]) < 1e-10):
                print(f"Adding fallback Born charge for {atom_type}")
                # Use a reasonable default tensor for missing charges
                if atom_type in ['O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I']:
                    # Electronegative elements
                    tensors[atom_type] = np.diag([-1.0, -1.0, -1.0])
                else:
                    # Electropositive elements
                    tensors[atom_type] = np.diag([1.0, 1.0, 1.0])
                
                # Add corresponding eigenvectors and eigenvalues
                eigenvectors[atom_type] = np.eye(3)
                eigenvalues[atom_type] = [-1.0, -1.0, -1.0] if atom_type in ['O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I'] else [1.0, 1.0, 1.0]
        
        self.born_charges = tensors
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues

    def bond_polarizability_derivative(self, atom1_idx, atom2_idx, bond_vec):
        """Calculate bond polarizability derivative using a physically-based model.
        
        This uses an improved model that properly accounts for bond direction,
        Born charges, and ionic polarizability effects.
        
        Parameters:
        -----------
        atom1_idx, atom2_idx : int
            Indices of the two atoms forming the bond
        bond_vec : numpy.ndarray
            Unit vector along the bond direction
            
        Returns:
        --------
        numpy.ndarray
            3x3 tensor representing the bond polarizability derivative
        """
        type1 = self.atoms[atom1_idx]['type']
        type2 = self.atoms[atom2_idx]['type']
        
        # Get Born charge tensors
        if not hasattr(self, 'born_charges') or type1 not in self.born_charges or type2 not in self.born_charges:
            # Use identity matrix as default if Born charges not available
            z1 = np.eye(3)
            z2 = np.eye(3)
        else:
            z1 = self.born_charges[type1]
            z2 = self.born_charges[type2]
        
        # Get bond direction as a column vector for tensor operations
        r = bond_vec.reshape(3, 1)  # Convert to column vector
        r_row = bond_vec.reshape(1, 3)  # Convert to row vector
        
        # Parameters for bond polarizability model
        # These parameters control the relative strength of different contributions
        ionic_strength = 1.0
        covalent_strength = 0.8
        
        # Construct the bond polarizability tensor using multiple physical contributions
        
        # 1. Ionic contribution: based on the Born charges of each atom
        # This represents how the atomic charges respond to electric field
        ionic_term = ionic_strength * (z1 @ r @ r_row + r @ r_row @ z2)
        
        # 2. Bond stretching contribution: represents how the bond polarizability 
        # changes when stretched along the bond direction
        # This is a standard bond polarizability model term
        alpha_parallel = 1.0  # Polarizability change along bond
        alpha_perp = 0.3      # Polarizability change perpendicular to bond
        
        # Calculate the anisotropic part using the bond direction
        stretch_term = (alpha_parallel - alpha_perp) * (r @ r_row)
        
        # 3. Construct the identity part (isotropic contribution)
        iso_term = alpha_perp * np.eye(3)
        
        # 4. Covalent contribution: reflects electron sharing between atoms
        # This term depends on the bond type
        bond_type = f"{type1}-{type2}"
        
        # Adjust bond coefficient based on bond type
        if bond_type in ['Si-O', 'O-Si']:
            # Strong covalent character for Si-O bonds
            cov_factor = 1.2
        elif bond_type in ['Al-O', 'O-Al']:
            # Slightly less covalent for Al-O
            cov_factor = 0.9
        else:
            # Default value for other bonds
            cov_factor = 0.5
            
        covalent_term = covalent_strength * cov_factor * (r @ r_row)
        
        # 5. Combine all contributions
        deriv_tensor = ionic_term + stretch_term + iso_term + covalent_term
        
        # Apply any special handling for specific atom types (optional)
        # For example, special contributions for tetrahedral coordination, etc.
        
        return deriv_tensor

    def calculate_raman_tensor(self, character=None):
        """
        Calculate the Raman tensor with enhanced debugging.
        
        Parameters:
        -----------
        character : str, optional
            Symmetry character to generate tensor for (e.g., 'A1', 'E', 'T2')
            If None, returns the general tensor.
        
        Returns:
        --------
        numpy.ndarray or list of numpy.ndarray
            If character is None, returns a single tensor.
            If character is 'all', returns a list of tensors for all symmetry types.
        """
        if not self.bonds:
            print("WARNING: No bonds found. Performing bond search...")
            self.find_bonds(cutoff=3.0)
            if not self.bonds:
                print("ERROR: Still no bonds found. Cannot calculate Raman tensor!")
                return np.zeros((3,3))
        
        # Initialize Raman tensor
        raman_tensor = np.zeros((3,3))
        
        print("\n=== Raman Tensor Calculation ===")
        print(f"Processing {len(self.bonds)} bonds...")
        
        # Track contribution from each bond
        contributions = []
        
        # Sum contributions from all bonds
        for idx, (i, j, bond_vec, bond_len) in enumerate(self.bonds):
            type1 = self.atoms[i]['type']
            type2 = self.atoms[j]['type']
            
            # Skip if either atom type is missing Born charges
            if type1 not in self.born_charges or type2 not in self.born_charges:
                print(f"WARNING: Missing Born charges for bond {type1}-{type2}, skipping")
                continue
            
            try:
                deriv = self.bond_polarizability_derivative(i, j, bond_vec)
                raman_tensor += deriv
                
                # Store contribution info
                norm = np.linalg.norm(deriv)
                contributions.append((f"{type1}-{type2}", bond_len, norm))
                
                if idx < 5:  # Show first 5 in detail
                    print(f"\nBond {idx+1}: {type1}-{type2}, length={bond_len:.3f}")
                    print("Derivative tensor:")
                    print(deriv)
            except Exception as e:
                print(f"ERROR processing bond {type1}-{type2}: {str(e)}")
        
        # Show contributions by magnitude
        print("\nTop bond contributions:")
        sorted_contributions = sorted(contributions, key=lambda x: x[2], reverse=True)
        for bond_type, length, norm in sorted_contributions[:5]:
            print(f"  {bond_type} ({length:.3f}Å): {norm:.6f}")
        
        # Normalize the tensor with safety check
        norm = np.linalg.norm(raman_tensor)
        print(f"\nRaw tensor norm: {norm:.6f}")
        
        if norm > 1e-6:  # Increased threshold from 1e-10 to 1e-6
            raman_tensor /= norm
            print("Normalized tensor:")
        else:
            # Instead of just warning, provide a default non-zero tensor
            print("WARNING: Zero or near-zero tensor! Applying fallback tensor.")
            # Create a fallback tensor with small but non-zero values
            raman_tensor = np.array([
                [0.8, 0.2, 0.1],
                [0.2, 0.8, 0.1],
                [0.1, 0.1, 0.6]
            ])
        
        print(raman_tensor)
        
        # If no character specified, return the calculated tensor
        if character is None:
            return raman_tensor
            
        # Get crystal point group
        point_group = getattr(self, 'point_group', None)
        if point_group is None:
            # Try to extract from space group
            space_group = getattr(self, 'space_group', '')
            if space_group:
                # This is a simplified mapping and may not be accurate for all cases
                if '432' in space_group or 'm-3m' in space_group:
                    point_group = 'm-3m'  # Cubic
                elif '4/mmm' in space_group:
                    point_group = '4/mmm'  # Tetragonal
                elif '6/mmm' in space_group:
                    point_group = '6/mmm'  # Hexagonal
                elif '-3m' in space_group:
                    point_group = '-3m'    # Trigonal
                elif 'mmm' in space_group:
                    point_group = 'mmm'    # Orthorhombic
                elif '2/m' in space_group:
                    point_group = '2/m'    # Monoclinic
                else:
                    point_group = '1'      # Default to Triclinic
        
        print(f"Point group for tensor calculation: {point_group}")
        
        # Generate multiple tensors based on point group symmetries
        if character == 'all':
            # Return a list of tensors for different characters
            tensors = []
            activities = []
            
            # Different point groups have different character sets
            if point_group in ['32', '3m', '3']:  # Trigonal (D3, C3v, C3)
                print(f"Generating tensors for trigonal point group: {point_group}")
                
                # A1/A1g mode (totally symmetric)
                a1_tensor = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]) * np.linalg.norm(raman_tensor)
                tensors.append(a1_tensor)
                activities.append("A1")
                
                # A2/A2g mode (if present)
                a2_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.1  # Often not Raman active
                tensors.append(a2_tensor)
                activities.append("A2")
                
                # E/Eg modes (doubly degenerate)
                e_tensor1 = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.8
                tensors.append(e_tensor1)
                activities.append("E")
                
                e_tensor2 = np.array([
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7
                tensors.append(e_tensor2)
                activities.append("E")
                
            elif point_group in ['-3', '-3m', '-32/m']:  # Trigonal with inversion (D3d)
                print(f"Generating tensors for D3d trigonal point group: {point_group}")
                
                # A1g mode
                a1g_tensor = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]) * np.linalg.norm(raman_tensor)
                tensors.append(a1g_tensor)
                activities.append("A1g")
                
                # A2g mode
                a2g_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.1
                tensors.append(a2g_tensor)
                activities.append("A2g")
                
                # Eg modes
                eg_tensor1 = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.8
                tensors.append(eg_tensor1)
                activities.append("Eg")
                
                eg_tensor2 = np.array([
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7
                tensors.append(eg_tensor2)
                activities.append("Eg")
                
            elif point_group in ['m-3m', 'm-3', '432', '-43m', '23']:  # Cubic
                # A1g mode
                a1g_tensor = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]) * np.linalg.norm(raman_tensor)
                tensors.append(a1g_tensor)
                activities.append("A1g")
                
                # Eg modes
                eg_tensor1 = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -2.0]
                ]) * np.linalg.norm(raman_tensor) * 0.6
                tensors.append(eg_tensor1)
                activities.append("Eg")
                
                eg_tensor2 = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.6
                tensors.append(eg_tensor2)
                activities.append("Eg")
                
                # T2g modes
                t2g_tensor1 = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.5
                tensors.append(t2g_tensor1)
                activities.append("T2g")
                
                t2g_tensor2 = np.array([
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.5
                tensors.append(t2g_tensor2)
                activities.append("T2g")
                
                t2g_tensor3 = np.array([
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.5
                tensors.append(t2g_tensor3)
                activities.append("T2g")
                
            elif point_group in ['6/mmm', '6/m', '622', '6mm', '-6m2', '-62m', '6', '-6']:  # Hexagonal (D6h)
                print(f"Generating tensors for hexagonal point group: {point_group}")
                
                # A1g mode (totally symmetric)
                a1g_tensor = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]) * np.linalg.norm(raman_tensor)
                tensors.append(a1g_tensor)
                activities.append("A1g")
                
                # A2g mode (antisymmetric)
                a2g_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7  # Usually not Raman active
                tensors.append(a2g_tensor)
                activities.append("A2g")
                
                # B1g mode
                b1g_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.6  # Usually not present in D6h
                tensors.append(b1g_tensor)
                activities.append("B1g")
                
                # B2g mode
                b2g_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.6  # Usually not present in D6h
                tensors.append(b2g_tensor)
                activities.append("B2g")
                
                # E1g modes (doubly degenerate)
                e1g_tensor1 = np.array([
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.8
                tensors.append(e1g_tensor1)
                activities.append("E1g")
                
                e1g_tensor2 = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.8
                tensors.append(e1g_tensor2)
                activities.append("E1g")
                
                # E2g modes (doubly degenerate)
                e2g_tensor1 = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.8
                tensors.append(e2g_tensor1)
                activities.append("E2g")
                
                e2g_tensor2 = np.array([
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.8
                tensors.append(e2g_tensor2)
                activities.append("E2g")
                
                # Add u-type modes (IR active)
                a1u_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.1  # Usually not Raman active
                tensors.append(a1u_tensor)
                activities.append("A1u")
                
                a2u_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.1  # Usually not Raman active
                tensors.append(a2u_tensor)
                activities.append("A2u")
                
                e1u_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.1  # Usually not Raman active
                tensors.append(e1u_tensor)
                activities.append("E1u")
                
                e2u_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.1  # Usually not Raman active
                tensors.append(e2u_tensor)
                activities.append("E2u")
                
            elif point_group in ['4/mmm', '4/m', '422', '4mm', '-42m', '-4m2', '4', '-4']:  # Tetragonal
                # A1g mode
                a1g_tensor = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]) * np.linalg.norm(raman_tensor)
                tensors.append(a1g_tensor)
                activities.append("A1g")
                
                # B1g mode
                b1g_tensor = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7
                tensors.append(b1g_tensor)
                activities.append("B1g")
                
                # B2g mode
                b2g_tensor = np.array([
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7
                tensors.append(b2g_tensor)
                activities.append("B2g")
                
                # Eg modes
                eg_tensor = np.array([
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.6
                tensors.append(eg_tensor)
                activities.append("Eg")
                
            elif point_group in ['mmm', '2/m2/m2/m', 'D2h']:  # Orthorhombic
                print(f"Generating tensors for orthorhombic point group: {point_group}")
                
                # Ag mode
                ag_tensor = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]) * np.linalg.norm(raman_tensor)
                tensors.append(ag_tensor)
                activities.append("Ag")
                
                # B1g mode
                b1g_tensor = np.array([
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7
                tensors.append(b1g_tensor)
                activities.append("B1g")
                
                # B2g mode
                b2g_tensor = np.array([
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7
                tensors.append(b2g_tensor)
                activities.append("B2g")
                
                # B3g mode
                b3g_tensor = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7
                tensors.append(b3g_tensor)
                activities.append("B3g")
                
            elif point_group in ['2/m', 'C2h']:  # Monoclinic
                print(f"Generating tensors for monoclinic point group: {point_group}")
                
                # Ag mode
                ag_tensor = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]) * np.linalg.norm(raman_tensor)
                tensors.append(ag_tensor)
                activities.append("Ag")
                
                # Bg mode
                bg_tensor = np.array([
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7
                tensors.append(bg_tensor)
                activities.append("Bg")
                
            elif point_group in ['1', '-1']:  # Triclinic
                print(f"Generating tensors for triclinic point group: {point_group}")
                
                # For triclinic, the characters are simple A or Ag
                if point_group == '-1':  # With inversion
                    # Ag mode (general)
                    ag_tensor = np.array([
                        [1.0, 0.3, 0.3],
                        [0.3, 1.0, 0.3],
                        [0.3, 0.3, 1.0]
                    ]) * np.linalg.norm(raman_tensor)
                    tensors.append(ag_tensor)
                    activities.append("Ag")
                else:
                    # A mode (general)
                    a_tensor = np.array([
                        [1.0, 0.3, 0.3],
                        [0.3, 1.0, 0.3],
                        [0.3, 0.3, 1.0]
                    ]) * np.linalg.norm(raman_tensor)
                    tensors.append(a_tensor)
                    activities.append("A")
                
            else:  # Default case - generate generic tensors
                print(f"Using default tensors for point group: {point_group}")
                
                # Totally symmetric (A1/A1g) mode
                a1_tensor = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]) * np.linalg.norm(raman_tensor)
                tensors.append(a1_tensor)
                activities.append("A1")
                
                # Non-totally symmetric mode with xy components
                e_tensor = np.array([
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.7
                tensors.append(e_tensor)
                activities.append("E")
                
                # Non-totally symmetric mode with xz components
                t_tensor = np.array([
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.6
                tensors.append(t_tensor)
                activities.append("T")
            
            # Save activities for visualization
            self.activities = activities
            self.raman_tensor = tensors
            return tensors
        
        # Look for specific character
        # First, handle general/common cases with shared tensor forms
        if character.upper() in ["A1", "A1G", "AG"]:
            # Totally symmetric mode
            return np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]) * np.linalg.norm(raman_tensor)
        
        # E-type modes (various notations)
        elif character.upper() in ["E", "EG"]:
            # E-type mode (xy plane)
            return np.array([
                [1.0, 0.3, 0.0],
                [0.3, -1.0, 0.0],
                [0.0, 0.0, 0.0]
            ]) * np.linalg.norm(raman_tensor) * 0.7
        
        # T-type modes for cubic
        elif character.upper() in ["T", "T2", "T2G"]:
            # T-type mode
            return np.array([
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0]
            ]) * np.linalg.norm(raman_tensor) * 0.6
            
        # Specific modes for hexagonal (D6h)
        elif character.upper() in ["E1G"]:
            # E1g mode for hexagonal (xz plane)
            return np.array([
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0]
            ]) * np.linalg.norm(raman_tensor) * 0.8
        elif character.upper() in ["E2G"]:
            # E2g mode for hexagonal (xy plane)
            return np.array([
                [0.7, 0.5, 0.0],
                [0.5, -0.7, 0.0],
                [0.0, 0.0, 0.0]
            ]) * np.linalg.norm(raman_tensor) * 0.8
            
        # Asymmetric modes (usually weak/inactive)
        elif character.upper() in ["A2", "A2G"]:
            # A2g mode - typically inactive in Raman
            return np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ]) * np.linalg.norm(raman_tensor) * 0.1
            
        # B-type modes for orthorhombic/tetragonal
        elif character.upper() in ["B1G", "B2G", "B3G"]:
            if character.upper() == "B1G":
                # B1g mode (xy components)
                return np.array([
                    [0.6, 0.0, 0.0],
                    [0.0, -0.6, 0.0],
                    [0.0, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.5
            elif character.upper() == "B2G":
                # B2g mode (xz components)
                return np.array([
                    [0.0, 0.0, 0.6],
                    [0.0, 0.0, 0.0],
                    [0.6, 0.0, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.5
            else:  # B3G
                # B3g mode (yz components)
                return np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.6],
                    [0.0, 0.6, 0.0]
                ]) * np.linalg.norm(raman_tensor) * 0.5
                
        # Monoclinic Bg mode
        elif character.upper() == "BG":
            # Bg mode for monoclinic
            return np.array([
                [0.0, 0.6, 0.0],
                [0.6, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ]) * np.linalg.norm(raman_tensor) * 0.5
            
        # Inactive u-type modes (IR active, Raman inactive)
        elif "U" in character.upper():
            # u-type modes are typically IR active but Raman inactive
            return np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ]) * np.linalg.norm(raman_tensor) * 0.1
            
        else:
            # Return the general tensor if character not recognized
            print(f"Character '{character}' not specifically handled; using general tensor")
            return raman_tensor

    def calculate_raman_spectrum(self, orientation=(0, 0, 0), polarization='VV'):
        """Calculate Raman spectrum for a given crystal orientation and polarization.
        
        Parameters:
        -----------
        orientation : tuple (phi, theta, psi)
            Euler angles in degrees for crystal orientation
        polarization : str
            Polarization configuration: 'VV', 'VH', 'HV', 'HH'
            
        Returns:
        --------
        list of (frequency, intensity, character) tuples
        """
        if not hasattr(self, 'raman_tensor') or self.raman_tensor is None:
            self.raman_tensor = self.calculate_raman_tensor()
        
        # Convert Euler angles to rotation matrix
        phi, theta, psi = [np.radians(angle) for angle in orientation]
        
        # Calculate rotation matrix using Euler angles
        R_z1 = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])
        
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        R_z2 = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        # Full rotation matrix
        R = R_z2 @ R_y @ R_z1
        
        # Rotate the Raman tensor
        rotated_tensor = R @ self.raman_tensor @ R.T
        
        # Define polarization vectors
        if polarization == 'VV':
            e_in = np.array([0, 1, 0])  # Vertical
            e_out = np.array([0, 1, 0])  # Vertical
        elif polarization == 'VH':
            e_in = np.array([0, 1, 0])   # Vertical
            e_out = np.array([1, 0, 0])  # Horizontal
        elif polarization == 'HV':
            e_in = np.array([1, 0, 0])   # Horizontal
            e_out = np.array([0, 1, 0])  # Vertical
        elif polarization == 'HH':
            e_in = np.array([1, 0, 0])   # Horizontal
            e_out = np.array([1, 0, 0])  # Horizontal
        elif polarization == 'UNPOL':
            # For unpolarized light, we'll calculate the average of all polarization combinations
            # This will be handled separately below
            e_in = None
            e_out = None
        else:
            raise ValueError(f"Unknown polarization: {polarization}")
        
        # Get frequencies from the mineral database if available
        if hasattr(self, 'frequencies') and len(self.frequencies) > 0:
            frequencies = self.frequencies
            base_intensities = self.intensities if hasattr(self, 'intensities') and len(self.intensities) > 0 else np.ones(len(frequencies))
            activities = self.activities if hasattr(self, 'activities') and len(self.activities) > 0 else [""] * len(frequencies)
        else:
            # Use placeholder frequencies
            frequencies = [i * 100 for i in range(1, 11)]  # Example: 100, 200, ..., 1000 cm⁻¹
            base_intensities = np.ones(len(frequencies))
            activities = [""] * len(frequencies)
        
        # Calculate intensity for each frequency mode using activity character
        mode_intensities = []
        for i, (freq, base_intensity, activity) in enumerate(zip(frequencies, base_intensities, activities)):
            # Basic intensity from polarizability calculation
            if polarization == 'UNPOL':
                # For unpolarized light, average over all polarization configurations
                polarization_configs = [
                    (np.array([0, 1, 0]), np.array([0, 1, 0])),  # VV
                    (np.array([0, 1, 0]), np.array([1, 0, 0])),  # VH
                    (np.array([1, 0, 0]), np.array([0, 1, 0])),  # HV
                    (np.array([1, 0, 0]), np.array([1, 0, 0]))   # HH
                ]
                avg_intensity = 0.0
                for e_in, e_out in polarization_configs:
                    # Calculate intensity for this polarization configuration and make sure it's a scalar
                    intensity_value = np.abs(e_out @ rotated_tensor @ e_in) ** 2
                    # If it's a tensor result, take the mean to get a scalar value
                    if isinstance(intensity_value, np.ndarray):
                        intensity_value = float(np.mean(intensity_value))
                    avg_intensity += intensity_value
                basic_intensity = float(avg_intensity / len(polarization_configs))
            else:
                # Calculate intensity for standard polarization
                intensity_value = np.abs(e_out @ rotated_tensor @ e_in) ** 2
                # Ensure we have a scalar value
                if isinstance(intensity_value, np.ndarray):
                    basic_intensity = float(np.mean(intensity_value))
                else:
                    basic_intensity = float(intensity_value)
            
            # Modify intensity based on mode character and orientation
            if activity.strip():
                # Scale based on whether this is an A1, E, or other type mode
                if 'A1' in activity or 'A₁' in activity:
                    # A1 mode: Strong dependence on orientation
                    character_factor = (np.cos(theta))**2
                elif 'E' in activity:
                    # E mode: Less orientation-dependent
                    character_factor = 0.5 + 0.5 * np.sin(theta)**2
                elif 'A2' in activity or 'A₂' in activity:
                    # A2 mode: Often vanishing in certain orientations
                    character_factor = np.sin(theta)**2
                else:
                    character_factor = 1.0
                
                # Apply the character factor + add random variation by mode
                # This adds more differentiation between modes for better optimization
                mode_specific_factor = 0.5 + np.sin(phi + 0.1*i)**2 + np.cos(psi + 0.2*i)**2
                final_intensity = basic_intensity * character_factor * mode_specific_factor * base_intensity
            else:
                # No character info, just use basic calculation
                final_intensity = basic_intensity * base_intensity
            
            mode_intensities.append(final_intensity)
        
        # Normalize intensities
        max_intensity = np.max(mode_intensities) if isinstance(mode_intensities[0], np.ndarray) else max(mode_intensities)
        if max_intensity > 0:
            mode_intensities = [float(i) / max_intensity for i in mode_intensities]
        
        # Return list of (frequency, intensity, character)
        return [(freq, intensity, activity) for freq, intensity, activity in zip(frequencies, mode_intensities, activities)]
    
    def calculate_orientation_dependent_intensity(self, frequency_idx, orientation, polarization='VV'):
        """Calculate intensity for a single frequency at given orientation."""
        spectrum = self.calculate_raman_spectrum(orientation, polarization)
        if frequency_idx < len(spectrum):
            return spectrum[frequency_idx][1]
        return 0.0

    def parse_cif_with_pymatgen(self):
        """Parse CIF file using pymatgen for proper crystallographic handling."""
        try:
            import importlib.util
            pymatgen_spec = importlib.util.find_spec("pymatgen")
            if pymatgen_spec is None:
                print("Pymatgen not installed. Use pip install pymatgen to enable advanced crystallographic analysis.")
                return False
                
            from pymatgen.io.cif import CifParser
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            import numpy as np
            
            print("Using pymatgen for advanced crystallographic analysis...")
            
            # Parse the CIF file
            parser = CifParser(self.cif_path)
            structure = parser.get_structures()[0]
            
            # Get space group info
            sga = SpacegroupAnalyzer(structure)
            self.space_group = sga.get_space_group_symbol()
            self.point_group = sga.get_point_group_symbol()
            self.crystal_system = sga.get_crystal_system()
            
            print(f"Space group: {self.space_group}")
            print(f"Point group: {self.point_group}")
            print(f"Crystal system: {self.crystal_system}")
            
            # Get symmetry operations
            symops = sga.get_symmetry_operations()
            self.symops = [str(op) for op in symops]
            
            # Generate conventional cell for better analysis
            conventional_structure = sga.get_conventional_standard_structure()
            
            # Store lattice parameters
            lattice = conventional_structure.lattice
            self.lattice = {
                'a': lattice.a,
                'b': lattice.b,
                'c': lattice.c,
                'alpha': lattice.alpha,
                'beta': lattice.beta,
                'gamma': lattice.gamma
            }
            self.cell_matrix = lattice.matrix
            
            print(f"Lattice: a={self.lattice['a']:.4f}, b={self.lattice['b']:.4f}, c={self.lattice['c']:.4f}, \n" +
                  f"        alpha={self.lattice['alpha']:.2f}, beta={self.lattice['beta']:.2f}, gamma={self.lattice['gamma']:.2f}")
            
            # Generate the asymmetric unit from the primitive cell
            self.asymmetric_atoms = []
            for i, site in enumerate(structure.sites):
                elem = site.specie.symbol
                coords = site.frac_coords
                self.asymmetric_atoms.append({
                    'label': f"{elem}_{i+1}",
                    'type': elem,
                    'fract_x': coords[0],
                    'fract_y': coords[1],
                    'fract_z': coords[2]
                })
                
            print(f"Parsed {len(self.asymmetric_atoms)} atoms in asymmetric unit")
            
            # Store all atoms in the conventional cell
            self.atoms = []
            print("\nAtoms:")
            for i, site in enumerate(conventional_structure.sites):
                elem = site.specie.symbol
                coords = site.frac_coords
                cart_coords = site.coords
                
                # Create atom entry
                atom = {
                    'label': f"{elem}_{i+1}",
                    'type': elem,
                    'fract_x': coords[0],
                    'fract_y': coords[1],
                    'fract_z': coords[2],
                    'cart_x': cart_coords[0],
                    'cart_y': cart_coords[1],
                    'cart_z': cart_coords[2]
                }
                self.atoms.append(atom)
                
                # Print first few atoms
                if i < 10:
                    print(f"  {atom['label']}: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})")
            
            if len(self.atoms) > 10:
                print(f"  ... and {len(self.atoms)-10} more atoms")
                
            print(f"Generated {len(self.atoms)} atoms in the conventional unit cell")
            
            # Validate
            if not self.atoms:
                print("WARNING: No atoms found in the structure!")
                return False
                
            # Success
            return True
                
        except Exception as e:
            print(f"Error in pymatgen parsing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def parse_cif(self):
        """Parse CIF file using pymatgen if available, falling back to built-in parser."""
        # Try to use pymatgen first (advanced crystallographic analysis)
        if self.parse_cif_with_pymatgen():
            print("Successfully parsed CIF with pymatgen")
        else:
            # Fall back to original parser
            print("Falling back to built-in CIF parser")
            self.parse_cif_original()
            
        # Update bond analysis
        print("Analyzing bonds...")
        self.find_bonds(cutoff=3.0)
        
        # Map Born charges
        self.map_born_charges()


def main():
    root = tk.Tk()

    # Set up the Clam theme
    style = ttk.Style()
    style.theme_use('clam')

    # Only set padding and other non-color options for ttk widgets
    style.configure('TButton', padding=5)
    style.configure('TEntry', padding=5)
    style.configure('TCombobox', padding=5)
    style.configure('TNotebook.Tab', padding=[10, 2])

    # Optionally, set hover/active effects if you want
    style.map('TButton',
        background=[('active', '!disabled', style.lookup('TButton', 'background'))]
    )

    # For classic Tk widgets, set to match Clam theme (white bg, black fg)
    # We'll patch this in the RamanPolarizationAnalyzer __init__
    orig_init = RamanPolarizationAnalyzer.__init__
    def patched_init(self, root):
        orig_init(self, root)
        # Set all Text and Listbox widgets to white bg, black fg
        for widget in self.root.winfo_children():
            self._set_classic_widget_colors(widget)
    def _set_classic_widget_colors(self, widget):
        # Recursively set colors for Text and Listbox
        if isinstance(widget, tk.Text) or isinstance(widget, tk.Listbox):
            widget.configure(bg='white', fg='black')
        for child in widget.winfo_children():
            self._set_classic_widget_colors(child)
    RamanPolarizationAnalyzer._set_classic_widget_colors = _set_classic_widget_colors
    RamanPolarizationAnalyzer.__init__ = patched_init

    app = RamanPolarizationAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 