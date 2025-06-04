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
from scipy.optimize import curve_fit, minimize

# Import strain tensor refinement module
try:
    from strain_raman_refinement import StrainRamanAnalyzer, RamanMode
    STRAIN_REFINEMENT_AVAILABLE = True
    print("✓ Strain tensor refinement module available")
except ImportError:
    STRAIN_REFINEMENT_AVAILABLE = False
    print("⚠ Strain tensor refinement module not available")
    print("  Make sure strain_raman_refinement.py is in the same directory")

class StressStrainAnalyzer:
    def __init__(self, root):
        """Initialize the Stress/Strain Analyzer
        
        Parameters:
        -----------
        root : tkinter.Tk
            The root window
        """
        self.root = root
        root.title("Raman Stress/Strain Analysis")
        root.geometry("1200x800")
        
        # Initialize variables
        self.current_spectrum = None
        self.original_spectrum = None
        self.reference_spectrum = None
        
        # Stress/strain analysis variables
        self.stress_strain_data = {}
        self.stress_coefficients = {}
        self.strain_analysis_results = {}
        self.strain_analyzer = None  # Will be initialized when crystal system is known
        self.gruneisen_data = {}  # Store Grüneisen parameters for modes
        self.strain_fitting_results = {}  # Store strain tensor fitting results
        
        # Peak fitting variables
        self.selected_peaks = []  # List of user-selected peak positions
        self.fitted_peaks = []    # List of fitted peak parameters
        self.peak_assignments = {}  # Dictionary mapping peak positions to vibrational modes
        self.frequency_shifts = {}  # Dictionary of calculated frequency shifts
        
        # Store references to plot components
        self.plot_components = {}
        
        # Create the main layout
        self.create_gui()
    
    def create_gui(self):
        """Create the main GUI layout."""
        # Create main frames
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        self.side_panel = ttk.Frame(main_frame, width=350)
        self.side_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.side_panel.pack_propagate(False)
        
        # Right panel for content/plots
        self.content_area = ttk.Frame(main_frame)
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup the stress/strain interface
        self.setup_stress_strain_interface()
        
        # Setup matplotlib plot
        self.setup_matplotlib_plot()
    
    def setup_stress_strain_interface(self):
        """Setup the Stress/Strain Analysis interface."""
        # Title
        ttk.Label(self.side_panel, text="Stress/Strain Analysis", font=('Arial', 14, 'bold')).pack(pady=(10, 20))
        
        # Check if strain refinement module is available
        if not STRAIN_REFINEMENT_AVAILABLE:
            warning_frame = ttk.LabelFrame(self.side_panel, text="Module Not Available", padding=10)
            warning_frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(warning_frame, text="Strain tensor refinement module not found.\n"
                                        "Advanced strain analysis features are disabled.", 
                     foreground="red", wraplength=300).pack()
        
        # Crystal system setup for strain analysis
        crystal_frame = ttk.LabelFrame(self.side_panel, text="Crystal System Setup", padding=10)
        crystal_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(crystal_frame, text="Crystal System:").pack(anchor=tk.W)
        self.strain_crystal_system_var = tk.StringVar(value="trigonal")
        crystal_combo = ttk.Combobox(crystal_frame, textvariable=self.strain_crystal_system_var,
                                   values=["cubic", "tetragonal", "trigonal", "hexagonal", 
                                          "orthorhombic", "monoclinic", "triclinic"], state="readonly")
        crystal_combo.pack(fill=tk.X, pady=2)
        crystal_combo.bind('<<ComboboxSelected>>', self.on_crystal_system_change)
        
        ttk.Button(crystal_frame, text="Initialize Strain Analyzer", command=self.initialize_strain_analyzer).pack(fill=tk.X, pady=2)
        
        # Status label for strain analyzer
        self.strain_analyzer_status = ttk.Label(crystal_frame, text="Not initialized", foreground="red")
        self.strain_analyzer_status.pack(pady=2)
        
        # Grüneisen parameters section
        gruneisen_frame = ttk.LabelFrame(self.side_panel, text="Grüneisen Parameters", padding=10)
        gruneisen_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(gruneisen_frame, text="Load from DFT Data", command=self.load_gruneisen_from_dft).pack(fill=tk.X, pady=2)
        ttk.Button(gruneisen_frame, text="Import Literature Values", command=self.import_gruneisen_literature).pack(fill=tk.X, pady=2)
        ttk.Button(gruneisen_frame, text="Calculate from Peaks", command=self.calculate_gruneisen_from_peaks).pack(fill=tk.X, pady=2)
        ttk.Button(gruneisen_frame, text="Use Default Values", command=self.use_default_gruneisen).pack(fill=tk.X, pady=2)
        
        # Data input section
        data_frame = ttk.LabelFrame(self.side_panel, text="Spectrum Data", padding=10)
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(data_frame, text="Load Spectrum", command=self.load_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Load Reference Spectrum", command=self.load_reference_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Generate Test Data", command=self.generate_strain_test_data).pack(fill=tk.X, pady=2)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(self.side_panel, text="Strain Tensor Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(analysis_frame, text="Peak Position Analysis", command=self.analyze_peak_positions_strain).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Full Spectrum Fitting", command=self.fit_strain_tensor_full).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Strain Gradient Analysis", command=self.analyze_strain_gradients).pack(fill=tk.X, pady=2)
        
        # Fitting options
        options_frame = ttk.LabelFrame(self.side_panel, text="Fitting Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.fit_gradients_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include strain gradients", 
                       variable=self.fit_gradients_var).pack(anchor=tk.W)
        
        self.use_weights_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Use intensity weights", 
                       variable=self.use_weights_var).pack(anchor=tk.W)
        
        # Legacy stress analysis section
        legacy_frame = ttk.LabelFrame(self.side_panel, text="Legacy Stress Analysis", padding=10)
        legacy_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(legacy_frame, text="Frequency Shift Analysis", command=self.analyze_frequency_shifts).pack(fill=tk.X, pady=2)
        ttk.Button(legacy_frame, text="Stress Tensor Refinement", command=self.refine_stress_tensor).pack(fill=tk.X, pady=2)
        
        # Results section
        results_frame = ttk.LabelFrame(self.side_panel, text="Results & Export", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(results_frame, text="Show Strain Analysis", command=self.show_strain_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(results_frame, text="Export Strain Results", command=self.export_strain_results).pack(fill=tk.X, pady=2)
        ttk.Button(results_frame, text="Generate Report", command=self.generate_strain_report).pack(fill=tk.X, pady=2)
    
    def setup_matplotlib_plot(self):
        """Setup the matplotlib plot area."""
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax.set_ylabel('Intensity (a.u.)')
        self.ax.set_title('Stress/Strain Analysis')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.content_area)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.content_area)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
    
    def load_spectrum(self):
        """Load a spectrum file."""
        file_path = filedialog.askopenfilename(
            title="Load Spectrum",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Try to load the spectrum data
            data = np.loadtxt(file_path, delimiter=None)
            
            if data.shape[1] >= 2:
                self.current_spectrum = {
                    'wavenumber': data[:, 0],
                    'intensity': data[:, 1]
                }
                self.original_spectrum = self.current_spectrum.copy()
                
                # Plot the spectrum
                self.ax.clear()
                self.ax.plot(self.current_spectrum['wavenumber'], self.current_spectrum['intensity'])
                self.ax.set_xlabel('Wavenumber (cm⁻¹)')
                self.ax.set_ylabel('Intensity (a.u.)')
                self.ax.set_title('Loaded Spectrum')
                self.canvas.draw()
                
                messagebox.showinfo("Success", f"Spectrum loaded successfully from {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Error", "File must contain at least two columns (wavenumber, intensity)")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load spectrum: {str(e)}")
    
    def on_crystal_system_change(self, event=None):
        """Handle crystal system change."""
        # Reset strain analyzer when crystal system changes
        self.strain_analyzer = None
        self.strain_analyzer_status.config(text="Not initialized", foreground="red")
    
    def initialize_strain_analyzer(self):
        """Initialize the strain analyzer with the selected crystal system."""
        if not STRAIN_REFINEMENT_AVAILABLE:
            messagebox.showwarning("Warning", "Strain refinement module not available.")
            return
        
        try:
            crystal_system = self.strain_crystal_system_var.get()
            
            # Initialize the strain analyzer
            self.strain_analyzer = StrainRamanAnalyzer(crystal_system=crystal_system)
            
            # Update status
            self.strain_analyzer_status.config(text=f"Initialized ({crystal_system})", foreground="green")
            
            messagebox.showinfo("Success", f"Strain analyzer initialized for {crystal_system} crystal system.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize strain analyzer: {str(e)}")
            self.strain_analyzer_status.config(text="Initialization failed", foreground="red")
    
    # Placeholder methods for all the strain analysis functionality
    def load_gruneisen_from_dft(self):
        """Load Grüneisen parameters from DFT data."""
        messagebox.showinfo("Info", "DFT data loading not yet implemented.")
    
    def import_gruneisen_literature(self):
        """Import Grüneisen parameters from literature."""
        messagebox.showinfo("Info", "Literature import not yet implemented.")
    
    def calculate_gruneisen_from_peaks(self):
        """Calculate Grüneisen parameters from peak data."""
        messagebox.showinfo("Info", "Peak-based calculation not yet implemented.")
    
    def use_default_gruneisen(self):
        """Use default Grüneisen parameters."""
        messagebox.showinfo("Info", "Default parameters loaded.")
    
    def load_reference_spectrum(self):
        """Load a reference spectrum."""
        messagebox.showinfo("Info", "Reference spectrum loading not yet implemented.")
    
    def generate_strain_test_data(self):
        """Generate test data for strain analysis."""
        messagebox.showinfo("Info", "Test data generation not yet implemented.")
    
    def analyze_peak_positions_strain(self):
        """Analyze peak positions for strain."""
        messagebox.showinfo("Info", "Peak position analysis not yet implemented.")
    
    def fit_strain_tensor_full(self):
        """Perform full spectrum strain tensor fitting."""
        messagebox.showinfo("Info", "Full spectrum fitting not yet implemented.")
    
    def analyze_strain_gradients(self):
        """Analyze strain gradients."""
        messagebox.showinfo("Info", "Strain gradient analysis not yet implemented.")
    
    def analyze_frequency_shifts(self):
        """Analyze frequency shifts for stress."""
        messagebox.showinfo("Info", "Frequency shift analysis not yet implemented.")
    
    def refine_stress_tensor(self):
        """Refine stress tensor."""
        messagebox.showinfo("Info", "Stress tensor refinement not yet implemented.")
    
    def show_strain_analysis(self):
        """Show strain analysis results."""
        messagebox.showinfo("Info", "Strain analysis display not yet implemented.")
    
    def export_strain_results(self):
        """Export strain analysis results."""
        messagebox.showinfo("Info", "Results export not yet implemented.")
    
    def generate_strain_report(self):
        """Generate strain analysis report."""
        messagebox.showinfo("Info", "Report generation not yet implemented.")

def main():
    """Main function to run the Stress/Strain Analyzer."""
    root = tk.Tk()
    app = StressStrainAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 