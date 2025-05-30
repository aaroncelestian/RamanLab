#!/usr/bin/env python3
# Batch Peak Fitting Module for ClaritySpectra
"""
Module for batch processing of Raman spectra with peak fitting.
Allows sequential refinement of peak positions, shapes, and backgrounds
across multiple spectra, with visualization of trends.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Menu
import pandas as pd
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Import the density analysis module
import sys
sys.path.append('Density')
from raman_density_analysis import RamanDensityAnalyzer

class BatchPeakFittingWindow:
    """Window for batch processing of Raman spectra with peak fitting."""
    
    def __init__(self, parent, raman):
        """
        Initialize the batch peak fitting window.
        
        Parameters:
        -----------
        parent : tk.Tk or tk.Toplevel
            Parent window
        raman : RamanSpectra
            Reference to the RamanSpectra instance
        """
        self.window = tk.Toplevel(parent)
        self.window.title("Batch Peak Fitting")
        self.window.geometry("1250x800")
        
        # Store references
        self.parent = parent
        self.raman = raman
        
        # Initialize file mapping structure
        self.spectra_files = []  # Maintains one-to-one correspondence with listbox indices
        
        # Initialize data storage
        self.current_spectrum_index = 0
        self.batch_results = []
        self.reference_peaks = None
        self.reference_background = None
        
        # Variables for fitted peaks
        self.peaks = []
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.current_model = tk.StringVar(value="Gaussian")
        self.residuals = None
        self.show_fitted_peaks = tk.BooleanVar(value=True)
        self.show_individual_peaks = tk.BooleanVar(value=True)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create GUI
        self.create_gui()
        
        # Set up window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.peak_visibility_vars = []
        self.batch_results = []
        self.spectra_files = []
        self.reference_peaks = None
        self.current_spectrum_index = -1
        self._stop_batch = False
        
        # Track manually skipped files
        self.manually_skipped_files = set()  # Set of file paths that are manually skipped
        
        # Initialize density analyzer
        self.density_analyzer = RamanDensityAnalyzer()
        self.density_results = []
        self.whewellite_calibration_value = None
    
    def create_menu_bar(self):
        """Create the menu bar with File and Help menus."""
        self.menu_bar = Menu(self.window)
        self.window.config(menu=self.menu_bar)
        
        # File menu
        file_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Add Files", command=self.add_files)
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Help menu
        self.help_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Help for Current Tab (F1)", command=self.show_current_help)
        self.help_menu.add_command(label="General Help", command=self.show_general_help)
        
        # Bind F1 key to show help
        self.window.bind("<F1>", lambda event: self.show_current_help())
        
        # Dictionary of help texts for each tab
        self.help_texts = {
            "General": (
                "Batch Peak Fitting Help",
                "This tool allows you to process multiple Raman spectra with consistent peak fitting parameters.\n\n"
                "General Workflow:\n"
                "1. Add spectrum files using the File Selection tab\n"
                "2. Select a spectrum to work with\n"
                "3. Subtract background in the Peaks tab\n"
                "4. Detect peaks automatically or add them manually\n"
                "5. Choose a peak model and fit the peaks\n"
                "6. Set this spectrum as a reference in the Batch tab\n"
                "7. Process all spectra with 'Apply to All'\n"
                "8. Analyze trends in the Fit Results tab\n"
                "9. Export results to CSV for further analysis\n\n"
                "Use F1 or click 'Help for Current Tab' at any time for context-specific help."
            ),
            "File Selection": (
                "File Selection Help",
                "This tab allows you to manage the spectra files to be processed:\n\n"
                "• Add Files: Import spectrum files (.txt, .csv) for batch processing\n"
                "• Remove Selected: Remove highlighted files from the list\n"
                "• Double-click a file to display it in the current spectrum view\n\n"
                "Navigation buttons at the bottom of the plot allow you to move between files."
            ),
            "Peaks": (
                "Peak Controls Help",
                "This tab contains tools for manipulating and fitting peaks:\n\n"
                "• Background: Set λ (smoothness) and p (asymmetry) parameters\n"
                "  for baseline correction, then click Subtract Background\n\n"
                "• Manual Peak Control: Add peaks by clicking on the spectrum\n"
                "  or delete selected peaks\n\n"
                "• Peak Detection (Interactive): Use sliders to adjust peak detection parameters:\n"
                "  - Height: Minimum peak height (0 = Auto detection)\n"
                "  - Distance: Minimum distance between peaks (0 = Auto detection)\n"
                "  - Prominence: Minimum peak prominence (0 = Auto detection)\n"
                "  Enable 'Auto-detect peaks as sliders change' for real-time peak detection\n"
                "  Use 'Reset to Auto' to return all parameters to automatic detection\n\n"
                "• Peak Model: Select the mathematical model to use for peak fitting\n"
                "  (Gaussian, Lorentzian, Pseudo-Voigt, or Asymmetric Voigt)\n\n"
                "• Fit Ranges: Specify wavenumber ranges for ROI (region of interest)\n"
                "  Use comma-separated ranges, e.g., 400-600,900-1100"
            ),
            "Batch": (
                "Batch Processing Help",
                "This tab allows you to process multiple spectra using consistent parameters:\n\n"
                "• Set as Reference: Save current spectrum's peaks and background as a template\n"
                "• Apply to All: Process all loaded spectra using the reference template\n"
                "• Stop: Interrupt batch processing\n"
                "• Export Results: Save batch results to a CSV file\n\n"
                "Progress Log shows the status of each spectrum processed."
            ),
            "Results": (
                "Results Tab Help",
                "This tab controls what is displayed in the Fit Results and Fit Stats tabs:\n\n"
                "• Peak Visibility: Toggle which peaks are shown in trend plots\n"
                "• Show All/Hide All: Quickly select or deselect all peaks\n"
                "• Show 95% Boundary: Display confidence intervals on trend plots\n"
                "• Show Grid Lines: Toggle grid visibility on trend plots\n\n"
                "Preview buttons allow you to view individual parameter plots in separate windows.\n\n"
                "Export controls let you save fit statistics plots as images or export all peak data as CSV.\n"
                "• Export Curve Data: Save all peak parameters (position, amplitude, width, eta) to CSV file\n"
                "• Export Fit Stats: Save plots of the best, median, and worst fits"
            ),
            "Current Spectrum": (
                "Current Spectrum Help",
                "This view shows the currently selected spectrum:\n\n"
                "• The top panel displays the raw spectrum, background, fitted peaks, and residuals\n"
                "• The bottom panel shows residuals (difference between data and fit)\n"
                "• Regions of interest (ROI) are highlighted in gray if specified\n"
                "• Individual peaks show R² values indicating goodness of fit\n\n"
                "Use the navigation buttons at the bottom to move between spectra."
            ),
            "Waterfall": (
                "Waterfall Plot Help",
                "This tab displays a stacked view of multiple spectra:\n\n"
                "• Skip: Set how many spectra to skip between each displayed line\n"
                "• Color: Choose a colormap for the stacked spectra\n"
                "• Colormap Min/Max: Adjust the colormap intensity range\n"
                "• Line Width: Set the thickness of spectrum lines\n"
                "• Y Offset: Adjust vertical spacing between spectra\n"
                "• Legend/Grid: Toggle display of legend and grid lines\n\n"
                "X-axis Range controls let you focus on specific wavenumber regions."
            ),
            "Heatmap": (
                "Heatmap Plot Help",
                "This tab shows intensity as a color-coded 2D map:\n\n"
                "• Colormap: Select color scheme for intensity visualization\n"
                "• Contrast: Adjust intensity range of the colormap\n"
                "• Brightness: Control overall brightness of the display\n"
                "• Gamma: Adjust the non-linear intensity mapping\n\n"
                "Use Reset Colors to return to default settings.\n"
                "X-axis Range controls let you focus on specific wavenumber regions."
            ),
            "Fit Results": (
                "Fit Results Help",
                "This tab displays trends in peak parameters across all spectra:\n\n"
                "• Position: How peak centers shift across spectra\n"
                "• Amplitude: How peak heights change\n"
                "• Width: Changes in peak widths\n"
                "• Eta: Mixing parameter for Voigt profiles (when applicable)\n\n"
                "Failed fits are excluded from plots. Use the Plot tab to control which peaks are displayed."
            ),
            "Fit Stats": (
                "Fit Statistics Help",
                "This tab shows fit quality metrics across all spectra:\n\n"
                "• The top row shows the best fits (highest R²)\n"
                "• The middle row shows median quality fits\n"
                "• The bottom row shows the worst fits (lowest R²)\n\n"
                "Each plot displays the original spectrum, background, and fit result.\n"
                "R² values indicate the goodness of fit within the ROI regions."
            ),
            "Density Analysis": (
                "Density Analysis Help",
                "This tab provides quantitative density analysis for correlation with micro-CT:\n\n"
                "• Calibration: Use a whewellite reference spectrum to calibrate the analysis\n"
                "• Single Spectrum: Analyze the currently selected spectrum for crystalline density\n"
                "• Batch Analysis: Process all loaded spectra to generate density profiles\n"
                "• Line Scan: Analyze spatial line scan data for density variation\n\n"
                "The Crystalline Density Index (CDI) quantifies the ratio of crystalline to organic content.\n"
                "Apparent density values can be directly correlated with micro-CT measurements.\n\n"
                "Key Features:\n"
                "• Automatic baseline correction using asymmetric least squares\n"
                "• COM peak analysis at 1462 cm⁻¹\n"
                "• Organic matrix and void space quantification\n"
                "• Export density profiles for micro-CT correlation"
            )
        }
        
    def show_current_help(self):
        """Show help for the current tab or view."""
        # Determine which tab is currently active
        try:
            # Check left notebook tabs
            current_left_tab = self.left_notebook.tab(self.left_notebook.select(), "text")
            help_key = current_left_tab
            
            # If visualization notebook exists and is visible, check its tabs too
            if hasattr(self, 'viz_notebook'):
                current_viz_tab = self.viz_notebook.tab(self.viz_notebook.select(), "text")
                # If user is looking at a visualization tab, use that context instead
                if self.viz_notebook.winfo_viewable():
                    help_key = current_viz_tab
        except:
            # Default to general help if we can't determine the current tab
            help_key = "General"
        
        # Get help text for the current context
        if help_key in self.help_texts:
            title, text = self.help_texts[help_key]
        else:
            title = "Help"
            text = "No specific help is available for this view."
        
        # Display help dialog
        self.show_help_dialog(title, text)
    
    def show_general_help(self):
        """Show general help information."""
        if "General" in self.help_texts:
            title, text = self.help_texts["General"]
        else:
            title = "General Help"
            text = "General help information is not available."
        
        # Display help dialog
        self.show_help_dialog(title, text)
    
    def show_help_dialog(self, title, text):
        """Display a help dialog with the given title and text."""
        help_dialog = tk.Toplevel(self.window)
        help_dialog.title(title)
        help_dialog.geometry("600x400")
        help_dialog.transient(self.window)  # Make dialog modal
        help_dialog.grab_set()  # Make dialog modal
        
        # Create a frame with padding
        frame = ttk.Frame(help_dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a scrolled text widget
        text_widget = tk.Text(frame, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Insert help text
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Add close button
        ttk.Button(help_dialog, text="Close", command=help_dialog.destroy).pack(pady=10)
    
    def create_gui(self):
        """Create the GUI elements."""
        # Main container
        main_container = ttk.Frame(self.window, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Use PanedWindow to eliminate unused space and allow user control
        self.main_paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # --- LEFT PANE: Controls Panel ---
        self.left_panel_container = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_panel_container, weight=1)  # Left panel gets 1 weight unit
        
        # --- RIGHT PANE: Plot Area ---
        self.right_panel_container = ttk.Frame(self.main_paned) 
        self.main_paned.add(self.right_panel_container, weight=3)  # Right panel gets 3 weight units (75% of space)
        
        # Create notebook for tabs
        self.left_notebook = ttk.Notebook(self.left_panel_container)
        self.left_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Bind tab change event to update help menu
        self.left_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # --- Tab 1: File Selection ---
        file_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(file_tab, text="File Selection")

        file_frame = ttk.LabelFrame(file_tab, text="File Selection: double click to display", padding=10)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, exportselection=False)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        
        # Add multiple event bindings for selection
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        self.file_listbox.bind('<Double-1>', self.on_file_select)
        button_frame = ttk.Frame(file_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected_files).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # --- Tab 2: Peak Controls ---
        peak_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(peak_tab, text="Peaks")

        # Create a canvas and scrollbar for scrollable content
        canvas = tk.Canvas(peak_tab)
        scrollbar_peaks = ttk.Scrollbar(peak_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure scrollable frame to expand with canvas width
        def configure_scrollable_frame(event):
            # Update scroll region when content changes
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Make scrollable frame width match canvas width
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", configure_scrollable_frame)

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_peaks.set)

        # Pack scrollbar first, then canvas to eliminate gap
        scrollbar_peaks.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel to canvas for smooth scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)

        controls_frame = ttk.LabelFrame(scrollable_frame, text="Peak Fitting Controls", padding=5)  # Reduced padding from 10 to 5
        controls_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Background controls
        bg_frame = ttk.LabelFrame(controls_frame, text="Background", padding=3)  # Reduced padding from 5 to 3
        bg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="λ (smoothness):").pack(anchor=tk.W)
        self.var_lambda = tk.StringVar(value="1e5")
        ttk.Entry(bg_frame, textvariable=self.var_lambda).pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="p (asymmetry):").pack(anchor=tk.W)
        self.var_p = tk.StringVar(value="0.01")
        ttk.Entry(bg_frame, textvariable=self.var_p).pack(fill=tk.X, pady=2)
        ttk.Button(bg_frame, text="Subtract Background", command=self.subtract_background).pack(fill=tk.X, pady=2)

        # Manual peak controls
        manual_frame = ttk.LabelFrame(controls_frame, text="Manual Peak Control", padding=3)  # Reduced padding from 5 to 3
        manual_frame.pack(fill=tk.X, pady=2)
        
        # Add/Delete buttons
        button_frame = ttk.Frame(manual_frame)
        button_frame.pack(fill=tk.X, pady=2)
        
        # Store reference to the manual peak button for styling
        self.manual_peak_button = ttk.Button(button_frame, text="Click to Add Peak", command=self.enable_peak_addition)
        self.manual_peak_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(button_frame, text="Delete Peak", command=self.show_peak_deletion_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Peak detection controls with interactive sliders
        peak_frame = ttk.LabelFrame(controls_frame, text="Peak Detection (Interactive)", padding=3)  # Reduced padding from 5 to 3
        peak_frame.pack(fill=tk.X, pady=2)
        
        # Height slider
        height_container = ttk.Frame(peak_frame)
        height_container.pack(fill=tk.X, pady=2)
        ttk.Label(height_container, text="Height:").pack(side=tk.LEFT)
        self.height_value_label = ttk.Label(height_container, text="Auto")
        self.height_value_label.pack(side=tk.RIGHT)
        
        self.var_height = tk.DoubleVar(value=0)  # 0 represents "Auto"
        self.height_slider = ttk.Scale(peak_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                     variable=self.var_height, command=lambda v: self.on_height_change(v))
        self.height_slider.pack(fill=tk.X, pady=(0, 5))
        
        # Distance slider
        distance_container = ttk.Frame(peak_frame)
        distance_container.pack(fill=tk.X, pady=2)
        ttk.Label(distance_container, text="Distance:").pack(side=tk.LEFT)
        self.distance_value_label = ttk.Label(distance_container, text="Auto")
        self.distance_value_label.pack(side=tk.RIGHT)
        
        self.var_distance = tk.DoubleVar(value=0)  # 0 represents "Auto"
        self.distance_slider = ttk.Scale(peak_frame, from_=0, to=50, orient=tk.HORIZONTAL,
                                       variable=self.var_distance, command=lambda v: self.on_distance_change(v))
        self.distance_slider.pack(fill=tk.X, pady=(0, 5))
        
        # Prominence slider
        prominence_container = ttk.Frame(peak_frame)
        prominence_container.pack(fill=tk.X, pady=2)
        ttk.Label(prominence_container, text="Prominence:").pack(side=tk.LEFT)
        self.prominence_value_label = ttk.Label(prominence_container, text="Auto")
        self.prominence_value_label.pack(side=tk.RIGHT)
        
        self.var_prominence = tk.DoubleVar(value=0)  # 0 represents "Auto"
        self.prominence_slider = ttk.Scale(peak_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                         variable=self.var_prominence, command=lambda v: self.on_prominence_change(v))
        self.prominence_slider.pack(fill=tk.X, pady=(0, 5))
        
        # Auto-detect checkbox
        self.auto_detect_var = tk.BooleanVar(value=True)
        auto_detect_cb = ttk.Checkbutton(peak_frame, text="Auto-detect peaks as sliders change", 
                                       variable=self.auto_detect_var)
        auto_detect_cb.pack(anchor=tk.W, pady=2)
        
        # Create a frame for the buttons
        peak_button_frame = ttk.Frame(peak_frame)
        peak_button_frame.pack(fill=tk.X, pady=2)
        
        # First row of buttons
        button_row1 = ttk.Frame(peak_button_frame)
        button_row1.pack(fill=tk.X, pady=1)
        ttk.Button(button_row1, text="Find Peaks", command=self.find_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(button_row1, text="Clear Peaks", command=self.clear_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        # Second row of buttons
        button_row2 = ttk.Frame(peak_button_frame)
        button_row2.pack(fill=tk.X, pady=1)
        ttk.Button(button_row2, text="Reset to Auto", command=self.reset_peak_params).pack(fill=tk.X, padx=1)

        # Model selection
        model_frame = ttk.LabelFrame(controls_frame, text="Peak Model", padding=3)  # Reduced padding from 5 to 3
        model_frame.pack(fill=tk.X, pady=2)
        self.current_model = tk.StringVar(value="Gaussian")
        model_combo = ttk.Combobox(model_frame, textvariable=self.current_model, 
                                 values=["Gaussian", "Lorentzian", "Pseudo-Voigt", "Asymmetric Voigt"])
        model_combo.pack(fill=tk.X, pady=2)
        
        # Parameter constraint controls
        constraint_frame = ttk.LabelFrame(model_frame, text="Parameter Constraints", padding=3)
        constraint_frame.pack(fill=tk.X, pady=3)
        
        self.fix_positions = tk.BooleanVar(value=False)
        self.fix_widths = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(constraint_frame, text="Fix Peak Positions", 
                       variable=self.fix_positions).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(constraint_frame, text="Fix Peak Widths", 
                       variable=self.fix_widths).pack(anchor=tk.W, pady=1)
        
        # Add tooltips/info
        info_label = ttk.Label(constraint_frame, 
                              text="Fixing parameters improves stability but reduces flexibility",
                              font=("", 8, "italic"), foreground="gray")
        info_label.pack(anchor=tk.W, pady=(2,0))
        
        ttk.Button(model_frame, text="Fit Peaks", command=self.fit_peaks).pack(fill=tk.X, pady=2)

        # In Peak Controls tab, add Fit Ranges entry
        fit_range_frame = ttk.LabelFrame(controls_frame, text="Fit Ranges (cm⁻¹)", padding=5)
        fit_range_frame.pack(fill=tk.X, pady=2)
        self.var_fit_ranges = tk.StringVar(value="")
        fit_range_entry = ttk.Entry(fit_range_frame, textvariable=self.var_fit_ranges)
        fit_range_entry.pack(fill=tk.X, pady=2)
        fit_range_entry.insert(0, "400-600,900-1100")  # Example default
        fit_range_entry_tooltip = ttk.Label(fit_range_frame, text="e.g. 400-600,900-1100", font=("", 8, "italic"), foreground="gray")
        fit_range_entry_tooltip.pack(anchor=tk.W)
        # Add Update ROI button
        ttk.Button(fit_range_frame, text="Update ROI", command=self.update_roi_regions).pack(fill=tk.X, pady=2)

        # File management controls
        file_mgmt_frame = ttk.LabelFrame(controls_frame, text="File Management", padding=5)
        file_mgmt_frame.pack(fill=tk.X, pady=2)
        
        # Skip/Unskip file button with dynamic text
        self.skip_button = ttk.Button(file_mgmt_frame, text="⚠ Skip This File", command=self.toggle_skip_current_file)
        self.skip_button.pack(fill=tk.X, pady=2)
        
        # Add tooltip/info
        skip_info_label = ttk.Label(file_mgmt_frame, 
                                   text="Excluded files won't appear in Fit Results or trend analysis",
                                   font=("", 8, "italic"), foreground="gray")
        skip_info_label.pack(anchor=tk.W, pady=(0,2))

        # --- Tab 3: Batch Processing ---
        batch_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(batch_tab, text="Batch")

        batch_frame = ttk.LabelFrame(batch_tab, text="Batch Processing", padding=10)
        batch_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Button(batch_frame, text="Set as Reference", command=self.set_reference).pack(fill=tk.X, pady=2)
        batch_buttons_row = ttk.Frame(batch_frame)
        batch_buttons_row.pack(fill=tk.X, pady=2)
        ttk.Button(batch_buttons_row, text="Apply to All", command=self.apply_to_all).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(batch_buttons_row, text="Stop", command=self.stop_batch).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(batch_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=2)



        # Add status text box with scrollbar
        status_frame = ttk.LabelFrame(batch_frame, text="Progress Log", padding=5)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create scrolled text widget
        self.batch_status_text = tk.Text(status_frame, height=10, width=40, wrap=tk.WORD, 
                                        font=("Courier", 9))
        self.batch_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        status_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, 
                                        command=self.batch_status_text.yview)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_status_text.config(yscrollcommand=status_scrollbar.set)
        
        # Set initial text
        self.batch_status_text.insert(tk.END, "Ready. Set a reference spectrum and click 'Apply to All' to begin batch processing.\n")
        self.batch_status_text.config(state=tk.DISABLED)  # Make read-only

        # --- Tab 4: Peak Visibility ---
        visibility_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(visibility_tab, text="Results")

        visibility_frame = ttk.LabelFrame(visibility_tab, text="Peak Visibility Controls", padding=10)
        visibility_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        # Add explanatory label
        ttk.Label(visibility_frame, text="(These controls affect only the Fit Results and Fit Stats tabs)", font=("", 9, "italic"), foreground="gray").pack(anchor=tk.W, pady=(0, 8))

        # Create a frame for peak visibility checkboxes
        self.peak_visibility_frame = ttk.Frame(visibility_frame)
        self.peak_visibility_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize peak visibility variables
        self.peak_visibility_vars = []
        
        # Add buttons to show/hide all peaks
        button_frame = ttk.Frame(visibility_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Show All", command=self.show_all_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="Hide All", command=self.hide_all_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Add new controls for 95% boundary, grid lines, and preview
        self.show_95_boundary = tk.BooleanVar(value=True)
        self.show_fit_grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(visibility_frame, text="Show 95% Boundary", variable=self.show_95_boundary, command=self.update_trends_plot).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(visibility_frame, text="Show Grid Lines", variable=self.show_fit_grid, command=self.update_trends_plot).pack(anchor=tk.W, pady=2)
        

        
        # Four separate preview buttons
        button_width = 22
        ttk.Button(visibility_frame, text="Preview Position Plot", command=lambda: self.preview_trends_subplot('position'), width=button_width).pack(anchor=tk.W, pady=2)
        ttk.Button(visibility_frame, text="Preview Amplitude Plot", command=lambda: self.preview_trends_subplot('amplitude'), width=button_width).pack(anchor=tk.W, pady=2)
        ttk.Button(visibility_frame, text="Preview Width Plot", command=lambda: self.preview_trends_subplot('width'), width=button_width).pack(anchor=tk.W, pady=2)
        ttk.Button(visibility_frame, text="Preview Eta Plot", command=lambda: self.preview_trends_subplot('eta'), width=button_width).pack(anchor=tk.W, pady=2)

        # Add export buttons for Fit Stats
        export_frame = ttk.LabelFrame(visibility_frame, text="Export Fit Stats", padding=5)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Save Individual Stats Plots", 
                  command=self.export_individual_fit_stats).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Save All Stats Plots", 
                  command=self.export_fit_stats_plot).pack(fill=tk.X, pady=2)
        
        # Add export CSV button
        export_csv_frame = ttk.LabelFrame(visibility_frame, text="Export Curve Data", padding=5)
        export_csv_frame.pack(fill=tk.X, pady=5)
        ttk.Button(export_csv_frame, text="Export Peak Curve Data (CSV)", 
                  command=self.export_curve_data_csv).pack(fill=tk.X, pady=2)




        # --- RIGHT PANEL: Visualization ---
        # Use the right_panel_container already created in PanedWindow
        self.viz_notebook = ttk.Notebook(self.right_panel_container)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)

        # Bind tab change event for visualization notebook
        self.viz_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Current spectrum tab
        current_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(current_frame, text="Current Spectrum")
        self.fig_current, (self.ax1_current, self.ax2_current) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
        self.canvas_current = FigureCanvasTkAgg(self.fig_current, master=current_frame)
        self.canvas_current.draw()
        self.canvas_current.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Add toolbar for current spectrum
        toolbar_frame = ttk.Frame(current_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_current = NavigationToolbar2Tk(self.canvas_current, toolbar_frame)
        self.toolbar_current.update()

        # --- Navigation controls at the bottom of the plot ---
        nav_frame = ttk.Frame(current_frame)
        nav_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        ttk.Button(nav_frame, text="<< First", command=lambda: self.navigate_spectrum(0)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(nav_frame, text="< Previous", command=lambda: self.navigate_spectrum(-1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(nav_frame, text="Next >", command=lambda: self.navigate_spectrum(1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(nav_frame, text="Last >>", command=lambda: self.navigate_spectrum(len(self.spectra_files)-1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        # Current spectrum info label
        self.current_spectrum_label = ttk.Label(current_frame)
        self.current_spectrum_label.pack(fill=tk.X, pady=2, side=tk.BOTTOM)

        # Waterfall plot tab
        waterfall_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(waterfall_frame, text="Waterfall")
        
        # Controls for waterfall plot
        waterfall_controls = ttk.Frame(waterfall_frame)
        waterfall_controls.pack(fill=tk.X, pady=5)
        
        # Create a frame for basic controls
        basic_controls = ttk.Frame(waterfall_controls)
        basic_controls.pack(fill=tk.X, pady=2)
        
        # First row: Skip, Color, Colormap Min, Max
        ttk.Label(basic_controls, text="Skip:").pack(side=tk.LEFT, padx=5)
        self.waterfall_skip = ttk.Spinbox(basic_controls, from_=1, to=100, width=5)
        self.waterfall_skip.set(1)
        self.waterfall_skip.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(basic_controls, text="Color:").pack(side=tk.LEFT, padx=5)
        self.waterfall_cmap = ttk.Combobox(basic_controls, values=[
            # Perceptually uniform sequential
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            # Sequential
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            # Diverging
            'coolwarm', 'RdBu_r', 'seismic', 'bwr', 'BrBG', 'PuOr',
            'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
            # Cyclic
            'twilight', 'twilight_shifted', 'hsv',
            # Qualitative
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2',
            # Custom options
            'all_black', 'black_to_darkgrey', 'darkgrey_to_black'
        ], state="readonly")
        self.waterfall_cmap.set('all_black')
        self.waterfall_cmap.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(basic_controls, text="Colormap Min:").pack(side=tk.LEFT, padx=2)
        self.waterfall_cmap_min = tk.DoubleVar(value=0.0)
        self.waterfall_cmap_min_entry = ttk.Entry(basic_controls, textvariable=self.waterfall_cmap_min, width=4)
        self.waterfall_cmap_min_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(basic_controls, text="Max:").pack(side=tk.LEFT, padx=2)
        self.waterfall_cmap_max = tk.DoubleVar(value=0.85)
        self.waterfall_cmap_max_entry = ttk.Entry(basic_controls, textvariable=self.waterfall_cmap_max, width=4)
        self.waterfall_cmap_max_entry.pack(side=tk.LEFT, padx=2)

        # Second row: Line Width, Y Offset, Legend, Grid
        secondary_controls = ttk.Frame(waterfall_controls)
        secondary_controls.pack(fill=tk.X, pady=2)
        ttk.Label(secondary_controls, text="Line Width:").pack(side=tk.LEFT, padx=5)
        self.waterfall_linewidth = tk.DoubleVar(value=1.0)
        self.waterfall_linewidth_spin = ttk.Spinbox(secondary_controls, from_=0.5, to=5.0, increment=0.1, 
                                                  textvariable=self.waterfall_linewidth, width=4)
        self.waterfall_linewidth_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(secondary_controls, text="Y Offset (a.u.):").pack(side=tk.LEFT, padx=5)
        self.waterfall_yoffset = tk.DoubleVar(value=100)  # Default value (absolute units)
        self.waterfall_yoffset_entry = ttk.Entry(secondary_controls, textvariable=self.waterfall_yoffset, width=6)
        self.waterfall_yoffset_entry.pack(side=tk.LEFT, padx=5)
        self.waterfall_show_legend = tk.BooleanVar(value=True)
        self.waterfall_show_grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(secondary_controls, text="Legend", variable=self.waterfall_show_legend, 
                       command=self.update_waterfall_plot).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(secondary_controls, text="Grid", variable=self.waterfall_show_grid, 
                       command=self.update_waterfall_plot).pack(side=tk.LEFT, padx=5)
        
        # Create a frame for X-axis range controls
        range_controls = ttk.LabelFrame(waterfall_controls, text="X-axis Range (cm⁻¹)", padding=5)
        range_controls.pack(fill=tk.X, pady=2)
        
        range_frame = ttk.Frame(range_controls)
        range_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(range_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        self.waterfall_xmin = ttk.Entry(range_frame, width=8)
        self.waterfall_xmin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(range_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        self.waterfall_xmax = ttk.Entry(range_frame, width=8)
        self.waterfall_xmax.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(range_frame, text="Reset Range", 
                  command=lambda: self.reset_waterfall_range()).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(range_frame, text="Update Plot", command=self.update_waterfall_plot).pack(side=tk.LEFT, padx=5)
        
        # Waterfall plot
        self.fig_waterfall, self.ax_waterfall = plt.subplots(figsize=(8, 6))
        self.canvas_waterfall = FigureCanvasTkAgg(self.fig_waterfall, master=waterfall_frame)
        self.canvas_waterfall.draw()
        self.canvas_waterfall.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(waterfall_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_waterfall = NavigationToolbar2Tk(self.canvas_waterfall, toolbar_frame)
        self.toolbar_waterfall.update()

        # Heatmap plot tab
        heatmap_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(heatmap_frame, text="Heatmap")
        
        # Controls for heatmap plot
        heatmap_controls = ttk.Frame(heatmap_frame)
        heatmap_controls.pack(fill=tk.X, pady=5)
        
        # Create a frame for basic controls and color adjustments
        basic_controls = ttk.Frame(heatmap_controls)
        basic_controls.pack(fill=tk.X, pady=2)
        
        # Colormap and color adjustments on one line
        ttk.Label(basic_controls, text="Colormap:").pack(side=tk.LEFT, padx=5)
        self.heatmap_cmap = ttk.Combobox(basic_controls, values=[
            # Perceptually uniform sequential
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            # Sequential
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            # Diverging
            'coolwarm', 'RdBu_r', 'seismic', 'bwr', 'BrBG', 'PuOr',
            'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
            # Cyclic
            'twilight', 'twilight_shifted', 'hsv',
            # Qualitative
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2'
        ])
        self.heatmap_cmap.set('viridis')
        self.heatmap_cmap.pack(side=tk.LEFT, padx=5)
        
        # Contrast control
        ttk.Label(basic_controls, text="Contrast:").pack(side=tk.LEFT, padx=5)
        self.heatmap_contrast = ttk.Scale(basic_controls, from_=0.1, to=2.0, orient=tk.HORIZONTAL, length=80)
        self.heatmap_contrast.set(1.0)
        self.heatmap_contrast.pack(side=tk.LEFT, padx=5)
        self.heatmap_contrast.bind("<Motion>", lambda e: self.update_heatmap_plot())
        
        # Brightness control
        ttk.Label(basic_controls, text="Bright:").pack(side=tk.LEFT, padx=5)
        self.heatmap_brightness = ttk.Scale(basic_controls, from_=0.0, to=2.0, orient=tk.HORIZONTAL, length=80)
        self.heatmap_brightness.set(1.0)
        self.heatmap_brightness.pack(side=tk.LEFT, padx=5)
        self.heatmap_brightness.bind("<Motion>", lambda e: self.update_heatmap_plot())
        
        # Gamma control
        ttk.Label(basic_controls, text="Gamma:").pack(side=tk.LEFT, padx=5)
        self.heatmap_gamma = ttk.Scale(basic_controls, from_=0.1, to=2.0, orient=tk.HORIZONTAL, length=80)
        self.heatmap_gamma.set(1.0)
        self.heatmap_gamma.pack(side=tk.LEFT, padx=5)
        self.heatmap_gamma.bind("<Motion>", lambda e: self.update_heatmap_plot())
        
        # Add reset button for color adjustments
        ttk.Button(basic_controls, text="Reset Colors", 
                  command=lambda: self.reset_heatmap_adjustments()).pack(side=tk.LEFT, padx=5)
                  
        # Add grid checkbox
        self.heatmap_show_grid = tk.BooleanVar(value=False)
        ttk.Checkbutton(basic_controls, text="Grid", variable=self.heatmap_show_grid, 
                       command=self.update_heatmap_plot).pack(side=tk.LEFT, padx=5)
        
        # Create a frame for X-axis range controls
        range_controls = ttk.LabelFrame(heatmap_controls, text="X-axis Range (cm⁻¹)", padding=5)
        range_controls.pack(fill=tk.X, pady=2)
        
        range_frame = ttk.Frame(range_controls)
        range_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(range_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        self.heatmap_xmin = ttk.Entry(range_frame, width=8)
        self.heatmap_xmin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(range_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        self.heatmap_xmax = ttk.Entry(range_frame, width=8)
        self.heatmap_xmax.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(range_frame, text="Reset Range", 
                  command=lambda: self.reset_heatmap_range()).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(range_frame, text="Update Plot", command=self.update_heatmap_plot).pack(side=tk.LEFT, padx=5)
        
        # Heatmap plot
        self.fig_heatmap, self.ax_heatmap = plt.subplots(figsize=(8, 6))
        self.canvas_heatmap = FigureCanvasTkAgg(self.fig_heatmap, master=heatmap_frame)
        self.canvas_heatmap.draw()
        self.canvas_heatmap.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(heatmap_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_heatmap = NavigationToolbar2Tk(self.canvas_heatmap, toolbar_frame)
        self.toolbar_heatmap.update()

        # Trends tab
        trends_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(trends_frame, text="Fit Results")
        self.fig_trends, self.ax_trends = plt.subplots(figsize=(8, 6))
        self.canvas_trends = FigureCanvasTkAgg(self.fig_trends, master=trends_frame)
        self.canvas_trends.draw()
        self.canvas_trends.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(trends_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_trends = NavigationToolbar2Tk(self.canvas_trends, toolbar_frame)
        self.toolbar_trends.update()

        # Fit Stats tab
        stats_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(stats_frame, text="Fit Stats")
        
        # Create a frame for the plot
        plot_frame = ttk.Frame(stats_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the figure and canvas
        self.fig_stats, self.ax_stats = plt.subplots(3, 3, figsize=(12, 12))
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=plot_frame)
        self.canvas_stats.draw()
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_stats = NavigationToolbar2Tk(self.canvas_stats, toolbar_frame)
        self.toolbar_stats.update()
        
        # Create a frame for the export buttons
        self.export_stats_buttons_frame = ttk.Frame(stats_frame)
        self.export_stats_buttons_frame.pack(side=tk.BOTTOM, pady=5)
        
        # Add export buttons
        self.export_individual_button = ttk.Button(
            self.export_stats_buttons_frame,
            text="Save Individual Plots",
            command=self.export_individual_fit_stats
        )
        
        # Density Analysis tab
        density_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(density_frame, text="Density Analysis")
        
        # Create main paned window for density analysis
        density_paned = ttk.PanedWindow(density_frame, orient=tk.HORIZONTAL)
        density_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        density_controls_frame = ttk.Frame(density_paned, width=300)
        density_paned.add(density_controls_frame, weight=1)
        
        # Create canvas and scrollbar for scrollable content
        density_canvas = tk.Canvas(density_controls_frame)
        density_scrollbar = ttk.Scrollbar(density_controls_frame, orient="vertical", command=density_canvas.yview)
        density_scrollable_frame = ttk.Frame(density_canvas)
        
        density_scrollable_frame.bind(
            "<Configure>",
            lambda e: density_canvas.configure(scrollregion=density_canvas.bbox("all"))
        )
        
        density_canvas.create_window((0, 0), window=density_scrollable_frame, anchor="nw")
        density_canvas.configure(yscrollcommand=density_scrollbar.set)
        
        density_scrollbar.pack(side="right", fill="y")
        density_canvas.pack(side="left", fill="both", expand=True)
        
        # Bind mousewheel for scrolling
        def _on_density_mousewheel(event):
            density_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        density_canvas.bind("<MouseWheel>", _on_density_mousewheel)
        
        # Calibration controls
        calib_frame = ttk.LabelFrame(density_scrollable_frame, text="Calibration", padding=10)
        calib_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(calib_frame, text="Load whewellite reference spectrum to calibrate analysis:").pack(anchor=tk.W, pady=2)
        ttk.Button(calib_frame, text="Load Whewellite Reference", 
                  command=self.load_whewellite_calibration).pack(fill=tk.X, pady=2)
        
        self.calib_status_label = ttk.Label(calib_frame, text="Status: Not calibrated", 
                                           foreground="red")
        self.calib_status_label.pack(anchor=tk.W, pady=2)
        
        # Single spectrum analysis
        single_frame = ttk.LabelFrame(density_scrollable_frame, text="Single Spectrum Analysis", padding=10)
        single_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(single_frame, text="Analyze Current Spectrum", 
                  command=self.analyze_current_spectrum_density).pack(fill=tk.X, pady=2)
        
        self.single_results_text = tk.Text(single_frame, height=6, width=40, wrap=tk.WORD, 
                                          font=("Courier", 9))
        single_scroll = ttk.Scrollbar(single_frame, orient=tk.VERTICAL, 
                                     command=self.single_results_text.yview)
        self.single_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=2)
        single_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=2)
        self.single_results_text.config(yscrollcommand=single_scroll.set)
        
        # Batch analysis controls
        batch_density_frame = ttk.LabelFrame(density_scrollable_frame, text="Batch Analysis", padding=10)
        batch_density_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(batch_density_frame, text="Analyze All Spectra", 
                  command=self.analyze_all_spectra_density).pack(fill=tk.X, pady=2)
        ttk.Button(batch_density_frame, text="Export Density Results", 
                  command=self.export_density_results).pack(fill=tk.X, pady=2)
        
        # Results summary
        results_frame = ttk.LabelFrame(density_scrollable_frame, text="Batch Results Summary", padding=10)
        results_frame.pack(fill=tk.X, pady=5)
        
        self.density_summary_text = tk.Text(results_frame, height=8, width=40, wrap=tk.WORD, 
                                           font=("Courier", 9))
        summary_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                      command=self.density_summary_text.yview)
        self.density_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=2)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=2)
        self.density_summary_text.config(yscrollcommand=summary_scroll.set)
        
        # Right panel for plot
        density_plot_frame = ttk.Frame(density_paned)
        density_paned.add(density_plot_frame, weight=2)
        
        # Create the density analysis plot
        self.fig_density, self.ax_density = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas_density = FigureCanvasTkAgg(self.fig_density, master=density_plot_frame)
        self.canvas_density.draw()
        self.canvas_density.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for density plot
        density_toolbar_frame = ttk.Frame(density_plot_frame)
        density_toolbar_frame.pack(fill=tk.X)
        self.toolbar_density = NavigationToolbar2Tk(self.canvas_density, density_toolbar_frame)
        self.toolbar_density.update()
        
        # Create status bar at the bottom of the window
        self.status_bar = ttk.Label(self.window, text="Press F1 for help with the current tab", relief=tk.SUNKEN, anchor=tk.W, padding=(10, 2))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def add_files(self):
        """Add files to the batch processing list."""
        files = filedialog.askopenfilenames(
            parent=self.window,
            title="Select Raman Spectra Files",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if files:
            was_empty = len(self.spectra_files) == 0
            for file in files:
                if file not in self.spectra_files:
                    self.spectra_files.append(file)
                    self.file_listbox.insert(tk.END, os.path.basename(file))
            
            # If this is the first time adding files, select and load the first file
            if was_empty and self.spectra_files:
                self.file_listbox.selection_set(0)
                self.load_spectrum(0)
    
    def remove_selected_files(self):
        """Remove selected files from the batch processing list."""
        selected_indices = self.file_listbox.curselection()
        for index in sorted(selected_indices, reverse=True):
            self.file_listbox.delete(index)
            del self.spectra_files[index]
        
        # Update current spectrum if needed
        if self.current_spectrum_index >= len(self.spectra_files):
            self.current_spectrum_index = len(self.spectra_files) - 1
            if self.current_spectrum_index >= 0:
                self.load_spectrum(self.current_spectrum_index)
            else:
                self.clear_plot()
        
        # Update waterfall and heatmap plots
        self.update_waterfall_plot()
        self.update_heatmap_plot()
    
    def navigate_spectrum(self, direction):
        """Navigate to the next or previous spectrum."""
        if not self.spectra_files:
            return
        
        # Get current selection
        selection = self.file_listbox.curselection()
        if not selection:
            # If no selection, start from the beginning
            current_index = 0
        else:
            current_index = selection[0]
        
        if direction == 0:  # First
            new_index = 0
        elif direction == -1:  # Previous
            new_index = max(0, current_index - 1)
        elif direction == 1:  # Next
            new_index = min(len(self.spectra_files) - 1, current_index + 1)
        else:  # Last
            new_index = len(self.spectra_files) - 1
        
        if new_index != current_index:
            # Update selection in listbox
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(new_index)
            self.file_listbox.see(new_index)
            # Load the spectrum
            self.load_spectrum(new_index)
    
    
    def detect_encoding_robust(self, file_path):
        """
        Detect file encoding robustly for cross-platform compatibility.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the file
            
        Returns:
        --------
        str
            Detected encoding
        """
        try:
            import chardet
            # Read a sample of the file to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                if result['encoding'] is None:
                    return 'utf-8'  # Default to utf-8 if detection fails
                return result['encoding']
        except ImportError:
            # If chardet is not available, try common encodings
            encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)  # Try to read a small portion
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8'  # Final fallback
        except Exception:
            return 'utf-8'  # Default fallback
    
    def load_spectrum_robust(self, file_path):
        """
        Load spectrum file with robust encoding detection and error handling.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the spectrum file
            
        Returns:
        --------
        tuple
            (wavenumbers, intensities) or (None, None) if failed
        """
        import pandas as pd
        import numpy as np
        
        # Try multiple methods in order of preference
        methods = [
            self._try_numpy_loadtxt,
            self._try_pandas_csv,
            self._try_manual_parsing
        ]
        
        for method in methods:
            try:
                wavenumbers, intensities = method(file_path)
                if wavenumbers is not None and intensities is not None:
                    return wavenumbers, intensities
            except Exception as e:
                continue
        
        return None, None
    
    def _try_numpy_loadtxt(self, file_path):
        """Try loading with numpy loadtxt with encoding detection."""
        import numpy as np
        
        # Detect encoding
        encoding = self.detect_encoding_robust(file_path)
        
        # Try multiple delimiter options
        delimiters = [None, '\t', ',', ';', ' ']
        
        for delimiter in delimiters:
            try:
                # numpy loadtxt doesn't directly support encoding, so we need to handle it differently
                # First, read and decode the file content
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    lines = f.readlines()
                
                # Filter out comment lines and empty lines
                data_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        data_lines.append(line)
                
                if not data_lines:
                    continue
                
                # Parse the data
                data = []
                for line in data_lines:
                    if delimiter is None:
                        # Auto-detect delimiter
                        for delim in ['\t', ',', ';', ' ']:
                            parts = line.split(delim)
                            if len(parts) >= 2:
                                try:
                                    wn = float(parts[0].strip())
                                    intensity = float(parts[1].strip())
                                    data.append([wn, intensity])
                                    break
                                except ValueError:
                                    continue
                    else:
                        parts = line.split(delimiter)
                        if len(parts) >= 2:
                            try:
                                wn = float(parts[0].strip())
                                intensity = float(parts[1].strip())
                                data.append([wn, intensity])
                            except ValueError:
                                continue
                
                if len(data) > 0:
                    data_array = np.array(data)
                    return data_array[:, 0], data_array[:, 1]
                    
            except Exception:
                continue
        
        return None, None
    
    def _try_pandas_csv(self, file_path):
        """Try loading with pandas with robust encoding detection."""
        import pandas as pd
        import numpy as np
        
        # Try multiple encoding and delimiter combinations
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']
        delimiters_to_try = [None, '\t', ',', ';', ' ']
        
        # First, try to detect encoding
        detected_encoding = self.detect_encoding_robust(file_path)
        encodings_to_try.insert(0, detected_encoding)
        
        for encoding in encodings_to_try:
            for delimiter in delimiters_to_try:
                try:
                    df = pd.read_csv(file_path, 
                                   sep=delimiter,
                                   engine='python', 
                                   header=None, 
                                   comment='#',
                                   encoding=encoding,
                                   on_bad_lines='skip')
                    
                    if len(df) > 0 and len(df.columns) >= 2:
                        # Use first two numeric columns
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            wavenumbers = df[numeric_cols[0]].values
                            intensities = df[numeric_cols[1]].values
                            return wavenumbers, intensities
                        elif len(df.columns) >= 2:
                            # Try to convert first two columns to numeric
                            try:
                                wavenumbers = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                                intensities = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
                                # Remove NaN values
                                valid_mask = ~(np.isnan(wavenumbers) | np.isnan(intensities))
                                if np.sum(valid_mask) > 0:
                                    return wavenumbers[valid_mask], intensities[valid_mask]
                            except:
                                continue
                
                except Exception:
                    continue
        
        return None, None
    
    def _try_manual_parsing(self, file_path):
        """Try manual parsing with robust encoding detection."""
        import numpy as np
        
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']
        detected_encoding = self.detect_encoding_robust(file_path)
        encodings_to_try.insert(0, detected_encoding)
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    wavenumbers = []
                    intensities = []
                    
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Try different delimiters
                        for delimiter in ['\t', ',', ';', ' ']:
                            parts = line.split(delimiter)
                            if len(parts) >= 2:
                                try:
                                    wn = float(parts[0].strip())
                                    intensity = float(parts[1].strip())
                                    wavenumbers.append(wn)
                                    intensities.append(intensity)
                                    break
                                except ValueError:
                                    continue
                    
                    if len(wavenumbers) > 0:
                        return np.array(wavenumbers), np.array(intensities)
            
            except Exception:
                continue
        
        return None, None


    def load_spectrum(self, index):
        """Load a spectrum from the list."""
        if not self.spectra_files or index < 0 or index >= len(self.spectra_files):
            self.clear_plot()
            return
            
        try:
            # Load the spectrum data
            file_path = self.spectra_files[index]
            # Load spectrum with robust encoding detection
            wavenumbers, intensities = self.load_spectrum_robust(file_path)
            if wavenumbers is None or intensities is None:
                raise Exception(f"Failed to load data from {file_path}")
            data = np.column_stack((wavenumbers, intensities))
            self.wavenumbers = data[:, 0]
            self.spectra = data[:, 1]
            self.original_spectra = np.copy(self.spectra)
            
            # Update current spectrum index and label
            self.current_spectrum_index = index
            # Update status
            filename = os.path.basename(self.spectra_files[index])
            # Check if this file is manually skipped
            current_file_path = self.spectra_files[index]
            is_skipped = current_file_path in self.manually_skipped_files
            
            if is_skipped:
                self.current_spectrum_label.config(text=f"Spectrum {index + 1}/{len(self.spectra_files)}: {filename} [SKIPPED]")
                # Update skip button text
                if hasattr(self, 'skip_button'):
                    self.skip_button.config(text="✓ Unskip This File")
            else:
                self.current_spectrum_label.config(text=f"Spectrum {index + 1}/{len(self.spectra_files)}: {filename}")
                # Update skip button text
                if hasattr(self, 'skip_button'):
                    self.skip_button.config(text="⚠ Skip This File")
            
            # Initialize peak fitting variables if they don't exist
            if not hasattr(self, 'peaks'):
                self.peaks = []
            if not hasattr(self, 'fit_params'):
                self.fit_params = []
            if not hasattr(self, 'fit_result'):
                self.fit_result = None
            if not hasattr(self, 'background'):
                self.background = None
            if not hasattr(self, 'residuals'):
                self.residuals = None
            if not hasattr(self, 'peak_r_squared'):
                self.peak_r_squared = []
                
            # Check if this spectrum has batch results and load them
            if hasattr(self, 'batch_results') and self.batch_results:
                for result in self.batch_results:
                    if 'file' in result and result['file'] == file_path:
                        # Load the saved fit parameters, peaks, and results
                        self.peaks = result['peaks'].copy() if 'peaks' in result and result['peaks'] is not None else []
                        self.fit_params = np.copy(result['fit_params']) if 'fit_params' in result and result['fit_params'] is not None else None
                        self.fit_cov = np.copy(result['fit_cov']) if 'fit_cov' in result and result['fit_cov'] is not None else None
                        self.background = np.copy(result['background']) if 'background' in result and result['background'] is not None else None
                        self.fit_result = np.copy(result['fit_result']) if 'fit_result' in result and result['fit_result'] is not None else None
                        
                        # Load stored R² values if available
                        if 'peak_r_squared' in result and result['peak_r_squared'] is not None:
                            self.peak_r_squared = result['peak_r_squared'].copy()
                        
                        # Recalculate residuals using background-subtracted data and fit
                        # Important: residuals should always be (subtracted spectrum - fit)
                        if self.fit_result is not None and self.background is not None:
                            # Calculate background-subtracted spectrum
                            background_subtracted = self.original_spectra - self.background
                            # Calculate residuals from the subtracted spectrum
                            self.residuals = background_subtracted - self.fit_result
                        break
            
            # Update the plot
            self.update_plot()
            
            # Update slider ranges based on new spectrum data
            self.update_slider_ranges()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load spectrum: {str(e)}", parent=self.window)
            self.clear_plot()
    
    def set_reference(self):
        """Set the current spectrum as the reference for batch processing."""
        if not hasattr(self, 'peaks') or not self.peaks:
            messagebox.showwarning("No Peaks", "Please detect and fit peaks first.", parent=self.window)
            return
            
        self.reference_peaks = self.peaks.copy()
        self.reference_background = self.background.copy() if self.background is not None else None
        messagebox.showinfo("Reference Set", "Current spectrum set as reference for batch processing.", parent=self.window)
    
    def apply_to_all(self):
        """Apply the current peak fitting parameters to all spectra."""
        if not self.reference_peaks:
            messagebox.showwarning("No Reference", "Please set a reference spectrum first.", parent=self.window)
            return
        
        self._stop_batch = False  # Reset stop flag
        # Clear previous batch results
        self.batch_results = []
        # Store current index
        current_index = self.current_spectrum_index
        
        # Enable text widget for updating
        self.batch_status_text.config(state=tk.NORMAL)
        self.batch_status_text.delete(1.0, tk.END)
        self.batch_status_text.insert(tk.END, "Starting batch processing...\n")
        self.batch_status_text.see(tk.END)
        self.window.update()
        
        # Process all spectra
        total_spectra = len(self.spectra_files)
        for i in range(total_spectra):
            if self._stop_batch:
                self.batch_status_text.insert(tk.END, "Batch processing stopped by user.\n")
                self.batch_status_text.see(tk.END)
                self.window.update()
                messagebox.showinfo("Stopped", "Batch processing was stopped by the user.", parent=self.window)
                break
        
            # Reset fit_failed flag for this iteration
            self.fit_failed = False
            
            try:
                # Update status text - only show which spectrum is being processed
                spectrum_name = os.path.basename(self.spectra_files[i])
                self.batch_status_text.insert(tk.END, f"Processing {i+1}/{total_spectra}: {spectrum_name}\n")
                self.batch_status_text.see(tk.END)
                self.window.update()
                
                self.load_spectrum(i)
                
                # Apply reference peaks
                self.peaks = self.reference_peaks.copy()
                
                # Subtract background using current spectrum's parameters
                try:
                    # Get parameters
                    lambda_val = float(self.var_lambda.get())
                    p_val = float(self.var_p.get())
                    
                    # Calculate background
                    L = len(self.spectra)
                    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
                    w = np.ones(L)
                    
                    for j in range(10):
                        W = sparse.spdiags(w, 0, L, L)
                        Z = W + lambda_val * D.dot(D.transpose())
                        z = spsolve(Z, w*self.spectra)
                        w = p_val * (self.spectra > z) + (1-p_val) * (self.spectra < z)
                    
                    self.background = z
                    self.spectra = self.original_spectra - self.background
                except Exception as e:
                    error_msg = f"Background subtraction failed: {str(e)}"
                    print(error_msg)
                    self.batch_status_text.insert(tk.END, f"  ✗ {error_msg}\n")
                    self.batch_status_text.see(tk.END)
                    self.window.update()
                    self.fit_failed = True
                
                # Fit peaks only if background subtraction succeeded
                if not self.fit_failed:                    
                    self.fit_peaks()
                    
                    # Get NLLS cycle count from the fit_peaks method
                    cycle_count = getattr(self, 'nlls_cycles', 0)
                    
                    if self.fit_failed:
                        self.batch_status_text.insert(tk.END, f"  ✗ Fit failed after {cycle_count} NLLS cycles\n")
                    else:
                        self.batch_status_text.insert(tk.END, f"  ✓ Fit successful: {cycle_count} NLLS cycles\n")
                    self.batch_status_text.see(tk.END)
                    self.window.update()
                
                # Store results - even if fit failed (with appropriate placeholders)
                result_entry = {
                    'file': self.spectra_files[i],
                    'peaks': self.peaks.copy() if hasattr(self, 'peaks') else [],
                    'fit_failed': self.fit_failed,  # Explicitly store fit failure status
                    'fit_params': np.copy(self.fit_params) if hasattr(self, 'fit_params') and self.fit_params is not None else None,
                    'fit_cov': np.copy(self.fit_cov) if hasattr(self, 'fit_cov') and self.fit_cov is not None else None,
                    'background': np.copy(self.background) if hasattr(self, 'background') and self.background is not None else None,
                    'fit_result': np.copy(self.fit_result) if hasattr(self, 'fit_result') and self.fit_result is not None else None,
                    'original_spectra': np.copy(self.original_spectra) if hasattr(self, 'original_spectra') else None,  # Store original spectrum too
                    'peak_r_squared': self.peak_r_squared.copy() if hasattr(self, 'peak_r_squared') and self.peak_r_squared is not None else [],
                    'nlls_cycles': getattr(self, 'nlls_cycles', 0)  # Store NLLS cycle count
                }
                
                self.batch_results.append(result_entry)
                
            except Exception as e:
                error_msg = f"Error processing spectrum {i+1}: {str(e)}"
                print(error_msg)
                self.batch_status_text.insert(tk.END, f"  ✗ {error_msg}\n")
                self.batch_status_text.see(tk.END)
                self.window.update()
                
                # Add placeholder entry for failed spectrum
                self.batch_results.append({
                    'file': self.spectra_files[i],
                    'peaks': [],
                    'fit_failed': True,
                    'fit_params': None,
                    'fit_cov': None,
                    'background': None,
                    'fit_result': None,
                    'original_spectra': np.copy(self.original_spectra) if hasattr(self, 'original_spectra') else None,
                    'peak_r_squared': [],
                    'nlls_cycles': 0  # Zero cycles for failed fits
                })
            
            # Update progress
            self.current_spectrum_label.config(text=f"Processing: {i+1}/{total_spectra}")
            self.window.update()
        
        # Restore original index
        if 0 <= current_index < len(self.spectra_files):
            self.load_spectrum(current_index)
        
        # Update peak visibility controls and trends plot
        self.update_peak_visibility_controls()
        self.update_trends_plot()
        self.update_fit_stats_plot()
        
        # Calculate and show statistics
        if self.batch_results:
            cycles_stats = [r.get('nlls_cycles', 0) for r in self.batch_results if not r.get('fit_failed', True)]
            if cycles_stats:
                avg_cycles = sum(cycles_stats) / len(cycles_stats)
                max_cycles = max(cycles_stats)
                min_cycles = min(cycles_stats)
                self.batch_status_text.insert(tk.END, f"\nNLLS cycle statistics:\n")
                self.batch_status_text.insert(tk.END, f"  Average: {avg_cycles:.1f}\n")
                self.batch_status_text.insert(tk.END, f"  Min: {min_cycles}\n")
                self.batch_status_text.insert(tk.END, f"  Max: {max_cycles}\n")
        
        # Final status update
        if not self._stop_batch:
            self.batch_status_text.insert(tk.END, "\nBatch processing completed successfully.\n")
            self.batch_status_text.see(tk.END)
            messagebox.showinfo("Complete", "Batch processing completed.", parent=self.window)
            
        # Make text widget read-only again
        self.batch_status_text.config(state=tk.DISABLED)
    
    def update_peak_visibility_controls(self):
        """Update the peak visibility checkboxes based on the current number of peaks."""
        # Clear existing checkboxes
        for widget in self.peak_visibility_frame.winfo_children():
            widget.destroy()
        self.peak_visibility_vars = []
        
        if not self.batch_results or len(self.batch_results) == 0:
            return
            
        # Get number of peaks
        model_type = self.current_model.get()
        if model_type == "Gaussian" or model_type == "Lorentzian":
            params_per_peak = 3
        elif model_type == "Pseudo-Voigt":
            params_per_peak = 4
        elif model_type == "Asymmetric Voigt":
            params_per_peak = 5
        else:
            params_per_peak = 3
            
        try:
            n_peaks = len(self.batch_results[0]['fit_params']) // params_per_peak
        except (KeyError, TypeError, AttributeError):
            return
        
        # Create checkboxes for each peak
        for i in range(n_peaks):
            var = tk.BooleanVar(value=True)
            self.peak_visibility_vars.append(var)
            cb = ttk.Checkbutton(
                self.peak_visibility_frame,
                text=f"Peak {i+1}",
                variable=var,
                command=self.update_trends_plot
            )
            cb.pack(anchor=tk.W, pady=2)
    
    def show_all_peaks(self):
        """Show all peaks in the trends plot."""
        for var in self.peak_visibility_vars:
            var.set(True)
        self.update_trends_plot()
    
    def hide_all_peaks(self):
        """Hide all peaks in the trends plot."""
        for var in self.peak_visibility_vars:
            var.set(False)
        self.update_trends_plot()
    
    def extract_trend_data(self):
        """Extracts parameter trends for all peaks and all batch results, according to the current model."""
        model_type = self.current_model.get()
        if model_type in ["Gaussian", "Lorentzian"]:
            params_per_peak = 3
        elif model_type == "Pseudo-Voigt":
            params_per_peak = 4
        elif model_type == "Asymmetric Voigt":
            params_per_peak = 5
        else:
            params_per_peak = 3
        n_peaks = None
        all_params = []
        all_covs = []  # Add storage for covariance matrices
        
        # Don't filter out manually skipped files - preserve original indexing
        # This maintains the time series integrity for kinetic studies
        for result in self.batch_results:
            # Check if this is a manually skipped file or failed fit
            manually_skipped = result.get('manually_skipped', False)
            fit_failed = result.get('fit_failed', False)
            fit_params = result.get('fit_params')
            fit_cov = result.get('fit_cov')  # Get covariance matrix
            
            if not manually_skipped and not fit_failed and fit_params is not None:
                if n_peaks is None:
                    n_peaks = len(fit_params) // params_per_peak
                all_params.append(np.array(fit_params).reshape(-1, params_per_peak))
                all_covs.append(fit_cov)  # Store covariance matrix
            else:
                # For failed fits or manually skipped files, append None to maintain array index correspondence
                all_params.append(None)
                all_covs.append(None)
        
        if n_peaks is None:
            return None, None, None, None
        
        # Use original spectrum numbering (0, 1, 2, ...) to preserve time series
        x = np.arange(len(self.batch_results))
        trends = []
        uncertainties = []  # Add storage for uncertainties
        
        for peak_idx in range(n_peaks):
            pos_vals, amp_vals, wid_vals, eta_vals, wid_left_vals, wid_right_vals = [], [], [], [], [], []
            pos_errs, amp_errs, wid_errs, eta_errs, wid_left_errs, wid_right_errs = [], [], [], [], [], []  # Add error storage
            
            for i, (params, cov) in enumerate(zip(all_params, all_covs)):
                # Initialize with NaN for all spectra indices
                pos_vals.append(np.nan)
                amp_vals.append(np.nan)
                wid_vals.append(np.nan)
                eta_vals.append(np.nan)
                wid_left_vals.append(np.nan)
                wid_right_vals.append(np.nan)
                
                pos_errs.append(None)
                amp_errs.append(None)
                wid_errs.append(None)
                eta_errs.append(None)
                wid_left_errs.append(None)
                wid_right_errs.append(None)
                
                # Only fill in values for successful fits
                if params is not None and peak_idx < params.shape[0]:
                    if model_type in ["Gaussian", "Lorentzian"]:
                        amp, pos, wid = params[peak_idx]
                        pos_vals[i] = pos
                        amp_vals[i] = amp
                        wid_vals[i] = wid
                        # Skip uncertainty calculation for constrained fits
                        # The covariance matrix dimensions don't match when parameters are fixed
                        # For now, set uncertainties to None to avoid crashes
                        pos_errs[i] = None
                        amp_errs[i] = None
                        wid_errs[i] = None
                    elif model_type == "Pseudo-Voigt":
                        amp, pos, wid, eta = params[peak_idx]
                        pos_vals[i] = pos
                        amp_vals[i] = amp
                        wid_vals[i] = wid
                        eta_vals[i] = eta
                        # Skip uncertainty calculation for constrained fits
                        pos_errs[i] = None
                        amp_errs[i] = None
                        wid_errs[i] = None
                        eta_errs[i] = None
                    elif model_type == "Asymmetric Voigt":
                        amp, pos, wid_left, wid_right, eta = params[peak_idx]
                        pos_vals[i] = pos
                        amp_vals[i] = amp
                        wid_left_vals[i] = wid_left
                        wid_right_vals[i] = wid_right
                        eta_vals[i] = eta
                        # Skip uncertainty calculation for constrained fits
                        pos_errs[i] = None
                        amp_errs[i] = None
                        wid_left_errs[i] = None
                        wid_right_errs[i] = None
                        eta_errs[i] = None
            
            trends.append({
                'pos': np.array(pos_vals),
                'amp': np.array(amp_vals),
                'wid': np.array(wid_vals),
                'eta': np.array(eta_vals),
                'wid_left': np.array(wid_left_vals),
                'wid_right': np.array(wid_right_vals)
            })
            uncertainties.append({
                'pos': np.array(pos_errs),
                'amp': np.array(amp_errs),
                'wid': np.array(wid_errs),
                'eta': np.array(eta_errs),
                'wid_left': np.array(wid_left_errs),
                'wid_right': np.array(wid_right_errs)
            })
        return x, trends, n_peaks, uncertainties

    def update_trends_plot(self):
        """Update the trends plot with batch processing results."""
        if not self.batch_results:
            self.fig_trends.clear()
            self.fig_trends.text(0.5, 0.5, 'No batch results to display.', ha='center', va='center')
            self.canvas_trends.draw()
            return

        # Always use 2x2 grid
        self.fig_trends.clear()
        axes = self.fig_trends.subplots(2, 2).flatten()
        subplot_titles = ["Position (cm⁻¹)", "Amplitude", "Width (cm⁻¹)", "Eta"]
        
        x, trends, n_peaks, uncertainties = self.extract_trend_data()
        if x is None or trends is None or n_peaks is None:
            for ax in axes:
                ax.text(0.5, 0.5, 'No fit results to display.', ha='center', va='center')
            self.fig_trends.tight_layout()
            self.canvas_trends.draw()
            return

        model_type = self.current_model.get()
        show_95 = self.show_95_boundary.get()
        show_grid = self.show_fit_grid.get()

        # Only plot peaks whose checkboxes are checked
        visible_peaks = [i for i in range(n_peaks) if i < len(self.peak_visibility_vars) and self.peak_visibility_vars[i].get()]

        # Count how many spectra had failed fits (excluding manually skipped)
        failed_fits_count = sum(1 for result in self.batch_results 
                               if result.get('fit_failed', False) and not result.get('manually_skipped', False))
        # Count how many spectra were manually refined
        manually_refined_count = sum(1 for result in self.batch_results 
                                   if result.get('manually_refined', False) and not result.get('manually_skipped', False))
        # Count how many spectra were manually skipped
        manually_skipped_count = sum(1 for result in self.batch_results if result.get('manually_skipped', False))
        
        if failed_fits_count > 0 or manually_refined_count > 0 or manually_skipped_count > 0:
            status_text = []
            if failed_fits_count > 0:
                status_text.append(f"{failed_fits_count} failed fits (shown as gaps)")
            if manually_skipped_count > 0:
                status_text.append(f"{manually_skipped_count} files manually skipped (shown as gaps)")
            if manually_refined_count > 0:
                status_text.append(f"{manually_refined_count} manually refined (★)")
            
            for ax in axes:
                ax.text(0.02, 0.98, "; ".join(status_text), 
                       transform=ax.transAxes, fontsize=8, va='top', ha='left', 
                       color='blue', bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

        # Add information about constraints if any are active
        constraints_active = False
        if hasattr(self, 'fix_positions') and hasattr(self, 'fix_widths'):
            constraints_active = self.fix_positions.get() or self.fix_widths.get()
        
        if constraints_active and show_95:
            # Add a note that uncertainty bands are disabled when constraints are used
            constraint_text = "Note: 95% confidence bands disabled when parameter constraints are active"
            axes[3].text(0.02, 0.02, constraint_text, 
                        transform=axes[3].transAxes, fontsize=8, va='bottom', ha='left', 
                        color='orange', style='italic',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'))

        # Create arrays to track which points are manually refined

        for peak_idx in visible_peaks:
            color = f"C{peak_idx}"
            
            # Position
            y = trends[peak_idx]['pos']
            yerr = uncertainties[peak_idx]['pos']
            valid_mask = ~np.isnan(y)
            
            if np.any(valid_mask):
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                yerr_valid = yerr[valid_mask]
                
                # Plot points
                axes[0].plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1}', color=color)
                
                
                # Draw smooth confidence bands without vertical lines
                if show_95:
                    # Find consecutive groups of points
                    # This identifies runs of adjacent indices in the original array
                    groups = []
                    current_group = []
                    for i, idx in enumerate(np.where(valid_mask)[0]):
                        if i == 0 or idx == np.where(valid_mask)[0][i-1] + 1:
                            # Consecutive point, add to current group
                            current_group.append(i)
                        else:
                            # Gap detected, start a new group
                            if current_group:
                                groups.append(current_group)
                            current_group = [i]
                    if current_group:
                        groups.append(current_group)
                    
                    # Draw each group as a separate smooth confidence band
                    for group in groups:
                        if len(group) < 2:
                            continue  # Need at least 2 points to draw a band
                            
                        # Get values for this group
                        xg = x_valid[group]
                        yg = y_valid[group]
                        yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                        
                        if len(yerrg) != len(xg):
                            continue  # Skip if not all errors are valid
                            
                        # Create a high-resolution interpolation for smooth curves
                        x_interp = np.linspace(xg[0], xg[-1], 100)
                        y_interp = np.interp(x_interp, xg, yg)
                        
                        # For the upper and lower bounds, interpolate separately
                        upper_bound = np.interp(x_interp, xg, yg + yerrg)
                        lower_bound = np.interp(x_interp, xg, yg - yerrg)
                        
                        # Fill once per group with a single call to avoid vertical lines
                        axes[0].fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
            
            # Amplitude
            y = trends[peak_idx]['amp']
            yerr = uncertainties[peak_idx]['amp']
            valid_mask = ~np.isnan(y)
            
            if np.any(valid_mask):
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                yerr_valid = yerr[valid_mask]
                
                axes[1].plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1}', color=color)
                
                
                if show_95:
                    # Find consecutive groups of points
                    groups = []
                    current_group = []
                    for i, idx in enumerate(np.where(valid_mask)[0]):
                        if i == 0 or idx == np.where(valid_mask)[0][i-1] + 1:
                            current_group.append(i)
                        else:
                            if current_group:
                                groups.append(current_group)
                            current_group = [i]
                    if current_group:
                        groups.append(current_group)
                    
                    for group in groups:
                        if len(group) < 2:
                            continue
                            
                        xg = x_valid[group]
                        yg = y_valid[group]
                        yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                        
                        if len(yerrg) != len(xg):
                            continue
                            
                        x_interp = np.linspace(xg[0], xg[-1], 100)
                        y_interp = np.interp(x_interp, xg, yg)
                        upper_bound = np.interp(x_interp, xg, yg + yerrg)
                        lower_bound = np.interp(x_interp, xg, yg - yerrg)
                        
                        axes[1].fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
            
            # Width
            if model_type == "Asymmetric Voigt":
                y_left = trends[peak_idx]['wid_left']
                y_right = trends[peak_idx]['wid_right']
                yerr_left = uncertainties[peak_idx]['wid_left']
                yerr_right = uncertainties[peak_idx]['wid_right']
                
                valid_mask_left = ~np.isnan(y_left)
                valid_mask_right = ~np.isnan(y_right)
                
                if np.any(valid_mask_left):
                    x_valid = x[valid_mask_left]
                    y_valid = y_left[valid_mask_left]
                    yerr_valid = yerr_left[valid_mask_left]
                    
                    axes[2].plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1} Left', color=color, alpha=0.7)
                    
                    
                    if show_95:
                        groups = []
                        current_group = []
                        for i, idx in enumerate(np.where(valid_mask_left)[0]):
                            if i == 0 or idx == np.where(valid_mask_left)[0][i-1] + 1:
                                current_group.append(i)
                            else:
                                if current_group:
                                    groups.append(current_group)
                                current_group = [i]
                        if current_group:
                            groups.append(current_group)
                        
                        for group in groups:
                            if len(group) < 2:
                                continue
                                
                            xg = x_valid[group]
                            yg = y_valid[group]
                            yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                            
                            if len(yerrg) != len(xg):
                                continue
                                
                            x_interp = np.linspace(xg[0], xg[-1], 100)
                            y_interp = np.interp(x_interp, xg, yg)
                            upper_bound = np.interp(x_interp, xg, yg + yerrg)
                            lower_bound = np.interp(x_interp, xg, yg - yerrg)
                            
                            axes[2].fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
                
                if np.any(valid_mask_right):
                    x_valid = x[valid_mask_right]
                    y_valid = y_right[valid_mask_right]
                    yerr_valid = yerr_right[valid_mask_right]
                    
                    axes[2].plot(x_valid, y_valid, '^', label=f'Peak {peak_idx+1} Right', color=color, alpha=0.7)
                    
                    
                    if show_95:
                        groups = []
                        current_group = []
                        for i, idx in enumerate(np.where(valid_mask_right)[0]):
                            if i == 0 or idx == np.where(valid_mask_right)[0][i-1] + 1:
                                current_group.append(i)
                            else:
                                if current_group:
                                    groups.append(current_group)
                                current_group = [i]
                        if current_group:
                            groups.append(current_group)
                        
                        for group in groups:
                            if len(group) < 2:
                                continue
                                
                            xg = x_valid[group]
                            yg = y_valid[group]
                            yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                            
                            if len(yerrg) != len(xg):
                                continue
                                
                            x_interp = np.linspace(xg[0], xg[-1], 100)
                            y_interp = np.interp(x_interp, xg, yg)
                            upper_bound = np.interp(x_interp, xg, yg + yerrg)
                            lower_bound = np.interp(x_interp, xg, yg - yerrg)
                            
                            axes[2].fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
            else:
                y = trends[peak_idx]['wid']
                yerr = uncertainties[peak_idx]['wid']
                valid_mask = ~np.isnan(y)
                
                if np.any(valid_mask):
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    yerr_valid = yerr[valid_mask]
                    
                    axes[2].plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1}', color=color)
                    
                    if show_95:
                        groups = []
                        current_group = []
                        for i, idx in enumerate(np.where(valid_mask)[0]):
                            if i == 0 or idx == np.where(valid_mask)[0][i-1] + 1:
                                current_group.append(i)
                            else:
                                if current_group:
                                    groups.append(current_group)
                                current_group = [i]
                        if current_group:
                            groups.append(current_group)
                        
                        for group in groups:
                            if len(group) < 2:
                                continue
                                
                            xg = x_valid[group]
                            yg = y_valid[group]
                            yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                            
                            if len(yerrg) != len(xg):
                                continue
                                
                            x_interp = np.linspace(xg[0], xg[-1], 100)
                            y_interp = np.interp(x_interp, xg, yg)
                            upper_bound = np.interp(x_interp, xg, yg + yerrg)
                            lower_bound = np.interp(x_interp, xg, yg - yerrg)
                            
                            axes[2].fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
            
            # Eta
            if model_type in ["Pseudo-Voigt", "Asymmetric Voigt"]:
                y = trends[peak_idx]['eta']
                yerr = uncertainties[peak_idx]['eta']
                valid_mask = ~np.isnan(y)
                
                if np.any(valid_mask):
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    yerr_valid = yerr[valid_mask]
                    
                    axes[3].plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1}', color=color)
                    
                    
                    if show_95:
                        groups = []
                        current_group = []
                        for i, idx in enumerate(np.where(valid_mask)[0]):
                            if i == 0 or idx == np.where(valid_mask)[0][i-1] + 1:
                                current_group.append(i)
                            else:
                                if current_group:
                                    groups.append(current_group)
                                current_group = [i]
                        if current_group:
                            groups.append(current_group)
                        
                        for group in groups:
                            if len(group) < 2:
                                continue
                                
                            xg = x_valid[group]
                            yg = y_valid[group]
                            yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                            
                            if len(yerrg) != len(xg):
                                continue
                                
                            x_interp = np.linspace(xg[0], xg[-1], 100)
                            y_interp = np.interp(x_interp, xg, yg)
                            upper_bound = np.interp(x_interp, xg, yg + yerrg)
                            lower_bound = np.interp(x_interp, xg, yg - yerrg)
                            
                            axes[3].fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)

        # If model does not support Eta, show message in bottom-right
        if model_type not in ["Pseudo-Voigt", "Asymmetric Voigt"]:
            axes[3].text(0.5, 0.5, 'Eta not available for current model', ha='center', va='center', transform=axes[3].transAxes)

        # Configure all axes
        for i, ax in enumerate(axes):
            ax.set_title(subplot_titles[i])
            ax.set_xlabel('Spectrum Number')
            if i == 0:
                ax.set_ylabel('Position (cm⁻¹)')
            elif i == 1:
                ax.set_ylabel('Amplitude')
            elif i == 2:
                ax.set_ylabel('Width (cm⁻¹)')
            elif i == 3:
                ax.set_ylabel('Eta')
            if show_grid:
                ax.grid(True, linestyle=':', color='gray', alpha=0.6)
            else:
                ax.grid(False)
            ax.legend()

        self.fig_trends.tight_layout()
        self.canvas_trends.draw()

    def _plot_single_trends_subplot(self, ax, which):
        """Helper to plot a single trends subplot (position, amplitude, width, eta) with current options."""
        if not self.batch_results:
            ax.text(0.5, 0.5, 'No batch results to display.', ha='center', va='center')
            return
        x, trends, n_peaks, uncertainties = self.extract_trend_data()
        if x is None or trends is None or n_peaks is None:
            ax.text(0.5, 0.5, 'No fit results to display.', ha='center', va='center')
            return
        model_type = self.current_model.get()
        show_95 = self.show_95_boundary.get()
        show_grid = self.show_fit_grid.get()
        visible_peaks = [i for i in range(n_peaks) if i < len(self.peak_visibility_vars) and self.peak_visibility_vars[i].get()]
        plotted = False
        
        # Count failed fits
        failed_fits_count = sum(1 for result in self.batch_results if result.get('fit_failed', False))
        if failed_fits_count > 0:
            ax.text(0.02, 0.98, f"Note: {failed_fits_count} failed fits excluded", 
                   transform=ax.transAxes, fontsize=8, va='top', ha='left', 
                   color='red', bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        # Create arrays to track which points are manually refined
        
        for peak_idx in visible_peaks:
            color = f"C{peak_idx}"
            if which == 'position':
                y = trends[peak_idx]['pos']
                yerr = uncertainties[peak_idx]['pos']
                valid_mask = ~np.isnan(y)
                
                if np.any(valid_mask):
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    yerr_valid = yerr[valid_mask]
                    
                    # Plot points
                    ax.plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1}', color=color)
                    
                    # Draw confidence bands
                    if show_95:
                        # Find consecutive groups of points
                        groups = []
                        current_group = []
                        for i, idx in enumerate(np.where(valid_mask)[0]):
                            if i == 0 or idx == np.where(valid_mask)[0][i-1] + 1:
                                current_group.append(i)
                            else:
                                if current_group:
                                    groups.append(current_group)
                                current_group = [i]
                        if current_group:
                            groups.append(current_group)
                        
                        # Draw each group as a separate smooth confidence band
                        for group in groups:
                            if len(group) < 2:
                                continue  # Need at least 2 points to draw a band
                                
                            # Get values for this group
                            xg = x_valid[group]
                            yg = y_valid[group]
                            yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                            
                            if len(yerrg) != len(xg):
                                continue  # Skip if not all errors are valid
                                
                            # Create a high-resolution interpolation for smooth curves
                            x_interp = np.linspace(xg[0], xg[-1], 100)
                            y_interp = np.interp(x_interp, xg, yg)
                            
                            # For the upper and lower bounds, interpolate separately
                            upper_bound = np.interp(x_interp, xg, yg + yerrg)
                            lower_bound = np.interp(x_interp, xg, yg - yerrg)
                            
                            # Fill once per group with a single call to avoid vertical lines
                            ax.fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
                
                ax.set_title('Peak Position')
                ax.set_ylabel('Position (cm⁻¹)')
                plotted = True
                
            elif which == 'amplitude':
                y = trends[peak_idx]['amp']
                yerr = uncertainties[peak_idx]['amp']
                valid_mask = ~np.isnan(y)
                
                if np.any(valid_mask):
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    yerr_valid = yerr[valid_mask]
                    
                    # Plot points
                    ax.plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1}', color=color)
                    
                    # Draw confidence bands
                    if show_95:
                        groups = []
                        current_group = []
                        for i, idx in enumerate(np.where(valid_mask)[0]):
                            if i == 0 or idx == np.where(valid_mask)[0][i-1] + 1:
                                current_group.append(i)
                            else:
                                if current_group:
                                    groups.append(current_group)
                                current_group = [i]
                        if current_group:
                            groups.append(current_group)
                        
                        for group in groups:
                            if len(group) < 2:
                                continue
                                
                            xg = x_valid[group]
                            yg = y_valid[group]
                            yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                            
                            if len(yerrg) != len(xg):
                                continue
                                
                            x_interp = np.linspace(xg[0], xg[-1], 100)
                            y_interp = np.interp(x_interp, xg, yg)
                            upper_bound = np.interp(x_interp, xg, yg + yerrg)
                            lower_bound = np.interp(x_interp, xg, yg - yerrg)
                            
                            ax.fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
                
                ax.set_title('Peak Amplitude')
                ax.set_ylabel('Amplitude')
                plotted = True
                
            elif which == 'width':
                if model_type == "Asymmetric Voigt":
                    y_left = trends[peak_idx]['wid_left']
                    y_right = trends[peak_idx]['wid_right']
                    yerr_left = uncertainties[peak_idx]['wid_left']
                    yerr_right = uncertainties[peak_idx]['wid_right']
                    
                    valid_mask_left = ~np.isnan(y_left)
                    valid_mask_right = ~np.isnan(y_right)
                    
                    if np.any(valid_mask_left):
                        x_valid = x[valid_mask_left]
                        y_valid = y_left[valid_mask_left]
                        yerr_valid = yerr_left[valid_mask_left]
                        
                        ax.plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1} Left', color=color, alpha=0.7)
                        
                        if show_95:
                            groups = []
                            current_group = []
                            for i, idx in enumerate(np.where(valid_mask_left)[0]):
                                if i == 0 or idx == np.where(valid_mask_left)[0][i-1] + 1:
                                    current_group.append(i)
                                else:
                                    if current_group:
                                        groups.append(current_group)
                                    current_group = [i]
                            if current_group:
                                groups.append(current_group)
                            
                            for group in groups:
                                if len(group) < 2:
                                    continue
                                    
                                xg = x_valid[group]
                                yg = y_valid[group]
                                yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                                
                        x_valid = x[valid_mask_right]
                        if np.any(manually_refined_valid):
                            x_refined = x_valid[manually_refined_valid]
                            y_refined = y_valid[manually_refined_valid]
                            ax.plot(x_refined, y_refined, '*', color=color, markersize=10, markeredgecolor='black', markeredgewidth=0.5)
                        
                        if show_95:
                            groups = []
                            current_group = []
                            for i, idx in enumerate(np.where(valid_mask_right)[0]):
                                if i == 0 or idx == np.where(valid_mask_right)[0][i-1] + 1:
                                    current_group.append(i)
                                else:
                                    if current_group:
                                        groups.append(current_group)
                                    current_group = [i]
                            if current_group:
                                groups.append(current_group)
                            
                            for group in groups:
                                if len(group) < 2:
                                    continue
                                    
                                xg = x_valid[group]
                                yg = y_valid[group]
                                yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                                
                                if len(yerrg) != len(xg):
                                    continue
                                    
                                x_interp = np.linspace(xg[0], xg[-1], 100)
                                y_interp = np.interp(x_interp, xg, yg)
                                upper_bound = np.interp(x_interp, xg, yg + yerrg)
                                lower_bound = np.interp(x_interp, xg, yg - yerrg)
                                
                                ax.fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
                else:
                    y = trends[peak_idx]['wid']
                    yerr = uncertainties[peak_idx]['wid']
                    valid_mask = ~np.isnan(y)
                    
                    if np.any(valid_mask):
                        x_valid = x[valid_mask]
                        y_valid = y[valid_mask]
                        yerr_valid = yerr[valid_mask]
                        
                        ax.plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1}', color=color)
                        
                        if show_95:
                            groups = []
                            current_group = []
                            for i, idx in enumerate(np.where(valid_mask)[0]):
                                if i == 0 or idx == np.where(valid_mask)[0][i-1] + 1:
                                    current_group.append(i)
                                else:
                                    if current_group:
                                        groups.append(current_group)
                                    current_group = [i]
                            if current_group:
                                groups.append(current_group)
                            
                            for group in groups:
                                if len(group) < 2:
                                    continue
                                    
                                xg = x_valid[group]
                                yg = y_valid[group]
                                yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                                x_interp = np.linspace(xg[0], xg[-1], 100)
                                y_interp = np.interp(x_interp, xg, yg)
                                upper_bound = np.interp(x_interp, xg, yg + yerrg)
                                lower_bound = np.interp(x_interp, xg, yg - yerrg)
                                
                                ax.fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
                                lower_bound = np.interp(x_interp, xg, yg - yerrg)
                                
                                ax.fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
                
                ax.set_title('Peak Width')
                ax.set_ylabel('Width (cm⁻¹)')
                plotted = True
                
            elif which == 'eta':
                if model_type in ["Pseudo-Voigt", "Asymmetric Voigt"]:
                    y = trends[peak_idx]['eta']
                    yerr = uncertainties[peak_idx]['eta']
                    valid_mask = ~np.isnan(y)
                    
                    if np.any(valid_mask):
                        x_valid = x[valid_mask]
                        y_valid = y[valid_mask]
                        yerr_valid = yerr[valid_mask]
                        
                        ax.plot(x_valid, y_valid, 'o', label=f'Peak {peak_idx+1}', color=color)
                        
                        if show_95:
                            groups = []
                            current_group = []
                            for i, idx in enumerate(np.where(valid_mask)[0]):
                                if i == 0 or idx == np.where(valid_mask)[0][i-1] + 1:
                                    current_group.append(i)
                                else:
                                    if current_group:
                                        groups.append(current_group)
                                    current_group = [i]
                            if current_group:
                                groups.append(current_group)
                            
                            for group in groups:
                                if len(group) < 2:
                                    continue
                                
                                xg = x_valid[group]
                                yg = y_valid[group]
                                yerrg = [e for i, e in enumerate(yerr_valid) if i in group and e is not None]
                                
                                if len(yerrg) != len(xg):
                                    continue
                                
                                x_interp = np.linspace(xg[0], xg[-1], 100)
                                y_interp = np.interp(x_interp, xg, yg)
                                upper_bound = np.interp(x_interp, xg, yg + yerrg)
                                lower_bound = np.interp(x_interp, xg, yg - yerrg)
                                
                                ax.fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
                    
                    ax.set_title('Peak Eta')
                    ax.set_ylabel('Eta')
                    plotted = True
                else:
                    ax.text(0.5, 0.5, 'Eta not available for current model', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Peak Eta')
                    ax.set_ylabel('Eta')
        
        ax.set_xlabel('Spectrum Number')
        if show_grid:
            ax.grid(True, linestyle=':', color='gray', alpha=0.6)
        else:
            ax.grid(False)
        ax.legend()

    def preview_trends_subplot(self, which):
        """Preview a single trends subplot (position, amplitude, width, eta) in a new window."""
        import matplotlib.pyplot as plt
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        preview_win = tk.Toplevel(self.window)
        preview_win.title(f"Preview {which.capitalize()} Plot")
        preview_win.geometry("700x500")  # Set a reasonable size for the preview window
        
        fig, ax = plt.subplots(figsize=(7, 5))
        # Call a helper to plot the selected subplot with current options
        self._plot_single_trends_subplot(ax, which)
        canvas = FigureCanvasTkAgg(fig, master=preview_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(preview_win)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def export_results(self):
        """Export batch processing results to CSV."""
        if not self.batch_results:
            messagebox.showwarning("No Results", "No batch processing results to export.", parent=self.window)
            return
            
        try:
            # Check if there are any manually skipped files
            skipped_count = sum(1 for result in self.batch_results if result.get('manually_skipped', False))
            
            include_skipped = True
            if skipped_count > 0:
                response = messagebox.askyesnocancel(
                    "Export Options",
                    f"Found {skipped_count} manually skipped file(s).\n\n"
                    "Include skipped files in export?\n\n"
                    "• Yes: Include all files (skipped files marked as failed)\n"
                    "• No: Exclude skipped files from export\n"
                    "• Cancel: Cancel export",
                    parent=self.window
                )
                if response is None:  # Cancel
                    return
                include_skipped = response
            
            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                parent=self.window,
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Save Batch Results"
            )
            
            if not file_path:
                return
                
            # Prepare data for export
            data = []
            for result in self.batch_results:
                # Skip manually skipped files if user chose not to include them
                if not include_skipped and result.get('manually_skipped', False):
                    continue
                    
                # Start with basic file info
                row = {'File': os.path.basename(result['file'])}
                
                # Check if this was a failed fit or manually skipped
                fit_failed = result.get('fit_failed', False)
                manually_skipped = result.get('manually_skipped', False)
                
                if manually_skipped:
                    row['Fit_Success'] = 'Skipped'
                else:
                    row['Fit_Success'] = 'No' if fit_failed else 'Yes'
                
                if not fit_failed and result['fit_params'] is not None:
                    # Get model type
                    model_type = self.current_model.get()
                    if model_type == "Gaussian" or model_type == "Lorentzian":
                        params_per_peak = 3
                    elif model_type == "Pseudo-Voigt":
                        params_per_peak = 4
                    elif model_type == "Asymmetric Voigt":
                        params_per_peak = 5
                    else:
                        params_per_peak = 3
                    
                    # Extract parameters for each peak
                    n_peaks = len(result['fit_params']) // params_per_peak
                    for i in range(n_peaks):
                        start_idx = i * params_per_peak
                        row[f'Peak_{i+1}_Position'] = result['fit_params'][start_idx+1]
                        row[f'Peak_{i+1}_Amplitude'] = result['fit_params'][start_idx]
                        row[f'Peak_{i+1}_Width'] = result['fit_params'][start_idx+2]
                        
                        if model_type == "Pseudo-Voigt":
                            row[f'Peak_{i+1}_Eta'] = result['fit_params'][start_idx+3]
                        elif model_type == "Asymmetric Voigt":
                            row[f'Peak_{i+1}_Width_Left'] = result['fit_params'][start_idx+2]
                            row[f'Peak_{i+1}_Width_Right'] = result['fit_params'][start_idx+3]
                            row[f'Peak_{i+1}_Eta'] = result['fit_params'][start_idx+4]
                else:
                    # For failed fits, add NaN placeholders for all peak parameters
                    model_type = self.current_model.get()
                    # Estimate number of peaks from the first successful fit in batch results
                    n_peaks = 0
                    for res in self.batch_results:
                        if not res.get('fit_failed', False) and res['fit_params'] is not None:
                            if model_type == "Gaussian" or model_type == "Lorentzian":
                                params_per_peak = 3
                            elif model_type == "Pseudo-Voigt":
                                params_per_peak = 4
                            elif model_type == "Asymmetric Voigt":
                                params_per_peak = 5
                            else:
                                params_per_peak = 3
                            n_peaks = len(res['fit_params']) // params_per_peak
                            break
                    
                    # Add placeholders for all peak parameters
                    for i in range(n_peaks):
                        row[f'Peak_{i+1}_Position'] = float('nan')
                        row[f'Peak_{i+1}_Amplitude'] = float('nan')
                        row[f'Peak_{i+1}_Width'] = float('nan')
                        
                        if model_type == "Pseudo-Voigt":
                            row[f'Peak_{i+1}_Eta'] = float('nan')
                        elif model_type == "Asymmetric Voigt":
                            row[f'Peak_{i+1}_Width_Left'] = float('nan')
                            row[f'Peak_{i+1}_Width_Right'] = float('nan')
                            row[f'Peak_{i+1}_Eta'] = float('nan')
                
                data.append(row)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Results saved to {file_path}", parent=self.window)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}", parent=self.window)
    
    def on_closing(self):
        """Handle window closing event."""
        self.window.destroy()
    
    def subtract_background(self):
        """Subtract background from the current spectrum using asymmetric least squares."""
        try:
            # Get parameters
            lambda_val = float(self.var_lambda.get())
            p_val = float(self.var_p.get())
            
            # Calculate background
            L = len(self.spectra)
            D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
            w = np.ones(L)
            
            for i in range(10):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lambda_val * D.dot(D.transpose())
                z = spsolve(Z, w*self.spectra)
                w = p_val * (self.spectra > z) + (1-p_val) * (self.spectra < z)
            
            self.background = z
            self.spectra = self.original_spectra - self.background
            
            # Update plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to subtract background: {str(e)}", parent=self.window)
    
    def find_peaks(self):
        """Find peaks in the current spectrum."""
        try:
            # Get parameters from sliders
            height_val = self.var_height.get()
            distance_val = self.var_distance.get()
            prominence_val = self.var_prominence.get()
            
            # Convert slider values (0 = Auto, >0 = actual value)
            height = None if height_val == 0 else height_val
            distance = None if distance_val == 0 else int(distance_val)
            prominence = None if prominence_val == 0 else prominence_val
            
            # Get a copy of the spectrum data to avoid modifying the original
            spectra_data = np.copy(self.spectra)
            
            # Ensure we have valid data
            if spectra_data is None or len(spectra_data) == 0:
                if hasattr(self, 'window'):  # Avoid error if called during auto-detect
                    messagebox.showerror("Error", "No spectrum data available", parent=self.window)
                return
                
            # Import scipy find_peaks function safely inside the try block
            try:
                from scipy.signal import find_peaks
                peak_indices, properties = find_peaks(
                    spectra_data, 
                    height=height, 
                    distance=distance,
                    prominence=prominence
                )
            except Exception as e:
                if hasattr(self, 'window'):  # Avoid error if called during auto-detect
                    messagebox.showerror("Error", f"Failed to execute scipy.signal.find_peaks: {str(e)}", parent=self.window)
                return
            
            # Store peak positions and intensities
            self.peaks = []
            for idx in peak_indices:
                idx_int = int(idx)  # Convert to int to avoid array indexing issues
                self.peaks.append({
                    'position': float(self.wavenumbers[idx_int]),
                    'intensity': float(spectra_data[idx_int]),
                    'index': idx_int
                })
            
            # Update plot
            self.update_plot()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in find_peaks: {error_details}")
            if hasattr(self, 'window'):  # Avoid error if called during auto-detect
                messagebox.showerror("Error", f"Failed to find peaks: {str(e)}", parent=self.window)

    def on_height_change(self, value):
        """Callback for height slider changes."""
        val = float(value)
        if val == 0:
            self.height_value_label.config(text="Auto")
        else:
            self.height_value_label.config(text=f"{val:.1f}")
        
        # Auto-detect peaks if enabled with throttling
        if hasattr(self, 'auto_detect_var') and self.auto_detect_var.get() and hasattr(self, 'spectra') and self.spectra is not None:
            # Cancel any pending auto-detection
            if hasattr(self, '_height_after_id'):
                self.window.after_cancel(self._height_after_id)
            # Schedule new auto-detection with delay
            self._height_after_id = self.window.after(200, self.find_peaks)

    def on_distance_change(self, value):
        """Callback for distance slider changes."""
        val = float(value)
        if val == 0:
            self.distance_value_label.config(text="Auto")
        else:
            self.distance_value_label.config(text=f"{int(val)}")
        
        # Auto-detect peaks if enabled
        if hasattr(self, 'auto_detect_var') and self.auto_detect_var.get() and hasattr(self, 'spectra') and self.spectra is not None:
            self.find_peaks()

    def on_prominence_change(self, value):
        """Callback for prominence slider changes."""
        val = float(value)
        if val == 0:
            self.prominence_value_label.config(text="Auto")
        else:
            self.prominence_value_label.config(text=f"{val:.1f}")
        
        # Auto-detect peaks if enabled
        if hasattr(self, 'auto_detect_var') and self.auto_detect_var.get() and hasattr(self, 'spectra') and self.spectra is not None:
            self.find_peaks()

    def reset_peak_params(self):
        """Reset all peak detection parameters to Auto."""
        self.var_height.set(0)
        self.var_distance.set(0)
        self.var_prominence.set(0)
        self.height_value_label.config(text="Auto")
        self.distance_value_label.config(text="Auto")
        self.prominence_value_label.config(text="Auto")
        
        # Auto-detect peaks if enabled with throttling
        if hasattr(self, 'auto_detect_var') and self.auto_detect_var.get() and hasattr(self, 'spectra') and self.spectra is not None:
            # Cancel any pending auto-detection
            for attr in ['_height_after_id', '_distance_after_id', '_prominence_after_id']:
                if hasattr(self, attr):
                    self.window.after_cancel(getattr(self, attr))
            # Schedule new auto-detection with delay
            self.window.after(200, self.find_peaks)
    
    def update_slider_ranges(self):
        """Update slider ranges based on current spectrum data."""
        if not hasattr(self, 'spectra') or self.spectra is None:
            return
        
        try:
            # Update height slider range based on spectrum intensity
            min_intensity = np.min(self.spectra)
            max_intensity = np.max(self.spectra)
            intensity_range = max_intensity - min_intensity
            
            # Set height range from 0 to max intensity with reasonable precision
            self.height_slider.config(to=max_intensity)
            
            # Update prominence slider range (typically smaller than height)
            self.prominence_slider.config(to=intensity_range * 0.5)
            
            # Distance slider range depends on spectrum length
            spectrum_length = len(self.spectra)
            self.distance_slider.config(to=min(50, spectrum_length // 10))
            
        except Exception as e:
            print(f"Error updating slider ranges: {e}")
            # Use default ranges if calculation fails
            self.height_slider.config(to=100)
            self.prominence_slider.config(to=50)
            self.distance_slider.config(to=50)
    
    def gaussian(self, x, a, x0, sigma):
        """Gaussian peak function."""
        # Use numpy operations for safety
        x_array = np.asarray(x)
        diff = np.subtract(x_array, x0)
        exponent = np.divide(np.negative(np.square(diff)), 2.0 * np.square(sigma))
        return a * np.exp(exponent)
    
    def lorentzian(self, x, a, x0, wid):
        """Lorentzian peak function."""
        # Use numpy operations for safety
        x_array = np.asarray(x)
        diff = np.subtract(x_array, x0)
        denom = np.add(np.square(diff), np.square(wid))
        return np.multiply(a, np.divide(np.square(wid), denom))
    
    def pseudo_voigt(self, x, a, x0, sigma, eta):
        """Pseudo-Voigt peak function."""
        # Safely compute as a weighted sum of Gaussian and Lorentzian components
        gaussian_part = self.gaussian(x, a, x0, sigma)
        lorentzian_part = self.lorentzian(x, a, x0, sigma)
        return np.add(np.multiply(eta, lorentzian_part), np.multiply(np.subtract(1.0, eta), gaussian_part))
    
    def asymmetric_voigt(self, x, amp, cen, wid_left, wid_right, eta=0.5):
        """
        Asymmetric Voigt function combining Gaussian and Lorentzian components
        with different widths on each side of the peak.
        
        Parameters:
        -----------
        x : array-like
            Wavenumber values
        amp : float
            Peak amplitude
        cen : float
            Peak center position
        wid_left : float
            Width parameter for the left side of the peak
        wid_right : float
            Width parameter for the right side of the peak
        eta : float, optional
            Mixing parameter between Gaussian and Lorentzian (0-1)
            Default is 0.5 (equal mix)
        
        Returns:
        --------
        array-like
            The asymmetric Voigt function evaluated at x
        """
        # Ensure we're working with numpy arrays
        x_array = np.asarray(x)
        cen_val = float(cen)
        
        # Initialize result array
        result = np.zeros_like(x_array)
        
        # Calculate the mask for left and right sides using numpy functions
        left_indices = np.nonzero(np.less_equal(x_array, cen_val))[0]
        right_indices = np.nonzero(np.greater(x_array, cen_val))[0]
        
        # Process left side
        if left_indices.size > 0:
            x_left = x_array[left_indices]
            g_left = np.exp(-np.square((x_left - cen_val) / wid_left))
            l_left = 1.0 / (1.0 + np.square((x_left - cen_val) / wid_left))
            result[left_indices] = amp * (eta * g_left + (1.0 - eta) * l_left)
        
        # Process right side
        if right_indices.size > 0:
            x_right = x_array[right_indices]
            g_right = np.exp(-np.square((x_right - cen_val) / wid_right))
            l_right = 1.0 / (1.0 + np.square((x_right - cen_val) / wid_right))
            result[right_indices] = amp * (eta * g_right + (1.0 - eta) * l_right)
        
        return result
    
    def fit_peaks(self):
        """Fit peaks to the current spectrum using the selected model."""
        try:
            if not self.peaks:
                messagebox.showwarning("No Peaks", "Please detect peaks first.", parent=self.window)
                return
            
            # Get model type
            model_type = self.current_model.get()
            
            # Select model function
            if model_type == "Gaussian":
                model_func = self.gaussian
                params_per_peak = 3
            elif model_type == "Lorentzian":
                model_func = self.lorentzian
                params_per_peak = 3
            elif model_type == "Pseudo-Voigt":
                model_func = self.pseudo_voigt
                params_per_peak = 4
            elif model_type == "Asymmetric Voigt":
                model_func = self.asymmetric_voigt
                params_per_peak = 5
            else:
                model_func = self.gaussian
                params_per_peak = 3

            # Parse fit ranges
            fit_ranges_str = self.var_fit_ranges.get().strip()
            roi_ranges = []
            if fit_ranges_str:
                try:
                    for part in fit_ranges_str.split(','):
                        if '-' in part:
                            min_w, max_w = map(float, part.split('-'))
                            roi_ranges.append((min_w, max_w))
                except Exception as e:
                    messagebox.showerror("Fit Range Error", f"Could not parse fit ranges: {fit_ranges_str}\nError: {e}", parent=self.window)
                    self.fit_failed = True
                    return
            
            # Filter peaks to only include those within ROI ranges
            peaks_in_roi = []
            for peak in self.peaks:
                peak_pos = peak['position']
                # Check if roi_ranges is empty using len() instead of direct boolean evaluation
                if len(roi_ranges) == 0:  # If no ROI specified, include all peaks
                    peaks_in_roi.append(peak)
                else:
                    for min_w, max_w in roi_ranges:
                        # Use explicit comparisons to avoid potential array logic issues
                        is_in_range = (min_w <= peak_pos) and (peak_pos <= max_w)
                        if is_in_range:
                            peaks_in_roi.append(peak)
                            break
            
            if not peaks_in_roi:
                messagebox.showwarning("No Peaks in ROI", "No peaks found within the specified ROI ranges.", parent=self.window)
                self.fit_failed = True
                return
            
            # Show constraint information
            constraint_info = self.get_constraint_info_text()
            print(f"Fitting {len(peaks_in_roi)} peaks with {model_type} model - {constraint_info}")
            
            # HARD CAP for Asymmetric Voigt
            if model_type == "Asymmetric Voigt" and len(peaks_in_roi) > 8:
                messagebox.showerror("Too Many Peaks", "Asymmetric Voigt fitting is limited to 8 peaks for stability. Please reduce the number of detected peaks (e.g., by increasing the prominence or height threshold).", parent=self.window)
                self.fit_failed = True
                return

            # Create mask for ROI ranges
            mask = np.zeros_like(self.wavenumbers, dtype=bool)
            if roi_ranges:
                for min_w, max_w in roi_ranges:
                    # Use numpy logical operations to avoid truth value ambiguity
                    new_mask = np.logical_and(
                        np.greater_equal(self.wavenumbers, min_w),
                        np.less_equal(self.wavenumbers, max_w)
                    )
                    mask = np.logical_or(mask, new_mask)
                x_fit = self.wavenumbers[mask]
                y_fit = self.spectra[mask]
            else:
                x_fit = self.wavenumbers
                y_fit = self.spectra
            
            # Prepare initial parameters and constraints
            initial_params = []
            bounds_lower = []
            bounds_upper = []
            
            # Track which parameters are fixed
            fixed_params = {}  # index -> fixed_value
            param_mapping = []  # Maps optimization parameter index to full parameter index
            full_param_count = 0
            
            for peak_idx, peak in enumerate(peaks_in_roi):
                base_idx = peak_idx * params_per_peak
                
                # Amplitude (never fixed)
                initial_params.append(peak['intensity'])
                bounds_lower.append(0)
                bounds_upper.append(np.inf)
                param_mapping.append(base_idx)  # amplitude at position 0 of each peak
                
                # Position (may be fixed)
                if self.fix_positions.get():
                    fixed_params[base_idx + 1] = peak['position']
                else:
                    initial_params.append(peak['position'])
                    bounds_lower.append(self.wavenumbers[0])
                    bounds_upper.append(self.wavenumbers[-1])
                    param_mapping.append(base_idx + 1)
                
                # Width parameters (may be fixed)
                if model_type == "Gaussian" or model_type == "Lorentzian":
                    if self.fix_widths.get():
                        fixed_params[base_idx + 2] = 10.0  # Default fixed width
                    else:
                        initial_params.append(10)  # Default width
                        bounds_lower.append(0.1)
                        bounds_upper.append(100)
                        param_mapping.append(base_idx + 2)
                        
                elif model_type == "Pseudo-Voigt":
                    if self.fix_widths.get():
                        fixed_params[base_idx + 2] = 10.0  # Default fixed width
                    else:
                        initial_params.append(10)  # Default width
                        bounds_lower.append(0.1)
                        bounds_upper.append(100)
                        param_mapping.append(base_idx + 2)
                    
                    # Eta parameter (never fixed for now)
                    initial_params.append(0.5)  # Default eta
                    bounds_lower.append(0)
                    bounds_upper.append(1)
                    param_mapping.append(base_idx + 3)
                    
                elif model_type == "Asymmetric Voigt":
                    if self.fix_widths.get():
                        fixed_params[base_idx + 2] = 10.0  # Default fixed left width
                        fixed_params[base_idx + 3] = 10.0  # Default fixed right width
                    else:
                        initial_params.append(10)  # Default left width
                        initial_params.append(10)  # Default right width
                        bounds_lower.extend([0.1, 0.1])
                        bounds_upper.extend([100, 100])
                        param_mapping.extend([base_idx + 2, base_idx + 3])
                    
                    # Eta parameter (never fixed for now)
                    initial_params.append(0.5)  # Default eta
                    bounds_lower.append(0)
                    bounds_upper.append(1)
                    param_mapping.append(base_idx + 4)
                    
                full_param_count += params_per_peak

            # Define combined model function that handles fixed parameters
            def combined_model_with_constraints(x, *opt_params):
                """Model function that reconstructs full parameter set including fixed values."""
                # Reconstruct full parameter array
                full_params = [0.0] * full_param_count
                
                # Fill in optimized parameters
                for opt_idx, full_idx in enumerate(param_mapping):
                    full_params[full_idx] = opt_params[opt_idx]
                
                # Fill in fixed parameters
                for fixed_idx, fixed_val in fixed_params.items():
                    full_params[fixed_idx] = fixed_val
                
                # Calculate model using full parameter set
                result = np.zeros_like(x)
                for i in range(0, len(full_params), params_per_peak):
                    if i + params_per_peak <= len(full_params):
                        peak_params = full_params[i:i+params_per_peak]
                        try:
                            peak_result = model_func(x, *peak_params)
                            result = result + peak_result
                        except Exception as e:
                            print(f"Error in model_func at i={i} with params {peak_params}: {e}")
                            continue
                return result
            
            # Add a counter for function evaluations (NLLS cycles)
            self.func_eval_count = 0
            
            # Wrap the model function to count evaluations
            def counting_model(x, *params):
                self.func_eval_count += 1
                return combined_model_with_constraints(x, *params)
            
            # Perform fit
            try:
                # Reset counter for each new fit
                self.func_eval_count = 0
                
                popt, pcov = curve_fit(counting_model, x_fit, y_fit,
                                     p0=initial_params,
                                     bounds=(bounds_lower, bounds_upper),
                                     maxfev=2000,
                                     full_output=False)
                                     
                # Store the number of function evaluations
                self.nlls_cycles = self.func_eval_count
                
            except Exception as e1:
                print(f"First fitting attempt failed: {str(e1)}")
                # Retry with higher maxfev
                try:
                    # Reset counter for retry
                    self.func_eval_count = 0
                    
                    popt, pcov = curve_fit(counting_model, x_fit, y_fit,
                                         p0=initial_params,
                                         bounds=(bounds_lower, bounds_upper),
                                         maxfev=20000,
                                         full_output=False)
                                         
                    # Store the number of function evaluations for the retry
                    self.nlls_cycles = self.func_eval_count
                    
                except Exception as e2:
                    print(f"Second fitting attempt failed: {str(e2)}")
                    # Fallback: use initial guess, no covariance, flag as failed
                    popt = np.array(initial_params)
                    pcov = None
                    
                    # Reconstruct full parameters for storage
                    full_popt = [0.0] * full_param_count
                    for opt_idx, full_idx in enumerate(param_mapping):
                        if opt_idx < len(popt):
                            full_popt[full_idx] = popt[opt_idx]
                    for fixed_idx, fixed_val in fixed_params.items():
                        full_popt[fixed_idx] = fixed_val
                    
                    self.fit_params = np.array(full_popt)
                    self.fit_cov = pcov
                    self.nlls_cycles = 0  # No successful cycles
                    
                    # Try to calculate fit result and residuals with initial parameters
                    try:
                        self.fit_result = combined_model_with_constraints(self.wavenumbers, *popt)
                        self.residuals = self.spectra - self.fit_result
                    except Exception as e3:
                        print(f"Failed to calculate fit with initial parameters: {str(e3)}")
                        # If even that fails, set empty arrays
                        self.fit_result = np.zeros_like(self.wavenumbers)
                        self.residuals = np.copy(self.spectra)
                    
                    self.fit_failed = True
                    messagebox.showwarning("Fit Failed", f"Failed to fit peaks after two attempts. Using initial guess for this spectrum.\n\nFirst error: {str(e1)}\nSecond error: {str(e2)}", parent=self.window)
                    self.update_plot()
                    self.update_peak_visibility_controls()
                    return
            
            # Reconstruct full parameter set from optimized parameters
            full_popt = [0.0] * full_param_count
            for opt_idx, full_idx in enumerate(param_mapping):
                full_popt[full_idx] = popt[opt_idx]
            for fixed_idx, fixed_val in fixed_params.items():
                full_popt[fixed_idx] = fixed_val
            
            # Create full initial parameters for validation
            full_initial_params = [0.0] * full_param_count
            opt_idx = 0
            for peak_idx, peak in enumerate(peaks_in_roi):
                base_idx = peak_idx * params_per_peak
                full_initial_params[base_idx] = peak['intensity']  # amplitude
                full_initial_params[base_idx + 1] = peak['position']  # position
                
                if model_type in ["Gaussian", "Lorentzian"]:
                    full_initial_params[base_idx + 2] = 10.0  # width
                elif model_type == "Pseudo-Voigt":
                    full_initial_params[base_idx + 2] = 10.0  # width
                    full_initial_params[base_idx + 3] = 0.5   # eta
                elif model_type == "Asymmetric Voigt":
                    full_initial_params[base_idx + 2] = 10.0  # left width
                    full_initial_params[base_idx + 3] = 10.0  # right width
                    full_initial_params[base_idx + 4] = 0.5   # eta
                
            # Validate fit parameters for realism before accepting the fit
            if not self.validate_fit_parameters(np.array(full_popt), np.array(full_initial_params), model_type, peaks_in_roi):
                print("Fit validation failed - parameters are unrealistic")
                self.fit_failed = True
                self.fit_params = np.array(full_initial_params)
                self.fit_cov = None
                self.nlls_cycles = self.func_eval_count
                
                # Calculate fit result with initial parameters for display
                try:
                    # Convert initial parameters to optimized parameter format
                    initial_opt_params = []
                    for full_idx in param_mapping:
                        initial_opt_params.append(full_initial_params[full_idx])
                    self.fit_result = combined_model_with_constraints(self.wavenumbers, *initial_opt_params)
                    if hasattr(self, 'original_spectra') and hasattr(self, 'background') and self.background is not None:
                        background_subtracted = np.subtract(self.original_spectra, self.background)
                        self.residuals = np.subtract(background_subtracted, self.fit_result)
                    else:
                        self.residuals = np.subtract(self.spectra, self.fit_result)
                except Exception:
                    self.fit_result = np.zeros_like(self.wavenumbers)
                    self.residuals = np.copy(self.spectra)
                
                self.update_plot()
                self.update_peak_visibility_controls()
                return
            
            # Store results using full parameter set
            self.fit_params = np.array(full_popt)
            self.fit_cov = pcov
            
            # Calculate fit result and residuals
            try:
                self.fit_result = combined_model_with_constraints(self.wavenumbers, *popt)
                
                # Calculate residuals (difference between background-subtracted data and fit)
                # Use background-subtracted spectrum for residuals calculation
                if hasattr(self, 'original_spectra') and hasattr(self, 'background') and self.background is not None:
                    # Calculate background-subtracted spectrum directly from original data
                    background_subtracted = np.subtract(self.original_spectra, self.background)
                    # Calculate residuals from the subtracted spectrum
                    self.residuals = np.subtract(background_subtracted, self.fit_result)
                else:
                    # Fallback if no background available
                    self.residuals = np.subtract(self.spectra, self.fit_result)
                
                # Calculate and store R² values for each peak
                self.peak_r_squared = self.calculate_peak_r_squared(self.spectra, self.fit_result, self.fit_params, model_type)
                
                self.fit_failed = False
                
                # Check if this spectrum is part of batch results and update it
                self.update_batch_results_if_present()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to calculate fit results: {str(e)}", parent=self.window)
                self.fit_failed = True
                return
            
            # Update plot
            self.update_plot()
            self.update_peak_visibility_controls()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fit peaks: {str(e)}", parent=self.window)
            self.fit_failed = True
    
    def update_plot(self):
        """Update the current spectrum plot."""
        # Safety check - do we have required data?
        if not hasattr(self, 'wavenumbers') or not hasattr(self, 'spectra'):
            return
            
        # Clear plots
        self.ax1_current.clear()
        self.ax2_current.clear()
        
        # Set title to current filename
        if hasattr(self, 'current_spectrum_index') and hasattr(self, 'spectra_files'):
            if self.current_spectrum_index >= 0 and self.current_spectrum_index < len(self.spectra_files):
                filename = os.path.basename(self.spectra_files[self.current_spectrum_index])
                self.ax1_current.set_title(f"Current Spectrum: {filename}")
        
        # Plot original spectrum
        if hasattr(self, 'original_spectra'):
            self.ax1_current.plot(self.wavenumbers, self.original_spectra, 'k-', label='Original Spectrum')
        else:
            self.ax1_current.plot(self.wavenumbers, self.spectra, 'k-', label='Spectrum')
            
        # Plot background if available
        if hasattr(self, 'background') and self.background is not None:
            self.ax1_current.plot(self.wavenumbers, self.background, 'b--', label='Background')
            
            # Also plot the background-subtracted data
            if hasattr(self, 'original_spectra'):
                background_subtracted = self.original_spectra - self.background
                self.ax1_current.plot(self.wavenumbers, background_subtracted, 'g-', alpha=0.7, label='Subtracted Spectrum')
        
        # Plot fitted peaks if available
        has_fit_result = (hasattr(self, 'fit_result') and 
                          self.fit_result is not None and 
                          isinstance(self.fit_result, np.ndarray) and 
                          len(self.fit_result) > 0)
                          
        if has_fit_result:
            wavenumbers = np.asarray(self.wavenumbers)
            fit_result = np.asarray(self.fit_result)
            self.ax1_current.plot(wavenumbers, fit_result, 'r-', label='Fit')
            
            # Plot individual peaks if requested
            show_individual = hasattr(self, 'show_individual_peaks') and self.show_individual_peaks.get()
            if show_individual:
                model_type = self.current_model.get()
                if model_type == "Gaussian" or model_type == "Lorentzian":
                    params_per_peak = 3
                elif model_type == "Pseudo-Voigt":
                    params_per_peak = 4
                elif model_type == "Asymmetric Voigt":
                    params_per_peak = 5
                else:
                    params_per_peak = 3
                
                # Calculate R² values for each peak only if we have fit_params and peaks
                has_fit_params = (hasattr(self, 'fit_params') and 
                                 self.fit_params is not None and 
                                 hasattr(self, 'peaks') and 
                                 isinstance(self.peaks, list) and len(self.peaks) > 0 and
                                 isinstance(self.fit_params, (list, np.ndarray)) and 
                                 len(self.fit_params) > 0)
                                
                if has_fit_params:
                    # Safely cast to numpy array if not already
                    fit_params = np.asarray(self.fit_params)
                    
                    # Get stored R² values or calculate new ones if not available
                    if hasattr(self, 'peak_r_squared') and self.peak_r_squared is not None:
                        peak_r_squared = self.peak_r_squared
                    else:
                        # Calculate R² values if not already stored
                        peak_r_squared = self.calculate_peak_r_squared(self.spectra, self.fit_result, fit_params, model_type)
                    
                    n_peaks = len(fit_params) // params_per_peak
                    for peak_idx in range(n_peaks):
                        i = peak_idx * params_per_peak
                        if i + params_per_peak <= len(fit_params):  # Safety check
                            peak_params = fit_params[i:i+params_per_peak]
                            
                            try:
                                if model_type == "Gaussian":
                                    peak = self.gaussian(self.wavenumbers, *peak_params)
                                elif model_type == "Lorentzian":
                                    peak = self.lorentzian(self.wavenumbers, *peak_params)
                                elif model_type == "Pseudo-Voigt":
                                    peak = self.pseudo_voigt(self.wavenumbers, *peak_params)
                                elif model_type == "Asymmetric Voigt":
                                    peak = self.asymmetric_voigt(self.wavenumbers, *peak_params)
                                else:
                                    peak = self.gaussian(self.wavenumbers, *peak_params)
                                
                                # Get R² value for this peak
                                if peak_idx < len(peak_r_squared):  # Safety check
                                    r2 = peak_r_squared[peak_idx]
                                    
                                    # Plot peak with label including R² value
                                    label = f'Peak {peak_idx+1} (R²={r2:.3f})'
                                    color = 'red' if r2 < 0.95 else 'g'
                                    self.ax1_current.plot(self.wavenumbers, peak, '--', color=color, alpha=0.5, label=label)
                            except Exception as e:
                                print(f"Error plotting peak {peak_idx}: {str(e)}")
        
        # Plot peak positions if available
        has_peaks = hasattr(self, 'peaks') and isinstance(self.peaks, list) and len(self.peaks) > 0
        if has_peaks:
            try:
                peak_positions = [peak['position'] for peak in self.peaks]
                peak_intensities = [peak['intensity'] for peak in self.peaks]
                self.ax1_current.plot(peak_positions, peak_intensities, 'ro', label='Peaks')
            except Exception as e:
                print(f"Error plotting peak positions: {str(e)}")
                
        # The rest of the plotting code remains mostly unchanged, but we'll add safety checks
        
        # Plot residuals if available
        has_residuals = (hasattr(self, 'residuals') and 
                        self.residuals is not None and 
                        isinstance(self.residuals, np.ndarray) and 
                        len(self.residuals) > 0)
                        
        if has_residuals:
            try:
                # Ensure we have valid arrays
                residuals_array = np.asarray(self.residuals)
                wavenumbers_array = np.asarray(self.wavenumbers)
                
                # Plot the residuals
                self.ax2_current.plot(wavenumbers_array, residuals_array, 'k-')
                self.ax2_current.axhline(y=0, color='r', linestyle='--')
                
                # Add colored fill for residuals
                positive_mask = np.greater(residuals_array, 0)
                negative_mask = np.less(residuals_array, 0)
                
                if np.any(positive_mask):
                    self.ax2_current.fill_between(
                        wavenumbers_array, 
                        residuals_array, 
                        0, 
                        where=positive_mask, 
                        color='red', 
                        alpha=0.3, 
                        interpolate=True
                    )
                
                if np.any(negative_mask):
                    self.ax2_current.fill_between(
                        wavenumbers_array, 
                        residuals_array, 
                        0, 
                        where=negative_mask, 
                        color='blue', 
                        alpha=0.3, 
                        interpolate=True
                    )
            except Exception as e:
                print(f"Error plotting residuals: {str(e)}")
        
        # Set labels and grid
        self.ax1_current.set_ylabel('Intensity')
        self.ax2_current.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax2_current.set_ylabel('Residuals')
        self.ax2_current.set_title('Residuals (Subtracted Spectrum - Fit)', fontsize=10)
        self.ax1_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        self.ax2_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        
        # Plot fit ranges (ROI) as grey regions if specified
        try:
            fit_ranges_str = self.var_fit_ranges.get().strip() if hasattr(self, 'var_fit_ranges') else ''
            roi_ranges = []
            if fit_ranges_str:
                for part in fit_ranges_str.split(','):
                    if '-' in part:
                        min_w, max_w = map(float, part.split('-'))
                        roi_ranges.append((min_w, max_w))
            
            # Plot ROI ranges as grey regions
            for min_w, max_w in roi_ranges:
                self.ax1_current.axvspan(min_w, max_w, color='lightgrey', alpha=0.3, zorder=0)
        except Exception as e:
            print(f"Error plotting ROI regions: {str(e)}")
            
        # Add legend and draw
        try:
            self.ax1_current.legend()
            self.canvas_current.draw()
        except Exception as e:
            print(f"Error finalizing plot: {str(e)}")
    
    def clear_plot(self):
        """Clear the current spectrum plot."""
        self.ax1_current.clear()
        self.ax2_current.clear()
        self.ax1_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        self.ax2_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        self.ax1_current.set_ylabel('Intensity')
        self.ax2_current.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax2_current.set_ylabel('Residuals')
        self.current_spectrum_label.config(text="No spectrum loaded")
        self.canvas_current.draw()
    
    def clear_peaks(self):
        """Clear detected peaks and fitted peak profiles for the current spectrum."""
        self.peaks = []
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        self.update_plot()

    def stop_batch(self):
        """Set a flag to stop the batch process."""
        self._stop_batch = True
        if hasattr(self, 'batch_status_text'):
            try:
                self.batch_status_text.config(state=tk.NORMAL)
                self.batch_status_text.insert(tk.END, "Stopping batch process (after current spectrum)...\n")
                self.batch_status_text.see(tk.END)
                self.batch_status_text.config(state=tk.DISABLED)
                self.window.update()
            except:
                pass
    
    def on_closing(self):
        """Handle window closing event."""
        self.window.destroy()
    
    def update_roi_regions(self):
        """Refresh the plot to show the current fit regions as shaded areas."""
        try:
            # Update the plot safely, avoiding any direct array comparisons
            self.update_plot()
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in update_roi_regions: {error_details}")
            messagebox.showerror("Error", f"Failed to update ROI regions: {str(e)}", parent=self.window)

    def enable_peak_addition(self):
        """Enable/disable mode to add peaks by clicking on the plot."""
        try:
            # Toggle the adding peak mode
            self.adding_peak = not getattr(self, 'adding_peak', False)
            
            if self.adding_peak:
                # Enable manual peak mode
                # Initialize peaks list if needed
                if not hasattr(self, 'peaks'):
                    self.peaks = []
                    
                # Change cursor to indicate interactive mode
                self.canvas_current.get_tk_widget().config(cursor="crosshair")
                
                # Update the plot title to show instructions
                self.ax1_current.set_title("MANUAL MODE: Click on peaks to add them • Click the red button or press ESC to finish")
                self.canvas_current.draw()
                
                # Store original toolbar state if it exists
                self.original_toolbar = getattr(self.toolbar_current, '_active', None)
                
                # Disable toolbar to prevent interaction conflicts
                if hasattr(self.toolbar_current, 'mode'):
                    self.toolbar_current.mode = ''
                self.toolbar_current._active = None
                
                # Connect the event handlers
                self.click_cid = self.canvas_current.mpl_connect('button_press_event', self.add_peak_on_click)
                self.key_cid = self.canvas_current.mpl_connect('key_press_event', self._on_peak_add_key)
                
                # Update button appearance to show active state
                self.manual_peak_button.configure(
                    text="🛑 Stop Adding Peaks",
                    style="Active.TButton"
                )
                
                # Create a custom style for the active button if it doesn't exist
                style = ttk.Style()
                style.configure("Active.TButton", 
                               foreground="white", 
                               background="red",
                               focuscolor="none")
                # For better cross-platform compatibility
                style.map("Active.TButton",
                         background=[('active', 'darkred'),
                                   ('pressed', 'darkred')])
            else:
                # Disable manual peak mode
                self.disable_peak_addition()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to toggle manual peak selection: {str(e)}", parent=self.window)
            # Ensure we're not in a broken state
            self.adding_peak = False
            self.canvas_current.get_tk_widget().config(cursor="arrow")

    def _on_peak_add_key(self, event):
        if event.key == 'escape':
            self.disable_peak_addition()
             
    def disable_peak_addition(self):
        """Disable manual peak addition mode."""
        # Restore cursor
        self.canvas_current.get_tk_widget().config(cursor="arrow")
         
        # Disconnect event handlers
        if hasattr(self, 'click_cid'):
            self.canvas_current.mpl_disconnect(self.click_cid)
        if hasattr(self, 'key_cid'):
            self.canvas_current.mpl_disconnect(self.key_cid)
         
        # Restore toolbar
        if hasattr(self, 'original_toolbar'):
            self.toolbar_current._active = self.original_toolbar
         
        # Reset flag
        self.adding_peak = False
         
        # Update title to show number of peaks
        if hasattr(self, 'peaks') and self.peaks:
            self.ax1_current.set_title(f'Raman Spectrum - {len(self.peaks)} Peaks Added')
        else:
            self.ax1_current.set_title('Raman Spectrum')
         
        # Reset button appearance to normal state
        self.manual_peak_button.configure(
            text="Click to Add Peak",
            style="TButton"  # Reset to default style
        )
         
        self.canvas_current.draw()
    
    def add_peak_on_click(self, event):
        """Handle click on the plot to add a peak."""
        if not getattr(self, 'adding_peak', False):
            return
        if event.button != 1:
            return
        if event.inaxes != self.ax1_current:
            return
             
        try:
            x, y = event.xdata, event.ydata
             
            # Initialize peaks list if needed
            if not hasattr(self, 'peaks'):
                self.peaks = []
             
            # Find the closest index in the wavenumbers array
            idx = np.argmin(np.abs(self.wavenumbers - x))
             
            # Check if a peak already exists very close to this position
            duplicate_threshold = 5.0  # cm⁻¹
            for existing_peak in self.peaks:
                if abs(existing_peak['position'] - self.wavenumbers[idx]) < duplicate_threshold:
                    # Peak already exists nearby, show message and return
                    messagebox.showinfo("Peak Exists", 
                                      f"A peak already exists near {self.wavenumbers[idx]:.1f} cm⁻¹.\n"
                                      f"Existing peak at {existing_peak['position']:.1f} cm⁻¹", 
                                      parent=self.window)
                    return
             
            # Add the peak
            new_peak = {
                'position': self.wavenumbers[idx],
                'intensity': y,
                'index': idx
            }
            self.peaks.append(new_peak)
             
            # Update the plot
            self.update_plot()
             
            # Update title with current count and instructions
            self.ax1_current.set_title(f'MANUAL MODE: {len(self.peaks)} peaks added • Click red button or press ESC to finish')
            self.canvas_current.draw()
             
            # Provide visual feedback (brief highlight)
            self.highlight_new_peak(self.wavenumbers[idx], y)
             
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add peak: {str(e)}", parent=self.window)
    
    def highlight_new_peak(self, x, y):
        """Briefly highlight a newly added peak."""
        try:
            # Add a temporary highlight circle
            highlight = self.ax1_current.plot(x, y, 'yo', markersize=15, alpha=0.7)[0]
            self.canvas_current.draw()
             
            # Remove the highlight after a short delay
            self.window.after(500, lambda: self.remove_highlight(highlight))
        except:
            pass  # If highlighting fails, just continue
    
    def remove_highlight(self, highlight):
        """Remove the temporary highlight."""
        try:
            highlight.remove()
            self.canvas_current.draw()
        except:
            pass  # If removal fails, just continue
    
    def show_peak_deletion_dialog(self):
        """Show a dialog to select peaks for deletion."""
        if not hasattr(self, 'peaks') or not self.peaks:
            messagebox.showinfo("No Peaks", "No peaks to delete.", parent=self.window)
            return
            
        # Create dialog window
        dialog = tk.Toplevel(self.window)
        dialog.title("Delete Peaks")
        dialog.geometry("300x400")
        
        # Create listbox for peak selection
        list_frame = ttk.Frame(dialog, padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        peak_list = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        peak_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=peak_list.yview)
        
        # Add peaks to listbox
        for i, peak in enumerate(self.peaks):
            peak_list.insert(tk.END, f"Peak {i+1}: {peak['position']:.2f} cm⁻¹")
        
        # Add buttons
        button_frame = ttk.Frame(dialog, padding=10)
        button_frame.pack(fill=tk.X)
        
        def delete_selected():
            # Get selected indices in reverse order to avoid index shifting
            selected_indices = sorted(peak_list.curselection(), reverse=True)
            
            if not selected_indices:
                messagebox.showinfo("No Selection", "Please select peaks to delete.", parent=self.window)
                return
                
            # Confirm deletion
            if messagebox.askyesno("Confirm Delete", f"Delete {len(selected_indices)} selected peak(s)?", parent=self.window):
                # Remove selected peaks
                for idx in selected_indices:
                    del self.peaks[idx]
                
                # Clear fit results
                self.fit_params = []
                self.fit_result = None
                self.residuals = None
                
                # Update plot
                self.update_plot()
                
                # Close dialog
                dialog.destroy()
        
        ttk.Button(button_frame, text="Delete Selected", command=delete_selected).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    def calculate_r_squared(self, y_true, y_pred):
            return "No constraints"
    
    def skip_current_file(self):
        """Manually skip the current file from trend analysis and batch results."""
        if not hasattr(self, 'current_spectrum_index') or self.current_spectrum_index < 0:
            messagebox.showwarning("No File Selected", "Please select a spectrum file first.", parent=self.window)
            return
            
        if self.current_spectrum_index >= len(self.spectra_files):
            messagebox.showwarning("Invalid Selection", "No valid spectrum file selected.", parent=self.window)
            return
            
        # Get the current file path
        current_file = self.spectra_files[self.current_spectrum_index]
        filename = os.path.basename(current_file)
        
        # Confirm the action
        response = messagebox.askyesno(
            "Skip File", 
            f"Skip '{filename}' from trend analysis?\n\n"
            "This file will be excluded from:\n"
            "• Fit Results plots\n"
            "• Fit Statistics\n"
            "• Exported data\n\n"
            "The file will remain in the file list but marked as skipped.",
            parent=self.window
        )
        
        if not response:
            return
            
        # Add to skipped files set
        self.manually_skipped_files.add(current_file)
        
        # Update batch results if this file exists there
        for i, result in enumerate(self.batch_results):
            if result.get('file') == current_file:
                self.batch_results[i]['manually_skipped'] = True
                break
        else:
            # If not in batch results, add it as skipped
            self.batch_results.append({
                'file': current_file,
                'peaks': [],
                'fit_failed': True,
                'fit_params': None,
                'fit_cov': None,
                'background': None,
                'fit_result': None,
                'original_spectra': None,
                'peak_r_squared': [],
                'nlls_cycles': 0,
                'manually_refined': False,
                'manually_skipped': True
            })
        
        # Update the current spectrum display to show it's skipped
        self.current_spectrum_label.config(text=f"Spectrum {self.current_spectrum_index + 1}/{len(self.spectra_files)}: {filename} [SKIPPED]")
        
        # Update trends plot to exclude this file
        self.update_trends_plot()
        self.update_fit_stats_plot()
        
        # Log the action
        if hasattr(self, 'batch_status_text'):
            try:
                self.batch_status_text.config(state=tk.NORMAL)
                self.batch_status_text.insert(tk.END, f"Manually skipped: {filename}\n")
                self.batch_status_text.see(tk.END)
                self.batch_status_text.config(state=tk.DISABLED)
            except:
                pass
        
        print(f"Manually skipped: {filename}")
        messagebox.showinfo("File Skipped", f"'{filename}' has been excluded from trend analysis.", parent=self.window)
    
    def unskip_current_file(self):
        """Remove the current file from the manually skipped list."""
        if not hasattr(self, 'current_spectrum_index') or self.current_spectrum_index < 0:
            return
            
        if self.current_spectrum_index >= len(self.spectra_files):
            return
            
        current_file = self.spectra_files[self.current_spectrum_index]
        filename = os.path.basename(current_file)
        
        # Remove from skipped files set
        self.manually_skipped_files.discard(current_file)
        
        # Update batch results
        for result in self.batch_results:
            if result.get('file') == current_file:
                result['manually_skipped'] = False
                break
        
        # Update display
        self.current_spectrum_label.config(text=f"Spectrum {self.current_spectrum_index + 1}/{len(self.spectra_files)}: {filename}")
        
        # Update plots
        self.update_trends_plot()
        self.update_fit_stats_plot()
        
        print(f"Un-skipped: {filename}")

    def toggle_skip_current_file(self):
        """Toggle the skip status of the current file."""
        if not hasattr(self, 'current_spectrum_index') or self.current_spectrum_index < 0:
            messagebox.showwarning("No File Selected", "Please select a spectrum file first.", parent=self.window)
            return
            
        if self.current_spectrum_index >= len(self.spectra_files):
            messagebox.showwarning("Invalid Selection", "No valid spectrum file selected.", parent=self.window)
            return
            
        current_file = self.spectra_files[self.current_spectrum_index]
        filename = os.path.basename(current_file)
        
        # Toggle skip status
        if current_file in self.manually_skipped_files:
            self.manually_skipped_files.discard(current_file)
            self.skip_button.config(text="⚠ Unskip This File")
        else:
            self.manually_skipped_files.add(current_file)
            self.skip_button.config(text="⚠ Skip This File")
        
        # Update batch results
        for result in self.batch_results:
            if result.get('file') == current_file:
                result['manually_skipped'] = current_file in self.manually_skipped_files
                break
        
        # Update display
        self.current_spectrum_label.config(text=f"Spectrum {self.current_spectrum_index + 1}/{len(self.spectra_files)}: {filename}")
        
        # Update trends plot to exclude this file
        self.update_trends_plot()
        self.update_fit_stats_plot()
        
        print(f"Skipped status changed for: {filename}")
        messagebox.showinfo("File Skipped Status", f"'{filename}' is now {'skipped' if current_file in self.manually_skipped_files else 'unskipped'}.", parent=self.window)

    def load_whewellite_calibration(self):
        """Load a whewellite reference spectrum for calibration."""
        file_path = filedialog.askopenfilename(parent=self.window,
            title="Select Whewellite Reference Spectrum",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Load the spectrum data
                wavenumbers, intensities = self.load_spectrum_robust(file_path)
                if wavenumbers is None or intensities is None:
                    raise Exception(f"Failed to load data from {file_path}")
                
                # Preprocess the spectrum
                wn, corrected_int = self.density_analyzer.preprocess_spectrum(wavenumbers, intensities)
                
                # Calculate calibration value using COM peak
                com_idx = np.argmin(np.abs(wn - self.density_analyzer.com_peaks['main']))
                baseline_mask = (wn >= self.density_analyzer.organic_regions['baseline'][0]) & \
                               (wn <= self.density_analyzer.organic_regions['baseline'][1])
                baseline_intensity = np.mean(corrected_int[baseline_mask])
                whewellite_peak_height = corrected_int[com_idx] - baseline_intensity
                
                self.whewellite_calibration_value = whewellite_peak_height
                self.calib_status_label.config(
                    text=f"Status: Calibrated (Peak height: {whewellite_peak_height:.2f})", 
                    foreground="green"
                )
                
                # Update the analyzer's reference value
                # Note: This would require modifying the RamanDensityAnalyzer class
                # For now, we'll store it and use it in our calculations
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load calibration spectrum: {str(e)}", parent=self.window)
                self.calib_status_label.config(text="Status: Not calibrated", foreground="red")

    def analyze_current_spectrum_density(self):
        """Analyze the current spectrum for density."""
        if not hasattr(self, 'wavenumbers') or not hasattr(self, 'intensities'):
            messagebox.showerror("Error", "No spectrum data available", parent=self.window)
            return
        
        try:
            # Preprocess the spectrum
            wn, corrected_int = self.density_analyzer.preprocess_spectrum(self.wavenumbers, self.intensities)
            
            # Calculate CDI
            cdi, metrics = self.density_analyzer.calculate_crystalline_density_index(wn, corrected_int)
            
            # Calculate apparent density
            apparent_density = self.density_analyzer.calculate_apparent_density(cdi)
            
            # Update results display
            self.single_results_text.delete(1.0, tk.END)
            results_text = f"""Current Spectrum Analysis:
============================
Crystalline Density Index: {cdi:.4f}
Apparent Density: {apparent_density:.4f} g/cm³

Detailed Metrics:
-----------------
COM Peak Height: {metrics['com_peak_height']:.2f}
Baseline Intensity: {metrics['baseline_intensity']:.2f}
Peak Width (FWHM): {metrics['peak_width']:.2f} cm⁻¹
Spectral Contrast: {metrics['spectral_contrast']:.4f}

Interpretation:
---------------
"""
            
            if cdi > 0.5:
                results_text += "High crystalline content (COM-rich region)\n"
            else:
                results_text += "Low crystalline content (organic-rich region)\n"
                
            self.single_results_text.insert(tk.END, results_text)
            
            # Update the density plot with current spectrum
            self.update_density_plot(wn, corrected_int, cdi, apparent_density, metrics)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze spectrum density: {str(e)}", parent=self.window)

    def analyze_all_spectra_density(self):
        """Analyze all spectra for density."""
        if not self.spectra_files:
            messagebox.showwarning("No Spectra", "No spectra to analyze.", parent=self.window)
            return
        
        try:
            self.density_results = []
            self.density_summary_text.delete(1.0, tk.END)
            
            densities = []
            cdis = []
            filenames = []
            
            self.density_summary_text.insert(tk.END, "Processing all spectra...\n\n")
            self.density_summary_text.see(tk.END)
            self.window.update()
            
            for i, file_path in enumerate(self.spectra_files):
                try:
                    # Load the spectrum data
                    wavenumbers, intensities = self.load_spectrum_robust(file_path)
                    if wavenumbers is None or intensities is None:
                        self.density_summary_text.insert(tk.END, f"ERROR: Failed to load {os.path.basename(file_path)}\n")
                        continue
                    
                    # Preprocess the spectrum
                    wn, corrected_int = self.density_analyzer.preprocess_spectrum(wavenumbers, intensities)
                    
                    # Calculate CDI
                    cdi, metrics = self.density_analyzer.calculate_crystalline_density_index(wn, corrected_int)
                    
                    # Calculate apparent density
                    apparent_density = self.density_analyzer.calculate_apparent_density(cdi)
                    
                    # Store results
                    result = {
                        'filename': os.path.basename(file_path),
                        'cdi': cdi,
                        'density': apparent_density,
                        'metrics': metrics
                    }
                    self.density_results.append(result)
                    densities.append(apparent_density)
                    cdis.append(cdi)
                    filenames.append(os.path.basename(file_path))
                    
                    # Update summary
                    self.density_summary_text.insert(tk.END, 
                        f"Spectrum {i+1:3d}: {os.path.basename(file_path)[:30]:30s} "
                        f"CDI: {cdi:.3f}  Density: {apparent_density:.3f} g/cm³\n")
                    
                    if i % 10 == 0:  # Update display every 10 spectra
                        self.density_summary_text.see(tk.END)
                        self.window.update()
                        
                except Exception as e:
                    self.density_summary_text.insert(tk.END, f"ERROR: {os.path.basename(file_path)}: {str(e)}\n")
                    continue
            
            if densities:
                # Calculate statistics
                mean_density = np.mean(densities)
                std_density = np.std(densities)
                min_density = np.min(densities)
                max_density = np.max(densities)
                mean_cdi = np.mean(cdis)
                
                summary_stats = f"""
Batch Analysis Complete!
========================
Processed: {len(densities)}/{len(self.spectra_files)} spectra

Density Statistics:
-------------------
Mean Density: {mean_density:.4f} ± {std_density:.4f} g/cm³
Range: {min_density:.4f} - {max_density:.4f} g/cm³
Mean CDI: {mean_cdi:.4f}

Classification:
---------------
Crystalline samples (CDI > 0.5): {sum(1 for cdi in cdis if cdi > 0.5)}/{len(cdis)}
Organic samples (CDI ≤ 0.5): {sum(1 for cdi in cdis if cdi <= 0.5)}/{len(cdis)}
"""
                self.density_summary_text.insert(tk.END, summary_stats)
                
                # Update batch density plot
                self.update_batch_density_plot(densities, cdis, filenames)
                
            else:
                self.density_summary_text.insert(tk.END, "\nNo spectra could be processed successfully.")
                
            self.density_summary_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze all spectra: {str(e)}", parent=self.window)

    def update_density_plot(self, wavenumbers, intensities, cdi, density, metrics):
        """Update the density analysis plot with current spectrum data."""
        try:
            # Clear all subplots
            for ax in self.ax_density.flat:
                ax.clear()
            
            # Plot 1: Spectrum with COM peak highlighted
            self.ax_density[0, 0].plot(wavenumbers, intensities, 'b-', linewidth=1)
            com_peak = self.density_analyzer.com_peaks['main']
            self.ax_density[0, 0].axvline(x=com_peak, color='r', linestyle='--', 
                                         label=f'COM peak ({com_peak} cm⁻¹)')
            self.ax_density[0, 0].set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_density[0, 0].set_ylabel('Intensity')
            self.ax_density[0, 0].set_title('Preprocessed Spectrum')
            self.ax_density[0, 0].legend()
            self.ax_density[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: CDI visualization
            colors = ['red', 'green']
            labels = ['Organic', 'Crystalline']
            values = [1-cdi, cdi]
            self.ax_density[0, 1].pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            self.ax_density[0, 1].set_title(f'Crystalline Density Index: {cdi:.3f}')
            
            # Plot 3: Density bar chart
            density_types = ['Organic Matrix', 'Calculated', 'Pure COM']
            density_values = [self.density_analyzer.densities['organic_matrix'], 
                            density, 
                            self.density_analyzer.densities['com_crystal']]
            colors = ['orange', 'blue', 'red']
            bars = self.ax_density[1, 0].bar(density_types, density_values, color=colors, alpha=0.7)
            self.ax_density[1, 0].set_ylabel('Density (g/cm³)')
            self.ax_density[1, 0].set_title('Density Comparison')
            self.ax_density[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, density_values):
                height = bar.get_height()
                self.ax_density[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                         f'{value:.3f}', ha='center', va='bottom')
            
            # Plot 4: Metrics summary
            metric_names = ['Peak Height', 'Baseline', 'Peak Width', 'Contrast']
            metric_values = [metrics['com_peak_height'], metrics['baseline_intensity'],
                           metrics['peak_width'], metrics['spectral_contrast']]
            
            self.ax_density[1, 1].barh(metric_names, metric_values, color='purple', alpha=0.7)
            self.ax_density[1, 1].set_xlabel('Value')
            self.ax_density[1, 1].set_title('Analysis Metrics')
            
            # Add value labels
            for i, value in enumerate(metric_values):
                self.ax_density[1, 1].text(value + max(metric_values)*0.01, i, 
                                         f'{value:.2f}', va='center')
            
            self.fig_density.tight_layout()
            self.canvas_density.draw()
            
        except Exception as e:
            print(f"Error updating density plot: {e}")

    def update_batch_density_plot(self, densities, cdis, filenames):
        """Update the density analysis plot with batch results."""
        try:
            # Clear all subplots
            for ax in self.ax_density.flat:
                ax.clear()
            
            # Plot 1: Density vs spectrum number
            spectrum_nums = range(1, len(densities) + 1)
            self.ax_density[0, 0].plot(spectrum_nums, densities, 'bo-', markersize=4, linewidth=1)
            self.ax_density[0, 0].axhline(y=np.mean(densities), color='r', linestyle='--', 
                                         label=f'Mean: {np.mean(densities):.3f}')
            self.ax_density[0, 0].set_xlabel('Spectrum Number')
            self.ax_density[0, 0].set_ylabel('Density (g/cm³)')
            self.ax_density[0, 0].set_title('Density Across All Spectra')
            self.ax_density[0, 0].legend()
            self.ax_density[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: CDI vs spectrum number
            self.ax_density[0, 1].plot(spectrum_nums, cdis, 'go-', markersize=4, linewidth=1)
            self.ax_density[0, 1].axhline(y=0.5, color='r', linestyle='--', 
                                         label='Classification threshold')
            self.ax_density[0, 1].set_xlabel('Spectrum Number')
            self.ax_density[0, 1].set_ylabel('Crystalline Density Index')
            self.ax_density[0, 1].set_title('CDI Across All Spectra')
            self.ax_density[0, 1].legend()
            self.ax_density[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Density histogram
            self.ax_density[1, 0].hist(densities, bins=min(20, len(densities)//2), 
                                      alpha=0.7, color='blue', edgecolor='black')
            self.ax_density[1, 0].axvline(x=np.mean(densities), color='r', linestyle='--', 
                                         linewidth=2, label=f'Mean: {np.mean(densities):.3f}')
            self.ax_density[1, 0].set_xlabel('Density (g/cm³)')
            self.ax_density[1, 0].set_ylabel('Frequency')
            self.ax_density[1, 0].set_title('Density Distribution')
            self.ax_density[1, 0].legend()
            self.ax_density[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: CDI vs Density scatter
            colors = ['red' if cdi <= 0.5 else 'blue' for cdi in cdis]
            self.ax_density[1, 1].scatter(cdis, densities, c=colors, alpha=0.6, s=30)
            self.ax_density[1, 1].axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
            self.ax_density[1, 1].set_xlabel('Crystalline Density Index')
            self.ax_density[1, 1].set_ylabel('Density (g/cm³)')
            self.ax_density[1, 1].set_title('CDI vs Density Correlation')
            self.ax_density[1, 1].grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='blue', label='Crystalline'),
                              Patch(facecolor='red', label='Organic')]
            self.ax_density[1, 1].legend(handles=legend_elements)
            
            self.fig_density.tight_layout()
            self.canvas_density.draw()
            
        except Exception as e:
            print(f"Error updating batch density plot: {e}")

    def export_density_results(self):
        """Export density analysis results to CSV."""
        if not self.density_results:
            messagebox.showwarning("No Results", "No density analysis results to export.", parent=self.window)
            return
        
        try:
            # Ask for file location
            file_path = filedialog.asksaveasfilename(parent=self.window,
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Save Density Results"
            )
            
            if not file_path:
                return
            
            # Prepare data for export
            data = []
            for i, result in enumerate(self.density_results):
                row = {
                    'Spectrum_Number': i + 1,
                    'Filename': result['filename'],
                    'Crystalline_Density_Index': result['cdi'],
                    'Apparent_Density_g_cm3': result['density'],
                    'COM_Peak_Height': result['metrics']['com_peak_height'],
                    'Baseline_Intensity': result['metrics']['baseline_intensity'],
                    'Peak_Width_cm1': result['metrics']['peak_width'],
                    'Spectral_Contrast': result['metrics']['spectral_contrast']
                }
                data.append(row)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Density results for {len(data)} spectra saved to {file_path}", parent=self.window)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export density results: {str(e)}", parent=self.window)
    
    def on_file_select(self, event):
        """Handle file selection in the listbox."""
        try:
            # Get the index of the clicked item
            index = self.file_listbox.index(f"@{event.x},{event.y}")
            # Load and display the selected spectrum
            self.load_spectrum(index)
            # Make sure this item is selected
            self.file_listbox.selection_set(index)
        except Exception as e:
            # If we can't get the index from the event, use the first selected item
            selection = self.file_listbox.curselection()
            if selection:
                self.load_spectrum(selection[0])
            else:
                self.clear_plot()
    
    def on_tab_changed(self, event=None):
        """Handle tab change events to update context-sensitive help."""
        # Update status bar with a hint about the current tab
        try:
            # Determine which tab is active
            if event and event.widget == self.left_notebook:
                current_tab = self.left_notebook.tab(self.left_notebook.select(), "text")
                if current_tab == "File Selection":
                    self.status_bar.config(text="File Selection: Add or remove spectrum files. Double-click to select. Press F1 for help.")
                elif current_tab == "Peaks":
                    self.status_bar.config(text="Peak Controls: Subtract background, detect/add peaks, and fit with different models. Press F1 for help.")
                elif current_tab == "Batch":
                    self.status_bar.config(text="Batch Processing: Set reference spectrum and apply to all files. Press F1 for help.")
                elif current_tab == "Results":
                    self.status_bar.config(text="Results: Configure which peaks are displayed and export data. Press F1 for help.")
            elif event and event.widget == self.viz_notebook:
                current_tab = self.viz_notebook.tab(self.viz_notebook.select(), "text")
                if current_tab == "Current Spectrum":
                    self.status_bar.config(text="Current Spectrum: View selected spectrum with fit results. Press F1 for help.")
                elif current_tab == "Waterfall":
                    self.status_bar.config(text="Waterfall Plot: View multiple spectra stacked vertically. Press F1 for help.")
                elif current_tab == "Heatmap":
                    self.status_bar.config(text="Heatmap: View spectra as a color-coded intensity map. Press F1 for help.")
                elif current_tab == "Fit Results":
                    self.status_bar.config(text="Fit Results: View trends in peak parameters across all spectra. Press F1 for help.")
                elif current_tab == "Fit Stats":
                    self.status_bar.config(text="Fit Statistics: Compare fit quality across different spectra. Press F1 for help.")
                elif current_tab == "Density Analysis":
                    self.status_bar.config(text="Density Analysis: Quantitative analysis for correlation with micro-CT. Press F1 for help.")
        except:
            # Default status message if something goes wrong
            self.status_bar.config(text="Press F1 for help with the current tab")
    
    def update_waterfall_plot(self):
        """Update the waterfall plot."""
        # Get current files from listbox
        current_files = []
        for i in range(self.file_listbox.size()):
            filename = self.file_listbox.get(i)
            # Find the full path in spectra_files
            for full_path in self.spectra_files:
                if os.path.basename(full_path) == filename:
                    current_files.append(full_path)
                    break
        
        if not current_files:
            self.fig_waterfall.clf()
            self.ax_waterfall = self.fig_waterfall.add_subplot(111)
            self.ax_waterfall.text(0.5, 0.5, 'No spectra to plot', ha='center', va='center', transform=self.ax_waterfall.transAxes)
            self.canvas_waterfall.draw()
            return

        try:
            # Clear the figure and recreate axes to prevent shrinking
            self.fig_waterfall.clf()
            self.ax_waterfall = self.fig_waterfall.add_subplot(111)
            
            skip = int(self.waterfall_skip.get())
            all_spectra = []
            all_wavenumbers = None
            for file_path in current_files[::skip]:
                # Load spectrum with robust encoding detection
                wavenumbers, intensities = self.load_spectrum_robust(file_path)
                if wavenumbers is None or intensities is None:
                    raise Exception(f"Failed to load data from {file_path}")
                data = np.column_stack((wavenumbers, intensities))
                wavenumbers = data[:, 0]
                spectrum = data[:, 1]
                if all_wavenumbers is None:
                    all_wavenumbers = wavenumbers
                all_spectra.append(spectrum)
            
            # Get X-axis range
            try:
                xmin = float(self.waterfall_xmin.get()) if self.waterfall_xmin.get() else all_wavenumbers[0]
                xmax = float(self.waterfall_xmax.get()) if self.waterfall_xmax.get() else all_wavenumbers[-1]
            except ValueError:
                xmin = all_wavenumbers[0]
                xmax = all_wavenumbers[-1]
            
            # Create mask for X-axis range
            mask = (all_wavenumbers >= xmin) & (all_wavenumbers <= xmax)
            wavenumbers_masked = all_wavenumbers[mask]
            all_spectra_masked = [spectrum[mask] for spectrum in all_spectra]

            # Determine color gradient
            cmap_name = self.waterfall_cmap.get() if hasattr(self, 'waterfall_cmap') else 'all_black'
            n_lines = len(all_spectra_masked)
            if cmap_name == 'all_black':
                colors = ['black'] * n_lines
            elif cmap_name == 'black_to_darkgrey':
                colors = [(i/(n_lines-1)*0.3, i/(n_lines-1)*0.3, i/(n_lines-1)*0.3) for i in range(n_lines)]
            elif cmap_name == 'darkgrey_to_black':
                colors = [((1-i/(n_lines-1))*0.3, (1-i/(n_lines-1))*0.3, (1-i/(n_lines-1))*0.3) for i in range(n_lines)]
            else:
                try:
                    cmap = plt.get_cmap(cmap_name)
                    # Get user-defined colormap min/max
                    try:
                        color_min = float(self.waterfall_cmap_min.get())
                        color_max = float(self.waterfall_cmap_max.get())
                        # Clamp to [0, 1]
                        color_min = max(0.0, min(1.0, color_min))
                        color_max = max(0.0, min(1.0, color_max))
                        if color_max <= color_min:
                            color_max = color_min + 0.01  # Ensure at least a small range
                    except Exception:
                        color_min, color_max = 0.0, 0.85
                    colors = [cmap(color_min + (color_max - color_min) * (i / (n_lines - 1))) for i in range(n_lines)]
                except Exception:
                    colors = ['black'] * n_lines
            linewidth = float(self.waterfall_linewidth.get()) if hasattr(self, 'waterfall_linewidth') else 1.5
            
            # Calculate the maximum intensity for proper scaling
            max_intensity = max(np.max(spectrum) for spectrum in all_spectra_masked)
            # Get Y offset from entry (absolute units)
            try:
                offset_step = float(self.waterfall_yoffset.get())
            except ValueError:
                offset_step = 100  # Default if invalid value
            
            for i, (spectrum, color) in enumerate(zip(all_spectra_masked, colors)):
                offset = i * offset_step
                self.ax_waterfall.plot(wavenumbers_masked, spectrum + offset, 
                                     label=f'Spectrum {i*skip + 1}', 
                                     color=color, 
                                     linewidth=linewidth)
            
            # Set tight limits for x and y axes
            self.ax_waterfall.set_xlim(wavenumbers_masked[0], wavenumbers_masked[-1])
            y_min = min(np.min(spectrum) for spectrum in all_spectra_masked)
            y_max = max(np.max(spectrum) + (len(all_spectra_masked) - 1) * offset_step for spectrum in all_spectra_masked)
            self.ax_waterfall.set_ylim(y_min, y_max)
            
            self.ax_waterfall.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_waterfall.set_ylabel('Intensity (a.u.)')
            self.ax_waterfall.set_title('Waterfall Plot')
            if self.waterfall_show_grid.get():
                self.ax_waterfall.grid(True, linestyle=':', color='gray', alpha=0.6)
            else:
                self.ax_waterfall.grid(False)
            if self.waterfall_show_legend.get():
                self.ax_waterfall.legend(loc='upper right', fontsize=8, frameon=True)
            
            # Ensure tight layout and maintain aspect ratio
            self.fig_waterfall.tight_layout()
            self.canvas_waterfall.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update waterfall plot: {str(e)}", parent=self.window)
    
    def reset_heatmap_adjustments(self):
        """Reset all heatmap color adjustments to default values."""
        self.heatmap_contrast.set(1.0)
        self.heatmap_brightness.set(1.0)
        self.heatmap_gamma.set(1.0)
        self.update_heatmap_plot()
    
    def reset_waterfall_range(self):
        """Reset waterfall plot X-axis range to full range."""
        self.waterfall_xmin.delete(0, tk.END)
        self.waterfall_xmax.delete(0, tk.END)
        self.update_waterfall_plot()
    
    def reset_heatmap_range(self):
        """Reset heatmap plot X-axis range to full range."""
        self.heatmap_xmin.delete(0, tk.END)
        self.heatmap_xmax.delete(0, tk.END)
        self.update_heatmap_plot()
    
    def update_heatmap_plot(self):
        """Update the heatmap plot."""
        # Get current files from listbox
        current_files = []
        for i in range(self.file_listbox.size()):
            filename = self.file_listbox.get(i)
            # Find the full path in spectra_files
            for full_path in self.spectra_files:
                if os.path.basename(full_path) == filename:
                    current_files.append(full_path)
                    break
        
        if not current_files:
            self.fig_heatmap.clf()
            self.ax_heatmap = self.fig_heatmap.add_subplot(111)
            self.ax_heatmap.text(0.5, 0.5, 'No spectra to plot', ha='center', va='center', transform=self.ax_heatmap.transAxes)
            self.canvas_heatmap.draw()
            return

        try:
            # Clear the figure and recreate axes to prevent shrinking
            self.fig_heatmap.clf()
            self.ax_heatmap = self.fig_heatmap.add_subplot(111)
            self._heatmap_colorbar = None
            all_spectra = []
            all_wavenumbers = None
            for file_path in current_files:
                # Load spectrum with robust encoding detection
                wavenumbers, intensities = self.load_spectrum_robust(file_path)
                if wavenumbers is None or intensities is None:
                    raise Exception(f"Failed to load data from {file_path}")
                data = np.column_stack((wavenumbers, intensities))
                wavenumbers = data[:, 0]
                spectrum = data[:, 1]
                if all_wavenumbers is None:
                    all_wavenumbers = wavenumbers
                all_spectra.append(spectrum)
            
            # Get X-axis range
            try:
                xmin = float(self.heatmap_xmin.get()) if self.heatmap_xmin.get() else all_wavenumbers[0]
                xmax = float(self.heatmap_xmax.get()) if self.heatmap_xmax.get() else all_wavenumbers[-1]
            except ValueError:
                xmin = all_wavenumbers[0]
                xmax = all_wavenumbers[-1]
            
            # Create mask for X-axis range
            mask = (all_wavenumbers >= xmin) & (all_wavenumbers <= xmax)
            wavenumbers_masked = all_wavenumbers[mask]
            # Apply mask to spectra data as well to match the wavenumber range
            spectra_array = np.array(all_spectra)[:, mask]
            
            # Apply color adjustments
            contrast = self.heatmap_contrast.get()
            brightness = self.heatmap_brightness.get()
            gamma = self.heatmap_gamma.get()
            
            # Normalize the data
            vmin, vmax = np.percentile(spectra_array, (2, 98))  # Use 2nd and 98th percentiles for better contrast
            normalized_data = (spectra_array - vmin) / (vmax - vmin)
            
            # Apply gamma correction
            normalized_data = np.power(normalized_data, 1/gamma)
            
            # Apply contrast
            normalized_data = (normalized_data - 0.5) * contrast + 0.5
            
            # Apply brightness
            normalized_data = normalized_data * brightness
            
            # Clip values to valid range
            normalized_data = np.clip(normalized_data, 0, 1)
            
            # Create custom colormap
            base_cmap = plt.get_cmap(self.heatmap_cmap.get())
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', base_cmap(np.linspace(0, 1, 256)))
            
            # Create the heatmap
            im = self.ax_heatmap.imshow(
                normalized_data,
                aspect='auto',
                extent=[wavenumbers_masked[0], wavenumbers_masked[-1], len(all_spectra), 0],
                cmap=custom_cmap,
                vmin=0,
                vmax=1
            )
            
            # Add colorbar
            if hasattr(self, '_heatmap_colorbar') and self._heatmap_colorbar is not None:
                self._heatmap_colorbar.remove()
            self._heatmap_colorbar = self.fig_heatmap.colorbar(im, ax=self.ax_heatmap)
            self._heatmap_colorbar.set_label('Intensity (a.u.)')
            
            self.ax_heatmap.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_heatmap.set_ylabel('Spectrum Number')
            self.ax_heatmap.set_title('Heatmap Plot')
            
            # Show or hide grid based on checkbox
            if self.heatmap_show_grid.get():
                self.ax_heatmap.grid(True, linestyle=':', color='gray', alpha=0.6)
            else:
                self.ax_heatmap.grid(False)
            
            # Ensure tight layout and maintain aspect ratio
            self.fig_heatmap.tight_layout()
            self.canvas_heatmap.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update heatmap plot: {str(e)}", parent=self.window)
    
    def export_individual_fit_stats(self):
        """Export individual fit statistics plots to separate image files."""
        try:
            # Ask for directory to save files
            save_dir = filedialog.askdirectory(parent=self.window, title="Select Directory to Save Individual Plots")
            if not save_dir:
                return
                
            # Get current model type for filename
            model_type = self.current_model.get()
            
            # Create a temporary figure for each metric
            metrics = ['position', 'amplitude', 'width', 'eta']
            for metric in metrics:
                # Skip eta for models that don't support it
                if metric == 'eta' and model_type not in ["Pseudo-Voigt", "Asymmetric Voigt"]:
                    continue
                    
                # Create figure for this metric
                fig, ax = plt.subplots(figsize=(8, 6))
                self._plot_single_trends_subplot(ax, metric)
                
                # Add title and labels
                ax.set_title(f'Peak {metric.capitalize()} Trends')
                ax.set_xlabel('Spectrum Number')
                if metric == 'position':
                    ax.set_ylabel('Position (cm⁻¹)')
                elif metric == 'amplitude':
                    ax.set_ylabel('Amplitude')
                elif metric == 'width':
                    ax.set_ylabel('Width (cm⁻¹)')
                elif metric == 'eta':
                    ax.set_ylabel('Eta')
                
                # Save figure
                filename = f"{model_type}_peak_{metric}.png"
                filepath = os.path.join(save_dir, filename)
                fig.tight_layout()
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
            messagebox.showinfo("Export Complete", f"Individual plots saved to {save_dir}", parent=self.window)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export individual plots: {str(e)}", parent=self.window)
    
    def export_curve_data_csv(self):
        """Export all peak curve data to a CSV file."""
        if not self.batch_results:
            messagebox.showwarning("No Results", "No batch processing results to export.", parent=self.window)
            return
            
        try:
            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                parent=self.window,
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Save Peak Curve Data"
            )
            
            if not file_path:
                return
                
            # Get data from extract_trend_data which already has all the peak information we need
            x, trends, n_peaks, uncertainties = self.extract_trend_data()
            if x is None or trends is None or n_peaks is None:
                messagebox.showwarning("No Data", "No trend data available to export.", parent=self.window)
                return
                
            # Get model type to determine column headers
            model_type = self.current_model.get()
            
            # Prepare data frame
            data = []
            # First add spectrum info (index and filename)
            for i, result in enumerate(self.batch_results):
                filename = os.path.basename(result['file'])
                fit_success = not result.get('fit_failed', True)
                row = {'Spectrum': i, 'File': filename, 'Fit_Success': fit_success}
                data.append(row)
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add column for each peak parameter
            for peak_idx in range(n_peaks):
                # Only include visible peaks
                if peak_idx < len(self.peak_visibility_vars) and self.peak_visibility_vars[peak_idx].get():
                    # Position
                    df[f'Peak_{peak_idx+1}_Position'] = trends[peak_idx]['pos']
                    df[f'Peak_{peak_idx+1}_Position_Error'] = [err if err is not None else np.nan for err in uncertainties[peak_idx]['pos']]
                    
                    # Amplitude
                    df[f'Peak_{peak_idx+1}_Amplitude'] = trends[peak_idx]['amp']
                    df[f'Peak_{peak_idx+1}_Amplitude_Error'] = [err if err is not None else np.nan for err in uncertainties[peak_idx]['amp']]
                    
                    # Width
                    if model_type == "Asymmetric Voigt":
                        df[f'Peak_{peak_idx+1}_Width_Left'] = trends[peak_idx]['wid_left']
                        df[f'Peak_{peak_idx+1}_Width_Left_Error'] = [err if err is not None else np.nan for err in uncertainties[peak_idx]['wid_left']]
                        df[f'Peak_{peak_idx+1}_Width_Right'] = trends[peak_idx]['wid_right']
                        df[f'Peak_{peak_idx+1}_Width_Right_Error'] = [err if err is not None else np.nan for err in uncertainties[peak_idx]['wid_right']]
                    else:
                        df[f'Peak_{peak_idx+1}_Width'] = trends[peak_idx]['wid']
                        df[f'Peak_{peak_idx+1}_Width_Error'] = [err if err is not None else np.nan for err in uncertainties[peak_idx]['wid']]
                    
                    # Eta (for Pseudo-Voigt and Asymmetric Voigt)
                    if model_type in ["Pseudo-Voigt", "Asymmetric Voigt"]:
                        df[f'Peak_{peak_idx+1}_Eta'] = trends[peak_idx]['eta']
                        df[f'Peak_{peak_idx+1}_Eta_Error'] = [err if err is not None else np.nan for err in uncertainties[peak_idx]['eta']]
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Peak curve data saved to {file_path}", parent=self.window)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export curve data: {str(e)}", parent=self.window)
    
    def export_fit_stats_plot(self):
        """Export the Fit Stats plot to an image file."""
        try:
            file_path = filedialog.asksaveasfilename(
                parent=self.window,
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ],
                title="Save Fit Stats Plot"
            )
            
            if file_path:
                # Save the current figure
                self.fig_stats.savefig(
                    file_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none'
                )
                
                messagebox.showinfo("Success", f"Plot saved successfully to:\n{file_path}", parent=self.window)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot: {str(e)}", parent=self.window)
    
    def update_fit_stats_plot(self):
        """Update the fit statistics plots to show best, median, and worst fits."""
        if not self.batch_results:
            # Clear all subplots
            for ax in self.ax_stats.flatten():
                ax.clear()
                ax.axis('off')
            self.canvas_stats.draw()
            return

        # Clear all subplots
        for ax in self.ax_stats.flatten():
            ax.clear()

        # For now, just show a message that this feature is being restored
        self.ax_stats[1, 1].text(0.5, 0.5, 'Fit Statistics Plot\n(Feature being restored)', 
                                ha='center', va='center', transform=self.ax_stats[1, 1].transAxes)
        
        self.canvas_stats.draw()
    
    def calculate_peak_r_squared(self, y_true, y_fit_total, peak_params, model_type):
        """Calculate R² values for individual peaks."""
        try:
            # For now, return empty list - this method needs to be restored from backup
            return []
        except Exception:
            return []
    
    def validate_fit_parameters(self, popt, initial_params, model_type, peaks_in_roi):
        """Validate fit parameters for realism."""
        try:
            # Basic validation - check if parameters are finite and reasonable
            if not np.all(np.isfinite(popt)):
                return False
            
            # Check if any amplitudes are negative
            model_type = self.current_model.get()
            if model_type in ["Gaussian", "Lorentzian"]:
                params_per_peak = 3
            elif model_type == "Pseudo-Voigt":
                params_per_peak = 4
            elif model_type == "Asymmetric Voigt":
                params_per_peak = 5
            else:
                params_per_peak = 3
            
            n_peaks = len(popt) // params_per_peak
            for i in range(n_peaks):
                amp = popt[i * params_per_peak]  # Amplitude is first parameter
                if amp < 0:
                    return False
            
            return True
        except Exception:
            return False
    
    def update_batch_results_if_present(self):
        """Update batch results if the current spectrum is part of batch results."""
        if not hasattr(self, 'batch_results') or not self.batch_results:
            return
            
        if not hasattr(self, 'current_spectrum_index') or self.current_spectrum_index < 0:
            return
            
        if self.current_spectrum_index >= len(self.spectra_files):
            return
            
        current_file = self.spectra_files[self.current_spectrum_index]
        
        # Find and update the result for this file
        for i, result in enumerate(self.batch_results):
            if result.get('file') == current_file:
                # Update with current fit results
                self.batch_results[i].update({
                    'peaks': self.peaks.copy() if hasattr(self, 'peaks') else [],
                    'fit_failed': getattr(self, 'fit_failed', True),
                    'fit_params': np.copy(self.fit_params) if hasattr(self, 'fit_params') and self.fit_params is not None else None,
                    'fit_cov': np.copy(self.fit_cov) if hasattr(self, 'fit_cov') and self.fit_cov is not None else None,
                    'background': np.copy(self.background) if hasattr(self, 'background') and self.background is not None else None,
                    'fit_result': np.copy(self.fit_result) if hasattr(self, 'fit_result') and self.fit_result is not None else None,
                    'peak_r_squared': self.peak_r_squared.copy() if hasattr(self, 'peak_r_squared') else [],
                    'manually_refined': True  # Mark as manually refined
                })
                break
    
    def get_constraint_info_text(self):
        """Get text describing current parameter constraints."""
        constraints = []
        if hasattr(self, 'fix_positions') and self.fix_positions.get():
            constraints.append("positions fixed")
        if hasattr(self, 'fix_widths') and self.fix_widths.get():
            constraints.append("widths fixed")
        
        if constraints:
            return ", ".join(constraints)
        else:
            return "No constraints"