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
                "• Peak Detection: Set height, distance, and prominence thresholds\n"
                "  to automatically find peaks, or click Clear Peaks to start over\n\n"
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
        
        # --- LEFT PANE: Replace with Notebook Tabs ---
        self.left_panel_width = 350
        self.left_panel_container = ttk.Frame(main_container, width=self.left_panel_width)
        self.left_panel_container.pack(side=tk.LEFT, fill=tk.Y)
        self.left_panel_container.pack_propagate(False)

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

        controls_frame = ttk.LabelFrame(peak_tab, text="Peak Fitting Controls", padding=10)
        controls_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Background controls
        bg_frame = ttk.LabelFrame(controls_frame, text="Background", padding=5)
        bg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="λ (smoothness):").pack(anchor=tk.W)
        self.var_lambda = tk.StringVar(value="1e5")
        ttk.Entry(bg_frame, textvariable=self.var_lambda).pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="p (asymmetry):").pack(anchor=tk.W)
        self.var_p = tk.StringVar(value="0.01")
        ttk.Entry(bg_frame, textvariable=self.var_p).pack(fill=tk.X, pady=2)
        ttk.Button(bg_frame, text="Subtract Background", command=self.subtract_background).pack(fill=tk.X, pady=2)

        # Manual peak controls
        manual_frame = ttk.LabelFrame(controls_frame, text="Manual Peak Control", padding=5)
        manual_frame.pack(fill=tk.X, pady=2)
        
        # Add/Delete buttons
        button_frame = ttk.Frame(manual_frame)
        button_frame.pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Click to Add Peak", command=self.enable_peak_addition).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="Delete Peak", command=self.show_peak_deletion_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Peak detection controls
        peak_frame = ttk.LabelFrame(controls_frame, text="Peak Detection", padding=5)
        peak_frame.pack(fill=tk.X, pady=2)
        ttk.Label(peak_frame, text="Height:").pack(anchor=tk.W)
        self.var_height = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_height).pack(fill=tk.X, pady=2)
        ttk.Label(peak_frame, text="Distance:").pack(anchor=tk.W)
        self.var_distance = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_distance).pack(fill=tk.X, pady=2)
        ttk.Label(peak_frame, text="Prominence:").pack(anchor=tk.W)
        self.var_prominence = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_prominence).pack(fill=tk.X, pady=2)
        
        # Create a frame for the buttons
        peak_button_frame = ttk.Frame(peak_frame)
        peak_button_frame.pack(fill=tk.X, pady=2)
        ttk.Button(peak_button_frame, text="Find Peaks", command=self.find_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(peak_button_frame, text="Clear Peaks", command=self.clear_peaks).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Model selection
        model_frame = ttk.LabelFrame(controls_frame, text="Peak Model", padding=5)
        model_frame.pack(fill=tk.X, pady=2)
        self.current_model = tk.StringVar(value="Gaussian")
        model_combo = ttk.Combobox(model_frame, textvariable=self.current_model, 
                                 values=["Gaussian", "Lorentzian", "Pseudo-Voigt", "Asymmetric Voigt"])
        model_combo.pack(fill=tk.X, pady=2)
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
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.viz_notebook = ttk.Notebook(right_panel)
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
        
        # Create status bar at the bottom of the window
        self.status_bar = ttk.Label(self.window, text="Press F1 for help with the current tab", relief=tk.SUNKEN, anchor=tk.W, padding=(10, 2))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def add_files(self):
        """Add files to the batch processing list."""
        files = filedialog.askopenfilenames(
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
    
    def load_spectrum(self, index):
        """Load a spectrum from the list."""
        if not self.spectra_files or index < 0 or index >= len(self.spectra_files):
            self.clear_plot()
            return
            
        try:
            # Load the spectrum data
            file_path = self.spectra_files[index]
            data = np.loadtxt(file_path)
            self.wavenumbers = data[:, 0]
            self.spectra = data[:, 1]
            self.original_spectra = np.copy(self.spectra)
            
            # Update current spectrum index and label
            self.current_spectrum_index = index
            self.current_spectrum_label.config(text=f"Current: {os.path.basename(file_path)}")
            
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
            
            # Update the plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load spectrum: {str(e)}")
            self.clear_plot()
    
    def set_reference(self):
        """Set the current spectrum as the reference for batch processing."""
        if not hasattr(self, 'peaks') or not self.peaks:
            messagebox.showwarning("No Peaks", "Please detect and fit peaks first.")
            return
            
        self.reference_peaks = self.peaks.copy()
        self.reference_background = self.background.copy() if self.background is not None else None
        messagebox.showinfo("Reference Set", "Current spectrum set as reference for batch processing.")
    
    def apply_to_all(self):
        """Apply the current peak fitting parameters to all spectra."""
        if not self.reference_peaks:
            messagebox.showwarning("No Reference", "Please set a reference spectrum first.")
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
                messagebox.showinfo("Stopped", "Batch processing was stopped by the user.")
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
            messagebox.showinfo("Complete", "Batch processing completed.")
            
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
        for result in self.batch_results:
            # Check if this is a failed fit
            fit_failed = result.get('fit_failed', False)
            fit_params = result.get('fit_params')
            fit_cov = result.get('fit_cov')  # Get covariance matrix
            
            if not fit_failed and fit_params is not None:
                if n_peaks is None:
                    n_peaks = len(fit_params) // params_per_peak
                all_params.append(np.array(fit_params).reshape(-1, params_per_peak))
                all_covs.append(fit_cov)  # Store covariance matrix
            else:
                # For failed fits, append None to maintain array index correspondence
                all_params.append(None)
                all_covs.append(None)
        
        if n_peaks is None:
            return None, None, None, None
        
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
                        # Calculate uncertainties from covariance matrix
                        if cov is not None:
                            start_idx = peak_idx * params_per_peak
                            pos_errs[i] = 1.96 * np.sqrt(cov[start_idx+1, start_idx+1])  # 95% confidence interval
                            amp_errs[i] = 1.96 * np.sqrt(cov[start_idx, start_idx])
                            wid_errs[i] = 1.96 * np.sqrt(cov[start_idx+2, start_idx+2])
                    elif model_type == "Pseudo-Voigt":
                        amp, pos, wid, eta = params[peak_idx]
                        pos_vals[i] = pos
                        amp_vals[i] = amp
                        wid_vals[i] = wid
                        eta_vals[i] = eta
                        if cov is not None:
                            start_idx = peak_idx * params_per_peak
                            pos_errs[i] = 1.96 * np.sqrt(cov[start_idx+1, start_idx+1])
                            amp_errs[i] = 1.96 * np.sqrt(cov[start_idx, start_idx])
                            wid_errs[i] = 1.96 * np.sqrt(cov[start_idx+2, start_idx+2])
                            eta_errs[i] = 1.96 * np.sqrt(cov[start_idx+3, start_idx+3])
                    elif model_type == "Asymmetric Voigt":
                        amp, pos, wid_left, wid_right, eta = params[peak_idx]
                        pos_vals[i] = pos
                        amp_vals[i] = amp
                        wid_left_vals[i] = wid_left
                        wid_right_vals[i] = wid_right
                        eta_vals[i] = eta
                        if cov is not None:
                            start_idx = peak_idx * params_per_peak
                            pos_errs[i] = 1.96 * np.sqrt(cov[start_idx+1, start_idx+1])
                            amp_errs[i] = 1.96 * np.sqrt(cov[start_idx, start_idx])
                            wid_left_errs[i] = 1.96 * np.sqrt(cov[start_idx+2, start_idx+2])
                            wid_right_errs[i] = 1.96 * np.sqrt(cov[start_idx+3, start_idx+3])
                            eta_errs[i] = 1.96 * np.sqrt(cov[start_idx+4, start_idx+4])
            
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

        # Count how many spectra had failed fits
        failed_fits_count = sum(1 for result in self.batch_results if result.get('fit_failed', False))
        if failed_fits_count > 0:
            for ax in axes:
                ax.text(0.02, 0.98, f"Note: {failed_fits_count} failed fits excluded", 
                       transform=ax.transAxes, fontsize=8, va='top', ha='left', 
                       color='red', bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

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
                                
                                if len(yerrg) != len(xg):
                                    continue
                                    
                                x_interp = np.linspace(xg[0], xg[-1], 100)
                                y_interp = np.interp(x_interp, xg, yg)
                                upper_bound = np.interp(x_interp, xg, yg + yerrg)
                                lower_bound = np.interp(x_interp, xg, yg - yerrg)
                                
                                ax.fill_between(x_interp, lower_bound, upper_bound, color=color, alpha=0.2)
                    
                    if np.any(valid_mask_right):
                        x_valid = x[valid_mask_right]
                        y_valid = y_right[valid_mask_right]
                        yerr_valid = yerr_right[valid_mask_right]
                        
                        ax.plot(x_valid, y_valid, '^', label=f'Peak {peak_idx+1} Right', color=color, alpha=0.7)
                        
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
                                
                                if len(yerrg) != len(xg):
                                    continue
                                    
                                x_interp = np.linspace(xg[0], xg[-1], 100)
                                y_interp = np.interp(x_interp, xg, yg)
                                upper_bound = np.interp(x_interp, xg, yg + yerrg)
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
        """Export the batch processing results to a CSV file."""
        if not self.batch_results:
            messagebox.showwarning("No Results", "No batch processing results to export.")
            return
            
        try:
            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Save Batch Results"
            )
            
            if not file_path:
                return
                
            # Prepare data for export
            data = []
            for result in self.batch_results:
                # Start with basic file info
                row = {'File': os.path.basename(result['file'])}
                
                # Check if this was a failed fit
                fit_failed = result.get('fit_failed', False)
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
            
            messagebox.showinfo("Export Complete", f"Results saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
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
            messagebox.showerror("Error", f"Failed to subtract background: {str(e)}")
    
    def find_peaks(self):
        """Find peaks in the current spectrum."""
        try:
            # Get parameters
            height = self.var_height.get()
            distance = self.var_distance.get()
            prominence = self.var_prominence.get()
            
            # Convert parameters to float if not "Auto" - use safe conversion
            try:
                height = float(height) if height != "Auto" else None
            except (ValueError, TypeError):
                height = None
                
            try:
                distance = int(float(distance)) if distance != "Auto" else None
            except (ValueError, TypeError):
                distance = None
                
            try:
                prominence = float(prominence) if prominence != "Auto" else None
            except (ValueError, TypeError):
                prominence = None
            
            # Get a copy of the spectrum data to avoid modifying the original
            spectra_data = np.copy(self.spectra)
            
            # Ensure we have valid data
            if spectra_data is None or len(spectra_data) == 0:
                messagebox.showerror("Error", "No spectrum data available")
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
                messagebox.showerror("Error", f"Failed to execute scipy.signal.find_peaks: {str(e)}")
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
            messagebox.showerror("Error", f"Failed to find peaks: {str(e)}")
    
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
                messagebox.showwarning("No Peaks", "Please detect peaks first.")
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
                    messagebox.showerror("Fit Range Error", f"Could not parse fit ranges: {fit_ranges_str}\nError: {e}")
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
                messagebox.showwarning("No Peaks in ROI", "No peaks found within the specified ROI ranges.")
                self.fit_failed = True
                return
            
            # HARD CAP for Asymmetric Voigt
            if model_type == "Asymmetric Voigt" and len(peaks_in_roi) > 8:
                messagebox.showerror("Too Many Peaks", "Asymmetric Voigt fitting is limited to 8 peaks for stability. Please reduce the number of detected peaks (e.g., by increasing the prominence or height threshold).")
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
            
            # Prepare initial parameters
            initial_params = []
            bounds_lower = []
            bounds_upper = []
            
            for peak in peaks_in_roi:
                # Initial amplitude
                initial_params.append(peak['intensity'])
                bounds_lower.append(0)
                bounds_upper.append(np.inf)
                
                # Initial position
                initial_params.append(peak['position'])
                bounds_lower.append(self.wavenumbers[0])
                bounds_upper.append(self.wavenumbers[-1])
                
                if model_type == "Gaussian" or model_type == "Lorentzian":
                    # Initial width
                    initial_params.append(10)  # Default width
                    bounds_lower.append(0.1)
                    bounds_upper.append(100)
                elif model_type == "Pseudo-Voigt":
                    # Initial width and eta
                    initial_params.append(10)  # Default width
                    initial_params.append(0.5)  # Default eta
                    bounds_lower.extend([0.1, 0])
                    bounds_upper.extend([100, 1])
                elif model_type == "Asymmetric Voigt":
                    # Initial left width, right width, eta
                    initial_params.append(10)  # Default left width (sigma_l)
                    initial_params.append(10)  # Default right width (sigma_r)
                    initial_params.append(0.5)  # Default eta
                    bounds_lower.extend([0.1, 0.1, 0])
                    bounds_upper.extend([100, 100, 1])

            # Check initial parameter count for Asymmetric Voigt
            if model_type == "Asymmetric Voigt" and len(initial_params) % 5 != 0:
                messagebox.showerror("Parameter Error", f"Initial parameter list for Asymmetric Voigt is not a multiple of 5 (got {len(initial_params)}). This indicates a bug or mismatch. Please clear peaks and try again.")
                self.fit_failed = True
                return
            
            # Define combined model function
            def combined_model(x, *params):
                result = np.zeros_like(x)
                for i in range(0, len(params), params_per_peak):
                    # Make sure we have enough parameters for this peak
                    if i + params_per_peak <= len(params):
                        peak_params = params[i:i+params_per_peak]
                        try:
                            # Add the contribution of this peak to the result
                            peak_result = model_func(x, *peak_params)
                            result = result + peak_result  # Use explicit addition instead of +=
                        except Exception as e:
                            print(f"Error in model_func at i={i} with params {peak_params}: {e}")
                            continue
                    else:
                        print(f"Skipping peak at i={i}, not enough parameters left")
                return result
            
            # Add a counter for function evaluations (NLLS cycles)
            self.func_eval_count = 0
            
            # Wrap the model function to count evaluations
            def counting_model(x, *params):
                self.func_eval_count += 1
                return combined_model(x, *params)
            
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
                    self.fit_params = popt
                    self.fit_cov = pcov
                    self.nlls_cycles = 0  # No successful cycles
                    
                    # Try to calculate fit result and residuals with initial parameters
                    try:
                        self.fit_result = combined_model(self.wavenumbers, *popt)
                        self.residuals = self.spectra - self.fit_result
                    except Exception as e3:
                        print(f"Failed to calculate fit with initial parameters: {str(e3)}")
                        # If even that fails, set empty arrays
                        self.fit_result = np.zeros_like(self.wavenumbers)
                        self.residuals = np.copy(self.spectra)
                    
                    self.fit_failed = True
                    messagebox.showwarning("Fit Failed", f"Failed to fit peaks after two attempts. Using initial guess for this spectrum.\n\nFirst error: {str(e1)}\nSecond error: {str(e2)}")
                    self.update_plot()
                    self.update_peak_visibility_controls()
                    return
                    
            # Robust check for Asymmetric Voigt
            if model_type == "Asymmetric Voigt":
                if len(popt) % 5 != 0:
                    messagebox.showerror("Fit Error", f"Fit failed: number of fit parameters ({len(popt)}) is not a multiple of 5. Try reducing the number of peaks or adjusting initial guesses.")
                    self.fit_failed = True
                    return
                    
            # Store results
            self.fit_params = popt
            self.fit_cov = pcov
            
            # Calculate fit result and residuals
            try:
                self.fit_result = combined_model(self.wavenumbers, *popt)
                self.residuals = np.subtract(self.spectra, self.fit_result)  # Use np.subtract to avoid array issues
                self.fit_failed = False
            except Exception as e:
                messagebox.showerror("Error", f"Failed to calculate fit results: {str(e)}")
                self.fit_failed = True
                return
            
            # Update plot
            self.update_plot()
            self.update_peak_visibility_controls()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fit peaks: {str(e)}")
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
        
        # Plot spectrum
        self.ax1_current.plot(self.wavenumbers, self.spectra, 'k-', label='Spectrum')
        
        # Plot background if available
        if hasattr(self, 'background') and self.background is not None:
            self.ax1_current.plot(self.wavenumbers, self.background, 'b--', label='Background')
        
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
                    
                    # Calculate R² values for each peak using the full fit result
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
            messagebox.showerror("Error", f"Failed to update ROI regions: {str(e)}")

    def enable_peak_addition(self):
        """Enable mode to add peaks by clicking on the plot. Stays active until Esc is pressed."""
        self.adding_peak = True
        self.canvas_current.mpl_connect('button_press_event', self.add_peak_on_click)
        self.canvas_current.mpl_connect('key_press_event', self._on_peak_add_key)
        self.ax1_current.set_title('Click to add peak. Press Esc to exit peak add mode.')
        self.canvas_current.draw()

    def _on_peak_add_key(self, event):
        if event.key == 'escape':
            self.adding_peak = False
            self.canvas_current.mpl_disconnect(self.canvas_current.mpl_connect('button_press_event', self.add_peak_on_click))
            self.canvas_current.mpl_disconnect(self.canvas_current.mpl_connect('key_press_event', self._on_peak_add_key))
            self.ax1_current.set_title('Raman Spectrum')
            self.canvas_current.draw()
            
    def add_peak_on_click(self, event):
        if not getattr(self, 'adding_peak', False):
            return
        if event.button != 1:
            return
        if event.inaxes != self.ax1_current:
            return
        try:
            position = event.xdata
            amplitude = event.ydata
            idx = np.argmin(np.abs(self.wavenumbers - position))
            new_peak = {
                'position': position,
                'intensity': amplitude,
                'index': idx
            }
            self.peaks.append(new_peak)
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add peak: {str(e)}")
    
    def show_peak_deletion_dialog(self):
        """Show a dialog to select peaks for deletion."""
        if not self.peaks:
            messagebox.showinfo("No Peaks", "No peaks to delete.")
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
                messagebox.showinfo("No Selection", "Please select peaks to delete.")
                return
                
            # Confirm deletion
            if messagebox.askyesno("Confirm Delete", 
                                 f"Delete {len(selected_indices)} selected peak(s)?"):
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
                data = np.loadtxt(file_path)
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
            messagebox.showerror("Error", f"Failed to update waterfall plot: {str(e)}")

    def reset_heatmap_adjustments(self):
        """Reset all heatmap color adjustments to default values."""
        self.heatmap_contrast.set(1.0)
        self.heatmap_brightness.set(1.0)
        self.heatmap_gamma.set(1.0)
        # Grid remains at its current setting when resetting colors
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
                data = np.loadtxt(file_path)
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
            spectra_array = np.array(all_spectra)
            
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
            if self._heatmap_colorbar is not None:
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
            messagebox.showerror("Error", f"Failed to update heatmap plot: {str(e)}")

    def calculate_r_squared(self, y_true, y_pred):
        """Calculate R-squared value for a fit using safe NumPy operations."""
        try:
            # Convert inputs to numpy arrays
            y_true_array = np.asarray(y_true)
            y_pred_array = np.asarray(y_pred)
            
            # Make sure arrays have the same shape
            if y_true_array.shape != y_pred_array.shape:
                print(f"Array shape mismatch: {y_true_array.shape} vs {y_pred_array.shape}")
                return 0.0
                
            # Calculate mean of true values
            y_mean = np.mean(y_true_array)
            
            # Calculate sum of squared residuals
            ss_res = np.sum(np.square(np.subtract(y_true_array, y_pred_array)))
            
            # Calculate total sum of squares
            ss_tot = np.sum(np.square(np.subtract(y_true_array, y_mean)))
            
            # Safely calculate R-squared
            if ss_tot > 0:
                r2 = 1.0 - (ss_res / ss_tot)
                return float(r2)  # Ensure we return a scalar
            else:
                return 0.0  # Default if ss_tot is zero
                
        except Exception as e:
            print(f"Error in calculate_r_squared: {str(e)}")
            return 0.0

    def calculate_peak_r_squared(self, y_true, y_fit_total, peak_params, model_type):
        """Calculate R-squared value for individual peaks."""
        peak_r_squared = []
        
        # Check if peaks or parameters are empty
        if not self.peaks or len(peak_params) == 0:
            return peak_r_squared
        
        # Determine number of parameters per peak based on model type
        if model_type == "Gaussian" or model_type == "Lorentzian":
            params_per_peak = 3
        elif model_type == "Pseudo-Voigt":
            params_per_peak = 4
        elif model_type == "Asymmetric Voigt":
            params_per_peak = 5
        else:
            params_per_peak = 3
            
        n_peaks = len(peak_params) // params_per_peak
        
        # Continue with calculation for each peak
        for peak_idx in range(n_peaks):
            try:
                # Get start index for this peak's parameters
                i = peak_idx * params_per_peak
                
                # Extract peak parameters
                if model_type == "Gaussian":
                    peak_y = self.gaussian(self.wavenumbers, *peak_params[i:i+3])
                    center = peak_params[i+1]
                    width = peak_params[i+2]
                    # FWHM for Gaussian
                    fwhm = 2.355 * width
                elif model_type == "Lorentzian":
                    peak_y = self.lorentzian(self.wavenumbers, *peak_params[i:i+3])
                    center = peak_params[i+1]
                    width = peak_params[i+2]
                    # FWHM for Lorentzian
                    fwhm = 2 * width
                elif model_type == "Pseudo-Voigt":
                    peak_y = self.pseudo_voigt(self.wavenumbers, *peak_params[i:i+4])
                    center = peak_params[i+1]
                    width = peak_params[i+2]
                    # Approximate FWHM for Pseudo-Voigt
                    fwhm = 2 * width
                elif model_type == "Asymmetric Voigt":
                    peak_y = self.asymmetric_voigt(self.wavenumbers, *peak_params[i:i+5])
                    center = peak_params[i+1]
                    width_left = peak_params[i+2]
                    width_right = peak_params[i+3]
                    # Average FWHM for asymmetric peak
                    fwhm = width_left + width_right
                else:
                    peak_y = self.gaussian(self.wavenumbers, *peak_params[i:i+3])
                    center = peak_params[i+1]
                    width = peak_params[i+2]
                    fwhm = 2.355 * width
                
                # Create a mask for a region around the peak (±2*FWHM)
                region_width = 2 * fwhm
                min_bound = center - region_width
                max_bound = center + region_width
                
                # Create mask indices for the peak region
                mask_indices = np.where(
                    np.logical_and(
                        np.greater_equal(self.wavenumbers, min_bound),
                        np.less_equal(self.wavenumbers, max_bound)
                    )
                )[0]
                
                # If mask is too narrow, widen it to ensure enough points
                if len(mask_indices) < 10:
                    region_width = 3 * fwhm
                    min_bound = center - region_width
                    max_bound = center + region_width
                    mask_indices = np.where(
                        np.logical_and(
                            np.greater_equal(self.wavenumbers, min_bound),
                            np.less_equal(self.wavenumbers, max_bound)
                        )
                    )[0]
                
                # Calculate contribution of this peak to the total fit
                if len(mask_indices) > 3:  # Need at least a few points for meaningful R²
                    # Get data, total fit, and individual peak in the region
                    y_true_region = y_true[mask_indices]
                    y_fit_region = y_fit_total[mask_indices]
                    peak_y_region = peak_y[mask_indices]
                    
                    # Calculate peak's contribution to the fit in this region
                    peak_contribution = np.sum(peak_y_region) / np.sum(y_fit_region) if np.sum(y_fit_region) > 0 else 0
                    
                    # Only consider regions where this peak is significant
                    if peak_contribution > 0.2:  # At least 20% contribution
                        # Calculate R² based on how well the total fit matches data in this region
                        ss_res = np.sum((y_true_region - y_fit_region) ** 2)
                        ss_tot = np.sum((y_true_region - np.mean(y_true_region)) ** 2)
                        
                        if ss_tot > 0:
                            r2 = 1 - (ss_res / ss_tot)
                            # Weight R² by peak's contribution to the total fit in this region
                            # r2 = r2 * peak_contribution
                        else:
                            r2 = 0.0
                    else:
                        # Peak has minimal contribution here
                        r2 = 0.0
                else:
                    r2 = 0.0
                
                peak_r_squared.append(r2)
            except Exception as e:
                # If any error occurs, add a default value
                print(f"Error calculating R² for peak {peak_idx}: {e}")
                peak_r_squared.append(0.0)
        
        return peak_r_squared

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

        # Calculate R² for each spectrum in batch results
        r2_values = []
        for i, result in enumerate(self.batch_results):
            if result.get('fit_failed', True) or result['fit_result'] is None or result['fit_params'] is None:
                r2_values.append((i, 0))  # Default to 0 for failed fits
                continue
                
            # Load the original spectrum data from file
            try:
                data = np.loadtxt(result['file'])
                wavenumbers = data[:, 0]
                original_spectrum = data[:, 1]
                
                # Get the background-subtracted spectrum used for fitting
                if result['background'] is not None:
                    background = result['background']
                    spectrum = original_spectrum - background
                else:
                    spectrum = original_spectrum
                
                # Parse fit ranges to only calculate R² within the ROI
                fit_ranges_str = self.var_fit_ranges.get().strip()
                roi_ranges = []
                if fit_ranges_str:
                    try:
                        for part in fit_ranges_str.split(','):
                            if '-' in part:
                                min_w, max_w = map(float, part.split('-'))
                                roi_ranges.append((min_w, max_w))
                    except:
                        # If parsing fails, use the entire range
                        pass
                
                # Create mask for ROI ranges
                if roi_ranges:
                    mask = np.zeros_like(wavenumbers, dtype=bool)
                    for min_w, max_w in roi_ranges:
                        new_mask = (wavenumbers >= min_w) & (wavenumbers <= max_w)
                        mask = mask | new_mask
                    
                    # Only use points within ROI for R² calculation
                    spectrum_roi = spectrum[mask]
                    fit_roi = result['fit_result'][mask]
                else:
                    # Use entire spectrum if no ROI specified
                    spectrum_roi = spectrum
                    fit_roi = result['fit_result']
                
                # Calculate R² only for the spectrum used for fitting (background-subtracted)
                # and only within the ROI where fitting was performed
                if len(spectrum_roi) > 0:
                    mean_data = np.mean(spectrum_roi)
                    ss_tot = np.sum((spectrum_roi - mean_data) ** 2)
                    ss_res = np.sum((spectrum_roi - fit_roi) ** 2)
                    
                    if ss_tot > 0:
                        r2 = 1 - (ss_res / ss_tot)
                    else:
                        r2 = 0
                else:
                    r2 = 0
                
                r2_values.append((i, r2))
            except Exception as e:
                print(f"Error calculating R² for spectrum {i}: {e}")
                r2_values.append((i, 0))
        
        # Sort by R² value
        r2_values.sort(key=lambda x: x[1])
        
        # Get the worst, median, and best spectra indices
        n_results = len(r2_values)
        if n_results < 9:
            # If fewer than 9 results, use all available
            indices_to_show = [x[0] for x in r2_values]
        else:
            # Get worst 3 (lowest R²)
            worst_indices = [x[0] for x in r2_values[:3]]
            
            # Get best 3 (highest R²)
            best_indices = [x[0] for x in r2_values[-3:]]
            best_indices.reverse()  # Show highest R² first
            
            # Get median 3
            median_start = n_results // 2 - 1
            median_indices = [x[0] for x in r2_values[median_start:median_start+3]]
            
            # Combine in order: best, median, worst
            indices_to_show = best_indices + median_indices + worst_indices
        
        # Plot each selected spectrum
        for i, plot_idx in enumerate(indices_to_show[:9]):  # Limit to 9 plots (3x3 grid)
            row = i // 3
            col = i % 3
            
            # Get result for this spectrum
            result = self.batch_results[plot_idx]
            
            try:
                # Load the original data
                data = np.loadtxt(result['file'])
                wavenumbers = data[:, 0]
                original_spectrum = data[:, 1]
                
                # Get filename for title
                filename = os.path.basename(result['file'])
                
                # Plot original spectrum
                self.ax_stats[row, col].plot(wavenumbers, original_spectrum, 'k-', label='Data')
                
                # Plot background if available
                if result['background'] is not None:
                    background = result['background']
                    self.ax_stats[row, col].plot(wavenumbers, background, 'b--', label='Background')
                    
                    # Plot background-subtracted spectrum
                    spectrum = original_spectrum - background
                    self.ax_stats[row, col].plot(wavenumbers, spectrum, 'g-', label='Subtracted', alpha=0.5)
                    
                    # Also plot the fit result on top of background-subtracted data
                    if result['fit_result'] is not None:
                        # Get correct R² value for this spectrum
                        r2 = 0
                        for idx, r2_val in r2_values:
                            if idx == plot_idx:
                                r2 = r2_val
                                break
                        
                        # Highlight ROI regions if specified
                        fit_ranges_str = self.var_fit_ranges.get().strip()
                        roi_ranges = []
                        if fit_ranges_str:
                            try:
                                for part in fit_ranges_str.split(','):
                                    if '-' in part:
                                        min_w, max_w = map(float, part.split('-'))
                                        roi_ranges.append((min_w, max_w))
                                        # Add shaded region for ROI
                                        self.ax_stats[row, col].axvspan(min_w, max_w, color='lightgrey', alpha=0.3, zorder=0)
                            except:
                                pass
                        
                        # Plot the fit on top of background-subtracted data
                        self.ax_stats[row, col].plot(wavenumbers, result['fit_result'], 'r-', label='Fit')
                        
                        # Set title with filename, spectrum number and R²
                        quality_label = "Best" if row == 0 else "Median" if row == 1 else "Worst"
                        self.ax_stats[row, col].set_title(f"{quality_label}: {filename}\nR²={r2:.3f}", fontsize=9)
                    else:
                        self.ax_stats[row, col].set_title(f"Spectrum {plot_idx+1} (No fit)", fontsize=9)
                else:
                    # No background case
                    if result['fit_result'] is not None:
                        r2 = 0
                        for idx, r2_val in r2_values:
                            if idx == plot_idx:
                                r2 = r2_val
                                break
                        self.ax_stats[row, col].plot(wavenumbers, result['fit_result'], 'r-', label='Fit')
                        quality_label = "Best" if row == 0 else "Median" if row == 1 else "Worst"
                        self.ax_stats[row, col].set_title(f"{quality_label}: {filename}\nR²={r2:.3f}", fontsize=9)
                    else:
                        self.ax_stats[row, col].set_title(f"Spectrum {plot_idx+1} (No fit)", fontsize=9)
                
                # Add grid
                self.ax_stats[row, col].grid(True, linestyle=':', color='gray', alpha=0.4)
                
                # Set axis labels only for edge plots
                if col == 0:
                    self.ax_stats[row, col].set_ylabel('Intensity')
                if row == 2:
                    self.ax_stats[row, col].set_xlabel('Wavenumber (cm⁻¹)')
                
                # Add legend for first plot only (to save space)
                if row == 0 and col == 0:
                    self.ax_stats[row, col].legend(fontsize=8, loc='upper right')
                
            except Exception as e:
                print(f"Error plotting spectrum {plot_idx}: {e}")
                self.ax_stats[row, col].text(0.5, 0.5, f"Error loading spectrum {plot_idx+1}", 
                                           ha='center', va='center', transform=self.ax_stats[row, col].transAxes)
        
        # Adjust layout
        self.fig_stats.tight_layout()
        self.canvas_stats.draw()

    def export_fit_stats_plot(self):
        """Export the Fit Stats plot to an image file."""
        try:
            file_path = filedialog.asksaveasfilename(
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
                # Get the file extension
                ext = os.path.splitext(file_path)[1].lower()
                
                # Create a new figure with the same size
                fig = plt.figure(figsize=(12, 12))
                axes = fig.subplots(3, 3)
                
                # Copy all content from the original figure
                for i, ax in enumerate(self.ax_stats.flatten()):
                    if not ax.has_data():
                        axes.flatten()[i].axis('off')
                        continue
                        
                    # Copy all lines
                    for line in ax.get_lines():
                        axes.flatten()[i].plot(line.get_xdata(), line.get_ydata(),
                                            color=line.get_color(),
                                            linestyle=line.get_linestyle(),
                                            linewidth=line.get_linewidth(),
                                            alpha=line.get_alpha(),
                                            label=line.get_label())
                    
                    # Copy background patches (ROI regions)
                    for patch in ax.patches:
                        if isinstance(patch, mpatches.Rectangle):
                            axes.flatten()[i].add_patch(plt.Rectangle(
                                patch.get_xy(),
                                patch.get_width(),
                                patch.get_height(),
                                color=patch.get_facecolor(),
                                alpha=patch.get_alpha()
                            ))
                    
                    # Copy text elements with consistent positioning
                    for text in ax.texts:
                        bbox = text.get_bbox_patch()
                        bbox_props = {
                            'facecolor': bbox.get_facecolor(),
                            'edgecolor': bbox.get_edgecolor(),
                            'alpha': bbox.get_alpha(),
                            'boxstyle': 'round,pad=0.3'
                        }
                        # Position text in upper right corner
                        axes.flatten()[i].text(0.98, 0.98, text.get_text(),
                                            transform=axes.flatten()[i].transAxes,
                                            verticalalignment='top',
                                            horizontalalignment='right',
                                            fontsize=text.get_fontsize(),
                                            bbox=bbox_props)
                    
                    # Copy title and labels
                    axes.flatten()[i].set_title(ax.get_title(), fontsize=9, pad=5)
                    axes.flatten()[i].set_xlabel(ax.get_xlabel())
                    axes.flatten()[i].set_ylabel(ax.get_ylabel())
                    
                    # Copy grid
                    axes.flatten()[i].grid(True, linestyle=':', color='gray', alpha=0.4)
                    
                    # Copy legend
                    if ax.get_legend():
                        axes.flatten()[i].legend(loc='upper right', fontsize=7, framealpha=0.8)
                    
                    # Set axis limits
                    axes.flatten()[i].set_xlim(ax.get_xlim())
                    axes.flatten()[i].set_ylim(ax.get_ylim())
                    
                    # Set tick parameters
                    axes.flatten()[i].tick_params(axis='both', which='major', labelsize=8)
                
                # Adjust layout
                fig.tight_layout(pad=2.0)
                
                # Save with high DPI for better quality
                fig.savefig(
                    file_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none'
                )
                
                # Close the temporary figure
                plt.close(fig)
                
                messagebox.showinfo("Success", f"Plot saved successfully to:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot: {str(e)}")

    def reset_heatmap_range(self):
        """Reset the X-axis range to full range."""
        if hasattr(self, 'wavenumbers') and self.wavenumbers is not None:
            self.heatmap_xmin.delete(0, tk.END)
            self.heatmap_xmax.delete(0, tk.END)
            self.heatmap_xmin.insert(0, f"{self.wavenumbers[0]:.1f}")
            self.heatmap_xmax.insert(0, f"{self.wavenumbers[-1]:.1f}")
            self.update_heatmap_plot()

    def reset_waterfall_range(self):
        """Reset the X-axis range to full range."""
        if hasattr(self, 'wavenumbers') and self.wavenumbers is not None:
            self.waterfall_xmin.delete(0, tk.END)
            self.waterfall_xmax.delete(0, tk.END)
            self.waterfall_xmin.insert(0, f"{self.wavenumbers[0]:.1f}")
            self.waterfall_xmax.insert(0, f"{self.wavenumbers[-1]:.1f}")
            # Reset Y offset to default (absolute units)
            self.waterfall_yoffset.set(100)
            self.update_waterfall_plot()

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

    def export_individual_fit_stats(self):
        """Export individual fit statistics plots to separate image files."""
        try:
            # Ask for directory to save files
            save_dir = filedialog.askdirectory(title="Select Directory to Save Individual Plots")
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
                
            messagebox.showinfo("Export Complete", f"Individual plots saved to {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export individual plots: {str(e)}")
            
    def export_curve_data_csv(self):
        """Export all peak curve data to a CSV file."""
        if not self.batch_results:
            messagebox.showwarning("No Results", "No batch processing results to export.")
            return
            
        try:
            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Save Peak Curve Data"
            )
            
            if not file_path:
                return
                
            # Get data from extract_trend_data which already has all the peak information we need
            x, trends, n_peaks, uncertainties = self.extract_trend_data()
            if x is None or trends is None or n_peaks is None:
                messagebox.showwarning("No Data", "No trend data available to export.")
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
            
            messagebox.showinfo("Export Complete", f"Peak curve data saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export curve data: {str(e)}")

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
        except:
            # Default status message if something goes wrong
            self.status_bar.config(text="Press F1 for help with the current tab")

    def show_general_help(self):
        """Show general help for the application."""
        title, text = self.help_texts["General"]
        self.show_help_dialog(title, text)