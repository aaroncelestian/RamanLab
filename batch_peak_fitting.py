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
from tkinter import ttk, messagebox, filedialog
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
        self.window.geometry("1400x800")
        self.window.minsize(1200, 700)
        
        # Store references
        self.parent = parent
        self.raman = raman
        
        # Initialize data storage
        self.spectra_files = []
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
        
        # Create GUI
        self.create_gui()
        
        # Set up window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
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

        # --- Tab 1: File Selection ---
        file_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(file_tab, text="File Selection")

        file_frame = ttk.LabelFrame(file_tab, text="File Selection", padding=10)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
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

        # --- Tab 4: Peak Visibility ---
        visibility_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(visibility_tab, text="Plot")

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
        export_frame.pack(fill=tk.X, pady=40)
        
        ttk.Button(export_frame, text="Save Individual Stats Plots", 
                  command=self.export_individual_fit_stats).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Save All Stats Plots", 
                  command=self.export_fit_stats_plot).pack(fill=tk.X, pady=2)




        # --- RIGHT PANEL: Visualization ---
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.viz_notebook = ttk.Notebook(right_panel)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)

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
        
        # Skip and color controls on one line
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
        
        ttk.Label(basic_controls, text="Line Width:").pack(side=tk.LEFT, padx=5)
        self.waterfall_linewidth = tk.DoubleVar(value=1.5)
        self.waterfall_linewidth_spin = ttk.Spinbox(basic_controls, from_=0.5, to=5.0, increment=0.1, 
                                                  textvariable=self.waterfall_linewidth, width=4)
        self.waterfall_linewidth_spin.pack(side=tk.LEFT, padx=5)
        
        # Add checkboxes for legend and grid
        self.waterfall_show_legend = tk.BooleanVar(value=True)
        self.waterfall_show_grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(basic_controls, text="Legend", variable=self.waterfall_show_legend, 
                       command=self.update_waterfall_plot).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(basic_controls, text="Grid", variable=self.waterfall_show_grid, 
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
            # Automatically load and plot the first file if this is the first time files are added
            if was_empty and self.spectra_files:
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
        
        if direction == 0:  # First
            new_index = 0
        elif direction == -1:  # Previous
            new_index = max(0, self.current_spectrum_index - 1)
        elif direction == 1:  # Next
            new_index = min(len(self.spectra_files) - 1, self.current_spectrum_index + 1)
        else:  # Last
            new_index = len(self.spectra_files) - 1
        
        if new_index != self.current_spectrum_index:
            self.load_spectrum(new_index)
    
    def load_spectrum(self, index):
        """Load a spectrum from the list."""
        if not self.spectra_files or index < 0 or index >= len(self.spectra_files):
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
            
            # Initialize peak fitting variables
            self.peaks = []
            self.fit_params = []
            self.fit_result = None
            self.background = None
            self.residuals = None
            
            # Update the plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load spectrum: {str(e)}")
    
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
        
        # Process all spectra
        for i in range(len(self.spectra_files)):
            if self._stop_batch:
                messagebox.showinfo("Stopped", "Batch processing was stopped by the user.")
                break
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
                messagebox.showerror("Error", f"Failed to subtract background for spectrum {i+1}: {str(e)}")
                continue
            
            # Fit peaks
            self.fit_peaks()
            
            # Store results
            self.batch_results.append({
                'file': self.spectra_files[i],
                'peaks': self.peaks.copy(),
                'fit_params': np.copy(self.fit_params) if self.fit_params is not None else None,
                'fit_cov': np.copy(self.fit_cov) if hasattr(self, 'fit_cov') and self.fit_cov is not None else None,
                'background': np.copy(self.background) if self.background is not None else None,
                'fit_result': np.copy(self.fit_result) if self.fit_result is not None else None
            })
            
            # Update progress
            self.current_spectrum_label.config(text=f"Processing: {i+1}/{len(self.spectra_files)}")
            self.window.update()
        
        # Update peak visibility controls and trends plot
        self.update_peak_visibility_controls()
        self.update_trends_plot()
        self.update_fit_stats_plot()  # Add this line to update the Fit Stats plot
        
        if not self._stop_batch:
            messagebox.showinfo("Complete", "Batch processing completed.")
    
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
            fit_params = result.get('fit_params')
            fit_cov = result.get('fit_cov')  # Get covariance matrix
            if fit_params is not None:
                if n_peaks is None:
                    n_peaks = len(fit_params) // params_per_peak
                all_params.append(np.array(fit_params).reshape(-1, params_per_peak))
                all_covs.append(fit_cov)  # Store covariance matrix
            else:
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
                if params is not None and peak_idx < params.shape[0]:
                    if model_type in ["Gaussian", "Lorentzian"]:
                        amp, pos, wid = params[peak_idx]
                        amp_vals.append(amp)
                        pos_vals.append(pos)
                        wid_vals.append(wid)
                        # Calculate uncertainties from covariance matrix
                        if cov is not None:
                            start_idx = peak_idx * params_per_peak
                            pos_errs.append(1.96 * np.sqrt(cov[start_idx+1, start_idx+1]))  # 95% confidence interval
                            amp_errs.append(1.96 * np.sqrt(cov[start_idx, start_idx]))
                            wid_errs.append(1.96 * np.sqrt(cov[start_idx+2, start_idx+2]))
                        else:
                            pos_errs.append(None)
                            amp_errs.append(None)
                            wid_errs.append(None)
                    elif model_type == "Pseudo-Voigt":
                        amp, pos, wid, eta = params[peak_idx]
                        amp_vals.append(amp)
                        pos_vals.append(pos)
                        wid_vals.append(wid)
                        eta_vals.append(eta)
                        if cov is not None:
                            start_idx = peak_idx * params_per_peak
                            pos_errs.append(1.96 * np.sqrt(cov[start_idx+1, start_idx+1]))
                            amp_errs.append(1.96 * np.sqrt(cov[start_idx, start_idx]))
                            wid_errs.append(1.96 * np.sqrt(cov[start_idx+2, start_idx+2]))
                            eta_errs.append(1.96 * np.sqrt(cov[start_idx+3, start_idx+3]))
                        else:
                            pos_errs.append(None)
                            amp_errs.append(None)
                            wid_errs.append(None)
                            eta_errs.append(None)
                    elif model_type == "Asymmetric Voigt":
                        amp, pos, wid_left, wid_right, eta = params[peak_idx]
                        amp_vals.append(amp)
                        pos_vals.append(pos)
                        wid_left_vals.append(wid_left)
                        wid_right_vals.append(wid_right)
                        eta_vals.append(eta)
                        if cov is not None:
                            start_idx = peak_idx * params_per_peak
                            pos_errs.append(1.96 * np.sqrt(cov[start_idx+1, start_idx+1]))
                            amp_errs.append(1.96 * np.sqrt(cov[start_idx, start_idx]))
                            wid_left_errs.append(1.96 * np.sqrt(cov[start_idx+2, start_idx+2]))
                            wid_right_errs.append(1.96 * np.sqrt(cov[start_idx+3, start_idx+3]))
                            eta_errs.append(1.96 * np.sqrt(cov[start_idx+4, start_idx+4]))
                        else:
                            pos_errs.append(None)
                            amp_errs.append(None)
                            wid_left_errs.append(None)
                            wid_right_errs.append(None)
                            eta_errs.append(None)
                else:
                    if model_type == "Asymmetric Voigt":
                        amp_vals.append(np.nan)
                        pos_vals.append(np.nan)
                        wid_left_vals.append(np.nan)
                        wid_right_vals.append(np.nan)
                        eta_vals.append(np.nan)
                        pos_errs.append(None)
                        amp_errs.append(None)
                        wid_left_errs.append(None)
                        wid_right_errs.append(None)
                        eta_errs.append(None)
                    elif model_type == "Pseudo-Voigt":
                        amp_vals.append(np.nan)
                        pos_vals.append(np.nan)
                        wid_vals.append(np.nan)
                        eta_vals.append(np.nan)
                        pos_errs.append(None)
                        amp_errs.append(None)
                        wid_errs.append(None)
                        eta_errs.append(None)
                    else:
                        amp_vals.append(np.nan)
                        pos_vals.append(np.nan)
                        wid_vals.append(np.nan)
                        pos_errs.append(None)
                        amp_errs.append(None)
                        wid_errs.append(None)
            trends.append({
                'pos': pos_vals,
                'amp': amp_vals,
                'wid': wid_vals,
                'eta': eta_vals,
                'wid_left': wid_left_vals,
                'wid_right': wid_right_vals
            })
            uncertainties.append({
                'pos': pos_errs,
                'amp': amp_errs,
                'wid': wid_errs,
                'eta': eta_errs,
                'wid_left': wid_left_errs,
                'wid_right': wid_right_errs
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

        for peak_idx in visible_peaks:
            color = f"C{peak_idx}"
            # Position
            y = np.array(trends[peak_idx]['pos'])
            yerr = np.array(uncertainties[peak_idx]['pos'])
            axes[0].plot(x, y, 'o', label=f'Peak {peak_idx+1}', color=color)
            if show_95:
                axes[0].fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
            # Amplitude
            y = np.array(trends[peak_idx]['amp'])
            yerr = np.array(uncertainties[peak_idx]['amp'])
            axes[1].plot(x, y, 'o', label=f'Peak {peak_idx+1}', color=color)
            if show_95:
                axes[1].fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
            # Width
            if model_type == "Asymmetric Voigt":
                y_left = np.array(trends[peak_idx]['wid_left'])
                y_right = np.array(trends[peak_idx]['wid_right'])
                yerr_left = np.array(uncertainties[peak_idx]['wid_left'])
                yerr_right = np.array(uncertainties[peak_idx]['wid_right'])
                axes[2].plot(x, y_left, 'o', label=f'Peak {peak_idx+1} Left', color=color, alpha=0.7)
                axes[2].plot(x, y_right, '^', label=f'Peak {peak_idx+1} Right', color=color, alpha=0.7)
                if show_95:
                    axes[2].fill_between(x, y_left - yerr_left, y_left + yerr_left, color=color, alpha=0.2)
                    axes[2].fill_between(x, y_right - yerr_right, y_right + yerr_right, color=color, alpha=0.2)
            else:
                y = np.array(trends[peak_idx]['wid'])
                yerr = np.array(uncertainties[peak_idx]['wid'])
                axes[2].plot(x, y, 'o', label=f'Peak {peak_idx+1}', color=color)
                if show_95:
                    axes[2].fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
            # Eta
            if model_type in ["Pseudo-Voigt", "Asymmetric Voigt"]:
                y = np.array(trends[peak_idx]['eta'])
                yerr = np.array(uncertainties[peak_idx]['eta'])
                axes[3].plot(x, y, 'o', label=f'Peak {peak_idx+1}', color=color)
                if show_95:
                    axes[3].fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

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
        for peak_idx in visible_peaks:
            color = f"C{peak_idx}"
            if which == 'position':
                y = np.array(trends[peak_idx]['pos'])
                yerr = np.array(uncertainties[peak_idx]['pos'])
                ax.plot(x, y, 'o', label=f'Peak {peak_idx+1}', color=color)
                if show_95:
                    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
                ax.set_title('Peak Position')
                ax.set_ylabel('Position (cm⁻¹)')
                plotted = True
            elif which == 'amplitude':
                y = np.array(trends[peak_idx]['amp'])
                yerr = np.array(uncertainties[peak_idx]['amp'])
                ax.plot(x, y, 'o', label=f'Peak {peak_idx+1}', color=color)
                if show_95:
                    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
                ax.set_title('Peak Amplitude')
                ax.set_ylabel('Amplitude')
                plotted = True
            elif which == 'width':
                if model_type == "Asymmetric Voigt":
                    y_left = np.array(trends[peak_idx]['wid_left'])
                    y_right = np.array(trends[peak_idx]['wid_right'])
                    yerr_left = np.array(uncertainties[peak_idx]['wid_left'])
                    yerr_right = np.array(uncertainties[peak_idx]['wid_right'])
                    ax.plot(x, y_left, 'o', label=f'Peak {peak_idx+1} Left', color=color, alpha=0.7)
                    ax.plot(x, y_right, '^', label=f'Peak {peak_idx+1} Right', color=color, alpha=0.7)
                    if show_95:
                        ax.fill_between(x, y_left - yerr_left, y_left + yerr_left, color=color, alpha=0.2)
                        ax.fill_between(x, y_right - yerr_right, y_right + yerr_right, color=color, alpha=0.2)
                else:
                    y = np.array(trends[peak_idx]['wid'])
                    yerr = np.array(uncertainties[peak_idx]['wid'])
                    ax.plot(x, y, 'o', label=f'Peak {peak_idx+1}', color=color)
                    if show_95:
                        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
                ax.set_title('Peak Width')
                ax.set_ylabel('Width (cm⁻¹)')
                plotted = True
            elif which == 'eta':
                if model_type in ["Pseudo-Voigt", "Asymmetric Voigt"]:
                    y = np.array(trends[peak_idx]['eta'])
                    yerr = np.array(uncertainties[peak_idx]['eta'])
                    ax.plot(x, y, 'o', label=f'Peak {peak_idx+1}', color=color)
                    if show_95:
                        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
                    ax.set_title('Peak Eta')
                    ax.set_ylabel('Eta')
                    plotted = True
        if which == 'eta' and not (model_type in ["Pseudo-Voigt", "Asymmetric Voigt"]):
            ax.text(0.5, 0.5, 'Eta not available for current model', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Peak Eta')
            ax.set_ylabel('Eta')
        ax.set_xlabel('Spectrum Number')
        if show_grid:
            ax.grid(True, linestyle=':', color='gray', alpha=0.6)
        else:
            ax.grid(False)
        ax.legend()
    
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
                row = {'File': os.path.basename(result['file'])}
                
                if result['fit_params'] is not None:
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
            
            # Convert parameters to float if not "Auto"
            height = float(height) if height != "Auto" else None
            distance = float(distance) if distance != "Auto" else None
            prominence = float(prominence) if prominence != "Auto" else None
            
            # Find peaks
            from scipy.signal import find_peaks
            peak_indices, properties = find_peaks(
                self.spectra, 
                height=height, 
                distance=distance,
                prominence=prominence
            )
            
            # Store peak positions and intensities
            self.peaks = []
            for idx in peak_indices:
                self.peaks.append({
                    'position': float(self.wavenumbers[int(idx)]),
                    'intensity': float(self.spectra[int(idx)]),
                    'index': int(idx)
                })
            
            # Update plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find peaks: {str(e)}")
    
    def gaussian(self, x, a, x0, sigma):
        """Gaussian peak function."""
        return a * np.exp(-(x-x0)**2/(2*sigma**2))
    
    def lorentzian(self, x, a, x0, gamma):
        """Lorentzian peak function."""
        return a * gamma**2 / ((x-x0)**2 + gamma**2)
    
    def pseudo_voigt(self, x, a, x0, sigma, eta):
        """Pseudo-Voigt peak function."""
        return eta * self.lorentzian(x, a, x0, sigma) + (1-eta) * self.gaussian(x, a, x0, sigma)
    
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
        # Create separate arrays for left and right sides
        left_mask = x <= cen
        right_mask = ~left_mask
        
        # Initialize result array
        result = np.zeros_like(x)
        
        # Calculate Gaussian and Lorentzian components for left side
        g_left = np.exp(-((x[left_mask] - cen) / wid_left) ** 2)
        l_left = 1 / (1 + ((x[left_mask] - cen) / wid_left) ** 2)
        
        # Calculate Gaussian and Lorentzian components for right side
        g_right = np.exp(-((x[right_mask] - cen) / wid_right) ** 2)
        l_right = 1 / (1 + ((x[right_mask] - cen) / wid_right) ** 2)
        
        # Combine components using eta parameter
        result[left_mask] = amp * (eta * g_left + (1 - eta) * l_left)
        result[right_mask] = amp * (eta * g_right + (1 - eta) * l_right)
        
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
                    return
            
            # Filter peaks to only include those within ROI ranges
            peaks_in_roi = []
            for peak in self.peaks:
                peak_pos = peak['position']
                if not roi_ranges:  # If no ROI specified, include all peaks
                    peaks_in_roi.append(peak)
                else:
                    for min_w, max_w in roi_ranges:
                        if min_w <= peak_pos <= max_w:
                            peaks_in_roi.append(peak)
                            break
            
            if not peaks_in_roi:
                messagebox.showwarning("No Peaks in ROI", "No peaks found within the specified ROI ranges.")
                return
            
            # HARD CAP for Asymmetric Voigt
            if model_type == "Asymmetric Voigt" and len(peaks_in_roi) > 8:
                messagebox.showerror("Too Many Peaks", "Asymmetric Voigt fitting is limited to 8 peaks for stability. Please reduce the number of detected peaks (e.g., by increasing the prominence or height threshold).")
                return

            # Create mask for ROI ranges
            mask = np.zeros_like(self.wavenumbers, dtype=bool)
            if roi_ranges:
                for min_w, max_w in roi_ranges:
                    mask |= (self.wavenumbers >= min_w) & (self.wavenumbers <= max_w)
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
                return
            
            # Define combined model function
            def combined_model(x, *params):
                result = np.zeros_like(x)
                for i in range(0, len(params), params_per_peak):
                    peak_params = params[i:i+params_per_peak]
                    if len(peak_params) != params_per_peak:
                        print(f"Skipping peak at i={i}, expected {params_per_peak} params, got {len(peak_params)}: {peak_params}")
                        continue
                    try:
                        result += model_func(x, *peak_params)
                    except Exception as e:
                        print(f"Error in model_func at i={i} with params {peak_params}: {e}")
                        continue
                return result
            
            # Perform fit
            try:
                popt, pcov = curve_fit(combined_model, x_fit, y_fit,
                                     p0=initial_params,
                                     bounds=(bounds_lower, bounds_upper),
                                     maxfev=2000)
            except Exception as e1:
                # Retry with higher maxfev
                try:
                    popt, pcov = curve_fit(combined_model, x_fit, y_fit,
                                         p0=initial_params,
                                         bounds=(bounds_lower, bounds_upper),
                                         maxfev=20000)
                except Exception as e2:
                    # Fallback: use initial guess, no covariance, flag as failed
                    popt = np.array(initial_params)
                    pcov = None
                    self.fit_params = popt
                    self.fit_cov = pcov
                    self.fit_result = combined_model(self.wavenumbers, *popt)
                    self.residuals = self.spectra - self.fit_result
                    self.fit_failed = True
                    messagebox.showwarning("Fit Failed", f"Failed to fit peaks after two attempts. Using initial guess for this spectrum.\n\nFirst error: {str(e1)}\nSecond error: {str(e2)}")
                    self.update_plot()
                    self.update_peak_visibility_controls()
                    return
            # Robust check for Asymmetric Voigt
            if model_type == "Asymmetric Voigt" and len(popt) % 5 != 0:
                messagebox.showerror("Fit Error", f"Fit failed: number of fit parameters ({len(popt)}) is not a multiple of 5. Try reducing the number of peaks or adjusting initial guesses.")
                return
            # Store results
            self.fit_params = popt
            self.fit_cov = pcov
            self.fit_result = combined_model(self.wavenumbers, *popt)
            self.residuals = self.spectra - self.fit_result
            self.fit_failed = False
            
            # Update plot
            self.update_plot()
            self.update_peak_visibility_controls()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fit peaks: {str(e)}")
    
    def update_plot(self):
        """Update the current spectrum plot."""
        if not hasattr(self, 'wavenumbers') or not hasattr(self, 'spectra'):
            return
            
        # Clear plots
        self.ax1_current.clear()
        self.ax2_current.clear()
        
        # Set title to current filename
        if self.current_spectrum_index >= 0 and self.current_spectrum_index < len(self.spectra_files):
            filename = os.path.basename(self.spectra_files[self.current_spectrum_index])
            self.ax1_current.set_title(f"Current Spectrum: {filename}")
        
        # --- Highlight fit regions ---
        fit_ranges_str = getattr(self, 'var_fit_ranges', None)
        if fit_ranges_str is not None:
            fit_ranges_str = fit_ranges_str.get().strip()
            if fit_ranges_str:
                for part in fit_ranges_str.split(','):
                    if '-' in part:
                        try:
                            min_w, max_w = map(float, part.split('-'))
                            self.ax1_current.axvspan(min_w, max_w, color='lightgrey', alpha=0.3, zorder=0)
                        except Exception:
                            pass
        
        # Plot spectrum
        self.ax1_current.plot(self.wavenumbers, self.spectra, 'k-', label='Spectrum')
        
        # Plot background if available
        if self.background is not None:
            self.ax1_current.plot(self.wavenumbers, self.background, 'b--', label='Background')
        
        # Plot fitted peaks if available
        if self.fit_result is not None:
            model_type = self.current_model.get()
            if model_type == "Gaussian" or model_type == "Lorentzian":
                params_per_peak = 3
            elif model_type == "Pseudo-Voigt":
                params_per_peak = 4
            elif model_type == "Asymmetric Voigt":
                params_per_peak = 5
            else:
                params_per_peak = 3
            
            self.ax1_current.plot(self.wavenumbers, self.fit_result, 'r-', label='Fit')
            
            # Calculate R² values for each peak
            peak_r_squared = self.calculate_peak_r_squared(self.spectra, self.fit_result, self.fit_params, model_type)
            
            # Plot individual peaks if requested
            if self.show_individual_peaks.get():
                for i in range(0, len(self.fit_params), params_per_peak):
                    peak_params = self.fit_params[i:i+params_per_peak]
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
                    peak_idx = i // params_per_peak
                    r2 = peak_r_squared[peak_idx]
                    
                    # Plot peak with label including R² value
                    label = f'Peak {peak_idx+1} (R²={r2:.3f})'
                    color = 'red' if r2 < 0.95 else 'g'
                    self.ax1_current.plot(self.wavenumbers, peak, '--', color=color, alpha=0.5, label=label)
        
        # Plot peak positions if available
        if self.peaks:
            peak_positions = [peak['position'] for peak in self.peaks]
            peak_intensities = [peak['intensity'] for peak in self.peaks]
            self.ax1_current.plot(peak_positions, peak_intensities, 'ro', label='Peaks')
        
        # Plot residuals if available
        if self.residuals is not None:
            self.ax2_current.plot(self.wavenumbers, self.residuals, 'k-')
            self.ax2_current.axhline(y=0, color='r', linestyle='--')
        
        # Set labels and grid
        self.ax1_current.set_ylabel('Intensity')
        self.ax2_current.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax2_current.set_ylabel('Residuals')
        self.ax1_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        self.ax2_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        
        # Add legend
        self.ax1_current.legend()
        
        # Draw canvas
        self.canvas_current.draw()
    
    def clear_plot(self):
        """Clear the current spectrum plot."""
        self.ax1_current.clear()
        self.ax2_current.clear()
        self.ax1_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        self.ax2_current.grid(True, linestyle=':', color='gray', alpha=0.6)
        self.ax1_current.set_ylabel('Intensity')
        self.ax2_current.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax2_current.set_ylabel('Residuals')
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
    
    def on_closing(self):
        """Handle window closing event."""
        self.window.destroy()
    
    def update_roi_regions(self):
        """Refresh the plot to show the current fit regions as shaded areas."""
        self.update_plot()

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
            elif cmap_name in ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'tab10', 'RdBu_r']:
                cmap = plt.get_cmap(cmap_name)
                colors = [cmap(i / (n_lines - 1)) for i in range(n_lines)]
            else:
                colors = ['black'] * n_lines
            linewidth = float(self.waterfall_linewidth.get()) if hasattr(self, 'waterfall_linewidth') else 1.5
            
            # Calculate the maximum intensity for proper scaling
            max_intensity = max(np.max(spectrum) for spectrum in all_spectra_masked)
            offset_step = 0.1 * max_intensity
            
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
            
            # Ensure tight layout and maintain aspect ratio
            self.fig_heatmap.tight_layout()
            self.canvas_heatmap.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update heatmap plot: {str(e)}")

    def preview_trends_subplot(self, which):
        """Preview a single trends subplot (position, amplitude, width, eta) in a new window."""
        import matplotlib.pyplot as plt
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        preview_win = tk.Toplevel(self.window)
        preview_win.title(f"Preview {which.capitalize()} Plot")
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

    def calculate_r_squared(self, y_true, y_pred):
        """Calculate R-squared value for a fit."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def calculate_peak_r_squared(self, y_true, y_pred, peak_params, model_type):
        """Calculate R-squared value for individual peaks."""
        peak_r_squared = []
        for i in range(0, len(peak_params), len(peak_params)//len(self.peaks)):
            if model_type == "Gaussian":
                peak_y = self.gaussian(self.wavenumbers, *peak_params[i:i+3])
            elif model_type == "Lorentzian":
                peak_y = self.lorentzian(self.wavenumbers, *peak_params[i:i+3])
            elif model_type == "Pseudo-Voigt":
                peak_y = self.pseudo_voigt(self.wavenumbers, *peak_params[i:i+4])
            elif model_type == "Asymmetric Voigt":
                peak_y = self.asymmetric_voigt(self.wavenumbers, *peak_params[i:i+5])
            else:
                peak_y = self.gaussian(self.wavenumbers, *peak_params[i:i+3])
            r2 = self.calculate_r_squared(y_true, peak_y)
            peak_r_squared.append(r2)
        return peak_r_squared

    def update_fit_stats_plot(self):
        """Update the Fit Stats plot showing best to worst fits."""
        if not self.batch_results:
            for ax in self.ax_stats.flatten():
                ax.text(0.5, 0.5, 'No batch results to display.', ha='center', va='center')
            self.canvas_stats.draw()
            return

        # Calculate R-squared values for all fits
        fit_qualities = []
        for i, result in enumerate(self.batch_results):
            if result['fit_params'] is not None:
                data = np.loadtxt(result['file'])
                wavenumbers = data[:, 0]
                original_spectrum = data[:, 1]
                
                # Apply background subtraction if available
                if result['background'] is not None:
                    spectrum = original_spectrum - result['background']
                else:
                    spectrum = original_spectrum
                
                # Get ROI ranges
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
                        return
                
                # Create mask for ROI ranges
                mask = np.zeros_like(wavenumbers, dtype=bool)
                if roi_ranges:
                    for min_w, max_w in roi_ranges:
                        mask |= (wavenumbers >= min_w) & (wavenumbers <= max_w)
                    x_fit = wavenumbers[mask]
                    y_fit = spectrum[mask]
                    fit_result = result['fit_result'][mask]
                else:
                    x_fit = wavenumbers
                    y_fit = spectrum
                    fit_result = result['fit_result']
                
                # Get model type and parameters
                model_type = self.current_model.get()
                if model_type == "Gaussian" or model_type == "Lorentzian":
                    params_per_peak = 3
                elif model_type == "Pseudo-Voigt":
                    params_per_peak = 4
                elif model_type == "Asymmetric Voigt":
                    params_per_peak = 5
                else:
                    params_per_peak = 3

                # Calculate overall R-squared
                overall_r2 = self.calculate_r_squared(y_fit, fit_result)
                
                # Calculate individual peak R-squared values
                peak_r2 = []
                for j in range(0, len(result['fit_params']), params_per_peak):
                    peak_params = result['fit_params'][j:j+params_per_peak]
                    if model_type == "Gaussian":
                        peak_y = self.gaussian(wavenumbers, *peak_params)
                        center = peak_params[1]
                        width = peak_params[2]
                        fwhm = 2.3548 * width  # FWHM for Gaussian
                    elif model_type == "Lorentzian":
                        peak_y = self.lorentzian(wavenumbers, *peak_params)
                        center = peak_params[1]
                        width = peak_params[2]
                        fwhm = 2 * width  # FWHM for Lorentzian
                    elif model_type == "Pseudo-Voigt":
                        peak_y = self.pseudo_voigt(wavenumbers, *peak_params)
                        center = peak_params[1]
                        width = peak_params[2]
                        fwhm = 2 * width  # Approximate FWHM for Pseudo-Voigt
                    elif model_type == "Asymmetric Voigt":
                        peak_y = self.asymmetric_voigt(wavenumbers, *peak_params)
                        center = peak_params[1]
                        width_left = peak_params[2]
                        width_right = peak_params[3]
                        fwhm = (2 * width_left + 2 * width_right) / 2  # Average FWHM
                    else:
                        peak_y = self.gaussian(wavenumbers, *peak_params)
                        center = peak_params[1]
                        width = peak_params[2]
                        fwhm = 2.3548 * width
                    # Mask for ±FWHM around center
                    mask_peak = (wavenumbers >= center - fwhm) & (wavenumbers <= center + fwhm)
                    if np.sum(mask_peak) < 3:
                        # If too few points, expand window
                        mask_peak = (wavenumbers >= center - 2*fwhm) & (wavenumbers <= center + 2*fwhm)
                    y_true_peak = spectrum[mask_peak]
                    y_pred_peak = peak_y[mask_peak]
                    if len(y_true_peak) > 1:
                        r2 = self.calculate_r_squared(y_true_peak, y_pred_peak)
                    else:
                        r2 = np.nan
                    peak_r2.append(r2)
                
                fit_qualities.append({
                    'spectrum_number': i + 1,
                    'overall_r2': overall_r2,
                    'peak_r2': peak_r2,
                    'wavenumbers': wavenumbers,
                    'original_spectrum': original_spectrum,
                    'spectrum': spectrum,
                    'fit_result': result['fit_result'],
                    'fit_params': result['fit_params'],
                    'background': result['background'],
                    'roi_ranges': roi_ranges,
                    'mask': mask
                })

        # Sort by overall R-squared (best to worst)
        fit_qualities.sort(key=lambda x: x['overall_r2'], reverse=True)

        # Select indices for best, median, and worst fits
        n_total = len(fit_qualities)
        n_plots = min(9, n_total)
        best_indices = list(range(min(3, n_total)))
        worst_indices = list(range(max(0, n_total-3), n_total))
        # For median, pick 3 evenly spaced in the middle
        if n_total >= 7:
            median_indices = [n_total//2 - 1, n_total//2, n_total//2 + 1]
        elif n_total > 3:
            # If not enough for 3 medians, pick what we can
            median_indices = list(range(3, min(6, n_total)))
        else:
            median_indices = []
        # Combine for plotting order: best, median, worst
        plot_indices = best_indices + median_indices + worst_indices
        # Remove duplicates and keep order
        seen = set()
        plot_indices_unique = []
        for idx in plot_indices:
            if idx not in seen and idx < n_total:
                plot_indices_unique.append(idx)
                seen.add(idx)
        # Pad with remaining fits if less than 9
        for idx in range(n_total):
            if len(plot_indices_unique) >= 9:
                break
            if idx not in seen:
                plot_indices_unique.append(idx)
                seen.add(idx)
        # Now plot in 3x3 grid
        for ax in self.ax_stats.flatten():
            ax.clear()
        # Add overall title
        self.fig_stats.suptitle("Fit Quality Overview (Best, Median, Worst)", fontsize=14, y=0.98)
        row_labels = ["Top 3 Fits", "Median 3 Fits", "Worst 3 Fits"]
        for i, idx in enumerate(plot_indices_unique[:9]):
            row = i // 3
            col = i % 3
            ax = self.ax_stats[row, col]
            fit_data = fit_qualities[idx]
            wavenumbers = fit_data['wavenumbers']
            spectrum = fit_data['spectrum']
            fit_result = fit_data['fit_result']
            # Plot original spectrum and fit
            ax.plot(wavenumbers, spectrum, 'k-', label='Spectrum', alpha=0.7, linewidth=1)
            ax.plot(wavenumbers, fit_result, 'r-', label='Fit', alpha=0.7, linewidth=1)
            # Plot background if available
            if fit_data['background'] is not None:
                ax.plot(wavenumbers, fit_data['background'], 'b--', label='Background', alpha=0.5, linewidth=0.8)
            # Highlight ROI regions
            for min_w, max_w in fit_data['roi_ranges']:
                ax.axvspan(min_w, max_w, color='lightgrey', alpha=0.3, zorder=0)
            # Plot individual peaks
            model_type = self.current_model.get()
            params_per_peak = 3 if model_type in ["Gaussian", "Lorentzian"] else 4 if model_type == "Pseudo-Voigt" else 5
            for j in range(0, len(fit_data['fit_params']), params_per_peak):
                peak_params = fit_data['fit_params'][j:j+params_per_peak]
                if model_type == "Gaussian":
                    peak_y = self.gaussian(wavenumbers, *peak_params)
                elif model_type == "Lorentzian":
                    peak_y = self.lorentzian(wavenumbers, *peak_params)
                elif model_type == "Pseudo-Voigt":
                    peak_y = self.pseudo_voigt(wavenumbers, *peak_params)
                elif model_type == "Asymmetric Voigt":
                    peak_y = self.asymmetric_voigt(wavenumbers, *peak_params)
                else:
                    peak_y = self.gaussian(wavenumbers, *peak_params)
                ax.plot(wavenumbers, peak_y, 'g--', alpha=0.5, linewidth=0.8)
            # Add R-squared values to plot with cleaner formatting
            ax.set_title(f"Spectrum {fit_data['spectrum_number']}\nR² = {fit_data['overall_r2']:.4f}", fontsize=9, pad=5)
            # Add peak R-squared values with cleaner formatting
            peak_r2_text = "\n".join([f"P{j+1}: {r2:.4f}" for j, r2 in enumerate(fit_data['peak_r2'])])
            ax.text(0.02, 0.98, peak_r2_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
            # Clean up axes
            ax.grid(True, linestyle=':', color='gray', alpha=0.4)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if i == 0:  # Only add legend to first plot
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
            # Add axis labels only to left and bottom plots
            if col == 0:
                # Add vertical row label for first column, rotated 90 degrees ccw
                ax.set_ylabel(row_labels[row], fontsize=11, labelpad=30, rotation=90, va='center')
            if row == 2:
                ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=8)
        # Hide unused subplots
        for i in range(len(plot_indices_unique), 9):
            row = i // 3
            col = i % 3
            self.ax_stats[row, col].axis('off')
        
        # Add export buttons
        if not hasattr(self, 'export_stats_buttons_frame'):
            self.export_stats_buttons_frame = ttk.Frame(self.viz_notebook.winfo_children()[3])  # Fit Stats tab
            self.export_stats_buttons_frame.pack(side=tk.BOTTOM, pady=5)
            
            # Button for saving individual plots
            self.export_individual_button = ttk.Button(
                self.export_stats_buttons_frame,
                text="Save Individual Plots",
                command=self.export_individual_fit_stats
            )
    
    def export_individual_fit_stats(self):
        """Export individual Fit Stats plots to image files."""
        if not self.batch_results:
            messagebox.showwarning("No Results", "No batch results to export.")
            return
            
        try:
            # Ask for directory to save files
            save_dir = filedialog.askdirectory(title="Select Directory to Save Individual Plots")
            if not save_dir:
                return
                
            # Ask for file format
            format_dialog = tk.Toplevel(self.window)
            format_dialog.title("Select Format")
            format_dialog.geometry("200x150")
            
            format_var = tk.StringVar(value="png")
            ttk.Radiobutton(format_dialog, text="PNG", variable=format_var, value="png").pack(pady=5)
            ttk.Radiobutton(format_dialog, text="PDF", variable=format_var, value="pdf").pack(pady=5)
            ttk.Radiobutton(format_dialog, text="SVG", variable=format_var, value="svg").pack(pady=5)
            
            def save_plots():
                format = format_var.get()
                format_dialog.destroy()
                
                # Create a new figure for each plot
                for i, ax in enumerate(self.ax_stats.flatten()):
                    if not ax.has_data():
                        continue
                        
                    # Create new figure with same size as subplot
                    fig = plt.figure(figsize=(6, 4))
                    new_ax = fig.add_subplot(111)
                    
                    # Copy all elements from the subplot
                    for line in ax.get_lines():
                        new_ax.plot(line.get_xdata(), line.get_ydata(),
                                  color=line.get_color(),
                                  linestyle=line.get_linestyle(),
                                  linewidth=line.get_linewidth(),
                                  alpha=line.get_alpha(),
                                  label=line.get_label())
                    
                    # Copy background patches (ROI regions)
                    for patch in ax.patches:
                        if isinstance(patch, mpatches.Rectangle):
                            new_ax.add_patch(plt.Rectangle(
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
                        new_ax.text(0.98, 0.98, text.get_text(),
                                  transform=new_ax.transAxes,
                                  verticalalignment='top',
                                  horizontalalignment='right',
                                  fontsize=text.get_fontsize(),
                                  bbox=bbox_props)
                    
                    # Copy title and labels
                    new_ax.set_title(ax.get_title())
                    new_ax.set_xlabel(ax.get_xlabel())
                    new_ax.set_ylabel(ax.get_ylabel())
                    
                    # Copy grid
                    new_ax.grid(True, linestyle=':', color='gray', alpha=0.4)
                    
                    # Copy legend
                    if ax.get_legend():
                        new_ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
                    
                    # Set axis limits
                    new_ax.set_xlim(ax.get_xlim())
                    new_ax.set_ylim(ax.get_ylim())
                    
                    # Save the figure
                    spectrum_num = int(ax.get_title().split()[1])  # Extract spectrum number from title
                    filename = f"spectrum_{spectrum_num}_fit.{format}"
                    filepath = os.path.join(save_dir, filename)
                    
                    fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)
                
                messagebox.showinfo("Success", f"Individual plots saved to:\n{save_dir}")
            
            ttk.Button(format_dialog, text="Save", command=save_plots).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export individual plots: {str(e)}")

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
            self.update_waterfall_plot()