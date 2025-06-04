#!/usr/bin/env python3
# Raman Database Browser Module for RamanLab
"""
Module for browsing and managing Raman spectra database.
"""

import os
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import matplotlib as mpl
# Configure matplotlib to use DejaVu Sans which supports mathematical symbols
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import random
import json
import time
import threading
import queue
import re
from scipy.signal import find_peaks

class RamanDatabase:
    """Raman database management system for spectral data."""
    
    def __init__(self, database_path=None):
        """
        Initialize the Raman database.
        
        Parameters:
        -----------
        database_path : str, optional
            Path to the database file (.pkl)
        """
        self.database_path = database_path or os.path.join(os.path.dirname(__file__), "raman_database.pkl")
        self._database = None  # Initialize as None for lazy loading
        self._spectrum_list = None  # Cache for spectrum list
        self._loaded = False  # Flag to check if database has been loaded
 
        
    def _load_database(self):
        """Load the database from file or create a new one if it doesn't exist."""
        if self._loaded and self._database is not None:
            return self._database
            
        print("Loading Raman database...")
        start_time = time.time()  # Track loading time
        
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    self._database = pickle.load(f)
                self._loaded = True
                load_time = time.time() - start_time
                print(f"Database loaded in {load_time:.2f} seconds with {len(self._database)} spectra.")
                return self._database
            except Exception as e:
                print(f"Error loading database: {e}")
                self._database = {}
                self._loaded = True
                return self._database
        
        self._database = {}
        self._loaded = True
        return self._database
    
    @property
    def database(self):
        """Lazy load the database when first accessed."""
        if not self._loaded:
            return self._load_database()
        return self._database
            
    def save_database(self):
        """Save the database to file."""
        # Make sure database is loaded before saving
        if not self._loaded:
            self._load_database()
            
        try:
            start_time = time.time()
            with open(self.database_path, 'wb') as f:
                pickle.dump(self._database, f)
            save_time = time.time() - start_time
            print(f"Database saved in {save_time:.2f} seconds.")
            
            # Reset cache after saving
            self._spectrum_list = None
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
            
    def get_spectra(self):
        """Get list of all spectra in the database with caching."""
        # Use cached list if available
        if self._spectrum_list is not None:
            return self._spectrum_list
        
        # Filter out special metadata keys (starting with __) and sort the list
        self._spectrum_list = [name for name in self.database.keys() if not name.startswith('__')]
        self._spectrum_list.sort()
        return self._spectrum_list
        
    def get_spectrum_data(self, name):
        """Get data for a specific spectrum."""
        return self.database.get(name, None)
        
    def add_spectrum(self, name, wavenumbers, intensities, metadata=None):
        """
        Add a new spectrum to the database.
        
        Parameters:
        -----------
        name : str
            Name of the spectrum
        wavenumbers : array-like
            Wavenumber data
        intensities : array-like
            Intensity data
        metadata : dict, optional
            Additional metadata
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if name in self.database:
            return False
            
        self.database[name] = {
            'wavenumbers': np.array(wavenumbers),
            'intensities': np.array(intensities),
            'metadata': metadata or {},
        }
        return True
        
    def get_peaks(self, name, height=None, prominence=None, distance=None):
        """
        Get peaks for a specific spectrum.
        
        Parameters:
        -----------
        name : str
            Name of the spectrum
        height : float, optional
            Minimum peak height
        prominence : float, optional
            Minimum peak prominence
        distance : float, optional
            Minimum distance between peaks
            
        Returns:
        --------
        dict
            Dictionary containing peak information
        """
        if name not in self.database:
            return None
            
        data = self.database[name]
        intensities = data['intensities']
        wavenumbers = data['wavenumbers']
        
        # Find peaks
        peaks, properties = find_peaks(intensities, height=height, prominence=prominence, distance=distance)
        
        return {
            'positions': wavenumbers[peaks],
            'heights': intensities[peaks],
            'properties': properties
        }
        
    def search_spectra(self, search_term):
        """
        Search for spectra by name only (not metadata).
        
        Parameters:
        -----------
        search_term : str
            Search term
            
        Returns:
        --------
        list
            List of matching spectrum names
        """
        search_lower = search_term.lower()
        matches = []
        
        for name in self.database.keys():
            if name.startswith('__'):
                continue
                
            # Search only in spectrum name
            if search_lower in name.lower():
                matches.append(name)
                
        return sorted(matches)
        
    def delete_spectrum(self, name):
        """Delete a spectrum from the database."""
        if name in self.database:
            del self.database[name]
            self._spectrum_list = None  # Reset cache
            return True
        return False 

class RamanDatabaseGUI:
    """GUI for Raman database browsing and management."""
    
    def __init__(self, parent=None):
        """Initialize the database GUI."""
        # Set the clam theme for better appearance
        style = ttk.Style()
        style.theme_use("clam")
        
        self.db = RamanDatabase()
        
        # Create window if no parent is provided
        if parent is None:
            # This is the root window for standalone use
            self.is_standalone = True
            self.window = tk.Tk()
        else:
            # This is being called from another application
            self.is_standalone = False
            self.window = tk.Toplevel(parent)
            
        # Configure window
        self.window.title("Raman Database Browser")
        self.window.geometry("1400x900")
        
        # Set the 'clam' theme to make the GUI consistent with other windows
        style = ttk.Style(self.window)
        style.theme_use('clam')
        
        # Configure minimal styling while maintaining default font sizes
        self.style = style
        self.style.configure('TLabelframe', borderwidth=1)
        self.style.configure('TLabelframe.Label')
        self.style.configure('TButton')
        self.style.configure('Small.TButton')
        self.style.configure('TLabel')
        self.style.configure('TEntry')
        self.style.configure('Treeview')
        self.style.configure('Treeview.Heading')
        self.style.configure('TCheckbutton')
        
        self.create_gui()
        
    def create_gui(self):
        """Create the GUI elements."""
        # Main frame
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.batch_import_tab = ttk.Frame(self.notebook)
        self.hey_classification_tab = ttk.Frame(self.notebook)
        self.add_spectrum_tab = ttk.Frame(self.notebook)
        self.batch_edit_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.main_tab, text="Database Browser")
        self.notebook.add(self.batch_import_tab, text="Batch Import")
        self.notebook.add(self.hey_classification_tab, text="Hey Classification")
        self.notebook.add(self.add_spectrum_tab, text="Add Spectrum")
        self.notebook.add(self.batch_edit_tab, text="Batch Edit")
        
        # Create content for each tab
        self.create_main_tab_content()
        self.create_batch_import_tab_content()
        self.create_hey_classification_tab_content()
        self.create_add_spectrum_tab_content()
        self.create_batch_edit_tab_content()
        
    def create_main_tab_content(self):
        """Create content for the main database browser tab."""
        # Left panel - spectrum list and controls
        left_panel = ttk.Frame(self.main_tab, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Search frame
        search_frame = ttk.LabelFrame(left_panel, text="Search", padding=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(fill=tk.X, pady=2)
        search_entry.bind('<Return>', self.perform_search)
        search_entry.bind('<KeyRelease>', self.perform_search)
        
        ttk.Button(search_frame, text="Clear Search", command=self.clear_search).pack(fill=tk.X, pady=2)
        
        # Controls at the top
        control_frame = ttk.LabelFrame(left_panel, text="Database Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Refresh List", command=self.update_spectrum_list).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Export Spectrum", command=self.export_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Delete Spectrum", command=self.delete_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Save Database", command=self.save_database).pack(fill=tk.X, pady=2)
        
        # Spectrum list
        list_frame = ttk.LabelFrame(left_panel, text="Raman Spectra", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.spectrum_listbox = tk.Listbox(listbox_frame)
        spectrum_scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.spectrum_listbox.yview)
        spectrum_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.spectrum_listbox.configure(yscrollcommand=spectrum_scrollbar.set)
        self.spectrum_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.spectrum_listbox.bind('<<ListboxSelect>>', self.on_spectrum_select)
        
        # Database statistics
        stats_frame = ttk.LabelFrame(left_panel, text="Database Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_text = tk.Text(stats_frame, height=4, width=30, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.X)
        self.stats_text.config(state=tk.DISABLED)
        
        # Right panel - spectrum details
        right_panel = ttk.Frame(self.main_tab)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Spectrum information
        info_frame = ttk.LabelFrame(right_panel, text="Spectrum Information", padding=5)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create a grid layout for info
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1
        ttk.Label(info_grid, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.name_label = ttk.Label(info_grid, text="", width=30, relief="solid", borderwidth=1, background="white")
        self.name_label.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        
        ttk.Label(info_grid, text="Mineral Name:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=3)
        self.mineral_name_label = ttk.Label(info_grid, text="", width=25, relief="solid", borderwidth=1, background="white")
        self.mineral_name_label.grid(row=0, column=3, sticky=tk.W+tk.E, padx=5, pady=3)
        
        # Row 2
        ttk.Label(info_grid, text="Data Points:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.data_points_label = ttk.Label(info_grid, text="", width=30, relief="solid", borderwidth=1, background="white")
        self.data_points_label.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        
        ttk.Label(info_grid, text="Hey Classification:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=3)
        self.hey_class_label = ttk.Label(info_grid, text="", width=25, relief="solid", borderwidth=1, background="white")
        self.hey_class_label.grid(row=1, column=3, sticky=tk.W+tk.E, padx=5, pady=3)
        
        # Row 3
        ttk.Label(info_grid, text="Wavenumber Range:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.range_label = ttk.Label(info_grid, text="", width=30, relief="solid", borderwidth=1, background="white")
        self.range_label.grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        
        ttk.Label(info_grid, text="RRUFF ID:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=3)
        self.rruff_id_label = ttk.Label(info_grid, text="", width=25, relief="solid", borderwidth=1, background="white")
        self.rruff_id_label.grid(row=2, column=3, sticky=tk.W+tk.E, padx=5, pady=3)
        
        # Row 4 - Chemical Family
        ttk.Label(info_grid, text="Chemical Family:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.chemical_family_label = ttk.Label(info_grid, text="", width=30, relief="solid", borderwidth=1, background="white")
        self.chemical_family_label.grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        
        # Row 5 - Description (spans full width)
        ttk.Label(info_grid, text="Description:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=3)
        self.description_label = ttk.Label(info_grid, text="", width=70, relief="solid", borderwidth=1, background="white")
        self.description_label.grid(row=4, column=1, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=3)
        
        # Control buttons
        button_frame = ttk.Frame(info_grid)
        button_frame.grid(row=0, column=4, rowspan=5, padx=5, pady=2, sticky=tk.N+tk.S)
        
        ttk.Button(button_frame, text="View Metadata", command=self.view_metadata).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Edit Metadata", command=self.edit_metadata).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Export Data", command=self.export_spectrum).pack(fill=tk.X, pady=2)
        
        # Configure column weights for proper expansion
        for i in range(4):
            info_grid.columnconfigure(i, weight=1)
            
        # Peak analysis frame
        peak_frame = ttk.LabelFrame(right_panel, text="Peak Analysis", padding=5)
        peak_frame.pack(fill=tk.X, pady=(0, 10))
        
        peak_controls = ttk.Frame(peak_frame)
        peak_controls.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(peak_controls, text="Min Height:").pack(side=tk.LEFT, padx=2)
        self.height_var = tk.StringVar(value="0.1")
        ttk.Entry(peak_controls, textvariable=self.height_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(peak_controls, text="Min Prominence:").pack(side=tk.LEFT, padx=2)
        self.prominence_var = tk.StringVar(value="0.05")
        ttk.Entry(peak_controls, textvariable=self.prominence_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(peak_controls, text="Find Peaks", command=self.find_peaks).pack(side=tk.LEFT, padx=10)
        ttk.Button(peak_controls, text="Clear Peaks", command=self.clear_peaks).pack(side=tk.LEFT, padx=2)
        
        # Peak table
        peak_table_frame = ttk.Frame(peak_frame)
        peak_table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("position", "height", "prominence")
        self.peak_tree = ttk.Treeview(peak_table_frame, columns=columns, show='headings', height=4)
        
        # Add scrollbar for peak table
        peak_scrollbar = ttk.Scrollbar(peak_table_frame, orient="vertical", command=self.peak_tree.yview)
        peak_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.peak_tree.configure(yscrollcommand=peak_scrollbar.set)
        self.peak_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure peak table columns
        self.peak_tree.heading("position", text="Position (cm^-1)")
        self.peak_tree.heading("height", text="Height")
        self.peak_tree.heading("prominence", text="Prominence")
        
        self.peak_tree.column("position", width=120)
        self.peak_tree.column("height", width=100)
        self.peak_tree.column("prominence", width=100)
        
        # Visualization panel
        viz_frame = ttk.LabelFrame(right_panel, text="Spectrum Visualization", padding=5)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
        # Plot controls
        plot_controls = ttk.Frame(viz_frame)
        plot_controls.pack(fill=tk.X, pady=(0, 5))
        
        # Show peaks checkbox
        self.show_peaks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plot_controls, text="Show Peaks", variable=self.show_peaks_var, 
                       command=self.update_plot).pack(side=tk.LEFT, padx=5)
        
        # Show grid checkbox
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plot_controls, text="Show Grid", variable=self.show_grid_var, 
                       command=self.update_plot).pack(side=tk.LEFT, padx=5)
        
        # X-axis range controls
        ttk.Label(plot_controls, text="X Range:").pack(side=tk.LEFT, padx=(10, 2))
        self.x_min_var = tk.StringVar()
        ttk.Entry(plot_controls, textvariable=self.x_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(plot_controls, text="to").pack(side=tk.LEFT, padx=2)
        self.x_max_var = tk.StringVar()
        ttk.Entry(plot_controls, textvariable=self.x_max_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(plot_controls, text="Update", command=self.update_plot).pack(side=tk.LEFT, padx=5)
        
        # Create a figure for the plot
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='#f8f8f8', constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar with a wrapper for compatibility
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Try to create toolbar with compatibility fallback
        try:
            self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            self.toolbar.update()
        except Exception:
            # Fallback for older versions of matplotlib
            class CompatNavToolbar(NavigationToolbar2Tk):
                def __init__(self, canvas, parent):
                    self.toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom', 'Save')]
                    super().__init__(canvas, parent)
                
                def _Button(self, text, image_file, toggle, command):
                    if hasattr(super(), '_Button'):
                        return super()._Button(text, image_file, toggle, command)
                    else:
                        # Alternative implementation for compatibility
                        return tk.Button(master=self, text=text, command=command)
                
                def _update_buttons_checked(self):
                    # Handle button state updates safely
                    try:
                        if hasattr(super(), '_update_buttons_checked'):
                            super()._update_buttons_checked()
                    except:
                        pass
            
            self.toolbar = CompatNavToolbar(self.canvas, toolbar_frame)
        
        # Initialize the plots and data
        self.current_spectrum = None
        self.current_peaks = None
        
        # Load and update the spectrum list
        self.update_spectrum_list()
        self.update_stats()
        
    def create_batch_import_tab_content(self):
        """Create content for the batch import tab."""
        # Directory selection frame
        dir_frame = ttk.LabelFrame(self.batch_import_tab, text="Directory Selection", padding=10)
        dir_frame.pack(fill=tk.X, pady=5)
        
        self.batch_dir_var = tk.StringVar()
        ttk.Label(dir_frame, text="Directory:").pack(anchor=tk.W)
        dir_entry_frame = ttk.Frame(dir_frame)
        dir_entry_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(dir_entry_frame, textvariable=self.batch_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_entry_frame, text="Browse", command=self.select_batch_directory).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Control frame
        control_frame = ttk.LabelFrame(self.batch_import_tab, text="Import Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=2)
        
        self.batch_import_button = ttk.Button(button_frame, text="Start Import", command=self.start_batch_import)
        self.batch_import_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.cancel_import_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_batch_import, state=tk.DISABLED)
        self.cancel_import_button.pack(side=tk.LEFT)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.batch_import_tab, text="Progress", padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Progress bar
        self.batch_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.batch_progress.pack(fill=tk.X, pady=2)
        
        # Status text
        text_frame = ttk.Frame(progress_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        self.batch_status_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.batch_status_text.yview)
        self.batch_status_text.config(yscrollcommand=scrollbar.set)
        
        self.batch_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize batch import variables
        self.batch_import_cancel = False
        self.batch_import_queue = queue.Queue()
        
    def create_hey_classification_tab_content(self):
        """Create content for the Hey classification update tab."""
        # File selection frame
        file_frame = ttk.LabelFrame(self.hey_classification_tab, text="Hey Classification File", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Select the Classification CSV file:").pack(anchor=tk.W, pady=(0, 5))
        
        file_entry_frame = ttk.Frame(file_frame)
        file_entry_frame.pack(fill=tk.X, pady=2)
        
        self.hey_file_var = tk.StringVar()
        ttk.Entry(file_entry_frame, textvariable=self.hey_file_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_entry_frame, text="Browse", command=self.browse_hey_file).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Control frame
        control_frame = ttk.LabelFrame(self.hey_classification_tab, text="Update Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=2)
        
        self.hey_update_button = ttk.Button(button_frame, text="Start Update", command=self.start_hey_classification_update)
        self.hey_update_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.hey_cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_hey_update, state=tk.DISABLED)
        self.hey_cancel_button.pack(side=tk.LEFT)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.hey_classification_tab, text="Progress", padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Progress bar
        self.hey_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.hey_progress.pack(fill=tk.X, pady=2)
        
        # Status text
        text_frame = ttk.Frame(progress_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        self.hey_status_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.hey_status_text.yview)
        self.hey_status_text.config(yscrollcommand=scrollbar.set)
        
        self.hey_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize hey classification variables
        self.hey_update_cancel = False
        self.hey_update_queue = queue.Queue()
        
    def create_add_spectrum_tab_content(self):
        """Create content for the Add Spectrum tab."""
        # Create main container
        main_container = ttk.Frame(self.add_spectrum_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create PanedWindow for adjustable divider
        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Create left panel for metadata forms and right panel for preview
        left_panel = ttk.Frame(paned)
        right_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)
        paned.add(right_panel, weight=3)

        # ===== LEFT PANEL: Metadata Forms =====
        # Create canvas and scrollbar for scrolling metadata forms
        canvas = tk.Canvas(left_panel)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # File selection frame
        file_frame = ttk.LabelFrame(scrollable_frame, text="Spectrum File", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.spectrum_file_var = tk.StringVar()
        
        ttk.Label(file_frame, text="File Path:").pack(anchor=tk.W)
        file_path_frame = ttk.Frame(file_frame)
        file_path_frame.pack(fill=tk.X, pady=2)
        
        ttk.Entry(file_path_frame, textvariable=self.spectrum_file_var, width=32).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_path_frame, text="Browse", command=self.browse_spectrum_file).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Load and preview button
        ttk.Button(file_frame, text="Load & Preview Spectrum", command=self.load_spectrum_preview).pack(pady=5)
        
        # Basic Information frame
        basic_frame = ttk.LabelFrame(scrollable_frame, text="Basic Information", padding=10)
        basic_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Spectrum Name
        ttk.Label(basic_frame, text="Spectrum Name*:").pack(anchor=tk.W)
        self.add_spectrum_name_var = tk.StringVar()
        ttk.Entry(basic_frame, textvariable=self.add_spectrum_name_var, width=32).pack(fill=tk.X, pady=2)
        
        # Mineral Name
        ttk.Label(basic_frame, text="Mineral Name:").pack(anchor=tk.W)
        self.add_mineral_name_var = tk.StringVar()
        ttk.Entry(basic_frame, textvariable=self.add_mineral_name_var, width=32).pack(fill=tk.X, pady=2)
        
        # Chemical Formula
        ttk.Label(basic_frame, text="Chemical Formula:").pack(anchor=tk.W)
        self.add_formula_var = tk.StringVar()
        ttk.Entry(basic_frame, textvariable=self.add_formula_var, width=40).pack(fill=tk.X, pady=2)
        
        # Chemical Family
        ttk.Label(basic_frame, text="Chemical Family:").pack(anchor=tk.W)
        self.add_chemical_family_var = tk.StringVar()
        ttk.Entry(basic_frame, textvariable=self.add_chemical_family_var, width=40).pack(fill=tk.X, pady=2)
        
        # Description
        ttk.Label(basic_frame, text="Description:").pack(anchor=tk.W)
        self.add_description_text = tk.Text(basic_frame, height=3, width=40)
        self.add_description_text.pack(fill=tk.X, pady=2)
        
        # Classification frame
        class_frame = ttk.LabelFrame(scrollable_frame, text="Classification", padding=10)
        class_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Hey Classification dropdown
        ttk.Label(class_frame, text="Hey Classification:").pack(anchor=tk.W)
        self.add_hey_class_var = tk.StringVar()
        hey_class_combo = ttk.Combobox(class_frame, textvariable=self.add_hey_class_var, width=32, state='readonly')
        hey_class_combo['values'] = (
            'Native Elements',
            'Sulfides and Sulfosalts', 
            'Oxides and Hydroxides',
            'Halides',
            'Carbonates',
            'Nitrates',
            'Borates', 
            'Sulfates',
            'Phosphates',
            'Silicates'
        )
        hey_class_combo.pack(fill=tk.X, pady=2)
        
        # Hey-Celestian Classification dropdown
        ttk.Label(class_frame, text="Hey-Celestian Classification:").pack(anchor=tk.W, pady=(5,0))
        self.add_hey_celestian_var = tk.StringVar()
        hey_celestian_combo = ttk.Combobox(class_frame, textvariable=self.add_hey_celestian_var, width=32, state='readonly')
        hey_celestian_combo['values'] = (
            'Framework Silicates',
            'Chain Silicates', 
            'Sheet Silicates',
            'Tectosilicates',
            'Inosilicates',
            'Phyllosilicates',
            'Cyclosilicates',
            'Sorosilicates', 
            'Nesosilicates',
            'Paired Tetrahedral Silicates',
            'Ring Silicates',
            'Carbonates - Calcite Group',
            'Carbonates - Aragonite Group',
            'Carbonates - Dolomite Group',
            'Carbonates - Other',
            'Sulfates - Barite Group',
            'Sulfates - Gypsum Group',
            'Sulfates - Anhydrite Group',
            'Sulfates - Other',
            'Phosphates - Apatite Group',
            'Phosphates - Other',
            'Oxides - Spinel Group',
            'Oxides - Hematite Group',
            'Oxides - Rutile Group',
            'Oxides - Corundum Group',
            'Oxides - Other',
            'Hydroxides',
            'Halides',
            'Sulfides',
            'Native Elements',
            'Borates',
            'Tungstates/Molybdates',
            'Vanadates/Arsenates',
            'Nitrates',
            'Organic Minerals'
        )
        hey_celestian_combo.pack(fill=tk.X, pady=2)
        
        # Crystal System
        ttk.Label(class_frame, text="Crystal System:").pack(anchor=tk.W, pady=(5,0))
        self.add_crystal_system_var = tk.StringVar()
        crystal_system_combo = ttk.Combobox(class_frame, textvariable=self.add_crystal_system_var, width=32, state='readonly')
        crystal_system_combo['values'] = ('Cubic', 'Tetragonal', 'Orthorhombic', 'Hexagonal', 'Trigonal', 'Monoclinic', 'Triclinic')
        crystal_system_combo.pack(fill=tk.X, pady=2)
        
        # Space Group
        ttk.Label(class_frame, text="Space Group:").pack(anchor=tk.W, pady=(5,0))
        self.add_space_group_var = tk.StringVar()
        ttk.Entry(class_frame, textvariable=self.add_space_group_var, width=32).pack(fill=tk.X, pady=2)
        
        # Measurement Conditions frame
        measurement_frame = ttk.LabelFrame(scrollable_frame, text="Measurement Conditions", padding=10)
        measurement_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create a grid for measurement conditions
        meas_grid = ttk.Frame(measurement_frame)
        meas_grid.pack(fill=tk.X)
        
        # Laser Wavelength and Power on same row
        ttk.Label(meas_grid, text="Laser λ (nm):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.add_laser_wavelength_var = tk.StringVar()
        ttk.Entry(meas_grid, textvariable=self.add_laser_wavelength_var, width=12).grid(row=0, column=1, padx=(0, 10), pady=2)
        
        ttk.Label(meas_grid, text="Power (mW):").grid(row=0, column=2, sticky=tk.W, padx=(0, 5), pady=2)
        self.add_laser_power_var = tk.StringVar()
        ttk.Entry(meas_grid, textvariable=self.add_laser_power_var, width=12).grid(row=0, column=3, pady=2)
        
        # Acquisition Time and Accumulations on same row
        ttk.Label(meas_grid, text="Time (s):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.add_acquisition_time_var = tk.StringVar()
        ttk.Entry(meas_grid, textvariable=self.add_acquisition_time_var, width=12).grid(row=1, column=1, padx=(0, 10), pady=2)
        
        ttk.Label(meas_grid, text="Accumulations:").grid(row=1, column=2, sticky=tk.W, padx=(0, 5), pady=2)
        self.add_accumulations_var = tk.StringVar()
        ttk.Entry(meas_grid, textvariable=self.add_accumulations_var, width=12).grid(row=1, column=3, pady=2)
        
        # Instrument on its own row
        ttk.Label(meas_grid, text="Instrument:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.add_instrument_var = tk.StringVar()
        ttk.Entry(meas_grid, textvariable=self.add_instrument_var, width=32).grid(row=2, column=1, columnspan=3, sticky=tk.W+tk.E, pady=2)
        
        # Polarization on its own row
        ttk.Label(meas_grid, text="Polarization:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.add_polarization_var = tk.StringVar()
        polarization_combo = ttk.Combobox(meas_grid, textvariable=self.add_polarization_var, width=32, state='readonly')
        polarization_combo['values'] = (
            'Unpolarized',
            'Linear (parallel)',
            'Linear (perpendicular)',
            'Linear (crossed)',
            'Circular (right)',
            'Circular (left)',
            'Custom',
            'Not specified'
        )
        polarization_combo.grid(row=3, column=1, columnspan=3, sticky=tk.W+tk.E, pady=2)
        
        meas_grid.columnconfigure(1, weight=1)
        meas_grid.columnconfigure(3, weight=1)
        
        # Sample Information frame
        sample_frame = ttk.LabelFrame(scrollable_frame, text="Sample Information", padding=10)
        sample_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Locality
        ttk.Label(sample_frame, text="Locality:").pack(anchor=tk.W)
        self.add_locality_var = tk.StringVar()
        ttk.Entry(sample_frame, textvariable=self.add_locality_var, width=32).pack(fill=tk.X, pady=2)
        
        # Sample ID and Owner on same row
        id_owner_frame = ttk.Frame(sample_frame)
        id_owner_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(id_owner_frame, text="Sample ID:").pack(side=tk.LEFT)
        self.add_sample_id_var = tk.StringVar()
        ttk.Entry(id_owner_frame, textvariable=self.add_sample_id_var, width=15).pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(id_owner_frame, text="Owner:").pack(side=tk.LEFT)
        self.add_owner_var = tk.StringVar()
        ttk.Entry(id_owner_frame, textvariable=self.add_owner_var, width=15).pack(side=tk.LEFT, padx=(5, 0))
        
        # Control buttons frame
        control_frame = ttk.Frame(scrollable_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Clear All Fields", command=self.clear_add_spectrum_fields).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Add to Database", command=self.add_spectrum_to_database).pack(side=tk.RIGHT, padx=5)
        
        # Pack canvas and scrollbar for left panel
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # ===== RIGHT PANEL: Spectrum Preview =====
        preview_frame = ttk.LabelFrame(right_panel, text="Spectrum Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create larger preview plot
        self.preview_fig, self.preview_ax = plt.subplots(figsize=(10, 6), facecolor='white')
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for the preview plot
        preview_toolbar_frame = ttk.Frame(preview_frame)
        preview_toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        try:
            self.preview_toolbar = NavigationToolbar2Tk(self.preview_canvas, preview_toolbar_frame)
            self.preview_toolbar.update()
        except Exception:
            pass  # Fallback if toolbar creation fails
        
        # Initialize preview
        self.preview_ax.text(0.5, 0.5, 'Load a spectrum file to see preview', 
                           ha='center', va='center', transform=self.preview_ax.transAxes, fontsize=14)
        self.preview_ax.set_xlabel('Wavenumber (cm^-1)')
        self.preview_ax.set_ylabel('Intensity (a.u.)')
        self.preview_ax.grid(True, alpha=0.3)
        self.preview_fig.tight_layout()
        self.preview_canvas.draw()
        
        # Initialize variables for loaded spectrum data
        self.loaded_wavenumbers = None
        self.loaded_intensities = None
        
    def update_spectrum_list(self):
        """Update the spectrum list from the database."""
        self.spectrum_listbox.delete(0, tk.END)
        
        search_term = self.search_var.get().strip()
        if search_term:
            spectra = self.db.search_spectra(search_term)
        else:
            spectra = self.db.get_spectra()
            
        for spectrum in sorted(spectra):
            self.spectrum_listbox.insert(tk.END, spectrum)
            
        self.update_stats()
            
    def on_spectrum_select(self, event=None):
        """Handle spectrum selection from the list."""
        if not self.spectrum_listbox.curselection():
            return
        
        try:
            # Get the selected spectrum
            index = self.spectrum_listbox.curselection()[0]
            spectrum_name = self.spectrum_listbox.get(index)
            
            # Get spectrum data
            spectrum_data = self.db.get_spectrum_data(spectrum_name)
            if not spectrum_data:
                return
                
            self.current_spectrum = spectrum_name
            
            # Extract information
            wavenumbers = spectrum_data.get('wavenumbers', [])
            intensities = spectrum_data.get('intensities', [])
            metadata = spectrum_data.get('metadata', {})
            
            # Update info labels
            self.name_label.config(text=spectrum_name)
            self.data_points_label.config(text=str(len(wavenumbers)))
            
            if len(wavenumbers) > 0:
                wn_range = f"{wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm^-1"
                self.range_label.config(text=wn_range)
            else:
                self.range_label.config(text="N/A")
            
            # Extract metadata fields
            mineral_name = metadata.get('NAME', metadata.get('Mineral Name', 'N/A'))
            hey_class = metadata.get('HEY CLASSIFICATION', metadata.get('Hey Classification', 'N/A'))
            rruff_id = metadata.get('RRUFF ID', metadata.get('RRUFFIDS', 'N/A'))
            chemical_family = metadata.get('CHEMICAL FAMILY', metadata.get('Chemical Family', 'N/A'))
            description = metadata.get('DESCRIPTION', metadata.get('Description', 'N/A'))
            
            self.mineral_name_label.config(text=str(mineral_name))
            self.hey_class_label.config(text=str(hey_class))
            self.rruff_id_label.config(text=str(rruff_id))
            self.chemical_family_label.config(text=str(chemical_family))
            self.description_label.config(text=str(description)[:80] + "..." if len(str(description)) > 80 else str(description))
            
            # Update plot
            self.update_plot()
            
        except Exception as e:
            print(f"Error selecting spectrum: {e}")
            messagebox.showerror("Error", f"Error loading spectrum: {str(e)}")
            
    def update_plot(self):
        """Update the spectrum plot."""
        if not self.current_spectrum:
            return
            
        try:
            spectrum_data = self.db.get_spectrum_data(self.current_spectrum)
            if not spectrum_data:
                return
                
            wavenumbers = spectrum_data.get('wavenumbers', [])
            intensities = spectrum_data.get('intensities', [])
            
            if len(wavenumbers) == 0 or len(intensities) == 0:
                return
                
            # Clear the plot
            self.ax.clear()
            
            # Plot spectrum
            self.ax.plot(wavenumbers, intensities, 'b-', linewidth=1.0, label=self.current_spectrum)
            
            # Plot peaks if enabled and available
            if self.show_peaks_var.get() and self.current_peaks:
                peak_positions = self.current_peaks['positions']
                peak_heights = self.current_peaks['heights']
                self.ax.plot(peak_positions, peak_heights, 'ro', markersize=5, label='Peaks')
                
                # Annotate peaks
                for pos, height in zip(peak_positions, peak_heights):
                    self.ax.annotate(f'{pos:.0f}', xy=(pos, height), xytext=(0, 10),
                                   textcoords='offset points', ha='center', fontsize=8)
            
            # Set labels and title
            self.ax.set_xlabel('Wavenumber (cm^-1)')
            self.ax.set_ylabel('Intensity (a.u.)')
            self.ax.set_title(f'Raman Spectrum: {self.current_spectrum}')
            
            # Set x-range if specified
            try:
                x_min = float(self.x_min_var.get())
                x_max = float(self.x_max_var.get())
                self.ax.set_xlim(x_min, x_max)
            except ValueError:
                pass  # Use auto range if invalid values
            
            # Grid
            if self.show_grid_var.get():
                self.ax.grid(True, alpha=0.3)
            
            # Legend
            self.ax.legend()
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plot: {e}")
            
    def find_peaks(self):
        """Find peaks in the current spectrum."""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "Please select a spectrum first.")
            return
            
        try:
            # Get parameters
            height = float(self.height_var.get()) if self.height_var.get() else None
            prominence = float(self.prominence_var.get()) if self.prominence_var.get() else None
            
            # Find peaks
            self.current_peaks = self.db.get_peaks(self.current_spectrum, height=height, prominence=prominence)
            
            if self.current_peaks:
                # Update peak table
                self.update_peak_table()
                
                # Update plot if peaks are shown
                if self.show_peaks_var.get():
                    self.update_plot()
                    
                messagebox.showinfo("Peaks Found", f"Found {len(self.current_peaks['positions'])} peaks.")
            else:
                messagebox.showinfo("No Peaks", "No peaks found with the current parameters.")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values. Please enter numbers.")
        except Exception as e:
            messagebox.showerror("Error", f"Error finding peaks: {str(e)}")
            
    def update_peak_table(self):
        """Update the peak table with current peaks."""
        # Clear existing items
        for item in self.peak_tree.get_children():
            self.peak_tree.delete(item)
            
        if not self.current_peaks:
            return
            
        positions = self.current_peaks['positions']
        heights = self.current_peaks['heights']
        properties = self.current_peaks.get('properties', {})
        prominences = properties.get('prominences', [None] * len(positions))
        
        for i, (pos, height, prom) in enumerate(zip(positions, heights, prominences)):
            prom_str = f"{prom:.3f}" if prom is not None else "N/A"
            self.peak_tree.insert("", tk.END, values=(f"{pos:.1f}", f"{height:.3f}", prom_str))
            
    def clear_peaks(self):
        """Clear current peaks."""
        self.current_peaks = None
        self.update_peak_table()
        if self.show_peaks_var.get():
            self.update_plot()
            
    def perform_search(self, event=None):
        """Perform search based on current search term."""
        self.update_spectrum_list()
        
    def clear_search(self):
        """Clear the search term and refresh list."""
        self.search_var.set("")
        self.update_spectrum_list()
        
    def view_metadata(self):
        """View full metadata for the selected spectrum."""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "Please select a spectrum first.")
            return
            
        spectrum_data = self.db.get_spectrum_data(self.current_spectrum)
        if not spectrum_data:
            return
            
        metadata = spectrum_data.get('metadata', {})
        
        # Create metadata window
        metadata_window = tk.Toplevel(self.window)
        metadata_window.title(f"Metadata: {self.current_spectrum}")
        metadata_window.geometry("600x500")
        
        # Create text widget with scrollbar
        frame = ttk.Frame(metadata_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Insert metadata
        text_widget.insert(tk.END, f"Spectrum: {self.current_spectrum}\n\n")
        text_widget.insert(tk.END, "Metadata:\n")
        text_widget.insert(tk.END, "-" * 50 + "\n")
        
        for key, value in metadata.items():
            text_widget.insert(tk.END, f"{key}: {value}\n")
            
        text_widget.config(state=tk.DISABLED)
        
    def edit_metadata(self):
        """Edit metadata for the selected spectrum."""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "Please select a spectrum first.")
            return
            
        spectrum_data = self.db.get_spectrum_data(self.current_spectrum)
        if not spectrum_data:
            return
            
        metadata = spectrum_data.get('metadata', {}).copy()
        
        # Create edit window
        edit_window = tk.Toplevel(self.window)
        edit_window.title(f"Edit Metadata: {self.current_spectrum}")
        edit_window.geometry("500x400")
        edit_window.grab_set()
        
        # Create scrollable frame
        main_frame = ttk.Frame(edit_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create entry widgets for each metadata field
        entries = {}
        
        # Common fields
        common_fields = ['NAME', 'HEY CLASSIFICATION', 'DESCRIPTION', 
                        'LOCALITY', 'IDEAL CHEMISTRY', 'CHEMICAL FAMILY']
        
        for field in common_fields:
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=2, padx=5)
            
            ttk.Label(frame, text=f"{field}:", width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(metadata.get(field, '')))
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
            entries[field] = var
            
        # Add existing metadata fields not in common_fields
        for key, value in metadata.items():
            if key not in common_fields:
                frame = ttk.Frame(scrollable_frame)
                frame.pack(fill=tk.X, pady=2, padx=5)
                
                ttk.Label(frame, text=f"{key}:", width=20).pack(side=tk.LEFT)
                var = tk.StringVar(value=str(value))
                entry = ttk.Entry(frame, textvariable=var)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
                entries[key] = var
        
        # Buttons
        button_frame = ttk.Frame(edit_window)
        button_frame.pack(fill=tk.X, pady=10, padx=10)
        
        def save_changes():
            new_metadata = {}
            for field, var in entries.items():
                value = var.get().strip()
                if value:  # Only save non-empty values
                    new_metadata[field] = value
                    
            # Update database
            spectrum_data['metadata'] = new_metadata
            self.db.database[self.current_spectrum] = spectrum_data
            
            # Refresh display
            self.on_spectrum_select()
            edit_window.destroy()
            messagebox.showinfo("Success", "Metadata updated successfully.")
            
        def cancel_changes():
            edit_window.destroy()
            
        ttk.Button(button_frame, text="Save", command=save_changes).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel_changes).pack(side=tk.RIGHT)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)

    def export_spectrum(self):
        """Export the selected spectrum to a file."""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "Please select a spectrum first.")
            return
            
        spectrum_data = self.db.get_spectrum_data(self.current_spectrum)
        if not spectrum_data:
            return
            
        # Ask for file location
        filename = filedialog.asksaveasfilename(
            title="Export Spectrum",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            wavenumbers = spectrum_data.get('wavenumbers', [])
            intensities = spectrum_data.get('intensities', [])
            metadata = spectrum_data.get('metadata', {})
            
            with open(filename, 'w') as f:
                # Write header with metadata
                f.write(f"# Raman Spectrum: {self.current_spectrum}\n")
                f.write("# Metadata:\n")
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("#\n")
                f.write("# Wavenumber(cm-1)\tIntensity(a.u.)\n")
                
                # Write data
                for wn, intensity in zip(wavenumbers, intensities):
                    f.write(f"{wn:.2f}\t{intensity:.6f}\n")
                    
            messagebox.showinfo("Success", f"Spectrum exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting spectrum: {str(e)}")
            
    def delete_spectrum(self):
        """Delete the selected spectrum."""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "Please select a spectrum first.")
            return
            
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete spectrum '{self.current_spectrum}'?\n\nThis action cannot be undone."
        )
        
        if result:
            if self.db.delete_spectrum(self.current_spectrum):
                messagebox.showinfo("Success", "Spectrum deleted successfully.")
                self.current_spectrum = None
                self.current_peaks = None
                self.update_spectrum_list()
                self.clear_info_labels()
                self.ax.clear()
                self.canvas.draw()
            else:
                messagebox.showerror("Error", "Failed to delete spectrum.")
                
    def clear_info_labels(self):
        """Clear all info labels."""
        self.name_label.config(text="")
        self.mineral_name_label.config(text="")
        self.data_points_label.config(text="")
        self.hey_class_label.config(text="")
        self.range_label.config(text="")
        self.rruff_id_label.config(text="")
        self.chemical_family_label.config(text="")
        self.description_label.config(text="")
        
    def save_database(self):
        """Save the database to file."""
        if self.db.save_database():
            messagebox.showinfo("Success", "Database saved successfully.")
        else:
            messagebox.showerror("Error", "Failed to save database.")
            
    def update_stats(self):
        """Update database statistics."""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        try:
            total_spectra = len(self.db.get_spectra())
            
            # Count Hey classifications
            hey_counts = {}
            for spectrum_data in self.db.database.values():
                if isinstance(spectrum_data, dict):
                    metadata = spectrum_data.get('metadata', {})
                    hey_class = metadata.get('HEY CLASSIFICATION', 'Unknown')
                    hey_counts[hey_class] = hey_counts.get(hey_class, 0) + 1
            
            self.stats_text.insert(tk.END, f"Total Spectra: {total_spectra}\n\n")
            
            if hey_counts:
                self.stats_text.insert(tk.END, "Hey Classifications:\n")
                for hey_class, count in sorted(hey_counts.items()):
                    self.stats_text.insert(tk.END, f"  {hey_class}: {count}\n")
            
        except Exception as e:
            self.stats_text.insert(tk.END, f"Error calculating stats: {e}")
            
        self.stats_text.config(state=tk.DISABLED)
        
    def run(self):
        """Run the GUI."""
        if self.is_standalone:
            self.window.mainloop()

    # Batch Import Methods
    def select_batch_directory(self):
        """Select directory for batch import."""
        directory = filedialog.askdirectory(title="Select Directory with Spectrum Files")
        if directory:
            self.batch_dir_var.set(directory)
            
    def start_batch_import(self):
        """Start the batch import process optimized for RRUFF database."""
        directory = self.batch_dir_var.get()
        if not directory or not os.path.exists(directory):
            messagebox.showerror("Error", "Please select a valid directory.")
            return
            
        # Get list of spectrum files (RRUFF and other common formats)
        extensions = ['.txt', '.csv', '.dat', '.asc', '.spc']
        files = []
        for ext in extensions:
            files.extend([f for f in os.listdir(directory) if f.lower().endswith(ext)])
        
        if not files:
            messagebox.showinfo("No Files", f"No spectrum files found in the selected directory.\nSupported formats: {', '.join(extensions)}")
            return
            
        # Check if this looks like a RRUFF directory
        rruff_files = [f for f in files if '__R0' in f or 'raman' in f.lower()]
        is_rruff_import = len(rruff_files) > len(files) * 0.3  # More than 30% look like RRUFF files
        
        # Show confirmation dialog with import details
        if is_rruff_import:
            message = f"RRUFF Database Import Detected!\n\n"
            message += f"Found {len(files)} spectrum files in directory.\n"
            message += f"Estimated RRUFF files: {len(rruff_files)}\n\n"
            message += "Import behavior:\n"
            message += "• Duplicates will be automatically skipped\n"
            message += "• RRUFF metadata will be automatically extracted\n"
            message += "• Comprehensive summary will be shown at the end\n\n"
            message += "Continue with batch import?"
        else:
            message = f"Standard Batch Import\n\n"
            message += f"Found {len(files)} spectrum files.\n\n"
            message += "Import behavior:\n"
            message += "• Duplicates will be automatically skipped\n"
            message += "• Basic metadata will be extracted from filenames\n"
            message += "• Summary will be shown at the end\n\n"
            message += "Continue with batch import?"
            
        if not messagebox.askyesno("Confirm Batch Import", message):
            return
            
        # Reset progress and status
        self.batch_progress['value'] = 0
        self.batch_status_text.delete(1.0, tk.END)
        self.batch_import_cancel = False
        
        # Show initial status
        if is_rruff_import:
            self.batch_status_text.insert(tk.END, "🔬 Starting RRUFF Database Import...\n")
            self.batch_status_text.insert(tk.END, f"Processing {len(files)} files with automatic duplicate handling.\n\n")
        else:
            self.batch_status_text.insert(tk.END, "📁 Starting Batch Import...\n")
            self.batch_status_text.insert(tk.END, f"Processing {len(files)} files.\n\n")
        
        # Update UI
        self.batch_import_button.config(state=tk.DISABLED)
        self.cancel_import_button.config(state=tk.NORMAL)
        
        # Start worker thread
        import threading
        worker_thread = threading.Thread(target=self._batch_import_worker, args=(directory, files))
        worker_thread.daemon = True
        worker_thread.start()
        
        # Start checking queue
        self._check_batch_queue()
        
    def _batch_import_worker(self, directory, files):
        """Worker thread for batch import with RRUFF database support."""
        successful_imports = 0
        skipped_duplicates = 0
        failed_imports = 0
        duplicate_details = []
        error_details = []
        
        for i, filename in enumerate(files):
            if self.batch_import_cancel:
                self.batch_import_queue.put(("status", f"Import cancelled by user.\n"))
                break
                
            file_path = os.path.join(directory, filename)
            
            try:
                # Update progress
                progress = int((i / len(files)) * 100)
                self.batch_import_queue.put(("progress", progress))
                self.batch_import_queue.put(("status", f"Processing: {filename}\n"))
                
                # Parse filename and determine file type
                spectrum_name = self._parse_filename(filename)
                
                # Check for duplicate before processing file
                if spectrum_name in self.db.database:
                    skipped_duplicates += 1
                    duplicate_details.append(spectrum_name)
                    self.batch_import_queue.put(("status", f"⚠ Skipped duplicate: {spectrum_name}\n"))
                    continue
                
                # Determine if this is a RRUFF file and parse accordingly
                is_rruff_file = self._is_rruff_file(filename)
                
                if is_rruff_file:
                    # Use RRUFF-specific parsing
                    wavenumbers, intensities, metadata = self._parse_rruff_file(file_path, filename)
                else:
                    # Use standard parsing for non-RRUFF files
                    wavenumbers, intensities, metadata = self._parse_standard_file(file_path, filename)
                
                if wavenumbers is not None and intensities is not None:
                    # Add to database
                    success = self.db.add_spectrum(spectrum_name, wavenumbers, intensities, metadata)
                    
                    if success:
                        successful_imports += 1
                        self.batch_import_queue.put(("status", f"✓ Imported: {spectrum_name}\n"))
                    else:
                        failed_imports += 1
                        error_msg = f"Database error for {filename}"
                        error_details.append((filename, error_msg))
                        self.batch_import_queue.put(("status", f"✗ Failed to add to database: {spectrum_name}\n"))
                else:
                    failed_imports += 1
                    error_msg = "Invalid file format or empty data"
                    error_details.append((filename, error_msg))
                    self.batch_import_queue.put(("status", f"✗ Failed to parse: {filename}\n"))
                
            except Exception as e:
                failed_imports += 1
                error_msg = str(e)
                error_details.append((filename, error_msg))
                self.batch_import_queue.put(("status", f"✗ Error processing {filename}: {error_msg}\n"))
        
        # Final comprehensive summary
        self.batch_import_queue.put(("progress", 100))
        self.batch_import_queue.put(("status", f"\n" + "="*60 + "\n"))
        self.batch_import_queue.put(("status", f"BATCH IMPORT SUMMARY\n"))
        self.batch_import_queue.put(("status", f"="*60 + "\n"))
        self.batch_import_queue.put(("status", f"Total files processed: {len(files)}\n"))
        self.batch_import_queue.put(("status", f"✓ Successfully imported: {successful_imports}\n"))
        self.batch_import_queue.put(("status", f"⚠ Skipped (duplicates): {skipped_duplicates}\n"))
        self.batch_import_queue.put(("status", f"✗ Failed imports: {failed_imports}\n"))
        
        # Show duplicate details if any
        if skipped_duplicates > 0:
            self.batch_import_queue.put(("status", f"\nDuplicate spectra skipped:\n"))
            for i, dup_name in enumerate(duplicate_details[:10]):  # Show first 10
                self.batch_import_queue.put(("status", f"  • {dup_name}\n"))
            if len(duplicate_details) > 10:
                self.batch_import_queue.put(("status", f"  ... and {len(duplicate_details) - 10} more\n"))
        
        # Show error details if any
        if failed_imports > 0:
            self.batch_import_queue.put(("status", f"\nFailed imports:\n"))
            for i, (filename, error) in enumerate(error_details[:5]):  # Show first 5 errors
                self.batch_import_queue.put(("status", f"  • {filename}: {error}\n"))
            if len(error_details) > 5:
                self.batch_import_queue.put(("status", f"  ... and {len(error_details) - 5} more errors\n"))
        
        # Final status
        self.batch_import_queue.put(("status", f"\n"))
        if successful_imports > 0:
            self.batch_import_queue.put(("status", f"Import completed successfully! Database now contains {successful_imports} new spectra.\n"))
        else:
            self.batch_import_queue.put(("status", f"No new spectra were imported.\n"))
            
        self.batch_import_queue.put(("complete", None))

    def _parse_filename(self, filename):
        """Parse filename to extract meaningful spectrum name."""
        # Remove file extension
        base_name = os.path.splitext(filename)[0]
        
        # RRUFF files often have format like: "Quartz__R040031__Raman__532__0.txt"
        # or "Calcite__R040032__Raman__785__0.txt"
        parts = base_name.split('__')
        
        if len(parts) >= 2:
            # Extract mineral name and RRUFF ID
            mineral_name = parts[0].replace('_', ' ').strip()
            rruff_id = parts[1] if parts[1].startswith('R') else None
            
            # Create descriptive name
            if rruff_id:
                spectrum_name = f"{mineral_name} ({rruff_id})"
            else:
                spectrum_name = mineral_name
        else:
            # Fallback to original filename
            spectrum_name = base_name.replace('_', ' ')
        
        return spectrum_name
    
    def _is_rruff_file(self, filename):
        """Check if the file is a RRUFF file."""
        # Check for common RRUFF filename patterns
        return '__R0' in filename or '__R1' in filename or 'raman' in filename.lower()
    
    def _parse_standard_file(self, file_path, filename):
        """Parse standard spectrum file (non-RRUFF) and extract data and metadata."""
        import time
        import re
        
        try:
            # Use simple numpy loadtxt for standard two-column format
            data = np.loadtxt(file_path)
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
            
            # Basic metadata from filename
            base_name = os.path.splitext(filename)[0]
            metadata = {
                'SOURCE_FILE': filename,
                'SOURCE': 'User Import',
                'IMPORT_DATE': time.strftime("%Y-%m-%d %H:%M:%S"),
                'DATA_POINTS': str(len(wavenumbers)),
                'WAVENUMBER_RANGE': f"{wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm^-1"
            }
            
            # Try to extract mineral name from filename
            # Clean up common separators and numbers
            cleaned_name = base_name.replace('_', ' ').replace('-', ' ')
            # Remove trailing numbers that might be sample numbers
            cleaned_name = re.sub(r'\s+\d+$', '', cleaned_name).strip()
            if cleaned_name:
                metadata['NAME'] = cleaned_name
                metadata['MINERAL_NAME'] = cleaned_name
            
            return wavenumbers, intensities, metadata
            
        except Exception as e:
            # If standard parsing fails, try to read as text file with more flexibility
            try:
                wavenumbers = []
                intensities = []
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    for line_num, line in enumerate(file):
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('%'):
                            try:
                                parts = line.replace(',', ' ').split()
                                if len(parts) >= 2:
                                    wavenumber = float(parts[0])
                                    intensity = float(parts[1])
                                    wavenumbers.append(wavenumber)
                                    intensities.append(intensity)
                            except (ValueError, IndexError):
                                continue
                
                if wavenumbers and intensities:
                    wavenumbers = np.array(wavenumbers)
                    intensities = np.array(intensities)
                    
                    base_name = os.path.splitext(filename)[0]
                    metadata = {
                        'SOURCE_FILE': filename,
                        'SOURCE': 'User Import',
                        'IMPORT_DATE': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'DATA_POINTS': str(len(wavenumbers)),
                        'WAVENUMBER_RANGE': f"{wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm^-1"
                    }
                    
                    # Try to extract mineral name
                    cleaned_name = base_name.replace('_', ' ').replace('-', ' ')
                    cleaned_name = re.sub(r'\s+\d+$', '', cleaned_name).strip()
                    if cleaned_name:
                        metadata['NAME'] = cleaned_name
                        metadata['MINERAL_NAME'] = cleaned_name
                    
                    return wavenumbers, intensities, metadata
                else:
                    return None, None, {}
                    
            except Exception as e2:
                raise Exception(f"Error parsing standard file: {str(e2)}")
    
    def _parse_rruff_file(self, file_path, filename):
        """Parse RRUFF spectrum file and extract data and metadata."""
        try:
            wavenumbers = []
            intensities = []
            metadata = {}
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
            
            # Parse header for metadata
            data_start_line = 0
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for metadata in comments or header lines
                if line.startswith('#') or line.startswith('##'):
                    # Parse metadata from comments
                    if ':' in line:
                        parts = line.lstrip('#').split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip().upper()
                            value = parts[1].strip()
                            metadata[key] = value
                
                # Find where data starts (first line with two numbers)
                elif line and not line.startswith('#'):
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            float(parts[0])
                            float(parts[1])
                            data_start_line = i
                            break
                    except ValueError:
                        continue
            
            # Parse spectral data
            for line in lines[data_start_line:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            wavenumber = float(parts[0])
                            intensity = float(parts[1])
                            wavenumbers.append(wavenumber)
                            intensities.append(intensity)
                    except (ValueError, IndexError):
                        continue
            
            # Extract additional metadata from filename
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split('__')
            
            if len(parts) >= 2:
                metadata['MINERAL_NAME'] = parts[0].replace('_', ' ')
                metadata['NAME'] = parts[0].replace('_', ' ')
                
                if len(parts) >= 3 and parts[1].startswith('R'):
                    metadata['RRUFF_ID'] = parts[1]
                    metadata['RRUFFIDS'] = parts[1]
                
                if len(parts) >= 4:
                    metadata['TECHNIQUE'] = parts[2]
                
                if len(parts) >= 5:
                    try:
                        laser_wavelength = int(parts[3])
                        metadata['LASER_WAVELENGTH'] = str(laser_wavelength)
                        metadata['LASER WAVELENGTH'] = f"{laser_wavelength} nm"
                    except ValueError:
                        pass
            
            # Add import metadata
            import time
            metadata['IMPORT_DATE'] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata['SOURCE_FILE'] = filename
            metadata['SOURCE'] = 'RRUFF Database'
            metadata['DATA_POINTS'] = str(len(wavenumbers))
            
            if wavenumbers:
                metadata['WAVENUMBER_RANGE'] = f"{min(wavenumbers):.1f} - {max(wavenumbers):.1f} cm^-1"
            
            # Convert to numpy arrays
            if wavenumbers and intensities:
                return np.array(wavenumbers), np.array(intensities), metadata
            else:
                return None, None, metadata
                
        except Exception as e:
            raise Exception(f"Error parsing file: {str(e)}")

    def _check_batch_queue(self):
        """Check the batch import queue for updates."""
        try:
            while True:
                try:
                    msg_type, data = self.batch_import_queue.get_nowait()
                    
                    if msg_type == "progress":
                        self.batch_progress['value'] = data
                    elif msg_type == "status":
                        self.batch_status_text.insert(tk.END, data)
                        self.batch_status_text.see(tk.END)
                    elif msg_type == "complete":
                        self.batch_import_button.config(state=tk.NORMAL)
                        self.cancel_import_button.config(state=tk.DISABLED)
                        self.update_spectrum_list()
                        self.update_stats()
                        return  # Stop checking
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            print(f"Error in batch queue check: {e}")
        
        # Schedule next check
        self.window.after(100, self._check_batch_queue)
        
    def cancel_batch_import(self):
        """Cancel batch import process."""
        self.batch_import_cancel = True
        self.cancel_import_button.config(state=tk.DISABLED)
        
    # Hey Classification Methods
    def load_hey_classification_data(self):
        """Load Hey classification data from CSV file."""
        csv_file = self.hey_file_var.get()
        if not csv_file or not os.path.exists(csv_file):
            messagebox.showerror("Error", "Please select a valid Hey_Classification.csv file.")
            return None
            
        try:
            df = pd.read_csv(csv_file)
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
            return None
    
    def browse_hey_file(self):
        """Browse for Hey classification CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select Hey_Classification.csv file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.hey_file_var.set(file_path)
    
    def cancel_hey_update(self):
        """Cancel the Hey classification update process."""
        self.hey_update_cancel = True
        self.hey_cancel_button.config(state=tk.DISABLED)
        self.hey_update_button.config(state=tk.NORMAL)

    def start_hey_classification_update(self):
        """Start Hey classification update process."""
        csv_file = self.hey_file_var.get()
        if not csv_file or not os.path.exists(csv_file):
            messagebox.showerror("Error", "Please select a valid Hey classification CSV file.")
            return
            
        if not self.db.database:
            messagebox.showinfo("Database", "The database is empty.")
            return
            
        # Load Hey classification data
        try:
            df = pd.read_csv(csv_file)
            hey_data = {}
            
            for _, row in df.iterrows():
                mineral_name = str(row.get('MINERAL_NAME', '')).strip()
                hey_class = str(row.get('HEY_CLASSIFICATION', '')).strip()
                
                if mineral_name and hey_class:
                    hey_data[mineral_name.upper()] = {
                        'HEY CLASSIFICATION': hey_class,
                        'STRUCTURAL GROUPNAME': str(row.get('STRUCTURAL_GROUPNAME', '')).strip(),
                        'CRYSTAL SYSTEM': str(row.get('CRYSTAL_SYSTEM', '')).strip(),
                        'SPACE GROUP': str(row.get('SPACE_GROUP', '')).strip(),
                        'OLDEST KNOWN AGE (MA)': str(row.get('OLDEST_KNOWN_AGE_MA', '')).strip(),
                        'PARAGENETIC MODE': str(row.get('PARAGENETIC_MODE', '')).strip()
                    }
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Hey classification data: {str(e)}")
            return
            
        # Clear log and reset progress
        self.hey_status_text.delete(1.0, tk.END)
        db_items = list(self.db.database.items())
        self.hey_progress["maximum"] = len(db_items)
        self.hey_progress["value"] = 0
        
        # Update UI state
        self.hey_update_button.config(state=tk.DISABLED)
        self.hey_cancel_button.config(state=tk.NORMAL)
        self.hey_update_cancel = False
        
        # Process each database entry
        updated = 0
        already_had = 0
        not_found = 0
        
        metadata_fields_updated = {
            "HEY CLASSIFICATION": 0,
            "STRUCTURAL GROUPNAME": 0,
            "CRYSTAL SYSTEM": 0,
            "SPACE GROUP": 0,
            "OLDEST KNOWN AGE (MA)": 0,
            "PARAGENETIC MODE": 0,
        }
        
        for i, (name, data) in enumerate(db_items):
            if self.hey_update_cancel:
                self.hey_status_text.insert(tk.END, "Update cancelled by user.\n")
                break
                
            # Update progress
            self.hey_progress["value"] = i
            percentage = int((i / len(db_items)) * 100) if len(db_items) > 0 else 0
            self.hey_status_text.insert(tk.END, f"Processing {name} - {percentage}% complete ({i}/{len(db_items)})\n")
            self.hey_status_text.see(tk.END)
            self.window.update_idletasks()
            
            # Ensure metadata exists
            if "metadata" not in data:
                data["metadata"] = {}
            metadata = data["metadata"]
            
            # Try to find mineral name
            mineral_name = None
            if "NAME" in metadata and metadata["NAME"]:
                mineral_name = metadata["NAME"]
            else:
                # Extract from spectrum name
                potential_name = name.split("__")[0].split("_")[0].split("-")[0].strip()
                cleaned_name = re.sub(r"[0-9_]+$", "", potential_name).strip()
                if cleaned_name:
                    mineral_name = cleaned_name
                    metadata["NAME"] = mineral_name
                    
            if not mineral_name:
                self.hey_status_text.insert(tk.END, f"Skipped {name}: No mineral name found\n")
                continue
                
            # Check if Hey Classification already exists
            had_hey_class = bool(metadata.get("HEY CLASSIFICATION"))
            if had_hey_class:
                already_had += 1
                
            # Look up in Hey data
            mineral_key = mineral_name.upper()
            if mineral_key in hey_data:
                fields_updated = []
                for field, value in hey_data[mineral_key].items():
                    if value and (not metadata.get(field) or field == "HEY CLASSIFICATION"):
                        metadata[field] = value
                        fields_updated.append(field)
                        metadata_fields_updated[field] = metadata_fields_updated.get(field, 0) + 1
                        
                if fields_updated:
                    updated += 1
                    self.hey_status_text.insert(tk.END, f"✓ Updated {name}: {', '.join(fields_updated)}\n")
                    
            else:
                not_found += 1
                self.hey_status_text.insert(tk.END, f"✗ Not found: {mineral_name}\n")
                
        # Final update
        self.hey_progress["value"] = len(db_items)
        self.hey_status_text.insert(tk.END, f"\nComplete! Updated: {updated}, Already had: {already_had}, Not found: {not_found}\n")
        
        # Show summary
        self.hey_status_text.insert(tk.END, f"\n--- Update Complete ---\n")
        self.hey_status_text.insert(tk.END, f"Updated: {updated}\nAlready had Hey Classification: {already_had}\nNot found: {not_found}\n")
        self.hey_status_text.insert(tk.END, "\nMetadata fields updated:\n")
        for field, count in metadata_fields_updated.items():
            if count > 0:
                self.hey_status_text.insert(tk.END, f"  - {field}: {count} entries\n")
                
        # Save database
        if self.db.save_database():
            self.hey_status_text.insert(tk.END, "Database saved successfully\n")
        else:
            self.hey_status_text.insert(tk.END, "Warning: Could not save database\n")
            
        # Update UI
        self.hey_update_button.config(state=tk.NORMAL)
        self.hey_cancel_button.config(state=tk.DISABLED)
        self.update_spectrum_list()
        self.update_stats()

    def add_spectrum_to_database(self):
        """Add a new spectrum to the database with full metadata."""
        # Check if spectrum data has been loaded
        if self.loaded_wavenumbers is None or self.loaded_intensities is None:
            messagebox.showerror("Error", "Please load a spectrum file first.")
            return
            
        # Get spectrum name
        spectrum_name = self.add_spectrum_name_var.get().strip()
        if not spectrum_name:
            messagebox.showerror("Error", "Please enter a spectrum name.")
            return
            
        # Check if spectrum already exists
        if spectrum_name in self.db.database:
            if not messagebox.askyesno("Spectrum Exists", 
                                     f"A spectrum named '{spectrum_name}' already exists. Replace it?"):
                return
                
        try:
            # Collect all metadata from the form
            metadata = {}
            
            # Basic information
            if self.add_mineral_name_var.get().strip():
                metadata['NAME'] = self.add_mineral_name_var.get().strip()
                metadata['Mineral Name'] = self.add_mineral_name_var.get().strip()
            
            if self.add_formula_var.get().strip():
                metadata['FORMULA'] = self.add_formula_var.get().strip()
                metadata['Chemical Formula'] = self.add_formula_var.get().strip()
                
            if self.add_chemical_family_var.get().strip():
                metadata['CHEMICAL FAMILY'] = self.add_chemical_family_var.get().strip()
                metadata['Chemical Family'] = self.add_chemical_family_var.get().strip()
                
            description = self.add_description_text.get(1.0, tk.END).strip()
            if description:
                metadata['DESCRIPTION'] = description
                metadata['Description'] = description
                
            # Classification
            if self.add_hey_class_var.get().strip():
                metadata['HEY CLASSIFICATION'] = self.add_hey_class_var.get().strip()
                metadata['Hey Classification'] = self.add_hey_class_var.get().strip()
                
            if self.add_hey_celestian_var.get().strip():
                metadata['HEY CELESTIAN CLASSIFICATION'] = self.add_hey_celestian_var.get().strip()
                metadata['Hey-Celestian Classification'] = self.add_hey_celestian_var.get().strip()
                
            if self.add_crystal_system_var.get().strip():
                metadata['CRYSTAL SYSTEM'] = self.add_crystal_system_var.get().strip()
                metadata['Crystal System'] = self.add_crystal_system_var.get().strip()
                
            if self.add_space_group_var.get().strip():
                metadata['SPACE GROUP'] = self.add_space_group_var.get().strip()
                metadata['Space Group'] = self.add_space_group_var.get().strip()
                
            # Measurement conditions
            if self.add_laser_wavelength_var.get().strip():
                metadata['LASER WAVELENGTH'] = self.add_laser_wavelength_var.get().strip()
                metadata['Laser Wavelength (nm)'] = self.add_laser_wavelength_var.get().strip()
                
            if self.add_laser_power_var.get().strip():
                metadata['LASER POWER'] = self.add_laser_power_var.get().strip()
                metadata['Laser Power (mW)'] = self.add_laser_power_var.get().strip()
                
            if self.add_acquisition_time_var.get().strip():
                metadata['ACQUISITION TIME'] = self.add_acquisition_time_var.get().strip()
                metadata['Acquisition Time (s)'] = self.add_acquisition_time_var.get().strip()
                
            if self.add_accumulations_var.get().strip():
                metadata['ACCUMULATIONS'] = self.add_accumulations_var.get().strip()
                metadata['Number of Accumulations'] = self.add_accumulations_var.get().strip()
                
            if self.add_instrument_var.get().strip():
                metadata['INSTRUMENT'] = self.add_instrument_var.get().strip()
                metadata['Instrument'] = self.add_instrument_var.get().strip()
                
            if self.add_polarization_var.get().strip():
                metadata['POLARIZATION'] = self.add_polarization_var.get().strip()
                metadata['Polarization'] = self.add_polarization_var.get().strip()
                
            # Sample information
            if self.add_locality_var.get().strip():
                metadata['LOCALITY'] = self.add_locality_var.get().strip()
                metadata['Locality'] = self.add_locality_var.get().strip()
                
            if self.add_sample_id_var.get().strip():
                metadata['SAMPLE ID'] = self.add_sample_id_var.get().strip()
                metadata['Sample ID'] = self.add_sample_id_var.get().strip()
                
            if self.add_owner_var.get().strip():
                metadata['OWNER'] = self.add_owner_var.get().strip()
                metadata['Owner'] = self.add_owner_var.get().strip()
                
            # Add import information
            import time
            metadata['IMPORT_DATE'] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata['SOURCE_FILE'] = os.path.basename(self.spectrum_file_var.get()) if self.spectrum_file_var.get() else "Manual Entry"
            metadata['DATA_POINTS'] = str(len(self.loaded_wavenumbers))
            metadata['WAVENUMBER_RANGE'] = f"{self.loaded_wavenumbers[0]:.1f} - {self.loaded_wavenumbers[-1]:.1f} cm^-1"
            
            # Add spectrum to database
            success = self.db.add_spectrum(spectrum_name, self.loaded_wavenumbers, self.loaded_intensities, metadata)
            
            if success:
                messagebox.showinfo("Success", f"Spectrum '{spectrum_name}' added successfully to the database.")
                
                # Update the spectrum list
                self.update_spectrum_list()
                self.update_stats()
                
                # Ask if user wants to clear the form
                if messagebox.askyesno("Clear Form", "Spectrum added successfully. Clear the form for a new entry?"):
                    self.clear_add_spectrum_fields()
                    
            else:
                messagebox.showerror("Error", "Failed to add spectrum to database.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error adding spectrum to database: {str(e)}")
    
    def browse_spectrum_file(self):
        """Browse for a spectrum file."""
        file_path = filedialog.askopenfilename(
            title="Select Spectrum File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.spectrum_file_var.set(file_path)
            
            # Auto-fill spectrum name from filename if empty
            if not self.add_spectrum_name_var.get().strip():
                filename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(filename)[0]
                self.add_spectrum_name_var.set(name_without_ext)

    def load_spectrum_preview(self):
        """Load and preview the selected spectrum file."""
        file_path = self.spectrum_file_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid spectrum file.")
            return
            
        try:
            # Load spectrum data
            data = np.loadtxt(file_path)
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
            
            # Update preview plot
            self.preview_ax.clear()
            self.preview_ax.plot(wavenumbers, intensities, 'b-', linewidth=1.0, label=os.path.basename(file_path))
            self.preview_ax.set_xlabel('Wavenumber (cm^-1)')
            self.preview_ax.set_ylabel('Intensity (a.u.)')
            self.preview_ax.set_title(f'Raman Spectrum: {os.path.basename(file_path)}')
            self.preview_fig.tight_layout()
            self.preview_canvas.draw()
            
            # Store loaded data
            self.loaded_wavenumbers = wavenumbers
            self.loaded_intensities = intensities
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading spectrum: {str(e)}")

    def clear_add_spectrum_fields(self):
        """Clear all fields in the Add Spectrum tab."""
        # Clear file selection
        self.spectrum_file_var.set("")
        
        # Clear basic information
        self.add_spectrum_name_var.set("")
        self.add_mineral_name_var.set("")
        self.add_formula_var.set("")
        self.add_chemical_family_var.set("")
        self.add_description_text.delete(1.0, tk.END)
        
        # Clear classification
        self.add_hey_class_var.set("")
        self.add_hey_celestian_var.set("")
        self.add_crystal_system_var.set("")
        self.add_space_group_var.set("")
        
        # Clear measurement conditions
        self.add_laser_wavelength_var.set("")
        self.add_laser_power_var.set("")
        self.add_acquisition_time_var.set("")
        self.add_accumulations_var.set("")
        self.add_instrument_var.set("")
        self.add_polarization_var.set("")
        
        # Clear sample information
        self.add_locality_var.set("")
        self.add_sample_id_var.set("")
        self.add_owner_var.set("")
        
        # Clear preview plot
        self.preview_ax.clear()
        self.preview_ax.text(0.5, 0.5, 'Load a spectrum file to see preview', 
                           ha='center', va='center', transform=self.preview_ax.transAxes)
        self.preview_ax.set_xlabel('Wavenumber (cm^-1)')
        self.preview_ax.set_ylabel('Intensity (a.u.)')
        self.preview_fig.tight_layout()
        self.preview_canvas.draw()
        
        # Clear loaded data
        self.loaded_wavenumbers = None
        self.loaded_intensities = None

    def create_batch_edit_tab_content(self):
        """Create content for the Batch Edit tab."""
        # Create main container with PanedWindow for adjustable layout
        main_container = ttk.Frame(self.batch_edit_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for search criteria and edit fields
        left_panel = ttk.Frame(paned)
        right_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=2)
        paned.add(right_panel, weight=3)
        
        # ===== LEFT PANEL: Search Criteria and Edit Fields =====
        
        # Search criteria frame
        search_frame = ttk.LabelFrame(left_panel, text="Search Criteria", padding=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Name pattern search
        ttk.Label(search_frame, text="Name contains:").pack(anchor=tk.W)
        self.batch_name_pattern_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=self.batch_name_pattern_var, width=30).pack(fill=tk.X, pady=2)
        
        # Metadata filters
        ttk.Label(search_frame, text="Hey Classification:").pack(anchor=tk.W, pady=(10, 0))
        self.batch_filter_hey_var = tk.StringVar()
        hey_filter_combo = ttk.Combobox(search_frame, textvariable=self.batch_filter_hey_var, width=30)
        hey_filter_combo['values'] = ['', 'Native Elements', 'Sulfides and Sulfosalts', 'Oxides and Hydroxides',
                                      'Halides', 'Carbonates', 'Nitrates', 'Borates', 'Sulfates', 
                                      'Phosphates', 'Silicates', 'Unknown']
        hey_filter_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(search_frame, text="Chemical Family:").pack(anchor=tk.W, pady=(5, 0))
        self.batch_filter_family_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=self.batch_filter_family_var, width=30).pack(fill=tk.X, pady=2)
        
        ttk.Label(search_frame, text="Mineral Name:").pack(anchor=tk.W, pady=(5, 0))
        self.batch_filter_mineral_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=self.batch_filter_mineral_var, width=30).pack(fill=tk.X, pady=2)
        
        # Search buttons
        search_button_frame = ttk.Frame(search_frame)
        search_button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(search_button_frame, text="Search", command=self.batch_search_spectra).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(search_button_frame, text="Clear Filters", command=self.clear_batch_filters).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Edit fields frame
        edit_frame = ttk.LabelFrame(left_panel, text="Batch Edit Fields", padding=10)
        edit_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Chemical Family
        chem_family_frame = ttk.Frame(edit_frame)
        chem_family_frame.pack(fill=tk.X, pady=2)
        self.batch_edit_family_var = tk.BooleanVar()
        ttk.Checkbutton(chem_family_frame, text="Chemical Family:", variable=self.batch_edit_family_var).pack(side=tk.LEFT)
        self.batch_new_family_var = tk.StringVar()
        ttk.Entry(chem_family_frame, textvariable=self.batch_new_family_var, width=20).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Hey Classification
        hey_frame = ttk.Frame(edit_frame)
        hey_frame.pack(fill=tk.X, pady=2)
        self.batch_edit_hey_var = tk.BooleanVar()
        ttk.Checkbutton(hey_frame, text="Hey Classification:", variable=self.batch_edit_hey_var).pack(side=tk.LEFT)
        self.batch_new_hey_var = tk.StringVar()
        hey_combo = ttk.Combobox(hey_frame, textvariable=self.batch_new_hey_var, width=20, state='readonly')
        hey_combo['values'] = ('Native Elements', 'Sulfides and Sulfosalts', 'Oxides and Hydroxides',
                              'Halides', 'Carbonates', 'Nitrates', 'Borates', 'Sulfates', 
                              'Phosphates', 'Silicates')
        hey_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Hey-Celestian Classification
        hey_cel_frame = ttk.Frame(edit_frame)
        hey_cel_frame.pack(fill=tk.X, pady=2)
        self.batch_edit_hey_celestian_var = tk.BooleanVar()
        ttk.Checkbutton(hey_cel_frame, text="Hey-Celestian:", variable=self.batch_edit_hey_celestian_var).pack(side=tk.LEFT)
        self.batch_new_hey_celestian_var = tk.StringVar()
        hey_cel_combo = ttk.Combobox(hey_cel_frame, textvariable=self.batch_new_hey_celestian_var, width=20, state='readonly')
        hey_cel_combo['values'] = ('Framework Silicates', 'Chain Silicates', 'Sheet Silicates', 'Tectosilicates',
                                   'Inosilicates', 'Phyllosilicates', 'Cyclosilicates', 'Sorosilicates', 
                                   'Nesosilicates', 'Paired Tetrahedral Silicates', 'Ring Silicates',
                                   'Carbonates - Calcite Group', 'Carbonates - Aragonite Group', 
                                   'Carbonates - Dolomite Group', 'Carbonates - Other',
                                   'Sulfates - Barite Group', 'Sulfates - Gypsum Group', 
                                   'Sulfates - Anhydrite Group', 'Sulfates - Other',
                                   'Phosphates - Apatite Group', 'Phosphates - Other',
                                   'Oxides - Spinel Group', 'Oxides - Hematite Group', 
                                   'Oxides - Rutile Group', 'Oxides - Corundum Group', 'Oxides - Other',
                                   'Hydroxides', 'Halides', 'Sulfides', 'Native Elements', 'Borates',
                                   'Tungstates/Molybdates', 'Vanadates/Arsenates', 'Nitrates', 'Organic Minerals')
        hey_cel_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Description
        desc_frame = ttk.Frame(edit_frame)
        desc_frame.pack(fill=tk.X, pady=2)
        self.batch_edit_description_var = tk.BooleanVar()
        ttk.Checkbutton(desc_frame, text="Description:", variable=self.batch_edit_description_var).pack(anchor=tk.W)
        self.batch_new_description_text = tk.Text(edit_frame, height=3, width=30)
        self.batch_new_description_text.pack(fill=tk.X, pady=2)
        
        # Description mode
        desc_mode_frame = ttk.Frame(edit_frame)
        desc_mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(desc_mode_frame, text="Description mode:").pack(side=tk.LEFT)
        self.batch_description_mode_var = tk.StringVar(value="replace")
        ttk.Radiobutton(desc_mode_frame, text="Replace", variable=self.batch_description_mode_var, value="replace").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(desc_mode_frame, text="Append", variable=self.batch_description_mode_var, value="append").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(desc_mode_frame, text="Prepend", variable=self.batch_description_mode_var, value="prepend").pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        action_frame = ttk.Frame(left_panel)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Preview Changes", command=self.preview_batch_changes).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(action_frame, text="Apply Changes", command=self.apply_batch_changes).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # ===== RIGHT PANEL: Results and Preview =====
        
        # Search results frame
        results_frame = ttk.LabelFrame(right_panel, text="Search Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Results info
        results_info_frame = ttk.Frame(results_frame)
        results_info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.batch_results_info_var = tk.StringVar(value="No search performed")
        ttk.Label(results_info_frame, textvariable=self.batch_results_info_var).pack(side=tk.LEFT)
        
        ttk.Button(results_info_frame, text="Select All", command=self.select_all_batch_results).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(results_info_frame, text="Deselect All", command=self.deselect_all_batch_results).pack(side=tk.RIGHT, padx=(0, 5))
        
        # Results tree with checkboxes
        tree_frame = ttk.Frame(results_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("name", "mineral", "hey_class", "chemical_family", "description")
        self.batch_results_tree = ttk.Treeview(tree_frame, columns=columns, show='tree headings', height=15)
        
        # Configure columns
        self.batch_results_tree.heading("#0", text="☐")
        self.batch_results_tree.heading("name", text="Spectrum Name")
        self.batch_results_tree.heading("mineral", text="Mineral")
        self.batch_results_tree.heading("hey_class", text="Hey Class")
        self.batch_results_tree.heading("chemical_family", text="Chemical Family")
        self.batch_results_tree.heading("description", text="Description")
        
        self.batch_results_tree.column("#0", width=30, minwidth=30)
        self.batch_results_tree.column("name", width=200, minwidth=150)
        self.batch_results_tree.column("mineral", width=150, minwidth=100)
        self.batch_results_tree.column("hey_class", width=150, minwidth=100)
        self.batch_results_tree.column("chemical_family", width=120, minwidth=80)
        self.batch_results_tree.column("description", width=200, minwidth=150)
        
        # Add scrollbar
        batch_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.batch_results_tree.yview)
        batch_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_results_tree.configure(yscrollcommand=batch_scrollbar.set)
        self.batch_results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind click event for checkbox toggle
        self.batch_results_tree.bind("<Button-1>", self.toggle_batch_selection)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(right_panel, text="Changes Preview", padding=10)
        preview_frame.pack(fill=tk.X, pady=(0, 0))
        
        self.batch_preview_text = tk.Text(preview_frame, height=8, width=50, wrap=tk.WORD)
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.batch_preview_text.yview)
        self.batch_preview_text.config(yscrollcommand=preview_scrollbar.set)
        
        self.batch_preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize tracking variables
        self.batch_selected_items = set()
        self.batch_search_results = []

    def batch_search_spectra(self):
        """Search for spectra based on the criteria."""
        name_pattern = self.batch_name_pattern_var.get().strip().lower()
        hey_filter = self.batch_filter_hey_var.get().strip()
        family_filter = self.batch_filter_family_var.get().strip().lower()
        mineral_filter = self.batch_filter_mineral_var.get().strip().lower()
        
        # Clear previous results
        for item in self.batch_results_tree.get_children():
            self.batch_results_tree.delete(item)
        self.batch_selected_items.clear()
        self.batch_search_results.clear()
        
        # Search through database
        matching_spectra = []
        
        for name, data in self.db.database.items():
            if name.startswith('__'):  # Skip metadata entries
                continue
                
            metadata = data.get('metadata', {})
            
            # Check name pattern
            if name_pattern and name_pattern not in name.lower():
                continue
                
            # Check Hey Classification
            if hey_filter:
                hey_class = metadata.get('HEY CLASSIFICATION', metadata.get('Hey Classification', ''))
                if hey_filter != hey_class:
                    continue
                    
            # Check Chemical Family
            if family_filter:
                chemical_family = metadata.get('CHEMICAL FAMILY', metadata.get('Chemical Family', '')).lower()
                if family_filter not in chemical_family:
                    continue
                    
            # Check Mineral Name
            if mineral_filter:
                mineral_name = metadata.get('NAME', metadata.get('Mineral Name', '')).lower()
                if mineral_filter not in mineral_name:
                    continue
            
            matching_spectra.append((name, metadata))
        
        # Populate results tree
        for name, metadata in sorted(matching_spectra):
            mineral_name = metadata.get('NAME', metadata.get('Mineral Name', 'N/A'))
            hey_class = metadata.get('HEY CLASSIFICATION', metadata.get('Hey Classification', 'N/A'))
            chemical_family = metadata.get('CHEMICAL FAMILY', metadata.get('Chemical Family', 'N/A'))
            description = metadata.get('DESCRIPTION', metadata.get('Description', 'N/A'))
            
            # Truncate long descriptions
            if len(str(description)) > 50:
                description = str(description)[:47] + "..."
            
            item_id = self.batch_results_tree.insert("", tk.END, text="☐",
                                                   values=(name, mineral_name, hey_class, 
                                                         chemical_family, description))
            self.batch_search_results.append(name)
        
        # Update info
        count = len(matching_spectra)
        self.batch_results_info_var.set(f"Found {count} matching spectra")
        
        # Clear preview
        self.batch_preview_text.delete(1.0, tk.END)
        self.batch_preview_text.insert(tk.END, f"Search completed. Found {count} spectra.\nSelect items and configure edit fields, then click 'Preview Changes'.")

    def clear_batch_filters(self):
        """Clear all search filters."""
        self.batch_name_pattern_var.set("")
        self.batch_filter_hey_var.set("")
        self.batch_filter_family_var.set("")
        self.batch_filter_mineral_var.set("")
        
        # Clear results
        for item in self.batch_results_tree.get_children():
            self.batch_results_tree.delete(item)
        self.batch_selected_items.clear()
        self.batch_search_results.clear()
        self.batch_results_info_var.set("No search performed")
        self.batch_preview_text.delete(1.0, tk.END)

    def toggle_batch_selection(self, event):
        """Toggle selection of batch edit items."""
        item = self.batch_results_tree.identify('item', event.x, event.y)
        if item:
            if item in self.batch_selected_items:
                self.batch_selected_items.remove(item)
                self.batch_results_tree.item(item, text="☐")
            else:
                self.batch_selected_items.add(item)
                self.batch_results_tree.item(item, text="☑")

    def select_all_batch_results(self):
        """Select all items in the batch results."""
        for item in self.batch_results_tree.get_children():
            self.batch_selected_items.add(item)
            self.batch_results_tree.item(item, text="☑")

    def deselect_all_batch_results(self):
        """Deselect all items in the batch results."""
        for item in self.batch_results_tree.get_children():
            self.batch_selected_items.discard(item)
            self.batch_results_tree.item(item, text="☐")

    def preview_batch_changes(self):
        """Preview the changes that will be made."""
        if not self.batch_selected_items:
            messagebox.showwarning("No Selection", "Please select at least one spectrum to edit.")
            return
        
        # Get selected spectrum names
        selected_names = []
        for item in self.batch_selected_items:
            values = self.batch_results_tree.item(item, 'values')
            if values:
                selected_names.append(values[0])  # First column is spectrum name
        
        if not selected_names:
            return
        
        # Clear preview
        self.batch_preview_text.delete(1.0, tk.END)
        
        # Check which fields are being edited
        changes_to_make = []
        
        if self.batch_edit_family_var.get():
            new_family = self.batch_new_family_var.get().strip()
            if new_family:
                changes_to_make.append(('Chemical Family', new_family))
        
        if self.batch_edit_hey_var.get():
            new_hey = self.batch_new_hey_var.get().strip()
            if new_hey:
                changes_to_make.append(('Hey Classification', new_hey))
        
        if self.batch_edit_hey_celestian_var.get():
            new_hey_cel = self.batch_new_hey_celestian_var.get().strip()
            if new_hey_cel:
                changes_to_make.append(('Hey-Celestian Classification', new_hey_cel))
        
        if self.batch_edit_description_var.get():
            new_desc = self.batch_new_description_text.get(1.0, tk.END).strip()
            if new_desc:
                mode = self.batch_description_mode_var.get()
                changes_to_make.append(('Description', f"{new_desc} ({mode})"))
        
        if not changes_to_make:
            self.batch_preview_text.insert(tk.END, "No changes configured. Please check the fields you want to edit and enter new values.")
            return
        
        # Generate preview
        self.batch_preview_text.insert(tk.END, f"BATCH EDIT PREVIEW\n")
        self.batch_preview_text.insert(tk.END, f"=" * 50 + "\n\n")
        self.batch_preview_text.insert(tk.END, f"Selected spectra: {len(selected_names)}\n\n")
        
        self.batch_preview_text.insert(tk.END, "Changes to apply:\n")
        for field, value in changes_to_make:
            self.batch_preview_text.insert(tk.END, f"  • {field}: {value}\n")
        self.batch_preview_text.insert(tk.END, "\n")
        
        # Show examples of changes for first few spectra
        self.batch_preview_text.insert(tk.END, "Examples (first 5 spectra):\n")
        self.batch_preview_text.insert(tk.END, "-" * 30 + "\n")
        
        for i, name in enumerate(selected_names[:5]):
            if name in self.db.database:
                self.batch_preview_text.insert(tk.END, f"\n{i+1}. {name}:\n")
                metadata = self.db.database[name].get('metadata', {})
                
                for field, new_value in changes_to_make:
                    if field == 'Chemical Family':
                        old_value = metadata.get('CHEMICAL FAMILY', metadata.get('Chemical Family', 'N/A'))
                        self.batch_preview_text.insert(tk.END, f"   Chemical Family: '{old_value}' → '{new_value}'\n")
                    elif field == 'Hey Classification':
                        old_value = metadata.get('HEY CLASSIFICATION', metadata.get('Hey Classification', 'N/A'))
                        self.batch_preview_text.insert(tk.END, f"   Hey Classification: '{old_value}' → '{new_value}'\n")
                    elif field == 'Hey-Celestian Classification':
                        old_value = metadata.get('HEY CELESTIAN CLASSIFICATION', metadata.get('Hey-Celestian Classification', 'N/A'))
                        self.batch_preview_text.insert(tk.END, f"   Hey-Celestian: '{old_value}' → '{new_value.split('(')[0].strip()}'\n")
                    elif field.startswith('Description'):
                        old_desc = metadata.get('DESCRIPTION', metadata.get('Description', 'N/A'))
                        mode = new_value.split('(')[-1].rstrip(')')
                        new_desc = new_value.split('(')[0].strip()
                        
                        if mode == 'replace':
                            result_desc = new_desc
                        elif mode == 'append':
                            result_desc = f"{old_desc} {new_desc}"
                        elif mode == 'prepend':
                            result_desc = f"{new_desc} {old_desc}"
                        
                        self.batch_preview_text.insert(tk.END, f"   Description ({mode}): '{old_desc[:30]}...' → '{result_desc[:30]}...'\n")
        
        if len(selected_names) > 5:
            self.batch_preview_text.insert(tk.END, f"\n... and {len(selected_names) - 5} more spectra\n")
        
        self.batch_preview_text.insert(tk.END, f"\nClick 'Apply Changes' to proceed with the batch edit.")

    def apply_batch_changes(self):
        """Apply the batch changes to selected spectra."""
        if not self.batch_selected_items:
            messagebox.showwarning("No Selection", "Please select at least one spectrum to edit.")
            return
        
        # Get selected spectrum names
        selected_names = []
        for item in self.batch_selected_items:
            values = self.batch_results_tree.item(item, 'values')
            if values:
                selected_names.append(values[0])
        
        if not selected_names:
            return
        
        # Check which fields are being edited
        changes_to_make = []
        
        if self.batch_edit_family_var.get():
            new_family = self.batch_new_family_var.get().strip()
            if new_family:
                changes_to_make.append(('family', new_family))
        
        if self.batch_edit_hey_var.get():
            new_hey = self.batch_new_hey_var.get().strip()
            if new_hey:
                changes_to_make.append(('hey', new_hey))
        
        if self.batch_edit_hey_celestian_var.get():
            new_hey_cel = self.batch_new_hey_celestian_var.get().strip()
            if new_hey_cel:
                changes_to_make.append(('hey_celestian', new_hey_cel))
        
        if self.batch_edit_description_var.get():
            new_desc = self.batch_new_description_text.get(1.0, tk.END).strip()
            if new_desc:
                mode = self.batch_description_mode_var.get()
                changes_to_make.append(('description', (new_desc, mode)))
        
        if not changes_to_make:
            messagebox.showerror("No Changes", "No changes configured. Please check the fields you want to edit and enter new values.")
            return
        
        # Confirm the operation
        result = messagebox.askyesno(
            "Confirm Batch Edit",
            f"Apply changes to {len(selected_names)} selected spectra?\n\n"
            f"This will modify the database. This action cannot be undone.\n\n"
            f"Click 'Yes' to proceed or 'No' to cancel."
        )
        
        if not result:
            return
        
        # Apply changes
        updated_count = 0
        error_count = 0
        
        for name in selected_names:
            try:
                if name in self.db.database:
                    metadata = self.db.database[name].get('metadata', {})
                    
                    for change_type, new_value in changes_to_make:
                        if change_type == 'family':
                            metadata['CHEMICAL FAMILY'] = new_value
                            metadata['Chemical Family'] = new_value
                        elif change_type == 'hey':
                            metadata['HEY CLASSIFICATION'] = new_value
                            metadata['Hey Classification'] = new_value
                        elif change_type == 'hey_celestian':
                            metadata['HEY CELESTIAN CLASSIFICATION'] = new_value
                            metadata['Hey-Celestian Classification'] = new_value
                        elif change_type == 'description':
                            new_desc, mode = new_value
                            old_desc = metadata.get('DESCRIPTION', metadata.get('Description', ''))
                            
                            if mode == 'replace':
                                result_desc = new_desc
                            elif mode == 'append':
                                result_desc = f"{old_desc} {new_desc}".strip()
                            elif mode == 'prepend':
                                result_desc = f"{new_desc} {old_desc}".strip()
                            
                            metadata['DESCRIPTION'] = result_desc
                            metadata['Description'] = result_desc
                    
                    # Update the database entry
                    self.db.database[name]['metadata'] = metadata
                    updated_count += 1
                    
            except Exception as e:
                print(f"Error updating {name}: {e}")
                error_count += 1
        
        # Save database
        if self.db.save_database():
            # Update the results display
            self.batch_search_spectra()  # Refresh search results
            self.update_spectrum_list()  # Update main browser list
            self.update_stats()  # Update statistics
            
            # Show success message
            message = f"Batch edit completed!\n\n"
            message += f"Successfully updated: {updated_count} spectra\n"
            if error_count > 0:
                message += f"Errors encountered: {error_count} spectra\n"
            message += f"\nDatabase saved successfully."
            
            messagebox.showinfo("Batch Edit Complete", message)
            
            # Clear preview
            self.batch_preview_text.delete(1.0, tk.END)
            self.batch_preview_text.insert(tk.END, f"Batch edit completed. {updated_count} spectra updated successfully.")
            
        else:
            messagebox.showerror("Save Error", "Changes were applied but failed to save database.")


def main():
    """Main function for standalone use."""
    app = RamanDatabaseGUI()
    app.run()


if __name__ == "__main__":
    main() 