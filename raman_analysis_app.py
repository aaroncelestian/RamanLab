#!/usr/bin/env python3
# Raman Spectrum Analysis Tool - GUI Application
# GUI for importing, analyzing, and identifying Raman spectra
"""
@author: AaronCelestian
ClaritySpectra
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg to prevent blank figure window
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import find_peaks
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
from pathlib import Path
import csv
from datetime import datetime
# from search_functions import plot_match_comparison, generate_correlation_heatmap, generate_match_report
# import types


# Import the RamanSpectra class
from raman_spectra import RamanSpectra

# Try importing reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Try importing scikit-learn for ML search
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False






class RamanAnalysisApp:
    """ClaritySpectra: GUI application for Raman spectrum analysis."""

    def __init__(self, root):
        """
        Initialize the GUI application.

        Parameters:
        -----------
        root : tk.Tk
            Root Tkinter window.
        """
        self.root = root
        self.root.title("ClaritySpectra: Raman Spectrum Analysis")
        self.root.geometry("1400x800")

        # Set minimum window size
        self.root.minsize(900, 600)
        
        # Create menu bar
        self.create_menu_bar()

        # Create Raman spectra instance
        self.raman = RamanSpectra()

        # Create GUI elements - this initializes all the UI components
        self.create_gui()
        
        # Create instance variable for db_stats_text as a backup precaution
        # It will be properly initialized in create_database_tab, but this ensures it exists
        if not hasattr(self, 'db_stats_text'):
            self.db_stats_text = None
            
        # Now that UI components are created, we can update database stats
        # Only if db_stats_text has been properly initialized
        if self.db_stats_text is not None:
            self.update_database_stats()

        

    def create_gui(self):
        """Create the GUI elements."""
        # Main frames
        self.frame_left = ttk.Frame(self.root, padding=10)
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_right = ttk.Frame(self.root, padding=10)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.Y)

        # Left panel - visualization
        self.create_visualization_panel()

        # Right panel - controls
        self.create_control_panel()

    def create_visualization_panel(self):
        """Create the visualization panel."""
        # Create visualization frame
        viz_frame = ttk.LabelFrame(self.frame_left, text="Spectrum Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Create initial empty figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax.set_ylabel('Intensity (a.u.)')
        self.ax.set_title('Raman Spectrum')

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()


    def create_menu_bar(self):
        """Create the application menu bar."""
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Spectrum", command=self.import_spectrum)
        file_menu.add_command(label="Save Current Spectrum", command=self.save_spectrum)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Process menu
        process_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Process", menu=process_menu)
        process_menu.add_command(label="Subtract Background", command=self.subtract_background)
        process_menu.add_command(label="Find Peaks", command=self.find_peaks)
        
        # Database menu
        database_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Database", menu=database_menu)
        database_menu.add_command(label="Add Current Spectrum", command=lambda: self.add_to_database())
        database_menu.add_command(label="View/Search Database", command=self.view_database)
        database_menu.add_command(label="Batch Import", command=self.batch_import)
        
        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about_dialog)
        


    def show_about_dialog(self):
        """Show the About dialog with application information."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About ClaritySpectra")
        about_window.geometry("400x400")
        about_window.resizable(False, False)
    
        # Make window modal
        about_window.transient(self.root)
        about_window.grab_set()
        
        # Application icon or logo could be added here
        
        # App title
        ttk.Label(
            about_window, 
            text="ClaritySpectra", 
            font=("TkDefaultFont", 14, "bold")
        ).pack(pady=(20, 5))
        
        # Version info
        ttk.Label(
            about_window, 
            text="Version 1.0"
        ).pack(pady=2)
        
        # Author info
        ttk.Label(
            about_window,
            text="Created by Aaron Celestian, Ph.D. \n"
                 "Natural History Museum of Los Angeles County \n"
                 "Department of Mineral Sciences",
        ).pack(pady=2)
        
        # Description
        description = ttk.Label(
            about_window,
            text="ClaritySpectra: A comprehensive tool for analyzing Raman spectra,\n"
                 "with features for background subtraction, peak finding,\n"
                 "and spectrum identification using database matching.",
            justify=tk.CENTER
        )
        description.pack(pady=10)
        
        # Copyright info
        ttk.Label(
            about_window,
            text="© 2025 Aaron Celestian. All rights reserved\n"
                 "RRUFF data use for minerals\n"
                 "https://www.rruff.info \n"
                 "Plastic data from SLOPP and SLOPP-E \n"
                 "https://tinyurl.com/2ek3zceb",
        ).pack(pady=10)
        
        # Close button
        ttk.Button(
            about_window, 
            text="Close", 
            command=about_window.destroy
        ).pack(pady=10)



    def create_control_panel(self):
        """Create the control panel with tabbed interface."""
        # Define frame width to ensure controls don't take too much space
        control_width = 300

        # Create notebook (tabbed panel)
        self.notebook = ttk.Notebook(self.frame_right, width=control_width)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.tab_file = ttk.Frame(self.notebook, padding=5)
        self.tab_process = ttk.Frame(self.notebook, padding=5)
        self.tab_search = ttk.Frame(self.notebook, padding=5)
        self.tab_database = ttk.Frame(self.notebook, padding=5)

        # Add tabs to notebook
        self.notebook.add(self.tab_file, text="File")
        self.notebook.add(self.tab_process, text="Processing")
        self.notebook.add(self.tab_search, text="Search-Match")
        self.notebook.add(self.tab_database, text="Database")

        # Create content for each tab
        self.create_file_tab()
        self.create_process_tab()
        self.create_search_tab()
        self.create_database_tab()

    def create_file_tab(self):
        """Create content for the file operations tab."""
        # File operations frame
        file_frame = ttk.LabelFrame(self.tab_file, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Button(file_frame, text="Import Spectrum", command=self.import_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Save Current Spectrum", command=self.save_spectrum).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Edit Metadata", command=self.edit_metadata).pack(fill=tk.X, pady=2)

        # Metadata display frame
        self.metadata_frame = ttk.LabelFrame(self.tab_file, text="Spectrum Metadata", padding=10)
        self.metadata_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.metadata_text = tk.Text(self.metadata_frame, height=10, width=30, wrap=tk.WORD)
        self.metadata_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.metadata_text.config(state=tk.DISABLED)

        # Add scrollbar to metadata
        scrollbar = ttk.Scrollbar(self.metadata_frame, orient=tk.VERTICAL, command=self.metadata_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.metadata_text.config(yscrollcommand=scrollbar.set)

    def create_process_tab(self):
        """Create content for the spectrum processing tab."""
        # Background subtraction frame
        bg_frame = ttk.LabelFrame(self.tab_process, text="Background Subtraction", padding=10)
        bg_frame.pack(fill=tk.X, pady=5)

        ttk.Label(bg_frame, text="Baseline λ:").pack(anchor=tk.W)
        self.var_lambda = tk.StringVar(value="1e5")
        ttk.Entry(bg_frame, textvariable=self.var_lambda).pack(fill=tk.X, pady=2)

        ttk.Label(bg_frame, text="Baseline p:").pack(anchor=tk.W)
        self.var_p = tk.StringVar(value="0.01")
        ttk.Entry(bg_frame, textvariable=self.var_p).pack(fill=tk.X, pady=2)

        ttk.Button(bg_frame, text="Subtract Background", command=self.subtract_background).pack(fill=tk.X, pady=5)

        # Peak finding frame
        peak_frame = ttk.LabelFrame(self.tab_process, text="Peak Finding", padding=10)
        peak_frame.pack(fill=tk.X, pady=5)

        ttk.Label(peak_frame, text="Peak Height Threshold: [0.1 - 1]").pack(anchor=tk.W)
        self.var_height = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_height).pack(fill=tk.X, pady=2)

        ttk.Label(peak_frame, text="Min. Peak Distance: [ca. 1 - 20]").pack(anchor=tk.W)
        self.var_distance = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_distance).pack(fill=tk.X, pady=2)

        ttk.Label(peak_frame, text="Peak Prominence: [0.5 - 5]").pack(anchor=tk.W)
        self.var_prominence = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_prominence).pack(fill=tk.X, pady=2)

        ttk.Button(peak_frame, text="Find Peaks", command=self.find_peaks).pack(fill=tk.X, pady=5)

        # Display options frame
        display_frame = ttk.LabelFrame(self.tab_process, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=5)

        self.var_show_background = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame,
            text="Show Background",
            variable=self.var_show_background,
            command=lambda: self.update_plot(include_background=self.var_show_background.get(),
                                            include_peaks=self.var_show_peaks.get())
        ).pack(anchor=tk.W, pady=2)

        self.var_show_peaks = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame,
            text="Show Peaks",
            variable=self.var_show_peaks,
            command=lambda: self.update_plot(include_background=self.var_show_background.get(),
                                            include_peaks=self.var_show_peaks.get())
        ).pack(anchor=tk.W, pady=2)

    def create_search_tab(self):
        """Create content for the search-match tab."""
        # Create notebook for search options
        search_notebook = ttk.Notebook(self.tab_search)
        search_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create tabs for basic and advanced search
        basic_search_tab = ttk.Frame(search_notebook)
        advanced_search_tab = ttk.Frame(search_notebook)

        search_notebook.add(basic_search_tab, text="Basic Search")
        search_notebook.add(advanced_search_tab, text="Advanced Search")

        # Basic search tab
        self.create_basic_search_tab(basic_search_tab)

        # Advanced search tab
        self.create_advanced_search_tab(advanced_search_tab)

        # Results display (common to both tabs)
        results_frame = ttk.LabelFrame(self.tab_search, text="Results Summary", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.match_result_text = tk.Text(results_frame, height=15, width=30, wrap=tk.WORD)
        self.match_result_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Add scrollbar to results summary
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.match_result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.match_result_text.config(yscrollcommand=scrollbar.set, state=tk.DISABLED)

    

    


    def update_hey_classification_database(self):
        """Update Hey Classification for all entries in the database."""
        if not self.raman.database:
            messagebox.showinfo("Database", "The database is empty.")
            return
        
        # Check if Hey Classification data is loaded
        if not hasattr(self.raman, 'hey_classification') or not self.raman.hey_classification:
            # Try to load Hey Classification data if not already loaded
            hey_csv_path = "RRUFF_Export_with_Hey_Classification.csv"
            if os.path.exists(hey_csv_path):
                self.raman.hey_classification = self.raman.load_hey_classification(hey_csv_path)
            
            # Check again if data was loaded
            if not hasattr(self.raman, 'hey_classification') or not self.raman.hey_classification:
                messagebox.showerror("Error", "Hey Classification data could not be loaded.")
                return
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Updating Hey Classification")
        progress_window.geometry("500x350")
        
        # Create progress bar
        ttk.Label(progress_window, text="Updating Hey Classification for database entries...").pack(pady=10)
        progress = ttk.Progressbar(progress_window, length=400, mode="determinate")
        progress.pack(pady=10)
        
        # Create status label
        status_var = tk.StringVar(value="Starting update...")
        status_label = ttk.Label(progress_window, textvariable=status_var)
        status_label.pack(pady=5)
        
        # Create log text area
        log_frame = ttk.Frame(progress_window)
        log_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        log_text = tk.Text(log_frame, height=10, width=50, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
        log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure tags for log text
        log_text.tag_configure("success", foreground="green")
        log_text.tag_configure("warning", foreground="orange")
        log_text.tag_configure("error", foreground="red")
        log_text.tag_configure("info", foreground="blue")
        
        # Get database items and update progress bar maximum
        db_items = list(self.raman.database.items())
        progress["maximum"] = len(db_items)
        
        # Statistics counters
        total = len(db_items)
        updated = 0
        already_had = 0
        not_found = 0
        
        # Update function for progress
        def update_progress(current, name, status, log_message=None, tag="info"):
            progress["value"] = current
            percentage = int((current / total) * 100) if total > 0 else 0
            status_var.set(f"{status} - {percentage}% complete ({current}/{total})")
            
            if log_message:
                log_text.insert(tk.END, log_message + "\n", tag)
                log_text.see(tk.END)
            
            progress_window.update_idletasks()
        
        # Process each database entry
        for i, (name, data) in enumerate(db_items):
            # Update progress display
            update_progress(i, name, f"Processing {name}")
            
            # Skip entries without metadata
            if 'metadata' not in data or not data['metadata']:
                # Try to create metadata if it doesn't exist
                if 'metadata' not in data:
                    data['metadata'] = {}
                
                # Add a note to log
                update_progress(i, name, f"Processing {name}", 
                              f"Note: Created metadata for {name}", "warning")
            
            metadata = data['metadata']
            
            # Check if Hey Classification already exists
            if 'HEY CLASSIFICATION' in metadata and metadata['HEY CLASSIFICATION']:
                already_had += 1
                update_progress(i, name, f"Processing {name}", 
                              f"Skipped {name}: Already has Hey Classification '{metadata['HEY CLASSIFICATION']}'", "info")
                continue
            
            # Try to find mineral name using different strategies
            mineral_name = None
            
            # 1. Check if NAME field exists in metadata
            if 'NAME' in metadata and metadata['NAME']:
                mineral_name = metadata['NAME']
            
            # 2. If no NAME in metadata, try to extract from the spectrum name (the database key)
            if not mineral_name:
                # Split by common separators and take the first part as potential mineral name
                potential_name = name.split('__')[0].split('_')[0].split('-')[0].strip()
                
                # Remove numbers and special characters at the end of the potential name
                import re
                cleaned_name = re.sub(r'[0-9_]+$', '', potential_name).strip()
                
                if cleaned_name:
                    mineral_name = cleaned_name
                    # Add the extracted name to metadata for future use
                    metadata['NAME'] = mineral_name
                    update_progress(i, name, f"Processing {name}", 
                                  f"Extracted mineral name '{mineral_name}' from spectrum name", "info")
                
            # If still no mineral name, skip this entry
            if not mineral_name:
                update_progress(i, name, f"Processing {name}", 
                              f"Skipped {name}: No mineral name in metadata and could not extract from name", "warning")
                continue
            
            # Try to match using the new advanced matching function
            matched_name, hey_class = self.raman.match_mineral_name(mineral_name)
            
            if hey_class:
                # Update the metadata with Hey Classification
                metadata['HEY CLASSIFICATION'] = hey_class
                updated += 1
                update_progress(i, name, f"Processing {name}", 
                              f"Updated {name}: Added Hey Classification '{hey_class}' (matched with '{matched_name}')", "success")
            else:
                not_found += 1
                update_progress(i, name, f"Processing {name}", 
                              f"Could not find Hey Classification for '{mineral_name}'", "error")
        
        # Final progress update
        update_progress(total, "", "Complete", 
                      f"Update complete! Updated: {updated}, Already had: {already_had}, Not found: {not_found}", "info")
        
        # Save the database
        saved = self.raman.save_database()
        if saved:
            log_text.insert(tk.END, f"Database saved successfully to {self.raman.db_path}\n", "success")
        else:
            log_text.insert(tk.END, "Warning: Could not save database\n", "error")
        log_text.see(tk.END)
        
        # Add close button
        ttk.Button(progress_window, text="Close", command=progress_window.destroy).pack(pady=10)
        
        # Update main GUI database statistics and filter options
        self.update_database_stats()
        self.update_metadata_filter_options()
        
        # Make progress window modal
        progress_window.transient(self.root)
        progress_window.grab_set()
        self.root.wait_window(progress_window)



    def view_hey_classification(self):
        """Create a window to browse Hey Classification hierarchy and associated minerals."""
        # Check if Hey Classification data is loaded
        if not hasattr(self.raman, 'hey_classification') or not self.raman.hey_classification:
            # Try to load Hey Classification data if not already loaded
            hey_csv_path = "RRUFF_Export_with_Hey_Classification.csv"
            if os.path.exists(hey_csv_path):
                self.raman.hey_classification = self.raman.load_hey_classification(hey_csv_path)
            
            # Check again if data was loaded
            if not hasattr(self.raman, 'hey_classification') or not self.raman.hey_classification:
                messagebox.showerror("Error", "Hey Classification data could not be loaded.")
                return
        
        # Create a new window
        hey_window = tk.Toplevel(self.root)
        hey_window.title("Hey Classification Browser")
        hey_window.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(hey_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a paned window for resizable sections
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for classification tree
        left_frame = ttk.Frame(paned, padding=5)
        paned.add(left_frame, weight=1)
        
        # Right frame for mineral list
        right_frame = ttk.Frame(paned, padding=5)
        paned.add(right_frame, weight=2)
        
        # Create tree structure for Hey Classification
        ttk.Label(left_frame, text="Hey Classification Hierarchy").pack(anchor=tk.W, pady=(0, 5))
        
        # Create treeview
        tree = ttk.Treeview(left_frame)
        tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar to tree
        tree_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Create listbox for minerals
        ttk.Label(right_frame, text="Minerals in Selected Classification").pack(anchor=tk.W, pady=(0, 5))
        
        # Search frame for minerals
        search_frame = ttk.Frame(right_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Mineral list with scrollbar
        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        mineral_listbox = tk.Listbox(list_frame, selectmode=tk.BROWSE)
        mineral_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=mineral_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        mineral_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        # Details section
        detail_frame = ttk.LabelFrame(right_frame, text="Mineral Details", padding=5)
        detail_frame.pack(fill=tk.X, pady=(5, 0))
        
        detail_text = tk.Text(detail_frame, height=8, wrap=tk.WORD)
        detail_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        detail_scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=detail_text.yview)
        detail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        detail_text.configure(yscrollcommand=detail_scrollbar.set)
        detail_text.config(state=tk.DISABLED)  # Read-only
        
        # Organize Hey Classification data by hierarchy
        hierarchy = {}
        minerals_by_class = {}
        
        # Function to get all the minerals for a specific Hey Classification
        def get_minerals_for_classification(classification):
            """Get all minerals for a specific Hey Classification."""
            if classification in minerals_by_class:
                return minerals_by_class[classification]
            
            minerals = []
            for mineral_name, class_name in self.raman.hey_classification.items():
                if class_name == classification:
                    minerals.append(mineral_name)
            
            minerals_by_class[classification] = sorted(minerals)
            return minerals_by_class[classification]
        
        # Analyze Hey Classification structure
        for mineral, classification in self.raman.hey_classification.items():
            # Skip if classification is missing
            if not classification:
                continue
            
            # Extract classification parts (assuming format like "X. Silicates - Tectosilicates")
            parts = classification.split(' - ')
            main_class = parts[0].strip()
            subclass = parts[1].strip() if len(parts) > 1 else ""
            
            # Build hierarchy
            if main_class not in hierarchy:
                hierarchy[main_class] = set()
            
            if subclass:
                hierarchy[main_class].add(subclass)
            
            # Build mineral lists by classification
            if classification not in minerals_by_class:
                minerals_by_class[classification] = []
            
            minerals_by_class[classification].append(mineral)
        
        # Sort minerals in each classification
        for classification in minerals_by_class:
            minerals_by_class[classification].sort()
        
        # Populate treeview with Hey Classification hierarchy
        # First, clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        # Add main classes as parent nodes
        for main_class in sorted(hierarchy.keys()):
            main_id = tree.insert('', 'end', text=main_class, values=(main_class, ''))
            
            # Add subclasses as child nodes
            subclasses = sorted(hierarchy[main_class]) if hierarchy[main_class] else []
            for subclass in subclasses:
                full_class = f"{main_class} - {subclass}"
                tree.insert(main_id, 'end', text=subclass, values=(full_class, ''))
        
        # Function to update mineral list when a classification is selected
        def on_classification_select(event):
            selected_items = tree.selection()
            if not selected_items:
                return
            
            # Get selected classification
            selected_item = selected_items[0]
            values = tree.item(selected_item, 'values')
            
            classification = values[0] if values else ""
            update_mineral_list(classification)
        
        # Function to update mineral list based on classification and search filter
        def update_mineral_list(classification=None, search_term=None):
            mineral_listbox.delete(0, tk.END)
            
            if not classification and not search_term:
                return
            
            # Get minerals for selected classification
            minerals = []
            if classification:
                minerals = get_minerals_for_classification(classification)
            else:
                # If no classification but searching, search all minerals
                minerals = sorted(self.raman.hey_classification.keys())
            
            # Apply search filter if provided
            if search_term:
                search_lower = search_term.lower()
                minerals = [m for m in minerals if search_lower in m.lower()]
            
            # Update listbox
            for mineral in minerals:
                mineral_listbox.insert(tk.END, mineral)
        
        # Function to display mineral details when selected
        def on_mineral_select(event):
            selected_indices = mineral_listbox.curselection()
            if not selected_indices:
                return
            
            # Get selected mineral
            selected_index = selected_indices[0]
            mineral_name = mineral_listbox.get(selected_index)
            
            # Clear detail text
            detail_text.config(state=tk.NORMAL)
            detail_text.delete(1.0, tk.END)
            
            # Add mineral name and classification
            hey_class = self.raman.hey_classification.get(mineral_name, "Unknown")
            detail_text.insert(tk.END, f"Mineral: {mineral_name}\n")
            detail_text.insert(tk.END, f"Hey Classification: {hey_class}\n\n")
            
            # Check if mineral is in database
            in_database = False
            spectra_count = 0
            
            # Look for mineral name in database (case insensitive)
            mineral_lower = mineral_name.lower()
            for name, data in self.raman.database.items():
                metadata = data.get('metadata', {})
                db_name = metadata.get('NAME', '')
                
                if db_name.lower() == mineral_lower:
                    in_database = True
                    spectra_count += 1
            
            if in_database:
                detail_text.insert(tk.END, f"In Database: Yes ({spectra_count} spectra)\n")
            else:
                detail_text.insert(tk.END, "In Database: No\n")
            
            detail_text.config(state=tk.DISABLED)
        
        # Function to search minerals
        def on_search(event=None):
            search_term = search_var.get().strip()
            
            # Get selected classification
            selected_items = tree.selection()
            classification = None
            if selected_items:
                values = tree.item(selected_items[0], 'values')
                classification = values[0] if values else None
            
            update_mineral_list(classification, search_term)
        
        # Bind events
        tree.bind('<<TreeviewSelect>>', on_classification_select)
        mineral_listbox.bind('<<ListboxSelect>>', on_mineral_select)
        search_entry.bind('<Return>', on_search)
        
        # Add search button
        ttk.Button(search_frame, text="Search", command=on_search).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(search_frame, text="Clear", 
                 command=lambda: [search_var.set(""), on_search()]).pack(side=tk.LEFT, padx=(5, 0))
        
        # Button to add selected mineral to database
        def import_selected_mineral():
            selected_indices = mineral_listbox.curselection()
            if not selected_indices:
                messagebox.showinfo("Selection Required", "Please select a mineral first.")
                return
            
            # Get selected mineral
            selected_index = selected_indices[0]
            mineral_name = mineral_listbox.get(selected_index)
            
            # Close the Hey Classification viewer
            hey_window.destroy()
            
            # Set the mineral name in the database tab for adding
            self.var_name.set(mineral_name)
            
            # Switch to the database tab
            self.notebook.select(self.tab_database)
            
            # Show message to guide the user
            messagebox.showinfo("Import Mineral", 
                              f"The mineral name '{mineral_name}' has been selected. \n\n"
                              "Please import a spectrum file first, then click 'Add Current Spectrum' "
                              "to add it to the database with this mineral name and Hey Classification.")
        
        # Add button to bottom of right frame
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Import Selected Mineral", command=import_selected_mineral).pack(side=tk.RIGHT)




# Add this button to the database_tab in create_database_tab method
    def create_database_tab(self):
        """Create content for the database operations tab."""
        # Add to database frame
        add_frame = ttk.LabelFrame(self.tab_database, text="Add to Database", padding=10)
        add_frame.pack(fill=tk.X, pady=5)

        ttk.Label(add_frame, text="Spectrum Name:").pack(anchor=tk.W)
        self.var_name = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.var_name).pack(fill=tk.X, pady=2)

        ttk.Button(add_frame, text="Add Current Spectrum", command=self.add_to_database).pack(fill=tk.X, pady=2)

        # Database management frame
        manage_frame = ttk.LabelFrame(self.tab_database, text="Database Management", padding=10)
        manage_frame.pack(fill=tk.X, pady=5)

        ttk.Button(manage_frame, text="Batch Import Spectra", command=self.batch_import).pack(fill=tk.X, pady=2)
        ttk.Button(manage_frame, text="View/Search Database", command=self.view_database).pack(fill=tk.X, pady=2)
        ttk.Button(manage_frame, text="Update Hey Classification", command=self.update_hey_classification_database).pack(fill=tk.X, pady=2)
        ttk.Button(manage_frame, text="Refresh Database Info", command=self.update_database_stats).pack(fill=tk.X, pady=2)

        # Database statistics frame
        stats_frame = ttk.LabelFrame(self.tab_database, text="Database Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create the db_stats_text widget - this must be initialized before update_database_stats is called
        self.db_stats_text = tk.Text(stats_frame, height=10, width=30, wrap=tk.WORD)
        self.db_stats_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.db_stats_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.db_stats_text.config(yscrollcommand=scrollbar.set)

        # Now it's safe to update the database statistics
        self.update_database_stats()

    
        

        
    def edit_metadata(self, name=None):
        """
        Edit metadata for the current spectrum or a specified database entry.
        
        Parameters:
        -----------
        name : str or None
            Name of the database entry to edit. If None, edit current spectrum metadata.
        """
        if name is None:
            # Edit current spectrum metadata
            if self.raman.current_spectra is None:
                messagebox.showerror("Error", "No spectrum loaded.")
                return
            metadata = self.raman.metadata
            title = "Edit Current Spectrum Metadata"
        else:
            # Edit database entry metadata
            if name not in self.raman.database:
                messagebox.showerror("Error", f"No spectrum named '{name}' in database.")
                return
            metadata = self.raman.database[name].get('metadata', {})
            title = f"Edit Metadata for {name}"
        
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("500x600")
        dialog.grab_set()  # Make modal
        
        # Create scrollable frame
        main_frame = ttk.Frame(dialog, padding=10)
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
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Keep track of entry widgets - store as instance variable for the dialog
        self.metadata_entries = {}
        
        # Add Hey Classification dropdown at the top
        hey_frame = ttk.LabelFrame(scrollable_frame, text="Hey Classification", padding=5)
        hey_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Get all available Hey Classifications
        hey_classes = []
        if hasattr(self.raman, 'hey_classification'):
            hey_classes = sorted(list(set(self.raman.hey_classification.values())))
        
        current_hey = metadata.get('HEY CLASSIFICATION', '')
        self.var_hey = tk.StringVar(value=current_hey)
        
        # Create combobox with Hey Classifications
        ttk.Label(hey_frame, text="Hey Classification:").pack(anchor=tk.W)
        hey_combo = ttk.Combobox(hey_frame, textvariable=self.var_hey, values=[""] + hey_classes)
        hey_combo.pack(fill=tk.X, pady=2)
        self.metadata_entries['HEY CLASSIFICATION'] = self.var_hey
        
        # Add mineral name field
        name_frame = ttk.LabelFrame(scrollable_frame, text="Mineral Name", padding=5)
        name_frame.pack(fill=tk.X, pady=5, padx=5)
        
        current_name = metadata.get('NAME', '')
        self.var_name = tk.StringVar(value=current_name)
        
        ttk.Label(name_frame, text="Mineral Name:").pack(anchor=tk.W)
        name_entry = ttk.Entry(name_frame, textvariable=self.var_name)
        name_entry.pack(fill=tk.X, pady=2)
        self.metadata_entries['NAME'] = self.var_name
        
        # Function to look up Hey Classification
        def lookup_hey_classification():
            mineral_name = self.var_name.get().strip()
            if not mineral_name:
                messagebox.showinfo("Lookup", "Please enter a mineral name first.")
                return
            
            # Try to find Hey Classification
            matched_name, classification = self.raman.match_mineral_name(mineral_name)
            
            if classification:
                self.var_hey.set(classification)
                messagebox.showinfo("Lookup Result", 
                                f"Found Hey Classification: {classification}\nMatched with: {matched_name}")
            else:
                messagebox.showinfo("Lookup Result", 
                                "No matching Hey Classification found for this mineral name.")
        
        ttk.Button(name_frame, text="Lookup Hey Classification", 
                   command=lookup_hey_classification).pack(pady=5)
        
        # Other metadata fields
        self.other_metadata_frame = ttk.LabelFrame(scrollable_frame, text="Other Metadata", padding=5)
        self.other_metadata_frame.pack(fill=tk.X, pady=5, padx=5, expand=True)
        
        # Priority fields to show at the top (excluding NAME and HEY CLASSIFICATION)
        priority_fields = ['RRUFFID', 'IDEAL CHEMISTRY', 'CHEMICAL FAMILY', 'LOCALITY', 'DESCRIPTION', 'URL']
        
        # Add priority fields first
        for field in priority_fields:
            if field in metadata:
                var = tk.StringVar(value=metadata[field])
                ttk.Label(self.other_metadata_frame, text=f"{field}:").pack(anchor=tk.W)
                ttk.Entry(self.other_metadata_frame, textvariable=var).pack(fill=tk.X, pady=2)
                self.metadata_entries[field] = var
        
        # Add remaining fields
        for field, value in metadata.items():
            if field not in self.metadata_entries and field not in ['NAME', 'HEY CLASSIFICATION']:
                var = tk.StringVar(value=value)
                ttk.Label(self.other_metadata_frame, text=f"{field}:").pack(anchor=tk.W)
                ttk.Entry(self.other_metadata_frame, textvariable=var).pack(fill=tk.X, pady=2)
                self.metadata_entries[field] = var
        
        # Add new field section
        new_field_frame = ttk.LabelFrame(scrollable_frame, text="Add New Field", padding=5)
        new_field_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.var_new_field = tk.StringVar()
        self.var_new_value = tk.StringVar()
        
        ttk.Label(new_field_frame, text="Field Name:").pack(anchor=tk.W)
        ttk.Entry(new_field_frame, textvariable=self.var_new_field).pack(fill=tk.X, pady=2)
        
        ttk.Label(new_field_frame, text="Field Value:").pack(anchor=tk.W)
        ttk.Entry(new_field_frame, textvariable=self.var_new_value).pack(fill=tk.X, pady=2)
        
        # Function to add new field to the form
        def add_new_field():
            field_name = self.var_new_field.get().strip().upper()
            field_value = self.var_new_value.get().strip()
            
            if not field_name or not field_value:
                messagebox.showinfo("Add Field", "Please enter both field name and value.")
                return
            
            if field_name in self.metadata_entries:
                if messagebox.askyesno("Field Exists", 
                                        f"Field '{field_name}' already exists. Update its value?"):
                    self.metadata_entries[field_name].set(field_value)
                    self.var_new_field.set("")
                    self.var_new_value.set("")
                return
            
            # Add new field to the form
            var = tk.StringVar(value=field_value)
            ttk.Label(self.other_metadata_frame, text=f"{field_name}:").pack(anchor=tk.W)
            ttk.Entry(self.other_metadata_frame, textvariable=var).pack(fill=tk.X, pady=2)
            self.metadata_entries[field_name] = var
            
            # Clear entry fields
            self.var_new_field.set("")
            self.var_new_value.set("")
        
        ttk.Button(new_field_frame, text="Add Field", command=add_new_field).pack(pady=5)
        
        # Button frame for save/cancel
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_metadata():
            # Update metadata with values from entry widgets
            updated_metadata = {}
            for field, var in self.metadata_entries.items():
                value = var.get().strip()
                if value:  # Only add non-empty fields
                    updated_metadata[field] = value
            
            if name is None:
                # Update current spectrum metadata
                self.raman.metadata = updated_metadata
                self.update_metadata_display()
                messagebox.showinfo("Metadata", "Current spectrum metadata updated.")
            else:
                # Update database entry metadata
                self.raman.database[name]['metadata'] = updated_metadata
                self.raman.save_database()
                messagebox.showinfo("Metadata", f"Metadata for '{name}' updated and saved to database.")
            
            # Update comboboxes for filters
            self.update_metadata_filter_options()
            dialog.destroy()
        
        ttk.Button(button_frame, text="Save Metadata", command=save_metadata).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
            
            
            


    # Standalone add_new_field method
    def add_new_field(self):
        """
        Add a new metadata field with the current values from the entry fields.
        This is a standalone method called from outside edit_metadata.
        """
        # Get values from the entry fields
        field_name = self.var_new_field.get().strip().upper()
        field_value = self.var_new_value.get().strip()
        
        if not field_name or not field_value:
            messagebox.showinfo("Add Field", "Please enter both field name and value.")
            return
        
        # Check if field already exists
        if field_name in self.metadata_entries:
            if messagebox.askyesno("Field Exists", 
                                f"Field '{field_name}' already exists. Update its value?"):
                self.metadata_entries[field_name].set(field_value)
                self.var_new_field.set("")
                self.var_new_value.set("")
            return
        
        # Add new field to the form
        self.metadata_entries[field_name] = tk.StringVar(value=field_value)
        ttk.Label(self.other_metadata_frame, text=f"{field_name}:").pack(anchor=tk.W)
        ttk.Entry(self.other_metadata_frame, textvariable=self.metadata_entries[field_name]).pack(fill=tk.X, pady=2)
        
        # Clear entry fields
        self.var_new_field.set("")
        self.var_new_value.set("")
        
        
    



    def create_basic_search_tab(self, parent):
        """Create basic search parameters interface."""
        # Search parameters frame
        params_frame = ttk.LabelFrame(parent, text="Search Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=5)

        ttk.Label(params_frame, text="Number of Matches:").pack(anchor=tk.W)
        self.var_n_matches = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.var_n_matches).pack(fill=tk.X, pady=2)

        ttk.Label(params_frame, text="Similarity Threshold:").pack(anchor=tk.W)
        self.var_corr_threshold = tk.StringVar(value="0.7")
        ttk.Entry(params_frame, textvariable=self.var_corr_threshold).pack(fill=tk.X, pady=2)

        # Matching algorithm selection
        ttk.Label(params_frame, text="Matching Algorithm:").pack(anchor=tk.W)
        self.var_algorithm = tk.StringVar(value="combined")
        algorithms = [
            ("Combined (Recommended)", "combined"),
            ("Correlation", "correlation"),
            ("Peak Matching", "peak"),
        ]
        if SKLEARN_AVAILABLE:
            algorithms.append(("Machine Learning (PCA)", "ml"))

        for text, value in algorithms:
            ttk.Radiobutton(
                params_frame,
                text=text,
                variable=self.var_algorithm,
                value=value
            ).pack(anchor=tk.W, pady=1)

        # Search button
        ttk.Button(params_frame, text="Search Match", command=self.search_match).pack(fill=tk.X, pady=5)

        # Display options frame (for controlling the main plot during comparison)
        visualization_frame = ttk.LabelFrame(parent, text="Visualization Options", padding=10)
        visualization_frame.pack(fill=tk.X, pady=5)

        self.var_show_diff = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            visualization_frame,
            text="Show Difference Plot",
            variable=self.var_show_diff
        ).pack(anchor=tk.W, pady=2)

        self.var_normalize = tk.BooleanVar(value=True) # Usually good to keep normalization for comparison
        ttk.Checkbutton(
            visualization_frame,
            text="Normalize Spectra (for Plot)",
            variable=self.var_normalize
        ).pack(anchor=tk.W, pady=2)

        self.var_highlight_peaks = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            visualization_frame,
            text="Highlight Matching Peaks",
            variable=self.var_highlight_peaks
        ).pack(anchor=tk.W, pady=2)




    

    def create_advanced_search_tab(self, parent):
        """Create advanced search options interface."""
        # Peak-based search frame
        peak_frame = ttk.LabelFrame(parent, text="Peak-Based Search Filter", padding=10)
        peak_frame.pack(fill=tk.X, pady=5)

        ttk.Label(peak_frame, text="Search by specific peak positions:").pack(anchor=tk.W)
        self.var_peak_positions = tk.StringVar()
        ttk.Entry(peak_frame, textvariable=self.var_peak_positions).pack(fill=tk.X, pady=2)
        
        # Fix for the font parameter issue - Create the label first, then pack it separately
        hint_label = ttk.Label(
            peak_frame,
            text="Format: comma-separated values (e.g., 1050, 1350)",
            font=('TkDefaultFont', 8) # Apply font during creation
        )
        hint_label.pack(anchor=tk.W) # Pack separately

        ttk.Label(peak_frame, text="Peak Tolerance (cm⁻¹):").pack(anchor=tk.W)
        self.var_peak_tolerance = tk.StringVar(value="10")
        ttk.Entry(peak_frame, textvariable=self.var_peak_tolerance).pack(fill=tk.X, pady=2)

        # Category filtering frame
        filter_frame = ttk.LabelFrame(parent, text="Metadata Filter Options", padding=10)
        filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_frame, text="Filter by Chemical Family:").pack(anchor=tk.W)
        self.var_chemical_family = tk.StringVar()
        self.chemical_family_combo = ttk.Combobox(filter_frame, textvariable=self.var_chemical_family, state="readonly")
        self.chemical_family_combo.pack(fill=tk.X, pady=2)
        
        # Add Hey Classification filter
        ttk.Label(filter_frame, text="Filter by Hey Classification:").pack(anchor=tk.W)
        self.var_hey_classification = tk.StringVar()
        self.hey_classification_combo = ttk.Combobox(filter_frame, textvariable=self.var_hey_classification, state="readonly")
        self.hey_classification_combo.pack(fill=tk.X, pady=2)
        
        # Update both dropdown options
        self.update_metadata_filter_options()

        # Advanced search button
        ttk.Button(parent, text="Advanced Search", command=self.advanced_search_match).pack(fill=tk.X, pady=5)
        
        
    
    def update_metadata_filter_options(self):
        """Update chemical family and Hey Classification combobox options from database metadata."""
        # Chemical Family options
        families = set()
        # Hey Classification options
        hey_classes = set()
        
        if hasattr(self, 'raman') and hasattr(self.raman, 'database') and self.raman.database:
            for data in self.raman.database.values():
                if 'metadata' in data and data.get('metadata'): # Check if metadata exists and is not empty
                    # Chemical Family
                    family = data['metadata'].get('CHEMICAL FAMILY')
                    if family: # Check if family exists and is not empty/None
                        families.add(family)
                    
                    # Hey Classification
                    hey_class = data['metadata'].get('HEY CLASSIFICATION')
                    if hey_class: # Check if Hey Classification exists and is not empty/None
                        hey_classes.add(hey_class)

        # Update Chemical Family combobox
        sorted_families = sorted(list(families))
        self.chemical_family_combo['values'] = [""] + sorted_families # Add empty option to disable filter
        self.var_chemical_family.set("") # Default to no filter
        
        # Update Hey Classification combobox
        sorted_hey_classes = sorted(list(hey_classes))
        self.hey_classification_combo['values'] = [""] + sorted_hey_classes # Add empty option to disable filter
        self.var_hey_classification.set("") # Default to no filter
    
    
    
    def update_chemical_family_options(self):
        """Update the chemical family dropdown options from database metadata."""
        # Check if the chemical_family_combo attribute exists
        if not hasattr(self, 'chemical_family_combo'):
            return
            
        # Get unique chemical families from database
        families = set()
        if hasattr(self, 'raman') and hasattr(self.raman, 'database'):
            for data in self.raman.database.values():
                if 'metadata' in data and data['metadata']:
                    family = data['metadata'].get('CHEMICAL FAMILY')
                    if family:  # Only add non-empty values
                        families.add(family)
        
        # Sort families for display
        sorted_families = sorted(list(families))
        
        # Update the combobox values
        self.chemical_family_combo['values'] = [""] + sorted_families
        
        # Reset the current selection to empty (no filter)
        if hasattr(self, 'var_chemical_family'):
            self.var_chemical_family.set("")
            
            


    def update_database_stats(self):
        """Update the database statistics display."""
        # Check if db_stats_text exists before trying to use it
        if not hasattr(self, 'db_stats_text'):
            # This method might have been called before UI components were created
            # Just return without updating stats
            return
        
        self.db_stats_text.config(state=tk.NORMAL)
        self.db_stats_text.delete(1.0, tk.END)

        if hasattr(self, 'raman') and hasattr(self.raman, 'database'):
            db_size = len(self.raman.database)
            self.db_stats_text.insert(tk.END, f"Total entries: {db_size}\n\n")

            if db_size > 0:
                # Count entries with metadata
                with_metadata = sum(1 for data in self.raman.database.values()
                                    if 'metadata' in data and data.get('metadata'))
                self.db_stats_text.insert(tk.END, f"Entries with metadata: {with_metadata}\n")

                # Count entries with peaks
                with_peaks = sum(1 for data in self.raman.database.values()
                                if 'peaks' in data and data.get('peaks') is not None)
                self.db_stats_text.insert(tk.END, f"Entries with peak data: {with_peaks}\n\n")

                # List a few example entries
                self.db_stats_text.insert(tk.END, "Sample entries:\n")
                for i, name in enumerate(list(self.raman.database.keys())[:5]):
                    self.db_stats_text.insert(tk.END, f"- {name}\n")

                if db_size > 5:
                    self.db_stats_text.insert(tk.END, f"... and {db_size-5} more\n")
            else:
                self.db_stats_text.insert(tk.END, "Database is empty.\n")
        else:
            self.db_stats_text.insert(tk.END, "Database not loaded or object missing.\n")

        self.db_stats_text.config(state=tk.DISABLED)
        
        # Update chemical family options if the method exists
        if hasattr(self, 'update_chemical_family_options'):
            self.update_chemical_family_options()


    def import_spectrum(self):
        """
        Import a Raman spectrum from a file.
        Opens a file dialog to select the file and loads it using the RamanSpectra class.
        """
        try:
            # Open file dialog to select spectrum file
            file_path = filedialog.askopenfilename(
                title="Select Raman Spectrum File",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:  # User cancelled the dialog
                return
                
            # Import the spectrum using the RamanSpectra class
            self.raman.import_spectrum(file_path)
            
            # Update the plot and metadata display
            self.update_plot()
            self.update_metadata_display()
            
            # Show success message
            messagebox.showinfo("Success", "Spectrum imported successfully!")
            
        except Exception as e:
            # Show error message if import fails
            messagebox.showerror("Error", f"Failed to import spectrum: {str(e)}")

    def update_metadata_display(self, metadata_dict=None):
        """Update the metadata display with provided or current spectrum metadata."""
        self.metadata_text.config(state=tk.NORMAL)
        self.metadata_text.delete(1.0, tk.END)

        meta_to_display = metadata_dict if metadata_dict is not None else self.raman.metadata

        if meta_to_display:
            # Order of important metadata fields
            important_fields = ['NAME', 'RRUFFID', 'IDEAL CHEMISTRY', 'CHEMICAL FAMILY', 'LOCALITY', 'DESCRIPTION', 'URL']

            # Display important fields first in order
            displayed_keys = set()
            for field in important_fields:
                if field in meta_to_display:
                    self.metadata_text.insert(tk.END, f"{field}: {meta_to_display[field]}\n")
                    displayed_keys.add(field)

            # Display other fields
            for field, value in meta_to_display.items():
                if field not in displayed_keys:
                    self.metadata_text.insert(tk.END, f"{field}: {value}\n")
        else:
            # If specific metadata wasn't passed, check current Raman object
            if metadata_dict is None and hasattr(self.raman, 'metadata') and self.raman.metadata:
                self.metadata_text.insert(tk.END, "No metadata available for current spectrum.")
            elif metadata_dict is None:
                self.metadata_text.insert(tk.END, "No spectrum loaded.")
            else:
                self.metadata_text.insert(tk.END, "No metadata available.")


        self.metadata_text.config(state=tk.DISABLED)

    def save_spectrum(self):
        """Save the current spectrum to a file."""
        try:
            if self.raman.current_spectra is None:
                messagebox.showerror("Error", "No spectrum loaded to save.")
                return
            
            # Get save location from user
            file_path = filedialog.asksaveasfilename(
                title="Save Raman Spectrum",
                defaultextension=".txt",
                filetypes=[
                    ("Text Files", "*.txt"),
                    ("CSV Files", "*.csv"),
                    ("All Files", "*.*")
                ]
            )
            
            if not file_path:
                return  # User cancelled
            
            # Try to save the spectrum
            success = self.raman.save_spectrum(file_path)
            
            if not success:
                messagebox.showerror("Error", "Failed to save spectrum.")
                return
            
            messagebox.showinfo("Success", "Spectrum saved successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save spectrum: {str(e)}")

    def subtract_background(self):
        """Subtract background from the current spectrum."""
        if self.raman.current_spectra is None:
            messagebox.showerror("Error", "No spectrum loaded.")
            return

        try:
            # Get parameters
            try:
                lam = float(self.var_lambda.get())
                p = float(self.var_p.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid baseline parameters. Please enter numeric values.")
                return

            # Subtract background
            corrected, baseline = self.raman.subtract_background(lam=lam, p=p)

            # Update plot
            self.update_plot(include_background=True, include_peaks=self.var_show_peaks.get()) # Keep peak display status

            messagebox.showinfo("Success", "Background subtracted successfully.")

        except Exception as e:
            messagebox.showerror("Background Subtraction Error", str(e))

    def find_peaks(self):
        """Find peaks in the current spectrum."""
        if self.raman.current_spectra is None:
            messagebox.showerror("Error", "No spectrum loaded.")
            return

        try:
            # Get parameters, handling "Auto"
            height_str = self.var_height.get().strip()
            distance_str = self.var_distance.get().strip()
            prominence_str = self.var_prominence.get().strip()

            # Determine spectrum for peak finding
            spectrum_to_use = self.raman.processed_spectra if self.raman.processed_spectra is not None else self.raman.current_spectra
            max_intensity = np.max(spectrum_to_use)

            try:
                # Convert height to appropriate value
                if height_str == "Auto":
                    height = None  # Let RamanSpectra class use default 5%
                elif height_str.endswith('%'):
                    # Handle percentage input
                    percentage = float(height_str.rstrip('%')) / 100.0
                    height = percentage * max_intensity
                else:
                    # Handle absolute value input
                    height_value = float(height_str)
                    # If value is less than 1.0, treat as percentage of max intensity
                    if height_value < 1.0 and height_value > 0:
                        height = height_value * max_intensity
                    else:
                        height = height_value

                # Handle other parameters similarly
                distance = None if distance_str == "Auto" else int(distance_str)
                prominence = None if prominence_str == "Auto" else float(prominence_str)
            except ValueError:
                messagebox.showerror("Error", "Invalid peak parameters. Please enter numeric values or 'Auto'.")
                return

            # Clear previous peaks
            self.raman.peaks = None
            
            # Find peaks using the RamanSpectra method
            peaks = self.raman.find_spectrum_peaks(
                height=height,
                distance=distance,
                prominence=prominence
            )

            # Update plot with new peaks
            self.var_show_peaks.set(True)  # Turn on peak display
            self.update_plot(include_background=self.var_show_background.get(), include_peaks=True)

            messagebox.showinfo("Success", f"Found {len(peaks['indices'])} peaks.")

        except Exception as e:
            messagebox.showerror("Peak Finding Error", str(e))

    def add_to_database(self):
        """Add the current spectrum to the database."""
        if self.raman.current_spectra is None:
            messagebox.showerror("Error", "No spectrum loaded.")
            return

        name = self.var_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please provide a name for the spectrum.")
            return

        try: # Wrap the operation in try/except
            # Check if name already exists in database
            if name in self.raman.database:
                overwrite = messagebox.askyesno(
                    "Confirmation",
                    f"'{name}' already exists in the database. Overwrite?"
                )
                if not overwrite:
                    return

            # Add to database (metadata is handled within RamanSpectra class)
            success = self.raman.add_to_database(name) # metadata is already in self.raman.metadata

            if success:
                messagebox.showinfo("Success", f"'{name}' added/updated in database.")
                # Update database statistics if tab exists
                if hasattr(self, 'db_stats_text'):
                    self.update_database_stats()
            else:
                messagebox.showerror("Error", f"Failed to add '{name}' to database.")

        except Exception as e:
            messagebox.showerror("Database Error", f"Error adding spectrum to database: {str(e)}")

    
    def batch_import(self):
        """Batch import spectra to the database with threaded processing."""
        import threading
        import queue
        
        directory = filedialog.askdirectory(title="Select Directory with Spectrum Files")
        if not directory:
            return

        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Import Progress")
        progress_window.geometry("500x400")  # Slightly larger for better log visibility

        # Create progress bar
        ttk.Label(progress_window, text="Importing spectra...").pack(pady=10)
        progress = ttk.Progressbar(progress_window, length=400, mode="determinate")
        progress.pack(pady=10)

        # Create status label
        status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(progress_window, textvariable=status_var)
        status_label.pack(pady=5)

        # Create log text area
        log_frame = ttk.Frame(progress_window)
        log_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        log_text = tk.Text(log_frame, height=15, width=60, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
        log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure tags for log text
        log_text.tag_configure("success", foreground="green")
        log_text.tag_configure("warning", foreground="orange")
        log_text.tag_configure("error", foreground="red")
        log_text.tag_configure("info", foreground="blue")

        # Get files list
        try:
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            if not files:
                messagebox.showinfo("Batch Import", "No files found in the selected directory.")
                progress_window.destroy()
                return
        except Exception as e:
            messagebox.showerror("Batch Import Error", f"Error reading directory: {str(e)}")
            progress_window.destroy()
            return

        # Update progress bar maximum
        progress["maximum"] = len(files)
        
        # Create a queue for thread-safe communication between worker thread and UI
        update_queue = queue.Queue()
        
        # Flag to indicate if processing should stop (for cancellation)
        stop_processing = threading.Event()
        
        # Function to process files in a separate thread
        def process_files_thread():
            # Statistics counters
            imported = 0
            skipped = 0
            errors = 0
            
            # Use a temporary RamanSpectra instance for processing
            temp_raman = RamanSpectra()
            
            for i, filename in enumerate(files):
                # Check if processing should stop
                if stop_processing.is_set():
                    break
                    
                filepath = os.path.join(directory, filename)
                current_status_msg = f"Processing {filename}"
                
                # Put progress update in queue
                update_queue.put(('progress', i, len(files), filename, current_status_msg))
                
                try:
                    # Import spectrum using the temporary instance
                    _, _ = temp_raman.import_spectrum(filepath)
                    
                    # Find peaks before adding
                    temp_raman.find_spectrum_peaks()
                    
                    # Generate name from metadata or filename
                    name = ""
                    if 'RRUFFID' in temp_raman.metadata and temp_raman.metadata['RRUFFID']:
                        name = temp_raman.metadata['RRUFFID']
                    elif 'NAME' in temp_raman.metadata and temp_raman.metadata['NAME']:
                        name = temp_raman.metadata['NAME']
                    
                    if not name:  # Fallback to filename if no suitable metadata
                        name, _ = os.path.splitext(filename)
                    name = name.strip()
                    
                    # Check if name already exists in the main database
                    if name in self.raman.database:
                        log_message = f"Skipped: '{name}' (from {filename}) already exists in database."
                        update_queue.put(('log', log_message, "info"))
                        skipped += 1
                        continue
                    
                    # Ensure Hey Classification is added to metadata if possible
                    if 'NAME' in temp_raman.metadata and 'HEY CLASSIFICATION' not in temp_raman.metadata:
                        mineral_name = temp_raman.metadata['NAME']
                        hey_class = temp_raman.get_hey_classification(mineral_name)
                        if hey_class:
                            temp_raman.metadata['HEY CLASSIFICATION'] = hey_class
                            update_queue.put(('log', f"Added Hey Classification '{hey_class}' for {mineral_name}", "success"))
                    
                    # Safely copy necessary data to main database
                    # We'll store the data temporarily and let the main thread do the actual database update
                    spectrum_data = {
                        'name': name,
                        'wavenumbers': temp_raman.current_wavenumbers.copy(),
                        'intensities': temp_raman.current_spectra.copy(),
                        'processed': temp_raman.processed_spectra.copy() if temp_raman.processed_spectra is not None else None,
                        'peaks': temp_raman.peaks.copy() if temp_raman.peaks is not None else None,
                        'metadata': temp_raman.metadata.copy()
                    }
                    
                    update_queue.put(('add_to_db', spectrum_data))
                    
                    # Log success
                    log_message = f"Imported '{name}' (from {filename})."
                    update_queue.put(('log', log_message, "success"))
                    imported += 1
                    
                except Exception as e:
                    errors += 1
                    log_message = f"Error importing/processing {filename}: {str(e)}"
                    update_queue.put(('log', log_message, "error"))
            
            # Final status update
            update_queue.put(('complete', imported, skipped, errors))
        
        # Function to check queue and update UI
        def check_queue():
            try:
                # Process all available updates in the queue
                while True:
                    item = update_queue.get_nowait()
                    message_type = item[0]
                    
                    if message_type == 'progress':
                        _, current, total, filename, status_msg = item
                        progress["value"] = current
                        percentage = int((current / total) * 100) if total > 0 else 0
                        status_var.set(f"{status_msg} - {percentage}% complete ({current}/{total})")
                    
                    elif message_type == 'log':
                        _, message, tag = item
                        log_text.insert(tk.END, message + "\n", tag)
                        log_text.see(tk.END)  # Auto-scroll to the end
                    
                    elif message_type == 'add_to_db':
                        _, spectrum_data = item
                        # Update the main database in the UI thread to avoid threading issues
                        name = spectrum_data['name']
                        
                        # Copy data to the main Raman object
                        self.raman.current_wavenumbers = spectrum_data['wavenumbers']
                        self.raman.current_spectra = spectrum_data['intensities']
                        self.raman.processed_spectra = spectrum_data['processed']
                        self.raman.peaks = spectrum_data['peaks']
                        self.raman.metadata = spectrum_data['metadata']
                        
                        # Add to database
                        self.raman.add_to_database(name)
                    
                    elif message_type == 'complete':
                        _, imported, skipped, errors = item
                        # Final status update
                        final_status = f"Complete! Imported: {imported}, Skipped: {skipped}, Errors: {errors}"
                        status_var.set(final_status)
                        log_text.insert(tk.END, f"\n--- {final_status} ---\n", "info")
                        
                        # Save the database
                        save_result = self.raman.save_database()
                        if save_result:
                            log_text.insert(tk.END, f"Database saved successfully to {self.raman.db_path}\n", "success")
                        else:
                            log_text.insert(tk.END, "Warning: Could not save database\n", "error")
                        
                        log_text.see(tk.END)
                        
                        # Update main GUI database statistics
                        self.update_database_stats()
                        
                        # Add close button
                        ttk.Button(progress_window, text="Close", 
                                  command=progress_window.destroy).pack(pady=10)
                        
                        # No need to keep checking the queue
                        return
                    
                    # Mark this item as processed
                    update_queue.task_done()
            
            except queue.Empty:
                # Queue is empty, check again after a delay
                progress_window.after(100, check_queue)
        
        # Add a cancel button
        def cancel_processing():
            stop_processing.set()
            status_var.set("Cancelling... Please wait.")
            cancel_button.config(state=tk.DISABLED)
        
        cancel_button = ttk.Button(progress_window, text="Cancel", command=cancel_processing)
        cancel_button.pack(pady=5)
        
        # Start the worker thread
        worker_thread = threading.Thread(target=process_files_thread)
        worker_thread.daemon = True  # Thread will terminate when main program exits
        worker_thread.start()
        
        # Start checking for updates
        progress_window.after(100, check_queue)
        
        # Make progress window modal
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Override window close button to set stop flag
        progress_window.protocol("WM_DELETE_WINDOW", cancel_processing)




    def view_database(self):
        """View the database contents in a new window."""
        if not self.raman.database:
            messagebox.showinfo("Database", "The database is empty.")
            return

        # Create a new window
        db_window = tk.Toplevel(self.root)
        db_window.title("Raman Spectra Database")
        db_window.geometry("800x600")

        # --- GUI Elements ---
        main_frame = ttk.Frame(db_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Search Frame
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=5)
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
        search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # List Frame (using Treeview for better structure)
        list_frame = ttk.LabelFrame(main_frame, text="Database Entries", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Treeview Columns
        columns = ('#0', 'name', 'points', 'family', 'hey_class')  # Added hey_class column
        tree = ttk.Treeview(list_frame, columns=columns[1:], show='headings', height=15)

        # Define headings
        tree.heading('name', text='Name/ID')
        tree.heading('points', text='Data Points')
        tree.heading('family', text='Chemical Family')
        tree.heading('hey_class', text='Hey Classification')  


        # Configure column widths (adjust as needed)
        tree.column('name', width=250, anchor=tk.W)
        tree.column('points', width=100, anchor=tk.CENTER)
        tree.column('family', width=200, anchor=tk.W)
        tree.column('hey_class', width=180, anchor=tk.W) 

        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Populate and Filter Logic ---
        db_items = list(self.raman.database.items()) # Get a list view

        def populate_treeview(search_term=""):
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
        
            search_lower = search_term.lower() if search_term else ""
        
            # Filter and insert entries
            for name, data in db_items:
                family = data.get('metadata', {}).get('CHEMICAL FAMILY', 'N/A')
                hey_class = data.get('metadata', {}).get('HEY CLASSIFICATION', 'N/A')  # Get Hey Classification
                points = len(data.get('wavenumbers', []))
                display_name = name # Use the key as the display name
        
                # Check if search term matches name, family, or Hey Classification
                matches_search = True # Assume match if no search term
                if search_lower:
                    matches_search = (search_lower in display_name.lower() or
                                   search_lower in family.lower() or
                                   search_lower in hey_class.lower())  # Include Hey Classification in search
        
                if matches_search:
                    # Use 'name' (the actual dict key) as the item ID (iid) for later retrieval
                    tree.insert('', tk.END, iid=name, values=(display_name, points, family, hey_class))  # Added hey_class

        # Search function
        def perform_search(event=None): # Added event for binding
            populate_treeview(search_var.get())

        search_entry.bind('<Return>', perform_search) # Bind Enter key
        ttk.Button(search_frame, text="Search", command=perform_search).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Clear", command=lambda: [search_var.set(""), populate_treeview()]).pack(side=tk.LEFT, padx=5)


        # --- Button Actions ---
        button_frame = ttk.Frame(main_frame, padding=(0, 5, 0, 0))
        button_frame.pack(fill=tk.X)

        def get_selected_name():
             selected_items = tree.selection()
             if not selected_items:
                 messagebox.showwarning("Selection Required", "Please select a spectrum from the list.")
                 return None
             # Use the item ID (iid), which we set as the dictionary key 'name'
             return selected_items[0]

        def view_selected():
            name = get_selected_name()
            if name:
                self.view_database_item(name) # Pass the actual key

        def remove_selected():
            name = get_selected_name()
            if name:
                self.remove_database_item(name, lambda: populate_treeview(search_var.get())) # Pass callback
                
        def edit_selected():
            name = get_selected_name()
            if name:
                # Call the metadata editor with the selected spectrum name
                self.edit_metadata(name)
                # Update treeview after editing
                populate_treeview(search_var.get())


        ttk.Button(button_frame, text="View in Main Plot", command=view_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Edit Metadata", command=edit_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=db_window.destroy).pack(side=tk.RIGHT, padx=5)


        # --- Initial Population ---
        populate_treeview()

        # Make window modal
        db_window.transient(self.root)
        db_window.grab_set()
        self.root.wait_window(db_window)

    def view_database_item(self, name):
        """View a database item in the main plot and update metadata display."""
        if name in self.raman.database:
            data = self.raman.database[name]

            # Update metadata display in the main GUI
            self.update_metadata_display(data.get('metadata', {}))

            # Update the main plot
            self.ax.clear()
            self.ax.plot(data['wavenumbers'], data['intensities'], label=f"DB: {name}")

            # Mark peaks if available
            if 'peaks' in data and data.get('peaks') is not None:
                try: # Add error handling for potentially malformed peak data
                    peak_wavenumbers = data['peaks'].get('wavenumbers')
                    peak_heights = data['peaks'].get('heights')
                    if peak_wavenumbers is not None and peak_heights is not None:
                         # Ensure heights correspond to the *plotted* intensities
                         # Need to find the intensity values at peak wavenumbers
                         peak_plot_heights = np.interp(peak_wavenumbers, data['wavenumbers'], data['intensities'])
                         self.ax.plot(peak_wavenumbers, peak_plot_heights, 'ro', label='DB Peaks')
                except Exception as e:
                    print(f"Warning: Could not plot peaks for {name}: {e}")


            self.ax.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax.set_ylabel('Intensity (a.u.)')
            self.ax.set_title(f'Database Spectrum: {name}')
            self.ax.legend()
            self.canvas.draw()
        else:
             messagebox.showerror("Error", f"Spectrum '{name}' not found in database.")

    def remove_database_item(self, name, refresh_callback=None):
        """Remove an item from the database after confirmation."""
        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to permanently remove '{name}' from the database?"
        )

        if confirm:
            try:
                success = self.raman.remove_from_database(name)
                if success:
                    messagebox.showinfo("Database", f"'{name}' removed from database.")
                    # Update stats in main GUI
                    self.update_database_stats()
                    # Refresh the database view window if a callback is provided
                    if refresh_callback:
                        refresh_callback()
                else:
                    messagebox.showerror("Error", f"Could not find '{name}' in the database to remove.")
            except Exception as e:
                 messagebox.showerror("Error", f"Error removing '{name}' from database: {str(e)}")

    # --- Search and Matching Logic ---

    def search_match(self):
        """Perform search using the basic parameters and selected algorithm."""
        if self._validate_search_conditions():
             try:
                 n_matches = int(self.var_n_matches.get())
                 correlation_threshold = float(self.var_corr_threshold.get())
                 algorithm = self.var_algorithm.get()

                 matches = self._perform_search(algorithm, n_matches, correlation_threshold)
                 self._process_and_display_matches(matches)

             except ValueError:
                 messagebox.showerror("Input Error", "Number of matches must be an integer and threshold must be a number.")
             except Exception as e:
                 messagebox.showerror("Search Error", f"An error occurred during search: {str(e)}")



    def advanced_search_match(self):
        """Perform search using advanced filters (peaks, chemical family, Hey Classification)."""
        if self._validate_search_conditions():
            try:
                # Basic params (still used for limiting results)
                n_matches = int(self.var_n_matches.get())
                threshold = float(self.var_corr_threshold.get())
    
                # Advanced Filters
                peak_positions = []
                peak_str = self.var_peak_positions.get().strip()
                if peak_str:
                    try:
                        peak_positions = [float(x.strip()) for x in peak_str.split(',')]
                    except ValueError:
                        messagebox.showerror("Input Error", "Invalid peak positions format. Use comma-separated numbers.")
                        return
    
                peak_tolerance = 10 # Default
                tolerance_str = self.var_peak_tolerance.get().strip()
                if tolerance_str:
                    try:
                        peak_tolerance = float(tolerance_str)
                    except ValueError:
                        messagebox.showwarning("Input Error", "Invalid peak tolerance. Using default (10 cm⁻¹).")
    
                chemical_family = self.var_chemical_family.get().strip() or None # None if empty
                hey_classification = self.var_hey_classification.get().strip() or None # None if empty
    
                # Perform filtered search
                matches = self._filtered_search(
                    peak_positions=peak_positions,
                    peak_tolerance=peak_tolerance,
                    chemical_family=chemical_family,
                    threshold=threshold,
                    hey_classification=hey_classification # Pass Hey Classification filter
                )
    
                # Sort and limit results
                matches.sort(key=lambda x: x[1], reverse=True)
                self._process_and_display_matches(matches[:n_matches])
    
            except ValueError:
                 messagebox.showerror("Input Error", "Number of matches or threshold is invalid.")
            except Exception as e:
                 messagebox.showerror("Advanced Search Error", f"An error occurred: {str(e)}")




    def _validate_search_conditions(self):
        """Check if a spectrum is loaded and the database is not empty."""
        if self.raman.current_spectra is None:
            messagebox.showerror("Error", "No spectrum loaded. Please import a spectrum first.")
            return False
        if not self.raman.database:
            messagebox.showerror("Error", "Database is empty. Please add spectra to the database.")
            return False
        return True

    def _perform_search(self, algorithm, n_matches, threshold):
        """Helper function to call the correct search method."""
        if algorithm == "correlation":
            return self.correlation_search(n_matches, threshold)
        elif algorithm == "peak":
            return self.peak_based_search(n_matches, threshold)
        elif algorithm == "ml" and SKLEARN_AVAILABLE:
            return self.ml_based_search(n_matches, threshold)
        elif algorithm == "ml": # Scikit-learn not available
            messagebox.showwarning("Missing Library","scikit-learn not found. Falling back to 'Combined' search.")
            return self.raman.search_match(n_matches, threshold) # Fallback to combined
        else: # Default to combined
            return self.raman.search_match(n_matches, threshold)

    def _process_and_display_matches(self, matches):
         """Update the results summary and display detailed results window."""
         self.match_result_text.config(state=tk.NORMAL)
         self.match_result_text.delete(1.0, tk.END)

         if matches:
             self.match_result_text.insert(tk.END, "=== Top 5 Search Results ===\n\n")
             for i, (name, score) in enumerate(matches[:5]):
                 confidence = self.get_confidence_level(score)
                 self.match_result_text.insert(tk.END, f"{i+1}. {name}\n")
                 self.match_result_text.insert(tk.END, f"   Score: {score:.4f} ({confidence})\n\n")

             # Display comprehensive results in a new window
             self.display_search_results(matches) # Pass all matches
         else:
             self.match_result_text.insert(tk.END, "No matches found matching the criteria.")

         self.match_result_text.config(state=tk.DISABLED)

    def correlation_search(self, n_matches, threshold):
        """Perform correlation-based search."""
        if self.raman.processed_spectra is not None:
            query_spectrum = self.raman.processed_spectra
        elif self.raman.current_spectra is not None:
             query_spectrum = self.raman.current_spectra
        else:
             raise ValueError("No spectrum data available for search.")

        query_wavenumbers = self.raman.current_wavenumbers
        matches = []

        # Normalize the query spectrum (0-1 range is better for correlation)
        query_min = np.min(query_spectrum)
        query_max = np.max(query_spectrum)
        query_norm = (query_spectrum - query_min) / (query_max - query_min) if (query_max > query_min) else query_spectrum

        for name, data in self.raman.database.items():
            db_intensities = data['intensities']
            db_wavenumbers = data['wavenumbers']

            # Interpolate database spectrum if wavenumbers don't match
            if not np.array_equal(query_wavenumbers, db_wavenumbers):
                db_intensities_interp = np.interp(query_wavenumbers, db_wavenumbers, db_intensities)
            else:
                db_intensities_interp = db_intensities

            # Normalize database spectrum
            db_min = np.min(db_intensities_interp)
            db_max = np.max(db_intensities_interp)
            db_norm = (db_intensities_interp - db_min) / (db_max - db_min) if (db_max > db_min) else db_intensities_interp

            # Calculate correlation coefficient
            try:
                # Ensure vectors are not constant
                if np.all(query_norm == query_norm[0]) or np.all(db_norm == db_norm[0]):
                     corr_coef = 0 # Correlation is undefined for constant vectors
                else:
                     corr_coef = np.corrcoef(query_norm, db_norm)[0, 1]
                if np.isnan(corr_coef): # Handle potential NaN from std dev calculation
                    corr_coef = 0
            except Exception: # Catch any other potential errors during calculation
                 corr_coef = 0


            if corr_coef >= threshold:
                matches.append((name, corr_coef))

        # Sort matches by correlation score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:n_matches]

    def peak_based_search(self, n_matches, threshold):
        """Perform peak-based search using Jaccard similarity."""
        if self.raman.peaks is None:
            # Find peaks if not already done for the current spectrum
            try:
                 self.find_peaks() # Use existing parameters or defaults
                 if self.raman.peaks is None: # Check if find_peaks failed
                     raise ValueError("Could not detect peaks in the query spectrum.")
            except Exception as e:
                 messagebox.showerror("Peak Search Error", f"Failed to find peaks for query spectrum: {e}")
                 return [] # Return empty list if peaks cannot be found


        if self.raman.peaks is None or not self.raman.peaks.get('wavenumbers').size:
             messagebox.showwarning("Peak Search", "No peaks found in the query spectrum to perform peak-based search.")
             return []

        query_peaks = self.raman.peaks['wavenumbers']
        matches = []
        tolerance = 10 # Default tolerance for peak matching (cm^-1)
        # Optionally get tolerance from GUI if available:
        if hasattr(self, 'var_peak_tolerance'):
            try:
                 tolerance = float(self.var_peak_tolerance.get())
            except ValueError:
                 pass # Use default if GUI value is invalid


        for name, data in self.raman.database.items():
            db_peak_data = data.get('peaks')
            if not db_peak_data or db_peak_data.get('wavenumbers') is None:
                continue # Skip database entries without pre-calculated peaks

            db_peaks_wavenumbers = db_peak_data['wavenumbers']
            if not db_peaks_wavenumbers.size:
                 continue # Skip if database entry has no peaks

            # Calculate Jaccard similarity (intersection over union)
            query_set = set(np.round(query_peaks / tolerance)) # Group peaks by tolerance
            db_set = set(np.round(db_peaks_wavenumbers / tolerance))

            intersection = len(query_set.intersection(db_set))
            union = len(query_set.union(db_set))

            jaccard = intersection / union if union > 0 else 0

            if jaccard >= threshold:
                matches.append((name, jaccard))

        # Sort matches by Jaccard score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:n_matches]

    def ml_based_search(self, n_matches, threshold):
        """Perform machine learning based search using PCA and cosine similarity."""
        if not SKLEARN_AVAILABLE:
             raise ImportError("scikit-learn library is required for ML-based search.")

        if self.raman.processed_spectra is not None:
            query_spectrum = self.raman.processed_spectra
        elif self.raman.current_spectra is not None:
             query_spectrum = self.raman.current_spectra
        else:
             raise ValueError("No spectrum data available for search.")

        query_wavenumbers = self.raman.current_wavenumbers

        # Prepare database spectra
        db_spectra_interp = []
        db_names = []
        min_len = len(query_wavenumbers) # Use query length as reference

        for name, data in self.raman.database.items():
            db_intensities = data['intensities']
            db_wavenumbers = data['wavenumbers']

            # Interpolate to match query wavenumbers
            if not np.array_equal(query_wavenumbers, db_wavenumbers):
                db_interp = np.interp(query_wavenumbers, db_wavenumbers, db_intensities)
            else:
                db_interp = db_intensities

            # Normalize (Max intensity to 1) - common preprocessing for PCA/similarity
            db_max = np.max(db_interp)
            if db_max > 0:
                db_norm = db_interp / db_max
            else:
                db_norm = db_interp # Avoid division by zero

            db_spectra_interp.append(db_norm)
            db_names.append(name)

        if not db_spectra_interp:
            return []

        # Normalize query spectrum similarly
        query_max = np.max(query_spectrum)
        query_norm = query_spectrum / query_max if query_max > 0 else query_spectrum

        # Combine query and database spectra for PCA fitting
        all_spectra = np.vstack(db_spectra_interp + [query_norm])

        # Data standardization (important for PCA)
        scaler = StandardScaler()
        spectra_scaled = scaler.fit_transform(all_spectra)

        # Apply PCA
        # Choose number of components (e.g., explain 95% variance or fixed number)
        n_components = min(10, spectra_scaled.shape[0], spectra_scaled.shape[1]) # Limit components
        if n_components < 1: # Handle cases with very few spectra/data points
             messagebox.showerror("ML Search Error", "Not enough data points or spectra for PCA.")
             return []

        pca = PCA(n_components=n_components)
        try:
            spectra_pca = pca.fit_transform(spectra_scaled)
        except ValueError as e:
             messagebox.showerror("PCA Error", f"Error during PCA calculation: {e}")
             return []


        # Separate query and database PCA components
        query_pca = spectra_pca[-1].reshape(1, -1)
        db_pca = spectra_pca[:-1]

        # Calculate cosine similarity in PCA space
        similarities = cosine_similarity(query_pca, db_pca)[0]

        # Filter by threshold and create matches list
        matches = []
        for i, similarity in enumerate(similarities):
            # Cosine similarity ranges from -1 to 1. Map to 0-1 if needed, or adjust threshold.
            # Assuming threshold is for 0-1 range, let's scale similarity: (sim + 1) / 2
            scaled_similarity = (similarity + 1) / 2
            if scaled_similarity >= threshold:
                matches.append((db_names[i], scaled_similarity))

        # Sort matches by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:n_matches]



    def _filtered_search(self, peak_positions, peak_tolerance, chemical_family, threshold, hey_classification=None):
        """Internal helper for advanced search applying filters."""
        if self.raman.processed_spectra is not None:
            query_spectrum = self.raman.processed_spectra
        elif self.raman.current_spectra is not None:
             query_spectrum = self.raman.current_spectra
        else:
             raise ValueError("No spectrum data available for search.")
    
        query_wavenumbers = self.raman.current_wavenumbers
        matches = []
    
        # Normalize query spectrum once (for correlation calculation)
        query_min = np.min(query_spectrum)
        query_max = np.max(query_spectrum)
        query_norm = (query_spectrum - query_min) / (query_max - query_min) if (query_max > query_min) else query_spectrum
    
    
        for name, data in self.raman.database.items():
            db_meta = data.get('metadata', {})
    
            # 1. Apply Chemical Family Filter
            if chemical_family:
                db_family = db_meta.get('CHEMICAL FAMILY')
                if not db_family or db_family.lower() != chemical_family.lower():
                    continue # Skip if family doesn't match
            
            # 2. Apply Hey Classification Filter (new)
            if hey_classification:
                db_hey_class = db_meta.get('HEY CLASSIFICATION')
                if not db_hey_class or db_hey_class.lower() != hey_classification.lower():
                    continue # Skip if Hey Classification doesn't match
    
            # 3. Apply Peak Position Filter
            if peak_positions:
                db_peak_data = data.get('peaks')
                # Ensure database entry has peaks calculated
                if not db_peak_data or db_peak_data.get('wavenumbers') is None:
                    # Option 1: Skip this entry
                     continue
                else:
                    db_peaks_wavenumbers = db_peak_data['wavenumbers']
                    if not db_peaks_wavenumbers.size:
                         continue # Skip if no peaks
    
                # Check if *all* specified query peaks have a match within tolerance
                all_query_peaks_match = True
                for query_peak_pos in peak_positions:
                    has_match = False
                    for db_peak_wn in db_peaks_wavenumbers:
                        if abs(query_peak_pos - db_peak_wn) <= peak_tolerance:
                            has_match = True
                            break
                    if not has_match:
                        all_query_peaks_match = False
                        break # No need to check further for this db entry
    
                if not all_query_peaks_match:
                    continue # Skip if not all query peaks matched
    
            # 4. Calculate Score (e.g., Correlation) for filtered items
            # If an entry passes all filters, calculate its similarity score
            db_intensities = data['intensities']
            db_wavenumbers = data['wavenumbers']
    
            if not np.array_equal(query_wavenumbers, db_wavenumbers):
                db_intensities_interp = np.interp(query_wavenumbers, db_wavenumbers, db_intensities)
            else:
                db_intensities_interp = db_intensities
    
            db_min = np.min(db_intensities_interp)
            db_max = np.max(db_intensities_interp)
            db_norm = (db_intensities_interp - db_min) / (db_max - db_min) if (db_max > db_min) else db_intensities_interp
    
            try:
                 if np.all(query_norm == query_norm[0]) or np.all(db_norm == db_norm[0]):
                      corr_coef = 0.0
                 else:
                      corr_coef = np.corrcoef(query_norm, db_norm)[0, 1]
                 if np.isnan(corr_coef): corr_coef = 0.0
            except Exception: corr_coef = 0.0
    
            # Use the correlation score, potentially combined with filter match quality later
            # For now, just use correlation if it passes filters
            if corr_coef >= threshold: # Apply threshold to the score
                matches.append((name, corr_coef))
    
        # Matches list now contains items that passed all filters AND the similarity threshold
        return matches

    # --- Results Display and Reporting ---


    def reset_figure_dimensions(self, fig, original_width=6, original_height=4):
        """
        Reset a matplotlib figure to its original dimensions.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure to reset
        original_width : float
            Original width in inches
        original_height : float
            Original height in inches
        """
        # Store original dimensions if not already stored
        if not hasattr(fig, '_original_dims'):
            fig._original_dims = (original_width, original_height)
        else:
            original_width, original_height = fig._original_dims
        
        # Reset the figure size
        fig.set_size_inches(original_width, original_height)
        
        # Reset the subplot parameters to prevent shrinking
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
        
        return fig
 
    

        
        

    def display_search_results(self, matches):
        """Display search results in a comprehensive window."""
        if not matches:
            messagebox.showinfo("Search Results", "No matches found.")
            return

        # Create a new window
        result_window = tk.Toplevel(self.root)
        result_window.title("Search Results")
        result_window.geometry("950x700")  # Adjusted size

        # --- Main Layout ---
        main_frame = ttk.Frame(result_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Paned window for resizable panels
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Left Panel: Match List
        left_frame = ttk.Frame(paned_window, padding=(0, 0, 5, 0))
        paned_window.add(left_frame, weight=1)

        # Right Panel: Visualization/Report Tabs
        right_frame = ttk.Frame(paned_window, padding=(5, 0, 0, 0))
        paned_window.add(right_frame, weight=3)

        # --- Left Panel Content (Match List using Treeview) ---
        match_frame = ttk.LabelFrame(left_frame, text="Matching Compounds", padding=10)
        match_frame.pack(fill=tk.BOTH, expand=True)

        match_columns = ('#0', 'rank', 'name', 'score', 'confidence')
        match_tree = ttk.Treeview(match_frame, columns=match_columns[1:], show='headings', height=20)

        # Define headings
        match_tree.heading('rank', text='Rank')
        match_tree.heading('name', text='Name/ID')
        match_tree.heading('score', text='Score')
        match_tree.heading('confidence', text='Confidence')

        # Configure column widths
        match_tree.column('rank', width=40, anchor=tk.CENTER, stretch=tk.NO)
        match_tree.column('name', width=180, anchor=tk.W)
        match_tree.column('score', width=80, anchor=tk.CENTER)
        match_tree.column('confidence', width=100, anchor=tk.W)

        # Add scrollbar
        match_scrollbar = ttk.Scrollbar(match_frame, orient=tk.VERTICAL, command=match_tree.yview)
        match_tree.configure(yscrollcommand=match_scrollbar.set)
        match_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        match_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Populate Treeview
        for i, (name, score) in enumerate(matches):
            confidence = self.get_confidence_level(score)
            # Use 'name' (dict key) as the item ID (iid)
            match_tree.insert('', tk.END, iid=name, values=(i + 1, name, f"{score:.4f}", confidence))

        # --- Right Panel Content (Tabs) ---
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Import necessary modules here rather than at the top level
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        # Tab 1: Spectral Comparison
        comparison_tab = ttk.Frame(notebook, padding=5)
        notebook.add(comparison_tab, text="Spectral Comparison")

        # Create a frame to properly contain the canvas with fixed dimensions
        canvas_container = ttk.Frame(comparison_tab)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create the comparison figure in a way that we'll completely recreate for each plot
        fig_comp = Figure(figsize=(6, 4), dpi=100)
        fig_comp._original_dims = (6, 4)  # Store original dimensions
        canvas_comp = FigureCanvasTkAgg(fig_comp, master=canvas_container)
        canvas_widget = canvas_comp.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for the comparison plot
        toolbar_frame_comp = ttk.Frame(canvas_container)
        toolbar_frame_comp.pack(fill=tk.X)
        toolbar_comp = NavigationToolbar2Tk(canvas_comp, toolbar_frame_comp)
        toolbar_comp.update()

        # Tab 2: Correlation Analysis
        correlation_tab = ttk.Frame(notebook, padding=5)
        notebook.add(correlation_tab, text="Correlation Analysis")
        
        # Correlation heatmap - create Figure directly
        fig_corr = Figure(figsize=(6, 4))
        fig_corr._original_dims = (6, 4)  # Store original dimensions
        canvas_corr = FigureCanvasTkAgg(fig_corr, master=correlation_tab)
        canvas_corr.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0,5))
        
        # Add toolbar for correlation plot
        toolbar_frame_corr = ttk.Frame(correlation_tab)
        toolbar_frame_corr.pack(fill=tk.X)
        toolbar_corr = NavigationToolbar2Tk(canvas_corr, toolbar_frame_corr)
        toolbar_corr.update()

        # Tab 3: Report
        report_tab = ttk.Frame(notebook, padding=5)
        notebook.add(report_tab, text="Report")
        report_text_frame = ttk.Frame(report_tab)
        report_text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        report_text = tk.Text(report_text_frame, wrap=tk.WORD, height=20)
        report_scrollbar = ttk.Scrollbar(report_text_frame, orient=tk.VERTICAL, command=report_text.yview)
        report_text.config(yscrollcommand=report_scrollbar.set)
        report_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add export buttons for the report
        button_frame = ttk.Frame(report_tab)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(button_frame, text="Export as PDF", 
                  command=lambda: self.export_report_as_pdf(report_text.get(1.0, tk.END))).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export as Text", 
                  command=lambda: self.export_report_as_txt(report_text.get(1.0, tk.END))).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export as CSV", 
                  command=lambda: self.export_results_as_csv(matches)).pack(side=tk.LEFT, padx=5)
        
        
        def redraw_plots(fig_comp, fig_corr, canvas_comp, canvas_corr):
            # Store current figure sizes
            comp_size = fig_comp.get_size_inches()
            corr_size = fig_corr.get_size_inches()
            
            # Temporarily resize (by a small amount) and then resize back
            # This triggers a complete redraw
            fig_comp.set_size_inches(comp_size[0] * 1.01, comp_size[1] * 1.01)
            canvas_comp.draw()
            fig_comp.set_size_inches(comp_size[0], comp_size[1])
            canvas_comp.draw()
            
            fig_corr.set_size_inches(corr_size[0] * 1.01, corr_size[1] * 1.01)
            canvas_corr.draw()
            fig_corr.set_size_inches(corr_size[0], corr_size[1])
            canvas_corr.draw()
        
        
        # --- Interaction Logic ---
        def create_comparison_plot(match_data):
            """Create a comparison plot between the query spectrum and selected match."""
            nonlocal fig_comp, canvas_comp
            
            # Clear the existing figure
            fig_comp.clear()
            
            # Create a fresh subplot
            ax_comp = fig_comp.add_subplot(111)
            
            # Plot data using the helper method
            self.plot_comparison_data(match_data, fig_comp, ax_comp)
            
            # Force a complete redraw
            redraw_plots(fig_comp, fig_corr, canvas_comp, canvas_corr)
            
            # Temporarily resize the container to force a complete refresh
            current_width = canvas_container.winfo_width()
            current_height = canvas_container.winfo_height()
            canvas_container.configure(width=current_width + 1, height=current_height + 1)
            canvas_container.update_idletasks()
            canvas_container.configure(width=current_width, height=current_height)
        
        def create_correlation_heatmap(match_data):
            """Create a correlation heatmap for the selected match."""
            nonlocal fig_corr, canvas_corr
            
            # Clear the existing figure
            fig_corr.clear()
            
            # Create a fresh subplot
            ax_corr = fig_corr.add_subplot(111)
            
            # Plot data using the helper method
            self.plot_correlation_data(match_data, fig_corr, ax_corr)
            
            # Force a complete redraw
            redraw_plots(fig_comp, fig_corr, canvas_comp, canvas_corr)
            
            # Temporarily resize the container to force a complete refresh
            current_width = canvas_container.winfo_width()
            current_height = canvas_container.winfo_height()
            canvas_container.configure(width=current_width + 1, height=current_height + 1)
            canvas_container.update_idletasks()
            canvas_container.configure(width=current_width, height=current_height)
        
        def generate_match_report(match_data):
            """Generate a report for the selected match."""
            # Clear existing report
            report_text.config(state=tk.NORMAL)
            report_text.delete(1.0, tk.END)
            
            # Generate the report for just this match
            self.generate_match_report([match_data], report_text)
            
        def on_match_select(event=None):
            """Handle the selection of a match from the search results."""
            # Get selected item from tree
            selected_items = match_tree.selection()
            if not selected_items:
                return
            
            # Get the match data
            item_id = selected_items[0]
            item_data = match_tree.item(item_id)
            match_name = item_id  # The item ID is the match name
            match_score = float(item_data['values'][2])  # Score is the third column
            
            # Create match data tuple
            match_data = (match_name, match_score)
            
            # Update the plots and report
            create_comparison_plot(match_data)
            create_correlation_heatmap(match_data)
            generate_match_report(match_data)
            
        # Bind the selection event
        match_tree.bind('<<TreeviewSelect>>', on_match_select)

        # --- Initial Display ---
        if matches:
            first_item_iid = matches[0][0]
            match_tree.selection_set(first_item_iid)
            match_tree.focus(first_item_iid)
            on_match_select()
        else:
            # Create empty plots for when there are no matches
            fig_comp.clear()
            ax_comp = fig_comp.add_subplot(111)
            ax_comp.text(0.5, 0.5, 'No matches to display.', ha='center', va='center', transform=ax_comp.transAxes)
            canvas_comp.draw()
            
            fig_corr.clear()
            ax_corr = fig_corr.add_subplot(111)
            ax_corr.text(0.5, 0.5, 'No matches for correlation.', ha='center', va='center', transform=ax_corr.transAxes)
            canvas_corr.draw()
            
            report_text.config(state=tk.NORMAL)
            report_text.delete(1.0, tk.END)
            report_text.insert(tk.END, "No matches found.")
            report_text.config(state=tk.DISABLED)

        # Make window modal
        result_window.transient(self.root)
        result_window.grab_set()
        self.root.wait_window(result_window)
        
      
            
            
    
    
    
    def plot_correlation_data(self, match_data, fig, ax):
        """Plot correlation heatmap for spectral regions between query and selected match."""
        if self.raman.current_spectra is None:
            ax.text(0.5, 0.5, 'No data for correlation heatmap.', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Reset figure dimensions
        self.reset_figure_dimensions(fig)
        
        # Get current spectrum (processed if available)
        if self.raman.processed_spectra is not None:
            current_spectrum = self.raman.processed_spectra
        else:
            current_spectrum = self.raman.current_spectra
        query_wavenumbers = self.raman.current_wavenumbers

        # Normalize query spectrum (0-1)
        query_max = np.max(current_spectrum)
        current_norm = current_spectrum / query_max if query_max > 0 else current_spectrum

        # Get match data
        match_name, score = match_data
        if match_name not in self.raman.database:
            ax.text(0.5, 0.5, f'Match {match_name} not found.', ha='center', va='center', transform=ax.transAxes)
            return
        
        match_data = self.raman.database[match_name]
        match_wavenumbers = match_data['wavenumbers']
        match_intensities = match_data['intensities']

        # Interpolate if needed
        if not np.array_equal(query_wavenumbers, match_wavenumbers):
            match_intensities_interp = np.interp(query_wavenumbers, match_wavenumbers, match_intensities)
        else:
            match_intensities_interp = match_intensities

        # Normalize match spectrum (0-1)
        match_max = np.max(match_intensities_interp)
        match_norm = match_intensities_interp / match_max if match_max > 0 else match_intensities_interp

        # Define spectral regions (adjust as needed)
        regions = [
            (200, 500, "Fingerprint Low"),
            (500, 1000, "Fingerprint Mid"),
            (1000, 1800, "Fingerprint High"),
            (2800, 3200, "CH Stretch"),
        ]

        region_scores = []
        region_labels = []

        for start, end, label in regions:
            indices = np.where((query_wavenumbers >= start) & (query_wavenumbers <= end))[0]
            if len(indices) > 1:  # Need at least 2 points for correlation
                region_current = current_norm[indices]
                region_match = match_norm[indices]

                # Calculate correlation coefficient for the region
                try:
                    if np.all(region_current == region_current[0]) or np.all(region_match == region_match[0]):
                        corr = 0.0  # Undefined for constant vectors
                    else:
                        corr = np.corrcoef(region_current, region_match)[0, 1]
                    if np.isnan(corr): corr = 0.0
                except Exception:
                    corr = 0.0  # Handle potential errors

                region_scores.append(corr)
                region_labels.append(f"{label}\n({start}-{end} cm⁻¹)")

        # Plot correlation heatmap
        if region_scores:
            # Use built-in colormaps directly from matplotlib
            import matplotlib.cm as cm
            cmap = cm.RdYlGn  # Red-Yellow-Green colormap (good for correlation)
            regions_array = np.array(region_scores).reshape(1, -1)

            im = ax.imshow(regions_array, cmap=cmap, aspect='auto', vmin=0, vmax=1)  # Correlation typically 0-1

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
            cbar.set_label('Region Correlation Coefficient')

            # Configure axes
            ax.set_yticks([])  # No Y ticks needed for a single row
            
            ax.set_xticks(np.arange(len(region_labels)))
            ax.set_xticklabels(region_labels, rotation=45, ha='right', fontsize=9)

            # Add correlation values as text
            for i, score in enumerate(region_scores):
                text_color = 'black' if 0.3 <= score <= 0.7 else 'white'  # Contrast based on score
                ax.text(i, 0, f"{score:.2f}", ha='center', va='center', color=text_color, fontweight='bold')

            ax.set_title(f'Spectral Region Correlation: Query vs. {match_name}')
        else:
            ax.text(0.5, 0.5, "Insufficient data or regions for correlation analysis",
                    ha='center', va='center', transform=ax.transAxes)

        # Make sure the figure is properly formatted
        fig.tight_layout()
    
    
    

    def plot_comparison_data(self, match_data, fig, ax):
        """Plot the comparison data without any existing plot state influence."""
        if self.raman.current_spectra is None:
            ax.text(0.5, 0.5, 'Query spectrum not loaded.', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Reset figure dimensions
        self.reset_figure_dimensions(fig)
        
        # Get current spectrum (processed if available)
        if self.raman.processed_spectra is not None:
            current_spectrum = self.raman.processed_spectra
            query_label = 'Query (Processed)'
        else:
            current_spectrum = self.raman.current_spectra
            query_label = 'Query (Raw)'
        query_wavenumbers = self.raman.current_wavenumbers
        
        # Normalize query spectrum (optional, based on GUI setting)
        normalize = self.var_normalize.get() if hasattr(self, 'var_normalize') else True
        if normalize:
            query_max = np.max(current_spectrum)
            current_norm = current_spectrum / query_max if query_max > 0 else current_spectrum
        else:
            current_norm = current_spectrum
        
        # Plot query spectrum
        ax.plot(query_wavenumbers, current_norm, 'b-', label=query_label, linewidth=1.5)
        
        # Get match data
        match_name, score = match_data
        if match_name not in self.raman.database:
            return  # Skip if somehow name is invalid
        
        match_data = self.raman.database[match_name]
        match_wavenumbers = match_data['wavenumbers']
        match_intensities = match_data['intensities']

        # Interpolate if needed
        if not np.array_equal(query_wavenumbers, match_wavenumbers):
            match_intensities_interp = np.interp(query_wavenumbers, match_wavenumbers, match_intensities)
        else:
            match_intensities_interp = match_intensities

        # Normalize match spectrum (optional)
        if normalize:
            match_max = np.max(match_intensities_interp)
            match_norm = match_intensities_interp / match_max if match_max > 0 else match_intensities_interp
        else:
            match_norm = match_intensities_interp

        # Plot match spectrum
        ax.plot(query_wavenumbers, match_norm, 'r-', 
                label=f'{match_name} ({score:.2f})', linewidth=1.0, alpha=0.8)

        # Add difference plot if requested
        show_diff = hasattr(self, 'var_show_diff') and self.var_show_diff.get()
        if show_diff:
            diff = current_norm - match_norm
            
            # Create twin axis for difference plot
            ax2 = ax.twinx()
            ax2.plot(query_wavenumbers, diff, 'k--', label='Difference', alpha=0.6, linewidth=0.8)
            ax2.set_ylabel('Difference', color='k')
            ax2.tick_params(axis='y', labelcolor='k')
            ax2.legend(loc='upper left', fontsize='small')

        # Highlight matching peaks (optional)
        highlight_peaks = hasattr(self, 'var_highlight_peaks') and self.var_highlight_peaks.get()
        if highlight_peaks and 'peaks' in match_data and match_data['peaks'] and self.raman.peaks:
            # Get peaks
            db_peak_wn = match_data['peaks'].get('wavenumbers', np.array([]))
            query_peak_wn = self.raman.peaks.get('wavenumbers', np.array([]))
            tolerance = 10  # Adjust tolerance

            # Find matching peaks
            for q_wn in query_peak_wn:
                for db_idx, m_wn in enumerate(db_peak_wn):
                    if abs(q_wn - m_wn) <= tolerance:
                        # Find corresponding heights in the *normalized* spectra for plotting
                        q_plot_idx = np.abs(query_wavenumbers - q_wn).argmin()
                        m_plot_idx = np.abs(query_wavenumbers - m_wn).argmin()
                        q_plot_height = current_norm[q_plot_idx]
                        m_plot_height = match_norm[m_plot_idx]

                        # Mark matching peaks on the plot
                        ax.plot(m_wn, m_plot_height, 'ro', markersize=5, alpha=0.7)
                        break  # Match found for this query peak

        # Configure plot appearance
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Intensity (Normalized)' if normalize else 'Intensity (a.u.)')
        ax.set_title(f'Query vs. {match_name}')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Make sure the figure is properly formatted
        fig.tight_layout()


    def generate_match_report(self, matches, text_widget):
        """
        Generate a detailed report of the search matches.
        
        Parameters:
        -----------
        matches : list of tuples
            List of (name, score) tuples for matched spectra
        text_widget : tk.Text
            Text widget to display the report
        """
        try:
            # Clear the text widget
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            
            if not matches:
                text_widget.insert(tk.END, "No matches found.\n")
                return
            
            # Add header
            text_widget.insert(tk.END, "Search Match Results\n")
            text_widget.insert(tk.END, "=" * 50 + "\n\n")
            
            # Add timestamp
            text_widget.insert(tk.END, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add current spectrum info if available
            if hasattr(self.raman, 'current_spectra') and self.raman.current_spectra is not None:
                text_widget.insert(tk.END, "Current Spectrum Information:\n")
                text_widget.insert(tk.END, "-" * 30 + "\n")
                if self.raman.metadata:
                    for key, value in self.raman.metadata.items():
                        text_widget.insert(tk.END, f"{key}: {value}\n")
                text_widget.insert(tk.END, "\n")
            
            # Add matches
            text_widget.insert(tk.END, "Top Matches:\n")
            text_widget.insert(tk.END, "-" * 30 + "\n")
            
            for i, (name, score) in enumerate(matches, 1):
                confidence = self.get_confidence_level(score)
                
                text_widget.insert(tk.END, f"\nMatch #{i}\n")
                text_widget.insert(tk.END, f"Name: {name}\n")
                text_widget.insert(tk.END, f"Match Score: {score:.2f}\n")
                text_widget.insert(tk.END, f"Confidence: {confidence}\n")
                
                # Get metadata if available
                metadata = None
                if name in self.raman.database:
                    metadata = self.raman.database[name].get('metadata', {})
                
                if metadata:
                    text_widget.insert(tk.END, "\nMetadata:\n")
                    for key, value in metadata.items():
                        text_widget.insert(tk.END, f"  {key}: {value}\n")
                
                text_widget.insert(tk.END, "\n" + "-" * 30 + "\n")
            
            # Add summary statistics
            text_widget.insert(tk.END, "\nSummary Statistics:\n")
            text_widget.insert(tk.END, "-" * 30 + "\n")
            text_widget.insert(tk.END, f"Total Matches: {len(matches)}\n")
            
            if matches:
                avg_score = sum(score for _, score in matches) / len(matches)
                text_widget.insert(tk.END, f"Average Match Score: {avg_score:.2f}\n")
            
            # Disable editing
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            # Handle any errors that occur during report generation
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, f"Error generating report: {str(e)}\n")
            text_widget.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")


    
    def generate_correlation_heatmap(self, matches, fig, ax, canvas):
        """Generate correlation heatmap for spectral regions between query and best match."""
        if self.raman.current_spectra is None or not matches:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No data for correlation heatmap.', ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return ax

        # Reset figure dimensions
        self.reset_figure_dimensions(fig)
        
        # Clear previous plot completely by clearing the whole figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Get current spectrum (processed if available)
        if self.raman.processed_spectra is not None:
            current_spectrum = self.raman.processed_spectra
        else:
            current_spectrum = self.raman.current_spectra
        query_wavenumbers = self.raman.current_wavenumbers

        # Normalize query spectrum (0-1)
        query_max = np.max(current_spectrum)
        current_norm = current_spectrum / query_max if query_max > 0 else current_spectrum

        # Get best match (first in the list)
        best_match_name = matches[0][0]
        if best_match_name not in self.raman.database:
            ax.text(0.5, 0.5, f'Best match {best_match_name} not found.', ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return ax
        
        # [Rest of the function continues as before]
        
        # Return the new axis
        return ax
    
        match_data = self.raman.database[best_match_name]
        match_wavenumbers = match_data['wavenumbers']
        match_intensities = match_data['intensities']
    
        # Interpolate if needed
        if not np.array_equal(query_wavenumbers, match_wavenumbers):
            match_intensities_interp = np.interp(query_wavenumbers, match_wavenumbers, match_intensities)
        else:
            match_intensities_interp = match_intensities
    
        # Normalize match spectrum (0-1)
        match_max = np.max(match_intensities_interp)
        match_norm = match_intensities_interp / match_max if match_max > 0 else match_intensities_interp
    
        # Define spectral regions (adjust as needed)
        regions = [
            (200, 500, "Fingerprint Low"),
            (500, 1000, "Fingerprint Mid"),
            (1000, 1800, "Fingerprint High"),
            (2800, 3200, "CH Stretch"),
            # (3200, 3700, "OH/NH Stretch")  # Example add more
        ]
    
        region_scores = []
        region_labels = []
    
        for start, end, label in regions:
            indices = np.where((query_wavenumbers >= start) & (query_wavenumbers <= end))[0]
            if len(indices) > 1:  # Need at least 2 points for correlation
                region_current = current_norm[indices]
                region_match = match_norm[indices]
    
                # Calculate correlation coefficient for the region
                try:
                    if np.all(region_current == region_current[0]) or np.all(region_match == region_match[0]):
                        corr = 0.0  # Undefined for constant vectors
                    else:
                        corr = np.corrcoef(region_current, region_match)[0, 1]
                    if np.isnan(corr): corr = 0.0
                except Exception:
                    corr = 0.0  # Handle potential errors
    
                region_scores.append(corr)
                region_labels.append(f"{label}\n({start}-{end} cm⁻¹)")
    
        # Plot correlation heatmap
        if region_scores:
            # Use built-in colormaps directly from the figure
            import matplotlib.cm as cm
            cmap = cm.RdYlGn  # Red-Yellow-Green colormap (good for correlation)
            regions_array = np.array(region_scores).reshape(1, -1)
    
            im = ax.imshow(regions_array, cmap=cmap, aspect='auto', vmin=0, vmax=1)  # Correlation typically 0-1
    
            # Add colorbar properly
            # Check if a colorbar already exists
            if hasattr(fig, '_colorbar'):
                fig._colorbar.remove()
            fig._colorbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
            fig._colorbar.set_label('Region Correlation Coefficient')
    
            # Configure axes
            ax.set_yticks([])  # No Y ticks needed for a single row
            
            ax.set_xticks(np.arange(len(region_labels)))
            ax.set_xticklabels(region_labels, rotation=45, ha='right', fontsize=9)
    
            # Add correlation values as text
            for i, score in enumerate(region_scores):
                text_color = 'black' if 0.3 <= score <= 0.7 else 'white'  # Contrast based on score
                ax.text(i, 0, f"{score:.2f}", ha='center', va='center', color=text_color, fontweight='bold')
    
            ax.set_title(f'Spectral Region Correlation: Query vs. {best_match_name}')
        else:
            ax.text(0.5, 0.5, "Insufficient data or regions for correlation analysis",
                    ha='center', va='center', transform=ax.transAxes)
    
        fig.tight_layout()  # Adjust layout
        canvas.draw()
        return ax 

    def export_report_as_pdf(self, report_text):
        """Export the report as a PDF file using reportlab."""
        if not REPORTLAB_AVAILABLE:
             messagebox.showerror("PDF Export Error", "ReportLab library not found. Cannot export as PDF.")
             return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save Report as PDF"
            )
            if not filename: return # User cancelled

            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            # Customize styles slightly
            styles['Title'].alignment = 1 # Center alignment
            styles['h2'].spaceBefore = 10
            styles['h2'].spaceAfter = 2
            styles['Normal'].spaceAfter = 4
            styles['Code'].fontSize = 8 # For peak list maybe?

            lines = report_text.split('\n')
            in_match_section = False

            for i, line in enumerate(lines):
                 line = line.strip()
                 if not line: # Skip empty lines unless needed for spacing
                      # elements.append(Spacer(1, 6)) # Add manual space if needed
                      continue

                 if line == "Raman Spectrum Analysis Report":
                      elements.append(Paragraph(line, styles['Title']))
                 elif line.startswith("="*5): # Separator
                      elements.append(Spacer(1, 12))
                 elif line in ["Sample Information", "Processing Information", "Match Results", "Analysis and Recommendations"]:
                      elements.append(Paragraph(line, styles['h2']))
                      in_match_section = (line == "Match Results")
                 elif line.startswith("-"*5): # Underline for sections
                      pass # Skip underline
                 elif in_match_section and line[0].isdigit() and ". " in line: # Match name line
                      elements.append(Paragraph(line, styles['h3'])) # Use h3 for match name
                 elif line.startswith("   Match Score:") or line.startswith("   Compound Information:"):
                      elements.append(Paragraph(line, styles['Normal']))
                 elif line.startswith("   - "): # Compound details
                      elements.append(Paragraph(line, styles['Normal']))
                 elif line.startswith("Detected Peaks"):
                      # Potentially style the peak list differently
                      parts = line.split(":", 1)
                      if len(parts) == 2:
                           elements.append(Paragraph(f"<b>{parts[0]}:</b> {parts[1]}", styles['Normal']))
                      else:
                           elements.append(Paragraph(line, styles['Normal']))
                 else: # Default normal style
                      elements.append(Paragraph(line, styles['Normal']))


            doc.build(elements)
            messagebox.showinfo("Export Successful", f"Report exported to {filename}")

        except Exception as e:
            messagebox.showerror("PDF Export Error", f"Error exporting report as PDF: {str(e)}")

    def export_report_as_txt(self, report_text):
         """Export the report as a plain text file."""
         try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Report as TXT"
            )
            if not filename: return # User cancelled

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)

            messagebox.showinfo("Export Successful", f"Report exported to {filename}")

         except Exception as e:
            messagebox.showerror("TXT Export Error", f"Error exporting report as TXT: {str(e)}")


    def export_results_as_csv(self, matches):
        """Export search results (matches) as a CSV file."""
        if not matches:
            messagebox.showerror("Export Error", "No matches to export.")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Results as CSV"
            )
            if not filename: return # User cancelled

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(["Rank", "Name", "Score", "Confidence",
                                 "Ideal Chemistry", "Chemical Family", "Locality"]) # Add more metadata?

                # Write data rows
                for i, (name, score) in enumerate(matches):
                    confidence = self.get_confidence_level(score)
                    chemistry = ""
                    family = ""
                    locality = ""

                    # Get metadata if available from database
                    if name in self.raman.database:
                        data = self.raman.database[name]
                        if 'metadata' in data and data.get('metadata'):
                            chemistry = data['metadata'].get('IDEAL CHEMISTRY', '')
                            family = data['metadata'].get('CHEMICAL FAMILY', '')
                            locality = data['metadata'].get('LOCALITY', '')

                    writer.writerow([i + 1, name, f"{score:.4f}", confidence,
                                     chemistry, family, locality])

            messagebox.showinfo("Export Successful", f"Results exported to {filename}")

        except Exception as e:
            messagebox.showerror("CSV Export Error", f"Error exporting results as CSV: {str(e)}")

    # --- Helper Methods ---

    def get_confidence_level(self, score):
        """Convert numeric score (0-1) to confidence level description."""
        # Adjust thresholds based on expected score range and meaning
        if score >= 0.95: return "Excellent"
        elif score >= 0.85: return "Good"
        elif score >= 0.75: return "Possible"
        elif score >= 0.65: return "Weak"
        else: return "Poor"

    # def update_plot(self, include_background=False, include_peaks=False):
    #     """Update the main plot with current/processed spectrum, background, and peaks."""
    #     if self.raman.current_spectra is None:
    #         self.ax.clear()
    #         self.ax.text(0.5, 0.5, 'No Spectrum Loaded', ha='center', va='center', 
    #                      transform=self.ax.transAxes)
    #         self.canvas.draw()
    #         return
    
    #     # Clear existing plot
    #     self.ax.clear()
    
    #     # Plot processed spectrum if available, otherwise raw
    #     spectrum_to_plot = self.raman.current_spectra
    #     label = 'Raw Spectrum'
    #     color = 'blue'
    #     if self.raman.processed_spectra is not None:
    #         spectrum_to_plot = self.raman.processed_spectra
    #         label = 'Processed Spectrum'
    #         color = 'green'
        
    #     # Plot the spectrum using the EXISTING axis object
    #     self.ax.plot(self.raman.current_wavenumbers, spectrum_to_plot, color=color, 
    #                  label=label, linewidth=1.5)
        
    #     # Plot background if requested and available
    #     if include_background and self.raman.background is not None:
    #         self.ax.plot(self.raman.current_wavenumbers, self.raman.background, 
    #                      'r--', label='Background', linewidth=1)
        
    #     # # Mark peaks if requested and available
    #     # if include_peaks and self.raman.peaks is not None:
    #     #     peak_indices = self.raman.peaks.get('indices')
    #     #     if peak_indices is not None:
    #     #         # Use indices to get heights from the plotted spectrum
    #     #         peak_heights = spectrum_to_plot[peak_indices]
    #     #         peak_wavenumbers = self.raman.current_wavenumbers[peak_indices]
    #     #         self.ax.plot(peak_wavenumbers, peak_heights, 'ro', label='Peaks', markersize=4)
        
        
    #     # Mark peaks if requested and available
    #     if include_peaks and self.raman.peaks is not None:
    #         peak_indices = self.raman.peaks.get('indices')
    #         if peak_indices is not None:
    #             # Use indices to get heights from the plotted spectrum
    #             peak_heights = spectrum_to_plot[peak_indices]
    #             peak_wavenumbers = self.raman.current_wavenumbers[peak_indices]
    #             self.ax.plot(peak_wavenumbers, peak_heights, 'ro', label='Peaks', markersize=4)
                
    #             # Add wavenumber annotations for each peak
    #             for i, (wn, height) in enumerate(zip(peak_wavenumbers, peak_heights)):
    #                 self.ax.annotate(
    #                     f"{wn:.1f}",  # Format to 1 decimal place
    #                     xy=(wn, height),  # Point to annotate
    #                     xytext=(0, 5),  # Offset text by 5 points above
    #                     textcoords='offset points',
    #                     ha='center',  # Horizontally center text
    #                     fontsize=8,  # Small font to avoid crowding
    #                 )
    # # Configure plot appearance
    #     self.ax.set_xlabel('Wavenumber (cm⁻¹)')
    #     self.ax.set_ylabel('Intensity (a.u.)')
    #     self.ax.set_title('Raman Spectrum Analysis')
    #     self.ax.legend()
    #     self.ax.grid(True, linestyle=':', alpha=0.6)
        
    #     # Update the canvas - this is critical
    #     self.canvas.draw()
    
    
    def update_plot(self, include_background=False, include_peaks=False):
        """Update the main plot with current/processed spectrum, background, and peaks."""
        if self.raman.current_spectra is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No Spectrum Loaded', ha='center', va='center', 
                         transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        # Completely clear the figure and create a new axis
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        # Plot processed spectrum if available, otherwise raw
        spectrum_to_plot = self.raman.current_spectra
        label = 'Raw Spectrum'
        color = 'blue'
        if self.raman.processed_spectra is not None:
            spectrum_to_plot = self.raman.processed_spectra
            label = 'Processed Spectrum'
            color = 'green'
        
        # Plot the spectrum
        self.ax.plot(self.raman.current_wavenumbers, spectrum_to_plot, color=color, 
                     label=label, linewidth=1.5)
        
        # Plot background if requested and available
        if include_background and self.raman.background is not None:
            self.ax.plot(self.raman.current_wavenumbers, self.raman.background, 
                         'r--', label='Background', linewidth=1)
        
        # Mark peaks if requested and available
        if include_peaks and self.raman.peaks is not None:
            peak_indices = self.raman.peaks.get('indices')
            if peak_indices is not None:
                # Use indices to get heights from the plotted spectrum
                peak_heights = spectrum_to_plot[peak_indices]
                peak_wavenumbers = self.raman.current_wavenumbers[peak_indices]
                self.ax.plot(peak_wavenumbers, peak_heights, 'ro', label='Peaks', markersize=4)
                
                # Add wavenumber annotations for each peak
                for i, (wn, height) in enumerate(zip(peak_wavenumbers, peak_heights)):
                    self.ax.annotate(
                        f"{wn:.1f}",  # Format to 1 decimal place
                        xy=(wn, height),  # Point to annotate
                        xytext=(0, 5),  # Offset text by 5 points above
                        textcoords='offset points',
                        ha='center',  # Horizontally center text
                        fontsize=8,  # Small font to avoid crowding
                    )
        
        # Configure plot appearance
        self.ax.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax.set_ylabel('Intensity (a.u.)')
        self.ax.set_title('Raman Spectrum Analysis')
        self.ax.legend()
        self.ax.grid(True, linestyle=':', alpha=0.6)
        
        # Update the canvas
        self.canvas.draw()

# --- Main Execution ---
def main():
    """Main function to run the application."""
    
    # Close any existing matplotlib figures
    plt.close('all')
    
    root = tk.Tk()
    # Optional: Apply a theme
    try:
        style = ttk.Style(root)
        # Available themes: 'clam', 'alt', 'default', 'classic'
        # Some systems might have 'vista', 'xpnative'
        available_themes = style.theme_names()
        # print(f"Available themes: {available_themes}")
        if 'clam' in available_themes:
             style.theme_use('clam')
        elif 'vista' in available_themes:
             style.theme_use('vista')
    except Exception as e:
        print(f"Could not set theme: {e}")

    app = RamanAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()