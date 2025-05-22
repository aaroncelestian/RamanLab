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

matplotlib.use("Agg")  # Set backend to Agg to prevent blank figure window
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import find_peaks, savgol_filter
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
from pathlib import Path
import csv
from datetime import datetime
from io import BytesIO
from version import __version__, __author__, __copyright__

# from cnn_model import RamanCNN  # Import the RamanCNN class
# from search_functions import plot_match_comparison, generate_correlation_heatmap, generate_match_report
# import types


# Import the RamanSpectra class
from raman_spectra import RamanSpectra

# Import the Peak Fitting Window
from peak_fitting import PeakFittingWindow

# Try importing reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
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
        self.root.title("ClaritySpectra")
        self.root.geometry(
            "1350x900"
        )  # Increased from 1200x900 to accommodate wider tabs
        self.root.minsize(1150, 700)  # Increased minimum width from 1000 to 1150

        # Initialize RamanSpectra instance
        self.raman = RamanSpectra()

        # Initialize selected match
        self.selected_match = tk.StringVar()

        # Create GUI
        self.create_gui()

        # Create instance variable for db_stats_text as a backup precaution
        # It will be properly initialized in create_database_tab, but this ensures it exists
        if not hasattr(self, "db_stats_text"):
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

        self.frame_right = ttk.Frame(
            self.root, padding=10, width=450
        )  # Increased width from default
        self.frame_right.pack(side=tk.RIGHT, fill=tk.Y)
        self.frame_right.pack_propagate(False)  # Prevent the frame from shrinking

        # Left panel - visualization
        self.create_visualization_panel()

        # Right panel - controls
        self.create_control_panel()

    def create_visualization_panel(self):
        """Create the visualization panel."""
        # Create visualization frame
        viz_frame = ttk.LabelFrame(
            self.frame_left, text="Spectrum Visualization", padding=10
        )
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Create initial empty figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Raman Spectrum")

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
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Spectrum", command=self.import_spectrum)
        file_menu.add_command(label="Save Spectrum", command=self.save_spectrum)
        file_menu.add_command(label="Batch Import", command=self.batch_import)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Peak Fitting", command=self.open_peak_fitting)
        analysis_menu.add_command(
            label="Batch Peak Fitting", command=self.open_batch_peak_fitting
        )
        analysis_menu.add_command(
            label="2D Map Analysis", command=self.open_2d_map_analysis
        )
        analysis_menu.add_command(
            label="Raman Group Analysis", command=self.open_raman_group_analysis
        )
        analysis_menu.add_command(
            label="Mixed Mineral Analysis", command=self.analyze_mixed_minerals
        )

        # Database menu
        database_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Database", menu=database_menu)
        database_menu.add_command(
            label="Add Current Spectrum", command=lambda: self.add_to_database()
        )
        database_menu.add_command(
            label="View/Search Database", command=self.view_database
        )
        database_menu.add_command(label="Batch Import", command=self.batch_import)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
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
            about_window, text="ClaritySpectra", font=("TkDefaultFont", 14, "bold")
        ).pack(pady=(20, 5))

        # Version info
        ttk.Label(about_window, text=f"Version {__version__}").pack(pady=2)

        # Author info
        ttk.Label(
            about_window,
            text=f"Created by {__author__} \n"
            "Natural History Museum of Los Angeles County \n"
            "Department of Mineral Sciences",
        ).pack(pady=2)

        # Description
        description = ttk.Label(
            about_window,
            text="ClaritySpectra: A comprehensive tool for analyzing Raman spectra,\n"
            "with features for background subtraction, peak finding,\n"
            "and spectrum identification using database matching.",
            justify=tk.CENTER,
        )
        description.pack(pady=10)

        # Copyright info
        ttk.Label(
            about_window,
            text=f"{__copyright__}\n"
            "RRUFF data use for minerals\n"
            "https://www.rruff.info \n"
            "Plastic data from SLOPP and SLOPP-E \n"
            "https://tinyurl.com/2ek3zceb",
        ).pack(pady=10)

        # Close button
        ttk.Button(about_window, text="Close", command=about_window.destroy).pack(
            pady=10
        )

    def create_control_panel(self):
        """Create the control panel with tabs for different functions."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(
            self.frame_right, width=350
        )  # Set explicit width for notebook
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.tab_file = ttk.Frame(
            self.notebook, width=340
        )  # Set explicit width for tab frames
        self.tab_process = ttk.Frame(self.notebook, width=340)
        self.tab_search = ttk.Frame(self.notebook, width=340)
        self.tab_database = ttk.Frame(self.notebook, width=340)
        self.tab_peak_fitting = ttk.Frame(
            self.notebook, width=340
        )  # New tab for peak fitting

        # Prevent tabs from shrinking
        self.tab_file.pack_propagate(False)
        self.tab_process.pack_propagate(False)
        self.tab_search.pack_propagate(False)
        self.tab_database.pack_propagate(False)
        self.tab_peak_fitting.pack_propagate(False)

        # Add tabs to notebook
        self.notebook.add(self.tab_file, text="File")
        self.notebook.add(self.tab_process, text="Process")
        self.notebook.add(self.tab_search, text="Search")
        self.notebook.add(self.tab_database, text="Database")
        self.notebook.add(
            self.tab_peak_fitting, text="Advanced Analysis"
        )  # Changed label from "Peak Fitting"

        # Create content for each tab
        self.create_file_tab()
        self.create_process_tab()
        self.create_search_tab(self.tab_search)  # Pass the parent tab as argument
        self.create_database_tab()
        self.create_advanced_analysis_tab()  # Changed from create_peak_fitting_tab

    def create_file_tab(self):
        """Create content for the file operations tab."""
        # File operations frame
        file_frame = ttk.LabelFrame(self.tab_file, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            file_frame, text="Import Spectrum", command=self.import_spectrum
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            file_frame, text="Save Current Spectrum", command=self.save_spectrum
        ).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Edit Metadata", command=self.edit_metadata).pack(
            fill=tk.X, pady=2
        )
        

        # Metadata display frame
        self.metadata_frame = ttk.LabelFrame(
            self.tab_file, text="Spectrum Metadata", padding=10
        )
        self.metadata_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.metadata_text = tk.Text(
            self.metadata_frame, height=10, width=30, wrap=tk.WORD
        )
        self.metadata_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.metadata_text.config(state=tk.DISABLED)

        # Add scrollbar to metadata
        scrollbar = ttk.Scrollbar(
            self.metadata_frame, orient=tk.VERTICAL, command=self.metadata_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.metadata_text.config(yscrollcommand=scrollbar.set)

    def create_process_tab(self):
        """Create content for the spectrum processing tab."""
        # Background subtraction frame
        bg_frame = ttk.LabelFrame(
            self.tab_process, text="Background Subtraction", padding=10
        )
        bg_frame.pack(fill=tk.X, pady=5)

        ttk.Label(bg_frame, text="Baseline λ:").pack(anchor=tk.W)
        self.var_lambda = tk.StringVar(value="1e5")
        ttk.Entry(bg_frame, textvariable=self.var_lambda).pack(fill=tk.X, pady=2)

        ttk.Label(bg_frame, text="Baseline p:").pack(anchor=tk.W)
        self.var_p = tk.StringVar(value="0.01")
        ttk.Entry(bg_frame, textvariable=self.var_p).pack(fill=tk.X, pady=2)

        ttk.Button(
            bg_frame, text="Subtract Background", command=self.subtract_background
        ).pack(fill=tk.X, pady=5)

        # Savitzky-Golay smoothing frame
        sg_frame = ttk.LabelFrame(
            self.tab_process, text="Savitzky-Golay Smoothing", padding=10
        )
        sg_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sg_frame, text="Window Length (odd, >= 5):").pack(anchor=tk.W)
        self.var_sg_window = tk.StringVar(value="11")
        ttk.Entry(sg_frame, textvariable=self.var_sg_window).pack(fill=tk.X, pady=2)

        ttk.Label(sg_frame, text="Polynomial Order (>= 2):").pack(anchor=tk.W)
        self.var_sg_poly = tk.StringVar(value="3")
        ttk.Entry(sg_frame, textvariable=self.var_sg_poly).pack(fill=tk.X, pady=2)

        # Button frame for Apply and Preview
        sg_button_frame = ttk.Frame(sg_frame)
        sg_button_frame.pack(fill=tk.X, pady=2)
        ttk.Button(
            sg_button_frame, text="Apply Smoothing", command=self.apply_savgol_smoothing
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(
            sg_button_frame,
            text="Preview Smoothing",
            command=self.preview_savgol_smoothing,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Peak finding frame
        peak_frame = ttk.LabelFrame(self.tab_process, text="Peak Finding", padding=10)
        peak_frame.pack(fill=tk.X, pady=5)

        ttk.Label(peak_frame, text="Peak Height Threshold:").pack(anchor=tk.W)
        self.var_height = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_height).pack(fill=tk.X, pady=2)

        ttk.Label(peak_frame, text="Min. Peak Distance:").pack(anchor=tk.W)
        self.var_distance = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_distance).pack(fill=tk.X, pady=2)

        ttk.Label(peak_frame, text="Peak Prominence:").pack(anchor=tk.W)
        self.var_prominence = tk.StringVar(value="Auto")
        ttk.Entry(peak_frame, textvariable=self.var_prominence).pack(fill=tk.X, pady=2)

        ttk.Button(peak_frame, text="Find Peaks", command=self.find_peaks).pack(
            fill=tk.X, pady=5
        )

        # Display options frame
        display_frame = ttk.LabelFrame(
            self.tab_process, text="Display Options", padding=10
        )
        display_frame.pack(fill=tk.X, pady=5)

        self.var_show_background = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame,
            text="Show Background",
            variable=self.var_show_background,
            command=lambda: self.update_plot(
                include_background=self.var_show_background.get(),
                include_peaks=self.var_show_peaks.get(),
            ),
        ).pack(anchor=tk.W, pady=2)

        self.var_show_peaks = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame,
            text="Show Peaks",
            variable=self.var_show_peaks,
            command=lambda: self.update_plot(
                include_background=self.var_show_background.get(),
                include_peaks=self.var_show_peaks.get(),
            ),
        ).pack(anchor=tk.W, pady=2)

        # # Advanced Analysis frame
        # advanced_frame = ttk.LabelFrame(
        #     self.tab_process, text="Advanced Analysis", padding=10
        # )
        # advanced_frame.pack(fill=tk.X, pady=5)

        # ttk.Button(
        #     advanced_frame, text="Peak Fitting", command=self.open_peak_fitting
        # ).pack(fill=tk.X, pady=2)
        # ttk.Button(
        #     advanced_frame,
        #     text="Batch Peak Fitting",
        #     command=self.open_batch_peak_fitting,
        # ).pack(fill=tk.X, pady=2)
        # ttk.Button(
        #     advanced_frame, text="2D Map Analysis", command=self.open_2d_map_analysis
        # ).pack(fill=tk.X, pady=2)
        # ttk.Button(
        #     advanced_frame,
        #     text="Raman Group Analysis",
        #     command=self.open_raman_group_analysis,
        # ).pack(fill=tk.X, pady=2)
        # ttk.Button(
        #     advanced_frame,
        #     text="Mixed Mineral Analysis",
        #     command=self.analyze_mixed_minerals,
        # ).pack(fill=tk.X, pady=2)

    def create_search_tab(self, parent):
        """Create content for the search-match tab."""
        # Create notebook for search options
        search_notebook = ttk.Notebook(parent)
        search_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs for different search methods
        self.tab_basic_search = ttk.Frame(search_notebook)
        self.tab_advanced_search = ttk.Frame(search_notebook)

        # Add tabs to notebook
        search_notebook.add(self.tab_basic_search, text="Basic Search")
        search_notebook.add(self.tab_advanced_search, text="Advanced Search")

        # Create content for each search tab
        self.create_basic_search_tab(self.tab_basic_search)
        self.create_advanced_search_tab(self.tab_advanced_search)

        # Results display (common to both tabs)
        results_frame = ttk.LabelFrame(
            self.tab_search, text="Results Summary", padding=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.match_result_text = tk.Text(
            results_frame, height=15, width=30, wrap=tk.WORD
        )
        self.match_result_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Add scrollbar to results summary
        scrollbar = ttk.Scrollbar(
            results_frame, orient=tk.VERTICAL, command=self.match_result_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.match_result_text.config(yscrollcommand=scrollbar.set, state=tk.DISABLED)

    def update_hey_classification_database(self):
        """Update Hey Classification and additional metadata for all entries in the database."""
        if not self.raman.database:
            messagebox.showinfo("Database", "The database is empty.")
            return

        # Check if Hey Classification data is loaded
        if (
            not hasattr(self.raman, "hey_classification")
            or not self.raman.hey_classification
        ):
            # Try to load Hey Classification data if not already loaded
            hey_csv_path = "RRUFF_Export_with_Hey_Classification.csv"
            if os.path.exists(hey_csv_path):
                self.raman.hey_classification = self.raman.load_hey_classification(
                    hey_csv_path
                )

            # Check again if data was loaded
            if (
                not hasattr(self.raman, "hey_classification")
                or not self.raman.hey_classification
            ):
                messagebox.showerror(
                    "Error", "Hey Classification data could not be loaded."
                )
                return

        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Updating Hey Classification and Metadata")
        progress_window.geometry("500x350")

        # Create progress bar
        ttk.Label(
            progress_window,
            text="Updating Hey Classification and metadata for database entries...",
        ).pack(pady=10)
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
        log_scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=log_text.yview
        )
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

        # Track additional metadata fields for stats
        metadata_fields_updated = {
            "HEY CLASSIFICATION": 0,
            "STRUCTURAL GROUPNAME": 0,
            "CRYSTAL SYSTEM": 0,
            "SPACE GROUP": 0,
            "OLDEST KNOWN AGE (MA)": 0,
            "PARAGENETIC MODE": 0,
        }

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
            if "metadata" not in data or not data["metadata"]:
                # Try to create metadata if it doesn't exist
                if "metadata" not in data:
                    data["metadata"] = {}

                # Add a note to log
                update_progress(
                    i,
                    name,
                    f"Processing {name}",
                    f"Note: Created metadata for {name}",
                    "warning",
                )

            metadata = data["metadata"]

            # Check if Hey Classification already exists - we'll still add other metadata
            had_hey_class = False
            if "HEY CLASSIFICATION" in metadata and metadata["HEY CLASSIFICATION"]:
                had_hey_class = True
                already_had += 1
                update_progress(
                    i,
                    name,
                    f"Processing {name}",
                    f"Note: {name} already has Hey Classification '{metadata['HEY CLASSIFICATION']}'",
                    "info",
                )

            # Try to find mineral name using different strategies
            mineral_name = None

            # 1. Check if NAME field exists in metadata
            if "NAME" in metadata and metadata["NAME"]:
                mineral_name = metadata["NAME"]

            # 2. If no NAME in metadata, try to extract from the spectrum name (the database key)
            if not mineral_name:
                # Split by common separators and take the first part as potential mineral name
                potential_name = name.split("__")[0].split("_")[0].split("-")[0].strip()

                # Remove numbers and special characters at the end of the potential name
                import re

                cleaned_name = re.sub(r"[0-9_]+$", "", potential_name).strip()

                if cleaned_name:
                    mineral_name = cleaned_name
                    # Add the extracted name to metadata for future use
                    metadata["NAME"] = mineral_name
                    update_progress(
                        i,
                        name,
                        f"Processing {name}",
                        f"Extracted mineral name '{mineral_name}' from spectrum name",
                        "info",
                    )

            # If still no mineral name, skip this entry
            if not mineral_name:
                update_progress(
                    i,
                    name,
                    f"Processing {name}",
                    f"Skipped {name}: No mineral name in metadata and could not extract from name",
                    "warning",
                )
                continue

            # Try to match using the advanced matching function
            matched_name, matched_metadata = self.raman.match_mineral_name(mineral_name)

            if matched_metadata:
                # Track which fields were updated
                fields_updated = []

                # Update all available metadata fields
                for field, value in matched_metadata.items():
                    # Skip if the field already exists and has a value (except for HEY CLASSIFICATION if updating all)
                    if (
                        field in metadata
                        and metadata[field]
                        and (field != "HEY CLASSIFICATION" or had_hey_class)
                    ):
                        continue

                    # Update the field
                    metadata[field] = value
                    fields_updated.append(field)
                    metadata_fields_updated[field] = (
                        metadata_fields_updated.get(field, 0) + 1
                    )

                if fields_updated:
                    updated += 1
                    update_progress(
                        i,
                        name,
                        f"Processing {name}",
                        f"Updated {name}: Added fields {', '.join(fields_updated)} (matched with '{matched_name}')",
                        "success",
                    )
                else:
                    update_progress(
                        i,
                        name,
                        f"Processing {name}",
                        f"No new metadata fields to update for {name}",
                        "info",
                    )
            else:
                not_found += 1
                update_progress(
                    i,
                    name,
                    f"Processing {name}",
                    f"Could not find Hey Classification data for '{mineral_name}'",
                    "error",
                )

        # Final progress update
        update_progress(
            total,
            "",
            "Complete",
            f"Update complete! Updated: {updated}, Already had Hey Classification: {already_had}, Not found: {not_found}",
            "info",
        )

        # Show breakdown of fields updated
        log_text.insert(tk.END, "\nMetadata fields updated:\n", "info")
        for field, count in metadata_fields_updated.items():
            if count > 0:
                log_text.insert(tk.END, f"  - {field}: {count} entries\n", "info")

        # Save the database
        saved = self.raman.save_database()
        if saved:
            log_text.insert(
                tk.END,
                f"Database saved successfully to {self.raman.db_path}\n",
                "success",
            )
        else:
            log_text.insert(tk.END, "Warning: Could not save database\n", "error")
        log_text.see(tk.END)

        # Add close button
        ttk.Button(progress_window, text="Close", command=progress_window.destroy).pack(
            pady=10
        )

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
        if (
            not hasattr(self.raman, "hey_classification")
            or not self.raman.hey_classification
        ):
            # Try to load Hey Classification data if not already loaded
            hey_csv_path = "RRUFF_Export_with_Hey_Classification.csv"
            if os.path.exists(hey_csv_path):
                self.raman.hey_classification = self.raman.load_hey_classification(
                    hey_csv_path
                )

            # Check again if data was loaded
            if (
                not hasattr(self.raman, "hey_classification")
                or not self.raman.hey_classification
            ):
                messagebox.showerror(
                    "Error", "Hey Classification data could not be loaded."
                )
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
        ttk.Label(left_frame, text="Hey Classification Hierarchy").pack(
            anchor=tk.W, pady=(0, 5)
        )

        # Create treeview
        tree = ttk.Treeview(left_frame)
        tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Add scrollbar to tree
        tree_scrollbar = ttk.Scrollbar(
            left_frame, orient=tk.VERTICAL, command=tree.yview
        )
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=tree_scrollbar.set)

        # Create listbox for minerals
        ttk.Label(right_frame, text="Minerals in Selected Classification").pack(
            anchor=tk.W, pady=(0, 5)
        )

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

        list_scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=mineral_listbox.yview
        )
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        mineral_listbox.configure(yscrollcommand=list_scrollbar.set)

        # Details section
        detail_frame = ttk.LabelFrame(right_frame, text="Mineral Details", padding=5)
        detail_frame.pack(fill=tk.X, pady=(5, 0))

        detail_text = tk.Text(detail_frame, height=8, wrap=tk.WORD)
        detail_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        detail_scrollbar = ttk.Scrollbar(
            detail_frame, orient=tk.VERTICAL, command=detail_text.yview
        )
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

            # Extract classification parts (assuming format like "X. Silicate - TectoSilicate")
            parts = classification.split(" - ")
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
            main_id = tree.insert("", "end", text=main_class, values=(main_class, ""))

            # Add subclasses as child nodes
            subclasses = sorted(hierarchy[main_class]) if hierarchy[main_class] else []
            for subclass in subclasses:
                full_class = f"{main_class} - {subclass}"
                tree.insert(main_id, "end", text=subclass, values=(full_class, ""))

        # Function to update mineral list when a classification is selected
        def on_classification_select(event):
            selected_items = tree.selection()
            if not selected_items:
                return

            # Get selected classification
            selected_item = selected_items[0]
            values = tree.item(selected_item, "values")

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
                metadata = data.get("metadata", {})
                db_name = metadata.get("NAME", "")

                if db_name.lower() == mineral_lower:
                    in_database = True
                    spectra_count += 1

            if in_database:
                detail_text.insert(
                    tk.END, f"In Database: Yes ({spectra_count} spectra)\n"
                )
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
                values = tree.item(selected_items[0], "values")
                classification = values[0] if values else None

            update_mineral_list(classification, search_term)

        # Bind events
        tree.bind("<<TreeviewSelect>>", on_classification_select)
        mineral_listbox.bind("<<ListboxSelect>>", on_mineral_select)
        search_entry.bind("<Return>", on_search)

        # Add search button
        ttk.Button(search_frame, text="Search", command=on_search).pack(
            side=tk.LEFT, padx=(5, 0)
        )
        ttk.Button(
            search_frame,
            text="Clear",
            command=lambda: [search_var.set(""), on_search()],
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Button to add selected mineral to database
        def import_selected_mineral():
            selected_indices = mineral_listbox.curselection()
            if not selected_indices:
                messagebox.showinfo(
                    "Selection Required", "Please select a mineral first."
                )
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
            messagebox.showinfo(
                "Import Mineral",
                f"The mineral name '{mineral_name}' has been selected. \n\n"
                "Please import a spectrum file first, then click 'Add Current Spectrum' "
                "to add it to the database with this mineral name and Hey Classification.",
            )

        # Add button to bottom of right frame
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            button_frame,
            text="Import Selected Mineral",
            command=import_selected_mineral,
        ).pack(side=tk.RIGHT)

    # Add this button to the database_tab in create_database_tab method
    def create_database_tab(self):
        """Create content for the database operations tab."""
        # Add to database frame
        add_frame = ttk.LabelFrame(
            self.tab_database, text="Add to Database", padding=10
        )
        add_frame.pack(fill=tk.X, pady=5)

        ttk.Label(add_frame, text="Spectrum Name:").pack(anchor=tk.W)
        self.var_name = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.var_name).pack(fill=tk.X, pady=2)

        ttk.Button(
            add_frame, text="Add Current Spectrum", command=self.add_to_database
        ).pack(fill=tk.X, pady=2)

        # Database management frame
        manage_frame = ttk.LabelFrame(
            self.tab_database, text="Database Management", padding=10
        )
        manage_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            manage_frame, text="Batch Import Spectra", command=self.batch_import
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            manage_frame, text="View/Search Database", command=self.view_database
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            manage_frame,
            text="Update Hey Classification",
            command=self.update_hey_classification_database,
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            manage_frame,
            text="Refresh Database Info",
            command=self.update_database_stats,
        ).pack(fill=tk.X, pady=2)
        
        # Add the new button to launch the mineral database browser
        ttk.Button(
            manage_frame,
            text="View/Edit Mineral Character Info",
            command=self.launch_mineral_database_browser,
        ).pack(fill=tk.X, pady=2)

        # Database statistics frame
        stats_frame = ttk.LabelFrame(
            self.tab_database, text="Database Statistics", padding=10
        )
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create the db_stats_text widget - this must be initialized before update_database_stats is called
        self.db_stats_text = tk.Text(stats_frame, height=10, width=30, wrap=tk.WORD)
        self.db_stats_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            stats_frame, orient=tk.VERTICAL, command=self.db_stats_text.yview
        )
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
        try:
            if name is None:
                # Edit current spectrum metadata
                if self.raman.current_spectra is None:
                    messagebox.showerror("Error", "No spectrum loaded.")
                    return
                metadata = (
                    self.raman.metadata.copy()
                    if hasattr(self.raman, "metadata")
                    else {}
                )
                title = "Edit Current Spectrum Metadata"
            else:
                # Edit database entry metadata
                if name not in self.raman.database:
                    messagebox.showerror(
                        "Error", f"No spectrum named '{name}' in database."
                    )
                    return
                metadata = self.raman.database[name].get("metadata", {}).copy()
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
            scrollbar = ttk.Scrollbar(
                main_frame, orient="vertical", command=canvas.yview
            )

            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Keep track of entry widgets - store as instance variable for the dialog
            self.metadata_entries = {}

            # Define common metadata fields that should always be shown
            # This list is based on the field mappings in RamanSpectra.import_spectrum and other common fields
            common_fields = [
                "NAME",
                "HEY CLASSIFICATION",
                "RRUFFIDS",
                "IDEAL CHEMISTRY",
                "CHEMICAL FAMILY",
                "CRYSTAL SYSTEMS",
                "SPACE GROUPS",
                "LOCALITY",
                "DESCRIPTION",
                "URL",
                "PARAGENETIC MODES",
                "MINERAL FAMILY",
                "CHEMICAL FORMULA",
                "NOTES",
                "AGE",
            ]

            # Add Hey Classification dropdown at the top
            hey_frame = ttk.LabelFrame(
                scrollable_frame, text="Hey Classification", padding=5
            )
            hey_frame.pack(fill=tk.X, pady=5, padx=5)

            # Get all available Hey Classifications
            hey_classes = []
            try:
                if hasattr(self.raman, "hey_classification"):
                    # Safely extract Hey classifications, handling potential dict values
                    classifications = []
                    for val in self.raman.hey_classification.values():
                        if isinstance(val, dict) and "HEY CLASSIFICATION" in val:
                            classifications.append(val["HEY CLASSIFICATION"])
                        elif isinstance(val, str):
                            classifications.append(val)
                    hey_classes = sorted(list(set(filter(None, classifications))))
            except Exception as e:
                print(f"Error getting Hey classifications: {str(e)}")

            current_hey = metadata.get("HEY CLASSIFICATION", "")
            self.var_hey = tk.StringVar(value=current_hey)

            # Create combobox with Hey Classifications
            ttk.Label(hey_frame, text="Hey Classification:").pack(anchor=tk.W)
            hey_combo = ttk.Combobox(
                hey_frame, textvariable=self.var_hey, values=[""] + hey_classes
            )
            hey_combo.pack(fill=tk.X, pady=2)
            self.metadata_entries["HEY CLASSIFICATION"] = self.var_hey

            # Add mineral name field
            name_frame = ttk.LabelFrame(
                scrollable_frame, text="Mineral Name", padding=5
            )
            name_frame.pack(fill=tk.X, pady=5, padx=5)

            current_name = metadata.get("NAME", "")
            self.var_name = tk.StringVar(value=current_name)

            ttk.Label(name_frame, text="Mineral Name:").pack(anchor=tk.W)
            name_entry = ttk.Entry(name_frame, textvariable=self.var_name)
            name_entry.pack(fill=tk.X, pady=2)
            self.metadata_entries["NAME"] = self.var_name

            # Function to look up Hey Classification
            def lookup_hey_classification():
                mineral_name = self.var_name.get().strip()
                if not mineral_name:
                    messagebox.showinfo("Lookup", "Please enter a mineral name first.")
                    return

                try:
                    # Try to find Hey Classification
                    matched_name, classification = self.raman.match_mineral_name(
                        mineral_name
                    )

                    if classification:
                        # Handle both string and dictionary classification results
                        if (
                            isinstance(classification, dict)
                            and "HEY CLASSIFICATION" in classification
                        ):
                            self.var_hey.set(classification["HEY CLASSIFICATION"])
                            messagebox.showinfo(
                                "Lookup Result",
                                f"Found Hey Classification: {classification['HEY CLASSIFICATION']}\nMatched with: {matched_name}",
                            )
                        elif isinstance(classification, str):
                            self.var_hey.set(classification)
                            messagebox.showinfo(
                                "Lookup Result",
                                f"Found Hey Classification: {classification}\nMatched with: {matched_name}",
                            )
                    else:
                        messagebox.showinfo(
                            "Lookup Result",
                            "No matching Hey Classification found for this mineral name.",
                        )
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"Error looking up Hey Classification: {str(e)}"
                    )

            ttk.Button(
                name_frame,
                text="Lookup Hey Classification",
                command=lookup_hey_classification,
            ).pack(pady=5)

            # Common metadata fields frame
            common_metadata_frame = ttk.LabelFrame(
                scrollable_frame, text="Common Metadata", padding=5
            )
            common_metadata_frame.pack(fill=tk.X, pady=5, padx=5, expand=True)

            # Add all common fields first (excluding NAME and HEY CLASSIFICATION which are already added)
            for field in common_fields:
                if field not in ["NAME", "HEY CLASSIFICATION"]:
                    var = tk.StringVar(value=metadata.get(field, ""))
                    ttk.Label(common_metadata_frame, text=f"{field}:").pack(anchor=tk.W)
                    ttk.Entry(common_metadata_frame, textvariable=var).pack(
                        fill=tk.X, pady=2
                    )
                    self.metadata_entries[field] = var

            # Other metadata fields that exist in the current metadata but aren't in common_fields
            other_metadata_frame = ttk.LabelFrame(
                scrollable_frame, text="Other Metadata", padding=5
            )
            other_metadata_frame.pack(fill=tk.X, pady=5, padx=5, expand=True)
            self.other_metadata_frame = other_metadata_frame

            # Add remaining fields from current metadata
            for field, value in metadata.items():
                if field not in self.metadata_entries and field not in [
                    "NAME",
                    "HEY CLASSIFICATION",
                ]:
                    var = tk.StringVar(value=value)
                    ttk.Label(other_metadata_frame, text=f"{field}:").pack(anchor=tk.W)
                    ttk.Entry(other_metadata_frame, textvariable=var).pack(
                        fill=tk.X, pady=2
                    )
                    self.metadata_entries[field] = var

            # Add new field section
            new_field_frame = ttk.LabelFrame(
                scrollable_frame, text="Add New Field", padding=5
            )
            new_field_frame.pack(fill=tk.X, pady=5, padx=5)

            self.var_new_field = tk.StringVar()
            self.var_new_value = tk.StringVar()

            ttk.Label(new_field_frame, text="Field Name:").pack(anchor=tk.W)
            ttk.Entry(new_field_frame, textvariable=self.var_new_field).pack(
                fill=tk.X, pady=2
            )

            ttk.Label(new_field_frame, text="Field Value:").pack(anchor=tk.W)
            ttk.Entry(new_field_frame, textvariable=self.var_new_value).pack(
                fill=tk.X, pady=2
            )

            # Function to add new field to the form
            def add_new_field():
                field_name = self.var_new_field.get().strip().upper()
                field_value = self.var_new_value.get().strip()

                if not field_name or not field_value:
                    messagebox.showinfo(
                        "Add Field", "Please enter both field name and value."
                    )
                    return

                if field_name in self.metadata_entries:
                    if messagebox.askyesno(
                        "Field Exists",
                        f"Field '{field_name}' already exists. Update its value?",
                    ):
                        self.metadata_entries[field_name].set(field_value)
                        self.var_new_field.set("")
                        self.var_new_value.set("")
                    return

                # Add new field to the form
                var = tk.StringVar(value=field_value)
                ttk.Label(self.other_metadata_frame, text=f"{field_name}:").pack(
                    anchor=tk.W
                )
                ttk.Entry(self.other_metadata_frame, textvariable=var).pack(
                    fill=tk.X, pady=2
                )
                self.metadata_entries[field_name] = var

                # Clear entry fields
                self.var_new_field.set("")
                self.var_new_value.set("")

            ttk.Button(new_field_frame, text="Add Field", command=add_new_field).pack(
                pady=5
            )

            # Button frame for save/cancel
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill=tk.X, pady=10)

            def save_metadata():
                try:
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
                        messagebox.showinfo(
                            "Metadata", "Current spectrum metadata updated."
                        )
                    else:
                        # Update database entry metadata
                        self.raman.database[name]["metadata"] = updated_metadata
                        self.raman.save_database()
                        messagebox.showinfo(
                            "Metadata",
                            f"Metadata for '{name}' updated and saved to database.",
                        )

                    # Update comboboxes for filters
                    self.update_metadata_filter_options()
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Error saving metadata: {str(e)}")

            ttk.Button(button_frame, text="Save Metadata", command=save_metadata).pack(
                side=tk.RIGHT, padx=5
            )
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
                side=tk.RIGHT, padx=5
            )

        except Exception as e:
            messagebox.showerror(
                "Error", f"An error occurred opening the metadata editor: {str(e)}"
            )

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
            if messagebox.askyesno(
                "Field Exists",
                f"Field '{field_name}' already exists. Update its value?",
            ):
                self.metadata_entries[field_name].set(field_value)
                self.var_new_field.set("")
                self.var_new_value.set("")
            return

        # Add new field to the form
        self.metadata_entries[field_name] = tk.StringVar(value=field_value)
        ttk.Label(self.other_metadata_frame, text=f"{field_name}:").pack(anchor=tk.W)
        ttk.Entry(
            self.other_metadata_frame, textvariable=self.metadata_entries[field_name]
        ).pack(fill=tk.X, pady=2)

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
        ttk.Entry(params_frame, textvariable=self.var_corr_threshold).pack(
            fill=tk.X, pady=2
        )

        # Matching algorithm selection
        ttk.Label(params_frame, text="Matching Algorithm:").pack(anchor=tk.W)
        self.var_algorithm = tk.StringVar(value="combined")
        algorithms = [
            ("Combined (Recommended)", "combined"),
            ("Correlation", "correlation"),
            ("Peak Matching", "peak"),
        ]
        if SKLEARN_AVAILABLE:
            algorithms.append(("Machine Learning (DTW)", "ml"))

        for text, value in algorithms:
            ttk.Radiobutton(
                params_frame, text=text, variable=self.var_algorithm, value=value
            ).pack(anchor=tk.W, pady=1)

        # Search button
        ttk.Button(params_frame, text="Search Match", command=self.search_match).pack(
            fill=tk.X, pady=5
        )

        # Display options frame (for controlling the main plot during comparison)
        visualization_frame = ttk.LabelFrame(
            parent, text="Visualization Options", padding=10
        )
        visualization_frame.pack(fill=tk.X, pady=5)

        self.var_show_diff = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            visualization_frame,
            text="Show Difference Plot",
            variable=self.var_show_diff,
        ).pack(anchor=tk.W, pady=2)

        self.var_normalize = tk.BooleanVar(
            value=True
        )  # Usually good to keep normalization for comparison
        ttk.Checkbutton(
            visualization_frame,
            text="Normalize Spectra (for Plot)",
            variable=self.var_normalize,
        ).pack(anchor=tk.W, pady=2)

        self.var_highlight_peaks = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            visualization_frame,
            text="Highlight Matching Peaks",
            variable=self.var_highlight_peaks,
        ).pack(anchor=tk.W, pady=2)

    def create_advanced_search_tab(self, parent):
        """Create advanced search options interface."""
        # Peak-based search frame
        peak_frame = ttk.LabelFrame(parent, text="Peak-Based Search Filter", padding=10)
        peak_frame.pack(fill=tk.X, pady=5)

        ttk.Label(
            peak_frame, text="Search by specific peak positions (strict matching):"
        ).pack(anchor=tk.W)
        self.var_peak_positions = tk.StringVar()
        ttk.Entry(peak_frame, textvariable=self.var_peak_positions).pack(
            fill=tk.X, pady=2
        )

        # Fix for the font parameter issue - Create the label first, then pack it separately
        hint_label = ttk.Label(
            peak_frame,
            text="Format: comma-separated values (e.g., 1050, 1350). Results must contain ALL specified peaks.",
            font=("TkDefaultFont", 8),  # Apply font during creation
        )
        hint_label.pack(anchor=tk.W)  # Pack separately

        ttk.Label(peak_frame, text="Peak Tolerance (cm⁻¹):").pack(anchor=tk.W)
        self.var_peak_tolerance = tk.StringVar(value="10")
        ttk.Entry(peak_frame, textvariable=self.var_peak_tolerance).pack(
            fill=tk.X, pady=2
        )

        tolerance_hint = ttk.Label(
            peak_frame,
            text="Tolerance defines how close a database peak must be to match a specified peak position.",
            font=("TkDefaultFont", 8),
        )
        tolerance_hint.pack(anchor=tk.W)

        # Category filtering frame
        filter_frame = ttk.LabelFrame(
            parent, text="Metadata Filter Options", padding=10
        )
        filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_frame, text="Filter by Chemical Family:").pack(anchor=tk.W)
        self.var_chemical_family = tk.StringVar()
        self.chemical_family_combo = ttk.Combobox(
            filter_frame, textvariable=self.var_chemical_family, state="readonly"
        )
        self.chemical_family_combo.pack(fill=tk.X, pady=2)

        # Add Hey Classification filter
        ttk.Label(filter_frame, text="Filter by Hey Classification:").pack(anchor=tk.W)
        self.var_hey_classification = tk.StringVar()
        self.hey_classification_combo = ttk.Combobox(
            filter_frame, textvariable=self.var_hey_classification, state="readonly"
        )
        self.hey_classification_combo.pack(fill=tk.X, pady=2)

        # Add Chemistry Elements filter
        elements_frame = ttk.LabelFrame(
            filter_frame, text="Filter by Chemistry Elements", padding=5
        )
        elements_frame.pack(fill=tk.X, pady=5)

        # Only these elements
        ttk.Label(elements_frame, text="Only these elements (comma-separated):").pack(
            anchor=tk.W
        )
        self.var_only_elements = tk.StringVar()
        ttk.Entry(elements_frame, textvariable=self.var_only_elements).pack(
            fill=tk.X, pady=2
        )

        # Required elements
        ttk.Label(elements_frame, text="Required elements (comma-separated):").pack(
            anchor=tk.W
        )
        self.var_required_elements = tk.StringVar()
        ttk.Entry(elements_frame, textvariable=self.var_required_elements).pack(
            fill=tk.X, pady=2
        )

        # Exclude elements
        ttk.Label(elements_frame, text="Exclude elements (comma-separated):").pack(
            anchor=tk.W
        )
        self.var_exclude_elements = tk.StringVar()
        ttk.Entry(elements_frame, textvariable=self.var_exclude_elements).pack(
            fill=tk.X, pady=2
        )

        # Update both dropdown options
        self.update_metadata_filter_options()

        # Scoring threshold
        threshold_frame = ttk.Frame(parent)
        threshold_frame.pack(fill=tk.X, pady=5)

        ttk.Label(threshold_frame, text="Correlation Threshold:").pack(side=tk.LEFT)
        self.var_adv_corr_threshold = tk.StringVar(value="0.7")
        ttk.Entry(
            threshold_frame, textvariable=self.var_adv_corr_threshold, width=8
        ).pack(side=tk.LEFT, padx=5)

        threshold_hint = ttk.Label(
            threshold_frame,
            text="Used as secondary score after peak matching",
            font=("TkDefaultFont", 8),
        )
        threshold_hint.pack(side=tk.LEFT, padx=5)

        # Advanced search button
        ttk.Button(
            parent, text="Advanced Search", command=self.advanced_search_match
        ).pack(fill=tk.X, pady=5)

        # Add Mixed Mineral Analysis button
        ttk.Button(
            parent, text="Mixed Mineral Analysis", command=self.analyze_mixed_minerals
        ).pack(fill=tk.X, pady=5)

    def update_metadata_filter_options(self):
        """Update chemical family and Hey Classification combobox options from database metadata."""
        # Chemical Family options - now using the dedicated method to ensure consistency
        if hasattr(self, "chemical_family_combo"):
            self.update_chemical_family_options()

        # Hey Classification options
        hey_classes = set()

        if (
            hasattr(self, "raman")
            and hasattr(self.raman, "database")
            and self.raman.database
        ):
            for data in self.raman.database.values():
                if "metadata" in data and data.get(
                    "metadata"
                ):  # Check if metadata exists and is not empty
                    # Hey Classification
                    hey_class = data["metadata"].get("HEY CLASSIFICATION")
                    if hey_class and isinstance(
                        hey_class, str
                    ):  # Check if Hey Classification exists and is not empty/None
                        hey_classes.add(hey_class.strip())

        # Update Hey Classification combobox if it exists
        if hasattr(self, "hey_classification_combo"):
            sorted_hey_classes = sorted(list(hey_classes))
            self.hey_classification_combo["values"] = [
                ""
            ] + sorted_hey_classes  # Add empty option to disable filter

            # Keep current selection if valid, otherwise reset
            current_hey = self.var_hey_classification.get()
            if current_hey and current_hey not in sorted_hey_classes:
                self.var_hey_classification.set(
                    ""
                )  # Reset if current value not in new list

    def update_chemical_family_options(self):
        """Update the chemical family dropdown options from database metadata."""
        # Check if the chemical_family_combo attribute exists
        if not hasattr(self, "chemical_family_combo"):
            print("Chemical family combo not found - returning early")
            return

        # Get unique chemical families from database - ONLY those explicitly in metadata
        families = set()

        if hasattr(self, "raman") and hasattr(self, "raman.database"):
            print(f"Database has {len(self.raman.database)} entries")
            for name, data in self.raman.database.items():
                if "metadata" in data and data["metadata"]:
                    family = data["metadata"].get("CHEMICAL FAMILY")
                    if family and isinstance(
                        family, str
                    ):  # Only add non-empty, string values
                        clean_family = family.strip()
                        families.add(clean_family)
                        print(f"Entry '{name}' has CHEMICAL FAMILY: '{clean_family}'")

                    # We're no longer extracting families from Hey Classification
                    # This ensures only explicit Chemical Family values are shown in the filter

        # Sort families for display
        sorted_families = sorted(list(families))

        # Print debug info
        print(
            f"Found {len(sorted_families)} unique chemical families explicitly in database metadata"
        )
        if sorted_families:
            print(f"Chemical families in dropdown: {sorted_families}")

        # Update the combobox values
        actual_dropdown_values = [""] + sorted_families
        self.chemical_family_combo["values"] = actual_dropdown_values

        # Verify the values were set correctly
        try:
            actual_values = self.chemical_family_combo["values"]
            print(f"Values actually in dropdown after setting: {actual_values}")

            # Ensure compatibility with Tkinter handling of values
            if isinstance(actual_values, str):
                print("Warning: Values were converted to string by Tkinter")
                # Attempt to fix by reapplying as a tuple
                self.chemical_family_combo["values"] = tuple(actual_dropdown_values)
        except Exception as e:
            print(f"Error verifying dropdown values: {str(e)}")

        # Keep the current selection if it exists in the new values, otherwise reset
        current_value = self.var_chemical_family.get()
        if current_value and current_value not in sorted_families:
            print(f"Resetting selection because '{current_value}' not in new values")
            self.var_chemical_family.set("")

    def update_database_stats(self):
        """Update the database statistics display."""
        # Check if db_stats_text exists before trying to use it
        if not hasattr(self, "db_stats_text"):
            # This method might have been called before UI components were created
            # Just return without updating stats
            return

        self.db_stats_text.config(state=tk.NORMAL)
        self.db_stats_text.delete(1.0, tk.END)

        if hasattr(self, "raman") and hasattr(self.raman, "database"):
            db_size = len(self.raman.database)
            self.db_stats_text.insert(tk.END, f"Total entries: {db_size}\n\n")

            if db_size > 0:
                # Count entries with metadata
                with_metadata = sum(
                    1
                    for data in self.raman.database.values()
                    if "metadata" in data and data.get("metadata")
                )
                self.db_stats_text.insert(
                    tk.END, f"Entries with metadata: {with_metadata}\n"
                )

                # Count entries with peaks
                with_peaks = sum(
                    1
                    for data in self.raman.database.values()
                    if "peaks" in data and data.get("peaks") is not None
                )
                self.db_stats_text.insert(
                    tk.END, f"Entries with peak data: {with_peaks}\n\n"
                )

                # List a few example entries
                self.db_stats_text.insert(tk.END, "Sample entries:\n")
                for i, name in enumerate(list(self.raman.database.keys())[:5]):
                    self.db_stats_text.insert(tk.END, f"- {name}\n")

                if db_size > 5:
                    self.db_stats_text.insert(tk.END, f"... and {db_size-5} more\n")
            else:
                self.db_stats_text.insert(tk.END, "Database is empty.\n")
        else:
            self.db_stats_text.insert(
                tk.END, "Database not loaded or object missing.\n"
            )

        self.db_stats_text.config(state=tk.DISABLED)

        # Update chemical family options if the method exists
        if hasattr(self, "update_chemical_family_options"):
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
                    ("All files", "*.*"),
                ],
            )

            if not file_path:  # User cancelled the dialog
                return

            # Import the spectrum using the RamanSpectra class
            self.raman.import_spectrum(file_path)

            # Update the plot and metadata display
            self.update_plot()
            self.update_metadata_display()
             
            # Update status in title bar
            filename = os.path.basename(file_path)
            self.root.title(f"ClaritySpectra: Raman Spectrum Analysis - {filename}")

        except Exception as e:
            # Show error message if import fails
            messagebox.showerror("Error", f"Failed to import spectrum: {str(e)}")

    def update_metadata_display(self, metadata_dict=None):
        """Update the metadata display with provided or current spectrum metadata."""
        self.metadata_text.config(state=tk.NORMAL)
        self.metadata_text.delete(1.0, tk.END)

        meta_to_display = (
            metadata_dict if metadata_dict is not None else self.raman.metadata
        )

        if meta_to_display:
            # Order of important metadata fields
            important_fields = [
                "NAME",
                "RRUFF ID",
                "IDEAL CHEMISTRY",
                "CHEMICAL FAMILY",
                "LOCALITY",
                "DESCRIPTION",
                "URL",
            ]

            # Display important fields first in order
            displayed_keys = set()
            for field in important_fields:
                if field in meta_to_display:
                    self.metadata_text.insert(
                        tk.END, f"{field}: {meta_to_display[field]}\n"
                    )
                    displayed_keys.add(field)

            # Add peak fitting information if available
            if hasattr(self.raman, "fitted_peaks") and self.raman.fitted_peaks:
                self.metadata_text.insert(tk.END, "\n=== Peak Fitting Results ===\n")

                # Add fitting model and quality info
                if self.raman.peak_fit_result:
                    model = self.raman.peak_fit_result.get("model", "Unknown")
                    r_squared = self.raman.peak_fit_result.get("r_squared", 0)
                    self.metadata_text.insert(tk.END, f"Model: {model}\n")
                    self.metadata_text.insert(tk.END, f"R-squared: {r_squared:.4f}\n")

                # Add peak information
                self.metadata_text.insert(
                    tk.END, f"Total peaks: {len(self.raman.fitted_peaks)}\n"
                )
                self.metadata_text.insert(tk.END, "Peak positions (cm⁻¹):\n")

                # Display each peak
                for i, peak in enumerate(self.raman.fitted_peaks):
                    position = peak["position"]
                    intensity = peak["intensity"]
                    width = peak["width"]
                    self.metadata_text.insert(
                        tk.END,
                        f"  Peak {i+1}: {position:.2f} cm⁻¹ (I={intensity:.2f}, W={width:.2f})\n",
                    )

                # Add this section to displayed keys to prevent duplication
                displayed_keys.add("PEAK_FIT")

            # Display peak fit section from metadata if present
            elif "PEAK_FIT" in meta_to_display:
                self.metadata_text.insert(tk.END, "\n=== Peak Fitting Results ===\n")
                peak_fit_data = meta_to_display["PEAK_FIT"]

                for key, value in peak_fit_data.items():
                    self.metadata_text.insert(tk.END, f"{key}: {value}\n")

                displayed_keys.add("PEAK_FIT")

            # Display other fields
            for field, value in meta_to_display.items():
                if field not in displayed_keys:
                    self.metadata_text.insert(tk.END, f"{field}: {value}\n")
        else:
            # If specific metadata wasn't passed, check current Raman object
            if (
                metadata_dict is None
                and hasattr(self.raman, "metadata")
                and self.raman.metadata
            ):
                self.metadata_text.insert(
                    tk.END, "No metadata available for current spectrum."
                )
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
                    ("All Files", "*.*"),
                ],
            )

            if not file_path:
                return  # User cancelled

            # Try to save the spectrum
            success = self.raman.save_spectrum(file_path)

            if not success:
                messagebox.showerror("Error", "Failed to save spectrum.")
                return

            # Update title bar with success message instead of showing a dialog
            filename = os.path.basename(file_path)
            self.root.title(
                f"ClaritySpectra: Raman Spectrum Analysis - {filename} saved"
            )

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
                messagebox.showerror(
                    "Error", "Invalid baseline parameters. Please enter numeric values."
                )
                return

            # Subtract background
            corrected, baseline = self.raman.subtract_background(lam=lam, p=p)

            # Ensure background display is off by default
            self.var_show_background.set(False)

            # Update plot without showing background, respecting current peak display setting
            self.update_plot(
                include_background=False, include_peaks=self.var_show_peaks.get()
            )

            # Update status in title bar instead of showing a dialog
            current_title = self.root.title()
            if " - Background Subtracted" not in current_title:
                self.root.title(f"{current_title} - Background Subtracted")

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
            spectrum_to_use = (
                self.raman.processed_spectra
                if self.raman.processed_spectra is not None
                else self.raman.current_spectra
            )
            max_intensity = np.max(spectrum_to_use)

            try:
                # Convert height to appropriate value
                if height_str == "Auto":
                    height = None  # Let RamanSpectra class use default 5%
                elif height_str.endswith("%"):
                    # Handle percentage input
                    percentage = float(height_str.rstrip("%")) / 100.0
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
                messagebox.showerror(
                    "Error",
                    "Invalid peak parameters. Please enter numeric values or 'Auto'.",
                )
                return

            # Clear previous peaks
            self.raman.peaks = None

            # Find peaks using the RamanSpectra method
            peaks = self.raman.find_spectrum_peaks(
                height=height, distance=distance, prominence=prominence
            )

            # Update plot with new peaks
            self.var_show_peaks.set(True)  # Turn on peak display
            self.update_plot(
                include_background=self.var_show_background.get(), include_peaks=True
            )

            # Update title bar with peak count instead of showing dialog
            current_title = self.root.title()
            base_title = current_title.split(" - ")[
                0
            ]  # Get the base title without additional info
            self.root.title(f"{base_title} - {len(peaks['indices'])} Peaks Detected")

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

        try:  # Wrap the operation in try/except
            # Check if name already exists in database
            if name in self.raman.database:
                overwrite = messagebox.askyesno(
                    "Confirmation",
                    f"'{name}' already exists in the database. Overwrite?",
                )
                if not overwrite:
                    return

            # Check if CHEMICAL FAMILY is missing but HEY CLASSIFICATION exists
            if hasattr(self.raman, "metadata"):
                # Ensure metadata is a dictionary
                if self.raman.metadata is None:
                    self.raman.metadata = {}

                # Try to extract CHEMICAL FAMILY from HEY CLASSIFICATION if available
                if (
                    "CHEMICAL FAMILY" not in self.raman.metadata
                    or not self.raman.metadata["CHEMICAL FAMILY"]
                ) and "HEY CLASSIFICATION" in self.raman.metadata:
                    hey_class = self.raman.metadata.get("HEY CLASSIFICATION")
                    if hey_class and isinstance(hey_class, str) and " - " in hey_class:
                        # Extract chemical family from Hey Classification (format is "Family - Group")
                        family = hey_class.split(" - ")[0].strip()
                        if family:
                            self.raman.metadata["CHEMICAL FAMILY"] = family
                            print(
                                f"Added CHEMICAL FAMILY '{family}' from Hey Classification"
                            )

            # Add to database (metadata is handled within RamanSpectra class)
            success = self.raman.add_to_database(
                name
            )  # metadata is already in self.raman.metadata

            if success:
                # Update title bar with success message instead of showing dialog
                self.root.title(
                    f"ClaritySpectra: Raman Spectrum Analysis - '{name}' added to database"
                )

                # Update database statistics if tab exists
                if hasattr(self, "db_stats_text"):
                    self.update_database_stats()

                # Update metadata filter options to include any new chemical families
                self.update_metadata_filter_options()
            else:
                messagebox.showerror("Error", f"Failed to add '{name}' to database.")

        except Exception as e:
            messagebox.showerror(
                "Database Error", f"Error adding spectrum to database: {str(e)}"
            )

    def batch_import(self):
        """Batch import spectra to the database with threaded processing."""
        import threading
        import queue

        directory = filedialog.askdirectory(
            title="Select Directory with Spectrum Files"
        )
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
        log_scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=log_text.yview
        )
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
            files = [
                f
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
            ]
            if not files:
                messagebox.showinfo(
                    "Batch Import", "No files found in the selected directory."
                )
                progress_window.destroy()
                return
        except Exception as e:
            messagebox.showerror(
                "Batch Import Error", f"Error reading directory: {str(e)}"
            )
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
                update_queue.put(
                    ("progress", i, len(files), filename, current_status_msg)
                )

                try:
                    # Import spectrum using the temporary instance
                    _, _ = temp_raman.import_spectrum(filepath)

                    # Find peaks before adding
                    temp_raman.find_spectrum_peaks()

                    # Generate name from metadata or filename
                    name = ""
                    if (
                        "RRUFFID" in temp_raman.metadata
                        and temp_raman.metadata["RRUFFID"]
                    ):
                        name = temp_raman.metadata["RRUFFID"]
                    elif "NAME" in temp_raman.metadata and temp_raman.metadata["NAME"]:
                        name = temp_raman.metadata["NAME"]

                    if not name:  # Fallback to filename if no suitable metadata
                        name, _ = os.path.splitext(filename)
                    name = name.strip()

                    # Check if name already exists in the main database
                    if name in self.raman.database:
                        log_message = f"Skipped: '{name}' (from {filename}) already exists in database."
                        update_queue.put(("log", log_message, "info"))
                        skipped += 1
                        continue

                    # Ensure Hey Classification is added to metadata if possible
                    if (
                        "NAME" in temp_raman.metadata
                        and "HEY CLASSIFICATION" not in temp_raman.metadata
                    ):
                        mineral_name = temp_raman.metadata["NAME"]
                        hey_class = temp_raman.get_hey_classification(mineral_name)
                        if hey_class:
                            temp_raman.metadata["HEY CLASSIFICATION"] = hey_class
                            update_queue.put(
                                (
                                    "log",
                                    f"Added Hey Classification '{hey_class}' for {mineral_name}",
                                    "success",
                                )
                            )

                    # Safely copy necessary data to main database
                    # We'll store the data temporarily and let the main thread do the actual database update
                    spectrum_data = {
                        "name": name,
                        "wavenumbers": temp_raman.current_wavenumbers.copy(),
                        "intensities": temp_raman.current_spectra.copy(),
                        "processed": (
                            temp_raman.processed_spectra.copy()
                            if temp_raman.processed_spectra is not None
                            else None
                        ),
                        "peaks": (
                            temp_raman.peaks.copy()
                            if temp_raman.peaks is not None
                            else None
                        ),
                        "metadata": temp_raman.metadata.copy(),
                    }

                    update_queue.put(("add_to_db", spectrum_data))

                    # Log success
                    log_message = f"Imported '{name}' (from {filename})."
                    update_queue.put(("log", log_message, "success"))
                    imported += 1

                except Exception as e:
                    errors += 1
                    log_message = f"Error importing/processing {filename}: {str(e)}"
                    update_queue.put(("log", log_message, "error"))

            # Final status update
            update_queue.put(("complete", imported, skipped, errors))

        # Function to check queue and update UI
        def check_queue():
            try:
                # Process all available updates in the queue
                while True:
                    item = update_queue.get_nowait()
                    message_type = item[0]

                    if message_type == "progress":
                        _, current, total, filename, status_msg = item
                        progress["value"] = current
                        percentage = int((current / total) * 100) if total > 0 else 0
                        status_var.set(
                            f"{status_msg} - {percentage}% complete ({current}/{total})"
                        )

                    elif message_type == "log":
                        _, message, tag = item
                        log_text.insert(tk.END, message + "\n", tag)
                        log_text.see(tk.END)  # Auto-scroll to the end

                    elif message_type == "add_to_db":
                        _, spectrum_data = item
                        # Update the main database in the UI thread to avoid threading issues
                        name = spectrum_data["name"]

                        # Copy data to the main Raman object
                        self.raman.current_wavenumbers = spectrum_data["wavenumbers"]
                        self.raman.current_spectra = spectrum_data["intensities"]
                        self.raman.processed_spectra = spectrum_data["processed"]
                        self.raman.peaks = spectrum_data["peaks"]
                        self.raman.metadata = spectrum_data["metadata"]

                        # Add to database
                        self.raman.add_to_database(name)

                    elif message_type == "complete":
                        _, imported, skipped, errors = item
                        # Final status update
                        final_status = f"Complete! Imported: {imported}, Skipped: {skipped}, Errors: {errors}"
                        status_var.set(final_status)
                        log_text.insert(tk.END, f"\n--- {final_status} ---\n", "info")

                        # Save the database
                        save_result = self.raman.save_database()
                        if save_result:
                            log_text.insert(
                                tk.END,
                                f"Database saved successfully to {self.raman.db_path}\n",
                                "success",
                            )
                        else:
                            log_text.insert(
                                tk.END, "Warning: Could not save database\n", "error"
                            )

                        log_text.see(tk.END)

                        # Update main GUI database statistics
                        self.update_database_stats()

                        # Add close button
                        ttk.Button(
                            progress_window,
                            text="Close",
                            command=progress_window.destroy,
                        ).pack(pady=10)

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

        cancel_button = ttk.Button(
            progress_window, text="Cancel", command=cancel_processing
        )
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

    def open_peak_fitting(self):
        """Open the peak fitting window."""
        # Check if spectra data is loaded
        if self.raman.current_wavenumbers is None or self.raman.current_spectra is None:
            messagebox.showinfo("No Data", "Please load spectrum data first.")
            return

        try:
            # Check if processed spectra exists, otherwise use current spectra
            spectra_to_use = (
                self.raman.processed_spectra
                if self.raman.processed_spectra is not None
                else self.raman.current_spectra
            )

            # Open the peak fitting window
            PeakFittingWindow(
                self.root, self.raman, self.raman.current_wavenumbers, spectra_to_use
            )

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to open peak fitting window: {str(e)}"
            )

    def view_database(self):
        """View the database contents in a new window."""
        if not self.raman.database:
            messagebox.showinfo("Database", "The database is empty.")
            return

        # Create a new window
        db_window = tk.Toplevel(self.root)
        db_window.title("Raman Spectra Database")
        db_window.geometry(
            "1300x800"
        )  # Slightly increased width for paragenetic modes column

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

        # Treeview Columns - Added paragenetic modes column
        columns = (
            "#0",
            "name",
            "points",
            "mineral_name",
            "rruff_id",
            "ima_number",
            "hey_class",
            "chemical_family",
            "ideal_chemistry",
            "crystal_system",
            "space_groups",
            "locality",
            "paragenetic_modes",
            "chemistry_elements",
            "description",
        )
        tree = ttk.Treeview(list_frame, columns=columns[1:], show="headings", height=15)

        # Define headings
        tree.heading("name", text="Name/ID")
        tree.heading("points", text="Data Points")
        tree.heading("mineral_name", text="Mineral Name")
        tree.heading("rruff_id", text="RRUFF ID")
        tree.heading("ima_number", text="IMA Number")
        tree.heading("hey_class", text="Hey Classification")
        tree.heading("chemical_family", text="Chemical Family")
        tree.heading("ideal_chemistry", text="Ideal Chemistry")
        tree.heading("crystal_system", text="Crystal System")
        tree.heading("space_groups", text="Space Groups")
        tree.heading("locality", text="Locality")
        tree.heading("paragenetic_modes", text="Paragenetic Modes")
        tree.heading("chemistry_elements", text="Chemistry Elements")
        tree.heading("description", text="Description")

        # Configure column widths
        tree.column("name", width=150, anchor=tk.W)
        tree.column("points", width=80, anchor=tk.CENTER)
        tree.column("mineral_name", width=150, anchor=tk.W)
        tree.column("rruff_id", width=100, anchor=tk.W)
        tree.column("ima_number", width=100, anchor=tk.W)
        tree.column("hey_class", width=150, anchor=tk.W)
        tree.column("chemical_family", width=150, anchor=tk.W)
        tree.column("ideal_chemistry", width=150, anchor=tk.W)
        tree.column("crystal_system", width=120, anchor=tk.W)
        tree.column("space_groups", width=120, anchor=tk.W)
        tree.column("locality", width=150, anchor=tk.W)
        tree.column("paragenetic_modes", width=150, anchor=tk.W)
        tree.column("chemistry_elements", width=150, anchor=tk.W)
        tree.column("description", width=200, anchor=tk.W)

        # Add scrollbars
        yscrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        xscrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=yscrollbar.set, xscrollcommand=xscrollbar.set)

        # Pack scrollbars and tree
        yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Populate and Filter Logic ---
        db_items = list(self.raman.database.items())

        def populate_treeview(search_term=""):
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)

            search_lower = search_term.lower() if search_term else ""

            # Filter and insert entries
            for name, data in db_items:
                metadata = data.get("metadata", {})

                # Extract all metadata fields using exact keys
                mineral_name = metadata.get("NAME", "N/A")
                rruff_id = metadata.get("RRUFF ID", "N/A")
                ima_number = metadata.get("IMA NUMBER", "N/A")
                hey_class = metadata.get("HEY CLASSIFICATION", "N/A")
                chemical_family = metadata.get("CHEMICAL FAMILY", "N/A")
                ideal_chemistry = metadata.get("IDEAL CHEMISTRY", "N/A")
                crystal_system = metadata.get("CRYSTAL SYSTEM", "N/A")
                space_groups = metadata.get("SPACE GROUPS", "N/A")
                locality = metadata.get("LOCALITY", "N/A")
                paragenetic_modes = metadata.get("PARAGENETIC MODES", "N/A")
                chemistry_elements = metadata.get("CHEMISTRY ELEMENTS", "N/A")
                description = metadata.get("DESCRIPTION", "N/A")

                points = len(data.get("wavenumbers", []))
                display_name = name

                # Check if search term matches any field
                matches_search = True
                if search_lower:
                    matches_search = (
                        search_lower in display_name.lower()
                        or search_lower in mineral_name.lower()
                        or search_lower in str(rruff_id).lower()
                        or search_lower in str(ima_number).lower()
                        or search_lower in hey_class.lower()
                        or search_lower in chemical_family.lower()
                        or search_lower in ideal_chemistry.lower()
                        or search_lower in crystal_system.lower()
                        or search_lower in space_groups.lower()
                        or search_lower in locality.lower()
                        or search_lower in paragenetic_modes.lower()
                        or search_lower in chemistry_elements.lower()
                        or search_lower in description.lower()
                    )

                if matches_search:
                    tree.insert(
                        "",
                        tk.END,
                        iid=name,
                        values=(
                            display_name,
                            points,
                            mineral_name,
                            rruff_id,
                            ima_number,
                            hey_class,
                            chemical_family,
                            ideal_chemistry,
                            crystal_system,
                            space_groups,
                            locality,
                            paragenetic_modes,
                            chemistry_elements,
                            description,
                        ),
                    )

        # Search function
        def perform_search(event=None):
            populate_treeview(search_var.get())

        search_entry.bind("<Return>", perform_search)
        ttk.Button(search_frame, text="Search", command=perform_search).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            search_frame,
            text="Clear",
            command=lambda: [search_var.set(""), populate_treeview()],
        ).pack(side=tk.LEFT, padx=5)

        # --- Button Actions ---
        button_frame = ttk.Frame(main_frame, padding=(0, 5, 0, 0))
        button_frame.pack(fill=tk.X)

        def get_selected_name():
            selected_items = tree.selection()
            if not selected_items:
                messagebox.showwarning(
                    "Selection Required", "Please select a spectrum from the list."
                )
                return None
            return selected_items[0]

        def view_selected():
            name = get_selected_name()
            if name:
                self.view_database_item(name)

        def remove_selected():
            name = get_selected_name()
            if name:
                self.remove_database_item(
                    name, lambda: populate_treeview(search_var.get())
                )

        def edit_selected():
            name = get_selected_name()
            if name:
                self.edit_metadata(name)
                populate_treeview(search_var.get())

        ttk.Button(button_frame, text="View in Main Plot", command=view_selected).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Edit Metadata", command=edit_selected).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Remove Selected", command=remove_selected).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Close", command=db_window.destroy).pack(
            side=tk.RIGHT, padx=5
        )

        # --- Initial Population ---
        populate_treeview()

        # Make window modal
        db_window.transient(self.root)
        db_window.grab_set()
        self.root.wait_window(db_window)

    def view_database_item(self, name):
        """
        View a specific spectrum from the database.

        Parameters:
        -----------
        name : str
            Name/identifier of the spectrum to view.
        """
        if name not in self.raman.database:
            messagebox.showerror("Error", f"Spectrum '{name}' not found in database.")
            return

        # Get data
        data = self.raman.database[name]

        # Set as current spectrum
        self.raman.current_wavenumbers = data["wavenumbers"]
        self.raman.current_spectra = data["intensities"]

        # Set processed spectrum if available
        if "processed" in data:
            self.raman.processed_spectra = data["processed"]
        else:
            self.raman.processed_spectra = None

        # Set background if available
        if "background" in data:
            self.raman.background = data["background"]
        else:
            self.raman.background = None

        # Set peak data if available
        if "peaks" in data:
            self.raman.peaks = data["peaks"]
        else:
            self.raman.peaks = None

        # Set peak fitting data if available
        if "fitted_peaks" in data:
            self.raman.fitted_peaks = data["fitted_peaks"]
        else:
            self.raman.fitted_peaks = None

        if "peak_fit_result" in data:
            # Need to reconstruct the full peak_fit_result with fit curve
            # We'll do this when needed in update_plot to avoid unnecessary computations
            self.raman.peak_fit_result = data["peak_fit_result"]
        else:
            self.raman.peak_fit_result = None

        # Set metadata
        if "metadata" in data:
            self.raman.metadata = data["metadata"]
        else:
            self.raman.metadata = {}

        # Update display
        self.update_metadata_display()
        self.update_plot(include_background=False, include_peaks=False)

        # Update name input field
        self.var_name.set(name)

        # Update title bar
        self.root.title(f"ClaritySpectra: Raman Spectrum Analysis - {name}")

    def remove_database_item(self, name, refresh_callback=None):
        """Remove an item from the database after confirmation."""
        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to permanently remove '{name}' from the database?",
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
                    messagebox.showerror(
                        "Error", f"Could not find '{name}' in the database to remove."
                    )
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error removing '{name}' from database: {str(e)}"
                )

    # --- Search and Matching Logic ---

    def search_match(self):
        """Perform search using the basic parameters and selected algorithm."""
        if self._validate_search_conditions():
            try:
                try:
                    n_matches = int(self.var_n_matches.get().strip())
                except Exception as e:
                    messagebox.showerror(
                        "Input Error",
                        f"Number of matches must be an integer. Value: '{self.var_n_matches.get()}'\nError: {e}",
                    )
                    return
                try:
                    correlation_threshold = float(self.var_corr_threshold.get().strip())
                except Exception as e:
                    messagebox.showerror(
                        "Input Error",
                        f"Similarity threshold must be a number. Value: '{self.var_corr_threshold.get()}'\nError: {e}",
                    )
                    return
                algorithm = self.var_algorithm.get()
                # Store search parameters for reporting
                self.raman.last_search_algorithm = algorithm
                self.raman.last_search_threshold = correlation_threshold
                matches = self._perform_search(
                    algorithm, n_matches, correlation_threshold
                )
                self._process_and_display_matches(matches)
            except Exception as e:
                messagebox.showerror(
                    "Search Error", f"An error occurred during search: {str(e)}"
                )

    def advanced_search_match(self):
        """Perform search using advanced filters (peaks, chemical family, Hey Classification)."""
        if self._validate_search_conditions():
            try:
                try:
                    n_matches = int(self.var_n_matches.get().strip())
                except Exception as e:
                    messagebox.showerror(
                        "Input Error",
                        f"Number of matches must be an integer. Value: '{self.var_n_matches.get()}'\nError: {e}",
                    )
                    return
                try:
                    threshold = float(
                        self.var_adv_corr_threshold.get().strip()
                        if hasattr(self, "var_adv_corr_threshold")
                        else self.var_corr_threshold.get().strip()
                    )
                except Exception as e:
                    val = (
                        self.var_adv_corr_threshold.get()
                        if hasattr(self, "var_adv_corr_threshold")
                        else self.var_corr_threshold.get()
                    )
                    messagebox.showerror(
                        "Input Error",
                        f"Similarity threshold must be a number. Value: '{val}'\nError: {e}",
                    )
                    return
                # Store search parameters for reporting
                self.raman.last_search_algorithm = "Advanced (Filtered)"
                self.raman.last_search_threshold = threshold
                # Advanced Filters
                peak_positions = []
                peak_str = self.var_peak_positions.get().strip()
                if peak_str:
                    try:
                        peak_positions = [float(x.strip()) for x in peak_str.split(",")]
                    except ValueError:
                        messagebox.showerror(
                            "Input Error",
                            "Invalid peak positions format. Use comma-separated numbers.",
                        )
                        return

                peak_tolerance = 10  # Default
                tolerance_str = self.var_peak_tolerance.get().strip()
                if tolerance_str:
                    try:
                        peak_tolerance = float(tolerance_str)
                    except ValueError:
                        messagebox.showwarning(
                            "Input Error",
                            "Invalid peak tolerance. Using default (10 cm⁻¹).",
                        )

                chemical_family = (
                    self.var_chemical_family.get().strip() or None
                )  # None if empty
                hey_classification = (
                    self.var_hey_classification.get().strip() or None
                )  # None if empty

                # Get chemistry elements filters
                only_elements = None
                required_elements = None
                exclude_elements = None

                # Process "only these elements" filter
                only_elements_str = self.var_only_elements.get().strip()
                if only_elements_str:
                    only_elements = [
                        elem.strip().upper() for elem in only_elements_str.split(",")
                    ]

                # Process "required elements" filter
                required_elements_str = self.var_required_elements.get().strip()
                if required_elements_str:
                    required_elements = [
                        elem.strip().upper()
                        for elem in required_elements_str.split(",")
                    ]

                # Process "exclude elements" filter
                exclude_elements_str = self.var_exclude_elements.get().strip()
                if exclude_elements_str:
                    exclude_elements = [
                        elem.strip().upper() for elem in exclude_elements_str.split(",")
                    ]

                # Show a message to indicate search criteria
                if peak_positions:
                    messagebox.showinfo(
                        "Search Criteria",
                        f"Searching for spectra that contain ALL specified peaks: {', '.join([str(p) for p in peak_positions])}\n"
                        + f"With tolerance: ±{peak_tolerance} cm⁻¹",
                    )

                # Perform filtered search
                matches = self._filtered_search(
                    peak_positions=peak_positions,
                    peak_tolerance=peak_tolerance,
                    chemical_family=chemical_family,
                    threshold=threshold,
                    hey_classification=hey_classification,
                    only_elements=only_elements,
                    required_elements=required_elements,
                    exclude_elements=exclude_elements,
                )

                # Sort and limit results
                matches.sort(key=lambda x: x[1], reverse=True)
                self._process_and_display_matches(matches[:n_matches])

            except ValueError:
                messagebox.showerror(
                    "Input Error", "Number of matches or threshold is invalid."
                )
            except Exception as e:
                messagebox.showerror(
                    "Advanced Search Error", f"An error occurred: {str(e)}"
                )

    def _validate_search_conditions(self):
        """Check if a spectrum is loaded and the database is not empty."""
        if self.raman.current_spectra is None:
            messagebox.showerror(
                "Error", "No spectrum loaded. Please import a spectrum first."
            )
            return False
        if not self.raman.database:
            messagebox.showerror(
                "Error", "Database is empty. Please add spectra to the database."
            )
            return False
        return True

    def _perform_search(self, algorithm, n_matches, threshold, spectrum=None):
        """Helper function to call the correct search method."""
        if algorithm == "correlation":
            return self.correlation_search(n_matches, threshold, spectrum)
        elif algorithm == "peak":
            return self.peak_based_search(n_matches, threshold)
        elif algorithm == "ml" and SKLEARN_AVAILABLE:
            return self.ml_based_search(n_matches, threshold)
        elif algorithm == "ml":  # Scikit-learn not available
            messagebox.showwarning(
                "Missing Library",
                "scikit-learn not found. Falling back to 'Combined' search.",
            )
            return self.raman.search_match(n_matches, threshold)  # Fallback to combined
        else:  # Default to combined
            return self.raman.search_match(n_matches, threshold)

    def _process_and_display_matches(self, matches):
        """Update the results summary and display detailed results window."""
        self.match_result_text.config(state=tk.NORMAL)
        self.match_result_text.delete(1.0, tk.END)

        if matches:
            self.match_result_text.insert(tk.END, "=== Top 5 Search Results ===\n\n")
            for i, (name, score) in enumerate(matches[:5]):
                confidence = self.get_confidence_level(score)
                # Get mineral name from metadata if available
                mineral_name = name
                if (
                    name in self.raman.database
                    and "metadata" in self.raman.database[name]
                ):
                    metadata = self.raman.database[name]["metadata"]
                    if "NAME" in metadata and metadata["NAME"]:
                        mineral_name = metadata["NAME"]

                self.match_result_text.insert(tk.END, f"{i+1}. {mineral_name}\n")
                self.match_result_text.insert(
                    tk.END, f"   Score: {score:.4f} ({confidence})\n\n"
                )

            # Display comprehensive results in a new window
            self.display_search_results(matches)  # Pass all matches
        else:
            self.match_result_text.insert(
                tk.END, "No matches found matching the criteria."
            )

        self.match_result_text.config(state=tk.DISABLED)

    def correlation_search(self, n_matches, threshold, spectrum=None):
        """Perform correlation-based search."""
        # Update the search algorithm name for more detail
        self.raman.last_search_algorithm = "Correlation-based"
        self.raman.last_search_threshold = threshold

        if spectrum is not None:
            query_spectrum = spectrum
        elif self.raman.processed_spectra is not None:
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
        query_norm = (
            (query_spectrum - query_min) / (query_max - query_min)
            if (query_max > query_min)
            else query_spectrum
        )

        for name, data in self.raman.database.items():
            db_intensities = data["intensities"]
            db_wavenumbers = data["wavenumbers"]

            # Interpolate database spectrum if wavenumbers don't match
            if not np.array_equal(query_wavenumbers, db_wavenumbers):
                db_intensities_interp = np.interp(
                    query_wavenumbers, db_wavenumbers, db_intensities
                )
            else:
                db_intensities_interp = db_intensities

            # Normalize database spectrum
            db_min = np.min(db_intensities_interp)
            db_max = np.max(db_intensities_interp)
            db_norm = (
                (db_intensities_interp - db_min) / (db_max - db_min)
                if (db_max > db_min)
                else db_intensities_interp
            )

            # Calculate correlation coefficient
            try:
                # Ensure vectors are not constant
                if np.all(query_norm == query_norm[0]) or np.all(db_norm == db_norm[0]):
                    corr_coef = 0  # Correlation is undefined for constant vectors
                else:
                    corr_coef = np.corrcoef(query_norm, db_norm)[0, 1]
                if np.isnan(corr_coef):  # Handle potential NaN from std dev calculation
                    corr_coef = 0
            except Exception:  # Catch any other potential errors during calculation
                corr_coef = 0

            if corr_coef >= threshold:
                matches.append((name, corr_coef))

        # Sort matches by correlation score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:n_matches]

    def peak_based_search(self, n_matches, threshold):
        """Perform peak-based search using Jaccard similarity."""
        # Update the search algorithm name for more detail
        self.raman.last_search_algorithm = "Peak-based (Jaccard)"
        self.raman.last_search_threshold = threshold

        if self.raman.peaks is None:
            # Find peaks if not already done for the current spectrum
            try:
                self.find_peaks()  # Use existing parameters or defaults
                if self.raman.peaks is None:  # Check if find_peaks failed
                    raise ValueError("Could not detect peaks in the query spectrum.")
            except Exception as e:
                messagebox.showerror(
                    "Peak Search Error", f"Failed to find peaks for query spectrum: {e}"
                )
                return []  # Return empty list if peaks cannot be found

        if self.raman.peaks is None or not self.raman.peaks.get("wavenumbers").size:
            messagebox.showwarning(
                "Peak Search",
                "No peaks found in the query spectrum to perform peak-based search.",
            )
            return []

        query_peaks = self.raman.peaks["wavenumbers"]
        matches = []
        tolerance = 10  # Default tolerance for peak matching (cm^-1)
        # Optionally get tolerance from GUI if available:
        if hasattr(self, "var_peak_tolerance"):
            try:
                tolerance = float(self.var_peak_tolerance.get())
            except ValueError:
                pass  # Use default if GUI value is invalid

        for name, data in self.raman.database.items():
            db_peak_data = data.get("peaks")
            if not db_peak_data or db_peak_data.get("wavenumbers") is None:
                continue  # Skip database entries without pre-calculated peaks

            db_peaks_wavenumbers = db_peak_data["wavenumbers"]
            if not db_peaks_wavenumbers.size:
                continue  # Skip if database entry has no peaks

            # Calculate Jaccard similarity (intersection over union)
            query_set = set(
                np.round(query_peaks / tolerance)
            )  # Group peaks by tolerance
            db_set = set(np.round(db_peaks_wavenumbers / tolerance))

            intersection = len(query_set.intersection(db_set))
            union = len(query_set.union(db_set))

            jaccard = intersection / union if union > 0 else 0

            if jaccard >= threshold:
                matches.append((name, jaccard))

        # Sort matches by Jaccard score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:n_matches]

    def ml_based_search(self, n_matches, threshold, spectrum=None):
                """Perform machine learning based search using DTW (fastdtw) similarity.
                
                Parameters:
                -----------
                n_matches : int
                    Number of top matches to return
                threshold : float
                    Similarity threshold (0-1)
                spectrum : array-like, optional
                    Custom spectrum to search with. If None, uses self.raman.processed_spectra
                    or self.raman.current_spectra
                """
                from scipy.spatial.distance import cdist
                self.raman.last_search_algorithm = "DTW-based (fastdtw)"
                self.raman.last_search_threshold = threshold
                
                # Use provided spectrum if available, otherwise use class's spectra
                if spectrum is not None:
                    query_spectrum = spectrum
                elif self.raman.processed_spectra is not None:
                    query_spectrum = self.raman.processed_spectra
                elif self.raman.current_spectra is not None:
                    query_spectrum = self.raman.current_spectra
                else:
                    raise ValueError("No spectrum data available for search.")
                
                query_wavenumbers = self.raman.current_wavenumbers
                matches = []
                db_names = []
                db_spectra_interp = []
                
                for name, data in self.raman.database.items():
                    db_intensities = data["intensities"]
                    db_wavenumbers = data["wavenumbers"]
                    # Interpolate to match query wavenumbers
                    if not np.array_equal(query_wavenumbers, db_wavenumbers):
                        db_interp = np.interp(query_wavenumbers, db_wavenumbers, db_intensities)
                    else:
                        db_interp = db_intensities
                    db_spectra_interp.append(db_interp)
                    db_names.append(name)
                
                # Normalize query spectrum
                query_max = np.max(query_spectrum)
                query_norm = query_spectrum / query_max if query_max > 0 else query_spectrum
                query_norm = np.asarray(query_norm).flatten()  # Ensure 1-D
                
                # Compute distances for each database spectrum
                distances = []
                for i, db_spec in enumerate(db_spectra_interp):
                    db_max = np.max(db_spec)
                    db_norm = db_spec / db_max if db_max > 0 else db_spec
                    db_norm = np.asarray(db_norm).flatten()  # Ensure 1-D
                    
                    # Reshape arrays for cdist (needs 2D arrays)
                    query_2d = query_norm.reshape(1, -1)
                    db_2d = db_norm.reshape(1, -1)
                    # Compute Euclidean distance
                    distance = cdist(query_2d, db_2d, "euclidean")[0][0]
                    distances.append((db_names[i], distance))
                
                # Normalize distances to [0, 1] range
                max_distance = max(dist for _, dist in distances)
                normalized_distances = [(name, dist / max_distance) for name, dist in distances]
                
                # Convert normalized distance to similarity using a more gradual transformation
                # Using exp(-x) gives a smoother falloff than 1/(1+x)
                matches = [
                    (name, np.exp(-dist))
                    for name, dist in normalized_distances
                    if np.exp(-dist) >= threshold
                ]
                matches.sort(key=lambda x: x[1], reverse=True)
                return matches[:n_matches]

    def _filtered_search(
        self,
        peak_positions,
        peak_tolerance,
        chemical_family,
        threshold,
        hey_classification=None,
        only_elements=None,
        required_elements=None,
        exclude_elements=None,
    ):
        """Perform a filtered search using peak positions, chemical family, and/or Hey Classification.

        Args:
            peak_positions : list
                List of peak positions to search for.
            peak_tolerance : float
                Tolerance for peak matching in cm⁻¹.
            chemical_family : str
                Chemical family to filter by.
            threshold : float
                Correlation threshold for matching.
            hey_classification : str
                Hey Classification to filter by.
            only_elements : list
                List of elements to include in the search.
            required_elements : list
                List of elements to require in the search.
            exclude_elements : list
                List of elements to exclude from the search.

        Returns:
            list: List of (name, score) tuples for matching spectra.
        """
        matches = []

        # Get the query spectrum
        if self.raman.processed_spectra is not None:
            query_spectrum = self.raman.processed_spectra
        elif self.raman.current_spectra is not None:
            query_spectrum = self.raman.current_spectra
        else:
            raise ValueError("No spectrum data available for search.")

        query_wavenumbers = self.raman.current_wavenumbers

        # Normalize query spectrum once (for correlation calculation)
        query_min = np.min(query_spectrum)
        query_max = np.max(query_spectrum)
        query_norm = (
            (query_spectrum - query_min) / (query_max - query_min)
            if (query_max > query_min)
            else query_spectrum
        )

        # Iterate through all database entries
        for name, data in self.raman.database.items():
            db_meta = data.get("metadata", {})
            match_score = 0.0

            # Apply chemical family filter
            if chemical_family:
                db_family = db_meta.get("chemical_family")
                if not db_family or db_family.lower() != chemical_family.lower():
                    continue

            # Apply Hey Classification filter
            if hey_classification:
                db_hey_class = db_meta.get("hey_classification")
                if (
                    not db_hey_class
                    or db_hey_class.lower() != hey_classification.lower()
                ):
                    continue

            # Apply chemistry elements filters
            chemistry_elements = db_meta.get(
                "CHEMISTRY ELEMENTS", ""
            )  # Note: Using uppercase field name
            if chemistry_elements:
                # Convert to uppercase and split, then strip whitespace
                elements = [
                    elem.strip().upper() for elem in chemistry_elements.split(",")
                ]

                # Check "only these elements" filter
                if only_elements:
                    # Convert both lists to sets for comparison
                    db_elements_set = set(elements)
                    required_set = set(only_elements)
                    if db_elements_set != required_set:
                        continue

                # Check "required elements" filter
                if required_elements:
                    # Convert both lists to sets for comparison
                    db_elements_set = set(elements)
                    required_set = set(required_elements)
                    if not required_set.issubset(db_elements_set):
                        continue

                # Check "exclude elements" filter
                if exclude_elements:
                    # Convert both lists to sets for comparison
                    db_elements_set = set(elements)
                    exclude_set = set(exclude_elements)
                    if db_elements_set.intersection(exclude_set):
                        continue

            # If peak positions are specified, check if the spectrum contains all of them
            if peak_positions:
                db_peak_data = data.get("peaks")
                if not db_peak_data or db_peak_data.get("wavenumbers") is None:
                    continue

                db_peaks = db_peak_data["wavenumbers"]
                if not db_peaks.size:
                    continue

                # Check if all specified peaks are present within tolerance
                all_peaks_found = True
                for target_peak in peak_positions:
                    peak_found = False
                    for peak in db_peaks:
                        if abs(peak - target_peak) <= peak_tolerance:
                            peak_found = True
                            break
                    if not peak_found:
                        all_peaks_found = False
                        break

                if not all_peaks_found:
                    continue

            # Calculate correlation score
            db_intensities = data["intensities"]
            db_wavenumbers = data["wavenumbers"]

            if not np.array_equal(query_wavenumbers, db_wavenumbers):
                db_intensities_interp = np.interp(
                    query_wavenumbers, db_wavenumbers, db_intensities
                )
            else:
                db_intensities_interp = db_intensities

            db_min = np.min(db_intensities_interp)
            db_max = np.max(db_intensities_interp)
            db_norm = (
                (db_intensities_interp - db_min) / (db_max - db_min)
                if (db_max > db_min)
                else db_intensities_interp
            )

            try:
                if np.all(query_norm == query_norm[0]) or np.all(db_norm == db_norm[0]):
                    corr_coef = 0.0
                else:
                    corr_coef = np.corrcoef(query_norm, db_norm)[0, 1]
                if np.isnan(corr_coef):
                    corr_coef = 0.0
            except Exception:
                corr_coef = 0.0

            match_score = corr_coef

            # Add to matches if it meets the threshold
            if match_score >= threshold:
                matches.append((name, match_score))

        # Sort matches by score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
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
        if not hasattr(fig, "_original_dims"):
            fig._original_dims = (original_width, original_height)
        else:
            original_width, original_height = fig._original_dims

        # Reset the figure size
        fig.set_size_inches(original_width, original_height)

        # Reset the subplot parameters to prevent shrinking
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)

        return fig

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
                text_widget.config(state=tk.DISABLED)
                return

            # Add header
            text_widget.insert(tk.END, "Raman Spectrum Analysis Report\n")
            text_widget.insert(tk.END, "=" * 50 + "\n\n")

            # Add timestamp
            text_widget.insert(
                tk.END,
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            )

            # === Sample Information ===
            text_widget.insert(tk.END, "Sample Information\n")
            text_widget.insert(tk.END, "-" * 25 + "\n")
            if (
                hasattr(self.raman, "current_spectra")
                and self.raman.current_spectra is not None
            ):
                if self.raman.metadata:
                    # Add specific metadata fields if available
                    if "NAME" in self.raman.metadata:
                        text_widget.insert(
                            tk.END, f"Sample Name: {self.raman.metadata['NAME']}\n"
                        )

                    # Add Hey Classification if available
                    if "HEY CLASSIFICATION" in self.raman.metadata:
                        text_widget.insert(
                            tk.END,
                            f"Hey Classification: {self.raman.metadata['HEY CLASSIFICATION']}\n",
                        )

                    # Add structural information
                    if "CRYSTAL SYSTEM" in self.raman.metadata:
                        text_widget.insert(
                            tk.END,
                            f"Crystal System: {self.raman.metadata['CRYSTAL SYSTEM']}\n",
                        )
                    if "SPACE GROUP" in self.raman.metadata:
                        text_widget.insert(
                            tk.END,
                            f"Space Group: {self.raman.metadata['SPACE GROUP']}\n",
                        )

                    # Add chemical information
                    if "CHEMICAL FAMILY" in self.raman.metadata:
                        text_widget.insert(
                            tk.END,
                            f"Chemical Family: {self.raman.metadata['CHEMICAL FAMILY']}\n",
                        )
                    if "IDEAL CHEMISTRY" in self.raman.metadata:
                        text_widget.insert(
                            tk.END,
                            f"Chemistry: {self.raman.metadata['IDEAL CHEMISTRY']}\n",
                        )

                    # Add origin information
                    if "PARAGENETIC MODE" in self.raman.metadata:
                        text_widget.insert(
                            tk.END,
                            f"Paragenetic Mode: {self.raman.metadata['PARAGENETIC MODE']}\n",
                        )
                    if "OLDEST KNOWN AGE (MA)" in self.raman.metadata:
                        text_widget.insert(
                            tk.END,
                            f"Oldest Known Age (Ma): {self.raman.metadata['OLDEST KNOWN AGE (MA)']}\n",
                        )

                    # Add locality
                    if "LOCALITY" in self.raman.metadata:
                        text_widget.insert(
                            tk.END, f"Locality: {self.raman.metadata['LOCALITY']}\n"
                        )

                    # Add other metadata fields
                    for key, value in self.raman.metadata.items():
                        if key not in [
                            "NAME",
                            "HEY CLASSIFICATION",
                            "CRYSTAL SYSTEM",
                            "SPACE GROUP",
                            "CHEMICAL FAMILY",
                            "IDEAL CHEMISTRY",
                            "PARAGENETIC MODE",
                            "OLDEST KNOWN AGE (MA)",
                            "LOCALITY",
                        ]:
                            text_widget.insert(tk.END, f"{key}: {value}\n")
                else:
                    text_widget.insert(
                        tk.END, "No metadata available for current spectrum.\n"
                    )
            else:
                text_widget.insert(tk.END, "No current spectrum loaded.\n")

            text_widget.insert(tk.END, "\n")

            # === Processing Information ===
            text_widget.insert(tk.END, "Processing Information\n")
            text_widget.insert(tk.END, "-" * 25 + "\n")

            # Processing details
            if (
                hasattr(self.raman, "processed_spectra")
                and self.raman.processed_spectra is not None
            ):
                text_widget.insert(tk.END, "Background Subtraction: Applied\n")
            else:
                text_widget.insert(tk.END, "Background Subtraction: Not Applied\n")

            if hasattr(self.raman, "peaks") and self.raman.peaks is not None:
                num_peaks = len(self.raman.peaks["wavenumbers"])
                peak_list = ", ".join(
                    [f"{wn:.1f}" for wn in self.raman.peaks["wavenumbers"]]
                )
                text_widget.insert(
                    tk.END, f"Peak Detection: {num_peaks} peaks detected\n"
                )
                text_widget.insert(tk.END, f"Detected Peaks: {peak_list}\n")
            else:
                text_widget.insert(tk.END, "Peak Detection: Not Applied\n")

            # Add search settings
            text_widget.insert(
                tk.END,
                f"Search Algorithm: {self.raman.last_search_algorithm if hasattr(self.raman, 'last_search_algorithm') else 'Unknown'}\n",
            )
            text_widget.insert(
                tk.END,
                f"Match Threshold: {self.raman.last_search_threshold if hasattr(self.raman, 'last_search_threshold') else 'Unknown'}\n",
            )
            text_widget.insert(tk.END, "\n")

            # === Match Results ===
            text_widget.insert(tk.END, "Match Results\n")
            text_widget.insert(tk.END, "-" * 25 + "\n")

            for i, (name, score) in enumerate(matches, 1):
                confidence = self.get_confidence_level(score)

                # Get mineral name from metadata if available
                mineral_name = name
                if (
                    name in self.raman.database
                    and "metadata" in self.raman.database[name]
                ):
                    metadata = self.raman.database[name].get("metadata", {})
                    if "NAME" in metadata and metadata["NAME"]:
                        mineral_name = metadata["NAME"]

                text_widget.insert(tk.END, f"{i}. {mineral_name}\n")
                text_widget.insert(
                    tk.END, f"   Match Score: {score:.4f} ({confidence} confidence)\n"
                )

                # Get metadata if available
                if name in self.raman.database:
                    metadata = self.raman.database[name].get("metadata", {})

                    if metadata:
                        text_widget.insert(tk.END, f"   Compound Information:\n")

                        # Hey Classification
                        if "HEY CLASSIFICATION" in metadata:
                            text_widget.insert(
                                tk.END,
                                f"   - Hey Classification: {metadata['HEY CLASSIFICATION']}\n",
                            )

                        # Structural information
                        if "CRYSTAL SYSTEM" in metadata:
                            text_widget.insert(
                                tk.END,
                                f"   - Crystal System: {metadata['CRYSTAL SYSTEM']}\n",
                            )
                        if "SPACE GROUP" in metadata:
                            text_widget.insert(
                                tk.END, f"   - Space Group: {metadata['SPACE GROUP']}\n"
                            )

                        # Chemical information
                        if "CHEMICAL FAMILY" in metadata:
                            text_widget.insert(
                                tk.END,
                                f"   - Chemical Family: {metadata['CHEMICAL FAMILY']}\n",
                            )
                        if "IDEAL CHEMISTRY" in metadata:
                            text_widget.insert(
                                tk.END,
                                f"   - Chemistry: {metadata['IDEAL CHEMISTRY']}\n",
                            )

                        # Origin information
                        if "PARAGENETIC MODE" in metadata:
                            text_widget.insert(
                                tk.END,
                                f"   - Paragenetic Mode: {metadata['PARAGENETIC MODE']}\n",
                            )
                        if "OLDEST KNOWN AGE (MA)" in metadata:
                            text_widget.insert(
                                tk.END,
                                f"   - Oldest Known Age (Ma): {metadata['OLDEST KNOWN AGE (MA)']}\n",
                            )

                        # Locality
                        if "LOCALITY" in metadata:
                            text_widget.insert(
                                tk.END, f"   - Locality: {metadata['LOCALITY']}\n"
                            )

                # Add separator between matches
                text_widget.insert(tk.END, "\n")

            # === Analysis and Recommendations ===
            text_widget.insert(tk.END, "Analysis and Recommendations\n")
            text_widget.insert(tk.END, "-" * 25 + "\n")

            if matches:
                best_match = matches[0]
                best_name, best_score = best_match

                # Get mineral name for best match
                best_mineral_name = best_name
                if (
                    best_name in self.raman.database
                    and "metadata" in self.raman.database[best_name]
                ):
                    metadata = self.raman.database[best_name].get("metadata", {})
                    if "NAME" in metadata and metadata["NAME"]:
                        best_mineral_name = metadata["NAME"]

                confidence = self.get_confidence_level(best_score)

                if best_score >= 0.85:
                    text_widget.insert(
                        tk.END,
                        f"The best match ({best_mineral_name}) has a {confidence} confidence level.\n",
                    )
                    text_widget.insert(
                        tk.END,
                        "This indicates a high probability of a correct match.\n",
                    )
                elif best_score >= 0.75:
                    text_widget.insert(
                        tk.END,
                        f"The best match ({best_mineral_name}) has a {confidence} confidence level.\n",
                    )
                    text_widget.insert(
                        tk.END,
                        "Consider additional analysis to confirm this identification.\n",
                    )
                else:
                    text_widget.insert(
                        tk.END,
                        f"The best match ({best_mineral_name}) has a {confidence} confidence level.\n",
                    )
                    text_widget.insert(
                        tk.END,
                        "The match quality is low. Consider additional techniques for identification.\n",
                    )

                # Check for multiple similar matches
                if len(matches) > 1:
                    second_best = matches[1]
                    _, second_score = second_best
                    score_diff = best_score - second_score

                    if score_diff < 0.05:
                        text_widget.insert(
                            tk.END,
                            "Note: Multiple close matches found. Consider comparing the spectra carefully.\n",
                        )

            # Summary stats
            text_widget.insert(tk.END, "\nSummary Statistics:\n")
            text_widget.insert(tk.END, f"Total Matches: {len(matches)}\n")

            if matches:
                avg_score = sum(score for _, score in matches) / len(matches)
                text_widget.insert(tk.END, f"Average Match Score: {avg_score:.4f}\n")

                # Score range
                min_score = min(score for _, score in matches)
                max_score = max(score for _, score in matches)
                text_widget.insert(
                    tk.END, f"Score Range: {min_score:.4f} - {max_score:.4f}\n"
                )

            # Disable editing
            text_widget.config(state=tk.DISABLED)

        except Exception as e:
            # Handle any errors that occur during report generation
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, f"Error generating report: {str(e)}\n")
            text_widget.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")

    def export_report_as_pdf(self, report_text, matches=None):
        """Export the report as a PDF file using reportlab."""
        if not REPORTLAB_AVAILABLE:
            messagebox.showerror(
                "PDF Export Error", "ReportLab library not found. Cannot export as PDF."
            )
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save Report as PDF",
            )
            if not filename:
                return  # User cancelled

            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            # Customize styles slightly
            styles["Title"].alignment = 1  # Center alignment
            styles["h2"].spaceBefore = 10
            styles["h2"].spaceAfter = 2
            styles["Normal"].spaceAfter = 4
            styles["Code"].fontSize = 8  # For peak list maybe?

            lines = report_text.split("\n")
            in_match_section = False

            # Get the selected match from the matches list
            match_name = None
            match_score = 0

            if matches and len(matches) > 0:
                # Get the selected match from the matches list
                selected_match = None
                selected_match_name = self.selected_match.get()

                if selected_match_name:
                    # Find the selected match in the matches list
                    for name, score in matches:
                        if name == selected_match_name:
                            selected_match = (name, score)
                            break

                if selected_match:
                    match_name, match_score = selected_match
                else:
                    # Fallback to first match if no selection found
                    match_name, match_score = matches[0]
            else:
                # Try to extract match information from report text as fallback
                for j, l in enumerate(lines):
                    if "Match Results" in l:
                        # Look for the first match after this line
                        for k in range(j + 1, len(lines)):
                            if (
                                lines[k].strip()
                                and lines[k][0].isdigit()
                                and ". " in lines[k]
                            ):
                                match_text = lines[k].strip()
                                match_name = match_text.split(". ")[1].split("\n")[0]
                                # Try to find the score in the next line
                                if (
                                    k + 1 < len(lines)
                                    and "Match Score:" in lines[k + 1]
                                ):
                                    try:
                                        score_text = (
                                            lines[k + 1]
                                            .strip()
                                            .split("Match Score:")[1]
                                            .split("(")[0]
                                            .strip()
                                        )
                                        match_score = float(score_text)
                                    except:
                                        pass
                                break
                        break

            # Print debug info
            print(f"Match name found: {match_name}, Score: {match_score}")
            if match_name:
                print(f"Match in database: {match_name in self.raman.database}")

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:  # Skip empty lines unless needed for spacing
                    # elements.append(Spacer(1, 6)) # Add manual space if needed
                    continue

                if line == "Raman Spectrum Analysis Report":
                    elements.append(Paragraph(line, styles["Title"]))

                    # After title, add the fit image and mineral vibrations analysis image
                    # Only try to add images if we have a valid match
                    if match_name and match_name in self.raman.database:
                        match_data = (match_name, match_score)

                        # Create a temporary figure for the comparison plot
                        fit_fig = plt.figure(figsize=(7, 4), dpi=150)
                        fit_ax = fit_fig.add_subplot(111)
                        self.plot_comparison_data(match_data, fit_fig, fit_ax)

                        # Save the figure to a BytesIO object
                        fit_buffer = BytesIO()
                        fit_fig.savefig(fit_buffer, format="png", dpi=150)
                        fit_buffer.seek(0)

                        # Add the fit image to the report
                        elements.append(Spacer(1, 10))
                        elements.append(
                            Paragraph("Spectral Fit Comparison", styles["h2"])
                        )
                        fit_image = Image(fit_buffer, width=450, height=250)
                        elements.append(fit_image)
                        elements.append(Spacer(1, 10))

                        # Create a temporary figure for the mineral vibrations analysis
                        vib_fig = plt.figure(figsize=(7, 4), dpi=150)
                        vib_ax = vib_fig.add_subplot(111)
                        self.plot_mineral_vibrations(match_data, vib_fig, vib_ax)

                        # Save the figure to a BytesIO object
                        vib_buffer = BytesIO()
                        vib_fig.savefig(vib_buffer, format="png", dpi=150)
                        vib_buffer.seek(0)

                        # Add the mineral vibrations image to the report
                        elements.append(
                            Paragraph("Mineral Vibrations Analysis", styles["h2"])
                        )
                        vib_image = Image(vib_buffer, width=450, height=250)
                        elements.append(vib_image)
                        elements.append(Spacer(1, 15))

                        # Clean up
                        plt.close(fit_fig)
                        plt.close(vib_fig)
                    else:
                        # If there's no valid match, add a message explaining why images are missing
                        elements.append(Spacer(1, 10))
                        elements.append(
                            Paragraph(
                                "Note: Spectral comparison and mineral vibration images could not be generated because no valid match was found.",
                                styles["Normal"],
                            )
                        )
                        elements.append(Spacer(1, 10))
                elif line.startswith("=" * 5):  # Separator
                    elements.append(Spacer(1, 12))
                elif line in [
                    "Sample Information",
                    "Processing Information",
                    "Match Results",
                    "Analysis and Recommendations",
                ]:
                    elements.append(Paragraph(line, styles["h2"]))
                    in_match_section = line == "Match Results"
                elif line.startswith("-" * 5):  # Underline for sections
                    pass  # Skip underline
                elif (
                    in_match_section and line[0].isdigit() and ". " in line
                ):  # Match name line
                    elements.append(
                        Paragraph(line, styles["h3"])
                    )  # Use h3 for match name
                elif line.startswith("   Match Score:") or line.startswith(
                    "   Compound Information:"
                ):
                    elements.append(Paragraph(line, styles["Normal"]))
                elif line.startswith("   - "):  # Compound details
                    elements.append(Paragraph(line, styles["Normal"]))
                elif line.startswith("Detected Peaks"):
                    # Potentially style the peak list differently
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        elements.append(
                            Paragraph(
                                f"<b>{parts[0]}:</b> {parts[1]}", styles["Normal"]
                            )
                        )
                    else:
                        elements.append(Paragraph(line, styles["Normal"]))
                else:  # Default normal style
                    elements.append(Paragraph(line, styles["Normal"]))

            doc.build(elements)
            messagebox.showinfo("Export Successful", f"Report exported to {filename}")

        except Exception as e:
            messagebox.showerror(
                "PDF Export Error", f"Error exporting report as PDF: {str(e)}"
            )
            # Print detailed error for debugging
            import traceback

            traceback.print_exc()

    def export_report_as_txt(self, report_text):
        """Export the report as a plain text file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Report as TXT",
            )
            if not filename:
                return  # User cancelled

            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_text)

            messagebox.showinfo("Export Successful", f"Report exported to {filename}")

        except Exception as e:
            messagebox.showerror(
                "TXT Export Error", f"Error exporting report as TXT: {str(e)}"
            )

    def export_results_as_csv(self, matches):
        """Export search results (matches) as a CSV file with comprehensive metadata."""
        if not matches:
            messagebox.showerror("Export Error", "No matches to export.")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Results as CSV",
            )
            if not filename:
                return  # User cancelled

            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # Write header with extended metadata fields
                header = [
                    "Rank",
                    "Name",
                    "Score",
                    "Confidence",
                    "Hey Classification",
                    "Chemical Family",
                    "Crystal System",
                    "Space Group",
                    "Paragenetic Mode",
                    "Oldest Known Age (Ma)",
                    "Ideal Chemistry",
                    "Locality",
                ]
                writer.writerow(header)

                # Write data rows
                for i, (name, score) in enumerate(matches):
                    confidence = self.get_confidence_level(score)

                    # Initialize metadata fields with empty values
                    hey_class = ""
                    family = ""
                    crystal_system = ""
                    space_group = ""
                    paragenetic_mode = ""
                    oldest_age = ""
                    chemistry = ""
                    locality = ""

                    # Get metadata if available from database
                    if name in self.raman.database:
                        data = self.raman.database[name]
                        if "metadata" in data and data.get("metadata"):
                            metadata = data["metadata"]
                            hey_class = metadata.get("HEY CLASSIFICATION", "")
                            family = metadata.get("CHEMICAL FAMILY", "")
                            crystal_system = metadata.get("CRYSTAL SYSTEM", "")
                            space_group = metadata.get("SPACE GROUP", "")
                            paragenetic_mode = metadata.get("PARAGENETIC MODES", "")
                            oldest_age = metadata.get("OLDEST KNOWN AGE (MA)", "")
                            chemistry = metadata.get("IDEAL CHEMISTRY", "")
                            locality = metadata.get("LOCALITY", "")

                    # Write row with all metadata fields
                    row = [
                        i + 1,
                        name,
                        f"{score:.4f}",
                        confidence,
                        hey_class,
                        family,
                        crystal_system,
                        space_group,
                        paragenetic_mode,
                        oldest_age,
                        chemistry,
                        locality,
                    ]
                    writer.writerow(row)

            messagebox.showinfo("Export Successful", f"Results exported to {filename}")

        except Exception as e:
            messagebox.showerror(
                "CSV Export Error", f"Error exporting results as CSV: {str(e)}"
            )

    # --- Helper Methods ---

    def get_confidence_level(self, score):
        """Convert numeric score (0-1) to confidence level description."""
        # Adjust thresholds based on expected score range and meaning
        if score >= 0.95:
            return "Excellent"
        elif score >= 0.85:
            return "Good"
        elif score >= 0.75:
            return "Possible"
        elif score >= 0.65:
            return "Weak"
        else:
            return "Poor"

    def on_match_select(self, event=None):
        """Handle the selection of a match from the search results."""
        selected_indices = self.matches_listbox.curselection()
        if not selected_indices:
            return

        # Get the first selected match
        match_text = self.matches_listbox.get(selected_indices[0])

        # Use the stored mapping to get the original filename and score
        if hasattr(self, "match_data_map") and match_text in self.match_data_map:
            match_name, score = self.match_data_map[match_text]
        else:
            # Fallback to the old method (parsing the text)
            if " (Score:" in match_text:
                match_name = match_text.split(" (Score:")[0]
                try:
                    score = float(match_text.split("Score: ")[1].split(")")[0])
                except:
                    score = 0.0
            else:
                match_name = match_text
                score = 0.0

        if match_name not in self.raman.database:
            return

        # Update comparison plot
        fig_comp = self.results_figures["comparison"]
        fig_comp.clear()
        ax_comp = fig_comp.add_subplot(111)
        self.plot_comparison_data((match_name, score), fig_comp, ax_comp)
        self.results_figures["canvas_comp"].draw()

        # Get the x-axis ticks and labels from the comparison plot
        comp_xticks = ax_comp.get_xticks()
        comp_xticklabels = ax_comp.get_xticklabels()
        comp_xlim = ax_comp.get_xlim()

        # Store the figure dimensions and position for the comparison plot
        self.comparison_fig_position = {
            "position": ax_comp.get_position(),
            "subplots_params": {
                "left": fig_comp.subplotpars.left,
                "right": fig_comp.subplotpars.right,
                "top": fig_comp.subplotpars.top,
                "bottom": fig_comp.subplotpars.bottom,
            },
        }

        # Update mineral vibration analysis
        if (
            "mineral" in self.results_figures
            and "canvas_mineral" in self.results_figures
        ):
            fig_mineral = self.results_figures["mineral"]
            canvas_mineral = self.results_figures["canvas_mineral"]

            # Call the plot_mineral_vibrations method to generate the mineral vibration analysis
            ax_mineral = self.plot_mineral_vibrations(
                (match_name, score), fig_mineral, None, ax_comp
            )

            # Apply the same x-axis ticks and limits from the comparison plot
            ax_mineral.set_xlim(comp_xlim)
            ax_mineral.set_xticks(comp_xticks)

            # Format the tick labels to match
            ax_mineral.xaxis.set_major_formatter(ax_comp.xaxis.get_major_formatter())

            # Apply any additional x-axis styling (grid, minor ticks, etc.)
            ax_mineral.grid(
                True, axis="x", linestyle=":", color="gray", alpha=0.6
            )  # Match the grid style

            # Ensure exact x-axis alignment by matching the figure dimension ratios
            if hasattr(self, "comparison_fig_position"):
                # Set the subplot position to match the comparison plot's horizontal positioning
                # But maintain the current vertical position
                comp_pos = self.comparison_fig_position["position"]
                min_pos = ax_mineral.get_position()

                # Create a new position that matches x-axis positioning of comparison plot
                # but preserves the vertical position of mineral plot
                new_pos = [comp_pos.x0, min_pos.y0, comp_pos.width, min_pos.height]
                ax_mineral.set_position(new_pos)

                # Also match the figure-level subplot parameters for left and right margins
                comp_params = self.comparison_fig_position["subplots_params"]
                fig_mineral.subplots_adjust(
                    left=comp_params["left"],
                    right=comp_params["right"],
                    # Keep current vertical margins
                    top=fig_mineral.subplotpars.top,
                    bottom=fig_mineral.subplotpars.bottom,
                )

            # Update the mineral plot
            canvas_mineral.draw_idle()

        # Always update metadata window if it exists
        if hasattr(self, "metadata_window") and self.metadata_window is not None:
            # Check if window still exists (not closed)
            if self.metadata_window.winfo_exists():
                # Update the metadata content with current selection
                self.update_metadata_window()
            else:
                # Window was closed but reference still exists, set to None
                self.metadata_window = None

    def update_plot(self, include_background=False, include_peaks=False):
        """Update the main plot with current/processed spectrum, background, and peaks."""
        if self.raman.current_spectra is None:
            self.ax.clear()
            self.ax.text(
                0.5,
                0.5,
                "No Spectrum Loaded",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            self.canvas.draw()
            return

        # Completely clear the figure and create a new axis
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        # Plot processed spectrum if available, otherwise raw
        spectrum_to_plot = self.raman.current_spectra
        label = "Raw Spectrum"
        color = "blue"
        if self.raman.processed_spectra is not None:
            spectrum_to_plot = self.raman.processed_spectra
            label = "Processed Spectrum"
            color = "green"

        # Plot the spectrum
        self.ax.plot(
            self.raman.current_wavenumbers,
            spectrum_to_plot,
            color=color,
            label=label,
            linewidth=1.5,
        )

        # Plot background if requested and available
        if include_background and self.raman.background is not None:
            self.ax.plot(
                self.raman.current_wavenumbers,
                self.raman.background,
                "r--",
                label="Background",
                linewidth=1,
            )

        # Mark peaks if requested and available
        if include_peaks and self.raman.peaks is not None:
            peak_indices = self.raman.peaks.get("indices")
            if peak_indices is not None:
                # Use indices to get heights from the plotted spectrum
                peak_heights = spectrum_to_plot[peak_indices]
                peak_wavenumbers = self.raman.current_wavenumbers[peak_indices]
                self.ax.plot(
                    peak_wavenumbers, peak_heights, "ro", label="Peaks", markersize=4
                )

                # Add wavenumber annotations for each peak
                for i, (wn, height) in enumerate(zip(peak_wavenumbers, peak_heights)):
                    self.ax.annotate(
                        f"{wn:.1f}",  # Format to 1 decimal place
                        xy=(wn, height),  # Point to annotate
                        xytext=(0, 5),  # Offset text by 5 points above
                        textcoords="offset points",
                        ha="center",  # Horizontally center text
                        fontsize=8,  # Small font to avoid crowding
                    )

        # Show fitted peaks if available
        if hasattr(self.raman, "fitted_peaks") and self.raman.fitted_peaks is not None:
            # Plot the overall fit curve if available
            if (
                hasattr(self.raman, "peak_fit_result")
                and self.raman.peak_fit_result is not None
            ):
                if "fit_curve" in self.raman.peak_fit_result:
                    self.ax.plot(
                        self.raman.current_wavenumbers,
                        self.raman.peak_fit_result["fit_curve"],
                        "k-",
                        label=f"Peak Fit ({self.raman.peak_fit_result['model']})",
                        linewidth=1.5,
                        alpha=0.7,
                    )

            # Plot individual fitted peaks
            colors = [
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
            ]
            for i, peak in enumerate(self.raman.fitted_peaks):
                color = colors[i % len(colors)]
                peak_pos = peak["position"]
                peak_amp = peak["intensity"]
                peak_width = peak["width"]
                model = peak["model"]

                # Plot peak marker
                self.ax.plot(peak_pos, peak_amp, "o", color=color, markersize=5)

                # Add annotation
                self.ax.annotate(
                    f"{peak_pos:.1f}",
                    xy=(peak_pos, peak_amp),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )

                # Plot individual peak model if desired
                # (Uncomment the section below if you want to see individual peak curves)
                """
                peak_x = np.linspace(
                    peak_pos - 3*peak_width, 
                    peak_pos + 3*peak_width, 
                    100
                )
                
                if model == "Gaussian":
                    from scipy.stats import norm
                    norm_scale = peak_amp / (norm.pdf(0) * peak_width)
                    peak_y = norm_scale * norm.pdf((peak_x - peak_pos) / peak_width)
                    self.ax.plot(peak_x, peak_y, '--', color=color, alpha=0.5, 
                               label=f'Peak {i+1} ({peak_pos:.1f})')
                
                elif model == "Lorentzian":
                    from scipy.stats import cauchy
                    cauchy_scale = peak_amp / (cauchy.pdf(0) * peak_width)
                    peak_y = cauchy_scale * cauchy.pdf((peak_x - peak_pos) / peak_width)
                    self.ax.plot(peak_x, peak_y, '--', color=color, alpha=0.5, 
                               label=f'Peak {i+1} ({peak_pos:.1f})')
                
                elif model == "Pseudo-Voigt":
                    # Pseudo-Voigt is a linear combination of Gaussian and Lorentzian
                    eta = peak.get('eta', 0.5)
                    from scipy.stats import norm, cauchy
                    norm_scale = peak_amp / (norm.pdf(0) * peak_width)
                    cauchy_scale = peak_amp / (cauchy.pdf(0) * peak_width)
                    
                    gauss_component = norm_scale * norm.pdf((peak_x - peak_pos) / peak_width)
                    lorentz_component = cauchy_scale * cauchy.pdf((peak_x - peak_pos) / peak_width)
                    
                    peak_y = eta * gauss_component + (1-eta) * lorentz_component
                    self.ax.plot(peak_x, peak_y, '--', color=color, alpha=0.5, 
                               label=f'Peak {i+1} ({peak_pos:.1f})')
                """

            # Show fit quality if available
            if (
                hasattr(self.raman, "peak_fit_result")
                and self.raman.peak_fit_result is not None
                and "r_squared" in self.raman.peak_fit_result
            ):

                r_squared = self.raman.peak_fit_result["r_squared"]
                model = self.raman.peak_fit_result["model"]
                n_peaks = len(self.raman.fitted_peaks)

                # Add fit information to the plot
                fit_info = f"Fit: {model}, {n_peaks} peaks, R² = {r_squared:.4f}"
                self.ax.text(
                    0.02,
                    0.95,
                    fit_info,
                    transform=self.ax.transAxes,
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
                )

        # Configure plot appearance
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Raman Spectrum Analysis")
        self.ax.legend()
        self.ax.grid(True, linestyle=":", color="gray", alpha=0.6)

        # Update the canvas
        self.canvas.draw()

    def generate_correlation_heatmap(self, matches, fig, ax, canvas):
        """Generate correlation heatmap for spectral regions between query and best match."""
        if self.raman.current_spectra is None or not matches:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No data for correlation heatmap.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
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
        current_norm = (
            current_spectrum / query_max if query_max > 0 else current_spectrum
        )

        # Get best match (first in the list)
        best_match_name = matches[0][0]
        if best_match_name not in self.raman.database:
            ax.text(
                0.5,
                0.5,
                f"Best match {best_match_name} not found.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            canvas.draw()
            return ax

        # Get mineral name from metadata if available
        mineral_name = best_match_name
        if "metadata" in self.raman.database[best_match_name]:
            metadata = self.raman.database[best_match_name].get("metadata", {})
            if "NAME" in metadata and metadata["NAME"]:
                mineral_name = metadata["NAME"]

        match_data = self.raman.database[best_match_name]
        match_wavenumbers = match_data["wavenumbers"]
        match_intensities = match_data["intensities"]

        # Interpolate if needed
        if not np.array_equal(query_wavenumbers, match_wavenumbers):
            match_intensities_interp = np.interp(
                query_wavenumbers, match_wavenumbers, match_intensities
            )
        else:
            match_intensities_interp = match_intensities

        # Normalize match spectrum (0-1)
        match_max = np.max(match_intensities_interp)
        match_norm = (
            match_intensities_interp / match_max
            if match_max > 0
            else match_intensities_interp
        )

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
            indices = np.where(
                (query_wavenumbers >= start) & (query_wavenumbers <= end)
            )[0]
            if len(indices) > 1:  # Need at least 2 points for correlation
                region_current = current_norm[indices]
                region_match = match_norm[indices]

                # Calculate correlation coefficient for the region
                try:
                    if np.all(region_current == region_current[0]) or np.all(
                        region_match == region_match[0]
                    ):
                        corr = 0.0  # Undefined for constant vectors
                    else:
                        corr = np.corrcoef(region_current, region_match)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
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

            im = ax.imshow(
                regions_array, cmap=cmap, aspect="auto", vmin=0, vmax=1
            )  # Correlation typically 0-1

            # Add colorbar properly
            # Check if a colorbar already exists
            if hasattr(fig, "_colorbar"):
                fig._colorbar.remove()
            fig._colorbar = fig.colorbar(
                im, ax=ax, orientation="vertical", fraction=0.05, pad=0.1
            )
            fig._colorbar.set_label("Region Correlation Coefficient")

            # Configure axes
            ax.set_yticks([])  # No Y ticks needed for a single row

            ax.set_xticks(np.arange(len(region_labels)))
            ax.set_xticklabels(region_labels, rotation=45, ha="right", fontsize=9)

            # Add correlation values as text
            for i, score in enumerate(region_scores):
                text_color = (
                    "black" if 0.3 <= score <= 0.7 else "white"
                )  # Contrast based on score
                ax.text(
                    i,
                    0,
                    f"{score:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )

            ax.set_title(f"Spectral Region Correlation: Query vs. {mineral_name}")
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data or regions for correlation analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        fig.tight_layout()  # Adjust layout
        canvas.draw()
        return ax

    def save_overlay_plot(self, fig):
        """Save the overlay plot to a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Save Plot", f"Plot saved successfully to {file_path}")

    def plot_comparison_data(self, match_data, fig, ax):
        """Plot comparison between query spectrum and selected match.

        Parameters:
        -----------
        match_data : tuple
            Tuple containing (name, score) for the selected match
        fig : matplotlib.figure.Figure
            Figure object to plot on
        ax : matplotlib.axes.Axes
            Axes object to plot on
        """
        # Clear previous plots
        if ax is None or not ax:
            fig.clear()
            ax = fig.add_subplot(111)
        else:
            ax.clear()

        # Check if we have a valid spectrum loaded
        if self.raman.current_spectra is None:
            ax.text(
                0.5,
                0.5,
                "No spectrum loaded for comparison.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Get current spectrum (processed if available)
        if self.raman.processed_spectra is not None:
            query_spectrum = self.raman.processed_spectra
        else:
            query_spectrum = self.raman.current_spectra
        query_wavenumbers = self.raman.current_wavenumbers

        # Get match data
        match_name, score = match_data

        # Get mineral name from metadata if available
        mineral_name = match_name
        if (
            match_name in self.raman.database
            and "metadata" in self.raman.database[match_name]
        ):
            metadata = self.raman.database[match_name].get("metadata", {})
            if "NAME" in metadata and metadata["NAME"]:
                mineral_name = metadata["NAME"]

        # Check if match exists in database
        if match_name not in self.raman.database:
            ax.text(
                0.5,
                0.5,
                f"Match {mineral_name} not found.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        match_data = self.raman.database[match_name]

        # Handle both data formats (for backward compatibility)
        if "wavenumbers" in match_data and "intensities" in match_data:
            match_wavenumbers = match_data["wavenumbers"]
            match_intensities = match_data["intensities"]
        elif "wavenumber" in match_data and "intensity" in match_data:
            match_wavenumbers = match_data["wavenumber"]
            match_intensities = match_data["intensity"]
        else:
            ax.text(
                0.5,
                0.5,
                f"Match data format not recognized.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Determine if we need to normalize
        normalize = hasattr(self, "var_normalize") and self.var_normalize.get()

        # Normalize query spectrum (if requested)
        if normalize:
            query_max = np.max(query_spectrum)
            query_norm = query_spectrum / query_max if query_max > 0 else query_spectrum
        else:
            query_norm = query_spectrum

        # Interpolate match spectrum if the wavenumbers differ
        if not np.array_equal(query_wavenumbers, match_wavenumbers):
            match_intensities_interp = np.interp(
                query_wavenumbers, match_wavenumbers, match_intensities
            )
        else:
            match_intensities_interp = match_intensities

        # Normalize match spectrum (if requested)
        if normalize:
            match_max = np.max(match_intensities_interp)
            match_norm = (
                match_intensities_interp / match_max
                if match_max > 0
                else match_intensities_interp
            )
        else:
            match_norm = match_intensities_interp

        # Plot query spectrum
        ax.plot(
            query_wavenumbers, query_norm, "b-", label="Query Spectrum", linewidth=1.5
        )

        # Plot match spectrum
        ax.plot(
            query_wavenumbers,
            match_norm,
            "r-",
            label=f"{mineral_name} ({score:.2f})",
            linewidth=1.0,
            alpha=0.8,
        )

        # Add difference plot if requested
        show_diff = hasattr(self, "var_show_diff") and self.var_show_diff.get()
        if show_diff:
            diff = query_norm - match_norm
            ax.plot(query_wavenumbers, diff, "k--", label="Difference", alpha=0.6)

        # Highlight matching peaks if requested
        if hasattr(self, "var_highlight_peaks") and self.var_highlight_peaks.get():
            # Get peaks from both spectra
            query_peaks = (
                self.raman.peaks["wavenumbers"] if self.raman.peaks is not None else []
            )

            if "peaks" in match_data and match_data["peaks"] is not None:
                match_peaks = match_data["peaks"].get("wavenumbers", [])

                # Find matching peaks (within tolerance)
                tolerance = 10  # Default peak match tolerance in cm^-1
                if hasattr(self, "var_peak_tolerance"):
                    try:
                        tolerance = float(self.var_peak_tolerance.get())
                    except (ValueError, TypeError):
                        pass

                # Plot matching peaks
                for q_peak in query_peaks:
                    for m_peak in match_peaks:
                        if abs(q_peak - m_peak) <= tolerance:
                            # Mark this as a matching peak
                            q_height = np.interp(q_peak, query_wavenumbers, query_norm)
                            m_height = np.interp(m_peak, query_wavenumbers, match_norm)

                            # Draw highlights for matching peaks
                            ax.axvline(
                                x=q_peak, ymax=0.1, color="g", linestyle=":", alpha=0.5
                            )
                            ax.plot(q_peak, q_height, "go", markersize=5)
                            ax.plot(m_peak, m_height, "go", markersize=5)

                            # Optionally add peak position annotation
                            ax.annotate(
                                f"{q_peak:.1f}",
                                xy=(q_peak, q_height),
                                xytext=(0, 10),
                                textcoords="offset points",
                                ha="center",
                                fontsize=8,
                            )
                            break  # Once matched, move to next query peak

        # Configure plot appearance
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (Normalized)" if normalize else "Intensity (a.u.)")
        ax.set_title(f"Query vs. {mineral_name}")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(True, linestyle=":", color="gray", alpha=0.6)

        # Store the x-axis range for alignment with the mineral vibrations plot
        self.comparison_x_range = ax.get_xlim()

        # Make the figure tight in the plot window
        # Adjust margins to minimize whitespace - more compact around the plot
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)

        # Use tight_layout for remaining adjustments
        try:
            fig.tight_layout(pad=0.5)  # Smaller padding for tighter layout
        except:
            # If tight_layout fails, continue without it
            pass

        return ax

    def plot_correlation_data(self, match_data, fig, ax):
        """Plot correlation heatmap for spectral regions between query and selected match."""
        # Clear the entire figure first to ensure a clean start
        fig.clear()
        # Create a new axes object
        ax = fig.add_subplot(111)

        # Check if we have a valid spectrum loaded
        if self.raman.current_spectra is None:
            ax.text(
                0.5,
                0.5,
                "No data for correlation heatmap.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        # Get current spectrum (processed if available)
        if self.raman.processed_spectra is not None:
            current_spectrum = self.raman.processed_spectra
        else:
            current_spectrum = self.raman.current_spectra
        query_wavenumbers = self.raman.current_wavenumbers

        # Get match data
        match_name, score = match_data

        # Get mineral name from metadata if available
        mineral_name = match_name
        if (
            match_name in self.raman.database
            and "metadata" in self.raman.database[match_name]
        ):
            metadata = self.raman.database[match_name].get("metadata", {})
            if "NAME" in metadata and metadata["NAME"]:
                mineral_name = metadata["NAME"]

        # Check if match exists in database
        if match_name not in self.raman.database:
            ax.text(
                0.5,
                0.5,
                f"Match {mineral_name} not found.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        match_data = self.raman.database[match_name]

        # Get wavenumbers and intensities from match data - handle both formats
        if "wavenumbers" in match_data and "intensities" in match_data:
            match_wavenumbers = match_data["wavenumbers"]
            match_intensities = match_data["intensities"]
        elif "wavenumber" in match_data and "intensity" in match_data:
            match_wavenumbers = match_data["wavenumber"]
            match_intensities = match_data["intensity"]
        else:
            ax.text(
                0.5,
                0.5,
                f"Match data format not recognized.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        # Normalize query spectrum (0-1)
        query_max = np.max(current_spectrum)
        current_norm = (
            current_spectrum / query_max if query_max > 0 else current_spectrum
        )

        # Interpolate match spectrum if the wavenumbers differ
        if not np.array_equal(query_wavenumbers, match_wavenumbers):
            match_intensities_interp = np.interp(
                query_wavenumbers, match_wavenumbers, match_intensities
            )
        else:
            match_intensities_interp = match_intensities

        # Normalize match spectrum
        match_max = np.max(match_intensities_interp)
        match_norm = (
            match_intensities_interp / match_max
            if match_max > 0
            else match_intensities_interp
        )

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
            indices = np.where(
                (query_wavenumbers >= start) & (query_wavenumbers <= end)
            )[0]
            if len(indices) > 1:  # Need at least 2 points for correlation
                region_current = current_norm[indices]
                region_match = match_norm[indices]

                # Calculate correlation coefficient for the region
                try:
                    if np.all(region_current == region_current[0]) or np.all(
                        region_match == region_match[0]
                    ):
                        corr = 0.0  # Undefined for constant vectors
                    else:
                        corr = np.corrcoef(region_current, region_match)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0  # Handle potential errors

                region_scores.append(corr)
                region_labels.append(f"{label}\n({start}-{end} cm⁻¹)")

        # Plot correlation heatmap
        if region_scores:
            # Use built-in colormaps
            import matplotlib.cm as cm

            cmap = cm.RdYlGn  # Red-Yellow-Green colormap (good for correlation)
            regions_array = np.array(region_scores).reshape(1, -1)

            # Add buffer to left and right sides to match the comparison plot width
            # Also increase bottom margin to prevent x-axis label cutoff
            fig.subplots_adjust(left=0.12, right=0.95, bottom=0.28)

            # Create the heatmap without colorbar
            im = ax.imshow(
                regions_array, cmap=cmap, aspect="auto", vmin=0, vmax=1
            )  # Correlation typically 0-1

            # Safely remove existing colorbar if it exists
            try:
                if hasattr(fig, "_colorbar") and fig._colorbar is not None:
                    try:
                        fig._colorbar.remove()
                    except:
                        # If removal fails, set to None
                        fig._colorbar = None
            except:
                # If any error in checking or removing, ensure _colorbar is None
                fig._colorbar = None

            # No longer creating colorbar
            # The colorbar creation code has been removed

            # Configure axes
            ax.set_yticks([])  # No Y ticks needed for a single row

            ax.set_xticks(np.arange(len(region_labels)))
            ax.set_xticklabels(region_labels, rotation=45, ha="right", fontsize=9)

            # Add correlation values as text
            for i, score in enumerate(region_scores):
                text_color = (
                    "black" if 0.3 <= score <= 0.7 else "white"
                )  # Contrast based on score
                ax.text(
                    i,
                    0,
                    f"{score:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )

            ax.set_title(f"Spectral Region Correlation: Query vs. {mineral_name}")
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for correlation analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # We already set custom subplots_adjust, so don't use tight_layout
        # as it would override our custom spacing
        # fig.tight_layout()
        return ax

    def plot_mineral_vibrations(self, match_data, fig, ax, comparison_ax=None):
        """
        Plot vertical bars representing specific mineral-related vibrations between query and match.

        Parameters:
        -----------
        match_data : tuple
            Tuple containing (name, score) for the selected match
        fig : matplotlib.figure.Figure
            Figure object to plot on
        ax : matplotlib.axes.Axes
            Axes object to plot on
        """
        # Clear the entire figure first to ensure a clean start
        fig.clear()
        # Create a new axes object
        ax = fig.add_subplot(111)

        # Check if we have a valid spectrum loaded
        if self.raman.current_spectra is None:
            ax.text(
                0.5,
                0.5,
                "No data for mineral vibration analysis.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        # Get current spectrum (processed if available)
        if self.raman.processed_spectra is not None:
            current_spectrum = self.raman.processed_spectra
        else:
            current_spectrum = self.raman.current_spectra
        query_wavenumbers = self.raman.current_wavenumbers

        # Get match data
        match_name, score = match_data

        # Get mineral name from metadata if available
        mineral_name = match_name
        if (
            match_name in self.raman.database
            and "metadata" in self.raman.database[match_name]
        ):
            metadata = self.raman.database[match_name].get("metadata", {})
            if "NAME" in metadata and metadata["NAME"]:
                mineral_name = metadata["NAME"]

        # Check if match exists in database
        if match_name not in self.raman.database:
            ax.text(
                0.5,
                0.5,
                f"Match {mineral_name} not found.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        match_data = self.raman.database[match_name]

        # Get wavenumbers and intensities from match data - handle both formats
        if "wavenumbers" in match_data and "intensities" in match_data:
            match_wavenumbers = match_data["wavenumbers"]
            match_intensities = match_data["intensities"]
        elif "wavenumber" in match_data and "intensity" in match_data:
            match_wavenumbers = match_data["wavenumber"]
            match_intensities = match_data["intensity"]
        else:
            ax.text(
                0.5,
                0.5,
                f"Match data format not recognized.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        # Normalize query spectrum (0-1)
        query_max = np.max(current_spectrum)
        current_norm = (
            current_spectrum / query_max if query_max > 0 else current_spectrum
        )

        # Interpolate match spectrum if the wavenumbers differ
        if not np.array_equal(query_wavenumbers, match_wavenumbers):
            match_intensities_interp = np.interp(
                query_wavenumbers, match_wavenumbers, match_intensities
            )
        else:
            match_intensities_interp = match_intensities

        # Normalize match spectrum
        match_max = np.max(match_intensities_interp)
        match_norm = (
            match_intensities_interp / match_max
            if match_max > 0
            else match_intensities_interp
        )

        # Define specific mineral vibration regions with group categories
        # Extended with new mineral groups
        mineral_regions = [
            # Silicate vibrations
            ("Silicate", 450, 550, "Si-O-Si 3MR Stretch"),
            ("Silicate", 600, 680, "Si-O-Si"),
            ("Silicate", 850, 1000, "Si-O Stretch Q$^{2}$,Q$^{3}$"),
            ("Silicate", 1050, 1200, "Si-O-Si Stretch Q$^{0}$"),
            # Carbonate vibrations
            ("Carbonate", 700, 740, "CO$_3$ Bend v$_2$ "),
            ("Carbonate", 1050, 1090, "CO$_3$ Stretch v$_4$"),
            # Phosphate vibrations
            ("Phosphate", 550, 620, "PO$_4$ Bend v$_4$"),
            ("Phosphate", 950, 970, "PO$_4$ Stretch v$_1$"),
            ("Phosphate", 1030, 1080, "PO$_4$ Asym"),
            # Arsenate vibrations
            ("Arsenate", 420, 460, "AsO$_4$ Bend v$_2$"),
            ("Arsenate", 810, 855, "AsO$_4$ Stretch v$_1$"),
            ("Arsenate", 780, 880, "AsO$_3$ Stretch v$_3$"),
            # Sulfate vibrations
            ("Sulfate", 450, 500, "SO$_4$ Bend v$_2$"),
            ("Sulfate", 975, 1010, "SO$_4$ Stretch v$_1$"),
            ("Sulfate", 1100, 1150, "SO$_4$ Asym v$_3$"),
            # Oxide vibrations
            ("Oxide", 300, 350, "Metal-O Stretch"),
            ("Oxide", 400, 450, "Metal-O-Metal Bend"),
            ("Oxide", 500, 600, "M-O Lattice"),
            # Hydroxide vibrations
            ("Hydroxide", 3500, 3650, "OH Stretch"),
            ("Hydroxide", 600, 900, "M-OH Bend"),
            ("Hydroxide", 1600, 1650, "HOH Bend"),
            # Sulfide vibrations
            ("Sulfide", 300, 400, "Metal-S Stretch"),
            ("Sulfide", 200, 280, "S-S Stretch"),
            ("Sulfide", 350, 420, "M-S-M Bend"),
            # Sulfosalt vibrations
            ("Sulfosalt", 300, 360, "Sb-S Stretch"),
            ("Sulfosalt", 330, 380, "As-S Stretch"),
            ("Sulfosalt", 250, 290, "S-S Stretch"),
            # Vanadate vibrations
            ("Vanadate", 800, 860, "V-O Stretch v$_1$"),
            ("Vanadate", 780, 820, "V-O-V Asym v$_3$"),
            ("Vanadate", 400, 450, "V-O Bend v$_4$"),
            # Borate vibrations
            ("Borate", 650, 700, "BO$_3$ Bend"),
            ("Borate", 880, 950, "BO$_3$ Stretch"),
            ("Borate", 1300, 1400, "BO$_3$ Asym"),
            # Water vibrations
            ("OH/H$_2$O", 3200, 3500, "H$_2$O Stretch"),
            ("OH/H$_2$O", 1600, 1650, "H$_2$O Bend"),
            ("OH/H$_2$O", 500, 800, "H$_2$O Libration"),
            # Oxalate vibrations
            ("Oxalate", 1455, 1490, "C-O Stretch"),
            ("Oxalate", 900, 920, "C-C Stretch"),
            ("Oxalate", 850, 870, "O-C-O Bend"),
        ]

        # Calculate correlation for each region
        region_scores = []
        region_data = []

        # Dictionary to store group correlations
        group_correlations = {}
        group_weights = {}

        # Define region importance factors
        region_importance = {
            "Carbonate": 1.0,
            "Sulfate": 1.0,
            "Phosphate": 1.0,
            "Silicate": 1.0,
            "OH/H$_2$O": 0.5,  # Reduce weight of OH/H2O regions
            "Vanadate": 1.0,
            "Borate": 1.0,
            "Oxalate": 1.0,
        }
        filtered_regions = [region for region in mineral_regions if region[1] <= max(query_wavenumbers) and region[2] >= min(query_wavenumbers)]
        # First pass: calculate individual region correlations
        for group, start, end, label in filtered_regions:
            indices = np.where(
                (query_wavenumbers >= start) & (query_wavenumbers <= end)
            )[0]
            if len(indices) > 1:  # Need at least 2 points for correlation
                region_current = current_norm[indices]
                region_match = match_norm[indices]

                # Calculate correlation coefficient for the region
                try:
                    if np.all(region_current == region_current[0]) or np.all(
                        region_match == region_match[0]
                    ):
                        corr = 0.0  # Undefined for constant vectors
                    else:
                        corr = np.corrcoef(region_current, region_match)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0  # Handle potential errors
            else:
                corr = 0.0  # If this region doesn't have enough data points

            region_scores.append(corr)
            region_data.append((group, start, end, label, corr))

            # Initialize group correlation tracking
            if group not in group_correlations:
                group_correlations[group] = []
                group_weights[group] = []

            # Store correlation and weight for this region
            group_correlations[group].append(corr)

            # Calculate weight based on region width and importance factor
            width = end - start
            # Use a larger normalization factor (2000 cm^-1) and apply importance factor
            weight = (width / 2000.0) * region_importance.get(group, 1.0)
            group_weights[group].append(weight)

        # Second pass: calculate weighted group correlations
        weighted_group_scores = {}
        for group in group_correlations:
            if len(group_correlations[group]) > 0:
                # Calculate weighted average correlation for the group
                weighted_corr = np.average(
                    group_correlations[group], weights=group_weights[group]
                )
                weighted_group_scores[group] = weighted_corr

        # Update region scores with group context
        for i, (group, start, end, label, corr) in enumerate(region_data):
            # Adjust individual region score based on group correlation
            group_corr = weighted_group_scores.get(group, 0.0)
            # Weight the individual region score with the group correlation
            adjusted_corr = 0.7 * corr + 0.3 * group_corr
            region_scores[i] = adjusted_corr

        # Use a colormap for correlation values (for reference, will be used for the bars)
        import matplotlib.cm as cm

        cmap = cm.RdYlGn  # Red-Yellow-Green colormap (good for correlation)

        # Match x-axis with comparison plot if available
        if hasattr(self, "comparison_x_range") and self.comparison_x_range is not None:
            x_min, x_max = self.comparison_x_range
        else:
            # Fallback: use spectral range plus a small buffer
            x_min = min(query_wavenumbers)
            x_max = max(query_wavenumbers)
            x_range = x_max - x_min
            x_min = max(0, x_min - 0.15 * x_range)  # Ensure no negative wavenumbers
            x_max = x_max + 0.02 * x_range

        # Stretch the x-axis more to the right by extending the x_max
        x_max_extended = (
            x_max + (x_max - x_min) * 0.1
        )  # Add 10% more space to the right

        # Set the new x-axis limits
        ax.set_xlim(x_min, x_max_extended)
        ax.set_ylim(0, 1)  # Correlation values range from 0 to 1

        # Add gray dotted vertical grid lines
        ax.grid(True, axis="x", linestyle=":", color="gray", alpha=0.6)

        # Group by mineral types for better visualization
        groups = {}
        for item in region_data:
            group = item[0]
            if group not in groups:
                groups[group] = []
            groups[group].append(item)

        # Calculate average correlation for each group
        group_averages = {}
        for group_name, group_items in groups.items():
            correlations = [item[4] for item in group_items]  # Get correlation values
            group_averages[group_name] = np.mean(correlations) if correlations else 0.0

        # Define y-positions for each group - adjusted to accommodate more groups
        group_positions = {
            "Silicate": 0.94,
            "Carbonate": 0.88,
            "Phosphate": 0.82,
            "Arsenate": 0.76,
            "Sulfate": 0.70,
            "Oxide": 0.64,
            "Hydroxide": 0.58,
            "Sulfide": 0.52,
            "Sulfosalt": 0.46,
            "Vanadate": 0.40,
            "Borate": 0.34,
            "OH/H$_2$O": 0.28,
            "Oxalate": 0.22,
        }

        # Define height for bars - reduced to fit more groups
        bar_height = 0.05  # Reduced from 0.10 to 0.05

        # Create a collection of tooltip data to enable hover functionality
        self.tooltip_data = []

        # Plot each vibration region as a vertical bar and add group labels
        for group_name, group_items in groups.items():
            y_pos = group_positions.get(group_name, 0.5)
            avg_corr = group_averages[group_name]

            # Add group name with average correlation on the left side of the plot
            group_label = f"{group_name} (Avg: {avg_corr:.2f})"
            ax.text(
                x_min - 0.05 * (x_max - x_min),
                y_pos,
                group_label,
                fontsize=7,
                ha="left",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

            # Plot each bar in the group
            for _, start, end, label, corr in group_items:
                # Skip regions that are outside our x-axis limits
                if end < x_min or start > x_max_extended:
                    continue

                # Calculate bar width and position
                width = end - start

                # Get color based on correlation value
                color = cmap(corr)

                # Create bar
                rect = plt.Rectangle(
                    (start, y_pos - bar_height / 2),
                    width=width,
                    height=bar_height,
                    facecolor=color,
                    edgecolor="black",
                    alpha=0.8,
                )
                ax.add_patch(rect)

                # Add just the correlation value to the center of wider bars
                if width > 70:  # Only add text if bar is wide enough
                    ax.text(
                        start + width / 2,
                        y_pos,
                        f"{corr:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",  # Reduced font size
                        color="black" if 0.3 <= corr <= 0.7 else "white",
                    )

                # Store data for hover tooltips - include the rectangle, tooltip info, the color, and the region range
                tooltip_info = f"{group_name}: {label}\nRange: {start}-{end} cm⁻¹\nCorrelation: {corr:.2f}"
                self.tooltip_data.append((rect, tooltip_info, color, start, end))

        # Add a small color reference at the bottom of the plot instead of a vertical colorbar
        # Create a horizontal color gradient bar
        gradient_width = (
            x_max_extended - x_min
        ) * 0.7  # Make it wider to use more space
        gradient_height = 0.02
        gradient_x = (
            x_min + (x_max_extended - x_min) * 0.15
        )  # Position slightly left of center
        gradient_y = 0.03  # Near the bottom

        # Create the gradient rectangle
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax.imshow(
            gradient,
            aspect="auto",
            extent=[
                gradient_x,
                gradient_x + gradient_width,
                gradient_y - gradient_height / 2,
                gradient_y + gradient_height / 2,
            ],
            cmap=cmap,
        )

        # Add min/max labels for the color gradient ON TOP of the bar
        ax.text(
            gradient_x,
            gradient_y + gradient_height / 2 + 0.005,
            "Low Correlation (0.0)",
            ha="left",
            va="bottom",
            fontsize=6,
            color="dimgray",
        )
        ax.text(
            gradient_x + gradient_width,
            gradient_y + gradient_height / 2 + 0.005,
            "High Correlation (1.0)",
            ha="right",
            va="bottom",
            fontsize=6,
            color="dimgray",
        )

        # Set labels and title
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        # ax.set_ylabel('Mineral Groups', labelpad=20)  # Added labelpad to move label further left
        ax.set_title(f"Mineral Vibration Correlation: Query vs. {mineral_name}")

        # Remove y-axis ticks as they're not needed
        ax.set_yticks([])

        # Adjust the figure to add more whitespace above and below the plot
        fig.subplots_adjust(
            left=0.12, right=0.98, top=0.88, bottom=0.16
        )  # Increased top and bottom margins

        # Configure tooltip functionality
        self.setup_tooltip_event_handling(fig, ax, comparison_ax)

        return ax

    def setup_tooltip_event_handling(self, fig, ax, comparison_ax=None):
        """Set up mouseover tooltips for the mineral vibration plot."""
        from matplotlib.backend_bases import FigureCanvasBase

        # Tooltip annotation - starts hidden (without styling yet)
        self.tooltip = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, -70),  # Position below the cursor (-70 points in y direction)
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5", fc="white", alpha=0.9, edgecolor="black"
            ),
            arrowprops=dict(arrowstyle="->"),
            visible=False,
            fontsize=9,  # Slightly larger font for better readability
            color="navy",
        )  # Dark blue for better contrast and legibility

        def hover(event):
            # Check if we have tooltip data
            if not hasattr(self, "tooltip_data") or not self.tooltip_data:
                return

            # Check if mouse is over the axis
            if event.inaxes != ax:
                self.tooltip.set_visible(False)
                fig.canvas.draw_idle()
                # Remove overlay if mouse leaves
                if (
                    comparison_ax is not None
                    and hasattr(self, "_comparison_overlay")
                    and self._comparison_overlay is not None
                ):
                    self._comparison_overlay.remove()
                    self._comparison_overlay = None
                    comparison_ax.figure.canvas.draw_idle()
                return

            # Check if mouse is over any rectangle
            for rect, tooltip_text, color, start, end in self.tooltip_data:
                contains, _ = rect.contains(event)
                if contains:
                    # Update tooltip text and position
                    self.tooltip.set_text(tooltip_text)
                    self.tooltip.xy = (event.xdata, event.ydata)

                    # Position the tooltip below the cursor (let y positioning be handled by the xytext offset)
                    # Center it horizontally with the cursor
                    self.tooltip.xyann = (0, -70)  # Fixed position below cursor

                    # Set tooltip background color to match the rectangle with 90% opacity
                    # Extract RGB components from the color
                    r, g, b, _ = color

                    # Make the background slightly lighter for better text contrast
                    # Blend with white (70% original color, 30% white)
                    lighter_r = 0.7 * r + 0.3
                    lighter_g = 0.7 * g + 0.3
                    lighter_b = 0.7 * b + 0.3

                    self.tooltip.get_bbox_patch().set(
                        fc=(lighter_r, lighter_g, lighter_b, 0.9), ec=color
                    )

                    # Make sure tooltip is visible
                    self.tooltip.set_visible(True)
                    fig.canvas.draw_idle()
                    # Add overlay to comparison plot
                    if comparison_ax is not None:
                        if (
                            hasattr(self, "_comparison_overlay")
                            and self._comparison_overlay is not None
                        ):
                            self._comparison_overlay.remove()
                            self._comparison_overlay = None
                        self._comparison_overlay = comparison_ax.axvspan(
                            start, end, color=color, alpha=0.3, zorder=0
                        )
                        comparison_ax.figure.canvas.draw_idle()
                    return

            # If we get here, mouse is not over any rectangle
            self.tooltip.set_visible(False)
            fig.canvas.draw_idle()
            # Remove overlay if not hovering any region
            if (
                comparison_ax is not None
                and hasattr(self, "_comparison_overlay")
                and self._comparison_overlay is not None
            ):
                self._comparison_overlay.remove()
                self._comparison_overlay = None
                comparison_ax.figure.canvas.draw_idle()

        # Connect the event handlers
        fig.canvas.mpl_connect("motion_notify_event", hover)

    def display_search_results(self, matches):
        """Display search results with comparison plots and mineral vibration analysis."""
        if not matches:
            messagebox.showinfo("Search Results", "No matches found.")
            return

        # Create a new window for results
        results_window = tk.Toplevel(self.root)
        results_window.title("Search Results")
        results_window.geometry(
            "1200x950"
        )  # Increased height to accommodate more whitespace

        # Store reference to the results window
        self.results_window = results_window

        # Variable to track the metadata window
        self.metadata_window = None

        # Store the x-axis range for alignment between plots
        self.comparison_x_range = None

        # Create main container
        main_container = ttk.Frame(results_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create left panel for results list with reduced width
        left_panel = ttk.Frame(
            main_container, width=200
        )  # Reduced width from 300 to 200
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(
            False
        )  # Prevent the frame from shrinking to fit its contents

        # Create right panel for plots - will take more space
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create listbox for matches with increased height
        matches_frame = ttk.LabelFrame(left_panel, text="Matches", padding=5)
        matches_frame.pack(
            fill=tk.BOTH, expand=True, pady=5
        )  # Set to expand=True to take available space

        # Use grid for more control over the matches listbox and scrollbar
        matches_frame.columnconfigure(0, weight=1)
        matches_frame.rowconfigure(0, weight=1)

        # Add scrollbar to matches list
        scrollbar = ttk.Scrollbar(matches_frame)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Create listbox with explicitly set height of 15 items
        self.matches_listbox = tk.Listbox(
            matches_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.BROWSE,
            height=15,  # Explicitly set height to 15 lines
        )
        self.matches_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.matches_listbox.yview)

        # Add separator for visual spacing
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Create smaller overlay selection panel with reduced width
        overlay_frame = ttk.LabelFrame(left_panel, text="Overlay Selection", padding=5)
        overlay_frame.pack(fill=tk.X, pady=5)

        self.overlay_listbox = tk.Listbox(
            overlay_frame, height=8, selectmode=tk.MULTIPLE
        )
        self.overlay_listbox.pack(fill=tk.X, expand=True, side=tk.LEFT)

        overlay_scrollbar = ttk.Scrollbar(
            overlay_frame, orient=tk.VERTICAL, command=self.overlay_listbox.yview
        )
        overlay_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.overlay_listbox.config(yscrollcommand=overlay_scrollbar.set)

        # Add another separator for visual spacing
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Store original match data for reference
        self.match_data_map = {}

        # Populate listbox with matches
        for match in matches:
            if isinstance(match, tuple):
                # Handle tuple format (name, score)
                name, score = match
                # Get mineral name from metadata if available
                mineral_name = name
                if (
                    name in self.raman.database
                    and "metadata" in self.raman.database[name]
                ):
                    metadata = self.raman.database[name]["metadata"]
                    if "NAME" in metadata and metadata["NAME"]:
                        mineral_name = metadata["NAME"]

                display_text = f"{mineral_name} (Score: {score:.2f})"
                self.matches_listbox.insert(tk.END, display_text)
                # Store mapping between display text and original name
                self.match_data_map[display_text] = (name, score)
            elif isinstance(match, dict):
                # Handle dictionary format
                name = match["name"]
                score = match["score"]
                # Get mineral name from metadata if available
                mineral_name = name
                if (
                    name in self.raman.database
                    and "metadata" in self.raman.database[name]
                ):
                    metadata = self.raman.database[name]["metadata"]
                    if "NAME" in metadata and metadata["NAME"]:
                        mineral_name = metadata["NAME"]

                display_text = f"{mineral_name} (Score: {score:.2f})"
                self.matches_listbox.insert(tk.END, display_text)
                # Store mapping between display text and original name
                self.match_data_map[display_text] = (name, score)
            else:
                # Handle string format
                self.matches_listbox.insert(tk.END, str(match))
                self.match_data_map[str(match)] = (str(match), 0.0)

        # Button frames for controls and export - stacked vertically in left panel
        # Style the buttons with consistent width and padding
        button_style = {"width": 15, "padding": 3}

        # Control buttons frame
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(fill=tk.X, pady=5)

        # Add View Metadata button at the top of the controls
        metadata_button = ttk.Button(
            controls_frame,
            text="View Metadata",
            command=self.show_metadata_window,
            **button_style,
        )
        metadata_button.pack(fill=tk.X, pady=2)

        # Add to Overlay button
        add_button = ttk.Button(
            controls_frame,
            text="Add to Overlay",
            command=self.add_selected_to_overlay,
            **button_style,
        )
        add_button.pack(fill=tk.X, pady=2)

        # Remove from Overlay button
        remove_button = ttk.Button(
            controls_frame,
            text="Remove from Overlay",
            command=self.remove_from_overlay,
            **button_style,
        )
        remove_button.pack(fill=tk.X, pady=2)

        # Overlay Selected button
        overlay_button = ttk.Button(
            controls_frame,
            text="Overlay Selected",
            command=self.overlay_from_selection,
            **button_style,
        )
        overlay_button.pack(fill=tk.X, pady=2)

        # Add separator between controls and export buttons
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Export buttons frame - vertically stacked
        export_frame = ttk.Frame(left_panel)
        export_frame.pack(fill=tk.X, pady=5)

        # Export As label
        ttk.Label(export_frame, text="Export As:").pack(anchor=tk.W, padx=5)

        # Add hidden report text widget for export functionality (not shown in GUI)
        self.report_text = tk.Text(left_panel)
        self.report_text.pack_forget()  # Hide this widget

        # Export buttons stacked vertically
        pdf_button = ttk.Button(
            export_frame,
            text="PDF Export",
            command=lambda matches=matches: self.generate_match_report(
                matches, self.report_text
            )
            or self.export_report_as_pdf(self.report_text.get(1.0, tk.END), matches),
            **button_style,
        )
        pdf_button.pack(fill=tk.X, pady=2)

        txt_button = ttk.Button(
            export_frame,
            text="TXT Export",
            command=lambda matches=matches: self.generate_match_report(
                matches, self.report_text
            )
            or self.export_report_as_txt(self.report_text.get(1.0, tk.END)),
            **button_style,
        )
        txt_button.pack(fill=tk.X, pady=2)

        csv_button = ttk.Button(
            export_frame,
            text="CSV Export",
            command=lambda: self.export_results_as_csv(matches),
            **button_style,
        )
        csv_button.pack(fill=tk.X, pady=2)

        # Create comparison plot - give it more vertical space now
        comparison_frame = ttk.LabelFrame(
            right_panel, text="Comparison Plot", padding=5
        )
        comparison_frame.pack(
            fill=tk.BOTH, expand=True, pady=(0, 15)
        )  # Increased bottom padding

        fig_comp = plt.figure(figsize=(6, 5))  # Taller figure for comparison plot
        canvas_comp = FigureCanvasTkAgg(fig_comp, master=comparison_frame)
        canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar for comparison plot
        toolbar_frame = ttk.Frame(comparison_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas_comp, toolbar_frame)
        toolbar.update()

        # Create mineral vibrations analysis pane with more padding
        mineral_frame = ttk.LabelFrame(
            right_panel, text="Mineral Vibrations Analysis", padding=10
        )  # Increased padding
        mineral_frame.pack(
            fill=tk.BOTH, expand=True, pady=(15, 5)
        )  # Added top and bottom padding

        fig_mineral = plt.figure(figsize=(6, 5))  # Taller figure for mineral analysis
        canvas_mineral = FigureCanvasTkAgg(fig_mineral, master=mineral_frame)
        canvas_mineral.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=5, pady=5
        )  # Added padding inside the frame

        # Store references to figures and canvases
        self.results_figures = {
            "comparison": fig_comp,
            "mineral": fig_mineral,
            "canvas_comp": canvas_comp,
            "canvas_mineral": canvas_mineral,
        }

        # Bind the selection event
        self.matches_listbox.bind("<<ListboxSelect>>", self.on_match_select)

        # Automatically select the first match
        if self.matches_listbox.size() > 0:
            self.matches_listbox.selection_set(0)
            self.matches_listbox.see(0)
            self.on_match_select()  # Manually trigger the selection event

    def show_metadata_window(self):
        """
        Open a new window showing metadata for the currently selected match.
        If the window is already open, bring it to the front.
        """
        # Check if a match is selected
        selected_indices = self.matches_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo(
                "Selection Required", "Please select a spectrum from the list."
            )
            return

        # Create window if it doesn't exist, or focus if it does
        if (
            not hasattr(self, "metadata_window")
            or self.metadata_window is None
            or not self.metadata_window.winfo_exists()
        ):
            # Create a new window
            self.metadata_window = tk.Toplevel(self.results_window)
            self.metadata_window.title("Spectrum Metadata")
            self.metadata_window.geometry("400x500")

            # Make it transient to the results window
            self.metadata_window.transient(self.results_window)

            # Create a frame for the metadata content
            content_frame = ttk.Frame(self.metadata_window, padding=10)
            content_frame.pack(fill=tk.BOTH, expand=True)

            # Create text widget for displaying metadata
            self.metadata_text = tk.Text(content_frame, wrap=tk.WORD)
            self.metadata_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

            # Add scrollbar
            scrollbar = ttk.Scrollbar(
                content_frame, orient=tk.VERTICAL, command=self.metadata_text.yview
            )
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.metadata_text.config(yscrollcommand=scrollbar.set)

            # Create button to close the window
            ttk.Button(
                self.metadata_window, text="Close", command=self.metadata_window.destroy
            ).pack(pady=10)
        else:
            # If window exists, bring it to front
            self.metadata_window.lift()
            self.metadata_window.focus_set()

        # Update the content
        self.update_metadata_window()

    def update_metadata_window(self):
        """Update the content of the metadata window with current selection."""
        if not hasattr(self, "metadata_text") or not hasattr(self, "matches_listbox"):
            return

        # Get the selected match
        selected_indices = self.matches_listbox.curselection()
        if not selected_indices:
            return

        selected_index = selected_indices[0]
        selected_text = self.matches_listbox.get(selected_index)

        # Get the actual database name from our mapping
        if hasattr(self, "match_data_map") and selected_text in self.match_data_map:
            db_name, score = self.match_data_map[selected_text]
        else:
            # Try to parse the name from the display text
            if " (Score:" in selected_text:
                db_name = selected_text.split(" (Score:")[0]
            else:
                db_name = selected_text

        # Clear the text widget
        self.metadata_text.config(state=tk.NORMAL)
        self.metadata_text.delete(1.0, tk.END)

        # Check if the spectrum exists in the database
        if db_name in self.raman.database:
            # Get metadata
            spectrum_data = self.raman.database[db_name]
            metadata = spectrum_data.get("metadata", {})

            if metadata:
                # Add a header
                self.metadata_text.insert(tk.END, f"Metadata for: {db_name}\n")
                self.metadata_text.insert(tk.END, "-" * 40 + "\n\n")

                # Order of important metadata fields
                important_fields = [
                    "NAME",
                    "RRUFFIDS",
                    "HEY CLASSIFICATION",
                    "IDEAL CHEMISTRY",
                    "CHEMICAL FAMILY",
                    "CRYSTAL SYSTEMS",
                    "SPACE GROUPS",
                    "LOCALITY",
                    "DESCRIPTION",
                    "URL",
                    "PARAGENETIC MODES",
                    "AGE",
                    "CHEMICAL FORMULA",
                    "NOTES",
                ]

                # Display important fields first in order
                displayed_keys = set()
                for field in important_fields:
                    if field in metadata and metadata[field]:
                        self.metadata_text.insert(
                            tk.END, f"{field}: {metadata[field]}\n\n"
                        )
                        displayed_keys.add(field)

                # Display other fields
                for field, value in metadata.items():
                    if field not in displayed_keys and value:
                        self.metadata_text.insert(tk.END, f"{field}: {value}\n\n")
            else:
                self.metadata_text.insert(
                    tk.END, f"No metadata available for {db_name}."
                )
        else:
            self.metadata_text.insert(
                tk.END, f"Spectrum '{db_name}' not found in database."
            )

        # Make the text widget read-only
        self.metadata_text.config(state=tk.DISABLED)

        # Update window title with spectrum name
        if hasattr(self, "metadata_window") and self.metadata_window is not None:
            self.metadata_window.title(f"Metadata: {db_name}")

    def add_selected_to_overlay(self):
        """Add selected spectra to the overlay list."""
        selected_indices = self.matches_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo(
                "Selection Required", "Please select at least one spectrum."
            )
            return

        for index in selected_indices:
            match_text = self.matches_listbox.get(index)
            if match_text in self.match_data_map:
                self.overlay_listbox.insert(tk.END, match_text)

    def remove_from_overlay(self):
        """Remove selected spectra from the overlay list."""
        selected_indices = self.overlay_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo(
                "Selection Required", "Please select at least one spectrum."
            )
            return

        for index in selected_indices:
            self.overlay_listbox.delete(index)

    def overlay_from_selection(self):
        """Overlay selected spectra on the main plot."""
        selected_indices = self.overlay_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo(
                "Selection Required", "Please select at least one spectrum."
            )
            return

        # Get selected spectra
        selected_spectra = [
            self.match_data_map[self.matches_listbox.get(index)]
            for index in selected_indices
        ]

        # Update the main plot
        self.update_plot(include_peaks=True)

        # Create overlay window
        self.create_overlay_window(
            [i for i in range(self.overlay_listbox.size())], from_overlay_list=True
        )

    def create_overlay_window(self, selected_indices, from_overlay_list=False):
        """Create an overlay window with selected spectra."""
        if not hasattr(self, "raman") or self.raman.current_spectra is None:
            messagebox.showerror("Error", "No spectrum loaded to overlay with.")
            return

        # Create overlay window
        overlay_window = tk.Toplevel(self.root)
        overlay_window.title("Spectra Overlay")
        overlay_window.geometry("1000x800")

        # Create main container
        main_container = ttk.Frame(overlay_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create control panel
        control_frame = ttk.LabelFrame(
            main_container, text="Global Controls", padding=5
        )
        control_frame.pack(fill=tk.X, pady=5)

        # Add normalization checkbox
        normalize_var = tk.BooleanVar(value=True)
        normalize_check = ttk.Checkbutton(
            control_frame, text="Normalize Spectra", variable=normalize_var
        )
        normalize_check.pack(side=tk.LEFT, padx=20)

        # Add manual search
        search_frame = ttk.Frame(control_frame)
        search_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(search_frame, text="Search Database:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(search_frame, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_button = ttk.Button(
            search_frame,
            text="Add",
            command=lambda: self.add_spectrum_from_search(search_entry.get()),
        )
        search_button.pack(side=tk.LEFT)

        # Add save button
        save_button = ttk.Button(
            control_frame, text="Save Plot", command=lambda: self.save_overlay_plot(fig)
        )
        save_button.pack(side=tk.RIGHT, padx=5)

        # Create spectrum control panel with transparency sliders for each spectrum
        spectrum_control_frame = ttk.LabelFrame(
            main_container, text="Spectrum Controls", padding=5
        )
        spectrum_control_frame.pack(fill=tk.X, pady=5)

        # Add scrollable frame for spectrum controls
        canvas_frame = ttk.Frame(spectrum_control_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbar
        control_canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=control_canvas.yview
        )
        scrollable_frame = ttk.Frame(control_canvas)

        # Configure scrollable frame to update the canvas scrollregion when its size changes
        def configure_scroll_region(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        scrollable_frame.bind("<Configure>", configure_scroll_region)

        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)

        control_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Store transparency variables in a dictionary
        transparency_vars = {}

        # Add control for current spectrum
        current_frame = ttk.Frame(scrollable_frame)
        current_frame.pack(fill=tk.X, pady=2)

        # Color indicator
        color_indicator = ttk.Frame(current_frame, width=15, height=15)
        color_indicator.pack(side=tk.LEFT, padx=5)
        color_indicator.configure(style="CurrentSpectrum.TFrame")

        # Add style for current spectrum
        style = ttk.Style()
        style.configure("CurrentSpectrum.TFrame", background="blue")

        # Checkbox to show/hide
        show_current_var = tk.BooleanVar(value=True)
        show_current_check = ttk.Checkbutton(
            current_frame, text="Current Spectrum", variable=show_current_var
        )
        show_current_check.pack(side=tk.LEFT, padx=5)

        # Transparency slider
        ttk.Label(current_frame, text="Transparency:").pack(side=tk.LEFT, padx=5)
        current_transparency_var = tk.DoubleVar(value=0.0)  # 0.0 for full opacity
        current_transparency_scale = ttk.Scale(
            current_frame,
            from_=0.0,
            to=0.9,
            variable=current_transparency_var,
            orient=tk.HORIZONTAL,
            length=100,
        )
        current_transparency_scale.pack(side=tk.LEFT, padx=5)

        # Store in dictionary
        transparency_vars["current"] = {
            "var": current_transparency_var,
            "show": show_current_var,
            "name": "Current Spectrum",
        }

        # Get the list of spectra to overlay and create controls for each
        spectrum_names = []
        colors = plt.cm.tab10.colors  # Use matplotlib color cycle

        if from_overlay_list:
            # Get spectra from overlay_listbox
            spectrum_names = [self.overlay_listbox.get(i) for i in selected_indices]

        # Create controls for each spectrum
        for i, spectrum_text in enumerate(spectrum_names):
            spectrum_frame = ttk.Frame(scrollable_frame)
            spectrum_frame.pack(fill=tk.X, pady=2)

            # Color indicator - use color from matplotlib color cycle
            color_idx = (i + 1) % len(
                colors
            )  # +1 because index 0 is for current spectrum
            color_indicator = ttk.Frame(spectrum_frame, width=15, height=15)
            color_indicator.pack(side=tk.LEFT, padx=5)
            # Add style for this spectrum with its color
            style_name = f"Spectrum{i}.TFrame"
            style.configure(style_name, background=self._rgb_to_hex(colors[color_idx]))
            color_indicator.configure(style=style_name)

            # Extract name from the listbox text (handle both formats)
            if " (Score:" in spectrum_text:
                display_name = spectrum_text.split(" (Score:")[0]
            else:
                display_name = spectrum_text

            # Checkbox to show/hide
            show_var = tk.BooleanVar(value=True)
            show_check = ttk.Checkbutton(
                spectrum_frame, text=display_name, variable=show_var
            )
            show_check.pack(side=tk.LEFT, padx=5)

            # Transparency slider
            ttk.Label(spectrum_frame, text="Transparency:").pack(side=tk.LEFT, padx=5)
            spectrum_transparency_var = tk.DoubleVar(value=0.3)  # Default transparency
            spectrum_transparency_scale = ttk.Scale(
                spectrum_frame,
                from_=0.0,
                to=0.9,
                variable=spectrum_transparency_var,
                orient=tk.HORIZONTAL,
                length=100,
            )
            spectrum_transparency_scale.pack(side=tk.LEFT, padx=5)

            # Store in dictionary with spectrum text as key
            transparency_vars[spectrum_text] = {
                "var": spectrum_transparency_var,
                "show": show_var,
                "name": display_name,
                "color": colors[color_idx],
            }

        # Create plot area
        plot_frame = ttk.Frame(main_container)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        fig = plt.figure(figsize=(8, 6))
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # Store references
        self.overlay_fig = fig
        self.overlay_canvas = canvas
        self.overlay_transparency_vars = transparency_vars
        self.overlay_normalize = normalize_var
        self.control_canvas = control_canvas  # Store reference to the control canvas

        # Initial plot
        self.update_overlay_plot(selected_indices, from_overlay_list)

        # Create update function that captures the current state
        def update_overlay(*args):
            self.update_overlay_plot(selected_indices, from_overlay_list)

        # Bind transparency changes and checkbox changes
        for spectrum_id, controls in transparency_vars.items():
            controls["var"].trace_add("write", update_overlay)
            controls["show"].trace_add("write", update_overlay)

        # Bind normalization change
        normalize_var.trace_add("write", update_overlay)

    def _rgb_to_hex(self, rgb_tuple):
        """Convert RGB tuple (0-1 range) to hex color string."""
        rgb_int = tuple(int(x * 255) for x in rgb_tuple[:3])
        return "#{:02x}{:02x}{:02x}".format(*rgb_int)

    def add_spectrum_from_search(self, search_term):
        """Add a spectrum from the search results to the overlay."""
        if not search_term:
            return

        # Search the database
        matches = []
        for name in self.raman.database:
            if search_term.lower() in name.lower():
                matches.append(name)

            # Also search in metadata NAME field if available
            if "metadata" in self.raman.database[name]:
                metadata = self.raman.database[name].get("metadata", {})
                if "NAME" in metadata and metadata["NAME"]:
                    mineral_name = metadata["NAME"]
                    if search_term.lower() in mineral_name.lower():
                        matches.append(name)

        # Remove duplicates
        matches = list(set(matches))

        if not matches:
            messagebox.showinfo("Search", "No matches found.")
            return

        # Create selection window
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Spectrum")
        select_window.geometry("400x300")

        # Create listbox
        listbox = tk.Listbox(select_window, selectmode=tk.MULTIPLE)
        listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add matches to listbox
        for match in matches:
            # Get mineral name from metadata if available
            display_name = match
            if "metadata" in self.raman.database[match]:
                metadata = self.raman.database[match].get("metadata", {})
                if "NAME" in metadata and metadata["NAME"]:
                    display_name = metadata["NAME"]

            listbox.insert(tk.END, f"{display_name} ({match})")

        # Add select button
        ttk.Button(
            select_window,
            text="Add Selected",
            command=lambda: self.add_selected_spectra(
                listbox.curselection(), matches, select_window
            ),
        ).pack(pady=5)

    def add_selected_spectra(self, indices, matches, window=None):
        """Add selected spectra from search results to the overlay."""
        if not indices:
            return

        # Get current selections
        current_items = self.overlay_listbox.get(0, tk.END)

        # List of newly added spectra
        newly_added = []

        # Add new selections
        for idx in indices:
            match_name = matches[idx]

            # Get mineral name from metadata if available
            display_name = match_name
            if "metadata" in self.raman.database[match_name]:
                metadata = self.raman.database[match_name].get("metadata", {})
                if "NAME" in metadata and metadata["NAME"]:
                    display_name = metadata["NAME"]

            display_text = f"{display_name} (Score: N/A)"

            # Check if already in overlay
            if display_text not in current_items:
                self.overlay_listbox.insert(tk.END, display_text)
                # Add to match_data_map if not already there
                if display_text not in self.match_data_map:
                    self.match_data_map[display_text] = (match_name, 0.0)
                newly_added.append(display_text)

        # Close the selection window if provided
        if window:
            window.destroy()

        # Update the overlay plot if it exists
        if hasattr(self, "overlay_fig") and hasattr(self, "overlay_canvas"):
            # Add transparency controls for newly added spectra
            if hasattr(self, "overlay_transparency_vars") and newly_added:
                self._add_spectrum_controls(newly_added)

            # Update the plot with all items
            self.update_overlay_plot(
                range(self.overlay_listbox.size()), from_overlay_list=True
            )

    def _add_spectrum_controls(self, spectrum_names):
        """Add transparency controls for newly added spectra."""
        if not hasattr(self, "overlay_transparency_vars"):
            return

        # Find the scrollable frame - it's the first child of the control_canvas
        if (
            not hasattr(self, "control_canvas")
            or not self.control_canvas.winfo_children()
        ):
            return

        scrollable_frame = self.control_canvas.winfo_children()[0]

        # Get the style for color
        style = ttk.Style()

        # Get color palette
        colors = plt.cm.tab10.colors

        # Get number of existing controls to determine color index
        existing_count = (
            len(self.overlay_transparency_vars) - 1
        )  # -1 for the "current" spectrum

        # Create controls for each new spectrum
        for i, spectrum_text in enumerate(spectrum_names):
            # Skip if already has controls
            if spectrum_text in self.overlay_transparency_vars:
                continue

            spectrum_frame = ttk.Frame(scrollable_frame)
            spectrum_frame.pack(fill=tk.X, pady=2)

            # Color indicator - use color from matplotlib color cycle
            color_idx = (existing_count + i + 1) % len(
                colors
            )  # +1 because index 0 is for current spectrum
            color_indicator = ttk.Frame(spectrum_frame, width=15, height=15)
            color_indicator.pack(side=tk.LEFT, padx=5)

            # Add style for this spectrum with its color
            style_name = f"Spectrum{existing_count + i}.TFrame"
            style.configure(style_name, background=self._rgb_to_hex(colors[color_idx]))
            color_indicator.configure(style=style_name)

            # Extract name from the listbox text (handle both formats)
            if " (Score:" in spectrum_text:
                display_name = spectrum_text.split(" (Score:")[0]
            else:
                display_name = spectrum_text

            # Checkbox to show/hide
            show_var = tk.BooleanVar(value=True)
            show_check = ttk.Checkbutton(
                spectrum_frame, text=display_name, variable=show_var
            )
            show_check.pack(side=tk.LEFT, padx=5)

            # Transparency slider
            ttk.Label(spectrum_frame, text="Transparency:").pack(side=tk.LEFT, padx=5)
            spectrum_transparency_var = tk.DoubleVar(value=0.3)  # Default transparency
            spectrum_transparency_scale = ttk.Scale(
                spectrum_frame,
                from_=0.0,
                to=0.9,
                variable=spectrum_transparency_var,
                orient=tk.HORIZONTAL,
                length=100,
            )
            spectrum_transparency_scale.pack(side=tk.LEFT, padx=5)

            # Store in dictionary with spectrum text as key
            self.overlay_transparency_vars[spectrum_text] = {
                "var": spectrum_transparency_var,
                "show": show_var,
                "name": display_name,
                "color": colors[color_idx],
            }

            # Create update function that updates the plot
            def update_overlay(*args):
                self.update_overlay_plot(
                    range(self.overlay_listbox.size()), from_overlay_list=True
                )

            # Add trace to update plot when changed
            spectrum_transparency_var.trace_add("write", update_overlay)
            show_var.trace_add("write", update_overlay)

    def update_overlay_plot(self, selected_indices, from_overlay_list=False):
        """Update the overlay plot with current settings."""
        if not hasattr(self, "overlay_fig"):
            return

        # Store current axis limits if they exist
        xlim = None
        ylim = None
        if hasattr(self, "overlay_fig") and self.overlay_fig.axes:
            try:
                current_ax = self.overlay_fig.axes[0]
                xlim = current_ax.get_xlim()
                ylim = current_ax.get_ylim()
            except Exception:
                pass

        # Instead of clearing the figure, just clear the axes
        fig = self.overlay_fig
        if fig.axes:
            ax = fig.axes[0]
            ax.clear()
        else:
            ax = fig.add_subplot(111)

        # Store initial axis limits if not already stored
        if not hasattr(self, "_initial_axis_limits"):
            self._initial_axis_limits = None

        # Track if any items have been plotted for legend
        has_items_to_display = False

        # Get current spectrum
        if self.raman.processed_spectra is not None:
            current_intensity = self.raman.processed_spectra
            current_wavenumbers = self.raman.current_wavenumbers
        else:
            current_intensity = self.raman.current_spectra
            current_wavenumbers = self.raman.current_wavenumbers

        if current_intensity is None or current_wavenumbers is None:
            ax.text(
                0.5,
                0.5,
                "No spectrum loaded",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self.overlay_canvas.draw()
            return

        # Check if we should plot the current spectrum
        if hasattr(self, "overlay_transparency_vars"):
            transparency_vars = self.overlay_transparency_vars
            if (
                "current" in transparency_vars
                and transparency_vars["current"]["show"].get()
            ):
                # Normalize if requested
                if self.overlay_normalize.get():
                    current_max = np.max(current_intensity)
                    current_intensity = (
                        current_intensity / current_max
                        if current_max > 0
                        else current_intensity
                    )

                # Plot current spectrum with its transparency value
                alpha = (
                    1.0 - transparency_vars["current"]["var"].get()
                )  # Convert to opacity
                ax.plot(
                    current_wavenumbers,
                    current_intensity,
                    label="Current Spectrum",
                    linewidth=2,
                    color="blue",
                    alpha=alpha,
                )
                has_items_to_display = True

        # Get the list of spectra to plot
        if from_overlay_list:
            # Get spectra from overlay_listbox
            if isinstance(selected_indices, range):
                # Convert range to list for better handling
                selected_indices = list(selected_indices)

            # Get all spectrum items from overlay listbox
            spectrum_items = [self.overlay_listbox.get(i) for i in selected_indices]
        else:
            # Get spectra from matches_listbox
            spectrum_items = [self.matches_listbox.get(i) for i in selected_indices]

        # Keep track of colors used for consistent coloring
        colors = plt.cm.tab10.colors
        color_map = {}

        # Plot selected spectra from database
        for idx, match_text in enumerate(spectrum_items):
            # Skip if this spectrum should be hidden (check the show/hide checkbox)
            if hasattr(self, "overlay_transparency_vars"):
                if (
                    match_text in self.overlay_transparency_vars
                    and not self.overlay_transparency_vars[match_text]["show"].get()
                ):
                    continue

            # Extract name from the listbox text (handle both formats)
            if " (Score:" in match_text:
                match_name = match_text.split(" (Score:")[0]
            else:
                match_name = match_text

            # If we have match_data_map, use it to get the original filename
            if hasattr(self, "match_data_map") and match_text in self.match_data_map:
                file_name, _ = self.match_data_map[match_text]
            else:
                file_name = match_name

            # Get spectrum data from database
            if file_name in self.raman.database:
                db_data = self.raman.database[file_name]

                # Handle both old and new data formats
                if "spectrum" in db_data:
                    # New format
                    db_wavenumbers = db_data["spectrum"]["wavenumbers"]
                    db_intensity = db_data["spectrum"]["intensity"]
                else:
                    # Old format
                    db_wavenumbers = db_data["wavenumbers"]
                    db_intensity = db_data["intensities"]

                # Normalize if requested
                if self.overlay_normalize.get():
                    db_max = np.max(db_intensity)
                    db_intensity = db_intensity / db_max if db_max > 0 else db_intensity

                # Get display name from metadata if available
                display_name = match_name
                if "metadata" in db_data and "NAME" in db_data["metadata"]:
                    display_name = db_data["metadata"]["NAME"]

                # Get transparency and color settings
                alpha = 1.0  # Default
                color = None  # Default

                if (
                    hasattr(self, "overlay_transparency_vars")
                    and match_text in self.overlay_transparency_vars
                ):
                    # Get transparency from slider
                    transparency = self.overlay_transparency_vars[match_text][
                        "var"
                    ].get()
                    alpha = 1.0 - transparency  # Convert to opacity

                    # Get color if defined
                    if "color" in self.overlay_transparency_vars[match_text]:
                        color = self.overlay_transparency_vars[match_text]["color"]
                else:
                    # Assign a color based on index
                    color_idx = (idx + 1) % len(
                        colors
                    )  # +1 because index 0 is for current spectrum
                    color = colors[color_idx]

                # Make sure we have a color
                if color is None:
                    color_idx = (idx + 1) % len(
                        colors
                    )  # +1 because index 0 is for current spectrum
                    color = colors[color_idx]

                # Store color mapping for legend
                color_map[display_name] = color

                # Plot with individual transparency
                ax.plot(
                    db_wavenumbers,
                    db_intensity,
                    label=display_name,
                    color=color,
                    alpha=alpha,
                )
                has_items_to_display = True

        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel(
            "Intensity (Normalized)"
            if self.overlay_normalize.get()
            else "Intensity (a.u.)"
        )
        ax.set_title("Spectra Overlay")

        # Only add legend if there are items to display
        if has_items_to_display:
            ax.legend()

        ax.grid(True, linestyle=":", color="gray", alpha=0.6)

        # Draw the canvas first to establish the default view
        self.overlay_canvas.draw()

        # Store initial axis limits if not already stored
        if self._initial_axis_limits is None:
            self._initial_axis_limits = (ax.get_xlim(), ax.get_ylim())

        # Restore previous limits if they exist
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            self.overlay_canvas.draw()

        # Set up event handling for home button
        def on_limits_changed(event):
            if event.name == "xlim_changed" or event.name == "ylim_changed":
                # Get current limits
                current_xlim = ax.get_xlim()
                current_ylim = ax.get_ylim()

                # Get initial limits
                init_xlim, init_ylim = self._initial_axis_limits

                # Check if this is a home button press (limits match initial)
                if np.allclose(current_xlim, init_xlim) and np.allclose(
                    current_ylim, init_ylim
                ):
                    # Use the larger range for each axis
                    x_range = max(
                        current_xlim[1] - current_xlim[0], init_xlim[1] - init_xlim[0]
                    )
                    y_range = max(
                        current_ylim[1] - current_ylim[0], init_ylim[1] - init_ylim[0]
                    )

                    # Set new limits
                    ax.set_xlim(init_xlim[0], init_xlim[0] + x_range)
                    ax.set_ylim(init_ylim[0], init_ylim[0] + y_range)
                    self.overlay_canvas.draw()

        # Connect the event handlers
        ax.callbacks.connect("xlim_changed", on_limits_changed)
        ax.callbacks.connect("ylim_changed", on_limits_changed)

    def apply_savgol_smoothing(self):
        """Apply Savitzky-Golay smoothing to the current spectrum."""
        if self.raman.current_spectra is None:
            messagebox.showerror("Error", "No spectrum loaded.")
            return
        try:
            window_length = int(self.var_sg_window.get())
            polyorder = int(self.var_sg_poly.get())
            # Ensure window_length is odd and >= 5
            if window_length < 5 or window_length % 2 == 0:
                messagebox.showerror(
                    "Error", "Window length must be an odd integer >= 5."
                )
                return
            if polyorder < 2 or polyorder >= window_length:
                messagebox.showerror(
                    "Error",
                    "Polynomial order must be >= 2 and less than window length.",
                )
                return
            # Apply smoothing to the processed spectrum if it exists, else to the raw spectrum
            if self.raman.processed_spectra is not None:
                smoothed = savgol_filter(
                    self.raman.processed_spectra, window_length, polyorder
                )
                self.raman.processed_spectra = smoothed
            else:
                smoothed = savgol_filter(
                    self.raman.current_spectra, window_length, polyorder
                )
                self.raman.processed_spectra = smoothed
            self.update_plot(
                include_background=self.var_show_background.get(),
                include_peaks=self.var_show_peaks.get(),
            )
        except Exception as e:
            messagebox.showerror(
                "Smoothing Error", f"Failed to apply smoothing: {str(e)}"
            )

    def preview_savgol_smoothing(self):
        """Preview Savitzky-Golay smoothing on the plot without modifying the data."""
        if self.raman.current_spectra is None:
            messagebox.showerror("Error", "No spectrum loaded.")
            return
        try:
            window_length = int(self.var_sg_window.get())
            polyorder = int(self.var_sg_poly.get())
            if window_length < 5 or window_length % 2 == 0:
                messagebox.showerror(
                    "Error", "Window length must be an odd integer >= 5."
                )
                return
            if polyorder < 2 or polyorder >= window_length:
                messagebox.showerror(
                    "Error",
                    "Polynomial order must be >= 2 and less than window length.",
                )
                return
            # Use processed spectrum if it exists, else raw
            if self.raman.processed_spectra is not None:
                spectrum = self.raman.processed_spectra
            else:
                spectrum = self.raman.current_spectra
            smoothed = savgol_filter(spectrum, window_length, polyorder)
            # Plot preview: overlay smoothed spectrum in orange
            self.ax.clear()
            self.ax.plot(
                self.raman.current_wavenumbers,
                spectrum,
                color="blue",
                label="Original",
                linewidth=1.5,
            )
            self.ax.plot(
                self.raman.current_wavenumbers,
                smoothed,
                color="orange",
                label="S-G Preview",
                linewidth=1.5,
                linestyle="--",
            )
            self.ax.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax.set_ylabel("Intensity (a.u.)")
            self.ax.set_title("Savitzky-Golay Smoothing Preview")
            self.ax.legend()
            self.ax.grid(True, linestyle=":", color="gray", alpha=0.6)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror(
                "Smoothing Preview Error", f"Failed to preview smoothing: {str(e)}"
            )

    def create_advanced_analysis_tab(self):
        """Create content for the advanced analysis tab."""
        # Peak fitting button with styling
        style = ttk.Style()
        style.configure("PeakFitting.TButton", background="#4a7a96", foreground="white")
        ttk.Button(
            self.tab_peak_fitting,
            text="Open Peak Fitting Window",
            command=self.open_peak_fitting,
            style="PeakFitting.TButton",
        ).pack(fill=tk.X, pady=10, padx=5)

        # Batch peak fitting button
        ttk.Button(
            self.tab_peak_fitting,
            text="Open Batch Peak Fitting",
            command=self.open_batch_peak_fitting,
            style="PeakFitting.TButton",
        ).pack(fill=tk.X, pady=10, padx=5)

        # Add 2D Map Analysis button
        ttk.Button(
            self.tab_peak_fitting,
            text="2D Map Analysis",
            command=self.open_2d_map_analysis,
            style="PeakFitting.TButton",
        ).pack(fill=tk.X, pady=10, padx=5)

        # Add Raman Group Analysis button
        ttk.Button(
            self.tab_peak_fitting,
            text="Raman Group Analysis",
            command=self.open_raman_group_analysis,
            style="PeakFitting.TButton",
        ).pack(fill=tk.X, pady=10, padx=5)
        
        # Add Raman Polarization Analysis button
        ttk.Button(
            self.tab_peak_fitting,
            text="Raman Polarization Analysis",
            command=self.open_polarization_analysis,
            style="PeakFitting.TButton",
        ).pack(fill=tk.X, pady=10, padx=5)

    def open_polarization_analysis(self):
        """Open the Raman polarization analysis window."""
        try:
            print("Opening Polarization Analysis window...")
            from raman_polarization_analyzer_OLD import RamanPolarizationAnalyzer
            
            # Create a new top-level window
            polarization_window = tk.Toplevel(self.root)
            polarization_window.title("Raman Polarization Analysis")
            polarization_window.geometry("1200x800")
            
            # Create the analyzer instance
            analyzer = RamanPolarizationAnalyzer(polarization_window)
            
            # Store the analyzer instance in the window to prevent garbage collection
            polarization_window.analyzer = analyzer
            
            # Add a protocol to handle window close properly
            def on_window_close():
                # Clean up any resources if needed
                if hasattr(polarization_window, 'analyzer'):
                    del polarization_window.analyzer
                polarization_window.destroy()
                
            polarization_window.protocol("WM_DELETE_WINDOW", on_window_close)
            
            print("Polarization Analysis window opened successfully")
        except Exception as e:
            print(f"Error opening Polarization Analysis window: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            messagebox.showerror(
                "Error", f"Failed to open polarization analysis window: {str(e)}"
            )
            
    def open_batch_peak_fitting(self):
        """Open the batch peak fitting window."""
        try:
            from batch_peak_fitting import BatchPeakFittingWindow

            BatchPeakFittingWindow(self.root, self)
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to open batch peak fitting window: {str(e)}"
            )

    def open_2d_map_analysis(self):
        """Open the 2D Map Analysis window."""
        try:
            from map_analysis_2d import TwoDMapAnalysisWindow

            TwoDMapAnalysisWindow(self.root, self)
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to open 2D Map Analysis window: {str(e)}"
            )

    def open_raman_group_analysis(self):
        """Open the Raman Group Analysis window."""
        try:
            from raman_group_analysis import RamanGroupAnalysisWindow

            RamanGroupAnalysisWindow(self.root, self)
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to open Raman Group Analysis window: {str(e)}"
            )

    def get_mineral_display_name(self, mineral_key):
        """
        Get the display name for a mineral from the database.
        Uses the NAME field from metadata if available, otherwise falls back to the mineral key.
        """
        try:
            mineral_data = self.raman.database[mineral_key]
            if "metadata" in mineral_data and "NAME" in mineral_data["metadata"]:
                return mineral_data["metadata"]["NAME"]
        except Exception:
            pass
        return mineral_key



    def launch_mineral_database_browser(self):
        """Launch the mineral database browser within the same process."""
        try:
            # Try to use the standalone browser first (which handles DataFrame issues)
            try:
                import browse_mineral_database_standalone
                browse_mineral_database_standalone.main()
                return
            except ImportError:
                pass
                
            # Direct instantiation fallback
            from mineral_database import MineralDatabaseGUI
            
            # Create a new toplevel window for the mineral database
            db_window = tk.Toplevel(self.root)
            db_window.title("Mineral Database Browser")
            db_window.geometry("1000x800")
            
            # Create the mineral database GUI with the toplevel as parent
            db_gui = MineralDatabaseGUI(parent=db_window)
            
            # Fix for DataFrame truth value issues
            if hasattr(db_gui, 'is_standalone'):
                if hasattr(db_gui.is_standalone, 'empty'):  # It's a DataFrame
                    db_gui.is_standalone = bool(not db_gui.is_standalone.empty)
                else:
                    db_gui.is_standalone = bool(db_gui.is_standalone)
            
            # This prevents the window from being garbage collected
            db_window.db_gui = db_gui
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch mineral database browser: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
    
    def analyze_mixed_minerals(self):
        """Analyze a spectrum containing multiple minerals using simultaneous multi-component fitting with vibrational mode guidance."""
        # Create a new window
        window = tk.Toplevel(self.root)
        window.title("Mixed Mineral Analysis")
        window.geometry("1200x800")

        # Create main container
        main_frame = ttk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create left panel for controls - increase width to accommodate buttons
        control_panel = ttk.Frame(main_frame, width=350)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.pack_propagate(False)

        # Store selected minerals and their spectra
        selected_minerals = {}
        mineral_weights = {}
        # Dictionary to store display names
        mineral_display_names = {}


        # Create figure with subplots
        fig = Figure(figsize=(10, 8))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)

        # Create axes
        ax1 = fig.add_subplot(gs[0])  # Original spectrum
        ax2 = fig.add_subplot(gs[1])  # Combined fit
        ax3 = fig.add_subplot(gs[2])  # Residual
        ax4 = fig.add_subplot(gs[3])  # Vibrational mode analysis
        
        # Add more padding to all plots
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.08)





        # Create right panel for plots
        plot_panel = ttk.Frame(main_frame)
        plot_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=plot_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, plot_panel)
        toolbar.update()
        # No redundant canvas packing here



        current_spectrum = self.raman.processed_spectra
        current_wavenumbers = self.raman.current_wavenumbers

        # Fallback if processed spectrum is None or empty
        if (
            current_spectrum is None or
            current_wavenumbers is None or
            not hasattr(current_spectrum, '__len__') or
            not hasattr(current_wavenumbers, '__len__') or
            len(current_spectrum) == 0 or
            len(current_wavenumbers) == 0
        ):
            current_spectrum = self.raman.current_spectra
            current_wavenumbers = self.raman.current_wavenumbers



        # Get current spectrum data
        # current_spectrum = self.raman.processed_spectra
        # current_wavenumbers = self.raman.current_wavenumbers

        # Plot original spectrum
        ax1.plot(current_wavenumbers, current_spectrum, "b-", label="Original Spectrum")
        # No title for cleaner look
        ax1.set_xlabel("Wavenumber (cm⁻¹)")
        ax1.set_ylabel("Intensity")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()




        # Data check: Prevent plotting if data is missing
        if (
            current_spectrum is not None and current_wavenumbers is not None
            and hasattr(current_spectrum, '__len__') and hasattr(current_wavenumbers, '__len__')
            and len(current_spectrum) > 0 and len(current_wavenumbers) > 0
            and len(current_spectrum) == len(current_wavenumbers)
        ):
            ax1.plot(current_wavenumbers, current_spectrum, "b-", label="Original Spectrum")
        else:
            ax1.text(
                0.5, 0.5, "No spectrum data available",
                ha='center', va='center', transform=ax1.transAxes, fontsize=14, color='red'
            )
            print("Nothing to plot: spectrum or wavenumbers missing or mismatched.")
    


        # Store selected minerals and their spectra
        selected_minerals = {}
        mineral_weights = {}

        # Add button frame - moved before minerals frame for better visibility
        btn_frame = ttk.Frame(control_panel)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        # Use grid layout for buttons to ensure they're all visible
        ttk.Button(btn_frame, text="Add Mineral", command=lambda: add_mineral()).grid(
            row=0, column=0, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(
            btn_frame, text="Remove Mineral", command=lambda: remove_mineral()
        ).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(
            btn_frame, text="Search Residual", command=lambda: dtw_search_on_residual()
        ).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(
            btn_frame,
            text="Analyze Vibrations",
            command=lambda: update_vibration_analysis(),
        ).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Configure grid columns to have equal width

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        # Create selected minerals listbox
        minerals_frame = ttk.LabelFrame(control_panel, text="Selected Minerals")
        minerals_frame.pack(fill=tk.X, padx=5, pady=5)
        selected_minerals_listbox = tk.Listbox(minerals_frame, height=6)
        selected_minerals_listbox.pack(fill=tk.X, expand=True)

        # Add fit metrics frame to display R², RMSE, etc.
        metrics_frame = ttk.LabelFrame(control_panel, text="Fit Quality Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)

        # Grid for metrics
        ttk.Label(metrics_frame, text="Overall R²:").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        r2_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=r2_var).grid(
            row=0, column=1, sticky="e", padx=5, pady=2
        )

        ttk.Label(metrics_frame, text="RMSE:").grid(
            row=1, column=0, sticky="w", padx=5, pady=2
        )
        rmse_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=rmse_var).grid(
            row=1, column=1, sticky="e", padx=5, pady=2
        )

        # Add mineral contributions frame
        contrib_frame = ttk.LabelFrame(control_panel, text="Mineral Contributions")
        contrib_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a text widget to display mineral contributions
        contrib_text = tk.Text(contrib_frame, height=6, width=30, wrap=tk.WORD)
        contrib_text.pack(fill=tk.BOTH, expand=True)
        contrib_text.config(state=tk.DISABLED)

        
        # Define vibration groups for all operations
        vib_groups = ["Silicate", "Carbonate", "Phosphate", "Sulfate", "Hydroxide", 
                      "Arsenate", "Oxide", "Sulfide", "Sulfosalt", "Vanadate", 
                      "Borate", "OH/H₂O", "Oxalate"]
        
        # Add a frame for vibration display control
        vib_display_frame = ttk.LabelFrame(control_panel, text="Vibrational Mode Display")
        vib_display_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Create a canvas with scrollbar for the checkboxes to allow for scrolling
        checkbox_canvas = tk.Canvas(vib_display_frame, height=120, width=300)
        checkbox_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        checkbox_scrollbar = ttk.Scrollbar(vib_display_frame, orient="vertical", 
                                         command=checkbox_canvas.yview)
        checkbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure the canvas
        checkbox_canvas.configure(yscrollcommand=checkbox_scrollbar.set)
        checkbox_canvas.bind('<Configure>', 
                          lambda e: checkbox_canvas.configure(scrollregion=checkbox_canvas.bbox("all")))
        
        # Create a frame inside the canvas to hold checkboxes
        checkbox_frame = ttk.Frame(checkbox_canvas)
        checkbox_frame.columnconfigure(0, weight=1)
        checkbox_frame.columnconfigure(1, weight=1)
        checkbox_frame.columnconfigure(2, weight=1)
        checkbox_canvas.create_window((0, 0), window=checkbox_frame, anchor="nw", width=checkbox_canvas.winfo_width())

        def update_frame_width(event):
            checkbox_canvas.itemconfig(
                1,  # First item is the window
                width=event.width  # Set width to canvas width
            )
            
        checkbox_canvas.bind("<Configure>", update_frame_width)
        
        # Create variables to track checkbox states
        vib_group_vars = {}
        for i, group in enumerate(vib_groups):
            var = tk.BooleanVar(value=False)  # Default all to deselected
            vib_group_vars[group] = var
            
            # Calculate row and column for 2-column layout
            row = i // 3
            column = i % 3
            
            ttk.Checkbutton(
                checkbox_frame, 
                text=group, 
                variable=var,
                command=lambda g=group: update_vibration_analysis()
            ).grid(row=row, column=column, sticky="ew", padx=5, pady=2)
            
        # Add select all/none buttons
        select_btn_frame = ttk.Frame(vib_display_frame)
        select_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        def select_all_vibrations():
            for var in vib_group_vars.values():
                var.set(True)
            update_vibration_analysis()
            
        def deselect_all_vibrations():
            for var in vib_group_vars.values():
                var.set(False)
            update_vibration_analysis()
            
        ttk.Button(select_btn_frame, text="Select All", command=select_all_vibrations).grid(
            row=0, column=0, padx=5, pady=2, sticky="ew")
        ttk.Button(select_btn_frame, text="Deselect All", command=deselect_all_vibrations).grid(
            row=0, column=1, padx=5, pady=2, sticky="ew")
        
        select_btn_frame.columnconfigure(0, weight=1)
        select_btn_frame.columnconfigure(1, weight=1)

        # # Add a frame for vibration-based search
        # vib_search_frame = ttk.LabelFrame(
        #     control_panel, text="Search by Vibrational Group"
        # )
        # vib_search_frame.pack(fill=tk.X, padx=5, pady=5)

        # # Define the vibration groups for search
        # vib_groups = ["Silicate", "Carbonate", "Phosphate", "Sulfate", "Hydroxide", "Arsenate", "Oxide", "Sulfide", "Sulfosalt", "Vanadate", "Borate", "OH/H₂O", "Oxalate"]
        # vib_group_var = tk.StringVar()
        # vib_group_var.set(vib_groups[0])

        # # Add a dropdown for vibration group selection
        # ttk.Label(vib_search_frame, text="Vibration Group:").pack(
        #     anchor="w", padx=5, pady=2
        # )
        # ttk.Combobox(
        #     vib_search_frame,
        #     textvariable=vib_group_var,
        #     values=vib_groups,
        #     state="readonly",
        # ).pack(fill=tk.X, padx=5, pady=2)

        # # Add search button
        # ttk.Button(
        #     vib_search_frame,
        #     text="Search by Vibration",
        #     command=lambda: search_by_vibration_group(vib_group_var.get()),
        # ).pack(fill=tk.X, padx=5, pady=5)

        def update_fit():
            if not selected_minerals:
                # Clear metrics if no minerals selected
                r2_var.set("N/A")
                rmse_var.set("N/A")
                contrib_text.config(state=tk.NORMAL)
                contrib_text.delete(1.0, tk.END)
                contrib_text.config(state=tk.DISABLED)
                return

            A = np.column_stack([spectrum for spectrum in selected_minerals.values()])
            b = current_spectrum
            try:
                from scipy.optimize import nnls

                weights, residual = nnls(A, b)
                for i, mineral_name in enumerate(selected_minerals.keys()):
                    mineral_weights[mineral_name] = weights[i]
                fit = np.sum(
                    [w * s for w, s in zip(weights, selected_minerals.values())], axis=0
                )
                residual = current_spectrum - fit

                # Calculate fit quality metrics
                # R² calculation (coefficient of determination)
                ss_total = np.sum((current_spectrum - np.mean(current_spectrum)) ** 2)
                ss_residual = np.sum(residual**2)
                r_squared = 1 - (ss_residual / ss_total)

                # RMSE calculation
                rmse = np.sqrt(np.mean(residual**2))

                # Update metric variables
                r2_var.set(f"{r_squared:.4f}")
                rmse_var.set(f"{rmse:.2f}")

                # Update mineral contributions text
                contrib_text.config(state=tk.NORMAL)
                contrib_text.delete(1.0, tk.END)

                # Calculate total weight for percentage
                total_weight = sum(weights)

                # Add header
                contrib_text.insert(tk.END, "Mineral\tWeight\t%\n", "header")
                contrib_text.tag_configure("header", font=("TkDefaultFont", 9, "bold"))

                # Add each mineral's contribution
                for i, (mineral_name, weight) in enumerate(
                    zip(selected_minerals.keys(), weights)
                ):
                    percentage = (
                        (weight / total_weight) * 100 if total_weight > 0 else 0
                    )
                    display_name = self.get_mineral_display_name(mineral_name)
                    display_name = (
                        display_name[:15] + "..."
                        if len(display_name) > 15
                        else display_name
                    )
                    contrib_text.insert(
                        tk.END, f"{display_name}\t{weight:.2f}\t{percentage:.1f}%\n"
                    )

                contrib_text.config(state=tk.DISABLED)

                # Update plots
                ax2.clear()
                ax2.plot(
                    current_wavenumbers,
                    current_spectrum,
                    "b-",
                    label="Original",
                    alpha=0.5,
                )
                ax2.plot(current_wavenumbers, fit, "r-", label=f"Combined Fit (R² = {r_squared:.4f})")
                for mineral_key, spectrum in selected_minerals.items():
                    weight = mineral_weights[mineral_key]
                    display_name = self.get_mineral_display_name(mineral_key)
                    ax2.plot(
                        current_wavenumbers,
                        weight * spectrum,
                        "--",
                        label=f"{display_name} ({weight:.2f})",
                        alpha=0.7,
                    )
                # No title for cleaner look (R² value will be in legend)
                ax2.set_xlabel("Wavenumber (cm⁻¹)")
                ax2.set_ylabel("Intensity")
                ax2.grid(True, linestyle=":", alpha=0.6)
                ax2.legend()

                ax3.clear()
                ax3.plot(current_wavenumbers, residual, "g-", label="Residual")
                # No title for cleaner look
                ax3.set_xlabel("Wavenumber (cm⁻¹)")
                ax3.set_ylabel("Intensity")
                ax3.grid(True, linestyle=":", alpha=0.6)
                ax3.legend()
                canvas.draw()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update fit: {str(e)}")

        def update_vibration_analysis():
            """Update the vibrational mode analysis plot based on selected minerals."""
            if not selected_minerals:
                messagebox.showinfo("Info", "Please add at least one mineral first.")
                return

            ax4.clear()

            # Define common mineral vibrational regions
            mineral_regions = [
                # Silicate vibrations
                ("Silicate", 450, 550, "Si-O-Si 3MR Stretch"),
                ("Silicate", 600, 680, "Si-O-Si"),
                ("Silicate", 850, 1000, "Si-O Stretch Q$^{2}$,Q$^{3}$"),
                ("Silicate", 1050, 1200, "Si-O-Si Stretch Q$^{0}$"),
                # Carbonate vibrations
                ("Carbonate", 700, 740, "CO$_3$ Bend v$_2$ "),
                ("Carbonate", 1050, 1090, "CO$_3$ Stretch v$_4$"),
                # Phosphate vibrations
                ("Phosphate", 550, 620, "PO$_4$ Bend v$_4$"),
                ("Phosphate", 950, 970, "PO$_4$ Stretch v$_1$"),
                ("Phosphate", 1030, 1080, "PO$_4$ Asym"),
                # Sulfate vibrations
                ("Sulfate", 450, 500, "SO$_4$ Bend v$_2$"),
                ("Sulfate", 975, 1010, "SO$_4$ Stretch v$_1$"),
                ("Sulfate", 1100, 1150, "SO$_4$ Asym v$_3$"),
                # Arsenate vibrations
                ("Arsenate", 420, 460, "AsO$_4$ Bend v$_2$"),
                ("Arsenate", 810, 855, "AsO$_4$ Stretch v$_1$"),
                ("Arsenate", 780, 880, "AsO$_3$ Stretch v$_3$"),
                # Oxide vibrations
                ("Oxide", 300, 350, "Metal-O Stretch"),
                ("Oxide", 400, 450, "Metal-O-Metal Bend"),
                ("Oxide", 500, 600, "M-O Lattice"),
                # Hydroxide vibrations
                ("Hydroxide", 3500, 3650, "OH Stretch"),
                ("Hydroxide", 600, 900, "M-OH Bend"),
                ("Hydroxide", 1600, 1650, "HOH Bend"),
                # Sulfide vibrations
                ("Sulfide", 300, 400, "Metal-S Stretch"),
                ("Sulfide", 200, 280, "S-S Stretch"),
                ("Sulfide", 350, 420, "M-S-M Bend"),
                # Sulfosalt vibrations
                ("Sulfosalt", 300, 360, "Sb-S Stretch"),
                ("Sulfosalt", 330, 380, "As-S Stretch"),
                ("Sulfosalt", 250, 290, "S-S Stretch"),
                # Vanadate vibrations
                ("Vanadate", 800, 860, "V-O Stretch v$_1$"),
                ("Vanadate", 780, 820, "V-O-V Asym v$_3$"),
                ("Vanadate", 400, 450, "V-O Bend v$_4$"),
                # Borate vibrations
                ("Borate", 650, 700, "BO$_3$ Bend"),
                ("Borate", 880, 950, "BO$_3$ Stretch"),
                ("Borate", 1300, 1400, "BO$_3$ Asym"),
                # Water vibrations
                ("OH/H₂O", 3200, 3500, "H$_2$O Stretch"),
                ("OH/H₂O", 1600, 1650, "H$_2$O Bend"),
                ("OH/H₂O", 500, 800, "H$_2$O Libration"),
                # Oxalate vibrations
                ("Oxalate", 1455, 1490, "C-O Stretch"),
                ("Oxalate", 900, 920, "C-C Stretch"),
                ("Oxalate", 850, 870, "O-C-O Bend"),
            ]

            # Plot original spectrum with lower opacity
            ax4.plot(
                current_wavenumbers, current_spectrum, "k-", alpha=0.3, label="Original"
            )

            # Use different colors for different mineral groups
            group_colors = {
                "Silicate": "blue",
                "Carbonate": "green",
                "Phosphate": "red",
                "Sulfate": "purple",
                "Arsenate": "orange",
                "Oxide": "brown",
                "Hydroxide": "cyan",
                "Sulfide": "olive",
                "Sulfosalt": "pink",
                "Vanadate": "indigo",
                "Borate": "gold",
                "OH/H₂O": "lightblue",
                "Oxalate": "darkgreen",
            }
            # Filter mineral regions based on checkbox selection
            filtered_regions = [region for region in mineral_regions
                               if region[0] in vib_group_vars and vib_group_vars[region[0]].get()]

            # Add shaded regions for selected vibrational modes
            for group, start, end, label in filtered_regions:
                color = group_colors.get(group, "gray")
                ax4.axvspan(start, end, alpha=0.2, color=color)
                # Add text at the center of each region
                center = (start + end) / 2
                ax4.text(
                    center,
                    ax4.get_ylim()[1] * 0.9,
                    label,
                    fontsize=8,
                    ha="center",
                    va="top",
                    rotation=90,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

            # Add a legend for mineral groups - only for selected groups
            from matplotlib.patches import Patch
            
            selected_groups = [group for group in group_colors.keys() 
                              if group in vib_group_vars and vib_group_vars[group].get()]
            
            if selected_groups:
                legend_elements = [
                    Patch(facecolor=group_colors[group], alpha=0.2, label=group)
                    for group in selected_groups
                ]
                ax4.legend(handles=legend_elements, loc="upper right")

            # No title for cleaner look
            ax4.set_xlabel("Wavenumber (cm⁻¹)")
            ax4.set_ylabel("Intensity")
            ax4.grid(True, linestyle=":", alpha=0.6)

            canvas.draw()
        

        
        def add_mineral():
            
            matches = self._perform_search("correlation", 5, 0.5)
            if not matches:
                messagebox.showwarning("Warning", "No good matches found.")
                return
            select_window = tk.Toplevel(window)
            select_window.title("Select Mineral")
            select_window.geometry("400x300")
            match_frame = ttk.Frame(select_window)
            match_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            match_listbox = tk.Listbox(match_frame)
            match_listbox.pack(fill=tk.BOTH, expand=True)
            for match_name, score in matches:
                display_name = self.get_mineral_display_name(match_name)
                match_listbox.insert(tk.END, f"{display_name} (Score: {score:.2f})")    

            def on_select():
                selection = match_listbox.curselection()
                if not selection:
                    return
                match_name = matches[selection[0]][0]
                if match_name in selected_minerals:
                    messagebox.showwarning(
                        "Warning", "This mineral is already selected."
                    )
                    return
                match_data = self.raman.database[match_name]
                match_spectrum = match_data["intensities"]
                match_wavenumbers = match_data["wavenumbers"]
                match_spectrum_interp = np.interp(
                    current_wavenumbers, match_wavenumbers, match_spectrum
                )
                selected_minerals[match_name] = match_spectrum_interp
                mineral_weights[match_name] = 1.0
                display_name = self.get_mineral_display_name(match_name)
                selected_minerals_listbox.insert(tk.END, display_name)
                update_fit()
                select_window.destroy()

            ttk.Button(select_window, text="Select", command=on_select).pack(pady=5)

        def remove_mineral():
            selection = selected_minerals_listbox.curselection()
            if not selection:
                return
            display_name = selected_minerals_listbox.get(selection[0])  # Add this line

            mineral_key = None
            for key in list(selected_minerals.keys()):
                if self.get_mineral_display_name(key) == display_name:
                    mineral_key = key
                    break
            if mineral_key is not None:
                del selected_minerals[mineral_key]
                del mineral_weights[mineral_key]
                selected_minerals_listbox.delete(selection[0])
                update_fit()

        def dtw_search_on_residual():
            """Search for a second phase in the residual spectrum using DTW."""
            if not selected_minerals:
                messagebox.showwarning(
                    "Warning", "Please add at least one mineral first."
                )
                return

            # Get the residual spectrum
            A = np.column_stack([spectrum for spectrum in selected_minerals.values()])
            b = current_spectrum
            try:
                from scipy.optimize import nnls

                weights, nnls_residual = nnls(A, b)
                # Calculate the actual residual spectrum
                fit = np.sum(
                    [w * s for w, s in zip(weights, selected_minerals.values())], axis=0
                )
                residual_spectrum = current_spectrum - fit

                # Initialize matches list
                matches = []
                
                # Update residual plot to highlight that we're searching on it
                ax3.clear()
                ax3.plot(
                    current_wavenumbers,
                    residual_spectrum,
                    "g-",
                    label="Residual (Searching...)",
                    linewidth=2,
                )
                # No title for cleaner look
                ax3.set_xlabel("Wavenumber (cm⁻¹)")
                ax3.set_ylabel("Intensity")
                ax3.grid(True, linestyle=":", alpha=0.6)
                ax3.legend()
                canvas.draw()
                
                # Force immediate GUI update
                window.update()
                
                # Define function to display search results
                def display_search_results():
                    # Create selection window
                    select_window = tk.Toplevel(window)
                    select_window.title("Select Mineral from Residual")
                    select_window.geometry("400x300")
                    match_frame = ttk.Frame(select_window)
                    match_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    match_listbox = tk.Listbox(match_frame)
                    match_listbox.pack(fill=tk.BOTH, expand=True)

                    # Add a title to explain what we're doing
                    ttk.Label(
                        match_frame,
                        text="Minerals found in the residual spectrum:",
                        font=("TkDefaultFont", 9, "bold"),
                    ).pack(side=tk.TOP, pady=(0, 5))

                    for match_name, score in matches:
                        display_name = self.get_mineral_display_name(match_name)
                        match_listbox.insert(tk.END, f"{display_name} (Score: {score:.2f})")

                    def on_select():
                        selection = match_listbox.curselection()
                        if not selection:
                            return
                        match_name = matches[selection[0]][0]
                        if match_name in selected_minerals:
                            messagebox.showwarning(
                                "Warning", "This mineral is already selected."
                            )
                            return
                        match_data = self.raman.database[match_name]
                        match_spectrum = match_data["intensities"]
                        match_wavenumbers = match_data["wavenumbers"]
                        match_spectrum_interp = np.interp(
                            current_wavenumbers, match_wavenumbers, match_spectrum
                        )
                        selected_minerals[match_name] = match_spectrum_interp
                        mineral_weights[match_name] = 1.0
                        display_name = self.get_mineral_display_name(match_name)
                        selected_minerals_listbox.insert(tk.END, display_name)
                        update_fit()
                        select_window.destroy()

                    ttk.Button(select_window, text="Select", command=on_select).pack(pady=5)
                
                # Define function to process search results
                def show_results():
                    # Still no results, show warning
                    if not matches:
                        messagebox.showwarning(
                            "Warning", "No good matches found in residual."
                        )
                        # Restore original residual plot
                        ax3.clear()
                        ax3.plot(
                            current_wavenumbers, residual_spectrum, "g-", label="Residual"
                        )
                        # No title for cleaner look
                        ax3.set_xlabel("Wavenumber (cm⁻¹)")
                        ax3.set_ylabel("Intensity")
                        ax3.grid(True, linestyle=":", alpha=0.6)
                        ax3.legend()
                        canvas.draw()
                        return
                    else:
                        # Create selection window with search results
                        display_search_results()
                
                # Define the search thread function
                def search_thread_function():
                    nonlocal matches
                    # Perform the search on the residual spectrum
                    # Use correlation search which works with the spectrum parameter
                    matches = self.ml_based_search(5, 0.5, spectrum=residual_spectrum)

                    # If no results, try with a lower threshold
                    if not matches:
                        matches = self.ml_based_search(
                            5, 0.3, spectrum=residual_spectrum
                        )
                    
                    # Process results in the main thread
                    window.after(0, show_results)
                
                # Start the search thread
                import threading
                search_thread = threading.Thread(target=search_thread_function)
                search_thread.daemon = True
                search_thread.start()
                
                # Return early - the thread will handle the rest
                return
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to calculate residual: {str(e)}")
                return

            # Create selection window
            select_window = tk.Toplevel(window)
            select_window.title("Select Mineral from Residual")
            select_window.geometry("400x300")
            match_frame = ttk.Frame(select_window)
            match_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            match_listbox = tk.Listbox(match_frame)
            match_listbox.pack(fill=tk.BOTH, expand=True)

            # Add a title to explain what we're doing
            ttk.Label(
                match_frame,
                text="Minerals found in the residual spectrum:",
                font=("TkDefaultFont", 9, "bold"),
            ).pack(side=tk.TOP, pady=(0, 5))

            for match_name, score in matches:
                display_name = self.get_mineral_display_name(match_name)
                match_listbox.insert(tk.END, f"{display_name} (Score: {score:.2f})")

            def on_select():
                selection = match_listbox.curselection()
                if not selection:
                    return
                match_name = matches[selection[0]][0]
                if match_name in selected_minerals:
                    messagebox.showwarning(
                        "Warning", "This mineral is already selected."
                    )
                    return
                match_data = self.raman.database[match_name]
                match_spectrum = match_data["intensities"]
                match_wavenumbers = match_data["wavenumbers"]
                match_spectrum_interp = np.interp(
                    current_wavenumbers, match_wavenumbers, match_spectrum
                )
                selected_minerals[match_name] = match_spectrum_interp
                mineral_weights[match_name] = 1.0
                display_name = self.get_mineral_display_name(match_name)
                selected_minerals_listbox.insert(tk.END, display_name)
                update_fit()
                select_window.destroy()

                # Reset the residual plot
                ax3.clear()
                ax3.plot(current_wavenumbers, residual_spectrum, "g-", label="Residual")
                # No title for cleaner look
                ax3.set_xlabel("Wavenumber (cm⁻¹)")
                ax3.set_ylabel("Intensity")
                ax3.grid(True, linestyle=":", alpha=0.6)
                ax3.legend()
                canvas.draw()

            ttk.Button(select_window, text="Select", command=on_select).pack(pady=5)

        # def search_by_vibration_group(group_name):
        #     """Search for minerals based on characteristic vibration bands for selected group."""
        #     # TODO: need to make sure this searches the Hey Classification instead of vibrational mode
        #     # Define vibration ranges for each group
        #     vibration_bands = {
        #         "Silicate": [(450, 550), (600, 680), (850, 1000), (1050, 1200)],
        #         "Carbonate": [(700, 740), (1050, 1090)],
        #         "Phosphate": [(550, 620), (950, 970), (1030, 1080)],
        #         "Sulfate": [(450, 500), (975, 1010), (1100, 1150)],
        #         "Hydroxide": [(3500, 3650), (600, 900), (1600, 1650)],
        #         "Arsenate": [(420, 460), (810, 855), (780, 880)],
        #         "Oxide": [(300, 350), (400, 450), (500, 600)],
        #         "Sulfide": [(300, 400), (200, 280), (350, 420)],
        #         "Sulfosalt": [(300, 360), (330, 380), (250, 290)],
        #         "Vanadate": [(800, 860), (780, 820), (400, 450)],
        #         "Borate": [(650, 700), (880, 950), (1300, 1400)],
        #         "OH/H₂O": [(3200, 3500), (1600, 1650), (500, 800)],
        #         "Oxalate": [(1455, 1490), (900, 920), (850, 870)],
        #     }

        #     if group_name not in vibration_bands:
        #         messagebox.showwarning(
        #             "Warning", f"No vibration bands defined for {group_name}."
        #         )
        #         return

        #     # Create a peak list based on the vibration bands midpoints
        #     peak_positions = []
        #     for band_start, band_end in vibration_bands[group_name]:
        #         peak_positions.append((band_start + band_end) / 2)

        #     # Use filtered search with peak positions
        #     try:
        #         # Get a higher number of potential matches with a lower threshold
        #         matches = self._filtered_search(
        #             peak_positions=peak_positions,
        #             peak_tolerance=20,  # Higher tolerance for vibration band matching
        #             chemical_family="",  # No chemical family filter
        #             threshold=0.3,  # Lower threshold for more results
        #             hey_classification=True,
        #         )

        #         # Sort by score and limit
        #         matches.sort(key=lambda x: x[1], reverse=True)
        #         matches = matches[:10]  # Get top 10

        #         if not matches:
        #             messagebox.showwarning(
        #                 "No Matches",
        #                 f"No matches found for {group_name} vibration bands.",
        #             )
        #             return

        #         # Create selection window
        #         select_window = tk.Toplevel(window)
        #         select_window.title(f"Select {group_name} Mineral")
        #         select_window.geometry("400x300")

        #         # Add explanation label
        #         ttk.Label(
        #             select_window,
        #             text=f"Minerals with {group_name.lower()} vibration bands:",
        #             font=("TkDefaultFont", 10, "bold"),
        #         ).pack(pady=(10, 5))

        #         # Create frame for list
        #         match_frame = ttk.Frame(select_window)
        #         match_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        #         # Create listbox with scrollbar
        #         match_listbox = tk.Listbox(match_frame)
        #         scrollbar = ttk.Scrollbar(
        #             match_frame, orient=tk.VERTICAL, command=match_listbox.yview
        #         )
        #         match_listbox.config(yscrollcommand=scrollbar.set)

        #         scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        #         match_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        #         # Populate listbox
        #         for match_name, score in matches:
        #             match_listbox.insert(tk.END, f"{match_name} (Score: {score:.2f})")

        #         def on_select():
        #             selection = match_listbox.curselection()
        #             if not selection:
        #                 return
        #             match_name = matches[selection[0]][0]
        #             if match_name in selected_minerals:
        #                 messagebox.showwarning(
        #                     "Warning", "This mineral is already selected."
        #                 )
        #                 return

        #             match_data = self.raman.database[match_name]
        #             match_spectrum = match_data["intensities"]
        #             match_wavenumbers = match_data["wavenumbers"]
        #             match_spectrum_interp = np.interp(
        #                 current_wavenumbers, match_wavenumbers, match_spectrum
        #             )
        #             selected_minerals[match_name] = match_spectrum_interp
        #             mineral_weights[match_name] = 1.0
        #             selected_minerals_listbox.insert(tk.END, match_name)
        #             update_fit()
        #             select_window.destroy()

        #         # Add buttons
        #         btn_frame = ttk.Frame(select_window)
        #         btn_frame.pack(fill=tk.X, pady=5)

        #         ttk.Button(btn_frame, text="Select", command=on_select).pack(
        #             side=tk.LEFT, padx=5
        #         )
        #         ttk.Button(
        #             btn_frame, text="Cancel", command=select_window.destroy
        #         ).pack(side=tk.RIGHT, padx=5)

        #     except Exception as e:
        #         messagebox.showerror(
        #             "Search Error", f"Error searching by vibration group: {str(e)}"
        #         )


# --- Main Execution ---
def main():
    """Main function to run the application."""

    # Close any existing matplotlib figures
    plt.close("all")

    root = tk.Tk()
    # Optional: Apply a theme
    try:
        style = ttk.Style(root)
        # Available themes: 'clam', 'alt', 'default', 'classic'
        # Some systems might have 'vista', 'xpnative'
        available_themes = style.theme_names()
        # print(f"Available themes: {available_themes}")
        if "clam" in available_themes:
            style.theme_use("clam")
        elif "vista" in available_themes:
            style.theme_use("vista")
    except Exception as e:
        print(f"Could not set theme: {e}")

    app = RamanAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

