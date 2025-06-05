#!/usr/bin/env python3
"""
RamanLab Qt6 - Database Browser Window
Qt6 conversion of the database browser functionality
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
import time

# Matplotlib imports
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar

# Qt6 imports
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, QPushButton,
    QLineEdit, QListWidget, QTextEdit, QTreeWidget, QTreeWidgetItem, QSplitter,
    QGroupBox, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QFileDialog, QMessageBox, QFormLayout, QGridLayout, QFrame, QListWidgetItem,
    QProgressDialog, QSizePolicy
)
from PySide6.QtCore import Qt, QStandardPaths, QTimer, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from scipy.signal import find_peaks


class DatabaseBrowserQt6(QDialog):
    """Qt6 Database Browser Window for RamanLab."""
    
    def __init__(self, raman_db, parent=None):
        """
        Initialize the database browser.
        
        Parameters:
        -----------
        raman_db : RamanSpectraQt6
            Database instance
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.raman_db = raman_db
        self.current_spectrum = None
        self.current_peaks = None
        
        # Set window properties
        self.setWindowTitle("RamanLab - Database Browser")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Create the UI
        self.setup_ui()
        
        # Initialize data
        self.update_spectrum_list()
        self.update_stats()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_main_browser_tab()
        self.create_add_spectrum_tab()
        self.create_batch_import_tab()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def create_main_browser_tab(self):
        """Create the main database browser tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel
        self.create_left_panel(splitter)
        
        # Right panel
        self.create_right_panel(splitter)
        
        # Set splitter proportions and prevent resizing
        splitter.setSizes([350, 850])
        # Prevent automatic resizing of panels
        splitter.setChildrenCollapsible(False)
        # Keep the left panel at a fixed size
        splitter.widget(0).setMinimumWidth(350)
        splitter.widget(0).setMaximumWidth(350)
        
        self.tab_widget.addTab(tab, "Database Browser")
    
    def create_left_panel(self, parent):
        """Create the left panel with search, list, and controls."""
        left_widget = QWidget()
        # Set fixed width for the left panel
        left_widget.setFixedWidth(350)
        left_layout = QVBoxLayout(left_widget)
        
        # Search group
        search_group = QGroupBox("Search")
        search_layout = QVBoxLayout(search_group)
        
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Search spectra...")
        self.search_entry.textChanged.connect(self.perform_search)
        search_layout.addWidget(self.search_entry)
        
        clear_search_btn = QPushButton("Clear Search")
        clear_search_btn.clicked.connect(self.clear_search)
        search_layout.addWidget(clear_search_btn)
        
        left_layout.addWidget(search_group)
        
        # Controls group
        controls_group = QGroupBox("Database Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.update_spectrum_list)
        controls_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("Export Spectrum")
        export_btn.clicked.connect(self.export_spectrum)
        controls_layout.addWidget(export_btn)
        
        delete_btn = QPushButton("Delete Spectrum")
        delete_btn.clicked.connect(self.delete_spectrum)
        controls_layout.addWidget(delete_btn)
        
        save_btn = QPushButton("Save Database")
        save_btn.clicked.connect(self.save_database)
        controls_layout.addWidget(save_btn)
        
        left_layout.addWidget(controls_group)
        
        # Spectrum list group
        list_group = QGroupBox("Raman Spectra")
        list_layout = QVBoxLayout(list_group)
        
        self.spectrum_list = QListWidget()
        self.spectrum_list.currentItemChanged.connect(self.on_spectrum_select)
        list_layout.addWidget(self.spectrum_list)
        
        left_layout.addWidget(list_group)
        
        # Statistics group
        stats_group = QGroupBox("Database Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(120)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        left_layout.addWidget(stats_group)
        
        parent.addWidget(left_widget)
    
    def create_right_panel(self, parent):
        """Create the right panel with spectrum details and visualization."""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Spectrum information group
        info_group = QGroupBox("Spectrum Information")
        info_layout = QGridLayout(info_group)
        
        # Create info labels
        row = 0
        info_layout.addWidget(QLabel("Name:"), row, 0)
        self.name_label = QLabel("")
        self.name_label.setStyleSheet("border: 1px solid gray; padding: 2px; background: white;")
        # Prevent text from causing layout changes
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumHeight(40)
        self.name_label.setMinimumHeight(25)
        info_layout.addWidget(self.name_label, row, 1, 1, 2)  # Span 2 columns now
        
        info_layout.addWidget(QLabel("Data Points:"), row, 3)
        self.data_points_label = QLabel("")
        self.data_points_label.setStyleSheet("border: 1px solid gray; padding: 2px; background: white;")
        self.data_points_label.setMinimumHeight(25)
        info_layout.addWidget(self.data_points_label, row, 4)
        
        row += 1
        info_layout.addWidget(QLabel("Wavenumber Range:"), row, 0)
        self.range_label = QLabel("")
        self.range_label.setStyleSheet("border: 1px solid gray; padding: 2px; background: white;")
        self.range_label.setMinimumHeight(25)
        info_layout.addWidget(self.range_label, row, 1, 1, 2)  # Span 2 columns
        
        row += 1
        info_layout.addWidget(QLabel("Description:"), row, 0)
        self.description_label = QLabel("")
        self.description_label.setStyleSheet("border: 1px solid gray; padding: 2px; background: white;")
        # Allow description to wrap but with fixed height
        self.description_label.setWordWrap(True)
        self.description_label.setMaximumHeight(60)
        self.description_label.setMinimumHeight(25)
        info_layout.addWidget(self.description_label, row, 1, 1, 4)  # Span all remaining columns
        
        # Control buttons
        button_layout = QVBoxLayout()
        view_metadata_btn = QPushButton("View Metadata")
        view_metadata_btn.clicked.connect(self.view_metadata)
        button_layout.addWidget(view_metadata_btn)
        
        edit_metadata_btn = QPushButton("Edit Metadata")
        edit_metadata_btn.clicked.connect(self.edit_metadata)
        button_layout.addWidget(edit_metadata_btn)
        
        load_spectrum_btn = QPushButton("Load in Main App")
        load_spectrum_btn.clicked.connect(self.load_spectrum_in_main)
        button_layout.addWidget(load_spectrum_btn)
        
        info_layout.addLayout(button_layout, 0, 5, 3, 1)  # Adjust column position
        
        right_layout.addWidget(info_group)
        
        # Peak analysis group
        peak_group = QGroupBox("Peak Analysis")
        peak_layout = QVBoxLayout(peak_group)
        
        # Peak controls
        peak_controls_layout = QHBoxLayout()
        
        peak_controls_layout.addWidget(QLabel("Min Height:"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.0, 10.0)
        self.height_spin.setSingleStep(0.1)
        self.height_spin.setValue(0.1)
        peak_controls_layout.addWidget(self.height_spin)
        
        peak_controls_layout.addWidget(QLabel("Min Prominence:"))
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.0, 1.0)
        self.prominence_spin.setSingleStep(0.01)
        self.prominence_spin.setValue(0.05)
        peak_controls_layout.addWidget(self.prominence_spin)
        
        peak_controls_layout.addWidget(QLabel("Min Distance:"))
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 100)
        self.distance_spin.setValue(10)
        peak_controls_layout.addWidget(self.distance_spin)
        
        find_peaks_btn = QPushButton("Find Peaks")
        find_peaks_btn.clicked.connect(self.find_peaks)
        peak_controls_layout.addWidget(find_peaks_btn)
        
        clear_peaks_btn = QPushButton("Clear Peaks")
        clear_peaks_btn.clicked.connect(self.clear_peaks)
        peak_controls_layout.addWidget(clear_peaks_btn)
        
        peak_controls_layout.addStretch()
        peak_layout.addLayout(peak_controls_layout)
        
        # Peak comparison section
        peak_comparison_layout = QHBoxLayout()
        
        # Stored peaks column
        stored_group = QGroupBox("Stored in Database")
        stored_layout = QVBoxLayout(stored_group)
        self.stored_peaks_list = QListWidget()
        self.stored_peaks_list.setMaximumHeight(100)
        stored_layout.addWidget(self.stored_peaks_list)
        peak_comparison_layout.addWidget(stored_group)
        
        # Detected peaks column  
        detected_group = QGroupBox("Currently Detected")
        detected_layout = QVBoxLayout(detected_group)
        self.detected_peaks_list = QListWidget()
        self.detected_peaks_list.setMaximumHeight(100)
        detected_layout.addWidget(self.detected_peaks_list)
        peak_comparison_layout.addWidget(detected_group)
        
        peak_layout.addLayout(peak_comparison_layout)
        
        # Peak action buttons
        peak_action_layout = QHBoxLayout()
        
        commit_peaks_btn = QPushButton("Commit Detected Peaks to Database")
        commit_peaks_btn.clicked.connect(self.commit_peaks_to_database)
        commit_peaks_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1E7E34;
            }
        """)
        peak_action_layout.addWidget(commit_peaks_btn)
        
        show_all_peaks_btn = QPushButton("View All Database Peaks")
        show_all_peaks_btn.clicked.connect(self.show_all_database_peaks)
        show_all_peaks_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C757D;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5A6268;
            }
        """)
        peak_action_layout.addWidget(show_all_peaks_btn)
        
        peak_action_layout.addStretch()
        peak_layout.addLayout(peak_action_layout)
        
        right_layout.addWidget(peak_group)
        
        # Visualization group
        viz_group = QGroupBox("Spectrum Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Plot controls
        plot_controls_layout = QHBoxLayout()
        self.show_peaks_checkbox = QCheckBox("Show Peaks")
        self.show_peaks_checkbox.setChecked(True)
        self.show_peaks_checkbox.toggled.connect(self.update_plot)
        plot_controls_layout.addWidget(self.show_peaks_checkbox)
        plot_controls_layout.addStretch()
        viz_layout.addLayout(plot_controls_layout)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, viz_group)
        
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        # Create the plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Raman Spectrum")
        self.ax.grid(True, alpha=0.3)
        
        right_layout.addWidget(viz_group)
        
        parent.addWidget(right_widget)
    
    def create_add_spectrum_tab(self):
        """Create the add spectrum tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Add a new spectrum to the database by importing from file or manual entry.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # File import group
        file_group = QGroupBox("Import from File")
        file_layout = QVBoxLayout(file_group)
        
        file_controls = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select spectrum file...")
        file_controls.addWidget(self.file_path_edit)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_spectrum_file)
        file_controls.addWidget(browse_btn)
        
        file_layout.addLayout(file_controls)
        
        preview_btn = QPushButton("Preview Spectrum")
        preview_btn.clicked.connect(self.preview_spectrum)
        file_layout.addWidget(preview_btn)
        
        layout.addWidget(file_group)
        
        # Metadata group
        metadata_group = QGroupBox("Spectrum Metadata")
        metadata_layout = QFormLayout(metadata_group)
        
        self.spectrum_name_edit = QLineEdit()
        metadata_layout.addRow("Spectrum Name:", self.spectrum_name_edit)
        
        self.mineral_name_edit = QLineEdit()
        metadata_layout.addRow("Mineral Name:", self.mineral_name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        metadata_layout.addRow("Description:", self.description_edit)
        
        self.laser_wavelength_edit = QLineEdit()
        self.laser_wavelength_edit.setPlaceholderText("e.g., 532 nm")
        metadata_layout.addRow("Laser Wavelength:", self.laser_wavelength_edit)
        
        layout.addWidget(metadata_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add to Database")
        add_btn.clicked.connect(self.add_spectrum_to_database)
        button_layout.addWidget(add_btn)
        
        clear_btn = QPushButton("Clear Fields")
        clear_btn.clicked.connect(self.clear_add_fields)
        button_layout.addWidget(clear_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Add Spectrum")
    
    def create_batch_import_tab(self):
        """Create the batch import tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Import multiple spectrum files from a directory or migrate existing database.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Database Migration Section
        migration_group = QGroupBox("Database Migration")
        migration_layout = QVBoxLayout(migration_group)
        
        migration_info = QLabel(
            "If you have an existing raman_database.pkl file from the Tkinter version, "
            "use this option to migrate it to the Qt6 format. The migration will "
            "preserve all your spectra and metadata."
        )
        migration_info.setWordWrap(True)
        migration_layout.addWidget(migration_info)
        
        migration_buttons = QHBoxLayout()
        
        migrate_btn = QPushButton("Migrate Legacy Database")
        migrate_btn.clicked.connect(self.migrate_legacy_database)
        migration_buttons.addWidget(migrate_btn)
        
        browse_pkl_btn = QPushButton("Browse for PKL File")
        browse_pkl_btn.clicked.connect(self.browse_pkl_file)
        migration_buttons.addWidget(browse_pkl_btn)
        
        migration_buttons.addStretch()
        migration_layout.addLayout(migration_buttons)
        
        # Migration status
        self.migration_status_label = QLabel("Ready to migrate database...")
        migration_layout.addWidget(self.migration_status_label)
        
        layout.addWidget(migration_group)
        
        # Database Export/Import Section
        export_import_group = QGroupBox("Database Export/Import")
        export_import_layout = QVBoxLayout(export_import_group)
        
        export_import_info = QLabel(
            "Export your migrated database for distribution, or import a pre-migrated "
            "database to avoid manual migration on new installations."
        )
        export_import_info.setWordWrap(True)
        export_import_layout.addWidget(export_import_info)
        
        export_import_buttons = QHBoxLayout()
        
        export_db_btn = QPushButton("Export Database")
        export_db_btn.clicked.connect(self.export_database)
        export_import_buttons.addWidget(export_db_btn)
        
        import_db_btn = QPushButton("Import Database")
        import_db_btn.clicked.connect(self.import_database)
        export_import_buttons.addWidget(import_db_btn)
        
        export_import_buttons.addStretch()
        export_import_layout.addLayout(export_import_buttons)
        
        # Export/Import status
        self.export_import_status_label = QLabel("Ready for database export/import operations...")
        export_import_layout.addWidget(self.export_import_status_label)
        
        layout.addWidget(export_import_group)
        
        # Directory selection for batch import
        dir_group = QGroupBox("Batch Import from Directory")
        dir_layout = QVBoxLayout(dir_group)
        
        dir_controls = QHBoxLayout()
        self.batch_dir_edit = QLineEdit()
        self.batch_dir_edit.setPlaceholderText("Select directory containing spectrum files...")
        dir_controls.addWidget(self.batch_dir_edit)
        
        browse_dir_btn = QPushButton("Browse Directory")
        browse_dir_btn.clicked.connect(self.browse_batch_directory)
        dir_controls.addWidget(browse_dir_btn)
        
        dir_layout.addLayout(dir_controls)
        layout.addWidget(dir_group)
        
        # Import settings
        settings_group = QGroupBox("Import Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.file_pattern_edit = QLineEdit("*.txt")
        settings_layout.addRow("File Pattern:", self.file_pattern_edit)
        
        self.skip_existing_checkbox = QCheckBox()
        self.skip_existing_checkbox.setChecked(True)
        settings_layout.addRow("Skip Existing:", self.skip_existing_checkbox)
        
        layout.addWidget(settings_group)
        
        # Progress
        progress_group = QGroupBox("Import Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.batch_progress = QProgressBar()
        progress_layout.addWidget(self.batch_progress)
        
        self.batch_status_label = QLabel("Ready to import...")
        progress_layout.addWidget(self.batch_status_label)
        
        layout.addWidget(progress_group)
        
        # Action buttons
        batch_button_layout = QHBoxLayout()
        
        start_batch_btn = QPushButton("Start Batch Import")
        start_batch_btn.clicked.connect(self.start_batch_import)
        batch_button_layout.addWidget(start_batch_btn)
        
        cancel_batch_btn = QPushButton("Cancel Import")
        cancel_batch_btn.clicked.connect(self.cancel_batch_import)
        batch_button_layout.addWidget(cancel_batch_btn)
        
        batch_button_layout.addStretch()
        layout.addLayout(batch_button_layout)
        
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Batch Import & Migration")
    
    # Event handlers and functionality methods
    def update_spectrum_list(self):
        """Update the spectrum list."""
        self.spectrum_list.clear()
        
        # Get all spectra from database
        for name in sorted(self.raman_db.database.keys()):
            item = QListWidgetItem(name)
            self.spectrum_list.addItem(item)
    
    def on_spectrum_select(self, current, previous):
        """Handle spectrum selection."""
        if current is None:
            return
        
        spectrum_name = current.text()
        self.current_spectrum = self.raman_db.database.get(spectrum_name)
        
        if self.current_spectrum:
            # Update name label with mineral name if available, otherwise use spectrum name
            metadata = self.current_spectrum.get('metadata', {})
            mineral_name = metadata.get('mineral_name', '').strip()
            
            # NOTE: Database metadata keys are stored in ALL UPPERCASE (e.g., 'DESCRIPTION', 'NAME', 'LOCALITY')
            # Always check for uppercase versions when accessing metadata fields
            
            if mineral_name and mineral_name.lower() != 'unknown':
                # Use mineral name from metadata if available and not 'unknown'
                display_name = mineral_name
            else:
                # Try to extract mineral name from filename pattern
                # Look for pattern like "Mineral__R123456__..." and extract "Mineral"
                if '__' in spectrum_name:
                    potential_mineral = spectrum_name.split('__')[0].strip()
                    # Check if it looks like a mineral name (not empty and reasonable length)
                    if potential_mineral and len(potential_mineral) > 1 and len(potential_mineral) < 50:
                        display_name = potential_mineral
                    else:
                        display_name = spectrum_name
                else:
                    # Fall back to full spectrum name if no pattern found
                    display_name = spectrum_name
            
            self.name_label.setText(display_name)
            self.data_points_label.setText(str(len(self.current_spectrum['wavenumbers'])))
            
            # Handle description - check multiple possible keys (database uses uppercase keys)
            description = ''
            description_keys = ['description', 'Description', 'DESCRIPTION', 'desc', 'notes', 'Notes', 'comment', 'Comment']
            for key in description_keys:
                if key in metadata and metadata[key]:
                    description = str(metadata[key]).strip()
                    break
            
            if not description:
                description = 'No description'
            self.description_label.setText(description)
            
            wavenumbers = np.array(self.current_spectrum['wavenumbers'])
            self.range_label.setText(f"{wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
            
            # Clear any detected peaks when switching spectra
            self.current_peaks = None
            
            # Update peak lists to show stored peaks
            self.update_peak_lists()
            
            # Update plot
            self.update_plot()
    
    def update_plot(self):
        """Update the spectrum plot."""
        self.ax.clear()
        
        if self.current_spectrum:
            wavenumbers = np.array(self.current_spectrum['wavenumbers'])
            intensities = np.array(self.current_spectrum['intensities'])
            
            # Plot spectrum
            self.ax.plot(wavenumbers, intensities, 'b-', linewidth=1)
            
            # Plot peaks if enabled and available
            if self.show_peaks_checkbox.isChecked() and self.current_peaks is not None:
                peak_positions = wavenumbers[self.current_peaks]
                peak_intensities = intensities[self.current_peaks]
                self.ax.plot(peak_positions, peak_intensities, 'ro', markersize=6)
            
            self.ax.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax.set_ylabel("Intensity (a.u.)")
            self.ax.set_title("Raman Spectrum")
            self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def find_peaks(self):
        """Find peaks in the current spectrum."""
        if not self.current_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        intensities = np.array(self.current_spectrum['intensities'])
        wavenumbers = np.array(self.current_spectrum['wavenumbers'])
        
        # Get parameters
        height_threshold = self.height_spin.value() * np.max(intensities)
        prominence_threshold = self.prominence_spin.value() * np.max(intensities)
        distance = self.distance_spin.value()
        
        # Find peaks
        self.current_peaks, properties = find_peaks(
            intensities,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=distance
        )
        
        # Update peak lists
        self.update_peak_lists()
        
        # Update plot
        self.update_plot()
        
        print(f"DEBUG: Found {len(self.current_peaks)} peaks at wavenumbers: {wavenumbers[self.current_peaks]}")
    
    def update_peak_lists(self):
        """Update the stored and detected peak lists."""
        self.stored_peaks_list.clear()
        self.detected_peaks_list.clear()
        
        # Show stored peaks from database
        if self.current_spectrum:
            stored_peaks = self.get_stored_peaks_for_spectrum()
            for peak in stored_peaks:
                item_text = f"{peak:.1f} cm⁻¹"
                self.stored_peaks_list.addItem(item_text)
        
        # Show currently detected peaks
        if self.current_peaks is not None and self.current_spectrum:
            wavenumbers = np.array(self.current_spectrum['wavenumbers'])
            
            for peak_idx in self.current_peaks:
                item_text = f"{wavenumbers[peak_idx]:.1f} cm⁻¹"
                self.detected_peaks_list.addItem(item_text)
    
    def clear_peaks(self):
        """Clear the peaks."""
        self.current_peaks = None
        self.update_peak_lists()
        self.update_plot()

    def get_stored_peaks_for_spectrum(self):
        """Get stored peak positions for the current spectrum."""
        if not self.current_spectrum:
            return []
        
        peaks_data = self.current_spectrum.get('peaks', [])
        peak_positions = []
        
        # Handle different peak storage formats (same logic as main app)
        if isinstance(peaks_data, dict) and peaks_data.get("wavenumbers") is not None:
            # Original app format
            db_peaks = peaks_data["wavenumbers"]
            if hasattr(db_peaks, 'tolist'):
                peak_positions = db_peaks.tolist()
            else:
                peak_positions = list(db_peaks)
        elif isinstance(peaks_data, (list, tuple)):
            # Check if these are wavenumber values or indices
            spectrum_wavenumbers = self.current_spectrum.get('wavenumbers', [])
            if (len(spectrum_wavenumbers) > 0 and 
                all(isinstance(p, (int, float)) and 0 <= p < len(spectrum_wavenumbers) for p in peaks_data if p is not None)):
                # Legacy format: convert indices to wavenumbers
                spectrum_wavenumbers = np.array(spectrum_wavenumbers)
                for peak_idx in peaks_data:
                    if peak_idx is not None and 0 <= int(peak_idx) < len(spectrum_wavenumbers):
                        peak_positions.append(float(spectrum_wavenumbers[int(peak_idx)]))
            else:
                # Direct wavenumber values
                peak_positions = [float(p) for p in peaks_data if p is not None]
        
        return peak_positions

    def commit_peaks_to_database(self):
        """Commit the currently detected peaks to the database, overwriting stored peaks."""
        if not self.current_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        if self.current_peaks is None or len(self.current_peaks) == 0:
            reply = QMessageBox.question(
                self,
                "No Peaks Detected",
                "No peaks are currently detected. Do you want to clear the stored peaks for this spectrum?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            peak_wavenumbers = []
        else:
            # Convert detected peak indices to wavenumber values
            wavenumbers = np.array(self.current_spectrum['wavenumbers'])
            peak_wavenumbers = wavenumbers[self.current_peaks].tolist()
        
        # Get the spectrum name
        current_item = self.spectrum_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "No spectrum selected.")
            return
        
        spectrum_name = current_item.text()
        
        # Confirm the action
        peak_count = len(peak_wavenumbers)
        reply = QMessageBox.question(
            self,
            "Commit Peaks",
            f"This will overwrite the stored peaks for '{spectrum_name}' with {peak_count} newly detected peaks.\n\n"
            f"Peak positions: {[f'{p:.1f}' for p in peak_wavenumbers[:5]]}{'...' if peak_count > 5 else ''}\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Update the peaks in the database
                self.raman_db.database[spectrum_name]['peaks'] = peak_wavenumbers
                
                # Save the database
                self.raman_db.save_database()
                
                # Update the current spectrum object
                self.current_spectrum['peaks'] = peak_wavenumbers
                
                # Refresh the peak lists
                self.update_peak_lists()
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Successfully committed {peak_count} peaks to database for '{spectrum_name}'!"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to commit peaks to database:\n{str(e)}"
                )

    def show_all_database_peaks(self):
        """Show a dialog with all stored peaks in the database for verification."""
        if not self.raman_db.database:
            QMessageBox.information(self, "Empty Database", "No spectra in database to show peaks for.")
            return
        
        # Create dialog to show stored peaks (moved from main app)
        dialog = QDialog(self)
        dialog.setWindowTitle("Database Peak Verification")
        dialog.setMinimumSize(800, 500)
        layout = QVBoxLayout(dialog)
        
        # Create table to show peaks
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Spectrum Name", "Number of Peaks", "Peak Positions (cm⁻¹)"])
        table.horizontalHeader().setStretchLastSection(True)
        
        # Populate table with database peak info
        row = 0
        for name, data in self.raman_db.database.items():
            peaks_data = data.get('peaks', [])
            
            # Use the same logic as get_stored_peaks_for_spectrum
            peak_positions = []
            if isinstance(peaks_data, dict) and peaks_data.get("wavenumbers") is not None:
                # Original app format
                db_peaks = peaks_data["wavenumbers"]
                if hasattr(db_peaks, 'tolist'):
                    peak_positions = db_peaks.tolist()
                else:
                    peak_positions = list(db_peaks)
            elif isinstance(peaks_data, (list, tuple)):
                # Check if these are wavenumber values or indices
                spectrum_wavenumbers = data.get('wavenumbers', [])
                if (len(spectrum_wavenumbers) > 0 and 
                    all(isinstance(p, (int, float)) and 0 <= p < len(spectrum_wavenumbers) for p in peaks_data if p is not None)):
                    # Legacy format: convert indices to wavenumbers
                    spectrum_wavenumbers = np.array(spectrum_wavenumbers)
                    for peak_idx in peaks_data:
                        if peak_idx is not None and 0 <= int(peak_idx) < len(spectrum_wavenumbers):
                            peak_positions.append(float(spectrum_wavenumbers[int(peak_idx)]))
                else:
                    # Direct wavenumber values
                    peak_positions = [float(p) for p in peaks_data if p is not None]
            
            table.setRowCount(row + 1)
            
            # Spectrum name
            table.setItem(row, 0, QTableWidgetItem(name))
            
            # Number of peaks
            table.setItem(row, 1, QTableWidgetItem(str(len(peak_positions))))
            
            # Peak positions (rounded to 1 decimal)
            if peak_positions:
                peaks_str = ", ".join([f"{p:.1f}" for p in sorted(peak_positions)])
            else:
                peaks_str = "No peaks stored"
            table.setItem(row, 2, QTableWidgetItem(peaks_str))
            
            row += 1
        
        # Make table read-only
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # Info label
        info_label = QLabel(
            "This shows the peak positions currently stored in the database.\n"
            "To modify peaks: select a spectrum, detect new peaks, then click 'Commit Detected Peaks to Database'."
        )
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def perform_search(self):
        """Perform search based on the search term."""
        search_term = self.search_entry.text().lower()
        
        # Clear and repopulate list based on search
        self.spectrum_list.clear()
        
        for name in sorted(self.raman_db.database.keys()):
            if search_term in name.lower():
                item = QListWidgetItem(name)
                self.spectrum_list.addItem(item)
    
    def clear_search(self):
        """Clear the search."""
        self.search_entry.clear()
        self.update_spectrum_list()
    
    def view_metadata(self):
        """View metadata for the current spectrum."""
        if not self.current_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        metadata = self.current_spectrum.get('metadata', {})
        
        # Create metadata viewer dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Spectrum Metadata")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        
        # Format metadata display
        metadata_text = "Spectrum Metadata:\n\n"
        for key, value in metadata.items():
            metadata_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        if not metadata:
            metadata_text += "No metadata available."
        
        text_edit.setPlainText(metadata_text)
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def edit_metadata(self):
        """Edit metadata for the current spectrum."""
        if not self.current_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        current_item = self.spectrum_list.currentItem()
        if not current_item:
            return
        
        spectrum_name = current_item.text()
        metadata = self.current_spectrum.get('metadata', {}).copy()  # Work with a copy
        
        # Create metadata editor dialog
        dialog = MetadataEditorDialog(metadata, spectrum_name, self)
        
        if dialog.exec() == QDialog.Accepted:
            # Get the updated metadata
            updated_metadata = dialog.get_metadata()
            
            # Update the spectrum in the database
            self.current_spectrum['metadata'] = updated_metadata
            
            # Save to database
            try:
                self.raman_db.save_database()
                
                # Update the display
                self.on_spectrum_select(current_item, None)  # Refresh the display
                
                QMessageBox.information(self, "Success", "Metadata updated successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save metadata:\n{str(e)}")

    def load_spectrum_in_main(self):
        """Load the selected spectrum in the main application."""
        if not self.current_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        # Get parent application and load spectrum
        if self.parent():
            parent_app = self.parent()
            if hasattr(parent_app, 'current_wavenumbers'):
                parent_app.current_wavenumbers = np.array(self.current_spectrum['wavenumbers'])
                parent_app.current_intensities = np.array(self.current_spectrum['intensities'])
                parent_app.processed_intensities = parent_app.current_intensities.copy()
                parent_app.detected_peaks = np.array(self.current_spectrum['peaks']) if self.current_spectrum['peaks'] else None
                
                # Update main app plot
                parent_app.update_plot()
                
                # Update info display
                metadata = self.current_spectrum.get('metadata', {})
                info_text = f"Loaded from database: {self.name_label.text()}\n"
                info_text += f"Data points: {len(parent_app.current_wavenumbers)}\n"
                info_text += f"Wavenumber range: {parent_app.current_wavenumbers.min():.1f} - {parent_app.current_wavenumbers.max():.1f} cm⁻¹\n"
                if metadata.get('mineral_name'):
                    info_text += f"Mineral: {metadata['mineral_name']}\n"
                
                parent_app.info_text.setPlainText(info_text)
                parent_app.status_bar.showMessage(f"Loaded from database: {self.name_label.text()}")
                
                QMessageBox.information(self, "Success", f"Spectrum loaded in main application!")

    def export_spectrum(self):
        """Export the selected spectrum."""
        if not self.current_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Spectrum",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                wavenumbers = np.array(self.current_spectrum['wavenumbers'])
                intensities = np.array(self.current_spectrum['intensities'])
                data = np.column_stack([wavenumbers, intensities])
                np.savetxt(file_path, data, delimiter='\t', header='Wavenumber\tIntensity')
                QMessageBox.information(self, "Success", f"Spectrum exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export spectrum:\n{str(e)}")

    def delete_spectrum(self):
        """Delete the selected spectrum."""
        current_item = self.spectrum_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a spectrum to delete.")
            return
        
        spectrum_name = current_item.text()
        
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete '{spectrum_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.raman_db.remove_from_database(spectrum_name)
            if success:
                self.update_spectrum_list()
                self.update_stats()
                self.clear_spectrum_display()

    def clear_spectrum_display(self):
        """Clear the spectrum display."""
        self.current_spectrum = None
        self.current_peaks = None
        
        # Clear info labels
        self.name_label.setText("")
        self.data_points_label.setText("")
        self.range_label.setText("")
        self.description_label.setText("")
        
        # Clear peak table
        self.peak_table.clear()
        
        # Clear plot
        self.ax.clear()
        self.canvas.draw()

    def save_database(self):
        """Save the database."""
        success = self.raman_db.save_database()
        if success:
            QMessageBox.information(self, "Success", "Database saved successfully!")
        else:
            QMessageBox.critical(self, "Error", "Failed to save database.")

    def update_stats(self):
        """Update database statistics."""
        stats = self.raman_db.get_database_stats()
        
        stats_text = f"Database Statistics:\n\n"
        stats_text += f"Total Spectra: {stats['total_spectra']}\n"
        stats_text += f"Avg Data Points: {stats['avg_data_points']:.0f}\n"
        stats_text += f"Avg Peaks: {stats['avg_peaks']:.1f}\n"
        stats_text += f"File Size: {stats['database_size']}"
        
        self.stats_text.setPlainText(stats_text)

    # Add Spectrum Tab Methods
    def browse_spectrum_file(self):
        """Browse for spectrum file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Spectrum File",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt *.csv *.dat);;All files (*.*)"
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
            # Auto-fill spectrum name from filename
            filename = Path(file_path).stem
            self.spectrum_name_edit.setText(filename)

    def preview_spectrum(self):
        """Preview the selected spectrum file."""
        file_path = self.file_path_edit.text()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Invalid File", "Please select a valid spectrum file.")
            return
        
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 2 and data.shape[1] >= 2:
                QMessageBox.information(
                    self,
                    "Preview",
                    f"File: {Path(file_path).name}\n"
                    f"Data points: {len(data)}\n"
                    f"Wavenumber range: {data[:, 0].min():.1f} - {data[:, 0].max():.1f} cm⁻¹\n"
                    f"Intensity range: {data[:, 1].min():.2e} - {data[:, 1].max():.2e}"
                )
            else:
                QMessageBox.warning(self, "Invalid Format", "File must contain at least two columns.")
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Failed to preview file:\n{str(e)}")

    def add_spectrum_to_database(self):
        """Add spectrum to database."""
        file_path = self.file_path_edit.text()
        spectrum_name = self.spectrum_name_edit.text()
        
        if not file_path or not spectrum_name:
            QMessageBox.warning(self, "Missing Information", "Please provide file path and spectrum name.")
            return
        
        if spectrum_name in self.raman_db.database:
            QMessageBox.warning(self, "Name Exists", f"Spectrum '{spectrum_name}' already exists in database.")
            return
        
        try:
            # Load data
            data = np.loadtxt(file_path)
            if data.ndim == 2 and data.shape[1] >= 2:
                wavenumbers = data[:, 0]
                intensities = data[:, 1]
            else:
                raise ValueError("Invalid file format")
            
            # Collect metadata
            metadata = {
                'mineral_name': self.mineral_name_edit.text(),
                'description': self.description_edit.toPlainText(),
                'laser_wavelength': self.laser_wavelength_edit.text(),
                'source_file': file_path
            }
            
            # Add to database
            success = self.raman_db.add_to_database(spectrum_name, wavenumbers, intensities, metadata)
            
            if success:
                QMessageBox.information(self, "Success", f"Spectrum '{spectrum_name}' added to database!")
                self.update_spectrum_list()
                self.update_stats()
                self.clear_add_fields()
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import spectrum:\n{str(e)}")

    def clear_add_fields(self):
        """Clear all add spectrum fields."""
        self.file_path_edit.clear()
        self.spectrum_name_edit.clear()
        self.mineral_name_edit.clear()
        self.description_edit.clear()
        self.laser_wavelength_edit.clear()

    # Batch Import Methods
    def browse_batch_directory(self):
        """Browse for batch import directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        )
        
        if directory:
            self.batch_dir_edit.setText(directory)

    def start_batch_import(self):
        """Start batch import process."""
        directory = self.batch_dir_edit.text()
        if not directory or not os.path.exists(directory):
            QMessageBox.warning(self, "Invalid Directory", "Please select a valid directory.")
            return
        
        # This would implement batch import functionality
        QMessageBox.information(self, "Coming Soon", "Batch import will be implemented in the next version.")

    def cancel_batch_import(self):
        """Cancel batch import process."""
        # Implementation for canceling batch import
        pass

    def migrate_legacy_database(self):
        """Migrate legacy raman_database.pkl to Qt6 format."""
        # Look for raman_database.pkl in current directory
        legacy_db_path = "raman_database.pkl"
        
        if not os.path.exists(legacy_db_path):
            QMessageBox.warning(
                self, 
                "Legacy Database Not Found", 
                f"Could not find raman_database.pkl in the current directory.\n\n"
                f"Current directory: {os.getcwd()}\n\n"
                f"Use 'Browse for PKL File' to locate your database file manually."
            )
            return
        
        # Confirm migration
        reply = QMessageBox.question(
            self,
            "Confirm Migration",
            f"Found legacy database: {legacy_db_path}\n"
            f"Size: {os.path.getsize(legacy_db_path) / (1024*1024):.1f} MB\n\n"
            f"This will migrate the database to Qt6 format at:\n"
            f"{self.raman_db.db_path}\n\n"
            f"Any existing Qt6 database will be backed up.\n\n"
            f"Continue with migration?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self._perform_migration(legacy_db_path)

    def browse_pkl_file(self):
        """Browse for legacy database PKL file to migrate."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Legacy Raman Database File",
            os.getcwd(),
            "Pickle files (*.pkl);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        # Confirm migration
        reply = QMessageBox.question(
            self,
            "Confirm Migration",
            f"Selected database: {os.path.basename(file_path)}\n"
            f"Size: {os.path.getsize(file_path) / (1024*1024):.1f} MB\n\n"
            f"This will migrate the database to Qt6 format at:\n"
            f"{self.raman_db.db_path}\n\n"
            f"Any existing Qt6 database will be backed up.\n\n"
            f"Continue with migration?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self._perform_migration(file_path)

    def _perform_migration(self, legacy_db_path):
        """Perform the actual database migration."""
        try:
            # Update status
            self.migration_status_label.setText("Starting migration...")
            QApplication.processEvents()
            
            # Backup existing Qt6 database if it exists
            if os.path.exists(self.raman_db.db_path):
                backup_path = str(self.raman_db.db_path) + f".backup_{int(time.time())}"
                os.rename(self.raman_db.db_path, backup_path)
                self.migration_status_label.setText(f"Backed up existing database to {os.path.basename(backup_path)}")
                QApplication.processEvents()
            
            # Load legacy database
            self.migration_status_label.setText("Loading legacy database...")
            QApplication.processEvents()
            
            with open(legacy_db_path, 'rb') as f:
                legacy_db = pickle.load(f)
            
            if not isinstance(legacy_db, dict):
                raise ValueError("Legacy database is not in expected dictionary format")
            
            self.migration_status_label.setText(f"Loaded {len(legacy_db)} entries from legacy database")
            QApplication.processEvents()
            
            # Migrate entries
            success_count = 0
            error_count = 0
            
            # Show progress dialog for large databases
            if len(legacy_db) > 10:
                progress = QProgressDialog("Migrating database entries...", "Cancel", 0, len(legacy_db), self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
            else:
                progress = None
            
            for i, (name, entry) in enumerate(legacy_db.items()):
                if progress:
                    if progress.wasCanceled():
                        break
                    progress.setValue(i)
                    progress.setLabelText(f"Migrating: {name}")
                    QApplication.processEvents()
                
                try:
                    # Skip metadata entries
                    if name.startswith('__'):
                        continue
                    
                    # Extract data from legacy format
                    wavenumbers = entry.get('wavenumbers', [])
                    intensities = entry.get('intensities', [])
                    metadata = entry.get('metadata', {})
                    peaks = entry.get('peaks', [])
                    
                    # Convert to numpy arrays if needed
                    if not isinstance(wavenumbers, np.ndarray):
                        wavenumbers = np.array(wavenumbers)
                    if not isinstance(intensities, np.ndarray):
                        intensities = np.array(intensities)
                    
                    # Add timestamp if not present
                    if 'timestamp' not in metadata:
                        from datetime import datetime
                        metadata['timestamp'] = datetime.now().isoformat()
                    
                    # Add to Qt6 database
                    success = self.raman_db.add_to_database(
                        name=name,
                        wavenumbers=wavenumbers,
                        intensities=intensities,
                        metadata=metadata,
                        peaks=peaks
                    )
                    
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    print(f"Error migrating {name}: {str(e)}")
            
            if progress:
                progress.close()
            
            # Update UI
            self.update_spectrum_list()
            self.update_stats()
            
            # Show success message
            message = (
                f"Database Migration Complete!\n\n"
                f"✅ Successfully migrated: {success_count} spectra\n"
                f"❌ Failed migrations: {error_count} spectra\n\n"
                f"Qt6 database location:\n{self.raman_db.db_path}\n\n"
                f"You can now use all Qt6 database features!"
            )
            
            QMessageBox.information(self, "Migration Complete", message)
            self.migration_status_label.setText(f"Migration complete: {success_count} spectra migrated successfully")
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            QMessageBox.critical(self, "Migration Error", error_msg)
            self.migration_status_label.setText(f"Migration failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def export_database(self):
        """Export the current database to a distributable file."""
        if not self.raman_db.database or len(self.raman_db.database) == 0:
            QMessageBox.warning(
                self,
                "Empty Database",
                "The database is empty. Nothing to export.\n\n"
                "Import or migrate some spectra first."
            )
            return
        
        # Get save location
        default_filename = f"RamanLab_Database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export RamanLab Database",
            os.path.join(os.path.expanduser("~"), "Desktop", default_filename),
            "SQLite Database (*.sqlite);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            self.export_import_status_label.setText("Exporting database...")
            QApplication.processEvents()
            
            # Get database stats for confirmation
            stats = self.raman_db.get_database_stats()
            
            # Confirm export
            reply = QMessageBox.question(
                self,
                "Confirm Database Export",
                f"Export database with {stats['total_spectra']} spectra?\n\n"
                f"Database size: {stats['database_size']}\n"
                f"Export to: {os.path.basename(file_path)}\n\n"
                f"This will create a complete, portable database file.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply != QMessageBox.Yes:
                self.export_import_status_label.setText("Export cancelled.")
                return
            
            # Save current database
            self.raman_db.save_database()
            
            # Copy the database file
            import shutil
            if os.path.exists(file_path):
                os.remove(file_path)
            
            shutil.copy2(self.raman_db.db_path, file_path)
            
            # Create an info file alongside the database
            info_file = file_path.replace('.sqlite', '_info.txt')
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"RamanLab Database Export\n")
                f.write(f"{'='*40}\n\n")
                f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Database Version: Qt6 SQLite Format\n\n")
                f.write(f"Database Statistics:\n")
                f.write(f"  Total Spectra: {stats['total_spectra']}\n")
                f.write(f"  Average Data Points: {stats['avg_data_points']:.0f}\n")
                f.write(f"  Average Peaks: {stats['avg_peaks']:.1f}\n")
                f.write(f"  File Size: {stats['database_size']}\n\n")
                f.write(f"Installation Instructions:\n")
                f.write(f"1. Copy this .sqlite file to your RamanLab directory\n")
                f.write(f"2. Use 'Database > Import Database' in RamanLab\n")
                f.write(f"3. Select this file to load all {stats['total_spectra']} spectra\n\n")
                f.write(f"No migration required - ready to use!\n")
            
            # Success message
            message = (
                f"Database exported successfully!\n\n"
                f"📄 Database file: {os.path.basename(file_path)}\n"
                f"📋 Info file: {os.path.basename(info_file)}\n\n"
                f"Contains {stats['total_spectra']} spectra ready for distribution.\n\n"
                f"Recipients can import this database directly without migration!"
            )
            
            QMessageBox.information(self, "Export Complete", message)
            self.export_import_status_label.setText(f"Exported {stats['total_spectra']} spectra successfully")
            
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            QMessageBox.critical(self, "Export Error", error_msg)
            self.export_import_status_label.setText(f"Export failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def import_database(self):
        """Import a pre-migrated database file."""
        # Browse for database file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import RamanLab Database",
            os.path.expanduser("~"),
            "SQLite Database (*.sqlite);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "File Error", f"File not found: {file_path}")
            return
        
        try:
            self.export_import_status_label.setText("Validating database file...")
            QApplication.processEvents()
            
            # Validate the database file by trying to read it
            import sqlite3
            with sqlite3.connect(file_path) as conn:
                cursor = conn.cursor()
                # Check if it has the expected table structure
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                if 'spectra' not in tables:
                    raise ValueError("Not a valid RamanLab database file")
                
                # Get basic stats
                cursor.execute("SELECT COUNT(*) FROM spectra")
                count = cursor.fetchone()[0]
            
            # Confirm import
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            reply = QMessageBox.question(
                self,
                "Confirm Database Import",
                f"Import database from: {os.path.basename(file_path)}\n"
                f"Contains approximately {count} spectra\n"
                f"File size: {file_size:.1f} MB\n\n"
                f"This will replace your current database.\n"
                f"Your existing database will be backed up first.\n\n"
                f"Continue with import?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No  # Default to No for safety
            )
            
            if reply != QMessageBox.Yes:
                self.export_import_status_label.setText("Import cancelled.")
                return
            
            self.export_import_status_label.setText("Backing up current database...")
            QApplication.processEvents()
            
            # Backup current database if it exists
            if os.path.exists(self.raman_db.db_path):
                backup_path = str(self.raman_db.db_path) + f".backup_{int(time.time())}"
                import shutil
                shutil.copy2(self.raman_db.db_path, backup_path)
                backup_msg = f"Current database backed up to: {os.path.basename(backup_path)}"
            else:
                backup_msg = "No existing database to backup."
            
            self.export_import_status_label.setText("Importing new database...")
            QApplication.processEvents()
            
            # Copy the new database file
            import shutil
            shutil.copy2(file_path, self.raman_db.db_path)
            
            # Reload the database in memory
            self.raman_db.load_database()
            
            # Update UI
            self.update_spectrum_list()
            self.update_stats()
            
            # Success message
            stats = self.raman_db.get_database_stats()
            message = (
                f"Database imported successfully!\n\n"
                f"✅ Imported: {stats['total_spectra']} spectra\n"
                f"✅ Database size: {stats['database_size']}\n\n"
                f"{backup_msg}\n\n"
                f"The new database is now active and ready to use!"
            )
            
            QMessageBox.information(self, "Import Complete", message)
            self.export_import_status_label.setText(f"Imported {stats['total_spectra']} spectra successfully")
            
        except Exception as e:
            error_msg = f"Import failed: {str(e)}"
            QMessageBox.critical(self, "Import Error", error_msg)
            self.export_import_status_label.setText(f"Import failed: {str(e)}")
            import traceback
            traceback.print_exc()
        

class MetadataEditorDialog(QDialog):
    """Dialog for editing spectrum metadata."""
    
    def __init__(self, metadata, spectrum_name, parent=None):
        super().__init__(parent)
        self.metadata = metadata.copy()
        self.spectrum_name = spectrum_name
        self.field_widgets = {}  # Store references to input widgets
        
        self.setWindowTitle(f"Edit Metadata - {spectrum_name}")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        self.setup_ui()
        self.populate_fields()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # Left-justify content
        
        # Title
        title = QLabel(f"Editing metadata for: {self.spectrum_name}")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignLeft)  # Left-align title
        layout.addWidget(title)
        
        # Create scrollable area for metadata fields
        from PySide6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.fields_layout = QFormLayout(scroll_widget)
        
        # Configure form layout for left-justified, wider fields
        self.fields_layout.setLabelAlignment(Qt.AlignLeft)
        self.fields_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.fields_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        
        # Set the label-to-field ratio (25% labels, 75% fields)
        self.fields_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(350)
        layout.addWidget(scroll_area)
        
        # Add new field section
        new_field_group = QGroupBox("Add New Field")
        new_field_layout = QHBoxLayout(new_field_group)
        
        new_field_layout.addWidget(QLabel("Field Name:"))
        self.new_field_name = QLineEdit()
        self.new_field_name.setPlaceholderText("Enter field name...")
        new_field_layout.addWidget(self.new_field_name)
        
        new_field_layout.addWidget(QLabel("Value:"))
        self.new_field_value = QLineEdit()
        self.new_field_value.setPlaceholderText("Enter field value...")
        new_field_layout.addWidget(self.new_field_value)
        
        add_field_btn = QPushButton("Add Field")
        add_field_btn.clicked.connect(self.add_new_field)
        new_field_layout.addWidget(add_field_btn)
        
        layout.addWidget(new_field_group)
        
        # Instructions
        instructions = QLabel(
            "• Edit existing fields directly in the form above\n"
            "• Use 'Add New Field' to add custom metadata\n"
            "• Clear field contents to effectively remove them\n"
            "• Click 'Save Changes' to update the database"
        )
        instructions.setStyleSheet("color: #666; font-size: 10px; padding: 10px;")
        instructions.setAlignment(Qt.AlignLeft)  # Left-align instructions
        layout.addWidget(instructions)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(self.accept)
        save_btn.setDefault(True)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        reset_btn = QPushButton("Reset to Original")
        reset_btn.clicked.connect(self.reset_fields)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def populate_fields(self):
        """Populate the form with existing metadata fields."""
        # Define common fields with better labels and order
        common_fields = [
            ('mineral_name', 'Mineral Name'),
            ('description', 'Description'),
            ('laser_wavelength', 'Laser Wavelength'),
            ('source_file', 'Source File'),
            ('timestamp', 'Timestamp'),
            ('sample_id', 'Sample ID'),
            ('location', 'Location/Origin'),
            ('operator', 'Operator'),
            ('instrument', 'Instrument'),
            ('acquisition_time', 'Acquisition Time'),
            ('laser_power', 'Laser Power'),
            ('integration_time', 'Integration Time'),
            ('temperature', 'Temperature'),
            ('pressure', 'Pressure'),
            ('notes', 'Additional Notes')
        ]
        
        # Add common fields first (whether they exist or not)
        for field_key, field_label in common_fields:
            value = self.metadata.get(field_key, '')
            self.add_field_widget(field_key, field_label, value)
        
        # Add any additional custom fields that aren't in the common list
        common_keys = [key for key, _ in common_fields]
        for key, value in self.metadata.items():
            if key not in common_keys:
                # Format the key for display
                display_label = key.replace('_', ' ').title()
                self.add_field_widget(key, display_label, str(value))
    
    def add_field_widget(self, field_key, field_label, value):
        """Add a field widget to the form."""
        if field_key == 'description' or field_key == 'notes':
            # Use QTextEdit for longer text fields
            widget = QTextEdit()
            widget.setMaximumHeight(80)
            widget.setPlainText(str(value))
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            # Use QLineEdit for single-line fields
            widget = QLineEdit()
            widget.setText(str(value))
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if field_key == 'timestamp':
                widget.setReadOnly(True)  # Timestamps shouldn't be manually edited
                widget.setStyleSheet("background-color: #f0f0f0;")
        
        # Set minimum width to ensure fields take up sufficient space
        widget.setMinimumWidth(300)
        
        # Add the widget directly to the form layout (no delete buttons)
        self.fields_layout.addRow(field_label + ":", widget)
        self.field_widgets[field_key] = widget
    
    def add_new_field(self):
        """Add a new metadata field."""
        field_name = self.new_field_name.text().strip()
        field_value = self.new_field_value.text().strip()
        
        if not field_name:
            QMessageBox.warning(self, "Invalid Field", "Please enter a field name.")
            return
        
        # Convert field name to a valid key
        field_key = field_name.lower().replace(' ', '_').replace('-', '_')
        
        # Check if field already exists
        if field_key in self.field_widgets:
            QMessageBox.warning(self, "Field Exists", f"Field '{field_name}' already exists.")
            return
        
        # Add the field
        self.add_field_widget(field_key, field_name, field_value)
        
        # Clear the input fields
        self.new_field_name.clear()
        self.new_field_value.clear()
    
    def reset_fields(self):
        """Reset all fields to their original values."""
        reply = QMessageBox.question(
            self,
            "Reset Fields",
            "Are you sure you want to reset all fields to their original values?\n\nThis will lose any unsaved changes.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear existing fields
            self.field_widgets.clear()
            
            # Clear the form layout
            while self.fields_layout.rowCount() > 0:
                self.fields_layout.removeRow(0)
            
            # Repopulate with original values
            self.populate_fields()
    
    def get_metadata(self):
        """Get the current metadata from the form fields."""
        updated_metadata = {}
        
        for field_key, widget in self.field_widgets.items():
            if isinstance(widget, QTextEdit):
                value = widget.toPlainText().strip()
            else:
                value = widget.text().strip()
            
            # Only include non-empty values
            if value:
                updated_metadata[field_key] = value
        
        return updated_metadata
        