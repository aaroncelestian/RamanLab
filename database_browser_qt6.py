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
from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar

# Qt6 imports
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, QPushButton,
    QLineEdit, QListWidget, QTextEdit, QTreeWidget, QTreeWidgetItem, QSplitter,
    QGroupBox, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QFileDialog, QMessageBox, QFormLayout, QGridLayout, QFrame, QListWidgetItem,
    QProgressDialog, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QStandardPaths, QTimer, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from scipy.signal import find_peaks

# Hey Classification System imports
try:
    from Hey_class.improved_hey_classification import ImprovedHeyClassifier, HEY_CATEGORIES
    from Hey_class.raman_vibrational_classifier import HeyCelestianClassifier
    from Hey_class.improved_element_extraction import extract_elements_from_formula
    HEY_CLASSIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"Hey Classification system not available: {e}")
    HEY_CLASSIFICATION_AVAILABLE = False
    HEY_CATEGORIES = {}


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
        
        # Lazy loading for performance with large databases
        self.all_spectrum_names = []  # Full list of spectrum names
        self.MAX_INITIAL_ITEMS = 100  # Only show first 100 initially
        self.LOAD_BATCH_SIZE = 50  # Load 50 more items when scrolling
        self.current_loaded_count = 0  # Track how many items are currently loaded
        self.is_loading_more = False  # Prevent multiple simultaneous loads
        
        # Hey Index tab variables
        self.current_hey_spectrum = None
        
        # Hey Classification System variables
        self.hey_classifier = None
        self.hey_celestian_classifier = None
        self.current_classification_spectrum = None
        
        # Initialize Hey classifiers if available
        if HEY_CLASSIFICATION_AVAILABLE:
            try:
                self.hey_classifier = ImprovedHeyClassifier()
                self.hey_celestian_classifier = HeyCelestianClassifier()
                print("Hey Classification System initialized successfully")
            except Exception as e:
                print(f"Error initializing Hey classifiers: {e}")
                self.hey_classifier = None
                self.hey_celestian_classifier = None
        
        # Set window properties
        self.setWindowTitle("RamanLab - Database Browser")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Create the UI
        self.setup_ui()
        
        # Initialize data
        self.update_spectrum_list()
        self.update_stats()
        self.update_hey_index_data()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_main_browser_tab()
        self.create_hey_index_tab()
        self.create_hey_classification_tab()
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
    
    def create_hey_index_tab(self):
        """Create the Hey Index editing tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel for search and list
        self.create_hey_index_left_panel(splitter)
        
        # Right panel for editing and metadata
        self.create_hey_index_right_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([350, 850])
        splitter.setChildrenCollapsible(False)
        splitter.widget(0).setMinimumWidth(350)
        splitter.widget(0).setMaximumWidth(350)
        
        self.tab_widget.addTab(tab, "Hey Index")
    
    def create_hey_classification_tab(self):
        """Create the Hey Classification tab with subtabs for dual classification system."""
        if not HEY_CLASSIFICATION_AVAILABLE:
            # Create a simple info tab if the system isn't available
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            info_label = QLabel(
                "Hey Classification System Not Available\n\n"
                "The Hey Classification system requires the Hey_class module.\n"
                "Please ensure the Hey_class folder is in your RamanLab directory."
            )
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("color: #666; font-size: 12px; padding: 20px;")
            layout.addWidget(info_label)
            
            self.tab_widget.addTab(tab, "Hey Classification (Unavailable)")
            return
        
        # Create main tab widget
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        # Add header info
        header_label = QLabel("🔬 Hey & Hey-Celestian Classification System")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        header_label.setStyleSheet("color: #2E7D32; padding: 10px; background: #E8F5E8; border-radius: 5px; margin-bottom: 10px;")
        main_layout.addWidget(header_label)
        
        # Create subtab widget
        self.hey_classification_tabs = QTabWidget()
        main_layout.addWidget(self.hey_classification_tabs)
        
        # Create subtabs
        self.create_single_classification_tab()
        self.create_batch_classification_tab()
        self.create_classification_viewer_tab()
        self.create_hey_celestian_groups_tab()
        
        self.tab_widget.addTab(tab, "Hey Classification")
    
    def create_hey_index_left_panel(self, parent):
        """Create the left panel for Hey Index tab."""
        left_widget = QWidget()
        left_widget.setFixedWidth(350)
        left_layout = QVBoxLayout(left_widget)
        
        # Hey Index search group
        hey_search_group = QGroupBox("Search by Hey Index")
        hey_search_layout = QVBoxLayout(hey_search_group)
        
        # Hey Index search controls
        search_controls = QHBoxLayout()
        
        self.hey_index_search = QLineEdit()
        self.hey_index_search.setPlaceholderText("Enter Hey Index (e.g., 2.45)")
        self.hey_index_search.textChanged.connect(self.search_by_hey_index)
        search_controls.addWidget(self.hey_index_search)
        
        search_hey_btn = QPushButton("Search")
        search_hey_btn.clicked.connect(self.search_by_hey_index)
        search_controls.addWidget(search_hey_btn)
        
        hey_search_layout.addLayout(search_controls)
        
        # Hey Index range search
        range_controls = QHBoxLayout()
        range_controls.addWidget(QLabel("Range:"))
        
        self.hey_min_spin = QDoubleSpinBox()
        self.hey_min_spin.setRange(1.0, 10.0)
        self.hey_min_spin.setSingleStep(0.1)
        self.hey_min_spin.setValue(1.0)
        self.hey_min_spin.setPrefix("Min: ")
        range_controls.addWidget(self.hey_min_spin)
        
        self.hey_max_spin = QDoubleSpinBox()
        self.hey_max_spin.setRange(1.0, 10.0)
        self.hey_max_spin.setSingleStep(0.1)
        self.hey_max_spin.setValue(10.0)
        self.hey_max_spin.setPrefix("Max: ")
        range_controls.addWidget(self.hey_max_spin)
        
        search_range_btn = QPushButton("Search Range")
        search_range_btn.clicked.connect(self.search_hey_range)
        range_controls.addWidget(search_range_btn)
        
        hey_search_layout.addLayout(range_controls)
        
        # Filter controls
        filter_controls = QHBoxLayout()
        
        show_unassigned_btn = QPushButton("Show Unassigned")
        show_unassigned_btn.clicked.connect(self.show_unassigned_hey)
        filter_controls.addWidget(show_unassigned_btn)
        
        show_all_btn = QPushButton("Show All")
        show_all_btn.clicked.connect(self.show_all_hey_spectra)
        filter_controls.addWidget(show_all_btn)
        
        hey_search_layout.addLayout(filter_controls)
        
        left_layout.addWidget(hey_search_group)
        
        # Spectra list with Hey Index info
        list_group = QGroupBox("Spectra with Hey Index")
        list_layout = QVBoxLayout(list_group)
        
        self.hey_spectrum_list = QListWidget()
        self.hey_spectrum_list.currentItemChanged.connect(self.on_hey_spectrum_select)
        list_layout.addWidget(self.hey_spectrum_list)
        
        left_layout.addWidget(list_group)
        
        # Hey Index statistics
        stats_group = QGroupBox("Hey Index Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.hey_stats_text = QTextEdit()
        self.hey_stats_text.setMaximumHeight(120)
        self.hey_stats_text.setReadOnly(True)
        stats_layout.addWidget(self.hey_stats_text)
        
        left_layout.addWidget(stats_group)
        
        parent.addWidget(left_widget)
    
    def create_hey_index_right_panel(self, parent):
        """Create the right panel for Hey Index editing."""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Current spectrum info group
        current_info_group = QGroupBox("Current Spectrum")
        current_info_layout = QGridLayout(current_info_group)
        
        current_info_layout.addWidget(QLabel("Name:"), 0, 0)
        self.hey_name_label = QLabel("")
        self.hey_name_label.setStyleSheet("border: 1px solid gray; padding: 2px; background: white;")
        self.hey_name_label.setWordWrap(True)
        self.hey_name_label.setMaximumHeight(40)
        current_info_layout.addWidget(self.hey_name_label, 0, 1, 1, 2)
        
        current_info_layout.addWidget(QLabel("Current Hey Index:"), 1, 0)
        self.current_hey_label = QLabel("Not assigned")
        self.current_hey_label.setStyleSheet("border: 1px solid gray; padding: 2px; background: white; font-weight: bold;")
        current_info_layout.addWidget(self.current_hey_label, 1, 1)
        
        current_info_layout.addWidget(QLabel("Mineral Name:"), 1, 2)
        self.hey_mineral_label = QLabel("")
        self.hey_mineral_label.setStyleSheet("border: 1px solid gray; padding: 2px; background: white;")
        current_info_layout.addWidget(self.hey_mineral_label, 1, 3)
        
        right_layout.addWidget(current_info_group)
        
        # Hey Index editing group
        edit_group = QGroupBox("Edit Hey Index")
        edit_layout = QVBoxLayout(edit_group)
        
        # Hey Index dropdown selection
        input_controls = QHBoxLayout()
        
        input_controls.addWidget(QLabel("Select Hey Classification:"))
        self.hey_index_dropdown = QComboBox()
        self.hey_index_dropdown.setMinimumWidth(300)
        
        # Populate with all 32 Hey Index groups
        hey_classifications = [
            "-- Select Classification --",
            "Native Elements",
            "Sulfides",
            "Sulfosalts", 
            "Halides",
            "Oxides",
            "Hydroxides",
            "Oxides and Hydroxides",
            "Carbonates",
            "Nitrates",
            "Borates",
            "Sulfates",
            "Chromates",
            "Molybdates and Tungstates",
            "Phosphates",
            "Arsenates",
            "Vanadates",
            "Silicates",
            "Silicates of Aluminum",
            "Framework Silicates",
            "Tectosilicates",
            "Chain Silicates",
            "Inosilicates",
            "Single Chain Silicates",
            "Double Chain Silicates",
            "Sheet Silicates",
            "Phyllosilicates",
            "Ring Silicates",
            "Cyclosilicates",
            "Isolated Tetrahedra Silicates",
            "Nesosilicates",
            "Linked Tetrahedra Silicates",
            "Sorosilicates"
        ]
        
        self.hey_index_dropdown.addItems(hey_classifications)
        input_controls.addWidget(self.hey_index_dropdown)
        
        input_controls.addStretch()
        edit_layout.addLayout(input_controls)
        
        # Action buttons
        action_controls = QHBoxLayout()
        
        save_hey_btn = QPushButton("Save Hey Index")
        save_hey_btn.clicked.connect(self.save_hey_index)
        save_hey_btn.setStyleSheet("""
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
        """)
        action_controls.addWidget(save_hey_btn)
        
        clear_hey_btn = QPushButton("Clear Hey Index")
        clear_hey_btn.clicked.connect(self.clear_hey_index)
        clear_hey_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC3545;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C82333;
            }
        """)
        action_controls.addWidget(clear_hey_btn)
        
        action_controls.addStretch()
        edit_layout.addLayout(action_controls)
        
        right_layout.addWidget(edit_group)
        
        # Full metadata display group
        metadata_group = QGroupBox("Complete Metadata")
        metadata_layout = QVBoxLayout(metadata_group)
        
        self.hey_metadata_text = QTextEdit()
        self.hey_metadata_text.setReadOnly(True)
        self.hey_metadata_text.setFont(QFont("Courier", 9))
        metadata_layout.addWidget(self.hey_metadata_text)
        
        # Metadata actions
        metadata_actions = QHBoxLayout()
        
        refresh_metadata_btn = QPushButton("Refresh Metadata")
        refresh_metadata_btn.clicked.connect(self.refresh_hey_metadata)
        metadata_actions.addWidget(refresh_metadata_btn)
        
        edit_full_metadata_btn = QPushButton("Edit Full Metadata")
        edit_full_metadata_btn.clicked.connect(self.edit_hey_metadata)
        metadata_actions.addWidget(edit_full_metadata_btn)
        
        metadata_actions.addStretch()
        metadata_layout.addLayout(metadata_actions)
        
        right_layout.addWidget(metadata_group)
        
        parent.addWidget(right_widget)
    
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
        
        # Add batch peak detection button
        batch_peaks_btn = QPushButton("Batch Peak Detection")
        batch_peaks_btn.clicked.connect(self.show_batch_peak_detection_dialog)
        batch_peaks_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        controls_layout.addWidget(batch_peaks_btn)
        
        save_btn = QPushButton("Save Database")
        save_btn.clicked.connect(self.save_database)
        controls_layout.addWidget(save_btn)
        
        left_layout.addWidget(controls_group)
        
        # Spectrum list group
        list_group = QGroupBox("Raman Spectra")
        list_layout = QVBoxLayout(list_group)
        
        self.spectrum_list = QListWidget()
        self.spectrum_list.currentItemChanged.connect(self.on_spectrum_select)
        
        # Connect scroll bar to detect when user reaches the bottom
        scrollbar = self.spectrum_list.verticalScrollBar()
        scrollbar.valueChanged.connect(self.on_spectrum_list_scroll)
        
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
        
        # Create scrollable area for metadata fields
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Basic Information group
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout(basic_group)
        
        self.spectrum_name_edit = QLineEdit()
        basic_layout.addRow("Spectrum Name:", self.spectrum_name_edit)
        
        self.mineral_name_edit = QLineEdit()
        basic_layout.addRow("Mineral Name:", self.mineral_name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        basic_layout.addRow("Description:", self.description_edit)
        
        scroll_layout.addWidget(basic_group)
        
        # Chemical Information group
        chemical_group = QGroupBox("Chemical Information")
        chemical_layout = QFormLayout(chemical_group)
        
        self.composition_edit = QLineEdit()
        self.composition_edit.setPlaceholderText("e.g., SiO2, CaCO3, Al2SiO5, (Ca,Mg)CO3")
        chemical_layout.addRow("Chemical Formula:", self.composition_edit)
        
        # Hey Classification dropdown
        self.hey_classification_combo = QComboBox()
        self.hey_classification_combo.setEditable(True)
        self.hey_classification_combo.addItem("")  # Empty option
        # Add all Hey categories
        for category_id, category_name in sorted(HEY_CATEGORIES.items(), key=lambda x: int(x[0])):
            display_text = f"{category_id}: {category_name}"
            self.hey_classification_combo.addItem(display_text, category_id)
        chemical_layout.addRow("Hey Classification:", self.hey_classification_combo)
        
        self.hey_celestian_edit = QLineEdit()
        self.hey_celestian_edit.setPlaceholderText("e.g., Tectosilicate, Nesosilicate, Carbonate")
        chemical_layout.addRow("Hey-Celestian Group:", self.hey_celestian_edit)
        
        scroll_layout.addWidget(chemical_group)
        
        # Crystallographic Information group
        crystal_group = QGroupBox("Crystallographic Information")
        crystal_layout = QFormLayout(crystal_group)
        
        self.crystal_system_combo = QComboBox()
        crystal_systems = ["", "Cubic", "Tetragonal", "Orthorhombic", "Hexagonal", 
                          "Trigonal", "Monoclinic", "Triclinic"]
        self.crystal_system_combo.addItems(crystal_systems)
        crystal_layout.addRow("Crystal System:", self.crystal_system_combo)
        
        self.space_group_edit = QLineEdit()
        self.space_group_edit.setPlaceholderText("e.g., Pm3m, P6_3/mmc, Fd3m")
        crystal_layout.addRow("Space Group:", self.space_group_edit)
        
        scroll_layout.addWidget(crystal_group)
        
        # Specimen Information group
        specimen_group = QGroupBox("Specimen Information")
        specimen_layout = QFormLayout(specimen_group)
        
        self.locality_edit = QLineEdit()
        self.locality_edit.setPlaceholderText("e.g., Franklin, Sussex County, New Jersey, USA")
        specimen_layout.addRow("Locality:", self.locality_edit)
        
        self.museum_catalog_edit = QLineEdit()
        self.museum_catalog_edit.setPlaceholderText("e.g., NMNH 123456, USNM 789012")
        specimen_layout.addRow("Museum Catalog #:", self.museum_catalog_edit)
        
        self.owner_edit = QLineEdit()
        self.owner_edit.setPlaceholderText("e.g., Smithsonian Institution, Private Collection")
        specimen_layout.addRow("Owner/Institution:", self.owner_edit)
        
        self.collection_date_edit = QLineEdit()
        self.collection_date_edit.setPlaceholderText("e.g., 2024-01-15, January 2024, Unknown")
        specimen_layout.addRow("Collection Date:", self.collection_date_edit)
        
        self.collector_edit = QLineEdit()
        self.collector_edit.setPlaceholderText("e.g., John Smith, Field Team Alpha")
        specimen_layout.addRow("Collector:", self.collector_edit)
        
        scroll_layout.addWidget(specimen_group)
        
        # Analytical Information group
        analytical_group = QGroupBox("Analytical Information")
        analytical_layout = QFormLayout(analytical_group)
        
        self.laser_wavelength_edit = QLineEdit()
        self.laser_wavelength_edit.setPlaceholderText("e.g., 532 nm, 785 nm")
        analytical_layout.addRow("Laser Wavelength:", self.laser_wavelength_edit)
        
        self.laser_power_edit = QLineEdit()
        self.laser_power_edit.setPlaceholderText("e.g., 10 mW, 5%")
        analytical_layout.addRow("Laser Power:", self.laser_power_edit)
        
        self.acquisition_time_edit = QLineEdit()
        self.acquisition_time_edit.setPlaceholderText("e.g., 30 seconds, 2x30s")
        analytical_layout.addRow("Acquisition Time:", self.acquisition_time_edit)
        
        self.instrument_edit = QLineEdit()
        self.instrument_edit.setPlaceholderText("e.g., Horiba LabRAM HR Evolution")
        analytical_layout.addRow("Instrument:", self.instrument_edit)
        
        self.objective_edit = QLineEdit()
        self.objective_edit.setPlaceholderText("e.g., 50x LWD, 100x")
        analytical_layout.addRow("Objective:", self.objective_edit)
        
        scroll_layout.addWidget(analytical_group)
        
        # Additional Notes group
        notes_group = QGroupBox("Additional Information")
        notes_layout = QFormLayout(notes_group)
        
        self.reference_edit = QLineEdit()
        self.reference_edit.setPlaceholderText("e.g., Smith et al. 2024, Internal Lab Data")
        notes_layout.addRow("Reference:", self.reference_edit)
        
        self.quality_rating_combo = QComboBox()
        quality_ratings = ["", "Excellent", "Good", "Fair", "Poor"]
        self.quality_rating_combo.addItems(quality_ratings)
        notes_layout.addRow("Spectrum Quality:", self.quality_rating_combo)
        
        self.preparation_edit = QLineEdit()
        self.preparation_edit.setPlaceholderText("e.g., Polished section, Powder, Oriented crystal")
        notes_layout.addRow("Sample Preparation:", self.preparation_edit)
        
        self.comments_edit = QTextEdit()
        self.comments_edit.setMaximumHeight(60)
        self.comments_edit.setPlaceholderText("Additional comments, observations, or notes...")
        notes_layout.addRow("Comments:", self.comments_edit)
        
        scroll_layout.addWidget(notes_group)
        
        # Set the scroll widget
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)
        layout.addWidget(scroll_area)
        
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
        
        self.tab_widget.addTab(tab, "Add Spectrum")
    
    def create_batch_import_tab(self):
        """Create the import/export tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Import multiple spectrum files from a directory or export/import database files.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Database Export/Import Section
        export_import_group = QGroupBox("Database Export/Import")
        export_import_layout = QVBoxLayout(export_import_group)
        
        export_import_info = QLabel(
            "Export complete database or filtered subsets for distribution, or import databases from other users."
        )
        export_import_info.setWordWrap(True)
        export_import_layout.addWidget(export_import_info)
        
        # Export options
        export_section = QGroupBox("Export Options")
        export_section_layout = QVBoxLayout(export_section)
        
        # Full database export
        full_export_layout = QHBoxLayout()
        export_db_btn = QPushButton("Export Full Database")
        export_db_btn.clicked.connect(self.export_database)
        full_export_layout.addWidget(export_db_btn)
        full_export_layout.addStretch()
        export_section_layout.addLayout(full_export_layout)
        
        # Filtered export - split into left controls and right preview
        filtered_main_layout = QHBoxLayout()
        
        # Left side - Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Filter configuration
        filter_config_layout = QFormLayout()
        
        # Filter type selection
        self.export_filter_type = QComboBox()
        self.export_filter_type.addItems([
            "Chemical Family",
            "Mineral Name",
            "Hey Classification",
            "Celestian Group",
            "Custom Metadata Field"
        ])
        self.export_filter_type.currentTextChanged.connect(self.update_export_filter_values)
        self.export_filter_type.currentTextChanged.connect(self.toggle_custom_field_visibility)
        filter_config_layout.addRow("Filter by:", self.export_filter_type)
        
        # Filter value selection
        self.export_filter_value = QComboBox()
        self.export_filter_value.setEditable(True)
        filter_config_layout.addRow("Value:", self.export_filter_value)
        
        # Custom metadata field (for custom filter type)
        self.export_custom_field = QLineEdit()
        self.export_custom_field.setPlaceholderText("Enter metadata field name...")
        self.export_custom_field.setVisible(False)
        filter_config_layout.addRow("Field Name:", self.export_custom_field)
        
        controls_layout.addLayout(filter_config_layout)
        
        # Control buttons
        filter_buttons_layout = QVBoxLayout()
        
        preview_filter_btn = QPushButton("Preview Filter")
        preview_filter_btn.clicked.connect(self.preview_export_filter)
        filter_buttons_layout.addWidget(preview_filter_btn)
        
        debug_metadata_btn = QPushButton("Debug Metadata")
        debug_metadata_btn.clicked.connect(self.debug_metadata_keys)
        filter_buttons_layout.addWidget(debug_metadata_btn)
        
        export_filtered_btn = QPushButton("Export Filtered")
        export_filtered_btn.clicked.connect(self.export_filtered_database)
        filter_buttons_layout.addWidget(export_filtered_btn)
        
        controls_layout.addLayout(filter_buttons_layout)
        controls_layout.addStretch()
        
        # Right side - Preview panel
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        preview_label = QLabel("Filter Preview:")
        preview_label.setStyleSheet("QLabel { font-weight: bold; }")
        preview_layout.addWidget(preview_label)
        
        # Filter preview results with scroll area
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setMinimumHeight(200)
        preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.export_filter_preview = QLabel("Select filter criteria and click 'Preview Filter' to see matching spectra...")
        self.export_filter_preview.setWordWrap(True)
        self.export_filter_preview.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 8px; border: 1px solid #ccc; }")
        self.export_filter_preview.setAlignment(Qt.AlignTop)
        
        preview_scroll.setWidget(self.export_filter_preview)
        preview_layout.addWidget(preview_scroll)
        
        # Add both sides to the main layout
        filtered_main_layout.addWidget(controls_widget, 1)  # 1/3 width
        filtered_main_layout.addWidget(preview_widget, 2)   # 2/3 width
        
        export_section_layout.addLayout(filtered_main_layout)
        
        export_import_layout.addWidget(export_section)
        
        # Import section
        import_section = QGroupBox("Import Options")
        import_section_layout = QHBoxLayout(import_section)
        
        import_db_btn = QPushButton("Import Database")
        import_db_btn.clicked.connect(self.import_database)
        import_section_layout.addWidget(import_db_btn)
        
        import_section_layout.addStretch()
        export_import_layout.addWidget(import_section)
        
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
        
        # Initialize filter values
        self.update_export_filter_values()
        
        self.tab_widget.addTab(tab, "Import/Export")
    
    # Event handlers and functionality methods
    def update_spectrum_list(self):
        """Update the spectrum list with lazy loading for performance."""
        start_time = time.time()
        
        self.spectrum_list.clear()
        
        # Store full sorted list
        self.all_spectrum_names = sorted(self.raman_db.database.keys())
        total_spectra = len(self.all_spectrum_names)
        
        # Only populate first MAX_INITIAL_ITEMS for instant loading
        items_to_show = min(self.MAX_INITIAL_ITEMS, total_spectra)
        self.current_loaded_count = items_to_show
        
        print(f"🚀 LAZY LOADING: Populating {items_to_show} of {total_spectra:,} spectra")
        
        for name in self.all_spectrum_names[:items_to_show]:
            item = QListWidgetItem(name)
            self.spectrum_list.addItem(item)
        
        elapsed = time.time() - start_time
        print(f"⏱️  Spectrum list populated: {elapsed:.2f}s ({items_to_show} items)")
        
        if total_spectra > self.MAX_INITIAL_ITEMS:
            print(f"ℹ️  Showing first {items_to_show} of {total_spectra:,} spectra. Scroll down to load more.")
    
    def on_spectrum_list_scroll(self, value):
        """Handle scroll events to load more items when reaching the bottom."""
        if self.is_loading_more:
            return  # Already loading, don't trigger again
        
        scrollbar = self.spectrum_list.verticalScrollBar()
        
        # Check if we're near the bottom (within 10% of max)
        max_value = scrollbar.maximum()
        if max_value == 0:
            return  # No scrollbar needed
        
        threshold = max_value * 0.9  # Trigger at 90% scroll
        
        if value >= threshold:
            # Check if there are more items to load
            total_spectra = len(self.all_spectrum_names)
            if self.current_loaded_count < total_spectra:
                self.load_more_spectra()
    
    def load_more_spectra(self):
        """Load the next batch of spectra into the list."""
        if self.is_loading_more:
            return
        
        self.is_loading_more = True
        
        try:
            total_spectra = len(self.all_spectrum_names)
            start_index = self.current_loaded_count
            end_index = min(start_index + self.LOAD_BATCH_SIZE, total_spectra)
            
            if start_index >= total_spectra:
                return  # Already loaded everything
            
            print(f"📥 Loading more spectra: {start_index} to {end_index} of {total_spectra:,}")
            
            # Add next batch of items
            for name in self.all_spectrum_names[start_index:end_index]:
                item = QListWidgetItem(name)
                self.spectrum_list.addItem(item)
            
            self.current_loaded_count = end_index
            
            remaining = total_spectra - end_index
            if remaining > 0:
                print(f"✅ Loaded {end_index} of {total_spectra:,} spectra ({remaining:,} remaining)")
            else:
                print(f"✅ All {total_spectra:,} spectra loaded")
        
        finally:
            self.is_loading_more = False
    
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
            
            # Automatically load stored peaks when spectrum is selected
            stored_peaks = self.get_stored_peaks_for_spectrum()
            print(f"DEBUG SELECT: Found {len(stored_peaks)} stored peaks for {spectrum_name}")
            print(f"DEBUG SELECT: Stored peak indices: {stored_peaks}")
            if len(stored_peaks) > 0:
                wavenumbers = np.array(self.current_spectrum['wavenumbers'])
                stored_wavenumbers = wavenumbers[stored_peaks] if len(stored_peaks) > 0 else []
                print(f"DEBUG SELECT: Stored peak wavenumbers: {stored_wavenumbers}")
                self.current_peaks = stored_peaks
                print(f"DEBUG SELECT: Set current_peaks to stored peaks")
            else:
                self.current_peaks = None
                print(f"DEBUG SELECT: Set current_peaks to None (no stored peaks)")
            
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
            if self.show_peaks_checkbox.isChecked():
                # Get stored peaks from database
                stored_peaks = self.get_stored_peaks_for_spectrum()
                
                # Show stored peaks in green
                if len(stored_peaks) > 0:
                    stored_peak_positions = wavenumbers[stored_peaks]
                    stored_peak_intensities = intensities[stored_peaks]
                    self.ax.plot(stored_peak_positions, stored_peak_intensities, 'go', 
                               markersize=8, label='Stored Peaks', markeredgecolor='darkgreen', markeredgewidth=1.5)
                
                # Show newly detected peaks in red (if different from stored)
                if self.current_peaks is not None:
                    # Check if current peaks are different from stored peaks
                    if len(stored_peaks) == 0 or not np.array_equal(np.sort(self.current_peaks), np.sort(stored_peaks)):
                        peak_positions = wavenumbers[self.current_peaks]
                        peak_intensities = intensities[self.current_peaks]
                        self.ax.plot(peak_positions, peak_intensities, 'ro', 
                                   markersize=6, label='Detected Peaks', markeredgecolor='darkred', markeredgewidth=1.5)
                
                # Add legend if there are peaks to show
                if len(stored_peaks) > 0 or (self.current_peaks is not None and len(self.current_peaks) > 0):
                    self.ax.legend(loc='upper right')
            
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
            stored_peak_indices = self.get_stored_peaks_for_spectrum()
            wavenumbers = np.array(self.current_spectrum['wavenumbers'])
            
            for peak_idx in stored_peak_indices:
                if peak_idx < len(wavenumbers):
                    item_text = f"{wavenumbers[peak_idx]:.1f} cm⁻¹"
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
        print(f"DEBUG PEAKS: Raw peaks_data from spectrum: {peaks_data}")
        print(f"DEBUG PEAKS: Type of peaks_data: {type(peaks_data)}")
        
        peak_positions = []
        
        # Handle different peak storage formats
        if isinstance(peaks_data, dict) and peaks_data.get("wavenumbers") is not None:
            # Original app format - convert wavenumbers to indices
            peak_wavenumbers = np.array(peaks_data["wavenumbers"])
            peak_positions = np.searchsorted(self.current_spectrum['wavenumbers'], peak_wavenumbers)
            print(f"DEBUG PEAKS: Dict format - wavenumbers {peak_wavenumbers} -> indices {peak_positions}")
        elif isinstance(peaks_data, (list, tuple)):
            # Check if these are wavenumber values or indices
            wavenumbers = np.array(self.current_spectrum['wavenumbers'])
            min_wn = wavenumbers.min()
            max_wn = wavenumbers.max()
            
            # Improved logic: if values are in the wavenumber range AND are decimals, treat as wavenumbers
            # If they're all integers and small (< 10% of array length), treat as indices
            all_in_wn_range = all(min_wn <= p <= max_wn for p in peaks_data if p is not None)
            has_decimals = any(float(p) != int(p) for p in peaks_data if p is not None)
            all_small_integers = all(isinstance(p, (int, float)) and float(p) == int(p) and 0 <= p < len(wavenumbers)/10 for p in peaks_data if p is not None)
            
            print(f"DEBUG PEAKS: Detection logic - all_in_wn_range: {all_in_wn_range}, has_decimals: {has_decimals}, all_small_integers: {all_small_integers}")
            print(f"DEBUG PEAKS: Wavenumber range: {min_wn:.1f} - {max_wn:.1f}")
            
            if all_small_integers and not has_decimals:
                # Treat as indices (old format)
                peak_positions = np.array([int(p) for p in peaks_data if p is not None], dtype=np.int64)
                print(f"DEBUG PEAKS: List format (indices) - {peaks_data} -> {peak_positions}")
            else:
                # Treat as wavenumber values (new format) - convert to indices
                peak_wavenumbers = np.array([float(p) for p in peaks_data if p is not None])
                peak_positions = np.searchsorted(wavenumbers, peak_wavenumbers)
                print(f"DEBUG PEAKS: List format (wavenumbers) - {peak_wavenumbers} -> indices {peak_positions}")
        else:
            peak_positions = None
            print(f"DEBUG PEAKS: Unrecognized format - returning empty")
        
        if peak_positions is not None:
            print(f"DEBUG PEAKS: Final peak_positions: {peak_positions}")
        else:
            print(f"DEBUG PEAKS: Final peak_positions: []")
            peak_positions = []
        
        return peak_positions

    def commit_peaks_to_database(self):
        """Commit the currently detected peaks to the database, overwriting stored peaks."""
        if not self.current_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        # DEBUG: Print current state
        print(f"DEBUG COMMIT: self.current_peaks = {self.current_peaks}")
        print(f"DEBUG COMMIT: type(self.current_peaks) = {type(self.current_peaks)}")
        if hasattr(self.current_peaks, '__len__'):
            print(f"DEBUG COMMIT: len(self.current_peaks) = {len(self.current_peaks)}")
        
        if self.current_peaks is None or len(self.current_peaks) == 0:
            reply = QMessageBox.question(
                self,
                "No Peaks Detected",
                f"No peaks are currently detected (current_peaks = {self.current_peaks}). Do you want to clear the stored peaks for this spectrum?",
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
            print(f"DEBUG COMMIT: Converting {len(self.current_peaks)} peak indices to wavenumbers")
            print(f"DEBUG COMMIT: Peak indices: {self.current_peaks}")
            print(f"DEBUG COMMIT: Peak wavenumbers: {peak_wavenumbers}")
        
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
                print(f"DEBUG COMMIT: Updating peaks for spectrum '{spectrum_name}'")
                print(f"DEBUG COMMIT: Old peaks in database: {self.raman_db.database[spectrum_name].get('peaks', 'NO PEAKS KEY')}")
                
                # Update the peaks in the database
                self.raman_db.database[spectrum_name]['peaks'] = peak_wavenumbers
                print(f"DEBUG COMMIT: New peaks set in database: {self.raman_db.database[spectrum_name]['peaks']}")
                
                # Save the database
                print(f"DEBUG COMMIT: Calling save_database()...")
                save_result = self.raman_db.save_database()
                print(f"DEBUG COMMIT: save_database() returned: {save_result}")
                
                # Verify the save worked by reloading the spectrum data
                print(f"DEBUG COMMIT: Verifying save - checking peaks in database...")
                verified_peaks = self.raman_db.database[spectrum_name].get('peaks', [])
                print(f"DEBUG COMMIT: Verified peaks after save: {verified_peaks}")
                
                # Update the current spectrum object
                self.current_spectrum['peaks'] = peak_wavenumbers
                print(f"DEBUG COMMIT: Updated current_spectrum['peaks'] to: {self.current_spectrum['peaks']}")
                
                # Clear current detected peaks to force refresh from database
                old_current_peaks = self.current_peaks
                self.current_peaks = None
                print(f"DEBUG COMMIT: Cleared current_peaks (was: {old_current_peaks}, now: {self.current_peaks})")
                
                # Refresh the peak lists
                print(f"DEBUG COMMIT: Calling update_peak_lists()...")
                self.update_peak_lists()
                
                # Update the plot to show the new stored peaks
                print(f"DEBUG COMMIT: Calling update_plot()...")
                self.update_plot()
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Successfully committed {peak_count} peaks to database for '{spectrum_name}'!\n\nDEBUG: Peaks saved as {peak_wavenumbers}"
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
        """Perform search based on the search term with lazy loading."""
        start_time = time.time()
        search_term = self.search_entry.text().lower()
        
        # Clear list
        self.spectrum_list.clear()
        
        if not search_term:
            # No search term - show first MAX_INITIAL_ITEMS
            items_to_show = min(self.MAX_INITIAL_ITEMS, len(self.all_spectrum_names))
            for name in self.all_spectrum_names[:items_to_show]:
                item = QListWidgetItem(name)
                self.spectrum_list.addItem(item)
            
            if len(self.all_spectrum_names) > self.MAX_INITIAL_ITEMS:
                print(f"ℹ️  Showing first {items_to_show} of {len(self.all_spectrum_names):,} spectra")
        else:
            # Search in full list (fast - just string matching)
            matching_names = [name for name in self.all_spectrum_names if search_term in name.lower()]
            
            # Limit results to 1000 for performance
            MAX_SEARCH_RESULTS = 1000
            items_to_show = min(MAX_SEARCH_RESULTS, len(matching_names))
            
            for name in matching_names[:items_to_show]:
                item = QListWidgetItem(name)
                self.spectrum_list.addItem(item)
            
            elapsed = time.time() - start_time
            print(f"⏱️  Search completed: {elapsed:.2f}s ({len(matching_names)} matches, showing {items_to_show})")
            
            if len(matching_names) > MAX_SEARCH_RESULTS:
                print(f"ℹ️  Showing first {items_to_show} of {len(matching_names):,} matching spectra. Refine search for more specific results.")
    
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
                
                # Handle different peak storage formats
                peaks_data = self.current_spectrum.get('peaks', [])
                if isinstance(peaks_data, dict) and peaks_data.get("wavenumbers") is not None:
                    # Original app format - convert wavenumbers to indices
                    peak_wavenumbers = np.array(peaks_data["wavenumbers"])
                    parent_app.detected_peaks = np.searchsorted(parent_app.current_wavenumbers, peak_wavenumbers)
                elif isinstance(peaks_data, (list, tuple)):
                    # Check if these are wavenumber values or indices
                    if all(isinstance(p, (int, float)) and 0 <= p < len(parent_app.current_wavenumbers) for p in peaks_data if p is not None):
                        # Already indices
                        parent_app.detected_peaks = np.array([int(p) for p in peaks_data if p is not None], dtype=np.int64)
                    else:
                        # Direct wavenumber values - convert to indices
                        peak_wavenumbers = np.array([float(p) for p in peaks_data if p is not None])
                        parent_app.detected_peaks = np.searchsorted(parent_app.current_wavenumbers, peak_wavenumbers)
                else:
                    parent_app.detected_peaks = None
                
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
            
            # Collect comprehensive metadata
            metadata = {
                # Basic information
                'mineral_name': self.mineral_name_edit.text(),
                'description': self.description_edit.toPlainText(),
                'source_file': file_path,
                
                # Chemical information
                'composition': self.composition_edit.text(),
                'chemical_formula': self.composition_edit.text(),  # Alias for compatibility
                
                # Hey classification
                'hey_classification': self.hey_classification_combo.currentText(),
                'hey_classification_id': self.hey_classification_combo.currentData(),
                'hey_celestian_group': self.hey_celestian_edit.text(),
                
                # Crystallographic information
                'crystal_system': self.crystal_system_combo.currentText(),
                'space_group': self.space_group_edit.text(),
                
                # Specimen information
                'locality': self.locality_edit.text(),
                'museum_catalog_number': self.museum_catalog_edit.text(),
                'owner': self.owner_edit.text(),
                'institution': self.owner_edit.text(),  # Alias for compatibility
                'collection_date': self.collection_date_edit.text(),
                'collector': self.collector_edit.text(),
                
                # Analytical information
                'laser_wavelength': self.laser_wavelength_edit.text(),
                'laser_power': self.laser_power_edit.text(),
                'acquisition_time': self.acquisition_time_edit.text(),
                'instrument': self.instrument_edit.text(),
                'objective': self.objective_edit.text(),
                
                # Additional information
                'reference': self.reference_edit.text(),
                'quality_rating': self.quality_rating_combo.currentText(),
                'sample_preparation': self.preparation_edit.text(),
                'comments': self.comments_edit.toPlainText(),
                
                # Add timestamp
                'timestamp': datetime.now().isoformat(),
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
            }
            
            # Remove empty values to keep database clean
            metadata = {k: v for k, v in metadata.items() if v and v.strip()}
            
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
        # File and basic information
        self.file_path_edit.clear()
        self.spectrum_name_edit.clear()
        self.mineral_name_edit.clear()
        self.description_edit.clear()
        
        # Chemical information
        self.composition_edit.clear()
        self.hey_classification_combo.setCurrentIndex(0)
        self.hey_celestian_edit.clear()
        
        # Crystallographic information
        self.crystal_system_combo.setCurrentIndex(0)
        self.space_group_edit.clear()
        
        # Specimen information
        self.locality_edit.clear()
        self.museum_catalog_edit.clear()
        self.owner_edit.clear()
        self.collection_date_edit.clear()
        self.collector_edit.clear()
        
        # Analytical information
        self.laser_wavelength_edit.clear()
        self.laser_power_edit.clear()
        self.acquisition_time_edit.clear()
        self.instrument_edit.clear()
        self.objective_edit.clear()
        
        # Additional information
        self.reference_edit.clear()
        self.quality_rating_combo.setCurrentIndex(0)
        self.preparation_edit.clear()
        self.comments_edit.clear()

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
        """Migrate legacy database files to Qt6 format."""
        # Show message that legacy files are no longer provided
        QMessageBox.information(
            self, 
            "Legacy Database Migration", 
            "Legacy database files (raman_database.pkl) are no longer provided.\n\n"
            "If you have an existing legacy database file from a previous installation, "
            "use 'Browse for PKL File' to locate and migrate it manually.\n\n"
            "Otherwise, you can start with a fresh database or import spectra individually."
        )
        return

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
        default_filename = f"RamanLab_Database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export RamanLab Database",
            os.path.join(os.path.expanduser("~"), "Desktop", default_filename),
            "RamanLab Database (*.pkl);;All files (*.*)"
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
            info_file = file_path.replace('.pkl', '_info.txt')
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"RamanLab Database Export\n")
                f.write(f"{'='*40}\n\n")
                f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Database Version: Qt6 Pickle Format\n\n")
                f.write(f"Database Statistics:\n")
                f.write(f"  Total Spectra: {stats['total_spectra']}\n")
                f.write(f"  Average Data Points: {stats['avg_data_points']:.0f}\n")
                f.write(f"  Average Peaks: {stats['avg_peaks']:.1f}\n")
                f.write(f"  File Size: {stats['database_size']}\n\n")
                f.write(f"Installation Instructions:\n")
                f.write(f"1. Copy this .pkl file to your RamanLab directory\n")
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
        """Import a database file (pickle format)."""
        # Browse for database file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import RamanLab Database",
            os.path.expanduser("~"),
            "RamanLab Database (*.pkl);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "File Error", f"File not found: {file_path}")
            return
        
        try:
            self.export_import_status_label.setText("Validating database file...")
            QApplication.processEvents()
            
            # Load and validate pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate pickle structure
            if not isinstance(data, dict) or 'spectra' not in data:
                raise ValueError("Not a valid RamanLab database file")
            
            count = len(data['spectra'])
            file_type = "RamanLab Database"
            
            # Show filter information if available
            metadata = data.get('metadata', {})
            filter_info = ""
            if metadata.get('export_type') == 'filtered':
                filter_type = metadata.get('filter_type', 'Unknown')
                filter_value = metadata.get('filter_value', 'Unknown')
                filter_info = f"\nFilter: {filter_type} = '{filter_value}'"
            
            # Confirm import
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            reply = QMessageBox.question(
                self,
                "Confirm Database Import",
                f"Import {file_type} from: {os.path.basename(file_path)}\n"
                f"Contains {count} spectra{filter_info}\n"
                f"File size: {file_size:.1f} MB\n\n"
                f"This will merge with your current database.\n"
                f"Existing spectra with same names will be updated.\n\n"
                f"Continue with import?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No  # Default to No for safety
            )
            
            if reply != QMessageBox.Yes:
                self.export_import_status_label.setText("Import cancelled.")
                return
            
            # Handle pickle file import (merge)
            self.export_import_status_label.setText("Importing database...")
            QApplication.processEvents()
            
            # Load pickle data
            with open(file_path, 'rb') as f:
                import_data = pickle.load(f)
            
            spectra_to_import = import_data['spectra']
            imported_count = 0
            updated_count = 0
            
            # Merge spectra into current database
            for spectrum_name, spectrum_data in spectra_to_import.items():
                if spectrum_name in self.raman_db.database:
                    updated_count += 1
                else:
                    imported_count += 1
                
                # Add/update spectrum in database
                self.raman_db.database[spectrum_name] = spectrum_data
            
            # Save the updated database
            self.raman_db.save_database()
            backup_msg = f"Merged {imported_count} new and updated {updated_count} existing spectra."
            
            # Update UI
            self.update_spectrum_list()
            self.update_stats()
            
            # Success message
            stats = self.raman_db.get_database_stats()
            
            message = (
                f"Database imported successfully!\n\n"
                f"✅ Total spectra in database: {stats['total_spectra']}\n"
                f"✅ Database size: {stats['database_size']}\n"
                f"🔍 Source: {file_type}{filter_info}\n\n"
                f"{backup_msg}\n\n"
                f"The spectra have been merged into your database!"
            )
            self.export_import_status_label.setText(f"Imported {count} spectra successfully")
            
            QMessageBox.information(self, "Import Complete", message)
            
        except Exception as e:
            error_msg = f"Import failed: {str(e)}"
            QMessageBox.critical(self, "Import Error", error_msg)
            self.export_import_status_label.setText(f"Import failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def toggle_custom_field_visibility(self):
        """Toggle visibility of custom field input based on filter type selection."""
        is_custom = self.export_filter_type.currentText() == "Custom Metadata Field"
        self.export_custom_field.setVisible(is_custom)
    
    def update_export_filter_values(self):
        """Update the filter value dropdown based on the selected filter type."""
        self.export_filter_value.clear()
        filter_type = self.export_filter_type.currentText()
        
        try:
            if filter_type == "Chemical Family":
                # Get unique chemical families from metadata
                families = set()
                all_metadata_keys = set()  # For debugging
                
                for spectrum_data in self.raman_db.database.values():
                    metadata = spectrum_data.get('metadata', {})
                    all_metadata_keys.update(metadata.keys())  # Collect all keys for debugging
                    
                    # Check multiple possible keys for chemical family (expanded list)
                    family_keys = [
                        'chemical_family', 'Chemical_Family', 'CHEMICAL_FAMILY', 
                        'family', 'Family', 'FAMILY',
                        'chemical_type', 'Chemical_Type', 'CHEMICAL_TYPE',
                        'type', 'Type', 'TYPE',
                        'material_type', 'Material_Type', 'MATERIAL_TYPE',
                        'category', 'Category', 'CATEGORY'
                    ]
                    
                    # Also check for any key containing 'family', 'type', or 'category' (case insensitive)
                    for key in metadata.keys():
                        key_lower = key.lower()
                        if any(term in key_lower for term in ['family', 'type', 'category', 'plastic']):
                            family_keys.append(key)
                    
                    # Remove duplicates while preserving order
                    family_keys = list(dict.fromkeys(family_keys))
                    
                    for key in family_keys:
                        if key in metadata and metadata[key]:
                            family_value = str(metadata[key]).strip()
                            # Filter out empty, unknown, or nonsensical values
                            if (family_value and 
                                family_value.lower() not in ['unknown', 'none', 'n/a', '', 'null', 'nan'] and
                                len(family_value) < 100 and  # Avoid very long text
                                not family_value.startswith('http')):  # Avoid URLs
                                families.add(family_value)
                                print(f"DEBUG: Found chemical family '{family_value}' in key '{key}'")
                                break
                
                # Debug output
                print(f"DEBUG: Found {len(families)} chemical families: {sorted(families)}")
                print(f"DEBUG: Sample metadata keys found: {sorted(list(all_metadata_keys))[:20]}")
                
                # If no families found, add a debug option
                if len(families) == 0:
                    self.export_filter_value.addItem("(No chemical families found - check metadata)")
                else:
                    self.export_filter_value.addItems(sorted(families))
                
            elif filter_type == "Mineral Name":
                # Get unique mineral names
                minerals = set()
                for spectrum_data in self.raman_db.database.values():
                    metadata = spectrum_data.get('metadata', {})
                    # Enhanced search for mineral names
                    name_keys = [
                        'mineral_name', 'Mineral_Name', 'MINERAL_NAME', 
                        'name', 'Name', 'NAME',
                        'mineral', 'Mineral', 'MINERAL',
                        'species', 'Species', 'SPECIES'
                    ]
                    
                    # Also check for any key containing 'mineral', 'name', or 'species'
                    for key in metadata.keys():
                        key_lower = key.lower()
                        if any(term in key_lower for term in ['mineral', 'name', 'species']) and 'file' not in key_lower:
                            name_keys.append(key)
                    
                    name_keys = list(dict.fromkeys(name_keys))
                    
                    for key in name_keys:
                        if key in metadata and metadata[key]:
                            mineral_name = str(metadata[key]).strip()
                            if (mineral_name and 
                                mineral_name.lower() not in ['unknown', 'none', 'n/a', '', 'null', 'nan'] and
                                len(mineral_name) < 100 and
                                not mineral_name.startswith('http')):
                                minerals.add(mineral_name)
                                print(f"DEBUG: Found mineral name '{mineral_name}' in key '{key}'")
                                break
                
                if len(minerals) == 0:
                    self.export_filter_value.addItem("(No mineral names found)")
                else:
                    self.export_filter_value.addItems(sorted(minerals))
                
            elif filter_type == "Hey Classification":
                # Get unique Hey classifications
                hey_classes = set()
                for spectrum_data in self.raman_db.database.values():
                    metadata = spectrum_data.get('metadata', {})
                    # Enhanced search for Hey classifications
                    hey_keys = [
                        'hey_classification', 'Hey_Classification', 'HEY_CLASSIFICATION',
                        'hey_class', 'Hey_Class', 'HEY_CLASS',
                        'hey', 'Hey', 'HEY'
                    ]
                    
                    # Also check for any key containing 'hey'
                    for key in metadata.keys():
                        key_lower = key.lower()
                        if 'hey' in key_lower:
                            hey_keys.append(key)
                    
                    hey_keys = list(dict.fromkeys(hey_keys))
                    
                    for key in hey_keys:
                        if key in metadata and metadata[key]:
                            hey_value = str(metadata[key]).strip()
                            if (hey_value and 
                                hey_value.lower() not in ['unknown', 'none', 'n/a', '', 'null', 'nan'] and
                                len(hey_value) < 100):
                                hey_classes.add(hey_value)
                                print(f"DEBUG: Found Hey classification '{hey_value}' in key '{key}'")
                                break
                
                if len(hey_classes) == 0:
                    self.export_filter_value.addItem("(No Hey classifications found)")
                else:
                    self.export_filter_value.addItems(sorted(hey_classes))
                
            elif filter_type == "Celestian Group":
                # Get unique Celestian groups
                celestian_groups = set()
                for spectrum_data in self.raman_db.database.values():
                    metadata = spectrum_data.get('metadata', {})
                    # Enhanced search for Celestian groups
                    celestian_keys = [
                        'celestian_group', 'Celestian_Group', 'CELESTIAN_GROUP',
                        'celestian', 'Celestian', 'CELESTIAN',
                        'group', 'Group', 'GROUP'
                    ]
                    
                    # Also check for any key containing 'celestian' or 'group'
                    for key in metadata.keys():
                        key_lower = key.lower()
                        if 'celestian' in key_lower or ('group' in key_lower and 'file' not in key_lower):
                            celestian_keys.append(key)
                    
                    celestian_keys = list(dict.fromkeys(celestian_keys))
                    
                    for key in celestian_keys:
                        if key in metadata and metadata[key]:
                            celestian_value = str(metadata[key]).strip()
                            if (celestian_value and 
                                celestian_value.lower() not in ['unknown', 'none', 'n/a', '', 'null', 'nan'] and
                                len(celestian_value) < 100):
                                celestian_groups.add(celestian_value)
                                print(f"DEBUG: Found Celestian group '{celestian_value}' in key '{key}'")
                                break
                
                if len(celestian_groups) == 0:
                    self.export_filter_value.addItem("(No Celestian groups found)")
                else:
                    self.export_filter_value.addItems(sorted(celestian_groups))
                
            elif filter_type == "Custom Metadata Field":
                # For custom fields, user will type both field name and value
                self.export_filter_value.addItem("Enter value...")
                
        except Exception as e:
            print(f"Error updating filter values: {e}")
    
    def preview_export_filter(self):
        """Preview which spectra match the current filter criteria."""
        filter_type = self.export_filter_type.currentText()
        filter_value = self.export_filter_value.currentText().strip()
        
        if not filter_value or filter_value == "Enter value...":
            self.export_filter_preview.setText("Please select or enter a filter value.")
            return
        
        try:
            matching_spectra = self.get_filtered_spectra(filter_type, filter_value)
            
            if len(matching_spectra) == 0:
                self.export_filter_preview.setText(f"No spectra found matching '{filter_value}' in {filter_type}.")
            else:
                # Show first few matches and total count
                preview_list = list(matching_spectra.keys())[:10]
                preview_text = f"Found {len(matching_spectra)} spectra matching '{filter_value}' in {filter_type}:\n\n"
                preview_text += "\n".join(f"• {name}" for name in preview_list)
                if len(matching_spectra) > 10:
                    preview_text += f"\n... and {len(matching_spectra) - 10} more"
                self.export_filter_preview.setText(preview_text)
                
        except Exception as e:
            self.export_filter_preview.setText(f"Error previewing filter: {str(e)}")
    
    def get_filtered_spectra(self, filter_type, filter_value):
        """Get spectra that match the specified filter criteria."""
        matching_spectra = {}
        
        for spectrum_name, spectrum_data in self.raman_db.database.items():
            metadata = spectrum_data.get('metadata', {})
            
            match_found = False
            
            if filter_type == "Chemical Family":
                # Use the same enhanced search logic as update_export_filter_values
                family_keys = [
                    'chemical_family', 'Chemical_Family', 'CHEMICAL_FAMILY', 
                    'family', 'Family', 'FAMILY',
                    'chemical_type', 'Chemical_Type', 'CHEMICAL_TYPE',
                    'type', 'Type', 'TYPE',
                    'material_type', 'Material_Type', 'MATERIAL_TYPE',
                    'category', 'Category', 'CATEGORY'
                ]
                
                # Also check for any key containing 'family', 'type', or 'category' (case insensitive)
                for key in metadata.keys():
                    key_lower = key.lower()
                    if any(term in key_lower for term in ['family', 'type', 'category', 'plastic']):
                        family_keys.append(key)
                
                # Remove duplicates while preserving order
                family_keys = list(dict.fromkeys(family_keys))
                
                for key in family_keys:
                    if key in metadata and str(metadata[key]).strip() == filter_value:
                        match_found = True
                        print(f"DEBUG: Found match for '{filter_value}' in key '{key}' for spectrum '{spectrum_name}'")
                        break
                        
            elif filter_type == "Mineral Name":
                # Enhanced search for mineral names
                name_keys = [
                    'mineral_name', 'Mineral_Name', 'MINERAL_NAME', 
                    'name', 'Name', 'NAME',
                    'mineral', 'Mineral', 'MINERAL',
                    'species', 'Species', 'SPECIES'
                ]
                
                # Also check for any key containing 'mineral', 'name', or 'species'
                for key in metadata.keys():
                    key_lower = key.lower()
                    if any(term in key_lower for term in ['mineral', 'name', 'species']) and 'file' not in key_lower:
                        name_keys.append(key)
                
                name_keys = list(dict.fromkeys(name_keys))
                
                for key in name_keys:
                    if key in metadata and str(metadata[key]).strip() == filter_value:
                        match_found = True
                        print(f"DEBUG: Found match for '{filter_value}' in key '{key}' for spectrum '{spectrum_name}'")
                        break
                        
            elif filter_type == "Hey Classification":
                # Enhanced search for Hey classifications
                hey_keys = [
                    'hey_classification', 'Hey_Classification', 'HEY_CLASSIFICATION',
                    'hey_class', 'Hey_Class', 'HEY_CLASS',
                    'hey', 'Hey', 'HEY'
                ]
                
                # Also check for any key containing 'hey'
                for key in metadata.keys():
                    key_lower = key.lower()
                    if 'hey' in key_lower:
                        hey_keys.append(key)
                
                hey_keys = list(dict.fromkeys(hey_keys))
                
                for key in hey_keys:
                    if key in metadata and str(metadata[key]).strip() == filter_value:
                        match_found = True
                        print(f"DEBUG: Found match for '{filter_value}' in key '{key}' for spectrum '{spectrum_name}'")
                        break
                        
            elif filter_type == "Celestian Group":
                # Enhanced search for Celestian groups
                celestian_keys = [
                    'celestian_group', 'Celestian_Group', 'CELESTIAN_GROUP',
                    'celestian', 'Celestian', 'CELESTIAN',
                    'group', 'Group', 'GROUP'
                ]
                
                # Also check for any key containing 'celestian' or 'group'
                for key in metadata.keys():
                    key_lower = key.lower()
                    if 'celestian' in key_lower or ('group' in key_lower and 'file' not in key_lower):
                        celestian_keys.append(key)
                
                celestian_keys = list(dict.fromkeys(celestian_keys))
                
                for key in celestian_keys:
                    if key in metadata and str(metadata[key]).strip() == filter_value:
                        match_found = True
                        print(f"DEBUG: Found match for '{filter_value}' in key '{key}' for spectrum '{spectrum_name}'")
                        break
                        
            elif filter_type == "Custom Metadata Field":
                custom_field = self.export_custom_field.text().strip()
                if custom_field and custom_field in metadata:
                    if str(metadata[custom_field]).strip() == filter_value:
                        match_found = True
            
            if match_found:
                matching_spectra[spectrum_name] = spectrum_data
        
        return matching_spectra
    
    def export_filtered_database(self):
        """Export a filtered subset of the database."""
        filter_type = self.export_filter_type.currentText()
        filter_value = self.export_filter_value.currentText().strip()
        
        if not filter_value or filter_value == "Enter value...":
            QMessageBox.warning(self, "No Filter Value", "Please select or enter a filter value.")
            return
        
        try:
            # Get filtered spectra
            matching_spectra = self.get_filtered_spectra(filter_type, filter_value)
            
            if len(matching_spectra) == 0:
                QMessageBox.warning(
                    self, 
                    "No Matching Spectra", 
                    f"No spectra found matching '{filter_value}' in {filter_type}."
                )
                return
            
            # Get save location
            safe_filter_value = "".join(c for c in filter_value if c.isalnum() or c in (' ', '-', '_')).rstrip()
            default_filename = f"RamanLab_{filter_type.replace(' ', '_')}_{safe_filter_value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                f"Export Filtered Database ({filter_type}: {filter_value})",
                os.path.join(os.path.expanduser("~"), "Desktop", default_filename),
                "Pickle Files (*.pkl);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            # Confirm export
            reply = QMessageBox.question(
                self,
                "Confirm Filtered Export",
                f"Export {len(matching_spectra)} spectra?\n\n"
                f"Filter: {filter_type} = '{filter_value}'\n"
                f"Export to: {os.path.basename(file_path)}\n\n"
                f"This will create a portable database subset.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply != QMessageBox.Yes:
                self.export_import_status_label.setText("Filtered export cancelled.")
                return
            
            self.export_import_status_label.setText("Exporting filtered database...")
            QApplication.processEvents()
            
            # Create export data structure
            export_data = {
                'metadata': {
                    'export_type': 'filtered',
                    'filter_type': filter_type,
                    'filter_value': filter_value,
                    'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_spectra': len(matching_spectra),
                    'ramanlab_version': 'Qt6'
                },
                'spectra': matching_spectra
            }
            
            # Save to pickle file
            with open(file_path, 'wb') as f:
                pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Create an info file alongside the pickle
            info_file = file_path.replace('.pkl', '_info.txt')
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"RamanLab Filtered Database Export\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Filter Type: {filter_type}\n")
                f.write(f"Filter Value: {filter_value}\n")
                f.write(f"Total Spectra: {len(matching_spectra)}\n\n")
                f.write(f"Included Spectra:\n")
                for i, name in enumerate(sorted(matching_spectra.keys()), 1):
                    f.write(f"{i:3d}. {name}\n")
                f.write(f"\nInstallation Instructions:\n")
                f.write(f"1. Copy the .pkl file to your RamanLab directory\n")
                f.write(f"2. Use 'Database > Import Database' in RamanLab\n")
                f.write(f"3. Select the .pkl file to load all {len(matching_spectra)} spectra\n\n")
                f.write(f"This filtered subset can be shared with other RamanLab users!\n")
            
            # Success message
            message = (
                f"Filtered database exported successfully!\n\n"
                f"📄 Database file: {os.path.basename(file_path)}\n"
                f"📋 Info file: {os.path.basename(info_file)}\n"
                f"🔍 Filter: {filter_type} = '{filter_value}'\n\n"
                f"Contains {len(matching_spectra)} spectra ready for sharing!\n\n"
                f"Recipients can import this filtered database directly into RamanLab."
            )
            
            QMessageBox.information(self, "Filtered Export Complete", message)
            self.export_import_status_label.setText(f"Exported {len(matching_spectra)} filtered spectra successfully")
            
        except Exception as e:
            error_msg = f"Filtered export failed: {str(e)}"
            QMessageBox.critical(self, "Export Error", error_msg)
            self.export_import_status_label.setText(f"Filtered export failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def debug_metadata_keys(self):
        """Debug method to show all metadata keys and their values from a sample of spectra."""
        try:
            if not self.raman_db.database:
                self.export_filter_preview.setText("No spectra in database to debug.")
                return
            
            # Get first 5 spectra for debugging
            sample_spectra = list(self.raman_db.database.items())[:5]
            
            debug_text = "DEBUG: Metadata Keys Analysis\n" + "="*50 + "\n\n"
            
            all_keys = set()
            for spectrum_name, spectrum_data in sample_spectra:
                metadata = spectrum_data.get('metadata', {})
                all_keys.update(metadata.keys())
                
                debug_text += f"Spectrum: {spectrum_name}\n"
                debug_text += f"Metadata keys ({len(metadata)}):\n"
                
                for key, value in metadata.items():
                    value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    debug_text += f"  • {key}: {value_str}\n"
                
                debug_text += "\n"
            
            debug_text += f"\nALL UNIQUE METADATA KEYS ({len(all_keys)}):\n"
            debug_text += "\n".join(f"• {key}" for key in sorted(all_keys))
            
            # Look for potential chemical family keys
            potential_family_keys = []
            for key in all_keys:
                key_lower = key.lower()
                if any(term in key_lower for term in ['family', 'type', 'category', 'plastic', 'chemical', 'material']):
                    potential_family_keys.append(key)
            
            if potential_family_keys:
                debug_text += f"\n\nPOTENTIAL CHEMICAL FAMILY KEYS:\n"
                debug_text += "\n".join(f"• {key}" for key in sorted(potential_family_keys))
            
            self.export_filter_preview.setText(debug_text)
            
        except Exception as e:
            self.export_filter_preview.setText(f"Debug error: {str(e)}")
    
    # Hey Index Tab Methods
    def update_hey_index_data(self):
        """Initialize the Hey Index tab with all spectra."""
        self.show_all_hey_spectra()
        self.update_hey_stats()
    
    def show_all_hey_spectra(self):
        """Show all spectra in the Hey Index list."""
        self.hey_spectrum_list.clear()
        
        for name, data in sorted(self.raman_db.database.items()):
            metadata = data.get('metadata', {})
            hey_index = self.get_hey_index_from_metadata(metadata)
            
            if hey_index is not None:
                display_text = f"{name} - Hey Index: {hey_index}"
            else:
                display_text = f"{name} - No Hey Index"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, name)  # Store the spectrum name
            self.hey_spectrum_list.addItem(item)
    
    def get_hey_index_from_metadata(self, metadata):
        """Extract Hey Index from metadata, checking various possible keys."""
        # Check for direct numeric Hey Index keys first
        direct_keys = [
            'hey_index', 'Hey_Index', 'HEY_INDEX', 'hey', 'Hey', 'HEY',
            'Hey Index', 'Hey-Index', 'Hey_index', 'hey-index', 'HeyIndex'
        ]
        
        for key in direct_keys:
            if key in metadata:
                try:
                    return float(metadata[key])
                except (ValueError, TypeError):
                    continue
        
        # Check for keys that contain "hey" and "index" (case insensitive)
        for key, value in metadata.items():
            key_lower = key.lower()
            if 'hey' in key_lower and 'index' in key_lower:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue
        
        # Check for Hey Classification and return the text value
        classification_keys = [
            'Hey Classification', 'hey_classification', 'Hey_Classification',
            'HEY_CLASSIFICATION', 'HEY CLASSIFICATION', 'hey classification', 
            'HeyClassification'
        ]
        
        for key in classification_keys:
            if key in metadata:
                classification = str(metadata[key]).strip()
                if classification:  # Return the actual classification text
                    return classification
        
        return None
    

    
    def search_by_hey_index(self):
        """Search spectra by specific Hey Index value."""
        search_text = self.hey_index_search.text().strip()
        
        if not search_text:
            self.show_all_hey_spectra()
            return
        
        try:
            target_hey = float(search_text)
            tolerance = 0.05  # ±0.05 tolerance for floating point comparison
            
            self.hey_spectrum_list.clear()
            
            for name, data in sorted(self.raman_db.database.items()):
                metadata = data.get('metadata', {})
                hey_index = self.get_hey_index_from_metadata(metadata)
                
                if hey_index is not None and abs(hey_index - target_hey) <= tolerance:
                    display_text = f"{name} - Hey Index: {hey_index}"
                    item = QListWidgetItem(display_text)
                    item.setData(Qt.UserRole, name)
                    self.hey_spectrum_list.addItem(item)
            
            if self.hey_spectrum_list.count() == 0:
                item = QListWidgetItem(f"No spectra found with Hey Index ≈ {target_hey}")
                self.hey_spectrum_list.addItem(item)
                
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid numeric Hey Index value.")
    
    def search_hey_range(self):
        """Search spectra within a Hey Index range."""
        min_hey = self.hey_min_spin.value()
        max_hey = self.hey_max_spin.value()
        
        if min_hey > max_hey:
            QMessageBox.warning(self, "Invalid Range", "Minimum value cannot be greater than maximum value.")
            return
        
        self.hey_spectrum_list.clear()
        
        for name, data in sorted(self.raman_db.database.items()):
            metadata = data.get('metadata', {})
            hey_index = self.get_hey_index_from_metadata(metadata)
            
            if hey_index is not None and min_hey <= hey_index <= max_hey:
                display_text = f"{name} - Hey Index: {hey_index}"
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, name)
                self.hey_spectrum_list.addItem(item)
        
        if self.hey_spectrum_list.count() == 0:
            item = QListWidgetItem(f"No spectra found with Hey Index {min_hey} - {max_hey}")
            self.hey_spectrum_list.addItem(item)
    
    def show_unassigned_hey(self):
        """Show spectra without assigned Hey Index."""
        self.hey_spectrum_list.clear()
        
        for name, data in sorted(self.raman_db.database.items()):
            metadata = data.get('metadata', {})
            hey_index = self.get_hey_index_from_metadata(metadata)
            
            if hey_index is None:
                display_text = f"{name} - No Hey Index"
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, name)
                self.hey_spectrum_list.addItem(item)
        
        if self.hey_spectrum_list.count() == 0:
            item = QListWidgetItem("All spectra have assigned Hey Index values")
            self.hey_spectrum_list.addItem(item)
    
    def on_hey_spectrum_select(self, current, previous):
        """Handle Hey Index spectrum selection."""
        if current is None:
            return
        
        spectrum_name = current.data(Qt.UserRole)
        if not spectrum_name:
            # Clear display if no valid spectrum
            self.clear_hey_display()
            return
        
        self.current_hey_spectrum = self.raman_db.database.get(spectrum_name)
        
        if self.current_hey_spectrum:
            metadata = self.current_hey_spectrum.get('metadata', {})
            
            # Update spectrum name display
            mineral_name = metadata.get('mineral_name', '').strip()
            if mineral_name and mineral_name.lower() != 'unknown':
                display_name = mineral_name
            else:
                # Extract from filename if possible
                if '__' in spectrum_name:
                    potential_mineral = spectrum_name.split('__')[0].strip()
                    display_name = potential_mineral if potential_mineral else spectrum_name
                else:
                    display_name = spectrum_name
            
            self.hey_name_label.setText(display_name)
            self.hey_mineral_label.setText(mineral_name or 'Not specified')
            
            # Update current Hey Index display
            hey_index = self.get_hey_index_from_metadata(metadata)
            if hey_index is not None:
                self.current_hey_label.setText(f"{hey_index}")
                # Set the dropdown to the current classification if it exists in the list
                dropdown_index = self.hey_index_dropdown.findText(hey_index)
                if dropdown_index >= 0:
                    self.hey_index_dropdown.setCurrentIndex(dropdown_index)
                else:
                    self.hey_index_dropdown.setCurrentIndex(0)  # "-- Select Classification --"
            else:
                self.current_hey_label.setText("Not assigned")
                self.hey_index_dropdown.setCurrentIndex(0)  # "-- Select Classification --"
            
            # Update full metadata display
            self.refresh_hey_metadata()
    
    def clear_hey_display(self):
        """Clear the Hey Index display."""
        self.current_hey_spectrum = None
        self.hey_name_label.setText("")
        self.hey_mineral_label.setText("")
        self.current_hey_label.setText("Not assigned")
        self.hey_index_dropdown.setCurrentIndex(0)  # "-- Select Classification --"
        self.hey_metadata_text.clear()
    
    def save_hey_index(self):
        """Save the Hey Index value to the database."""
        if not self.current_hey_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        current_item = self.hey_spectrum_list.currentItem()
        if not current_item:
            return
        
        spectrum_name = current_item.data(Qt.UserRole)
        if not spectrum_name:
            return
        
        selected_classification = self.hey_index_dropdown.currentText()
        
        # Check if a valid classification is selected
        if selected_classification == "-- Select Classification --":
            QMessageBox.warning(self, "No Selection", "Please select a Hey Classification from the dropdown.")
            return
        
        # Confirm the action
        reply = QMessageBox.question(
            self,
            "Save Hey Index",
            f"Save Hey Classification '{selected_classification}' for '{spectrum_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Update the metadata
                if 'metadata' not in self.raman_db.database[spectrum_name]:
                    self.raman_db.database[spectrum_name]['metadata'] = {}
                
                self.raman_db.database[spectrum_name]['metadata']['HEY CLASSIFICATION'] = selected_classification
                
                # Save the database
                self.raman_db.save_database()
                
                # Update the current spectrum object
                self.current_hey_spectrum['metadata']['HEY CLASSIFICATION'] = selected_classification
                
                # Update displays
                self.current_hey_label.setText(f"{selected_classification}")
                self.refresh_hey_metadata()
                self.update_hey_stats()
                
                # Update the list display
                self.show_all_hey_spectra()
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Hey Classification '{selected_classification}' saved for '{spectrum_name}'!"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save Hey Index:\n{str(e)}"
                )
    
    def clear_hey_index(self):
        """Clear the Hey Index value from the database."""
        if not self.current_hey_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        current_item = self.hey_spectrum_list.currentItem()
        if not current_item:
            return
        
        spectrum_name = current_item.data(Qt.UserRole)
        if not spectrum_name:
            return
        
        # Confirm the action
        reply = QMessageBox.question(
            self,
            "Clear Hey Index",
            f"Remove Hey Index assignment for '{spectrum_name}'?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                metadata = self.raman_db.database[spectrum_name].get('metadata', {})
                
                # Remove Hey Index from all possible keys
                hey_keys = ['hey_index', 'Hey_Index', 'HEY_INDEX', 'hey', 'Hey', 'HEY']
                for key in hey_keys:
                    if key in metadata:
                        del metadata[key]
                
                # Save the database
                self.raman_db.save_database()
                
                # Update displays
                self.current_hey_label.setText("Not assigned")
                self.refresh_hey_metadata()
                self.update_hey_stats()
                
                # Update the list display
                self.show_all_hey_spectra()
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Hey Index cleared for '{spectrum_name}'!"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to clear Hey Index:\n{str(e)}"
                )
    
    def refresh_hey_metadata(self):
        """Refresh the metadata display for the current spectrum."""
        if not self.current_hey_spectrum:
            self.hey_metadata_text.clear()
            return
        
        metadata = self.current_hey_spectrum.get('metadata', {})
        
        # Format metadata display with proper alignment
        metadata_text = "Complete Spectrum Metadata:\n"
        metadata_text += "=" * 50 + "\n\n"
        
        if metadata:
            # Sort metadata keys for consistent display
            for key in sorted(metadata.keys()):
                value = metadata[key]
                # Format key for display
                display_key = key.replace('_', ' ').title()
                
                # Special formatting for Hey Index
                if 'hey' in key.lower():
                    metadata_text += f"🔍 {display_key:20}: {value}\n"
                else:
                    metadata_text += f"   {display_key:20}: {value}\n"
        else:
            metadata_text += "No metadata available for this spectrum.\n"
        
        # Add spectrum data info
        metadata_text += f"\n{'-' * 50}\n"
        metadata_text += "Spectrum Data Information:\n"
        metadata_text += f"   Wavenumber Points   : {len(self.current_hey_spectrum.get('wavenumbers', []))}\n"
        metadata_text += f"   Intensity Points    : {len(self.current_hey_spectrum.get('intensities', []))}\n"
        
        wavenumbers = np.array(self.current_hey_spectrum.get('wavenumbers', []))
        if len(wavenumbers) > 0:
            metadata_text += f"   Wavenumber Range    : {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹\n"
        
        peaks = self.current_hey_spectrum.get('peaks', [])
        if peaks:
            metadata_text += f"   Stored Peaks        : {len(peaks)} peaks\n"
        
        self.hey_metadata_text.setPlainText(metadata_text)
    
    def edit_hey_metadata(self):
        """Edit the full metadata for the current spectrum."""
        if not self.current_hey_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        current_item = self.hey_spectrum_list.currentItem()
        if not current_item:
            return
        
        spectrum_name = current_item.data(Qt.UserRole)
        if not spectrum_name:
            return
        
        metadata = self.current_hey_spectrum.get('metadata', {}).copy()
        
        # Create metadata editor dialog (reuse existing one)
        dialog = MetadataEditorDialog(metadata, spectrum_name, self)
        
        if dialog.exec() == QDialog.Accepted:
            # Get the updated metadata
            updated_metadata = dialog.get_metadata()
            
            # Update the spectrum in the database
            self.current_hey_spectrum['metadata'] = updated_metadata
            self.raman_db.database[spectrum_name]['metadata'] = updated_metadata
            
            # Save to database
            try:
                self.raman_db.save_database()
                
                # Update the display
                self.on_hey_spectrum_select(current_item, None)  # Refresh the display
                self.update_hey_stats()
                self.show_all_hey_spectra()
                
                QMessageBox.information(self, "Success", "Metadata updated successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save metadata:\n{str(e)}")
    
    def update_hey_stats(self):
        """Update Hey Index statistics."""
        hey_values = []
        assigned_count = 0
        total_count = len(self.raman_db.database)
        
        for name, data in self.raman_db.database.items():
            metadata = data.get('metadata', {})
            hey_index = self.get_hey_index_from_metadata(metadata)
            
            if hey_index is not None:
                hey_values.append(hey_index)
                assigned_count += 1
        
        stats_text = "Hey Index Statistics:\n\n"
        stats_text += f"Total Spectra: {total_count}\n"
        stats_text += f"With Hey Index: {assigned_count}\n"
        stats_text += f"Unassigned: {total_count - assigned_count}\n"
        
        if hey_values:
            # Count the frequency of each classification
            from collections import Counter
            classification_counts = Counter(hey_values)
            
            stats_text += f"\nHey Classifications Found:\n"
            for classification, count in sorted(classification_counts.items()):
                percentage = (count / assigned_count) * 100
                stats_text += f"  {classification}: {count} ({percentage:.1f}%)\n"
        
        self.hey_stats_text.setPlainText(stats_text)
    
    # Hey Classification Tab Methods
    def create_single_classification_tab(self):
        """Create subtab for single spectrum classification."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - spectrum selection
        left_widget = QWidget()
        left_widget.setFixedWidth(350)
        left_layout = QVBoxLayout(left_widget)
        
        # Spectrum selection group
        selection_group = QGroupBox("Select Spectrum for Classification")
        selection_layout = QVBoxLayout(selection_group)
        
        # Search for spectra
        search_layout = QHBoxLayout()
        self.classification_search = QLineEdit()
        self.classification_search.setPlaceholderText("Search spectra...")
        self.classification_search.textChanged.connect(self.search_classification_spectra)
        search_layout.addWidget(self.classification_search)
        
        clear_search_btn = QPushButton("Clear")
        clear_search_btn.clicked.connect(self.clear_classification_search)
        search_layout.addWidget(clear_search_btn)
        selection_layout.addLayout(search_layout)
        
        # Spectrum list
        self.classification_spectrum_list = QListWidget()
        self.classification_spectrum_list.currentItemChanged.connect(self.on_classification_spectrum_select)
        selection_layout.addWidget(self.classification_spectrum_list)
        
        left_layout.addWidget(selection_group)
        
        # Current classification status
        status_group = QGroupBox("Current Classifications")
        status_layout = QVBoxLayout(status_group)
        
        self.current_classifications_text = QTextEdit()
        self.current_classifications_text.setMaximumHeight(150)
        self.current_classifications_text.setReadOnly(True)
        status_layout.addWidget(self.current_classifications_text)
        
        left_layout.addWidget(status_group)
        
        splitter.addWidget(left_widget)
        
        # Right panel - classification controls and results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Current spectrum info
        info_group = QGroupBox("Current Spectrum Information")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("Name:"), 0, 0)
        self.classification_name_label = QLabel("")
        self.classification_name_label.setStyleSheet("border: 1px solid gray; padding: 4px; background: white;")
        info_layout.addWidget(self.classification_name_label, 0, 1, 1, 2)
        
        info_layout.addWidget(QLabel("Chemical Formula:"), 1, 0)
        self.classification_formula_edit = QLineEdit()
        self.classification_formula_edit.setPlaceholderText("Enter or extract chemical formula")
        info_layout.addWidget(self.classification_formula_edit, 1, 1)
        
        extract_btn = QPushButton("Extract from Metadata")
        extract_btn.clicked.connect(self.extract_formula_from_metadata)
        info_layout.addWidget(extract_btn, 1, 2)
        
        debug_btn = QPushButton("Show Metadata")
        debug_btn.clicked.connect(self.show_metadata_debug)
        info_layout.addWidget(debug_btn, 2, 2)
        
        info_layout.addWidget(QLabel("Elements:"), 2, 0)
        self.classification_elements_label = QLabel("")
        self.classification_elements_label.setStyleSheet("border: 1px solid gray; padding: 4px; background: white;")
        info_layout.addWidget(self.classification_elements_label, 2, 1, 1, 2)
        
        right_layout.addWidget(info_group)
        
        # Classification options
        options_group = QGroupBox("Classification Options")
        options_layout = QVBoxLayout(options_group)
        
        # Checkboxes for which systems to use
        classifier_options = QHBoxLayout()
        self.use_traditional_hey = QCheckBox("Traditional Hey Classification")
        self.use_traditional_hey.setChecked(True)
        classifier_options.addWidget(self.use_traditional_hey)
        
        self.use_hey_celestian = QCheckBox("Hey-Celestian Vibrational Classification")
        self.use_hey_celestian.setChecked(True)
        classifier_options.addWidget(self.use_hey_celestian)
        options_layout.addLayout(classifier_options)
        
        # Hey-Celestian confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Hey-Celestian Confidence Threshold:"))
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.0, 1.0)
        self.confidence_threshold.setSingleStep(0.1)
        self.confidence_threshold.setValue(0.5)
        threshold_layout.addWidget(self.confidence_threshold)
        threshold_layout.addStretch()
        options_layout.addLayout(threshold_layout)
        
        # CUSTOM: Add peak matching tolerance control
        peak_tolerance_layout = QHBoxLayout()
        peak_tolerance_layout.addWidget(QLabel("Peak Matching Tolerance (cm⁻¹):"))
        self.peak_tolerance = QDoubleSpinBox()
        self.peak_tolerance.setRange(5.0, 100.0)
        self.peak_tolerance.setSingleStep(5.0)
        self.peak_tolerance.setValue(20.0)
        peak_tolerance_layout.addWidget(self.peak_tolerance)
        peak_tolerance_layout.addStretch()
        options_layout.addLayout(peak_tolerance_layout)
        
        # CUSTOM: Add spectral analysis options
        spectral_options = QHBoxLayout()
        self.enable_peak_matching = QCheckBox("Enable Peak Matching Analysis")
        self.enable_peak_matching.setChecked(True)
        spectral_options.addWidget(self.enable_peak_matching)
        
        self.enable_quality_assessment = QCheckBox("Enable Spectral Quality Assessment")
        self.enable_quality_assessment.setChecked(True)
        spectral_options.addWidget(self.enable_quality_assessment)
        options_layout.addLayout(spectral_options)
        
        right_layout.addWidget(options_group)
        
        # Classification action buttons
        action_layout = QHBoxLayout()
        
        classify_btn = QPushButton("🔍 Classify Spectrum")
        classify_btn.clicked.connect(self.classify_current_spectrum)
        classify_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        action_layout.addWidget(classify_btn)
        
        save_results_btn = QPushButton("💾 Save to Database")
        save_results_btn.clicked.connect(self.save_classification_results)
        save_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #388E3C;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """)
        action_layout.addWidget(save_results_btn)
        
        action_layout.addStretch()
        right_layout.addLayout(action_layout)
        
        # Results display
        results_group = QGroupBox("Classification Results")
        results_layout = QVBoxLayout(results_group)
        
        self.classification_results_text = QTextEdit()
        self.classification_results_text.setReadOnly(True)
        self.classification_results_text.setFont(QFont("Courier", 10))
        results_layout.addWidget(self.classification_results_text)
        
        right_layout.addWidget(results_group)
        
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setSizes([350, 850])
        splitter.setChildrenCollapsible(False)
        
        self.hey_classification_tabs.addTab(tab, "Single Classification")
        
        # Initialize spectrum list
        self.update_classification_spectrum_list()
    
    def create_batch_classification_tab(self):
        """Create subtab for batch classification processing."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        info_label = QLabel(
            "Batch classify all spectra in the database using both Traditional Hey and Hey-Celestian systems.\n"
            "This will analyze chemical formulas and apply both classification approaches simultaneously."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background: #FFF3E0; padding: 10px; border-radius: 5px; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Processing options
        options_group = QGroupBox("Batch Processing Options")
        options_layout = QVBoxLayout(options_group)
        
        # System selection
        systems_layout = QHBoxLayout()
        self.batch_use_traditional = QCheckBox("Process with Traditional Hey Classification")
        self.batch_use_traditional.setChecked(True)
        systems_layout.addWidget(self.batch_use_traditional)
        
        self.batch_use_celestian = QCheckBox("Process with Hey-Celestian Classification")
        self.batch_use_celestian.setChecked(True)  
        systems_layout.addWidget(self.batch_use_celestian)
        options_layout.addLayout(systems_layout)
        
        # Processing settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Hey-Celestian Confidence Threshold:"))
        self.batch_confidence_threshold = QDoubleSpinBox()
        self.batch_confidence_threshold.setRange(0.0, 1.0)
        self.batch_confidence_threshold.setSingleStep(0.1)
        self.batch_confidence_threshold.setValue(0.5)
        settings_layout.addWidget(self.batch_confidence_threshold)
        
        self.overwrite_existing = QCheckBox("Overwrite existing classifications")
        settings_layout.addWidget(self.overwrite_existing)
        
        settings_layout.addStretch()
        options_layout.addLayout(settings_layout)
        
        layout.addWidget(options_group)
        
        # Progress and status
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.batch_classification_progress = QProgressBar()
        progress_layout.addWidget(self.batch_classification_progress)
        
        self.batch_classification_status_label = QLabel("Ready to process batch classification...")
        progress_layout.addWidget(self.batch_classification_status_label)
        
        layout.addWidget(progress_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        start_batch_btn = QPushButton("🚀 Start Batch Classification")
        start_batch_btn.clicked.connect(self.start_batch_classification)
        start_batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        button_layout.addWidget(start_batch_btn)
        
        cancel_batch_btn = QPushButton("❌ Cancel Processing")
        cancel_batch_btn.clicked.connect(self.cancel_batch_classification)
        button_layout.addWidget(cancel_batch_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Results summary
        summary_group = QGroupBox("Batch Results Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.batch_classification_results_text = QTextEdit()
        self.batch_classification_results_text.setReadOnly(True)
        self.batch_classification_results_text.setMaximumHeight(200)
        summary_layout.addWidget(self.batch_classification_results_text)
        
        layout.addWidget(summary_group)
        
        layout.addStretch()
        
        self.hey_classification_tabs.addTab(tab, "Batch Processing")
    
    def create_classification_viewer_tab(self):
        """Create subtab for viewing and managing existing classifications."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - filter and search
        left_widget = QWidget()
        left_widget.setFixedWidth(350)
        left_layout = QVBoxLayout(left_widget)
        
        # Filter options
        filter_group = QGroupBox("Filter Spectra")
        filter_layout = QVBoxLayout(filter_group)
        
        # Classification status filters
        status_filters = QHBoxLayout()
        self.show_classified = QCheckBox("Show Classified")
        self.show_classified.setChecked(True)
        status_filters.addWidget(self.show_classified)
        
        self.show_unclassified = QCheckBox("Show Unclassified")
        self.show_unclassified.setChecked(True)
        status_filters.addWidget(self.show_unclassified)
        filter_layout.addLayout(status_filters)
        
        # Hey category filter
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Hey Category:"))
        self.hey_category_filter = QComboBox()
        self.hey_category_filter.addItem("All Categories")
        for category_id, category_name in HEY_CATEGORIES.items():
            self.hey_category_filter.addItem(f"{category_id}: {category_name}")
        category_layout.addWidget(self.hey_category_filter)
        filter_layout.addLayout(category_layout)
        
        # Filter buttons
        filter_buttons = QHBoxLayout()
        apply_filter_btn = QPushButton("Apply Filter")
        apply_filter_btn.clicked.connect(self.apply_classification_filter)
        filter_buttons.addWidget(apply_filter_btn)
        
        clear_filter_btn = QPushButton("Clear Filter")
        clear_filter_btn.clicked.connect(self.clear_classification_filter)
        filter_buttons.addWidget(clear_filter_btn)
        filter_layout.addLayout(filter_buttons)
        
        left_layout.addWidget(filter_group)
        
        # Classification statistics
        stats_group = QGroupBox("Classification Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.classification_stats_text = QTextEdit()
        self.classification_stats_text.setReadOnly(True)
        self.classification_stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.classification_stats_text)
        
        refresh_stats_btn = QPushButton("Refresh Statistics")
        refresh_stats_btn.clicked.connect(self.update_classification_stats)
        stats_layout.addWidget(refresh_stats_btn)
        
        left_layout.addWidget(stats_group)
        
        splitter.addWidget(left_widget)
        
        # Right panel - spectrum list and details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Classified spectra list
        list_group = QGroupBox("Classified Spectra")
        list_layout = QVBoxLayout(list_group)
        
        self.classified_spectra_list = QListWidget()
        self.classified_spectra_list.currentItemChanged.connect(self.on_classified_spectrum_select)
        list_layout.addWidget(self.classified_spectra_list)
        
        right_layout.addWidget(list_group)
        
        # Classification details
        details_group = QGroupBox("Classification Details")
        details_layout = QVBoxLayout(details_group)
        
        self.classification_details_text = QTextEdit()
        self.classification_details_text.setReadOnly(True)
        self.classification_details_text.setFont(QFont("Courier", 10))
        details_layout.addWidget(self.classification_details_text)
        
        # Action buttons for selected spectrum
        detail_buttons = QHBoxLayout()
        
        reclassify_btn = QPushButton("🔄 Reclassify")
        reclassify_btn.clicked.connect(self.reclassify_selected_spectrum)
        detail_buttons.addWidget(reclassify_btn)
        
        clear_classification_btn = QPushButton("🗑️ Clear Classification")
        clear_classification_btn.clicked.connect(self.clear_selected_classification)
        detail_buttons.addWidget(clear_classification_btn)
        
        detail_buttons.addStretch()
        details_layout.addLayout(detail_buttons)
        
        right_layout.addWidget(details_group)
        
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setSizes([350, 850])
        splitter.setChildrenCollapsible(False)
        
        self.hey_classification_tabs.addTab(tab, "Classification Viewer")
        
        # Initialize the viewer
        self.update_classification_stats()
        self.update_classified_spectra_list()
    
    def create_hey_celestian_groups_tab(self):
        """Create subtab for viewing Hey-Celestian classification groups."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - groups list
        left_widget = QWidget()
        left_widget.setFixedWidth(350)
        left_layout = QVBoxLayout(left_widget)
        
        groups_group = QGroupBox("Hey-Celestian Vibrational Groups")
        groups_layout = QVBoxLayout(groups_group)
        
        self.celestian_groups_list = QListWidget()
        self.celestian_groups_list.currentItemChanged.connect(self.on_celestian_group_select)
        groups_layout.addWidget(self.celestian_groups_list)
        
        left_layout.addWidget(groups_group)
        
        # Add group statistics
        group_stats_group = QGroupBox("Group Statistics")
        group_stats_layout = QVBoxLayout(group_stats_group)
        
        self.group_stats_text = QTextEdit()
        self.group_stats_text.setReadOnly(True)
        self.group_stats_text.setMaximumHeight(150)
        group_stats_layout.addWidget(self.group_stats_text)
        
        left_layout.addWidget(group_stats_group)
        
        splitter.addWidget(left_widget)
        
        # Right panel - group details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Group information
        info_group = QGroupBox("Group Information")
        info_layout = QVBoxLayout(info_group)
        
        self.group_info_text = QTextEdit()
        self.group_info_text.setReadOnly(True)
        self.group_info_text.setFont(QFont("Arial", 11))
        info_layout.addWidget(self.group_info_text)
        
        right_layout.addWidget(info_group)
        
        # Characteristic modes
        modes_group = QGroupBox("Characteristic Vibrational Modes")
        modes_layout = QVBoxLayout(modes_group)
        
        self.characteristic_modes_text = QTextEdit()
        self.characteristic_modes_text.setReadOnly(True)
        self.characteristic_modes_text.setFont(QFont("Courier", 10))
        modes_layout.addWidget(self.characteristic_modes_text)
        
        right_layout.addWidget(modes_group)
        
        # Analysis strategy
        strategy_group = QGroupBox("Recommended Raman Analysis Strategy")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.analysis_strategy_text = QTextEdit()
        self.analysis_strategy_text.setReadOnly(True)
        self.analysis_strategy_text.setFont(QFont("Arial", 10))
        strategy_layout.addWidget(self.analysis_strategy_text)
        
        right_layout.addWidget(strategy_group)
        
        splitter.addWidget(right_widget)
        
        # Set splitter proportions  
        splitter.setSizes([350, 850])
        splitter.setChildrenCollapsible(False)
        
        self.hey_classification_tabs.addTab(tab, "Hey-Celestian Groups")
        
        # Initialize the groups list
        self.populate_celestian_groups_list()
    
    # Hey Classification Supporting Methods
    def update_classification_spectrum_list(self):
        """Update the spectrum list for classification."""
        if not hasattr(self, 'classification_spectrum_list'):
            return
            
        self.classification_spectrum_list.clear()
        
        # Get all spectra from database
        search_term = getattr(self, 'classification_search', None)
        search_text = search_term.text().lower() if search_term else ""
        
        for name in sorted(self.raman_db.database.keys()):
            if not search_text or search_text in name.lower():
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, name)
                self.classification_spectrum_list.addItem(item)
    
    def search_classification_spectra(self):
        """Search spectra for classification."""
        self.update_classification_spectrum_list()
    
    def clear_classification_search(self):
        """Clear classification search."""
        if hasattr(self, 'classification_search'):
            self.classification_search.clear()
            self.update_classification_spectrum_list()
    
    def on_classification_spectrum_select(self, current, previous):
        """Handle spectrum selection for classification."""
        if current is None:
            return
        
        spectrum_name = current.data(Qt.UserRole)
        if not spectrum_name:
            return
        
        self.current_classification_spectrum = self.raman_db.database.get(spectrum_name)
        
        if self.current_classification_spectrum:
            metadata = self.current_classification_spectrum.get('metadata', {})
            
            # Update name display
            mineral_name = metadata.get('mineral_name', '').strip()
            if mineral_name and mineral_name.lower() != 'unknown':
                display_name = mineral_name
            else:
                if '__' in spectrum_name:
                    potential_mineral = spectrum_name.split('__')[0].strip()
                    display_name = potential_mineral if potential_mineral else spectrum_name
                else:
                    display_name = spectrum_name
            
            self.classification_name_label.setText(display_name)
            
            # Try to extract chemical formula from metadata
            self.extract_formula_from_metadata()
            
            # Update current classifications display
            self.update_current_classifications_display()
    
    def extract_formula_from_metadata(self):
        """Extract chemical formula from metadata."""
        if not self.current_classification_spectrum:
            return
        
        metadata = self.current_classification_spectrum.get('metadata', {})
        
        # Debug: Show what metadata keys are available
        print(f"DEBUG: Available metadata keys: {list(metadata.keys())}")
        
        # Look for chemical formula in various metadata fields (expanded list)
        formula_keys = [
            # Primary formula keys (based on your database structure)
            'IDEAL CHEMISTRY', 'MEASURED CHEMISTRY',  # Your database format
            'idealchemistry', 'IDEALCHEMISTRY', 'Ideal_Chemistry', 'ideal_chemistry',
            'measured chemistry', 'MEASURED_CHEMISTRY', 'Measured_Chemistry',
            
            # Standard formula keys
            'chemical_formula', 'formula', 'chemistry', 'mineral_chemistry', 'composition',
            'CHEMICAL_FORMULA', 'FORMULA', 'CHEMISTRY', 'MINERAL_CHEMISTRY', 'COMPOSITION',
            'Chemical_Formula', 'Formula', 'Chemistry', 'Mineral_Chemistry', 'Composition',
            'chemical formula', 'Chemistry Formula', 'Mineral Formula', 'Chemical Composition',
            
            # RRUFF-specific keys
            'RRUFF Chemistry (concise)', 'RRUFF_Chemistry_concise', 'rruff_chemistry',
            'RRUFF Chemistry', 'Chemistry_Elements', 'chemistry_elements'
        ]
        
        formula = ""
        found_key = None
        
        for key in formula_keys:
            if key in metadata and metadata[key]:
                formula = str(metadata[key]).strip()
                found_key = key
                print(f"DEBUG: Found formula '{formula}' in key '{key}'")
                break
        
        if not formula:
            # If no formula found, show available metadata for debugging
            print("DEBUG: No formula found. Available metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        self.classification_formula_edit.setText(formula)
        
        # Extract elements if formula is available
        if formula:
            try:
                elements = extract_elements_from_formula(formula)
                self.classification_elements_label.setText(", ".join(elements))
                print(f"DEBUG: Extracted elements: {elements}")
            except Exception as e:
                error_msg = f"Error extracting elements: {str(e)}"
                self.classification_elements_label.setText(error_msg)
                print(f"DEBUG: {error_msg}")
        else:
            self.classification_elements_label.setText("No formula available")
            
            # Show a helpful message in the formula field if no formula found
            if metadata:
                self.classification_formula_edit.setPlaceholderText("No formula found in metadata - enter manually")
            else:
                self.classification_formula_edit.setPlaceholderText("No metadata available - enter formula manually")
    
    def update_current_classifications_display(self):
        """Update the display of current classifications."""
        if not self.current_classification_spectrum:
            self.current_classifications_text.clear()
            return
        
        metadata = self.current_classification_spectrum.get('metadata', {})
        
        # Look for existing classifications
        classifications = []
        
        # Traditional Hey classification
        hey_keys = ['hey_classification', 'Hey_Classification', 'HEY_CLASSIFICATION', 'hey_id', 'hey_name']
        for key in hey_keys:
            if key in metadata and metadata[key]:
                classifications.append(f"Traditional Hey: {metadata[key]}")
                break
        
        # Hey-Celestian classification
        celestian_keys = ['hey_celestian_group', 'Hey_Celestian_Group', 'hey_celestian_classification']
        for key in celestian_keys:
            if key in metadata and metadata[key]:
                confidence = metadata.get('hey_celestian_confidence', 'N/A')
                classifications.append(f"Hey-Celestian: {metadata[key]} (Confidence: {confidence})")
                break
        
        if not classifications:
            classifications.append("No classifications found")
        
        self.current_classifications_text.setPlainText("\n".join(classifications))
    
    def classify_current_spectrum(self):
        """Classify the currently selected spectrum."""
        if not self.current_classification_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        if not self.hey_classifier and not self.hey_celestian_classifier:
            QMessageBox.warning(self, "No Classifiers", "Classification system not available.")
            return
        
        formula = self.classification_formula_edit.text().strip()
        if not formula:
            QMessageBox.warning(self, "No Formula", "Please enter a chemical formula.")
            return
        
        # CUSTOM FORMULA PREPROCESSING - Add your modifications here
        # Example: Clean up common formula formatting issues
        formula = formula.replace(" ", "")  # Remove spaces
        formula = formula.replace("·", ".")  # Replace center dot with period
        formula = formula.replace("*", ".")  # Replace asterisk with period
        
        # Preserve superscript/subscript notation (^2+^, _3_, etc.) - no modifications needed
        # This format maintains charge states and stoichiometry information
        
        # Example: Validate formula format
        if not self.validate_formula_format(formula):
            QMessageBox.warning(self, "Invalid Formula", 
                              f"Formula '{formula}' doesn't appear to be in a valid format.\n"
                              f"Please use standard chemical notation (e.g., CaCO3, Al2O3).")
            return
        
        # Extract elements with enhanced error handling
        try:
            elements = extract_elements_from_formula(formula)
            elements_str = ", ".join(elements)
            
            # CUSTOM ELEMENT VALIDATION - Add your checks here
            if len(elements) == 0:
                QMessageBox.warning(self, "No Elements", "No elements could be extracted from the formula.")
                return
            
            # Example: Check for problematic elements
            problematic_elements = {'X', 'Unknown', '?'}
            if any(elem in problematic_elements for elem in elements):
                reply = QMessageBox.question(
                    self, "Unknown Elements", 
                    f"Formula contains unknown elements: {elements}\n"
                    f"Continue with classification?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
                    
        except Exception as e:
            # Enhanced error handling for complex formula formats
            error_msg = f"Error parsing formula: {str(e)}"
            print(f"DEBUG: Formula parsing error for '{formula}': {str(e)}")
            
            # Try to give helpful suggestions for the user's format
            if '^' in formula and '_' in formula:
                reply = QMessageBox.question(
                    self, "Formula Parsing Error", 
                    f"{error_msg}\n\n"
                    f"Your formula '{formula}' uses superscript/subscript notation.\n"
                    f"The element extraction function may not fully support this format.\n\n"
                    f"Would you like to continue anyway? (Classification may still work)",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    # Try manual element extraction as fallback
                    elements = self.extract_elements_manual_fallback(formula)
                    elements_str = ", ".join(elements)
                    if len(elements) == 0:
                        QMessageBox.critical(self, "No Elements", "Could not extract any elements from the formula.")
                        return
                else:
                    return
            else:
                QMessageBox.critical(self, "Formula Error", error_msg)
                return
        
        # CUSTOM CLASSIFICATION PREPARATION
        # You can add spectrum-specific analysis here
        spectrum_data = {
            'formula': formula,
            'elements': elements,
            'elements_str': elements_str,
            'spectrum_name': self.classification_name_label.text(),
            'wavenumbers': self.current_classification_spectrum.get('wavenumbers', []),
            'intensities': self.current_classification_spectrum.get('intensities', [])
        }
        
        # Perform classifications with enhanced results
        results = []
        
        # Traditional Hey classification
        if self.use_traditional_hey.isChecked() and self.hey_classifier:
            hey_results = self.perform_hey_classification(spectrum_data)
            results.extend(hey_results)
        
        # Hey-Celestian classification
        if self.use_hey_celestian.isChecked() and self.hey_celestian_classifier:
            celestian_results = self.perform_celestian_classification(spectrum_data)
            results.extend(celestian_results)
        
        # CUSTOM ADDITIONAL ANALYSIS - Add your own classification methods here
        additional_results = self.perform_additional_analysis(spectrum_data)
        if additional_results:
            results.extend(additional_results)
        
        # Display results
        if results:
            self.classification_results_text.setPlainText("\n".join(results))
        else:
            self.classification_results_text.setPlainText("No classification methods selected.")
    
    def validate_formula_format(self, formula):
        """Validate that the formula is in a reasonable format."""
        import re
        # Enhanced regex for chemical formulas: letters, numbers, parentheses, dots, 
        # superscript (^) and subscript (_) notation for charge states and indices
        pattern = r'^[A-Za-z0-9().·*\-_^+]+$'
        return bool(re.match(pattern, formula))
    
    def extract_elements_manual_fallback(self, formula):
        """Fallback method to extract elements from complex formula formats."""
        import re
        
        # Remove superscript/subscript notation to focus on element symbols
        # Example: Fe^2+^_3_Al_2_(SiO_4_)_3_ -> Fe Al Si O
        clean_formula = formula
        
        # Remove charge indicators (^2+^, ^3+^, etc.)
        clean_formula = re.sub(r'\^[0-9]*[+-]*\^', '', clean_formula)
        
        # Remove subscript numbers (_3_, _2_, etc.)
        clean_formula = re.sub(r'_[0-9]+_', '', clean_formula)
        
        # Remove standalone numbers and parentheses for element extraction
        clean_formula = re.sub(r'[0-9().]', '', clean_formula)
        
        # Extract element symbols (1-2 letter combinations where first is uppercase)
        element_pattern = r'[A-Z][a-z]?'
        elements = re.findall(element_pattern, clean_formula)
        
        # Remove duplicates while preserving order
        unique_elements = []
        for element in elements:
            if element not in unique_elements:
                unique_elements.append(element)
        
        print(f"DEBUG: Manual element extraction from '{formula}' -> {unique_elements}")
        return unique_elements
    
    def perform_hey_classification(self, spectrum_data):
        """Perform Traditional Hey classification with enhanced error handling."""
        results = []
        try:
            hey_result = self.hey_classifier.classify_mineral(
                spectrum_data['formula'], 
                spectrum_data['elements_str']
            )
            
            results.append(f"🔸 Traditional Hey Classification:")
            results.append(f"   Category ID: {hey_result['id']}")
            results.append(f"   Category: {hey_result['name']}")
            
            # CUSTOM: Add more detailed Hey classification info
            if 'confidence' in hey_result:
                results.append(f"   Confidence: {hey_result['confidence']:.3f}")
            if 'description' in hey_result:
                results.append(f"   Description: {hey_result['description']}")
            
            results.append("")
            
        except Exception as e:
            results.append(f"🔸 Traditional Hey Classification: Error - {str(e)}")
            results.append("")
            
        return results
    
    def perform_celestian_classification(self, spectrum_data):
        """Perform Hey-Celestian classification with enhanced results."""
        results = []
        try:
            # Extract spectral data for analysis
            wavenumbers = spectrum_data.get('wavenumbers', [])
            intensities = spectrum_data.get('intensities', [])
            
            # Get detected peaks from the spectrum
            detected_peaks = []
            if len(wavenumbers) > 0 and len(intensities) > 0:
                # Convert stored peaks to wavenumber positions if they're indices
                stored_peaks = spectrum_data.get('peaks', [])
                
                # Handle different peak storage formats
                peak_values = []
                if isinstance(stored_peaks, dict):
                    # New format: peaks stored as dictionary with 'wavenumbers' key
                    if 'wavenumbers' in stored_peaks:
                        peak_values = stored_peaks['wavenumbers']
                    elif 'indices' in stored_peaks:
                        # Convert indices to wavenumbers
                        indices = stored_peaks['indices']
                        wn_array = np.array(wavenumbers)
                        peak_values = wn_array[indices]
                elif isinstance(stored_peaks, (list, tuple, np.ndarray)):
                    # Legacy format: peaks stored as list/array
                    peak_values = stored_peaks
                
                # Convert peak_values to detected_peaks list, handling numpy arrays properly
                if hasattr(peak_values, '__len__') and len(peak_values) > 0:
                    try:
                        wn_array = np.array(wavenumbers)
                        detected_peaks = []
                        
                        # Convert to list if it's a numpy array
                        if hasattr(peak_values, 'tolist'):
                            peak_list = peak_values.tolist()
                        else:
                            peak_list = list(peak_values)
                        
                        # Check if these are indices or wavenumber values
                        for p in peak_list:
                            if p is None:
                                continue
                            
                            # Convert to Python native type to avoid array boolean issues
                            p_val = float(p) if hasattr(p, 'dtype') else float(p)
                            
                            # If it looks like an index (integer in valid range), convert to wavenumber
                            if isinstance(p_val, float) and p_val == int(p_val) and 0 <= int(p_val) < len(wn_array):
                                detected_peaks.append(float(wn_array[int(p_val)]))
                            else:
                                # Treat as wavenumber value
                                detected_peaks.append(p_val)
                    except Exception as e:
                        print(f"DEBUG: Error processing stored peaks: {e}")
                        detected_peaks = []
                else:
                    detected_peaks = []
            
            # Perform classification with spectral data
            celestian_result = self.hey_celestian_classifier.classify_mineral(
                spectrum_data['formula'], 
                spectrum_data['elements_str'], 
                spectrum_data['spectrum_name'],
                wavenumbers=wavenumbers,
                intensities=intensities,
                detected_peaks=detected_peaks
            )
            
            confidence = celestian_result.get('confidence', 0.0)
            threshold = self.confidence_threshold.value()
            
            results.append(f"🔹 Hey-Celestian Vibrational Classification:")
            results.append(f"   Best Match: {celestian_result['best_group_name']}")
            results.append(f"   Group ID: {celestian_result['best_group_id']}")
            results.append(f"   Confidence: {confidence:.3f}")
            results.append(f"   Threshold: {threshold:.3f}")
            results.append(f"   Status: {'✅ High Confidence' if confidence >= threshold else '⚠️ Low Confidence'}")
            results.append(f"   Reasoning: {celestian_result['reasoning']}")
            
            # Enhanced spectral analysis results
            spectral_analysis = celestian_result.get('spectral_analysis')
            if spectral_analysis:
                results.append("")
                results.append("   📊 Spectral Analysis Results:")
                results.append(f"     • Total Peaks Detected: {spectral_analysis['total_peaks']}")
                results.append(f"     • Assigned Peaks: {len(spectral_analysis['peak_assignments'])}")
                results.append(f"     • Unassigned Peaks: {len(spectral_analysis['unassigned_peaks'])}")
                
                if spectral_analysis['chemical_constraints']:
                    results.append(f"     • Chemical Constraints: {', '.join(spectral_analysis['chemical_constraints'])}")
                
                # Show peak assignments
                peak_assignments = celestian_result.get('peak_assignments', [])
                if peak_assignments:
                    results.append("")
                    results.append("   🎯 Peak Assignments:")
                    for i, assignment in enumerate(peak_assignments[:5]):  # Show top 5
                        quality_icon = {'excellent': '🟢', 'good': '🟡', 'tentative': '🟠', 'poor': '🔴'}.get(assignment['assignment_quality'], '⚪')
                        results.append(f"     {quality_icon} {assignment['detected_peak']:.1f} cm⁻¹ → {assignment['mode_description']} (conf: {assignment['confidence']:.2f})")
                    
                    if len(peak_assignments) > 5:
                        results.append(f"     ... and {len(peak_assignments) - 5} more assignments")
                
                # Show unassigned peaks if any
                unassigned_peaks = celestian_result.get('unassigned_peaks', [])
                if unassigned_peaks:
                    results.append("")
                    results.append("   ❓ Unassigned Peaks:")
                    unassigned_str = ", ".join([f"{peak:.1f}" for peak in unassigned_peaks[:10]])
                    if len(unassigned_peaks) > 10:
                        unassigned_str += f" ... and {len(unassigned_peaks) - 10} more"
                    results.append(f"     {unassigned_str} cm⁻¹")
            else:
                results.append("")
                results.append("   ⚠️ No spectral data available for peak analysis")
                results.append("   Classification based on chemical composition only")
            
            # Initialize expected_modes to ensure it's always defined
            expected_modes = []
            
            # CUSTOM: Enhanced vibrational mode analysis with spectral confirmation
            if detected_peaks and hasattr(self.hey_celestian_classifier, 'group_expected_modes'):
                best_group_id = celestian_result['best_group_id']
                expected_modes = self.hey_celestian_classifier.group_expected_modes.get(best_group_id, [])
                
                if expected_modes:
                    results.append("")
                    results.append("   🔬 Expected vs Detected Vibrational Modes:")
                    
                    # Use the peak matcher to show expected modes
                    peak_matches = self.hey_celestian_classifier.peak_matcher.match_peaks_to_modes(
                        detected_peaks, expected_modes
                    )
                    
                    if peak_matches:
                        for match in peak_matches[:3]:  # Show top 3 matches
                            status = "✅" if match['confidence'] > 0.7 else "⚠️" if match['confidence'] > 0.4 else "❌"
                            results.append(f"     {status} Expected: {match['expected_peak']:.0f} cm⁻¹ | Found: {match['detected_peak']:.1f} cm⁻¹ | {match['mode_description']}")
                    else:
                        results.append("     No strong matches found with expected modes")
                        # Show a few expected modes for reference
                        for mode in expected_modes[:3]:
                            results.append(f"     📋 Expected: {mode['peak']:.0f} cm⁻¹ - {mode['description']}")
            
            # CUSTOM: Add spectral matching analysis (conditional)
            if (self.enable_peak_matching.isChecked() and 
                len(spectrum_data['wavenumbers']) > 0 and expected_modes):
                matching_peaks = self.find_matching_peaks(spectrum_data, expected_modes)
                if matching_peaks:
                    results.append("")
                    results.append("   🎯 Additional Peak Matching Analysis:")
                    for match in matching_peaks:
                        results.append(f"     • {match['expected']:.1f} cm⁻¹ → Found at {match['actual']:.1f} cm⁻¹ (Δ={match['delta']:.1f})")
            
            results.append("")
            
        except Exception as e:
            results.append(f"🔹 Hey-Celestian Classification: Error - {str(e)}")
            results.append("")
            
        return results
    
    def perform_additional_analysis(self, spectrum_data):
        """Perform additional custom analysis."""
        results = []
        
        # CUSTOM ANALYSIS EXAMPLE 1: Element-based warnings
        problematic_combinations = [
            (['Fe', 'S'], "⚠️  Iron-sulfur compounds may show complex Raman spectra"),
            (['Cu', 'O'], "⚠️  Copper oxides are sensitive to laser heating"),
            (['C'], "💎 Carbon-based minerals: Check for diamond/graphite peaks"),
        ]
        
        elements = spectrum_data['elements']
        for element_combo, warning in problematic_combinations:
            if all(elem in elements for elem in element_combo):
                if not results:  # Add header only once
                    results.append("🔍 Additional Analysis:")
                results.append(f"   {warning}")
        
        # CUSTOM ANALYSIS EXAMPLE 2: Spectral quality assessment (conditional)
        if (self.enable_quality_assessment.isChecked() and 
            len(spectrum_data['wavenumbers']) > 0):
            quality_assessment = self.assess_spectral_quality(spectrum_data)
            if quality_assessment:
                if not results:
                    results.append("🔍 Additional Analysis:")
                results.extend([f"   {line}" for line in quality_assessment])
        
        if results:
            results.append("")  # Add spacing
            
        return results
    
    def find_matching_peaks(self, spectrum_data, expected_modes):
        """Find peaks in the spectrum that match expected vibrational modes."""
        if not spectrum_data['wavenumbers'] or not spectrum_data['intensities']:
            return []
        
        wavenumbers = np.array(spectrum_data['wavenumbers'])
        intensities = np.array(spectrum_data['intensities'])
        
        # Simple peak detection (you can make this more sophisticated)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(intensities, height=0.1*np.max(intensities), distance=10)
        detected_peaks = wavenumbers[peaks]
        
        matches = []
        tolerance = self.peak_tolerance.value()  # Use UI control value
        
        for mode in expected_modes[:3]:  # Check top 3 expected modes
            expected_peak = mode['peak']
            
            # Find closest detected peak
            if len(detected_peaks) > 0:
                distances = np.abs(detected_peaks - expected_peak)
                closest_idx = np.argmin(distances)
                closest_peak = detected_peaks[closest_idx]
                delta = abs(closest_peak - expected_peak)
                
                if delta <= tolerance:
                    matches.append({
                        'expected': expected_peak,
                        'actual': closest_peak,
                        'delta': delta,
                        'description': mode['description']
                    })
        
        return matches
    
    def assess_spectral_quality(self, spectrum_data):
        """Assess the quality of the spectrum for classification."""
        assessment = []
        
        wavenumbers = np.array(spectrum_data['wavenumbers'])
        intensities = np.array(spectrum_data['intensities'])
        
        if len(wavenumbers) == 0:
            return ["No spectral data available"]
        
        # Check spectral range
        wn_min, wn_max = wavenumbers.min(), wavenumbers.max()
        spectral_range = wn_max - wn_min
        
        if spectral_range < 1000:
            assessment.append(f"⚠️  Limited spectral range ({spectral_range:.0f} cm⁻¹)")
        elif spectral_range > 3000:
            assessment.append(f"✅ Good spectral range ({spectral_range:.0f} cm⁻¹)")
        
        # Check signal-to-noise ratio (simplified)
        if len(intensities) > 10:
            noise_estimate = np.std(intensities[:10])  # Estimate noise from first 10 points
            signal_max = np.max(intensities)
            snr = signal_max / noise_estimate if noise_estimate > 0 else float('inf')
            
            if snr < 10:
                assessment.append(f"⚠️  Low signal-to-noise ratio (~{snr:.1f})")
            elif snr > 50:
                assessment.append(f"✅ Good signal-to-noise ratio (~{snr:.1f})")
        
        # Check for common Raman issues
        if np.any(intensities < 0):
            assessment.append("⚠️  Negative intensities detected (baseline issue?)")
        
        return assessment
    
    def save_classification_results(self):
        """Save classification results to database."""
        if not self.current_classification_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        # Get spectrum name
        current_item = self.classification_spectrum_list.currentItem()
        if not current_item:
            return
        
        spectrum_name = current_item.data(Qt.UserRole)
        formula = self.classification_formula_edit.text().strip()
        
        if not formula:
            QMessageBox.warning(self, "No Formula", "Please enter a chemical formula first.")
            return
        
        try:
            elements = extract_elements_from_formula(formula)
            elements_str = ", ".join(elements)
        except Exception as e:
            QMessageBox.critical(self, "Formula Error", f"Error parsing formula: {str(e)}")
            return
        
        # Perform classifications and save to metadata
        metadata = self.current_classification_spectrum.get('metadata', {})
        saved_classifications = []
        
        # Traditional Hey classification
        if self.use_traditional_hey.isChecked() and self.hey_classifier:
            try:
                hey_result = self.hey_classifier.classify_mineral(formula, elements_str)
                metadata['hey_classification_id'] = hey_result['id']
                metadata['hey_classification_name'] = hey_result['name']
                metadata['hey_classification_formula'] = formula
                saved_classifications.append(f"Traditional Hey: {hey_result['name']}")
            except Exception as e:
                QMessageBox.warning(self, "Classification Error", f"Traditional Hey classification failed: {str(e)}")
        
        # Hey-Celestian classification
        if self.use_hey_celestian.isChecked() and self.hey_celestian_classifier:
            try:
                spectrum_name_for_classification = self.classification_name_label.text()
                # Get spectral data for classification
                wavenumbers = self.current_classification_spectrum.get('wavenumbers', [])
                intensities = self.current_classification_spectrum.get('intensities', [])
                
                # Get detected peaks
                detected_peaks = []
                stored_peaks = self.current_classification_spectrum.get('peaks', [])
                if stored_peaks and wavenumbers:
                    try:
                        wn_array = np.array(wavenumbers)
                        # Fix: Properly handle numpy scalars and arrays in boolean contexts
                        all_are_indices = True
                        for p in stored_peaks:
                            if p is None:
                                continue
                            # Convert to Python native type to avoid array boolean issues
                            p_val = float(p) if hasattr(p, 'dtype') else p
                            if not (isinstance(p_val, (int, float)) and 0 <= p_val < len(wn_array)):
                                all_are_indices = False
                                break
                        
                        if all_are_indices:
                            detected_peaks = []
                            for p in stored_peaks:
                                if p is not None:
                                    p_idx = int(float(p))  # Handle numpy scalars
                                    if 0 <= p_idx < len(wn_array):
                                        detected_peaks.append(float(wn_array[p_idx]))
                        else:
                            detected_peaks = []
                            for p in stored_peaks:
                                if p is not None:
                                    detected_peaks.append(float(p))
                    except Exception as e:
                        print(f"DEBUG: Error processing peaks for classification: {e}")
                
                celestian_result = self.hey_celestian_classifier.classify_mineral(
                    formula, elements_str, spectrum_name_for_classification,
                    wavenumbers=wavenumbers,
                    intensities=intensities, 
                    detected_peaks=detected_peaks
                )
                
                metadata['hey_celestian_group_id'] = celestian_result['best_group_id']
                metadata['hey_celestian_group_name'] = celestian_result['best_group_name']
                metadata['hey_celestian_confidence'] = celestian_result['confidence']
                metadata['hey_celestian_reasoning'] = celestian_result['reasoning']
                metadata['hey_celestian_formula'] = formula
                
                saved_classifications.append(f"Hey-Celestian: {celestian_result['best_group_name']} ({celestian_result['confidence']:.3f})")
            except Exception as e:
                QMessageBox.warning(self, "Classification Error", f"Hey-Celestian classification failed: {str(e)}")
        
        if saved_classifications:
            # Update the database
            self.raman_db.database[spectrum_name]['metadata'] = metadata
            
            # Save to database
            try:
                self.raman_db.save_database()
                
                # Update displays
                self.update_current_classifications_display()
                self.update_classification_stats()
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Saved classifications for '{spectrum_name}':\n" + "\n".join(saved_classifications)
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save to database: {str(e)}")
        else:
            QMessageBox.warning(self, "No Results", "No classifications were generated to save.")
    
    def start_batch_classification(self):
        """Start batch classification of all spectra."""
        if not self.hey_classifier and not self.hey_celestian_classifier:
            QMessageBox.warning(self, "No Classifiers", "Classification system not available.")
            return
        
        if not self.batch_use_traditional.isChecked() and not self.batch_use_celestian.isChecked():
            QMessageBox.warning(self, "No Methods", "Please select at least one classification method.")
            return
        
        # Confirm batch processing
        total_spectra = len(self.raman_db.database)
        reply = QMessageBox.question(
            self,
            "Confirm Batch Processing",
            f"This will process {total_spectra} spectra with the selected classification methods.\n\n"
            f"Traditional Hey: {'✅ Enabled' if self.batch_use_traditional.isChecked() else '❌ Disabled'}\n"
            f"Hey-Celestian: {'✅ Enabled' if self.batch_use_celestian.isChecked() else '❌ Disabled'}\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Process batch
        self.batch_classification_progress.setMaximum(total_spectra)
        self.batch_classification_progress.setValue(0)
        
        processed = 0
        hey_success = 0
        celestian_success = 0
        errors = 0
        
        for spectrum_name, spectrum_data in self.raman_db.database.items():
            self.batch_classification_status_label.setText(f"Processing: {spectrum_name}")
            QApplication.processEvents()
            
            metadata = spectrum_data.get('metadata', {})
            
            # Skip if not overwriting and classifications exist
            if not self.overwrite_existing.isChecked():
                has_hey = any(key in metadata for key in ['hey_classification_id', 'hey_classification_name'])
                has_celestian = any(key in metadata for key in ['hey_celestian_group_id', 'hey_celestian_group_name'])
                
                if has_hey and has_celestian:
                    processed += 1
                    self.batch_classification_progress.setValue(processed)
                    continue
            
#            Try to extract formula from metadata
            formula_keys = [
                # Primary formula keys (based on your database structure)
                'IDEAL CHEMISTRY', 'MEASURED CHEMISTRY',
                'idealchemistry', 'IDEALCHEMISTRY', 'Ideal_Chemistry', 'ideal_chemistry',
                'measured chemistry', 'MEASURED_CHEMISTRY', 'Measured_Chemistry',
                # Standard formula keys
                'chemical_formula', 'formula', 'chemistry', 'mineral_chemistry', 'composition',
                'CHEMICAL_FORMULA', 'FORMULA', 'CHEMISTRY', 'MINERAL_CHEMISTRY', 'COMPOSITION'
            ]
            formula = ""
            
            for key in formula_keys:
                 if key in metadata and metadata[key]:
                     formula = str(metadata[key]).strip()
                     break
            
            if not formula:
                errors += 1
                processed += 1
                self.batch_classification_progress.setValue(processed)
                continue
            
            try:
                elements = extract_elements_from_formula(formula)
                elements_str = ", ".join(elements)
                
                # Traditional Hey classification
                if self.batch_use_traditional.isChecked() and self.hey_classifier:
                    try:
                        hey_result = self.hey_classifier.classify_mineral(formula, elements_str)
                        metadata['hey_classification_id'] = hey_result['id']
                        metadata['hey_classification_name'] = hey_result['name']
                        metadata['hey_classification_formula'] = formula
                        hey_success += 1
                    except Exception:
                        pass
                
                # Hey-Celestian classification  
                if self.batch_use_celestian.isChecked() and self.hey_celestian_classifier:
                    try:
                        # Extract mineral name for classification
                        mineral_name = metadata.get('mineral_name', spectrum_name)
                        celestian_result = self.hey_celestian_classifier.classify_mineral(
                            formula, elements_str, mineral_name
                        )
                        
                        if celestian_result['confidence'] >= self.batch_confidence_threshold.value():
                            metadata['hey_celestian_group_id'] = celestian_result['best_group_id']
                            metadata['hey_celestian_group_name'] = celestian_result['best_group_name']
                            metadata['hey_celestian_confidence'] = celestian_result['confidence']
                            metadata['hey_celestian_reasoning'] = celestian_result['reasoning']
                            metadata['hey_celestian_formula'] = formula
                            
                            # Save spectral analysis results if available
                            if celestian_result.get('spectral_analysis'):
                                spectral_analysis = celestian_result['spectral_analysis']
                                metadata['hey_celestian_peaks_detected'] = spectral_analysis['total_peaks']
                                metadata['hey_celestian_peaks_assigned'] = len(spectral_analysis['peak_assignments'])
                                metadata['hey_celestian_spectral_confidence'] = spectral_analysis['confidence_score']
                            
                            celestian_success += 1
                    except Exception:
                        pass
                
                # Update metadata
                spectrum_data['metadata'] = metadata
                
            except Exception:
                errors += 1
            
            processed += 1
            self.batch_classification_progress.setValue(processed)
        
        # Save database
        try:
            self.raman_db.save_database()
            
            # Update displays
            self.update_classification_stats()
            self.update_classified_spectra_list()
            
            # Show results
            results_text = f"Batch Classification Complete!\n\n"
            results_text += f"📊 Processing Summary:\n"
            results_text += f"   Total Spectra: {total_spectra}\n"
            results_text += f"   Processed: {processed}\n\n"
            results_text += f"🔸 Traditional Hey Classifications: {hey_success}\n"
            results_text += f"🔹 Hey-Celestian Classifications: {celestian_success}\n"
            results_text += f"❌ Errors/Skipped: {errors}\n\n"
            results_text += f"Database updated successfully!"
            
            self.batch_classification_results_text.setPlainText(results_text)
            self.batch_classification_status_label.setText("Batch classification completed!")
            
            QMessageBox.information(self, "Batch Complete", results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save database: {str(e)}")
    
    def cancel_batch_classification(self):
        """Cancel batch classification."""
        # Implementation for canceling would go here
        self.batch_classification_status_label.setText("Batch classification cancelled.")
    
    def update_classification_stats(self):
        """Update classification statistics."""
        if not hasattr(self, 'classification_stats_text'):
            return
            
        total_spectra = len(self.raman_db.database)
        hey_classified = 0
        celestian_classified = 0
        both_classified = 0
        
        hey_categories = {}
        celestian_groups = {}
        
        for spectrum_name, spectrum_data in self.raman_db.database.items():
            metadata = spectrum_data.get('metadata', {})
            
            has_hey = any(key in metadata for key in ['hey_classification_id', 'hey_classification_name'])
            has_celestian = any(key in metadata for key in ['hey_celestian_group_id', 'hey_celestian_group_name'])
            
            if has_hey:
                hey_classified += 1
                hey_name = metadata.get('hey_classification_name', 'Unknown')
                hey_categories[hey_name] = hey_categories.get(hey_name, 0) + 1
            
            if has_celestian:
                celestian_classified += 1
                celestian_name = metadata.get('hey_celestian_group_name', 'Unknown')
                celestian_groups[celestian_name] = celestian_groups.get(celestian_name, 0) + 1
            
            if has_hey and has_celestian:
                both_classified += 1
        
        stats_text = f"Classification Statistics:\n\n"
        stats_text += f"📊 Overview:\n"
        stats_text += f"   Total Spectra: {total_spectra}\n"
        stats_text += f"   Hey Classified: {hey_classified} ({hey_classified/total_spectra*100:.1f}%)\n"
        stats_text += f"   Celestian Classified: {celestian_classified} ({celestian_classified/total_spectra*100:.1f}%)\n"
        stats_text += f"   Both Classified: {both_classified} ({both_classified/total_spectra*100:.1f}%)\n\n"
        
        if hey_categories:
            stats_text += f"🔸 Top Hey Categories:\n"
            for category, count in sorted(hey_categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                stats_text += f"   {category}: {count}\n"
            stats_text += "\n"
        
        if celestian_groups:
            stats_text += f"🔹 Top Celestian Groups:\n"
            for group, count in sorted(celestian_groups.items(), key=lambda x: x[1], reverse=True)[:5]:
                stats_text += f"   {group}: {count}\n"
        
        self.classification_stats_text.setPlainText(stats_text)
    
    def update_classified_spectra_list(self):
        """Update the list of classified spectra."""
        if not hasattr(self, 'classified_spectra_list'):
            return
            
        self.classified_spectra_list.clear()
        
        for spectrum_name, spectrum_data in sorted(self.raman_db.database.items()):
            metadata = spectrum_data.get('metadata', {})
            
            has_hey = any(key in metadata for key in ['hey_classification_id', 'hey_classification_name'])
            has_celestian = any(key in metadata for key in ['hey_celestian_group_id', 'hey_celestian_group_name'])
            
            show_spectrum = False
            if self.show_classified.isChecked() and (has_hey or has_celestian):
                show_spectrum = True
            if self.show_unclassified.isChecked() and not (has_hey or has_celestian):
                show_spectrum = True
            
            if show_spectrum:
                # Create display text
                status_icons = []
                if has_hey:
                    status_icons.append("🔸")
                if has_celestian:
                    status_icons.append("🔹")
                if not status_icons:
                    status_icons.append("❌")
                
                display_text = f"{''.join(status_icons)} {spectrum_name}"
                
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, spectrum_name)
                self.classified_spectra_list.addItem(item)
    
    def apply_classification_filter(self):
        """Apply classification filter."""
        self.update_classified_spectra_list()
    
    def clear_classification_filter(self):
        """Clear classification filter."""
        self.show_classified.setChecked(True)
        self.show_unclassified.setChecked(True)
        self.hey_category_filter.setCurrentIndex(0)
        self.update_classified_spectra_list()
    
    def on_classified_spectrum_select(self, current, previous):
        """Handle classified spectrum selection."""
        if current is None:
            return
        
        spectrum_name = current.data(Qt.UserRole)
        if not spectrum_name:
            return
        
        spectrum_data = self.raman_db.database.get(spectrum_name)
        if not spectrum_data:
            return
        
        metadata = spectrum_data.get('metadata', {})
        
        # Display classification details
        details = f"Classification Details for: {spectrum_name}\n"
        details += "=" * 60 + "\n\n"
        
        # Traditional Hey classification
        if any(key in metadata for key in ['hey_classification_id', 'hey_classification_name']):
            details += "🔸 Traditional Hey Classification:\n"
            details += f"   ID: {metadata.get('hey_classification_id', 'N/A')}\n"
            details += f"   Name: {metadata.get('hey_classification_name', 'N/A')}\n"
            details += f"   Formula: {metadata.get('hey_classification_formula', 'N/A')}\n\n"
        else:
            details += "🔸 Traditional Hey Classification: Not assigned\n\n"
        
        # Hey-Celestian classification
        if any(key in metadata for key in ['hey_celestian_group_id', 'hey_celestian_group_name']):
            details += "🔹 Hey-Celestian Vibrational Classification:\n"
            details += f"   Group ID: {metadata.get('hey_celestian_group_id', 'N/A')}\n"
            details += f"   Group Name: {metadata.get('hey_celestian_group_name', 'N/A')}\n"
            details += f"   Confidence: {metadata.get('hey_celestian_confidence', 'N/A')}\n"
            details += f"   Reasoning: {metadata.get('hey_celestian_reasoning', 'N/A')}\n"
            details += f"   Formula: {metadata.get('hey_celestian_formula', 'N/A')}\n\n"
        else:
            details += "🔹 Hey-Celestian Classification: Not assigned\n\n"
        
        # Additional metadata
        details += "📋 Additional Metadata:\n"
        for key, value in metadata.items():
            if not key.startswith(('hey_', 'Hey_')):
                details += f"   {key}: {value}\n"
        
        self.classification_details_text.setPlainText(details)
    
    def reclassify_selected_spectrum(self):
        """Reclassify the selected spectrum."""
        current_item = self.classified_spectra_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a spectrum first.")
            return
        
        spectrum_name = current_item.data(Qt.UserRole)
        
        # Switch to single classification tab and load this spectrum
        self.hey_classification_tabs.setCurrentIndex(0)  # Single Classification tab
        
        # Find and select the spectrum in the classification list
        for i in range(self.classification_spectrum_list.count()):
            item = self.classification_spectrum_list.item(i)
            if item.data(Qt.UserRole) == spectrum_name:
                self.classification_spectrum_list.setCurrentItem(item)
                break
    
    def clear_selected_classification(self):
        """Clear classification for selected spectrum."""
        current_item = self.classified_spectra_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a spectrum first.")
            return
        
        spectrum_name = current_item.data(Qt.UserRole)
        
        reply = QMessageBox.question(
            self,
            "Clear Classifications",
            f"Remove all classifications for '{spectrum_name}'?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            spectrum_data = self.raman_db.database.get(spectrum_name)
            if spectrum_data:
                metadata = spectrum_data.get('metadata', {})
                
                # Remove classification keys
                keys_to_remove = [
                    'hey_classification_id', 'hey_classification_name', 'hey_classification_formula',
                    'hey_celestian_group_id', 'hey_celestian_group_name', 'hey_celestian_confidence',
                    'hey_celestian_reasoning', 'hey_celestian_formula'
                ]
                
                for key in keys_to_remove:
                    metadata.pop(key, None)
                
                spectrum_data['metadata'] = metadata
                
                # Save database
                try:
                    self.raman_db.save_database()
                    
                    # Update displays
                    self.update_classification_stats()
                    self.update_classified_spectra_list()
                    self.classification_details_text.clear()
                    
                    QMessageBox.information(self, "Success", f"Classifications cleared for '{spectrum_name}'!")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Failed to save changes: {str(e)}")
    
    def populate_celestian_groups_list(self):
        """Populate the Hey-Celestian groups list."""
        if not hasattr(self, 'celestian_groups_list') or not self.hey_celestian_classifier:
            return
        
        self.celestian_groups_list.clear()
        
        # Get vibrational groups from the classifier
        groups = self.hey_celestian_classifier.vibrational_groups
        
        for group_id, group_info in groups.items():
            display_text = f"{group_id}: {group_info['name']}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, group_id)
            self.celestian_groups_list.addItem(item)
    
    def on_celestian_group_select(self, current, previous):
        """Handle Hey-Celestian group selection."""
        if current is None or not self.hey_celestian_classifier:
            return
        
        group_id = current.data(Qt.UserRole)
        if not group_id:
            return
        
        # Get group information
        group_info = self.hey_celestian_classifier.get_classification_info(group_id)
        if not group_info:
            return
        
        # Display group information
        info_text = f"{group_info['name']}\n"
        info_text += "=" * len(group_info['name']) + "\n\n"
        info_text += f"Description: {group_info['description']}\n\n"
        info_text += f"Typical Wavenumber Range: {group_info['typical_range']}\n\n"
        info_text += "Example Minerals:\n"
        for example in group_info['examples']:
            info_text += f"  • {example}\n"
        
        self.group_info_text.setPlainText(info_text)
        
        # Display characteristic modes
        modes = self.hey_celestian_classifier.get_characteristic_modes_for_group(group_id)
        if modes:
            modes_text = "Characteristic Vibrational Modes:\n\n"
            for mode in modes:
                modes_text += f"Mode: {mode['description']}\n"
                modes_text += f"  Peak Position: {mode['peak']} cm⁻¹\n"
                modes_text += f"  Range: {mode['range'][0]} - {mode['range'][1]} cm⁻¹\n\n"
        else:
            modes_text = "No specific characteristic modes defined for this group."
        
        self.characteristic_modes_text.setPlainText(modes_text)
        
        # Display analysis strategy
        strategy = self.hey_celestian_classifier.suggest_raman_analysis_strategy(group_id)
        if strategy:
            strategy_text = f"Recommended Raman Analysis Strategy:\n\n"
            strategy_text += f"Focus Regions: {strategy.get('focus_regions', 'Not specified')}\n\n"
            strategy_text += f"Key Peaks to Look For: {strategy.get('key_peaks', 'Not specified')}\n\n"
            strategy_text += f"Analysis Tips: {strategy.get('analysis_tips', 'Not specified')}\n\n"
            strategy_text += f"Common Interferences: {strategy.get('interferences', 'Not specified')}\n"
        else:
            strategy_text = "No specific analysis strategy available for this group."
        
        self.analysis_strategy_text.setPlainText(strategy_text)
        
        # Update group statistics
        self.update_group_statistics(group_id, group_info['name'])
    
    def update_group_statistics(self, group_id, group_name):
        """Update statistics for the selected group."""
        if not hasattr(self, 'group_stats_text'):
            return
        
        # Count spectra in this group
        group_count = 0
        total_classified = 0
        
        for spectrum_name, spectrum_data in self.raman_db.database.items():
            metadata = spectrum_data.get('metadata', {})
            
            if metadata.get('hey_celestian_group_id') == group_id:
                group_count += 1
            
            if any(key in metadata for key in ['hey_celestian_group_id', 'hey_celestian_group_name']):
                total_classified += 1
        
        stats_text = f"Group Statistics:\n\n"
        stats_text += f"Group: {group_name}\n"
        stats_text += f"Spectra in Group: {group_count}\n"
        stats_text += f"Total Classified: {total_classified}\n"
        stats_text += f"Percentage: {group_count/total_classified*100:.1f}%" if total_classified > 0 else "Percentage: 0%"
        
        self.group_stats_text.setPlainText(stats_text)
    
    def show_metadata_debug(self):
        """Show metadata debug information for the current spectrum."""
        if not self.current_classification_spectrum:
            QMessageBox.warning(self, "No Spectrum", "Please select a spectrum first.")
            return
        
        # Get spectrum name
        current_item = self.classification_spectrum_list.currentItem()
        spectrum_name = current_item.data(Qt.UserRole) if current_item else "Unknown"
        
        metadata = self.current_classification_spectrum.get('metadata', {})
        
        # Create debug dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Metadata Debug - {spectrum_name}")
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        info_label = QLabel(
            "This shows all metadata keys and values for the selected spectrum.\n"
            "Look for keys that might contain chemical formulas or composition data."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background: #E3F2FD; padding: 10px; border-radius: 5px; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Metadata display
        metadata_text = QTextEdit()
        metadata_text.setReadOnly(True)
        metadata_text.setFont(QFont("Courier", 10))
        
        if metadata:
            debug_content = f"Metadata for: {spectrum_name}\n"
            debug_content += "=" * 60 + "\n\n"
            debug_content += f"Total metadata keys: {len(metadata)}\n\n"
            
            # Group metadata by potential relevance
            formula_related = []
            classification_related = []
            other_metadata = []
            
            for key, value in metadata.items():
                key_lower = key.lower()
                if any(term in key_lower for term in ['formula', 'chemistry', 'composition', 'element']):
                    formula_related.append((key, value))
                elif any(term in key_lower for term in ['classification', 'hey', 'celestian', 'group']):
                    classification_related.append((key, value))
                else:
                    other_metadata.append((key, value))
            
            if formula_related:
                debug_content += "🧪 FORMULA/CHEMISTRY RELATED METADATA:\n"
                debug_content += "-" * 40 + "\n"
                for key, value in formula_related:
                    debug_content += f"  {key}: {value}\n"
                debug_content += "\n"
            
            if classification_related:
                debug_content += "🔬 CLASSIFICATION RELATED METADATA:\n"
                debug_content += "-" * 40 + "\n"
                for key, value in classification_related:
                    debug_content += f"  {key}: {value}\n"
                debug_content += "\n"
            
            if other_metadata:
                debug_content += "📋 OTHER METADATA:\n"
                debug_content += "-" * 40 + "\n"
                for key, value in other_metadata:
                    debug_content += f"  {key}: {value}\n"
                debug_content += "\n"
            
            # Add suggestion
            if formula_related:
                debug_content += "💡 SUGGESTIONS:\n"
                debug_content += "-" * 40 + "\n"
                debug_content += "Found formula-related metadata! The system should automatically\n"
                debug_content += "extract formulas from these fields. If it's not working, the keys\n"
                debug_content += "might need to be added to the formula_keys list in the code.\n\n"
                debug_content += "You can manually copy the formula from above and paste it into\n"
                debug_content += "the Chemical Formula field in the classification interface.\n"
            else:
                debug_content += "⚠️  NO FORMULA METADATA FOUND:\n"
                debug_content += "-" * 40 + "\n"
                debug_content += "No chemistry/formula metadata detected. You'll need to:\n"
                debug_content += "1. Manually enter the chemical formula, OR\n"
                debug_content += "2. Add formula metadata to your spectra, OR\n"
                debug_content += "3. Check if the formula is stored under a different key name\n"
                
        else:
            debug_content = f"No metadata found for: {spectrum_name}\n\n"
            debug_content += "This spectrum has no metadata attached. To use the classification\n"
            debug_content += "system, you'll need to manually enter the chemical formula.\n\n"
            debug_content += "Consider adding metadata to your spectra using the metadata editor\n"
            debug_content += "in the main database browser tab."
        
        metadata_text.setPlainText(debug_content)
        layout.addWidget(metadata_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec()

    def show_batch_peak_detection_dialog(self):
        """Show the batch peak detection dialog."""
        if not self.raman_db.database:
            QMessageBox.warning(self, "Empty Database", "No spectra in database to process.")
            return
        
        dialog = BatchPeakDetectionDialog(self.raman_db, self)
        dialog.exec()
        
        # Refresh displays after batch processing
        self.update_spectrum_list()
        self.update_stats()
        if hasattr(self, 'hey_spectrum_list'):
            self.show_all_hey_spectra()


class BatchPeakDetectionDialog(QDialog):
    """Dialog for batch peak detection processing."""
    
    def __init__(self, raman_db, parent=None):
        super().__init__(parent)
        self.raman_db = raman_db
        self.processing_cancelled = False
        
        self.setWindowTitle("Batch Peak Detection")
        self.setMinimumSize(700, 600)
        self.resize(800, 650)
        
        self.setup_ui()
        self.analyze_current_peaks()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Warning header
        warning_label = QLabel(
            "⚠️ BATCH PEAK DETECTION WARNING ⚠️\n\n"
            "This will reprocess ALL spectra in your database and OVERWRITE existing peak data.\n"
            "Make sure to backup your database before proceeding!"
        )
        warning_label.setAlignment(Qt.AlignCenter)
        warning_label.setStyleSheet("""
            background-color: #FFF3CD;
            color: #856404;
            border: 2px solid #FFEAA7;
            border-radius: 8px;
            padding: 15px;
            font-weight: bold;
            font-size: 12px;
        """)
        layout.addWidget(warning_label)
        
        # Current database analysis
        analysis_group = QGroupBox("Current Database Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(120)
        self.analysis_text.setFont(QFont("Courier", 9))
        analysis_layout.addWidget(self.analysis_text)
        
        refresh_analysis_btn = QPushButton("Refresh Analysis")
        refresh_analysis_btn.clicked.connect(self.analyze_current_peaks)
        analysis_layout.addWidget(refresh_analysis_btn)
        
        layout.addWidget(analysis_group)
        
        # Peak detection parameters
        params_group = QGroupBox("Peak Detection Parameters")
        params_layout = QGridLayout(params_group)
        
        # Height threshold (relative to max intensity)
        params_layout.addWidget(QLabel("Height Threshold:"), 0, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.01, 1.0)
        self.height_spin.setSingleStep(0.01)
        self.height_spin.setValue(0.05)  # 5% of max intensity
        self.height_spin.setToolTip("Minimum peak height as fraction of maximum intensity")
        params_layout.addWidget(self.height_spin, 0, 1)
        params_layout.addWidget(QLabel("(fraction of max intensity)"), 0, 2)
        
        # Prominence threshold  
        params_layout.addWidget(QLabel("Prominence Threshold:"), 1, 0)
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.001, 0.5)
        self.prominence_spin.setSingleStep(0.005)
        self.prominence_spin.setValue(0.02)  # 2% of max intensity
        self.prominence_spin.setToolTip("Minimum peak prominence as fraction of maximum intensity")
        params_layout.addWidget(self.prominence_spin, 1, 1)
        params_layout.addWidget(QLabel("(fraction of max intensity)"), 1, 2)
        
        # Distance between peaks
        params_layout.addWidget(QLabel("Min Distance Between Peaks:"), 2, 0)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 100)
        self.distance_spin.setValue(5)
        self.distance_spin.setToolTip("Minimum number of data points between peaks")
        params_layout.addWidget(self.distance_spin, 2, 1)
        params_layout.addWidget(QLabel("(data points)"), 2, 2)
        
        # Width constraints
        params_layout.addWidget(QLabel("Min Peak Width:"), 3, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 50)
        self.width_spin.setValue(2)
        self.width_spin.setToolTip("Minimum peak width in data points")
        params_layout.addWidget(self.width_spin, 3, 1)
        params_layout.addWidget(QLabel("(data points)"), 3, 2)
        
        layout.addWidget(params_group)
        
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)
        
        # Checkboxes for processing options
        self.backup_original_peaks = QCheckBox("Backup original peaks to metadata")
        self.backup_original_peaks.setChecked(True)
        self.backup_original_peaks.setToolTip("Save current peaks as 'original_peaks' before overwriting")
        options_layout.addWidget(self.backup_original_peaks)
        
        self.skip_empty_spectra = QCheckBox("Skip spectra with no intensity data")
        self.skip_empty_spectra.setChecked(True)
        options_layout.addWidget(self.skip_empty_spectra)
        
        self.require_minimum_peaks = QCheckBox("Require minimum number of peaks")
        self.require_minimum_peaks.setChecked(False)
        options_layout.addWidget(self.require_minimum_peaks)
        
        min_peaks_layout = QHBoxLayout()
        min_peaks_layout.addWidget(QLabel("Minimum peaks required:"))
        self.min_peaks_spin = QSpinBox()
        self.min_peaks_spin.setRange(1, 50)
        self.min_peaks_spin.setValue(3)
        self.min_peaks_spin.setEnabled(False)
        min_peaks_layout.addWidget(self.min_peaks_spin)
        min_peaks_layout.addStretch()
        options_layout.addLayout(min_peaks_layout)
        
        # Connect checkbox to enable/disable spin box
        self.require_minimum_peaks.toggled.connect(self.min_peaks_spin.setEnabled)
        
        layout.addWidget(options_group)
        
        # Preview section
        preview_group = QGroupBox("Preview Results")
        preview_layout = QVBoxLayout(preview_group)
        
        preview_controls = QHBoxLayout()
        
        preview_btn = QPushButton("Preview Detection on Sample")
        preview_btn.clicked.connect(self.preview_peak_detection)
        preview_controls.addWidget(preview_btn)
        
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(1, 50)
        self.sample_size_spin.setValue(5)
        self.sample_size_spin.setPrefix("Sample size: ")
        preview_controls.addWidget(self.sample_size_spin)
        
        preview_controls.addStretch()
        preview_layout.addLayout(preview_controls)
        
        self.preview_results_text = QTextEdit()
        self.preview_results_text.setReadOnly(True)
        self.preview_results_text.setMaximumHeight(150)
        self.preview_results_text.setFont(QFont("Courier", 9))
        preview_layout.addWidget(self.preview_results_text)
        
        layout.addWidget(preview_group)
        
        # Progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to process batch peak detection...")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        start_btn = QPushButton("🚀 Start Batch Processing")
        start_btn.clicked.connect(self.start_batch_processing)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC3545;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #C82333;
            }
        """)
        button_layout.addWidget(start_btn)
        
        # Debug button for testing database save
        debug_save_btn = QPushButton("🔧 Test Database Save")
        debug_save_btn.clicked.connect(self.test_database_save)
        debug_save_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C757D;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5A6268;
            }
        """)
        button_layout.addWidget(debug_save_btn)
        
        # Debug button for testing peak detection comparison
        debug_peaks_btn = QPushButton("🔍 Test Peak Detection")
        debug_peaks_btn.clicked.connect(self.test_peak_detection_comparison)
        debug_peaks_btn.setStyleSheet("""
            QPushButton {
                background-color: #17A2B8;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        button_layout.addWidget(debug_peaks_btn)
        
        # Add verification button for current spectrum
        verify_btn = QPushButton("📋 Verify Current Spectrum")
        verify_btn.clicked.connect(self.verify_current_spectrum_detection)
        verify_btn.setToolTip("Compare individual vs batch detection for currently selected spectrum")
        verify_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        button_layout.addWidget(verify_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def analyze_current_peaks(self):
        """Analyze the current peak data in the database."""
        total_spectra = len(self.raman_db.database)
        spectra_with_peaks = 0
        total_peaks = 0
        peak_counts = []
        
        for name, data in self.raman_db.database.items():
            peaks = data.get('peaks', [])
            if peaks and len(peaks) > 0:
                spectra_with_peaks += 1
                peak_count = len(peaks)
                total_peaks += peak_count
                peak_counts.append(peak_count)
        
        analysis_text = f"Database Peak Analysis:\n"
        analysis_text += f"{'='*40}\n"
        analysis_text += f"Total Spectra: {total_spectra}\n"
        analysis_text += f"Spectra with Peaks: {spectra_with_peaks} ({spectra_with_peaks/total_spectra*100:.1f}%)\n"
        analysis_text += f"Spectra without Peaks: {total_spectra - spectra_with_peaks}\n\n"
        
        if peak_counts:
            avg_peaks = np.mean(peak_counts)
            min_peaks = min(peak_counts)
            max_peaks = max(peak_counts)
            
            analysis_text += f"Peak Statistics:\n"
            analysis_text += f"  Total Peaks: {total_peaks}\n"
            analysis_text += f"  Average Peaks per Spectrum: {avg_peaks:.1f}\n"
            analysis_text += f"  Range: {min_peaks} - {max_peaks} peaks\n\n"
            
            # Show distribution
            poor_quality = sum(1 for count in peak_counts if count < 5)
            good_quality = sum(1 for count in peak_counts if 5 <= count < 20)
            high_quality = sum(1 for count in peak_counts if count >= 20)
            
            analysis_text += f"Quality Assessment:\n"
            analysis_text += f"  Poor (< 5 peaks): {poor_quality} spectra\n"
            analysis_text += f"  Good (5-19 peaks): {good_quality} spectra\n"
            analysis_text += f"  High (20+ peaks): {high_quality} spectra\n"
        else:
            analysis_text += "No peak data found in database.\n"
        
        self.analysis_text.setPlainText(analysis_text)
    
    def preview_peak_detection(self):
        """Preview peak detection on a sample of spectra."""
        sample_size = self.sample_size_spin.value()
        
        # Get a random sample of spectra
        import random
        spectrum_names = list(self.raman_db.database.keys())
        sample_names = random.sample(spectrum_names, min(sample_size, len(spectrum_names)))
        
        preview_text = f"Preview Results (Sample of {len(sample_names)} spectra):\n"
        preview_text += f"{'='*50}\n\n"
        
        for name in sample_names:
            spectrum_data = self.raman_db.database[name]
            wavenumbers = np.array(spectrum_data.get('wavenumbers', []))
            intensities = np.array(spectrum_data.get('intensities', []))
            
            if len(intensities) == 0:
                preview_text += f"{name}: No intensity data\n"
                continue
            
            # Current peaks
            current_peaks = spectrum_data.get('peaks', [])
            current_count = len(current_peaks) if current_peaks else 0
            
            # Detect new peaks
            try:
                detected_peaks = self.detect_peaks_for_spectrum(intensities)
                new_count = len(detected_peaks)
                
                # Get wavenumber positions
                if len(wavenumbers) > 0:
                    new_positions = wavenumbers[detected_peaks]
                    pos_str = f"[{', '.join([f'{p:.0f}' for p in new_positions[:5]])}{'...' if new_count > 5 else ''}]"
                else:
                    pos_str = f"[indices: {detected_peaks[:5].tolist()}{'...' if new_count > 5 else ''}]"
                
                preview_text += f"{name}:\n"
                preview_text += f"  Current: {current_count} peaks\n"
                preview_text += f"  New: {new_count} peaks {pos_str}\n"
                preview_text += f"  Change: {new_count - current_count:+d} peaks\n\n"
                
            except Exception as e:
                preview_text += f"{name}: Error - {str(e)}\n\n"
        
        self.preview_results_text.setPlainText(preview_text)
    
    def detect_peaks_for_spectrum(self, intensities):
        """Detect peaks for a single spectrum using current parameters."""
        if len(intensities) == 0:
            return np.array([])
        
        # Get parameters - MATCH EXACTLY the individual peak detection method
        max_intensity = np.max(intensities)
        height_threshold = self.height_spin.value() * max_intensity
        prominence_threshold = self.prominence_spin.value() * max_intensity
        distance = self.distance_spin.value()
        
        print(f"DEBUG: Peak detection params - height: {height_threshold:.3f}, prominence: {prominence_threshold:.3f}, distance: {distance}")
        print(f"DEBUG: Max intensity: {max_intensity:.3f}, height %: {self.height_spin.value():.3f}, prominence %: {self.prominence_spin.value():.3f}")
        
        # Find peaks using SAME method as individual detection - REMOVED WIDTH PARAMETER
        peaks, properties = find_peaks(
            intensities,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=distance
        )
        
        print(f"DEBUG: Found {len(peaks)} peaks at indices: {peaks}")
        
        return peaks
    
    def start_batch_processing(self):
        """Start the batch peak detection processing."""
        # Final confirmation
        total_spectra = len(self.raman_db.database)
        reply = QMessageBox.question(
            self,
            "Confirm Batch Processing",
            f"⚠️ FINAL WARNING ⚠️\n\n"
            f"This will process {total_spectra} spectra and OVERWRITE all existing peak data.\n\n"
            f"Parameters:\n"
            f"• Height Threshold: {self.height_spin.value():.3f}\n"
            f"• Prominence Threshold: {self.prominence_spin.value():.3f}\n"
            f"• Min Distance: {self.distance_spin.value()} points\n\n"
            f"Backup original peaks: {'Yes' if self.backup_original_peaks.isChecked() else 'No'}\n\n"
            f"Are you absolutely sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Start processing
        self.processing_cancelled = False
        self.progress_bar.setMaximum(total_spectra)
        self.progress_bar.setValue(0)
        
        processed = 0
        successful = 0
        skipped = 0
        errors = 0
        
        for spectrum_name, spectrum_data in self.raman_db.database.items():
            if self.processing_cancelled:
                break
            
            self.status_label.setText(f"Processing: {spectrum_name}")
            QApplication.processEvents()
            
            try:
                wavenumbers = np.array(spectrum_data.get('wavenumbers', []))
                intensities = np.array(spectrum_data.get('intensities', []))
                
                # Skip if no intensity data
                if len(intensities) == 0:
                    if self.skip_empty_spectra.isChecked():
                        skipped += 1
                        processed += 1
                        self.progress_bar.setValue(processed)
                        continue
                
                # Backup original peaks if requested
                if self.backup_original_peaks.isChecked():
                    original_peaks = spectrum_data.get('peaks', [])
                    if original_peaks:
                        if 'metadata' not in spectrum_data:
                            spectrum_data['metadata'] = {}
                        spectrum_data['metadata']['original_peaks_backup'] = original_peaks.copy()
                        spectrum_data['metadata']['peak_backup_timestamp'] = datetime.now().isoformat()
                
                # Detect new peaks
                detected_peaks = self.detect_peaks_for_spectrum(intensities)
                
                # Check minimum peaks requirement
                if (self.require_minimum_peaks.isChecked() and 
                    len(detected_peaks) < self.min_peaks_spin.value()):
                    skipped += 1
                    processed += 1
                    self.progress_bar.setValue(processed)
                    continue
                
                # Convert peak indices to wavenumber values for storage
                if len(wavenumbers) > 0:
                    peak_wavenumbers = wavenumbers[detected_peaks].tolist()
                else:
                    # If no wavenumbers, store indices
                    peak_wavenumbers = detected_peaks.tolist()
                
                # Update peaks in database - ensure we're modifying the actual database entry
                spectrum_data['peaks'] = peak_wavenumbers
                
                # CRITICAL: Also update the main database dictionary directly to ensure changes persist
                self.raman_db.database[spectrum_name]['peaks'] = peak_wavenumbers
                
                # Add processing metadata
                if 'metadata' not in spectrum_data:
                    spectrum_data['metadata'] = {}
                if 'metadata' not in self.raman_db.database[spectrum_name]:
                    self.raman_db.database[spectrum_name]['metadata'] = {}
                
                processing_metadata = {
                    'batch_peak_detection_timestamp': datetime.now().isoformat(),
                    'batch_peak_detection_params': {
                        'height_threshold': self.height_spin.value(),
                        'prominence_threshold': self.prominence_spin.value(),
                        'min_distance': self.distance_spin.value(),
                        'min_width': self.width_spin.value()
                    },
                    'batch_peak_count': len(detected_peaks)
                }
                
                spectrum_data['metadata'].update(processing_metadata)
                self.raman_db.database[spectrum_name]['metadata'].update(processing_metadata)
                
                print(f"DEBUG: Updated {spectrum_name} with {len(detected_peaks)} peaks")
                
                successful += 1
                
                # Immediate verification that the change was made
                verify_peaks = self.raman_db.database[spectrum_name].get('peaks', [])
                verify_metadata = self.raman_db.database[spectrum_name].get('metadata', {})
                has_batch_timestamp = 'batch_peak_detection_timestamp' in verify_metadata
                print(f"DEBUG: Verification - {spectrum_name} now has {len(verify_peaks)} peaks, batch metadata: {has_batch_timestamp}")
                
            except Exception as e:
                print(f"Error processing {spectrum_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                errors += 1
            
            processed += 1
            self.progress_bar.setValue(processed)
        
        # Save database with enhanced error checking
        try:
            print(f"DEBUG: Attempting to save database to: {self.raman_db.db_path}")
            print(f"DEBUG: Database type: {type(self.raman_db)}")
            print(f"DEBUG: Database has save_database method: {hasattr(self.raman_db, 'save_database')}")
            
            # Check database contents before save
            total_in_memory = len(self.raman_db.database)
            sample_spectrum = list(self.raman_db.database.keys())[0] if self.raman_db.database else None
            if sample_spectrum:
                sample_peaks = self.raman_db.database[sample_spectrum].get('peaks', [])
                sample_metadata = self.raman_db.database[sample_spectrum].get('metadata', {})
                has_batch_meta = 'batch_peak_detection_timestamp' in sample_metadata
                print(f"DEBUG: Sample check before save - {sample_spectrum}: {len(sample_peaks)} peaks, batch metadata: {has_batch_meta}")
            
            # Force a database save and verify it worked
            save_result = self.raman_db.save_database()
            print(f"DEBUG: Database save result: {save_result}")
            
            # Verify the database file was actually updated
            import os
            if os.path.exists(self.raman_db.db_path):
                file_stats = os.stat(self.raman_db.db_path)
                print(f"DEBUG: Database file size: {file_stats.st_size} bytes")
                print(f"DEBUG: Database last modified: {datetime.fromtimestamp(file_stats.st_mtime)}")
                
                # Try to reload database and check if changes persist
                print(f"DEBUG: Testing database reload...")
                original_database = self.raman_db.database.copy()
                
                try:
                    self.raman_db.load_database()
                    
                    # Check if our sample spectrum still has the new data
                    if sample_spectrum and sample_spectrum in self.raman_db.database:
                        reloaded_peaks = self.raman_db.database[sample_spectrum].get('peaks', [])
                        reloaded_metadata = self.raman_db.database[sample_spectrum].get('metadata', {})
                        reloaded_batch_meta = 'batch_peak_detection_timestamp' in reloaded_metadata
                        print(f"DEBUG: After reload - {sample_spectrum}: {len(reloaded_peaks)} peaks, batch metadata: {reloaded_batch_meta}")
                        
                        if len(reloaded_peaks) != len(sample_peaks) or not reloaded_batch_meta:
                            print("DEBUG: ⚠️ WARNING: Data was not properly saved! Changes lost after reload.")
                            # Restore the original data
                            self.raman_db.database = original_database
                        else:
                            print("DEBUG: ✅ SUCCESS: Data persisted after reload!")
                    
                except Exception as reload_error:
                    print(f"DEBUG: Error during reload test: {reload_error}")
                    # Restore original data if reload failed
                    self.raman_db.database = original_database
                
            else:
                print(f"DEBUG: Database file not found at {self.raman_db.db_path}")
            
            save_success = True
        except Exception as e:
            save_success = False
            error_msg = f"Failed to save database: {str(e)}"
            print(f"DEBUG: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Save Error", error_msg)
        
        # Show results
        if save_success:
            # Verify the data was actually saved by checking a few random spectra
            verification_success = True
            verification_details = []
            
            sample_names = list(self.raman_db.database.keys())[:3]  # Check first 3 spectra
            for name in sample_names:
                spectrum_data = self.raman_db.database.get(name)
                if spectrum_data:
                    peaks = spectrum_data.get('peaks', [])
                    metadata = spectrum_data.get('metadata', {})
                    has_batch_metadata = 'batch_peak_detection_timestamp' in metadata
                    
                    verification_details.append(f"  {name}: {len(peaks)} peaks, batch metadata: {has_batch_metadata}")
                    
                    if not has_batch_metadata and successful > 0:
                        verification_success = False
            
            results_text = f"Batch Peak Detection Complete!\n\n"
            results_text += f"📊 Processing Results:\n"
            results_text += f"   Total Spectra: {total_spectra}\n"
            results_text += f"   Successfully Processed: {successful}\n"
            results_text += f"   Skipped: {skipped}\n"
            results_text += f"   Errors: {errors}\n\n"
            results_text += f"Database saved successfully!\n\n"
            
            if verification_success:
                results_text += f"✅ Verification: Changes appear to be saved correctly\n\n"
            else:
                results_text += f"⚠️ Verification: Some changes may not have been saved properly\n\n"
            
            results_text += f"Sample verification:\n"
            results_text += "\n".join(verification_details)
            results_text += f"\n\n💡 Tip: Use the main browser to verify results."
            
            QMessageBox.information(self, "Processing Complete", results_text)
            self.status_label.setText("Batch processing completed successfully!")
        
        self.close()
    
    def test_database_save(self):
        """Test the database save functionality for debugging."""
        print("DEBUG: ========== DATABASE SAVE TEST ==========")
        
        try:
            # Get a sample spectrum to modify
            if not self.raman_db.database:
                QMessageBox.warning(self, "No Data", "No spectra in database to test with.")
                return
            
            sample_name = list(self.raman_db.database.keys())[0]
            sample_spectrum = self.raman_db.database[sample_name]
            
            print(f"DEBUG: Testing with spectrum: {sample_name}")
            
            # Save original state
            original_peaks = sample_spectrum.get('peaks', []).copy()
            original_metadata = sample_spectrum.get('metadata', {}).copy()
            
            print(f"DEBUG: Original peaks count: {len(original_peaks)}")
            print(f"DEBUG: Original has test metadata: {'test_save_timestamp' in original_metadata}")
            
            # Make a temporary change
            test_peaks = [100.0, 200.0, 300.0]  # Test wavenumber values
            test_timestamp = datetime.now().isoformat()
            
            self.raman_db.database[sample_name]['peaks'] = test_peaks
            if 'metadata' not in self.raman_db.database[sample_name]:
                self.raman_db.database[sample_name]['metadata'] = {}
            self.raman_db.database[sample_name]['metadata']['test_save_timestamp'] = test_timestamp
            
            print(f"DEBUG: Set test peaks: {test_peaks}")
            print(f"DEBUG: Set test timestamp: {test_timestamp}")
            
            # Try to save
            print(f"DEBUG: Attempting save to: {self.raman_db.db_path}")
            save_result = self.raman_db.save_database()
            print(f"DEBUG: Save result: {save_result}")
            
            # Check file modification time
            import os
            if os.path.exists(self.raman_db.db_path):
                file_stats = os.stat(self.raman_db.db_path)
                print(f"DEBUG: File modified: {datetime.fromtimestamp(file_stats.st_mtime)}")
            
            # Try to reload and verify
            self.raman_db.load_database()
            
            reloaded_spectrum = self.raman_db.database.get(sample_name)
            if reloaded_spectrum:
                reloaded_peaks = reloaded_spectrum.get('peaks', [])
                reloaded_metadata = reloaded_spectrum.get('metadata', {})
                reloaded_timestamp = reloaded_metadata.get('test_save_timestamp')
                
                print(f"DEBUG: After reload - peaks: {reloaded_peaks}")
                print(f"DEBUG: After reload - timestamp: {reloaded_timestamp}")
                
                if reloaded_peaks == test_peaks and reloaded_timestamp == test_timestamp:
                    result_msg = "✅ SUCCESS: Database save and reload working correctly!"
                    print(f"DEBUG: {result_msg}")
                    QMessageBox.information(self, "Test Result", result_msg)
                else:
                    result_msg = f"❌ FAILURE: Data not saved correctly!\nExpected peaks: {test_peaks}\nActual peaks: {reloaded_peaks}\nExpected timestamp: {test_timestamp}\nActual timestamp: {reloaded_timestamp}"
                    print(f"DEBUG: {result_msg}")
                    QMessageBox.warning(self, "Test Result", result_msg)
            else:
                result_msg = "❌ FAILURE: Spectrum not found after reload!"
                print(f"DEBUG: {result_msg}")
                QMessageBox.critical(self, "Test Result", result_msg)
            
            # Restore original state
            self.raman_db.database[sample_name]['peaks'] = original_peaks
            self.raman_db.database[sample_name]['metadata'] = original_metadata
            self.raman_db.save_database()
            
            print("DEBUG: Restored original state")
            
        except Exception as e:
            error_msg = f"Error during database save test: {str(e)}"
            print(f"DEBUG: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Test Error", error_msg)
        
        print("DEBUG: ========== TEST COMPLETE ==========")
    
    def test_peak_detection_comparison(self):
        """Test peak detection to compare with individual detection."""
        print("DEBUG: ========== PEAK DETECTION COMPARISON TEST ==========")
        
        try:
            # Get the parent browser to access individual detection parameters
            browser = self.parent()
            if not browser or not hasattr(browser, 'current_spectrum'):
                QMessageBox.warning(self, "No Data", "No current spectrum in browser to test with.")
                return
            
            current_spectrum = browser.current_spectrum
            if not current_spectrum:
                QMessageBox.warning(self, "No Data", "Please select a spectrum in the main browser first.")
                return
            
            wavenumbers = np.array(current_spectrum['wavenumbers'])
            intensities = np.array(current_spectrum['intensities'])
            
            print(f"DEBUG: Testing with spectrum having {len(intensities)} data points")
            print(f"DEBUG: Intensity range: {intensities.min():.3f} to {intensities.max():.3f}")
            
            # Get individual detection parameters
            if hasattr(browser, 'height_spin'):
                individual_height = browser.height_spin.value()
                individual_prominence = browser.prominence_spin.value() 
                individual_distance = browser.distance_spin.value()
                
                print(f"DEBUG: Individual detection params - height: {individual_height}, prominence: {individual_prominence}, distance: {individual_distance}")
                
                # Set batch parameters to match individual
                self.height_spin.setValue(individual_height)
                self.prominence_spin.setValue(individual_prominence)
                self.distance_spin.setValue(individual_distance)
                
                print(f"DEBUG: Set batch params to match individual")
            else:
                print("DEBUG: Individual detection controls not found, using batch defaults")
            
            # Get batch detection parameters
            batch_height = self.height_spin.value()
            batch_prominence = self.prominence_spin.value()
            batch_distance = self.distance_spin.value()
            batch_width = self.width_spin.value()
            
            print(f"DEBUG: Batch detection params - height: {batch_height}, prominence: {batch_prominence}, distance: {batch_distance}, width: {batch_width}")
            
            # Test individual-style detection
            max_intensity = np.max(intensities)
            individual_height_threshold = batch_height * max_intensity
            individual_prominence_threshold = batch_prominence * max_intensity
            
            print(f"DEBUG: Individual-style thresholds - height: {individual_height_threshold:.3f}, prominence: {individual_prominence_threshold:.3f}")
            
            # Detect peaks using individual method style
            individual_peaks, _ = find_peaks(
                intensities,
                height=individual_height_threshold,
                prominence=individual_prominence_threshold, 
                distance=batch_distance
            )
            
            print(f"DEBUG: Individual-style detection found {len(individual_peaks)} peaks")
            if len(individual_peaks) > 0:
                individual_wavenumbers = wavenumbers[individual_peaks]
                print(f"DEBUG: Individual peak positions: {individual_wavenumbers[:10]}")  # Show first 10
            
            # Test batch detection method
            batch_peaks = self.detect_peaks_for_spectrum(intensities)
            
            print(f"DEBUG: Batch detection found {len(batch_peaks)} peaks")
            if len(batch_peaks) > 0:
                batch_wavenumbers = wavenumbers[batch_peaks]
                print(f"DEBUG: Batch peak positions: {batch_wavenumbers[:10]}")  # Show first 10
            
            # Compare results
            if np.array_equal(individual_peaks, batch_peaks):
                result_msg = "✅ SUCCESS: Peak detection methods match perfectly!"
                print(f"DEBUG: {result_msg}")
                QMessageBox.information(self, "Test Result", result_msg)
            else:
                diff_count = abs(len(individual_peaks) - len(batch_peaks))
                result_msg = f"❌ MISMATCH: Peak detection differs!\n\nIndividual method: {len(individual_peaks)} peaks\nBatch method: {len(batch_peaks)} peaks\nDifference: {diff_count} peaks\n\nSee console for detailed peak positions."
                print(f"DEBUG: {result_msg}")
                QMessageBox.warning(self, "Test Result", result_msg)
            
        except Exception as e:
            error_msg = f"Error during peak detection test: {str(e)}"
            print(f"DEBUG: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Test Error", error_msg)
        
        print("DEBUG: ========== PEAK DETECTION TEST COMPLETE ==========")

    def verify_current_spectrum_detection(self):
        """Verify that batch and individual detection give same results for currently selected spectrum."""
        # Get the currently selected spectrum from the parent browser
        parent_browser = self.parent()
        if not hasattr(parent_browser, 'current_spectrum') or not parent_browser.current_spectrum:
            QMessageBox.warning(self, "No Selection", "Please select a spectrum in the main browser first.")
            return
            
        current_spectrum = parent_browser.current_spectrum
        spectrum_name = parent_browser.spectrum_list.currentItem().text() if parent_browser.spectrum_list.currentItem() else "Unknown"
        
        intensities = np.array(current_spectrum.get('intensities', []))
        wavenumbers = np.array(current_spectrum.get('wavenumbers', []))
        
        if len(intensities) == 0:
            QMessageBox.warning(self, "No Data", f"No intensity data found for {spectrum_name}")
            return
        
        # Get current parameters from the main browser (individual detection)
        main_height = parent_browser.height_spin.value()
        main_prominence = parent_browser.prominence_spin.value() 
        main_distance = parent_browser.distance_spin.value()
        
        # Get current parameters from batch dialog
        batch_height = self.height_spin.value()
        batch_prominence = self.prominence_spin.value()
        batch_distance = self.distance_spin.value()
        
        # Simulate individual peak detection (exactly as done in main browser)
        max_intensity = np.max(intensities)
        height_threshold = main_height * max_intensity
        prominence_threshold = main_prominence * max_intensity
        
        individual_peaks, _ = find_peaks(
            intensities,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=main_distance
        )
        
        # Test batch detection with same parameters as main browser
        old_height = self.height_spin.value()
        old_prominence = self.prominence_spin.value()
        old_distance = self.distance_spin.value()
        
        self.height_spin.setValue(main_height)
        self.prominence_spin.setValue(main_prominence)
        self.distance_spin.setValue(main_distance)
        
        batch_peaks = self.detect_peaks_for_spectrum(intensities)
        
        # Restore original batch parameters
        self.height_spin.setValue(old_height)
        self.prominence_spin.setValue(old_prominence)
        self.distance_spin.setValue(old_distance)
        
        # Create results dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Verification Results: {spectrum_name}")
        dialog.setMinimumSize(700, 600)
        layout = QVBoxLayout(dialog)
        
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        layout.addWidget(text_widget)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        # Generate comparison report
        results = []
        results.append(f"=== VERIFICATION REPORT: {spectrum_name} ===\n")
        
        results.append("PARAMETERS COMPARISON:")
        results.append(f"Main Browser -> Height: {main_height:.3f}, Prominence: {main_prominence:.3f}, Distance: {main_distance}")
        results.append(f"Batch Dialog -> Height: {batch_height:.3f}, Prominence: {batch_prominence:.3f}, Distance: {batch_distance}")
        param_match = (main_height == batch_height and main_prominence == batch_prominence and main_distance == batch_distance)
        results.append(f"Parameters Match: {'✓ YES' if param_match else '✗ NO'}")
        results.append("")
        
        results.append("PEAK DETECTION RESULTS:")
        results.append(f"Individual Method: {len(individual_peaks)} peaks")
        results.append(f"Batch Method: {len(batch_peaks)} peaks")
        
        peaks_match = np.array_equal(individual_peaks, batch_peaks)
        results.append(f"Results Match: {'✓ YES' if peaks_match else '✗ NO'}")
        results.append("")
        
        if len(wavenumbers) > 0:
            results.append("PEAK POSITIONS:")
            if len(individual_peaks) > 0:
                individual_positions = wavenumbers[individual_peaks]
                results.append(f"Individual: {[f'{p:.1f}' for p in individual_positions]}")
            else:
                results.append("Individual: No peaks found")
                
            if len(batch_peaks) > 0:
                batch_positions = wavenumbers[batch_peaks]
                results.append(f"Batch: {[f'{p:.1f}' for p in batch_positions]}")
            else:
                results.append("Batch: No peaks found")
        
        results.append("")
        results.append("STORED PEAKS IN DATABASE:")
        stored_peaks = current_spectrum.get('peaks', [])
        if stored_peaks:
            if isinstance(stored_peaks, (list, tuple)) and len(stored_peaks) > 0:
                # Check if these are wavenumbers or indices
                if len(wavenumbers) > 0 and all(isinstance(p, (int, float)) for p in stored_peaks):
                    max_wn = max(wavenumbers)
                    if all(p <= max_wn for p in stored_peaks):
                        results.append(f"Stored: {[f'{p:.1f}' for p in stored_peaks]} (wavenumbers)")
                    else:
                        # Likely indices - convert to wavenumbers
                        try:
                            stored_positions = [wavenumbers[int(p)] for p in stored_peaks if 0 <= int(p) < len(wavenumbers)]
                            results.append(f"Stored: {[f'{p:.1f}' for p in stored_positions]} (converted from indices)")
                        except:
                            results.append(f"Stored: {stored_peaks} (indices - conversion failed)")
                else:
                    results.append(f"Stored: {stored_peaks}")
            else:
                results.append("Stored: Empty list")
        else:
            results.append("Stored: No peaks stored")
        
        results.append("")
        if not peaks_match:
            results.append("⚠️ ISSUE DETECTED: Individual and batch detection produce different results!")
            results.append("This indicates a bug in the peak detection algorithms.")
            results.append("")
            results.append("DEBUGGING INFO:")
            results.append(f"Max intensity: {max_intensity:.3f}")
            results.append(f"Height threshold: {height_threshold:.3f}")
            results.append(f"Prominence threshold: {prominence_threshold:.3f}")
        else:
            results.append("✅ SUCCESS: Both methods produce identical results!")
            if not param_match:
                results.append("Note: Parameters were different but results matched after synchronization.")
        
        text_widget.setPlainText('\n'.join(results))
        dialog.exec_()


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
        