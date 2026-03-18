#!/usr/bin/env python3
"""
Peak Parameter Discriminant Analysis Tool for RamanLab
=======================================================
Generalized tool for discriminating between material phases/types based on
peak fitting parameters (center, FWHM, R², etc.) from batch processing results.

Features:
- GUI-based group definition with filename pattern matching
- Automated discriminant plot generation
- Linear Discriminant Analysis (LDA) with LOOCV validation
- Statistical summaries and export capabilities
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import fnmatch
import warnings
warnings.filterwarnings('ignore')

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QLineEdit, QTextEdit, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QSpinBox, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QTabWidget, QCheckBox,
    QProgressDialog, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. LDA analysis will be disabled.")


class PeakDiscriminantAnalysisDialog(QDialog):
    """Dialog for peak parameter-based discriminant analysis."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Peak Parameter Discriminant Analysis")
        self.resize(1400, 900)
        
        # Data storage
        self.batch_results = None
        self.peaks_df = None
        self.groups = {}  # {group_name: [filename_patterns]}
        self.group_colors = {}
        self.selected_peak_number = 1
        
        # Color palette for groups
        self.color_palette = [
            '#2166ac', '#d6604d', '#4daf4a', '#984ea3', 
            '#ff7f00', '#a65628', '#f781bf', '#999999'
        ]
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Create splitter for left panel and right panel
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Plots and results
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 1000])
        layout.addWidget(splitter)
    
    def create_left_panel(self):
        """Create left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File loading section
        file_group = QGroupBox("1. Load Batch Results")
        file_layout = QVBoxLayout(file_group)
        
        load_btn = QPushButton("📂 Load batch_results.pkl")
        load_btn.clicked.connect(self.load_batch_results)
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        file_layout.addWidget(load_btn)
        
        self.file_status_label = QLabel("No file loaded")
        self.file_status_label.setWordWrap(True)
        file_layout.addWidget(self.file_status_label)
        
        layout.addWidget(file_group)
        
        # Peak selection section
        peak_group = QGroupBox("2. Select Peak to Analyze")
        peak_layout = QVBoxLayout(peak_group)
        
        peak_h_layout = QHBoxLayout()
        peak_h_layout.addWidget(QLabel("Peak Number:"))
        self.peak_number_spin = QSpinBox()
        self.peak_number_spin.setMinimum(1)
        self.peak_number_spin.setMaximum(10)
        self.peak_number_spin.setValue(1)
        self.peak_number_spin.valueChanged.connect(self.update_peak_info)
        peak_h_layout.addWidget(self.peak_number_spin)
        peak_h_layout.addStretch()
        peak_layout.addLayout(peak_h_layout)
        
        self.peak_info_label = QLabel("Select a peak to analyze")
        self.peak_info_label.setWordWrap(True)
        peak_layout.addWidget(self.peak_info_label)
        
        layout.addWidget(peak_group)
        
        # Group definition section
        groups_group = QGroupBox("3. Define Classification Groups")
        groups_layout = QVBoxLayout(groups_group)
        
        # Auto-suggest button
        suggest_btn = QPushButton("🤖 Auto-Suggest Groups from Filenames")
        suggest_btn.clicked.connect(self.auto_suggest_groups)
        suggest_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                padding: 6px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        groups_layout.addWidget(suggest_btn)
        
        groups_layout.addWidget(QLabel("Or manually define groups:"))
        
        # Group name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Group Name:"))
        self.group_name_input = QLineEdit()
        self.group_name_input.setPlaceholderText("e.g., Anatase, Magnéli, Rutile")
        name_layout.addWidget(self.group_name_input)
        groups_layout.addLayout(name_layout)
        
        # Pattern input
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Filename Pattern:"))
        self.pattern_input = QLineEdit()
        self.pattern_input.setPlaceholderText("e.g., *switzerland*.txt or 2_03.txt")
        pattern_layout.addWidget(self.pattern_input)
        groups_layout.addLayout(pattern_layout)
        
        # Add group button
        add_group_btn = QPushButton("➕ Add Group")
        add_group_btn.clicked.connect(self.add_group)
        groups_layout.addWidget(add_group_btn)
        
        # Groups list
        self.groups_list = QListWidget()
        self.groups_list.setMaximumHeight(150)
        groups_layout.addWidget(self.groups_list)
        
        # Remove group button
        remove_group_btn = QPushButton("➖ Remove Selected Group")
        remove_group_btn.clicked.connect(self.remove_group)
        groups_layout.addWidget(remove_group_btn)
        
        layout.addWidget(groups_group)
        
        # Analysis controls
        analysis_group = QGroupBox("4. Run Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.run_analysis_btn = QPushButton("🚀 Generate Discriminant Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setEnabled(False)
        self.run_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 12pt;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        analysis_layout.addWidget(self.run_analysis_btn)
        
        # Export options
        export_btn = QPushButton("💾 Export Results")
        export_btn.clicked.connect(self.export_results)
        analysis_layout.addWidget(export_btn)
        
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self):
        """Create right panel with tabs for plots and results."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        self.tabs = QTabWidget()
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Courier", 10))
        summary_layout.addWidget(self.summary_text)
        self.tabs.addTab(summary_tab, "📊 Summary")
        
        # Plot tabs (will be created dynamically)
        self.plot_tabs = {}
        
        layout.addWidget(self.tabs)
        return panel
    
    def load_batch_results(self):
        """Load batch_results.pkl file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Batch Results",
            "",
            "Pickle Files (*.pkl);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'rb') as f:
                self.batch_results = pickle.load(f)
            
            # Extract peaks dataframe
            if 'peaks_df' in self.batch_results:
                self.peaks_df = self.batch_results['peaks_df']
            else:
                QMessageBox.warning(
                    self,
                    "Invalid File",
                    "The selected file does not contain 'peaks_df'. "
                    "Please select a valid batch_results.pkl file."
                )
                return
            
            # Update UI
            n_spectra = len(self.peaks_df['filename'].unique())
            n_peaks = len(self.peaks_df)
            self.file_status_label.setText(
                f"✅ Loaded: {Path(file_path).name}\n"
                f"Spectra: {n_spectra}\n"
                f"Total peaks: {n_peaks}"
            )
            
            # Update peak number range
            max_peak = self.peaks_df['peak_number'].max()
            self.peak_number_spin.setMaximum(int(max_peak))
            
            self.update_peak_info()
            self.run_analysis_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load batch results:\n{str(e)}"
            )
    
    def update_peak_info(self):
        """Update peak information display."""
        if self.peaks_df is None:
            return
        
        peak_num = self.peak_number_spin.value()
        peak_data = self.peaks_df[self.peaks_df['peak_number'] == peak_num]
        
        if len(peak_data) == 0:
            self.peak_info_label.setText(f"No data for peak {peak_num}")
            return
        
        center_range = f"{peak_data['peak_center'].min():.1f} - {peak_data['peak_center'].max():.1f}"
        fwhm_range = f"{peak_data['fwhm'].min():.1f} - {peak_data['fwhm'].max():.1f}"
        
        self.peak_info_label.setText(
            f"Peak {peak_num} found in {len(peak_data)} spectra\n"
            f"Center range: {center_range} cm⁻¹\n"
            f"FWHM range: {fwhm_range} cm⁻¹"
        )
    
    def add_group(self):
        """Add a new classification group."""
        group_name = self.group_name_input.text().strip()
        pattern = self.pattern_input.text().strip()
        
        if not group_name or not pattern:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter both group name and filename pattern."
            )
            return
        
        # Add to groups dictionary
        if group_name not in self.groups:
            self.groups[group_name] = []
            # Assign color
            color_idx = len(self.groups) - 1
            self.group_colors[group_name] = self.color_palette[color_idx % len(self.color_palette)]
        
        self.groups[group_name].append(pattern)
        
        # Update list display
        self.update_groups_list()
        
        # Clear inputs
        self.group_name_input.clear()
        self.pattern_input.clear()
    
    def remove_group(self):
        """Remove selected group."""
        current_item = self.groups_list.currentItem()
        if not current_item:
            return
        
        group_name = current_item.text().split(':')[0].strip()
        if group_name in self.groups:
            del self.groups[group_name]
            del self.group_colors[group_name]
            self.update_groups_list()
    
    def auto_suggest_groups(self):
        """Automatically suggest groups based on filename patterns."""
        if self.peaks_df is None:
            QMessageBox.warning(self, "No Data", "Please load batch results first.")
            return
        
        # Get unique filenames
        filenames = self.peaks_df['filename'].unique()
        
        if len(filenames) < 2:
            QMessageBox.information(
                self,
                "Insufficient Data",
                "Need at least 2 files to suggest groups."
            )
            return
        
        # Analyze filename patterns
        suggestions = self._analyze_filename_patterns(filenames)
        
        if not suggestions:
            QMessageBox.information(
                self,
                "No Patterns Found",
                "Could not identify clear grouping patterns in filenames.\n\n"
                "Try manually defining groups with patterns like:\n"
                "• *location_name* for location-based grouping\n"
                "• prefix_* for prefix-based grouping\n"
                "• *_color_* for color/type-based grouping"
            )
            return
        
        # Show suggestions dialog
        self.show_suggestions_dialog(suggestions, filenames)
    
    def _analyze_filename_patterns(self, filenames):
        """Analyze filenames and suggest grouping patterns."""
        import re
        from collections import defaultdict
        
        suggestions = []
        matched_files = set()  # Track which files have been matched
        
        # Strategy 1: Common words/tokens in filenames (case-insensitive)
        word_groups = defaultdict(list)
        for fname in filenames:
            # Extract words (split by _, -, space, numbers, dots)
            words = re.findall(r'[a-zA-Z]+', fname.lower())
            for word in words:
                if len(word) > 2:  # Lowered threshold to catch more patterns
                    word_groups[word].append(fname)
        
        # Find words that appear in multiple files but not all
        for word, files in word_groups.items():
            if 2 <= len(files) < len(filenames) * 0.9:  # 2+ files but not everything
                suggestions.append({
                    'name': word.capitalize(),
                    'pattern': f'*{word}*',
                    'count': len(files),
                    'files': files
                })
                matched_files.update(files)
        
        # Strategy 2: Numeric prefixes (e.g., "2_03.txt", "3_04.txt")
        numeric_prefix_groups = defaultdict(list)
        for fname in filenames:
            match = re.match(r'^(\d+)_', fname)
            if match:
                prefix = match.group(1)
                numeric_prefix_groups[prefix].append(fname)
        
        for prefix, files in numeric_prefix_groups.items():
            if len(files) >= 1:
                suggestions.append({
                    'name': f'Numeric_{prefix}',
                    'pattern': f'{prefix}_*',
                    'count': len(files),
                    'files': files
                })
                matched_files.update(files)
        
        # Strategy 3: Files with dots/periods (e.g., "1.01.txt", "2.03.txt")
        dot_pattern_groups = defaultdict(list)
        for fname in filenames:
            # Match patterns like "1.01", "2.03"
            match = re.match(r'^(\d+)\.', fname)
            if match:
                prefix = match.group(1)
                dot_pattern_groups[prefix].append(fname)
        
        for prefix, files in dot_pattern_groups.items():
            if len(files) >= 1:
                suggestions.append({
                    'name': f'Dot_{prefix}',
                    'pattern': f'{prefix}.*',
                    'count': len(files),
                    'files': files
                })
                matched_files.update(files)
        
        # Strategy 4: Color keywords
        color_keywords = ['yellow', 'blue', 'red', 'green', 'white', 'black', 'brown', 'pink', 'orange', 'purple']
        for color in color_keywords:
            matching = [f for f in filenames if color.lower() in f.lower()]
            if len(matching) >= 1:  # Even single matches
                suggestions.append({
                    'name': color.capitalize(),
                    'pattern': f'*{color}*',
                    'count': len(matching),
                    'files': matching
                })
                matched_files.update(matching)
        
        # Strategy 5: Location/country patterns
        location_keywords = ['switzerland', 'usa', 'canada', 'germany', 'france', 'italy', 
                           'spain', 'china', 'japan', 'australia', 'brazil', 'mexico']
        for location in location_keywords:
            matching = [f for f in filenames if location.lower() in f.lower()]
            if len(matching) >= 1:  # Even single matches
                suggestions.append({
                    'name': location.capitalize(),
                    'pattern': f'*{location}*',
                    'count': len(matching),
                    'files': matching
                })
                matched_files.update(matching)
        
        # Strategy 6: Common prefixes (first N characters)
        prefix_groups = defaultdict(list)
        for fname in filenames:
            # Try different prefix lengths
            for prefix_len in [3, 4, 5]:
                if len(fname) >= prefix_len:
                    prefix = fname[:prefix_len]
                    prefix_groups[prefix].append(fname)
        
        for prefix, files in prefix_groups.items():
            if 2 <= len(files) < len(filenames) * 0.9:
                suggestions.append({
                    'name': f'Prefix_{prefix}',
                    'pattern': f'{prefix}*',
                    'count': len(files),
                    'files': files
                })
                matched_files.update(files)
        
        # Strategy 7: Catch remaining unmatched files
        unmatched = set(filenames) - matched_files
        if unmatched:
            suggestions.append({
                'name': 'Other_Files',
                'pattern': '*',  # Will need manual refinement
                'count': len(unmatched),
                'files': list(unmatched)
            })
        
        # Remove duplicates (same files in multiple suggestions)
        # Keep only unique suggestions based on file sets
        seen_file_sets = set()
        unique_suggestions = []
        for suggestion in suggestions:
            file_set = frozenset(suggestion['files'])
            if file_set not in seen_file_sets:
                seen_file_sets.add(file_set)
                unique_suggestions.append(suggestion)
        
        # Sort by count (most files first)
        unique_suggestions.sort(key=lambda x: x['count'], reverse=True)
        
        return unique_suggestions
    
    def show_suggestions_dialog(self, suggestions, all_filenames):
        """Show dialog with suggested groups for user to select."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Suggested Groups")
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        info_label = QLabel(
            "✨ Select patterns to merge into a single group, or add them individually.\n"
            "Selected patterns will be combined into one classification group."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #E3F2FD;")
        layout.addWidget(info_label)
        
        # Table of suggestions
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(['Select', 'Pattern Name', 'Pattern', 'Matches'])
        table.setRowCount(len(suggestions))
        table.setSelectionBehavior(QTableWidget.SelectRows)
        
        checkboxes = []
        for i, suggestion in enumerate(suggestions):
            # Checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(False)  # Start unchecked for manual selection
            checkboxes.append(checkbox)
            table.setCellWidget(i, 0, checkbox)
            
            # Pattern name (not editable - just for reference)
            name_item = QTableWidgetItem(suggestion['name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 1, name_item)
            
            # Pattern (not editable)
            pattern_item = QTableWidgetItem(suggestion['pattern'])
            pattern_item.setFlags(pattern_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 2, pattern_item)
            
            # Match count
            count_item = QTableWidgetItem(f"{suggestion['count']} files")
            count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 3, count_item)
        
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(table)
        
        # Show example files for selected suggestion
        example_label = QLabel("Example files (click a row to preview):")
        layout.addWidget(example_label)
        
        example_list = QListWidget()
        example_list.setMaximumHeight(100)
        layout.addWidget(example_list)
        
        def show_examples(row, col):
            if 0 <= row < len(suggestions):
                example_list.clear()
                for fname in suggestions[row]['files'][:10]:  # Show first 10
                    example_list.addItem(fname)
                if len(suggestions[row]['files']) > 10:
                    example_list.addItem(f"... and {len(suggestions[row]['files']) - 10} more")
        
        table.cellClicked.connect(show_examples)
        
        # Group name input for merged group
        merge_layout = QHBoxLayout()
        merge_layout.addWidget(QLabel("Merged Group Name:"))
        merged_group_name_input = QLineEdit()
        merged_group_name_input.setPlaceholderText("Name for selected patterns (e.g., Anatase, Magnéli)")
        merge_layout.addWidget(merged_group_name_input)
        layout.addLayout(merge_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        merge_btn = QPushButton("🔗 Merge Selected into One Group")
        merge_btn.clicked.connect(lambda: self.apply_merged_suggestions(
            table, checkboxes, suggestions, merged_group_name_input, dialog
        ))
        merge_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
        """)
        button_layout.addWidget(merge_btn)
        
        add_individual_btn = QPushButton("➕ Add Selected as Separate Groups")
        add_individual_btn.clicked.connect(lambda: self.apply_suggestions(
            table, checkboxes, suggestions, dialog
        ))
        add_individual_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
        """)
        button_layout.addWidget(add_individual_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def apply_merged_suggestions(self, table, checkboxes, suggestions, group_name_input, dialog):
        """Merge selected patterns into a single group."""
        group_name = group_name_input.text().strip()
        
        if not group_name:
            QMessageBox.warning(
                self,
                "No Group Name",
                "Please enter a name for the merged group."
            )
            return
        
        # Collect selected patterns
        selected_patterns = []
        for i, checkbox in enumerate(checkboxes):
            if checkbox.isChecked():
                pattern = table.item(i, 2).text().strip()
                if pattern:
                    selected_patterns.append(pattern)
        
        if not selected_patterns:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select at least one pattern to merge."
            )
            return
        
        # Create or update group with all selected patterns
        if group_name not in self.groups:
            self.groups[group_name] = []
            # Assign color
            color_idx = len(self.groups) - 1
            self.group_colors[group_name] = self.color_palette[color_idx % len(self.color_palette)]
        
        # Add all patterns to this group
        for pattern in selected_patterns:
            if pattern not in self.groups[group_name]:
                self.groups[group_name].append(pattern)
        
        self.update_groups_list()
        dialog.accept()
        
        QMessageBox.information(
            self,
            "Group Created",
            f"Created group '{group_name}' with {len(selected_patterns)} pattern(s):\n\n" +
            "\n".join(f"  • {p}" for p in selected_patterns)
        )
    
    def apply_suggestions(self, table, checkboxes, suggestions, dialog):
        """Apply selected group suggestions as separate groups."""
        added_count = 0
        
        for i, checkbox in enumerate(checkboxes):
            if checkbox.isChecked():
                # Get values from table
                group_name = table.item(i, 1).text().strip()
                pattern = table.item(i, 2).text().strip()
                
                if group_name and pattern:
                    # Add as separate group
                    if group_name not in self.groups:
                        self.groups[group_name] = []
                        # Assign color
                        color_idx = len(self.groups) - 1
                        self.group_colors[group_name] = self.color_palette[color_idx % len(self.color_palette)]
                    
                    if pattern not in self.groups[group_name]:
                        self.groups[group_name].append(pattern)
                        added_count += 1
        
        self.update_groups_list()
        dialog.accept()
        
        if added_count > 0:
            QMessageBox.information(
                self,
                "Groups Added",
                f"Added {added_count} separate group(s)."
            )
    
    def update_groups_list(self):
        """Update the groups list display."""
        self.groups_list.clear()
        for group_name, patterns in self.groups.items():
            patterns_str = ', '.join(patterns)
            item_text = f"{group_name}: {patterns_str}"
            self.groups_list.addItem(item_text)
    
    def assign_spectra_to_groups(self, peak_data):
        """Assign spectra to groups based on filename patterns."""
        peak_data = peak_data.copy()
        peak_data['group'] = 'Unclassified'
        
        for group_name, patterns in self.groups.items():
            for pattern in patterns:
                # Match filenames
                mask = peak_data['filename'].apply(
                    lambda x: fnmatch.fnmatch(x, pattern)
                )
                peak_data.loc[mask, 'group'] = group_name
        
        return peak_data
    
    def run_analysis(self):
        """Run the discriminant analysis."""
        if self.peaks_df is None:
            QMessageBox.warning(self, "No Data", "Please load batch results first.")
            return
        
        if not self.groups:
            QMessageBox.warning(
                self,
                "No Groups",
                "Please define at least one classification group."
            )
            return
        
        try:
            # Extract selected peak data
            peak_num = self.peak_number_spin.value()
            peak_data = self.peaks_df[self.peaks_df['peak_number'] == peak_num].copy()
            
            if len(peak_data) == 0:
                QMessageBox.warning(
                    self,
                    "No Data",
                    f"No data found for peak {peak_num}"
                )
                return
            
            # Assign groups
            peak_data = self.assign_spectra_to_groups(peak_data)
            
            # Check if we have classified spectra
            classified = peak_data[peak_data['group'] != 'Unclassified']
            if len(classified) == 0:
                QMessageBox.warning(
                    self,
                    "No Matches",
                    "No spectra matched the defined patterns."
                )
                return
            
            # Generate summary
            self.generate_summary(peak_data)
            
            # Generate plots
            self.generate_plots(peak_data)
            
            # Run LDA if sklearn available and multiple groups
            unique_groups = peak_data[peak_data['group'] != 'Unclassified']['group'].unique()
            if SKLEARN_AVAILABLE and len(unique_groups) >= 2:
                self.run_lda_analysis(peak_data)
            
            QMessageBox.information(
                self,
                "Analysis Complete",
                "Discriminant analysis completed successfully!"
            )
            
        except Exception as e:
            import traceback
            QMessageBox.critical(
                self,
                "Analysis Error",
                f"Failed to run analysis:\n{str(e)}\n\n{traceback.format_exc()}"
            )
    
    def generate_summary(self, peak_data):
        """Generate statistical summary."""
        summary = f"PEAK PARAMETER DISCRIMINANT ANALYSIS\n"
        summary += f"{'='*60}\n\n"
        summary += f"Peak Number: {self.peak_number_spin.value()}\n"
        summary += f"Total Spectra: {len(peak_data)}\n\n"
        
        summary += f"CLASSIFICATION GROUPS:\n"
        summary += f"{'-'*60}\n"
        
        for group_name in sorted(self.groups.keys()):
            group_data = peak_data[peak_data['group'] == group_name]
            n = len(group_data)
            
            if n == 0:
                summary += f"\n{group_name}: No matches\n"
                continue
            
            summary += f"\n{group_name} (n={n}):\n"
            summary += f"  Peak Center:  {group_data['peak_center'].mean():.2f} ± {group_data['peak_center'].std():.2f} cm⁻¹\n"
            summary += f"                range [{group_data['peak_center'].min():.2f}, {group_data['peak_center'].max():.2f}]\n"
            summary += f"  FWHM:         {group_data['fwhm'].mean():.2f} ± {group_data['fwhm'].std():.2f} cm⁻¹\n"
            summary += f"                range [{group_data['fwhm'].min():.2f}, {group_data['fwhm'].max():.2f}]\n"
            
            if 'total_r2' in group_data.columns:
                summary += f"  R²:           {group_data['total_r2'].mean():.5f} ± {group_data['total_r2'].std():.5f}\n"
                summary += f"                range [{group_data['total_r2'].min():.5f}, {group_data['total_r2'].max():.5f}]\n"
        
        # Unclassified
        unclassified = peak_data[peak_data['group'] == 'Unclassified']
        if len(unclassified) > 0:
            summary += f"\nUnclassified: {len(unclassified)} spectra\n"
        
        self.summary_text.setText(summary)
    
    def generate_plots(self, peak_data):
        """Generate discriminant plots."""
        # Clear existing plot tabs
        for tab_name in list(self.plot_tabs.keys()):
            idx = self.tabs.indexOf(self.plot_tabs[tab_name])
            if idx >= 0:
                self.tabs.removeTab(idx)
        self.plot_tabs.clear()
        
        # Plot 1: Peak Center vs FWHM
        self.create_scatter_plot(
            peak_data,
            'peak_center',
            'fwhm',
            'Peak Center (cm⁻¹)',
            'FWHM (cm⁻¹)',
            'Peak Center vs FWHM',
            'center_vs_fwhm'
        )
        
        # Plot 2: R² vs Peak Center (if available)
        if 'total_r2' in peak_data.columns:
            self.create_scatter_plot(
                peak_data,
                'peak_center',
                'total_r2',
                'Peak Center (cm⁻¹)',
                'Fit Quality (R²)',
                'Peak Center vs R²',
                'center_vs_r2'
            )
            
            # Plot 3: FWHM vs R²
            self.create_scatter_plot(
                peak_data,
                'fwhm',
                'total_r2',
                'FWHM (cm⁻¹)',
                'Fit Quality (R²)',
                'FWHM vs R²',
                'fwhm_vs_r2'
            )
    
    def create_scatter_plot(self, peak_data, x_col, y_col, x_label, y_label, title, tab_name):
        """Create a scatter plot for discriminant analysis."""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot each group
        for group_name in sorted(self.groups.keys()):
            group_data = peak_data[peak_data['group'] == group_name]
            if len(group_data) == 0:
                continue
            
            color = self.group_colors[group_name]
            ax.scatter(
                group_data[x_col],
                group_data[y_col],
                c=color,
                s=80,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.5,
                label=f"{group_name} (n={len(group_data)})"
            )
            
            # Add 2σ confidence ellipse for groups with enough data
            if len(group_data) >= 3:
                self.add_confidence_ellipse(
                    ax,
                    group_data[x_col].values,
                    group_data[y_col].values,
                    color,
                    n_std=2.0
                )
        
        # Plot unclassified
        unclassified = peak_data[peak_data['group'] == 'Unclassified']
        if len(unclassified) > 0:
            ax.scatter(
                unclassified[x_col],
                unclassified[y_col],
                c='gray',
                s=40,
                alpha=0.3,
                marker='x',
                label=f"Unclassified (n={len(unclassified)})"
            )
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Create tab
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, tab)
        tab_layout.addWidget(toolbar)
        tab_layout.addWidget(canvas)
        
        self.tabs.addTab(tab, title)
        self.plot_tabs[tab_name] = tab
    
    def add_confidence_ellipse(self, ax, x, y, color, n_std=2.0):
        """Add confidence ellipse to plot."""
        cov = np.cov(x, y)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        
        ellipse = Ellipse(
            (np.mean(x), np.mean(y)),
            width,
            height,
            angle=angle,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
            linestyle='--',
            linewidth=1.5
        )
        ax.add_patch(ellipse)
    
    def run_lda_analysis(self, peak_data):
        """Run Linear Discriminant Analysis."""
        # Filter to classified data only
        classified = peak_data[peak_data['group'] != 'Unclassified'].copy()
        
        if len(classified) < 10:
            return  # Not enough data
        
        # Prepare data
        groups = classified['group'].values
        unique_groups = sorted(classified['group'].unique())
        
        # Encode groups as integers
        group_to_int = {g: i for i, g in enumerate(unique_groups)}
        y = np.array([group_to_int[g] for g in groups])
        
        # Features: peak_center, fwhm, total_r2 (if available)
        feature_cols = ['peak_center', 'fwhm']
        if 'total_r2' in classified.columns:
            feature_cols.append('total_r2')
        
        X = classified[feature_cols].values
        
        # Run LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        lda_scores = lda.transform(X)
        
        # LOOCV
        loo = LeaveOneOut()
        y_pred_loo = cross_val_predict(LinearDiscriminantAnalysis(), X, y, cv=loo)
        
        # Generate LDA summary
        lda_summary = f"\n\nLINEAR DISCRIMINANT ANALYSIS\n"
        lda_summary += f"{'='*60}\n\n"
        lda_summary += f"Features used: {', '.join(feature_cols)}\n"
        lda_summary += f"Number of groups: {len(unique_groups)}\n\n"
        
        lda_summary += "LOOCV Classification Report:\n"
        lda_summary += classification_report(y, y_pred_loo, target_names=unique_groups)
        
        lda_summary += f"\nLOOCV Accuracy: {accuracy_score(y, y_pred_loo):.4f}\n"
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred_loo)
        lda_summary += f"\nConfusion Matrix:\n"
        lda_summary += "Predicted →\n"
        header = "True ↓".ljust(20) + "  ".join([g[:10].ljust(10) for g in unique_groups])
        lda_summary += header + "\n"
        for i, group in enumerate(unique_groups):
            row = group[:18].ljust(20) + "  ".join([str(cm[i, j]).ljust(10) for j in range(len(unique_groups))])
            lda_summary += row + "\n"
        
        # Append to summary
        current_summary = self.summary_text.toPlainText()
        self.summary_text.setText(current_summary + lda_summary)
        
        # Create LDA plot
        self.create_lda_plot(lda_scores, y, unique_groups)
    
    def create_lda_plot(self, lda_scores, y, group_names):
        """Create LDA discriminant plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, group_name in enumerate(group_names):
            mask = y == i
            scores = lda_scores[mask, 0] if lda_scores.shape[1] > 0 else lda_scores[mask]
            jitter = np.random.normal(0, 0.05, len(scores))
            
            color = self.group_colors.get(group_name, 'gray')
            ax.scatter(
                scores,
                jitter,
                c=color,
                s=80,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.5,
                label=f"{group_name} (n={len(scores)})"
            )
        
        ax.set_xlabel('Linear Discriminant 1 (LD1)', fontsize=12)
        ax.set_ylabel('Jitter (display only)', fontsize=11)
        ax.set_title('Linear Discriminant Analysis', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])
        
        # Create tab
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, tab)
        tab_layout.addWidget(toolbar)
        tab_layout.addWidget(canvas)
        
        self.tabs.addTab(tab, "LDA Analysis")
        self.plot_tabs['lda'] = tab
    
    def export_results(self):
        """Export analysis results."""
        if self.peaks_df is None:
            QMessageBox.warning(self, "No Data", "No analysis results to export.")
            return
        
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder"
        )
        
        if not folder:
            return
        
        try:
            # Export summary text
            summary_path = Path(folder) / "discriminant_analysis_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(self.summary_text.toPlainText())
            
            # Export plots
            for tab_name, tab_widget in self.plot_tabs.items():
                # Find canvas in tab
                for child in tab_widget.findChildren(FigureCanvas):
                    fig = child.figure
                    plot_path = Path(folder) / f"plot_{tab_name}.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported to:\n{folder}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results:\n{str(e)}"
            )


def launch_peak_discriminant_analysis(parent=None):
    """Launch the Peak Discriminant Analysis dialog."""
    dialog = PeakDiscriminantAnalysisDialog(parent)
    dialog.exec()
