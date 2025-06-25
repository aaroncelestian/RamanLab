#!/usr/bin/env python3
"""
Qt6 Interface for Mixed Mineral Spectral Fitting with Database Integration
=======================================================================

Integrates the advanced mixed mineral spectral fitting capabilities with
the main RamanLab Qt6 application and mineral database for proper identification.
"""

import numpy as np
import sys
import sqlite3
import os
from pathlib import Path

# Import Qt6
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, 
    QPushButton, QTextEdit, QGroupBox, QProgressBar, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QCheckBox, QDoubleSpinBox, QSpinBox, QComboBox, QListWidget,
    QListWidgetItem, QApplication, QFrame, QInputDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor

# Import matplotlib for Qt6
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from polarization_ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar, apply_theme

# Import our spectral fitting module
from mixed_mineral_spectral_fitting import MixedMineralFitter, FittingMethod, PhaseInfo

# Import database functionality  
try:
    from raman_spectra_qt6 import RamanSpectraQt6
    RAMAN_DB_AVAILABLE = True
except ImportError:
    try:
        from core.database import MineralDatabase
        RAMAN_DB_AVAILABLE = False
    except ImportError:
        RAMAN_DB_AVAILABLE = False


class MineralSelectionDialog(QDialog):
    """Dialog for selecting mineral matches from database search results."""
    
    def __init__(self, parent, detected_peaks, search_results):
        super().__init__(parent)
        self.detected_peaks = detected_peaks
        self.search_results = search_results
        self.selected_mineral = None
        
        self.setup_ui()
        self.populate_results()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Select Mineral Match")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("🔍 Select the best mineral match for detected peaks:")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        layout.addWidget(header_label)
        
        # Detected peaks info
        peaks_info = QLabel(f"Detected Peaks: {', '.join(f'{p:.1f}' for p in self.detected_peaks[:10])} cm⁻¹")
        peaks_info.setStyleSheet("margin: 5px; color: #666;")
        layout.addWidget(peaks_info)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Rank", "Mineral Name", "Formula", "Match Score", "Key Peaks"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        layout.addWidget(self.results_table)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("📊 Preview Match")
        self.preview_btn.clicked.connect(self.preview_match)
        self.preview_btn.setEnabled(False)
        button_layout.addWidget(self.preview_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        select_btn = QPushButton("✅ Select This Mineral")
        select_btn.clicked.connect(self.accept_selection)
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        button_layout.addWidget(select_btn)
        
        layout.addLayout(button_layout)
        
        # Connect selection change
        self.results_table.selectionModel().selectionChanged.connect(
            self.on_selection_changed
        )
        
    def populate_results(self):
        """Populate the results table with search matches."""
        self.results_table.setRowCount(len(self.search_results))
        
        for i, result in enumerate(self.search_results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.results_table.setItem(i, 1, QTableWidgetItem(result['name']))
            self.results_table.setItem(i, 2, QTableWidgetItem(result['formula']))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{result['score']:.3f}"))
            
            # Format key peaks
            key_peaks = result.get('key_peaks', [])
            peaks_str = ', '.join(f'{p:.1f}' for p in key_peaks[:5])
            if len(key_peaks) > 5:
                peaks_str += "..."
            self.results_table.setItem(i, 4, QTableWidgetItem(peaks_str))
        
        # Auto-select the first (best) result
        if len(self.search_results) > 0:
            self.results_table.selectRow(0)
            
    def on_selection_changed(self):
        """Handle selection change."""
        selected_rows = self.results_table.selectionModel().selectedRows()
        self.preview_btn.setEnabled(len(selected_rows) > 0)
        
    def preview_match(self):
        """Preview the selected mineral match."""
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        mineral_data = self.search_results[row]
        
        # Create a simple preview dialog
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle(f"Preview: {mineral_data['name']}")
        preview_dialog.resize(500, 400)
        
        layout = QVBoxLayout(preview_dialog)
        
        # Mineral info
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_content = f"""
<h3>{mineral_data['name']}</h3>
<p><b>Formula:</b> {mineral_data['formula']}</p>
<p><b>Match Score:</b> {mineral_data['score']:.3f}</p>
<p><b>Crystal System:</b> {mineral_data.get('crystal_system', 'Unknown')}</p>

<h4>Expected Raman Peaks:</h4>
<ul>
"""
        for peak in mineral_data.get('raman_peaks', []):
            intensity = peak.get('intensity', 'medium')
            info_content += f"<li>{peak['frequency']:.1f} cm⁻¹ ({intensity})</li>"
        
        info_content += """
</ul>

<h4>Peak Matching:</h4>
<p>This mineral's expected peaks will be compared with your detected peaks using constrained fitting.</p>
"""
        info_text.setHtml(info_content)
        layout.addWidget(info_text)
        
        # Close button
        close_btn = QPushButton("Close Preview")
        close_btn.clicked.connect(preview_dialog.close)
        layout.addWidget(close_btn)
        
        preview_dialog.exec()
        
    def accept_selection(self):
        """Accept the selected mineral."""
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a mineral from the list.")
            return
            
        row = selected_rows[0].row()
        self.selected_mineral = self.search_results[row]
        self.accept()


class MajorPhaseConfirmationDialog(QDialog):
    """Dialog for confirming the detected major phase."""
    
    def __init__(self, parent, major_phase, search_results=None):
        super().__init__(parent)
        self.major_phase = major_phase
        self.search_results = search_results or []
        self.confirmed_phase = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Confirm Major Phase")
        self.setModal(True)
        self.resize(700, 500)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("🎯 Confirm Major Phase Detection")
        header_label.setStyleSheet("font-weight: bold; font-size: 16px; margin: 10px; color: #2C3E50;")
        layout.addWidget(header_label)
        
        # Detected phase info
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Box)
        info_frame.setStyleSheet("background-color: #F8F9FA; border: 1px solid #E9ECEF; border-radius: 8px; margin: 5px;")
        info_layout = QVBoxLayout(info_frame)
        
        phase_label = QLabel(f"<h3>Detected: {self.major_phase.name}</h3>")
        phase_label.setStyleSheet("color: #2C3E50; margin: 5px;")
        info_layout.addWidget(phase_label)
        
        details_layout = QHBoxLayout()
        
        # Left column
        left_info = QVBoxLayout()
        left_info.addWidget(QLabel(f"<b>Formula:</b> {self.major_phase.formula}"))
        left_info.addWidget(QLabel(f"<b>Confidence:</b> {self.major_phase.confidence:.3f}"))
        left_info.addWidget(QLabel(f"<b>Search Method:</b> {self.major_phase.search_method}"))
        left_info.addWidget(QLabel(f"<b>Est. Abundance:</b> {self.major_phase.abundance:.1%}"))
        details_layout.addLayout(left_info)
        
        # Right column - peaks
        right_info = QVBoxLayout()
        peaks_str = ", ".join(f"{p:.1f}" for p in self.major_phase.expected_peaks[:8])
        if len(self.major_phase.expected_peaks) > 8:
            peaks_str += "..."
        right_info.addWidget(QLabel(f"<b>Key Peaks (cm⁻¹):</b>"))
        peaks_label = QLabel(peaks_str)
        peaks_label.setWordWrap(True)
        peaks_label.setStyleSheet("color: #495057; font-family: monospace; margin-left: 10px;")
        right_info.addWidget(peaks_label)
        details_layout.addLayout(right_info)
        
        info_layout.addLayout(details_layout)
        layout.addWidget(info_frame)
        
        # Confidence indicator
        confidence_frame = QFrame()
        if self.major_phase.confidence >= 0.7:
            confidence_color = "#28A745"  # Green
            confidence_text = "High Confidence"
            confidence_icon = "✅"
        elif self.major_phase.confidence >= 0.5:
            confidence_color = "#FFC107"  # Yellow
            confidence_text = "Medium Confidence"
            confidence_icon = "⚠"
        else:
            confidence_color = "#DC3545"  # Red
            confidence_text = "Low Confidence - Review Recommended"
            confidence_icon = "⚠"
            
        confidence_frame.setStyleSheet(f"background-color: {confidence_color}15; border: 2px solid {confidence_color}; border-radius: 8px; margin: 5px;")
        confidence_layout = QHBoxLayout(confidence_frame)
        confidence_layout.addWidget(QLabel(f"{confidence_icon} {confidence_text}"))
        layout.addWidget(confidence_frame)
        
        # Alternative options if available
        if self.search_results and len(self.search_results) > 1:
            alternatives_label = QLabel("Alternative Options:")
            alternatives_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
            layout.addWidget(alternatives_label)
            
            self.alternatives_table = QTableWidget()
            self.alternatives_table.setColumnCount(4)
            self.alternatives_table.setHorizontalHeaderLabels([
                "Rank", "Mineral Name", "Formula", "Confidence"
            ])
            self.alternatives_table.horizontalHeader().setStretchLastSection(True)
            self.alternatives_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.alternatives_table.setMaximumHeight(150)
            
            # Populate alternatives (skip the first one as it's already shown)
            alt_results = self.search_results[1:6]  # Show up to 5 alternatives
            self.alternatives_table.setRowCount(len(alt_results))
            
            for i, result in enumerate(alt_results):
                self.alternatives_table.setItem(i, 0, QTableWidgetItem(str(i + 2)))
                self.alternatives_table.setItem(i, 1, QTableWidgetItem(result['name']))
                self.alternatives_table.setItem(i, 2, QTableWidgetItem(result.get('metadata', {}).get('formula', 'Unknown')))
                self.alternatives_table.setItem(i, 3, QTableWidgetItem(f"{result['score']:.3f}"))
            
            layout.addWidget(self.alternatives_table)
            
            self.alternatives_table.cellDoubleClicked.connect(self.select_alternative)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Manual selection button
        manual_btn = QPushButton("🔍 Manual Selection")
        manual_btn.clicked.connect(self.manual_selection)
        manual_btn.setToolTip("Choose from all database search results")
        button_layout.addWidget(manual_btn)
        
        button_layout.addStretch()
        
        # Cancel button
        cancel_btn = QPushButton("❌ Cancel Analysis")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        # Confirm button
        confirm_btn = QPushButton("✅ Confirm & Continue")
        confirm_btn.clicked.connect(self.confirm_phase)
        confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        button_layout.addWidget(confirm_btn)
        
        layout.addLayout(button_layout)
        
    def select_alternative(self, row, column):
        """Select an alternative phase from the table."""
        if hasattr(self, 'alternatives_table') and self.search_results:
            alt_result = self.search_results[row + 1]  # +1 because we skipped the first result
            
            # Create new phase from alternative
            from mixed_mineral_spectral_fitting import PhaseInfo
            alternative_phase = PhaseInfo(
                name=alt_result['name'],
                formula=alt_result.get('metadata', {}).get('formula', 'Unknown'),
                expected_peaks=alt_result.get('peaks', [])[:10],
                peak_tolerances=[5.0] * min(10, len(alt_result.get('peaks', []))),
                peak_intensities=[1.0] * min(10, len(alt_result.get('peaks', []))),
                constraints=[],
                confidence=alt_result['score'],
                abundance=0.8,
                search_method=alt_result.get('search_method', 'correlation'),
                database_metadata=alt_result.get('metadata', {})
            )
            
            self.confirmed_phase = alternative_phase
            self.accept()
        
    def manual_selection(self):
        """Open manual selection dialog."""
        if self.search_results:
            dialog = MineralSelectionDialog(self, self.major_phase.expected_peaks, self.search_results)
            if dialog.exec() == QDialog.Accepted and dialog.selected_mineral:
                # Create phase from selected mineral
                selected = dialog.selected_mineral
                from mixed_mineral_spectral_fitting import PhaseInfo
                
                manual_phase = PhaseInfo(
                    name=selected['name'],
                    formula=selected.get('formula', 'Unknown'),
                    expected_peaks=selected.get('key_peaks', selected.get('peaks', []))[:10],
                    peak_tolerances=[5.0] * min(10, len(selected.get('key_peaks', selected.get('peaks', [])))),
                    peak_intensities=[1.0] * min(10, len(selected.get('key_peaks', selected.get('peaks', [])))),
                    constraints=[],
                    confidence=selected['score'],
                    abundance=0.8,
                    search_method='manual_selection',
                    database_metadata=selected.get('metadata', {})
                )
                
                self.confirmed_phase = manual_phase
                self.accept()
        
    def confirm_phase(self):
        """Confirm the originally detected phase."""
        self.confirmed_phase = self.major_phase
        self.accept()


class MixedMineralAnalysisQt6(QDialog):
    """Qt6 interface for mixed mineral spectral analysis."""
    
    def __init__(self, parent, wavenumbers, intensities):
        super().__init__(parent)
        
        apply_theme('compact')
        
        self.parent = parent
        self.wavenumbers = np.array(wavenumbers)
        self.intensities = np.array(intensities)
        
        # Initialize database first
        if RAMAN_DB_AVAILABLE:
            self.raman_database = RamanSpectraQt6()  # This is the proven RamanLab database
        else:
            self.raman_database = None
        
        # Initialize the fitter with database integration
        self.fitter = MixedMineralFitter(self.wavenumbers, self.intensities, raman_db=self.raman_database)
        
        # Initialize legacy database for fallback
        try:
            from core.database import MineralDatabase
            self.mineral_database = MineralDatabase()
        except ImportError:
            self.mineral_database = None
            
        self.sql_database_path = None
        
        # Analysis state
        self.major_phase = None
        self.minor_phases = []
        self.global_fit_result = None
        self.quantification_results = None
        
        self.setup_ui()
        self.setWindowTitle("Mixed Mineral Spectral Analysis with Database")
        self.resize(1200, 800)
        
        # Initialize database after UI is set up (for logging)
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize mineral database connections."""
        try:
            # Log database system being used
            if self.raman_database is not None:
                self.log_status("✓ Using RamanLab correlation-based database system")
                
                # Check if RamanLab database has loaded data
                if hasattr(self.raman_database, 'database') and self.raman_database.database:
                    self.log_status(f"✓ RamanLab database loaded with {len(self.raman_database.database)} spectra")
                else:
                    self.log_status("⚠ RamanLab database appears empty - will load/search from SQL")
            else:
                self.log_status("⚠ RamanLab database not available - using fallback")
            
            # Look for SQLite database - including the specific file mentioned by user
            potential_sql_paths = [
                'RamanLab_Database_20250602.sqlite',  # User mentioned this specific file
                'RamanLab_Database.sqlite',
                'mineral_database.sqlite',
                os.path.join(os.path.dirname(__file__), 'RamanLab_Database_20250602.sqlite'),
                os.path.join(os.path.dirname(__file__), 'RamanLab_Database.sqlite'),
                os.path.join(os.path.dirname(__file__), '..', 'RamanLab_Database_20250602.sqlite'),
                os.path.join(os.path.dirname(__file__), '..', 'RamanLab_Database.sqlite'),
                # Check for common SQLite database names
                'raman_database.sqlite',
                'ramanlab.sqlite'
            ]
            
            for path in potential_sql_paths:
                if os.path.exists(path):
                    self.sql_database_path = path
                    self.log_status(f"✓ Found SQL database: {os.path.basename(path)}")
                    break
            
            if not self.sql_database_path:
                self.log_status("⚠ No SQL database found - using built-in data only")
            
            # Initialize legacy database if available
            if self.mineral_database is not None:
                try:
                    database_paths = [
                        'mineral_database.pkl',
                        'mineral_database.py', 
                        os.path.join(os.path.dirname(__file__), 'mineral_database.pkl'),
                        os.path.join(os.path.dirname(__file__), 'core', 'mineral_database.pkl'),
                        os.path.join(os.path.dirname(__file__), '..', 'mineral_database.pkl')
                    ]
                    
                    success = self.mineral_database.load_database(database_paths)
                    if success:
                        self.log_status(f"✓ Loaded fallback database with {len(self.mineral_database.mineral_list)} minerals")
                    else:
                        self.log_status("⚠ Using built-in minimal mineral database for fallback")
                except Exception as e:
                    self.log_status(f"Fallback database warning: {str(e)}")
            
        except Exception as e:
            self.log_status(f"Database initialization warning: {str(e)}")
    
    def search_minerals_by_peaks(self, peaks, max_results=15):
        """
        Search for minerals matching detected peaks.
        
        Parameters:
        -----------
        peaks : array-like
            Detected peak positions in cm⁻¹
        max_results : int
            Maximum number of results to return (default 15 to ensure good selection)
            
        Returns:
        --------
        list
            List of mineral matches with scores
        """
        results = []
        
        # Ensure peaks is a valid array
        if peaks is None or (hasattr(peaks, '__len__') and len(peaks) == 0):
            self.log_status("No peaks provided for mineral search")
            return results
        
        # Convert peaks to list if it's an array to avoid boolean context issues
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()
        
        self.log_status(f"Searching for minerals matching peaks: {[f'{p:.1f}' for p in peaks]}")
        
        # Calculate scores for all minerals first
        all_candidates = []
        for mineral_name, mineral_data in self.mineral_database.database.items():
            score = self._calculate_peak_match_score(peaks, mineral_data)
            self.log_status(f"  {mineral_data.get('name', mineral_name)}: score={score:.3f}")
            
            if score > 0.0:  # Include any non-zero match
                all_candidates.append({
                    'name': mineral_data.get('name', mineral_name),
                    'formula': mineral_data.get('formula', 'Unknown'),
                    'score': score,
                    'crystal_system': mineral_data.get('crystal_system', 'Unknown'),
                    'raman_peaks': mineral_data.get('raman_modes', []),
                    'key_peaks': [mode['frequency'] for mode in mineral_data.get('raman_modes', [])],
                    'source': 'built-in'
                })
        
        # Sort by score (highest first)
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Use tiered thresholds to ensure we get at least 10 results while maintaining quality
        thresholds = [0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]  # Start strict, then relax
        min_results_target = 10
        
        self.log_status(f"Using adaptive threshold approach to ensure at least {min_results_target} results...")
        
        for threshold in thresholds:
            results = [candidate for candidate in all_candidates if candidate['score'] >= threshold]
            
            self.log_status(f"With threshold {threshold:.2f}: {len(results)} candidates")
            
            if len(results) >= min_results_target:
                self.log_status(f"✓ Found {len(results)} candidates with threshold {threshold:.2f} - sufficient results!")
                break
        
        # If we still don't have enough, take the top scoring ones regardless of threshold
        if len(results) < min_results_target:
            results = all_candidates[:min_results_target]
            self.log_status(f"Using top {len(results)} candidates (all available, relaxed thresholds)")
        
        # Search in SQL database if available
        if self.sql_database_path:
            try:
                sql_results = self._search_sql_database(peaks, max_results)
                results.extend(sql_results)
            except Exception as e:
                self.log_status(f"SQL search warning: {str(e)}")
        
        # Only add synthetic matches if we still don't have enough results after everything
        if len(results) < min_results_target:
            needed = min_results_target - len(results)
            self.log_status(f"Adding {needed} synthetic matches to reach {min_results_target} total results")
            
            for i, peak in enumerate(peaks[:needed]):  # Use detected peaks
                synthetic_peaks = [peak, peak*0.95, peak*1.05]  # Very close related peaks
                results.append({
                    'name': f'Synthetic_Match_{i+1}',
                    'formula': 'Unknown',
                    'score': 0.01 - i*0.001,  # Very low scores
                    'crystal_system': 'Unknown',
                    'raman_peaks': [{'frequency': p, 'intensity': 'medium'} for p in synthetic_peaks],
                    'key_peaks': synthetic_peaks,
                    'source': 'synthetic'
                })
        
        # Final sort and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        final_results = results[:max_results]
        
        self.log_status(f"Final results: {len(final_results)} candidates")
        self.log_status(f"Best match: {final_results[0]['name']} (score: {final_results[0]['score']:.3f})")
        if len(final_results) > 1:
            self.log_status(f"Second best: {final_results[1]['name']} (score: {final_results[1]['score']:.3f})")
        
        return final_results
    
    def _calculate_peak_match_score(self, detected_peaks, mineral_data):
        """Calculate how well detected peaks match a mineral's expected peaks."""
        raman_modes = mineral_data.get('raman_modes', [])
        if not raman_modes:
            return 0.0
        
        expected_peaks = [mode['frequency'] for mode in raman_modes]
        if not expected_peaks:
            return 0.0
        
        # Use stricter tolerance for better precision
        tolerance = 8.0  # Reduced from 25.0 to 8.0 cm⁻¹ for stricter matching
        matched_peaks = 0
        total_intensity_weight = 0
        exact_matches = 0  # Count very close matches (within 3 cm⁻¹)
        
        # Forward matching (expected -> detected) with penalty for missing key peaks
        for expected_peak in expected_peaks:
            # Find closest detected peak
            distances = [abs(dp - expected_peak) for dp in detected_peaks]
            min_distance = min(distances) if distances else float('inf')
            
            if min_distance <= tolerance:
                # Weight by intensity and proximity
                intensity_weight = self._get_intensity_weight(
                    next((mode for mode in raman_modes if mode['frequency'] == expected_peak), {})
                )
                proximity_factor = 1.0 - (min_distance / tolerance)
                total_intensity_weight += intensity_weight * proximity_factor
                matched_peaks += 1
                
                # Bonus for very close matches
                if min_distance <= 3.0:
                    exact_matches += 1
        
        # Reverse matching (detected -> expected) - penalize extra unmatched peaks
        reverse_matches = 0
        for detected_peak in detected_peaks:
            distances = [abs(ep - detected_peak) for ep in expected_peaks]
            min_distance = min(distances) if distances else float('inf')
            if min_distance <= tolerance:
                reverse_matches += 1
        
        # Calculate comprehensive score
        if len(expected_peaks) == 0:
            return 0.0
        
        # Coverage: what fraction of expected peaks are matched
        coverage_score = matched_peaks / len(expected_peaks)
        
        # Precision: what fraction of detected peaks match expected
        precision_score = reverse_matches / max(len(detected_peaks), 1) if detected_peaks else 0
        
        # Intensity-weighted score
        intensity_score = total_intensity_weight / max(len(expected_peaks), 1)
        
        # Exact match bonus
        exact_match_bonus = exact_matches / max(len(detected_peaks), 1) if detected_peaks else 0
        
        # Weighted combination favoring coverage and exact matches
        final_score = (coverage_score * 0.4 + 
                      precision_score * 0.3 + 
                      intensity_score * 0.2 + 
                      exact_match_bonus * 0.1)
        
        # Apply penalties for poor matches
        if coverage_score < 0.3:  # Less than 30% of expected peaks matched
            final_score *= 0.5
        
        if precision_score < 0.5:  # More than 50% of detected peaks are unmatched
            final_score *= 0.7
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def _get_intensity_weight(self, raman_mode):
        """Get intensity weight from raman mode data."""
        intensity_map = {
            'very_strong': 1.0,
            'strong': 0.8,
            'medium': 0.6,
            'weak': 0.4,
            'very_weak': 0.2
        }
        
        intensity = raman_mode.get('intensity', 'medium')
        return intensity_map.get(intensity, 0.6)
    
    def _search_sql_database(self, peaks, max_results=10):
        """Search the SQL database for mineral matches."""
        results = []
        
        # Ensure peaks is valid
        if peaks is None or (hasattr(peaks, '__len__') and len(peaks) == 0):
            return results
        
        try:
            with sqlite3.connect(self.sql_database_path) as conn:
                cursor = conn.cursor()
                
                # Query spectra with peaks data
                cursor.execute("""
                    SELECT name, metadata 
                    FROM spectra 
                    WHERE peaks IS NOT NULL 
                    AND peaks != ''
                    LIMIT 1000
                """)
                
                # Collect all candidates first
                all_sql_candidates = []
                
                for row in cursor.fetchall():
                    spectrum_name, metadata_str = row
                    
                    try:
                        # Parse metadata to extract mineral info
                        import json
                        metadata = json.loads(metadata_str) if metadata_str else {}
                        
                        # Get mineral name and formula from metadata
                        mineral_name = metadata.get('mineral_name', spectrum_name)
                        formula = metadata.get('formula', 'Unknown')
                        
                        # Extract peak positions from metadata or name
                        spectrum_peaks = self._extract_peaks_from_spectrum_data(row, conn)
                        
                        if spectrum_peaks and len(spectrum_peaks) > 0:
                            score = self._calculate_peak_match_score_simple(peaks, spectrum_peaks)
                            
                            if score > 0.0:  # Include any non-zero match
                                all_sql_candidates.append({
                                    'name': mineral_name,
                                    'formula': formula,
                                    'score': score,
                                    'crystal_system': metadata.get('crystal_system', 'Unknown'),
                                    'raman_peaks': [{'frequency': p, 'intensity': 'medium'} for p in spectrum_peaks],
                                    'key_peaks': spectrum_peaks,
                                    'source': 'database'
                                })
                    except Exception as e:
                        continue  # Skip problematic entries
                
                # Sort by score and apply tiered thresholds like the main search
                all_sql_candidates.sort(key=lambda x: x['score'], reverse=True)
                
                # Use more lenient thresholds for SQL database to get more variety
                sql_thresholds = [0.15, 0.1, 0.05, 0.02, 0.01]
                target_sql_results = min(5, max_results // 2)  # Aim for about half the results from SQL
                
                for threshold in sql_thresholds:
                    sql_results = [candidate for candidate in all_sql_candidates if candidate['score'] >= threshold]
                    self.log_status(f"SQL DB with threshold {threshold:.2f}: {len(sql_results)} candidates")
                    
                    if len(sql_results) >= target_sql_results:
                        break
                
                # Take the best ones up to our target
                results = sql_results[:target_sql_results]
                
                if len(results) > 0:
                    self.log_status(f"Selected {len(results)} candidates from SQL database")
                    for result in results[:3]:  # Log top 3
                        self.log_status(f"  SQL: {result['name']} (score: {result['score']:.3f})")
                        
        except Exception as e:
            raise Exception(f"SQL database search failed: {str(e)}")
        
        return results
    
    def _extract_peaks_from_spectrum_data(self, row, cursor):
        """Extract peak positions from spectrum database entry."""
        spectrum_name = row[0]
        
        # Try to get peaks from stored peaks data
        cursor.execute("SELECT peaks FROM spectra WHERE name = ?", (spectrum_name,))
        peaks_result = cursor.fetchone()
        
        if peaks_result and peaks_result[0]:
            try:
                # Parse stored peaks (assuming JSON format)
                import json
                peaks_data = json.loads(peaks_result[0])
                if isinstance(peaks_data, list):
                    return [float(p) for p in peaks_data[:20]]  # Limit to 20 peaks
            except:
                pass
        
        return []
    
    def _calculate_peak_match_score_simple(self, detected_peaks, reference_peaks):
        """Simple peak matching score calculation with strict tolerance."""
        # Ensure inputs are valid
        if (reference_peaks is None or 
            (hasattr(reference_peaks, '__len__') and len(reference_peaks) == 0) or
            detected_peaks is None or
            (hasattr(detected_peaks, '__len__') and len(detected_peaks) == 0)):
            return 0.0
        
        # Convert arrays to lists to avoid boolean context issues
        if isinstance(reference_peaks, np.ndarray):
            reference_peaks = reference_peaks.tolist()
        if isinstance(detected_peaks, np.ndarray):
            detected_peaks = detected_peaks.tolist()
        
        tolerance = 8.0  # Reduced from 25.0 for stricter matching
        matched_forward = 0
        matched_reverse = 0
        exact_matches = 0
        
        # Forward matching: reference -> detected
        for ref_peak in reference_peaks:
            distances = [abs(dp - ref_peak) for dp in detected_peaks]
            if len(distances) > 0:
                min_distance = min(distances)
                if min_distance <= tolerance:
                    matched_forward += 1
                    if min_distance <= 3.0:  # Very close match bonus
                        exact_matches += 1
        
        # Reverse matching: detected -> reference
        for det_peak in detected_peaks:
            distances = [abs(rp - det_peak) for rp in reference_peaks]
            if len(distances) > 0:
                min_distance = min(distances)
                if min_distance <= tolerance:
                    matched_reverse += 1
        
        # Calculate scores
        forward_score = matched_forward / len(reference_peaks) if len(reference_peaks) > 0 else 0
        reverse_score = matched_reverse / len(detected_peaks) if len(detected_peaks) > 0 else 0
        exact_bonus = exact_matches / len(detected_peaks) if len(detected_peaks) > 0 else 0
        
        # Weighted combination
        combined_score = (forward_score * 0.5 + reverse_score * 0.4 + exact_bonus * 0.1)
        
        # Apply penalty for low coverage
        if forward_score < 0.3:
            combined_score *= 0.5
        
        return combined_score
    
    def log_status(self, message):
        """Log a status message."""
        # Handle case where UI isn't set up yet
        if hasattr(self, 'status_text') and self.status_text is not None:
            self.status_text.append(f"• {message}")
            self.status_text.ensureCursorVisible()
        else:
            print(f"• {message}")
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel - controls
        control_panel = self.create_control_panel()
        main_splitter.addWidget(control_panel)
        
        # Right panel - visualization
        viz_panel = self.create_visualization_panel()
        main_splitter.addWidget(viz_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 800])
        
        # Status and buttons
        button_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        button_layout.addWidget(self.progress_bar)
        
        self.run_analysis_btn = QPushButton("🔬 Run Complete Analysis")
        self.run_analysis_btn.clicked.connect(self.run_complete_analysis)
        self.run_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        button_layout.addWidget(self.run_analysis_btn)
        
        export_btn = QPushButton("📊 Export Results")
        export_btn.clicked.connect(self.export_results)
        button_layout.addWidget(export_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
    def create_control_panel(self):
        """Create the control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Preprocessing controls
        preprocess_group = QGroupBox("1. Preprocessing")
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        self.remove_bg_cb = QCheckBox("Remove Background")
        self.remove_bg_cb.setChecked(True)
        preprocess_layout.addWidget(self.remove_bg_cb)
        
        self.smooth_cb = QCheckBox("Smooth Spectrum")
        self.smooth_cb.setChecked(True)
        preprocess_layout.addWidget(self.smooth_cb)
        
        self.normalize_cb = QCheckBox("Normalize Intensity")
        self.normalize_cb.setChecked(True)
        preprocess_layout.addWidget(self.normalize_cb)
        
        layout.addWidget(preprocess_group)
        
        # Phase detection controls
        detection_group = QGroupBox("2. Phase Detection")
        detection_layout = QVBoxLayout(detection_group)
        
        detection_layout.addWidget(QLabel("Peak Prominence:"))
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.01, 1.0)
        self.prominence_spin.setValue(0.1)
        self.prominence_spin.setSingleStep(0.01)
        detection_layout.addWidget(self.prominence_spin)
        
        detection_layout.addWidget(QLabel("Min Phase Abundance:"))
        self.min_abundance_spin = QDoubleSpinBox()
        self.min_abundance_spin.setRange(0.01, 0.5)
        self.min_abundance_spin.setValue(0.05)
        self.min_abundance_spin.setSingleStep(0.01)
        detection_layout.addWidget(self.min_abundance_spin)
        
        layout.addWidget(detection_group)
        
        # Fitting controls
        fitting_group = QGroupBox("3. Fitting Parameters")
        fitting_layout = QVBoxLayout(fitting_group)
        
        fitting_layout.addWidget(QLabel("Peak Shape:"))
        self.peak_shape_combo = QComboBox()
        self.peak_shape_combo.addItems(["Pseudo-Voigt", "Gaussian", "Lorentzian"])
        fitting_layout.addWidget(self.peak_shape_combo)
        
        self.use_constraints_cb = QCheckBox("Use Physicochemical Constraints")
        self.use_constraints_cb.setChecked(True)
        fitting_layout.addWidget(self.use_constraints_cb)
        
        self.iterative_cb = QCheckBox("Iterative Refinement")
        self.iterative_cb.setChecked(True)
        fitting_layout.addWidget(self.iterative_cb)
        
        layout.addWidget(fitting_group)
        
        # Results table
        results_group = QGroupBox("4. Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Phase", "Formula", "Abundance (%)", "Confidence"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        # Status display
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        layout.addStretch()
        return panel
        
    def create_visualization_panel(self):
        """Create the visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Initialize plots
        self.ax_main = self.figure.add_subplot(2, 2, 1)
        self.ax_residual = self.figure.add_subplot(2, 2, 2)
        self.ax_components = self.figure.add_subplot(2, 2, 3)
        self.ax_fit_quality = self.figure.add_subplot(2, 2, 4)
        
        self.figure.tight_layout()
        
        # Initial plot
        self.plot_original_spectrum()
        
        return panel
        
    def plot_original_spectrum(self):
        """Plot the original spectrum."""
        self.ax_main.clear()
        self.ax_main.plot(self.wavenumbers, self.intensities, 'b-', label='Original')
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Original Spectrum')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        self.canvas.draw()
        
    def run_complete_analysis(self):
        """
        Run complete mixed mineral analysis using integrated RamanLab search system.
        
        Uses proven correlation-based search for major phases, DTW for residual analysis.
        """
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.run_analysis_btn.setEnabled(False)
            
            self.log_status("🔬 Starting integrated mixed mineral analysis...")
            self.log_status("=" * 60)
            
            # Step 1: Initialize integrated fitter
            self.log_status("Step 1: Initializing integrated fitter...")
            self.progress_bar.setValue(10)
            
            # Create new fitter with database integration
            from mixed_mineral_spectral_fitting import MixedMineralFitter
            
            # Pass our RamanLab database instance for proven correlation search
            integrated_fitter = MixedMineralFitter(
                self.wavenumbers, 
                self.intensities, 
                raman_db=self.raman_database  # Use RamanLab correlation-based database
            )
            
            # Preprocess spectrum
            processed = integrated_fitter.preprocess_spectrum(
                remove_background=self.remove_bg_cb.isChecked(),
                smooth_spectrum=self.smooth_cb.isChecked(),
                normalize=False  # Keep original intensities for better correlation
            )
            
            # Step 2: Detect major phase with proven correlation search
            self.log_status("Step 2: Detecting major phase with correlation-based search...")
            self.progress_bar.setValue(25)
            
            try:
                major_phase = integrated_fitter.detect_major_phase(
                    peak_prominence=self.prominence_spin.value(),
                    peak_height=0.02,
                    correlation_threshold=0.5,  # Reasonable correlation threshold
                    n_matches=15
                )
                
                self.log_status(f"✓ Major phase detected: {major_phase.name}")
                self.log_status(f"  Search method: {major_phase.search_method}")
                self.log_status(f"  Confidence: {major_phase.confidence:.3f}")
                self.log_status(f"  Formula: {major_phase.formula}")
                self.log_status(f"  Key peaks: {[f'{p:.1f}' for p in major_phase.expected_peaks[:5]]}")
                
                # Get the search results for confirmation dialog
                search_results = integrated_fitter.db_interface.search_correlation_based(
                    self.wavenumbers,
                    integrated_fitter.processed_intensities,
                    n_matches=15,
                    threshold=0.3
                )
                
                # Step 2.5: USER CONFIRMATION of major phase
                self.log_status("⏸ Waiting for user confirmation of major phase...")
                confirmation_dialog = MajorPhaseConfirmationDialog(self, major_phase, search_results)
                
                if confirmation_dialog.exec() == QDialog.Accepted:
                    if confirmation_dialog.confirmed_phase:
                        self.major_phase = confirmation_dialog.confirmed_phase
                        self.log_status(f"✅ User confirmed major phase: {self.major_phase.name}")
                        if self.major_phase != major_phase:
                            self.log_status(f"   (Changed from: {major_phase.name})")
                    else:
                        self.log_status("❌ No phase confirmed")
                        return
                else:
                    self.log_status("❌ Analysis cancelled by user")
                    return
                
                # Store for interface compatibility
                self.fitter = integrated_fitter  # Update to use integrated fitter
                
                if self.major_phase.confidence < 0.6:
                    self.log_status("⚠ Low confidence - proceeding with user confirmation")
                    
            except Exception as e:
                self.log_status(f"❌ Major phase detection failed: {str(e)}")
                # Fallback to old method
                self.log_status("Falling back to peak-based detection...")
                try:
                    # Create a simple peak-based phase
                    from scipy.signal import find_peaks
                    import numpy as np  # Import numpy here for scoping
                    
                    peaks, _ = find_peaks(
                        self.intensities,
                        prominence=self.prominence_spin.value(),
                        height=0.02,
                        distance=5
                    )
                    
                    if len(peaks) > 0:
                        peak_positions = self.wavenumbers[peaks]
                        peak_intensities = self.intensities[peaks]
                        
                        # Sort by intensity
                        sorted_indices = np.argsort(peak_intensities)[::-1]
                        peak_positions = peak_positions[sorted_indices]
                        peak_intensities = peak_intensities[sorted_indices]
                        
                        from mixed_mineral_spectral_fitting import PhaseInfo
                        self.major_phase = PhaseInfo(
                            name="Unknown_Major_Phase",
                            formula="Unknown",
                            expected_peaks=peak_positions[:10].tolist(),
                            peak_tolerances=[5.0] * min(10, len(peak_positions)),
                            peak_intensities=(peak_intensities[:10] / np.max(peak_intensities[:10])).tolist(),
                            constraints=[],
                            confidence=0.5,
                            abundance=0.8,
                            search_method='peak_detection',
                            database_metadata={}
                        )
                        
                        # Use the integrated fitter
                        self.fitter = integrated_fitter
                    else:
                        raise ValueError("No peaks detected")
                        
                except Exception as fallback_error:
                    self.log_status(f"❌ Fallback also failed: {str(fallback_error)}")
                    return
                
            # Update the fitter to use the confirmed major phase
            self.fitter.major_phase = self.major_phase
            
            # Step 3: Fit major phase
            self.log_status("Step 3: Fitting confirmed major phase...")
            self.progress_bar.setValue(40)
            
            fitting_method = self.get_fitting_method()
            major_fit = self.fitter.fit_major_phase(
                fitting_method=fitting_method,
                use_constraints=self.use_constraints_cb.isChecked()
            )
            
            self.log_status(f"✓ Major phase fit R²: {major_fit.r_squared:.4f}")
            
            # Update abundance estimate
            self.major_phase.abundance = self.fitter._estimate_phase_abundance(
                major_fit.individual_components[self.major_phase.name]
            )
            self.log_status(f"  Estimated abundance: {self.major_phase.abundance:.1%}")
            
            # Step 4: Analyze residual with weighted correction
            self.log_status("Step 4: Analyzing residual spectrum with overlap correction...")
            self.progress_bar.setValue(60)
            
            residual = self.fitter.analyze_residual_spectrum(
                major_fit,
                weighted_analysis=True,
                overlap_correction=True
            )
            
            import numpy as np  # Ensure numpy is available here
            residual_rms = np.sqrt(np.mean(residual**2))
            self.log_status(f"  Residual RMS: {residual_rms:.4f}")
            
            # Step 5: DTW search with user selection and Hey-Celestian classification
            self.log_status("Step 5: DTW search for minor phases with user selection...")
            self.progress_bar.setValue(75)
            
            try:
                # Perform DTW search
                dtw_results = self.fitter.db_interface.search_dtw_based(
                    self.wavenumbers,
                    residual,
                    n_matches=20,  # Get more candidates for user selection
                    window_size=15
                )
                
                if dtw_results:
                    self.log_status(f"✓ Found {len(dtw_results)} DTW matches")
                    self.log_status("⏸ Opening DTW analysis dialog for user selection...")
                    
                    # Open the new residual analysis dialog
                    residual_dialog = ResidualAnalysisDialog(
                        self, residual, self.wavenumbers, dtw_results, self.fitter
                    )
                    
                    if residual_dialog.exec() == QDialog.Accepted:
                        self.minor_phases = residual_dialog.selected_phases
                        
                        if self.minor_phases:
                            self.log_status(f"✅ User selected {len(self.minor_phases)} minor phases:")
                            for i, phase in enumerate(self.minor_phases):
                                self.log_status(f"  {i+1}. {phase.name} ({phase.search_method})")
                                self.log_status(f"     DTW Score: {phase.confidence:.3f}, Abundance: {phase.abundance:.1%}")
                                self.log_status(f"     Formula: {phase.formula}")
                        else:
                            self.log_status("  No minor phases selected by user")
                    else:
                        self.log_status("❌ DTW analysis cancelled by user")
                        self.minor_phases = []
                else:
                    self.log_status("  No DTW matches found")
                    self.minor_phases = []
                    
            except Exception as e:
                self.log_status(f"⚠ DTW-based search failed: {str(e)}")
                # Fallback to peak-based search
                self.log_status("Falling back to peak-based minor phase detection...")
                try:
                    from scipy.signal import find_peaks
                    import numpy as np
                    
                    residual_peaks, _ = find_peaks(
                        residual,
                        prominence=np.max(residual) * 0.1,
                        height=np.max(residual) * 0.05,
                        distance=2
                    )
                    
                    if len(residual_peaks) > 0:
                        residual_peak_positions = self.wavenumbers[residual_peaks]
                        minor_search_results = self.search_minerals_by_peaks(
                            residual_peak_positions, max_results=10
                        )
                        
                        self.minor_phases = []
                        for result in minor_search_results[:3]:
                            if result['score'] > 0.1:
                                abundance = self._estimate_minor_phase_abundance(residual, result)
                                if abundance >= self.min_abundance_spin.value():
                                    minor_phase = self._create_phase_from_mineral(result, abundance)
                                    self.minor_phases.append(minor_phase)
                    else:
                        self.minor_phases = []
                        
                except Exception as fallback_error:
                    self.log_status(f"Fallback also failed: {str(fallback_error)}")
                    self.minor_phases = []
            
            # Step 6: Global fitting
            if self.minor_phases:
                self.log_status("Step 6: Performing global fitting of all phases...")
                self.progress_bar.setValue(85)
                
                try:
                    self.global_fit_result = self.fitter.perform_global_fitting(
                        self.major_phase,
                        self.minor_phases,
                        fitting_method=fitting_method,
                        use_constraints=self.use_constraints_cb.isChecked(),
                        iterative_refinement=self.iterative_cb.isChecked()
                    )
                    
                    self.log_status(f"✓ Global fit R²: {self.global_fit_result.r_squared:.4f}")
                    self.log_status(f"  Reduced χ²: {self.global_fit_result.reduced_chi_squared:.4f}")
                    
                except Exception as e:
                    self.log_status(f"⚠ Global fitting failed: {str(e)}")
                    self.log_status("Using major phase fit only")
                    self.global_fit_result = major_fit
                    
            else:
                self.log_status("Step 6: Using major phase only (no minor phases)")
                self.global_fit_result = major_fit
            
            # Step 7: Quantification
            self.log_status("Step 7: Quantifying phases...")
            self.progress_bar.setValue(95)
            
            self.quantification_results = self.fitter.quantify_phases(self.global_fit_result)
            
            self.log_status("✓ Phase quantification results:")
            for phase_name, quant in self.quantification_results.items():
                self.log_status(f"  {phase_name}: {quant['abundance']:.1%} ± {quant['uncertainty']:.1%}")
                self.log_status(f"    Confidence: {quant['confidence']:.3f}")
            
            # Update displays
            self.update_results_table()
            self.update_plots()
                
            self.progress_bar.setValue(100)
            self.log_status("=" * 60)
            self.log_status("✅ Integrated mixed mineral analysis completed successfully!")
            
            QMessageBox.information(self, "Analysis Complete", 
                                  "Mixed mineral analysis completed successfully!\n"
                                  "Used correlation-based search for major phase and "
                                  "DTW analysis for minor phases.")
            
        except Exception as e:
            self.log_status(f"❌ Analysis failed: {str(e)}")
            import traceback
            self.log_status("Full traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self.log_status(f"  {line}")
            QMessageBox.critical(self, "Analysis Error", 
                               f"Analysis failed:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.run_analysis_btn.setEnabled(True)
            
    def get_fitting_method(self):
        """Get the selected fitting method."""
        method_map = {
            "Pseudo-Voigt": FittingMethod.PSEUDO_VOIGT,
            "Gaussian": FittingMethod.GAUSSIAN,
            "Lorentzian": FittingMethod.LORENTZIAN
        }
        return method_map[self.peak_shape_combo.currentText()]
        
    def update_results_table(self):
        """Update the results table with quantification data."""
        if not self.quantification_results:
            return
            
        self.results_table.setRowCount(len(self.quantification_results))
        
        for i, (phase_name, data) in enumerate(self.quantification_results.items()):
            # Find corresponding phase info
            phase_info = None
            if self.major_phase and self.major_phase.name == phase_name:
                phase_info = self.major_phase
            else:
                for minor_phase in self.minor_phases:
                    if minor_phase.name == phase_name:
                        phase_info = minor_phase
                        break
            
            formula = phase_info.formula if phase_info else "Unknown"
            
            self.results_table.setItem(i, 0, QTableWidgetItem(phase_name))
            self.results_table.setItem(i, 1, QTableWidgetItem(formula))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{data['abundance']:.1%} ± {data['uncertainty']:.1%}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{data['confidence']:.3f}"))
            
    def update_plots(self):
        """Update all plots with enhanced, clean visualizations."""
        if not self.global_fit_result:
            return
            
        import numpy as np
        
        # Enhanced color scheme
        main_colors = ['#2C3E50', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#34495E']
        
        # Main spectrum plot - Enhanced
        self.ax_main.clear()
        self.ax_main.plot(self.wavenumbers, self.intensities, color='#2C3E50', linewidth=1.5, 
                         label='Original', alpha=0.8)
        self.ax_main.plot(self.wavenumbers, self.global_fit_result.fitted_spectrum, 
                         color='#E74C3C', linewidth=2, label='Fitted')
        
        # Add R² to title
        r2 = self.global_fit_result.r_squared
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
        self.ax_main.set_ylabel('Intensity', fontsize=10)
        self.ax_main.set_title(f'Spectrum Fitting Results (R² = {r2:.4f})', fontweight='bold')
        self.ax_main.legend(fontsize=9)
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        self.ax_main.spines['top'].set_visible(False)
        self.ax_main.spines['right'].set_visible(False)
        
        # Residual plot - Enhanced with positive residual highlighting
        self.ax_residual.clear()
        residual = self.global_fit_result.residual
        self.ax_residual.plot(self.wavenumbers, residual, color='#27AE60', linewidth=1.5, 
                             label='Residual')
        
        # Highlight positive residual (potential minor phases)
        positive_residual = np.maximum(residual, 0)
        self.ax_residual.fill_between(self.wavenumbers, 0, positive_residual, 
                                     color='#E67E22', alpha=0.3, label='Positive Residual')
        
        self.ax_residual.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add RMS info
        residual_rms = np.sqrt(np.mean(residual**2))
        self.ax_residual.text(0.98, 0.95, f'RMS: {residual_rms:.4f}', 
                             transform=self.ax_residual.transAxes, fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                             horizontalalignment='right', verticalalignment='top')
        
        self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
        self.ax_residual.set_ylabel('Residual Intensity', fontsize=10)
        self.ax_residual.set_title('Fitting Residual', fontweight='bold')
        self.ax_residual.legend(fontsize=9)
        self.ax_residual.grid(True, alpha=0.3, linestyle='--')
        self.ax_residual.spines['top'].set_visible(False)
        self.ax_residual.spines['right'].set_visible(False)
        
        # Component spectra - Enhanced with abundances
        self.ax_components.clear()
        for i, (phase_name, component) in enumerate(self.global_fit_result.individual_components.items()):
            color = main_colors[i % len(main_colors)]
            
            # Truncate long names and add abundance if available
            label = phase_name[:15] + "..." if len(phase_name) > 15 else phase_name
            if hasattr(self, 'quantification_results') and phase_name in self.quantification_results:
                abundance = self.quantification_results[phase_name]['abundance']
                label += f" ({abundance:.1%})"
            
            self.ax_components.plot(self.wavenumbers, component, color=color, 
                                   linewidth=1.5, label=label, alpha=0.8)
        
        self.ax_components.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
        self.ax_components.set_ylabel('Intensity', fontsize=10)
        self.ax_components.set_title('Individual Phase Components', fontweight='bold')
        self.ax_components.legend(fontsize=8, loc='upper right')
        self.ax_components.grid(True, alpha=0.3, linestyle='--')
        self.ax_components.spines['top'].set_visible(False)
        self.ax_components.spines['right'].set_visible(False)
        
        # Phase abundances - Enhanced pie chart with confidence info
        self.ax_fit_quality.clear()
        if hasattr(self, 'quantification_results') and self.quantification_results:
            phases = list(self.quantification_results.keys())
            abundances = [data['abundance'] for data in self.quantification_results.values()]
            confidences = [data.get('confidence', 0.5) for data in self.quantification_results.values()]
            
            # Filter significant phases
            significant_data = []
            other_abundance = 0
            
            for phase, abundance, confidence in zip(phases, abundances, confidences):
                if abundance > 0.01:  # >1%
                    label = phase[:12] + "..." if len(phase) > 12 else phase
                    # Add confidence as subtitle
                    label += f"\n(conf: {confidence:.2f})"
                    significant_data.append((label, abundance, confidence))
                else:
                    other_abundance += abundance
            
            if other_abundance > 0.01:
                significant_data.append((f"Others\n({other_abundance:.1%})", other_abundance, 0.5))
            
            if significant_data:
                labels, values, confs = zip(*significant_data)
                
                # Color by abundance level
                colors = []
                for val in values:
                    if val > 0.5:
                        colors.append('#E74C3C')  # Red for major
                    elif val > 0.2:
                        colors.append('#F39C12')  # Orange for significant  
                    else:
                        colors.append('#3498DB')  # Blue for minor
                
                wedges, texts, autotexts = self.ax_fit_quality.pie(
                    values, labels=labels, autopct='%1.1f%%', startangle=90,
                    colors=colors, textprops={'fontsize': 8}
                )
                
                self.ax_fit_quality.set_title('Phase Abundances\n(with confidence)', fontweight='bold')
            else:
                self.ax_fit_quality.text(0.5, 0.5, 'No significant\nphases detected', 
                                        ha='center', va='center', fontsize=10)
                self.ax_fit_quality.set_title('Phase Abundances', fontweight='bold')
        else:
            self.ax_fit_quality.text(0.5, 0.5, 'Analysis in progress...', 
                                    ha='center', va='center', fontsize=10)
            self.ax_fit_quality.set_title('Phase Abundances', fontweight='bold')
        
        # Enhanced layout with title
        self.figure.suptitle('Mixed Mineral Analysis Results', fontsize=13, fontweight='bold', y=0.98)
        self.figure.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas.draw()
        
    def _create_phase_from_mineral(self, mineral_data, abundance=0.8):
        """Create PhaseInfo from selected mineral data."""
        try:
            # Extract Raman peaks with better error handling
            raman_peaks = mineral_data.get('key_peaks', [])
            if not raman_peaks and 'raman_peaks' in mineral_data:
                raman_peaks = [peak['frequency'] for peak in mineral_data['raman_peaks']]
            
            # Ensure we have peaks and they are valid numbers
            if not raman_peaks or not isinstance(raman_peaks, (list, tuple, np.ndarray)):
                raman_peaks = [400, 600, 800, 1000]  # Fallback generic peaks
            
            # Convert to list and filter valid peaks
            if isinstance(raman_peaks, np.ndarray):
                raman_peaks = raman_peaks.tolist()
            
            # Ensure all peaks are valid numbers
            valid_peaks = []
            for peak in raman_peaks:
                if isinstance(peak, (int, float)) and not np.isnan(peak) and 50 <= peak <= 4000:
                    valid_peaks.append(float(peak))
            
            if not valid_peaks:
                valid_peaks = [400, 600, 800, 1000]  # Fallback
            
            # Limit to reasonable number of peaks
            valid_peaks = valid_peaks[:10]
            
            return PhaseInfo(
                name=mineral_data.get('name', 'Unknown'),
                formula=mineral_data.get('formula', 'Unknown'),
                expected_peaks=valid_peaks,
                peak_tolerances=[5.0] * len(valid_peaks),
                peak_intensities=[1.0] * len(valid_peaks),
                constraints=[],
                confidence=mineral_data.get('score', 0.7),
                abundance=abundance
            )
        except Exception as e:
            self.log_status(f"Error creating phase from mineral data: {str(e)}")
            # Return a basic fallback phase
            return PhaseInfo(
                name='Unknown_Phase',
                formula='Unknown',
                expected_peaks=[400, 600, 800, 1000],
                peak_tolerances=[5.0, 5.0, 5.0, 5.0],
                peak_intensities=[1.0, 1.0, 1.0, 1.0],
                constraints=[],
                confidence=0.5,
                abundance=abundance
            )
    
    def _apply_als_refinement(self, fit_result):
        """Apply ALS (Asymmetric Least Squares) refinement to improve fitting."""
        try:
            import numpy as np  # Import numpy here for scope safety
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
            
            # Get current fit and residual
            fitted_spectrum = fit_result.fitted_spectrum
            original_intensities = self.fitter.processed_intensities
            
            # ALS baseline correction on residual
            residual = original_intensities - fitted_spectrum
            
            # Apply ALS to identify systematic baseline issues
            baseline_corrected = self._als_baseline_correction(residual)
            
            # Create improved spectrum
            improved_spectrum = fitted_spectrum + baseline_corrected
            
            # Calculate new residual and metrics
            new_residual = original_intensities - improved_spectrum
            
            # Calculate new R²
            ss_res = np.sum(new_residual ** 2)
            ss_tot = np.sum((original_intensities - np.mean(original_intensities)) ** 2)
            new_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Only return if improvement is significant
            if new_r_squared > fit_result.r_squared + 0.01:  # At least 1% improvement
                # Create new FitResult
                from mixed_mineral_spectral_fitting import FitResult
                
                return FitResult(
                    phases=fit_result.phases,
                    fitted_spectrum=improved_spectrum,
                    residual=new_residual,
                    individual_components=fit_result.individual_components,
                    r_squared=new_r_squared,
                    reduced_chi_squared=fit_result.reduced_chi_squared,
                    uncertainties=fit_result.uncertainties,
                    convergence_info={'refinement': 'ALS', 'original_r2': fit_result.r_squared}
                )
        
        except Exception as e:
            self.log_status(f"ALS refinement error: {str(e)}")
        
        return None
    
    def _als_baseline_correction(self, spectrum, lam=1e4, p=0.01, niter=10):
        """ALS baseline correction algorithm."""
        try:
            import numpy as np  # Import numpy here for scope safety
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
            
            # Ensure spectrum is a valid numpy array
            if not isinstance(spectrum, np.ndarray):
                spectrum = np.array(spectrum)
            
            L = len(spectrum)
            if L < 3:
                return np.zeros_like(spectrum)
                
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            w = np.ones(L)
            
            for i in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w * spectrum)
                
                # Fix array boolean context issue by using explicit array operations
                condition1 = spectrum > z
                condition2 = spectrum <= z
                w = p * condition1.astype(float) + (1-p) * condition2.astype(float)
            
            return z - np.mean(z)  # Return baseline-corrected component
            
        except Exception as e:
            import numpy as np  # Import numpy here too for the exception handler
            self.log_status(f"ALS baseline correction error: {str(e)}")
            return np.zeros_like(spectrum) if isinstance(spectrum, np.ndarray) else np.zeros(len(spectrum))
    
    def _estimate_minor_phase_abundance(self, residual, mineral_data):
        """Estimate abundance of a minor phase from residual spectrum."""
        try:
            import numpy as np  # Import numpy here for scope safety
            # Get expected peaks for this mineral
            key_peaks = mineral_data.get('key_peaks', [])
            if not key_peaks:
                return 0.05  # Default small abundance
            
            # Find residual intensity at expected peak positions
            total_residual_intensity = 0
            matched_peaks = 0
            
            for peak_freq in key_peaks[:5]:  # Check top 5 peaks
                # Find closest wavenumber index
                idx = np.argmin(np.abs(self.wavenumbers - peak_freq))
                if abs(self.wavenumbers[idx] - peak_freq) <= 10:  # Within 10 cm⁻¹
                    total_residual_intensity += max(0, residual[idx])
                    matched_peaks += 1
            
            if matched_peaks > 0:
                avg_residual_intensity = total_residual_intensity / matched_peaks
                # Normalize by total spectrum intensity
                total_spectrum_intensity = np.sum(self.fitter.processed_intensities)
                abundance = (avg_residual_intensity * matched_peaks) / total_spectrum_intensity
                return min(0.3, max(0.02, abundance))  # Clamp between 2% and 30%
            
            return 0.05  # Default
            
        except Exception as e:
            return 0.05
    
    def export_results(self):
        """Export analysis results to file."""
        if not self.global_fit_result or not self.quantification_results:
            QMessageBox.warning(self, "No Results", "No analysis results to export.")
            return
            
        try:
            # Generate comprehensive report
            report = self.fitter.generate_analysis_report(
                self.global_fit_result, 
                self.quantification_results
            )
            
            # Save to file
            from PySide6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Analysis Report", 
                "mixed_mineral_analysis_report.txt",
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(report)
                    
                QMessageBox.information(self, "Export Complete", 
                                      f"Analysis report exported to:\n{file_path}")
                                      
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export results:\n{str(e)}")


class ResidualAnalysisDialog(QDialog):
    """
    Dialog for DTW-based residual analysis with Hey-Celestian classification.
    
    This dialog allows users to:
    1. View DTW search results for minor phases
    2. See Hey-Celestian classification groups
    3. Select specific correlative modes
    4. Choose from chemical constraints
    """
    
    def __init__(self, parent, residual_spectrum, wavenumbers, dtw_results, fitter):
        super().__init__(parent)
        self.residual_spectrum = residual_spectrum
        self.wavenumbers = wavenumbers
        self.dtw_results = dtw_results
        self.fitter = fitter
        self.selected_phases = []
        
        self.setup_ui()
        self.populate_results()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("🔍 Residual Analysis - DTW Search & Chemical Classification")
        self.setModal(True)
        self.resize(1200, 800)
        
        main_layout = QHBoxLayout(self)
        
        # Left panel - DTW Results and Selection
        left_panel = self.create_results_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Visualization and Chemical Analysis
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 1)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        # Advanced options
        self.show_correlations_btn = QPushButton("📊 Show Chemical Correlations")
        self.show_correlations_btn.clicked.connect(self.show_chemical_correlations)
        button_layout.addWidget(self.show_correlations_btn)
        
        self.filter_by_group_btn = QPushButton("🧪 Filter by Hey-Celestian Group")
        self.filter_by_group_btn.clicked.connect(self.filter_by_classification_group)
        button_layout.addWidget(self.filter_by_group_btn)
        
        button_layout.addStretch()
        
        # Cancel/Accept
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        accept_btn = QPushButton("✅ Add Selected Phases")
        accept_btn.clicked.connect(self.accept_selection)
        accept_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        button_layout.addWidget(accept_btn)
        
        # Add button layout to main layout
        main_layout_with_buttons = QVBoxLayout()
        main_layout_with_buttons.addLayout(main_layout)
        main_layout_with_buttons.addLayout(button_layout)
        
        # Replace the main layout
        self.setLayout(main_layout_with_buttons)
        
    def create_results_panel(self):
        """Create the DTW results and selection panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("🔍 DTW Search Results for Residual Spectrum")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2C3E50; margin: 10px;")
        layout.addWidget(header)
        
        # Residual info
        import numpy as np
        residual_max = np.max(self.residual_spectrum)
        residual_rms = np.sqrt(np.mean(self.residual_spectrum**2))
        info_label = QLabel(f"Residual Max: {residual_max:.4f} | RMS: {residual_rms:.4f}")
        info_label.setStyleSheet("color: #666; margin: 5px;")
        layout.addWidget(info_label)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "✓", "Rank", "Mineral Name", "DTW Score", "Formula", "Hey-Celestian Group", "Key Peaks"
        ])
        
        # Set column widths
        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(True)  # Last column stretches
        self.results_table.setColumnWidth(0, 30)   # Checkbox
        self.results_table.setColumnWidth(1, 50)   # Rank
        self.results_table.setColumnWidth(2, 200)  # Name
        self.results_table.setColumnWidth(3, 80)   # Score
        self.results_table.setColumnWidth(4, 120)  # Formula
        self.results_table.setColumnWidth(5, 150)  # Hey-Celestian
        
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        layout.addWidget(self.results_table)
        
        # Selection controls
        selection_group = QGroupBox("Selection Controls")
        selection_layout = QVBoxLayout(selection_group)
        
        select_all_btn = QPushButton("Select All Top 5")
        select_all_btn.clicked.connect(self.select_top_phases)
        selection_layout.addWidget(select_all_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_selection)
        selection_layout.addWidget(clear_btn)
        
        # Abundance threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Min Abundance:"))
        self.abundance_threshold = QDoubleSpinBox()
        self.abundance_threshold.setRange(0.01, 0.5)
        self.abundance_threshold.setValue(0.05)
        self.abundance_threshold.setSingleStep(0.01)
        self.abundance_threshold.setSuffix("%")
        threshold_layout.addWidget(self.abundance_threshold)
        selection_layout.addLayout(threshold_layout)
        
        layout.addWidget(selection_group)
        
        return panel
        
    def create_visualization_panel(self):
        """Create the visualization and chemical analysis panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create figure for visualization
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        self.figure = Figure(figsize=(8, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Chemical analysis info
        self.chemical_info = QTextEdit()
        self.chemical_info.setMaximumHeight(200)
        self.chemical_info.setReadOnly(True)
        layout.addWidget(self.chemical_info)
        
        self.plot_residual_analysis()
        
        return panel
        
    def populate_results(self):
        """Populate the results table with DTW search results."""
        self.results_table.setRowCount(len(self.dtw_results))
        
        for i, result in enumerate(self.dtw_results):
            # Checkbox for selection
            checkbox = QCheckBox()
            self.results_table.setCellWidget(i, 0, checkbox)
            
            # Rank
            self.results_table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
            
            # Mineral name
            self.results_table.setItem(i, 2, QTableWidgetItem(result['name']))
            
            # DTW Score
            score_item = QTableWidgetItem(f"{result['score']:.4f}")
            if result['score'] > 0.7:
                score_item.setBackground(QColor(40, 167, 69, 50))  # Green
            elif result['score'] > 0.5:
                score_item.setBackground(QColor(255, 193, 7, 50))  # Yellow
            else:
                score_item.setBackground(QColor(220, 53, 69, 50))  # Red
            self.results_table.setItem(i, 3, score_item)
            
            # Formula
            formula = result.get('metadata', {}).get('formula', 'Unknown')
            self.results_table.setItem(i, 4, QTableWidgetItem(formula))
            
            # Hey-Celestian Classification
            hey_group = self.get_hey_celestian_group(result)
            self.results_table.setItem(i, 5, QTableWidgetItem(hey_group))
            
            # Key peaks
            peaks = result.get('peaks', [])
            if peaks:
                peaks_str = ', '.join(f'{p:.1f}' for p in peaks[:4])
                if len(peaks) > 4:
                    peaks_str += "..."
            else:
                peaks_str = "No peaks"
            self.results_table.setItem(i, 6, QTableWidgetItem(peaks_str))
        
        # Connect row selection to visualization update
        self.results_table.itemSelectionChanged.connect(self.update_chemical_analysis)
        
    def get_hey_celestian_group(self, result):
        """Get Hey-Celestian classification group for a result."""
        try:
            # Try to import and use the Hey-Celestian classifier
            from Hey_class.improved_hey_classification import ImprovedHeyClassifier
            
            classifier = ImprovedHeyClassifier()
            formula = result.get('metadata', {}).get('formula', '')
            elements_str = result.get('metadata', {}).get('elements', '')
            
            if formula:
                classification = classifier.classify_mineral(formula, elements_str)
                return classification.get('name', 'Unknown')[:25] + "..."
            else:
                return "Unknown"
                
        except ImportError:
            return "Classification N/A"
        except Exception:
            return "Classification Error"
    
    def plot_residual_analysis(self):
        """Plot residual spectrum analysis."""
        self.figure.clear()
        
        # Create subplots
        ax1 = self.figure.add_subplot(3, 1, 1)
        ax2 = self.figure.add_subplot(3, 1, 2)
        ax3 = self.figure.add_subplot(3, 1, 3)
        
        # Plot 1: Residual spectrum with peaks
        ax1.plot(self.wavenumbers, self.residual_spectrum, 'g-', linewidth=1, label='Residual')
        
        # Find and mark peaks
        from scipy.signal import find_peaks
        import numpy as np
        peaks, properties = find_peaks(
            self.residual_spectrum,
            prominence=np.max(self.residual_spectrum) * 0.1,
            height=np.max(self.residual_spectrum) * 0.05
        )
        
        if len(peaks) > 0:
            ax1.plot(self.wavenumbers[peaks], self.residual_spectrum[peaks], 'ro', 
                    markersize=6, label=f'{len(peaks)} Peaks')
        
        ax1.set_title('Residual Spectrum with Detected Peaks')
        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Residual Intensity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: DTW score distribution
        if self.dtw_results:
            scores = [r['score'] for r in self.dtw_results[:20]]  # Top 20
            names = [r['name'][:15] + "..." if len(r['name']) > 15 else r['name'] 
                    for r in self.dtw_results[:20]]
            
            bars = ax2.bar(range(len(scores)), scores, alpha=0.7)
            
            # Color bars by score
            for i, (bar, score) in enumerate(zip(bars, scores)):
                if score > 0.7:
                    bar.set_color('#28A745')  # Green
                elif score > 0.5:
                    bar.set_color('#FFC107')  # Yellow
                else:
                    bar.set_color('#DC3545')  # Red
            
            ax2.set_title('DTW Match Scores (Top 20)')
            ax2.set_xlabel('Match Rank')
            ax2.set_ylabel('DTW Score')
            ax2.set_xticks(range(0, len(scores), 2))
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Hey-Celestian group distribution
        if self.dtw_results:
            groups = []
            for result in self.dtw_results[:10]:  # Top 10
                group = self.get_hey_celestian_group(result)
                groups.append(group[:20] + "..." if len(group) > 20 else group)
            
            from collections import Counter
            group_counts = Counter(groups)
            
            if group_counts:
                labels, counts = zip(*group_counts.most_common(5))
                ax3.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Hey-Celestian Classification Distribution')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_chemical_analysis(self):
        """Update chemical analysis info when selection changes."""
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        if row >= len(self.dtw_results):
            return
            
        result = self.dtw_results[row]
        
        # Generate chemical analysis text
        analysis_text = f"""
<h3>🧪 Chemical Analysis: {result['name']}</h3>

<b>DTW Match Information:</b>
• DTW Score: {result['score']:.4f}
• Search Method: {result.get('search_method', 'dtw')}
• Match Confidence: {result.get('confidence', result['score']):.3f}

<b>Chemical Properties:</b>
• Formula: {result.get('metadata', {}).get('formula', 'Unknown')}
• Elements: {result.get('metadata', {}).get('elements', 'Unknown')}
• Crystal System: {result.get('metadata', {}).get('crystal_system', 'Unknown')}

<b>Hey-Celestian Classification:</b>
• Group: {self.get_hey_celestian_group(result)}
• Classification provides expected vibrational modes and chemical constraints

<b>Expected Raman Characteristics:</b>
"""
        
        # Add expected peaks and vibrational modes
        peaks = result.get('peaks', [])
        if peaks:
            analysis_text += f"• Key Peaks: {', '.join(f'{p:.1f}' for p in peaks[:8])} cm⁻¹\n"
        
        # Add Hey-Celestian vibrational mode predictions
        hey_group = self.get_hey_celestian_group(result)
        vibrational_info = self.get_vibrational_mode_info(hey_group, result)
        analysis_text += vibrational_info
        
        self.chemical_info.setHtml(analysis_text)
    
    def get_vibrational_mode_info(self, hey_group, result):
        """Get vibrational mode information based on Hey-Celestian classification."""
        formula = result.get('metadata', {}).get('formula', '')
        
        vibrational_info = "\n<b>Expected Vibrational Modes:</b>\n"
        
        # Basic chemical group analysis
        if 'carbonate' in hey_group.lower() or 'CO3' in formula:
            vibrational_info += "• Carbonate ν₁: ~1085 cm⁻¹ (strong)\n"
            vibrational_info += "• Carbonate ν₄: ~712 cm⁻¹ (medium)\n"
        elif 'sulfate' in hey_group.lower() or 'SO4' in formula:
            vibrational_info += "• Sulfate ν₁: ~1008 cm⁻¹ (strong)\n"
            vibrational_info += "• Sulfate ν₃: ~1100-1200 cm⁻¹ (medium)\n"
        elif 'phosphate' in hey_group.lower() or 'PO4' in formula:
            vibrational_info += "• Phosphate ν₁: ~960 cm⁻¹ (strong)\n"
            vibrational_info += "• Phosphate ν₃: ~1000-1100 cm⁻¹ (medium)\n"
        elif 'silicate' in hey_group.lower() or 'Si' in formula:
            vibrational_info += "• Si-O stretching: 800-1200 cm⁻¹\n"
            vibrational_info += "• Si-O-Si bending: 400-600 cm⁻¹\n"
        elif 'oxide' in hey_group.lower():
            vibrational_info += "• Metal-O stretching: 200-800 cm⁻¹\n"
            vibrational_info += "• Lattice modes: 100-400 cm⁻¹\n"
        else:
            vibrational_info += "• Mode assignment requires detailed analysis\n"
        
        return vibrational_info
    
    def show_chemical_correlations(self):
        """Show chemical correlation analysis dialog."""
        selected_rows = self.get_selected_results()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select at least one phase to analyze correlations.")
            return
        
        correlation_dialog = ChemicalCorrelationDialog(self, selected_rows, self.residual_spectrum, self.wavenumbers)
        correlation_dialog.exec()
    
    def filter_by_classification_group(self):
        """Filter results by Hey-Celestian classification group."""
        groups = set()
        for result in self.dtw_results:
            group = self.get_hey_celestian_group(result)
            groups.add(group)
        
        group_list = sorted(list(groups))
        
        # Show selection dialog
        selected_group, ok = QInputDialog.getItem(
            self, "Filter by Classification Group", 
            "Select Hey-Celestian group to filter by:", 
            group_list, 0, False
        )
        
        if ok and selected_group:
            self.filter_results_by_group(selected_group)
    
    def filter_results_by_group(self, selected_group):
        """Filter the results table by classification group."""
        for row in range(self.results_table.rowCount()):
            group_item = self.results_table.item(row, 5)  # Hey-Celestian group column
            if group_item:
                show_row = selected_group in group_item.text()
                self.results_table.setRowHidden(row, not show_row)
    
    def select_top_phases(self):
        """Select top 5 phases automatically."""
        for row in range(min(5, self.results_table.rowCount())):
            checkbox = self.results_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(True)
    
    def clear_selection(self):
        """Clear all phase selections."""
        for row in range(self.results_table.rowCount()):
            checkbox = self.results_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(False)
    
    def get_selected_results(self):
        """Get list of selected results."""
        selected = []
        for row in range(self.results_table.rowCount()):
            checkbox = self.results_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                if row < len(self.dtw_results):
                    selected.append(self.dtw_results[row])
        return selected
    
    def accept_selection(self):
        """Accept the selected phases."""
        selected = self.get_selected_results()
        
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select at least one phase to add.")
            return
        
        # Convert selected results to PhaseInfo objects
        from mixed_mineral_spectral_fitting import PhaseInfo
        import numpy as np
        
        self.selected_phases = []
        for result in selected:
            # Estimate abundance based on residual
            residual_max = np.max(self.residual_spectrum)
            estimated_abundance = min(0.3, max(0.05, residual_max * 0.5))  # Rough estimate
            
            # Check against threshold
            if estimated_abundance >= self.abundance_threshold.value() / 100:
                phase = PhaseInfo(
                    name=result['name'],
                    formula=result.get('metadata', {}).get('formula', 'Unknown'),
                    expected_peaks=result.get('peaks', [])[:10],
                    peak_tolerances=[3.0] * min(10, len(result.get('peaks', []))),
                    peak_intensities=[1.0] * min(10, len(result.get('peaks', []))),
                    constraints=[],
                    confidence=result['score'],
                    abundance=estimated_abundance,
                    search_method='dtw_user_selected',
                    database_metadata=result.get('metadata', {})
                )
                self.selected_phases.append(phase)
        
        if not self.selected_phases:
            QMessageBox.warning(self, "Threshold Filter", 
                              f"No phases meet the minimum abundance threshold of {self.abundance_threshold.value():.1f}%.")
            return
        
        self.accept()


class ChemicalCorrelationDialog(QDialog):
    """Dialog for showing chemical correlations and correlative mode selection."""
    
    def __init__(self, parent, selected_results, residual_spectrum, wavenumbers):
        super().__init__(parent)
        self.selected_results = selected_results
        self.residual_spectrum = residual_spectrum
        self.wavenumbers = wavenumbers
        
        self.setup_ui()
        self.analyze_correlations()
        
    def setup_ui(self):
        """Set up the correlation analysis UI."""
        self.setWindowTitle("🧪 Chemical Correlation Analysis")
        self.setModal(True)
        self.resize(1000, 700)
        
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("🧪 Chemical Correlation & Correlative Mode Analysis")
        header.setStyleSheet("font-weight: bold; font-size: 16px; color: #2C3E50; margin: 10px;")
        layout.addWidget(header)
        
        # Correlation results
        self.correlation_table = QTableWidget()
        self.correlation_table.setColumnCount(5)
        self.correlation_table.setHorizontalHeaderLabels([
            "Phase", "Chemical Group", "Correlative Modes", "Peak Match", "Confidence"
        ])
        layout.addWidget(self.correlation_table)
        
        # Mode selection
        mode_group = QGroupBox("Select Specific Correlative Modes")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_list = QListWidget()
        self.mode_list.setSelectionMode(QListWidget.MultiSelection)
        mode_layout.addWidget(self.mode_list)
        
        layout.addWidget(mode_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("✅ Apply Selected Modes")
        apply_btn.clicked.connect(self.apply_selected_modes)
        button_layout.addWidget(apply_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def analyze_correlations(self):
        """Analyze chemical correlations for selected phases."""
        self.correlation_table.setRowCount(len(self.selected_results))
        all_modes = []
        
        for i, result in enumerate(self.selected_results):
            # Phase name
            self.correlation_table.setItem(i, 0, QTableWidgetItem(result['name']))
            
            # Chemical group (simplified)
            formula = result.get('metadata', {}).get('formula', '')
            chemical_group = self.determine_chemical_group(formula)
            self.correlation_table.setItem(i, 1, QTableWidgetItem(chemical_group))
            
            # Correlative modes
            modes = self.get_correlative_modes(formula, chemical_group)
            modes_str = ', '.join(modes)
            self.correlation_table.setItem(i, 2, QTableWidgetItem(modes_str))
            all_modes.extend(modes)
            
            # Peak match assessment
            peaks = result.get('peaks', [])
            match_quality = self.assess_peak_match(peaks)
            self.correlation_table.setItem(i, 3, QTableWidgetItem(match_quality))
            
            # Confidence
            confidence = result.get('score', 0.5)
            conf_item = QTableWidgetItem(f"{confidence:.3f}")
            if confidence > 0.7:
                conf_item.setBackground(QColor(40, 167, 69, 50))
            elif confidence > 0.5:
                conf_item.setBackground(QColor(255, 193, 7, 50))
            else:
                conf_item.setBackground(QColor(220, 53, 69, 50))
            self.correlation_table.setItem(i, 4, conf_item)
        
        # Populate mode selection list
        unique_modes = sorted(set(all_modes))
        for mode in unique_modes:
            self.mode_list.addItem(mode)
    
    def determine_chemical_group(self, formula):
        """Determine chemical group from formula."""
        if 'CO3' in formula or 'CO_3' in formula:
            return "Carbonate"
        elif 'SO4' in formula or 'SO_4' in formula:
            return "Sulfate"
        elif 'PO4' in formula or 'PO_4' in formula:
            return "Phosphate"
        elif 'SiO' in formula:
            return "Silicate"
        elif any(ox in formula for ox in ['O2', 'O_2', 'O3', 'O_3']):
            return "Oxide"
        else:
            return "Other"
    
    def get_correlative_modes(self, formula, chemical_group):
        """Get correlative modes for chemical group."""
        if chemical_group == "Carbonate":
            return ["ν₁(CO₃)", "ν₄(CO₃)", "lattice"]
        elif chemical_group == "Sulfate":
            return ["ν₁(SO₄)", "ν₃(SO₄)", "ν₄(SO₄)"]
        elif chemical_group == "Phosphate":
            return ["ν₁(PO₄)", "ν₃(PO₄)", "ν₄(PO₄)"]
        elif chemical_group == "Silicate":
            return ["Si-O stretch", "Si-O-Si bend", "lattice"]
        elif chemical_group == "Oxide":
            return ["M-O stretch", "lattice", "breathing"]
        else:
            return ["unknown"]
    
    def assess_peak_match(self, peaks):
        """Assess how well peaks match the residual."""
        if not peaks:
            return "No peaks"
        
        # Simple assessment based on number of peaks in residual range
        residual_peaks = len([p for p in peaks if 200 <= p <= 1200])
        
        if residual_peaks >= 3:
            return "Good match"
        elif residual_peaks >= 2:
            return "Fair match"
        else:
            return "Weak match"
    
    def apply_selected_modes(self):
        """Apply selected correlative modes for enhanced analysis."""
        selected_modes = [item.text() for item in self.mode_list.selectedItems()]
        
        if not selected_modes:
            QMessageBox.information(self, "No Selection", "No correlative modes selected.")
            return
        
        QMessageBox.information(
            self, "Modes Applied", 
            f"Selected correlative modes will be emphasized in fitting:\n\n" +
            "\n".join(f"• {mode}" for mode in selected_modes)
        )
        
        self.close()


def launch_mixed_mineral_analysis(parent, wavenumbers, intensities):
    """Launch the mixed mineral analysis dialog."""
    try:
        dialog = MixedMineralAnalysisQt6(parent, wavenumbers, intensities)
        dialog.exec()
    except Exception as e:
        print(f"Error launching mixed mineral analysis: {str(e)}")
        print("Please check the console for detailed error information.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test the interface with synthetic data
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Generate test data
    wavenumbers = np.linspace(100, 1200, 1100)
    intensities = np.random.normal(0.1, 0.02, len(wavenumbers))
    
    # Add some synthetic peaks
    for peak_pos in [206, 465, 696, 1085]:  # Quartz-like
        intensities += 0.8 * np.exp(-(wavenumbers - peak_pos)**2 / (2 * 8**2))
    
    for peak_pos in [288, 508, 760]:  # Feldspar-like
        intensities += 0.3 * np.exp(-(wavenumbers - peak_pos)**2 / (2 * 6**2))
    
    dialog = MixedMineralAnalysisQt6(None, wavenumbers, intensities)
    dialog.show()
    
    sys.exit(app.exec()) 