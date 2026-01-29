"""
Microplastic Detection Tab for 2D Map Analysis

Provides UI for detecting microplastics in Raman spectroscopy maps.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QCheckBox, QLabel, QSlider, 
                              QProgressBar, QTextEdit, QComboBox, QDoubleSpinBox,
                              QGridLayout, QFrame, QSpinBox)
from typing import Dict, Optional
from PySide6.QtCore import Qt, Signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

# Import custom toolbar from RamanLab core
try:
    import sys
    from pathlib import Path
    # Add parent directory to path to access core module
    core_path = Path(__file__).parent.parent.parent.parent / 'core'
    if str(core_path) not in sys.path:
        sys.path.insert(0, str(core_path))
    from matplotlib_config import CompactNavigationToolbar as NavigationToolbar
except ImportError:
    # Fallback to standard toolbar
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class MicroplasticDetectionTab(QWidget):
    """Tab for microplastic detection in 2D Raman maps."""
    
    # Signals
    detection_started = Signal()
    detection_stopped = Signal()
    detection_completed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.detection_results = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # ===== Visualization Area =====
        viz_panel = self.create_visualization_panel()
        layout.addWidget(viz_panel, stretch=1)
        
        # ===== Status and Results =====
        status_panel = self.create_status_panel()
        layout.addWidget(status_panel)
        
    def create_control_panel(self):
        """Create the control panel with detection settings."""
        from PySide6.QtWidgets import QScrollArea
        
        # Main panel container
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # === Scrollable Controls Section ===
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMaximumHeight(400)  # Limit height to make room for spectrum
        
        # Controls widget inside scroll area
        controls_widget = QWidget()
        layout = QVBoxLayout(controls_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # === Database Info (read-only, shows status) ===
        self.db_info_label = QLabel()
        self.update_database_status()
        layout.addWidget(self.db_info_label)
        
        # === Plastic Type Selection ===
        plastic_group = QGroupBox("Plastic Types to Detect")
        plastic_layout = QGridLayout(plastic_group)
        
        self.plastic_checkboxes = {}
        plastic_types = [
            ('PE', 'Polyethylene', 'Bags, bottles, films'),
            ('PP', 'Polypropylene', 'Containers, caps, fibers'),
            ('PS', 'Polystyrene', 'Foam, packaging'),
            ('PET', 'PET', 'Bottles, polyester fibers'),
            ('PVC', 'PVC', 'Pipes, films, coatings'),
            ('PMMA', 'Acrylic (PMMA)', 'Sheets, beads, paints'),
            ('PA', 'Nylon (PA)', 'Fibers, textiles')
        ]
        
        for idx, (code, name, desc) in enumerate(plastic_types):
            row = idx // 2
            col = (idx % 2) * 2
            
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)  # All enabled by default
            checkbox.setToolTip(f"{name}: {desc}")
            self.plastic_checkboxes[code] = checkbox
            
            plastic_layout.addWidget(checkbox, row, col, 1, 2)
        
        # Select/Deselect all buttons + Conservative Mode
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_plastics)
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all_plastics)
        
        # Conservative Mode button
        conservative_btn = QPushButton("🎯 Conservative Mode")
        conservative_btn.setToolTip("Select only PE, PP, PS (most common) and use high threshold")
        conservative_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        conservative_btn.clicked.connect(self.set_conservative_mode)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addWidget(conservative_btn)
        button_layout.addStretch()
        
        plastic_layout.addLayout(button_layout, (len(plastic_types) + 1) // 2, 0, 1, 4)
        layout.addWidget(plastic_group)
        
        # === Detection Parameters ===
        params_group = QGroupBox("Detection Parameters")
        params_group.setMaximumWidth(400)  # Limit width
        params_layout = QGridLayout(params_group)
        
        # Detection threshold
        params_layout.addWidget(QLabel("Detection Threshold:"), 0, 0)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(30)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        params_layout.addWidget(self.threshold_slider, 0, 1)
        
        self.threshold_label = QLabel("0.30")
        self.threshold_label.setMinimumWidth(50)
        params_layout.addWidget(self.threshold_label, 0, 2)
        
        # Baseline Correction Presets
        params_layout.addWidget(QLabel("Baseline Correction:"), 1, 0)
        self.baseline_preset_combo = QComboBox()
        self.baseline_preset_combo.addItems([
            "Rolling Ball (Fast) - Recommended",
            "ALS (Fast)",
            "ALS (Moderate)",
            "ALS (Aggressive)",
            "ALS (Balanced)",
            "ALS (Conservative)",
            "ALS (Ultra Smooth)"
        ])
        self.baseline_preset_combo.setCurrentText("Rolling Ball (Fast) - Recommended")  # Default
        self.baseline_preset_combo.setToolTip(
            "Baseline correction methods:\n"
            "• Rolling Ball (Fast): Very fast, good for fluorescence (~30 sec for 82k spectra)\n"
            "• ALS (Fast): λ=1e5, p=0.01, 5 iter - Quick ALS (~2-5 min)\n"
            "• ALS (Moderate): λ=1e5, p=0.01, 10 iter - Standard (~4-8 min)\n"
            "• ALS (Aggressive): λ=1e4, p=0.05, 15 iter - Strong removal (~6-12 min)\n"
            "• ALS (Balanced): λ=5e5, p=0.02, 12 iter - Middle ground (~5-10 min)\n"
            "• ALS (Conservative): λ=1e6, p=0.001, 10 iter - Gentle (~4-8 min)\n"
            "• ALS (Ultra Smooth): λ=1e7, p=0.002, 20 iter - Very smooth (~8-15 min)"
        )
        params_layout.addWidget(self.baseline_preset_combo, 1, 1)
        
        # Test Baseline button
        self.test_baseline_btn = QPushButton("🔬 Test")
        self.test_baseline_btn.setToolTip("Test baseline correction on selected spectrum")
        self.test_baseline_btn.clicked.connect(self.open_baseline_tester)
        self.test_baseline_btn.setMaximumWidth(80)
        params_layout.addWidget(self.test_baseline_btn, 1, 2)
        
        # Peak enhancement
        params_layout.addWidget(QLabel("Peak Enhancement:"), 2, 0)
        self.enhancement_spin = QSpinBox()
        self.enhancement_spin.setRange(5, 21)
        self.enhancement_spin.setValue(11)
        self.enhancement_spin.setSingleStep(2)
        self.enhancement_spin.setToolTip("Smoothing window size (odd numbers only)")
        params_layout.addWidget(self.enhancement_spin, 2, 1, 1, 2)
        
        # Detection method
        params_layout.addWidget(QLabel("Detection Method:"), 3, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            'Peak-based (Fast)',
            'Database Correlation (Accurate)',
            'Hybrid (Recommended)',
            'Template Matching (Best for Noisy Data)'
        ])
        self.method_combo.setCurrentIndex(3)  # Default to Template Matching
        self.method_combo.setToolTip(
            "Peak-based: Fast, uses characteristic peaks\n"
            "Database: Slower, matches against reference spectra\n"
            "Hybrid: Uses both methods\n"
            "Template Matching: Uses curated plastic spectra from database (best for noisy data)"
        )
        params_layout.addWidget(self.method_combo, 3, 1, 1, 2)
        
        layout.addWidget(params_group)
        
        # === Action Buttons ===
        # Row 1: Start and Stop buttons
        button_row1 = QHBoxLayout()
        
        self.scan_btn = QPushButton("🔍 Scan Map for Microplastics")
        self.scan_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        self.scan_btn.clicked.connect(self.start_detection)
        button_row1.addWidget(self.scan_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)
        button_row1.addWidget(self.stop_btn)
        
        layout.addLayout(button_row1)
        
        # Row 2: Export and Stats buttons
        button_row2 = QHBoxLayout()
        
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        button_row2.addWidget(self.export_btn)
        
        self.stats_btn = QPushButton("📊 Show Statistics")
        self.stats_btn.setEnabled(False)
        self.stats_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.stats_btn.clicked.connect(self.show_statistics)
        button_row2.addWidget(self.stats_btn)
        
        layout.addLayout(button_row2)
        
        # Row 3: Refine Results button
        button_row3 = QHBoxLayout()
        
        self.refine_btn = QPushButton("🎯 Refine Results (Spatial Filter)")
        self.refine_btn.setEnabled(False)
        self.refine_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.refine_btn.clicked.connect(self.refine_results)
        button_row3.addWidget(self.refine_btn)
        
        layout.addLayout(button_row3)
        
        # Set the controls widget in the scroll area
        scroll_area.setWidget(controls_widget)
        main_layout.addWidget(scroll_area)
        
        # === Spectrum Display (below scroll area, always visible) ===
        spectrum_group = QGroupBox("Selected Spectrum (Click map to view)")
        spectrum_layout = QVBoxLayout(spectrum_group)
        spectrum_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create matplotlib figure for spectrum with proper DPI
        self.spectrum_figure = Figure(figsize=(5, 3.5), dpi=100)
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        self.spectrum_canvas.setMinimumHeight(250)
        self.spectrum_canvas.setMaximumHeight(350)
        
        # Add navigation toolbar for spectrum
        spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, self)
        
        spectrum_layout.addWidget(spectrum_toolbar)
        spectrum_layout.addWidget(self.spectrum_canvas)
        main_layout.addWidget(spectrum_group)
        
        # Create initial empty spectrum plot
        ax = self.spectrum_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Click on detection map\nto view spectrum',
               ha='center', va='center', fontsize=10, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        self.spectrum_figure.tight_layout()
        self.spectrum_canvas.draw_idle()
        
        return panel
    
    def create_visualization_panel(self):
        """Create the visualization panel with matplotlib figures."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # === Detection Maps ===
        maps_group = QGroupBox("Detection Results - Spatial Maps (Click to view spectrum)")
        maps_layout = QVBoxLayout(maps_group)
        
        # Create matplotlib figure for detection maps
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Add navigation toolbar for maps
        toolbar = NavigationToolbar(self.canvas, self)
        
        maps_layout.addWidget(toolbar)
        maps_layout.addWidget(self.canvas)
        layout.addWidget(maps_group)
        
        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        
        # Store for later use
        self.selected_position = None
        
        # Create initial empty plots
        self.create_empty_plots()
        
        return panel
    
    def create_empty_plots(self):
        """Create empty placeholder plots."""
        self.figure.clear()
        
        # Create 2x4 grid for up to 7 plastic types + composite
        self.axes = []
        for i in range(8):
            ax = self.figure.add_subplot(2, 4, i + 1)
            ax.text(0.5, 0.5, 'No data\n\nRun detection to see results',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            self.axes.append(ax)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def create_status_panel(self):
        """Create status and progress panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        self.status_text.setPlaceholderText("Status messages will appear here...")
        layout.addWidget(self.status_text)
        
        return panel
    
    def update_database_status(self):
        """Update the database status label."""
        # Check if parent window has database loaded
        # The database is loaded from RamanLab_Database_20250602.pkl and filtered for plastics
        if hasattr(self.parent_window, 'database') and self.parent_window.database:
            n_entries = len(self.parent_window.database)
            self.db_info_label.setText(f"📚 Database: {n_entries:,} plastic spectra loaded")
            self.db_info_label.setStyleSheet("""
                QLabel {
                    background-color: #e8f5e9;
                    color: #2e7d32;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
            self.log_status(f"✓ Using {n_entries} plastic reference spectra from database")
        else:
            self.db_info_label.setText("⚠️ No plastic spectra in database\n(Use File → Load Database to load plastic references)")
            self.db_info_label.setStyleSheet("""
                QLabel {
                    background-color: #fff3e0;
                    color: #e65100;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
            self.log_status("ℹ️ No plastic database loaded - will use peak-based detection only")
    
    def select_all_plastics(self):
        """Select all plastic types."""
        for checkbox in self.plastic_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_plastics(self):
        """Deselect all plastic types."""
        for checkbox in self.plastic_checkboxes.values():
            checkbox.setChecked(False)
    
    def set_conservative_mode(self):
        """Set conservative detection mode: PE, PP, PS only with high threshold."""
        # Deselect all first
        for code, checkbox in self.plastic_checkboxes.items():
            checkbox.setChecked(code in ['PE', 'PP', 'PS'])
        
        # Set high threshold (60%)
        self.threshold_slider.setValue(60)
        
        # Log the change
        self.log_status("🎯 Conservative Mode enabled:")
        self.log_status("  • Detecting only PE, PP, PS (most common microplastics)")
        self.log_status("  • Threshold set to 0.60 (reduces false positives)")
        self.log_status("  • Recommended for environmental samples")
    
    def get_selected_plastics(self):
        """Get list of selected plastic types."""
        return [code for code, checkbox in self.plastic_checkboxes.items() 
                if checkbox.isChecked()]
    
    def update_threshold_label(self):
        """Update threshold label when slider changes."""
        value = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"{value:.2f}")
    
    def get_detection_parameters(self):
        """Get current detection parameters."""
        # Baseline correction presets
        baseline_presets = {
            "Rolling Ball (Fast) - Recommended": {"method": "rolling_ball", "window": 100},
            "ALS (Fast)": {"method": "als", "lam": 1e5, "p": 0.01, "niter": 5},
            "ALS (Moderate)": {"method": "als", "lam": 1e5, "p": 0.01, "niter": 10},
            "ALS (Aggressive)": {"method": "als", "lam": 1e4, "p": 0.05, "niter": 15},
            "ALS (Balanced)": {"method": "als", "lam": 5e5, "p": 0.02, "niter": 12},
            "ALS (Conservative)": {"method": "als", "lam": 1e6, "p": 0.001, "niter": 10},
            "ALS (Ultra Smooth)": {"method": "als", "lam": 1e7, "p": 0.002, "niter": 20}
        }
        
        selected_preset = self.baseline_preset_combo.currentText()
        baseline_params = baseline_presets.get(selected_preset, baseline_presets["Rolling Ball (Fast) - Recommended"])
        
        return {
            'plastic_types': self.get_selected_plastics(),
            'threshold': self.threshold_slider.value() / 100.0,
            'baseline': baseline_params,
            'enhancement_window': self.enhancement_spin.value(),
            'method': self.method_combo.currentText()
        }
    
    def start_detection(self):
        """Start microplastic detection."""
        # Update database status before starting
        self.update_database_status()
        
        # Enable stop button, disable start button
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.stats_btn.setEnabled(False)
        self.refine_btn.setEnabled(False)
        
        # This will be connected to the parent window's detection method
        if hasattr(self.parent_window, 'run_microplastic_detection'):
            self.parent_window.run_microplastic_detection()
    
    def stop_detection(self):
        """Stop ongoing detection."""
        self.log_status("⏹ Stopping detection...")
        self.stop_btn.setEnabled(False)  # Disable stop button immediately
        # Emit signal to parent to stop the worker thread
        self.detection_stopped.emit()
    
    def update_progress(self, value, message=""):
        """Update progress bar and status."""
        self.progress_bar.setValue(value)
        if message:
            self.log_status(message)
    
    def log_status(self, message):
        """Add message to status log."""
        self.status_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def reset_buttons(self):
        """Reset button states after detection completes or fails."""
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def display_results(self, results: Dict[str, np.ndarray], map_shape: tuple = None):
        """Display detection results in the visualization panel.
        
        Args:
            results: Dictionary of plastic type -> score map arrays
            map_shape: Tuple of (n_rows, n_cols) for reshaping 1D arrays into 2D maps
        """
        from map_analysis_2d.analysis.microplastic_detector import MicroplasticDetector
        
        # Store full results (including metadata like _match_details)
        self.detection_results = results
        self.log_status(f"✓ Stored detection results with {len(results)} entries")
        
        # Filter out metadata entries (starting with _) for visualization
        score_maps = {k: v for k, v in results.items() if not k.startswith('_')}
        
        # Clear previous plots
        self.figure.clear()
        
        # Get threshold from UI
        threshold = self.threshold_slider.value() / 100.0
        
        # Create detector instance for plastic info
        detector = MicroplasticDetector()
        
        # Create subplots for each plastic type
        n_plastics = len(score_maps)
        if n_plastics == 0:
            return
        
        # Calculate grid layout
        cols = min(3, n_plastics)
        rows = (n_plastics + cols - 1) // cols
        
        for i, (plastic_type, score_map) in enumerate(score_maps.items()):
            # Create subplot
            ax = self.figure.add_subplot(rows, cols, i + 1)
            
            # Get plastic info
            plastic_info = detector.PLASTIC_SIGNATURES.get(plastic_type, {})
            plastic_name = plastic_info.get('name', plastic_type)
            plastic_color = plastic_info.get('color', '#FF6B6B')
            
            # Reshape if needed using actual map dimensions
            if score_map.ndim == 1:
                if map_shape is not None:
                    # Use provided map dimensions
                    score_map = score_map.reshape(map_shape)
                else:
                    # Fallback to square assumption
                    size = int(np.sqrt(len(score_map)))
                    if size * size == len(score_map):
                        score_map = score_map.reshape(size, size)
                    else:
                        self.log_status(f"⚠️ Warning: Cannot reshape {plastic_type} map (size={len(score_map)})")
                        continue
            
            # Mask out scores below threshold for clearer visualization
            masked_map = np.copy(score_map)
            masked_map[masked_map < threshold] = 0
            
            # Plot heatmap
            im = ax.imshow(masked_map, cmap='hot', interpolation='nearest',
                          vmin=0, vmax=1, aspect='auto')
            ax.set_title(f'{plastic_name}\n({np.sum(score_map > threshold)} detections)',
                        fontweight='bold')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Detection Score', rotation=270, labelpad=15)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Enable export, statistics, and refine buttons
        self.export_btn.setEnabled(True)
        self.stats_btn.setEnabled(True)
        self.refine_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Log summary
        total_detections = sum(np.sum(score_map > threshold) for score_map in score_maps.values())
        self.log_status(f"✅ Detection complete! Found {total_detections} potential microplastic locations (threshold: {threshold:.2f})")
    
    def show_statistics(self):
        """Show comprehensive statistics dialog."""
        if self.detection_results is None:
            self.log_status("⚠️ No detection results available")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Results", "Please run a detection first before viewing statistics.")
            return
        
        self.log_status("📊 Opening statistics dialog...")
        
        try:
            from map_analysis_2d.ui.dialogs.statistics_dialog import MicroplasticStatisticsDialog
            from map_analysis_2d.analysis.microplastic_detector import MicroplasticDetector
            
            self.log_status("✓ Imported statistics dialog")
            
            # Create detector instance
            detector = MicroplasticDetector()
            
            # Load references if available
            if hasattr(self.parent_window, 'database') and self.parent_window.database:
                # Database is already filtered for plastics
                detector.load_plastic_references(self.parent_window.database)
                self.log_status(f"✓ Loaded {len(self.parent_window.database)} references")
            
            # Show dialog
            self.log_status("Creating statistics dialog...")
            dialog = MicroplasticStatisticsDialog(
                self.detection_results,
                self.parent_window.map_data if hasattr(self.parent_window, 'map_data') else None,
                detector,
                self
            )
            self.log_status("✓ Dialog created, showing...")
            dialog.exec()
            self.log_status("✓ Statistics dialog closed")
            
        except ImportError as e:
            self.log_status(f"❌ Import error: {str(e)}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Import Error", 
                               f"Could not import statistics dialog:\n{str(e)}\n\nMake sure statistics_dialog.py exists.")
            import traceback
            traceback.print_exc()
        except Exception as e:
            self.log_status(f"❌ Error showing statistics: {str(e)}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Statistics Error", 
                               f"Error showing statistics:\n{str(e)}\n\nCheck console for details.")
            import traceback
            traceback.print_exc()
    
    def export_results(self):
        """Export detection results."""
        if self.detection_results is None:
            return
        
        from PySide6.QtWidgets import QFileDialog
        
        # Get export file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Microplastic Detection Results", "",
            "CSV File (*.csv);;NumPy Array (*.npy);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.csv'):
                self.export_to_csv(file_path)
            elif file_path.endswith('.npy'):
                self.export_to_npy(file_path)
            
            self.log_status(f"💾 Results exported to: {file_path}")
        except Exception as e:
            self.log_status(f"❌ Export failed: {str(e)}")
    
    def export_to_csv(self, file_path):
        """Export results to CSV format."""
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            plastic_types = list(self.detection_results.keys())
            writer.writerow(['X', 'Y'] + plastic_types + ['Max_Score', 'Detected_Type'])
            
            # Get map dimensions
            first_map = self.detection_results[plastic_types[0]]
            if first_map.ndim == 1:
                size = int(np.sqrt(len(first_map)))
                shape = (size, size)
            else:
                shape = first_map.shape
            
            # Write data
            for ix in range(shape[0]):
                for iy in range(shape[1]):
                    scores = []
                    for ptype in plastic_types:
                        score_map = self.detection_results[ptype]
                        if score_map.ndim == 1:
                            idx = ix * shape[1] + iy
                            score = score_map[idx]
                        else:
                            score = score_map[ix, iy]
                        scores.append(score)
                    
                    max_score = max(scores)
                    detected_type = plastic_types[scores.index(max_score)] if max_score > 0.3 else 'None'
                    
                    writer.writerow([ix, iy] + scores + [max_score, detected_type])
    
    def export_to_npy(self, file_path):
        """Export results to NumPy format."""
        np.save(file_path, self.detection_results)
    
    def on_map_click(self, event):
        """Handle click on detection map to show spectrum."""
        if event.inaxes is None or not hasattr(self.parent_window, 'map_data'):
            return
        
        # Get click coordinates (these are image pixel coordinates)
        x_click = int(round(event.xdata))
        y_click = int(round(event.ydata))
        
        # Get map data from parent
        map_data = self.parent_window.map_data
        if map_data is None:
            return
        
        # Find the spectrum at this position
        try:
            # Get all unique positions
            all_positions = [(spec.x_pos, spec.y_pos) for spec in map_data.spectra.values()]
            unique_x = sorted(set(pos[0] for pos in all_positions))
            unique_y = sorted(set(pos[1] for pos in all_positions))
            
            # Map click coordinates to actual spectrum positions
            # The click coordinates are indices into the 2D map array
            if y_click < len(unique_y) and x_click < len(unique_x):
                actual_x = unique_x[x_click]
                actual_y = unique_y[y_click]
                
                # Find spectrum at this actual position
                spectrum = None
                for spec in map_data.spectra.values():
                    if spec.x_pos == actual_x and spec.y_pos == actual_y:
                        spectrum = spec
                        break
                
                if spectrum is not None:
                    self.selected_position = (x_click, y_click)
                    self.display_spectrum(spectrum, x_click, y_click)
                    self.log_status(f"📊 Spectrum at map position ({x_click}, {y_click}) = actual position ({actual_x}, {actual_y})")
                else:
                    self.log_status(f"⚠️ No spectrum found at actual position ({actual_x}, {actual_y})")
            else:
                self.log_status(f"⚠️ Click outside map bounds: ({x_click}, {y_click})")
        
        except Exception as e:
            self.log_status(f"❌ Error displaying spectrum: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_spectrum(self, spectrum, x_pos, y_pos):
        """Display the selected spectrum with overlaid reference spectra."""
        self.spectrum_figure.clear()
        ax = self.spectrum_figure.add_subplot(111)
        
        # Get wavenumbers and intensities
        wavenumbers = spectrum.wavenumbers
        intensities = spectrum.intensities.copy()
        
        # Apply baseline correction using the same method as detection
        from map_analysis_2d.analysis.microplastic_detector import MicroplasticDetector
        
        # Get baseline parameters from UI
        params = self.get_detection_parameters()
        baseline_params = params.get('baseline', {})
        
        # Apply baseline correction
        if baseline_params.get('method') == 'rolling_ball':
            # Use fast rolling ball
            from scipy.ndimage import minimum_filter1d
            baseline = minimum_filter1d(intensities, size=100, mode='nearest')
            intensities = intensities - baseline
        else:
            # Use ALS
            intensities = MicroplasticDetector.baseline_als(
                intensities,
                lam=baseline_params.get('lam', 1e6),
                p=baseline_params.get('p', 0.001),
                niter=baseline_params.get('niter', 10)
            )
        
        # Normalize spectrum for comparison
        intensities_norm = intensities - np.min(intensities)
        if np.max(intensities_norm) > 0:
            intensities_norm = intensities_norm / np.max(intensities_norm)
        
        # Plot spectrum
        ax.plot(wavenumbers, intensities_norm, 'b-', linewidth=2, 
                label='Map Spectrum (baseline corrected)', alpha=0.8)
        
        # Get detection scores and overlay reference spectra
        if self.detection_results is not None:
            scores_list = []
            
            # Calculate flat index for this position
            flat_idx = None
            if hasattr(self.parent_window, 'map_data'):
                map_data = self.parent_window.map_data
                all_positions = [(spec.x_pos, spec.y_pos) for spec in map_data.spectra.values()]
                unique_y = sorted(set(pos[1] for pos in all_positions))
                unique_x = sorted(set(pos[0] for pos in all_positions))
                
                try:
                    # Map click position to actual indices
                    if x_pos < len(unique_x) and y_pos < len(unique_y):
                        flat_idx = y_pos * len(unique_x) + x_pos
                except (ValueError, IndexError):
                    pass
            
            # Collect scores for each plastic type
            for plastic_type, score_map in self.detection_results.items():
                if plastic_type.startswith('_'):
                    continue  # Skip metadata
                
                # Get score at this position
                try:
                    if hasattr(score_map, 'ndim') and score_map.ndim == 2:
                        if y_pos < score_map.shape[0] and x_pos < score_map.shape[1]:
                            score = score_map[y_pos, x_pos]
                        else:
                            score = 0.0
                    elif flat_idx is not None and flat_idx < len(score_map):
                        score = score_map[flat_idx]
                    else:
                        score = 0.0
                except (IndexError, TypeError):
                    score = 0.0
                
                scores_list.append((plastic_type, float(score)))
            
            # Sort by score
            scores_list.sort(key=lambda x: x[1], reverse=True)
            
            # Get best match info from stored details (for template matching)
            match_info = None
            match_details = self.detection_results.get('_match_details', {})
            match_info_dict = self.detection_results.get('_match_info', {})
            
            if flat_idx is not None:
                match_info = match_details.get(flat_idx) or match_info_dict.get(flat_idx)
            
            # Overlay best matching reference spectrum
            best_match_type, best_score = scores_list[0] if scores_list else (None, 0)
            
            if best_score > 0.3:  # Only overlay if score is significant
                ref_spectrum = None
                matched_name = None
                
                # Try to get reference from stored match info
                if match_info and match_info.get('best_match'):
                    matched_name = match_info['best_match']
                elif match_info and match_info.get('matched_name'):
                    matched_name = match_info['matched_name']
                
                # Try to load reference spectrum from database
                if matched_name and hasattr(self.parent_window, 'database') and self.parent_window.database:
                    if matched_name in self.parent_window.database:
                        db_entry = self.parent_window.database[matched_name]
                        ref_wn = np.array(db_entry.get('wavenumbers', []))
                        ref_int = np.array(db_entry.get('intensities', []))
                        if len(ref_wn) > 0 and len(ref_int) > 0:
                            ref_spectrum = (ref_wn, ref_int)
                
                # Fallback: Create synthetic reference from peak signatures
                if ref_spectrum is None:
                    detector = MicroplasticDetector()
                    plastic_info = detector.PLASTIC_SIGNATURES.get(best_match_type, {})
                    if 'peaks' in plastic_info:
                        ref_wn = wavenumbers
                        ref_int = np.zeros_like(wavenumbers, dtype=float)
                        
                        for peak_pos in plastic_info['peaks']:
                            sigma = 20 / 2.355
                            gaussian = np.exp(-((wavenumbers - peak_pos)**2) / (2 * sigma**2))
                            ref_int += gaussian
                        
                        ref_spectrum = (ref_wn, ref_int)
                        matched_name = plastic_info.get('name', best_match_type)
                
                # Overlay the reference spectrum
                if ref_spectrum is not None:
                    ref_wn, ref_int = ref_spectrum
                    
                    # Normalize reference spectrum
                    ref_int_norm = ref_int - np.min(ref_int)
                    if np.max(ref_int_norm) > 0:
                        ref_int_norm = ref_int_norm / np.max(ref_int_norm)
                    
                    # Interpolate to match wavenumber range
                    if len(ref_wn) > 0:
                        ref_int_interp = np.interp(wavenumbers, ref_wn, ref_int_norm)
                    else:
                        ref_int_interp = ref_int_norm
                    
                    # Get color for this plastic type
                    detector = MicroplasticDetector()
                    plastic_info = detector.PLASTIC_SIGNATURES.get(best_match_type, {})
                    plastic_color = plastic_info.get('color', '#FF0000')
                    
                    display_name = matched_name if matched_name else best_match_type
                    
                    # Shorten display name for legend (truncate long database names)
                    short_name = display_name
                    if len(display_name) > 25:
                        short_name = display_name[:22] + "..."
                    
                    # Plot reference spectrum
                    ax.plot(wavenumbers, ref_int_interp * 0.9, color=plastic_color, 
                           linewidth=2, linestyle='--', alpha=0.7,
                           label=f'{short_name} ({best_score:.2f})')
            
            # Build compact score summary for title area
            top_scores = scores_list[:3]  # Top 3 only
            score_summary = " | ".join([f"{p}: {s:.2f}" for p, s in top_scores if s > 0.1])
        
        # Set labels and title
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=9)
        ax.set_ylabel('Intensity (norm.)', fontsize=9)
        
        # Compact title with position and top score
        title = f'Position ({x_pos}, {y_pos})'
        if self.detection_results and score_summary:
            title += f'  [{score_summary}]'
        ax.set_title(title, fontsize=9, fontweight='bold')
        
        # Legend outside plot area or compact
        ax.legend(fontsize=7, loc='upper right', framealpha=0.8, 
                  handlelength=1.5, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Note: X-axis is NOT inverted to match data order (low to high wavenumber)
        # This ensures template matching overlays align correctly with the spectrum
        
        # Force proper layout and canvas update
        self.spectrum_figure.tight_layout(pad=0.5)
        self.spectrum_canvas.draw_idle()
        self.spectrum_canvas.flush_events()
        
        self.log_status(f"📊 Displaying spectrum at ({x_pos}, {y_pos})")
    
    def refine_results(self):
        """Apply spatial filtering to refine detection results.
        
        This removes noise while keeping:
        1. Clusters of detections (real particles have spatial extent)
        2. Isolated high-score pixels that stand out from neighbors (small particles)
        """
        if self.detection_results is None:
            self.log_status("⚠️ No detection results to refine")
            return
        
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox, QGroupBox
        from scipy import ndimage
        
        # Create settings dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Refine Detection Results")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)
        
        # Info label
        info = QLabel(
            "Spatial filtering removes random noise while keeping real detections.\n"
            "• Clusters: Groups of adjacent pixels (real particles have area)\n"
            "• Weak Signal Mode: Uses adaptive thresholds to preserve weak but real signals"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Settings group
        settings_group = QGroupBox("Filter Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Erosion iterations (shrink detection regions)
        erosion_row = QHBoxLayout()
        erosion_row.addWidget(QLabel("Erosion iterations:"))
        self.erosion_spin = QSpinBox()
        self.erosion_spin.setRange(0, 10)
        self.erosion_spin.setValue(1)
        self.erosion_spin.setToolTip("Shrink detection regions by this many pixels.\nRemoves thin/noisy connections between clusters.")
        erosion_row.addWidget(self.erosion_spin)
        settings_layout.addLayout(erosion_row)
        
        # Minimum cluster size (after erosion)
        cluster_row = QHBoxLayout()
        cluster_row.addWidget(QLabel("Min cluster size (pixels):"))
        self.min_cluster_spin = QSpinBox()
        self.min_cluster_spin.setRange(1, 500)
        self.min_cluster_spin.setValue(5)
        self.min_cluster_spin.setToolTip("Minimum cluster size AFTER erosion.\nReal particles should have a core of at least this size.")
        cluster_row.addWidget(self.min_cluster_spin)
        settings_layout.addLayout(cluster_row)
        
        # Weak signal preservation mode
        self.weak_signal_mode = QCheckBox("Weak Signal Mode (Recommended for microplastics)")
        self.weak_signal_mode.setChecked(True)
        self.weak_signal_mode.setToolTip(
            "Uses adaptive thresholds based on local contrast.\n"
            "Keeps weak signals that stand out from background.\n"
            "Recommended when detecting weak plastics in noisy data."
        )
        settings_layout.addWidget(self.weak_signal_mode)
        
        # Minimum relative score (only used when weak signal mode is OFF)
        score_row = QHBoxLayout()
        score_row.addWidget(QLabel("Min cluster mean score:"))
        self.min_score_spin = QDoubleSpinBox()
        self.min_score_spin.setRange(0.0, 1.0)
        self.min_score_spin.setValue(0.55)
        self.min_score_spin.setSingleStep(0.05)
        self.min_score_spin.setToolTip("Absolute threshold (only used if Weak Signal Mode is OFF).\nClusters with mean score below this are removed.")
        score_row.addWidget(self.min_score_spin)
        settings_layout.addLayout(score_row)
        
        # Minimum local contrast (for weak signal mode)
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Min local contrast (σ):"))
        self.min_contrast_spin = QDoubleSpinBox()
        self.min_contrast_spin.setRange(1.0, 10.0)
        self.min_contrast_spin.setValue(2.0)
        self.min_contrast_spin.setSingleStep(0.5)
        self.min_contrast_spin.setToolTip(
            "Minimum standard deviations above local background.\n"
            "2.0 = keep signals 2σ above neighbors (recommended).\n"
            "Higher = more conservative (fewer detections)."
        )
        contrast_row.addWidget(self.min_contrast_spin)
        settings_layout.addLayout(contrast_row)
        
        layout.addWidget(settings_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply Filter")
        apply_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec() != QDialog.Accepted:
            return
        
        # Get settings
        erosion_iterations = self.erosion_spin.value()
        min_cluster_size = self.min_cluster_spin.value()
        min_mean_score = self.min_score_spin.value()
        weak_signal_mode = self.weak_signal_mode.isChecked()
        min_contrast_sigma = self.min_contrast_spin.value()
        
        mode_str = "Weak Signal (Adaptive)" if weak_signal_mode else f"Absolute (>{min_mean_score:.2f})"
        self.log_status(f"🎯 Applying spatial filter (mode: {mode_str}, erosion: {erosion_iterations}, min cluster: {min_cluster_size})...")
        
        # Get current threshold
        threshold = self.threshold_slider.value() / 100.0
        
        # Process each plastic type
        refined_results = {}
        total_before = 0
        total_after = 0
        
        for plastic_type, score_map in self.detection_results.items():
            if plastic_type.startswith('_'):
                refined_results[plastic_type] = score_map
                continue
            
            # Ensure score_map is 2D
            original_shape = score_map.shape
            if score_map.ndim == 1:
                # Try to reshape to 2D - assume square or get from map_data
                if hasattr(self.parent_window, 'map_data'):
                    map_data = self.parent_window.map_data
                    all_positions = [(spec.x_pos, spec.y_pos) for spec in map_data.spectra.values()]
                    unique_x = sorted(set(pos[0] for pos in all_positions))
                    unique_y = sorted(set(pos[1] for pos in all_positions))
                    n_rows, n_cols = len(unique_y), len(unique_x)
                    if n_rows * n_cols == len(score_map):
                        score_map_2d = score_map.reshape(n_rows, n_cols)
                    else:
                        # Can't reshape properly, skip refinement for this type
                        self.log_status(f"⚠️ Cannot reshape {plastic_type} score map for spatial filtering")
                        refined_results[plastic_type] = score_map
                        continue
                else:
                    # Assume square
                    side = int(np.sqrt(len(score_map)))
                    if side * side == len(score_map):
                        score_map_2d = score_map.reshape(side, side)
                    else:
                        refined_results[plastic_type] = score_map
                        continue
            else:
                score_map_2d = score_map
            
            # Create binary detection mask
            detection_mask = score_map_2d > threshold
            total_before += np.sum(detection_mask)
            
            # Apply morphological erosion to shrink regions and break weak connections
            if erosion_iterations > 0:
                eroded_mask = ndimage.binary_erosion(detection_mask, iterations=erosion_iterations)
                
                # CRITICAL: Find isolated pixels that were completely removed by erosion
                # These need special handling since they disappear entirely
                isolated_pixels = detection_mask & ~ndimage.binary_dilation(eroded_mask, iterations=erosion_iterations + 1)
            else:
                eroded_mask = detection_mask
                isolated_pixels = np.zeros_like(detection_mask, dtype=bool)
            
            # Label connected components on eroded mask
            labeled_array, num_features = ndimage.label(eroded_mask)
            
            # Create refined mask
            refined_mask = np.zeros_like(detection_mask)
            clusters_kept = 0
            clusters_removed = 0
            
            for label_id in range(1, num_features + 1):
                component_mask = labeled_array == label_id
                component_size = np.sum(component_mask)
                
                # Get scores for this cluster (from original, not eroded)
                component_scores = score_map_2d[component_mask]
                max_score = np.max(component_scores)
                mean_score = np.mean(component_scores)
                
                # Check if cluster passes filters
                keep_cluster = False
                
                if weak_signal_mode:
                    # ADAPTIVE MODE: Use local contrast analysis
                    # Calculate local background around this cluster
                    # Dilate cluster mask to get surrounding region
                    dilated_region = ndimage.binary_dilation(component_mask, iterations=3)
                    background_mask = dilated_region & ~component_mask
                    
                    if np.sum(background_mask) > 0:
                        background_scores = score_map_2d[background_mask]
                        bg_mean = np.mean(background_scores)
                        bg_std = np.std(background_scores)
                        
                        # For isolated detections, background is often all zeros
                        # In this case, any detection above threshold is significant
                        if bg_mean < threshold * 0.5 and bg_std < threshold * 0.3:
                            # Background is essentially noise/zero
                            # Keep if detection is clearly above threshold
                            if mean_score > threshold * 1.2:  # 20% above threshold
                                keep_cluster = True
                            elif max_score > 0.7:  # Or strong peak
                                keep_cluster = True
                        else:
                            # Normal contrast calculation
                            if bg_std > 0:
                                contrast = (mean_score - bg_mean) / bg_std
                            else:
                                # No variation but non-zero background
                                contrast = (mean_score - bg_mean) / (threshold * 0.1)
                            
                            # Keep if cluster stands out from local background
                            if contrast >= min_contrast_sigma:
                                keep_cluster = True
                        
                        # Additional rules regardless of contrast
                        # Keep if cluster is large enough (real particles have area)
                        if component_size >= min_cluster_size and mean_score > threshold:
                            keep_cluster = True
                        # Always keep very strong signals
                        if max_score > 0.8:
                            keep_cluster = True
                    else:
                        # No background to compare - isolated at edge
                        # Keep if above threshold (it's isolated, so likely real)
                        if mean_score > threshold:
                            keep_cluster = True
                        elif max_score > 0.7:
                            keep_cluster = True
                else:
                    # ABSOLUTE MODE: Use fixed thresholds (original behavior)
                    # Keep if cluster is large enough AND has high enough mean score
                    if component_size >= min_cluster_size and mean_score >= min_mean_score:
                        keep_cluster = True
                    # Always keep very strong signals
                    elif max_score > 0.8:
                        keep_cluster = True
                
                if keep_cluster:
                    # Recover pixels that were in original detection and connected to this cluster
                    if erosion_iterations > 0:
                        # Dilate back to original extent, but ONLY keep original detections
                        dilated = ndimage.binary_dilation(component_mask, iterations=erosion_iterations)
                        # Only recover pixels that were originally detected
                        recovered = dilated & detection_mask
                        refined_mask |= recovered
                    else:
                        refined_mask |= component_mask
                    clusters_kept += 1
                else:
                    clusters_removed += 1
            
            # Handle isolated pixels that were completely removed by erosion
            if erosion_iterations > 0 and np.any(isolated_pixels):
                # Label isolated pixel groups
                isolated_labels, n_isolated = ndimage.label(isolated_pixels)
                
                for iso_id in range(1, n_isolated + 1):
                    iso_mask = isolated_labels == iso_id
                    iso_scores = score_map_2d[iso_mask]
                    iso_mean = np.mean(iso_scores)
                    iso_max = np.max(iso_scores)
                    iso_size = np.sum(iso_mask)
                    
                    # Evaluate isolated pixels using same criteria
                    keep_isolated = False
                    
                    if weak_signal_mode:
                        # For isolated pixels, if they're above threshold, they're likely real
                        # (Random noise doesn't typically create isolated high-scoring pixels)
                        if iso_mean > threshold * 1.2:
                            keep_isolated = True
                        elif iso_max > 0.7:
                            keep_isolated = True
                        elif iso_size >= min_cluster_size and iso_mean > threshold:
                            keep_isolated = True
                    else:
                        # Absolute mode
                        if iso_size >= min_cluster_size and iso_mean >= min_mean_score:
                            keep_isolated = True
                        elif iso_max > 0.8:
                            keep_isolated = True
                    
                    if keep_isolated:
                        refined_mask |= iso_mask
                        clusters_kept += 1
                    else:
                        clusters_removed += 1
            
            # Apply refined mask to scores (zero out filtered pixels)
            refined_scores = score_map_2d.copy()
            refined_scores[~refined_mask] = 0
            
            # Reshape back to original if needed
            if score_map.ndim == 1:
                refined_scores = refined_scores.flatten()
            
            refined_results[plastic_type] = refined_scores
            total_after += np.sum(refined_mask)
            
            self.log_status(f"  {plastic_type}: kept {clusters_kept} clusters, removed {clusters_removed}")
        
        # Update results
        self.detection_results = refined_results
        
        # Get map shape for proper visualization
        map_shape = None
        if hasattr(self.parent_window, 'map_data'):
            map_data = self.parent_window.map_data
            map_shape = (len(map_data.y_positions), len(map_data.x_positions))
        
        # Redisplay with map dimensions
        self.display_results(refined_results, map_shape=map_shape)
        
        # Log summary
        removed = total_before - total_after
        self.log_status(f"✅ Refinement complete: {total_before:,} → {total_after:,} detections ({removed:,} removed, {removed/max(1,total_before)*100:.1f}% filtered)")
    
    def open_baseline_tester(self):
        """Open interactive baseline correction tester dialog."""
        from PySide6.QtWidgets import QMessageBox
        
        # Check if we have a selected spectrum
        if not hasattr(self, 'selected_position') or self.selected_position is None:
            QMessageBox.information(
                self, "No Spectrum Selected",
                "Please click on a spectrum in the map to select it first.\n\n"
                "Then click 'Test' to try different baseline correction methods."
            )
            return
        
        # Get the selected spectrum
        if not hasattr(self.parent_window, 'map_data') or self.parent_window.map_data is None:
            QMessageBox.warning(self, "No Data", "No map data loaded.")
            return
        
        map_data = self.parent_window.map_data
        x_pos, y_pos = self.selected_position
        
        # Find the spectrum
        unique_x = sorted(set(s.x_pos for s in map_data.spectra.values()))
        unique_y = sorted(set(s.y_pos for s in map_data.spectra.values()))
        
        if x_pos < len(unique_x) and y_pos < len(unique_y):
            actual_x = unique_x[x_pos]
            actual_y = unique_y[y_pos]
            
            spectrum = None
            for spec in map_data.spectra.values():
                if spec.x_pos == actual_x and spec.y_pos == actual_y:
                    spectrum = spec
                    break
            
            if spectrum is not None:
                # Open the baseline tester dialog
                try:
                    from map_analysis_2d.ui.dialogs.baseline_tester_dialog import BaselineTesterDialog
                    
                    dialog = BaselineTesterDialog(
                        spectrum.wavenumbers,
                        spectrum.intensities,
                        self
                    )
                    
                    if dialog.exec():
                        # User clicked "Apply to All Spectra"
                        method, params = dialog.get_selected_method()
                        
                        # Update the baseline preset combo to match
                        method_map = {
                            ("rolling_ball", 100): "Rolling Ball (Fast) - Recommended",
                            ("als", (1e5, 0.01, 5)): "ALS (Fast)",
                            ("als", (1e5, 0.01, 10)): "ALS (Moderate)",
                            ("als", (1e6, 0.01, 10)): "ALS (Aggressive)",
                            ("als", (1e6, 0.001, 10)): "ALS (Conservative)",
                            ("als", (1e7, 0.002, 20)): "ALS (Ultra Smooth)"
                        }
                        
                        # Try to match to a preset
                        matched = False
                        if method == "rolling_ball":
                            key = (method, params.get("window", 100))
                        else:
                            key = (method, (params.get("lam", 1e6), params.get("p", 0.001), params.get("niter", 10)))
                        
                        if key in method_map:
                            self.baseline_preset_combo.setCurrentText(method_map[key])
                            matched = True
                        
                        if matched:
                            self.log_status(f"✓ Baseline method updated to: {self.baseline_preset_combo.currentText()}")
                        else:
                            self.log_status(f"✓ Custom baseline parameters set: {method} with {params}")
                            QMessageBox.information(
                                self, "Custom Parameters",
                                f"Custom baseline parameters selected:\n"
                                f"Method: {method}\n"
                                f"Parameters: {params}\n\n"
                                f"Note: These custom parameters will be used for detection."
                            )
                
                except ImportError as e:
                    QMessageBox.critical(
                        self, "Import Error",
                        f"Could not load baseline tester dialog:\n{str(e)}"
                    )
                    import traceback
                    traceback.print_exc()
            else:
                QMessageBox.warning(self, "Spectrum Not Found", "Could not find spectrum at selected position.")
        else:
            QMessageBox.warning(self, "Invalid Position", "Selected position is out of bounds.")
