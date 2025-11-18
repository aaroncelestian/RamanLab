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
        params_layout.addWidget(self.baseline_preset_combo, 1, 1, 1, 2)
        
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
            'Hybrid (Recommended)'
        ])
        self.method_combo.setCurrentIndex(2)
        self.method_combo.setToolTip(
            "Peak-based: Fast, uses characteristic peaks\n"
            "Database: Slower, matches against reference spectra\n"
            "Hybrid: Uses both methods"
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
        
    def display_results(self, results: Dict[str, np.ndarray]):
        """Display detection results in the visualization panel."""
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
            
            # Reshape if needed (assuming square map)
            if score_map.ndim == 1:
                size = int(np.sqrt(len(score_map)))
                if size * size == len(score_map):
                    score_map = score_map.reshape(size, size)
            
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
        
        # Enable export and statistics
        self.export_btn.setEnabled(True)
        self.stats_btn.setEnabled(True)
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
        intensities = spectrum.intensities
        
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
        ax.plot(wavenumbers, intensities_norm, 'b-', linewidth=2, label='Map Spectrum (baseline corrected)', alpha=0.8)
        
        # Get detection scores and overlay reference spectra
        if self.detection_results is not None:
            score_text = f"Position: ({x_pos}, {y_pos})\\n\\n"
            scores_list = []
            
            # Test against negative controls
            detector = MicroplasticDetector()
            negative_match, negative_score = detector.test_negative_controls(
                wavenumbers, intensities_norm, baseline_corrected=True
            )
            
            for plastic_type, score_map in self.detection_results.items():
                if plastic_type.startswith('_'):
                    continue  # Skip metadata
                
                # Get score at this position
                if score_map.ndim == 2:
                    score = score_map[y_pos, x_pos]
                else:
                    # Flatten index
                    size = int(np.sqrt(len(score_map)))
                    idx = y_pos * size + x_pos
                    if idx < len(score_map):
                        score = score_map[idx]
                    else:
                        score = 0.0
                
                # Calculate confidence level
                window_scores = {}  # Would need to store these during detection
                confidence, conf_details = detector.calculate_confidence_level(
                    score, window_scores, negative_score
                )
                
                # Color code by confidence
                conf_emoji = {
                    'HIGH': '🟢',
                    'MEDIUM': '🟡',
                    'LOW': '🟠',
                    'VERY_LOW': '🔴'
                }.get(confidence, '⚪')
                
                scores_list.append((plastic_type, score, confidence))
                score_text += f"{conf_emoji} {plastic_type}: {score:.3f} ({confidence})\\n"
            
            # Add negative control info
            score_text += f"\\n⚠️ Best Non-Plastic Match:\\n"
            score_text += f"   {negative_match}: {negative_score:.3f}"
            
            # Sort by score and overlay top matches
            scores_list.sort(key=lambda x: x[1], reverse=True)
            
            # Get reference spectra from parent's detector
            from map_analysis_2d.analysis.microplastic_detector import MicroplasticDetector
            detector = MicroplasticDetector()
            
            # Try to load references if available
            n_refs = 0
            if hasattr(self.parent_window, 'database') and self.parent_window.database:
                # Database is already filtered for plastics, so pass it directly
                n_refs = len(self.parent_window.database)
                detector.load_plastic_references(self.parent_window.database)
                self.log_status(f"Loaded {n_refs} plastic references for overlay")
            else:
                self.log_status("⚠️ No database available for reference overlay")
            
            # Check if we have stored match details for this position
            match_details = self.detection_results.get('_match_details', {})
            self.log_status(f"🔍 Found {len(match_details)} stored matches in detection results")
            
            # Calculate flat index for this position
            flat_idx = None
            if hasattr(self.parent_window, 'map_data'):
                map_data = self.parent_window.map_data
                all_positions = [(spec.x_pos, spec.y_pos) for spec in map_data.spectra.values()]
                unique_y = sorted(set(pos[1] for pos in all_positions))
                unique_x = sorted(set(pos[0] for pos in all_positions))
                
                # Find the index in the flattened array
                # The detector processes spectra in row-major order
                try:
                    y_idx = unique_y.index(y_pos)
                    x_idx = unique_x.index(x_pos)
                    flat_idx = y_idx * len(unique_x) + x_idx
                    self.log_status(f"📍 Position ({x_pos}, {y_pos}) → flat index {flat_idx}")
                except ValueError:
                    self.log_status(f"⚠️ Position ({x_pos}, {y_pos}) not found in map")
            
            # Get match info from stored details
            match_info = match_details.get(flat_idx) if flat_idx is not None else None
            if match_info:
                self.log_status(f"✓ Found match info: {match_info.get('matched_name', 'Unknown')}")
            else:
                self.log_status(f"ℹ️ No match info stored for this position")
            
            # Overlay best match (highest score)
            best_match_type, best_score = scores_list[0] if scores_list else (None, 0)
            
            if best_score > 0.1:  # Only overlay if score is significant
                # Get reference spectrum for best match
                ref_spectrum = None
                matched_name = None
                
                # First try to use stored match info
                if match_info and match_info.get('matched_name'):
                    matched_name = match_info['matched_name']
                    self.log_status(f"✓ Using stored match: {matched_name}")
                    
                    # Get the actual reference spectrum from database
                    if n_refs > 0 and matched_name in self.parent_window.database:
                        db_entry = self.parent_window.database[matched_name]
                        ref_wn = db_entry['wavenumbers']
                        ref_int = db_entry['intensities']
                        ref_spectrum = (ref_wn, ref_int)
                        self.log_status(f"✓ Loaded reference spectrum: {matched_name}")
                
                # Fallback: Try to get from loaded references by plastic type
                if ref_spectrum is None and n_refs > 0 and best_match_type in detector.reference_spectra:
                    ref_wn, ref_int = detector.reference_spectra[best_match_type]
                    ref_spectrum = (ref_wn, ref_int)
                    matched_name = best_match_type
                    self.log_status(f"✓ Using database reference for {best_match_type}")
                
                # If no database reference, create synthetic reference from peak signatures
                if ref_spectrum is None:
                    self.log_status(f"ℹ️ No database reference for {best_match_type}, using peak signatures")
                    plastic_info = detector.PLASTIC_SIGNATURES.get(best_match_type, {})
                    if 'peaks' in plastic_info:
                        # Create synthetic spectrum from peak positions
                        ref_wn = wavenumbers  # Use same wavenumber range
                        ref_int = np.zeros_like(wavenumbers)
                        
                        # Add Gaussian peaks at expected positions
                        for peak_pos in plastic_info['peaks']:
                            # Find closest wavenumber
                            idx = np.argmin(np.abs(wavenumbers - peak_pos))
                            # Add Gaussian peak (width ~20 cm-1)
                            sigma = 20 / 2.355  # FWHM to sigma
                            gaussian = np.exp(-((wavenumbers - peak_pos)**2) / (2 * sigma**2))
                            ref_int += gaussian
                        
                        ref_spectrum = (ref_wn, ref_int)
                
                # Overlay the reference spectrum
                if ref_spectrum is not None:
                    ref_wn, ref_int = ref_spectrum
                    
                    # Normalize reference spectrum
                    ref_int_norm = ref_int - np.min(ref_int)
                    if np.max(ref_int_norm) > 0:
                        ref_int_norm = ref_int_norm / np.max(ref_int_norm)
                    
                    # Interpolate to match wavenumber range if needed
                    if not np.array_equal(ref_wn, wavenumbers):
                        ref_int_interp = np.interp(wavenumbers, ref_wn, ref_int_norm)
                    else:
                        ref_int_interp = ref_int_norm
                    
                    # Get plastic info for color and name
                    plastic_info = detector.PLASTIC_SIGNATURES.get(best_match_type, {})
                    plastic_name = plastic_info.get('name', best_match_type)
                    plastic_color = plastic_info.get('color', '#FF0000')
                    
                    # Use matched name if available, otherwise use plastic type
                    display_name = matched_name if matched_name else plastic_name
                    
                    # Plot reference spectrum with offset for visibility
                    ax.plot(wavenumbers, ref_int_interp * 0.9, color=plastic_color, 
                           linewidth=2, linestyle='--', alpha=0.7,
                           label=f'{display_name}\n(score: {best_score:.3f})')
                    
                    self.log_status(f"📊 Overlaid {display_name} reference (score: {best_score:.3f})")
            
            # Add score text to plot
            ax.text(0.02, 0.98, score_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=9)
        ax.set_ylabel('Intensity (normalized)', fontsize=9)
        ax.set_title(f'Spectrum at Position ({x_pos}, {y_pos})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Force proper layout and canvas update
        self.spectrum_figure.tight_layout()
        self.spectrum_canvas.draw_idle()
        self.spectrum_canvas.flush_events()
        
        self.log_status(f"📊 Displaying spectrum at ({x_pos}, {y_pos})")
