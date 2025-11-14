"""
Microplastic Detection Tab for 2D Map Analysis

Provides UI for detecting microplastics in Raman spectroscopy maps.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QCheckBox, QLabel, QSlider, 
                              QProgressBar, QTextEdit, QComboBox, QDoubleSpinBox,
                              QGridLayout, QFrame, QSpinBox)
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
        panel = QGroupBox("Microplastic Detection Settings")
        layout = QVBoxLayout(panel)
        
        # === Database Loading ===
        db_button = QPushButton("📁 Load Reference Database")
        db_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        db_button.clicked.connect(lambda: self.parent_window.load_database_file() if hasattr(self.parent_window, 'load_database_file') else None)
        layout.addWidget(db_button)
        
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
        
        # Select/Deselect all buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_plastics)
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all_plastics)
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addStretch()
        
        plastic_layout.addLayout(button_layout, (len(plastic_types) + 1) // 2, 0, 1, 4)
        layout.addWidget(plastic_group)
        
        # === Detection Parameters ===
        params_group = QGroupBox("Detection Parameters")
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
        button_layout = QHBoxLayout()
        
        self.scan_btn = QPushButton("🔍 Scan Map for Microplastics")
        self.scan_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        self.scan_btn.clicked.connect(self.start_detection)
        button_layout.addWidget(self.scan_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)
        button_layout.addWidget(self.stop_btn)
        
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        
        # === Spectrum Display (in control panel) ===
        spectrum_group = QGroupBox("Selected Spectrum (Click map to view)")
        spectrum_layout = QVBoxLayout(spectrum_group)
        spectrum_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create matplotlib figure for spectrum with proper DPI
        self.spectrum_figure = Figure(figsize=(5, 4), dpi=100)
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        self.spectrum_canvas.setMinimumHeight(300)
        
        # Add navigation toolbar for spectrum
        spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, self)
        
        spectrum_layout.addWidget(spectrum_toolbar)
        spectrum_layout.addWidget(self.spectrum_canvas)
        layout.addWidget(spectrum_group)
        
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
    
    def select_all_plastics(self):
        """Select all plastic types."""
        for checkbox in self.plastic_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_plastics(self):
        """Deselect all plastic types."""
        for checkbox in self.plastic_checkboxes.values():
            checkbox.setChecked(False)
    
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
        selected = self.get_selected_plastics()
        if not selected:
            self.log_status("⚠️ Please select at least one plastic type to detect")
            return
        
        self.log_status(f"🔍 Starting detection for: {', '.join(selected)}")
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        
        # Emit signal to parent window
        self.detection_started.emit()
    
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
    
    def display_results(self, results):
        """Display detection results as spatial maps."""
        self.detection_results = results
        self.figure.clear()
        
        # Get current threshold
        threshold = self.threshold_slider.value() / 100.0
        
        plastic_types = list(results.keys())
        n_types = len(plastic_types)
        
        # Determine grid layout
        if n_types <= 4:
            nrows, ncols = 1, n_types
        else:
            nrows, ncols = 2, 4
        
        # Plot each plastic type
        for idx, plastic_type in enumerate(plastic_types):
            if idx >= 8:  # Max 8 plots
                break
            
            ax = self.figure.add_subplot(nrows, ncols, idx + 1)
            score_map = results[plastic_type]
            
            # Get plastic info
            from map_analysis_2d.analysis.microplastic_detector import MicroplasticDetector
            detector = MicroplasticDetector()
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
        
        # Enable export
        self.export_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Log summary
        total_detections = sum(np.sum(results[pt] > threshold) for pt in plastic_types)
        self.log_status(f"✅ Detection complete! Found {total_detections} potential microplastic locations (threshold: {threshold:.2f})")
    
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
            score_text = f"Position: ({x_pos}, {y_pos})\\n"
            scores_list = []
            
            for plastic_type, score_map in self.detection_results.items():
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
                
                scores_list.append((plastic_type, score))
                score_text += f"{plastic_type}: {score:.3f}\\n"
            
            # Sort by score and overlay top matches
            scores_list.sort(key=lambda x: x[1], reverse=True)
            
            # Get reference spectra from parent's detector
            from map_analysis_2d.analysis.microplastic_detector import MicroplasticDetector
            detector = MicroplasticDetector()
            
            # Try to load references if available
            n_refs = 0
            if hasattr(self.parent_window, 'database') and self.parent_window.database:
                n_refs = detector.load_plastic_references(self.parent_window.database, 'Plastics')
                self.log_status(f"Loaded {n_refs} plastic references for overlay")
            else:
                self.log_status("⚠️ No database available for reference overlay")
            
            # Overlay top 3 matches with score > 0.1
            if n_refs > 0:
                colors = ['r', 'g', 'orange']
                overlays_added = 0
                
                for i, (plastic_type, score) in enumerate(scores_list[:3]):
                    if score > 0.1 and i < len(colors):
                        # Try to find reference spectrum
                        ref_spectrum = None
                        plastic_name = detector.PLASTIC_SIGNATURES[plastic_type]['name'].lower()
                        
                        # Try exact match first
                        if plastic_type in detector.reference_spectra:
                            ref_spectrum = detector.reference_spectra[plastic_type]
                            self.log_status(f"Found exact match for {plastic_type}")
                        else:
                            # Try fuzzy match
                            for key, spectrum in detector.reference_spectra.items():
                                if plastic_name in key.lower() or plastic_type.lower() in key.lower():
                                    ref_spectrum = spectrum
                                    self.log_status(f"Matched {plastic_type} to reference: {key}")
                                    break
                        
                        if ref_spectrum is not None:
                            ref_wn, ref_int = ref_spectrum
                            
                            # Normalize reference spectrum
                            ref_int_norm = ref_int - np.min(ref_int)
                            if np.max(ref_int_norm) > 0:
                                ref_int_norm = ref_int_norm / np.max(ref_int_norm)
                            
                            # Plot reference spectrum
                            ax.plot(ref_wn, ref_int_norm, colors[i], linewidth=2, 
                                   label=f'{plastic_type} Ref (score: {score:.3f})',
                                   alpha=0.7, linestyle='--')
                            overlays_added += 1
                        else:
                            self.log_status(f"⚠️ No reference found for {plastic_type}")
                
                if overlays_added == 0:
                    self.log_status("⚠️ No reference spectra could be matched")
            else:
                self.log_status("⚠️ No plastic references in database")
            
            # Position text box in upper left, but below the title
            ax.text(0.02, 0.95, score_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
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
