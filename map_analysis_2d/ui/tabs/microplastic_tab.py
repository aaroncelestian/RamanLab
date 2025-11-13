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
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import numpy as np


class MicroplasticDetectionTab(QWidget):
    """Tab for microplastic detection in 2D Raman maps."""
    
    # Signals
    detection_started = Signal()
    detection_completed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.detection_results = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # ===== Control Panel =====
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
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
        
        # Baseline correction aggressiveness
        params_layout.addWidget(QLabel("Baseline Correction:"), 1, 0)
        self.baseline_combo = QComboBox()
        self.baseline_combo.addItems(['Mild', 'Moderate', 'Aggressive', 'Very Aggressive'])
        self.baseline_combo.setCurrentText('Aggressive')
        self.baseline_combo.setToolTip("Higher = more fluorescence removal")
        params_layout.addWidget(self.baseline_combo, 1, 1, 1, 2)
        
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
        
        return panel
    
    def create_visualization_panel(self):
        """Create the visualization panel with matplotlib figures."""
        panel = QGroupBox("Detection Results - Spatial Maps")
        layout = QVBoxLayout(panel)
        
        # Create matplotlib figure with subplots
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
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
    
    def update_threshold_label(self, value):
        """Update threshold label when slider changes."""
        self.threshold_label.setText(f"{value/100:.2f}")
    
    def get_selected_plastics(self):
        """Get list of selected plastic types."""
        return [code for code, checkbox in self.plastic_checkboxes.items() 
                if checkbox.isChecked()]
    
    def get_detection_parameters(self):
        """Get current detection parameters."""
        # Baseline correction parameters
        baseline_params = {
            'Mild': {'lam': 1e5, 'p': 0.01},
            'Moderate': {'lam': 1e6, 'p': 0.005},
            'Aggressive': {'lam': 1e7, 'p': 0.001},
            'Very Aggressive': {'lam': 1e8, 'p': 0.0001}
        }
        
        return {
            'plastic_types': self.get_selected_plastics(),
            'threshold': self.threshold_slider.value() / 100.0,
            'baseline': baseline_params[self.baseline_combo.currentText()],
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
        self.log_status("⏹ Detection stopped by user")
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
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
    
    def display_results(self, results):
        """Display detection results as spatial maps."""
        self.detection_results = results
        self.figure.clear()
        
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
            
            # Plot heatmap
            im = ax.imshow(score_map, cmap='hot', interpolation='nearest',
                          vmin=0, vmax=1, aspect='auto')
            ax.set_title(f'{plastic_name}\n({np.sum(score_map > 0.3)} detections)',
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
        total_detections = sum(np.sum(results[pt] > 0.3) for pt in plastic_types)
        self.log_status(f"✅ Detection complete! Found {total_detections} potential microplastic locations")
    
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
