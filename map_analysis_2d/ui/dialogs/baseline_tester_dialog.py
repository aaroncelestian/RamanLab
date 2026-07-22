"""
Interactive Baseline Correction Tester Dialog

Allows users to test different baseline correction methods on a selected spectrum
before applying to the entire dataset.
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QLabel, QComboBox, QDoubleSpinBox,
                              QSpinBox, QGridLayout)
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Import custom toolbar
try:
    import sys
    from pathlib import Path
    core_path = Path(__file__).parent.parent.parent.parent / 'core'
    if str(core_path) not in sys.path:
        sys.path.insert(0, str(core_path))
    from matplotlib_config import CompactNavigationToolbar as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class BaselineTesterDialog(QDialog):
    """Interactive dialog for testing baseline correction methods."""
    
    def __init__(self, wavenumbers, intensities, parent=None):
        super().__init__(parent)
        self.wavenumbers = wavenumbers
        self.raw_intensities = intensities.copy()
        self.corrected_intensities = intensities.copy()
        self.selected_method = None
        self.selected_params = None
        
        self.setWindowTitle("Baseline Correction Tester")
        self.setMinimumSize(900, 600)
        self.setup_ui()
        self.update_correction()  # Apply initial baseline correction
        self.update_plot()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # === Controls Panel ===
        controls = self.create_controls()
        layout.addWidget(controls)
        
        # === Plot Area ===
        plot_widget = self.create_plot_area()
        layout.addWidget(plot_widget, stretch=1)
        
        # === Action Buttons ===
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply to All Spectra")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.apply_btn.clicked.connect(self.accept)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.apply_btn)
        
        layout.addLayout(button_layout)
        
    def create_controls(self):
        """Create the controls panel."""
        group = QGroupBox("Baseline Correction Method")
        layout = QGridLayout(group)
        
        # Method selection
        layout.addWidget(QLabel("Method:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Rolling Ball (Fast)",
            "SNIP (Fast)",
            "SNIP (Moderate)",
            "Custom SNIP",
            "ALS (Fast)",
            "ALS (Moderate)",
            "ALS (Aggressive)",
            "ALS (Conservative)",
            "ALS (Ultra Smooth)",
            "Custom ALS"
        ])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        layout.addWidget(self.method_combo, 0, 1, 1, 2)
        
        # Rolling Ball window size
        layout.addWidget(QLabel("Rolling Ball Window:"), 1, 0)
        self.rb_window_spin = QSpinBox()
        self.rb_window_spin.setRange(10, 500)
        self.rb_window_spin.setValue(100)
        self.rb_window_spin.setSuffix(" points")
        self.rb_window_spin.valueChanged.connect(self.update_correction)
        layout.addWidget(self.rb_window_spin, 1, 1)
        
        # SNIP iterations
        layout.addWidget(QLabel("SNIP Iterations:"), 2, 0)
        self.snip_iter_spin = QSpinBox()
        self.snip_iter_spin.setRange(1, 200)
        self.snip_iter_spin.setValue(40)
        self.snip_iter_spin.setToolTip(
            "Clipping window iterations (higher = broader baseline features removed; "
            "too high can erode wide peaks)"
        )
        self.snip_iter_spin.valueChanged.connect(self.update_correction)
        layout.addWidget(self.snip_iter_spin, 2, 1)
        
        # ALS Lambda parameter
        layout.addWidget(QLabel("ALS Lambda (λ):"), 3, 0)
        self.als_lambda_spin = QDoubleSpinBox()
        self.als_lambda_spin.setRange(1e3, 1e9)
        self.als_lambda_spin.setValue(1e6)
        self.als_lambda_spin.setDecimals(0)
        self.als_lambda_spin.setSingleStep(1e5)
        self.als_lambda_spin.setToolTip("Smoothness parameter (higher = smoother baseline)")
        self.als_lambda_spin.valueChanged.connect(self.update_correction)
        layout.addWidget(self.als_lambda_spin, 3, 1)
        
        # ALS p parameter
        layout.addWidget(QLabel("ALS p:"), 4, 0)
        self.als_p_spin = QDoubleSpinBox()
        self.als_p_spin.setRange(0.0001, 0.1)
        self.als_p_spin.setValue(0.001)
        self.als_p_spin.setDecimals(4)
        self.als_p_spin.setSingleStep(0.001)
        self.als_p_spin.setToolTip("Asymmetry parameter (lower = more aggressive removal)")
        self.als_p_spin.valueChanged.connect(self.update_correction)
        layout.addWidget(self.als_p_spin, 4, 1)
        
        # ALS iterations
        layout.addWidget(QLabel("ALS Iterations:"), 5, 0)
        self.als_iter_spin = QSpinBox()
        self.als_iter_spin.setRange(1, 50)
        self.als_iter_spin.setValue(10)
        self.als_iter_spin.valueChanged.connect(self.update_correction)
        layout.addWidget(self.als_iter_spin, 5, 1)
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        layout.addWidget(self.info_label, 6, 0, 1, 3)
        
        self.on_method_changed(self.method_combo.currentText())
        
        return group
        
    def create_plot_area(self):
        """Create the plot area."""
        widget = QGroupBox("Spectrum Preview")
        layout = QVBoxLayout(widget)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Add toolbar
        toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
        return widget
        
    def on_method_changed(self, method):
        """Handle method selection change."""
        is_rolling_ball = "Rolling Ball" in method
        is_snip = "SNIP" in method
        is_als = "ALS" in method
        is_custom_als = method == "Custom ALS"
        is_custom_snip = method == "Custom SNIP"
        
        # Enable/disable controls based on method
        self.rb_window_spin.setEnabled(is_rolling_ball)
        self.snip_iter_spin.setEnabled(is_snip)
        self.als_lambda_spin.setEnabled(is_als)
        self.als_p_spin.setEnabled(is_als)
        self.als_iter_spin.setEnabled(is_als)
        
        # Set preset values for non-custom methods
        if is_snip and not is_custom_snip:
            snip_presets = {
                "SNIP (Fast)": 20,
                "SNIP (Moderate)": 40,
            }
            if method in snip_presets:
                self.snip_iter_spin.setValue(snip_presets[method])
        
        if is_als and not is_custom_als:
            presets = {
                "ALS (Fast)": (1e5, 0.01, 5),
                "ALS (Moderate)": (1e5, 0.01, 10),
                "ALS (Aggressive)": (1e6, 0.01, 10),
                "ALS (Conservative)": (1e6, 0.001, 10),
                "ALS (Ultra Smooth)": (1e7, 0.002, 20)
            }
            if method in presets:
                lam, p, niter = presets[method]
                self.als_lambda_spin.setValue(lam)
                self.als_p_spin.setValue(p)
                self.als_iter_spin.setValue(niter)
        
        # Update info label
        if is_rolling_ball:
            self.info_label.setText(
                "Rolling Ball: Fast method using minimum filter. Good for broad fluorescence backgrounds."
            )
        elif is_snip:
            self.info_label.setText(
                "SNIP: Statistics-sensitive Non-linear Iterative Peak-clipping. "
                "Fast on large maps; higher iterations remove broader background."
            )
        else:
            self.info_label.setText(
                "ALS: Asymmetric Least Squares. Higher λ = smoother, lower p = more aggressive."
            )
        
        self.update_correction()
        
    def update_correction(self):
        """Apply baseline correction and update plot."""
        method = self.method_combo.currentText()
        
        if "Rolling Ball" in method:
            window = self.rb_window_spin.value()
            baseline = minimum_filter1d(self.raw_intensities, size=window, mode='nearest')
            self.corrected_intensities = self.raw_intensities - baseline
            self.selected_method = "rolling_ball"
            self.selected_params = {"window": window}
        elif "SNIP" in method:
            iterations = self.snip_iter_spin.value()
            self.corrected_intensities = self.baseline_snip(
                self.raw_intensities, iterations=iterations
            )
            self.selected_method = "snip"
            self.selected_params = {"iterations": iterations}
        else:
            lam = self.als_lambda_spin.value()
            p = self.als_p_spin.value()
            niter = self.als_iter_spin.value()
            self.corrected_intensities = self.baseline_als(self.raw_intensities, lam, p, niter)
            self.selected_method = "als"
            self.selected_params = {"lam": lam, "p": p, "niter": niter}
        
        self.update_plot()
        
    @staticmethod
    def baseline_snip(intensities, iterations=40, decreasing=True):
        """
        SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) baseline.
        
        Uses an LLS transform for improved clipping on Raman-like spectra.
        Returns baseline-corrected intensities.
        """
        y = np.asarray(intensities, dtype=np.float64)
        n = y.size
        if n < 3 or iterations < 1:
            return y.copy()
        
        iterations = min(int(iterations), max(1, n // 2 - 1))
        
        # LLS transform (safe for zeros / small negatives)
        offset = max(0.0, -float(np.min(y))) + 1.0
        z = np.log(np.log(np.sqrt(y + offset) + 1.0) + 1.0)
        
        window_sizes = (
            range(iterations, 0, -1) if decreasing else range(1, iterations + 1)
        )
        for p in window_sizes:
            z[p:-p] = np.minimum(z[p:-p], 0.5 * (z[:-2 * p] + z[2 * p:]))
        
        baseline = (np.exp(np.exp(z) - 1.0) - 1.0) ** 2 - offset
        return y - baseline
        
    @staticmethod
    def baseline_als(intensities, lam=1e6, p=0.001, niter=10):
        """Asymmetric Least Squares baseline correction."""
        L = len(intensities)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        
        for i in range(niter):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w * intensities)
            w = p * (intensities > z) + (1 - p) * (intensities < z)
        
        return intensities - z
        
    def update_plot(self):
        """Update the plot with current data."""
        self.ax.clear()
        
        # Calculate baseline for visualization
        baseline = self.raw_intensities - self.corrected_intensities
        
        # Plot raw spectrum (with fluorescence)
        self.ax.plot(self.wavenumbers, self.raw_intensities, 
                    'gray', alpha=0.6, linewidth=1.5, label='Raw (with fluorescence)', zorder=1)
        
        # Plot estimated baseline
        self.ax.plot(self.wavenumbers, baseline,
                    'orange', alpha=0.7, linewidth=2, linestyle='--', label='Estimated Baseline', zorder=2)
        
        # Plot corrected spectrum (peaks only)
        self.ax.plot(self.wavenumbers, self.corrected_intensities, 
                    'red', linewidth=2, label='Baseline Corrected', zorder=3)
        
        self.ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
        self.ax.set_ylabel('Intensity (a.u.)', fontsize=11)
        self.ax.set_title('Baseline Correction Preview - Adjust parameters and watch the red line change!', 
                         fontsize=12, fontweight='bold')
        self.ax.legend(loc='best', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        
        # Add text showing method and params
        method_text = f"Method: {self.selected_method}"
        if self.selected_params:
            if self.selected_method == "rolling_ball":
                method_text += f"\nWindow: {self.selected_params.get('window', 100)}"
            elif self.selected_method == "snip":
                method_text += f"\nIterations: {self.selected_params.get('iterations', 40)}"
            else:
                method_text += (
                    f"\nλ={self.selected_params.get('lam', 1e6):.0e}, "
                    f"p={self.selected_params.get('p', 0.001):.4f}, "
                    f"iter={self.selected_params.get('niter', 10)}"
                )
        
        self.ax.text(0.02, 0.98, method_text, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9, family='monospace')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def get_selected_method(self):
        """Get the selected baseline correction method and parameters."""
        return self.selected_method, self.selected_params
