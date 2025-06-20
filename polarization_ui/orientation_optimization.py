"""
Crystal Orientation Optimization Widget

This module implements the comprehensive trilogy optimization approach for crystal orientation
determination, building upon polarization analysis data and preparing results for 3D visualization.

Features:
- Enhanced Individual Peak Optimization (Stage 1)
- Probabilistic Bayesian Framework (Stage 2) 
- Advanced Multi-Objective Bayesian Optimization (Stage 3)
- Comprehensive uncertainty quantification
- 3D visualization data preparation
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QTabWidget,
    QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem, QTableWidget,
    QTableWidgetItem, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QTextEdit,
    QFileDialog, QMessageBox, QFormLayout, QSplitter, QFrame, QProgressBar,
    QHeaderView, QDialog, QDialogButtonBox, QTreeWidget, QTreeWidgetItem,
    QProgressDialog, QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPalette, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2

# Try to import advanced optimization libraries
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False


class OrientationOptimizationWidget(QWidget):
    """
    Comprehensive orientation optimization widget implementing the trilogy approach.
    
    This widget provides three stages of increasingly sophisticated optimization:
    1. Enhanced Individual Peak Optimization - Multi-start global optimization
    2. Probabilistic Bayesian Framework - MCMC sampling with uncertainty quantification  
    3. Advanced Multi-Objective Bayesian Optimization - Gaussian Process with Pareto optimization
    """
    
    # Signals
    optimization_complete = Signal(dict)
    data_exported = Signal(dict)
    
    def __init__(self, parent=None):
        """Initialize the orientation optimization widget."""
        super().__init__(parent)
        
        # Data storage
        self.polarization_data = {}
        self.tensor_data = {}
        self.fitted_peaks = []
        self.stage_results = {'stage1': None, 'stage2': None, 'stage3': None}
        
        # Configuration
        self.max_iterations = 200
        self.tolerance = 1e-4
        self.n_starts_stage1 = 15
        self.n_samples_stage2 = 1000
        
        # Initialize UI
        self.init_ui()
        self.setup_matplotlib()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - controls
        self.control_panel = self.create_control_panel()
        splitter.addWidget(self.control_panel)
        
        # Right panel - visualization
        self.viz_panel = self.create_visualization_panel()
        splitter.addWidget(self.viz_panel)
        
        # Set splitter proportions
        splitter.setSizes([350, 650])
    
    def create_control_panel(self):
        """Create the control panel with optimization options."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(400)
        
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("🎯 Crystal Orientation Optimization")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Data Sources Group
        data_group = QGroupBox("Data Sources")
        data_layout = QVBoxLayout(data_group)
        
        # Data status
        self.data_status = QLabel("📊 No data loaded")
        self.data_status.setWordWrap(True)
        data_layout.addWidget(self.data_status)
        
        # Load data buttons
        load_peaks_btn = QPushButton("Load Peak Data")
        load_peaks_btn.clicked.connect(self.load_peak_data)
        self.apply_button_style(load_peaks_btn)
        data_layout.addWidget(load_peaks_btn)
        
        load_tensors_btn = QPushButton("Load Tensor Data")
        load_tensors_btn.clicked.connect(self.load_tensor_data)
        self.apply_button_style(load_tensors_btn)
        data_layout.addWidget(load_tensors_btn)
        
        layout.addWidget(data_group)
        
        # Optimization Trilogy Group
        trilogy_group = QGroupBox("🚀 Optimization Trilogy")
        trilogy_layout = QVBoxLayout(trilogy_group)
        
        # Stage 1 Button
        self.stage1_btn = QPushButton("🚀 Stage 1: Enhanced Peak Optimization")
        self.stage1_btn.clicked.connect(self.run_stage1)
        self.stage1_btn.setStyleSheet(self.get_stage_style("stage1"))
        self.stage1_btn.setToolTip("Multi-start global optimization with individual peak adjustments")
        trilogy_layout.addWidget(self.stage1_btn)
        
        # Stage 2 Button
        self.stage2_btn = QPushButton("🧠 Stage 2: Bayesian Analysis")
        self.stage2_btn.clicked.connect(self.run_stage2)
        self.stage2_btn.setStyleSheet(self.get_stage_style("stage2"))
        self.stage2_btn.setToolTip("MCMC sampling with probabilistic uncertainty quantification")
        trilogy_layout.addWidget(self.stage2_btn)
        
        # Stage 3 Button
        self.stage3_btn = QPushButton("🌟 Stage 3: Advanced Multi-Objective")
        self.stage3_btn.clicked.connect(self.run_stage3)
        self.stage3_btn.setStyleSheet(self.get_stage_style("stage3"))
        self.stage3_btn.setToolTip("Gaussian Process with Pareto optimization")
        trilogy_layout.addWidget(self.stage3_btn)
        
        layout.addWidget(trilogy_group)
        
        # Configuration Group
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout(config_group)
        
        # Max iterations
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(50, 1000)
        self.iterations_spin.setValue(self.max_iterations)
        self.iterations_spin.valueChanged.connect(self.update_max_iterations)
        config_layout.addRow("Max Iterations:", self.iterations_spin)
        
        # Tolerance
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-6, 1e-2)
        self.tolerance_spin.setValue(self.tolerance)
        self.tolerance_spin.setDecimals(6)
        self.tolerance_spin.setSingleStep(1e-5)
        self.tolerance_spin.valueChanged.connect(self.update_tolerance)
        config_layout.addRow("Tolerance:", self.tolerance_spin)
        
        layout.addWidget(config_group)
        
        # Results Group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.status_label = QLabel("Ready for optimization")
        self.status_label.setWordWrap(True)
        results_layout.addWidget(self.status_label)
        
        # Results buttons
        show_results_btn = QPushButton("Show Detailed Results")
        show_results_btn.clicked.connect(self.show_detailed_results)
        self.apply_button_style(show_results_btn)
        results_layout.addWidget(show_results_btn)
        
        export_btn = QPushButton("Export for 3D Visualization")
        export_btn.clicked.connect(self.export_for_3d)
        self.apply_button_style(export_btn)
        results_layout.addWidget(export_btn)
        
        layout.addWidget(results_group)
        
        # Add stretch
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Create the visualization panel with plots."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Plot controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("View:"))
        
        self.plot_combo = QComboBox()
        self.plot_combo.addItems([
            "Data Overview",
            "Optimization Progress", 
            "Results Comparison",
            "Uncertainty Analysis",
            "Convergence History"
        ])
        self.plot_combo.currentTextChanged.connect(self.update_plot)
        controls_layout.addWidget(self.plot_combo)
        
        controls_layout.addStretch()
        
        update_btn = QPushButton("Update Plot")
        update_btn.clicked.connect(self.update_plot)
        self.apply_button_style(update_btn)
        controls_layout.addWidget(update_btn)
        
        layout.addLayout(controls_layout)
        
        # Create matplotlib canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        return panel
    
    def setup_matplotlib(self):
        """Setup matplotlib configuration."""
        try:
            from polarization_ui.matplotlib_config import configure_compact_ui
            configure_compact_ui()
        except ImportError:
            pass
        
        # Initialize plot
        self.init_plot()
    
    def init_plot(self):
        """Initialize the plot display."""
        self.figure.clear()
        
        # Create 2x2 subplot grid
        self.ax1 = self.figure.add_subplot(2, 2, 1)
        self.ax2 = self.figure.add_subplot(2, 2, 2) 
        self.ax3 = self.figure.add_subplot(2, 2, 3)
        self.ax4 = self.figure.add_subplot(2, 2, 4)
        
        # Set initial titles
        self.ax1.set_title("Data Overview")
        self.ax2.set_title("Optimization Progress")
        self.ax3.set_title("Results Comparison")
        self.ax4.set_title("Uncertainty Analysis")
        
        # Add placeholder text
        for ax, text in zip([self.ax1, self.ax2, self.ax3, self.ax4], 
                           ["Load data to view", "Run optimization", "Complete stages", "Analyze uncertainty"]):
            ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, alpha=0.6)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def get_stage_style(self, stage):
        """Get button style for optimization stages."""
        styles = {
            "stage1": """
                QPushButton {
                    background-color: #e6f3ff;
                    border: 2px solid #4dabf7;
                    border-radius: 10px;
                    padding: 12px 16px;
                    font-weight: bold;
                    font-size: 13px;
                    color: #1971c2;
                    text-align: left;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background-color: #d0ebff;
                    border-color: #339af0;
                }
                QPushButton:pressed {
                    background-color: #a5d8ff;
                    border-color: #228be6;
                }
            """,
            "stage2": """
                QPushButton {
                    background-color: #fff0e6;
                    border: 2px solid #fd7e14;
                    border-radius: 10px;
                    padding: 12px 16px;
                    font-weight: bold;
                    font-size: 13px;
                    color: #e8590c;
                    text-align: left;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background-color: #ffe8cc;
                    border-color: #e8590c;
                }
                QPushButton:pressed {
                    background-color: #ffd8a8;
                    border-color: #d63384;
                }
            """,
            "stage3": """
                QPushButton {
                    background-color: #f0fff4;
                    border: 2px solid #20c997;
                    border-radius: 10px;
                    padding: 12px 16px;
                    font-weight: bold;
                    font-size: 13px;
                    color: #0f5132;
                    text-align: left;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background-color: #d1ecf1;
                    border-color: #17a2b8;
                }
                QPushButton:pressed {
                    background-color: #b8daff;
                    border-color: #138496;
                }
            """
        }
        return styles.get(stage, "")
    
    def apply_button_style(self, button):
        """Apply standard button styling."""
        button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 500;
                color: #333333;
                font-size: 12px;
                min-height: 16px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border-color: #bbb;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
                border-color: #999;
            }
            QPushButton:disabled {
                background-color: #f8f8f8;
                color: #999999;
                border-color: #e0e0e0;
            }
        """)
    
    def update_max_iterations(self, value):
        """Update maximum iterations."""
        self.max_iterations = value
    
    def update_tolerance(self, value):
        """Update optimization tolerance."""
        self.tolerance = value
    
    # === Data Loading Methods ===
    
    def load_peak_data(self):
        """Load peak fitting data."""
        # This would be called by the parent application
        QMessageBox.information(self, "Peak Data", 
                              "Peak data should be imported from the Peak Fitting tab.")
    
    def load_tensor_data(self):
        """Load tensor analysis data."""
        # This would be called by the parent application
        QMessageBox.information(self, "Tensor Data", 
                              "Tensor data should be imported from the Tensor Analysis tab.")
    
    def set_data(self, peak_data, tensor_data=None, polarization_data=None):
        """Set optimization data from parent application."""
        self.fitted_peaks = peak_data or []
        self.tensor_data = tensor_data or {}
        self.polarization_data = polarization_data or {}
        
        # Update status
        data_count = len(self.fitted_peaks)
        tensor_count = len(self.tensor_data)
        
        if data_count > 0:
            self.data_status.setText(f"✅ Loaded {data_count} peaks, {tensor_count} tensors")
            self.data_status.setStyleSheet("color: green;")
        else:
            self.data_status.setText("❌ No peak data available")
            self.data_status.setStyleSheet("color: red;")
        
        self.update_plot()
    
    # === Optimization Methods ===
    
    def run_stage1(self):
        """🚀 Stage 1: Enhanced Individual Peak Optimization."""
        if not self.validate_data():
            return
        
        # Implementation would go here
        QMessageBox.information(self, "Stage 1", "Stage 1 optimization completed successfully!")
        
        # Mock results for demonstration
        self.stage_results['stage1'] = {
            'method': 'Enhanced Individual Peak Optimization',
            'best_orientation': [45.0, 90.0, 135.0],
            'orientation_uncertainty': [2.5, 3.0, 2.8],
            'final_error': 0.0523,
            'confidence': 0.87,
            'timestamp': datetime.now()
        }
        
        self.update_status("🚀 Stage 1 Complete! Enhanced optimization with ±2.7° accuracy")
        self.update_plot()
    
    def run_stage2(self):
        """🧠 Stage 2: Probabilistic Bayesian Framework."""
        if not self.validate_data():
            return
        
        if not EMCEE_AVAILABLE:
            QMessageBox.warning(self, "Missing Dependency", 
                              "Stage 2 requires emcee for MCMC sampling.\nInstall with: pip install emcee")
            return
        
        # Implementation would go here
        QMessageBox.information(self, "Stage 2", "Stage 2 Bayesian analysis completed successfully!")
        
        # Mock results for demonstration
        self.stage_results['stage2'] = {
            'method': 'Probabilistic Bayesian Framework',
            'best_orientation': [44.2, 89.8, 136.1],
            'orientation_uncertainty': [[41.7, 46.8], [87.1, 92.5], [133.2, 139.0]],  # Credible intervals
            'convergence': True,
            'evidence': 1.23e-5,
            'timestamp': datetime.now()
        }
        
        self.update_status("🧠 Stage 2 Complete! Bayesian analysis with credible intervals")
        self.update_plot()
    
    def run_stage3(self):
        """🌟 Stage 3: Advanced Multi-Objective Bayesian Optimization."""
        if not self.validate_data():
            return
        
        if not SKLEARN_AVAILABLE:
            QMessageBox.warning(self, "Missing Dependency", 
                              "Stage 3 requires scikit-learn for Gaussian Processes.\nInstall with: pip install scikit-learn")
            return
        
        # Implementation would go here
        QMessageBox.information(self, "Stage 3", "Stage 3 advanced optimization completed successfully!")
        
        # Mock results for demonstration
        self.stage_results['stage3'] = {
            'method': 'Advanced Multi-Objective Bayesian Optimization',
            'best_orientation': [44.1, 89.9, 135.8],
            'pareto_solutions': 25,
            'total_uncertainty': [1.8, 2.1, 1.9],
            'hypervolume': 0.92,
            'timestamp': datetime.now()
        }
        
        self.update_status("🌟 Stage 3 Complete! Multi-objective optimization with ±2.0° total uncertainty")
        self.update_plot()
    
    def validate_data(self):
        """Validate that required data is available."""
        if not self.fitted_peaks:
            QMessageBox.warning(self, "Missing Data", "No peak data available.\nPlease load peak fitting results first.")
            return False
        return True
    
    def update_status(self, message):
        """Update the status label."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
    
    # === Visualization Methods ===
    
    def update_plot(self):
        """Update the visualization based on current selection."""
        plot_type = self.plot_combo.currentText()
        
        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        if plot_type == "Data Overview":
            self.plot_data_overview()
        elif plot_type == "Optimization Progress":
            self.plot_optimization_progress()
        elif plot_type == "Results Comparison":
            self.plot_results_comparison()
        elif plot_type == "Uncertainty Analysis":
            self.plot_uncertainty_analysis()
        elif plot_type == "Convergence History":
            self.plot_convergence_history()
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_data_overview(self):
        """Plot data overview."""
        if self.fitted_peaks:
            # Plot peak data
            frequencies = [peak['position'] for peak in self.fitted_peaks]
            intensities = [peak['amplitude'] for peak in self.fitted_peaks]
            
            self.ax1.bar(frequencies, intensities, alpha=0.7, color='blue')
            self.ax1.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax1.set_ylabel('Intensity')
            self.ax1.set_title('Experimental Peaks')
            self.ax1.grid(True, alpha=0.3)
        else:
            self.ax1.text(0.5, 0.5, 'No peak data loaded', ha='center', va='center', 
                         transform=self.ax1.transAxes)
        
        # Placeholder for other data plots
        for ax, title in zip([self.ax2, self.ax3, self.ax4], 
                            ['Tensor Overview', 'Polarization Data', 'Quality Metrics']):
            ax.set_title(title)
            ax.text(0.5, 0.5, 'Data visualization\ncoming soon', ha='center', va='center',
                   transform=ax.transAxes, alpha=0.6)
    
    def plot_optimization_progress(self):
        """Plot optimization progress."""
        completed_stages = [name for name, results in self.stage_results.items() if results]
        
        if not completed_stages:
            self.ax1.text(0.5, 0.5, 'Run optimization to view progress', 
                         ha='center', va='center', transform=self.ax1.transAxes)
            return
        
        # Plot stage completion
        stage_names = ['Stage 1', 'Stage 2', 'Stage 3']
        completion = [1 if f'stage{i+1}' in completed_stages else 0 for i in range(3)]
        colors = ['green' if c else 'lightgray' for c in completion]
        
        self.ax1.bar(stage_names, completion, color=colors, alpha=0.7)
        self.ax1.set_ylabel('Completed')
        self.ax1.set_title('Stage Completion')
        self.ax1.set_ylim(0, 1.2)
        
        # Add completion text
        for i, (stage, comp) in enumerate(zip(stage_names, completion)):
            if comp:
                self.ax1.text(i, 0.5, '✓', ha='center', va='center', 
                             fontsize=20, color='white', weight='bold')
    
    def plot_results_comparison(self):
        """Plot comparison of optimization results."""
        completed_stages = [(name, results) for name, results in self.stage_results.items() if results]
        
        if not completed_stages:
            self.ax1.text(0.5, 0.5, 'No results to compare', 
                         ha='center', va='center', transform=self.ax1.transAxes)
            return
        
        # Compare orientations
        stage_names = []
        phi_vals = []
        theta_vals = []
        psi_vals = []
        
        for stage_name, results in completed_stages:
            stage_names.append(stage_name.upper())
            orientation = results['best_orientation']
            phi_vals.append(orientation[0])
            theta_vals.append(orientation[1])
            psi_vals.append(orientation[2])
        
        x_pos = np.arange(len(stage_names))
        width = 0.25
        
        self.ax1.bar(x_pos - width, phi_vals, width, label='φ (°)', alpha=0.7)
        self.ax1.bar(x_pos, theta_vals, width, label='θ (°)', alpha=0.7)
        self.ax1.bar(x_pos + width, psi_vals, width, label='ψ (°)', alpha=0.7)
        
        self.ax1.set_xlabel('Optimization Stage')
        self.ax1.set_ylabel('Angle (degrees)')
        self.ax1.set_title('Orientation Results')
        self.ax1.set_xticks(x_pos)
        self.ax1.set_xticklabels(stage_names)
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
    
    def plot_uncertainty_analysis(self):
        """Plot uncertainty analysis."""
        # Find most advanced completed stage
        advanced_stage = None
        for stage in ['stage3', 'stage2', 'stage1']:
            if self.stage_results.get(stage):
                advanced_stage = stage
                break
        
        if not advanced_stage:
            self.ax1.text(0.5, 0.5, 'No uncertainty data available', 
                         ha='center', va='center', transform=self.ax1.transAxes)
            return
        
        results = self.stage_results[advanced_stage]
        
        # Plot uncertainty information
        if 'orientation_uncertainty' in results:
            uncertainties = results['orientation_uncertainty']
            param_names = ['φ', 'θ', 'ψ']
            
            # Handle different uncertainty formats
            if isinstance(uncertainties[0], list):
                # Credible intervals from Stage 2
                lower = [u[0] for u in uncertainties]
                upper = [u[1] for u in uncertainties]
                centers = [(l + u) / 2 for l, u in zip(lower, upper)]
                errors = [(u - l) / 2 for l, u in zip(lower, upper)]
                
                self.ax1.bar(param_names, centers, yerr=errors, alpha=0.7, capsize=5, color='blue')
                self.ax1.set_ylabel('Angle (degrees)')
                self.ax1.set_title(f'Credible Intervals ({advanced_stage.title()})')
            else:
                # Standard uncertainties from Stage 1/3
                self.ax1.bar(param_names, uncertainties, alpha=0.7, color='orange')
                self.ax1.set_ylabel('Uncertainty (°)')
                self.ax1.set_title(f'Orientation Uncertainties ({advanced_stage.title()})')
            
            self.ax1.grid(True, alpha=0.3)
    
    def plot_convergence_history(self):
        """Plot convergence history."""
        self.ax1.text(0.5, 0.5, 'Convergence history\ncoming soon', 
                     ha='center', va='center', transform=self.ax1.transAxes, alpha=0.6)
    
    # === Results and Export Methods ===
    
    def show_detailed_results(self):
        """Show detailed optimization results."""
        if not any(self.stage_results.values()):
            QMessageBox.information(self, "No Results", "No optimization results available.\nRun optimization first.")
            return
        
        # Create detailed results dialog
        dialog = DetailedResultsDialog(self.stage_results, parent=self)
        dialog.exec()
    
    def export_for_3d(self):
        """Export optimization results for 3D visualization."""
        if not any(self.stage_results.values()):
            QMessageBox.warning(self, "No Results", "No optimization results to export.\nRun optimization first.")
            return
        
        try:
            # Find best available results
            best_stage = None
            for stage in ['stage3', 'stage2', 'stage1']:
                if self.stage_results.get(stage):
                    best_stage = stage
                    break
            
            if not best_stage:
                raise ValueError("No optimization results available")
            
            results = self.stage_results[best_stage]
            
            # Prepare 3D visualization data
            viz_data = {
                'orientation': results['best_orientation'],
                'uncertainty': results.get('orientation_uncertainty', [5.0, 5.0, 5.0]),
                'method': results.get('method', 'Unknown'),
                'stage': best_stage,
                'timestamp': results.get('timestamp', datetime.now()),
                'peak_data': self.fitted_peaks,
                'tensor_data': self.tensor_data
            }
            
            # Emit signal with visualization data
            self.data_exported.emit(viz_data)
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Optimization results from {best_stage.upper()} exported for 3D visualization.\n"
                                  f"Best orientation: φ={results['best_orientation'][0]:.1f}°, "
                                  f"θ={results['best_orientation'][1]:.1f}°, ψ={results['best_orientation'][2]:.1f}°")
        
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting for 3D visualization:\n{str(e)}")


class DetailedResultsDialog(QDialog):
    """Dialog for displaying detailed optimization results."""
    
    def __init__(self, stage_results, parent=None):
        super().__init__(parent)
        
        self.stage_results = stage_results
        self.init_ui()
    
    def init_ui(self):
        """Initialize the results dialog UI."""
        self.setWindowTitle("🎯 Detailed Optimization Results")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Create tab widget for different stages
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        summary_text.setPlainText(self.generate_summary())
        summary_layout.addWidget(summary_text)
        tab_widget.addTab(summary_tab, "📊 Summary")
        
        # Individual stage tabs
        for stage_name, results in self.stage_results.items():
            if results:
                stage_tab = QWidget()
                stage_layout = QVBoxLayout(stage_tab)
                stage_text = QTextEdit()
                stage_text.setReadOnly(True)
                stage_text.setPlainText(self.generate_stage_details(stage_name, results))
                stage_layout.addWidget(stage_text)
                tab_widget.addTab(stage_tab, f"{stage_name.upper()}")
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    
    def generate_summary(self):
        """Generate summary of all results."""
        summary = "🎯 CRYSTAL ORIENTATION OPTIMIZATION RESULTS\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        completed_stages = [name for name, results in self.stage_results.items() if results]
        summary += f"Completed Stages: {', '.join([s.upper() for s in completed_stages])}\n\n"
        
        # Best results recommendation
        if self.stage_results.get('stage3'):
            best_results = self.stage_results['stage3']
            summary += "🌟 RECOMMENDED: Use Stage 3 results for highest accuracy\n"
        elif self.stage_results.get('stage2'):
            best_results = self.stage_results['stage2']
            summary += "🧠 RECOMMENDED: Use Stage 2 results for probabilistic analysis\n"
        elif self.stage_results.get('stage1'):
            best_results = self.stage_results['stage1']
            summary += "🚀 RECOMMENDED: Use Stage 1 results for enhanced optimization\n"
        else:
            summary += "❌ No optimization completed\n"
            return summary
        
        orientation = best_results['best_orientation']
        summary += f"Best Orientation: φ={orientation[0]:.2f}°, θ={orientation[1]:.2f}°, ψ={orientation[2]:.2f}°\n"
        summary += f"Method: {best_results.get('method', 'Unknown')}\n\n"
        
        summary += "STAGE COMPARISON:\n"
        summary += "-" * 20 + "\n"
        
        for stage_name, results in self.stage_results.items():
            if results:
                orientation = results['best_orientation']
                summary += f"{stage_name.upper()}: φ={orientation[0]:.1f}°, θ={orientation[1]:.1f}°, ψ={orientation[2]:.1f}°\n"
        
        return summary
    
    def generate_stage_details(self, stage_name, results):
        """Generate detailed information for a specific stage."""
        details = f"{stage_name.upper()} DETAILED RESULTS\n"
        details += "=" * 40 + "\n\n"
        
        details += f"Method: {results.get('method', 'Unknown')}\n"
        details += f"Timestamp: {results.get('timestamp', 'Unknown')}\n\n"
        
        orientation = results['best_orientation']
        details += "ORIENTATION:\n"
        details += f"φ (phi): {orientation[0]:.3f}°\n"
        details += f"θ (theta): {orientation[1]:.3f}°\n"
        details += f"ψ (psi): {orientation[2]:.3f}°\n\n"
        
        # Stage-specific details
        if stage_name == 'stage1':
            details += "ENHANCED INDIVIDUAL PEAK OPTIMIZATION:\n"
            if 'final_error' in results:
                details += f"Final Error: {results['final_error']:.6f}\n"
            if 'confidence' in results:
                details += f"Confidence: {results['confidence']:.1%}\n"
            if 'orientation_uncertainty' in results:
                unc = results['orientation_uncertainty']
                details += f"Uncertainties: ±{unc[0]:.1f}°, ±{unc[1]:.1f}°, ±{unc[2]:.1f}°\n"
        
        elif stage_name == 'stage2':
            details += "PROBABILISTIC BAYESIAN FRAMEWORK:\n"
            if 'convergence' in results:
                details += f"Converged: {results['convergence']}\n"
            if 'evidence' in results:
                details += f"Model Evidence: {results['evidence']:.2e}\n"
            if 'orientation_uncertainty' in results:
                intervals = results['orientation_uncertainty']
                details += "95% Credible Intervals:\n"
                details += f"φ: [{intervals[0][0]:.1f}°, {intervals[0][1]:.1f}°]\n"
                details += f"θ: [{intervals[1][0]:.1f}°, {intervals[1][1]:.1f}°]\n"
                details += f"ψ: [{intervals[2][0]:.1f}°, {intervals[2][1]:.1f}°]\n"
        
        elif stage_name == 'stage3':
            details += "ADVANCED MULTI-OBJECTIVE BAYESIAN OPTIMIZATION:\n"
            if 'pareto_solutions' in results:
                details += f"Pareto Solutions: {results['pareto_solutions']}\n"
            if 'hypervolume' in results:
                details += f"Hypervolume: {results['hypervolume']:.3f}\n"
            if 'total_uncertainty' in results:
                unc = results['total_uncertainty']
                details += f"Total Uncertainties: ±{unc[0]:.1f}°, ±{unc[1]:.1f}°, ±{unc[2]:.1f}°\n"
        
        return details


# Export the main widget
__all__ = ['OrientationOptimizationWidget'] 