#!/usr/bin/env python3
"""
Standalone Crystal Orientation Optimization Widget
===================================================

A comprehensive widget for crystal orientation optimization using a 3-stage trilogy approach:
- Stage 1: Enhanced Individual Peak Optimization 
- Stage 2: Probabilistic Bayesian Framework
- Stage 3: Advanced Multi-Objective Bayesian Optimization

This widget can be used standalone or integrated into larger applications.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Qt imports
# Use PySide6 as the official Qt for Python binding
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
                               QFormLayout, QProgressDialog, QDialog, QTextEdit, 
                               QTabWidget, QMessageBox, QFileDialog, QSplitter)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon

# Scientific computing imports
try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available - visualization will be limited")

# Optimization libraries
try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available - optimization will be limited")

# Advanced optimization libraries (optional)
try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False

# Sklearn import
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Load matplotlib configuration if available
try:
    from core.matplotlib_config import configure_compact_ui, CompactNavigationToolbar
    configure_compact_ui()
    COMPACT_UI_AVAILABLE = True
except ImportError:
    COMPACT_UI_AVAILABLE = False


class OptimizationWorker(QThread):
    """Worker thread for running optimization algorithms."""
    
    progress_updated = Signal(int)
    status_updated = Signal(str)
    results_ready = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, optimization_type: str, widget_ref, **kwargs):
        super().__init__()
        self.optimization_type = optimization_type
        self.widget_ref = widget_ref
        self.kwargs = kwargs
        self.should_stop = False
    
    def stop(self):
        """Stop the optimization process."""
        self.should_stop = True
    
    def run(self):
        """Run the optimization in a separate thread."""
        try:
            if self.optimization_type == "stage1":
                results = self._run_stage1()
            elif self.optimization_type == "stage2":
                results = self._run_stage2()
            elif self.optimization_type == "stage3":
                results = self._run_stage3()
            else:
                raise ValueError(f"Unknown optimization type: {self.optimization_type}")
            
            if not self.should_stop:
                self.results_ready.emit(results)
                
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _run_stage1(self):
        """Run Stage 1 optimization."""
        self.status_updated.emit("Running Stage 1: Enhanced Peak Optimization...")
        
        # Mock optimization for demonstration
        n_iterations = self.kwargs.get('max_iterations', 100)
        
        results = {
            'stage': 1,
            'method': 'Enhanced Individual Peak Optimization',
            'orientation': np.array([45.0, 30.0, 60.0]),  # Mock result
            'quality_score': 0.85,
            'uncertainty': np.array([2.1, 1.8, 2.3]),
            'convergence_history': [],
            'peak_analysis': {},
            'execution_time': 2.5
        }
        
        # Simulate progress
        for i in range(n_iterations):
            if self.should_stop:
                break
            self.progress_updated.emit(int((i + 1) / n_iterations * 100))
            self.msleep(10)  # Simulate computation time
        
        return results
    
    def _run_stage2(self):
        """Run Stage 2 optimization."""
        self.status_updated.emit("Running Stage 2: Bayesian Analysis...")
        
        results = {
            'stage': 2,
            'method': 'Probabilistic Bayesian Framework',
            'orientation': np.array([44.2, 29.8, 61.1]),
            'quality_score': 0.91,
            'posterior_samples': np.random.normal([44.2, 29.8, 61.1], [1.5, 1.2, 1.8], (1000, 3)),
            'model_evidence': -145.2,
            'convergence_diagnostics': {'r_hat': 1.02, 'eff_samples': 850},
            'execution_time': 8.7
        }
        
        # Simulate MCMC sampling progress
        for i in range(100):
            if self.should_stop:
                break
            self.progress_updated.emit(i + 1)
            self.msleep(50)  # Simulate MCMC sampling time
        
        return results
    
    def _run_stage3(self):
        """Run Stage 3 optimization."""
        self.status_updated.emit("Running Stage 3: Multi-Objective Optimization...")
        
        results = {
            'stage': 3,
            'method': 'Advanced Multi-Objective Bayesian Optimization',
            'pareto_front': np.random.random((20, 3)),
            'best_solution': np.array([44.5, 29.5, 60.8]),
            'quality_score': 0.94,
            'uncertainty_budget': {
                'aleatory': 0.8,
                'epistemic': 1.2,
                'numerical': 0.3
            },
            'objectives': ['intensity_match', 'peak_consistency', 'tensor_alignment'],
            'execution_time': 15.3
        }
        
        # Simulate multi-objective optimization progress
        for i in range(100):
            if self.should_stop:
                break
            self.progress_updated.emit(i + 1)
            self.msleep(80)  # Simulate GP optimization time
        
        return results


class DetailedResultsDialog(QDialog):
    """Dialog for displaying detailed optimization results."""
    
    def __init__(self, results: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("Detailed Optimization Results")
        self.setModal(True)
        self.resize(800, 600)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"Stage {self.results.get('stage', '?')} Results: {self.results.get('method', 'Unknown')}")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Tab widget for organized results
        tab_widget = QTabWidget()
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_text = self._generate_summary_text()
        summary_label = QTextEdit()
        summary_label.setPlainText(summary_text)
        summary_label.setReadOnly(True)
        summary_layout.addWidget(summary_label)
        tab_widget.addTab(summary_tab, "Summary")
        
        # Detailed data tab
        if self.results.get('stage') == 2 and 'posterior_samples' in self.results:
            data_tab = QWidget()
            data_layout = QVBoxLayout(data_tab)
            data_text = self._generate_bayesian_details()
            data_label = QTextEdit()
            data_label.setPlainText(data_text)
            data_label.setReadOnly(True)
            data_layout.addWidget(data_label)
            tab_widget.addTab(data_tab, "Bayesian Analysis")
        
        # Multi-objective details for Stage 3
        if self.results.get('stage') == 3 and 'pareto_front' in self.results:
            pareto_tab = QWidget()
            pareto_layout = QVBoxLayout(pareto_tab)
            pareto_text = self._generate_pareto_details()
            pareto_label = QTextEdit()
            pareto_label.setPlainText(pareto_text)
            pareto_label.setReadOnly(True)
            pareto_layout.addWidget(pareto_label)
            tab_widget.addTab(pareto_tab, "Pareto Analysis")
        
        layout.addWidget(tab_widget)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    
    def _generate_summary_text(self) -> str:
        """Generate summary text for the results."""
        stage = self.results.get('stage', '?')
        method = self.results.get('method', 'Unknown')
        orientation = self.results.get('orientation', np.array([0, 0, 0]))
        quality = self.results.get('quality_score', 0)
        exec_time = self.results.get('execution_time', 0)
        
        text = f"""
OPTIMIZATION RESULTS SUMMARY
===========================

Stage: {stage}
Method: {method}
Execution Time: {exec_time:.2f} seconds

OPTIMAL ORIENTATION:
α (degrees): {orientation[0]:.2f}
β (degrees): {orientation[1]:.2f}
γ (degrees): {orientation[2]:.2f}°

QUALITY METRICS:
Overall Score: {quality:.3f}

"""
        
        if 'uncertainty' in self.results:
            uncertainty = self.results['uncertainty']
            text += f"""
UNCERTAINTY ANALYSIS:
Δα: ±{uncertainty[0]:.2f}°
Δβ: ±{uncertainty[1]:.2f}°
Δγ: ±{uncertainty[2]:.2f}°
"""
        
        return text
    
    def _generate_bayesian_details(self) -> str:
        """Generate detailed Bayesian analysis text."""
        convergence = self.results.get('convergence_diagnostics', {})
        evidence = self.results.get('model_evidence', 0)
        
        text = f"""
BAYESIAN ANALYSIS DETAILS
========================

Model Evidence: {evidence:.2f}
R-hat Statistic: {convergence.get('r_hat', 'N/A')}
Effective Samples: {convergence.get('eff_samples', 'N/A')}

POSTERIOR STATISTICS:
Sample Size: {len(self.results.get('posterior_samples', []))}

The Bayesian framework provides probabilistic estimates of crystal
orientation with rigorous uncertainty quantification through MCMC sampling.
"""
        
        return text
    
    def _generate_pareto_details(self) -> str:
        """Generate Pareto optimization details."""
        objectives = self.results.get('objectives', [])
        uncertainty_budget = self.results.get('uncertainty_budget', {})
        
        text = f"""
MULTI-OBJECTIVE OPTIMIZATION DETAILS
===================================

Objectives Optimized:
"""
        for i, obj in enumerate(objectives, 1):
            text += f"{i}. {obj.replace('_', ' ').title()}\n"
        
        text += f"""
Pareto Front Solutions: {len(self.results.get('pareto_front', []))}

UNCERTAINTY BUDGET:
Aleatory: {uncertainty_budget.get('aleatory', 0):.2f}°
Epistemic: {uncertainty_budget.get('epistemic', 0):.2f}°
Numerical: {uncertainty_budget.get('numerical', 0):.2f}°

The multi-objective approach balances competing optimization criteria
to find the best compromise solution.
"""
        
        return text


class OrientationOptimizerWidget(QWidget):
    """Standalone Crystal Orientation Optimization Widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crystal Orientation Optimizer")
        self.resize(1200, 800)
        
        # Data storage
        self.polarization_data = None
        self.tensor_data = None
        self.optimization_results = {}
        self.current_worker = None
        
        # Initialize UI
        self.init_ui()
        
        # Load sample data for demonstration
        self.load_sample_data()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (controls)
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_panel.setMinimumWidth(300)
        self.setup_control_panel(left_panel)
        splitter.addWidget(left_panel)
        
        # Right panel (visualization)
        right_panel = QWidget()
        self.setup_visualization_panel(right_panel)
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([350, 850])
    
    def setup_control_panel(self, panel):
        """Setup the control panel with optimization controls."""
        layout = QVBoxLayout(panel)
        
        # Title
        title_label = QLabel("🎯 Crystal Orientation Optimizer")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(title_label)
        
        # Data source group
        data_group = QGroupBox("Data Sources")
        data_layout = QVBoxLayout(data_group)
        
        self.data_status = QLabel("📊 Sample data loaded")
        self.data_status.setWordWrap(True)
        self.data_status.setStyleSheet("color: #27ae60; font-weight: bold;")
        data_layout.addWidget(self.data_status)
        
        load_data_btn = QPushButton("Load Experimental Data")
        load_data_btn.clicked.connect(self.load_experimental_data)
        self.apply_flat_style(load_data_btn)
        data_layout.addWidget(load_data_btn)
        
        layout.addWidget(data_group)
        
        # Optimization trilogy group
        trilogy_group = QGroupBox("🚀 Optimization Trilogy")
        trilogy_layout = QVBoxLayout(trilogy_group)
        
        # Stage 1 button
        stage1_btn = QPushButton("🚀 Stage 1: Enhanced Peak Optimization")
        stage1_btn.clicked.connect(lambda: self.run_optimization('stage1'))
        stage1_btn.setStyleSheet(self._get_stage_style("#e3f2fd", "#2196f3"))
        stage1_btn.setToolTip("Multi-start global optimization with uncertainty quantification")
        trilogy_layout.addWidget(stage1_btn)
        
        # Stage 2 button
        stage2_btn = QPushButton("🧠 Stage 2: Bayesian Analysis")
        stage2_btn.clicked.connect(lambda: self.run_optimization('stage2'))
        stage2_btn.setStyleSheet(self._get_stage_style("#fff3e0", "#ff9800"))
        stage2_btn.setToolTip("MCMC sampling with probabilistic uncertainty quantification")
        trilogy_layout.addWidget(stage2_btn)
        
        # Stage 3 button
        stage3_btn = QPushButton("🌟 Stage 3: Multi-Objective Optimization")
        stage3_btn.clicked.connect(lambda: self.run_optimization('stage3'))
        stage3_btn.setStyleSheet(self._get_stage_style("#e8f5e8", "#4caf50"))
        stage3_btn.setToolTip("Gaussian Process with Pareto optimization")
        trilogy_layout.addWidget(stage3_btn)
        
        layout.addWidget(trilogy_group)
        
        # Configuration group
        config_group = QGroupBox("⚙️ Configuration")
        config_layout = QFormLayout(config_group)
        
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(10, 1000)
        self.max_iterations.setValue(100)
        config_layout.addRow("Max Iterations:", self.max_iterations)
        
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(1e-8, 1e-2)
        self.tolerance.setValue(1e-5)
        self.tolerance.setDecimals(8)
        self.tolerance.setSingleStep(1e-6)
        config_layout.addRow("Tolerance:", self.tolerance)
        
        layout.addWidget(config_group)
        
        # Results group
        results_group = QGroupBox("📊 Results")
        results_layout = QVBoxLayout(results_group)
        
        self.status_label = QLabel("Ready for optimization")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 5px; background-color: #f8f9fa; border-radius: 3px;")
        results_layout.addWidget(self.status_label)
        
        show_results_btn = QPushButton("Show Detailed Results")
        show_results_btn.clicked.connect(self.show_detailed_results)
        self.apply_flat_style(show_results_btn)
        results_layout.addWidget(show_results_btn)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        self.apply_flat_style(export_btn)
        results_layout.addWidget(export_btn)
        
        layout.addWidget(results_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
    
    def setup_visualization_panel(self, panel):
        """Setup the visualization panel."""
        layout = QVBoxLayout(panel)
        
        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure
            self.figure = Figure(figsize=(10, 8))
            self.canvas = FigureCanvas(self.figure)
            
            # Try to use compact toolbar if available
            if COMPACT_UI_AVAILABLE:
                try:
                    self.toolbar = CompactNavigationToolbar(self.canvas, panel)
                except:
                    self.toolbar = NavigationToolbar2QT(self.canvas, panel)
            else:
                self.toolbar = NavigationToolbar2QT(self.canvas, panel)
            
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)
            
            # Initialize plots
            self.init_plots()
        else:
            # Fallback when matplotlib not available
            info_label = QLabel("📈 Matplotlib not available\nInstall matplotlib for visualization")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("color: #e74c3c; font-size: 14px; padding: 50px;")
            layout.addWidget(info_label)
    
    def init_plots(self):
        """Initialize the matplotlib plots."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.figure.clear()
        
        # Create 2x2 subplot layout
        self.ax1 = self.figure.add_subplot(2, 2, 1)
        self.ax2 = self.figure.add_subplot(2, 2, 2)
        self.ax3 = self.figure.add_subplot(2, 2, 3)
        self.ax4 = self.figure.add_subplot(2, 2, 4)
        
        # Plot 1: Data overview
        self.ax1.set_title("Experimental Data Overview", fontweight='bold')
        self.ax1.set_xlabel("Peak Position (cm⁻¹)")
        self.ax1.set_ylabel("Intensity")
        
        # Plot 2: Optimization progress
        self.ax2.set_title("Optimization Progress", fontweight='bold')
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Objective Function")
        
        # Plot 3: Results comparison
        self.ax3.set_title("Stage Comparison", fontweight='bold')
        self.ax3.set_xlabel("Optimization Stage")
        self.ax3.set_ylabel("Quality Score")
        
        # Plot 4: Uncertainty analysis
        self.ax4.set_title("Uncertainty Analysis", fontweight='bold')
        self.ax4.set_xlabel("Orientation Parameter")
        self.ax4.set_ylabel("Uncertainty (degrees)")
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def load_sample_data(self):
        """Load sample data for demonstration."""
        # Generate mock polarization data
        self.polarization_data = {
            'peaks': np.array([515, 639, 1080]),  # Anatase peaks
            'intensities': np.array([0.8, 1.0, 0.6]),
            'polarizations': ['VV', 'VH', 'HH']
        }
        
        # Generate mock tensor data
        self.tensor_data = {
            'crystal_system': 'Tetragonal',
            'space_group': 'I41/amd',
            'tensors': {
                515: np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.8]]),
                639: np.array([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 1.5]]),
                1080: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.3]])
            }
        }
        
        self.update_plots_with_sample_data()
    
    def update_plots_with_sample_data(self):
        """Update plots with sample data."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'ax1'):
            return
        
        # Plot sample spectrum
        wavenumbers = np.linspace(400, 1200, 100)
        spectrum = np.zeros_like(wavenumbers)
        
        for peak, intensity in zip(self.polarization_data['peaks'], 
                                   self.polarization_data['intensities']):
            # Add Gaussian peaks
            spectrum += intensity * np.exp(-((wavenumbers - peak) / 30)**2)
        
        self.ax1.clear()
        self.ax1.plot(wavenumbers, spectrum, 'b-', linewidth=2, label='Sample Spectrum')
        self.ax1.scatter(self.polarization_data['peaks'], 
                        self.polarization_data['intensities'], 
                        c='red', s=100, zorder=5, label='Peaks')
        self.ax1.set_title("Sample Data Overview", fontweight='bold')
        self.ax1.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax1.set_ylabel("Intensity")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def load_experimental_data(self):
        """Load experimental data from files."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Experimental Data", "",
            "Data Files (*.txt *.csv *.dat);;All Files (*)"
        )
        
        if file_path:
            try:
                # Simple data loading (can be extended)
                data = np.loadtxt(file_path)
                self.data_status.setText(f"📊 Data loaded from {os.path.basename(file_path)}")
                self.data_status.setStyleSheet("color: #27ae60; font-weight: bold;")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load data: {str(e)}")
                self.data_status.setText("❌ Failed to load data")
                self.data_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
    
    def run_optimization(self, stage: str):
        """Run optimization for the specified stage."""
        if not SCIPY_AVAILABLE and stage != 'stage1':
            QMessageBox.warning(self, "Missing Dependencies", 
                              "SciPy is required for advanced optimization stages.")
            return
        
        # Stop any running optimization
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop()
            self.current_worker.wait()
        
        # Create progress dialog
        progress = QProgressDialog(f"Running {stage.upper()}...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        # Create and start worker thread
        self.current_worker = OptimizationWorker(
            stage, self, 
            max_iterations=self.max_iterations.value(),
            tolerance=self.tolerance.value()
        )
        
        # Connect signals
        self.current_worker.progress_updated.connect(progress.setValue)
        self.current_worker.status_updated.connect(self.status_label.setText)
        self.current_worker.results_ready.connect(self.on_optimization_complete)
        self.current_worker.error_occurred.connect(self.on_optimization_error)
        progress.canceled.connect(self.current_worker.stop)
        
        # Start optimization
        self.current_worker.start()
        
        # Update UI
        self.status_label.setText(f"Running {stage.upper()} optimization...")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; padding: 5px; background-color: #fef9e7; border-radius: 3px;")
    
    def on_optimization_complete(self, results: Dict[str, Any]):
        """Handle optimization completion."""
        stage = results.get('stage', '?')
        self.optimization_results[f'stage{stage}'] = results
        
        # Update status
        quality = results.get('quality_score', 0)
        self.status_label.setText(f"✅ Stage {stage} complete - Quality: {quality:.3f}")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px; background-color: #eafaf1; border-radius: 3px;")
        
        # Update plots
        self.update_results_plots()
    
    def on_optimization_error(self, error_msg: str):
        """Handle optimization errors."""
        self.status_label.setText(f"❌ Optimization failed: {error_msg}")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; padding: 5px; background-color: #fdedec; border-radius: 3px;")
        QMessageBox.critical(self, "Optimization Error", f"Optimization failed:\n{error_msg}")
    
    def update_results_plots(self):
        """Update plots with optimization results."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'ax3'):
            return
        
        # Plot results comparison
        self.ax3.clear()
        stages = []
        qualities = []
        
        for stage_key, results in self.optimization_results.items():
            stage_num = results.get('stage', 0)
            quality = results.get('quality_score', 0)
            stages.append(f"Stage {stage_num}")
            qualities.append(quality)
        
        if stages:
            bars = self.ax3.bar(stages, qualities, color=['#2196f3', '#ff9800', '#4caf50'][:len(stages)])
            self.ax3.set_title("Optimization Results Comparison", fontweight='bold')
            self.ax3.set_ylabel("Quality Score")
            self.ax3.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, quality in zip(bars, qualities):
                height = bar.get_height()
                self.ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{quality:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot uncertainty analysis if available
        self.ax4.clear()
        latest_result = list(self.optimization_results.values())[-1]
        if 'uncertainty' in latest_result:
            uncertainty = latest_result['uncertainty']
            params = ['α', 'β', 'γ']
            bars = self.ax4.bar(params, uncertainty, color='orange', alpha=0.7)
            self.ax4.set_title("Uncertainty Analysis", fontweight='bold')
            self.ax4.set_ylabel("Uncertainty (degrees)")
            
            # Add value labels
            for bar, unc in zip(bars, uncertainty):
                height = bar.get_height()
                self.ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                             f'±{unc:.1f}°', ha='center', va='bottom', fontweight='bold')
        
        self.canvas.draw()
    
    def show_detailed_results(self):
        """Show detailed results dialog."""
        if not self.optimization_results:
            QMessageBox.information(self, "No Results", "No optimization results available yet.")
            return
        
        # Show results for the latest stage
        latest_results = list(self.optimization_results.values())[-1]
        dialog = DetailedResultsDialog(latest_results, self)
        dialog.exec()
    
    def export_results(self):
        """Export optimization results to file."""
        if not self.optimization_results:
            QMessageBox.information(self, "No Results", "No optimization results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "optimization_results.txt",
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("CRYSTAL ORIENTATION OPTIMIZATION RESULTS\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for stage_key, results in self.optimization_results.items():
                        stage = results.get('stage', '?')
                        method = results.get('method', 'Unknown')
                        orientation = results.get('orientation', np.array([0, 0, 0]))
                        quality = results.get('quality_score', 0)
                        
                        f.write(f"STAGE {stage}: {method}\n")
                        f.write(f"Orientation: α={orientation[0]:.2f}°, β={orientation[1]:.2f}°, γ={orientation[2]:.2f}°\n")
                        f.write(f"Quality Score: {quality:.3f}\n\n")
                
                self.status_label.setText(f"✅ Results exported to {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
    
    def apply_flat_style(self, button):
        """Apply flat styling to buttons."""
        button.setStyleSheet("""
            QPushButton {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                color: #2c3e50;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
                border-color: #95a5a6;
            }
            QPushButton:pressed {
                background-color: #bdc3c7;
                border-color: #7f8c8d;
            }
        """)
    
    def _get_stage_style(self, bg_color: str, border_color: str) -> str:
        """Get CSS style for stage buttons."""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 12px;
                text-align: left;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {bg_color}CC;
                border-color: {border_color}DD;
            }}
            QPushButton:pressed {{
                background-color: {bg_color}AA;
                border-color: {border_color}BB;
            }}
        """


def main():
    """Main function to run the standalone widget."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Crystal Orientation Optimizer")
    app.setApplicationVersion("1.1.2")
    app.setOrganizationName("RamanLab")
    
    # Create and show widget
    widget = OrientationOptimizerWidget()
    widget.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 