#!/usr/bin/env python3
"""
Advanced Jupyter Console for RamanLab
=====================================

A standalone Jupyter console application for advanced analysis of RamanLab batch processing data.
This module provides a full IPython/Jupyter experience with:
- Interactive Python console
- Rich output (plots, tables, HTML)
- Pre-loaded RamanLab data from pickle files
- Custom analysis functions
- Integration with pandas, numpy, matplotlib

Usage:
    python advanced_jupyter_console.py [pickle_file]
    
Or launch from RamanLab Advanced tab.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Qt imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QMessageBox, QTextEdit, QSplitter, QGroupBox,
                               QTabWidget, QComboBox, QSpinBox, QDoubleSpinBox,
                               QCheckBox, QProgressBar, QStatusBar, QDialog, 
                               QScrollArea, QFrame)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QFont, QIcon, QPixmap

# Try to import Jupyter console components
try:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget
    from qtconsole.manager import QtKernelManager
    from jupyter_client import find_connection_file
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    print("Warning: Jupyter console components not available. Install with: pip install qtconsole jupyter-client ipykernel")

class RamanLabDataProcessor:
    """Helper class to process and prepare RamanLab data for analysis."""
    
    @staticmethod
    def load_pickle_data(file_path):
        """Load and validate RamanLab pickle data."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(data, list):
                raise ValueError("Expected list of batch results")
            
            return data
        except Exception as e:
            raise ValueError(f"Failed to load pickle data: {str(e)}")
    
    @staticmethod
    def convert_to_pandas(batch_data):
        """Convert batch results to pandas DataFrames."""
        results = {
            'summary': [],
            'peaks': [],
            'spectra': {}
        }
        
        for i, result in enumerate(batch_data):
            if not isinstance(result, dict):
                continue
                
            # Extract summary information
            summary_row = {
                'file_index': i,
                'filename': result.get('filename', f'spectrum_{i}'),
                'n_peaks': len(result.get('peaks', [])),
                'total_r2': result.get('total_r2', None),
                'processing_time': result.get('processing_time', None)
            }
            results['summary'].append(summary_row)
            
            # Extract peak information
            peaks = result.get('peaks', [])
            for j, peak in enumerate(peaks):
                if isinstance(peak, dict):
                    peak_row = {
                        'file_index': i,
                        'filename': result.get('filename', f'spectrum_{i}'),
                        'peak_index': j,
                        'position': peak.get('position', None),
                        'height': peak.get('height', None),
                        'width': peak.get('width', None),
                        'area': peak.get('area', None),
                        'r2': peak.get('r2', None)
                    }
                    results['peaks'].append(peak_row)
            
            # Store spectral data
            if 'wavenumbers' in result and 'intensities' in result:
                results['spectra'][result.get('filename', f'spectrum_{i}')] = {
                    'wavenumbers': np.array(result['wavenumbers']),
                    'intensities': np.array(result['intensities']),
                    'background': np.array(result.get('background', [])),
                    'fitted_peaks': result.get('fitted_peaks', None)
                }
        
        # Convert to DataFrames
        summary_df = pd.DataFrame(results['summary'])
        peaks_df = pd.DataFrame(results['peaks'])
        
        return summary_df, peaks_df, results['spectra']

class EmbeddedJupyterConsole(RichJupyterWidget):
    """Embedded Jupyter console widget for RamanLab Analysis."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.kernel_manager = None
        self.kernel_client = None
        self.setup_console()
        self.setup_kernel()
    
    def setup_kernel(self):
        """Start a new Jupyter kernel."""
        try:
            # Create kernel manager
            self.kernel_manager = QtKernelManager(kernel_name='python3')
            self.kernel_manager.start_kernel()
            
            # Create kernel client
            kernel_client = self.kernel_manager.client()
            kernel_client.start_channels()
            
            # This is the proper way to connect to RichJupyterWidget
            self.kernel_client = kernel_client
            
            # Wait a moment for kernel to be ready, then show welcome message
            QTimer.singleShot(2000, self.show_welcome_message)
            
        except Exception as e:
            print(f"Failed to start Jupyter kernel: {e}")
            self.kernel_client = None
    
    def show_welcome_message(self):
        """Show the welcome message once kernel is ready."""
        welcome_code = '''
print("""
RamanLab Advanced Analysis Console
=================================

Welcome to the RamanLab Jupyter console! This environment provides full Python/IPython
functionality for advanced analysis of your batch processing data.

Available variables will be loaded when you select a pickle file:
• summary_df    - Summary statistics for all spectra
• peaks_df      - Detailed peak information
• spectra_dict  - Dictionary of spectral data
• batch_data    - Raw batch processing results

Common analysis patterns:
• summary_df.describe()                    - Basic statistics
• peaks_df.groupby('filename').size()      - Peaks per spectrum
• plt.plot(spectra_dict['file.txt']['wavenumbers'], 
           spectra_dict['file.txt']['intensities'])  - Plot spectrum

Type 'help()' for Python help, or '?' after any object for detailed information.
Use Tab for auto-completion and Shift+Tab for function signatures.
""")
'''
        if self.kernel_client:
            self.execute(welcome_code, hidden=False)
    
    def setup_console(self):
        """Configure the console appearance and behavior."""
        # Console styling
        self.set_default_style(colors='linux')  # or 'lightbg' for light theme
        
        # Enable syntax highlighting
        self.syntax_style = 'monokai'
        
        # Set font
        font = QFont()
        font.setFamily('Consolas')  # or 'Monaco', 'Courier New'
        font.setPointSize(10)
        self.setFont(font)
    
    def inject_ramanlab_data(self, summary_df, peaks_df, spectra_dict, batch_data):
        """Inject RamanLab data into the kernel namespace."""
        if not self.kernel_client:
            return
            
        try:
            # Store data in the widget for access by the kernel
            self._summary_df = summary_df
            self._peaks_df = peaks_df
            self._spectra_dict = spectra_dict
            self._batch_data = batch_data
            
            # Prepare data injection code
            code_lines = [
                "# RamanLab data injection",
                "import numpy as np",
                "import pandas as pd", 
                "import matplotlib.pyplot as plt",
                "from pathlib import Path",
                "",
                "# Configure matplotlib for better plots",
                "plt.style.use('default')",
                "plt.rcParams['figure.figsize'] = (10, 6)",
                "plt.rcParams['font.size'] = 12",
                "",
                "print('📊 Loading RamanLab data...')"
            ]
            
            # Execute initial setup
            self.execute('\n'.join(code_lines), hidden=False)
            
            # Convert DataFrames to JSON for transmission to kernel
            summary_json = summary_df.to_json(orient='records')
            peaks_json = peaks_df.to_json(orient='records')
            
            # Inject the DataFrames
            df_injection_code = f"""
# Load DataFrames from JSON
import json
summary_data = json.loads('''{summary_json}''')
peaks_data = json.loads('''{peaks_json}''')

summary_df = pd.DataFrame(summary_data)
peaks_df = pd.DataFrame(peaks_data)
"""
            
            self.execute(df_injection_code, hidden=False)
            
            # Inject spectral data dictionary
            spectra_code_lines = ["# Load spectral data", "spectra_dict = {}"]
            for filename, spectrum_data in spectra_dict.items():
                # Convert numpy arrays to lists for JSON serialization
                wavenumbers_list = spectrum_data['wavenumbers'].tolist()
                intensities_list = spectrum_data['intensities'].tolist()
                background_list = spectrum_data['background'].tolist() if len(spectrum_data['background']) > 0 else []
                
                spectra_code_lines.append(f"""
spectra_dict['{filename}'] = {{
    'wavenumbers': np.array({wavenumbers_list}),
    'intensities': np.array({intensities_list}),
    'background': np.array({background_list}),
    'fitted_peaks': {spectrum_data.get('fitted_peaks', None)}
}}""")
            
            self.execute('\n'.join(spectra_code_lines), hidden=False)
            
            # Store raw batch data (simplified)
            batch_data_code = f"""
# Raw batch data (simplified - access via spectra_dict for full data)
batch_data = {len(batch_data)} # Number of processed spectra
"""
            
            self.execute(batch_data_code, hidden=False)
            
            # Create summary message
            summary_code = f"""
# Data summary
n_spectra = len(summary_df)
n_peaks_total = len(peaks_df)
n_spectra_with_data = len(spectra_dict)

print(f'✅ Data loaded successfully!')
print(f'📈 {{n_spectra}} spectra processed')
print(f'🔍 {{n_peaks_total}} peaks detected')
print(f'📊 {{n_spectra_with_data}} spectra with full data')
print()
print('Available variables:')
print('• summary_df    - Summary statistics DataFrame')
print('• peaks_df      - Peak parameters DataFrame') 
print('• spectra_dict  - Dictionary of spectral data')
print('• batch_data    - Number of processed spectra')
print()
print('Try: summary_df.head() or peaks_df.describe()')
"""
            
            self.execute(summary_code, hidden=False)
            
        except Exception as e:
            error_msg = f"Failed to inject data: {str(e)}"
            self.execute(f"print('❌ {error_msg}')", hidden=False)
    
    def shutdown_kernel(self):
        """Clean shutdown of the kernel."""
        if self.kernel_client:
            self.kernel_client.stop_channels()
        if self.kernel_manager:
            self.kernel_manager.shutdown_kernel()

class SimpleCodeEditor(QTextEdit):
    """Simple code editor as fallback when Jupyter is not available."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_editor()
        self.namespace = {}
    
    def setup_editor(self):
        """Setup the code editor."""
        # Set font
        font = QFont()
        font.setFamily('Consolas')
        font.setPointSize(10)
        self.setFont(font)
        
        # Set placeholder text
        self.setPlaceholderText("""# RamanLab Analysis Console (Fallback Mode)
# Jupyter console not available. Basic Python execution only.

# Example analysis:
print("Available data:", list(locals().keys()))
if 'summary_df' in locals():
    print("Summary statistics:")
    print(summary_df.describe())
""")
    
    def execute_code(self):
        """Execute the code in the editor."""
        code = self.toPlainText()
        if not code.strip():
            return
            
        try:
            # Capture output
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Execute code
            exec(code, self.namespace)
            
            # Get output
            output = captured_output.getvalue()
            sys.stdout = old_stdout
            
            if output:
                self.append(f"\n--- Output ---\n{output}")
                
        except Exception as e:
            self.append(f"\n--- Error ---\n{str(e)}")
    
    def inject_data(self, summary_df, peaks_df, spectra_dict, batch_data):
        """Inject data into the namespace."""
        self.namespace.update({
            'summary_df': summary_df,
            'peaks_df': peaks_df,
            'spectra_dict': spectra_dict,
            'batch_data': batch_data,
            'np': np,
            'pd': pd,
            'plt': plt
        })

class AdvancedJupyterConsole(QMainWindow):
    """Main window for the advanced Jupyter console."""
    
    def __init__(self, pickle_file=None):
        super().__init__()
        self.pickle_file = pickle_file
        self.batch_data = None
        self.summary_df = None
        self.peaks_df = None
        self.spectra_dict = None
        
        self.setup_ui()
        self.setup_matplotlib_config()
        
        # Load data if provided
        if pickle_file:
            self.load_pickle_file(pickle_file)
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("RamanLab Advanced Analysis Console")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Create main content area
        if JUPYTER_AVAILABLE:
            try:
                self.console = EmbeddedJupyterConsole()
                layout.addWidget(self.console)
            except Exception as e:
                print(f"Failed to create Jupyter console: {e}")
                # Fallback to simple editor
                fallback_widget = self.create_fallback_widget()
                layout.addWidget(fallback_widget)
        else:
            # Fallback: simple code editor
            fallback_widget = self.create_fallback_widget()
            layout.addWidget(fallback_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load a pickle file to begin analysis")
    
    def create_control_panel(self):
        """Create the control panel."""
        panel = QGroupBox("Data Management")
        layout = QHBoxLayout(panel)
        
        # File selection
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.file_label)
        
        # Load button
        load_btn = QPushButton("Load Pickle File")
        load_btn.clicked.connect(self.select_pickle_file)
        layout.addWidget(load_btn)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.clicked.connect(self.refresh_data)
        refresh_btn.setEnabled(False)
        self.refresh_btn = refresh_btn
        layout.addWidget(refresh_btn)
        
        # Clear button
        if JUPYTER_AVAILABLE:
            clear_btn = QPushButton("🗑️ Clear Console")
            clear_btn.clicked.connect(self.clear_console)
            layout.addWidget(clear_btn)
        
        # Python Examples button
        examples_btn = QPushButton("📚 Python Examples")
        examples_btn.clicked.connect(self.show_examples_window)
        examples_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(examples_btn)
        
        layout.addStretch()
        return panel
    
    def create_fallback_widget(self):
        """Create fallback widget when Jupyter is not available."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Warning message
        warning = QLabel("⚠️ Jupyter Console Not Available")
        warning.setStyleSheet("color: orange; font-weight: bold; font-size: 14px;")
        layout.addWidget(warning)
        
        info = QLabel("Install Jupyter components for full functionality:\n"
                     "pip install qtconsole jupyter-client ipykernel")
        info.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info)
        
        # Simple code editor
        self.code_editor = SimpleCodeEditor()
        layout.addWidget(self.code_editor)
        
        # Execute button
        execute_btn = QPushButton("Execute Code")
        execute_btn.clicked.connect(self.code_editor.execute_code)
        layout.addWidget(execute_btn)
        
        return widget
    
    def setup_matplotlib_config(self):
        """Setup matplotlib configuration."""
        try:
            # Try to import and use the RamanLab matplotlib config
            from matplotlib_config import setup_matplotlib_style
            setup_matplotlib_style()
        except ImportError:
            # Fallback matplotlib configuration
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
    
    def select_pickle_file(self):
        """Select a pickle file to load."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select RamanLab Pickle File", 
            "", "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            self.load_pickle_file(file_path)
    
    def load_pickle_file(self, file_path):
        """Load and process a pickle file."""
        try:
            # Load raw data
            with open(file_path, 'rb') as f:
                raw_data = pickle.load(f)
            
            # Check if it's the new pandas format or old batch format
            if isinstance(raw_data, dict) and 'summary_df' in raw_data:
                # New pandas format - use directly
                self.summary_df = raw_data['summary_df']
                self.peaks_df = raw_data['peaks_df']
                self.spectra_dict = raw_data['spectra_dict']
                self.batch_data = raw_data  # Store entire structure
                
                # Extract count for status
                spectrum_count = len(self.summary_df) if hasattr(self.summary_df, '__len__') else 0
                
            else:
                # Old batch format - convert using existing processor
                self.batch_data = RamanLabDataProcessor.load_pickle_data(file_path)
                
                # Convert to pandas
                self.summary_df, self.peaks_df, self.spectra_dict = \
                    RamanLabDataProcessor.convert_to_pandas(self.batch_data)
                
                spectrum_count = len(self.batch_data)
            
            # Update UI
            self.pickle_file = file_path
            self.file_label.setText(f"Loaded: {Path(file_path).name}")
            self.refresh_btn.setEnabled(True)
            
            # Inject data
            self.refresh_data()
            
            # Update status
            self.status_bar.showMessage(
                f"Loaded {spectrum_count} spectra from {Path(file_path).name}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load pickle file:\n{str(e)}")
    
    def refresh_data(self):
        """Refresh data in the console."""
        if not self.batch_data:
            return
            
        try:
            if JUPYTER_AVAILABLE and hasattr(self, 'console'):
                self.console.inject_ramanlab_data(
                    self.summary_df, self.peaks_df, 
                    self.spectra_dict, self.batch_data
                )
            elif hasattr(self, 'code_editor'):
                self.code_editor.inject_data(
                    self.summary_df, self.peaks_df,
                    self.spectra_dict, self.batch_data
                )
                
        except Exception as e:
            QMessageBox.warning(self, "Data Refresh Error", f"Failed to refresh data:\n{str(e)}")
    
    def clear_console(self):
        """Clear the console."""
        if JUPYTER_AVAILABLE and hasattr(self, 'console'):
            self.console.clear()
    
    def show_examples_window(self):
        """Show the Python examples window."""
        if not hasattr(self, 'examples_window') or self.examples_window is None:
            self.examples_window = PythonExamplesWindow(self)
        
        self.examples_window.show()
        self.examples_window.raise_()
        self.examples_window.activateWindow()
    
    def closeEvent(self, event):
        """Handle window close event."""
        if JUPYTER_AVAILABLE and hasattr(self, 'console'):
            self.console.shutdown_kernel()
        event.accept()


class PythonExamplesWindow(QDialog):
    """Persistent window showing Python examples for Raman spectroscopy."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RamanLab Python Examples - Your Guide to Spectral Analysis")
        self.setGeometry(200, 200, 1200, 800)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Make the window persistent
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        
        self.setup_ui()
        
        # Use matplotlib config for consistency
        try:
            from polarization_ui.matplotlib_config import setup_matplotlib
            setup_matplotlib()
        except ImportError:
            pass
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("🐍 Python Examples for Raman Spectroscopy & Batch Processing")
        header.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2E7D32;
                padding: 10px;
                background-color: #E8F5E8;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Info label
        info_label = QLabel("💡 Click examples to copy to clipboard, then paste into console!")
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # Create tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Basic Data Exploration
        self.tabs.addTab(self.create_basic_tab(), "📊 Basic Data Exploration")
        
        # Tab 2: Peak Analysis
        self.tabs.addTab(self.create_peak_analysis_tab(), "🔍 Peak Analysis")
        
        # Tab 3: Plotting & Visualization
        self.tabs.addTab(self.create_plotting_tab(), "📈 Plotting & Visualization")
        
        # Tab 4: Batch Processing
        self.tabs.addTab(self.create_batch_tab(), "⚙️ Batch Processing")
        
        # Tab 5: Advanced Analysis
        self.tabs.addTab(self.create_advanced_tab(), "🎯 Advanced Analysis")
        
        # Tab 6: Export & Save
        self.tabs.addTab(self.create_export_tab(), "💾 Export & Save")
        
        layout.addWidget(self.tabs)
        
        # Close button
        close_btn = QPushButton("Close Examples Window")
        close_btn.clicked.connect(self.hide)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        layout.addWidget(close_btn)
    
    def create_basic_tab(self):
        """Create basic data exploration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        examples = [
            ("View loaded data summary", """
# Check what data is available
print("Summary DataFrame:")
print(summary_df.head())
print(f"Shape: {summary_df.shape}")

print("\\nPeaks DataFrame:")
print(peaks_df.head())
print(f"Shape: {peaks_df.shape}")

print("\\nSpectra available:")
print(list(spectra_dict.keys())[:5])  # First 5 spectra
"""),
            
            ("Basic statistics", """
# Summary statistics for all spectra
print("Summary Statistics:")
print(summary_df.describe())

print("\\nPeak Position Statistics:")
# Handle both old and new column names
position_col = 'peak_center' if 'peak_center' in peaks_df.columns else 'position'
if position_col in peaks_df.columns:
    print(peaks_df[position_col].describe())
else:
    print("No peak position data available")

print("\\nR² Statistics:")
r2_col = 'total_r2' if 'total_r2' in peaks_df.columns else 'r2'
if r2_col in peaks_df.columns:
    print(peaks_df[r2_col].describe())
else:
    print("No R² data available")
"""),
            
            ("Find spectra with most peaks", """
# Find files with the most peaks
top_spectra = summary_df.nlargest(10, 'n_peaks')
print("Spectra with most peaks:")
print(top_spectra[['filename', 'n_peaks', 'total_r2']])
"""),
            
            ("View specific spectrum data", """
# Access spectral data for a specific file
filename = list(spectra_dict.keys())[0]  # First spectrum
spectrum_data = spectra_dict[filename]

print(f"Spectrum: {filename}")
print(f"Wavenumber range: {spectrum_data['wavenumbers'].min():.1f} - {spectrum_data['wavenumbers'].max():.1f} cm⁻¹")
print(f"Data points: {len(spectrum_data['wavenumbers'])}")

# Show what data components are available
print("\\nAvailable data components:")
for key in spectrum_data.keys():
    if key not in ['filename', 'region_start', 'region_end']:
        data_array = spectrum_data[key]
        if hasattr(data_array, '__len__'):
            print(f"  {key}: {len(data_array)} points")
        else:
            print(f"  {key}: {data_array}")

# Show data ranges
if 'original_intensities' in spectrum_data:
    print(f"\\nOriginal intensity range: {spectrum_data['original_intensities'].min():.1f} - {spectrum_data['original_intensities'].max():.1f}")
print(f"Background-corrected range: {spectrum_data['intensities'].min():.1f} - {spectrum_data['intensities'].max():.1f}")
if 'background' in spectrum_data:
    print(f"Background range: {spectrum_data['background'].min():.1f} - {spectrum_data['background'].max():.1f}")
"""),
            
            ("Filter data by criteria", """
# Filter spectra by number of peaks
high_peak_spectra = summary_df[summary_df['n_peaks'] >= 5]
print(f"Spectra with ≥5 peaks: {len(high_peak_spectra)}")

# Filter by R² value
good_fits = summary_df[summary_df['total_r2'] >= 0.9]
print(f"Spectra with R² ≥ 0.9: {len(good_fits)}")

# Filter peaks by position (e.g., find quartz peak around 464 cm⁻¹)
position_col = 'peak_center' if 'peak_center' in peaks_df.columns else 'position'
if position_col in peaks_df.columns:
    quartz_peaks = peaks_df[(peaks_df[position_col] >= 460) & (peaks_df[position_col] <= 470)]
    print(f"Peaks near 464 cm⁻¹ (quartz): {len(quartz_peaks)}")
else:
    print("No position data available")
""")
        ]
        
        for title, code in examples:
            scroll_layout.addWidget(self.create_example_widget(title, code))
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def create_peak_analysis_tab(self):
        """Create peak analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        examples = [
            ("Identify common mineral peaks", """
# Common Raman peaks for mineral identification
peak_ranges = {
    'Quartz': [(464, 468), (206, 210)],
    'Calcite': [(1085, 1090), (712, 716)],
    'Feldspar': [(507, 513), (476, 480)],
    'Muscovite': [(700, 710), (258, 265)],
    'Pyrite': [(342, 348), (378, 383)]
}

position_col = 'peak_center' if 'peak_center' in peaks_df.columns else 'position'
if position_col in peaks_df.columns:
    for mineral, ranges in peak_ranges.items():
        print(f"\\n{mineral} peaks:")
        for min_pos, max_pos in ranges:
            mineral_peaks = peaks_df[(peaks_df[position_col] >= min_pos) & (peaks_df[position_col] <= max_pos)]
            print(f"  {min_pos}-{max_pos} cm⁻¹: {len(mineral_peaks)} peaks found")
            if len(mineral_peaks) > 0:
                print(f"    Files: {mineral_peaks['filename'].unique()[:3]}")  # First 3 files
else:
    print("No position data available for mineral identification")
"""),
            
            ("Peak intensity analysis", """
# Analyze peak intensities
height_col = 'amplitude' if 'amplitude' in peaks_df.columns else 'height'
position_col = 'peak_center' if 'peak_center' in peaks_df.columns else 'position'

if height_col in peaks_df.columns:
    print("Peak intensity statistics:")
    print(f"Mean height: {peaks_df[height_col].mean():.2f}")
    print(f"Max height: {peaks_df[height_col].max():.2f}")
    print(f"Min height: {peaks_df[height_col].min():.2f}")
    
    # Find strongest peaks
    strongest_peaks = peaks_df.nlargest(10, height_col)
    print("\\nStrongest peaks:")
    cols = ['filename', position_col, height_col]
    if 'width' in peaks_df.columns:
        cols.append('width')
    print(strongest_peaks[cols])
else:
    print("No intensity data available")
"""),
            
            ("Peak width analysis", """
# Analyze peak widths (FWHM)
print("Peak width statistics:")
print(f"Mean FWHM: {peaks_df['width'].mean():.2f} cm⁻¹")
print(f"Standard deviation: {peaks_df['width'].std():.2f} cm⁻¹")

# Find narrow vs broad peaks
narrow_peaks = peaks_df[peaks_df['width'] < 10]
broad_peaks = peaks_df[peaks_df['width'] > 30]

print(f"\\nNarrow peaks (<10 cm⁻¹): {len(narrow_peaks)}")
print(f"Broad peaks (>30 cm⁻¹): {len(broad_peaks)}")
"""),
            
            ("Peak quality assessment", """
# Assess peak fitting quality
r2_col = 'total_r2' if 'total_r2' in peaks_df.columns else 'r2'
position_col = 'peak_center' if 'peak_center' in peaks_df.columns else 'position'
height_col = 'amplitude' if 'amplitude' in peaks_df.columns else 'height'

if r2_col in peaks_df.columns:
    print("Peak fitting quality:")
    good_peaks = peaks_df[peaks_df[r2_col] >= 0.95]
    poor_peaks = peaks_df[peaks_df[r2_col] < 0.8]
    
    print(f"High quality peaks (R² ≥ 0.95): {len(good_peaks)}")
    print(f"Poor quality peaks (R² < 0.8): {len(poor_peaks)}")
    
    # Show worst fitting peaks
    worst_peaks = peaks_df.nsmallest(5, r2_col)
    print("\\nWorst fitting peaks:")
    cols = ['filename', position_col, height_col, r2_col]
    print(worst_peaks[cols])
else:
    print("No R² data available for quality assessment")
""")
        ]
        
        for title, code in examples:
            scroll_layout.addWidget(self.create_example_widget(title, code))
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def create_plotting_tab(self):
        """Create plotting and visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        examples = [
            ("Plot a single spectrum", """
# Plot first spectrum
filename = list(spectra_dict.keys())[0]
spectrum_data = spectra_dict[filename]

plt.figure(figsize=(12, 6))
plt.plot(spectrum_data['wavenumbers'], spectrum_data['intensities'], 'b-', linewidth=1)
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.title(f'Raman Spectrum: {filename}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),

            ("Complete spectral analysis plot", """
# Plot comprehensive spectral analysis for first spectrum
filename = list(spectra_dict.keys())[0]
spectrum_data = spectra_dict[filename]

plt.figure(figsize=(15, 10))

# Main spectrum with components
plt.subplot(2, 2, 1)
if 'original_intensities' in spectrum_data:
    plt.plot(spectrum_data['wavenumbers'], spectrum_data['original_intensities'], 'k-', alpha=0.7, label='Original')
if 'background' in spectrum_data:
    plt.plot(spectrum_data['wavenumbers'], spectrum_data['background'], 'r--', label='Background')
plt.plot(spectrum_data['wavenumbers'], spectrum_data['intensities'], 'b-', label='Corrected')
if 'fitted_peaks' in spectrum_data:
    plt.plot(spectrum_data['wavenumbers'], spectrum_data['fitted_peaks'], 'g-', label='Fitted')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.title(f'Complete Analysis: {filename}')
plt.legend()
plt.grid(True, alpha=0.3)

# Background-corrected spectrum only
plt.subplot(2, 2, 2)
plt.plot(spectrum_data['wavenumbers'], spectrum_data['intensities'], 'b-', linewidth=1.5)
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.title('Background-Corrected Spectrum')
plt.grid(True, alpha=0.3)

# Fitted peaks overlay
plt.subplot(2, 2, 3)
plt.plot(spectrum_data['wavenumbers'], spectrum_data['intensities'], 'b-', alpha=0.7, label='Data')
if 'fitted_peaks' in spectrum_data:
    plt.plot(spectrum_data['wavenumbers'], spectrum_data['fitted_peaks'], 'r-', linewidth=2, label='Fit')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.title('Peak Fitting Results')
plt.legend()
plt.grid(True, alpha=0.3)

# Residuals
plt.subplot(2, 2, 4)
if 'residuals' in spectrum_data:
    plt.plot(spectrum_data['wavenumbers'], spectrum_data['residuals'], 'g-', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Residuals')
    plt.title('Fitting Residuals')
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No residuals available', ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
"""),
            
            ("Plot multiple spectra comparison", """
# Compare first 3 spectra
plt.figure(figsize=(12, 8))
filenames = list(spectra_dict.keys())[:3]

for i, filename in enumerate(filenames):
    spectrum_data = spectra_dict[filename]
    plt.plot(spectrum_data['wavenumbers'], spectrum_data['intensities'] + i*1000, 
             label=filename, linewidth=1)

plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity (offset)')
plt.title('Comparison of Multiple Raman Spectra')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
            
            ("Plot peak distribution histogram", """
# Plot peak position distribution
position_col = 'peak_center' if 'peak_center' in peaks_df.columns else 'position'
if position_col in peaks_df.columns:
    plt.figure(figsize=(12, 6))
    plt.hist(peaks_df[position_col], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Peak Position (cm⁻¹)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Peak Positions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No position data available for histogram")
"""),
            
            ("Plot peak intensity vs position", """
# Scatter plot of peak intensity vs position
position_col = 'peak_center' if 'peak_center' in peaks_df.columns else 'position'
height_col = 'amplitude' if 'amplitude' in peaks_df.columns else 'height'

if position_col in peaks_df.columns and height_col in peaks_df.columns:
    plt.figure(figsize=(12, 6))
    plt.scatter(peaks_df[position_col], peaks_df[height_col], alpha=0.6, s=30)
    plt.xlabel('Peak Position (cm⁻¹)')
    plt.ylabel('Peak Height')
    plt.title('Peak Intensity vs Position')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Missing position or intensity data for scatter plot")
""")
        ]
        
        for title, code in examples:
            scroll_layout.addWidget(self.create_example_widget(title, code))
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def create_batch_tab(self):
        """Create batch processing tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        examples = [
            ("Processing statistics overview", """
# Get overview of batch processing results
print("Batch Processing Summary:")
print(f"Total files processed: {len(summary_df)}")
print(f"Total peaks detected: {len(peaks_df)}")
print(f"Average peaks per spectrum: {summary_df['n_peaks'].mean():.1f}")
print(f"Average R²: {summary_df['total_r2'].mean():.3f}")

# Processing time statistics
if 'processing_time' in summary_df.columns:
    print(f"Average processing time: {summary_df['processing_time'].mean():.2f} seconds")
    print(f"Total processing time: {summary_df['processing_time'].sum():.1f} seconds")
"""),
            
            ("Find problematic spectra", """
# Identify spectra that might need attention
print("Problematic Spectra Analysis:")

# No peaks found
no_peaks = summary_df[summary_df['n_peaks'] == 0]
print(f"Spectra with no peaks: {len(no_peaks)}")

# Poor fitting
poor_fits = summary_df[summary_df['total_r2'] < 0.7]
print(f"Spectra with poor fits (R² < 0.7): {len(poor_fits)}")

# Too many peaks (might be noise)
too_many_peaks = summary_df[summary_df['n_peaks'] > 20]
print(f"Spectra with >20 peaks: {len(too_many_peaks)}")

if len(poor_fits) > 0:
    print("\\nPoor fitting spectra:")
    print(poor_fits[['filename', 'n_peaks', 'total_r2']])
"""),
            
            ("Export batch results", """
# Export processed results to CSV files
import pandas as pd

# Export summary
summary_df.to_csv('batch_summary.csv', index=False)
print("Exported batch_summary.csv")

# Export peaks
peaks_df.to_csv('batch_peaks.csv', index=False)
print("Exported batch_peaks.csv")

# Export only high-quality peaks
high_quality_peaks = peaks_df[peaks_df['r2'] >= 0.9]
high_quality_peaks.to_csv('high_quality_peaks.csv', index=False)
print(f"Exported {len(high_quality_peaks)} high-quality peaks to high_quality_peaks.csv")
""")
        ]
        
        for title, code in examples:
            scroll_layout.addWidget(self.create_example_widget(title, code))
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def create_advanced_tab(self):
        """Create advanced analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        examples = [
            ("Principal Component Analysis (PCA)", """
# Perform PCA on spectral data
from sklearn.decomposition import PCA
import numpy as np

# Prepare data matrix
spectra_list = []
filenames_list = []

for filename, spectrum_data in spectra_dict.items():
    spectra_list.append(spectrum_data['intensities'])
    filenames_list.append(filename)

if len(spectra_list) > 3:
    # Stack spectra into matrix
    X = np.vstack(spectra_list)
    
    # Perform PCA
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(X)
    
    print("PCA Results:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Plot PCA results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA Score Plot')
    
    plt.subplot(1, 2, 2)
    plt.plot(pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    
    plt.tight_layout()
    plt.show()
else:
    print("Need at least 4 spectra for PCA analysis")
"""),
            
            ("Spectral similarity analysis", """
# Calculate spectral similarity using correlation
import numpy as np
from scipy.stats import pearsonr

# Get first few spectra for comparison
filenames = list(spectra_dict.keys())[:5]
similarity_matrix = np.zeros((len(filenames), len(filenames)))

for i, fname1 in enumerate(filenames):
    for j, fname2 in enumerate(filenames):
        if i <= j:
            spec1 = spectra_dict[fname1]['intensities']
            spec2 = spectra_dict[fname2]['intensities']
            
            # Ensure same length
            min_len = min(len(spec1), len(spec2))
            corr, _ = pearsonr(spec1[:min_len], spec2[:min_len])
            similarity_matrix[i, j] = corr
            similarity_matrix[j, i] = corr

# Plot similarity matrix
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(filenames)), [f.split('/')[-1][:10] for f in filenames], rotation=45)
plt.yticks(range(len(filenames)), [f.split('/')[-1][:10] for f in filenames])
plt.title('Spectral Similarity Matrix')
plt.tight_layout()
plt.show()

print("Similarity Matrix:")
print(similarity_matrix)
""")
        ]
        
        for title, code in examples:
            scroll_layout.addWidget(self.create_example_widget(title, code))
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def create_export_tab(self):
        """Create export and save tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        examples = [
            ("Export data to Excel", """
# Export all data to Excel with multiple sheets
with pd.ExcelWriter('ramanlab_analysis.xlsx', engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    peaks_df.to_excel(writer, sheet_name='Peaks', index=False)
    
    # Create summary statistics sheet
    stats_data = {
        'Metric': ['Total Files', 'Total Peaks', 'Avg Peaks/File', 'Avg R²', 'Success Rate %'],
        'Value': [
            len(summary_df),
            len(peaks_df),
            f"{summary_df['n_peaks'].mean():.1f}",
            f"{summary_df['total_r2'].mean():.3f}",
            f"{(len(summary_df[summary_df['n_peaks'] > 0]) / len(summary_df) * 100):.1f}"
        ]
    }
    pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)
    
print("Exported ramanlab_analysis.xlsx with multiple sheets")
"""),
            
            ("Save figures as high-quality images", """
# Save current figure as high-quality image
plt.figure(figsize=(12, 8))

# Example plot
filename = list(spectra_dict.keys())[0]
spectrum_data = spectra_dict[filename]
plt.plot(spectrum_data['wavenumbers'], spectrum_data['intensities'], 'b-', linewidth=1)
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.title(f'Raman Spectrum: {filename}')
plt.grid(True, alpha=0.3)

# Save in multiple formats
plt.savefig('raman_spectrum.png', dpi=300, bbox_inches='tight')
plt.savefig('raman_spectrum.pdf', bbox_inches='tight')
plt.savefig('raman_spectrum.svg', bbox_inches='tight')
plt.show()

print("Saved spectrum as PNG, PDF, and SVG")
""")
        ]
        
        for title, code in examples:
            scroll_layout.addWidget(self.create_example_widget(title, code))
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def create_example_widget(self, title, code):
        """Create a widget for a single example."""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 5px;
                margin: 2px;
                background-color: #f9f9f9;
            }
        """)
        
        layout = QVBoxLayout(widget)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2E7D32;
                padding: 5px;
                background-color: #E8F5E8;
                border-radius: 3px;
            }
        """)
        layout.addWidget(title_label)
        
        # Code
        code_text = QTextEdit()
        code_text.setPlainText(code.strip())
        code_text.setReadOnly(True)
        code_text.setMaximumHeight(200)
        
        # Code styling - use cross-platform fallback
        font = QFont()
        font.setFamily('Monaco')  # Better for Mac
        if not font.exactMatch():
            font.setFamily('Courier New')  # Windows fallback
        if not font.exactMatch():
            font.setFamily('monospace')  # Generic fallback
        font.setPointSize(9)
        code_text.setFont(font)
        code_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #f8f8f2;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        
        layout.addWidget(code_text)
        
        # Copy button
        copy_btn = QPushButton("📋 Copy to Clipboard")
        copy_btn.clicked.connect(lambda: self.copy_to_clipboard(code.strip()))
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(copy_btn)
        
        return widget
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
        # Show temporary feedback
        self.show_copy_feedback()
    
    def show_copy_feedback(self):
        """Show temporary feedback that text was copied."""
        # This could be enhanced with a popup or status message
        pass
    
    def closeEvent(self, event):
        """Handle close event by hiding instead of closing."""
        event.ignore()
        self.hide()


def main():
    """Main function."""
    app = QApplication(sys.argv)
    
    # Get pickle file from command line if provided
    pickle_file = None
    if len(sys.argv) > 1:
        pickle_file = sys.argv[1]
    
    # Create and show the console
    console = AdvancedJupyterConsole(pickle_file)
    console.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 