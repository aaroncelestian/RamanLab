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
                               QScrollArea, QFrame, QLineEdit, QInputDialog)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QFont, QIcon, QPixmap, QClipboard

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
    def standardize_dataframe_columns(df, df_type='peaks'):
        """
        Standardize DataFrame column names to ensure consistent access patterns.
        
        Args:
            df: pandas DataFrame to standardize
            df_type: 'peaks' or 'summary' to determine which standardization to apply
            
        Returns:
            DataFrame with standardized column names
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        
        if df_type == 'peaks':
            # Define mapping from possible column names to standard names
            column_mappings = {
                # Position/wavenumber mappings
                'position': ['position', 'peak_position', 'center', 'peak_center', 'wavenumber', 'wn', 'freq', 'frequency'],
                # Height/intensity mappings  
                'height': ['height', 'amplitude', 'intensity', 'amp', 'peak_height', 'peak_amplitude', 'peak_intensity'],
                # Width mappings
                'width': ['width', 'fwhm', 'sigma', 'peak_width', 'full_width_half_max'],
                # Area mappings
                'area': ['area', 'peak_area', 'integrated_area', 'integration'],
                # R² mappings
                'r2': ['r2', 'r_squared', 'r_sq', 'fit_quality', 'goodness_of_fit', 'correlation'],
                # File identification
                'filename': ['filename', 'file', 'name', 'file_name', 'spectrum_name'],
                'file_index': ['file_index', 'index', 'file_id', 'spectrum_index'],
                'peak_index': ['peak_index', 'peak_id', 'peak_number', 'peak_num']
            }
            
        elif df_type == 'summary':
            column_mappings = {
                'filename': ['filename', 'file', 'name', 'file_name', 'spectrum_name'],
                'file_index': ['file_index', 'index', 'file_id', 'spectrum_index'],
                'n_peaks': ['n_peaks', 'num_peaks', 'peak_count', 'peaks_found', 'number_of_peaks'],
                'total_r2': ['total_r2', 'r2', 'r_squared', 'overall_r2', 'fit_quality'],
                'processing_time': ['processing_time', 'time', 'duration', 'elapsed_time']
            }
        else:
            return df_copy
            
        # Apply mappings
        current_columns = df_copy.columns.tolist()
        rename_dict = {}
        
        for standard_name, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in current_columns and standard_name not in current_columns:
                    rename_dict[possible_name] = standard_name
                    break  # Use first match found
        
        if rename_dict:
            df_copy = df_copy.rename(columns=rename_dict)
            print(f"📋 Standardized columns: {rename_dict}")
            
        return df_copy

    @staticmethod
    def convert_to_pandas(batch_data):
        """Convert batch results to pandas DataFrames with standardized columns."""
        results = {
            'summary': [],
            'peaks': [],
            'spectra': {}
        }
        
        for i, result in enumerate(batch_data):
            if not isinstance(result, dict):
                continue
                
            # Extract summary information with flexible field names
            summary_row = {
                'file_index': i,
                'filename': result.get('filename', f'spectrum_{i}'),
                'n_peaks': len(result.get('peaks', [])),
                'total_r2': result.get('total_r2', result.get('r2', result.get('r_squared', None))),
                'processing_time': result.get('processing_time', result.get('time', None))
            }
            results['summary'].append(summary_row)
            
            # Extract peak information with flexible field names
            peaks = result.get('peaks', [])
            for j, peak in enumerate(peaks):
                if isinstance(peak, dict):
                    # Try multiple possible field names for each parameter
                    peak_row = {
                        'file_index': i,
                        'filename': result.get('filename', f'spectrum_{i}'),
                        'peak_index': j,
                        'position': peak.get('position', peak.get('center', peak.get('peak_position', peak.get('wavenumber', None)))),
                        'height': peak.get('height', peak.get('amplitude', peak.get('intensity', peak.get('amp', None)))),
                        'width': peak.get('width', peak.get('fwhm', peak.get('sigma', None))),
                        'area': peak.get('area', peak.get('peak_area', None)),
                        'r2': peak.get('r2', peak.get('r_squared', peak.get('fit_quality', None)))
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
        
        # Apply column standardization
        summary_df = RamanLabDataProcessor.standardize_dataframe_columns(summary_df, 'summary')
        peaks_df = RamanLabDataProcessor.standardize_dataframe_columns(peaks_df, 'peaks')
        
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
        """Inject RamanLab data into the kernel namespace using the proven Analysis tab approach."""
        if not self.kernel_client:
            return
            
        try:
            # Store data in the widget for access by the kernel
            self._summary_df = summary_df
            self._peaks_df = peaks_df
            self._spectra_dict = spectra_dict
            self._batch_data = batch_data
            
            # Prepare initial setup code
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
            
            # Calculate basic statistics
            n_spectra = len(summary_df)
            n_peaks_total = len(peaks_df)
            n_spectra_with_data = len(spectra_dict)
            avg_peaks = summary_df['n_peaks'].mean() if 'n_peaks' in summary_df.columns else 0
            avg_r2 = summary_df['total_r2'].mean() if 'total_r2' in summary_df.columns else 0
            
            # Use the proven approach from Analysis tab - inject pandas DataFrames directly
            # Convert DataFrames to string representation for execution
            
            # Inject summary DataFrame
            summary_csv = summary_df.to_csv(index=False)
            summary_injection = f"""
# Create summary DataFrame using CSV approach (same as Analysis tab)
from io import StringIO
summary_csv_data = '''{summary_csv}'''
summary_df = pd.read_csv(StringIO(summary_csv_data))
"""
            self.execute(summary_injection, hidden=True)
            
            # Inject peaks DataFrame  
            peaks_csv = peaks_df.to_csv(index=False)
            peaks_injection = f"""
# Create peaks DataFrame using CSV approach (same as Analysis tab)
peaks_csv_data = '''{peaks_csv}'''
peaks_df = pd.read_csv(StringIO(peaks_csv_data))
"""
            self.execute(peaks_injection, hidden=True)
            
            # Inject basic spectral information and show completion message
            filenames_list = list(spectra_dict.keys())
            completion_injection = f"""
# Create spectra dictionary with filenames (same as Analysis tab)
spectra_dict = {dict.fromkeys(filenames_list, 'Available in console')}
batch_data = {len(batch_data)}  # Number of processed spectra

# Basic statistics
n_spectra = {n_spectra}
n_peaks_total = {n_peaks_total}
n_spectra_with_data = {n_spectra_with_data}
avg_peaks_per_spectrum = {avg_peaks:.1f}
avg_r2 = {avg_r2:.3f}

print('✅ RamanLab data ready!')
print(f'📈 {{n_spectra}} spectra processed')
print(f'🔍 {{n_peaks_total}} peaks detected')
print(f'📊 {{n_spectra_with_data}} spectra with full data')
print(f'⚡ Average: {{avg_peaks_per_spectrum}} peaks/spectrum')
print(f'📐 Average R²: {{avg_r2:.3f}}')
print()
print('💡 Your data is now available as:')
print(f'• summary_df - Summary statistics DataFrame ({{len(summary_df)}} rows)')
print(f'• peaks_df - Peak parameters DataFrame ({{len(peaks_df)}} rows)')
print(f'• spectra_dict - Dictionary of spectrum filenames ({{len(spectra_dict)}} files)')
print('• batch_data - Number of processed spectra')
print()
print('📋 Standardized column names:')
print(f'• summary_df columns: {{list(summary_df.columns)}}')
print(f'• peaks_df columns: {{list(peaks_df.columns)}}')
print()
print('🎯 Ready-to-use examples:')
print('>>> summary_df.head()')
print('>>> peaks_df.describe()')
print('>>> peaks_df.groupby("filename").size()')
print('>>> peaks_df[peaks_df["r2"] >= 0.7]  # Good quality peaks')
"""
            
            self.execute(completion_injection, hidden=False)
            
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
            from polarization_ui.matplotlib_config import setup_matplotlib_style
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
    """Window showing Python examples with custom example management."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("RamanLab Analysis Examples")
        self.setModal(False)
        self.resize(900, 700)
        
        # Path for custom examples file
        from pathlib import Path
        self.custom_examples_file = Path.home() / '.ramanlab' / 'custom_examples.json'
        self.custom_examples_file.parent.mkdir(exist_ok=True)
        
        # Load custom examples
        self.custom_examples = self.load_custom_examples()
        
        # Set up UI
        self.setup_ui()
        
        # Timer for copy feedback
        self.copy_timer = QTimer()
        self.copy_timer.timeout.connect(self.show_copy_feedback)
        self.copy_timer.setSingleShot(True)
    
    def setup_ui(self):
        """Set up the examples window UI with custom examples integration."""
        layout = QVBoxLayout(self)
        
        # Header with title and add example button
        header_layout = QHBoxLayout()
        title_label = QLabel("RamanLab Analysis Examples")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin: 10px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Add example button
        add_example_btn = QPushButton("➕ Add Custom Example")
        add_example_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        add_example_btn.clicked.connect(self.show_add_example_dialog)
        header_layout.addWidget(add_example_btn)
        
        layout.addLayout(header_layout)
        
        # Tab widget for different categories
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_basic_tab(), "📊 Basic")
        self.tab_widget.addTab(self.create_peak_analysis_tab(), "🔍 Peak Analysis") 
        self.tab_widget.addTab(self.create_plotting_tab(), "📈 Plotting")
        
        # Add custom tabs for each custom category
        for category in self.custom_examples.keys():
            tab = self.create_custom_tab(category)
            self.tab_widget.addTab(tab, f"⭐ {category}")
        
        layout.addWidget(self.tab_widget)
        
        # Footer with info
        footer_label = QLabel("💡 Tip: Click ➕ to save your own code snippets for later use!")
        footer_label.setStyleSheet("color: #7f8c8d; font-style: italic; margin: 5px;")
        layout.addWidget(footer_label)
    
    def load_custom_examples(self):
        """Load custom examples from JSON file."""
        import json
        try:
            if self.custom_examples_file.exists():
                with open(self.custom_examples_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Warning: Could not load custom examples: {e}")
            return {}
    
    def save_custom_examples(self):
        """Save custom examples to JSON file."""
        import json
        try:
            with open(self.custom_examples_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_examples, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving custom examples: {e}")
            return False
    
    def add_custom_example(self, category, title, code, description=""):
        """Add a new custom example."""
        if category not in self.custom_examples:
            self.custom_examples[category] = []
        
        # Check if example already exists
        for example in self.custom_examples[category]:
            if example['title'] == title:
                # Update existing example
                example['code'] = code
                example['description'] = description
                self.save_custom_examples()
                return True
        
        # Add new example
        self.custom_examples[category].append({
            'title': title,
            'code': code,
            'description': description,
            'created': pd.Timestamp.now().isoformat()
        })
        
        if self.save_custom_examples():
            self.refresh_tabs()
            return True
        return False
    
    def delete_custom_example(self, category, title):
        """Delete a custom example."""
        if category in self.custom_examples:
            self.custom_examples[category] = [
                ex for ex in self.custom_examples[category] 
                if ex['title'] != title
            ]
            if not self.custom_examples[category]:
                del self.custom_examples[category]
            self.save_custom_examples()
            self.refresh_tabs()
    
    def show_add_example_dialog(self):
        """Show dialog to add a new custom example."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Custom Example")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Category selection
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        category_combo = QComboBox()
        category_combo.addItems(["Basic Analysis", "Peak Analysis", "Plotting", "Custom"])
        category_combo.setEditable(True)
        cat_layout.addWidget(category_combo)
        layout.addLayout(cat_layout)
        
        # Title input
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        title_edit = QLineEdit()
        title_edit.setPlaceholderText("My Analysis Example")
        title_layout.addWidget(title_edit)
        layout.addLayout(title_layout)
        
        # Description input
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel("Description (optional):"))
        desc_edit = QTextEdit()
        desc_edit.setMaximumHeight(80)
        desc_edit.setPlaceholderText("Brief description of what this example does...")
        desc_layout.addWidget(desc_edit)
        layout.addLayout(desc_layout)
        
        # Code input
        code_layout = QVBoxLayout()
        code_layout.addWidget(QLabel("Code:"))
        code_edit = QTextEdit()
        code_edit.setPlaceholderText("# Your Python code here...\nprint('Hello RamanLab!')")
        
        # Set up code editor styling
        font = QFont()
        font.setFamily('Consolas')
        font.setPointSize(10)
        code_edit.setFont(font)
        
        code_layout.addWidget(code_edit)
        layout.addLayout(code_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("💾 Save Example")
        cancel_btn = QPushButton("❌ Cancel")
        
        save_btn.clicked.connect(lambda: self.save_from_dialog(
            dialog, category_combo.currentText(), title_edit.text(), 
            code_edit.toPlainText(), desc_edit.toPlainText()
        ))
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def save_from_dialog(self, dialog, category, title, code, description):
        """Save example from dialog input."""
        if not title.strip():
            QMessageBox.warning(dialog, "Missing Title", "Please enter a title for your example.")
            return
        
        if not code.strip():
            QMessageBox.warning(dialog, "Missing Code", "Please enter some code for your example.")
            return
        
        if self.add_custom_example(category, title.strip(), code.strip(), description.strip()):
            QMessageBox.information(dialog, "Success", f"Example '{title}' saved successfully!")
            dialog.accept()
        else:
            QMessageBox.warning(dialog, "Error", "Failed to save example. Please try again.")
    
    def refresh_tabs(self):
        """Refresh all tabs to show updated custom examples."""
        # Clear and rebuild tabs
        self.tab_widget.clear()
        self.tab_widget.addTab(self.create_basic_tab(), "📊 Basic")
        self.tab_widget.addTab(self.create_peak_analysis_tab(), "🔍 Peak Analysis") 
        self.tab_widget.addTab(self.create_plotting_tab(), "📈 Plotting")
        
        # Add custom tabs for each custom category
        for category in self.custom_examples.keys():
            tab = self.create_custom_tab(category)
            self.tab_widget.addTab(tab, f"⭐ {category}")
    
    def create_custom_tab(self, category):
        """Create tab for custom examples in a specific category."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header with add/manage buttons
        header_layout = QHBoxLayout()
        header_label = QLabel(f"Custom Examples: {category}")
        header_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(header_label)
        
        header_layout.addStretch()
        
        add_btn = QPushButton("➕ Add Example")
        add_btn.clicked.connect(self.show_add_example_dialog)
        header_layout.addWidget(add_btn)
        
        layout.addLayout(header_layout)
        
        # Scroll area for examples
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Add examples for this category
        if category in self.custom_examples:
            for example in self.custom_examples[category]:
                example_widget = self.create_custom_example_widget(
                    category, example['title'], example['code'], example.get('description', '')
                )
                scroll_layout.addWidget(example_widget)
        
        # Add spacing at the end
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        return widget
    
    def create_custom_example_widget(self, category, title, code, description=""):
        """Create widget for a custom example with delete option."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 5px; margin: 2px; }")
        
        layout = QVBoxLayout(frame)
        
        # Header with title and controls
        header_layout = QHBoxLayout()
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Control buttons
        copy_btn = QPushButton("📋")
        copy_btn.setToolTip("Copy to clipboard")
        copy_btn.setMaximumWidth(30)
        copy_btn.clicked.connect(lambda: self.copy_to_clipboard(code))
        
        edit_btn = QPushButton("✏️")
        edit_btn.setToolTip("Edit example")
        edit_btn.setMaximumWidth(30)
        edit_btn.clicked.connect(lambda: self.edit_custom_example(category, title, code, description))
        
        delete_btn = QPushButton("🗑️")
        delete_btn.setToolTip("Delete example")
        delete_btn.setMaximumWidth(30)
        delete_btn.clicked.connect(lambda: self.confirm_delete_example(category, title))
        
        header_layout.addWidget(copy_btn)
        header_layout.addWidget(edit_btn)
        header_layout.addWidget(delete_btn)
        
        layout.addLayout(header_layout)
        
        # Description if available
        if description:
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #7f8c8d; font-style: italic; margin-bottom: 5px;")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
        
        # Code display
        code_display = QTextEdit()
        code_display.setPlainText(code)
        code_display.setReadOnly(True)
        code_display.setMaximumHeight(150)
        
        # Set up font
        font = QFont()
        font.setFamily('Consolas')
        font.setPointSize(9)
        code_display.setFont(font)
        code_display.setStyleSheet("background-color: #f8f9fa; border: 1px solid #e9ecef;")
        
        layout.addWidget(code_display)
        
        return frame
    
    def edit_custom_example(self, category, title, code, description):
        """Edit an existing custom example."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Example: {title}")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Title (read-only for editing)
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        title_edit = QLineEdit(title)
        title_layout.addWidget(title_edit)
        layout.addLayout(title_layout)
        
        # Description
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        desc_edit = QTextEdit()
        desc_edit.setPlainText(description)
        desc_edit.setMaximumHeight(80)
        desc_layout.addWidget(desc_edit)
        layout.addLayout(desc_layout)
        
        # Code
        code_layout = QVBoxLayout()
        code_layout.addWidget(QLabel("Code:"))
        code_edit = QTextEdit()
        code_edit.setPlainText(code)
        
        font = QFont()
        font.setFamily('Consolas')
        font.setPointSize(10)
        code_edit.setFont(font)
        
        code_layout.addWidget(code_edit)
        layout.addLayout(code_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("💾 Save Changes")
        cancel_btn = QPushButton("❌ Cancel")
        
        def save_changes():
            new_title = title_edit.text().strip()
            new_code = code_edit.toPlainText().strip()
            new_desc = desc_edit.toPlainText().strip()
            
            if not new_title or not new_code:
                QMessageBox.warning(dialog, "Missing Information", "Title and code cannot be empty.")
                return
            
            # Delete old example if title changed
            if new_title != title:
                self.delete_custom_example(category, title)
            
            if self.add_custom_example(category, new_title, new_code, new_desc):
                QMessageBox.information(dialog, "Success", "Example updated successfully!")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Error", "Failed to update example.")
        
        save_btn.clicked.connect(save_changes)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def confirm_delete_example(self, category, title):
        """Confirm deletion of custom example."""
        reply = QMessageBox.question(
            self, "Delete Example", 
            f"Are you sure you want to delete the example '{title}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.delete_custom_example(category, title)
            QMessageBox.information(self, "Deleted", f"Example '{title}' has been deleted.")
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard."""
        try:
            import sys
            if hasattr(sys, 'platform'):
                from PySide6.QtGui import QClipboard
                clipboard = QApplication.clipboard()
                clipboard.setText(text)
                self.copy_timer.start(2000)  # Show feedback for 2 seconds
        except Exception as e:
            print(f"Could not copy to clipboard: {e}")
    
    def show_copy_feedback(self):
        """Show brief feedback that text was copied."""
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage("Code copied to clipboard!", 2000)
        else:
            print("✓ Code copied to clipboard!")
    
    def create_example_widget(self, title, code):
        """Create a widget for displaying built-in examples (maintains compatibility)."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 5px; margin: 2px; }")
        
        layout = QVBoxLayout(frame)
        
        # Header with title and copy button
        header_layout = QHBoxLayout()
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Copy button
        copy_btn = QPushButton("📋 Copy")
        copy_btn.clicked.connect(lambda: self.copy_to_clipboard(code))
        copy_btn.setMaximumWidth(70)
        
        # Quick save button for built-in examples
        save_btn = QPushButton("💾 Save")
        save_btn.setToolTip("Save as custom example")
        save_btn.clicked.connect(lambda: self.quick_save_example(title, code))
        save_btn.setMaximumWidth(70)
        
        header_layout.addWidget(save_btn)
        header_layout.addWidget(copy_btn)
        
        layout.addLayout(header_layout)
        
        # Code display
        code_display = QTextEdit()
        code_display.setPlainText(code)
        code_display.setReadOnly(True)
        code_display.setMaximumHeight(150)
        
        # Set up font
        font = QFont()
        font.setFamily('Consolas')
        font.setPointSize(9)
        code_display.setFont(font)
        code_display.setStyleSheet("background-color: #f8f9fa; border: 1px solid #e9ecef;")
        
        layout.addWidget(code_display)
        
        return frame
    
    def quick_save_example(self, title, code):
        """Quick save a built-in example as a custom example."""
        # Show simplified save dialog
        category, ok = QInputDialog.getItem(
            self, "Save Example", "Choose category for this example:",
            ["Basic Analysis", "Peak Analysis", "Plotting", "My Examples"], 
            0, True
        )
        
        if ok and category:
            new_title, ok2 = QInputDialog.getText(
                self, "Example Title", "Enter a title for your saved example:",
                text=f"My {title}"
            )
            
            if ok2 and new_title.strip():
                if self.add_custom_example(category, new_title.strip(), code):
                    QMessageBox.information(self, "Saved!", f"Example saved as '{new_title}' in {category}")
                else:
                    QMessageBox.warning(self, "Error", "Could not save example. Please try again.")
    
    def closeEvent(self, event):
        """Handle close event by hiding instead of closing."""
        event.ignore()
        self.hide()

    def create_basic_tab(self):
        """Create basic data exploration tab with safe examples."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        examples = [
            ("Check available data (Safe)", """
# Safe data availability check
import pandas as pd
import numpy as np

print("=== RamanLab Data Availability Check ===")

# Check if main variables exist
variables_to_check = ['summary_df', 'peaks_df', 'spectra_dict', 'batch_data']
available_vars = []

for var_name in variables_to_check:
    if var_name in globals():
        available_vars.append(var_name)
        var_data = globals()[var_name]
        if hasattr(var_data, '__len__'):
            print(f"✓ {var_name}: Available ({len(var_data)} items)")
        else:
            print(f"✓ {var_name}: Available")
    else:
        print(f"✗ {var_name}: Not available")

if 'summary_df' in available_vars:
    print(f"\\nSummary DataFrame shape: {summary_df.shape}")
    print(f"Summary columns: {list(summary_df.columns)}")

if 'peaks_df' in available_vars:
    print(f"\\nPeaks DataFrame shape: {peaks_df.shape}")
    print(f"Peaks columns: {list(peaks_df.columns)}")

if 'spectra_dict' in available_vars:
    print(f"\\nSpectra dictionary: {len(spectra_dict)} spectra available")
    if len(spectra_dict) > 0:
        first_key = list(spectra_dict.keys())[0]
        first_spectrum = spectra_dict[first_key]
        print(f"First spectrum keys: {list(first_spectrum.keys())}")

print("\\n=== Data Check Complete ===")
"""),

            ("View data safely", """
# Safe data viewing with error handling
try:
    if 'summary_df' in globals() and len(summary_df) > 0:
        print("Summary DataFrame (first 5 rows):")
        print(summary_df.head())
        print(f"\\nShape: {summary_df.shape}")
    else:
        print("Summary DataFrame not available or empty")
        
    if 'peaks_df' in globals() and len(peaks_df) > 0:
        print("\\nPeaks DataFrame (first 5 rows):")
        print(peaks_df.head())
        print(f"\\nShape: {peaks_df.shape}")
    else:
        print("\\nPeaks DataFrame not available or empty")
        
    if 'spectra_dict' in globals() and len(spectra_dict) > 0:
        print(f"\\nSpectra available: {len(spectra_dict)}")
        print("First 3 spectrum names:")
        for i, name in enumerate(list(spectra_dict.keys())[:3]):
            print(f"  {i+1}. {name}")
    else:
        print("\\nNo spectra dictionary available")
        
except Exception as e:
    print(f"Error viewing data: {e}")
"""),

            ("Safe statistics with smart columns", """
# Safe statistics with intelligent column detection
def get_safe_statistics():
    try:
        if 'summary_df' not in globals() or len(summary_df) == 0:
            print("No summary data available")
            return
            
        print("=== RamanLab Statistics Summary ===")
        print(f"Total files processed: {len(summary_df)}")
        
        # Smart column detection for peaks
        peak_cols = [col for col in summary_df.columns if 'peak' in col.lower() and ('count' in col.lower() or 'n_' in col.lower())]
        if not peak_cols:
            peak_cols = [col for col in summary_df.columns if col in ['n_peaks', 'num_peaks', 'peak_count']]
        
        if peak_cols:
            peak_col = peak_cols[0]
            total_peaks = summary_df[peak_col].sum()
            avg_peaks = summary_df[peak_col].mean()
            print(f"Total peaks detected: {total_peaks}")
            print(f"Average peaks per spectrum: {avg_peaks:.1f}")
            
            # Success rate
            success_count = len(summary_df[summary_df[peak_col] > 0])
            success_rate = (success_count / len(summary_df)) * 100
            print(f"Success rate: {success_rate:.1f}% ({success_count}/{len(summary_df)} files)")
        else:
            print("Peak count information not available")
            
        # Smart R² detection
        r2_cols = [col for col in summary_df.columns if 'r2' in col.lower() or 'r_squared' in col.lower()]
        if r2_cols:
            r2_col = r2_cols[0]
            avg_r2 = summary_df[r2_col].mean()
            good_fits = len(summary_df[summary_df[r2_col] >= 0.9])
            print(f"Average R²: {avg_r2:.3f}")
            print(f"High quality fits (R² ≥ 0.9): {good_fits}")
        else:
            print("R² information not available")
            
        print("\\n" + "="*40)
        
    except Exception as e:
        print(f"Error generating statistics: {e}")

# Run the safe statistics
get_safe_statistics()
"""),

            ("Safe spectrum plotting", """
# Safe single spectrum plot
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum_safely(spectrum_index=0):
    try:
        if 'spectra_dict' not in globals() or len(spectra_dict) == 0:
            print("No spectra data available for plotting")
            return
            
        filenames = list(spectra_dict.keys())
        if spectrum_index >= len(filenames):
            print(f"Index {spectrum_index} too high. Available: 0-{len(filenames)-1}")
            return
            
        filename = filenames[spectrum_index]
        spectrum_data = spectra_dict[filename]
        
        # Check for required data
        if 'wavenumbers' not in spectrum_data or 'intensities' not in spectrum_data:
            print(f"Spectrum {filename} missing wavenumbers or intensities")
            return
            
        wavenumbers = np.array(spectrum_data['wavenumbers'])
        intensities = np.array(spectrum_data['intensities'])
        
        # Validate data
        if len(wavenumbers) == 0 or len(intensities) == 0:
            print(f"Empty data arrays for {filename}")
            return
            
        if len(wavenumbers) != len(intensities):
            print(f"Mismatched array lengths")
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(wavenumbers, intensities, 'b-', linewidth=1)
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title(f'Raman Spectrum: {filename}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Successfully plotted: {filename}")
        
    except Exception as e:
        print(f"Error plotting spectrum: {e}")

# Plot the first available spectrum
plot_spectrum_safely(0)
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

for mineral, ranges in peak_ranges.items():
    print(f"\\n{mineral} peaks:")
    for min_pos, max_pos in ranges:
        mineral_peaks = peaks_df[(peaks_df['position'] >= min_pos) & (peaks_df['position'] <= max_pos)]
        print(f"  {min_pos}-{max_pos} cm⁻¹: {len(mineral_peaks)} peaks found")
        if len(mineral_peaks) > 0:
            print(f"    Files: {mineral_peaks['filename'].unique()[:3]}")  # First 3 files
"""),
            
            ("Peak intensity analysis", """
# Analyze peak intensities
print("Peak intensity statistics:")
print(f"Mean height: {peaks_df['height'].mean():.2f}")
print(f"Max height: {peaks_df['height'].max():.2f}")
print(f"Min height: {peaks_df['height'].min():.2f}")

# Find strongest peaks
strongest_peaks = peaks_df.nlargest(10, 'height')
print("\\nStrongest peaks:")
print(strongest_peaks[['filename', 'position', 'height', 'width']])
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
print("Peak fitting quality:")
good_peaks = peaks_df[peaks_df['r2'] >= 0.95]
poor_peaks = peaks_df[peaks_df['r2'] < 0.8]

print(f"High quality peaks (R² ≥ 0.95): {len(good_peaks)}")
print(f"Poor quality peaks (R² < 0.8): {len(poor_peaks)}")

# Show worst fitting peaks
worst_peaks = peaks_df.nsmallest(5, 'r2')
print("\\nWorst fitting peaks:")
print(worst_peaks[['filename', 'position', 'height', 'r2']])
"""),
            
            ("Peak clustering by position", """
# Group peaks by position to find common frequencies
import numpy as np

# Create position bins (every 10 cm⁻¹)
bin_width = 10
min_pos = peaks_df['position'].min()
max_pos = peaks_df['position'].max()
bins = np.arange(min_pos, max_pos + bin_width, bin_width)

# Count peaks in each bin
peak_counts, bin_edges = np.histogram(peaks_df['position'], bins=bins)

# Find most common peak regions
common_regions = []
for i, count in enumerate(peak_counts):
    if count > 5:  # More than 5 peaks in this region
        common_regions.append((bin_edges[i], bin_edges[i+1], count))

print("Common peak regions:")
for start, end, count in sorted(common_regions, key=lambda x: x[2], reverse=True):
    print(f"  {start:.0f}-{end:.0f} cm⁻¹: {count} peaks")
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
plt.figure(figsize=(12, 6))
plt.hist(peaks_df['position'], bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Peak Position (cm⁻¹)')
plt.ylabel('Frequency')
plt.title('Distribution of Peak Positions')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
            
            ("Plot peak intensity vs position", """
# Scatter plot of peak intensity vs position
plt.figure(figsize=(12, 6))
plt.scatter(peaks_df['position'], peaks_df['height'], alpha=0.6, s=30)
plt.xlabel('Peak Position (cm⁻¹)')
plt.ylabel('Peak Height')
plt.title('Peak Intensity vs Position')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
            
            ("Create a heat map of peak occurrences", """
# Create heatmap showing peak occurrence across samples
import numpy as np

# Get unique filenames and create position bins
filenames = summary_df['filename'].unique()[:20]  # First 20 files
pos_bins = np.arange(200, 1800, 20)  # 20 cm⁻¹ bins from 200-1800

# Create matrix for heatmap
heatmap_data = np.zeros((len(filenames), len(pos_bins)-1))

for i, filename in enumerate(filenames):
    file_peaks = peaks_df[peaks_df['filename'] == filename]
    if len(file_peaks) > 0:
        hist, _ = np.histogram(file_peaks['position'], bins=pos_bins)
        heatmap_data[i, :] = hist

# Plot heatmap
plt.figure(figsize=(15, 8))
plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Peak Count')
plt.xlabel('Wavenumber Bins')
plt.ylabel('Sample Index')
plt.title('Peak Occurrence Heatmap')

# Add wavenumber labels
tick_positions = range(0, len(pos_bins)-1, 10)
tick_labels = [f"{pos_bins[i]:.0f}" for i in tick_positions]
plt.xticks(tick_positions, tick_labels, rotation=45)

plt.tight_layout()
plt.show()
"""),
            
            ("Plot R² quality assessment", """
# Plot R² distribution and quality assessment
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# R² histogram
ax1.hist(peaks_df['r2'].dropna(), bins=30, alpha=0.7, edgecolor='black')
ax1.set_xlabel('R² Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Peak Fitting Quality Distribution')
ax1.grid(True, alpha=0.3)

# R² vs peak position
ax2.scatter(peaks_df['position'], peaks_df['r2'], alpha=0.6, s=30)
ax2.set_xlabel('Peak Position (cm⁻¹)')
ax2.set_ylabel('R² Value')
ax2.set_title('Fitting Quality vs Peak Position')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
        ]
        
        for title, code in examples:
            scroll_layout.addWidget(self.create_example_widget(title, code))
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget


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