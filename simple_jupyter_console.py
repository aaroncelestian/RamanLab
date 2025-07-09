#!/usr/bin/env python3
"""
Simplified Jupyter Console for RamanLab
=======================================

A simplified version of the Jupyter console that should work more reliably.
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
                               QMessageBox, QTextEdit, QGroupBox, QDialog, QTabWidget, 
                               QScrollArea, QSplitter, QFrame)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

# Try to import Jupyter console components
try:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget
    from qtconsole.manager import QtKernelManager
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    print("Warning: Jupyter console components not available.")

class SimpleJupyterConsole(QMainWindow):
    """Simplified Jupyter console for RamanLab."""
    
    def __init__(self, pickle_file=None):
        super().__init__()
        self.pickle_file = pickle_file
        self.batch_data = None
        self.summary_df = None
        self.peaks_df = None
        self.spectra_dict = None
        
        self.setup_ui()
        
        # Load data if provided
        if pickle_file:
            self.load_pickle_file(pickle_file)
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("RamanLab Jupyter Console (Simplified)")
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Create console area
        if JUPYTER_AVAILABLE:
            try:
                self.console_widget = self.create_jupyter_console()
                layout.addWidget(self.console_widget)
                self.console_type = "jupyter"
            except Exception as e:
                print(f"Failed to create Jupyter console: {e}")
                self.console_widget = self.create_fallback_console()
                layout.addWidget(self.console_widget)
                self.console_type = "fallback"
        else:
            self.console_widget = self.create_fallback_console()
            layout.addWidget(self.console_widget)
            self.console_type = "fallback"
    
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
        
        # Clear Console button
        clear_btn = QPushButton("🧹 Clear Console")
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
    
    def create_jupyter_console(self):
        """Create a Jupyter console widget."""
        # Create a simple RichJupyterWidget without custom kernel setup
        console = RichJupyterWidget()
        
        # Configure appearance
        console.set_default_style(colors='linux')
        console.syntax_style = 'monokai'
        
        # Set font - use cross-platform fallback
        font = QFont()
        font.setFamily('Monaco')  # Better for Mac
        if not font.exactMatch():
            font.setFamily('Courier New')  # Windows fallback
        if not font.exactMatch():
            font.setFamily('monospace')  # Generic fallback
        font.setPointSize(10)
        console.setFont(font)
        
        return console
    
    def create_fallback_console(self):
        """Create a fallback text console."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Warning
        warning = QLabel("⚠️ Using Fallback Console")
        warning.setStyleSheet("color: orange; font-weight: bold;")
        layout.addWidget(warning)
        
        # Text area
        self.text_console = QTextEdit()
        self.text_console.setReadOnly(True)
        
        # Set font - use cross-platform fallback
        font = QFont()
        font.setFamily('Monaco')  # Better for Mac
        if not font.exactMatch():
            font.setFamily('Courier New')  # Windows fallback
        if not font.exactMatch():
            font.setFamily('monospace')  # Generic fallback
        font.setPointSize(10)
        self.text_console.setFont(font)
        
        self.text_console.setPlainText("""
RamanLab Console (Fallback Mode)
==============================

This is a simplified console. Load a pickle file to see data summary.

For full Jupyter functionality, install:
pip install qtconsole jupyter-client ipykernel
""")
        
        layout.addWidget(self.text_console)
        return widget
    
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
            # Load data
            with open(file_path, 'rb') as f:
                self.batch_data = pickle.load(f)
            
            # Convert to pandas
            self.summary_df, self.peaks_df, self.spectra_dict = \
                self.convert_to_pandas(self.batch_data)
            
            # Update UI
            self.pickle_file = file_path
            self.file_label.setText(f"Loaded: {Path(file_path).name}")
            self.refresh_btn.setEnabled(True)
            
            # Show data
            self.refresh_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load pickle file:\n{str(e)}")
    
    def convert_to_pandas(self, batch_data):
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
    
    def refresh_data(self):
        """Refresh data display."""
        if not self.batch_data:
            return
        
        if self.console_type == "jupyter":
            self.inject_jupyter_data()
        else:
            self.show_fallback_data()
    
    def inject_jupyter_data(self):
        """Inject data into Jupyter console."""
        if not hasattr(self.console_widget, 'kernel_client') or not self.console_widget.kernel_client:
            # Start a kernel if not available
            try:
                from qtconsole.manager import QtKernelManager
                km = QtKernelManager()
                km.start_kernel()
                kc = km.client()
                kc.start_channels()
                self.console_widget.kernel_client = kc
                
                # Wait a moment then inject data
                QTimer.singleShot(1000, self._do_inject_data)
            except Exception as e:
                print(f"Failed to start kernel: {e}")
        else:
            self._do_inject_data()
    
    def _do_inject_data(self):
        """Actually inject the data."""
        try:
            # Create data injection code
            code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load summary data
summary_data = {self.summary_df.to_dict('records')}
summary_df = pd.DataFrame(summary_data)

# Load peaks data  
peaks_data = {self.peaks_df.to_dict('records')}
peaks_df = pd.DataFrame(peaks_data)

# Create spectra dictionary
spectra_dict = {{}}
"""
            
            # Add spectral data
            for filename, spectrum_data in self.spectra_dict.items():
                code += f"""
spectra_dict['{filename}'] = {{
    'wavenumbers': np.array({spectrum_data['wavenumbers'].tolist()}),
    'intensities': np.array({spectrum_data['intensities'].tolist()}),
    'background': np.array({spectrum_data['background'].tolist() if len(spectrum_data['background']) > 0 else []}),
}}
"""
            
            code += f"""
print("✅ RamanLab data loaded successfully!")
print(f"📈 {{len(summary_df)}} spectra processed")
print(f"🔍 {{len(peaks_df)}} peaks detected")
print(f"📊 {{len(spectra_dict)}} spectra with full data")
print()
print("Available variables:")
print("• summary_df    - Summary statistics DataFrame")
print("• peaks_df      - Peak parameters DataFrame") 
print("• spectra_dict  - Dictionary of spectral data")
print()
print("Try: summary_df.head() or peaks_df.describe()")
"""
            
            # Execute the code
            self.console_widget.execute(code, hidden=False)
            
        except Exception as e:
            print(f"Failed to inject data: {e}")
    
    def show_fallback_data(self):
        """Show data in fallback console."""
        text = f"""
RamanLab Data Summary
===================

File: {Path(self.pickle_file).name if self.pickle_file else 'Unknown'}
Spectra processed: {len(self.summary_df)}
Total peaks detected: {len(self.peaks_df)}
Spectra with full data: {len(self.spectra_dict)}

Summary DataFrame shape: {self.summary_df.shape}
Peaks DataFrame shape: {self.peaks_df.shape}

Summary statistics:
{self.summary_df.describe()}

Peak position statistics:
{self.peaks_df['position'].describe() if 'position' in self.peaks_df.columns else 'No position data'}

For full analysis capabilities, install Jupyter components:
pip install qtconsole jupyter-client ipykernel
"""
        
        self.text_console.setPlainText(text)
    
    def clear_console(self):
        """Clear the console."""
        if self.console_type == "jupyter":
            try:
                self.console_widget.clear()
            except:
                # Fallback to executing clear command
                self.console_widget.execute("clear", hidden=False)
        else:
            self.text_console.clear()
    
    def show_examples_window(self):
        """Show the Python examples window."""
        if not hasattr(self, 'examples_window') or self.examples_window is None:
            self.examples_window = PythonExamplesWindow(self)
        
        self.examples_window.show()
        self.examples_window.raise_()
        self.examples_window.activateWindow()


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
            from core.matplotlib_config import setup_matplotlib
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
"""),

            ("Safe data export", """
# Safe data export with validation
def export_data_safely():
    try:
        import pandas as pd
        from datetime import datetime
        
        # Create timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("=== Safe Data Export ===")
        
        # Export summary data
        if 'summary_df' in globals() and len(summary_df) > 0:
            summary_filename = f"ramanlab_summary_{timestamp}.csv"
            summary_df.to_csv(summary_filename, index=False)
            print(f"✓ Exported summary data: {summary_filename} ({len(summary_df)} rows)")
        else:
            print("✗ No summary data to export")
            
        # Export peaks data
        if 'peaks_df' in globals() and len(peaks_df) > 0:
            peaks_filename = f"ramanlab_peaks_{timestamp}.csv"
            peaks_df.to_csv(peaks_filename, index=False)
            print(f"✓ Exported peaks data: {peaks_filename} ({len(peaks_df)} rows)")
        else:
            print("✗ No peaks data to export")
            
        print(f"\\nFiles saved with timestamp: {timestamp}")
        
    except Exception as e:
        print(f"Error during export: {e}")

# Run safe export
export_data_safely()
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
            
            ("Batch quality control", """
# Quality control metrics for batch processing
print("Quality Control Metrics:")

# Calculate success rate
total_files = len(summary_df)
successful_files = len(summary_df[summary_df['n_peaks'] > 0])
success_rate = (successful_files / total_files) * 100

print(f"Success rate: {success_rate:.1f}% ({successful_files}/{total_files})")

# Peak detection consistency
peak_std = summary_df['n_peaks'].std()
peak_mean = summary_df['n_peaks'].mean()
cv = (peak_std / peak_mean) * 100  # Coefficient of variation

print(f"Peak detection consistency (CV): {cv:.1f}%")

# Fitting quality consistency
r2_mean = summary_df['total_r2'].mean()
r2_std = summary_df['total_r2'].std()

print(f"Average fitting quality: {r2_mean:.3f} ± {r2_std:.3f}")
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
    console = SimpleJupyterConsole(pickle_file)
    console.show()
    console.raise_()  # Bring window to front
    console.activateWindow()  # Give it focus
    
    print(f"RamanLab Jupyter Console started successfully!")
    print(f"Window geometry: {console.geometry()}")
    print(f"Window visible: {console.isVisible()}")
    print("Look for the '📚 Python Examples' button in the control panel!")
    print("The button should be GREEN and located at the top of the window.")
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 