import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend for PySide6 compatibility
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QWidget, QPushButton, QLabel, QListWidget, QListWidgetItem,
                              QSplitter, QTextEdit, QSpinBox, QCheckBox, QGroupBox,
                              QMessageBox, QProgressDialog, QTabWidget, QTableWidget,
                              QTableWidgetItem, QHeaderView, QSlider, QDoubleSpinBox)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer
from PySide6.QtGui import QFont, QPalette, QColor
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.metrics import r2_score
import warnings
from typing import Dict, List, Tuple, Optional
import time
import json

# Import RamanLab utilities
try:
    from pkl_utils import load_raman_database
    RAMANLAB_AVAILABLE = True
except ImportError:
    RAMANLAB_AVAILABLE = False

# Try to import matplotlib config (required for RamanLab styling)
try:
    from core.matplotlib_config import (
        configure_compact_ui,
        apply_theme,
        CompactNavigationToolbar,
        get_toolbar_class
    )
    MATPLOTLIB_CONFIG_AVAILABLE = True
    print("✅ Successfully imported RamanLab matplotlib configuration from core")
except ImportError as e:
    MATPLOTLIB_CONFIG_AVAILABLE = False
    # Fallback to standard toolbar
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as CompactNavigationToolbar
    print(f"⚠️ Warning: matplotlib_config not available - using default styling. Error: {e}")

class PeakSelectionCanvas(FigureCanvas):
    """Interactive matplotlib canvas for peak selection and spectrum overlay."""
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(12, 8))
        super().__init__(self.figure)
        self.setParent(parent)
        
        # Matplotlib configuration is applied globally in __init__ - no need to apply again here
        
        self.selected_peaks = []
        self.user_wavenumbers = None
        self.user_spectrum = None
        self.reference_wavenumbers = None
        self.reference_spectrum = None
        self.synthetic_spectrum = None
        self.residual_spectrum = None
        
        # Create subplots
        self.setup_plots()
        
        # Connect click events
        self.mpl_connect('button_press_event', self.on_click)
        
    def setup_plots(self):
        """Setup the plot layout with multiple subplots."""
        self.figure.clear()
        
        # Create 2x2 subplot layout
        self.ax_overlay = self.figure.add_subplot(2, 2, 1)
        self.ax_residual = self.figure.add_subplot(2, 2, 2)
        self.ax_synthetic = self.figure.add_subplot(2, 2, 3)
        self.ax_analysis = self.figure.add_subplot(2, 2, 4)
        
        # Set titles
        self.ax_overlay.set_title('Spectrum Overlay & Peak Selection', fontweight='bold')
        self.ax_residual.set_title('Current Residual', fontweight='bold')
        self.ax_synthetic.set_title('Cumulative Synthetic Spectrum', fontweight='bold')
        self.ax_analysis.set_title('Component Analysis', fontweight='bold')
        
        # Set labels
        for ax in [self.ax_overlay, self.ax_residual, self.ax_synthetic]:
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity (normalized)')
            ax.grid(True, alpha=0.3)
        
        self.ax_analysis.set_xlabel('Component')
        self.ax_analysis.set_ylabel('Contribution')
        self.ax_analysis.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.draw()
    
    def update_overlay(self, user_wavenumbers, user_spectrum, 
                      reference_wavenumbers, reference_spectrum, 
                      reference_name="Reference"):
        """Update the overlay plot with user and reference spectra."""
        self.user_wavenumbers = user_wavenumbers
        self.user_spectrum = user_spectrum / np.max(user_spectrum)  # Normalize
        self.reference_wavenumbers = reference_wavenumbers
        self.reference_spectrum = reference_spectrum / np.max(reference_spectrum)  # Normalize
        
        # Clear overlay plot
        self.ax_overlay.clear()
        self.ax_overlay.set_title('Spectrum Overlay & Peak Selection', fontweight='bold')
        self.ax_overlay.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_overlay.set_ylabel('Intensity (normalized)')
        self.ax_overlay.grid(True, alpha=0.3)
        
        # Plot spectra
        self.ax_overlay.plot(user_wavenumbers, self.user_spectrum, 'k-', 
                           linewidth=2, label='User Data', alpha=0.8)
        self.ax_overlay.plot(reference_wavenumbers, self.reference_spectrum, 'r-', 
                           linewidth=1.5, label=f'{reference_name}', alpha=0.7)
        
        # Mark selected peaks
        for peak_wn in self.selected_peaks:
            self.ax_overlay.axvline(peak_wn, color='blue', linestyle='--', 
                                  alpha=0.7, linewidth=1)
            
        self.ax_overlay.legend()
        self.draw()
    
    def on_click(self, event):
        """Handle mouse clicks for peak selection."""
        if event.inaxes == self.ax_overlay and event.button == 1:  # Left click
            if self.user_wavenumbers is not None:
                # Find nearest wavenumber
                click_wn = event.xdata
                if click_wn is not None:
                    # Find closest wavenumber in user data
                    idx = np.argmin(np.abs(self.user_wavenumbers - click_wn))
                    selected_wn = self.user_wavenumbers[idx]
                    
                    if selected_wn not in self.selected_peaks:
                        self.selected_peaks.append(selected_wn)
                        print(f"🔧 DEBUG: Peak selected at {selected_wn:.1f} cm⁻¹")
                        print(f"🔧 DEBUG: Total selected peaks: {len(self.selected_peaks)}")
                        
                        # Re-draw overlay with new peak marked
                        if self.reference_wavenumbers is not None:
                            self.update_overlay(self.user_wavenumbers, self.user_spectrum * np.max(self.user_spectrum),
                                              self.reference_wavenumbers, self.reference_spectrum * np.max(self.reference_spectrum))
                        else:
                            self.display_user_spectrum()
                        
                        # Notify main window of peak selection
                        if hasattr(self, 'main_window') and hasattr(self.main_window, 'on_peak_selected'):
                            print(f"🔧 DEBUG: Calling main_window.on_peak_selected({selected_wn:.1f})")
                            self.main_window.on_peak_selected(selected_wn)
                        else:
                            print(f"🔧 DEBUG: Main window reference not found or no on_peak_selected method!")
                            print(f"🔧 DEBUG: main_window exists: {hasattr(self, 'main_window')}")
                            if hasattr(self, 'main_window'):
                                print(f"🔧 DEBUG: main_window type: {type(self.main_window)}")
                                print(f"🔧 DEBUG: has on_peak_selected: {hasattr(self.main_window, 'on_peak_selected')}")
    
    def clear_selected_peaks(self):
        """Clear all selected peaks."""
        self.selected_peaks = []
        if self.user_wavenumbers is not None:
            if self.reference_wavenumbers is not None:
                self.update_overlay(self.user_wavenumbers, self.user_spectrum * np.max(self.user_spectrum),
                                  self.reference_wavenumbers, self.reference_spectrum * np.max(self.reference_spectrum))
            else:
                self.display_user_spectrum()
    
    def display_user_spectrum(self):
        """Display only the user spectrum without reference."""
        if self.user_wavenumbers is None or self.user_spectrum is None:
            return
            
        # Normalize spectrum
        normalized_spectrum = self.user_spectrum / np.max(self.user_spectrum)
        
        # Clear overlay plot
        self.ax_overlay.clear()
        self.ax_overlay.set_title('Spectrum Overlay & Peak Selection', fontweight='bold')
        self.ax_overlay.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_overlay.set_ylabel('Intensity (normalized)')
        self.ax_overlay.grid(True, alpha=0.3)
        
        # Plot user spectrum
        self.ax_overlay.plot(self.user_wavenumbers, normalized_spectrum, 'k-', 
                           linewidth=2, label='User Data', alpha=0.8)
        
        # Mark selected peaks
        for peak_wn in self.selected_peaks:
            self.ax_overlay.axvline(peak_wn, color='blue', linestyle='--', 
                                  alpha=0.7, linewidth=1)
            
        self.ax_overlay.legend()
        self.draw()
    
    def update_residual(self, residual_spectrum):
        """Update the residual plot."""
        self.residual_spectrum = residual_spectrum
        
        self.ax_residual.clear()
        self.ax_residual.set_title('Current Residual', fontweight='bold')
        self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_residual.set_ylabel('Intensity')
        self.ax_residual.grid(True, alpha=0.3)
        
        if self.user_wavenumbers is not None:
            self.ax_residual.plot(self.user_wavenumbers, residual_spectrum, 'g-', 
                                linewidth=1.5, label='Residual')
            self.ax_residual.legend()
        
        self.draw()
    
    def update_synthetic(self, synthetic_spectrum):
        """Update the synthetic spectrum plot."""
        self.synthetic_spectrum = synthetic_spectrum
        
        self.ax_synthetic.clear()
        self.ax_synthetic.set_title('Cumulative Synthetic Spectrum', fontweight='bold')
        self.ax_synthetic.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_synthetic.set_ylabel('Intensity')
        self.ax_synthetic.grid(True, alpha=0.3)
        
        if self.user_wavenumbers is not None:
            self.ax_synthetic.plot(self.user_wavenumbers, self.user_spectrum, 'k-', 
                                 linewidth=2, label='Original', alpha=0.6)
            self.ax_synthetic.plot(self.user_wavenumbers, synthetic_spectrum, 'b-', 
                                 linewidth=2, label='Synthetic', alpha=0.8)
            self.ax_synthetic.legend()
        
        self.draw()
    
    def update_component_analysis(self, iteration_data, synthetic_components=None):
        """Update the component analysis plot with pie chart of relative contributions."""
        self.ax_analysis.clear()
        self.ax_analysis.set_title('Mineral Contributions', fontweight='bold', fontsize=12)
        
        if len(iteration_data) == 0:
            self.ax_analysis.text(0.5, 0.5, 'No components fitted yet\nClick peaks to start analysis', 
                                ha='center', va='center', transform=self.ax_analysis.transAxes,
                                fontsize=11, alpha=0.6)
            self.ax_analysis.axis('off')  # Hide axes for cleaner look
            self.draw()
            return
        
        if synthetic_components is None or len(synthetic_components) == 0:
            self.ax_analysis.text(0.5, 0.5, 'Component data not available', 
                                ha='center', va='center', transform=self.ax_analysis.transAxes,
                                fontsize=11, alpha=0.6)
            self.ax_analysis.axis('off')
            self.draw()
            return
        
        # Calculate relative contributions based on integrated intensity
        component_names = []
        relative_contributions = []
        # Professional color palette optimized for scientific visualization
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        total_contribution = 0
        for component in synthetic_components:
            # Calculate integrated intensity (area under curve) for this component
            fitted_spectrum = component['fitted_spectrum']
            integrated_intensity = np.trapz(fitted_spectrum, dx=1.0)  # Simple integration
            relative_contributions.append(integrated_intensity)
            total_contribution += integrated_intensity
            
            # Prepare display name
            mineral_name = component['mineral']
            if len(mineral_name) > 20:
                display_name = mineral_name[:17] + '...'
            else:
                display_name = mineral_name
            component_names.append(display_name)
        
        # Normalize to percentages
        if total_contribution > 0:
            percentages = [(contrib / total_contribution) * 100 for contrib in relative_contributions]
        else:
            percentages = [100.0 / len(relative_contributions)] * len(relative_contributions)
        
        # Create pie chart with RamanLab styling
        wedges, texts, autotexts = self.ax_analysis.pie(
            percentages, 
            labels=component_names, 
            colors=colors[:len(component_names)],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 9, 'fontweight': 'bold', 'fontfamily': 'Arial'},
            pctdistance=0.85,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
        )
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Add center circle for donut chart effect (more modern look)
        centre_circle = plt.Circle((0,0), 0.50, fc='white', linewidth=2, edgecolor='lightgray')
        self.ax_analysis.add_artist(centre_circle)
        
        # Add summary text in center
        total_r_sq = iteration_data[-1]['r_squared'] if iteration_data else 0
        total_peaks = sum(len(data['peaks']) for data in iteration_data)
        center_text = f"Total R²\n{total_r_sq:.3f}\n\n{total_peaks} peaks"
        self.ax_analysis.text(0, 0, center_text, ha='center', va='center', 
                             fontsize=10, fontweight='bold', color='darkslategray')
        
        # Ensure equal aspect ratio for circular pie
        self.ax_analysis.axis('equal')
        
        # Add subtitle with component count
        component_count = len(synthetic_components)
        subtitle = f"{component_count} Component{'s' if component_count != 1 else ''} Fitted"
        self.ax_analysis.set_title(f'Mineral Contributions\n{subtitle}', 
                                 fontweight='bold', fontsize=11, pad=20)
        
        self.draw()

class DatabaseSearchWorker(QObject):
    """Worker thread for database searching."""
    
    search_completed = Signal(list)
    progress_updated = Signal(int)
    
    def __init__(self, database, spectrum, wavenumbers, top_n=10):
        super().__init__()
        self.database = database
        self.spectrum = spectrum
        self.wavenumbers = wavenumbers
        self.top_n = top_n
    
    def search_database(self):
        """Search database for top matches."""
        matches = []
        total_minerals = len(self.database)
        
        for i, (mineral_name, mineral_data) in enumerate(self.database.items()):
            try:
                # Interpolate reference to match experimental wavenumbers
                ref_wavenumbers = mineral_data['wavenumbers']
                ref_intensities = mineral_data['intensities']
                
                # Check wavenumber overlap
                wn_min, wn_max = np.min(self.wavenumbers), np.max(self.wavenumbers)
                ref_min, ref_max = np.min(ref_wavenumbers), np.max(ref_wavenumbers)
                
                overlap = min(wn_max, ref_max) - max(wn_min, ref_min)
                total_range = max(wn_max, ref_max) - min(wn_min, ref_min)
                
                if overlap / total_range < 0.3:  # Need at least 30% overlap
                    continue
                
                # Interpolate to common grid
                ref_interp = np.interp(self.wavenumbers, ref_wavenumbers, ref_intensities)
                ref_interp = ref_interp / np.max(ref_interp)  # Normalize
                
                # Calculate correlation
                correlation = np.corrcoef(self.spectrum, ref_interp)[0, 1]
                
                if not np.isnan(correlation):
                    matches.append({
                        'mineral': mineral_name,
                        'correlation': correlation,
                        'wavenumbers': ref_wavenumbers,
                        'intensities': ref_intensities,
                        'metadata': mineral_data.get('metadata', {})
                    })
                
                # Update progress
                progress = int((i + 1) / total_minerals * 100)
                self.progress_updated.emit(progress)
                
            except Exception as e:
                continue
        
        # Sort by correlation and take top N
        matches.sort(key=lambda x: x['correlation'], reverse=True)
        top_matches = matches[:self.top_n]
        
        self.search_completed.emit(top_matches)

class InteractiveMixtureAnalyzer(QMainWindow):
    """Interactive mixture analysis interface."""
    
    def __init__(self):
        super().__init__()
        
        # Apply compact UI configuration for consistent toolbar sizing (EARLY - like main RamanLab app)
        if MATPLOTLIB_CONFIG_AVAILABLE:
            try:
                apply_theme('compact')
            except Exception as e:
                print(f"⚠️ Warning: Could not apply RamanLab matplotlib styling: {e}")
        
        # Load RamanLab database
        self.database = self.load_database()
        
        # Analysis state
        self.user_wavenumbers = None
        self.user_spectrum = None
        self.original_spectrum = None
        self.current_residual = None
        self.synthetic_components = []  # List of fitted components
        self.cumulative_synthetic = None
        self.iteration_history = []
        self.selected_reference = None
        
        # UI setup
        self.setup_ui()
        self.setWindowTitle("RamanLab Interactive Mixture Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply additional UI styling (but NOT matplotlib config - that's done above)
        self.apply_ui_styling()
        
        # Final synchronization after UI is complete
        QTimer.singleShot(100, self.synchronize_peak_selection)
    
    def preselect_mineral(self, mineral_name, confidence_score=None):
        """Pre-select a specific mineral from the database and display it."""
        if not self.database:
            return
        
        # Search for the mineral in the database
        best_match = None
        best_score = 0
        
        for db_mineral_name, mineral_data in self.database.items():
            # Check if this is the target mineral (case insensitive, partial match)
            if mineral_name.lower() in db_mineral_name.lower():
                # Calculate a simple score based on name similarity
                score = len(mineral_name) / len(db_mineral_name)
                if score > best_score:
                    best_score = score
                    best_match = {
                        'mineral': db_mineral_name,
                        'correlation': confidence_score or 0.93,  # Use provided confidence or default
                        'wavenumbers': mineral_data['wavenumbers'],
                        'intensities': mineral_data['intensities'],
                        'metadata': mineral_data.get('metadata', {})
                    }
        
        if best_match:
            # Add to search results
            self.results_list.clear()
            item = QListWidgetItem(f"1. {best_match['mineral']} (corr: {best_match['correlation']:.3f})")
            item.setData(Qt.UserRole, best_match)
            self.results_list.addItem(item)
            
            # Auto-select the first item
            self.results_list.setCurrentRow(0)
            self.on_result_selected(item)
            
            self.log_status(f"🎯 Pre-selected mineral: {best_match['mineral']}")
            if confidence_score:
                self.log_status(f"   Search confidence: {confidence_score:.3f}")
            
            # Synchronize peak selection after mineral is selected
            QTimer.singleShot(200, self.synchronize_peak_selection)
        else:
            self.log_status(f"⚠️ Could not find '{mineral_name}' in database")
    
    def preselect_exact_mineral(self, match_data):
        """Pre-select exact mineral data from search results."""
        if not match_data:
            return
        
        # Extract data from the match
        mineral_name = match_data.get('name', 'Unknown')
        metadata = match_data.get('metadata', {})
        display_name = metadata.get('NAME') or metadata.get('mineral_name') or mineral_name
        score = match_data.get('score', 0.0)
        wavenumbers = match_data.get('wavenumbers', [])
        intensities = match_data.get('intensities', [])
        
        if len(wavenumbers) == 0 or len(intensities) == 0:
            self.log_status(f"⚠️ No spectral data available for {display_name}")
            return
        
        # Create the exact match data structure for the mixture analyzer
        exact_match = {
            'mineral': display_name,
            'correlation': score,
            'wavenumbers': np.array(wavenumbers),
            'intensities': np.array(intensities),
            'metadata': metadata
        }
        
        # Set as selected reference
        self.selected_reference = exact_match
        
        # Add to search results list
        self.results_list.clear()
        item = QListWidgetItem(f"1. {display_name} (corr: {score:.3f})")
        item.setData(Qt.UserRole, exact_match)
        self.results_list.addItem(item)
        
        # Auto-select the item
        self.results_list.setCurrentRow(0)
        
        # Update overlay plot with the exact selected data
        if self.user_wavenumbers is not None and self.current_residual is not None:
            self.plot_canvas.update_overlay(
                self.user_wavenumbers, self.current_residual,
                exact_match['wavenumbers'], exact_match['intensities'],
                reference_name=display_name
            )
        
        self.log_status(f"🎯 Pre-selected exact mineral: {display_name}")
        self.log_status(f"   Search confidence: {score:.3f}")
        
        # Synchronize peak selection after mineral is selected
        QTimer.singleShot(200, self.synchronize_peak_selection)
    
    def load_database(self):
        """Load the complete RamanLab_Database_20250602.pkl for mixture analysis."""
        # Load the complete RamanLab database with experimental Raman spectra
        if RAMANLAB_AVAILABLE:
            try:
                from pkl_utils import load_raman_database
                print("🔍 Loading complete RamanLab_Database_20250602.pkl for mixture analysis...")
                database = load_raman_database()  # This loads RamanLab_Database_20250602.pkl
                
                print(f"📊 Raw database contains {len(database)} experimental spectra entries")
                
                # Convert and filter for mixture analysis (no artificial limits)
                converted_database = {}
                processed_minerals = set()  # Track unique mineral names to avoid duplicates
                skipped_count = 0
                
                for entry_name, entry_data in database.items():
                    try:
                        # Check if entry has required spectral data
                        if ('wavenumbers' in entry_data and 'intensities' in entry_data and
                            len(entry_data['wavenumbers']) > 0 and len(entry_data['intensities']) > 0):
                            
                            wavenumbers = np.array(entry_data['wavenumbers'])
                            intensities = np.array(entry_data['intensities'])
                            
                            # Quality check: ensure valid spectral data
                            if len(wavenumbers) < 10 or len(intensities) < 10:
                                skipped_count += 1
                                continue
                            
                            # Normalize intensities
                            max_intensity = np.max(intensities)
                            if max_intensity > 0:
                                intensities = intensities / max_intensity
                            else:
                                skipped_count += 1
                                continue
                            
                            # Extract clean mineral name for mixture analysis
                            clean_name = entry_name.split('__')[0] if '__' in entry_name else entry_name
                            
                            # Store all valid entries (keep duplicates as they may have different conditions)
                            # Use a suffix for duplicates to maintain multiple spectra per mineral
                            unique_key = clean_name
                            counter = 1
                            while unique_key in converted_database:
                                counter += 1
                                unique_key = f"{clean_name}_{counter}"
                            
                            converted_database[unique_key] = {
                                'wavenumbers': wavenumbers,
                                'intensities': intensities,
                                'metadata': {
                                    'source': 'RamanLab_Database_20250602.pkl',
                                    'original_name': entry_name,
                                    'mineral_base_name': clean_name,
                                    **entry_data.get('metadata', {})
                                }
                            }
                            
                    except Exception as e:
                        skipped_count += 1
                        continue
                
                print(f"✅ Loaded complete RamanLab_Database_20250602.pkl with {len(converted_database)} spectra")
                print(f"   📈 Processed: {len(database)} → {len(converted_database)} (skipped {skipped_count} invalid)")
                print("   🔬 Full experimental database for comprehensive mixture analysis")
                
                # Count unique mineral base names
                unique_minerals = set()
                for key, data in converted_database.items():
                    unique_minerals.add(data['metadata']['mineral_base_name'])
                
                print(f"   🧪 Unique mineral types: {len(unique_minerals)}")
                
                # Show sample of loaded minerals
                sample_names = list(converted_database.keys())[:8]
                print(f"   Sample entries: {', '.join(sample_names)}")
                
                return converted_database
                
            except Exception as e:
                print(f"❌ Error loading RamanLab_Database_20250602.pkl: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback to test database
        print("⚠️ Using test database for demonstration")
        return self.create_test_database()
    
    def create_test_database(self):
        """Create test database for demonstration."""
        wavenumbers = np.linspace(200, 1200, 500)
        
        database = {}
        
        # Quartz
        quartz_intensities = (
            0.3 * np.exp(-((wavenumbers - 465)**2) / (2 * 15**2)) +
            0.1 * np.exp(-((wavenumbers - 207)**2) / (2 * 10**2))
        )
        database['Quartz'] = {
            'wavenumbers': wavenumbers,
            'intensities': quartz_intensities / np.max(quartz_intensities),
            'metadata': {'formula': 'SiO2', 'test_mineral': True}
        }
        
        # Calcite
        calcite_intensities = (
            0.4 * np.exp(-((wavenumbers - 1086)**2) / (2 * 20**2)) +
            0.2 * np.exp(-((wavenumbers - 712)**2) / (2 * 12**2)) +
            0.15 * np.exp(-((wavenumbers - 282)**2) / (2 * 8**2))
        )
        database['Calcite'] = {
            'wavenumbers': wavenumbers,
            'intensities': calcite_intensities / np.max(calcite_intensities),
            'metadata': {'formula': 'CaCO3', 'test_mineral': True}
        }
        
        # Feldspar
        feldspar_intensities = (
            0.25 * np.exp(-((wavenumbers - 508)**2) / (2 * 18**2)) +
            0.2 * np.exp(-((wavenumbers - 477)**2) / (2 * 15**2)) +
            0.15 * np.exp(-((wavenumbers - 288)**2) / (2 * 10**2))
        )
        database['Feldspar'] = {
            'wavenumbers': wavenumbers,
            'intensities': feldspar_intensities / np.max(feldspar_intensities),
            'metadata': {'formula': 'KAlSi3O8', 'test_mineral': True}
        }
        
        print(f"✅ Created test database with {len(database)} minerals")
        return database
    
    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel for controls
        self.setup_control_panel(splitter)
        
        # Right panel for plotting
        self.setup_plot_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
    
    def setup_control_panel(self, parent):
        """Setup the left control panel."""
        control_widget = QWidget()
        parent.addWidget(control_widget)
        
        layout = QVBoxLayout(control_widget)
        
        # Data loading section
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout(data_group)
        
        self.load_data_btn = QPushButton("Load Spectrum Data")
        self.load_data_btn.clicked.connect(self.load_spectrum_data)
        data_layout.addWidget(self.load_data_btn)
        
        self.demo_data_btn = QPushButton("Use Demo Data")
        self.demo_data_btn.clicked.connect(self.load_demo_data)
        data_layout.addWidget(self.demo_data_btn)
        
        self.data_status_label = QLabel("No data loaded")
        data_layout.addWidget(self.data_status_label)
        
        layout.addWidget(data_group)
        
        # Search section
        search_group = QGroupBox("Database Search")
        search_layout = QVBoxLayout(search_group)
        
        self.search_btn = QPushButton("Search Database")
        self.search_btn.clicked.connect(self.search_database)
        self.search_btn.setEnabled(False)
        search_layout.addWidget(self.search_btn)
        
        self.search_residual_btn = QPushButton("Search on Residual")
        self.search_residual_btn.clicked.connect(self.search_on_residual)
        self.search_residual_btn.setEnabled(False)
        search_layout.addWidget(self.search_residual_btn)
        
        layout.addWidget(search_group)
        
        # Search results
        results_group = QGroupBox("Search Results (Top 10)")
        results_layout = QVBoxLayout(results_group)
        
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_result_selected)
        results_layout.addWidget(self.results_list)
        
        layout.addWidget(results_group)
        
        # Peak selection section
        peaks_group = QGroupBox("Peak Selection")
        peaks_layout = QVBoxLayout(peaks_group)
        
        self.selected_peaks_label = QLabel("Selected peaks: None")
        peaks_layout.addWidget(self.selected_peaks_label)
        
        self.clear_peaks_btn = QPushButton("Clear Selected Peaks")
        self.clear_peaks_btn.clicked.connect(self.clear_selected_peaks)
        peaks_layout.addWidget(self.clear_peaks_btn)
        
        self.fit_peaks_btn = QPushButton("Fit Selected Peaks")
        self.fit_peaks_btn.clicked.connect(self.fit_selected_peaks)
        self.fit_peaks_btn.setEnabled(False)
        peaks_layout.addWidget(self.fit_peaks_btn)
        
        layout.addWidget(peaks_group)
        
        # Analysis controls
        analysis_group = QGroupBox("Analysis Control")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.iteration_count_label = QLabel("Iterations: 0")
        analysis_layout.addWidget(self.iteration_count_label)
        
        self.current_r_squared_label = QLabel("Current R²: 0.000")
        analysis_layout.addWidget(self.current_r_squared_label)
        
        self.reset_analysis_btn = QPushButton("Reset Analysis")
        self.reset_analysis_btn.clicked.connect(self.reset_analysis)
        analysis_layout.addWidget(self.reset_analysis_btn)
        
        self.finalize_btn = QPushButton("Finalize Analysis")
        self.finalize_btn.clicked.connect(self.finalize_analysis)
        self.finalize_btn.setEnabled(False)
        analysis_layout.addWidget(self.finalize_btn)
        
        layout.addWidget(analysis_group)
        
        # Component summary
        summary_group = QGroupBox("Components Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.components_table = QTableWidget(0, 3)
        self.components_table.setHorizontalHeaderLabels(["Mineral", "R²", "Peaks"])
        self.components_table.horizontalHeader().setStretchLastSection(True)
        summary_layout.addWidget(self.components_table)
        
        layout.addWidget(summary_group)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setPlainText("Ready to load spectrum data...")
        layout.addWidget(self.status_text)
    
    def setup_plot_panel(self, parent):
        """Setup the right plot panel with RamanLab-styled matplotlib controls."""
        self.plot_canvas = PeakSelectionCanvas()  # Create canvas without parent
        self.plot_canvas.main_window = self  # Store direct reference to main window
        
        # Add matplotlib navigation toolbar with RamanLab styling
        if MATPLOTLIB_CONFIG_AVAILABLE:
            try:
                # Use the optimized toolbar class from matplotlib_config
                toolbar_class = get_toolbar_class('compact')
                self.toolbar = toolbar_class(self.plot_canvas, self)
            except Exception as e:
                print(f"⚠️ Using fallback toolbar: {e}")
                self.toolbar = CompactNavigationToolbar(self.plot_canvas, self)
        else:
            self.toolbar = CompactNavigationToolbar(self.plot_canvas, self)
        
        # Create layout for plot panel
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.plot_canvas)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)
        
        # Create widget to contain the layout
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        parent.addWidget(plot_widget)
        
        # Ensure peak selection state is synchronized
        self.synchronize_peak_selection()
    
    def synchronize_peak_selection(self):
        """Synchronize peak selection state between canvas and main window."""
        if hasattr(self, 'plot_canvas') and hasattr(self, 'fit_peaks_btn'):
            # Check if there are already selected peaks and update the UI
            if self.plot_canvas.selected_peaks:
                self.update_selected_peaks_display()
                self.fit_peaks_btn.setEnabled(True)
                peaks_str = ", ".join([f"{peak:.1f}" for peak in self.plot_canvas.selected_peaks])
                self.log_status(f"🔄 Synchronized {len(self.plot_canvas.selected_peaks)} selected peaks: {peaks_str}")
            else:
                self.update_selected_peaks_display()
                self.fit_peaks_btn.setEnabled(False)
    
    def apply_ui_styling(self):
        """Apply UI styling (matplotlib configuration is applied earlier in __init__)."""
        # Set consistent font matching other RamanLab applications
        font = QFont("Arial", 10)
        self.setFont(font)
        
        # Apply consistent window styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
            QPushButton:pressed {
                background-color: #d9d9d9;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #999999;
            }
        """)
        
        # Log styling status
        if MATPLOTLIB_CONFIG_AVAILABLE:
            self.log_status("✅ Applied RamanLab styling configuration")
        else:
            self.log_status("⚠️ Using default styling (matplotlib_config not available)")
    
    def load_spectrum_data(self):
        """Load spectrum data from file."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Spectrum Data", "", 
            "Text files (*.txt *.csv *.dat);;All files (*)"
        )
        
        if file_path:
            try:
                data = np.loadtxt(file_path)
                if data.shape[1] >= 2:
                    self.user_wavenumbers = data[:, 0]
                    self.user_spectrum = data[:, 1]
                    self.original_spectrum = self.user_spectrum.copy()
                    self.current_residual = self.user_spectrum.copy()
                    
                    self.data_status_label.setText(f"Loaded: {len(self.user_wavenumbers)} points")
                    self.search_btn.setEnabled(True)
                    self.log_status(f"✅ Loaded spectrum data: {Path(file_path).name}")
                    self.log_status(f"   Range: {self.user_wavenumbers[0]:.1f} - {self.user_wavenumbers[-1]:.1f} cm⁻¹")
                    
                    # Pass data to plot canvas and display the loaded spectrum immediately
                    self.plot_canvas.user_wavenumbers = self.user_wavenumbers
                    self.plot_canvas.user_spectrum = self.user_spectrum
                    self.plot_canvas.display_user_spectrum()
                else:
                    QMessageBox.warning(self, "Error", "File must have at least 2 columns (wavenumber, intensity)")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load file: {e}")
    
    def load_demo_data(self):
        """Load demonstration mixture data."""
        # Create synthetic mixture
        wavenumbers = np.linspace(200, 1200, 500)
        
        # Mixture of quartz + calcite + feldspar
        mixture_spectrum = (
            0.5 * np.exp(-((wavenumbers - 465)**2) / (2 * 15**2)) +  # Quartz
            0.3 * np.exp(-((wavenumbers - 1086)**2) / (2 * 20**2)) +  # Calcite
            0.2 * np.exp(-((wavenumbers - 508)**2) / (2 * 18**2)) +   # Feldspar
            0.05 * np.random.normal(0, 0.02, len(wavenumbers))       # Noise
        )
        
        mixture_spectrum = np.maximum(mixture_spectrum, 0)
        
        self.user_wavenumbers = wavenumbers
        self.user_spectrum = mixture_spectrum
        self.original_spectrum = mixture_spectrum.copy()
        self.current_residual = mixture_spectrum.copy()
        
        self.data_status_label.setText(f"Demo data: {len(wavenumbers)} points")
        self.search_btn.setEnabled(True)
        self.log_status("✅ Loaded demo mixture data (Quartz + Calcite + Feldspar)")
        
        # Pass data to plot canvas and display the demo spectrum immediately
        self.plot_canvas.user_wavenumbers = self.user_wavenumbers
        self.plot_canvas.user_spectrum = self.user_spectrum
        self.plot_canvas.display_user_spectrum()
    
    def search_database(self):
        """Search database for matches to current spectrum."""
        if self.user_wavenumbers is None:
            return
        
        self.log_status("🔍 Searching database for matches...")
        self.search_btn.setEnabled(False)
        
        # Create progress dialog
        progress = QProgressDialog("Searching database...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # Create worker thread for search
        self.search_worker = DatabaseSearchWorker(
            self.database, 
            self.current_residual / np.max(self.current_residual),  # Normalize
            self.user_wavenumbers, 
            top_n=10
        )
        
        # Connect signals
        self.search_worker.progress_updated.connect(progress.setValue)
        self.search_worker.search_completed.connect(self.on_search_completed)
        self.search_worker.search_completed.connect(progress.close)
        
        # Start search
        QTimer.singleShot(100, self.search_worker.search_database)
    
    def search_on_residual(self):
        """Search database for matches to current residual."""
        if self.current_residual is None:
            return
        
        self.log_status("🔍 Searching database on residual...")
        self.search_residual_btn.setEnabled(False)
        
        # Create progress dialog
        progress = QProgressDialog("Searching on residual...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # Create worker thread for search
        self.search_worker = DatabaseSearchWorker(
            self.database, 
            self.current_residual / np.max(self.current_residual),  # Normalize
            self.user_wavenumbers, 
            top_n=10
        )
        
        # Connect signals
        self.search_worker.progress_updated.connect(progress.setValue)
        self.search_worker.search_completed.connect(self.on_search_completed)
        self.search_worker.search_completed.connect(progress.close)
        
        # Start search
        QTimer.singleShot(100, self.search_worker.search_database)
    
    def on_search_completed(self, matches):
        """Handle search completion."""
        self.search_btn.setEnabled(True)
        self.search_residual_btn.setEnabled(True)
        
        # Populate results list
        self.results_list.clear()
        
        for i, match in enumerate(matches):
            correlation = match['correlation']
            mineral = match['mineral']
            
            item = QListWidgetItem(f"{i+1}. {mineral} (corr: {correlation:.3f})")
            item.setData(Qt.UserRole, match)
            self.results_list.addItem(item)
        
        self.log_status(f"✅ Found {len(matches)} matches")
        if matches:
            self.log_status(f"   Best match: {matches[0]['mineral']} (correlation: {matches[0]['correlation']:.3f})")
    
    def on_result_selected(self, item):
        """Handle selection of a search result."""
        match_data = item.data(Qt.UserRole)
        if match_data:
            self.selected_reference = match_data
            mineral_name = match_data['mineral']
            correlation = match_data['correlation']
            
            # Update overlay plot
            self.plot_canvas.update_overlay(
                self.user_wavenumbers, self.current_residual,
                match_data['wavenumbers'], match_data['intensities'],
                reference_name=mineral_name
            )
            
            # Clear selected peaks for new selection
            self.plot_canvas.clear_selected_peaks()
            self.update_selected_peaks_display()
            
            self.log_status(f"📊 Selected: {mineral_name} (correlation: {correlation:.3f})")
            self.log_status("   Click on peaks in the overlay plot to select them for fitting")
    
    def on_peak_selected(self, peak_wavenumber):
        """Handle peak selection from plot."""
        print(f"🔧 DEBUG: on_peak_selected called with peak {peak_wavenumber:.1f}")
        print(f"🔧 DEBUG: plot_canvas.selected_peaks = {self.plot_canvas.selected_peaks}")
        print(f"🔧 DEBUG: fit_peaks_btn exists = {hasattr(self, 'fit_peaks_btn')}")
        
        self.update_selected_peaks_display()
        
        # Enable fit button if we have peaks and a reference
        has_peaks = len(self.plot_canvas.selected_peaks) > 0
        has_reference = self.selected_reference is not None
        should_enable = has_peaks and has_reference
        
        print(f"🔧 DEBUG: has_peaks = {has_peaks}, has_reference = {has_reference}")
        print(f"🔧 DEBUG: should_enable button = {should_enable}")
        
        if hasattr(self, 'fit_peaks_btn'):
            self.fit_peaks_btn.setEnabled(should_enable)
            print(f"🔧 DEBUG: Button enabled = {self.fit_peaks_btn.isEnabled()}")
        
        peaks_str = ", ".join([f"{peak:.1f}" for peak in self.plot_canvas.selected_peaks])
        self.log_status(f"🎯 Selected peak at {peak_wavenumber:.1f} cm⁻¹")
        self.log_status(f"   All selected: {peaks_str}")
        
        if not has_reference:
            self.log_status("⚠️ No reference mineral selected - select from search results first")
    
    def update_selected_peaks_display(self):
        """Update the selected peaks display."""
        if self.plot_canvas.selected_peaks:
            peaks_str = ", ".join([f"{peak:.1f}" for peak in self.plot_canvas.selected_peaks])
            self.selected_peaks_label.setText(f"Selected peaks: {peaks_str}")
        else:
            self.selected_peaks_label.setText("Selected peaks: None")
    
    def clear_selected_peaks(self):
        """Clear all selected peaks."""
        self.plot_canvas.clear_selected_peaks()
        self.update_selected_peaks_display()
        self.fit_peaks_btn.setEnabled(False)
        self.log_status("🧹 Cleared all selected peaks")
    
    def fit_selected_peaks(self):
        """Fit pseudo-Voigt peaks to selected positions."""
        if not self.plot_canvas.selected_peaks or self.selected_reference is None:
            return
        
        self.log_status("🔧 Fitting pseudo-Voigt peaks to selected positions...")
        
        try:
            # Fit peaks using pseudo-Voigt profiles
            fitted_component = self.fit_pseudovoigt_peaks(
                self.user_wavenumbers, 
                self.current_residual,
                self.plot_canvas.selected_peaks,
                self.selected_reference['mineral']
            )
            
            if fitted_component is not None:
                # Add to synthetic components
                self.synthetic_components.append(fitted_component)
                
                # Update cumulative synthetic spectrum
                self.update_cumulative_synthetic()
                
                # Update residual
                self.update_residual()
                
                # Update plots
                self.update_all_plots()
                
                # Update iteration counter
                self.iteration_history.append({
                    'mineral': fitted_component['mineral'],
                    'r_squared': fitted_component['r_squared'],
                    'peaks': fitted_component['peaks'].copy()
                })
                
                self.iteration_count_label.setText(f"Iterations: {len(self.iteration_history)}")
                
                # Update components table
                self.update_components_table()
                
                # Enable residual search and finalize
                self.search_residual_btn.setEnabled(True)
                self.finalize_btn.setEnabled(True)
                
                self.log_status(f"✅ Fitted {len(fitted_component['peaks'])} peaks for {fitted_component['mineral']}")
                self.log_status(f"   Component R²: {fitted_component['r_squared']:.4f}")
                
                # Clear selected peaks for next iteration
                self.plot_canvas.clear_selected_peaks()
                self.update_selected_peaks_display()
                self.fit_peaks_btn.setEnabled(False)
            
        except Exception as e:
            self.log_status(f"❌ Peak fitting failed: {e}")
            QMessageBox.warning(self, "Fitting Error", f"Could not fit peaks: {e}")
    
    def estimate_peak_width_from_spectrum(self, wavenumbers, spectrum, peak_center):
        """Estimate peak width from the actual spectrum around a given position."""
        try:
            # Find the peak center index
            center_idx = np.argmin(np.abs(wavenumbers - peak_center))
            peak_intensity = spectrum[center_idx]
            
            # Find the half-maximum intensity
            half_max = peak_intensity * 0.5
            
            # Search for full width at half maximum (FWHM)
            # Look left from center
            left_idx = center_idx
            while left_idx > 0 and spectrum[left_idx] > half_max:
                left_idx -= 1
            
            # Look right from center
            right_idx = center_idx
            while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
                right_idx += 1
            
            # Calculate FWHM
            if left_idx < center_idx and right_idx > center_idx:
                fwhm = wavenumbers[right_idx] - wavenumbers[left_idx]
                # Convert FWHM to reasonable width estimate
                # For Gaussian: FWHM ≈ 2.355 * sigma
                # For practical fitting, use FWHM directly as a good width estimate
                estimated_width = max(2.0, min(50.0, fwhm))  # Constrain to reasonable range
            else:
                # Fallback: estimate based on local noise and peak sharpness
                window_size = min(20, len(spectrum) // 10)
                start_idx = max(0, center_idx - window_size)
                end_idx = min(len(spectrum), center_idx + window_size)
                
                # Calculate local standard deviation as width estimate
                local_spectrum = spectrum[start_idx:end_idx]
                local_variation = np.std(local_spectrum)
                
                # Use 3-8 times the wavenumber spacing as a reasonable default
                avg_spacing = np.median(np.diff(wavenumbers))
                estimated_width = max(5.0, min(25.0, avg_spacing * 6))
            
            return estimated_width
            
        except Exception as e:
            print(f"⚠️ Error estimating peak width: {e}")
            return 8.0  # Reasonable default
    
    def fit_pseudovoigt_peaks(self, wavenumbers, spectrum, peak_positions, mineral_name):
        """Fit pseudo-Voigt peaks at specified positions with realistic parameters."""
        try:
            # Prepare initial parameters with realistic estimation
            initial_params = []
            bounds_lower = []
            bounds_upper = []
            
            # Calculate average peak spacing for width estimation
            wavenumber_spacing = np.median(np.diff(wavenumbers))
            
            for peak_wn in peak_positions:
                # Find peak intensity and estimate width from actual spectrum
                idx = np.argmin(np.abs(wavenumbers - peak_wn))
                peak_intensity = spectrum[idx]
                
                # Estimate peak width from the spectrum at this position
                estimated_width = self.estimate_peak_width_from_spectrum(wavenumbers, spectrum, peak_wn)
                
                # Use estimated width, but constrain to reasonable range
                initial_sigma = max(2.0, min(20.0, estimated_width * 0.6))  # Gaussian component
                initial_gamma = max(1.0, min(15.0, estimated_width * 0.4))  # Lorentzian component
                
                print(f"🔧 Peak at {peak_wn:.1f}: intensity={peak_intensity:.1f}, estimated_width={estimated_width:.1f}")
                print(f"🔧   Initial: sigma={initial_sigma:.1f}, gamma={initial_gamma:.1f}")
                
                # Parameters: [amplitude, center, sigma, gamma] for each peak
                initial_params.extend([
                    peak_intensity,    # amplitude
                    peak_wn,          # center
                    initial_sigma,    # sigma (Gaussian width)
                    initial_gamma     # gamma (Lorentzian width)
                ])
                
                # More realistic bounds based on the estimated width
                center_tolerance = max(3.0, estimated_width * 0.5)
                max_width = max(30.0, estimated_width * 3.0)
                
                bounds_lower.extend([
                    0.0,                        # amplitude >= 0
                    peak_wn - center_tolerance, # center with realistic tolerance
                    0.5,                        # sigma >= 0.5
                    0.1                         # gamma >= 0.1
                ])
                
                bounds_upper.extend([
                    peak_intensity * 5.0,       # amplitude <= 5x initial (more flexible)
                    peak_wn + center_tolerance, # center with realistic tolerance
                    max_width,                  # sigma based on estimated width
                    max_width * 0.8             # gamma slightly smaller than sigma
                ])
            
            def multi_pseudovoigt(x, *params):
                """Multi-peak pseudo-Voigt function."""
                result = np.zeros_like(x)
                num_peaks = len(params) // 4
                
                for i in range(num_peaks):
                    base_idx = i * 4
                    if base_idx + 3 < len(params):
                        amplitude = params[base_idx]
                        center = params[base_idx + 1]
                        sigma = params[base_idx + 2]
                        gamma = params[base_idx + 3]
                        
                        # Pseudo-Voigt (weighted sum of Gaussian and Lorentzian)
                        gaussian = np.exp(-0.5 * ((x - center) / sigma)**2)
                        lorentzian = 1 / (1 + ((x - center) / gamma)**2)
                        
                        # 50:50 mix of Gaussian and Lorentzian
                        pseudo_voigt = amplitude * (0.5 * gaussian + 0.5 * lorentzian)
                        result += pseudo_voigt
                
                return result
            
            # Perform fitting with improved settings
            print(f"🔧 Fitting {len(peak_positions)} peaks for {mineral_name}")
            print(f"🔧 Initial params count: {len(initial_params)}, bounds: {len(bounds_lower)}-{len(bounds_upper)}")
            
            popt, pcov = curve_fit(
                multi_pseudovoigt, wavenumbers, spectrum,
                p0=initial_params,
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000,  # Increased iterations
                method='trf'   # Trust Region Reflective algorithm (better for bounds)
            )
            
            # Generate fitted spectrum
            fitted_spectrum = multi_pseudovoigt(wavenumbers, *popt)
            
            # Calculate fit quality metrics
            residual = spectrum - fitted_spectrum
            r_squared = r2_score(spectrum, fitted_spectrum)
            rmse = np.sqrt(np.mean(residual**2))
            
            print(f"🔧 Fit quality: R² = {r_squared:.4f}, RMSE = {rmse:.2f}")
            
            # Log fitted parameters for debugging
            for i, peak_wn in enumerate(peak_positions):
                amp = popt[i*4]
                center = popt[i*4 + 1]
                sigma = popt[i*4 + 2]
                gamma = popt[i*4 + 3]
                print(f"🔧 Peak {i+1} at {peak_wn:.1f}: fitted center={center:.1f}, amp={amp:.1f}, σ={sigma:.1f}, γ={gamma:.1f}")
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            component = {
                'mineral': mineral_name,
                'peaks': np.array(peak_positions),
                'fitted_spectrum': fitted_spectrum,
                'parameters': popt,
                'parameter_errors': param_errors,
                'r_squared': r_squared,
                'profile_type': 'pseudo_voigt'
            }
            
            return component
            
        except Exception as e:
            raise Exception(f"Pseudo-Voigt fitting failed: {e}")
    
    def update_cumulative_synthetic(self):
        """Update the cumulative synthetic spectrum."""
        if not self.synthetic_components:
            self.cumulative_synthetic = np.zeros_like(self.user_wavenumbers)
        else:
            self.cumulative_synthetic = np.sum([
                comp['fitted_spectrum'] for comp in self.synthetic_components
            ], axis=0)
    
    def update_residual(self):
        """Update the current residual."""
        if self.cumulative_synthetic is not None:
            self.current_residual = self.original_spectrum - self.cumulative_synthetic
            self.current_residual = np.maximum(self.current_residual, 0)  # No negative values
    
    def update_all_plots(self):
        """Update all plots with current data."""
        # Update residual plot
        if self.current_residual is not None:
            self.plot_canvas.update_residual(self.current_residual)
        
        # Update synthetic plot
        if self.cumulative_synthetic is not None:
            self.plot_canvas.update_synthetic(self.cumulative_synthetic)
        
        # Update fit quality
        self.plot_canvas.update_component_analysis(self.iteration_history, self.synthetic_components)
        
        # Update R² display
        if self.cumulative_synthetic is not None:
            overall_r_squared = r2_score(self.original_spectrum, self.cumulative_synthetic)
            self.current_r_squared_label.setText(f"Current R²: {overall_r_squared:.4f}")
    
    def update_components_table(self):
        """Update the components summary table."""
        self.components_table.setRowCount(len(self.synthetic_components))
        
        for i, component in enumerate(self.synthetic_components):
            mineral_item = QTableWidgetItem(component['mineral'])
            r_squared_item = QTableWidgetItem(f"{component['r_squared']:.4f}")
            peaks_item = QTableWidgetItem(f"{len(component['peaks'])}")
            
            self.components_table.setItem(i, 0, mineral_item)
            self.components_table.setItem(i, 1, r_squared_item)
            self.components_table.setItem(i, 2, peaks_item)
    
    def reset_analysis(self):
        """Reset the analysis to start over."""
        reply = QMessageBox.question(
            self, "Reset Analysis", 
            "Are you sure you want to reset the analysis? All fitted components will be lost.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.synthetic_components = []
            self.cumulative_synthetic = None
            self.current_residual = self.original_spectrum.copy() if self.original_spectrum is not None else None
            self.iteration_history = []
            
            # Reset UI
            self.iteration_count_label.setText("Iterations: 0")
            self.current_r_squared_label.setText("Current R²: 0.000")
            self.components_table.setRowCount(0)
            self.results_list.clear()
            
            # Clear plots
            self.plot_canvas.setup_plots()
            
            # Reset buttons
            self.search_residual_btn.setEnabled(False)
            self.finalize_btn.setEnabled(False)
            
            self.log_status("🔄 Analysis reset - ready to start over")
    
    def finalize_analysis(self):
        """Finalize the analysis and show final results."""
        if not self.synthetic_components:
            return
        
        # Calculate final statistics
        overall_r_squared = r2_score(self.original_spectrum, self.cumulative_synthetic)
        rms_residual = np.sqrt(np.mean(self.current_residual**2))
        
        # Create results summary
        results_text = f"""
🎯 FINAL MIXTURE ANALYSIS RESULTS
{'='*50}

Components Identified: {len(self.synthetic_components)}
Overall R²: {overall_r_squared:.4f}
RMS Residual: {rms_residual:.6f}

Individual Components:
"""
        
        for i, component in enumerate(self.synthetic_components):
            results_text += f"\n{i+1}. {component['mineral']}"
            results_text += f"\n   Peaks fitted: {len(component['peaks'])}"
            results_text += f"\n   Individual R²: {component['r_squared']:.4f}"
            results_text += f"\n   Peak positions: {', '.join([f'{p:.1f}' for p in component['peaks']])} cm⁻¹"
        
        results_text += f"\n\n{'='*50}"
        
        # Show results dialog
        QMessageBox.information(self, "Analysis Complete", results_text)
        
        self.log_status("🎉 Analysis finalized!")
        self.log_status(f"   Final R²: {overall_r_squared:.4f}")
        self.log_status(f"   Components: {len(self.synthetic_components)}")
    
    def log_status(self, message):
        """Log a status message."""
        current_text = self.status_text.toPlainText()
        new_text = f"{current_text}\n{message}"
        
        # Keep only last 20 lines
        lines = new_text.split('\n')
        if len(lines) > 20:
            lines = lines[-20:]
            new_text = '\n'.join(lines)
        
        self.status_text.setPlainText(new_text)
        
        # Scroll to bottom
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.status_text.setTextCursor(cursor)

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("RamanLab Interactive Mixture Analysis")
    app.setApplicationVersion("1.3.0")
    app.setOrganizationName("RamanLab")
    
    # Create and show main window
    window = InteractiveMixtureAnalyzer()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 