#!/usr/bin/env python3
"""
Raman Spectrum Analysis Tool - Qt6 Compatible Core Spectra Class
Core functionality for importing, analyzing, and identifying Raman spectra with Qt6 integration
"""

import os
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import correlation
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

# Qt6 imports for dialogs and file operations
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QApplication
from PySide6.QtCore import QStandardPaths, Qt

# Qt6 imports for parent widget reference
try:
    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtCore import QStandardPaths
    HAS_QT = True
except ImportError:
    HAS_QT = False


class RamanSpectraQt6:
    """Qt6-compatible class for handling and analyzing Raman spectra data."""
    
    def __init__(self, parent_widget=None):
        """Initialize the RamanSpectra object."""
        self.parent = parent_widget
        self.current_spectra = None
        self.current_wavenumbers = None
        self.processed_spectra = None
        self.peaks = None
        self.background = None
        self.metadata = {}
        self.database = {}
        
        # Set up database path with fallback logic
        self._setup_database_path()
        
        # Try to load existing database
        self.load_database()
    
    def _setup_database_path(self):
        """Setup the database path with fallback to script directory for compatibility."""
        # Primary path: Documents/RamanLab_Qt6 directory (for user data)
        docs_path = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        self.db_directory = Path(docs_path) / "RamanLab_Qt6"
        primary_db_path = self.db_directory / "RamanLab_Database_20250602.pkl"
        
        # Fallback path: Script directory (for compatibility with Database Manager)
        script_dir = Path(__file__).parent
        fallback_db_path = script_dir / "RamanLab_Database_20250602.pkl"
        
        # Create Documents directory if it doesn't exist
        self.db_directory.mkdir(exist_ok=True)
        
        # Check which database exists and use that path
        if os.path.exists(primary_db_path):
            self.db_path = primary_db_path
            print(f"Using database from Documents folder: {primary_db_path}")
        elif os.path.exists(fallback_db_path):
            self.db_path = fallback_db_path
            print(f"Using database from script directory: {fallback_db_path}")
        else:
            # Default to primary path for new databases
            self.db_path = primary_db_path
            print(f"No existing database found, will create at: {primary_db_path}")
    
    def add_to_database(self, name, wavenumbers, intensities, metadata=None, peaks=None):
        """
        Add a spectrum to the database.
        
        Parameters:
        -----------
        name : str
            Unique name for the spectrum
        wavenumbers : array-like
            Wavenumber data
        intensities : array-like
            Intensity data
        metadata : dict, optional
            Additional metadata
        peaks : array-like, optional
            Peak positions
        """
        try:
            # Create database entry
            entry = {
                'name': name,
                'wavenumbers': np.array(wavenumbers).tolist(),
                'intensities': np.array(intensities).tolist(),
                'peaks': peaks.tolist() if peaks is not None else [],
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to database
            self.database[name] = entry
            
            # Save database
            self.save_database()
            
            return True
            
        except Exception as e:
            if self.parent:
                QMessageBox.critical(
                    self.parent, 
                    "Database Error", 
                    f"Failed to add spectrum to database:\n{str(e)}"
                )
            return False
    
    def remove_from_database(self, name):
        """Remove a spectrum from the database."""
        try:
            if name in self.database:
                del self.database[name]
                self.save_database()
                return True
            else:
                if self.parent:
                    QMessageBox.warning(
                        self.parent,
                        "Not Found",
                        f"Spectrum '{name}' not found in database."
                    )
                return False
                
        except Exception as e:
            if self.parent:
                QMessageBox.critical(
                    self.parent,
                    "Database Error", 
                    f"Failed to remove spectrum from database:\n{str(e)}"
                )
            return False
    
    def save_database(self):
        """Save the database to file."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.database, f)
            return True
            
        except Exception as e:
            if self.parent:
                QMessageBox.critical(
                    self.parent,
                    "Save Error",
                    f"Failed to save database:\n{str(e)}"
                )
            return False
    
    def load_database(self):
        """Load the database from file."""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    self.database = pickle.load(f)
                print(f"✓ Database loaded successfully from: {self.db_path}")
                print(f"  Database contains {len(self.database)} entries")
                if len(self.database) > 0:
                    sample_keys = list(self.database.keys())[:3]
                    print(f"  Sample entries: {sample_keys}")
                return True
            else:
                # Create empty database
                print(f"⚠ Database file not found at: {self.db_path}")
                print("  Creating empty database")
                self.database = {}
                return True
                
        except Exception as e:
            print(f"❌ Error loading database from {self.db_path}: {e}")
            if self.parent:
                QMessageBox.critical(
                    self.parent,
                    "Load Error",
                    f"Failed to load database:\n{str(e)}\n\nCreating new database."
                )
            self.database = {}
            return False
    
    def get_database_stats(self):
        """Get database statistics."""
        if not self.database:
            return {
                'total_spectra': 0,
                'avg_data_points': 0,
                'avg_peaks': 0,
                'database_size': '0 KB'
            }
        
        total_spectra = len(self.database)
        total_points = sum(len(entry['wavenumbers']) for entry in self.database.values())
        total_peaks = sum(len(entry['peaks']) for entry in self.database.values())
        
        # Get database file size
        db_size = 0
        if os.path.exists(self.db_path):
            db_size = os.path.getsize(self.db_path)
        
        # Format file size
        if db_size < 1024:
            size_str = f"{db_size} B"
        elif db_size < 1024 * 1024:
            size_str = f"{db_size / 1024:.1f} KB"
        else:
            size_str = f"{db_size / (1024 * 1024):.1f} MB"
        
        return {
            'total_spectra': total_spectra,
            'avg_data_points': total_points / total_spectra if total_spectra > 0 else 0,
            'avg_peaks': total_peaks / total_spectra if total_spectra > 0 else 0,
            'database_size': size_str
        }
    
    def search_database(self, query_wavenumbers, query_intensities, n_matches=5, threshold=0.7):
        """
        Search database for similar spectra using correlation.
        
        Parameters:
        -----------
        query_wavenumbers : array-like
            Query spectrum wavenumbers
        query_intensities : array-like
            Query spectrum intensities
        n_matches : int
            Number of top matches to return
        threshold : float
            Minimum correlation threshold
            
        Returns:
        --------
        list
            List of matches with scores
        """
        if not self.database:
            return []
        
        matches = []
        
        try:
            # Progress dialog for long searches
            if len(self.database) > 10 and self.parent:
                progress = QProgressDialog(
                    "Searching database...", "Cancel", 0, len(self.database), self.parent
                )
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                QApplication.processEvents()
            else:
                progress = None
            
            for i, (name, entry) in enumerate(self.database.items()):
                if progress:
                    if progress.wasCanceled():
                        break
                    progress.setValue(i)
                    progress.setLabelText(f"Comparing with {name}...")
                    QApplication.processEvents()
                
                try:
                    # Get database spectrum
                    db_wavenumbers = np.array(entry['wavenumbers'])
                    db_intensities = np.array(entry['intensities'])
                    
                    # Interpolate to common wavenumber range
                    common_min = max(query_wavenumbers.min(), db_wavenumbers.min())
                    common_max = min(query_wavenumbers.max(), db_wavenumbers.max())
                    
                    if common_min >= common_max:
                        continue  # No overlap
                    
                    # Create common wavenumber grid
                    common_wavenumbers = np.linspace(common_min, common_max, 500)
                    
                    # Interpolate both spectra
                    query_interp = np.interp(common_wavenumbers, query_wavenumbers, query_intensities)
                    db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)
                    
                    # Normalize intensities
                    query_norm = (query_interp - query_interp.mean()) / query_interp.std()
                    db_norm = (db_interp - db_interp.mean()) / db_interp.std()
                    
                    # Calculate correlation
                    correlation_score = 1 - correlation(query_norm, db_norm)
                    
                    if correlation_score >= threshold:
                        matches.append({
                            'name': name,
                            'score': correlation_score,
                            'metadata': entry.get('metadata', {}),
                            'peaks': entry.get('peaks', []),
                            'timestamp': entry.get('timestamp', 'Unknown')
                        })
                
                except Exception as e:
                    print(f"Error comparing with {name}: {e}")
                    continue
            
            if progress:
                progress.close()
            
            # Sort by score and return top matches
            matches.sort(key=lambda x: x['score'], reverse=True)
            return matches[:n_matches]
            
        except Exception as e:
            if self.parent:
                QMessageBox.critical(
                    self.parent,
                    "Search Error",
                    f"Database search failed:\n{str(e)}"
                )
            return []
    
    def export_database(self, export_path, format='json'):
        """
        Export database to different formats.
        
        Parameters:
        -----------
        export_path : str
            Path to export file
        format : str
            Export format ('json', 'csv')
        """
        try:
            if format.lower() == 'json':
                with open(export_path, 'w') as f:
                    json.dump(self.database, f, indent=2)
                    
            elif format.lower() == 'csv':
                # Create CSV with basic info
                data = []
                for name, entry in self.database.items():
                    data.append({
                        'name': name,
                        'data_points': len(entry['wavenumbers']),
                        'peaks_found': len(entry['peaks']),
                        'timestamp': entry.get('timestamp', 'Unknown'),
                        'wavenumber_range_min': min(entry['wavenumbers']),
                        'wavenumber_range_max': max(entry['wavenumbers']),
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(export_path, index=False)
            
            return True
            
        except Exception as e:
            if self.parent:
                QMessageBox.critical(
                    self.parent,
                    "Export Error",
                    f"Failed to export database:\n{str(e)}"
                )
            return False
    
    def import_database(self, import_path):
        """Import database from file."""
        try:
            if import_path.endswith('.pkl'):
                with open(import_path, 'rb') as f:
                    imported_db = pickle.load(f)
            elif import_path.endswith('.json'):
                with open(import_path, 'r') as f:
                    imported_db = json.load(f)
            else:
                if self.parent:
                    QMessageBox.warning(
                        self.parent,
                        "Unsupported Format",
                        "Only .pkl and .json files are supported for import."
                    )
                return False
            
            # Merge with existing database
            conflicts = []
            for name in imported_db:
                if name in self.database:
                    conflicts.append(name)
            
            if conflicts and self.parent:
                reply = QMessageBox.question(
                    self.parent,
                    "Conflicts Found",
                    f"Found {len(conflicts)} conflicting entries. Overwrite them?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False
            
            # Import the database
            self.database.update(imported_db)
            self.save_database()
            
            return True
            
        except Exception as e:
            if self.parent:
                QMessageBox.critical(
                    self.parent,
                    "Import Error",
                    f"Failed to import database:\n{str(e)}"
                )
            return False
    
    def baseline_als(self, y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares Smoothing for baseline correction.
        
        Parameters:
        -----------
        y : array-like
            Input spectrum.
        lam : float
            Smoothness parameter (default: 1e5).
        p : float
            Asymmetry parameter (default: 0.01).
        niter : int
            Number of iterations (default: 10).
            
        Returns:
        --------
        array-like
            Estimated baseline.
        """
        L = len(y)
        D = csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        
        for i in range(niter):
            W = csc_matrix((w, (np.arange(L), np.arange(L))))
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        
        return z
    
    def subtract_background_als(self, wavenumbers, intensities, lam=1e5, p=0.01, niter=10):
        """
        Subtract background using ALS algorithm.
        
        Parameters:
        -----------
        wavenumbers : array-like
            Wavenumber array.
        intensities : array-like
            Intensity array.
        lam : float
            Smoothness parameter for baseline correction.
        p : float
            Asymmetry parameter for baseline correction.
        niter : int
            Number of iterations for baseline correction.
            
        Returns:
        --------
        tuple
            (corrected_intensities, baseline)
        """
        # Compute the baseline
        baseline = self.baseline_als(intensities, lam, p, niter)
        
        # Subtract baseline from spectrum
        corrected_spectrum = intensities - baseline
        
        # Set negative values to zero (optional)
        corrected_spectrum[corrected_spectrum < 0] = 0
        
        return corrected_spectrum, baseline 