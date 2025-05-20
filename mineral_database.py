#!/usr/bin/env python3
# Mineral Database Module for ClaritySpectra
"""
Module for managing a database of mineral Raman modes.
"""

import os
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import matplotlib as mpl
# Configure matplotlib to use DejaVu Sans which supports mathematical symbols
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import random
import json
import subprocess
import sys
import time
import math

class MineralDatabase:
    """Mineral database management system for Raman modes."""
    
    def __init__(self, database_path=None):
        """
        Initialize the mineral database.
        
        Parameters:
        -----------
        database_path : str, optional
            Path to the database file (.pkl)
        """
        self.database_path = database_path or os.path.join(os.path.dirname(__file__), "mineral_modes.pkl")
        self._database = None  # Initialize as None for lazy loading
        self._mineral_list = None  # Cache for mineral list
        self._loaded = False  # Flag to check if database has been loaded
 
        
    def _load_database(self):
        """Load the database from file or create a new one if it doesn't exist."""
        if self._loaded and self._database is not None:
            return self._database
            
        print("Loading mineral database...")
        start_time = time.time()  # Track loading time
        
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    self._database = pickle.load(f)
                self._loaded = True
                load_time = time.time() - start_time
                print(f"Database loaded in {load_time:.2f} seconds with {len(self._database)} minerals.")
                return self._database
            except Exception as e:
                print(f"Error loading database: {e}")
                self._database = {}
                self._loaded = True
                return self._database
        
        self._database = {}
        self._loaded = True
        return self._database
    
    @property
    def database(self):
        """Lazy load the database when first accessed."""
        if not self._loaded:
            return self._load_database()
        return self._database
            
    def save_database(self):
        """Save the database to file."""
        # Make sure database is loaded before saving
        if not self._loaded:
            self._load_database()
            
        try:
            start_time = time.time()
            with open(self.database_path, 'wb') as f:
                pickle.dump(self._database, f)
            save_time = time.time() - start_time
            print(f"Database saved in {save_time:.2f} seconds.")
            
            # Reset cache after saving
            self._mineral_list = None
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
            
    def get_minerals(self):
        """Get list of all minerals in the database with caching."""
        # Use cached list if available
        if self._mineral_list is not None:
            return self._mineral_list
        
        # Filter out special metadata keys (starting with __) and sort the list
        self._mineral_list = [name for name in self.database.keys() if not name.startswith('__')]
        self._mineral_list.sort()
        return self._mineral_list
        
    def get_mineral_data(self, name):
        """Get data for a specific mineral."""
        return self.database.get(name, None)
        
    def add_mineral(self, name, crystal_system=None, point_group=None, space_group=None):
        """
        Add a new mineral to the database.
        
        Parameters:
        -----------
        name : str
            Name of the mineral
        crystal_system : str, optional
            Crystal system (e.g., cubic, tetragonal)
        point_group : str, optional
            Point group (e.g., Oh, D4h)
        space_group : str, optional
            Space group (e.g., Fm-3m, P21/c)
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if name in self.database:
            return False
            
        self.database[name] = {
            'name': name,
            'crystal_system': crystal_system,
            'point_group': point_group,
            'space_group': space_group,
            'modes': [],
        }
        return True
        
    def add_mode(self, mineral_name, position, symmetry, intensity=1.0):
        """
        Add a Raman mode to a mineral.
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral
        position : float
            Wavenumber (cm-1)
        symmetry : str
            Symmetry character (e.g., A1g, Eg)
        intensity : float, optional
            Relative intensity (default: 1.0)
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if mineral_name not in self.database:
            return False
            
        mode = (float(position), str(symmetry), float(intensity))
        self.database[mineral_name]['modes'].append(mode)
        return True
        
    def get_minerals(self):
        """Get list of all minerals in the database."""
        return list(self.database.keys())
        
    def get_mineral_data(self, name):
        """Get data for a specific mineral."""
        return self.database.get(name, None)
        
    def get_modes(self, name):
        """Get Raman modes for a specific mineral."""
        print(f"DEBUG: get_modes called for mineral: {name}")
        
        # First, check for conversion cache marker
        conversion_cache_key = f"__converted_{name.replace(' ', '_')}"
        conversion_already_done = conversion_cache_key in self.database
        
        if name in self.database:
            # Look for modes in the standard key
            modes = self.database[name].get('modes', [])
            
            # If modes exist, return them directly without attempting conversion
            if modes and len(modes) > 0:
                print(f"DEBUG: Found {len(modes)} existing modes for {name}, returning without conversion")
                return modes
                
            # If no modes found and conversion hasn't been attempted, try other keys
            if (not modes or len(modes) == 0) and not conversion_already_done:
                print(f"DEBUG: No modes found in 'modes' key for {name}, looking in alternative locations...")
                
                # Check for phonon_modes and attempt conversion
                mineral_data = self.database[name]
                
                # Print all keys to help debug
                print(f"DEBUG: Available keys for {name}: {list(mineral_data.keys())}")
                
                # Attempt to convert phonon modes to Raman modes
                conversion_success = self.convert_phonon_to_raman_modes(name)
                if conversion_success:
                    print(f"DEBUG: Successfully converted phonon modes to Raman modes for {name}")
                    # Get the newly converted modes
                    modes = mineral_data.get('modes', [])
                    
                    # Save the database to persist the converted modes
                    save_success = self.save_database()
                    if save_success:
                        print(f"DEBUG: Database saved after mode conversion for {name}")
                    else:
                        print(f"DEBUG: Failed to save database after mode conversion for {name}")
                else:
                    # If conversion failed, look for modes in various possible keys
                    print(f"DEBUG: Conversion failed or no phonon data found, checking other keys for {name}")
                    for key in ['phonon_modes', 'raman_modes', 'modes_data', 'mode_list', 'peaks']:
                        if key in mineral_data:
                            value = mineral_data[key]
                            print(f"DEBUG: Found potential mode data in key '{key}', type: {type(value)}")
                            
                            # Instead of trying to convert automatically, just log the info
                            if isinstance(value, pd.DataFrame):
                                print(f"DEBUG: DataFrame found in '{key}' with shape {value.shape}")
                                print(f"DEBUG: Cannot auto-convert DataFrame to modes - try using add_example_modes")
                            elif isinstance(value, list) and len(value) > 0:
                                print(f"DEBUG: List found in '{key}' with {len(value)} items")
                                print(f"DEBUG: First item type: {type(value[0])}")
                                # Don't try auto-conversion here
            
            print(f"DEBUG: Returning {len(modes)} modes for {name}")
            return modes
        else:
            print(f"DEBUG: Mineral {name} not found in database")
        return []
        
    def import_from_peak_fitting(self, mineral_name, peak_data):
        """
        Import peak data from peak_fitting.py output.
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral to add/update
        peak_data : list
            List of peak data from peak fitting
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if mineral_name not in self.database:
            self.add_mineral(mineral_name)
            
        for peak in peak_data:
            position = peak.get('position', 0)
            # Default to A1g symmetry if not specified
            symmetry = peak.get('symmetry', 'A1g')
            intensity = peak.get('amplitude', 1.0)
            self.add_mode(mineral_name, position, symmetry, intensity)
            
        return True
        
    def import_from_pkl(self, file_path):
        """
        Import data from another pickle file.
        
        Parameters:
        -----------
        file_path : str
            Path to the pickle file
            
        Returns:
        --------
        int
            Number of minerals imported
        """
        try:
            with open(file_path, 'rb') as f:
                imported_data = pickle.load(f)
                
            if not isinstance(imported_data, dict):
                return 0
                
            count = 0
            for name, data in imported_data.items():
                if name not in self.database:
                    self.database[name] = data
                    count += 1
                    
            return count
        except Exception as e:
            print(f"Error importing database: {e}")
            return 0

    def delete_mineral(self, name):
        """
        Delete a mineral from the database.
        
        Parameters:
        -----------
        name : str
            Name of the mineral to delete
            
        Returns:
        --------
        bool
            True if mineral was deleted, False if mineral not found
        """
        if name in self.database:
            del self.database[name]
            # Clear mineral list cache after deletion
            self._mineral_list = None
            return True
        return False

    def import_from_csv(self, mineral_name, csv_type, file_path):
        """
        Import data from CSV files (dielectric_tensors, born_charges, info).
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral to add/update
        csv_type : str
            Type of CSV data ('dielectric_tensors', 'born_charges', 'info')
        file_path : str
            Path to the CSV file
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Create the mineral if it doesn't exist
            if mineral_name not in self.database:
                self.add_mineral(mineral_name)
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Process based on CSV type
            if csv_type == 'dielectric_tensors':
                # Extract the dielectric tensor
                dielectric_tensor = []
                if 'Ɛ∞' in ' '.join(df['Tensor']):
                    # Create a 3x3 dielectric tensor
                    e_inf = np.zeros((3, 3))
                    for i, row in df[df['Tensor'] == 'Ɛ∞'].iterrows():
                        comp = row['Component'].lower()
                        idx1 = 0 if 'xx' in comp else (1 if 'yy' in comp else 2)
                        idx2 = 0 if 'xx' in comp else (1 if 'yy' in comp else 2)
                        e_inf[idx1, idx2] = float(row['X'])
                        if 'Y' in df.columns and not pd.isna(row['Y']):
                            e_inf[idx1, 1] = float(row['Y'])
                        if 'Z' in df.columns and not pd.isna(row['Z']):
                            e_inf[idx1, 2] = float(row['Z'])
                    
                    dielectric_tensor = e_inf.tolist()
                
                # Store the tensor in the database
                if dielectric_tensor:
                    self.database[mineral_name]['dielectric_tensor'] = dielectric_tensor
                
                # Store the entire dataframe for future reference
                self.database[mineral_name]['dielectric_tensor_full'] = df
                
                return True
                
            elif csv_type == 'born_charges':
                # Extract Born charges
                born_charges = []
                atoms = df['Atom'].unique()
                
                for atom in atoms:
                    atom_data = df[df['Atom'] == atom]
                    
                    # Group rows for each unique atom
                    charges = []
                    for i in range(0, len(atom_data), 3):  # Each atom has 3 rows (xx, yy, zz)
                        atom_charge = np.zeros((3, 3))
                        for j in range(3):
                            if i+j < len(atom_data):
                                row = atom_data.iloc[i+j]
                                comp = row['Component'].lower()
                                idx1 = 0 if 'xx' in comp else (1 if 'yy' in comp else 2)
                                atom_charge[idx1, 0] = float(row['X'])
                                atom_charge[idx1, 1] = float(row['Y']) if 'Y' in df.columns and not pd.isna(row['Y']) else 0.0
                                atom_charge[idx1, 2] = float(row['Z']) if 'Z' in df.columns and not pd.isna(row['Z']) else 0.0
                        
                        charges.append(atom_charge.tolist())
                    
                    born_charges.append({
                        'atom': atom,
                        'charge': charges[0] if charges else []
                    })
                
                # Store the Born charges in the database
                if born_charges:
                    self.database[mineral_name]['born_charges'] = born_charges
                
                # Store the entire dataframe for future reference
                self.database[mineral_name]['born_charges_full'] = df
                
                return True
                
            elif csv_type == 'info':
                # Extract mineral information
                if not df.empty:
                    row = df.iloc[0]
                    
                    # Update space group information
                    if 'experimental_symmetry_space_group_symbol' in row:
                        space_group = row['experimental_symmetry_space_group_symbol']
                        if pd.notna(space_group):
                            self.database[mineral_name]['space_group'] = space_group
                    
                    # Derive crystal system from space group number or symbol
                    if 'experimental_symmetry_space_group_number' in row:
                        sg_number = row['experimental_symmetry_space_group_number']
                        if pd.notna(sg_number):
                            # Pass the raw space group number to _derive_crystal_system
                            # which will handle conversion and special cases
                            crystal_system = self._derive_crystal_system(sg_number)
                            if crystal_system:
                                self.database[mineral_name]['crystal_system'] = crystal_system
                    
                    # Derive point group from space group
                    if 'experimental_symmetry_space_group_symbol' in row:
                        sg_symbol = row['experimental_symmetry_space_group_symbol']
                        if pd.notna(sg_symbol):
                            point_group = self._derive_point_group(sg_symbol)
                            if point_group:
                                self.database[mineral_name]['point_group'] = point_group
                    
                    # Store chemical formula
                    if 'chemical_formula' in row:
                        formula = row['chemical_formula']
                        if pd.notna(formula):
                            self.database[mineral_name]['chemical_formula'] = formula
                
                # Store the entire dataframe for future reference
                self.database[mineral_name]['info_full'] = df
                
                return True
                
            return False
            
        except Exception as e:
            print(f"Error importing CSV: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _derive_crystal_system(self, space_group_number):
        """Derive crystal system from space group number."""
        # Skip the entire conversion process if we're not given a valid input
        if space_group_number is None:
            return ""
            
        # If space_group_number is a string, try to extract numeric value
        if isinstance(space_group_number, str):
            # Handle special cases first
            if 'xx' in space_group_number.lower():
                return ""  # Invalid space group
                
            import re
            # Look for patterns like "Exp 176, Theo 173" or "Theo 14" and extract the first number
            match = re.search(r'(\d+)', space_group_number)
            if match:
                try:
                    # Extract the first numeric value found
                    space_group_number = int(match.group(1))
                except (ValueError, TypeError):
                    return ""
            else:
                return ""
                
        # Now process the numeric space group
        try:
            space_group_number = int(space_group_number)
            if 1 <= space_group_number <= 2:
                return "Triclinic"
            elif 3 <= space_group_number <= 15:
                return "Monoclinic"
            elif 16 <= space_group_number <= 74:
                return "Orthorhombic"
            elif 75 <= space_group_number <= 142:
                return "Tetragonal"
            elif 143 <= space_group_number <= 167:
                return "Trigonal"
            elif 168 <= space_group_number <= 194:
                return "Hexagonal"
            elif 195 <= space_group_number <= 230:
                return "Cubic"
        except (ValueError, TypeError):
            # If conversion still fails, don't raise an error, just return empty string
            pass
            
        return ""
        
    def _derive_point_group(self, space_group_symbol):
        """Derive point group from space group symbol."""
        # Extended mapping from space group to point group
        sg_to_pg = {
            'P1': '1', 'P-1': '-1',  # Triclinic
            'P2': '2', 'P21': '2', 'C2': '2', 'Pm': 'm', 'Pc': 'm', 'Cm': 'm', 'Cc': 'm',
            'P2/m': '2/m', 'P21/m': '2/m', 'C2/m': '2/m', 'P2/c': '2/m', 'P21/c': '2/m', 'C2/c': '2/m',  # Monoclinic
            'P222': '222', 'P2221': '222', 'Pmmm': 'mmm', 'Pnma': 'mmm',  # Orthorhombic
            'P4': '4', 'P-4': '-4', 'P4/m': '4/m', 'P422': '422', 'P4mm': '4mm', 'P-42m': '-42m', 'P4/mmm': '4/mmm',  # Tetragonal
            'P3': '3', 'P-3': '-3', 'P321': '32', 'P3m1': '3m', 'P-3m1': '-3m',  # Trigonal
            'R-3c': '3m', 'R3c': '3m', 'R-3': '-3', 'R3': '3', 'R32': '32', 'R3m': '3m', 'R-3m': '-3m',
            'P6': '6', 'P-6': '-6', 'P6/m': '6/m', 'P622': '622', 'P6mm': '6mm', 'P-6m2': '-6m2', 'P6/mmm': '6/mmm',  # Hexagonal
            'P23': '23', 'P213': '23', 'Pm-3': 'm-3', 'Pn-3': 'm-3',
            'P432': '432', 'P-43m': '-43m', 'Pm-3m': 'm-3m', 'Pn-3m': 'm-3m', 'Fm-3m': 'm-3m', 'Fd-3m': 'm-3m',  # Cubic
        }
        # Try to find a direct match (case-insensitive, ignore spaces)
        sg_symbol_clean = space_group_symbol.replace(' ', '').upper()
        for sg, pg in sg_to_pg.items():
            if sg_symbol_clean == sg.replace(' ', '').upper():
                return pg
        # Fallback: handle common trigonal/hexagonal rhombohedral cases
        if sg_symbol_clean.startswith('R-3C') or sg_symbol_clean.startswith('R3C'):
            return '3m'
        if sg_symbol_clean.startswith('R-3M') or sg_symbol_clean.startswith('R3M'):
            return '-3m'
        if sg_symbol_clean.startswith('R-3') or sg_symbol_clean.startswith('R3'):
            return '-3'
        # Fallback: try to extract the point group from the symbol (very basic)
        if sg_symbol_clean.startswith('P3') or sg_symbol_clean.startswith('R3'):
            return '3'
        if sg_symbol_clean.startswith('P6'):
            return '6'
        if sg_symbol_clean.startswith('P4'):
            return '4'
        if sg_symbol_clean.startswith('P2'):
            return '2'
        if sg_symbol_clean.startswith('P1'):
            return '1'
        # If no match, return empty string
        return ""

    def convert_phonon_to_raman_modes(self, mineral_name):
        """Convert phonon modes to Raman modes format."""
        try:
            # Get mineral data without triggering auto-conversion
            mineral_data = self.database.get(mineral_name)
            if not mineral_data:
                print(f"DEBUG: No mineral data found for '{mineral_name}'")
                return False
                
            print(f"DEBUG: Converting phonon modes for '{mineral_name}'")
            
            # IMPORTANT: Create a unique cache key for checking if this conversion has been done
            conversion_cache_key = f"__converted_{mineral_name.replace(' ', '_')}"
            if conversion_cache_key in self.database:
                print(f"DEBUG: Conversion already performed for '{mineral_name}', skipping")
                return True
                
            # Check if modes already exist and have data - if so, skip conversion 
            if 'modes' in mineral_data and isinstance(mineral_data['modes'], list) and len(mineral_data['modes']) > 0:
                print(f"DEBUG: '{mineral_name}' already has {len(mineral_data['modes'])} modes, skipping conversion")
                
                # Mark this mineral as having been converted (if not already marked)
                if conversion_cache_key not in self.database:
                    print(f"DEBUG: Marking '{mineral_name}' as already converted")
                    self.database[conversion_cache_key] = True
                
                return True
            
            # Import pandas for dataframe operations only when needed
            import pandas as pd
            import numpy as np
            
            # Check for phonon modes in different possible keys - use a more efficient approach
            phonon_modes = None
            potential_keys = ['phonon_modes', 'phonon_mode', 'phonons', 'modes_full', 'all_modes']
            
            # First try to find any valid phonon data
            for key in potential_keys:
                if key in mineral_data:
                    value = mineral_data[key]
                    
                    # Handle DataFrames specifically
                    if isinstance(value, pd.DataFrame):
                        if not value.empty:
                            try:
                                phonon_modes = value.to_dict('records')
                                print(f"DEBUG: Converted DataFrame in '{key}' to {len(phonon_modes)} records")
                                break
                            except Exception as e:
                                print(f"DEBUG: Error converting DataFrame in '{key}': {e}")
                                continue
                    elif value:  # Non-empty list, dict, etc.
                        phonon_modes = value
                        print(f"DEBUG: Found phonon modes in key '{key}', count: {len(phonon_modes) if hasattr(phonon_modes, '__len__') else 'unknown'}")
                        break
            
            # If no valid phonon modes found through standard keys, look for mode-like data in any key
            if phonon_modes is None:
                for key, value in mineral_data.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict) and any(k in value[0] for k in ['frequency', 'position', 'mode']):
                        phonon_modes = value
                        print(f"DEBUG: Found potential mode data in key '{key}', count: {len(phonon_modes)}")
                        break
            
            # If still no modes found, return failure
            if phonon_modes is None:
                print(f"DEBUG: No phonon mode data found for '{mineral_name}' under any key")
                return False
            
            # Initialize modes list
            if 'modes' not in mineral_data:
                mineral_data['modes'] = []
                
            # Process modes more efficiently
            modes_added = 0
            processed_frequencies = set()  # Track already processed frequencies to avoid duplicates
            
            for mode in phonon_modes:
                try:
                    # Get frequency/position (try multiple possible keys efficiently)
                    pos = None
                    if isinstance(mode, dict):
                        # Try common keys in order of likelihood
                        for key in ['frequency', 'position', 'freq', 'wavenumber', 'TO_Frequency', 'Frequency']:
                            if key in mode:
                                try:
                                    pos = float(mode[key])
                                    break
                                except (ValueError, TypeError):
                                    # Try to extract number from string
                                    try:
                                        import re
                                        match = re.search(r'(\d+\.?\d*)', str(mode[key]))
                                        if match:
                                            pos = float(match.group(1))
                                            break
                                    except:
                                        pass
                    
                        # Try Mode field with frequency in parentheses
                        if pos is None and 'Mode' in mode and isinstance(mode['Mode'], str):
                            import re
                            match = re.search(r'\((\d+\.?\d*)\)', mode['Mode'])
                            if match:
                                pos = float(match.group(1))
                
                    # Skip if no valid position/frequency found
                    if pos is None:
                        continue
                        
                    # Skip duplicate frequencies (within a small tolerance)
                    duplicate = False
                    for existing_pos in processed_frequencies:
                        if abs(existing_pos - pos) < 0.5:  # 0.5 cm-1 tolerance
                            duplicate = True
                            break
                    if duplicate:
                        continue
                        
                    # Extract symmetry - prioritize specific fields
                    sym = None
                    for key in ['Activity', 'symmetry', 'sym', 'irrep', 'mode_symmetry', 'character', 'Mode']:
                        if key in mode:
                            try:
                                value = str(mode[key])
                                if key == 'Mode' and isinstance(value, str):
                                    # Extract symmetry part
                                    import re
                                    match = re.search(r'(A1g|A2g|B1g|B2g|Eg|A1u|A2u|B1u|B2u|Eu|A1|A2|B1|B2|E|T1g|T2g|T1u|T2u)', value)
                                    if match:
                                        sym = match.group(1)
                                    else:
                                        sym = value.split('(')[0].strip()
                                else:
                                    sym = value
                                if sym:
                                    break
                            except:
                                pass
                    
                    # Default symmetry if not found
                    if not sym:
                        sym = "A1g"
                    
                    # Get intensity - prioritize specific fields
                    intensity = None
                    for key in ['I_Total', 'intensity', 'raman_intensity', 'int', 'raman_int', 'amplitude']:
                        if key in mode:
                            try:
                                val = mode[key]
                                if val is not None:
                                    intensity = float(val)
                                    break
                            except:
                                pass
                    
                    # Default intensity if not found
                    if intensity is None or intensity <= 0:
                        # Check activity to determine default intensity
                        activity = None
                        for key in ['activity', 'Activity']:
                            if key in mode:
                                activity = str(mode[key]).lower()
                                break
                                
                        if activity and 'raman' in activity:
                            intensity = 0.75  # Default for Raman active modes
                        elif activity and 'ir' in activity:
                            intensity = 0.1   # Lower intensity for IR modes
                        else:
                            intensity = 0.3   # General default intensity
                    
                    # Add the mode
                    mineral_data['modes'].append((pos, sym, intensity))
                    processed_frequencies.add(pos)
                    modes_added += 1
                    
                except Exception as e:
                    print(f"DEBUG: Error processing mode: {e}")
                    continue
            
            # Sort the modes by position (ascending)
            if mineral_data['modes']:
                mineral_data['modes'].sort(key=lambda x: x[0])
            
            # Set conversion cache flag to avoid repeat conversions
            if modes_added > 0:
                self.database[conversion_cache_key] = True
                
            print(f"DEBUG: Added {modes_added} modes for '{mineral_name}'")
            return modes_added > 0
            
        except Exception as e:
            print(f"DEBUG ERROR: Exception in convert_phonon_to_raman_modes for '{mineral_name}': {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def search_minerals(self, search_term):
        """
        Search for minerals matching the search term.
        
        Parameters:
        -----------
        search_term : str
            Search term to match against mineral names and other properties.
            
        Returns:
        --------
        list
            List of matching mineral names
        """
        # Convert search_term to string to prevent errors with numeric inputs
        search_term = str(search_term).lower()
        results = []
        
        # Search in mineral names and attributes
        for name in self.database.keys():
            # Skip special metadata keys (conversion cache markers start with __)
            if name.startswith('__'):
                continue
                
            # Check name
            if search_term in name.lower():
                results.append(name)
                continue
                
            # Get mineral data
            mineral_data = self.database[name]
            
            # Skip if mineral_data is not a dictionary
            if not isinstance(mineral_data, dict):
                continue
                
            # Check chemical formula if available
            if 'chemical_formula' in mineral_data and mineral_data['chemical_formula']:
                formula = str(mineral_data['chemical_formula']).lower()
                if search_term in formula:
                    results.append(name)
                    continue
            
            # Check crystal system if available
            if 'crystal_system' in mineral_data and mineral_data['crystal_system']:
                crystal_system = str(mineral_data['crystal_system']).lower()
                if search_term in crystal_system:
                    results.append(name)
                    continue
                    
            # Check space group if available
            if 'space_group' in mineral_data and mineral_data['space_group']:
                space_group = str(mineral_data['space_group']).lower()
                if search_term in space_group:
                    results.append(name)
                    continue
        
        return results

    def get_default_crystal_model(self, mineral_name):
        """
        Generate a default crystal model based on the crystal system.
        Returns dictionary with vertices, edges, and default orientation.
        
        Parameters:
        -----------
        mineral_name : str
            Name of the mineral
            
        Returns:
        --------
        dict
            Crystal model with vertices, edges, and orientation information
        """
        mineral_data = self.get_mineral_data(mineral_name)
        if not mineral_data:
            return None
            
        crystal_system = mineral_data.get('crystal_system', '').lower()
        
        # Default model with no specific shape
        model = {
            'vertices': [],
            'edges': [],
            'faces': [],
            'crystal_system': crystal_system,
            'point_group': mineral_data.get('point_group', ''),
            'space_group': mineral_data.get('space_group', ''),
            'default_orientation': {'alpha': 0, 'beta': 0, 'gamma': 0}
        }
        
        # Generate vertices and edges based on crystal system
        if 'cubic' in crystal_system:
            # Simple cube for cubic system
            a = 1.0  # Unit length
            vertices = [
                [-a/2, -a/2, -a/2], [a/2, -a/2, -a/2], [a/2, a/2, -a/2], [-a/2, a/2, -a/2],
                [-a/2, -a/2, a/2], [a/2, -a/2, a/2], [a/2, a/2, a/2], [-a/2, a/2, a/2]
            ]
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
            ]
            model['vertices'] = vertices
            model['edges'] = edges
            model['axes'] = {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]}
            
        elif 'tetragonal' in crystal_system:
            # Tetragonal prism
            a = 1.0
            c = 1.5  # c axis typically longer
            vertices = [
                [-a/2, -a/2, -c/2], [a/2, -a/2, -c/2], [a/2, a/2, -c/2], [-a/2, a/2, -c/2],
                [-a/2, -a/2, c/2], [a/2, -a/2, c/2], [a/2, a/2, c/2], [-a/2, a/2, c/2]
            ]
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
            ]
            model['vertices'] = vertices
            model['edges'] = edges
            model['axes'] = {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]}
            
        elif 'hexagonal' in crystal_system or 'trigonal' in crystal_system:
            # Hexagonal prism
            a = 1.0
            c = 1.6  # c axis typically longer
            
            # Create a regular hexagon for the base
            import math
            vertices = []
            for i in range(6):
                angle = 2 * math.pi * i / 6
                x = a * math.cos(angle)
                y = a * math.sin(angle)
                vertices.append([x, y, -c/2])  # Bottom face
            
            for i in range(6):
                angle = 2 * math.pi * i / 6
                x = a * math.cos(angle)
                y = a * math.sin(angle)
                vertices.append([x, y, c/2])   # Top face
            
            # Create edges
            edges = []
            for i in range(6):
                edges.append([i, (i+1)%6])           # Bottom hexagon
                edges.append([i+6, ((i+1)%6)+6])     # Top hexagon
                edges.append([i, i+6])               # Connecting edges
            
            model['vertices'] = vertices
            model['edges'] = edges
            
            # Standard orientation: c axis along z, a in xy plane
            model['axes'] = {'a': [1, 0, 0], 'b': [-0.5, 0.866, 0], 'c': [0, 0, 1]}
            
        elif 'orthorhombic' in crystal_system:
            # Orthorhombic prism (rectangle)
            a, b, c = 1.0, 1.3, 1.7  # Three different lengths
            vertices = [
                [-a/2, -b/2, -c/2], [a/2, -b/2, -c/2], [a/2, b/2, -c/2], [-a/2, b/2, -c/2],
                [-a/2, -b/2, c/2], [a/2, -b/2, c/2], [a/2, b/2, c/2], [-a/2, b/2, c/2]
            ]
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
            ]
            model['vertices'] = vertices
            model['edges'] = edges
            model['axes'] = {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]}
            
        elif 'monoclinic' in crystal_system:
            # Monoclinic prism (with beta angle ≠ 90°)
            a, b, c = 1.0, 1.2, 1.4
            beta = 105 * (math.pi / 180)  # ~105° is common for monoclinic
            
            # Create vertices with beta angle between a and c
            sin_beta = math.sin(beta)
            cos_beta = math.cos(beta)
            
            vertices = [
                [-a/2, -b/2, -c/2], [a/2, -b/2, -c/2], [a/2, b/2, -c/2], [-a/2, b/2, -c/2],
                [-a/2 + c*cos_beta, -b/2, c*sin_beta - c/2], 
                [a/2 + c*cos_beta, -b/2, c*sin_beta - c/2],
                [a/2 + c*cos_beta, b/2, c*sin_beta - c/2],
                [-a/2 + c*cos_beta, b/2, c*sin_beta - c/2]
            ]
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
            ]
            model['vertices'] = vertices
            model['edges'] = edges
            model['axes'] = {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [cos_beta, 0, sin_beta]}
            model['default_orientation'] = {'alpha': 0, 'beta': beta * (180/math.pi), 'gamma': 0}
            
        elif 'triclinic' in crystal_system:
            # Triclinic crystal (all angles ≠ 90°)
            a, b, c = 1.0, 1.1, 1.3
            alpha = 95 * (math.pi / 180)  # ~95° between b and c
            beta = 105 * (math.pi / 180)  # ~105° between a and c
            gamma = 85 * (math.pi / 180)  # ~85° between a and b
            
            # This is a simplified triclinic cell representation
            sin_alpha = math.sin(alpha)
            cos_alpha = math.cos(alpha)
            sin_beta = math.sin(beta)
            cos_beta = math.cos(beta)
            sin_gamma = math.sin(gamma)
            cos_gamma = math.cos(gamma)
            
            # Create a general triclinic cell (simplified)
            vertices = [
                [0, 0, 0], 
                [a, 0, 0], 
                [a + b*cos_gamma, b*sin_gamma, 0], 
                [b*cos_gamma, b*sin_gamma, 0],
                [c*cos_beta, c*cos_alpha*sin_beta, c*sin_alpha*sin_beta], 
                [a + c*cos_beta, c*cos_alpha*sin_beta, c*sin_alpha*sin_beta],
                [a + b*cos_gamma + c*cos_beta, b*sin_gamma + c*cos_alpha*sin_beta, c*sin_alpha*sin_beta],
                [b*cos_gamma + c*cos_beta, b*sin_gamma + c*cos_alpha*sin_beta, c*sin_alpha*sin_beta]
            ]
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
            ]
            model['vertices'] = vertices
            model['edges'] = edges
            model['axes'] = {
                'a': [1, 0, 0], 
                'b': [cos_gamma, sin_gamma, 0], 
                'c': [cos_beta, cos_alpha*sin_beta, sin_alpha*sin_beta]
            }
            model['default_orientation'] = {
                'alpha': alpha * (180/math.pi), 
                'beta': beta * (180/math.pi), 
                'gamma': gamma * (180/math.pi)
            }
            
        else:
            # Default to a simple cube if crystal system is unknown
            a = 1.0
            vertices = [
                [-a/2, -a/2, -a/2], [a/2, -a/2, -a/2], [a/2, a/2, -a/2], [-a/2, a/2, -a/2],
                [-a/2, -a/2, a/2], [a/2, -a/2, a/2], [a/2, a/2, a/2], [-a/2, a/2, a/2]
            ]
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
            ]
            model['vertices'] = vertices
            model['edges'] = edges
            model['axes'] = {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]}
        
        # Add centers for drawing reference axes
        model['center'] = [0, 0, 0]
        
        # Store this model in the mineral data for future use
        mineral_data['crystal_model'] = model
        self.save_database()
        
        return model

    def update_eigenvectors(self, csv_dir="wurm_data/csv"):
        """
        Update eigenvectors for minerals from CSV files in the specified directory.
        
        This method updates both the dielectric tensor and born charges eigenvectors
        for all minerals where corresponding CSV files are found.
        
        Parameters:
        -----------
        csv_dir : str, optional
            Path to the directory containing the CSV files
            Files should be named MINERAL_born_charges.csv and MINERAL_dielectric_tensors.csv
            
        Returns:
        --------
        tuple
            (processed_count, updated_born, updated_dielectric) - counts of updated entries
        """
        # Make sure the database is loaded
        if not self._loaded:
            self._load_database()
            
        print("Starting eigenvector update process...")
        processed_count = 0
        updated_dielectric = 0
        updated_born = 0
        
        # Get all minerals in the database
        minerals = self.get_minerals()
        total_minerals = len(minerals)
        print(f"Found {total_minerals} minerals in the database")
        
        # Process each mineral
        for i, mineral in enumerate(minerals):
            # Skip special entries that start with "__"
            if mineral.startswith('__'):
                continue
                
            # Get mineral data
            mineral_data = self.get_mineral_data(mineral)
            if not isinstance(mineral_data, dict):
                continue
                
            # Create the mineral name for file lookup (uppercase and replace spaces with underscores)
            mineral_base = mineral.upper().replace(" ", "_")
            
            # For user feedback
            if (i+1) % 10 == 0 or i+1 == total_minerals:
                print(f"Processing mineral {i+1}/{total_minerals}: {mineral}")
            
            # --- Update Born Charges ---
            born_csv = os.path.join(csv_dir, f"{mineral_base}_born_charges.csv")
            if os.path.exists(born_csv):
                try:
                    df = pd.read_csv(born_csv)
                    
                    # Group by atom, take first occurrence of each (xx, yy, zz triplet)
                    eigenv_dict = {}
                    for atom in df['Atom'].unique():
                        atom_rows = df[df['Atom'] == atom]
                        # Get the first xx component row for each atom
                        xx_rows = atom_rows[atom_rows['Component'] == 'xx']
                        if not xx_rows.empty:
                            first_row = xx_rows.iloc[0]
                            # Check if Eig columns exist
                            if all(col in first_row.index for col in ['Eig1', 'Eig2', 'Eig3']):
                                eigenv = [float(first_row['Eig1']), float(first_row['Eig2']), float(first_row['Eig3'])]
                                eigenv_dict.setdefault(atom, []).append(eigenv)
                    
                    # Now update the database if we have born_charges
                    if 'born_charges' in mineral_data and eigenv_dict:
                        born_updated = False
                        
                        # If born_charges is a DataFrame, convert to list of dicts
                        if isinstance(mineral_data['born_charges'], pd.DataFrame):
                            try:
                                mineral_data['born_charges'] = mineral_data['born_charges'].to_dict('records')
                            except:
                                # If conversion fails, create empty list
                                mineral_data['born_charges'] = []
                        
                        # Ensure it's a list
                        if not isinstance(mineral_data['born_charges'], list):
                            mineral_data['born_charges'] = []
                        
                        # Update each atom entry
                        for atom_entry in mineral_data['born_charges']:
                            if isinstance(atom_entry, dict) and 'atom' in atom_entry:
                                atom_name = atom_entry['atom']
                                if atom_name in eigenv_dict:
                                    atom_entry['EigenV'] = eigenv_dict[atom_name][0]  # Use first set of eigenvectors
                                    born_updated = True
                        
                        if born_updated:
                            updated_born += 1
                except Exception as e:
                    print(f"Error processing {born_csv}: {e}")
            
            # --- Update Dielectric Tensor ---
            diel_csv = os.path.join(csv_dir, f"{mineral_base}_dielectric_tensors.csv")
            if os.path.exists(diel_csv):
                try:
                    df = pd.read_csv(diel_csv)
                    
                    # Only use the rows where Tensor == 'Ɛ∞'
                    diel_rows = df[df['Tensor'] == 'Ɛ∞']
                    
                    # Get the eigenvectors for xx, yy, zz
                    eigenv = []
                    if not diel_rows.empty:
                        for comp in ['xx', 'yy', 'zz']:
                            row = diel_rows[diel_rows['Component'] == comp]
                            if not row.empty:
                                row = row.iloc[0]
                                if all(col in row.index for col in ['Eig1', 'Eig2', 'Eig3']):
                                    eigenv.append([float(row['Eig1']), float(row['Eig2']), float(row['Eig3'])])
                    
                    # Update the database if we have eigenvectors
                    if eigenv and len(eigenv) == 3:
                        if 'dielectric_tensor' in mineral_data:
                            # Convert to dict if it's a numpy array
                            if isinstance(mineral_data['dielectric_tensor'], np.ndarray):
                                mineral_data['dielectric_tensor'] = {'tensor': mineral_data['dielectric_tensor'].tolist()}
                            
                            # Convert to dict if it's a list
                            if isinstance(mineral_data['dielectric_tensor'], list):
                                mineral_data['dielectric_tensor'] = {'tensor': mineral_data['dielectric_tensor']}
                            
                            # Now add the eigenvectors
                            if isinstance(mineral_data['dielectric_tensor'], dict):
                                mineral_data['dielectric_tensor']['EigenV'] = eigenv
                                updated_dielectric += 1
                        else:
                            # Create new dielectric tensor entry
                            mineral_data['dielectric_tensor'] = {'EigenV': eigenv}
                            updated_dielectric += 1
                except Exception as e:
                    print(f"Error processing {diel_csv}: {e}")
            
            # Count as processed if we updated either born charges or dielectric tensor
            if os.path.exists(born_csv) or os.path.exists(diel_csv):
                processed_count += 1
        
        # Save the database after all updates
        print(f"Saving updates to database...")
        success = self.save_database()
        
        # Print summary
        print("\nEigenvector Update Summary:")
        print(f"Total minerals processed: {processed_count}/{total_minerals}")
        print(f"Updated Born charges: {updated_born}")
        print(f"Updated Dielectric tensors: {updated_dielectric}")
        print(f"Database save {'successful' if success else 'failed'}")
        
        return (processed_count, updated_born, updated_dielectric)

class MineralDatabaseGUI:
    """GUI for mineral database management."""
    
    def __init__(self, parent=None):
        """Initialize the database GUI."""
        # Set the clam theme for better appearance
        style = ttk.Style()
        style.theme_use("clam")
        
        self.db = MineralDatabase()
        
        # Create window if no parent is provided
        if parent is None:
            # This is the root window for standalone use
            self.is_standalone = True
            self.window = tk.Tk()
        else:
            # This is being called from another application
            self.is_standalone = False
            self.window = tk.Toplevel(parent)
            
        # Set the 'clam' theme to make the GUI consistent with other windows
        style = ttk.Style(self.window)
        style.theme_use('clam')
        
        # Configure font sizes - using default system sizes
        default_font = None
        button_font = None
        
        # Configure minimal styling while maintaining default font sizes
        self.style = style
        self.style.configure('TLabelframe', borderwidth=1)
        self.style.configure('TLabelframe.Label')
        self.style.configure('TButton')
        self.style.configure('Small.TButton')
        self.style.configure('TLabel')
        self.style.configure('TEntry')
        self.style.configure('Treeview')
        self.style.configure('Treeview.Heading')
        self.style.configure('TCheckbutton')
        
        self.create_gui()
        
    def create_gui(self):
        """Create the GUI elements."""
        # Main frame
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - mineral list and controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Controls at the top
        control_frame = ttk.LabelFrame(left_panel, text="Database Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Add Mineral", command=self.add_mineral).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Delete Mineral", command=self.delete_mineral).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Import from Peak Fitting", command=self.import_from_peak_fitting).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Import from PKL File", command=self.import_from_pkl).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Save Database", command=self.save_database).pack(fill=tk.X, pady=2)
        
        # Mineral list
        list_frame = ttk.LabelFrame(left_panel, text="Minerals", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.mineral_listbox = tk.Listbox(list_frame)
        self.mineral_listbox.pack(fill=tk.BOTH, expand=True)
        self.mineral_listbox.bind('<<ListboxSelect>>', self.on_mineral_select)
        
        # Right panel - mineral details
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Mineral information with direct labels
        info_frame = ttk.LabelFrame(right_panel, text="Mineral Information", padding=5)
        info_frame.pack(fill=tk.X, pady=(0, 10))  # Increase bottom padding
        
        # We'll still use StringVars for initial setup but configure the labels directly
        self.mineral_name_var = tk.StringVar(value="")
        self.crystal_system_var = tk.StringVar(value="")
        self.point_group_var = tk.StringVar(value="")
        self.space_group_var = tk.StringVar(value="")
        self.raman_modes_count_var = tk.StringVar(value="")
        self.phonon_data_var = tk.StringVar(value="")
        self.key_features_var = tk.StringVar(value="")
        
        # Create a simple grid layout for info
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1
        ttk.Label(info_grid, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.name_label = ttk.Label(info_grid, text="", width=25, relief="solid", borderwidth=1, background="white")
        self.name_label.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        
        ttk.Label(info_grid, text="Crystal System:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=3)
        self.crystal_system_label = ttk.Label(info_grid, text="", width=25, relief="solid", borderwidth=1, background="white")
        self.crystal_system_label.grid(row=0, column=3, sticky=tk.W+tk.E, padx=5, pady=3)
        
        # Row 2
        ttk.Label(info_grid, text="Point Group:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.point_group_label = ttk.Label(info_grid, text="", width=25, relief="solid", borderwidth=1, background="white")
        self.point_group_label.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        
        ttk.Label(info_grid, text="Space Group:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=3)
        self.space_group_label = ttk.Label(info_grid, text="", width=25, relief="solid", borderwidth=1, background="white")
        self.space_group_label.grid(row=1, column=3, sticky=tk.W+tk.E, padx=5, pady=3)
        
        # Row 3 - More compact layout for Available Data
        ttk.Label(info_grid, text="Raman Modes:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.raman_modes_label = ttk.Label(info_grid, text="", width=8, relief="solid", borderwidth=1, background="white")
        self.raman_modes_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(info_grid, text="Available Data:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=3)
        self.available_data_label = ttk.Label(info_grid, text="", width=25, relief="solid", borderwidth=1, background="white")
        self.available_data_label.grid(row=2, column=3, sticky=tk.W+tk.E, padx=5, pady=3)
        
        # Row 4 with reduced height
        ttk.Label(info_grid, text="Key Features:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.key_features_label = ttk.Label(info_grid, text="", width=60, relief="solid", borderwidth=1, background="white")
        self.key_features_label.grid(row=3, column=1, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=3)
        
        # Control buttons
        button_frame = ttk.Frame(info_grid)
        button_frame.grid(row=0, column=4, rowspan=4, padx=5, pady=2, sticky=tk.N+tk.S)
        
        ttk.Button(button_frame, text="Edit Entry", command=self.edit_mineral_entry).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Show Info Popup", 
                  command=lambda: self.show_info_popup_for_selected()).pack(fill=tk.X, pady=2)
                  
        # Configure column weights for proper expansion
        for i in range(4):
            info_grid.columnconfigure(i, weight=1)
            
        # Add a frame for advanced tensor viewing options - moved before Raman modes
        advanced_frame = ttk.LabelFrame(right_panel, text="Advanced Properties", padding=5)
        advanced_frame.pack(fill=tk.X, pady=(5, 5))  # Add top padding too
        
        advanced_buttons = ttk.Frame(advanced_frame)
        advanced_buttons.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(advanced_buttons, text="View Dielectric Tensor", 
                  command=self.view_dielectric_tensor).pack(side=tk.LEFT, padx=2)
        ttk.Button(advanced_buttons, text="View Born Charges", 
                  command=self.view_born_charges).pack(side=tk.LEFT, padx=2)
        ttk.Button(advanced_buttons, text="View Full Phonon Modes", 
                  command=self.view_phonon_modes).pack(side=tk.LEFT, padx=2)
        ttk.Button(advanced_buttons, text="Debug Modes", 
                  command=self.debug_mineral_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(advanced_buttons, text="Add Example Modes", 
                  command=self.add_example_modes).pack(side=tk.LEFT, padx=2)
        ttk.Button(advanced_buttons, text="Batch Convert Phonon Modes", 
                  command=self.batch_convert_phonon_to_raman).pack(side=tk.LEFT, padx=2)
        ttk.Button(advanced_buttons, text="Update Eigenvectors from CSV", 
                  command=self.update_eigenvectors).pack(side=tk.LEFT, padx=2)
        
        # Raman modes panel - reduce vertical space
        modes_frame = ttk.LabelFrame(right_panel, text="Raman Modes", padding=5)
        modes_frame.pack(fill=tk.BOTH, pady=(0, 5))
        
        # Buttons for mode management
        mode_buttons = ttk.Frame(modes_frame)
        mode_buttons.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(mode_buttons, text="Add Mode", command=self.add_mode).pack(side=tk.LEFT, padx=2)
        ttk.Button(mode_buttons, text="Edit Mode", command=self.edit_mode).pack(side=tk.LEFT, padx=2)
        ttk.Button(mode_buttons, text="Delete Mode", command=self.delete_mode).pack(side=tk.LEFT, padx=2)
        
        # Table for modes - set a fixed height to limit vertical space
        columns = ("position", "symmetry", "intensity")
        
        # Create a frame to hold the treeview and scrollbar
        tree_frame = ttk.Frame(modes_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the treeview with scrollbar
        self.modes_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=5)
        
        # Add vertical scrollbar
        modes_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.modes_tree.yview)
        modes_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure the treeview to use the scrollbar
        self.modes_tree.configure(yscrollcommand=modes_scrollbar.set)
        self.modes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure the columns
        self.modes_tree.heading("position", text="Position (cm⁻¹)")
        self.modes_tree.heading("symmetry", text="Character")
        self.modes_tree.heading("intensity", text="Relative Intensity")
        
        self.modes_tree.column("position", width=150)
        self.modes_tree.column("symmetry", width=150)
        self.modes_tree.column("intensity", width=150)
        
        # Visualization panel - give it more vertical space
        viz_frame = ttk.LabelFrame(right_panel, text="Visualization", padding=5)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
        # Create a figure for the plot with high DPI and a custom style
        plt.style.use('seaborn-v0_8-whitegrid')
        # Decrease figure height from 6 to 5 and set better figure parameters
        self.fig = plt.figure(figsize=(6, 5), dpi=100, facecolor='#f8f8f8', constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        # Adjust the plot position to leave more room for labels
        self.ax.set_position([0.12, 0.15, 0.80, 0.78])
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Add toolbar with configuration - make it more compact
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X)
        
        # Fix for matplotlib compatibility issues
        try:
            self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        except TypeError as e:
            # More comprehensive fallback for matplotlib version differences
            print(f"DEBUG: NavigationToolbar compatibility issue: {e}")
            class CompatNavToolbar(NavigationToolbar2Tk):
                def __init__(self, canvas, parent):
                    # For older matplotlib that uses a different constructor
                    self.canvas = canvas
                    self.window = parent
                    NavigationToolbar2Tk.__init__(self, canvas, parent)
                    
                def _Button(self, text, image_file, toggle, command):
                    # Handle both older and newer versions
                    try:
                        # Newer versions expect a tooltip parameter (not supported in older versions)
                        return NavigationToolbar2Tk._Button(self, text, image_file, toggle, command, tooltip="")
                    except TypeError:
                        # Older versions don't use tooltip
                        return NavigationToolbar2Tk._Button(self, text, image_file, toggle, command)
                        
                def _update_buttons_checked(self):
                    # Some older versions don't have this method
                    if hasattr(NavigationToolbar2Tk, '_update_buttons_checked'):
                        try:
                            NavigationToolbar2Tk._update_buttons_checked(self)
                        except:
                            pass
        
            self.toolbar = CompatNavToolbar(self.canvas, toolbar_frame)
        
        self.toolbar.update()
        
        # Configure the toolbar buttons to be smaller
        for child in self.toolbar.winfo_children():
            if isinstance(child, tk.Button):
                child.configure(padx=2, pady=2)
                
        # Add additional plot controls under the toolbar - make more compact
        controls_frame = ttk.Frame(viz_frame)
        controls_frame.pack(fill=tk.X, pady=2)
        
        # Create a row for controls
        control_row = ttk.Frame(controls_frame)
        control_row.pack(fill=tk.X)
        
        # Add x-range control
        ttk.Label(control_row, text="X-Range:").pack(side=tk.LEFT, padx=(0, 5))
        self.x_min_var = tk.StringVar(value="50")
        self.x_max_var = tk.StringVar(value="1500")
        ttk.Entry(control_row, textvariable=self.x_min_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(control_row, text="-").pack(side=tk.LEFT)
        ttk.Entry(control_row, textvariable=self.x_max_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # Add peak width control
        ttk.Label(control_row, text="Peak Width:").pack(side=tk.LEFT, padx=(10, 5))
        self.peak_width_var = tk.StringVar(value="5")
        ttk.Entry(control_row, textvariable=self.peak_width_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # Add toggle grid button
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_row, text="Grid", variable=self.show_grid_var, 
                       command=self.update_current_plot).pack(side=tk.LEFT, padx=(10, 5))
        
        # Add line thickness control
        ttk.Label(control_row, text="Line:").pack(side=tk.LEFT, padx=(10, 5))
        self.line_width_var = tk.DoubleVar(value=1.0)
        line_slider = ttk.Scale(control_row, from_=0.5, to=2.5, length=60, 
                              variable=self.line_width_var, orient=tk.HORIZONTAL,
                              command=lambda x: self.update_current_plot())
        line_slider.pack(side=tk.LEFT, padx=2)
        
        # Add update plot button
        ttk.Button(control_row, text="Update Plot", command=self.update_current_plot, 
                  style="Small.TButton").pack(side=tk.RIGHT, padx=5)
        
        # Add event callbacks for plot navigation
        self.canvas.mpl_connect('draw_event', self.on_plot_navigate)
        
        # Callback for key presses on the plot
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Populate the list
        self.update_mineral_list()
        
        # Select a random mineral by default if available
        self.select_random_mineral()
        
    def update_mineral_list(self):
        """Update the mineral list from the database."""
        self.mineral_listbox.delete(0, tk.END)
        for mineral in sorted(self.db.get_minerals()):
            self.mineral_listbox.insert(tk.END, mineral)
            
    def on_mineral_select(self, event=None):
        """Handle mineral selection from the list."""
        print("DEBUG: on_mineral_select called")
        
        # If no selection or empty listbox, return
        if not self.mineral_listbox.curselection():
            print("DEBUG: No selection in mineral_listbox")
            return
        
        try:
            # Get the selected mineral
            index = self.mineral_listbox.curselection()[0]
            mineral_name = self.mineral_listbox.get(index)
            print(f"DEBUG: Selected mineral_name = {mineral_name}")
            
            # Get mineral data
            mineral_data = self.db.get_mineral_data(mineral_name)
            if not mineral_data:
                print("DEBUG: No mineral_data found")
                return
                
            # Skip entries that are not dictionaries (marker entries)
            if not isinstance(mineral_data, dict):
                print(f"DEBUG: Skipping {mineral_name} - not a valid mineral record (type: {type(mineral_data)})")
                messagebox.showinfo("Information", f"{mineral_name} is a special marker entry, not a mineral record.")
                return
            
            print(f"DEBUG: mineral_data has {len(mineral_data)} keys: {list(mineral_data.keys())}")
            
            # Get modes data - don't try auto-conversion
            modes = self.db.get_modes(mineral_name)
            mode_count = len(modes) if isinstance(modes, list) else 0
            
            # For now, just add example modes if there are none
            if mode_count == 0:
                print(f"DEBUG: No modes found for {mineral_name}, adding example modes")
                self.add_example_modes_for_mineral(mineral_name)
                # Refresh modes after adding examples
                modes = self.db.get_modes(mineral_name)
                mode_count = len(modes) if isinstance(modes, list) else 0
                print(f"DEBUG: Added {mode_count} example modes")
            
            # Extract information with fallbacks
            crystal_system = mineral_data.get('crystal_system', '')
            point_group = mineral_data.get('point_group', '')
            space_group = mineral_data.get('space_group', '')
            print(f"DEBUG: crystal_system={crystal_system}, point_group={point_group}, space_group={space_group}")
            
            # Create label text directly (for debugging)
            display_name = mineral_name
            if 'chemical_formula' in mineral_data:
                formula = mineral_data.get('chemical_formula')
                print(f"DEBUG: chemical_formula type = {type(formula)}, value = {formula}")
                if formula is not None and formula != '':
                    display_name = f"{mineral_name} ({formula})"
                    print(f"DEBUG: Set Name with formula: {display_name}")
            
            # Update text labels (but don't show popup)
            try:
                self.name_label.config(text=display_name)
                self.crystal_system_label.config(text=str(crystal_system))
                self.point_group_label.config(text=str(point_group))
                self.space_group_label.config(text=str(space_group))
                self.raman_modes_label.config(text=str(mode_count))
                
                # Available data checks
                data_types = []
                for key in ['phonon_modes', 'dielectric_tensor', 'born_charges']:
                    if key in mineral_data:
                        # Safely check if data exists without triggering DataFrame truth value error
                        if isinstance(mineral_data[key], pd.DataFrame):
                            if not mineral_data[key].empty:
                                data_type = key.split('_')[0].capitalize()
                                data_types.append(data_type)
                        elif mineral_data[key] is not None:
                            data_type = key.split('_')[0].capitalize()
                            data_types.append(data_type)
                
                self.available_data_label.config(text=", ".join(data_types) if data_types else "None")
                
                # Key features
                if mode_count > 0:
                    # Sort modes by intensity to find strongest peaks
                    sorted_modes = sorted(modes, key=lambda x: x[2], reverse=True)
                    top_modes = sorted_modes[:3] if len(sorted_modes) >= 3 else sorted_modes
                    positions = [f"{pos:.0f}" for pos, _, _ in top_modes]
                    key_features = f"Strong peaks at {', '.join(positions)} cm⁻¹"
                else:
                    key_features = "No peaks identified"
                
                self.key_features_label.config(text=key_features)
                
            except Exception as e:
                print(f"DEBUG ERROR: Failed to update labels: {e}")
                import traceback
                traceback.print_exc()
                
            print("DEBUG: Calling update_modes_table")
            # Update modes table
            try:
                self.update_modes_table(mineral_name)
                print("DEBUG: update_modes_table completed successfully")
            except Exception as e:
                print(f"DEBUG ERROR: Failed to update modes table: {e}")
                import traceback
                traceback.print_exc()
            
            print("DEBUG: Calling update_plot")
            # Update plot
            try:
                self.update_plot(mineral_name)
                print("DEBUG: update_plot completed successfully")
            except Exception as e:
                print(f"DEBUG ERROR: Failed to update plot: {e}")
                import traceback
                traceback.print_exc()
                try:
                    messagebox.showinfo("Plot Error", 
                        f"Could not update the visualization plot. Error: {str(e)}")
                except:
                    print("DEBUG ERROR: Failed to show error messagebox")
                
        except Exception as e:
            print(f"DEBUG ERROR: Unexpected error in on_mineral_select: {e}")
            import traceback
            traceback.print_exc()
            
    def update_modes_table(self, mineral_name):
        """Update the modes table for the selected mineral."""
        try:
            # Clear the tree
            self.modes_tree.delete(*self.modes_tree.get_children())
            
            # Get the modes
            modes = self.db.get_modes(mineral_name)
            if not modes:
                print("DEBUG: No modes to display in table")
                return
                
            # Add modes to the tree
            for mode in modes:
                try:
                    position, symmetry, intensity = mode
                    self.modes_tree.insert('', tk.END, values=(f"{position:.2f}", symmetry, f"{intensity:.3f}"))
                except Exception as e:
                    print(f"DEBUG: Error adding mode to table: {e}")
                    continue
        except Exception as e:
            print(f"DEBUG: Error in update_modes_table: {e}")
            import traceback
            traceback.print_exc()
            
    def show_info_popup_for_selected(self):
        """Show info popup for the currently selected mineral."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        mineral_data = self.db.get_mineral_data(mineral_name)
        
        if not mineral_data:
            messagebox.showwarning("Warning", f"No data found for {mineral_name}.")
            return
            
        self.show_info_popup(mineral_name, mineral_data)
        
    def show_info_popup(self, mineral_name, mineral_data):
        """Show mineral information in a simple popup window."""
        try:
            # Create popup window
            popup = tk.Toplevel(self.window)
            popup.title(f"Mineral Information: {mineral_name}")
            popup.geometry("500x300")
            popup.transient(self.window)
            
            # Extract information safely without triggering DataFrame truth value errors
            crystal_system = str(mineral_data.get('crystal_system', 'Not specified'))
            point_group = str(mineral_data.get('point_group', 'Not specified'))
            space_group = str(mineral_data.get('space_group', 'Not specified'))
            
            # Handle formula safely
            name_display = mineral_name
            if 'chemical_formula' in mineral_data:
                formula = mineral_data.get('chemical_formula')
                if formula is not None and formula != '':  # Avoid checking truth value directly
                    name_display = f"{mineral_name} ({formula})"
                
            # Create text widget to display information
            info_text = tk.Text(popup, wrap=tk.WORD, height=15, width=60)
            info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Insert information
            info_text.insert(tk.END, f"Name: {name_display}\n\n")
            info_text.insert(tk.END, f"Crystal System: {crystal_system}\n")
            info_text.insert(tk.END, f"Point Group: {point_group}\n")
            info_text.insert(tk.END, f"Space Group: {space_group}\n\n")
            
            # Add mode information safely
            modes = self.db.get_modes(mineral_name)
            mode_count = len(modes) if isinstance(modes, list) else 0
            info_text.insert(tk.END, f"Raman Modes: {mode_count}\n\n")
            
            # Add key modes
            if mode_count > 0:
                info_text.insert(tk.END, "Key Modes:\n")
                sorted_modes = sorted(modes, key=lambda x: x[2], reverse=True)
                for i, (pos, sym, intensity) in enumerate(sorted_modes[:5]):
                    info_text.insert(tk.END, f"  {pos:.1f} cm⁻¹ ({sym}): {intensity:.3f}\n")
                    if i >= 4:  # Only show the first 5
                        break
            
            # Check for additional data
            data_sections = []
            if 'phonon_modes' in mineral_data:
                phonon_data = mineral_data['phonon_modes']
                if isinstance(phonon_data, pd.DataFrame) and not phonon_data.empty:
                    data_sections.append(f"Phonon Modes: {len(phonon_data)} modes available")
                elif isinstance(phonon_data, list) and phonon_data:
                    data_sections.append(f"Phonon Modes: {len(phonon_data)} modes available")
            
            if 'dielectric_tensor' in mineral_data and mineral_data['dielectric_tensor']:
                data_sections.append("Dielectric Tensor: Available")
                
            if 'born_charges' in mineral_data and mineral_data['born_charges']:
                data_sections.append("Born Charges: Available")
                
            if data_sections:
                info_text.insert(tk.END, "\nAdditional Data:\n")
                for section in data_sections:
                    info_text.insert(tk.END, f"  {section}\n")
                
            # Make text widget read-only
            info_text.config(state=tk.DISABLED)
            
            # Add close button
            ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)
            
        except Exception as e:
            print(f"DEBUG ERROR: Error in show_info_popup: {e}")
            import traceback
            traceback.print_exc()

    def add_example_modes_for_mineral(self, mineral_name):
        """Add example modes for a mineral that has no modes defined."""
        print(f"DEBUG: Adding example modes for {mineral_name}")
        
        # Create some example modes based on mineral name characteristics
        modes = []
        
        # Check if mineral name contains common elements to create somewhat realistic modes
        name_upper = mineral_name.upper()
        
        if "QUARTZ" in name_upper or "SIO2" in name_upper:
            # Quartz-like modes
            modes = [
                (128.0, "A1", 0.5),
                (206.0, "E", 0.7),
                (464.0, "A1", 1.0),
                (696.0, "E", 0.2),
                (796.0, "E", 0.3),
                (1085.0, "A1", 0.5)
            ]
        elif "CALCITE" in name_upper or "ARAGONITE" in name_upper or "CACO3" in name_upper:
            # Carbonate modes
            modes = [
                (154.0, "Eg", 0.3),
                (281.0, "Eg", 0.4),
                (712.0, "A1g", 1.0),
                (1087.0, "A1g", 0.8)
            ]
        elif "TITANIUM" in name_upper or "ANATASE" in name_upper or "TIO2" in name_upper:
            # Titanium oxide modes
            modes = [
                (144.0, "Eg", 1.0),
                (197.0, "Eg", 0.2),
                (399.0, "B1g", 0.3),
                (513.0, "A1g", 0.4),
                (639.0, "Eg", 0.5)
            ]
        elif "IRON" in name_upper or "HEMATITE" in name_upper or "FE2O3" in name_upper:
            # Iron oxide modes
            modes = [
                (226.0, "A1g", 0.7),
                (245.0, "Eg", 0.5),
                (292.0, "Eg", 0.8),
                (411.0, "Eg", 0.4),
                (498.0, "A1g", 0.6),
                (610.0, "Eg", 0.3)
            ]
        else:
            # Generic random peaks for any other mineral type
            import random
            # Generate 3-7 random modes
            num_modes = random.randint(3, 7)
            symmetries = ["A1g", "A2g", "B1g", "B2g", "Eg"]
            
            for _ in range(num_modes):
                position = random.uniform(100, 1200)  # Random position between 100-1200 cm-1
                symmetry = random.choice(symmetries)
                intensity = random.uniform(0.3, 1.0)
                modes.append((position, symmetry, intensity))
                
            # Sort by position
            modes.sort(key=lambda x: x[0])
        
        # Add the modes to the database
        if modes:
            print(f"DEBUG: Adding {len(modes)} example modes to {mineral_name}")
            mineral_data = self.db.get_mineral_data(mineral_name)
            if mineral_data:
                mineral_data['modes'] = modes
                print(f"DEBUG: Example modes added successfully")
                return True
            else:
                print(f"DEBUG: Failed to add example modes - mineral data not found")
        
        return False

    def add_mineral(self):
        """Add a new mineral to the database."""
        name = simpledialog.askstring("Add Mineral", "Enter mineral name:")
        if not name:
            return
            
        crystal_system = simpledialog.askstring("Crystal System", "Enter crystal system (optional):")
        point_group = simpledialog.askstring("Point Group", "Enter point group (optional):")
        space_group = simpledialog.askstring("Space Group", "Enter space group (optional):")
        
        success = self.db.add_mineral(name, crystal_system, point_group, space_group)
        if success:
            self.update_mineral_list()
            messagebox.showinfo("Success", f"Mineral '{name}' added successfully.")
        else:
            messagebox.showerror("Error", f"Mineral '{name}' already exists.")
            
    def edit_mineral_entry(self):
        """Open a dialog to edit all parameters of the selected mineral."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        
        # Get current mineral data
        mineral_data = self.db.get_mineral_data(mineral_name)
        if not mineral_data or not isinstance(mineral_data, dict):
            messagebox.showwarning("Warning", f"Cannot edit {mineral_name}. Invalid data format.")
            return
        
        # Create a new dialog window
        dialog = tk.Toplevel(self.window)
        dialog.title(f"Edit Mineral Entry - {mineral_name}")
        dialog.geometry("600x500")
        dialog.transient(self.window)
        dialog.grab_set()  # Make dialog modal
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for different tab sections
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Basic Info tab
        basic_tab = ttk.Frame(notebook)
        notebook.add(basic_tab, text="Basic Information")
        
        # Grid for the fields
        basic_grid = ttk.Frame(basic_tab, padding=10)
        basic_grid.pack(fill=tk.BOTH, expand=True)
        
        # Name field (display only, can't change the key)
        ttk.Label(basic_grid, text="Mineral Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        name_var = tk.StringVar(value=mineral_name)
        ttk.Entry(basic_grid, textvariable=name_var, state='readonly', width=30).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Chemical formula
        ttk.Label(basic_grid, text="Chemical Formula:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        formula_var = tk.StringVar(value=mineral_data.get('chemical_formula', ''))
        ttk.Entry(basic_grid, textvariable=formula_var, width=30).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Crystal System
        ttk.Label(basic_grid, text="Crystal System:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        crystal_system_var = tk.StringVar(value=mineral_data.get('crystal_system', ''))
        crystal_systems = ['', 'Cubic', 'Hexagonal', 'Trigonal', 'Tetragonal', 'Orthorhombic', 'Monoclinic', 'Triclinic']
        ttk.Combobox(basic_grid, textvariable=crystal_system_var, values=crystal_systems, width=28).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Point Group (Hermann-Mauguin)
        ttk.Label(basic_grid, text="Point Group (H-M):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        hm_point_group_var = tk.StringVar(value=mineral_data.get('hermann_mauguin_point_group', ''))
        hm_point_groups = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm', 
                         '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-62m', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        ttk.Combobox(basic_grid, textvariable=hm_point_group_var, values=hm_point_groups, width=28).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Point Group (Schoenflies)
        ttk.Label(basic_grid, text="Point Group (Schoenflies):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        schoenflies_point_group_var = tk.StringVar(value=mineral_data.get('schoenflies_point_group', ''))
        schoenflies_point_groups = ['C1', 'Ci', 'C2', 'Cs', 'C2h', 'D2', 'C2v', 'D2h', 'C4', 'S4', 'C4h', 'D4', 'C4v', 'D2d', 'D4h',
                                  'C3', 'C3i', 'D3', 'C3v', 'D3d', 'C6', 'C3h', 'C6h', 'D6', 'C6v', 'D3h', 'D6h', 'T', 'Th', 'O', 'Td', 'Oh']
        ttk.Combobox(basic_grid, textvariable=schoenflies_point_group_var, values=schoenflies_point_groups, width=28).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Space Group
        ttk.Label(basic_grid, text="Space Group:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        space_group_var = tk.StringVar(value=mineral_data.get('space_group', ''))
        ttk.Entry(basic_grid, textvariable=space_group_var, width=30).grid(row=5, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Auto-update point groups based on selection
        # Create mapping dictionaries for point group conversions
        hm_to_schoenflies = {
            '1': 'C1', '-1': 'Ci', '2': 'C2', 'm': 'Cs', '2/m': 'C2h', '222': 'D2', 'mm2': 'C2v', 'mmm': 'D2h',
            '4': 'C4', '-4': 'S4', '4/m': 'C4h', '422': 'D4', '4mm': 'C4v', '-42m': 'D2d', '4/mmm': 'D4h',
            '3': 'C3', '-3': 'C3i', '32': 'D3', '3m': 'C3v', '-3m': 'D3d', '6': 'C6', '-6': 'C3h', '6/m': 'C6h',
            '622': 'D6', '6mm': 'C6v', '-62m': 'D3h', '6/mmm': 'D6h', '23': 'T', 'm-3': 'Th',
            '432': 'O', '-43m': 'Td', 'm-3m': 'Oh'
        }
        schoenflies_to_hm = {v: k for k, v in hm_to_schoenflies.items()}
        
        # Helper function to update the other point group notation
        def update_point_group_notations(source):
            if source == 'hm':
                # Update Schoenflies from Hermann-Mauguin
                hm_value = hm_point_group_var.get()
                if hm_value in hm_to_schoenflies:
                    schoenflies_point_group_var.set(hm_to_schoenflies[hm_value])
            else:
                # Update Hermann-Mauguin from Schoenflies
                schoenflies_value = schoenflies_point_group_var.get()
                if schoenflies_value in schoenflies_to_hm:
                    hm_point_group_var.get() != schoenflies_to_hm[schoenflies_value] and hm_point_group_var.set(schoenflies_to_hm[schoenflies_value])
        
        # Add trace callbacks to automatically update related fields
        hm_point_group_var.trace_add('write', lambda *args: update_point_group_notations('hm'))
        schoenflies_point_group_var.trace_add('write', lambda *args: update_point_group_notations('schoenflies'))
        
        # Add Advanced tab for other properties
        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text="Advanced Properties")
        
        # Create a text widget for advanced editing (JSON format)
        ttk.Label(advanced_tab, text="Advanced properties (JSON format):", padding=5).pack(anchor=tk.W)
        
        # Create a filtered copy of the mineral data for advanced editing
        import json
        advanced_data = {}
        # Include only selected fields that are not in the basic tab
        for key, value in mineral_data.items():
            if key not in ['name', 'chemical_formula', 'crystal_system', 'point_group', 'space_group', 
                          'hermann_mauguin_point_group', 'schoenflies_point_group', 'modes']:
                if isinstance(value, (str, int, float, bool, list, dict)) and key != 'modes':
                    advanced_data[key] = value
        
        # Create text widget with scrollbar
        advanced_frame = ttk.Frame(advanced_tab, padding=5)
        advanced_frame.pack(fill=tk.BOTH, expand=True)
        
        advanced_text = tk.Text(advanced_frame, wrap=tk.WORD, height=15, width=60)
        advanced_scrollbar = ttk.Scrollbar(advanced_frame, orient="vertical", command=advanced_text.yview)
        advanced_text.configure(yscrollcommand=advanced_scrollbar.set)
        
        advanced_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        advanced_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert the JSON data
        try:
            # Format advanced data as JSON and insert if not empty
            if advanced_data:
                # Handle NumPy arrays for JSON serialization
                def numpy_handler(obj):
                    import numpy as np
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, (np.complex, np.complex64, np.complex128)):
                        return {"real": obj.real, "imag": obj.imag}
                    else:
                        return str(obj)
                
                # Use the custom encoder to handle NumPy arrays
                advanced_text.insert(tk.END, json.dumps(advanced_data, indent=2, default=numpy_handler))
            else:
                # If no advanced data exists, insert empty object
                advanced_text.insert(tk.END, "{}")
        except Exception as e:
            print(f"Error formatting advanced data: {str(e)}")
            advanced_text.insert(tk.END, "{}\n# Previous data conversion error.\n# Edit with caution.")
        
        # Create buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_changes():
            try:
                # Get values from the basic tab
                new_formula = formula_var.get()
                new_crystal_system = crystal_system_var.get()
                new_hm_point_group = hm_point_group_var.get()
                new_schoenflies_point_group = schoenflies_point_group_var.get()
                new_space_group = space_group_var.get()
                
                # Update the mineral data
                if new_formula:
                    mineral_data['chemical_formula'] = new_formula
                
                mineral_data['crystal_system'] = new_crystal_system
                mineral_data['hermann_mauguin_point_group'] = new_hm_point_group
                mineral_data['schoenflies_point_group'] = new_schoenflies_point_group
                mineral_data['point_group'] = new_schoenflies_point_group  # Keep this for compatibility
                mineral_data['space_group'] = new_space_group
                
                # Try to parse and update advanced properties
                try:
                    # Get the text and clean it up - make sure we have just one JSON object
                    advanced_json = advanced_text.get(1.0, tk.END).strip()
                    
                    # Skip JSON parsing entirely if the text is empty or just whitespace
                    if advanced_json and advanced_json != "{}":
                        # Make sure the JSON is valid - it must start with { and end with }
                        if not (advanced_json.startswith('{') and advanced_json.endswith('}')):
                            raise ValueError("Advanced properties must be a valid JSON object starting with { and ending with }")
                        
                        try:
                            # Parse the JSON
                            advanced_props = json.loads(advanced_json)
                            
                            if not isinstance(advanced_props, dict):
                                raise ValueError("Advanced properties must be a JSON object (dictionary)")
                            
                            # Update with advanced properties
                            for key, value in advanced_props.items():
                                if key not in ['name', 'chemical_formula', 'crystal_system', 'point_group', 'space_group',
                                              'hermann_mauguin_point_group', 'schoenflies_point_group', 'modes']:
                                    mineral_data[key] = value
                        except json.JSONDecodeError as e:
                            # Provide more helpful error message with line/column information
                            line_info = f"line {e.lineno}, column {e.colno}"
                            raise ValueError(f"Error in advanced properties JSON at {line_info}: {e.msg}\nPlease check format - it should be valid JSON.")
                    # Else: If empty or just "{}", do nothing with advanced properties
                except Exception as e:
                    raise ValueError(f"Error processing advanced properties: {str(e)}")
                
                # Save the database
                self.db.save_database()
                
                # Update the display
                self.on_mineral_select(None)
                
                # Close the dialog
                dialog.destroy()
                
                # Show success message
                messagebox.showinfo("Success", f"Mineral '{mineral_name}' updated successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save changes: {str(e)}")
        
        # Add save and cancel buttons
        ttk.Button(button_frame, text="Save Changes", command=save_changes).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Make dialog modal
        dialog.wait_window()
    
    def update_mineral_info(self):
        """Legacy method - redirects to edit_mineral_entry."""
        self.edit_mineral_entry()
            
    def add_mode(self):
        """Add a new Raman mode to the selected mineral."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        
        # Ask user if they want to add single or multiple modes
        mode_option = messagebox.askquestion("Add Modes", 
                                          "Would you like to add multiple modes at once?",
                                          icon='question')
        
        if mode_option == 'yes':
            # Add multiple modes
            self.add_multiple_modes(mineral_name)
        else:
            # Add single mode (original functionality)
            # Get mode details
            position = simpledialog.askfloat("Peak Position", "Enter peak position (cm⁻¹):")
            if position is None:
                return
                
            symmetry = simpledialog.askstring("Symmetry", "Enter symmetry character:")
            if not symmetry:
                return
                
            intensity = simpledialog.askfloat("Intensity", "Enter relative intensity (default: 1.0):", initialvalue=1.0)
            if intensity is None:
                intensity = 1.0
                
            # Add mode
            success = self.db.add_mode(mineral_name, position, symmetry, intensity)
            if success:
                self.update_modes_table(mineral_name)
                self.update_plot(mineral_name)
                messagebox.showinfo("Success", f"Raman mode added to '{mineral_name}'.")
    
    def add_multiple_modes(self, mineral_name):
        """Add multiple Raman modes at once to a mineral."""
        # Create a new dialog window
        dialog = tk.Toplevel(self.window)
        dialog.title(f"Add Multiple Modes - {mineral_name}")
        dialog.geometry("600x400")
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Create a frame for the table
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        ttk.Label(frame, text="Enter multiple Raman modes (position, symmetry, intensity):").pack(pady=(0, 10))
        
        # Create a table for entering multiple modes
        # Headers
        headers_frame = ttk.Frame(frame)
        headers_frame.pack(fill=tk.X)
        
        ttk.Label(headers_frame, text="Position (cm⁻¹)", width=15).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(headers_frame, text="Symmetry", width=15).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(headers_frame, text="Intensity", width=15).grid(row=0, column=2, padx=5, pady=5)
        
        # Create a canvas with scrollbar for the entries
        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        
        entries_frame = ttk.Frame(canvas)
        entries_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=entries_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Entry rows (start with 10 rows, can add more)
        mode_entries = []
        for i in range(10):
            row_frame = ttk.Frame(entries_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            position_var = tk.StringVar()
            symmetry_var = tk.StringVar()
            intensity_var = tk.StringVar(value="1.0")  # Default intensity
            
            position_entry = ttk.Entry(row_frame, textvariable=position_var, width=15)
            position_entry.grid(row=0, column=0, padx=5)
            
            symmetry_entry = ttk.Entry(row_frame, textvariable=symmetry_var, width=15)
            symmetry_entry.grid(row=0, column=1, padx=5)
            
            intensity_entry = ttk.Entry(row_frame, textvariable=intensity_var, width=15)
            intensity_entry.grid(row=0, column=2, padx=5)
            
            mode_entries.append((position_var, symmetry_var, intensity_var))
        
        # Button to add more rows
        def add_row():
            row_frame = ttk.Frame(entries_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            position_var = tk.StringVar()
            symmetry_var = tk.StringVar()
            intensity_var = tk.StringVar(value="1.0")  # Default intensity
            
            position_entry = ttk.Entry(row_frame, textvariable=position_var, width=15)
            position_entry.grid(row=0, column=0, padx=5)
            
            symmetry_entry = ttk.Entry(row_frame, textvariable=symmetry_var, width=15)
            symmetry_entry.grid(row=0, column=1, padx=5)
            
            intensity_entry = ttk.Entry(row_frame, textvariable=intensity_var, width=15)
            intensity_entry.grid(row=0, column=2, padx=5)
            
            mode_entries.append((position_var, symmetry_var, intensity_var))
            
            # Scroll to the bottom
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.yview_moveto(1.0)
        
        # Button to import from clipboard
        def import_from_clipboard():
            try:
                clipboard = dialog.clipboard_get()
                lines = clipboard.strip().split('\n')
                
                # Clear existing entries first
                for pos_var, sym_var, int_var in mode_entries:
                    pos_var.set("")
                    sym_var.set("")
                    int_var.set("")
                
                # Add more rows if needed
                while len(mode_entries) < len(lines):
                    add_row()
                
                # Fill in values from clipboard
                for i, line in enumerate(lines):
                    if i >= len(mode_entries):
                        break
                        
                    # Try to parse the line as comma or tab-separated values
                    parts = line.split('\t') if '\t' in line else line.split(',')
                    parts = [p.strip() for p in parts]
                    
                    if len(parts) >= 1:
                        mode_entries[i][0].set(parts[0])  # Position
                    if len(parts) >= 2:
                        mode_entries[i][1].set(parts[1])  # Character
                    if len(parts) >= 3:
                        mode_entries[i][2].set(parts[2])  # Intensity
            except Exception as e:
                messagebox.showerror("Import Error", f"Error importing from clipboard: {str(e)}")
        
        # Buttons frame
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Add Row", command=add_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Import from Clipboard", command=import_from_clipboard).pack(side=tk.LEFT, padx=5)
        
        # Save function
        def save_modes():
            added_count = 0
            error_count = 0
            
            for pos_var, sym_var, int_var in mode_entries:
                # Skip empty rows
                if not pos_var.get() or not sym_var.get():
                    continue
                    
                try:
                    position = float(pos_var.get())
                    symmetry = sym_var.get()
                    
                    # Handle empty or invalid intensity
                    intensity = 1.0
                    if int_var.get():
                        try:
                            intensity = float(int_var.get())
                        except ValueError:
                            pass
                    
                    # Add mode to database
                    success = self.db.add_mode(mineral_name, position, symmetry, intensity)
                    if success:
                        added_count += 1
                    else:
                        error_count += 1
                except ValueError:
                    error_count += 1
            
            # Update the UI
            self.update_modes_table(mineral_name)
            self.update_plot(mineral_name)
            
            # Show result message
            if added_count > 0:
                result_msg = f"Successfully added {added_count} modes to '{mineral_name}'."
                if error_count > 0:
                    result_msg += f"\nFailed to add {error_count} modes."
                messagebox.showinfo("Success", result_msg)
            else:
                messagebox.showwarning("Warning", "No modes were added.")
            
            # Close dialog
            dialog.destroy()
        
        # Add save and cancel buttons
        ttk.Button(buttons_frame, text="Save", command=save_modes).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Make the dialog modal
        dialog.wait_window()
        
    def edit_mode(self):
        """Edit an existing Raman mode."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        selection = self.modes_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "No mode selected.")
            return
            
        # Get mineral and mode index
        mineral_index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(mineral_index)
        mode_index = self.modes_tree.index(selection[0])
        
        # Get current values
        modes = self.db.get_modes(mineral_name)
        if mode_index >= len(modes):
            return
            
        current_position, current_symmetry, current_intensity = modes[mode_index]
        
        # Get new values
        position = simpledialog.askfloat("Peak Position", "Enter peak position (cm⁻¹):", 
                                        initialvalue=current_position)
        if position is None:
            return
            
        symmetry = simpledialog.askstring("Symmetry", "Enter symmetry character:", 
                                         initialvalue=current_symmetry)
        if not symmetry:
            return
            
        intensity = simpledialog.askfloat("Intensity", "Enter relative intensity:", 
                                         initialvalue=current_intensity)
        if intensity is None:
            return
            
        # Update mode
        modes[mode_index] = (position, symmetry, intensity)
        
        # Update display
        self.update_modes_table(mineral_name)
        self.update_plot(mineral_name)
        
    def delete_mode(self):
        """Delete a Raman mode from the selected mineral."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        selection = self.modes_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "No mode selected.")
            return
            
        # Get mineral and mode index
        mineral_index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(mineral_index)
        mode_index = self.modes_tree.index(selection[0])
        
        # Confirm deletion
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to delete this mode?")
        if not confirm:
            return
            
        # Delete mode
        modes = self.db.get_modes(mineral_name)
        if mode_index < len(modes):
            del modes[mode_index]
            
            # Update display
            self.update_modes_table(mineral_name)
            self.update_plot(mineral_name)
            
    def import_from_peak_fitting(self):
        """Import data from peak_fitting.py output."""
        # This would typically involve getting data from peak_fitting.py results
        # For now, we'll use a placeholder asking for mineral name and data
        mineral_name = simpledialog.askstring("Import from Peak Fitting", 
                                            "Enter mineral name:")
        if not mineral_name:
            return
        
        messagebox.showinfo("Import from Peak Fitting", 
                          "This would normally import data directly from peak_fitting.py results.\n\n"
                          "For this demo, please use 'Add Mode' to add peaks manually.")
        
    def import_from_pkl(self):
        """Import data from another pickle file."""
        try:
            # Use our launcher script to start the PKL editor
            import subprocess
            import os
            import sys
            
            # Get the path to the current Python interpreter
            python_executable = sys.executable
            
            # Get the path to the launcher script
            launcher_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch_pkl_editor.py")
            
            if os.path.exists(launcher_path):
                # Launch the script and wait for it to complete
                self.window.iconify()  # Minimize the main window while editing
                
                # Show a message to the user
                messagebox.showinfo("PKL Editor Launching", 
                                  "The PKL File Editor will now launch.\n\n"
                                  "Please use it to select, edit, and import your PKL file.\n\n"
                                  "Click OK to continue.")
                
                # Launch the script
                result = subprocess.call([python_executable, launcher_path])
                
                # Restore the main window
                self.window.deiconify()
                
                # Refresh the mineral list after import
                self.update_mineral_list()
                
                if result == 0:
                    messagebox.showinfo("Import Complete", 
                                      "The PKL File Editor has closed.\n\n"
                                      "Any imported minerals have been added to the database.")
            else:
                raise FileNotFoundError(f"Could not find launcher script at {launcher_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch PKL File Editor: {str(e)}")
            
            # Fall back to the original import method if there's an error
            file_path = filedialog.askopenfilename(
                title="Select Pickle File",
                filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
                
            count = self.db.import_from_pkl(file_path)
            
            if count > 0:
                self.update_mineral_list()
                messagebox.showinfo("Import Successful", f"Imported {count} minerals from file.")
            else:
                messagebox.showwarning("Import Failed", "No minerals were imported from the file.")

    def save_database(self):
        """Save the database to file."""
        success = self.db.save_database()
        
        if success:
            messagebox.showinfo("Save Successful", "Database saved successfully.")
        else:
            messagebox.showerror("Save Failed", "Failed to save database.")
            
    def run(self):
        """Run the GUI application."""
        # Only call mainloop if we're running in standalone mode
        if hasattr(self, 'is_standalone') and self.is_standalone:
            # Update the mineral list before showing the window
            self.update_mineral_list()
            # Start the mainloop
            self.window.mainloop()
        else:
            # For embedded mode, just update the mineral list and return
            # The parent application's mainloop will handle the window
            self.update_mineral_list()

    def update_current_plot(self):
        """Update the current plot with new settings."""
        if not self.mineral_listbox.curselection():
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        self.update_plot(mineral_name)
        
    def on_key_press(self, event):
        """Handle key press events on the plot."""
        if event.key == 'g':
            # Toggle grid
            self.show_grid_var.set(not self.show_grid_var.get())
            self.update_current_plot()

    def select_random_mineral(self):
        """Select a random mineral from the database and display it."""
        minerals = self.db.get_minerals()
        if not minerals:
            return
            
        # Choose a random mineral
        random_mineral = random.choice(list(minerals))
        
        # Find the index in the listbox
        for i in range(self.mineral_listbox.size()):
            if self.mineral_listbox.get(i) == random_mineral:
                # Select it in the listbox
                self.mineral_listbox.selection_clear(0, tk.END)
                self.mineral_listbox.selection_set(i)
                self.mineral_listbox.see(i)
                
                # Trigger the selection event manually
                self.on_mineral_select(None)
                break

    def on_plot_navigate(self, event):
        """Update the range fields when the plot view changes through navigation."""
        # Only update if we have xlim
        if hasattr(self.ax, 'get_xlim'):
            x_min, x_max = self.ax.get_xlim()
            # Update the entry fields without triggering a redraw
            self.x_min_var.set(f"{x_min:.0f}")
            self.x_max_var.set(f"{x_max:.0f}")

    def add_example_modes(self):
        """Add example Raman modes to the selected mineral for demonstration purposes."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        
        # Special cases for specific minerals
        if "DIAMOND" in mineral_name.upper():
            # Diamond has a very simple Raman spectrum with one strong peak
            example_modes = [
                (1332.5, "F2g", 1.0),  # Main diamond peak
            ]
            if "(4H)" in mineral_name:
                # For 4H polytype, add some additional features
                example_modes.extend([
                    (776.0, "E2", 0.3),  # Folded optical modes for 4H polytype
                    (964.0, "A1", 0.2),
                ])
        elif "QUARTZ" in mineral_name.upper():
            # Quartz has several characteristic peaks
            example_modes = [
                (128.0, "A1", 0.5),
                (206.0, "E", 0.7),
                (265.0, "E", 0.3),
                (355.0, "A1", 0.4),
                (394.0, "A1", 0.3),
                (464.0, "A1", 1.0),  # Main quartz peak
                (696.0, "E", 0.2),
                (796.0, "E", 0.3),
                (1085.0, "A1", 0.5),
            ]
        else:
            # Generic example for other minerals
            example_modes = [
                (142.5, "Eg", 0.8),
                (196.3, "A1g", 1.0),
                (394.7, "B1g", 0.5),
                (515.2, "A1g", 0.9),
                (639.0, "Eg", 0.7)
            ]
        
        # Add the example modes
        modes_added = 0
        
        # Clear existing modes if any
        mineral_data = self.db.get_mineral_data(mineral_name)
        if mineral_data and 'modes' in mineral_data:
            # Ask user if they want to replace existing modes
            if len(mineral_data['modes']) > 0:
                response = messagebox.askyesno("Replace Modes", 
                          f"This mineral already has {len(mineral_data['modes'])} modes defined. Replace them?")
                if response:
                    mineral_data['modes'] = []
                else:
                    # Keep existing modes, add new ones that don't overlap
                    existing_positions = [pos for pos, _, _ in mineral_data['modes']]
                    example_modes = [mode for mode in example_modes 
                                    if not any(abs(mode[0] - pos) < 5.0 for pos in existing_positions)]
        
        # Add the modes
        for position, symmetry, intensity in example_modes:
            success = self.db.add_mode(mineral_name, position, symmetry, intensity)
            if success:
                modes_added += 1
                
        # Update UI
        if modes_added > 0:
            self.update_modes_table(mineral_name)
            self.update_plot(mineral_name)
            messagebox.showinfo("Success", f"Added {modes_added} example Raman modes to {mineral_name}.")
        else:
            messagebox.showwarning("Warning", "Failed to add example modes.")

    def view_dielectric_tensor(self):
        """Display dielectric tensor for the selected mineral."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        
        # Create a new window for the dielectric tensor
        window = tk.Toplevel(self.window)
        window.title(f"Dielectric Tensor - {mineral_name}")
        window.geometry("600x400")
        window.transient(self.window)
        
        # Create main frame
        main_frame = ttk.Frame(window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Ensure numpy is available
        import numpy as np
        
        # Check if dielectric tensor exists in the database
        mineral_data = self.db.get_mineral_data(mineral_name)
        dielectric_tensor = mineral_data.get('dielectric_tensor', None)
        
        # Convert dielectric tensor to a proper numpy array if it's not already
        if dielectric_tensor is not None:
            try:
                # If it's a dictionary, extract the array data
                if isinstance(dielectric_tensor, dict):
                    if 'data' in dielectric_tensor:
                        dielectric_tensor = dielectric_tensor['data']
                    elif 'matrix' in dielectric_tensor:
                        dielectric_tensor = dielectric_tensor['matrix']
                    else:
                        # Create a 3x3 array from dictionary values
                        temp_tensor = np.zeros((3, 3))
                        for i in range(3):
                            for j in range(3):
                                key = f"{i}{j}"
                                if key in dielectric_tensor:
                                    temp_tensor[i, j] = float(dielectric_tensor[key])
                        dielectric_tensor = temp_tensor
                        
                # If it's a list or tuple, convert to numpy array
                if isinstance(dielectric_tensor, (list, tuple)):
                    dielectric_tensor = np.array(dielectric_tensor, dtype=float)
                    
                # Ensure it's a proper 3x3 matrix
                if dielectric_tensor.shape != (3, 3):
                    # Try to reshape if possible
                    if dielectric_tensor.size == 9:
                        dielectric_tensor = dielectric_tensor.reshape(3, 3)
                    else:
                        # If not possible, create a default diagonal tensor
                        print(f"DEBUG: Dielectric tensor has invalid shape {dielectric_tensor.shape}, using default")
                        dielectric_tensor = np.diag([1.0, 1.0, 1.0])
                        
            except Exception as e:
                print(f"DEBUG: Error processing dielectric tensor: {e}")
                # Create a default tensor
                dielectric_tensor = np.diag([1.0, 1.0, 1.0])
    
        if dielectric_tensor is None:
            # No tensor available, show input fields to create one
            ttk.Label(main_frame, text="No dielectric tensor available for this mineral.").pack(pady=10)
            
            ttk.Label(main_frame, text="The dielectric tensor describes how the material responds to electric fields.\n"
                                     "It relates the electric displacement field to the electric field.").pack(pady=5)
            
            # Create a frame for the tensor
            tensor_frame = ttk.LabelFrame(main_frame, text="Dielectric Tensor Input", padding=10)
            tensor_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # Create a 3x3 grid of entry widgets
            tensor_vars = []
            for i in range(3):
                row_vars = []
                row_frame = ttk.Frame(tensor_frame)
                row_frame.pack(fill=tk.X, pady=5)
                
                for j in range(3):
                    var = tk.StringVar(value="0.0")
                    ttk.Entry(row_frame, textvariable=var, width=10).pack(side=tk.LEFT, padx=5)
                    row_vars.append(var)
                tensor_vars.append(row_vars)
            
            # Add a button to save the tensor
            def save_tensor():
                try:
                    tensor = []
                    for row_vars in tensor_vars:
                        tensor_row = [float(var.get()) for var in row_vars]
                        tensor.append(tensor_row)
                        
                    mineral_data['dielectric_tensor'] = np.array(tensor)
                    messagebox.showinfo("Success", "Dielectric tensor saved.")
                    window.destroy()
                except ValueError:
                    messagebox.showerror("Error", "Invalid tensor values. Please enter numeric values only.")
            
            ttk.Button(main_frame, text="Save Tensor", command=save_tensor).pack(pady=10)
            
            # Add example button
            def use_example():
                # Example dielectric tensor for a typical mineral (e.g., quartz)
                example = [
                    [4.52, 0.0, 0.0],
                    [0.0, 4.52, 0.0],
                    [0.0, 0.0, 4.64]
                ]
                
                for i in range(3):
                    for j in range(3):
                        tensor_vars[i][j].set(f"{example[i][j]:.2f}")
                        
            ttk.Button(main_frame, text="Use Example", command=use_example).pack(pady=5)
            
        else:
            # Display the existing tensor
            ttk.Label(main_frame, text="Dielectric Tensor:").pack(pady=10)
            
            # Create a frame for the tensor display
            tensor_frame = ttk.Frame(main_frame)
            tensor_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # Display the tensor values
            for i in range(3):
                row_frame = ttk.Frame(tensor_frame)
                row_frame.pack(fill=tk.X, pady=5)
                
                for j in range(3):
                    try:
                        value = dielectric_tensor[i][j]
                        if isinstance(value, (int, float)):
                            ttk.Label(row_frame, text=f"{value:.4f}", width=10, 
                                      relief="solid", padding=5).pack(side=tk.LEFT, padx=5)
                        else:
                            ttk.Label(row_frame, text=str(value), width=10, 
                                      relief="solid", padding=5).pack(side=tk.LEFT, padx=5)
                    except (IndexError, TypeError):
                        ttk.Label(row_frame, text="N/A", width=10, 
                                  relief="solid", padding=5).pack(side=tk.LEFT, padx=5)
            
            # Show Eigenvectors if they exist
            if isinstance(mineral_data['dielectric_tensor'], dict) and 'EigenV' in mineral_data['dielectric_tensor']:
                eigenvectors = mineral_data['dielectric_tensor']['EigenV']
                
                # Create section for eigenvectors
                eigenv_frame = ttk.LabelFrame(main_frame, text="Eigenvectors", padding=5)
                eigenv_frame.pack(fill=tk.X, pady=10)
                
                ttk.Label(eigenv_frame, text="Eigenvector values:").pack(anchor=tk.W, pady=5)
                
                # Display the eigenvector values
                for i, eigenv in enumerate(eigenvectors):
                    row_frame = ttk.Frame(eigenv_frame)
                    row_frame.pack(fill=tk.X, pady=2)
                    
                    ttk.Label(row_frame, text=f"Eigenvector {i+1}:", width=15).pack(side=tk.LEFT)
                    
                    for value in eigenv:
                        ttk.Label(row_frame, text=f"{value:.4f}", width=10, 
                                  relief="solid", padding=2).pack(side=tk.LEFT, padx=2)
            
            # Add a button to edit the tensor
            def edit_tensor():
                # Create entry widgets to allow editing
                for widget in tensor_frame.winfo_children():
                    widget.destroy()
                
                tensor_vars = []
                for i in range(3):
                    row_vars = []
                    row_frame = ttk.Frame(tensor_frame)
                    row_frame.pack(fill=tk.X, pady=5)
                    
                    for j in range(3):
                        try:
                            value = dielectric_tensor[i][j]
                            var = tk.StringVar(value=f"{value:.4f}")
                        except (IndexError, TypeError):
                            var = tk.StringVar(value="0.0")
                            
                        ttk.Entry(row_frame, textvariable=var, width=10).pack(side=tk.LEFT, padx=5)
                        row_vars.append(var)
                    tensor_vars.append(row_vars)
                
                # Add button to save changes
                def save_changes():
                    try:
                        tensor = []
                        for row_vars in tensor_vars:
                            tensor_row = [float(var.get()) for var in row_vars]
                            tensor.append(tensor_row)
                            
                        mineral_data['dielectric_tensor'] = np.array(tensor)
                        messagebox.showinfo("Success", "Dielectric tensor updated.")
                        window.destroy()
                    except ValueError:
                        messagebox.showerror("Error", "Invalid tensor values. Please enter numeric values only.")
                
                edit_btn.pack_forget()
                ttk.Button(main_frame, text="Save Changes", command=save_changes).pack(pady=10)
            
            edit_btn = ttk.Button(main_frame, text="Edit Tensor", command=edit_tensor)
            edit_btn.pack(pady=10)
    
    def view_born_charges(self):
        """Display Born effective charges for the selected mineral."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        
        # Create a new window for the Born charges
        window = tk.Toplevel(self.window)
        window.title(f"Born Effective Charges - {mineral_name}")
        window.geometry("700x500")
        window.transient(self.window)
        
        # Create main frame
        main_frame = ttk.Frame(window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Check if Born charges exist in the database
        mineral_data = self.db.get_mineral_data(mineral_name)
        born_charges = mineral_data.get('born_charges', None)
        
        # Ensure pandas is imported here for local reference
        import pandas as pd
        
        # Convert DataFrame to list of dictionaries if needed
        if isinstance(born_charges, pd.DataFrame):
            try:
                born_charges = born_charges.to_dict('records')
                print(f"DEBUG: Converted DataFrame to {len(born_charges)} records in view_born_charges")
            except Exception as e:
                print(f"DEBUG: Error converting DataFrame in view_born_charges: {e}")
                born_charges = []
        
        if born_charges is None:
            # No Born charges available, show message and input option
            ttk.Label(main_frame, text="No Born effective charges available for this mineral.").pack(pady=10)
            
            ttk.Label(main_frame, text="Born effective charges describe how atomic displacements create electric polarization.\n"
                                     "They are crucial for understanding IR activity and LO-TO splitting.").pack(pady=5)
            
            # Add a button to add example Born charges
            def add_example_charges():
                # Create example Born charges for a typical mineral
                # For simplicity, we'll use simple examples for 3 atoms
                example = [
                    {"atom": "Si", "charge": [[2.3, 0.0, 0.0], [0.0, 2.3, 0.0], [0.0, 0.0, 2.3]]},
                    {"atom": "O1", "charge": [[-1.1, 0.0, 0.0], [0.0, -1.1, 0.0], [0.0, 0.0, -1.2]]},
                    {"atom": "O2", "charge": [[-1.2, 0.1, 0.0], [0.1, -1.1, 0.0], [0.0, 0.0, -1.1]]}
                ]
                
                mineral_data['born_charges'] = example
                messagebox.showinfo("Success", "Example Born charges added.")
                window.destroy()
                
            ttk.Button(main_frame, text="Add Example Born Charges", command=add_example_charges).pack(pady=10)
            
        else:
            # Display the existing Born charges
            ttk.Label(main_frame, text="Born Effective Charges:").pack(pady=10)
            
            # Create a frame with a scrollbar
            canvas_frame = ttk.Frame(main_frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            canvas = tk.Canvas(canvas_frame)
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            
            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
                                # Display each atom's Born charges
            for i, atom_data in enumerate(born_charges):
                # Handle both dictionary and list formats for atom_data
                if isinstance(atom_data, dict):
                    atom_name = atom_data.get('atom', f'Atom {i+1}')
                    charge_tensor = atom_data.get('charge', [])
                    # Get eigenvectors if they exist
                    eigenvectors = atom_data.get('EigenV', None)
                else:
                    # If atom_data is a list (raw CSV data), use row as atom number
                    atom_name = f'Atom {i+1}'
                    charge_tensor = atom_data if isinstance(atom_data, list) else []
                    eigenvectors = None
                
                atom_frame = ttk.LabelFrame(scrollable_frame, text=f"Atom: {atom_name}", padding=5)
                atom_frame.pack(fill=tk.X, expand=True, pady=5, padx=5)
                
                # Display charge tensor
                tensor_frame = ttk.LabelFrame(atom_frame, text="Charge Tensor", padding=5)
                tensor_frame.pack(fill=tk.X, expand=True, pady=2)
                
                for j, row in enumerate(charge_tensor):
                    row_frame = ttk.Frame(tensor_frame)
                    row_frame.pack(fill=tk.X, pady=2)
                    
                    if isinstance(row, list):
                        for k, value in enumerate(row):
                            if isinstance(value, (int, float)):
                                ttk.Label(row_frame, text=f"{value:.4f}", width=10,
                                         relief="solid", padding=3).pack(side=tk.LEFT, padx=3)
                            else:
                                ttk.Label(row_frame, text=str(value), width=10,
                                         relief="solid", padding=3).pack(side=tk.LEFT, padx=3)
                    else:
                        # Handle non-list row
                        ttk.Label(row_frame, text=str(row), width=30,
                                 relief="solid", padding=3).pack(side=tk.LEFT, padx=3)
                
                # Display eigenvectors if they exist
                if eigenvectors is not None:
                    eigenv_frame = ttk.LabelFrame(atom_frame, text="Eigenvectors", padding=5)
                    eigenv_frame.pack(fill=tk.X, expand=True, pady=2)
                    
                    # Display as a row
                    row_frame = ttk.Frame(eigenv_frame)
                    row_frame.pack(fill=tk.X, pady=2)
                    
                    for i, value in enumerate(eigenvectors):
                        ttk.Label(row_frame, text=f"Eig{i+1}: {value:.4f}", width=12,
                                 relief="solid", padding=3).pack(side=tk.LEFT, padx=3)
            
            # Add button to edit Born charges
            def edit_born_charges():
                # Create a new window for editing
                edit_window = tk.Toplevel(window)
                edit_window.title(f"Edit Born Charges - {mineral_name}")
                edit_window.geometry("700x500")
                edit_window.transient(window)
                
                # To be implemented in a real application
                ttk.Label(edit_window, text="Born charge editing would be implemented here.").pack(pady=20)
                ttk.Button(edit_window, text="Close", command=edit_window.destroy).pack(pady=10)
                
            ttk.Button(main_frame, text="Edit Born Charges", command=edit_born_charges).pack(pady=10)
    
    def view_phonon_modes(self):
        """Display full phonon mode list for the selected mineral."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        
        # Create a new window for the phonon modes
        window = tk.Toplevel(self.window)
        window.title(f"Phonon Modes - {mineral_name}")
        window.geometry("800x600")
        window.transient(self.window)
        
        # Create main frame
        main_frame = ttk.Frame(window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Check if phonon_modes exist in the database
        mineral_data = self.db.get_mineral_data(mineral_name)
        phonon_modes = mineral_data.get('phonon_modes', None)
        
        # Convert DataFrame to list of dictionaries if needed
        if isinstance(phonon_modes, pd.DataFrame):
            try:
                phonon_modes = phonon_modes.to_dict('records')
                print(f"DEBUG: Converted DataFrame to {len(phonon_modes)} records in view_phonon_modes")
            except Exception as e:
                print(f"DEBUG: Error converting DataFrame in view_phonon_modes: {e}")
                phonon_modes = []
        
        raman_modes = mineral_data.get('modes', [])
        
        # Create a notebook for multiple tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tab for current Raman modes
        raman_tab = ttk.Frame(notebook)
        notebook.add(raman_tab, text="Raman Modes")
        
        # Create a treeview for Raman modes
        columns = ("position", "symmetry", "intensity", "activity")
        raman_tree = ttk.Treeview(raman_tab, columns=columns, show='headings')
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(raman_tab, orient="vertical", command=raman_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        raman_tree.configure(yscrollcommand=tree_scroll.set)
        raman_tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure columns
        raman_tree.heading("position", text="Position (cm⁻¹)")
        raman_tree.heading("symmetry", text="Character")
        raman_tree.heading("intensity", text="Intensity")
        raman_tree.heading("activity", text="Activity")
        
        raman_tree.column("position", width=150)
        raman_tree.column("symmetry", width=150)
        raman_tree.column("intensity", width=150)
        raman_tree.column("activity", width=150)
        
        # Populate with current Raman modes
        for position, symmetry, intensity in raman_modes:
            raman_tree.insert('', tk.END, values=(f"{position:.2f}", symmetry, f"{intensity:.3f}", "Raman"))
        
        # Tab for full phonon modes (if available)
        phonon_tab = ttk.Frame(notebook)
        notebook.add(phonon_tab, text="All Phonon Modes")
        
        if phonon_modes is None:
            # No phonon modes available
            ttk.Label(phonon_tab, text="No phonon modes available for this mineral.").pack(pady=20)
            
            # Add button to add example phonon modes
            def add_example_phonon_modes():
                # Create example phonon modes
                example = []
                # Generate some example modes across the spectrum
                symmetries = ["A1g", "B1g", "B2g", "Eg", "A1u", "B1u", "B2u", "Eu"]
                activities = ["Raman", "IR", "Raman+IR", "Silent"]
                
                import random
                for i in range(20):  # Generate 20 example modes
                    freq = random.uniform(50, 1200)  # Random frequency between 50-1200 cm-1
                    sym = random.choice(symmetries)
                    act = random.choice(activities)
                    intensity = random.uniform(0.1, 1.0) if "Raman" in act else 0.0
                    ir_intensity = random.uniform(0.1, 1.0) if "IR" in act else 0.0
                    
                    example.append({
                        "frequency": freq,
                        "symmetry": sym,
                        "activity": act,
                        "raman_intensity": intensity,
                        "ir_intensity": ir_intensity,
                        "eigenvector": [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(3)]  # Random eigenvector
                    })
                
                # Sort by frequency
                example.sort(key=lambda x: x["frequency"])
                mineral_data['phonon_modes'] = example
                
                messagebox.showinfo("Success", "Example phonon modes added.")
                window.destroy()
                self.view_phonon_modes()  # Reopen the window with the new data
                
            ttk.Button(phonon_tab, text="Add Example Phonon Modes", command=add_example_phonon_modes).pack(pady=10)
            
        else:
            # Create a treeview for all phonon modes
            columns = ("frequency", "symmetry", "activity", "raman_int", "ir_int")
            phonon_tree = ttk.Treeview(phonon_tab, columns=columns, show='headings')
            
            # Add scrollbar
            tree_scroll = ttk.Scrollbar(phonon_tab, orient="vertical", command=phonon_tree.yview)
            tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            phonon_tree.configure(yscrollcommand=tree_scroll.set)
            phonon_tree.pack(fill=tk.BOTH, expand=True)
            
            # Configure columns
            phonon_tree.heading("frequency", text="Frequency (cm⁻¹)")
            phonon_tree.heading("symmetry", text="Character")
            phonon_tree.heading("activity", text="Activity")
            phonon_tree.heading("raman_int", text="Raman Int.")
            phonon_tree.heading("ir_int", text="IR Int.")
            
            phonon_tree.column("frequency", width=150)
            phonon_tree.column("symmetry", width=100)
            phonon_tree.column("activity", width=150)
            phonon_tree.column("raman_int", width=100)
            phonon_tree.column("ir_int", width=100)
            
            # Populate with phonon modes
            for mode in phonon_modes:
                freq = mode.get("frequency", 0)
                sym = mode.get("symmetry", "")
                act = mode.get("activity", "")
                raman_int = mode.get("raman_intensity", 0)
                ir_int = mode.get("ir_intensity", 0)
                
                phonon_tree.insert('', tk.END, values=(
                    f"{freq:.2f}", sym, act, 
                    f"{raman_int:.3f}" if raman_int > 0 else "-", 
                    f"{ir_int:.3f}" if ir_int > 0 else "-"
                ))
            
            # Add a tab for eigenvector visualization (placeholder for future implementation)
            eigenvector_tab = ttk.Frame(notebook)
            notebook.add(eigenvector_tab, text="Eigenvectors")
            
            ttk.Label(eigenvector_tab, text="Eigenvector visualization would be implemented here.\n"
                                         "This would show the atomic displacements for selected modes.", 
                     font=('Arial', 10)).pack(pady=20)
        
        # Button to export data to CSV
        def export_to_csv():
            file_path = filedialog.asksaveasfilename(
                title="Export Phonon Modes",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                defaultextension=".csv"
            )
            
            if not file_path:
                return
                
            try:
                import csv
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    if phonon_modes is not None:
                        writer.writerow(["Frequency (cm⁻¹)", "Symmetry", "Activity", "Raman Intensity", "IR Intensity"])
                        
                        # Write data
                        for mode in phonon_modes:
                            writer.writerow([
                                f"{mode.get('frequency', 0):.2f}",
                                mode.get("symmetry", ""),
                                mode.get("activity", ""),
                                f"{mode.get('raman_intensity', 0):.3f}",
                                f"{mode.get('ir_intensity', 0):.3f}"
                            ])
                    else:
                        # Just export the Raman modes
                        writer.writerow(["Position (cm⁻¹)", "Symmetry", "Intensity", "Activity"])
                        
                        for position, symmetry, intensity in raman_modes:
                            writer.writerow([f"{position:.2f}", symmetry, f"{intensity:.3f}", "Raman"])
                    
                messagebox.showinfo("Export Successful", f"Data exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")
        
        ttk.Button(main_frame, text="Export to CSV", command=export_to_csv).pack(pady=10)

    def debug_mineral_data(self):
        """Debug the mineral data in the database."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        
        # Get mineral data
        mineral_data = self.db.get_mineral_data(mineral_name)
        if not mineral_data:
            messagebox.showwarning("Warning", "No mineral data found.")
            return
            
        # Create a new window for debugging
        debug_window = tk.Toplevel(self.window)
        debug_window.title(f"Debug - {mineral_name}")
        debug_window.geometry("800x600")
        debug_window.transient(self.window)
        
        # Create a frame for action buttons
        action_frame = ttk.LabelFrame(debug_window, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=5)
        
        # Button to force conversion of phonon modes to Raman modes
        def force_convert_modes():
            if 'phonon_modes' in mineral_data:
                try:
                    # Create or clear the modes array
                    mineral_data['modes'] = []
                    
                    # Check if it's a DataFrame and convert to records if needed
                    phonon_modes = mineral_data['phonon_modes']
                    if isinstance(phonon_modes, pd.DataFrame):
                        try:
                            phonon_modes = phonon_modes.to_dict('records')
                            print(f"DEBUG: Converted DataFrame to {len(phonon_modes)} records")
                        except Exception as e:
                            print(f"DEBUG: Error converting DataFrame: {e}")
                            messagebox.showerror("Error", f"Error converting DataFrame: {e}")
                            return
                    
                    # Convert all phonon modes to Raman modes regardless of activity
                    for mode in phonon_modes:
                        try:
                            # Try to get frequency from different possible keys
                            freq = None
                            for key in ['frequency', 'freq', 'position', 'wavenumber', 'TO_Frequency', 'Frequency']:
                                if key in mode:
                                    try:
                                        freq = float(mode[key])
                                        break
                                    except (ValueError, TypeError):
                                        # Try to extract number from string if it's not directly convertible
                                        try:
                                            import re
                                            match = re.search(r'(\d+\.?\d*)', str(mode[key]))
                                            if match:
                                                freq = float(match.group(1))
                                                print(f"DEBUG: Extracted frequency {freq} from {key}")
                                                break
                                        except:
                                            continue
                            
                            # Special case for Mode column with frequency in parentheses
                            if freq is None and 'Mode' in mode and isinstance(mode['Mode'], str):
                                mode_text = str(mode['Mode'])
                                # Try to extract frequency from format like "A1g (625)"
                                import re
                                freq_match = re.search(r'\((\d+\.?\d*)\)', mode_text)
                                if freq_match:
                                    try:
                                        freq = float(freq_match.group(1))
                                        print(f"DEBUG: Extracted frequency {freq} from Mode text: {mode_text}")
                                    except (ValueError, TypeError):
                                        pass
                            
                            if freq is None:
                                print(f"DEBUG: Skipping mode, no frequency found. Keys: {list(mode.keys())}")
                                continue
                                
                            # Try to get symmetry - prioritize Activity column for character
                            sym = None
                            
                            # First, check explicitly for Activity column
                            if 'Activity' in mode:
                                try:
                                    sym = str(mode['Activity']).strip()
                                    print(f"DEBUG: Using Activity '{sym}' as character")
                                except Exception:
                                    pass
                            
                            # If no Activity found, try other keys
                            if not sym:
                                for key in ['symmetry', 'sym', 'irrep', 'character', 'Mode']:
                                    if key in mode:
                                        try:
                                            if key == 'Mode' and isinstance(mode[key], str):
                                                # Extract symmetry from Mode string
                                                mode_text = str(mode[key])
                                                import re
                                                sym_match = re.search(r'(A1g|A2g|B1g|B2g|Eg|A1u|A2u|B1u|B2u|Eu|A1|A2|B1|B2|E|T1g|T2g|T1u|T2u)', mode_text)
                                                if sym_match:
                                                    sym = sym_match.group(1)
                                                else:
                                                    sym = mode_text.split('(')[0].strip()
                                                print(f"DEBUG: Extracted symmetry '{sym}' from Mode: {mode_text}")
                                            else:
                                                sym = str(mode[key])
                                                print(f"DEBUG: Found symmetry '{sym}' from key '{key}'")
                                            break
                                        except Exception:
                                            continue
                            
                            # Default if nothing found
                            if not sym:
                                sym = "A1g"  # Default symmetry if none found
                                print(f"DEBUG: Using default symmetry A1g")
                            
                            # Try to get intensity - prioritize I_Total
                            intensity = None
                            
                            # First try I_Total
                            if 'I_Total' in mode:
                                try:
                                    val = mode['I_Total']
                                    if val is not None:
                                        intensity = float(val)
                                        print(f"DEBUG: Using I_Total={intensity} for intensity")
                                except (ValueError, TypeError):
                                    print(f"DEBUG: Could not convert I_Total value to float")
                            
                            # Fall back to other fields if needed
                            if intensity is None:
                                for key in ['intensity', 'raman_intensity', 'int', 'raman_int']:
                                    if key in mode:
                                        try:
                                            val = mode[key]
                                            if val is not None:
                                                intensity = float(val)
                                                print(f"DEBUG: Falling back to {key}={intensity} for intensity")
                                                break
                                        except (ValueError, TypeError):
                                            continue
                            
                            if intensity is None or intensity <= 0:
                                intensity = 0.5  # Default intensity if none
                            
                            mineral_data['modes'].append((freq, sym, intensity))
                            print(f"DEBUG: Added mode at {freq} cm-1 with {sym} symmetry and {intensity} intensity")
                        except Exception as e:
                            print(f"DEBUG: Error processing mode: {e}")
                    
                    # Update the view
                    self.update_modes_table(mineral_name)
                    self.update_plot(mineral_name)
                    
                    # Refresh the debug data display
                    text_widget.delete(1.0, tk.END)
                    text_widget.insert(tk.END, json.dumps(mineral_data, indent=4))
                    
                    messagebox.showinfo("Success", f"Converted {len(mineral_data['modes'])} phonon modes to Raman modes.")
                except Exception as e:
                    messagebox.showerror("Error", f"Error converting phonon modes: {str(e)}")
            else:
                messagebox.showwarning("Warning", "No phonon modes found to convert.")
        
        # Button to attempt database repair if structure is incorrect
        def repair_database_structure():
            # Common repairs to try
            changes_made = []
            
            # 1. Ensure 'modes' exists
            if 'modes' not in mineral_data:
                mineral_data['modes'] = []
                changes_made.append("Created 'modes' array")
            
            # 2. Check if modes might be stored under a different key
            for key in mineral_data:
                if key != 'modes' and isinstance(mineral_data[key], list) and len(mineral_data[key]) > 0:
                    # Check if this could be a list of modes
                    first_item = mineral_data[key][0]
                    if (isinstance(first_item, tuple) or isinstance(first_item, list)) and len(first_item) == 3:
                        # This looks like a mode tuple (position, symmetry, intensity)
                        mineral_data['modes'] = mineral_data[key]
                        changes_made.append(f"Copied modes from '{key}' to 'modes'")
                        break
            
            # 3. Convert any object-style modes to tuple format
            if 'modes' in mineral_data and isinstance(mineral_data['modes'], list):
                for i, mode in enumerate(mineral_data['modes']):
                    if isinstance(mode, dict):
                        # Convert dict mode to tuple
                        position = mode.get('position', mode.get('frequency', 0))
                        symmetry = mode.get('symmetry', 'A')
                        intensity = mode.get('intensity', mode.get('raman_intensity', 1.0))
                        mineral_data['modes'][i] = (position, symmetry, intensity)
                        changes_made.append(f"Converted mode {i} from dict to tuple format")
            
            # Update the view if changes were made
            if changes_made:
                self.update_modes_table(mineral_name)
                self.update_plot(mineral_name)
                
                # Refresh the debug data display
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, json.dumps(mineral_data, indent=4))
                
                messagebox.showinfo("Repair Results", "The following repairs were made:\n" + "\n".join(changes_made))
            else:
                messagebox.showinfo("Repair Results", "No issues found that needed repair.")
        
        # Add action buttons
        ttk.Button(action_frame, text="Convert Phonon to Raman Modes", 
                  command=force_convert_modes).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Repair Database Structure", 
                  command=repair_database_structure).pack(side=tk.LEFT, padx=5)
        
        # Status frame shows basic info
        status_frame = ttk.LabelFrame(debug_window, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Display basic counts
        modes_count = len(mineral_data.get('modes', []))
        
        # Handle phonon_modes correctly whether it's a DataFrame or list
        phonon_modes = mineral_data.get('phonon_modes', [])
        if isinstance(phonon_modes, pd.DataFrame):
            phonon_modes_count = len(phonon_modes)
        else:
            phonon_modes_count = len(phonon_modes)
        has_dielectric = 'dielectric_tensor' in mineral_data
        has_born_charges = 'born_charges' in mineral_data
        
        status_text = f"Raman Modes: {modes_count}\n"
        status_text += f"Phonon Modes: {phonon_modes_count}\n"
        status_text += f"Has Dielectric Tensor: {'Yes' if has_dielectric else 'No'}\n"
        status_text += f"Has Born Charges: {'Yes' if has_born_charges else 'No'}"
        
        ttk.Label(status_frame, text=status_text).pack(anchor=tk.W)
        
        # Create a text widget to display the mineral data
        text_widget = tk.Text(debug_window, wrap=tk.WORD, width=80, height=20)
        text_widget.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add scrollbar to text widget
        text_scroll = ttk.Scrollbar(text_widget, orient="vertical", command=text_widget.yview)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=text_scroll.set)
        
        # Convert the mineral data to a readable format
        data_str = json.dumps(mineral_data, indent=4)
        
        # Insert the data into the text widget
        text_widget.insert(tk.END, data_str)
        
        # Add buttons for file operations
        file_frame = ttk.Frame(debug_window)
        file_frame.pack(fill=tk.X, pady=5)
        
        def save_debug_data():
            file_path = filedialog.asksaveasfilename(
                title="Save Debug Data",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                defaultextension=".json"
            )
            
            if not file_path:
                return
                
            try:
                with open(file_path, 'w') as f:
                    f.write(data_str)
                messagebox.showinfo("Success", f"Debug data saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving debug data: {str(e)}")
        
        ttk.Button(file_frame, text="Save Debug Data", command=save_debug_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Close", command=debug_window.destroy).pack(side=tk.RIGHT, padx=5)

    def delete_mineral(self):
        """Delete the selected mineral from the database."""
        if not self.mineral_listbox.curselection():
            messagebox.showwarning("Warning", "No mineral selected.")
            return
            
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        
        # Confirm deletion
        confirm = messagebox.askyesno("Confirm Deletion", 
                                    f"Are you sure you want to delete '{mineral_name}'?\n\nThis action cannot be undone.")
        if not confirm:
            return
            
        # Delete the mineral
        success = self.db.delete_mineral(mineral_name)
        
        if success:
            # Update the mineral list
            self.update_mineral_list()
            
            # Clear the info panel
            self.mineral_name_var.set("")
            self.crystal_system_var.set("")
            self.point_group_var.set("")
            self.space_group_var.set("")
            
            # Clear the modes table
            self.modes_tree.delete(*self.modes_tree.get_children())
            
            # Clear the plot
            self.ax.clear()
            self.ax.set_title("Raman Spectrum")
            self.ax.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax.set_ylabel("Normalized Intensity")
            self.canvas.draw()
            
            messagebox.showinfo("Success", f"Mineral '{mineral_name}' deleted successfully.")
        else:
            messagebox.showerror("Error", f"Failed to delete mineral '{mineral_name}'.")
            
        # Save the database automatically after deletion
        self.db.save_database()

    def force_update_display(self):
        """Manually force the GUI to update the mineral information display."""
        print("DEBUG: Manual refresh triggered")
        
        # Check if there's a currently selected mineral
        if not self.mineral_listbox.curselection():
            print("DEBUG: No mineral selected during manual refresh")
            # Set a clear message in the name field
            self.mineral_name_var.set("No mineral selected")
            self.crystal_system_var.set("")
            self.point_group_var.set("")
            self.space_group_var.set("")
            self.raman_modes_count_var.set("")
            self.phonon_data_var.set("")
            self.key_features_var.set("")
            return
        
        # Get the currently selected mineral
        index = self.mineral_listbox.curselection()[0]
        mineral_name = self.mineral_listbox.get(index)
        print(f"DEBUG: Manual refresh for mineral: {mineral_name}")
        
        # Get the current mineral data
        mineral_data = self.db.get_mineral_data(mineral_name)
        if not mineral_data:
            print(f"DEBUG: No data found for {mineral_name}")
            return
            
        # Extract data and update display
        crystal_system = mineral_data.get('crystal_system', '')
        point_group = mineral_data.get('point_group', '')
        space_group = mineral_data.get('space_group', '')
        
        # Manually set each field and force update
        if 'chemical_formula' in mineral_data and mineral_data['chemical_formula']:
            formula = mineral_data['chemical_formula']
            display_name = f"{mineral_name} ({formula})"
        else:
            display_name = mineral_name
            
        # Set each StringVar and force GUI update after each one
        self.mineral_name_var.set(display_name)
        self.window.update_idletasks()
        
        self.crystal_system_var.set(crystal_system)
        self.window.update_idletasks()
        
        self.point_group_var.set(point_group)
        self.window.update_idletasks()
        
        self.space_group_var.set(space_group)
        self.window.update_idletasks()
        
        # Get mode count
        modes = self.db.get_modes(mineral_name)
        self.raman_modes_count_var.set(f"{len(modes)}" if modes else "0")
        self.window.update_idletasks()
        
        # Check available data
        has_phonon_data = False
        for key in ['phonon_modes', 'phonon_mode', 'phonons', 'modes_full', 'all_modes']:
            if key in mineral_data and mineral_data[key]:
                has_phonon_data = True
                break
                
        has_dielectric = 'dielectric_tensor' in mineral_data and mineral_data['dielectric_tensor']
        has_born_charges = 'born_charges' in mineral_data and mineral_data['born_charges']
        
        data_status = []
        if has_phonon_data:
            data_status.append("Phonon")
        if has_dielectric:
            data_status.append("Dielectric")
        if has_born_charges:
            data_status.append("Born")
            
        self.phonon_data_var.set(", ".join(data_status) if data_status else "No")
        self.window.update_idletasks()
        
        # Set key features
        key_features = ""
        if modes:
            # Sort modes by intensity to find the strongest peaks
            sorted_modes = sorted(modes, key=lambda x: x[2], reverse=True)
            strong_modes = sorted_modes[:3] if len(sorted_modes) >= 3 else sorted_modes
            
            # Format the key features text
            positions = [f"{pos:.0f}" for pos, _, _ in strong_modes]
            key_features = f"Strong peaks at {', '.join(positions)} cm⁻¹"
            
            # Add symmetry information if available
            symmetries = set([sym for _, sym, _ in modes])
            if len(symmetries) <= 5:  # Only show if not too many
                key_features += f" ({', '.join(symmetries)})"
        else:
            key_features = "No characteristic peaks identified"
            
        self.key_features_var.set(key_features)
        self.window.update_idletasks()
        
        print("DEBUG: Manual refresh completed")
        # Final force refresh of the entire window
        self.window.update()

    def update_plot(self, mineral_name):
        """
        Update the visualization plot.
        
        The plot shows:
        - Blue solid line: Total combined Raman spectrum (sum of all modes)
        - Colored dashed lines: Individual contributions from each Raman mode
        - Colored markers: Peak positions with wavenumber and symmetry labels
        """
        try:
            self.ax.clear()
            
            # Get the modes data
            modes = self.db.get_modes(mineral_name)
            if not modes or len(modes) == 0:
                self.ax.text(0.5, 0.5, "No Raman modes available", 
                          ha='center', va='center', fontsize=12,
                          transform=self.ax.transAxes, color='red')
                self.canvas.draw()
                return
                
            # Get plot parameters from controls
            try:
                x_min = float(self.x_min_var.get())
                x_max = float(self.x_max_var.get())
                peak_width_base = float(self.peak_width_var.get())
                line_width = self.line_width_var.get() if hasattr(self, 'line_width_var') else 1.0
            except (ValueError, AttributeError):
                x_min, x_max = 50, 1500
                peak_width_base = 5
                line_width = 1.0
                
            # Generate x-axis (wavenumbers) with enough points for smooth curves
            x = np.linspace(x_min, x_max, 2000)
            
            # Extract positions, symmetries, and intensities
            positions = []
            symmetries = []
            intensities = []
            
            for mode in modes:
                try:
                    pos, sym, intensity = mode
                    positions.append(float(pos))
                    symmetries.append(str(sym))
                    intensities.append(float(intensity))
                except (ValueError, TypeError):
                    continue
            
            # If no valid modes, show message and return
            if not positions:
                self.ax.text(0.5, 0.5, "No valid Raman modes to display", 
                          ha='center', va='center', fontsize=12,
                          transform=self.ax.transAxes, color='red')
                self.canvas.draw()
                return
            
            # Normalize intensities to a 0-1 scale while preserving relative ratios
            max_intensity = max(intensities)
            if max_intensity > 0:
                normalized_intensities = [i/max_intensity for i in intensities]
            else:
                normalized_intensities = intensities
            
            # Create list of (position, symmetry, normalized_intensity) tuples
            normalized_modes = list(zip(positions, symmetries, normalized_intensities))
            
            # Generate spectrum
            total_y = np.zeros_like(x)
            mode_contributions = []
            peak_y_values = {}
            
            # Calculate each mode's contribution with higher magnification
            for position, symmetry, intensity in normalized_modes:
                # Skip very low intensity modes
                if intensity < 0.01:  # 1% threshold
                    continue
                    
                # Calculate Lorentzian profile with higher amplitude
                width = peak_width_base * (0.8 + 0.4 * intensity)
                gamma = width
                # Use a much higher amplitude factor to make peaks visible
                lorentzian = intensity * 50.0 * (gamma/2)**2 / ((x - position)**2 + (gamma/2)**2)
                
                # Add to the total spectrum
                total_y += lorentzian
                
                # Store for individual plotting
                mode_contributions.append((position, symmetry, intensity, lorentzian))
                
                # Store peak height for label placement
                peak_index = np.abs(x - position).argmin()
                peak_y_values[position] = lorentzian[peak_index]
            
            # Normalize the total spectrum to a maximum of 1.0
            if np.max(total_y) > 0:
                scale_factor = 1.0 / np.max(total_y)
                total_y *= scale_factor
                # Scale individual contributions by the same factor
                for i in range(len(mode_contributions)):
                    pos, sym, intensity, lorentzian = mode_contributions[i]
                    mode_contributions[i] = (pos, sym, intensity, lorentzian * scale_factor)
                    if pos in peak_y_values:
                        peak_y_values[pos] *= scale_factor
            
            # Plot the total spectrum
            total_line, = self.ax.plot(x, total_y, '-', lw=line_width, color='#1f77b4', alpha=0.9, 
                                label='Total Spectrum')
            
            # Plot individual mode contributions
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            mode_lines = []
            for i, (position, symmetry, intensity, lorentzian) in enumerate(mode_contributions):
                if x_min <= position <= x_max:
                    color = colors[i % len(colors)]
                    mode_line, = self.ax.plot(x, lorentzian, '--', lw=0.75, color=color, alpha=0.5, 
                                        label=f"{position:.0f} cm⁻¹ ({symmetry})")
                    mode_lines.append(mode_line)
                    
                    # Add marker at the peak position
                    marker_height = 0.05
                    self.ax.plot([position], [marker_height], 'v', ms=4, color=color, alpha=0.8)
                    
                    # Add text annotation
                    peak_height = peak_y_values.get(position, 0)
                    label_offset = peak_height * 1.05
                    self.ax.text(position, label_offset, f"{position:.0f}\n{symmetry}", 
                              ha='center', va='bottom', fontsize=8, color=color)
                    
                    # Add vertical line
                    self.ax.axvline(x=position, linestyle='--', linewidth=0.3, color=color, alpha=0.4)
            
            # Set labels and title - use DejaVu Sans which supports all mathematical symbols
            self.ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9, labelpad=5, fontfamily='DejaVu Sans')
            self.ax.set_ylabel("Normalized Intensity", fontsize=9, labelpad=5, fontfamily='DejaVu Sans')
            
            # Set axis limits
            self.ax.set_ylim(bottom=0, top=1.1)
            self.ax.set_xlim(x_min, x_max)
            
            # Add grid if enabled
            if hasattr(self, 'show_grid_var') and self.show_grid_var.get():
                self.ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
            else:
                self.ax.grid(False)
            
            # Set title with DejaVu Sans font
            self.ax.set_title(f"{mineral_name} - Raman Spectrum", fontsize=10, fontfamily='DejaVu Sans')
            
            # Add a legend for the plot elements
            # Limit to just a few representative items to avoid overcrowding
            legend_items = [total_line]
            legend_labels = ['Total Spectrum']
            
            # Only include up to 3 modes in the legend to avoid cluttering
            for i, line in enumerate(mode_lines[:3]):
                legend_items.append(line)
                pos, sym, _, _ = mode_contributions[i]
                legend_labels.append(f"{pos:.0f} cm⁻¹ ({sym})")
            
            # If more modes exist, add an ellipsis entry
            if len(mode_lines) > 3:
                dummy_line, = self.ax.plot([], [], '--', color='gray')
                legend_items.append(dummy_line)
                legend_labels.append('...more peaks')
            
            # Add the legend with smaller font outside the plot area - use DejaVu Sans
            self.ax.legend(legend_items, legend_labels, loc='upper center', 
                          bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, 
                          frameon=True, framealpha=0.8, prop={'family': 'DejaVu Sans'})
            
            # Update layout
            self.fig.tight_layout(pad=0.3, rect=[0.03, 0.03, 0.97, 0.90])  # Adjust rect to make room for legend
            self.canvas.draw()
            
        except Exception as e:
            print(f"Plot error: {e}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error updating plot: {str(e)}", 
                      ha='center', va='center', fontsize=10,
                      transform=self.ax.transAxes, color='red')
            self.canvas.draw()

    def update_eigenvectors(self):
        """Update eigenvectors from CSV files for all minerals."""
        # Create a progress window
        progress_window = tk.Toplevel(self.window)
        progress_window.title("Updating Eigenvectors")
        progress_window.geometry("400x300")
        progress_window.transient(self.window)
        
        # Progress frame
        progress_frame = ttk.Frame(progress_window, padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        status_var = tk.StringVar(value="Starting eigenvector update...")
        status_label = ttk.Label(progress_frame, textvariable=status_var)
        status_label.pack(pady=10)
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=10)
        
        # Results text
        results_text = tk.Text(progress_frame, height=10, width=40)
        results_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollbar for text
        scrollbar = ttk.Scrollbar(results_text, command=results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.config(yscrollcommand=scrollbar.set)
        
        # Ask for the CSV directory - default to wurm_data/csv
        csv_dir = filedialog.askdirectory(
            title="Select directory containing CSV files",
            initialdir="wurm_data/csv"
        )
        
        # If user cancels, use default directory
        if not csv_dir:
            csv_dir = "wurm_data/csv"
            
        # Update the window to show we're starting
        status_var.set(f"Starting eigenvector update from {csv_dir}...")
        progress_window.update()
        
        try:
            # Use the new method in MineralDatabase
            results_text.insert(tk.END, f"Processing CSV files from {csv_dir}\n")
            progress_window.update()
            
            # Start an update thread to avoid freezing the GUI
            import threading
            
            def update_thread():
                try:
                    processed, born_updated, diel_updated = self.db.update_eigenvectors(csv_dir)
                    
                    # Update results when done
                    progress_var.set(100)
                    status_var.set("Update completed!")
                    
                    # Show results
                    results_text.insert(tk.END, f"\nResults:\n")
                    results_text.insert(tk.END, f"Minerals processed: {processed}\n")
                    results_text.insert(tk.END, f"Born charges updated: {born_updated}\n")
                    results_text.insert(tk.END, f"Dielectric tensors updated: {diel_updated}\n")
                    
                    # Refresh the mineral list and current display
                    self.update_mineral_list()
                    if self.mineral_listbox.curselection():
                        self.on_mineral_select()
                        
                except Exception as e:
                    results_text.insert(tk.END, f"Error: {str(e)}\n")
                
                # Add close button when thread is complete
                ttk.Button(progress_frame, text="Close", 
                          command=progress_window.destroy).pack(pady=10)
            
            # Start the thread
            threading.Thread(target=update_thread).start()
            
        except Exception as e:
            results_text.insert(tk.END, f"Error: {str(e)}\n")
            # Add close button
            ttk.Button(progress_frame, text="Close", 
                      command=progress_window.destroy).pack(pady=10)
                      
    def batch_convert_phonon_to_raman(self):
        """Convert phonon modes to Raman modes for all minerals in the database."""
        # Create a progress window
        progress_window = tk.Toplevel(self.window)
        progress_window.title("Converting Phonon Modes")
        progress_window.geometry("400x300")
        progress_window.transient(self.window)
        
        # Progress frame
        progress_frame = ttk.Frame(progress_window, padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        status_var = tk.StringVar(value="Starting conversion...")
        status_label = ttk.Label(progress_frame, textvariable=status_var)
        status_label.pack(pady=10)
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=10)
        
        # Results text
        results_text = tk.Text(progress_frame, height=10, width=40)
        results_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollbar for text
        scrollbar = ttk.Scrollbar(results_text, command=results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.config(yscrollcommand=scrollbar.set)
        
        # Update the window immediately to show progress
        progress_window.update()
        
        # Get the list of minerals
        minerals = self.db.get_minerals()
        total_minerals = len(minerals)
        
        # Track statistics
        converted_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each mineral
        for i, mineral_name in enumerate(minerals):
            try:
                # Update progress
                progress_percent = (i / total_minerals) * 100
                progress_var.set(progress_percent)
                status_var.set(f"Converting {mineral_name}... ({i+1}/{total_minerals})")
                progress_window.update()
                
                # Get mineral data
                mineral_data = self.db.get_mineral_data(mineral_name)
                
                # Skip if already has Raman modes
                if 'modes' in mineral_data and mineral_data['modes'] and len(mineral_data['modes']) > 0:
                    results_text.insert(tk.END, f"Skipped {mineral_name}: Already has {len(mineral_data['modes'])} Raman modes\n")
                    skipped_count += 1
                    continue
                
                # Convert phonon modes to Raman modes
                success = self.db.convert_phonon_to_raman_modes(mineral_name)
                
                if success:
                    # Get the new count of modes
                    mode_count = len(mineral_data.get('modes', []))
                    results_text.insert(tk.END, f"Converted {mineral_name}: Added {mode_count} Raman modes\n")
                    converted_count += 1
                else:
                    # Try adding example modes instead
                    if self.add_example_modes_for_mineral(mineral_name):
                        mode_count = len(mineral_data.get('modes', []))
                        results_text.insert(tk.END, f"Added examples for {mineral_name}: {mode_count} modes\n")
                        converted_count += 1
                    else:
                        results_text.insert(tk.END, f"Failed to convert {mineral_name}\n")
                        error_count += 1
                
                # Scroll to the bottom of results
                results_text.see(tk.END)
                progress_window.update()
                
            except Exception as e:
                results_text.insert(tk.END, f"Error with {mineral_name}: {str(e)}\n")
                error_count += 1
                
                # Continue with next mineral despite errors
                continue
        
        # Save the database
        if converted_count > 0:
            success = self.db.save_database()
            if success:
                results_text.insert(tk.END, "\nDatabase saved successfully.\n")
            else:
                results_text.insert(tk.END, "\nFailed to save database.\n")
        
        # Show final summary
        status_var.set("Conversion complete")
        progress_var.set(100)
        
        summary = f"\nSummary:\n"
        summary += f"Total minerals: {total_minerals}\n"
        summary += f"Converted: {converted_count}\n"
        summary += f"Skipped (already had modes): {skipped_count}\n"
        summary += f"Errors: {error_count}\n"
        
        results_text.insert(tk.END, summary)
        results_text.see(tk.END)
        
        # Add close button
        ttk.Button(progress_frame, text="Close", 
                  command=progress_window.destroy).pack(pady=10)
                  
        # Refresh the mineral list and current display after batch conversion
        self.update_mineral_list()
        if self.mineral_listbox.curselection():
            self.on_mineral_select()

def import_from_peak_fitting(peak_fitting_results, mineral_name, crystal_system=None, point_group=None, space_group=None):
    """
    Import data from peak_fitting.py results into the database.
    
    Parameters:
    -----------
    peak_fitting_results : dict
        Results from peak_fitting.py
    mineral_name : str
        Name of the mineral
    crystal_system : str, optional
        Crystal system
    point_group : str, optional
        Point group
    space_group : str, optional
        Space group
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    db = MineralDatabase()
    
    # Add mineral if it doesn't exist
    if mineral_name not in db.get_minerals():
        db.add_mineral(mineral_name, crystal_system, point_group, space_group)
    
    # Convert peak fitting results to database format
    peak_data = []
    for peak in peak_fitting_results.get('peaks', []):
        peak_data.append({
            'position': peak.get('center', 0),
            'amplitude': peak.get('amplitude', 1.0),
            'symmetry': peak.get('symmetry', 'A1g')  # Default symmetry
        })
    
    # Import peaks
    success = db.import_from_peak_fitting(mineral_name, peak_data)
    
    # Save database
    if success:
        db.save_database()
    
    return success

def update_point_group_notations():
    """
    Update all minerals in the database to add both Hermann-Mauguin and Schoenflies point group notations.
    - Derives point group from space group if not present
    - Converts from Hermann-Mauguin to Schoenflies 
    - Stores both notations in separate keys
    """
    db = MineralDatabase()
    
    # Hermann-Mauguin to Schoenflies mapping
    hm_to_schoenflies = {
        '1': 'C1', '-1': 'Ci',
        '2': 'C2', 'm': 'Cs', '2/m': 'C2h',
        '222': 'D2', 'mm2': 'C2v', 'mmm': 'D2h',
        '4': 'C4', '-4': 'S4', '4/m': 'C4h',
        '422': 'D4', '4mm': 'C4v', '-42m': 'D2d', '4/mmm': 'D4h',
        '3': 'C3', '-3': 'C3i',
        '32': 'D3', '3m': 'C3v', '-3m': 'D3d',
        '6': 'C6', '-6': 'C3h', '6/m': 'C6h',
        '622': 'D6', '6mm': 'C6v', '-62m': 'D3h', '6/mmm': 'D6h',
        '23': 'T', 'm-3': 'Th',
        '432': 'O', '-43m': 'Td', 'm-3m': 'Oh'
    }
    
    # Track statistics
    total_minerals = 0
    updated_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Process each mineral
    for mineral_name in db.get_minerals():
        total_minerals += 1
        mineral_data = db.get_mineral_data(mineral_name)
        
        # Skip entries that are not dictionaries (like boolean values or special markers)
        if not isinstance(mineral_data, dict):
            print(f"Skipping {mineral_name}: Not a valid mineral data record (type: {type(mineral_data)})")
            skipped_count += 1
            continue
            
        if not mineral_data:
            print(f"Error: No data found for {mineral_name}")
            skipped_count += 1
            continue
            
        # Step 1: Get/derive point group in Hermann-Mauguin notation
        hm_point_group = ""
        
        # Check if there's already a point group stored
        current_pg = mineral_data.get('point_group', '')
        
        # If current point group looks like Schoenflies (has letters), try reverse lookup
        if current_pg and any(s in current_pg for s in ['C', 'D', 'T', 'O', 'I', 'S']):
            # Reverse lookup from Schoenflies to Hermann-Mauguin
            reversed_mapping = {v: k for k, v in hm_to_schoenflies.items()}
            if current_pg in reversed_mapping:
                hm_point_group = reversed_mapping[current_pg]
        else:
            # If already in Hermann-Mauguin format or empty
            hm_point_group = current_pg
        
        # If still no point group, try to derive from space group
        if not hm_point_group:
            space_group = mineral_data.get('space_group', '')
            if space_group:
                try:
                    # Handle numeric or string space groups
                    if isinstance(space_group, (int, float, np.integer, np.floating)):
                        # If it's a number, use the _derive_crystal_system to get crystal system
                        crystal_system = db._derive_crystal_system(space_group)
                        # Set a default point group based on the crystal system
                        if 'Triclinic' in crystal_system:
                            hm_point_group = '-1'
                        elif 'Monoclinic' in crystal_system:
                            hm_point_group = '2/m'
                        elif 'Orthorhombic' in crystal_system:
                            hm_point_group = 'mmm'
                        elif 'Tetragonal' in crystal_system:
                            hm_point_group = '4/mmm'
                        elif 'Trigonal' in crystal_system:
                            hm_point_group = '-3m'
                        elif 'Hexagonal' in crystal_system:
                            hm_point_group = '6/mmm'
                        elif 'Cubic' in crystal_system:
                            hm_point_group = 'm-3m'
                        else:
                            # Default for unknown
                            hm_point_group = '1'
                    else:
                        # String space group, use the existing derive function
                        derived_pg = db._derive_point_group(space_group)
                        if derived_pg:
                            hm_point_group = derived_pg
                except Exception as e:
                    print(f"Error deriving point group from space group for {mineral_name}: {e}")
        
        # Step 2: Convert Hermann-Mauguin to Schoenflies
        schoenflies_point_group = ""
        if hm_point_group in hm_to_schoenflies:
            schoenflies_point_group = hm_to_schoenflies[hm_point_group]
        
        # If still no valid point group, assign based on crystal system
        if not hm_point_group or not schoenflies_point_group:
            crystal_system = mineral_data.get('crystal_system', '').lower()
            
            if 'cubic' in crystal_system:
                hm_point_group = 'm-3m'
                schoenflies_point_group = 'Oh'
            elif 'hexagonal' in crystal_system:
                hm_point_group = '6/mmm'
                schoenflies_point_group = 'D6h'
            elif 'trigonal' in crystal_system:
                hm_point_group = '-3m'
                schoenflies_point_group = 'D3d'
            elif 'tetragonal' in crystal_system:
                hm_point_group = '4/mmm'
                schoenflies_point_group = 'D4h'
            elif 'orthorhombic' in crystal_system:
                hm_point_group = 'mmm'
                schoenflies_point_group = 'D2h'
            elif 'monoclinic' in crystal_system:
                hm_point_group = '2/m'
                schoenflies_point_group = 'C2h'
            elif 'triclinic' in crystal_system:
                hm_point_group = '-1'
                schoenflies_point_group = 'Ci'
            else:
                # If all else fails, set to lowest symmetry
                hm_point_group = '1'
                schoenflies_point_group = 'C1'
        
        # Step 3: Update the mineral data with both notations
        if hm_point_group and schoenflies_point_group:
            mineral_data['hermann_mauguin_point_group'] = hm_point_group
            mineral_data['schoenflies_point_group'] = schoenflies_point_group
            
            # Keep the original point_group field with Schoenflies for compatibility
            mineral_data['point_group'] = schoenflies_point_group
            
            updated_count += 1
            print(f"Updated {mineral_name}: {hm_point_group} (HM) → {schoenflies_point_group} (Schoenflies)")
        else:
            failed_count += 1
            print(f"Failed to determine point group for {mineral_name}")
    
    # Save the database
    if updated_count > 0:
        success = db.save_database()
        if success:
            print(f"Database saved successfully.")
        else:
            print(f"Failed to save database.")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total minerals: {total_minerals}")
    print(f"Updated with both notations: {updated_count}")
    print(f"Failed to determine point groups: {failed_count}")
    print(f"Skipped entries: {skipped_count}")
    
    return updated_count > 0

if __name__ == "__main__":
    import sys

    
    # Check if an argument is passed
    if len(sys.argv) > 1 and sys.argv[1] == '--update-point-groups':
        # Only update point group notations
        print("Checking and updating point group notations in the database...")
        update_point_group_notations()
        print("Point group update completed. You can now run the program normally.")
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Usage:")
        print("  python mineral_database.py            - Run the GUI")
        print("  python mineral_database.py --update-point-groups - Update point groups in database")
        sys.exit(0)
    else:
        # Run the GUI as normal
        app = MineralDatabaseGUI()
        app.run()