#!/usr/bin/env python3
"""
Line Scan Raman Spectrum Splitter

This module provides functionality to split line scan Raman spectroscopy data files
into individual spectrum files that can be imported into the RamanLab 
batch processing system.

Input format: 
- Row 1: Raman shift values (wavenumbers)
- Column 1: Scan numbers/positions  
- Each subsequent column: Intensity data for one spectrum

Output: Individual .txt files in two-column format (wavenumber, intensity)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re


class LineScanSplitter:
    """Class to handle splitting line scan Raman data into individual spectrum files."""
    
    def __init__(self):
        """Initialize the LineScanSplitter."""
        self.raman_shifts = None
        self.spectra_data = None
        self.scan_numbers = None
        self.output_directory = None
    
    def load_line_scan_file(self, file_path):
        """
        Load a line scan Raman data file.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the line scan data file
            
        Returns:
        --------
        bool
            True if successfully loaded, False otherwise
        """
        try:
            # Try reading with different delimiters
            delimiters = ['\t', ',', ';', ' ']
            
            for delimiter in delimiters:
                try:
                    # Read the file
                    data = pd.read_csv(file_path, delimiter=delimiter, header=None)
                    
                    # Check if we have at least 2 rows and 2 columns
                    if data.shape[0] >= 2 and data.shape[1] >= 2:
                        # Extract Raman shifts (first row, excluding first column)
                        self.raman_shifts = data.iloc[0, 1:].values.astype(float)
                        
                        # Extract scan numbers (first column, excluding first row)
                        self.scan_numbers = data.iloc[1:, 0].values
                        
                        # Extract spectra data (excluding first row and first column)
                        self.spectra_data = data.iloc[1:, 1:].values.astype(float)
                        
                        print(f"Successfully loaded line scan data:")
                        print(f"  - {len(self.raman_shifts)} wavenumber points")
                        print(f"  - {len(self.scan_numbers)} spectra")
                        print(f"  - Wavenumber range: {self.raman_shifts.min():.1f} - {self.raman_shifts.max():.1f} cm⁻¹")
                        
                        return True
                        
                except Exception as e:
                    continue
            
            # If all delimiters failed, try numpy loadtxt
            try:
                data = np.loadtxt(file_path)
                if data.shape[0] >= 2 and data.shape[1] >= 2:
                    self.raman_shifts = data[0, 1:]
                    self.scan_numbers = data[1:, 0]
                    self.spectra_data = data[1:, 1:]
                    
                    print(f"Successfully loaded line scan data with numpy:")
                    print(f"  - {len(self.raman_shifts)} wavenumber points")
                    print(f"  - {len(self.scan_numbers)} spectra")
                    print(f"  - Wavenumber range: {self.raman_shifts.min():.1f} - {self.raman_shifts.max():.1f} cm⁻¹")
                    
                    return True
            except Exception as e:
                pass
            
            print(f"Error: Could not parse file {file_path}")
            return False
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return False
    
    def validate_data(self):
        """
        Validate that the loaded data is consistent.
        
        Returns:
        --------
        bool
            True if data is valid, False otherwise
        """
        if self.raman_shifts is None or self.spectra_data is None:
            print("Error: No data loaded")
            return False
        
        if len(self.raman_shifts) != self.spectra_data.shape[1]:
            print(f"Error: Mismatch between wavenumber points ({len(self.raman_shifts)}) and spectrum data columns ({self.spectra_data.shape[1]})")
            return False
        
        if len(self.scan_numbers) != self.spectra_data.shape[0]:
            print(f"Error: Mismatch between scan numbers ({len(self.scan_numbers)}) and spectrum data rows ({self.spectra_data.shape[0]})")
            return False
        
        return True
    
    def split_to_individual_files(self, output_directory, base_filename=None, include_scan_number=True):
        """
        Split the line scan data into individual spectrum files.
        
        Parameters:
        -----------
        output_directory : str or Path
            Directory to save individual spectrum files
        base_filename : str, optional
            Base name for output files. If None, uses input filename
        include_scan_number : bool, default True
            Whether to include scan number in filename
            
        Returns:
        --------
        list
            List of created file paths
        """
        if not self.validate_data():
            return []
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if base_filename is None:
            base_filename = "spectrum"
        
        # Clean base filename to make it filesystem-safe
        base_filename = re.sub(r'[<>:"/\\|?*]', '_', base_filename)
        
        created_files = []
        
        for i in range(len(self.scan_numbers)):
            scan_num = self.scan_numbers[i]
            intensities = self.spectra_data[i, :]
            
            # Create filename
            if include_scan_number:
                filename = f"{base_filename}_scan_{scan_num}.txt"
            else:
                filename = f"{base_filename}_{i+1:03d}.txt"
            
            file_path = output_dir / filename
            
            try:
                # Create two-column data (wavenumber, intensity)
                spectrum_data = np.column_stack((self.raman_shifts, intensities))
                
                # Save as tab-delimited text file
                np.savetxt(file_path, spectrum_data, delimiter='\t', 
                          fmt='%.3f\t%.6f', 
                          header=f"Wavenumber (cm-1)\tIntensity\n# Scan number: {scan_num}",
                          comments='')
                
                created_files.append(str(file_path))
                
            except Exception as e:
                print(f"Error saving spectrum {i+1}: {e}")
                continue
        
        print(f"Successfully created {len(created_files)} individual spectrum files in {output_directory}")
        return created_files
    
    def get_info(self):
        """
        Get information about the loaded data.
        
        Returns:
        --------
        dict
            Dictionary with data information
        """
        if not self.validate_data():
            return {}
        
        return {
            'num_spectra': len(self.scan_numbers),
            'num_wavenumbers': len(self.raman_shifts),
            'wavenumber_range': (self.raman_shifts.min(), self.raman_shifts.max()),
            'scan_numbers': list(self.scan_numbers),
            'intensity_range': (self.spectra_data.min(), self.spectra_data.max()),
            'data_shape': self.spectra_data.shape
        }


class LineScanSplitterGUI:
    """GUI interface for the LineScanSplitter."""
    
    def __init__(self):
        """Initialize the GUI."""
        self.splitter = LineScanSplitter()
        self.create_gui()
    
    def create_gui(self):
        """Create the GUI interface."""
        self.root = tk.Tk()
        self.root.title("Line Scan Raman Splitter")
        self.root.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Line Scan Raman Spectrum Splitter", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection
        ttk.Label(main_frame, text="Input File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.file_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.file_path_var, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 5))
        ttk.Button(main_frame, text="Browse", command=self.browse_input_file).grid(row=1, column=2, pady=5)
        
        # Output directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_dir_var, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 5))
        ttk.Button(main_frame, text="Browse", command=self.browse_output_dir).grid(row=2, column=2, pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        options_frame.columnconfigure(1, weight=1)
        
        # Base filename
        ttk.Label(options_frame, text="Base Filename:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.base_filename_var = tk.StringVar(value="spectrum")
        ttk.Entry(options_frame, textvariable=self.base_filename_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # Include scan number option
        self.include_scan_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include scan number in filename", 
                       variable=self.include_scan_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        ttk.Button(buttons_frame, text="Load & Preview", command=self.load_and_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Split Files", command=self.split_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # Info text area
        info_frame = ttk.LabelFrame(main_frame, text="File Information", padding="10")
        info_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(info_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.info_text = tk.Text(text_frame, height=10, width=70, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure main frame row weights
        main_frame.rowconfigure(5, weight=1)
    
    def browse_input_file(self):
        """Browse for input file."""
        filename = filedialog.askopenfilename(
            title="Select Line Scan Raman Data File",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            # Auto-set output directory to same as input file
            if not self.output_dir_var.get():
                input_dir = os.path.dirname(filename)
                output_dir = os.path.join(input_dir, "split_spectra")
                self.output_dir_var.set(output_dir)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def load_and_preview(self):
        """Load file and show preview information."""
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select an input file")
            return
        
        # Clear previous info
        self.info_text.delete(1.0, tk.END)
        
        # Load file
        if self.splitter.load_line_scan_file(file_path):
            info = self.splitter.get_info()
            
            preview_text = f"""File loaded successfully!

Data Summary:
- Number of spectra: {info['num_spectra']}
- Number of wavenumber points: {info['num_wavenumbers']}
- Wavenumber range: {info['wavenumber_range'][0]:.1f} - {info['wavenumber_range'][1]:.1f} cm⁻¹
- Intensity range: {info['intensity_range'][0]:.1f} - {info['intensity_range'][1]:.1f}
- Data dimensions: {info['data_shape'][0]} spectra × {info['data_shape'][1]} points

Scan Numbers: {', '.join(map(str, info['scan_numbers']))}

Ready to split into individual files.
"""
            self.info_text.insert(1.0, preview_text)
        else:
            self.info_text.insert(1.0, "Error: Could not load the file. Please check the file format.")
    
    def split_files(self):
        """Split the loaded data into individual files."""
        if self.splitter.raman_shifts is None:
            messagebox.showerror("Error", "Please load a file first")
            return
        
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        base_filename = self.base_filename_var.get()
        if not base_filename:
            base_filename = "spectrum"
        
        try:
            created_files = self.splitter.split_to_individual_files(
                output_dir, 
                base_filename,
                self.include_scan_var.get()
            )
            
            if created_files:
                success_text = f"""
Split completed successfully!

Created {len(created_files)} individual spectrum files in:
{output_dir}

Files created:
{chr(10).join([os.path.basename(f) for f in created_files[:10]])}
{'...' if len(created_files) > 10 else ''}

These files can now be imported into the RamanLab batch processing system.
"""
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(1.0, success_text)
                
                messagebox.showinfo("Success", f"Successfully created {len(created_files)} spectrum files!")
            else:
                messagebox.showerror("Error", "No files were created. Check the console for error messages.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error during splitting: {e}")
    
    def clear_all(self):
        """Clear all fields and data."""
        self.file_path_var.set("")
        self.output_dir_var.set("")
        self.base_filename_var.set("spectrum")
        self.include_scan_var.set(True)
        self.info_text.delete(1.0, tk.END)
        self.splitter = LineScanSplitter()
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()


def split_line_scan_file(input_file, output_directory=None, base_filename=None, include_scan_number=True):
    """
    Convenience function to split a line scan file programmatically.
    
    Parameters:
    -----------
    input_file : str or Path
        Path to the line scan data file
    output_directory : str or Path, optional
        Directory to save individual files. If None, creates 'split_spectra' in input directory
    base_filename : str, optional
        Base name for output files. If None, uses input filename
    include_scan_number : bool, default True
        Whether to include scan number in filename
    
    Returns:
    --------
    list
        List of created file paths
    """
    splitter = LineScanSplitter()
    
    if not splitter.load_line_scan_file(input_file):
        return []
    
    if output_directory is None:
        input_path = Path(input_file)
        output_directory = input_path.parent / "split_spectra"
    
    if base_filename is None:
        input_path = Path(input_file)
        base_filename = input_path.stem
    
    return splitter.split_to_individual_files(output_directory, base_filename, include_scan_number)


if __name__ == "__main__":
    # Run GUI if script is executed directly
    gui = LineScanSplitterGUI()
    gui.run() 