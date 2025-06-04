#!/usr/bin/env python3
"""
Hey-Celestian Frequency Analyzer

This module implements systematic frequency analysis for the Hey-Celestian classification system.
It establishes reference frequency ranges for characteristic modes and quantifies systematic 
deviations within each anionic group.

Key Features:
- Reference frequency ranges for each Hey-Celestian group
- Polymerization tracking for silicates (Q^0 to Q^4 species)
- Cation substitution effects analysis
- Mode splitting pattern detection
- Peak parameter correlation analysis

Author: Aaron Celestian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, pearsonr
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import os

class HeyCelestianFrequencyAnalyzer:
    """
    Systematic frequency analysis for Hey-Celestian classification groups.
    """
    
    def __init__(self):
        """Initialize the frequency analyzer with reference ranges."""
        self.reference_ranges = self._load_reference_ranges()
        self.polymerization_ranges = self._load_polymerization_ranges()
        self.cation_properties = self._load_cation_properties()
        
    def _load_reference_ranges(self):
        """Load reference frequency ranges for each Hey-Celestian group."""
        return {
            "Framework Modes - Tetrahedral": {
                "primary_modes": [
                    {"name": "Si-O-Si bending", "range": (450, 480), "intensity": "strong", "characteristic": True},
                    {"name": "T-O symmetric stretch", "range": (800, 850), "intensity": "medium", "characteristic": False},
                    {"name": "T-O asymmetric stretch", "range": (1000, 1200), "intensity": "very_strong", "characteristic": True}
                ],
                "secondary_modes": [
                    {"name": "Ring breathing", "range": (300, 400), "intensity": "weak", "characteristic": False},
                    {"name": "Framework deformation", "range": (500, 600), "intensity": "weak", "characteristic": False}
                ]
            },
            "Framework Modes - Octahedral": {
                "primary_modes": [
                    {"name": "M-O stretch", "range": (600, 800), "intensity": "strong", "characteristic": True},
                    {"name": "M-O bend", "range": (300, 500), "intensity": "medium", "characteristic": True}
                ]
            },
            "Characteristic Vibrational Mode - Carbonate": {
                "primary_modes": [
                    {"name": "ν₁ symmetric stretch", "range": (1080, 1090), "intensity": "very_strong", "characteristic": True},
                    {"name": "ν₄ in-plane bend", "range": (710, 720), "intensity": "medium", "characteristic": True}
                ],
                "secondary_modes": [
                    {"name": "ν₂ out-of-plane bend", "range": (870, 880), "intensity": "weak", "characteristic": False},
                    {"name": "ν₃ asymmetric stretch", "range": (1400, 1500), "intensity": "weak", "characteristic": False}
                ]
            },
            "Characteristic Vibrational Mode - Sulfate": {
                "primary_modes": [
                    {"name": "ν₁ symmetric stretch", "range": (1000, 1020), "intensity": "very_strong", "characteristic": True},
                    {"name": "ν₄ asymmetric bend", "range": (600, 650), "intensity": "medium", "characteristic": True}
                ]
            },
            "Characteristic Vibrational Mode - Phosphate": {
                "primary_modes": [
                    {"name": "ν₁ symmetric stretch", "range": (950, 980), "intensity": "very_strong", "characteristic": True},
                    {"name": "ν₄ asymmetric bend", "range": (550, 650), "intensity": "medium", "characteristic": True}
                ]
            },
            "Chain Modes - Single Chain": {
                "primary_modes": [
                    {"name": "Si-O-Si chain stretch", "range": (650, 680), "intensity": "strong", "characteristic": True},
                    {"name": "Si-O stretch", "range": (950, 1050), "intensity": "strong", "characteristic": True}
                ]
            },
            "Chain Modes - Double Chain": {
                "primary_modes": [
                    {"name": "Si-O-Si double chain stretch", "range": (665, 675), "intensity": "strong", "characteristic": True},
                    {"name": "Si-O stretch", "range": (920, 1020), "intensity": "strong", "characteristic": True}
                ]
            },
            "Ring Modes - Cyclosilicates": {
                "primary_modes": [
                    {"name": "Ring breathing", "range": (620, 660), "intensity": "strong", "characteristic": True},
                    {"name": "Si-O stretch", "range": (900, 1000), "intensity": "medium", "characteristic": True}
                ]
            },
            "Layer Modes - Sheet Silicates": {
                "primary_modes": [
                    {"name": "Si-O stretch", "range": (950, 1100), "intensity": "medium", "characteristic": True},
                    {"name": "OH stretch", "range": (3500, 3700), "intensity": "strong", "characteristic": True}
                ],
                "secondary_modes": [
                    {"name": "Layer deformation", "range": (200, 400), "intensity": "weak", "characteristic": False}
                ]
            }
        }
    
    def _load_polymerization_ranges(self):
        """Load silicate polymerization (Q^n species) frequency ranges."""
        return {
            "Q0": {  # Isolated tetrahedra
                "range": (850, 900),
                "description": "Isolated SiO₄ tetrahedra",
                "examples": ["Olivine", "Garnet", "Zircon"],
                "peak_width": "narrow"
            },
            "Q1": {  # Chain end groups
                "range": (900, 950),
                "description": "Chain end groups, sorosilicates",
                "examples": ["Akermanite", "Gehlenite"],
                "peak_width": "narrow"
            },
            "Q2": {  # Chain middle groups
                "range": (950, 1000),
                "description": "Chain silicates, ring silicates",
                "examples": ["Pyroxenes", "Amphiboles"],
                "peak_width": "medium"
            },
            "Q3": {  # Sheet structures
                "range": (1000, 1050),
                "description": "Sheet silicates",
                "examples": ["Micas", "Clay minerals"],
                "peak_width": "medium"
            },
            "Q4": {  # Framework structures
                "range": (1050, 1100),
                "description": "Framework silicates",
                "examples": ["Quartz", "Feldspar", "Zeolites"],
                "peak_width": "broad"
            }
        }
    
    def _load_cation_properties(self):
        """Load cation properties for substitution effect analysis."""
        return {
            "Ca2+": {"radius": 1.00, "electronegativity": 1.0, "field_strength": 2.0},
            "Mg2+": {"radius": 0.72, "electronegativity": 1.31, "field_strength": 3.85},
            "Fe2+": {"radius": 0.78, "electronegativity": 1.83, "field_strength": 3.29},
            "Mn2+": {"radius": 0.83, "electronegativity": 1.55, "field_strength": 2.90},
            "Na+": {"radius": 1.02, "electronegativity": 0.93, "field_strength": 0.98},
            "K+": {"radius": 1.38, "electronegativity": 0.82, "field_strength": 0.52},
            "Al3+": {"radius": 0.54, "electronegativity": 1.61, "field_strength": 10.31},
            "Si4+": {"radius": 0.40, "electronegativity": 1.90, "field_strength": 25.0},
            "Ti4+": {"radius": 0.61, "electronegativity": 1.54, "field_strength": 10.75}
        }
    
    def analyze_frequency_deviations(self, mineral_data, hey_group):
        """
        Analyze systematic frequency deviations within a Hey-Celestian group.
        
        Parameters:
        -----------
        mineral_data : dict
            Dictionary containing mineral spectrum data
        hey_group : str
            Hey-Celestian classification group
            
        Returns:
        --------
        dict : Analysis results including deviations and correlations
        """
        if hey_group not in self.reference_ranges:
            return {"error": f"Unknown Hey-Celestian group: {hey_group}"}
        
        results = {
            "group": hey_group,
            "reference_ranges": self.reference_ranges[hey_group],
            "observed_peaks": [],
            "deviations": [],
            "correlations": {},
            "polymerization_analysis": None,
            "splitting_patterns": []
        }
        
        # Extract peaks from spectrum
        wavenumbers = mineral_data.get("wavenumbers", [])
        intensities = mineral_data.get("intensities", [])
        
        if len(wavenumbers) == 0 or len(intensities) == 0:
            return {"error": "No spectral data available"}
        
        # Find peaks in the spectrum
        peaks, properties = find_peaks(intensities, prominence=0.1, height=0.1)
        peak_positions = wavenumbers[peaks]
        peak_intensities = intensities[peaks]
        
        # Analyze each reference mode
        for mode_type in ["primary_modes", "secondary_modes"]:
            if mode_type in self.reference_ranges[hey_group]:
                for mode in self.reference_ranges[hey_group][mode_type]:
                    mode_analysis = self._analyze_mode_deviation(
                        peak_positions, peak_intensities, mode
                    )
                    if mode_analysis:
                        results["observed_peaks"].append(mode_analysis)
                        if mode_analysis["deviation"] is not None:
                            results["deviations"].append(mode_analysis["deviation"])
        
        # Perform polymerization analysis for silicates
        if "silicate" in hey_group.lower() or "framework" in hey_group.lower():
            results["polymerization_analysis"] = self._analyze_polymerization(
                peak_positions, peak_intensities
            )
        
        # Detect splitting patterns
        results["splitting_patterns"] = self._detect_splitting_patterns(
            peak_positions, peak_intensities
        )
        
        # Calculate correlation statistics
        if results["deviations"]:
            results["correlations"]["mean_deviation"] = np.mean(results["deviations"])
            results["correlations"]["std_deviation"] = np.std(results["deviations"])
        
        return results
    
    def _analyze_mode_deviation(self, peak_positions, peak_intensities, mode):
        """Analyze deviation of observed peaks from reference mode."""
        ref_range = mode["range"]
        ref_center = (ref_range[0] + ref_range[1]) / 2
        
        # Find peaks within or near the reference range
        tolerance = 50  # cm⁻¹ tolerance
        candidates = []
        
        for i, pos in enumerate(peak_positions):
            if ref_range[0] - tolerance <= pos <= ref_range[1] + tolerance:
                candidates.append({
                    "position": pos,
                    "intensity": peak_intensities[i],
                    "distance_from_center": abs(pos - ref_center)
                })
        
        if not candidates:
            return None
        
        # Select the closest peak to the reference center
        best_match = min(candidates, key=lambda x: x["distance_from_center"])
        
        deviation = best_match["position"] - ref_center
        
        return {
            "mode_name": mode["name"],
            "reference_range": ref_range,
            "reference_center": ref_center,
            "observed_position": best_match["position"],
            "observed_intensity": best_match["intensity"],
            "deviation": deviation,
            "relative_deviation": deviation / ref_center * 100,
            "within_range": ref_range[0] <= best_match["position"] <= ref_range[1]
        }
    
    def _analyze_polymerization(self, peak_positions, peak_intensities):
        """Analyze silicate polymerization based on Si-O stretching frequencies."""
        polymerization_results = {
            "detected_species": [],
            "polymerization_index": None,
            "structural_complexity": None
        }
        
        # Check for each Q^n species
        for qn_species, qn_data in self.polymerization_ranges.items():
            qn_range = qn_data["range"]
            
            # Find peaks in this range
            matching_peaks = []
            for i, pos in enumerate(peak_positions):
                if qn_range[0] <= pos <= qn_range[1]:
                    matching_peaks.append({
                        "position": pos,
                        "intensity": peak_intensities[i],
                        "species": qn_species
                    })
            
            if matching_peaks:
                # Take the strongest peak in this range
                strongest_peak = max(matching_peaks, key=lambda x: x["intensity"])
                polymerization_results["detected_species"].append({
                    "species": qn_species,
                    "position": strongest_peak["position"],
                    "intensity": strongest_peak["intensity"],
                    "description": qn_data["description"]
                })
        
        # Calculate polymerization index
        if polymerization_results["detected_species"]:
            # Simple polymerization index based on highest Q^n species present
            max_qn = max([int(species["species"][1]) for species in polymerization_results["detected_species"]])
            polymerization_results["polymerization_index"] = max_qn / 4.0  # Normalize to 0-1
            
            # Assess structural complexity
            if max_qn <= 1:
                polymerization_results["structural_complexity"] = "Low (isolated/chain end units)"
            elif max_qn <= 2:
                polymerization_results["structural_complexity"] = "Medium (chains/rings)"
            elif max_qn <= 3:
                polymerization_results["structural_complexity"] = "High (sheets)"
            else:
                polymerization_results["structural_complexity"] = "Very High (frameworks)"
        
        return polymerization_results
    
    def _detect_splitting_patterns(self, peak_positions, peak_intensities):
        """Detect characteristic doublets and multiplets that indicate symmetry reduction."""
        splitting_patterns = []
        
        # Look for doublets (peaks separated by 5-50 cm⁻¹)
        for i in range(len(peak_positions) - 1):
            separation = peak_positions[i + 1] - peak_positions[i]
            if 5 <= separation <= 50:
                intensity_ratio = peak_intensities[i] / peak_intensities[i + 1]
                
                splitting_patterns.append({
                    "type": "doublet",
                    "positions": [peak_positions[i], peak_positions[i + 1]],
                    "separation": separation,
                    "intensity_ratio": intensity_ratio,
                    "interpretation": self._interpret_splitting(separation, intensity_ratio)
                })
        
        # Look for triplets and higher multiplets
        # Use DBSCAN clustering to find groups of closely spaced peaks
        if len(peak_positions) >= 3:
            positions_2d = peak_positions.reshape(-1, 1)
            clustering = DBSCAN(eps=30, min_samples=3).fit(positions_2d)
            
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # Ignore noise points
                    cluster_peaks = peak_positions[clustering.labels_ == cluster_id]
                    cluster_intensities = peak_intensities[clustering.labels_ == cluster_id]
                    
                    if len(cluster_peaks) >= 3:
                        splitting_patterns.append({
                            "type": "multiplet",
                            "positions": cluster_peaks.tolist(),
                            "intensities": cluster_intensities.tolist(),
                            "span": cluster_peaks.max() - cluster_peaks.min(),
                            "interpretation": "Possible symmetry reduction or site disorder"
                        })
        
        return splitting_patterns
    
    def _interpret_splitting(self, separation, intensity_ratio):
        """Interpret the cause of peak splitting based on separation and intensity ratio."""
        if 5 <= separation <= 15:
            if 0.8 <= intensity_ratio <= 1.2:
                return "Likely cation ordering or symmetry reduction"
            else:
                return "Possible site disorder or solid solution"
        elif 15 <= separation <= 30:
            return "Possible different coordination environments"
        elif 30 <= separation <= 50:
            return "Different structural sites or phase separation"
        else:
            return "Unknown splitting mechanism"
    
    def analyze_cation_substitution_effects(self, mineral_series_data):
        """
        Analyze cation substitution effects across a mineral series.
        
        Parameters:
        -----------
        mineral_series_data : list
            List of dictionaries containing mineral data and composition
            
        Returns:
        --------
        dict : Correlation analysis results
        """
        if len(mineral_series_data) < 3:
            return {"error": "Need at least 3 samples for correlation analysis"}
        
        results = {
            "correlations": {},
            "trends": {},
            "statistical_tests": {}
        }
        
        # Extract peak positions and cation properties
        peak_data = []
        cation_data = []
        
        for sample in mineral_series_data:
            # Analyze each sample
            analysis = self.analyze_frequency_deviations(
                sample["spectrum_data"], 
                sample["hey_group"]
            )
            
            # Extract peak positions for characteristic modes
            characteristic_peaks = [
                peak for peak in analysis["observed_peaks"] 
                if peak and peak.get("within_range", False)
            ]
            
            if characteristic_peaks and "composition" in sample:
                peak_data.append(characteristic_peaks)
                cation_data.append(sample["composition"])
        
        # Perform correlation analysis
        if len(peak_data) >= 3:
            results = self._perform_correlation_analysis(peak_data, cation_data)
        
        return results
    
    def _perform_correlation_analysis(self, peak_data, cation_data):
        """Perform statistical correlation analysis between peak parameters and cation properties."""
        correlations = {}
        
        # For each mode, analyze correlation with cation properties
        for mode_idx, mode_name in enumerate(["primary_mode_1", "primary_mode_2"]):
            mode_positions = []
            avg_cation_radius = []
            avg_electronegativity = []
            avg_field_strength = []
            
            for i, sample_peaks in enumerate(peak_data):
                if mode_idx < len(sample_peaks):
                    mode_positions.append(sample_peaks[mode_idx]["observed_position"])
                    
                    # Calculate average cation properties
                    composition = cation_data[i]
                    total_cations = sum(composition.values())
                    
                    weighted_radius = sum(
                        frac * self.cation_properties.get(cation, {}).get("radius", 0)
                        for cation, frac in composition.items()
                    ) / total_cations if total_cations > 0 else 0
                    
                    weighted_electronegativity = sum(
                        frac * self.cation_properties.get(cation, {}).get("electronegativity", 0)
                        for cation, frac in composition.items()
                    ) / total_cations if total_cations > 0 else 0
                    
                    weighted_field_strength = sum(
                        frac * self.cation_properties.get(cation, {}).get("field_strength", 0)
                        for cation, frac in composition.items()
                    ) / total_cations if total_cations > 0 else 0
                    
                    avg_cation_radius.append(weighted_radius)
                    avg_electronegativity.append(weighted_electronegativity)
                    avg_field_strength.append(weighted_field_strength)
            
            if len(mode_positions) >= 3:
                # Calculate correlations
                if avg_cation_radius:
                    r_radius, p_radius = pearsonr(mode_positions, avg_cation_radius)
                    correlations[f"{mode_name}_vs_radius"] = {"r": r_radius, "p": p_radius}
                
                if avg_electronegativity:
                    r_en, p_en = pearsonr(mode_positions, avg_electronegativity)
                    correlations[f"{mode_name}_vs_electronegativity"] = {"r": r_en, "p": p_en}
                
                if avg_field_strength:
                    r_fs, p_fs = pearsonr(mode_positions, avg_field_strength)
                    correlations[f"{mode_name}_vs_field_strength"] = {"r": r_fs, "p": p_fs}
        
        return {"correlations": correlations}

class HeyCelestianFrequencyGUI:
    """GUI for Hey-Celestian frequency analysis."""
    
    def __init__(self, parent, mineral_database):
        """Initialize the GUI."""
        self.window = tk.Toplevel(parent)
        self.window.title("Hey-Celestian Frequency Analyzer")
        self.window.geometry("1200x800")
        
        self.mineral_database = mineral_database
        self.analyzer = HeyCelestianFrequencyAnalyzer()
        
        self.create_gui()
    
    def create_gui(self):
        """Create the GUI elements."""
        # Create notebook for different analysis types
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_single_mineral_tab()
        self.create_group_analysis_tab()
        self.create_substitution_tab()
        self.create_polymerization_tab()
        
    def create_single_mineral_tab(self):
        """Create tab for single mineral frequency analysis."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Single Mineral Analysis")
        
        # Selection frame
        selection_frame = ttk.LabelFrame(tab, text="Mineral Selection")
        selection_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(selection_frame, text="Select Mineral:").pack(anchor="w")
        self.mineral_var = tk.StringVar()
        self.mineral_combo = ttk.Combobox(selection_frame, textvariable=self.mineral_var, 
                                         state="readonly", width=50)
        self.mineral_combo.pack(fill="x", padx=5, pady=2)
        
        # Update mineral list
        self.update_mineral_list()
        
        # Analysis button
        ttk.Button(selection_frame, text="Analyze Frequency Deviations", 
                  command=self.analyze_single_mineral).pack(pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(tab, text="Analysis Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create text widget for results
        self.results_text = tk.Text(results_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_group_analysis_tab(self):
        """Create tab for Hey-Celestian group analysis."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Group Analysis")
        
        # Control panel
        control_frame = ttk.LabelFrame(tab, text="Group Analysis Controls")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Group selection
        ttk.Label(control_frame, text="Select Hey-Celestian Group:").pack(anchor="w")
        self.group_var = tk.StringVar()
        self.group_combo = ttk.Combobox(control_frame, textvariable=self.group_var,
                                       state="readonly", width=50)
        # Populate with available groups
        available_groups = list(self.analyzer.reference_ranges.keys())
        self.group_combo["values"] = available_groups
        self.group_combo.pack(fill="x", padx=5, pady=2)
        
        # Analysis options
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill="x", pady=5)
        
        self.analyze_deviations_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Analyze Frequency Deviations",
                       variable=self.analyze_deviations_var).pack(anchor="w")
        
        self.track_polymerization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Track Polymerization (Silicates only)",
                       variable=self.track_polymerization_var).pack(anchor="w")
        
        self.detect_splitting_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Detect Mode Splitting Patterns",
                       variable=self.detect_splitting_var).pack(anchor="w")
        
        # Analysis buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=5)
        
        ttk.Button(button_frame, text="Analyze Selected Group",
                  command=self.analyze_hey_group).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Export Group Analysis",
                  command=self.export_group_analysis).pack(side="left", padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(tab, text="Group Analysis Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create notebook for different result views
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Statistics tab
        self.stats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.stats_frame, text="Statistics")
        
        self.group_stats_text = tk.Text(self.stats_frame, wrap=tk.WORD)
        stats_scrollbar = ttk.Scrollbar(self.stats_frame, orient="vertical", 
                                       command=self.group_stats_text.yview)
        self.group_stats_text.configure(yscrollcommand=stats_scrollbar.set)
        self.group_stats_text.pack(side="left", fill="both", expand=True)
        stats_scrollbar.pack(side="right", fill="y")
        
        # Visualization tab
        self.viz_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.viz_frame, text="Visualizations")
        
        # Create matplotlib figure for group analysis plots
        self.group_fig = Figure(figsize=(12, 8), dpi=100)
        self.group_canvas = FigureCanvasTkAgg(self.group_fig, self.viz_frame)
        self.group_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def create_substitution_tab(self):
        """Create tab for cation substitution analysis."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Substitution Effects")
        
        # Implementation placeholder
        ttk.Label(tab, text="Cation Substitution Analysis Coming Soon", 
                 font=("Arial", 16)).pack(expand=True)
    
    def create_polymerization_tab(self):
        """Create tab for polymerization analysis."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Polymerization Analysis")
        
        # Implementation placeholder
        ttk.Label(tab, text="Polymerization Analysis Coming Soon", 
                 font=("Arial", 16)).pack(expand=True)
    
    def update_mineral_list(self):
        """Update the mineral selection combobox."""
        try:
            minerals = self.mineral_database.get_minerals()  # This returns a list
            self.mineral_combo["values"] = sorted(minerals)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mineral list: {str(e)}")
    
    def analyze_single_mineral(self):
        """Analyze frequency deviations for selected mineral."""
        mineral_name = self.mineral_var.get()
        if not mineral_name:
            messagebox.showwarning("Warning", "Please select a mineral first.")
            return
        
        try:
            # Get mineral data
            mineral_data = self.mineral_database.get_mineral_data(mineral_name)
            if not mineral_data:
                messagebox.showerror("Error", f"No data found for {mineral_name}")
                return
            
            # Get Hey-Celestian classification
            hey_group = mineral_data.get("metadata", {}).get("HEY-CELESTIAN GROUP NAME", "Unknown")
            
            if hey_group == "Unknown":
                messagebox.showwarning("Warning", f"No Hey-Celestian classification found for {mineral_name}")
                return
            
            # Prepare spectrum data
            modes = mineral_data.get("modes", {})
            if not modes:
                messagebox.showerror("Error", f"No spectral modes found for {mineral_name}")
                return
            
            # Extract wavenumbers and intensities
            wavenumbers = []
            intensities = []
            for position, mode_data in modes.items():
                wavenumbers.append(float(position))
                intensities.append(float(mode_data.get("intensity", 1.0)))
            
            spectrum_data = {
                "wavenumbers": np.array(wavenumbers),
                "intensities": np.array(intensities)
            }
            
            # Perform analysis
            results = self.analyzer.analyze_frequency_deviations(spectrum_data, hey_group)
            
            # Display results
            self.display_analysis_results(mineral_name, hey_group, results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def display_analysis_results(self, mineral_name, hey_group, results):
        """Display the analysis results in the text widget."""
        self.results_text.delete(1.0, tk.END)
        
        if "error" in results:
            self.results_text.insert(tk.END, f"Error: {results['error']}\n")
            return
        
        # Header
        self.results_text.insert(tk.END, f"Hey-Celestian Frequency Analysis\n")
        self.results_text.insert(tk.END, f"="*50 + "\n\n")
        self.results_text.insert(tk.END, f"Mineral: {mineral_name}\n")
        self.results_text.insert(tk.END, f"Hey-Celestian Group: {hey_group}\n\n")
        
        # Observed peaks
        self.results_text.insert(tk.END, "Observed Characteristic Modes:\n")
        self.results_text.insert(tk.END, "-"*30 + "\n")
        
        for peak in results["observed_peaks"]:
            if peak:
                self.results_text.insert(tk.END, f"Mode: {peak['mode_name']}\n")
                self.results_text.insert(tk.END, f"  Reference Range: {peak['reference_range'][0]}-{peak['reference_range'][1]} cm⁻¹\n")
                self.results_text.insert(tk.END, f"  Observed Position: {peak['observed_position']:.1f} cm⁻¹\n")
                self.results_text.insert(tk.END, f"  Deviation: {peak['deviation']:.1f} cm⁻¹ ({peak['relative_deviation']:.2f}%)\n")
                self.results_text.insert(tk.END, f"  Within Range: {'Yes' if peak['within_range'] else 'No'}\n\n")
        
        # Polymerization analysis
        if results["polymerization_analysis"]:
            poly_data = results["polymerization_analysis"]
            self.results_text.insert(tk.END, "Polymerization Analysis:\n")
            self.results_text.insert(tk.END, "-"*30 + "\n")
            
            if poly_data["detected_species"]:
                for species in poly_data["detected_species"]:
                    self.results_text.insert(tk.END, f"{species['species']}: {species['position']:.1f} cm⁻¹ - {species['description']}\n")
                
                if poly_data["polymerization_index"] is not None:
                    self.results_text.insert(tk.END, f"\nPolymerization Index: {poly_data['polymerization_index']:.2f}\n")
                    self.results_text.insert(tk.END, f"Structural Complexity: {poly_data['structural_complexity']}\n\n")
        
        # Splitting patterns
        if results["splitting_patterns"]:
            self.results_text.insert(tk.END, "Splitting Patterns:\n")
            self.results_text.insert(tk.END, "-"*30 + "\n")
            
            for pattern in results["splitting_patterns"]:
                self.results_text.insert(tk.END, f"Type: {pattern['type'].title()}\n")
                self.results_text.insert(tk.END, f"Positions: {', '.join([f'{pos:.1f}' for pos in pattern['positions']])} cm⁻¹\n")
                if "separation" in pattern:
                    self.results_text.insert(tk.END, f"Separation: {pattern['separation']:.1f} cm⁻¹\n")
                self.results_text.insert(tk.END, f"Interpretation: {pattern['interpretation']}\n\n")
        
        # Summary statistics
        if results["correlations"]:
            self.results_text.insert(tk.END, "Summary Statistics:\n")
            self.results_text.insert(tk.END, "-"*30 + "\n")
            
            if "mean_deviation" in results["correlations"]:
                self.results_text.insert(tk.END, f"Mean Deviation: {results['correlations']['mean_deviation']:.2f} cm⁻¹\n")
                self.results_text.insert(tk.END, f"Standard Deviation: {results['correlations']['std_deviation']:.2f} cm⁻¹\n")
    
    def analyze_hey_group(self):
        """Analyze all minerals in a selected Hey-Celestian group."""
        selected_group = self.group_var.get()
        if not selected_group:
            messagebox.showwarning("Warning", "Please select a Hey-Celestian group first.")
            return
        
        try:
            # Get all minerals with this Hey-Celestian classification
            all_mineral_names = self.mineral_database.get_minerals()  # This returns a list
            group_minerals = []
            
            for mineral_name in all_mineral_names:
                mineral_data = self.mineral_database.get_mineral_data(mineral_name)
                if mineral_data:
                    hey_group = mineral_data.get("metadata", {}).get("HEY-CELESTIAN GROUP NAME", "")
                    if hey_group == selected_group:
                        group_minerals.append((mineral_name, mineral_data))
            
            if not group_minerals:
                messagebox.showinfo("Info", f"No minerals found for group: {selected_group}")
                return
            
            # Analyze each mineral in the group
            group_results = []
            for mineral_name, mineral_data in group_minerals:
                modes = mineral_data.get("modes", {})
                if modes:
                    # Extract spectrum data
                    wavenumbers = []
                    intensities = []
                    for position, mode_data in modes.items():
                        wavenumbers.append(float(position))
                        intensities.append(float(mode_data.get("intensity", 1.0)))
                    
                    spectrum_data = {
                        "wavenumbers": np.array(wavenumbers),
                        "intensities": np.array(intensities)
                    }
                    
                    # Analyze this mineral
                    analysis = self.analyzer.analyze_frequency_deviations(spectrum_data, selected_group)
                    if "error" not in analysis:
                        group_results.append({
                            "mineral_name": mineral_name,
                            "analysis": analysis
                        })
            
            # Generate group statistics and visualizations
            self.display_group_analysis_results(selected_group, group_results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Group analysis failed: {str(e)}")
    
    def display_group_analysis_results(self, group_name, group_results):
        """Display group analysis results."""
        self.group_stats_text.delete(1.0, tk.END)
        
        # Header
        self.group_stats_text.insert(tk.END, f"Hey-Celestian Group Analysis: {group_name}\n")
        self.group_stats_text.insert(tk.END, f"="*60 + "\n\n")
        self.group_stats_text.insert(tk.END, f"Number of minerals analyzed: {len(group_results)}\n\n")
        
        # Collect all deviations for statistical analysis
        all_deviations = []
        mode_deviations = {}
        polymerization_data = []
        splitting_patterns = []
        
        for result in group_results:
            analysis = result["analysis"]
            
            # Collect deviations
            if analysis.get("deviations"):
                all_deviations.extend(analysis["deviations"])
            
            # Collect mode-specific deviations
            for peak in analysis.get("observed_peaks", []):
                if peak:
                    mode_name = peak["mode_name"]
                    if mode_name not in mode_deviations:
                        mode_deviations[mode_name] = []
                    mode_deviations[mode_name].append(peak["deviation"])
            
            # Collect polymerization data
            if analysis.get("polymerization_analysis"):
                poly_data = analysis["polymerization_analysis"]
                if poly_data.get("polymerization_index") is not None:
                    polymerization_data.append({
                        "mineral": result["mineral_name"],
                        "index": poly_data["polymerization_index"],
                        "complexity": poly_data["structural_complexity"]
                    })
            
            # Collect splitting patterns
            if analysis.get("splitting_patterns"):
                splitting_patterns.extend([
                    {"mineral": result["mineral_name"], "pattern": pattern}
                    for pattern in analysis["splitting_patterns"]
                ])
        
        # Display overall statistics
        if all_deviations:
            self.group_stats_text.insert(tk.END, "Overall Frequency Deviation Statistics:\n")
            self.group_stats_text.insert(tk.END, "-"*40 + "\n")
            self.group_stats_text.insert(tk.END, f"Mean deviation: {np.mean(all_deviations):.2f} cm⁻¹\n")
            self.group_stats_text.insert(tk.END, f"Standard deviation: {np.std(all_deviations):.2f} cm⁻¹\n")
            self.group_stats_text.insert(tk.END, f"Range: {np.min(all_deviations):.2f} to {np.max(all_deviations):.2f} cm⁻¹\n\n")
        
        # Display mode-specific statistics
        if mode_deviations:
            self.group_stats_text.insert(tk.END, "Mode-Specific Deviation Statistics:\n")
            self.group_stats_text.insert(tk.END, "-"*40 + "\n")
            for mode_name, deviations in mode_deviations.items():
                self.group_stats_text.insert(tk.END, f"{mode_name}:\n")
                self.group_stats_text.insert(tk.END, f"  Mean: {np.mean(deviations):.2f} cm⁻¹\n")
                self.group_stats_text.insert(tk.END, f"  Std: {np.std(deviations):.2f} cm⁻¹\n")
                self.group_stats_text.insert(tk.END, f"  Range: {np.min(deviations):.2f} to {np.max(deviations):.2f} cm⁻¹\n\n")
        
        # Display polymerization analysis
        if polymerization_data:
            self.group_stats_text.insert(tk.END, "Polymerization Analysis Summary:\n")
            self.group_stats_text.insert(tk.END, "-"*40 + "\n")
            indices = [p["index"] for p in polymerization_data]
            self.group_stats_text.insert(tk.END, f"Average polymerization index: {np.mean(indices):.3f}\n")
            self.group_stats_text.insert(tk.END, f"Range: {np.min(indices):.3f} to {np.max(indices):.3f}\n\n")
            
            # Group by complexity
            complexity_counts = {}
            for p in polymerization_data:
                complexity = p["complexity"]
                if complexity not in complexity_counts:
                    complexity_counts[complexity] = 0
                complexity_counts[complexity] += 1
            
            self.group_stats_text.insert(tk.END, "Structural Complexity Distribution:\n")
            for complexity, count in complexity_counts.items():
                self.group_stats_text.insert(tk.END, f"  {complexity}: {count} minerals\n")
            self.group_stats_text.insert(tk.END, "\n")
        
        # Display splitting patterns summary
        if splitting_patterns:
            self.group_stats_text.insert(tk.END, "Mode Splitting Patterns Summary:\n")
            self.group_stats_text.insert(tk.END, "-"*40 + "\n")
            self.group_stats_text.insert(tk.END, f"Total patterns detected: {len(splitting_patterns)}\n")
            
            pattern_types = {}
            for sp in splitting_patterns:
                pattern_type = sp["pattern"]["type"]
                if pattern_type not in pattern_types:
                    pattern_types[pattern_type] = 0
                pattern_types[pattern_type] += 1
            
            for pattern_type, count in pattern_types.items():
                self.group_stats_text.insert(tk.END, f"  {pattern_type.title()}: {count}\n")
        
        # Create visualizations
        self.create_group_visualizations(group_name, group_results, mode_deviations, polymerization_data)
    
    def create_group_visualizations(self, group_name, group_results, mode_deviations, polymerization_data):
        """Create visualizations for group analysis."""
        self.group_fig.clear()
        
        # Determine subplot layout based on available data
        n_plots = 2  # Always have deviation histogram and box plot
        if polymerization_data:
            n_plots += 1
        if mode_deviations and len(mode_deviations) > 1:
            n_plots += 1
        
        # Create subplot grid
        if n_plots <= 2:
            rows, cols = 1, 2
        elif n_plots <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        plot_idx = 1
        
        # 1. Deviation histogram
        ax1 = self.group_fig.add_subplot(rows, cols, plot_idx)
        plot_idx += 1
        
        all_deviations = []
        for result in group_results:
            if result["analysis"].get("deviations"):
                all_deviations.extend(result["analysis"]["deviations"])
        
        if all_deviations:
            ax1.hist(all_deviations, bins=20, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Frequency Deviation (cm⁻¹)')
            ax1.set_ylabel('Count')
            ax1.set_title(f'Deviation Distribution - {group_name}')
            ax1.axvline(np.mean(all_deviations), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_deviations):.2f}')
            ax1.legend()
        
        # 2. Mode-specific box plot
        if mode_deviations and len(mode_deviations) > 1:
            ax2 = self.group_fig.add_subplot(rows, cols, plot_idx)
            plot_idx += 1
            
            mode_names = list(mode_deviations.keys())
            mode_values = [mode_deviations[name] for name in mode_names]
            
            ax2.boxplot(mode_values, labels=mode_names)
            ax2.set_ylabel('Frequency Deviation (cm⁻¹)')
            ax2.set_title('Mode-Specific Deviations')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Polymerization scatter plot
        if polymerization_data:
            ax3 = self.group_fig.add_subplot(rows, cols, plot_idx)
            plot_idx += 1
            
            mineral_names = [p["mineral"] for p in polymerization_data]
            indices = [p["index"] for p in polymerization_data]
            
            ax3.scatter(range(len(indices)), indices, alpha=0.7)
            ax3.set_xlabel('Mineral Index')
            ax3.set_ylabel('Polymerization Index')
            ax3.set_title('Polymerization Index Distribution')
            ax3.set_xticks(range(len(mineral_names)))
            ax3.set_xticklabels(mineral_names, rotation=45, ha='right')
        
        # 4. Reference range comparison
        if len(group_results) > 1:
            ax4 = self.group_fig.add_subplot(rows, cols, plot_idx)
            plot_idx += 1
            
            # Create a comparison of observed vs reference ranges
            reference_ranges = self.analyzer.reference_ranges.get(group_name, {})
            if "primary_modes" in reference_ranges:
                mode_centers = []
                mode_names = []
                observed_positions = []
                
                for mode in reference_ranges["primary_modes"]:
                    mode_names.append(mode["name"][:20])  # Truncate long names
                    mode_centers.append((mode["range"][0] + mode["range"][1]) / 2)
                    
                    # Collect observed positions for this mode
                    mode_positions = []
                    for result in group_results:
                        for peak in result["analysis"].get("observed_peaks", []):
                            if peak and peak["mode_name"] == mode["name"]:
                                mode_positions.append(peak["observed_position"])
                    
                    if mode_positions:
                        observed_positions.append(np.mean(mode_positions))
                    else:
                        observed_positions.append(None)
                
                # Plot reference vs observed
                x_pos = range(len(mode_names))
                ax4.bar([x - 0.2 for x in x_pos], mode_centers, width=0.4, 
                       label='Reference', alpha=0.7)
                
                observed_clean = [pos for pos in observed_positions if pos is not None]
                x_pos_clean = [x for x, pos in zip(x_pos, observed_positions) if pos is not None]
                
                if observed_clean:
                    ax4.bar([x + 0.2 for x in x_pos_clean], observed_clean, width=0.4, 
                           label='Observed', alpha=0.7)
                
                ax4.set_xlabel('Vibrational Modes')
                ax4.set_ylabel('Frequency (cm⁻¹)')
                ax4.set_title('Reference vs Observed Frequencies')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(mode_names, rotation=45, ha='right')
                ax4.legend()
        
        plt.tight_layout()
        self.group_canvas.draw()
    
    def export_group_analysis(self):
        """Export group analysis results to CSV."""
        selected_group = self.group_var.get()
        if not selected_group:
            messagebox.showwarning("Warning", "Please run group analysis first.")
            return
        
        # Implementation for CSV export
        messagebox.showinfo("Info", "Export functionality coming soon!")
    
def open_hey_celestian_frequency_analyzer(parent, mineral_database):
    """Open the Hey-Celestian Frequency Analyzer window."""
    try:
        HeyCelestianFrequencyGUI(parent, mineral_database)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open Hey-Celestian Frequency Analyzer: {str(e)}")

if __name__ == "__main__":
    # Test the analyzer
    analyzer = HeyCelestianFrequencyAnalyzer()
    print("Hey-Celestian Frequency Analyzer initialized successfully!")
    print(f"Available groups: {list(analyzer.reference_ranges.keys())}")
    print(f"Polymerization species: {list(analyzer.polymerization_ranges.keys())}") 