"""
Missing methods that need to be added to the RamanAnalysisApp class in raman_analysis_app.py
These should be added as methods within the class.
"""

def open_multi_spectrum_manager(self):
    """Open the Multi-Spectrum Manager window."""
    try:
        from multi_spectrum_manager import MultiSpectrumManager
        MultiSpectrumManager(self.root)
    except Exception as e:
        messagebox.showerror(
            "Error", f"Failed to open Multi-Spectrum Manager: {str(e)}"
        )

def open_hey_celestian_frequency_analyzer(self):
    """Open the Hey Celestian Frequency Analyzer window."""
    try:
        from hey_celestian_frequency_analyzer import open_hey_celestian_frequency_analyzer
        # Pass the mineral database to the analyzer
        mineral_database = getattr(self.raman, 'database', {})
        open_hey_celestian_frequency_analyzer(self.root, mineral_database)
    except Exception as e:
        messagebox.showerror(
            "Error", f"Failed to open Hey Celestian Frequency Analyzer: {str(e)}"
        )

def open_chemical_strain_analyzer(self):
    """Open the Chemical Strain Analyzer window."""
    try:
        from chemical_strain_analyzer import ChemicalStrainAnalysisApp
        # Create a new top-level window
        analyzer_window = tk.Toplevel(self.root)
        analyzer_window.title("Chemical Strain Analysis")
        analyzer_window.geometry("1400x900")
        
        # Create the analyzer instance
        analyzer = ChemicalStrainAnalysisApp(analyzer_window)
        
        # Store the analyzer instance in the window to prevent garbage collection
        analyzer_window.analyzer = analyzer
    except Exception as e:
        messagebox.showerror(
            "Error", f"Failed to open Chemical Strain Analyzer: {str(e)}"
        )

def open_stress_strain_analyzer(self):
    """Open the Stress Strain Analyzer window."""
    try:
        from stress_strain_analyzer import StressStrainAnalyzer
        # Create a new top-level window
        analyzer_window = tk.Toplevel(self.root)
        analyzer_window.title("Stress/Strain Analysis")
        analyzer_window.geometry("1200x800")
        
        # Create the analyzer instance
        analyzer = StressStrainAnalyzer(analyzer_window)
        
        # Store the analyzer instance in the window to prevent garbage collection
        analyzer_window.analyzer = analyzer
    except Exception as e:
        messagebox.showerror(
            "Error", f"Failed to open Stress Strain Analyzer: {str(e)}"
        ) 