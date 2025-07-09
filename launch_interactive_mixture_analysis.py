#!/usr/bin/env python3
"""
RamanLab Interactive Mixture Analysis Launcher

This script launches the interactive mixture analysis interface where users can:
1. Search the database for spectrum matches
2. Overlay and select matching peaks interactively
3. Fit pseudo-Voigt peaks to build synthetic spectra iteratively
4. Search residuals for additional components
5. Build complete mixture models through guided analysis

Usage:
    python launch_interactive_mixture_analysis.py
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Launch the interactive mixture analysis interface."""
    print("🚀 Launching RamanLab Interactive Mixture Analysis...")
    
    try:
        # Import the interactive analyzer
        from raman_mixture_analysis_interactive import main as run_analyzer
        
        print("✅ Successfully imported interactive mixture analysis")
        print("🎯 Starting interactive interface...")
        print("=" * 60)
        print("INTERACTIVE MIXTURE ANALYSIS WORKFLOW:")
        print("1. Load your spectrum data (or use demo data)")
        print("2. Search database for matches")
        print("3. Select best match from top 10 results")
        print("4. Click on peaks in overlay plot to select them")
        print("5. Fit pseudo-Voigt peaks to selected positions")
        print("6. Search residual for additional components")
        print("7. Repeat steps 3-6 until satisfied")
        print("8. Finalize analysis to see complete results")
        print("=" * 60)
        
        # Launch the application
        return run_analyzer()
        
    except ImportError as e:
        print(f"❌ Error: Could not import interactive mixture analysis: {e}")
        print("Make sure you have all required dependencies:")
        print("  - PySide6 (pip install PySide6)")
        print("  - matplotlib")
        print("  - numpy")
        print("  - scipy")
        print("  - scikit-learn")
        return 1
        
    except Exception as e:
        print(f"❌ Error launching interactive mixture analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 