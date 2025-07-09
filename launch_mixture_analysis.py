#!/usr/bin/env python3
"""
Launch script for RamanLab Mixture Analysis

UPDATED: Now launches the new Interactive Mixture Analysis by default.
The new interface provides expert-guided, iterative mixture analysis with:
- Interactive peak selection by clicking
- Real-time spectrum overlay and residual analysis  
- Pseudo-Voigt peak fitting
- Database search and component tracking

Old automated analysis has been removed - use interactive version only
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the mixture analysis interface."""
    print("🔬 RamanLab - Interactive Mixture Analysis (NEW)")
    print("=" * 50)
    print("🎯 Launching expert-guided interactive interface...")
    print("   (Old automated analysis has been removed)")
    print("")
    
    try:
        # Import and run the NEW interactive mixture analysis
        from raman_mixture_analysis_interactive import main as run_interactive_gui
        return run_interactive_gui()
    except ImportError as e:
        print(f"❌ Error importing NEW interactive mixture analysis: {e}")
        print("Please ensure raman_mixture_analysis_interactive.py is in the same directory.")
        print("Make sure you have PySide6 installed: pip install PySide6")
        print("The old automated version has been removed.")
        return 1
    except Exception as e:
        print(f"❌ Error running interactive mixture analysis: {e}")
        print("Please check your installation.")
        print("The old automated version has been removed.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 