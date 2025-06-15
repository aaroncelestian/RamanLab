#!/usr/bin/env python3
"""
Test script to verify that the quantitative analysis import fix works.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path (like main.py does)
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_quantitative_analysis_import():
    """Test the quantitative analysis import with the same conditions as the main app."""
    
    print("=" * 60)
    print("TESTING QUANTITATIVE ANALYSIS IMPORT FIX")
    print("=" * 60)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Python path includes: {sys.path[:3]}...")
    print()
    
    try:
        # Test the path resolution logic from the main window
        print("Testing path resolution logic...")
        
        # Simulate the path logic from main_window.py
        # This file is in map_analysis_2d/test_import_fix.py
        # So we need to get to map_analysis_2d directory
        map_analysis_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Also try the parent directory in case we're running from RamanLab root
        parent_map_analysis_dir = os.path.join(os.path.dirname(map_analysis_dir), 'map_analysis_2d')
        
        print(f"Map analysis directory: {map_analysis_dir}")
        print(f"Parent map analysis directory: {parent_map_analysis_dir}")
        
        # Add both potential paths
        for path in [map_analysis_dir, parent_map_analysis_dir]:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
                print(f"Added to Python path: {path}")
        
        print()
        print("Testing imports...")
        
        # Test the imports
        from integrate_quantitative_analysis import QuantitativeAnalysisIntegrator
        print("✅ Successfully imported QuantitativeAnalysisIntegrator")
        
        # Test robust import logic for analysis modules
        try:
            from analysis.quantitative_analysis import RobustQuantitativeAnalyzer
        except ImportError:
            from map_analysis_2d.analysis.quantitative_analysis import RobustQuantitativeAnalyzer
        print("✅ Successfully imported RobustQuantitativeAnalyzer")
        
        try:
            from analysis.ml_class_flip_detector import MLClassFlipDetector
        except ImportError:
            from map_analysis_2d.analysis.ml_class_flip_detector import MLClassFlipDetector
        print("✅ Successfully imported MLClassFlipDetector")
        
        # Test creating instances
        print()
        print("Testing object creation...")
        
        integrator = QuantitativeAnalysisIntegrator()
        print("✅ Successfully created QuantitativeAnalysisIntegrator instance")
        
        analyzer = RobustQuantitativeAnalyzer()
        print("✅ Successfully created RobustQuantitativeAnalyzer instance")
        
        detector = MLClassFlipDetector()
        print("✅ Successfully created MLClassFlipDetector instance")
        
        print()
        print("🎉 ALL TESTS PASSED!")
        print("The quantitative analysis import fix is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quantitative_analysis_import()
    sys.exit(0 if success else 1) 