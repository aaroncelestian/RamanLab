#!/usr/bin/env python3

"""
Comprehensive Workflow Guide for Raman Polarization Analyzer 3D Visualization

This guide will help you get your actual data working with the 3D visualization.
"""

import tkinter as tk
from tkinter import messagebox
import raman_polarization_analyzer as rpa

def check_data_status():
    """Check what data is currently available."""
    print("=== CHECKING YOUR DATA STATUS ===")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = rpa.RamanPolarizationAnalyzer(root)
        
        # Check each data type
        has_spectrum = (hasattr(app, 'current_spectrum') and app.current_spectrum) or \
                      (hasattr(app, 'wavenumbers') and app.wavenumbers is not None)
        has_fitted_peaks = hasattr(app, 'fitted_peaks') and bool(app.fitted_peaks)
        has_tensors = hasattr(app, 'raman_tensors') and bool(app.raman_tensors)
        has_structure = hasattr(app, 'crystal_structure') and app.crystal_structure
        has_optimization = hasattr(app, 'optimized_orientation') and app.optimized_orientation
        
        print("\n📊 DATA AVAILABILITY:")
        print(f"  Spectrum Data:      {'✅ Available' if has_spectrum else '❌ Missing'}")
        print(f"  Fitted Peaks:       {'✅ Available' if has_fitted_peaks else '❌ Missing'}")
        print(f"  Tensor Data:        {'✅ Available' if has_tensors else '❌ Missing'}")
        print(f"  Crystal Structure:  {'✅ Available' if has_structure else '❌ Missing'}")
        print(f"  Optimization:       {'✅ Available' if has_optimization else '❌ Missing'}")
        
        if has_fitted_peaks:
            print(f"  Number of fitted peaks: {len(app.fitted_peaks)}")
        
        return has_spectrum, has_fitted_peaks, has_tensors, has_structure, has_optimization
        
    finally:
        root.destroy()

def show_workflow_steps():
    """Show the complete workflow to get 3D visualization working."""
    
    has_spectrum, has_fitted_peaks, has_tensors, has_structure, has_optimization = check_data_status()
    
    print("\n🚀 WORKFLOW TO GET 3D VISUALIZATION WORKING:")
    print("=" * 60)
    
    if not has_spectrum:
        print("\n📈 STEP 1: Load Your Spectrum Data")
        print("   • Go to the 'Spectrum Analysis' tab")
        print("   • Click 'Load Spectrum' button")
        print("   • Select your .txt, .csv, or .spc file")
        print("   • Make sure it has wavenumber and intensity columns")
        print("   ⚠️  This step is REQUIRED for everything else!")
    else:
        print("\n✅ STEP 1: Spectrum data already loaded")
    
    if not has_fitted_peaks:
        print("\n🔍 STEP 2: Fit Peaks in Your Spectrum")
        print("   • Go to the 'Peak Fitting' tab")
        print("   • Use 'Auto Peak Detection' or manually add peaks")
        print("   • Adjust peak parameters (position, width, height)")
        print("   • Click 'Fit All Peaks' to get fitted parameters")
        print("   • You should see peaks listed in the results")
        print("   ⚠️  This creates the peak data needed for tensors!")
    else:
        print("\n✅ STEP 2: Fitted peaks already available")
    
    if not has_tensors:
        print("\n🔬 STEP 3: Create Tensor Data")
        print("   • Go to the '3D Visualization' tab")
        print("   • In the 'Data' sub-tab, click:")
        print("     - '📈 Create from Fitted Peaks' (uses your peak fitting results)")
        print("     - OR '🚀 Auto-Import All Available Data'")
        print("   • This creates Raman tensors from your fitted peaks")
        print("   ⚠️  This is what creates the red ellipsoids!")
    else:
        print("\n✅ STEP 3: Tensor data already available")
    
    print("\n🎯 STEP 4: Enable 3D Visualization")
    print("   • Go to the '3D Visualization' tab")
    print("   • Click '✨ Quick Setup (Demo + Enable All)' for instant results")
    print("   • OR manually enable in 'Display' sub-tab:")
    print("     - ☑️ Crystal Shape")
    print("     - ☑️ Raman Tensor Ellipsoid")
    print("     - ☑️ Laser Geometry")
    print("     - ☑️ Coordinate Axes")
    
    print("\n🎮 STEP 5: Interact with Your Data")
    print("   • Use the 'Orientation' sub-tab sliders to rotate crystal")
    print("   • Select different peaks in the dropdown")
    print("   • Watch the spectrum update in real-time")
    print("   • Use animation controls for automatic rotation")
    
    print("\n" + "=" * 60)
    print("💡 QUICK SOLUTIONS:")
    
    if not has_spectrum and not has_fitted_peaks:
        print("\n🚨 NO DATA DETECTED - Try these options:")
        print("   Option A: Use Demo Data")
        print("   • Go to 3D Visualization → Data tab")
        print("   • Click '✨ Quick Setup (Demo + Enable All)'")
        print("   • This loads example data to test the interface")
        print()
        print("   Option B: Load Your Own Data")
        print("   • Load spectrum file in Spectrum Analysis tab")
        print("   • Fit peaks in Peak Fitting tab")
        print("   • Then use 3D Visualization")
    
    elif has_spectrum and not has_fitted_peaks:
        print("\n📊 SPECTRUM LOADED - Need to fit peaks:")
        print("   • Go to Peak Fitting tab")
        print("   • Use Auto Peak Detection")
        print("   • Fit the peaks")
        print("   • Then return to 3D Visualization")
    
    elif has_fitted_peaks and not has_tensors:
        print("\n🔍 PEAKS FITTED - Need to create tensors:")
        print("   • Go to 3D Visualization → Data tab")
        print("   • Click '📈 Create from Fitted Peaks'")
        print("   • This will create tensor ellipsoids from your peaks")
    
    print("\n🆘 TROUBLESHOOTING:")
    print("   • If ellipsoids don't appear: Check tensor scale in Display tab")
    print("   • If spectrum is empty: Make sure real-time calculation is enabled")
    print("   • If nothing works: Use Demo Data first to test interface")
    print("   • Check the Status panel for detailed information")

def create_test_workflow():
    """Create a test workflow with sample data."""
    print("\n🧪 CREATING TEST WORKFLOW...")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = rpa.RamanPolarizationAnalyzer(root)
        
        # Create sample spectrum data
        import numpy as np
        wavenumbers = np.linspace(200, 1200, 1000)
        intensities = np.zeros_like(wavenumbers)
        
        # Add some peaks
        peak_positions = [300, 500, 700, 900, 1100]
        for pos in peak_positions:
            peak = np.exp(-((wavenumbers - pos) / 30)**2)
            intensities += peak * np.random.uniform(0.5, 1.5)
        
        # Add noise
        intensities += np.random.normal(0, 0.05, len(intensities))
        intensities = np.maximum(intensities, 0)
        
        # Set spectrum data
        app.wavenumbers = wavenumbers
        app.intensities = intensities
        app.current_spectrum = {
            'wavenumbers': wavenumbers,
            'intensities': intensities,
            'name': 'Test Spectrum'
        }
        
        # Create fitted peaks
        app.fitted_peaks = []
        for i, pos in enumerate(peak_positions):
            # Create a simple peak object
            peak = type('Peak', (), {})()
            peak.center = pos
            peak.amplitude = np.random.uniform(0.5, 1.5)
            peak.sigma = 30.0
            app.fitted_peaks.append(peak)
        
        print(f"✅ Created test spectrum with {len(peak_positions)} peaks")
        print(f"   Peak positions: {peak_positions}")
        
        # Now create tensors
        success = app.create_tensors_from_fitted_peaks()
        if success:
            print("✅ Successfully created tensor data from test peaks")
            print("   Now you can use the 3D visualization!")
        else:
            print("❌ Failed to create tensor data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating test workflow: {e}")
        return False
    finally:
        root.destroy()

if __name__ == "__main__":
    print("🔬 RAMAN POLARIZATION ANALYZER - 3D VISUALIZATION WORKFLOW GUIDE")
    print("=" * 70)
    
    # Check current status
    show_workflow_steps()
    
    # Ask if user wants to create test data
    print("\n" + "=" * 70)
    response = input("\n❓ Would you like to create test data to try the workflow? (y/n): ")
    
    if response.lower().startswith('y'):
        success = create_test_workflow()
        if success:
            print("\n🎉 Test data created! Now:")
            print("   1. Run your main application")
            print("   2. Go to 3D Visualization tab")
            print("   3. Click 'Auto-Import All Available Data'")
            print("   4. You should see tensor ellipsoids!")
    
    print("\n📚 For more help, check the status panel in the 3D Visualization tab")
    print("💡 Remember: Demo Data button always works for testing!") 