#!/usr/bin/env python3

import tkinter as tk
import raman_polarization_analyzer as rpa
import numpy as np

print('🧪 CREATING TEST DATA FOR YOUR APPLICATION...')

root = tk.Tk()
root.withdraw()

try:
    app = rpa.RamanPolarizationAnalyzer(root)
    
    # Create sample spectrum data
    wavenumbers = np.linspace(200, 1200, 1000)
    intensities = np.zeros_like(wavenumbers)
    
    # Add some realistic peaks
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
        peak = type('Peak', (), {})()
        peak.center = pos
        peak.amplitude = np.random.uniform(0.5, 1.5)
        peak.sigma = 30.0
        app.fitted_peaks.append(peak)
    
    print(f'✅ Created test spectrum with {len(peak_positions)} peaks')
    print(f'   Peak positions: {peak_positions}')
    
    # Now create tensors
    success = app.create_tensors_from_fitted_peaks()
    if success:
        print('✅ Successfully created tensor data from test peaks')
        print('   Tensor data is now available for 3D visualization!')
    else:
        print('❌ Failed to create tensor data')
    
finally:
    root.destroy()

print('\n🎉 TEST DATA CREATED SUCCESSFULLY!')
print('\nNOW FOLLOW THESE STEPS:')
print('1. Run your main application: python3 raman_polarization_analyzer.py')
print('2. Go to the "3D Visualization" tab')
print('3. Click the "Data" sub-tab')
print('4. Click "🚀 Auto-Import All Available Data"')
print('5. You should see tensor ellipsoids appear!')
print('\nAlternatively, click "✨ Quick Setup (Demo + Enable All)" for instant results!') 