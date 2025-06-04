#!/usr/bin/env python3

import tkinter as tk
import raman_polarization_analyzer as rpa
import numpy as np

def debug_user_data():
    """Debug the user's actual data to understand the structure."""
    print("=== Debugging User's Actual Data ===")
    
    # Create application instance
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    try:
        app = rpa.RamanPolarizationAnalyzer(root)
        
        print("\n1. CHECKING SPECTRUM DATA:")
        print(f"   current_spectrum: {hasattr(app, 'current_spectrum') and app.current_spectrum is not None}")
        if hasattr(app, 'current_spectrum') and app.current_spectrum:
            print(f"   Type: {type(app.current_spectrum)}")
            if isinstance(app.current_spectrum, dict):
                print(f"   Keys: {list(app.current_spectrum.keys())}")
        
        print(f"   wavenumbers: {hasattr(app, 'wavenumbers') and app.wavenumbers is not None}")
        if hasattr(app, 'wavenumbers') and app.wavenumbers is not None:
            print(f"   Length: {len(app.wavenumbers)}")
            print(f"   Range: {np.min(app.wavenumbers):.1f} - {np.max(app.wavenumbers):.1f} cm⁻¹")
        
        print(f"   intensities: {hasattr(app, 'intensities') and app.intensities is not None}")
        if hasattr(app, 'intensities') and app.intensities is not None:
            print(f"   Length: {len(app.intensities)}")
            print(f"   Range: {np.min(app.intensities):.3f} - {np.max(app.intensities):.3f}")
        
        print("\n2. DETAILED FITTED PEAKS ANALYSIS:")
        print(f"   fitted_peaks exists: {hasattr(app, 'fitted_peaks')}")
        if hasattr(app, 'fitted_peaks'):
            print(f"   fitted_peaks is not None: {app.fitted_peaks is not None}")
            print(f"   fitted_peaks bool: {bool(app.fitted_peaks)}")
            print(f"   fitted_peaks type: {type(app.fitted_peaks)}")
            
            if app.fitted_peaks:
                print(f"   Number of fitted peaks: {len(app.fitted_peaks)}")
                
                # Show first few peaks in detail
                for i, peak in enumerate(app.fitted_peaks[:3]):  # Show first 3 peaks
                    print(f"\n   Peak {i+1}:")
                    print(f"     Type: {type(peak)}")
                    if hasattr(peak, '__dict__'):
                        print(f"     Attributes: {list(peak.__dict__.keys())}")
                        for attr, value in peak.__dict__.items():
                            print(f"       {attr}: {value}")
                    elif isinstance(peak, dict):
                        print(f"     Keys: {list(peak.keys())}")
                        for key, value in peak.items():
                            print(f"       {key}: {value}")
                    else:
                        print(f"     Value: {peak}")
                        
                    if i >= 2:  # Only show first 3
                        break
        
        print("\n3. CHECKING TENSOR DATA:")
        print(f"   raman_tensors: {hasattr(app, 'raman_tensors') and bool(app.raman_tensors)}")
        if hasattr(app, 'raman_tensors'):
            print(f"   raman_tensors type: {type(app.raman_tensors)}")
            print(f"   raman_tensors content: {app.raman_tensors}")
        
        print(f"   tensor_data_3d: {hasattr(app, 'tensor_data_3d') and app.tensor_data_3d is not None}")
        if hasattr(app, 'tensor_data_3d'):
            print(f"   tensor_data_3d type: {type(app.tensor_data_3d)}")
            print(f"   tensor_data_3d content: {app.tensor_data_3d}")
        
        print("\n4. CHECKING MINERAL/DATABASE DATA:")
        print(f"   mineral_data: {hasattr(app, 'mineral_data') and app.mineral_data is not None}")
        if hasattr(app, 'mineral_data'):
            print(f"   mineral_data type: {type(app.mineral_data)}")
            print(f"   mineral_data content: {app.mineral_data}")
        
        print("\n5. CHECKING CRYSTAL STRUCTURE:")
        print(f"   structure_data: {hasattr(app, 'structure_data') and app.structure_data is not None}")
        if hasattr(app, 'structure_data'):
            print(f"   structure_data type: {type(app.structure_data)}")
            print(f"   structure_data content: {app.structure_data}")
        
        print(f"   crystal_structure: {hasattr(app, 'crystal_structure') and app.crystal_structure is not None}")
        if hasattr(app, 'crystal_structure'):
            print(f"   crystal_structure type: {type(app.crystal_structure)}")
            print(f"   crystal_structure content: {app.crystal_structure}")
        
        print("\n6. CHECKING OPTIMIZATION DATA:")
        print(f"   optimized_orientation: {hasattr(app, 'optimized_orientation') and app.optimized_orientation is not None}")
        if hasattr(app, 'optimized_orientation'):
            print(f"   optimized_orientation: {app.optimized_orientation}")
            
        print(f"   optimization_results: {hasattr(app, 'optimization_results') and app.optimization_results is not None}")
        if hasattr(app, 'optimization_results'):
            print(f"   optimization_results: {app.optimization_results}")
        
        print("\n=== CREATING TENSOR DATA FROM FITTED PEAKS ===")
        
        # Try to create tensor data from fitted peaks
        if hasattr(app, 'fitted_peaks') and app.fitted_peaks:
            try:
                print("Attempting to convert fitted peaks to tensor data...")
                
                # Create basic tensor data structure
                tensor_data = {}
                
                for i, peak in enumerate(app.fitted_peaks):
                    # Extract peak information
                    if hasattr(peak, 'center'):
                        wavenumber = peak.center
                    elif hasattr(peak, 'position'):
                        wavenumber = peak.position
                    elif isinstance(peak, dict) and 'center' in peak:
                        wavenumber = peak['center']
                    elif isinstance(peak, dict) and 'position' in peak:
                        wavenumber = peak['position']
                    else:
                        print(f"   Could not extract wavenumber from peak {i}: {peak}")
                        continue
                    
                    # Extract intensity
                    if hasattr(peak, 'amplitude'):
                        intensity = peak.amplitude
                    elif hasattr(peak, 'height'):
                        intensity = peak.height
                    elif hasattr(peak, 'intensity'):
                        intensity = peak.intensity
                    elif isinstance(peak, dict) and 'amplitude' in peak:
                        intensity = peak['amplitude']
                    elif isinstance(peak, dict) and 'height' in peak:
                        intensity = peak['height']
                    elif isinstance(peak, dict) and 'intensity' in peak:
                        intensity = peak['intensity']
                    else:
                        intensity = 1.0  # Default intensity
                    
                    # Create a simple tensor (identity matrix scaled by intensity)
                    tensor = np.eye(3) * intensity
                    
                    tensor_data[f"{wavenumber:.1f}"] = {
                        'tensor': tensor,
                        'wavenumber': wavenumber,
                        'intensity': intensity
                    }
                    
                    print(f"   Created tensor for peak at {wavenumber:.1f} cm⁻¹ (I={intensity:.2f})")
                
                print(f"\nCreated tensor data for {len(tensor_data)} peaks")
                print(f"Peak wavenumbers: {list(tensor_data.keys())}")
                
                # Save this data to the app
                app.raman_tensors = tensor_data
                app.tensor_data_3d = tensor_data
                
                print("✅ Successfully created tensor data from fitted peaks!")
                
            except Exception as e:
                print(f"❌ Error creating tensor data: {e}")
                import traceback
                traceback.print_exc()
        
    finally:
        root.destroy()

if __name__ == "__main__":
    debug_user_data() 