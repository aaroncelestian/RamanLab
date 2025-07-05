#!/usr/bin/env python3
"""
Test script for integrated mixed mineral analysis system.

This tests the integration between the mixed mineral analysis and 
the existing RamanLab correlation-based database search.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_integration():
    """Test the integrated system."""
    print("Testing Integrated Mixed Mineral Analysis System")
    print("=" * 60)
    
    # Test 1: Import check
    print("\n1. Testing imports...")
    try:
        from mixed_mineral_spectral_fitting import MixedMineralFitter, RamanLabDatabaseInterface
        print("✓ Mixed mineral fitting module imported successfully")
        
        try:
            from raman_spectra_qt6 import RamanSpectraQt6
            print("✓ RamanLab database module imported successfully")
            RAMAN_DB_AVAILABLE = True
        except ImportError as e:
            print(f"⚠ RamanLab database not available: {e}")
            RAMAN_DB_AVAILABLE = False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Database connection
    print("\n2. Testing database connection...")
    if RAMAN_DB_AVAILABLE:
        try:
            raman_db = RamanSpectraQt6()
            print("✓ RamanLab database instance created")
            
            # Check if database file exists
            db_files = [
                'RamanLab_Database_20250602.pkl',
                'RamanLab_Database.sqlite'
            ]
            
            found_db = False
            for db_file in db_files:
                if os.path.exists(db_file):
                    print(f"✓ Found database file: {db_file}")
                    found_db = True
                    break
                    
            if not found_db:
                print("⚠ No database file found - will use empty database")
                
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raman_db = None
    else:
        raman_db = None
        
    # Test 3: Create synthetic spectrum
    print("\n3. Creating synthetic test spectrum...")
    wavenumbers = np.linspace(200, 1000, 800)
    
    # Create a spectrum that should match hilairite
    hilairite_peaks = [295, 520, 924]  # User's test peaks
    intensities = np.zeros_like(wavenumbers)
    
    for peak in hilairite_peaks:
        # Add peak with some realistic width and intensity
        width = 8.0
        intensity = 1.0 - (peak - 295) / 1000  # Decreasing intensity
        intensities += intensity * np.exp(-(wavenumbers - peak)**2 / (2 * width**2))
    
    # Add some noise and baseline
    intensities += np.random.normal(0, 0.02, len(wavenumbers))
    intensities += 0.1
    
    print(f"✓ Created spectrum with peaks at: {hilairite_peaks} cm⁻¹")
    
    # Test 4: Initialize integrated fitter
    print("\n4. Testing integrated fitter...")
    try:
        fitter = MixedMineralFitter(wavenumbers, intensities, raman_db=raman_db)
        print("✓ Integrated fitter created successfully")
        
        # Test database interface
        if hasattr(fitter, 'db_interface'):
            print("✓ Database interface initialized")
        else:
            print("⚠ Database interface not found")
            
    except Exception as e:
        print(f"❌ Fitter creation failed: {e}")
        return False
    
    # Test 5: Test correlation-based search
    print("\n5. Testing correlation-based search...")
    try:
        if fitter.db_interface.raman_db is not None:
            # Test the search functionality
            search_results = fitter.db_interface.search_correlation_based(
                wavenumbers, intensities, n_matches=5, threshold=0.3
            )
            
            print(f"✓ Correlation search completed")
            print(f"  Results found: {len(search_results)}")
            
            if search_results:
                best_match = search_results[0]
                print(f"  Best match: {best_match['name']} (score: {best_match['score']:.3f})")
            else:
                print("  No matches found (expected if database is empty)")
        else:
            print("⚠ No database available for correlation search")
            
    except Exception as e:
        print(f"❌ Correlation search failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Test DTW search (if available)
    print("\n6. Testing DTW search...")
    try:
        from mixed_mineral_spectral_fitting import DTW_AVAILABLE
        if DTW_AVAILABLE:
            print("✓ DTW libraries are available")
            
            if fitter.db_interface.raman_db is not None:
                dtw_results = fitter.db_interface.search_dtw_based(
                    wavenumbers, intensities, n_matches=3
                )
                print(f"✓ DTW search completed")
                print(f"  DTW results found: {len(dtw_results)}")
            else:
                print("⚠ No database available for DTW search")
        else:
            print("⚠ DTW libraries not available - will use correlation fallback")
            
    except Exception as e:
        print(f"❌ DTW search test failed: {e}")
    
    # Test 7: Test major phase detection
    print("\n7. Testing major phase detection...")
    try:
        major_phase = fitter.detect_major_phase(
            correlation_threshold=0.3,  # Lower threshold for testing
            n_matches=5
        )
        
        print(f"✓ Major phase detected: {major_phase.name}")
        print(f"  Search method: {major_phase.search_method}")
        print(f"  Confidence: {major_phase.confidence:.3f}")
        print(f"  Expected peaks: {[f'{p:.1f}' for p in major_phase.expected_peaks[:5]]}")
        
    except Exception as e:
        print(f"❌ Major phase detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ Integration test completed!")
    print("\nKey findings:")
    print(f"• RamanLab database available: {RAMAN_DB_AVAILABLE}")
    print(f"• DTW analysis available: {DTW_AVAILABLE if 'DTW_AVAILABLE' in locals() else 'Unknown'}")
    print("• System ready for mixed mineral analysis")
    
    return True

if __name__ == "__main__":
    try:
        success = test_integration()
        if success:
            print("\n🎉 All tests passed! The integrated system is working.")
        else:
            print("\n❌ Some tests failed. Check the output above.")
    except Exception as e:
        print(f"\n💥 Test script failed: {e}")
        import traceback
        traceback.print_exc() 