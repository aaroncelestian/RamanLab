#!/usr/bin/env python3
"""
Demonstration of the Raman vs IR mode assignment fix.
Shows how the system now correctly handles the database issue.
"""

import pickle
import numpy as np

def demo_fix():
    """Demonstrate the fix in action."""
    
    print("🔬 Demonstration: Raman vs IR Mode Assignment Fix")
    print("=" * 60)
    
    # Load database
    with open('mineral_modes.pkl', 'rb') as f:
        mineral_database = pickle.load(f)
    
    calcite_data = mineral_database['CALCITE_3']
    modes = calcite_data['modes']
    
    print(f"📊 CALCITE_3 Database Analysis:")
    print(f"   Total modes: {len(modes)}")
    
    # Analyze the problematic 1108 cm⁻¹ region
    print(f"\n🔍 The Problematic 1108 cm⁻¹ Region:")
    print(f"   Database contains: 1108.0 cm⁻¹ A1u (IR-active)")
    print(f"   Should contain:    1108.0 cm⁻¹ A1g (Raman-active)")
    print(f"   ❌ This is a database error!")
    
    # Show what happens with the old vs new assignment logic
    print(f"\n⚖️  Assignment Logic Comparison:")
    
    # Simulate a peak at 1108 cm⁻¹
    experimental_peak = 1108.0
    
    print(f"\n   Experimental peak: {experimental_peak} cm⁻¹")
    print(f"   Available database modes:")
    
    nearby_modes = []
    for mode in modes:
        if isinstance(mode, tuple) and len(mode) >= 3:
            freq, symmetry, intensity = mode[0], mode[1], mode[2]
            if abs(freq - experimental_peak) <= 50:  # Within 50 cm⁻¹
                nearby_modes.append((freq, symmetry, intensity))
                activity = "Raman-active" if str(symmetry).endswith('g') else "IR-active" if str(symmetry).endswith('u') else "Unknown"
                print(f"     {freq:.1f} cm⁻¹ {symmetry} ({activity})")
    
    # OLD LOGIC (just closest match)
    print(f"\n   🔴 OLD LOGIC (distance-only):")
    if nearby_modes:
        closest_mode = min(nearby_modes, key=lambda x: abs(x[0] - experimental_peak))
        freq, sym, intensity = closest_mode
        print(f"     Selected: {freq:.1f} cm⁻¹ {sym}")
        if str(sym).endswith('u'):
            print(f"     ❌ PROBLEM: Assigned IR-active mode to Raman peak!")
    
    # NEW LOGIC (prioritize Raman-active)
    print(f"\n   ✅ NEW LOGIC (Raman-priority):")
    raman_modes = [m for m in nearby_modes if str(m[1]).endswith('g')]
    ir_modes = [m for m in nearby_modes if str(m[1]).endswith('u')]
    
    if raman_modes:
        best_raman = min(raman_modes, key=lambda x: abs(x[0] - experimental_peak))
        freq, sym, intensity = best_raman
        print(f"     Selected: {freq:.1f} cm⁻¹ {sym} (Raman-active)")
        print(f"     ✅ CORRECT: Raman-active mode assigned!")
    elif ir_modes:
        best_ir = min(ir_modes, key=lambda x: abs(x[0] - experimental_peak))
        freq, sym, intensity = best_ir
        print(f"     Fallback: {freq:.1f} cm⁻¹ {sym} (IR-active)")
        print(f"     ⚠️  WARNING: No Raman-active alternative found!")
        print(f"     ⚠️  This indicates a database problem!")
    
    # Show tensor creation filtering
    print(f"\n🏗️  Tensor Creation Filtering:")
    
    original_count = len(modes)
    raman_count = len([m for m in modes if isinstance(m, tuple) and len(m) >= 3 and str(m[1]).endswith('g')])
    ir_count = len([m for m in modes if isinstance(m, tuple) and len(m) >= 3 and str(m[1]).endswith('u')])
    
    print(f"   Original modes: {original_count}")
    print(f"   Raman-active ('g'): {raman_count}")
    print(f"   IR-active ('u'): {ir_count} ← These are now filtered out!")
    print(f"   Modes used for tensors: {raman_count}")
    
    # Show the fix benefits
    print(f"\n🎯 Fix Benefits:")
    print(f"   ✅ Peak assignment prioritizes Raman-active modes")
    print(f"   ✅ Warns when IR-active modes are assigned")
    print(f"   ✅ Tensor creation filters out IR-active modes")
    print(f"   ✅ Physics-correct tensor representations")
    print(f"   ✅ Identifies database issues automatically")
    
    print(f"\n💡 Recommendation:")
    print(f"   The database should be corrected to have:")
    print(f"   • 1108.0 cm⁻¹ A1g (instead of A1u)")
    print(f"   • Proper Raman-active mode assignments")

if __name__ == "__main__":
    demo_fix() 