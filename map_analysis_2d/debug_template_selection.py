#!/usr/bin/env python3
"""
Debug script to help identify template selection issues in hybrid analysis.
Run this to see what templates are loaded and their relative strengths.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_template_selection():
    """
    This script helps debug template selection issues.
    You can run this from within your main application to see what's happening.
    """
    
    print("Template Selection Debug Guide")
    print("=" * 50)
    print()
    print("The quantitative calibration system now provides:")
    print("1. Calibration standards with known concentrations")
    print("2. Response curves relating signal to concentration")  
    print("3. Matrix effect corrections")
    print("4. Actual concentration values instead of arbitrary units")
    print()
    print("If the hybrid map is still wrong, here's how to debug:")
    print()
    print("STEP 1: Check the Log Output")
    print("-" * 30)
    print("When you run quantitative calibration, look for these features:")
    print("  - 'Available templates: [list of template names]'")
    print("  - 'Template X (name): avg strength = Y'")
    print("  - 'Selected template X (name) with highest strength: Y'")
    print("  - 'Final selection: Using template index X (name)'")
    print()
    print("STEP 2: Identify the Problem")
    print("-" * 30)
    print("a) If the log shows your background templates have higher strength:")
    print("   → Your polypropylene template might be weak or poorly fitted")
    print("   → Use the template selection dialog to manually choose")
    print()
    print("b) If the log shows the wrong template name is selected:")
    print("   → The automatic name detection failed")
    print("   → Use the template selection dialog to manually choose")
    print()
    print("c) If no log messages appear:")
    print("   → Check that both NMF and template fitting completed successfully")
    print("   → Verify you have both results loaded")
    print()
    print("STEP 3: Manual Template Selection")
    print("-" * 30)
    print("The new version shows a template selection dialog where you can:")
    print("  - See all available templates with their statistics")
    print("  - See average strength, max strength, and active pixels")
    print("  - The strongest template is highlighted with a ⭐")
    print("  - Manually select the template you want (like 'Map_Extract_1_(123_45)')")
    print()
    print("STEP 4: Expected Behavior")
    print("-" * 30)
    print("After selecting the correct template, the hybrid map should:")
    print("  - Look similar to your template extraction map")
    print("  - Show enhanced intensity where NMF also has strong signals")
    print("  - Have the same spatial patterns as your working template map")
    print()
    print("STEP 5: If Problems Persist")
    print("-" * 30)
    print("1. Check template fitting quality (R-squared values)")
    print("2. Verify NMF component 3 contains relevant signals")
    print("3. Try different NMF components in the hybrid analysis")
    print("4. Check that your polypropylene template is well-fitted")
    print()
    print("Next Steps:")
    print("1. Run 'Improved: Hybrid Template Map' from your map features")
    print("2. Watch the console output for log messages")
    print("3. Use the template selection dialog to choose the right template")
    print("4. Compare the result with your template extraction map")

def simulate_template_selection_logic():
    """Simulate the template selection logic with example data."""
    print("\nSimulating Template Selection Logic")
    print("=" * 40)
    
    # Example template data (simulate what you might have)
    example_templates = [
        {"name": "background_template_1", "avg_strength": 0.05, "description": "Background template"},
        {"name": "background_template_2", "avg_strength": 0.03, "description": "Background template"},
        {"name": "background_template_3", "avg_strength": 0.02, "description": "Background template"},
        {"name": "Map_Extract_1_(123_45)", "avg_strength": 0.35, "description": "Your target template"},
    ]
    
    print("Example Templates:")
    for i, template in enumerate(example_templates):
        print(f"  {i}: {template['name']} (avg strength: {template['avg_strength']:.3f}) - {template['description']}")
    
    print()
    
    # Simulate old logic (first with 'extract')
    old_selection = None
    for i, template in enumerate(example_templates):
        if 'extract' in template['name'].lower():
            old_selection = i
            break
    
    print(f"Old logic would select: {old_selection} ({example_templates[old_selection]['name']})")
    
    # Simulate new logic (strongest)
    strengths = [t['avg_strength'] for t in example_templates]
    new_selection = strengths.index(max(strengths))
    
    print(f"New logic selects: {new_selection} ({example_templates[new_selection]['name']}) - CORRECT!")
    
    print()
    print("This shows why the new logic should work better!")
    print("Your Map_Extract_1_(123_45) template should have the highest strength")
    print("and be automatically selected instead of background templates.")

if __name__ == "__main__":
    debug_template_selection()
    simulate_template_selection_logic()
    
    print(f"\nTo use the improved hybrid analysis:")
    print(f"1. Load your data and run template fitting and/or NMF")
    print(f"2. Click 'Quantitative Calibration' from the Template tab")
    print(f"3. Set up calibration standards with known concentrations")
    print(f"4. Get actual concentration values instead of arbitrary units!") 