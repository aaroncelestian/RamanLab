#!/usr/bin/env python3
"""
Quick script to calculate polypropylene statistics from template fitting results.
This bypasses the hybrid analysis issues and gives you direct PP1 statistics.
"""

def calculate_pp1_statistics(main_window):
    """Calculate and print polypropylene statistics directly."""
    
    if not hasattr(main_window, 'template_fitting_results') or not main_window.template_fitting_results:
        print("❌ No template fitting results found. Run template fitting first.")
        return None
        
    # Get template data
    template_names = main_window.template_fitting_results['template_names']
    template_coefficients = main_window.template_fitting_results['coefficients']
    r_squared_values = main_window.template_fitting_results['r_squared']
    
    print(f"📊 Template Fitting Results Analysis")
    print(f"Templates: {template_names}")
    print(f"Total spectra fitted: {len(template_coefficients)}")
    
    # Find polypropylene template (flexible detection)
    pp_idx = None
    for i, name in enumerate(template_names):
        name_lower = str(name).lower()
        if any(hint in name_lower for hint in ['pp1', 'pp', 'polyprop', 'plastic', 'propyl']):
            pp_idx = i
            print(f"✅ Auto-detected polypropylene template: '{name}' at index {i}")
            break
    
    if pp_idx is None:
        print("❌ Could not auto-detect polypropylene template")
        print(f"Available templates: {template_names}")
        print("Please specify which template index represents polypropylene:")
        for i, name in enumerate(template_names):
            print(f"  {i}: {name}")
        return None
        
    
    # Calculate statistics
    total_spectra = len(template_coefficients)
    good_fits = 0
    pp_detections = 0
    high_confidence = 0
    medium_confidence = 0
    
    pp_contributions = []
    pp_relatives = []
    
    for pos_key, coeffs in template_coefficients.items():
        r_squared = r_squared_values.get(pos_key, 0)
        
        # Only count good fits
        if r_squared >= 0.5:
            good_fits += 1
            
            # Get polypropylene contribution
            pp_contrib = coeffs[pp_idx] if pp_idx < len(coeffs) else 0
            total_contrib = sum(coeffs[:len(template_names)])
            pp_relative = (pp_contrib / total_contrib) if total_contrib > 1e-10 else 0
            
            pp_contributions.append(pp_contrib)
            pp_relatives.append(pp_relative)
            
            # Count detections
            if pp_relative >= 0.2:  # 20% threshold
                pp_detections += 1
                
                if pp_relative >= 0.5:  # 50% threshold
                    high_confidence += 1
                elif pp_relative >= 0.2:
                    medium_confidence += 1
    
    # Print results
    print(f"\n🧬 POLYPROPYLENE DETECTION RESULTS:")
    print(f"=" * 50)
    print(f"Good fit spectra: {good_fits:,} / {total_spectra:,} ({(good_fits/total_spectra)*100:.1f}%)")
    print(f"PP detections (>20%): {pp_detections:,} ({(pp_detections/total_spectra)*100:.3f}%)")
    print(f"High confidence (>50%): {high_confidence:,} ({(high_confidence/total_spectra)*100:.3f}%)")
    print(f"Medium confidence (20-50%): {medium_confidence:,} ({(medium_confidence/total_spectra)*100:.3f}%)")
    
    if pp_contributions:
        import numpy as np
        pp_contributions = np.array(pp_contributions)
        pp_relatives = np.array(pp_relatives)
        
        print(f"\n📈 CONTRIBUTION STATISTICS:")
        print(f"Mean absolute contribution: {np.mean(pp_contributions):.6f}")
        print(f"Max absolute contribution: {np.max(pp_contributions):.6f}")
        print(f"Mean relative contribution: {np.mean(pp_relatives):.3f}")
        print(f"Max relative contribution: {np.max(pp_relatives):.3f}")
        
        print(f"\n✅ SUMMARY:")
        detection_rate = (pp_detections/total_spectra)*100
        if detection_rate > 1.0:
            print(f"⚠️  Significant polypropylene contamination: {detection_rate:.3f}%")
        elif detection_rate > 0.1:
            print(f"📊 Moderate polypropylene presence: {detection_rate:.3f}%")
        else:
            print(f"🔍 Trace polypropylene detected: {detection_rate:.3f}%")
            
        print(f"This is based on pure template fitting (no NMF interference)")
        
        return {
            'total_spectra': total_spectra,
            'good_fits': good_fits,
            'pp_detections': pp_detections,
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'detection_percentage': detection_rate,
            'mean_contribution': np.mean(pp_contributions),
            'max_contribution': np.max(pp_contributions),
            'mean_relative': np.mean(pp_relatives),
            'max_relative': np.max(pp_relatives)
        }
    
    return None

def calculate_pp_statistics_manual(main_window, template_index):
    """Calculate polypropylene statistics for a specific template index."""
    
    if not hasattr(main_window, 'template_fitting_results') or not main_window.template_fitting_results:
        print("❌ No template fitting results found. Run template fitting first.")
        return None
        
    # Get template data
    template_names = main_window.template_fitting_results['template_names']
    template_coefficients = main_window.template_fitting_results['coefficients']
    r_squared_values = main_window.template_fitting_results['r_squared']
    
    if template_index >= len(template_names) or template_index < 0:
        print(f"❌ Invalid template index {template_index}. Available indices: 0-{len(template_names)-1}")
        return None
        
    pp_template_name = template_names[template_index]
    print(f"📊 Analyzing template {template_index}: '{pp_template_name}' as polypropylene")
    print(f"Total spectra fitted: {len(template_coefficients)}")
    
    # Calculate statistics (same logic as main function)
    total_spectra = len(template_coefficients)
    good_fits = 0
    pp_detections = 0
    high_confidence = 0
    medium_confidence = 0
    
    pp_contributions = []
    pp_relatives = []
    
    for pos_key, coeffs in template_coefficients.items():
        r_squared = r_squared_values.get(pos_key, 0)
        
        if r_squared >= 0.5:
            good_fits += 1
            
            pp_contrib = coeffs[template_index] if template_index < len(coeffs) else 0
            total_contrib = sum(coeffs[:len(template_names)])
            pp_relative = (pp_contrib / total_contrib) if total_contrib > 1e-10 else 0
            
            pp_contributions.append(pp_contrib)
            pp_relatives.append(pp_relative)
            
            if pp_relative >= 0.2:
                pp_detections += 1
                
                if pp_relative >= 0.5:
                    high_confidence += 1
                elif pp_relative >= 0.2:
                    medium_confidence += 1
    
    # Print results
    print(f"\n🧬 POLYPROPYLENE DETECTION RESULTS:")
    print(f"=" * 50)
    print(f"Good fit spectra: {good_fits:,} / {total_spectra:,} ({(good_fits/total_spectra)*100:.1f}%)")
    print(f"PP detections (>20%): {pp_detections:,} ({(pp_detections/total_spectra)*100:.3f}%)")
    print(f"High confidence (>50%): {high_confidence:,} ({(high_confidence/total_spectra)*100:.3f}%)")
    print(f"Medium confidence (20-50%): {medium_confidence:,} ({(medium_confidence/total_spectra)*100:.3f}%)")
    
    if pp_contributions:
        import numpy as np
        pp_contributions = np.array(pp_contributions)
        pp_relatives = np.array(pp_relatives)
        
        print(f"\n📈 CONTRIBUTION STATISTICS:")
        print(f"Mean absolute contribution: {np.mean(pp_contributions):.6f}")
        print(f"Max absolute contribution: {np.max(pp_contributions):.6f}")
        print(f"Mean relative contribution: {np.mean(pp_relatives):.3f}")
        print(f"Max relative contribution: {np.max(pp_relatives):.3f}")
        
        print(f"\n✅ SUMMARY:")
        detection_rate = (pp_detections/total_spectra)*100
        if detection_rate > 1.0:
            print(f"⚠️  Significant polypropylene contamination: {detection_rate:.3f}%")
        elif detection_rate > 0.1:
            print(f"📊 Moderate polypropylene presence: {detection_rate:.3f}%")
        else:
            print(f"🔍 Trace polypropylene detected: {detection_rate:.3f}%")
            
        print(f"This is based on pure template fitting (no NMF interference)")
        
        return {
            'template_name': pp_template_name,
            'template_index': template_index,
            'total_spectra': total_spectra,
            'good_fits': good_fits,
            'pp_detections': pp_detections,
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'detection_percentage': detection_rate,
            'mean_contribution': np.mean(pp_contributions),
            'max_contribution': np.max(pp_contributions),
            'mean_relative': np.mean(pp_relatives),
            'max_relative': np.max(pp_relatives)
        }
    
    return None

if __name__ == "__main__":
    print("This script should be run from within the main application")
    print("Use: calculate_pp1_statistics(main_window)")
    print("Or: calculate_pp_statistics_manual(main_window, template_index)") 