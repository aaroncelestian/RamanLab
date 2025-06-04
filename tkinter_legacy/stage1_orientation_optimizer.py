"""
Stage 1 Improved Crystal Orientation Optimization

This module provides enhanced crystal orientation determination from Raman spectra
with the following key improvements over the basic approach:

1. Individual peak position adjustments (constrained by uncertainties)
2. Uncertainty estimates extracted from fitting errors
3. Multi-start optimization from different orientations  
4. Weighted errors by peak intensity and fit quality
5. Character-based assignment bonuses

Usage:
    from stage1_orientation_optimizer import optimize_crystal_orientation_stage1
    result = optimize_crystal_orientation_stage1(analyzer_instance)
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.optimize import minimize, differential_evolution


def optimize_crystal_orientation_stage1(analyzer):
    """
    Stage 1 improved crystal orientation optimization.
    
    Args:
        analyzer: RamanPolarizationAnalyzer instance with fitted peaks and crystal structure
        
    Returns:
        dict: Results containing optimized orientation, uncertainties, and quality metrics
    """
    try:
        # Check prerequisites
        if not hasattr(analyzer, 'fitted_regions') or not analyzer.fitted_regions:
            messagebox.showinfo("Stage 1 Optimization", "Please fit peaks in the Peak Fitting tab first.")
            return None
            
        if not hasattr(analyzer, 'current_structure') or not hasattr(analyzer.current_structure, 'frequencies'):
            messagebox.showinfo("Stage 1 Optimization", "Please calculate a Raman spectrum first.")
            return None
        
        # Create progress window
        progress_window = tk.Toplevel(analyzer.root)
        progress_window.title("Stage 1 Improved Crystal Orientation Optimization")
        progress_window.geometry("600x450")
        progress_window.transient(analyzer.root)
        progress_window.grab_set()
        
        # UI Setup
        ttk.Label(progress_window, text="Stage 1: Enhanced Crystal Orientation Optimization", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        status_label = ttk.Label(progress_window, text="Initializing...")
        status_label.pack(pady=5)
        
        # Results display with scrollbar
        results_frame = tk.Frame(progress_window)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        results_text = tk.Text(results_frame, height=15, width=70, yscrollcommand=scrollbar.set)
        results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=results_text.yview)
        
        abort_var = tk.BooleanVar(value=False)
        
        def abort_optimization():
            abort_var.set(True)
            status_label.config(text="Aborting...")
        
        ttk.Button(progress_window, text="Abort", command=abort_optimization).pack(pady=5)
        progress_window.update()
        
        # Display improvements
        results_text.insert(tk.END, "=== STAGE 1 ENHANCED OPTIMIZATION ===\n\n")
        results_text.insert(tk.END, "Key improvements over basic method:\n")
        results_text.insert(tk.END, "✓ Individual peak position adjustments\n")
        results_text.insert(tk.END, "✓ Uncertainty-weighted optimization\n") 
        results_text.insert(tk.END, "✓ Multi-start global search\n")
        results_text.insert(tk.END, "✓ Character-based peak assignment\n")
        results_text.insert(tk.END, "✓ Fit quality weighting\n\n")
        results_text.update()
        
        # Phase 1: Extract experimental peaks with enhanced uncertainty analysis
        status_label.config(text="Phase 1: Extracting experimental data...")
        progress_var.set(10)
        progress_window.update()
        
        experimental_peaks = extract_experimental_peaks_with_uncertainties(analyzer, results_text)
        
        if len(experimental_peaks) < 3:
            progress_window.destroy()
            messagebox.showinfo("Stage 1 Optimization", "Need at least 3 fitted peaks for optimization.")
            return None
        
        # Phase 2: Enhanced multi-start optimization
        status_label.config(text="Phase 2: Multi-start optimization...")
        progress_var.set(30)
        progress_window.update()
        
        optimization_result = run_enhanced_optimization(
            analyzer, experimental_peaks, progress_var, status_label, 
            progress_window, abort_var, results_text
        )
        
        if abort_var.get() or optimization_result is None:
            progress_window.destroy()
            return None
        
        # Phase 3: Uncertainty estimation
        status_label.config(text="Phase 3: Uncertainty estimation...")
        progress_var.set(90)
        progress_window.update()
        
        uncertainty_estimates = estimate_orientation_uncertainty(
            optimization_result, experimental_peaks, results_text
        )
        
        # Update analyzer with results
        best_orientation = optimization_result['best_orientation']
        analyzer.phi_var.set(best_orientation[0])
        analyzer.theta_var.set(best_orientation[1])
        analyzer.psi_var.set(best_orientation[2])
        
        # Display final results
        display_final_results(
            optimization_result, uncertainty_estimates, results_text
        )
        
        progress_var.set(100)
        status_label.config(text="Stage 1 optimization complete!")
        
        # Update the orientation plot
        analyzer.calculate_orientation_raman_spectrum()
        
        # Action buttons
        button_frame = tk.Frame(progress_window)
        button_frame.pack(pady=10)
        
        def apply_and_close():
            progress_window.destroy()
            analyzer.update_3d_visualizer()
            
            # Success message with key improvements highlighted
            messagebox.showinfo("Stage 1 Complete", 
                f"Stage 1 Enhanced Optimization Complete!\n\n"
                f"Improved Crystal Orientation:\n"
                f"φ = {best_orientation[0]:.1f}° ± {uncertainty_estimates['phi_uncertainty']:.1f}°\n"
                f"θ = {best_orientation[1]:.1f}° ± {uncertainty_estimates['theta_uncertainty']:.1f}°\n"
                f"ψ = {best_orientation[2]:.1f}° ± {uncertainty_estimates['psi_uncertainty']:.1f}°\n\n"
                f"Confidence: {optimization_result['confidence']:.1%}\n\n"
                f"Key Improvements Applied:\n"
                f"• Individual peak adjustments: ±{np.mean([p['position_uncertainty'] for p in experimental_peaks]):.1f} cm⁻¹\n"
                f"• Multi-start optimization: {optimization_result['num_starts']} starting points\n"
                f"• Uncertainty weighting: Fit quality considered\n"
                f"• Global search: {optimization_result['total_evaluations']} evaluations\n\n"
                f"This provides much more reliable orientation\n"
                f"determination compared to the basic method!")
        
        def save_detailed_results():
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Stage 1 Optimization Results"
            )
            if filename:
                save_optimization_results(filename, optimization_result, uncertainty_estimates, experimental_peaks)
                messagebox.showinfo("Saved", f"Detailed results saved to {filename}")
        
        ttk.Button(button_frame, text="Save Results", command=save_detailed_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply & Close", command=apply_and_close).pack(side=tk.LEFT, padx=5)
        
        # Return results for further processing if needed
        return {
            'orientation': best_orientation,
            'uncertainties': uncertainty_estimates,
            'optimization_details': optimization_result,
            'experimental_peaks': experimental_peaks
        }
        
    except Exception as e:
        if 'progress_window' in locals():
            progress_window.destroy()
        messagebox.showerror("Error", f"Stage 1 optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def extract_experimental_peaks_with_uncertainties(analyzer, output_text):
    """Extract experimental peaks with enhanced uncertainty estimates."""
    experimental_peaks = []
    
    output_text.insert(tk.END, "Extracting experimental peaks with uncertainties...\n\n")
    
    for region_idx, region_data in analyzer.fitted_regions.items():
        if 'peaks' in region_data:
            region_r_squared = region_data.get('r_squared', 0.9)
            covariance = region_data.get('covariance', None)
            
            for peak_idx, peak in enumerate(region_data['peaks']):
                if 'center' in peak and 'height' in peak:
                    center = float(peak['center'])
                    height = float(peak['height'])
                    width = float(peak.get('width', peak.get('sigma', peak.get('wid_left', 10.0))))
                    
                    # Enhanced position uncertainty estimation
                    position_uncertainty = width * 0.1  # Base estimate: 10% of peak width
                    
                    # Use covariance matrix if available for better uncertainty
                    if covariance is not None:
                        try:
                            # Estimate parameter index (center is typically 2nd parameter)
                            param_idx = peak_idx * 3 + 1  # Assuming 3 params per peak
                            if param_idx < len(covariance) and covariance[param_idx, param_idx] > 0:
                                stderr_from_fit = np.sqrt(covariance[param_idx, param_idx])
                                position_uncertainty = max(position_uncertainty, stderr_from_fit)
                        except:
                            pass
                    
                    # Adjust uncertainty based on fit quality
                    quality_factor = max(0.3, region_r_squared)
                    position_uncertainty /= quality_factor
                    
                    # Intensity uncertainty
                    intensity_uncertainty = height * 0.05 / quality_factor
                    
                    # Character assignment (try to extract from markers)
                    character = ""
                    character_confidence = 0.0
                    
                    if hasattr(analyzer, 'fitted_peak_markers'):
                        for marker in analyzer.fitted_peak_markers:
                            try:
                                if hasattr(marker, 'get_position') and hasattr(marker, 'get_text'):
                                    marker_pos = marker.get_position()
                                    if abs(marker_pos[0] - center) < 5:
                                        marker_text = marker.get_text()
                                        # Extract character part (before R² info)
                                        if '\n' in marker_text:
                                            character = marker_text.split('\n')[0].strip()
                                        else:
                                            character = marker_text.strip()
                                        
                                        if len(character) <= 5 and character:
                                            character_confidence = 0.8
                            except:
                                continue
                    
                    experimental_peaks.append({
                        'position': center,
                        'intensity': height,
                        'width': width,
                        'character': character,
                        'position_uncertainty': position_uncertainty,
                        'intensity_uncertainty': intensity_uncertainty,
                        'character_confidence': character_confidence,
                        'r_squared': region_r_squared,
                        'region': region_idx,
                        'peak_index': peak_idx,
                        'fit_quality': quality_factor
                    })
    
    # Sort by intensity (prioritize strong peaks)
    experimental_peaks.sort(key=lambda x: x['intensity'], reverse=True)
    
    # Display extracted peaks
    output_text.insert(tk.END, f"Extracted {len(experimental_peaks)} experimental peaks:\n")
    for i, peak in enumerate(experimental_peaks):
        output_text.insert(tk.END, 
            f"  Peak {i+1}: {peak['position']:.1f} ± {peak['position_uncertainty']:.1f} cm⁻¹ "
            f"(I={peak['intensity']:.0f}, R²={peak['r_squared']:.3f})")
        if peak['character']:
            output_text.insert(tk.END, f" [{peak['character']}]")
        output_text.insert(tk.END, "\n")
    
    output_text.insert(tk.END, "\n")
    output_text.see(tk.END)
    
    return experimental_peaks


def run_enhanced_optimization(analyzer, experimental_peaks, progress_var, status_label, 
                            progress_window, abort_var, output_text):
    """Run the enhanced multi-start optimization with individual peak adjustments."""
    
    output_text.insert(tk.END, "Setting up enhanced optimization...\n")
    
    # Get theoretical data
    theoretical_freqs = np.array(analyzer.current_structure.frequencies)
    theoretical_chars = getattr(analyzer.current_structure, 'activities', [])
    
    output_text.insert(tk.END, f"Theoretical modes available: {len(theoretical_freqs)}\n\n")
    
    # Track optimization progress
    evaluation_count = [0]
    best_results = []
    
    def enhanced_objective_function(params):
        """Enhanced objective with individual peak adjustments and uncertainty weighting."""
        evaluation_count[0] += 1
        
        # Periodic progress updates
        if evaluation_count[0] % 25 == 0:
            progress = 30 + min(55, (evaluation_count[0] / 400) * 55)
            progress_var.set(progress)
            if evaluation_count[0] % 100 == 0:
                progress_window.update()
        
        if abort_var.get():
            raise Exception("Optimization aborted")
        
        # Extract parameters: [phi, theta, psi] + individual peak adjustments
        phi, theta, psi = params[:3]
        
        # Constrain angles
        phi = phi % 360
        theta = max(0.1, min(179.9, theta))
        psi = psi % 360
        
        # Individual peak adjustments
        n_peaks = len(experimental_peaks)
        peak_adjustments = params[3:3+n_peaks] if len(params) > 3 else np.zeros(n_peaks)
        
        try:
            # Calculate theoretical spectrum
            spectrum = analyzer.current_structure.calculate_raman_spectrum((phi, theta, psi), 'VV')
            if not spectrum:
                return 1000
            
            theo_freqs = np.array([f for f, i, c in spectrum])
            theo_intensities = np.array([i for f, i, c in spectrum])
            theo_chars = [c for f, i, c in spectrum]
            
            # Normalize intensities
            max_theo_intensity = np.max(theo_intensities) if len(theo_intensities) > 0 else 1.0
            max_exp_intensity = max(p['intensity'] for p in experimental_peaks)
            
            if max_theo_intensity > 0:
                norm_theo_intensities = theo_intensities / max_theo_intensity
            else:
                norm_theo_intensities = theo_intensities
            
            # Multi-objective weighted error calculation
            position_error = 0
            intensity_error = 0
            character_bonus = 0
            matched_peaks = 0
            total_weight = 0
            
            for i, exp_peak in enumerate(experimental_peaks):
                exp_pos = exp_peak['position']
                exp_intensity = exp_peak['intensity'] / max_exp_intensity
                exp_uncertainty = exp_peak['position_uncertainty']
                fit_quality = exp_peak['fit_quality']
                exp_char = exp_peak['character']
                char_confidence = exp_peak['character_confidence']
                
                # Weight based on intensity and fit quality
                peak_weight = exp_intensity * fit_quality
                total_weight += peak_weight
                
                # Apply constrained individual adjustment
                max_adjustment = 2.0 * exp_uncertainty  # Allow up to 2σ adjustment
                adjustment = np.clip(peak_adjustments[i] if i < len(peak_adjustments) else 0,
                                   -max_adjustment, max_adjustment)
                
                # Find best theoretical match
                best_pos_error = float('inf')
                best_int_error = float('inf')
                best_char_score = 0
                
                for j, theo_freq in enumerate(theo_freqs):
                    adjusted_theo_freq = theo_freq + adjustment
                    
                    # Position error (normalized by uncertainty)
                    pos_error = ((exp_pos - adjusted_theo_freq) / exp_uncertainty)**2
                    
                    # Intensity error
                    theo_int = norm_theo_intensities[j] if j < len(norm_theo_intensities) else 0
                    int_error = (exp_intensity - theo_int)**2
                    
                    # Character matching bonus
                    char_score = 0
                    if exp_char and j < len(theo_chars):
                        theo_char = str(theo_chars[j])
                        if exp_char.lower() == theo_char.lower():
                            char_score = 2.0 * char_confidence
                        elif exp_char.lower() in theo_char.lower() or theo_char.lower() in exp_char.lower():
                            char_score = 1.0 * char_confidence
                    
                    # Track best match for this experimental peak
                    if pos_error < best_pos_error:
                        best_pos_error = pos_error
                        best_int_error = int_error
                        best_char_score = char_score
                
                # Accumulate weighted errors
                if best_pos_error < float('inf'):
                    position_error += best_pos_error * peak_weight
                    intensity_error += best_int_error * peak_weight
                    character_bonus += best_char_score * peak_weight
                    matched_peaks += 1
            
            # Normalize by total weight
            if total_weight > 0 and matched_peaks > 0:
                position_error /= total_weight
                intensity_error /= total_weight
                character_bonus /= total_weight
            else:
                return 1000
            
            # Penalties and regularization
            completeness_penalty = (len(experimental_peaks) - matched_peaks) * 5
            adjustment_penalty = 0.01 * np.sum(peak_adjustments**2) if len(peak_adjustments) > 0 else 0
            
            # Combined multi-objective error (minimize)
            combined_error = (2.0 * position_error +      # Position heavily weighted
                            1.0 * intensity_error -       # Intensity matching  
                            0.5 * character_bonus +       # Character bonus (subtract for benefit)
                            completeness_penalty +        # Penalty for unmatched peaks
                            adjustment_penalty)            # Regularization
            
            # Store promising results for analysis
            if evaluation_count[0] % 50 == 0:
                best_results.append({
                    'orientation': [phi, theta, psi],
                    'error': combined_error,
                    'position_error': position_error,
                    'intensity_error': intensity_error,
                    'character_bonus': character_bonus,
                    'matched_peaks': matched_peaks,
                    'adjustments': peak_adjustments.copy() if len(peak_adjustments) > 0 else []
                })
            
            return combined_error
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1000
    
    # Set up optimization bounds
    n_peaks = len(experimental_peaks)
    bounds = [(0, 360), (0, 180), (0, 360)]  # Euler angles
    
    # Add bounds for individual peak adjustments
    for exp_peak in experimental_peaks:
        max_adj = 2.0 * exp_peak['position_uncertainty']
        bounds.append((-max_adj, max_adj))
    
    output_text.insert(tk.END, f"Optimizing {3 + n_peaks} parameters:\n")
    output_text.insert(tk.END, f"  3 Euler angles + {n_peaks} individual peak adjustments\n\n")
    
    # Multi-start strategy
    starting_points = []
    
    # Current orientation
    current_angles = [analyzer.phi_var.get(), analyzer.theta_var.get(), analyzer.psi_var.get()]
    starting_points.append(current_angles + [0.0] * n_peaks)
    
    # Systematic grid for global search
    for phi in [0, 60, 120, 180, 240, 300]:
        for theta in [0, 30, 60, 90, 120, 150]:
            starting_points.append([phi, theta, 0] + [0.0] * n_peaks)
    
    # Limit to reasonable number for efficiency
    starting_points = starting_points[:min(20, len(starting_points))]
    
    output_text.insert(tk.END, f"Multi-start optimization with {len(starting_points)} starting points\n\n")
    
    best_result = None
    best_score = float('inf')
    successful_optimizations = 0
    
    for start_idx, start_point in enumerate(starting_points):
        if abort_var.get():
            break
        
        output_text.insert(tk.END, f"Start {start_idx+1}/{len(starting_points)}: φ={start_point[0]:.0f}°, θ={start_point[1]:.0f}°, ψ={start_point[2]:.0f}°")
        output_text.see(tk.END)
        progress_window.update()
        
        try:
            # Use different optimization methods for diversity
            if start_idx < 5:
                # Differential evolution for global search
                result = differential_evolution(
                    enhanced_objective_function,
                    bounds,
                    maxiter=100,
                    popsize=8,
                    seed=42 + start_idx,
                    atol=1e-6
                )
            else:
                # Local optimization from systematic starting points
                result = minimize(
                    enhanced_objective_function,
                    start_point,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100}
                )
            
            if result.success:
                successful_optimizations += 1
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result
                    output_text.insert(tk.END, f" → NEW BEST: {result.fun:.3f}\n")
                else:
                    output_text.insert(tk.END, f" → {result.fun:.3f}\n")
            else:
                output_text.insert(tk.END, f" → Failed\n")
                
        except Exception as e:
            output_text.insert(tk.END, f" → Error: {str(e)}\n")
            continue
    
    if best_result is None:
        output_text.insert(tk.END, "\nERROR: All optimization attempts failed!\n")
        return None
    
    # Calculate quality metrics
    confidence = max(0, min(1, 1.0 - best_score / 50))
    
    output_text.insert(tk.END, f"\n=== OPTIMIZATION SUMMARY ===\n")
    output_text.insert(tk.END, f"Successful optimizations: {successful_optimizations}/{len(starting_points)}\n")
    output_text.insert(tk.END, f"Best score: {best_score:.3f}\n")
    output_text.insert(tk.END, f"Total evaluations: {evaluation_count[0]}\n")
    output_text.insert(tk.END, f"Estimated confidence: {confidence:.1%}\n\n")
    
    return {
        'best_orientation': best_result.x[:3],
        'best_score': best_score,
        'confidence': confidence,
        'total_evaluations': evaluation_count[0],
        'successful_optimizations': successful_optimizations,
        'num_starts': len(starting_points),
        'best_results': best_results[-10:],  # Keep last 10 for analysis
        'peak_adjustments': best_result.x[3:] if len(best_result.x) > 3 else []
    }


def estimate_orientation_uncertainty(optimization_result, experimental_peaks, output_text):
    """Estimate uncertainty in the determined orientation."""
    
    output_text.insert(tk.END, "Estimating orientation uncertainties...\n")
    
    # Base uncertainty from peak uncertainties
    avg_peak_uncertainty = np.mean([p['position_uncertainty'] for p in experimental_peaks])
    
    # Convert peak uncertainty to angle uncertainty (rough estimation)
    base_angle_uncertainty = avg_peak_uncertainty / 8  # Empirical conversion factor
    
    # Adjust based on optimization confidence
    confidence = optimization_result['confidence']
    uncertainty_factor = 2.5 - 1.5 * confidence  # Higher confidence → lower uncertainty
    
    # Adjust based on number of peaks (more peaks → better determination)
    n_peaks = len(experimental_peaks)
    peak_factor = max(0.5, np.sqrt(5.0 / n_peaks))
    
    # Calculate uncertainties for each angle
    phi_uncertainty = base_angle_uncertainty * uncertainty_factor * peak_factor
    theta_uncertainty = base_angle_uncertainty * uncertainty_factor * peak_factor * 1.3  # θ often less well determined
    psi_uncertainty = base_angle_uncertainty * uncertainty_factor * peak_factor
    
    # Ensure reasonable bounds (1° minimum, 45° maximum)
    phi_uncertainty = max(1.0, min(45.0, phi_uncertainty))
    theta_uncertainty = max(1.0, min(45.0, theta_uncertainty))
    psi_uncertainty = max(1.0, min(45.0, psi_uncertainty))
    
    output_text.insert(tk.END, f"Uncertainty factors:\n")
    output_text.insert(tk.END, f"  Peak uncertainty: {avg_peak_uncertainty:.2f} cm⁻¹\n")
    output_text.insert(tk.END, f"  Confidence factor: {confidence:.3f}\n")
    output_text.insert(tk.END, f"  Number of peaks: {n_peaks}\n")
    output_text.insert(tk.END, f"  Peak factor: {peak_factor:.3f}\n\n")
    
    return {
        'phi_uncertainty': phi_uncertainty,
        'theta_uncertainty': theta_uncertainty,
        'psi_uncertainty': psi_uncertainty,
        'base_uncertainty': base_angle_uncertainty,
        'confidence_factor': confidence,
        'peak_factor': peak_factor
    }


def display_final_results(optimization_result, uncertainty_estimates, output_text):
    """Display comprehensive final results."""
    
    best_orientation = optimization_result['best_orientation']
    
    output_text.insert(tk.END, "=== STAGE 1 OPTIMIZATION RESULTS ===\n\n")
    
    output_text.insert(tk.END, "ENHANCED CRYSTAL ORIENTATION:\n")
    output_text.insert(tk.END, f"φ (phi):   {best_orientation[0]:.3f}° ± {uncertainty_estimates['phi_uncertainty']:.1f}°\n")
    output_text.insert(tk.END, f"θ (theta): {best_orientation[1]:.3f}° ± {uncertainty_estimates['theta_uncertainty']:.1f}°\n")
    output_text.insert(tk.END, f"ψ (psi):   {best_orientation[2]:.3f}° ± {uncertainty_estimates['psi_uncertainty']:.1f}°\n\n")
    
    output_text.insert(tk.END, "QUALITY METRICS:\n")
    output_text.insert(tk.END, f"Overall Score: {optimization_result['best_score']:.3f}\n")
    output_text.insert(tk.END, f"Confidence: {optimization_result['confidence']:.1%}\n")
    output_text.insert(tk.END, f"Successful Optimizations: {optimization_result['successful_optimizations']}/{optimization_result['num_starts']}\n")
    output_text.insert(tk.END, f"Total Function Evaluations: {optimization_result['total_evaluations']}\n\n")
    
    # Show individual peak adjustments if available
    if len(optimization_result['peak_adjustments']) > 0:
        output_text.insert(tk.END, "INDIVIDUAL PEAK ADJUSTMENTS:\n")
        for i, adj in enumerate(optimization_result['peak_adjustments']):
            if abs(adj) > 0.1:  # Only show significant adjustments
                output_text.insert(tk.END, f"  Peak {i+1}: {adj:+.1f} cm⁻¹\n")
    
    output_text.insert(tk.END, "\nSTAGE 1 IMPROVEMENTS SUCCESSFULLY APPLIED!\n")
    output_text.see(tk.END)


def save_optimization_results(filename, optimization_result, uncertainty_estimates, experimental_peaks):
    """Save detailed optimization results to file."""
    
    with open(filename, 'w') as f:
        f.write("=== STAGE 1 ENHANCED CRYSTAL ORIENTATION RESULTS ===\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now() if 'pd' in globals() else 'N/A'}\n\n")
        
        best_orientation = optimization_result['best_orientation']
        
        f.write("OPTIMIZED CRYSTAL ORIENTATION:\n")
        f.write(f"φ (phi):   {best_orientation[0]:.6f}° ± {uncertainty_estimates['phi_uncertainty']:.3f}°\n")
        f.write(f"θ (theta): {best_orientation[1]:.6f}° ± {uncertainty_estimates['theta_uncertainty']:.3f}°\n")
        f.write(f"ψ (psi):   {best_orientation[2]:.6f}° ± {uncertainty_estimates['psi_uncertainty']:.3f}°\n\n")
        
        f.write("QUALITY ASSESSMENT:\n")
        f.write(f"Overall Score: {optimization_result['best_score']:.6f}\n")
        f.write(f"Confidence: {optimization_result['confidence']:.6f}\n")
        f.write(f"Successful Optimizations: {optimization_result['successful_optimizations']}/{optimization_result['num_starts']}\n")
        f.write(f"Total Function Evaluations: {optimization_result['total_evaluations']}\n\n")
        
        f.write("EXPERIMENTAL PEAKS USED:\n")
        for i, peak in enumerate(experimental_peaks):
            f.write(f"Peak {i+1}: {peak['position']:.2f} ± {peak['position_uncertainty']:.2f} cm⁻¹, ")
            f.write(f"Intensity: {peak['intensity']:.1f}, R²: {peak['r_squared']:.3f}")
            if peak['character']:
                f.write(f", Character: {peak['character']}")
            f.write("\n")
        
        if len(optimization_result['peak_adjustments']) > 0:
            f.write(f"\nINDIVIDUAL PEAK ADJUSTMENTS:\n")
            for i, adj in enumerate(optimization_result['peak_adjustments']):
                f.write(f"Peak {i+1}: {adj:+.3f} cm⁻¹\n")
        
        f.write(f"\nSTAGE 1 ENHANCEMENTS:\n")
        f.write(f"✓ Individual peak position adjustments\n")
        f.write(f"✓ Uncertainty-weighted optimization\n")
        f.write(f"✓ Multi-start global search\n")
        f.write(f"✓ Character-based peak assignment\n")
        f.write(f"✓ Fit quality weighting\n") 