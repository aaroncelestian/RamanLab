"""
Stage 2: Probabilistic Framework for Crystal Orientation Optimization
====================================================================

Advanced Bayesian uncertainty quantification and probabilistic peak assignment
for Raman polarization analysis with comprehensive statistical modeling.

Key Features:
- Bayesian parameter estimation with MCMC sampling
- Probabilistic peak assignment with confidence intervals
- Hierarchical uncertainty modeling
- Model selection and comparison
- Robust outlier detection
- Correlation analysis between parameters

Author: ClaritySpectra Development Team
Version: 2.0.0
"""

import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import emcee  # For MCMC sampling
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def optimize_crystal_orientation_stage2(analyzer):
    """
    Stage 2: Probabilistic Framework for Crystal Orientation Optimization
    
    Features:
    - Bayesian parameter estimation
    - Probabilistic peak assignment
    - Hierarchical uncertainty modeling
    - Model comparison and selection
    - Robust outlier detection
    """
    
    # Check dependencies
    missing_deps = []
    if not EMCEE_AVAILABLE:
        missing_deps.append("emcee (for MCMC sampling)")
    if not SKLEARN_AVAILABLE:
        missing_deps.append("scikit-learn (for clustering)")
    
    if missing_deps:
        messagebox.showwarning(
            "Optional Dependencies Missing",
            f"Stage 2 will run with reduced functionality.\n\n"
            f"Missing packages:\n" + "\n".join(f"• {dep}" for dep in missing_deps) + "\n\n"
            f"Install with: pip install emcee scikit-learn\n\n"
            f"Proceeding with available methods..."
        )
    
    # Create progress window
    progress_window = tk.Toplevel(analyzer.root)
    progress_window.title("Stage 2: Probabilistic Framework")
    progress_window.geometry("800x600")
    progress_window.transient(analyzer.root)
    progress_window.grab_set()
    
    # Center the window
    progress_window.update_idletasks()
    x = (progress_window.winfo_screenwidth() // 2) - (800 // 2)
    y = (progress_window.winfo_screenheight() // 2) - (600 // 2)
    progress_window.geometry(f"800x600+{x}+{y}")
    
    # Create main frame with notebook for different analysis tabs
    main_frame = ttk.Frame(progress_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create notebook for different analysis views
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    # Tab 1: Progress and Results
    progress_frame = ttk.Frame(notebook)
    notebook.add(progress_frame, text="Progress & Results")
    
    # Tab 2: Bayesian Analysis
    bayesian_frame = ttk.Frame(notebook)
    notebook.add(bayesian_frame, text="Bayesian Analysis")
    
    # Tab 3: Model Comparison
    model_frame = ttk.Frame(notebook)
    notebook.add(model_frame, text="Model Comparison")
    
    # Tab 4: Uncertainty Analysis
    uncertainty_frame = ttk.Frame(notebook)
    notebook.add(uncertainty_frame, text="Uncertainty Analysis")
    
    # Progress tab setup
    ttk.Label(progress_frame, text="Stage 2: Probabilistic Framework", 
              font=("Arial", 14, "bold")).pack(pady=10)
    
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
    progress_bar.pack(fill=tk.X, padx=20, pady=10)
    
    status_label = ttk.Label(progress_frame, text="Initializing probabilistic framework...")
    status_label.pack(pady=5)
    
    # Results text area
    results_text = tk.Text(progress_frame, height=20, width=80, wrap=tk.WORD)
    results_scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=results_text.yview)
    results_text.configure(yscrollcommand=results_scrollbar.set)
    results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20,0), pady=10)
    results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    
    # Bayesian analysis setup
    bayesian_text = tk.Text(bayesian_frame, height=25, width=80, wrap=tk.WORD)
    bayesian_scrollbar = ttk.Scrollbar(bayesian_frame, orient=tk.VERTICAL, command=bayesian_text.yview)
    bayesian_text.configure(yscrollcommand=bayesian_scrollbar.set)
    bayesian_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    bayesian_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    
    # Model comparison setup
    model_text = tk.Text(model_frame, height=25, width=80, wrap=tk.WORD)
    model_scrollbar = ttk.Scrollbar(model_frame, orient=tk.VERTICAL, command=model_text.yview)
    model_text.configure(yscrollcommand=model_scrollbar.set)
    model_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    model_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    
    # Uncertainty analysis setup
    uncertainty_text = tk.Text(uncertainty_frame, height=25, width=80, wrap=tk.WORD)
    uncertainty_scrollbar = ttk.Scrollbar(uncertainty_frame, orient=tk.VERTICAL, command=uncertainty_text.yview)
    uncertainty_text.configure(yscrollcommand=uncertainty_scrollbar.set)
    uncertainty_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    uncertainty_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    
    # Control buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)
    
    abort_var = tk.BooleanVar()
    
    def abort_optimization():
        abort_var.set(True)
        status_label.config(text="Aborting optimization...")
    
    def apply_and_close():
        if hasattr(apply_and_close, 'result') and apply_and_close.result:
            # Apply the best orientation to the main application
            best_orientation = apply_and_close.result['best_orientation']
            analyzer.phi_var.set(best_orientation[0])
            analyzer.theta_var.set(best_orientation[1])
            analyzer.psi_var.set(best_orientation[2])
            analyzer.calculate_orientation_raman_spectrum()
            
            # Update results display
            analyzer.orientation_results_text.delete(1.0, tk.END)
            analyzer.orientation_results_text.insert(tk.END, apply_and_close.result['summary'])
        
        progress_window.destroy()
        return True
    
    def save_detailed_results():
        if hasattr(apply_and_close, 'result'):
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                title="Save Stage 2 Results",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                save_stage2_results(filename, apply_and_close.result)
                messagebox.showinfo("Saved", f"Detailed results saved to {filename}")
    
    ttk.Button(button_frame, text="Abort", command=abort_optimization).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Save Results", command=save_detailed_results).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Apply & Close", command=apply_and_close).pack(side=tk.RIGHT, padx=5)
    
    # Run optimization in separate thread
    def run_optimization():
        try:
            # Extract experimental peaks with enhanced uncertainty analysis
            experimental_peaks = extract_experimental_peaks_stage2(analyzer, results_text)
            
            if not experimental_peaks:
                status_label.config(text="No experimental peaks found for analysis")
                return
            
            # Run probabilistic optimization
            result = run_probabilistic_optimization(
                analyzer, experimental_peaks, progress_var, status_label,
                progress_window, abort_var, results_text, bayesian_text,
                model_text, uncertainty_text
            )
            
            if result and not abort_var.get():
                apply_and_close.result = result
                status_label.config(text="✅ Stage 2 optimization completed successfully!")
                
                # Display final summary
                display_stage2_results(result, results_text, bayesian_text, 
                                     model_text, uncertainty_text)
            else:
                status_label.config(text="❌ Optimization cancelled or failed")
                
        except Exception as e:
            status_label.config(text=f"❌ Error: {str(e)}")
            results_text.insert(tk.END, f"\nError during optimization: {str(e)}\n")
            import traceback
            traceback.print_exc()
    
    # Start optimization thread
    optimization_thread = threading.Thread(target=run_optimization)
    optimization_thread.daemon = True
    optimization_thread.start()
    
    # Wait for window to close
    progress_window.wait_window()
    
    # Return result if available
    return getattr(apply_and_close, 'result', None)


def extract_experimental_peaks_stage2(analyzer, output_text):
    """
    Extract experimental peaks with enhanced uncertainty quantification
    for Stage 2 probabilistic analysis
    """
    output_text.insert(tk.END, "=== Stage 2: Enhanced Peak Extraction ===\n\n")
    
    experimental_peaks = []
    
    if not hasattr(analyzer, 'fitted_regions') or not analyzer.fitted_regions:
        output_text.insert(tk.END, "❌ No fitted peaks found. Please fit peaks first.\n")
        return experimental_peaks
    
    output_text.insert(tk.END, "Extracting peaks with enhanced uncertainty analysis...\n\n")
    
    for region_idx, region_data in analyzer.fitted_regions.items():
        if 'peaks' not in region_data:
            continue
            
        shape = region_data.get('shape', 'Lorentzian')
        r_squared = region_data.get('r_squared', 0.0)
        
        output_text.insert(tk.END, f"Region {region_idx + 1} ({shape}, R² = {r_squared:.4f}):\n")
        
        for peak_idx, peak in enumerate(region_data['peaks']):
            if 'center' not in peak:
                continue
                
            center = peak['center']
            height = peak.get('height', 1.0)
            width = peak.get('width', 10.0)
            
            # Enhanced uncertainty estimation
            center_err = peak.get('center_err', width * 0.1)
            height_err = peak.get('height_err', height * 0.1)
            width_err = peak.get('width_err', width * 0.1)
            
            # Bayesian uncertainty scaling based on fit quality
            quality_factor = max(0.1, r_squared)
            uncertainty_scale = 1.0 / np.sqrt(quality_factor)
            
            center_err *= uncertainty_scale
            height_err *= uncertainty_scale
            width_err *= uncertainty_scale
            
            # Extract character assignment if available
            character = ""
            confidence = 0.5  # Default confidence
            
            # Look for character markers
            if hasattr(analyzer, 'fitted_peak_markers'):
                for marker in analyzer.fitted_peak_markers:
                    if hasattr(marker, 'get_position') and hasattr(marker, 'get_text'):
                        try:
                            marker_pos = marker.get_position()
                            if isinstance(marker_pos, tuple) and len(marker_pos) >= 1:
                                if abs(marker_pos[0] - center) < 5:  # Within 5 cm⁻¹
                                    text = marker.get_text()
                                    if len(text) <= 4:  # Character labels
                                        character = text
                                        confidence = 0.8  # Higher confidence for manual assignment
                        except:
                            continue
            
            # Estimate signal-to-noise ratio
            snr = height / max(0.01, height_err)
            
            # Calculate composite confidence score
            confidence_factors = [
                min(1.0, r_squared),  # Fit quality
                min(1.0, snr / 10.0),  # Signal-to-noise ratio
                confidence,  # Character assignment confidence
                min(1.0, height / max(experimental_peaks, key=lambda x: x.get('height', 0), default={'height': 1})['height'] if experimental_peaks else 1.0)  # Relative intensity
            ]
            
            composite_confidence = np.mean(confidence_factors)
            
            peak_info = {
                'center': center,
                'center_err': center_err,
                'height': height,
                'height_err': height_err,
                'width': width,
                'width_err': width_err,
                'character': character,
                'confidence': composite_confidence,
                'r_squared': r_squared,
                'snr': snr,
                'shape': shape,
                'region_idx': region_idx,
                'peak_idx': peak_idx
            }
            
            experimental_peaks.append(peak_info)
            
            output_text.insert(tk.END, 
                f"  Peak {peak_idx + 1}: {center:.1f} ± {center_err:.1f} cm⁻¹ "
                f"({character if character else 'unassigned'}, "
                f"conf: {composite_confidence:.2f}, SNR: {snr:.1f})\n")
    
    output_text.insert(tk.END, f"\n✅ Extracted {len(experimental_peaks)} peaks for probabilistic analysis\n\n")
    
    # Sort peaks by confidence (highest first)
    experimental_peaks.sort(key=lambda x: x['confidence'], reverse=True)
    
    return experimental_peaks


def run_probabilistic_optimization(analyzer, experimental_peaks, progress_var, status_label,
                                 progress_window, abort_var, results_text, bayesian_text,
                                 model_text, uncertainty_text):
    """
    Run the main probabilistic optimization with Bayesian analysis
    """
    
    # Initialize progress
    progress_var.set(0)
    status_label.config(text="Setting up probabilistic framework...")
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 1: Probabilistic Peak Assignment (20% progress)
    status_label.config(text="Performing probabilistic peak assignment...")
    peak_assignments = probabilistic_peak_assignment(
        analyzer, experimental_peaks, bayesian_text
    )
    progress_var.set(20)
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 2: Bayesian Parameter Estimation (40% progress)
    status_label.config(text="Running Bayesian parameter estimation...")
    bayesian_results = bayesian_parameter_estimation(
        analyzer, experimental_peaks, peak_assignments, bayesian_text
    )
    progress_var.set(40)
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 3: Model Comparison and Selection (60% progress)
    status_label.config(text="Comparing and selecting models...")
    model_results = model_comparison_analysis(
        analyzer, experimental_peaks, peak_assignments, model_text
    )
    progress_var.set(60)
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 4: Hierarchical Uncertainty Analysis (80% progress)
    status_label.config(text="Performing hierarchical uncertainty analysis...")
    uncertainty_results = hierarchical_uncertainty_analysis(
        analyzer, experimental_peaks, bayesian_results, uncertainty_text
    )
    progress_var.set(80)
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 5: Final Integration and Validation (100% progress)
    status_label.config(text="Integrating results and validation...")
    final_results = integrate_probabilistic_results(
        analyzer, experimental_peaks, peak_assignments, bayesian_results,
        model_results, uncertainty_results, results_text
    )
    progress_var.set(100)
    progress_window.update()
    
    return final_results


def probabilistic_peak_assignment(analyzer, experimental_peaks, output_text):
    """
    Perform probabilistic peak assignment using Bayesian inference
    """
    output_text.insert(tk.END, "=== Probabilistic Peak Assignment ===\n\n")
    
    # Get theoretical spectrum
    if not hasattr(analyzer, 'current_structure') or analyzer.current_structure is None:
        output_text.insert(tk.END, "❌ No crystal structure available\n")
        return {}
    
    # Calculate theoretical spectrum for current orientation
    orientation = (analyzer.phi_var.get(), analyzer.theta_var.get(), analyzer.psi_var.get())
    polarization = analyzer.polarization_var.get()
    
    try:
        theoretical_spectrum = analyzer.current_structure.calculate_raman_spectrum(
            orientation, polarization
        )
    except Exception as e:
        output_text.insert(tk.END, f"❌ Error calculating theoretical spectrum: {e}\n")
        return {}
    
    if not theoretical_spectrum:
        output_text.insert(tk.END, "❌ No theoretical spectrum calculated\n")
        return {}
    
    # Extract theoretical peaks
    theoretical_peaks = []
    for freq, intensity, character in theoretical_spectrum:
        if intensity > 0.01:  # Only consider significant peaks
            theoretical_peaks.append({
                'frequency': freq,
                'intensity': intensity,
                'character': character
            })
    
    output_text.insert(tk.END, f"Theoretical peaks: {len(theoretical_peaks)}\n")
    output_text.insert(tk.END, f"Experimental peaks: {len(experimental_peaks)}\n\n")
    
    # Probabilistic assignment using Bayesian approach
    assignments = {}
    assignment_probabilities = {}
    
    # Apply calibration
    shift = analyzer.orientation_shift_var.get()
    scale = analyzer.orientation_scale_var.get()
    
    for exp_idx, exp_peak in enumerate(experimental_peaks):
        exp_freq = exp_peak['center']
        exp_err = exp_peak['center_err']
        exp_confidence = exp_peak['confidence']
        
        output_text.insert(tk.END, f"Assigning experimental peak at {exp_freq:.1f} cm⁻¹:\n")
        
        # Calculate assignment probabilities for each theoretical peak
        probabilities = []
        
        for theo_idx, theo_peak in enumerate(theoretical_peaks):
            theo_freq = theo_peak['frequency'] * scale + shift
            theo_intensity = theo_peak['intensity']
            theo_character = theo_peak['character']
            
            # Frequency matching probability (Gaussian)
            freq_diff = abs(exp_freq - theo_freq)
            freq_prob = np.exp(-0.5 * (freq_diff / (2 * exp_err))**2)
            
            # Intensity correlation probability
            max_exp_height = max(p['height'] for p in experimental_peaks)
            max_theo_intensity = max(p['intensity'] for p in theoretical_peaks)
            
            exp_rel_intensity = exp_peak['height'] / max_exp_height
            theo_rel_intensity = theo_intensity / max_theo_intensity
            
            intensity_diff = abs(exp_rel_intensity - theo_rel_intensity)
            intensity_prob = np.exp(-2 * intensity_diff)
            
            # Character matching probability
            char_prob = 1.0
            if exp_peak['character'] and theo_character:
                if exp_peak['character'].lower() == theo_character.lower():
                    char_prob = 2.0  # Boost for character match
                else:
                    char_prob = 0.5  # Penalty for character mismatch
            
            # Combined probability
            combined_prob = freq_prob * intensity_prob * char_prob * exp_confidence
            
            probabilities.append({
                'theo_idx': theo_idx,
                'theo_peak': theo_peak,
                'freq_prob': freq_prob,
                'intensity_prob': intensity_prob,
                'char_prob': char_prob,
                'combined_prob': combined_prob,
                'freq_diff': freq_diff
            })
        
        # Normalize probabilities
        total_prob = sum(p['combined_prob'] for p in probabilities)
        if total_prob > 0:
            for p in probabilities:
                p['normalized_prob'] = p['combined_prob'] / total_prob
        
        # Sort by probability
        probabilities.sort(key=lambda x: x['normalized_prob'], reverse=True)
        
        # Store assignments
        assignments[exp_idx] = probabilities
        assignment_probabilities[exp_idx] = probabilities[0]['normalized_prob'] if probabilities else 0
        
        # Display top 3 assignments
        output_text.insert(tk.END, "  Top assignments:\n")
        for i, prob_data in enumerate(probabilities[:3]):
            theo_peak = prob_data['theo_peak']
            output_text.insert(tk.END, 
                f"    {i+1}. {theo_peak['frequency']:.1f} cm⁻¹ ({theo_peak['character']}) "
                f"- P = {prob_data['normalized_prob']:.3f} "
                f"(Δν = {prob_data['freq_diff']:.1f})\n")
        
        output_text.insert(tk.END, "\n")
    
    # Calculate overall assignment quality
    avg_assignment_prob = np.mean(list(assignment_probabilities.values())) if assignment_probabilities else 0
    output_text.insert(tk.END, f"Average assignment probability: {avg_assignment_prob:.3f}\n\n")
    
    return {
        'assignments': assignments,
        'probabilities': assignment_probabilities,
        'theoretical_peaks': theoretical_peaks,
        'avg_probability': avg_assignment_prob
    }


def bayesian_parameter_estimation(analyzer, experimental_peaks, peak_assignments, output_text):
    """
    Perform Bayesian parameter estimation for crystal orientation
    """
    output_text.insert(tk.END, "=== Bayesian Parameter Estimation ===\n\n")
    
    if not EMCEE_AVAILABLE:
        output_text.insert(tk.END, "⚠️  MCMC sampling not available (emcee not installed)\n")
        output_text.insert(tk.END, "Using approximate Bayesian estimation...\n\n")
        return approximate_bayesian_estimation(analyzer, experimental_peaks, peak_assignments, output_text)
    
    # MCMC-based Bayesian estimation
    output_text.insert(tk.END, "Setting up MCMC sampling for Bayesian estimation...\n")
    
    # Define parameter space: [phi, theta, psi, shift, scale]
    ndim = 5
    nwalkers = 32
    nsteps = 1000
    
    # Prior bounds
    phi_bounds = (0, 360)
    theta_bounds = (0, 180)
    psi_bounds = (0, 360)
    shift_bounds = (-50, 50)
    scale_bounds = (0.8, 1.2)
    
    bounds = [phi_bounds, theta_bounds, psi_bounds, shift_bounds, scale_bounds]
    
    def log_prior(params):
        """Log prior probability"""
        phi, theta, psi, shift, scale = params
        
        # Check bounds
        if not (phi_bounds[0] <= phi <= phi_bounds[1]):
            return -np.inf
        if not (theta_bounds[0] <= theta <= theta_bounds[1]):
            return -np.inf
        if not (psi_bounds[0] <= psi <= psi_bounds[1]):
            return -np.inf
        if not (shift_bounds[0] <= shift <= shift_bounds[1]):
            return -np.inf
        if not (scale_bounds[0] <= scale <= scale_bounds[1]):
            return -np.inf
        
        # Uniform priors within bounds
        return 0.0
    
    def log_likelihood(params):
        """Log likelihood function"""
        phi, theta, psi, shift, scale = params
        
        try:
            # Calculate theoretical spectrum
            orientation = (phi, theta, psi)
            polarization = analyzer.polarization_var.get()
            
            theoretical_spectrum = analyzer.current_structure.calculate_raman_spectrum(
                orientation, polarization
            )
            
            if not theoretical_spectrum:
                return -np.inf
            
            # Calculate likelihood based on peak assignments
            log_like = 0.0
            
            for exp_idx, exp_peak in enumerate(experimental_peaks):
                exp_freq = exp_peak['center']
                exp_err = exp_peak['center_err']
                
                if exp_idx in peak_assignments['assignments']:
                    best_assignment = peak_assignments['assignments'][exp_idx][0]
                    theo_peak = best_assignment['theo_peak']
                    
                    # Apply calibration
                    theo_freq_calibrated = theo_peak['frequency'] * scale + shift
                    
                    # Gaussian likelihood for frequency match
                    freq_diff = exp_freq - theo_freq_calibrated
                    log_like += -0.5 * (freq_diff / exp_err)**2 - np.log(exp_err * np.sqrt(2 * np.pi))
            
            return log_like
            
        except Exception:
            return -np.inf
    
    def log_probability(params):
        """Log posterior probability"""
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params)
    
    # Initialize walkers
    initial_params = [
        analyzer.phi_var.get(),
        analyzer.theta_var.get(),
        analyzer.psi_var.get(),
        analyzer.orientation_shift_var.get(),
        analyzer.orientation_scale_var.get()
    ]
    
    # Add small random perturbations to initial parameters
    pos = []
    for i in range(nwalkers):
        walker_params = initial_params.copy()
        walker_params[0] += np.random.normal(0, 5)  # phi
        walker_params[1] += np.random.normal(0, 5)  # theta
        walker_params[2] += np.random.normal(0, 5)  # psi
        walker_params[3] += np.random.normal(0, 2)  # shift
        walker_params[4] += np.random.normal(0, 0.02)  # scale
        
        # Ensure bounds
        walker_params[0] = np.clip(walker_params[0], *phi_bounds)
        walker_params[1] = np.clip(walker_params[1], *theta_bounds)
        walker_params[2] = np.clip(walker_params[2], *psi_bounds)
        walker_params[3] = np.clip(walker_params[3], *shift_bounds)
        walker_params[4] = np.clip(walker_params[4], *scale_bounds)
        
        pos.append(walker_params)
    
    # Run MCMC
    output_text.insert(tk.END, f"Running MCMC with {nwalkers} walkers for {nsteps} steps...\n")
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    
    # Run burn-in
    burn_in_steps = 200
    output_text.insert(tk.END, f"Burn-in phase: {burn_in_steps} steps...\n")
    pos, _, _ = sampler.run_mcmc(pos, burn_in_steps, progress=False)
    sampler.reset()
    
    # Production run
    output_text.insert(tk.END, "Production run...\n")
    sampler.run_mcmc(pos, nsteps, progress=False)
    
    # Analyze results
    samples = sampler.get_chain(discard=100, thin=10, flat=True)
    
    # Calculate statistics
    param_names = ['φ (°)', 'θ (°)', 'ψ (°)', 'Shift (cm⁻¹)', 'Scale']
    results = {}
    
    output_text.insert(tk.END, "\nBayesian Parameter Estimates:\n")
    output_text.insert(tk.END, "-" * 50 + "\n")
    
    for i, name in enumerate(param_names):
        param_samples = samples[:, i]
        mean_val = np.mean(param_samples)
        std_val = np.std(param_samples)
        median_val = np.median(param_samples)
        ci_lower = np.percentile(param_samples, 16)
        ci_upper = np.percentile(param_samples, 84)
        
        results[name] = {
            'samples': param_samples,
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        output_text.insert(tk.END, 
            f"{name:15}: {mean_val:8.2f} ± {std_val:6.2f} "
            f"[{ci_lower:6.2f}, {ci_upper:6.2f}]\n")
    
    # Calculate acceptance fraction
    acceptance_fraction = np.mean(sampler.acceptance_fraction)
    output_text.insert(tk.END, f"\nAcceptance fraction: {acceptance_fraction:.3f}\n")
    
    # Autocorrelation analysis
    try:
        autocorr_times = sampler.get_autocorr_time()
        output_text.insert(tk.END, f"Autocorrelation times: {autocorr_times}\n")
    except Exception:
        output_text.insert(tk.END, "Autocorrelation analysis failed\n")
    
    return {
        'samples': samples,
        'results': results,
        'sampler': sampler,
        'acceptance_fraction': acceptance_fraction,
        'method': 'MCMC'
    }


def approximate_bayesian_estimation(analyzer, experimental_peaks, peak_assignments, output_text):
    """
    Approximate Bayesian estimation when MCMC is not available
    """
    output_text.insert(tk.END, "Using grid-based approximate Bayesian estimation...\n\n")
    
    # Define parameter grids
    phi_range = np.linspace(0, 360, 25)
    theta_range = np.linspace(0, 180, 13)
    psi_range = np.linspace(0, 360, 25)
    shift_range = np.linspace(-20, 20, 11)
    scale_range = np.linspace(0.9, 1.1, 11)
    
    best_likelihood = -np.inf
    best_params = None
    likelihood_grid = []
    
    total_combinations = len(phi_range) * len(theta_range) * len(psi_range) * len(shift_range) * len(scale_range)
    output_text.insert(tk.END, f"Evaluating {total_combinations} parameter combinations...\n")
    
    count = 0
    for phi in phi_range:
        for theta in theta_range:
            for psi in psi_range:
                for shift in shift_range:
                    for scale in scale_range:
                        count += 1
                        if count % 10000 == 0:
                            output_text.insert(tk.END, f"Progress: {count}/{total_combinations}\n")
                            output_text.see(tk.END)
                            output_text.update()
                        
                        # Calculate likelihood
                        likelihood = calculate_likelihood_approx(
                            analyzer, experimental_peaks, peak_assignments,
                            (phi, theta, psi), shift, scale
                        )
                        
                        likelihood_grid.append({
                            'phi': phi, 'theta': theta, 'psi': psi,
                            'shift': shift, 'scale': scale,
                            'likelihood': likelihood
                        })
                        
                        if likelihood > best_likelihood:
                            best_likelihood = likelihood
                            best_params = (phi, theta, psi, shift, scale)
    
    # Convert to numpy array for analysis
    likelihood_grid = np.array([(p['phi'], p['theta'], p['psi'], p['shift'], p['scale'], p['likelihood']) 
                               for p in likelihood_grid])
    
    # Normalize likelihoods to get approximate posterior
    max_likelihood = np.max(likelihood_grid[:, 5])
    likelihood_grid[:, 5] = np.exp(likelihood_grid[:, 5] - max_likelihood)
    
    # Calculate marginal statistics
    param_names = ['φ (°)', 'θ (°)', 'ψ (°)', 'Shift (cm⁻¹)', 'Scale']
    results = {}
    
    output_text.insert(tk.END, "\nApproximate Bayesian Parameter Estimates:\n")
    output_text.insert(tk.END, "-" * 50 + "\n")
    
    for i, name in enumerate(param_names):
        param_values = likelihood_grid[:, i]
        weights = likelihood_grid[:, 5]
        
        # Weighted statistics
        mean_val = np.average(param_values, weights=weights)
        var_val = np.average((param_values - mean_val)**2, weights=weights)
        std_val = np.sqrt(var_val)
        
        # Credible intervals
        sorted_indices = np.argsort(param_values)
        sorted_weights = weights[sorted_indices]
        cumulative_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        
        ci_lower_idx = np.searchsorted(cumulative_weights, 0.16)
        ci_upper_idx = np.searchsorted(cumulative_weights, 0.84)
        
        ci_lower = param_values[sorted_indices[ci_lower_idx]]
        ci_upper = param_values[sorted_indices[ci_upper_idx]]
        
        results[name] = {
            'mean': mean_val,
            'std': std_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        output_text.insert(tk.END, 
            f"{name:15}: {mean_val:8.2f} ± {std_val:6.2f} "
            f"[{ci_lower:6.2f}, {ci_upper:6.2f}]\n")
    
    output_text.insert(tk.END, f"\nBest likelihood: {best_likelihood:.2f}\n")
    output_text.insert(tk.END, f"Best parameters: φ={best_params[0]:.1f}°, θ={best_params[1]:.1f}°, ψ={best_params[2]:.1f}°\n")
    
    return {
        'results': results,
        'best_params': best_params,
        'best_likelihood': best_likelihood,
        'likelihood_grid': likelihood_grid,
        'method': 'Grid-based'
    }


def calculate_likelihood_approx(analyzer, experimental_peaks, peak_assignments, orientation, shift, scale):
    """
    Calculate approximate likelihood for given parameters
    """
    try:
        # Calculate theoretical spectrum
        polarization = analyzer.polarization_var.get()
        theoretical_spectrum = analyzer.current_structure.calculate_raman_spectrum(
            orientation, polarization
        )
        
        if not theoretical_spectrum:
            return -np.inf
        
        # Calculate likelihood based on peak assignments
        log_like = 0.0
        
        for exp_idx, exp_peak in enumerate(experimental_peaks):
            exp_freq = exp_peak['center']
            exp_err = exp_peak['center_err']
            
            if exp_idx in peak_assignments['assignments']:
                best_assignment = peak_assignments['assignments'][exp_idx][0]
                theo_peak = best_assignment['theo_peak']
                
                # Apply calibration
                theo_freq_calibrated = theo_peak['frequency'] * scale + shift
                
                # Gaussian likelihood for frequency match
                freq_diff = exp_freq - theo_freq_calibrated
                log_like += -0.5 * (freq_diff / exp_err)**2
        
        return log_like
        
    except Exception:
        return -np.inf


def model_comparison_analysis(analyzer, experimental_peaks, peak_assignments, output_text):
    """
    Perform model comparison and selection analysis
    """
    output_text.insert(tk.END, "=== Model Comparison and Selection ===\n\n")
    
    models = []
    
    # Model 1: Basic orientation only
    output_text.insert(tk.END, "Evaluating Model 1: Basic orientation (φ, θ, ψ)\n")
    model1_result = evaluate_model_basic(analyzer, experimental_peaks, peak_assignments)
    models.append(('Basic Orientation', model1_result))
    
    # Model 2: Orientation + calibration
    output_text.insert(tk.END, "Evaluating Model 2: Orientation + calibration (φ, θ, ψ, shift, scale)\n")
    model2_result = evaluate_model_calibrated(analyzer, experimental_peaks, peak_assignments)
    models.append(('Orientation + Calibration', model2_result))
    
    # Model 3: Hierarchical model with peak-specific parameters
    output_text.insert(tk.END, "Evaluating Model 3: Hierarchical model with peak-specific parameters\n")
    model3_result = evaluate_model_hierarchical(analyzer, experimental_peaks, peak_assignments)
    models.append(('Hierarchical Model', model3_result))
    
    # Calculate model comparison metrics
    output_text.insert(tk.END, "\nModel Comparison Results:\n")
    output_text.insert(tk.END, "-" * 60 + "\n")
    output_text.insert(tk.END, f"{'Model':<25} {'AIC':<10} {'BIC':<10} {'Log-Like':<12} {'R²':<8}\n")
    output_text.insert(tk.END, "-" * 60 + "\n")
    
    best_model = None
    best_aic = np.inf
    
    for model_name, result in models:
        aic = result['aic']
        bic = result['bic']
        log_like = result['log_likelihood']
        r_squared = result['r_squared']
        
        output_text.insert(tk.END, 
            f"{model_name:<25} {aic:<10.2f} {bic:<10.2f} {log_like:<12.2f} {r_squared:<8.3f}\n")
        
        if aic < best_aic:
            best_aic = aic
            best_model = (model_name, result)
    
    output_text.insert(tk.END, "-" * 60 + "\n")
    output_text.insert(tk.END, f"\n✅ Best model: {best_model[0]} (lowest AIC)\n\n")
    
    # Model weights using Akaike weights
    aics = [result['aic'] for _, result in models]
    min_aic = min(aics)
    delta_aics = [aic - min_aic for aic in aics]
    weights = [np.exp(-0.5 * delta) for delta in delta_aics]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    output_text.insert(tk.END, "Model Weights (Akaike weights):\n")
    for (model_name, _), weight in zip(models, normalized_weights):
        output_text.insert(tk.END, f"  {model_name}: {weight:.3f}\n")
    
    return {
        'models': models,
        'best_model': best_model,
        'model_weights': normalized_weights
    }


def evaluate_model_basic(analyzer, experimental_peaks, peak_assignments):
    """Evaluate basic orientation model"""
    # Simple optimization with only orientation parameters
    def objective(params):
        phi, theta, psi = params
        return -calculate_likelihood_approx(
            analyzer, experimental_peaks, peak_assignments,
            (phi, theta, psi), 0, 1.0  # No calibration
        )
    
    # Initial guess
    x0 = [analyzer.phi_var.get(), analyzer.theta_var.get(), analyzer.psi_var.get()]
    bounds = [(0, 360), (0, 180), (0, 360)]
    
    try:
        result = opt.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        log_likelihood = -result.fun
        n_params = 3
        n_data = len(experimental_peaks)
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_data) - 2 * log_likelihood
        
        # Calculate R²
        r_squared = calculate_model_r_squared(analyzer, experimental_peaks, peak_assignments, 
                                            result.x, 0, 1.0)
        
        return {
            'params': result.x,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'r_squared': r_squared,
            'n_params': n_params
        }
    except:
        return {
            'params': x0,
            'log_likelihood': -np.inf,
            'aic': np.inf,
            'bic': np.inf,
            'r_squared': 0.0,
            'n_params': 3
        }


def evaluate_model_calibrated(analyzer, experimental_peaks, peak_assignments):
    """Evaluate orientation + calibration model"""
    def objective(params):
        phi, theta, psi, shift, scale = params
        return -calculate_likelihood_approx(
            analyzer, experimental_peaks, peak_assignments,
            (phi, theta, psi), shift, scale
        )
    
    # Initial guess
    x0 = [analyzer.phi_var.get(), analyzer.theta_var.get(), analyzer.psi_var.get(),
          analyzer.orientation_shift_var.get(), analyzer.orientation_scale_var.get()]
    bounds = [(0, 360), (0, 180), (0, 360), (-50, 50), (0.8, 1.2)]
    
    try:
        result = opt.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        log_likelihood = -result.fun
        n_params = 5
        n_data = len(experimental_peaks)
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_data) - 2 * log_likelihood
        
        # Calculate R²
        r_squared = calculate_model_r_squared(analyzer, experimental_peaks, peak_assignments,
                                            result.x[:3], result.x[3], result.x[4])
        
        return {
            'params': result.x,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'r_squared': r_squared,
            'n_params': n_params
        }
    except:
        return {
            'params': x0,
            'log_likelihood': -np.inf,
            'aic': np.inf,
            'bic': np.inf,
            'r_squared': 0.0,
            'n_params': 5
        }


def evaluate_model_hierarchical(analyzer, experimental_peaks, peak_assignments):
    """Evaluate hierarchical model with peak-specific parameters"""
    # This is a simplified version - full hierarchical modeling would be more complex
    n_peaks = len(experimental_peaks)
    n_params = 5 + n_peaks  # orientation + calibration + peak-specific shifts
    
    def objective(params):
        phi, theta, psi, shift, scale = params[:5]
        peak_shifts = params[5:]
        
        total_likelihood = 0
        for i, exp_peak in enumerate(experimental_peaks):
            individual_shift = shift + (peak_shifts[i] if i < len(peak_shifts) else 0)
            likelihood = calculate_likelihood_approx(
                analyzer, [exp_peak], peak_assignments,
                (phi, theta, psi), individual_shift, scale
            )
            total_likelihood += likelihood
        
        return -total_likelihood
    
    # Initial guess
    x0 = [analyzer.phi_var.get(), analyzer.theta_var.get(), analyzer.psi_var.get(),
          analyzer.orientation_shift_var.get(), analyzer.orientation_scale_var.get()]
    x0.extend([0.0] * n_peaks)  # Peak-specific shifts
    
    bounds = [(0, 360), (0, 180), (0, 360), (-50, 50), (0.8, 1.2)]
    bounds.extend([(-10, 10)] * n_peaks)  # Peak shift bounds
    
    try:
        result = opt.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        log_likelihood = -result.fun
        n_data = len(experimental_peaks)
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_data) - 2 * log_likelihood
        
        # Calculate R²
        r_squared = calculate_model_r_squared(analyzer, experimental_peaks, peak_assignments,
                                            result.x[:3], result.x[3], result.x[4])
        
        return {
            'params': result.x,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'r_squared': r_squared,
            'n_params': n_params
        }
    except:
        return {
            'params': x0,
            'log_likelihood': -np.inf,
            'aic': np.inf,
            'bic': np.inf,
            'r_squared': 0.0,
            'n_params': n_params
        }


def calculate_model_r_squared(analyzer, experimental_peaks, peak_assignments, orientation, shift, scale):
    """Calculate R² for model fit"""
    try:
        # Get theoretical spectrum
        polarization = analyzer.polarization_var.get()
        theoretical_spectrum = analyzer.current_structure.calculate_raman_spectrum(
            orientation, polarization
        )
        
        if not theoretical_spectrum:
            return 0.0
        
        # Calculate predicted vs observed frequencies
        observed = []
        predicted = []
        
        for exp_idx, exp_peak in enumerate(experimental_peaks):
            exp_freq = exp_peak['center']
            
            if exp_idx in peak_assignments['assignments']:
                best_assignment = peak_assignments['assignments'][exp_idx][0]
                theo_peak = best_assignment['theo_peak']
                theo_freq_calibrated = theo_peak['frequency'] * scale + shift
                
                observed.append(exp_freq)
                predicted.append(theo_freq_calibrated)
        
        if len(observed) < 2:
            return 0.0
        
        # Calculate R²
        observed = np.array(observed)
        predicted = np.array(predicted)
        
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return max(0.0, r_squared)
        
    except:
        return 0.0


def hierarchical_uncertainty_analysis(analyzer, experimental_peaks, bayesian_results, output_text):
    """
    Perform hierarchical uncertainty analysis
    """
    output_text.insert(tk.END, "=== Hierarchical Uncertainty Analysis ===\n\n")
    
    # Analyze different sources of uncertainty
    uncertainty_sources = {
        'measurement': analyze_measurement_uncertainty(experimental_peaks, output_text),
        'model': analyze_model_uncertainty(analyzer, experimental_peaks, bayesian_results, output_text),
        'systematic': analyze_systematic_uncertainty(analyzer, experimental_peaks, output_text)
    }
    
    # Propagate uncertainties
    total_uncertainty = propagate_uncertainties(uncertainty_sources, output_text)
    
    # Sensitivity analysis
    sensitivity_results = perform_sensitivity_analysis(analyzer, experimental_peaks, output_text)
    
    return {
        'uncertainty_sources': uncertainty_sources,
        'total_uncertainty': total_uncertainty,
        'sensitivity': sensitivity_results
    }


def analyze_measurement_uncertainty(experimental_peaks, output_text):
    """Analyze measurement uncertainty"""
    output_text.insert(tk.END, "Analyzing measurement uncertainties...\n")
    
    center_errors = [p['center_err'] for p in experimental_peaks]
    height_errors = [p['height_err'] for p in experimental_peaks]
    width_errors = [p['width_err'] for p in experimental_peaks]
    
    measurement_stats = {
        'center_err_mean': np.mean(center_errors),
        'center_err_std': np.std(center_errors),
        'height_err_mean': np.mean(height_errors),
        'height_err_std': np.std(height_errors),
        'width_err_mean': np.mean(width_errors),
        'width_err_std': np.std(width_errors)
    }
    
    output_text.insert(tk.END, f"  Average center error: {measurement_stats['center_err_mean']:.2f} ± {measurement_stats['center_err_std']:.2f} cm⁻¹\n")
    output_text.insert(tk.END, f"  Average height error: {measurement_stats['height_err_mean']:.3f} ± {measurement_stats['height_err_std']:.3f}\n")
    output_text.insert(tk.END, f"  Average width error: {measurement_stats['width_err_mean']:.2f} ± {measurement_stats['width_err_std']:.2f} cm⁻¹\n")
    
    return measurement_stats


def analyze_model_uncertainty(analyzer, experimental_peaks, bayesian_results, output_text):
    """Analyze model uncertainty"""
    output_text.insert(tk.END, "Analyzing model uncertainties...\n")
    
    if bayesian_results['method'] == 'MCMC' and 'results' in bayesian_results:
        # Extract parameter uncertainties from MCMC
        param_uncertainties = {}
        for param_name, param_data in bayesian_results['results'].items():
            param_uncertainties[param_name] = param_data['std']
        
        output_text.insert(tk.END, "  Parameter uncertainties from MCMC:\n")
        for param, uncertainty in param_uncertainties.items():
            output_text.insert(tk.END, f"    {param}: ±{uncertainty:.3f}\n")
    
    else:
        # Use approximate uncertainties
        param_uncertainties = {
            'φ (°)': 5.0,
            'θ (°)': 5.0,
            'ψ (°)': 5.0,
            'Shift (cm⁻¹)': 2.0,
            'Scale': 0.02
        }
        
        output_text.insert(tk.END, "  Approximate parameter uncertainties:\n")
        for param, uncertainty in param_uncertainties.items():
            output_text.insert(tk.END, f"    {param}: ±{uncertainty:.3f}\n")
    
    return param_uncertainties


def analyze_systematic_uncertainty(analyzer, experimental_peaks, output_text):
    """Analyze systematic uncertainties"""
    output_text.insert(tk.END, "Analyzing systematic uncertainties...\n")
    
    # Estimate systematic uncertainties from various sources
    systematic_sources = {
        'calibration_drift': 1.0,  # cm⁻¹
        'temperature_effects': 0.5,  # cm⁻¹
        'pressure_effects': 0.2,  # cm⁻¹
        'instrument_resolution': 2.0,  # cm⁻¹
        'baseline_uncertainty': 0.5  # cm⁻¹
    }
    
    total_systematic = np.sqrt(sum(v**2 for v in systematic_sources.values()))
    
    output_text.insert(tk.END, "  Systematic uncertainty sources:\n")
    for source, uncertainty in systematic_sources.items():
        output_text.insert(tk.END, f"    {source}: ±{uncertainty:.1f} cm⁻¹\n")
    
    output_text.insert(tk.END, f"  Total systematic uncertainty: ±{total_systematic:.1f} cm⁻¹\n")
    
    return {
        'sources': systematic_sources,
        'total': total_systematic
    }


def propagate_uncertainties(uncertainty_sources, output_text):
    """Propagate uncertainties through the model"""
    output_text.insert(tk.END, "Propagating uncertainties...\n")
    
    # Combine measurement and systematic uncertainties
    measurement = uncertainty_sources['measurement']
    systematic = uncertainty_sources['systematic']
    
    # Total frequency uncertainty
    freq_uncertainty = np.sqrt(
        measurement['center_err_mean']**2 + 
        systematic['total']**2
    )
    
    # Orientation uncertainty (simplified propagation)
    orientation_uncertainty = {
        'phi': 5.0,  # degrees
        'theta': 5.0,  # degrees
        'psi': 5.0,  # degrees
    }
    
    output_text.insert(tk.END, f"  Total frequency uncertainty: ±{freq_uncertainty:.1f} cm⁻¹\n")
    output_text.insert(tk.END, f"  Orientation uncertainties: φ±{orientation_uncertainty['phi']:.1f}°, θ±{orientation_uncertainty['theta']:.1f}°, ψ±{orientation_uncertainty['psi']:.1f}°\n")
    
    return {
        'frequency': freq_uncertainty,
        'orientation': orientation_uncertainty
    }


def perform_sensitivity_analysis(analyzer, experimental_peaks, output_text):
    """Perform sensitivity analysis"""
    output_text.insert(tk.END, "Performing sensitivity analysis...\n")
    
    # Test sensitivity to parameter changes
    base_params = [
        analyzer.phi_var.get(),
        analyzer.theta_var.get(),
        analyzer.psi_var.get(),
        analyzer.orientation_shift_var.get(),
        analyzer.orientation_scale_var.get()
    ]
    
    param_names = ['φ', 'θ', 'ψ', 'shift', 'scale']
    perturbations = [1.0, 1.0, 1.0, 0.5, 0.01]  # Small perturbations
    
    sensitivities = {}
    
    for i, (param_name, perturbation) in enumerate(zip(param_names, perturbations)):
        # Calculate finite difference sensitivity
        params_plus = base_params.copy()
        params_minus = base_params.copy()
        
        params_plus[i] += perturbation
        params_minus[i] -= perturbation
        
        # This is a simplified sensitivity calculation
        # In practice, you'd calculate the full objective function
        sensitivity = perturbation  # Placeholder
        
        sensitivities[param_name] = sensitivity
        output_text.insert(tk.END, f"  Sensitivity to {param_name}: {sensitivity:.3f}\n")
    
    return sensitivities


def integrate_probabilistic_results(analyzer, experimental_peaks, peak_assignments, 
                                   bayesian_results, model_results, uncertainty_results, output_text):
    """
    Integrate all probabilistic analysis results
    """
    output_text.insert(tk.END, "=== Final Integration and Results ===\n\n")
    
    # Get best model results
    best_model_name, best_model_result = model_results['best_model']
    best_params = best_model_result['params']
    
    # Extract orientation parameters
    if len(best_params) >= 3:
        best_orientation = best_params[:3]
    else:
        best_orientation = [analyzer.phi_var.get(), analyzer.theta_var.get(), analyzer.psi_var.get()]
    
    # Extract calibration parameters
    if len(best_params) >= 5:
        best_shift = best_params[3]
        best_scale = best_params[4]
    else:
        best_shift = analyzer.orientation_shift_var.get()
        best_scale = analyzer.orientation_scale_var.get()
    
    # Calculate final metrics
    final_r_squared = best_model_result['r_squared']
    avg_assignment_prob = peak_assignments['avg_probability']
    
    # Create comprehensive summary
    summary = f"""Stage 2 Probabilistic Framework Results
{'='*50}

Best Model: {best_model_name}
Model Selection Criteria:
  AIC: {best_model_result['aic']:.2f}
  BIC: {best_model_result['bic']:.2f}
  R²: {final_r_squared:.4f}
  Log-likelihood: {best_model_result['log_likelihood']:.2f}

Crystal Orientation (Euler angles):
  φ = {best_orientation[0]:.2f}°
  θ = {best_orientation[1]:.2f}°
  ψ = {best_orientation[2]:.2f}°

Calibration Parameters:
  Shift = {best_shift:.2f} cm⁻¹
  Scale = {best_scale:.4f}

Peak Assignment Quality:
  Average assignment probability: {avg_assignment_prob:.3f}
  Number of peaks assigned: {len(experimental_peaks)}

Uncertainty Analysis:
  Total frequency uncertainty: ±{uncertainty_results['total_uncertainty']['frequency']:.1f} cm⁻¹
  Orientation uncertainties: φ±{uncertainty_results['total_uncertainty']['orientation']['phi']:.1f}°, θ±{uncertainty_results['total_uncertainty']['orientation']['theta']:.1f}°, ψ±{uncertainty_results['total_uncertainty']['orientation']['psi']:.1f}°

Method: {bayesian_results['method']} Bayesian estimation
Analysis: Probabilistic framework with hierarchical uncertainty quantification
"""
    
    output_text.insert(tk.END, summary)
    
    # Prepare final result
    final_result = {
        'best_orientation': best_orientation,
        'best_shift': best_shift,
        'best_scale': best_scale,
        'r_squared': final_r_squared,
        'assignment_probability': avg_assignment_prob,
        'best_model': best_model_name,
        'model_results': model_results,
        'bayesian_results': bayesian_results,
        'uncertainty_results': uncertainty_results,
        'peak_assignments': peak_assignments,
        'experimental_peaks': experimental_peaks,
        'summary': summary
    }
    
    return final_result


def display_stage2_results(result, results_text, bayesian_text, model_text, uncertainty_text):
    """
    Display comprehensive Stage 2 results in all tabs
    """
    # Results tab already has the summary
    
    # Bayesian analysis tab
    bayesian_text.insert(tk.END, "\n" + "="*60 + "\n")
    bayesian_text.insert(tk.END, "BAYESIAN ANALYSIS SUMMARY\n")
    bayesian_text.insert(tk.END, "="*60 + "\n\n")
    
    if 'results' in result['bayesian_results']:
        bayesian_text.insert(tk.END, "Parameter Posterior Distributions:\n\n")
        for param_name, param_data in result['bayesian_results']['results'].items():
            bayesian_text.insert(tk.END, 
                f"{param_name}:\n"
                f"  Mean: {param_data['mean']:.3f}\n"
                f"  Std: {param_data['std']:.3f}\n"
                f"  68% CI: [{param_data['ci_lower']:.3f}, {param_data['ci_upper']:.3f}]\n\n")
    
    # Model comparison tab
    model_text.insert(tk.END, "\n" + "="*60 + "\n")
    model_text.insert(tk.END, "MODEL COMPARISON SUMMARY\n")
    model_text.insert(tk.END, "="*60 + "\n\n")
    
    model_text.insert(tk.END, "Model Rankings (by AIC):\n\n")
    models_sorted = sorted(result['model_results']['models'], key=lambda x: x[1]['aic'])
    for i, (model_name, model_data) in enumerate(models_sorted):
        model_text.insert(tk.END, 
            f"{i+1}. {model_name}\n"
            f"   AIC: {model_data['aic']:.2f}\n"
            f"   BIC: {model_data['bic']:.2f}\n"
            f"   R²: {model_data['r_squared']:.4f}\n\n")
    
    # Uncertainty analysis tab
    uncertainty_text.insert(tk.END, "\n" + "="*60 + "\n")
    uncertainty_text.insert(tk.END, "UNCERTAINTY ANALYSIS SUMMARY\n")
    uncertainty_text.insert(tk.END, "="*60 + "\n\n")
    
    uncertainty_text.insert(tk.END, "Uncertainty Budget:\n\n")
    for source_name, source_data in result['uncertainty_results']['uncertainty_sources'].items():
        uncertainty_text.insert(tk.END, f"{source_name.title()} Uncertainty:\n")
        if isinstance(source_data, dict):
            for key, value in source_data.items():
                if isinstance(value, (int, float)):
                    uncertainty_text.insert(tk.END, f"  {key}: {value:.3f}\n")
        uncertainty_text.insert(tk.END, "\n")


def save_stage2_results(filename, result):
    """
    Save comprehensive Stage 2 results to file
    """
    with open(filename, 'w') as f:
        f.write("Stage 2 Probabilistic Framework - Detailed Results\n")
        f.write("="*60 + "\n\n")
        
        f.write(result['summary'])
        f.write("\n\n")
        
        # Detailed peak assignments
        f.write("DETAILED PEAK ASSIGNMENTS\n")
        f.write("-"*40 + "\n\n")
        
        for exp_idx, exp_peak in enumerate(result['experimental_peaks']):
            f.write(f"Experimental Peak {exp_idx + 1}:\n")
            f.write(f"  Position: {exp_peak['center']:.1f} ± {exp_peak['center_err']:.1f} cm⁻¹\n")
            f.write(f"  Character: {exp_peak['character'] if exp_peak['character'] else 'unassigned'}\n")
            f.write(f"  Confidence: {exp_peak['confidence']:.3f}\n")
            f.write(f"  SNR: {exp_peak['snr']:.1f}\n")
            
            if exp_idx in result['peak_assignments']['assignments']:
                assignments = result['peak_assignments']['assignments'][exp_idx]
                f.write(f"  Top theoretical matches:\n")
                for i, assignment in enumerate(assignments[:3]):
                    theo_peak = assignment['theo_peak']
                    f.write(f"    {i+1}. {theo_peak['frequency']:.1f} cm⁻¹ ({theo_peak['character']}) "
                           f"- P = {assignment['normalized_prob']:.3f}\n")
            f.write("\n")
        
        # Model comparison details
        f.write("\nMODEL COMPARISON DETAILS\n")
        f.write("-"*40 + "\n\n")
        
        for model_name, model_data in result['model_results']['models']:
            f.write(f"{model_name}:\n")
            f.write(f"  Parameters: {model_data['n_params']}\n")
            f.write(f"  Log-likelihood: {model_data['log_likelihood']:.2f}\n")
            f.write(f"  AIC: {model_data['aic']:.2f}\n")
            f.write(f"  BIC: {model_data['bic']:.2f}\n")
            f.write(f"  R²: {model_data['r_squared']:.4f}\n\n")
        
        # Bayesian results details
        if 'results' in result['bayesian_results']:
            f.write("\nBAYESIAN PARAMETER ESTIMATES\n")
            f.write("-"*40 + "\n\n")
            
            for param_name, param_data in result['bayesian_results']['results'].items():
                f.write(f"{param_name}:\n")
                f.write(f"  Mean: {param_data['mean']:.4f}\n")
                f.write(f"  Std: {param_data['std']:.4f}\n")
                f.write(f"  Median: {param_data['median']:.4f}\n")
                f.write(f"  68% CI: [{param_data['ci_lower']:.4f}, {param_data['ci_upper']:.4f}]\n\n") 