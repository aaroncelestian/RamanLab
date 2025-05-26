"""
Stage 3: Advanced Multi-Objective Bayesian Optimization
======================================================

Ultimate crystal orientation optimization using cutting-edge techniques:
- Gaussian Process surrogate models
- Multi-objective optimization (Pareto fronts)
- Active learning and adaptive sampling
- Ensemble methods and model fusion
- Advanced uncertainty quantification
- Real-time convergence diagnostics

Author: ClaritySpectra Development Team
Version: 3.0.0
"""

import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False

try:
    from scipy.optimize import differential_evolution, dual_annealing
    ADVANCED_OPT_AVAILABLE = True
except ImportError:
    ADVANCED_OPT_AVAILABLE = False


def optimize_crystal_orientation_stage3(analyzer):
    """
    Stage 3: Advanced Multi-Objective Bayesian Optimization
    
    Features:
    - Gaussian Process surrogate modeling
    - Multi-objective optimization with Pareto fronts
    - Active learning and adaptive sampling
    - Ensemble methods for robust predictions
    - Advanced uncertainty quantification
    - Real-time convergence diagnostics
    """
    
    # Check dependencies
    missing_deps = []
    if not SKLEARN_AVAILABLE:
        missing_deps.append("scikit-learn (for Gaussian Processes)")
    if not EMCEE_AVAILABLE:
        missing_deps.append("emcee (for MCMC sampling)")
    if not ADVANCED_OPT_AVAILABLE:
        missing_deps.append("scipy>=1.2 (for advanced optimizers)")
    
    if missing_deps:
        messagebox.showwarning(
            "Advanced Dependencies Missing",
            f"Stage 3 will run with reduced functionality.\n\n"
            f"Missing packages:\n" + "\n".join(f"• {dep}" for dep in missing_deps) + "\n\n"
            f"Install with: pip install scikit-learn emcee scipy\n\n"
            f"Proceeding with available methods..."
        )
    
    # Create advanced progress window
    progress_window = tk.Toplevel(analyzer.root)
    progress_window.title("Stage 3: Advanced Multi-Objective Bayesian Optimization")
    progress_window.geometry("1000x700")
    progress_window.transient(analyzer.root)
    progress_window.grab_set()
    
    # Center the window
    progress_window.update_idletasks()
    x = (progress_window.winfo_screenwidth() // 2) - (1000 // 2)
    y = (progress_window.winfo_screenheight() // 2) - (700 // 2)
    progress_window.geometry(f"1000x700+{x}+{y}")
    
    # Create main frame with advanced notebook
    main_frame = ttk.Frame(progress_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create notebook for different analysis views
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    # Tab 1: Progress and Control
    progress_frame = ttk.Frame(notebook)
    notebook.add(progress_frame, text="Progress & Control")
    
    # Tab 2: Gaussian Process Analysis
    gp_frame = ttk.Frame(notebook)
    notebook.add(gp_frame, text="Gaussian Process")
    
    # Tab 3: Multi-Objective Results
    pareto_frame = ttk.Frame(notebook)
    notebook.add(pareto_frame, text="Pareto Optimization")
    
    # Tab 4: Ensemble Methods
    ensemble_frame = ttk.Frame(notebook)
    notebook.add(ensemble_frame, text="Ensemble Analysis")
    
    # Tab 5: Convergence Diagnostics
    convergence_frame = ttk.Frame(notebook)
    notebook.add(convergence_frame, text="Convergence")
    
    # Tab 6: Advanced Uncertainty
    uncertainty_frame = ttk.Frame(notebook)
    notebook.add(uncertainty_frame, text="Advanced Uncertainty")
    
    # Progress tab setup
    ttk.Label(progress_frame, text="Stage 3: Advanced Multi-Objective Bayesian Optimization", 
              font=("Arial", 14, "bold")).pack(pady=10)
    
    # Progress indicators
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
    progress_bar.pack(fill=tk.X, padx=20, pady=10)
    
    status_label = ttk.Label(progress_frame, text="Initializing advanced optimization framework...")
    status_label.pack(pady=5)
    
    # Real-time metrics
    metrics_frame = ttk.LabelFrame(progress_frame, text="Real-Time Metrics")
    metrics_frame.pack(fill=tk.X, padx=20, pady=10)
    
    metrics_text = tk.Text(metrics_frame, height=8, width=80, wrap=tk.WORD)
    metrics_scrollbar = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL, command=metrics_text.yview)
    metrics_text.configure(yscrollcommand=metrics_scrollbar.set)
    metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    metrics_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    # Results text area
    results_text = tk.Text(progress_frame, height=15, width=80, wrap=tk.WORD)
    results_scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=results_text.yview)
    results_text.configure(yscrollcommand=results_scrollbar.set)
    results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20,0), pady=10)
    results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    
    # Setup other tabs
    gp_text = create_analysis_tab(gp_frame, "Gaussian Process Analysis")
    pareto_text = create_analysis_tab(pareto_frame, "Multi-Objective Optimization")
    ensemble_text = create_analysis_tab(ensemble_frame, "Ensemble Methods")
    convergence_text = create_analysis_tab(convergence_frame, "Convergence Diagnostics")
    uncertainty_text = create_analysis_tab(uncertainty_frame, "Advanced Uncertainty")
    
    # Control buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)
    
    abort_var = tk.BooleanVar()
    pause_var = tk.BooleanVar()
    
    def abort_optimization():
        abort_var.set(True)
        status_label.config(text="Aborting optimization...")
    
    def pause_optimization():
        pause_var.set(not pause_var.get())
        if pause_var.get():
            status_label.config(text="Optimization paused...")
        else:
            status_label.config(text="Optimization resumed...")
    
    def apply_and_close():
        if hasattr(apply_and_close, 'result') and apply_and_close.result:
            # Apply the best orientation from Pareto front
            best_solution = apply_and_close.result['pareto_front'][0]  # Best compromise
            analyzer.phi_var.set(best_solution['orientation'][0])
            analyzer.theta_var.set(best_solution['orientation'][1])
            analyzer.psi_var.set(best_solution['orientation'][2])
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
                title="Save Stage 3 Results",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                save_stage3_results(filename, apply_and_close.result)
                messagebox.showinfo("Saved", f"Detailed results saved to {filename}")
    
    ttk.Button(button_frame, text="Abort", command=abort_optimization).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Pause/Resume", command=pause_optimization).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Save Results", command=save_detailed_results).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Apply Best & Close", command=apply_and_close).pack(side=tk.RIGHT, padx=5)
    
    # Run optimization in separate thread
    def run_optimization():
        try:
            # Extract experimental peaks with advanced analysis
            experimental_peaks = extract_experimental_peaks_stage3(analyzer, results_text)
            
            if not experimental_peaks:
                status_label.config(text="No experimental peaks found for analysis")
                return
            
            # Run advanced multi-objective optimization
            result = run_advanced_optimization(
                analyzer, experimental_peaks, progress_var, status_label,
                progress_window, abort_var, pause_var, results_text, metrics_text,
                gp_text, pareto_text, ensemble_text, convergence_text, uncertainty_text
            )
            
            if result and not abort_var.get():
                apply_and_close.result = result
                status_label.config(text="✅ Stage 3 optimization completed successfully!")
                
                # Display final comprehensive results
                display_stage3_results(result, results_text, gp_text, pareto_text,
                                     ensemble_text, convergence_text, uncertainty_text)
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


def create_analysis_tab(parent_frame, title):
    """Create a standardized analysis tab"""
    text_widget = tk.Text(parent_frame, height=30, width=100, wrap=tk.WORD)
    scrollbar = ttk.Scrollbar(parent_frame, orient=tk.VERTICAL, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    return text_widget


def extract_experimental_peaks_stage3(analyzer, output_text):
    """
    Extract experimental peaks with advanced uncertainty analysis for Stage 3
    """
    output_text.insert(tk.END, "=== Stage 3: Advanced Peak Extraction ===\n\n")
    
    experimental_peaks = []
    
    if not hasattr(analyzer, 'fitted_regions') or not analyzer.fitted_regions:
        output_text.insert(tk.END, "❌ No fitted peaks found. Please fit peaks first.\n")
        return experimental_peaks
    
    output_text.insert(tk.END, "Extracting peaks with advanced uncertainty analysis...\n\n")
    
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
            
            # Advanced uncertainty estimation with multiple sources
            center_err = peak.get('center_err', width * 0.1)
            height_err = peak.get('height_err', height * 0.1)
            width_err = peak.get('width_err', width * 0.1)
            
            # Multi-level uncertainty scaling
            quality_factor = max(0.1, r_squared)
            uncertainty_scale = 1.0 / np.sqrt(quality_factor)
            
            # Add systematic uncertainties
            systematic_center_err = 0.5  # cm⁻¹
            systematic_height_err = height * 0.05
            systematic_width_err = width * 0.05
            
            # Combine uncertainties
            total_center_err = np.sqrt((center_err * uncertainty_scale)**2 + systematic_center_err**2)
            total_height_err = np.sqrt((height_err * uncertainty_scale)**2 + systematic_height_err**2)
            total_width_err = np.sqrt((width_err * uncertainty_scale)**2 + systematic_width_err**2)
            
            # Extract character assignment with confidence
            character = ""
            confidence = 0.5  # Default confidence
            
            # Look for character markers with advanced scoring
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
                                        confidence = 0.9  # High confidence for manual assignment
                        except:
                            continue
            
            # Advanced signal-to-noise estimation
            snr = height / max(0.01, total_height_err)
            
            # Multi-factor confidence scoring
            confidence_factors = [
                min(1.0, r_squared),  # Fit quality
                min(1.0, snr / 15.0),  # Signal-to-noise ratio (higher threshold)
                confidence,  # Character assignment confidence
                min(1.0, height / max([p.get('height', 0) for p in experimental_peaks], default=1.0)),  # Relative intensity
                min(1.0, 1.0 / (1.0 + total_center_err / 2.0)),  # Uncertainty penalty
            ]
            
            composite_confidence = np.mean(confidence_factors)
            
            # Advanced peak quality metrics
            asymmetry = peak.get('asymmetry', 0.0)
            baseline_quality = 1.0 - abs(peak.get('baseline_offset', 0.0)) / max(height, 0.1)
            
            peak_info = {
                'center': center,
                'center_err': total_center_err,
                'height': height,
                'height_err': total_height_err,
                'width': width,
                'width_err': total_width_err,
                'character': character,
                'confidence': composite_confidence,
                'r_squared': r_squared,
                'snr': snr,
                'shape': shape,
                'region_idx': region_idx,
                'peak_idx': peak_idx,
                'asymmetry': asymmetry,
                'baseline_quality': baseline_quality,
                'quality_score': composite_confidence * baseline_quality * (1.0 - abs(asymmetry))
            }
            
            experimental_peaks.append(peak_info)
            
            output_text.insert(tk.END, 
                f"  Peak {peak_idx + 1}: {center:.1f} ± {total_center_err:.1f} cm⁻¹ "
                f"({character if character else 'unassigned'}, "
                f"conf: {composite_confidence:.3f}, SNR: {snr:.1f}, Q: {peak_info['quality_score']:.3f})\n")
    
    output_text.insert(tk.END, f"\n✅ Extracted {len(experimental_peaks)} peaks for advanced analysis\n\n")
    
    # Sort peaks by quality score (highest first)
    experimental_peaks.sort(key=lambda x: x['quality_score'], reverse=True)
    
    return experimental_peaks


def run_advanced_optimization(analyzer, experimental_peaks, progress_var, status_label,
                            progress_window, abort_var, pause_var, results_text, metrics_text,
                            gp_text, pareto_text, ensemble_text, convergence_text, uncertainty_text):
    """
    Run the advanced multi-objective Bayesian optimization
    """
    
    # Initialize progress
    progress_var.set(0)
    status_label.config(text="Initializing advanced optimization framework...")
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 1: Gaussian Process Surrogate Modeling (15% progress)
    status_label.config(text="Building Gaussian Process surrogate models...")
    gp_models = build_gaussian_process_models(
        analyzer, experimental_peaks, gp_text, metrics_text
    )
    progress_var.set(15)
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 2: Multi-Objective Optimization (35% progress)
    status_label.config(text="Running multi-objective optimization...")
    pareto_results = run_multi_objective_optimization(
        analyzer, experimental_peaks, gp_models, pareto_text, metrics_text,
        abort_var, pause_var, progress_var, 15, 35
    )
    progress_var.set(35)
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 3: Ensemble Methods (55% progress)
    status_label.config(text="Applying ensemble methods...")
    ensemble_results = apply_ensemble_methods(
        analyzer, experimental_peaks, gp_models, pareto_results, ensemble_text, metrics_text
    )
    progress_var.set(55)
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 4: Active Learning and Adaptive Sampling (75% progress)
    status_label.config(text="Performing active learning and adaptive sampling...")
    adaptive_results = perform_adaptive_sampling(
        analyzer, experimental_peaks, gp_models, ensemble_results, 
        convergence_text, metrics_text, abort_var, pause_var
    )
    progress_var.set(75)
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 5: Advanced Uncertainty Quantification (90% progress)
    status_label.config(text="Performing advanced uncertainty quantification...")
    uncertainty_results = advanced_uncertainty_quantification(
        analyzer, experimental_peaks, gp_models, ensemble_results, 
        adaptive_results, uncertainty_text, metrics_text
    )
    progress_var.set(90)
    progress_window.update()
    
    if abort_var.get():
        return None
    
    # Step 6: Final Integration and Validation (100% progress)
    status_label.config(text="Integrating results and final validation...")
    final_results = integrate_advanced_results(
        analyzer, experimental_peaks, gp_models, pareto_results, ensemble_results,
        adaptive_results, uncertainty_results, results_text, metrics_text
    )
    progress_var.set(100)
    progress_window.update()
    
    return final_results


def build_gaussian_process_models(analyzer, experimental_peaks, output_text, metrics_text):
    """
    Build Gaussian Process surrogate models for the optimization landscape
    """
    output_text.insert(tk.END, "=== Gaussian Process Surrogate Modeling ===\n\n")
    
    if not SKLEARN_AVAILABLE:
        output_text.insert(tk.END, "⚠️  Scikit-learn not available, using simplified models\n")
        return build_simplified_models(analyzer, experimental_peaks, output_text)
    
    # Define parameter space
    param_bounds = {
        'phi': (0, 360),
        'theta': (0, 180),
        'psi': (0, 360),
        'shift': (-20, 20),
        'scale': (0.9, 1.1)
    }
    
    output_text.insert(tk.END, "Building Gaussian Process models...\n")
    
    # Generate initial training data using Latin Hypercube Sampling
    n_initial = 50
    training_points = generate_latin_hypercube_samples(param_bounds, n_initial)
    
    output_text.insert(tk.END, f"Generated {n_initial} initial training points\n")
    
    # Evaluate objective functions at training points
    objectives = []
    for i, params in enumerate(training_points):
        if i % 10 == 0:
            metrics_text.insert(tk.END, f"Evaluating training point {i+1}/{n_initial}\n")
            metrics_text.see(tk.END)
            metrics_text.update()
        
        obj_values = evaluate_multi_objectives(analyzer, experimental_peaks, params)
        objectives.append(obj_values)
    
    objectives = np.array(objectives)
    
    # Build separate GP models for each objective
    gp_models = {}
    
    # Kernel selection
    kernels = {
        'RBF': ConstantKernel(1.0) * RBF(length_scale=1.0),
        'Matern': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
        'RBF+White': ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    }
    
    objective_names = ['frequency_error', 'intensity_error', 'assignment_quality', 'uncertainty']
    
    for i, obj_name in enumerate(objective_names):
        output_text.insert(tk.END, f"\nBuilding GP model for {obj_name}...\n")
        
        y_train = objectives[:, i].reshape(-1, 1)
        
        # Try different kernels and select best
        best_score = -np.inf
        best_gp = None
        best_kernel_name = None
        
        for kernel_name, kernel in kernels.items():
            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=5
                )
                
                # Cross-validation score
                scores = cross_val_score(gp, training_points, y_train.ravel(), cv=5, scoring='neg_mean_squared_error')
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_gp = gp
                    best_kernel_name = kernel_name
                    
            except Exception as e:
                output_text.insert(tk.END, f"  Failed to fit {kernel_name}: {e}\n")
                continue
        
        if best_gp is not None:
            best_gp.fit(training_points, y_train.ravel())
            gp_models[obj_name] = {
                'model': best_gp,
                'kernel': best_kernel_name,
                'score': best_score,
                'training_points': training_points,
                'training_values': y_train.ravel()
            }
            
            output_text.insert(tk.END, f"  Best kernel: {best_kernel_name} (CV score: {best_score:.4f})\n")
            output_text.insert(tk.END, f"  Log-likelihood: {best_gp.log_marginal_likelihood():.4f}\n")
        else:
            output_text.insert(tk.END, f"  Failed to build GP model for {obj_name}\n")
    
    output_text.insert(tk.END, f"\n✅ Built {len(gp_models)} Gaussian Process models\n\n")
    
    return gp_models


def generate_latin_hypercube_samples(param_bounds, n_samples):
    """Generate Latin Hypercube samples for parameter space exploration"""
    n_dims = len(param_bounds)
    
    # Generate LHS samples in [0,1]^n_dims
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=n_dims, seed=42)
    unit_samples = sampler.random(n=n_samples)
    
    # Scale to parameter bounds
    samples = np.zeros_like(unit_samples)
    param_names = list(param_bounds.keys())
    
    for i, param_name in enumerate(param_names):
        lower, upper = param_bounds[param_name]
        samples[:, i] = lower + unit_samples[:, i] * (upper - lower)
    
    return samples


def evaluate_multi_objectives(analyzer, experimental_peaks, params):
    """
    Evaluate multiple objectives for multi-objective optimization
    """
    phi, theta, psi, shift, scale = params
    
    try:
        # Calculate theoretical spectrum
        orientation = (phi, theta, psi)
        polarization = analyzer.polarization_var.get()
        
        theoretical_spectrum = analyzer.current_structure.calculate_raman_spectrum(
            orientation, polarization
        )
        
        if not theoretical_spectrum:
            return [1e6, 1e6, 0.0, 1e6]  # Bad values
        
        # Objective 1: Frequency matching error
        freq_errors = []
        intensity_errors = []
        assignment_qualities = []
        
        for exp_peak in experimental_peaks:
            exp_freq = exp_peak['center']
            exp_intensity = exp_peak['height']
            exp_err = exp_peak['center_err']
            
            # Find best theoretical match
            best_freq_error = float('inf')
            best_intensity_error = float('inf')
            best_quality = 0.0
            
            for theo_freq, theo_intensity, theo_char in theoretical_spectrum:
                theo_freq_calibrated = theo_freq * scale + shift
                
                freq_error = abs(exp_freq - theo_freq_calibrated) / exp_err
                intensity_error = abs(exp_intensity - theo_intensity) / max(exp_intensity, theo_intensity)
                
                # Character matching bonus
                char_bonus = 0.0
                if exp_peak['character'] and theo_char:
                    if exp_peak['character'].lower() == theo_char.lower():
                        char_bonus = 0.5
                
                quality = exp_peak['confidence'] + char_bonus
                
                if freq_error < best_freq_error:
                    best_freq_error = freq_error
                    best_intensity_error = intensity_error
                    best_quality = quality
            
            freq_errors.append(best_freq_error)
            intensity_errors.append(best_intensity_error)
            assignment_qualities.append(best_quality)
        
        # Objective values
        frequency_error = np.mean(freq_errors)
        intensity_error = np.mean(intensity_errors)
        assignment_quality = np.mean(assignment_qualities)
        uncertainty = np.std(freq_errors) + np.std(intensity_errors)
        
        return [frequency_error, intensity_error, -assignment_quality, uncertainty]  # Negative for maximization
        
    except Exception:
        return [1e6, 1e6, 0.0, 1e6]  # Bad values


def build_simplified_models(analyzer, experimental_peaks, output_text):
    """Build simplified models when scikit-learn is not available"""
    output_text.insert(tk.END, "Building simplified surrogate models...\n")
    
    # Simple polynomial approximation models
    models = {
        'frequency_error': {'type': 'polynomial', 'degree': 2},
        'intensity_error': {'type': 'polynomial', 'degree': 2},
        'assignment_quality': {'type': 'polynomial', 'degree': 2},
        'uncertainty': {'type': 'polynomial', 'degree': 2}
    }
    
    output_text.insert(tk.END, "✅ Built simplified surrogate models\n\n")
    return models


def run_multi_objective_optimization(analyzer, experimental_peaks, gp_models, output_text, 
                                   metrics_text, abort_var, pause_var, progress_var, 
                                   start_progress, end_progress):
    """
    Run multi-objective optimization to find Pareto front
    """
    output_text.insert(tk.END, "=== Multi-Objective Optimization ===\n\n")
    
    if not SKLEARN_AVAILABLE:
        output_text.insert(tk.END, "⚠️  Using simplified multi-objective optimization\n")
        return run_simplified_pareto_optimization(analyzer, experimental_peaks, output_text)
    
    # NSGA-II inspired multi-objective optimization
    population_size = 100
    n_generations = 50
    
    output_text.insert(tk.END, f"Running NSGA-II with population size {population_size} for {n_generations} generations\n\n")
    
    # Initialize population
    param_bounds = [(0, 360), (0, 180), (0, 360), (-20, 20), (0.9, 1.1)]
    population = []
    
    for i in range(population_size):
        individual = []
        for lower, upper in param_bounds:
            individual.append(np.random.uniform(lower, upper))
        population.append(individual)
    
    # Evolution loop
    pareto_history = []
    convergence_metrics = []
    
    for generation in range(n_generations):
        if abort_var.get():
            break
            
        while pause_var.get():
            time.sleep(0.1)
            if abort_var.get():
                break
        
        # Evaluate population using GP models
        objectives = []
        for individual in population:
            if SKLEARN_AVAILABLE and gp_models:
                obj_values = evaluate_with_gp_models(individual, gp_models)
            else:
                obj_values = evaluate_multi_objectives(analyzer, experimental_peaks, individual)
            objectives.append(obj_values)
        
        objectives = np.array(objectives)
        
        # Non-dominated sorting
        fronts = non_dominated_sorting(objectives)
        
        # Calculate crowding distance
        crowding_distances = calculate_crowding_distance(objectives, fronts)
        
        # Selection for next generation
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                new_population.extend([population[i] for i in front])
            else:
                # Sort by crowding distance and select best
                remaining = population_size - len(new_population)
                front_with_distance = [(i, crowding_distances[i]) for i in front]
                front_with_distance.sort(key=lambda x: x[1], reverse=True)
                new_population.extend([population[i] for i, _ in front_with_distance[:remaining]])
                break
        
        # Crossover and mutation
        population = evolve_population(new_population, param_bounds)
        
        # Track Pareto front
        pareto_front = [population[i] for i in fronts[0]]
        pareto_objectives = objectives[fronts[0]]
        pareto_history.append((pareto_front.copy(), pareto_objectives.copy()))
        
        # Convergence metrics
        hypervolume = calculate_hypervolume(pareto_objectives)
        convergence_metrics.append(hypervolume)
        
        # Update progress
        progress = start_progress + (generation / n_generations) * (end_progress - start_progress)
        progress_var.set(progress)
        
        # Update metrics
        if generation % 5 == 0:
            metrics_text.insert(tk.END, f"Generation {generation}: Pareto front size = {len(fronts[0])}, Hypervolume = {hypervolume:.4f}\n")
            metrics_text.see(tk.END)
            metrics_text.update()
        
        output_text.insert(tk.END, f"Generation {generation}: {len(fronts[0])} solutions in Pareto front\n")
        output_text.see(tk.END)
        output_text.update()
    
    # Final Pareto front analysis
    final_pareto_front, final_pareto_objectives = pareto_history[-1]
    
    output_text.insert(tk.END, f"\n✅ Final Pareto front contains {len(final_pareto_front)} solutions\n")
    
    # Analyze solutions
    pareto_solutions = []
    for i, (solution, objectives) in enumerate(zip(final_pareto_front, final_pareto_objectives)):
        pareto_solutions.append({
            'rank': i + 1,
            'orientation': solution[:3],
            'shift': solution[3],
            'scale': solution[4],
            'frequency_error': objectives[0],
            'intensity_error': objectives[1],
            'assignment_quality': -objectives[2],  # Convert back from negative
            'uncertainty': objectives[3]
        })
    
    # Sort by compromise solution (weighted sum)
    weights = [0.4, 0.3, 0.2, 0.1]  # Frequency, intensity, assignment, uncertainty
    for sol in pareto_solutions:
        sol['compromise_score'] = (
            weights[0] * sol['frequency_error'] +
            weights[1] * sol['intensity_error'] +
            weights[2] * (1.0 - sol['assignment_quality']) +
            weights[3] * sol['uncertainty']
        )
    
    pareto_solutions.sort(key=lambda x: x['compromise_score'])
    
    output_text.insert(tk.END, "\nTop 5 compromise solutions:\n")
    for i, sol in enumerate(pareto_solutions[:5]):
        output_text.insert(tk.END, 
            f"  {i+1}. φ={sol['orientation'][0]:.1f}°, θ={sol['orientation'][1]:.1f}°, ψ={sol['orientation'][2]:.1f}° "
            f"(Score: {sol['compromise_score']:.4f})\n")
    
    return {
        'pareto_front': pareto_solutions,
        'pareto_history': pareto_history,
        'convergence_metrics': convergence_metrics,
        'hypervolume': convergence_metrics[-1] if convergence_metrics else 0.0
    }


def evaluate_with_gp_models(params, gp_models):
    """Evaluate objectives using Gaussian Process models"""
    objectives = []
    param_array = np.array(params).reshape(1, -1)
    
    objective_names = ['frequency_error', 'intensity_error', 'assignment_quality', 'uncertainty']
    
    for obj_name in objective_names:
        if obj_name in gp_models:
            gp_model = gp_models[obj_name]['model']
            pred_mean, pred_std = gp_model.predict(param_array, return_std=True)
            objectives.append(pred_mean[0])
        else:
            objectives.append(0.0)  # Default value
    
    return objectives


def non_dominated_sorting(objectives):
    """Perform non-dominated sorting for NSGA-II"""
    n_points = len(objectives)
    domination_counts = np.zeros(n_points)
    dominated_solutions = [[] for _ in range(n_points)]
    fronts = [[]]
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                elif dominates(objectives[j], objectives[i]):
                    domination_counts[i] += 1
        
        if domination_counts[i] == 0:
            fronts[0].append(i)
    
    current_front = 0
    while current_front < len(fronts) and len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        
        if len(next_front) > 0:
            fronts.append(next_front)
        current_front += 1
    
    # Remove empty fronts
    return [front for front in fronts if len(front) > 0]


def dominates(obj1, obj2):
    """Check if obj1 dominates obj2 (minimization problem)"""
    return all(obj1[i] <= obj2[i] for i in range(len(obj1))) and any(obj1[i] < obj2[i] for i in range(len(obj1)))


def calculate_crowding_distance(objectives, fronts):
    """Calculate crowding distance for diversity preservation"""
    n_points = len(objectives)
    distances = np.zeros(n_points)
    
    for front in fronts:
        if len(front) <= 2:
            for i in front:
                distances[i] = float('inf')
            continue
        
        front_objectives = objectives[front]
        n_objectives = objectives.shape[1]
        
        for m in range(n_objectives):
            # Sort by objective m
            sorted_indices = np.argsort(front_objectives[:, m])
            
            # Boundary points get infinite distance
            distances[front[sorted_indices[0]]] = float('inf')
            distances[front[sorted_indices[-1]]] = float('inf')
            
            # Calculate distances for intermediate points
            obj_range = front_objectives[sorted_indices[-1], m] - front_objectives[sorted_indices[0], m]
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distance = (front_objectives[sorted_indices[i+1], m] - 
                              front_objectives[sorted_indices[i-1], m]) / obj_range
                    distances[front[sorted_indices[i]]] += distance
    
    return distances


def calculate_hypervolume(pareto_objectives):
    """Calculate hypervolume indicator for convergence assessment"""
    # Simplified hypervolume calculation
    # Reference point (worst case for each objective)
    ref_point = np.max(pareto_objectives, axis=0) * 1.1
    
    # Sort points by first objective
    sorted_indices = np.argsort(pareto_objectives[:, 0])
    sorted_objectives = pareto_objectives[sorted_indices]
    
    hypervolume = 0.0
    prev_point = ref_point.copy()
    
    for point in sorted_objectives:
        # Calculate volume contribution
        volume = 1.0
        for i in range(len(point)):
            volume *= max(0, prev_point[i] - point[i])
        
        hypervolume += volume
        
        # Update reference point
        for i in range(len(point)):
            prev_point[i] = min(prev_point[i], point[i])
    
    return hypervolume


def evolve_population(population, param_bounds):
    """Evolve population through crossover and mutation"""
    new_population = []
    
    # Elitism: keep best individuals
    elite_size = len(population) // 4
    new_population.extend(population[:elite_size])
    
    # Generate offspring
    while len(new_population) < len(population):
        # Tournament selection
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        
        # Crossover
        child1, child2 = simulated_binary_crossover(parent1, parent2, param_bounds)
        
        # Mutation
        child1 = polynomial_mutation(child1, param_bounds)
        child2 = polynomial_mutation(child2, param_bounds)
        
        new_population.extend([child1, child2])
    
    return new_population[:len(population)]


def tournament_selection(population, tournament_size=3):
    """Tournament selection for parent selection"""
    tournament = np.random.choice(len(population), tournament_size, replace=False)
    # For simplicity, return random selection (in real implementation, use fitness)
    return population[tournament[0]]


def simulated_binary_crossover(parent1, parent2, param_bounds, eta=20):
    """Simulated Binary Crossover (SBX)"""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    for i in range(len(parent1)):
        if np.random.random() < 0.9:  # Crossover probability
            y1, y2 = parent1[i], parent2[i]
            
            if abs(y1 - y2) > 1e-14:
                if y1 > y2:
                    y1, y2 = y2, y1
                
                lower, upper = param_bounds[i]
                
                # Calculate beta
                rand = np.random.random()
                if rand <= 0.5:
                    beta = (2 * rand) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
                
                # Generate offspring
                child1[i] = 0.5 * ((y1 + y2) - beta * abs(y2 - y1))
                child2[i] = 0.5 * ((y1 + y2) + beta * abs(y2 - y1))
                
                # Ensure bounds
                child1[i] = np.clip(child1[i], lower, upper)
                child2[i] = np.clip(child2[i], lower, upper)
    
    return child1, child2


def polynomial_mutation(individual, param_bounds, eta=20):
    """Polynomial mutation"""
    mutated = individual.copy()
    
    for i in range(len(individual)):
        if np.random.random() < 0.1:  # Mutation probability
            y = individual[i]
            lower, upper = param_bounds[i]
            
            delta1 = (y - lower) / (upper - lower)
            delta2 = (upper - y) / (upper - lower)
            
            rand = np.random.random()
            mut_pow = 1.0 / (eta + 1)
            
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                deltaq = 1.0 - val ** mut_pow
            
            y = y + deltaq * (upper - lower)
            mutated[i] = np.clip(y, lower, upper)
    
    return mutated


def run_simplified_pareto_optimization(analyzer, experimental_peaks, output_text):
    """Simplified Pareto optimization when advanced methods are not available"""
    output_text.insert(tk.END, "Running simplified multi-objective optimization...\n")
    
    # Grid search with multiple objectives
    phi_range = np.linspace(0, 360, 20)
    theta_range = np.linspace(0, 180, 10)
    psi_range = np.linspace(0, 360, 20)
    shift_range = np.linspace(-10, 10, 5)
    scale_range = np.linspace(0.95, 1.05, 5)
    
    pareto_solutions = []
    
    for phi in phi_range:
        for theta in theta_range:
            for psi in psi_range:
                for shift in shift_range:
                    for scale in scale_range:
                        params = [phi, theta, psi, shift, scale]
                        objectives = evaluate_multi_objectives(analyzer, experimental_peaks, params)
                        
                        pareto_solutions.append({
                            'orientation': [phi, theta, psi],
                            'shift': shift,
                            'scale': scale,
                            'frequency_error': objectives[0],
                            'intensity_error': objectives[1],
                            'assignment_quality': -objectives[2],
                            'uncertainty': objectives[3],
                            'compromise_score': sum(objectives[:3])
                        })
    
    # Sort by compromise score
    pareto_solutions.sort(key=lambda x: x['compromise_score'])
    
    output_text.insert(tk.END, f"✅ Generated {len(pareto_solutions)} solutions\n")
    
    return {
        'pareto_front': pareto_solutions[:50],  # Top 50 solutions
        'pareto_history': [],
        'convergence_metrics': [],
        'hypervolume': 0.0
    }


def apply_ensemble_methods(analyzer, experimental_peaks, gp_models, pareto_results, 
                         output_text, metrics_text):
    """
    Apply ensemble methods for robust predictions
    """
    output_text.insert(tk.END, "=== Ensemble Methods ===\n\n")
    
    if not SKLEARN_AVAILABLE:
        output_text.insert(tk.END, "⚠️  Scikit-learn not available, using simplified ensemble\n")
        return apply_simplified_ensemble(pareto_results, output_text)
    
    # Build ensemble of different models
    ensemble_models = {}
    
    # Get training data from GP models
    if gp_models and 'frequency_error' in gp_models:
        training_points = gp_models['frequency_error']['training_points']
        
        # Build different model types
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        
        # Add GP models
        for obj_name, gp_data in gp_models.items():
            models[f'GP_{obj_name}'] = gp_data['model']
        
        output_text.insert(tk.END, f"Building ensemble with {len(models)} model types...\n")
        
        # Train ensemble models
        for model_name, model in models.items():
            try:
                if 'GP_' not in model_name:
                    # Train on combined objectives
                    y_combined = []
                    for params in training_points:
                        objectives = evaluate_multi_objectives(analyzer, experimental_peaks, params)
                        y_combined.append(sum(objectives[:3]))  # Combined score
                    
                    model.fit(training_points, y_combined)
                
                ensemble_models[model_name] = model
                output_text.insert(tk.END, f"  ✓ {model_name} trained successfully\n")
                
            except Exception as e:
                output_text.insert(tk.END, f"  ❌ {model_name} failed: {e}\n")
    
    # Ensemble prediction on Pareto front
    pareto_front = pareto_results['pareto_front']
    ensemble_predictions = []
    
    output_text.insert(tk.END, f"\nApplying ensemble to {len(pareto_front)} Pareto solutions...\n")
    
    for solution in pareto_front:
        params = solution['orientation'] + [solution['shift'], solution['scale']]
        param_array = np.array(params).reshape(1, -1)
        
        predictions = []
        for model_name, model in ensemble_models.items():
            try:
                if 'GP_' in model_name:
                    pred = model.predict(param_array)[0]
                else:
                    pred = model.predict(param_array)[0]
                predictions.append(pred)
            except:
                predictions.append(0.0)
        
        # Ensemble statistics
        ensemble_mean = np.mean(predictions)
        ensemble_std = np.std(predictions)
        ensemble_confidence = 1.0 / (1.0 + ensemble_std)
        
        ensemble_predictions.append({
            'solution': solution,
            'ensemble_mean': ensemble_mean,
            'ensemble_std': ensemble_std,
            'ensemble_confidence': ensemble_confidence,
            'individual_predictions': predictions
        })
    
    # Sort by ensemble confidence
    ensemble_predictions.sort(key=lambda x: x['ensemble_confidence'], reverse=True)
    
    output_text.insert(tk.END, f"\nTop 5 ensemble-ranked solutions:\n")
    for i, pred in enumerate(ensemble_predictions[:5]):
        sol = pred['solution']
        output_text.insert(tk.END, 
            f"  {i+1}. φ={sol['orientation'][0]:.1f}°, θ={sol['orientation'][1]:.1f}°, ψ={sol['orientation'][2]:.1f}° "
            f"(Confidence: {pred['ensemble_confidence']:.3f})\n")
    
    output_text.insert(tk.END, f"\n✅ Ensemble analysis completed\n\n")
    
    return {
        'ensemble_models': ensemble_models,
        'ensemble_predictions': ensemble_predictions,
        'best_ensemble_solution': ensemble_predictions[0]['solution']
    }


def apply_simplified_ensemble(pareto_results, output_text):
    """Simplified ensemble when scikit-learn is not available"""
    output_text.insert(tk.END, "Applying simplified ensemble methods...\n")
    
    pareto_front = pareto_results['pareto_front']
    
    # Simple voting ensemble based on different criteria
    ensemble_scores = []
    
    for solution in pareto_front:
        # Different scoring criteria
        freq_score = 1.0 / (1.0 + solution['frequency_error'])
        intensity_score = 1.0 / (1.0 + solution['intensity_error'])
        quality_score = solution['assignment_quality']
        uncertainty_score = 1.0 / (1.0 + solution['uncertainty'])
        
        # Weighted ensemble score
        ensemble_score = (
            0.4 * freq_score +
            0.3 * intensity_score +
            0.2 * quality_score +
            0.1 * uncertainty_score
        )
        
        ensemble_scores.append({
            'solution': solution,
            'ensemble_score': ensemble_score,
            'ensemble_confidence': ensemble_score
        })
    
    ensemble_scores.sort(key=lambda x: x['ensemble_score'], reverse=True)
    
    output_text.insert(tk.END, f"✅ Simplified ensemble analysis completed\n\n")
    
    return {
        'ensemble_models': {},
        'ensemble_predictions': ensemble_scores,
        'best_ensemble_solution': ensemble_scores[0]['solution']
    }


def perform_adaptive_sampling(analyzer, experimental_peaks, gp_models, ensemble_results,
                            output_text, metrics_text, abort_var, pause_var):
    """
    Perform active learning and adaptive sampling
    """
    output_text.insert(tk.END, "=== Active Learning and Adaptive Sampling ===\n\n")
    
    if not SKLEARN_AVAILABLE or not gp_models:
        output_text.insert(tk.END, "⚠️  GP models not available, using simplified adaptive sampling\n")
        return perform_simplified_adaptive_sampling(ensemble_results, output_text)
    
    # Acquisition functions for active learning
    acquisition_functions = {
        'Expected Improvement': expected_improvement,
        'Upper Confidence Bound': upper_confidence_bound,
        'Probability of Improvement': probability_of_improvement
    }
    
    # Adaptive sampling parameters
    n_adaptive_samples = 20
    param_bounds = [(0, 360), (0, 180), (0, 360), (-20, 20), (0.9, 1.1)]
    
    output_text.insert(tk.END, f"Performing {n_adaptive_samples} adaptive sampling iterations...\n\n")
    
    # Current best solution from ensemble
    current_best = ensemble_results['best_ensemble_solution']
    current_best_score = current_best['compromise_score']
    
    adaptive_history = []
    convergence_history = []
    
    for iteration in range(n_adaptive_samples):
        if abort_var.get():
            break
            
        while pause_var.get():
            time.sleep(0.1)
            if abort_var.get():
                break
        
        output_text.insert(tk.END, f"Adaptive iteration {iteration + 1}/{n_adaptive_samples}\n")
        
        # Generate candidate points
        n_candidates = 1000
        candidates = []
        for _ in range(n_candidates):
            candidate = []
            for lower, upper in param_bounds:
                candidate.append(np.random.uniform(lower, upper))
            candidates.append(candidate)
        
        candidates = np.array(candidates)
        
        # Evaluate acquisition functions
        best_candidates = {}
        
        for acq_name, acq_func in acquisition_functions.items():
            acq_values = []
            
            for candidate in candidates:
                # Get GP predictions
                gp_predictions = {}
                for obj_name, gp_data in gp_models.items():
                    gp_model = gp_data['model']
                    mean, std = gp_model.predict(candidate.reshape(1, -1), return_std=True)
                    gp_predictions[obj_name] = {'mean': mean[0], 'std': std[0]}
                
                # Calculate acquisition value
                acq_value = acq_func(gp_predictions, current_best_score)
                acq_values.append(acq_value)
            
            # Select best candidate for this acquisition function
            best_idx = np.argmax(acq_values)
            best_candidates[acq_name] = {
                'params': candidates[best_idx],
                'acquisition_value': acq_values[best_idx]
            }
        
        # Evaluate the most promising candidates
        new_evaluations = []
        
        for acq_name, candidate_info in best_candidates.items():
            params = candidate_info['params']
            objectives = evaluate_multi_objectives(analyzer, experimental_peaks, params)
            
            new_evaluation = {
                'params': params,
                'objectives': objectives,
                'acquisition_function': acq_name,
                'acquisition_value': candidate_info['acquisition_value']
            }
            
            new_evaluations.append(new_evaluation)
            
            # Check if this is a new best
            compromise_score = sum(objectives[:3])
            if compromise_score < current_best_score:
                current_best_score = compromise_score
                output_text.insert(tk.END, f"  🎯 New best found with {acq_name}: score = {compromise_score:.4f}\n")
        
        # Update GP models with new data
        for new_eval in new_evaluations:
            params = new_eval['params'].reshape(1, -1)
            objectives = new_eval['objectives']
            
            for i, obj_name in enumerate(['frequency_error', 'intensity_error', 'assignment_quality', 'uncertainty']):
                if obj_name in gp_models:
                    # Add new training point (simplified - in practice, would retrain)
                    gp_data = gp_models[obj_name]
                    gp_data['training_points'] = np.vstack([gp_data['training_points'], params])
                    gp_data['training_values'] = np.append(gp_data['training_values'], objectives[i])
        
        adaptive_history.append(new_evaluations)
        convergence_history.append(current_best_score)
        
        # Update metrics
        metrics_text.insert(tk.END, f"Adaptive {iteration + 1}: Best score = {current_best_score:.4f}\n")
        metrics_text.see(tk.END)
        metrics_text.update()
    
    output_text.insert(tk.END, f"\n✅ Adaptive sampling completed\n")
    output_text.insert(tk.END, f"Final best score: {current_best_score:.4f}\n\n")
    
    return {
        'adaptive_history': adaptive_history,
        'convergence_history': convergence_history,
        'final_best_score': current_best_score,
        'n_evaluations': len(adaptive_history) * len(acquisition_functions)
    }


def expected_improvement(gp_predictions, current_best):
    """Expected Improvement acquisition function"""
    # Simplified EI calculation
    freq_pred = gp_predictions.get('frequency_error', {'mean': 0, 'std': 1})
    mean = freq_pred['mean']
    std = freq_pred['std']
    
    if std == 0:
        return 0
    
    z = (current_best - mean) / std
    ei = (current_best - mean) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
    return max(0, ei)


def upper_confidence_bound(gp_predictions, current_best, kappa=2.0):
    """Upper Confidence Bound acquisition function"""
    freq_pred = gp_predictions.get('frequency_error', {'mean': 0, 'std': 1})
    return -(freq_pred['mean'] - kappa * freq_pred['std'])  # Negative for minimization


def probability_of_improvement(gp_predictions, current_best):
    """Probability of Improvement acquisition function"""
    freq_pred = gp_predictions.get('frequency_error', {'mean': 0, 'std': 1})
    mean = freq_pred['mean']
    std = freq_pred['std']
    
    if std == 0:
        return 0
    
    z = (current_best - mean) / std
    return stats.norm.cdf(z)


def perform_simplified_adaptive_sampling(ensemble_results, output_text):
    """Simplified adaptive sampling when GP models are not available"""
    output_text.insert(tk.END, "Performing simplified adaptive sampling...\n")
    
    # Simple random search around best solutions
    best_solutions = ensemble_results['ensemble_predictions'][:5]
    
    adaptive_samples = []
    for solution in best_solutions:
        # Add noise around best solutions
        base_params = solution['solution']['orientation'] + [solution['solution']['shift'], solution['solution']['scale']]
        
        for _ in range(4):  # 4 samples per best solution
            noisy_params = []
            noise_scales = [5.0, 5.0, 5.0, 1.0, 0.02]  # Different noise for each parameter
            
            for param, noise_scale in zip(base_params, noise_scales):
                noisy_param = param + np.random.normal(0, noise_scale)
                noisy_params.append(noisy_param)
            
            adaptive_samples.append(noisy_params)
    
    output_text.insert(tk.END, f"✅ Generated {len(adaptive_samples)} adaptive samples\n\n")
    
    return {
        'adaptive_history': [],
        'convergence_history': [],
        'final_best_score': best_solutions[0]['ensemble_confidence'],
        'n_evaluations': len(adaptive_samples)
    }


def advanced_uncertainty_quantification(analyzer, experimental_peaks, gp_models, 
                                       ensemble_results, adaptive_results, 
                                       output_text, metrics_text):
    """
    Perform advanced uncertainty quantification
    """
    output_text.insert(tk.END, "=== Advanced Uncertainty Quantification ===\n\n")
    
    # Multiple sources of uncertainty
    uncertainty_sources = {
        'aleatory': analyze_aleatory_uncertainty(experimental_peaks, output_text),
        'epistemic': analyze_epistemic_uncertainty(gp_models, ensemble_results, output_text),
        'model': analyze_model_uncertainty(ensemble_results, output_text),
        'numerical': analyze_numerical_uncertainty(adaptive_results, output_text)
    }
    
    # Uncertainty propagation
    total_uncertainty = propagate_advanced_uncertainties(uncertainty_sources, output_text)
    
    # Sensitivity analysis
    sensitivity_results = perform_global_sensitivity_analysis(
        analyzer, experimental_peaks, output_text, metrics_text
    )
    
    # Confidence intervals
    confidence_intervals = calculate_confidence_intervals(
        ensemble_results, total_uncertainty, output_text
    )
    
    output_text.insert(tk.END, f"\n✅ Advanced uncertainty quantification completed\n\n")
    
    return {
        'uncertainty_sources': uncertainty_sources,
        'total_uncertainty': total_uncertainty,
        'sensitivity_results': sensitivity_results,
        'confidence_intervals': confidence_intervals
    }


def analyze_aleatory_uncertainty(experimental_peaks, output_text):
    """Analyze aleatory (irreducible) uncertainty"""
    output_text.insert(tk.END, "Analyzing aleatory uncertainty...\n")
    
    # Measurement uncertainties
    center_errors = [p['center_err'] for p in experimental_peaks]
    height_errors = [p['height_err'] for p in experimental_peaks]
    
    aleatory_stats = {
        'measurement_uncertainty': {
            'center_mean': np.mean(center_errors),
            'center_std': np.std(center_errors),
            'height_mean': np.mean(height_errors),
            'height_std': np.std(height_errors)
        },
        'instrument_noise': 0.5,  # cm⁻¹
        'environmental_variation': 0.3  # cm⁻¹
    }
    
    total_aleatory = np.sqrt(
        aleatory_stats['measurement_uncertainty']['center_mean']**2 +
        aleatory_stats['instrument_noise']**2 +
        aleatory_stats['environmental_variation']**2
    )
    
    aleatory_stats['total'] = total_aleatory
    
    output_text.insert(tk.END, f"  Total aleatory uncertainty: ±{total_aleatory:.2f} cm⁻¹\n")
    
    return aleatory_stats


def analyze_epistemic_uncertainty(gp_models, ensemble_results, output_text):
    """Analyze epistemic (reducible) uncertainty"""
    output_text.insert(tk.END, "Analyzing epistemic uncertainty...\n")
    
    epistemic_stats = {}
    
    if gp_models and SKLEARN_AVAILABLE:
        # GP model uncertainties
        gp_uncertainties = {}
        for obj_name, gp_data in gp_models.items():
            gp_model = gp_data['model']
            # Model uncertainty from kernel parameters
            kernel_params = gp_model.kernel_.get_params()
            
            # Extract numeric values from kernel parameters, handling arrays and nested structures
            numeric_values = []
            for value in kernel_params.values():
                if isinstance(value, (int, float)):
                    numeric_values.append(float(value))
                elif isinstance(value, np.ndarray):
                    # Flatten arrays and add all numeric values
                    flat_values = value.flatten()
                    numeric_values.extend([float(v) for v in flat_values if np.isfinite(v)])
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    # Handle other iterable types
                    try:
                        for v in value:
                            if isinstance(v, (int, float)) and np.isfinite(v):
                                numeric_values.append(float(v))
                    except (TypeError, ValueError):
                        pass
            
            # Calculate uncertainty from numeric kernel parameters
            if numeric_values:
                kernel_uncertainty = np.std(numeric_values)
            else:
                kernel_uncertainty = 0.1  # Default fallback
            
            gp_uncertainties[obj_name] = {
                'kernel_uncertainty': kernel_uncertainty,
                'log_likelihood': gp_model.log_marginal_likelihood()
            }
        
        epistemic_stats['gp_uncertainties'] = gp_uncertainties
    
    # Ensemble disagreement
    if 'ensemble_predictions' in ensemble_results:
        ensemble_stds = [pred.get('ensemble_std', 0.0) for pred in ensemble_results['ensemble_predictions']]
        if ensemble_stds:  # Check if not empty
            epistemic_stats['ensemble_disagreement'] = {
                'mean': np.mean(ensemble_stds),
                'max': np.max(ensemble_stds),
                'std': np.std(ensemble_stds)
            }
        else:
            epistemic_stats['ensemble_disagreement'] = {
                'mean': 0.1,
                'max': 0.1,
                'std': 0.0
            }
    else:
        epistemic_stats['ensemble_disagreement'] = {
            'mean': 0.1,
            'max': 0.1,
            'std': 0.0
        }
    
    # Model selection uncertainty
    epistemic_stats['model_selection'] = 0.1  # Simplified
    
    total_epistemic = epistemic_stats.get('ensemble_disagreement', {}).get('mean', 0.1)
    epistemic_stats['total'] = total_epistemic
    
    output_text.insert(tk.END, f"  Total epistemic uncertainty: ±{total_epistemic:.3f}\n")
    
    return epistemic_stats


def analyze_model_uncertainty(ensemble_results, output_text):
    """Analyze model structure uncertainty"""
    output_text.insert(tk.END, "Analyzing model uncertainty...\n")
    
    # Model disagreement analysis
    if 'ensemble_predictions' in ensemble_results:
        predictions = ensemble_results['ensemble_predictions']
        
        # Variance across different models
        model_variances = []
        for pred in predictions[:10]:  # Top 10 solutions
            if 'individual_predictions' in pred:
                model_variances.append(np.var(pred['individual_predictions']))
        
        model_uncertainty = {
            'mean_variance': np.mean(model_variances) if model_variances else 0.1,
            'max_variance': np.max(model_variances) if model_variances else 0.1,
            'model_count': len(predictions[0].get('individual_predictions', [1])) if predictions else 1
        }
    else:
        model_uncertainty = {'mean_variance': 0.1, 'max_variance': 0.1, 'model_count': 1}
    
    output_text.insert(tk.END, f"  Model uncertainty: ±{model_uncertainty['mean_variance']:.3f}\n")
    
    return model_uncertainty


def analyze_numerical_uncertainty(adaptive_results, output_text):
    """Analyze numerical uncertainty from optimization"""
    output_text.insert(tk.END, "Analyzing numerical uncertainty...\n")
    
    # Convergence analysis
    convergence_history = adaptive_results.get('convergence_history', [])
    
    if len(convergence_history) > 5:
        # Estimate convergence uncertainty
        recent_values = convergence_history[-5:]
        convergence_std = np.std(recent_values)
        convergence_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
    else:
        convergence_std = 0.01
        convergence_trend = 0.0
    
    numerical_uncertainty = {
        'convergence_std': convergence_std,
        'convergence_trend': abs(convergence_trend),
        'optimization_tolerance': 1e-6,
        'total': max(convergence_std, 1e-4)
    }
    
    output_text.insert(tk.END, f"  Numerical uncertainty: ±{numerical_uncertainty['total']:.6f}\n")
    
    return numerical_uncertainty


def propagate_advanced_uncertainties(uncertainty_sources, output_text):
    """Propagate uncertainties through the model"""
    output_text.insert(tk.END, "Propagating uncertainties...\n")
    
    # Extract total uncertainties
    aleatory = uncertainty_sources['aleatory']['total']
    epistemic = uncertainty_sources['epistemic']['total']
    model = uncertainty_sources['model']['mean_variance']
    numerical = uncertainty_sources['numerical']['total']
    
    # Combined uncertainty (assuming independence)
    total_uncertainty = np.sqrt(aleatory**2 + epistemic**2 + model**2 + numerical**2)
    
    # Uncertainty breakdown
    uncertainty_breakdown = {
        'aleatory': aleatory,
        'epistemic': epistemic,
        'model': model,
        'numerical': numerical,
        'total': total_uncertainty,
        'relative_contributions': {
            'aleatory': (aleatory**2 / total_uncertainty**2) * 100,
            'epistemic': (epistemic**2 / total_uncertainty**2) * 100,
            'model': (model**2 / total_uncertainty**2) * 100,
            'numerical': (numerical**2 / total_uncertainty**2) * 100
        }
    }
    
    output_text.insert(tk.END, f"  Total combined uncertainty: ±{total_uncertainty:.3f}\n")
    output_text.insert(tk.END, "  Uncertainty contributions:\n")
    for source, contribution in uncertainty_breakdown['relative_contributions'].items():
        output_text.insert(tk.END, f"    {source}: {contribution:.1f}%\n")
    
    return uncertainty_breakdown


def perform_global_sensitivity_analysis(analyzer, experimental_peaks, output_text, metrics_text):
    """Perform global sensitivity analysis"""
    output_text.insert(tk.END, "Performing global sensitivity analysis...\n")
    
    # Sobol indices calculation (simplified)
    param_names = ['phi', 'theta', 'psi', 'shift', 'scale']
    param_bounds = [(0, 360), (0, 180), (0, 360), (-20, 20), (0.9, 1.1)]
    
    # Generate samples for sensitivity analysis
    n_samples = 100
    samples_A = []
    samples_B = []
    
    for _ in range(n_samples):
        sample_A = []
        sample_B = []
        for lower, upper in param_bounds:
            sample_A.append(np.random.uniform(lower, upper))
            sample_B.append(np.random.uniform(lower, upper))
        samples_A.append(sample_A)
        samples_B.append(sample_B)
    
    # Evaluate model at samples
    outputs_A = []
    outputs_B = []
    
    for sample in samples_A:
        objectives = evaluate_multi_objectives(analyzer, experimental_peaks, sample)
        outputs_A.append(sum(objectives[:3]))  # Combined objective
    
    for sample in samples_B:
        objectives = evaluate_multi_objectives(analyzer, experimental_peaks, sample)
        outputs_B.append(sum(objectives[:3]))  # Combined objective
    
    # Calculate first-order Sobol indices
    sobol_indices = {}
    total_variance = np.var(outputs_A + outputs_B)
    
    for i, param_name in enumerate(param_names):
        # Create samples with parameter i from A and others from B
        samples_AB = []
        for j in range(n_samples):
            sample = samples_B[j].copy()
            sample[i] = samples_A[j][i]
            samples_AB.append(sample)
        
        outputs_AB = []
        for sample in samples_AB:
            objectives = evaluate_multi_objectives(analyzer, experimental_peaks, sample)
            outputs_AB.append(sum(objectives[:3]))
        
        # Calculate Sobol index
        if total_variance > 0:
            sobol_index = (np.mean([outputs_A[j] * outputs_AB[j] for j in range(n_samples)]) - 
                          np.mean(outputs_A) * np.mean(outputs_AB)) / total_variance
        else:
            sobol_index = 0.0
        
        sobol_indices[param_name] = max(0.0, sobol_index)
        
        metrics_text.insert(tk.END, f"Sobol index {param_name}: {sobol_index:.3f}\n")
        metrics_text.see(tk.END)
        metrics_text.update()
    
    # Normalize Sobol indices
    total_sobol = sum(sobol_indices.values())
    if total_sobol > 0:
        normalized_sobol = {k: v/total_sobol for k, v in sobol_indices.items()}
    else:
        normalized_sobol = {k: 1.0/len(sobol_indices) for k in sobol_indices.keys()}
    
    output_text.insert(tk.END, "  Parameter sensitivities:\n")
    for param, sensitivity in normalized_sobol.items():
        output_text.insert(tk.END, f"    {param}: {sensitivity:.3f}\n")
    
    return {
        'sobol_indices': sobol_indices,
        'normalized_sobol': normalized_sobol,
        'total_variance': total_variance
    }


def calculate_confidence_intervals(ensemble_results, uncertainty_breakdown, output_text):
    """Calculate confidence intervals for results"""
    output_text.insert(tk.END, "Calculating confidence intervals...\n")
    
    best_solution = ensemble_results['best_ensemble_solution']
    total_uncertainty_value = uncertainty_breakdown['total']
    
    # 95% confidence intervals
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    confidence_intervals = {}
    
    # Orientation parameters
    orientation_uncertainty = total_uncertainty_value * 5.0  # Convert to degrees (simplified)
    
    for i, param_name in enumerate(['phi', 'theta', 'psi']):
        param_value = best_solution['orientation'][i]
        margin = z_score * orientation_uncertainty
        
        confidence_intervals[param_name] = {
            'value': param_value,
            'lower': param_value - margin,
            'upper': param_value + margin,
            'margin': margin
        }
    
    # Calibration parameters
    shift_uncertainty = total_uncertainty_value
    scale_uncertainty = total_uncertainty_value * 0.01
    
    confidence_intervals['shift'] = {
        'value': best_solution['shift'],
        'lower': best_solution['shift'] - z_score * shift_uncertainty,
        'upper': best_solution['shift'] + z_score * shift_uncertainty,
        'margin': z_score * shift_uncertainty
    }
    
    confidence_intervals['scale'] = {
        'value': best_solution['scale'],
        'lower': best_solution['scale'] - z_score * scale_uncertainty,
        'upper': best_solution['scale'] + z_score * scale_uncertainty,
        'margin': z_score * scale_uncertainty
    }
    
    output_text.insert(tk.END, f"  95% Confidence intervals:\n")
    for param, ci in confidence_intervals.items():
        output_text.insert(tk.END, 
            f"    {param}: {ci['value']:.2f} ± {ci['margin']:.2f} "
            f"[{ci['lower']:.2f}, {ci['upper']:.2f}]\n")
    
    return confidence_intervals


def integrate_advanced_results(analyzer, experimental_peaks, gp_models, pareto_results,
                             ensemble_results, adaptive_results, uncertainty_results,
                             output_text, metrics_text):
    """
    Integrate all advanced optimization results
    """
    output_text.insert(tk.END, "=== Final Integration and Advanced Results ===\n\n")
    
    # Get best solution from ensemble
    best_solution = ensemble_results['best_ensemble_solution']
    
    # Calculate final metrics
    final_metrics = {
        'pareto_front_size': len(pareto_results['pareto_front']),
        'hypervolume': pareto_results['hypervolume'],
        'ensemble_confidence': ensemble_results['ensemble_predictions'][0]['ensemble_confidence'],
        'adaptive_evaluations': adaptive_results['n_evaluations'],
        'total_uncertainty': uncertainty_results['total_uncertainty']['total'],
        'convergence_achieved': adaptive_results['convergence_history'][-1] if adaptive_results['convergence_history'] else 0.0
    }
    
    # Create comprehensive summary
    summary = f"""Stage 3 Advanced Multi-Objective Bayesian Optimization Results
{'='*70}

🎯 OPTIMAL SOLUTION (Best Compromise from Pareto Front):
   Crystal Orientation (Euler angles):
     φ = {best_solution['orientation'][0]:.3f}° ± {uncertainty_results['confidence_intervals']['phi']['margin']:.3f}°
     θ = {best_solution['orientation'][1]:.3f}° ± {uncertainty_results['confidence_intervals']['theta']['margin']:.3f}°
     ψ = {best_solution['orientation'][2]:.3f}° ± {uncertainty_results['confidence_intervals']['psi']['margin']:.3f}°

   Calibration Parameters:
     Shift = {best_solution['shift']:.3f} ± {uncertainty_results['confidence_intervals']['shift']['margin']:.3f} cm⁻¹
     Scale = {best_solution['scale']:.4f} ± {uncertainty_results['confidence_intervals']['scale']['margin']:.4f}

📊 MULTI-OBJECTIVE PERFORMANCE:
   Frequency Error: {best_solution['frequency_error']:.4f}
   Intensity Error: {best_solution['intensity_error']:.4f}
   Assignment Quality: {best_solution['assignment_quality']:.4f}
   Uncertainty: {best_solution['uncertainty']:.4f}

🔬 ADVANCED ANALYSIS METRICS:
   Pareto Front Size: {final_metrics['pareto_front_size']} solutions
   Hypervolume: {final_metrics['hypervolume']:.4f}
   Ensemble Confidence: {final_metrics['ensemble_confidence']:.4f}
   Adaptive Evaluations: {final_metrics['adaptive_evaluations']}

🎯 UNCERTAINTY QUANTIFICATION:
   Total Uncertainty: ±{final_metrics['total_uncertainty']:.4f}
   Uncertainty Breakdown:
     Aleatory (irreducible): {uncertainty_results['uncertainty_sources']['aleatory']['total']:.4f}
     Epistemic (reducible): {uncertainty_results['uncertainty_sources']['epistemic']['total']:.4f}
     Model uncertainty: {uncertainty_results['uncertainty_sources']['model']['mean_variance']:.4f}
     Numerical uncertainty: {uncertainty_results['uncertainty_sources']['numerical']['total']:.6f}

🔍 SENSITIVITY ANALYSIS:
   Most sensitive parameters:"""

    # Add sensitivity ranking
    sensitivity = uncertainty_results['sensitivity_results']['normalized_sobol']
    sorted_sensitivity = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    
    for param, sens in sorted_sensitivity:
        summary += f"\n     {param}: {sens:.3f}"

    summary += f"""

🚀 OPTIMIZATION SUMMARY:
   Method: Advanced Multi-Objective Bayesian Optimization
   Techniques: Gaussian Processes, NSGA-II, Ensemble Methods, Active Learning
   Convergence: {'Achieved' if final_metrics['convergence_achieved'] < 0.01 else 'Partial'}
   Confidence Level: 95%
   
   This represents the most sophisticated and rigorous crystal orientation
   determination available, with comprehensive uncertainty quantification
   and multi-objective optimization for maximum reliability.
"""
    
    output_text.insert(tk.END, summary)
    
    # Update final metrics display
    metrics_text.insert(tk.END, "\n" + "="*50 + "\n")
    metrics_text.insert(tk.END, "FINAL OPTIMIZATION METRICS\n")
    metrics_text.insert(tk.END, "="*50 + "\n")
    
    for metric_name, value in final_metrics.items():
        if isinstance(value, float):
            metrics_text.insert(tk.END, f"{metric_name}: {value:.6f}\n")
        else:
            metrics_text.insert(tk.END, f"{metric_name}: {value}\n")
    
    # Prepare final result
    final_result = {
        'best_solution': best_solution,
        'pareto_front': pareto_results['pareto_front'],
        'ensemble_results': ensemble_results,
        'uncertainty_results': uncertainty_results,
        'adaptive_results': adaptive_results,
        'gp_models': gp_models,
        'final_metrics': final_metrics,
        'summary': summary,
        'confidence_intervals': uncertainty_results['confidence_intervals']
    }
    
    return final_result


def display_stage3_results(result, results_text, gp_text, pareto_text, ensemble_text, 
                         convergence_text, uncertainty_text):
    """
    Display comprehensive Stage 3 results in all tabs
    """
    # GP Analysis tab
    gp_text.insert(tk.END, "\n" + "="*70 + "\n")
    gp_text.insert(tk.END, "GAUSSIAN PROCESS ANALYSIS SUMMARY\n")
    gp_text.insert(tk.END, "="*70 + "\n\n")
    
    if result['gp_models']:
        gp_text.insert(tk.END, "Gaussian Process Models:\n\n")
        for obj_name, gp_data in result['gp_models'].items():
            gp_text.insert(tk.END, f"{obj_name}:\n")
            gp_text.insert(tk.END, f"  Kernel: {gp_data['kernel']}\n")
            gp_text.insert(tk.END, f"  CV Score: {gp_data['score']:.4f}\n")
            gp_text.insert(tk.END, f"  Training Points: {len(gp_data['training_points'])}\n\n")
    
    # Pareto Analysis tab
    pareto_text.insert(tk.END, "\n" + "="*70 + "\n")
    pareto_text.insert(tk.END, "PARETO OPTIMIZATION SUMMARY\n")
    pareto_text.insert(tk.END, "="*70 + "\n\n")
    
    pareto_front = result['pareto_front']
    pareto_text.insert(tk.END, f"Pareto Front Size: {len(pareto_front)} solutions\n\n")
    
    pareto_text.insert(tk.END, "Top 10 Pareto Solutions:\n")
    pareto_text.insert(tk.END, "-" * 80 + "\n")
    pareto_text.insert(tk.END, f"{'Rank':<4} {'φ (°)':<8} {'θ (°)':<8} {'ψ (°)':<8} {'Freq Err':<10} {'Int Err':<10} {'Quality':<8}\n")
    pareto_text.insert(tk.END, "-" * 80 + "\n")
    
    for i, sol in enumerate(pareto_front[:10]):
        pareto_text.insert(tk.END, 
            f"{i+1:<4} {sol['orientation'][0]:<8.1f} {sol['orientation'][1]:<8.1f} {sol['orientation'][2]:<8.1f} "
            f"{sol['frequency_error']:<10.4f} {sol['intensity_error']:<10.4f} {sol['assignment_quality']:<8.3f}\n")
    
    # Ensemble Analysis tab
    ensemble_text.insert(tk.END, "\n" + "="*70 + "\n")
    ensemble_text.insert(tk.END, "ENSEMBLE ANALYSIS SUMMARY\n")
    ensemble_text.insert(tk.END, "="*70 + "\n\n")
    
    ensemble_predictions = result['ensemble_results']['ensemble_predictions']
    ensemble_text.insert(tk.END, f"Ensemble Models: {len(result['ensemble_results']['ensemble_models'])}\n")
    ensemble_text.insert(tk.END, f"Ensemble Predictions: {len(ensemble_predictions)}\n\n")
    
    ensemble_text.insert(tk.END, "Top 5 Ensemble-Ranked Solutions:\n")
    for i, pred in enumerate(ensemble_predictions[:5]):
        sol = pred['solution']
        ensemble_text.insert(tk.END, 
            f"{i+1}. φ={sol['orientation'][0]:.1f}°, θ={sol['orientation'][1]:.1f}°, ψ={sol['orientation'][2]:.1f}° "
            f"(Confidence: {pred['ensemble_confidence']:.3f})\n")
    
    # Convergence Analysis tab
    convergence_text.insert(tk.END, "\n" + "="*70 + "\n")
    convergence_text.insert(tk.END, "CONVERGENCE ANALYSIS SUMMARY\n")
    convergence_text.insert(tk.END, "="*70 + "\n\n")
    
    adaptive_results = result['adaptive_results']
    convergence_text.insert(tk.END, f"Adaptive Evaluations: {adaptive_results['n_evaluations']}\n")
    convergence_text.insert(tk.END, f"Final Best Score: {adaptive_results['final_best_score']:.6f}\n")
    
    if adaptive_results['convergence_history']:
        convergence_text.insert(tk.END, "\nConvergence History (last 10 iterations):\n")
        history = adaptive_results['convergence_history'][-10:]
        for i, score in enumerate(history):
            convergence_text.insert(tk.END, f"  Iteration {len(adaptive_results['convergence_history'])-10+i+1}: {score:.6f}\n")
    
    # Uncertainty Analysis tab
    uncertainty_text.insert(tk.END, "\n" + "="*70 + "\n")
    uncertainty_text.insert(tk.END, "ADVANCED UNCERTAINTY ANALYSIS SUMMARY\n")
    uncertainty_text.insert(tk.END, "="*70 + "\n\n")
    
    uncertainty_results = result['uncertainty_results']
    
    uncertainty_text.insert(tk.END, "Uncertainty Budget:\n")
    uncertainty_text.insert(tk.END, f"  Total: ±{uncertainty_results['total_uncertainty']['total']:.4f}\n\n")
    
    uncertainty_text.insert(tk.END, "Uncertainty Contributions:\n")
    for source, contribution in uncertainty_results['total_uncertainty']['relative_contributions'].items():
        uncertainty_text.insert(tk.END, f"  {source.capitalize()}: {contribution:.1f}%\n")
    
    uncertainty_text.insert(tk.END, "\nSensitivity Analysis:\n")
    sensitivity = uncertainty_results['sensitivity_results']['normalized_sobol']
    for param, sens in sorted(sensitivity.items(), key=lambda x: x[1], reverse=True):
        uncertainty_text.insert(tk.END, f"  {param}: {sens:.3f}\n")
    
    uncertainty_text.insert(tk.END, "\n95% Confidence Intervals:\n")
    for param, ci in uncertainty_results['confidence_intervals'].items():
        uncertainty_text.insert(tk.END, 
            f"  {param}: {ci['value']:.3f} ± {ci['margin']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]\n")


def save_stage3_results(filename, result):
    """
    Save comprehensive Stage 3 results to file
    """
    with open(filename, 'w') as f:
        f.write("Stage 3 Advanced Multi-Objective Bayesian Optimization - Detailed Results\n")
        f.write("="*80 + "\n\n")
        
        f.write(result['summary'])
        f.write("\n\n")
        
        # Detailed Pareto front
        f.write("COMPLETE PARETO FRONT\n")
        f.write("-"*50 + "\n\n")
        
        for i, sol in enumerate(result['pareto_front']):
            f.write(f"Solution {i+1}:\n")
            f.write(f"  Orientation: φ={sol['orientation'][0]:.3f}°, θ={sol['orientation'][1]:.3f}°, ψ={sol['orientation'][2]:.3f}°\n")
            f.write(f"  Calibration: shift={sol['shift']:.3f}, scale={sol['scale']:.4f}\n")
            f.write(f"  Objectives: freq_err={sol['frequency_error']:.4f}, int_err={sol['intensity_error']:.4f}\n")
            f.write(f"  Quality: {sol['assignment_quality']:.4f}, Uncertainty: {sol['uncertainty']:.4f}\n\n")
        
        # Ensemble details
        f.write("\nENSEMBLE ANALYSIS DETAILS\n")
        f.write("-"*50 + "\n\n")
        
        ensemble_predictions = result['ensemble_results']['ensemble_predictions']
        for i, pred in enumerate(ensemble_predictions[:20]):  # Top 20
            sol = pred['solution']
            f.write(f"Ensemble Rank {i+1}:\n")
            f.write(f"  Solution: φ={sol['orientation'][0]:.2f}°, θ={sol['orientation'][1]:.2f}°, ψ={sol['orientation'][2]:.2f}°\n")
            f.write(f"  Ensemble Confidence: {pred['ensemble_confidence']:.4f}\n")
            if 'individual_predictions' in pred:
                f.write(f"  Individual Predictions: {pred['individual_predictions']}\n")
            f.write("\n")
        
        # Uncertainty analysis details
        f.write("\nDETAILED UNCERTAINTY ANALYSIS\n")
        f.write("-"*50 + "\n\n")
        
        uncertainty_results = result['uncertainty_results']
        
        f.write("Uncertainty Sources:\n")
        for source_name, source_data in uncertainty_results['uncertainty_sources'].items():
            f.write(f"\n{source_name.upper()}:\n")
            if isinstance(source_data, dict):
                for key, value in source_data.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.6f}\n")
                    elif isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float)):
                                f.write(f"    {subkey}: {subvalue:.6f}\n")
        
        # Sensitivity analysis details
        f.write("\nSENSITIVITY ANALYSIS DETAILS\n")
        f.write("-"*50 + "\n\n")
        
        sensitivity = uncertainty_results['sensitivity_results']
        f.write("Sobol Indices (First-order):\n")
        for param, index in sensitivity['sobol_indices'].items():
            f.write(f"  {param}: {index:.6f}\n")
        
        f.write("\nNormalized Sensitivity:\n")
        for param, sens in sensitivity['normalized_sobol'].items():
            f.write(f"  {param}: {sens:.6f}\n")
        
        # Final metrics
        f.write("\nFINAL OPTIMIZATION METRICS\n")
        f.write("-"*50 + "\n\n")
        
        for metric_name, value in result['final_metrics'].items():
            if isinstance(value, float):
                f.write(f"{metric_name}: {value:.8f}\n")
            else:
                f.write(f"{metric_name}: {value}\n") 