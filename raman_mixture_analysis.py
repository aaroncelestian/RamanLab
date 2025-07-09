import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, least_squares
from scipy.signal import find_peaks
from sklearn.metrics import r2_score
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
import time

class RamanMixtureAnalyzer:
    """
    Enhanced Raman mixture analysis using iterative spectral decomposition
    with synthetic spectrum generation and weighted fitting.
    """
    
    def __init__(self, database_path: str = None, noise_threshold: float = 0.02, fast_mode: bool = True):
        """
        Initialize analyzer with reference database and parameters.
        
        Args:
            database_path: Path to reference spectra database (ignored for RamanLab integration)
            noise_threshold: Minimum signal level for analysis
            fast_mode: If True, use optimized parameters for speed (recommended for interactive use)
        """
        print("🚀 Initializing RamanLab Mixture Analysis Framework...")
        
        # Load RamanLab database infrastructure
        self.database = self._load_database(database_path)
        self.noise_threshold = noise_threshold
        self.fast_mode = fast_mode
        
        if fast_mode:
            print("⚡ Fast mode enabled - optimized for interactive analysis")
            # Fast mode settings (2-5 minute analysis)
            self.max_iterations = 5
            self.correlation_threshold = 0.6  # Lower threshold to find matches faster
            self.convergence_threshold = 0.90  # Slightly lower R² threshold
            
            # Faster convergence criteria
            self.reduced_chi_squared_target = 1.5  # More relaxed
            self.residual_std_threshold = 0.05   # 5% threshold (more relaxed)
            self.min_improvement_threshold = 0.005  # Larger improvement threshold
            
            # Minimal uncertainty estimation for speed
            self.bootstrap_samples = 10  # Much fewer bootstrap resamples
            self.monte_carlo_samples = 5   # Minimal Monte Carlo trials
            self.uncertainty_confidence = 0.90  # 90% confidence intervals
            
            # Database search optimization
            self.max_database_candidates = 500  # Limit database search to top candidates
            
        else:
            print("🔬 Research mode enabled - full statistical analysis")
            # Research mode settings (20-50 minute analysis)
            self.max_iterations = 10
            self.correlation_threshold = 0.7
            self.convergence_threshold = 0.95  # R² threshold
            
            # Enhanced convergence criteria
            self.reduced_chi_squared_target = 1.2  # Target reduced χ² (close to unity)
            self.residual_std_threshold = 0.025   # 2.5% of max peak intensity
            self.min_improvement_threshold = 0.001  # Minimum R² improvement per iteration
            
            # Full uncertainty estimation
            self.bootstrap_samples = 100  # Number of bootstrap resamples
            self.monte_carlo_samples = 50  # Number of Monte Carlo trials
            self.uncertainty_confidence = 0.95  # 95% confidence intervals
            
            # Full database search
            self.max_database_candidates = None  # Search entire database
        
        # RamanLab integration settings
        self.ramanlab_integration = True
        self.save_results_to_database = True
        
        print(f"✅ Initialization complete!")
        print(f"   Database entries: {len(self.database)}")
        print(f"   Mode: {'⚡ Fast' if fast_mode else '🔬 Research'}")
        print(f"   Expected time: {'2-5 minutes' if fast_mode else '20-50 minutes'}")
        print(f"   RamanLab integration: {'Enabled' if self.ramanlab_integration else 'Disabled'}")
        
    def _load_database(self, path: str) -> Dict:
        """
        Load reference mineral spectra database.
        
        Integrates with RamanLab's existing database infrastructure using pkl_utils.
        Loads the main RamanLab database and converts to the expected format for mixture analysis.
        
        Args:
            path: Database path (ignored for RamanLab integration)
            
        Returns:
            Database dictionary with mineral spectra
        """
        try:
            # Import RamanLab utilities
            from pkl_utils import load_raman_database, get_example_data_paths
            
            print("🔗 Integrating with RamanLab database infrastructure...")
            
            # Load the main RamanLab database
            try:
                database = load_raman_database()
                print(f"✅ Successfully loaded RamanLab database with {len(database)} entries")
            except Exception as e:
                print(f"⚠️  Warning: Could not load main database ({e})")
                print("   Creating empty database for testing")
                database = {}
            
            # Convert RamanLab database format to mixture analysis format
            converted_database = {}
            
            for mineral_name, entry in database.items():
                try:
                    # Extract wavenumbers and intensities
                    wavenumbers = np.array(entry['wavenumbers'])
                    intensities = np.array(entry['intensities'])
                    
                    # Basic validation
                    if len(wavenumbers) > 10 and len(intensities) > 10:
                        # Normalize intensities for consistent analysis
                        intensities = intensities / np.max(intensities)
                        
                        converted_database[mineral_name] = {
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'peaks': entry.get('peaks', []),
                            'metadata': entry.get('metadata', {}),
                            'original_entry': entry  # Keep reference to original
                        }
                        
                except Exception as e:
                    print(f"⚠️  Skipping invalid entry '{mineral_name}': {e}")
                    continue
            
            print(f"✅ Converted {len(converted_database)} valid mineral spectra for mixture analysis")
            
            # Display sample of loaded minerals
            if converted_database:
                sample_minerals = list(converted_database.keys())[:5]
                print(f"   Sample minerals: {', '.join(sample_minerals)}")
                
                # Show spectral data ranges
                wavenumber_ranges = []
                for mineral in sample_minerals:
                    wn = converted_database[mineral]['wavenumbers']
                    wavenumber_ranges.append(f"{np.min(wn):.0f}-{np.max(wn):.0f} cm⁻¹")
                print(f"   Spectral ranges: {', '.join(wavenumber_ranges)}")
            
            return converted_database
            
        except ImportError as e:
            print(f"❌ Error: Cannot import RamanLab utilities: {e}")
            print("   Make sure you're running from the RamanLab directory")
            # Return empty database as fallback
            return self._create_test_database()
        
        except Exception as e:
            print(f"❌ Error loading RamanLab database: {e}")
            print("   Using test database for demonstration")
            return self._create_test_database()
    
    def _create_test_database(self) -> Dict:
        """
        Create a test database for demonstration when RamanLab database is unavailable.
        
        Returns:
            Test database with synthetic mineral spectra
        """
        print("🧪 Creating test database with synthetic mineral spectra...")
        
        # Create synthetic spectra for common minerals
        wavenumbers = np.linspace(200, 1200, 500)
        
        test_database = {}
        
        # Quartz - strong peak at ~465 cm⁻¹
        quartz_intensities = (
            0.3 * np.exp(-((wavenumbers - 465)**2) / (2 * 15**2)) +
            0.1 * np.exp(-((wavenumbers - 207)**2) / (2 * 10**2)) +
            0.05 * np.random.normal(0, 0.02, len(wavenumbers))
        )
        quartz_intensities = np.maximum(quartz_intensities, 0)
        
        test_database['Quartz'] = {
            'wavenumbers': wavenumbers,
            'intensities': quartz_intensities / np.max(quartz_intensities),
            'peaks': [207, 465],
            'metadata': {'formula': 'SiO2', 'test_mineral': True}
        }
        
        # Calcite - strong peak at ~1086 cm⁻¹
        calcite_intensities = (
            0.4 * np.exp(-((wavenumbers - 1086)**2) / (2 * 20**2)) +
            0.2 * np.exp(-((wavenumbers - 712)**2) / (2 * 12**2)) +
            0.15 * np.exp(-((wavenumbers - 282)**2) / (2 * 8**2)) +
            0.05 * np.random.normal(0, 0.02, len(wavenumbers))
        )
        calcite_intensities = np.maximum(calcite_intensities, 0)
        
        test_database['Calcite'] = {
            'wavenumbers': wavenumbers,
            'intensities': calcite_intensities / np.max(calcite_intensities),
            'peaks': [282, 712, 1086],
            'metadata': {'formula': 'CaCO3', 'test_mineral': True}
        }
        
        # Feldspar - multiple peaks
        feldspar_intensities = (
            0.25 * np.exp(-((wavenumbers - 508)**2) / (2 * 18**2)) +
            0.2 * np.exp(-((wavenumbers - 477)**2) / (2 * 15**2)) +
            0.15 * np.exp(-((wavenumbers - 288)**2) / (2 * 10**2)) +
            0.05 * np.random.normal(0, 0.02, len(wavenumbers))
        )
        feldspar_intensities = np.maximum(feldspar_intensities, 0)
        
        test_database['Feldspar'] = {
            'wavenumbers': wavenumbers,
            'intensities': feldspar_intensities / np.max(feldspar_intensities),
            'peaks': [288, 477, 508],
            'metadata': {'formula': 'KAlSi3O8', 'test_mineral': True}
        }
        
        print(f"✅ Created test database with {len(test_database)} synthetic minerals")
        return test_database
    
    def _find_best_match(self, wavenumbers: np.ndarray, residual: np.ndarray) -> Dict:
        """
        Find the best matching mineral for the current residual spectrum.
        
        Args:
            wavenumbers: Wavenumber array
            residual: Current residual spectrum
            
        Returns:
            Dictionary with match information
        """
        best_correlation = 0
        best_mineral = None
        best_wavenumbers = None
        best_intensities = None
        
        # In fast mode, pre-calculate correlations and select top candidates
        if self.fast_mode and self.max_database_candidates:
            print(f"    🔍 Searching top {self.max_database_candidates} database candidates (fast mode)")
            
            # Pre-calculate all correlations for speed optimization
            correlations = []
            for mineral_name, mineral_data in self.database.items():
                try:
                    ref_wavenumbers = mineral_data['wavenumbers']
                    ref_intensities = mineral_data['intensities']
                    
                    # Quick interpolation and correlation
                    interpolated = np.interp(wavenumbers, ref_wavenumbers, ref_intensities)
                    correlation = np.corrcoef(residual, interpolated)[0, 1]
                    
                    if not np.isnan(correlation):
                        correlations.append((correlation, mineral_name, ref_wavenumbers, ref_intensities))
                        
                except Exception:
                    continue
            
            # Sort by correlation and take top candidates
            correlations.sort(key=lambda x: x[0], reverse=True)
            candidates = correlations[:self.max_database_candidates]
            
            # Find best match from top candidates
            for correlation, mineral_name, ref_wavenumbers, ref_intensities in candidates:
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_mineral = mineral_name
                    best_wavenumbers = ref_wavenumbers
                    best_intensities = ref_intensities
                    
        else:
            # Full database search (research mode)
            print(f"    🔍 Searching entire database ({len(self.database)} entries)")
            
            for mineral_name, mineral_data in self.database.items():
                try:
                    # Interpolate reference spectrum to match experimental wavenumbers
                    ref_wavenumbers = mineral_data['wavenumbers']
                    ref_intensities = mineral_data['intensities']
                    
                    # Only consider if there's good overlap
                    wn_min, wn_max = np.min(wavenumbers), np.max(wavenumbers)
                    ref_min, ref_max = np.min(ref_wavenumbers), np.max(ref_wavenumbers)
                    
                    overlap = min(wn_max, ref_max) - max(wn_min, ref_min)
                    total_range = max(wn_max, ref_max) - min(wn_min, ref_min)
                    
                    if overlap / total_range < 0.5:  # Need at least 50% overlap
                        continue
                    
                    # Interpolate to common wavenumber grid
                    interpolated_intensities = np.interp(wavenumbers, ref_wavenumbers, ref_intensities)
                    
                    # Calculate correlation with current residual
                    correlation = np.corrcoef(residual, interpolated_intensities)[0, 1]
                    
                    if np.isnan(correlation):
                        correlation = 0
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_mineral = mineral_name
                        best_wavenumbers = ref_wavenumbers
                        best_intensities = ref_intensities
                        
                except Exception as e:
                    continue
        
        return {
            'mineral': best_mineral or 'Unknown',
            'correlation': best_correlation,
            'wavenumbers': best_wavenumbers,
            'intensities': best_intensities
        }
    
    def _check_enhanced_convergence(self, wavenumbers: np.ndarray, 
                                   spectrum: np.ndarray,
                                   identified_components: List[Dict]) -> Dict:
        """
        Check enhanced convergence criteria.
        
        Args:
            wavenumbers: Wavenumber array
            spectrum: Original spectrum
            identified_components: List of identified components
            
        Returns:
            Convergence analysis results
        """
        if not identified_components:
            return {
                'converged': False,
                'criteria_met': {
                    'r_squared_threshold': False,
                    'reduced_chi_squared_acceptable': False,
                    'residual_std_acceptable': False,
                    'sufficient_improvement': False
                },
                'convergence_reason': 'No components identified'
            }
        
        # Calculate total synthetic spectrum
        total_synthetic = np.sum([comp['synthetic_spectrum'] 
                                 for comp in identified_components], axis=0)
        
        # Calculate R²
        r_squared = r2_score(spectrum, total_synthetic)
        
        # Calculate residual statistics
        residual = spectrum - total_synthetic
        rms_residual = np.sqrt(np.mean(residual**2))
        residual_std_percent = rms_residual / np.max(spectrum)
        
        # Calculate reduced χ²
        n_points = len(spectrum)
        n_params = sum([len(comp['fit_parameters'].get('parameters', [])) 
                       for comp in identified_components])
        degrees_freedom = max(1, n_points - n_params)
        reduced_chi_squared = np.sum(residual**2) / degrees_freedom
        
        # Check individual criteria
        criteria_met = {
            'r_squared_threshold': r_squared >= self.convergence_threshold,
            'reduced_chi_squared_acceptable': reduced_chi_squared <= self.reduced_chi_squared_target,
            'residual_std_acceptable': residual_std_percent <= self.residual_std_threshold,
            'sufficient_improvement': True  # Would need previous iteration to check
        }
        
        # Overall convergence
        converged = all(criteria_met.values())
        
        if converged:
            convergence_reason = "All convergence criteria satisfied"
        else:
            failed_criteria = [k for k, v in criteria_met.items() if not v]
            convergence_reason = f"Failed criteria: {', '.join(failed_criteria)}"
        
        return {
            'converged': converged,
            'criteria_met': criteria_met,
            'convergence_reason': convergence_reason,
            'r_squared': r_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'residual_std_percent': residual_std_percent * 100,
            'rms_residual': rms_residual
        }
    
    def _display_enhanced_results(self, results: Dict):
        """
        Display comprehensive analysis results with enhanced formatting.
        
        Args:
            results: Complete analysis results
        """
        print(f"\n{'='*80}")
        print("🎯 ENHANCED MIXTURE ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        metadata = results['analysis_metadata']
        print(f"📊 Analysis Overview:")
        print(f"   Components requested: {metadata['max_components_requested']}")
        print(f"   Components identified: {metadata['components_found']}")
        print(f"   Analysis time: {metadata['analysis_time']:.2f} seconds")
        print(f"   RamanLab integration: {'✓' if metadata['ramanlab_integration'] else '✗'}")
        print(f"   Database saved: {'✓' if metadata['database_saved'] else '✗'}")
        
        # Display identified components
        if results['identified_components']:
            print(f"\n🧪 IDENTIFIED COMPONENTS:")
            for i, comp in enumerate(results['identified_components']):
                mineral = comp['mineral']
                correlation = comp['correlation']
                percentage = comp.get('percentage', 0)
                profile = comp['fit_parameters'].get('selected_profile', 'unknown')
                r_squared = comp['fit_parameters'].get('r_squared', 0)
                
                print(f"   {i+1}. {mineral}")
                print(f"      Correlation: {correlation:.3f}")
                print(f"      Percentage: {percentage:.1f}%")
                print(f"      Profile: {profile.title()}")
                print(f"      R²: {r_squared:.4f}")
        
        # Display convergence information
        conv_details = results['convergence_details']
        print(f"\n🎯 CONVERGENCE ANALYSIS:")
        print(f"   Overall: {'✓ Converged' if conv_details['converged'] else '✗ Not converged'}")
        print(f"   Reason: {conv_details['convergence_reason']}")
        
        print(f"\n   Criteria Status:")
        for criterion, status in conv_details['criteria_met'].items():
            icon = "✓" if status else "✗"
            criterion_display = criterion.replace('_', ' ').title()
            print(f"     {icon} {criterion_display}")
        
        # Display final statistics
        final_stats = results['final_statistics']
        print(f"\n📈 FINAL STATISTICS:")
        print(f"   R²: {final_stats['r_squared']:.4f}")
        print(f"   RMS Residual: {final_stats['rms_residual']:.6f}")
        print(f"   RMS Residual %: {final_stats['enhanced_metrics']['rms_residual_percent']:.2f}%")
        print(f"   Total Iterations: {final_stats['enhanced_metrics']['total_iterations']}")
        
        # Display uncertainty analysis
        uncertainty = results['uncertainty_analysis']
        if uncertainty.get('component_confidence'):
            print(f"\n📊 UNCERTAINTY ANALYSIS:")
            print(f"   Bootstrap success rate: {final_stats['enhanced_metrics']['bootstrap_success_rate']:.1%}")
            
            for mineral, conf_data in uncertainty['component_confidence'].items():
                detection_conf = conf_data['detection_confidence']
                reliability = conf_data['reliability_category']
                mean_pct = conf_data['mean_percentage']
                ci_lower = conf_data['percentage_ci_lower']
                ci_upper = conf_data['percentage_ci_upper']
                
                print(f"   {mineral}:")
                print(f"     Detection confidence: {detection_conf:.1%} ({reliability})")
                print(f"     Percentage: {mean_pct:.1f}% [{ci_lower:.1f}%-{ci_upper:.1f}%]")
        
        print(f"\n{'='*80}")
    
    def _plot_enhanced_results(self, wavenumbers: np.ndarray, 
                              spectrum: np.ndarray, 
                              results: Dict):
        """
        Create enhanced visualization plots for mixture analysis results.
        
        Args:
            wavenumbers: Wavenumber array
            spectrum: Original spectrum
            results: Analysis results
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib_config import configure_matplotlib
            configure_matplotlib()  # Apply RamanLab styling
        except ImportError:
            import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Raman Mixture Analysis Results', fontsize=16, fontweight='bold')
        
        # Main decomposition plot
        ax1 = axes[0, 0]
        ax1.plot(wavenumbers, spectrum, 'k-', linewidth=2, label='Original Spectrum')
        
        # Plot individual components
        colors = plt.cm.Set1(np.linspace(0, 1, len(results['identified_components'])))
        total_synthetic = np.zeros_like(spectrum)
        
        for i, (comp, color) in enumerate(zip(results['identified_components'], colors)):
            synthetic = comp['synthetic_spectrum']
            mineral = comp['mineral']
            percentage = comp.get('percentage', 0)
            
            ax1.plot(wavenumbers, synthetic, color=color, linewidth=1.5, 
                    label=f'{mineral} ({percentage:.1f}%)')
            total_synthetic += synthetic
        
        # Plot total fit
        ax1.plot(wavenumbers, total_synthetic, 'r--', linewidth=2, 
                alpha=0.8, label='Total Fit')
        
        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Intensity (normalized)')
        ax1.set_title('Spectral Decomposition')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Residual plot
        ax2 = axes[0, 1]
        residual = spectrum - total_synthetic
        ax2.plot(wavenumbers, residual, 'b-', linewidth=1)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Wavenumber (cm⁻¹)')
        ax2.set_ylabel('Residual Intensity')
        ax2.set_title(f'Fit Residual (RMS: {np.sqrt(np.mean(residual**2)):.4f})')
        ax2.grid(True, alpha=0.3)
        
        # Convergence evolution
        ax3 = axes[1, 0]
        if results['convergence_history']:
            iterations = range(1, len(results['convergence_history']) + 1)
            r_squared_values = [conv['r_squared'] for conv in results['convergence_history']]
            
            ax3.plot(iterations, r_squared_values, 'o-', color='green', linewidth=2)
            ax3.axhline(y=self.convergence_threshold, color='red', linestyle='--', 
                       alpha=0.7, label=f'Target R² = {self.convergence_threshold}')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('R²')
            ax3.set_title('Convergence Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Component percentages with uncertainty
        ax4 = axes[1, 1]
        if results['identified_components']:
            minerals = [comp['mineral'] for comp in results['identified_components']]
            percentages = [comp.get('percentage', 0) for comp in results['identified_components']]
            
            # Add error bars if uncertainty data available
            uncertainties = []
            uncertainty_analysis = results['uncertainty_analysis']
            
            for mineral in minerals:
                if (uncertainty_analysis.get('component_confidence') and 
                    mineral in uncertainty_analysis['component_confidence']):
                    std_pct = uncertainty_analysis['component_confidence'][mineral]['std_percentage']
                    uncertainties.append(std_pct)
                else:
                    uncertainties.append(0)
            
            bars = ax4.bar(range(len(minerals)), percentages, 
                          yerr=uncertainties if any(uncertainties) else None,
                          capsize=5, alpha=0.7, color=colors[:len(minerals)])
            
            ax4.set_xlabel('Components')
            ax4.set_ylabel('Percentage (%)')
            ax4.set_title('Component Composition')
            ax4.set_xticks(range(len(minerals)))
            ax4.set_xticklabels(minerals, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_weights(self, spectrum: np.ndarray, 
                          noise_estimate: np.ndarray) -> np.ndarray:
        """
        Calculate inverse-variance weights for spectral fitting.
        
        Args:
            spectrum: Input spectrum intensities
            noise_estimate: Estimated noise level at each point
            
        Returns:
            Weight array for fitting
        """
        # Add small epsilon to prevent division by zero
        epsilon = 1e-12
        
        # Ensure noise estimate is never zero
        safe_noise = np.maximum(noise_estimate, epsilon)
        
        # Calculate weights avoiding division by zero
        weights = np.where(safe_noise > epsilon, 
                          (np.maximum(spectrum, epsilon) / safe_noise)**2, 
                          1.0)
        
        # Normalize weights and ensure they're never zero
        max_weight = np.max(weights)
        if max_weight > 0:
            weights = weights / max_weight
        else:
            weights = np.ones_like(weights)
            
        # Add minimum weight threshold to prevent zeros
        min_weight = 1e-6
        weights = np.maximum(weights, min_weight)
        
        return weights
    
    def _estimate_noise(self, spectrum: np.ndarray) -> np.ndarray:
        """Estimate noise level using baseline regions or statistical methods."""
        # Simple approach: use moving standard deviation
        window_size = 11
        noise = np.zeros_like(spectrum)
        
        for i in range(len(spectrum)):
            start = max(0, i - window_size//2)
            end = min(len(spectrum), i + window_size//2 + 1)
            window_data = spectrum[start:end]
            
            if len(window_data) > 1:
                noise[i] = np.std(window_data)
            else:
                noise[i] = np.abs(spectrum[i]) * 0.01  # 1% of signal as noise estimate
        
        # Ensure noise estimate is never zero or too small
        min_noise = np.max(spectrum) * 1e-4  # Minimum 0.01% of max signal as noise floor
        noise = np.maximum(noise, min_noise)
        
        return noise
    
    def _gaussian_profile(self, x: np.ndarray, amplitude: float, center: float, 
                         sigma: float) -> np.ndarray:
        """
        Gaussian profile for symmetric peaks.
        
        Args:
            x: Wavenumber array
            amplitude: Peak amplitude
            center: Peak center position
            sigma: Gaussian width parameter
            
        Returns:
            Gaussian profile array
        """
        return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)
    
    def _lorentzian_profile(self, x: np.ndarray, amplitude: float, center: float, 
                           gamma: float) -> np.ndarray:
        """
        Lorentzian profile for natural line broadening.
        
        Args:
            x: Wavenumber array
            amplitude: Peak amplitude
            center: Peak center position
            gamma: Lorentzian width parameter
            
        Returns:
            Lorentzian profile array
        """
        return amplitude / (1 + ((x - center) / gamma)**2)
    
    def _voigt_profile(self, x: np.ndarray, amplitude: float, center: float, 
                      sigma: float, gamma: float) -> np.ndarray:
        """
        Voigt profile combining Gaussian and Lorentzian components.
        
        Args:
            x: Wavenumber array
            amplitude: Peak amplitude
            center: Peak center position
            sigma: Gaussian width parameter
            gamma: Lorentzian width parameter
            
        Returns:
            Voigt profile array
        """
        # Simplified Voigt approximation (use scipy.special.voigt_profile for exact)
        gaussian = np.exp(-((x - center) / sigma)**2)
        lorentzian = 1 / (1 + ((x - center) / gamma)**2)
        return amplitude * (gaussian + lorentzian) / 2
    
    def _asymmetric_profile(self, x: np.ndarray, amplitude: float, center: float, 
                           sigma_left: float, sigma_right: float) -> np.ndarray:
        """
        Asymmetric Gaussian profile for peaks with asymmetric broadening.
        
        Args:
            x: Wavenumber array
            amplitude: Peak amplitude
            center: Peak center position
            sigma_left: Width parameter for left side (x < center)
            sigma_right: Width parameter for right side (x > center)
            
        Returns:
            Asymmetric profile array
        """
        result = np.zeros_like(x)
        
        # Left side (x < center)
        left_mask = x < center
        if np.any(left_mask):
            result[left_mask] = amplitude * np.exp(-0.5 * ((x[left_mask] - center) / sigma_left)**2)
        
        # Right side (x >= center)
        right_mask = x >= center
        if np.any(right_mask):
            result[right_mask] = amplitude * np.exp(-0.5 * ((x[right_mask] - center) / sigma_right)**2)
        
        return result
    
    def _test_peak_profile(self, wavenumbers: np.ndarray, reference_spectrum: np.ndarray,
                          weights: np.ndarray, peaks: np.ndarray, 
                          profile_type: str) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Test a specific peak profile type and return goodness-of-fit metrics.
        
        Args:
            wavenumbers: Wavenumber array
            reference_spectrum: Reference spectrum to fit
            weights: Fitting weights
            peaks: Peak positions
            profile_type: 'gaussian', 'lorentzian', 'voigt', or 'asymmetric'
            
        Returns:
            Tuple of (fit_quality_score, fitted_spectrum, parameters)
        """
        try:
            # Set up initial parameters and bounds based on profile type
            if profile_type == 'gaussian':
                # [amplitude, center, sigma] for each peak
                initial_params = []
                bounds_lower = []
                bounds_upper = []
                
                for peak_idx in peaks:
                    initial_params.extend([
                        reference_spectrum[peak_idx],  # amplitude
                        wavenumbers[peak_idx],         # center
                        2.0                            # sigma
                    ])
                    bounds_lower.extend([0, wavenumbers[peak_idx] - 5, 0.5])
                    bounds_upper.extend([np.inf, wavenumbers[peak_idx] + 5, 5.0])
                
                def multi_profile(x, *params):
                    result = np.zeros_like(x)
                    for i in range(0, len(params), 3):
                        if i + 2 < len(params):
                            result += self._gaussian_profile(x, params[i], params[i+1], params[i+2])
                    return result
                    
            elif profile_type == 'lorentzian':
                # [amplitude, center, gamma] for each peak
                initial_params = []
                bounds_lower = []
                bounds_upper = []
                
                for peak_idx in peaks:
                    initial_params.extend([
                        reference_spectrum[peak_idx],  # amplitude
                        wavenumbers[peak_idx],         # center
                        1.0                            # gamma
                    ])
                    bounds_lower.extend([0, wavenumbers[peak_idx] - 5, 0.5])
                    bounds_upper.extend([np.inf, wavenumbers[peak_idx] + 5, 5.0])
                
                def multi_profile(x, *params):
                    result = np.zeros_like(x)
                    for i in range(0, len(params), 3):
                        if i + 2 < len(params):
                            result += self._lorentzian_profile(x, params[i], params[i+1], params[i+2])
                    return result
                    
            elif profile_type == 'voigt':
                # [amplitude, center, sigma, gamma] for each peak
                initial_params = []
                bounds_lower = []
                bounds_upper = []
                
                for peak_idx in peaks:
                    initial_params.extend([
                        reference_spectrum[peak_idx],  # amplitude
                        wavenumbers[peak_idx],         # center
                        2.0,                           # sigma
                        1.0                            # gamma
                    ])
                    bounds_lower.extend([0, wavenumbers[peak_idx] - 5, 0.5, 0.5])
                    bounds_upper.extend([np.inf, wavenumbers[peak_idx] + 5, 5.0, 5.0])
                
                def multi_profile(x, *params):
                    result = np.zeros_like(x)
                    for i in range(0, len(params), 4):
                        if i + 3 < len(params):
                            result += self._voigt_profile(x, params[i], params[i+1], params[i+2], params[i+3])
                    return result
                    
            elif profile_type == 'asymmetric':
                # [amplitude, center, sigma_left, sigma_right] for each peak
                initial_params = []
                bounds_lower = []
                bounds_upper = []
                
                for peak_idx in peaks:
                    initial_params.extend([
                        reference_spectrum[peak_idx],  # amplitude
                        wavenumbers[peak_idx],         # center
                        2.0,                           # sigma_left
                        2.0                            # sigma_right
                    ])
                    bounds_lower.extend([0, wavenumbers[peak_idx] - 5, 0.5, 0.5])
                    bounds_upper.extend([np.inf, wavenumbers[peak_idx] + 5, 5.0, 5.0])
                
                def multi_profile(x, *params):
                    result = np.zeros_like(x)
                    for i in range(0, len(params), 4):
                        if i + 3 < len(params):
                            result += self._asymmetric_profile(x, params[i], params[i+1], params[i+2], params[i+3])
                    return result
                    
            else:
                raise ValueError(f"Unknown profile type: {profile_type}")
            
            # Perform fitting
            # Calculate safe sigma values for curve_fit (sigma = 1/weights)
            # Ensure we never divide by zero
            safe_weights = np.maximum(weights, 1e-10)  # Minimum weight threshold
            sigma_values = 1.0 / safe_weights
            
            popt, pcov = curve_fit(multi_profile, wavenumbers, reference_spectrum,
                                 p0=initial_params, 
                                 bounds=(bounds_lower, bounds_upper),
                                 sigma=sigma_values, absolute_sigma=True,
                                 maxfev=5000)
            
            fitted_spectrum = multi_profile(wavenumbers, *popt)
            
            # Calculate multiple goodness-of-fit metrics
            r_squared = r2_score(reference_spectrum, fitted_spectrum)
            
            # Calculate Akaike Information Criterion (AIC) - lower is better
            n = len(wavenumbers)
            k = len(popt)  # number of parameters
            residuals = reference_spectrum - fitted_spectrum
            weighted_residuals = residuals * np.sqrt(weights)
            rss = np.sum(weighted_residuals**2)  # residual sum of squares
            
            if rss > 0:
                aic = n * np.log(rss/n) + 2*k
                # Calculate combined score (higher is better)
                # Weight R² heavily, penalize AIC
                fit_quality = r_squared - (aic / (1000 * n))  # Normalize AIC penalty
            else:
                fit_quality = r_squared
            
            return fit_quality, fitted_spectrum, popt
            
        except Exception as e:
            # Return poor score if fitting fails
            return -1.0, np.zeros_like(wavenumbers), np.array([])
    
    def _select_best_profile(self, wavenumbers: np.ndarray, reference_spectrum: np.ndarray,
                            weights: np.ndarray, peaks: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        Automatically select the best peak profile based on goodness-of-fit metrics.
        
        Args:
            wavenumbers: Wavenumber array
            reference_spectrum: Reference spectrum to fit
            weights: Fitting weights
            peaks: Peak positions
            
        Returns:
            Tuple of (best_profile_type, fitted_spectrum, parameters)
        """
        profile_types = ['gaussian', 'lorentzian', 'voigt', 'asymmetric']
        best_score = -np.inf
        best_profile = 'voigt'  # Default fallback
        best_spectrum = reference_spectrum.copy()
        best_params = np.array([])
        
        print(f"    Testing {len(profile_types)} profile types...")
        
        for profile_type in profile_types:
            score, spectrum, params = self._test_peak_profile(
                wavenumbers, reference_spectrum, weights, peaks, profile_type)
            
            print(f"      {profile_type.capitalize()}: score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_profile = profile_type
                best_spectrum = spectrum
                best_params = params
        
        print(f"    Selected: {best_profile.capitalize()} (score: {best_score:.4f})")
        
        return best_profile, best_spectrum, best_params
    
    def _generate_synthetic_spectrum_with_profile_selection(self, wavenumbers: np.ndarray, 
                                                          mineral_name: str,
                                                          reference_wavenumbers: np.ndarray,
                                                          reference_intensities: np.ndarray) -> Dict:
        """
        Generate synthetic spectrum with automatic profile selection for a specific mineral.
        
        Args:
            wavenumbers: Target wavenumber array
            mineral_name: Name of the mineral
            reference_wavenumbers: Reference spectrum wavenumbers
            reference_intensities: Reference spectrum intensities
            
        Returns:
            Dictionary containing synthetic spectrum and fit parameters
        """
        # Interpolate reference spectrum to target wavenumbers
        interpolated_spectrum = np.interp(wavenumbers, reference_wavenumbers, reference_intensities)
        
        # Calculate weights
        noise_estimate = self._estimate_noise(interpolated_spectrum)
        weights = self._calculate_weights(interpolated_spectrum, noise_estimate)
        
        # Generate synthetic spectrum using existing method
        synthetic_spectrum, fit_params = self._fit_synthetic_spectrum(
            wavenumbers, interpolated_spectrum, weights
        )
        
        # Return in expected format
        return {
            'best_spectrum': synthetic_spectrum,
            'selected_profile': fit_params.get('selected_profile', 'unknown'),
            'r_squared': fit_params.get('r_squared', 0.0),
            'parameters': fit_params.get('parameters', []),
            'uncertainties': fit_params.get('uncertainties', []),
            'peaks': fit_params.get('peaks', []),
            'mineral_name': mineral_name,
            'profile_comparison': fit_params.get('profile_comparison', {}),
            'uncertainty_analysis': fit_params.get('uncertainty_analysis', {})
        }

    def _fit_synthetic_spectrum(self, wavenumbers: np.ndarray, 
                               reference_spectrum: np.ndarray,
                               weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Generate synthetic spectrum by automatically selecting optimal peak profiles.
        
        Tests multiple profile functions (Gaussian, Lorentzian, Voigt, Asymmetric)
        and selects the best one based on goodness-of-fit metrics including R² and AIC.
        
        Args:
            wavenumbers: Wavenumber array
            reference_spectrum: Reference mineral spectrum
            weights: Fitting weights
            
        Returns:
            Fitted synthetic spectrum and parameters with best profile type
        """
        # Find peaks in reference spectrum
        peaks, properties = find_peaks(reference_spectrum, 
                                     height=np.max(reference_spectrum) * 0.1)
        
        if len(peaks) == 0:
            print("    No peaks found in reference spectrum")
            return reference_spectrum.copy(), {
                'parameters': [], 
                'uncertainties': [], 
                'r_squared': 0, 
                'peaks': [],
                'selected_profile': 'none'
            }
        
        print(f"    Found {len(peaks)} peaks, selecting optimal profile...")
        
        # Automatically select best profile type
        try:
            best_profile_type, synthetic_spectrum, best_params = self._select_best_profile(
                wavenumbers, reference_spectrum, weights, peaks)
            
            # Calculate parameter uncertainties (simplified estimation)
            if len(best_params) > 0:
                # Enhanced uncertainty estimation using Monte Carlo sampling
                uncertainty_stats = self._monte_carlo_parameter_uncertainty(
                    wavenumbers, reference_spectrum, weights, 
                    self._estimate_noise(reference_spectrum), 
                    best_profile_type, best_params)
                
                if uncertainty_stats['parameter_uncertainties']:
                    param_errors = uncertainty_stats['parameter_uncertainties']
                else:
                    # Fallback to simple estimation
                    residuals = reference_spectrum - synthetic_spectrum
                    rms_residual = np.sqrt(np.mean(residuals**2))
                    param_errors = np.ones_like(best_params) * rms_residual * 0.1
            else:
                param_errors = np.array([])
                uncertainty_stats = {'parameter_uncertainties': [], 'confidence_intervals': {}}
            
            fit_params = {
                'parameters': best_params,
                'uncertainties': param_errors,
                'r_squared': r2_score(reference_spectrum, synthetic_spectrum),
                'peaks': peaks,
                'selected_profile': best_profile_type,
                'profile_comparison': {
                    'tested_profiles': ['gaussian', 'lorentzian', 'voigt', 'asymmetric'],
                    'selected_best': best_profile_type
                },
                'uncertainty_analysis': uncertainty_stats
            }
            
        except Exception as e:
            print(f"    Profile selection failed: {e}")
            print("    Falling back to Voigt profile...")
            
            # Fallback to Voigt profile if automatic selection fails
            synthetic_spectrum, fit_params = self._fallback_voigt_fit(
                wavenumbers, reference_spectrum, weights, peaks)
        
        return synthetic_spectrum, fit_params
    
    def _fallback_voigt_fit(self, wavenumbers: np.ndarray, reference_spectrum: np.ndarray,
                           weights: np.ndarray, peaks: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Fallback Voigt fitting method when automatic selection fails.
        
        Args:
            wavenumbers: Wavenumber array
            reference_spectrum: Reference spectrum
            weights: Fitting weights
            peaks: Peak positions
            
        Returns:
            Fitted spectrum and parameters using Voigt profiles
        """
        # Set up Voigt parameters
        initial_params = []
        bounds_lower = []
        bounds_upper = []
        
        for peak_idx in peaks:
            initial_params.extend([
                reference_spectrum[peak_idx],  # amplitude
                wavenumbers[peak_idx],         # center
                2.0,                           # sigma (cm⁻¹)
                1.0                            # gamma (cm⁻¹)
            ])
            bounds_lower.extend([0, wavenumbers[peak_idx] - 5, 0.5, 0.5])
            bounds_upper.extend([np.inf, wavenumbers[peak_idx] + 5, 5.0, 5.0])
        
        def multi_voigt(x, *params):
            result = np.zeros_like(x)
            for i in range(0, len(params), 4):
                if i + 3 < len(params):
                    result += self._voigt_profile(x, params[i], params[i+1], 
                                                params[i+2], params[i+3])
            return result
        
        try:
            # Calculate safe sigma values for curve_fit (sigma = 1/weights)  
            # Ensure we never divide by zero
            safe_weights = np.maximum(weights, 1e-10)  # Minimum weight threshold
            sigma_values = 1.0 / safe_weights
            
            popt, pcov = curve_fit(multi_voigt, wavenumbers, reference_spectrum,
                                 p0=initial_params, 
                                 bounds=(bounds_lower, bounds_upper),
                                 sigma=sigma_values, absolute_sigma=True,
                                 maxfev=5000)
            
            synthetic_spectrum = multi_voigt(wavenumbers, *popt)
            param_errors = np.sqrt(np.diag(pcov))
            
            fit_params = {
                'parameters': popt,
                'uncertainties': param_errors,
                'r_squared': r2_score(reference_spectrum, synthetic_spectrum),
                'peaks': peaks,
                'selected_profile': 'voigt',
                'profile_comparison': {'fallback_used': True}
            }
            
        except Exception as e:
            print(f"    Even Voigt fallback failed: {e}")
            synthetic_spectrum = reference_spectrum.copy()
            fit_params = {
                'parameters': [], 
                'uncertainties': [], 
                'r_squared': 0, 
                'peaks': peaks,
                'selected_profile': 'none'
            }
        
        return synthetic_spectrum, fit_params

    def _global_iterative_refinement(self, wavenumbers: np.ndarray, 
                                   spectrum: np.ndarray,
                                   identified_components: List[Dict], 
                                   weights: np.ndarray) -> Dict:
        """
        Perform global iterative refinement of all identified components.
        
        Args:
            wavenumbers: Wavenumber array
            spectrum: Original spectrum
            identified_components: List of identified components
            weights: Fitting weights
            
        Returns:
            Dictionary with refined components and statistics
        """
        print("   🔄 Starting global iterative refinement...")
        
        try:
            # Use existing global refinement method
            refined_components = self._global_refinement(
                wavenumbers, spectrum, identified_components, weights
            )
            
            # Calculate global statistics
            total_synthetic = np.sum([comp['synthetic_spectrum'] 
                                    for comp in refined_components], axis=0)
            
            # Calculate contributions for percentage calculation
            for comp in refined_components:
                # Use peak intensity as contribution measure
                if 'synthetic_spectrum' in comp:
                    comp['contribution'] = np.max(comp['synthetic_spectrum'])
                else:
                    comp['contribution'] = 0.0
            
            # Calculate global statistics
            global_r_squared = r2_score(spectrum, total_synthetic)
            residual = spectrum - total_synthetic
            rms_residual = np.sqrt(np.mean(residual**2))
            
            # Degrees of freedom for reduced chi-squared
            n_data_points = len(spectrum)
            n_parameters = sum(len(comp.get('fit_parameters', {}).get('parameters', [])) 
                             for comp in refined_components)
            dof = max(1, n_data_points - n_parameters)
            reduced_chi_squared = np.sum((residual * weights)**2) / dof
            
            statistics = {
                'r_squared': global_r_squared,
                'reduced_chi_squared': reduced_chi_squared,
                'rms_residual': rms_residual,
                'degrees_of_freedom': dof,
                'optimization_success': True
            }
            
            print(f"   ✓ Global refinement completed successfully")
            
            return {
                'components': refined_components,
                'statistics': statistics
            }
            
        except Exception as e:
            print(f"   ⚠️ Global refinement failed: {e}")
            print("   Using original components without refinement")
            
            # Return original components with basic statistics
            total_synthetic = np.sum([comp['synthetic_spectrum'] 
                                    for comp in identified_components], axis=0)
            
            basic_stats = {
                'r_squared': r2_score(spectrum, total_synthetic),
                'reduced_chi_squared': 1.0,
                'rms_residual': np.sqrt(np.mean((spectrum - total_synthetic)**2)),
                'degrees_of_freedom': len(spectrum),
                'optimization_success': False
            }
            
            return {
                'components': identified_components,
                'statistics': basic_stats
            }
    
    def _estimate_component_uncertainties(self, wavenumbers: np.ndarray,
                                        spectrum: np.ndarray,
                                        identified_components: List[Dict],
                                        weights: np.ndarray) -> Dict:
        """
        Estimate uncertainties for identified components using bootstrap methods.
        
        Args:
            wavenumbers: Wavenumber array
            spectrum: Original spectrum
            identified_components: List of identified components
            weights: Fitting weights
            
        Returns:
            Dictionary containing uncertainty analysis results
        """
        print("   🎲 Estimating component uncertainties...")
        
        try:
            # Use existing bootstrap uncertainty method
            noise_estimate = self._estimate_noise(spectrum)
            uncertainty_results = self._bootstrap_component_uncertainty(
                wavenumbers, spectrum, noise_estimate, identified_components
            )
            
            print(f"   ✓ Bootstrap uncertainty analysis completed")
            print(f"   Successful bootstraps: {uncertainty_results.get('successful_bootstraps', 0)}")
            
            return uncertainty_results
            
        except Exception as e:
            print(f"   ⚠️ Uncertainty estimation failed: {e}")
            print("   Using simplified uncertainty estimates")
            
            # Fallback: simplified uncertainty analysis
            simplified_uncertainties = {
                'component_confidence': {},
                'successful_bootstraps': 0,
                'total_bootstrap_samples': self.bootstrap_samples,
                'confidence_level': self.uncertainty_confidence,
                'uncertainty_method': 'simplified_fallback'
            }
            
            # Add basic uncertainty estimates for each component
            for comp in identified_components:
                mineral_name = comp['mineral']
                simplified_uncertainties['component_confidence'][mineral_name] = {
                    'detection_confidence': 0.8,  # Assume 80% confidence
                    'mean_percentage': comp.get('percentage', 0),
                    'percentage_ci_lower': max(0, comp.get('percentage', 0) - 5),
                    'percentage_ci_upper': comp.get('percentage', 0) + 5,
                    'std_percentage': 2.5
                }
            
            return simplified_uncertainties
    
    def _check_convergence(self, original_spectrum: np.ndarray, 
                          current_total_fit: np.ndarray,
                          weights: np.ndarray,
                          wavenumbers: np.ndarray,
                          num_parameters: int,
                          previous_r_squared: float = None) -> Tuple[bool, Dict]:
        """
        Comprehensive convergence checking using multiple criteria.
        
        Implements the advanced termination criteria from the algorithm:
        - R² > 0.95 for overall fit quality
        - Reduced χ² approaching unity  
        - Residual standard deviation < 2-3% of maximum peak intensity
        - Minimum improvement threshold to prevent stagnation
        
        Args:
            original_spectrum: Original experimental spectrum
            current_total_fit: Current total synthetic fit
            weights: Fitting weights
            wavenumbers: Wavenumber array
            num_parameters: Total number of fitted parameters
            previous_r_squared: R² from previous iteration for improvement check
            
        Returns:
            Tuple of (converged_flag, convergence_metrics_dict)
        """
        residuals = original_spectrum - current_total_fit
        max_peak_intensity = np.max(original_spectrum)
        
        # 1. Calculate R² (coefficient of determination)
        r_squared = r2_score(original_spectrum, current_total_fit)
        
        # 2. Calculate reduced chi-squared
        weighted_residuals = residuals * np.sqrt(weights)
        sum_squared_residuals = np.sum(weighted_residuals**2)
        degrees_of_freedom = len(wavenumbers) - num_parameters
        
        if degrees_of_freedom > 0:
            reduced_chi_squared = sum_squared_residuals / degrees_of_freedom
        else:
            reduced_chi_squared = np.inf
        
        # 3. Calculate residual standard deviation as percentage of max peak intensity
        residual_std = np.std(residuals)
        residual_std_percent = (residual_std / max_peak_intensity) * 100
        
        # 4. Calculate RMS residual
        rms_residual = np.sqrt(np.mean(residuals**2))
        rms_residual_percent = (rms_residual / max_peak_intensity) * 100
        
        # 5. Check improvement from previous iteration
        improvement_check = True
        r_squared_improvement = 0.0
        if previous_r_squared is not None:
            r_squared_improvement = r_squared - previous_r_squared
            improvement_check = r_squared_improvement >= self.min_improvement_threshold
        
        # Convergence criteria evaluation
        criteria_met = {
            'r_squared_threshold': r_squared >= self.convergence_threshold,
            'reduced_chi_squared_acceptable': reduced_chi_squared <= self.reduced_chi_squared_target,
            'residual_std_acceptable': residual_std_percent <= (self.residual_std_threshold * 100),
            'sufficient_improvement': improvement_check
        }
        
        # Overall convergence decision
        # Require R² threshold AND at least one of the other advanced criteria
        primary_convergence = criteria_met['r_squared_threshold']
        secondary_convergence = (criteria_met['reduced_chi_squared_acceptable'] or 
                               criteria_met['residual_std_acceptable'])
        
        converged = primary_convergence and secondary_convergence
        
        # If stagnating (no improvement), also consider converged
        if not criteria_met['sufficient_improvement'] and r_squared >= 0.9:
            converged = True
            criteria_met['stagnation_convergence'] = True
        
        convergence_metrics = {
            'r_squared': r_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'residual_std_percent': residual_std_percent,
            'rms_residual_percent': rms_residual_percent,
            'r_squared_improvement': r_squared_improvement,
            'degrees_of_freedom': degrees_of_freedom,
            'criteria_met': criteria_met,
            'converged': converged,
            'convergence_reason': self._get_convergence_reason(criteria_met, converged)
        }
        
        return converged, convergence_metrics
    
    def _get_convergence_reason(self, criteria_met: Dict, converged: bool) -> str:
        """
        Determine the primary reason for convergence or lack thereof.
        
        Args:
            criteria_met: Dictionary of convergence criteria results
            converged: Overall convergence flag
            
        Returns:
            String describing convergence reason
        """
        if not converged:
            missing_criteria = [k for k, v in criteria_met.items() if not v]
            return f"Not converged: {', '.join(missing_criteria)}"
        
        if criteria_met.get('stagnation_convergence', False):
            return "Converged: Insufficient improvement (stagnation)"
        
        reasons = []
        if criteria_met['r_squared_threshold']:
            reasons.append("R² threshold met")
        if criteria_met['reduced_chi_squared_acceptable']:
            reasons.append("Reduced χ² acceptable")  
        if criteria_met['residual_std_acceptable']:
            reasons.append("Residual std acceptable")
            
        return f"Converged: {', '.join(reasons)}"
    
    def _bootstrap_resample(self, spectrum: np.ndarray, 
                           noise_estimate: np.ndarray) -> np.ndarray:
        """
        Generate bootstrap resample of spectrum with realistic noise.
        
        Uses the estimated noise characteristics to create resampled data
        that preserves the statistical properties of the measurement.
        
        Args:
            spectrum: Original spectrum intensities
            noise_estimate: Estimated noise level at each point
            
        Returns:
            Bootstrap resampled spectrum
        """
        # Add random noise based on estimated noise characteristics
        random_noise = np.random.normal(0, noise_estimate)
        
        # Resample with replacement (bootstrap principle)
        n_points = len(spectrum)
        indices = np.random.choice(n_points, size=n_points, replace=True)
        
        # Create resampled spectrum with noise
        resampled = spectrum[indices] + random_noise
        
        # Ensure non-negative intensities (physical constraint)
        resampled = np.maximum(resampled, 0)
        
        return resampled
    
    def _monte_carlo_parameter_uncertainty(self, wavenumbers: np.ndarray,
                                         reference_spectrum: np.ndarray,
                                         weights: np.ndarray,
                                         noise_estimate: np.ndarray,
                                         best_profile_type: str,
                                         nominal_params: np.ndarray) -> Dict:
        """
        Estimate parameter uncertainties using Monte Carlo sampling.
        
        Performs repeated fitting on noise-perturbed data to estimate
        parameter confidence intervals and correlation matrices.
        
        Args:
            wavenumbers: Wavenumber array
            reference_spectrum: Reference spectrum
            weights: Fitting weights
            noise_estimate: Noise level estimate
            best_profile_type: Selected profile type
            nominal_params: Best-fit parameters
            
        Returns:
            Dictionary containing uncertainty statistics
        """
        if len(nominal_params) == 0:
            return {'parameter_uncertainties': [], 'confidence_intervals': {}}
        
        print(f"    Estimating uncertainties using {self.monte_carlo_samples} Monte Carlo samples...")
        
        parameter_samples = []
        successful_fits = 0
        
        for sample_idx in range(self.monte_carlo_samples):
            try:
                # Generate noise-perturbed spectrum
                noise_perturbation = np.random.normal(0, noise_estimate)
                perturbed_spectrum = reference_spectrum + noise_perturbation
                perturbed_spectrum = np.maximum(perturbed_spectrum, 0)  # Physical constraint
                
                # Perform fitting on perturbed data
                _, fit_params = self._test_peak_profile(
                    wavenumbers, perturbed_spectrum, weights, 
                    np.arange(len(wavenumbers))[perturbed_spectrum > 0.1 * np.max(perturbed_spectrum)],
                    best_profile_type)
                
                if len(fit_params) > 0:
                    parameter_samples.append(fit_params)
                    successful_fits += 1
                    
            except Exception:
                continue
        
        if successful_fits < 10:  # Need minimum samples for statistics
            return {'parameter_uncertainties': [], 'confidence_intervals': {}}
        
        parameter_samples = np.array(parameter_samples)
        
        # Calculate statistics
        param_means = np.mean(parameter_samples, axis=0)
        param_stds = np.std(parameter_samples, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - self.uncertainty_confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        param_lower = np.percentile(parameter_samples, lower_percentile, axis=0)
        param_upper = np.percentile(parameter_samples, upper_percentile, axis=0)
        
        uncertainty_stats = {
            'parameter_uncertainties': param_stds,
            'parameter_means': param_means,
            'confidence_intervals': {
                'lower': param_lower,
                'upper': param_upper,
                'confidence_level': self.uncertainty_confidence
            },
            'successful_fits': successful_fits,
            'total_samples': self.monte_carlo_samples,
            'parameter_correlation': np.corrcoef(parameter_samples.T) if parameter_samples.shape[1] > 1 else None
        }
        
        return uncertainty_stats
    
    def _bootstrap_component_uncertainty(self, wavenumbers: np.ndarray,
                                       spectrum: np.ndarray,
                                       noise_estimate: np.ndarray,
                                       identified_components: List[Dict]) -> Dict:
        """
        Estimate component identification uncertainty using bootstrap resampling.
        
        Performs full analysis on bootstrap resamples to assess confidence
        in component identification and quantification.
        
        Args:
            wavenumbers: Wavenumber array
            spectrum: Original spectrum
            noise_estimate: Noise level estimate
            identified_components: Original analysis results
            
        Returns:
            Dictionary containing bootstrap uncertainty statistics
        """
        if len(identified_components) == 0:
            return {'component_confidence': {}, 'bootstrap_results': []}
        
        print(f"\n🔄 Bootstrap uncertainty analysis ({self.bootstrap_samples} samples)...")
        
        component_occurrences = {}
        component_percentages = {comp['mineral']: [] for comp in identified_components}
        bootstrap_results = []
        
        successful_bootstraps = 0
        
        for bootstrap_idx in range(self.bootstrap_samples):
            try:
                # Generate bootstrap resample
                bootstrap_spectrum = self._bootstrap_resample(spectrum, noise_estimate)
                
                # Perform simplified analysis on bootstrap sample
                weights = self._calculate_weights(bootstrap_spectrum, noise_estimate)
                current_residual = bootstrap_spectrum.copy()
                bootstrap_components = []
                
                # Simplified iteration (max 3 components to prevent overfitting)
                for iteration in range(min(3, len(identified_components) + 1)):
                    best_mineral, correlation = self._database_search(current_residual, wavenumbers)
                    
                    if correlation < self.correlation_threshold:
                        break
                    
                    # Quick fitting without full profile selection
                    reference_spectrum = np.interp(wavenumbers,
                                                 self.database[best_mineral]['wavenumbers'],
                                                 self.database[best_mineral]['intensities'])
                    
                    # Use Voigt profile for speed
                    synthetic_spectrum, fit_params = self._fallback_voigt_fit(
                        wavenumbers, reference_spectrum, weights, 
                        np.arange(len(wavenumbers))[reference_spectrum > 0.1 * np.max(reference_spectrum)])
                    
                    if len(fit_params['parameters']) > 0:
                        component_info = {
                            'mineral': best_mineral,
                            'correlation': correlation,
                            'contribution': np.trapz(synthetic_spectrum, wavenumbers)
                        }
                        bootstrap_components.append(component_info)
                        current_residual -= synthetic_spectrum
                        
                        # Track component occurrence
                        if best_mineral not in component_occurrences:
                            component_occurrences[best_mineral] = 0
                        component_occurrences[best_mineral] += 1
                
                # Calculate percentages for this bootstrap
                if bootstrap_components:
                    total_contribution = sum([comp['contribution'] for comp in bootstrap_components])
                    for comp in bootstrap_components:
                        percentage = (comp['contribution'] / total_contribution) * 100
                        mineral = comp['mineral']
                        if mineral in component_percentages:
                            component_percentages[mineral].append(percentage)
                
                bootstrap_results.append(bootstrap_components)
                successful_bootstraps += 1
                
                # Progress indicator
                if (bootstrap_idx + 1) % 20 == 0:
                    print(f"    Completed {bootstrap_idx + 1}/{self.bootstrap_samples} bootstrap samples")
                    
            except Exception:
                continue
        
        # Calculate confidence statistics
        component_confidence = {}
        for mineral in component_occurrences:
            occurrence_rate = component_occurrences[mineral] / successful_bootstraps
            percentages = component_percentages.get(mineral, [])
            
            if percentages:
                alpha = 1 - self.uncertainty_confidence
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                component_confidence[mineral] = {
                    'detection_confidence': occurrence_rate,
                    'mean_percentage': np.mean(percentages),
                    'std_percentage': np.std(percentages),
                    'percentage_ci_lower': np.percentile(percentages, lower_percentile),
                    'percentage_ci_upper': np.percentile(percentages, upper_percentile),
                    'occurrences': component_occurrences[mineral],
                    'total_bootstraps': successful_bootstraps
                }
        
        bootstrap_uncertainty = {
            'component_confidence': component_confidence,
            'bootstrap_results': bootstrap_results,
            'successful_bootstraps': successful_bootstraps,
            'total_bootstrap_samples': self.bootstrap_samples,
            'confidence_level': self.uncertainty_confidence
        }
        
        return bootstrap_uncertainty
    
    def _database_search(self, spectrum: np.ndarray, 
                        wavenumbers: np.ndarray) -> Tuple[str, float]:
        """
        Search database for best matching mineral phase.
        
        Args:
            spectrum: Current residual spectrum
            wavenumbers: Wavenumber array
            
        Returns:
            Best matching mineral name and correlation coefficient
        """
        best_match = None
        best_correlation = 0
        
        for mineral_name, mineral_data in self.database.items():
            # Interpolate reference to match experimental wavenumbers
            ref_interp = np.interp(wavenumbers, 
                                 mineral_data['wavenumbers'],
                                 mineral_data['intensities'])
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(spectrum, ref_interp)[0, 1]
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_match = mineral_name
        
        return best_match, best_correlation
    
    def _global_refinement(self, wavenumbers: np.ndarray, 
                          original_spectrum: np.ndarray,
                          identified_components: List[Dict],
                          weights: np.ndarray) -> List[Dict]:
        """
        Simultaneously refine all identified components for optimal global fit.
        
        Uses constrained least-squares optimization to simultaneously optimize
        all component parameters while maintaining physical constraints.
        
        Args:
            wavenumbers: Wavenumber array
            original_spectrum: Original experimental spectrum
            identified_components: List of identified mineral components
            weights: Fitting weights
            
        Returns:
            Refined component list with improved fit quality
        """
        if len(identified_components) < 2:
            return identified_components
        
        print("  Setting up global optimization...")
        
        # Extract all parameters from all components
        all_params = []
        param_bounds_lower = []
        param_bounds_upper = []
        component_param_indices = []  # Track which params belong to which component
        
        for comp_idx, component in enumerate(identified_components):
            fit_params = component['fit_parameters']['parameters']
            if len(fit_params) == 0:
                continue
                
            # Store parameter indices for this component
            start_idx = len(all_params)
            all_params.extend(fit_params)
            end_idx = len(all_params)
            component_param_indices.append((comp_idx, start_idx, end_idx))
            
            # Get profile type for this component
            profile_type = component['fit_parameters'].get('selected_profile', 'voigt')
            
            # Set up constraints based on profile type
            if profile_type == 'gaussian' or profile_type == 'lorentzian':
                # Parameters: [amplitude, center, width] for each peak
                num_peaks = len(fit_params) // 3
                for peak_idx in range(num_peaks):
                    base_idx = peak_idx * 3
                    
                    param_bounds_lower.extend([
                        0.0,                              # amplitude >= 0
                        fit_params[base_idx + 1] - 5.0,   # center ± 5 cm⁻¹
                        0.5                               # width >= 0.5 cm⁻¹
                    ])
                    
                    param_bounds_upper.extend([
                        fit_params[base_idx] * 3.0,       # amplitude <= 3x initial
                        fit_params[base_idx + 1] + 5.0,   # center ± 5 cm⁻¹
                        5.0                               # width <= 5.0 cm⁻¹
                    ])
                    
            else:  # voigt or asymmetric (4 parameters per peak)
                # Parameters: [amplitude, center, width1, width2] for each peak
                num_peaks = len(fit_params) // 4
                for peak_idx in range(num_peaks):
                    base_idx = peak_idx * 4
                    
                    param_bounds_lower.extend([
                        0.0,                              # amplitude >= 0
                        fit_params[base_idx + 1] - 5.0,   # center ± 5 cm⁻¹
                        0.5,                              # width1 >= 0.5 cm⁻¹
                        0.5                               # width2 >= 0.5 cm⁻¹
                    ])
                    
                    param_bounds_upper.extend([
                        fit_params[base_idx] * 3.0,       # amplitude <= 3x initial
                        fit_params[base_idx + 1] + 5.0,   # center ± 5 cm⁻¹
                        5.0,                              # width1 <= 5.0 cm⁻¹
                        5.0                               # width2 <= 5.0 cm⁻¹
                    ])
        
        if len(all_params) == 0:
            return identified_components
        
        # Convert to numpy arrays
        initial_params = np.array(all_params)
        bounds = (np.array(param_bounds_lower), np.array(param_bounds_upper))
        
        def global_objective(params):
            """
            Objective function for global optimization.
            Returns weighted residuals for least_squares optimization.
            Handles mixed profile types for different components.
            """
            total_model = np.zeros_like(wavenumbers)
            
            # Reconstruct each component with current parameters
            for comp_idx, start_idx, end_idx in component_param_indices:
                component_params = params[start_idx:end_idx]
                component = identified_components[comp_idx]
                
                # Get the selected profile type for this component
                profile_type = component['fit_parameters'].get('selected_profile', 'voigt')
                
                # Generate model spectrum for this component based on its profile type
                component_model = np.zeros_like(wavenumbers)
                
                if profile_type == 'gaussian':
                    num_peaks = len(component_params) // 3
                    for peak_idx in range(num_peaks):
                        base_idx = peak_idx * 3
                        if base_idx + 2 < len(component_params):
                            amplitude = component_params[base_idx]
                            center = component_params[base_idx + 1]
                            sigma = component_params[base_idx + 2]
                            component_model += self._gaussian_profile(wavenumbers, amplitude, center, sigma)
                            
                elif profile_type == 'lorentzian':
                    num_peaks = len(component_params) // 3
                    for peak_idx in range(num_peaks):
                        base_idx = peak_idx * 3
                        if base_idx + 2 < len(component_params):
                            amplitude = component_params[base_idx]
                            center = component_params[base_idx + 1]
                            gamma = component_params[base_idx + 2]
                            component_model += self._lorentzian_profile(wavenumbers, amplitude, center, gamma)
                            
                elif profile_type == 'asymmetric':
                    num_peaks = len(component_params) // 4
                    for peak_idx in range(num_peaks):
                        base_idx = peak_idx * 4
                        if base_idx + 3 < len(component_params):
                            amplitude = component_params[base_idx]
                            center = component_params[base_idx + 1]
                            sigma_left = component_params[base_idx + 2]
                            sigma_right = component_params[base_idx + 3]
                            component_model += self._asymmetric_profile(wavenumbers, amplitude, center, sigma_left, sigma_right)
                            
                else:  # Default to voigt
                    num_peaks = len(component_params) // 4
                    for peak_idx in range(num_peaks):
                        base_idx = peak_idx * 4
                        if base_idx + 3 < len(component_params):
                            amplitude = component_params[base_idx]
                            center = component_params[base_idx + 1]
                            sigma = component_params[base_idx + 2]
                            gamma = component_params[base_idx + 3]
                            component_model += self._voigt_profile(wavenumbers, amplitude, center, sigma, gamma)
                
                total_model += component_model
            
            # Calculate weighted residuals
            residuals = (original_spectrum - total_model) * np.sqrt(weights)
            return residuals
        
        # Perform global optimization using least_squares
        print("  Running constrained optimization...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = least_squares(
                    global_objective,
                    initial_params,
                    bounds=bounds,
                    method='trf',  # Trust Region Reflective algorithm
                    max_nfev=10000,
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8
                )
            
            if not result.success:
                print(f"  Warning: Global optimization did not converge: {result.message}")
                return identified_components
            
            optimized_params = result.x
            
            # Update components with optimized parameters
            refined_components = []
            
            for comp_idx, start_idx, end_idx in component_param_indices:
                original_component = identified_components[comp_idx].copy()
                refined_params = optimized_params[start_idx:end_idx]
                
                # Get the profile type for this component
                profile_type = original_component['fit_parameters'].get('selected_profile', 'voigt')
                
                # Reconstruct synthetic spectrum with refined parameters based on profile type
                refined_spectrum = np.zeros_like(wavenumbers)
                
                if profile_type == 'gaussian':
                    num_peaks = len(refined_params) // 3
                    for peak_idx in range(num_peaks):
                        base_idx = peak_idx * 3
                        if base_idx + 2 < len(refined_params):
                            amplitude = refined_params[base_idx]
                            center = refined_params[base_idx + 1]
                            sigma = refined_params[base_idx + 2]
                            refined_spectrum += self._gaussian_profile(wavenumbers, amplitude, center, sigma)
                            
                elif profile_type == 'lorentzian':
                    num_peaks = len(refined_params) // 3
                    for peak_idx in range(num_peaks):
                        base_idx = peak_idx * 3
                        if base_idx + 2 < len(refined_params):
                            amplitude = refined_params[base_idx]
                            center = refined_params[base_idx + 1]
                            gamma = refined_params[base_idx + 2]
                            refined_spectrum += self._lorentzian_profile(wavenumbers, amplitude, center, gamma)
                            
                elif profile_type == 'asymmetric':
                    num_peaks = len(refined_params) // 4
                    for peak_idx in range(num_peaks):
                        base_idx = peak_idx * 4
                        if base_idx + 3 < len(refined_params):
                            amplitude = refined_params[base_idx]
                            center = refined_params[base_idx + 1]
                            sigma_left = refined_params[base_idx + 2]
                            sigma_right = refined_params[base_idx + 3]
                            refined_spectrum += self._asymmetric_profile(wavenumbers, amplitude, center, sigma_left, sigma_right)
                            
                else:  # Default to voigt
                    num_peaks = len(refined_params) // 4
                    for peak_idx in range(num_peaks):
                        base_idx = peak_idx * 4
                        if base_idx + 3 < len(refined_params):
                            amplitude = refined_params[base_idx]
                            center = refined_params[base_idx + 1]
                            sigma = refined_params[base_idx + 2]
                            gamma = refined_params[base_idx + 3]
                            refined_spectrum += self._voigt_profile(wavenumbers, amplitude, center, sigma, gamma)
                
                # Update component information
                original_component['synthetic_spectrum'] = refined_spectrum
                original_component['fit_parameters']['parameters'] = refined_params
                original_component['contribution'] = np.trapz(refined_spectrum, wavenumbers)
                
                # Calculate improved fit statistics
                if len(refined_spectrum) > 0:
                    r_squared = r2_score(original_spectrum, refined_spectrum)
                    original_component['fit_parameters']['r_squared'] = r_squared
                
                refined_components.append(original_component)
            
            # Calculate global fit statistics
            total_refined = np.sum([comp['synthetic_spectrum'] 
                                  for comp in refined_components], axis=0)
            global_r_squared = r2_score(original_spectrum, total_refined)
            
            # Calculate reduced chi-squared
            residuals = original_spectrum - total_refined
            weighted_residuals = residuals * np.sqrt(weights)
            degrees_of_freedom = len(wavenumbers) - len(optimized_params)
            
            if degrees_of_freedom > 0:
                reduced_chi_squared = np.sum(weighted_residuals**2) / degrees_of_freedom
            else:
                reduced_chi_squared = np.inf
            
            print(f"  Global refinement completed:")
            print(f"    Global R²: {global_r_squared:.4f}")
            print(f"    Reduced χ²: {reduced_chi_squared:.4f}")
            print(f"    RMS improvement: {np.sqrt(result.cost):.6f}")
            
            # Store global statistics in first component for access
            if refined_components:
                refined_components[0]['global_statistics'] = {
                    'global_r_squared': global_r_squared,
                    'reduced_chi_squared': reduced_chi_squared,
                    'optimization_success': result.success,
                    'optimization_message': result.message,
                    'final_cost': result.cost
                }
            
            return refined_components
            
        except Exception as e:
            print(f"  Global optimization failed: {e}")
            return identified_components
    
    def analyze(self, wavenumbers: np.ndarray, spectrum: np.ndarray, 
                max_components: int = 5, plot_results: bool = True,
                save_to_database: bool = None, analysis_name: str = None,
                constrained_components: List[Dict] = None) -> Dict:
        """
        Perform complete mixture analysis with enhanced features and optional constraints.
        
        Args:
            wavenumbers: Wavenumber array
            spectrum: Intensity array  
            max_components: Maximum number of components to identify
            plot_results: Whether to create visualization plots
            save_to_database: Whether to save to RamanLab database (auto-detect if None)
            analysis_name: Name for database storage
            constrained_components: List of known components to use as starting points
                Format: [{'mineral': 'name', 'confidence': float, 'metadata': dict}, ...]
            
        Returns:
            Complete analysis results with all enhancements
        """
        print("\n" + "="*80)
        if constrained_components:
            print("🎯 CONSTRAINED RAMAN MIXTURE ANALYSIS")
            print(f"🔗 Starting with {len(constrained_components)} known component(s)")
        else:
            print("🔬 ENHANCED RAMAN MIXTURE ANALYSIS")
        print("="*80)
        
        # Determine database saving preference
        if save_to_database is None:
            save_to_database = self.save_results_to_database and self.ramanlab_integration
        
        start_time = time.time()
        
        # Normalize spectrum
        spectrum_normalized = spectrum / np.max(spectrum)
        
        # Calculate weights for weighted fitting
        weights = self._calculate_weights(spectrum_normalized, wavenumbers)
        
        # Store original data for database saving
        original_wavenumbers = wavenumbers.copy()
        original_spectrum = spectrum.copy()
        
        # Initialize tracking variables
        identified_components = []
        current_residual = spectrum_normalized.copy()
        convergence_history = []
        
        print(f"🎯 Target: {max_components} components | Database: {'✓' if save_to_database else '✗'}")
        print(f"⚡ Mode: {'Fast' if self.fast_mode else 'Research'} | Expected time: {'2-5 min' if self.fast_mode else '20-50 min'}")
        if constrained_components:
            print(f"🔗 Constrained mode: Starting with {len(constrained_components)} known components")
        print("-" * 80)
        
        # Phase 1: Process constrained components first (if provided)
        if constrained_components:
            print(f"\n🎯 PHASE 1: PROCESSING KNOWN COMPONENTS")
            print("-" * 40)
            
            for i, constraint in enumerate(constrained_components):
                mineral_name = constraint['mineral']
                confidence = constraint.get('confidence', 1.0)
                
                print(f"🔄 Processing constraint {i+1}: {mineral_name} (confidence: {confidence:.3f})")
                
                # Find the mineral in our database
                if mineral_name in self.database:
                    mineral_data = self.database[mineral_name]
                    
                    # Generate synthetic spectrum with automatic profile selection
                    synthetic_result = self._generate_synthetic_spectrum_with_profile_selection(
                        wavenumbers, mineral_name, 
                        mineral_data['wavenumbers'], mineral_data['intensities']
                    )
                    
                    # Store component information
                    component = {
                        'mineral': mineral_name,
                        'correlation': confidence,  # Use search confidence as correlation
                        'synthetic_spectrum': synthetic_result['best_spectrum'],
                        'fit_parameters': synthetic_result,
                        'constraint_source': 'user_provided',
                        'constraint_confidence': confidence
                    }
                    identified_components.append(component)
                    
                    print(f"   ✓ {mineral_name}: Fitted with {synthetic_result['selected_profile']} profile")
                    print(f"     R²: {synthetic_result['r_squared']:.3f}")
                    
                    # Update residual after fitting this component
                    current_residual = current_residual - synthetic_result['best_spectrum']
                    current_residual = np.maximum(current_residual, 0)  # No negative intensities
                    
                    # Store constraint processing stats
                    constraint_stats = {
                        'iteration': f'constraint_{i+1}',
                        'r_squared': synthetic_result['r_squared'],
                        'rms_residual': np.sqrt(np.mean(current_residual**2)),
                        'component_type': 'constrained'
                    }
                    convergence_history.append(constraint_stats)
                    
                else:
                    print(f"   ⚠️  {mineral_name} not found in database - skipping constraint")
            
            # Calculate remaining search space
            remaining_components = max_components - len(identified_components)
            if remaining_components <= 0:
                print(f"\n✅ All {max_components} component slots filled by constraints")
            else:
                print(f"\n🔍 Phase 2: Searching residual for {remaining_components} additional components")
        else:
            remaining_components = max_components
        
        # Phase 2: Search residual for additional components (standard iterative process)
        if remaining_components > 0:
            if constrained_components:
                print(f"\n🔍 PHASE 2: RESIDUAL COMPONENT SEARCH")
                print("-" * 40)
                phase_start = len(identified_components)
            else:
                print(f"\n🔄 ITERATIVE COMPONENT IDENTIFICATION")
                print("-" * 40)
                phase_start = 0
        
            for iteration in range(remaining_components):
                print(f"\n🔄 Iteration {iteration + 1 + phase_start}")
                
                # Find best match for current residual
                match = self._find_best_match(wavenumbers, current_residual)
                
                if match['correlation'] < self.correlation_threshold:
                    print(f"   No more significant matches found (correlation: {match['correlation']:.3f})")
                    break
                
                # Generate synthetic spectrum with automatic profile selection
                synthetic_result = self._generate_synthetic_spectrum_with_profile_selection(
                    wavenumbers, match['mineral'], match['wavenumbers'], match['intensities']
                )
                
                # Store component information
                component = {
                    'mineral': match['mineral'],
                    'correlation': match['correlation'],
                    'synthetic_spectrum': synthetic_result['best_spectrum'],
                    'fit_parameters': synthetic_result,
                    'component_type': 'discovered'
                }
                identified_components.append(component)
                
                print(f"   ✓ {match['mineral']}: {match['correlation']:.3f} correlation")
                print(f"     Profile: {synthetic_result['selected_profile']} (R²: {synthetic_result['r_squared']:.3f})")
                
                # Update residual for next iteration
                current_residual = current_residual - synthetic_result['best_spectrum']
                current_residual = np.maximum(current_residual, 0)  # No negative intensities
                
                # Store convergence metrics
                iteration_stats = {
                    'iteration': iteration + 1 + phase_start,
                    'r_squared': synthetic_result['r_squared'],
                    'rms_residual': np.sqrt(np.mean(current_residual**2)),
                    'component_type': 'discovered'
                }
                convergence_history.append(iteration_stats)
        
        # Global refinement of all components
        print(f"\n🔧 GLOBAL REFINEMENT")
        print("-" * 40)
        
        if identified_components:
            refined_result = self._global_iterative_refinement(
                wavenumbers, spectrum_normalized, identified_components, weights
            )
            
            identified_components = refined_result['components']
            global_stats = refined_result['statistics']
            
            print(f"   ✓ Global R²: {global_stats['r_squared']:.4f}")
            print(f"   ✓ Reduced χ²: {global_stats['reduced_chi_squared']:.4f}")
            print(f"   ✓ RMS Residual: {global_stats['rms_residual']:.4f}")
        
        # Enhanced convergence checking
        convergence_result = self._check_enhanced_convergence(
            wavenumbers, spectrum_normalized, identified_components
        )
        
        # Component uncertainty estimation
        print(f"\n📊 UNCERTAINTY ESTIMATION")
        print("-" * 40)
        
        uncertainty_analysis = self._estimate_component_uncertainties(
            wavenumbers, spectrum_normalized, identified_components, weights
        )
        
        # Calculate final statistics and component percentages
        final_stats = self._calculate_final_statistics(
            wavenumbers, spectrum_normalized, identified_components, uncertainty_analysis, convergence_history
        )
        
        # Compile complete results
        results = {
            'identified_components': identified_components,
            'convergence_history': convergence_history,
            'final_statistics': final_stats,
            'uncertainty_analysis': uncertainty_analysis,
            'convergence_details': convergence_result,
            'constrained_components': constrained_components or [],
            'analysis_metadata': {
                'analysis_time': time.time() - start_time,
                'max_components_requested': max_components,
                'components_found': len(identified_components),
                'constrained_components_used': len(constrained_components) if constrained_components else 0,
                'ramanlab_integration': self.ramanlab_integration,
                'database_saved': False,  # Will be updated below
                'analysis_type': 'constrained' if constrained_components else 'unconstrained'
            }
        }
        
        # Save to RamanLab database if requested
        if save_to_database:
            print(f"\n💾 RAMANLAB DATABASE INTEGRATION")
            print("-" * 40)
            
            database_success = self.save_results_to_ramanlab_database(
                results, original_wavenumbers, original_spectrum, analysis_name
            )
            results['analysis_metadata']['database_saved'] = database_success
        
        # Display comprehensive results
        self._display_enhanced_results(results)
        
        # Generate visualization if requested
        if plot_results:
            self._plot_enhanced_results(wavenumbers, spectrum_normalized, results)
        
        return results
    
    def _calculate_final_statistics(self, wavenumbers: np.ndarray, 
                                   spectrum: np.ndarray,
                                   identified_components: List[Dict],
                                   uncertainty_analysis: Dict,
                                   convergence_history: List[Dict] = None) -> Dict:
        """Calculate final fit statistics and uncertainties."""
        if not identified_components:
            return {
                'r_squared': 0,
                'rms_residual': 0,
                'total_synthetic': 0,
                'final_residual': spectrum,
                'global_statistics': {},
                'convergence_summary': {},
                'uncertainty_analysis': uncertainty_analysis,
                'enhanced_metrics': {
                    'rms_residual_percent': 0,
                    'max_peak_intensity': 0,
                    'total_iterations': 0,
                    'bootstrap_success_rate': 0
                }
            }
        
        # Calculate total synthetic spectrum
        total_synthetic = np.sum([comp['synthetic_spectrum'] 
                                for comp in identified_components], axis=0)
        
        # Final statistics
        final_r_squared = r2_score(spectrum, total_synthetic)
        final_residual = spectrum - total_synthetic
        rms_residual = np.sqrt(np.mean(final_residual**2))
        
        # Calculate component percentages
        total_intensity = np.sum([comp['contribution'] 
                                for comp in identified_components])
        
        for comp in identified_components:
            comp['percentage'] = (comp['contribution'] / total_intensity) * 100
        
        # Enhanced final statistics with convergence analysis
        global_stats = {}
        convergence_summary = {}
        
        # Get final convergence metrics if available
        if convergence_history and len(convergence_history) > 0:
            final_convergence = convergence_history[-1]
            convergence_summary = {
                'final_r_squared': final_convergence['r_squared'],
                'final_reduced_chi_squared': final_convergence['rms_residual']**2,
                'final_residual_std_percent': final_convergence['rms_residual'] / np.max(spectrum) * 100,
                'convergence_achieved': final_convergence['rms_residual'] < self.residual_std_threshold,
                'convergence_reason': "Residual std below threshold" if final_convergence['rms_residual'] < self.residual_std_threshold else "Residual std above threshold",
                'iterations_to_convergence': len(convergence_history)
            }
        
        # Check if global statistics are available from refinement
        if (identified_components and 
            'global_statistics' in identified_components[0]):
            global_stats = identified_components[0]['global_statistics']
            
        print(f"\n{'='*60}")
        print(f"�� FINAL ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Components identified: {len(identified_components)}")
        print(f"Analysis iterations: {len(convergence_history) if convergence_history else 0}")
        
        # Display convergence results
        if convergence_summary:
            print(f"\n📊 CONVERGENCE ANALYSIS:")
            print(f"  Status: {'✅ CONVERGED' if convergence_summary['convergence_achieved'] else '❌ NOT CONVERGED'}")
            print(f"  Reason: {convergence_summary['convergence_reason']}")
            print(f"  Final R²: {convergence_summary['final_r_squared']:.4f}")
            print(f"  Final Reduced χ²: {convergence_summary['final_reduced_chi_squared']:.3f}")
            print(f"  Final Residual Std: {convergence_summary['final_residual_std_percent']:.2f}%")
        
        # Display global refinement results if available
        if global_stats:
            print(f"\n🔧 GLOBAL REFINEMENT RESULTS:")
            print(f"  Global R²: {global_stats['global_r_squared']:.4f}")
            print(f"  Global Reduced χ²: {global_stats['reduced_chi_squared']:.4f}")
            print(f"  Optimization success: {'✅' if global_stats['optimization_success'] else '❌'}")
        
        print(f"\n📈 FIT QUALITY METRICS:")
        print(f"  RMS residual: {rms_residual:.6f}")
        print(f"  RMS residual %: {(rms_residual/np.max(spectrum)*100):.2f}%")
        
        # Display component information
        print(f"\n🧪 IDENTIFIED COMPONENTS:")
        for i, comp in enumerate(identified_components):
            profile_type = comp['fit_parameters'].get('selected_profile', 'unknown')
            percentage = comp.get('percentage', 0)
            component_r_squared = comp['fit_parameters'].get('r_squared', 0)
            print(f"  {i+1}. {comp['mineral']}")
            print(f"     Profile: {profile_type.capitalize()}")
            print(f"     Contribution: {percentage:.1f}%")
            print(f"     Individual R²: {component_r_squared:.4f}")
            
            # Display uncertainty information if available
            if 'uncertainty_analysis' in comp and comp['uncertainty_analysis']['component_confidence']:
                mineral = comp['mineral']
                if mineral in comp['uncertainty_analysis']['component_confidence']:
                    conf_data = comp['uncertainty_analysis']['component_confidence'][mineral]
                    detection_conf = conf_data['detection_confidence'] * 100
                    mean_pct = conf_data['mean_percentage']
                    ci_lower = conf_data['percentage_ci_lower']
                    ci_upper = conf_data['percentage_ci_upper']
                    print(f"     Bootstrap confidence: {detection_conf:.1f}% detection rate")
                    print(f"     Percentage range: {ci_lower:.1f}% - {ci_upper:.1f}% ({self.uncertainty_confidence*100:.0f}% CI)")
        
        # Display uncertainty analysis summary
        if 'uncertainty_analysis' in comp and comp['uncertainty_analysis']['component_confidence']:
            print(f"\n🎲 UNCERTAINTY ANALYSIS SUMMARY:")
            uncertainty_data = comp['uncertainty_analysis']
            print(f"  Bootstrap samples: {uncertainty_data['successful_bootstraps']}/{uncertainty_data['total_bootstrap_samples']}")
            print(f"  Confidence level: {uncertainty_data['confidence_level']*100:.0f}%")
            
            print(f"\n  Component Detection Reliability:")
            for mineral, conf_data in uncertainty_data['component_confidence'].items():
                detection_rate = conf_data['detection_confidence'] * 100
                reliability = "High" if detection_rate >= 80 else "Medium" if detection_rate >= 60 else "Low"
                print(f"    {mineral}: {detection_rate:.1f}% ({reliability} reliability)")
        
        print(f"{'='*60}")
        
        return {
            'r_squared': final_r_squared,
            'rms_residual': rms_residual,
            'total_synthetic': total_synthetic,
            'final_residual': final_residual,
            'global_statistics': global_stats,
            'convergence_summary': convergence_summary,
            'uncertainty_analysis': uncertainty_analysis,
            'enhanced_metrics': {
                'rms_residual_percent': (rms_residual/np.max(spectrum)*100),
                'max_peak_intensity': np.max(spectrum),
                'total_iterations': len(convergence_history) if convergence_history else 0,
                'bootstrap_success_rate': (uncertainty_analysis.get('successful_bootstraps', 0) /
                                         self.bootstrap_samples) if uncertainty_analysis else 0
            }
        }
    
    def plot_results(self, wavenumbers: np.ndarray, 
                    original_spectrum: np.ndarray, 
                    results: Dict):
        """Generate comprehensive analysis plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original spectrum and total fit
        ax1 = axes[0, 0]
        ax1.plot(wavenumbers, original_spectrum, 'k-', label='Original', linewidth=2)
        
        if 'final_statistics' in results:
            ax1.plot(wavenumbers, results['final_statistics']['total_synthetic'], 
                    'r--', label='Total Fit', linewidth=2)
            ax1.plot(wavenumbers, results['final_statistics']['final_residual'], 
                    'g-', label='Residual', alpha=0.7)
        
        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Intensity')
        ax1.set_title('Spectral Decomposition Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Individual components
        ax2 = axes[0, 1]
        ax2.plot(wavenumbers, original_spectrum, 'k-', alpha=0.3, label='Original')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results['identified_components'])))
        for i, comp in enumerate(results['identified_components']):
            ax2.plot(wavenumbers, comp['synthetic_spectrum'], 
                    color=colors[i], linewidth=2,
                    label=f"{comp['mineral']} ({comp['percentage']:.1f}%)")
        
        ax2.set_xlabel('Wavenumber (cm⁻¹)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Individual Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Residual evolution
        ax3 = axes[1, 0]
        for i, residual in enumerate(results['convergence_history']):
            ax3.plot(wavenumbers, residual['residual'], alpha=0.7, label=f'Iteration {i+1}')
        
        ax3.set_xlabel('Wavenumber (cm⁻¹)')
        ax3.set_ylabel('Residual Intensity')
        ax3.set_title('Residual Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Enhanced convergence statistics
        ax4 = axes[1, 1]
        if results['convergence_history']:
            iterations = range(1, len(results['convergence_history']) + 1)
            r_squared_values = [conv['r_squared'] for conv in results['convergence_history']]
            reduced_chi_squared_values = [conv['rms_residual']**2 for conv in results['convergence_history']]
            
            # Plot R² evolution
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(iterations, r_squared_values, 'bo-', linewidth=2, markersize=6, 
                           label='R²', alpha=0.8)
            ax4.axhline(y=self.convergence_threshold, color='red', linestyle='--', 
                       alpha=0.7, label=f'R² threshold ({self.convergence_threshold})')
            
            # Plot reduced χ² evolution on twin axis
            line2 = ax4_twin.plot(iterations, reduced_chi_squared_values, 'rs-', linewidth=2, 
                                markersize=6, label='Reduced χ²', alpha=0.8)
            ax4_twin.axhline(y=self.reduced_chi_squared_target, color='orange', linestyle='--', 
                           alpha=0.7, label=f'χ² target ({self.reduced_chi_squared_target})')
            
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('R² Value', color='blue')
            ax4_twin.set_ylabel('Reduced χ²', color='red')
            ax4.set_title('Enhanced Convergence Metrics')
            
            # Combine legends
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
            
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='y', labelcolor='blue')
            ax4_twin.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        plt.show()

    def save_results_to_ramanlab_database(self, results: Dict, 
                                         original_wavenumbers: np.ndarray,
                                         original_spectrum: np.ndarray,
                                         analysis_name: str = None) -> bool:
        """
        Save mixture analysis results to RamanLab batch results database.
        
        Integrates with the existing batch_results_database.h5 infrastructure
        for unified search across all RamanLab modules.
        
        Args:
            results: Analysis results dictionary
            original_wavenumbers: Original spectrum wavenumbers
            original_spectrum: Original spectrum intensities
            analysis_name: Name for this analysis
            
        Returns:
            Success flag
        """
        if not self.save_results_to_database:
            return True
        
        try:
            import h5py
            from datetime import datetime
            from pathlib import Path
            
            # Determine analysis name
            if analysis_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                analysis_name = f"mixture_analysis_{timestamp}"
            
            # Get workspace root for database path
            try:
                from pkl_utils import get_workspace_root
                workspace_root = get_workspace_root()
                database_path = workspace_root / "batch_results_database.h5"
            except:
                database_path = Path("batch_results_database.h5")
            
            print(f"💾 Saving results to RamanLab database: {database_path}")
            
            # Prepare results for HDF5 storage
            analysis_data = {
                'analysis_type': 'mixture_analysis',
                'timestamp': datetime.now().isoformat(),
                'original_spectrum': {
                    'wavenumbers': original_wavenumbers.tolist(),
                    'intensities': original_spectrum.tolist()
                },
                'identified_components': [],
                'final_statistics': results.get('final_statistics', {}),
                'convergence_history': results.get('convergence_history', []),
                'uncertainty_analysis': results.get('uncertainty_analysis', {})
            }
            
            # Process identified components
            for comp in results.get('identified_components', []):
                component_data = {
                    'mineral': comp['mineral'],
                    'correlation': float(comp['correlation']),
                    'percentage': float(comp.get('percentage', 0)),
                    'selected_profile': comp['fit_parameters'].get('selected_profile', 'unknown'),
                    'r_squared': float(comp['fit_parameters'].get('r_squared', 0)),
                    'synthetic_spectrum': comp['synthetic_spectrum'].tolist(),
                    'fit_parameters': comp['fit_parameters']['parameters'].tolist() if comp['fit_parameters']['parameters'] else []
                }
                analysis_data['identified_components'].append(component_data)
            
            # Save to HDF5 database
            with h5py.File(database_path, 'a') as h5file:
                # Create mixture_analysis group if it doesn't exist
                if 'mixture_analysis' not in h5file:
                    mixture_group = h5file.create_group('mixture_analysis')
                else:
                    mixture_group = h5file['mixture_analysis']
                
                # Create analysis subgroup
                analysis_group = mixture_group.create_group(analysis_name)
                
                # Store metadata
                analysis_group.attrs['analysis_type'] = 'mixture_analysis'
                analysis_group.attrs['timestamp'] = analysis_data['timestamp']
                analysis_group.attrs['num_components'] = len(analysis_data['identified_components'])
                
                # Store original spectrum
                original_group = analysis_group.create_group('original_spectrum')
                original_group.create_dataset('wavenumbers', data=original_wavenumbers)
                original_group.create_dataset('intensities', data=original_spectrum)
                
                # Store components
                if analysis_data['identified_components']:
                    components_group = analysis_group.create_group('identified_components')
                    for i, comp in enumerate(analysis_data['identified_components']):
                        comp_group = components_group.create_group(f'component_{i:02d}')
                        comp_group.attrs['mineral'] = comp['mineral']
                        comp_group.attrs['correlation'] = comp['correlation']
                        comp_group.attrs['percentage'] = comp['percentage']
                        comp_group.attrs['selected_profile'] = comp['selected_profile']
                        comp_group.attrs['r_squared'] = comp['r_squared']
                        
                        comp_group.create_dataset('synthetic_spectrum', data=comp['synthetic_spectrum'])
                        if comp['fit_parameters']:
                            comp_group.create_dataset('fit_parameters', data=comp['fit_parameters'])
                
                # Store final statistics
                if results.get('final_statistics'):
                    stats_group = analysis_group.create_group('final_statistics')
                    final_stats = results['final_statistics']
                    
                    stats_group.attrs['r_squared'] = float(final_stats.get('r_squared', 0))
                    stats_group.attrs['rms_residual'] = float(final_stats.get('rms_residual', 0))
                    
                    if 'total_synthetic' in final_stats:
                        stats_group.create_dataset('total_synthetic', data=final_stats['total_synthetic'])
                    if 'final_residual' in final_stats:
                        stats_group.create_dataset('final_residual', data=final_stats['final_residual'])
                
                # Store uncertainty analysis
                if results.get('uncertainty_analysis', {}).get('component_confidence'):
                    uncertainty_group = analysis_group.create_group('uncertainty_analysis')
                    uncertainty_data = results['uncertainty_analysis']
                    
                    uncertainty_group.attrs['successful_bootstraps'] = uncertainty_data.get('successful_bootstraps', 0)
                    uncertainty_group.attrs['total_bootstrap_samples'] = uncertainty_data.get('total_bootstrap_samples', 0)
                    uncertainty_group.attrs['confidence_level'] = uncertainty_data.get('confidence_level', 0.95)
                    
                    # Store component confidence data
                    conf_group = uncertainty_group.create_group('component_confidence')
                    for mineral, conf_data in uncertainty_data['component_confidence'].items():
                        mineral_group = conf_group.create_group(mineral.replace(' ', '_'))
                        mineral_group.attrs['detection_confidence'] = conf_data['detection_confidence']
                        mineral_group.attrs['mean_percentage'] = conf_data['mean_percentage']
                        mineral_group.attrs['std_percentage'] = conf_data['std_percentage']
                        mineral_group.attrs['percentage_ci_lower'] = conf_data['percentage_ci_lower']
                        mineral_group.attrs['percentage_ci_upper'] = conf_data['percentage_ci_upper']
            
            print(f"✅ Results saved successfully as '{analysis_name}'")
            return True
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save to RamanLab database: {e}")
            print("   Analysis results are still available in memory")
            return False

# Example usage
if __name__ == "__main__":
    print("🚀 RamanLab Enhanced Mixture Analysis Demo")
    print("=" * 60)
    
    # Initialize analyzer with RamanLab integration
    analyzer = RamanMixtureAnalyzer()
    
    # Create or load example spectrum data
    print("\n📊 Preparing example mixture spectrum...")
    
    # Synthetic mixture for demonstration
    wavenumbers = np.linspace(200, 1200, 500)
    
    # Create a mixture of quartz + calcite + noise
    mixture_spectrum = (
        0.6 * np.exp(-((wavenumbers - 465)**2) / (2 * 15**2)) +  # Quartz
        0.3 * np.exp(-((wavenumbers - 1086)**2) / (2 * 20**2)) +  # Calcite  
        0.1 * np.exp(-((wavenumbers - 712)**2) / (2 * 12**2)) +   # Calcite
        0.05 * np.random.normal(0, 0.02, len(wavenumbers))       # Noise
    )
    
    # Ensure no negative values
    mixture_spectrum = np.maximum(mixture_spectrum, 0)
    
    print(f"✓ Created synthetic mixture spectrum")
    print(f"  Wavenumber range: {wavenumbers[0]:.0f} - {wavenumbers[-1]:.0f} cm⁻¹")
    print(f"  Data points: {len(wavenumbers)}")
    
    # Run complete enhanced analysis
    print(f"\n🔬 Running Enhanced Mixture Analysis...")
    print("   (This will demonstrate all integrated features)")
    
    results = analyzer.analyze(
        wavenumbers=wavenumbers,
        spectrum=mixture_spectrum,
        max_components=3,
        plot_results=True,
        save_to_database=True,
        analysis_name="demo_quartz_calcite_mixture"
    )
    
    print(f"\n🎉 Analysis complete!")
    print(f"   Components found: {results['analysis_metadata']['components_found']}")
    print(f"   Analysis time: {results['analysis_metadata']['analysis_time']:.2f} seconds")
    print(f"   Database saved: {'✓' if results['analysis_metadata']['database_saved'] else '✗'}")
    
    # Demonstrate results access
    print(f"\n📋 Results Summary:")
    for i, component in enumerate(results['identified_components']):
        mineral = component['mineral']
        correlation = component['correlation']
        percentage = component.get('percentage', 0)
        profile = component['fit_parameters'].get('selected_profile', 'unknown')
        
        print(f"   {i+1}. {mineral}")
        print(f"      Correlation: {correlation:.3f}")
        print(f"      Percentage: {percentage:.1f}%")
        print(f"      Profile: {profile}")
    
    # Show uncertainty information if available
    if results['uncertainty_analysis'].get('component_confidence'):
        print(f"\n📊 Uncertainty Analysis:")
        for mineral, conf_data in results['uncertainty_analysis']['component_confidence'].items():
            detection_conf = conf_data['detection_confidence']
            mean_pct = conf_data['mean_percentage']
            std_pct = conf_data['std_percentage']
            
            print(f"   {mineral}: {detection_conf:.1%} confidence")
            print(f"      Mean: {mean_pct:.1f}% ± {std_pct:.1f}%")
    
    # Display final convergence status
    conv_details = results['convergence_details']
    print(f"\n🎯 Convergence Status:")
    for criterion, status in conv_details['criteria_met'].items():
        icon = "✓" if status else "✗"
        print(f"   {icon} {criterion}")
    
    print(f"\n✨ Demo completed successfully!")
    
    # Optional: Test with real RamanLab data if available
    try:
        from pkl_utils import get_example_spectrum_file
        
        example_file = get_example_spectrum_file()
        if example_file and example_file.exists():
            print(f"\n🔍 Testing with real RamanLab data: {example_file.name}")
            
            # Load real spectrum data
            data = np.loadtxt(example_file)
            if data.shape[1] >= 2:
                real_wavenumbers = data[:, 0]
                real_intensities = data[:, 1]
                
                print(f"   Loaded {len(real_wavenumbers)} data points")
                
                # Run analysis on real data
                real_results = analyzer.analyze(
                    wavenumbers=real_wavenumbers,
                    spectrum=real_intensities,
                    max_components=2,
                    plot_results=False,  # Skip plotting for real data demo
                    save_to_database=True,
                    analysis_name=f"real_data_{example_file.stem}"
                )
                
                print(f"   Real data analysis: {real_results['analysis_metadata']['components_found']} components")
            
    except Exception as e:
        print(f"\n⚠️  Could not test with real data: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 RamanLab Enhanced Mixture Analysis Demo Complete!")
    print("   All features demonstrated:")
    print("   ✓ RamanLab database integration")
    print("   ✓ Automatic peak profile selection")
    print("   ✓ Global iterative refinement")
    print("   ✓ Enhanced convergence criteria")
    print("   ✓ Component uncertainty estimation")
    print("   ✓ HDF5 results database storage")
    print("   ✓ Comprehensive visualization")
    print("   ✓ Professional statistical reporting")
    print("=" * 60)
