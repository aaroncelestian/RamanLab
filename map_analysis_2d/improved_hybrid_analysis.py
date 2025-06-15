#!/usr/bin/env python3
"""
Improved Hybrid Analysis for Raman Spectroscopy

This module addresses the key limitations in the existing hybrid analysis:
1. Scale & Alignment Issues (Point 2)
2. Algorithmic Limitations (Point 3) 
3. Data Quality Dependencies (Point 4)

Key improvements:
- Adaptive scaling and normalization
- Intelligent method weighting
- Dynamic threshold optimization
- Quality-based processing
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality assessment for each analysis method."""
    mean_confidence: float
    std_confidence: float
    outlier_fraction: float
    data_range: Tuple[float, float]
    noise_level: float

@dataclass
class AdaptiveParams:
    """Adaptively determined parameters."""
    nmf_threshold: float
    template_threshold: float
    method_weights: Dict[str, float]
    scaling_factors: Dict[str, float]

class ImprovedHybridAnalyzer:
    """Enhanced hybrid analysis with adaptive processing."""
    
    def __init__(self):
        self.nmf_scaler = RobustScaler()
        self.template_scaler = RobustScaler()
        self.quality_metrics = {}
        self.adaptive_params = None
        
    def assess_data_quality(self, nmf_data: np.ndarray, template_data: Dict, 
                           r_squared_data: Dict) -> Dict[str, QualityMetrics]:
        """
        IMPROVEMENT 4: Data Quality Dependencies
        Automatically assess quality to guide processing decisions.
        """
        quality_metrics = {}
        
        # Assess NMF data quality
        nmf_flat = np.array(nmf_data).flatten()
        nmf_valid = nmf_flat[np.isfinite(nmf_flat)]
        
        if len(nmf_valid) > 0:
            # Outlier detection using IQR
            q75, q25 = np.percentile(nmf_valid, [75, 25])
            iqr = q75 - q25
            outliers = ((nmf_valid < (q25 - 1.5 * iqr)) | 
                       (nmf_valid > (q75 + 1.5 * iqr)))
            outlier_fraction = np.sum(outliers) / len(nmf_valid)
            
            # Noise estimation using MAD
            median_val = np.median(nmf_valid)
            mad = np.median(np.abs(nmf_valid - median_val))
            noise_level = 1.4826 * mad
            
            quality_metrics['nmf'] = QualityMetrics(
                mean_confidence=np.mean(nmf_valid),
                std_confidence=np.std(nmf_valid),
                outlier_fraction=outlier_fraction,
                data_range=(np.min(nmf_valid), np.max(nmf_valid)),
                noise_level=noise_level
            )
        
        # Assess template data quality using R-squared values
        if template_data and r_squared_data:
            r2_values = [r_squared_data.get(pos, 0.0) for pos in template_data.keys()]
            r2_valid = [r for r in r2_values if np.isfinite(r)]
            
            if r2_valid:
                poor_fits = np.sum(np.array(r2_valid) < 0.1) / len(r2_valid)
                
                quality_metrics['template'] = QualityMetrics(
                    mean_confidence=np.mean(r2_valid),
                    std_confidence=np.std(r2_valid),
                    outlier_fraction=poor_fits,
                    data_range=(0.0, 1.0),  # R-squared range
                    noise_level=np.std(r2_valid)
                )
        
        self.quality_metrics = quality_metrics
        return quality_metrics
    
    def adaptive_scaling(self, nmf_data: np.ndarray, template_data: Dict) -> Tuple[np.ndarray, Dict]:
        """
        IMPROVEMENT 2: Scale & Alignment Issues
        Robust scaling that handles different data ranges and outliers.
        """
        logger.info("Applying adaptive scaling...")
        
        # Robust scaling for NMF data
        nmf_reshaped = nmf_data.flatten().reshape(-1, 1)
        nmf_scaled = self.nmf_scaler.fit_transform(nmf_reshaped).flatten()
        
        # Quality-based post-processing
        if 'nmf' in self.quality_metrics:
            quality = self.quality_metrics['nmf']
            
            # Handle high noise levels
            if quality.noise_level > quality.std_confidence * 0.5:
                # Apply gentle smoothing
                kernel = np.array([0.25, 0.5, 0.25])
                nmf_padded = np.pad(nmf_scaled, 1, mode='edge')
                nmf_scaled = np.convolve(nmf_padded, kernel, mode='valid')
                logger.info(f"Applied noise reduction (noise: {quality.noise_level:.3f})")
            
            # Handle outliers
            if quality.outlier_fraction > 0.15:
                # Clip to robust percentiles
                p5, p95 = np.percentile(nmf_scaled, [5, 95])
                nmf_scaled = np.clip(nmf_scaled, p5, p95)
                logger.info(f"Applied outlier clipping (outliers: {quality.outlier_fraction:.1%})")
        
        # Scale template data consistently
        template_scaled = {}
        if template_data:
            all_coeffs = [c for coeffs in template_data.values() for c in coeffs]
            if all_coeffs:
                coeffs_array = np.array(all_coeffs).reshape(-1, 1)
                scaled_coeffs = self.template_scaler.fit_transform(coeffs_array).flatten()
                
                # Redistribute to original structure
                idx = 0
                for pos_key, orig_coeffs in template_data.items():
                    n_coeffs = len(orig_coeffs)
                    template_scaled[pos_key] = scaled_coeffs[idx:idx+n_coeffs].tolist()
                    idx += n_coeffs
        
        return nmf_scaled, template_scaled
    
    def determine_adaptive_parameters(self, nmf_data: np.ndarray, r_squared_data: Dict) -> AdaptiveParams:
        """
        IMPROVEMENT 3: Algorithmic Limitations
        Dynamic parameter optimization based on data characteristics.
        """
        logger.info("Determining adaptive parameters...")
        
        # Auto-optimize NMF threshold using statistical methods
        nmf_valid = nmf_data[np.isfinite(nmf_data)]
        
        if len(nmf_valid) > 10:
            # Use elbow method on sorted data
            sorted_nmf = np.sort(nmf_valid)[::-1]
            
            # Calculate rate of change to find elbow
            changes = np.diff(sorted_nmf)
            if len(changes) > 5:
                # Find where rate of change slows down significantly
                change_rates = np.diff(changes)
                if len(change_rates) > 0:
                    elbow_idx = np.argmin(change_rates) + 2
                    nmf_threshold = sorted_nmf[min(elbow_idx, len(sorted_nmf)-1)]
                else:
                    nmf_threshold = np.percentile(nmf_valid, 80)
            else:
                nmf_threshold = np.percentile(nmf_valid, 80)
        else:
            nmf_threshold = 1.0  # Fallback
        
        # Ensure reasonable bounds
        nmf_mean = np.mean(nmf_valid) if len(nmf_valid) > 0 else 0
        nmf_std = np.std(nmf_valid) if len(nmf_valid) > 0 else 1
        nmf_threshold = max(nmf_threshold, nmf_mean + 0.3 * nmf_std)
        nmf_threshold = min(nmf_threshold, np.percentile(nmf_valid, 90))
        
        # Auto-optimize template threshold
        template_threshold = 0.25  # Default
        if r_squared_data:
            r2_values = [r for r in r_squared_data.values() if np.isfinite(r)]
            if r2_values:
                r2_mean = np.mean(r2_values)
                r2_std = np.std(r2_values)
                # Adaptive threshold: slightly below mean
                template_threshold = max(0.1, r2_mean - 0.3 * r2_std)
                template_threshold = min(0.7, template_threshold)
        
        # Quality-based method weighting
        method_weights = {'nmf': 0.5, 'template': 0.5}  # Default
        
        if self.quality_metrics:
            nmf_quality = self.quality_metrics.get('nmf')
            template_quality = self.quality_metrics.get('template')
            
            if nmf_quality and template_quality:
                # Weight based on quality scores
                nmf_score = ((1 - nmf_quality.outlier_fraction) * 
                           (1 - min(1.0, nmf_quality.noise_level)))
                template_score = (template_quality.mean_confidence * 
                                (1 - template_quality.outlier_fraction))
                
                total_score = nmf_score + template_score
                if total_score > 0:
                    method_weights['nmf'] = nmf_score / total_score
                    method_weights['template'] = template_score / total_score
        
        # Determine scaling factors for balanced contribution
        scaling_factors = {'nmf': 1.0, 'template': 1.0}
        
        if 'nmf' in self.quality_metrics and 'template' in self.quality_metrics:
            nmf_range = (self.quality_metrics['nmf'].data_range[1] - 
                        self.quality_metrics['nmf'].data_range[0])
            template_range = (self.quality_metrics['template'].data_range[1] - 
                            self.quality_metrics['template'].data_range[0])
            
            # Balance contributions if ranges are very different
            if nmf_range > 0 and template_range > 0:
                ratio = template_range / nmf_range
                if ratio > 3:
                    scaling_factors['template'] = 0.6
                elif ratio < 0.33:
                    scaling_factors['nmf'] = 0.6
        
        params = AdaptiveParams(
            nmf_threshold=nmf_threshold,
            template_threshold=template_threshold,
            method_weights=method_weights,
            scaling_factors=scaling_factors
        )
        
        logger.info(f"Adaptive params: NMF thresh={nmf_threshold:.3f}, "
                   f"Template thresh={template_threshold:.3f}, "
                   f"Weights: NMF={method_weights['nmf']:.3f}, "
                   f"Template={method_weights['template']:.3f}")
        
        self.adaptive_params = params
        return params
    
    def enhanced_combination(self, nmf_intensity: float, template_strength: float, 
                           r_squared: float) -> Dict[str, float]:
        """
        IMPROVEMENT 3: Enhanced combination with non-linear strategies.
        """
        if not self.adaptive_params:
            raise ValueError("Must determine adaptive parameters first")
        
        params = self.adaptive_params
        
        # Validate and clean inputs
        nmf_val = max(0, float(nmf_intensity)) if np.isfinite(nmf_intensity) else 0.0
        template_val = max(0, float(template_strength)) if np.isfinite(template_strength) else 0.0
        r2_val = max(0, min(1, float(r_squared))) if np.isfinite(r_squared) else 0.0
        
        # Apply scaling factors
        nmf_scaled = nmf_val * params.scaling_factors['nmf']
        template_scaled = template_val * params.scaling_factors['template']
        
        # Enhanced combination logic
        if template_scaled > params.template_threshold and r2_val > 0.1:
            # Template-dominant with quality weighting
            base_intensity = template_scaled
            
            # Non-linear NMF boost
            if nmf_scaled > params.nmf_threshold:
                boost_factor = 1 + np.tanh(nmf_scaled / params.nmf_threshold - 1)
            else:
                boost_factor = 1.0
            
            # Non-linear quality weighting for better discrimination
            quality_weight = 1 + np.power(r2_val, 2)
            
            hybrid_intensity = base_intensity * boost_factor * quality_weight
            confidence = r2_val * min(1.0, boost_factor)
            
        elif nmf_scaled > params.nmf_threshold:
            # NMF-dominant (more conservative)
            base_intensity = nmf_scaled * 0.6
            
            # Template support
            if template_scaled > 0:
                support_factor = 1 + 0.4 * np.sqrt(template_scaled / max(0.01, params.template_threshold))
                base_intensity *= support_factor
            
            hybrid_intensity = base_intensity
            confidence = min(0.6, nmf_scaled / params.nmf_threshold * 0.4)
            
        else:
            # Weak signal region
            weighted_avg = (nmf_scaled * params.method_weights['nmf'] + 
                          template_scaled * params.method_weights['template'])
            hybrid_intensity = weighted_avg * 0.4
            confidence = min(0.4, weighted_avg)
        
        return {
            'hybrid_intensity': float(hybrid_intensity),
            'confidence_score': float(confidence),
            'nmf_contribution': float(nmf_scaled * params.method_weights['nmf']),
            'template_contribution': float(template_scaled * params.method_weights['template'])
        }
    
    def process_improved_analysis(self, nmf_data: np.ndarray, template_data: Dict,
                                r_squared_data: Dict, positions: List) -> Dict:
        """Main processing function for improved hybrid analysis."""
        logger.info("Starting improved hybrid analysis...")
        
        # Step 1: Assess data quality
        quality_metrics = self.assess_data_quality(nmf_data, template_data, r_squared_data)
        
        # Step 2: Apply adaptive scaling
        nmf_scaled, template_scaled = self.adaptive_scaling(nmf_data, template_data)
        
        # Step 3: Determine adaptive parameters
        adaptive_params = self.determine_adaptive_parameters(nmf_scaled, r_squared_data)
        
        # Step 4: Process each position
        results = {
            'positions': {},
            'summary': {
                'total_positions': len(positions),
                'high_confidence_count': 0,
                'processing_errors': 0
            },
            'quality_metrics': quality_metrics,
            'adaptive_parameters': adaptive_params
        }
        
        high_confidence_count = 0
        processing_errors = 0
        
        for i, pos in enumerate(positions):
            try:
                pos_key = tuple(pos) if isinstance(pos, (list, tuple)) else pos
                
                # Get data for this position
                nmf_intensity = nmf_scaled[i] if i < len(nmf_scaled) else 0.0
                template_coeffs = template_scaled.get(pos_key, [0.0])
                template_strength = template_coeffs[0] if template_coeffs else 0.0
                r_squared = r_squared_data.get(pos_key, 0.0)
                
                # Apply enhanced combination
                result = self.enhanced_combination(nmf_intensity, template_strength, r_squared)
                results['positions'][pos_key] = result
                
                # Count high confidence detections
                if result['confidence_score'] > 0.5:
                    high_confidence_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing position {pos}: {e}")
                processing_errors += 1
        
        # Update summary
        results['summary'].update({
            'high_confidence_count': high_confidence_count,
            'processing_errors': processing_errors,
            'success_rate': (len(positions) - processing_errors) / len(positions) if positions else 0
        })
        
        logger.info(f"Improved analysis complete. Success rate: {results['summary']['success_rate']:.1%}")
        return results


def integrate_improved_hybrid_analysis(main_window, nmf_component_index=2):
    """
    Integration function to replace the original hybrid analysis with improved version.
    """
    try:
        # Validate prerequisites
        if not hasattr(main_window, 'template_fitting_results') or not main_window.template_fitting_results:
            raise ValueError("Template fitting results required")
        
        if not hasattr(main_window, 'nmf_results') or not main_window.nmf_results:
            raise ValueError("NMF results required")
        
        # Extract data
        nmf_components = main_window.nmf_results.get('components', [])
        if nmf_component_index >= len(nmf_components):
            raise ValueError(f"NMF component {nmf_component_index} not available")
        
        nmf_data = np.array(nmf_components[nmf_component_index])
        template_coefficients = main_window.template_fitting_results['coefficients']
        r_squared_values = main_window.template_fitting_results.get('r_squared', {})
        positions = [(spec.x_pos, spec.y_pos) for spec in main_window.map_data.spectra.values()]
        
        # Create and run improved analyzer
        analyzer = ImprovedHybridAnalyzer()
        results = analyzer.process_improved_analysis(
            nmf_data, template_coefficients, r_squared_values, positions
        )
        
        # Store results
        main_window.improved_hybrid_results = results
        
        # Log improvements
        logger.info("=== IMPROVED HYBRID ANALYSIS COMPLETE ===")
        logger.info(f"Quality-based weighting: {analyzer.adaptive_params.method_weights}")
        logger.info(f"Adaptive thresholds: NMF={analyzer.adaptive_params.nmf_threshold:.3f}, "
                   f"Template={analyzer.adaptive_params.template_threshold:.3f}")
        logger.info(f"Success rate: {results['summary']['success_rate']:.1%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Improved hybrid analysis failed: {e}")
        raise 