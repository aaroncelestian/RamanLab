"""
Quantitative Component Analysis for Raman Spectroscopy

This module provides robust quantitative analysis by combining multiple methods
to identify component presence and estimate percentages with confidence intervals.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

# Import the class flip detector
try:
    from .ml_class_flip_detector import MLClassFlipDetector
except ImportError:
    # Fallback for standalone usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ml_class_flip_detector import MLClassFlipDetector

logger = logging.getLogger(__name__)


@dataclass
class ComponentResult:
    """Results for a single component analysis."""
    component_name: str
    intensity_map: np.ndarray
    confidence_map: np.ndarray
    percentage_map: np.ndarray
    detection_map: np.ndarray  # Binary detection (above threshold)
    statistics: Dict[str, float]
    method_contributions: Dict[str, np.ndarray]


@dataclass
class QuantitativeResults:
    """Complete quantitative analysis results."""
    components: List[ComponentResult]
    summary_statistics: Dict[str, Any]
    method_weights: Dict[str, float]
    quality_metrics: Dict[str, float]


class RobustQuantitativeAnalyzer:
    """
    Robust quantitative analysis combining multiple detection methods.
    
    This analyzer addresses common issues:
    - Template-only: too sensitive, many false positives
    - NMF-only: scale issues, underestimation
    - ML methods: training data mismatch, class imbalance
    - Hybrid methods: scale mismatch, alignment issues
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the quantitative analyzer.
        
        Args:
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.template_manager = None
        self.nmf_results = None
        self.ml_results = None
        self.map_shape = None
        self.wavenumbers = None
        
    def set_template_results(self, template_manager, template_coefficients: np.ndarray, 
                           template_r_squared: np.ndarray):
        """Set template fitting results."""
        self.template_manager = template_manager
        self.template_coefficients = template_coefficients
        self.template_r_squared = template_r_squared
        
    def set_nmf_results(self, nmf_components: np.ndarray, nmf_feature_components: np.ndarray):
        """Set NMF results."""
        self.nmf_components = nmf_components
        self.nmf_feature_components = nmf_feature_components
        
    def set_ml_results(self, ml_probabilities: np.ndarray, ml_predictions: np.ndarray,
                       auto_detect_class_flip: bool = True, expected_positive_rate: float = 0.05):
        """
        Set ML classification results with automatic class flip detection.
        
        Args:
            ml_probabilities: ML class probabilities
            ml_predictions: ML class predictions
            auto_detect_class_flip: Whether to automatically detect and correct class flipping
            expected_positive_rate: Expected rate of positive detections for flip detection
        """
        
        # Store original results
        self.ml_probabilities_original = ml_probabilities.copy()
        self.ml_predictions_original = ml_predictions.copy()
        
        # Auto-detect class flipping if enabled
        if auto_detect_class_flip:
            logger.info("Checking for ML class label flipping...")
            
            # Initialize class flip detector
            flip_detector = MLClassFlipDetector()
            
            # Get template and NMF results for comparison (if available)
            template_results = getattr(self, 'template_coefficients', None)
            nmf_results = getattr(self, 'nmf_components', None)
            
            # Detect class flipping
            flip_result = flip_detector.detect_class_flip(
                ml_probabilities=ml_probabilities,
                ml_predictions=ml_predictions,
                template_results=template_results,
                nmf_results=nmf_results,
                expected_positive_rate=expected_positive_rate
            )
            
            # Store flip detection results
            self.class_flip_result = flip_result
            self.flip_detector = flip_detector
            
            # Apply correction if flip detected
            if flip_result['flip_detected']:
                logger.warning(f"Class flip detected (confidence: {flip_result['flip_confidence']:.3f})")
                logger.info(f"Recommended target class: {flip_result['recommended_target_class']}")
                
                # Correct the class flip
                corrected_probs, corrected_preds = flip_detector.correct_class_flip(
                    ml_probabilities, ml_predictions
                )
                
                self.ml_probabilities = corrected_probs
                self.ml_predictions = corrected_preds
                
                logger.info("Applied class flip correction to ML results")
                
                # Log the correction details
                original_minority_rate = flip_result['diagnostic_info']['minority_rate']
                new_minority_rate = np.mean(corrected_preds == flip_result['recommended_target_class'])
                logger.info(f"Class distribution changed: {original_minority_rate:.3f} -> {new_minority_rate:.3f}")
                
            else:
                logger.info("No class flip detected - using original ML results")
                self.ml_probabilities = ml_probabilities
                self.ml_predictions = ml_predictions
                
        else:
            # Use results as-is
            self.ml_probabilities = ml_probabilities
            self.ml_predictions = ml_predictions
            self.class_flip_result = None
        
    def set_map_info(self, map_shape: Tuple[int, int], wavenumbers: np.ndarray):
        """Set map geometry and wavenumber information."""
        self.map_shape = map_shape
        self.wavenumbers = wavenumbers
        
    def analyze_component(self, component_name: str, template_index: Optional[int] = None,
                         nmf_component: Optional[int] = None, 
                         target_class_index: Optional[int] = None) -> ComponentResult:
        """
        Perform comprehensive quantitative analysis for a single component.
        
        Args:
            component_name: Name of the component to analyze
            template_index: Index of template (if using template method)
            nmf_component: Index of NMF component (if using NMF method)
            target_class_index: Index of target class (if using ML method)
            
        Returns:
            ComponentResult with quantitative analysis
        """
        logger.info(f"Starting quantitative analysis for component: {component_name}")
        
        # Collect available methods and their results
        method_results = {}
        method_weights = {}
        
        # 1. Template-based analysis (if available)
        if (template_index is not None and 
            hasattr(self, 'template_coefficients') and 
            template_index < self.template_coefficients.shape[1]):
                
            template_result = self._analyze_template_method(template_index)
            method_results['template'] = template_result
            method_weights['template'] = self._calculate_template_weight(template_result)
            logger.info(f"Template method: weight = {method_weights['template']:.3f}")
            
        # 2. NMF-based analysis (if available)
        if (nmf_component is not None and 
            hasattr(self, 'nmf_components') and 
            nmf_component < self.nmf_components.shape[1]):
                
            nmf_result = self._analyze_nmf_method(nmf_component)
            method_results['nmf'] = nmf_result
            method_weights['nmf'] = self._calculate_nmf_weight(nmf_result)
            logger.info(f"NMF method: weight = {method_weights['nmf']:.3f}")
            
        # 3. ML-based analysis (if available)
        if (target_class_index is not None and 
            hasattr(self, 'ml_probabilities') and 
            target_class_index < self.ml_probabilities.shape[1]):
                
            ml_result = self._analyze_ml_method(target_class_index)
            method_results['ml'] = ml_result
            method_weights['ml'] = self._calculate_ml_weight(ml_result)
            logger.info(f"ML method: weight = {method_weights['ml']:.3f}")
            
        if not method_results:
            raise ValueError(f"No valid methods available for component {component_name}")
            
        # Combine methods using weighted ensemble
        combined_result = self._combine_methods(method_results, method_weights)
        
        # Calculate final statistics
        statistics = self._calculate_component_statistics(combined_result, method_results)
        
        # Create detection map based on confidence threshold
        detection_map = combined_result['confidence'] >= self.confidence_threshold
        
        return ComponentResult(
            component_name=component_name,
            intensity_map=combined_result['intensity'],
            confidence_map=combined_result['confidence'],
            percentage_map=combined_result['percentage'],
            detection_map=detection_map,
            statistics=statistics,
            method_contributions=method_results
        )
        
    def _analyze_template_method(self, template_index: int) -> Dict[str, np.ndarray]:
        """Analyze using template fitting method."""
        coefficients = self.template_coefficients[:, template_index]
        r_squared = self.template_r_squared[:, template_index]
        
        # Intensity: use coefficients directly
        intensity = np.clip(coefficients, 0, None)  # Ensure non-negative
        
        # Confidence: based on R-squared and coefficient magnitude
        # High R² + significant coefficient = high confidence
        coeff_percentile = np.percentile(coefficients[coefficients > 0], 90)
        normalized_coeff = np.clip(coefficients / (coeff_percentile + 1e-6), 0, 1)
        confidence = r_squared * normalized_coeff
        
        # Percentage: normalize coefficients to percentage-like values
        max_coeff = np.percentile(coefficients[coefficients > 0], 95)
        percentage = np.clip(coefficients / (max_coeff + 1e-6) * 100, 0, 100)
        
        return {
            'intensity': intensity,
            'confidence': confidence,
            'percentage': percentage,
            'quality': r_squared
        }
        
    def _analyze_nmf_method(self, nmf_component: int) -> Dict[str, np.ndarray]:
        """Analyze using NMF method."""
        component_values = self.nmf_components[:, nmf_component]
        
        # Intensity: use component values directly
        intensity = np.clip(component_values, 0, None)
        
        # Confidence: based on relative magnitude within component
        # Higher values within this component get higher confidence
        percentile_95 = np.percentile(component_values, 95)
        percentile_50 = np.percentile(component_values, 50)
        
        if percentile_95 > percentile_50:
            confidence = np.clip((component_values - percentile_50) / 
                               (percentile_95 - percentile_50), 0, 1)
        else:
            confidence = np.ones_like(component_values) * 0.5
            
        # Percentage: normalize to percentage scale
        max_val = np.percentile(component_values[component_values > 0], 90)
        percentage = np.clip(component_values / (max_val + 1e-6) * 100, 0, 100)
        
        # Quality: measure of component separation (higher variance = better separation)
        quality_score = np.var(component_values) / (np.mean(component_values) + 1e-6)
        quality = np.full_like(component_values, min(quality_score / 10, 1.0))  # Normalize
        
        return {
            'intensity': intensity,
            'confidence': confidence,
            'percentage': percentage,
            'quality': quality
        }
        
    def _analyze_ml_method(self, class_index: int) -> Dict[str, np.ndarray]:
        """Analyze using ML classification method."""
        probabilities = self.ml_probabilities[:, class_index]
        predictions = self.ml_predictions
        
        # Intensity: use probabilities directly
        intensity = probabilities
        
        # Confidence: probabilities are already confidence measures
        confidence = probabilities
        
        # Percentage: scale probabilities to percentage
        percentage = probabilities * 100
        
        # Quality: measure of prediction certainty
        # High probabilities (close to 0 or 1) indicate high certainty
        prediction_certainty = np.abs(probabilities - 0.5) * 2  # 0-1 scale
        quality = prediction_certainty
        
        return {
            'intensity': intensity,
            'confidence': confidence,
            'percentage': percentage,
            'quality': quality
        }
        
    def _calculate_template_weight(self, template_result: Dict[str, np.ndarray]) -> float:
        """Calculate weight for template method based on quality."""
        mean_r_squared = np.mean(template_result['quality'])
        detection_rate = np.mean(template_result['confidence'] > 0.1)  # How many pixels detected
        
        # Template method gets higher weight when:
        # - Good average fit quality (R²)
        # - Reasonable detection rate (not too sparse, not too dense)
        optimal_detection_rate = 0.1  # 10% seems reasonable for many components
        detection_penalty = abs(detection_rate - optimal_detection_rate) / optimal_detection_rate
        
        weight = mean_r_squared * (1 - min(detection_penalty, 0.5))
        return max(weight, 0.1)  # Minimum weight
        
    def _calculate_nmf_weight(self, nmf_result: Dict[str, np.ndarray]) -> float:
        """Calculate weight for NMF method based on quality."""
        mean_quality = np.mean(nmf_result['quality'])
        dynamic_range = np.ptp(nmf_result['intensity'])  # Peak-to-peak range
        
        # NMF gets higher weight when component has good separation
        # and reasonable dynamic range
        if dynamic_range > 0:
            normalized_range = min(dynamic_range / np.mean(nmf_result['intensity']), 10)
            weight = mean_quality * (normalized_range / 10)
        else:
            weight = 0.1
            
        return max(weight, 0.1)
        
    def _calculate_ml_weight(self, ml_result: Dict[str, np.ndarray]) -> float:
        """Calculate weight for ML method based on quality."""
        mean_certainty = np.mean(ml_result['quality'])
        detection_rate = np.mean(ml_result['confidence'] > 0.5)
        
        # ML method gets lower weight if detection rate is too extreme
        # (suggests training data mismatch)
        if detection_rate < 0.001 or detection_rate > 0.5:
            penalty = 0.5  # Reduce weight significantly
        else:
            penalty = 0.0
            
        weight = mean_certainty * (1 - penalty)
        return max(weight, 0.05)  # Lower minimum weight than other methods
        
    def _combine_methods(self, method_results: Dict[str, Dict], 
                        method_weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Combine multiple methods using weighted ensemble."""
        
        # Normalize weights
        total_weight = sum(method_weights.values())
        normalized_weights = {k: v/total_weight for k, v in method_weights.items()}
        
        logger.info(f"Normalized method weights: {normalized_weights}")
        
        # Initialize combined results
        n_pixels = len(list(method_results.values())[0]['intensity'])
        combined = {
            'intensity': np.zeros(n_pixels),
            'confidence': np.zeros(n_pixels),
            'percentage': np.zeros(n_pixels)
        }
        
        # Weighted combination
        for method_name, weight in normalized_weights.items():
            method_data = method_results[method_name]
            combined['intensity'] += weight * method_data['intensity']
            combined['confidence'] += weight * method_data['confidence']  
            combined['percentage'] += weight * method_data['percentage']
            
        # Apply ensemble confidence boost for agreement
        if len(method_results) > 1:
            agreement_boost = self._calculate_method_agreement(method_results)
            combined['confidence'] = np.minimum(combined['confidence'] * agreement_boost, 1.0)
            
        return combined
        
    def _calculate_method_agreement(self, method_results: Dict[str, Dict]) -> np.ndarray:
        """Calculate boost factor based on method agreement."""
        
        if len(method_results) < 2:
            return np.ones(len(list(method_results.values())[0]['intensity']))
        
        # Calculate pairwise correlations between method confidence values
        methods = list(method_results.keys())
        correlations = []
        
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                conf1 = method_results[methods[i]]['confidence']
                conf2 = method_results[methods[j]]['confidence']
                
                # Calculate correlation
                if np.std(conf1) > 1e-6 and np.std(conf2) > 1e-6:
                    corr = np.corrcoef(conf1, conf2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if correlations:
            mean_correlation = np.mean(correlations)
            # Boost factor: 1.0 (no boost) to 1.5 (50% boost) based on agreement
            boost_factor = 1.0 + 0.5 * max(mean_correlation, 0)
        else:
            boost_factor = 1.0
            
        return np.full(len(list(method_results.values())[0]['intensity']), boost_factor)
        
    def _calculate_component_statistics(self, combined_result: Dict[str, np.ndarray],
                                      method_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate comprehensive statistics for the component."""
        
        intensity = combined_result['intensity']
        confidence = combined_result['confidence']
        percentage = combined_result['percentage']
        
        # Detection statistics
        detected_pixels = confidence >= self.confidence_threshold
        n_detected = np.sum(detected_pixels)
        detection_percentage = (n_detected / len(intensity)) * 100
        
        # Intensity statistics (for detected pixels only)
        if n_detected > 0:
            mean_intensity = np.mean(intensity[detected_pixels])
            std_intensity = np.std(intensity[detected_pixels])
            mean_percentage = np.mean(percentage[detected_pixels])
            std_percentage = np.std(percentage[detected_pixels])
        else:
            mean_intensity = 0.0
            std_intensity = 0.0
            mean_percentage = 0.0
            std_percentage = 0.0
            
        # Overall statistics
        overall_mean_intensity = np.mean(intensity)
        overall_mean_confidence = np.mean(confidence)
        
        # Method contribution statistics
        method_contributions = {}
        for method_name in method_results.keys():
            method_detected = np.sum(method_results[method_name]['confidence'] >= 0.5)
            method_contributions[f'{method_name}_detection_count'] = int(method_detected)
            method_contributions[f'{method_name}_detection_percent'] = (method_detected / len(intensity)) * 100
            
        return {
            'total_pixels': len(intensity),
            'detected_pixels': int(n_detected),
            'detection_percentage': detection_percentage,
            'mean_intensity_detected': mean_intensity,
            'std_intensity_detected': std_intensity,
            'mean_percentage_detected': mean_percentage,
            'std_percentage_detected': std_percentage,
            'overall_mean_intensity': overall_mean_intensity,
            'overall_mean_confidence': overall_mean_confidence,
            'confidence_threshold_used': self.confidence_threshold,
            **method_contributions
        }
        
    def generate_summary_report(self, results: List[ComponentResult]) -> str:
        """Generate a human-readable summary report."""
        
        report = ["=" * 60]
        report.append("QUANTITATIVE COMPONENT ANALYSIS REPORT")
        report.append("=" * 60)
        
        for result in results:
            report.append(f"\nComponent: {result.component_name}")
            report.append("-" * 40)
            
            stats = result.statistics
            report.append(f"Detection Summary:")
            report.append(f"  • Total pixels analyzed: {stats['total_pixels']:,}")
            report.append(f"  • Pixels with component detected: {stats['detected_pixels']:,}")
            report.append(f"  • Detection percentage: {stats['detection_percentage']:.2f}%")
            
            if stats['detected_pixels'] > 0:
                report.append(f"\nQuantitative Results (detected pixels only):")
                report.append(f"  • Average component percentage: {stats['mean_percentage_detected']:.1f}% ± {stats['std_percentage_detected']:.1f}%")
                report.append(f"  • Average intensity: {stats['mean_intensity_detected']:.3f} ± {stats['std_intensity_detected']:.3f}")
                
            report.append(f"\nMethod Contributions:")
            for key, value in stats.items():
                if '_detection_percent' in key:
                    method_name = key.replace('_detection_percent', '').title()
                    report.append(f"  • {method_name}: {value:.2f}% of pixels")
                    
        return "\n".join(report) 