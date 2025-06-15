#!/usr/bin/env python3
"""
ML Class Flip Detection and Correction

This module detects when ML classification results have flipped class labels
(common with class imbalance) and provides correction methods.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from scipy import stats

logger = logging.getLogger(__name__)


class MLClassFlipDetector:
    """
    Detects and corrects class label flipping in ML classification results.
    
    This is particularly common with imbalanced datasets where the minority
    class (target material) gets confused with the majority class (background).
    """
    
    def __init__(self):
        """Initialize the class flip detector."""
        self.flip_detected = False
        self.confidence_score = 0.0
        self.diagnostic_info = {}
        
    def detect_class_flip(self, ml_probabilities: np.ndarray, 
                         ml_predictions: np.ndarray,
                         template_results: Optional[np.ndarray] = None,
                         nmf_results: Optional[np.ndarray] = None,
                         expected_positive_rate: float = 0.05) -> Dict[str, any]:
        """
        Detect if ML class labels are flipped.
        
        Args:
            ml_probabilities: ML class probabilities (n_pixels, n_classes)
            ml_predictions: ML class predictions (n_pixels,)
            template_results: Template fitting results for comparison (optional)
            nmf_results: NMF results for comparison (optional)
            expected_positive_rate: Expected rate of positive detections (0.01-0.1)
            
        Returns:
            Dictionary with flip detection results
        """
        
        logger.info("Analyzing ML classification for potential class flipping...")
        
        n_pixels = len(ml_predictions)
        n_classes = ml_probabilities.shape[1]
        
        # Get class counts
        class_counts = {}
        for i in range(n_classes):
            class_counts[i] = np.sum(ml_predictions == i)
            
        logger.info(f"Class counts: {class_counts}")
        
        # Identify minority and majority classes
        minority_class = min(class_counts.keys(), key=lambda x: class_counts[x])
        majority_class = max(class_counts.keys(), key=lambda x: class_counts[x])
        
        minority_rate = class_counts[minority_class] / n_pixels
        majority_rate = class_counts[majority_class] / n_pixels
        
        logger.info(f"Minority class {minority_class}: {minority_rate:.3f} ({class_counts[minority_class]} pixels)")
        logger.info(f"Majority class {majority_class}: {majority_rate:.3f} ({class_counts[majority_class]} pixels)")
        
        # Initialize detection results
        flip_indicators = []
        diagnostic_info = {
            'minority_class': minority_class,
            'majority_class': majority_class,
            'minority_rate': minority_rate,
            'majority_rate': majority_rate,
            'expected_positive_rate': expected_positive_rate
        }
        
        # Test 1: Check detection rate vs expectation
        rate_test = self._test_detection_rate(minority_rate, expected_positive_rate)
        flip_indicators.append(rate_test)
        diagnostic_info['rate_test'] = rate_test
        
        # Test 2: Compare with template results (if available)
        if template_results is not None:
            template_test = self._test_template_agreement(
                ml_probabilities, ml_predictions, template_results, 
                minority_class, majority_class
            )
            flip_indicators.append(template_test)
            diagnostic_info['template_test'] = template_test
            
        # Test 3: Compare with NMF results (if available)
        if nmf_results is not None:
            nmf_test = self._test_nmf_agreement(
                ml_probabilities, ml_predictions, nmf_results,
                minority_class, majority_class
            )
            flip_indicators.append(nmf_test)
            diagnostic_info['nmf_test'] = nmf_test
            
        # Test 4: Probability distribution analysis
        prob_test = self._test_probability_distributions(ml_probabilities, minority_class, majority_class)
        flip_indicators.append(prob_test)
        diagnostic_info['probability_test'] = prob_test
        
        # Overall flip detection decision
        flip_scores = [test['flip_likelihood'] for test in flip_indicators if test is not None]
        
        if flip_scores:
            overall_flip_score = np.mean(flip_scores)
            flip_detected = overall_flip_score > 0.5
            confidence = abs(overall_flip_score - 0.5) * 2  # 0-1 scale
        else:
            overall_flip_score = 0.5
            flip_detected = False
            confidence = 0.0
            
        self.flip_detected = flip_detected
        self.confidence_score = confidence
        self.diagnostic_info = diagnostic_info
        
        result = {
            'flip_detected': flip_detected,
            'flip_confidence': confidence,
            'flip_score': overall_flip_score,
            'minority_class_original': minority_class,
            'majority_class_original': majority_class,
            'recommended_target_class': majority_class if flip_detected else minority_class,
            'diagnostic_info': diagnostic_info,
            'flip_indicators': flip_indicators
        }
        
        logger.info(f"Class flip detection: {'FLIP DETECTED' if flip_detected else 'NO FLIP'} "
                   f"(confidence: {confidence:.3f})")
        
        return result
        
    def _test_detection_rate(self, minority_rate: float, expected_rate: float) -> Dict[str, any]:
        """Test if detection rate suggests class flipping."""
        
        # If minority rate is much smaller than expected, might be flipped
        rate_ratio = minority_rate / expected_rate
        
        if rate_ratio < 0.1:  # Much fewer detections than expected
            flip_likelihood = 0.8
            reason = f"Very low detection rate ({minority_rate:.3f}) vs expected ({expected_rate:.3f})"
        elif rate_ratio < 0.5:  # Somewhat fewer than expected
            flip_likelihood = 0.6
            reason = f"Low detection rate ({minority_rate:.3f}) vs expected ({expected_rate:.3f})"
        elif rate_ratio > 2.0:  # More than expected
            flip_likelihood = 0.2
            reason = f"Higher than expected detection rate ({minority_rate:.3f}) vs expected ({expected_rate:.3f})"
        else:  # About right
            flip_likelihood = 0.3
            reason = f"Detection rate ({minority_rate:.3f}) close to expected ({expected_rate:.3f})"
            
        return {
            'test_name': 'detection_rate',
            'flip_likelihood': flip_likelihood,
            'reason': reason,
            'minority_rate': minority_rate,
            'expected_rate': expected_rate,
            'rate_ratio': rate_ratio
        }
        
    def _test_template_agreement(self, ml_probabilities: np.ndarray, ml_predictions: np.ndarray,
                               template_results: np.ndarray, minority_class: int, 
                               majority_class: int) -> Optional[Dict[str, any]]:
        """Test agreement between ML results and template fitting."""
        
        try:
            # Use template coefficients as ground truth indicator
            template_strength = np.max(template_results, axis=1) if len(template_results.shape) > 1 else template_results
            
            # Strong template detections (top 10%)
            template_threshold = np.percentile(template_strength, 90)
            strong_template_pixels = template_strength > template_threshold
            
            if np.sum(strong_template_pixels) < 10:  # Too few for reliable comparison
                return None
                
            # Check which ML class agrees better with strong template detections
            minority_agreement = np.mean(ml_predictions[strong_template_pixels] == minority_class)
            majority_agreement = np.mean(ml_predictions[strong_template_pixels] == majority_class)
            
            if majority_agreement > minority_agreement:
                flip_likelihood = 0.8
                reason = f"Majority class agrees better with template ({majority_agreement:.3f} vs {minority_agreement:.3f})"
            else:
                flip_likelihood = 0.2
                reason = f"Minority class agrees better with template ({minority_agreement:.3f} vs {majority_agreement:.3f})"
                
            return {
                'test_name': 'template_agreement',
                'flip_likelihood': flip_likelihood,
                'reason': reason,
                'minority_agreement': minority_agreement,
                'majority_agreement': majority_agreement,
                'strong_template_pixels': np.sum(strong_template_pixels)
            }
            
        except Exception as e:
            logger.warning(f"Template agreement test failed: {str(e)}")
            return None
            
    def _test_nmf_agreement(self, ml_probabilities: np.ndarray, ml_predictions: np.ndarray,
                          nmf_results: np.ndarray, minority_class: int, 
                          majority_class: int) -> Optional[Dict[str, any]]:
        """Test agreement between ML results and NMF components."""
        
        try:
            # Find the NMF component with best separation
            best_component = 0
            best_separation = 0
            
            for i in range(nmf_results.shape[1]):
                component_values = nmf_results[:, i]
                separation = np.std(component_values) / (np.mean(component_values) + 1e-6)
                if separation > best_separation:
                    best_separation = separation
                    best_component = i
                    
            nmf_values = nmf_results[:, best_component]
            nmf_threshold = np.percentile(nmf_values, 90)
            strong_nmf_pixels = nmf_values > nmf_threshold
            
            if np.sum(strong_nmf_pixels) < 10:  # Too few for reliable comparison
                return None
                
            # Check which ML class agrees better with strong NMF detections
            minority_agreement = np.mean(ml_predictions[strong_nmf_pixels] == minority_class)
            majority_agreement = np.mean(ml_predictions[strong_nmf_pixels] == majority_class)
            
            if majority_agreement > minority_agreement:
                flip_likelihood = 0.7
                reason = f"Majority class agrees better with NMF ({majority_agreement:.3f} vs {minority_agreement:.3f})"
            else:
                flip_likelihood = 0.3
                reason = f"Minority class agrees better with NMF ({minority_agreement:.3f} vs {majority_agreement:.3f})"
                
            return {
                'test_name': 'nmf_agreement',
                'flip_likelihood': flip_likelihood,
                'reason': reason,
                'minority_agreement': minority_agreement,
                'majority_agreement': majority_agreement,
                'best_nmf_component': best_component,
                'strong_nmf_pixels': np.sum(strong_nmf_pixels)
            }
            
        except Exception as e:
            logger.warning(f"NMF agreement test failed: {str(e)}")
            return None
            
    def _test_probability_distributions(self, ml_probabilities: np.ndarray, 
                                      minority_class: int, majority_class: int) -> Dict[str, any]:
        """Test probability distributions for signs of class confusion."""
        
        minority_probs = ml_probabilities[:, minority_class]
        majority_probs = ml_probabilities[:, majority_class]
        
        # High-confidence predictions for each class
        high_conf_threshold = 0.8
        
        high_conf_minority = np.sum(minority_probs > high_conf_threshold)
        high_conf_majority = np.sum(majority_probs > high_conf_threshold)
        
        # If minority class has very few high-confidence predictions, might be flipped
        if high_conf_minority == 0 and high_conf_majority > 100:
            flip_likelihood = 0.7
            reason = f"No high-confidence minority predictions vs {high_conf_majority} majority"
        elif high_conf_minority < 10 and high_conf_majority > 1000:
            flip_likelihood = 0.6
            reason = f"Very few high-confidence minority predictions ({high_conf_minority} vs {high_conf_majority})"
        else:
            flip_likelihood = 0.4
            reason = f"Reasonable confidence distribution ({high_conf_minority} minority, {high_conf_majority} majority)"
            
        return {
            'test_name': 'probability_distribution',
            'flip_likelihood': flip_likelihood,
            'reason': reason,
            'high_conf_minority': high_conf_minority,
            'high_conf_majority': high_conf_majority,
            'minority_mean_prob': np.mean(minority_probs),
            'majority_mean_prob': np.mean(majority_probs)
        }
        
    def correct_class_flip(self, ml_probabilities: np.ndarray, 
                          ml_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct class flipping by swapping class labels.
        
        Args:
            ml_probabilities: Original probabilities
            ml_predictions: Original predictions
            
        Returns:
            Corrected probabilities and predictions
        """
        
        if not self.flip_detected:
            logger.info("No class flip detected - returning original results")
            return ml_probabilities.copy(), ml_predictions.copy()
            
        logger.info(f"Correcting class flip (confidence: {self.confidence_score:.3f})")
        
        # Swap probabilities (flip columns)
        corrected_probabilities = ml_probabilities[:, ::-1]  # Reverse column order
        
        # Swap predictions  
        corrected_predictions = 1 - ml_predictions  # Flip 0<->1 for binary classification
        
        return corrected_probabilities, corrected_predictions
        
    def generate_diagnostic_report(self) -> str:
        """Generate a human-readable diagnostic report."""
        
        if not hasattr(self, 'diagnostic_info'):
            return "No diagnostic information available. Run detect_class_flip() first."
            
        report = ["ML CLASS FLIP DIAGNOSTIC REPORT"]
        report.append("=" * 40)
        
        info = self.diagnostic_info
        
        report.append(f"Flip Detection: {'YES' if self.flip_detected else 'NO'}")
        report.append(f"Confidence: {self.confidence_score:.3f}")
        report.append("")
        
        report.append("Class Distribution:")
        report.append(f"  Minority class {info['minority_class']}: {info['minority_rate']:.3f} ({info['minority_rate']*100:.1f}%)")
        report.append(f"  Majority class {info['majority_class']}: {info['majority_rate']:.3f} ({info['majority_rate']*100:.1f}%)")
        report.append(f"  Expected positive rate: {info['expected_positive_rate']:.3f} ({info['expected_positive_rate']*100:.1f}%)")
        report.append("")
        
        report.append("Individual Tests:")
        for test_name, test_result in info.items():
            if isinstance(test_result, dict) and 'test_name' in test_result:
                report.append(f"  {test_result['test_name']}: {test_result['flip_likelihood']:.3f}")
                report.append(f"    Reason: {test_result['reason']}")
                
        if self.flip_detected:
            report.append("")
            report.append("RECOMMENDATION:")
            report.append(f"  Use class {info['majority_class']} as target class (instead of {info['minority_class']})")
            report.append("  Apply class flip correction before quantitative analysis")
            
        return "\n".join(report) 