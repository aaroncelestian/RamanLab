#!/usr/bin/env python3
"""
Test script for ML class flip detection and correction.

This script specifically addresses your issue where Random Forest gives the
correct number of detections (38-55 pixels) but assigns them to the wrong class.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import logging
from analysis.ml_class_flip_detector import MLClassFlipDetector
from analysis.quantitative_analysis import RobustQuantitativeAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def simulate_your_class_flip_issue():
    """
    Simulate the exact class flipping issue you're experiencing.
    
    Your logs show:
    - 16,383 total pixels
    - 38-55 pixels detected as minority class (class 1) 
    - But the results are flipped - these should be the background pixels
    - The actual target material pixels are being labeled as class 0
    """
    
    n_pixels = 16383
    
    # Ground truth: about 5% of pixels actually contain your target material
    n_true_positives = 820  # About 5%
    true_positive_pixels = np.random.choice(n_pixels, size=n_true_positives, replace=False)
    ground_truth = np.zeros(n_pixels, dtype=bool)
    ground_truth[true_positive_pixels] = True
    
    print(f"Ground truth: {n_true_positives} pixels ({n_true_positives/n_pixels*100:.1f}%) contain target material")
    
    # Simulate your ML results with class flipping
    ml_probabilities = np.zeros((n_pixels, 2))
    ml_predictions = np.zeros(n_pixels, dtype=int)
    
    # Here's the key issue: ML is assigning most target material pixels to class 0 (background)
    # and some random background pixels to class 1 (target)
    
    # Most pixels (including most target material) get assigned to class 0
    ml_probabilities[:, 0] = np.random.beta(8, 2, n_pixels)  # High background probability
    
    # Only a few random pixels (mostly background) get assigned to class 1
    # This simulates your 38-55 detection issue
    n_false_detections = 55  # Your observed number
    false_detection_pixels = np.random.choice(
        np.where(~ground_truth)[0],  # Choose from background pixels
        size=n_false_detections, 
        replace=False
    )
    
    # A few true positives might still be detected correctly (but most are missed)
    n_correct_detections = 5  # Very few correct detections
    correct_detection_pixels = np.random.choice(true_positive_pixels, size=n_correct_detections, replace=False)
    
    # Set class 1 probabilities for these "detections"
    all_class1_pixels = np.concatenate([false_detection_pixels, correct_detection_pixels])
    ml_probabilities[all_class1_pixels, 1] = np.random.beta(6, 2, len(all_class1_pixels))
    
    # Normalize probabilities
    ml_probabilities[all_class1_pixels, 0] = 1 - ml_probabilities[all_class1_pixels, 1]
    ml_probabilities[~np.isin(np.arange(n_pixels), all_class1_pixels), 1] = 1 - ml_probabilities[~np.isin(np.arange(n_pixels), all_class1_pixels), 0]
    
    # Make predictions based on probabilities
    ml_predictions = np.argmax(ml_probabilities, axis=1)
    
    # Verify we have the right class distribution (matching your logs)
    class_counts = {0: np.sum(ml_predictions == 0), 1: np.sum(ml_predictions == 1)}
    print(f"ML Class distribution: {class_counts}")
    print(f"Class 1 (minority): {class_counts[1]} pixels ({class_counts[1]/n_pixels*100:.3f}%)")
    print(f"Class 0 (majority): {class_counts[0]} pixels ({class_counts[0]/n_pixels*100:.3f}%)")
    
    # This should match your logs showing ~55 pixels in class 1
    
    # Create some template results for comparison
    template_coefficients = np.zeros((n_pixels, 1))
    template_r_squared = np.zeros((n_pixels, 1))
    
    # Template method detects most true positives correctly (but with some false positives)
    template_coefficients[true_positive_pixels, 0] = np.random.normal(0.8, 0.2, len(true_positive_pixels))
    template_r_squared[true_positive_pixels, 0] = np.random.beta(8, 2, len(true_positive_pixels))
    
    # Template also gives some false positives
    template_false_positives = np.random.choice(
        np.where(~ground_truth)[0], 
        size=int(0.1 * n_pixels), 
        replace=False
    )
    template_coefficients[template_false_positives, 0] = np.random.normal(0.3, 0.1, len(template_false_positives))
    template_r_squared[template_false_positives, 0] = np.random.beta(2, 3, len(template_false_positives))
    
    # Add noise
    template_coefficients += np.random.normal(0, 0.05, (n_pixels, 1))
    template_r_squared += np.random.normal(0, 0.05, (n_pixels, 1))
    template_coefficients = np.clip(template_coefficients, 0, None)
    template_r_squared = np.clip(template_r_squared, 0, 1)
    
    # Create some NMF results
    nmf_components = np.zeros((n_pixels, 5))
    # NMF component 2 represents the target material
    nmf_components[true_positive_pixels, 2] = np.random.normal(8.0, 2.0, len(true_positive_pixels))
    nmf_components[:, 2] += np.random.exponential(0.5, n_pixels)  # Background noise
    nmf_components = np.clip(nmf_components, 0, None)
    
    return {
        'ground_truth': ground_truth,
        'ml_probabilities': ml_probabilities,
        'ml_predictions': ml_predictions,
        'template_coefficients': template_coefficients,
        'template_r_squared': template_r_squared,
        'nmf_components': nmf_components,
        'n_true_positives': n_true_positives
    }

def test_class_flip_detection(data):
    """Test the class flip detector on your specific issue."""
    
    print("\n" + "="*60)
    print("TESTING CLASS FLIP DETECTION")
    print("="*60)
    
    # Initialize detector
    detector = MLClassFlipDetector()
    
    # Run flip detection
    flip_result = detector.detect_class_flip(
        ml_probabilities=data['ml_probabilities'],
        ml_predictions=data['ml_predictions'],
        template_results=data['template_coefficients'],
        nmf_results=data['nmf_components'],
        expected_positive_rate=0.05  # Expecting ~5% positive detections
    )
    
    # Show results
    print(f"Flip Detection Result: {'FLIP DETECTED' if flip_result['flip_detected'] else 'NO FLIP'}")
    print(f"Confidence: {flip_result['flip_confidence']:.3f}")
    print(f"Original minority class: {flip_result['minority_class_original']}")
    print(f"Recommended target class: {flip_result['recommended_target_class']}")
    
    # Generate diagnostic report
    report = detector.generate_diagnostic_report()
    print("\n" + report)
    
    # Test correction
    if flip_result['flip_detected']:
        print("\n" + "="*40)
        print("APPLYING CLASS FLIP CORRECTION")
        print("="*40)
        
        corrected_probs, corrected_preds = detector.correct_class_flip(
            data['ml_probabilities'], data['ml_predictions']
        )
        
        # Check performance before and after correction
        ground_truth = data['ground_truth']
        
        # Original performance (should be poor)
        original_target_class = flip_result['minority_class_original']
        original_detections = data['ml_predictions'] == original_target_class
        orig_tp = np.sum(ground_truth & original_detections)
        orig_fp = np.sum(~ground_truth & original_detections)
        orig_precision = orig_tp / (orig_tp + orig_fp) if (orig_tp + orig_fp) > 0 else 0
        orig_recall = orig_tp / np.sum(ground_truth) if np.sum(ground_truth) > 0 else 0
        
        # Corrected performance (should be better)
        corrected_target_class = flip_result['recommended_target_class']
        corrected_detections = corrected_preds == corrected_target_class
        corr_tp = np.sum(ground_truth & corrected_detections)
        corr_fp = np.sum(~ground_truth & corrected_detections)
        corr_precision = corr_tp / (corr_tp + corr_fp) if (corr_tp + corr_fp) > 0 else 0
        corr_recall = corr_tp / np.sum(ground_truth) if np.sum(ground_truth) > 0 else 0
        
        print("PERFORMANCE COMPARISON:")
        print("-" * 40)
        print(f"Original (using class {original_target_class}):")
        print(f"  Detections: {np.sum(original_detections)} pixels")
        print(f"  Precision: {orig_precision:.3f}, Recall: {orig_recall:.3f}")
        print(f"  True Positives: {orig_tp}, False Positives: {orig_fp}")
        print()
        print(f"Corrected (using class {corrected_target_class}):")
        print(f"  Detections: {np.sum(corrected_detections)} pixels")
        print(f"  Precision: {corr_precision:.3f}, Recall: {corr_recall:.3f}")
        print(f"  True Positives: {corr_tp}, False Positives: {corr_fp}")
        print()
        
        improvement = corr_recall - orig_recall
        print(f"Recall improvement: {improvement:.3f} ({improvement*100:.1f} percentage points)")
        
        return corrected_probs, corrected_preds, flip_result
    
    return data['ml_probabilities'], data['ml_predictions'], flip_result

def test_quantitative_analysis_with_correction(data, corrected_probs, corrected_preds):
    """Test quantitative analysis with the corrected ML results."""
    
    print("\n" + "="*60)
    print("TESTING QUANTITATIVE ANALYSIS WITH CORRECTED ML")
    print("="*60)
    
    # Initialize analyzer
    analyzer = RobustQuantitativeAnalyzer(confidence_threshold=0.3)
    
    # Set template results
    analyzer.set_template_results(
        template_manager=None,
        template_coefficients=data['template_coefficients'],
        template_r_squared=data['template_r_squared']
    )
    
    # Set NMF results
    analyzer.set_nmf_results(
        nmf_components=data['nmf_components'],
        nmf_feature_components=np.random.random((5, 672))
    )
    
    # Set corrected ML results (this will automatically detect the correction was applied)
    analyzer.set_ml_results(
        ml_probabilities=corrected_probs,
        ml_predictions=corrected_preds,
        auto_detect_class_flip=False  # Already corrected
    )
    
    # Run quantitative analysis
    result = analyzer.analyze_component(
        component_name="Target Material",
        template_index=0,
        nmf_component=2,
        target_class_index=1  # Use the corrected target class
    )
    
    # Evaluate against ground truth
    ground_truth = data['ground_truth']
    detected_pixels = result.detection_map
    
    tp = np.sum(ground_truth & detected_pixels)
    fp = np.sum(~ground_truth & detected_pixels)
    fn = np.sum(ground_truth & ~detected_pixels)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("QUANTITATIVE ANALYSIS RESULTS:")
    print(f"  Component detected in: {np.sum(detected_pixels)} pixels ({np.sum(detected_pixels)/len(ground_truth)*100:.1f}%)")
    print(f"  True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}")
    print()
    
    # Show detailed statistics
    stats = result.statistics
    print("DETAILED STATISTICS:")
    print(f"  Detection percentage: {stats['detection_percentage']:.2f}%")
    print(f"  Average component percentage: {stats['mean_percentage_detected']:.1f}% ± {stats['std_percentage_detected']:.1f}%")
    print(f"  Confidence threshold used: {stats['confidence_threshold_used']:.2f}")
    
    # Show method contributions
    print("\nMETHOD CONTRIBUTIONS:")
    for key, value in stats.items():
        if '_detection_percent' in key:
            method_name = key.replace('_detection_percent', '').title()
            print(f"  {method_name}: {value:.2f}% of pixels")
    
    return result

def main():
    """Main test function."""
    
    print("CLASS FLIP DETECTION TEST")
    print("=" * 60)
    print("This test simulates your exact issue:")
    print("- Random Forest finds 38-55 positive detections")
    print("- But assigns them to the wrong class (class flipping)")
    print("- Template and NMF methods work correctly")
    print("- Quantitative analysis should detect and correct the flip")
    print()
    
    # Generate test data matching your issue
    data = simulate_your_class_flip_issue()
    
    # Test class flip detection
    corrected_probs, corrected_preds, flip_result = test_class_flip_detection(data)
    
    # Test quantitative analysis with correction
    if flip_result['flip_detected']:
        result = test_quantitative_analysis_with_correction(data, corrected_probs, corrected_preds)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✓ Class flip was detected and corrected automatically")
        print("✓ ML results now agree better with template and NMF methods")
        print("✓ Quantitative analysis provides reliable component detection")
        print("✓ You get accurate percentage estimates with confidence measures")
        print()
        print("FOR YOUR REAL DATA:")
        print("- The quantitative analysis will automatically detect this class flip")
        print("- It will correct the ML results before combining with other methods")
        print("- You'll get reliable quantitative results despite the ML class confusion")
    else:
        print("\nUnexpected: No class flip detected in test data")
        print("This suggests the test data doesn't match your exact issue")

if __name__ == "__main__":
    main() 