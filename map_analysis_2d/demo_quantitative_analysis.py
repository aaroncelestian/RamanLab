#!/usr/bin/env python3
"""
Demonstration of Robust Quantitative Analysis

This script shows how to use the new quantitative analysis approach to get
reliable component identification and percentage estimates from your Raman maps.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from analysis.quantitative_analysis import RobustQuantitativeAnalyzer, ComponentResult

# Simple matplotlib configuration
plt.style.use('default')
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 10})

def simulate_realistic_data():
    """
    Simulate realistic Raman map data with known ground truth for validation.
    This represents the issues you're experiencing.
    """
    n_pixels = 16383  # Similar to your map size
    n_wavenumbers = 672
    
    # Create ground truth: 5% of pixels contain your component of interest
    true_component_pixels = np.random.choice(n_pixels, size=int(0.05 * n_pixels), replace=False)
    ground_truth = np.zeros(n_pixels, dtype=bool)
    ground_truth[true_component_pixels] = True
    
    # Simulate template fitting results (your current issue: too many detections)
    template_coefficients = np.zeros((n_pixels, 1))
    template_r_squared = np.zeros((n_pixels, 1))
    
    # True positives: good fits where component actually exists
    template_coefficients[true_component_pixels, 0] = np.random.normal(0.8, 0.2, len(true_component_pixels))
    template_r_squared[true_component_pixels, 0] = np.random.beta(8, 2, len(true_component_pixels))  # High R²
    
    # False positives: template method detects component where it doesn't exist (your issue)
    false_positive_pixels = np.random.choice(
        np.where(~ground_truth)[0], 
        size=int(0.15 * n_pixels),  # 15% false positive rate (too high!)
        replace=False
    )
    template_coefficients[false_positive_pixels, 0] = np.random.normal(0.3, 0.1, len(false_positive_pixels))
    template_r_squared[false_positive_pixels, 0] = np.random.beta(2, 3, len(false_positive_pixels))  # Lower R²
    
    # Add noise to all measurements
    template_coefficients += np.random.normal(0, 0.05, (n_pixels, 1))
    template_r_squared += np.random.normal(0, 0.05, (n_pixels, 1))
    template_coefficients = np.clip(template_coefficients, 0, None)
    template_r_squared = np.clip(template_r_squared, 0, 1)
    
    # Simulate NMF results (your current issue: underestimation, scale mismatch)
    nmf_components = np.zeros((n_pixels, 5))
    
    # True positives: strong NMF signal where component exists
    nmf_components[true_component_pixels, 2] = np.random.normal(8.0, 2.0, len(true_component_pixels))
    
    # False negatives: weak NMF signal where component exists (underestimation issue)
    weak_detection_pixels = np.random.choice(true_component_pixels, size=len(true_component_pixels)//3, replace=False)
    nmf_components[weak_detection_pixels, 2] = np.random.normal(2.0, 0.5, len(weak_detection_pixels))
    
    # Background noise in NMF
    nmf_components[:, 2] += np.random.exponential(0.5, n_pixels)
    nmf_components = np.clip(nmf_components, 0, None)
    
    # Simulate ML results (your current issue: class imbalance, few detections)
    ml_probabilities = np.zeros((n_pixels, 2))  # [background, component]
    ml_predictions = np.zeros(n_pixels, dtype=int)
    
    # Background class (majority)
    ml_probabilities[~ground_truth, 0] = np.random.beta(8, 2, np.sum(~ground_truth))  # High background prob
    ml_probabilities[~ground_truth, 1] = 1 - ml_probabilities[~ground_truth, 0]
    
    # Component class (minority) - but with some missed detections
    detected_by_ml = np.random.choice(true_component_pixels, size=len(true_component_pixels)//2, replace=False)  # Only half detected
    ml_probabilities[detected_by_ml, 1] = np.random.beta(6, 2, len(detected_by_ml))  # High component prob
    ml_probabilities[detected_by_ml, 0] = 1 - ml_probabilities[detected_by_ml, 1]
    
    # Missed detections by ML
    missed_by_ml = np.setdiff1d(true_component_pixels, detected_by_ml)
    ml_probabilities[missed_by_ml, 0] = np.random.beta(6, 3, len(missed_by_ml))
    ml_probabilities[missed_by_ml, 1] = 1 - ml_probabilities[missed_by_ml, 0]
    
    ml_predictions = np.argmax(ml_probabilities, axis=1)
    
    return {
        'ground_truth': ground_truth,
        'template_coefficients': template_coefficients,
        'template_r_squared': template_r_squared,
        'nmf_components': nmf_components,
        'ml_probabilities': ml_probabilities,
        'ml_predictions': ml_predictions,
        'n_pixels': n_pixels,
        'true_positive_rate_expected': 0.05
    }

def evaluate_method_performance(data):
    """Evaluate how each individual method performs vs ground truth."""
    
    ground_truth = data['ground_truth']
    n_true_positives = np.sum(ground_truth)
    
    print("INDIVIDUAL METHOD PERFORMANCE")
    print("=" * 50)
    print(f"Ground truth: {n_true_positives} pixels ({n_true_positives/len(ground_truth)*100:.1f}%) contain component")
    print()
    
    # Template method performance
    template_detections = data['template_coefficients'][:, 0] > 0.2  # Simple threshold
    template_tp = np.sum(ground_truth & template_detections)
    template_fp = np.sum(~ground_truth & template_detections)
    template_fn = np.sum(ground_truth & ~template_detections)
    
    template_precision = template_tp / (template_tp + template_fp) if (template_tp + template_fp) > 0 else 0
    template_recall = template_tp / (template_tp + template_fn) if (template_tp + template_fn) > 0 else 0
    
    print(f"Template Method:")
    print(f"  Detected: {np.sum(template_detections)} pixels ({np.sum(template_detections)/len(ground_truth)*100:.1f}%)")
    print(f"  True Positives: {template_tp}, False Positives: {template_fp}, False Negatives: {template_fn}")
    print(f"  Precision: {template_precision:.3f}, Recall: {template_recall:.3f}")
    print(f"  Issue: Too many false positives (overestimation)")
    print()
    
    # NMF method performance
    nmf_detections = data['nmf_components'][:, 2] > 3.0  # Simple threshold
    nmf_tp = np.sum(ground_truth & nmf_detections)
    nmf_fp = np.sum(~ground_truth & nmf_detections)
    nmf_fn = np.sum(ground_truth & ~nmf_detections)
    
    nmf_precision = nmf_tp / (nmf_tp + nmf_fp) if (nmf_tp + nmf_fp) > 0 else 0
    nmf_recall = nmf_tp / (nmf_tp + nmf_fn) if (nmf_tp + nmf_fn) > 0 else 0
    
    print(f"NMF Method:")
    print(f"  Detected: {np.sum(nmf_detections)} pixels ({np.sum(nmf_detections)/len(ground_truth)*100:.1f}%)")
    print(f"  True Positives: {nmf_tp}, False Positives: {nmf_fp}, False Negatives: {nmf_fn}")
    print(f"  Precision: {nmf_precision:.3f}, Recall: {nmf_recall:.3f}")
    print(f"  Issue: Poor recall (underestimation)")
    print()
    
    # ML method performance
    ml_detections = data['ml_predictions'] == 1
    ml_tp = np.sum(ground_truth & ml_detections)
    ml_fp = np.sum(~ground_truth & ml_detections)
    ml_fn = np.sum(ground_truth & ~ml_detections)
    
    ml_precision = ml_tp / (ml_tp + ml_fp) if (ml_tp + ml_fp) > 0 else 0
    ml_recall = ml_tp / (ml_tp + ml_fn) if (ml_tp + ml_fn) > 0 else 0
    
    print(f"ML Method:")
    print(f"  Detected: {np.sum(ml_detections)} pixels ({np.sum(ml_detections)/len(ground_truth)*100:.1f}%)")
    print(f"  True Positives: {ml_tp}, False Positives: {ml_fp}, False Negatives: {ml_fn}")
    print(f"  Precision: {ml_precision:.3f}, Recall: {ml_recall:.3f}")
    print(f"  Issue: Poor recall due to class imbalance")
    print()

def demonstrate_quantitative_analysis(data):
    """Demonstrate the new quantitative analysis approach."""
    
    print("ROBUST QUANTITATIVE ANALYSIS")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = RobustQuantitativeAnalyzer(confidence_threshold=0.3)
    
    # Set up the data
    analyzer.set_template_results(
        template_manager=None,  # Not needed for this demo
        template_coefficients=data['template_coefficients'],
        template_r_squared=data['template_r_squared']
    )
    
    analyzer.set_nmf_results(
        nmf_components=data['nmf_components'],
        nmf_feature_components=np.random.random((5, 672))  # Dummy for demo
    )
    
    analyzer.set_ml_results(
        ml_probabilities=data['ml_probabilities'],
        ml_predictions=data['ml_predictions']
    )
    
    # Analyze the component using all available methods
    result = analyzer.analyze_component(
        component_name="Polypropylene",
        template_index=0,
        nmf_component=2,
        target_class_index=1
    )
    
    # Evaluate performance vs ground truth
    ground_truth = data['ground_truth']
    detected_pixels = result.detection_map
    
    tp = np.sum(ground_truth & detected_pixels)
    fp = np.sum(~ground_truth & detected_pixels)
    fn = np.sum(ground_truth & ~detected_pixels)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"QUANTITATIVE ANALYSIS RESULTS:")
    print(f"  Component detected in: {np.sum(detected_pixels)} pixels ({np.sum(detected_pixels)/len(ground_truth)*100:.1f}%)")
    print(f"  True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}")
    print()
    
    # Show detailed statistics
    print("DETAILED COMPONENT STATISTICS:")
    for key, value in result.statistics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Generate and show summary report
    report = analyzer.generate_summary_report([result])
    print("SUMMARY REPORT:")
    print(report)
    
    return result, analyzer

def create_comparison_plots(data, result):
    """Create plots comparing individual methods vs the quantitative approach."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reshape data for visualization (create a square-ish map)
    n_pixels = len(data['ground_truth'])
    map_size = int(np.sqrt(n_pixels))
    if map_size * map_size < n_pixels:
        map_size += 1
    
    # Pad data to make it square
    pad_size = map_size * map_size - n_pixels
    
    def reshape_for_plot(arr):
        padded = np.pad(arr, (0, pad_size), mode='constant', constant_values=0)
        return padded.reshape(map_size, map_size)
    
    # Ground truth
    axes[0, 0].imshow(reshape_for_plot(data['ground_truth'].astype(float)), cmap='Reds')
    axes[0, 0].set_title('Ground Truth\n(What we want to detect)')
    axes[0, 0].axis('off')
    
    # Template method
    template_map = data['template_coefficients'][:, 0]
    im1 = axes[0, 1].imshow(reshape_for_plot(template_map), cmap='Blues')
    axes[0, 1].set_title('Template Method\n(Too many detections)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # NMF method
    nmf_map = data['nmf_components'][:, 2]
    im2 = axes[0, 2].imshow(reshape_for_plot(nmf_map), cmap='Greens')
    axes[0, 2].set_title('NMF Method\n(Underestimation)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    
    # ML method
    ml_map = data['ml_probabilities'][:, 1]  # Component probability
    im3 = axes[1, 0].imshow(reshape_for_plot(ml_map), cmap='Purples')
    axes[1, 0].set_title('ML Method\n(Class imbalance issues)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # Quantitative analysis - confidence
    confidence_map = result.confidence_map
    im4 = axes[1, 1].imshow(reshape_for_plot(confidence_map), cmap='Oranges')
    axes[1, 1].set_title('Quantitative Analysis\n(Confidence Map)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    # Quantitative analysis - detection
    detection_map = result.detection_map.astype(float)
    axes[1, 2].imshow(reshape_for_plot(detection_map), cmap='Reds')
    axes[1, 2].set_title('Quantitative Analysis\n(Final Detection)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Comparison: Individual Methods vs Quantitative Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.show()

def main():
    """Main demonstration function."""
    
    print("ROBUST QUANTITATIVE ANALYSIS DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows how the new quantitative analysis approach")
    print("addresses the issues you're experiencing with individual methods.")
    print()
    
    # Generate realistic test data
    print("1. Generating realistic test data (simulating your issues)...")
    data = simulate_realistic_data()
    print(f"   Created map with {data['n_pixels']} pixels")
    print(f"   Ground truth: {np.sum(data['ground_truth'])} pixels contain component")
    print()
    
    # Evaluate individual method performance
    print("2. Evaluating individual method performance...")
    evaluate_method_performance(data)
    
    # Demonstrate quantitative analysis
    print("3. Applying robust quantitative analysis...")
    result, analyzer = demonstrate_quantitative_analysis(data)
    
    # Create visualization
    print("4. Creating comparison plots...")
    create_comparison_plots(data, result)
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FROM QUANTITATIVE ANALYSIS:")
    print("=" * 60)
    print("• Template method alone: High false positive rate (overestimation)")
    print("• NMF method alone: Poor recall (underestimation)")  
    print("• ML method alone: Class imbalance issues (too few detections)")
    print("• Quantitative approach: Combines strengths, reduces weaknesses")
    print("• Provides confidence measures and reliable percentage estimates")
    print("• Method weights adjust based on data quality automatically")
    print()
    
    print("NEXT STEPS FOR YOUR REAL DATA:")
    print("• Load your actual template, NMF, and ML results")
    print("• Run quantitative analysis on your component of interest")
    print("• Use confidence threshold to control detection sensitivity")
    print("• Get reliable percentage estimates with uncertainty bounds")

if __name__ == "__main__":
    main() 