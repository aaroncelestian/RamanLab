#!/usr/bin/env python3
"""
Demonstration of Robust Quantitative Analysis

This script shows how to use the new quantitative analysis approach to get
reliable component identification and percentage estimates from your Raman maps.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from analysis.quantitative_analysis import RobustQuantitativeAnalyzer, ComponentResult

# Import safe file handling
from pkl_utils import get_workspace_root, get_example_data_paths, print_available_example_files, get_example_spectrum_file
from utils.file_loaders import load_spectrum_file
from pathlib import Path

def setup_matplotlib_config():
    """Set up matplotlib configuration using the workspace config."""
    try:
        # Try to import the matplotlib config
        workspace_root = get_workspace_root()
        polarization_ui_dir = workspace_root / "polarization_ui"
        
        if (polarization_ui_dir / "matplotlib_config.py").exists():
            import sys
            sys.path.insert(0, str(polarization_ui_dir))
            import matplotlib_config
            print("✅ Using RamanLab matplotlib configuration")
            return True
        else:
            # Use default matplotlib settings
            plt.style.use('default')
            plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 10})
            print("⚠️  Using default matplotlib configuration")
            return False
    except Exception as e:
        print(f"❌ Error setting up matplotlib: {e}")
        plt.style.use('default')
        plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 10})
        return False

def load_real_map_data():
    """
    Load real map data if available.
    
    Returns:
        dict: Dictionary containing real map data or None if not available
    """
    try:
        # Get available example files
        paths = get_example_data_paths()
        
        # Look for pickle files that might contain map data
        map_data_files = []
        for key, path in paths.items():
            if isinstance(path, Path) and path.suffix == '.pkl':
                if 'batch_results' in key or 'map' in key.lower():
                    map_data_files.append((key, path))
        
        if map_data_files:
            print(f"📄 Found {len(map_data_files)} potential map data files:")
            for key, path in map_data_files:
                print(f"   • {key}: {path.name}")
            
            # Try to load the first one
            key, path = map_data_files[0]
            print(f"\n📊 Loading map data from: {path.name}")
            
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✅ Successfully loaded map data:")
            print(f"   • Data type: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"   • Shape: {data.shape}")
            elif isinstance(data, dict):
                print(f"   • Dictionary keys: {list(data.keys())[:10]}...")  # First 10 keys
                
            return data
            
        else:
            print("⚠️  No map data files found")
            return None
            
    except Exception as e:
        print(f"❌ Error loading map data: {e}")
        return None

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
    
    print("🔍 INDIVIDUAL METHOD PERFORMANCE")
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
    
    print(f"📊 Template Method:")
    print(f"   Detected: {np.sum(template_detections)} pixels ({np.sum(template_detections)/len(ground_truth)*100:.1f}%)")
    print(f"   True Positives: {template_tp}, False Positives: {template_fp}, False Negatives: {template_fn}")
    print(f"   Precision: {template_precision:.3f}, Recall: {template_recall:.3f}")
    print(f"   Issue: Too many false positives (overestimation)")
    print()
    
    # NMF method performance
    nmf_detections = data['nmf_components'][:, 2] > 3.0  # Simple threshold
    nmf_tp = np.sum(ground_truth & nmf_detections)
    nmf_fp = np.sum(~ground_truth & nmf_detections)
    nmf_fn = np.sum(ground_truth & ~nmf_detections)
    
    nmf_precision = nmf_tp / (nmf_tp + nmf_fp) if (nmf_tp + nmf_fp) > 0 else 0
    nmf_recall = nmf_tp / (nmf_tp + nmf_fn) if (nmf_tp + nmf_fn) > 0 else 0
    
    print(f"📊 NMF Method:")
    print(f"   Detected: {np.sum(nmf_detections)} pixels ({np.sum(nmf_detections)/len(ground_truth)*100:.1f}%)")
    print(f"   True Positives: {nmf_tp}, False Positives: {nmf_fp}, False Negatives: {nmf_fn}")
    print(f"   Precision: {nmf_precision:.3f}, Recall: {nmf_recall:.3f}")
    print(f"   Issue: Poor recall (underestimation)")
    print()
    
    # ML method performance
    ml_detections = data['ml_predictions'] == 1
    ml_tp = np.sum(ground_truth & ml_detections)
    ml_fp = np.sum(~ground_truth & ml_detections)
    ml_fn = np.sum(ground_truth & ~ml_detections)
    
    ml_precision = ml_tp / (ml_tp + ml_fp) if (ml_tp + ml_fp) > 0 else 0
    ml_recall = ml_tp / (ml_tp + ml_fn) if (ml_tp + ml_fn) > 0 else 0
    
    print(f"📊 ML Method:")
    print(f"   Detected: {np.sum(ml_detections)} pixels ({np.sum(ml_detections)/len(ground_truth)*100:.1f}%)")
    print(f"   True Positives: {ml_tp}, False Positives: {ml_fp}, False Negatives: {ml_fn}")
    print(f"   Precision: {ml_precision:.3f}, Recall: {ml_recall:.3f}")
    print(f"   Issue: Poor recall due to class imbalance")
    print()

def demonstrate_quantitative_analysis(data):
    """Demonstrate the new quantitative analysis approach."""
    
    print("🔬 ROBUST QUANTITATIVE ANALYSIS")
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 Quantitative Analysis Results:")
    print(f"   • Component: {result.component_name}")
    print(f"   • Detected pixels: {np.sum(detected_pixels)} ({np.sum(detected_pixels)/len(ground_truth)*100:.1f}%)")
    print(f"   • Confidence levels: {len(result.confidence_levels)} levels")
    print(f"   • Agreement score: {result.agreement_score:.3f}")
    print(f"   • Precision: {precision:.3f}")
    print(f"   • Recall: {recall:.3f}")
    print(f"   • F1-Score: {f1:.3f}")
    
    return result

def create_comparison_plots(data, result):
    """Create comprehensive comparison plots."""
    
    print("\n📊 Creating Comparison Plots...")
    
    # Setup plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Quantitative Analysis Comparison', fontsize=16, fontweight='bold')
    
    def reshape_for_plot(arr):
        """Reshape 1D array to 2D for plotting."""
        if len(arr) == 16383:
            # Try to make a roughly square image
            side = int(np.sqrt(len(arr)))
            if side * side < len(arr):
                side += 1
            padded = np.zeros(side * side)
            padded[:len(arr)] = arr
            return padded.reshape(side, side)
        else:
            return arr.reshape(-1, 1)
    
    # Plot 1: Ground Truth
    ax = axes[0, 0]
    ground_truth_2d = reshape_for_plot(data['ground_truth'])
    im1 = ax.imshow(ground_truth_2d, cmap='Reds', aspect='auto')
    ax.set_title('Ground Truth')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(im1, ax=ax, shrink=0.8)
    
    # Plot 2: Template Method
    ax = axes[0, 1]
    template_2d = reshape_for_plot(data['template_coefficients'][:, 0])
    im2 = ax.imshow(template_2d, cmap='Blues', aspect='auto')
    ax.set_title('Template Method')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax, shrink=0.8)
    
    # Plot 3: NMF Method
    ax = axes[0, 2]
    nmf_2d = reshape_for_plot(data['nmf_components'][:, 2])
    im3 = ax.imshow(nmf_2d, cmap='Greens', aspect='auto')
    ax.set_title('NMF Method')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(im3, ax=ax, shrink=0.8)
    
    # Plot 4: ML Method
    ax = axes[1, 0]
    ml_2d = reshape_for_plot(data['ml_probabilities'][:, 1])
    im4 = ax.imshow(ml_2d, cmap='Purples', aspect='auto')
    ax.set_title('ML Method')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(im4, ax=ax, shrink=0.8)
    
    # Plot 5: Quantitative Analysis Result
    ax = axes[1, 1]
    result_2d = reshape_for_plot(result.detection_map.astype(float))
    im5 = ax.imshow(result_2d, cmap='Oranges', aspect='auto')
    ax.set_title('Quantitative Analysis')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(im5, ax=ax, shrink=0.8)
    
    # Plot 6: Performance Comparison
    ax = axes[1, 2]
    methods = ['Template', 'NMF', 'ML', 'Quantitative']
    
    # Calculate F1 scores for each method
    f1_scores = []
    
    for method_data in [data['template_coefficients'][:, 0] > 0.2,
                       data['nmf_components'][:, 2] > 3.0,
                       data['ml_predictions'] == 1,
                       result.detection_map]:
        
        tp = np.sum(data['ground_truth'] & method_data)
        fp = np.sum(~data['ground_truth'] & method_data)
        fn = np.sum(data['ground_truth'] & ~method_data)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    bars = ax.bar(methods, f1_scores, color=['blue', 'green', 'purple', 'orange'], alpha=0.7)
    ax.set_title('F1-Score Comparison')
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    workspace_root = get_workspace_root()
    results_dir = workspace_root / "quantitative_analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    plot_file = results_dir / "quantitative_analysis_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {plot_file}")
    
    plt.show()

def main():
    """Main demonstration function."""
    print("🔬 Quantitative Analysis Demonstration")
    print("=" * 50)
    
    # Set up matplotlib configuration
    setup_matplotlib_config()
    
    # Show available example files
    print("\n📁 Available Example Data Files:")
    print_available_example_files()
    
    # Try to load real map data
    print("\n📊 Loading Real Map Data:")
    real_data = load_real_map_data()
    
    if real_data:
        print("✅ Real map data loaded successfully!")
        print("   This demo will use synthetic data for controlled testing,")
        print("   but the methods can be applied to your real data.")
    else:
        print("⚠️  No real map data found, using synthetic data for demonstration.")
    
    # Generate synthetic data for demonstration
    print("\n🧪 Generating Synthetic Data for Controlled Testing:")
    data = simulate_realistic_data()
    
    print(f"   • Total pixels: {data['n_pixels']:,}")
    print(f"   • True component pixels: {np.sum(data['ground_truth'])} ({np.sum(data['ground_truth'])/data['n_pixels']*100:.1f}%)")
    print(f"   • Template detections: {np.sum(data['template_coefficients'][:, 0] > 0.2)}")
    print(f"   • NMF detections: {np.sum(data['nmf_components'][:, 2] > 3.0)}")
    print(f"   • ML detections: {np.sum(data['ml_predictions'] == 1)}")
    
    # Evaluate individual methods
    print("\n📈 Evaluating Individual Methods:")
    evaluate_method_performance(data)
    
    # Demonstrate quantitative analysis
    print("\n🔬 Demonstrating Quantitative Analysis:")
    result = demonstrate_quantitative_analysis(data)
    
    # Create comparison plots
    print("\n📊 Creating Visualization:")
    create_comparison_plots(data, result)
    
    # Save results
    workspace_root = get_workspace_root()
    results_dir = workspace_root / "quantitative_analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "quantitative_analysis_results.txt"
    with open(results_file, 'w') as f:
        f.write("Quantitative Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Component: {result.component_name}\n")
        f.write(f"Detected pixels: {np.sum(result.detection_map)}\n")
        f.write(f"Agreement score: {result.agreement_score:.3f}\n")
        f.write(f"Confidence levels: {len(result.confidence_levels)}\n")
    
    print(f"📄 Results saved to: {results_file}")
    
    print("\n✅ Demonstration Complete!")
    print("   • Individual method analysis: ✅")
    print("   • Quantitative analysis: ✅")
    print("   • Comparison plots: ✅")
    print("   • Results saved: ✅")

if __name__ == "__main__":
    main() 