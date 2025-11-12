#!/usr/bin/env python3
"""
Test script for the extracted tab functionality.

This script tests that all tabs can be imported and instantiated correctly.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_tab_imports():
    """Test that all tab modules can be imported."""
    print("Testing tab imports...")
    
    try:
        # Test main module import
        from cluster_analysis import RamanClusterAnalysisQt6
        print("✓ Main cluster analysis module imported successfully")
        
        # Test individual tab imports
        from cluster_analysis.ui.tabs.import_tab import ImportTab
        print("✓ ImportTab imported successfully")
        
        from cluster_analysis.ui.tabs.clustering_tab import ClusteringTab
        print("✓ ClusteringTab imported successfully")
        
        from cluster_analysis.ui.tabs.visualization_tabs import VisualizationTab, DendrogramTab, HeatmapTab, ScatterTab
        print("✓ Visualization tabs imported successfully")
        
        from cluster_analysis.ui.tabs.analysis_tab import AnalysisTab
        print("✓ AnalysisTab imported successfully")
        
        from cluster_analysis.ui.tabs.refinement_tab import RefinementTab
        print("✓ RefinementTab imported successfully")
        
        from cluster_analysis.ui.tabs.advanced_tabs import (TimeSeriesTab, KineticsTab, 
                                                          StructuralAnalysisTab, ValidationTab, 
                                                          AdvancedStatisticsTab)
        print("✓ Advanced analysis tabs imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_ui_module_exports():
    """Test that UI module exports all tabs correctly."""
    print("\nTesting UI module exports...")
    
    try:
        from cluster_analysis.ui import (
            ImportTab, ClusteringTab, VisualizationTab, DendrogramTab, 
            HeatmapTab, ScatterTab, AnalysisTab, RefinementTab, 
            TimeSeriesTab, KineticsTab, StructuralAnalysisTab, 
            ValidationTab, AdvancedStatisticsTab
        )
        
        tabs = [
            ImportTab, ClusteringTab, VisualizationTab, DendrogramTab,
            HeatmapTab, ScatterTab, AnalysisTab, RefinementTab,
            TimeSeriesTab, KineticsTab, StructuralAnalysisTab,
            ValidationTab, AdvancedStatisticsTab
        ]
        
        print(f"✓ All {len(tabs)} tabs exported correctly from UI module")
        return True
        
    except ImportError as e:
        print(f"✗ UI module export failed: {e}")
        return False

def test_tab_instantiation():
    """Test that tabs can be instantiated (without full main window)."""
    print("\nTesting tab instantiation...")
    
    try:
        from PySide6.QtWidgets import QApplication
        
        # Create QApplication if it doesn't exist
        if not QApplication.instance():
            app = QApplication([])
        
        # Import tabs
        from cluster_analysis.ui.tabs.import_tab import ImportTab
        from cluster_analysis.ui.tabs.clustering_tab import ClusteringTab
        from cluster_analysis.ui.tabs.visualization_tabs import VisualizationTab
        from cluster_analysis.ui.tabs.analysis_tab import AnalysisTab
        from cluster_analysis.ui.tabs.refinement_tab import RefinementTab
        
        # Mock parent window for testing
        class MockParent:
            def select_import_folder(self):
                pass
            def open_database_import_dialog(self):
                pass
            def update_performance_controls(self):
                pass
            def update_preprocessing_controls(self):
                pass
            def update_carbon_controls(self):
                pass
            def run_clustering(self):
                pass
            def run_probabilistic_clustering(self):
                pass
            def export_analysis_results(self):
                pass
            def toggle_refinement_mode(self):
                pass
            def undo_last_action(self):
                pass
            def split_selected_cluster(self):
                pass
            def merge_selected_clusters(self):
                pass
            def reset_selection(self):
                pass
            def apply_refinement(self):
                pass
            def cancel_refinement(self):
                pass
            def update_refinement_plot(self):
                pass
            def export_visualization(self):
                pass
            def update_dendrogram(self):
                pass
            def update_heatmap(self):
                pass
            def update_scatter_plot(self):
                pass
            def analyze_time_series(self):
                pass
            def plot_time_series(self):
                pass
            def fit_kinetics_model(self):
                pass
            def predict_kinetics(self):
                pass
            def analyze_peak_positions(self):
                pass
            def calculate_band_ratios(self):
                pass
            def extract_structural_parameters(self):
                pass
            def calculate_silhouette_score(self):
                pass
            def calculate_davies_bouldin(self):
                pass
            def calculate_calinski_harabasz(self):
                pass
            def find_optimal_clusters(self):
                pass
            def perform_anova_test(self):
                pass
            def perform_kruskal_test(self):
                pass
            def perform_manova_test(self):
                pass
            def perform_detailed_pca(self):
                pass
            def perform_factor_analysis(self):
                pass
            def plot_probability_heatmap(self):
                pass
            def print_carbon_feature_analysis(self):
                pass
            def suggest_clustering_improvements(self):
                pass
            def show_nmf_clustering_info(self):
                pass
            def plot_dendrogram(self, param):
                pass
            def import_from_main_app(self):
                pass
            def import_single_file_map(self):
                pass
            def start_batch_import(self):
                pass
            def append_data(self):
                pass
        
        mock_parent = MockParent()
        
        # Test basic tab instantiation
        import_tab = ImportTab(mock_parent)
        print("✓ ImportTab instantiated successfully")
        
        clustering_tab = ClusteringTab(mock_parent)
        print("✓ ClusteringTab instantiated successfully")
        
        viz_tab = VisualizationTab(mock_parent)
        print("✓ VisualizationTab instantiated successfully")
        
        analysis_tab = AnalysisTab(mock_parent)
        print("✓ AnalysisTab instantiated successfully")
        
        refinement_tab = RefinementTab(mock_parent)
        print("✓ RefinementTab instantiated successfully")
        
        # Test that tabs have expected methods
        assert hasattr(import_tab, 'get_import_progress'), "ImportTab missing get_import_progress method"
        assert hasattr(clustering_tab, 'get_clustering_controls'), "ClusteringTab missing get_clustering_controls method"
        assert hasattr(viz_tab, 'get_viz_tab_widget'), "VisualizationTab missing get_viz_tab_widget method"
        assert hasattr(analysis_tab, 'get_analysis_results_text'), "AnalysisTab missing get_analysis_results_text method"
        assert hasattr(refinement_tab, 'get_refinement_controls'), "RefinementTab missing get_refinement_controls method"
        
        print("✓ All tabs have expected methods")
        return True
        
    except Exception as e:
        print(f"✗ Tab instantiation failed: {e}")
        return False

def test_main_window_with_extracted_tabs():
    """Test that main window works with extracted tabs."""
    print("\nTesting main window with extracted tabs...")
    
    try:
        from PySide6.QtWidgets import QApplication
        
        # Create QApplication if it doesn't exist
        if not QApplication.instance():
            app = QApplication([])
        
        from cluster_analysis import RamanClusterAnalysisQt6
        
        # Test main window creation
        main_window = RamanClusterAnalysisQt6()
        print("✓ Main window created successfully")
        
        # Test that all tabs are present
        tab_widget = main_window.tab_widget
        tab_count = tab_widget.count()
        
        expected_tabs = [
            "Import", "Clustering", "Visualization", "Analysis", 
            "Refinement", "Time Series", "Kinetics", "Structural", 
            "Validation", "Statistics"
        ]
        
        print(f"✓ Found {tab_count} tabs in main window")
        
        for i in range(tab_count):
            tab_text = tab_widget.tabText(i)
            assert tab_text in expected_tabs, f"Unexpected tab: {tab_text}"
            print(f"  ✓ Tab {i}: {tab_text}")
        
        # Test that tab references exist
        assert hasattr(main_window, 'import_tab'), "Main window missing import_tab reference"
        assert hasattr(main_window, 'clustering_tab'), "Main window missing clustering_tab reference"
        assert hasattr(main_window, 'visualization_tab'), "Main window missing visualization_tab reference"
        assert hasattr(main_window, 'analysis_tab'), "Main window missing analysis_tab reference"
        assert hasattr(main_window, 'refinement_tab'), "Main window missing refinement_tab reference"
        
        print("✓ All tab references present in main window")
        return True
        
    except Exception as e:
        print(f"✗ Main window test failed: {e}")
        return False

def main():
    """Run all tab extraction tests."""
    print("=" * 60)
    print("Testing Extracted Tab Functionality")
    print("=" * 60)
    
    tests = [
        test_tab_imports,
        test_ui_module_exports,
        test_tab_instantiation,
        test_main_window_with_extracted_tabs
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Tab Extraction Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tab extraction tests passed!")
        print("\n📋 Summary of extracted tabs:")
        print("  • ImportTab - Data import and performance controls")
        print("  • ClusteringTab - Clustering parameters and execution")
        print("  • VisualizationTab - Dendrogram, heatmap, and scatter plots")
        print("  • AnalysisTab - Results display and export")
        print("  • RefinementTab - Interactive cluster refinement")
        print("  • TimeSeriesTab - Time-series progression analysis")
        print("  • KineticsTab - Kinetics modeling")
        print("  • StructuralAnalysisTab - Structural characterization")
        print("  • ValidationTab - Cluster validation metrics")
        print("  • AdvancedStatisticsTab - Statistical analysis tools")
        return 0
    else:
        print("❌ Some tab extraction tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
