"""
Statistical analysis implementations for cluster validation.
These methods will be integrated into the main cluster analysis class.
"""

# This file contains the implementations that need to be copied into main.py
# to replace the stub methods for statistical analysis.

ANOVA_IMPLEMENTATION = '''
    def perform_anova_test(self):
        """Perform ANOVA test on cluster features."""
        try:
            # Check for clustering results
            if 'labels' not in self.cluster_data or 'features_scaled' not in self.cluster_data:
                QMessageBox.warning(self, "No Data", "Please perform clustering first.")
                return
            
            from scipy import stats
            import matplotlib.pyplot as plt
            
            labels = self.cluster_data['labels']
            features = self.cluster_data['features_scaled']
            n_clusters = len(np.unique(labels))
            
            if n_clusters < 2:
                QMessageBox.warning(self, "Insufficient Clusters", 
                                  "Need at least 2 clusters for ANOVA test.")
                return
            
            self.statusBar().showMessage("Performing ANOVA test...")
            
            # Perform ANOVA for each feature
            n_features = features.shape[1]
            f_statistics = np.zeros(n_features)
            p_values = np.zeros(n_features)
            
            for i in range(n_features):
                # Group features by cluster
                groups = [features[labels == k, i] for k in range(n_clusters)]
                f_stat, p_val = stats.f_oneway(*groups)
                f_statistics[i] = f_stat
                p_values[i] = p_val
            
            # Find significant features
            significant_mask = p_values < 0.05
            n_significant = np.sum(significant_mask)
            
            # Display results
            text = "ANOVA TEST RESULTS\\n"
            text += "=" * 50 + "\\n\\n"
            text += f"Number of Clusters: {n_clusters}\\n"
            text += f"Number of Features: {n_features}\\n"
            text += f"Significant Features (p < 0.05): {n_significant} ({100*n_significant/n_features:.1f}%)\\n\\n"
            
            text += "INTERPRETATION:\\n"
            text += "-" * 50 + "\\n"
            if n_significant > n_features * 0.5:
                text += "✓ STRONG SEPARATION: More than half of features differ significantly\\n"
                text += "  between clusters. Your clustering captures real differences.\\n"
            elif n_significant > n_features * 0.2:
                text += "✓ MODERATE SEPARATION: A substantial portion of features differ\\n"
                text += "  significantly. Clusters show meaningful differences.\\n"
            else:
                text += "⚠ WEAK SEPARATION: Few features differ significantly between\\n"
                text += "  clusters. Consider different clustering parameters.\\n"
            
            text += "\\n"
            text += "TOP 10 DISCRIMINATING FEATURES:\\n"
            text += "-" * 50 + "\\n"
            text += f"{'Feature':<10} {'F-statistic':<15} {'p-value':<15} {'Significance'}\\n"
            text += "-" * 50 + "\\n"
            
            # Sort by F-statistic
            top_indices = np.argsort(f_statistics)[::-1][:10]
            for idx in top_indices:
                sig_level = "***" if p_values[idx] < 0.001 else "**" if p_values[idx] < 0.01 else "*" if p_values[idx] < 0.05 else "ns"
                text += f"{idx:<10} {f_statistics[idx]:<15.2f} {p_values[idx]:<15.6f} {sig_level}\\n"
            
            text += "\\n*** p<0.001, ** p<0.01, * p<0.05, ns = not significant\\n"
            
            self.advanced_statistics_tab.stats_results_text.setText(text)
            
            # Create visualization
            fig = self.advanced_statistics_tab.stats_fig
            fig.clear()
            ax = fig.add_subplot(111)
            
            # Plot F-statistics
            x = np.arange(n_features)
            ax.plot(x, f_statistics, 'b-', linewidth=1, label='F-statistic')
            ax.fill_between(x, 0, f_statistics, alpha=0.3)
            
            # Add significance threshold line
            f_crit = stats.f.ppf(0.95, n_clusters-1, len(labels)-n_clusters)
            ax.axhline(y=f_crit, color='r', linestyle='--', linewidth=2, 
                      label=f'Critical F (p=0.05): {f_crit:.2f}')
            
            ax.set_xlabel('Feature Index', fontsize=10)
            ax.set_ylabel('F-statistic', fontsize=10)
            ax.set_title(f'ANOVA F-Statistics Across Features\\n({n_significant} significant features)', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            self.advanced_statistics_tab.stats_canvas.draw()
            
            self.statusBar().showMessage(f"ANOVA test complete: {n_significant} significant features", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error performing ANOVA test:\\n{str(e)}")
            import traceback
            traceback.print_exc()
'''

print("Statistical implementations ready. Copy these into main.py to replace stub methods.")
