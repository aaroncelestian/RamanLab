def update_scatter_plot_improved(self):
    """
    Enhanced UMAP implementation specifically optimized for carbon soot clustering.
    Features improved parameter optimization and feature selection for tighter clusters.
    """
    if (self.cluster_data['features_scaled'] is None or 
        self.cluster_data['labels'] is None):
        return
    
    try:
        self.viz_ax.clear()
        
        # Clear previous scatter points data and hover annotation
        self.scatter_points = []
        self.hover_annotation = None
        
        # Get visualization method
        method = self.visualization_method_combo.currentText()
        
        # Use carbon-specific features instead of generic features
        if method == 'UMAP' and UMAP_AVAILABLE:
            # Extract carbon-specific features first
            print("Extracting carbon-specific features...")
            carbon_features = self.extract_carbon_specific_features(
                self.cluster_data['intensities'], 
                self.cluster_data['wavenumbers']
            )
            
            # Apply feature selection for better discrimination
            carbon_features = self.select_discriminatory_carbon_features(carbon_features)
            
            # Scale the carbon-specific features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(carbon_features)
        else:
            features_scaled = self.cluster_data['features_scaled']
        
        labels = self.cluster_data['labels']
        
        # Get hover labels
        hover_labels = self.get_hover_labels()
        
        if method == 'PCA':
            # Standard PCA implementation
            pca = PCA(n_components=2)
            coords = pca.fit_transform(features_scaled)
            
            self.cluster_data['pca_coords'] = coords
            self.cluster_data['pca_model'] = pca
            
            xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)'
            ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
            title = 'PCA Visualization of Clusters'
            
        elif method == 'UMAP' and UMAP_AVAILABLE:
            # Optimized UMAP parameters for carbon soot discrimination with tighter clusters
            n_samples = len(features_scaled)
            
            # More aggressive parameters for tighter clustering
            n_neighbors = max(8, min(25, n_samples // 4))  # Smaller neighborhood for tighter clusters
            min_dist = 0.001  # Even smaller for very tight clusters
            metric = 'cosine'  # Best for spectral data
            spread = 0.3  # Tighter spread for better separation
            
            # Override with UI values if available
            if hasattr(self, 'umap_n_neighbors'):
                n_neighbors = min(self.umap_n_neighbors.value(), n_samples - 1)
            if hasattr(self, 'umap_min_dist'):
                min_dist = self.umap_min_dist.value()
            if hasattr(self, 'umap_metric'):
                metric = self.umap_metric.currentText()
            if hasattr(self, 'umap_spread'):
                spread = self.umap_spread.value()
            
            try:
                print(f"Running optimized UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
                
                # Create UMAP with highly optimized parameters for carbon discrimination
                umap_model = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    spread=spread,
                    random_state=None,  # Remove to enable parallel processing
                    n_jobs=-1,  # Use all available CPU cores
                    # Enhanced parameters for tighter carbon clustering
                    local_connectivity=3.0,   # Higher local connectivity for tighter groups
                    repulsion_strength=3.0,   # Stronger repulsion for better separation
                    negative_sample_rate=15,  # More negative samples for cleaner structure
                    transform_queue_size=16.0, # Larger queue for stability
                    a=None, b=None,           # Let UMAP optimize these
                    init='spectral',          # Better initialization for spectral data
                    densmap=False,            # Focus on topology, not density
                    dens_lambda=3.0,          # Higher for cleaner topology
                    dens_frac=0.2,            # Lower fraction for tighter groups
                    dens_var_shift=0.05,      # Smaller shift for stability
                    output_dens=False,
                    verbose=True,
                    # Additional optimization for carbon materials
                    learning_rate=2.0,        # Higher learning rate for faster convergence
                    n_epochs=500,             # More epochs for better optimization
                    min_dist_scale=1.2,       # Scale min_dist for better embedding
                    set_op_mix_ratio=0.8      # Higher ratio for better global structure
                )
                
                coords = umap_model.fit_transform(features_scaled)
                
                # Store UMAP results
                self.cluster_data['umap_coords'] = coords
                self.cluster_data['umap_model'] = umap_model
                self.cluster_data['carbon_features'] = carbon_features
                self.cluster_data['carbon_features_scaled'] = features_scaled
                
                xlabel = f'UMAP 1 (carbon-optimized: {metric} metric)'
                ylabel = f'UMAP 2 (neighbors={n_neighbors}, min_dist={min_dist:.4f}, tighter clusters)'
                title = 'UMAP Visualization - Enhanced Carbon Discrimination'
                
                print("Enhanced UMAP completed successfully")
                
            except Exception as e:
                print(f"Enhanced UMAP failed: {e}")
                print("Falling back to conservative UMAP settings...")
                
                try:
                    # Fallback UMAP with conservative but still optimized settings
                    fallback_umap = umap.UMAP(
                        n_components=2,
                        n_neighbors=min(15, n_samples - 1),
                        min_dist=0.01,
                        metric='cosine',
                        spread=0.5,
                        local_connectivity=2.0,
                        repulsion_strength=2.0,
                        random_state=None,  # Remove to enable parallel processing
                        n_jobs=-1,  # Use all available CPU cores
                        n_epochs=300
                    )
                    coords = fallback_umap.fit_transform(features_scaled)
                    
                    xlabel = 'UMAP 1 (fallback - optimized)'
                    ylabel = 'UMAP 2 (fallback - optimized)'
                    title = 'UMAP Visualization (Conservative Optimization)'
                    
                except Exception as e2:
                    print(f"Fallback UMAP also failed: {e2}")
                    # Final fallback to PCA
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(features_scaled)
                    xlabel = f'PC1 (UMAP failed - {pca.explained_variance_ratio_[0]:.1%})'
                    ylabel = f'PC2 (UMAP failed - {pca.explained_variance_ratio_[1]:.1%})'
                    title = 'PCA Visualization (UMAP Failed)'
        
        else:
            # Fallback to PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(features_scaled)
            xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
            ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
            title = 'PCA Visualization'
        
        # Create enhanced scatter plot with better styling for clearer separation
        unique_labels = np.unique(labels)
        
        # Get colormap
        colormap_name = self.colormap_combo.currentText() if hasattr(self, 'colormap_combo') else 'Set1'
        reverse_colormap = self.reverse_colormap_cb.isChecked() if hasattr(self, 'reverse_colormap_cb') else False
        
        try:
            colormap = plt.cm.get_cmap(colormap_name)
            if reverse_colormap:
                colormap = colormap.reversed()
            colors = colormap(np.linspace(0, 1, len(unique_labels)))
        except Exception as e:
            print(f"Error loading colormap '{colormap_name}': {e}")
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        # Plot clusters with enhanced visualization for better separation visibility
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_coords = coords[mask]
            cluster_hover_labels = [hover_labels[j] for j in range(len(hover_labels)) if mask[j]]
            
            # Create scatter plot for this cluster with enhanced styling
            scatter = self.viz_ax.scatter(
                cluster_coords[:, 0], 
                cluster_coords[:, 1], 
                c=[colors[i]], 
                label=f'Cluster {label} (n={len(cluster_coords)})',
                alpha=0.85,
                s=80,  # Larger points for better visibility
                edgecolors='black',
                linewidths=1.5  # Thicker edge for clearer boundaries
            )
            
            # Store scatter point information for hover detection
            self.scatter_points.append({
                'scatter': scatter,
                'coords': cluster_coords,
                'labels': cluster_hover_labels,
                'cluster': label
            })
            
            # Add cluster centroid with enhanced visibility
            centroid = np.mean(cluster_coords, axis=0)
            self.viz_ax.scatter(centroid[0], centroid[1], 
                              c='black', marker='x', s=300, linewidths=4,
                              zorder=10)  # Ensure centroids are on top
            
            # Add cluster boundary estimation (convex hull)
            if len(cluster_coords) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(cluster_coords)
                    hull_points = cluster_coords[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
                    self.viz_ax.plot(hull_points[:, 0], hull_points[:, 1], 
                                   color=colors[i], alpha=0.3, linewidth=2, linestyle='--')
                except:
                    pass  # Skip hull if computation fails
        
        self.viz_ax.set_xlabel(xlabel)
        self.viz_ax.set_ylabel(ylabel)
        self.viz_ax.set_title(title)
        self.viz_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.viz_ax.grid(True, alpha=0.3)
        
        # Add enhanced text box with optimization info
        if method == 'UMAP':
            info_text = f"Enhanced Carbon Clustering\nOptimized for: Charcoal/Diesel/Car Exhaust\nFeatures: Selective D/G features\nTighter clustering parameters applied"
            self.viz_ax.text(0.02, 0.98, info_text, transform=self.viz_ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round',
                           facecolor='lightgreen', alpha=0.8), fontsize=9)
        
        self.viz_fig.tight_layout()
        self.viz_canvas.draw()
        
        # Enable export buttons
        if hasattr(self, 'export_folders_btn') and unique_labels is not None:
            self.export_folders_btn.setEnabled(True)
            self.export_summed_btn.setEnabled(True)
            self.export_overview_btn.setEnabled(True)
            self.export_status.setText(f"Ready to export {len(unique_labels)} enhanced clusters")
        
        # Print enhanced feature analysis for debugging
        if method == 'UMAP' and 'carbon_features' in self.cluster_data:
            self.print_enhanced_carbon_analysis()
        
    except Exception as e:
        print(f"Error updating scatter plot: {str(e)}")
        import traceback
        traceback.print_exc()

def select_discriminatory_carbon_features(self, carbon_features):
    """
    Select the most discriminatory features for charcoal/diesel/car exhaust classification.
    """
    try:
        if carbon_features.shape[1] < 17:
            return carbon_features  # Return as-is if not full feature set
        
        # Feature indices based on extract_carbon_specific_features
        feature_names = [
            'D_peak_position',     # 0 - Important for disorder level
            'D_intensity',         # 1 - Key discriminator
            'D_width',             # 2 - Crystallite size indicator
            'D_integrated',        # 3 - Total D-band contribution
            'G_peak_position',     # 4 - Graphitic quality
            'G_intensity',         # 5 - Key discriminator
            'G_width',             # 6 - Graphitic order
            'G_integrated',        # 7 - Total G-band contribution
            'ID_IG_ratio',         # 8 - Most important discriminator
            'D_prime_intensity',   # 9 - Defect indicator
            'D_prime_G_ratio',     # 10 - Defect/graphitic ratio
            'Low_freq_integrated', # 11 - Structural differences
            'RBM_intensity',       # 12 - Nanotube indicator
            '2D_intensity',        # 13 - Graphitic layers
            '2D_G_ratio',          # 14 - Layer quality
            'G_asymmetry',         # 15 - Crystallinity
            'Background_slope'     # 16 - Amorphous content
        ]
        
        # Select most discriminatory features for carbon soot types
        # Based on literature and empirical carbon analysis
        selected_indices = [
            0,   # D_peak_position - shifts with disorder
            1,   # D_intensity - varies significantly between types
            2,   # D_width - crystallite size differences
            4,   # G_peak_position - graphitic quality varies
            5,   # G_intensity - reference for normalization
            6,   # G_width - order parameter
            8,   # ID_IG_ratio - THE key discriminator for carbon
            10,  # D_prime_G_ratio - defect characterization
            11,  # Low_freq_integrated - structural fingerprint
            15,  # G_asymmetry - crystallinity measure
            16   # Background_slope - amorphous content
        ]
        
        # Extract selected features
        selected_features = carbon_features[:, selected_indices]
        
        print(f"Selected {len(selected_indices)} discriminatory features from {carbon_features.shape[1]} total features")
        print(f"Selected features: {[feature_names[i] for i in selected_indices]}")
        
        return selected_features
        
    except Exception as e:
        print(f"Error in feature selection: {e}")
        return carbon_features  # Return original if selection fails

def print_enhanced_carbon_analysis(self):
    """Enhanced analysis specifically for charcoal/diesel/car exhaust discrimination."""
    try:
        if 'carbon_features' not in self.cluster_data:
            return
        
        features = self.cluster_data['carbon_features']
        labels = self.cluster_data['labels']
        unique_labels = np.unique(labels)
        
        print("\n=== Enhanced Carbon Discrimination Analysis ===")
        print("Optimized for: Charcoal vs Diesel vs Car Exhaust")
        
        # Focus on key discriminatory features
        if features.shape[1] >= 11:  # Check if we have selected features
            key_features = {
                'ID_IG_ratio': 8,      # Most important
                'D_peak_position': 0,   # Disorder indicator  
                'G_peak_position': 4,   # Graphitic quality
                'D_width': 2,          # Crystallite size
                'G_asymmetry': 10,      # Crystallinity (if selected features)
                'Background_slope': 10  # Amorphous content (if selected features)
            }
        else:
            # Standard feature indices
            key_features = {
                'ID_IG_ratio': 8,
                'D_peak_position': 0,
                'G_peak_position': 4,
                'D_width': 2
            }
        
        for feature_name, idx in key_features.items():
            if idx < features.shape[1]:
                print(f"\n{feature_name}:")
                cluster_means = []
                cluster_stds = []
                
                for label in unique_labels:
                    mask = labels == label
                    feature_values = features[mask, idx]
                    mean_val = np.mean(feature_values)
                    std_val = np.std(feature_values)
                    cluster_means.append(mean_val)
                    cluster_stds.append(std_val)
                    print(f"  Cluster {label}: {mean_val:.4f} ± {std_val:.4f}")
                
                # Calculate cluster separation
                if len(cluster_means) > 1:
                    separation = (max(cluster_means) - min(cluster_means)) / np.mean(cluster_stds)
                    print(f"  Separation ratio: {separation:.2f} {'(Good)' if separation > 2 else '(Poor)'}")
        
        # Specific interpretations for carbon types
        if 8 < features.shape[1]:  # ID/IG ratio available
            print(f"\n=== Carbon Type Predictions (Based on ID/IG Ratios) ===")
            for label in unique_labels:
                mask = labels == label
                id_ig_values = features[mask, 8] if features.shape[1] > 8 else [0]
                mean_ratio = np.mean(id_ig_values)
                
                # Carbon type interpretation
                if mean_ratio > 2.0:
                    carbon_type = "Highly disordered soot (likely diesel exhaust)"
                elif mean_ratio > 1.2:
                    carbon_type = "Moderately disordered carbon (car exhaust/fresh soot)"
                elif mean_ratio > 0.7:
                    carbon_type = "Partially graphitized carbon (aged soot/low-temp charcoal)"
                else:
                    carbon_type = "Well-ordered carbon (high-temp charcoal/graphitic)"
                
                print(f"Cluster {label} (ID/IG = {mean_ratio:.3f}): {carbon_type}")
        
        # Calculate overall clustering quality
        self.calculate_clustering_quality_metrics()
        
    except Exception as e:
        print(f"Error in enhanced carbon analysis: {e}")

def calculate_clustering_quality_metrics(self):
    """Calculate metrics to assess clustering quality."""
    try:
        if 'carbon_features_scaled' not in self.cluster_data or 'labels' not in self.cluster_data:
            return
        
        features = self.cluster_data['carbon_features_scaled']
        labels = self.cluster_data['labels']
        
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        # Calculate clustering quality metrics
        silhouette_avg = silhouette_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        
        print(f"\n=== Clustering Quality Metrics ===")
        print(f"Silhouette Score: {silhouette_avg:.3f} (higher is better, >0.5 is good)")
        print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f} (higher is better)")
        print(f"Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better, <1.0 is good)")
        
        # Overall assessment
        quality_score = 0
        if silhouette_avg > 0.5: quality_score += 1
        if davies_bouldin < 1.0: quality_score += 1
        if calinski_harabasz > 100: quality_score += 1
        
        if quality_score >= 2:
            assessment = "GOOD - Clusters are well-separated"
        elif quality_score == 1:
            assessment = "MODERATE - Some improvement possible"
        else:
            assessment = "POOR - Consider parameter adjustment"
        
        print(f"Overall Assessment: {assessment}")
        
        return {
            'silhouette': silhouette_avg,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin,
            'quality_score': quality_score
        }
        
    except Exception as e:
        print(f"Error calculating clustering metrics: {e}")
        return None

def suggest_clustering_improvements(self):
    """Enhanced clustering improvement suggestions with specific parameter recommendations."""
    if (self.cluster_data['labels'] is None or 
        'carbon_features' not in self.cluster_data):
        print("No clustering data available. Run clustering with Carbon Soot Optimization first.")
        return
    
    try:
        features = self.cluster_data['carbon_features']
        labels = self.cluster_data['labels']
        unique_labels = np.unique(labels)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE CLUSTERING IMPROVEMENT ANALYSIS")
        print("="*60)
        print(f"Current clusters: {len(unique_labels)}")
        print(f"Sample types expected: Charcoal, Diesel Exhaust, Car Exhaust")
        print(f"Total samples: {len(labels)}")
        
        # 1. Calculate current clustering quality
        quality_metrics = self.calculate_clustering_quality_metrics()
        
        # 2. Analyze cluster separation for key features
        print("\n" + "-"*40)
        print("FEATURE DISCRIMINATION ANALYSIS")
        print("-"*40)
        
        separation_scores = {}
        feature_names = ['D_peak_position', 'D_intensity', 'D_width', 'G_peak_position', 
                        'G_intensity', 'G_width', 'ID_IG_ratio', 'Background_slope']
        
        for i, feature_name in enumerate(feature_names):
            if i < features.shape[1]:
                cluster_means = []
                cluster_stds = []
                
                for label in unique_labels:
                    mask = labels == label
                    feature_values = features[mask, i]
                    cluster_means.append(np.mean(feature_values))
                    cluster_stds.append(np.std(feature_values))
                
                if len(cluster_means) > 1:
                    # Calculate Fisher's discriminant ratio
                    between_var = np.var(cluster_means)
                    within_var = np.mean([std**2 for std in cluster_stds])
                    if within_var > 0:
                        separation_score = between_var / within_var
                    else:
                        separation_score = 0
                    
                    separation_scores[feature_name] = separation_score
                    status = "EXCELLENT" if separation_score > 5 else "GOOD" if separation_score > 2 else "POOR"
                    print(f"{feature_name}: {separation_score:.3f} ({status})")
        
        # 3. Specific recommendations based on analysis
        print("\n" + "-"*40)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("-"*40)
        
        recommendations = []
        
        # Check cluster count vs expected
        if len(unique_labels) == 4 and len(unique_labels) > 3:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Extra cluster detected',
                'recommendation': 'Try n_clusters=3 in hierarchical clustering, or increase UMAP min_dist to 0.01-0.05',
                'action': 'Reduce cluster sensitivity'
            })
        elif len(unique_labels) < 3:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Too few clusters',
                'recommendation': 'Decrease UMAP min_dist to 0.0001, increase n_neighbors to 15-20',
                'action': 'Increase cluster sensitivity'
            })
        
        # Check feature separation quality
        poor_features = [name for name, score in separation_scores.items() if score < 2]
        if len(poor_features) > 3:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': f'Poor feature separation in {len(poor_features)} features',
                'recommendation': 'Try different preprocessing: enable enhanced baseline correction and fluorescence removal',
                'action': 'Improve data quality'
            })
        
        # Check silhouette score
        if quality_metrics and quality_metrics['silhouette'] < 0.3:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Poor cluster separation (silhouette < 0.3)',
                'recommendation': 'Increase UMAP repulsion_strength to 4.0, decrease spread to 0.2',
                'action': 'Enhance cluster separation'
            })
        elif quality_metrics and quality_metrics['silhouette'] < 0.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': 'Moderate cluster separation',
                'recommendation': 'Fine-tune UMAP: try n_neighbors=10-15, min_dist=0.001-0.01',
                'action': 'Optimize parameters'
            })
        
        # ID/IG ratio specific recommendations
        if 8 < features.shape[1]:  # ID/IG ratio available
            id_ig_separation = separation_scores.get('ID_IG_ratio', 0)
            if id_ig_separation < 3:
                recommendations.append({
                    'priority': 'HIGH',
                    'issue': 'Poor ID/IG ratio discrimination',
                    'recommendation': 'Check D and G band fitting quality. Consider manual peak fitting for better accuracy.',
                    'action': 'Improve peak detection'
                })
        
        # Check for potential outliers
        from scipy.stats import zscore
        z_scores = np.abs(zscore(features, axis=0))
        outlier_samples = np.any(z_scores > 3.0, axis=1)
        n_outliers = np.sum(outlier_samples)
        
        if n_outliers > len(features) * 0.1:  # More than 10% outliers
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': f'{n_outliers} potential outlier samples detected',
                'recommendation': 'Review data quality, check for measurement errors or contamination',
                'action': 'Data quality check'
            })
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']} PRIORITY] {rec['issue']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Action: {rec['action']}")
        
        # 4. Parameter optimization suggestions
        print("\n" + "-"*40)
        print("PARAMETER OPTIMIZATION GUIDE")
        print("-"*40)
        
        print("\nFor TIGHTER clusters (less scatter):")
        print("  - Decrease min_dist: 0.0001 - 0.001")
        print("  - Increase repulsion_strength: 3.0 - 5.0")  
        print("  - Decrease spread: 0.1 - 0.3")
        print("  - Increase local_connectivity: 3.0 - 5.0")
        
        print("\nFor BETTER separation between clusters:")
        print("  - Increase n_neighbors: 15 - 25")
        print("  - Use cosine metric (best for spectral data)")
        print("  - Increase negative_sample_rate: 15 - 20")
        print("  - Try n_epochs: 500 - 1000")
        
        print("\nFor SPECIFIC carbon type discrimination:")
        print("  - Focus on ID/IG ratio (most important)")
        print("  - D-band position distinguishes disorder levels")
        print("  - G-band width indicates crystallinity")
        print("  - Background slope shows amorphous content")
        
        # 5. Expected clustering patterns
        print("\n" + "-"*40)
        print("EXPECTED CLUSTERING PATTERNS")
        print("-"*40)
        
        print("\nCharcoal (high-temperature):")
        print("  - Low ID/IG ratio (0.5-1.0)")
        print("  - G-band ~1580 cm⁻¹")
        print("  - Narrow D and G bands")
        print("  - Low background slope")
        
        print("\nDiesel exhaust:")
        print("  - High ID/IG ratio (1.5-3.0)")
        print("  - D-band ~1350 cm⁻¹")
        print("  - Broad D band")
        print("  - High background slope")
        
        print("\nCar exhaust (gasoline):")
        print("  - Medium ID/IG ratio (1.0-2.0)")
        print("  - D-band ~1340-1350 cm⁻¹")
        print("  - Intermediate band widths")
        print("  - Medium background slope")
        
        # 6. Action plan
        print("\n" + "-"*40)
        print("IMMEDIATE ACTION PLAN")
        print("-"*40)
        
        print("\n1. QUICK FIXES (try first):")
        print("   - Set UMAP min_dist = 0.001")
        print("   - Set UMAP n_neighbors = 15")
        print("   - Set repulsion_strength = 3.5")
        print("   - Set spread = 0.2")
        print("   - Enable carbon-specific preprocessing")
        
        print("\n2. IF STILL SCATTERED:")
        print("   - Try min_dist = 0.0001 (very tight)")
        print("   - Increase repulsion_strength to 5.0")
        print("   - Use feature selection (focus on ID/IG, D/G positions)")
        
        print("\n3. IF CLUSTERS OVERLAP:")
        print("   - Check data quality and preprocessing")
        print("   - Consider that some samples may be intermediate types")
        print("   - Try different clustering algorithm (GMM, spectral clustering)")
        
        print("\n4. DATA QUALITY CHECKS:")
        print("   - Verify D and G band peaks are properly identified")
        print("   - Check for cosmic rays or measurement artifacts")
        print("   - Ensure consistent measurement conditions")
        
        # 7. Return quantitative assessment
        if quality_metrics:
            overall_quality = quality_metrics['quality_score']
            if overall_quality >= 2:
                assessment = "GOOD - Minor tweaks may help"
            elif overall_quality == 1:
                assessment = "MODERATE - Parameter optimization recommended"
            else:
                assessment = "POOR - Major improvements needed"
            
            print(f"\nOVERALL ASSESSMENT: {assessment}")
            print(f"Quality Score: {overall_quality}/3")
        
        return {
            'recommendations': recommendations,
            'separation_scores': separation_scores,
            'quality_metrics': quality_metrics,
            'n_outliers': n_outliers if 'n_outliers' in locals() else 0
        }
        
    except Exception as e:
        print(f"Error in clustering improvement analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def optimize_clustering_parameters_automatically(self):
    """Automatically optimize UMAP parameters based on current data characteristics."""
    try:
        if 'carbon_features_scaled' not in self.cluster_data:
            print("No scaled carbon features available for optimization")
            return None
        
        features = self.cluster_data['carbon_features_scaled']
        n_samples = len(features)
        
        print("\n" + "="*50)
        print("AUTOMATIC PARAMETER OPTIMIZATION")
        print("="*50)
        
        # Test different parameter combinations
        param_combinations = [
            # (n_neighbors, min_dist, repulsion_strength, spread)
            (10, 0.001, 3.5, 0.2),  # Tight clusters
            (15, 0.001, 3.0, 0.3),  # Balanced
            (12, 0.0005, 4.0, 0.15), # Very tight
            (20, 0.01, 2.5, 0.4),   # More relaxed
            (8, 0.0001, 5.0, 0.1),  # Extremely tight
        ]
        
        best_score = -1
        best_params = None
        best_coords = None
        results = []
        
        from sklearn.metrics import silhouette_score
        
        for i, (n_neighbors, min_dist, repulsion, spread) in enumerate(param_combinations):
            try:
                print(f"\nTesting combination {i+1}/5: n_neighbors={n_neighbors}, min_dist={min_dist}")
                
                # Adjust n_neighbors for small datasets
                n_neighbors_adj = min(n_neighbors, n_samples - 1)
                
                # Create UMAP model
                umap_model = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors_adj,
                    min_dist=min_dist,
                    metric='cosine',
                    spread=spread,
                    random_state=None,  # Remove to enable parallel processing
                    n_jobs=-1,  # Use all available CPU cores
                    local_connectivity=3.0,
                    repulsion_strength=repulsion,
                    negative_sample_rate=15,
                    n_epochs=300,
                    verbose=False
                )
                
                # Fit and transform
                coords = umap_model.fit_transform(features)
                
                # Perform clustering on the UMAP coordinates
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                temp_labels = kmeans.fit_predict(coords)
                
                # Calculate quality metrics
                silhouette = silhouette_score(coords, temp_labels)
                
                # Calculate cluster compactness (lower is better for tight clusters)
                compactness = 0
                for label in np.unique(temp_labels):
                    cluster_coords = coords[temp_labels == label]
                    if len(cluster_coords) > 1:
                        centroid = np.mean(cluster_coords, axis=0)
                        distances = np.sqrt(np.sum((cluster_coords - centroid)**2, axis=1))
                        compactness += np.mean(distances)
                compactness /= len(np.unique(temp_labels))
                
                # Combined score (higher silhouette, lower compactness)
                combined_score = silhouette - compactness * 0.1
                
                results.append({
                    'params': (n_neighbors_adj, min_dist, repulsion, spread),
                    'silhouette': silhouette,
                    'compactness': compactness,
                    'combined_score': combined_score,
                    'coords': coords
                })
                
                print(f"  Silhouette: {silhouette:.3f}, Compactness: {compactness:.3f}, Score: {combined_score:.3f}")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_params = (n_neighbors_adj, min_dist, repulsion, spread)
                    best_coords = coords
                    
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        # Print optimization results
        print(f"\n" + "-"*40)
        print("OPTIMIZATION RESULTS")
        print("-"*40)
        
        if best_params:
            n_neighbors, min_dist, repulsion, spread = best_params
            print(f"\nBEST PARAMETERS (Score: {best_score:.3f}):")
            print(f"  n_neighbors: {n_neighbors}")
            print(f"  min_dist: {min_dist}")
            print(f"  repulsion_strength: {repulsion}")
            print(f"  spread: {spread}")
            
            # Apply best parameters to UI if available
            if hasattr(self, 'umap_n_neighbors'):
                self.umap_n_neighbors.setValue(n_neighbors)
            if hasattr(self, 'umap_min_dist'):
                self.umap_min_dist.setValue(min_dist)
            if hasattr(self, 'umap_spread'):
                self.umap_spread.setValue(spread)
            
            print(f"\n✓ Best parameters applied to UI controls")
            print(f"✓ Click 'Update UMAP' to see optimized clustering")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'best_coords': best_coords,
                'all_results': results
            }
        else:
            print("\nNo successful parameter combinations found")
            return None
            
    except Exception as e:
        print(f"Error in automatic parameter optimization: {e}")
        return None