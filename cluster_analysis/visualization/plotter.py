"""
Visualization Plotter for RamanLab Cluster Analysis

This module contains plotting and visualization methods for cluster analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, SpectralEmbedding
import seaborn as sns


class ClusterPlotter:
    """Plotting utilities for cluster analysis visualization."""
    
    def __init__(self):
        self.figure = None
        self.canvas = None
        self.current_plot_type = None
    
    def create_figure_canvas(self, parent=None):
        """Create a matplotlib figure and canvas for Qt integration."""
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        if parent:
            parent.addWidget(self.canvas)
        return self.canvas
    
    def plot_cluster_scatter(self, features, labels, method='pca', title=None):
        """Plot cluster visualization using specified dimensionality reduction method."""
        try:
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Apply dimensionality reduction
            if method.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(features)
                explained_var = reducer.explained_variance_ratio_
                subtitle = f"PCA (Explained variance: {explained_var.sum()*100:.1f}%)"
                
            elif method.lower() == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
                coords_2d = reducer.fit_transform(features)
                subtitle = "t-SNE"
                
            elif method.lower() == 'mds':
                reducer = MDS(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(features)
                subtitle = "Multidimensional Scaling"
                
            elif method.lower() == 'spectral':
                reducer = SpectralEmbedding(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(features)
                subtitle = "Spectral Embedding"
                
            else:
                raise ValueError(f"Unknown visualization method: {method}")
            
            # Get unique labels and colors
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            # Plot each cluster
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                          c=[colors[i]], label=f'Cluster {label}', 
                          alpha=0.7, s=50)
            
            # Set labels and title
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'Cluster Visualization - {subtitle}')
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Update canvas
            self.figure.tight_layout()
            if self.canvas:
                self.canvas.draw()
            
            self.current_plot_type = f'scatter_{method}'
            
        except Exception as e:
            print(f"Error in cluster scatter plot: {str(e)}")
            self._plot_error_message(f"Error in {method} plot: {str(e)}")
    
    def plot_dendrogram(self, linkage_matrix, labels=None, title=None):
        """Plot hierarchical clustering dendrogram."""
        try:
            from scipy.cluster.hierarchy import dendrogram
            
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot dendrogram
            dendrogram(linkage_matrix, 
                      labels=labels if labels is not None else None,
                      ax=ax,
                      leaf_rotation=90,
                      leaf_font_size=8)
            
            ax.set_title(title or 'Hierarchical Clustering Dendrogram')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Distance')
            
            # Update canvas
            self.figure.tight_layout()
            if self.canvas:
                self.canvas.draw()
            
            self.current_plot_type = 'dendrogram'
            
        except Exception as e:
            print(f"Error in dendrogram plot: {str(e)}")
            self._plot_error_message(f"Error in dendrogram plot: {str(e)}")
    
    def plot_heatmap(self, data, labels=None, title=None):
        """Plot heatmap of data with optional cluster labels."""
        try:
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Sort by labels if provided
            if labels is not None:
                sort_idx = np.argsort(labels)
                sorted_data = data[sort_idx]
            else:
                sorted_data = data
            
            # Create heatmap
            sns.heatmap(sorted_data, ax=ax, cmap='viridis', 
                       cbar_kws={'label': 'Intensity'})
            
            ax.set_title(title or 'Data Heatmap')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Sample Index')
            
            # Update canvas
            self.figure.tight_layout()
            if self.canvas:
                self.canvas.draw()
            
            self.current_plot_type = 'heatmap'
            
        except Exception as e:
            print(f"Error in heatmap plot: {str(e)}")
            self._plot_error_message(f"Error in heatmap plot: {str(e)}")
    
    def plot_cluster_centroids(self, cluster_data, title=None):
        """Plot cluster centroids/average spectra."""
        try:
            if 'labels' not in cluster_data or 'intensities' not in cluster_data:
                raise ValueError("Missing required data for centroid plot")
            
            labels = cluster_data['labels']
            intensities = cluster_data['intensities']
            wavenumbers = cluster_data.get('wavenumbers', np.arange(intensities.shape[1]))
            
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Calculate centroids for each cluster
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                centroid = np.mean(intensities[mask], axis=0)
                
                ax.plot(wavenumbers, centroid, 
                       color=colors[i], label=f'Cluster {label}', 
                       linewidth=2)
            
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity (normalized)')
            ax.set_title(title or 'Cluster Centroids')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Update canvas
            self.figure.tight_layout()
            if self.canvas:
                self.canvas.draw()
            
            self.current_plot_type = 'centroids'
            
        except Exception as e:
            print(f"Error in centroid plot: {str(e)}")
            self._plot_error_message(f"Error in centroid plot: {str(e)}")
    
    def plot_silhouette_analysis(self, features, labels, title=None):
        """Plot silhouette analysis for clustering evaluation."""
        try:
            from sklearn.metrics import silhouette_samples, silhouette_score
            
            # Calculate silhouette scores
            silhouette_avg = silhouette_score(features, labels)
            sample_silhouette_values = silhouette_samples(features, labels)
            
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot silhouette for each cluster
            unique_labels = np.unique(labels)
            y_lower = 10
            
            for i, label in enumerate(unique_labels):
                # Get silhouette values for current cluster
                ith_cluster_silhouette_values = sample_silhouette_values[labels == label]
                ith_cluster_silhouette_values.sort()
                
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = plt.cm.Set3(i / len(unique_labels))
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                               0, ith_cluster_silhouette_values,
                               facecolor=color, edgecolor=color, alpha=0.7)
                
                # Label the silhouette plots with their cluster numbers
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
                
                y_lower = y_upper + 10  # 10 for spacing between clusters
            
            ax.set_xlabel('Silhouette coefficient values')
            ax.set_ylabel('Cluster label')
            
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'Silhouette Analysis (Average score: {silhouette_avg:.3f})')
            
            # Vertical line for average silhouette score
            ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                      label=f'Average: {silhouette_avg:.3f}')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Update canvas
            self.figure.tight_layout()
            if self.canvas:
                self.canvas.draw()
            
            self.current_plot_type = 'silhouette'
            
        except Exception as e:
            print(f"Error in silhouette plot: {str(e)}")
            self._plot_error_message(f"Error in silhouette plot: {str(e)}")
    
    def save_figure(self, filename, dpi=300, format='png'):
        """Save the current figure to file."""
        try:
            if self.figure is None:
                raise ValueError("No figure to save")
            
            self.figure.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"Error saving figure: {str(e)}")
            return False
    
    def clear_plot(self):
        """Clear the current plot."""
        if self.figure:
            self.figure.clear()
            if self.canvas:
                self.canvas.draw()
            self.current_plot_type = None
    
    def _plot_error_message(self, message):
        """Plot an error message when plotting fails."""
        if self.figure:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, message, transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            if self.canvas:
                self.canvas.draw()
