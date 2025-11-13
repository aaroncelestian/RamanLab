"""
Clustering Engine for RamanLab Cluster Analysis

This module contains the core clustering algorithms and methods.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import multiprocessing
import sklearn


class ClusteringEngine:
    """Core clustering engine for Raman spectroscopy data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.last_algorithm_used = None
    
    def perform_hierarchical_clustering(self, features, n_clusters, linkage_method, distance_metric):
        """Perform clustering on the features with automatic algorithm selection for performance.
        
        For large datasets (>5000 spectra), uses MiniBatchKMeans for speed.
        For medium datasets (1000-5000 spectra), uses standard KMeans.
        For small datasets (<1000 spectra), uses hierarchical clustering.
        """
        try:
            n_samples = features.shape[0]
            
            # Get number of available CPU cores
            n_jobs = multiprocessing.cpu_count()
            
            # Check sklearn version for n_jobs support with better parsing
            try:
                version_parts = sklearn.__version__.split('.')
                sklearn_version = tuple(int(x.split('rc')[0].split('dev')[0]) for x in version_parts[:2])
            except:
                # Fallback: assume old version if parsing fails
                sklearn_version = (0, 19)
            
            print(f"\n🔍 Debug: scikit-learn version {sklearn.__version__} -> {sklearn_version}")
            
            supports_minibatch_njobs = sklearn_version >= (1, 0)  # MiniBatchKMeans got n_jobs in 1.0+
            supports_kmeans_njobs = sklearn_version >= (0, 20)  # KMeans has had n_jobs since 0.20
            
            print(f"   MiniBatchKMeans n_jobs support: {supports_minibatch_njobs}")
            print(f"   KMeans n_jobs support: {supports_kmeans_njobs}")
            
            # Automatic algorithm selection based on dataset size
            if n_samples > 5000:
                # Very large dataset - use MiniBatchKMeans for speed
                print(f"\n⚡ Large dataset detected ({n_samples:,} spectra)")
                if supports_minibatch_njobs:
                    print(f"   Using MiniBatchKMeans with {n_jobs} CPU cores for optimal performance...")
                else:
                    print(f"   Using MiniBatchKMeans (single-core - upgrade to sklearn 1.0+ for multi-core)")
                
                # MiniBatchKMeans with progress tracking and parallel processing (if supported)
                batch_size = min(1000, n_samples // 10)
                kmeans_params = {
                    'n_clusters': n_clusters,
                    'batch_size': batch_size,
                    'max_iter': 100,
                    'random_state': 42,
                    'verbose': 0,
                    'n_init': 3
                }
                
                # Try to add n_jobs if supported (check actual parameter support)
                if supports_minibatch_njobs:
                    try:
                        import inspect
                        sig = inspect.signature(MiniBatchKMeans.__init__)
                        if 'n_jobs' in sig.parameters:
                            kmeans_params['n_jobs'] = -1
                            print(f"   ✓ n_jobs parameter confirmed available")
                    except:
                        print(f"   ⚠ Could not verify n_jobs parameter, using single-core")
                
                kmeans = MiniBatchKMeans(**kmeans_params)
                labels = kmeans.fit_predict(features)
                
                # Create synthetic linkage matrix for compatibility
                linkage_matrix = self._create_synthetic_linkage(features, labels, n_clusters)
                distance_matrix = pdist(features, metric='euclidean')
                algorithm_used = "MiniBatchKMeans"
                self.last_algorithm_used = algorithm_used
                
            elif n_samples > 1000:
                # Medium dataset - use standard KMeans
                print(f"\n⚡ Medium dataset detected ({n_samples:,} spectra)")
                if supports_kmeans_njobs:
                    print(f"   Using KMeans with {n_jobs} CPU cores for optimal performance...")
                else:
                    print(f"   Using KMeans (single-core - upgrade to sklearn 0.20+ for multi-core)")
                
                kmeans_params = {
                    'n_clusters': n_clusters,
                    'max_iter': 300,
                    'random_state': 42,
                    'verbose': 0,
                    'n_init': 10
                }
                
                # Try to add n_jobs if supported
                if supports_kmeans_njobs:
                    try:
                        import inspect
                        sig = inspect.signature(KMeans.__init__)
                        if 'n_jobs' in sig.parameters:
                            kmeans_params['n_jobs'] = -1
                            print(f"   ✓ n_jobs parameter confirmed available")
                    except:
                        print(f"   ⚠ Could not verify n_jobs parameter, using single-core")
                
                kmeans = KMeans(**kmeans_params)
                labels = kmeans.fit_predict(features)
                
                # Create synthetic linkage matrix for compatibility
                linkage_matrix = self._create_synthetic_linkage(features, labels, n_clusters)
                distance_matrix = pdist(features, metric='euclidean')
                algorithm_used = "KMeans"
                self.last_algorithm_used = algorithm_used
                
            else:
                # Small dataset - use hierarchical clustering
                print(f"\n⚡ Small dataset detected ({n_samples:,} spectra)")
                print(f"   Using hierarchical clustering with {linkage_method} linkage...")
                
                # Calculate distance matrix
                distance_matrix = pdist(features, metric=distance_metric.lower())
                
                # Perform hierarchical clustering
                linkage_matrix = linkage(distance_matrix, method=linkage_method.lower())
                
                # Cut dendrogram to get cluster labels
                labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust') - 1
                algorithm_used = "Hierarchical"
                self.last_algorithm_used = algorithm_used
            
            print(f"   ✓ Clustering completed using {algorithm_used}")
            print(f"   ✓ Found {n_clusters} clusters from {n_samples:,} spectra")
            
            return labels, linkage_matrix, distance_matrix, algorithm_used
            
        except Exception as e:
            print(f"❌ Clustering failed: {str(e)}")
            raise e
    
    def _create_synthetic_linkage(self, features, labels, n_clusters):
        """Create a synthetic linkage matrix for KMeans results to maintain compatibility."""
        try:
            # Calculate cluster centers
            cluster_centers = []
            for i in range(n_clusters):
                cluster_mask = labels == i
                if np.any(cluster_mask):
                    cluster_centers.append(np.mean(features[cluster_mask], axis=0))
                else:
                    cluster_centers.append(np.zeros(features.shape[1]))
            
            cluster_centers = np.array(cluster_centers)
            
            # Calculate distances between cluster centers
            center_distances = pdist(cluster_centers, metric='euclidean')
            
            # Create a simple linkage matrix
            # This is a simplified version - for full compatibility, you might want
            # to implement a more sophisticated method
            linkage_matrix = linkage(center_distances, method='ward')
            
            return linkage_matrix
            
        except Exception as e:
            print(f"Warning: Could not create synthetic linkage matrix: {str(e)}")
            # Return a minimal linkage matrix for compatibility
            return np.array([[0, 1, 1.0, 2]])
    
    def run_probabilistic_clustering(self, features, n_components):
        """Run probabilistic clustering with GMM and hierarchical sub-typing."""
        try:
            # 1. First-level clustering with GMM
            gmm = GaussianMixture(n_components=n_components, 
                                covariance_type='full', 
                                random_state=42)
            
            # Get cluster probabilities and hard assignments
            gmm.fit(features)  # First fit the model
            cluster_probs = gmm.predict_proba(features)  # Then get probabilities
            hard_labels = gmm.predict(features)  # Get hard assignments
            
            # Store results
            results = {
                'cluster_probs': cluster_probs,
                'labels': hard_labels,
                'gmm': gmm,
                'subtypes': {}
            }
            
            # 2. Hierarchical sub-typing
            results['subtypes'] = self._identify_subtypes(features, hard_labels, n_components)
            
            return results
            
        except Exception as e:
            print(f"Error in probabilistic clustering: {str(e)}")
            raise e
    
    def _identify_subtypes(self, features, labels, n_clusters):
        """Identify sub-types within each cluster using hierarchical clustering."""
        subtypes = {}
        
        for cluster_id in range(n_clusters):
            # Get samples in this cluster
            mask = (labels == cluster_id)
            cluster_features = features[mask]
            
            if len(cluster_features) < 5:  # Skip small clusters
                continue
                
            try:
                # Determine number of sub-clusters (you can make this configurable)
                n_subtypes = min(3, len(cluster_features) // 5)
                
                if n_subtypes > 1:
                    # Calculate distance matrix and linkage
                    dist_matrix = pdist(cluster_features, 'euclidean')
                    Z = linkage(dist_matrix, method='ward')
                    
                    # Get sub-cluster labels
                    sub_labels = fcluster(Z, t=n_subtypes, criterion='maxclust')
                    
                    # Store results
                    subtypes[cluster_id] = {
                        'linkage': Z,
                        'n_subtypes': n_subtypes,
                        'sub_labels': sub_labels,
                        'sample_indices': np.where(mask)[0]  # Store original indices
                    }
                    
            except Exception as e:
                print(f"Error in sub-clustering cluster {cluster_id}: {str(e)}")
        
        return subtypes
    
    def perform_kmeans_clustering(self, features, n_clusters, use_minibatch=False):
        """Perform K-Means or MiniBatchKMeans clustering.
        
        Args:
            features: Feature matrix
            n_clusters: Number of clusters
            use_minibatch: If True, use MiniBatchKMeans; otherwise use standard KMeans
        
        Returns:
            labels, linkage_matrix, distance_matrix, algorithm_used
        """
        try:
            import inspect
            n_samples = features.shape[0]
            n_cores = multiprocessing.cpu_count()
            
            if use_minibatch:
                print(f"Running MiniBatchKMeans on {n_samples:,} samples...")
                batch_size = min(1000, n_samples // 10)
                
                # Build parameters, checking for n_jobs support
                kmeans_params = {
                    'n_clusters': n_clusters,
                    'batch_size': batch_size,
                    'max_iter': 100,
                    'random_state': 42,
                    'n_init': 3
                }
                
                # Check if n_jobs is supported
                sig = inspect.signature(MiniBatchKMeans.__init__)
                if 'n_jobs' in sig.parameters:
                    kmeans_params['n_jobs'] = -1
                    print(f"   Using {n_cores} CPU cores")
                else:
                    print(f"   Single-threaded (sklearn version doesn't support n_jobs for MiniBatchKMeans)")
                
                kmeans = MiniBatchKMeans(**kmeans_params)
                algorithm_used = "MiniBatchKMeans"
            else:
                print(f"Running KMeans on {n_samples:,} samples...")
                
                # Build parameters, checking for n_jobs support
                kmeans_params = {
                    'n_clusters': n_clusters,
                    'max_iter': 300,
                    'random_state': 42,
                    'n_init': 10
                }
                
                # Check if n_jobs is supported (should be available in KMeans)
                sig = inspect.signature(KMeans.__init__)
                if 'n_jobs' in sig.parameters:
                    kmeans_params['n_jobs'] = -1
                    print(f"   Using {n_cores} CPU cores")
                else:
                    print(f"   Single-threaded")
                
                kmeans = KMeans(**kmeans_params)
                algorithm_used = "KMeans"
            
            labels = kmeans.fit_predict(features)
            
            # Create synthetic linkage matrix for compatibility
            linkage_matrix = self._create_synthetic_linkage(features, labels, n_clusters)
            distance_matrix = None  # Not needed for K-Means
            
            self.last_algorithm_used = algorithm_used
            print(f"✓ {algorithm_used} completed: {n_clusters} clusters from {n_samples:,} spectra")
            
            return labels, linkage_matrix, distance_matrix, algorithm_used
            
        except Exception as e:
            print(f"❌ K-Means clustering failed: {str(e)}")
            raise e
    
    def perform_dbscan_clustering(self, features, eps, min_samples):
        """Perform DBSCAN clustering.
        
        Args:
            features: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
        
        Returns:
            labels, linkage_matrix, distance_matrix, algorithm_used
        """
        try:
            from sklearn.cluster import DBSCAN
            
            n_samples = features.shape[0]
            print(f"Running DBSCAN on {n_samples:,} samples (eps={eps}, min_samples={min_samples})...")
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(features)
            
            # DBSCAN labels: -1 for noise, 0+ for clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"✓ DBSCAN completed: {n_clusters} clusters, {n_noise} noise points")
            
            # Create synthetic linkage matrix for compatibility
            linkage_matrix = None
            distance_matrix = None
            algorithm_used = "DBSCAN"
            self.last_algorithm_used = algorithm_used
            
            return labels, linkage_matrix, distance_matrix, algorithm_used
            
        except Exception as e:
            print(f"❌ DBSCAN clustering failed: {str(e)}")
            raise e
    
    def scale_features(self, features):
        """Scale features using StandardScaler."""
        return self.scaler.fit_transform(features)
    
    def apply_pca_reduction(self, features, n_components):
        """Apply PCA dimensionality reduction."""
        if n_components >= features.shape[1]:
            return features, None
        
        pca_reducer = PCA(n_components=n_components)
        features_reduced = pca_reducer.fit_transform(features)
        
        variance_explained = np.sum(pca_reducer.explained_variance_ratio_)
        print(f"   ✓ Variance explained: {variance_explained*100:.2f}%")
        print(f"   ✓ Dimensionality reduction: {features_reduced.shape[1]/features.shape[1]:.1%}")
        
        return features_reduced, pca_reducer
