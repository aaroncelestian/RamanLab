"""
Advanced Statistical Methods for Raman Cluster Analysis
Supplementary methods for the Advanced Statistics tab.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, f_oneway
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def plot_discriminant_analysis(stats_fig, stats_canvas, X_lda, labels, explained_variance, cv_scores):
    """Plot discriminant analysis results."""
    stats_fig.clear()
    
    # Create subplots
    ax1 = stats_fig.add_subplot(2, 2, 1)  # LDA scatter plot
    ax2 = stats_fig.add_subplot(2, 2, 2)  # Explained variance
    ax3 = stats_fig.add_subplot(2, 2, 3)  # Cross-validation scores
    ax4 = stats_fig.add_subplot(2, 2, 4)  # Cluster separation
    
    # 1. LDA scatter plot (first two components)
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if X_lda.shape[1] >= 2:
            ax1.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                       c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
        else:
            # If only one component, plot against index
            ax1.scatter(range(np.sum(mask)), X_lda[mask, 0], 
                       c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
    
    if X_lda.shape[1] >= 2:
        ax1.set_xlabel(f'LD1 ({explained_variance[0]:.1%} variance)')
        ax1.set_ylabel(f'LD2 ({explained_variance[1]:.1%} variance)')
    else:
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel(f'LD1 ({explained_variance[0]:.1%} variance)')
    ax1.set_title('Linear Discriminant Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Explained variance plot
    components = range(1, len(explained_variance) + 1)
    ax2.bar(components, explained_variance, color='skyblue', edgecolor='navy')
    ax2.set_xlabel('Linear Discriminant')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Explained Variance by Component')
    ax2.set_xticks(components)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cross-validation scores
    fold_numbers = range(1, len(cv_scores) + 1)
    bars = ax3.bar(fold_numbers, cv_scores, color='lightgreen', edgecolor='darkgreen')
    ax3.axhline(y=np.mean(cv_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(cv_scores):.3f}')
    ax3.set_xlabel('Cross-Validation Fold')
    ax3.set_ylabel('Accuracy Score')
    ax3.set_title('Cross-Validation Performance')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cluster separation analysis
    if X_lda.shape[1] >= 1:
        # Calculate pairwise distances between cluster centroids
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(X_lda[mask], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        n_clusters = len(unique_labels)
        
        # Calculate separation matrix
        separation_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    separation_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])
        
        # Plot separation heatmap
        im = ax4.imshow(separation_matrix, cmap='viridis', aspect='auto')
        ax4.set_xticks(range(n_clusters))
        ax4.set_yticks(range(n_clusters))
        ax4.set_xticklabels([f'C{label}' for label in unique_labels])
        ax4.set_yticklabels([f'C{label}' for label in unique_labels])
        ax4.set_title('Cluster Separation Matrix')
        
        # Add text annotations
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    ax4.text(j, i, f'{separation_matrix[i, j]:.2f}', 
                           ha='center', va='center', color='white')
        
        stats_fig.colorbar(im, ax=ax4)
    
    stats_fig.tight_layout()
    stats_canvas.draw()


def display_discriminant_results(stats_results, cv_scores, explained_variance, lda):
    """Display discriminant analysis results."""
    results_text = "Linear Discriminant Analysis Results\n"
    results_text += "=" * 40 + "\n\n"
    
    results_text += f"Cross-Validation Performance:\n"
    results_text += f"• Mean accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}\n"
    results_text += f"• Best fold: {np.max(cv_scores):.3f}\n"
    results_text += f"• Worst fold: {np.min(cv_scores):.3f}\n\n"
    
    results_text += f"Explained Variance by Component:\n"
    for i, var in enumerate(explained_variance):
        results_text += f"• LD{i+1}: {var:.3f} ({var*100:.1f}%)\n"
    
    cumulative_var = np.cumsum(explained_variance)
    results_text += f"\nCumulative Explained Variance:\n"
    for i, cum_var in enumerate(cumulative_var):
        results_text += f"• First {i+1} component(s): {cum_var:.3f} ({cum_var*100:.1f}%)\n"
    
    # Model interpretation
    results_text += f"\nModel Interpretation:\n"
    if np.mean(cv_scores) >= 0.9:
        interpretation = "Excellent cluster separation"
    elif np.mean(cv_scores) >= 0.8:
        interpretation = "Good cluster separation"
    elif np.mean(cv_scores) >= 0.7:
        interpretation = "Moderate cluster separation"
    else:
        interpretation = "Poor cluster separation"
    
    results_text += f"• Classification quality: {interpretation}\n"
    results_text += f"• Number of discriminant functions: {len(explained_variance)}\n"
    
    stats_results.setText(results_text)


def perform_permanova(intensities, labels, alpha, n_permutations):
    """Perform PERMANOVA test."""
    # Calculate distance matrix
    distances = pdist(intensities, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Calculate observed F-statistic
    def calculate_f_stat(dist_matrix, group_labels):
        unique_groups = np.unique(group_labels)
        n_total = len(group_labels)
        n_groups = len(unique_groups)
        
        # Total sum of squares
        grand_mean_distances = np.mean(dist_matrix)
        total_ss = np.sum((dist_matrix - grand_mean_distances) ** 2)
        
        # Within-group sum of squares
        within_ss = 0
        for group in unique_groups:
            group_mask = group_labels == group
            group_distances = dist_matrix[np.ix_(group_mask, group_mask)]
            group_mean = np.mean(group_distances)
            within_ss += np.sum((group_distances - group_mean) ** 2)
        
        # Between-group sum of squares
        between_ss = total_ss - within_ss
        
        # Degrees of freedom
        df_between = n_groups - 1
        df_within = n_total - n_groups
        
        # F-statistic
        if df_within > 0 and within_ss > 0:
            f_stat = (between_ss / df_between) / (within_ss / df_within)
        else:
            f_stat = 0
        
        return f_stat, df_between, df_within
    
    observed_f, df_between, df_within = calculate_f_stat(distance_matrix, labels)
    
    # Permutation test
    permuted_f_stats = []
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(labels)
        perm_f, _, _ = calculate_f_stat(distance_matrix, permuted_labels)
        permuted_f_stats.append(perm_f)
    
    permuted_f_stats = np.array(permuted_f_stats)
    p_value = np.sum(permuted_f_stats >= observed_f) / n_permutations
    
    return {
        'test_statistic': observed_f,
        'p_value': p_value,
        'df_between': df_between,
        'df_within': df_within,
        'significant': p_value < alpha,
        'permuted_stats': permuted_f_stats,
        'method': 'PERMANOVA'
    }


def perform_anosim(intensities, labels, alpha, n_permutations):
    """Perform ANOSIM test."""
    # Calculate distance matrix
    distances = pdist(intensities, metric='euclidean')
    distance_matrix = squareform(distances)
    
    def calculate_anosim_r(dist_matrix, group_labels):
        unique_groups = np.unique(group_labels)
        n_samples = len(group_labels)
        
        # Calculate within-group and between-group distances
        within_distances = []
        between_distances = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if group_labels[i] == group_labels[j]:
                    within_distances.append(dist_matrix[i, j])
                else:
                    between_distances.append(dist_matrix[i, j])
        
        within_distances = np.array(within_distances)
        between_distances = np.array(between_distances)
        
        # Calculate ANOSIM R statistic
        if len(within_distances) > 0 and len(between_distances) > 0:
            mean_within = np.mean(within_distances)
            mean_between = np.mean(between_distances)
            r_stat = (mean_between - mean_within) / (mean_between + mean_within) * 2
        else:
            r_stat = 0
        
        return r_stat
    
    observed_r = calculate_anosim_r(distance_matrix, labels)
    
    # Permutation test
    permuted_r_stats = []
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(labels)
        perm_r = calculate_anosim_r(distance_matrix, permuted_labels)
        permuted_r_stats.append(perm_r)
    
    permuted_r_stats = np.array(permuted_r_stats)
    p_value = np.sum(permuted_r_stats >= observed_r) / n_permutations
    
    return {
        'test_statistic': observed_r,
        'p_value': p_value,
        'significant': p_value < alpha,
        'permuted_stats': permuted_r_stats,
        'method': 'ANOSIM'
    }


def perform_kruskal_wallis(intensities, labels, alpha):
    """Perform Kruskal-Wallis test."""
    # Test each wavenumber separately
    n_features = intensities.shape[1]
    h_statistics = []
    p_values = []
    
    unique_labels = np.unique(labels)
    
    for feature_idx in range(n_features):
        feature_data = intensities[:, feature_idx]
        group_data = [feature_data[labels == label] for label in unique_labels]
        
        try:
            h_stat, p_val = kruskal(*group_data)
            h_statistics.append(h_stat)
            p_values.append(p_val)
        except:
            h_statistics.append(0)
            p_values.append(1)
    
    h_statistics = np.array(h_statistics)
    p_values = np.array(p_values)
    
    # Multiple testing correction (Bonferroni)
    corrected_p_values = p_values * n_features
    corrected_p_values = np.minimum(corrected_p_values, 1.0)
    
    # Overall test result
    significant_features = np.sum(corrected_p_values < alpha)
    overall_significant = significant_features > 0
    
    return {
        'test_statistic': np.mean(h_statistics),
        'p_value': np.min(p_values),
        'corrected_p_values': corrected_p_values,
        'significant': overall_significant,
        'significant_features': significant_features,
        'total_features': n_features,
        'method': 'Kruskal-Wallis'
    }


def perform_anova(intensities, labels, alpha):
    """Perform one-way ANOVA test."""
    # Test each wavenumber separately
    n_features = intensities.shape[1]
    f_statistics = []
    p_values = []
    
    unique_labels = np.unique(labels)
    
    for feature_idx in range(n_features):
        feature_data = intensities[:, feature_idx]
        group_data = [feature_data[labels == label] for label in unique_labels]
        
        try:
            f_stat, p_val = f_oneway(*group_data)
            f_statistics.append(f_stat)
            p_values.append(p_val)
        except:
            f_statistics.append(0)
            p_values.append(1)
    
    f_statistics = np.array(f_statistics)
    p_values = np.array(p_values)
    
    # Multiple testing correction (Bonferroni)
    corrected_p_values = p_values * n_features
    corrected_p_values = np.minimum(corrected_p_values, 1.0)
    
    # Overall test result
    significant_features = np.sum(corrected_p_values < alpha)
    overall_significant = significant_features > 0
    
    return {
        'test_statistic': np.mean(f_statistics),
        'p_value': np.min(p_values),
        'corrected_p_values': corrected_p_values,
        'significant': overall_significant,
        'significant_features': significant_features,
        'total_features': n_features,
        'method': 'ANOVA'
    }


def plot_significance_results(stats_fig, stats_canvas, results, method, wavenumbers=None):
    """Plot statistical significance results."""
    stats_fig.clear()
    
    if method in ['PERMANOVA', 'ANOSIM']:
        # Permutation-based methods
        ax1 = stats_fig.add_subplot(2, 2, 1)  # Null distribution
        ax2 = stats_fig.add_subplot(2, 2, 2)  # Test result
        
        # Plot null distribution
        permuted_stats = results['permuted_stats']
        observed_stat = results['test_statistic']
        
        ax1.hist(permuted_stats, bins=50, alpha=0.7, color='lightblue', 
                edgecolor='black', label='Null distribution')
        ax1.axvline(x=observed_stat, color='red', linestyle='--', linewidth=2,
                   label=f'Observed: {observed_stat:.3f}')
        ax1.set_xlabel(f'{method} Statistic')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{method} Null Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot test result
        p_value = results['p_value']
        significance = "Significant" if results['significant'] else "Not Significant"
        
        colors = ['red' if results['significant'] else 'green']
        bars = ax2.bar(['Test Result'], [p_value], color=colors, alpha=0.7)
        ax2.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='α = 0.05')
        ax2.set_ylabel('p-value')
        ax2.set_title(f'{method} Test Result\n{significance}')
        ax2.set_ylim(0, max(0.1, p_value * 1.2))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add text annotation
        ax2.text(0, p_value + 0.01, f'p = {p_value:.4f}', 
                ha='center', va='bottom', fontweight='bold')
        
    else:
        # Feature-wise methods (ANOVA, Kruskal-Wallis)
        ax1 = stats_fig.add_subplot(2, 2, 1)  # p-value distribution
        ax2 = stats_fig.add_subplot(2, 2, 2)  # Significant features
        ax3 = stats_fig.add_subplot(2, 2, 3)  # Corrected p-values
        
        corrected_p_values = results['corrected_p_values']
        
        # p-value distribution
        ax1.hist(corrected_p_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        ax1.set_xlabel('Corrected p-value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Corrected p-values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Significant features summary
        significant_features = results['significant_features']
        total_features = results['total_features']
        
        labels = ['Significant', 'Not Significant']
        sizes = [significant_features, total_features - significant_features]
        colors = ['red', 'lightgray']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Feature Significance\n({significant_features}/{total_features} significant)')
        
        # Plot corrected p-values vs wavenumber
        if wavenumbers is not None and len(wavenumbers) == len(corrected_p_values):
            significant_mask = corrected_p_values < 0.05
            ax3.scatter(wavenumbers[~significant_mask], corrected_p_values[~significant_mask], 
                       c='gray', alpha=0.5, s=20, label='Not significant')
            if np.any(significant_mask):
                ax3.scatter(wavenumbers[significant_mask], corrected_p_values[significant_mask], 
                           c='red', s=30, label='Significant')
            ax3.axhline(y=0.05, color='black', linestyle='--', alpha=0.7)
            ax3.set_xlabel('Wavenumber (cm⁻¹)')
            ax3.set_ylabel('Corrected p-value')
            ax3.set_title('Significance by Wavenumber')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    stats_fig.tight_layout()
    stats_canvas.draw()


def display_significance_results(stats_results, results, method, alpha):
    """Display statistical significance results."""
    results_text = f"Statistical Significance Testing - {method}\n"
    results_text += "=" * 50 + "\n\n"
    
    results_text += f"Test Parameters:\n"
    results_text += f"• Significance level (α): {alpha:.3f}\n"
    
    if method in ['PERMANOVA', 'ANOSIM']:
        results_text += f"• Permutations: {len(results['permuted_stats'])}\n\n"
        
        results_text += f"Test Results:\n"
        results_text += f"• {method} statistic: {results['test_statistic']:.4f}\n"
        results_text += f"• p-value: {results['p_value']:.4f}\n"
        results_text += f"• Significant: {'Yes' if results['significant'] else 'No'}\n\n"
        
        if method == 'PERMANOVA':
            results_text += f"• Degrees of freedom (between): {results['df_between']}\n"
            results_text += f"• Degrees of freedom (within): {results['df_within']}\n"
        
        # Interpretation
        if results['significant']:
            results_text += f"\nInterpretation:\n"
            results_text += f"The clusters show statistically significant differences "
            results_text += f"(p < {alpha:.3f}). The observed clustering is unlikely "
            results_text += f"to have occurred by chance.\n"
        else:
            results_text += f"\nInterpretation:\n"
            results_text += f"The clusters do not show statistically significant differences "
            results_text += f"(p ≥ {alpha:.3f}). The observed clustering may have "
            results_text += f"occurred by chance.\n"
    
    else:
        # Feature-wise methods
        results_text += f"\nTest Results:\n"
        results_text += f"• Average test statistic: {results['test_statistic']:.4f}\n"
        results_text += f"• Minimum p-value: {results['p_value']:.4f}\n"
        results_text += f"• Significant features: {results['significant_features']}/{results['total_features']}\n"
        results_text += f"• Percentage significant: {results['significant_features']/results['total_features']*100:.1f}%\n"
        results_text += f"• Overall significant: {'Yes' if results['significant'] else 'No'}\n\n"
        
        # Interpretation
        if results['significant']:
            results_text += f"Interpretation:\n"
            results_text += f"Multiple spectral features show statistically significant "
            results_text += f"differences between clusters after Bonferroni correction. "
            results_text += f"The clustering reveals meaningful chemical differences.\n"
        else:
            results_text += f"Interpretation:\n"
            results_text += f"No spectral features show statistically significant "
            results_text += f"differences between clusters after multiple testing correction. "
            results_text += f"The clustering may not reflect meaningful chemical differences.\n"
    
    stats_results.setText(results_text) 