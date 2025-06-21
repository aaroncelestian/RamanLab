"""
Strain Visualization for Battery Analysis
========================================

Visualization tools for LiMn2O4 strain analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union
import sys
import os

# Add parent directory to path for matplotlib config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the matplotlib config if available (following user rules)
try:
    from polarization_ui.matplotlib_config import *
except ImportError:
    # Fallback to standard matplotlib settings
    plt.style.use('default')

class StrainVisualizer:
    """Visualization tools for battery strain analysis results"""
    
    def __init__(self, figure_size: Tuple[int, int] = (12, 8)):
        """Initialize strain visualizer"""
        self.figure_size = figure_size
        
        # Color schemes for different data types
        self.colors = {
            'strain': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'peaks': ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'phases': ['#ff6d6d', '#4ecdc4', '#45b7d1', '#f9ca24'],
            'electrochemical': ['#2c3e50', '#e74c3c']
        }
        
        # Mode colors for consistent plotting
        self.mode_colors = {
            'A1g': '#1f77b4',
            'Eg': '#ff7f0e', 
            'T2g_1': '#2ca02c',
            'T2g_2': '#d62728',
            'Li_O': '#9467bd',
            'Disorder': '#8c564b'
        }
    
    def plot_time_series_overview(self, results: Dict, save_path: str = None) -> plt.Figure:
        """Create overview plot of time series strain analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LiMn2O4 Time Series Strain Analysis Overview', fontsize=16, fontweight='bold')
        
        times = np.array(results['time_points'])
        
        # 1. Strain evolution
        ax1 = axes[0, 0]
        strain_data = results['strain_evolution']
        
        strain_labels = ['ε₁₁', 'ε₂₂', 'ε₃₃', 'ε₁₂', 'ε₁₃', 'ε₂₃']
        for i, label in enumerate(strain_labels):
            ax1.plot(times, strain_data[:, i], 'o-', 
                    color=self.colors['strain'][i], label=label, markersize=4)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Strain')
        ax1.set_title('Strain Tensor Evolution')
        ax1.legend(ncol=2, fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Composition evolution
        ax2 = axes[0, 1]
        if 'composition_evolution' in results:
            composition = results['composition_evolution']
            ax2.plot(times, composition, 'o-', color=self.colors['electrochemical'][0], 
                    markersize=6, label='H fraction')
            ax2.plot(times, 1-composition, 'o-', color=self.colors['electrochemical'][1], 
                    markersize=6, label='Li fraction')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Composition')
        ax2.set_title('H/Li Composition Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Jahn-Teller parameter evolution
        ax3 = axes[1, 0]
        if 'jt_evolution' in results:
            jt_param = results['jt_evolution']
            ax3.plot(times, jt_param, 'o-', color=self.colors['phases'][0], 
                    markersize=6, linewidth=2)
        
        ax3.set_xlabel('Time (s)')  
        ax3.set_ylabel('JT Parameter')
        ax3.set_title('Jahn-Teller Distortion')
        ax3.grid(True, alpha=0.3)
        
        # 4. Peak tracking
        ax4 = axes[1, 1]
        peak_tracking = results.get('peak_tracking', {})
        
        for mode_name, tracking_data in peak_tracking.items():
            if tracking_data['frequencies'] and mode_name in self.mode_colors:
                freqs = np.array(tracking_data['frequencies'])
                ax4.plot(times[:len(freqs)], freqs, 'o-', 
                        color=self.mode_colors[mode_name], 
                        label=mode_name, markersize=4)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Frequency (cm⁻¹)')
        ax4.set_title('Peak Frequency Evolution') 
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_strain_tensor_3d(self, strain_tensor: np.ndarray, 
                             title: str = "Strain State", 
                             save_path: str = None) -> plt.Figure:
        """3D visualization of strain tensor as ellipsoid"""
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert strain vector to 3x3 tensor
        strain_matrix = np.array([
            [strain_tensor[0], strain_tensor[3], strain_tensor[4]],
            [strain_tensor[3], strain_tensor[1], strain_tensor[5]],
            [strain_tensor[4], strain_tensor[5], strain_tensor[2]]
        ])
        
        # Eigenvalue decomposition for principal strains
        eigenvals, eigenvecs = np.linalg.eigh(strain_matrix)
        
        # Create strain ellipsoid
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        
        # Scale factors based on strain eigenvalues
        scale_factor = 100  # Amplify for visualization
        a = 1 + scale_factor * eigenvals[0]
        b = 1 + scale_factor * eigenvals[1]  
        c = 1 + scale_factor * eigenvals[2]
        
        # Ellipsoid coordinates
        x = a * np.outer(np.cos(u), np.sin(v))
        y = b * np.outer(np.sin(u), np.sin(v))
        z = c * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Rotate by eigenvectors
        coords = np.stack([x.flatten(), y.flatten(), z.flatten()])
        rotated = eigenvecs @ coords
        
        x_rot = rotated[0].reshape(x.shape)
        y_rot = rotated[1].reshape(y.shape)
        z_rot = rotated[2].reshape(z.shape)
        
        # Plot ellipsoid
        surf = ax.plot_surface(x_rot, y_rot, z_rot, alpha=0.6, 
                             cmap='viridis', linewidth=0)
        
        # Add principal strain axes
        axis_length = 2.0
        for i, (eval, evec) in enumerate(zip(eigenvals, eigenvecs.T)):
            ax.quiver(0, 0, 0, evec[0]*axis_length, evec[1]*axis_length, evec[2]*axis_length,
                     color=self.colors['strain'][i], arrow_length_ratio=0.1, linewidth=3,
                     label=f'ε₁={eval:.4f}' if i==0 else f'ε₂={eval:.4f}' if i==1 else f'ε₃={eval:.4f}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        # Color bar
        plt.colorbar(surf, ax=ax, shrink=0.5, label='Strain Magnitude')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

if __name__ == "__main__":
    print("Strain Visualization Module for Battery Analysis")
    print("=" * 50)
    
    visualizer = StrainVisualizer()
    print("Visualizer initialized")