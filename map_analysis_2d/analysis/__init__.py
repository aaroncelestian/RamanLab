"""
Analysis module for Raman spectroscopy data.

This module contains various analysis algorithms including:
- Principal Component Analysis (PCA)
- Non-negative Matrix Factorization (NMF)
- Machine Learning Classification (Random Forest, Clustering)
- Microplastic Detection (specialized for weak signals)
"""

from .pca_analysis import PCAAnalyzer
from .nmf_analysis import NMFAnalyzer
from .ml_classification import RandomForestAnalyzer, UnsupervisedAnalyzer, SpectrumLoader
from .microplastic_detector import MicroplasticDetector

__all__ = [
    'PCAAnalyzer',
    'NMFAnalyzer', 
    'RandomForestAnalyzer',
    'UnsupervisedAnalyzer',
    'SpectrumLoader',
    'MicroplasticDetector'
] 