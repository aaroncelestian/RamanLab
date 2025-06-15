#!/usr/bin/env python3
"""
Top Spectra Selector for Quantitative Analysis Results

Identifies and ranks the best-fitting spectra from the positive class
for display in the Results tab, providing validation and quality control.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpectrumRanking:
    """Information about a ranked spectrum."""
    pixel_index: int
    map_coordinates: Tuple[int, int]  # (row, col) in 2D map
    confidence_score: float
    component_percentage: float
    template_fit_quality: Optional[float] = None
    nmf_component_strength: Optional[float] = None
    ml_probability: Optional[float] = None
    combined_quality_score: float = 0.0
    reasons: List[str] = None  # Why this spectrum was selected
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class TopSpectraSelector:
    """
    Selects and ranks the top-performing spectra from quantitative analysis results.
    
    This class identifies the best examples of the target component by combining
    multiple quality metrics from template fitting, NMF, and ML methods.
    """
    
    def __init__(self, n_top_spectra: int = 5):
        """
        Initialize the top spectra selector.
        
        Args:
            n_top_spectra: Number of top spectra to select
        """
        self.n_top_spectra = n_top_spectra
        self.ranking_weights = {
            'confidence': 0.4,      # Primary weight on confidence
            'template_quality': 0.3, # Template fitting R²
            'component_strength': 0.2, # NMF component strength
            'ml_probability': 0.1   # ML probability (less weight due to potential issues)
        }
        
    def select_top_spectra(self, quantitative_result, 
                          template_coefficients: Optional[np.ndarray] = None,
                          template_r_squared: Optional[np.ndarray] = None,
                          nmf_components: Optional[np.ndarray] = None,
                          ml_probabilities: Optional[np.ndarray] = None,
                          map_shape: Optional[Tuple[int, int]] = None,
                          template_index: Optional[int] = None,
                          nmf_component_index: Optional[int] = None,
                          ml_class_index: Optional[int] = None) -> List[SpectrumRanking]:
        """
        Select and rank the top spectra from quantitative analysis results.
        
        Args:
            quantitative_result: ComponentResult from quantitative analysis
            template_coefficients: Template fitting coefficients (optional)
            template_r_squared: Template fitting R² values (optional)
            nmf_components: NMF component values (optional)
            ml_probabilities: ML class probabilities (optional)
            map_shape: Shape of the 2D map for coordinate conversion
            template_index: Index of template used in analysis
            nmf_component_index: Index of NMF component used in analysis
            ml_class_index: Index of ML class used in analysis
            
        Returns:
            List of SpectrumRanking objects for top spectra
        """
        
        logger.info(f"Selecting top {self.n_top_spectra} spectra from quantitative analysis results")
        
        # Get detected pixels (above confidence threshold)
        detected_pixels = quantitative_result.detection_map
        detected_indices = np.where(detected_pixels)[0]
        
        if len(detected_indices) == 0:
            logger.warning("No pixels detected above confidence threshold")
            return []
            
        logger.info(f"Found {len(detected_indices)} detected pixels to rank")
        
        # Create rankings for all detected pixels
        rankings = []
        
        for pixel_idx in detected_indices:
            ranking = self._create_spectrum_ranking(
                pixel_idx=pixel_idx,
                quantitative_result=quantitative_result,
                template_coefficients=template_coefficients,
                template_r_squared=template_r_squared,
                nmf_components=nmf_components,
                ml_probabilities=ml_probabilities,
                map_shape=map_shape,
                template_index=template_index,
                nmf_component_index=nmf_component_index,
                ml_class_index=ml_class_index
            )
            
            if ranking:
                rankings.append(ranking)
                
        # Sort by combined quality score (descending)
        rankings.sort(key=lambda x: x.combined_quality_score, reverse=True)
        
        # Return top N
        top_rankings = rankings[:self.n_top_spectra]
        
        logger.info(f"Selected top {len(top_rankings)} spectra")
        for i, ranking in enumerate(top_rankings):
            logger.info(f"  Rank {i+1}: Pixel {ranking.pixel_index}, "
                       f"Confidence: {ranking.confidence_score:.3f}, "
                       f"Quality: {ranking.combined_quality_score:.3f}")
            
        return top_rankings
        
    def _create_spectrum_ranking(self, pixel_idx: int, quantitative_result,
                               template_coefficients: Optional[np.ndarray],
                               template_r_squared: Optional[np.ndarray],
                               nmf_components: Optional[np.ndarray],
                               ml_probabilities: Optional[np.ndarray],
                               map_shape: Optional[Tuple[int, int]],
                               template_index: Optional[int],
                               nmf_component_index: Optional[int],
                               ml_class_index: Optional[int]) -> Optional[SpectrumRanking]:
        """Create a ranking object for a single spectrum."""
        
        try:
            # Basic information
            confidence = quantitative_result.confidence_map[pixel_idx]
            percentage = quantitative_result.percentage_map[pixel_idx]
            
            # Convert to map coordinates
            if map_shape:
                row = pixel_idx // map_shape[1]
                col = pixel_idx % map_shape[1]
                map_coordinates = (row, col)
            else:
                # Use linear index as fallback
                map_coordinates = (pixel_idx, 0)
                
            # Extract method-specific quality metrics
            template_quality = None
            if template_r_squared is not None and template_index is not None:
                if template_index < template_r_squared.shape[1]:
                    template_quality = template_r_squared[pixel_idx, template_index]
                    
            nmf_strength = None
            if nmf_components is not None and nmf_component_index is not None:
                if nmf_component_index < nmf_components.shape[1]:
                    nmf_strength = nmf_components[pixel_idx, nmf_component_index]
                    
            ml_prob = None
            if ml_probabilities is not None and ml_class_index is not None:
                if ml_class_index < ml_probabilities.shape[1]:
                    ml_prob = ml_probabilities[pixel_idx, ml_class_index]
                    
            # Calculate combined quality score
            quality_score = self._calculate_combined_quality_score(
                confidence, template_quality, nmf_strength, ml_prob
            )
            
            # Determine reasons for selection
            reasons = self._generate_selection_reasons(
                confidence, percentage, template_quality, nmf_strength, ml_prob
            )
            
            return SpectrumRanking(
                pixel_index=pixel_idx,
                map_coordinates=map_coordinates,
                confidence_score=confidence,
                component_percentage=percentage,
                template_fit_quality=template_quality,
                nmf_component_strength=nmf_strength,
                ml_probability=ml_prob,
                combined_quality_score=quality_score,
                reasons=reasons
            )
            
        except Exception as e:
            logger.warning(f"Failed to create ranking for pixel {pixel_idx}: {str(e)}")
            return None
            
    def _calculate_combined_quality_score(self, confidence: float,
                                        template_quality: Optional[float],
                                        nmf_strength: Optional[float],
                                        ml_prob: Optional[float]) -> float:
        """Calculate a combined quality score from multiple metrics."""
        
        score = 0.0
        total_weight = 0.0
        
        # Confidence score (always available)
        score += confidence * self.ranking_weights['confidence']
        total_weight += self.ranking_weights['confidence']
        
        # Template quality (R²)
        if template_quality is not None:
            score += template_quality * self.ranking_weights['template_quality']
            total_weight += self.ranking_weights['template_quality']
            
        # NMF component strength (normalized)
        if nmf_strength is not None:
            # Normalize to 0-1 range (assuming max strength around 10-20)
            normalized_strength = min(nmf_strength / 20.0, 1.0)
            score += normalized_strength * self.ranking_weights['component_strength']
            total_weight += self.ranking_weights['component_strength']
            
        # ML probability
        if ml_prob is not None:
            score += ml_prob * self.ranking_weights['ml_probability']
            total_weight += self.ranking_weights['ml_probability']
            
        # Normalize by total weight used
        if total_weight > 0:
            score = score / total_weight
            
        return score
        
    def _generate_selection_reasons(self, confidence: float, percentage: float,
                                   template_quality: Optional[float],
                                   nmf_strength: Optional[float],
                                   ml_prob: Optional[float]) -> List[str]:
        """Generate human-readable reasons for spectrum selection."""
        
        reasons = []
        
        # Confidence-based reasons
        if confidence >= 0.8:
            reasons.append("Very high confidence detection")
        elif confidence >= 0.6:
            reasons.append("High confidence detection")
        elif confidence >= 0.4:
            reasons.append("Moderate confidence detection")
            
        # Template-based reasons
        if template_quality is not None:
            if template_quality >= 0.9:
                reasons.append("Excellent template fit (R² ≥ 0.9)")
            elif template_quality >= 0.7:
                reasons.append("Good template fit (R² ≥ 0.7)")
            elif template_quality >= 0.5:
                reasons.append("Moderate template fit (R² ≥ 0.5)")
                
        # NMF-based reasons
        if nmf_strength is not None:
            if nmf_strength >= 10.0:
                reasons.append("Strong NMF component signal")
            elif nmf_strength >= 5.0:
                reasons.append("Moderate NMF component signal")
                
        # ML-based reasons
        if ml_prob is not None:
            if ml_prob >= 0.8:
                reasons.append("High ML classification probability")
            elif ml_prob >= 0.6:
                reasons.append("Moderate ML classification probability")
                
        # Percentage-based reasons
        if percentage >= 80:
            reasons.append("Very high component percentage")
        elif percentage >= 60:
            reasons.append("High component percentage")
        elif percentage >= 40:
            reasons.append("Moderate component percentage")
            
        # Ensure at least one reason
        if not reasons:
            reasons.append("Detected above confidence threshold")
            
        return reasons
        
    def generate_top_spectra_report(self, top_spectra: List[SpectrumRanking],
                                   component_name: str) -> str:
        """Generate a human-readable report of top spectra."""
        
        if not top_spectra:
            return f"No high-quality spectra found for {component_name}."
            
        report = [f"TOP {len(top_spectra)} SPECTRA FOR {component_name.upper()}"]
        report.append("=" * 60)
        
        for i, spectrum in enumerate(top_spectra):
            report.append(f"\nRank {i+1}: Pixel {spectrum.pixel_index} "
                         f"(Map position: {spectrum.map_coordinates[0]}, {spectrum.map_coordinates[1]})")
            report.append(f"  Confidence Score: {spectrum.confidence_score:.3f}")
            report.append(f"  Component Percentage: {spectrum.component_percentage:.1f}%")
            report.append(f"  Combined Quality Score: {spectrum.combined_quality_score:.3f}")
            
            if spectrum.template_fit_quality is not None:
                report.append(f"  Template Fit Quality (R²): {spectrum.template_fit_quality:.3f}")
                
            if spectrum.nmf_component_strength is not None:
                report.append(f"  NMF Component Strength: {spectrum.nmf_component_strength:.2f}")
                
            if spectrum.ml_probability is not None:
                report.append(f"  ML Probability: {spectrum.ml_probability:.3f}")
                
            report.append(f"  Selection Reasons:")
            for reason in spectrum.reasons:
                report.append(f"    • {reason}")
                
        return "\n".join(report)
        
    def get_spectrum_indices_for_extraction(self, top_spectra: List[SpectrumRanking]) -> List[int]:
        """
        Get the pixel indices for spectrum extraction.
        
        This can be used to extract the actual spectral data for display.
        
        Args:
            top_spectra: List of SpectrumRanking objects
            
        Returns:
            List of pixel indices for spectrum extraction
        """
        return [spectrum.pixel_index for spectrum in top_spectra]
        
    def set_ranking_weights(self, confidence: float = 0.4, template_quality: float = 0.3,
                           component_strength: float = 0.2, ml_probability: float = 0.1):
        """
        Adjust the ranking weights for different quality metrics.
        
        Args:
            confidence: Weight for confidence score
            template_quality: Weight for template fitting quality
            component_strength: Weight for NMF component strength
            ml_probability: Weight for ML probability
        """
        
        # Normalize weights to sum to 1
        total = confidence + template_quality + component_strength + ml_probability
        
        self.ranking_weights = {
            'confidence': confidence / total,
            'template_quality': template_quality / total,
            'component_strength': component_strength / total,
            'ml_probability': ml_probability / total
        }
        
        logger.info(f"Updated ranking weights: {self.ranking_weights}")


def create_ui_integration_helper():
    """Helper function for integrating with the Results tab UI."""
    
    integration_code = '''
    # Integration with Results Tab UI
    
    def update_results_tab_with_top_spectra(self, quantitative_result, 
                                           original_spectral_data):
        """
        Update the Results tab to show top 5 interesting spectra from quantitative analysis.
        
        Args:
            quantitative_result: ComponentResult from quantitative analysis
            original_spectral_data: Original spectral data for extraction
        """
        
        from analysis.top_spectra_selector import TopSpectraSelector
        
        # Initialize selector
        selector = TopSpectraSelector(n_top_spectra=5)
        
        # Get the data needed for ranking
        template_coeffs = getattr(self, 'template_coefficients', None)
        template_r2 = getattr(self, 'template_r_squared', None)
        nmf_components = getattr(self, 'nmf_analyzer', None)
        ml_probs = getattr(self, 'classification_results', {}).get('probabilities', None)
        
        # Extract NMF components if available
        nmf_data = None
        if nmf_components and hasattr(nmf_components, 'components'):
            nmf_data = nmf_components.get_components()
            
        # Select top spectra
        top_spectra = selector.select_top_spectra(
            quantitative_result=quantitative_result,
            template_coefficients=template_coeffs,
            template_r_squared=template_r2,
            nmf_components=nmf_data,
            ml_probabilities=ml_probs,
            map_shape=self.map_shape,
            template_index=0,  # Adjust as needed
            nmf_component_index=2,  # Adjust as needed
            ml_class_index=1  # Adjust as needed (use corrected class)
        )
        
        # Extract spectral data for the top spectra
        spectrum_indices = selector.get_spectrum_indices_for_extraction(top_spectra)
        
        # Update Results tab
        self.clear_results_tab_spectra()
        
        for i, (spectrum_ranking, pixel_idx) in enumerate(zip(top_spectra, spectrum_indices)):
            # Extract spectrum data
            wavenumbers = original_spectral_data['wavenumbers']
            intensities = original_spectral_data['intensities'][pixel_idx]
            
            # Create label with quality information
            label = f"Top {i+1}: Pixel {pixel_idx} (Conf: {spectrum_ranking.confidence_score:.2f}, " \\
                   f"Pct: {spectrum_ranking.component_percentage:.1f}%)"
            
            # Add to Results tab
            self.add_spectrum_to_results_tab(
                wavenumbers=wavenumbers,
                intensities=intensities,
                label=label,
                pixel_index=pixel_idx,
                ranking_info=spectrum_ranking
            )
            
        # Generate and display summary
        report = selector.generate_top_spectra_report(top_spectra, quantitative_result.component_name)
        self.update_results_tab_summary(report)
        
        logger.info(f"Updated Results tab with {len(top_spectra)} top spectra")
    '''
    
    return integration_code 