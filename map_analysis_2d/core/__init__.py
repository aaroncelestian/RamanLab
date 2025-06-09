"""Core functionality for 2D Map Analysis."""

from .spectrum_data import SpectrumData
from .cosmic_ray_detection import CosmicRayConfig, SimpleCosmicRayManager
from .template_management import TemplateSpectrum, TemplateSpectraManager
from .file_io import RamanMapData

__all__ = [
    'SpectrumData', 'RamanMapData',
    'CosmicRayConfig', 'SimpleCosmicRayManager',
    'TemplateSpectrum', 'TemplateSpectraManager'
] 