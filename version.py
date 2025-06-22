#!/usr/bin/env python3
"""
Version information for RamanLab Qt6
"""

__version__ = "1.0.1"
__version_info__ = (1, 0, 1)
__author__ = "Aaron J. Celestian, Ph.D."
__copyright__ = "Copyright 2025, RamanLab"
__description__ = "Cross-platform Raman spectrum analysis tool built with Qt6"

# Release information
__release_date__ = "2025-01-26"
__release_name__ = "First Update"
__release_status__ = "stable"

# Version history
__changes__ = {
    "1.0.1": {
        "date": "2025-01-26",
        "name": "First Update",
        "description": "Minor updates and improvements to RamanLab Qt6",
        "major_features": [],
        "breaking_changes": [],
        "technical_notes": [
            "Version bump to 1.0.1",
            "Ongoing development and refinements"
        ]
    },
    "1.0.0": {
        "date": "2025-01-26",
        "name": "Debut Release",
        "description": "Initial stable release of RamanLab Qt6",
        "major_features": [
            "Complete Qt6 GUI framework implementation",
            "Cross-platform compatibility (macOS, Windows, Linux)",
            "Multi-spectrum management system",
            "Advanced Raman analysis tools",
            "Modern dependency management",
            "Comprehensive documentation"
        ],
        "breaking_changes": [
            "Migration from legacy tkinter to Qt6",
            "Updated Python requirements (3.8+)",
            "New project structure and organization"
        ],
        "technical_notes": [
            "Built with PySide6/PyQt6",
            "Modern packaging and distribution",
            "Updated scientific computing stack",
            "Enhanced user interface design"
        ]
    }
}

# Compatibility information
__python_requires__ = ">=3.8"
__qt_version__ = "6.5+"
__platforms__ = ["Windows", "macOS", "Linux"]

def get_version():
    """Return the current version string."""
    return __version__

def get_version_info():
    """Return the current version as a tuple."""
    return __version_info__

def get_full_version():
    """Return detailed version information."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "release_date": __release_date__,
        "release_name": __release_name__,
        "status": __release_status__,
        "python_requires": __python_requires__,
        "qt_version": __qt_version__,
        "platforms": __platforms__
    } 