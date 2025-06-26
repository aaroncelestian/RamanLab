#!/usr/bin/env python3
"""
Version information for RamanLab Qt6
"""

__version__ = "1.0.4"
__version_info__ = (1, 0, 4)
__author__ = "Aaron J. Celestian Ph.D."
__copyright__ = "Copyright 2025, RamanLab"
__description__ = "Cross-platform Raman spectrum analysis tool built with Qt6"

# Release information
__release_date__ = "2025-01-27"
__release_name__ = "Windows Database Connection Fix"
__release_status__ = "stable"

# Version history
__changes__ = {
    "1.0.4": {
        "date": "2025-01-27",
        "name": "Windows Database Connection Fix",
        "description": "Fixed critical Windows database connection issue preventing Search/Match functionality",
        "major_features": [
            "Fixed Windows database path mismatch causing 'Empty Database' errors",
            "Enhanced database path detection with intelligent fallback logic",
            "Improved cross-platform database compatibility",
            "Added comprehensive database loading debug information"
        ],
        "breaking_changes": [],
        "technical_notes": [
            "RamanSpectraQt6 now checks both Documents/RamanLab_Qt6/ and script directory for databases",
            "Priority order: Documents folder (primary) → Script directory (fallback) → Create new",
            "Enhanced load_database() method with detailed console debugging output",
            "Maintains backward compatibility with existing installations",
            "All dependent modules (Multi-Spectrum Manager, Cluster Analysis) benefit automatically",
            "Polarization Analyzer already used correct script directory path"
        ],
        "bug_fixes": [
            "Windows users can now access Search/Match functionality without 'Empty Database' errors",
            "Database Manager and Search functionality now use consistent database sources",
            "Resolved path separator issues on Windows systems",
            "Fixed Qt6 QStandardPaths inconsistencies across platforms"
        ]
    },
    "1.0.3": {
        "date": "2025-01-26",
        "name": "Database Manager & Debug Improvements",
        "description": "Added database manager functionality and various debugging improvements",
        "major_features": [
            "Integrated database manager for better data management",
            "Removed database pulldown menu for cleaner interface",
            "Various debugging improvements and fixes"
        ],
        "breaking_changes": [],
        "technical_notes": [
            "Removed Database menu from menu bar for cleaner UI",
            "Database functionality still accessible through Database tab",
            "Enhanced debugging and error handling throughout application",
            "Improved code organization and maintenance"
        ]
    },
    "1.0.2": {
        "date": "2025-01-26",
        "name": "DTW Performance Enhancement",
        "description": "Enhanced DTW algorithm performance and user experience",
        "major_features": [
            "DTW performance warning dialog with time estimates",
            "Enhanced progress tracking for slow algorithms",
            "Unified search architecture for better consistency"
        ],
        "breaking_changes": [],
        "technical_notes": [
            "Added DTW algorithm performance warning dialog",
            "Real-time progress updates every 10% for DTW and Combined algorithms",
            "Unified Basic and Advanced search to use optimized search_filtered_candidates()",
            "Early termination optimization when sufficient matches found",
            "Improved UI responsiveness during long searches with QApplication.processEvents()"
        ]
    },
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