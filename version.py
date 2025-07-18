#!/usr/bin/env python3
"""
Version information for RamanLab Qt6
"""

__version__ = "1.1.4"
__version_info__ = (1, 1, 4)
__author__ = "Aaron J. Celestian Ph.D."
__maintainer__ = "Aaron J. Celestian Ph.D."
__email__ = "aaron.celestian@gmail.com"
__copyright__ = "Copyright 2025, RamanLab"
__description__ = "Cross-platform Raman spectrum analysis tool built with Qt6"

# Release information
__release_date__ = "2025-01-28"
__release_name__ = "Jupyter Console Integration & Enhanced Dependency Management"
__release_status__ = "stable"

# Version history
__changes__ = {
    "1.1.4": {
        "date": "2025-01-28",
        "name": "Jupyter Console Integration & Enhanced Dependency Management",
        "description": "Added interactive Jupyter console integration and comprehensive dependency management tools",
        "major_features": [
            "🐍 Interactive Jupyter console integration with qtconsole, jupyter-client, and ipykernel",
            "📦 Automated dependency update script with interactive and command-line modes",
            "🔍 Enhanced dependency checker with Jupyter package detection and component status",
            "📋 Comprehensive user documentation and troubleshooting guides",
            "💻 Live data access and custom analysis capabilities within RamanLab",
            "🛠️ Multiple update options: automatic, manual, and full system updates"
        ],
        "breaking_changes": [],
        "technical_notes": [
            "Added qtconsole>=5.4.0 for interactive console widget integration",
            "Added jupyter-client>=7.0.0 for kernel management and communication",
            "Added ipykernel>=6.0.0 for enhanced Python environment support",
            "Created update_dependencies.py script with --jupyter, --core, --all flags",
            "Enhanced check_dependencies.py with Jupyter package detection",
            "Added Interactive Console component to availability tracking",
            "Implemented virtual environment detection and warnings",
            "Added graceful fallback - RamanLab works fine without Jupyter packages"
        ],
        "bug_fixes": [
            "Improved dependency management workflow for users",
            "Enhanced user experience with clear update instructions",
            "Added comprehensive troubleshooting documentation",
            "Fixed dependency checking loop for packages with 4-tuple format"
        ],
        "future_roadmap": [
            "Enhanced interactive console features and RamanLab integration",
            "Advanced scripting capabilities within the console environment",
            "Custom analysis workflow development tools"
        ]
    },
    "1.1.3": {
        "date": "2025-01-28",
        "name": "Matplotlib Toolbar Fix",
        "description": "Fixed matplotlib toolbar button vertical centering across all RamanLab modules",
        "major_features": [],
        "breaking_changes": [],
        "technical_notes": [
            "Fixed matplotlib toolbar vertical centering by simplifying CSS approach and removing complex property overrides",
            "Adopted proven stylesheet approach from batch processing monitor for consistent button alignment",
            "Removed problematic Qt property overrides that were causing alignment issues",
            "Simplified CompactNavigationToolbar implementation for better cross-platform compatibility"
        ],
        "bug_fixes": [
            "Fixed matplotlib toolbar button vertical centering across all modules",
            "Resolved CSS alignment issues in polarization_ui/matplotlib_config.py",
            "Improved toolbar appearance consistency across different RamanLab components"
        ],
        "future_roadmap": []
    },
    "1.1.2": {
        "date": "2025-01-28",
        "name": "Major Feature Update: Auto-Update & Advanced Batch Processing",
        "description": "Revolutionary update introducing automatic background updates and completely redesigned sophisticated batch peak fitting system",
        "major_features": [
            "🚀 Automatic background git pull functionality for seamless updates",
            "🔬 Completely new sophisticated batch peak fitting system with advanced algorithms",
            "🎯 Enhanced peak detection and fitting algorithms with improved accuracy",
            "🔄 Streamlined interface with removal of legacy batch processing entry points",
            "📊 Advanced batch processing capabilities with sophisticated analysis workflows",
            "🛠️ Improved code architecture and maintainability throughout the system"
        ],
        "breaking_changes": [
            "Legacy batch peak fitting button removed from main interface (replaced with integrated system)",
            "Old batch processing methods deprecated in favor of new sophisticated implementation"
        ],
        "technical_notes": [
            "Implemented automatic background git pull system for continuous updates",
            "Completely redesigned batch peak fitting architecture with advanced algorithms",
            "New sophisticated peak detection and analysis workflows",
            "Enhanced error handling and robustness in batch processing",
            "Improved performance and accuracy in peak fitting algorithms",
            "Removed legacy batch peak fitting button from raman_analysis_app_qt6.py Advanced tab",
            "Eliminated redundant launch_batch_peak_fitting() method and supporting infrastructure",
            "Cleaned up obsolete code while maintaining backward compatibility",
            "Enhanced system architecture for future feature expansion"
        ],
        "bug_fixes": [
            "Eliminated confusion from duplicate batch processing entry points",
            "Improved system reliability and error handling",
            "Enhanced overall application consistency and user experience",
            "Fixed legacy code inconsistencies and maintenance issues"
        ],
        "future_roadmap": [
            "Continued improvements to the new batch peak fitting system",
            "Additional sophisticated analysis algorithms planned",
            "Enhanced automation features in development",
            "Expanded integration capabilities for advanced workflows"
        ]
    },
    "1.1.0": {
        "date": "2025-01-28",
        "name": "Batch Peak Fitting Module Enhancement",
        "description": "Major improvements to batch peak fitting module availability and robustness",
        "major_features": [
            "Fixed batch peak fitting module import issues for all users",
            "Added comprehensive dependency checker with GUI integration",
            "Enhanced error handling with fallback peak detection methods",
            "Implemented graceful degradation when core modules unavailable",
            "Added detailed diagnostic tools and troubleshooting guidance"
        ],
        "breaking_changes": [],
        "technical_notes": [
            "Added robust import path setup with automatic RamanLab root detection",
            "Enhanced BatchPeakFittingAdapter with fallback scipy-based peak detection",
            "Comprehensive MODULE_STATUS tracking for all dependencies",
            "Updated launch function with detailed module availability checking",
            "Completely rewrote check_dependencies.py for batch peak fitting diagnostics",
            "Added GUI-based dependency checker accessible from Advanced tab",
            "Enhanced error messages throughout to guide users toward solutions"
        ],
        "bug_fixes": [
            "Fixed 'ModuleNotFoundError: no module named core.config_manager' issue",
            "Resolved batch peak fitting 'Module not available' errors",
            "Fixed Python path resolution issues preventing module imports",
            "Enhanced import error handling with specific troubleshooting steps",
            "Added missing __init__.py file verification and creation guidance"
        ]
    },
    "1.0.5": {
        "date": "2025-01-28",
        "name": "Peak Fitting Enhancement",
        "description": "Major improvements to batch peak fitting algorithms and reliability",
        "major_features": [
            "Simplified and more robust background subtraction algorithms",
            "Improved peak detection with baseline-relative thresholding",
            "Enhanced peak fitting parameter estimation and bounds",
            "Better initial width estimation based on peak spacing",
            "More generous fitting bounds for improved convergence"
        ],
        "breaking_changes": [],
        "technical_notes": [
            "Replaced complex background subtraction with simplified rolling minimum approach",
            "Added try/catch error handling with fallback methods for all background algorithms",
            "Peak detection now uses signal above baseline for better sensitivity",
            "Increased curve fitting iterations (5000) and improved algorithm (TRF)",
            "Better edge filtering and noise handling in peak detection",
            "Baseline estimation using rolling minimum window for peak detection"
        ],
        "bug_fixes": [
            "Fixed poor peak fitting quality caused by overly complex background algorithms",
            "Resolved background subtraction artifacts that interfered with peak fitting",
            "Improved peak detection reliability on noisy spectra",
            "Fixed restrictive parameter bounds that prevented proper convergence",
            "Enhanced error handling prevents crashes during problematic fits"
        ]
    },
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
        "author": __author__,
        "maintainer": __maintainer__,
        "email": __email__,
        "python_requires": __python_requires__,
        "qt_version": __qt_version__,
        "platforms": __platforms__
    } 