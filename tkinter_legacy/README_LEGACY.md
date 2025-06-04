# Tkinter Legacy Files

This folder contains all the original tkinter-based GUI files that have been moved here to keep them separate from the new Qt6/PySide6 implementation.

## Main Application Files
- `raman_analysis_app.py` - Original main tkinter application (replaced by `raman_analysis_app_qt6.py`)
- `raman_analysis_app_old.py` - Even older version of the main application
- `main.py` - Original tkinter main entry point (replaced by `main_qt6.py`)
- `multi_spectrum_manager.py` - Original tkinter spectrum manager (replaced by `multi_spectrum_manager_qt6.py`)

## Analysis Tools
- `batch_peak_fitting.py` - Original tkinter batch peak fitting (replaced by `batch_peak_fitting_qt6.py`)
- `peak_fitting.py` - Original tkinter peak fitting module (replaced by `peak_fitting_qt6.py`)
- `mineral_database.py` - Original tkinter database browser (replaced by `database_browser_qt6.py`)
- `raman_database_browser.py` - Another tkinter database browser
- `map_analysis_2d.py` - Original tkinter 2D map analysis (replaced by `map_analysis_2d_qt6.py`)
- `raman_cluster_analysis.py` - Original tkinter cluster analysis (replaced by `raman_cluster_analysis_qt6.py`)
- `raman_group_analysis.py` - Tkinter group analysis tool

## Specialized Analysis Tools
- `raman_polarization_analyzer.py` - Original tkinter polarization analyzer (replaced by `raman_polarization_analyzer_modular_qt6.py`)
- `stress_strain_analyzer.py` - Tkinter stress/strain analysis tool
- `chemical_strain_analyzer.py` - Tkinter chemical strain analysis tool
- `raman_tensor_3d_visualization.py` - Tkinter 3D tensor visualization

## Optimization Tools
- `stage1_orientation_optimizer.py` - Tkinter stage 1 optimizer
- `stage2_probabilistic_optimizer.py` - Tkinter stage 2 optimizer  
- `stage3_advanced_optimizer.py` - Tkinter stage 3 optimizer

## Utility Files
- `raman_spectra.py` - Original tkinter raman spectra utilities (replaced by `raman_spectra_qt6.py`)
- `line_scan_splitter.py` - Tkinter line scan splitter tool
- `update_checker.py` - Tkinter update checker
- `hey_celestian_frequency_analyzer.py` - Tkinter frequency analyzer
- `mixed_mineral_enhancement.py` - Tkinter mineral enhancement tool
- `workflow_guide.py` - Tkinter workflow guide

## Backup Files
- Various `.backup` and `.utf8_backup` files from the tkinter versions

## Test Files
Located in `test_files/` subdirectory:
- Various test and debug files that used tkinter GUI components

## Hey Classification System
Located in `Hey_class/` subdirectory:
- Complete Hey mineral classification system implemented in tkinter

## Dependencies
- `requirements.txt` - Original requirements file for tkinter version (superseded by `requirements_qt6.txt`)

---

**Note**: These files are kept for reference and backup purposes. The active development should use the Qt6/PySide6 versions in the main directory. 