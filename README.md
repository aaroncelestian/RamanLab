# RamanLab

**Advanced Raman Spectroscopy Analysis Suite**

RamanLab is a comprehensive Python-based application for Raman spectroscopy data analysis, featuring peak fitting, database management, cluster analysis, 2D mapping, and advanced visualization tools.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8 or higher**
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 8GB minimum (16GB+ recommended for large datasets)
- **Storage**: 500MB for application + 250MB for database

### Installation

1. **Clone or download the repository:**
   ```bash
   git clone https://github.com/aaroncelestian/RamanLab.git
   cd RamanLab
   ```

2. **Create a dedicated Python environment** (Recommended):
   
   It is strongly recommended to create a dedicated virtual environment for RamanLab to avoid dependency conflicts:
   
   ```bash
   python -m venv ramanlab_env
   ```
   
   **Activate the environment:**
   - **macOS/Linux**:
     ```bash
     source ramanlab_env/bin/activate
     ```
   - **Windows**:
     ```bash
     ramanlab_env\Scripts\activate
     ```
   
   You should see `(ramanlab_env)` in your terminal prompt when the environment is active.

3. **Install dependencies:**
   ```bash
   pip install -r requirements_qt6.txt
   ```

4. **Verify installation** (Recommended):
   ```bash
   python check_dependencies.py
   ```
   
   This will check:
   - ✅ Python version compatibility
   - ✅ All required packages and versions
   - ✅ Qt6 framework availability
   - ✅ System resources (RAM, CPU, disk)
   - ✅ Database files presence
   - ✅ Component availability
   
   If issues are found, the script provides specific installation commands.

5. **Download the database file** (Required - 200MB+):
   
   The main Raman spectra database is not included in the repository due to its size.
   
   **Download link:** https://zenodo.org/records/15742626
   
   **Installation locations** (choose one):
   
   - **Option 1 (Recommended)**: User Documents folder
     - **Windows**: `C:\Users\<YourUsername>\Documents\RamanLab_Qt6\RamanLab_Database_20250602.pkl`
     - **macOS**: `~/Documents/RamanLab_Qt6/RamanLab_Database_20250602.pkl`
     - **Linux**: `~/Documents/RamanLab_Qt6/RamanLab_Database_20250602.pkl`
   
   - **Option 2**: Application directory
     - Place `RamanLab_Database_20250602.pkl` in the same folder as the Python scripts

6. **Launch RamanLab:**
   ```bash
   python main_qt6.py
   ```

### Optional: Install Desktop Icon

For convenient access, install a desktop shortcut/application icon:

```bash
python install_desktop_icon.py
```

This will create:
- **Windows**: Desktop shortcut (.lnk) with icon
- **macOS**: Application bundle in ~/Applications
- **Linux**: Desktop entry in applications menu and desktop

**To uninstall the icon:**
```bash
python install_desktop_icon.py --uninstall
```


---

## 📦 Database Files

RamanLab requires two database files:

### 1. Main Raman Database (Required)
- **File**: `RamanLab_Database_20250602.pkl`
- **Size**: ~200MB
- **Contains**: 6,939+ experimental Raman spectra
- **Format**: Python pickle (.pkl)
- **Download**: https://zenodo.org/records/15742626

### 2. Mineral Modes Database (Optional)
- **File**: `mineral_modes.pkl`
- **Size**: ~50MB
- **Contains**: Mineral vibrational mode data
- **Format**: Python pickle (.pkl)

**Note**: If the database is not found, RamanLab will start with an empty database and show a warning dialog. You can import spectra manually or download the database file separately.

---

## 🎯 Features

### Core Analysis Tools

- **Peak Fitting**: Multi-peak Lorentzian, Gaussian, Voigt, and asymmetric Voigt fitting
- **Baseline Correction**: ALS, polynomial, rolling ball, and manual methods
- **Database Management**: Store, search, and compare 6,939+ reference spectra
- **Batch Processing**: Process multiple files with automated workflows

### Advanced Modules

- **Cluster Analysis**: Hierarchical clustering, K-means, UMAP visualization
- **2D Raman Mapping**: Spatial analysis with cosmic ray removal and parallel processing
- **Polarization Analysis**: Crystal orientation and tensor analysis
- **Machine Learning**: Automated classification and feature extraction

### Performance Features

- **Parallel Processing**: Utilizes all CPU cores for faster analysis
- **Large Dataset Support**: Handles 80,000+ spectra with caching
- **Real-time Preview**: Live parameter adjustment with instant feedback

---

## 📚 Application Modules

### Main Application
```bash
python raman_analysis_app_qt6.py
```
- Peak detection and fitting
- Baseline correction
- Database search and management
- Spectrum visualization

### Cluster Analysis
```bash
python raman_cluster_analysis_qt6.py
```
- Hierarchical clustering
- K-means clustering
- UMAP dimensionality reduction
- Feature importance analysis

### 2D Map Analysis
```bash
python map_analysis_2d_qt6.py
```
- Spatial Raman mapping
- Integrated intensity maps
- Peak position/width mapping
- Cosmic ray removal

### Database Browser
```bash
python database_browser_qt6.py
```
- Browse 6,939+ reference spectra
- Search by name, chemical family, or metadata
- Export filtered subsets
- Database statistics

### Polarization Analyzer
```bash
python raman_polarization_analyzer_qt6.py
```
- Crystal orientation analysis
- Raman tensor calculations
- Polarization-dependent measurements

---

## 🔧 Configuration

### Database Paths

RamanLab searches for databases in this order:

1. **User Documents folder**: e.g. `~/Documents/RamanLab_Qt6/`
2. **Application directory**: Same folder as Python scripts

### Performance Settings

For optimal performance on multi-core systems:
- Parallel processing is enabled by default
- Adjust `n_jobs` parameters in code if needed
- See `performance_fixes_summary.txt` for details

---

## 📖 Usage Examples

### Basic Workflow

1. **Load a spectrum**: File → Import Spectrum
2. **Apply baseline correction**: Process tab → Select method → Apply
3. **Detect peaks**: Process tab → Detect Peaks
4. **Fit peaks**: Process tab → Fit Peaks
5. **Save results**: File → Save or Export

### Database Search

1. **Load your spectrum**: Import your data
2. **Search database**: Search tab → Search Database
3. **View matches**: Results show top matches with correlation scores
4. **Compare spectra**: Click on matches to overlay with your data

### Batch Processing

1. **Open Advanced Baseline Subtraction**: Advanced tab
2. **Select folder**: Browse to folder with multiple files
3. **Configure parameters**: Choose baseline method and settings
4. **Process Batch**: All files processed automatically

---

## 🐛 Troubleshooting

### Dependency Check Failures

**Problem**: `check_dependencies.py` reports missing or outdated packages

**Solution**:
1. **Ensure virtual environment is activated** (if you created one):
   ```bash
   # macOS/Linux:
   source ramanlab_env/bin/activate
   
   # Windows:
   ramanlab_env\Scripts\activate
   ```

2. **Install all requirements**:
   ```bash
   pip install -r requirements_qt6.txt
   ```

3. **Upgrade outdated packages**:
   ```bash
   pip install --upgrade <package-name>
   ```

4. **Check Python version**:
   ```bash
   python --version  # Should be 3.8 or higher
   ```

The dependency checker provides specific commands for any issues found.

---

### "Database Not Found" Warning

**Problem**: Application starts with empty database (0 spectra)

**Solution**:
1. Download `RamanLab_Database_20250602.pkl` (200MB+)
2. Place in `Documents/RamanLab_Qt6/` folder
3. Restart RamanLab

### Import Errors

**Problem**: Cannot import database files

**Solution**:
- Ensure file is `.pkl` format (not `.sqlite`)
- Check file has `'spectra'` dictionary key
- Try re-downloading the database file

### Performance Issues

**Problem**: Slow processing on large datasets

**Solutions**:
- Enable parallel processing (default)
- Use data truncation to reduce wavenumber range
- Apply spectral downsampling for clustering
- Check `performance_fixes_summary.txt`

### Window Too Large for Screen

**Problem**: Application window doesn't fit on laptop screens

**Solution**:
- Minimum resolution: 1024x600
- Scroll areas enabled for small screens
- Resize window as needed

---

## 📝 File Formats

### Supported Import Formats

- **Text files** (`.txt`): Tab or comma-delimited
- **CSV files** (`.csv`): Comma-separated values
- **Database files** (`.pkl`): RamanLab pickle format

### Expected Data Format

```
# Optional header lines (start with #)
Wavenumber    Intensity
100.0         1234.5
101.0         1245.8
...
```

### Export Formats

- **Processed spectra**: `.txt` with metadata headers
- **Peak parameters**: `.csv` with detailed fit results
- **Database exports**: `.pkl` with full metadata

---

## 🔬 Technical Details

### Database Structure

The database uses Python pickle format with this structure:

```python
{
    'metadata': {
        'export_date': '2025-06-02',
        'total_spectra': 6939,
        'ramanlab_version': 'Qt6'
    },
    'spectra': {
        'spectrum_name': {
            'wavenumbers': [100, 101, ...],
            'intensities': [1234, 1245, ...],
            'peaks': [...],
            'metadata': {...}
        },
        ...
    }
}
```

### System Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB+ for large datasets
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Qt**: PySide6 (Qt6)

### Dependencies

See `requirements_qt6.txt` for full list. Key packages:
- PySide6 (Qt6 GUI)
- NumPy, SciPy (numerical computing)
- Matplotlib (plotting)
- scikit-learn (machine learning)
- joblib (parallel processing)

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License

Copyright (c) 2025 Aaron Celestian

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 🙏 Acknowledgements

RamanLab incorporates reference data from several excellent open-access databases:

### RRUFF Database
The mineral Raman spectra database includes data from the **RRUFF Project** (https://rruff.info), a comprehensive database of Raman spectra, X-ray diffraction, and chemistry data for minerals. We gratefully acknowledge the RRUFF Project and its contributors for making this valuable resource freely available to the scientific community.

**Citation**: Lafuente, B., Downs, R.T., Yang, H., and Stone, N. (2015) The power of databases: the RRUFF project. In: Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, eds. Berlin, Germany, W. De Gruyter, pp 1-30.

### WURM Database
Raman spectra for organic and biological materials are sourced from the **WURM (Wollongong University Raman Microscopy) Database** (https://wurm.info), a comprehensive spectral library for organic materials, biological samples, and minerals.

**Citation**: WURM Database, University of Wollongong, https://wurm.info

### SLOPP/E Plastics Database
Microplastic identification capabilities are enhanced by reference spectra from the **SLOPP/E (Spectral Library of Plastic Particles) Database**, providing comprehensive Raman spectra for various plastic types and additives.

**Citation**: SLOPP/E Database, https://www.slopp-e.de

---

## 📖 Citation

If you use RamanLab or the database in your research, please cite:

**Database:**
```
RamanLab Database (2025). Zenodo. https://doi.org/10.5281/zenodo.15742626
```

**Software:**
```
Celestian, A. (2025). RamanLab: A Comprehensive Cross-Platform Software Suite 
for Advanced Raman Spectroscopy Analysis. Authorea.
https://doi.org/10.22541/au.175269971.11603902/v1
```

**DOI**: https://doi.org/10.22541/au.175269971.11603902/v1

**Full Article**: https://www.authorea.com/users/19309/articles/1315548-ramanlab-a-comprehensive-cross-platform-software-suite-for-advanced-raman-spectroscopy-analysis

The database is permanently archived on Zenodo with DOI: `10.5281/zenodo.15742626`

---

## 📧 Contact

For questions, bug reports, or support:
- **Support Forum**: https://ramanlab.freeforums.net/#category-3
- **GitHub Issues**: https://github.com/aaroncelestian/RamanLab/issues
- **Email**: aaron.celestian@gmail.com
- **Repository**: https://github.com/aaroncelestian/RamanLab
- **Database**: https://zenodo.org/records/15742626

---

## 🔄 Version History

### Current Version
- Qt6 compatible
- 6,939+ reference spectra
- Parallel processing support
- Enhanced performance for large datasets

### Recent Updates
- Removed confusing SQLite references (database is pickle format)
- Added auto-preview in baseline subtraction
- Fixed data truncation validation
- Improved small screen support
- Enhanced database import/export

---

## 📚 Additional Documentation

- `FAST_LAUNCH_README.txt` - Performance optimization tips
- `performance_fixes_summary.txt` - Parallel processing details
- `requirements_qt6.txt` - Python dependencies

---

## ⚠️ Important Notes

1. **Database file is required** for full functionality (200MB download)
2. **First launch** may show "Database Not Found" - this is normal if you haven't downloaded it yet
3. **Large datasets** (>10,000 spectra) automatically use optimized algorithms
4. **Backup your data** before major operations
5. **Database format is pickle** (.pkl), not SQLite

---

**Thank you for using RamanLab!** 🔬✨
