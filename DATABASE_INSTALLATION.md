# RamanLab Database Installation Guide

## 📦 Required Database File

**File Name**: `RamanLab_Database_20250602.pkl`  
**Size**: ~200MB  
**Contains**: 6,939+ experimental Raman spectra  
**Format**: Python pickle (.pkl)

---

## 🚨 Important: Database Not Included in Repository

Due to the large file size (200MB+), the database is **not included** in the Git repository. You must download it separately.

---

## 📥 Download Instructions

### Step 1: Download the Database

**Download Link**: https://zenodo.org/records/15742626

The database is hosted on Zenodo, a trusted research data repository that provides:
- ✅ Permanent, citable DOI
- ✅ Free, open access
- ✅ Reliable long-term storage
- ✅ Fast download speeds

### Step 2: Verify the Download

After downloading, verify:
- ✅ File name: `RamanLab_Database_20250602.pkl`
- ✅ File size: ~200MB
- ✅ File extension: `.pkl` (not `.sqlite` or other)

---

## 📂 Installation Locations

Choose **ONE** of the following locations:

### Option 1: User Documents Folder (Recommended)

This is the preferred location as it keeps user data separate from application files.

#### Windows
```
C:\Users\<YourUsername>\Documents\RamanLab_Qt6\RamanLab_Database_20250602.pkl
```

**Steps:**
1. Open File Explorer
2. Navigate to `Documents` folder
3. Create a new folder named `RamanLab_Qt6` (if it doesn't exist)
4. Copy `RamanLab_Database_20250602.pkl` into this folder

#### macOS
```
/Users/<YourUsername>/Documents/RamanLab_Qt6/RamanLab_Database_20250602.pkl
```

**Steps:**
1. Open Finder
2. Go to your `Documents` folder
3. Create a new folder named `RamanLab_Qt6` (if it doesn't exist)
4. Copy `RamanLab_Database_20250602.pkl` into this folder

#### Linux
```
/home/<YourUsername>/Documents/RamanLab_Qt6/RamanLab_Database_20250602.pkl
```

**Steps:**
1. Open your file manager
2. Navigate to `Documents` folder
3. Create a new folder named `RamanLab_Qt6` (if it doesn't exist)
4. Copy `RamanLab_Database_20250602.pkl` into this folder

---

### Option 2: Application Directory

Place the database file in the same folder where the RamanLab Python scripts are located.

**Example:**
```
RamanLab/
├── raman_analysis_app_qt6.py
├── raman_spectra_qt6.py
├── RamanLab_Database_20250602.pkl  ← Place here
└── ...
```

**Steps:**
1. Navigate to your RamanLab installation folder
2. Copy `RamanLab_Database_20250602.pkl` into this folder
3. The file should be in the same directory as `raman_analysis_app_qt6.py`

---

## ✅ Verification

### Check if Database is Loaded

1. Launch RamanLab:
   ```bash
   python raman_analysis_app_qt6.py
   ```

2. Check the console output for:
   ```
   ✓ Database loaded successfully from: <path>
     Database contains 6939 entries
   ```

3. In the application:
   - Go to **Database** tab
   - You should see database statistics showing 6,939+ spectra
   - Search functionality should work

### If Database is NOT Found

You will see:
- Console message: `⚠ Database file not found at: <path>`
- Warning dialog: "Database Not Found"
- Database tab shows 0 spectra

**Solution**: Double-check the file location and name match exactly.

---

## 🔍 Troubleshooting

### Problem: "Database Not Found" Warning

**Symptoms:**
- Warning dialog appears on startup
- Database tab shows 0 spectra
- Search returns no results

**Solutions:**

1. **Check file location:**
   - Verify the file is in one of the two locations above
   - File must be named exactly: `RamanLab_Database_20250602.pkl`

2. **Check file permissions:**
   - Ensure you have read permissions for the file
   - On Linux/Mac: `chmod 644 RamanLab_Database_20250602.pkl`

3. **Verify file integrity:**
   - Re-download if file size doesn't match (~200MB)
   - Check file isn't corrupted

4. **Check Documents folder:**
   - Windows: Make sure `Documents` folder exists in your user profile
   - macOS/Linux: Verify `~/Documents` is accessible

### Problem: "Not a valid RamanLab database file"

**Cause:** File is corrupted or wrong format

**Solutions:**
1. Re-download the database file
2. Ensure file extension is `.pkl` (not `.sqlite`, `.txt`, etc.)
3. Don't rename the file
4. Don't edit or modify the file

### Problem: Import Fails with Pickle Error

**Cause:** Python version mismatch or corrupted file

**Solutions:**
1. Ensure you're using Python 3.10 or higher
2. Re-download the database file
3. Check console for specific error message

---

## 📊 Database Contents

Once loaded, the database contains:

- **6,939+ spectra** from various minerals and materials
- **Wavenumber range**: Typically 100-4000 cm⁻¹
- **Metadata**: Chemical family, mineral name, collection info
- **Peak data**: Pre-identified peak positions
- **Search capability**: Find similar spectra by correlation

---

## 🔄 Updating the Database

If a new database version is released:

1. Download the new database file
2. **Backup your current database** (if you've added custom spectra)
3. Replace the old file with the new one
4. Keep the same filename: `RamanLab_Database_20250602.pkl`
5. Restart RamanLab

---

## 💾 Backup Recommendations

**Important**: If you add your own spectra to the database:

1. **Export your additions:**
   - Database Browser → Export Database
   - Save as a separate `.pkl` file

2. **Regular backups:**
   - Copy `RamanLab_Database_20250602.pkl` to a backup location
   - Recommended: Weekly backups if actively adding data

3. **Before updates:**
   - Always backup before replacing with a new database version

---

## 🆘 Still Having Issues?

If you continue to have problems:

1. **Check console output** for detailed error messages
2. **Verify Python version**: `python --version` (should be 3.10+)
3. **Check dependencies**: `pip install -r requirements_qt6.txt`
4. **Contact support** with:
   - Operating system and version
   - Python version
   - Console error messages
   - Screenshot of the warning dialog

---

## 📧 Support

For database download links or installation help:
- **Support Forum**: https://ramanlab.freeforums.net/#category-3
- **GitHub Issues**: https://github.com/aaroncelestian/RamanLab/issues
- **Email**: aaron.celestian@gmail.com
- **Repository**: https://github.com/aaroncelestian/RamanLab
- **Database**: https://zenodo.org/records/15742626

---

## ✨ Quick Reference

| Platform | Recommended Location |
|----------|---------------------|
| **Windows** | `C:\Users\<You>\Documents\RamanLab_Qt6\RamanLab_Database_20250602.pkl` |
| **macOS** | `~/Documents/RamanLab_Qt6/RamanLab_Database_20250602.pkl` |
| **Linux** | `~/Documents/RamanLab_Qt6/RamanLab_Database_20250602.pkl` |
| **Alternative** | Same folder as `raman_analysis_app_qt6.py` |

**File Size**: ~200MB  
**Format**: Python pickle (.pkl)  
**Required**: Yes (for full functionality)

---

**Happy analyzing!** 🔬
