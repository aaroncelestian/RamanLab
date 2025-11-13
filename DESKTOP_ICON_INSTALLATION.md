# RamanLab Desktop Icon Installation Guide

## 📱 Quick Start

Install a desktop shortcut or application icon for easy access to RamanLab:

```bash
python install_desktop_icon.py
```

This cross-platform installer will automatically detect your operating system and create the appropriate shortcut.

---

## 🖥️ Platform-Specific Details

### Windows

**What Gets Created:**
- Desktop shortcut (`.lnk` file) with RamanLab icon
- Double-click to launch RamanLab

**Location:**
```
C:\Users\<YourUsername>\Desktop\RamanLab.lnk
```

**Features:**
- Uses high-quality icon (`RamanLab_icon_HQ.ico`)
- Sets working directory automatically
- Launches with your Python environment

**Requirements:**
- Python installed
- Optional: `pywin32` and `winshell` packages for advanced shortcuts
  ```bash
  pip install pywin32 winshell
  ```
- Without these packages, a basic `.bat` file will be created instead

---

### macOS

**What Gets Created:**
- Full application bundle (`.app`) in your Applications folder
- Appears in Launchpad and Spotlight
- Can be added to Dock

**Location:**
```
~/Applications/RamanLab.app
```

**Features:**
- Native macOS application bundle
- High-resolution icon (`RamanLab_icon.icns`)
- Conda environment support
- Retina display compatible

**Usage:**
1. After installation, find "RamanLab" in Launchpad
2. Or navigate to `~/Applications/RamanLab.app`
3. Drag to Dock for quick access
4. Double-click to launch

**Note:** The installer will attempt to add RamanLab to your Dock automatically.

---

### Linux

**What Gets Created:**
- Desktop entry file (`.desktop`) in applications menu
- Desktop shortcut (if Desktop folder exists)
- PNG icon converted from ICO

**Locations:**
```
~/.local/share/applications/ramanlab.desktop
~/Desktop/RamanLab.desktop (if Desktop exists)
```

**Features:**
- Appears in application launcher (GNOME, KDE, etc.)
- Desktop shortcut for quick access
- Categorized under Science → Education → Physics

**Requirements:**
- Optional: `pillow` or ImageMagick for icon conversion
  ```bash
  pip install pillow
  # OR
  sudo apt-get install imagemagick  # Debian/Ubuntu
  ```

**Usage:**
1. Search for "RamanLab" in your application launcher
2. Or double-click desktop shortcut
3. Pin to favorites/taskbar as desired

---

## 🔧 Advanced Installation

### Custom Installation Locations

The installer automatically uses:
- **Windows**: Desktop folder
- **macOS**: `~/Applications` (user applications)
- **Linux**: `~/.local/share/applications` (user applications)

### Icon Files

RamanLab includes multiple icon formats:
- `RamanLab_icon_HQ.ico` - High-quality Windows icon
- `RamanLab_icon.icns` - macOS icon bundle
- `RamanLab_icon_*.ico` - Various sizes (32x32, 64x64, 128x128, 256x256)

The installer automatically selects the appropriate icon for your platform.

### Launcher Scripts

The installer looks for these launcher scripts (in order):
1. `launch_ramanlab_fast.py` (optimized launcher)
2. `launch_ramanlab.py` (standard launcher)
3. Falls back to `raman_analysis_app_qt6.py`

---

## 🗑️ Uninstallation

Remove the desktop icon/shortcut:

```bash
python install_desktop_icon.py --uninstall
```

This will remove:
- **Windows**: Desktop shortcut files
- **macOS**: Application bundle from ~/Applications
- **Linux**: Desktop entry and desktop shortcut

---

## 🐛 Troubleshooting

### Windows Issues

**Problem**: "No module named 'winshell'"

**Solution**: Install optional Windows packages:
```bash
pip install pywin32 winshell
```

Or use the basic `.bat` file that gets created automatically.

---

**Problem**: Icon doesn't appear on shortcut

**Solution**: 
1. Right-click the shortcut
2. Select "Properties"
3. Click "Change Icon"
4. Browse to RamanLab folder and select `RamanLab_icon_HQ.ico`

---

### macOS Issues

**Problem**: "RamanLab.app" can't be opened because it is from an unidentified developer

**Solution**:
1. Right-click (or Control-click) on RamanLab.app
2. Select "Open" from the menu
3. Click "Open" in the dialog
4. macOS will remember this choice

Or disable Gatekeeper for this app:
```bash
xattr -cr ~/Applications/RamanLab.app
```

---

**Problem**: Application doesn't launch from Dock

**Solution**: The app bundle includes conda environment setup. If issues persist:
1. Open Terminal
2. Navigate to RamanLab directory
3. Run: `python raman_analysis_app_qt6.py`
4. Check console for error messages

---

### Linux Issues

**Problem**: Desktop entry doesn't appear in application menu

**Solution**: Update desktop database manually:
```bash
update-desktop-database ~/.local/share/applications
```

Or log out and log back in.

---

**Problem**: Icon doesn't display

**Solution**: Install Pillow for icon conversion:
```bash
pip install pillow
python install_desktop_icon.py
```

Or manually convert icon:
```bash
convert RamanLab_icon_HQ.ico RamanLab_icon.png
```

---

**Problem**: Desktop shortcut not executable

**Solution**: Make it executable:
```bash
chmod +x ~/Desktop/RamanLab.desktop
```

---

## 📋 Manual Installation

If the automatic installer doesn't work, you can create shortcuts manually:

### Windows (Manual)

Create a batch file `RamanLab.bat` on your desktop:

```batch
@echo off
cd /d "C:\Path\To\RamanLab"
python raman_analysis_app_qt6.py
pause
```

Replace `C:\Path\To\RamanLab` with your actual path.

---

### macOS (Manual)

Create a shell script `RamanLab.command` on your desktop:

```bash
#!/bin/bash
cd "/Path/To/RamanLab"
python raman_analysis_app_qt6.py
```

Make it executable:
```bash
chmod +x ~/Desktop/RamanLab.command
```

---

### Linux (Manual)

Create `~/.local/share/applications/ramanlab.desktop`:

```ini
[Desktop Entry]
Version=1.0
Type=Application
Name=RamanLab
Comment=Raman Spectroscopy Analysis Suite
Exec=python /path/to/RamanLab/raman_analysis_app_qt6.py
Path=/path/to/RamanLab
Terminal=false
Categories=Science;Education;Physics;
```

Replace `/path/to/RamanLab` with your actual path.

---

## ✨ Features

### Windows Shortcut Features
- ✅ Custom icon
- ✅ Working directory set
- ✅ Description/tooltip
- ✅ Double-click to launch

### macOS App Bundle Features
- ✅ Native .app bundle
- ✅ Launchpad integration
- ✅ Spotlight searchable
- ✅ Dock support
- ✅ Retina display icon
- ✅ Conda environment setup

### Linux Desktop Entry Features
- ✅ Application menu integration
- ✅ Desktop shortcut
- ✅ Categorized (Science)
- ✅ Custom icon
- ✅ Startup notification

---

## 🔄 Updating

If you move the RamanLab folder or update Python:

1. **Uninstall old shortcut:**
   ```bash
   python install_desktop_icon.py --uninstall
   ```

2. **Reinstall with new location:**
   ```bash
   python install_desktop_icon.py
   ```

---

## 📧 Support

If you encounter issues with desktop icon installation:

- **Support Forum**: https://ramanlab.freeforums.net/#category-3
- **GitHub Issues**: https://github.com/aaroncelestian/RamanLab/issues
- **Email**: aaron.celestian@gmail.com

---

## 📝 Technical Details

### Installer Script

The `install_desktop_icon.py` script:
- Detects operating system automatically
- Locates Python executable
- Finds RamanLab directory
- Creates platform-specific shortcuts
- Sets appropriate permissions
- Handles icon files

### Icon Formats

- **Windows**: `.ico` format (multi-resolution)
- **macOS**: `.icns` format (Apple icon bundle)
- **Linux**: `.png` format (converted from .ico)

### Launcher Priority

1. `launch_ramanlab_fast.py` - Optimized startup
2. `launch_ramanlab.py` - Standard launcher
3. `raman_analysis_app_qt6.py` - Main application

---

**Enjoy quick access to RamanLab!** 🚀
