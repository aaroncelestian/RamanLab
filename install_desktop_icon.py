#!/usr/bin/env python3
"""
RamanLab Desktop Icon Installer
==================================

Cross-platform installer for RamanLab desktop shortcuts.
Supports Windows, macOS, and Linux.

Usage:
    python install_desktop_icon.py

Author: Aaron Celestian
Version: 1.0.0
License: MIT
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

class DesktopIconInstaller:
    """Handles desktop icon installation across different platforms."""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.script_dir = Path(__file__).parent.absolute()
        self.app_name = "RamanLab"
        self.app_description = "Raman Spectroscopy Analysis Suite"
        self.main_script = self.script_dir / "launch_ramanlab_fast.py"
        self.fallback_script = self.script_dir / "launch_ramanlab.py"
        
        # Icon files
        self.icon_ico = self.script_dir / "RamanLab_icon_HQ.ico"  # Windows
        self.icon_icns = self.script_dir / "RamanLab_icon.icns"  # macOS
        self.icon_png = None  # We'll create this for Linux
        
    def install(self):
        """Install desktop icon based on the current platform."""
        print(f"🚀 Installing {self.app_name} desktop icon...")
        print(f"📍 Platform detected: {self.platform.title()}")
        print(f"📂 Application directory: {self.script_dir}")
        
        # Verify main script exists (try fast launcher first, fallback to regular)
        if not self.main_script.exists():
            if self.fallback_script.exists():
                print(f"⚠️  Fast launcher not found, using regular launcher: {self.fallback_script}")
                self.main_script = self.fallback_script
            else:
                print(f"❌ Error: Main script not found at {self.main_script}")
                print("💡 Make sure you're running this script from the RamanLab directory.")
                return False
            
        try:
            if self.platform == "windows":
                return self._install_windows()
            elif self.platform == "darwin":  # macOS
                return self._install_macos()
            elif self.platform == "linux":
                return self._install_linux()
            else:
                print(f"❌ Unsupported platform: {self.platform}")
                return False
                
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    def _install_windows(self):
        """Install desktop shortcut on Windows."""
        try:
            import winshell
            from win32com.client import Dispatch
            print("✅ Windows COM libraries available")
        except ImportError:
            print("⚠️  Windows COM libraries not available. Installing basic shortcut...")
            return self._install_windows_basic()
            
        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, f"{self.app_name}.lnk")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{self.main_script}"'
        shortcut.WorkingDirectory = str(self.script_dir)
        shortcut.Description = self.app_description
        
        if self.icon_ico.exists():
            shortcut.IconLocation = str(self.icon_ico)
            
        shortcut.save()
        
        print(f"✅ Windows desktop shortcut created: {shortcut_path}")
        return True
    
    def _install_windows_basic(self):
        """Install basic Windows shortcut without COM libraries."""
        desktop = Path.home() / "Desktop"
        shortcut_path = desktop / f"{self.app_name}.bat"
        
        batch_content = f'''@echo off
cd /d "{self.script_dir}"
"{sys.executable}" "{self.main_script}"
pause
'''
        
        with open(shortcut_path, 'w') as f:
            f.write(batch_content)
            
        print(f"✅ Windows batch file created: {shortcut_path}")
        print("💡 To add an icon, right-click the .bat file → Properties → Change Icon")
        return True
    
    def _install_macos(self):
        """Install application bundle on macOS."""
        applications_dir = Path.home() / "Applications"
        app_bundle_dir = applications_dir / f"{self.app_name}.app"
        contents_dir = app_bundle_dir / "Contents"
        macos_dir = contents_dir / "MacOS"
        resources_dir = contents_dir / "Resources"
        
        # Create app bundle structure
        for directory in [app_bundle_dir, contents_dir, macos_dir, resources_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create Info.plist
        info_plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>{self.app_name}</string>
    <key>CFBundleIdentifier</key>
    <string>com.celestian.ramanlab</string>
    <key>CFBundleName</key>
    <string>{self.app_name}</string>
    <key>CFBundleDisplayName</key>
    <string>{self.app_name}</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>RMNL</string>
    <key>CFBundleIconFile</key>
    <string>RamanLab_icon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.12</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsArbitraryLoads</key>
        <true/>
    </dict>
</dict>
</plist>'''
        
        with open(contents_dir / "Info.plist", 'w') as f:
            f.write(info_plist)
        
        # Create executable script with conda environment setup
        executable_script = f'''#!/bin/bash
cd "{self.script_dir}"

# Set up the conda environment PATH (fixes desktop launch issues)
export PATH="/opt/anaconda3/bin:/opt/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# Initialize conda if needed
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate base
fi

# Launch RamanLab with full conda environment
"{sys.executable}" "{self.main_script}"
'''
        
        executable_path = macos_dir / self.app_name
        with open(executable_path, 'w') as f:
            f.write(executable_script)
        
        # Make executable
        os.chmod(executable_path, 0o755)
        
        # Copy icon if available
        if self.icon_icns.exists():
            shutil.copy2(self.icon_icns, resources_dir / "RamanLab_icon.icns")
        
        print(f"✅ macOS app bundle created: {app_bundle_dir}")
        
        # Try to add to Dock (optional)
        try:
            subprocess.run([
                "defaults", "write", "com.apple.dock", "persistent-apps", "-array-add",
                f"<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>{app_bundle_dir}</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>"
            ], check=True)
            subprocess.run(["killall", "Dock"], check=True)
            print("✅ Added to Dock")
        except subprocess.CalledProcessError:
            print("💡 App bundle created. You can manually drag it to your Dock if desired.")
        
        return True
    
    def _install_linux(self):
        """Install desktop entry on Linux."""
        # Create PNG icon for Linux (convert from ICO if needed)
        self._create_png_icon()
        
        desktop_file_content = f'''[Desktop Entry]
Version=1.0
Type=Application
Name={self.app_name}
Comment={self.app_description}
Exec={sys.executable} "{self.main_script}"
Icon={self.icon_png or "python"}
Path={self.script_dir}
Terminal=false
Categories=Science;Education;Physics;
StartupNotify=true
StartupWMClass={self.app_name}
'''
        
        # Install to user's applications directory
        local_apps_dir = Path.home() / ".local" / "share" / "applications"
        local_apps_dir.mkdir(parents=True, exist_ok=True)
        
        desktop_file_path = local_apps_dir / f"{self.app_name.lower()}.desktop"
        
        with open(desktop_file_path, 'w') as f:
            f.write(desktop_file_content)
        
        # Make executable
        os.chmod(desktop_file_path, 0o755)
        
        # Also create on Desktop if exists
        desktop_dir = Path.home() / "Desktop"
        if desktop_dir.exists():
            desktop_shortcut = desktop_dir / f"{self.app_name}.desktop"
            shutil.copy2(desktop_file_path, desktop_shortcut)
            os.chmod(desktop_shortcut, 0o755)
            print(f"✅ Desktop shortcut created: {desktop_shortcut}")
        
        # Update desktop database
        try:
            subprocess.run(["update-desktop-database", str(local_apps_dir)], 
                         check=True, capture_output=True)
            print("✅ Desktop database updated")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("💡 Desktop database update skipped (update-desktop-database not found)")
        
        print(f"✅ Linux desktop entry created: {desktop_file_path}")
        return True
    
    def _create_png_icon(self):
        """Create PNG icon for Linux from ICO file."""
        if not self.icon_ico.exists():
            print("⚠️  ICO icon file not found, using default Python icon")
            return
            
        png_path = self.script_dir / "RamanLab_icon.png"
        
        try:
            from PIL import Image
            
            # Convert ICO to PNG
            with Image.open(self.icon_ico) as img:
                # Get the largest size
                img_sizes = img.info.get('sizes', [(32, 32)])
                largest_size = max(img_sizes, key=lambda x: x[0] * x[1])
                img.size = largest_size
                img.save(png_path, "PNG")
            
            self.icon_png = png_path
            print(f"✅ PNG icon created: {png_path}")
            
        except ImportError:
            print("⚠️  PIL not available, trying ImageMagick...")
            try:
                subprocess.run([
                    "convert", str(self.icon_ico), str(png_path)
                ], check=True, capture_output=True)
                self.icon_png = png_path
                print(f"✅ PNG icon created with ImageMagick: {png_path}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️  Could not create PNG icon. Using default.")
    
    def uninstall(self):
        """Remove desktop icons/shortcuts."""
        print(f"🗑️  Removing {self.app_name} desktop shortcuts...")
        
        removed = False
        
        if self.platform == "windows":
            desktop = Path.home() / "Desktop"
            for pattern in [f"{self.app_name}.lnk", f"{self.app_name}.bat"]:
                shortcut_path = desktop / pattern
                if shortcut_path.exists():
                    shortcut_path.unlink()
                    print(f"✅ Removed: {shortcut_path}")
                    removed = True
                    
        elif self.platform == "darwin":
            app_bundle = Path.home() / "Applications" / f"{self.app_name}.app"
            if app_bundle.exists():
                shutil.rmtree(app_bundle)
                print(f"✅ Removed: {app_bundle}")
                removed = True
                
        elif self.platform == "linux":
            # Remove from applications
            local_apps_dir = Path.home() / ".local" / "share" / "applications"
            desktop_file = local_apps_dir / f"{self.app_name.lower()}.desktop"
            if desktop_file.exists():
                desktop_file.unlink()
                print(f"✅ Removed: {desktop_file}")
                removed = True
            
            # Remove from desktop
            desktop_dir = Path.home() / "Desktop"
            desktop_shortcut = desktop_dir / f"{self.app_name}.desktop"
            if desktop_shortcut.exists():
                desktop_shortcut.unlink()
                print(f"✅ Removed: {desktop_shortcut}")
                removed = True
            
            # Remove PNG icon
            png_icon = self.script_dir / "RamanLab_icon.png"
            if png_icon.exists():
                png_icon.unlink()
                print(f"✅ Removed: {png_icon}")
        
        if not removed:
            print("ℹ️  No shortcuts found to remove.")
        else:
            print("✅ Uninstall complete!")


def main():
    """Main installer function."""
    print("=" * 60)
    print("🔬 RamanLab Desktop Icon Installer v1.0.0")
    print("=" * 60)
    
    installer = DesktopIconInstaller()
    
    if len(sys.argv) > 1 and sys.argv[1] in ["--uninstall", "-u", "uninstall"]:
        installer.uninstall()
    else:
        success = installer.install()
        if success:
            print("\n🎉 Installation successful!")
            print(f"🚀 You can now launch {installer.app_name} from your desktop or applications menu.")
            print("\n💡 To uninstall, run: python install_desktop_icon.py --uninstall")
        else:
            print("\n❌ Installation failed!")
            sys.exit(1)


if __name__ == "__main__":
    main() 