import os
import shutil
import subprocess

def create_icns_from_pngs():
    """
    Create a .icns file from the macOS-style PNG icons using iconutil
    """
    
    # Create the iconset directory
    iconset_dir = "RamanLab_mac.iconset"
    if os.path.exists(iconset_dir):
        shutil.rmtree(iconset_dir)
    os.makedirs(iconset_dir)
    
    # Map our PNG files to the required iconset naming convention
    icon_mappings = {
        "RamanLab_mac_16.png": "icon_16x16.png",
        "RamanLab_mac_32.png": "icon_16x16@2x.png",  # 32px for 16@2x
        "RamanLab_mac_32.png": "icon_32x32.png",
        "RamanLab_mac_64.png": "icon_32x32@2x.png",  # 64px for 32@2x
        "RamanLab_mac_128.png": "icon_128x128.png",
        "RamanLab_mac_256.png": "icon_128x128@2x.png",  # 256px for 128@2x
        "RamanLab_mac_256.png": "icon_256x256.png",
        "RamanLab_mac_512.png": "icon_256x256@2x.png",  # 512px for 256@2x
        "RamanLab_mac_512.png": "icon_512x512.png",
        "RamanLab_mac_1024.png": "icon_512x512@2x.png"  # 1024px for 512@2x
    }
    
    # Copy files to iconset directory with proper names
    for source_file, target_name in icon_mappings.items():
        if os.path.exists(source_file):
            shutil.copy2(source_file, os.path.join(iconset_dir, target_name))
            print(f"Copied {source_file} -> {target_name}")
        else:
            print(f"Warning: {source_file} not found")
    
    # Use iconutil to create the .icns file
    try:
        result = subprocess.run([
            "iconutil", "-c", "icns", iconset_dir, "-o", "RamanLab_mac.icns"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully created RamanLab_mac.icns")
            # Clean up the iconset directory
            shutil.rmtree(iconset_dir)
            print("🧹 Cleaned up temporary iconset directory")
        else:
            print(f"❌ Error creating .icns file: {result.stderr}")
            
    except FileNotFoundError:
        print("❌ iconutil command not found. This script requires macOS.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    create_icns_from_pngs() 