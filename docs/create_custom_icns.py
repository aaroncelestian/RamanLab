import os
import shutil
import subprocess
from PIL import Image

def create_custom_icon_set():
    """
    Create a complete icon set from the custom RL icon and generate .icns file
    """
    
    # Check if our custom icon exists
    if not os.path.exists("RamanLab_custom_rl.png"):
        print("❌ RamanLab_custom_rl.png not found. Run create_custom_rl_icon.py first.")
        return
    
    # Load the base image
    base_img = Image.open("RamanLab_custom_rl.png").convert("RGBA")
    print("📷 Loaded base custom RL icon")
    
    # Create multiple sizes
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    for size in sizes:
        resized = base_img.resize((size, size), Image.LANCZOS)
        filename = f"RamanLab_custom_{size}.png"
        resized.save(filename)
        print(f"✅ Created {filename}")
    
    # Create the iconset directory
    iconset_dir = "RamanLab_custom.iconset"
    if os.path.exists(iconset_dir):
        shutil.rmtree(iconset_dir)
    os.makedirs(iconset_dir)
    print(f"📁 Created {iconset_dir} directory")
    
    # Map our PNG files to the required iconset naming convention
    icon_mappings = {
        "RamanLab_custom_16.png": "icon_16x16.png",
        "RamanLab_custom_32.png": "icon_16x16@2x.png",  # 32px for 16@2x
        "RamanLab_custom_32.png": "icon_32x32.png",
        "RamanLab_custom_64.png": "icon_32x32@2x.png",  # 64px for 32@2x
        "RamanLab_custom_128.png": "icon_128x128.png",
        "RamanLab_custom_256.png": "icon_128x128@2x.png",  # 256px for 128@2x
        "RamanLab_custom_256.png": "icon_256x256.png",
        "RamanLab_custom_512.png": "icon_256x256@2x.png",  # 512px for 256@2x
        "RamanLab_custom_512.png": "icon_512x512.png",
        "RamanLab_custom_1024.png": "icon_512x512@2x.png"  # 1024px for 512@2x
    }
    
    # Copy files to iconset directory with proper names
    for source_file, target_name in icon_mappings.items():
        if os.path.exists(source_file):
            shutil.copy2(source_file, os.path.join(iconset_dir, target_name))
            print(f"📋 Copied {source_file} -> {target_name}")
        else:
            print(f"⚠️  Warning: {source_file} not found")
    
    # Use iconutil to create the .icns file
    try:
        result = subprocess.run([
            "iconutil", "-c", "icns", iconset_dir, "-o", "RamanLab_custom.icns"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully created RamanLab_custom.icns")
            # Clean up the iconset directory
            shutil.rmtree(iconset_dir)
            print("🧹 Cleaned up temporary iconset directory")
            
            # Show file info
            if os.path.exists("RamanLab_custom.icns"):
                file_size = os.path.getsize("RamanLab_custom.icns")
                print(f"📊 Icon file size: {file_size:,} bytes")
            
        else:
            print(f"❌ Error creating .icns file: {result.stderr}")
            
    except FileNotFoundError:
        print("❌ iconutil command not found. This script requires macOS.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    create_custom_icon_set() 