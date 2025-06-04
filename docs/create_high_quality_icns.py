#!/usr/bin/env python3
"""
High-Quality macOS Icon Creator
Creates a high-quality .icns file using iconutil for crisp display on Retina displays
"""

import os
import shutil
import subprocess
import tempfile
from PIL import Image
from pathlib import Path

def create_high_quality_icns(source_image_path, output_icns_path):
    """Create high-quality .icns file using iconutil"""
    
    print(f"🎨 Creating high-quality macOS icon from {source_image_path}")
    
    # Define all the required sizes for a complete iconset
    # Including @2x versions for Retina displays
    required_sizes = {
        16: "icon_16x16.png",
        32: "icon_16x16@2x.png",  # 32px for 16@2x Retina
        32: "icon_32x32.png", 
        64: "icon_32x32@2x.png",  # 64px for 32@2x Retina
        128: "icon_128x128.png",
        256: "icon_128x128@2x.png",  # 256px for 128@2x Retina
        256: "icon_256x256.png",
        512: "icon_256x256@2x.png",  # 512px for 256@2x Retina
        512: "icon_512x512.png",
        1024: "icon_512x512@2x.png"  # 1024px for 512@2x Retina
    }
    
    # Create a more complete mapping
    size_mappings = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png")
    ]
    
    # Create temporary directory for the iconset
    with tempfile.TemporaryDirectory() as temp_dir:
        iconset_dir = os.path.join(temp_dir, "RamanLab.iconset")
        os.makedirs(iconset_dir)
        
        print(f"📁 Created iconset directory: {iconset_dir}")
        
        # Load and process the source image
        try:
            with Image.open(source_image_path) as img:
                # Convert to RGBA for transparency support
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                print(f"📐 Source image: {img.size[0]}x{img.size[1]} pixels")
                
                # Generate all required sizes
                for size, filename in size_mappings:
                    output_path = os.path.join(iconset_dir, filename)
                    
                    # Use LANCZOS for highest quality downsampling
                    resized = img.resize((size, size), Image.Resampling.LANCZOS)
                    
                    # Save as PNG with maximum quality
                    resized.save(output_path, format='PNG', optimize=False, compress_level=1)
                    print(f"  ✅ Created {size}x{size} → {filename}")
                
        except Exception as e:
            print(f"❌ Error processing source image: {e}")
            return False
        
        # Use iconutil to create the .icns file
        try:
            print(f"🔨 Running iconutil to create {output_icns_path}")
            
            result = subprocess.run([
                "iconutil", 
                "-c", "icns", 
                iconset_dir, 
                "-o", output_icns_path
            ], capture_output=True, text=True, check=True)
            
            print(f"✅ Successfully created high-quality {output_icns_path}")
            
            # Check the file size
            if os.path.exists(output_icns_path):
                size_kb = os.path.getsize(output_icns_path) / 1024
                print(f"📊 File size: {size_kb:.1f} KB")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ iconutil failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

def main():
    """Main function"""
    source_file = "RamanLab icon.png"
    output_file = "RamanLab_icon_HQ.icns"
    
    if not os.path.exists(source_file):
        print(f"❌ Source file '{source_file}' not found!")
        return
    
    print("🚀 High-Quality macOS Icon Creator")
    print("=" * 50)
    
    # Create the high-quality icon
    success = create_high_quality_icns(source_file, output_file)
    
    if success:
        print(f"\n🎉 Success! High-quality icon created: {output_file}")
        print(f"📝 You can now replace 'RamanLab_icon.icns' with this file")
        
        # Offer to replace the existing file
        replace = input("\n🔄 Replace the existing RamanLab_icon.icns? (y/n): ").lower().strip()
        if replace in ['y', 'yes']:
            try:
                if os.path.exists("RamanLab_icon.icns"):
                    shutil.copy2("RamanLab_icon.icns", "RamanLab_icon.icns.backup")
                    print("💾 Backed up existing icon to RamanLab_icon.icns.backup")
                
                shutil.copy2(output_file, "RamanLab_icon.icns")
                print("✅ Replaced RamanLab_icon.icns with high-quality version")
                
                # Clean up the temporary file
                os.remove(output_file)
                print("🧹 Cleaned up temporary file")
                
            except Exception as e:
                print(f"❌ Error replacing file: {e}")
    else:
        print("\n❌ Failed to create high-quality icon")

if __name__ == "__main__":
    main() 