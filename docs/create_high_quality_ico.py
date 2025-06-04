#!/usr/bin/env python3
"""
High-Quality Windows Icon Creator
Creates a high-quality .ico file with multiple resolutions for crisp display on all Windows systems
"""

import os
import shutil
from PIL import Image
from pathlib import Path

def extract_largest_from_icns(icns_path):
    """Extract the largest resolution image from an .icns file"""
    print(f"📱 Extracting largest image from {icns_path}")
    
    try:
        with Image.open(icns_path) as img:
            # Find the largest resolution in the .icns file
            largest_size = 0
            largest_img = None
            
            # Iterate through all the images in the .icns file
            for i in range(100):  # Check up to 100 frames
                try:
                    img.seek(i)
                    current_size = img.size[0]  # Assuming square images
                    if current_size > largest_size:
                        largest_size = current_size
                        largest_img = img.copy()
                    print(f"  Found {img.size[0]}x{img.size[1]} resolution")
                except EOFError:
                    break
            
            if largest_img:
                print(f"✅ Extracted largest image: {largest_size}x{largest_size}")
                return largest_img
            else:
                print("❌ No images found in .icns file")
                return None
                
    except Exception as e:
        print(f"❌ Error reading .icns file: {e}")
        return None

def create_high_quality_ico(source_image, output_ico_path):
    """Create high-quality .ico file with multiple sizes"""
    
    print(f"🖼️ Creating high-quality Windows icon")
    
    # Windows icon sizes including high-DPI support
    # Standard sizes + larger sizes for modern high-DPI displays
    ico_sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
    
    print(f"📐 Target sizes: {', '.join(f'{s}x{s}' for s in ico_sizes)}")
    
    try:
        # Convert to RGBA for transparency support
        if source_image.mode != 'RGBA':
            source_image = source_image.convert('RGBA')
        
        print(f"📐 Source image: {source_image.size[0]}x{source_image.size[1]} pixels, mode: {source_image.mode}")
        
        # Create high-quality resized versions
        icon_images = []
        
        for size in ico_sizes:
            # Use LANCZOS for highest quality downsampling
            resized = source_image.resize((size, size), Image.Resampling.LANCZOS)
            
            # Apply slight sharpening for small sizes to improve clarity
            if size <= 32:
                from PIL import ImageFilter
                resized = resized.filter(ImageFilter.UnsharpMask(radius=0.5, percent=50, threshold=0))
            
            icon_images.append(resized)
            print(f"  ✅ Created {size}x{size} version")
        
        # Save as high-quality .ico file
        print(f"💾 Saving high-quality .ico file...")
        
        # Use the highest quality settings
        icon_images[0].save(
            output_ico_path,
            format='ICO',
            sizes=[(img.width, img.height) for img in icon_images],
            # Include all images in the .ico file
            append_images=icon_images[1:],
            # Use maximum quality settings
            optimize=False,  # Don't optimize to maintain quality
            quality=100      # Maximum quality
        )
        
        print(f"✅ Successfully created high-quality {output_ico_path}")
        
        # Check the file size
        if os.path.exists(output_ico_path):
            size_kb = os.path.getsize(output_ico_path) / 1024
            print(f"📊 File size: {size_kb:.1f} KB")
            
            # Show comparison with old file
            old_file = "RamanLab_icon.ico"
            if os.path.exists(old_file):
                old_size_kb = os.path.getsize(old_file) / 1024
                improvement = (size_kb / old_size_kb) if old_size_kb > 0 else 0
                print(f"📈 Size increase: {improvement:.1f}x larger (from {old_size_kb:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating .ico file: {e}")
        return False

def compare_ico_files():
    """Compare the quality metrics of different .ico files"""
    print("\n🔍 Icon Quality Comparison")
    print("=" * 30)
    
    files_to_check = [
        ("RamanLab_icon.ico", "Original"),
        ("RamanLab_icon_HQ.ico", "High Quality")
    ]
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            size_kb = os.path.getsize(filename) / 1024
            
            # Try to open and analyze the icon
            try:
                with Image.open(filename) as img:
                    # Count the number of different sizes in the .ico
                    img.load()
                    sizes = []
                    try:
                        for i in range(100):  # Check up to 100 frames
                            img.seek(i)
                            sizes.append(f"{img.size[0]}x{img.size[1]}")
                    except EOFError:
                        pass
                    
                    unique_sizes = list(set(sizes))
                    print(f"{description:12} | Size: {size_kb:6.1f} KB | Resolutions: {len(unique_sizes)} | {', '.join(sorted(unique_sizes, key=lambda x: int(x.split('x')[0])))}")
                    
            except Exception as e:
                print(f"{description:12} | Size: {size_kb:6.1f} KB | Error reading: {e}")
        else:
            print(f"{description:12} | Not found")

def main():
    """Main function"""
    icns_file = "RamanLab_icon.icns"
    output_file = "RamanLab_icon_HQ.ico"
    
    print("🚀 High-Quality Windows Icon Creator")
    print("=" * 50)
    
    # Check if we have the high-quality .icns file
    if not os.path.exists(icns_file):
        print(f"❌ Source file '{icns_file}' not found!")
        print(f"   Please make sure the high-quality .icns file exists.")
        return
    
    # Extract the largest image from the .icns file
    source_image = extract_largest_from_icns(icns_file)
    if not source_image:
        print("❌ Could not extract source image from .icns file")
        return
    
    # Create the high-quality icon
    success = create_high_quality_ico(source_image, output_file)
    
    if success:
        print(f"\n🎉 Success! High-quality Windows icon created: {output_file}")
        
        # Show comparison
        compare_ico_files()
        
        # Offer to replace the existing file
        replace = input(f"\n🔄 Replace the existing RamanLab_icon.ico? (y/n): ").lower().strip()
        if replace in ['y', 'yes']:
            try:
                if os.path.exists("RamanLab_icon.ico"):
                    shutil.copy2("RamanLab_icon.ico", "RamanLab_icon.ico.backup")
                    print("💾 Backed up existing icon to RamanLab_icon.ico.backup")
                
                shutil.copy2(output_file, "RamanLab_icon.ico")
                print("✅ Replaced RamanLab_icon.ico with high-quality version")
                
                # Clean up the temporary file
                os.remove(output_file)
                print("🧹 Cleaned up temporary file")
                
                print(f"\n📝 The new high-quality .ico file is now ready!")
                print(f"   File size improved from ~1.2 KB to much higher quality")
                print(f"   Includes {len([16, 20, 24, 32, 40, 48, 64, 96, 128, 256])} different resolutions")
                
            except Exception as e:
                print(f"❌ Error replacing file: {e}")
    else:
        print("\n❌ Failed to create high-quality Windows icon")

if __name__ == "__main__":
    main() 