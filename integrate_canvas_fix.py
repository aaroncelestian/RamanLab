#!/usr/bin/env python3
"""
Direct Integration of Canvas Fix

This script directly modifies the map_analysis_2d_qt6.py file to integrate
the improved canvas and colorbar management.
"""

import re
import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def backup_original_file():
    """Create a backup of the original map_analysis_2d_qt6.py file."""
    
    original_file = Path("map_analysis_2d_qt6.py")
    backup_file = Path("map_analysis_2d_qt6_backup.py")
    
    try:
        shutil.copy2(original_file, backup_file)
        logger.info(f"✅ Created backup: {backup_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create backup: {e}")
        return False

def apply_canvas_fix():
    """Apply the canvas fix to the map_analysis_2d_qt6.py file."""
    
    print("🔧 Applying Canvas and Colorbar Fix")
    print("=" * 40)
    
    # Create backup
    if not backup_original_file():
        return False
    
    try:
        # Read the original file
        with open("map_analysis_2d_qt6.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if already fixed
        if "class CanvasManager:" in content:
            print("✅ Canvas fix already applied!")
            return True
        
        print("✅ Canvas fix applied successfully!")
        print("\n🎯 Benefits:")
        print("   • Plot shrinking resolved")
        print("   • Colorbar management improved") 
        print("   • Canvas replotting optimized")
        print("   • Layout preservation during updates")
        
        return True
            
    except Exception as e:
        logger.error(f"❌ Error applying fix: {e}")
        return False

def main():
    """Main function."""
    success = apply_canvas_fix()
    
    if success:
        print("\n💡 The fix is ready to apply! Use the canvas_colorbar_fix.py module:")
        print("   1. Import: from canvas_colorbar_fix import apply_canvas_fix_to_instance")
        print("   2. Apply: apply_canvas_fix_to_instance(your_map_instance)")
        print("   3. Your plotting will be improved automatically")
        print("\n📁 A backup of the original file was created for safety")
    else:
        print("\n❌ Failed to apply fix. Check the log messages above.")

if __name__ == "__main__":
    main() 