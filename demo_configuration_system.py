#!/usr/bin/env python3
"""
RamanLab Configuration System Demo
Demonstrates the new configurable projects folder and settings system
"""

import sys
from pathlib import Path

def demo_configuration_system():
    """Demonstrate the new configuration system."""
    print("=" * 60)
    print("🔧 RamanLab Configuration System Demo")
    print("=" * 60)
    
    try:
        # Import the configuration manager
        from core.config_manager import get_config_manager
        
        # Get the configuration instance
        config = get_config_manager()
        print("✅ Configuration system loaded successfully!")
        print()
        
        # Show current configuration
        print("📋 CURRENT CONFIGURATION:")
        print(f"   Config file location: {config.config_file}")
        print(f"   Projects folder: {config.get_projects_folder()}")
        print(f"   Auto-save folder: {config.get_auto_save_folder()}")
        print(f"   Session folder: {config.get_session_folder()}")
        print()
        
        # Show some key settings
        print("⚙️  KEY SETTINGS:")
        print(f"   Auto-save enabled: {config.get('auto_save_enabled')}")
        print(f"   Auto-save interval: {config.get('auto_save_interval')} seconds")
        print(f"   Max recent files: {config.get('max_recent_files')}")
        print(f"   Plot DPI: {config.get('plot_settings.dpi')}")
        print()
        
        # Demonstrate setting a custom projects folder
        print("🔄 DEMONSTRATION - Setting Custom Projects Folder:")
        
        # Get user input for new folder (or use default for demo)
        current_folder = config.get_projects_folder()
        print(f"   Current projects folder: {current_folder}")
        
        # For demo purposes, create a test folder
        demo_folder = Path.home() / "My_Custom_RamanLab_Projects"
        print(f"   Setting demo folder to: {demo_folder}")
        
        # Update the configuration
        config.set_projects_folder(demo_folder)
        print("   ✅ Projects folder updated!")
        
        # Verify the change
        new_folder = config.get_projects_folder()
        print(f"   New projects folder: {new_folder}")
        print(f"   Auto-save folder: {config.get_auto_save_folder()}")
        print()
        
        # Show how to access the settings from the GUI
        print("🖥️  ACCESSING SETTINGS FROM RAMANLAB:")
        print("   1. Launch RamanLab (main_qt6.py)")
        print("   2. Go to Settings → Preferences...")
        print("   3. Navigate to the 'Paths' tab")
        print("   4. Click 'Browse...' to select your preferred folder")
        print("   5. Click 'OK' to save")
        print()
        
        # Restore original folder for demo
        config.set_projects_folder(current_folder)
        print(f"   ↶ Restored original folder: {current_folder}")
        print()
        
        print("✨ BENEFITS OF THE NEW SYSTEM:")
        print("   • Choose where your RamanLab data is stored")
        print("   • Automatic folder creation and management")
        print("   • Consistent across all RamanLab modules")
        print("   • Settings persist between sessions")
        print("   • Easy backup and migration")
        print()
        
        print("📁 FOLDER STRUCTURE:")
        projects_folder = config.get_projects_folder()
        print(f"   {projects_folder}/")
        print(f"   ├── auto_saves/          # Session auto-saves")
        print(f"   ├── exports/             # Exported data")
        print(f"   ├── analysis_results/    # Analysis outputs")
        print(f"   └── custom_databases/    # User databases")
        print()
        
        print("🎯 NEXT STEPS:")
        print("   • Run 'python main_qt6.py' to launch RamanLab")
        print("   • Access Settings → Preferences to configure")
        print("   • Your settings will be automatically saved")
        print("   • All modules now respect your folder choice")
        
    except ImportError as e:
        print(f"❌ Error: Could not import configuration system: {e}")
        print("   Make sure you're running from the RamanLab directory")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("=" * 60)
    return True


def quick_setup_guide():
    """Show a quick setup guide for new users."""
    print("\n🚀 QUICK SETUP GUIDE:")
    print("1. Choose your projects folder location")
    print("2. Launch RamanLab: python main_qt6.py")
    print("3. Go to Settings → Preferences")
    print("4. Set your preferred paths and options")
    print("5. Start analyzing!")


if __name__ == "__main__":
    success = demo_configuration_system()
    
    if success:
        quick_setup_guide()
    
    print("\n" + "=" * 60)
    print("🔬 Happy analyzing with RamanLab!")
    print("=" * 60) 