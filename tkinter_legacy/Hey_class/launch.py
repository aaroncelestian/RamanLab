#!/usr/bin/env python3
"""
🚀 Hey Classification System Launcher
Simple launcher for all Hey Classification tools
"""

import os
import sys

def main():
    print("🔬 Hey Classification System Launcher")
    print("=" * 40)
    print()
    print("Choose what you want to do:")
    print()
    print("1. 🧪 Quick Examples (see how it works)")
    print("2. 🔍 Interactive Classifier (classify minerals)")
    print("3. 📊 Batch Process CSV file")
    print("4. 📈 Create Validation Report")
    print("5. 🧬 Update Element Extraction")
    print("6. ❓ Help & Documentation")
    print("7. 🚪 Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                print("\n🧪 Running Quick Examples...")
                os.system("python quick_start.py")
                break
                
            elif choice == '2':
                print("\n🔍 Starting Interactive Classifier...")
                os.system("echo '1' | python example_usage.py")
                break
                
            elif choice == '3':
                print("\n📊 Starting Batch Processor...")
                os.system("echo '2' | python example_usage.py")
                break
                
            elif choice == '4':
                print("\n📈 Creating Validation Report...")
                os.system("echo '4' | python example_usage.py")
                break
                
            elif choice == '5':
                print("\n🧬 Updating Element Extraction...")
                os.system("echo '3' | python example_usage.py")
                break
                
            elif choice == '6':
                show_help()
                break
                
            elif choice == '7':
                print("\n👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

def show_help():
    """Show help and documentation."""
    print("\n📚 Hey Classification System Help")
    print("=" * 35)
    print()
    print("🎯 WHAT IT DOES:")
    print("   Automatically classifies minerals according to Hey's")
    print("   classification system based on chemical formulas.")
    print()
    print("📁 FILES:")
    print("   • improved_hey_classification.py - Main classifier")
    print("   • improved_element_extraction.py - Element parser")
    print("   • example_usage.py - Interactive interface")
    print("   • quick_start.py - Quick examples")
    print()
    print("🔧 USAGE:")
    print("   1. Quick Examples: See how it works with sample minerals")
    print("   2. Interactive: Classify individual minerals manually")
    print("   3. Batch Process: Process entire CSV files")
    print("   4. Validation: Compare with existing classifications")
    print("   5. Element Update: Fix element extraction in CSV")
    print()
    print("📊 ACCURACY:")
    print("   Current system achieves 62.5% accuracy on RRUFF data")
    print("   (3,748 out of 5,997 minerals correctly classified)")
    print()
    print("💡 TIPS:")
    print("   • Use 'Quick Examples' first to understand the system")
    print("   • For CSV files, ensure they have 'RRUFF Chemistry (concise)'")
    print("     and 'Chemistry Elements' columns")
    print("   • Complex formulas are automatically parsed for elements")
    print()
    print("🆘 NEED HELP?")
    print("   Run 'python quick_start.py' for working examples")
    print("   Check the validation reports for accuracy insights")

if __name__ == "__main__":
    main() 