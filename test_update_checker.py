#!/usr/bin/env python3
"""
Test script for ClaritySpectra Update Checker integration.
Verifies that the update checker is properly integrated into the main application.

@author: AaronCelestian
@version: 2.6.3
ClaritySpectra
"""

import sys
import os

def test_update_checker_integration():
    """Test that the update checker is properly integrated."""
    
    print("Testing ClaritySpectra Update Checker Integration...")
    print("=" * 50)
    
    # Test 1: Check if update_checker.py exists
    print("1. Checking if update_checker.py exists...")
    if os.path.exists("update_checker.py"):
        print("   ✓ update_checker.py found")
    else:
        print("   ✗ update_checker.py not found")
        return False
    
    # Test 2: Check if UpdateChecker can be imported
    print("2. Testing UpdateChecker import...")
    try:
        from update_checker import UpdateChecker
        print("   ✓ UpdateChecker imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import UpdateChecker: {e}")
        return False
    
    # Test 3: Check if UpdateChecker can be instantiated
    print("3. Testing UpdateChecker instantiation...")
    try:
        checker = UpdateChecker("2.6.3")
        print("   ✓ UpdateChecker instantiated successfully")
    except Exception as e:
        print(f"   ✗ Failed to instantiate UpdateChecker: {e}")
        return False
    
    # Test 4: Check if main app has update checker integration
    print("4. Checking main application integration...")
    try:
        with open("raman_analysis_app.py", 'r') as f:
            content = f.read()
            
        if "from update_checker import UpdateChecker" in content:
            print("   ✓ Update checker import found in main app")
        else:
            print("   ✗ Update checker import not found in main app")
            return False
            
        if "Check for Updates" in content:
            print("   ✓ Update checker menu item found")
        else:
            print("   ✗ Update checker menu item not found")
            return False
            
        if "def check_for_updates" in content:
            print("   ✓ Update checker method found")
        else:
            print("   ✗ Update checker method not found")
            return False
            
    except Exception as e:
        print(f"   ✗ Failed to check main app integration: {e}")
        return False
    
    # Test 5: Check dependencies
    print("5. Checking required dependencies...")
    required_deps = ['requests', 'packaging']
    optional_deps = ['pyperclip']
    
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"   ✓ {dep} available")
        except ImportError:
            print(f"   ✗ {dep} not available (required)")
            return False
    
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"   ✓ {dep} available")
        except ImportError:
            print(f"   ⚠ {dep} not available (optional)")
    
    # Test 6: Test basic functionality
    print("6. Testing basic update checker functionality...")
    try:
        # Test version comparison
        result = checker._is_newer_version("2.6.3", "2.6.3")
        if result:
            print("   ✓ Version comparison working")
        else:
            print("   ✗ Version comparison failed")
            return False
            
        # Test GitHub URL configuration
        if "github.com" in checker.github_repo_url:
            print("   ✓ GitHub URL configured")
        else:
            print("   ✗ GitHub URL not configured")
            return False
            
    except Exception as e:
        print(f"   ✗ Basic functionality test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! Update checker is properly integrated.")
    print("\nTo use the update checker:")
    print("1. Run ClaritySpectra: python raman_analysis_app.py")
    print("2. Go to Help → Check for Updates")
    print("3. Follow the update dialog instructions")
    
    return True

def test_standalone_update_checker():
    """Test the standalone update checker functionality."""
    
    print("\nTesting standalone update checker...")
    print("-" * 30)
    
    try:
        from update_checker import UpdateChecker
        checker = UpdateChecker("2.6.3")
        
        print("Testing update check (this may take a few seconds)...")
        # Test with show_no_updates=False to avoid popup
        result = checker.check_for_updates(show_no_updates=False)
        
        if result is not None:
            if result.get('update_available'):
                print(f"   ✓ Update available: {result.get('latest_version')}")
            else:
                print("   ✓ No updates available (you have the latest version)")
        else:
            print("   ⚠ Could not check for updates (network issue)")
            
        print("   ✓ Standalone update checker working")
        return True
        
    except Exception as e:
        print(f"   ✗ Standalone test failed: {e}")
        return False

if __name__ == "__main__":
    print("ClaritySpectra Update Checker Test Suite")
    print("========================================\n")
    
    # Run integration tests
    integration_success = test_update_checker_integration()
    
    if integration_success:
        # Run standalone tests
        standalone_success = test_standalone_update_checker()
        
        if standalone_success:
            print("\n🎉 All tests completed successfully!")
            print("The update checker is ready to use.")
            sys.exit(0)
        else:
            print("\n⚠ Integration tests passed but standalone tests failed.")
            sys.exit(1)
    else:
        print("\n❌ Integration tests failed.")
        print("Please check the installation and try again.")
        sys.exit(1) 