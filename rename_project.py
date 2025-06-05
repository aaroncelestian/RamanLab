#!/usr/bin/env python3
import os
import re
from pathlib import Path

def should_process_file(file_path):
    """Check if a file should be processed based on its extension and name."""
    # Skip binary files and certain directories
    skip_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib', '.exe', '.bin', '.dat', '.pkl'}
    skip_dirs = {'.git', '__pycache__', 'venv', 'env', 'node_modules', 'build', 'dist'}
    
    # Get file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    # Skip if extension is in skip list
    if ext in skip_extensions:
        return False
    
    # Skip if file is in a directory we want to ignore
    for skip_dir in skip_dirs:
        if skip_dir in file_path.split(os.sep):
            return False
    
    return True

def process_file(file_path):
    """Process a single file to replace ClaritySpectra with RamanLab."""
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file contains the old name
        if 'ClaritySpectra' not in content:
            return False
        
        # Replace all occurrences
        new_content = content.replace('ClaritySpectra', 'RamanLab')
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Track statistics
    total_files = 0
    modified_files = 0
    errors = 0
    
    print("Starting project rename from ClaritySpectra to RamanLab...")
    
    # Walk through all files in the project
    for root, dirs, files in os.walk(project_root):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip the rename script itself
            if file == 'rename_project.py':
                continue
            
            if should_process_file(file_path):
                total_files += 1
                try:
                    if process_file(file_path):
                        modified_files += 1
                        print(f"Updated: {file_path}")
                except Exception as e:
                    errors += 1
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Print summary
    print("\nRename operation completed!")
    print(f"Total files processed: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Errors encountered: {errors}")

if __name__ == "__main__":
    main() 