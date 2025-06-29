#!/usr/bin/env python3
"""
RamanLab Duplicate Code Detection Tool
Finds and reports duplicate classes, methods, and code blocks
"""

import os
import re
import hashlib
from collections import defaultdict
from pathlib import Path

class DuplicateDetector:
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.duplicates_found = False
        
    def find_duplicate_classes(self):
        """Find duplicate class definitions across files"""
        print("🔍 Checking for duplicate class definitions...")
        
        classes = defaultdict(list)
        class_pattern = re.compile(r'^class\s+(\w+)\s*\(.*\):')
        
        for py_file in self.root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        match = class_pattern.match(line.strip())
                        if match:
                            class_name = match.group(1)
                            classes[class_name].append((py_file, line_num))
            except Exception as e:
                print(f"⚠️  Error reading {py_file}: {e}")
        
        # Report duplicates
        for class_name, locations in classes.items():
            if len(locations) > 1:
                self.duplicates_found = True
                print(f"❌ DUPLICATE CLASS: {class_name}")
                for file_path, line_num in locations:
                    print(f"   📁 {file_path}:{line_num}")
                print()
    
    def find_duplicate_methods_in_file(self, file_path):
        """Find duplicate method definitions within a single file"""
        methods = defaultdict(list)
        method_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    match = method_pattern.match(line)
                    if match:
                        method_name = match.group(1)
                        methods[method_name].append(line_num)
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            return False
        
        has_duplicates = False
        for method_name, line_nums in methods.items():
            if len(line_nums) > 1:
                has_duplicates = True
                self.duplicates_found = True
                print(f"❌ DUPLICATE METHOD in {file_path}: {method_name}")
                print(f"   📍 Lines: {', '.join(map(str, line_nums))}")
        
        return has_duplicates
    
    def find_large_files(self, threshold=3000):
        """Find files that are suspiciously large (might contain duplicates)"""
        print(f"🔍 Checking for files larger than {threshold} lines...")
        
        for py_file in self.root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                    
                if line_count > threshold:
                    print(f"⚠️  LARGE FILE: {py_file} ({line_count} lines)")
                    
                    # Check if this large file has duplicate methods
                    if self.find_duplicate_methods_in_file(py_file):
                        print(f"   🚨 Contains duplicate methods!")
                    print()
            except Exception as e:
                print(f"⚠️  Error reading {py_file}: {e}")
    
    def find_similar_code_blocks(self, min_lines=10):
        """Find similar code blocks that might be duplicated"""
        print(f"🔍 Checking for similar code blocks ({min_lines}+ lines)...")
        
        code_hashes = defaultdict(list)
        
        for py_file in self.root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Check sliding windows of code
                for i in range(len(lines) - min_lines + 1):
                    block = ''.join(lines[i:i + min_lines])
                    # Normalize whitespace for comparison
                    normalized = re.sub(r'\s+', ' ', block.strip())
                    if normalized:  # Skip empty blocks
                        block_hash = hashlib.md5(normalized.encode()).hexdigest()
                        code_hashes[block_hash].append((py_file, i + 1, i + min_lines))
            except Exception as e:
                print(f"⚠️  Error reading {py_file}: {e}")
        
        # Report duplicates
        for block_hash, locations in code_hashes.items():
            if len(locations) > 1:
                self.duplicates_found = True
                print(f"❌ SIMILAR CODE BLOCK found in {len(locations)} locations:")
                for file_path, start_line, end_line in locations:
                    print(f"   📁 {file_path}:{start_line}-{end_line}")
                print()
    
    def run_full_check(self):
        """Run all duplicate detection checks"""
        print("🚀 Starting duplicate code detection...")
        print("=" * 60)
        
        self.find_duplicate_classes()
        self.find_large_files()
        self.find_similar_code_blocks()
        
        print("=" * 60)
        if self.duplicates_found:
            print("❌ DUPLICATES FOUND! Please review and clean up.")
            return False
        else:
            print("✅ No duplicates detected!")
            return True

if __name__ == "__main__":
    detector = DuplicateDetector()
    success = detector.run_full_check()
    exit(0 if success else 1) 