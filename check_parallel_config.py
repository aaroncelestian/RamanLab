#!/usr/bin/env python3
"""
Parallel Processing Configuration Checker for RamanLab

This script scans the codebase for potential parallel processing bottlenecks
and reports on the configuration of multi-core settings.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def scan_file(filepath):
    """Scan a Python file for parallel processing configurations."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        # Check for n_jobs limitations
        for i, line in enumerate(lines, 1):
            # Check for n_jobs=1 or other small numbers
            if re.search(r'n_jobs\s*=\s*[1-9](?!\d)', line):
                issues.append({
                    'file': filepath,
                    'line': i,
                    'type': 'n_jobs_limited',
                    'content': line.strip(),
                    'severity': 'HIGH'
                })
            
            # Check for min(cpu_count(), N) patterns
            if re.search(r'min\s*\(\s*.*cpu_count.*,\s*\d+\s*\)', line):
                issues.append({
                    'file': filepath,
                    'line': i,
                    'type': 'cpu_count_limited',
                    'content': line.strip(),
                    'severity': 'HIGH'
                })
            
            # Check for max_workers with small numbers
            if re.search(r'max_workers\s*=\s*[1-9](?!\d)', line):
                issues.append({
                    'file': filepath,
                    'line': i,
                    'type': 'max_workers_limited',
                    'content': line.strip(),
                    'severity': 'MEDIUM'
                })
            
            # Check for thread environment variables set to 1
            if re.search(r'(OMP|MKL|OPENBLAS|NUMEXPR)_NUM_THREADS.*=.*[\'"]1[\'"]', line):
                if not line.strip().startswith('#'):
                    issues.append({
                        'file': filepath,
                        'line': i,
                        'type': 'thread_env_limited',
                        'content': line.strip(),
                        'severity': 'CRITICAL'
                    })
        
        # Check for good patterns (n_jobs=-1)
        good_patterns = []
        for i, line in enumerate(lines, 1):
            if 'n_jobs=-1' in line or 'n_jobs = -1' in line:
                good_patterns.append({
                    'file': filepath,
                    'line': i,
                    'type': 'n_jobs_optimized',
                    'content': line.strip()
                })
        
        return issues, good_patterns
        
    except Exception as e:
        return [], []

def scan_directory(root_dir):
    """Scan all Python files in directory."""
    all_issues = []
    all_good = []
    
    root_path = Path(root_dir)
    
    # Scan Python files
    for py_file in root_path.rglob('*.py'):
        # Skip __pycache__ and .git directories
        if '__pycache__' in str(py_file) or '.git' in str(py_file):
            continue
            
        issues, good = scan_file(py_file)
        all_issues.extend(issues)
        all_good.extend(good)
    
    # Scan shell scripts
    for sh_file in root_path.rglob('*.sh'):
        try:
            with open(sh_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                if re.search(r'export\s+(OMP|MKL|OPENBLAS|NUMEXPR)_NUM_THREADS=1', line):
                    if not line.strip().startswith('#'):
                        all_issues.append({
                            'file': sh_file,
                            'line': i,
                            'type': 'thread_env_limited',
                            'content': line.strip(),
                            'severity': 'CRITICAL'
                        })
        except:
            pass
    
    return all_issues, all_good

def print_report(issues, good_patterns):
    """Print a formatted report."""
    print("=" * 80)
    print("RAMANLAB PARALLEL PROCESSING CONFIGURATION REPORT")
    print("=" * 80)
    print()
    
    # Group issues by severity
    by_severity = defaultdict(list)
    for issue in issues:
        by_severity[issue['severity']].append(issue)
    
    # Print issues
    if issues:
        print("⚠️  POTENTIAL BOTTLENECKS FOUND:")
        print()
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity in by_severity:
                print(f"\n{severity} PRIORITY ({len(by_severity[severity])} issues):")
                print("-" * 80)
                
                for issue in by_severity[severity]:
                    rel_path = os.path.relpath(issue['file'])
                    print(f"\n  File: {rel_path}")
                    print(f"  Line: {issue['line']}")
                    print(f"  Type: {issue['type']}")
                    print(f"  Code: {issue['content']}")
    else:
        print("✅ NO BOTTLENECKS FOUND!")
    
    print()
    print("=" * 80)
    print(f"OPTIMIZED CONFIGURATIONS: {len(good_patterns)}")
    print("=" * 80)
    
    if good_patterns:
        # Group by file
        by_file = defaultdict(list)
        for pattern in good_patterns:
            by_file[pattern['file']].append(pattern)
        
        print(f"\nFound {len(good_patterns)} optimized parallel configurations across {len(by_file)} files:")
        for filepath in sorted(by_file.keys()):
            rel_path = os.path.relpath(filepath)
            print(f"  ✓ {rel_path} ({len(by_file[filepath])} instances)")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total issues found: {len(issues)}")
    print(f"Total optimized configs: {len(good_patterns)}")
    
    if issues:
        print("\n⚠️  Action required: Review and fix the issues listed above")
    else:
        print("\n✅ All parallel processing configurations are optimized!")
    print()

def main():
    """Main function."""
    script_dir = Path(__file__).parent
    
    print("Scanning RamanLab codebase for parallel processing configurations...")
    print(f"Root directory: {script_dir}")
    print()
    
    issues, good_patterns = scan_directory(script_dir)
    print_report(issues, good_patterns)

if __name__ == "__main__":
    main()
