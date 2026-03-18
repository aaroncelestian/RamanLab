#!/usr/bin/env python3
"""Test script to reproduce the import error."""

import sys
import os
import numpy as np
from pathlib import Path

# Simulate the parse_spectrum_file method
def parse_metadata_line(line, metadata):
    """Parse a metadata line starting with #."""
    content = line[1:].strip()
    
    if not content:
        return
    
    for separator in [':', '=']:
        if separator in content:
            parts = content.split(separator, 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                # Try to convert numeric values to appropriate types
                try:
                    if '.' not in value:
                        metadata[key] = int(value)
                    else:
                        metadata[key] = float(value)
                except ValueError:
                    metadata[key] = value
                return
    
    if 'comments' not in metadata:
        metadata['comments'] = []
    metadata['comments'].append(content)

def detect_delimiter(line, file_extension):
    """Detect the delimiter used in the data file."""
    if file_extension == '.csv':
        if ',' in line:
            return ','
    
    # Count occurrences of different delimiters - ensure all are integers
    comma_count = int(line.count(','))
    tab_count = int(line.count('\t'))
    space_parts = [x for x in line.split(' ') if x.strip()]
    space_count = max(0, len(space_parts) - 1)
    
    print(f"DEBUG: comma_count={comma_count} (type={type(comma_count).__name__})")
    print(f"DEBUG: tab_count={tab_count} (type={type(tab_count).__name__})")
    print(f"DEBUG: space_count={space_count} (type={type(space_count).__name__})")
    
    # Choose delimiter with highest count
    if comma_count > 0 and comma_count >= tab_count and comma_count >= space_count:
        return ','
    elif tab_count > 0 and tab_count >= space_count:
        return '\t'
    else:
        return None

# Test with the demo file
test_file = '/Users/aaroncelestian/Python/RamanLab/demo_data/alps_switzerland_01.txt'

try:
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Testing file: {test_file}")
    print(f"Total lines: {len(lines)}")
    
    # Test first data line
    first_line = lines[0].strip()
    print(f"\nFirst line: {repr(first_line)}")
    
    file_extension = Path(test_file).suffix.lower()
    print(f"File extension: {file_extension}")
    
    delimiter = detect_delimiter(first_line, file_extension)
    print(f"Detected delimiter: {repr(delimiter)}")
    
    print("\n✓ No error occurred!")
    
except Exception as e:
    import traceback
    print(f"\n✗ ERROR: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()
