#!/bin/bash
# Clean up iCloud conflict files

echo "Finding iCloud conflict files..."
find . -name "*conflicted copy*" -type f

echo ""
echo "Do you want to delete these conflict files? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    find . -name "*conflicted copy*" -type f -delete
    echo "Conflict files deleted."
else
    echo "No files deleted."
fi
