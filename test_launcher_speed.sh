#!/bin/bash
# Test different launcher speeds

echo "=========================================="
echo "RamanLab Launcher Speed Comparison"
echo "=========================================="
echo ""

echo "Testing launchers (will open and close windows)..."
echo "Press Ctrl+C after each window appears to test next launcher"
echo ""

# Test 1: Original launcher
echo "1. Testing run_ramanlab.py (optimized with preloading)..."
echo "   Command: python3 run_ramanlab.py"
time python3 run_ramanlab.py &
LAUNCHER_PID=$!
sleep 3
kill $LAUNCHER_PID 2>/dev/null
echo ""

# Test 2: Fast launcher
echo "2. Testing launch_ramanlab_fast.py (no dependency check)..."
echo "   Command: python3 launch_ramanlab_fast.py"
time python3 launch_ramanlab_fast.py &
LAUNCHER_PID=$!
sleep 3
kill $LAUNCHER_PID 2>/dev/null
echo ""

# Test 3: Ultra fast launcher
echo "3. Testing launch_ramanlab_ultra_fast.py (with splash screen)..."
echo "   Command: python3 launch_ramanlab_ultra_fast.py"
time python3 launch_ramanlab_ultra_fast.py &
LAUNCHER_PID=$!
sleep 3
kill $LAUNCHER_PID 2>/dev/null
echo ""

echo "=========================================="
echo "Comparison complete!"
echo "=========================================="
echo ""
echo "Recommended launcher for daily use:"
echo "  python3 run_ramanlab.py"
echo ""
echo "Or create an alias in your ~/.zshrc:"
echo "  alias ramanlab='cd ~/Python/RamanLab && python3 run_ramanlab.py'"
