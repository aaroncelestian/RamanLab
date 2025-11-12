#!/bin/bash
# Benchmark your actual application launch time

echo "Benchmarking RamanLab application launch..."
echo ""

# Find your main launch script
if [ -f "run_ramanlab.sh" ]; then
    LAUNCHER="./run_ramanlab.sh"
elif [ -f "launch_ramanlab.py" ]; then
    LAUNCHER="python3 launch_ramanlab.py"
elif [ -f "run_ramanlab.py" ]; then
    LAUNCHER="python3 run_ramanlab.py"
else
    echo "Could not find launcher script"
    echo "Available launch scripts:"
    ls -1 launch*.py run*.py 2>/dev/null
    exit 1
fi

echo "Using launcher: $LAUNCHER"
echo ""

# Time the launch
echo "Launching application..."
time $LAUNCHER &
APP_PID=$!

echo ""
echo "Application launched with PID: $APP_PID"
echo ""
echo "Compare this time to your home iMac."
echo "If similar, the slowness is elsewhere."
