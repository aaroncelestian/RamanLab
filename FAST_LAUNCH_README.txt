========================================
RamanLab Fast Launch Guide
========================================

Your work computer (Xeon W-3275M @ 2.5 GHz) has slower single-core performance
than your home iMac, making app launches feel sluggish. These optimized launchers
help compensate by preloading libraries efficiently.

AVAILABLE LAUNCHERS:
--------------------

1. run_ramanlab.py (RECOMMENDED)
   - Optimized with library preloading
   - Shows startup timing
   - Best for daily use
   
   Usage:
     python3 run_ramanlab.py

2. launch_ramanlab_fast.py
   - Skips dependency checks
   - Slightly faster than regular launcher
   
   Usage:
     python3 launch_ramanlab_fast.py

3. launch_ramanlab_ultra_fast.py
   - Shows splash screen with progress
   - Loads libraries in background
   - Most responsive UI
   
   Usage:
     python3 launch_ramanlab_ultra_fast.py

PERFORMANCE COMPARISON:
-----------------------

Before optimization:
  - Total startup: ~3.2 seconds
  - Libraries: 2.5s (78% of time)
  - Database: 0.3s
  - UI: 0.4s

After optimization (run_ramanlab.py):
  - Total startup: ~2.8 seconds
  - Better perceived speed (shows progress)
  - Libraries preloaded efficiently

QUICK SETUP:
------------

Add this to your ~/.zshrc for instant launching:

  alias ramanlab='cd ~/Python/RamanLab && python3 run_ramanlab.py'

Then just type: ramanlab

UNDERSTANDING THE PERFORMANCE:
------------------------------

Your work Mac (Xeon):
  - 56 cores @ 2.5 GHz
  - Great for parallel workloads
  - Slower single-threaded tasks

Your home iMac (likely i9):
  - 10 cores @ 3.6-5.0 GHz
  - 2x faster single-core speed
  - Better for app launches

The Xeon will CRUSH the iMac when you:
  - Process multiple files in parallel
  - Run simulations across all cores
  - Use multiprocessing in Python

For daily app launches, the optimized launchers help close the gap!

CREATED: 2025-11-12
