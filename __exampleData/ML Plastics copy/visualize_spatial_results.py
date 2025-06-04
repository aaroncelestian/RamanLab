import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.widgets import Slider, CheckButtons
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import simpledialog

def extract_coordinates(filename):
    # Extract Y and X coordinates from filename using regex
    match = re.search(r'Y(\d+)_X(\d+)', filename)
    if match:
        y = int(match.group(1))
        x = int(match.group(2))
        return y, x
    return None, None

# Read the results CSV file
df = pd.read_csv('unknown_spectra_results.csv')

# Extract coordinates from filenames
coordinates = df['File'].apply(extract_coordinates)
df['Y'] = [coord[0] for coord in coordinates]
df['X'] = [coord[1] for coord in coordinates]

# --- Tkinter dropdown for colormap selection ---
def get_colormap_tkinter():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    cmap_options = ['inferno', 'viridis', 'seismic']
    selected = simpledialog.askstring(
        "Colormap Selection",
        "Enter colormap (inferno, viridis, seismic):",
        initialvalue='inferno')
    if selected not in cmap_options:
        selected = 'inferno'
    root.destroy()
    return selected

selected_cmap = get_colormap_tkinter()

plt.style.use('default')
fig = plt.figure(figsize=(16, 12))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Main plot panel
ax = plt.axes([0.1, 0.25, 0.8, 0.6])

# Smoothing slider (centered above plot)
slider_width = 0.09
slider_height = 0.04
slider_y = 0.88
slider_label_height = 0.018
slider_label_pad = 0.006
slider_start_x = 0.46  # Centered

smooth_ax = plt.axes([slider_start_x, slider_y, slider_width, slider_height])
smooth_slider = Slider(smooth_ax, '', 0, 1, valinit=0, valstep=0.01, color='lightgray')
smooth_label_ax = plt.axes([slider_start_x, slider_y + slider_height + slider_label_pad, slider_width, slider_label_height])
smooth_label_ax.axis('off')
smooth_label_ax.text(0.5, 0.5, 'Smoothing', ha='center', va='center', fontsize=9)

# Filter Perfect Corr checkbox (horizontally aligned with smoothing slider label)
checkbox_width = slider_width
checkbox_height = slider_label_height
checkbox_x = slider_start_x + slider_width + 0.03  # 0.03 gap to the right of the slider label
checkbox_y = slider_y + slider_height + slider_label_pad
checkbox_ax = plt.axes([checkbox_x, checkbox_y, checkbox_width, checkbox_height])
checkbox_ax.set_frame_on(False)
checkbox = CheckButtons(checkbox_ax, ['Filter Perfect Corr'], [False])
for label in checkbox.labels:
    label.set_fontsize(9)

# Prepare data for plot
x_coords = sorted(df['X'].unique())
y_coords = sorted(df['Y'].unique())
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)
X, Y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
class_a_grid = np.zeros_like(X, dtype=float)
class_b_grid = np.zeros_like(X, dtype=float)
for _, row in df.iterrows():
    x, y = row['X'], row['Y']
    if row['Prediction'] == 'Class A':
        class_a_grid[y-y_min, x-x_min] = row['Confidence']
    else:
        class_b_grid[y-y_min, x-x_min] = row['Confidence']
difference_grid = class_a_grid - class_b_grid
original_grid = difference_grid.copy()

# State for filtering
filter_corr_state = {'active': False}

def get_filtered_grid():
    if filter_corr_state['active']:
        mask = (class_a_grid == 1) | (class_b_grid == 1)
        return np.where(mask, -1.0, difference_grid)
    else:
        return difference_grid

# Plot
im = ax.imshow(get_filtered_grid(), cmap=selected_cmap, alpha=0.8,
               extent=[x_min-0.5, x_max+0.5, y_max+0.5, y_min-0.5],
               interpolation='nearest', vmin=-1, vmax=1)
cbar_ax = fig.add_axes([0.1, 0.18, 0.8, 0.03])
cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Class Difference (A - B)')
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.set_ticklabels(['Strong B', 'Weak B', 'Neutral', 'Weak A', 'Strong A'])
cbar.ax.tick_params(labelsize=10)

ax.set_title('Spatial Distribution of Class Differences', pad=20, fontsize=14)
ax.set_xlabel('X Position (pixels)', fontsize=12, labelpad=10)
ax.set_ylabel('Y Position (pixels)', fontsize=12, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

# Update functions
def update_smoothing(val):
    grid = get_filtered_grid()
    if val > 0:
        smoothed = gaussian_filter(grid, sigma=val)
        im.set_data(smoothed)
    else:
        im.set_data(grid)
    fig.canvas.draw_idle()

def on_checkbox(label):
    filter_corr_state['active'] = checkbox.get_status()[0]
    update_smoothing(smooth_slider.val)

# Connect widgets to update functions
smooth_slider.on_changed(update_smoothing)
checkbox.on_clicked(on_checkbox)

plt.show() 