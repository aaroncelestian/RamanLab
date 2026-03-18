import struct
import numpy as np

file_path = '/Users/aaroncelestian/Python/RamanLab/demo_data/tavetch_switzerland_05.l6s'

with open(file_path, 'rb') as f:
    data = f.read()

print(f'File size: {len(data)} bytes')
print(f'Header: {data[:8].decode("ascii", errors="ignore")}')

# Look for "Spectrum" string
spectrum_pos = data.find(b'Spectrum')
print(f'\n"Spectrum" found at offset: {spectrum_pos}')

# From hex dump, we see float data starting around 0x200 (512)
# The pattern shows: 28 af c7 42 = float value
# Let's check different offsets more carefully
print('\nSearching for data start offset...')
print('Checking offsets from hex dump analysis:')

for offset in [0x1FC, 0x200, 0x204, 0x208]:
    test_data = np.frombuffer(data[offset:offset+40], dtype=np.float32)
    print(f'  Offset {offset} (0x{offset:03x}): {test_data[:5]}')

# Based on hex dump: 28 af c7 42 86 f2 d8 42 ca 6b f2 42
# These are little-endian float32 values
# Let's try offset right before these bytes
data_offset_search = data.find(b'\x28\xaf\xc7\x42')
print(f'\nFound signature bytes at offset: {data_offset_search}')

best_offset = data_offset_search if data_offset_search > 0 else 0x200

data_start = best_offset
remaining = len(data) - data_start
aligned_size = (remaining // 4) * 4
float_data = np.frombuffer(data[data_start:data_start+aligned_size], dtype=np.float32)

# Try reading as pairs (wavenumber, intensity)
estimated_points = len(float_data) // 2
print(f'\nTotal float32 values: {len(float_data)}')
print(f'Estimated spectral points: {estimated_points}')

pairs = float_data[:estimated_points*2].reshape(-1, 2)
wavenumbers = pairs[:, 0]
intensities = pairs[:, 1]

print(f'\nWavenumbers range: {wavenumbers.min():.2f} to {wavenumbers.max():.2f}')
print(f'Intensities range: {intensities.min():.2f} to {intensities.max():.2f}')
print(f'\nFirst 10 data points:')
for i in range(10):
    print(f'  {wavenumbers[i]:8.2f} cm^-1 : {intensities[i]:10.2f}')
print(f'\nLast 5 data points:')
for i in range(-5, 0):
    print(f'  {wavenumbers[i]:8.2f} cm^-1 : {intensities[i]:10.2f}')

# Analyze spacing
print('\n' + '='*60)
print('SPACING ANALYSIS')
print('='*60)
diffs = np.diff(wavenumbers)
valid_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
if len(valid_diffs) > 0:
    print(f'Spacing range: {np.min(valid_diffs):.2f} to {np.max(valid_diffs):.2f} cm^-1')
    print(f'Mean spacing: {np.mean(valid_diffs):.2f} cm^-1')
    print(f'Median spacing: {np.median(valid_diffs):.2f} cm^-1')
    print(f'\nSpacing distribution:')
    print(f'  < 10 cm^-1: {np.sum(valid_diffs < 10)} points')
    print(f'  10-50 cm^-1: {np.sum((valid_diffs >= 10) & (valid_diffs < 50))} points')
    print(f'  50-200 cm^-1: {np.sum((valid_diffs >= 50) & (valid_diffs < 200))} points')
    print(f'  > 200 cm^-1: {np.sum(valid_diffs >= 200)} points')
