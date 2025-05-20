import pickle
import os

# Path to the mineral_modes.pkl file
pkl_path = "mineral_modes.pkl"

# Load the data
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# Print information about the original data
print(f"Original data type: {type(data)}")
print(f"Original number of entries: {len(data)}")

# Count entries with "__converted_" in their names
converted_entries = [key for key in data.keys() if "__converted_" in key]
print(f"Number of __converted_ entries: {len(converted_entries)}")
print(f"__converted_ entries to remove: {converted_entries}")

# Create a backup of the original file
backup_path = f"{pkl_path}.bak"
if not os.path.exists(backup_path):
    print(f"Creating backup at {backup_path}")
    with open(pkl_path, "rb") as src, open(backup_path, "wb") as dst:
        dst.write(src.read())

# Remove all entries with "__converted_" in their names
for key in converted_entries:
    del data[key]

print(f"New number of entries: {len(data)}")

# Save the modified data back to the file
with open(pkl_path, "wb") as f:
    pickle.dump(data, f)

print(f"Successfully removed all __converted_ entries from {pkl_path}") 