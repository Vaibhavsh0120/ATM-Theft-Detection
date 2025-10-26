import os
import glob
from tqdm import tqdm

# This script merges the 16 classes from the dataset into 2 final classes.
# CLASS 0: Face_Covered
# CLASS 1: Face_Uncovered

#
# --- THIS IS THE CORRECTED MAPPING ---
#
# Based on the 0-indexed order of your class list:
# 0: balaclava, 1: concealing glasses, 2: cover, 3: face, 4: hand, 5: head,
# 6: Helmet, 7: mask, 8: medicine mask, 9: non-concealing glasses, 10: normal,
# 11: nothing, 12: person-with-mask, 13: person-without-mask, 14: scarf, 15: thief_mask
#
CLASS_REMAPPING = {
    # --- Map to NEW ID 0 (Face_Covered) ---
    0: 0,  # balaclava
    1: 0,  # concealing glasses 
    2: 0,  # cover 
    4: 0,  # hand 
    6: 0,  # Helmet
    7: 0,  # mask
    8: 0,  # medicine mask
    12: 0, # person-with-mask
    14: 0, # scarf
    15: 0, # thief_mask

    # --- Map to NEW ID 1 (Face_Uncovered) ---
    3: 1,  # face
    5: 1,  # head 
    9: 1,  # non-concealing glasses
    10: 1, # normal
    11: 1, # nothing
    13: 1, # person-without-mask
}

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming the dataset folder is still named 'ATM-Theft-Detection-2'
# If it has a new name, change it here.
dataset_root = os.path.join(script_dir, 'ATM-Theft-Detection-3') 

# Check if dataset directory exists
if not os.path.exists(dataset_root):
    print(f"Error: Dataset directory not found at '{dataset_root}'")
    print("Please ensure the dataset folder is in the same directory as this script.")
    exit()

# Find all label files in train, valid, and test directories
label_files = []
for split in ['train', 'valid', 'test']:
    # Path relative to the script's directory
    path_pattern = os.path.join(dataset_root, split, 'labels', '*.txt')
    label_files.extend(glob.glob(path_pattern))

if not label_files:
    print(f"Warning: No label files found in '{dataset_root}'. Is the dataset populated?")
    exit()

print(f"Found {len(label_files)} label files to process.")
remapped_count = 0
unmapped_ids = set()

# Process each label file with a progress bar
for file_path in tqdm(label_files, desc="Merging Labels"):
    temp_lines = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        processed_line_in_file = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                old_class_id = int(parts[0])
                if old_class_id in CLASS_REMAPPING:
                    new_class_id = CLASS_REMAPPING[old_class_id]
                    new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                    temp_lines.append(new_line)
                    processed_line_in_file = True
                else:
                    # Collect any IDs that are not in our map
                    unmapped_ids.add(old_class_id)

            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed line in {file_path}: {line.strip()}")

        # Only write back to the file if we successfully processed at least one line
        if processed_line_in_file:
            with open(file_path, 'w') as f:
                f.writelines(temp_lines)
            remapped_count += 1

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

print(f"\n✅ Success! Processed and remapped labels in {remapped_count} files.")
if unmapped_ids:
    print(f"Warning: The following class IDs were found in your files but were not in the remapping dict: {sorted(list(unmapped_ids))}")
    print("These classes were skipped and not included in the new label files.")