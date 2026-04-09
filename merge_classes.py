import os
import glob
from tqdm import tqdm

# This script merges the 21 classes from the dataset (version 4) into 2 final classes.
# CLASS 0: Face_Covered
# CLASS 1: Face_Uncovered

#
# --- THIS IS THE CORRECTED MAPPING (based on data.yaml) ---
#
# 0: Cuong, 1: Helmet, 2: Hung, 3: Lau-Ka-Fai, 4: Trung, 5: Tuan,
# 6: Vu, 7: balaclava, 8: concealing glasses, 9: cover, 10: face,
# 11: hand, 12: mask, 13: medicine mask, 14: non-concealing glasses,
# 15: normal, 16: nothing, 17: person-with-mask,
# 18: person-without-mask, 19: scarf, 20: thief_mask
#
CLASS_REMAPPING = {
    # --- Map to NEW ID 0 (Face_Covered) ---
    1: 0,  # Helmet
    7: 0,  # balaclava
    8: 0,  # concealing glasses
    9: 0,  # cover
    11: 0, # hand (assuming hand covers face)
    12: 0, # mask
    13: 0, # medicine mask
    17: 0, # person-with-mask
    19: 0, # scarf
    20: 0, # thief_mask

    # --- Map to NEW ID 1 (Face_Uncovered) ---
    0: 1,  # Cuong
    2: 1,  # Hung
    3: 1,  # Lau-Ka-Fai
    4: 1,  # Trung
    5: 1,  # Tuan
    6: 1,  # Vu
    10: 1, # face
    14: 1, # non-concealing glasses
    15: 1, # normal
    16: 1, # nothing
    18: 1, # person-without-mask
}

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Point to the dataset folder from setup.py
dataset_root = os.path.join(script_dir, 'ATM-Theft-Detection-4') 

# Check if dataset directory exists
if not os.path.exists(dataset_root):
    print(f"Error: Dataset directory not found at '{dataset_root}'")
    print("Please run 'python setup.py' first.")
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