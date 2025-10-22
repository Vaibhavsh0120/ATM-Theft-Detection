import os
import glob
from tqdm import tqdm

# This script merges the original 13 classes from the dataset into 2 final classes.
# CLASS 0: Face_Covered
# CLASS 1: Face_Uncovered

CLASS_REMAPPING = {
    # Old class IDs to be mapped to NEW ID 0 (Face_Covered)
    0: 0,  # balaclava
    1: 0,  # concealing glasses
    2: 0,  # cover
    3: 0,  # hand
    4: 0,  # mask
    5: 0,  # medicine mask
    9: 0,  # person-with-mask
    11: 0, # scarf
    12: 0, # thief_mask

    # Old class IDs to be mapped to NEW ID 1 (Face_Uncovered)
    6: 1,  # non-concealing glasses
    7: 1,  # normal
    8: 1,  # nothing
    10: 1, # person-without-mask
}

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)
dataset_root = os.path.join(script_dir, 'ATM-Theft-Detection-2') # Dataset folder relative to script

# Find all label files in train, valid, and test directories
label_files = []
for split in ['train', 'valid', 'test']:
    # Path relative to the script's directory
    path_pattern = os.path.join(dataset_root, split, 'labels', '*.txt')
    label_files.extend(glob.glob(path_pattern))

if not os.path.exists(dataset_root):
     print(f"Error: Dataset directory not found at '{dataset_root}'")
     print("Please run 'python setup.py' first from the ATM_Face_Recognition directory.")
     exit()

if not label_files:
    print(f"Warning: No label files found in '{dataset_root}'. Did you run `setup.py` to download the dataset first?")
    exit()

# Process each label file with a progress bar
for file_path in tqdm(label_files, desc="Merging Labels"):
    temp_lines = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

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
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed line in {file_path}: {line.strip()}")

        # Write the modified lines back to the file
        with open(file_path, 'w') as f:
            f.writelines(temp_lines)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


print(f"\n✅ Success! Processed and remapped labels in {len(label_files)} files within '{os.path.basename(dataset_root)}'.")