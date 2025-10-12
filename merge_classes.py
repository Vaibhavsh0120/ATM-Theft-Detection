import os
import glob
from tqdm import tqdm # Import the tqdm library

# Mapping based on your provided data.yaml file
CLASS_REMAPPING = {
    # Old IDs to be mapped to NEW ID 0 (Face_Covered)
    0: 0,  # balaclava
    1: 0,  # concealing glasses
    2: 0,  # cover
    3: 0,  # hand
    4: 0,  # mask
    5: 0,  # medicine mask
    9: 0,  # person-with-mask
    11: 0, # scarf
    12: 0, # thief_mask

    # Old IDs to be mapped to NEW ID 1 (Face_Uncovered)
    6: 1,  # non-concealing glasses
    7: 1,  # normal
    8: 1,  # nothing
    10: 1, # person-without-mask
}

# Find all label files in train, valid, and test directories
label_files = []
for split in ['train', 'valid', 'test']:
    label_files.extend(glob.glob(os.path.join(split, 'labels', '*.txt')))

# THE ONLY CHANGE IS HERE: We wrap the loop with tqdm()
# Process each label file with a progress bar
for file_path in tqdm(label_files, desc="Merging Labels"):
    temp_lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        old_class_id = int(parts[0])

        if old_class_id in CLASS_REMAPPING:
            new_class_id = CLASS_REMAPPING[old_class_id]
            new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
            temp_lines.append(new_line)

    # Write the modified lines back to the file
    with open(file_path, 'w') as f:
        f.writelines(temp_lines)

print(f"\nSuccess! Processed and remapped labels in {len(label_files)} files.")