from __future__ import annotations

import sys
import math
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repo_paths import DATASET_DIR

REMAPPED_MARKER = DATASET_DIR / ".labels_remapped"

# CLASS 0: Face_Covered
# CLASS 1: Face_Uncovered
CLASS_REMAPPING = {
    1: 0,  # Helmet
    7: 0,  # balaclava
    8: 0,  # concealing glasses
    9: 0,  # cover
    11: 0,  # hand
    12: 0,  # mask
    13: 0,  # medicine mask
    17: 0,  # person-with-mask
    19: 0,  # scarf
    20: 0,  # thief_mask
    0: 1,  # Cuong
    2: 1,  # Hung
    3: 1,  # Lau-Ka-Fai
    4: 1,  # Trung
    5: 1,  # Tuan
    6: 1,  # Vu
    10: 1,  # face
    14: 1,  # non-concealing glasses
    15: 1,  # normal
    16: 1,  # nothing
    18: 1,  # person-without-mask
}


def iter_label_files() -> list[Path]:
    label_files: list[Path] = []
    for split in ("train", "valid", "test"):
        label_files.extend(sorted((DATASET_DIR / split / "labels").glob("*.txt")))
    return label_files


def mapped_lines(file_path: Path) -> list[str]:
    updated_lines: list[str] = []

    for line in file_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        try:
            old_class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"Malformed class in '{file_path}': {line}") from exc

        if old_class_id not in CLASS_REMAPPING:
            raise ValueError(f"Unknown class {old_class_id} in '{file_path}'; no labels were discarded.")

        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid bounding box in '{file_path}': {line}") from exc
        if len(coords) != 4 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
            raise ValueError(f"Invalid YOLO detection label in '{file_path}': {line}")

        new_class_id = CLASS_REMAPPING[old_class_id]
        updated_lines.append(f"{new_class_id} {' '.join(parts[1:])}".rstrip())
    return updated_lines


def remap_file(file_path: Path, unmapped_ids: set[int]) -> bool:
    updated_lines = mapped_lines(file_path)
    if updated_lines:
        file_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return bool(updated_lines)


def main() -> None:
    if not DATASET_DIR.exists():
        print(f"Dataset directory not found at '{DATASET_DIR}'.")
        print("Run 'python training/scripts/bootstrap_assets.py' first.")
        raise SystemExit(1)

    if REMAPPED_MARKER.exists():
        print(f"Dataset labels already appear remapped at '{REMAPPED_MARKER}'.")
        print(
            "Run 'python training/scripts/bootstrap_assets.py' to restore a fresh "
            "Roboflow export before remapping again."
        )
        raise SystemExit(1)

    label_files = iter_label_files()
    if not label_files:
        print(f"No label files were found under '{DATASET_DIR}'.")
        raise SystemExit(1)

    metadata_path = DATASET_DIR / "data.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    expected_names = [
        "Cuong",
        "Helmet",
        "Hung",
        "Lau-Ka-Fai",
        "Trung",
        "Tuan",
        "Vu",
        "balaclava",
        "concealing glasses",
        "cover",
        "face",
        "hand",
        "mask",
        "medicine mask",
        "non-concealing glasses",
        "normal",
        "nothing",
        "person-with-mask",
        "person-without-mask",
        "scarf",
        "thief_mask",
    ]
    names = metadata.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if names != expected_names:
        raise ValueError(
            "Dataset class order differs from the supported 21-class export. Download a fresh version 4 dataset."
        )
    backup_root = DATASET_DIR / ".original-labels"
    if backup_root.exists():
        raise ValueError(
            "An incomplete remap backup exists at .original-labels. Restore a fresh dataset before retrying."
        )
    # Validate the entire export before changing a single label.
    for file_path in label_files:
        mapped_lines(file_path)
    for split in ("train", "valid", "test"):
        source = DATASET_DIR / split / "labels"
        if source.is_dir():
            shutil.copytree(source, backup_root / split)
    shutil.copy2(metadata_path, backup_root / "data.yaml")
    print(f"Found {len(label_files)} label files to process; originals backed up in '{backup_root}'.")
    remapped_count = 0
    unmapped_ids: set[int] = set()

    for file_path in tqdm(label_files, desc="Merging Labels"):
        if remap_file(file_path, unmapped_ids):
            remapped_count += 1

    if remapped_count == 0:
        print("No label files were remapped. Check that the dataset contains the original 21-class labels.")
        raise SystemExit(1)

    metadata.update(nc=2, names=["Face_Covered", "Face_Uncovered"])
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    REMAPPED_MARKER.write_text(
        "This dataset export has already been remapped into the 2-class training set.\n",
        encoding="utf-8",
    )
    print(f"Processed and remapped labels in {remapped_count} files.")
    if unmapped_ids:
        print(f"Unmapped class IDs were skipped: {sorted(unmapped_ids)}")


if __name__ == "__main__":
    main()
