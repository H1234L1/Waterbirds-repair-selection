from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
GROUP_NAMES = {
    0: "landbird on land",
    1: "landbird on water",
    2: "waterbird on land",
    3: "waterbird on water",
}


def find_dataset_root(path: str | Path) -> Path:
    """Find the directory containing metadata.csv below path (one nested level is common)."""
    path = Path(path).expanduser().resolve()
    candidates = [path, path / "waterbird_complete95_forest2water2"]
    for candidate in candidates:
        if (candidate / "metadata.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"metadata.csv not found under {path}. Run scripts/download_waterbirds.py first."
    )


def load_metadata(data_dir: str | Path) -> tuple[Path, pd.DataFrame]:
    root = find_dataset_root(data_dir)
    metadata = pd.read_csv(root / "metadata.csv")
    required = {"img_filename", "y", "place", "split"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")
    metadata = metadata.copy()
    for column in ("y", "place", "split"):
        metadata[column] = metadata[column].astype(int)
    if not set(metadata.y.unique()).issubset({0, 1}):
        raise ValueError("bird label y must be binary (0=landbird, 1=waterbird)")
    if not set(metadata.place.unique()).issubset({0, 1}):
        raise ValueError("place must be binary (0=land, 1=water)")
    if not set(metadata.split.unique()).issubset(SPLIT_NAMES):
        raise ValueError("split must use official codes 0=train, 1=val, 2=test")
    metadata["group"] = 2 * metadata["y"] + metadata["place"]
    return root, metadata


def dataset_statistics(metadata: pd.DataFrame) -> dict:
    stats: dict[str, dict] = {}
    for split_id, split_name in SPLIT_NAMES.items():
        frame = metadata.loc[metadata.split == split_id]
        stats[split_name] = {
            "num_samples": int(len(frame)),
            "class_counts": {
                "landbird": int((frame.y == 0).sum()),
                "waterbird": int((frame.y == 1).sum()),
            },
            "group_counts": {
                GROUP_NAMES[group]: int((frame.group == group).sum()) for group in range(4)
            },
        }
    return stats


class WaterbirdsDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        transform: Callable | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> None:
        if split not in SPLIT_NAMES.values():
            raise ValueError(f"unknown split {split!r}; expected one of {list(SPLIT_NAMES.values())}")
        self.root, all_metadata = load_metadata(data_dir)
        if metadata is not None:
            all_metadata = metadata
        split_id = {name: idx for idx, name in SPLIT_NAMES.items()}[split]
        self.metadata = all_metadata.loc[all_metadata.split == split_id].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = self.root / str(row.img_filename)
        if not image_path.is_file():
            raise FileNotFoundError(f"image listed in metadata does not exist: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row.y), int(row.place), int(row.group)

