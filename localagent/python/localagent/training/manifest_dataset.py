from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from localagent.vision import build_training_transforms, load_rgb_image


class ManifestImageDataset:
    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        split: str,
        label_to_index: dict[str, int],
        image_size: int = 224,
        cache_dir: Path | None = None,
        cache_format: str = "png",
        transform: Any | None = None,
    ) -> None:
        self.image_size = image_size
        self.label_to_index = label_to_index
        self.cache_dir = cache_dir
        self.cache_suffix = ".raw" if cache_format == "raw" else ".png"
        self.transform = transform or build_training_transforms(image_size)
        self.cached_transform = build_training_transforms(image_size, pre_resized=True)
        self.records = list(
            frame.filter(
                pl.col("is_valid")
                & (pl.col("label") != "unknown")
                & (pl.col("split") == split)
            )
            .sort("sample_id")
            .select("sample_id", "image_path", "label")
            .iter_rows(named=True)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path, use_cached_transform = self._resolve_image_path(record)
        rgb_image = load_rgb_image(
            image_path,
            raw_image_size=self.image_size if image_path.suffix.lower() == ".raw" else None,
        )
        transform = self.cached_transform if use_cached_transform else self.transform
        tensor = transform(rgb_image)
        label_index = self.label_to_index[str(record["label"])]
        return tensor, label_index

    def _resolve_image_path(self, record: dict[str, Any]) -> tuple[Path, bool]:
        if self.cache_dir is not None:
            cached_path = self.cache_dir / f"{record['sample_id']}{self.cache_suffix}"
            if cached_path.is_file():
                return cached_path, True
        return Path(str(record["image_path"])), False
