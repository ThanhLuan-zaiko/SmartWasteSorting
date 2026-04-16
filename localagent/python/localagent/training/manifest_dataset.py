from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import polars as pl

from localagent.vision import build_training_transforms, load_bgr_image


class ManifestImageDataset:
    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        split: str,
        label_to_index: dict[str, int],
        image_size: int = 224,
        transform: Any | None = None,
    ) -> None:
        self.label_to_index = label_to_index
        self.transform = transform or build_training_transforms(image_size)
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
        image = load_bgr_image(Path(str(record["image_path"])))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_image)
        label_index = self.label_to_index[str(record["label"])]
        return tensor, label_index
