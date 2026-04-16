from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from localagent.domain import WasteSample

try:
    import polars as pl
except ImportError:  # pragma: no cover - dependency is installed by uv sync
    pl = None


class DatasetIndex:
    def __init__(self, records: Sequence[WasteSample]) -> None:
        self.records = list(records)

    @classmethod
    def from_paths(
        cls,
        image_paths: Iterable[Path],
        default_label: str = "unlabeled",
    ) -> DatasetIndex:
        records = [
            WasteSample(
                sample_id=image_path.stem,
                image_path=image_path,
                label=default_label,
            )
            for image_path in image_paths
        ]
        return cls(records)

    def to_polars(self) -> Any:
        if pl is None:
            raise RuntimeError("polars is not available. Run `uv sync` inside localagent first.")
        return pl.DataFrame(
            {
                "sample_id": [record.sample_id for record in self.records],
                "image_path": [str(record.image_path) for record in self.records],
                "label": [record.label for record in self.records],
            }
        )

    @classmethod
    def from_polars(cls, frame: Any) -> DatasetIndex:
        records: list[WasteSample] = []
        for row in frame.iter_rows(named=True):
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"sample_id", "image_path", "label"}
            }
            records.append(
                WasteSample(
                    sample_id=str(row["sample_id"]),
                    image_path=Path(str(row["image_path"])),
                    label=str(row.get("label", "unknown")),
                    metadata=metadata,
                )
            )
        return cls(records)

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> DatasetIndex:
        if pl is None:
            raise RuntimeError("polars is not available. Run `uv sync` inside localagent first.")
        frame = pl.read_parquet(manifest_path)
        return cls.from_polars(frame)

    def write_manifest(self, output_path: Path) -> Path:
        frame = self.to_polars()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output_path)
        return output_path

    def __len__(self) -> int:
        return len(self.records)
