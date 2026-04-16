from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from localagent.config import DatasetPipelineConfig
from localagent.data import DatasetPipeline


def _write_rgb_image(path: Path, width: int, height: int, value: int) -> None:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Unable to write image: {path}")


def _build_config(
    tmp_path: Path,
    *,
    min_width: int = 32,
    min_height: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> DatasetPipelineConfig:
    return DatasetPipelineConfig(
        raw_dataset_dir=tmp_path / "dataset",
        manifest_dir=tmp_path / "artifacts" / "manifests",
        report_dir=tmp_path / "artifacts" / "reports",
        min_width=min_width,
        min_height=min_height,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=7,
    )


def test_scan_marks_invalid_small_and_duplicate_images(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    config.raw_dataset_dir.mkdir(parents=True)

    canonical = config.raw_dataset_dir / "a_valid.jpg"
    duplicate = config.raw_dataset_dir / "b_duplicate.jpg"
    too_small = config.raw_dataset_dir / "c_small.jpg"
    broken = config.raw_dataset_dir / "d_broken.jpg"

    _write_rgb_image(canonical, width=64, height=64, value=90)
    shutil.copyfile(canonical, duplicate)
    _write_rgb_image(too_small, width=8, height=8, value=120)
    broken.write_bytes(b"not-a-real-image")

    frame = DatasetPipeline(config).scan()

    rows = {row["file_name"]: row for row in frame.iter_rows(named=True)}

    assert frame.height == 4
    assert rows["a_valid.jpg"]["is_valid"] is True
    assert rows["a_valid.jpg"]["is_duplicate"] is False

    assert rows["b_duplicate.jpg"]["is_duplicate"] is True
    assert rows["b_duplicate.jpg"]["duplicate_of"] == rows["a_valid.jpg"]["sample_id"]
    assert rows["b_duplicate.jpg"]["is_valid"] is False
    assert rows["b_duplicate.jpg"]["quarantine_reason"] == "duplicate"

    assert rows["c_small.jpg"]["is_too_small"] is True
    assert rows["c_small.jpg"]["is_valid"] is False
    assert rows["c_small.jpg"]["quarantine_reason"] == "too_small"

    assert rows["d_broken.jpg"]["decode_ok"] is False
    assert rows["d_broken.jpg"]["is_valid"] is False
    assert rows["d_broken.jpg"]["quarantine_reason"] == "decode_failed"


def test_run_all_writes_manifest_and_reports(tmp_path: Path) -> None:
    config = _build_config(
        tmp_path,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    config.raw_dataset_dir.mkdir(parents=True)

    for index in range(10):
        _write_rgb_image(
            config.raw_dataset_dir / f"plastic_{index}.jpg",
            width=64,
            height=64,
            value=40 + index,
        )
        _write_rgb_image(
            config.raw_dataset_dir / f"glass_{index}.jpg",
            width=64,
            height=64,
            value=90 + index,
        )

    pipeline = DatasetPipeline(config)
    frame = pipeline.run_all()

    assert config.manifest_path.is_file()
    assert config.manifest_csv_path.is_file()
    assert config.summary_path.is_file()
    assert config.split_summary_path.is_file()
    assert config.quality_summary_path.is_file()
    assert config.extension_summary_path.is_file()
    assert config.label_summary_path.is_file()

    manifest_frame = pl.read_parquet(config.manifest_path)
    valid_frame = manifest_frame.filter(pl.col("is_valid"))
    split_counts = {
        (row["label"], row["split"]): row["count"]
        for row in valid_frame.group_by("label", "split").len(name="count").iter_rows(named=True)
    }

    assert frame.height == 20
    assert valid_frame.height == 20
    assert split_counts == {
        ("glass", "train"): 6,
        ("glass", "val"): 2,
        ("glass", "test"): 2,
        ("plastic", "train"): 6,
        ("plastic", "val"): 2,
        ("plastic", "test"): 2,
    }

    summary = json.loads(config.summary_path.read_text(encoding="utf-8"))
    assert summary["total_files"] == 20
    assert summary["valid_files"] == 20
    assert summary["duplicate_files"] == 0
    assert summary["label_counts"]["glass"] == 10
    assert summary["label_counts"]["plastic"] == 10


def test_frame_construction_handles_late_duplicate_string_values(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    pipeline = DatasetPipeline(config)

    records: list[dict[str, object | None]] = []
    for index in range(120):
        records.append(
            {
                "sample_id": f"sample_{index}",
                "image_path": f"/tmp/sample_{index}.jpg",
                "relative_path": f"sample_{index}.jpg",
                "file_name": f"sample_{index}.jpg",
                "extension": ".jpg",
                "file_size": 1024,
                "width": 64,
                "height": 64,
                "channels": 3,
                "decode_ok": True,
                "decode_error": None,
                "label": "unknown",
                "content_hash": f"hash_{index}",
                "is_duplicate": False,
                "duplicate_of": None,
                "is_too_small": False,
                "is_valid": True,
                "quarantine_reason": None,
                "split": None,
            }
        )

    records[-1]["is_duplicate"] = True
    records[-1]["duplicate_of"] = "sample_0"
    records[-1]["is_valid"] = False
    records[-1]["quarantine_reason"] = "duplicate"

    frame = pipeline._frame_from_records(records)

    assert frame.height == 120
    assert frame.get_column("duplicate_of").tail(1).item() == "sample_0"


def test_scan_infers_normalized_label_names(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    config.raw_dataset_dir.mkdir(parents=True)
    _write_rgb_image(
        config.raw_dataset_dir / "Miscellaneous Trash_1.jpg",
        width=64,
        height=64,
        value=80,
    )

    frame = DatasetPipeline(config).scan()
    row = frame.row(0, named=True)

    assert row["raw_label"] == "Miscellaneous Trash"
    assert row["label"] == "miscellaneous_trash"
