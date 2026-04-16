from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from localagent.config import AgentPaths, DatasetPipelineConfig, TrainingConfig
from localagent.data import DatasetPipeline
from localagent.training import WasteTrainer


def _write_rgb_image(path: Path, width: int, height: int, value: int) -> None:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Unable to write image: {path}")


def test_trainer_builds_datasets_and_exports_labels(tmp_path: Path) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    for index in range(10):
        _write_rgb_image(
            raw_dataset_dir / f"plastic_{index}.jpg",
            width=64,
            height=64,
            value=30 + index,
        )
        _write_rgb_image(
            raw_dataset_dir / f"glass_{index}.jpg",
            width=64,
            height=64,
            value=90 + index,
        )

    pipeline_config = DatasetPipelineConfig(
        raw_dataset_dir=raw_dataset_dir,
        manifest_dir=tmp_path / "artifacts" / "manifests",
        report_dir=tmp_path / "artifacts" / "reports",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=3,
    )
    manifest_frame = DatasetPipeline(pipeline_config).run_all()

    training_config = TrainingConfig(
        batch_size=4,
        epochs=1,
        num_workers=0,
        manifest_path=pipeline_config.manifest_path,
        labels_output_path=tmp_path / "models" / "labels.json",
        checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
    )
    trainer = WasteTrainer(
        AgentPaths(
            project_root=tmp_path,
            dataset_dir=tmp_path / "datasets",
            model_dir=tmp_path / "models",
            artifact_dir=tmp_path / "artifacts",
            log_dir=tmp_path / "logs",
            config_dir=tmp_path / "configs",
        ).ensure_layout(),
        training_config,
    )

    summary = trainer.summarize_training_plan(manifest_frame=manifest_frame)
    datasets, label_to_index = trainer.build_datasets(manifest_frame=manifest_frame)
    labels_path = trainer.export_label_index(manifest_frame=manifest_frame)

    assert summary["num_classes"] == 2
    assert summary["split_counts"] == {"train": 12, "val": 4, "test": 4}
    assert label_to_index == {"glass": 0, "plastic": 1}
    assert len(datasets["train"]) == 12
    assert len(datasets["val"]) == 4
    assert len(datasets["test"]) == 4
    assert json.loads(labels_path.read_text(encoding="utf-8")) == ["glass", "plastic"]
