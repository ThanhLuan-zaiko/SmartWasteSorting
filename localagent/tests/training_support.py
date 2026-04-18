from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from localagent.config import AgentPaths, DatasetPipelineConfig, TrainingConfig
from localagent.data import DatasetPipeline
from localagent.training import WasteTrainer


def build_agent_paths(tmp_path: Path) -> AgentPaths:
    return AgentPaths(
        project_root=tmp_path,
        dataset_dir=tmp_path / "datasets",
        model_dir=tmp_path / "models",
        artifact_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        config_dir=tmp_path / "configs",
    ).ensure_layout()


def build_trainer(tmp_path: Path, training_config: TrainingConfig) -> WasteTrainer:
    return WasteTrainer(build_agent_paths(tmp_path), training_config)


def write_rgb_image(path: Path, width: int, height: int, value: int) -> None:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Unable to write image: {path}")


def build_two_class_manifest(
    tmp_path: Path,
    *,
    per_class: int = 6,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    random_seed: int = 19,
) -> DatasetPipelineConfig:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True, exist_ok=True)

    for index in range(per_class):
        write_rgb_image(
            raw_dataset_dir / f"glass_{index}.jpg",
            width=64,
            height=64,
            value=20 + index,
        )
        write_rgb_image(
            raw_dataset_dir / f"plastic_{index}.jpg",
            width=64,
            height=64,
            value=60 + index,
        )

    pipeline_config = DatasetPipelineConfig(
        raw_dataset_dir=raw_dataset_dir,
        manifest_dir=tmp_path / "artifacts" / "manifests",
        report_dir=tmp_path / "artifacts" / "reports",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        show_progress=False,
    )
    DatasetPipeline(pipeline_config).run_all()
    return pipeline_config
