from __future__ import annotations

from pathlib import Path
from typing import Any

from localagent.config import AgentPaths, TrainingConfig
from localagent.data import DatasetIndex


class WasteTrainer:
    def __init__(self, paths: AgentPaths, config: TrainingConfig) -> None:
        self.paths = paths
        self.config = config

    def summarize_training_plan(self, dataset_index: DatasetIndex | None = None) -> dict[str, Any]:
        return {
            "experiment_name": self.config.experiment_name,
            "image_size": self.config.image_size,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "dataset_size": len(dataset_index) if dataset_index is not None else 0,
            "artifact_dir": str(self.paths.artifact_dir),
        }

    def build_model_stub(self) -> Any:
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 4),
        )

    def export_onnx_stub(self, output_path: Path | None = None) -> Path:
        destination = output_path or self.paths.model_dir / "waste_classifier.onnx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch(exist_ok=True)
        return destination
