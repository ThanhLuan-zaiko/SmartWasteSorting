from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class AgentPaths:
    project_root: Path = field(default_factory=_project_root)
    dataset_dir: Path = field(default_factory=lambda: _project_root() / "datasets")
    model_dir: Path = field(default_factory=lambda: _project_root() / "models")
    artifact_dir: Path = field(default_factory=lambda: _project_root() / "artifacts")
    log_dir: Path = field(default_factory=lambda: _project_root() / "logs")
    config_dir: Path = field(default_factory=lambda: _project_root() / "configs")

    def ensure_layout(self) -> AgentPaths:
        for path in (
            self.dataset_dir,
            self.model_dir,
            self.artifact_dir,
            self.log_dir,
            self.config_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(slots=True)
class RuntimeConfig:
    model_path: Path = field(
        default_factory=lambda: _project_root() / "models" / "waste_classifier.onnx"
    )
    labels_path: Path = field(default_factory=lambda: _project_root() / "models" / "labels.json")
    device: str = "cpu"
    score_threshold: float = 0.45
    server_host: str = "127.0.0.1"
    server_port: int = 8080

    @property
    def base_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"


@dataclass(slots=True)
class TrainingConfig:
    experiment_name: str = "baseline-waste-sorter"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    num_workers: int = 4
    learning_rate: float = 1e-3
