from __future__ import annotations

import math
import sys
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
    training_preset: str | None = None
    experiment_name: str = "baseline-waste-sorter"
    model_name: str = "mobilenet_v3_small"
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    num_workers: int = 0 if sys.platform.startswith("win") else 4
    prefetch_factor: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    manifest_path: Path = field(
        default_factory=lambda: _project_root()
        / "artifacts"
        / "manifests"
        / "dataset_manifest.parquet"
    )
    labels_output_path: Path = field(
        default_factory=lambda: _project_root() / "models" / "labels.json"
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "checkpoints"
    )
    resume_from_checkpoint: Path | None = None
    cache_dir: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "cache" / "training"
    )
    cache_format: str = "png"
    use_rust_image_cache: bool = True
    force_rebuild_cache: bool = False
    class_bias_strategy: str = "none"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-3
    enable_early_stopping: bool = True
    show_progress: bool = True
    device: str = "auto"


@dataclass(slots=True)
class DatasetPipelineConfig:
    raw_dataset_dir: Path = field(default_factory=lambda: _project_root() / "dataset")
    manifest_dir: Path = field(default_factory=lambda: _project_root() / "artifacts" / "manifests")
    report_dir: Path = field(default_factory=lambda: _project_root() / "artifacts" / "reports")
    manifest_name: str = "dataset_manifest.parquet"
    manifest_csv_name: str = "dataset_manifest.csv"
    min_width: int = 32
    min_height: int = 32
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    hash_algorithm: str = "sha256"
    infer_labels_from_filename: bool = True
    unknown_label: str = "unknown"
    show_progress: bool = True

    @property
    def manifest_path(self) -> Path:
        return self.manifest_dir / self.manifest_name

    @property
    def manifest_csv_path(self) -> Path:
        return self.manifest_dir / self.manifest_csv_name

    @property
    def summary_path(self) -> Path:
        return self.report_dir / "summary.json"

    @property
    def split_summary_path(self) -> Path:
        return self.report_dir / "split_summary.csv"

    @property
    def quality_summary_path(self) -> Path:
        return self.report_dir / "quality_summary.csv"

    @property
    def extension_summary_path(self) -> Path:
        return self.report_dir / "extension_summary.csv"

    @property
    def label_summary_path(self) -> Path:
        return self.report_dir / "label_summary.csv"

    def ensure_layout(self) -> DatasetPipelineConfig:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        return self

    def validate(self) -> DatasetPipelineConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if self.min_width <= 0 or self.min_height <= 0:
            raise ValueError("min_width and min_height must be greater than zero")
        return self
