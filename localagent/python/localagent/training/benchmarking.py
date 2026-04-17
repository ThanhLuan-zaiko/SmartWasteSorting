from __future__ import annotations

import ctypes
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localagent.config import TrainingConfig

SUPPORTED_TRAINING_BACKENDS = ("pytorch", "rust_tch")


@dataclass(slots=True)
class ExperimentSpec:
    schema_version: int
    experiment_name: str
    training_backend: str
    training_preset: str | None
    model_name: str
    pretrained_backbone: bool
    freeze_backbone: bool
    image_size: int
    batch_size: int
    epochs: int
    num_workers: int
    device: str
    learning_rate: float
    weight_decay: float
    manifest_path: str
    checkpoint_dir: str
    labels_output_path: str
    onnx_output_path: str
    cache_dir: str
    cache_format: str
    normalization_preset: str
    class_bias_strategy: str
    use_rust_image_cache: bool
    force_rebuild_cache: bool
    onnx_opset: int
    export_batch_size: int
    early_stopping: dict[str, Any]

    @classmethod
    def from_training_config(cls, config: TrainingConfig) -> ExperimentSpec:
        return cls(
            schema_version=1,
            experiment_name=config.experiment_name,
            training_backend=config.training_backend,
            training_preset=config.training_preset,
            model_name=config.model_name,
            pretrained_backbone=config.pretrained_backbone,
            freeze_backbone=config.freeze_backbone,
            image_size=config.image_size,
            batch_size=config.batch_size,
            epochs=config.epochs,
            num_workers=config.num_workers,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            manifest_path=str(config.manifest_path),
            checkpoint_dir=str(config.checkpoint_dir),
            labels_output_path=str(config.labels_output_path),
            onnx_output_path=str(config.onnx_output_path),
            cache_dir=str(config.cache_dir),
            cache_format=config.cache_format,
            normalization_preset=config.normalization_preset,
            class_bias_strategy=config.class_bias_strategy,
            use_rust_image_cache=config.use_rust_image_cache,
            force_rebuild_cache=config.force_rebuild_cache,
            onnx_opset=config.onnx_opset,
            export_batch_size=config.export_batch_size,
            early_stopping={
                "enabled": config.enable_early_stopping,
                "patience": config.early_stopping_patience,
                "min_delta": config.early_stopping_min_delta,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "training_backend": self.training_backend,
            "training_preset": self.training_preset,
            "model_name": self.model_name,
            "pretrained_backbone": self.pretrained_backbone,
            "freeze_backbone": self.freeze_backbone,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "num_workers": self.num_workers,
            "device": self.device,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "manifest_path": self.manifest_path,
            "checkpoint_dir": self.checkpoint_dir,
            "labels_output_path": self.labels_output_path,
            "onnx_output_path": self.onnx_output_path,
            "cache_dir": self.cache_dir,
            "cache_format": self.cache_format,
            "normalization_preset": self.normalization_preset,
            "class_bias_strategy": self.class_bias_strategy,
            "use_rust_image_cache": self.use_rust_image_cache,
            "force_rebuild_cache": self.force_rebuild_cache,
            "onnx_opset": self.onnx_opset,
            "export_batch_size": self.export_batch_size,
            "early_stopping": self.early_stopping,
        }

    def write_json(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path


class ProcessMemorySampler:
    def __init__(self, poll_interval_seconds: float = 0.05) -> None:
        self.poll_interval_seconds = poll_interval_seconds
        self.peak_rss_bytes = current_rss_bytes()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ProcessMemorySampler:
        if self.peak_rss_bytes is None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval_seconds * 4.0)
        current = current_rss_bytes()
        if current is not None:
            self.peak_rss_bytes = max(self.peak_rss_bytes or 0, current)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            current = current_rss_bytes()
            if current is not None:
                self.peak_rss_bytes = max(self.peak_rss_bytes or 0, current)
            time.sleep(self.poll_interval_seconds)


def current_rss_bytes() -> int | None:
    if sys.platform.startswith("win"):
        return _windows_working_set_bytes()
    return _posix_peak_rss_bytes()


def bytes_to_megabytes(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def compare_benchmark_reports(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = _read_json(left_path)
    right = _read_json(right_path)

    def _stage_duration(payload: dict[str, Any], stage_name: str) -> float | None:
        stages = payload.get("stages")
        if not isinstance(stages, dict):
            return None
        stage = stages.get(stage_name)
        if not isinstance(stage, dict):
            return None
        value = stage.get("duration_seconds")
        return float(value) if isinstance(value, (int, float)) else None

    def _metric(payload: dict[str, Any], key: str) -> float | None:
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            return None
        value = metrics.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    comparison = {
        "schema_version": 1,
        "left_report_path": str(left_path),
        "right_report_path": str(right_path),
        "left_backend": left.get("training_backend"),
        "right_backend": right.get("training_backend"),
        "left_experiment_name": left.get("experiment_name"),
        "right_experiment_name": right.get("experiment_name"),
        "duration_delta_seconds": _delta(
            _metric(left, "total_duration_seconds"),
            _metric(right, "total_duration_seconds"),
        ),
        "fit_stage_delta_seconds": _delta(
            _stage_duration(left, "fit"),
            _stage_duration(right, "fit"),
        ),
        "accuracy_delta": _delta(_metric(left, "accuracy"), _metric(right, "accuracy")),
        "macro_f1_delta": _delta(_metric(left, "macro_f1"), _metric(right, "macro_f1")),
        "weighted_f1_delta": _delta(
            _metric(left, "weighted_f1"),
            _metric(right, "weighted_f1"),
        ),
    }
    return comparison


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return right - left


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark report must be a JSON object: {path}")
    return payload


def _windows_working_set_bytes() -> int | None:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    success = get_process_memory_info(
        get_current_process(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not success:
        return None
    return int(counters.WorkingSetSize)


def _posix_peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - platform specific
        return None

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024
