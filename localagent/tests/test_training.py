from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
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


def test_trainer_warm_cache_prefers_cached_images(tmp_path: Path) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    for index in range(6):
        _write_rgb_image(
            raw_dataset_dir / f"plastic_{index}.jpg",
            width=64,
            height=64,
            value=40 + index,
        )

    pipeline_config = DatasetPipelineConfig(
        raw_dataset_dir=raw_dataset_dir,
        manifest_dir=tmp_path / "artifacts" / "manifests",
        report_dir=tmp_path / "artifacts" / "reports",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=11,
        show_progress=False,
    )
    manifest_frame = DatasetPipeline(pipeline_config).run_all()

    training_config = TrainingConfig(
        batch_size=2,
        epochs=1,
        num_workers=0,
        manifest_path=pipeline_config.manifest_path,
        labels_output_path=tmp_path / "models" / "labels.json",
        checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
        cache_dir=tmp_path / "artifacts" / "cache",
        show_progress=False,
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

    class StubRustAcceleration:
        def is_available(self) -> bool:
            return True

        def prepare_image_cache(
            self,
            entries: list[tuple[str, str]],
            *,
            cache_dir: Path,
            failure_report_path: Path | None = None,
            image_size: int,
            force: bool = False,
            show_progress: bool = True,
        ) -> dict[str, object]:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for sample_id, image_path in entries:
                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Unable to read {image_path}")
                resized = cv2.resize(image, (image_size, image_size))
                output_path = cache_dir / f"{sample_id}.png"
                if force or not output_path.exists():
                    if not cv2.imwrite(str(output_path), resized):
                        raise RuntimeError(f"Unable to write {output_path}")
            if failure_report_path is not None:
                failure_report_path.parent.mkdir(parents=True, exist_ok=True)
                failure_report_path.write_text("[]", encoding="utf-8")
            return {
                "total": len(entries),
                "processed": len(entries),
                "skipped": 0,
                "errors": 0,
                "cache_dir": str(cache_dir),
                "failure_report_path": (
                    None if failure_report_path is None else str(failure_report_path)
                ),
                "image_size": image_size,
            }

    trainer.rust_acceleration = StubRustAcceleration()

    cache_summary = trainer.warm_image_cache(manifest_frame=manifest_frame)

    assert cache_summary is not None
    cache_dir = Path(str(cache_summary["cache_dir"]))
    datasets, _ = trainer.build_datasets(manifest_frame=manifest_frame, cache_dir=cache_dir)
    image_path, use_cached = datasets["train"]._resolve_image_path(datasets["train"].records[0])

    assert use_cached is True
    assert image_path.is_file()
    assert image_path.suffix == ".png"


def test_fit_stops_early_when_validation_loss_stalls(tmp_path: Path) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    for index in range(6):
        _write_rgb_image(
            raw_dataset_dir / f"glass_{index}.jpg",
            width=64,
            height=64,
            value=20 + index,
        )
        _write_rgb_image(
            raw_dataset_dir / f"plastic_{index}.jpg",
            width=64,
            height=64,
            value=60 + index,
        )

    pipeline_config = DatasetPipelineConfig(
        raw_dataset_dir=raw_dataset_dir,
        manifest_dir=tmp_path / "artifacts" / "manifests",
        report_dir=tmp_path / "artifacts" / "reports",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=19,
        show_progress=False,
    )
    DatasetPipeline(pipeline_config).run_all()

    training_config = TrainingConfig(
        epochs=10,
        batch_size=2,
        num_workers=0,
        pretrained_backbone=False,
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
        manifest_path=pipeline_config.manifest_path,
        labels_output_path=tmp_path / "models" / "labels.json",
        checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
        show_progress=False,
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

    trainer.warm_image_cache = lambda *args, **kwargs: None  # type: ignore[method-assign]
    trainer.build_dataloaders = lambda *args, **kwargs: (  # type: ignore[method-assign]
        {"train": [0], "val": [0], "test": [0]},
        {"glass": 0, "plastic": 1},
    )

    def fake_build_model(num_classes: int = 4):
        trainer._last_model_metadata = {
            "model_name": "mobilenet_v3_small",
            "pretrained_backbone_requested": False,
            "pretrained_backbone_loaded": False,
            "freeze_backbone": True,
        }
        return torch.nn.Linear(4, num_classes)

    metric_sequence = iter(
        [
            {"loss": 1.20, "accuracy": 0.30},
            {"loss": 1.00, "accuracy": 0.35},
            {"loss": 1.10, "accuracy": 0.40},
            {"loss": 0.90, "accuracy": 0.45},
            {"loss": 1.05, "accuracy": 0.42},
            {"loss": 0.91, "accuracy": 0.44},
            {"loss": 1.00, "accuracy": 0.43},
            {"loss": 0.92, "accuracy": 0.44},
            {"loss": 0.93, "accuracy": 0.46},
        ]
    )

    trainer.build_model_stub = fake_build_model  # type: ignore[method-assign]
    trainer._run_epoch = lambda *args, **kwargs: next(metric_sequence)  # type: ignore[method-assign]

    summary = trainer.fit()

    assert summary["stopped_early"] is True
    assert summary["best_epoch"] == 2
    assert summary["epochs_completed"] == 4
    assert "Early stopping triggered" in str(summary["stop_reason"])
    assert Path(str(summary["checkpoint_path"])).is_file()


def test_warm_cache_recovers_rust_failures_with_opencv(tmp_path: Path) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)
    failing_image = raw_dataset_dir / "paper_1.jpg"
    _write_rgb_image(failing_image, width=64, height=64, value=77)

    pipeline_config = DatasetPipelineConfig(
        raw_dataset_dir=raw_dataset_dir,
        manifest_dir=tmp_path / "artifacts" / "manifests",
        report_dir=tmp_path / "artifacts" / "reports",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        random_seed=5,
        show_progress=False,
    )
    manifest_frame = DatasetPipeline(pipeline_config).run_all()

    training_config = TrainingConfig(
        batch_size=1,
        epochs=1,
        num_workers=0,
        manifest_path=pipeline_config.manifest_path,
        checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
        cache_dir=tmp_path / "artifacts" / "cache",
        show_progress=False,
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

    class StubRustFailureAcceleration:
        def is_available(self) -> bool:
            return True

        def prepare_image_cache(
            self,
            entries: list[tuple[str, str]],
            *,
            cache_dir: Path,
            failure_report_path: Path | None = None,
            image_size: int,
            force: bool = False,
            show_progress: bool = True,
        ) -> dict[str, object]:
            del cache_dir, image_size, force, show_progress
            assert failure_report_path is not None
            failure_report_path.parent.mkdir(parents=True, exist_ok=True)
            failure_report_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": entries[0][0],
                            "image_path": entries[0][1],
                            "error": "rust decode failed",
                        }
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {
                "total": len(entries),
                "processed": 0,
                "skipped": 0,
                "errors": 1,
                "cache_dir": str(trainer._cache_dir()),
                "failure_report_path": str(failure_report_path),
                "image_size": training_config.image_size,
            }

    trainer.rust_acceleration = StubRustFailureAcceleration()

    summary = trainer.warm_image_cache(manifest_frame=manifest_frame)

    assert summary is not None
    assert summary["processed"] == 1
    assert summary["errors"] == 0
    assert summary["fallback_processed"] == 1
    assert summary["fallback_errors"] == 0
    report_path = Path(str(summary["failure_report_path"]))
    assert json.loads(report_path.read_text(encoding="utf-8")) == []
    sample_id = str(manifest_frame.row(0, named=True)["sample_id"])
    assert (trainer._cache_dir() / f"{sample_id}.png").is_file()
