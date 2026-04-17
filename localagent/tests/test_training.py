from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from localagent.config import AgentPaths, DatasetPipelineConfig, TrainingConfig
from localagent.data import DatasetPipeline
from localagent.training import SUPPORTED_CNN_MODELS, WasteTrainer


def _write_rgb_image(path: Path, width: int, height: int, value: int) -> None:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Unable to write image: {path}")


def _build_two_class_manifest(
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
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        show_progress=False,
    )
    DatasetPipeline(pipeline_config).run_all()
    return pipeline_config


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
            cache_format: str = "png",
            force: bool = False,
            show_progress: bool = True,
        ) -> dict[str, object]:
            assert cache_format == "png"
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


def test_build_model_stub_supports_multiple_cnn_backbones(tmp_path: Path) -> None:
    trainer = WasteTrainer(
        AgentPaths(
            project_root=tmp_path,
            dataset_dir=tmp_path / "datasets",
            model_dir=tmp_path / "models",
            artifact_dir=tmp_path / "artifacts",
            log_dir=tmp_path / "logs",
            config_dir=tmp_path / "configs",
        ).ensure_layout(),
        TrainingConfig(pretrained_backbone=False),
    )

    for model_name in SUPPORTED_CNN_MODELS:
        trainer.config.model_name = model_name
        trainer.config.freeze_backbone = True
        model = trainer.build_model_stub(num_classes=3)

        if model_name == "resnet18":
            assert model.fc.out_features == 3
            frozen_backbone = [
                param.requires_grad
                for name, param in model.named_parameters()
                if not name.startswith("fc.")
            ]
            trainable_head = [
                param.requires_grad
                for name, param in model.named_parameters()
                if name.startswith("fc.")
            ]
        else:
            assert model.classifier[-1].out_features == 3
            frozen_backbone = [param.requires_grad for param in model.features.parameters()]
            trainable_head = [param.requires_grad for param in model.classifier.parameters()]

        assert frozen_backbone
        assert all(flag is False for flag in frozen_backbone)
        assert trainable_head
        assert any(flag is True for flag in trainable_head)
        assert trainer._last_model_metadata["model_name"] == model_name


def test_build_model_stub_rejects_unknown_cnn_backbone(tmp_path: Path) -> None:
    trainer = WasteTrainer(
        AgentPaths(
            project_root=tmp_path,
            dataset_dir=tmp_path / "datasets",
            model_dir=tmp_path / "models",
            artifact_dir=tmp_path / "artifacts",
            log_dir=tmp_path / "logs",
            config_dir=tmp_path / "configs",
        ).ensure_layout(),
        TrainingConfig(model_name="cnn_does_not_exist", pretrained_backbone=False),
    )

    try:
        trainer.build_model_stub(num_classes=3)
    except ValueError as error:
        assert "Unsupported model_name" in str(error)
    else:
        raise AssertionError("Expected ValueError for unsupported model_name")


def test_trainer_supports_raw_rust_cache_for_large_epoch_runs(tmp_path: Path) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    for index in range(4):
        _write_rgb_image(
            raw_dataset_dir / f"plastic_{index}.jpg",
            width=64,
            height=64,
            value=50 + index,
        )

    pipeline_config = DatasetPipelineConfig(
        raw_dataset_dir=raw_dataset_dir,
        manifest_dir=tmp_path / "artifacts" / "manifests",
        report_dir=tmp_path / "artifacts" / "reports",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        random_seed=13,
        show_progress=False,
    )
    manifest_frame = DatasetPipeline(pipeline_config).run_all()

    training_config = TrainingConfig(
        batch_size=2,
        epochs=1,
        num_workers=0,
        manifest_path=pipeline_config.manifest_path,
        checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
        cache_dir=tmp_path / "artifacts" / "cache",
        cache_format="raw",
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

    class StubRustAccelerationRaw:
        def is_available(self) -> bool:
            return True

        def prepare_image_cache(
            self,
            entries: list[tuple[str, str]],
            *,
            cache_dir: Path,
            failure_report_path: Path | None = None,
            image_size: int,
            cache_format: str = "png",
            force: bool = False,
            show_progress: bool = True,
        ) -> dict[str, object]:
            assert cache_format == "raw"
            del force, show_progress
            cache_dir.mkdir(parents=True, exist_ok=True)
            for sample_id, image_path in entries:
                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Unable to read {image_path}")
                resized = cv2.resize(image, (image_size, image_size))
                output_path = cache_dir / f"{sample_id}.raw"
                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).tofile(output_path)
            if failure_report_path is not None:
                failure_report_path.parent.mkdir(parents=True, exist_ok=True)
                failure_report_path.write_text("[]", encoding="utf-8")
            return {
                "total": len(entries),
                "processed": len(entries),
                "skipped": 0,
                "errors": 0,
                "cache_dir": str(cache_dir),
                "cache_format": cache_format,
                "failure_report_path": (
                    None if failure_report_path is None else str(failure_report_path)
                ),
                "image_size": image_size,
            }

    trainer.rust_acceleration = StubRustAccelerationRaw()

    cache_summary = trainer.warm_image_cache(manifest_frame=manifest_frame)

    assert cache_summary is not None
    cache_dir = Path(str(cache_summary["cache_dir"]))
    datasets, _ = trainer.build_datasets(manifest_frame=manifest_frame, cache_dir=cache_dir)
    image_path, use_cached = datasets["train"]._resolve_image_path(datasets["train"].records[0])
    tensor, label_index = datasets["train"][0]

    assert use_cached is True
    assert image_path.suffix == ".raw"
    assert tensor.shape == (3, training_config.image_size, training_config.image_size)
    assert label_index == 0


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
    assert Path(str(summary["latest_checkpoint_path"])).is_file()


def test_fit_can_resume_from_latest_checkpoint(tmp_path: Path) -> None:
    pipeline_config = _build_two_class_manifest(tmp_path / "first_run", per_class=4, random_seed=23)

    def build_trainer(config: TrainingConfig) -> WasteTrainer:
        return WasteTrainer(
            AgentPaths(
                project_root=tmp_path,
                dataset_dir=tmp_path / "datasets",
                model_dir=tmp_path / "models",
                artifact_dir=tmp_path / "artifacts",
                log_dir=tmp_path / "logs",
                config_dir=tmp_path / "configs",
            ).ensure_layout(),
            config,
        )

    initial_config = TrainingConfig(
        epochs=2,
        batch_size=2,
        num_workers=0,
        pretrained_backbone=False,
        manifest_path=pipeline_config.manifest_path,
        checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
        show_progress=False,
    )
    initial_trainer = build_trainer(initial_config)
    initial_trainer.warm_image_cache = lambda *args, **kwargs: None  # type: ignore[method-assign]
    initial_trainer.build_dataloaders = lambda *args, **kwargs: (  # type: ignore[method-assign]
        {"train": [0], "val": [0]},
        {"glass": 0, "plastic": 1},
    )

    def fake_build_model(num_classes: int = 4):
        initial_trainer._last_model_metadata = {
            "model_name": "mobilenet_v3_small",
            "pretrained_backbone_requested": False,
            "pretrained_backbone_loaded": False,
            "freeze_backbone": True,
        }
        return torch.nn.Linear(4, num_classes)

    initial_metrics = iter(
        [
            {"loss": 1.20, "accuracy": 0.40},
            {"loss": 1.00, "accuracy": 0.45},
            {"loss": 1.10, "accuracy": 0.42},
            {"loss": 0.90, "accuracy": 0.50},
        ]
    )
    initial_trainer.build_model_stub = fake_build_model  # type: ignore[method-assign]
    initial_trainer._run_epoch = lambda *args, **kwargs: next(initial_metrics)  # type: ignore[method-assign]

    first_summary = initial_trainer.fit()

    resumed_config = TrainingConfig(
        epochs=4,
        batch_size=2,
        num_workers=0,
        pretrained_backbone=False,
        manifest_path=pipeline_config.manifest_path,
        checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
        resume_from_checkpoint=Path(str(first_summary["latest_checkpoint_path"])),
        show_progress=False,
    )
    resumed_trainer = build_trainer(resumed_config)
    resumed_trainer.warm_image_cache = lambda *args, **kwargs: None  # type: ignore[method-assign]
    resumed_trainer.build_dataloaders = lambda *args, **kwargs: (  # type: ignore[method-assign]
        {"train": [0], "val": [0]},
        {"glass": 0, "plastic": 1},
    )
    resumed_trainer.build_model_stub = lambda num_classes=4: torch.nn.Linear(4, num_classes)  # type: ignore[method-assign]
    resumed_metrics = iter(
        [
            {"loss": 0.95, "accuracy": 0.52},
            {"loss": 0.85, "accuracy": 0.57},
            {"loss": 0.90, "accuracy": 0.55},
            {"loss": 0.80, "accuracy": 0.60},
        ]
    )
    resumed_trainer._run_epoch = lambda *args, **kwargs: next(resumed_metrics)  # type: ignore[method-assign]

    resumed_summary = resumed_trainer.fit()

    assert resumed_summary["epochs_completed"] == 4
    assert resumed_summary["best_epoch"] == 4
    assert resumed_summary["history"][0]["epoch"] == 1
    assert resumed_summary["history"][-1]["epoch"] == 4
    assert resumed_summary["resume_from_checkpoint"] == str(first_summary["latest_checkpoint_path"])


def test_fit_handles_keyboard_interrupt_and_saves_latest_checkpoint(tmp_path: Path) -> None:
    pipeline_config = _build_two_class_manifest(
        tmp_path / "interrupt_run",
        per_class=4,
        random_seed=29,
    )

    training_config = TrainingConfig(
        epochs=5,
        batch_size=2,
        num_workers=0,
        pretrained_backbone=False,
        manifest_path=pipeline_config.manifest_path,
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
        {"train": [0], "val": [0]},
        {"glass": 0, "plastic": 1},
    )
    trainer.build_model_stub = lambda num_classes=4: torch.nn.Linear(4, num_classes)  # type: ignore[method-assign]

    call_index = {"value": 0}

    def interrupting_epoch(*args, **kwargs):
        call_index["value"] += 1
        if call_index["value"] == 1:
            return {"loss": 1.10, "accuracy": 0.45}
        if call_index["value"] == 2:
            return {"loss": 0.95, "accuracy": 0.50}
        raise KeyboardInterrupt

    trainer._run_epoch = interrupting_epoch  # type: ignore[method-assign]

    summary = trainer.fit()

    assert summary["interrupted"] is True
    assert summary["epochs_completed"] == 1
    assert "Training interrupted by user" in str(summary["stop_reason"])
    assert Path(str(summary["latest_checkpoint_path"])).is_file()


def test_fit_writes_evaluation_report_and_confusion_matrix(tmp_path: Path) -> None:
    pipeline_config = _build_two_class_manifest(tmp_path / "eval_run", per_class=4, random_seed=31)

    training_config = TrainingConfig(
        epochs=1,
        batch_size=2,
        num_workers=0,
        pretrained_backbone=False,
        manifest_path=pipeline_config.manifest_path,
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
    trainer.build_model_stub = lambda num_classes=4: torch.nn.Linear(4, num_classes)  # type: ignore[method-assign]

    metrics = iter(
        [
            {"loss": 1.00, "accuracy": 0.50},
            {"loss": 0.90, "accuracy": 0.55},
            {
                "loss": 0.80,
                "accuracy": 0.75,
                "predictions": [0, 1, 1, 1],
                "targets": [0, 1, 0, 1],
            },
        ]
    )
    trainer._run_epoch = lambda *args, **kwargs: next(metrics)  # type: ignore[method-assign]

    summary = trainer.fit()
    report_path = Path(str(summary["evaluation_report_path"]))
    confusion_path = Path(str(summary["confusion_matrix_path"]))
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.is_file()
    assert confusion_path.is_file()
    assert summary["evaluation_summary"]["accuracy"] == 0.75
    assert report_payload["per_class"]["glass"]["support"] == 2
    assert report_payload["confusion_matrix"][0][1] == 1


def test_fit_prints_epoch_progress_when_progress_bars_are_disabled(
    tmp_path: Path,
    capsys,
) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    for index in range(4):
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
        random_seed=7,
        show_progress=False,
    )
    DatasetPipeline(pipeline_config).run_all()

    training_config = TrainingConfig(
        epochs=3,
        batch_size=2,
        num_workers=0,
        pretrained_backbone=False,
        manifest_path=pipeline_config.manifest_path,
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
    trainer.build_model_stub = lambda num_classes=4: torch.nn.Linear(4, num_classes)  # type: ignore[method-assign]

    metric_sequence = iter(
        [
            {"loss": 1.00, "accuracy": 0.50},
            {"loss": 0.90, "accuracy": 0.55},
            {"loss": 0.95, "accuracy": 0.52},
            {"loss": 0.85, "accuracy": 0.58},
            {"loss": 0.92, "accuracy": 0.53},
            {"loss": 0.82, "accuracy": 0.60},
            {"loss": 0.80, "accuracy": 0.61},
        ]
    )
    trainer._run_epoch = lambda *args, **kwargs: next(metric_sequence)  # type: ignore[method-assign]

    trainer.fit()
    captured = capsys.readouterr().out

    assert "Progress bars disabled; reporting text progress snapshots" in captured
    assert "epoch 1/3 | 33.33%" in captured
    assert "epoch 2/3 | 66.67%" in captured
    assert "epoch 3/3 | 100.00%" in captured


def test_run_epoch_prints_batch_snapshots_when_progress_bars_are_disabled(
    tmp_path: Path,
    capsys,
) -> None:
    training_config = TrainingConfig(
        batch_size=1,
        epochs=1,
        num_workers=0,
        pretrained_backbone=False,
        manifest_path=tmp_path / "artifacts" / "manifests" / "dataset_manifest.parquet",
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

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, 2),
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = [
        (torch.randn(1, 3, 4, 4), torch.tensor([0])),
        (torch.randn(1, 3, 4, 4), torch.tensor([1])),
        (torch.randn(1, 3, 4, 4), torch.tensor([0])),
    ]

    trainer._run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        training=True,
        stage_name="train 1/1000",
    )
    captured = capsys.readouterr().out

    assert "train 1/1000 | batch 1/3 | 33.33%" in captured
    assert "train 1/1000 | batch 2/3 | 66.67%" in captured
    assert "train 1/1000 | batch 3/3 | 100.00%" in captured


def test_build_dataloaders_can_balance_classes_with_sampler_bias(tmp_path: Path) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    for index in range(6):
        _write_rgb_image(
            raw_dataset_dir / f"plastic_{index}.jpg",
            width=64,
            height=64,
            value=40 + index,
        )
    _write_rgb_image(raw_dataset_dir / "glass_0.jpg", width=64, height=64, value=90)

    pipeline_config = DatasetPipelineConfig(
        raw_dataset_dir=raw_dataset_dir,
        manifest_dir=tmp_path / "artifacts" / "manifests",
        report_dir=tmp_path / "artifacts" / "reports",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        random_seed=21,
        show_progress=False,
    )
    manifest_frame = DatasetPipeline(pipeline_config).run_all()

    training_config = TrainingConfig(
        batch_size=2,
        epochs=1,
        num_workers=0,
        manifest_path=pipeline_config.manifest_path,
        class_bias_strategy="sampler",
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

    loaders, _ = trainer.build_dataloaders(manifest_frame=manifest_frame)
    weight_map = trainer.class_weight_map(manifest_frame)

    assert loaders["train"].sampler.__class__.__name__ == "WeightedRandomSampler"
    assert weight_map["glass"] > weight_map["plastic"]
    assert trainer.train_imbalance_ratio(manifest_frame) > 1.0


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
            cache_format: str = "png",
            force: bool = False,
            show_progress: bool = True,
        ) -> dict[str, object]:
            del cache_dir, image_size, cache_format, force, show_progress
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
