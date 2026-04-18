from __future__ import annotations

import json
from pathlib import Path

import cv2
from localagent.config import DatasetPipelineConfig, TrainingConfig
from localagent.data import DatasetPipeline
from localagent.training import SUPPORTED_CNN_MODELS

from tests.training_support import build_trainer, write_rgb_image


def test_trainer_builds_datasets_and_exports_labels(tmp_path: Path) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    for index in range(10):
        write_rgb_image(
            raw_dataset_dir / f"plastic_{index}.jpg",
            width=64,
            height=64,
            value=30 + index,
        )
        write_rgb_image(
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

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            batch_size=4,
            epochs=1,
            num_workers=0,
            manifest_path=pipeline_config.manifest_path,
            labels_output_path=tmp_path / "models" / "labels.json",
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
        ),
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
        write_rgb_image(
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

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            batch_size=2,
            epochs=1,
            num_workers=0,
            manifest_path=pipeline_config.manifest_path,
            labels_output_path=tmp_path / "models" / "labels.json",
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            cache_dir=tmp_path / "artifacts" / "cache",
            show_progress=False,
        ),
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
    trainer = build_trainer(tmp_path, TrainingConfig(pretrained_backbone=False))

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
    trainer = build_trainer(
        tmp_path,
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
        write_rgb_image(
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

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            batch_size=2,
            epochs=1,
            num_workers=0,
            manifest_path=pipeline_config.manifest_path,
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            cache_dir=tmp_path / "artifacts" / "cache",
            cache_format="raw",
            show_progress=False,
        ),
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
    assert tensor.shape == (3, trainer.config.image_size, trainer.config.image_size)
    assert label_index == 0


def test_build_dataloaders_can_balance_classes_with_sampler_bias(tmp_path: Path) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    for index in range(6):
        write_rgb_image(
            raw_dataset_dir / f"plastic_{index}.jpg",
            width=64,
            height=64,
            value=40 + index,
        )
    write_rgb_image(raw_dataset_dir / "glass_0.jpg", width=64, height=64, value=90)

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

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            batch_size=2,
            epochs=1,
            num_workers=0,
            manifest_path=pipeline_config.manifest_path,
            class_bias_strategy="sampler",
            show_progress=False,
        ),
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
    write_rgb_image(failing_image, width=64, height=64, value=77)

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

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            batch_size=1,
            epochs=1,
            num_workers=0,
            manifest_path=pipeline_config.manifest_path,
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            cache_dir=tmp_path / "artifacts" / "cache",
            show_progress=False,
        ),
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
                "image_size": trainer.config.image_size,
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
