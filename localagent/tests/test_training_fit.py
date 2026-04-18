from __future__ import annotations

from pathlib import Path

import torch
from localagent.config import DatasetPipelineConfig, TrainingConfig
from localagent.data import DatasetPipeline

from tests.training_support import build_trainer, build_two_class_manifest, write_rgb_image


def test_fit_stops_early_when_validation_loss_stalls(tmp_path: Path) -> None:
    pipeline_config = build_two_class_manifest(tmp_path, per_class=6, random_seed=19)

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
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
        ),
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
    pipeline_config = build_two_class_manifest(tmp_path / "first_run", per_class=4, random_seed=23)

    initial_trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            pretrained_backbone=False,
            manifest_path=pipeline_config.manifest_path,
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            show_progress=False,
        ),
    )
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

    resumed_trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            epochs=4,
            batch_size=2,
            num_workers=0,
            pretrained_backbone=False,
            manifest_path=pipeline_config.manifest_path,
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            resume_from_checkpoint=Path(str(first_summary["latest_checkpoint_path"])),
            show_progress=False,
        ),
    )
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
    pipeline_config = build_two_class_manifest(
        tmp_path / "interrupt_run",
        per_class=4,
        random_seed=29,
    )

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            epochs=5,
            batch_size=2,
            num_workers=0,
            pretrained_backbone=False,
            manifest_path=pipeline_config.manifest_path,
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            show_progress=False,
        ),
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


def test_fit_prints_epoch_progress_when_progress_bars_are_disabled(
    tmp_path: Path,
    capsys,
) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    for index in range(4):
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
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=7,
        show_progress=False,
    )
    DatasetPipeline(pipeline_config).run_all()

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            epochs=3,
            batch_size=2,
            num_workers=0,
            pretrained_backbone=False,
            manifest_path=pipeline_config.manifest_path,
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            show_progress=False,
        ),
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
    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            batch_size=1,
            epochs=1,
            num_workers=0,
            pretrained_backbone=False,
            manifest_path=tmp_path / "artifacts" / "manifests" / "dataset_manifest.parquet",
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            show_progress=False,
        ),
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
