from __future__ import annotations

import json
from pathlib import Path

import torch
from localagent.config import TrainingConfig
from localagent.training import compare_benchmark_reports

from tests.training_support import build_trainer, build_two_class_manifest


def test_fit_writes_evaluation_report_and_confusion_matrix(tmp_path: Path) -> None:
    pipeline_config = build_two_class_manifest(tmp_path / "eval_run", per_class=4, random_seed=31)

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            epochs=1,
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


def test_evaluate_uses_saved_checkpoint_and_writes_report(tmp_path: Path) -> None:
    pipeline_config = build_two_class_manifest(
        tmp_path / "evaluate_run",
        per_class=4,
        random_seed=37,
    )

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            epochs=1,
            batch_size=2,
            num_workers=0,
            pretrained_backbone=False,
            image_size=8,
            manifest_path=pipeline_config.manifest_path,
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            show_progress=False,
        ),
    )

    trainer.warm_image_cache = lambda *args, **kwargs: None  # type: ignore[method-assign]
    trainer.build_dataloaders = lambda *args, **kwargs: (  # type: ignore[method-assign]
        {"test": [0]},
        {"glass": 0, "plastic": 1},
    )

    def fake_build_model(num_classes: int = 4):
        trainer._last_model_metadata = {
            "model_name": "mobilenet_v3_small",
            "pretrained_backbone_requested": False,
            "pretrained_backbone_loaded": False,
            "freeze_backbone": True,
            "image_size": trainer.config.image_size,
            "normalization_preset": trainer.config.normalization_preset,
        }
        return torch.nn.Linear(4, num_classes)

    trainer.build_model_stub = fake_build_model  # type: ignore[method-assign]
    model = fake_build_model(num_classes=2)
    trainer._save_training_checkpoint(
        checkpoint_path=trainer._checkpoint_path(),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=None,
        labels=["glass", "plastic"],
        history=[],
        best_epoch=1,
        best_loss=0.5,
        best_model_state_dict=model.state_dict(),
        last_completed_epoch=1,
        target_epochs=1,
        checkpoint_kind="best",
        interrupted=False,
        stopped_early=False,
        stop_reason=None,
    )
    trainer._run_epoch = lambda *args, **kwargs: {  # type: ignore[method-assign]
        "loss": 0.25,
        "accuracy": 0.75,
        "predictions": [0, 1, 1, 1],
        "targets": [0, 1, 0, 1],
    }

    summary = trainer.evaluate()
    report_path = Path(str(summary["evaluation_report_path"]))
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.is_file()
    assert summary["accuracy"] == 0.75
    assert summary["split"] == "test"
    assert report_payload["loss"] == 0.25
    assert report_payload["confusion_matrix"][0][1] == 1


def test_export_onnx_writes_manifest_and_export_report(tmp_path: Path) -> None:
    pipeline_config = build_two_class_manifest(
        tmp_path / "export_run",
        per_class=4,
        random_seed=41,
    )

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            epochs=1,
            batch_size=2,
            num_workers=0,
            pretrained_backbone=False,
            image_size=8,
            manifest_path=pipeline_config.manifest_path,
            labels_output_path=tmp_path / "models" / "labels.json",
            onnx_output_path=tmp_path / "models" / "waste_classifier.onnx",
            model_manifest_output_path=tmp_path / "models" / "model_manifest.json",
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            show_progress=False,
        ),
    )

    def fake_build_model(num_classes: int = 4):
        trainer._last_model_metadata = {
            "model_name": "linear_export_stub",
            "pretrained_backbone_requested": False,
            "pretrained_backbone_loaded": False,
            "freeze_backbone": True,
            "image_size": trainer.config.image_size,
            "normalization_preset": trainer.config.normalization_preset,
        }
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                3 * trainer.config.image_size * trainer.config.image_size,
                num_classes,
            ),
        )

    trainer.build_model_stub = fake_build_model  # type: ignore[method-assign]
    model = fake_build_model(num_classes=2)
    trainer._save_training_checkpoint(
        checkpoint_path=trainer._checkpoint_path(),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=None,
        labels=["glass", "plastic"],
        history=[],
        best_epoch=1,
        best_loss=0.4,
        best_model_state_dict=model.state_dict(),
        last_completed_epoch=1,
        target_epochs=1,
        checkpoint_kind="best",
        interrupted=False,
        stopped_early=False,
        stop_reason=None,
    )

    summary = trainer.export_onnx()
    export_report = json.loads(trainer._export_report_path().read_text(encoding="utf-8"))
    model_manifest = json.loads(trainer._model_manifest_path().read_text(encoding="utf-8"))
    labels_payload = json.loads(trainer.config.labels_output_path.read_text(encoding="utf-8"))

    assert Path(str(summary["onnx_path"])).is_file()
    assert summary["verification"]["verified"] is True
    assert export_report["onnx_path"] == str(trainer.config.onnx_output_path)
    assert model_manifest["onnx"]["verification"]["verified"] is True
    assert model_manifest["labels"] == ["glass", "plastic"]
    assert labels_payload == ["glass", "plastic"]


def test_build_artifact_report_bundles_existing_training_outputs(tmp_path: Path) -> None:
    pipeline_config = build_two_class_manifest(
        tmp_path / "bundle_run",
        per_class=2,
        random_seed=43,
    )
    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            manifest_path=pipeline_config.manifest_path,
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            show_progress=False,
        ),
    )

    trainer._write_json(trainer._training_report_path(), {"epochs_completed": 1})
    trainer._write_json(trainer._evaluation_report_path(), {"accuracy": 0.5})
    trainer._write_json(
        trainer._export_report_path(),
        {"onnx_path": "models/waste_classifier.onnx"},
    )
    trainer._write_json(trainer._model_manifest_path(), {"labels": ["glass", "plastic"]})

    report = trainer.build_artifact_report()

    assert Path(str(trainer._artifact_bundle_path())).is_file()
    assert report["training"]["epochs_completed"] == 1
    assert report["evaluation"]["accuracy"] == 0.5
    assert report["export"]["onnx_path"] == "models/waste_classifier.onnx"
    assert report["model_manifest"]["labels"] == ["glass", "plastic"]


def test_export_experiment_spec_writes_backend_metadata(tmp_path: Path) -> None:
    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            training_backend="pytorch",
            experiment_name="demo-spec",
            show_progress=False,
        ),
    )

    spec_path = trainer.export_experiment_spec()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))

    assert spec_path.is_file()
    assert payload["experiment_name"] == "demo-spec"
    assert payload["training_backend"] == "pytorch"
    assert payload["backend_capability"]["supported"] is True


def test_benchmark_writes_report_for_pytorch_backend(tmp_path: Path) -> None:
    pipeline_config = build_two_class_manifest(
        tmp_path / "benchmark_run",
        per_class=2,
        random_seed=47,
    )
    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            training_backend="pytorch",
            experiment_name="benchmark-demo",
            manifest_path=pipeline_config.manifest_path,
            show_progress=False,
        ),
    )

    trainer.fit = lambda manifest_path=None: {  # type: ignore[method-assign]
        "best_checkpoint_path": str(tmp_path / "artifacts" / "checkpoints" / "benchmark-demo.pt"),
        "best_loss": 0.42,
        "best_epoch": 2,
        "epochs_completed": 2,
        "training_report_path": str(trainer._training_report_path()),
    }
    trainer.evaluate = lambda **kwargs: {  # type: ignore[method-assign]
        "evaluation_report_path": str(trainer._evaluation_report_path()),
        "accuracy": 0.8,
        "macro_f1": 0.75,
        "weighted_f1": 0.77,
    }
    trainer.export_onnx = lambda **kwargs: {  # type: ignore[method-assign]
        "export_report_path": str(trainer._export_report_path()),
        "verification": {"verified": True},
    }
    trainer.build_artifact_report = lambda manifest_path=None: {  # type: ignore[method-assign]
        "artifact_bundle_path": str(trainer._artifact_bundle_path()),
    }

    summary = trainer.benchmark()
    report_path = trainer._benchmark_report_path()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.is_file()
    assert summary["status"] == "completed"
    assert payload["metrics"]["best_loss"] == 0.42
    assert payload["metrics"]["accuracy"] == 0.8
    assert payload["metrics"]["onnx_verified"] is True
    assert set(payload["stages"]) == {"fit", "evaluate", "export_onnx", "report"}


def test_benchmark_marks_rust_backend_as_unsupported(tmp_path: Path) -> None:
    pipeline_config = build_two_class_manifest(
        tmp_path / "unsupported_run",
        per_class=2,
        random_seed=53,
    )
    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            training_backend="rust_tch",
            experiment_name="rust-preview",
            manifest_path=pipeline_config.manifest_path,
            show_progress=False,
        ),
    )

    summary = trainer.benchmark()

    assert summary["status"] == "unsupported"
    assert summary["backend_capability"]["backend"] == "rust_tch"
    assert summary["stages"] == {}


def test_compare_benchmark_reports_returns_stage_and_metric_deltas(tmp_path: Path) -> None:
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(
            {
                "training_backend": "pytorch",
                "experiment_name": "baseline",
                "stages": {"fit": {"duration_seconds": 12.0}},
                "metrics": {
                    "total_duration_seconds": 20.0,
                    "accuracy": 0.78,
                    "macro_f1": 0.74,
                    "weighted_f1": 0.76,
                },
            }
        ),
        encoding="utf-8",
    )
    right_path.write_text(
        json.dumps(
            {
                "training_backend": "rust_tch",
                "experiment_name": "candidate",
                "stages": {"fit": {"duration_seconds": 10.0}},
                "metrics": {
                    "total_duration_seconds": 17.5,
                    "accuracy": 0.8,
                    "macro_f1": 0.75,
                    "weighted_f1": 0.78,
                },
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_benchmark_reports(left_path, right_path)

    assert comparison["duration_delta_seconds"] == -2.5
    assert comparison["fit_stage_delta_seconds"] == -2.0
    assert abs(float(comparison["accuracy_delta"]) - 0.02) < 1e-9
