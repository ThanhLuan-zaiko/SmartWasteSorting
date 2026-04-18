from __future__ import annotations

import csv
from pathlib import Path

import localagent.training.trainer as trainer_module
import torch
from localagent.config import DatasetPipelineConfig, TrainingConfig
from localagent.data import DatasetPipeline

from tests.training_support import build_trainer, write_rgb_image


def test_pseudo_label_updates_manifest_with_confidence_gate(tmp_path: Path, monkeypatch) -> None:
    raw_dataset_dir = tmp_path / "dataset"
    raw_dataset_dir.mkdir(parents=True)

    image_values = {
        "seed_dark_1.jpg": 10,
        "seed_bright_1.jpg": 240,
        "candidate_dark_1.jpg": 12,
        "candidate_mid_1.jpg": 128,
    }
    for file_name, value in image_values.items():
        write_rgb_image(raw_dataset_dir / file_name, width=64, height=64, value=value)

    pipeline_config = DatasetPipelineConfig(
        raw_dataset_dir=raw_dataset_dir,
        manifest_dir=tmp_path / "artifacts" / "manifests",
        report_dir=tmp_path / "artifacts" / "reports",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=17,
        infer_labels_from_filename=False,
        show_progress=False,
    )
    pipeline = DatasetPipeline(pipeline_config)
    pipeline.run_all()

    manifest_frame = pipeline.load_manifest()
    rows_by_name = {
        str(row["file_name"]): row for row in manifest_frame.iter_rows(named=True)
    }
    labels_file = tmp_path / "seed_labels.csv"
    with labels_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "status"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": rows_by_name["seed_dark_1.jpg"]["sample_id"],
                "label": "glass",
                "status": "labeled",
            }
        )
        writer.writerow(
            {
                "sample_id": rows_by_name["seed_bright_1.jpg"]["sample_id"],
                "label": "plastic",
                "status": "labeled",
            }
        )
    pipeline.import_labels(labels_file)

    trainer = build_trainer(
        tmp_path,
        TrainingConfig(
            batch_size=1,
            epochs=1,
            num_workers=0,
            pretrained_backbone=False,
            manifest_path=pipeline_config.manifest_path,
            checkpoint_dir=tmp_path / "artifacts" / "checkpoints",
            pseudo_label_confidence_threshold=0.85,
            pseudo_label_margin_threshold=0.15,
            show_progress=False,
        ),
    )

    class BrightnessModel(torch.nn.Module):
        def forward(self, images):
            mean = images.mean(dim=(1, 2, 3))
            return torch.stack([(0.5 - mean) * 12.0, (mean - 0.5) * 12.0], dim=1)

    monkeypatch.setattr(
        trainer,
        "build_model_stub",
        lambda num_classes=4: BrightnessModel(),
    )
    monkeypatch.setattr(
        trainer,
        "_load_checkpoint_payload",
        lambda checkpoint_path: {
            "labels": ["glass", "plastic"],
            "best_model_state_dict": BrightnessModel().state_dict(),
            "training_config": {
                "model_name": "resnet18",
                "image_size": 64,
                "normalization_preset": "imagenet",
            },
        },
    )
    monkeypatch.setattr(
        trainer_module,
        "build_training_transforms",
        lambda image_size, normalization_preset="imagenet": (
            lambda image: torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        ),
    )

    summary = trainer.pseudo_label()
    updated_manifest = trainer.load_manifest()
    updated_rows = {
        str(row["file_name"]): row for row in updated_manifest.iter_rows(named=True)
    }

    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 1
    assert updated_rows["candidate_dark_1.jpg"]["annotation_status"] == "pseudo_labeled"
    assert updated_rows["candidate_dark_1.jpg"]["label_source"] == "model_pseudo"
    assert updated_rows["candidate_dark_1.jpg"]["label"] == "glass"
    assert updated_rows["candidate_mid_1.jpg"]["annotation_status"] == "unlabeled"
    assert updated_rows["candidate_mid_1.jpg"]["review_status"] == "pseudo_rejected"
    assert Path(str(summary["pseudo_label_report_path"])).is_file()
