from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import polars as pl

from localagent.config import AgentPaths, TrainingConfig
from localagent.data import DatasetIndex
from localagent.training.manifest_dataset import ManifestImageDataset


class WasteTrainer:
    def __init__(self, paths: AgentPaths, config: TrainingConfig) -> None:
        self.paths = paths
        self.config = config

    def summarize_training_plan(
        self,
        dataset_index: DatasetIndex | None = None,
        manifest_frame: pl.DataFrame | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "experiment_name": self.config.experiment_name,
            "image_size": self.config.image_size,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "dataset_size": len(dataset_index) if dataset_index is not None else 0,
            "artifact_dir": str(self.paths.artifact_dir),
            "manifest_path": str(self.config.manifest_path),
            "device": self.config.device,
        }
        frame = manifest_frame
        if frame is None and self.config.manifest_path.exists():
            frame = self.load_manifest()

        if frame is not None:
            labeled_frame = self._labeled_frame(frame)
            split_counts = {
                row["split"]: row["count"]
                for row in labeled_frame.group_by("split")
                .len(name="count")
                .iter_rows(named=True)
            }
            class_names = self.class_names(frame)
            summary.update(
                {
                    "dataset_size": labeled_frame.height,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "split_counts": split_counts,
                }
            )
        return summary

    def build_model_stub(self, num_classes: int = 4) -> Any:
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def export_onnx_stub(self, output_path: Path | None = None) -> Path:
        destination = output_path or self.paths.model_dir / "waste_classifier.onnx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch(exist_ok=True)
        return destination

    def load_manifest(self, manifest_path: Path | None = None) -> pl.DataFrame:
        source = manifest_path or self.config.manifest_path
        if not source.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {source}")
        return pl.read_parquet(source)

    def class_names(self, manifest_frame: pl.DataFrame) -> list[str]:
        return (
            manifest_frame.filter(pl.col("is_valid") & (pl.col("label") != "unknown"))
            .get_column("label")
            .unique()
            .sort()
            .to_list()
        )

    def build_label_index(self, manifest_frame: pl.DataFrame) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.class_names(manifest_frame))}

    def build_datasets(
        self,
        manifest_path: Path | None = None,
        manifest_frame: pl.DataFrame | None = None,
    ) -> tuple[dict[str, ManifestImageDataset], dict[str, int]]:
        frame = manifest_frame if manifest_frame is not None else self.load_manifest(manifest_path)
        label_to_index = self.build_label_index(frame)
        datasets = {
            split: ManifestImageDataset(
                frame,
                split=split,
                label_to_index=label_to_index,
                image_size=self.config.image_size,
            )
            for split in ("train", "val", "test")
        }
        return datasets, label_to_index

    def build_dataloaders(
        self,
        manifest_path: Path | None = None,
        manifest_frame: pl.DataFrame | None = None,
    ):
        from torch.utils.data import DataLoader

        datasets, label_to_index = self.build_datasets(manifest_path, manifest_frame)
        loaders = {
            split: DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == "train"),
                num_workers=self.config.num_workers,
            )
            for split, dataset in datasets.items()
            if len(dataset) > 0
        }
        return loaders, label_to_index

    def export_label_index(
        self,
        manifest_path: Path | None = None,
        output_path: Path | None = None,
        manifest_frame: pl.DataFrame | None = None,
    ) -> Path:
        frame = manifest_frame if manifest_frame is not None else self.load_manifest(manifest_path)
        labels = self.class_names(frame)
        destination = output_path or self.config.labels_output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        return destination

    def fit(
        self,
        manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        import torch
        import torch.nn as nn

        frame = self.load_manifest(manifest_path)
        loaders, label_to_index = self.build_dataloaders(manifest_frame=frame)
        if "train" not in loaders:
            raise RuntimeError("Training split is empty. Re-run the dataset pipeline first.")
        if not label_to_index:
            raise RuntimeError("No labels available in manifest. Cannot train a classifier.")

        model = self.build_model_stub(num_classes=len(label_to_index))
        device = self._resolve_device()
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history: list[dict[str, float | int]] = []
        best_epoch = 0
        best_loss = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(
                model=model,
                loader=loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                training=True,
            )

            val_metrics = (
                self._run_epoch(
                    model=model,
                    loader=loaders["val"],
                    criterion=criterion,
                    optimizer=None,
                    device=device,
                    training=False,
                )
                if "val" in loaders
                else None
            )

            metric_source = val_metrics or train_metrics
            if metric_source["loss"] < best_loss:
                best_loss = float(metric_source["loss"])
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

            epoch_summary: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
            }
            if val_metrics is not None:
                epoch_summary.update(
                    {
                        "val_loss": float(val_metrics["loss"]),
                        "val_accuracy": float(val_metrics["accuracy"]),
                    }
                )
            history.append(epoch_summary)

        model.load_state_dict(best_state_dict)
        checkpoint_path = self._checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "labels": self.class_names(frame),
                "history": history,
                "experiment_name": self.config.experiment_name,
                "best_epoch": best_epoch,
            },
            checkpoint_path,
        )
        labels_path = self.export_label_index(manifest_frame=frame)

        summary = {
            "checkpoint_path": str(checkpoint_path),
            "labels_path": str(labels_path),
            "num_classes": len(label_to_index),
            "best_epoch": best_epoch,
            "history": history,
        }

        if "test" in loaders:
            test_metrics = self._run_epoch(
                model=model,
                loader=loaders["test"],
                criterion=criterion,
                optimizer=None,
                device=device,
                training=False,
            )
            summary["test_loss"] = float(test_metrics["loss"])
            summary["test_accuracy"] = float(test_metrics["accuracy"])

        return summary

    def _labeled_frame(self, manifest_frame: pl.DataFrame) -> pl.DataFrame:
        return manifest_frame.filter(pl.col("is_valid") & (pl.col("label") != "unknown"))

    def _resolve_device(self) -> str:
        import torch

        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _checkpoint_path(self) -> Path:
        return self.config.checkpoint_dir / f"{self.config.experiment_name}.pt"

    def _run_epoch(
        self,
        *,
        model,
        loader,
        criterion,
        optimizer,
        device: str,
        training: bool,
    ) -> dict[str, float]:
        import torch

        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                logits = model(images)
                loss = criterion(logits, labels)

                if training and optimizer is not None:
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.item()) * int(labels.size(0))
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_examples += int(labels.size(0))

        if total_examples == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }
