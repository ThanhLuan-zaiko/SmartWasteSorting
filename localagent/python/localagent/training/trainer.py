from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from localagent.bridge import RustAccelerationBridge
from localagent.config import AgentPaths, TrainingConfig
from localagent.data import DatasetIndex
from localagent.training.manifest_dataset import ManifestImageDataset
from localagent.utils import TerminalProgressBar


class WasteTrainer:
    def __init__(self, paths: AgentPaths, config: TrainingConfig) -> None:
        self.paths = paths
        self.config = config
        self.rust_acceleration = RustAccelerationBridge()
        self._last_model_metadata: dict[str, Any] = {}

    def summarize_training_plan(
        self,
        dataset_index: DatasetIndex | None = None,
        manifest_frame: pl.DataFrame | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "experiment_name": self.config.experiment_name,
            "model_name": self.config.model_name,
            "pretrained_backbone": self.config.pretrained_backbone,
            "freeze_backbone": self.config.freeze_backbone,
            "image_size": self.config.image_size,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "dataset_size": len(dataset_index) if dataset_index is not None else 0,
            "artifact_dir": str(self.paths.artifact_dir),
            "manifest_path": str(self.config.manifest_path),
            "device": self.config.device,
            "resolved_device": self._resolve_device(),
            "rust_acceleration_available": self.rust_acceleration.is_available(),
            "cache_dir": str(self._cache_dir()),
            "cache_failure_report_path": str(self._cache_failure_report_path()),
            "early_stopping_patience": self.config.early_stopping_patience,
            "early_stopping_min_delta": self.config.early_stopping_min_delta,
            "enable_early_stopping": self.config.enable_early_stopping,
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
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        if self.config.model_name != "mobilenet_v3_small":
            raise ValueError(f"Unsupported model_name: {self.config.model_name}")

        pretrained_loaded = False
        if self.config.pretrained_backbone:
            try:
                model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
                pretrained_loaded = True
            except Exception as error:  # pragma: no cover - depends on local weight cache/network
                print(
                    "Unable to load pretrained MobileNetV3-Small weights "
                    f"({type(error).__name__}: {error}). Falling back to random init."
                )
                model = mobilenet_v3_small(weights=None)
        else:
            model = mobilenet_v3_small(weights=None)

        model.classifier[-1] = nn.Linear(int(model.classifier[-1].in_features), num_classes)
        if self.config.freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False

        self._last_model_metadata = {
            "model_name": self.config.model_name,
            "pretrained_backbone_requested": self.config.pretrained_backbone,
            "pretrained_backbone_loaded": pretrained_loaded,
            "freeze_backbone": self.config.freeze_backbone,
        }
        return model

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
        cache_dir: Path | None = None,
    ) -> tuple[dict[str, ManifestImageDataset], dict[str, int]]:
        frame = manifest_frame if manifest_frame is not None else self.load_manifest(manifest_path)
        label_to_index = self.build_label_index(frame)
        datasets = {
            split: ManifestImageDataset(
                frame,
                split=split,
                label_to_index=label_to_index,
                image_size=self.config.image_size,
                cache_dir=cache_dir,
            )
            for split in ("train", "val", "test")
        }
        return datasets, label_to_index

    def build_dataloaders(
        self,
        manifest_path: Path | None = None,
        manifest_frame: pl.DataFrame | None = None,
        cache_dir: Path | None = None,
    ):
        from torch.utils.data import DataLoader

        datasets, label_to_index = self.build_datasets(
            manifest_path,
            manifest_frame,
            cache_dir=cache_dir,
        )
        resolved_device = self._resolve_device()
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": (resolved_device == "cuda"),
            "persistent_workers": (self.config.num_workers > 0),
        }
        if self.config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.config.prefetch_factor
        loaders = {
            split: DataLoader(
                dataset,
                shuffle=(split == "train"),
                **loader_kwargs,
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
        cache_summary = self.warm_image_cache(manifest_frame=frame)
        cache_dir = Path(cache_summary["cache_dir"]) if cache_summary is not None else None
        loaders, label_to_index = self.build_dataloaders(manifest_frame=frame, cache_dir=cache_dir)
        if "train" not in loaders:
            raise RuntimeError("Training split is empty. Re-run the dataset pipeline first.")
        if not label_to_index:
            raise RuntimeError("No labels available in manifest. Cannot train a classifier.")

        model = self.build_model_stub(num_classes=len(label_to_index))
        device = self._resolve_device()
        print(f"Training on device: {device}")
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history: list[dict[str, float | int]] = []
        best_epoch = 0
        best_loss = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        stopped_early = False
        stop_reason: str | None = None

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(
                model=model,
                loader=loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                training=True,
                stage_name=f"train {epoch}/{self.config.epochs}",
            )

            val_metrics = (
                self._run_epoch(
                    model=model,
                    loader=loaders["val"],
                    criterion=criterion,
                    optimizer=None,
                    device=device,
                    training=False,
                    stage_name=f"val {epoch}/{self.config.epochs}",
                )
                if "val" in loaders
                else None
            )

            metric_source = val_metrics or train_metrics
            current_loss = float(metric_source["loss"])
            if current_loss < best_loss - self.config.early_stopping_min_delta:
                best_loss = current_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

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

            if self._should_stop_early(epochs_without_improvement):
                stopped_early = True
                stop_reason = (
                    "Early stopping triggered after "
                    f"{epochs_without_improvement} epoch(s) without improvement "
                    f"greater than {self.config.early_stopping_min_delta}."
                )
                print(stop_reason)
                break

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
                "model": self._last_model_metadata,
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
            },
            checkpoint_path,
        )
        labels_path = self.export_label_index(manifest_frame=frame)

        summary = {
            "checkpoint_path": str(checkpoint_path),
            "labels_path": str(labels_path),
            "num_classes": len(label_to_index),
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "history": history,
            "device": device,
            "model": self._last_model_metadata,
            "cache": cache_summary,
        }

        if "test" in loaders:
            test_metrics = self._run_epoch(
                model=model,
                loader=loaders["test"],
                criterion=criterion,
                optimizer=None,
                device=device,
                training=False,
                stage_name="test",
            )
            summary["test_loss"] = float(test_metrics["loss"])
            summary["test_accuracy"] = float(test_metrics["accuracy"])

        return summary

    def warm_image_cache(
        self,
        *,
        manifest_path: Path | None = None,
        manifest_frame: pl.DataFrame | None = None,
    ) -> dict[str, Any] | None:
        if not self.config.use_rust_image_cache:
            return None
        if not self.rust_acceleration.is_available():
            print("Rust acceleration is unavailable. Training will read original images.")
            return None

        frame = manifest_frame if manifest_frame is not None else self.load_manifest(manifest_path)
        entries = self._cache_entries(frame)
        if not entries:
            return None

        cache_dir = self._cache_dir()
        failure_report_path = self._cache_failure_report_path()
        cache_dir.mkdir(parents=True, exist_ok=True)
        summary = self.rust_acceleration.prepare_image_cache(
            entries,
            cache_dir=cache_dir,
            failure_report_path=failure_report_path,
            image_size=self.config.image_size,
            force=self.config.force_rebuild_cache,
            show_progress=self.config.show_progress,
        )
        if summary is None:
            return None

        print(
            "Rust cache summary: "
            f"processed={summary['processed']}, "
            f"skipped={summary['skipped']}, "
            f"errors={summary['errors']}"
        )
        summary = self._recover_cache_failures(summary)
        if int(summary.get("fallback_processed", 0)) > 0:
            print(
                "OpenCV fallback summary: "
                f"processed={summary['fallback_processed']}, "
                f"errors={summary['fallback_errors']}"
            )
        if int(summary["errors"]) > 0 and summary.get("failure_report_path"):
            print(f"Rust cache failure report: {summary['failure_report_path']}")
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

    def _cache_dir(self) -> Path:
        return self.config.cache_dir / f"{self.config.image_size}px"

    def _cache_failure_report_path(self) -> Path:
        return (
            self.paths.artifact_dir
            / "reports"
            / f"training_cache_failures_{self.config.image_size}px.json"
        )

    def _cache_entries(self, manifest_frame: pl.DataFrame) -> list[tuple[str, str]]:
        return [
            (str(row["sample_id"]), str(row["image_path"]))
            for row in self._labeled_frame(manifest_frame)
            .sort("sample_id")
            .select("sample_id", "image_path")
            .iter_rows(named=True)
        ]

    def _recover_cache_failures(self, summary: dict[str, Any]) -> dict[str, Any]:
        failure_report_path_raw = summary.get("failure_report_path")
        if not failure_report_path_raw:
            summary["fallback_processed"] = 0
            summary["fallback_errors"] = int(summary.get("errors", 0))
            return summary

        failure_report_path = Path(str(failure_report_path_raw))
        if not failure_report_path.is_file():
            summary["fallback_processed"] = 0
            summary["fallback_errors"] = int(summary.get("errors", 0))
            return summary

        failures = json.loads(failure_report_path.read_text(encoding="utf-8"))
        if not isinstance(failures, list) or not failures:
            summary["fallback_processed"] = 0
            summary["fallback_errors"] = 0
            summary["errors"] = 0
            return summary

        cache_dir = Path(str(summary["cache_dir"]))
        progress = TerminalProgressBar(
            total=len(failures),
            description="opencv fallback",
            enabled=self.config.show_progress,
        )
        recovered = 0
        remaining_failures: list[dict[str, Any]] = []
        for index, record in enumerate(failures, start=1):
            if self._write_cache_with_opencv(
                image_path=Path(str(record["image_path"])),
                output_path=cache_dir / f"{record['sample_id']}.png",
            ):
                recovered += 1
            else:
                remaining_failures.append(record)
            progress.update(index, postfix=str(record["sample_id"]))
        progress.close(
            summary=(
                "OpenCV fallback: "
                f"recovered={recovered} remaining={len(remaining_failures)}"
            )
        )

        failure_report_path.write_text(
            json.dumps(remaining_failures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["processed"] = int(summary.get("processed", 0)) + recovered
        summary["errors"] = len(remaining_failures)
        summary["fallback_processed"] = recovered
        summary["fallback_errors"] = len(remaining_failures)
        return summary

    def _write_cache_with_opencv(self, *, image_path: Path, output_path: Path) -> bool:
        try:
            encoded = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                return False

            resized = cv2.resize(
                image,
                (self.config.image_size, self.config.image_size),
                interpolation=cv2.INTER_AREA,
            )
            success, buffer = cv2.imencode(".png", resized)
            if not success:
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            buffer.tofile(output_path)
            return True
        except Exception:
            return False

    def _run_epoch(
        self,
        *,
        model,
        loader,
        criterion,
        optimizer,
        device: str,
        training: bool,
        stage_name: str,
    ) -> dict[str, float]:
        import torch

        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        context = torch.enable_grad() if training else torch.inference_mode()
        progress = TerminalProgressBar(
            total=len(loader),
            description=stage_name,
            enabled=self.config.show_progress,
        )
        with context:
            for batch_index, (images, labels) in enumerate(loader, start=1):
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

                running_loss = total_loss / max(total_examples, 1)
                running_accuracy = total_correct / max(total_examples, 1)
                progress.update(
                    batch_index,
                    postfix=f"loss={running_loss:.4f} acc={running_accuracy:.3f}",
                )

        if total_examples == 0:
            progress.close(summary=f"{stage_name}: no samples")
            return {"loss": 0.0, "accuracy": 0.0}

        metrics = {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }
        progress.close(
            summary=(
                f"{stage_name}: loss={metrics['loss']:.4f} "
                f"accuracy={metrics['accuracy']:.3f}"
            )
        )
        return metrics

    def _should_stop_early(self, epochs_without_improvement: int) -> bool:
        if not self.config.enable_early_stopping:
            return False
        if self.config.early_stopping_patience <= 0:
            return False
        return epochs_without_improvement >= self.config.early_stopping_patience
