from __future__ import annotations

import copy
import csv
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import polars as pl

from localagent.bridge import RustAccelerationBridge
from localagent.config import AgentPaths, DatasetPipelineConfig, TrainingConfig
from localagent.data import DatasetIndex
from localagent.training.benchmarking import (
    ExperimentSpec,
    ProcessMemorySampler,
    bytes_to_megabytes,
    compare_benchmark_reports,
)
from localagent.training.manifest_dataset import ManifestImageDataset
from localagent.utils import TerminalProgressBar
from localagent.vision import build_training_transforms, load_rgb_image, normalization_stats

SUPPORTED_CNN_MODELS = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet18",
    "efficientnet_b0",
)

ACCEPTED_LABEL_SOURCES = {"curated", "cluster_review", "model_pseudo"}
ACCEPTED_ANNOTATION_STATUSES = {"labeled", "pseudo_labeled"}


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
            "training_preset": self.config.training_preset,
            "experiment_name": self.config.experiment_name,
            "training_backend": self.config.training_backend,
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
            "training_report_path": str(self._training_report_path()),
            "experiment_spec_path": str(self._experiment_spec_path()),
            "benchmark_report_path": str(self._benchmark_report_path()),
            "resume_from_checkpoint": (
                None
                if self.config.resume_from_checkpoint is None
                else str(self.config.resume_from_checkpoint)
            ),
            "device": self.config.device,
            "resolved_device": self._resolve_device(),
            "rust_acceleration_available": self.rust_acceleration.is_available(),
            "best_checkpoint_path": str(self._checkpoint_path()),
            "latest_checkpoint_path": str(self._latest_checkpoint_path()),
            "cache_dir": str(self._cache_dir()),
            "cache_format": self.config.cache_format,
            "cache_failure_report_path": str(self._cache_failure_report_path()),
            "evaluation_report_path": str(self._evaluation_report_path()),
            "confusion_matrix_path": str(self._confusion_matrix_path()),
            "export_report_path": str(self._export_report_path()),
            "artifact_bundle_path": str(self._artifact_bundle_path()),
            "onnx_output_path": str(self.config.onnx_output_path),
            "model_manifest_path": str(self._model_manifest_path()),
            "onnx_opset": self.config.onnx_opset,
            "verify_onnx": self.config.verify_onnx,
            "export_batch_size": self.config.export_batch_size,
            "normalization_preset": self.config.normalization_preset,
            "class_bias_strategy": self.config.class_bias_strategy,
            "early_stopping_patience": self.config.early_stopping_patience,
            "early_stopping_min_delta": self.config.early_stopping_min_delta,
            "enable_early_stopping": self.config.enable_early_stopping,
            "pseudo_label_confidence_threshold": self.config.pseudo_label_confidence_threshold,
            "pseudo_label_margin_threshold": self.config.pseudo_label_margin_threshold,
            "backend_capability": self._training_backend_capability(),
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
                    "train_label_counts": self.train_label_counts(frame),
                    "train_imbalance_ratio": self.train_imbalance_ratio(frame),
                    "effective_training_mode": self._effective_training_mode(frame),
                }
            )
            if self.config.class_bias_strategy != "none":
                summary["class_weight_map"] = self.class_weight_map(frame)
        return summary

    def build_model_stub(self, num_classes: int = 4) -> Any:
        import torch.nn as nn
        from torchvision.models import (
            EfficientNet_B0_Weights,
            MobileNet_V3_Large_Weights,
            MobileNet_V3_Small_Weights,
            ResNet18_Weights,
            efficientnet_b0,
            mobilenet_v3_large,
            mobilenet_v3_small,
            resnet18,
        )

        model_factories = {
            "mobilenet_v3_small": (
                mobilenet_v3_small,
                MobileNet_V3_Small_Weights.DEFAULT,
            ),
            "mobilenet_v3_large": (
                mobilenet_v3_large,
                MobileNet_V3_Large_Weights.DEFAULT,
            ),
            "resnet18": (
                resnet18,
                ResNet18_Weights.DEFAULT,
            ),
            "efficientnet_b0": (
                efficientnet_b0,
                EfficientNet_B0_Weights.DEFAULT,
            ),
        }
        if self.config.model_name not in model_factories:
            raise ValueError(f"Unsupported model_name: {self.config.model_name}")

        factory, default_weights = model_factories[self.config.model_name]
        pretrained_loaded = False
        if self.config.pretrained_backbone:
            try:
                model = factory(weights=default_weights)
                pretrained_loaded = True
            except Exception as error:  # pragma: no cover - depends on local weight cache/network
                print(
                    f"Unable to load pretrained weights for {self.config.model_name} "
                    f"({type(error).__name__}: {error}). Falling back to random init."
                )
                model = factory(weights=None)
        else:
            model = factory(weights=None)

        self._replace_classifier_head(model, num_classes=num_classes, nn_module=nn)
        if self.config.freeze_backbone:
            self._freeze_model_backbone(model)

        self._last_model_metadata = {
            "training_backend": self.config.training_backend,
            "model_name": self.config.model_name,
            "pretrained_backbone_requested": self.config.pretrained_backbone,
            "pretrained_backbone_loaded": pretrained_loaded,
            "freeze_backbone": self.config.freeze_backbone,
            "image_size": self.config.image_size,
            "normalization_preset": self.config.normalization_preset,
        }
        return model

    def _replace_classifier_head(self, model, *, num_classes: int, nn_module) -> None:
        if self.config.model_name in {
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "efficientnet_b0",
        }:
            model.classifier[-1] = nn_module.Linear(
                int(model.classifier[-1].in_features),
                num_classes,
            )
            return
        if self.config.model_name == "resnet18":
            model.fc = nn_module.Linear(int(model.fc.in_features), num_classes)
            return
        raise ValueError(f"Unsupported model_name: {self.config.model_name}")

    def _freeze_model_backbone(self, model) -> None:
        if self.config.model_name in {
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "efficientnet_b0",
        }:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
            return
        if self.config.model_name == "resnet18":
            for name, parameter in model.named_parameters():
                if not name.startswith("fc."):
                    parameter.requires_grad = False
            return
        raise ValueError(f"Unsupported model_name: {self.config.model_name}")

    def load_manifest(self, manifest_path: Path | None = None) -> pl.DataFrame:
        source = manifest_path or self.config.manifest_path
        if not source.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {source}")
        return self._ensure_manifest_columns(pl.read_parquet(source))

    def class_names(self, manifest_frame: pl.DataFrame) -> list[str]:
        return (
            self._labeled_frame(manifest_frame)
            .get_column("label")
            .unique()
            .sort()
            .to_list()
        )

    def build_label_index(self, manifest_frame: pl.DataFrame) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.class_names(manifest_frame))}

    def train_label_counts(self, manifest_frame: pl.DataFrame) -> dict[str, int]:
        counts = {label: 0 for label in self.class_names(manifest_frame)}
        for row in (
            self._labeled_frame(manifest_frame)
            .filter(pl.col("split") == "train")
            .group_by("label")
            .len(name="count")
            .iter_rows(named=True)
        ):
            counts[str(row["label"])] = int(row["count"])
        return counts

    def train_imbalance_ratio(self, manifest_frame: pl.DataFrame) -> float:
        counts = [count for count in self.train_label_counts(manifest_frame).values() if count > 0]
        if not counts:
            return 0.0
        return max(counts) / min(counts)

    def class_weight_map(self, manifest_frame: pl.DataFrame) -> dict[str, float]:
        class_names = self.class_names(manifest_frame)
        train_labels = [
            str(row["label"])
            for row in self._labeled_frame(manifest_frame)
            .filter(pl.col("split") == "train")
            .sort("sample_id")
            .select("label")
            .iter_rows(named=True)
        ]
        rust_result = self.rust_acceleration.compute_class_weight_map(train_labels, class_names)
        if rust_result is not None:
            return {str(label): float(weight) for label, weight in rust_result.items()}

        counts = self.train_label_counts(manifest_frame)
        present_labels = [label for label, count in counts.items() if count > 0]
        if not present_labels:
            return {label: 1.0 for label in counts}

        total_samples = sum(counts[label] for label in present_labels)
        num_present_labels = len(present_labels)
        return {
            label: (
                0.0
                if counts[label] <= 0
                else float(total_samples / (num_present_labels * counts[label]))
            )
            for label in counts
        }

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
                cache_format=self.config.cache_format,
                normalization_preset=self.config.normalization_preset,
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

        frame = manifest_frame if manifest_frame is not None else self.load_manifest(manifest_path)
        datasets, label_to_index = self.build_datasets(
            manifest_frame=frame,
            cache_dir=cache_dir,
        )
        train_sampler = (
            self._build_train_sampler(
                datasets["train"],
                weight_map=self.class_weight_map(frame),
            )
            if "train" in datasets
            else None
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
        loaders = {}
        for split, dataset in datasets.items():
            if len(dataset) == 0:
                continue

            split_kwargs = dict(loader_kwargs)
            shuffle = split == "train"
            if split == "train" and train_sampler is not None:
                split_kwargs["sampler"] = train_sampler
                shuffle = False
            loaders[split] = DataLoader(dataset, shuffle=shuffle, **split_kwargs)
        return loaders, label_to_index

    def export_label_index(
        self,
        manifest_path: Path | None = None,
        output_path: Path | None = None,
        manifest_frame: pl.DataFrame | None = None,
    ) -> Path:
        frame = manifest_frame if manifest_frame is not None else self.load_manifest(manifest_path)
        labels = self.class_names(frame)
        return self._write_labels_payload(labels, output_path=output_path)

    def build_experiment_spec(self) -> dict[str, Any]:
        spec = ExperimentSpec.from_training_config(self.config)
        payload = spec.to_dict()
        payload["generated_at"] = self._timestamp_now()
        payload["backend_capability"] = self._training_backend_capability()
        return payload

    def export_experiment_spec(self, output_path: Path | None = None) -> Path:
        destination = output_path or self._experiment_spec_path()
        spec = ExperimentSpec.from_training_config(self.config)
        payload = spec.to_dict()
        payload["generated_at"] = self._timestamp_now()
        payload["backend_capability"] = self._training_backend_capability()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return destination

    def benchmark(
        self,
        *,
        manifest_path: Path | None = None,
        compare_to: Path | None = None,
    ) -> dict[str, Any]:
        manifest_frame = self.load_manifest(manifest_path)
        capability = self._training_backend_capability()
        benchmark_path = self._benchmark_report_path()
        spec_path = self.export_experiment_spec()
        report: dict[str, Any] = {
            "schema_version": 1,
            "generated_at": self._timestamp_now(),
            "experiment_name": self.config.experiment_name,
            "training_backend": self.config.training_backend,
            "benchmark_report_path": str(benchmark_path),
            "experiment_spec_path": str(spec_path),
            "manifest_path": str(self.config.manifest_path),
            "backend_capability": capability,
            "summary": self.summarize_training_plan(manifest_frame=manifest_frame),
            "stages": {},
            "metrics": {},
        }
        if not capability["supported"]:
            report["status"] = "unsupported"
            if compare_to is not None:
                report["comparison"] = self._compare_benchmark_path(compare_to, current=report)
            self._write_json(benchmark_path, report)
            return report

        try:
            fit_summary = self._run_benchmark_stage(
                report["stages"],
                "fit",
                lambda: self.fit(manifest_path=manifest_path),
            )
            checkpoint_path = Path(str(fit_summary["best_checkpoint_path"]))
            evaluation_summary = self._run_benchmark_stage(
                report["stages"],
                "evaluate",
                lambda: self.evaluate(
                    manifest_path=manifest_path,
                    checkpoint_path=checkpoint_path,
                ),
            )
            export_summary = self._run_benchmark_stage(
                report["stages"],
                "export_onnx",
                lambda: self.export_onnx(
                    manifest_path=manifest_path,
                    checkpoint_path=checkpoint_path,
                ),
            )
            bundle_summary = self._run_benchmark_stage(
                report["stages"],
                "report",
                lambda: self.build_artifact_report(manifest_path=manifest_path),
            )
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            self._write_json(benchmark_path, report)
            raise

        total_duration_seconds = sum(
            float(stage.get("duration_seconds", 0.0))
            for stage in report["stages"].values()
            if isinstance(stage, dict)
        )
        peak_stage_rss_mb = max(
            (
                float(stage["peak_rss_mb"])
                for stage in report["stages"].values()
                if isinstance(stage, dict) and stage.get("peak_rss_mb") is not None
            ),
            default=0.0,
        )

        evaluation_metrics = (
            evaluation_summary if isinstance(evaluation_summary, dict) else None
        )
        report["status"] = "completed"
        report["metrics"] = {
            "total_duration_seconds": total_duration_seconds,
            "peak_stage_rss_mb": peak_stage_rss_mb,
            "best_loss": float(fit_summary["best_loss"]),
            "best_epoch": int(fit_summary["best_epoch"]),
            "epochs_completed": int(fit_summary["epochs_completed"]),
            "accuracy": (
                None
                if evaluation_metrics is None
                else float(evaluation_metrics.get("accuracy"))
            ),
            "macro_f1": (
                None
                if evaluation_metrics is None
                else float(evaluation_metrics.get("macro_f1"))
            ),
            "weighted_f1": (
                None
                if evaluation_metrics is None
                else float(evaluation_metrics.get("weighted_f1"))
            ),
            "onnx_verified": bool(export_summary.get("verification", {}).get("verified")),
            "artifact_bundle_path": str(bundle_summary.get("artifact_bundle_path")),
        }
        if compare_to is not None:
            report["comparison"] = self._compare_benchmark_path(compare_to, current=report)
        self._write_json(benchmark_path, report)
        return report

    def _compute_best_loss_from_history(self, history: list[dict[str, Any]]) -> float:
        best_loss = float("inf")
        for epoch_summary in history:
            metric_source = epoch_summary.get("val_loss", epoch_summary.get("train_loss"))
            if metric_source is None:
                continue
            best_loss = min(best_loss, float(metric_source))
        return best_loss

    def _load_resume_state(
        self,
        *,
        checkpoint_path: Path,
        model,
        optimizer,
        class_names: list[str],
        device: str,
    ) -> dict[str, Any]:
        import torch

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")

        checkpoint_labels = payload.get("labels")
        if checkpoint_labels is not None and list(checkpoint_labels) != class_names:
            raise ValueError(
                "Checkpoint labels do not match the current manifest labels. "
                f"{checkpoint_path}"
            )

        model_state_dict = self._checkpoint_state_dict(payload, checkpoint_path=checkpoint_path)
        if model_state_dict is None:
            raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
        model.load_state_dict(model_state_dict)

        optimizer_state_dict = payload.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            self._move_optimizer_state_to_device(optimizer, device=device)

        history_raw = payload.get("history", [])
        history = list(history_raw) if isinstance(history_raw, list) else []
        best_state_dict = payload.get("best_model_state_dict")
        if best_state_dict is None:
            best_state_dict = copy.deepcopy(model_state_dict)

        best_loss_raw = payload.get("best_loss")
        best_loss = (
            float(best_loss_raw)
            if best_loss_raw is not None
            else self._compute_best_loss_from_history(history)
        )
        if best_loss == float("inf"):
            best_loss = float("inf")

        last_completed_epoch_raw = payload.get("last_completed_epoch", len(history))
        last_completed_epoch = int(last_completed_epoch_raw)
        best_epoch = int(payload.get("best_epoch", 0))
        print(
            "Resuming training from "
            f"{checkpoint_path} starting at epoch {last_completed_epoch + 1}."
        )
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_model_state_dict": best_state_dict,
            "last_completed_epoch": last_completed_epoch,
        }

    def _move_optimizer_state_to_device(self, optimizer, *, device: str) -> None:
        import torch

        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def _checkpoint_payload(
        self,
        *,
        model_state_dict: dict[str, Any],
        optimizer_state_dict: dict[str, Any] | None,
        labels: list[str],
        history: list[dict[str, Any]],
        best_epoch: int,
        best_loss: float,
        best_model_state_dict: dict[str, Any],
        last_completed_epoch: int,
        target_epochs: int,
        checkpoint_kind: str,
        interrupted: bool,
        stopped_early: bool,
        stop_reason: str | None,
        evaluation_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "best_model_state_dict": best_model_state_dict,
            "labels": labels,
            "history": history,
            "experiment_name": self.config.experiment_name,
            "training_preset": self.config.training_preset,
            "training_backend": self.config.training_backend,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "last_completed_epoch": last_completed_epoch,
            "target_epochs": target_epochs,
            "checkpoint_kind": checkpoint_kind,
            "resume_from_checkpoint": (
                None
                if self.config.resume_from_checkpoint is None
                else str(self.config.resume_from_checkpoint)
            ),
            "model": self._last_model_metadata,
            "training_config": {
                "training_backend": self.config.training_backend,
                "model_name": self.config.model_name,
                "image_size": self.config.image_size,
                "normalization_preset": self.config.normalization_preset,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "cache_format": self.config.cache_format,
                "class_bias_strategy": self.config.class_bias_strategy,
                "onnx_opset": self.config.onnx_opset,
            },
            "cache_format": self.config.cache_format,
            "class_bias_strategy": self.config.class_bias_strategy,
            "stopped_early": stopped_early,
            "interrupted": interrupted,
            "stop_reason": stop_reason,
            "evaluation_summary": evaluation_summary,
        }

    def _save_training_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        model_state_dict: dict[str, Any],
        optimizer_state_dict: dict[str, Any] | None,
        labels: list[str],
        history: list[dict[str, Any]],
        best_epoch: int,
        best_loss: float,
        best_model_state_dict: dict[str, Any],
        last_completed_epoch: int,
        target_epochs: int,
        checkpoint_kind: str,
        interrupted: bool,
        stopped_early: bool,
        stop_reason: str | None,
        evaluation_summary: dict[str, Any] | None = None,
    ) -> Path:
        import torch

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._checkpoint_payload(
                model_state_dict=model_state_dict,
                optimizer_state_dict=optimizer_state_dict,
                labels=labels,
                history=history,
                best_epoch=best_epoch,
                best_loss=best_loss,
                best_model_state_dict=best_model_state_dict,
                last_completed_epoch=last_completed_epoch,
                target_epochs=target_epochs,
                checkpoint_kind=checkpoint_kind,
                interrupted=interrupted,
                stopped_early=stopped_early,
                stop_reason=stop_reason,
                evaluation_summary=evaluation_summary,
            ),
            checkpoint_path,
        )
        return checkpoint_path

    def _build_classification_report(
        self,
        *,
        predictions: list[int],
        targets: list[int],
        class_names: list[str],
    ) -> dict[str, Any]:
        rust_report = self.rust_acceleration.build_classification_report(
            predictions=predictions,
            targets=targets,
            class_names=class_names,
        )
        if rust_report is not None:
            return rust_report

        num_classes = len(class_names)
        confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for target, prediction in zip(targets, predictions, strict=False):
            if 0 <= int(target) < num_classes and 0 <= int(prediction) < num_classes:
                confusion_matrix[int(target)][int(prediction)] += 1

        per_class: dict[str, dict[str, float | int]] = {}
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0
        total_support = 0
        total_correct = 0

        for class_index, label in enumerate(class_names):
            tp = int(confusion_matrix[class_index][class_index])
            support = int(sum(confusion_matrix[class_index]))
            predicted_count = int(sum(row[class_index] for row in confusion_matrix))
            fp = predicted_count - tp
            fn = support - tp
            precision = (tp / predicted_count) if predicted_count > 0 else 0.0
            recall = (tp / support) if support > 0 else 0.0
            f1 = (
                (2.0 * precision * recall / (precision + recall))
                if (precision + recall) > 0.0
                else 0.0
            )
            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
            weighted_f1 += f1 * support
            total_support += support
            total_correct += tp

        divisor = max(num_classes, 1)
        return {
            "labels": class_names,
            "num_samples": total_support,
            "accuracy": (total_correct / total_support) if total_support > 0 else 0.0,
            "macro_precision": macro_precision / divisor,
            "macro_recall": macro_recall / divisor,
            "macro_f1": macro_f1 / divisor,
            "weighted_f1": (weighted_f1 / total_support) if total_support > 0 else 0.0,
            "per_class": per_class,
            "confusion_matrix": confusion_matrix,
        }

    def _write_evaluation_artifacts(self, evaluation: dict[str, Any]) -> tuple[Path, Path]:
        report_path = self._evaluation_report_path()
        confusion_path = self._confusion_matrix_path()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        labels = list(evaluation.get("labels", []))
        matrix = list(evaluation.get("confusion_matrix", []))
        with confusion_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label", *labels])
            for label, row in zip(labels, matrix, strict=False):
                writer.writerow([label, *row])
        return report_path, confusion_path

    def _training_backend_capability(self) -> dict[str, Any]:
        if self.config.training_backend == "pytorch":
            return {
                "backend": "pytorch",
                "supported": True,
                "maturity": "stable",
                "notes": "Full training loop uses PyTorch autograd and optimizer.",
            }
        if self.config.training_backend == "rust_tch":
            return {
                "backend": "rust_tch",
                "supported": False,
                "maturity": "planned",
                "notes": (
                    "Rust training backend scaffolding is present, but the full tch-based "
                    "training core is not implemented in this build."
                ),
            }
        return {
            "backend": self.config.training_backend,
            "supported": False,
            "maturity": "unknown",
            "notes": "Unsupported training backend.",
        }

    def _ensure_training_backend_supported(self, *, operation: str) -> None:
        capability = self._training_backend_capability()
        if capability["supported"]:
            return
        raise NotImplementedError(
            f"Training backend `{self.config.training_backend}` is unavailable for {operation}. "
            f"{capability['notes']}"
        )

    def _run_benchmark_stage(
        self,
        stages: dict[str, dict[str, Any]],
        stage_name: str,
        runner,
    ) -> dict[str, Any]:
        started_at = self._timestamp_now()
        started = perf_counter()
        with ProcessMemorySampler() as memory_sampler:
            try:
                result = runner()
            except Exception as error:
                duration_seconds = perf_counter() - started
                stages[stage_name] = {
                    "started_at": started_at,
                    "duration_seconds": duration_seconds,
                    "peak_rss_mb": bytes_to_megabytes(memory_sampler.peak_rss_bytes),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                raise

        duration_seconds = perf_counter() - started
        stages[stage_name] = {
            "started_at": started_at,
            "duration_seconds": duration_seconds,
            "peak_rss_mb": bytes_to_megabytes(memory_sampler.peak_rss_bytes),
            "status": "completed",
        }
        if isinstance(result, dict):
            stages[stage_name]["result_path"] = self._stage_result_path(result)
            return result
        return {"value": result}

    def _compare_benchmark_path(
        self,
        compare_to: Path,
        *,
        current: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if current is None:
            return compare_benchmark_reports(compare_to, self._benchmark_report_path())

        temp_path = self._benchmark_report_path().with_name(
            f"{self.config.experiment_name}_benchmark.preview.json"
        )
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            return compare_benchmark_reports(compare_to, temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _stage_result_path(self, result: dict[str, Any]) -> str | None:
        for key in (
            "training_report_path",
            "evaluation_report_path",
            "export_report_path",
            "artifact_bundle_path",
            "benchmark_report_path",
            "checkpoint_path",
        ):
            value = result.get(key)
            if value is not None:
                return str(value)
        return None

    def evaluate(
        self,
        *,
        manifest_path: Path | None = None,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any]:
        self._ensure_training_backend_supported(operation="evaluate")
        import torch.nn as nn

        frame = self.load_manifest(manifest_path)
        checkpoint_source = (
            checkpoint_path
            or self.config.resume_from_checkpoint
            or self._checkpoint_path()
        )
        payload = self._load_checkpoint_payload(checkpoint_source)
        overrides = self._checkpoint_training_overrides(payload)

        with self._temporary_config(**overrides):
            cache_summary = self.warm_image_cache(manifest_frame=frame)
            cache_dir = Path(cache_summary["cache_dir"]) if cache_summary is not None else None
            loaders, label_to_index = self.build_dataloaders(
                manifest_frame=frame,
                cache_dir=cache_dir,
            )
            class_names = list(payload.get("labels") or self.class_names(frame))
            self._require_trainable_classes(class_names)
            split_name = self._evaluation_split_name(loaders)
            if split_name is None:
                raise RuntimeError("Evaluation requires a non-empty test or validation split.")

            model = self.build_model_stub(num_classes=len(class_names))
            state_dict = self._checkpoint_state_dict(payload, checkpoint_path=checkpoint_source)
            if state_dict is None:
                raise ValueError(f"Checkpoint has no model weights: {checkpoint_source}")
            model.load_state_dict(state_dict)

            device = self._resolve_device()
            model.to(device)
            criterion = nn.CrossEntropyLoss(
                weight=self._loss_class_weights(frame, label_to_index, device=device)
            )
            metrics = self._run_epoch(
                model=model,
                loader=loaders[split_name],
                criterion=criterion,
                optimizer=None,
                device=device,
                training=False,
                stage_name=split_name,
                collect_predictions=True,
            )

        predictions = metrics.get("predictions")
        targets = metrics.get("targets")
        if not isinstance(predictions, list) or not isinstance(targets, list):
            raise RuntimeError("Evaluation did not return predictions/targets for reporting.")

        evaluation = self._build_classification_report(
            predictions=predictions,
            targets=targets,
            class_names=class_names,
        )
        evaluation.update(
            {
                "schema_version": 1,
                "generated_at": self._timestamp_now(),
                "experiment_name": self.config.experiment_name,
                "checkpoint_path": str(checkpoint_source),
                "split": split_name,
                "loss": float(metrics["loss"]),
                "device": self._resolve_device(),
            }
        )
        report_path, confusion_path = self._write_evaluation_artifacts(evaluation)
        summary = {
            "checkpoint_path": str(checkpoint_source),
            "training_backend": self.config.training_backend,
            "evaluation_report_path": str(report_path),
            "confusion_matrix_path": str(confusion_path),
            "split": split_name,
            "loss": float(metrics["loss"]),
            "accuracy": float(evaluation["accuracy"]),
            "macro_f1": float(evaluation["macro_f1"]),
            "weighted_f1": float(evaluation["weighted_f1"]),
            "num_samples": int(evaluation["num_samples"]),
            "cache": cache_summary,
        }
        return summary

    def export_onnx(
        self,
        *,
        manifest_path: Path | None = None,
        checkpoint_path: Path | None = None,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        self._ensure_training_backend_supported(operation="export-onnx")
        import onnx
        import onnxruntime as ort
        import torch

        frame = (
            self.load_manifest(manifest_path)
            if (manifest_path or self.config.manifest_path.exists())
            else None
        )
        checkpoint_source = (
            checkpoint_path
            or self.config.resume_from_checkpoint
            or self._checkpoint_path()
        )
        payload = self._load_checkpoint_payload(checkpoint_source)
        overrides = self._checkpoint_training_overrides(payload)
        class_names = list(
            payload.get("labels") or (self.class_names(frame) if frame is not None else [])
        )
        self._require_trainable_classes(class_names)
        destination = output_path or self.config.onnx_output_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        with self._temporary_config(**overrides):
            model = self.build_model_stub(num_classes=len(class_names))
            state_dict = self._checkpoint_state_dict(payload, checkpoint_path=checkpoint_source)
            if state_dict is None:
                raise ValueError(f"Checkpoint has no model weights: {checkpoint_source}")
            model.load_state_dict(state_dict)
            model.eval()
            model.cpu()

            dummy_input = torch.randn(
                self.config.export_batch_size,
                3,
                self.config.image_size,
                self.config.image_size,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                pytorch_logits = model(dummy_input).detach().cpu().numpy()

            torch.onnx.export(
                model,
                dummy_input,
                str(destination),
                export_params=True,
                do_constant_folding=True,
                dynamo=False,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={
                    "images": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                },
                opset_version=self.config.onnx_opset,
            )

            onnx_model = onnx.load(str(destination))
            onnx.checker.check_model(onnx_model)

            verification: dict[str, Any]
            if self.config.verify_onnx:
                session = ort.InferenceSession(
                    str(destination),
                    providers=["CPUExecutionProvider"],
                )
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                onnx_logits = np.asarray(
                    session.run(
                        [output_name],
                        {input_name: dummy_input.detach().cpu().numpy()},
                    )[0]
                )
                max_abs_diff = float(np.max(np.abs(onnx_logits - pytorch_logits)))
                if max_abs_diff > 1e-3:
                    raise ValueError(
                        "ONNX verification failed because PyTorch and ONNX outputs diverged "
                        f"(max_abs_diff={max_abs_diff:.6f})."
                    )
                verification = {
                    "verified": True,
                    "provider": "CPUExecutionProvider",
                    "input_name": input_name,
                    "output_name": output_name,
                    "max_abs_diff": max_abs_diff,
                }
            else:
                verification = {"verified": False, "skipped": True}

        labels_path = self._write_labels_payload(class_names)
        export_summary = {
            "schema_version": 1,
            "generated_at": self._timestamp_now(),
            "experiment_name": self.config.experiment_name,
            "training_backend": self.config.training_backend,
            "checkpoint_path": str(checkpoint_source),
            "onnx_path": str(destination),
            "labels_path": str(labels_path),
            "model_name": str(overrides.get("model_name", self.config.model_name)),
            "image_size": int(overrides.get("image_size", self.config.image_size)),
            "normalization_preset": str(
                overrides.get("normalization_preset", self.config.normalization_preset)
            ),
            "onnx_opset": self.config.onnx_opset,
            "input_spec": {
                "name": "images",
                "dtype": "float32",
                "shape": [
                    self.config.export_batch_size,
                    3,
                    self.config.image_size,
                    self.config.image_size,
                ],
                "dynamic_axes": {"0": "batch_size"},
            },
            "output_spec": {"name": "logits", "dtype": "float32"},
            "num_classes": len(class_names),
            "labels": class_names,
            "verification": verification,
        }
        model_manifest = self._build_model_manifest(
            export_summary=export_summary,
            checkpoint_payload=payload,
        )
        self._write_json(self._export_report_path(), export_summary)
        self._write_json(self._model_manifest_path(), model_manifest)
        export_summary["export_report_path"] = str(self._export_report_path())
        export_summary["model_manifest_path"] = str(self._model_manifest_path())
        return export_summary

    def build_artifact_report(self, manifest_path: Path | None = None) -> dict[str, Any]:
        frame = (
            self.load_manifest(manifest_path)
            if (manifest_path or self.config.manifest_path.exists())
            else None
        )
        report = {
            "schema_version": 1,
            "generated_at": self._timestamp_now(),
            "experiment_name": self.config.experiment_name,
            "training_backend": self.config.training_backend,
            "artifact_bundle_path": str(self._artifact_bundle_path()),
            "training_plan": self.summarize_training_plan(manifest_frame=frame),
            "dataset_summary": self._read_json(
                self.paths.artifact_dir / "reports" / "summary.json"
            ),
            "pseudo_label": self._read_json(self._pseudo_label_report_path()),
            "training": self._read_json(self._training_report_path()),
            "evaluation": self._read_json(self._evaluation_report_path()),
            "export": self._read_json(self._export_report_path()),
            "benchmark": self._read_json(self._benchmark_report_path()),
            "experiment_spec": self._read_json(self._experiment_spec_path()),
            "model_manifest": self._read_json(self._model_manifest_path()),
        }
        self._write_json(self._artifact_bundle_path(), report)
        return report

    def pseudo_label(
        self,
        *,
        manifest_path: Path | None = None,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any]:
        self._ensure_training_backend_supported(operation="pseudo-label")
        import torch

        frame = self.load_manifest(manifest_path)
        checkpoint_source = (
            checkpoint_path
            or self.config.resume_from_checkpoint
            or self._checkpoint_path()
        )
        payload = self._load_checkpoint_payload(checkpoint_source)
        overrides = self._checkpoint_training_overrides(payload)
        class_names = list(payload.get("labels") or self.class_names(frame))
        self._require_trainable_classes(class_names)
        candidates = list(self._pseudo_label_candidates(frame).iter_rows(named=True))

        summary: dict[str, Any] = {
            "schema_version": 1,
            "generated_at": self._timestamp_now(),
            "experiment_name": self.config.experiment_name,
            "training_backend": self.config.training_backend,
            "manifest_path": str(self.config.manifest_path),
            "checkpoint_path": str(checkpoint_source),
            "candidate_count": len(candidates),
            "accepted_count": 0,
            "rejected_count": 0,
            "confidence_threshold": self.config.pseudo_label_confidence_threshold,
            "margin_threshold": self.config.pseudo_label_margin_threshold,
            "pseudo_label_report_path": str(self._pseudo_label_report_path()),
        }
        if not candidates:
            self._write_json(self._pseudo_label_report_path(), summary)
            return summary

        assignments: dict[str, dict[str, Any]] = {}
        accepted_scores: list[float] = []
        with self._temporary_config(**overrides):
            model = self.build_model_stub(num_classes=len(class_names))
            state_dict = self._checkpoint_state_dict(payload, checkpoint_path=checkpoint_source)
            if state_dict is None:
                raise ValueError(f"Checkpoint has no model weights: {checkpoint_source}")
            model.load_state_dict(state_dict)
            device = self._resolve_device()
            model.to(device)
            model.eval()
            transform = build_training_transforms(
                self.config.image_size,
                normalization_preset=self.config.normalization_preset,
            )
            progress = TerminalProgressBar(
                total=len(candidates),
                description="pseudo label",
                enabled=self.config.show_progress,
            )
            for row in candidates:
                image = load_rgb_image(Path(str(row["image_path"])))
                tensor = transform(image).unsqueeze(0).to(device)
                with torch.inference_mode():
                    probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
                ranking = np.argsort(probabilities)[::-1]
                top1_index = int(ranking[0])
                top2_score = float(probabilities[int(ranking[1])]) if ranking.size > 1 else 0.0
                score = float(probabilities[top1_index])
                margin = float(score - top2_score)
                predicted_label = str(class_names[top1_index])
                accepted = (
                    score >= self.config.pseudo_label_confidence_threshold
                    and margin >= self.config.pseudo_label_margin_threshold
                )
                if accepted:
                    accepted_scores.append(score)
                assignments[str(row["sample_id"])] = {
                    "predicted_label": predicted_label,
                    "score": score,
                    "margin": margin,
                    "accepted": accepted,
                }
                progress.advance(postfix=str(row["file_name"]))
            progress.close(
                summary=(
                    "Pseudo-label summary: "
                    f"accepted={sum(1 for value in assignments.values() if value['accepted'])}, "
                    f"rejected={sum(1 for value in assignments.values() if not value['accepted'])}"
                )
            )

        updated_records: list[dict[str, Any]] = []
        accepted_count = 0
        rejected_count = 0
        for row in frame.iter_rows(named=True):
            record = dict(row)
            assignment = assignments.get(str(record["sample_id"]))
            if assignment is None:
                updated_records.append(record)
                continue
            if bool(assignment["accepted"]):
                accepted_count += 1
                updated_records.append(
                    self._apply_manifest_label_update(
                        record,
                        label=str(assignment["predicted_label"]),
                        status="pseudo_labeled",
                        label_source="model_pseudo",
                        review_status="pseudo_accepted",
                        suggested_label=str(assignment["predicted_label"]),
                        suggested_label_source="model_pseudo",
                        pseudo_label_score=float(assignment["score"]),
                        pseudo_label_margin=float(assignment["margin"]),
                    )
                )
                continue
            rejected_count += 1
            record["suggested_label"] = str(assignment["predicted_label"])
            record["suggested_label_source"] = "model_pseudo"
            record["pseudo_label_score"] = float(assignment["score"])
            record["pseudo_label_margin"] = float(assignment["margin"])
            record["review_status"] = "pseudo_rejected"
            updated_records.append(record)

        pipeline = self._dataset_pipeline()
        updated_frame = pipeline.assign_splits(pipeline._frame_from_records(updated_records))
        pipeline.write_manifest(updated_frame)
        report_summary = pipeline.generate_reports(updated_frame)
        summary.update(
            {
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "average_accepted_score": (
                    float(sum(accepted_scores) / len(accepted_scores))
                    if accepted_scores
                    else None
                ),
                "training_ready_files": int(report_summary["training_ready_files"]),
                "effective_training_mode": report_summary.get("effective_training_mode"),
            }
        )
        self._write_json(self._pseudo_label_report_path(), summary)
        return summary

    def fit(
        self,
        manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        self._ensure_training_backend_supported(operation="fit")
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

        class_names = self.class_names(frame)
        self._require_trainable_classes(class_names)
        model = self.build_model_stub(num_classes=len(label_to_index))
        device = self._resolve_device()
        print(f"Training on device: {device}")
        model.to(device)

        criterion = nn.CrossEntropyLoss(
            weight=self._loss_class_weights(frame, label_to_index, device=device)
        )
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history: list[dict[str, float | int]] = []
        best_epoch = 0
        best_loss = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())
        last_completed_epoch = 0
        epochs_without_improvement = 0
        stopped_early = False
        interrupted = False
        stop_reason: str | None = None
        evaluation_summary: dict[str, Any] | None = None
        best_checkpoint_path = self._checkpoint_path()
        latest_checkpoint_path = self._latest_checkpoint_path()

        if self.config.resume_from_checkpoint is not None:
            resume_state = self._load_resume_state(
                checkpoint_path=self.config.resume_from_checkpoint,
                model=model,
                optimizer=optimizer,
                class_names=class_names,
                device=device,
            )
            history = resume_state["history"]
            best_epoch = int(resume_state["best_epoch"])
            best_loss = float(resume_state["best_loss"])
            best_state_dict = copy.deepcopy(resume_state["best_model_state_dict"])
            last_completed_epoch = int(resume_state["last_completed_epoch"])
            if self.config.epochs <= last_completed_epoch:
                raise ValueError(
                    "Requested epochs must be greater than the completed epochs in the "
                    f"resume checkpoint ({last_completed_epoch})."
                )

        if not self.config.show_progress:
            print(
                "Progress bars disabled; reporting text progress snapshots "
                "during training and epoch summaries."
            )

        try:
            for epoch in range(last_completed_epoch + 1, self.config.epochs + 1):
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
                last_completed_epoch = epoch
                self._save_training_checkpoint(
                    checkpoint_path=latest_checkpoint_path,
                    model_state_dict=copy.deepcopy(model.state_dict()),
                    optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                    labels=class_names,
                    history=history,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                    best_model_state_dict=copy.deepcopy(best_state_dict),
                    last_completed_epoch=last_completed_epoch,
                    target_epochs=self.config.epochs,
                    checkpoint_kind="last",
                    interrupted=False,
                    stopped_early=False,
                    stop_reason=None,
                )
                if best_epoch == epoch:
                    self._save_training_checkpoint(
                        checkpoint_path=best_checkpoint_path,
                        model_state_dict=copy.deepcopy(best_state_dict),
                        optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                        labels=class_names,
                        history=history,
                        best_epoch=best_epoch,
                        best_loss=best_loss,
                        best_model_state_dict=copy.deepcopy(best_state_dict),
                        last_completed_epoch=last_completed_epoch,
                        target_epochs=self.config.epochs,
                        checkpoint_kind="best",
                        interrupted=False,
                        stopped_early=False,
                        stop_reason=None,
                    )

                self._print_epoch_summary(
                    epoch_summary,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                )

                if self._should_stop_early(epochs_without_improvement):
                    stopped_early = True
                    stop_reason = (
                        "Early stopping triggered after "
                        f"{epochs_without_improvement} epoch(s) without improvement "
                        f"greater than {self.config.early_stopping_min_delta}."
                    )
                    print(stop_reason)
                    break
        except KeyboardInterrupt:
            interrupted = True
            stop_reason = (
                "Training interrupted by user. "
                f"Saved latest checkpoint to {latest_checkpoint_path}."
            )
            print(stop_reason)
            self._save_training_checkpoint(
                checkpoint_path=latest_checkpoint_path,
                model_state_dict=copy.deepcopy(model.state_dict()),
                optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                labels=class_names,
                history=history,
                best_epoch=best_epoch,
                best_loss=best_loss,
                best_model_state_dict=copy.deepcopy(best_state_dict),
                last_completed_epoch=last_completed_epoch,
                target_epochs=self.config.epochs,
                checkpoint_kind="last",
                interrupted=True,
                stopped_early=False,
                stop_reason=stop_reason,
            )

        labels_path = self.export_label_index(manifest_frame=frame)

        summary = {
            "schema_version": 1,
            "generated_at": self._timestamp_now(),
            "checkpoint_path": str(best_checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "training_preset": self.config.training_preset,
            "training_backend": self.config.training_backend,
            "experiment_name": self.config.experiment_name,
            "labels_path": str(labels_path),
            "training_report_path": str(self._training_report_path()),
            "benchmark_report_path": str(self._benchmark_report_path()),
            "experiment_spec_path": str(self._experiment_spec_path()),
            "num_classes": len(label_to_index),
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "interrupted": interrupted,
            "stop_reason": stop_reason,
            "history": history,
            "device": device,
            "model": self._last_model_metadata,
            "resume_from_checkpoint": (
                None
                if self.config.resume_from_checkpoint is None
                else str(self.config.resume_from_checkpoint)
            ),
            "cache_format": self.config.cache_format,
            "class_bias_strategy": self.config.class_bias_strategy,
            "class_weight_map": self.class_weight_map(frame),
            "cache": cache_summary,
        }

        if interrupted:
            self._write_json(self._training_report_path(), summary)
            return summary

        model.load_state_dict(best_state_dict)

        if "test" in loaders:
            test_metrics = self._run_epoch(
                model=model,
                loader=loaders["test"],
                criterion=criterion,
                optimizer=None,
                device=device,
                training=False,
                stage_name="test",
                collect_predictions=True,
            )
            summary["test_loss"] = float(test_metrics["loss"])
            summary["test_accuracy"] = float(test_metrics["accuracy"])
            predictions = test_metrics.get("predictions")
            targets = test_metrics.get("targets")
            if isinstance(predictions, list) and isinstance(targets, list):
                evaluation = self._build_classification_report(
                    predictions=predictions,
                    targets=targets,
                    class_names=class_names,
                )
                evaluation.update(
                    {
                        "schema_version": 1,
                        "generated_at": self._timestamp_now(),
                        "experiment_name": self.config.experiment_name,
                        "checkpoint_path": str(best_checkpoint_path),
                        "split": "test",
                        "loss": float(test_metrics["loss"]),
                        "device": device,
                    }
                )
                report_path, confusion_path = self._write_evaluation_artifacts(evaluation)
                evaluation_summary = {
                    "accuracy": float(evaluation["accuracy"]),
                    "macro_f1": float(evaluation["macro_f1"]),
                    "weighted_f1": float(evaluation["weighted_f1"]),
                }
                summary["evaluation_report_path"] = str(report_path)
                summary["confusion_matrix_path"] = str(confusion_path)
                summary["evaluation_summary"] = evaluation_summary

        self._save_training_checkpoint(
            checkpoint_path=best_checkpoint_path,
            model_state_dict=copy.deepcopy(best_state_dict),
            optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
            labels=class_names,
            history=history,
            best_epoch=best_epoch,
            best_loss=best_loss,
            best_model_state_dict=copy.deepcopy(best_state_dict),
            last_completed_epoch=last_completed_epoch,
            target_epochs=self.config.epochs,
            checkpoint_kind="best",
            interrupted=False,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            evaluation_summary=evaluation_summary,
        )

        self._write_json(self._training_report_path(), summary)
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
            cache_format=self.config.cache_format,
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

    def _ensure_manifest_columns(self, frame: pl.DataFrame) -> pl.DataFrame:
        result = frame
        if "raw_label" not in result.columns:
            result = result.with_columns(pl.col("label").alias("raw_label"))
        if "curated_label" not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.String).alias("curated_label"))
        if "suggested_label" not in result.columns:
            result = result.with_columns(
                pl.when(pl.col("label") != "unknown")
                .then(pl.col("label"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("suggested_label")
            )
        if "suggested_label_source" not in result.columns:
            result = result.with_columns(
                pl.when(pl.col("label") != "unknown")
                .then(pl.lit("filename"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("suggested_label_source")
            )
        if "label_source" not in result.columns:
            result = result.with_columns(
                pl.when(pl.col("label") != "unknown")
                .then(pl.lit("filename"))
                .otherwise(pl.lit("unknown"))
                .alias("label_source")
            )
        if "annotation_status" not in result.columns:
            result = result.with_columns(
                pl.when(pl.col("label") != "unknown")
                .then(pl.lit("inferred"))
                .otherwise(pl.lit("unlabeled"))
                .alias("annotation_status")
            )
        if "annotated_at" not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.String).alias("annotated_at"))
        if "pseudo_label_score" not in result.columns:
            result = result.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("pseudo_label_score")
            )
        if "pseudo_label_margin" not in result.columns:
            result = result.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("pseudo_label_margin")
            )
        if "review_status" not in result.columns:
            result = result.with_columns(pl.lit("unreviewed").alias("review_status"))
        return result

    def _labeled_frame(self, manifest_frame: pl.DataFrame) -> pl.DataFrame:
        frame = self._ensure_manifest_columns(manifest_frame)
        base_filter = (
            pl.col("is_valid")
            & (pl.col("label") != "unknown")
            & pl.col("annotation_status").is_in(["inferred", "labeled", "pseudo_labeled"])
        )
        if self._has_accepted_labels(frame):
            return frame.filter(
                base_filter
                & pl.col("label_source").is_in(sorted(ACCEPTED_LABEL_SOURCES))
                & pl.col("annotation_status").is_in(sorted(ACCEPTED_ANNOTATION_STATUSES))
            )
        return frame.filter(base_filter)

    def _require_trainable_classes(self, class_names: list[str]) -> None:
        if len(class_names) < 2:
            raise RuntimeError(
                "Training-ready manifest must contain at least 2 classes. "
                f"Found {len(class_names)} class(es): {class_names}. "
                "If your filenames only infer one label, export the labeling template "
                "and import curated labels first."
            )

    def _has_accepted_labels(self, manifest_frame: pl.DataFrame) -> bool:
        frame = self._ensure_manifest_columns(manifest_frame)
        return (
            frame.filter(
                pl.col("is_valid")
                & (pl.col("label") != "unknown")
                & pl.col("label_source").is_in(sorted(ACCEPTED_LABEL_SOURCES))
                & pl.col("annotation_status").is_in(sorted(ACCEPTED_ANNOTATION_STATUSES))
            ).height
            > 0
        )

    def _effective_training_mode(self, manifest_frame: pl.DataFrame) -> str:
        return (
            "accepted_labels_only"
            if self._has_accepted_labels(manifest_frame)
            else "weak_inferred"
        )

    def _pseudo_label_candidates(self, manifest_frame: pl.DataFrame) -> pl.DataFrame:
        frame = self._ensure_manifest_columns(manifest_frame)
        return frame.filter(
            pl.col("is_valid")
            & pl.col("label_source").is_in(["unknown", "filename"])
            & ~pl.col("annotation_status").is_in(["excluded", "labeled", "pseudo_labeled"])
        )

    def _apply_manifest_label_update(
        self,
        record: dict[str, Any],
        *,
        label: str,
        status: str,
        label_source: str,
        review_status: str,
        suggested_label: str | None,
        suggested_label_source: str | None,
        pseudo_label_score: float | None = None,
        pseudo_label_margin: float | None = None,
    ) -> dict[str, Any]:
        next_record = dict(record)
        next_record["label"] = label if label else "unknown"
        next_record["label_source"] = label_source if label else "unknown"
        next_record["annotation_status"] = status
        next_record["annotated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        next_record["suggested_label"] = suggested_label
        next_record["suggested_label_source"] = suggested_label_source
        next_record["review_status"] = review_status
        next_record["pseudo_label_score"] = pseudo_label_score
        next_record["pseudo_label_margin"] = pseudo_label_margin
        return next_record

    def _dataset_pipeline(self):
        from localagent.data import DatasetPipeline

        return DatasetPipeline(
            DatasetPipelineConfig(
                raw_dataset_dir=self.paths.project_root / "dataset",
                manifest_dir=self.config.manifest_path.parent,
                report_dir=self.paths.artifact_dir / "reports",
                manifest_name=self.config.manifest_path.name,
                manifest_csv_name=f"{self.config.manifest_path.stem}.csv",
                labeling_template_name=DatasetPipelineConfig().labeling_template_name,
                cluster_review_template_name=DatasetPipelineConfig().cluster_review_template_name,
                embeddings_name=DatasetPipelineConfig().embeddings_name,
                show_progress=self.config.show_progress,
            )
        )

    def _write_labels_payload(
        self,
        labels: list[str],
        output_path: Path | None = None,
    ) -> Path:
        destination = output_path or self.config.labels_output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        return destination

    def _load_checkpoint_payload(self, checkpoint_path: Path) -> dict[str, Any]:
        import torch

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
        return payload

    def _checkpoint_state_dict(
        self,
        payload: dict[str, Any],
        *,
        checkpoint_path: Path,
    ) -> dict[str, Any] | None:
        if "best_model_state_dict" in payload:
            state_dict = payload["best_model_state_dict"]
            if state_dict is not None:
                return state_dict
        if "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
            if state_dict is not None:
                return state_dict
        return None

    def _checkpoint_training_overrides(self, payload: dict[str, Any]) -> dict[str, Any]:
        training_config = payload.get("training_config")
        if not isinstance(training_config, dict):
            training_config = {}
        return {
            "model_name": str(training_config.get("model_name", self.config.model_name)),
            "image_size": int(training_config.get("image_size", self.config.image_size)),
            "normalization_preset": str(
                training_config.get("normalization_preset", self.config.normalization_preset)
            ),
            "cache_format": str(training_config.get("cache_format", self.config.cache_format)),
        }

    @contextmanager
    def _temporary_config(self, **overrides: Any):
        original_values = {
            key: getattr(self.config, key)
            for key, value in overrides.items()
            if value is not None and hasattr(self.config, key)
        }
        for key, value in overrides.items():
            if value is not None and hasattr(self.config, key):
                setattr(self.config, key, value)
        try:
            yield
        finally:
            for key, value in original_values.items():
                setattr(self.config, key, value)

    def _evaluation_split_name(self, loaders: dict[str, Any]) -> str | None:
        if "test" in loaders:
            return "test"
        if "val" in loaders:
            return "val"
        return None

    def _build_model_manifest(
        self,
        *,
        export_summary: dict[str, Any],
        checkpoint_payload: dict[str, Any],
    ) -> dict[str, Any]:
        mean, std = normalization_stats(str(export_summary["normalization_preset"]))
        evaluation_payload = self._read_json(self._evaluation_report_path())
        evaluation_summary = checkpoint_payload.get("evaluation_summary")
        if evaluation_summary is None and isinstance(evaluation_payload, dict):
            evaluation_summary = {
                "accuracy": evaluation_payload.get("accuracy"),
                "macro_f1": evaluation_payload.get("macro_f1"),
                "weighted_f1": evaluation_payload.get("weighted_f1"),
            }
        return {
            "schema_version": 1,
            "generated_at": self._timestamp_now(),
            "experiment_name": self.config.experiment_name,
            "model_name": export_summary["model_name"],
            "onnx_path": export_summary["onnx_path"],
            "labels_path": export_summary["labels_path"],
            "checkpoint_path": export_summary["checkpoint_path"],
            "labels": export_summary["labels"],
            "image_size": export_summary["image_size"],
            "normalization": {
                "preset": export_summary["normalization_preset"],
                "mean": list(mean),
                "std": list(std),
            },
            "onnx": {
                "opset": export_summary["onnx_opset"],
                "input_spec": export_summary["input_spec"],
                "output_spec": export_summary["output_spec"],
                "verification": export_summary["verification"],
            },
            "evaluation_summary": evaluation_summary,
        }

    def _write_json(self, output_path: Path, payload: dict[str, Any]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def _timestamp_now(self) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def _resolve_device(self) -> str:
        import torch

        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _checkpoint_path(self) -> Path:
        return self.config.checkpoint_dir / f"{self.config.experiment_name}.pt"

    def _latest_checkpoint_path(self) -> Path:
        return self.config.checkpoint_dir / f"{self.config.experiment_name}.last.pt"

    def _evaluation_report_path(self) -> Path:
        return (
            self.paths.artifact_dir
            / "reports"
            / f"{self.config.experiment_name}_evaluation.json"
        )

    def _confusion_matrix_path(self) -> Path:
        return (
            self.paths.artifact_dir
            / "reports"
            / f"{self.config.experiment_name}_confusion_matrix.csv"
        )

    def _training_report_path(self) -> Path:
        return self.paths.artifact_dir / "reports" / f"{self.config.experiment_name}_training.json"

    def _experiment_spec_path(self) -> Path:
        return (
            self.paths.artifact_dir
            / "reports"
            / f"{self.config.experiment_name}_experiment_spec.json"
        )

    def _benchmark_report_path(self) -> Path:
        return (
            self.paths.artifact_dir
            / "reports"
            / f"{self.config.experiment_name}_benchmark.json"
        )

    def _pseudo_label_report_path(self) -> Path:
        return (
            self.paths.artifact_dir
            / "reports"
            / f"{self.config.experiment_name}_pseudo_label.json"
        )

    def _export_report_path(self) -> Path:
        return self.paths.artifact_dir / "reports" / f"{self.config.experiment_name}_export.json"

    def _artifact_bundle_path(self) -> Path:
        return self.paths.artifact_dir / "reports" / f"{self.config.experiment_name}_report.json"

    def _model_manifest_path(self) -> Path:
        return self.config.model_manifest_output_path

    def _cache_dir(self) -> Path:
        suffix = "" if self.config.cache_format == "png" else f"-{self.config.cache_format}"
        return self.config.cache_dir / f"{self.config.image_size}px{suffix}"

    def _cache_failure_report_path(self) -> Path:
        suffix = "" if self.config.cache_format == "png" else f"_{self.config.cache_format}"
        return (
            self.paths.artifact_dir
            / "reports"
            / f"training_cache_failures_{self.config.image_size}px{suffix}.json"
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
                output_path=cache_dir / f"{record['sample_id']}{self._cache_file_suffix()}",
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

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.config.cache_format == "raw":
                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).tofile(output_path)
                return True

            success, buffer = cv2.imencode(".png", resized)
            if not success:
                return False
            buffer.tofile(output_path)
            return True
        except Exception:
            return False

    def _cache_file_suffix(self) -> str:
        return ".raw" if self.config.cache_format == "raw" else ".png"

    def _build_train_sampler(
        self,
        dataset: ManifestImageDataset,
        *,
        weight_map: dict[str, float],
    ):
        if self.config.class_bias_strategy not in {"sampler", "both"}:
            return None

        import torch
        from torch.utils.data import WeightedRandomSampler

        sample_weights = [float(weight_map[str(record["label"])]) for record in dataset.records]
        if not sample_weights:
            return None
        return WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )

    def _loss_class_weights(
        self,
        manifest_frame: pl.DataFrame,
        label_to_index: dict[str, int],
        *,
        device: str,
    ):
        if self.config.class_bias_strategy not in {"loss", "both"}:
            return None

        import torch

        weight_map = self.class_weight_map(manifest_frame)
        weights = torch.ones(len(label_to_index), dtype=torch.float32, device=device)
        for label, index in label_to_index.items():
            weights[index] = float(weight_map.get(label, 1.0))
        return weights

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
        collect_predictions: bool = False,
    ) -> dict[str, Any]:
        import torch

        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        predictions: list[int] = []
        targets: list[int] = []

        context = torch.enable_grad() if training else torch.inference_mode()
        progress = TerminalProgressBar(
            total=len(loader),
            description=stage_name,
            enabled=self.config.show_progress,
        )
        log_interval = self._batch_log_interval(len(loader))
        with context:
            for batch_index, (images, labels) in enumerate(loader, start=1):
                images = images.to(device)
                labels = labels.to(device)

                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                logits = model(images)
                loss = criterion(logits, labels)
                predicted = logits.argmax(dim=1)

                if training and optimizer is not None:
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.item()) * int(labels.size(0))
                total_correct += int((predicted == labels).sum().item())
                total_examples += int(labels.size(0))
                if collect_predictions:
                    predictions.extend(predicted.detach().cpu().tolist())
                    targets.extend(labels.detach().cpu().tolist())

                running_loss = total_loss / max(total_examples, 1)
                running_accuracy = total_correct / max(total_examples, 1)
                progress.update(
                    batch_index,
                    postfix=f"loss={running_loss:.4f} acc={running_accuracy:.3f}",
                )
                if self._should_log_batch_snapshot(
                    batch_index=batch_index,
                    total_batches=len(loader),
                    log_interval=log_interval,
                ):
                    self._print_batch_snapshot(
                        stage_name=stage_name,
                        batch_index=batch_index,
                        total_batches=len(loader),
                        loss=running_loss,
                        accuracy=running_accuracy,
                    )

        if total_examples == 0:
            progress.close(summary=f"{stage_name}: no samples")
            return {"loss": 0.0, "accuracy": 0.0}

        metrics = {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }
        if collect_predictions:
            metrics["predictions"] = predictions
            metrics["targets"] = targets
        progress.close(
            summary=(
                f"{stage_name}: loss={metrics['loss']:.4f} "
                f"accuracy={metrics['accuracy']:.3f}"
            )
        )
        return metrics

    def _print_epoch_summary(
        self,
        epoch_summary: dict[str, float | int],
        *,
        best_epoch: int,
        best_loss: float,
    ) -> None:
        if self.config.show_progress:
            return

        epoch = int(epoch_summary["epoch"])
        progress_pct = (epoch / max(self.config.epochs, 1)) * 100.0
        parts = [
            f"epoch {epoch}/{self.config.epochs}",
            f"{progress_pct:.2f}%",
            f"train_loss={float(epoch_summary['train_loss']):.4f}",
            f"train_acc={float(epoch_summary['train_accuracy']):.3f}",
        ]
        if "val_loss" in epoch_summary:
            parts.append(f"val_loss={float(epoch_summary['val_loss']):.4f}")
        if "val_accuracy" in epoch_summary:
            parts.append(f"val_acc={float(epoch_summary['val_accuracy']):.3f}")
        parts.append(f"best_epoch={best_epoch}")
        parts.append(f"best_loss={best_loss:.4f}")
        print(" | ".join(parts))

    def _batch_log_interval(self, total_batches: int) -> int:
        if total_batches <= 0:
            return 1
        return max(1, total_batches // 10)

    def _should_log_batch_snapshot(
        self,
        *,
        batch_index: int,
        total_batches: int,
        log_interval: int,
    ) -> bool:
        if self.config.show_progress or total_batches <= 0:
            return False
        return (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % max(log_interval, 1) == 0
        )

    def _print_batch_snapshot(
        self,
        *,
        stage_name: str,
        batch_index: int,
        total_batches: int,
        loss: float,
        accuracy: float,
    ) -> None:
        if self.config.show_progress:
            return
        progress_pct = (batch_index / max(total_batches, 1)) * 100.0
        print(
            f"{stage_name} | batch {batch_index}/{total_batches} | "
            f"{progress_pct:.2f}% | loss={loss:.4f} acc={accuracy:.3f}"
        )

    def _should_stop_early(self, epochs_without_improvement: int) -> bool:
        if not self.config.enable_early_stopping:
            return False
        if self.config.early_stopping_patience <= 0:
            return False
        return epochs_without_improvement >= self.config.early_stopping_patience
