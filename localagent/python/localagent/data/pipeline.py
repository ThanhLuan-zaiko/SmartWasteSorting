from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from localagent.config import DatasetPipelineConfig
from localagent.data.discovery import EmbeddingArtifact, cluster_embeddings, extract_embeddings
from localagent.utils import TerminalProgressBar

MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "sample_id": pl.String,
    "image_path": pl.String,
    "relative_path": pl.String,
    "file_name": pl.String,
    "extension": pl.String,
    "file_size": pl.Int64,
    "width": pl.Int64,
    "height": pl.Int64,
    "channels": pl.Int64,
    "decode_ok": pl.Boolean,
    "decode_error": pl.String,
    "raw_label": pl.String,
    "curated_label": pl.String,
    "suggested_label": pl.String,
    "suggested_label_source": pl.String,
    "label": pl.String,
    "label_source": pl.String,
    "annotation_status": pl.String,
    "annotated_at": pl.String,
    "pseudo_label_score": pl.Float64,
    "pseudo_label_margin": pl.Float64,
    "review_status": pl.String,
    "cluster_id": pl.Int64,
    "cluster_distance": pl.Float64,
    "cluster_size": pl.Int64,
    "is_cluster_outlier": pl.Boolean,
    "content_hash": pl.String,
    "is_duplicate": pl.Boolean,
    "duplicate_of": pl.String,
    "is_too_small": pl.Boolean,
    "is_valid": pl.Boolean,
    "quarantine_reason": pl.String,
    "split": pl.String,
}

LABEL_PATTERN = re.compile(r"(.+?)[_\- ]\d+$")
ALLOWED_LABEL_STATUSES = {"labeled", "unlabeled", "excluded", "pseudo_labeled"}
TRAINABLE_ANNOTATION_STATUSES = {"inferred", "labeled", "pseudo_labeled"}
ACCEPTED_LABEL_SOURCES = {"curated", "cluster_review", "model_pseudo"}
ACCEPTED_ANNOTATION_STATUSES = {"labeled", "pseudo_labeled"}
CLUSTER_PREVIEW_CLUSTER_LIMIT = 8
CLUSTER_PREVIEW_MEMBER_LIMIT = 4


class DatasetPipeline:
    def __init__(self, config: DatasetPipelineConfig | None = None) -> None:
        self.config = (config or DatasetPipelineConfig()).validate().ensure_layout()

    def scan(self) -> pl.DataFrame:
        image_paths = self._iter_image_paths()
        progress = TerminalProgressBar(
            total=len(image_paths),
            description="scan dataset",
            enabled=self.config.show_progress,
        )
        records: list[dict[str, Any]] = []
        for image_path in image_paths:
            records.append(self._inspect_image(image_path))
            progress.advance(postfix=image_path.name)
        progress.close(summary=f"Scanned {len(records)} images from {self.config.raw_dataset_dir}")

        self._mark_duplicates(records)
        for record in records:
            self._apply_quality_flags(record)
        return self._frame_from_records(records)

    def run_scan(self) -> pl.DataFrame:
        frame = self.scan()
        self.write_manifest(frame)
        return frame

    def assign_splits(self, frame: pl.DataFrame) -> pl.DataFrame:
        normalized_frame = self._ensure_manifest_columns(frame)
        has_accepted_labels = self._has_accepted_labels(normalized_frame)
        eligible_rows = (
            self._training_ready_frame(normalized_frame)
            .select("sample_id", "label")
            .sort(["label", "sample_id"])
            .iter_rows(named=True)
        )
        split_map: dict[str, str] = {}
        grouped_ids: dict[str, list[str]] = {}
        for row in eligible_rows:
            grouped_ids.setdefault(str(row["label"]), []).append(str(row["sample_id"]))

        rng = random.Random(self.config.random_seed)
        for label in sorted(grouped_ids):
            sample_ids = list(grouped_ids[label])
            rng.shuffle(sample_ids)

            train_cutoff = int(len(sample_ids) * self.config.train_ratio)
            val_cutoff = train_cutoff + int(len(sample_ids) * self.config.val_ratio)

            for index, sample_id in enumerate(sample_ids):
                if index < train_cutoff:
                    split_map[sample_id] = "train"
                elif index < val_cutoff:
                    split_map[sample_id] = "val"
                else:
                    split_map[sample_id] = "test"

        frame_without_split = normalized_frame.drop("split")
        return (
            frame_without_split.with_columns(
                pl.struct(["sample_id", "label", "is_valid", "annotation_status", "label_source"])
                .map_elements(
                    lambda row: self._resolve_split(
                        row,
                        split_map,
                        has_accepted_labels=has_accepted_labels,
                    ),
                    return_dtype=pl.String,
                )
                .alias("split")
            )
            .sort("relative_path")
        )

    def run_split(self) -> pl.DataFrame:
        frame = self.load_manifest()
        split_frame = self.assign_splits(frame)
        self.write_manifest(split_frame)
        return split_frame

    def generate_reports(self, frame: pl.DataFrame) -> dict[str, Any]:
        training_ready_frame = self._training_ready_frame(frame)
        accepted_frame = frame.filter(
            pl.col("label_source").is_in(sorted(ACCEPTED_LABEL_SOURCES))
            & (pl.col("label") != self.config.unknown_label)
        )
        summary = {
            "dataset_root": str(self.config.raw_dataset_dir),
            "manifest_path": str(self.config.manifest_path),
            "total_files": frame.height,
            "valid_files": frame.filter(pl.col("is_valid")).height,
            "invalid_files": frame.filter(~pl.col("is_valid")).height,
            "decode_failed_files": frame.filter(~pl.col("decode_ok")).height,
            "too_small_files": frame.filter(pl.col("is_too_small")).height,
            "duplicate_files": frame.filter(pl.col("is_duplicate")).height,
            "training_ready_files": training_ready_frame.height,
            "effective_training_mode": self._effective_training_mode(frame),
            "split_counts": self._count_mapping(frame, "split"),
            "label_counts": self._count_mapping(frame, "label"),
            "trainable_label_counts": self._count_mapping(training_ready_frame, "label"),
            "label_source_counts": self._count_mapping(frame, "label_source"),
            "accepted_label_source_counts": self._count_mapping(accepted_frame, "label_source"),
            "annotation_status_counts": self._count_mapping(frame, "annotation_status"),
            "review_status_counts": self._count_mapping(frame, "review_status"),
            "extension_counts": self._count_mapping(frame, "extension"),
            "quarantine_counts": self._count_mapping(frame, "quarantine_reason"),
            "clustered_files": frame.filter(pl.col("cluster_id").is_not_null()).height,
            "cluster_outlier_files": frame.filter(pl.col("is_cluster_outlier")).height,
            "embedding_artifact_exists": self.config.embeddings_path.is_file(),
            "cluster_summary_exists": self.config.cluster_summary_path.is_file(),
            "width_stats": self._dimension_stats(frame, "width"),
            "height_stats": self._dimension_stats(frame, "height"),
            **self._build_cluster_preview_summary(frame),
        }

        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        self.config.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._write_count_table(frame, "split", self.config.split_summary_path)
        self._write_count_table(frame, "quarantine_reason", self.config.quality_summary_path)
        self._write_count_table(frame, "extension", self.config.extension_summary_path)
        self._write_count_table(frame, "label", self.config.label_summary_path)
        return summary

    def run_report(self) -> dict[str, Any]:
        frame = self.load_manifest()
        return self.generate_reports(frame)

    def embed_dataset(self) -> dict[str, Any]:
        frame = self.load_manifest()
        records = self._discovery_ready_records(frame)
        artifact, summary = extract_embeddings(
            records,
            image_size=224,
            show_progress=self.config.show_progress,
        )
        artifact_path = artifact.save(self.config.embeddings_path)
        payload = {
            **summary,
            "embedding_path": str(artifact_path),
            "manifest_path": str(self.config.manifest_path),
        }
        self.config.embedding_summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.generate_reports(frame)
        return payload

    def cluster_dataset(self, *, requested_clusters: int | None = None) -> dict[str, Any]:
        frame = self.load_manifest()
        records = self._discovery_ready_records(frame)
        if not records:
            raise ValueError("No valid images are available for clustering.")
        embeddings = EmbeddingArtifact.load(self.config.embeddings_path)
        index_by_sample_id = {
            str(sample_id): index for index, sample_id in enumerate(embeddings.sample_ids.tolist())
        }
        missing_ids = [
            str(record["sample_id"])
            for record in records
            if str(record["sample_id"]) not in index_by_sample_id
        ]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                "Embeddings are stale and do not cover the current manifest. "
                f"Re-run `embed`. Missing sample_ids: {preview}"
            )

        cluster_artifact, summary = cluster_embeddings(
            embeddings.vectors,
            seed=self.config.random_seed,
            requested_clusters=requested_clusters,
        )

        updates = {
            str(sample_id): cluster_artifact.assignment_for(index)
            for index, sample_id in enumerate(embeddings.sample_ids.tolist())
        }
        updated_frame = self._apply_cluster_updates(frame, updates)
        self.write_manifest(updated_frame)
        payload = {
            **summary,
            "manifest_path": str(self.config.manifest_path),
            "embedding_path": str(self.config.embeddings_path),
            "cluster_summary_path": str(self.config.cluster_summary_path),
        }
        self.config.cluster_summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.generate_reports(updated_frame)
        return payload

    def export_cluster_review(
        self,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        frame = self.load_manifest()
        destination = output_path or self.config.cluster_review_template_path
        existing_rows = (
            self._load_existing_cluster_review_rows(destination)
            if destination.exists()
            else {}
        )
        rows, stale_reset_count = self._build_cluster_review_rows(
            frame,
            existing_rows=existing_rows,
        )
        if not rows:
            raise ValueError("No cluster assignments are available. Run `cluster` first.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "cluster_id",
                    "cluster_size",
                    "outlier_count",
                    "representative_sample_ids",
                    "representative_paths",
                    "current_majority_label",
                    "label",
                    "status",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return {
            "review_path": str(destination),
            "cluster_count": len(rows),
            "stale_reset_count": stale_reset_count,
            "manifest_path": str(self.config.manifest_path),
        }

    def promote_cluster_labels(self, review_file: Path) -> dict[str, Any]:
        frame = self.load_manifest()
        current_rows = self._build_cluster_review_rows(frame)[0]
        current_rows_by_cluster = {
            int(row["cluster_id"]): row for row in current_rows if row.get("cluster_id")
        }
        assignments, stale_cluster_ids = self._load_cluster_review_assignments(
            review_file,
            current_rows_by_cluster=current_rows_by_cluster,
        )
        if not assignments:
            if stale_cluster_ids:
                stale_preview = ", ".join(str(cluster_id) for cluster_id in stale_cluster_ids[:5])
                raise ValueError(
                    "No current cluster review rows found in "
                    f"{review_file}. Stale cluster reviews: {stale_preview}. "
                    "Re-export or reload cluster review and try again."
                )
            raise ValueError(f"No cluster review rows found in {review_file}")

        updated_records: list[dict[str, Any]] = []
        promoted_rows = 0
        excluded_rows = 0
        touched_clusters: set[int] = set()
        for row in frame.iter_rows(named=True):
            record = dict(row)
            cluster_id = row.get("cluster_id")
            if cluster_id is None:
                updated_records.append(record)
                continue
            cluster_assignment = assignments.get(int(cluster_id))
            if cluster_assignment is None or bool(record.get("is_cluster_outlier")):
                updated_records.append(record)
                continue
            touched_clusters.add(int(cluster_id))
            status = str(cluster_assignment["annotation_status"])
            normalized_label = str(cluster_assignment["label"])
            if status == "excluded":
                excluded_rows += 1
            elif status == "labeled":
                promoted_rows += 1
            updated_records.append(
                self._apply_label_update(
                    record,
                    label=normalized_label,
                    status=status,
                    label_source="cluster_review" if status == "labeled" else "unknown",
                    review_status=(
                        "cluster_accepted"
                        if status == "labeled"
                        else "cluster_excluded"
                        if status == "excluded"
                        else "cluster_pending"
                    ),
                    curated_label=None,
                    suggested_label=(
                        normalized_label if status == "labeled" else record.get("suggested_label")
                    ),
                    suggested_label_source=(
                        "cluster_review"
                        if status == "labeled"
                        else record.get("suggested_label_source")
                    ),
                )
            )

        updated_frame = self.assign_splits(self._frame_from_records(updated_records))
        self.write_manifest(updated_frame)
        report_summary = self.generate_reports(updated_frame)
        return {
            "review_path": str(review_file),
            "manifest_path": str(self.config.manifest_path),
            "clusters_applied": len(touched_clusters),
            "promoted_rows": promoted_rows,
            "excluded_rows": excluded_rows,
            "stale_cluster_count": len(stale_cluster_ids),
            "training_ready_files": int(report_summary["training_ready_files"]),
        }

    def export_labeling_template(
        self,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        frame = self.load_manifest() if self.config.manifest_path.exists() else self.run_scan()
        destination = output_path or self.config.labeling_template_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, str]] = []
        for row in frame.sort("relative_path").iter_rows(named=True):
            label_source = str(row.get("label_source") or "unknown")
            annotation_status = str(row.get("annotation_status") or "unlabeled")
            current_label = str(row.get("label") or self.config.unknown_label)
            curated_label = (
                current_label
                if label_source == "curated" and current_label != self.config.unknown_label
                else ""
            )
            curated_status = "labeled" if curated_label else "unlabeled"
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "relative_path": str(row["relative_path"]),
                    "file_name": str(row["file_name"]),
                    "suggested_label": str(row.get("suggested_label") or ""),
                    "current_label": current_label,
                    "current_label_source": label_source,
                    "current_annotation_status": annotation_status,
                    "cluster_id": (
                        ""
                        if row.get("cluster_id") is None
                        else str(int(row["cluster_id"]))
                    ),
                    "review_status": str(row.get("review_status") or ""),
                    "label": curated_label,
                    "status": curated_status,
                    "notes": "",
                }
            )

        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "relative_path",
                    "file_name",
                    "suggested_label",
                    "current_label",
                    "current_label_source",
                    "current_annotation_status",
                    "cluster_id",
                    "review_status",
                    "label",
                    "status",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return {
            "template_path": str(destination),
            "total_rows": len(rows),
            "manifest_path": str(self.config.manifest_path),
        }

    def import_labels(self, labels_file: Path) -> dict[str, Any]:
        frame = self.load_manifest()
        assignments = self._load_label_assignments(labels_file)
        if not assignments:
            raise ValueError(f"No labels found in {labels_file}")

        manifest_ids = set(frame.get_column("sample_id").to_list())
        assignment_ids = set(assignments)
        missing_ids = sorted(assignment_ids - manifest_ids)
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                "Label file contains unknown sample_id values not present in the manifest: "
                f"{preview}"
            )

        updated_records: list[dict[str, Any]] = []
        updated_count = 0
        labeled_count = 0
        excluded_count = 0
        unlabeled_count = 0
        for row in frame.iter_rows(named=True):
            record = dict(row)
            assignment = assignments.get(str(record["sample_id"]))
            if assignment is not None:
                record = self._apply_label_update(
                    record,
                    label=str(assignment["label"]),
                    status=str(assignment["annotation_status"]),
                    label_source=str(assignment["label_source"]),
                    review_status=(
                        "curated"
                        if str(assignment["annotation_status"]) == "labeled"
                        else "excluded"
                        if str(assignment["annotation_status"]) == "excluded"
                        else "pending_review"
                    ),
                    curated_label=(
                        str(assignment["label"])
                        if str(assignment["annotation_status"]) == "labeled"
                        else None
                    ),
                    suggested_label=record.get("suggested_label"),
                    suggested_label_source=record.get("suggested_label_source"),
                    annotated_at=assignment.get("annotated_at"),
                )
                updated_count += 1
                status = str(record["annotation_status"])
                if status == "labeled":
                    labeled_count += 1
                elif status == "excluded":
                    excluded_count += 1
                else:
                    unlabeled_count += 1
            updated_records.append(record)

        updated_frame = self.assign_splits(self._frame_from_records(updated_records))
        self.write_manifest(updated_frame)
        report_summary = self.generate_reports(updated_frame)
        validation_summary = self.validate_labels(updated_frame)
        return {
            "labels_file": str(labels_file),
            "manifest_path": str(self.config.manifest_path),
            "updated_rows": updated_count,
            "labeled_rows": labeled_count,
            "excluded_rows": excluded_count,
            "unlabeled_rows": unlabeled_count,
            "training_ready_files": int(report_summary["training_ready_files"]),
            "validation": validation_summary,
        }

    def validate_labels(self, frame: pl.DataFrame | None = None) -> dict[str, Any]:
        source_frame = frame if frame is not None else self.load_manifest()
        training_ready_frame = self._training_ready_frame(source_frame)
        curated_frame = source_frame.filter(pl.col("label_source") == "curated")
        source_counts = self._count_mapping(source_frame, "label_source")
        status_counts = self._count_mapping(source_frame, "annotation_status")
        review_status_counts = self._count_mapping(source_frame, "review_status")
        class_names = self.class_names(source_frame)
        warnings: list[str] = []
        if len(class_names) < 2:
            warnings.append("Training requires at least 2 classes in the current manifest.")
        if training_ready_frame.height == 0:
            warnings.append("No training-ready samples remain after validation filters.")
        if (
            not self._has_accepted_labels(source_frame)
            and int(source_counts.get("filename", 0)) > 0
        ):
            warnings.append(
                "The manifest still relies entirely on filename-derived labels."
            )
        if (
            self._has_accepted_labels(source_frame)
            and int(source_counts.get("filename", 0)) > 0
        ):
            warnings.append(
                "Filename-derived labels are now treated as weak hints only because "
                "accepted labels exist in the manifest."
            )
        if int(status_counts.get("unlabeled", 0)) > 0:
            warnings.append("Some samples are still unlabeled and will be excluded from training.")

        summary = {
            "manifest_path": str(self.config.manifest_path),
            "num_classes": len(class_names),
            "class_names": class_names,
            "total_files": source_frame.height,
            "training_ready_files": training_ready_frame.height,
            "curated_files": curated_frame.height,
            "effective_training_mode": self._effective_training_mode(source_frame),
            "label_source_counts": source_counts,
            "annotation_status_counts": status_counts,
            "review_status_counts": review_status_counts,
            "train_label_counts": self.train_label_counts(source_frame),
            "warnings": warnings,
        }
        return summary

    def run_all(self) -> pl.DataFrame:
        frame = self.scan()
        split_frame = self.assign_splits(frame)
        self.write_manifest(split_frame)
        self.generate_reports(split_frame)
        return split_frame

    def load_manifest(self) -> pl.DataFrame:
        if not self.config.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {self.config.manifest_path}. Run the `scan` command first."
            )
        return self._ensure_manifest_columns(pl.read_parquet(self.config.manifest_path))

    def write_manifest(self, frame: pl.DataFrame) -> None:
        self.config.manifest_dir.mkdir(parents=True, exist_ok=True)
        normalized_frame = self._ensure_manifest_columns(frame)
        normalized_frame.sort("relative_path").write_parquet(self.config.manifest_path)
        normalized_frame.sort("relative_path").write_csv(self.config.manifest_csv_path)

    def class_names(self, frame: pl.DataFrame) -> list[str]:
        return (
            self._training_ready_frame(frame)
            .get_column("label")
            .unique()
            .sort()
            .to_list()
        )

    def train_label_counts(self, frame: pl.DataFrame) -> dict[str, int]:
        counts = {label: 0 for label in self.class_names(frame)}
        for row in (
            self._training_ready_frame(frame)
            .filter(pl.col("split") == "train")
            .group_by("label")
            .len(name="count")
            .iter_rows(named=True)
        ):
            counts[str(row["label"])] = int(row["count"])
        return counts

    def _ensure_manifest_columns(self, frame: pl.DataFrame) -> pl.DataFrame:
        result = frame
        if "raw_label" not in result.columns:
            result = result.with_columns(pl.col("label").alias("raw_label"))
        if "curated_label" not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.String).alias("curated_label"))
        if "suggested_label" not in result.columns:
            result = result.with_columns(
                pl.when(pl.col("label") != self.config.unknown_label)
                .then(pl.col("label"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("suggested_label")
            )
        if "suggested_label_source" not in result.columns:
            result = result.with_columns(
                pl.when(pl.col("label") != self.config.unknown_label)
                .then(pl.lit("filename"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("suggested_label_source")
            )
        if "label_source" not in result.columns:
            result = result.with_columns(
                pl.when(pl.col("label") != self.config.unknown_label)
                .then(pl.lit("filename"))
                .otherwise(pl.lit("unknown"))
                .alias("label_source")
            )
        if "annotation_status" not in result.columns:
            result = result.with_columns(
                pl.when(pl.col("label") != self.config.unknown_label)
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
        if "cluster_id" not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.Int64).alias("cluster_id"))
        if "cluster_distance" not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias("cluster_distance"))
        if "cluster_size" not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.Int64).alias("cluster_size"))
        if "is_cluster_outlier" not in result.columns:
            result = result.with_columns(
                pl.lit(False, dtype=pl.Boolean).alias("is_cluster_outlier")
            )

        return result.select(
            [
                (
                    pl.col(name).cast(data_type, strict=False)
                    if name in result.columns
                    else pl.lit(None, dtype=data_type)
                ).alias(name)
                for name, data_type in MANIFEST_SCHEMA.items()
            ]
        )

    def _iter_image_paths(self) -> list[Path]:
        if not self.config.raw_dataset_dir.exists():
            raise FileNotFoundError(
                f"Raw dataset directory does not exist: {self.config.raw_dataset_dir}"
            )
        allowed_extensions = {extension.lower() for extension in self.config.allowed_extensions}
        return sorted(
            (
                path
                for path in self.config.raw_dataset_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in allowed_extensions
            ),
            key=lambda path: path.relative_to(self.config.raw_dataset_dir).as_posix().lower(),
        )

    def _inspect_image(self, image_path: Path) -> dict[str, Any]:
        relative_path = image_path.relative_to(self.config.raw_dataset_dir).as_posix()
        raw_label, label = self._infer_label(image_path.stem)
        label_source = "filename" if label != self.config.unknown_label else "unknown"
        annotation_status = "inferred" if label != self.config.unknown_label else "unlabeled"
        width: int | None = None
        height: int | None = None
        channels: int | None = None
        decode_ok = False
        decode_error: str | None = None

        try:
            encoded = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if image is None:
                decode_error = "decode_failed"
            else:
                decode_ok = True
                if image.ndim == 2:
                    height, width = image.shape
                    channels = 1
                else:
                    height, width = image.shape[:2]
                    channels = int(image.shape[2])
        except Exception as error:  # pragma: no cover - defensive guard for OS/image issues
            decode_error = f"decode_failed:{type(error).__name__}"

        return {
            "sample_id": self._build_sample_id(relative_path, image_path.stem),
            "image_path": str(image_path),
            "relative_path": relative_path,
            "file_name": image_path.name,
            "extension": image_path.suffix.lower(),
            "file_size": image_path.stat().st_size,
            "width": width,
            "height": height,
            "channels": channels,
            "decode_ok": decode_ok,
            "decode_error": decode_error,
            "raw_label": raw_label,
            "curated_label": None,
            "suggested_label": (label if label != self.config.unknown_label else None),
            "suggested_label_source": ("filename" if label != self.config.unknown_label else None),
            "label": label,
            "label_source": label_source,
            "annotation_status": annotation_status,
            "annotated_at": None,
            "pseudo_label_score": None,
            "pseudo_label_margin": None,
            "review_status": "unreviewed",
            "cluster_id": None,
            "cluster_distance": None,
            "cluster_size": None,
            "is_cluster_outlier": False,
            "content_hash": self._hash_file(image_path),
            "is_duplicate": False,
            "duplicate_of": None,
            "is_too_small": False,
            "is_valid": False,
            "quarantine_reason": None,
            "split": None,
        }

    def _mark_duplicates(self, records: list[dict[str, Any]]) -> None:
        canonical_by_hash: dict[str, str] = {}
        for record in records:
            content_hash = str(record["content_hash"])
            canonical_sample_id = canonical_by_hash.get(content_hash)
            if canonical_sample_id is None:
                canonical_by_hash[content_hash] = str(record["sample_id"])
                continue
            record["is_duplicate"] = True
            record["duplicate_of"] = canonical_sample_id

    def _apply_quality_flags(self, record: dict[str, Any]) -> None:
        reason: str | None = None
        is_too_small = False

        if not bool(record["decode_ok"]):
            reason = "decode_failed"
        else:
            width = int(record["width"])
            height = int(record["height"])
            if width < self.config.min_width or height < self.config.min_height:
                reason = "too_small"
                is_too_small = True
            elif bool(record["is_duplicate"]):
                reason = "duplicate"

        record["is_too_small"] = is_too_small
        record["quarantine_reason"] = reason
        record["is_valid"] = reason is None

    def _frame_from_records(self, records: list[dict[str, Any]]) -> pl.DataFrame:
        if not records:
            return pl.DataFrame(
                {
                    name: pl.Series(name, [], dtype=data_type)
                    for name, data_type in MANIFEST_SCHEMA.items()
                }
            )

        frame = pl.DataFrame(
            records,
            schema=MANIFEST_SCHEMA,
            strict=False,
            infer_schema_length=None,
        )
        return frame.select(
            [
                pl.col(name).cast(data_type, strict=False).alias(name)
                for name, data_type in MANIFEST_SCHEMA.items()
            ]
        )

    def _infer_label(self, stem: str) -> tuple[str, str]:
        if not self.config.infer_labels_from_filename:
            return (self.config.unknown_label, self.config.unknown_label)

        match = LABEL_PATTERN.match(stem)
        raw_label = (match.group(1) if match else stem).strip()
        normalized_label = self._normalize_label(raw_label)
        return (
            raw_label or self.config.unknown_label,
            normalized_label or self.config.unknown_label,
        )

    def _normalize_label(self, label: str) -> str:
        normalized = re.sub(r"[^0-9a-zA-Z]+", "_", label.casefold()).strip("_")
        return normalized

    def _build_sample_id(self, relative_path: str, stem: str) -> str:
        safe_stem = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in stem.lower()
        ).strip("-")
        digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
        return f"{safe_stem or 'sample'}-{digest}"

    def _hash_file(self, image_path: Path) -> str:
        digest = hashlib.new(self.config.hash_algorithm)
        with image_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _effective_training_mode(self, frame: pl.DataFrame) -> str:
        return "accepted_labels_only" if self._has_accepted_labels(frame) else "weak_inferred"

    def _has_accepted_labels(self, frame: pl.DataFrame) -> bool:
        normalized_frame = self._ensure_manifest_columns(frame)
        return (
            normalized_frame.filter(
                pl.col("is_valid")
                & (pl.col("label") != self.config.unknown_label)
                & pl.col("label_source").is_in(sorted(ACCEPTED_LABEL_SOURCES))
                & pl.col("annotation_status").is_in(sorted(ACCEPTED_ANNOTATION_STATUSES))
            ).height
            > 0
        )

    def _discovery_ready_records(self, frame: pl.DataFrame) -> list[dict[str, Any]]:
        return list(
            frame.filter(pl.col("is_valid") & pl.col("decode_ok"))
            .sort("relative_path")
            .iter_rows(named=True)
        )

    def _apply_cluster_updates(
        self,
        frame: pl.DataFrame,
        updates: dict[str, dict[str, int | float | bool]],
    ) -> pl.DataFrame:
        records: list[dict[str, Any]] = []
        for row in frame.iter_rows(named=True):
            record = dict(row)
            update = updates.get(str(record["sample_id"]))
            if update is not None:
                record.update(update)
            else:
                record["cluster_id"] = None
                record["cluster_distance"] = None
                record["cluster_size"] = None
                record["is_cluster_outlier"] = False
            records.append(record)
        return self.assign_splits(self._frame_from_records(records))

    def _build_cluster_review_rows(
        self,
        frame: pl.DataFrame,
        *,
        existing_rows: dict[int, dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, str]], int]:
        rows: list[dict[str, str]] = []
        stale_reset_count = 0
        grouped_rows = self._cluster_groups(frame)

        for cluster_id in sorted(grouped_rows):
            members = grouped_rows[cluster_id]
            representative_members = members[: min(5, len(members))]
            label_counts: dict[str, int] = {}
            for member in members:
                label = str(member.get("label") or self.config.unknown_label)
                if label == self.config.unknown_label:
                    continue
                label_counts[label] = label_counts.get(label, 0) + 1
            majority_label = (
                max(label_counts.items(), key=lambda item: (item[1], item[0]))[0]
                if label_counts
                else ""
            )
            row = {
                "cluster_id": str(cluster_id),
                "cluster_size": str(len(members)),
                "outlier_count": str(
                    sum(1 for member in members if member["is_cluster_outlier"])
                ),
                "representative_sample_ids": "|".join(
                    str(member["sample_id"]) for member in representative_members
                ),
                "representative_paths": "|".join(
                    str(member["relative_path"]) for member in representative_members
                ),
                "current_majority_label": majority_label,
                "label": "",
                "status": "unlabeled",
                "notes": "",
            }
            existing_row = (existing_rows or {}).get(cluster_id)
            if existing_row is not None:
                if self._cluster_review_row_is_current(existing_row, row):
                    label_text = str(existing_row.get("label", "") or "").strip()
                    status = self._normalize_annotation_status(
                        existing_row.get("status"),
                        label_text=label_text,
                    )
                    row["status"] = status
                    row["label"] = self._normalize_label(label_text) if status == "labeled" else ""
                    row["notes"] = str(existing_row.get("notes", "") or "")
                else:
                    stale_reset_count += 1
            rows.append(row)
        return rows, stale_reset_count

    def _cluster_groups(self, frame: pl.DataFrame) -> dict[int, list[dict[str, Any]]]:
        clustered_rows = [
            dict(row)
            for row in frame.filter(pl.col("cluster_id").is_not_null())
            .sort(["cluster_id", "cluster_distance", "relative_path"], nulls_last=True)
            .iter_rows(named=True)
        ]
        grouped_rows: dict[int, list[dict[str, Any]]] = {}
        for row in clustered_rows:
            grouped_rows.setdefault(int(row["cluster_id"]), []).append(row)
        return grouped_rows

    def _build_cluster_preview_summary(self, frame: pl.DataFrame) -> dict[str, Any]:
        grouped_rows = self._cluster_groups(frame)
        ordered_cluster_ids = sorted(
            grouped_rows,
            key=lambda cluster_id: (-len(grouped_rows[cluster_id]), cluster_id),
        )
        previews: list[dict[str, Any]] = []

        for cluster_id in ordered_cluster_ids[:CLUSTER_PREVIEW_CLUSTER_LIMIT]:
            members = grouped_rows[cluster_id]
            label_counts: dict[str, int] = {}
            review_status_counts: dict[str, int] = {}
            representatives: list[dict[str, Any]] = []

            for member in members:
                label = str(member.get("label") or self.config.unknown_label)
                if label != self.config.unknown_label:
                    label_counts[label] = label_counts.get(label, 0) + 1
                review_status = str(member.get("review_status") or "unreviewed")
                review_status_counts[review_status] = review_status_counts.get(review_status, 0) + 1

            for member in members[:CLUSTER_PREVIEW_MEMBER_LIMIT]:
                representatives.append(
                    {
                        "sample_id": str(member["sample_id"]),
                        "relative_path": str(member["relative_path"]),
                        "label": str(member.get("label") or self.config.unknown_label),
                        "label_source": str(member.get("label_source") or "unknown"),
                        "annotation_status": str(
                            member.get("annotation_status") or "unlabeled"
                        ),
                        "review_status": str(member.get("review_status") or "unreviewed"),
                        "cluster_distance": (
                            None
                            if member.get("cluster_distance") is None
                            else float(member["cluster_distance"])
                        ),
                    }
                )

            previews.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "outlier_count": sum(
                        1 for member in members if bool(member.get("is_cluster_outlier"))
                    ),
                    "current_majority_label": (
                        max(label_counts.items(), key=lambda item: (item[1], item[0]))[0]
                        if label_counts
                        else None
                    ),
                    "review_status_counts": review_status_counts,
                    "representatives": representatives,
                }
            )

        return {
            "cluster_preview_total": len(grouped_rows),
            "cluster_preview_truncated": len(grouped_rows) > len(previews),
            "cluster_previews": previews,
        }

    def _apply_label_update(
        self,
        record: dict[str, Any],
        *,
        label: str,
        status: str,
        label_source: str,
        review_status: str,
        curated_label: str | None,
        suggested_label: Any,
        suggested_label_source: Any,
        annotated_at: Any | None = None,
        pseudo_label_score: float | None = None,
        pseudo_label_margin: float | None = None,
    ) -> dict[str, Any]:
        next_record = dict(record)
        accepted = status in {"labeled", "pseudo_labeled"} and label != self.config.unknown_label
        next_record["label"] = label if accepted else self.config.unknown_label
        next_record["label_source"] = label_source if accepted else "unknown"
        next_record["annotation_status"] = status
        next_record["annotated_at"] = (
            annotated_at
            if annotated_at is not None
            else (
                datetime.now(UTC).isoformat(timespec="seconds")
                if status in {"labeled", "excluded", "pseudo_labeled"}
                else None
            )
        )
        next_record["curated_label"] = curated_label
        next_record["suggested_label"] = suggested_label
        next_record["suggested_label_source"] = suggested_label_source
        next_record["pseudo_label_score"] = pseudo_label_score
        next_record["pseudo_label_margin"] = pseudo_label_margin
        next_record["review_status"] = review_status
        return next_record

    def _resolve_split(
        self,
        row: dict[str, Any],
        split_map: dict[str, str],
        *,
        has_accepted_labels: bool,
    ) -> str:
        if not bool(row["is_valid"]):
            return "excluded"
        if str(row["label"]) == self.config.unknown_label:
            return "excluded"
        if str(row["annotation_status"]) not in TRAINABLE_ANNOTATION_STATUSES:
            return "excluded"
        if has_accepted_labels and str(row.get("label_source")) not in ACCEPTED_LABEL_SOURCES:
            return "excluded"
        return split_map.get(str(row["sample_id"]), "excluded")

    def _count_mapping(self, frame: pl.DataFrame, column_name: str) -> dict[str, int]:
        if frame.is_empty():
            return {}
        rows = (
            frame.group_by(column_name)
            .len(name="count")
            .sort(column_name, nulls_last=True)
            .iter_rows(named=True)
        )
        return {
            ("null" if row[column_name] is None else str(row[column_name])): int(row["count"])
            for row in rows
        }

    def _dimension_stats(
        self,
        frame: pl.DataFrame,
        column_name: str,
    ) -> dict[str, float | int | None]:
        valid_frame = frame.filter(pl.col("decode_ok") & pl.col(column_name).is_not_null())
        if valid_frame.is_empty():
            return {"min": None, "median": None, "max": None}

        stats = valid_frame.select(
            pl.col(column_name).min().alias("min"),
            pl.col(column_name).median().alias("median"),
            pl.col(column_name).max().alias("max"),
        ).row(0, named=True)
        return {
            "min": int(stats["min"]) if stats["min"] is not None else None,
            "median": float(stats["median"]) if stats["median"] is not None else None,
            "max": int(stats["max"]) if stats["max"] is not None else None,
        }

    def _write_count_table(self, frame: pl.DataFrame, column_name: str, output_path: Path) -> None:
        table = frame.group_by(column_name).len(name="count").sort(column_name, nulls_last=True)
        table.write_csv(output_path)

    def _training_ready_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        normalized_frame = self._ensure_manifest_columns(frame)
        allowed_statuses = (
            sorted(ACCEPTED_ANNOTATION_STATUSES)
            if self._has_accepted_labels(normalized_frame)
            else sorted(TRAINABLE_ANNOTATION_STATUSES)
        )
        base_filter = (
            pl.col("is_valid")
            & (pl.col("label") != self.config.unknown_label)
            & pl.col("annotation_status").is_in(allowed_statuses)
        )
        if self._has_accepted_labels(normalized_frame):
            return normalized_frame.filter(
                base_filter & pl.col("label_source").is_in(sorted(ACCEPTED_LABEL_SOURCES))
            )
        return normalized_frame.filter(base_filter)

    def _load_label_assignments(self, labels_file: Path) -> dict[str, dict[str, Any]]:
        suffix = labels_file.suffix.lower()
        if suffix == ".csv":
            rows = self._read_label_csv(labels_file)
        elif suffix in {".jsonl", ".ndjson"}:
            rows = self._read_label_jsonl(labels_file)
        elif suffix == ".json":
            rows = self._read_label_json(labels_file)
        else:
            raise ValueError(
                f"Unsupported labels file format: {labels_file}. Use .csv, .json, or .jsonl"
            )

        assignments: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError("Each label record must include a non-empty sample_id.")
            label_text = str(row.get("label", "") or "").strip()
            raw_status = row.get("status", row.get("annotation_status"))
            status = self._normalize_annotation_status(raw_status, label_text=label_text)
            normalized_label = (
                self._normalize_label(label_text) if label_text else self.config.unknown_label
            )
            if status == "labeled" and normalized_label == self.config.unknown_label:
                raise ValueError(
                    f"Label record for {sample_id} is marked labeled but has no usable label."
                )
            if status != "labeled":
                normalized_label = self.config.unknown_label

            assignments[sample_id] = {
                "label": normalized_label,
                "label_source": "curated" if status == "labeled" else "unknown",
                "annotation_status": status,
                "annotated_at": (
                    str(row.get("annotated_at")).strip()
                    if row.get("annotated_at") not in {None, ""}
                    else (
                        datetime.now(UTC).isoformat(timespec="seconds")
                        if status in {"labeled", "excluded"}
                        else None
                    )
                ),
            }
        return assignments

    def _load_cluster_review_assignments(
        self,
        review_file: Path,
        *,
        current_rows_by_cluster: dict[int, dict[str, str]] | None = None,
    ) -> tuple[dict[int, dict[str, Any]], list[int]]:
        rows = self._read_label_csv(review_file)
        assignments: dict[int, dict[str, Any]] = {}
        stale_cluster_ids: list[int] = []
        for row in rows:
            raw_cluster_id = str(row.get("cluster_id", "")).strip()
            if not raw_cluster_id:
                continue
            cluster_id = int(raw_cluster_id)
            current_row = (current_rows_by_cluster or {}).get(cluster_id)
            if current_rows_by_cluster is not None:
                if current_row is None:
                    stale_cluster_ids.append(cluster_id)
                    continue
                if not self._cluster_review_row_is_current(row, current_row):
                    stale_cluster_ids.append(cluster_id)
                    continue
            label_text = str(row.get("label", "") or "").strip()
            status = self._normalize_annotation_status(
                row.get("status"),
                label_text=label_text,
            )
            if status == "unlabeled" and not label_text:
                continue
            normalized_label = (
                self._normalize_label(label_text) if label_text else self.config.unknown_label
            )
            if status == "labeled" and normalized_label == self.config.unknown_label:
                raise ValueError(
                    "Cluster review row for cluster "
                    f"{cluster_id} is labeled but has no usable label."
                )
            assignments[cluster_id] = {
                "label": normalized_label,
                "annotation_status": status,
            }
        stale_cluster_ids = sorted(set(stale_cluster_ids))
        return assignments, stale_cluster_ids

    def _load_existing_cluster_review_rows(self, review_file: Path) -> dict[int, dict[str, Any]]:
        rows = self._read_label_csv(review_file)
        existing_rows: dict[int, dict[str, Any]] = {}
        for row in rows:
            raw_cluster_id = str(row.get("cluster_id", "")).strip()
            if not raw_cluster_id:
                continue
            existing_rows[int(raw_cluster_id)] = dict(row)
        return existing_rows

    def _cluster_review_row_is_current(
        self,
        review_row: dict[str, Any],
        current_row: dict[str, Any],
    ) -> bool:
        return self._cluster_review_fingerprint(review_row) == self._cluster_review_fingerprint(
            current_row
        )

    def _cluster_review_fingerprint(self, row: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("cluster_id", "")).strip(),
            str(row.get("cluster_size", "")).strip(),
            str(row.get("outlier_count", "")).strip(),
            str(row.get("representative_sample_ids", "")).strip(),
        )

    def _normalize_annotation_status(self, value: Any, *, label_text: str) -> str:
        if value is None or str(value).strip() == "":
            return "labeled" if label_text else "unlabeled"

        normalized = str(value).strip().casefold()
        if normalized in {"pseudo_labeled", "pseudo-labelled", "pseudo"}:
            return "pseudo_labeled"
        if normalized in {"labeled", "labelled", "curated"}:
            return "labeled"
        if normalized in {"unlabeled", "unlabelled", "pending", "unknown"}:
            return "unlabeled"
        if normalized in {"excluded", "skip", "ignored"}:
            return "excluded"
        raise ValueError(f"Unsupported annotation status: {value}")

    def _read_label_csv(self, labels_file: Path) -> list[dict[str, Any]]:
        with labels_file.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _read_label_json(self, labels_file: Path) -> list[dict[str, Any]]:
        payload = json.loads(labels_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON label files must contain a list of records.")
        return [dict(row) for row in payload]

    def _read_label_jsonl(self, labels_file: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in labels_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError("Each JSONL line must be an object.")
            rows.append(dict(payload))
        return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset pipeline for localagent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--raw-dir", type=Path, default=None)
    common_parent.add_argument("--manifest-dir", type=Path, default=None)
    common_parent.add_argument("--report-dir", type=Path, default=None)
    common_parent.add_argument("--min-width", type=int, default=None)
    common_parent.add_argument("--min-height", type=int, default=None)
    common_parent.add_argument("--train-ratio", type=float, default=None)
    common_parent.add_argument("--val-ratio", type=float, default=None)
    common_parent.add_argument("--test-ratio", type=float, default=None)
    common_parent.add_argument("--seed", type=int, default=None)
    common_parent.add_argument("--num-clusters", type=int, default=None)
    common_parent.add_argument("--no-filename-labels", action="store_true")
    common_parent.add_argument("--no-progress", action="store_true")

    for command in ("scan", "split", "report", "run-all", "validate-labels", "embed", "cluster"):
        subparsers.add_parser(command, parents=[common_parent])

    export_parser = subparsers.add_parser("export-labeling-template", parents=[common_parent])
    export_parser.add_argument("--output", type=Path, default=None)

    cluster_review_export = subparsers.add_parser(
        "export-cluster-review",
        parents=[common_parent],
    )
    cluster_review_export.add_argument("--output", type=Path, default=None)

    import_parser = subparsers.add_parser("import-labels", parents=[common_parent])
    import_parser.add_argument("--labels-file", type=Path, required=True)

    promote_parser = subparsers.add_parser("promote-cluster-labels", parents=[common_parent])
    promote_parser.add_argument("--review-file", type=Path, required=True)

    return parser


def build_config(args: argparse.Namespace) -> DatasetPipelineConfig:
    defaults = DatasetPipelineConfig()
    return DatasetPipelineConfig(
        raw_dataset_dir=args.raw_dir or defaults.raw_dataset_dir,
        manifest_dir=args.manifest_dir or defaults.manifest_dir,
        report_dir=args.report_dir or defaults.report_dir,
        manifest_name=defaults.manifest_name,
        manifest_csv_name=defaults.manifest_csv_name,
        labeling_template_name=defaults.labeling_template_name,
        cluster_review_template_name=defaults.cluster_review_template_name,
        embeddings_name=defaults.embeddings_name,
        min_width=defaults.min_width if args.min_width is None else args.min_width,
        min_height=defaults.min_height if args.min_height is None else args.min_height,
        train_ratio=defaults.train_ratio if args.train_ratio is None else args.train_ratio,
        val_ratio=defaults.val_ratio if args.val_ratio is None else args.val_ratio,
        test_ratio=defaults.test_ratio if args.test_ratio is None else args.test_ratio,
        random_seed=defaults.random_seed if args.seed is None else args.seed,
        allowed_extensions=defaults.allowed_extensions,
        hash_algorithm=defaults.hash_algorithm,
        infer_labels_from_filename=(
            defaults.infer_labels_from_filename
            if not getattr(args, "no_filename_labels", False)
            else False
        ),
        unknown_label=defaults.unknown_label,
        show_progress=not args.no_progress,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pipeline = DatasetPipeline(build_config(args))

    if args.command == "scan":
        frame = pipeline.run_scan()
        print(f"Scanned {frame.height} files into {pipeline.config.manifest_path}")
        return 0

    if args.command == "split":
        frame = pipeline.run_split()
        print(f"Assigned splits for {frame.filter(pl.col('is_valid')).height} valid files")
        return 0

    if args.command == "report":
        summary = pipeline.run_report()
        print(
            f"Wrote dataset report to {pipeline.config.summary_path} "
            f"({summary['total_files']} files)"
        )
        return 0

    if args.command == "embed":
        summary = pipeline.embed_dataset()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "cluster":
        summary = pipeline.cluster_dataset(requested_clusters=args.num_clusters)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "export-labeling-template":
        summary = pipeline.export_labeling_template(output_path=args.output)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "export-cluster-review":
        summary = pipeline.export_cluster_review(output_path=args.output)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "import-labels":
        summary = pipeline.import_labels(args.labels_file)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "promote-cluster-labels":
        summary = pipeline.promote_cluster_labels(args.review_file)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "validate-labels":
        summary = pipeline.validate_labels()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    frame = pipeline.run_all()
    print(
        "Completed scan, split, and report for "
        f"{frame.height} files. Manifest: {pipeline.config.manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
