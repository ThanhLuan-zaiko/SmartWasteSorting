from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from localagent.config import DatasetPipelineConfig
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
    "label": pl.String,
    "content_hash": pl.String,
    "is_duplicate": pl.Boolean,
    "duplicate_of": pl.String,
    "is_too_small": pl.Boolean,
    "is_valid": pl.Boolean,
    "quarantine_reason": pl.String,
    "split": pl.String,
}

LABEL_PATTERN = re.compile(r"(.+?)[_\- ]\d+$")


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
        eligible_rows = (
            frame.filter(pl.col("is_valid") & (pl.col("label") != self.config.unknown_label))
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

        frame_without_split = frame.drop("split")
        return (
            frame_without_split.with_columns(
                pl.struct(["sample_id", "label", "is_valid"])
                .map_elements(
                    lambda row: self._resolve_split(row, split_map),
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
        summary = {
            "dataset_root": str(self.config.raw_dataset_dir),
            "manifest_path": str(self.config.manifest_path),
            "total_files": frame.height,
            "valid_files": frame.filter(pl.col("is_valid")).height,
            "invalid_files": frame.filter(~pl.col("is_valid")).height,
            "decode_failed_files": frame.filter(~pl.col("decode_ok")).height,
            "too_small_files": frame.filter(pl.col("is_too_small")).height,
            "duplicate_files": frame.filter(pl.col("is_duplicate")).height,
            "training_ready_files": frame.filter(
                pl.col("is_valid") & (pl.col("label") != self.config.unknown_label)
            ).height,
            "split_counts": self._count_mapping(frame, "split"),
            "label_counts": self._count_mapping(frame, "label"),
            "extension_counts": self._count_mapping(frame, "extension"),
            "quarantine_counts": self._count_mapping(frame, "quarantine_reason"),
            "width_stats": self._dimension_stats(frame, "width"),
            "height_stats": self._dimension_stats(frame, "height"),
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
        return pl.read_parquet(self.config.manifest_path)

    def write_manifest(self, frame: pl.DataFrame) -> None:
        self.config.manifest_dir.mkdir(parents=True, exist_ok=True)
        frame.sort("relative_path").write_parquet(self.config.manifest_path)
        frame.sort("relative_path").write_csv(self.config.manifest_csv_path)

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
            "label": label,
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

    def _resolve_split(self, row: dict[str, Any], split_map: dict[str, str]) -> str:
        if not bool(row["is_valid"]):
            return "excluded"
        if str(row["label"]) == self.config.unknown_label:
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
    common_parent.add_argument("--no-progress", action="store_true")

    for command in ("scan", "split", "report", "run-all"):
        subparsers.add_parser(command, parents=[common_parent])

    return parser


def build_config(args: argparse.Namespace) -> DatasetPipelineConfig:
    defaults = DatasetPipelineConfig()
    return DatasetPipelineConfig(
        raw_dataset_dir=args.raw_dir or defaults.raw_dataset_dir,
        manifest_dir=args.manifest_dir or defaults.manifest_dir,
        report_dir=args.report_dir or defaults.report_dir,
        manifest_name=defaults.manifest_name,
        manifest_csv_name=defaults.manifest_csv_name,
        min_width=defaults.min_width if args.min_width is None else args.min_width,
        min_height=defaults.min_height if args.min_height is None else args.min_height,
        train_ratio=defaults.train_ratio if args.train_ratio is None else args.train_ratio,
        val_ratio=defaults.val_ratio if args.val_ratio is None else args.val_ratio,
        test_ratio=defaults.test_ratio if args.test_ratio is None else args.test_ratio,
        random_seed=defaults.random_seed if args.seed is None else args.seed,
        allowed_extensions=defaults.allowed_extensions,
        hash_algorithm=defaults.hash_algorithm,
        infer_labels_from_filename=defaults.infer_labels_from_filename,
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

    frame = pipeline.run_all()
    print(
        "Completed scan, split, and report for "
        f"{frame.height} files. Manifest: {pipeline.config.manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
