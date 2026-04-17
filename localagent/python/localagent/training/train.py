from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from localagent.config import AgentPaths, TrainingConfig
from localagent.training import WasteTrainer

TRAINING_PRESETS: dict[str, dict[str, object]] = {
    "cpu_fast": {
        "model_name": "mobilenet_v3_small",
        "image_size": 160,
        "batch_size": 32,
        "cache_format": "raw",
        "class_bias_strategy": "loss",
    },
    "cpu_balanced": {
        "model_name": "resnet18",
        "image_size": 224,
        "batch_size": 16,
        "cache_format": "raw",
        "class_bias_strategy": "loss",
    },
    "cpu_stronger": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 8,
        "cache_format": "raw",
        "class_bias_strategy": "loss",
    },
}


def _resolve_value(
    explicit_value,
    *,
    preset: dict[str, object],
    preset_key: str,
    default_value,
):
    if explicit_value is not None:
        return explicit_value
    if preset_key in preset:
        return preset[preset_key]
    return default_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training utilities for localagent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, default=None)
    common.add_argument("--training-preset", choices=tuple(TRAINING_PRESETS), default=None)
    common.add_argument("--experiment-name", type=str, default=None)
    common.add_argument("--model-name", type=str, default=None)
    common.add_argument("--no-pretrained", action="store_true")
    common.add_argument("--train-backbone", action="store_true")
    common.add_argument("--image-size", type=int, default=None)
    common.add_argument("--batch-size", type=int, default=None)
    common.add_argument("--epochs", type=int, default=None)
    common.add_argument("--num-workers", type=int, default=None)
    common.add_argument("--device", type=str, default=None)
    common.add_argument("--cache-dir", type=Path, default=None)
    common.add_argument("--resume-from", type=Path, default=None)
    common.add_argument("--checkpoint", type=Path, default=None)
    common.add_argument("--onnx-output", type=Path, default=None)
    common.add_argument("--cache-format", choices=("png", "raw"), default=None)
    common.add_argument("--no-rust-cache", action="store_true")
    common.add_argument("--force-cache", action="store_true")
    common.add_argument("--class-bias", choices=("none", "loss", "sampler", "both"), default=None)
    common.add_argument("--early-stopping-patience", type=int, default=None)
    common.add_argument("--early-stopping-min-delta", type=float, default=None)
    common.add_argument("--disable-early-stopping", action="store_true")
    common.add_argument("--onnx-opset", type=int, default=None)
    common.add_argument("--export-batch-size", type=int, default=None)
    common.add_argument("--skip-onnx-verify", action="store_true")
    common.add_argument("--no-progress", action="store_true")

    subparsers.add_parser("summary", parents=[common])
    subparsers.add_parser("export-labels", parents=[common])
    subparsers.add_parser("warm-cache", parents=[common])
    subparsers.add_parser("fit", parents=[common])
    subparsers.add_parser("evaluate", parents=[common])
    subparsers.add_parser("export-onnx", parents=[common])
    subparsers.add_parser("report", parents=[common])
    return parser


def build_config(args: argparse.Namespace) -> TrainingConfig:
    defaults = TrainingConfig()
    preset_name = args.training_preset
    preset = TRAINING_PRESETS.get(preset_name, {})
    return TrainingConfig(
        training_preset=preset_name,
        experiment_name=(
            defaults.experiment_name if args.experiment_name is None else args.experiment_name
        ),
        model_name=_resolve_value(
            args.model_name,
            preset=preset,
            preset_key="model_name",
            default_value=defaults.model_name,
        ),
        pretrained_backbone=not args.no_pretrained,
        freeze_backbone=not args.train_backbone,
        image_size=_resolve_value(
            args.image_size,
            preset=preset,
            preset_key="image_size",
            default_value=defaults.image_size,
        ),
        batch_size=_resolve_value(
            args.batch_size,
            preset=preset,
            preset_key="batch_size",
            default_value=defaults.batch_size,
        ),
        epochs=defaults.epochs if args.epochs is None else args.epochs,
        num_workers=defaults.num_workers if args.num_workers is None else args.num_workers,
        prefetch_factor=defaults.prefetch_factor,
        learning_rate=defaults.learning_rate,
        weight_decay=defaults.weight_decay,
        manifest_path=defaults.manifest_path if args.manifest is None else args.manifest,
        labels_output_path=defaults.labels_output_path,
        onnx_output_path=(
            defaults.onnx_output_path if args.onnx_output is None else args.onnx_output
        ),
        model_manifest_output_path=defaults.model_manifest_output_path,
        checkpoint_dir=defaults.checkpoint_dir,
        resume_from_checkpoint=(
            defaults.resume_from_checkpoint
            if args.resume_from is None
            else args.resume_from
        ),
        cache_dir=defaults.cache_dir if args.cache_dir is None else args.cache_dir,
        cache_format=_resolve_value(
            args.cache_format,
            preset=preset,
            preset_key="cache_format",
            default_value=defaults.cache_format,
        ),
        onnx_opset=defaults.onnx_opset if args.onnx_opset is None else args.onnx_opset,
        verify_onnx=not args.skip_onnx_verify,
        export_batch_size=(
            defaults.export_batch_size
            if args.export_batch_size is None
            else args.export_batch_size
        ),
        normalization_preset=defaults.normalization_preset,
        use_rust_image_cache=not args.no_rust_cache,
        force_rebuild_cache=bool(args.force_cache),
        class_bias_strategy=_resolve_value(
            args.class_bias,
            preset=preset,
            preset_key="class_bias_strategy",
            default_value=defaults.class_bias_strategy,
        ),
        early_stopping_patience=(
            defaults.early_stopping_patience
            if args.early_stopping_patience is None
            else args.early_stopping_patience
        ),
        early_stopping_min_delta=(
            defaults.early_stopping_min_delta
            if args.early_stopping_min_delta is None
            else args.early_stopping_min_delta
        ),
        enable_early_stopping=not args.disable_early_stopping,
        show_progress=not args.no_progress,
        device=defaults.device if args.device is None else args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    trainer = WasteTrainer(AgentPaths().ensure_layout(), build_config(args))

    if args.command == "summary":
        summary = trainer.summarize_training_plan()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "export-labels":
        labels_path = trainer.export_label_index()
        print(f"Exported labels to {labels_path}")
        return 0

    if args.command == "warm-cache":
        summary = trainer.warm_image_cache()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "evaluate":
        summary = trainer.evaluate(checkpoint_path=args.checkpoint)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "export-onnx":
        summary = trainer.export_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.onnx_output,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "report":
        summary = trainer.build_artifact_report()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    result = trainer.fit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
