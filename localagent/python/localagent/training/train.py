from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from localagent.config import AgentPaths, TrainingConfig
from localagent.training import WasteTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training utilities for localagent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, default=None)
    common.add_argument("--model-name", type=str, default=None)
    common.add_argument("--no-pretrained", action="store_true")
    common.add_argument("--train-backbone", action="store_true")
    common.add_argument("--image-size", type=int, default=None)
    common.add_argument("--batch-size", type=int, default=None)
    common.add_argument("--epochs", type=int, default=None)
    common.add_argument("--num-workers", type=int, default=None)
    common.add_argument("--device", type=str, default=None)
    common.add_argument("--cache-dir", type=Path, default=None)
    common.add_argument("--no-rust-cache", action="store_true")
    common.add_argument("--force-cache", action="store_true")
    common.add_argument("--early-stopping-patience", type=int, default=None)
    common.add_argument("--early-stopping-min-delta", type=float, default=None)
    common.add_argument("--disable-early-stopping", action="store_true")
    common.add_argument("--no-progress", action="store_true")

    subparsers.add_parser("summary", parents=[common])
    subparsers.add_parser("export-labels", parents=[common])
    subparsers.add_parser("warm-cache", parents=[common])
    subparsers.add_parser("fit", parents=[common])
    return parser


def build_config(args: argparse.Namespace) -> TrainingConfig:
    defaults = TrainingConfig()
    return TrainingConfig(
        experiment_name=defaults.experiment_name,
        model_name=defaults.model_name if args.model_name is None else args.model_name,
        pretrained_backbone=not args.no_pretrained,
        freeze_backbone=not args.train_backbone,
        image_size=defaults.image_size if args.image_size is None else args.image_size,
        batch_size=defaults.batch_size if args.batch_size is None else args.batch_size,
        epochs=defaults.epochs if args.epochs is None else args.epochs,
        num_workers=defaults.num_workers if args.num_workers is None else args.num_workers,
        prefetch_factor=defaults.prefetch_factor,
        learning_rate=defaults.learning_rate,
        weight_decay=defaults.weight_decay,
        manifest_path=defaults.manifest_path if args.manifest is None else args.manifest,
        labels_output_path=defaults.labels_output_path,
        checkpoint_dir=defaults.checkpoint_dir,
        cache_dir=defaults.cache_dir if args.cache_dir is None else args.cache_dir,
        use_rust_image_cache=not args.no_rust_cache,
        force_rebuild_cache=bool(args.force_cache),
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

    result = trainer.fit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
