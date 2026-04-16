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
    common.add_argument("--batch-size", type=int, default=None)
    common.add_argument("--epochs", type=int, default=None)
    common.add_argument("--device", type=str, default=None)

    subparsers.add_parser("summary", parents=[common])
    subparsers.add_parser("export-labels", parents=[common])
    subparsers.add_parser("fit", parents=[common])
    return parser


def build_config(args: argparse.Namespace) -> TrainingConfig:
    defaults = TrainingConfig()
    return TrainingConfig(
        experiment_name=defaults.experiment_name,
        image_size=defaults.image_size,
        batch_size=defaults.batch_size if args.batch_size is None else args.batch_size,
        epochs=defaults.epochs if args.epochs is None else args.epochs,
        num_workers=defaults.num_workers,
        learning_rate=defaults.learning_rate,
        weight_decay=defaults.weight_decay,
        manifest_path=defaults.manifest_path if args.manifest is None else args.manifest,
        labels_output_path=defaults.labels_output_path,
        checkpoint_dir=defaults.checkpoint_dir,
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

    result = trainer.fit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
