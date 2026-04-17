from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._extension import load_extension


@dataclass(slots=True)
class RustAccelerationBridge:
    def is_available(self) -> bool:
        return load_extension() is not None

    def prepare_image_cache(
        self,
        entries: list[tuple[str, str]],
        *,
        cache_dir: Path,
        failure_report_path: Path | None = None,
        image_size: int,
        cache_format: str = "png",
        force: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any] | None:
        extension = load_extension()
        if extension is None:
            return None

        payload = extension.prepare_image_cache(
            entries=entries,
            cache_dir=str(cache_dir),
            failure_report_path=(
                None if failure_report_path is None else str(failure_report_path)
            ),
            image_size=int(image_size),
            cache_format=str(cache_format),
            force=bool(force),
            show_progress=bool(show_progress),
        )
        return json.loads(payload)

    def compute_class_weight_map(
        self,
        train_labels: list[str],
        class_names: list[str],
    ) -> dict[str, float] | None:
        extension = load_extension()
        if extension is None:
            return None

        payload = extension.compute_class_weight_map_json(
            train_labels=list(train_labels),
            class_names=list(class_names),
        )
        return json.loads(payload)

    def build_classification_report(
        self,
        *,
        predictions: list[int],
        targets: list[int],
        class_names: list[str],
    ) -> dict[str, Any] | None:
        extension = load_extension()
        if extension is None:
            return None

        payload = extension.build_classification_report_json(
            predictions=list(predictions),
            targets=list(targets),
            class_names=list(class_names),
        )
        return json.loads(payload)
