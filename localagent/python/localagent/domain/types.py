from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WasteSample:
    sample_id: str
    image_path: Path
    label: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    label: str
    score: float


@dataclass(slots=True)
class ClassificationResult:
    sample_id: str
    predictions: list[Prediction]
    backend: str
