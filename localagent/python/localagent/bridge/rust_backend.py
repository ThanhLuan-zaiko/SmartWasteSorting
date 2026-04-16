from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from localagent.config import RuntimeConfig


def _load_extension() -> Any | None:
    try:
        return import_module("localagent._rust")
    except ImportError:
        return None


@dataclass(slots=True)
class RustBackendBridge:
    config: RuntimeConfig

    def is_available(self) -> bool:
        return _load_extension() is not None

    def create_backend(self) -> Any | None:
        extension = _load_extension()
        if extension is None:
            return None
        return extension.RustBackend(
            model_path=str(self.config.model_path),
            labels_path=str(self.config.labels_path),
            device=self.config.device,
            score_threshold=float(self.config.score_threshold),
        )

    def ping(self) -> str | None:
        extension = _load_extension()
        if extension is None:
            return None
        return extension.ping()
