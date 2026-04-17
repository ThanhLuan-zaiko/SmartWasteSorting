from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from localagent.config import RuntimeConfig

from ._extension import load_extension


@dataclass(slots=True)
class RustBackendBridge:
    config: RuntimeConfig

    def is_available(self) -> bool:
        return load_extension() is not None

    def create_backend(self) -> Any | None:
        extension = load_extension()
        if extension is None:
            return None
        return extension.RustBackend(
            model_path=str(self.config.model_path),
            labels_path=str(self.config.labels_path),
            device=self.config.device,
            score_threshold=float(self.config.score_threshold),
        )

    def ping(self) -> str | None:
        extension = load_extension()
        if extension is None:
            return None
        return extension.ping()
