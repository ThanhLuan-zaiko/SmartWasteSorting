from __future__ import annotations

import json

from localagent.bridge import RustBackendBridge
from localagent.config import RuntimeConfig
from localagent.domain import ClassificationResult, Prediction


class InferencePipeline:
    def __init__(
        self,
        config: RuntimeConfig,
        bridge: RustBackendBridge | None = None,
    ) -> None:
        self.config = config
        self.bridge = bridge or RustBackendBridge(config)

    def classify(self, sample_id: str) -> ClassificationResult:
        backend = self.bridge.create_backend()
        if backend is None:
            return ClassificationResult(
                sample_id=sample_id,
                predictions=[Prediction(label="pending-model", score=0.0)],
                backend="python-fallback",
            )

        payload = json.loads(backend.classify_stub(sample_id))
        predictions = [
            Prediction(label=item["label"], score=float(item["score"]))
            for item in payload.get("predictions", [])
        ]
        return ClassificationResult(
            sample_id=payload.get("sample_id", sample_id),
            predictions=predictions,
            backend=payload.get("backend", "rust"),
        )
