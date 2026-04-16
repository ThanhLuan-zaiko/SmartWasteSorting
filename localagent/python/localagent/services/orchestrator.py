from __future__ import annotations

from localagent.config import AgentPaths, RuntimeConfig, TrainingConfig
from localagent.inference import InferencePipeline
from localagent.training import WasteTrainer


class LocalWasteAgent:
    def __init__(
        self,
        paths: AgentPaths | None = None,
        training: TrainingConfig | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.paths = (paths or AgentPaths()).ensure_layout()
        self.training = training or TrainingConfig()
        self.runtime = runtime or RuntimeConfig()
        self.trainer = WasteTrainer(self.paths, self.training)
        self.inference = InferencePipeline(self.runtime)

    def prepare_workspace(self) -> AgentPaths:
        return self.paths.ensure_layout()

    def training_plan(self) -> dict[str, object]:
        return self.trainer.summarize_training_plan()

    def classify(self, sample_id: str):
        return self.inference.classify(sample_id)
