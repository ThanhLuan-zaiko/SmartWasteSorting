from .config import AgentPaths, DatasetPipelineConfig, RuntimeConfig, TrainingConfig
from .data import DatasetIndex
from .services import LocalWasteAgent

__all__ = [
    "AgentPaths",
    "DatasetIndex",
    "DatasetPipeline",
    "DatasetPipelineConfig",
    "LocalWasteAgent",
    "RuntimeConfig",
    "TrainingConfig",
]


def __getattr__(name: str):
    if name == "DatasetPipeline":
        from .data.pipeline import DatasetPipeline

        return DatasetPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
