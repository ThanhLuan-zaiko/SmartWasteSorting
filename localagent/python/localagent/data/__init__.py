from .dataset import DatasetIndex

__all__ = ["DatasetIndex", "DatasetPipeline", "main"]


def __getattr__(name: str):
    if name in {"DatasetPipeline", "main"}:
        from .pipeline import DatasetPipeline, main

        exports = {
            "DatasetPipeline": DatasetPipeline,
            "main": main,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
