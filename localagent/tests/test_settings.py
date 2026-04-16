from __future__ import annotations

from pathlib import Path

from localagent.config import AgentPaths, RuntimeConfig, TrainingConfig


def test_agent_paths_default_to_localagent_root() -> None:
    paths = AgentPaths()

    assert paths.project_root.name == "localagent"
    assert paths.dataset_dir == paths.project_root / "datasets"
    assert paths.config_dir == paths.project_root / "configs"


def test_agent_paths_ensure_layout_creates_directories(tmp_path: Path) -> None:
    paths = AgentPaths(
        project_root=tmp_path,
        dataset_dir=tmp_path / "datasets",
        model_dir=tmp_path / "models",
        artifact_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        config_dir=tmp_path / "configs",
    )

    paths.ensure_layout()

    assert paths.dataset_dir.is_dir()
    assert paths.model_dir.is_dir()
    assert paths.artifact_dir.is_dir()
    assert paths.log_dir.is_dir()
    assert paths.config_dir.is_dir()


def test_runtime_and_training_defaults_are_stable() -> None:
    runtime = RuntimeConfig()
    training = TrainingConfig()

    assert runtime.server_port == 8080
    assert runtime.base_url == "http://127.0.0.1:8080"
    assert training.batch_size == 16
    assert training.image_size == 224
