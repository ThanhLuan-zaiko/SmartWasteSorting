from __future__ import annotations

from pathlib import Path

from localagent.config import AgentPaths, DatasetPipelineConfig, RuntimeConfig, TrainingConfig


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
    assert runtime.model_manifest_path.name == "model_manifest.json"
    assert runtime.artifact_dir.name == "artifacts"
    assert runtime.experiment_name == "baseline-waste-sorter"
    assert training.training_preset is None
    assert training.model_name == "mobilenet_v3_small"
    assert training.pretrained_backbone is True
    assert training.batch_size == 16
    assert training.image_size == 224
    assert training.resume_from_checkpoint is None
    assert training.cache_format == "png"
    assert training.onnx_output_path.name == "waste_classifier.onnx"
    assert training.model_manifest_output_path.name == "model_manifest.json"
    assert training.onnx_opset == 17
    assert training.verify_onnx is True
    assert training.class_bias_strategy == "none"
    assert training.enable_early_stopping is True
    assert training.manifest_path.name == "dataset_manifest.parquet"
    assert training.labels_output_path.name == "labels.json"


def test_dataset_pipeline_defaults_are_stable() -> None:
    config = DatasetPipelineConfig()

    assert config.raw_dataset_dir.name == "dataset"
    assert config.manifest_path.name == "dataset_manifest.parquet"
    assert config.report_dir.name == "reports"
    assert config.random_seed == 42
    assert config.label_summary_path.name == "label_summary.csv"
    assert config.labeling_template_path.name == "labeling_template.csv"
