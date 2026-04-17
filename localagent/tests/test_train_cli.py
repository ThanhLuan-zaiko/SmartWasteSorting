from __future__ import annotations

from localagent.config import TrainingConfig
from localagent.training.train import build_config, build_parser


def test_build_config_uses_default_experiment_name() -> None:
    parser = build_parser()

    args = parser.parse_args(["fit"])
    config = build_config(args)

    assert config.experiment_name == TrainingConfig().experiment_name


def test_build_config_accepts_custom_experiment_name() -> None:
    parser = build_parser()

    args = parser.parse_args(["fit", "--experiment-name", "baseline-waste-sorter-e15-cpu"])
    config = build_config(args)

    assert config.experiment_name == "baseline-waste-sorter-e15-cpu"


def test_build_config_accepts_cache_format_and_class_bias() -> None:
    parser = build_parser()

    args = parser.parse_args(["fit", "--cache-format", "raw", "--class-bias", "both"])
    config = build_config(args)

    assert config.cache_format == "raw"
    assert config.class_bias_strategy == "both"


def test_build_config_accepts_resume_checkpoint_path() -> None:
    parser = build_parser()

    args = parser.parse_args(["fit", "--resume-from", "artifacts/checkpoints/demo.last.pt"])
    config = build_config(args)

    assert str(config.resume_from_checkpoint).endswith("artifacts\\checkpoints\\demo.last.pt")


def test_build_config_accepts_cnn_model_name() -> None:
    parser = build_parser()

    args = parser.parse_args(["fit", "--model-name", "resnet18"])
    config = build_config(args)

    assert config.model_name == "resnet18"


def test_build_config_applies_cpu_fast_preset() -> None:
    parser = build_parser()

    args = parser.parse_args(["fit", "--training-preset", "cpu_fast"])
    config = build_config(args)

    assert config.training_preset == "cpu_fast"
    assert config.model_name == "mobilenet_v3_small"
    assert config.image_size == 160
    assert config.batch_size == 32
    assert config.cache_format == "raw"
    assert config.class_bias_strategy == "loss"


def test_explicit_flags_override_training_preset() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "fit",
            "--training-preset",
            "cpu_balanced",
            "--model-name",
            "mobilenet_v3_large",
            "--batch-size",
            "24",
        ]
    )
    config = build_config(args)

    assert config.training_preset == "cpu_balanced"
    assert config.model_name == "mobilenet_v3_large"
    assert config.batch_size == 24
    assert config.image_size == 224
    assert config.cache_format == "raw"
