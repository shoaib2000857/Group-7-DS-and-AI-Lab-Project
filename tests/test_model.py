"""Unit tests for model architectures and configuration loading."""

from src.config import load_yaml_config


def test_model_config_loads() -> None:
    """Verify that model.yaml loads with expected top-level keys."""
    cfg = load_yaml_config("model.yaml")
    assert "model" in cfg
    assert "parameters" in cfg
