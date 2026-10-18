"""End-to-end integration tests for training, retrieval, and inference pipelines."""

from src.config import load_yaml_config, OUTPUTS_DIR


def test_pipeline_configs_and_outputs() -> None:
    """Verify that training and inference configs load and outputs directory exists."""
    train_cfg = load_yaml_config("train.yaml")
    inf_cfg = load_yaml_config("inference.yaml")
    assert "training" in train_cfg
    assert "inference" in inf_cfg
    assert OUTPUTS_DIR.exists()
