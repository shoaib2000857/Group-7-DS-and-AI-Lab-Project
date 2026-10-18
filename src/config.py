"""Central configuration loader and path definitions for the project."""

from pathlib import Path
from typing import Any, Dict
import yaml

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Standard directory paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

CONFIGS_DIR = ROOT_DIR / "configs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DOCS_DIR = ROOT_DIR / "docs"


def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML configuration file from the configs/ directory.

    Args:
        config_name: Filename of the YAML config (e.g., 'model.yaml').

    Returns:
        Dictionary containing the parsed YAML configuration.
    """
    config_path = CONFIGS_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
