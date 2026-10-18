"""Unit tests for data loading and preprocessing pipelines."""

from src.config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR


def test_data_directories_exist() -> None:
    """Verify that the required data subdirectories exist."""
    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert SAMPLE_DATA_DIR.exists()
