import pytest
from src.football_tracking.config import load_config, validate_config
CONFIG_PATH = "configs/default.yaml"

def test_load_config_returns_dictionary():
    """Check that the YAML configuration is loaded as a dictionary."""
    config = load_config(CONFIG_PATH)
    assert isinstance(config, dict)

def test_load_config_contains_required_sections():
    """Check that all required configuration sections exist."""
    config = load_config(CONFIG_PATH)
    expected_sections = [
        "project",
        "paths",
        "video",
        "detection",
        "tracking",
        "team_classification",
        "calibration",
        "visualization",
    ]

    for section in expected_sections:
        assert section in config

def test_validate_config_accepts_valid_configuration():
    """Check that a valid configuration passes validation."""
    config = load_config(CONFIG_PATH)

    validate_config(config)

def test_validate_config_rejects_invalid_confidence():
    """Check that confidence values above 1 are rejected."""
    config = load_config(CONFIG_PATH)
    config["detection"]["confidence_threshold"] = 1.5

    with pytest.raises(ValueError):
        validate_config(config)

def test_validate_config_rejects_invalid_nms():
    """Check that negative NMS thresholds are rejected."""
    config = load_config(CONFIG_PATH)
    config["detection"]["nms_threshold"] = -0.1

    with pytest.raises(ValueError):
        validate_config(config)

def test_validate_config_rejects_invalid_number_of_teams():
    """Check that fewer than two teams are rejected."""
    config = load_config(CONFIG_PATH)
    config["team_classification"]["number_of_teams"] = 1

    with pytest.raises(ValueError):
        validate_config(config)

def test_validate_config_rejects_negative_track_buffer():
    """Check that negative tracking buffers are rejected."""
    config = load_config(CONFIG_PATH)
    config["tracking"]["lost_track_buffer"] = -5

    with pytest.raises(ValueError):
        validate_config(config)
