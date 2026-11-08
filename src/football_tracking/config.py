from pathlib import Path

import yaml


VALID_CLASSES = {"player", "goalkeeper", "referee", "ball"}


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a YAML dictionary."
        )

    return config


def validate_config(config: dict) -> None:
    required_sections = [
        "project",
        "paths",
        "video",
        "detection",
        "tracking",
        "team_classification",
        "calibration",
        "visualization",
    ]

    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(
                f"Missing or invalid configuration section: {section}"
            )

    detection = config["detection"]
    tracking = config["tracking"]
    team_classification = config["team_classification"]

    confidence = detection.get("confidence_threshold")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError(
            "confidence_threshold must be between 0 and 1."
        )

    nms = detection.get("nms_threshold")

    if (
        isinstance(nms, bool)
        or not isinstance(nms, (int, float))
        or not 0 <= nms <= 1
    ):
        raise ValueError(
            "nms_threshold must be between 0 and 1."
        )

    class_thresholds = detection.get(
        "class_confidence_thresholds", {}
    )

    if not isinstance(class_thresholds, dict):
        raise ValueError(
            "class_confidence_thresholds must be a dictionary."
        )

    unknown_classes = set(class_thresholds) - VALID_CLASSES

    if unknown_classes:
        raise ValueError(
            f"Unknown classes in class_confidence_thresholds: "
            f"{sorted(unknown_classes)}"
        )

    for class_name, threshold in class_thresholds.items():
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                f"Confidence threshold for '{class_name}' "
                "must be between 0 and 1."
            )

    number_of_teams = team_classification.get("number_of_teams")

    if (
        isinstance(number_of_teams, bool)
        or not isinstance(number_of_teams, int)
        or number_of_teams < 2
    ):
        raise ValueError(
            "number_of_teams must be an integer greater than or equal to 2."
        )

    lost_track_buffer = tracking.get("lost_track_buffer")

    if (
        isinstance(lost_track_buffer, bool)
        or not isinstance(lost_track_buffer, int)
        or lost_track_buffer < 0
    ):
        raise ValueError(
            "lost_track_buffer must be a non-negative integer."
        )