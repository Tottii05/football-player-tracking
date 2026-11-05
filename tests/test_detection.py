
import numpy as np
import pytest

from src.football_tracking.detection.base import Detection, ObjectDetector
from src.football_tracking.detection.classes import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    get_class_id,
    get_class_name,
)


# Detection tests

def test_detection_accepts_valid_data():
    detection = Detection(
        bbox=(10, 20, 100, 200),
        confidence=0.95,
        class_id=0,
        class_name="player",
    )

    assert detection.bbox == (10.0, 20.0, 100.0, 200.0)
    assert detection.confidence == 0.95
    assert detection.class_id == 0
    assert detection.class_name == "player"


def test_detection_rejects_invalid_bbox():
    with pytest.raises(ValueError):
        Detection((100, 20, 10, 200), 0.9, 0, "player")


def test_detection_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        Detection((10, 20, 100, 200), 1.5, 0, "player")


def test_detection_rejects_invalid_class_id():
    with pytest.raises(ValueError):
        Detection((10, 20, 100, 200), 0.9, -1, "player")


def test_detection_rejects_empty_class_name():
    with pytest.raises(ValueError):
        Detection((10, 20, 100, 200), 0.9, 0, " ")


def test_detection_is_immutable():
    detection = Detection(
        bbox=(10, 20, 100, 200),
        confidence=0.9,
        class_id=0,
        class_name="player",
    )

    with pytest.raises(AttributeError):
        detection.confidence = 0.5


# ObjectDetector tests

def test_object_detector_requires_predict_implementation():
    with pytest.raises(TypeError):
        ObjectDetector()


def test_custom_detector_implements_common_interface():
    class DummyDetector(ObjectDetector):
        def predict(self, frame: np.ndarray):
            return [
                Detection(
                    bbox=(10, 20, 100, 200),
                    confidence=0.95,
                    class_id=0,
                    class_name="player",
                )
            ]

    detector = DummyDetector()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    detections = detector.predict(frame)

    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert detections[0].class_name == "player"


def test_object_detector_can_be_closed():
    class DummyDetector(ObjectDetector):
        def predict(self, frame: np.ndarray):
            return []

    detector = DummyDetector()
    detector.close()


# Class mapping tests

def test_class_mapping_contains_expected_classes():
    assert CLASS_TO_ID == {
        "player": 0,
        "goalkeeper": 1,
        "referee": 2,
        "ball": 3,
    }


def test_class_mapping_is_bidirectional():
    for class_name, class_id in CLASS_TO_ID.items():
        assert ID_TO_CLASS[class_id] == class_name
        assert get_class_id(class_name) == class_id
        assert get_class_name(class_id) == class_name


def test_get_class_name_rejects_unknown_id():
    with pytest.raises(ValueError):
        get_class_name(99)


def test_get_class_id_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_class_id("stadium")


def test_get_class_id_normalizes_class_name():
    assert get_class_id("  PLAYER  ") == 0


def test_get_class_name_rejects_boolean():
    with pytest.raises(TypeError):
        get_class_name(True)
