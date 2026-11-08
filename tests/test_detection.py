import numpy as np
import pytest
from types import SimpleNamespace
from src.football_tracking.detection.detector import RFDETRDetector
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

# RF-DETR detector tests

def test_detector_applies_class_specific_confidence_thresholds():
    class DummyRFDETRModel:
        class_names = [
            "ball",
            "goalkeeper",
            "player",
            "referee",
        ]

        def predict(self, frame, threshold):
            self.received_threshold = threshold

            return SimpleNamespace(
                xyxy=np.array([
                    [10, 10, 30, 30],
                    [40, 10, 60, 30],
                    [70, 10, 90, 30],
                    [100, 10, 120, 30],
                ]),
                confidence=np.array([
                    0.40,
                    0.40,
                    0.40,
                    0.40,
                ]),
                class_id=np.array([0, 1, 2, 3]),
            )

    model = DummyRFDETRModel()

    detector = RFDETRDetector(
        checkpoint_path="unused.ckpt",
        confidence_threshold=0.30,
        class_confidence_thresholds={
            "ball": 0.50,
            "goalkeeper": 0.50,
            "player": 0.30,
            "referee": 0.30,
        },
        use_fp16=False,
        model=model,
    )

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    detections = detector.predict(frame)

    assert len(detections) == 2

    detected_classes = [
        detection.class_name
        for detection in detections
    ]

    assert detected_classes == ["player", "referee"]

    detector.close()


def test_detector_uses_lowest_configured_inference_threshold():
    class DummyRFDETRModel:
        class_names = [
            "ball",
            "goalkeeper",
            "player",
            "referee",
        ]

        def predict(self, frame, threshold):
            self.received_threshold = threshold

            return SimpleNamespace(
                xyxy=np.empty((0, 4)),
                confidence=np.array([]),
                class_id=np.array([]),
            )

    model = DummyRFDETRModel()

    detector = RFDETRDetector(
        checkpoint_path="unused.ckpt",
        confidence_threshold=0.30,
        class_confidence_thresholds={
            "ball": 0.50,
            "goalkeeper": 0.50,
            "player": 0.20,
            "referee": 0.40,
        },
        use_fp16=False,
        model=model,
    )

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    detections = detector.predict(frame)

    assert detections == []
    assert model.received_threshold == 0.20

    detector.close()
