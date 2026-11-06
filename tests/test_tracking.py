
import pytest

from football_tracking.detection.base import Detection
from football_tracking.tracking import ObjectTracker, TrackedObject


def make_detection(
    x1: float = 100,
    confidence: float = 0.9,
    class_name: str = "player",
) -> Detection:
    return Detection(
        bbox=(x1, 100, x1 + 50, 200),
        confidence=confidence,
        class_id=0,
        class_name=class_name,
    )


def make_tracker() -> ObjectTracker:
    return ObjectTracker(
        track_activation_threshold=0.35,
        lost_track_buffer=30,
        minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.6,
        frame_rate=25,
        minimum_consecutive_frames=2,
    )


def test_tracker_returns_confirmed_objects():
    tracker = make_tracker()

    # First detection: the track is not confirmed yet.
    assert tracker.update([make_detection()]) == []

    # Second consecutive detection: the track is confirmed.
    results = tracker.update([make_detection()])

    assert len(results) == 1
    assert isinstance(results[0], TrackedObject)
    assert results[0].track_id >= 0


def test_tracker_preserves_id_when_object_moves():
    tracker = make_tracker()

    tracker.update([make_detection(x1=100)])
    first_results = tracker.update([make_detection(x1=105)])

    assert len(first_results) == 1
    original_id = first_results[0].track_id

    results = tracker.update([make_detection(x1=110)])

    assert len(results) == 1
    assert results[0].track_id == original_id
    assert results[0].bbox[0] == 110.0


def test_tracker_recovers_track_after_empty_frame():
    tracker = make_tracker()

    tracker.update([make_detection(x1=100)])
    initial_results = tracker.update([make_detection(x1=105)])

    original_id = initial_results[0].track_id

    # One frame without detections.
    assert tracker.update([]) == []

    # The object reappears within the lost-track buffer.
    results = tracker.update([make_detection(x1=110)])

    assert len(results) == 1
    assert results[0].track_id == original_id


def test_tracker_ignores_ball_detections():
    tracker = make_tracker()

    results = tracker.update([
        make_detection(class_name="ball")
    ])

    assert results == []


def test_tracker_reset_clears_previous_tracks():
    tracker = make_tracker()

    tracker.update([make_detection()])
    results = tracker.update([make_detection()])

    assert len(results) == 1

    tracker.reset()

    # A new track needs confirmation again.
    assert tracker.update([make_detection()]) == []

    results = tracker.update([make_detection()])

    assert len(results) == 1
    assert results[0].track_id >= 0


def test_tracker_rejects_non_list_input():
    tracker = make_tracker()

    with pytest.raises(TypeError):
        tracker.update(None)


def test_tracker_rejects_invalid_list_elements():
    tracker = make_tracker()

    with pytest.raises(TypeError):
        tracker.update(["not a Detection"])


def test_tracker_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        ObjectTracker(track_activation_threshold=1.5)
