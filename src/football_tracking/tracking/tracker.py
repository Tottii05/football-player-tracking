from dataclasses import dataclass
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from football_tracking.detection.base import Detection
from football_tracking.detection.classes import get_class_id, get_class_name

@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

class ObjectTracker:
    TRACKABLE_CLASSES = {"player", "goalkeeper", "referee"}

    def __init__(
        self,
        track_activation_threshold: float = 0.35,
        lost_track_buffer: int = 30,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
        frame_rate: float = 25.0,
        minimum_consecutive_frames: int = 2,
    ) -> None:
        if not 0.0 <= track_activation_threshold <= 1.0:
            raise ValueError("track_activation_threshold must be between 0 and 1")

        if not 0.0 <= high_conf_det_threshold <= 1.0:
            raise ValueError("high_conf_det_threshold must be between 0 and 1")

        if not 0.0 <= minimum_iou_threshold <= 1.0:
            raise ValueError("minimum_iou_threshold must be between 0 and 1")

        if lost_track_buffer < 1:
            raise ValueError("lost_track_buffer must be positive")

        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")

        if minimum_consecutive_frames < 1:
            raise ValueError("minimum_consecutive_frames must be positive")

        self._tracker = ByteTrackTracker(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_iou_threshold=minimum_iou_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        if not isinstance(detections, list):
            raise TypeError("detections must be a list of Detection objects")

        if not all(isinstance(detection, Detection) for detection in detections):
            raise TypeError("Every element in detections must be a Detection")

        trackable = [
            detection
            for detection in detections
            if detection.class_name in self.TRACKABLE_CLASSES
        ]

        if not trackable:
            self._tracker.update(sv.Detections.empty())
            return []

        supervision_detections = sv.Detections(
            xyxy=np.asarray([d.bbox for d in trackable], dtype=np.float32),
            confidence=np.asarray([d.confidence for d in trackable], dtype=np.float32),
            class_id=np.asarray([get_class_id(d.class_name) for d in trackable], dtype=int))

        tracked = self._tracker.update(supervision_detections)

        if tracked.tracker_id is None:
            return []

        results = []

        for bbox, confidence, class_id, track_id in zip(tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id):
            if track_id < 0:
                continue

            if class_id is None or confidence is None:
                continue

            class_id = int(class_id)
            results.append(
                TrackedObject(track_id=int(track_id), bbox=tuple(float(value) for value in bbox), confidence=float(confidence), class_id=class_id, class_name=get_class_name(class_id)))
        return results

    def reset(self) -> None:
        self._tracker.reset()