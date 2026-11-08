from __future__ import annotations
import cv2
import numpy as np
from football_tracking.detection.base import Detection
from football_tracking.tracking.tracker import TrackedObject

class FrameAnnotator:
    COLORS = {
        "player": (0, 200, 0),
        "goalkeeper": (255, 165, 0),
        "referee": (0, 165, 255),
        "ball": (0, 0, 255),
    }

    def __init__(self, show_labels: bool = True, show_confidence: bool = True, show_tracking_ids: bool = True,) -> None:
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_tracking_ids = show_tracking_ids

    @staticmethod
    def _calculate_iou(bbox_a: tuple[float, float, float, float], bbox_b: tuple[float, float, float, float],) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(0.0, intersection_x2 - intersection_x1)
        intersection_height = max(0.0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union_area = area_a + area_b - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def _find_track_id(self, detection: Detection, tracked_objects: list[TrackedObject],) -> int | None:
        if detection.class_name == "ball":
            return None

        best_iou = 0.0
        best_track_id = None

        for tracked_object in tracked_objects:
            if tracked_object.class_name != detection.class_name:
                continue

            iou = self._calculate_iou(detection.bbox, tracked_object.bbox)

            if iou > best_iou:
                best_iou = iou
                best_track_id = tracked_object.track_id

        if best_iou < 0.5:
            return None

        return best_track_id

    def annotate(self, frame: np.ndarray, detections: list[Detection], tracked_objects: list[TrackedObject]) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3).")

        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
            color = self.COLORS.get(detection.class_name, (255, 255, 255))
            track_id = self._find_track_id(detection, tracked_objects)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label_parts = []

            if self.show_labels:
                label_parts.append(detection.class_name)

            if self.show_tracking_ids and track_id is not None:
                label_parts.append(f"ID: {track_id}")

            if self.show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")

            if label_parts:
                label = " | ".join(label_parts)
                (text_width, text_height), baseline = (cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1))
                label_y = max(y1, text_height + 8)
                cv2.rectangle(annotated_frame, (x1, label_y - text_height - 8), (x1 + text_width + 6, label_y + baseline - 4), color, thickness=-1)
                cv2.putText(annotated_frame, label, (x1 + 3, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame