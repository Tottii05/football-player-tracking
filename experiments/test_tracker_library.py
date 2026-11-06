import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

def make_detections(x1: float, confidence: float = 0.9):
    return sv.Detections(
        xyxy=np.array([[x1, 100, x1 + 50, 200]], dtype=np.float32), confidence=np.array([confidence], dtype=np.float32), class_id=np.array([0], dtype=int))


tracker = ByteTrackTracker(
    lost_track_buffer=30,
    frame_rate=25,
    track_activation_threshold=0.35,
    minimum_consecutive_frames=2,
    minimum_iou_threshold=0.1,
    high_conf_det_threshold=0.6,
)

print("Fotograma 1")
result = tracker.update(make_detections(100))
print("IDs:", result.tracker_id)
print("\nFotograma 2")
result = tracker.update(make_detections(105))
print("IDs:", result.tracker_id)
print("\nFotograma 3")
result = tracker.update(make_detections(110))
print("IDs:", result.tracker_id)
print("\nFotograma 4: detección vacía")
result = tracker.update(sv.Detections.empty())
print("IDs:", result.tracker_id)
print("\nFotograma 5: jugador reaparece")
result = tracker.update(make_detections(115))
print("IDs:", result.tracker_id)
print("\nReiniciando tracker...")
tracker.reset()
result = tracker.update(make_detections(100))
print("IDs después de reset:", result.tracker_id)
