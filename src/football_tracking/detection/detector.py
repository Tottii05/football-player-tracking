from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from football_tracking.detection.base import Detection, ObjectDetector
from football_tracking.detection.classes import get_class_id


class RFDETRDetector(ObjectDetector):
    """RF-DETR implementation of the common object detector interface."""

    VALID_CLASSES = {"player", "goalkeeper", "referee", "ball"}

    def __init__(
        self,
        checkpoint_path: str | Path,
        confidence_threshold: float = 0.30,
        class_confidence_thresholds: Mapping[str, float] | None = None,
        use_fp16: bool = True,
        model=None,
    ) -> None:

        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not 0.0 <= confidence_threshold <= 1.0
        ):
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = float(confidence_threshold)
        self.use_fp16 = use_fp16
        self._closed = False

        self.class_confidence_thresholds = {}

        if class_confidence_thresholds is not None:
            if not isinstance(class_confidence_thresholds, Mapping):
                raise TypeError(
                    "class_confidence_thresholds must be a mapping."
                )

            for class_name, threshold in class_confidence_thresholds.items():
                if class_name not in self.VALID_CLASSES:
                    raise ValueError(
                        f"Unsupported class threshold: {class_name}"
                    )

                if (
                    isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float))
                    or not 0.0 <= threshold <= 1.0
                ):
                    raise ValueError(
                        f"Threshold for '{class_name}' "
                        "must be between 0 and 1."
                    )

                self.class_confidence_thresholds[class_name] = float(
                    threshold
                )

        if model is None:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"RF-DETR checkpoint not found: {self.checkpoint_path}"
                )

            from rfdetr import RFDETRNano

            model = RFDETRNano.from_checkpoint(
                str(self.checkpoint_path)
            )

        self._model = model

        self._configure_inference()
        self._configure_classes()

    def _configure_inference(self) -> None:
        """Enable FP16 inference when CUDA is available."""

        if not self.use_fp16:
            return

        import torch

        if torch.cuda.is_available():
            self._model.inference(dtype=torch.float16)

    def _configure_classes(self) -> None:
        """Build the mapping between RF-DETR names and project class IDs."""

        model_classes = getattr(self._model, "class_names", None)

        if not isinstance(model_classes, (list, tuple)):
            raise ValueError(
                "RF-DETR model must expose class_names as a list or tuple."
            )

        self._class_mapping = {}

        for model_class_id, class_name in enumerate(model_classes):
            if not isinstance(class_name, str):
                raise ValueError(
                    "RF-DETR class names must be strings."
                )

            class_name = class_name.strip().lower()

            try:
                canonical_id = get_class_id(class_name)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported RF-DETR class: {class_name}"
                ) from exc

            self._class_mapping[model_class_id] = (
                canonical_id,
                class_name,
            )

        detected_classes = {
            class_name
            for _, class_name in self._class_mapping.values()
        }

        missing_classes = self.VALID_CLASSES - detected_classes

        if missing_classes:
            raise ValueError(
                f"RF-DETR checkpoint is missing classes: "
                f"{sorted(missing_classes)}"
            )

    def _get_inference_threshold(self) -> float:
        """Return the lowest threshold needed to preserve candidate detections."""

        thresholds = [
            self.confidence_threshold,
            *self.class_confidence_thresholds.values(),
        ]

        return min(thresholds)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR frame and return normalized detections."""

        if self._closed:
            raise RuntimeError("Detector is closed.")

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "frame must have shape (height, width, 3)."
            )

        if frame.size == 0:
            raise ValueError("frame cannot be empty.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        predictions = self._model.predict(
            frame_rgb,
            threshold=self._get_inference_threshold(),
        )

        if (
            predictions.xyxy is None
            or predictions.confidence is None
            or predictions.class_id is None
        ):
            raise RuntimeError(
                "RF-DETR returned incomplete detection data."
            )

        boxes = np.asarray(predictions.xyxy)
        confidences = np.asarray(predictions.confidence)
        class_ids = np.asarray(predictions.class_id)

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise RuntimeError(
                "RF-DETR returned invalid bounding boxes."
            )

        if not (
            len(boxes) == len(confidences) == len(class_ids)
        ):
            raise RuntimeError(
                "RF-DETR returned inconsistent detection arrays."
            )

        height, width = frame.shape[:2]
        results = []

        for bbox, confidence, model_class_id in zip(
            boxes, confidences, class_ids, strict=True
        ):
            model_class_id = int(model_class_id)

            if model_class_id not in self._class_mapping:
                raise RuntimeError(
                    f"Unknown RF-DETR class ID: {model_class_id}"
                )

            canonical_id, class_name = self._class_mapping[
                model_class_id
            ]

            confidence = float(confidence)

            class_threshold = self.class_confidence_thresholds.get(
                class_name,
                self.confidence_threshold,
            )

            if confidence < class_threshold:
                continue

            x1, y1, x2, y2 = map(float, bbox)

            x1 = float(np.clip(x1, 0, width))
            y1 = float(np.clip(y1, 0, height))
            x2 = float(np.clip(x2, 0, width))
            y2 = float(np.clip(y2, 0, height))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=canonical_id,
                    class_name=class_name,
                )
            )

        return results

    def close(self) -> None:
        """Release the model reference."""

        self._model = None
        self._closed = True