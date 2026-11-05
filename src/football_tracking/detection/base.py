"""Base interfaces and data structures for object detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Represents a single detected object."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    def __post_init__(self):
        if not isinstance(self.bbox, (tuple, list)) or len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly four coordinates.")

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in self.bbox
        ):
            raise ValueError("bbox coordinates must be finite numbers.")

        x1, y1, x2, y2 = self.bbox

        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height.")

        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be between 0 and 1.")

        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id < 0
        ):
            raise ValueError("class_id must be a non-negative integer.")

        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a non-empty string.")

        object.__setattr__(self, "bbox", tuple(float(v) for v in self.bbox))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "class_name", self.class_name.strip())


class ObjectDetector(ABC):
    """Abstract interface for object detection models."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a frame."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources used by the detector."""