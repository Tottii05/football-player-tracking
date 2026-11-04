from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np

@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata describing a video file."""
    width: int
    height: int
    fps: float
    frame_count: int

class VideoReader:
    """Read a video file one frame at a time."""
    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        self._capture: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    def open(self) -> None:
        """Open the video and read its metadata."""
        if self.is_opened:
            return

        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video: {self.video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or fps <= 0:
            capture.release()
            raise ValueError(f"Video has invalid metadata: {self.video_path}")

        self._capture = capture
        self._metadata = VideoMetadata(width=width, height=height, fps=fps, frame_count=frame_count)

    @property
    def is_opened(self) -> bool:
        """Return whether the video is currently open."""
        return (self._capture is not None and self._capture.isOpened())

    @property
    def metadata(self) -> VideoMetadata:
        """Return video metadata. The video must be open."""
        if self._metadata is None:
            raise RuntimeError("Video metadata is unavailable. Open the video first.")
        return self._metadata

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    @property
    def fps(self) -> float:
        return self.metadata.fps

    @property
    def frame_count(self) -> int:
        return self.metadata.frame_count

    def read(self) -> np.ndarray | None:
        """
        Read the next frame.

        Returns:
            A BGR image as a NumPy array, or None when the video ends.
        """
        if not self.is_opened:
            raise RuntimeError("Video is not open. Use VideoReader as a context manager or call open() first.")
        success, frame = self._capture.read()

        if not success:
            return None

        return frame

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over all frames until the video ends."""
        while True:
            frame = self.read()
            if frame is None:
                break

            yield frame

    def close(self) -> None:
        """Release the video capture resource."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
