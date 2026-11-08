from pathlib import Path
import cv2
import numpy as np

class VideoWriter:
    def __init__(self, output_path: str | Path, fps: float, frame_size: tuple[int, int], codec: str = "mp4v",) -> None:
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self.frame_size = frame_size
        self.codec = codec
        self._writer: cv2.VideoWriter | None = None

        if self.fps <= 0:
            raise ValueError("FPS must be greater than zero.")

        if (len(self.frame_size) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.frame_size)):
            raise ValueError("frame_size must contain positive integers (width, height).")

        if len(self.codec) != 4:
            raise ValueError("Codec must contain exactly 4 characters.")

    @property
    def is_opened(self) -> bool:
        """Return whether the video writer is currently open."""
        return (self._writer is not None and self._writer.isOpened())

    def open(self) -> None:
        """Create the output video file."""
        if self.is_opened:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, self.frame_size)

        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not create output video: {self.output_path}")
        self._writer = writer

    def write(self, frame: np.ndarray) -> None:
        """Write one BGR frame to the output video."""
        if not self.is_opened:
            raise RuntimeError("Video writer is not open. Use VideoWriter as a context manager or call open() first.")

        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be a 3-channel BGR image.")
        expected_width, expected_height = self.frame_size
        actual_height, actual_width = frame.shape[:2]

        if (actual_width != expected_width or actual_height != expected_height):
            raise ValueError("Frame dimensions do not match the configured frame_size. "
                f"Expected {self.frame_size}, "
                f"received {(actual_width, actual_height)}.")
        self._writer.write(frame)

    def close(self) -> None:
        """Release the video writer resource."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
