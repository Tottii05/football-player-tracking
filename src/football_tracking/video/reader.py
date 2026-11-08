from pathlib import Path
from typing import Iterator, Optional
import av
import cv2
import numpy as np

class VideoReader:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {self.video_path}"
            )

        self.container = av.open(
            str(self.video_path),
            options={"ignore_editlist": "1"},
        )

        video_streams = self.container.streams.video

        if not video_streams:
            self.container.close()
            raise ValueError(
                f"No video stream found: {self.video_path}"
            )

        self.stream = video_streams[0]

        self.fps = float(self.stream.average_rate or 0)

        if self.fps <= 0:
            self.container.close()
            raise ValueError(
                f"Invalid video FPS: {self.video_path}"
            )

        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height

        self.frame_count = self.stream.frames

        if not self.frame_count:
            self.frame_count = 0

        self.duration = (
            float(self.stream.duration * self.stream.time_base)
            if self.stream.duration is not None
            else 0.0
        )

        self._frame_iterator = None
        self._closed = False

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield decoded frames in BGR format."""

        if self._closed:
            raise RuntimeError("VideoReader is already closed.")

        if self._frame_iterator is None:
            self._frame_iterator = self.container.decode(
                self.stream
            )

        for frame in self._frame_iterator:
            yield frame.to_ndarray(format="bgr24")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one frame."""

        if self._closed:
            return False, None

        if self._frame_iterator is None:
            self._frame_iterator = self.container.decode(
                self.stream
            )

        try:
            frame = next(self._frame_iterator)
            return True, frame.to_ndarray(format="bgr24")
        except StopIteration:
            return False, None
    @property
    def is_opened(self) -> bool:
        return not self._closed

    def release(self) -> None:
        """Release video resources."""

        if not self._closed:
            self.container.close()
            self._closed = True
            self._frame_iterator = None

    def get(self, property_id: int) -> float:
        """Get common video properties."""

        properties = {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
            cv2.CAP_PROP_POS_MSEC: 0.0,
        }

        return float(properties.get(property_id, 0.0))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
