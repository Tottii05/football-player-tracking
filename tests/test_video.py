import cv2
import numpy as np
import pytest
from src.football_tracking.video.reader import VideoReader
from src.football_tracking.video.writer import VideoWriter

def create_test_video(path, frame_count=5, width=320, height=240, fps=10):
    """Create a small synthetic video for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    if not writer.isOpened():
        pytest.skip("MP4 video codec is not available in this environment.")

    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (index * 30, 100, 200)
        writer.write(frame)
    writer.release()

def test_video_reader_opens_video_and_reads_metadata(tmp_path):
    video_path = tmp_path / "input.mp4"
    create_test_video(video_path)
    with VideoReader(video_path) as reader:
        assert reader.is_opened
        assert reader.width == 320
        assert reader.height == 240
        assert reader.fps > 0
        assert reader.frame_count == 5

def test_video_reader_iterates_over_all_frames(tmp_path):
    video_path = tmp_path / "input.mp4"
    create_test_video(video_path)
    with VideoReader(video_path) as reader:
        frames = list(reader)
    assert len(frames) == 5
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)

def test_video_reader_releases_resource_after_context(tmp_path):
    video_path = tmp_path / "input.mp4"
    create_test_video(video_path)
    reader = VideoReader(video_path)
    with reader:
        assert reader.is_opened
    assert not reader.is_opened

def test_video_reader_raises_error_for_missing_file(tmp_path):
    video_path = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError):
        with VideoReader(video_path):
            pass

def test_video_writer_creates_output_video(tmp_path):
    output_path = tmp_path / "output" / "result.mp4"
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with VideoWriter(output_path, fps=10, frame_size=(320, 240)) as writer:
        for _ in range(5):
            writer.write(frame)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_video_writer_rejects_wrong_frame_dimensions(tmp_path):
    output_path = tmp_path / "output.mp4"
    wrong_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with VideoWriter(output_path, fps=10, frame_size=(320, 240)) as writer:
        with pytest.raises(ValueError):
            writer.write(wrong_frame)

def test_video_writer_releases_resource_after_context(tmp_path):
    output_path = tmp_path / "output.mp4"
    writer = VideoWriter(output_path, fps=10, frame_size=(320, 240),)
    with writer:
        assert writer.is_opened
    assert not writer.is_opened

def test_video_writer_rejects_invalid_fps(tmp_path):
    with pytest.raises(ValueError):
        VideoWriter(tmp_path / "output.mp4", fps=0, frame_size=(320, 240),)

def test_video_writer_rejects_invalid_frame_size(tmp_path):
    with pytest.raises(ValueError):
        VideoWriter(tmp_path / "output.mp4", fps=10, frame_size=(0, 240),)