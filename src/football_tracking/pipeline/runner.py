from __future__ import annotations
from pathlib import Path
import cv2
from football_tracking.config import load_config, validate_config
from football_tracking.detection.detector import RFDETRDetector
from football_tracking.tracking.tracker import ObjectTracker
from football_tracking.video.reader import VideoReader
from football_tracking.video.writer import VideoWriter
from football_tracking.analytics.annotators import FrameAnnotator

class FootballTrackingPipeline:
    def __init__(self, config_path: str | Path, checkpoint_path: str | Path,) -> None:
        self.config_path = Path(config_path).resolve()
        self.project_root = self.config_path.parent.parent
        self.config = load_config(self.config_path)
        validate_config(self.config)
        self.checkpoint_path = self._resolve_path(checkpoint_path)

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)

        if path.is_absolute():
            return path

        return self.project_root / path

    def run(self, input_path: str | Path, output_path: str | Path | None = None,) -> dict:
        input_path = self._resolve_path(input_path)

        if not input_path.is_file():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {self.checkpoint_path}")

        paths_config = self.config["paths"]
        video_config = self.config["video"]
        detection_config = self.config["detection"]
        tracking_config = self.config["tracking"]
        visualization_config = self.config["visualization"]
        save_video = visualization_config.get("save_video", True)

        if save_video:
            if output_path is None:
                output_dir = self._resolve_path(paths_config["output_dir"])
                output_path = (output_dir / f"{input_path.stem}_tracked.mp4")
            else:
                output_path = self._resolve_path(output_path)

            output_path = Path(output_path)

            if output_path.resolve() == input_path.resolve():
                raise ValueError("Input and output video paths must be different.")

        resize_width = video_config.get("resize_width")
        resize_height = video_config.get("resize_height")

        if (resize_width is None) != (resize_height is None):
            raise ValueError("resize_width and resize_height must both be set or both be null.")

        if resize_width is not None:
            for name, value in (("resize_width", resize_width), ("resize_height", resize_height)):
                if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                    raise ValueError(f"{name} must be a positive integer.")

        confidence_threshold = float(detection_config["confidence_threshold"])
        with VideoReader(input_path) as reader:
            original_fps = reader.fps

            if original_fps <= 0:
                raise ValueError("Input video has an invalid FPS.")

            tracker = ObjectTracker(
                track_activation_threshold=tracking_config["track_activation_threshold"],
                lost_track_buffer=tracking_config["lost_track_buffer"],
                minimum_iou_threshold=tracking_config["minimum_iou_threshold"],
                high_conf_det_threshold=tracking_config["high_conf_det_threshold"],
                frame_rate=original_fps,
                minimum_consecutive_frames=tracking_config["minimum_consecutive_frames"])

            detector = RFDETRDetector(
                checkpoint_path=self.checkpoint_path,
                confidence_threshold=confidence_threshold,
                class_confidence_thresholds=detection_config.get(
                    "class_confidence_thresholds", {}
                ),
                use_fp16=True,
                )
            annotator = FrameAnnotator(
                show_labels=visualization_config.get("show_labels", True),
                show_confidence=visualization_config.get("show_confidence", True), 
                show_tracking_ids=visualization_config.get("show_tracking_ids", True))

            output_fps = (float(video_config["target_fps"]) if video_config.get("target_fps") is not None else original_fps)
            output_size = ((resize_width, resize_height) if resize_width is not None else (reader.width, reader.height))
            writer = None

            try:
                if save_video:
                    writer = VideoWriter(output_path=output_path, fps=output_fps, frame_size=output_size)
                    writer.open()

                frame_count = 0
                total_detections = 0
                total_tracked_objects = 0

                print(f"Input video: {input_path}")
                print(f"Resolution: {reader.width}x{reader.height}")
                print(f"FPS: {original_fps:.2f}")
                print(f"Frames: {reader.frame_count}")
                print("Starting football tracking pipeline...")

                for frame in reader:
                    frame_count += 1

                    if resize_width is not None:
                        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)

                    detections = detector.predict(frame)
                    tracked_objects = tracker.update(detections)
                    annotated_frame = annotator.annotate(frame=frame, detections=detections, tracked_objects=tracked_objects)

                    if writer is not None:
                        writer.write(annotated_frame)

                    total_detections += len(detections)
                    total_tracked_objects += len(tracked_objects)

                    if frame_count % 50 == 0:
                        print(f"Processed {frame_count} frames | "f"Detections: {total_detections} | "f"Tracked objects: {total_tracked_objects}")

                print("Pipeline completed.")
                print(f"Processed frames: {frame_count}")
                print(f"Total detections: {total_detections}")
                print(f"Total tracked objects: {total_tracked_objects}")

                if writer is not None:
                    print(f"Output video: {output_path}")

                return {
                    "input_path": str(input_path),
                    "output_path": (
                        str(output_path) if save_video else None
                    ),
                    "frames_processed": frame_count,
                    "total_detections": total_detections,
                    "total_tracked_objects": total_tracked_objects,
                }

            finally:
                if writer is not None:
                    writer.close()

                detector.close()
