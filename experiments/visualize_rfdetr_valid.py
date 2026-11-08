
from __future__ import annotations

from pathlib import Path

import cv2
from rfdetr import RFDETRNano


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "football_players_detection"
IMAGE_DIR = DATASET_DIR / "valid" / "images"
LABEL_DIR = DATASET_DIR / "valid" / "labels"

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "rfdetr_nano_football"
    / "checkpoint_best_ema.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rfdetr_valid_diagnostic"

CONFIDENCE_THRESHOLDS = (0.10, 0.30, 0.50)

CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]

GT_COLOR = (255, 0, 0)

CLASS_COLORS = {
    "ball": (0, 255, 255),
    "goalkeeper": (255, 0, 255),
    "player": (0, 200, 0),
    "referee": (255, 100, 0),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_box(frame, bbox, label, color, thickness=2):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 6, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_ground_truth(frame, label_path):
    height, width = frame.shape[:2]

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()

            if not values:
                continue

            class_id, xc, yc, bw, bh = map(float, values)
            class_id = int(class_id)

            if not 0 <= class_id < len(CLASS_NAMES):
                continue

            bbox = (
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            )

            draw_box(
                frame,
                bbox,
                f"GT {CLASS_NAMES[class_id]}",
                GT_COLOR,
            )


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(IMAGE_DIR)

    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(LABEL_DIR)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError("No validation images found.")

    print("Loading trained RF-DETR checkpoint...")
    model = RFDETRNano.from_checkpoint(CHECKPOINT)

    if list(model.class_names) != CLASS_NAMES:
        raise ValueError(f"Unexpected classes: {model.class_names}")

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Images: {len(image_paths)}")
    print(f"Thresholds: {CONFIDENCE_THRESHOLDS}")

    for threshold in CONFIDENCE_THRESHOLDS:
        threshold_dir = OUTPUT_DIR / f"confidence_{threshold:.2f}"
        threshold_dir.mkdir(parents=True, exist_ok=True)

        total_by_class = {name: 0 for name in CLASS_NAMES}

        print(f"\nProcessing confidence threshold: {threshold:.2f}")

        for index, image_path in enumerate(image_paths, start=1):
            original = cv2.imread(str(image_path))

            if original is None:
                raise RuntimeError(f"Could not read {image_path}")

            label_path = LABEL_DIR / f"{image_path.stem}.txt"

            if not label_path.is_file():
                raise FileNotFoundError(label_path)

            image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            detections = model.predict(
                image_rgb,
                threshold=threshold,
            )

            prediction_frame = original.copy()

            for bbox, confidence, class_id in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
            ):
                class_id = int(class_id)

                if not 0 <= class_id < len(CLASS_NAMES):
                    continue

                class_name = CLASS_NAMES[class_id]
                total_by_class[class_name] += 1

                draw_box(
                    prediction_frame,
                    bbox,
                    f"{class_name} {float(confidence):.2f}",
                    CLASS_COLORS[class_name],
                )

            # Ground truth is drawn on a separate copy.
            ground_truth_frame = original.copy()
            draw_ground_truth(ground_truth_frame, label_path)

            # Save predictions and ground truth separately.
            cv2.imwrite(
                str(threshold_dir / image_path.name),
                prediction_frame,
            )

            cv2.imwrite(
                str(threshold_dir / f"{image_path.stem}_GT{image_path.suffix}"),
                ground_truth_frame,
            )

            print(f"[{index}/{len(image_paths)}] {image_path.name}")

        print(f"\nThreshold: {threshold:.2f}")

        for class_name, count in total_by_class.items():
            print(f"{class_name}: {count} predictions")

        print(f"Results: {threshold_dir}")

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
