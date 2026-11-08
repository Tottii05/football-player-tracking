
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rfdetr_threshold_evaluation"

CONFIDENCE_THRESHOLDS = (0.10, 0.30, 0.50)
MIN_INFERENCE_CONFIDENCE = 0.001
IOU_THRESHOLD = 0.50

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_ground_truth(label_path: Path, image_width: int, image_height: int):
    """Load YOLO labels and convert normalized XYWH boxes to absolute XYXY."""

    ground_truth = defaultdict(list)

    if not label_path.exists():
        return ground_truth

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.strip().split()

            if len(values) != 5:
                print(
                    f"Warning: invalid label at {label_path.name}:{line_number}"
                )
                continue

            class_id = int(float(values[0]))
            center_x, center_y, width, height = map(float, values[1:])

            x1 = (center_x - width / 2) * image_width
            y1 = (center_y - height / 2) * image_height
            x2 = (center_x + width / 2) * image_width
            y2 = (center_y + height / 2) * image_height

            x1 = max(0.0, min(float(image_width), x1))
            y1 = max(0.0, min(float(image_height), y1))
            x2 = max(0.0, min(float(image_width), x2))
            y2 = max(0.0, min(float(image_height), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            ground_truth[class_id].append(
                np.array([x1, y1, x2, y2], dtype=np.float32)
            )

    return ground_truth


def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calculate intersection over union between two XYXY boxes."""

    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def extract_predictions(model, image_bgr: np.ndarray):
    """Run inference on the original image and return standard predictions."""

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detections = model.predict(
        image_rgb,
        threshold=MIN_INFERENCE_CONFIDENCE,
    )

    boxes = np.asarray(detections.xyxy, dtype=np.float32)

    if detections.confidence is None or detections.class_id is None:
        return []

    confidences = np.asarray(detections.confidence, dtype=np.float32)
    class_ids = np.asarray(detections.class_id, dtype=np.int32)

    predictions = []

    for box, confidence, class_id in zip(
        boxes, confidences, class_ids
    ):
        if not np.isfinite(box).all() or not np.isfinite(confidence):
            continue

        predictions.append(
            {
                "bbox": box,
                "confidence": float(confidence),
                "class_id": int(class_id),
            }
        )

    return predictions


def match_predictions(
    predictions,
    ground_truth_by_image,
    class_id: int,
    confidence_threshold: float | None,
):
    """
    Match predictions to ground truth, separately for each image.

    Each ground-truth object can be matched only once.
    Matching requires the same class and IoU >= IOU_THRESHOLD.
    """

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for image_name, image_predictions in predictions.items():
        gt_boxes = ground_truth_by_image[image_name].get(class_id, [])

        candidates = [
            prediction
            for prediction in image_predictions
            if prediction["class_id"] == class_id
            and (
                confidence_threshold is None
                or prediction["confidence"] >= confidence_threshold
            )
        ]

        candidates.sort(
            key=lambda prediction: prediction["confidence"],
            reverse=True,
        )

        matched_gt = set()

        for prediction in candidates:
            best_iou = 0.0
            best_gt_index = None

            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched_gt:
                    continue

                iou = calculate_iou(prediction["bbox"], gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_gt_index is not None and best_iou >= IOU_THRESHOLD:
                true_positives += 1
                matched_gt.add(best_gt_index)
            else:
                false_positives += 1

        false_negatives += len(gt_boxes) - len(matched_gt)

    return true_positives, false_positives, false_negatives


def calculate_ap50(predictions, ground_truth_by_image, class_id: int):
    """Calculate interpolated AP at IoU 0.50 for one class."""

    total_ground_truth = sum(
        len(image_gt.get(class_id, []))
        for image_gt in ground_truth_by_image.values()
    )

    if total_ground_truth == 0:
        return None

    ranked_predictions = []

    for image_name, image_predictions in predictions.items():
        for prediction in image_predictions:
            if prediction["class_id"] == class_id:
                ranked_predictions.append(
                    (
                        prediction["confidence"],
                        image_name,
                        prediction["bbox"],
                    )
                )

    ranked_predictions.sort(key=lambda item: item[0], reverse=True)

    matched_by_image = {
        image_name: set()
        for image_name in ground_truth_by_image
    }

    tp_values = []
    fp_values = []

    for _, image_name, predicted_box in ranked_predictions:
        gt_boxes = ground_truth_by_image[image_name].get(class_id, [])

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_by_image[image_name]:
                continue

            iou = calculate_iou(predicted_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= IOU_THRESHOLD:
            matched_by_image[image_name].add(best_gt_index)
            tp_values.append(1)
            fp_values.append(0)
        else:
            tp_values.append(0)
            fp_values.append(1)

    if not ranked_predictions:
        return 0.0

    cumulative_tp = np.cumsum(tp_values)
    cumulative_fp = np.cumsum(fp_values)

    recalls = cumulative_tp / total_ground_truth
    precisions = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, 1
    )

    # Add endpoints and calculate the interpolated precision envelope.
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(
            precisions[index], precisions[index + 1]
        )

    change_points = np.where(recalls[1:] != recalls[:-1])[0]

    ap = np.sum(
        (recalls[change_points + 1] - recalls[change_points])
        * precisions[change_points + 1]
    )

    return float(ap)


def calculate_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No validation images found in {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trained RF-DETR checkpoint...")
    print(f"Checkpoint: {CHECKPOINT}")
    model = RFDETRNano.from_checkpoint(CHECKPOINT)

    class_names = list(model.class_names)
    print(f"Model classes: {class_names}")

    predictions_by_image = {}
    ground_truth_by_image = {}

    print(f"\nEvaluating {len(image_paths)} validation images...")

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: could not read {image_path.name}")
            continue

        height, width = image.shape[:2]

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        ground_truth_by_image[image_path.name] = load_ground_truth(
            label_path, width, height
        )

        # Inference receives the original image, without GT drawings.
        predictions_by_image[image_path.name] = extract_predictions(
            model, image
        )

        print(f"[{index}/{len(image_paths)}] {image_path.name}")

    if not predictions_by_image:
        raise RuntimeError("No images could be evaluated.")

    results = []

    print("\n" + "=" * 78)
    print("THRESHOLD EVALUATION — IoU 0.50")
    print("=" * 78)

    for class_id, class_name in enumerate(class_names):
        total_gt = sum(
            len(image_gt.get(class_id, []))
            for image_gt in ground_truth_by_image.values()
        )

        print(f"\nClass: {class_name} | Ground truth: {total_gt}")

        ap50 = calculate_ap50(
            predictions_by_image,
            ground_truth_by_image,
            class_id,
        )

        for threshold in CONFIDENCE_THRESHOLDS:
            tp, fp, fn = match_predictions(
                predictions_by_image,
                ground_truth_by_image,
                class_id,
                threshold,
            )

            precision, recall, f1 = calculate_metrics(tp, fp, fn)

            result = {
                "threshold": threshold,
                "class": class_name,
                "ground_truth": total_gt,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "AP50": ap50,
            }

            results.append(result)

            print(
                f"  threshold={threshold:.2f} | "
                f"TP={tp:4d} FP={fp:4d} FN={fn:4d} | "
                f"P={precision:.4f} R={recall:.4f} "
                f"F1={f1:.4f} AP50={ap50:.4f}"
            )

    csv_path = OUTPUT_DIR / "threshold_metrics.csv"

    fieldnames = [
        "threshold",
        "class",
        "ground_truth",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "AP50",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 78)
    print(f"Evaluation completed.")
    print(f"CSV results: {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
