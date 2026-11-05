from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from rfdetr import RFDETRNano

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "football_players_detection"
    / "valid"
    / "images"
)
LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "football_players_detection"
    / "valid"
    / "labels"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rfdetr_evaluation"
CONFIDENCE_THRESHOLD = 0.001
REPORT_CONFIDENCE = 0.30

# Internal evaluation classes:
# 0 = person
# 1 = sports ball

# RF-DETR pretrained COCO IDs -> evaluation IDs
MODEL_CLASS_MAP = {
    1: 0,   # person
    37: 1,  # sports ball
}

# Original Roboflow dataset IDs -> evaluation IDs
DATASET_CLASS_MAP = {
    0: 1,  # ball -> sports ball
    1: 0,  # goalkeeper -> person
    2: 0,  # player -> person
    3: 0,  # referee -> person
}

CLASS_NAMES = {
    0: "person",
    1: "sports ball",
}

IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)

def load_ground_truth(label_path, image_width, image_height):
    """Load YOLO annotations and convert them to absolute XYXY boxes."""
    ground_truth = []

    if not label_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de anotaciones: {label_path}")

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.strip().split()

            if not values:
                continue

            if len(values) != 5:
                raise ValueError(f"Formato incorrecto en {label_path}, "f"línea {line_number}")

            original_class = int(values[0])

            if original_class not in DATASET_CLASS_MAP:
                raise ValueError(f"Clase desconocida {original_class} "f"en {label_path}")
            x_center, y_center, width, height = map(float, values[1:])
            x1 = (x_center - width / 2) * image_width
            y1 = (y_center - height / 2) * image_height
            x2 = (x_center + width / 2) * image_width
            y2 = (y_center + height / 2) * image_height

            ground_truth.append({
                "class_id": DATASET_CLASS_MAP[original_class],
                "bbox": np.array([x1, y1, x2, y2]),
            })

    return ground_truth

def calculate_iou(box_a, box_b):
    """Calculate Intersection over Union between two XYXY boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union

def calculate_precision_recall(predictions, ground_truths, confidence_threshold=0.30, iou_threshold=0.50,):
    """Calculate precision and recall at fixed confidence and IoU."""
    true_positives = 0
    false_positives = 0
    total_ground_truths = sum(len(items) for items in ground_truths.values())

    for class_id in CLASS_NAMES:
        class_predictions = [
            prediction
            for prediction in predictions
            if prediction["class_id"] == class_id
            and prediction["confidence"] >= confidence_threshold
        ]

        class_predictions.sort(key=lambda item: item["confidence"], reverse=True)
        matched = defaultdict(set)

        for prediction in class_predictions:
            image_id = prediction["image_id"]

            candidates = ground_truths[image_id]
            best_iou = 0.0
            best_index = None

            for index, gt in enumerate(candidates):
                if gt["class_id"] != class_id:
                    continue

                if index in matched[image_id]:
                    continue

                iou = calculate_iou(prediction["bbox"], gt["bbox"])

                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index is not None and best_iou >= iou_threshold:
                true_positives += 1
                matched[image_id].add(best_index)
            else:
                false_positives += 1

    false_negatives = total_ground_truths - true_positives
    precision = (true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0)
    recall = (true_positives / total_ground_truths if total_ground_truths > 0 else 0.0)

    return precision, recall, true_positives, false_positives, false_negatives


def calculate_ap(predictions, ground_truths, class_id, iou_threshold):
    """Calculate interpolated Average Precision for one class."""
    total_ground_truths = sum(1 for items in ground_truths.values() for item in items if item["class_id"] == class_id)

    if total_ground_truths == 0:
        return 0.0

    class_predictions = [
        prediction
        for prediction in predictions
        if prediction["class_id"] == class_id
    ]

    class_predictions.sort(
        key=lambda item: item["confidence"], reverse=True)

    matched = defaultdict(set)
    true_positives = []
    false_positives = []

    for prediction in class_predictions:
        image_id = prediction["image_id"]
        candidates = ground_truths[image_id]
        best_iou = 0.0
        best_index = None

        for index, gt in enumerate(candidates):
            if gt["class_id"] != class_id:
                continue

            if index in matched[image_id]:
                continue

            iou = calculate_iou(prediction["bbox"], gt["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            true_positives.append(1)
            false_positives.append(0)
            matched[image_id].add(best_index)
        else:
            true_positives.append(0)
            false_positives.append(1)

    if not true_positives:
        return 0.0

    tp_cumulative = np.cumsum(true_positives)
    fp_cumulative = np.cumsum(false_positives)
    recalls = tp_cumulative / total_ground_truths
    precisions = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)

    # 101-point interpolated AP
    recall_points = np.linspace(0, 1, 101)
    interpolated_precisions = []

    for recall_point in recall_points:
        valid = precisions[recalls >= recall_point]
        interpolated_precisions.append(np.max(valid) if len(valid) else 0.0)

    return float(np.mean(interpolated_precisions))

def main():
    print("=" * 60)
    print("RF-DETR NANO - EVALUACIÓN SOBRE DATASET DE FÚTBOL")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError("No se encuentran las carpetas de imágenes o etiquetas.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    image_paths = sorted(path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if not image_paths:
        raise RuntimeError("No se han encontrado imágenes.")

    print(f"Imágenes de evaluación: {len(image_paths)}")
    print("\nInicializando RF-DETR Nano...")
    model = RFDETRNano()
    print("Optimizando inferencia FP16...")
    model.inference(dtype=torch.float16)
    predictions = []
    ground_truths = {}
    unknown_model_classes = set()

    print("\nEjecutando evaluación...\n")

    for index, image_path in enumerate(image_paths, start=1):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

        height, width = image.shape[:2]
        ground_truths[image_path.name] = load_ground_truth(label_path, width, height)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = model.predict(image_rgb, threshold=CONFIDENCE_THRESHOLD,)

        for bbox, confidence, original_class in zip(detections.xyxy, detections.confidence, detections.class_id,):
            original_class = int(original_class)

            if original_class not in MODEL_CLASS_MAP:
                unknown_model_classes.add(original_class)
                continue

            predictions.append({
                "image_id": image_path.name,
                "class_id": MODEL_CLASS_MAP[original_class],
                "confidence": float(confidence),
                "bbox": np.asarray(bbox, dtype=float),
            })

        print(f"[{index}/{len(image_paths)}] {image_path.name}")
    print("\nCalculando métricas...")

    precision, recall, tp, fp, fn = calculate_precision_recall(predictions, ground_truths, confidence_threshold=REPORT_CONFIDENCE, iou_threshold=0.50,)

    print("\n" + "=" * 60)
    print("RESULTADOS GLOBALES")
    print("=" * 60)
    print(f"Predicciones: {len(predictions)}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Precision @ IoU 0.50, conf 0.30: {precision:.4f}")
    print(f"Recall @ IoU 0.50, conf 0.30:    {recall:.4f}")
    print("\n" + "=" * 60)
    print("AVERAGE PRECISION POR CLASE")
    print("=" * 60)

    all_aps = []

    for class_id, class_name in CLASS_NAMES.items():
        aps = []

        for iou_threshold in IOU_THRESHOLDS:
            ap = calculate_ap(predictions, ground_truths, class_id, float(iou_threshold))
            aps.append(ap)

        ap50 = aps[0]
        ap50_95 = float(np.mean(aps))
        all_aps.append(ap50_95)

        print(f"\n{class_name}:")
        print(f"  AP50:    {ap50:.4f}")
        print(f"  AP50-95: {ap50_95:.4f}")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"mAP50:    {np.mean([calculate_ap(predictions, ground_truths, c, 0.50) for c in CLASS_NAMES]):.4f}")
    print(f"mAP50-95: {np.mean(all_aps):.4f}")

    if unknown_model_classes:
        print("\nIDs de clases del modelo ignorados:",sorted(unknown_model_classes),)

    print("\nNOTA: evaluación agrupada en person y sports ball.")
    print("No representa todavía una evaluación de las cuatro clases.")
    print(f"\nEvaluación completada. Imágenes: {len(image_paths)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
