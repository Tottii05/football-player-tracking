
from __future__ import annotations

from pathlib import Path

import cv2
from rfdetr import RFDETRNano


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "football_players_detection"
    / "test"
    / "images"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "rfdetr_nano_football"
    / "checkpoint_best_ema.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rfdetr_test_preview"

CONFIDENCE_THRESHOLD = 0.30
RESOLUTION = 384

CLASS_COLORS = {
    "ball": (0, 255, 255),
    "goalkeeper": (255, 0, 255),
    "player": (0, 200, 0),
    "referee": (255, 100, 0),
}


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"No existe el directorio: {IMAGE_DIR}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {CHECKPOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        raise RuntimeError("No se han encontrado imágenes de test.")

    print("Cargando RF-DETR Nano...")
    model = RFDETRNano.from_checkpoint(CHECKPOINT)

    print(f"Clases: {model.class_names}")
    print(f"Imágenes: {len(image_paths)}")
    print(f"Umbral de confianza: {CONFIDENCE_THRESHOLD}")

    total_detections = 0

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"No se pudo leer: {image_path.name}")
            continue

        height, width = frame.shape[:2]

        # RF-DETR recibe imágenes en formato RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = model.predict(
            image_rgb,
            threshold=CONFIDENCE_THRESHOLD,
        )

        image_detections = 0

        for bbox, confidence, class_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
        ):
            class_id = int(class_id)

            if class_id < 0 or class_id >= len(model.class_names):
                continue

            class_name = model.class_names[class_id]
            color = CLASS_COLORS.get(class_name, (255, 255, 255))

            x1, y1, x2, y2 = map(int, bbox)

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = f"{class_name}: {float(confidence):.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 7, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            image_detections += 1

        total_detections += image_detections

        output_path = OUTPUT_DIR / image_path.name

        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"No se pudo guardar: {output_path}")

        print(
            f"{image_path.name}: "
            f"{image_detections} detecciones -> {output_path.name}"
        )

    print("\n" + "=" * 50)
    print("VISUALIZACIÓN COMPLETADA")
    print("=" * 50)
    print(f"Imágenes procesadas: {len(image_paths)}")
    print(f"Detecciones totales: {total_detections}")
    print(f"Resultados: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
