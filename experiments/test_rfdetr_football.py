from pathlib import Path
import cv2
import torch
from rfdetr import RFDETRNano
import supervision as sv

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "football_players_detection"
    / "valid"
    / "images"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rfdetr_preview"

# Configuración
CONFIDENCE_THRESHOLD = 0.3
MAX_IMAGES = 12

def main():
    print("=" * 55)
    print("RF-DETR NANO - DETECCIÓN EN FÚTBOL REAL")
    print("=" * 55)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"No se encuentra el directorio: {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    image_paths = image_paths[:MAX_IMAGES]

    if not image_paths:
        raise RuntimeError("No se han encontrado imágenes.")

    print(f"\nImágenes seleccionadas: {len(image_paths)}")
    print("\nInicializando RF-DETR Nano...")
    model = RFDETRNano()
    print("Optimizando modelo con FP16...")
    model.inference(dtype=torch.float16)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print("\nProcesando imágenes...\n")

    for index, image_path in enumerate(image_paths, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[ERROR] No se pudo leer: {image_path.name}")
            continue

        # RF-DETR recibe la imagen en formato RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = model.predict(image_rgb, threshold=CONFIDENCE_THRESHOLD)

        labels = [
            f"{int(class_id)}: {confidence:.2f}"
            for class_id, confidence in zip(
                detections.class_id,
                detections.confidence
            )
        ]

        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        output_path = OUTPUT_DIR / f"{image_path.stem}_pred.jpg"
        cv2.imwrite(str(output_path), annotated_image)

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}: "
            f"{len(detections)} detecciones"
        )

    print("\n" + "=" * 55)
    print("EXPERIMENTO COMPLETADO")
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print("=" * 55)

if __name__ == "__main__":
    main()
