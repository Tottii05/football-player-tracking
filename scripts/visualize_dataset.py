
"""Visualize YOLO annotations on football dataset images."""

from pathlib import Path
import random

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "football_players_detection"

SPLIT = "train"
NUM_IMAGES = 12
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee",
}

COLORS = {
    0: (0, 255, 255),
    1: (255, 0, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_preview"


def draw_annotations(image, label_path):
    """Draw YOLO bounding boxes on an image."""

    height, width = image.shape[:2]

    if not label_path.exists():
        return image

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    for line in lines:
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(float(values[0]))
        x_center, y_center, box_width, box_height = map(float, values[1:])

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        color = COLORS[class_id]
        class_name = CLASS_NAMES[class_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            image,
            class_name,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image


def main():
    images_dir = DATASET_DIR / SPLIT / "images"
    labels_dir = DATASET_DIR / SPLIT / "labels"

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    random.seed(RANDOM_SEED)
    selected_images = random.sample(
        image_paths,
        min(NUM_IMAGES, len(image_paths)),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in selected_images:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"

        image = draw_annotations(image, label_path)

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), image)

        print(f"Saved: {output_path}")

    print(f"\nPreview generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
