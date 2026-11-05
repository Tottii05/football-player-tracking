from collections import Counter
from pathlib import Path
import math
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "football_players_detection"
SPLITS = ("train", "valid", "test")

# IDs in the original Roboflow dataset.
CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def validate_dataset():
    """Validate images, labels and YOLO annotations."""

    yaml_path = DATASET_DIR / "data.yaml"

    if not yaml_path.is_file():
        print(f"ERROR: data.yaml not found: {yaml_path}")
        return False

    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        print("ERROR: data.yaml must contain a YAML dictionary.")
        return False

    expected_names = ["ball", "goalkeeper", "player", "referee"]

    if config.get("nc") != len(CLASS_NAMES):
        print("ERROR: Incorrect number of classes in data.yaml.")
        return False

    if config.get("names") != expected_names:
        print("ERROR: Class names or their order do not match.")
        print(f"Expected: {expected_names}")
        print(f"Found: {config.get('names')}")
        return False

    total_images = 0
    total_annotations = 0
    total_empty_labels = 0
    class_counts = Counter()
    errors = []

    for split in SPLITS:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            errors.append(f"{split}: missing images or labels directory.")
            continue

        images = {
            path.stem: path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        labels = {
            path.stem: path
            for path in labels_dir.glob("*.txt")
            if path.is_file()
        }

        missing_labels = set(images) - set(labels)
        missing_images = set(labels) - set(images)

        for stem in sorted(missing_labels):
            errors.append(f"{split}: missing label for image {stem}")

        for stem in sorted(missing_images):
            errors.append(f"{split}: missing image for label {stem}")

        split_annotations = 0
        split_empty = 0

        for stem in sorted(set(images) & set(labels)):
            label_path = labels[stem]

            try:
                content = label_path.read_text(encoding="utf-8").strip()
            except OSError as error:
                errors.append(f"{label_path}: cannot read file: {error}")
                continue

            if not content:
                split_empty += 1
                continue

            for line_number, line in enumerate(content.splitlines(), start=1):
                values = line.split()

                if len(values) != 5:
                    errors.append(f"{label_path.name}:{line_number}: expected 5 values (class x_center y_center width height).")
                    continue

                try:
                    class_value = float(values[0])
                    coordinates = [float(value) for value in values[1:]]
                except ValueError:
                    errors.append(f"{label_path.name}:{line_number}: non-numeric value.")
                    continue

                if (not math.isfinite(class_value) or not class_value.is_integer() or int(class_value) not in CLASS_NAMES):
                    errors.append(f"{label_path.name}:{line_number}: invalid class ID.")
                    continue

                if not all(math.isfinite(value) for value in coordinates):
                    errors.append(f"{label_path.name}:{line_number}: non-finite coordinates.")
                    continue

                x_center, y_center, width, height = coordinates

                if not all(0 <= value <= 1 for value in coordinates):
                    errors.append(f"{label_path.name}:{line_number}: coordinates must be normalized between 0 and 1.")
                    continue

                if width <= 0 or height <= 0:
                    errors.append(f"{label_path.name}:{line_number}: box width and height must be positive.")
                    continue

                class_id = int(class_value)
                class_counts[class_id] += 1
                split_annotations += 1

        total_images += len(images)
        total_annotations += split_annotations
        total_empty_labels += split_empty

        print(
            f"{split:>5}: {len(images):>4} images | "
            f"{len(labels):>4} labels | "
            f"{split_annotations:>5} annotations | "
            f"{split_empty:>3} empty labels"
        )

    print("\nAnnotations by class (original dataset IDs):")

    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name:<12} {class_counts[class_id]}")

    print(f"\nTotal images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Empty label files: {total_empty_labels}")

    if errors:
        print(f"\nVALIDATION FAILED: {len(errors)} issue(s) found.")

        for error in errors[:30]:
            print(f"  - {error}")

        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more issues.")

        return False

    print("\nVALIDATION PASSED: dataset structure and labels are valid.")
    return True


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
