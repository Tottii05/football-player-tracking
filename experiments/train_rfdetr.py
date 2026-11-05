from __future__ import annotations
import argparse
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "football_players_detection"
DATA_YAML = DATASET_DIR / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rfdetr_nano_football"
MODEL_NAME = "RF-DETR Nano"
RESOLUTION = 384
TRAINING_CONFIG = {
    "epochs": 20,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "eval_batch_size": 1,
    "num_workers": 0,
    "amp_dtype": "auto",
    "early_stopping": True,
    "early_stopping_patience": 5,
    "checkpoint_interval": 5,
    "eval_interval": 1,
    "log_per_class_metrics": True,
    "run_test": False,
    "seed": 42,
    "num_sanity_val_steps": 2,
    "tensorboard": True,
    "wandb": False,
    "save_dataset_grids": True,
}

EXPECTED_CLASSES = [
    "ball",
    "goalkeeper",
    "player",
    "referee",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def validate_dataset() -> None:
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"No se encuentra: {DATA_YAML}")

    with DATA_YAML.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("El contenido de data.yaml no es válido.")

    names = data.get("names")

    if names != EXPECTED_CLASSES:
        raise ValueError("El orden de las clases no coincide con el esperado.\n" f"Esperado: {EXPECTED_CLASSES}\n" f"Encontrado: {names}")

    print(f"Dataset: {DATASET_DIR}")
    print(f"Clases: {names}")

    for split in ("train", "val", "test"):
        split_value = data.get(split)

        if not isinstance(split_value, str):
            raise ValueError(f"Falta una ruta válida para el conjunto '{split}' en data.yaml.")

        images_dir = (DATASET_DIR / split_value).resolve()
        labels_dir = (images_dir.parent / "labels").resolve()

        if not images_dir.is_dir():
            raise FileNotFoundError(f"No existe el directorio de imágenes de {split}: {images_dir}")

        if not labels_dir.is_dir():
            raise FileNotFoundError(f"No existe el directorio de etiquetas de {split}: {labels_dir}")

        images = {
            path.stem
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        labels = {
            path.stem
            for path in labels_dir.glob("*.txt")
            if path.is_file()
        }

        missing_labels = images - labels
        extra_labels = labels - images

        if missing_labels or extra_labels:
            raise ValueError(f"Los archivos de {split} no están emparejados.\n" f"Imágenes sin etiqueta: {len(missing_labels)}\n" f"Etiquetas sin imagen: {len(extra_labels)}")

        print(f"{split:>5}: {len(images):>4} imágenes | "f"{len(labels):>4} etiquetas | OK")

    print("\nDataset validado correctamente.")

def train() -> None:
    from rfdetr import RFDETRNano
    print("\n" + "=" * 60)
    print("RF-DETR NANO - FOOTBALL FINE-TUNING")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. No iniciaremos el entrenamiento en CPU.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Resolución: {RESOLUTION}x{RESOLUTION}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiment_info = {
        "model": MODEL_NAME,
        "rfdetr_version": version("rfdetr"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "resolution": RESOLUTION,
        "dataset": str(DATASET_DIR),
        "classes": EXPECTED_CLASSES,
        "training_config": TRAINING_CONFIG,
        "created_at": datetime.now().isoformat(),
    }

    config_path = OUTPUT_DIR / "experiment_config.json"
    config_path.write_text(json.dumps(experiment_info, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"\nConfiguración guardada en: {config_path}")
    print("\nInicializando RF-DETR Nano...")
    model = RFDETRNano()
    print("\nIniciando fine-tuning...\n")
    model.train(dataset_dir=str(DATASET_DIR), dataset_file="roboflow", output_dir=str(OUTPUT_DIR), resolution=RESOLUTION, project="football-player-tracking", run="rfdetr-nano-384", **TRAINING_CONFIG)
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 60)
    print(f"Resultados: {OUTPUT_DIR}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning de RF-DETR Nano para fútbol.")
    parser.add_argument("--check-only", action="store_true", help="Valida el dataset sin iniciar el entrenamiento.")
    args = parser.parse_args()
    validate_dataset()

    if args.check_only:
        print("\nModo de comprobación: no se ha iniciado ningún entrenamiento.")
        return

    train()


if __name__ == "__main__":
    main()
