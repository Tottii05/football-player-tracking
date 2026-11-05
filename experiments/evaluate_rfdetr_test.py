from __future__ import annotations
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import torch
from rfdetr import RFDETRNano

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "football_players_detection"
CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "rfdetr_nano_football"
    / "checkpoint_best_ema.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rfdetr_test_evaluation"
RESOLUTION = 384

def main() -> None:
    print("=" * 60)
    print("RF-DETR NANO - TEST EVALUATION")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. Comprueba tu entorno.")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"No se encuentra el checkpoint: {CHECKPOINT}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No se encuentra el dataset: {DATASET_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT.name}")
    print(f"Resolución: {RESOLUTION}")
    print("\nCargando modelo...")

    model = RFDETRNano.from_checkpoint(CHECKPOINT)

    print(f"Modelo: {type(model).__name__}")
    print(f"Clases: {model.class_names}")
    print("\nEvaluando conjunto TEST...")
    print("Este proceso puede tardar unos minutos.\n")

    metrics = model.evaluate(split="test", dataset_dir=str(DATASET_DIR), device="cuda", resolution=RESOLUTION, batch_size=1, eval_batch_size=1, num_workers=0, log_per_class_metrics=True, output_dir=str(OUTPUT_DIR))

    results = {
        "experiment": "RF-DETR Nano fine-tuned",
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_DIR.relative_to(PROJECT_ROOT)),
        "split": "test",
        "resolution": RESOLUTION,
        "model": type(model).__name__,
        "classes": model.class_names,
        "rfdetr_version": version("rfdetr"),
        "evaluation_date": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "metrics": metrics,
    }

    results_path = OUTPUT_DIR / "test_metrics.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RESULTADOS TEST")
    print("=" * 60)

    for name, value in sorted(metrics.items()):
        print(f"{name:<35} {value:.4f}")

    print(f"\nResultados guardados en:\n{results_path}")
    print("\nEvaluación completada correctamente.")


if __name__ == "__main__":
    main()
