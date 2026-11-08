from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from football_tracking.pipeline.runner import FootballTrackingPipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Football player detection and tracking pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input video.")
    parser.add_argument("--output", type=Path, default=None, help="Path to the output video. Optional.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml", help="Path to the YAML configuration file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "rfdetr_nano_football"
            / "checkpoint_best_ema.pth"
        ), help="Path to the trained RF-DETR checkpoint.")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        pipeline = FootballTrackingPipeline(config_path=args.config, checkpoint_path=args.checkpoint)
        results = pipeline.run(input_path=args.input, output_path=args.output)
        print("\nExecution summary")
        print("-----------------")

        for key, value in results.items():
            print(f"{key}: {value}")

    except Exception as error:
        print(f"\nPipeline failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
