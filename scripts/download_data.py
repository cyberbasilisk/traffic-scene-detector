"""Download a class-filtered subset of Open Images V7 with FiftyOne.

Quick check that the download works. Use scripts/prepare_data.py to build the YOLO dataset.

Open Images annotations are CC BY 4.0 and the images are mostly CC BY 2.0.
Record sources and attribution in docs/dataset_card.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_detector.classes import OPEN_IMAGES_DOWNLOAD_CLASSES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    import fiftyone.zoo as foz  # imported here so --help works without FiftyOne installed

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=args.split,
        label_types=["detections"],
        classes=OPEN_IMAGES_DOWNLOAD_CLASSES,
        max_samples=args.max_samples,
    )
    print(f"Downloaded {len(dataset)} samples for split '{args.split}'.")


if __name__ == "__main__":
    main()
