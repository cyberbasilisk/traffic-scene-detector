"""Convert a class-filtered Open Images subset into YOLO format under data/processed.

STATUS: not yet run against real data. Check FiftyOne's current docs if anything fails.

Run it again with different --classes or --max-samples to add images to the same output
folder (for example, extra images that contain license plates). Files with the same name
are overwritten with identical content.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_detector.classes import OPEN_IMAGES_DOWNLOAD_CLASSES  # noqa: E402
from traffic_detector.convert import detections_to_yolo_lines  # noqa: E402

# Open Images split name -> folder name used by configs/data.yaml.
SPLIT_DIRS = {"train": "train", "validation": "val", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=sorted(SPLIT_DIRS))
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=OPEN_IMAGES_DOWNLOAD_CLASSES,
        help="Open Images classes to request (images with at least one are loaded)",
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    import fiftyone.zoo as foz  # imported here so --help works without FiftyOne installed

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=args.split,
        label_types=["detections"],
        classes=args.classes,
        max_samples=args.max_samples,
        drop_existing_dataset=True,
    )

    split_dir = SPLIT_DIRS[args.split]
    images_dir = args.out / "images" / split_dir
    labels_dir = args.out / "labels" / split_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    for sample in dataset:
        detections = sample.ground_truth.detections if sample.ground_truth else []
        lines = detections_to_yolo_lines(detections, stats)
        if not lines:
            stats["images_skipped_no_labels"] += 1
            continue

        source = Path(sample.filepath)
        shutil.copy2(source, images_dir / source.name)
        (labels_dir / f"{source.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["images_written"] += 1

    print(f"Output: {args.out} ({split_dir})")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
