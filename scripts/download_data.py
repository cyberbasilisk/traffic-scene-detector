"""Download a class-filtered subset of Open Images V7 with FiftyOne.

STATUS: skeleton, not yet run. Check the FiftyOne docs for the current API before use,
and verify the label strings against the Open Images class list.

Open Images annotations are CC BY 4.0 and the images are mostly CC BY 2.0.
Record sources and attribution in docs/dataset_card.md.
"""

from __future__ import annotations

import argparse

# Open Images class names to request (see src/traffic_detector/classes.py for the mapping).
OPEN_IMAGES_CLASSES = [
    "Person",
    "Car",
    "Bus",
    "Truck",
    "Van",
    "Motorcycle",
    "Vehicle registration plate",
]


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
        classes=OPEN_IMAGES_CLASSES,
        max_samples=args.max_samples,
    )
    print(f"Downloaded {len(dataset)} samples for split '{args.split}'.")
    # TODO (Day 2): map labels with traffic_detector.classes.map_label, convert boxes with
    # traffic_detector.labels.rel_xywh_to_yolo_line, and write YOLO-format files to
    # data/processed/{images,labels}/<split>.


if __name__ == "__main__":
    main()
