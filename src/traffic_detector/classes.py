"""Project classes and the mapping from source-dataset labels to them."""

from __future__ import annotations

# Project classes, in YOLO class-id order. Keep in sync with configs/data.yaml.
CLASS_NAMES: tuple[str, ...] = ("person", "vehicle", "license_plate")
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Open Images label (lower-case) -> project class.
# Several vehicle types are merged into one class for Part 1.
# TODO: verify these strings against the Open Images V7 class list.
OPEN_IMAGES_TO_PROJECT: dict[str, str] = {
    "person": "person",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "van": "vehicle",
    "motorcycle": "vehicle",
    "vehicle registration plate": "license_plate",
}


def map_label(source_label: str) -> int | None:
    """Return the project class id for a source label, or None to drop the box."""
    project_name = OPEN_IMAGES_TO_PROJECT.get(source_label.strip().lower())
    return CLASS_IDS[project_name] if project_name is not None else None
