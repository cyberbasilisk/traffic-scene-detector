"""Project classes and the mapping from source-dataset labels to them."""

from __future__ import annotations

# Project classes, in YOLO class-id order. Keep in sync with configs/data.yaml.
CLASS_NAMES: tuple[str, ...] = ("person", "vehicle", "license_plate")
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Open Images label (lower-case) -> project class.
# Label names checked against a 50-image validation sample.
# Open Images has separate labels for "Man", "Woman", "Boy" and "Girl"; they all map to person
# and overlapping duplicates are removed later (see labels.dedupe_boxes).
OPEN_IMAGES_TO_PROJECT: dict[str, str] = {
    "person": "person",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "van": "vehicle",
    "motorcycle": "vehicle",
    "vehicle registration plate": "license_plate",
}

# Generic parent labels that can include boats, aircraft and so on.
# Excluded (and counted separately) until they have been reviewed visually.
AMBIGUOUS_LABELS: frozenset[str] = frozenset({"vehicle", "land vehicle"})

# Open Images class names to request when downloading (exact capitalisation).
OPEN_IMAGES_DOWNLOAD_CLASSES: list[str] = [
    "Person",
    "Man",
    "Woman",
    "Boy",
    "Girl",
    "Car",
    "Bus",
    "Truck",
    "Van",
    "Motorcycle",
    "Vehicle registration plate",
]


def map_label(source_label: str) -> int | None:
    """Return the project class id for a source label, or None to drop the box."""
    project_name = OPEN_IMAGES_TO_PROJECT.get(source_label.strip().lower())
    return CLASS_IDS[project_name] if project_name is not None else None
