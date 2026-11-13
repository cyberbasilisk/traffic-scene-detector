"""Convert Open Images detections into YOLO label lines for the project's classes."""

from __future__ import annotations

from collections import Counter

from .classes import AMBIGUOUS_LABELS, CLASS_NAMES, map_label
from .labels import dedupe_boxes, rel_xywh_to_yolo_line

# Open Images attributes that mark boxes we do not want to train on.
# IsGroupOf: one box covering many objects. IsDepiction: drawings, paintings, statues.
SKIP_ATTRIBUTES = ("IsGroupOf", "IsDepiction")


def detections_to_yolo_lines(detections, stats: Counter | None = None) -> list[str]:
    """Convert one image's detections to YOLO label lines.

    ``detections`` is any iterable of objects with ``label`` and ``bounding_box``
    (relative [x, y, w, h], top-left origin) and optional Open Images attributes.
    Counts of what was kept or dropped are added to ``stats``.
    """
    if stats is None:
        stats = Counter()

    boxes = []
    for det in detections:
        class_id = map_label(det.label)
        if class_id is None:
            reason = "ambiguous" if det.label.strip().lower() in AMBIGUOUS_LABELS else "unmapped"
            stats[f"dropped_{reason}"] += 1
            continue

        skipped = [attr for attr in SKIP_ATTRIBUTES if getattr(det, attr, False)]
        if skipped:
            stats[f"dropped_{skipped[0]}"] += 1
            continue

        # Clamp to the image; Open Images boxes can stray slightly outside 0-1.
        x, y, w, h = det.bounding_box
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(1.0, x + w), min(1.0, y + h)
        if x1 <= x0 or y1 <= y0:
            stats["dropped_degenerate"] += 1
            continue
        boxes.append((class_id, x0, y0, x1 - x0, y1 - y0))

    kept = dedupe_boxes(boxes)
    stats["dropped_duplicate"] += len(boxes) - len(kept)
    for class_id, *_ in kept:
        stats[f"kept_{CLASS_NAMES[class_id]}"] += 1
    return [rel_xywh_to_yolo_line(*box) for box in kept]
