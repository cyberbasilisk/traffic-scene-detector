"""Helpers for writing YOLO-format label lines: 'class cx cy w h' (all normalised 0-1)."""

from __future__ import annotations


def to_yolo_line(
    class_id: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
) -> str:
    """Convert a pixel-coordinate box (x1, y1, x2, y2) to a YOLO label line.

    The box is clamped to the image first. Raises ValueError for an invalid image size
    or a box with no area after clamping.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image width and height must be positive.")

    x1 = max(0.0, min(float(xmin), img_w))
    y1 = max(0.0, min(float(ymin), img_h))
    x2 = max(0.0, min(float(xmax), img_w))
    y2 = max(0.0, min(float(ymax), img_h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Degenerate box: no area after clamping to the image.")

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def rel_xywh_to_yolo_line(class_id: int, x: float, y: float, w: float, h: float) -> str:
    """Convert a relative top-left box (x, y, w, h in 0-1) to a YOLO label line."""
    if w <= 0 or h <= 0:
        raise ValueError("Box width and height must be positive.")
    cx = x + w / 2
    cy = y + h / 2
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
