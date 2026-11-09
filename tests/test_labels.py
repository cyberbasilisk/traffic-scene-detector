import pytest

from traffic_detector.labels import rel_xywh_to_yolo_line, to_yolo_line


def test_pixel_box_to_yolo():
    assert to_yolo_line(1, 100, 50, 300, 150, 400, 200) == "1 0.500000 0.500000 0.500000 0.500000"


def test_box_is_clamped_to_image():
    assert to_yolo_line(0, -10, -10, 50, 50, 100, 100) == "0 0.250000 0.250000 0.500000 0.500000"


def test_degenerate_box_raises():
    with pytest.raises(ValueError):
        to_yolo_line(0, 10, 10, 10, 20, 100, 100)


def test_box_fully_outside_image_raises():
    with pytest.raises(ValueError):
        to_yolo_line(0, 150, 150, 200, 200, 100, 100)


def test_bad_image_size_raises():
    with pytest.raises(ValueError):
        to_yolo_line(0, 0, 0, 10, 10, 0, 100)


def test_relative_box_to_yolo():
    assert rel_xywh_to_yolo_line(2, 0.25, 0.25, 0.5, 0.5) == "2 0.500000 0.500000 0.500000 0.500000"


def test_relative_box_bad_size_raises():
    with pytest.raises(ValueError):
        rel_xywh_to_yolo_line(0, 0.1, 0.1, 0.0, 0.2)
