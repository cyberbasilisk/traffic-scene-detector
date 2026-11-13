from collections import Counter
from types import SimpleNamespace

from traffic_detector.convert import detections_to_yolo_lines


def det(label, box=(0.1, 0.1, 0.2, 0.2), **attrs):
    return SimpleNamespace(label=label, bounding_box=list(box), **attrs)


def test_maps_known_labels_and_drops_unmapped():
    stats = Counter()
    lines = detections_to_yolo_lines([det("Car"), det("Trousers", (0.5, 0.5, 0.1, 0.1))], stats)
    assert len(lines) == 1
    assert lines[0].startswith("1 ")
    assert stats["dropped_unmapped"] == 1
    assert stats["kept_vehicle"] == 1


def test_group_of_boxes_are_skipped():
    stats = Counter()
    assert detections_to_yolo_lines([det("Person", IsGroupOf=True)], stats) == []
    assert stats["dropped_IsGroupOf"] == 1


def test_depictions_are_skipped():
    stats = Counter()
    assert detections_to_yolo_lines([det("Person", IsDepiction=True)], stats) == []
    assert stats["dropped_IsDepiction"] == 1


def test_ambiguous_labels_are_counted_separately():
    stats = Counter()
    assert detections_to_yolo_lines([det("Land vehicle")], stats) == []
    assert stats["dropped_ambiguous"] == 1


def test_person_and_man_on_same_object_become_one_box():
    stats = Counter()
    box = (0.1, 0.1, 0.3, 0.5)
    lines = detections_to_yolo_lines([det("Person", box), det("Man", box)], stats)
    assert len(lines) == 1
    assert lines[0].startswith("0 ")
    assert stats["dropped_duplicate"] == 1


def test_box_is_clamped_to_image():
    lines = detections_to_yolo_lines([det("Car", (0.9, 0.9, 0.3, 0.3))])
    assert lines == ["1 0.950000 0.950000 0.100000 0.100000"]


def test_box_outside_image_is_dropped_as_degenerate():
    stats = Counter()
    assert detections_to_yolo_lines([det("Car", (1.2, 0.1, 0.1, 0.1))], stats) == []
    assert stats["dropped_degenerate"] == 1


def test_missing_attributes_default_to_false():
    assert len(detections_to_yolo_lines([det("Car")])) == 1
