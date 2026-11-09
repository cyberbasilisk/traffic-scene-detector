from traffic_detector.classes import CLASS_IDS, CLASS_NAMES, map_label


def test_class_order():
    assert CLASS_NAMES == ("person", "vehicle", "license_plate")
    assert CLASS_IDS["license_plate"] == 2


def test_known_labels_map_to_project_classes():
    assert map_label("Person") == 0
    assert map_label("Car") == 1
    assert map_label("Truck") == 1
    assert map_label("Vehicle registration plate") == 2


def test_unknown_label_is_dropped():
    assert map_label("Tree") is None


def test_label_whitespace_and_case_are_ignored():
    assert map_label("  CAR ") == 1
