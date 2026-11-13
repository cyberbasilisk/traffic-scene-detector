from traffic_detector.classes import (
    AMBIGUOUS_LABELS,
    CLASS_IDS,
    CLASS_NAMES,
    OPEN_IMAGES_DOWNLOAD_CLASSES,
    map_label,
)


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


def test_gendered_person_labels_map_to_person():
    for label in ("Man", "Woman", "Boy", "Girl"):
        assert map_label(label) == 0


def test_ambiguous_labels_are_not_mapped():
    assert map_label("Vehicle") is None
    assert map_label("Land vehicle") is None
    assert {"vehicle", "land vehicle"} == set(AMBIGUOUS_LABELS)


def test_download_classes_all_map_to_project_classes():
    for label in OPEN_IMAGES_DOWNLOAD_CLASSES:
        assert map_label(label) is not None
