import pytest

pytest.importorskip("yaml")

from traffic_detector.classes import CLASS_NAMES  # noqa: E402
from traffic_detector.config import load_yaml  # noqa: E402


def test_data_config_matches_project_classes():
    cfg = load_yaml("configs/data.yaml")
    assert tuple(cfg["names"][i] for i in sorted(cfg["names"])) == CLASS_NAMES
