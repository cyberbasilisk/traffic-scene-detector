"""Tiny config loader so runs are driven by YAML files, not hardcoded values."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
