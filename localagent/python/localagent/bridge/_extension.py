from __future__ import annotations

from importlib import import_module
from typing import Any


def load_extension() -> Any | None:
    try:
        return import_module("localagent._rust")
    except ImportError:
        return None
