# YOLOv5 compatibility wrapper for local utils package.
# Expose common symbols imported by older YOLOv5 modules that expect
# `from utils import TryExcept, emojis, threaded, notebook_init`.

from __future__ import annotations

try:
    from ultralytics.utils import TryExcept, emojis, threaded
except ImportError as exc:
    raise ImportError(
        "Could not import TryExcept/emojis/threaded from ultralytics.utils. "
        "Install the ultralytics package in the current environment."
    ) from exc

__all__ = ["TryExcept", "emojis", "threaded", "notebook_init"]


def notebook_init():
    """Compatibility no-op for older yolov5 benchmarks."""
    return
