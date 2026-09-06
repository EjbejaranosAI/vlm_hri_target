"""Detección de personas + VLM (Qwen2-VL) para acción/estado social en HRI."""

from __future__ import annotations

# Debe importarse antes que cualquier otro submódulo: fija los límites de
# hilos de CPU antes de que algo más importe torch/cv2 (ver config.py).
from . import config as config  # noqa: F401,E402
