"""Carga las plantillas de prompt del VLM desde prompts/*.json — el texto
vive como dato editable fuera del código (una plantilla por archivo). Este
módulo solo expone las mismas constantes que antes (``COMPACT_WITH_SOCIAL``,
etc.) para que ``pipeline.py``/``pose_pipeline.py`` no necesiten cambios.

Cada plantilla se rellena con ``str.format(**kwargs)`` en tiempo de ejecución
con los datos dinámicos de cada petición (mapeo de IDs detectados, claves JSON
requeridas, ejemplo, etc.). Las llaves dobles ``{{`` / ``}}`` dentro de la
plantilla son llaves literales de JSON (se convierten en ``{`` / ``}`` al
formatear); las llaves simples como ``{clip}`` son los marcadores que
``pipeline.py`` rellena.
"""

from __future__ import annotations

import json
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load(name: str) -> str:
    with open(_PROMPTS_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)["template"]


# —— Modo compacto (mínimo consumo de tokens + alta precisión) ——————————
COMPACT_WITH_SOCIAL = _load("compact_with_social")
COMPACT_NO_SOCIAL = _load("compact_no_social")

# —— Modo completo (instrucciones estructuradas por jerarquía) ——————————
FULL_WITH_SOCIAL = _load("full_with_social")
FULL_NO_SOCIAL = _load("full_no_social")

NO_PEOPLE = _load("no_people")
MOTION_HINT_VIDEO = _load("motion_hint_video")
MOTION_COLOR_HINT = _load("motion_color_hint")
