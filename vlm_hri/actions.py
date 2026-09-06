"""Normalización y clasificación de texto de acción (vocabulario libre del VLM).

Sin dependencias internas — módulo base usado por social_state.py, motion.py
y vlm/parsing.py.
"""

from __future__ import annotations

import re


def _is_id_label(text: str) -> bool:
    t = str(text).strip().upper()
    return t == "ID" or bool(re.match(r"^ID\d+$", t))


def _is_sitting_phrase(low: str) -> bool:
    """Sentado real (evita falsos positivos tipo 'sit' dentro de otras palabras)."""
    return bool(
        re.search(r"\bsitting\b", low)
        or re.search(r"\bseated\b", low)
        or re.search(r"\bsit\b", low)
        or re.search(r"\bsits\b", low)
        or re.search(r"\bon\s+(?:a\s+)?(?:chair|bench|sofa|couch|seat)\b", low)
    )


def _action_traits(low: str) -> dict[str, bool]:
    return {
        "sitting": _is_sitting_phrase(low),
        "standing": bool(
            re.search(r"\bstanding\b", low)
            or re.search(r"\bstood\b", low)
            or re.search(r"\bstand(?:ing|s)?\b", low)
        ),
        "walking": bool(
            re.search(r"\bwalking\b", low)
            or re.search(r"\bwalk(?:ing|s|ed)?\b", low)
        ),
        "running": bool(re.search(r"\brunning\b", low) or re.search(r"\bjogg", low)),
        "talking": bool(
            re.search(r"\btalk", low)
            or re.search(r"\bspeak", low)
            or re.search(r"\bconvers", low)
        ),
        "smiling": bool(re.search(r"\bsmil", low)),
        "phone": bool(re.search(r"\bphone\b", low) or re.search(r"\bmobile\b", low)),
    }


def _action_secondary_parts(tr: dict[str, bool]) -> list[str]:
    parts: list[str] = []
    for label, key in (
        ("talking", "talking"),
        ("smiling", "smiling"),
        ("using phone", "phone"),
    ):
        if tr[key]:
            parts.append(label)
    return parts


_SINGLE_WORD_ACTION_CANON: dict[str, str] = {
    "sit": "sitting", "sits": "sitting", "sitting": "sitting", "seated": "sitting",
    "stand": "standing", "stands": "standing", "stood": "standing", "standing": "standing",
    "walk": "walking", "walks": "walking", "walked": "walking", "walking": "walking",
    "run": "running", "runs": "running", "running": "running",
    "talk": "talking", "talks": "talking", "talking": "talking",
}


def normalize_action(text: str) -> str:
    """Limpia espacios/puntuación y canonicaliza variantes de tense/plural de
    una sola palabra (p. ej. 'stood' -> 'standing'). Cualquier frase más
    descriptiva se deja TAL CUAL (p. ej. 'sit and eat', 'eating a sandwich',
    'sitting and reading') — antes se reconstruía siempre desde un set
    cerrado de palabras clave (walk/run/sit/stand/talk/smile/phone) y
    cualquier actividad fuera de ese set se perdía en silencio (medido:
    "eating" quedaba reducido a solo "standing")."""
    t = " ".join(str(text).strip().split())
    if not t:
        return "unknown"
    t = t.rstrip(".,;:")[:80]
    if _is_id_label(t):
        return "unknown"
    low = t.lower()
    if low in _SINGLE_WORD_ACTION_CANON:
        return _SINGLE_WORD_ACTION_CANON[low]
    return t


def prioritize_movement_action(action: str) -> str:
    """Para navegación/HRI, si la persona se está desplazando esa señal debe
    dominar el estado social (MOVING) sobre cualquier otra acción secundaria
    del mismo trozo: al robot le importa primero si hay alguien en movimiento
    delante. La intención secundaria (p. ej. hablar) NO se descarta —se
    mantiene anexada en la etiqueta (p.ej. "walking and talking")— por si es
    relevante para decidir a quién puede acercarse el robot después."""
    if not action:
        return action
    tr = _action_traits(action.lower())
    base = "running" if tr.get("running") else "walking" if tr.get("walking") else None
    if base is None:
        return action
    extras = _action_secondary_parts(tr)
    return " and ".join([base] + extras) if extras else base
