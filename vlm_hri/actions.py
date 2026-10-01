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
    phone = bool(re.search(r"\bphone\b", low) or re.search(r"\bmobile\b", low))
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
        # "talking ON THE PHONE" no es conversación cara a cara (ENGAGED) --
        # es una llamada, tarea solitaria (BUSY, ver trait "phone"). Excluir
        # el caso "phone" de este trait evita que "talking on phone" entre
        # por la rama ENGAGED (que exige corroboración de otra persona
        # ENGAGED cerca por LiDAR) y acabe cayendo hasta AVAILABLE cuando esa
        # corroboración nunca puede existir para una llamada.
        "talking": bool(
            not phone
            and (
                re.search(r"\btalk", low)
                or re.search(r"\bspeak", low)
                or re.search(r"\bconvers", low)
            )
        ),
        "smiling": bool(re.search(r"\bsmil", low)),
        "phone": phone,
        # Actividad secundaria ligera, interrumpible (AVAILABLE) -- distinta
        # de "phone"/"read"/"work" (BUSY, tarea absorbente que es de mala
        # educación interrumpir).
        "light_secondary": bool(
            re.search(r"\beat", low)
            or re.search(r"\bdrink", low)
            or re.search(r"\bpray", low)
        ),
        # Tarea solitaria absorbente (BUSY) más allá de "phone" -- antes solo
        # cubría read/work/laptop/book/writ/typ (duplicado en social_state.py
        # en dos sitios); se unifica aquí y se amplía con think/study/focus/
        # computer/screen, que el VLM ya puede describir con el vocabulario
        # nuevo del prompt pero que antes de esto no disparaban BUSY -- caían
        # al branch genérico de sitting/standing y se quedaban en AVAILABLE.
        "solo_task": bool(
            re.search(r"\bread", low)
            or re.search(r"\bwork", low)
            or re.search(r"\blaptop\b", low)
            or re.search(r"\bbook\b", low)
            or re.search(r"\bwrit", low)
            or re.search(r"\btyp", low)
            or re.search(r"\bthink", low)
            or re.search(r"\bstud", low)
            or re.search(r"\bfocus", low)
            or re.search(r"\bconcentrat", low)
            or re.search(r"\bcomputer\b", low)
            or re.search(r"\bscreen\b", low)
        ),
        # Interaccion con el entorno (no con otra persona) -- ENGAGED.
        "env_interaction": bool(
            re.search(r"\bphoto", low)
            or re.search(r"\bpictur", low)
            or re.search(r"\bfilm", low)
        ),
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
