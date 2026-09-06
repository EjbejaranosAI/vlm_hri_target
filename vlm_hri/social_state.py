"""Estados sociales (capa robot): constantes, normalización y reconciliación con la acción del VLM."""

from __future__ import annotations

import os

from .actions import _action_traits, normalize_action


SOCIAL_ATTENTIVE = "ATTENTIVE"


SOCIAL_AVAILABLE = "AVAILABLE"


SOCIAL_BUSY = "BUSY"


SOCIAL_ENGAGED = "ENGAGED"


SOCIAL_MOVING = "MOVING"


SOCIAL_UNKNOWN = "UNKNOWN"


_ACTION_TO_SOCIAL: dict[str, str] = {
    "standing": SOCIAL_AVAILABLE,
    "sitting": SOCIAL_AVAILABLE,
    "idle": SOCIAL_AVAILABLE,
    "looking": SOCIAL_ATTENTIVE,
    "greeting": SOCIAL_ATTENTIVE,
    "waving": SOCIAL_ATTENTIVE,
    "talking": SOCIAL_ENGAGED,
    "phone": SOCIAL_BUSY,
    "reading": SOCIAL_BUSY,
    "working": SOCIAL_BUSY,
    "blocking": SOCIAL_ENGAGED,
    "walking": SOCIAL_MOVING,
    "running": SOCIAL_MOVING,
}


_SOCIAL_SUITABILITY: dict[str, str] = {
    SOCIAL_AVAILABLE: "High",
    SOCIAL_ATTENTIVE: "Very High",
    SOCIAL_BUSY: "Low",
    SOCIAL_ENGAGED: "Low",
    SOCIAL_MOVING: "Low",
    SOCIAL_UNKNOWN: "?",
}


_SOCIAL_ALIASES: dict[str, str] = {
    "available": SOCIAL_AVAILABLE,
    "free": SOCIAL_AVAILABLE,
    "idle": SOCIAL_AVAILABLE,
    "interruptible": SOCIAL_AVAILABLE,
    "attentive": SOCIAL_ATTENTIVE,
    "aware": SOCIAL_ATTENTIVE,
    "receptive": SOCIAL_ATTENTIVE,
    "busy": SOCIAL_BUSY,
    "occupied": SOCIAL_BUSY,
    "phone": SOCIAL_BUSY,
    "reading": SOCIAL_BUSY,
    "engaged": SOCIAL_ENGAGED,
    "conversation": SOCIAL_ENGAGED,
    "moving": SOCIAL_MOVING,
    "motion": SOCIAL_MOVING,
    "locomotion": SOCIAL_MOVING,
}


def social_state_mode() -> str:
    m = os.environ.get("SOCIAL_STATE_MODE", "vlm").strip().lower()
    return m if m in ("vlm", "map") else "vlm"


def social_vlm_fallback_map() -> bool:
    return os.environ.get("SOCIAL_VLM_FALLBACK_MAP", "1") == "1"


def social_reconcile_action() -> bool:
    """Si 1, corrige social incoherente con la acción (p. ej. walking+ENGAGED → MOVING)."""
    return os.environ.get("SOCIAL_RECONCILE_ACTION", "1") == "1"


def use_vlm_social_states() -> bool:
    return social_state_mode() == "vlm"


def _is_social_vocabulary(word: str) -> bool:
    """True si la palabra es un estado social, no una acción observable."""
    low = word.lower().strip()
    if low in ("available", "attentive", "engaged", "moving"):
        return True
    return low in _SOCIAL_ALIASES


def coerce_action_social(action: str, social: str) -> tuple[str, str]:
    """
    Si el VLM puso ENGAGED/ATTENTIVE/… en "action", moverlo a social y dejar action=unknown.
    No toca acciones reales (walking, talking, …).
    """
    act = normalize_action(action)
    soc = social
    if act != "unknown" and _is_social_vocabulary(act):
        inferred = normalize_social_state(act)
        if soc == SOCIAL_UNKNOWN and inferred != SOCIAL_UNKNOWN:
            soc = inferred
        act = "unknown"
    return act, soc


def action_to_social_state(action: str) -> str:
    """Mapea acción (simple o compuesta) → estado de interacción."""
    act = normalize_action(action)
    if not act or act == "unknown":
        return SOCIAL_UNKNOWN
    if act in _ACTION_TO_SOCIAL:
        return _ACTION_TO_SOCIAL[act]
    low = act.lower()
    tr = _action_traits(low)
    if tr["walking"] or tr["running"]:
        return SOCIAL_MOVING
    if tr["talking"]:
        return SOCIAL_ENGAGED
    if tr["phone"]:
        return SOCIAL_BUSY
    if tr["smiling"]:
        return SOCIAL_ATTENTIVE
    if tr["sitting"] and not tr["standing"]:
        return SOCIAL_AVAILABLE
    if tr["standing"]:
        return SOCIAL_AVAILABLE
    if any(k in low for k in ("read", "work", "laptop", "book")):
        return SOCIAL_BUSY
    if any(k in low for k in ("greet", "wave", "look", "watch", "eye")):
        return SOCIAL_ATTENTIVE
    return SOCIAL_UNKNOWN


def reconcile_social_with_action(action: str, social: str) -> str:
    """Alinea estado social con acción simple o compuesta."""
    act = normalize_action(action)
    if act == "unknown":
        return social if social != SOCIAL_UNKNOWN else SOCIAL_UNKNOWN
    low = act.lower()
    tr = _action_traits(low)
    if tr["walking"] or tr["running"]:
        # ENGAGED gana sobre MOVING si el VLM ya juzgó (en su campo 's',
        # independiente del texto de 'a') que hay una interacción real en
        # curso — caminar mientras se conversa/saluda a alguien es
        # interacción, no solo tránsito.
        if social == SOCIAL_ENGAGED:
            return SOCIAL_ENGAGED
        return SOCIAL_MOVING
    if tr["talking"]:
        return SOCIAL_ENGAGED
    if tr["phone"]:
        return SOCIAL_BUSY
    if tr["smiling"]:
        return SOCIAL_ATTENTIVE
    if tr["sitting"] and not tr["standing"]:
        # Igual que la rama "standing" de abajo: sentado sin una actividad
        # reconocida (talking/phone/smiling ya se descartaron arriba) no debe
        # pisar un estado social válido que el VLM ya haya dado (p. ej. BUSY
        # para "sit and eat") — antes esto forzaba AVAILABLE siempre, perdiendo
        # cualquier actividad fuera del set walk/talk/phone/smile.
        if social in (SOCIAL_ENGAGED, SOCIAL_ATTENTIVE, SOCIAL_BUSY):
            return social
        return social if social != SOCIAL_UNKNOWN else SOCIAL_AVAILABLE
    if tr["standing"]:
        # talking/smiling ya se descartaron arriba (si alguno fuera True, ya
        # habríamos retornado) — no hace falta repetir esos dos checks aquí.
        if social == SOCIAL_MOVING:
            return SOCIAL_AVAILABLE
        if social in (SOCIAL_ENGAGED, SOCIAL_ATTENTIVE, SOCIAL_BUSY):
            return social
        return social if social != SOCIAL_UNKNOWN else SOCIAL_AVAILABLE
    derived = action_to_social_state(act)
    if derived != SOCIAL_UNKNOWN:
        return derived
    return social if social != SOCIAL_UNKNOWN else SOCIAL_UNKNOWN


def social_interaction_hint(state: str) -> str:
    return _SOCIAL_SUITABILITY.get(state, "?")


def social_states_from_actions(actions: dict[int, str]) -> dict[int, str]:
    return {pid: action_to_social_state(act) for pid, act in actions.items()}


_COMPACT_SOCIAL: dict[str, str] = {
    "A": SOCIAL_AVAILABLE,
    "T": SOCIAL_ATTENTIVE,
    "B": SOCIAL_BUSY,
    "E": SOCIAL_ENGAGED,
    "M": SOCIAL_MOVING,
}


def normalize_social_state(text: str) -> str:
    """ATTENTIVE | AVAILABLE | BUSY | ENGAGED | MOVING o abreviatura A|T|B|E|M."""
    t = " ".join(str(text).strip().split()).upper().rstrip(".,;:")
    if not t:
        return SOCIAL_UNKNOWN
    if t in _COMPACT_SOCIAL:
        return _COMPACT_SOCIAL[t]
    if t in (
        SOCIAL_AVAILABLE,
        SOCIAL_ATTENTIVE,
        SOCIAL_BUSY,
        SOCIAL_ENGAGED,
        SOCIAL_MOVING,
    ):
        return t
    low = t.lower().replace("-", " ").replace("_", " ")
    if low in _SOCIAL_ALIASES:
        return _SOCIAL_ALIASES[low]
    for word in low.split():
        if word in _SOCIAL_ALIASES:
            return _SOCIAL_ALIASES[word]
    return SOCIAL_UNKNOWN


def _never_unknown(state: str) -> str:
    """El robot necesita una de las 5 categorías reales para decidir a quién
    acercarse, no un "no sé": si ninguna señal permite clasificar a alguien,
    se asume AVAILABLE -- el estado más neutro (no afirma que esté atento,
    ocupado, en conversación ni en movimiento)."""
    return state if state != SOCIAL_UNKNOWN else SOCIAL_AVAILABLE


def resolve_social_states(
    actions: dict[int, str],
    parsed_social: dict[int, str] | None,
    person_ids: list[int] | None = None,
) -> dict[int, str]:
    """vlm: usa parsed_social; map: deriva de acciones. Nunca devuelve
    SOCIAL_UNKNOWN (ver _never_unknown)."""
    pids = person_ids if person_ids is not None else list(actions.keys())
    if social_state_mode() == "map":
        base = social_states_from_actions(actions)
        return {pid: _never_unknown(base.get(pid, SOCIAL_UNKNOWN)) for pid in pids}
    parsed = parsed_social or {}
    out: dict[int, str] = {}
    for pid in pids:
        act = actions.get(pid, "unknown")
        s = normalize_social_state(parsed.get(pid, ""))
        if s == SOCIAL_UNKNOWN and social_vlm_fallback_map():
            s = action_to_social_state(act)
        if social_reconcile_action():
            s = reconcile_social_with_action(act, s)
        out[pid] = _never_unknown(s)
    return out


def merge_social_votes(votes: list[dict[int, str]]) -> dict[int, str]:
    from collections import Counter

    if not votes:
        return {}
    all_pids = set().union(*(v.keys() for v in votes))
    out: dict[int, str] = {}
    for pid in all_pids:
        c = Counter(
            v[pid]
            for v in votes
            if pid in v and v[pid] and v[pid] != SOCIAL_UNKNOWN
        )
        out[pid] = c.most_common(1)[0][0] if c else SOCIAL_UNKNOWN
    return out
