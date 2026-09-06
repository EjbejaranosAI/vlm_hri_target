"""Prompts del VLM: plantillas (JSON en prompt_templates/) + construcción según modo/social.

Dos categorías, una por archivo — cada una con las variantes con/sin estado
social (y, en "full", la nota anti-sesgo de video) bajo una clave de máximo
dos palabras (``with_social``, ``no_social``, ``motion_hint``):

- ``COMPACT``: mínimo consumo de tokens, para uso normal (recomendado).
- ``FULL``: instrucciones largas y estructuradas por jerarquía (más tokens,
  úsese solo si el modo compacto da resultados pobres para el caso de uso).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import vlm_prompt_mode
from ..detection import sort_dets_left_right
from ..drawing import pid_label
from ..social_state import use_vlm_social_states

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompt_templates"


def _load_category(name: str) -> dict[str, str]:
    """{clave (<=2 palabras): texto de plantilla} de prompt_templates/{name}.json."""
    with open(_PROMPTS_DIR / f"{name}.json", encoding="utf-8") as f:
        raw = json.load(f)
    return {key: entry["template"] for key, entry in raw.items()}


COMPACT = _load_category("compact")
FULL = _load_category("full")

# Snippets pequeños de propósito general (no son "el prompt" en sí, sino
# texto que se antepone/pospone al elegido arriba) — no dependen de
# compact/full, viven como constantes simples en vez de una plantilla más.
NO_PEOPLE = "{}"
MOTION_COLOR_HINT = (
    "Sensor hint: BLUE box = walking detected; RED box = stationary. This hint can be "
    "noisy (e.g. a person just appearing/re-detected can flash BLUE without moving) — "
    "use it as a prior, but only label someone as walking/MOVING if you also see them "
    "actually changing position across the clip.\n"
)


def _vlm_prompt_compact(dets: list[dict], *, video: bool) -> str:
    """Prompt corto con ejemplo JSON por ID (evita respuesta plana {\"a\":...})."""
    pids = sorted({d["pid"] for d in dets})
    keys = ",".join(str(p) for p in pids)
    id_map = ", ".join(f'{pid_label(d["pid"])}→"{d["pid"]}"' for d in dets)
    clip = "1s video" if video else "image"
    if use_vlm_social_states():
        # "a" usa un placeholder (no una palabra real como "stand"/"walk") a
        # propósito: un ejemplo con una acción real de muestra ancla al modelo
        # a repetir ESA palabra sin importar lo que diga el texto de reglas
        # (medido: con "stand"/"walk" de ejemplo, casi todo salía "standing"
        # aunque la persona estuviera comiendo o mirando la cámara). "s" sí
        # puede usar letras reales porque ese es un enum cerrado a propósito.
        example = ",".join(
            f'"{p}":{{"a":"<action>","s":"{"A" if i == 0 else "M"}"}}'
            for i, p in enumerate(pids[:2])
        )
        if len(pids) > 2:
            example += ",..."
        return COMPACT["with_social"].format(
            clip=clip, id_map=id_map, example=example, keys=keys
        )
    example = ",".join(f'"{p}":"<action>"' for p in pids[:2])
    return COMPACT["no_social"].format(clip=clip, id_map=id_map, example=example, keys=keys)


def _vlm_prompt_full(dets: list[dict], *, video: bool) -> str:
    pids = sorted({d["pid"] for d in dets})
    mapping = "\n".join(
        f'- Box label {pid_label(d["pid"])} (person inside that box) → JSON key "{d["pid"]}"'
        for d in dets
    )
    ctx = "this short video clip" if video else "this image"
    motion_hint = FULL["motion_hint"] if video else ""
    keys = ", ".join(str(p) for p in pids)
    if use_vlm_social_states():
        slots = ",".join(
            f'"{p}":{{"action":"<what they do>","social":"<STATE>"}}' for p in pids
        )
        return FULL["with_social"].format(
            ctx=ctx,
            n_people=len(pids),
            motion_hint=motion_hint,
            mapping=mapping,
            slots=slots,
            keys=keys,
        )
    slots = ",".join(f'"{p}":"<action>"' for p in pids)
    return FULL["no_social"].format(
        ctx=ctx,
        n_people=len(pids),
        motion_hint=motion_hint,
        mapping=mapping,
        slots=slots,
        keys=keys,
    )


def vlm_prompt(dets: list[dict], *, video: bool = False) -> str:
    ordered = sort_dets_left_right([d for d in dets if "pid" in d])
    if not ordered:
        return NO_PEOPLE
    if vlm_prompt_mode() == "compact":
        return _vlm_prompt_compact(ordered, video=video)
    return _vlm_prompt_full(ordered, video=video)
