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

from ..config import vlm_prompt_mode, vlm_target_source
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
    "Each person's box and skeleton is colored by a noisy motion sensor: BLUE = walking "
    "detected, RED = still. Use MOVING only if you also see them change position in the clip.\n"
)


_TARGET_FLAG_LINE = (
    '- "t": 1 if this is THE person the robot should approach to start an interaction, else 0. '
    "Give 1 to EXACTLY ONE person if anyone is ATTENTIVE or AVAILABLE (ATTENTIVE first, then AVAILABLE, "
    "then the closest); give 0 to everyone ONLY if every person is BUSY, ENGAGED or MOVING.\n"
)


def _vlm_prompt_compact(dets: list[dict], *, video: bool) -> str:
    """Prompt corto con ejemplo JSON por ID (evita respuesta plana {\"a\":...})."""
    pids = sorted({d["pid"] for d in dets})
    keys = ",".join(str(p) for p in pids)
    id_map = ", ".join(f'{pid_label(d["pid"])}→"{d["pid"]}"' for d in dets)
    clip = "1s video" if video else "image"
    if use_vlm_social_states():
        # Placeholders en vez de valores reales a propósito: un ejemplo con
        # valores concretos ancla al modelo a repetirlos (medido: con
        # "stand"/"walk" de ejemplo casi todo salía "standing"; con "s":"A"
        # para el primer ID y "s":"E" para el segundo, el ejemplo sugería
        # "persona 1 libre, persona 2 en conversación").
        # "t" (1/0) por persona en vez de un "target":<ID> global al final:
        # medido en video, el "target" global salía null en 56/58 trozos aun
        # con gente ATTENTIVE/AVAILABLE; con un flag por persona el modelo
        # decide sobre cada una, y de los logits sale P(t=1) por persona
        # (ver inference.target_flag_probabilities).
        # Con VLM_TARGET_SOURCE=rules (default) el target sale de reglas sobre
        # 's' y el prompt no lo pide (menos tokens y menos con qué confundirse).
        ask_target = vlm_target_source() == "vlm"
        flag = ',"t":<0|1>' if ask_target else ""
        example = ",".join(f'"{p}":{{"a":"<action>","s":"<intention>"{flag}}}' for p in pids[:2])
        if len(pids) > 2:
            example += ",..."
        return COMPACT["with_social"].format(
            clip=clip, id_map=id_map, example=example, keys=keys,
            target_line=_TARGET_FLAG_LINE if ask_target else "",
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
            f'"{p}":{{"action":"<what they do>","social":"<intention>"}}' for p in pids
        ) + ',"target":<key>'
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


def vlm_prompt_person(*, video: bool = True) -> str:
    """Prompt de una sola persona (la de la caja amarilla, ver
    pose.gait.annotate_for_vlm_person)."""
    return COMPACT["per_person"].format(clip="1s video" if video else "image")


def vlm_prompt(dets: list[dict], *, video: bool = False) -> str:
    ordered = sort_dets_left_right([d for d in dets if "pid" in d])
    if not ordered:
        return NO_PEOPLE
    if vlm_prompt_mode() == "compact":
        return _vlm_prompt_compact(ordered, video=video)
    return _vlm_prompt_full(ordered, video=video)
