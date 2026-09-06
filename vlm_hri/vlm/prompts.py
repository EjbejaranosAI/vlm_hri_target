"""Prompts del VLM: plantillas (JSON en prompt_templates/) + construcción según modo/social."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import vlm_prompt_mode
from ..detection import sort_dets_left_right
from ..drawing import pid_label
from ..social_state import use_vlm_social_states

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompt_templates"


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
        return COMPACT_WITH_SOCIAL.format(
            clip=clip, id_map=id_map, example=example, keys=keys
        )
    example = ",".join(f'"{p}":"<action>"' for p in pids[:2])
    return COMPACT_NO_SOCIAL.format(clip=clip, id_map=id_map, example=example, keys=keys)


def _vlm_prompt_full(dets: list[dict], *, video: bool) -> str:
    pids = sorted({d["pid"] for d in dets})
    mapping = "\n".join(
        f'- Box label {pid_label(d["pid"])} (person inside that box) → JSON key "{d["pid"]}"'
        for d in dets
    )
    ctx = "this short video clip" if video else "this image"
    motion_hint = MOTION_HINT_VIDEO if video else ""
    keys = ", ".join(str(p) for p in pids)
    if use_vlm_social_states():
        slots = ",".join(
            f'"{p}":{{"action":"<what they do>","social":"<STATE>"}}' for p in pids
        )
        return FULL_WITH_SOCIAL.format(
            ctx=ctx,
            n_people=len(pids),
            motion_hint=motion_hint,
            mapping=mapping,
            slots=slots,
            keys=keys,
        )
    slots = ",".join(f'"{p}":"<action>"' for p in pids)
    return FULL_NO_SOCIAL.format(
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
