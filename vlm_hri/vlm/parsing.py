"""Parseo tolerante del JSON (posiblemente truncado) que devuelve el VLM."""

from __future__ import annotations

import json
import re

from ..actions import _is_id_label, normalize_action
from ..social_state import SOCIAL_UNKNOWN, coerce_action_social, normalize_social_state


def _parse_json_key(k) -> int | None:
    key = str(k).strip().strip('"').upper().removeprefix("ID")
    return int(key) if key.isdigit() else None


def _is_flat_action_social_dict(data: dict) -> bool:
    """VLM devolvió un solo {a,s} en vez de {1:{a,s}, 2:{a,s}}."""
    keys = {str(k).lower() for k in data}
    allowed = {"a", "s", "action", "social", "social_state", "state"}
    return bool(keys) and keys <= allowed


_FLAT_ACTION_SOCIAL_RE = re.compile(r"^(.*?),\s*([ATBEM])\s*$", re.IGNORECASE)


def _extract_action_social(v: object) -> tuple[str, str]:
    """De un valor JSON (str o dict) extrae acción y estado social.
    String plano "accion,X" (formato compacto actual) o "accion" solo."""
    if isinstance(v, str) and v.strip():
        m = _FLAT_ACTION_SOCIAL_RE.match(v.strip())
        if m:
            return coerce_action_social(
                normalize_action(m.group(1)), normalize_social_state(m.group(2))
            )
        return coerce_action_social(normalize_action(v), SOCIAL_UNKNOWN)
    if not isinstance(v, dict):
        return "unknown", SOCIAL_UNKNOWN
    act = v.get("action") or v.get("a") or v.get("actions")
    if isinstance(act, list) and act:
        act = act[0]
    action = normalize_action(str(act)) if act else "unknown"
    social_raw = v.get("social") or v.get("s") or v.get("social_state") or v.get("state")
    social = (
        normalize_social_state(str(social_raw)) if social_raw is not None else SOCIAL_UNKNOWN
    )
    return coerce_action_social(action, social)


def parse_vlm_json(raw: str, person_ids: list[int]) -> dict[int, str]:
    actions, _ = parse_vlm_response(raw, person_ids)
    return actions


def parse_vlm_response(
    raw: str, person_ids: list[int]
) -> tuple[dict[int, str], dict[int, str]]:
    """Acciones + estados sociales parseados del JSON del VLM."""
    allowed = set(person_ids)
    actions = {pid: "unknown" for pid in person_ids}
    social = {pid: SOCIAL_UNKNOWN for pid in person_ids}
    if not raw or not person_ids:
        return actions, social
    text = ("{" + raw.lstrip("{")).strip()
    if text.count('"') % 2:
        text += '"'
    text += "}" * max(0, text.count("{") - text.count("}"))
    data = None
    for blob in (text, text.replace("'", '"')):
        try:
            data = json.loads(blob)
            break
        except json.JSONDecodeError:
            pass
    if isinstance(data, dict) and _is_flat_action_social_dict(data):
        act, soc = _extract_action_social(data)
        if _is_id_label(act):
            pid_hint = _parse_json_key(act)
            act = "unknown"
        else:
            pid_hint = None
        targets = (
            [pid_hint]
            if pid_hint is not None and pid_hint in allowed
            else (person_ids if len(person_ids) == 1 else [])
        )
        for pid in targets:
            if act != "unknown":
                actions[pid] = act
            if soc != SOCIAL_UNKNOWN:
                social[pid] = soc
    elif isinstance(data, dict):
        for k, v in data.items():
            pid = _parse_json_key(k)
            if pid is None or pid not in allowed:
                continue
            if isinstance(v, dict):
                nested_pid = _parse_json_key(v.get("id", v.get("person_id", k)))
                if nested_pid is not None and nested_pid in allowed:
                    pid = nested_pid
            act, soc = _extract_action_social(v)
            if _is_id_label(act):
                act = "unknown"
            if act != "unknown":
                actions[pid] = act
            if soc != SOCIAL_UNKNOWN:
                social[pid] = soc
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            pid = _parse_json_key(item.get("id", item.get("person_id")))
            if pid is None or pid not in allowed:
                continue
            act, soc = _extract_action_social(item)
            if act != "unknown":
                actions[pid] = act
            if soc != SOCIAL_UNKNOWN:
                social[pid] = soc
    for m in re.finditer(r'["\']?(\d+)["\']?\s*:\s*"([^"]*)"', text):
        pid = int(m.group(1))
        if pid in allowed and actions[pid] == "unknown" and m.group(2).strip():
            act_raw, soc_raw = _extract_action_social(m.group(2))
            act, soc = coerce_action_social(
                act_raw, soc_raw if soc_raw != SOCIAL_UNKNOWN else social[pid]
            )
            if act != "unknown":
                actions[pid] = act
            if soc != SOCIAL_UNKNOWN:
                social[pid] = soc
    for m in re.finditer(
        r'["\']?(\d+)["\']?\s*:\s*\{[^}]*"(?:action|a)"\s*:\s*"([^"]+)"', text
    ):
        pid = int(m.group(1))
        if pid in allowed:
            act, soc = coerce_action_social(
                normalize_action(m.group(2)), social[pid]
            )
            if act != "unknown":
                actions[pid] = act
            elif soc != SOCIAL_UNKNOWN:
                actions[pid] = "unknown"
            if soc != SOCIAL_UNKNOWN:
                social[pid] = soc
    for m in re.finditer(
        r'["\']?(\d+)["\']?\s*:\s*\{[^}]*"(?:social|s)"\s*:\s*"([^"]+)"', text, re.I
    ):
        pid = int(m.group(1))
        if pid in allowed:
            social[pid] = normalize_social_state(m.group(2))
    for pid in person_ids:
        actions[pid], social[pid] = coerce_action_social(actions[pid], social[pid])
    return actions, social
