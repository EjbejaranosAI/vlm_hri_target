"""Parseo tolerante del JSON (posiblemente truncado) que devuelve el VLM."""

from __future__ import annotations

import ast
import json
import re

from ..actions import _is_id_label, normalize_action
from ..social_state import (
    SOCIAL_UNKNOWN,
    coerce_action_social,
    normalize_confidence,
    normalize_interrupt_cost,
    normalize_social_state,
)


def _parse_json_key(k) -> int | None:
    key = str(k).strip().strip('"').upper().removeprefix("ID")
    return int(key) if key.isdigit() else None


_TARGET_KEYS = ("target", "target_id")
# Flag por persona del prompt compacto: {"1": {"a":..,"s":..,"t":1}, ...}
_TARGET_FLAG_KEYS = ("t", "is_target")
_TARGET_CONF_KEYS = ("target_conf", "target_confidence", "target_prob", "p")

# Clave sin comillas tras "{" o "," -- Qwen2-VL devuelve a menudo
# `{1: {"a": ...}}` (medido en vivo: 34/35 trozos), que no es JSON válido.
_BARE_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*|\d+)\s*:')


def _repair_truncated(raw: str) -> str:
    """Cierra comillas/llaves que quedaron abiertas si generate() cortó por
    max_new_tokens."""
    text = ("{" + raw.lstrip("{")).strip()
    if text.count('"') % 2:
        text += '"'
    return text + "}" * max(0, text.count("{") - text.count("}"))


def load_vlm_json(raw: str) -> object | None:
    """Objeto JSON de la respuesta del VLM, tolerando lo que Qwen2-VL suele
    emitir mal: claves sin comillas (`{1: {...}}`), comillas simples, `None`
    estilo Python y respuestas truncadas. None si nada de eso lo arregla."""
    if not raw:
        return None
    text = _repair_truncated(raw)
    quoted = _BARE_KEY_RE.sub(r'\1"\2":', text)
    for blob in (text, quoted, quoted.replace("'", '"')):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def _is_flat_action_social_dict(data: dict) -> bool:
    """VLM devolvió un solo {a,s} en vez de {1:{a,s}, 2:{a,s}}."""
    keys = {str(k).lower() for k in data}
    allowed = {
        "a", "s", "action", "social", "social_state", "state",
        "c", "confidence", "ic", "interrupt_cost", "cost",
        *_TARGET_KEYS, *_TARGET_CONF_KEYS, *_TARGET_FLAG_KEYS,
    }
    return bool(keys) and keys <= allowed


_FLAT_ACTION_SOCIAL_RE = re.compile(r"^(.*?),\s*([ATBEM])\s*$", re.IGNORECASE)


def _extract_action_social(v: object) -> tuple[str, str, int | None, int | None]:
    """De un valor JSON (str o dict) extrae acción, estado social, confianza
    (0-2) y costo de interrupción (0-2). String plano "accion,X" (formato
    compacto actual) o "accion" solo no tienen sitio para confianza/costo
    (van siempre None ahí -- solo existen en el formato {a,s,c,ic}/
    {action,social,confidence,interrupt_cost})."""
    if isinstance(v, str) and v.strip():
        m = _FLAT_ACTION_SOCIAL_RE.match(v.strip())
        if m:
            action, social = coerce_action_social(
                normalize_action(m.group(1)), normalize_social_state(m.group(2))
            )
        else:
            action, social = coerce_action_social(normalize_action(v), SOCIAL_UNKNOWN)
        return action, social, None, None
    if not isinstance(v, dict):
        return "unknown", SOCIAL_UNKNOWN, None, None
    act = v.get("action") or v.get("a") or v.get("actions")
    if isinstance(act, list) and act:
        act = act[0]
    action = normalize_action(str(act)) if act else "unknown"
    social_raw = v.get("social") or v.get("s") or v.get("social_state") or v.get("state")
    social = (
        normalize_social_state(str(social_raw)) if social_raw is not None else SOCIAL_UNKNOWN
    )
    action, social = coerce_action_social(action, social)
    confidence = normalize_confidence(v.get("confidence", v.get("c")))
    interrupt_cost = normalize_interrupt_cost(
        v.get("interrupt_cost", v.get("ic", v.get("cost")))
    )
    return action, social, confidence, interrupt_cost


def parse_vlm_json(raw: str, person_ids: list[int]) -> dict[int, str]:
    actions, _, _, _ = parse_vlm_response(raw, person_ids)
    return actions


def parse_vlm_response(
    raw: str, person_ids: list[int]
) -> tuple[dict[int, str], dict[int, str], dict[int, int | None], dict[int, int | None]]:
    """Acciones + estados sociales + confianza/costo de interrupción
    (últimos dos: None por pid si el VLM no los dio -- se distingue a
    propósito de un 0 real) parseados del JSON del VLM."""
    allowed = set(person_ids)
    actions = {pid: "unknown" for pid in person_ids}
    social = {pid: SOCIAL_UNKNOWN for pid in person_ids}
    confidence: dict[int, int | None] = {pid: None for pid in person_ids}
    interrupt_cost: dict[int, int | None] = {pid: None for pid in person_ids}
    if not raw or not person_ids:
        return actions, social, confidence, interrupt_cost
    text = _repair_truncated(raw)
    data = load_vlm_json(raw)
    if isinstance(data, dict) and _is_flat_action_social_dict(data):
        act, soc, conf, cost = _extract_action_social(data)
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
            if conf is not None:
                confidence[pid] = conf
            if cost is not None:
                interrupt_cost[pid] = cost
    elif isinstance(data, dict):
        for k, v in data.items():
            pid = _parse_json_key(k)
            if pid is None or pid not in allowed:
                continue
            if isinstance(v, dict):
                nested_pid = _parse_json_key(v.get("id", v.get("person_id", k)))
                if nested_pid is not None and nested_pid in allowed:
                    pid = nested_pid
            act, soc, conf, cost = _extract_action_social(v)
            if _is_id_label(act):
                act = "unknown"
            if act != "unknown":
                actions[pid] = act
            if soc != SOCIAL_UNKNOWN:
                social[pid] = soc
            if conf is not None:
                confidence[pid] = conf
            if cost is not None:
                interrupt_cost[pid] = cost
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            pid = _parse_json_key(item.get("id", item.get("person_id")))
            if pid is None or pid not in allowed:
                continue
            act, soc, conf, cost = _extract_action_social(item)
            if act != "unknown":
                actions[pid] = act
            if soc != SOCIAL_UNKNOWN:
                social[pid] = soc
            if conf is not None:
                confidence[pid] = conf
            if cost is not None:
                interrupt_cost[pid] = cost
    for m in re.finditer(r'["\']?(\d+)["\']?\s*:\s*"([^"]*)"', text):
        pid = int(m.group(1))
        if pid in allowed and actions[pid] == "unknown" and m.group(2).strip():
            act_raw, soc_raw, _, _ = _extract_action_social(m.group(2))
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
    for m in re.finditer(
        r'["\']?(\d+)["\']?\s*:\s*\{[^}]*"(?:confidence|c)"\s*:\s*"?(\d+)"?', text, re.I
    ):
        pid = int(m.group(1))
        if pid in allowed:
            conf = normalize_confidence(m.group(2))
            if conf is not None:
                confidence[pid] = conf
    for m in re.finditer(
        r'["\']?(\d+)["\']?\s*:\s*\{[^}]*"(?:interrupt_cost|ic|cost)"\s*:\s*"?(\d+)"?',
        text, re.I,
    ):
        pid = int(m.group(1))
        if pid in allowed:
            cost = normalize_interrupt_cost(m.group(2))
            if cost is not None:
                interrupt_cost[pid] = cost
    for pid in person_ids:
        actions[pid], social[pid] = coerce_action_social(actions[pid], social[pid])
    return actions, social, confidence, interrupt_cost


def _parse_target_conf(v: object) -> float | None:
    """Confianza del target en [0, 1]; acepta también porcentaje (0-100)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(str(v).strip().rstrip("%"))
    except ValueError:
        return None
    if 1.0 < f <= 100.0:
        f /= 100.0
    return max(0.0, min(1.0, f))


_TARGET_RE = re.compile(r'["\']?target(?:_id)?["\']?\s*:\s*["\']?(?:ID)?(\d+|null|none)', re.I)
_TARGET_CONF_RE = re.compile(
    r'["\']?target_(?:conf|confidence|prob)["\']?\s*:\s*["\']?([0-9]*\.?[0-9]+)', re.I
)


def _flag_is_set(v: object) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().strip('"').lower() in ("1", "true", "yes")


_TARGET_FLAG_RE = re.compile(r'["\']?(\d+)["\']?\s*:\s*\{[^{}]*?(?<![A-Za-z_])["\']?t["\']?\s*:\s*["\']?(1|0|true|false)', re.I)


def parse_vlm_target_flags(raw: str, person_ids: list[int]) -> list[int] | None:
    """pids con "t": 1, en el orden en que salen. None si NINGUNA persona
    trae el campo "t" (respuesta en el formato viejo "target": <ID>, o
    truncada antes del primer "t") -- [] si lo traen y todos son 0."""
    allowed = set(person_ids)
    data = load_vlm_json(raw)
    if isinstance(data, dict):
        seen_flag = False
        flagged = []
        for k, v in data.items():
            pid = _parse_json_key(k)
            if pid not in allowed or not isinstance(v, dict):
                continue
            flag = next((v[f] for f in _TARGET_FLAG_KEYS if f in v), None)
            if flag is None:
                continue
            seen_flag = True
            if _flag_is_set(flag):
                flagged.append(pid)
        return flagged if seen_flag else None
    matches = [(int(m.group(1)), _flag_is_set(m.group(2))) for m in _TARGET_FLAG_RE.finditer(raw or "")]
    matches = [(pid, f) for pid, f in matches if pid in allowed]
    if not matches:
        return None
    return [pid for pid, f in matches if f]


def parse_vlm_target(raw: str, person_ids: list[int]) -> tuple[int | None, float | None]:
    """(pid, confianza 0-1) del target que eligió el VLM, o (None, None) si
    no eligió a nadie, el campo falta o el ID no es una de las personas que
    se le pasaron (un ID inventado no se acepta).

    Formato actual: flag "t" por persona (ver parse_vlm_target_flags) --
    gana el primer "t": 1 (si hay varios, inference elige el de mayor
    P(t=1) de los logits). Si ninguna persona trae "t", cae al formato
    viejo de un "target": <ID> global (prompt completo/full.json)."""
    flagged = parse_vlm_target_flags(raw, person_ids)
    if flagged is not None:
        return (flagged[0], None) if flagged else (None, None)
    allowed = set(person_ids)
    data = load_vlm_json(raw)
    target_raw: object = None
    conf_raw: object = None
    if isinstance(data, dict):
        target_raw = next((data[k] for k in _TARGET_KEYS if k in data), None)
        conf_raw = next((data[k] for k in _TARGET_CONF_KEYS if k in data), None)
        if not any(k in data for k in _TARGET_KEYS):
            # El modelo a veces mete "target" dentro del objeto de una persona
            # (medido: `{1: {"a": ..., "s": ..., "target": null, ...}}`):
            # `true` = esa persona, un ID = ese ID.
            for k, v in data.items():
                if not isinstance(v, dict):
                    continue
                inner = next((v[t] for t in _TARGET_KEYS if t in v), None)
                if inner is True:
                    inner = k
                if inner is not None and inner is not False:
                    target_raw = inner
                    conf_raw = next((v[c] for c in _TARGET_CONF_KEYS if c in v), None)
                    break
    elif raw:
        text = _repair_truncated(raw)
        m = _TARGET_RE.search(text)
        target_raw = m.group(1) if m else None
        m = _TARGET_CONF_RE.search(text)
        conf_raw = m.group(1) if m else None
    pid = _parse_json_key(target_raw) if target_raw is not None else None
    if pid is None or pid not in allowed:
        return None, None
    return pid, _parse_target_conf(conf_raw)
