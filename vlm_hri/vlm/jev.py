"""Backend VLM_BACKEND=vlm_jev: modelos JEV-HRI (ejbejaranos/JEV-HRI-*).

A diferencia del VLM generativo (inference.py) no escribe texto: con UN
prefill de la imagen + un forward batched (persona x pregunta) devuelve
probabilidades calibradas sobre opciones cerradas (protocolo hri-bare-v1, ver
el README del repo del modelo). Lo que se pierde: no hay "acción" libre
(actions = "unknown" para todos). Lo que se gana: ~100 ms por imagen y
P(estado) por persona.

Preguntas que usa (VLM_JEV_CONFIG, por defecto configs/hri_intent.yaml del repo):
- intent (choice)            -> social_states + social_probs
- open_to_interact (noul)    -> target_probs (P(true) por persona)
- interruption_cost (score)  -> interrupt_cost (0-4 reescalado a 0-2)
y la política del YAML (jev_hri.policy.select_target) da target_pid.

El código del motor (jev_hri/) viaja en el propio repo del modelo y se
importa desde el snapshot descargado -- así se usa exactamente el protocolo
con el que se entrenó/evaluó en vez de una copia que pueda divergir.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from ..config import VLM_DEBUG, VLM_JEV_CONFIG, VLM_JEV_QUANT
from ..social_state import SOCIAL_UNKNOWN, normalize_social_state
from .inference import VlmInferenceResult

_INTENT = "intent"
_OPEN = "open_to_interact"
_COST = "interruption_cost"


def _snapshot(model_id: str) -> Path:
    if Path(model_id).exists():
        return Path(model_id)
    from huggingface_hub import snapshot_download

    # El adaptador LoRA ya va fusionado en model.safetensors.
    return Path(snapshot_download(model_id, ignore_patterns=["adapter/*", "*.png", "*ipynb_checkpoints*"]))


def robot_policy(policy: dict) -> dict:
    """Política del YAML sin BUSY como "último recurso": el robot solo se
    dirige a ATTENTIVE/AVAILABLE (lo mismo que acepta target_filter); BUSY
    queda solo para el VLM generativo."""
    return {**policy, "priority": [s for s in policy.get("priority", []) if s != "BUSY"]}


def load_jev(model_id: str, device=None):
    """(JevEngine, policy) listo para infer_intention_jev. `device` se ignora:
    jev_hri carga con device_map="cuda" (igual que en su benchmark)."""
    path = _snapshot(model_id)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from jev_hri.engine import JevEngine, Task, load_vlm

    cfg = yaml.safe_load(open(path / VLM_JEV_CONFIG, encoding="utf-8"))
    cal = json.load(open(path / "calibration.json", encoding="utf-8"))
    model, proc = load_vlm(str(path), quant=VLM_JEV_QUANT)
    engine = JevEngine(model, proc, Task(cfg), protocol="bare", calibration={"temperature": cal["per_kind"]})
    return engine, robot_policy(cfg["policy"])


def _best_frame(raw_frames: list[tuple[np.ndarray, list[dict]]], pids: list[int]) -> int:
    """Índice del frame más reciente en el que se ven más pids del trozo: JEV
    es de una sola imagen, y en el último frame puede faltar alguien que sí
    estuvo en el trozo (oclusión de un frame)."""
    want = set(pids)
    best, best_n = len(raw_frames) - 1, -1
    for i, (_, dets) in enumerate(raw_frames):
        n = len(want & {d["pid"] for d in dets if "pid" in d})
        if n >= best_n:
            best, best_n = i, n
    return best


def infer_intention_jev(
    engine,
    policy: dict,
    raw_frames: list[tuple[np.ndarray, list[dict]]],
    pids: list[int],
) -> VlmInferenceResult:
    """Una inferencia JEV sobre el mejor frame del trozo. `raw_frames` =
    [(frame BGR SIN anotar, dets)]: las cajas + etiquetas P1..Pn las dibuja
    jev_hri.annotate.mark_persons, igual que en el entrenamiento."""
    from jev_hri.annotate import mark_persons
    from jev_hri.policy import select_target

    if not raw_frames or not pids:
        return VlmInferenceResult({}, {}, 0.0)
    frame_bgr, dets = raw_frames[_best_frame(raw_frames, pids)]
    # Mismo orden que mark_persons (centro x, sort estable) -> P{k+1} = dets[k].
    want = set(pids)
    dets = sorted([d for d in dets if d.get("pid") in want], key=lambda d: d["x1"] + d["x2"])
    if not dets:
        return VlmInferenceResult({}, {}, 0.0)
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img, labels = mark_persons(pil, [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])
    pid_of = {lab: d["pid"] for lab, d in zip(labels, dets)}

    t0 = time.perf_counter()
    out, timing = engine.predict(img, labels)
    elapsed = time.perf_counter() - t0

    actions: dict[int, str] = {}
    social: dict[int, str] = {}
    social_probs: dict[int, dict[str, float]] = {}
    target_probs: dict[int, float] = {}
    interrupt_cost: dict[int, int] = {}
    for lab, pid in pid_of.items():
        ans = out.get(lab, {})
        actions[pid] = "unknown"
        intent = ans.get(_INTENT)
        social[pid] = normalize_social_state(intent["value"]) if intent else SOCIAL_UNKNOWN
        if intent:
            social_probs[pid] = intent["probabilities"]
        if _OPEN in ans:
            target_probs[pid] = ans[_OPEN]["probability"]
        if _COST in ans:
            # score 0-4 (minimal..very high) -> 0-2 de interrupt_cost.
            interrupt_cost[pid] = int(min(2, max(0, round(ans[_COST]["value"] / 2))))

    choice = select_target(out, policy)
    target_pid = pid_of.get(choice["target"])
    target_conf = None
    if target_pid is not None:
        # P(estado por el que lo eligió la política) de esa persona.
        target_conf = out[choice["target"]][_INTENT]["probabilities"][choice["state"]]
    if VLM_DEBUG:
        print(
            f"  JEV {labels}->{list(pid_of.values())} social={social} P(open)={target_probs} "
            f"cost={interrupt_cost} target={target_pid} ({choice['reason']}) p={target_conf} "
            f"[{timing['latency_s'] * 1000:.0f} ms, {timing['decision_rows']} filas]",
            flush=True,
        )
    return VlmInferenceResult(
        actions, social, elapsed, {}, interrupt_cost, target_pid, target_conf, target_probs, social_probs,
    )


def warmup_jev(engine, policy: dict) -> None:
    infer_intention_jev(
        engine, policy,
        [(np.zeros((240, 320, 3), dtype=np.uint8), [{"pid": 1, "x1": 0, "y1": 0, "x2": 80, "y2": 120}])],
        [1],
    )
