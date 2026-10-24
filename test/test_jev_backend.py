"""Mapeo de la salida JEV (P1..Pn) a pids del tracker -- con un engine falso,
sin GPU. Usa jev_hri (annotate/policy) del snapshot del modelo en .hf_cache;
se salta si no está descargado."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from vlm_hri.config import ROOT

_SNAP = sorted((ROOT / ".hf_cache" / "hub" / "models--ejbejaranos--JEV-HRI-Qwen3-VL-2B" / "snapshots").glob("*"))
if not _SNAP:
    pytest.skip("snapshot JEV-HRI-Qwen3-VL-2B no descargado", allow_module_level=True)
sys.path.insert(0, str(_SNAP[-1]))

from vlm_hri.vlm.jev import _best_frame, infer_intention_jev, robot_policy  # noqa: E402

_STATES = ["ATTENTIVE", "AVAILABLE", "BUSY", "ENGAGED", "MOVING"]


def _choice(state: str, p: float) -> dict:
    rest = (1 - p) / 4
    probs = {s: (p if s == state else rest) for s in _STATES}
    return {"type": "choice", "value": state, "confidence": p, "probabilities": probs}


class _FakeEngine:
    def __init__(self, by_label: dict[str, tuple[str, float, float, float]]):
        self.by_label = by_label
        self.seen_labels = None

    def predict(self, image, labels):
        self.seen_labels = list(labels)
        out = {}
        for lab in labels:
            state, p, p_open, cost = self.by_label[lab]
            out[lab] = {
                "intent": _choice(state, p),
                "open_to_interact": {"type": "noul", "value": p_open >= 0.5, "probability": p_open},
                "interruption_cost": {"type": "score", "value": cost, "probabilities": {}},
            }
        return out, {"latency_s": 0.01, "decision_rows": 3 * len(labels)}


def _policy():
    cfg = yaml.safe_load(open(_SNAP[-1] / "configs" / "hri_intent.yaml"))
    return robot_policy(cfg["policy"])


def _det(pid, x1, x2):
    return {"pid": pid, "x1": x1, "y1": 10, "x2": x2, "y2": 200}


def test_labels_map_left_to_right_to_tracker_pids():
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    # pid 7 a la izquierda, pid 3 a la derecha -> P1 = 7, P2 = 3.
    dets = [_det(3, 400, 500), _det(7, 20, 120)]
    eng = _FakeEngine({"P1": ("BUSY", 0.7, 0.2, 3.0), "P2": ("ATTENTIVE", 0.8, 0.9, 0.4)})
    r = infer_intention_jev(eng, _policy(), [(frame, dets)], [3, 7])
    assert eng.seen_labels == ["P1", "P2"]
    assert r.social_states == {7: "BUSY", 3: "ATTENTIVE"}
    assert r.actions == {7: "unknown", 3: "unknown"}
    assert r.target_probs == {7: 0.2, 3: 0.9}
    assert r.interrupt_cost == {7: 2, 3: 0}
    assert r.target_pid == 3
    assert r.target_confidence == pytest.approx(0.8)
    assert r.social_probs[3]["ATTENTIVE"] == pytest.approx(0.8)


def test_no_eligible_target_is_none():
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    eng = _FakeEngine({"P1": ("MOVING", 0.9, 0.1, 4.0), "P2": ("ENGAGED", 0.6, 0.3, 3.0)})
    r = infer_intention_jev(eng, _policy(), [(frame, [_det(1, 0, 50), _det(2, 300, 400)])], [1, 2])
    assert r.target_pid is None and r.target_confidence is None


def test_busy_is_never_target():
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    eng = _FakeEngine({"P1": ("BUSY", 0.9, 0.6, 1.0), "P2": ("ENGAGED", 0.6, 0.3, 3.0)})
    r = infer_intention_jev(eng, _policy(), [(frame, [_det(1, 0, 50), _det(2, 300, 400)])], [1, 2])
    assert r.social_states[1] == "BUSY"
    assert r.target_pid is None


def test_best_frame_prefers_latest_with_most_pids():
    f = np.zeros((10, 10, 3), dtype=np.uint8)
    frames = [(f, [_det(1, 0, 1), _det(2, 2, 3)]), (f, [_det(1, 0, 1), _det(2, 2, 3)]), (f, [_det(1, 0, 1)])]
    assert _best_frame(frames, [1, 2]) == 1
