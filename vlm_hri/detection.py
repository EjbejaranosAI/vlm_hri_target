"""Detección de personas (YOLO), tracking entre frames y utilidades de caja."""

from __future__ import annotations

import numpy as np
import torch
from ultralytics import YOLO

from .config import (
    MAX_TRACK_IDS,
    TRACK_IOU,
    YOLO_CONF,
    YOLO_IOU,
    YOLO_MAX_DET,
    YOLO_POST_NMS_IOU,
)


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(dev)}")
    return dev


def box_iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ub = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (ua + ub - inter)


def nms_boxes(dets: list[dict], iou_thresh: float) -> list[dict]:
    if len(dets) <= 1:
        return dets
    kept = []
    for d in sorted(dets, key=lambda x: x["conf"], reverse=True):
        if all(box_iou(d, k) < iou_thresh for k in kept):
            kept.append(d)
    return kept


def assign_spatial_ids(dets: list[dict]) -> list[dict]:
    ordered = sorted(dets, key=lambda d: ((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2))
    for i, d in enumerate(ordered, 1):
        d["pid"] = i
    return ordered


def _box_center(d: dict) -> tuple[float, float]:
    return (d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2


def _center_distance(a: dict, b: dict) -> float:
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def track_detections(prev: list[dict], curr: list[dict], iou_thresh: float = TRACK_IOU) -> list[dict]:
    """Mantiene el mismo ID entre frames (IoU; si falla, proximidad de centro)."""
    if not prev:
        return assign_spatial_ids(curr)
    out: list[dict] = []
    used: set[int] = set()
    for c in curr:
        best_p, best_score = None, -1.0
        for p in prev:
            if p["pid"] in used:
                continue
            iou = box_iou(c, p)
            if iou >= iou_thresh:
                score = iou
            else:
                bw = max(c["x2"] - c["x1"], 1)
                dist = _center_distance(c, p)
                score = (1.0 - dist / (bw * 2.5)) if dist < bw * 2.5 else -1.0
            if score > best_score:
                best_score, best_p = score, p
        row = dict(c)
        if best_p and best_score > 0:
            row["pid"] = best_p["pid"]
            used.add(best_p["pid"])
        out.append(row)
    next_id = max((p["pid"] for p in prev), default=0) + 1
    for row in out:
        if "pid" not in row:
            if next_id <= MAX_TRACK_IDS:
                row["pid"] = next_id
                next_id += 1
            elif prev:
                nearest = min(prev, key=lambda p: _center_distance(row, p))
                row["pid"] = nearest["pid"]
            else:
                row["pid"] = 1
    return sorted(out, key=lambda d: d["pid"])


def stable_person_ids(
    frame_rows: list[dict],
    *,
    max_ids: int = MAX_TRACK_IDS,
    min_frames: int | None = None,
) -> list[int]:
    """IDs que aparecen bastante en el clip (evita tracks fantasma de 1–2 frames)."""
    from collections import Counter

    if not frame_rows:
        return []
    if min_frames is None:
        n_frames = max(r["frame"] for r in frame_rows) + 1
        min_frames = max(8, int(n_frames * 0.08))
    counts = Counter(r["person_id"] for r in frame_rows)
    ranked = [pid for pid, n in counts.most_common() if n >= min_frames]
    return ranked[:max_ids]


def pick_representative_frame(
    all_frame_dets: list[list[dict]], person_ids: list[int]
) -> tuple[int, list[dict]]:
    """Frame con más IDs estables visibles (mejor para VLM que 48 keys)."""
    best_i, best_dets, best_n = 0, [], -1
    want = set(person_ids)
    for i, dets in enumerate(all_frame_dets):
        n = sum(1 for d in dets if d["pid"] in want)
        if n > best_n:
            best_n, best_i, best_dets = n, i, dets
    return best_i, [d for d in best_dets if d["pid"] in want]


def detect_people(yolo: YOLO, bgr: np.ndarray) -> list[dict]:
    res = yolo.predict(
        bgr,
        classes=[0],
        verbose=False,
        device=0,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        max_det=YOLO_MAX_DET,
    )[0]
    dets: list[dict] = []
    if res.boxes is None:
        return dets
    h, w = bgr.shape[:2]
    for j in range(len(res.boxes)):
        x1, y1, x2, y2 = res.boxes.xyxy[j].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cid = int(res.boxes.cls[j].item())
        dets.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_name": res.names.get(cid, "person"),
                "conf": float(res.boxes.conf[j].item()),
            }
        )
    return nms_boxes(dets, YOLO_POST_NMS_IOU)


def sort_dets_left_right(dets: list[dict]) -> list[dict]:
    return sorted(dets, key=lambda d: ((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2))


def dets_for_vlm_prompt(
    buffer_dets: list[list[dict]], person_ids: list[int]
) -> list[dict]:
    """Una caja por ID para el prompt VLM (mejor área en el trozo, no solo frame central)."""
    want = set(person_ids)
    best: dict[int, tuple[float, dict]] = {}
    for row in buffer_dets:
        for d in row:
            pid = d.get("pid")
            if pid not in want:
                continue
            area = float((d["x2"] - d["x1"]) * (d["y2"] - d["y1"]))
            if pid not in best or area > best[pid][0]:
                best[pid] = (area, d)
    return [best[pid][1] for pid in person_ids if pid in best]


_yolo_warmed_shapes: set[tuple[int, int]] = set()


def warmup_yolo(yolo: YOLO, shape: tuple[int, int, int] = (480, 640, 3)) -> None:
    """cuDNN (benchmark=True) autotunea el algoritmo de convolución por CADA
    forma (H,W) de entrada distinta la primera vez que la ve — ese autotuneo
    puede costar 20-40s (no ms), y si se calienta con una forma genérica pero
    el video real tiene otra, el coste completo se paga igualmente dentro de
    los primeros 1-2 frames "cronometrados". Por eso cada forma nueva debe
    calentarse explícitamente con sus dimensiones reales antes de medir."""
    hw = (shape[0], shape[1])
    if hw in _yolo_warmed_shapes:
        return
    detect_people(yolo, np.zeros(shape, dtype=np.uint8))
    _yolo_warmed_shapes.add(hw)
