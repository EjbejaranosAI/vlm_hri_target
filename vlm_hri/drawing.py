"""Dibujo de cajas/paneles sobre el frame y formateo de etiquetas."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .actions import normalize_action
from .config import UI_BANNER_FACTOR, UI_REF_PX, UI_SCALE_MAX, UI_SCALE_MIN, VIDEO_VLM_FPS
from .detection import sort_dets_left_right
from .social_state import (
    SOCIAL_ATTENTIVE,
    SOCIAL_AVAILABLE,
    SOCIAL_BUSY,
    SOCIAL_ENGAGED,
    SOCIAL_MOVING,
    SOCIAL_UNKNOWN,
    action_to_social_state,
    social_state_mode,
)


_BOX_BGR = (255, 90, 210)


_BOX_EDGE_BGR = (160, 0, 120)


_LABEL_BG_BGR = (48, 24, 56)


_LABEL_FG_BGR = (255, 255, 255)


_BANNER_BG_BGR = (32, 28, 40)


_BANNER_ACCENT_BGR = (255, 90, 210)


_BANNER_TEXT_BGR = (248, 248, 252)


def pid_label(pid: int) -> str:
    return f"ID{pid}"


def annotate_for_vlm(frame_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    """Frame para el VLM: cajas púrpuras con etiqueta solo ID{n} (sin clase ni acción)."""
    out = frame_bgr.copy()
    for d in sort_dets_left_right(dets):
        draw_box(out, d["x1"], d["y1"], d["x2"], d["y2"], pid_label(d["pid"]))
    return out


_SOCIAL_BOX_BGR: dict[str, tuple[int, int, int]] = {
    SOCIAL_AVAILABLE: (80, 175, 76),  # verde
    SOCIAL_ATTENTIVE: (212, 188, 0),  # cian
    SOCIAL_BUSY: (0, 165, 255),  # naranja
    SOCIAL_ENGAGED: (54, 67, 244),  # rojo
    SOCIAL_MOVING: (243, 150, 33),  # azul
    SOCIAL_UNKNOWN: _BOX_BGR,  # púrpura (default actual)
}


def social_box_color(social_state: str | None) -> tuple[int, int, int]:
    return _SOCIAL_BOX_BGR.get(social_state or SOCIAL_UNKNOWN, _BOX_BGR)


def format_detection_label(
    pid: int,
    action: str,
    *,
    social_state: str | None = None,
    class_name: str = "person",
    show_action: bool = True,
) -> str:
    """Etiqueta en caja: ID + estado social (+ acción observable)."""
    state = social_state or (
        action_to_social_state(action)
        if social_state_mode() == "map"
        else SOCIAL_UNKNOWN
    )
    act = normalize_action(action) if action else ""
    if state != SOCIAL_UNKNOWN:
        if show_action and act and act != "unknown":
            return f"ID{pid}: {state} ({act})"
        return f"ID{pid}: {state}"
    if act and act != "unknown":
        return f"ID{pid}: {act}"
    return f"ID{pid} {class_name}"


def ui_scale(h: int, w: int, *, banner: bool = False) -> float:
    """Escala UI al lado corto del frame (video pequeño → etiquetas pequeñas)."""
    s = min(h, w) / UI_REF_PX
    s = max(UI_SCALE_MIN, min(UI_SCALE_MAX, s))
    if banner:
        s *= UI_BANNER_FACTOR
    return s


def draw_box(
    img, x1, y1, x2, y2, label: str, *, color: tuple[int, int, int] | None = None
) -> None:
    box_color = color if color is not None else _BOX_BGR
    h, w = img.shape[:2]
    s = ui_scale(h, w)
    box_w = max(x2 - x1, 1)
    # El trazo de color debe ser MÁS grueso que el borde oscuro: si es más
    # fino, la compresión H.264 lo diluye/mezcla con el borde y el color por
    # estado deja de distinguirse (se ve todo como el borde oscuro/púrpura).
    t_edge = max(1, int(round(s)))
    t_box = max(2, int(round(1.8 * s)))
    cv2.rectangle(img, (x1, y1), (x2, y2), _BOX_EDGE_BGR, t_edge)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, t_box)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = min(0.55 * s, max(0.32, box_w / 220.0))
    thick = max(1, int(round(min(s, 1.2))))
    pad = max(4, int(5 * s))
    (tw, text_h), baseline = cv2.getTextSize(label, font, fs, thick)
    box_h = text_h + baseline + 2 * pad
    ty_top = y1 + pad
    if ty_top + box_h > y2 - pad:
        ty_top = y1 - box_h - pad
        if ty_top < 0:
            ty_top = y2 + pad
    tx = max(0, min(x1, w - tw - 2 * pad - 1))
    ty_top = max(0, min(ty_top, h - box_h - 1))
    br_x = min(w - 1, tx + tw + 2 * pad)
    br_y = min(h - 1, ty_top + box_h)
    cv2.rectangle(img, (tx, ty_top), (br_x, br_y), _LABEL_BG_BGR, -1)
    cv2.rectangle(img, (tx, ty_top), (br_x, br_y), box_color, max(1, thick // 2))
    cv2.putText(
        img, label, (tx + pad, br_y - pad - 1), font, fs, _LABEL_FG_BGR, thick, cv2.LINE_AA
    )


def draw_banner(img, yolo_s: float, vlm_s: float, n: int) -> None:
    """Banner para pipeline de imágenes (1 YOLO + 1 VLM por imagen)."""
    total = yolo_s + vlm_s
    _draw_banner_lines(
        img,
        [
            f"Total pipeline: {total:.3f} s",
            f"  YOLO {yolo_s:.3f} s | VLM {vlm_s:.3f} s",
            f"Personas: {n}",
        ],
        banner=True,
        corner="bl",
    )


def draw_banner_video(
    img,
    *,
    yolo_total_s: float,
    n_frames: int,
    vlm_total_s: float,
    vlm_calls: int,
    n_people: int,
    pipeline_total_s: float,
    frame_i: int | None = None,
    vlm_input_kind: str = "chunks",
    chunk_sec: float = 0.0,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> None:
    """Banner inferior izquierdo: tiempos YOLO, VLM acumulado y total del pipeline."""
    yolo_ms = 1000.0 * yolo_total_s / max(n_frames, 1)
    vlm_avg = vlm_total_s / max(vlm_calls, 1)
    other_s = max(0.0, pipeline_total_s - yolo_total_s - vlm_total_s)
    lines = [
        f"Total pipeline: {pipeline_total_s:.1f}s",
        f"  YOLO {yolo_total_s:.1f}s | VLM {vlm_total_s:.2f}s | otro {other_s:.1f}s",
        f"YOLO: {yolo_ms:.0f} ms/frame",
        f"VLM acum: {vlm_total_s:.2f}s ({vlm_calls} inf, ~{vlm_avg:.2f}s/inf)",
    ]
    if vlm_input_kind == "chunks" and chunk_sec > 0:
        ci = (chunk_index or 0) + 1
        cc = chunk_count or vlm_calls
        lines.append(f"Trozo {ci}/{cc} cada {chunk_sec:.0f}s")
    elif vlm_input_kind == "video":
        lines.append(f"VLM: clip @ {VIDEO_VLM_FPS} fps")
    else:
        lines.append(f"VLM: imagen f{frame_i or 0}")
    lines.append(f"Personas: {n_people}")
    corner = os.environ.get("UI_BANNER_CORNER", "br").strip().lower()
    if corner not in ("bl", "br", "tl", "tr"):
        corner = "br"
    _draw_banner_lines(img, lines, banner=True, corner=corner)


def _draw_banner_lines(
    img, lines: list[str], *, banner: bool = False, corner: str = "bl"
) -> None:
    """Banner con texto nítido (LINE_8, píxeles enteros; evita blur de LINE_AA en video)."""
    h, w = img.shape[:2]
    s = ui_scale(h, w, banner=banner)
    font = cv2.FONT_HERSHEY_DUPLEX if banner else cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.52, min(0.85, 0.5 * s))
    thick = 1
    pad = int(max(10, 10 * s))
    gap = int(max(6, 5 * s))
    metrics = []
    for ln in lines:
        (tw, text_h), baseline = cv2.getTextSize(ln, font, fs, thick)
        metrics.append((ln, tw, text_h, baseline))
    bw = int(max(m[1] for m in metrics) + 2 * pad)
    bh = int(sum(m[2] + m[3] + gap for m in metrics) - gap + 2 * pad)
    margin = int(max(10, 10 * s))
    if corner == "bl":
        x0, y0 = margin, max(margin, h - bh - margin)
    elif corner == "br":
        x0, y0 = max(margin, w - bw - margin), max(margin, h - bh - margin)
    elif corner == "tr":
        x0, y0 = max(margin, w - bw - margin), margin
    else:
        x0, y0 = margin, margin
    x1, y1 = x0 + bw, y0 + bh
    cv2.rectangle(img, (x0, y0), (x1, y1), _BANNER_BG_BGR, -1)
    border = max(2, int(round(s)))
    cv2.rectangle(img, (x0, y0), (x1, y1), _BANNER_ACCENT_BGR, border)
    cy = int(y0 + pad)
    tx = int(x0 + pad)
    for ln, _, text_h, baseline in metrics:
        cy = int(cy + text_h)
        cv2.putText(
            img, ln, (tx, cy), font, fs, _BANNER_TEXT_BGR, thick, cv2.LINE_8
        )
        cy = int(cy + baseline + gap)
