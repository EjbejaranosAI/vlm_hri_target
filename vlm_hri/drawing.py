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


_BANNER_BG_BGR = (38, 26, 22)


_BANNER_ACCENT_BGR = (255, 90, 210)


_BANNER_TEXT_BGR = (248, 248, 252)


# Etiqueta ("YOLO", "VLM", ...) en un gris azulado apagado -- el valor
# (el número que importa) va en _BANNER_TEXT_BGR (blanco casi puro), para que
# el ojo distinga de un vistazo la etiqueta del dato sin depender de negrita
# (las fuentes Hershey de OpenCV no tienen peso variable real).
_BANNER_LABEL_BGR = (170, 158, 148)


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
    """Barra de tiempos/latencia: una sola fila horizontal que ocupa todo el
    ancho del frame (arriba o abajo según UI_BANNER_SIDE), con los tiempos
    clave repartidos de izquierda a derecha en vez de apilados verticalmente.

    Cada segmento es (etiqueta, valor) -- la etiqueta se dibuja en un tono
    apagado y el valor en blanco brillante, para que el ojo distinga el dato
    de su descripción sin depender de negrita (los fonts Hershey de OpenCV no
    tienen peso variable real)."""
    yolo_ms = 1000.0 * yolo_total_s / max(n_frames, 1)
    vlm_avg = vlm_total_s / max(vlm_calls, 1)
    other_s = max(0.0, pipeline_total_s - yolo_total_s - vlm_total_s)
    segments = [
        ("Total", f"{pipeline_total_s:.1f}s"),
        ("YOLO", f"{yolo_total_s:.1f}s ({yolo_ms:.0f} ms/f)"),
        ("VLM", f"{vlm_total_s:.2f}s (~{vlm_avg:.2f}s/inf, {vlm_calls})"),
        ("Otro", f"{other_s:.1f}s"),
    ]
    if vlm_input_kind == "chunks" and chunk_sec > 0:
        ci = (chunk_index or 0) + 1
        cc = chunk_count or vlm_calls
        segments.append(("Trozo", f"{ci}/{cc} @ {chunk_sec:.0f}s"))
    elif vlm_input_kind == "video":
        segments.append(("Clip", f"@ {VIDEO_VLM_FPS} fps"))
    else:
        segments.append(("Imagen", f"f{frame_i or 0}"))
    segments.append(("Personas", str(n_people)))
    side = os.environ.get("UI_BANNER_SIDE", "bottom").strip().lower()
    _draw_banner_row(img, segments, side=side if side in ("top", "bottom") else "bottom")


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


def _segment_width(font, fs: float, thick: int, label: str, value: str) -> int:
    (lw, _), _ = cv2.getTextSize(f"{label} ", font, fs, thick)
    (vw, _), _ = cv2.getTextSize(value, font, fs, thick)
    return lw + vw


def _draw_banner_row(
    img, segments: list[tuple[str, str]], *, side: str = "bottom"
) -> None:
    """Barra horizontal de ancho completo (menos un margen a cada lado) con
    `segments` (etiqueta, valor) repartidos de izquierda a derecha en una
    sola fila — el primero pegado al margen izquierdo, el último al derecho,
    separadores verticales a mitad de cada hueco.

    Si el contenido no cabe a la escala normal (video muy angosto o muchos
    segmentos), primero se encoge la fuente (hasta un piso legible) y, si aun
    así no cabe, se sueltan los segmentos menos importantes (los últimos de
    la lista) uno a uno — nunca se deja texto que se salga del frame."""
    h, w = img.shape[:2]
    s = ui_scale(h, w, banner=True)
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = max(0.5, min(0.8, 0.48 * s))
    fs_floor = 0.32
    thick = 1
    margin = int(max(8, 8 * s))
    inner_pad = int(max(10, 10 * s))
    pad_y = int(max(8, 8 * s))
    min_gap = int(max(14, 14 * s))

    x0, x1 = margin, max(margin + 1, w - margin)
    inner_w = max(1, (x1 - x0) - 2 * inner_pad)

    segs = list(segments)
    while True:
        widths = [_segment_width(font, fs, thick, lbl, val) for lbl, val in segs]
        gaps = max(len(segs) - 1, 1)
        needed = sum(widths) + min_gap * gaps
        if needed <= inner_w or (fs <= fs_floor and len(segs) <= 1):
            break
        if fs > fs_floor:
            # Escala proporcional al espacio que sobra/falta, no a tientas:
            # una sola corrección basta salvo por el redondeo de cv2.
            fs = max(fs_floor, fs * (inner_w / max(needed, 1)))
        else:
            segs = segs[:-1]  # sin espacio ni encogiendo más: suelta el último (menos crítico)

    widths = [_segment_width(font, fs, thick, lbl, val) for lbl, val in segs]
    (_, text_h), baseline = cv2.getTextSize("Hg", font, fs, thick)
    bar_h = text_h + baseline + 2 * pad_y

    y0 = margin if side == "top" else max(margin, h - bar_h - margin)
    y1 = y0 + bar_h
    cv2.rectangle(img, (x0, y0), (x1, y1), _BANNER_BG_BGR, -1)
    border = max(2, int(round(s)))
    cv2.rectangle(img, (x0, y0), (x1, y1), _BANNER_ACCENT_BGR, border)

    n = len(segs)
    gaps = max(n - 1, 1)
    extra = inner_w - sum(widths)
    gap_w = max(min_gap, extra / gaps) if n > 1 else 0.0

    cy = int(y0 + pad_y + text_h)
    cx = float(x0 + inner_pad)
    for i, ((label, value), tw) in enumerate(zip(segs, widths)):
        (lw, _), _ = cv2.getTextSize(f"{label} ", font, fs, thick)
        cv2.putText(img, label, (int(cx), cy), font, fs, _BANNER_LABEL_BGR, thick, cv2.LINE_8)
        cv2.putText(
            img, value, (int(cx) + lw, cy), font, fs, _BANNER_TEXT_BGR, thick, cv2.LINE_8
        )
        cx += tw + gap_w
        if i < n - 1:
            sep_x = int(cx - gap_w / 2)
            cv2.line(img, (sep_x, y0 + 3), (sep_x, y1 - 3), _BANNER_ACCENT_BGR, 1)
