"""Código compartido: YOLO, VLM, dibujo, parseo JSON."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import NamedTuple

import prompts as VLM_PROMPTS

# torch (intra/interop) + cv2 por defecto reclaman todos los cores (32 en esta
# máquina); al correr YOLO (GPU) + decode/encode de video (CPU) en el mismo
# proceso, la sobre-suscripción de hilos genera contención (context-switch
# thrashing) y puede hacer que fases CPU-ligeras tarden 5-8x más de lo normal.
# Se fija un límite razonable ANTES de importar torch/cv2 para que ambos lo
# respeten desde su inicialización.
_CPU_THREADS = int(os.environ.get("PIPELINE_CPU_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from ultralytics import YOLO

torch.set_num_threads(_CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # ya se usó paralelismo interop en este proceso; no se puede cambiar
cv2.setNumThreads(_CPU_THREADS)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
for _p in (ROOT / ".hf_cache", ROOT / ".mplconfig"):
    _p.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS = "yolo11n.pt"
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.55"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.40"))
YOLO_MAX_DET = int(os.environ.get("YOLO_MAX_DET", "30"))
YOLO_POST_NMS_IOU = float(os.environ.get("YOLO_POST_NMS_IOU", "0.50"))
TRACK_IOU = float(os.environ.get("TRACK_IOU", "0.45"))
MAX_TRACK_IDS = int(os.environ.get("MAX_TRACK_IDS", "12"))
UI_REF_PX = int(os.environ.get("UI_REF_PX", "720"))
UI_SCALE_MIN = float(os.environ.get("UI_SCALE_MIN", "0.35"))
UI_SCALE_MAX = float(os.environ.get("UI_SCALE_MAX", "1.75"))
UI_BANNER_FACTOR = float(os.environ.get("UI_BANNER_FACTOR", "0.62"))
VIDEO_VLM_CHUNK_SEC = float(os.environ.get("VIDEO_VLM_CHUNK_SEC", "1"))
VLM_ID = os.environ.get("VLM_ID", "Qwen/Qwen2-VL-2B-Instruct")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VLM_DEBUG = os.environ.get("VLM_DEBUG", "0") == "1"
VLM_SAVE_INPUT = os.environ.get("VLM_SAVE_INPUT", "0") == "1"

_PATCH = 28 * 28
_PROFILES = {
    "stream": ("448,384", 448, 80),
    "fast": ("512,384", 512, 96),
    "balanced": ("640,512", 768, 120),
    "quality": ("768,640", 1024, 140),
}


def vlm_resize_limits() -> tuple[tuple[int, ...], int, int]:
    """(max_sides, max_pixels, max_new_cap) según VLM_PROFILE / env."""
    prof_name = os.environ.get("VLM_PROFILE", "balanced").lower()
    prof = _PROFILES.get(prof_name, _PROFILES["balanced"])
    raw_side = os.environ.get("VLM_MAX_SIDE", prof[0])
    sides = tuple(int(x) for x in raw_side.split(",") if x.strip())
    max_pixels = int(os.environ.get("VLM_MAX_PIXELS", str(prof[1] * _PATCH)))
    max_new_cap = int(os.environ.get("VLM_MAX_NEW_CAP", str(prof[2])))
    return sides, max_pixels, max_new_cap


VLM_MAX_SIDE, VLM_MAX_PIXELS, VLM_MAX_NEW_CAP = vlm_resize_limits()


def chunk_vlm_use_image(chunk_sec: float) -> bool:
    """auto: vídeo por defecto; imagen solo si VLM_CHUNK_AUTO_IMAGE_SEC > 0 y trozo corto."""
    mode = os.environ.get("VLM_CHUNK_INPUT", "auto").strip().lower()
    if mode in ("image", "img", "1", "true", "yes"):
        return True
    if mode in ("video", "vid", "0", "false", "no"):
        return False
    limit = float(os.environ.get("VLM_CHUNK_AUTO_IMAGE_SEC", "0"))
    return limit > 0 and chunk_sec <= limit


def vlm_prompt_mode() -> str:
    """full = prompt largo; compact = corto (menos latencia, default)."""
    m = os.environ.get("VLM_PROMPT_MODE", "compact").strip().lower()
    return m if m in ("full", "compact") else "compact"


def video_vlm_fps(chunk_sec: float) -> float:
    """Muestreo temporal del clip (compact: ~3 frames/s de trozo)."""
    raw = os.environ.get("VIDEO_VLM_FPS")
    if raw is not None and str(raw).strip() != "":
        return float(raw)
    target = 3.0 if vlm_prompt_mode() == "compact" else 4.0
    return max(2.0, min(6.0, target / max(chunk_sec, 0.25)))


VLM_LOAD_IN_4BIT = os.environ.get("VLM_LOAD_IN_4BIT", "1") == "1"
VLM_BATCH_SIZE = int(os.environ.get("VLM_BATCH_SIZE", "16"))
# qwen_vl_utils ignora min/max_pixels del processor para clips de video y usa
# su propio límite interno (~768 tokens/frame ≈ 602K px) salvo que se pase
# explícito por mensaje; en videos de resolución media/alta (p.ej. 960x540)
# eso deja frames sin recortar y cuesta ~40% más de latencia sin mejorar el
# resultado. Se fuerza aquí un techo más bajo para todos los clips de video.
VLM_VIDEO_MAX_PIXELS = int(os.environ.get("VLM_VIDEO_MAX_PIXELS", str(384 * _PATCH)))
VLM_COMPILE = os.environ.get("VLM_COMPILE", "0") == "1"
VLM_WARMUP = os.environ.get("VLM_WARMUP", "1") == "1"
VIDEO_VLM_FPS = float(os.environ.get("VIDEO_VLM_FPS", "2"))
# Estados sociales: SOCIAL_STATE_MODE=vlm (default) | map

_VLM_ATTN = os.environ.get("VLM_ATTN", "auto")
if _VLM_ATTN == "auto":
    try:
        import flash_attn  # noqa: F401

        _VLM_ATTN = "flash_attention_2"
    except ImportError:
        _VLM_ATTN = "sdpa"

# Púrpura visible (BGR): borde oscuro + caja brillante
_BOX_BGR = (255, 90, 210)
_BOX_EDGE_BGR = (160, 0, 120)
_LABEL_BG_BGR = (48, 24, 56)
_LABEL_FG_BGR = (255, 255, 255)
_BANNER_BG_BGR = (32, 28, 40)
_BANNER_ACCENT_BGR = (255, 90, 210)
_BANNER_TEXT_BGR = (248, 248, 252)


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


def load_vlm(model_id: str, device: torch.device):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _, max_pixels, _ = vlm_resize_limits()
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256 * _PATCH, max_pixels=max_pixels
    )
    kwargs: dict = {"torch_dtype": dtype}
    if VLM_LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["attn_implementation"] = _VLM_ATTN
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if not VLM_LOAD_IN_4BIT:
        model = model.to(device)
    model.eval()
    if VLM_COMPILE:
        model = torch.compile(model, mode="reduce-overhead")
    return model, processor


def pid_label(pid: int) -> str:
    return f"ID{pid}"


def _parse_json_key(k) -> int | None:
    key = str(k).strip().strip('"').upper().removeprefix("ID")
    return int(key) if key.isdigit() else None


def _is_id_label(text: str) -> bool:
    t = str(text).strip().upper()
    return t == "ID" or bool(re.match(r"^ID\d+$", t))


def _is_flat_action_social_dict(data: dict) -> bool:
    """VLM devolvió un solo {a,s} en vez de {1:{a,s}, 2:{a,s}}."""
    keys = {str(k).lower() for k in data}
    allowed = {"a", "s", "action", "social", "social_state", "state"}
    return bool(keys) and keys <= allowed


def sort_dets_left_right(dets: list[dict]) -> list[dict]:
    return sorted(dets, key=lambda d: ((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2))


def annotate_for_vlm(frame_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    """Frame para el VLM: cajas púrpuras con etiqueta solo ID{n} (sin clase ni acción)."""
    out = frame_bgr.copy()
    for d in sort_dets_left_right(dets):
        draw_box(out, d["x1"], d["y1"], d["x2"], d["y2"], pid_label(d["pid"]))
    return out


# Estados de interacción social (capa robot; se derivan de la acción del VLM)
SOCIAL_ATTENTIVE = "ATTENTIVE"
SOCIAL_AVAILABLE = "AVAILABLE"
SOCIAL_BUSY = "BUSY"
SOCIAL_ENGAGED = "ENGAGED"
SOCIAL_MOVING = "MOVING"
SOCIAL_UNKNOWN = "UNKNOWN"

# Color de caja por estado social (BGR) — un vistazo distingue quién está
# libre/atento/ocupado/en conversación/en movimiento sin leer la etiqueta.
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


_ACTION_TO_SOCIAL: dict[str, str] = {
    "standing": SOCIAL_AVAILABLE,
    "sitting": SOCIAL_AVAILABLE,
    "idle": SOCIAL_AVAILABLE,
    "looking": SOCIAL_ATTENTIVE,
    "greeting": SOCIAL_ATTENTIVE,
    "waving": SOCIAL_ATTENTIVE,
    "talking": SOCIAL_ENGAGED,
    "phone": SOCIAL_BUSY,
    "reading": SOCIAL_BUSY,
    "working": SOCIAL_BUSY,
    "blocking": SOCIAL_ENGAGED,
    "walking": SOCIAL_MOVING,
    "running": SOCIAL_MOVING,
}

_SOCIAL_SUITABILITY: dict[str, str] = {
    SOCIAL_AVAILABLE: "High",
    SOCIAL_ATTENTIVE: "Very High",
    SOCIAL_BUSY: "Low",
    SOCIAL_ENGAGED: "Low",
    SOCIAL_MOVING: "Low",
    SOCIAL_UNKNOWN: "?",
}

_SOCIAL_ALIASES: dict[str, str] = {
    "available": SOCIAL_AVAILABLE,
    "free": SOCIAL_AVAILABLE,
    "idle": SOCIAL_AVAILABLE,
    "interruptible": SOCIAL_AVAILABLE,
    "attentive": SOCIAL_ATTENTIVE,
    "aware": SOCIAL_ATTENTIVE,
    "receptive": SOCIAL_ATTENTIVE,
    "busy": SOCIAL_BUSY,
    "occupied": SOCIAL_BUSY,
    "phone": SOCIAL_BUSY,
    "reading": SOCIAL_BUSY,
    "engaged": SOCIAL_ENGAGED,
    "conversation": SOCIAL_ENGAGED,
    "moving": SOCIAL_MOVING,
    "motion": SOCIAL_MOVING,
    "locomotion": SOCIAL_MOVING,
}


class VlmInferenceResult(NamedTuple):
    actions: dict[int, str]
    social_states: dict[int, str]
    elapsed_s: float


def social_state_mode() -> str:
    m = os.environ.get("SOCIAL_STATE_MODE", "vlm").strip().lower()
    return m if m in ("vlm", "map") else "vlm"


def social_vlm_fallback_map() -> bool:
    return os.environ.get("SOCIAL_VLM_FALLBACK_MAP", "1") == "1"


def social_reconcile_action() -> bool:
    """Si 1, corrige social incoherente con la acción (p. ej. walking+ENGAGED → MOVING)."""
    return os.environ.get("SOCIAL_RECONCILE_ACTION", "1") == "1"


def use_vlm_social_states() -> bool:
    return social_state_mode() == "vlm"


def _is_sitting_phrase(low: str) -> bool:
    """Sentado real (evita falsos positivos tipo 'sit' dentro de otras palabras)."""
    return bool(
        re.search(r"\bsitting\b", low)
        or re.search(r"\bseated\b", low)
        or re.search(r"\bsit\b", low)
        or re.search(r"\bsits\b", low)
        or re.search(r"\bon\s+(?:a\s+)?(?:chair|bench|sofa|couch|seat)\b", low)
    )


def _action_traits(low: str) -> dict[str, bool]:
    return {
        "sitting": _is_sitting_phrase(low),
        "standing": bool(
            re.search(r"\bstanding\b", low)
            or re.search(r"\bstood\b", low)
            or re.search(r"\bstand(?:ing|s)?\b", low)
        ),
        "walking": bool(
            re.search(r"\bwalking\b", low)
            or re.search(r"\bwalk(?:ing|s|ed)?\b", low)
        ),
        "running": bool(re.search(r"\brunning\b", low) or re.search(r"\bjogg", low)),
        "talking": bool(
            re.search(r"\btalk", low)
            or re.search(r"\bspeak", low)
            or re.search(r"\bconvers", low)
        ),
        "smiling": bool(re.search(r"\bsmil", low)),
        "phone": bool(re.search(r"\bphone\b", low) or re.search(r"\bmobile\b", low)),
    }


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


def chunk_motion_enabled() -> bool:
    return os.environ.get("CHUNK_MOTION_HINT", "1") == "1"


def chunk_motion_by_pid(
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    frame_size: tuple[int, int],
) -> dict[int, str]:
    """Desplazamiento del centro de la caja en el trozo → walking.

    Umbrales calibrados empíricamente sobre CSVs de detección reales (videos
    con caminata real vs. videos de gente quieta): a ~25 fps una persona
    caminando desplaza el centro de su caja solo ~1-3 px POR FRAME (no ~40+ px
    como asumían los umbrales anteriores, que exigían ~0.14·alto por frame y
    nunca se cumplían — por eso "walking" casi nunca se detectaba por
    cinemática y la gente en movimiento quedaba como AVAILABLE). Aquí solo se
    exige desplazamiento total del trozo (relativo al alto de la caja) y
    direccionalidad neta/recorrido para filtrar el zigzag de tracking de
    alguien parado.
    """
    if not buffer_dets or not person_ids:
        return {}
    total_floor_px = float(os.environ.get("CHUNK_MOTION_PX", "6"))
    rel_total = float(os.environ.get("CHUNK_MOTION_TOTAL_REL", "0.06"))
    # Persona parada con jitter de tracking (re-detección, cambio de postura,
    # oclusión parcial) acumula desplazamiento en zigzag: mucho "path" pero
    # poco avance neto. Caminar de verdad es direccional: el desplazamiento
    # neto (inicio→fin) se acerca al camino recorrido.
    #
    # Límite conocido (medido, no solo teórico): un balanceo de peso MUY lento
    # de pie (p. ej. alguien hablando/sonriendo a cámara) puede verse igual de
    # "direccional" que caminar si su período de vaivén es más largo que el
    # propio clip — solo se observa un tramo de la oscilación, nunca el
    # regreso. Subir este umbral NO lo arregla (se probó empíricamente: subirlo
    # perdía caminatas reales sin quitar ese falso positivo, porque el vaivén
    # medido llegaba a la misma direccionalidad que un paso lento real).
    # Arreglarlo de raíz requiere una ventana temporal más larga que un trozo
    # (para ver si la persona vuelve cerca de su posición de origen) o una
    # señal distinta a la posición del centro de la caja (p. ej. pose/marcha).
    min_directionality = float(os.environ.get("CHUNK_MOTION_DIRECTIONALITY", "0.35"))
    out: dict[int, str] = {}
    for pid in person_ids:
        pts: list[tuple[float, float]] = []
        heights: list[float] = []
        for row in buffer_dets:
            for d in row:
                if d["pid"] == pid:
                    pts.append((0.5 * (d["x1"] + d["x2"]), 0.5 * (d["y1"] + d["y2"])))
                    bh = d["y2"] - d["y1"]
                    if bh > 24:
                        heights.append(bh)
                    break
        if len(pts) < 4:
            continue
        dist = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            dist += (dx * dx + dy * dy) ** 0.5
        avg_h = sum(heights) / len(heights) if heights else 80.0
        total_thr = max(total_floor_px, rel_total * avg_h)
        net_dx = pts[-1][0] - pts[0][0]
        net_dy = pts[-1][1] - pts[0][1]
        net_disp = (net_dx * net_dx + net_dy * net_dy) ** 0.5
        directionality = net_disp / dist if dist > 1e-6 else 0.0
        if dist >= total_thr and directionality >= min_directionality:
            out[pid] = "walking"
    return out


def chunk_posture_hints(
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    motion: dict[int, str],
) -> dict[int, str]:
    """Silueta ancha + poco desplazamiento → probablemente sentado."""
    if os.environ.get("CHUNK_POSTURE_HINT", "0") != "1":
        return {}
    aspect_min = float(os.environ.get("CHUNK_SIT_ASPECT_MIN", "0.88"))
    vert_max = float(os.environ.get("CHUNK_SIT_VERT_REL", "0.05"))
    out: dict[int, str] = {}
    for row in buffer_dets:
        if not row:
            continue
        h_img = max(d["y2"] for d in row) - min(d["y1"] for d in row)
        if h_img < 48:
            h_img = 480.0
        break
    else:
        h_img = 480.0
    for pid in person_ids:
        if motion.get(pid) == "walking":
            continue
        aspects: list[float] = []
        ys: list[float] = []
        for row in buffer_dets:
            for d in row:
                if d["pid"] == pid:
                    bh = d["y2"] - d["y1"]
                    bw = d["x2"] - d["x1"]
                    if bh > 24:
                        aspects.append(bw / bh)
                        ys.append(0.5 * (d["y1"] + d["y2"]))
                    break
        if not aspects or sum(aspects) / len(aspects) < aspect_min:
            continue
        vert_span = (max(ys) - min(ys)) if len(ys) >= 2 else 0.0
        if vert_span <= vert_max * h_img:
            out[pid] = "sitting"
    return out


def _action_secondary_parts(tr: dict[str, bool]) -> list[str]:
    parts: list[str] = []
    for label, key in (
        ("talking", "talking"),
        ("smiling", "smiling"),
        ("using phone", "phone"),
    ):
        if tr[key]:
            parts.append(label)
    return parts


def apply_chunk_kinematic_hints(
    actions: dict[int, str],
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    frame_size: tuple[int, int],
    *,
    moving_pids: set[int] | None = None,
) -> dict[int, str]:
    """Combina VLM + movimiento de cajas en el trozo (walking/sitting).

    El desplazamiento real de la caja es la señal más fiable de que alguien
    camina — prima sobre lo que haya dicho el VLM (p. ej. "talking"), salvo
    que el VLM afirme sentado (incompatible con caminar). Cualquier intención
    secundaria detectada (hablar, sonreír, teléfono) se conserva anexada en
    vez de descartarse, para no perder señales de posible interacción.

    `moving_pids`, si se pasa, reemplaza el cálculo de bbox-motion de aquí:
    el pipeline de pose (run_video_pose.refine_labels_pose) ya resuelve
    "quién se mueve de verdad" combinando marcha real + profundidad + bbox +
    histéresis + debias de grupo ANTES de llamar a esta función — recalcular
    bbox-motion otra vez aquí (una señal más cruda, pensada para el pipeline
    SIN pose que no tiene nada mejor) podía pisar esa decisión ya tomada. Sin
    `moving_pids` (pipeline sin pose), el comportamiento es el de siempre."""
    if not buffer_dets:
        return actions
    if moving_pids is not None:
        motion = {pid: "walking" for pid in moving_pids}
    elif chunk_motion_enabled():
        motion = chunk_motion_by_pid(buffer_dets, person_ids, frame_size)
    else:
        return actions
    posture = chunk_posture_hints(buffer_dets, person_ids, motion)
    out: dict[int, str] = {}
    for pid in person_ids:
        raw = actions.get(pid, "unknown")
        act = normalize_action(raw)
        tr = _action_traits(act.lower())
        extras = _action_secondary_parts(tr)
        has_semantic = tr["talking"] or tr["smiling"] or tr["phone"]
        clearly_sitting = tr["sitting"] and not tr["standing"]
        if motion.get(pid) == "walking" and not clearly_sitting:
            # Antes esto solo conservaba talking/smiling/phone (extras) y
            # descartaba cualquier otra descripción del VLM (p. ej.
            # "greeting person", "gesturing") — con el prompt ahora en texto
            # libre, eso perdía la interacción real y forzaba MOVING sobre
            # cualquier cosa. Se conserva el texto tal cual (solo se
            # reemplaza la postura de pie por "walking", que ya la implica).
            if act.lower() in ("unknown", "standing", "stand", "sitting", "sit", "", "walking", "walk"):
                out[pid] = "walking"
            else:
                stripped = re.sub(r"^(standing|stand)\s+and\s+", "", act, flags=re.IGNORECASE)
                out[pid] = f"walking and {stripped}"
            continue
        if (
            posture.get(pid) == "sitting"
            and not tr["walking"]
            and not has_semantic
            and act in ("unknown", "standing", "sitting", "")
        ):
            base = "sitting"
            out[pid] = " and ".join([base] + extras) if extras else base
            continue
        out[pid] = act
    for pid, act in actions.items():
        if pid not in out:
            out[pid] = normalize_action(act)
    return out


def group_talk_debias_enabled() -> bool:
    return os.environ.get("GROUP_TALK_DEBIAS", "1") == "1"


def infer_posture_light(
    buffer_dets: list[list[dict]], person_ids: list[int]
) -> dict[int, str]:
    """standing | sitting a partir de forma de caja (siempre, para enriquecer etiquetas)."""
    aspects: dict[int, list[float]] = {p: [] for p in person_ids}
    heights: dict[int, list[float]] = {p: [] for p in person_ids}
    for row in buffer_dets:
        for d in row:
            pid = d.get("pid")
            if pid not in aspects:
                continue
            bh = d["y2"] - d["y1"]
            bw = d["x2"] - d["x1"]
            if bh > 20:
                aspects[pid].append(bw / bh)
                heights[pid].append(bh)
    med_h = 0.0
    all_h = [h for hs in heights.values() for h in hs]
    if all_h:
        med_h = sorted(all_h)[len(all_h) // 2]
    sit_aspect = float(os.environ.get("POSTURE_SIT_ASPECT", "0.80"))
    sit_h_frac = float(os.environ.get("POSTURE_SIT_HEIGHT_FRAC", "0.90"))
    out: dict[int, str] = {}
    for pid in person_ids:
        asp_vals = aspects.get(pid) or []
        h_vals = heights.get(pid) or []
        if not asp_vals:
            out[pid] = "standing"
            continue
        asp = sum(asp_vals) / len(asp_vals)
        h = sum(h_vals) / len(h_vals)
        if med_h > 0 and asp >= sit_aspect and h <= sit_h_frac * med_h:
            out[pid] = "sitting"
        else:
            out[pid] = "standing"
    return out


def enrich_action_posture(
    actions: dict[int, str],
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    *,
    legs_visible: dict[int, bool] | None = None,
) -> dict[int, str]:
    """Antepone la postura ('standing'/'sitting') SOLO cuando el VLM no la
    mencionó (p. ej. dijo solo 'talking' o 'eating' -> 'standing and eating').

    Si el texto YA trae su propia postura (p. ej. 'sit and eat', 'standing
    and talking' — lo normal ahora que el prompt pide describir la actividad
    real), se deja tal cual. ANTES esto se reconstruía siempre desde un set
    cerrado de palabras clave (walk/run/sit/stand/talk/smile/phone), y
    cualquier actividad fuera de ese set se perdía en silencio (medido:
    "eating" quedaba reducido a solo "standing", con la persona sentada
    comiendo reportada como si solo estuviera de pie).

    `legs_visible` (pipeline de pose, ver pose_pipeline.chunk_legs_visible_by_pid):
    si dice explícitamente False para un pid (piernas fuera de cuadro, p. ej.
    muy cerca de la cámara), NO se antepone ninguna postura adivinada — se
    confía en el texto del VLM tal cual, siguiendo la misma lógica que el
    prompt (no adivinar lo que no se puede ver). Ausente del dict (sin pose,
    o sin dato de piernas ese trozo) cae al comportamiento de siempre
    (adivinar por forma de caja vía infer_posture_light)."""
    if not buffer_dets:
        return actions
    postures = infer_posture_light(buffer_dets, person_ids)
    out: dict[int, str] = {}
    for pid in person_ids:
        act = normalize_action(actions.get(pid, "unknown"))
        if act == "unknown":
            out[pid] = act
            continue
        tr = _action_traits(act.lower())
        if tr["walking"] or tr["running"] or tr["sitting"] or tr["standing"]:
            out[pid] = act
            continue
        if legs_visible is not None and legs_visible.get(pid) is False:
            out[pid] = act
            continue
        pos = postures.get(pid, "standing")
        out[pid] = f"{pos} and {act}"
    for pid, act in actions.items():
        if pid not in out:
            out[pid] = normalize_action(act)
    return out


def debias_group_talking(
    actions: dict[int, str],
    social: dict[int, str],
    person_ids: list[int],
    buffer_dets: list[list[dict]],
) -> tuple[dict[int, str], dict[int, str]]:
    """
    El VLM suele poner talk a todo el grupo. Quita talk a oyentes / escena completa.

    Solo se aplica con 3+ personas: con exactamente 2, "ambas hablando" suele
    ser una conversación real (el caso más común en los videos de prueba) y
    no un sesgo del VLM, así que no se debe rebajar de ENGAGED a ATTENTIVE.
    Tampoco se toca nunca a quien el movimiento real ya confirmó caminando
    (esa señal manda sobre la postura inferida aquí).
    """
    if not group_talk_debias_enabled() or len(person_ids) < 3 or not buffer_dets:
        return actions, social
    out_a = {pid: normalize_action(actions.get(pid, "unknown")) for pid in person_ids}
    out_s = dict(social)
    postures = infer_posture_light(buffer_dets, person_ids)
    n = len(person_ids)

    def is_talking(pid: int) -> bool:
        return _action_traits(out_a.get(pid, "").lower())["talking"]

    def is_walking(pid: int) -> bool:
        return _action_traits(out_a.get(pid, "").lower())["walking"]

    talkers = [p for p in person_ids if is_talking(p)]
    if len(talkers) < 2:
        return out_a, out_s

    # Sentado en escena con muchos talk → oyente, no hablante
    if len(talkers) >= n - 1:
        for pid in talkers:
            if is_walking(pid):
                continue
            if postures.get(pid) == "sitting":
                out_a[pid] = "sitting"
                if normalize_social_state(str(out_s.get(pid, ""))) == SOCIAL_ENGAGED:
                    out_s[pid] = SOCIAL_ATTENTIVE

    talkers = [p for p in person_ids if is_talking(p)]
    if len(talkers) >= n:
        for pid in person_ids:
            if is_walking(pid):
                continue
            out_a[pid] = postures.get(pid, "standing")
            soc = normalize_social_state(str(out_s.get(pid, "")))
            if soc in (SOCIAL_ENGAGED, SOCIAL_UNKNOWN):
                out_s[pid] = SOCIAL_ATTENTIVE
    elif len(talkers) >= n - 1:
        for pid in talkers:
            if is_walking(pid):
                continue
            soc = normalize_social_state(str(out_s.get(pid, "")))
            if soc in (SOCIAL_AVAILABLE, SOCIAL_ATTENTIVE):
                out_a[pid] = postures.get(pid, "standing")

    for pid in actions:
        if pid not in out_a:
            out_a[pid] = normalize_action(actions[pid])
    return out_a, out_s


def refine_chunk_labels(
    actions: dict[int, str],
    social: dict[int, str],
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    frame_size: tuple[int, int],
    *,
    moving_pids: set[int] | None = None,
    legs_visible: dict[int, bool] | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Post-proceso VLM: cinemática, postura, debias de talk grupal, social coherente.

    `moving_pids`/`legs_visible`: señales del pipeline de pose (marcha real +
    piernas visibles), ver apply_chunk_kinematic_hints/enrich_action_posture.
    Sin pose (pipeline clásico), ambos quedan en None y el comportamiento es
    el de siempre."""
    acts = apply_chunk_kinematic_hints(
        actions, buffer_dets, person_ids, frame_size, moving_pids=moving_pids
    )
    acts, social = debias_group_talking(acts, social, person_ids, buffer_dets)
    acts = enrich_action_posture(acts, buffer_dets, person_ids, legs_visible=legs_visible)
    social = resolve_social_states(acts, social, person_ids)
    return acts, social


_SINGLE_WORD_ACTION_CANON: dict[str, str] = {
    "sit": "sitting", "sits": "sitting", "sitting": "sitting", "seated": "sitting",
    "stand": "standing", "stands": "standing", "stood": "standing", "standing": "standing",
    "walk": "walking", "walks": "walking", "walked": "walking", "walking": "walking",
    "run": "running", "runs": "running", "running": "running",
    "talk": "talking", "talks": "talking", "talking": "talking",
}


def normalize_action(text: str) -> str:
    """Limpia espacios/puntuación y canonicaliza variantes de tense/plural de
    una sola palabra (p. ej. 'stood' -> 'standing'). Cualquier frase más
    descriptiva se deja TAL CUAL (p. ej. 'sit and eat', 'eating a sandwich',
    'sitting and reading') — antes se reconstruía siempre desde un set
    cerrado de palabras clave (walk/run/sit/stand/talk/smile/phone) y
    cualquier actividad fuera de ese set se perdía en silencio (medido:
    "eating" quedaba reducido a solo "standing")."""
    t = " ".join(str(text).strip().split())
    if not t:
        return "unknown"
    t = t.rstrip(".,;:")[:80]
    if _is_id_label(t):
        return "unknown"
    low = t.lower()
    if low in _SINGLE_WORD_ACTION_CANON:
        return _SINGLE_WORD_ACTION_CANON[low]
    return t


def _is_social_vocabulary(word: str) -> bool:
    """True si la palabra es un estado social, no una acción observable."""
    low = word.lower().strip()
    if low in ("available", "attentive", "engaged", "moving"):
        return True
    return low in _SOCIAL_ALIASES


def coerce_action_social(action: str, social: str) -> tuple[str, str]:
    """
    Si el VLM puso ENGAGED/ATTENTIVE/… en "action", moverlo a social y dejar action=unknown.
    No toca acciones reales (walking, talking, …).
    """
    act = normalize_action(action)
    soc = social
    if act != "unknown" and _is_social_vocabulary(act):
        inferred = normalize_social_state(act)
        if soc == SOCIAL_UNKNOWN and inferred != SOCIAL_UNKNOWN:
            soc = inferred
        act = "unknown"
    return act, soc


def action_to_social_state(action: str) -> str:
    """Mapea acción (simple o compuesta) → estado de interacción."""
    act = normalize_action(action)
    if not act or act == "unknown":
        return SOCIAL_UNKNOWN
    if act in _ACTION_TO_SOCIAL:
        return _ACTION_TO_SOCIAL[act]
    low = act.lower()
    tr = _action_traits(low)
    if tr["walking"] or tr["running"]:
        return SOCIAL_MOVING
    if tr["talking"]:
        return SOCIAL_ENGAGED
    if tr["phone"]:
        return SOCIAL_BUSY
    if tr["smiling"]:
        return SOCIAL_ATTENTIVE
    if tr["sitting"] and not tr["standing"]:
        return SOCIAL_AVAILABLE
    if tr["standing"]:
        return SOCIAL_AVAILABLE
    if any(k in low for k in ("read", "work", "laptop", "book")):
        return SOCIAL_BUSY
    if any(k in low for k in ("greet", "wave", "look", "watch", "eye")):
        return SOCIAL_ATTENTIVE
    return SOCIAL_UNKNOWN


def reconcile_social_with_action(action: str, social: str) -> str:
    """Alinea estado social con acción simple o compuesta."""
    act = normalize_action(action)
    if act == "unknown":
        return social if social != SOCIAL_UNKNOWN else SOCIAL_UNKNOWN
    low = act.lower()
    tr = _action_traits(low)
    if tr["walking"] or tr["running"]:
        # ENGAGED gana sobre MOVING si el VLM ya juzgó (en su campo 's',
        # independiente del texto de 'a') que hay una interacción real en
        # curso — caminar mientras se conversa/saluda a alguien es
        # interacción, no solo tránsito.
        if social == SOCIAL_ENGAGED:
            return SOCIAL_ENGAGED
        return SOCIAL_MOVING
    if tr["talking"]:
        return SOCIAL_ENGAGED
    if tr["phone"]:
        return SOCIAL_BUSY
    if tr["smiling"]:
        return SOCIAL_ATTENTIVE
    if tr["sitting"] and not tr["standing"]:
        # Igual que la rama "standing" de abajo: sentado sin una actividad
        # reconocida (talking/phone/smiling ya se descartaron arriba) no debe
        # pisar un estado social válido que el VLM ya haya dado (p. ej. BUSY
        # para "sit and eat") — antes esto forzaba AVAILABLE siempre, perdiendo
        # cualquier actividad fuera del set walk/talk/phone/smile.
        if social in (SOCIAL_ENGAGED, SOCIAL_ATTENTIVE, SOCIAL_BUSY):
            return social
        return social if social != SOCIAL_UNKNOWN else SOCIAL_AVAILABLE
    if tr["standing"]:
        # talking/smiling ya se descartaron arriba (si alguno fuera True, ya
        # habríamos retornado) — no hace falta repetir esos dos checks aquí.
        if social == SOCIAL_MOVING:
            return SOCIAL_AVAILABLE
        if social in (SOCIAL_ENGAGED, SOCIAL_ATTENTIVE, SOCIAL_BUSY):
            return social
        return social if social != SOCIAL_UNKNOWN else SOCIAL_AVAILABLE
    derived = action_to_social_state(act)
    if derived != SOCIAL_UNKNOWN:
        return derived
    return social if social != SOCIAL_UNKNOWN else SOCIAL_UNKNOWN


def social_interaction_hint(state: str) -> str:
    return _SOCIAL_SUITABILITY.get(state, "?")


def social_states_from_actions(actions: dict[int, str]) -> dict[int, str]:
    return {pid: action_to_social_state(act) for pid, act in actions.items()}


_COMPACT_SOCIAL: dict[str, str] = {
    "A": SOCIAL_AVAILABLE,
    "T": SOCIAL_ATTENTIVE,
    "B": SOCIAL_BUSY,
    "E": SOCIAL_ENGAGED,
    "M": SOCIAL_MOVING,
}


def normalize_social_state(text: str) -> str:
    """ATTENTIVE | AVAILABLE | BUSY | ENGAGED | MOVING o abreviatura A|T|B|E|M."""
    t = " ".join(str(text).strip().split()).upper().rstrip(".,;:")
    if not t:
        return SOCIAL_UNKNOWN
    if t in _COMPACT_SOCIAL:
        return _COMPACT_SOCIAL[t]
    if t in (
        SOCIAL_AVAILABLE,
        SOCIAL_ATTENTIVE,
        SOCIAL_BUSY,
        SOCIAL_ENGAGED,
        SOCIAL_MOVING,
    ):
        return t
    low = t.lower().replace("-", " ").replace("_", " ")
    if low in _SOCIAL_ALIASES:
        return _SOCIAL_ALIASES[low]
    for word in low.split():
        if word in _SOCIAL_ALIASES:
            return _SOCIAL_ALIASES[word]
    return SOCIAL_UNKNOWN


def resolve_social_states(
    actions: dict[int, str],
    parsed_social: dict[int, str] | None,
    person_ids: list[int] | None = None,
) -> dict[int, str]:
    """vlm: usa parsed_social; map: deriva de acciones."""
    pids = person_ids if person_ids is not None else list(actions.keys())
    if social_state_mode() == "map":
        base = social_states_from_actions(actions)
        return {pid: base.get(pid, SOCIAL_UNKNOWN) for pid in pids}
    parsed = parsed_social or {}
    out: dict[int, str] = {}
    for pid in pids:
        act = actions.get(pid, "unknown")
        s = normalize_social_state(parsed.get(pid, ""))
        if s == SOCIAL_UNKNOWN and social_vlm_fallback_map():
            s = action_to_social_state(act)
        if social_reconcile_action():
            s = reconcile_social_with_action(act, s)
        out[pid] = s
    return out


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
        return VLM_PROMPTS.COMPACT_WITH_SOCIAL.format(
            clip=clip, id_map=id_map, example=example, keys=keys
        )
    example = ",".join(f'"{p}":"<action>"' for p in pids[:2])
    return VLM_PROMPTS.COMPACT_NO_SOCIAL.format(clip=clip, id_map=id_map, example=example, keys=keys)


def _vlm_prompt_full(dets: list[dict], *, video: bool) -> str:
    pids = sorted({d["pid"] for d in dets})
    mapping = "\n".join(
        f'- Box label {pid_label(d["pid"])} (person inside that box) → JSON key "{d["pid"]}"'
        for d in dets
    )
    ctx = "this short video clip" if video else "this image"
    motion_hint = VLM_PROMPTS.MOTION_HINT_VIDEO if video else ""
    keys = ", ".join(str(p) for p in pids)
    if use_vlm_social_states():
        slots = ",".join(
            f'"{p}":{{"action":"<what they do>","social":"<STATE>"}}' for p in pids
        )
        return VLM_PROMPTS.FULL_WITH_SOCIAL.format(
            ctx=ctx,
            n_people=len(pids),
            motion_hint=motion_hint,
            mapping=mapping,
            slots=slots,
            keys=keys,
        )
    slots = ",".join(f'"{p}":"<action>"' for p in pids)
    return VLM_PROMPTS.FULL_NO_SOCIAL.format(
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
        return VLM_PROMPTS.NO_PEOPLE
    if vlm_prompt_mode() == "compact":
        return _vlm_prompt_compact(ordered, video=video)
    return _vlm_prompt_full(ordered, video=video)


def actions_for_ids(actions: dict[int, str], person_ids: list[int]) -> dict[int, str]:
    """Devuelve acciones solo para los IDs pedidos (descarta claves extra del modelo)."""
    return {pid: actions.get(pid, "unknown") for pid in person_ids}


def resize_max_side(pil: Image.Image, max_side: int) -> Image.Image:
    w, h = pil.size
    m = max(w, h)
    if m <= max_side:
        return pil
    s = max_side / m
    return pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)


def _vlm_max_new_tokens(n_people: int) -> int:
    _, _, max_new_cap = vlm_resize_limits()
    if vlm_prompt_mode() == "compact":
        extra = 10 if use_vlm_social_states() else 0
        budget = max(28, 10 + 9 * n_people + extra)
    else:
        extra = 28 if use_vlm_social_states() else 0
        budget = max(40, 20 + 14 * n_people + extra)
    return min(max_new_cap, budget)


def prioritize_movement_action(action: str) -> str:
    """Para navegación/HRI, si la persona se está desplazando esa señal debe
    dominar el estado social (MOVING) sobre cualquier otra acción secundaria
    del mismo trozo: al robot le importa primero si hay alguien en movimiento
    delante. La intención secundaria (p. ej. hablar) NO se descarta —se
    mantiene anexada en la etiqueta (p.ej. "walking and talking")— por si es
    relevante para decidir a quién puede acercarse el robot después."""
    if not action:
        return action
    tr = _action_traits(action.lower())
    base = "running" if tr.get("running") else "walking" if tr.get("walking") else None
    if base is None:
        return action
    extras = _action_secondary_parts(tr)
    return " and ".join([base] + extras) if extras else base


def _postprocess_vlm_raw(
    raw: str, person_ids: list[int], elapsed: float
) -> VlmInferenceResult:
    raw = "{" + raw.lstrip("{").strip()
    parsed_actions, parsed_social = parse_vlm_response(raw, person_ids)
    coerced_actions: dict[int, str] = {}
    coerced_social: dict[int, str] = {}
    for pid in person_ids:
        a, s = coerce_action_social(
            parsed_actions.get(pid, "unknown"),
            parsed_social.get(pid, SOCIAL_UNKNOWN),
        )
        a = prioritize_movement_action(a)
        coerced_actions[pid] = a
        coerced_social[pid] = s
    actions = actions_for_ids(coerced_actions, person_ids)
    social = resolve_social_states(actions, coerced_social, person_ids)
    if VLM_DEBUG or all(a == "unknown" for a in actions.values()):
        print(f"  VLM raw: {raw!r}", flush=True)
        print(
            f"  VLM IDs {person_ids} → actions={actions} social={social} "
            f"(mode={social_state_mode()})",
            flush=True,
        )
    return VlmInferenceResult(actions, social, elapsed)


def _generate_json(
    model, processor, messages: list, person_ids: list[int]
) -> VlmInferenceResult:
    max_new = _vlm_max_new_tokens(len(person_ids))
    dev = next(model.parameters()).device
    chat = processor.apply_chat_template(
        messages + [{"role": "assistant", "content": "{"}],
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    imgs, vids = process_vision_info(messages) if process_vision_info else (None, None)
    t0 = time.perf_counter()
    inputs = processor(
        text=[chat], images=imgs, videos=vids, padding=True, return_tensors="pt"
    ).to(dev)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, use_cache=True
        )
    raw = processor.batch_decode(
        [o[len(i) :] for i, o in zip(inputs.input_ids, out)],
        skip_special_tokens=True,
    )[0]
    elapsed = time.perf_counter() - t0
    return _postprocess_vlm_raw(raw, person_ids, elapsed)


def _generate_json_batch(
    model, processor, batch_messages: list[list], batch_person_ids: list[list[int]]
) -> list[VlmInferenceResult]:
    """Como _generate_json pero procesa varios clips en una sola pasada del modelo.

    La GPU está infrautilizada con batch=1 en este modelo pequeño (~2-2.3x más
    rápido por trozo con batch=4-6, medido empíricamente); agrupar trozos ya
    disponibles en disco (modo offline) reduce el tiempo total sin tocar la
    calidad del resultado por trozo.
    """
    if len(batch_messages) == 1:
        return [_generate_json(model, processor, batch_messages[0], batch_person_ids[0])]
    max_new = max(_vlm_max_new_tokens(len(pids)) for pids in batch_person_ids)
    dev = next(model.parameters()).device
    chats = [
        processor.apply_chat_template(
            m + [{"role": "assistant", "content": "{"}],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        for m in batch_messages
    ]
    videos: list = []
    for m in batch_messages:
        _, vids = process_vision_info(m) if process_vision_info else (None, None)
        videos.extend(vids or [])
    t0 = time.perf_counter()
    inputs = processor(
        text=chats, images=None, videos=videos, padding=True, return_tensors="pt"
    ).to(dev)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, use_cache=True
        )
    raws = processor.batch_decode(
        [o[len(i) :] for i, o in zip(inputs.input_ids, out)],
        skip_special_tokens=True,
    )
    elapsed_each = (time.perf_counter() - t0) / len(batch_messages)
    return [
        _postprocess_vlm_raw(raw, pids, elapsed_each)
        for raw, pids in zip(raws, batch_person_ids)
    ]


def merge_action_votes(votes: list[dict[int, str]]) -> dict[int, str]:
    from collections import Counter

    if not votes:
        return {}
    all_pids = set().union(*(v.keys() for v in votes))
    out: dict[int, str] = {}
    for pid in all_pids:
        c = Counter(
            v[pid] for v in votes if pid in v and v[pid] and v[pid] != "unknown"
        )
        out[pid] = c.most_common(1)[0][0] if c else "unknown"
    return out


def actions_for_video_frame(
    frame_i: int, keyframe_actions: list[tuple[int, dict[int, str]]]
) -> dict[int, str]:
    """Acumula acciones de keyframes ya vistos (cada keyframe puede actualizar IDs)."""
    if not keyframe_actions:
        return {}
    out: dict[int, str] = {}
    for kfi, acts in keyframe_actions:
        if kfi <= frame_i:
            for pid, act in acts.items():
                if act and act != "unknown":
                    out[pid] = act
    return out


def social_for_video_frame(
    frame_i: int, keyframe_social: list[tuple[int, dict[int, str]]]
) -> dict[int, str]:
    """Igual que actions_for_video_frame pero para estados sociales."""
    if not keyframe_social:
        return {}
    out: dict[int, str] = {}
    for kfi, states in keyframe_social:
        if kfi <= frame_i:
            for pid, st in states.items():
                if st and st != SOCIAL_UNKNOWN:
                    out[pid] = st
    return out


def infer_actions_image(
    model,
    processor,
    frame_bgr: np.ndarray,
    dets: list[dict],
    *,
    vlm_input_path: Path | None = None,
    already_annotated: bool = False,
) -> VlmInferenceResult:
    dets = sort_dets_left_right([d for d in dets if "pid" in d])
    person_ids = [d["pid"] for d in dets]
    if not person_ids:
        return VlmInferenceResult({}, {}, 0.0)
    annotated = frame_bgr if already_annotated else annotate_for_vlm(frame_bgr, dets)
    if vlm_input_path is not None:
        vlm_input_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vlm_input_path), annotated)
    pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    last_err = None
    for max_side in vlm_resize_limits()[0]:
        pil_r = resize_max_side(pil, max_side)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_r},
                    {"type": "text", "text": vlm_prompt(dets, video=False)},
                ],
            }
        ]
        try:
            return _generate_json(model, processor, messages, person_ids)
        except Exception as e:
            last_err = e
            if "memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                continue
            raise
    if last_err:
        print(f"  VLM error: {last_err}", flush=True)
    return VlmInferenceResult(
        {pid: "unknown" for pid in person_ids},
        {pid: SOCIAL_UNKNOWN for pid in person_ids},
        0.0,
    )


def infer_actions_chunk(
    model,
    processor,
    *,
    chunk_sec: float,
    frames: list[np.ndarray],
    person_ids: list[int],
    dets: list[dict],
    video_path: Path | None = None,
) -> VlmInferenceResult:
    """Trozo corto → 1 imagen (rápido); trozo largo → clip MP4."""
    if chunk_vlm_use_image(chunk_sec):
        mid = frames[len(frames) // 2]
        return infer_actions_image(
            model, processor, mid, dets, already_annotated=True
        )
    if video_path is None or not video_path.is_file():
        mid = frames[len(frames) // 2]
        return infer_actions_image(
            model, processor, mid, dets, already_annotated=True
        )
    return infer_actions_video(
        model, processor, video_path, person_ids, dets=dets, chunk_sec=chunk_sec
    )


def _read_video_frames_cv2(video_path: Path, target_fps: float) -> list[Image.Image]:
    """Decodifica el clip con OpenCV y lo re-muestrea a ~target_fps.

    Se usa en vez de dejar que qwen_vl_utils abra el .mp4 él mismo (decord/
    torchvision/torchcodec, en ese orden de preferencia): los tres backends
    exigen que la versión de ffmpeg del sistema y la de la wheel instalada
    coincidan exactamente (torchvision.io.read_video ya no existe en
    versiones nuevas; torchcodec/decord fallan por conflictos de libstdc++ —
    ver .venv/bin/activate). OpenCV ya es dependencia dura del resto del
    pipeline y decodifica el mismo archivo sin ese problema, así que evitamos
    la clase entera de fallos pasándole los frames ya decodificados
    (qwen_vl_utils acepta ``"video"`` como lista de imágenes, no solo ruta)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frames_bgr: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_bgr.append(frame)
    finally:
        cap.release()
    if not frames_bgr:
        return []
    total = len(frames_bgr)
    nframes = max(1, min(total, round(total / src_fps * target_fps))) if src_fps > 0 else total
    idx = np.linspace(0, total - 1, nframes).round().astype(int)
    return [Image.fromarray(cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)) for i in idx]


def _video_messages(
    video_path: Path, person_ids: list[int], dets: list[dict] | None, vfps: float
) -> list:
    prompt_dets = dets if dets else [{"pid": p, "x1": 0, "y1": 0, "x2": 0, "y2": 0} for p in person_ids]
    frames = _read_video_frames_cv2(video_path, vfps)
    if frames:
        video_content = {
            "type": "video",
            "video": frames,
            "max_pixels": VLM_VIDEO_MAX_PIXELS,
        }
    else:
        # Último recurso (no debería pasar: cv2 ya escribió este mismo
        # archivo) — deja que qwen_vl_utils lo intente por su cuenta.
        video_content = {
            "type": "video",
            "video": str(video_path.resolve()),
            "fps": vfps,
            "max_pixels": VLM_VIDEO_MAX_PIXELS,
        }
    return [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": vlm_prompt(prompt_dets, video=True)},
            ],
        }
    ]


def infer_actions_video(
    model,
    processor,
    video_path: Path,
    person_ids: list[int],
    *,
    dets: list[dict] | None = None,
    chunk_sec: float | None = None,
) -> VlmInferenceResult:
    """VLM sobre clip MP4 anotado (muestreo temporal con video_vlm_fps)."""
    vfps = video_vlm_fps(chunk_sec if chunk_sec is not None else 1.0)
    messages = _video_messages(video_path, person_ids, dets, vfps)
    return _generate_json(model, processor, messages, person_ids)


def infer_actions_video_batch(
    model,
    processor,
    items: list[tuple[Path, list[int], list[dict] | None]],
    *,
    chunk_sec: float | None = None,
) -> list[VlmInferenceResult]:
    """Batched infer_actions_video: agrupa varios trozos ya extraídos en una
    sola llamada a generate() (offline/batch, todos los clips ya existen en
    disco). Si el batch falla (p.ej. OOM), reintenta trozo a trozo."""
    if not items:
        return []
    vfps = video_vlm_fps(chunk_sec if chunk_sec is not None else 1.0)
    batch_messages = [_video_messages(vp, pids, dets, vfps) for vp, pids, dets in items]
    batch_person_ids = [pids for _, pids, _ in items]
    try:
        return _generate_json_batch(model, processor, batch_messages, batch_person_ids)
    except Exception as e:
        if "memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
            raise
        torch.cuda.empty_cache()
        print(f"  Aviso: batch VLM falló ({e}); reintentando trozo a trozo …", flush=True)
        return [
            infer_actions_video(model, processor, vp, pids, dets=dets, chunk_sec=chunk_sec)
            for vp, pids, dets in items
        ]


def merge_social_votes(votes: list[dict[int, str]]) -> dict[int, str]:
    from collections import Counter

    if not votes:
        return {}
    all_pids = set().union(*(v.keys() for v in votes))
    out: dict[int, str] = {}
    for pid in all_pids:
        c = Counter(
            v[pid]
            for v in votes
            if pid in v and v[pid] and v[pid] != SOCIAL_UNKNOWN
        )
        out[pid] = c.most_common(1)[0][0] if c else SOCIAL_UNKNOWN
    return out


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


def warmup_vlm(model, processor) -> None:
    infer_actions_image(
        model,
        processor,
        np.zeros((240, 320, 3), dtype=np.uint8),
        [{"pid": 1, "x1": 0, "y1": 0, "x2": 80, "y2": 120}],
    )
