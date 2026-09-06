"""Constantes de configuración (variables de entorno) y bootstrap de proceso.

Debe ser el PRIMER módulo del paquete en importarse (lo garantiza
``vlm_hri/__init__.py``): fija los límites de hilos de CPU antes de que
cualquier otro módulo importe ``torch``/``cv2`` por primera vez.
"""

from __future__ import annotations

import os
from pathlib import Path

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
import torch

torch.set_num_threads(_CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # ya se usó paralelismo interop en este proceso; no se puede cambiar
cv2.setNumThreads(_CPU_THREADS)

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
for _p in (ROOT / ".hf_cache", ROOT / ".mplconfig"):
    _p.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS = str(ROOT / "models" / "yolo11n.pt")
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

POSE_YOLO_WEIGHTS = str(ROOT / "models" / "yolo26n-pose.pt")
