"""Orquestación de la inferencia VLM: prompts -> generate() -> parseo -> reconciliación."""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None  # type: ignore[misc, assignment]

from ..actions import prioritize_movement_action
from ..config import (
    VLM_DEBUG,
    VLM_VIDEO_MAX_PIXELS,
    chunk_vlm_use_image,
    vlm_prompt_mode,
    vlm_resize_limits,
    video_vlm_fps,
)
from ..detection import sort_dets_left_right
from ..drawing import annotate_for_vlm
from ..social_state import (
    SOCIAL_UNKNOWN,
    coerce_action_social,
    resolve_social_states,
    social_state_mode,
    use_vlm_social_states,
)
from .parsing import parse_vlm_response
from .prompts import vlm_prompt


class VlmInferenceResult(NamedTuple):
    actions: dict[int, str]
    social_states: dict[int, str]
    elapsed_s: float


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


def warmup_vlm(model, processor) -> None:
    infer_actions_image(
        model,
        processor,
        np.zeros((240, 320, 3), dtype=np.uint8),
        [{"pid": 1, "x1": 0, "y1": 0, "x2": 80, "y2": 120}],
    )
