"""Pipeline en streaming (cámara o video en vivo) CON POSE: detección+pose
cada frame, marcha/profundidad/histéresis/compensación de cámara/ATTENTIVE
por mirada/target por trozo, VLM en hilo aparte para no bloquear la captura.

Copia de run_stream.py adaptada a pose_pipeline.py — módulo APARTE, no toca
run_stream.py/run_video.py/pipeline.py. Reutiliza de run_stream.py todo lo que
no cambia (vista previa mpv/ffplay/http, ventana OpenCV, cola acotada) y de
run_video_pose.py la lógica de refinamiento por trozo (pose+gait+debias).

  python main_pose.py stream -i input_videos/walking.mp4 --preview
  python main_pose.py stream --camera 0 --display
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
from ultralytics import YOLO

import pipeline as P
import pose_pipeline as PP
import run_video_pose as RVP
from run_video import (
    _crop_frame,
    _even_dims,
    _realtime_metrics,
    ensure_opencv_video,
    finalize_video_h264,
    print_vlm_latency_report,
    vlm_timing_stats,
)
from run_stream import (
    HttpPreview,
    LivePreview,
    _build_segment_summary,
    _graphical_session_ok,
    _open_live_preview,
    _opencv_gui_available,
    _preview_player_available,
    _resolve_display,
    _safe_destroy_windows,
    _submit_chunk,
    _write_chunk_video,
)

ROOT = P.ROOT

STREAM_OUTPUT_DIR = Path(os.environ.get("STREAM_POSE_OUTPUT_DIR", str(ROOT / "output")))

# Factor de escala SOLO para la ventana de --display (cv2.imshow) — no afecta
# la resolución real usada por YOLO/VLM ni lo que se guarda a disco.
DISPLAY_SCALE = float(os.environ.get("DISPLAY_SCALE", "2.0"))


@dataclass
class RawFrame:
    """Frame crudo + detecciones (con keypoints) tal cual salen de Fase 1,
    ANTES de saber azul/rojo (eso solo se sabe al cerrar el trozo)."""
    frame_i: int
    frame_bgr: np.ndarray
    dets: list[dict]


@dataclass
class StreamOutput:
    writer: cv2.VideoWriter
    lock: threading.Lock
    live_preview: LivePreview | None
    w: int
    h: int
    fps: float
    chunk_sec: float
    t_pipeline0: float
    state: StreamState


@dataclass
class VlmChunkJob:
    chunk_index: int
    start_frame: int
    raw_frames: list[RawFrame]
    person_ids: list[int]
    out_path: Path
    fps: float
    size: tuple[int, int]


@dataclass
class StreamState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    actions: dict[int, str] = field(default_factory=dict)
    social_states: dict[int, str] = field(default_factory=dict)
    segment_actions: list[tuple[int, dict[int, str]]] = field(default_factory=list)
    segment_social: list[tuple[int, dict[int, str]]] = field(default_factory=list)
    vlm_chunk_timings: list[dict] = field(default_factory=list)
    vlm_total_s: float = 0.0
    vlm_calls: int = 0
    vlm_pending: bool = False
    yolo_total_s: float = 0.0
    frames_done: int = 0


def _render_output_frame(
    rf: RawFrame,
    actions: dict[int, str],
    social_states: dict[int, str],
    *,
    w: int,
    h: int,
    fps: float,
    chunk_sec: float,
    chunk_index: int,
    chunk_count: int,
    yolo_total_s: float,
    vlm_total_s: float,
    vlm_calls: int,
    pipeline_total_s: float,
    vlm_pending: bool,
) -> np.ndarray:
    """Video final para el usuario (no el que ve el VLM): caja de color + ID +
    panel lateral + resalte del target — igual estilo que run_video_pose.py."""
    ann = rf.frame_bgr.copy()
    target_pid = PP.pick_interaction_target(
        rf.dets, {d["pid"]: social_states.get(d["pid"]) for d in rf.dets}
    )
    panel_entries: list[tuple[int, str, tuple[int, int, int]]] = []
    for d in rf.dets:
        social_state = social_states.get(d["pid"])
        color = P.social_box_color(social_state)
        PP.draw_box_only(ann, d["x1"], d["y1"], d["x2"], d["y2"], d["pid"], color)
        act = P.normalize_action(actions.get(d["pid"], ""))
        state_txt = social_state or "UNKNOWN"
        text = f"{state_txt} ({act})" if act and act != "unknown" else state_txt
        if d["pid"] == target_pid:
            text += " -Target-"
            cv2.rectangle(
                ann, (d["x1"] - 3, d["y1"] - 3), (d["x2"] + 3, d["y2"] + 3),
                (0, 255, 255), 2,
            )
        panel_entries.append((d["pid"], text, color))
    PP.draw_side_panel(ann, panel_entries)
    P.draw_banner_video(
        ann,
        yolo_total_s=yolo_total_s,
        n_frames=max(rf.frame_i, 1),
        vlm_total_s=vlm_total_s,
        vlm_calls=max(vlm_calls, 1),
        n_people=len({d["pid"] for d in rf.dets}),
        pipeline_total_s=pipeline_total_s,
        frame_i=rf.frame_i,
        vlm_input_kind="chunks",
        chunk_sec=chunk_sec,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
    )
    if vlm_pending:
        cv2.putText(
            ann, "VLM...",
            (int(10 * P.ui_scale(h, w)), int(h - 30 * P.ui_scale(h, w))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * P.ui_scale(h, w), (255, 200, 255), 1, cv2.LINE_8,
        )
    return ann


def _flush_chunk_to_output(job: VlmChunkJob, output: StreamOutput) -> None:
    with output.state.lock:
        actions = dict(output.state.actions)
        social_states = dict(output.state.social_states)
        yolo_acum = output.state.yolo_total_s
        vlm_acum = output.state.vlm_total_s
        vlm_n = output.state.vlm_calls
        vlm_pending = output.state.vlm_pending
    pipeline_now = time.perf_counter() - output.t_pipeline0
    chunk_count = max(job.chunk_index + 1, vlm_n)
    with output.lock:
        for rf in job.raw_frames:
            ann = _render_output_frame(
                rf, actions, social_states,
                w=output.w, h=output.h, fps=output.fps, chunk_sec=output.chunk_sec,
                chunk_index=job.chunk_index, chunk_count=chunk_count,
                yolo_total_s=yolo_acum, vlm_total_s=vlm_acum, vlm_calls=vlm_n,
                pipeline_total_s=pipeline_now, vlm_pending=vlm_pending,
            )
            output.writer.write(ann)
            if output.live_preview is not None:
                try:
                    output.live_preview.write(ann)
                except BrokenPipeError:
                    pass


def _vlm_worker(
    queue: Queue,
    vlm,
    processor,
    state: StreamState,
    stop: threading.Event,
    output: StreamOutput | None,
) -> None:
    motion_hyst = PP.MotionHysteresis()
    while True:
        if stop.is_set():
            try:
                job: VlmChunkJob = queue.get_nowait()
            except Empty:
                break
        else:
            try:
                job = queue.get(timeout=0.25)
            except Empty:
                continue
        with state.lock:
            state.vlm_pending = True

        buffer_dets = [rf.dets for rf in job.raw_frames]
        pids = job.person_ids
        gait = PP.chunk_gait_by_pid(buffer_dets, pids)
        depth = PP.chunk_depth_motion_by_pid(buffer_dets, pids)
        raw_moving = {p for p in pids if gait.get(p) or depth.get(p)}
        moving_pids = motion_hyst.update(job.chunk_index, raw_moving, set(pids))

        vlm_frames = [
            PP.annotate_for_vlm_pose(rf.frame_bgr, rf.dets, moving_pids) for rf in job.raw_frames
        ]
        dets_prompt = P.dets_for_vlm_prompt(buffer_dets, pids)

        chunk_sec = len(vlm_frames) / max(job.fps, 1e-6)
        t0 = time.perf_counter()
        try:
            if P.chunk_vlm_use_image(chunk_sec):
                mid = vlm_frames[len(vlm_frames) // 2]
                vlm_out = P.infer_actions_image(vlm, processor, mid, dets_prompt, already_annotated=True)
            elif _write_chunk_video(vlm_frames, job.out_path, job.fps, job.size):
                vlm_out = P.infer_actions_video(
                    vlm, processor, job.out_path, pids, dets=dets_prompt, chunk_sec=chunk_sec
                )
            else:
                mid = vlm_frames[len(vlm_frames) // 2]
                vlm_out = P.infer_actions_image(vlm, processor, mid, dets_prompt, already_annotated=True)
        except Exception as e:
            print(f"  [VLM trozo {job.chunk_index}] error: {e}", flush=True)
            vlm_out = P.VlmInferenceResult({}, {}, 0.0)

        acts = dict(vlm_out.actions)
        social = dict(vlm_out.social_states)
        if pids:
            # Misma lógica de refinamiento que el modo offline (run_video_pose.py):
            # debias del sesgo grupal de "walking", forzado a caminar si hay
            # evidencia independiente, respaldo por cinemática de caja
            # (con compensación de movimiento de cámara), y ATTENTIVE por mirada.
            acts, social = RVP.refine_labels_pose(
                acts, social, buffer_dets, pids, job.size, moving_pids_hint=moving_pids
            )
            social = RVP.upgrade_attentive_by_gaze(social, buffer_dets, pids)

        dt = time.perf_counter() - t0
        with state.lock:
            state.vlm_total_s += dt
            state.vlm_calls += 1
            state.vlm_pending = False
            for pid, act in acts.items():
                if act and act != "unknown":
                    state.actions[pid] = act
            for pid, st in social.items():
                if st and st != P.SOCIAL_UNKNOWN:
                    state.social_states[pid] = st
            state.segment_actions.append((job.start_frame, dict(acts)))
            state.segment_social.append((job.start_frame, dict(social)))
            state.vlm_chunk_timings.append(
                {"chunk_index": job.chunk_index, "start_frame": job.start_frame, "latency_s": round(vlm_out.elapsed_s, 4)}
            )
        if output is not None and job.raw_frames:
            _flush_chunk_to_output(job, output)
        print(
            f"  [VLM trozo {job.chunk_index}] f{job.start_frame}+ actions={acts} social={social} "
            f"(inferencia {vlm_out.elapsed_s:.2f}s, acum VLM {state.vlm_total_s:.2f}s)",
            flush=True,
        )


def run(
    *,
    video_path: Path | None = None,
    camera: int | None = None,
    output_dir: Path | None = None,
    display: bool = False,
    preview: bool = False,
    realtime: bool = False,
    max_frames: int | None = None,
    yolo: YOLO | None = None,
    yolo_pose: YOLO | None = None,
    vlm=None,
    processor=None,
    device=None,
) -> Path:
    if (video_path is None) == (camera is None):
        raise SystemExit("Indica --input VIDEO o --camera N (no ambos).")

    chunk_sec = float(os.environ.get("VIDEO_VLM_CHUNK_SEC", str(P.VIDEO_VLM_CHUNK_SEC)))
    if device is None:
        device = P.resolve_device()
    out_name = video_path.stem.replace("_h264", "") if video_path else f"camera_{camera}"
    out = Path(output_dir or STREAM_OUTPUT_DIR) / out_name
    out.mkdir(parents=True, exist_ok=True)
    chunk_dir = out / "vlm_chunks"
    chunk_dir.mkdir(exist_ok=True)

    if video_path is not None:
        video_path = ensure_opencv_video(Path(video_path))
        cap = cv2.VideoCapture(str(video_path))
        source = "video"
    else:
        cap = cv2.VideoCapture(int(camera))
        source = "camera"
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la fuente de video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if source == "camera":
        fps = float(os.environ.get("STREAM_CAMERA_FPS", "30"))
    w, h = _even_dims(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    chunk_frames = max(1, int(chunk_sec * fps))

    if display and not _opencv_gui_available():
        if _preview_player_available():
            preview, display = True, False
        else:
            display = False
    elif preview and not _preview_player_available():
        if os.environ.get("STREAM_PREVIEW_HTTP", "1") == "0":
            preview = False
    if preview:
        _resolve_display(quiet=False)

    print(f"Stream [pose]: {out_name} | fuente={source} | trozo={chunk_sec}s ({chunk_frames} fr) | salida: {out}", flush=True)

    if yolo is None:
        yolo = YOLO(P.YOLO_WEIGHTS)
    if yolo_pose is None:
        yolo_pose = YOLO(PP.POSE_YOLO_WEIGHTS)
    P.warmup_yolo(yolo, shape=(h, w, 3))
    P.warmup_yolo(yolo_pose, shape=(h, w, 3))
    if vlm is None or processor is None:
        print("Cargando VLM …", flush=True)
        t_load = time.perf_counter()
        vlm, processor = P.load_vlm(P.VLM_ID, device)
        if P.VLM_WARMUP:
            P.warmup_vlm(vlm, processor)
        print(f"  VLM listo en {time.perf_counter() - t_load:.1f}s", flush=True)

    state = StreamState()
    stop = threading.Event()
    out_mp4 = out / "annotated_stream.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    live_preview: LivePreview | None = None
    want_preview = preview
    preview_gave_up = False
    t_pipeline0 = time.perf_counter()
    writer_lock = threading.Lock()
    stream_output = StreamOutput(
        writer=writer, lock=writer_lock, live_preview=None,
        w=w, h=h, fps=fps, chunk_sec=chunk_sec, t_pipeline0=t_pipeline0, state=state,
    )
    vlm_queue: Queue = Queue(maxsize=8)
    worker = threading.Thread(
        target=_vlm_worker, args=(vlm_queue, vlm, processor, state, stop, stream_output), daemon=True
    )
    worker.start()

    prev_dets: list[dict] = []
    chunk_raw: list[RawFrame] = []
    frame_i = 0
    chunk_i = 0
    next_chunk_at = chunk_frames

    def submit_chunk() -> None:
        nonlocal chunk_raw, chunk_i, next_chunk_at
        if not chunk_raw:
            return
        pids = sorted({d["pid"] for rf in chunk_raw for d in rf.dets})[: PP.MAX_VLM_PEOPLE]
        _submit_chunk(
            vlm_queue,
            VlmChunkJob(
                chunk_index=chunk_i,
                start_frame=frame_i - len(chunk_raw),
                raw_frames=list(chunk_raw),
                person_ids=pids,
                out_path=chunk_dir / f"chunk_{chunk_i:04d}.mp4",
                fps=fps,
                size=(w, h),
            ),
            stream_output,
        )
        chunk_i += 1
        next_chunk_at += chunk_frames
        chunk_raw = []

    try:
        while True:
            t_fr0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_i >= max_frames:
                break

            frame = _crop_frame(frame, w, h)
            raw = P.detect_people(yolo, frame)
            dets = P.track_detections(prev_dets, raw) if prev_dets else P.assign_spatial_ids(raw)
            prev_dets = dets
            pose_raw = PP.detect_people_pose(yolo_pose, frame)
            PP.attach_pose_keypoints(dets, pose_raw)

            chunk_raw.append(RawFrame(frame_i, frame.copy(), [dict(d) for d in dets]))

            if display:
                with state.lock:
                    actions = dict(state.actions)
                    social_states = dict(state.social_states)
                    vlm_pending = state.vlm_pending
                    yolo_acum, vlm_acum, vlm_n = state.yolo_total_s, state.vlm_total_s, state.vlm_calls
                try:
                    preview_ann = _render_output_frame(
                        chunk_raw[-1], actions, social_states,
                        w=w, h=h, fps=fps, chunk_sec=chunk_sec, chunk_index=chunk_i, chunk_count=chunk_i + 1,
                        yolo_total_s=yolo_acum, vlm_total_s=vlm_acum, vlm_calls=vlm_n,
                        pipeline_total_s=time.perf_counter() - t_pipeline0, vlm_pending=vlm_pending,
                    )
                    display_ann = cv2.resize(
                        preview_ann, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    cv2.imshow("stream-pose", display_ann)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    display = False
                    print("  imshow no disponible; continuando solo guardando mp4 …", flush=True)

            if want_preview and not preview_gave_up:
                if live_preview is None:
                    live_preview = _open_live_preview(w, h, fps)
                    stream_output.live_preview = live_preview
                    if live_preview is None:
                        preview_gave_up = True
                        want_preview = False
                elif not live_preview.alive():
                    preview_gave_up = True
                    want_preview = False
                    stream_output.live_preview = None
                    print("  Ventana de vista previa cerrada; sigue el mp4 en disco.", flush=True)

            yolo_dt = time.perf_counter() - t_fr0
            with state.lock:
                state.yolo_total_s += yolo_dt
                state.frames_done = frame_i + 1
            frame_i += 1

            if frame_i >= next_chunk_at and len(chunk_raw) >= chunk_frames:
                submit_chunk()

            if realtime and source == "video":
                time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t_fr0)))
    finally:
        submit_chunk()
        stop.set()
        worker.join(timeout=300)
        cap.release()
        writer.release()
        if live_preview is not None:
            live_preview.close()
        if display:
            _safe_destroy_windows()

    finalize_video_h264(out_mp4)
    pipeline_total_s = time.perf_counter() - t_pipeline0

    with state.lock:
        segment_actions = list(state.segment_actions)
        segment_social = list(state.segment_social)
        vlm_chunk_timings = list(state.vlm_chunk_timings)
        final_actions = dict(state.actions)
        final_social = dict(state.social_states)
        vlm_total_s = state.vlm_total_s
        vlm_calls = state.vlm_calls
        yolo_total_s = state.yolo_total_s

    rt = _realtime_metrics(
        frames=frame_i, fps=fps, yolo_s=yolo_total_s, vlm_s=vlm_total_s,
        render_s=0.0, encode_s=0.0, pipeline_s=pipeline_total_s,
    )
    summary = {
        "source": source,
        "video": video_path.name if video_path else None,
        "camera": camera,
        "frames": frame_i,
        "fps": fps,
        "chunk_sec": chunk_sec,
        "vlm_chunks_submitted": chunk_i,
        "vlm_inferences": vlm_calls,
        "yolo_total_s": round(yolo_total_s, 4),
        "vlm_total_s": round(vlm_total_s, 4),
        "pipeline_total_s": round(pipeline_total_s, 4),
        **rt,
        "social_state_mode": P.social_state_mode(),
        "detector": "pose+gait (stream)",
        "actions_final": {str(k): v for k, v in final_actions.items()},
        "social_states_final": {str(k): v for k, v in final_social.items()},
        "vlm_chunk_timings": vlm_chunk_timings,
        "vlm_chunk_latencies_s": [t["latency_s"] for t in vlm_chunk_timings],
        **vlm_timing_stats([t["latency_s"] for t in vlm_chunk_timings]),
        "segments": _build_segment_summary(segment_actions, segment_social, vlm_chunk_timings),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    vname = video_path.name if video_path else f"camera_{camera}"
    print_vlm_latency_report(vname, vlm_chunk_timings)
    print(
        f"\nStream [pose] listo → {out} ({pipeline_total_s:.1f}s, {frame_i} frames, "
        f"VLM {vlm_calls} trozos, factor RT {rt['realtime_factor']:.2f}x)",
        flush=True,
    )
    print(f"  Ver resultado: {out_mp4}", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="YOLO+pose+VLM en streaming")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", type=Path, help="Video archivo (simula cámara)")
    src.add_argument("--camera", type=int, help="Índice cámara (0 = default)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--display", action="store_true", help="Ventana OpenCV")
    p.add_argument("--preview", action="store_true", help="Vista previa en vivo con ffplay/mpv")
    p.add_argument("--realtime", action="store_true", help="Con video: ritmo real (1/fps)")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()
    run(
        video_path=args.input, camera=args.camera, output_dir=args.output,
        display=args.display, preview=args.preview, realtime=args.realtime,
        max_frames=args.max_frames,
    )
