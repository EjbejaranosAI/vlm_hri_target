"""Pipeline en streaming (cámara o video en vivo): YOLO cada frame + VLM cada
~N segundos en un hilo aparte, para no bloquear la captura.

Un solo entrypoint `run()` para las dos variantes que existían como módulos
separados (run_stream.py / run_stream_pose.py): `use_pose=True` (default,
con marcha por piernas + ATTENTIVE por mirada + target de interacción, vía
yolo26n-pose) o `use_pose=False` (solo detección yolo11n). Igual que en
runners/video.py, las dos implementaciones divergen de verdad (dataclasses,
worker VLM, render final) así que se mantienen separadas
(`_run_plain`/`_run_pose`) en vez de forzar un solo cuerpo. Sin batch
(`run_all`) para el modo pose: nunca existió (run_stream.py sí lo tenía,
pero no era alcanzable desde ningún entrypoint documentado).

  python main.py stream -i input_videos/walking.mp4 --preview
  python main.py stream --camera 0 --display
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

from .. import preview as stream_preview, video_io
from ..actions import normalize_action
from ..config import (
    MAX_TRACK_IDS,
    POSE_YOLO_WEIGHTS,
    ROOT,
    VIDEO_VLM_CHUNK_SEC,
    VLM_ID,
    VLM_WARMUP,
    YOLO_WEIGHTS,
    chunk_vlm_use_image,
    video_vlm_fps,
    vlm_prompt_mode,
)
from ..detection import (
    assign_spatial_ids,
    detect_people,
    dets_for_vlm_prompt,
    resolve_device,
    track_detections,
    warmup_yolo,
)
from ..drawing import annotate_for_vlm, draw_banner_video, draw_box, format_detection_label, social_box_color, ui_scale
from ..motion import refine_chunk_labels
from ..pose import gait as pose_gait
from ..social_state import SOCIAL_UNKNOWN, social_state_mode, social_states_from_actions
from ..vlm.inference import VlmInferenceResult, infer_actions_image, infer_actions_video, warmup_vlm
from ..vlm.model import load_vlm

ROOT = ROOT
_STREAM_OUTPUT_DIR_PLAIN = Path(os.environ.get("STREAM_OUTPUT_DIR", str(ROOT / "output_stream")))
_STREAM_OUTPUT_DIR_POSE = Path(os.environ.get("STREAM_POSE_OUTPUT_DIR", str(ROOT / "output")))

# Factor de escala SOLO para la ventana de --display (cv2.imshow) del modo
# pose -- no afecta la resolución real usada por YOLO/VLM ni lo guardado a disco.
DISPLAY_SCALE = float(os.environ.get("DISPLAY_SCALE", "2.0"))


@dataclass
class DisplayFrame:
    """Frame crudo + detecciones; se escribe al mp4 cuando el VLM termina el trozo."""
    frame_i: int
    frame_bgr: np.ndarray
    dets: list[dict]


@dataclass
class StreamOutput:
    writer: cv2.VideoWriter
    lock: threading.Lock
    live_preview: stream_preview.LivePreview | None
    w: int
    h: int
    fps: float
    chunk_sec: float
    chunk_index: int
    t_pipeline0: float
    state: StreamState


@dataclass
class VlmChunkJob:
    chunk_index: int
    start_frame: int
    frames: list[np.ndarray]
    dets_mid: list[dict]
    person_ids: list[int]
    out_path: Path
    fps: float
    size: tuple[int, int]
    display_frames: list[DisplayFrame] = field(default_factory=list)
    buffer_dets: list[list[dict]] = field(default_factory=list)


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
    last_vlm_s: float = 0.0
    yolo_total_s: float = 0.0
    frames_done: int = 0


def _write_chunk_video(
    frames: list[np.ndarray], path: Path, fps: float, size: tuple[int, int]
) -> bool:
    w, h = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for fr in frames:
        writer.write(fr)
    writer.release()
    return path.is_file() and path.stat().st_size > 0


def _submit_chunk(
    queue: Queue, job: VlmChunkJob, output: StreamOutput | None = None
) -> None:
    """Cola acotada: si está llena, escribe el trozo más antiguo (etiquetas actuales) y lo quita."""
    while queue.full():
        try:
            old = queue.get_nowait()
            if output is not None and old.display_frames:
                _flush_chunk_to_output(old, output)
        except Empty:
            break
    queue.put(job)


def _render_output_frame(
    df: DisplayFrame,
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
    ann = df.frame_bgr.copy()
    for d in df.dets:
        draw_box(
            ann,
            d["x1"],
            d["y1"],
            d["x2"],
            d["y2"],
            _label_for_det(d, actions, social_states),
        )
    draw_banner_video(
        ann,
        yolo_total_s=yolo_total_s,
        n_frames=max(df.frame_i, 1),
        vlm_total_s=vlm_total_s,
        vlm_calls=max(vlm_calls, 1),
        n_people=len({d["pid"] for d in df.dets}),
        pipeline_total_s=pipeline_total_s,
        frame_i=df.frame_i,
        vlm_input_kind="chunks",
        chunk_sec=chunk_sec,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
    )
    if vlm_pending:
        cv2.putText(
            ann,
            "VLM...",
            (int(10 * ui_scale(h, w)), int(h - 30 * ui_scale(h, w))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * ui_scale(h, w),
            (255, 200, 255),
            1,
            cv2.LINE_8,
        )
    return ann


def _flush_chunk_to_output(job: VlmChunkJob, output: StreamOutput) -> None:
    """Escribe el trozo con las etiquetas del VLM (sincronizadas con ese segundo de vídeo)."""
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
        for df in job.display_frames:
            ann = _render_output_frame(
                df,
                actions,
                social_states,
                w=output.w,
                h=output.h,
                fps=output.fps,
                chunk_sec=output.chunk_sec,
                chunk_index=job.chunk_index,
                chunk_count=chunk_count,
                yolo_total_s=yolo_acum,
                vlm_total_s=vlm_acum,
                vlm_calls=vlm_n,
                pipeline_total_s=pipeline_now,
                vlm_pending=vlm_pending,
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
        chunk_sec = len(job.frames) / max(job.fps, 1e-6)
        t0 = time.perf_counter()
        try:
            if chunk_vlm_use_image(chunk_sec):
                mid = job.frames[len(job.frames) // 2]
                vlm_out = infer_actions_image(
                    vlm, processor, mid, job.dets_mid, already_annotated=True
                )
            elif _write_chunk_video(job.frames, job.out_path, job.fps, job.size):
                vlm_out = infer_actions_video(
                    vlm,
                    processor,
                    job.out_path,
                    job.person_ids,
                    dets=job.dets_mid,
                    chunk_sec=chunk_sec,
                )
            else:
                mid = job.frames[len(job.frames) // 2]
                vlm_out = infer_actions_image(
                    vlm, processor, mid, job.dets_mid, already_annotated=True
                )
        except Exception as e:
            print(f"  [VLM trozo {job.chunk_index}] error: {e}", flush=True)
            vlm_out = VlmInferenceResult({}, {}, 0.0)
        acts = dict(vlm_out.actions)
        social = dict(vlm_out.social_states)
        if job.buffer_dets and job.person_ids:
            acts, social = refine_chunk_labels(
                acts, social, job.buffer_dets, job.person_ids, job.size
            )
        dt = time.perf_counter() - t0
        with state.lock:
            state.vlm_total_s += dt
            state.vlm_calls += 1
            state.last_vlm_s = dt
            state.vlm_pending = False
            for pid, act in acts.items():
                if act and act != "unknown":
                    state.actions[pid] = act
            for pid, st in social.items():
                if st and st != SOCIAL_UNKNOWN:
                    state.social_states[pid] = st
            state.segment_actions.append((job.start_frame, dict(acts)))
            state.segment_social.append((job.start_frame, dict(social)))
            state.vlm_chunk_timings.append(
                {
                    "chunk_index": job.chunk_index,
                    "start_frame": job.start_frame,
                    "latency_s": round(vlm_out.elapsed_s, 4),
                }
            )
        if output is not None and job.display_frames:
            _flush_chunk_to_output(job, output)
        print(
            f"  [VLM trozo {job.chunk_index}] f{job.start_frame}+ "
            f"actions={acts} social={social} "
            f"(inferencia {vlm_out.elapsed_s:.2f}s, acum VLM {state.vlm_total_s:.2f}s, "
            f"mode={social_state_mode()})",
            flush=True,
        )


def _build_segment_summary(
    segment_actions: list[tuple[int, dict[int, str]]],
    segment_social: list[tuple[int, dict[int, str]]],
    vlm_chunk_timings: list[dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    timings = vlm_chunk_timings or []
    for i, (sf, acts) in enumerate(segment_actions):
        if i < len(segment_social):
            soc = segment_social[i][1]
        else:
            soc = social_states_from_actions(acts)
        row: dict = {
            "start_frame": sf,
            "actions": {str(k): v for k, v in acts.items()},
            "social_states": {str(k): v for k, v in soc.items()},
        }
        if i < len(timings):
            row["chunk_index"] = timings[i].get("chunk_index", i)
            row["vlm_latency_s"] = timings[i].get("latency_s")
        rows.append(row)
    return rows


def _label_for_det(
    d: dict, actions: dict[int, str], social_states: dict[int, str]
) -> str:
    pid = d["pid"]
    return format_detection_label(
        pid,
        actions.get(pid, ""),
        social_state=social_states.get(pid),
        class_name=d.get("class_name", "person"),
    )


def _run_plain(
    *,
    video_path: Path | None = None,
    camera: int | None = None,
    output_dir: Path | None = None,
    display: bool = False,
    preview: bool = False,
    realtime: bool = False,
    max_frames: int | None = None,
    yolo: YOLO | None = None,
    vlm=None,
    processor=None,
    device=None,
) -> Path:
    if (video_path is None) == (camera is None):
        raise SystemExit("Indica --input VIDEO o --camera N (no ambos).")

    chunk_sec = float(os.environ.get("VIDEO_VLM_CHUNK_SEC", str(VIDEO_VLM_CHUNK_SEC)))
    if device is None:
        device = resolve_device()
    out_name = video_path.stem.replace("_h264", "") if video_path else f"camera_{camera}"
    out = Path(output_dir or _STREAM_OUTPUT_DIR_PLAIN) / out_name
    out.mkdir(parents=True, exist_ok=True)
    chunk_dir = out / "vlm_chunks"
    chunk_dir.mkdir(exist_ok=True)

    if video_path is not None:
        video_path = video_io.ensure_opencv_video(Path(video_path))
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
    w, h = video_io._even_dims(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    chunk_frames = max(1, int(chunk_sec * fps))

    if display and not stream_preview._opencv_gui_available():
        if stream_preview._preview_player_available():
            print(
                "OpenCV sin ventanas → vista previa con mpv/ffplay (--preview).",
                flush=True,
            )
            preview = True
            display = False
        else:
            print(
                "Aviso: sin GUI OpenCV ni ffplay. Solo se guarda annotated_stream.mp4.\n"
                "  Vista previa: sudo apt install ffmpeg  y usa --preview\n"
                "  O: pip install opencv-python y DISPLAY=:0 con --display",
                flush=True,
            )
            display = False
    elif preview and not stream_preview._preview_player_available():
        if os.environ.get("STREAM_PREVIEW_HTTP", "1") == "0":
            print("Aviso: instala mpv o ffmpeg (ffplay). Sin vista previa.", flush=True)
            preview = False

    if preview:
        stream_preview._resolve_display(quiet=False)

    print(f"Stream: {out_name} | fuente={source} | trozo={chunk_sec}s ({chunk_frames} fr)", flush=True)
    if chunk_vlm_use_image(chunk_sec):
        vlm_chunk_mode = "imagen (1 frame/trozo)"
    else:
        vfps = video_vlm_fps(chunk_sec)
        vlm_chunk_mode = f"video ~{vfps:.0f} fps/trozo (1 s de movimiento)"
    print(
        f"VLM={VLM_ID} | perfil={os.environ.get('VLM_PROFILE', 'balanced')} | "
        f"prompt={vlm_prompt_mode()} | trozo→{vlm_chunk_mode} | "
        f"social={social_state_mode()} | salida: {out}",
        flush=True,
    )
    if display:
        print("Ventana OpenCV (pulsa q para salir).", flush=True)
    elif preview:
        pace = "ritmo del video" if realtime else "máxima velocidad"
        print(
            f"Vista previa en vivo ({pace}); cierra la ventana del reproductor para parar.",
            flush=True,
        )
    print(
        f"Grabando: {out / 'annotated_stream.mp4'} "
        f"(cada trozo se escribe al terminar el VLM → etiquetas alineadas al segundo)",
        flush=True,
    )

    if yolo is None:
        yolo = YOLO(YOLO_WEIGHTS)
    if vlm is None or processor is None:
        print("Cargando VLM …", flush=True)
        t_load = time.perf_counter()
        vlm, processor = load_vlm(VLM_ID, device)
        if VLM_WARMUP:
            warmup_vlm(vlm, processor)
        print(f"  VLM listo en {time.perf_counter() - t_load:.1f}s", flush=True)

    state = StreamState()
    stop = threading.Event()
    out_mp4 = out / "annotated_stream.mp4"
    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    live_preview: stream_preview.LivePreview | None = None
    want_preview = preview
    preview_gave_up = False
    t_pipeline0 = time.perf_counter()
    writer_lock = threading.Lock()
    stream_output = StreamOutput(
        writer=writer,
        lock=writer_lock,
        live_preview=None,
        w=w,
        h=h,
        fps=fps,
        chunk_sec=chunk_sec,
        chunk_index=0,
        t_pipeline0=t_pipeline0,
        state=state,
    )
    vlm_queue: Queue = Queue(maxsize=8)
    worker = threading.Thread(
        target=_vlm_worker,
        args=(vlm_queue, vlm, processor, state, stop, stream_output),
        daemon=True,
    )
    worker.start()

    prev_dets: list[dict] = []
    buffer_ann: list[np.ndarray] = []
    buffer_dets: list[list[dict]] = []
    chunk_display: list[DisplayFrame] = []
    frame_i = 0
    chunk_i = 0
    next_chunk_at = chunk_frames

    try:
        while True:
            t_fr0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_i >= max_frames:
                break

            frame = video_io._crop_frame(frame, w, h)
            raw = detect_people(yolo, frame)
            if prev_dets:
                dets = track_detections(prev_dets, raw)
            else:
                dets = assign_spatial_ids(raw)
            prev_dets = dets

            chunk_display.append(
                DisplayFrame(frame_i, frame.copy(), [{**d} for d in dets])
            )

            if display:
                with state.lock:
                    actions = dict(state.actions)
                    social_states = dict(state.social_states)
                    vlm_pending = state.vlm_pending
                    yolo_acum = state.yolo_total_s
                    vlm_acum = state.vlm_total_s
                    vlm_n = state.vlm_calls
                try:
                    preview_ann = _render_output_frame(
                        chunk_display[-1],
                        actions,
                        social_states,
                        w=w,
                        h=h,
                        fps=fps,
                        chunk_sec=chunk_sec,
                        chunk_index=chunk_i,
                        chunk_count=chunk_i + 1,
                        yolo_total_s=yolo_acum,
                        vlm_total_s=vlm_acum,
                        vlm_calls=vlm_n,
                        pipeline_total_s=time.perf_counter() - t_pipeline0,
                        vlm_pending=vlm_pending,
                    )
                    cv2.imshow("stream", preview_ann)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    display = False
                    print(
                        "  imshow no disponible; continuando solo guardando mp4 …",
                        flush=True,
                    )

            if want_preview and not preview_gave_up:
                if live_preview is None:
                    live_preview = stream_preview._open_live_preview(w, h, fps)
                    stream_output.live_preview = live_preview
                    if live_preview is None:
                        preview_gave_up = True
                        want_preview = False
                elif not live_preview.alive():
                    preview_gave_up = True
                    want_preview = False
                    stream_output.live_preview = None
                    print(
                        "  Ventana de vista previa cerrada; sigue el mp4 en disco.",
                        flush=True,
                    )

            buffer_ann.append(annotate_for_vlm(frame, dets))
            buffer_dets.append(dets)

            yolo_dt = time.perf_counter() - t_fr0
            with state.lock:
                state.yolo_total_s += yolo_dt
                state.frames_done = frame_i + 1

            frame_i += 1

            if frame_i >= next_chunk_at and len(buffer_ann) >= chunk_frames:
                pids = sorted(
                    {
                        det["pid"]
                        for row in buffer_dets[-chunk_frames:]
                        for det in row
                    }
                )
                pids = pids[: MAX_TRACK_IDS]
                dets_prompt = dets_for_vlm_prompt(buffer_dets, pids)
                _submit_chunk(
                    vlm_queue,
                    VlmChunkJob(
                        chunk_index=chunk_i,
                        start_frame=frame_i - len(buffer_ann),
                        frames=list(buffer_ann),
                        dets_mid=dets_prompt,
                        person_ids=pids,
                        out_path=chunk_dir / f"chunk_{chunk_i:04d}.mp4",
                        fps=fps,
                        size=(w, h),
                        display_frames=list(chunk_display),
                        buffer_dets=[list(row) for row in buffer_dets],
                    ),
                    stream_output,
                )
                chunk_i += 1
                next_chunk_at += chunk_frames
                buffer_ann = []
                buffer_dets = []
                chunk_display = []

            if realtime and source == "video":
                time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t_fr0)))

    finally:
        if buffer_ann and buffer_dets:
            pids = sorted({d["pid"] for row in buffer_dets for d in row})[: MAX_TRACK_IDS]
            dets_prompt = dets_for_vlm_prompt(buffer_dets, pids)
            _submit_chunk(
                vlm_queue,
                VlmChunkJob(
                    chunk_index=chunk_i,
                    start_frame=frame_i - len(buffer_ann),
                    frames=list(buffer_ann),
                    dets_mid=dets_prompt,
                    person_ids=pids,
                    out_path=chunk_dir / f"chunk_{chunk_i:04d}.mp4",
                    fps=fps,
                    size=(w, h),
                    display_frames=list(chunk_display),
                    buffer_dets=[list(row) for row in buffer_dets],
                ),
                stream_output,
            )
            chunk_i += 1

        stop.set()
        worker.join(timeout=300)
        # Trozos aún en cola al parar: el worker los vacía y escribe al mp4.
        cap.release()
        writer.release()
        if live_preview is not None:
            live_preview.close()
        if display:
            stream_preview._safe_destroy_windows()

    video_io.finalize_video_h264(out_mp4)
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

    rt = video_io._realtime_metrics(
        frames=frame_i,
        fps=fps,
        yolo_s=yolo_total_s,
        vlm_s=vlm_total_s,
        render_s=0.0,
        encode_s=0.0,
        pipeline_s=pipeline_total_s,
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
        "social_state_mode": social_state_mode(),
        "actions_final": {str(k): v for k, v in final_actions.items()},
        "social_states_final": {str(k): v for k, v in final_social.items()},
        "vlm_chunk_timings": vlm_chunk_timings,
        "vlm_chunk_latencies_s": [t["latency_s"] for t in vlm_chunk_timings],
        **video_io.vlm_timing_stats([t["latency_s"] for t in vlm_chunk_timings]),
        "segments": _build_segment_summary(
            segment_actions, segment_social, vlm_chunk_timings
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    vname = video_path.name if video_path else (f"camera_{camera}" if camera is not None else out_name)
    video_io.print_vlm_latency_report(vname, vlm_chunk_timings)
    print(
        f"\nStream listo → {out} ({pipeline_total_s:.1f}s, {frame_i} frames, "
        f"VLM {vlm_calls} trozos, factor RT {rt['realtime_factor']:.2f}x)",
        flush=True,
    )
    print(f"  Ver resultado: {out_mp4}", flush=True)
    return out


def _run_all_plain(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    preview: bool = False,
    realtime: bool = False,
) -> list[Path]:
    """Todos los videos de input_videos/ con pipeline stream + resumen batch."""
    

    indir = Path(input_dir or ROOT / "input_videos")
    base_out = Path(output_dir or _STREAM_OUTPUT_DIR_PLAIN)
    paths = video_io.list_videos(indir)
    if not paths:
        raise SystemExit(f"No hay videos en {indir}")

    print(f"Stream batch: {len(paths)} videos en {indir}\n", flush=True)
    device = resolve_device()
    yolo = YOLO(YOLO_WEIGHTS)
    print("Cargando VLM una vez para todo el batch …", flush=True)
    t0 = time.perf_counter()
    vlm, processor = load_vlm(VLM_ID, device)
    if VLM_WARMUP:
        warmup_vlm(vlm, processor)
    print(f"  VLM listo en {time.perf_counter() - t0:.1f}s\n", flush=True)

    outs: list[Path] = []
    rows: list[dict] = []
    for i, path in enumerate(paths, 1):
        print(f"{'=' * 60}\n[{i}/{len(paths)}] {path.name}\n{'=' * 60}")
        out = _run_plain(
            video_path=path,
            output_dir=base_out,
            preview=preview,
            realtime=realtime,
            yolo=yolo,
            vlm=vlm,
            processor=processor,
            device=device,
        )
        outs.append(out)
        row = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        row["render_s"] = row.get("render_s", 0)
        row["encode_s"] = row.get("encode_s", 0)
        rows.append(row)

    video_io.write_batch_summary(rows, base_out)
    (base_out / "batch_summary_stream.json").write_text(
        (base_out / "batch_summary.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return outs


@dataclass
class RawFrame:
    """Frame crudo + detecciones (con keypoints) tal cual salen de Fase 1,
    ANTES de saber azul/rojo (eso solo se sabe al cerrar el trozo)."""
    frame_i: int
    frame_bgr: np.ndarray
    dets: list[dict]


def _scaled_raw_frame(rf: RawFrame, factor: float) -> RawFrame:
    """Agranda el frame Y las coordenadas de detecciones/keypoints por
    `factor` ANTES de dibujar encima -- para la ventana --display (DISPLAY_SCALE).

    Escalar la imagen ya dibujada (texto/cajas/esqueleto) con cv2.resize se ve
    borroso: la interpolación difumina los bordes nítidos del texto y las
    líneas finas. Escalando el frame crudo primero y dibujando después, cada
    elemento se dibuja nítido directamente al tamaño final (ui_scale ya
    calcula el grosor/tamaño de fuente a partir de las dimensiones de la
    imagen, así que sale proporcionalmente más grande, no borroso)."""
    if factor == 1.0:
        return rf
    frame = cv2.resize(rf.frame_bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    dets: list[dict] = []
    for d in rf.dets:
        nd = dict(d)
        for k in ("x1", "y1", "x2", "y2"):
            nd[k] = int(round(d[k] * factor))
        kpts = d.get("kpts")
        if kpts is not None:
            kpts = kpts.copy()
            kpts[:, 0] *= factor
            kpts[:, 1] *= factor
            nd["kpts"] = kpts
        dets.append(nd)
    return RawFrame(rf.frame_i, frame, dets)


@dataclass
class _PoseStreamOutput:
    writer: cv2.VideoWriter
    lock: threading.Lock
    live_preview: stream_preview.LivePreview | None
    w: int
    h: int
    fps: float
    chunk_sec: float
    t_pipeline0: float
    state: _PoseStreamState
    draw_pose: bool = False


@dataclass
class _PoseVlmChunkJob:
    chunk_index: int
    start_frame: int
    raw_frames: list[RawFrame]
    person_ids: list[int]
    out_path: Path
    fps: float
    size: tuple[int, int]


@dataclass
class _PoseStreamState:
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


def _render_output_frame_pose(
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
    draw_pose: bool = False,
) -> np.ndarray:
    """Video final para el usuario (no el que ve el VLM): caja de color + ID +
    panel lateral + resalte del target — igual estilo que run_video_pose.py.
    `draw_pose=True`: superpone además el esqueleto COCO-17 (visualización)."""
    ann = rf.frame_bgr.copy()
    target_pid = pose_gait.pick_interaction_target(
        rf.dets, {d["pid"]: social_states.get(d["pid"]) for d in rf.dets}
    )
    panel_entries: list[tuple[int, str, tuple[int, int, int], bool]] = []
    for d in rf.dets:
        social_state = social_states.get(d["pid"])
        color = social_box_color(social_state)
        pose_gait.draw_box_only(ann, d["x1"], d["y1"], d["x2"], d["y2"], d["pid"], color)
        if draw_pose:
            pose_gait.draw_pose_skeleton(ann, d.get("kpts"))
        act = normalize_action(actions.get(d["pid"], ""))
        state_txt = social_state or "UNKNOWN"
        text = f"{state_txt} ({act})" if act and act != "unknown" else state_txt
        is_target = d["pid"] == target_pid
        if is_target:
            cv2.rectangle(
                ann, (d["x1"] - 3, d["y1"] - 3), (d["x2"] + 3, d["y2"] + 3),
                (0, 255, 255), 2,
            )
        panel_entries.append((d["pid"], text, color, is_target))
    pose_gait.draw_side_panel(ann, panel_entries)
    draw_banner_video(
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
            (int(10 * ui_scale(h, w)), int(h - 30 * ui_scale(h, w))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui_scale(h, w), (255, 200, 255), 1, cv2.LINE_8,
        )
    return ann


def _flush_chunk_to_output_pose(job: _PoseVlmChunkJob, output: _PoseStreamOutput) -> None:
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
            ann = _render_output_frame_pose(
                rf, actions, social_states,
                w=output.w, h=output.h, fps=output.fps, chunk_sec=output.chunk_sec,
                chunk_index=job.chunk_index, chunk_count=chunk_count,
                yolo_total_s=yolo_acum, vlm_total_s=vlm_acum, vlm_calls=vlm_n,
                pipeline_total_s=pipeline_now, vlm_pending=vlm_pending,
                draw_pose=output.draw_pose,
            )
            output.writer.write(ann)
            if output.live_preview is not None:
                try:
                    output.live_preview.write(ann)
                except BrokenPipeError:
                    pass


def _vlm_worker_pose(
    queue: Queue,
    vlm,
    processor,
    state: _PoseStreamState,
    stop: threading.Event,
    output: _PoseStreamOutput | None,
) -> None:
    motion_hyst = pose_gait.MotionHysteresis()
    while True:
        if stop.is_set():
            try:
                job: _PoseVlmChunkJob = queue.get_nowait()
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
        gait = pose_gait.chunk_gait_by_pid(buffer_dets, pids)
        depth = pose_gait.chunk_depth_motion_by_pid(buffer_dets, pids)
        raw_moving = {p for p in pids if gait.get(p) or depth.get(p)}
        moving_pids = motion_hyst.update(job.chunk_index, raw_moving, set(pids))

        vlm_frames = [
            pose_gait.annotate_for_vlm_pose(rf.frame_bgr, rf.dets, moving_pids) for rf in job.raw_frames
        ]
        dets_prompt = dets_for_vlm_prompt(buffer_dets, pids)

        chunk_sec = len(vlm_frames) / max(job.fps, 1e-6)
        t0 = time.perf_counter()
        try:
            if chunk_vlm_use_image(chunk_sec):
                mid = vlm_frames[len(vlm_frames) // 2]
                vlm_out = infer_actions_image(vlm, processor, mid, dets_prompt, already_annotated=True)
            elif _write_chunk_video(vlm_frames, job.out_path, job.fps, job.size):
                vlm_out = infer_actions_video(
                    vlm, processor, job.out_path, pids, dets=dets_prompt, chunk_sec=chunk_sec
                )
            else:
                mid = vlm_frames[len(vlm_frames) // 2]
                vlm_out = infer_actions_image(vlm, processor, mid, dets_prompt, already_annotated=True)
        except Exception as e:
            print(f"  [VLM trozo {job.chunk_index}] error: {e}", flush=True)
            vlm_out = VlmInferenceResult({}, {}, 0.0)

        acts = dict(vlm_out.actions)
        social = dict(vlm_out.social_states)
        if pids:
            # Misma lógica de refinamiento que el modo offline (run_video_pose.py):
            # debias del sesgo grupal de "walking", forzado a caminar si hay
            # evidencia independiente, respaldo por cinemática de caja
            # (con compensación de movimiento de cámara), y ATTENTIVE por mirada.
            acts, social = pose_gait.refine_labels_pose(
                acts, social, buffer_dets, pids, job.size, moving_pids_hint=moving_pids
            )
            social = pose_gait.upgrade_attentive_by_gaze(social, buffer_dets, pids)

        dt = time.perf_counter() - t0
        with state.lock:
            state.vlm_total_s += dt
            state.vlm_calls += 1
            state.vlm_pending = False
            for pid, act in acts.items():
                if act and act != "unknown":
                    state.actions[pid] = act
            for pid, st in social.items():
                if st and st != SOCIAL_UNKNOWN:
                    state.social_states[pid] = st
            state.segment_actions.append((job.start_frame, dict(acts)))
            state.segment_social.append((job.start_frame, dict(social)))
            state.vlm_chunk_timings.append(
                {"chunk_index": job.chunk_index, "start_frame": job.start_frame, "latency_s": round(vlm_out.elapsed_s, 4)}
            )
        if output is not None and job.raw_frames:
            _flush_chunk_to_output_pose(job, output)
        print(
            f"  [VLM trozo {job.chunk_index}] f{job.start_frame}+ actions={acts} social={social} "
            f"(inferencia {vlm_out.elapsed_s:.2f}s, acum VLM {state.vlm_total_s:.2f}s)",
            flush=True,
        )


def _run_pose(
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
    draw_pose: bool = False,
) -> Path:
    if (video_path is None) == (camera is None):
        raise SystemExit("Indica --input VIDEO o --camera N (no ambos).")

    chunk_sec = float(os.environ.get("VIDEO_VLM_CHUNK_SEC", str(VIDEO_VLM_CHUNK_SEC)))
    if device is None:
        device = resolve_device()
    out_name = video_path.stem.replace("_h264", "") if video_path else f"camera_{camera}"
    out = Path(output_dir or _STREAM_OUTPUT_DIR_POSE) / out_name
    out.mkdir(parents=True, exist_ok=True)
    chunk_dir = out / "vlm_chunks"
    chunk_dir.mkdir(exist_ok=True)

    if video_path is not None:
        video_path = video_io.ensure_opencv_video(Path(video_path))
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
    w, h = video_io._even_dims(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    chunk_frames = max(1, int(chunk_sec * fps))

    if display and not stream_preview._opencv_gui_available():
        if stream_preview._preview_player_available():
            preview, display = True, False
        else:
            display = False
    elif preview and not stream_preview._preview_player_available():
        if os.environ.get("STREAM_PREVIEW_HTTP", "1") == "0":
            preview = False
    if preview:
        stream_preview._resolve_display(quiet=False)

    print(f"Stream [pose]: {out_name} | fuente={source} | trozo={chunk_sec}s ({chunk_frames} fr) | salida: {out}", flush=True)

    if yolo is None:
        yolo = YOLO(YOLO_WEIGHTS)
    if yolo_pose is None:
        yolo_pose = YOLO(POSE_YOLO_WEIGHTS)
    warmup_yolo(yolo, shape=(h, w, 3))
    warmup_yolo(yolo_pose, shape=(h, w, 3))
    if vlm is None or processor is None:
        print("Cargando VLM …", flush=True)
        t_load = time.perf_counter()
        vlm, processor = load_vlm(VLM_ID, device)
        if VLM_WARMUP:
            warmup_vlm(vlm, processor)
        print(f"  VLM listo en {time.perf_counter() - t_load:.1f}s", flush=True)

    state = _PoseStreamState()
    stop = threading.Event()
    out_mp4 = out / "annotated_stream.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    live_preview: stream_preview.LivePreview | None = None
    want_preview = preview
    preview_gave_up = False
    t_pipeline0 = time.perf_counter()
    writer_lock = threading.Lock()
    stream_output = _PoseStreamOutput(
        writer=writer, lock=writer_lock, live_preview=None,
        w=w, h=h, fps=fps, chunk_sec=chunk_sec, t_pipeline0=t_pipeline0, state=state,
        draw_pose=draw_pose,
    )
    vlm_queue: Queue = Queue(maxsize=8)
    worker = threading.Thread(
        target=_vlm_worker_pose, args=(vlm_queue, vlm, processor, state, stop, stream_output), daemon=True
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
        all_pids = sorted({d["pid"] for rf in chunk_raw for d in rf.dets})
        # Igual que runners/video.py: prioriza a quien está más cerca de la
        # cámara (mayor área de caja), no a quien tiene el pid numérico más
        # bajo -- antes este truncado por orden de pid siempre se quedaba con
        # los primeros IDs asignados en la sesión, sin importar qué tan
        # relevantes (cercanos) fueran.
        ranking_rows = [
            {"person_id": d["pid"], "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"]}
            for rf in chunk_raw
            for d in rf.dets
        ]
        pids = pose_gait.closest_n_pids(ranking_rows, all_pids, pose_gait.MAX_VLM_PEOPLE)
        _submit_chunk(
            vlm_queue,
            _PoseVlmChunkJob(
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

            frame = video_io._crop_frame(frame, w, h)
            raw = detect_people(yolo, frame)
            dets = track_detections(prev_dets, raw) if prev_dets else assign_spatial_ids(raw)
            prev_dets = dets
            pose_raw = pose_gait.detect_people_pose(yolo_pose, frame)
            pose_gait.attach_pose_keypoints(dets, pose_raw)

            chunk_raw.append(RawFrame(frame_i, frame.copy(), [dict(d) for d in dets]))

            if display:
                with state.lock:
                    actions = dict(state.actions)
                    social_states = dict(state.social_states)
                    vlm_pending = state.vlm_pending
                    yolo_acum, vlm_acum, vlm_n = state.yolo_total_s, state.vlm_total_s, state.vlm_calls
                try:
                    disp_rf = _scaled_raw_frame(chunk_raw[-1], DISPLAY_SCALE)
                    disp_h, disp_w = disp_rf.frame_bgr.shape[:2]
                    display_ann = _render_output_frame_pose(
                        disp_rf, actions, social_states,
                        w=disp_w, h=disp_h, fps=fps, chunk_sec=chunk_sec, chunk_index=chunk_i, chunk_count=chunk_i + 1,
                        yolo_total_s=yolo_acum, vlm_total_s=vlm_acum, vlm_calls=vlm_n,
                        pipeline_total_s=time.perf_counter() - t_pipeline0, vlm_pending=vlm_pending,
                        draw_pose=draw_pose,
                    )
                    cv2.imshow("stream-pose", display_ann)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    display = False
                    print("  imshow no disponible; continuando solo guardando mp4 …", flush=True)

            if want_preview and not preview_gave_up:
                if live_preview is None:
                    live_preview = stream_preview._open_live_preview(w, h, fps)
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
            stream_preview._safe_destroy_windows()

    video_io.finalize_video_h264(out_mp4)
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

    rt = video_io._realtime_metrics(
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
        "social_state_mode": social_state_mode(),
        "detector": "pose+gait (stream)",
        "actions_final": {str(k): v for k, v in final_actions.items()},
        "social_states_final": {str(k): v for k, v in final_social.items()},
        "vlm_chunk_timings": vlm_chunk_timings,
        "vlm_chunk_latencies_s": [t["latency_s"] for t in vlm_chunk_timings],
        **video_io.vlm_timing_stats([t["latency_s"] for t in vlm_chunk_timings]),
        "segments": _build_segment_summary(segment_actions, segment_social, vlm_chunk_timings),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    vname = video_path.name if video_path else f"camera_{camera}"
    video_io.print_vlm_latency_report(vname, vlm_chunk_timings)
    print(
        f"\nStream [pose] listo → {out} ({pipeline_total_s:.1f}s, {frame_i} frames, "
        f"VLM {vlm_calls} trozos, factor RT {rt['realtime_factor']:.2f}x)",
        flush=True,
    )
    print(f"  Ver resultado: {out_mp4}", flush=True)
    return out


def run(
    *,
    video_path: Path | None = None,
    camera: int | None = None,
    output_dir: Path | None = None,
    use_pose: bool = True,
    draw_pose: bool = False,
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
    """Cámara o video en vivo. `use_pose=True` (default): detector de pose +
    marcha por piernas, ATTENTIVE por mirada y target de interacción
    (yolo26n-pose). `use_pose=False`: solo detección (yolo11n). `draw_pose`
    (solo con use_pose): dibuja también el esqueleto COCO-17 en el video/
    ventana final, no solo en lo que ve el VLM."""
    if use_pose:
        return _run_pose(
            video_path=video_path, camera=camera, output_dir=output_dir,
            display=display, preview=preview, realtime=realtime, max_frames=max_frames,
            yolo=yolo, yolo_pose=yolo_pose, vlm=vlm, processor=processor, device=device,
            draw_pose=draw_pose,
        )
    return _run_plain(
        video_path=video_path, camera=camera, output_dir=output_dir,
        display=display, preview=preview, realtime=realtime, max_frames=max_frames,
        yolo=yolo, vlm=vlm, processor=processor, device=device,
    )


def run_all(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    use_pose: bool = True,
    preview: bool = False,
    realtime: bool = False,
) -> list[Path]:
    """Todos los videos de `input_dir` en modo streaming + resumen batch.
    Solo soporta `use_pose=False`: el modo pose nunca tuvo un batch (no era
    alcanzable desde ningún entrypoint documentado)."""
    if use_pose:
        raise NotImplementedError(
            "run_all en modo streaming con pose no está implementado; usa use_pose=False."
        )
    return _run_all_plain(input_dir, output_dir, preview=preview, realtime=realtime)
