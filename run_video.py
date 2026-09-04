"""Pipeline de video: YOLO en todos los frames → video anotado → VLM sobre el clip."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

import pipeline as P

ROOT = P.ROOT

VIDEO_INPUT = Path(
    os.environ.get(
        "VIDEO_INPUT",
        str(ROOT / "input_videos" / "youtube_short_935cb0IrwEw.mp4"),
    )
)
VIDEO_OUTPUT_DIR = Path(os.environ.get("VIDEO_OUTPUT_DIR", str(ROOT / "output_video")))


def _opencv_can_read(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    ok, frame = cap.read()
    cap.release()
    return ok and frame is not None


def _even_dims(w: int, h: int) -> tuple[int, int]:
    return w - (w % 2), h - (h % 2)


def vlm_timing_stats(latencies_s: list[float]) -> dict:
    """Promedio / min / max de latencia por inferencia VLM (un trozo o keyframe)."""
    if not latencies_s:
        return {
            "vlm_latency_count": 0,
            "vlm_latency_avg_s": 0.0,
            "vlm_latency_min_s": 0.0,
            "vlm_latency_max_s": 0.0,
        }
    return {
        "vlm_latency_count": len(latencies_s),
        "vlm_latency_avg_s": round(sum(latencies_s) / len(latencies_s), 4),
        "vlm_latency_min_s": round(min(latencies_s), 4),
        "vlm_latency_max_s": round(max(latencies_s), 4),
    }


def print_vlm_latency_report(video_name: str, chunk_rows: list[dict]) -> None:
    """Imprime tabla de latencias VLM por trozo y el promedio del video."""
    if not chunk_rows:
        return
    latencies = [float(r["latency_s"]) for r in chunk_rows]
    st = vlm_timing_stats(latencies)
    print(f"  VLM por trozo ({video_name}):", flush=True)
    for r in chunk_rows:
        print(
            f"    trozo {r.get('chunk_index', '?')} f{r.get('start_frame', '?')}: "
            f"{r['latency_s']:.3f}s",
            flush=True,
        )
    print(
        f"  → Promedio VLM/video: {st['vlm_latency_avg_s']:.3f}s "
        f"(min {st['vlm_latency_min_s']:.3f}s, max {st['vlm_latency_max_s']:.3f}s, "
        f"n={st['vlm_latency_count']})",
        flush=True,
    )


def _crop_frame(frame, w: int, h: int):
    return frame[:h, :w]


def finalize_video_h264(path: Path, *, crf: int = 16) -> None:
    """Re-codifica con H.264 (evita borrosidad del codec mp4v de OpenCV)."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    tmp = path.with_name(f"{path.stem}_x264{path.suffix}")
    preset = os.environ.get("ENCODE_PRESET", "veryfast")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Aviso: no se re-codificó {path.name} ({e})", flush=True)
        if tmp.is_file():
            tmp.unlink(missing_ok=True)


class FFmpegPipeWriter:
    """Escribe frames BGR directo a ffmpeg (libx264) por pipe. A diferencia de
    cv2.VideoWriter+mp4v, no cuantiza mal colores saturados en secuencias de
    bajo movimiento, y evita el doble encode con pérdida (mp4v → H.264)."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        w, h = size
        preset = os.environ.get("ENCODE_PRESET", "veryfast")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def write(self, frame) -> None:
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()


def extract_video_segment(
    src: Path,
    dst: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    size: tuple[int, int],
) -> bool:
    w, h = size
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for _ in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    return dst.is_file() and dst.stat().st_size > 0


def ensure_opencv_video(path: Path) -> Path:
    """OpenCV suele fallar con AV1/HEVC; convertimos a H.264 (yuv420p) con ffmpeg."""
    if _opencv_can_read(path):
        return path
    cache = path.parent / f"{path.stem}_h264{path.suffix}"
    if cache.is_file() and _opencv_can_read(cache):
        print(f"Usando caché H.264: {cache.name}")
        return cache
    print(f"OpenCV no lee {path.name} (p. ej. AV1). Convirtiendo a H.264 …")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(cache),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if not _opencv_can_read(cache):
        raise SystemExit(f"No se pudo convertir el video para OpenCV: {cache}")
    print(f"Listo: {cache}")
    return cache


def _realtime_metrics(
    *,
    frames: int,
    fps: float,
    yolo_s: float,
    vlm_s: float,
    render_s: float,
    encode_s: float,
    pipeline_s: float,
) -> dict:
    dur = frames / max(fps, 1e-6)
    other = max(0.0, pipeline_s - yolo_s - vlm_s - render_s - encode_s)
    ms_pf = 1000.0 * pipeline_s / max(frames, 1)
    proc_fps = frames / max(pipeline_s, 1e-6)
    rt_factor = pipeline_s / max(dur, 1e-6)
    return {
        "video_duration_s": round(dur, 3),
        "pipeline_ms_per_frame": round(ms_pf, 2),
        "processing_fps": round(proc_fps, 2),
        "realtime_factor": round(rt_factor, 2),
        "yolo_realtime_ok": yolo_s < dur,
        "pipeline_realtime_ok": pipeline_s < dur,
        "other_s": round(other, 4),
        "encode_s": round(encode_s, 4),
    }


def run(
    video_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    yolo: YOLO | None = None,
    vlm=None,
    processor=None,
    device=None,
) -> Path:
    video_path = Path(video_path or VIDEO_INPUT)
    if not video_path.is_file():
        raise SystemExit(f"No existe el video: {video_path}")
    video_path = ensure_opencv_video(video_path)
    t_pipeline0 = time.perf_counter()

    out = Path(output_dir or VIDEO_OUTPUT_DIR) / video_path.stem.replace("_h264", "")
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = _even_dims(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    yolo_mp4 = out / "annotated_yolo.mp4"
    vlm_mp4 = out / "vlm_input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(yolo_mp4), fourcc, fps, (w, h))
    vlm_writer = cv2.VideoWriter(str(vlm_mp4), fourcc, fps, (w, h))

    if device is None:
        device = P.resolve_device()
    print(f"Video: {video_path.name} ({n_frames} frames @ {fps:.1f} fps)")
    print(f"VLM={P.VLM_ID} | salida: {out}")
    if yolo is None:
        yolo = YOLO(P.YOLO_WEIGHTS)
    P.warmup_yolo(yolo, shape=(h, w, 3))

    # Config de trozos VLM calculada antes de la Fase 1 para poder escribir
    # cada trozo directamente durante el único recorrido secuencial del video
    # (evita reabrir/seekear vlm_mp4 por cada trozo en la Fase 2).
    chunk_sec = float(os.environ.get("VIDEO_VLM_CHUNK_SEC", str(P.VIDEO_VLM_CHUNK_SEC)))
    vlm_mode = os.environ.get(
        "VIDEO_VLM_MODE", "chunks" if chunk_sec > 0 else "video"
    ).lower()
    chunk_frames = max(1, int(chunk_sec * fps)) if chunk_sec > 0 else n_frames
    use_img = P.chunk_vlm_use_image(chunk_sec)
    chunk_dir = out / "vlm_chunks"
    write_chunks_inline = vlm_mode == "chunks" and chunk_sec > 0 and not use_img
    if write_chunks_inline:
        chunk_dir.mkdir(exist_ok=True)
    chunk_writer = None
    chunk_writer_ci = -1

    # —— Fase 1: YOLO en todo el video + tracking de IDs ——
    print("Fase 1/2: YOLO + tracking …", flush=True)
    t_yolo0 = time.perf_counter()
    prev_dets: list[dict] = []
    frame_rows: list[dict] = []
    all_frame_dets: list[list[dict]] = []
    all_pids: set[int] = set()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _crop_frame(frame, w, h)
        raw = P.detect_people(yolo, frame)
        if prev_dets:
            dets = P.track_detections(prev_dets, raw)
        else:
            dets = P.assign_spatial_ids(raw)
        prev_dets = dets
        all_frame_dets.append(dets)
        for d in dets:
            all_pids.add(d["pid"])

        ann = frame.copy()
        for d in dets:
            P.draw_box(
                ann, d["x1"], d["y1"], d["x2"], d["y2"], f"ID{d['pid']} {d['class_name']}"
            )
        writer.write(ann)
        ann_vlm = P.annotate_for_vlm(frame, dets)
        vlm_writer.write(ann_vlm)
        if write_chunks_inline:
            ci = frame_idx // chunk_frames
            if ci != chunk_writer_ci:
                if chunk_writer is not None:
                    chunk_writer.release()
                chunk_writer_ci = ci
                chunk_writer = cv2.VideoWriter(
                    str(chunk_dir / f"chunk_{ci:03d}.mp4"), fourcc, fps, (w, h)
                )
            chunk_writer.write(ann_vlm)

        for d in dets:
            frame_rows.append(
                {
                    "frame": frame_idx,
                    "time_s": round(frame_idx / fps, 4),
                    "person_id": d["pid"],
                    "x1": d["x1"],
                    "y1": d["y1"],
                    "x2": d["x2"],
                    "y2": d["y2"],
                    "class_name": d["class_name"],
                    "conf": round(d["conf"], 4),
                }
            )
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  … frame {frame_idx}/{n_frames}", flush=True)

    cap.release()
    writer.release()
    vlm_writer.release()
    if chunk_writer is not None:
        chunk_writer.release()
    yolo_s = time.perf_counter() - t_yolo0
    print(f"  YOLO listo: {frame_idx} frames en {yolo_s:.1f}s → {yolo_mp4.name}")
    print(f"  Clip VLM: {vlm_mp4.name}")

    pd.DataFrame(frame_rows).to_csv(out / "detections_per_frame.csv", index=False)

    if not all_pids:
        print("No se detectaron personas; se omite VLM.")
        return out

    stable_ids = P.stable_person_ids(frame_rows)
    vlm_ids = stable_ids if stable_ids else sorted(all_pids)[: P.MAX_TRACK_IDS]
    print(
        f"  Tracks totales: {len(all_pids)} | VLM sobre {len(vlm_ids)} IDs estables: {vlm_ids}",
        flush=True,
    )

    # —— Fase 2: VLM (por defecto trozos de VIDEO_VLM_CHUNK_SEC segundos) ——
    print(
        f"Fase 2/2: VLM (modo={vlm_mode}, trozo={chunk_sec}s, "
        f"social={P.social_state_mode()}) …",
        flush=True,
    )
    vlm_load_s = 0.0
    if vlm is None or processor is None:
        t_load = time.perf_counter()
        vlm, processor = P.load_vlm(P.VLM_ID, device)
        vlm_load_s = time.perf_counter() - t_load
        if P.VLM_WARMUP:
            P.warmup_vlm(vlm, processor)
    proc = processor

    rep_idx, rep_dets = P.pick_representative_frame(all_frame_dets, vlm_ids)
    segment_actions: list[tuple[int, dict[int, str]]] = []
    segment_social: list[tuple[int, dict[int, str]]] = []
    vlm_chunk_timings: list[dict] = []
    vlm_total_s = 0.0
    vlm_input_kind = vlm_mode
    n_chunks = max(1, (frame_idx + chunk_frames - 1) // chunk_frames)

    if vlm_mode == "chunks" and chunk_sec > 0:
        vlm_input_kind = "chunks"
        vlm_chunk_mode = (
            "imagen (frame central)"
            if use_img
            else f"video @ {P.VIDEO_VLM_FPS} fps"
        )
        print(
            f"  VLM: {n_chunks} trozos de ~{chunk_sec}s ({chunk_frames} fr/trozo) → {vlm_chunk_mode}",
            flush=True,
        )
        cap_vlm = cv2.VideoCapture(str(vlm_mp4)) if use_img else None

        def _fallback_image(sf: int, ef: int):
            rel_idx, dets_frame = P.pick_representative_frame(
                all_frame_dets[sf:ef], vlm_ids
            )
            if not dets_frame:
                return None
            mid = sf + rel_idx
            cap_m = cap_vlm if cap_vlm is not None else cv2.VideoCapture(str(vlm_mp4))
            cap_m.set(cv2.CAP_PROP_POS_FRAMES, mid)
            _, fr = cap_m.read()
            if cap_vlm is None:
                cap_m.release()
            if fr is None:
                return None
            return P.infer_actions_image(vlm, proc, fr, dets_frame, already_annotated=True)

        def _emit(ci: int, sf: int, ef: int, vlm_out) -> None:
            nonlocal vlm_total_s
            if vlm_out is None:
                return
            vlm_total_s += vlm_out.elapsed_s
            segment_actions.append((sf, vlm_out.actions))
            segment_social.append((sf, vlm_out.social_states))
            vlm_chunk_timings.append(
                {
                    "chunk_index": ci,
                    "start_frame": sf,
                    "latency_s": round(vlm_out.elapsed_s, 4),
                }
            )
            print(
                f"    trozo {ci + 1}/{n_chunks} frames {sf}-{ef}: "
                f"actions={vlm_out.actions} social={vlm_out.social_states} "
                f"(inferencia {vlm_out.elapsed_s:.2f}s, acum VLM {vlm_total_s:.2f}s)",
                flush=True,
            )

        if use_img:
            for ci, sf in enumerate(range(0, frame_idx, chunk_frames)):
                ef = min(frame_idx, sf + chunk_frames)
                if ef <= sf:
                    continue
                # Escoge, dentro del trozo, el frame donde más IDs objetivo son
                # visibles a la vez (no siempre el geométricamente central) —
                # evita perder a alguien solo porque el punto medio lo tapaba.
                rel_idx, dets_mid = P.pick_representative_frame(
                    all_frame_dets[sf:ef], vlm_ids
                )
                if not dets_mid:
                    continue
                mid = sf + rel_idx
                assert cap_vlm is not None
                cap_vlm.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ok_m, fr = cap_vlm.read()
                if not ok_m or fr is None:
                    continue
                vlm_out = P.infer_actions_image(
                    vlm, proc, fr, dets_mid, already_annotated=True
                )
                buffer_dets = all_frame_dets[sf:ef]
                acts, soc = P.refine_chunk_labels(
                    vlm_out.actions, vlm_out.social_states, buffer_dets, vlm_ids, (w, h)
                )
                vlm_out = vlm_out._replace(actions=acts, social_states=soc)
                _emit(ci, sf, ef, vlm_out)
            cap_vlm.release()
        else:
            # Los trozos ya se escribieron en disco durante la Fase 1
            # (write_chunks_inline); aquí solo se agrupan en lotes para VLM.
            pending = []
            for ci, sf in enumerate(range(0, frame_idx, chunk_frames)):
                ef = min(frame_idx, sf + chunk_frames)
                if ef <= sf:
                    continue
                # El clip ya muestra cada frame con sus cajas reales; para el
                # texto del prompt basta la mejor caja por ID en TODO el trozo
                # (no solo el punto medio), así no se pierden IDs ocluidos justo
                # a mitad del trozo.
                dets_mid = P.dets_for_vlm_prompt(all_frame_dets[sf:ef], vlm_ids)
                if not dets_mid:
                    continue
                chunk_path = chunk_dir / f"chunk_{ci:03d}.mp4"
                if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
                    continue
                pending.append(
                    {"ci": ci, "sf": sf, "ef": ef, "path": chunk_path, "dets": dets_mid}
                )

            batch_size = max(1, P.VLM_BATCH_SIZE)
            for i in range(0, len(pending), batch_size):
                group = pending[i : i + batch_size]
                items = [(g["path"], vlm_ids, g["dets"]) for g in group]
                try:
                    results = P.infer_actions_video_batch(
                        vlm, proc, items, chunk_sec=chunk_sec
                    )
                except Exception as e:
                    results = []
                    for g in group:
                        print(
                            f"    trozo {g['ci'] + 1} falló ({e}); imagen f{g['sf']} …",
                            flush=True,
                        )
                        results.append(_fallback_image(g["sf"], g["ef"]))
                for g, vlm_out in zip(group, results):
                    if vlm_out is not None:
                        buffer_dets = all_frame_dets[g["sf"] : g["ef"]]
                        acts, soc = P.refine_chunk_labels(
                            vlm_out.actions, vlm_out.social_states, buffer_dets, vlm_ids, (w, h)
                        )
                        vlm_out = vlm_out._replace(actions=acts, social_states=soc)
                    _emit(g["ci"], g["sf"], g["ef"], vlm_out)

    elif vlm_mode == "video":
        print(f"  VLM: clip completo {vlm_mp4.name} @ {P.VIDEO_VLM_FPS} fps", flush=True)
        try:
            vlm_out = P.infer_actions_video(
                vlm, proc, vlm_mp4, vlm_ids, dets=rep_dets
            )
            acts, soc = P.refine_chunk_labels(
                vlm_out.actions, vlm_out.social_states, all_frame_dets, vlm_ids, (w, h)
            )
            vlm_out = vlm_out._replace(actions=acts, social_states=soc)
            segment_actions = [(0, vlm_out.actions)]
            segment_social = [(0, vlm_out.social_states)]
            vlm_chunk_timings = [
                {
                    "chunk_index": 0,
                    "start_frame": 0,
                    "latency_s": round(vlm_out.elapsed_s, 4),
                }
            ]
            vlm_total_s = vlm_out.elapsed_s
            print(
                f"  actions={vlm_out.actions} social={vlm_out.social_states}",
                flush=True,
            )
        except Exception as e:
            print(f"  VLM video falló ({e}); modo frame …", flush=True)
            vlm_mode = "frame"

    if vlm_mode == "frame":
        vlm_input_kind = "frame"
        n_kf = max(1, int(os.environ.get("VIDEO_VLM_KEYFRAMES", "1")))
        if n_kf == 1:
            kf_indices = [rep_idx]
        else:
            kf_indices = sorted(
                {
                    min(frame_idx - 1, int(i * max(frame_idx - 1, 1) / max(n_kf - 1, 1)))
                    for i in range(n_kf)
                }
            )
        print(f"  VLM: {len(kf_indices)} imagen(es) {kf_indices}", flush=True)
        vlm_in_dir = out / "vlm_input" if P.VLM_SAVE_INPUT else None
        cap2 = cv2.VideoCapture(str(video_path))
        for ki, kfi in enumerate(kf_indices):
            # Un solo keyframe no tiene desplazamiento propio; se usa una
            # ventana de frames alrededor para poder detectar caminata real
            # (sin esto, un keyframe "de pie" enmascara a alguien en movimiento)
            # y, dentro de ella, el frame con más IDs objetivo visibles a la
            # vez (evita perder a alguien solo porque el keyframe exacto lo
            # tapaba momentáneamente).
            win = max(int(round(fps)), 5)
            wsf, wef = max(0, kfi - win), min(len(all_frame_dets), kfi + win + 1)
            buffer_dets = all_frame_dets[wsf:wef]
            rel_idx, dets_kf = P.pick_representative_frame(buffer_dets, vlm_ids)
            if not dets_kf:
                continue
            kfi_eff = wsf + rel_idx
            cap2.set(cv2.CAP_PROP_POS_FRAMES, kfi_eff)
            _, frame_kf = cap2.read()
            if frame_kf is None:
                continue
            vlm_path = (vlm_in_dir / f"frame_{kfi:05d}.jpg") if vlm_in_dir else None
            vlm_out = P.infer_actions_image(
                vlm, proc, frame_kf, dets_kf, vlm_input_path=vlm_path
            )
            acts, soc = P.refine_chunk_labels(
                vlm_out.actions, vlm_out.social_states, buffer_dets, vlm_ids, (w, h)
            )
            vlm_out = vlm_out._replace(actions=acts, social_states=soc)
            vlm_total_s += vlm_out.elapsed_s
            segment_actions.append((kfi, vlm_out.actions))
            segment_social.append((kfi, vlm_out.social_states))
            vlm_chunk_timings.append(
                {
                    "chunk_index": ki,
                    "start_frame": kfi,
                    "latency_s": round(vlm_out.elapsed_s, 4),
                }
            )
            print(
                f"    img {ki + 1}/{len(kf_indices)} f{kfi}: "
                f"actions={vlm_out.actions} social={vlm_out.social_states} "
                f"(inferencia {vlm_out.elapsed_s:.2f}s, acum {vlm_total_s:.2f}s)",
                flush=True,
            )
        cap2.release()

    if not segment_actions:
        raise SystemExit("VLM no produjo acciones")

    keyframe_actions = segment_actions
    keyframe_social = segment_social
    actions = P.merge_action_votes([a for _, a in keyframe_actions])
    if P.social_state_mode() == "map":
        social_merged = P.social_states_from_actions(actions)
    else:
        social_merged = P.merge_social_votes([s for _, s in keyframe_social])
    vlm_s = vlm_total_s

    (out / "actions.json").write_text(
        json.dumps(
            {
                "social_state_mode": P.social_state_mode(),
                "merged": {str(k): v for k, v in actions.items()},
                "social_merged": {str(k): v for k, v in social_merged.items()},
                "keyframes": [
                    {
                        "frame": kfi,
                        "actions": {str(k): v for k, v in acts.items()},
                        "social_states": {
                            str(k): v
                            for k, v in (
                                keyframe_social[i][1]
                                if i < len(keyframe_social)
                                else P.social_states_from_actions(acts)
                            ).items()
                        },
                    }
                    for i, (kfi, acts) in enumerate(keyframe_actions)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Video final con acciones (sin re-ejecutar YOLO)
    print("Generando video con acciones …", flush=True)
    t_render0 = time.perf_counter()
    actions_mp4 = out / "annotated_actions.mp4"
    # cv2.VideoWriter con mp4v cuantiza mal colores saturados (ej. cajas rojas
    # de ENGAGED) cuando escribe muchos frames de bajo movimiento seguidos —
    # el re-encode a H.264 posterior no lo arregla porque el daño ya ocurrió
    # en ese paso con pérdida. Se escribe este video directo a ffmpeg (pipe
    # crudo → libx264) para no pasar por ese codec intermedio.
    writer2 = FFmpegPipeWriter(actions_mp4, fps, (w, h))
    cap3 = cv2.VideoCapture(str(video_path))
    fi = 0
    while True:
        ok, frame = cap3.read()
        if not ok or fi >= len(all_frame_dets):
            break
        frame = _crop_frame(frame, w, h)
        dets = all_frame_dets[fi]
        pipeline_total_s = time.perf_counter() - t_pipeline0
        frame_actions = P.actions_for_video_frame(fi, keyframe_actions)
        frame_social = P.social_for_video_frame(fi, keyframe_social)
        cur_chunk = (
            (fi // chunk_frames) if vlm_input_kind == "chunks" and chunk_sec > 0 else 0
        )
        ann = frame.copy()
        for d in dets:
            social_state = frame_social.get(d["pid"])
            label = P.format_detection_label(
                d["pid"],
                frame_actions.get(d["pid"], ""),
                social_state=social_state,
                class_name=d.get("class_name", "person"),
            )
            P.draw_box(
                ann,
                d["x1"],
                d["y1"],
                d["x2"],
                d["y2"],
                label,
                color=P.social_box_color(social_state),
            )
        P.draw_banner_video(
            ann,
            yolo_total_s=yolo_s,
            n_frames=frame_idx,
            vlm_total_s=vlm_total_s,
            vlm_calls=len(keyframe_actions),
            n_people=len(vlm_ids),
            pipeline_total_s=pipeline_total_s,
            frame_i=fi,
            vlm_input_kind=vlm_input_kind,
            chunk_sec=chunk_sec if vlm_input_kind == "chunks" else 0.0,
            chunk_index=cur_chunk,
            chunk_count=n_chunks,
        )
        writer2.write(ann)
        fi += 1
    cap3.release()
    writer2.release()
    render_s = time.perf_counter() - t_render0
    print("  Re-codificando H.264 (nitidez) …", flush=True)
    t_enc0 = time.perf_counter()
    finalize_video_h264(yolo_mp4)
    finalize_video_h264(vlm_mp4)
    encode_s = time.perf_counter() - t_enc0
    pipeline_total_s = time.perf_counter() - t_pipeline0
    rt = _realtime_metrics(
        frames=frame_idx,
        fps=fps,
        yolo_s=yolo_s,
        vlm_s=vlm_total_s,
        render_s=render_s,
        encode_s=encode_s,
        pipeline_s=pipeline_total_s,
    )
    print(
        f"  Tiempo total: {pipeline_total_s:.1f}s "
        f"(YOLO {yolo_s:.1f}s + VLM {vlm_total_s:.2f}s + render {render_s:.1f}s "
        f"+ encode {encode_s:.1f}s + carga VLM {vlm_load_s:.1f}s)",
        flush=True,
    )
    print(
        f"  Video {rt['video_duration_s']:.1f}s | factor tiempo real {rt['realtime_factor']:.2f}x "
        f"({'OK' if rt['pipeline_realtime_ok'] else 'más lento que tiempo real'})",
        flush=True,
    )

    full_actions = {pid: actions.get(pid, "unknown") for pid in sorted(all_pids)}
    summary = {
        "video": video_path.name,
        "frames": frame_idx,
        "fps": fps,
        "num_track_ids": len(all_pids),
        "vlm_person_ids": vlm_ids,
        "vlm_mode": vlm_input_kind,
        "vlm_chunk_sec": chunk_sec,
        "vlm_chunks": len(keyframe_actions),
        "vlm_segment_starts": [kfi for kfi, _ in keyframe_actions],
        "vlm_inferences": len(keyframe_actions),
        "yolo_total_s": round(yolo_s, 4),
        "yolo_ms_per_frame": round(1000.0 * yolo_s / max(frame_idx, 1), 2),
        "vlm_load_s": round(vlm_load_s, 4),
        "vlm_total_s": round(vlm_total_s, 4),
        "render_s": round(render_s, 4),
        "pipeline_total_s": round(pipeline_total_s, 4),
        **rt,
        "social_state_mode": P.social_state_mode(),
        "actions_merged": {str(k): v for k, v in actions.items()},
        "social_states_merged": {str(k): v for k, v in social_merged.items()},
        "actions": {str(k): v for k, v in full_actions.items()},
        "social_states": {
            str(k): v
            for k, v in P.resolve_social_states(
                full_actions,
                {int(k): v for k, v in social_merged.items()},
                sorted(all_pids),
            ).items()
        },
        "vlm_chunk_timings": vlm_chunk_timings,
        "vlm_chunk_latencies_s": [t["latency_s"] for t in vlm_chunk_timings],
        **vlm_timing_stats([t["latency_s"] for t in vlm_chunk_timings]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_vlm_latency_report(video_path.name, vlm_chunk_timings)
    print(f"Listo → {out} ({pipeline_total_s:.1f}s total)")
    return out


def list_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in P.VIDEO_EXTS
    )


def write_batch_summary(rows: list[dict], base_out: Path) -> Path:
    """Resumen global tras `python main.py videos` (tiempos + viabilidad tiempo real)."""
    if not rows:
        raise ValueError("Sin videos procesados")
    n = len(rows)
    all_chunk_latencies: list[float] = []
    for r in rows:
        all_chunk_latencies.extend(r.get("vlm_chunk_latencies_s") or [])
    global_vlm = vlm_timing_stats(all_chunk_latencies)
    per_video_avg = [r.get("vlm_latency_avg_s", 0.0) for r in rows if r.get("vlm_latency_count")]
    tot = {
        "videos": n,
        "total_frames": sum(r["frames"] for r in rows),
        "total_video_duration_s": round(sum(r["video_duration_s"] for r in rows), 2),
        "total_pipeline_s": round(sum(r["pipeline_total_s"] for r in rows), 2),
        "total_yolo_s": round(sum(r["yolo_total_s"] for r in rows), 2),
        "total_vlm_s": round(sum(r["vlm_total_s"] for r in rows), 2),
        "total_render_s": round(sum(r["render_s"] for r in rows), 2),
        "total_encode_s": round(sum(r["encode_s"] for r in rows), 2),
        "avg_realtime_factor": round(
            sum(r["realtime_factor"] for r in rows) / n, 2
        ),
        "yolo_realtime_count": sum(1 for r in rows if r["yolo_realtime_ok"]),
        "pipeline_realtime_count": sum(1 for r in rows if r["pipeline_realtime_ok"]),
        "vlm_inferences_total": global_vlm["vlm_latency_count"],
        "vlm_latency_global_avg_s": global_vlm["vlm_latency_avg_s"],
        "vlm_latency_global_min_s": global_vlm["vlm_latency_min_s"],
        "vlm_latency_global_max_s": global_vlm["vlm_latency_max_s"],
        "vlm_latency_avg_per_video_s": round(
            sum(per_video_avg) / len(per_video_avg), 4
        )
        if per_video_avg
        else 0.0,
    }
    yolo_ok = tot["yolo_realtime_count"]
    full_ok = tot["pipeline_realtime_count"]
    if full_ok == n:
        verdict = "Pipeline completo podría acercarse a tiempo real solo con optimización fuerte (hoy no)."
    elif yolo_ok == n:
        verdict = (
            "YOLO sí va en tiempo real; el cuello de botella es el VLM (trozos cada 2s). "
            "Para streaming: YOLO online + VLM cada N segundos en otro hilo."
        )
    else:
        verdict = "Ni YOLO ni pipeline completo alcanzan tiempo real en estos videos/resolución."

    batch = {
        "videos_processed": n,
        "totals": tot,
        "realtime_verdict": verdict,
        "per_video": rows,
    }
    base_out.mkdir(parents=True, exist_ok=True)
    out_json = base_out / "batch_summary.json"
    out_json.write_text(json.dumps(batch, indent=2), encoding="utf-8")

    import csv

    out_csv = base_out / "batch_summary.csv"
    fields = [
        "video",
        "frames",
        "video_duration_s",
        "pipeline_total_s",
        "yolo_total_s",
        "vlm_total_s",
        "vlm_inferences",
        "vlm_latency_count",
        "vlm_latency_avg_s",
        "vlm_latency_min_s",
        "vlm_latency_max_s",
        "render_s",
        "encode_s",
        "realtime_factor",
        "pipeline_realtime_ok",
        "yolo_realtime_ok",
        "processing_fps",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    timing_csv = base_out / "vlm_timings_per_chunk.csv"
    with timing_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["video", "chunk_index", "start_frame", "latency_s"],
        )
        w.writeheader()
        for r in rows:
            vname = r.get("video", "")
            for t in r.get("vlm_chunk_timings") or []:
                w.writerow(
                    {
                        "video": vname,
                        "chunk_index": t.get("chunk_index"),
                        "start_frame": t.get("start_frame"),
                        "latency_s": t.get("latency_s"),
                    }
                )

    print(f"\n{'=' * 60}")
    print("RESUMEN BATCH")
    print(f"  Videos: {n}")
    print(f"  Duración video total: {tot['total_video_duration_s']:.1f}s")
    print(f"  Tiempo procesado total: {tot['total_pipeline_s']:.1f}s")
    print(f"  YOLO: {tot['total_yolo_s']:.1f}s | VLM: {tot['total_vlm_s']:.1f}s")
    print(
        f"  VLM inferencias: {tot['vlm_inferences_total']} | "
        f"latencia media global: {tot['vlm_latency_global_avg_s']:.3f}s "
        f"(min {tot['vlm_latency_global_min_s']:.3f}s, max {tot['vlm_latency_global_max_s']:.3f}s)"
    )
    print(
        f"  Promedio de promedios por video: {tot['vlm_latency_avg_per_video_s']:.3f}s"
    )
    print(f"  Factor tiempo real medio: {tot['avg_realtime_factor']:.2f}x")
    print(f"  YOLO tiempo real: {yolo_ok}/{n} videos")
    print(f"  Pipeline completo tiempo real: {full_ok}/{n} videos")
    print(f"  → {verdict}")
    print(f"  Guardado: {out_json}")
    print(f"           {out_csv}")
    print(f"           {timing_csv}")
    print("=" * 60)
    return out_json


def run_all(input_dir: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    """Procesa todos los videos en input_videos/ + resumen batch."""
    indir = Path(input_dir or ROOT / "input_videos")
    base_out = Path(output_dir or VIDEO_OUTPUT_DIR)
    paths = list_videos(indir)
    if not paths:
        raise SystemExit(f"No hay videos en {indir}")
    print(f"Procesando {len(paths)} videos en {indir}\n")

    device = P.resolve_device()
    yolo = YOLO(P.YOLO_WEIGHTS)
    P.warmup_yolo(yolo)
    print("Cargando VLM una vez para todo el batch …", flush=True)
    t0 = time.perf_counter()
    vlm, processor = P.load_vlm(P.VLM_ID, device)
    if P.VLM_WARMUP:
        P.warmup_vlm(vlm, processor)
    print(f"  VLM listo en {time.perf_counter() - t0:.1f}s\n", flush=True)

    outs: list[Path] = []
    rows: list[dict] = []
    for i, path in enumerate(paths, 1):
        print(f"{'=' * 60}\n[{i}/{len(paths)}] {path.name}\n{'=' * 60}")
        out = run(
            video_path=path,
            output_dir=base_out,
            yolo=yolo,
            vlm=vlm,
            processor=processor,
            device=device,
        )
        outs.append(out)
        rows.append(json.loads((out / "summary.json").read_text(encoding="utf-8")))

    write_batch_summary(rows, base_out)
    return outs


if __name__ == "__main__":
    run()
