"""Pipeline de video offline: YOLO en todos los frames -> video anotado -> VLM.

Un solo entrypoint `run()`/`run_all()` para las dos variantes que existían
como módulos separados (run_video.py / run_video_pose.py): `use_pose=True`
(default, con marcha por piernas vía yolo26n-pose) o `use_pose=False` (solo
detección yolo11n, sin señal de gait). Las dos implementaciones divergen de
verdad en cómo procesan cada frame (lote+pose vs. frame a frame) y en el
render final, así que se mantienen como funciones privadas separadas
(`_run_plain`/`_run_pose`) en vez de forzar un solo cuerpo — `run()` solo
elige cuál correr.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

from .. import video_io
from ..config import (
    MAX_TRACK_IDS,
    POSE_YOLO_WEIGHTS,
    ROOT,
    VIDEO_VLM_CHUNK_SEC,
    VIDEO_VLM_FPS,
    VLM_BATCH_SIZE,
    VLM_ID,
    VLM_SAVE_INPUT,
    VLM_VIDEO_MAX_PIXELS,
    VLM_WARMUP,
    YOLO_WEIGHTS,
    chunk_vlm_use_image,
    video_vlm_fps,
)
from ..actions import normalize_action
from ..detection import (
    assign_spatial_ids,
    detect_people,
    dets_for_vlm_prompt,
    pick_representative_frame,
    resolve_device,
    stable_person_ids,
    track_detections,
    warmup_yolo,
)
from ..drawing import (
    annotate_for_vlm,
    draw_banner_video,
    draw_box,
    format_detection_label,
    social_box_color,
)
from ..motion import refine_chunk_labels
from ..pose import gait as pose_gait
from ..social_state import (
    merge_social_votes,
    resolve_social_states,
    social_state_mode,
    social_states_from_actions,
)
from ..vlm.inference import (
    _generate_json,
    _generate_json_batch,
    actions_for_video_frame,
    infer_actions_image,
    infer_actions_video,
    infer_actions_video_batch,
    merge_action_votes,
    social_for_video_frame,
    warmup_vlm,
)
from ..vlm.model import load_vlm

VIDEO_INPUT = Path(
    os.environ.get(
        "VIDEO_INPUT",
        str(ROOT / "input_videos" / "youtube_short_935cb0IrwEw.mp4"),
    )
)
_VIDEO_OUTPUT_DIR_PLAIN = Path(os.environ.get("VIDEO_OUTPUT_DIR", str(ROOT / "output_video")))
_VIDEO_OUTPUT_DIR_POSE = Path(os.environ.get("VIDEO_OUTPUT_DIR", str(ROOT / "output")))


def _run_plain(
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
    video_path = video_io.ensure_opencv_video(video_path)
    t_pipeline0 = time.perf_counter()

    out = Path(output_dir or _VIDEO_OUTPUT_DIR_PLAIN) / video_path.stem.replace("_h264", "")
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = video_io._even_dims(
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
        device = resolve_device()
    print(f"Video: {video_path.name} ({n_frames} frames @ {fps:.1f} fps)")
    print(f"VLM={VLM_ID} | salida: {out}")
    if yolo is None:
        yolo = YOLO(YOLO_WEIGHTS)
    warmup_yolo(yolo, shape=(h, w, 3))

    # Config de trozos VLM calculada antes de la Fase 1 para poder escribir
    # cada trozo directamente durante el único recorrido secuencial del video
    # (evita reabrir/seekear vlm_mp4 por cada trozo en la Fase 2).
    chunk_sec = float(os.environ.get("VIDEO_VLM_CHUNK_SEC", str(VIDEO_VLM_CHUNK_SEC)))
    vlm_mode = os.environ.get(
        "VIDEO_VLM_MODE", "chunks" if chunk_sec > 0 else "video"
    ).lower()
    chunk_frames = max(1, int(chunk_sec * fps)) if chunk_sec > 0 else n_frames
    use_img = chunk_vlm_use_image(chunk_sec)
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
        frame = video_io._crop_frame(frame, w, h)
        raw = detect_people(yolo, frame)
        if prev_dets:
            dets = track_detections(prev_dets, raw)
        else:
            dets = assign_spatial_ids(raw)
        prev_dets = dets
        all_frame_dets.append(dets)
        for d in dets:
            all_pids.add(d["pid"])

        ann = frame.copy()
        for d in dets:
            draw_box(
                ann, d["x1"], d["y1"], d["x2"], d["y2"], f"ID{d['pid']} {d['class_name']}"
            )
        writer.write(ann)
        ann_vlm = annotate_for_vlm(frame, dets)
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

    stable_ids = stable_person_ids(frame_rows)
    vlm_ids = stable_ids if stable_ids else sorted(all_pids)[: MAX_TRACK_IDS]
    print(
        f"  Tracks totales: {len(all_pids)} | VLM sobre {len(vlm_ids)} IDs estables: {vlm_ids}",
        flush=True,
    )

    # —— Fase 2: VLM (por defecto trozos de VIDEO_VLM_CHUNK_SEC segundos) ——
    print(
        f"Fase 2/2: VLM (modo={vlm_mode}, trozo={chunk_sec}s, "
        f"social={social_state_mode()}) …",
        flush=True,
    )
    vlm_load_s = 0.0
    if vlm is None or processor is None:
        t_load = time.perf_counter()
        vlm, processor = load_vlm(VLM_ID, device)
        vlm_load_s = time.perf_counter() - t_load
        if VLM_WARMUP:
            warmup_vlm(vlm, processor)
    proc = processor

    rep_idx, rep_dets = pick_representative_frame(all_frame_dets, vlm_ids)
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
            else f"video @ {VIDEO_VLM_FPS} fps"
        )
        print(
            f"  VLM: {n_chunks} trozos de ~{chunk_sec}s ({chunk_frames} fr/trozo) → {vlm_chunk_mode}",
            flush=True,
        )
        cap_vlm = cv2.VideoCapture(str(vlm_mp4)) if use_img else None

        def _fallback_image(sf: int, ef: int):
            rel_idx, dets_frame = pick_representative_frame(
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
            return infer_actions_image(vlm, proc, fr, dets_frame, already_annotated=True)

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
                rel_idx, dets_mid = pick_representative_frame(
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
                vlm_out = infer_actions_image(
                    vlm, proc, fr, dets_mid, already_annotated=True
                )
                buffer_dets = all_frame_dets[sf:ef]
                acts, soc = refine_chunk_labels(
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
                dets_mid = dets_for_vlm_prompt(all_frame_dets[sf:ef], vlm_ids)
                if not dets_mid:
                    continue
                chunk_path = chunk_dir / f"chunk_{ci:03d}.mp4"
                if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
                    continue
                pending.append(
                    {"ci": ci, "sf": sf, "ef": ef, "path": chunk_path, "dets": dets_mid}
                )

            batch_size = max(1, VLM_BATCH_SIZE)
            for i in range(0, len(pending), batch_size):
                group = pending[i : i + batch_size]
                items = [(g["path"], vlm_ids, g["dets"]) for g in group]
                try:
                    results = infer_actions_video_batch(
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
                        acts, soc = refine_chunk_labels(
                            vlm_out.actions, vlm_out.social_states, buffer_dets, vlm_ids, (w, h)
                        )
                        vlm_out = vlm_out._replace(actions=acts, social_states=soc)
                    _emit(g["ci"], g["sf"], g["ef"], vlm_out)

    elif vlm_mode == "video":
        print(f"  VLM: clip completo {vlm_mp4.name} @ {VIDEO_VLM_FPS} fps", flush=True)
        try:
            vlm_out = infer_actions_video(
                vlm, proc, vlm_mp4, vlm_ids, dets=rep_dets
            )
            acts, soc = refine_chunk_labels(
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
        vlm_in_dir = out / "vlm_input" if VLM_SAVE_INPUT else None
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
            rel_idx, dets_kf = pick_representative_frame(buffer_dets, vlm_ids)
            if not dets_kf:
                continue
            kfi_eff = wsf + rel_idx
            cap2.set(cv2.CAP_PROP_POS_FRAMES, kfi_eff)
            _, frame_kf = cap2.read()
            if frame_kf is None:
                continue
            vlm_path = (vlm_in_dir / f"frame_{kfi:05d}.jpg") if vlm_in_dir else None
            vlm_out = infer_actions_image(
                vlm, proc, frame_kf, dets_kf, vlm_input_path=vlm_path
            )
            acts, soc = refine_chunk_labels(
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
    actions = merge_action_votes([a for _, a in keyframe_actions])
    if social_state_mode() == "map":
        social_merged = social_states_from_actions(actions)
    else:
        social_merged = merge_social_votes([s for _, s in keyframe_social])
    vlm_s = vlm_total_s

    (out / "actions.json").write_text(
        json.dumps(
            {
                "social_state_mode": social_state_mode(),
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
                                else social_states_from_actions(acts)
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
    writer2 = video_io.FFmpegPipeWriter(actions_mp4, fps, (w, h))
    cap3 = cv2.VideoCapture(str(video_path))
    fi = 0
    while True:
        ok, frame = cap3.read()
        if not ok or fi >= len(all_frame_dets):
            break
        frame = video_io._crop_frame(frame, w, h)
        dets = all_frame_dets[fi]
        pipeline_total_s = time.perf_counter() - t_pipeline0
        frame_actions = actions_for_video_frame(fi, keyframe_actions)
        frame_social = social_for_video_frame(fi, keyframe_social)
        cur_chunk = (
            (fi // chunk_frames) if vlm_input_kind == "chunks" and chunk_sec > 0 else 0
        )
        ann = frame.copy()
        for d in dets:
            social_state = frame_social.get(d["pid"])
            label = format_detection_label(
                d["pid"],
                frame_actions.get(d["pid"], ""),
                social_state=social_state,
                class_name=d.get("class_name", "person"),
            )
            draw_box(
                ann,
                d["x1"],
                d["y1"],
                d["x2"],
                d["y2"],
                label,
                color=social_box_color(social_state),
            )
        draw_banner_video(
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
    video_io.finalize_video_h264(yolo_mp4)
    video_io.finalize_video_h264(vlm_mp4)
    encode_s = time.perf_counter() - t_enc0
    pipeline_total_s = time.perf_counter() - t_pipeline0
    rt = video_io._realtime_metrics(
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
        "social_state_mode": social_state_mode(),
        "actions_merged": {str(k): v for k, v in actions.items()},
        "social_states_merged": {str(k): v for k, v in social_merged.items()},
        "actions": {str(k): v for k, v in full_actions.items()},
        "social_states": {
            str(k): v
            for k, v in resolve_social_states(
                full_actions,
                {int(k): v for k, v in social_merged.items()},
                sorted(all_pids),
            ).items()
        },
        "vlm_chunk_timings": vlm_chunk_timings,
        "vlm_chunk_latencies_s": [t["latency_s"] for t in vlm_chunk_timings],
        **video_io.vlm_timing_stats([t["latency_s"] for t in vlm_chunk_timings]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    video_io.print_vlm_latency_report(video_path.name, vlm_chunk_timings)
    print(f"Listo → {out} ({pipeline_total_s:.1f}s total)")
    return out


def _run_all_plain(input_dir: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    """Procesa todos los videos en input_videos/ + resumen batch."""
    indir = Path(input_dir or ROOT / "input_videos")
    base_out = Path(output_dir or _VIDEO_OUTPUT_DIR_PLAIN)
    paths = video_io.list_videos(indir)
    if not paths:
        raise SystemExit(f"No hay videos en {indir}")
    print(f"Procesando {len(paths)} videos en {indir}\n")

    device = resolve_device()
    yolo = YOLO(YOLO_WEIGHTS)
    warmup_yolo(yolo)
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
            yolo=yolo,
            vlm=vlm,
            processor=processor,
            device=device,
        )
        outs.append(out)
        rows.append(json.loads((out / "summary.json").read_text(encoding="utf-8")))

    video_io.write_batch_summary(rows, base_out)
    return outs


def _run_pose(
    video_path: Path,
    output_dir: Path | None = None,
    *,
    yolo: YOLO | None = None,
    yolo_pose: YOLO | None = None,
    vlm=None,
    processor=None,
    device=None,
) -> Path:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise SystemExit(f"No existe el video: {video_path}")
    video_path = video_io.ensure_opencv_video(video_path)
    t_pipeline0 = time.perf_counter()

    out = Path(output_dir or _VIDEO_OUTPUT_DIR_POSE) / video_path.stem.replace("_h264", "")
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = video_io._even_dims(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    yolo_mp4 = out / "annotated_yolo.mp4"
    vlm_mp4 = out / "vlm_input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(yolo_mp4), fourcc, fps, (w, h))
    vlm_writer = cv2.VideoWriter(str(vlm_mp4), fourcc, fps, (w, h))

    if device is None:
        device = resolve_device()
    # cudnn.benchmark=True (activado por resolve_device) autotunea por cada
    # combinación (lote, alto, ancho) NUEVA que ve — 20-40s cada una, medido.
    # Bien para el pipeline estándar (siempre lote=1, una sola forma), pero
    # aquí el tamaño de lote cambia por video (chunk_frames depende del fps)
    # y otra vez en el último trozo de cada uno (el resto): sin desactivarlo,
    # cada video pagaba ese autotuneo de nuevo. Se reactiva antes del VLM
    # (Fase 2), que sí repite SIEMPRE la misma forma y se beneficia de él.
    torch.backends.cudnn.benchmark = False
    print(f"Video: {video_path.name} ({n_frames} frames @ {fps:.1f} fps) [pose+gait]")
    print(f"VLM={VLM_ID} | salida: {out}")
    if yolo is None:
        yolo = YOLO(YOLO_WEIGHTS)
    if yolo_pose is None:
        yolo_pose = YOLO(POSE_YOLO_WEIGHTS)
    warmup_yolo(yolo, shape=(h, w, 3))
    warmup_yolo(yolo_pose, shape=(h, w, 3))

    chunk_sec = float(os.environ.get("VIDEO_VLM_CHUNK_SEC", str(VIDEO_VLM_CHUNK_SEC)))
    chunk_frames = max(1, int(chunk_sec * fps)) if chunk_sec > 0 else n_frames
    chunk_dir = out / "vlm_chunks"
    chunk_dir.mkdir(exist_ok=True)

    # —— Fase 1: YOLO-pose + tracking, por LOTES de un trozo (~1s) a la vez ——
    # Los dos modelos (detector + pose) están infrautilizados llamándolos
    # frame a frame (~6-8ms/frame medido); agrupar un trozo entero en una
    # sola llamada a predict() baja eso a ~2-2.3ms/frame (~3x) sin cambiar el
    # resultado (mismos parámetros, mismos frames). El tracking (que sí
    # depende del frame anterior) se aplica DESPUÉS, secuencialmente, sobre
    # los resultados ya calculados del lote.
    print("Fase 1/2: YOLO-pose + tracking (por lotes) …", flush=True)
    t_yolo0 = time.perf_counter()
    prev_dets: list[dict] = []
    frame_rows: list[dict] = []
    all_frame_dets: list[list[dict]] = []
    all_pids: set[int] = set()
    frame_idx = 0
    motion_hyst = pose_gait.MotionHysteresis()
    chunk_moving_pids: dict[int, set[int]] = {}

    def process_chunk(raw_frames: list, ci: int) -> None:
        nonlocal prev_dets, frame_idx
        batch_det = pose_gait.detect_people_batch(yolo, raw_frames)
        batch_pose = pose_gait.detect_people_pose_batch(yolo_pose, raw_frames)
        dets_chunk: list[list[dict]] = []
        for raw in batch_det:
            if prev_dets:
                dets = track_detections(prev_dets, raw)
            else:
                dets = assign_spatial_ids(raw)
            prev_dets = dets
            dets_chunk.append(dets)
        for dets, pose_raw in zip(dets_chunk, batch_pose):
            pose_gait.attach_pose_keypoints(dets, pose_raw)

        for fi_local, (frame, dets) in enumerate(zip(raw_frames, dets_chunk)):
            fi = frame_idx + fi_local
            all_frame_dets.append(dets)
            for d in dets:
                all_pids.add(d["pid"])
            ann = frame.copy()
            for d in dets:
                draw_box(
                    ann, d["x1"], d["y1"], d["x2"], d["y2"], f"ID{d['pid']} {d['class_name']}"
                )
            writer.write(ann)
            for d in dets:
                frame_rows.append(
                    {
                        "frame": fi,
                        "time_s": round(fi / fps, 4),
                        "person_id": d["pid"],
                        "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                        "class_name": d["class_name"],
                        "conf": round(d["conf"], 4),
                    }
                )
        frame_idx += len(raw_frames)
        print(f"  … frame {frame_idx}/{n_frames}", flush=True)

        pids_in_chunk = sorted({d["pid"] for row in dets_chunk for d in row})
        gait = pose_gait.chunk_gait_by_pid(dets_chunk, pids_in_chunk)
        depth = pose_gait.chunk_depth_motion_by_pid(dets_chunk, pids_in_chunk)
        raw_moving = {pid for pid in pids_in_chunk if gait.get(pid) or depth.get(pid)}
        moving_pids = motion_hyst.update(ci, raw_moving, set(pids_in_chunk))
        chunk_moving_pids[ci] = moving_pids
        cw = cv2.VideoWriter(str(chunk_dir / f"chunk_{ci:03d}.mp4"), fourcc, fps, (w, h))
        for fr, dets in zip(raw_frames, dets_chunk):
            ann_vlm = pose_gait.annotate_for_vlm_pose(fr, dets, moving_pids)
            vlm_writer.write(ann_vlm)
            cw.write(ann_vlm)
        cw.release()

    raw_buf: list = []
    ci = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_buf.append(video_io._crop_frame(frame, w, h))
        if len(raw_buf) >= chunk_frames:
            process_chunk(raw_buf, ci)
            ci += 1
            raw_buf = []
    if raw_buf:
        process_chunk(raw_buf, ci)

    cap.release()
    writer.release()
    vlm_writer.release()
    yolo_s = time.perf_counter() - t_yolo0
    print(f"  YOLO-pose listo: {frame_idx} frames en {yolo_s:.1f}s → {yolo_mp4.name}")

    import pandas as pd

    pd.DataFrame(frame_rows).to_csv(out / "detections_per_frame.csv", index=False)

    if not all_pids:
        print("No se detectaron personas; se omite VLM.")
        raise SystemExit("Sin personas detectadas")

    stable_ids = stable_person_ids(frame_rows)
    vlm_ids = stable_ids if stable_ids else sorted(all_pids)[: MAX_TRACK_IDS]
    if len(vlm_ids) > pose_gait.MAX_VLM_PEOPLE:
        dropped = len(vlm_ids) - pose_gait.MAX_VLM_PEOPLE
        vlm_ids = pose_gait.closest_n_pids(frame_rows, vlm_ids, pose_gait.MAX_VLM_PEOPLE)
        print(
            f"  {dropped} persona(s) de sobra para el VLM (más de {pose_gait.MAX_VLM_PEOPLE}); "
            f"se priorizan las más cercanas a la cámara.",
            flush=True,
        )
    print(f"  Tracks totales: {len(all_pids)} | VLM sobre {len(vlm_ids)} IDs estables: {vlm_ids}", flush=True)

    # —— Fase 2: VLM sobre los trozos ya escritos en disco ——
    # Aquí la forma SÍ es estable (siempre el mismo resize de video/imagen),
    # así que cudnn.benchmark vuelve a valer la pena.
    torch.backends.cudnn.benchmark = True
    print(f"Fase 2/2: VLM (trozo={chunk_sec}s, social={social_state_mode()}) …", flush=True)
    vlm_load_s = 0.0
    if vlm is None or processor is None:
        t_load = time.perf_counter()
        vlm, processor = load_vlm(VLM_ID, device)
        vlm_load_s = time.perf_counter() - t_load
        if VLM_WARMUP:
            warmup_vlm(vlm, processor)
    proc = processor

    n_chunks = max(1, (frame_idx + chunk_frames - 1) // chunk_frames)
    segment_actions: list[tuple[int, dict[int, str]]] = []
    segment_social: list[tuple[int, dict[int, str]]] = []
    vlm_chunk_timings: list[dict] = []
    vlm_total_s = 0.0

    pending = []
    for ci, sf in enumerate(range(0, frame_idx, chunk_frames)):
        ef = min(frame_idx, sf + chunk_frames)
        if ef <= sf:
            continue
        dets_mid = dets_for_vlm_prompt(all_frame_dets[sf:ef], vlm_ids)
        if not dets_mid:
            continue
        chunk_path = chunk_dir / f"chunk_{ci:03d}.mp4"
        if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
            continue
        pending.append({"ci": ci, "sf": sf, "ef": ef, "path": chunk_path, "dets": dets_mid})

    def _emit(ci: int, sf: int, ef: int, vlm_out) -> None:
        nonlocal vlm_total_s
        if vlm_out is None:
            return
        vlm_total_s += vlm_out.elapsed_s
        segment_actions.append((sf, vlm_out.actions))
        segment_social.append((sf, vlm_out.social_states))
        vlm_chunk_timings.append(
            {"chunk_index": ci, "start_frame": sf, "latency_s": round(vlm_out.elapsed_s, 4)}
        )
        print(
            f"    trozo {ci + 1}/{n_chunks} frames {sf}-{ef}: "
            f"actions={vlm_out.actions} social={vlm_out.social_states} "
            f"(inferencia {vlm_out.elapsed_s:.2f}s, acum VLM {vlm_total_s:.2f}s)",
            flush=True,
        )

    vfps = video_vlm_fps(chunk_sec)
    batch_size = max(1, VLM_BATCH_SIZE)
    for i in range(0, len(pending), batch_size):
        group = pending[i : i + batch_size]
        # El prompt usa la MISMA función que pipeline.py + una nota sobre el
        # color de caja (pista de movimiento independiente del VLM).
        batch_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(g["path"].resolve()),
                            "fps": vfps,
                            "max_pixels": VLM_VIDEO_MAX_PIXELS,
                        },
                        {"type": "text", "text": pose_gait.vlm_prompt_with_motion_hint(g["dets"], video=True)},
                    ],
                }
            ]
            for g in group
        ]
        batch_person_ids = [vlm_ids for _ in group]
        try:
            results = _generate_json_batch(vlm, proc, batch_messages, batch_person_ids)
        except Exception as e:
            print(f"    lote falló ({e}); trozo a trozo …", flush=True)
            results = []
            for g, msgs in zip(group, batch_messages):
                try:
                    results.append(_generate_json(vlm, proc, msgs, vlm_ids))
                except Exception as e2:
                    print(f"    trozo {g['ci']} falló ({e2})", flush=True)
                    results.append(None)
        for g, vlm_out in zip(group, results):
            if vlm_out is not None:
                buffer_dets = all_frame_dets[g["sf"] : g["ef"]]
                acts, soc = pose_gait.refine_labels_pose(
                    vlm_out.actions, vlm_out.social_states, buffer_dets, vlm_ids, (w, h),
                    moving_pids_hint=chunk_moving_pids.get(g["ci"]),
                )
                soc = pose_gait.upgrade_attentive_by_gaze(soc, buffer_dets, vlm_ids)
                vlm_out = vlm_out._replace(actions=acts, social_states=soc)
            _emit(g["ci"], g["sf"], g["ef"], vlm_out)

    if not segment_actions:
        raise SystemExit("VLM no produjo acciones")

    keyframe_actions = segment_actions
    keyframe_social = segment_social
    actions = merge_action_votes([a for _, a in keyframe_actions])
    if social_state_mode() == "map":
        social_merged = social_states_from_actions(actions)
    else:
        social_merged = merge_social_votes([s for _, s in keyframe_social])

    (out / "actions.json").write_text(
        json.dumps(
            {
                "social_state_mode": social_state_mode(),
                "detector": "pose+gait (pose_pipeline.py)",
                "merged": {str(k): v for k, v in actions.items()},
                "social_merged": {str(k): v for k, v in social_merged.items()},
                "keyframes": [
                    {
                        "frame": kfi,
                        "actions": {str(k): v for k, v in acts.items()},
                        "social_states": {
                            str(k): v
                            for k, v in (
                                keyframe_social[i][1] if i < len(keyframe_social) else {}
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

    # —— Video final con acciones (igual que run_video.py) ——
    print("Generando video con acciones …", flush=True)
    t_render0 = time.perf_counter()
    actions_mp4 = out / "annotated_actions.mp4"
    writer2 = video_io.FFmpegPipeWriter(actions_mp4, fps, (w, h))
    cap3 = cv2.VideoCapture(str(video_path))
    fi = 0
    while True:
        ok, frame = cap3.read()
        if not ok or fi >= len(all_frame_dets):
            break
        frame = video_io._crop_frame(frame, w, h)
        dets = all_frame_dets[fi]
        pipeline_total_s = time.perf_counter() - t_pipeline0
        frame_actions = actions_for_video_frame(fi, keyframe_actions)
        frame_social = social_for_video_frame(fi, keyframe_social)
        cur_chunk = fi // chunk_frames
        ann = frame.copy()
        target_pid = pose_gait.pick_interaction_target(
            dets, {d["pid"]: frame_social.get(d["pid"]) for d in dets}
        )
        panel_entries: list[tuple[int, str, tuple[int, int, int]]] = []
        for d in dets:
            social_state = frame_social.get(d["pid"])
            color = social_box_color(social_state)
            pose_gait.draw_box_only(ann, d["x1"], d["y1"], d["x2"], d["y2"], d["pid"], color)
            act = normalize_action(frame_actions.get(d["pid"], ""))
            state_txt = social_state or "UNKNOWN"
            text = f"{state_txt} ({act})" if act and act != "unknown" else state_txt
            if d["pid"] == target_pid:
                text += " -Target-"
                # Resalte extra (amarillo) sobre la caja normal para que el
                # candidato a interacción se vea sin tener que leer el panel.
                cv2.rectangle(
                    ann, (d["x1"] - 3, d["y1"] - 3), (d["x2"] + 3, d["y2"] + 3),
                    (0, 255, 255), 2,
                )
            panel_entries.append((d["pid"], text, color))
        pose_gait.draw_side_panel(ann, panel_entries)
        draw_banner_video(
            ann, yolo_total_s=yolo_s, n_frames=frame_idx, vlm_total_s=vlm_total_s,
            vlm_calls=len(keyframe_actions), n_people=len(vlm_ids),
            pipeline_total_s=pipeline_total_s, frame_i=fi, vlm_input_kind="chunks",
            chunk_sec=chunk_sec, chunk_index=cur_chunk, chunk_count=n_chunks,
        )
        writer2.write(ann)
        fi += 1
    cap3.release()
    writer2.release()
    render_s = time.perf_counter() - t_render0
    print("  Re-codificando H.264 (nitidez) …", flush=True)
    t_enc0 = time.perf_counter()
    video_io.finalize_video_h264(yolo_mp4)
    video_io.finalize_video_h264(vlm_mp4)
    encode_s = time.perf_counter() - t_enc0
    pipeline_total_s = time.perf_counter() - t_pipeline0
    rt = video_io._realtime_metrics(
        frames=frame_idx, fps=fps, yolo_s=yolo_s, vlm_s=vlm_total_s,
        render_s=render_s, encode_s=encode_s, pipeline_s=pipeline_total_s,
    )
    print(
        f"  Tiempo total: {pipeline_total_s:.1f}s (YOLO-pose {yolo_s:.1f}s + VLM {vlm_total_s:.2f}s "
        f"+ render {render_s:.1f}s + encode {encode_s:.1f}s + carga VLM {vlm_load_s:.1f}s)",
        flush=True,
    )

    full_actions = {pid: actions.get(pid, "unknown") for pid in sorted(all_pids)}
    summary = {
        "video": video_path.name,
        "frames": frame_idx,
        "fps": fps,
        "num_track_ids": len(all_pids),
        "vlm_person_ids": vlm_ids,
        "vlm_mode": "chunks",
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
        "social_state_mode": social_state_mode(),
        "detector": "pose+gait",
        "actions_merged": {str(k): v for k, v in actions.items()},
        "social_states_merged": {str(k): v for k, v in social_merged.items()},
        "actions": {str(k): v for k, v in full_actions.items()},
        "social_states": {
            str(k): v
            for k, v in resolve_social_states(
                full_actions, {int(k): v for k, v in social_merged.items()}, sorted(all_pids)
            ).items()
        },
        "vlm_chunk_timings": vlm_chunk_timings,
        "vlm_chunk_latencies_s": [t["latency_s"] for t in vlm_chunk_timings],
        **video_io.vlm_timing_stats([t["latency_s"] for t in vlm_chunk_timings]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    video_io.print_vlm_latency_report(video_path.name, vlm_chunk_timings)
    print(f"Listo → {out} ({pipeline_total_s:.1f}s total)")
    return out


def _run_all_pose(input_dir: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    """Procesa todos los videos en input_videos/ con el pipeline de pose+gait."""
    indir = Path(input_dir or ROOT / "input_videos")
    base_out = Path(output_dir or _VIDEO_OUTPUT_DIR_POSE)
    paths = video_io.list_videos(indir)
    if not paths:
        raise SystemExit(f"No hay videos en {indir}")
    print(f"[pose+gait] Procesando {len(paths)} videos en {indir}\n")

    device = resolve_device()
    yolo = YOLO(YOLO_WEIGHTS)
    yolo_pose = YOLO(POSE_YOLO_WEIGHTS)
    warmup_yolo(yolo)
    warmup_yolo(yolo_pose)
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
        try:
            out = _run_pose(
                video_path=path, output_dir=base_out,
                yolo=yolo, yolo_pose=yolo_pose, vlm=vlm, processor=processor, device=device,
            )
        except SystemExit as e:
            print(f"  Omitido ({e})")
            continue
        outs.append(out)
        rows.append(json.loads((out / "summary.json").read_text(encoding="utf-8")))

    if rows:
        video_io.write_batch_summary(rows, base_out)
    return outs


def run(
    video_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    use_pose: bool = True,
    yolo: YOLO | None = None,
    yolo_pose: YOLO | None = None,
    vlm=None,
    processor=None,
    device=None,
) -> Path:
    """Procesa un video offline. `use_pose=True` (default): detector de pose +
    marcha por piernas (yolo26n-pose). `use_pose=False`: solo detección
    (yolo11n), sin señal de gait -- soporta además los modos de VLM
    alternativos (VIDEO_VLM_MODE=video|frame), que la variante de pose no
    tiene."""
    if use_pose:
        return _run_pose(
            video_path, output_dir,
            yolo=yolo, yolo_pose=yolo_pose, vlm=vlm, processor=processor, device=device,
        )
    return _run_plain(
        video_path, output_dir,
        yolo=yolo, vlm=vlm, processor=processor, device=device,
    )


def run_all(
    input_dir: Path | None = None, output_dir: Path | None = None, *, use_pose: bool = True
) -> list[Path]:
    """Procesa todos los videos de `input_dir` (default input_videos/)."""
    if use_pose:
        return _run_all_pose(input_dir, output_dir)
    return _run_all_plain(input_dir, output_dir)
