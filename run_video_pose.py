"""Variante de run_video.py que usa pose_pipeline.py (detector de pose +
marcha por piernas) en vez de solo detección de personas.

Deliberadamente un módulo APARTE — no toca run_video.py/pipeline.py/main.py.
Reutiliza de run_video.py todo lo que no cambia (IO de video, H.264, reporte
de tiempos, resumen batch) vía import. Solo soporta el modo por defecto
(trozos de ~1s, cada uno un clip de video para el VLM) — para los otros modos
(imagen, video completo, keyframes) usa run_video.py normal.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

import pipeline as P
import pose_pipeline as PP
import run_video as RV

ROOT = P.ROOT
VIDEO_OUTPUT_DIR = Path(os.environ.get("VIDEO_OUTPUT_DIR", str(ROOT / "output")))


def refine_labels_pose(
    actions: dict[int, str],
    social: dict[int, str],
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    frame_size: tuple[int, int],
    *,
    moving_pids_hint: set[int] | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Combina tres señales de movimiento en vez de una sola:
    1. El VLM ya vio la pose+color como pista y decidió con eso.
    2. El respaldo determinístico por cinemática de caja de pipeline.py
       (bbox-center, recalibrado).
    3. `moving_pids_hint`: marcha por piernas + cambio de profundidad de la
       caja + histéresis temporal (pose_pipeline.py), calculado en Fase 1.
    Si CUALQUIERA de las tres dice "se mueve", se respeta — medido: dejar
    todo en manos de una sola señal (el VLM+pista) perdía el caso difícil de
    caminata de frente a cámara.

    ANTES de eso, se corrige el sesgo contrario: el VLM a veces le pega
    "walking" a TODO el grupo a la vez sin que nadie se haya movido de verdad
    (medido: 4/5 personas "walking" con 3-36px de desplazamiento total en 60
    frames — nada). Si eso pasa, solo se respeta a quien SÍ tiene evidencia
    cinemática independiente."""
    # El respaldo por cinemática de caja de pipeline.py mide posición ABSOLUTA
    # en la imagen — si la cámara se mueve un poco (temblor, paneo), todas las
    # cajas se desplazan igual y parece que todo el mundo camina. Se le pasa
    # una copia con ese movimiento común (mediana entre personas) ya restado.
    compensated = PP.compensate_camera_motion(buffer_dets)
    bbox_confirmed = set(
        P.chunk_motion_by_pid(compensated, person_ids, frame_size).keys()
    )
    confirmed_moving = (moving_pids_hint or set()) | bbox_confirmed
    actions, social = PP.debias_group_walking(actions, social, person_ids, confirmed_moving)

    if moving_pids_hint:
        for pid in moving_pids_hint:
            if pid not in actions:
                continue
            act = P.normalize_action(actions[pid])
            tr = P._action_traits(act.lower())
            if not tr.get("walking"):
                extras = P._action_secondary_parts(tr)
                actions[pid] = " and ".join(["walking"] + extras) if extras else "walking"

    # Piernas visibles: si NO lo están (persona muy cerca de la cámara, solo
    # torso/cara en el frame), refine_chunk_labels no debe adivinar postura —
    # se confía en lo que el VLM describió (ver enrich_action_posture).
    legs_visible = PP.chunk_legs_visible_by_pid(compensated, person_ids)
    return P.refine_chunk_labels(
        actions, social, compensated, person_ids, frame_size,
        moving_pids=confirmed_moving, legs_visible=legs_visible,
    )


def upgrade_attentive_by_gaze(
    social: dict[int, str], buffer_dets: list[list[dict]], person_ids: list[int]
) -> dict[int, str]:
    """Corrige ATTENTIVE con la mirada real (marcha por pose, no el juicio
    suelto del VLM sobre "orientado a cámara"):
    - AVAILABLE → ATTENTIVE si SÍ mira de frente la mayoría del trozo.
    - ATTENTIVE → AVAILABLE si el VLM la puso pero la pose confirma que NO
      mira de frente (medido: el VLM a veces marca ATTENTIVE con la persona
      mirando a otro lado, basta con estar de pie/quieta orientada hacia la
      cámara en general).
    No toca ENGAGED/BUSY/MOVING, que ya son más específicos que ATTENTIVE."""
    facing = PP.chunk_facing_camera_by_pid(buffer_dets, person_ids)
    out = dict(social)
    for pid, is_facing in facing.items():
        cur = out.get(pid)
        if is_facing and cur == P.SOCIAL_AVAILABLE:
            out[pid] = P.SOCIAL_ATTENTIVE
        elif not is_facing and cur == P.SOCIAL_ATTENTIVE:
            out[pid] = P.SOCIAL_AVAILABLE
    return out


def run(
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
    video_path = RV.ensure_opencv_video(video_path)
    t_pipeline0 = time.perf_counter()

    out = Path(output_dir or VIDEO_OUTPUT_DIR) / video_path.stem.replace("_h264", "")
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = RV._even_dims(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    yolo_mp4 = out / "annotated_yolo.mp4"
    vlm_mp4 = out / "vlm_input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(yolo_mp4), fourcc, fps, (w, h))
    vlm_writer = cv2.VideoWriter(str(vlm_mp4), fourcc, fps, (w, h))

    if device is None:
        device = P.resolve_device()
    # cudnn.benchmark=True (activado por resolve_device) autotunea por cada
    # combinación (lote, alto, ancho) NUEVA que ve — 20-40s cada una, medido.
    # Bien para el pipeline estándar (siempre lote=1, una sola forma), pero
    # aquí el tamaño de lote cambia por video (chunk_frames depende del fps)
    # y otra vez en el último trozo de cada uno (el resto): sin desactivarlo,
    # cada video pagaba ese autotuneo de nuevo. Se reactiva antes del VLM
    # (Fase 2), que sí repite SIEMPRE la misma forma y se beneficia de él.
    torch.backends.cudnn.benchmark = False
    print(f"Video: {video_path.name} ({n_frames} frames @ {fps:.1f} fps) [pose+gait]")
    print(f"VLM={P.VLM_ID} | salida: {out}")
    if yolo is None:
        yolo = YOLO(P.YOLO_WEIGHTS)
    if yolo_pose is None:
        yolo_pose = YOLO(PP.POSE_YOLO_WEIGHTS)
    P.warmup_yolo(yolo, shape=(h, w, 3))
    P.warmup_yolo(yolo_pose, shape=(h, w, 3))

    chunk_sec = float(os.environ.get("VIDEO_VLM_CHUNK_SEC", str(P.VIDEO_VLM_CHUNK_SEC)))
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
    motion_hyst = PP.MotionHysteresis()
    chunk_moving_pids: dict[int, set[int]] = {}

    def process_chunk(raw_frames: list, ci: int) -> None:
        nonlocal prev_dets, frame_idx
        batch_det = PP.detect_people_batch(yolo, raw_frames)
        batch_pose = PP.detect_people_pose_batch(yolo_pose, raw_frames)
        dets_chunk: list[list[dict]] = []
        for raw in batch_det:
            if prev_dets:
                dets = P.track_detections(prev_dets, raw)
            else:
                dets = P.assign_spatial_ids(raw)
            prev_dets = dets
            dets_chunk.append(dets)
        for dets, pose_raw in zip(dets_chunk, batch_pose):
            PP.attach_pose_keypoints(dets, pose_raw)

        for fi_local, (frame, dets) in enumerate(zip(raw_frames, dets_chunk)):
            fi = frame_idx + fi_local
            all_frame_dets.append(dets)
            for d in dets:
                all_pids.add(d["pid"])
            ann = frame.copy()
            for d in dets:
                P.draw_box(
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
        gait = PP.chunk_gait_by_pid(dets_chunk, pids_in_chunk)
        depth = PP.chunk_depth_motion_by_pid(dets_chunk, pids_in_chunk)
        raw_moving = {pid for pid in pids_in_chunk if gait.get(pid) or depth.get(pid)}
        moving_pids = motion_hyst.update(ci, raw_moving, set(pids_in_chunk))
        chunk_moving_pids[ci] = moving_pids
        cw = cv2.VideoWriter(str(chunk_dir / f"chunk_{ci:03d}.mp4"), fourcc, fps, (w, h))
        for fr, dets in zip(raw_frames, dets_chunk):
            ann_vlm = PP.annotate_for_vlm_pose(fr, dets, moving_pids)
            vlm_writer.write(ann_vlm)
            cw.write(ann_vlm)
        cw.release()

    raw_buf: list = []
    ci = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_buf.append(RV._crop_frame(frame, w, h))
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

    stable_ids = P.stable_person_ids(frame_rows)
    vlm_ids = stable_ids if stable_ids else sorted(all_pids)[: P.MAX_TRACK_IDS]
    if len(vlm_ids) > PP.MAX_VLM_PEOPLE:
        dropped = len(vlm_ids) - PP.MAX_VLM_PEOPLE
        vlm_ids = PP.closest_n_pids(frame_rows, vlm_ids, PP.MAX_VLM_PEOPLE)
        print(
            f"  {dropped} persona(s) de sobra para el VLM (más de {PP.MAX_VLM_PEOPLE}); "
            f"se priorizan las más cercanas a la cámara.",
            flush=True,
        )
    print(f"  Tracks totales: {len(all_pids)} | VLM sobre {len(vlm_ids)} IDs estables: {vlm_ids}", flush=True)

    # —— Fase 2: VLM sobre los trozos ya escritos en disco ——
    # Aquí la forma SÍ es estable (siempre el mismo resize de video/imagen),
    # así que cudnn.benchmark vuelve a valer la pena.
    torch.backends.cudnn.benchmark = True
    print(f"Fase 2/2: VLM (trozo={chunk_sec}s, social={P.social_state_mode()}) …", flush=True)
    vlm_load_s = 0.0
    if vlm is None or processor is None:
        t_load = time.perf_counter()
        vlm, processor = P.load_vlm(P.VLM_ID, device)
        vlm_load_s = time.perf_counter() - t_load
        if P.VLM_WARMUP:
            P.warmup_vlm(vlm, processor)
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
        dets_mid = P.dets_for_vlm_prompt(all_frame_dets[sf:ef], vlm_ids)
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

    vfps = P.video_vlm_fps(chunk_sec)
    batch_size = max(1, P.VLM_BATCH_SIZE)
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
                            "max_pixels": P.VLM_VIDEO_MAX_PIXELS,
                        },
                        {"type": "text", "text": PP.vlm_prompt_with_motion_hint(g["dets"], video=True)},
                    ],
                }
            ]
            for g in group
        ]
        batch_person_ids = [vlm_ids for _ in group]
        try:
            results = P._generate_json_batch(vlm, proc, batch_messages, batch_person_ids)
        except Exception as e:
            print(f"    lote falló ({e}); trozo a trozo …", flush=True)
            results = []
            for g, msgs in zip(group, batch_messages):
                try:
                    results.append(P._generate_json(vlm, proc, msgs, vlm_ids))
                except Exception as e2:
                    print(f"    trozo {g['ci']} falló ({e2})", flush=True)
                    results.append(None)
        for g, vlm_out in zip(group, results):
            if vlm_out is not None:
                buffer_dets = all_frame_dets[g["sf"] : g["ef"]]
                acts, soc = refine_labels_pose(
                    vlm_out.actions, vlm_out.social_states, buffer_dets, vlm_ids, (w, h),
                    moving_pids_hint=chunk_moving_pids.get(g["ci"]),
                )
                soc = upgrade_attentive_by_gaze(soc, buffer_dets, vlm_ids)
                vlm_out = vlm_out._replace(actions=acts, social_states=soc)
            _emit(g["ci"], g["sf"], g["ef"], vlm_out)

    if not segment_actions:
        raise SystemExit("VLM no produjo acciones")

    keyframe_actions = segment_actions
    keyframe_social = segment_social
    actions = P.merge_action_votes([a for _, a in keyframe_actions])
    if P.social_state_mode() == "map":
        social_merged = P.social_states_from_actions(actions)
    else:
        social_merged = P.merge_social_votes([s for _, s in keyframe_social])

    (out / "actions.json").write_text(
        json.dumps(
            {
                "social_state_mode": P.social_state_mode(),
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
    writer2 = RV.FFmpegPipeWriter(actions_mp4, fps, (w, h))
    cap3 = cv2.VideoCapture(str(video_path))
    fi = 0
    while True:
        ok, frame = cap3.read()
        if not ok or fi >= len(all_frame_dets):
            break
        frame = RV._crop_frame(frame, w, h)
        dets = all_frame_dets[fi]
        pipeline_total_s = time.perf_counter() - t_pipeline0
        frame_actions = P.actions_for_video_frame(fi, keyframe_actions)
        frame_social = P.social_for_video_frame(fi, keyframe_social)
        cur_chunk = fi // chunk_frames
        ann = frame.copy()
        target_pid = PP.pick_interaction_target(
            dets, {d["pid"]: frame_social.get(d["pid"]) for d in dets}
        )
        panel_entries: list[tuple[int, str, tuple[int, int, int]]] = []
        for d in dets:
            social_state = frame_social.get(d["pid"])
            color = P.social_box_color(social_state)
            PP.draw_box_only(ann, d["x1"], d["y1"], d["x2"], d["y2"], d["pid"], color)
            act = P.normalize_action(frame_actions.get(d["pid"], ""))
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
        PP.draw_side_panel(ann, panel_entries)
        P.draw_banner_video(
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
    RV.finalize_video_h264(yolo_mp4)
    RV.finalize_video_h264(vlm_mp4)
    encode_s = time.perf_counter() - t_enc0
    pipeline_total_s = time.perf_counter() - t_pipeline0
    rt = RV._realtime_metrics(
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
        "social_state_mode": P.social_state_mode(),
        "detector": "pose+gait",
        "actions_merged": {str(k): v for k, v in actions.items()},
        "social_states_merged": {str(k): v for k, v in social_merged.items()},
        "actions": {str(k): v for k, v in full_actions.items()},
        "social_states": {
            str(k): v
            for k, v in P.resolve_social_states(
                full_actions, {int(k): v for k, v in social_merged.items()}, sorted(all_pids)
            ).items()
        },
        "vlm_chunk_timings": vlm_chunk_timings,
        "vlm_chunk_latencies_s": [t["latency_s"] for t in vlm_chunk_timings],
        **RV.vlm_timing_stats([t["latency_s"] for t in vlm_chunk_timings]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RV.print_vlm_latency_report(video_path.name, vlm_chunk_timings)
    print(f"Listo → {out} ({pipeline_total_s:.1f}s total)")
    return out


def run_all(input_dir: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    """Procesa todos los videos en input_videos/ con el pipeline de pose+gait."""
    indir = Path(input_dir or ROOT / "input_videos")
    base_out = Path(output_dir or VIDEO_OUTPUT_DIR)
    paths = RV.list_videos(indir)
    if not paths:
        raise SystemExit(f"No hay videos en {indir}")
    print(f"[pose+gait] Procesando {len(paths)} videos en {indir}\n")

    device = P.resolve_device()
    yolo = YOLO(P.YOLO_WEIGHTS)
    yolo_pose = YOLO(PP.POSE_YOLO_WEIGHTS)
    P.warmup_yolo(yolo)
    P.warmup_yolo(yolo_pose)
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
        try:
            out = run(
                video_path=path, output_dir=base_out,
                yolo=yolo, yolo_pose=yolo_pose, vlm=vlm, processor=processor, device=device,
            )
        except SystemExit as e:
            print(f"  Omitido ({e})")
            continue
        outs.append(out)
        rows.append(json.loads((out / "summary.json").read_text(encoding="utf-8")))

    if rows:
        RV.write_batch_summary(rows, base_out)
    return outs
