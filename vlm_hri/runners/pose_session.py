"""Sesión de streaming con pose reutilizable entre el CLI (cv2.VideoCapture,
ver runners/stream.py::_run_pose) y el nodo ROS2 (vlm_hri/ros/node.py) -- no
le importa de dónde vienen los frames/detecciones (propio YOLO+tracking o un
tracker LiDAR externo ya con IDs y posición real), solo necesita que alguien
llame `append_frame()`/`tick()` por cada frame ya detectado (+trackeado +
pose si aplica).

Toda la lógica de buffer de trozos, hilo VLM, refinamiento (marcha, mirada,
proximidad LiDAR) y render final vivía dentro de `_run_pose` mezclada con el
loop de captura de OpenCV -- se movió aquí tal cual, sin reescribir el
comportamiento, para que el nodo ROS2 no tenga que duplicarla.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Protocol

import cv2
import numpy as np

from .. import video_io
from ..clustering import cluster_by_distance
from ..config import VLM_SAVE_INPUT, chunk_vlm_use_image
from ..detection import dets_for_vlm_prompt
from ..drawing import draw_banner_video, social_box_color, ui_scale
from ..motion import strip_posture_prefix
from ..pose import gait as pose_gait
from ..social_state import SOCIAL_UNKNOWN
from ..target_filter import TargetFilter, TargetFilterConfig, chunk_record
from ..vlm.inference import VlmInferenceResult, infer_actions_image, infer_actions_video

_UNSET = object()  # distingue "target_pid no pasado" de "target_pid=None" (nadie elegible)


@dataclass
class RawFrame:
    """Frame crudo + detecciones (con keypoints) tal cual salen de Fase 1,
    ANTES de saber azul/rojo (eso solo se sabe al cerrar el trozo).

    Cada det puede traer opcionalmente `world_x`/`world_y` (posición real,
    p. ej. de un tracker LiDAR externo vía ROS2) -- ausentes en el CLI de
    cámara/video plano, donde no hay esa señal."""
    frame_i: int
    frame_bgr: np.ndarray
    dets: list[dict]


@dataclass
class PoseVlmChunkJob:
    chunk_index: int
    start_frame: int
    raw_frames: list[RawFrame]
    person_ids: list[int]
    out_path: Path
    fps: float
    size: tuple[int, int]


def _save_vlm_input(job: PoseVlmChunkJob, gait: dict[int, bool], moving_pids: set[int]) -> None:
    """VLM_SAVE_INPUT=1: junto al clip del VLM, el clip SIN anotar y las cajas
    por frame -- para re-probar prompts offline con la misma entrada."""
    import json

    video_io.write_chunk_video(
        [rf.frame_bgr for rf in job.raw_frames],
        job.out_path.with_name(job.out_path.stem + "_raw.mp4"), job.fps, job.size,
    )
    meta = {
        "chunk_index": job.chunk_index,
        "start_frame": job.start_frame,
        "person_ids": [int(p) for p in job.person_ids],
        "gait": sorted(int(p) for p in gait),
        "moving_pids": sorted(int(p) for p in moving_pids),
        "frames": [
            [{k: (int(d[k]) if k == "pid" else float(d[k])) for k in ("pid", "x1", "y1", "x2", "y2")} for d in rf.dets]
            for rf in job.raw_frames
        ],
    }
    job.out_path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


@dataclass
class PoseStreamState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    actions: dict[int, str] = field(default_factory=dict)
    social_states: dict[int, str] = field(default_factory=dict)
    pose: dict[int, str] = field(default_factory=dict)
    confidence: dict[int, int] = field(default_factory=dict)
    interrupt_cost: dict[int, int] = field(default_factory=dict)
    segment_actions: list[tuple[int, dict[int, str]]] = field(default_factory=list)
    segment_social: list[tuple[int, dict[int, str]]] = field(default_factory=list)
    segment_pose: list[tuple[int, dict[int, str]]] = field(default_factory=list)
    segment_confidence: list[tuple[int, dict[int, int]]] = field(default_factory=list)
    segment_interrupt_cost: list[tuple[int, dict[int, int]]] = field(default_factory=list)
    # Target vigente (el que va al overlay y a /vlm_hri/target_pose): el del
    # modo elegido de vlm_hri.target_filter -- "raw" = tal cual lo eligió el
    # VLM en el último trozo ("target"/"target_conf" del prompt), "ema"/"hyst"
    # = filtrado en el tiempo. None si nadie es apto.
    vlm_target: int | None = None
    vlm_target_conf: float | None = None
    # (start_frame, pid, conf) CRUDO del VLM por trozo, sin filtrar.
    segment_target: list[tuple[int, int | None, float | None]] = field(default_factory=list)
    # Votos + decisión de los 3 modos por trozo (ver target_filter.chunk_record).
    segment_target_filter: list[dict] = field(default_factory=list)
    vlm_chunk_timings: list[dict] = field(default_factory=list)
    vlm_total_s: float = 0.0
    vlm_calls: int = 0
    vlm_pending: bool = False
    yolo_total_s: float = 0.0
    frames_done: int = 0


class OutputSink(Protocol):
    """A dónde va el resultado renderizado -- el CLI escribe un mp4 (+ vista
    previa opcional), el nodo ROS2 publica tópicos. Ambos pueden coexistir en
    `PoseStreamSession.sinks` (lista)."""

    def write_frame(self, ann: np.ndarray) -> None: ...

    def write_social_update(
        self,
        actions: dict[int, str],
        social_states: dict[int, str],
        world_xy: dict[int, tuple[float, float]] | None,
        clusters: dict[int, int] | None,
        target_pid: int | None,
        *,
        confidence: dict[int, int] | None = None,
        interrupt_cost: dict[int, int] | None = None,
        pose: dict[int, str] | None = None,
        target_confidence: float | None = None,
    ) -> None: ...


@dataclass
class VideoFileSink:
    """Sink de hoy del CLI: escribe al mp4 de salida y, si hay, a la ventana/
    reproductor de vista previa en vivo -- tal cual el `_PoseStreamOutput` de
    antes, solo que ya no acopla la sesión al `cv2.VideoWriter` concreto."""

    writer: cv2.VideoWriter
    lock: threading.Lock
    live_preview: object | None = None  # vlm_hri.preview.LivePreview | None

    def write_frame(self, ann: np.ndarray) -> None:
        with self.lock:
            self.writer.write(ann)
            if self.live_preview is not None:
                try:
                    self.live_preview.write(ann)
                except BrokenPipeError:
                    pass

    def write_social_update(self, *args, **kwargs) -> None:
        pass  # el CLI no necesita esto, solo el video


def render_pose_frame(
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
    target_pid=_UNSET,
    target_confidence: float | None = None,
    pose: dict[int, str] | None = None,
) -> np.ndarray:
    """Video final para el usuario (no el que ve el VLM): caja de color + ID +
    etiqueta de postura/acción/intención pegada a cada caja + panel lateral +
    resalte del target. `draw_pose=True`: superpone además el esqueleto
    COCO-17 (visualización). `pose` (opcional, ver
    vlm_hri.pose.gait.chunk_posture_by_pid): postura verificada por
    pid -- si no se pasa, esa parte de la etiqueta muestra "?".

    `target_pid`: si el llamador ya lo tiene (el nodo ROS2: el que eligió el
    VLM en el último trozo), se pasa aquí para que el resalte de imagen use
    exactamente el mismo target que se publica en /vlm_hri/target_pose. Si no se pasa, se
    recalcula sin memoria (comportamiento de antes, usado por el preview de
    depuración de `stream.py --camera`, sin posición real que enviar a
    ningún lado)."""
    ann = rf.frame_bgr.copy()
    if target_pid is _UNSET:
        target_pid = pose_gait.pick_interaction_target(
            rf.dets, {d["pid"]: social_states.get(d["pid"]) for d in rf.dets}
        )
    pose = pose or {}
    panel_entries: list[tuple[int, str, tuple[int, int, int], bool]] = []
    for d in rf.dets:
        social_state = social_states.get(d["pid"])
        color = social_box_color(social_state)
        pose_gait.draw_box_only(ann, d["x1"], d["y1"], d["x2"], d["y2"], d["pid"], color)
        if draw_pose:
            pose_gait.draw_pose_skeleton(ann, d.get("kpts"))
        pose_txt = pose.get(d["pid"], "?")
        act = strip_posture_prefix(actions.get(d["pid"], "idle"))
        state_txt = social_state or "UNKNOWN"
        pose_gait.draw_box_details(
            ann, d["x1"], d["y1"], d["x2"], d["y2"], pose_txt, act, state_txt, color
        )
        text = f"{state_txt} | {pose_txt} | {act}"
        is_target = d["pid"] == target_pid
        if is_target and target_confidence is not None:
            text += f" | target p={target_confidence:.2f}"
        if is_target:
            cv2.rectangle(
                ann, (d["x1"] - 3, d["y1"] - 3), (d["x2"] + 3, d["y2"] + 3),
                (0, 255, 255), 2,
            )
            wx, wy = d.get("world_x"), d.get("world_y")
            if wx is not None and wy is not None:
                pose_gait.draw_target_coords(ann, d["x1"], d["y1"], float(wx), float(wy))
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


def _collect_world_xy(raw_frames: list[RawFrame]) -> dict[int, tuple[float, float]]:
    """Última posición real conocida por pid en el trozo (de tracks LiDAR
    externos vía ROS2, ver vlm_hri/ros/detections_bridge.py) -- vacío en el
    CLI de cámara/video plano, donde ningún det trae world_x/world_y."""
    out: dict[int, tuple[float, float]] = {}
    for rf in raw_frames:
        for d in rf.dets:
            wx, wy = d.get("world_x"), d.get("world_y")
            if wx is not None and wy is not None:
                out[d["pid"]] = (float(wx), float(wy))
    return out


def _target_in_frame(rf: RawFrame, target_pid: int | None) -> int | None:
    """El target del VLM solo vale en los frames donde esa persona aparece."""
    if target_pid is None or not any(d["pid"] == target_pid for d in rf.dets):
        return None
    return target_pid


def _flush_chunk_to_sinks(job: PoseVlmChunkJob, session: "PoseStreamSession") -> None:
    with session.state.lock:
        actions = dict(session.state.actions)
        social_states = dict(session.state.social_states)
        pose = dict(session.state.pose)
        confidence = dict(session.state.confidence)
        interrupt_cost = dict(session.state.interrupt_cost)
        vlm_target = session.state.vlm_target
        vlm_target_conf = session.state.vlm_target_conf
        target_rec = session.state.segment_target_filter[-1] if session.state.segment_target_filter else None
        yolo_acum = session.state.yolo_total_s
        vlm_acum = session.state.vlm_total_s
        vlm_n = session.state.vlm_calls
        vlm_pending = session.state.vlm_pending
    pipeline_now = time.perf_counter() - session.t_pipeline0
    chunk_count = max(job.chunk_index + 1, vlm_n)
    world_xy = _collect_world_xy(job.raw_frames)
    clusters = cluster_by_distance(world_xy) if world_xy else None
    target_pid = None
    for rf in job.raw_frames:
        target_pid = _target_in_frame(rf, vlm_target)
        ann = render_pose_frame(
            rf, actions, social_states,
            w=session.w, h=session.h, fps=session.fps, chunk_sec=session.chunk_sec,
            chunk_index=job.chunk_index, chunk_count=chunk_count,
            yolo_total_s=yolo_acum, vlm_total_s=vlm_acum, vlm_calls=vlm_n,
            pipeline_total_s=pipeline_now, vlm_pending=vlm_pending,
            draw_pose=session.draw_pose,
            target_pid=target_pid,
            target_confidence=vlm_target_conf,
            pose=pose,
        )
        for sink in session.sinks:
            sink.write_frame(ann)
    for sink in session.sinks:
        sink.write_social_update(
            actions, social_states, world_xy or None, clusters, target_pid,
            confidence=confidence, interrupt_cost=interrupt_cost, pose=pose,
            target_confidence=vlm_target_conf if target_pid is not None else None,
        )
        write_debug = getattr(sink, "write_target_debug", None)
        if write_debug is not None and target_rec is not None:
            write_debug(target_rec)


def _pose_vlm_worker(
    queue: Queue,
    vlm,
    processor,
    state: PoseStreamState,
    stop: threading.Event,
    session: "PoseStreamSession | None",
    target_filter: TargetFilter | None = None,
) -> None:
    motion_hyst = pose_gait.MotionHysteresis()
    target_filter = target_filter or TargetFilter(TargetFilterConfig())
    while True:
        if stop.is_set():
            try:
                job: PoseVlmChunkJob = queue.get_nowait()
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
        # chunk_depth_motion_by_pid (EXPERIMENTAL, sin calibrar -- ver su
        # docstring) queda fuera de raw_moving: en la práctica se dispara con
        # solo inclinarse/gesticular frente a la cámara (medido: alguien de
        # pie y quieto frente a un portátil, forzado a "walking" varios
        # trozos seguidos sin caminar). La marcha real (gait, ángulo de
        # rodilla) ya es suficiente y sí está calibrada.
        raw_moving = {p for p in pids if gait.get(p)}
        moving_pids = motion_hyst.update(job.chunk_index, raw_moving, set(pids))

        vlm_frames = [
            pose_gait.annotate_for_vlm_pose(rf.frame_bgr, rf.dets, moving_pids) for rf in job.raw_frames
        ]
        dets_prompt = dets_for_vlm_prompt(buffer_dets, pids)
        if VLM_SAVE_INPUT:
            _save_vlm_input(job, gait, moving_pids)

        chunk_sec = len(vlm_frames) / max(job.fps, 1e-6)
        t0 = time.perf_counter()
        try:
            if chunk_vlm_use_image(chunk_sec):
                mid = vlm_frames[len(vlm_frames) // 2]
                vlm_out = infer_actions_image(vlm, processor, mid, dets_prompt, already_annotated=True)
            elif video_io.write_chunk_video(vlm_frames, job.out_path, job.fps, job.size):
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
        conf = dict(vlm_out.confidence)
        cost = dict(vlm_out.interrupt_cost)
        if pids:
            # Misma lógica de refinamiento que el modo offline: debias del
            # sesgo grupal de "walking", forzado a caminar si hay evidencia
            # independiente, respaldo por cinemática de caja (con
            # compensación de movimiento de cámara), ATTENTIVE por mirada, y
            # (si hay posición real vía ROS2) ENGAGED corroborado por
            # proximidad LiDAR.
            world_xy = _collect_world_xy(job.raw_frames)
            clusters = cluster_by_distance(world_xy) if world_xy else None
            acts, social, pose = pose_gait.refine_labels_pose(
                acts, social, buffer_dets, pids, job.size, moving_pids_hint=moving_pids,
                world_xy=world_xy or None, clusters=clusters,
            )
            social = pose_gait.upgrade_attentive_by_gaze(social, buffer_dets, pids)
        else:
            pose = {}

        # Una muestra por trozo (no por frame), con el estado social ya
        # refinado -- los 3 modos se calculan siempre, `mode` elige cuál vale.
        seen_pids = set(pids)
        source = target_filter.cfg.source
        if source == "rules":
            # Voto crudo = reglas sobre las etiquetas 's' ya refinadas de este
            # trozo (ATTENTIVE > AVAILABLE > menor interrupt_cost > caja más
            # grande / dist_to_robot), con la última caja vista de cada pid.
            last_det: dict[int, dict] = {}
            for rf in job.raw_frames:
                for d in rf.dets:
                    if d["pid"] in seen_pids:
                        last_det[d["pid"]] = d
            raw_pid = pose_gait.pick_interaction_target(list(last_det.values()), social, cost or None)
            raw_conf = None
        else:
            raw_pid, raw_conf = vlm_out.target_pid, vlm_out.target_confidence
        decisions = target_filter.update(raw_pid, raw_conf, social, seen_pids)
        chosen = target_filter.selected(decisions)
        target_rec = chunk_record(
            job.chunk_index, job.start_frame, raw_pid, raw_conf,
            social, seen_pids, decisions, vlm_out.target_probs,
            source=source, vlm_pid=vlm_out.target_pid,
        )

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
            for pid, p in pose.items():
                state.pose[pid] = p
            for pid, c in conf.items():
                state.confidence[pid] = c
            for pid, c in cost.items():
                state.interrupt_cost[pid] = c
            state.segment_actions.append((job.start_frame, dict(acts)))
            state.segment_social.append((job.start_frame, dict(social)))
            state.segment_pose.append((job.start_frame, dict(pose)))
            state.segment_confidence.append((job.start_frame, dict(conf)))
            state.segment_interrupt_cost.append((job.start_frame, dict(cost)))
            state.vlm_chunk_timings.append(
                {"chunk_index": job.chunk_index, "start_frame": job.start_frame, "latency_s": round(vlm_out.elapsed_s, 4)}
            )
            # Se sobrescribe siempre (también con None): en modo "raw", si el
            # VLM dice que en este trozo nadie es apto, el robot deja de tener
            # target; en "ema"/"hyst" eso es solo un voto más.
            state.vlm_target = chosen.pid
            state.vlm_target_conf = chosen.confidence
            state.segment_target.append((job.start_frame, raw_pid, raw_conf))
            state.segment_target_filter.append(target_rec)
        if session is not None and job.raw_frames:
            _flush_chunk_to_sinks(job, session)
        print(
            f"  [VLM trozo {job.chunk_index}] f{job.start_frame}+ actions={acts} social={social} "
            f"gait={sorted(moving_pids)} target[{source}]={raw_pid} vlm={vlm_out.target_pid} p={vlm_out.target_confidence} "
            f"P(t=1)={ {p: round(v, 2) for p, v in vlm_out.target_probs.items()} } "
            f"[ema={decisions['ema'].pid} hyst={decisions['hyst'].pid} -> {target_filter.cfg.mode}={chosen.pid}] (inferencia {vlm_out.elapsed_s:.2f}s, acum VLM {state.vlm_total_s:.2f}s)",
            flush=True,
        )


class PoseStreamSession:
    """Envuelve el buffer de trozos + hilo VLM + refinamiento + render que
    antes vivía dentro de `_run_pose`. Quien la use (el CLI o el nodo ROS2)
    solo tiene que llamar, por cada frame ya detectado(+trackeado+pose):

        rf = session.append_frame(frame_bgr, dets)
        # (opcional: usar `rf`/`session.state` para una vista previa propia
        # ANTES de que este frame pueda disparar el envío de un trozo)
        session.tick(elapsed_s)

    y al terminar, `session.flush_and_stop()` para vaciar el último trozo
    pendiente, parar el hilo y obtener el resumen final."""

    def __init__(
        self,
        *,
        w: int,
        h: int,
        fps: float,
        chunk_sec: float,
        chunk_dir: Path,
        vlm,
        processor,
        draw_pose: bool = False,
        sinks: list[OutputSink] | None = None,
        target_filter: TargetFilterConfig | None = None,
        max_people: int | None = None,
    ) -> None:
        self.w, self.h, self.fps, self.chunk_sec = w, h, fps, chunk_sec
        # Cuántas personas (las de caja más grande = más cerca de la cámara)
        # recibe el VLM por trozo -- el resto se dibuja pero no se clasifica
        # ni puede ser target.
        self.max_people = max(1, max_people or pose_gait.MAX_VLM_PEOPLE)
        self.target_filter = TargetFilter(target_filter or TargetFilterConfig.from_env())
        self.chunk_dir = Path(chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.vlm, self.processor = vlm, processor
        self.draw_pose = draw_pose
        self.sinks: list[OutputSink] = sinks or []

        self.state = PoseStreamState()
        self.stop = threading.Event()
        self.t_pipeline0 = time.perf_counter()
        self.vlm_queue: Queue = Queue(maxsize=8)
        self._worker: threading.Thread | None = None

        self.chunk_frames = max(1, int(chunk_sec * fps))
        self.chunk_raw: list[RawFrame] = []
        self.frame_i = 0
        self.chunk_i = 0
        self.next_chunk_at = self.chunk_frames

    def start(self) -> None:
        self._worker = threading.Thread(
            target=_pose_vlm_worker,
            args=(self.vlm_queue, self.vlm, self.processor, self.state, self.stop, self, self.target_filter),
            daemon=True,
        )
        self._worker.start()

    def append_frame(self, frame_bgr: np.ndarray, dets: list[dict]) -> RawFrame:
        rf = RawFrame(self.frame_i, frame_bgr, dets)
        self.chunk_raw.append(rf)
        return rf

    def render_current(self, rf: RawFrame) -> np.ndarray:
        """Renderiza `rf` con el último estado (acción/social) ya conocido --
        para publicar cada frame en cuanto llega (ver vlm_hri/ros/node.py),
        en vez de esperar a que el VLM termine todo el trozo y volcar de
        golpe ~chunk_frames imágenes seguidas (se ve "cortado"/a saltos en un
        consumidor en vivo, aunque en un mp4 grabado no se note porque el
        archivo lleva su propio fps de reproducción)."""
        with self.state.lock:
            actions = dict(self.state.actions)
            social_states = dict(self.state.social_states)
            vlm_target = self.state.vlm_target
            vlm_target_conf = self.state.vlm_target_conf
            yolo_acum = self.state.yolo_total_s
            vlm_acum = self.state.vlm_total_s
            vlm_n = self.state.vlm_calls
            vlm_pending = self.state.vlm_pending
        return render_pose_frame(
            rf, actions, social_states,
            w=self.w, h=self.h, fps=self.fps, chunk_sec=self.chunk_sec,
            chunk_index=self.chunk_i, chunk_count=max(self.chunk_i, vlm_n),
            target_pid=_target_in_frame(rf, vlm_target),
            target_confidence=vlm_target_conf,
            yolo_total_s=yolo_acum, vlm_total_s=vlm_acum, vlm_calls=vlm_n,
            pipeline_total_s=time.perf_counter() - self.t_pipeline0, vlm_pending=vlm_pending,
            draw_pose=self.draw_pose,
        )

    def tick(self, elapsed_s: float = 0.0) -> None:
        """Contabilidad + chequeo de límite de trozo -- llamar una vez por
        frame, después de `append_frame()` (y de cualquier render de vista
        previa que quiera usar `chunk_raw[-1]` antes de que se dispare el
        envío del trozo, que vacía `chunk_raw`)."""
        with self.state.lock:
            self.state.yolo_total_s += elapsed_s
            self.state.frames_done = self.frame_i + 1
        self.frame_i += 1
        if self.frame_i >= self.next_chunk_at and len(self.chunk_raw) >= self.chunk_frames:
            self._submit_pending_chunk()

    def _submit_pending_chunk(self) -> None:
        if not self.chunk_raw:
            return
        all_pids = sorted({d["pid"] for rf in self.chunk_raw for d in rf.dets})
        # Prioriza a quien está más cerca de la cámara (mayor área de caja),
        # no a quien tiene el pid numérico más bajo.
        ranking_rows = [
            {"person_id": d["pid"], "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"]}
            for rf in self.chunk_raw
            for d in rf.dets
        ]
        pids = pose_gait.closest_n_pids(ranking_rows, all_pids, self.max_people)
        job = PoseVlmChunkJob(
            chunk_index=self.chunk_i,
            start_frame=self.frame_i - len(self.chunk_raw),
            raw_frames=list(self.chunk_raw),
            person_ids=pids,
            out_path=self.chunk_dir / f"chunk_{self.chunk_i:04d}.mp4",
            fps=self.fps,
            size=(self.w, self.h),
        )

        def _on_evict(old: PoseVlmChunkJob) -> None:
            if old.raw_frames:
                _flush_chunk_to_sinks(old, self)

        video_io.submit_chunk(self.vlm_queue, job, on_evict=_on_evict)
        self.chunk_i += 1
        self.next_chunk_at += self.chunk_frames
        self.chunk_raw = []

    def flush_and_stop(self) -> dict:
        """Cierra el trozo pendiente, para el hilo VLM, y devuelve una foto
        del estado final (mismas claves que antes se leían de `state` para
        armar `summary.json`) -- el llamador decide qué hacer con eso."""
        self._submit_pending_chunk()
        self.stop.set()
        if self._worker is not None:
            self._worker.join(timeout=300)
        with self.state.lock:
            return {
                "segment_actions": list(self.state.segment_actions),
                "segment_social": list(self.state.segment_social),
                "segment_confidence": list(self.state.segment_confidence),
                "segment_interrupt_cost": list(self.state.segment_interrupt_cost),
                "vlm_chunk_timings": list(self.state.vlm_chunk_timings),
                "final_actions": dict(self.state.actions),
                "final_social": dict(self.state.social_states),
                "final_confidence": dict(self.state.confidence),
                "final_interrupt_cost": dict(self.state.interrupt_cost),
                "segment_target": list(self.state.segment_target),
                "segment_target_filter": list(self.state.segment_target_filter),
                "vlm_total_s": self.state.vlm_total_s,
                "vlm_calls": self.state.vlm_calls,
                "yolo_total_s": self.state.yolo_total_s,
                "frame_i": self.frame_i,
                "chunk_i": self.chunk_i,
            }
