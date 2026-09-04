"""
Pipeline en streaming: YOLO cada frame + VLM cada N segundos en hilo aparte.

  python main.py stream -i input_videos/walking.mp4
  python main.py stream -i video.mp4 --display --realtime
  python main.py stream --camera 0 --display
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Empty, Queue
from socketserver import ThreadingMixIn

import cv2
import numpy as np
from ultralytics import YOLO

import pipeline as P
from run_video import (
    VIDEO_OUTPUT_DIR,
    _crop_frame,
    _even_dims,
    _realtime_metrics,
    ensure_opencv_video,
    finalize_video_h264,
    print_vlm_latency_report,
    vlm_timing_stats,
)

ROOT = P.ROOT

STREAM_OUTPUT_DIR = Path(os.environ.get("STREAM_OUTPUT_DIR", str(ROOT / "output_stream")))


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
    live_preview: LivePreview | None
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
        P.draw_box(
            ann,
            d["x1"],
            d["y1"],
            d["x2"],
            d["y2"],
            _label_for_det(d, actions, social_states),
        )
    P.draw_banner_video(
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
            (int(10 * P.ui_scale(h, w)), int(h - 30 * P.ui_scale(h, w))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * P.ui_scale(h, w),
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
            if P.chunk_vlm_use_image(chunk_sec):
                mid = job.frames[len(job.frames) // 2]
                vlm_out = P.infer_actions_image(
                    vlm, processor, mid, job.dets_mid, already_annotated=True
                )
            elif _write_chunk_video(job.frames, job.out_path, job.fps, job.size):
                vlm_out = P.infer_actions_video(
                    vlm,
                    processor,
                    job.out_path,
                    job.person_ids,
                    dets=job.dets_mid,
                    chunk_sec=chunk_sec,
                )
            else:
                mid = job.frames[len(job.frames) // 2]
                vlm_out = P.infer_actions_image(
                    vlm, processor, mid, job.dets_mid, already_annotated=True
                )
        except Exception as e:
            print(f"  [VLM trozo {job.chunk_index}] error: {e}", flush=True)
            vlm_out = P.VlmInferenceResult({}, {}, 0.0)
        acts = dict(vlm_out.actions)
        social = dict(vlm_out.social_states)
        if job.buffer_dets and job.person_ids:
            acts, social = P.refine_chunk_labels(
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
                if st and st != P.SOCIAL_UNKNOWN:
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
            f"mode={P.social_state_mode()})",
            flush=True,
        )


def _opencv_gui_available() -> bool:
    """OpenCV headless (sin GTK) no soporta imshow."""
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        return False
    try:
        cv2.namedWindow("__gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__gui_probe__")
        return True
    except cv2.error:
        return False
    except Exception:
        return False


def _safe_destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def _preview_player_available() -> bool:
    return bool(shutil.which("mpv") or shutil.which("ffplay"))


def _graphical_session_ok() -> bool:
    if os.name == "nt":
        return True
    return bool(
        os.environ.get("DISPLAY", "").strip()
        or os.environ.get("WAYLAND_DISPLAY", "").strip()
    )


def _player_env(display: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if display:
        env["DISPLAY"] = display
    if env.get("WAYLAND_DISPLAY"):
        env.pop("SDL_VIDEODRIVER", None)
    elif os.name != "nt":
        env.setdefault("SDL_VIDEODRIVER", "x11")
    return env


_display_probe_done = False


def _display_candidates() -> list[str]:
    cur = os.environ.get("DISPLAY", "").strip()
    out: list[str] = []
    if cur:
        out.append(cur)
    for d in (":0", ":1"):
        if d not in out:
            out.append(d)
    return out


def _mpv_rawvideo_probe(env: dict[str, str], w: int = 64, h: int = 48) -> tuple[bool, str]:
    if not shutil.which("mpv"):
        return False, "mpv no instalado"
    cmd = [
        "mpv",
        "--no-config",
        "--no-terminal",
        "--really-quiet",
        "--no-audio",
        f"--demuxer-rawvideo-w={w}",
        f"--demuxer-rawvideo-h={h}",
        "--demuxer-rawvideo-fps=10",
        "--demuxer-rawvideo-mp-format=bgr24",
        "--demuxer=rawvideo",
        "-",
    ]
    if env.get("WAYLAND_DISPLAY"):
        cmd.insert(1, "--gpu-context=wayland")
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            env=env,
        )
        time.sleep(0.25)
        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode(errors="replace").strip()
            return False, err or "mpv salió al iniciar"
        assert proc.stdin is not None
        proc.stdin.write(np.zeros((h, w, 3), dtype=np.uint8).tobytes())
        proc.stdin.flush()
        time.sleep(0.1)
        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode(errors="replace").strip()
            return False, err or "mpv cerró al recibir frames"
        proc.stdin.close()
        proc.wait(timeout=2)
        return True, ""
    except Exception as e:
        return False, str(e)


def _resolve_display(*, quiet: bool = False) -> bool:
    """Elige DISPLAY válido (:0 / :1) para mpv/ffplay."""
    global _display_probe_done
    if _display_probe_done:
        return _graphical_session_ok()
    _display_probe_done = True

    if os.environ.get("WAYLAND_DISPLAY"):
        ok, err = _mpv_rawvideo_probe(_player_env())
        if ok:
            if not quiet:
                print(
                    f"Wayland ({os.environ['WAYLAND_DISPLAY']}); vista previa con mpv.",
                    flush=True,
                )
            return True
        if not quiet and err:
            print(f"Aviso Wayland/mpv: {err}", flush=True)

    for disp in _display_candidates():
        ok, err = _mpv_rawvideo_probe(_player_env(disp))
        if ok:
            os.environ["DISPLAY"] = disp
            if not quiet:
                print(f"DISPLAY={disp} (vista previa)", flush=True)
            return True
        if not quiet and err:
            print(f"  DISPLAY={disp} no usable: {err[:120]}", flush=True)

    return _graphical_session_ok()


class LivePreview:
    """Reproductor en vivo por stdin (raw BGR)."""

    name = "player"

    def __init__(self, w: int, h: int, fps: float, env: dict[str, str] | None = None) -> None:
        self._last_err = ""
        env = env or _player_env()
        self._proc = self._spawn(w, h, fps, env)
        time.sleep(0.3)
        try:
            self.write(np.zeros((h, w, 3), dtype=np.uint8))
        except BrokenPipeError:
            self._last_err = self._read_stderr()
            raise RuntimeError(self._last_err or "Broken pipe al enviar el primer frame")
        time.sleep(0.1)
        if not self.alive():
            self._last_err = self._read_stderr()
            raise RuntimeError(
                self._last_err or "el proceso terminó al iniciar (¿DISPLAY o Wayland?)"
            )

    def _spawn(self, w: int, h: int, fps: float, env: dict[str, str]) -> subprocess.Popen:
        raise NotImplementedError

    def _read_stderr(self) -> str:
        if getattr(self, "_proc", None) is None or self._proc.stderr is None:
            return self._last_err
        try:
            return (self._proc.stderr.read() or b"").decode(errors="replace").strip()
        except Exception:
            return self._last_err

    def alive(self) -> bool:
        return self._proc.poll() is None

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._proc.stdin is not None and self.alive():
            self._proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
            self._proc.stdin.flush()

    def close(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class MpvPreview(LivePreview):
    name = "mpv"

    def _spawn(self, w: int, h: int, fps: float, env: dict[str, str]) -> subprocess.Popen:
        cmd = [
            "mpv",
            "--no-config",
            "--no-terminal",
            "--really-quiet",
            "--no-audio",
            "--title=Action recognition",
            f"--demuxer-rawvideo-w={w}",
            f"--demuxer-rawvideo-h={h}",
            f"--demuxer-rawvideo-fps={max(fps, 1.0)}",
            "--demuxer-rawvideo-mp-format=bgr24",
            "--demuxer=rawvideo",
            "-",
        ]
        if env.get("WAYLAND_DISPLAY"):
            cmd.insert(1, "--gpu-context=wayland")
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            env=env,
        )


class FfplayPreview(LivePreview):
    name = "ffplay"

    def _spawn(self, w: int, h: int, fps: float, env: dict[str, str]) -> subprocess.Popen:
        return subprocess.Popen(
            [
                "ffplay",
                "-loglevel",
                "error",
                "-window_title",
                "Action recognition",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-framedrop",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(max(fps, 1.0)),
                "-i",
                "pipe:0",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            env=env,
        )


class _ThreadingHttpServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class HttpPreview:
    """Vista previa MJPEG en el navegador (útil sin X11 o por SSH)."""

    name = "http"

    def __init__(self, w: int, h: int, fps: float, port: int | None = None) -> None:
        self._port = int(port or os.environ.get("STREAM_PREVIEW_PORT", "8765"))
        self._fps = max(fps, 1.0)
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stop = threading.Event()
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                pass

            def do_GET(self) -> None:
                if self.path not in ("/", "/video"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                interval = 1.0 / parent._fps
                while not parent._stop.is_set():
                    with parent._lock:
                        data = parent._jpeg
                    if data:
                        try:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    time.sleep(interval)

        for attempt in range(5):
            p = self._port + attempt
            try:
                self._server = _ThreadingHttpServer(("127.0.0.1", p), Handler)
                self._port = p
                break
            except OSError:
                if attempt == 4:
                    raise
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        self.write(blank)

    def alive(self) -> bool:
        return not self._stop.is_set()

    def write(self, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()

    def close(self) -> None:
        self._stop.set()
        self._server.shutdown()
        self._thread.join(timeout=2)


_preview_warned = False


def _open_live_preview(
    w: int, h: int, fps: float, *, allow_http: bool = True
) -> LivePreview | HttpPreview | None:
    global _preview_warned

    http_only = os.environ.get("STREAM_PREVIEW_HTTP_ONLY", "0") == "1"
    use_http = allow_http and os.environ.get("STREAM_PREVIEW_HTTP", "1") != "0"
    if not _graphical_session_ok() and not use_http:
        if not _preview_warned:
            _preview_warned = True
            print(
                "No hay DISPLAY ni WAYLAND. En escritorio: export DISPLAY=:0\n"
                "  O usa vista previa web (activada por defecto si falla mpv).",
                flush=True,
            )
        return None

    if _graphical_session_ok() and not http_only:
        _resolve_display(quiet=_preview_warned)
        env = _player_env()
        errors: list[str] = []
        for cls in (MpvPreview, FfplayPreview):
            if not shutil.which(cls.name):
                continue
            try:
                player = cls(w, h, fps, env=env)
                print(f"Ventana de vista previa: {cls.name}", flush=True)
                return player
            except Exception as e:
                detail = getattr(e, "args", (str(e),))[0]
                errors.append(f"{cls.name}: {detail}")

        if use_http:
            if not _preview_warned:
                print("mpv/ffplay no abrieron ventana; usando vista previa en el navegador.", flush=True)
                for msg in errors:
                    print(f"  - {msg}", flush=True)
        elif not _preview_warned:
            _preview_warned = True
            print("No se pudo abrir vista previa en vivo.", flush=True)
            for msg in errors:
                print(f"  - {msg}", flush=True)
            print(
                "  Prueba: export DISPLAY=:0  |  sudo apt install mpv ffmpeg\n"
                "  O: STREAM_PREVIEW_HTTP=1 (navegador http://127.0.0.1:8765/)",
                flush=True,
            )
            return None

    if use_http:
        try:
            player = HttpPreview(w, h, fps)
            print(
                f"Vista previa web: http://127.0.0.1:{player._port}/  "
                "(SSH: ssh -L 8765:127.0.0.1:8765 …)",
                flush=True,
            )
            _preview_warned = True
            return player
        except Exception as e:
            if not _preview_warned:
                _preview_warned = True
                print(f"No se pudo iniciar servidor web de preview: {e}", flush=True)
    return None


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
            soc = P.social_states_from_actions(acts)
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
    return P.format_detection_label(
        pid,
        actions.get(pid, ""),
        social_state=social_states.get(pid),
        class_name=d.get("class_name", "person"),
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
    w, h = _even_dims(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    chunk_frames = max(1, int(chunk_sec * fps))

    if display and not _opencv_gui_available():
        if _preview_player_available():
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
    elif preview and not _preview_player_available():
        if os.environ.get("STREAM_PREVIEW_HTTP", "1") == "0":
            print("Aviso: instala mpv o ffmpeg (ffplay). Sin vista previa.", flush=True)
            preview = False

    if preview:
        _resolve_display(quiet=False)

    print(f"Stream: {out_name} | fuente={source} | trozo={chunk_sec}s ({chunk_frames} fr)", flush=True)
    if P.chunk_vlm_use_image(chunk_sec):
        vlm_chunk_mode = "imagen (1 frame/trozo)"
    else:
        vfps = P.video_vlm_fps(chunk_sec)
        vlm_chunk_mode = f"video ~{vfps:.0f} fps/trozo (1 s de movimiento)"
    print(
        f"VLM={P.VLM_ID} | perfil={os.environ.get('VLM_PROFILE', 'balanced')} | "
        f"prompt={P.vlm_prompt_mode()} | trozo→{vlm_chunk_mode} | "
        f"social={P.social_state_mode()} | salida: {out}",
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
        yolo = YOLO(P.YOLO_WEIGHTS)
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
    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    live_preview: LivePreview | None = None
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

            frame = _crop_frame(frame, w, h)
            raw = P.detect_people(yolo, frame)
            if prev_dets:
                dets = P.track_detections(prev_dets, raw)
            else:
                dets = P.assign_spatial_ids(raw)
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
                    live_preview = _open_live_preview(w, h, fps)
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

            buffer_ann.append(P.annotate_for_vlm(frame, dets))
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
                pids = pids[: P.MAX_TRACK_IDS]
                dets_prompt = P.dets_for_vlm_prompt(buffer_dets, pids)
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
            pids = sorted({d["pid"] for row in buffer_dets for d in row})[: P.MAX_TRACK_IDS]
            dets_prompt = P.dets_for_vlm_prompt(buffer_dets, pids)
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
        "social_state_mode": P.social_state_mode(),
        "actions_final": {str(k): v for k, v in final_actions.items()},
        "social_states_final": {str(k): v for k, v in final_social.items()},
        "vlm_chunk_timings": vlm_chunk_timings,
        "vlm_chunk_latencies_s": [t["latency_s"] for t in vlm_chunk_timings],
        **vlm_timing_stats([t["latency_s"] for t in vlm_chunk_timings]),
        "segments": _build_segment_summary(
            segment_actions, segment_social, vlm_chunk_timings
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    vname = video_path.name if video_path else (f"camera_{camera}" if camera is not None else out_name)
    print_vlm_latency_report(vname, vlm_chunk_timings)
    print(
        f"\nStream listo → {out} ({pipeline_total_s:.1f}s, {frame_i} frames, "
        f"VLM {vlm_calls} trozos, factor RT {rt['realtime_factor']:.2f}x)",
        flush=True,
    )
    print(f"  Ver resultado: {out_mp4}", flush=True)
    return out


def run_all(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    preview: bool = False,
    realtime: bool = False,
) -> list[Path]:
    """Todos los videos de input_videos/ con pipeline stream + resumen batch."""
    from run_video import list_videos, write_batch_summary

    indir = Path(input_dir or ROOT / "input_videos")
    base_out = Path(output_dir or STREAM_OUTPUT_DIR)
    paths = list_videos(indir)
    if not paths:
        raise SystemExit(f"No hay videos en {indir}")

    print(f"Stream batch: {len(paths)} videos en {indir}\n", flush=True)
    device = P.resolve_device()
    yolo = YOLO(P.YOLO_WEIGHTS)
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

    write_batch_summary(rows, base_out)
    (base_out / "batch_summary_stream.json").write_text(
        (base_out / "batch_summary.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return outs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="YOLO+VLM en streaming")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", type=Path, help="Video archivo (simula cámara)")
    src.add_argument("--camera", type=int, help="Índice cámara (0 = default)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--display", action="store_true", help="Ventana OpenCV")
    p.add_argument(
        "--preview",
        action="store_true",
        help="Vista previa en vivo con ffplay",
    )
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Con video: esperar 1/fps entre frames (ritmo real)",
    )
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()
    run(
        video_path=args.input,
        camera=args.camera,
        output_dir=args.output,
        display=args.display,
        preview=args.preview,
        realtime=args.realtime,
        max_frames=args.max_frames,
    )
