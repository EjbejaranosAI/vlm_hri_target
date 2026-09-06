"""Vista previa en vivo del stream: ventana OpenCV, mpv/ffplay por pipe, o MJPEG por HTTP."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np


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
