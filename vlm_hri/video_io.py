"""IO de video compartido entre los runners: apertura/conversión, encode
H.264, escritura por ffmpeg-pipe, métricas de tiempo real y resumen batch.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2

from .config import VIDEO_EXTS


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


def list_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
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
