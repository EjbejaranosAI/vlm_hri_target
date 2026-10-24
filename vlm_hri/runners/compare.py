"""Comparación VLM generativo vs JEV-HRI (VLM_BACKEND=vlm | vlm_jev) sobre
los mismos videos, en el modo pose (el mismo camino que el nodo ROS2).

Cada backend corre en un subproceso propio (con su VLM_BACKEND, el modelo se
carga una sola vez para todos los videos y nunca hay dos en VRAM a la vez).
Por defecto a ritmo real (--realtime): así se ve lo que pasaría en el robot
-- cuántos trozos llega a inferir cada backend, cuántos se descartan y con
cuánto retraso (lag) llega cada etiqueta.

    python -m vlm_hri.runners.compare                      # todos los de input_videos/
    python -m vlm_hri.runners.compare -i input_videos/esperando.mp4 --overlay bayes

Salida (output/compare_backends/ por defecto):
    vlm/<video>/annotated_stream.mp4 + summary.json
    vlm_jev/<video>/annotated_stream.mp4 + summary.json
    side_by_side/<video>.mp4        izquierda vlm, derecha vlm_jev
    comparison.json / comparison.md tiempos, targets por modo y acuerdo
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from ..config import ROOT, VIDEO_EXTS
from ..target_filter import TARGET_FILTER_MODES

BACKENDS = ("vlm", "vlm_jev")
_DEFAULT_OUT = ROOT / "output" / "compare_backends"


def _videos(paths: list[Path] | None) -> list[Path]:
    if paths:
        return paths
    found = sorted(p for p in (ROOT / "input_videos").iterdir() if p.suffix.lower() in VIDEO_EXTS)
    # Algunos videos están dos veces (nombre corto + nombre largo del banco
    # de imágenes, mismo contenido): se deduplica por tamaño en disco.
    seen, out = set(), []
    for p in sorted(found, key=lambda p: (len(p.name), p.name)):
        if p.stat().st_size not in seen:
            seen.add(p.stat().st_size)
            out.append(p)
    return sorted(out)


def _worker(backend: str, videos: list[Path], out: Path, realtime: bool) -> None:
    """Dentro del subproceso (VLM_BACKEND ya fijado en el entorno)."""
    from ..detection import resolve_device
    from ..vlm.model import load_backend
    from . import stream

    device = resolve_device()
    vlm, processor = load_backend(device)
    for v in videos:
        print(f"[{backend}] {v.name}", flush=True)
        stream.run(video_path=v, output_dir=out, realtime=realtime, vlm=vlm, processor=processor, device=device)


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _backend_stats(summary: dict) -> dict:
    timings = summary.get("vlm_chunk_timings", [])
    lat = [t["latency_s"] for t in timings]
    lag = [t["lag_s"] for t in timings if "lag_s" in t]
    tf = summary.get("target_filter", {})
    return {
        "chunks_submitted": summary.get("vlm_chunks_submitted"),
        "inferences": len(lat),
        "dropped": max(0, (summary.get("vlm_chunks_submitted") or 0) - len(lat)),
        "latency_avg_s": round(statistics.mean(lat), 4) if lat else None,
        "latency_p50_s": round(_pct(lat, 0.5), 4) if lat else None,
        "latency_p95_s": round(_pct(lat, 0.95), 4) if lat else None,
        "lag_avg_s": round(statistics.mean(lag), 3) if lag else None,
        "lag_max_s": round(max(lag), 3) if lag else None,
        "inferences_per_s": round(len(lat) / summary["video_duration_s"], 3) if summary.get("video_duration_s") else None,
        "realtime_factor": summary.get("realtime_factor"),
        "target_metrics": tf.get("metrics", {}),
    }


def _agreement(a: dict, b: dict) -> dict:
    """Acuerdo entre backends por trozo común (mismo chunk_index): estado
    social por persona vista en ambos y target de cada modo."""
    ca = {c["chunk_index"]: c for c in a.get("target_filter", {}).get("chunks", [])}
    cb = {c["chunk_index"]: c for c in b.get("target_filter", {}).get("chunks", [])}
    common = sorted(set(ca) & set(cb))
    same_soc = total_soc = 0
    target_same = {m: 0 for m in TARGET_FILTER_MODES}
    for i in common:
        sa, sb = ca[i]["social"], cb[i]["social"]
        for pid in set(sa) & set(sb):
            total_soc += 1
            same_soc += sa[pid] == sb[pid]
        for m in TARGET_FILTER_MODES:
            target_same[m] += ca[i].get(m, {}).get("pid") == cb[i].get(m, {}).get("pid")
    n = len(common)
    return {
        "common_chunks": n,
        "social_agreement": round(same_soc / total_soc, 3) if total_soc else None,
        "target_agreement": {m: round(v / n, 3) if n else None for m, v in target_same.items()},
    }


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((34, frame.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


def _side_by_side(videos: list[Path], out: Path, titles: list[str]) -> bool:
    """Videos en columnas (izquierda a derecha), cada uno con su título."""
    from .. import video_io

    caps = [cv2.VideoCapture(str(v)) for v in videos]
    if not all(c.isOpened() for c in caps):
        for c in caps:
            c.release()
        return False
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    try:
        while True:
            reads = [c.read() for c in caps]
            if not all(ok for ok, _ in reads):
                break
            first = reads[0][1]
            cols = [
                _label(f if f.shape[:2] == first.shape[:2] else cv2.resize(f, (first.shape[1], first.shape[0])), t)
                for (_, f), t in zip(reads, titles)
            ]
            frame = np.hstack(cols)
            if writer is None:
                out.parent.mkdir(parents=True, exist_ok=True)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w - w % 2, h - h % 2))
            writer.write(frame[: frame.shape[0] - frame.shape[0] % 2, : frame.shape[1] - frame.shape[1] % 2])
    finally:
        for c in caps:
            c.release()
        if writer is not None:
            writer.release()
    if writer is None:
        return False
    video_io.finalize_video_h264(out)
    return True


def _title(name: str, stats: dict, overlay: str) -> str:
    return (
        f"{name}  |  latencia media {_fmt(stats['latency_avg_s'])}s  |  lag medio {_fmt(stats['lag_avg_s'])}s  |  "
        f"{stats['inferences']}/{stats['chunks_submitted']} trozos  |  target={overlay}"
    )


def _fmt(v, nd=2) -> str:
    if v is None:
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _report(rows: list[dict], out: Path, overlay: str, realtime: bool) -> None:
    (out / "comparison.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# VLM generativo vs JEV-HRI",
        "",
        f"Modo pose (mismo camino que el nodo ROS2), {'a ritmo real (--realtime)' if realtime else 'a máxima velocidad'}; "
        f"overlay de los videos: target `{overlay}`. Lag = cierre del trozo -> resultado del VLM "
        "(incluye la espera en cola).",
        "",
        "## Tiempos",
        "",
        "| video | backend | trozos | inferidos | descartados | latencia media (s) | p95 (s) | lag medio (s) | lag máx (s) | inferencias/s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for b in BACKENDS:
            s = r.get(b)
            if not s:
                continue
            lines.append(
                f"| {r['video']} | {b} | {s['chunks_submitted']} | {s['inferences']} | {s['dropped']} | "
                f"{_fmt(s['latency_avg_s'])} | {_fmt(s['latency_p95_s'])} | {_fmt(s['lag_avg_s'])} | "
                f"{_fmt(s['lag_max_s'])} | {_fmt(s['inferences_per_s'])} |"
            )
    lines += ["", "## Target por modo (coverage / cambios / pid_swaps)", ""]
    lines.append("| video | backend | " + " | ".join(TARGET_FILTER_MODES) + " |")
    lines.append("|---|---|" + "---|" * len(TARGET_FILTER_MODES))
    for r in rows:
        for b in BACKENDS:
            s = r.get(b)
            if not s:
                continue
            cells = []
            for m in TARGET_FILTER_MODES:
                tm = s["target_metrics"].get(m)
                cells.append(f"{tm['coverage']:.2f} / {tm['switches']} / {tm['pid_swaps']}" if tm else "–")
            lines.append(f"| {r['video']} | {b} | " + " | ".join(cells) + " |")
    lines += ["", "## Acuerdo entre backends (trozos comunes)", ""]
    lines.append("| video | trozos comunes | estado social | " + " | ".join(f"target {m}" for m in TARGET_FILTER_MODES) + " |")
    lines.append("|---|---|---|" + "---|" * len(TARGET_FILTER_MODES))
    for r in rows:
        a = r.get("agreement")
        if not a:
            continue
        lines.append(
            f"| {r['video']} | {a['common_chunks']} | {_fmt(a['social_agreement'])} | "
            + " | ".join(_fmt(a["target_agreement"][m]) for m in TARGET_FILTER_MODES) + " |"
        )
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Compara VLM_BACKEND=vlm vs vlm_jev sobre los mismos videos")
    p.add_argument("-i", "--input", type=Path, nargs="*", help="Videos (default: input_videos/ deduplicado)")
    p.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--no-realtime", action="store_true", help="Máxima velocidad en vez de ritmo real")
    p.add_argument("--overlay", choices=TARGET_FILTER_MODES, default="bayes", help="Target que se dibuja en los videos")
    p.add_argument("--backends", nargs="*", choices=BACKENDS, default=list(BACKENDS))
    p.add_argument("--report-only", action="store_true", help="Solo rehacer side-by-side + informe")
    p.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    args = p.parse_args()
    videos = _videos(args.input)
    realtime = not args.no_realtime

    if args.worker:
        _worker(args.worker, videos, args.output / args.worker, realtime)
        return

    if not args.report_only:
        for b in args.backends:
            env = {**os.environ, "VLM_BACKEND": b, "VLM_TARGET_FILTER": args.overlay}
            cmd = [sys.executable, "-m", "vlm_hri.runners.compare", "--worker", b, "--output", str(args.output),
                   "--overlay", args.overlay, "-i", *map(str, videos)]
            if not realtime:
                cmd.append("--no-realtime")
            print(f"=== {b}: {len(videos)} videos ===", flush=True)
            subprocess.run(cmd, env=env, cwd=ROOT, check=True)

    rows = []
    for v in videos:
        name = v.stem.replace("_h264", "")
        row: dict = {"video": name}
        summaries = {}
        for b in BACKENDS:
            f = args.output / b / name / "summary.json"
            if f.is_file():
                summaries[b] = json.loads(f.read_text(encoding="utf-8"))
                row[b] = _backend_stats(summaries[b])
        if len(summaries) == 2:
            row["agreement"] = _agreement(summaries["vlm"], summaries["vlm_jev"])
            _side_by_side(
                [args.output / b / name / "annotated_stream.mp4" for b in BACKENDS],
                args.output / "side_by_side" / f"{name}.mp4",
                [_title(b, row[b], args.overlay) for b in BACKENDS],
            )
        rows.append(row)
    _report(rows, args.output, args.overlay, realtime)
    print((args.output / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
