#!/usr/bin/env python3
"""Punto de entrada ALTERNO: pipeline con detector de pose (yolo26n-pose.pt)
+ marcha por piernas como pista de color para el VLM, en vez de solo
detección de personas. No toca main.py / run_video.py / pipeline.py.

  python main_pose.py videos [--input-dir input_videos] [--output DIR]
  python main_pose.py video --input video.mp4 [--output DIR]
  python main_pose.py stream -i video.mp4 [--preview] [--realtime]
  python main_pose.py stream --camera 0 [--preview] [--display]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_vid = sub.add_parser("video", help="Un archivo de video (offline)")
    p_vid.add_argument("--input", "-i", type=Path, required=True)
    p_vid.add_argument("--output", type=Path, default=None)

    p_all = sub.add_parser("videos", help="Todos los .mp4 de una carpeta (offline)")
    p_all.add_argument("--input-dir", type=Path, default=ROOT / "input_videos")
    p_all.add_argument("--output", type=Path, default=None)

    p_stream = sub.add_parser(
        "stream",
        help="Cámara o video en vivo (tiempo real): YOLO+pose cada frame, VLM en hilo aparte",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python main_pose.py stream -i input_videos/walking.mp4 --preview --realtime\n"
            "  python main_pose.py stream --camera 0 --display\n"
        ),
    )
    src = p_stream.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", type=Path, help="Un video .mp4 (simula cámara)")
    src.add_argument("--camera", type=int, help="Índice de cámara USB (0 = default)")
    p_stream.add_argument("--output", type=Path, default=None)
    p_stream.add_argument("--display", action="store_true", help="Ventana OpenCV (q=salir)")
    p_stream.add_argument("--preview", action="store_true", help="Vista previa mpv/ffplay o navegador")
    p_stream.add_argument("--realtime", action="store_true", help="Con --input: al fps del video")
    p_stream.add_argument("--max-frames", type=int, default=None, help="Limitar frames (pruebas)")

    args = parser.parse_args()

    if args.mode == "video":
        from run_video_pose import run

        run(video_path=args.input, output_dir=args.output)
    elif args.mode == "videos":
        from run_video_pose import run_all

        run_all(input_dir=args.input_dir, output_dir=args.output)
    else:
        from run_stream_pose import run as run_stream

        run_stream(
            video_path=args.input,
            camera=args.camera,
            output_dir=args.output,
            display=args.display,
            preview=args.preview,
            realtime=args.realtime,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
