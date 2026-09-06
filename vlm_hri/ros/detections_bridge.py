"""Convierte vision_msgs/Detection2DArray (publicado por dynamic_tracking en
/detections/tracked) a las estructuras internas de vlm_hri: una lista de
`dets` (mismo formato que produce vlm_hri.detection.detect_people, más
world_x/world_y) y un dict paralelo pid -> (map_x, map_y).

dynamic_tracking ya hace detección+tracking (LiDAR + YOLO auxiliar) con ID
estable y posición real en el mundo -- este módulo solo lee ese contrato, no
vuelve a detectar ni trackear nada.

Contrato de mensaje (confirmado leyendo dynamic-tracking/dynamic_tracking/
track_pipeline.py:tracks_to_detection_array, no un tipo custom):
  - `det.bbox.center.position.{x,y}` + `size_x`/`size_y` -- bbox en imagen,
    formato centro+tamaño.
  - `det.results[0].hypothesis.class_id` -- "{class_name}#{track_id}",
    p. ej. "person#7".
  - `det.results[0].hypothesis.score` -- confianza.
  - `det.results[0].pose.pose.position.{x,y}` -- posición real (map u odom
    según el perfil sim/robot de dynamic_tracking; se reenvía tal cual, sin
    tocar TF)."""

from __future__ import annotations


def _parse_class_id(class_id: str) -> tuple[str, int | None]:
    """"person#7" -> ("person", 7). Si no hay "#" o el sufijo no es un
    entero, el track_id queda en None (detección sin ID válido, se descarta
    -- vlm_hri necesita un pid estable para el buffer de trozos)."""
    if "#" not in class_id:
        return class_id, None
    name, _, suffix = class_id.rpartition("#")
    try:
        return name, int(suffix)
    except ValueError:
        return name, None


def detection2d_array_to_dets(
    msg, *, target_class: str = "person"
) -> tuple[list[dict], dict[int, tuple[float, float]]]:
    """(dets, world_xy) a partir de un vision_msgs/Detection2DArray.

    `dets` trae el mismo shape que el resto de vlm_hri espera
    (x1,y1,x2,y2,pid,class_name,conf) más world_x/world_y -- listo para
    pasarle a vlm_hri.runners.pose_session.PoseStreamSession.append_frame()
    (opcionalmente después de pegarle keypoints de pose propios vía
    vlm_hri.pose.gait.attach_pose_keypoints, ver ros/node.py).

    Detecciones sin track_id parseable o de otra clase (`target_class`) se
    descartan -- sin pid estable no hay con qué mantener un track a través
    de los trozos de ~1s del VLM."""
    dets: list[dict] = []
    world_xy: dict[int, tuple[float, float]] = {}
    for det in msg.detections:
        if not det.results:
            continue
        hyp = det.results[0]
        class_name, pid = _parse_class_id(hyp.hypothesis.class_id)
        if pid is None or class_name.lower() != target_class:
            continue
        cx = float(det.bbox.center.position.x)
        cy = float(det.bbox.center.position.y)
        hw = float(det.bbox.size_x) / 2.0
        hh = float(det.bbox.size_y) / 2.0
        dets.append(
            {
                "x1": int(round(cx - hw)),
                "y1": int(round(cy - hh)),
                "x2": int(round(cx + hw)),
                "y2": int(round(cy + hh)),
                "pid": pid,
                "class_name": class_name,
                "conf": float(hyp.hypothesis.score),
                "world_x": float(hyp.pose.pose.position.x),
                "world_y": float(hyp.pose.pose.position.y),
            }
        )
        world_xy[pid] = (float(hyp.pose.pose.position.x), float(hyp.pose.pose.position.y))
    return dets, world_xy
