"""OutputSink de ROS2: publica lo que hoy el CLI solo escribe a disco
(vlm_hri.runners.pose_session.VideoFileSink) como tópicos ROS2 en su lugar.

  /vlm_hri/image_annotated  sensor_msgs/Image        -- el mismo frame anotado
  /vlm_hri/social_states    std_msgs/String (JSON)   -- lista de personas
  /vlm_hri/target_pose      geometry_msgs/PoseStamped -- target elegido
  /vlm_hri/target_filter    std_msgs/String (JSON)   -- votos + decisión de
                            los 3 modos del filtro de target, una vez por
                            trozo (para grabar en rosbag y comparar)

Formato de /vlm_hri/social_states (decidido con el usuario: JSON plano, sin
paquete de interfaces propio para esta v1):
  [{"id": 7, "pose": "sitting", "action": "talking", "social_state": "ENGAGED",
    "confidence": 2, "interrupt_cost": 1,
    "map_x": 1.2, "map_y": 3.4, "cluster_id": 0, "is_target": false,
    "target_confidence": null}, ...]

`pose` ("sitting"/"standing"/"walking"/"lying"/"unknown") viene 100% de
geometría de pose (ángulo real de rodilla + marcha confirmada, ver
vlm_hri.pose.gait.chunk_posture_by_pid/refine_labels_pose) -- el VLM ya NO
la juzga en absoluto. `action` es solo la actividad específica que sí
describe el VLM (p. ej. "using phone", "reading", "talking"), sin la postura
mezclada -- separados a propósito para que cada consumidor de este tópico
pueda usarlos de forma independiente en vez de tener que parsear un texto
combinado tipo "sitting and reading".

`is_target`/`target_confidence` (0-1): el target lo elige el propio VLM
(campo "target"/"target_conf" del prompt); `target_confidence` solo viene en
la entrada del target, `null` en el resto.

`confidence`/`interrupt_cost` (0-2, juicio propio del VLM -- ver
vlm_hri/vlm/prompt_templates/*.json) vienen `null` si el VLM no los dio ese
chunk (no confundir con un 0 real)."""

from __future__ import annotations

import json

import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String

from ..motion import strip_posture_prefix
from ..pose.gait import draw_pose_skeleton


class RosPublisherSink:
    """Implementa el protocolo OutputSink (ver pose_session.py) publicando
    en vez de escribir a un archivo. `frame_id` debe ser el mismo frame en el
    que vienen las posiciones world_x/world_y del bridge de detecciones (el
    `tracking_frame` de dynamic_tracking: `map` en sim, `odom` en robot)."""

    def __init__(self, node, *, topic_prefix: str = "/vlm_hri", frame_id: str = "map") -> None:
        self._node = node
        self._frame_id = frame_id
        self._bridge = CvBridge()
        self._image_pub = node.create_publisher(Image, f"{topic_prefix}/image_annotated", 10)
        self._social_pub = node.create_publisher(String, f"{topic_prefix}/social_states", 10)
        self._target_pub = node.create_publisher(PoseStamped, f"{topic_prefix}/target_pose", 10)
        # Topic aparte (no mezclado con image_annotated) para poder
        # activarlo/desactivarlo desde RViz con el checkbox del Display, sin
        # afectar la imagen anotada principal -- ver dynamic_tracking.rviz.
        self._pose_pub = node.create_publisher(Image, f"{topic_prefix}/pose_skeleton", 10)
        self._target_filter_pub = node.create_publisher(String, f"{topic_prefix}/target_filter", 10)

    def write_target_debug(self, record: dict) -> None:
        """Una vez por trozo: `target_filter.chunk_record` (target crudo del
        VLM + lo que decidió cada modo raw/ema/hyst)."""
        self._target_filter_pub.publish(String(data=json.dumps(record)))

    def publish_frame(self, ann: np.ndarray) -> None:
        """Publica un frame ya renderizado -- llamado directamente por el
        nodo una vez por cada frame recibido (ver PoseStreamSession.
        render_current), a la velocidad real de la cámara. No pasa por
        write_frame/_flush_chunk_to_sinks: eso solo se dispara una vez por
        trozo (~1s) cuando el VLM termina, lo que publicaría de golpe todo
        el trozo en ráfaga en vez de en vivo."""
        msg = self._bridge.cv2_to_imgmsg(ann, encoding="bgr8")
        msg.header = Header(frame_id=self._frame_id)
        msg.header.stamp = self._node.get_clock().now().to_msg()
        self._image_pub.publish(msg)

    def write_frame(self, ann: np.ndarray) -> None:
        pass  # ver publish_frame -- el nodo publica en vivo, no en ráfaga por trozo

    def publish_pose_skeleton(self, frame_bgr: np.ndarray, dets: list[dict]) -> None:
        """Esqueleto COCO-17 sobre el frame crudo, en su propio topic --
        checkbox de RViz = suscriptor. Sin nadie mirando (checkbox apagado)
        no se dibuja nada, para no gastar CPU/copias de frame de gratis."""
        if self._pose_pub.get_subscription_count() == 0:
            return
        out = frame_bgr.copy()
        for d in dets:
            draw_pose_skeleton(out, d.get("kpts"))
        msg = self._bridge.cv2_to_imgmsg(out, encoding="bgr8")
        msg.header = Header(frame_id=self._frame_id)
        msg.header.stamp = self._node.get_clock().now().to_msg()
        self._pose_pub.publish(msg)

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
    ) -> None:
        world_xy = world_xy or {}
        clusters = clusters or {}
        confidence = confidence or {}
        interrupt_cost = interrupt_cost or {}
        pose = pose or {}
        payload = [
            {
                "id": pid,
                "pose": pose.get(pid, "unknown"),
                "action": strip_posture_prefix(actions.get(pid, "unknown")),
                "social_state": social_states.get(pid, "UNKNOWN"),
                "confidence": confidence.get(pid),
                "interrupt_cost": interrupt_cost.get(pid),
                "map_x": world_xy.get(pid, (None, None))[0],
                "map_y": world_xy.get(pid, (None, None))[1],
                "cluster_id": clusters.get(pid),
                "is_target": pid == target_pid,
                "target_confidence": target_confidence if pid == target_pid else None,
            }
            for pid in social_states
        ]
        self._social_pub.publish(String(data=json.dumps(payload)))

        if target_pid is not None and target_pid in world_xy:
            tx, ty = world_xy[target_pid]
            pose = PoseStamped()
            pose.header = Header(frame_id=self._frame_id)
            pose.header.stamp = self._node.get_clock().now().to_msg()
            pose.pose.position.x = float(tx)
            pose.pose.position.y = float(ty)
            pose.pose.orientation.w = 1.0
            self._target_pub.publish(pose)
