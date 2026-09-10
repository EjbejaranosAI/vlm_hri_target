"""OutputSink de ROS2: publica lo que hoy el CLI solo escribe a disco
(vlm_hri.runners.pose_session.VideoFileSink) como tópicos ROS2 en su lugar.

  /vlm_hri/image_annotated  sensor_msgs/Image        -- el mismo frame anotado
  /vlm_hri/social_states    std_msgs/String (JSON)   -- lista de personas
  /vlm_hri/target_pose      geometry_msgs/PoseStamped -- target elegido

Formato de /vlm_hri/social_states (decidido con el usuario: JSON plano, sin
paquete de interfaces propio para esta v1):
  [{"id": 7, "action": "talking", "social_state": "ENGAGED",
    "map_x": 1.2, "map_y": 3.4, "cluster_id": 0, "is_target": false}, ...]
"""

from __future__ import annotations

import json

import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String


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

    def write_social_update(
        self,
        actions: dict[int, str],
        social_states: dict[int, str],
        world_xy: dict[int, tuple[float, float]] | None,
        clusters: dict[int, int] | None,
        target_pid: int | None,
    ) -> None:
        world_xy = world_xy or {}
        clusters = clusters or {}
        payload = [
            {
                "id": pid,
                "action": actions.get(pid, "unknown"),
                "social_state": social_states.get(pid, "UNKNOWN"),
                "map_x": world_xy.get(pid, (None, None))[0],
                "map_y": world_xy.get(pid, (None, None))[1],
                "cluster_id": clusters.get(pid),
                "is_target": pid == target_pid,
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
