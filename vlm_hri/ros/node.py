"""Nodo ROS2: dos modos de entrada, elegidos con el parámetro
`use_dynamic_tracking` (default true):

- true (con dynamic-tracking, ver /home/edison/Desktop/PhD/RAL2026/dynamic-tracking):
  se suscribe a la imagen cruda del robot y a /detections/tracked
  (vision_msgs/Detection2DArray, ya con track_id estable y posición real en
  el mundo vía LiDAR 360°+YOLO auxiliar), y corre localmente SOLO el modelo
  de pose (para marcha/gait -- dynamic_tracking no hace pose).
- false (solo cámara, sin dynamic_tracking): se suscribe SOLO a la imagen y
  hace su propia detección+tracking de personas (mismo `detect_people` +
  `track_detections`/`assign_spatial_ids` que usa `main.py stream --camera`)
  -- sin posición real, así que el clustering/proximidad LiDAR queda inactivo
  (igual que el CLI de cámara plano).

Ambos modos alimentan el mismo PoseStreamSession que usa `python main.py stream`.

Requiere numpy<2 en el entorno del nodo -- cv_bridge (empaquetado con ROS2
Jazzy) está compilado contra numpy 1.x; con numpy>=2 (lo que trae este
venv por defecto, para torch/ultralytics) `cv_bridge` falla al importar.
Mismo problema y misma solución que ya documenta dynamic-tracking/README.md
("NumPy 2.x often breaks OpenCV/Ultralytics stacks").
"""

from __future__ import annotations

import threading
from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray

from ..config import POSE_YOLO_WEIGHTS, VIDEO_VLM_CHUNK_SEC, VLM_ID, VLM_WARMUP, YOLO_WEIGHTS
from ..detection import assign_spatial_ids, detect_people, resolve_device, track_detections, warmup_yolo
from ..pose import gait as pose_gait
from ..runners.pose_session import PoseStreamSession
from ..vlm.inference import warmup_vlm
from ..vlm.model import load_vlm
from .detections_bridge import detection2d_array_to_dets
from .sink import RosPublisherSink


def _sensor_qos(reliability: str, depth: int = 5) -> QoSProfile:
    rel = (
        QoSReliabilityPolicy.RELIABLE
        if reliability.strip().lower() == "reliable"
        else QoSReliabilityPolicy.BEST_EFFORT
    )
    return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=max(1, depth), reliability=rel)


class VlmHriNode(Node):
    def __init__(self) -> None:
        super().__init__("vlm_hri_node")

        self.declare_parameter("image_topic", "/head_front_camera/color/image_raw")
        self.declare_parameter("detections_topic", "/detections/tracked")
        self.declare_parameter("use_dynamic_tracking", True)
        self.declare_parameter("image_qos_reliability", "best_effort")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("chunk_sec", float(VIDEO_VLM_CHUNK_SEC))
        # Sin un fps real de video que medir (llegan detecciones por evento,
        # no frames a ritmo fijo) -- se asume uno, igual que ya hace el CLI
        # en modo --camera (STREAM_CAMERA_FPS), para el tamaño en frames de
        # cada trozo de ~1s del VLM.
        self.declare_parameter("assumed_fps", 30.0)
        self.declare_parameter("use_pose", True)
        self.declare_parameter("draw_pose", False)
        self.declare_parameter("sync_slop_sec", 0.10)
        self.declare_parameter("chunk_dir", str(Path.home() / ".ros" / "vlm_hri_chunks"))

        image_topic = str(self.get_parameter("image_topic").value)
        # Convención ROS image_transport: el tópico "…/compressed" publica
        # sensor_msgs/CompressedImage, no sensor_msgs/Image -- se detecta por
        # el nombre para no necesitar un parámetro aparte que el usuario
        # tenga que recordar poner en sincronía con el tópico.
        self._image_compressed = image_topic.rstrip("/").endswith("/compressed")
        self._use_dynamic_tracking = bool(self.get_parameter("use_dynamic_tracking").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        image_qos_str = str(self.get_parameter("image_qos_reliability").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._chunk_sec = float(self.get_parameter("chunk_sec").value)
        self._assumed_fps = float(self.get_parameter("assumed_fps").value)
        self._use_pose = bool(self.get_parameter("use_pose").value)
        self._draw_pose = bool(self.get_parameter("draw_pose").value)
        slop = float(self.get_parameter("sync_slop_sec").value)
        self._chunk_dir = Path(str(self.get_parameter("chunk_dir").value))

        self._bridge = CvBridge()
        self._device = resolve_device()
        self._yolo: YOLO | None = None
        self._prev_dets: list[dict] = []
        if not self._use_dynamic_tracking:
            self.get_logger().info("Cargando YOLO (detección+tracking propios, sin dynamic_tracking) …")
            self._yolo = YOLO(YOLO_WEIGHTS)
        self._yolo_pose: YOLO | None = None
        if self._use_pose:
            self.get_logger().info("Cargando YOLO-pose …")
            self._yolo_pose = YOLO(POSE_YOLO_WEIGHTS)
        self.get_logger().info("Cargando VLM …")
        self._vlm, self._processor = load_vlm(VLM_ID, self._device)
        if VLM_WARMUP:
            warmup_vlm(self._vlm, self._processor)
        self.get_logger().info("VLM listo")

        self._session: PoseStreamSession | None = None
        self._session_lock = threading.Lock()

        image_msg_type = CompressedImage if self._image_compressed else Image
        kind = "CompressedImage" if self._image_compressed else "Image"
        if self._use_dynamic_tracking:
            image_sub = Subscriber(self, image_msg_type, image_topic, qos_profile=_sensor_qos(image_qos_str))
            det_sub = Subscriber(self, Detection2DArray, detections_topic, qos_profile=10)
            self._sync = ApproximateTimeSynchronizer(
                [image_sub, det_sub], queue_size=16, slop=slop
            )
            self._sync.registerCallback(self._on_synced)
            self.get_logger().info(f"Suscrito a {image_topic} ({kind}) + {detections_topic} (slop={slop}s)")
        else:
            self.create_subscription(
                image_msg_type, image_topic, self._on_image_only, _sensor_qos(image_qos_str)
            )
            self.get_logger().info(
                f"Suscrito a {image_topic} ({kind}) -- detección/tracking propios, sin dynamic_tracking "
                "(sin posición LiDAR: clustering/proximidad ENGAGED inactivos)"
            )

    def _ensure_session(self, w: int, h: int) -> PoseStreamSession:
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is None:
                if self._yolo is not None:
                    warmup_yolo(self._yolo, shape=(h, w, 3))
                if self._yolo_pose is not None:
                    warmup_yolo(self._yolo_pose, shape=(h, w, 3))
                sink = RosPublisherSink(self, frame_id=self._frame_id)
                self._session = PoseStreamSession(
                    w=w, h=h, fps=self._assumed_fps, chunk_sec=self._chunk_sec,
                    chunk_dir=self._chunk_dir, vlm=self._vlm, processor=self._processor,
                    draw_pose=self._draw_pose, sinks=[sink],
                )
                self._session.start()
                self.get_logger().info(f"Sesión iniciada ({w}x{h} @ {self._assumed_fps} fps asumidos)")
        return self._session

    def _decode_image(self, image_msg: Image | CompressedImage):
        if self._image_compressed:
            return self._bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        return self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

    def _on_synced(self, image_msg: Image | CompressedImage, det_msg: Detection2DArray) -> None:
        frame = self._decode_image(image_msg)
        h, w = frame.shape[:2]
        session = self._ensure_session(w, h)

        dets, _world_xy = detection2d_array_to_dets(det_msg)
        if self._use_pose and self._yolo_pose is not None:
            pose_raw = pose_gait.detect_people_pose(self._yolo_pose, frame)
            pose_gait.attach_pose_keypoints(dets, pose_raw)

        session.append_frame(frame, dets)
        session.tick()

    def _on_image_only(self, image_msg: Image | CompressedImage) -> None:
        """Sin dynamic_tracking: detección+tracking propios (mismo criterio
        que `main.py stream --camera`), sin posición real -- world_x/world_y
        quedan ausentes en cada det, así que _collect_world_xy en
        pose_session.py no aporta nada y el chequeo de proximidad LiDAR
        queda en no-op automáticamente (igual que en el CLI plano)."""
        frame = self._decode_image(image_msg)
        h, w = frame.shape[:2]
        session = self._ensure_session(w, h)

        raw = detect_people(self._yolo, frame)
        dets = track_detections(self._prev_dets, raw) if self._prev_dets else assign_spatial_ids(raw)
        self._prev_dets = dets
        if self._use_pose and self._yolo_pose is not None:
            pose_raw = pose_gait.detect_people_pose(self._yolo_pose, frame)
            pose_gait.attach_pose_keypoints(dets, pose_raw)

        session.append_frame(frame, dets)
        session.tick()

    def destroy_node(self) -> None:
        if self._session is not None:
            self._session.flush_and_stop()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VlmHriNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
