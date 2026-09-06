"""Launch de vlm_hri_node -- mismo patrón profile:=sim|robot que
dynamic_tracking (ver dynamic-tracking/launch/dynamic_tracking.launch.py):
elige el YAML de config/ según el perfil, con args para override puntual sin
tocar el YAML."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_params_file(context):
    explicit = LaunchConfiguration("params_file").perform(context).strip()
    if explicit:
        return explicit
    profile = LaunchConfiguration("profile").perform(context).strip().lower()
    pkg = get_package_share_directory("vlm_hri_target")
    name = (
        "vlm_hri_params_sim.yaml" if profile == "sim" else "vlm_hri_params_robot.yaml"
    )
    return os.path.join(pkg, "config", name)


def _launch_setup(context, *args, **kwargs):
    params_path = _resolve_params_file(context)
    node_params = {
        "image_topic": LaunchConfiguration("image_topic"),
        "detections_topic": LaunchConfiguration("detections_topic"),
        "use_pose": LaunchConfiguration("use_pose"),
        "draw_pose": LaunchConfiguration("draw_pose"),
    }
    node = Node(
        package="vlm_hri_target",
        executable="vlm_hri_node",
        name="vlm_hri_node",
        parameters=[params_path, node_params],
        output="screen",
    )
    return [node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile",
                default_value="robot",
                description="robot (default, TIAGO real) o sim (Gazebo) -- decide el frame_id (odom/map)",
                choices=["robot", "sim"],
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value="",
                description="Override YAML; vacío = según profile",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/head_front_camera/color/image_raw",
                description="Cámara RGB del robot (sensor_msgs/Image)",
            ),
            DeclareLaunchArgument(
                "detections_topic",
                default_value="/detections/tracked",
                description="Salida de dynamic_tracking (vision_msgs/Detection2DArray, con track_id + posición real)",
            ),
            DeclareLaunchArgument(
                "use_pose",
                default_value="true",
                description="Corre YOLO-pose local para marcha/gait (dynamic_tracking no hace pose)",
            ),
            DeclareLaunchArgument(
                "draw_pose",
                default_value="false",
                description="Dibuja el esqueleto COCO-17 en /vlm_hri/image_annotated",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
