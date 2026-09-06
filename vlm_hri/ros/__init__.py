"""Puente ROS2: nodo, adaptador de mensajes y sink de publicación.

Aislado del resto del paquete a propósito -- `rclpy`/`vision_msgs`/
`cv_bridge` solo se importan aquí, nunca desde `main.py` ni el resto de
`vlm_hri/`, así que el CLI plano (`python main.py ...`) sigue sin depender de
tener ROS2 instalado."""

from __future__ import annotations
