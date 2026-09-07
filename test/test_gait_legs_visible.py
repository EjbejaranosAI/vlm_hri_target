"""Regresión: alguien sentado muy cerca de la cámara (solo torso/cara
visible, piernas fuera de cuadro) no debe salir como "walking"/MOVING solo
por la señal experimental de profundidad (chunk_depth_motion_by_pid, que se
confunde con gesticular/inclinarse/comer/hablar por teléfono -- ver su
docstring). Sin marcha real que confirmar (piernas no visibles ->
chunk_gait_by_pid no puede aportar nada), ese hint no debe forzar "walking"."""

import numpy as np

from vlm_hri.pose.gait import refine_labels_pose
from vlm_hri.social_state import SOCIAL_ATTENTIVE


def _torso_only_kpts() -> np.ndarray:
    """17x3 (COCO) -- cara/hombros con buena confianza, cadera/rodilla/
    tobillo por debajo de KPT_CONF_MIN (0.3): piernas fuera de cuadro."""
    kpts = np.zeros((17, 3), dtype=float)
    kpts[:, 2] = 0.1  # todo baja confianza por defecto
    for i in (0, 1, 2, 3, 4, 5, 6):  # nose, eyes, ears, shoulders
        kpts[i, 2] = 0.9
    return kpts


def _buffer_dets_growing_box(pid: int, n_frames: int = 10) -> list[list[dict]]:
    """Caja que crece de alto ~13% en el trozo (dispara chunk_depth_motion_by_pid,
    umbral DEPTH_MOTION_REL_CHANGE=0.08) simulando a alguien inclinándose hacia
    la cámara mientras come/gesticula -- CRECE SIMÉTRICA (mismo centro) para
    que chunk_motion_by_pid (centro de caja, señal independiente y confiable)
    no se dispare también y el test aísle de verdad la señal de profundidad."""
    out = []
    center_y = 300
    for i in range(n_frames):
        half_h = 100 + i * 1.5  # 100 -> 113.5, ~13.5% de crecimiento
        out.append(
            [
                {
                    "pid": pid,
                    "x1": 100,
                    "y1": int(center_y - half_h),
                    "x2": 300,
                    "y2": int(center_y + half_h),
                    "kpts": _torso_only_kpts(),
                }
            ]
        )
    return out


def test_depth_only_hint_does_not_force_walking_without_legs():
    buffer_dets = _buffer_dets_growing_box(pid=1)
    actions = {1: "eating"}
    social = {1: SOCIAL_ATTENTIVE}

    # Simula lo que ya decidió Fase 1 (gait+depth+histéresis, ver
    # _pose_vlm_worker): con piernas no visibles, gait no pudo aportar nada,
    # así que este hint viene solo de la señal de profundidad.
    acts, _soc = refine_labels_pose(
        actions, social, buffer_dets, [1], frame_size=(640, 480), moving_pids_hint={1}
    )

    assert "walking" not in acts[1].lower(), acts
    assert acts[1] == "eating", acts


def _standing_kpts() -> np.ndarray:
    """17x3 con piernas rectas y visibles (cadera/rodilla/tobillo en línea,
    coordenadas reales no-cero -- para que _knee_angle dé un ángulo válido,
    no solo alta confianza)."""
    kpts = np.zeros((17, 3), dtype=float)
    kpts[:, 2] = 0.9
    coords = {
        11: (140, 200), 12: (160, 200),  # caderas
        13: (140, 250), 14: (160, 250),  # rodillas
        15: (140, 300), 16: (160, 300),  # tobillos
    }
    for i, (x, y) in coords.items():
        kpts[i, 0], kpts[i, 1] = x, y
    return kpts


def test_kinematic_hint_never_rewrites_the_vlm_text_even_with_real_gait():
    """El VLM ya ve el clip de varios frames (+ pista de color) y decide el
    movimiento él mismo -- ni con evidencia cinemática real y confirmada
    (piernas visibles, moving_pids_hint activo) se le reescribe el texto de
    la acción. `moving_pids_hint`/`confirmed_moving` solo sirven para
    corregir el sesgo de grupo conocido (debias_group_walking) y para la
    pista de color del prompt, nunca para inyectar/quitar "walking" del
    texto -- ver refine_labels_pose."""
    buffer_dets = [
        [{"pid": 2, "x1": 100, "y1": 100, "x2": 200, "y2": 300, "kpts": _standing_kpts()}]
        for _ in range(10)
    ]
    actions = {2: "watch camera"}
    social = {2: SOCIAL_ATTENTIVE}

    acts, _soc = refine_labels_pose(
        actions, social, buffer_dets, [2], frame_size=(640, 480), moving_pids_hint={2}
    )

    # enrich_action_posture puede anteponer "standing"/"sitting" inferido de
    # la caja (eso es aparte, y sigue activo) -- lo que NO debe pasar es que
    # "walking" aparezca en el texto solo por la señal cinemática.
    assert "watch camera" in acts[2], acts
    assert "walking" not in acts[2].lower(), acts
