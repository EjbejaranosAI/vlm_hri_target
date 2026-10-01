"""chunk_gait_by_pid: ruido de keypoints de alguien quieto no es marcha;
la tijera de piernas real (oscilación L-R de ~1 paso) sí."""

import math

import numpy as np

from vlm_hri.pose.gait import (
    L_ANKLE, L_HIP, L_KNEE, R_ANKLE, R_HIP, R_KNEE, chunk_gait_by_pid,
)


def _kpts_with_knees(l_deg: float, r_deg: float) -> np.ndarray:
    """17x3 (COCO) con cadera-rodilla-tobillo formando el ángulo de rodilla dado."""
    kpts = np.zeros((17, 3), dtype=float)
    for hip, knee, ankle, ang, x0 in ((L_HIP, L_KNEE, L_ANKLE, l_deg, 100.0), (R_HIP, R_KNEE, R_ANKLE, r_deg, 140.0)):
        kpts[hip] = (x0, 100.0, 0.9)
        kpts[knee] = (x0, 200.0, 0.9)
        t = math.radians(180.0 - ang)  # 180° = pierna recta
        kpts[ankle] = (x0 + 100.0 * math.sin(t), 200.0 + 100.0 * math.cos(t), 0.9)
    return kpts


def _chunk(diffs: list[float]) -> list[list[dict]]:
    return [[{"pid": 1, "x1": 0, "y1": 0, "x2": 200, "y2": 400, "kpts": _kpts_with_knees(170.0, 170.0 - d)}] for d in diffs]


def test_standing_noise_with_spikes_is_not_gait():
    # Medido en vivo (esperando.mp4, gente en una cola): ruido de +-2° con un
    # pico aislado de un frame -- con el criterio viejo (5° max-min y 2
    # cruces de la media) esto salía como marcha.
    rng = np.random.default_rng(0)
    diffs = list(rng.uniform(-2, 2, 24))
    diffs[10] = 14.0
    assert chunk_gait_by_pid(_chunk(diffs), [1]) == {}


def test_slow_drift_is_not_gait():
    # Cambio de peso de una pierna a otra: deriva sin volver.
    diffs = list(np.linspace(-3, 6, 24))
    assert chunk_gait_by_pid(_chunk(diffs), [1]) == {}


def test_leg_scissor_is_gait():
    # ~1 paso por segundo, +-15° de diferencia L-R, 24 fps.
    diffs = [15.0 * math.sin(2 * math.pi * 1.0 * i / 24.0) for i in range(24)]
    assert chunk_gait_by_pid(_chunk(diffs), [1]) == {1: True}
