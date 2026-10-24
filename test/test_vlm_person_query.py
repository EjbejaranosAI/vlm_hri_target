"""Consulta por persona (VLM_QUERY_MODE=per_person) y mirada -> ATTENTIVE."""

import numpy as np

from vlm_hri.pose.gait import (
    L_EYE, L_SHOULDER, NOSE, R_EYE, R_SHOULDER, VLM_ANNOT_MAX_SIDE, annotate_for_vlm_person,
    attach_pose_keypoints, person_crop_box, upgrade_attentive_by_gaze,
)
from vlm_hri.social_state import (
    SOCIAL_ATTENTIVE, SOCIAL_AVAILABLE, SOCIAL_BUSY, SOCIAL_ENGAGED, SOCIAL_MOVING,
)
from vlm_hri.vlm.prompts import vlm_prompt_person


def test_person_frame_is_downscaled_and_only_that_box_is_drawn():
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    dets = [
        {"pid": 1, "x1": 100, "y1": 100, "x2": 900, "y2": 2000},
        {"pid": 2, "x1": 2000, "y1": 100, "x2": 2800, "y2": 2000},
    ]
    out = annotate_for_vlm_person(frame, dets, pid=2)
    assert max(out.shape[:2]) == VLM_ANNOT_MAX_SIDE
    s = VLM_ANNOT_MAX_SIDE / 3840
    left_half, right_half = out[:, : int(1500 * s)], out[:, int(1500 * s):]
    assert left_half.max() == 0  # la caja de la persona 1 NO se dibuja
    assert right_half.max() > 0


def test_person_prompt_asks_for_single_object():
    p = vlm_prompt_person(video=True)
    assert "YELLOW" in p and '{"a":' in p


def _facing_kpts() -> np.ndarray:
    k = np.zeros((17, 3), dtype=float)
    k[L_EYE] = (90, 50, 0.9)
    k[R_EYE] = (110, 50, 0.9)
    k[NOSE] = (100, 60, 0.9)
    return k


def test_gaze_promotes_engaged_but_not_busy():
    # Medido: el VLM confunde "sonríe mirando a la cámara" con "habla".
    buf = [[{"pid": 1, "kpts": _facing_kpts()}, {"pid": 2, "kpts": _facing_kpts()}] for _ in range(6)]
    out = upgrade_attentive_by_gaze({1: SOCIAL_ENGAGED, 2: SOCIAL_BUSY}, buf, [1, 2])
    assert out == {1: SOCIAL_ATTENTIVE, 2: SOCIAL_BUSY}


def test_gaze_promotion_off_keeps_engaged_but_still_degrades():
    # VLM_BACKEND=vlm_jev: sin promoción; degradar ATTENTIVE sin mirada sigue.
    buf = [[{"pid": 1, "kpts": _facing_kpts()}, {"pid": 2}] for _ in range(6)]
    out = upgrade_attentive_by_gaze({1: SOCIAL_ENGAGED, 2: SOCIAL_ATTENTIVE}, buf, [1, 2], promote=False)
    assert out == {1: SOCIAL_ENGAGED, 2: SOCIAL_AVAILABLE}


def _back_kpts() -> np.ndarray:
    k = np.zeros((17, 3), dtype=float)
    k[L_SHOULDER] = (80, 100, 0.9)
    k[R_SHOULDER] = (120, 100, 0.9)
    k[NOSE] = (100, 60, 0.05)
    return k


def test_back_turned_becomes_busy_but_moving_and_engaged_stay():
    buf = [[{"pid": p, "kpts": _back_kpts()} for p in (1, 2, 3, 4)] for _ in range(6)]
    social = {1: SOCIAL_AVAILABLE, 2: SOCIAL_ATTENTIVE, 3: SOCIAL_MOVING, 4: SOCIAL_ENGAGED}
    out = upgrade_attentive_by_gaze(social, buf, [1, 2, 3, 4])
    assert out == {1: SOCIAL_BUSY, 2: SOCIAL_BUSY, 3: SOCIAL_MOVING, 4: SOCIAL_ENGAGED}


def test_facing_person_next_to_back_turned_one_is_not_copied():
    buf = [[{"pid": 1, "kpts": _back_kpts()}, {"pid": 2, "kpts": _facing_kpts()}] for _ in range(6)]
    out = upgrade_attentive_by_gaze({1: SOCIAL_AVAILABLE, 2: SOCIAL_AVAILABLE}, buf, [1, 2])
    assert out == {1: SOCIAL_BUSY, 2: SOCIAL_ATTENTIVE}


def test_no_pose_data_does_not_force_busy():
    buf = [[{"pid": 1}] for _ in range(6)]
    assert upgrade_attentive_by_gaze({1: SOCIAL_AVAILABLE}, buf, [1]) == {1: SOCIAL_AVAILABLE}


def test_pose_keypoints_are_matched_one_to_one():
    # Dos poses que solapan más con la caja A: antes ambas iban a A (la
    # segunda pisaba a la primera) y B se quedaba sin keypoints.
    a = {"pid": 1, "x1": 0, "y1": 0, "x2": 100, "y2": 200}
    b = {"pid": 2, "x1": 60, "y1": 0, "x2": 160, "y2": 200}
    ka, kb = _facing_kpts(), _back_kpts()
    poses = [
        {"x1": 0, "y1": 0, "x2": 100, "y2": 200, "kpts": ka},
        {"x1": 40, "y1": 0, "x2": 140, "y2": 200, "kpts": kb},
    ]
    attach_pose_keypoints([a, b], poses)
    assert a["kpts"] is ka and b["kpts"] is kb


def test_person_crop_zooms_on_that_person():
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    frame[:, 2000:2800] = 200  # la persona 2
    dets = [
        {"pid": 1, "x1": 100, "y1": 100, "x2": 900, "y2": 2000},
        {"pid": 2, "x1": 2000, "y1": 100, "x2": 2800, "y2": 2000},
    ]
    crop = person_crop_box([dets, dets], 2, (3840, 2160))
    assert crop[0] >= 900 and crop[2] <= 3840  # no entra la persona 1
    out = annotate_for_vlm_person(frame, dets, pid=2, crop=crop)
    assert max(out.shape[:2]) == VLM_ANNOT_MAX_SIDE
    assert (out == 200).all(axis=2).mean() > 0.4  # la persona ocupa buena parte del clip
