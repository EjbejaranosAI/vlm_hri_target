"""Variante del pipeline con detector de POSE (yolo26n-pose.pt) en vez de solo
detección (yolo11n.pt). Añade una señal de movimiento independiente del VLM
—marcha real de piernas, no el centro de la caja— y la pasa al VLM como una
PISTA visual (caja azul=en movimiento, roja=parada, sin texto de estado) para
que el propio VLM decida la acción/intención final combinando esa pista con
lo que ve.

Reutiliza del resto del paquete todo lo que no cambia: carga/inferencia del
VLM, tracking, parseo de respuesta, dibujo del video final. Los umbrales de
marcha (MIN_AMPLITUDE_DEG/MIN_CROSSINGS_PER_CHUNK) se calibraron con datos
reales (control negativo de gente de pie vs. caminata lateral confirmada).
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np

from ..actions import _action_secondary_parts, _action_traits, normalize_action
from ..config import YOLO_CONF, YOLO_IOU, YOLO_MAX_DET, YOLO_POST_NMS_IOU
from ..detection import box_iou, nms_boxes, sort_dets_left_right
from ..drawing import pid_label, ui_scale
from ..motion import chunk_motion_by_pid, refine_chunk_labels
from ..clustering import same_cluster
from ..social_state import (
    SOCIAL_ATTENTIVE,
    SOCIAL_AVAILABLE,
    SOCIAL_ENGAGED,
    SOCIAL_MOVING,
    SOCIAL_UNKNOWN,
    normalize_social_state,
)
from ..vlm.prompts import MOTION_COLOR_HINT, vlm_prompt

# Tope de personas que se le mandan al VLM por trozo. Más gente = prompt más
# largo Y sobre todo más tokens de salida (uno por persona) = más latencia,
# medido: con 5 personas ya se acerca al techo de tokens antes de terminar de
# responder por todas. Se prioriza a quien está más cerca de la cámara (caja
# más grande) — quien está lejos importa menos para elegir con quién
# interactuar. Parámetro (no una constante quemada) para poder ampliarlo el
# día que se necesite atender a más gente a la vez.
MAX_VLM_PEOPLE = int(os.environ.get("MAX_VLM_PEOPLE", "10"))


def closest_n_pids(frame_rows: list[dict], pids: list[int], n: int) -> list[int]:
    """De `pids`, los `n` con mayor área de caja promedio (más cerca de la
    cámara), en ese orden. Si hay `n` o menos, los devuelve todos sin tocar
    el orden."""
    if len(pids) <= n:
        return list(pids)
    want = set(pids)
    areas: dict[int, list[float]] = {p: [] for p in pids}
    for row in frame_rows:
        pid = row.get("person_id")
        if pid in want:
            areas[pid].append((row["x2"] - row["x1"]) * (row["y2"] - row["y1"]))
    avg_area = {p: (sum(a) / len(a) if a else 0.0) for p, a in areas.items()}
    ranked = sorted(pids, key=lambda p: avg_area[p], reverse=True)
    return ranked[:n]

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
KPT_CONF_MIN = 0.3

# Pose COCO-17 completa (cabeza + brazos + tronco + piernas) — se dibuja sobre
# el frame que ve el VLM, no solo la caja de color, para darle más contexto
# visual (orientación de brazos/torso) del que puede sacar acción/intención.
SKELETON = [
    (L_ANKLE, L_KNEE), (L_KNEE, L_HIP), (R_ANKLE, R_KNEE), (R_KNEE, R_HIP),
    (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW), (L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST),
    (L_EYE, R_EYE), (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
    (L_EAR, L_SHOULDER), (R_EAR, R_SHOULDER),
]

# Calibrado con datos reales (control negativo
# (gente de pie confirmada) vs. caminata lateral confirmada → 5°/2 cruces da
# 74% de recall real con ~7% de falsos positivos (12°/2 daba solo 25% recall).
# Configurables por si un video concreto necesita más rigor que el default
# calibrado (subir cualquiera de los dos baja falsos positivos a costa de
# recall — ver MotionHysteresis para el otro control de rigor: confirmación
# por varios trozos antes de creer la evidencia cruda de este frame).
MIN_AMPLITUDE_DEG = float(os.environ.get("GAIT_MIN_AMPLITUDE_DEG", "5.0"))
MIN_CROSSINGS_PER_CHUNK = int(os.environ.get("GAIT_MIN_CROSSINGS", "2"))
# Muestras válidas (ambas rodillas visibles) mínimas en el trozo para evaluar
# la marcha — menos que esto es ruido, no una serie con la que medir
# amplitud/cruces de forma confiable.
MIN_GAIT_SAMPLES = int(os.environ.get("GAIT_MIN_SAMPLES", "4"))

COLOR_MOVING_BGR = (255, 0, 0)  # azul
COLOR_STANDING_BGR = (0, 0, 255)  # rojo
THIN_LINE_PX = 1


def detect_people_pose(yolo_pose, bgr: np.ndarray) -> list[dict]:
    """Como detection.detect_people pero cada detección incluye "kpts" (17,3)."""
    res = yolo_pose.predict(
        bgr,
        classes=[0],
        verbose=False,
        device=0,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        max_det=YOLO_MAX_DET,
    )[0]
    dets: list[dict] = []
    if res.boxes is None:
        return dets
    h, w = bgr.shape[:2]
    kpts_all = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
    for j in range(len(res.boxes)):
        x1, y1, x2, y2 = res.boxes.xyxy[j].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cid = int(res.boxes.cls[j].item())
        d = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class_name": res.names.get(cid, "person"),
            "conf": float(res.boxes.conf[j].item()),
        }
        if kpts_all is not None:
            d["kpts"] = kpts_all[j]
        dets.append(d)
    return nms_boxes(dets, YOLO_POST_NMS_IOU)


def _dets_from_result(res, bgr: np.ndarray, *, with_kpts: bool) -> list[dict]:
    dets: list[dict] = []
    if res.boxes is None:
        return dets
    h, w = bgr.shape[:2]
    kpts_all = (
        res.keypoints.data.cpu().numpy() if with_kpts and res.keypoints is not None else None
    )
    for j in range(len(res.boxes)):
        x1, y1, x2, y2 = res.boxes.xyxy[j].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cid = int(res.boxes.cls[j].item())
        d = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class_name": res.names.get(cid, "person"),
            "conf": float(res.boxes.conf[j].item()),
        }
        if kpts_all is not None:
            d["kpts"] = kpts_all[j]
        dets.append(d)
    return nms_boxes(dets, YOLO_POST_NMS_IOU)


def detect_people_batch(yolo, frames: list[np.ndarray]) -> list[list[dict]]:
    """Como detection.detect_people pero para VARIOS frames en una sola llamada
    a predict() — medido: ~6ms/frame uno a uno vs ~2ms/frame en lotes de 8+
    (la GPU está infrautilizada con lotes de 1 en modelos tan chicos)."""
    if not frames:
        return []
    results = yolo.predict(
        frames, classes=[0], verbose=False, device=0,
        conf=YOLO_CONF, iou=YOLO_IOU, max_det=YOLO_MAX_DET,
    )
    return [_dets_from_result(r, f, with_kpts=False) for r, f in zip(results, frames)]


def detect_people_pose_batch(yolo_pose, frames: list[np.ndarray]) -> list[list[dict]]:
    """Como detect_people_pose pero para VARIOS frames en una sola llamada."""
    if not frames:
        return []
    results = yolo_pose.predict(
        frames, classes=[0], verbose=False, device=0,
        conf=YOLO_CONF, iou=YOLO_IOU, max_det=YOLO_MAX_DET,
    )
    return [_dets_from_result(r, f, with_kpts=True) for r, f in zip(results, frames)]


# IoU mínimo para aceptar un emparejamiento pose↔caja trackeada; por debajo de
# esto se descarta (mejor sin keypoints ese frame que asignarlos a la persona
# equivocada).
POSE_MATCH_MIN_IOU = 0.3


def attach_pose_keypoints(tracked_dets: list[dict], pose_dets: list[dict]) -> None:
    """Empareja por IoU cada detección de POSE con su caja ya trackeada (de
    yolo11n, vía detection.detect_people + track_detections) y le añade "kpts".

    Deliberadamente NO se usa la pose para detectar/trackear: la caja de
    yolo26n-pose es menos estable frame a frame (reparte capacidad entre
    detectar y estimar pose) y romper el tracking en IDs nuevos a cada rato —
    medido en la práctica: una persona seguida sin cortes en 720 frames por
    yolo11n quedaba partida en 3-4 IDs distintos con el detector de pose. Se
    mantiene el tracking probado y estable, y la pose solo aporta keypoints
    encima de esas cajas ya trackeadas.
    """
    for pd_ in pose_dets:
        best, best_iou = None, 0.0
        for td in tracked_dets:
            iou = box_iou(pd_, td)
            if iou > best_iou:
                best_iou, best = iou, td
        if best is not None and best_iou >= POSE_MATCH_MIN_IOU and "kpts" in pd_:
            best["kpts"] = pd_["kpts"]


def _knee_angle(kpts: np.ndarray, hip_i: int, knee_i: int, ankle_i: int) -> float | None:
    hip, knee, ankle = kpts[hip_i], kpts[knee_i], kpts[ankle_i]
    if hip[2] < KPT_CONF_MIN or knee[2] < KPT_CONF_MIN or ankle[2] < KPT_CONF_MIN:
        return None
    v1 = hip[:2] - knee[:2]
    v2 = ankle[:2] - knee[:2]
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def compensate_camera_motion(buffer_dets: list[list[dict]]) -> list[list[dict]]:
    """Copia de `buffer_dets` con el desplazamiento COMÚN a todas las personas
    (la mediana, no el promedio, para que 1-2 personas caminando de verdad no
    lo distorsionen) restado acumulativamente de cada caja.

    Las señales de movimiento miden posición ABSOLUTA en la imagen — si la
    propia cámara se mueve un poco (temblor, paneo), TODAS las cajas se
    desplazan igual y parece que todo el mundo camina. Si varias personas se
    mueven exactamente igual al mismo tiempo, lo más probable es que sea la
    cámara, no ellas.

    Con menos de 2 personas simultáneas no hay forma de distinguir "la cámara
    se movió" de "esa persona caminó" — en ese caso se deja tal cual (sin
    compensar), igual que antes."""
    if not buffer_dets:
        return buffer_dets
    out: list[list[dict]] = [[dict(d) for d in row] for row in buffer_dets]
    cum_dx, cum_dy = 0.0, 0.0
    for i in range(1, len(buffer_dets)):
        prev_c = {
            d["pid"]: (0.5 * (d["x1"] + d["x2"]), 0.5 * (d["y1"] + d["y2"]))
            for d in buffer_dets[i - 1]
        }
        cur_c = {
            d["pid"]: (0.5 * (d["x1"] + d["x2"]), 0.5 * (d["y1"] + d["y2"]))
            for d in buffer_dets[i]
        }
        common = set(prev_c) & set(cur_c)
        if len(common) >= 2:
            dxs = [cur_c[p][0] - prev_c[p][0] for p in common]
            dys = [cur_c[p][1] - prev_c[p][1] for p in common]
            cum_dx += float(np.median(dxs))
            cum_dy += float(np.median(dys))
        if cum_dx or cum_dy:
            for d in out[i]:
                d["x1"] -= cum_dx
                d["x2"] -= cum_dx
                d["y1"] -= cum_dy
                d["y2"] -= cum_dy
    return out


def chunk_gait_by_pid(
    buffer_dets: list[list[dict]], person_ids: list[int]
) -> dict[int, bool]:
    """True si esa persona muestra tijera de piernas real (caminando) en el trozo.

    Caminar de verdad es una alternancia periódica del ángulo rodilla
    izquierda vs. derecha; balancearse de pie no la tiene. Umbrales
    calibrados empíricamente (ver MIN_AMPLITUDE_DEG/MIN_CROSSINGS_PER_CHUNK).
    """
    out: dict[int, bool] = {}
    for pid in person_ids:
        diffs: list[float] = []
        for row in buffer_dets:
            for d in row:
                if d.get("pid") != pid or "kpts" not in d:
                    continue
                l_ang = _knee_angle(d["kpts"], L_HIP, L_KNEE, L_ANKLE)
                r_ang = _knee_angle(d["kpts"], R_HIP, R_KNEE, R_ANKLE)
                if l_ang is not None and r_ang is not None:
                    diffs.append(l_ang - r_ang)
                break
        if len(diffs) < MIN_GAIT_SAMPLES:
            continue
        arr = np.array(diffs)
        amplitude = float(arr.max() - arr.min())
        signs = np.sign(arr - arr.mean())
        crossings = int(np.sum(signs[1:] != signs[:-1]))
        if amplitude >= MIN_AMPLITUDE_DEG and crossings >= MIN_CROSSINGS_PER_CHUNK:
            out[pid] = True
    return out


# EXPERIMENTAL — sin calibrar con datos reales (a diferencia de
# MIN_AMPLITUDE_DEG/MIN_CROSSINGS_PER_CHUNK). Deliberadamente laxo: basta con
# un par de frames buenos en el trozo para contar como "piernas visibles".
LEGS_VISIBLE_RATIO = 0.3


def chunk_legs_visible_by_pid(
    buffer_dets: list[list[dict]], person_ids: list[int]
) -> dict[int, bool]:
    """True si se vieron cadera/rodilla/tobillo (cualquier pierna) con
    confianza usable en suficientes frames del trozo como para confiar en
    CUALQUIER juicio de sentado/de pie sobre esa persona; False si la parte
    baja del cuerpo estuvo fuera de cuadro/ocluida la mayor parte del trozo
    (típico: persona muy cerca de la cámara, solo torso/cara en el frame) —
    en ese caso no debería aplicarse ningún heurístico de postura en absoluto
    (ver vlm_hri.motion.enrich_action_posture). Ausente del dict = sin datos
    de pose ese trozo (el llamador cae al comportamiento anterior)."""
    out: dict[int, bool] = {}
    for pid in person_ids:
        seen = 0
        legs_ok = 0
        for row in buffer_dets:
            for d in row:
                if d.get("pid") != pid or "kpts" not in d:
                    continue
                seen += 1
                l_ok = _knee_angle(d["kpts"], L_HIP, L_KNEE, L_ANKLE) is not None
                r_ok = _knee_angle(d["kpts"], R_HIP, R_KNEE, R_ANKLE) is not None
                if l_ok or r_ok:
                    legs_ok += 1
                break
        if seen == 0:
            continue
        out[pid] = (legs_ok / seen) >= LEGS_VISIBLE_RATIO
    return out


# EXPERIMENTAL — sin calibrar con datos reales todavía (a diferencia de
# MIN_AMPLITUDE_DEG/MIN_CROSSINGS_PER_CHUNK, que sí se midieron). Complementa
# la tijera de piernas: alguien caminando derecho hacia/desde la cámara apenas
# balancea las piernas lateralmente (por eso a veces no se detectaba), pero su
# caja SÍ crece o encoge de forma sostenida. Si en la práctica da muchos
# falsos positivos (p. ej. alguien agachándose), avisar para recalibrar igual
# que se hizo con la marcha.
DEPTH_MOTION_REL_CHANGE = float(os.environ.get("DEPTH_MOTION_REL_CHANGE", "0.08"))
MIN_DEPTH_SAMPLES = int(os.environ.get("DEPTH_MOTION_MIN_SAMPLES", "6"))


def chunk_depth_motion_by_pid(
    buffer_dets: list[list[dict]], person_ids: list[int]
) -> dict[int, bool]:
    """True si el alto de la caja crece/encoge de forma sostenida en el trozo
    (acercándose o alejándose de la cámara en línea recta)."""
    out: dict[int, bool] = {}
    for pid in person_ids:
        heights: list[float] = []
        for row in buffer_dets:
            for d in row:
                if d.get("pid") == pid:
                    heights.append(float(d["y2"] - d["y1"]))
                    break
        if len(heights) < MIN_DEPTH_SAMPLES:
            continue
        third = max(1, len(heights) // 3)
        h0 = float(np.mean(heights[:third]))
        h1 = float(np.mean(heights[-third:]))
        if h0 <= 1e-6:
            continue
        rel_change = abs(h1 - h0) / h0
        if rel_change >= DEPTH_MOTION_REL_CHANGE:
            out[pid] = True
    return out


def debias_group_walking(
    actions: dict[int, str],
    social: dict[int, str],
    person_ids: list[int],
    confirmed_moving: set[int],
) -> tuple[dict[int, str], dict[int, str]]:
    """El VLM a veces le pega "walking" a TODO el grupo a la vez — el mismo
    sesgo que ya se conocía con "talking" (ver pipeline.debias_group_talking),
    ahora confirmado también con "walking": medido en un video real, 4 de 5
    personas se marcaron "walking" en el mismo trozo con 3-36px de
    desplazamiento total de su caja en 60 frames (nada, ruido de detección).

    Si 3+ personas y la mayoría/todo el grupo se marca caminando a la vez,
    solo se respeta a quien tiene evidencia cinemática INDEPENDIENTE de
    movimiento real (`confirmed_moving`: marcha, profundidad o bbox-center,
    NO el propio juicio del VLM) — al resto se le quita el "walking" y se
    deja que el resto del pipeline le asigne su postura real.

    También aplica con solo 2 personas: "los dos caminando a la vez" sin
    evidencia independiente es el mismo sesgo, solo que con grupo más chico
    (medido: sesión con 2 personas sentadas —solo torso visible— marcadas
    "walking"/MOVING varios trozos seguidos sin ninguna evidencia cinemática)."""
    n = len(person_ids)
    if n < 2:
        return actions, social

    def is_walking(pid: int) -> bool:
        return _action_traits(normalize_action(actions.get(pid, "")).lower())["walking"]

    walkers = [p for p in person_ids if is_walking(p)]
    if len(walkers) < n - 1:
        return actions, social

    out_a, out_s = dict(actions), dict(social)
    for pid in walkers:
        if pid in confirmed_moving:
            continue
        out_a[pid] = "unknown"
        out_s[pid] = SOCIAL_UNKNOWN
    return out_a, out_s


def require_moving_evidence(
    actions: dict[int, str],
    social: dict[int, str],
    confirmed_moving: set[int],
) -> tuple[dict[int, str], dict[int, str]]:
    """Modo rígido (default): CUALQUIER "walking"/"running" — venga del VLM
    o de un forzado anterior — sin evidencia cinemática independiente
    confirmada (`confirmed_moving`: marcha real + profundidad + histéresis
    de varios trozos, o respaldo bbox-center) se quita, sin importar cuántas
    personas haya en cuadro.

    `debias_group_walking` solo corrige el sesgo cuando el GRUPO entero se
    marca caminando a la vez (necesita 2+ personas y que la mayoría lo diga)
    — deja sin cubrir el caso más común de falso positivo: una sola persona
    en cuadro (sin "grupo" que debiasear) a la que el VLM le pega "walking"
    de pura alucinación, o a la que la pose confirma solo por 2-3 trozos
    seguidos (afinado, pero no infalible). Aquí se exige la misma evidencia
    para cualquiera, esté solo o en grupo. Desactivable con
    STRICT_MOVING_EVIDENCE=0 si en la práctica pierde demasiadas caminatas
    reales (falsos negativos)."""
    if os.environ.get("STRICT_MOVING_EVIDENCE", "1") != "1":
        return actions, social
    out_a = dict(actions)
    out_s = dict(social)
    for pid, act in actions.items():
        if pid in confirmed_moving:
            continue
        norm = normalize_action(act)
        tr = _action_traits(norm.lower())
        if not (tr.get("walking") or tr.get("running")):
            continue
        out_a[pid] = _strip_walking_words(norm)
        if normalize_social_state(str(out_s.get(pid, ""))) == SOCIAL_MOVING:
            out_s[pid] = SOCIAL_UNKNOWN
    return out_a, out_s


_WALK_RUN_AND_RE = re.compile(r"\b(walking|running)\b\s+and\s+", re.IGNORECASE)
_AND_WALK_RUN_RE = re.compile(r"\s+and\s+\b(walking|running)\b", re.IGNORECASE)
_ONLY_WALK_RUN_RE = re.compile(r"^\s*(walking|running)\s*$", re.IGNORECASE)


def _strip_walking_words(act: str) -> str:
    """Quita 'walking'/'running' (con su 'and' adyacente) del texto, dejando
    el resto de la actividad intacta -- p. ej. 'walking and watch camera' ->
    'watch camera', no solo 'unknown'. Antes (require_moving_evidence) se
    reconstruía desde _action_secondary_parts, que solo reconoce talking/
    smiling/using phone y perdía cualquier otra actividad en texto libre."""
    out = _WALK_RUN_AND_RE.sub("", act)
    out = _AND_WALK_RUN_RE.sub("", out)
    out = _ONLY_WALK_RUN_RE.sub("", out)
    return out.strip() or "unknown"


def check_engaged_proximity(
    actions: dict[int, str],
    social: dict[int, str],
    person_ids: list[int],
    world_xy: dict[int, tuple[float, float]] | None,
    clusters: dict[int, int] | None,
    *,
    max_dist_m: float | None = None,
) -> dict[int, str]:
    """Corrige ENGAGED con la posición real (LiDAR), no solo el juicio del
    VLM/heurística sobre la imagen: alguien solo puede estar "hablando con
    otra persona" (ENGAGED) si hay AL MENOS otra persona ENGAGED en su mismo
    cluster de proximidad (`clusters`, ver vlm_hri.clustering). Sin eso, se
    baja a ATTENTIVE -- mismo espíritu que upgrade_attentive_by_gaze (exigir
    evidencia positiva, no solo "no desmentido"), pero con evidencia espacial
    real en vez de pose.

    Solo aplica en el pipeline ROS2 (con tracker LiDAR externo) -- sin
    `world_xy`/`clusters` (cámara/video plano, sin posición real) es un no-op,
    igual que el resto de este pipeline se comporta hoy. Desactivable con
    STRICT_ENGAGED_PROXIMITY=0."""
    if not world_xy or not clusters:
        return social
    if os.environ.get("STRICT_ENGAGED_PROXIMITY", "1") != "1":
        return social
    engaged = [
        pid for pid in person_ids
        if normalize_social_state(str(social.get(pid, ""))) == SOCIAL_ENGAGED
    ]
    if len(engaged) < 2:
        # Nadie mas ENGAGED en absoluto -> ninguno tiene con quien "hablar".
        out = dict(social)
        for pid in engaged:
            out[pid] = SOCIAL_ATTENTIVE
        return out
    out = dict(social)
    for pid in engaged:
        corroborated = any(
            other != pid and same_cluster(pid, other, clusters)
            for other in engaged
        )
        if not corroborated:
            out[pid] = SOCIAL_ATTENTIVE
    return out


# Cuántos trozos (~1s cada uno) se mantiene "moviéndose" tras la última
# evidencia real, para no perderlo en trozos donde la señal se ahoga en ruido
# (p. ej. alguien lejos, caja chica) — sin esto, un par de trozos "ciegos"
# de por medio ya tumbaban el veredicto final por mayoría de votos.
HYSTERESIS_CHUNKS = int(os.environ.get("HYSTERESIS_CHUNKS", "3"))

# Rigor para ENCENDER "moviéndose" (evita pasarle al VLM un falso positivo de
# un solo trozo ruidoso como pista de color): no basta con evidencia cruda
# (gait o profundidad) en el trozo actual — se exige que aparezca en al menos
# `min_confirm_chunks` de los últimos `confirm_window_chunks` trozos. Antes,
# UN trozo con evidencia espuria (p. ej. un mal enganche de keypoints) no solo
# marcaba ESE trozo como moviéndose: por la histéresis de apagado, se
# propagaba `hold_chunks` trozos MÁS — un solo falso positivo se convertía en
# varios segundos de pista azul incorrecta para el VLM.
MIN_CONFIRM_CHUNKS = int(os.environ.get("GAIT_MIN_CONFIRM_CHUNKS", "2"))
CONFIRM_WINDOW_CHUNKS = int(os.environ.get("GAIT_CONFIRM_WINDOW_CHUNKS", "3"))


class MotionHysteresis:
    """Estado de "moviéndose" por persona a lo largo del video.

    Se ENCIENDE solo tras confirmar evidencia real (marcha o cambio de
    profundidad) en `min_confirm_chunks` de los últimos `confirm_window_chunks`
    trozos — no con un solo trozo ruidoso — y una vez encendida se mantiene
    (histéresis) `hold_chunks` trozos tras la última evidencia, para no
    parpadear en trozos donde la señal se ahoga en ruido."""

    def __init__(
        self,
        hold_chunks: int = HYSTERESIS_CHUNKS,
        min_confirm_chunks: int = MIN_CONFIRM_CHUNKS,
        confirm_window_chunks: int = CONFIRM_WINDOW_CHUNKS,
    ) -> None:
        self.hold_chunks = hold_chunks
        self.min_confirm_chunks = min_confirm_chunks
        self.confirm_window_chunks = max(confirm_window_chunks, min_confirm_chunks)
        self._raw_history: dict[int, list[int]] = {}
        self._last_moving_ci: dict[int, int] = {}

    def update(self, ci: int, raw_moving_pids: set[int], present_pids: set[int]) -> set[int]:
        for pid in present_pids:
            hist = self._raw_history.setdefault(pid, [])
            if pid in raw_moving_pids:
                hist.append(ci)
            self._raw_history[pid] = [
                c for c in hist if ci - c < self.confirm_window_chunks
            ]
        out: set[int] = set()
        for pid in present_pids:
            if len(self._raw_history.get(pid, [])) >= self.min_confirm_chunks:
                self._last_moving_ci[pid] = ci
            last = self._last_moving_ci.get(pid)
            if last is not None and ci - last <= self.hold_chunks:
                out.add(pid)
        return out


def draw_thin_pose_box(
    img: np.ndarray, x1, y1, x2, y2, label: str, kpts: np.ndarray | None, *, moving: bool
) -> None:
    """Caja delgada + ID (sin texto de estado) + esqueleto COCO-17 completo,
    todo del mismo color: azul si se mueve, rojo si no."""
    color = COLOR_MOVING_BGR if moving else COLOR_STANDING_BGR
    cv2.rectangle(img, (x1, y1), (x2, y2), color, THIN_LINE_PX)
    if kpts is not None:
        for a, b in SKELETON:
            pa, pb = kpts[a], kpts[b]
            if pa[2] >= KPT_CONF_MIN and pb[2] >= KPT_CONF_MIN:
                cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, THIN_LINE_PX)
        for x, y, c in kpts:
            if c >= KPT_CONF_MIN:
                cv2.circle(img, (int(x), int(y)), 2, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.4
    (tw, th), baseline = cv2.getTextSize(label, font, fs, 1)
    ty = max(th + 2, y1 - 2)
    cv2.rectangle(img, (x1, ty - th - 2), (x1 + tw + 4, ty + baseline), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 2, ty - 1), font, fs, (255, 255, 255), 1, cv2.LINE_AA)


POSE_SKELETON_BGR = (0, 215, 255)  # amarillo/naranja: no se confunde con los
# colores de estado social (verde/cian/naranja/rojo/azul) ni con el azul/rojo
# de moving/standing de draw_thin_pose_box (ese es para el clip del VLM).


def draw_pose_skeleton(img: np.ndarray, kpts: np.ndarray | None) -> None:
    """Esqueleto COCO-17 en color fijo, para visualizar la pose en el video
    final que ve el usuario — independiente del color azul/rojo de
    draw_thin_pose_box (ese codifica movimiento y es solo para el clip que ve
    el VLM). Pensado para depurar/inspeccionar la pose, no para decidir nada."""
    if kpts is None:
        return
    for a, b in SKELETON:
        pa, pb = kpts[a], kpts[b]
        if pa[2] >= KPT_CONF_MIN and pb[2] >= KPT_CONF_MIN:
            cv2.line(
                img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                POSE_SKELETON_BGR, THIN_LINE_PX,
            )
    for x, y, c in kpts:
        if c >= KPT_CONF_MIN:
            cv2.circle(img, (int(x), int(y)), 2, POSE_SKELETON_BGR, -1)


def draw_box_only(img: np.ndarray, x1, y1, x2, y2, pid: int, color: tuple[int, int, int]) -> None:
    """Caja de color CON el ID (chiquito, sin el estado/acción — eso va en el
    panel lateral) para poder saber cuál caja es cuál persona del panel."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = pid_label(pid)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    (tw, th), baseline = cv2.getTextSize(label, font, fs, 1)
    ty = max(th + 4, y1 - 2)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + baseline), color, -1)
    cv2.putText(img, label, (x1 + 3, ty - 1), font, fs, (255, 255, 255), 1, cv2.LINE_AA)


_PANEL_BG_BGR = (26, 22, 20)
_PANEL_BORDER_BGR = (90, 82, 76)
_TARGET_BADGE_BGR = (0, 255, 255)  # amarillo -- mismo acento que el resalte de caja


def draw_side_panel(
    img: np.ndarray,
    entries: list[tuple[int, str, tuple[int, int, int], bool]],
    *,
    side: str = "right",
) -> None:
    """Panel lateral con una fila por persona ("ID{n}: {estado/acción}"), en
    vez de una etiqueta encima de cada caja (que ensucia la imagen). Cada fila
    lleva una barra de acento a la izquierda con el color de su estado (para
    no depender solo del color del texto) y, si `is_target`, una insignia
    "TARGET" aparte en vez de texto pegado a la descripción."""
    if not entries:
        return
    h_img, w_img = img.shape[:2]
    s = ui_scale(h_img, w_img, banner=True)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.4, min(0.62, 0.55 * s))
    thick = 1
    pad = int(max(8, 9 * s))
    line_gap = int(max(5, 6 * s))
    accent_w = int(max(3, 4 * s))
    badge_gap = int(max(6, 7 * s))
    badge_pad = int(max(3, 4 * s))

    lines = [f"ID{pid}: {text}" for pid, text, _color, _t in entries]
    metrics = [cv2.getTextSize(ln, font, fs, thick) for ln in lines]
    badge_text = "TARGET"
    (badge_w, badge_h), badge_bl = cv2.getTextSize(badge_text, font, fs * 0.8, thick)
    badge_box_w = badge_w + 2 * badge_pad

    row_w = [
        tw + (badge_gap + badge_box_w if entry[3] else 0)
        for ((tw, _th), _bl), entry in zip(metrics, entries)
    ]
    panel_w = accent_w + max(row_w) + 2 * pad
    panel_h = sum(th + bl + line_gap for (tw, th), bl in metrics) - line_gap + 2 * pad
    margin = int(max(8, 10 * s))
    x0 = w_img - panel_w - margin if side == "right" else margin
    y0 = margin
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _PANEL_BG_BGR, -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), _PANEL_BORDER_BGR, 1)

    cy = y0 + pad
    for line, ((tw, th), bl), (_pid, _text, color, is_target) in zip(lines, metrics, entries):
        row_top, row_bot = cy - th, cy + bl
        cv2.rectangle(img, (x0, row_top), (x0 + accent_w, row_bot), color, -1)
        cy += th
        tx = x0 + accent_w + pad
        cv2.putText(img, line, (tx, cy), font, fs, color, thick, cv2.LINE_AA)
        if is_target:
            bx0 = tx + tw + badge_gap
            by1 = cy + badge_bl
            by0 = by1 - badge_h - badge_bl - 2 * badge_pad
            cv2.rectangle(img, (bx0, by0), (bx0 + badge_box_w, by1), _TARGET_BADGE_BGR, -1)
            cv2.putText(
                img, badge_text, (bx0 + badge_pad, by1 - badge_pad),
                font, fs * 0.8, (20, 20, 20), thick, cv2.LINE_AA,
            )
        cy += bl + line_gap


def annotate_for_vlm_pose(
    frame_bgr: np.ndarray, dets: list[dict], moving_pids: set[int]
) -> np.ndarray:
    """Frame para el VLM: esqueleto COCO-17 completo + caja delgada, azul (en
    movimiento) / roja (parado), solo con el ID (sin texto de estado) — el
    color YA es la pista de movimiento."""
    out = frame_bgr.copy()
    for d in sort_dets_left_right(dets):
        draw_thin_pose_box(
            out, d["x1"], d["y1"], d["x2"], d["y2"], pid_label(d["pid"]),
            d.get("kpts"), moving=d["pid"] in moving_pids,
        )
    return out


def vlm_prompt_with_motion_hint(dets: list[dict], *, video: bool = False) -> str:
    """Igual que pipeline.vlm_prompt pero añade una nota sobre el color de caja
    (pista de un sensor de movimiento independiente, no del propio VLM)."""
    base = vlm_prompt(dets, video=video)
    if not dets:
        return base
    return base + " " + MOTION_COLOR_HINT


# —— Mirando a cámara (para ATTENTIVE) ————————————————————————————
# EXPERIMENTAL, heurística simple sin calibrar: si los dos ojos Y la nariz
# están visibles con buena confianza, YOLO-pose casi siempre es porque ve la
# cara de frente (un perfil suele perder la confianza del ojo/oreja lejanos).
# No distingue "mirando AL ROBOT" de "mirando a otra cámara/punto" — es una
# aproximación de "cara de frente", no de dirección de mirada real.
FACE_CONF_MIN = 0.5


def is_facing_camera(kpts: np.ndarray) -> bool:
    nose, l_eye, r_eye = kpts[NOSE], kpts[L_EYE], kpts[R_EYE]
    return bool(nose[2] >= FACE_CONF_MIN and l_eye[2] >= FACE_CONF_MIN and r_eye[2] >= FACE_CONF_MIN)


FACING_MIN_SEEN = 4
FACING_TRUE_RATIO = 0.5
FACING_FALSE_RATIO = 0.2


def chunk_facing_camera_by_pid(
    buffer_dets: list[list[dict]], person_ids: list[int]
) -> dict[int, bool]:
    """True si la persona mira de frente a la cámara en la MAYORÍA de los
    frames del trozo; False si claramente NO mira de frente en casi ninguno;
    ausente (sin entrada) si no hay suficiente señal de pose para decidir en
    cualquier sentido — esa distinción importa porque una entrada False se usa
    para BAJAR un ATTENTIVE que el VLM puso por su cuenta (ver
    upgrade_attentive_by_gaze, en este mismo módulo), y eso solo es seguro con
    evidencia real de que NO mira, no con simple falta de datos."""
    out: dict[int, bool] = {}
    for pid in person_ids:
        seen = 0
        facing = 0
        for row in buffer_dets:
            for d in row:
                if d.get("pid") == pid and "kpts" in d:
                    seen += 1
                    if is_facing_camera(d["kpts"]):
                        facing += 1
                    break
        if seen < FACING_MIN_SEEN:
            continue
        ratio = facing / seen
        if ratio >= FACING_TRUE_RATIO:
            out[pid] = True
        elif ratio <= FACING_FALSE_RATIO:
            out[pid] = False
    return out


# —— Selección de a quién puede acercarse el robot ("-Target-") ————————
# Prioridad: ATTENTIVE (de pie, libre, mirando) > AVAILABLE (de pie, libre) >
# el resto (ENGAGED/BUSY/MOVING no son buenos candidatos: están ocupados,
# hablando con alguien más, o en movimiento). Entre empates, el más cercano
# a la cámara (caja más grande).
_TARGET_PRIORITY = {SOCIAL_ATTENTIVE: 0, SOCIAL_AVAILABLE: 1}


def pick_interaction_target(
    dets: list[dict], social_states: dict[int, str]
) -> int | None:
    """pid del mejor candidato para que el robot inicie interacción, o None
    si nadie en el frame está en un estado apto (todos ENGAGED/BUSY/MOVING)."""
    candidates = []
    for d in dets:
        pid = d["pid"]
        state = social_states.get(pid)
        if state not in _TARGET_PRIORITY:
            continue
        area = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
        candidates.append((_TARGET_PRIORITY[state], -area, pid))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


class TargetStability:
    """Mantiene el mismo target de interacción entre frames en vez de
    recalcularlo desde cero cada vez con pick_interaction_target.

    Sin esto, dos personas casi empatadas (misma prioridad de estado, área de
    caja parecida) pueden intercambiarse el "-Target-" de un frame a otro por
    puro jitter de detección — el robot solo puede acercarse a UN target real
    a la vez, no a uno que cambia constantemente. Se conserva el target
    actual mientras siga siendo elegible (ATTENTIVE/AVAILABLE); solo se
    recalcula cuando deja de serlo (sale de cuadro, pasa a ENGAGED/BUSY/
    MOVING, etc.)."""

    def __init__(self) -> None:
        self._current: int | None = None

    def update(self, dets: list[dict], social_states: dict[int, str]) -> int | None:
        candidates = []
        eligible: set[int] = set()
        for d in dets:
            pid = d["pid"]
            state = social_states.get(pid)
            if state not in _TARGET_PRIORITY:
                continue
            eligible.add(pid)
            area = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
            candidates.append((_TARGET_PRIORITY[state], -area, pid))
        if self._current is not None and self._current in eligible:
            return self._current
        if not candidates:
            self._current = None
            return None
        candidates.sort()
        self._current = candidates[0][2]
        return self._current


def refine_labels_pose(
    actions: dict[int, str],
    social: dict[int, str],
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    frame_size: tuple[int, int],
    *,
    moving_pids_hint: set[int] | None = None,
    world_xy: dict[int, tuple[float, float]] | None = None,
    clusters: dict[int, int] | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Combina tres señales de movimiento en vez de una sola:
    1. El VLM ya vio la pose+color como pista y decidió con eso.
    2. El respaldo determinístico por cinemática de caja de vlm_hri.motion
       (bbox-center, recalibrado).
    3. `moving_pids_hint`: marcha por piernas + cambio de profundidad de la
       caja + histéresis temporal (este módulo), calculado en Fase 1.
    Si CUALQUIERA de las tres dice "se mueve", se respeta — medido: dejar
    todo en manos de una sola señal (el VLM+pista) perdía el caso difícil de
    caminata de frente a cámara.

    ANTES de eso, se corrige el sesgo contrario: el VLM a veces le pega
    "walking" a TODO el grupo a la vez sin que nadie se haya movido de verdad
    (medido: 4/5 personas "walking" con 3-36px de desplazamiento total en 60
    frames — nada). Si eso pasa, solo se respeta a quien SÍ tiene evidencia
    cinemática independiente."""
    # El respaldo por cinemática de caja de vlm_hri.motion mide posición ABSOLUTA
    # en la imagen — si la cámara se mueve un poco (temblor, paneo), todas las
    # cajas se desplazan igual y parece que todo el mundo camina. Se le pasa
    # una copia con ese movimiento común (mediana entre personas) ya restado.
    compensated = compensate_camera_motion(buffer_dets)
    bbox_confirmed = set(
        chunk_motion_by_pid(compensated, person_ids, frame_size).keys()
    )
    confirmed_moving = (moving_pids_hint or set()) | bbox_confirmed
    actions, social = debias_group_walking(actions, social, person_ids, confirmed_moving)
    actions, social = require_moving_evidence(actions, social, confirmed_moving)

    if moving_pids_hint:
        for pid in moving_pids_hint:
            if pid not in actions:
                continue
            act = normalize_action(actions[pid])
            tr = _action_traits(act.lower())
            if tr.get("sitting") and not tr.get("standing"):
                # Sentado y caminando son físicamente incompatibles. Si el VLM
                # ya describió a esta persona como sentada, no se le fuerza
                # "walking" encima aunque la cinemática (gait/profundidad) lo
                # sugiera — es mucho más probable que sea ruido de esa señal
                # (gesticular, inclinarse a comer/alcanzar algo) que alguien
                # caminando sentado. Mismo criterio que ya usa
                # vlm_hri.motion.apply_chunk_kinematic_hints (clearly_sitting)
                # para el mismo caso en el pipeline sin pose — aquí faltaba.
                continue
            if not tr.get("walking"):
                extras = _action_secondary_parts(tr)
                actions[pid] = " and ".join(["walking"] + extras) if extras else "walking"

    # Piernas visibles: si NO lo están (persona muy cerca de la cámara, solo
    # torso/cara en el frame), refine_chunk_labels no debe adivinar postura —
    # se confía en lo que el VLM describió (ver enrich_action_posture).
    legs_visible = chunk_legs_visible_by_pid(compensated, person_ids)
    actions, social = refine_chunk_labels(
        actions, social, compensated, person_ids, frame_size,
        moving_pids=confirmed_moving, legs_visible=legs_visible,
    )
    social = check_engaged_proximity(actions, social, person_ids, world_xy, clusters)
    return actions, social


def upgrade_attentive_by_gaze(
    social: dict[int, str], buffer_dets: list[list[dict]], person_ids: list[int]
) -> dict[int, str]:
    """Corrige ATTENTIVE con la mirada real (pose), no el juicio suelto del
    VLM sobre "orientado a cámara":
    - AVAILABLE → ATTENTIVE si la pose SÍ confirma que mira de frente la
      mayoría del trozo.
    - ATTENTIVE → AVAILABLE si la pose NO lo confirma positivamente — ya sea
      porque confirma que mira a otro lado, o porque no hay suficientes
      datos de pose ese trozo para saberlo. Modo rígido (default): ATTENTIVE
      exige evidencia POSITIVA de mirada, no basta con que el VLM lo diga y
      nadie lo desmienta (medido: el VLM marca ATTENTIVE con la persona
      mirando claramente a otro lado, y muchas veces la pose no tiene
      suficiente confianza para "desmentirlo" con fuerza — sin exigir
      confirmación positiva, ese ATTENTIVE quedaba sin corregir).
    No toca ENGAGED/BUSY/MOVING, que ya son más específicos que ATTENTIVE.
    Desactivable con STRICT_ATTENTIVE_GAZE=0 (vuelve a solo desmentir con
    evidencia fuerte de NO mirar, dejando sin datos = sin tocar)."""
    facing = chunk_facing_camera_by_pid(buffer_dets, person_ids)
    strict = os.environ.get("STRICT_ATTENTIVE_GAZE", "1") == "1"
    out = dict(social)
    for pid in person_ids:
        cur = out.get(pid)
        is_facing = facing.get(pid)
        if is_facing is True and cur == SOCIAL_AVAILABLE:
            out[pid] = SOCIAL_ATTENTIVE
        elif cur == SOCIAL_ATTENTIVE and (
            is_facing is False or (strict and is_facing is None)
        ):
            out[pid] = SOCIAL_AVAILABLE
    return out
