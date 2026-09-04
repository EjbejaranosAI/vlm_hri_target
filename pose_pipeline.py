"""Variante del pipeline con detector de POSE (yolo26n-pose.pt) en vez de solo
detección (yolo11n.pt). Añade una señal de movimiento independiente del VLM
—marcha real de piernas, no el centro de la caja— y la pasa al VLM como una
PISTA visual (caja azul=en movimiento, roja=parada, sin texto de estado) para
que el propio VLM decida la acción/intención final combinando esa pista con
lo que ve.

Deliberadamente en un módulo aparte (junto con run_video_pose.py y
main_pose.py) para no tocar pipeline.py / run_video.py / main.py. Reutiliza
de `pipeline.py` todo lo que no cambia: carga/inferencia del VLM, tracking,
parseo de respuesta, dibujo del video final. Calibrado en pose_gait_test.py.
"""

from __future__ import annotations

import os

import numpy as np
import cv2

import pipeline as P
import prompts as VLM_PROMPTS

POSE_YOLO_WEIGHTS = "yolo26n-pose.pt"

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

# Calibrado con datos reales (ver pose_gait_test.py): control negativo
# (gente de pie confirmada) vs. caminata lateral confirmada → 5°/2 cruces da
# 74% de recall real con ~7% de falsos positivos (12°/2 daba solo 25% recall).
MIN_AMPLITUDE_DEG = 5.0
MIN_CROSSINGS_PER_CHUNK = 2

COLOR_MOVING_BGR = (255, 0, 0)  # azul
COLOR_STANDING_BGR = (0, 0, 255)  # rojo
THIN_LINE_PX = 1


def detect_people_pose(yolo_pose, bgr: np.ndarray) -> list[dict]:
    """Como pipeline.detect_people pero cada detección incluye "kpts" (17,3)."""
    res = yolo_pose.predict(
        bgr,
        classes=[0],
        verbose=False,
        device=0,
        conf=P.YOLO_CONF,
        iou=P.YOLO_IOU,
        max_det=P.YOLO_MAX_DET,
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
    return P.nms_boxes(dets, P.YOLO_POST_NMS_IOU)


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
    return P.nms_boxes(dets, P.YOLO_POST_NMS_IOU)


def detect_people_batch(yolo, frames: list[np.ndarray]) -> list[list[dict]]:
    """Como pipeline.detect_people pero para VARIOS frames en una sola llamada
    a predict() — medido: ~6ms/frame uno a uno vs ~2ms/frame en lotes de 8+
    (la GPU está infrautilizada con lotes de 1 en modelos tan chicos)."""
    if not frames:
        return []
    results = yolo.predict(
        frames, classes=[0], verbose=False, device=0,
        conf=P.YOLO_CONF, iou=P.YOLO_IOU, max_det=P.YOLO_MAX_DET,
    )
    return [_dets_from_result(r, f, with_kpts=False) for r, f in zip(results, frames)]


def detect_people_pose_batch(yolo_pose, frames: list[np.ndarray]) -> list[list[dict]]:
    """Como detect_people_pose pero para VARIOS frames en una sola llamada."""
    if not frames:
        return []
    results = yolo_pose.predict(
        frames, classes=[0], verbose=False, device=0,
        conf=P.YOLO_CONF, iou=P.YOLO_IOU, max_det=P.YOLO_MAX_DET,
    )
    return [_dets_from_result(r, f, with_kpts=True) for r, f in zip(results, frames)]


# IoU mínimo para aceptar un emparejamiento pose↔caja trackeada; por debajo de
# esto se descarta (mejor sin keypoints ese frame que asignarlos a la persona
# equivocada).
POSE_MATCH_MIN_IOU = 0.3


def attach_pose_keypoints(tracked_dets: list[dict], pose_dets: list[dict]) -> None:
    """Empareja por IoU cada detección de POSE con su caja ya trackeada (de
    yolo11n, vía pipeline.detect_people + track_detections) y le añade "kpts".

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
            iou = P.box_iou(pd_, td)
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
    izquierda vs. derecha; balancearse de pie no la tiene. Ver
    pose_gait_test.py para la calibración empírica de los umbrales.
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
        if len(diffs) < 4:
            continue
        arr = np.array(diffs)
        amplitude = float(arr.max() - arr.min())
        signs = np.sign(arr - arr.mean())
        crossings = int(np.sum(signs[1:] != signs[:-1]))
        if amplitude >= MIN_AMPLITUDE_DEG and crossings >= MIN_CROSSINGS_PER_CHUNK:
            out[pid] = True
    return out


# EXPERIMENTAL — sin calibrar con datos reales todavía (a diferencia de
# MIN_AMPLITUDE_DEG/MIN_CROSSINGS_PER_CHUNK, que sí se midieron). Complementa
# la tijera de piernas: alguien caminando derecho hacia/desde la cámara apenas
# balancea las piernas lateralmente (por eso a veces no se detectaba), pero su
# caja SÍ crece o encoge de forma sostenida. Si en la práctica da muchos
# falsos positivos (p. ej. alguien agachándose), avisar para recalibrar igual
# que se hizo con la marcha.
DEPTH_MOTION_REL_CHANGE = 0.08


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
        if len(heights) < 6:
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
    deja que el resto del pipeline le asigne su postura real."""
    n = len(person_ids)
    if n < 3:
        return actions, social

    def is_walking(pid: int) -> bool:
        return P._action_traits(P.normalize_action(actions.get(pid, "")).lower())["walking"]

    walkers = [p for p in person_ids if is_walking(p)]
    if len(walkers) < n - 1:
        return actions, social

    out_a, out_s = dict(actions), dict(social)
    for pid in walkers:
        if pid in confirmed_moving:
            continue
        out_a[pid] = "unknown"
        out_s[pid] = P.SOCIAL_UNKNOWN
    return out_a, out_s


# Cuántos trozos (~1s cada uno) se mantiene "moviéndose" tras la última
# evidencia real, para no perderlo en trozos donde la señal se ahoga en ruido
# (p. ej. alguien lejos, caja chica) — sin esto, un par de trozos "ciegos"
# de por medio ya tumbaban el veredicto final por mayoría de votos.
HYSTERESIS_CHUNKS = 3


class MotionHysteresis:
    """Estado de "moviéndose" por persona a lo largo del video: se enciende
    con evidencia real (marcha o cambio de profundidad) y se mantiene unos
    trozos después, apagándose solo tras varios trozos seguidos sin ninguna."""

    def __init__(self, hold_chunks: int = HYSTERESIS_CHUNKS) -> None:
        self.hold_chunks = hold_chunks
        self._last_moving_ci: dict[int, int] = {}

    def update(self, ci: int, raw_moving_pids: set[int], present_pids: set[int]) -> set[int]:
        for pid in raw_moving_pids:
            self._last_moving_ci[pid] = ci
        out: set[int] = set()
        for pid in present_pids:
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


def draw_box_only(img: np.ndarray, x1, y1, x2, y2, pid: int, color: tuple[int, int, int]) -> None:
    """Caja de color CON el ID (chiquito, sin el estado/acción — eso va en el
    panel lateral) para poder saber cuál caja es cuál persona del panel."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = P.pid_label(pid)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    (tw, th), baseline = cv2.getTextSize(label, font, fs, 1)
    ty = max(th + 4, y1 - 2)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + baseline), color, -1)
    cv2.putText(img, label, (x1 + 3, ty - 1), font, fs, (255, 255, 255), 1, cv2.LINE_AA)


def draw_side_panel(
    img: np.ndarray,
    entries: list[tuple[int, str, tuple[int, int, int]]],
    *,
    side: str = "right",
) -> None:
    """Panel lateral con "ID{n}: {estado/acción}" por persona, uno por línea,
    en vez de una etiqueta encima de cada caja (que ensucia la imagen)."""
    if not entries:
        return
    h_img, w_img = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    thick = 1
    pad = 10
    line_gap = 6
    lines = [f"ID{pid}: {text}" for pid, text, _ in entries]
    metrics = [cv2.getTextSize(ln, font, fs, thick) for ln in lines]
    panel_w = max(tw for (tw, _th), _bl in metrics) + 2 * pad
    panel_h = sum(th + bl + line_gap for (tw, th), bl in metrics) - line_gap + 2 * pad
    margin = 10
    x0 = w_img - panel_w - margin if side == "right" else margin
    y0 = margin
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), 1)
    cy = y0 + pad
    for line, ((tw, th), bl), (_pid, _text, color) in zip(lines, metrics, entries):
        cy += th
        cv2.putText(img, line, (x0 + pad, cy), font, fs, color, thick, cv2.LINE_AA)
        cy += bl + line_gap


def annotate_for_vlm_pose(
    frame_bgr: np.ndarray, dets: list[dict], moving_pids: set[int]
) -> np.ndarray:
    """Frame para el VLM: esqueleto COCO-17 completo + caja delgada, azul (en
    movimiento) / roja (parado), solo con el ID (sin texto de estado) — el
    color YA es la pista de movimiento."""
    out = frame_bgr.copy()
    for d in P.sort_dets_left_right(dets):
        draw_thin_pose_box(
            out, d["x1"], d["y1"], d["x2"], d["y2"], P.pid_label(d["pid"]),
            d.get("kpts"), moving=d["pid"] in moving_pids,
        )
    return out


def vlm_prompt_with_motion_hint(dets: list[dict], *, video: bool = False) -> str:
    """Igual que pipeline.vlm_prompt pero añade una nota sobre el color de caja
    (pista de un sensor de movimiento independiente, no del propio VLM)."""
    base = P.vlm_prompt(dets, video=video)
    if not dets:
        return base
    return base + " " + VLM_PROMPTS.MOTION_COLOR_HINT


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


def chunk_facing_camera_by_pid(
    buffer_dets: list[list[dict]], person_ids: list[int]
) -> dict[int, bool]:
    """True si la persona mira de frente a la cámara en la MAYORÍA de los
    frames del trozo (no solo un frame suelto, para evitar parpadeos)."""
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
        if seen >= 4 and facing / seen >= 0.5:
            out[pid] = True
    return out


# —— Selección de a quién puede acercarse el robot ("-Target-") ————————
# Prioridad: ATTENTIVE (de pie, libre, mirando) > AVAILABLE (de pie, libre) >
# el resto (ENGAGED/BUSY/MOVING no son buenos candidatos: están ocupados,
# hablando con alguien más, o en movimiento). Entre empates, el más cercano
# a la cámara (caja más grande).
_TARGET_PRIORITY = {P.SOCIAL_ATTENTIVE: 0, P.SOCIAL_AVAILABLE: 1}


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
