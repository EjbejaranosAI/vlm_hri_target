"""Refinamiento de etiquetas por trozo: cinemática de caja, postura, debias de grupo.

Combina la salida del VLM con señales geométricas del propio trozo de
detecciones (desplazamiento del centro de caja, forma/aspecto) para corregir
sesgos conocidos del VLM (p. ej. etiquetar "talking" a todo el grupo).
"""

from __future__ import annotations

import os
import re

from .actions import _action_secondary_parts, _action_traits, normalize_action
from .social_state import (
    SOCIAL_ATTENTIVE,
    SOCIAL_AVAILABLE,
    SOCIAL_ENGAGED,
    SOCIAL_UNKNOWN,
    normalize_social_state,
    resolve_social_states,
)


def chunk_motion_enabled() -> bool:
    return os.environ.get("CHUNK_MOTION_HINT", "1") == "1"


def chunk_motion_by_pid(
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    frame_size: tuple[int, int],
) -> dict[int, str]:
    """Desplazamiento del centro de la caja en el trozo → walking.

    Umbrales calibrados empíricamente sobre CSVs de detección reales (videos
    con caminata real vs. videos de gente quieta): a ~25 fps una persona
    caminando desplaza el centro de su caja solo ~1-3 px POR FRAME (no ~40+ px
    como asumían los umbrales anteriores, que exigían ~0.14·alto por frame y
    nunca se cumplían — por eso "walking" casi nunca se detectaba por
    cinemática y la gente en movimiento quedaba como AVAILABLE). Aquí solo se
    exige desplazamiento total del trozo (relativo al alto de la caja) y
    direccionalidad neta/recorrido para filtrar el zigzag de tracking de
    alguien parado.
    """
    if not buffer_dets or not person_ids:
        return {}
    total_floor_px = float(os.environ.get("CHUNK_MOTION_PX", "6"))
    rel_total = float(os.environ.get("CHUNK_MOTION_TOTAL_REL", "0.06"))
    # Persona parada con jitter de tracking (re-detección, cambio de postura,
    # oclusión parcial) acumula desplazamiento en zigzag: mucho "path" pero
    # poco avance neto. Caminar de verdad es direccional: el desplazamiento
    # neto (inicio→fin) se acerca al camino recorrido.
    #
    # Límite conocido (medido, no solo teórico): un balanceo de peso MUY lento
    # de pie (p. ej. alguien hablando/sonriendo a cámara) puede verse igual de
    # "direccional" que caminar si su período de vaivén es más largo que el
    # propio clip — solo se observa un tramo de la oscilación, nunca el
    # regreso. Subir este umbral NO lo arregla (se probó empíricamente: subirlo
    # perdía caminatas reales sin quitar ese falso positivo, porque el vaivén
    # medido llegaba a la misma direccionalidad que un paso lento real).
    # Arreglarlo de raíz requiere una ventana temporal más larga que un trozo
    # (para ver si la persona vuelve cerca de su posición de origen) o una
    # señal distinta a la posición del centro de la caja (p. ej. pose/marcha).
    min_directionality = float(os.environ.get("CHUNK_MOTION_DIRECTIONALITY", "0.35"))
    out: dict[int, str] = {}
    for pid in person_ids:
        pts: list[tuple[float, float]] = []
        heights: list[float] = []
        for row in buffer_dets:
            for d in row:
                if d["pid"] == pid:
                    pts.append((0.5 * (d["x1"] + d["x2"]), 0.5 * (d["y1"] + d["y2"])))
                    bh = d["y2"] - d["y1"]
                    if bh > 24:
                        heights.append(bh)
                    break
        if len(pts) < 4:
            continue
        dist = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            dist += (dx * dx + dy * dy) ** 0.5
        avg_h = sum(heights) / len(heights) if heights else 80.0
        total_thr = max(total_floor_px, rel_total * avg_h)
        net_dx = pts[-1][0] - pts[0][0]
        net_dy = pts[-1][1] - pts[0][1]
        net_disp = (net_dx * net_dx + net_dy * net_dy) ** 0.5
        directionality = net_disp / dist if dist > 1e-6 else 0.0
        if dist >= total_thr and directionality >= min_directionality:
            out[pid] = "walking"
    return out


def chunk_posture_hints(
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    motion: dict[int, str],
) -> dict[int, str]:
    """Silueta ancha + poco desplazamiento → probablemente sentado."""
    if os.environ.get("CHUNK_POSTURE_HINT", "0") != "1":
        return {}
    aspect_min = float(os.environ.get("CHUNK_SIT_ASPECT_MIN", "0.88"))
    vert_max = float(os.environ.get("CHUNK_SIT_VERT_REL", "0.05"))
    out: dict[int, str] = {}
    for row in buffer_dets:
        if not row:
            continue
        h_img = max(d["y2"] for d in row) - min(d["y1"] for d in row)
        if h_img < 48:
            h_img = 480.0
        break
    else:
        h_img = 480.0
    for pid in person_ids:
        if motion.get(pid) == "walking":
            continue
        aspects: list[float] = []
        ys: list[float] = []
        for row in buffer_dets:
            for d in row:
                if d["pid"] == pid:
                    bh = d["y2"] - d["y1"]
                    bw = d["x2"] - d["x1"]
                    if bh > 24:
                        aspects.append(bw / bh)
                        ys.append(0.5 * (d["y1"] + d["y2"]))
                    break
        if not aspects or sum(aspects) / len(aspects) < aspect_min:
            continue
        vert_span = (max(ys) - min(ys)) if len(ys) >= 2 else 0.0
        if vert_span <= vert_max * h_img:
            out[pid] = "sitting"
    return out


def apply_chunk_kinematic_hints(
    actions: dict[int, str],
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    frame_size: tuple[int, int],
    *,
    moving_pids: set[int] | None = None,
) -> dict[int, str]:
    """Combina VLM + movimiento de cajas en el trozo (walking/sitting).

    El desplazamiento real de la caja es la señal más fiable de que alguien
    camina — prima sobre lo que haya dicho el VLM (p. ej. "talking"), salvo
    que el VLM afirme sentado (incompatible con caminar). Cualquier intención
    secundaria detectada (hablar, sonreír, teléfono) se conserva anexada en
    vez de descartarse, para no perder señales de posible interacción.

    `moving_pids`, si se pasa, reemplaza el cálculo de bbox-motion de aquí:
    el pipeline de pose (run_video_pose.refine_labels_pose) ya resuelve
    "quién se mueve de verdad" combinando marcha real + profundidad + bbox +
    histéresis + debias de grupo ANTES de llamar a esta función — recalcular
    bbox-motion otra vez aquí (una señal más cruda, pensada para el pipeline
    SIN pose que no tiene nada mejor) podía pisar esa decisión ya tomada. Sin
    `moving_pids` (pipeline sin pose), el comportamiento es el de siempre."""
    if not buffer_dets:
        return actions
    if moving_pids is not None:
        motion = {pid: "walking" for pid in moving_pids}
    elif chunk_motion_enabled():
        motion = chunk_motion_by_pid(buffer_dets, person_ids, frame_size)
    else:
        return actions
    posture = chunk_posture_hints(buffer_dets, person_ids, motion)
    out: dict[int, str] = {}
    for pid in person_ids:
        raw = actions.get(pid, "unknown")
        act = normalize_action(raw)
        tr = _action_traits(act.lower())
        extras = _action_secondary_parts(tr)
        has_semantic = tr["talking"] or tr["smiling"] or tr["phone"]
        clearly_sitting = tr["sitting"] and not tr["standing"]
        if motion.get(pid) == "walking" and not clearly_sitting:
            # Antes esto solo conservaba talking/smiling/phone (extras) y
            # descartaba cualquier otra descripción del VLM (p. ej.
            # "greeting person", "gesturing") — con el prompt ahora en texto
            # libre, eso perdía la interacción real y forzaba MOVING sobre
            # cualquier cosa. Se conserva el texto tal cual (solo se
            # reemplaza la postura de pie por "walking", que ya la implica).
            if act.lower() in ("unknown", "standing", "stand", "sitting", "sit", "", "walking", "walk"):
                out[pid] = "walking"
            else:
                stripped = re.sub(r"^(standing|stand)\s+and\s+", "", act, flags=re.IGNORECASE)
                out[pid] = f"walking and {stripped}"
            continue
        if (
            posture.get(pid) == "sitting"
            and not tr["walking"]
            and not has_semantic
            and act in ("unknown", "standing", "sitting", "")
        ):
            base = "sitting"
            out[pid] = " and ".join([base] + extras) if extras else base
            continue
        out[pid] = act
    for pid, act in actions.items():
        if pid not in out:
            out[pid] = normalize_action(act)
    return out


def group_talk_debias_enabled() -> bool:
    return os.environ.get("GROUP_TALK_DEBIAS", "1") == "1"


def infer_posture_light(
    buffer_dets: list[list[dict]], person_ids: list[int]
) -> dict[int, str]:
    """standing | sitting a partir de forma de caja (siempre, para enriquecer etiquetas)."""
    aspects: dict[int, list[float]] = {p: [] for p in person_ids}
    heights: dict[int, list[float]] = {p: [] for p in person_ids}
    for row in buffer_dets:
        for d in row:
            pid = d.get("pid")
            if pid not in aspects:
                continue
            bh = d["y2"] - d["y1"]
            bw = d["x2"] - d["x1"]
            if bh > 20:
                aspects[pid].append(bw / bh)
                heights[pid].append(bh)
    med_h = 0.0
    all_h = [h for hs in heights.values() for h in hs]
    if all_h:
        med_h = sorted(all_h)[len(all_h) // 2]
    sit_aspect = float(os.environ.get("POSTURE_SIT_ASPECT", "0.80"))
    sit_h_frac = float(os.environ.get("POSTURE_SIT_HEIGHT_FRAC", "0.90"))
    out: dict[int, str] = {}
    for pid in person_ids:
        asp_vals = aspects.get(pid) or []
        h_vals = heights.get(pid) or []
        if not asp_vals:
            out[pid] = "standing"
            continue
        asp = sum(asp_vals) / len(asp_vals)
        h = sum(h_vals) / len(h_vals)
        if med_h > 0 and asp >= sit_aspect and h <= sit_h_frac * med_h:
            out[pid] = "sitting"
        else:
            out[pid] = "standing"
    return out


def enrich_action_posture(
    actions: dict[int, str],
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    *,
    legs_visible: dict[int, bool] | None = None,
) -> dict[int, str]:
    """Antepone la postura ('standing'/'sitting') SOLO cuando el VLM no la
    mencionó (p. ej. dijo solo 'talking' o 'eating' -> 'standing and eating').

    Si el texto YA trae su propia postura (p. ej. 'sit and eat', 'standing
    and talking' — lo normal ahora que el prompt pide describir la actividad
    real), se deja tal cual. ANTES esto se reconstruía siempre desde un set
    cerrado de palabras clave (walk/run/sit/stand/talk/smile/phone), y
    cualquier actividad fuera de ese set se perdía en silencio (medido:
    "eating" quedaba reducido a solo "standing", con la persona sentada
    comiendo reportada como si solo estuviera de pie).

    `legs_visible` (pipeline de pose, ver pose_pipeline.chunk_legs_visible_by_pid):
    si dice explícitamente False para un pid (piernas fuera de cuadro, p. ej.
    muy cerca de la cámara), NO se antepone ninguna postura adivinada — se
    confía en el texto del VLM tal cual, siguiendo la misma lógica que el
    prompt (no adivinar lo que no se puede ver). Ausente del dict (sin pose,
    o sin dato de piernas ese trozo) cae al comportamiento de siempre
    (adivinar por forma de caja vía infer_posture_light)."""
    if not buffer_dets:
        return actions
    postures = infer_posture_light(buffer_dets, person_ids)
    out: dict[int, str] = {}
    for pid in person_ids:
        act = normalize_action(actions.get(pid, "unknown"))
        if act == "unknown":
            out[pid] = act
            continue
        tr = _action_traits(act.lower())
        if tr["walking"] or tr["running"] or tr["sitting"] or tr["standing"]:
            out[pid] = act
            continue
        if legs_visible is not None and legs_visible.get(pid) is False:
            out[pid] = act
            continue
        pos = postures.get(pid, "standing")
        out[pid] = f"{pos} and {act}"
    for pid, act in actions.items():
        if pid not in out:
            out[pid] = normalize_action(act)
    return out


def debias_group_talking(
    actions: dict[int, str],
    social: dict[int, str],
    person_ids: list[int],
    buffer_dets: list[list[dict]],
) -> tuple[dict[int, str], dict[int, str]]:
    """
    El VLM suele poner talk a todo el grupo. Quita talk a oyentes / escena completa.

    Solo se aplica con 3+ personas: con exactamente 2, "ambas hablando" suele
    ser una conversación real (el caso más común en los videos de prueba) y
    no un sesgo del VLM, así que no se debe rebajar de ENGAGED a ATTENTIVE.
    Tampoco se toca nunca a quien el movimiento real ya confirmó caminando
    (esa señal manda sobre la postura inferida aquí).
    """
    if not group_talk_debias_enabled() or len(person_ids) < 3 or not buffer_dets:
        return actions, social
    out_a = {pid: normalize_action(actions.get(pid, "unknown")) for pid in person_ids}
    out_s = dict(social)
    postures = infer_posture_light(buffer_dets, person_ids)
    n = len(person_ids)

    def is_talking(pid: int) -> bool:
        return _action_traits(out_a.get(pid, "").lower())["talking"]

    def is_walking(pid: int) -> bool:
        return _action_traits(out_a.get(pid, "").lower())["walking"]

    talkers = [p for p in person_ids if is_talking(p)]
    if len(talkers) < 2:
        return out_a, out_s

    # Sentado en escena con muchos talk → oyente, no hablante
    if len(talkers) >= n - 1:
        for pid in talkers:
            if is_walking(pid):
                continue
            if postures.get(pid) == "sitting":
                out_a[pid] = "sitting"
                if normalize_social_state(str(out_s.get(pid, ""))) == SOCIAL_ENGAGED:
                    out_s[pid] = SOCIAL_ATTENTIVE

    talkers = [p for p in person_ids if is_talking(p)]
    if len(talkers) >= n:
        for pid in person_ids:
            if is_walking(pid):
                continue
            out_a[pid] = postures.get(pid, "standing")
            soc = normalize_social_state(str(out_s.get(pid, "")))
            if soc in (SOCIAL_ENGAGED, SOCIAL_UNKNOWN):
                out_s[pid] = SOCIAL_ATTENTIVE
    elif len(talkers) >= n - 1:
        for pid in talkers:
            if is_walking(pid):
                continue
            soc = normalize_social_state(str(out_s.get(pid, "")))
            if soc in (SOCIAL_AVAILABLE, SOCIAL_ATTENTIVE):
                out_a[pid] = postures.get(pid, "standing")

    for pid in actions:
        if pid not in out_a:
            out_a[pid] = normalize_action(actions[pid])
    return out_a, out_s


def refine_chunk_labels(
    actions: dict[int, str],
    social: dict[int, str],
    buffer_dets: list[list[dict]],
    person_ids: list[int],
    frame_size: tuple[int, int],
    *,
    moving_pids: set[int] | None = None,
    legs_visible: dict[int, bool] | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Post-proceso VLM: cinemática, postura, debias de talk grupal, social coherente.

    `moving_pids`/`legs_visible`: señales del pipeline de pose (marcha real +
    piernas visibles), ver apply_chunk_kinematic_hints/enrich_action_posture.
    Sin pose (pipeline clásico), ambos quedan en None y el comportamiento es
    el de siempre."""
    acts = apply_chunk_kinematic_hints(
        actions, buffer_dets, person_ids, frame_size, moving_pids=moving_pids
    )
    acts, social = debias_group_talking(acts, social, person_ids, buffer_dets)
    acts = enrich_action_posture(acts, buffer_dets, person_ids, legs_visible=legs_visible)
    social = resolve_social_states(acts, social, person_ids)
    return acts, social
