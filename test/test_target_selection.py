"""pick_interaction_target/TargetStability: prioridad ATTENTIVE>AVAILABLE,
desempate por costo de interrupción del VLM ('interrupt_cost') y cercanía
(dist_to_robot real si viene de un tracker LiDAR, si no área de caja), y el
filtro de confirmación por permanencia (N chunks de VLM SEGUIDOS con la
misma intención elegible antes de aceptar un target NUEVO)."""

from vlm_hri.pose.gait import TargetStability, pick_interaction_target
from vlm_hri.social_state import SOCIAL_ATTENTIVE, SOCIAL_AVAILABLE, SOCIAL_BUSY


def _det(pid: int, *, w: int = 50, h: int = 50, dist=None) -> dict:
    d = {"pid": pid, "x1": 0, "y1": 0, "x2": w, "y2": h}
    if dist is not None:
        d["dist_to_robot"] = dist
    return d


def test_attentive_beats_available():
    # 2 es ATTENTIVE con caja más chica (más lejos por el proxy de área) --
    # debe ganar igual por prioridad de estado, no por cercanía.
    dets = [_det(1, w=200, h=200), _det(2, w=10, h=10)]
    social = {1: SOCIAL_AVAILABLE, 2: SOCIAL_ATTENTIVE}
    assert pick_interaction_target(dets, social) == 2


def test_nobody_eligible_returns_none():
    dets = [_det(1), _det(2)]
    social = {1: SOCIAL_BUSY, 2: "ENGAGED"}
    assert pick_interaction_target(dets, social) is None


def test_area_tiebreak_without_interrupt_cost():
    # Misma prioridad (AVAILABLE); sin interrupt_cost, gana el más cercano
    # a la cámara (caja más grande) -- comportamiento previo intacto.
    dets = [_det(1, w=10, h=10), _det(2, w=200, h=200)]
    social = {1: SOCIAL_AVAILABLE, 2: SOCIAL_AVAILABLE}
    assert pick_interaction_target(dets, social) == 2


def test_interrupt_cost_tiebreak_beats_area():
    # Misma prioridad; 1 tiene caja más chica (más lejos) pero menor costo
    # de interrupción -- debe ganar 1 (el costo social pesa antes que la
    # cercanía por caja, ver comentario en gait.py).
    dets = [_det(1, w=10, h=10), _det(2, w=200, h=200)]
    social = {1: SOCIAL_AVAILABLE, 2: SOCIAL_AVAILABLE}
    interrupt_cost = {1: 0, 2: 2}
    assert pick_interaction_target(dets, social, interrupt_cost) == 1


def test_missing_interrupt_cost_defaults_conservative():
    # 1 tiene costo explícito bajo; 2 no tiene dato -> se asume costo alto
    # (2, conservador) y pierde frente a 1 aunque su caja sea más grande.
    dets = [_det(1, w=10, h=10), _det(2, w=200, h=200)]
    social = {1: SOCIAL_AVAILABLE, 2: SOCIAL_AVAILABLE}
    interrupt_cost = {1: 0}
    assert pick_interaction_target(dets, social, interrupt_cost) == 1


def test_dist_to_robot_overrides_area_proxy_when_present():
    # 1 tiene caja pequeña (proxy de área diría "lejos") pero dist_to_robot
    # real dice que está más cerca que 2 -- debe ganar 1.
    dets = [_det(1, w=10, h=10, dist=0.8), _det(2, w=200, h=200, dist=3.0)]
    social = {1: SOCIAL_AVAILABLE, 2: SOCIAL_AVAILABLE}
    assert pick_interaction_target(dets, social) == 1


def test_stability_sticky_no_reconfirmation_needed():
    """Una vez target, sigue siéndolo mientras sea elegible -- sin exigir
    reconfirmación chunk a chunk (eso solo aplica para ACEPTAR uno nuevo)."""
    ts = TargetStability(require_confirmation=True, min_confirm_chunks=2)
    ts.record_chunk({1: SOCIAL_ATTENTIVE}, {1})
    ts.record_chunk({1: SOCIAL_ATTENTIVE}, {1})
    assert ts.update([_det(1)], {1: SOCIAL_ATTENTIVE}) == 1
    # Un único chunk con otro estado elegible (AVAILABLE) para 1 no debería
    # importar -- sigue siendo el target actual mientras sea elegible.
    ts.record_chunk({1: SOCIAL_AVAILABLE}, {1})
    assert ts.update([_det(1)], {1: SOCIAL_AVAILABLE}) == 1


def test_stability_requires_n_consecutive_chunks_for_new_target():
    ts = TargetStability(require_confirmation=True, min_confirm_chunks=2)
    ts.record_chunk({1: SOCIAL_ATTENTIVE}, {1})
    # Solo 1 chunk confirmado todavía -- no debe aceptarse como target.
    assert ts.update([_det(1)], {1: SOCIAL_ATTENTIVE}) is None
    ts.record_chunk({1: SOCIAL_ATTENTIVE}, {1})
    # 2 chunks seguidos con la misma intención -- ya se acepta.
    assert ts.update([_det(1)], {1: SOCIAL_ATTENTIVE}) == 1


def test_stability_resets_confirmation_on_state_change():
    ts = TargetStability(require_confirmation=True, min_confirm_chunks=2)
    ts.record_chunk({1: SOCIAL_ATTENTIVE}, {1})
    ts.record_chunk({1: SOCIAL_AVAILABLE}, {1})  # cambia de intención -> resetea a 1
    assert ts.update([_det(1)], {1: SOCIAL_AVAILABLE}) is None
    ts.record_chunk({1: SOCIAL_AVAILABLE}, {1})
    assert ts.update([_det(1)], {1: SOCIAL_AVAILABLE}) == 1


def test_stability_resets_confirmation_when_pid_leaves_frame():
    ts = TargetStability(require_confirmation=True, min_confirm_chunks=2)
    ts.record_chunk({1: SOCIAL_ATTENTIVE}, {1})
    ts.record_chunk({}, set())  # 1 sale de cuadro este chunk
    ts.record_chunk({1: SOCIAL_ATTENTIVE}, {1})
    # De vuelta en cuadro, pero la racha se reinició -- solo 1 chunk.
    assert ts.update([_det(1)], {1: SOCIAL_ATTENTIVE}) is None


def test_stability_default_no_confirmation_required():
    """require_confirmation=False (default): comportamiento idéntico al de
    antes de este cambio -- sin record_chunk, el primer frame elegible ya
    puede ser target (pipelines offline que nunca llaman record_chunk)."""
    ts = TargetStability()
    assert ts.update([_det(1)], {1: SOCIAL_ATTENTIVE}) == 1
