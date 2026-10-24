"""TargetFilter: EMA + histéresis del target del VLM entre trozos. Un null o
un voto aislado no debe mover el target en modo "hyst"; un cambio sostenido
sí; MOVING suelta de inmediato; un pid ausente se olvida tras absent_chunks."""

import pytest

from vlm_hri.social_state import SOCIAL_ATTENTIVE, SOCIAL_AVAILABLE, SOCIAL_BUSY, SOCIAL_MOVING
from vlm_hri.target_filter import (
    TargetFilter,
    TargetFilterConfig,
    chunk_record,
    replay,
    summarize,
    track_metrics,
)

AV = SOCIAL_AVAILABLE


def _run(tf, seq, social=None, seen=None):
    """seq: lista de (raw_pid, raw_conf); devuelve las decisiones por trozo."""
    social = social or {1: AV, 2: AV}
    seen = seen or {1, 2}
    return [tf.update(pid, conf, social, seen) for pid, conf in seq]


def _hyst(out):
    return [d["hyst"].pid for d in out]


def test_raw_mode_passes_vlm_through():
    tf = TargetFilter(TargetFilterConfig(mode="raw"))
    out = _run(tf, [(1, 0.9), (None, None), (2, 0.8)])
    assert [tf.selected(d).pid for d in out] == [1, None, 2]
    assert out[0]["raw"].confidence == 0.9


def test_hyst_needs_min_stable_votes_to_enter():
    tf = TargetFilter(TargetFilterConfig())
    out = _run(tf, [(1, 0.9), (1, 0.9)])
    assert _hyst(out) == [None, 1]


def test_hyst_single_null_does_not_drop_target():
    tf = TargetFilter(TargetFilterConfig())
    out = _run(tf, [(1, 0.9)] * 4 + [(None, None)] + [(1, 0.9)])
    assert _hyst(out)[3:] == [1, 1, 1]
    # raw sí lo pierde en ese trozo
    assert out[4]["raw"].pid is None


def test_hyst_sustained_null_releases_target():
    # Sin respaldo: con alguien AVAILABLE, el null se reinterpreta (ver
    # test_fallback_*); aquí se prueba la mecánica pura de salida.
    tf = TargetFilter(TargetFilterConfig(fallback_weight=0.0))
    out = _run(tf, [(1, 0.9)] * 4 + [(None, None)] * 3)
    assert _hyst(out)[-1] is None


def test_hyst_single_vote_for_other_does_not_steal():
    tf = TargetFilter(TargetFilterConfig())
    out = _run(tf, [(1, 0.9)] * 4 + [(2, 0.9)] + [(1, 0.9)] * 2)
    assert set(_hyst(out)[3:]) == {1}


def test_hyst_sustained_vote_switches_target():
    tf = TargetFilter(TargetFilterConfig())
    out = _run(tf, [(1, 0.9)] * 4 + [(2, 0.9)] * 4)
    assert _hyst(out)[-1] == 2


def test_hyst_releases_immediately_when_target_starts_moving():
    tf = TargetFilter(TargetFilterConfig())
    _run(tf, [(1, 0.9)] * 4)
    d = tf.update(1, 0.9, {1: SOCIAL_MOVING, 2: AV}, {1, 2})
    assert d["hyst"].pid is None


def test_vote_for_ineligible_pid_counts_as_nobody():
    # El VLM eligió a 1 pero el refinamiento lo dejó BUSY -> voto a None.
    tf = TargetFilter(TargetFilterConfig())
    out = _run(tf, [(1, 0.9)] * 3, social={1: SOCIAL_BUSY, 2: AV})
    assert _hyst(out) == [None, None, None]
    assert out[-1]["raw"].pid == 1


def test_short_occlusion_keeps_target_then_forgets():
    tf = TargetFilter(TargetFilterConfig(absent_chunks=2))
    _run(tf, [(1, 0.9)] * 4)
    # 1 deja de verse: se mantiene 2 trozos (score congelado), luego se olvida
    hidden = [tf.update(None, None, {2: AV}, {2})["hyst"].pid for _ in range(3)]
    assert hidden[:2] == [1, 1]
    assert hidden[2] is None
    assert 1 not in tf.scores


def test_ema_mode_is_argmax_including_nobody():
    tf = TargetFilter(TargetFilterConfig(mode="ema", alpha=0.5, fallback_weight=0.0))
    out = _run(tf, [(1, 0.9), (None, None)])
    # trozo 0: s1=0.45, sNone=0.05 -> 1; trozo 1 (null): s1=0.225, sNone=0.525 -> None
    assert out[0]["ema"].pid == 1
    assert out[1]["ema"].pid is None
    assert tf.selected(out[1]) == out[1]["ema"]


def test_attentive_and_available_both_eligible():
    tf = TargetFilter(TargetFilterConfig())
    out = _run(tf, [(2, 1.0)] * 2, social={1: AV, 2: SOCIAL_ATTENTIVE})
    assert _hyst(out)[-1] == 2


def test_hyst_k_of_n_survives_single_interruption():
    # alpha=1: el líder de la EMA es el voto del trozo tal cual. Líderes
    # 1, 1, None, 1: nunca 3 seguidos, pero sí 3 de los últimos 4.
    seq = [(1, 1.0), (1, 1.0), (None, None), (1, 1.0)]
    seguidos = TargetFilter(TargetFilterConfig(alpha=1.0, min_stable=3, fallback_weight=0.0))
    assert _hyst(_run(seguidos, seq)) == [None] * 4
    k_de_n = TargetFilter(TargetFilterConfig(alpha=1.0, min_stable=3, stable_window=4, fallback_weight=0.0))
    assert _hyst(_run(k_de_n, seq)) == [None, None, None, 1]


def test_stable_window_below_min_stable_rejected():
    with pytest.raises(ValueError):
        TargetFilterConfig(min_stable=3, stable_window=2)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        TargetFilterConfig(mode="kalman")


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("VLM_TARGET_FILTER", "hyst")
    monkeypatch.setenv("VLM_TARGET_EMA_ALPHA", "0.7")
    cfg = TargetFilterConfig.from_env(min_stable=3, alpha=None)
    assert (cfg.mode, cfg.alpha, cfg.min_stable) == ("hyst", 0.7, 3)


def test_replay_reproduces_live_decisions():
    cfg = TargetFilterConfig(mode="hyst")
    tf = TargetFilter(cfg)
    social, seen = {1: AV, 2: AV}, {1, 2}
    seq = [(1, 0.9), (1, 0.8), (None, None), (2, 0.9), (1, 0.9), (2, 0.7), (2, 0.9), (2, 0.9)]
    live = [
        chunk_record(i, i * 10, pid, conf, social, seen, tf.update(pid, conf, social, seen))
        for i, (pid, conf) in enumerate(seq)
    ]
    assert replay(live, cfg) == live
    res = summarize(live, cfg, chunk_sec=1.0)
    assert res["metrics"]["hyst"]["switches"] <= res["metrics"]["raw"]["switches"]


def test_track_metrics():
    m = track_metrics([None, 1, 1, 2, None, 2], chunk_sec=1.0)
    assert m["switches"] == 4
    assert m["pid_swaps"] == 1
    assert m["drops"] == 1
    assert m["coverage"] == pytest.approx(4 / 6, abs=1e-4)


def test_fallback_null_with_attentive_keeps_target():
    # El VLM se contradice: marca a 1 ATTENTIVE pero devuelve target=null.
    tf = TargetFilter(TargetFilterConfig(fallback_weight=0.5))
    social = {1: SOCIAL_ATTENTIVE}
    _run(tf, [(1, 1.0)] * 3, social=social, seen={1})
    out = _run(tf, [(None, None)] * 6, social=social, seen={1})
    assert set(_hyst(out)) == {1}


def test_fallback_alone_does_not_create_target():
    # Solo votos de respaldo (nunca un voto real del VLM): score -> 0.5 sin
    # alcanzar enter_threshold, no se adopta target nuevo.
    tf = TargetFilter(TargetFilterConfig(fallback_weight=0.5))
    out = _run(tf, [(None, None)] * 10, social={1: SOCIAL_ATTENTIVE}, seen={1})
    assert set(_hyst(out)) == {None}


def test_fallback_disabled_drops_like_before():
    tf = TargetFilter(TargetFilterConfig(fallback_weight=0.0))
    social = {1: SOCIAL_ATTENTIVE}
    _run(tf, [(1, 1.0)] * 3, social=social, seen={1})
    out = _run(tf, [(None, None)] * 3, social=social, seen={1})
    assert _hyst(out)[-1] is None


def test_fallback_prefers_attentive_then_current():
    tf = TargetFilter(TargetFilterConfig())
    assert tf._fallback_pid({1: AV, 2: SOCIAL_ATTENTIVE}, {1, 2}) == 2
    tf._current = 3
    assert tf._fallback_pid({1: AV, 3: AV}, {1, 3}) == 3
    assert tf._fallback_pid({1: SOCIAL_BUSY}, {1}) is None


def test_ema_fallback_only_tie_goes_to_nobody():
    # Visto en video: 30 trozos con gente ATTENTIVE y target=null siempre --
    # pid y None empatan en ~0.5, ema no debe inventar un target.
    tf = TargetFilter(TargetFilterConfig(mode="ema"))
    out = _run(tf, [(None, None)] * 30, social={2: SOCIAL_ATTENTIVE}, seen={2})
    assert {d["ema"].pid for d in out} == {None}


def test_hyst_same_tier_tiebreak_flipping_does_not_block_target():
    # Medido en watching_robot_2.mp4: 5 personas ATTENTIVE con cajas
    # parecidas, el desempate por caja cambiaba de ganador cada trozo y la
    # histéresis no fijaba target en 14 trozos.
    at = SOCIAL_ATTENTIVE
    social = {p: at for p in range(1, 6)}
    tf = TargetFilter(TargetFilterConfig())
    out = _run(tf, [(p, None) for p in (5, 3, 5, 4, 5, 3, 1)], social=social, seen=set(social))
    hyst = _hyst(out)
    assert hyst[1] is not None and len(set(hyst[1:])) == 1


# --- voto continuo: "ema_c" y "bayes" ---------------------------------------

from vlm_hri.social_state import SOCIAL_ENGAGED  # noqa: E402
from vlm_hri.target_filter import target_evidence  # noqa: E402


def _cont(tf, evs, social=None):
    """evs: lista de {pid: e}; social por defecto AVAILABLE para todos."""
    out = []
    for ev in evs:
        soc = social or {p: AV for p in ev}
        out.append(tf.update(None, None, soc, set(ev), ev))
    return out


def test_evidence_uses_probs_and_pose_override():
    probs = {1: {"ATTENTIVE": 0.5, "AVAILABLE": 0.3, "BUSY": 0.2}, 2: {"ATTENTIVE": 0.6, "AVAILABLE": 0.2}}
    probs[4] = {"ENGAGED": 0.8, "ATTENTIVE": 0.1, "AVAILABLE": 0.05}
    # pid 2: el modelo dijo ATTENTIVE pero la marcha lo pasó a MOVING; pid 4:
    # dijo ENGAGED y la mirada lo pasó a ATTENTIVE -> en ambos manda la pose.
    ev = target_evidence({1: SOCIAL_ATTENTIVE, 2: SOCIAL_MOVING, 3: SOCIAL_ENGAGED, 4: SOCIAL_ATTENTIVE}, probs,
                         {1: SOCIAL_ATTENTIVE, 2: SOCIAL_ATTENTIVE, 3: SOCIAL_ENGAGED, 4: SOCIAL_ENGAGED})
    assert ev[1] == pytest.approx(0.8)
    assert ev[2] == pytest.approx(0.02)
    assert ev[3] == pytest.approx(0.1)  # sin probabilidades: tabla
    assert ev[4] == pytest.approx(0.9)


def test_ema_c_follows_continuous_evidence():
    tf = TargetFilter(TargetFilterConfig())
    out = _cont(tf, [{1: 0.9, 2: 0.3}] * 3)
    assert [d["ema_c"].pid for d in out] == [None, 1, 1]  # 0.45 < enter, luego 0.675


def test_bayes_creates_target_with_consistent_evidence():
    tf = TargetFilter(TargetFilterConfig())
    out = _cont(tf, [{1: 0.85, 2: 0.2}] * 4)
    assert out[-1]["bayes"].pid == 1
    assert tf.belief[1] > 0.5


def test_bayes_no_target_when_nobody_approachable():
    tf = TargetFilter(TargetFilterConfig())
    out = _cont(tf, [{1: 0.1, 2: 0.05}] * 5)
    assert all(d["bayes"].pid is None for d in out)


def test_bayes_single_contrary_chunk_does_not_switch():
    tf = TargetFilter(TargetFilterConfig())
    out = _cont(tf, [{1: 0.85, 2: 0.3}] * 4 + [{1: 0.3, 2: 0.9}] + [{1: 0.85, 2: 0.3}])
    assert [d["bayes"].pid for d in out][3:] == [1, 1, 1]


def test_bayes_sustained_challenger_switches():
    tf = TargetFilter(TargetFilterConfig())
    out = _cont(tf, [{1: 0.85, 2: 0.3}] * 4 + [{1: 0.2, 2: 0.9}] * 6)
    assert out[-1]["bayes"].pid == 2


def test_bayes_moving_releases_immediately():
    tf = TargetFilter(TargetFilterConfig())
    _cont(tf, [{1: 0.85}] * 4)
    d = tf.update(None, None, {1: SOCIAL_MOVING}, {1}, {1: 0.85})
    assert d["bayes"].pid is None


def test_replay_reproduces_continuous_modes():
    tf = TargetFilter(TargetFilterConfig())
    recs = []
    for i, ev in enumerate([{1: 0.85, 2: 0.2}] * 4):
        soc = {1: AV, 2: AV}
        d = tf.update(None, None, soc, {1, 2}, ev)
        recs.append(chunk_record(i, i * 25, None, None, soc, {1, 2}, d, evidence=ev))
    again = replay(recs, TargetFilterConfig())
    assert [c["bayes"]["pid"] for c in again] == [c["bayes"]["pid"] for c in recs]
    assert [c["ema_c"]["pid"] for c in again] == [c["ema_c"]["pid"] for c in recs]
    assert [c["ema_c_hyst"]["pid"] for c in again] == [c["ema_c_hyst"]["pid"] for c in recs]


def _ch(out):
    return [d["ema_c_hyst"].pid for d in out]


def test_ema_c_hyst_enters_holds_below_enter_then_releases():
    tf = TargetFilter(TargetFilterConfig())
    # s1: 0.45 (< enter), 0.675, 0.7875 -> líder 2 trozos -> entra en el 3º.
    out = _cont(tf, [{1: 0.9, 2: 0.3}] * 3)
    assert _ch(out) == [None, None, 1]
    # s1 = 0.49: ema_c lo pierde (< enter), ema_c_hyst lo conserva (>= exit).
    d = _cont(tf, [{1: 0.2, 2: 0.3}])[0]
    assert (d["ema_c"].pid, d["ema_c_hyst"].pid) == (None, 1)
    # s1 = 0.247 < exit -> se suelta.
    assert _ch(_cont(tf, [{1: 0.0, 2: 0.3}])) == [None]


def test_ema_c_hyst_single_better_chunk_does_not_steal():
    tf = TargetFilter(TargetFilterConfig())
    _cont(tf, [{1: 0.9, 2: 0.3}] * 3)
    out = _cont(tf, [{1: 0.3, 2: 1.0}] + [{1: 0.9, 2: 0.3}])
    assert out[0]["ema_c"].pid == 2  # ema_c sí salta
    assert _ch(out) == [1, 1]


def test_ema_c_hyst_sustained_challenger_switches():
    tf = TargetFilter(TargetFilterConfig())
    _cont(tf, [{1: 0.9, 2: 0.3}] * 3)
    assert _ch(_cont(tf, [{1: 0.3, 2: 1.0}] * 2)) == [1, 2]


def test_ema_c_hyst_moving_releases_immediately():
    tf = TargetFilter(TargetFilterConfig())
    _cont(tf, [{1: 0.9}] * 3)
    d = tf.update(None, None, {1: SOCIAL_MOVING}, {1}, {1: 0.9})
    assert d["ema_c_hyst"].pid is None


def test_auto_mode_follows_backend(monkeypatch):
    cfg = TargetFilterConfig(mode="auto")
    monkeypatch.setenv("VLM_BACKEND", "vlm_jev")
    assert cfg.resolved_mode() == "ema_c_hyst"
    monkeypatch.setenv("VLM_BACKEND", "vlm")
    assert cfg.resolved_mode() == "hyst"
    tf = TargetFilter(cfg)
    d = tf.update(1, 0.9, {1: AV}, {1}, {1: 0.9})
    assert tf.selected(d) is d["hyst"]
