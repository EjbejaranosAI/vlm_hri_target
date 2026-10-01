"""Filtro temporal del target de interacción (EMA + histéresis).

El VLM elige un target por trozo (~1s, campos "target"/"target_conf" del
prompt) y una sola clasificación es ruidosa: un `null` suelto o un voto
aislado por otra persona hacían que /vlm_hri/target_pose saltara o se
perdiera. Aquí se filtra en el tiempo, igual que se hace con emociones
frame a frame (EMA de probabilidades + histéresis):

a) EMA por "clase" -- cada pid visto + `None` ("nadie es apto") como una
   clase más. El voto de un trozo es una distribución que suma 1:
   target del VLM -> `target_conf` (1.0 si no la dio), `None` -> el resto.
   Si el VLM dijo `null`, o eligió a alguien que tras el refinamiento por
   pose/cinemática ya no es ATTENTIVE/AVAILABLE, el voto va a `None` --
   salvo que haya alguien elegible ese mismo trozo: el VLM se contradice
   (medido en vivo: `"s": "ATTENTIVE"` + `"target": null` en la misma
   respuesta, a pesar de que el prompt lo prohíbe), y entonces se vota por
   el mejor elegible (ATTENTIVE > AVAILABLE, desempate: target actual, luego
   mayor score) con peso `fallback_weight` en vez de por nadie. Con el
   default 0.5 el score de ese pid converge a 0.5 sin superarlo: solo con
   votos de respaldo alcanza para CONSERVAR un target (> exit) pero no para
   CREAR uno (>= enter); sí puede completar la racha de min_stable que
   empezó un voto real. 0 = desactivado (todo a `None`, como antes).
       s_t = alpha * p_t + (1 - alpha) * s_{t-1}
b) Histéresis -- el target solo cambia cuando un candidato (o `None`) lleva
   `min_stable` trozos SEGUIDOS como líder de la EMA y su score supera
   `enter_threshold`. El target actual se suelta antes si su score cae por
   debajo de `exit_threshold` (umbral de salida < entrada), si pasa a
   MOVING (salida rápida: no acercarse a alguien que se va) o si lleva más
   de `absent_chunks` trozos sin verse.

Un pid que deja de verse conserva su score congelado hasta `absent_chunks`
trozos (oclusión corta / pid que el VLM no recibió ese trozo) y luego se
olvida.

Los 3 modos ("raw", "ema", "hyst") se calculan SIEMPRE en paralelo sobre la
misma secuencia de votos -- `mode` solo decide cuál maneja al robot/overlay.
Así una sola corrida da la comparación sin el ruido de que el VLM no es
determinista entre corridas. `replay()` re-simula offline con otros
parámetros a partir de los votos guardados en summary.json:

    python -m vlm_hri.target_filter output/.../summary.json --alpha 0.6
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace

from .social_state import SOCIAL_ATTENTIVE, SOCIAL_AVAILABLE, SOCIAL_MOVING

TARGET_FILTER_MODES = ("raw", "ema", "hyst")
TARGET_SOURCES = ("rules", "vlm")
_ELIGIBLE = {SOCIAL_ATTENTIVE, SOCIAL_AVAILABLE}


@dataclass(frozen=True)
class TargetFilterConfig:
    mode: str = "raw"  # cuál de los 3 modos maneja al robot (los 3 se calculan igual)
    # De dónde sale el voto crudo de cada trozo: "rules" = pick_interaction_target
    # sobre las etiquetas 's' (ATTENTIVE > AVAILABLE > más cercano); "vlm" =
    # el target que eligió el propio VLM (flag "t"). Debe coincidir con
    # VLM_TARGET_SOURCE, que decide si el prompt pide "t" (ver
    # config.vlm_target_source) -- el nodo/CLI fijan ambos a la vez.
    source: str = "rules"
    alpha: float = 0.5
    enter_threshold: float = 0.5
    exit_threshold: float = 0.25
    min_stable: int = 2
    absent_chunks: int = 2
    fallback_weight: float = 0.5  # voto por el mejor elegible si el VLM dice null (0 = off)

    def __post_init__(self) -> None:
        if self.mode not in TARGET_FILTER_MODES:
            raise ValueError(f"target_filter mode={self.mode!r}, debe ser uno de {TARGET_FILTER_MODES}")
        if self.source not in TARGET_SOURCES:
            raise ValueError(f"target_filter source={self.source!r}, debe ser uno de {TARGET_SOURCES}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"target_filter alpha={self.alpha} fuera de (0, 1]")
        if not 0.0 <= self.fallback_weight <= 1.0:
            raise ValueError(f"target_filter fallback_weight={self.fallback_weight} fuera de [0, 1]")

    @classmethod
    def from_env(cls, **overrides) -> "TargetFilterConfig":
        """Variables de entorno VLM_TARGET_* como default; `overrides` con
        valor None se ignoran (p. ej. un flag de CLI no pasado)."""
        env = os.environ.get
        cfg = cls(
            mode=env("VLM_TARGET_FILTER", cls.mode).strip().lower(),
            source=env("VLM_TARGET_SOURCE", cls.source).strip().lower(),
            alpha=float(env("VLM_TARGET_EMA_ALPHA", cls.alpha)),
            enter_threshold=float(env("VLM_TARGET_ENTER", cls.enter_threshold)),
            exit_threshold=float(env("VLM_TARGET_EXIT", cls.exit_threshold)),
            min_stable=int(env("VLM_TARGET_MIN_STABLE", cls.min_stable)),
            absent_chunks=int(env("VLM_TARGET_ABSENT_CHUNKS", cls.absent_chunks)),
            fallback_weight=float(env("VLM_TARGET_FALLBACK_WEIGHT", cls.fallback_weight)),
        )
        return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


@dataclass(frozen=True)
class TargetDecision:
    pid: int | None
    confidence: float | None  # raw: target_conf del VLM; ema/hyst: score EMA del target


@dataclass
class TargetFilter:
    cfg: TargetFilterConfig = field(default_factory=TargetFilterConfig)
    # clave None = "nadie es apto"; siempre presente
    scores: dict[int | None, float] = field(default_factory=lambda: {None: 0.0})
    _absent: dict[int, int] = field(default_factory=dict)
    _current: int | None = None
    _streak_key: int | None = None
    _streak_n: int = 0

    def _votes(
        self,
        raw_pid: int | None,
        raw_conf: float | None,
        social_states: dict[int, str],
        seen: set[int],
    ) -> dict[int | None, float]:
        if raw_pid is not None and social_states.get(raw_pid) in _ELIGIBLE:
            c = 1.0 if raw_conf is None else min(1.0, max(0.0, float(raw_conf)))
            return {raw_pid: c, None: 1.0 - c}
        fb = self._fallback_pid(social_states, seen)
        if fb is None or self.cfg.fallback_weight <= 0.0:
            return {None: 1.0}
        w = self.cfg.fallback_weight
        return {fb: w, None: 1.0 - w}

    def _fallback_pid(self, social_states: dict[int, str], seen: set[int]) -> int | None:
        """Mejor elegible visto este trozo (para cuando el VLM dice null pero
        él mismo marcó a alguien ATTENTIVE/AVAILABLE)."""
        eligible = [p for p in seen if social_states.get(p) in _ELIGIBLE]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda p: (
                social_states[p] != SOCIAL_ATTENTIVE,
                p != self._current,
                -self.scores.get(p, 0.0),
                p,
            ),
        )

    def _update_scores(self, votes: dict[int | None, float], seen: set[int]) -> None:
        a = self.cfg.alpha
        for pid in seen | {k for k in votes if k is not None}:
            self.scores.setdefault(pid, 0.0)
        for key in list(self.scores):
            if key is not None and key not in seen:
                self._absent[key] = self._absent.get(key, 0) + 1
                if self._absent[key] > self.cfg.absent_chunks:
                    del self.scores[key]
                    del self._absent[key]
                continue  # congelado mientras no se ve
            self._absent.pop(key, None)
            self.scores[key] = a * votes.get(key, 0.0) + (1.0 - a) * self.scores[key]

    def _leader(self) -> int | None:
        # Empates: gana None (con solo votos de respaldo, pid y None convergen
        # ambos a fallback_weight=0.5 -- un empate no debe crear target), y
        # entre pids el menor (determinista).
        return min(self.scores, key=lambda k: (-self.scores[k], k is not None, k or 0))

    def update(
        self,
        raw_pid: int | None,
        raw_conf: float | None,
        social_states: dict[int, str],
        seen_pids: set[int],
    ) -> dict[str, TargetDecision]:
        """Llamar UNA vez por trozo de VLM (no por frame) con el target crudo,
        el estado social YA refinado de ese trozo y los pids que se le pasaron
        al VLM. Devuelve la decisión de los 3 modos."""
        seen = set(seen_pids)
        self._update_scores(self._votes(raw_pid, raw_conf, social_states, seen), seen)

        leader = self._leader()
        ema = TargetDecision(leader, self.scores[leader] if leader is not None else None)

        if leader == self._streak_key:
            self._streak_n += 1
        else:
            self._streak_key, self._streak_n = leader, 1

        cur = self._current
        if cur is not None and (
            cur not in self.scores
            or self.scores[cur] < self.cfg.exit_threshold
            or (cur in seen and social_states.get(cur) == SOCIAL_MOVING)
        ):
            cur = None
        if (
            leader != cur
            and self._streak_n >= self.cfg.min_stable
            and self.scores[leader] >= self.cfg.enter_threshold
        ):
            cur = leader
        self._current = cur
        hyst = TargetDecision(cur, self.scores[cur] if cur is not None else None)

        return {"raw": TargetDecision(raw_pid, raw_conf), "ema": ema, "hyst": hyst}

    def selected(self, decisions: dict[str, TargetDecision]) -> TargetDecision:
        return decisions[self.cfg.mode]


def chunk_record(
    chunk_index: int,
    start_frame: int,
    raw_pid: int | None,
    raw_conf: float | None,
    social_states: dict[int, str],
    seen_pids: set[int],
    decisions: dict[str, TargetDecision],
    target_probs: dict[int, float] | None = None,
    *,
    source: str = "vlm",
    vlm_pid: int | None = None,
) -> dict:
    """Entrada JSON-serializable por trozo: los votos (para `replay`) + lo
    que decidió cada modo. `target_probs`: P(t=1) por persona de los logits
    (solo informativo, el filtro vota con raw_pid/raw_conf)."""
    return {
        "chunk_index": chunk_index,
        "start_frame": start_frame,
        "raw_pid": raw_pid,
        "raw_conf": raw_conf,
        # "rules": raw_pid salió de las reglas y vlm_pid es lo que habría
        # elegido el VLM (None si el prompt no lo pidió); "vlm": son el mismo.
        "source": source,
        "vlm_pid": vlm_pid if source == "rules" else raw_pid,
        "target_probs": {str(p): round(v, 4) for p, v in (target_probs or {}).items()},
        "social": {str(p): s for p, s in social_states.items()},
        "seen": sorted(seen_pids),
        **{
            mode: {"pid": d.pid, "conf": None if d.confidence is None else round(d.confidence, 4)}
            for mode, d in decisions.items()
        },
    }


def replay(chunks: list[dict], cfg: TargetFilterConfig) -> list[dict]:
    """Re-simula el filtro con `cfg` sobre los votos guardados en
    `chunk_record`s -- para barrer alpha/umbrales sin volver a correr el VLM."""
    tf = TargetFilter(cfg)
    out = []
    for c in chunks:
        social = {int(p): s for p, s in c["social"].items()}
        seen = set(c["seen"])
        decisions = tf.update(c["raw_pid"], c["raw_conf"], social, seen)
        probs = {int(p): v for p, v in c.get("target_probs", {}).items()}
        out.append(chunk_record(
            c["chunk_index"], c["start_frame"], c["raw_pid"], c["raw_conf"], social, seen, decisions, probs,
            source=c.get("source", "vlm"), vlm_pid=c.get("vlm_pid"),
        ))
    return out


def track_metrics(pids: list[int | None], chunk_sec: float) -> dict:
    """Estabilidad de una secuencia de targets (uno por trozo):
    - switches: cualquier cambio (incluye ganar/perder target)
    - pid_swaps: cambio directo de una persona a OTRA (lo más grave para el robot)
    - drops: de una persona a None
    - coverage: fracción de trozos con target"""
    n = len(pids)
    pairs = list(zip(pids, pids[1:]))
    switches = sum(a != b for a, b in pairs)
    minutes = n * chunk_sec / 60.0
    return {
        "chunks": n,
        "switches": switches,
        "pid_swaps": sum(a is not None and b is not None and a != b for a, b in pairs),
        "drops": sum(a is not None and b is None for a, b in pairs),
        "coverage": round(sum(p is not None for p in pids) / n, 4) if n else 0.0,
        "switches_per_min": round(switches / minutes, 3) if minutes > 0 else 0.0,
    }


def summarize(chunks: list[dict], cfg: TargetFilterConfig, chunk_sec: float) -> dict:
    """Bloque "target_filter" de summary.json: config, votos/decisiones por
    trozo y métricas de los 3 modos."""
    return {
        "config": asdict(cfg),
        "metrics": {
            mode: track_metrics([c[mode]["pid"] for c in chunks], chunk_sec) for mode in TARGET_FILTER_MODES
        },
        "chunks": chunks,
    }


def _main() -> None:
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description="Re-simula el filtro de target sobre un summary.json")
    p.add_argument("summary", type=Path)
    p.add_argument("--alpha", type=float)
    p.add_argument("--enter", type=float, dest="enter_threshold")
    p.add_argument("--exit", type=float, dest="exit_threshold")
    p.add_argument("--min-stable", type=int)
    p.add_argument("--absent-chunks", type=int)
    p.add_argument("--fallback-weight", type=float)
    args = p.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    block = summary.get("target_filter")
    if not block:
        raise SystemExit("summary.json sin bloque 'target_filter' (corrida anterior a este filtro)")
    base = TargetFilterConfig.from_env(**block["config"])  # summaries viejos sin fallback_weight
    cfg = replace(base, **{k: v for k, v in vars(args).items() if k != "summary" and v is not None})
    chunks = replay(block["chunks"], cfg)
    res = summarize(chunks, cfg, float(summary.get("chunk_sec", 1.0)))
    print(json.dumps(res["config"]))
    for mode, m in res["metrics"].items():
        print(f"  {mode:5s} {m}")
    print("  trozo  raw   ema   hyst")
    for c in chunks:
        print(f"  {c['chunk_index']:5d}  {c['raw']['pid']!s:5s} {c['ema']['pid']!s:5s} {c['hyst']['pid']!s:5s}")


if __name__ == "__main__":
    _main()
