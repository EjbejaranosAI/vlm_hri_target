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
b) Histéresis -- el target solo cambia cuando un candidato (o `None`) ha
   sido líder de la EMA en al menos `min_stable` de los últimos
   `stable_window` trozos (0 = min_stable, es decir, `min_stable` SEGUIDOS)
   y su score supera `enter_threshold`. Con ventana > min_stable un voto
   suelto no reinicia la cuenta. El target actual se suelta antes si su score cae por
   debajo de `exit_threshold` (umbral de salida < entrada), si pasa a
   MOVING (salida rápida: no acercarse a alguien que se va) o si lleva más
   de `absent_chunks` trozos sin verse.

Un pid que deja de verse conserva su score congelado hasta `absent_chunks`
trozos (oclusión corta / pid que el VLM no recibió ese trozo) y luego se
olvida.

c) Voto continuo -- cada persona vista aporta una evidencia e_i en [0, 1]
   de que ES abordable (ver target_evidence: con VLM_BACKEND=vlm_jev,
   P(ATTENTIVE)+P(AVAILABLE) del modelo; con el VLM generativo, una tabla
   por estado ya refinado), no solo el ganador del trozo:
   - "ema_c": EMA por persona de e_i; target = argmax si >= enter_threshold.
   - "ema_c_hyst": la misma EMA por persona + histéresis (como b) pero
     sobre probabilidades en vez de votos duros): entra con >= enter, se
     suelta con < exit o MOVING, y otra persona solo lo roba si es líder en
     `min_stable` de los últimos `stable_window` trozos y >= enter.
   - "bayes": filtro de Bayes discreto (el caso exacto de un filtro de
     partículas cuando la hipótesis es finita): b(h) sobre
     h in {nadie, pid_1..pid_n}. Predicción: b <- stay*b + (1-stay)/|H|
     (el target real cambia poco entre trozos). Actualización: verosimilitud
     L(pid) = e/(1-e) (odds de que sea abordable), L(nadie) = 1; un pid no
     visto ese trozo no aporta (L = 1). Decisión: P(alguien) = 1 - b(nadie)
     >= enter para crear target y >= exit para conservarlo; un retador
     reemplaza al actual solo si su b le saca `bayes_margin` durante
     `min_stable` trozos seguidos (con varias personas igual de abordables
     la masa se reparte y el argmax solo cambiaría por ruido).

Todos los modos ("raw", "ema", "hyst", "ema_c", "ema_c_hyst", "bayes") se calculan SIEMPRE en
paralelo sobre la misma secuencia de votos -- `mode` solo decide cuál maneja al robot/overlay.
`mode="auto"` elige según VLM_BACKEND: "ema_c_hyst" con vlm_jev (tiene
probabilidades calibradas por persona), "hyst" con el VLM generativo (solo
etiquetas, votos duros).
Así una sola corrida da la comparación sin el ruido de que el VLM no es
determinista entre corridas. `replay()` re-simula offline con otros
parámetros a partir de los votos guardados en summary.json:

    python -m vlm_hri.target_filter output/.../summary.json --alpha 0.6
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import asdict, dataclass, field, replace

from .social_state import (
    SOCIAL_ATTENTIVE,
    SOCIAL_AVAILABLE,
    SOCIAL_BUSY,
    SOCIAL_ENGAGED,
    SOCIAL_MOVING,
    SOCIAL_UNKNOWN,
)

TARGET_FILTER_MODES = ("raw", "ema", "hyst", "ema_c", "ema_c_hyst", "bayes")
# Valores válidos de `mode`: los calculados + "auto" (se resuelve por backend).
TARGET_FILTER_CHOICES = TARGET_FILTER_MODES + ("auto",)
TARGET_SOURCES = ("rules", "vlm")
_ELIGIBLE = {SOCIAL_ATTENTIVE, SOCIAL_AVAILABLE}
# Evidencia de "es abordable" cuando no hay probabilidades del modelo (VLM
# generativo) o cuando la pose contradijo al modelo (p. ej. marcha -> MOVING).
_STATE_EVIDENCE = {
    SOCIAL_ATTENTIVE: 0.9,
    SOCIAL_AVAILABLE: 0.7,
    SOCIAL_BUSY: 0.1,
    SOCIAL_ENGAGED: 0.1,
    SOCIAL_MOVING: 0.02,
    SOCIAL_UNKNOWN: 0.5,
}
_EVIDENCE_CLIP = 0.02  # e en [clip, 1-clip]: odds acotadas a [1/49, 49]


def target_evidence(
    social_states: dict[int, str],
    social_probs: dict[int, dict[str, float]] | None = None,
    model_social: dict[int, str] | None = None,
) -> dict[int, float]:
    """e_i en [0, 1] = cuánto parece abordable cada persona este trozo.
    Con probabilidades (JEV): P(ATTENTIVE) + P(AVAILABLE); si el refinado
    por pose cambió el estado que dijo el modelo (marcha -> MOVING, de
    espaldas -> BUSY, mirada -> ATTENTIVE), manda la pose: valor de tabla
    de ese estado. Sin probabilidades: tabla por estado."""
    out = {}
    for pid, st in social_states.items():
        table = _STATE_EVIDENCE.get(st, _STATE_EVIDENCE[SOCIAL_UNKNOWN])
        probs = (social_probs or {}).get(pid)
        if not probs:
            out[pid] = table
            continue
        if (model_social or {}).get(pid, st) != st:
            out[pid] = table
        else:
            out[pid] = probs.get(SOCIAL_ATTENTIVE, 0.0) + probs.get(SOCIAL_AVAILABLE, 0.0)
    return out


# Trozos seguidos que el mismo pid debe ganar el voto crudo para que su voto
# cuente aunque haya otro del mismo nivel con más historia (ver _sticky_pid).
_CHALLENGER_RUN = 2


@dataclass(frozen=True)
class TargetFilterConfig:
    mode: str = "raw"  # cuál de los modos maneja al robot (todos se calculan igual); "auto" = por backend
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
    # "hyst"/"ema_c_hyst": min_stable de los últimos stable_window trozos (0 = min_stable, seguidos).
    stable_window: int = 0
    absent_chunks: int = 2
    fallback_weight: float = 0.5  # voto por el mejor elegible si el VLM dice null (0 = off)
    bayes_stay: float = 0.9  # "bayes": P(el target real sigue siendo el mismo entre trozos)
    bayes_margin: float = 0.1  # "bayes": ventaja en b() que necesita un retador para robar el target

    def __post_init__(self) -> None:
        if self.mode not in TARGET_FILTER_CHOICES:
            raise ValueError(f"target_filter mode={self.mode!r}, debe ser uno de {TARGET_FILTER_CHOICES}")
        if self.stable_window and self.stable_window < self.min_stable:
            raise ValueError(
                f"target_filter stable_window={self.stable_window} < min_stable={self.min_stable} (0 = min_stable)"
            )
        if self.source not in TARGET_SOURCES:
            raise ValueError(f"target_filter source={self.source!r}, debe ser uno de {TARGET_SOURCES}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"target_filter alpha={self.alpha} fuera de (0, 1]")
        if not 0.0 <= self.fallback_weight <= 1.0:
            raise ValueError(f"target_filter fallback_weight={self.fallback_weight} fuera de [0, 1]")
        if not 0.0 < self.bayes_stay < 1.0:
            raise ValueError(f"target_filter bayes_stay={self.bayes_stay} fuera de (0, 1)")

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
            stable_window=int(env("VLM_TARGET_STABLE_WINDOW", cls.stable_window)),
            absent_chunks=int(env("VLM_TARGET_ABSENT_CHUNKS", cls.absent_chunks)),
            fallback_weight=float(env("VLM_TARGET_FALLBACK_WEIGHT", cls.fallback_weight)),
            bayes_stay=float(env("VLM_TARGET_BAYES_STAY", cls.bayes_stay)),
            bayes_margin=float(env("VLM_TARGET_BAYES_MARGIN", cls.bayes_margin)),
        )
        return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})

    @property
    def window(self) -> int:
        return self.stable_window or self.min_stable

    def resolved_mode(self) -> str:
        """`mode` con "auto" resuelto: probabilidades por persona con JEV,
        votos duros con el VLM generativo."""
        if self.mode != "auto":
            return self.mode
        from .config import vlm_backend  # perezoso: config importa torch/cv2

        return "ema_c_hyst" if vlm_backend() == "vlm_jev" else "hyst"


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
    # Líderes de la EMA de los últimos `cfg.window` trozos (histéresis K de N).
    _leaders: deque = field(default_factory=deque)
    _raw_last: int | None = None
    _raw_run: int = 0
    # "ema_c": EMA de la evidencia continua por persona.
    cscores: dict[int, float] = field(default_factory=dict)
    _c_absent: dict[int, int] = field(default_factory=dict)
    # "ema_c_hyst": histéresis sobre cscores.
    _ch_current: int | None = None
    _ch_leaders: deque = field(default_factory=deque)
    # "bayes": creencia sobre quién es el target (None = nadie).
    belief: dict[int | None, float] = field(default_factory=lambda: {None: 1.0})
    _b_absent: dict[int, int] = field(default_factory=dict)
    _b_current: int | None = None
    _b_challenger: int | None = None
    _b_challenger_n: int = 0

    def _votes(
        self,
        raw_pid: int | None,
        raw_conf: float | None,
        social_states: dict[int, str],
        seen: set[int],
    ) -> dict[int | None, float]:
        if raw_pid is not None and social_states.get(raw_pid) in _ELIGIBLE:
            c = 1.0 if raw_conf is None else min(1.0, max(0.0, float(raw_conf)))
            return {self._sticky_pid(raw_pid, social_states, seen): c, None: 1.0 - c}
        fb = self._fallback_pid(social_states, seen)
        if fb is None or self.cfg.fallback_weight <= 0.0:
            return {None: 1.0}
        w = self.cfg.fallback_weight
        return {fb: w, None: 1.0 - w}

    def _sticky_pid(self, raw_pid: int, social_states: dict[int, str], seen: set[int]) -> int:
        """Entre personas del MISMO nivel que el voto crudo (todas ATTENTIVE o
        todas AVAILABLE), vota por el target actual y si no por el de mayor
        score acumulado, no por el ganador de este trozo. Medido en
        watching_robot_2.mp4 (5 personas mirando al robot, cajas de tamaño
        parecido): el desempate por caja más grande cambiaba de ganador casi
        cada trozo (5,3,5,4,5,3,1), los votos se repartían y la histéresis
        pasaba 14 trozos SIN target. Un nivel mejor (ATTENTIVE sobre
        AVAILABLE) sigue ganando siempre, y si el MISMO retador gana el voto
        crudo `_CHALLENGER_RUN` trozos seguidos (p. ej. alguien que se acerca
        y pasa a ser el más cercano) su voto cuenta tal cual."""
        if self._raw_run >= _CHALLENGER_RUN:
            return raw_pid  # mismo retador varios trozos seguidos: cambio real
        tier = social_states[raw_pid]
        same = [p for p in seen if social_states.get(p) == tier] or [raw_pid]
        if raw_pid not in same:
            same.append(raw_pid)
        return min(same, key=lambda p: (p != self._current, -self.scores.get(p, 0.0), p != raw_pid, p))

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

    def _age_out(self, store: dict, absent: dict[int, int], seen: set[int]) -> None:
        """Cuenta trozos sin ver cada pid de `store` y lo borra tras absent_chunks."""
        for key in [k for k in store if k is not None]:
            if key in seen:
                absent.pop(key, None)
                continue
            absent[key] = absent.get(key, 0) + 1
            if absent[key] > self.cfg.absent_chunks:
                del store[key]
                del absent[key]

    def _update_ema_c(self, evidence: dict[int, float], seen: set[int]) -> TargetDecision:
        self._age_out(self.cscores, self._c_absent, seen)
        a = self.cfg.alpha
        for pid in seen:
            self.cscores[pid] = a * evidence.get(pid, 0.0) + (1.0 - a) * self.cscores.get(pid, 0.0)
        if not self.cscores:
            return TargetDecision(None, None)
        leader = min(self.cscores, key=lambda p: (-self.cscores[p], p))
        if self.cscores[leader] < self.cfg.enter_threshold:
            return TargetDecision(None, None)
        return TargetDecision(leader, self.cscores[leader])

    def _push_leader(self, leaders: deque, leader: int | None) -> int:
        """Añade el líder de este trozo a la ventana y devuelve cuántas veces
        fue líder en los últimos `cfg.window` trozos (K de N)."""
        leaders.append(leader)
        while len(leaders) > self.cfg.window:
            leaders.popleft()
        return leaders.count(leader)

    def _update_ema_c_hyst(self, social_states: dict[int, str], seen: set[int]) -> TargetDecision:
        """Histéresis sobre `cscores` (ya actualizados por _update_ema_c)."""
        s = self.cscores

        def moving(p: int) -> bool:
            return p in seen and social_states.get(p) == SOCIAL_MOVING

        pids = [p for p in s if not moving(p)]
        leader = min(pids, key=lambda p: (-s[p], p)) if pids else None
        if leader is not None and s[leader] < self.cfg.enter_threshold:
            leader = None  # nadie llega a enter: cuenta como trozo sin candidato
        n = self._push_leader(self._ch_leaders, leader)
        cur = self._ch_current
        if cur is not None and (cur not in s or s[cur] < self.cfg.exit_threshold or moving(cur)):
            cur = None
        if leader is not None and leader != cur and n >= self.cfg.min_stable:
            cur = leader
        self._ch_current = cur
        return TargetDecision(cur, s[cur] if cur is not None else None)

    def _update_bayes(
        self, evidence: dict[int, float], social_states: dict[int, str], seen: set[int]
    ) -> TargetDecision:
        b = self.belief
        for pid in seen:
            b.setdefault(pid, 0.0)
        self._age_out(b, self._b_absent, seen)
        # Predicción: el target real se mantiene con prob. stay, si no salta
        # a cualquier hipótesis (incluida "nadie") por igual.
        stay, n = self.cfg.bayes_stay, len(b)
        total = sum(b.values()) or 1.0
        for h in b:
            b[h] = stay * b[h] / total + (1.0 - stay) / n
        # Actualización con la evidencia de este trozo.
        lo, hi = _EVIDENCE_CLIP, 1.0 - _EVIDENCE_CLIP
        for pid in b:
            if pid is not None and pid in seen:
                e = min(hi, max(lo, evidence.get(pid, 0.5)))
                b[pid] *= e / (1.0 - e)
        total = sum(b.values())
        for h in b:
            b[h] /= total

        p_any = 1.0 - b[None]
        # Nunca adoptar (ni como retador) a quien está MOVING este trozo.
        pids = [p for p in b if p is not None and not (p in seen and social_states.get(p) == SOCIAL_MOVING)]
        leader = min(pids, key=lambda p: (-b[p], p)) if pids else None
        cur = self._b_current
        if cur is not None and (
            cur not in b
            or p_any < self.cfg.exit_threshold
            or (cur in seen and social_states.get(cur) == SOCIAL_MOVING)
        ):
            cur = None
        if cur is None:
            self._b_challenger, self._b_challenger_n = None, 0
            if leader is not None and p_any >= self.cfg.enter_threshold:
                cur = leader
        elif leader != cur and b[leader] >= b[cur] + self.cfg.bayes_margin:
            if leader == self._b_challenger:
                self._b_challenger_n += 1
            else:
                self._b_challenger, self._b_challenger_n = leader, 1
            if self._b_challenger_n >= self.cfg.min_stable:
                cur = leader
                self._b_challenger, self._b_challenger_n = None, 0
        else:
            self._b_challenger, self._b_challenger_n = None, 0
        self._b_current = cur
        return TargetDecision(cur, b[cur] if cur is not None else None)

    def update(
        self,
        raw_pid: int | None,
        raw_conf: float | None,
        social_states: dict[int, str],
        seen_pids: set[int],
        evidence: dict[int, float] | None = None,
    ) -> dict[str, TargetDecision]:
        """Llamar UNA vez por trozo de VLM (no por frame) con el target crudo,
        el estado social YA refinado de ese trozo y los pids que se le pasaron
        al VLM. `evidence`: e_i por persona (ver target_evidence); None = la
        tabla por estado social. Devuelve la decisión de todos los modos."""
        seen = set(seen_pids)
        if evidence is None:
            evidence = target_evidence(social_states)
        self._raw_run = self._raw_run + 1 if raw_pid == self._raw_last else 1
        self._raw_last = raw_pid
        self._update_scores(self._votes(raw_pid, raw_conf, social_states, seen), seen)

        leader = self._leader()
        ema = TargetDecision(leader, self.scores[leader] if leader is not None else None)

        stable_n = self._push_leader(self._leaders, leader)

        cur = self._current
        if cur is not None and (
            cur not in self.scores
            or self.scores[cur] < self.cfg.exit_threshold
            or (cur in seen and social_states.get(cur) == SOCIAL_MOVING)
        ):
            cur = None
        if (
            leader != cur
            and stable_n >= self.cfg.min_stable
            and self.scores[leader] >= self.cfg.enter_threshold
        ):
            cur = leader
        self._current = cur
        hyst = TargetDecision(cur, self.scores[cur] if cur is not None else None)

        ema_c = self._update_ema_c(evidence, seen)  # antes que ema_c_hyst: actualiza cscores
        return {
            "raw": TargetDecision(raw_pid, raw_conf),
            "ema": ema,
            "hyst": hyst,
            "ema_c": ema_c,
            "ema_c_hyst": self._update_ema_c_hyst(social_states, seen),
            "bayes": self._update_bayes(evidence, social_states, seen),
        }

    def selected(self, decisions: dict[str, TargetDecision]) -> TargetDecision:
        return decisions[self.cfg.resolved_mode()]


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
    evidence: dict[int, float] | None = None,
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
        "evidence": {str(p): round(v, 4) for p, v in (evidence or {}).items()},
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
        evidence = {int(p): v for p, v in c.get("evidence", {}).items()} or None
        decisions = tf.update(c["raw_pid"], c["raw_conf"], social, seen, evidence)
        probs = {int(p): v for p, v in c.get("target_probs", {}).items()}
        out.append(chunk_record(
            c["chunk_index"], c["start_frame"], c["raw_pid"], c["raw_conf"], social, seen, decisions, probs,
            source=c.get("source", "vlm"), vlm_pid=c.get("vlm_pid"), evidence=evidence,
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
            mode: track_metrics([c.get(mode, {}).get("pid") for c in chunks], chunk_sec)
            for mode in TARGET_FILTER_MODES
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
    p.add_argument("--stable-window", type=int)
    p.add_argument("--absent-chunks", type=int)
    p.add_argument("--fallback-weight", type=float)
    p.add_argument("--bayes-stay", type=float)
    p.add_argument("--bayes-margin", type=float)
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
        print(f"  {mode:10s} {m}")
    print("  trozo  " + " ".join(f"{m:10s}" for m in TARGET_FILTER_MODES))
    for c in chunks:
        print(f"  {c['chunk_index']:5d}  " + " ".join(f"{c[m]['pid']!s:10s}" for m in TARGET_FILTER_MODES))


if __name__ == "__main__":
    _main()
