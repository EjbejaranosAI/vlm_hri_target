"""reconcile_social_with_action: MOVING solo se acepta si la propia acción
tiene indicios de caminar/correr -- un 's':M del VLM inconsistente con su
propio 'a' (p. ej. a="eating", s="M") no debe pasar tal cual."""

from vlm_hri.social_state import (
    SOCIAL_ATTENTIVE,
    SOCIAL_AVAILABLE,
    SOCIAL_BUSY,
    SOCIAL_MOVING,
    coerce_action_social,
    reconcile_social_with_action,
)


def test_moving_kept_when_action_says_walking():
    assert reconcile_social_with_action("walking", SOCIAL_MOVING) == SOCIAL_MOVING
    assert reconcile_social_with_action("running", SOCIAL_MOVING) == SOCIAL_MOVING
    assert reconcile_social_with_action("walking and talking", SOCIAL_MOVING) == SOCIAL_MOVING


def test_moving_downgraded_when_action_does_not_describe_movement():
    # "eating"/"reading" no mapean a ningún trait reconocido (ni walking,
    # talking, phone, smiling, sitting, standing) -- antes esto devolvía el
    # 's' crudo del VLM tal cual, aunque dijera M sin ninguna base en 'a'.
    assert reconcile_social_with_action("eating", SOCIAL_MOVING) != SOCIAL_MOVING
    assert reconcile_social_with_action("reading", SOCIAL_MOVING) != SOCIAL_MOVING
    # _never_unknown en resolve_social_states cae a AVAILABLE en producción;
    # aquí (llamada directa) el resultado intermedio es UNKNOWN.


def test_non_moving_social_states_unaffected_by_the_fix():
    assert reconcile_social_with_action("eating", SOCIAL_ATTENTIVE) == SOCIAL_ATTENTIVE
    assert reconcile_social_with_action("eating", SOCIAL_AVAILABLE) == SOCIAL_AVAILABLE


def test_bare_compact_letter_in_action_field_redirects_to_social():
    # Si el VLM no unió accion+social con una coma ("accion,M") y solo queda
    # la letra suelta en el campo de accion, se redirige a social en vez de
    # quedar como texto de accion literal ("M").
    assert coerce_action_social("M", "UNKNOWN") == ("unknown", "MOVING")
    assert coerce_action_social("attentive", "UNKNOWN") == ("unknown", "ATTENTIVE")


def test_activity_words_that_overlap_social_aliases_stay_as_actions():
    # "phone"/"reading" son acciones válidas y específicas (justo lo pedido
    # en el prompt) -- no deben tratarse como vocabulario social colado solo
    # porque también existen como alias en normalize_social_state.
    assert coerce_action_social("reading", "B") == ("reading", "B")
    assert coerce_action_social("phone", "UNKNOWN") == ("phone", "UNKNOWN")
    assert coerce_action_social("using phone", SOCIAL_BUSY) == ("using phone", SOCIAL_BUSY)
