"""parse_vlm_response/parse_vlm_target: respuestas reales de Qwen2-VL-2B
(claves sin comillas, "idle" como acción) y el target elegido por el VLM."""

from vlm_hri.social_state import SOCIAL_ATTENTIVE, SOCIAL_BUSY, SOCIAL_UNKNOWN
from vlm_hri.vlm.parsing import load_vlm_json, parse_vlm_response, parse_vlm_target


def test_unquoted_numeric_key_is_parsed():
    # Respuesta real medida en vivo (34/35 trozos): no es JSON válido.
    raw = '{1: {"a": "idle", "s": "ATTENTIVE"}}'
    assert load_vlm_json(raw) == {"1": {"a": "idle", "s": "ATTENTIVE"}}
    actions, social, _, _ = parse_vlm_response(raw, [1])
    assert actions == {1: "idle"}
    assert social == {1: SOCIAL_ATTENTIVE}


def test_idle_is_a_valid_action():
    actions, _, _, _ = parse_vlm_response('{"1": {"a": "idle", "s": "AVAILABLE"}}', [1])
    assert actions == {1: "idle"}


def test_intention_outside_the_five_is_unknown():
    _, social, _, _ = parse_vlm_response('{1: {"a": "idle", "s": "thinking"}}', [1])
    assert social == {1: SOCIAL_UNKNOWN}


def test_target_and_confidence():
    raw = (
        '{"1": {"a": "using phone", "s": "BUSY"}, "2": {"a": "waving", "s": "ATTENTIVE"}, '
        '"target": 2, "target_conf": 0.8}'
    )
    actions, social, _, _ = parse_vlm_response(raw, [1, 2])
    assert actions == {1: "using phone", 2: "waving"}
    assert social[1] == SOCIAL_BUSY
    assert parse_vlm_target(raw, [1, 2]) == (2, 0.8)


def test_target_with_unquoted_keys_and_null():
    assert parse_vlm_target('{1: {"a": "reading", "s": "BUSY"}, target: null, target_conf: 0.9}', [1]) == (None, None)
    assert parse_vlm_target('{1: {"a": "idle", "s": "AVAILABLE"}, target: 1, target_conf: 0.6}', [1]) == (1, 0.6)


def test_target_id_label_and_percentage():
    assert parse_vlm_target('{"target": "ID3", "target_conf": 75}', [3]) == (3, 0.75)


def test_target_not_in_people_is_rejected():
    assert parse_vlm_target('{"target": 9, "target_conf": 0.9}', [1, 2]) == (None, None)


def test_truncated_response_keeps_people_but_loses_target():
    raw = '{"1": {"a": "watching camera", "s": "ATTENTIVE"}, "target": 1, "target_co'
    actions, social, _, _ = parse_vlm_response(raw, [1])
    assert actions == {1: "watching camera"}
    assert social == {1: SOCIAL_ATTENTIVE}
    assert parse_vlm_target(raw, [1])[0] == 1


def test_target_nested_inside_person_object():
    raw = '{1: {"a": "idle", "s": "AVAILABLE", "target": 1, "target_conf": 0.7}}'
    assert parse_vlm_target(raw, [1]) == (1, 0.7)
    assert parse_vlm_target('{1: {"a": "idle", "s": "AVAILABLE", "target": true}}', [1]) == (1, None)
    assert parse_vlm_target('{1: {"a": "Moving", "s": "Available", "target": null}}', [1]) == (None, None)


class _FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.index(text)]


def _step(probs):
    import torch

    return torch.log(torch.tensor([probs]))


def test_target_probability_renormalized_over_candidates():
    from vlm_hri.vlm.inference import target_token_probability

    # Salida: `"target": "1"}` -- en el dígito, 'ID' (formato) se lleva 0.5.
    vocab = ['"target": "', "ID", "1", "2", '"}', "3"]
    gen_ids = [0, 2, 4]
    scores = (
        _step([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]),
        _step([0.02, 0.50, 0.36, 0.09, 0.02, 0.01]),  # '1'=0.36 vs '2'=0.09
        _step([0.01, 0.01, 0.01, 0.01, 0.95, 0.01]),
    )
    tok = _FakeTokenizer(vocab)
    # Candidatos 1 y 2: 0.36 / (0.36 + 0.09) = 0.8 (la masa de 'ID' no cuenta)
    assert abs(target_token_probability(tok, gen_ids, scores, 1, [1, 2]) - 0.8) < 1e-6
    # Una sola persona: 1.0 por definición
    assert abs(target_token_probability(tok, gen_ids, scores, 1, [1]) - 1.0) < 1e-6
    # ID distinto al que aparece en la salida -> no se inventa una probabilidad
    assert target_token_probability(tok, gen_ids, scores, 3, [1, 3]) is None


def test_target_probability_multi_digit_id():
    from vlm_hri.vlm.inference import target_token_probability

    # Salida: `"target": "ID12"}`
    vocab = ['"target": "', "ID", "1", "2", '"}', "3"]
    gen_ids = [0, 1, 2, 3, 4]
    scores = (
        _step([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]),
        _step([0.01, 0.95, 0.01, 0.01, 0.01, 0.01]),
        _step([0.05, 0.05, 0.60, 0.05, 0.05, 0.20]),  # '1' vs '3' (IDs 12 y 3)
        _step([0.05, 0.05, 0.05, 0.50, 0.05, 0.30]),  # '2' tal cual: 0.5
        _step([0.01, 0.01, 0.01, 0.01, 0.95, 0.01]),
    )
    p = target_token_probability(_FakeTokenizer(vocab), gen_ids, scores, 12, [12, 3])
    assert abs(p - (0.60 / 0.80) * 0.5) < 1e-6


def test_target_flag_per_person():
    from vlm_hri.vlm.parsing import parse_vlm_target_flags

    raw = '{1: {"a": "talking", "s": "ENGAGED", "t": 0}, 2: {"a": "waiting", "s": "AVAILABLE", "t": 1}}'
    assert parse_vlm_target_flags(raw, [1, 2]) == [2]
    assert parse_vlm_target(raw, [1, 2]) == (2, None)


def test_target_flag_all_zero_means_nobody():
    from vlm_hri.vlm.parsing import parse_vlm_target_flags

    raw = '{"1": {"a": "talking", "s": "ENGAGED", "t": 0}, "2": {"a": "talking", "s": "ENGAGED", "t": 0}}'
    assert parse_vlm_target_flags(raw, [1, 2]) == []
    assert parse_vlm_target(raw, [1, 2]) == (None, None)


def test_target_flag_absent_falls_back_to_global_target():
    from vlm_hri.vlm.parsing import parse_vlm_target_flags

    raw = '{"1": {"a": "idle", "s": "AVAILABLE"}, "target": 1}'
    assert parse_vlm_target_flags(raw, [1]) is None
    assert parse_vlm_target(raw, [1])[0] == 1


def test_target_flag_truncated_response_regex_fallback():
    from vlm_hri.vlm.parsing import parse_vlm_target_flags

    raw = '{"1": {"a": "idle", "s": "AVAILABLE", "t": 1}, "2": {"a": "phone", "s": "BU'
    assert parse_vlm_target_flags(raw, [1, 2]) == [1]


def test_target_flag_ignores_unknown_pid():
    from vlm_hri.vlm.parsing import parse_vlm_target_flags

    assert parse_vlm_target_flags('{"9": {"a": "idle", "s": "AVAILABLE", "t": 1}}', [1]) is None


def test_target_flag_probabilities_from_logits():
    from vlm_hri.vlm.inference import target_flag_probabilities

    # Salida: `"1": {"t": 0}, "2": {"t": 1}}`
    vocab = ['"1": {"t": ', "0", "1", '}, "2": {"t": ', "}}", "x"]
    tok = _FakeTokenizer(vocab)
    gen_ids = [0, 1, 3, 2, 4]
    scores = (
        _step([0.9, 0.02, 0.02, 0.02, 0.02, 0.02]),
        _step([0.02, 0.6, 0.2, 0.06, 0.06, 0.06]),  # flag de 1: P(1)=0.2/(0.6+0.2)=0.25
        _step([0.02, 0.02, 0.02, 0.9, 0.02, 0.02]),
        _step([0.02, 0.1, 0.3, 0.2, 0.2, 0.18]),  # flag de 2: P(1)=0.3/(0.1+0.3)=0.75
        _step([0.02, 0.02, 0.02, 0.02, 0.9, 0.02]),
    )
    probs = target_flag_probabilities(tok, gen_ids, scores, [1, 2])
    assert abs(probs[1] - 0.25) < 1e-6
    assert abs(probs[2] - 0.75) < 1e-6
