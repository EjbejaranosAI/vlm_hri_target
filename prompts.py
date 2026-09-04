"""Plantillas de prompt para el VLM — texto estático optimizado para
token-efficiency y precisión, separado de la lógica de construcción/parseo en
``pipeline.py``.

Cada plantilla se rellena con ``str.format(**kwargs)`` en tiempo de ejecución
con los datos dinámicos de cada petición (mapeo de IDs detectados, claves JSON
requeridas, ejemplo, etc.). Las llaves dobles ``{{`` / ``}}`` son llaves
literales de JSON (se convierten en ``{`` / ``}`` al formatear); las llaves
simples como ``{clip}`` son los marcadores que ``pipeline.py`` rellena.
"""

from __future__ import annotations

# —— Modo compacto (mínimo consumo de tokens + alta precisión) ——————————

COMPACT_WITH_SOCIAL = (
    "{clip}. Map: {id_map}.\n"
    "Keys required: {keys}. Return JSON format: {{{example}}}\n"
    "Rules per person ID:\n"
    "- 'a' (action): ONLY 'walk' (if moving), OR 'sit' (legs bent visible), OR 'talk' (mouth moving), OR 'stand', OR 'stand and talk'. Priority: walk > sit > talk > stand. "
    "Evaluate EACH person independently for 'talk' by THEIR OWN mouth — in a conversation, mark BOTH/ALL speakers who are visibly talking, not just one.\n"
    "- 's' (social): A (idle/not focused), T (facing camera), B (phone/reading/solo), E (talking to person), M (moving/walking).\n"
    "- If cropped/no legs visible -> assume 'stand', NOT 'sit'.\n"
    "Reply ONLY raw JSON. No markdown wrappers or extra text."
)

COMPACT_NO_SOCIAL = (
    "{clip}. Map: {id_map}.\n"
    "Keys required: {keys}. Output ONLY raw JSON: {{{example}}}.\n"
    "Describe each person in 1-3 words."
)

# —— Modo completo (instrucciones estructuradas por jerarquía) ——————————

FULL_WITH_SOCIAL = (
    "Context: {ctx} ({n_people} people with purple ID<number> boxes).\n"
    "{motion_hint}"
    "Mapping (Box Label -> Key):\n{mapping}\n\n"
    "Generate JSON for keys [{keys}] using schema {{{slots}}}:\n"
    "1. 'action':\n"
    "   - IF moving/walking -> 'walking' ONLY (do not combine).\n"
    "   - IF sitting -> 'sitting' (MUST see bent legs/seat, else default to 'standing').\n"
    "   - IF talking -> 'talking' (MUST see mouth moving).\n"
    "   - Combine non-moving actions with 'and' (e.g., 'standing and smiling').\n"
    "2. 'social': EXACTLY one of [ATTENTIVE, AVAILABLE, BUSY, ENGAGED, MOVING].\n"
    "   - ATTENTIVE: Looking/oriented at camera.\n"
    "   - AVAILABLE: Idle, not looking at camera/others.\n"
    "   - BUSY: Phone/reading/solo task.\n"
    "   - ENGAGED: Face-to-face talk with another.\n"
    "   - MOVING: Walking/in transit.\n\n"
    "Constraint: Do NOT put social states inside 'action'. Output ONLY raw JSON."
)

FULL_NO_SOCIAL = (
    "Context: {ctx} ({n_people} people with purple ID<number> boxes).\n"
    "{motion_hint}"
    "Mapping:\n{mapping}\n\n"
    "Describe action for keys [{keys}] in 1-3 words.\n"
    "Output ONLY raw JSON format: {{{slots}}}."
)

NO_PEOPLE = "{}"

MOTION_HINT_VIDEO = "Note: Track gait across clip. Do not label walking as standing from a single static frame.\n"

# Usado por pose_pipeline.py (variante con detector de pose): explica el color
# de la caja como una pista de un sensor de movimiento INDEPENDIENTE del VLM
# (marcha real de piernas, no el propio juicio visual del VLM sobre esa caja).
MOTION_COLOR_HINT = (
    "Sensor hint: BLUE box = walking detected; RED box = stationary. "
    "Trust hint unless visual evidence strictly contradicts it.\n"
)
