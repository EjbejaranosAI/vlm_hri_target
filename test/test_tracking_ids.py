"""track_detections nunca debe dar el mismo ID a dos personas del mismo frame."""

from vlm_hri.config import MAX_TRACK_IDS
from vlm_hri.detection import assign_spatial_ids, track_detections


def _box(x: float) -> dict:
    return {"x1": x, "y1": 0.0, "x2": x + 40.0, "y2": 200.0, "conf": 0.9}


def test_new_people_beyond_cap_get_unique_ids():
    # Multitud más grande que MAX_TRACK_IDS (medido: cola de ~15 personas en
    # esperando.mp4): antes los sobrantes heredaban el ID del más cercano.
    n = MAX_TRACK_IDS + 3
    prev = assign_spatial_ids([_box(100.0 * i) for i in range(MAX_TRACK_IDS)])
    curr = [_box(100.0 * i) for i in range(MAX_TRACK_IDS)] + [_box(5000.0 + 100.0 * i) for i in range(3)]
    out = track_detections(prev, curr)
    pids = [d["pid"] for d in out]
    assert len(pids) == n and len(set(pids)) == n


def test_freed_id_is_reused_but_not_from_previous_frame():
    prev = assign_spatial_ids([_box(0.0), _box(500.0)])  # pids 1, 2
    # La persona 1 sigue, la 2 se fue y aparece alguien lejos: no hereda el 2
    # (estaba en el frame anterior), recibe el 3.
    out = track_detections(prev, [_box(0.0), _box(3000.0)])
    assert sorted(d["pid"] for d in out) == [1, 3]
    # Un frame después el 2 ya está libre y se recicla para otra persona nueva.
    out2 = track_detections(out, [_box(0.0), _box(3000.0), _box(6000.0)])
    assert sorted(d["pid"] for d in out2) == [1, 2, 3]
