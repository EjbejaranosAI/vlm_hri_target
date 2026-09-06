"""Tests del puente ROS2 que no necesitan rclpy/vision_msgs instalados --
solo objetos con la misma forma que vision_msgs/Detection2DArray (mismo
patrón de test que dynamic_tracking/test/test_lidar_fusion.py)."""

from types import SimpleNamespace

from vlm_hri.ros.detections_bridge import _parse_class_id, detection2d_array_to_dets


def _make_det(cx, cy, sx, sy, class_id, score, wx, wy):
    return SimpleNamespace(
        bbox=SimpleNamespace(
            center=SimpleNamespace(position=SimpleNamespace(x=cx, y=cy)),
            size_x=sx,
            size_y=sy,
        ),
        results=[
            SimpleNamespace(
                hypothesis=SimpleNamespace(class_id=class_id, score=score),
                pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=wx, y=wy))),
            )
        ],
    )


def test_parse_class_id():
    assert _parse_class_id("person#7") == ("person", 7)
    assert _parse_class_id("person_dyn#3") == ("person_dyn", 3)
    assert _parse_class_id("person") == ("person", None)
    assert _parse_class_id("person#abc") == ("person", None)


def test_detection2d_array_to_dets_filters_and_converts():
    msg = SimpleNamespace(
        detections=[
            _make_det(100, 100, 40, 80, "person#1", 0.9, 1.5, 2.0),
            _make_det(300, 100, 40, 80, "person#2", 0.8, 5.0, 5.0),
            _make_det(50, 50, 20, 20, "chair#9", 0.7, 0.0, 0.0),  # otra clase
            _make_det(10, 10, 5, 5, "person", 0.5, 0.0, 0.0),  # sin track_id
        ]
    )
    dets, world_xy = detection2d_array_to_dets(msg)

    assert len(dets) == 2
    d1 = next(d for d in dets if d["pid"] == 1)
    assert (d1["x1"], d1["y1"], d1["x2"], d1["y2"]) == (80, 60, 120, 140)
    assert d1["class_name"] == "person"
    assert d1["conf"] == 0.9
    assert (d1["world_x"], d1["world_y"]) == (1.5, 2.0)
    assert world_xy == {1: (1.5, 2.0), 2: (5.0, 5.0)}


def test_detection2d_array_to_dets_empty():
    dets, world_xy = detection2d_array_to_dets(SimpleNamespace(detections=[]))
    assert dets == []
    assert world_xy == {}
