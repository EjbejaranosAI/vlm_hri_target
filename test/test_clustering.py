"""vlm_hri/clustering.py y vlm_hri/pose/gait.py::check_engaged_proximity."""

from vlm_hri.clustering import cluster_by_distance, same_cluster
from vlm_hri.pose.gait import check_engaged_proximity
from vlm_hri.social_state import SOCIAL_ATTENTIVE, SOCIAL_ENGAGED


def test_cluster_by_distance_groups_close_points():
    world = {1: (0.0, 0.0), 2: (0.4, 0.3), 3: (5.0, 5.0)}
    clusters = cluster_by_distance(world, max_dist_m=2.0)
    assert same_cluster(1, 2, clusters)
    assert not same_cluster(1, 3, clusters)
    assert not same_cluster(2, 3, clusters)


def test_cluster_by_distance_is_transitive():
    world = {1: (0.0, 0.0), 2: (1.8, 0.0), 3: (3.6, 0.0)}
    clusters = cluster_by_distance(world, max_dist_m=2.0)
    assert same_cluster(1, 3, clusters), "1 and 3 should merge transitively via 2"


def test_same_cluster_missing_pid_is_conservative():
    clusters = cluster_by_distance({1: (0.0, 0.0)}, max_dist_m=2.0)
    assert not same_cluster(1, 99, clusters)


def test_check_engaged_proximity_noop_without_world_xy():
    social = {1: SOCIAL_ENGAGED, 2: SOCIAL_ENGAGED}
    out = check_engaged_proximity({1: "talking", 2: "talking"}, social, [1, 2], None, None)
    assert out == social


def test_check_engaged_proximity_keeps_engaged_when_close():
    social = {1: SOCIAL_ENGAGED, 2: SOCIAL_ENGAGED}
    world = {1: (0.0, 0.0), 2: (0.5, 0.5)}
    clusters = cluster_by_distance(world, max_dist_m=2.0)
    out = check_engaged_proximity({1: "talking", 2: "talking"}, social, [1, 2], world, clusters)
    assert out[1] == SOCIAL_ENGAGED and out[2] == SOCIAL_ENGAGED


def test_check_engaged_proximity_downgrades_when_far():
    social = {1: SOCIAL_ENGAGED, 2: SOCIAL_ENGAGED}
    world = {1: (0.0, 0.0), 2: (10.0, 10.0)}
    clusters = cluster_by_distance(world, max_dist_m=2.0)
    out = check_engaged_proximity({1: "talking", 2: "talking"}, social, [1, 2], world, clusters)
    assert out[1] == SOCIAL_ATTENTIVE and out[2] == SOCIAL_ATTENTIVE


def test_check_engaged_proximity_downgrades_lone_engaged():
    social = {1: SOCIAL_ENGAGED, 2: SOCIAL_ATTENTIVE}
    world = {1: (0.0, 0.0), 2: (0.5, 0.5)}
    clusters = cluster_by_distance(world, max_dist_m=2.0)
    out = check_engaged_proximity({1: "talking", 2: "standing"}, social, [1, 2], world, clusters)
    assert out[1] == SOCIAL_ATTENTIVE
