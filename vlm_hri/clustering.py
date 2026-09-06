"""Agrupación de personas por proximidad real en el mundo (LiDAR), no por
posición en la imagen. Solo tiene sentido con datos de un tracker externo con
posición 3D/2D real (ver vlm_hri/ros/detections_bridge.py) -- el pipeline de
cámara/video plano no tiene esta señal y nunca llama a este módulo."""

from __future__ import annotations

import os


def _engaged_proximity_max_m() -> float:
    return float(os.environ.get("ENGAGED_PROXIMITY_MAX_M", "2.0"))


def cluster_by_distance(
    world_xy: dict[int, tuple[float, float]], max_dist_m: float | None = None
) -> dict[int, int]:
    """Agrupa pids cuya distancia real (mundo) es <= max_dist_m, transitivamente
    (unión de conjuntos: si A-B y B-C están cerca, A/B/C quedan en el mismo
    cluster aunque A-C no lo estén directamente). Devuelve {pid: cluster_id},
    con un id por cada componente conexa (arbitrario, solo para comparar
    igualdad de cluster entre dos pids)."""
    if max_dist_m is None:
        max_dist_m = _engaged_proximity_max_m()
    pids = list(world_xy.keys())
    parent = {pid: pid for pid in pids}

    def find(p: int) -> int:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(pids)):
        xi, yi = world_xy[pids[i]]
        for j in range(i + 1, len(pids)):
            xj, yj = world_xy[pids[j]]
            if ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5 <= max_dist_m:
                union(pids[i], pids[j])

    roots = {pid: find(pid) for pid in pids}
    # IDs de cluster compactos y estables (orden de aparición), no los pids
    # crudos de la raíz union-find -- más legibles al publicarlos.
    root_to_cluster: dict[int, int] = {}
    out: dict[int, int] = {}
    for pid in pids:
        root = roots[pid]
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        out[pid] = root_to_cluster[root]
    return out


def same_cluster(pid_a: int, pid_b: int, clusters: dict[int, int]) -> bool:
    """True si ambos pids están en el mismo cluster. Si a alguno le falta dato
    de cluster (sin posición del mundo ese trozo), no se puede afirmar que
    estén cerca -- devuelve False (mismo criterio conservador que el resto del
    refinamiento: sin evidencia, no se corrige)."""
    if pid_a not in clusters or pid_b not in clusters:
        return False
    return clusters[pid_a] == clusters[pid_b]
