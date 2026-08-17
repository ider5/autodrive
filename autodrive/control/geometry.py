import numpy as np


def nearest_index(path, x, y) -> int:
    """Linear scan; min hypot. Empty path: callers currently guard len < 2."""
    min_dist = float("inf")
    min_idx = 0
    for index, point in enumerate(path):
        distance = np.hypot(point[0] - x, point[1] - y)
        if distance < min_dist:
            min_dist = distance
            min_idx = index
    return min_idx


def wrap_angle(angle) -> float:
    """Map to [-pi, pi] via arctan2(sin, cos)."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def path_yaw(p1, p2) -> float:
    """arctan2(dy, dx)."""
    return float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))


def signed_cross_track(xy, p1, p2) -> float:
    """Stanley convention: np.cross(xy - p1, unit(p2-p1)).

    Degenerate segment length < 1e-6 returns 0.0.
    """
    xy = np.asarray(xy)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    path_vector = p2 - p1
    path_length = np.linalg.norm(path_vector)
    if path_length < 1e-6:
        return 0.0
    offset = xy - p1
    path_unit = path_vector / path_length
    return float(
        np.cross(
            [offset[0], offset[1], 0.0],
            [path_unit[0], path_unit[1], 0.0],
        )[2]
    )
