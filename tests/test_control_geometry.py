import numpy as np
import pytest

from autodrive.control.geometry import (
    nearest_index,
    path_yaw,
    signed_cross_track,
    wrap_angle,
)


def test_nearest_index_uses_first_minimum():
    path = [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]

    assert nearest_index(path, 1.0, 0.0) == 0


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (2 * np.pi + 0.25, 0.25),
        (-2 * np.pi - 0.25, -0.25),
    ],
)
def test_wrap_angle_maps_to_signed_pi_range(angle, expected):
    assert wrap_angle(angle) == pytest.approx(expected)


def test_path_yaw_uses_segment_direction():
    assert path_yaw([1.0, 1.0], [1.0, 3.0]) == pytest.approx(np.pi / 2)


def test_signed_cross_track_uses_stanley_sign_convention():
    assert signed_cross_track([0.0, 1.0], [0.0, 0.0], [1.0, 0.0]) == pytest.approx(
        -1.0
    )


def test_signed_cross_track_returns_zero_for_degenerate_segment():
    assert signed_cross_track([2.0, 3.0], [1.0, 1.0], [1.0, 1.0]) == 0.0
