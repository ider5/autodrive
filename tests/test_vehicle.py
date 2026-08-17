import numpy as np
import pytest

from vehicle_model import BicycleModel


def test_bicycle_model_constructor_defaults():
    vehicle = BicycleModel()

    assert vehicle.L == 2.0
    assert vehicle.width == 1.8
    assert vehicle.length == 4.0
    assert vehicle.dt == 0.1
    assert vehicle.max_steer == np.deg2rad(20.0)
    assert vehicle.max_v == 7.0
    assert vehicle.min_v == 0.0
    assert vehicle.max_a == 1.5
    assert vehicle.max_delta_dot == np.deg2rad(12.0)
    assert vehicle.max_jerk == 0.8


def test_bicycle_model_set_state_is_returned_by_get_state():
    vehicle = BicycleModel()

    vehicle.set_state(1.0, 2.0, 0.3, 4.0)

    assert vehicle.get_state() == [1.0, 2.0, 0.3, 4.0]


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (3.0 * np.pi, np.pi),
        (-3.0 * np.pi, -np.pi),
    ],
)
def test_bicycle_model_normalize_angle_maps_into_closed_pi_interval(
    angle, expected
):
    vehicle = BicycleModel()

    normalized = vehicle.normalize_angle(angle)

    assert normalized == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert -np.pi <= normalized <= np.pi


def test_bicycle_model_update_moves_straight_without_limiter_changes():
    vehicle = BicycleModel(x=0.0, y=0.0, yaw=0.0, v=1.0)

    vehicle.update(0.0, 0.0)

    assert vehicle.x == pytest.approx(0.1, rel=1e-12, abs=1e-12)
    assert vehicle.y == pytest.approx(0.0, rel=1e-12, abs=1e-12)
    assert vehicle.yaw == pytest.approx(0.0, rel=1e-12, abs=1e-12)
    assert vehicle.v == pytest.approx(1.0, rel=1e-12, abs=1e-12)


def test_bicycle_model_update_limits_acceleration_jerk():
    vehicle = BicycleModel(v=1.0)

    vehicle.update(1.5, 0.0)

    assert vehicle.a == pytest.approx(0.08, rel=1e-12, abs=1e-12)
    assert vehicle.v == pytest.approx(1.008, rel=1e-12, abs=1e-12)


def test_bicycle_model_update_limits_steering_rate():
    vehicle = BicycleModel(v=1.0)

    vehicle.update(0.0, np.deg2rad(20.0))

    assert vehicle.delta == pytest.approx(
        np.deg2rad(12.0) * 0.1, rel=1e-12, abs=1e-12
    )


def test_bicycle_model_update_clips_speed_to_maximum():
    vehicle = BicycleModel(v=6.9)

    for _ in range(100):
        vehicle.update(100.0, 0.0)
        assert vehicle.v <= 7.0

    assert vehicle.v == 7.0


def test_bicycle_model_update_clips_speed_to_minimum():
    vehicle = BicycleModel(v=0.1)

    for _ in range(100):
        vehicle.update(-100.0, 0.0)
        assert vehicle.v >= 0.0

    assert vehicle.v == 0.0


def test_bicycle_model_update_scales_acceleration_at_high_speed():
    vehicle = BicycleModel(v=0.0)
    for _ in range(19):
        vehicle.update(vehicle.max_a, 0.0)
    assert vehicle.prev_a == vehicle.max_a

    # Preserve the warmed-up acceleration limiter while isolating the v=6 scaling.
    vehicle.set_state(0.0, 0.0, 0.0, 6.0)
    speed_factor = 1.0 - 0.3 * 0.5
    expected_acceleration = vehicle.max_a * speed_factor

    vehicle.update(vehicle.max_a, 0.0)

    assert vehicle.a == pytest.approx(
        expected_acceleration, rel=1e-12, abs=1e-12
    )
    assert vehicle.v == pytest.approx(
        6.0 + expected_acceleration * vehicle.dt,
        rel=1e-12,
        abs=1e-12,
    )


def test_bicycle_model_update_returns_current_state_tuple():
    vehicle = BicycleModel(x=1.0, y=2.0, yaw=0.1, v=3.0)

    result = vehicle.update(0.0, 0.0)

    assert result == (vehicle.x, vehicle.y, vehicle.yaw, vehicle.v)
