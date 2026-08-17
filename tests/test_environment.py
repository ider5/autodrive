import numpy as np
import pytest

from autodrive.environment import Environment, ObstacleVehicle


def test_environment_default_geometry_and_lane_centers():
    environment = Environment()

    assert environment.road_length == 85.0
    assert environment.vehicle_length == 4.0
    assert environment.vehicle_width == 1.8
    assert environment.lane_width == 1.8 * 2.2
    assert environment.lane_width == pytest.approx(3.96, rel=1e-12, abs=1e-12)
    assert environment.num_lanes == 3
    assert environment.road_width == environment.lane_width * 3
    assert environment.get_lane_center(1) == 0.5 * environment.lane_width
    assert environment.get_lane_center(2) == 1.5 * environment.lane_width
    assert environment.get_lane_center(3) == 2.5 * environment.lane_width
    assert environment.get_lane_center(0) is None
    assert environment.get_lane_center(4) is None


def test_environment_default_start_and_end_points():
    environment = Environment()

    np.testing.assert_array_equal(
        environment.start_point,
        np.array([9.71, environment.get_lane_center(3)]),
    )
    np.testing.assert_array_equal(
        environment.end_point,
        np.array([80.0, environment.get_lane_center(1)]),
    )


def test_environment_default_obstacles():
    environment = Environment()
    expected_positions = [
        (25.0, environment.get_lane_center(3)),
        (48.27, environment.get_lane_center(2)),
        (48.27, environment.get_lane_center(1)),
        (60.0, environment.get_lane_center(1)),
    ]

    assert len(environment.obstacle_vehicles) == 4
    for obstacle, (expected_x, expected_y) in zip(
        environment.obstacle_vehicles, expected_positions
    ):
        assert obstacle.x == expected_x
        assert obstacle.y == expected_y
        assert obstacle.length == 4.0
        assert obstacle.width == 1.8
        assert obstacle.angle == 0.0


def test_environment_bounds_include_road_edges():
    environment = Environment()

    assert environment.is_out_of_bounds(0, 0) is False
    assert environment.is_out_of_bounds(-0.1, 1) is True
    assert environment.is_out_of_bounds(85.0, 0) is False
    assert environment.is_out_of_bounds(85.1, 0) is True


def test_environment_lane_membership_uses_requested_lane():
    environment = Environment()

    assert environment.is_within_lane(10.0, environment.get_lane_center(1), 1) is True
    assert environment.is_within_lane(10.0, environment.get_lane_center(2), 1) is False


def test_environment_obstacle_update_is_no_op():
    environment = Environment()
    before = [
        (obstacle.x, obstacle.y, obstacle.angle)
        for obstacle in environment.obstacle_vehicles
    ]

    environment.update_obstacles(0.1)

    after = [
        (obstacle.x, obstacle.y, obstacle.angle)
        for obstacle in environment.obstacle_vehicles
    ]
    assert after == before


def test_circle_collision_is_false_on_empty_lane_stretch():
    environment = Environment()

    assert (
        environment.is_collision(
            10.0, environment.get_lane_center(2), radius=0.5
        )
        is False
    )


def test_circle_collision_detects_obstacle_center():
    environment = Environment()

    assert (
        environment.is_collision(
            25.0, environment.get_lane_center(3), radius=1.5
        )
        is True
    )


def test_circle_collision_detects_lower_road_boundary():
    environment = Environment()

    assert environment.is_collision(10.0, 0.4, radius=0.5) is True


def test_circle_collision_detects_upper_road_boundary():
    environment = Environment()

    assert (
        environment.is_collision(
            10.0, environment.road_width - 0.4, radius=0.5
        )
        is True
    )


def test_rectangle_collision_is_false_at_start_point():
    environment = Environment()

    assert (
        environment.is_collision(
            environment.start_point[0],
            environment.start_point[1],
            radius=0,
            angle=0,
        )
        is False
    )


def test_rectangle_collision_detects_obstacle_center(capsys):
    environment = Environment()

    assert (
        environment.is_collision(
            25.0,
            environment.get_lane_center(3),
            radius=0,
            angle=0,
        )
        is True
    )
    capsys.readouterr()


def test_rectangle_collision_detects_position_below_road(capsys):
    environment = Environment()

    assert environment.is_collision(10.0, -5.0, radius=0, angle=0) is True
    capsys.readouterr()


def test_obstacle_vehicle_update_position_updates_coordinates_and_optional_angle():
    obstacle = ObstacleVehicle(1.0, 2.0, angle=0.1)

    obstacle.update_position(3.0, 4.0, angle=0.3)

    assert (obstacle.x, obstacle.y, obstacle.angle) == (3.0, 4.0, 0.3)

    obstacle.update_position(5.0, 6.0)

    assert (obstacle.x, obstacle.y, obstacle.angle) == (5.0, 6.0, 0.3)
