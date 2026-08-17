import numpy as np

from autodrive.config import (
    AStarPlannerConfig,
    MPCConfig,
    ObstacleSpec,
    PurePursuitConfig,
    RRTPlannerConfig,
    RRTStarPlannerConfig,
    ScenarioConfig,
    SimulationConfig,
    StanleyConfig,
    VehicleConfig,
)
from autodrive.environment import Environment
from autodrive.vehicle import BicycleModel


def test_config_defaults_match_current_simulation_values():
    assert VehicleConfig() == VehicleConfig(
        length=4.0,
        width=1.8,
        wheelbase=2.0,
        dt=0.1,
        max_steer_deg=20.0,
        max_v=7.0,
        min_v=0.0,
        max_a=1.5,
        max_delta_dot_deg=12.0,
        max_jerk=0.8,
        high_speed_accel_threshold=5.0,
    )
    assert ScenarioConfig() == ScenarioConfig(
        road_length=85.0,
        num_lanes=3,
        lane_width_over_vehicle_width=2.2,
        start_x=9.71,
        start_lane=3,
        end_x=80.0,
        end_lane=1,
        vehicle=VehicleConfig(),
        obstacles=(
            ObstacleSpec(25.0, 3),
            ObstacleSpec(48.27, 2),
            ObstacleSpec(48.27, 1),
            ObstacleSpec(60.0, 1),
        ),
    )
    assert SimulationConfig() == SimulationConfig(
        target_speed=4.0,
        sim_time=30.0,
        max_collision_count=15,
        goal_distance=2.0,
        dt=0.1,
    )
    assert RRTPlannerConfig() == RRTPlannerConfig(
        step_size=1.0,
        max_iter=20000,
        goal_sample_rate=30,
        max_turn_angle=10,
        safety_distance=1.5,
        smoothness=0.3,
    )
    assert AStarPlannerConfig() == AStarPlannerConfig(
        grid_resolution=0.5,
        safety_distance=1.5,
    )
    assert RRTStarPlannerConfig() == RRTStarPlannerConfig(
        step_size=1.5,
        max_iter=25000,
        goal_sample_rate=20,
        safety_distance=1.7,
        rewire_radius=3.0,
    )
    assert PurePursuitConfig() == PurePursuitConfig(dt=0.1, horizon=8)
    assert MPCConfig() == MPCConfig(dt=0.1, horizon=1)
    assert StanleyConfig() == StanleyConfig(dt=0.1, horizon=8)


def test_default_and_explicit_default_environment_geometry_match():
    default_environment = Environment()
    explicit_environment = Environment(ScenarioConfig())

    assert default_environment.road_length == explicit_environment.road_length
    assert default_environment.lane_width == explicit_environment.lane_width
    np.testing.assert_array_equal(
        default_environment.start_point, explicit_environment.start_point
    )
    np.testing.assert_array_equal(
        default_environment.end_point, explicit_environment.end_point
    )
    assert [
        (obstacle.x, obstacle.y)
        for obstacle in default_environment.obstacle_vehicles
    ] == [
        (obstacle.x, obstacle.y)
        for obstacle in explicit_environment.obstacle_vehicles
    ]


def test_environment_uses_custom_scenario_geometry():
    environment = Environment(
        ScenarioConfig(road_length=40.0, start_x=5.0, end_x=30.0)
    )

    assert environment.road_length == 40.0
    assert environment.start_point[0] == 5.0
    assert environment.end_point[0] == 30.0


def test_environment_extends_lane_colors_for_custom_lane_count():
    environment = Environment(ScenarioConfig(num_lanes=5, start_lane=5))

    assert len(environment.lane_colors) == 5


def test_bicycle_model_uses_vehicle_config_with_dt_argument_precedence():
    configured_vehicle = BicycleModel(
        dt=0.2,
        config=VehicleConfig(max_v=3.0, dt=0.05),
    )

    assert configured_vehicle.max_v == 3.0
    assert configured_vehicle.dt == 0.2
    assert BicycleModel().max_v == 7.0
