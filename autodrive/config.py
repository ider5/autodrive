from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class VehicleConfig:
    length: float = 4.0
    width: float = 1.8
    wheelbase: float = 2.0
    dt: float = 0.1
    max_steer_deg: float = 20.0
    max_v: float = 7.0
    min_v: float = 0.0
    max_a: float = 1.5
    max_delta_dot_deg: float = 12.0
    max_jerk: float = 0.8
    high_speed_accel_threshold: float = 5.0


@dataclass(frozen=True)
class ObstacleSpec:
    x: float
    lane_id: int


@dataclass(frozen=True)
class ScenarioConfig:
    road_length: float = 85.0
    num_lanes: int = 3
    lane_width_over_vehicle_width: float = 2.2
    start_x: float = 9.71
    start_lane: int = 3
    end_x: float = 80.0
    end_lane: int = 1
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    obstacles: Tuple[ObstacleSpec, ...] = (
        ObstacleSpec(25.0, 3),
        ObstacleSpec(48.27, 2),
        ObstacleSpec(48.27, 1),
        ObstacleSpec(60.0, 1),
    )


@dataclass(frozen=True)
class SimulationConfig:
    target_speed: float = 4.0
    sim_time: float = 30.0
    max_collision_count: int = 15
    goal_distance: float = 2.0
    dt: float = 0.1


@dataclass(frozen=True)
class RRTPlannerConfig:
    step_size: float = 1.0
    max_iter: int = 20000
    goal_sample_rate: int = 30
    max_turn_angle: float = 10
    safety_distance: float = 1.5
    smoothness: float = 0.3


@dataclass(frozen=True)
class AStarPlannerConfig:
    grid_resolution: float = 0.5
    safety_distance: float = 1.5


@dataclass(frozen=True)
class RRTStarPlannerConfig:
    step_size: float = 1.5
    max_iter: int = 25000
    goal_sample_rate: int = 20
    safety_distance: float = 1.7
    rewire_radius: float = 3.0


@dataclass(frozen=True)
class PurePursuitConfig:
    dt: float = 0.1
    horizon: int = 8


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.1
    horizon: int = 1


@dataclass(frozen=True)
class StanleyConfig:
    dt: float = 0.1
    horizon: int = 8
