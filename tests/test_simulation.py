from types import SimpleNamespace

import pytest

from autodrive.config import SimulationConfig
from autodrive.simulation.loop import run_simulation
from autodrive.vehicle import BicycleModel


class StubController:
    def __init__(self):
        self.target_speed = 4.0
        self.env = None
        self.calls = 0

    def calculate_steering(self, vehicle, path, road_width):
        self.calls += 1
        return 0.0, 0.0

    def set_path(self, path):
        pass


class StubEnvironment:
    road_width = 12.0

    def __init__(self, end_point, collides):
        self.end_point = end_point
        self.collides = collides

    def is_collision(self, *args, **kwargs):
        return self.collides


def run_stub_simulation(*, end_point, collides, config):
    env = StubEnvironment(end_point, collides)
    controller = StubController()
    controller.env = env
    vehicle = BicycleModel(x=0.0, y=0.0, yaw=0.0, v=0.0, dt=config.dt)
    path = [(0.0, 0.0), end_point]

    result = run_simulation(
        env=env,
        planner=SimpleNamespace(safety_distance=1.5),
        vehicle=vehicle,
        controller=controller,
        path=path,
        smooth_path=path,
        end_point=end_point,
        config=config,
        animate=False,
        plot=False,
    )
    return result, controller


def test_run_simulation_stops_at_goal():
    result, controller = run_stub_simulation(
        end_point=(0.0, 0.0),
        collides=False,
        config=SimulationConfig(sim_time=5.0),
    )

    assert result.stop_reason == "goal"
    assert result.distance_to_goal < 2.0
    assert controller.calls == 1


def test_run_simulation_stops_after_fifteen_consecutive_collisions():
    result, controller = run_stub_simulation(
        end_point=(1000.0, 1000.0),
        collides=True,
        config=SimulationConfig(max_collision_count=15),
    )

    assert result.stop_reason == "collision"
    assert result.collision_count == 15
    assert controller.calls == 15


def test_run_simulation_stops_at_timeout():
    result, controller = run_stub_simulation(
        end_point=(1000.0, 1000.0),
        collides=False,
        config=SimulationConfig(sim_time=0.3, dt=0.1),
    )

    assert result.stop_reason == "timeout"
    assert result.time == pytest.approx(0.3)
    assert controller.calls == 3
