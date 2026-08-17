from types import SimpleNamespace

import pytest

from autodrive.config import SimulationConfig
from autodrive.simulation import animation as animation_module
from autodrive.simulation import session as session_module
from autodrive.simulation.animation import create_animation
from autodrive.simulation.loop import run_simulation
from autodrive.simulation.session import _create_controller
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


@pytest.mark.parametrize("controller_choice", [1, 2, 3])
def test_create_controller_uses_vehicle_timestep(controller_choice, capsys):
    controller, _ = _create_controller(
        controller_choice, SimpleNamespace(dt=0.25)
    )
    capsys.readouterr()

    assert controller.dt == 0.25


def test_run_simulation_forwards_config_timestep_to_animation(monkeypatch):
    captured = {}

    def fake_create_animation(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        animation_module, "create_animation", fake_create_animation
    )
    config = SimulationConfig(sim_time=0.0, dt=0.25)
    env = StubEnvironment((10.0, 0.0), False)
    vehicle = BicycleModel(dt=config.dt)
    path = [[0.0, 0.0], [10.0, 0.0]]

    run_simulation(
        env=env,
        planner=SimpleNamespace(safety_distance=1.5),
        vehicle=vehicle,
        controller=StubController(),
        path=path,
        smooth_path=path,
        config=config,
        animate=True,
        plot=False,
        show=False,
    )

    assert captured["dt"] == 0.25


def test_animation_uses_configured_timestep_for_elapsed_time(monkeypatch):
    captured = {}

    def fake_func_animation(
        fig, animate_frame, frames, init_func, blit, interval
    ):
        init_func()
        artists = animate_frame(1)
        captured["info_text"] = artists[1].get_text()
        return SimpleNamespace()

    monkeypatch.setattr(
        animation_module.animation, "FuncAnimation", fake_func_animation
    )
    env = SimpleNamespace(
        road_width=12.0,
        road_length=85.0,
        vehicle_length=4.0,
        vehicle_width=1.8,
        plot_environment=lambda ax: None,
    )

    create_animation(
        env,
        [[0.0, 0.0], [1.0, 0.0]],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        dt=0.3,
        animate=False,
        show=False,
    )

    assert "0.3s" in captured["info_text"]


def test_run_session_without_show_suppresses_planner_window(
    monkeypatch, tmp_path
):
    save_calls = []
    path = [[0.0, 0.0], [1.0, 1.0]]

    class StubPlanner:
        safety_distance = 1.5

        def planning(self, *args):
            return path

        def save_and_show_results(self, saved_path, filename, show=True):
            save_calls.append((saved_path, filename, show))

    env = SimpleNamespace(
        start_point=path[0],
        end_point=path[-1],
        vehicle_width=1.8,
        vehicle_length=4.0,
    )
    monkeypatch.setattr(session_module, "Environment", lambda: env)
    monkeypatch.setattr(
        session_module,
        "_create_planner",
        lambda planning_choice, environment: (
            StubPlanner(),
            SimpleNamespace(),
            str(tmp_path / "path.png"),
        ),
    )
    monkeypatch.setattr(
        session_module,
        "run_simulation",
        lambda **kwargs: SimpleNamespace(stop_reason="goal"),
    )

    def fail_show():
        pytest.fail("run_session(show=False) must not call plt.show()")

    monkeypatch.setattr("matplotlib.pyplot.show", fail_show)

    session_module.run_session(
        3, 1, show=False, plot=False, animate=False
    )

    assert save_calls == [(path, str(tmp_path / "path.png"), False)]
