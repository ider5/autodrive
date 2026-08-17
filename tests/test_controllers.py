import pytest
import scipy.optimize as scipy_optimize

from environment import Environment
from mpc_controller import MPCController
from pure_pursuit_controller import CompatibleController
from stanley_controller import CompatibleStanleyController
from vehicle_model import BicycleModel


pytestmark = pytest.mark.filterwarnings(
    "ignore:Arrays of 2-dimensional vectors are deprecated:DeprecationWarning"
)

lane2 = Environment().get_lane_center(2)  # 1.5 * 3.96
PATH = [[float(x), float(lane2)] for x in range(0, 81, 2)]


def _vehicle(y=lane2):
    return BicycleModel(x=10.0, y=y, yaw=0.0, v=2.0, dt=0.1)


def _controller(controller_name):
    if controller_name == "pure_pursuit":
        controller = CompatibleController(dt=0.1, horizon=8)
    elif controller_name == "mpc":
        controller = MPCController(dt=0.1, horizon=1)
    else:
        controller = CompatibleStanleyController(dt=0.1, horizon=8)
    controller.target_speed = 4.0
    return controller


@pytest.mark.parametrize(
    ("controller_name", "expected"),
    [
        ("pure_pursuit", (0.0, 1.02)),
        ("mpc", (0.0, 0.5)),
        ("stanley", (0.0, 0.29999999999999993)),
    ],
)
def test_controller_output_on_path_at_mid_speed(
    controller_name, expected, capsys
):
    env = Environment()
    controller = _controller(controller_name)
    controller.set_path(PATH)

    actual = controller.calculate_steering(_vehicle(), PATH, env.road_width)

    capsys.readouterr()
    assert actual == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("controller_name", "expected"),
    [
        ("pure_pursuit", (-0.14497866312686414, 0.5492307692307693)),
        ("mpc", (0.08821898549276957, 0.5)),
        ("stanley", (-0.0074984380856760475, 0.29999999999999993)),
    ],
)
def test_controller_output_with_lateral_offset(
    controller_name, expected, capsys
):
    env = Environment()
    controller = _controller(controller_name)
    controller.set_path(PATH)

    actual = controller.calculate_steering(
        _vehicle(y=lane2 + 1.0), PATH, env.road_width
    )

    capsys.readouterr()
    assert actual == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("controller_name", "expected"),
    [
        ("pure_pursuit", (0.0, 0.0)),
        ("mpc", (0.0, 0.8)),
        ("stanley", (0.0, 0.0)),
    ],
)
def test_controller_output_with_empty_path(controller_name, expected, capsys):
    env = Environment()
    controller = _controller(controller_name)
    controller.set_path(None)

    actual = controller.calculate_steering(_vehicle(), None, env.road_width)

    capsys.readouterr()
    assert actual == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("controller_name", "expected"),
    [
        ("pure_pursuit", (0.0, 0.0)),
        ("mpc", (0.0, 0.8)),
        ("stanley", (0.0, 0.0)),
    ],
)
def test_controller_output_with_one_point_path(
    controller_name, expected, capsys
):
    env = Environment()
    path = [[0.0, lane2]]
    controller = _controller(controller_name)
    controller.set_path(path)

    actual = controller.calculate_steering(_vehicle(), path, env.road_width)

    capsys.readouterr()
    assert actual == pytest.approx(expected, abs=1e-10)


def test_stanley_boundary_protection_output(capsys):
    env = Environment()
    controller = _controller("stanley")
    controller.env = env
    controller.set_path(PATH)

    actual = controller.calculate_steering(
        _vehicle(y=0.4), PATH, env.road_width
    )

    captured = capsys.readouterr()
    assert "边界保护" in captured.out
    assert actual == pytest.approx(
        (-0.10471975511965978, 0.29999999999999993), abs=1e-10
    )


def test_pure_pursuit_near_boundary_output(capsys):
    env = Environment()
    controller = _controller("pure_pursuit")
    controller.env = env
    controller.set_path(PATH)

    actual = controller.calculate_steering(
        _vehicle(y=0.4), PATH, env.road_width
    )

    capsys.readouterr()
    assert actual == pytest.approx(
        (-0.17453292519943295, -0.8999999999999999), abs=1e-10
    )


def test_mpc_hardcoded_upper_boundary_output(capsys):
    env = Environment()
    controller = _controller("mpc")
    controller.env = env
    controller.set_path(PATH)

    actual = controller.calculate_steering(
        _vehicle(y=11.5), PATH, env.road_width
    )

    capsys.readouterr()
    assert actual == pytest.approx((-0.2181661564992912, 0.5), abs=1e-10)


def test_stanley_target_speed_is_capped_at_maximum(capsys):
    controller = CompatibleStanleyController(dt=0.1, horizon=8)

    controller.set_target_speed(10.0)

    capsys.readouterr()
    assert controller.target_speed == 4.0


def test_mpc_target_speed_is_capped_at_maximum(capsys):
    controller = MPCController(dt=0.1, horizon=1)

    controller.set_target_speed(10.0)

    capsys.readouterr()
    assert controller.target_speed == 4.0


def test_pure_pursuit_target_speed_is_set_without_cap(capsys):
    controller = CompatibleController(dt=0.1, horizon=8)

    controller.set_target_speed(4.0)

    capsys.readouterr()
    assert controller.target_speed == 4.0


def test_mpc_on_path_control_does_not_call_scipy_optimize(
    monkeypatch, capsys
):
    def fail_if_called(*args, **kwargs):
        pytest.fail("The active MPC control path must remain geometric")

    monkeypatch.setattr(scipy_optimize, "minimize", fail_if_called)
    env = Environment()
    controller = _controller("mpc")
    controller.set_path(PATH)

    controller.calculate_steering(_vehicle(), PATH, env.road_width)

    capsys.readouterr()
