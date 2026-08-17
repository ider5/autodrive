import pytest

from autodrive.control.protocol import Controller
from mpc_controller import MPCController
from pure_pursuit_controller import CompatibleController
from stanley_controller import CompatibleStanleyController


@pytest.mark.parametrize(
    "controller",
    [
        CompatibleController(dt=0.1, horizon=8),
        MPCController(dt=0.1, horizon=1),
        CompatibleStanleyController(dt=0.1, horizon=8),
    ],
)
def test_main_controller_implements_controller_protocol(controller):
    assert isinstance(controller, Controller)
