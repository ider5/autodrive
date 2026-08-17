import pytest

from autodrive.control.mpc import MPCController
from autodrive.control.protocol import Controller
from autodrive.control.pure_pursuit import CompatibleController
from autodrive.control.stanley import CompatibleStanleyController


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
