"""Public package import smoke tests."""

import autodrive
from autodrive.control.mpc import MPCController
from autodrive.control.pure_pursuit import CompatibleController
from autodrive.control.stanley import CompatibleStanleyController
from autodrive.environment import Environment, ObstacleVehicle
from autodrive.i18n import set_chinese_font, use_english_labels
from autodrive.planning.astar import AStar
from autodrive.planning.rrt import RRT
from autodrive.planning.rrt_star import RRTStar
from autodrive.vehicle import BicycleModel


def test_package_exports_public_classes():
    assert autodrive.Environment is Environment
    assert autodrive.ObstacleVehicle is ObstacleVehicle
    assert autodrive.BicycleModel is BicycleModel
    assert autodrive.AStar is AStar
    assert autodrive.RRT is RRT
    assert autodrive.RRTStar is RRTStar
    assert autodrive.CompatibleController is CompatibleController
    assert autodrive.MPCController is MPCController
    assert autodrive.CompatibleStanleyController is CompatibleStanleyController


def test_i18n_helpers_are_callable():
    assert callable(set_chinese_font)
    assert callable(use_english_labels)
