import font_support

from astar_path_planning import AStar as ShimAStar
from environment import Environment as ShimEnvironment
from mpc_controller import MPCController as ShimMPCController
from pure_pursuit_controller import CompatibleController as ShimCompatibleController
from rrt_path_planning import RRT as ShimRRT
from rrt_star_path_planning import RRTStar as ShimRRTStar
from stanley_controller import CompatibleStanleyController as ShimStanleyController
from vehicle_model import BicycleModel as ShimBicycleModel


def test_root_shims_reexport_package_classes():
    from autodrive.control.mpc import MPCController
    from autodrive.control.pure_pursuit import CompatibleController
    from autodrive.control.stanley import CompatibleStanleyController
    from autodrive.environment import Environment
    from autodrive.planning.astar import AStar
    from autodrive.planning.rrt import RRT
    from autodrive.planning.rrt_star import RRTStar
    from autodrive.vehicle import BicycleModel

    assert ShimEnvironment is Environment
    assert ShimBicycleModel is BicycleModel
    assert ShimAStar is AStar
    assert ShimRRT is RRT
    assert ShimRRTStar is RRTStar
    assert ShimCompatibleController is CompatibleController
    assert ShimMPCController is MPCController
    assert ShimStanleyController is CompatibleStanleyController


def test_font_support_exposes_live_package_labels():
    from autodrive import i18n
    from font_support import set_chinese_font, use_english_labels

    assert callable(set_chinese_font)
    labels = use_english_labels()
    assert labels is i18n.labels
    assert font_support.labels is i18n.labels
