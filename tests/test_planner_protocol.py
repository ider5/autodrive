import numpy as np
import pytest

from astar_path_planning import AStar
from autodrive.planning.protocol import PathPlanner
from environment import Environment
from rrt_path_planning import RRT
from rrt_star_path_planning import RRTStar


@pytest.fixture
def planners(capsys):
    env = Environment()
    instances = (
        RRT(env, max_iter=1),
        AStar(env),
        RRTStar(env, max_iter=1),
    )
    capsys.readouterr()
    return instances


def test_planners_satisfy_path_planner_protocol(planners):
    assert all(isinstance(planner, PathPlanner) for planner in planners)


def test_planners_save_nonempty_results_without_showing(planners, tmp_path):
    rrt, astar, rrt_star = planners
    path = [[10.0, 6.0], [12.0, 6.5], [14.0, 6.0]]
    smooth_path = [[10.0, 6.0], [12.0, 6.25], [14.0, 6.0]]
    astar.grid_map = np.zeros((astar.height, astar.width))

    outputs = {
        "rrt": tmp_path / "rrt.png",
        "astar": tmp_path / "astar.png",
        "rrt_star": tmp_path / "rrt_star.png",
    }

    rrt.save_and_show_results(path, smooth_path, outputs["rrt"], show=False)
    astar.save_and_show_results(path, smooth_path, outputs["astar"], show=False)
    rrt_star.save_and_show_results(path, outputs["rrt_star"], show=False)

    assert all(output.exists() and output.stat().st_size > 0 for output in outputs.values())
