import inspect
import math

import pytest

from environment import Environment
from astar_path_planning import AStar
from rrt_path_planning import RRT
from rrt_star_path_planning import RRTStar


ZIGZAG = [
    [10.0, 6.0],
    [20.0, 7.0],
    [30.0, 5.0],
    [40.0, 7.0],
    [50.0, 6.0],
]


@pytest.fixture
def env():
    return Environment()


@pytest.fixture
def rrt(env, capsys):
    planner = RRT(
        env,
        step_size=1.0,
        max_iter=20000,
        goal_sample_rate=30,
        max_turn_angle=10,
        safety_distance=1.5,
    )
    capsys.readouterr()
    return planner


@pytest.fixture
def astar(env, capsys):
    planner = AStar(env, grid_resolution=0.5, safety_distance=1.5)
    capsys.readouterr()
    return planner


@pytest.fixture
def rrt_star(env, capsys):
    planner = RRTStar(
        env,
        step_size=1.5,
        max_iter=25000,
        goal_sample_rate=20,
        safety_distance=1.7,
        rewire_radius=3.0,
    )
    capsys.readouterr()
    return planner


@pytest.mark.parametrize("planner_class", [RRT, AStar, RRTStar])
def test_planner_classes_expose_common_interface(planner_class):
    assert hasattr(planner_class, "planning")
    assert hasattr(planner_class, "smooth_path")
    assert hasattr(planner_class, "save_and_show_results")


@pytest.mark.parametrize("planner_class", [RRT, AStar])
def test_rrt_and_astar_save_results_accept_raw_and_smooth_paths(planner_class):
    parameters = inspect.signature(planner_class.save_and_show_results).parameters

    assert "path" in parameters
    assert "smooth_path" in parameters
    assert "filename" in parameters


def test_rrt_star_save_results_does_not_require_a_smooth_path_argument():
    parameters = inspect.signature(RRTStar.save_and_show_results).parameters

    assert "path" in parameters
    assert "filename" in parameters
    assert "smooth_path" not in parameters


def test_rrt_uses_requested_safety_distance_without_lane_constraints(rrt):
    assert rrt.safety_distance == 1.5
    assert getattr(rrt, "apply_lane_constraint", False) is False


def test_rrt_accepts_free_node(rrt, env):
    node = RRT.Node(10.0, env.get_lane_center(2))

    assert rrt._is_collision_free(node) is True


def test_rrt_rejects_node_at_first_obstacle_center(rrt, env):
    node = RRT.Node(25.0, env.get_lane_center(3))

    assert rrt._is_collision_free(node) is False


def test_rrt_rejects_node_below_safety_distance(rrt):
    node = RRT.Node(10.0, 0.5)

    assert rrt._is_collision_free(node) is False


def test_rrt_rejects_node_before_road_start(rrt):
    node = RRT.Node(-1.0, 6.0)

    assert rrt._is_collision_free(node) is False


def test_rrt_accepts_free_parent_child_segment(rrt, env):
    parent = RRT.Node(10.0, env.get_lane_center(2))
    child = RRT.Node(11.0, env.get_lane_center(2))
    child.parent = parent

    assert rrt._is_collision_free(child) is True


def test_rrt_rejects_parent_child_segment_through_obstacle_aabb(rrt, env):
    parent = RRT.Node(20.0, env.get_lane_center(3))
    child = RRT.Node(25.0, env.get_lane_center(3))
    child.parent = parent

    assert rrt._is_collision_free(child) is False


def test_rrt_star_accepts_free_segment(rrt_star, env):
    from_node = RRTStar.Node(10.0, env.get_lane_center(2))
    to_node = RRTStar.Node(12.0, env.get_lane_center(2))

    assert rrt_star._is_collision_free(from_node, to_node) is True


def test_rrt_star_rejects_segment_through_obstacle(rrt_star, env):
    from_node = RRTStar.Node(20.0, env.get_lane_center(3))
    to_node = RRTStar.Node(25.0, env.get_lane_center(3))

    assert rrt_star._is_collision_free(from_node, to_node) is False


def test_rrt_star_rejects_segment_ending_before_road_start(rrt_star, env):
    from_node = RRTStar.Node(10.0, env.get_lane_center(2))
    to_node = RRTStar.Node(-1.0, env.get_lane_center(2))

    assert rrt_star._is_collision_free(from_node, to_node) is False


def test_rrt_smooth_path_preserves_endpoints_and_locks_middle_point(rrt, capsys):
    smoothed = rrt.smooth_path(ZIGZAG, smoothness=0.3)
    capsys.readouterr()

    assert smoothed[0] == [10.0, 6.0]
    assert smoothed[-1] == [50.0, 6.0]
    assert smoothed[2] == pytest.approx([30.0, 6.119221308801128])


@pytest.mark.parametrize("short_path", [[], [[0, 0], [1, 1]]])
def test_rrt_smooth_path_returns_short_paths_unchanged(rrt, short_path):
    assert rrt.smooth_path(short_path) == short_path


def test_rrt_star_smooth_path_preserves_endpoints_and_locks_middle_point(
    rrt_star, capsys
):
    smoothed = rrt_star.smooth_path(ZIGZAG, smoothness=0.25)
    capsys.readouterr()

    assert smoothed[0] == [10.0, 6.0]
    assert smoothed[-1] == [50.0, 6.0]
    assert smoothed[2] == pytest.approx([30.0, 6.141367023536051])


def test_astar_smooth_path_applies_lane_constraint_without_changing_start_x(
    astar, env, capsys
):
    lane_center = env.get_lane_center(2)
    lane_path = [
        [10.0, lane_center],
        [20.0, lane_center],
        [30.0, lane_center],
        [40.0, lane_center],
        [50.0, lane_center],
    ]

    smoothed = astar.smooth_path(lane_path)
    capsys.readouterr()

    assert smoothed[0][0] == lane_path[0][0]


def test_astar_default_environment_path_is_safe_and_deterministic(
    astar, env, capsys
):
    path = astar.planning(
        env.start_point[0],
        env.start_point[1],
        env.end_point[0],
        env.end_point[1],
    )
    capsys.readouterr()

    assert isinstance(path, list)
    assert path
    assert all(isinstance(point, list) and len(point) == 2 for point in path)
    assert math.dist(path[0], env.start_point) <= 1.0
    assert math.dist(path[-1], env.end_point) <= 1.5
    assert all(
        not env.is_collision(x, y, radius=1.5)
        for x, y in path
    )
    assert len(path) == 58
