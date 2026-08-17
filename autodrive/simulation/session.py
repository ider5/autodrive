"""Planning, controller wiring, and simulation session orchestration."""

import numpy as np

from autodrive.config import (
    AStarPlannerConfig,
    MPCConfig,
    PurePursuitConfig,
    RRTPlannerConfig,
    RRTStarPlannerConfig,
    SimulationConfig,
    StanleyConfig,
)
from autodrive.control import (
    CompatibleController,
    CompatibleStanleyController,
    MPCController,
)
from autodrive.environment import Environment
from autodrive.planning import AStar, RRT, RRTStar
from autodrive.simulation.loop import run_simulation
from autodrive.vehicle import BicycleModel


def _create_planner(planning_choice, env):
    if planning_choice == 1:
        print("使用RRT路径规划算法")
        config = RRTPlannerConfig()
        planner = RRT(
            env,
            step_size=config.step_size,
            max_iter=config.max_iter,
            goal_sample_rate=config.goal_sample_rate,
            max_turn_angle=config.max_turn_angle,
            safety_distance=config.safety_distance,
        )
        filename = "rrt_path_planning.png"
    elif planning_choice == 2:
        print("使用A*路径规划算法")
        config = AStarPlannerConfig()
        planner = AStar(
            env,
            grid_resolution=config.grid_resolution,
            safety_distance=config.safety_distance,
        )
        filename = "astar_path_planning.png"
    else:
        print("使用RRT*路径规划算法（简化版）")
        config = RRTStarPlannerConfig()
        planner = RRTStar(
            env,
            step_size=config.step_size,
            max_iter=config.max_iter,
            goal_sample_rate=config.goal_sample_rate,
            safety_distance=config.safety_distance,
            rewire_radius=config.rewire_radius,
        )
        filename = "rrt_star_path_planning.png"
    return planner, config, filename


def _create_controller(controller_choice, vehicle):
    if controller_choice == 1:
        print("使用Pure Pursuit控制器")
        config = PurePursuitConfig()
        controller = CompatibleController(
            dt=config.dt, horizon=config.horizon
        )
        controller_name = "Pure Pursuit"
    elif controller_choice == 2:
        print("使用MPC (模型预测控制)控制器")
        config = MPCConfig()
        controller = MPCController(dt=config.dt, horizon=config.horizon)
        controller_name = "MPC"
    else:
        print("使用Stanley路径跟踪控制器")
        config = StanleyConfig()
        controller = CompatibleStanleyController(
            dt=config.dt, horizon=config.horizon
        )
        controller_name = "Stanley"
    return controller, controller_name


def run_session(
    planning_choice,
    controller_choice,
    *,
    simulation_config=None,
    show=True,
    animate=True,
    plot=True,
):
    """Plan a route, configure the vehicle, and run the simulation."""
    if simulation_config is None:
        simulation_config = SimulationConfig()

    env = Environment()
    planner, planner_config, filename = _create_planner(
        planning_choice, env
    )
    start_point = env.start_point
    end_point = env.end_point

    algorithm_names = {1: "RRT", 2: "A*", 3: "RRT*"}
    algorithm_name = algorithm_names[planning_choice]
    print(f"正在进行{algorithm_name}路径规划...")
    path = planner.planning(
        start_point[0], start_point[1], end_point[0], end_point[1]
    )
    if not path:
        print("无法找到路径！")
        return None

    if planning_choice == 1:
        print("RRT路径平滑...")
        smooth_path = planner.smooth_path(
            path, smoothness=planner_config.smoothness
        )
    elif planning_choice == 2:
        print("A*算法使用原始路径，不进行平滑处理")
        smooth_path = path
    else:
        print("RRT*算法使用原始路径（未进行平滑处理）")
        smooth_path = path

    print("保存路径规划结果...")
    if planning_choice == 3:
        planner.save_and_show_results(path, filename)
    else:
        planner.save_and_show_results(path, smooth_path, filename)

    vehicle = BicycleModel(dt=simulation_config.dt)
    vehicle.width = env.vehicle_width
    vehicle.length = env.vehicle_length
    initial_speed = 0.0
    if path and len(path) > 1:
        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
        initial_yaw = np.arctan2(dy, dx)
        print(
            f"计算得到的初始航向角: "
            f"{np.rad2deg(initial_yaw):.2f}度"
        )
        vehicle.set_state(
            path[0][0], path[0][1], initial_yaw, initial_speed
        )
    else:
        initial_yaw = 0.0
        vehicle.set_state(
            start_point[0], start_point[1], initial_yaw, initial_speed
        )

    controller, controller_name = _create_controller(
        controller_choice, vehicle
    )
    controller.target_speed = simulation_config.target_speed
    controller.env = env
    controller.set_path(smooth_path)

    print(
        f"设置控制器目标速度: "
        f"{simulation_config.target_speed} m/s"
    )
    print(
        f"车辆初始位置: ({vehicle.x:.2f}, {vehicle.y:.2f}), "
        f"航向: {np.rad2deg(vehicle.yaw):.2f}度"
    )
    if smooth_path and len(smooth_path) >= 3:
        print(f"控制器路径前3点: {smooth_path[:3]}")
    path_types = {
        1: "RRT平滑路径",
        2: "A*原始路径",
        3: "RRT*原始路径",
    }
    print(f"使用路径类型: {path_types[planning_choice]}")
    print(f"使用控制器: {controller_name}")

    return run_simulation(
        env=env,
        planner=planner,
        vehicle=vehicle,
        controller=controller,
        path=path,
        smooth_path=smooth_path,
        end_point=end_point,
        config=simulation_config,
        animate=animate,
        plot=plot,
        show=show,
    )
