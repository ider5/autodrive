"""Closed-loop vehicle simulation."""

from dataclasses import dataclass
from typing import List

import numpy as np

from autodrive.config import SimulationConfig


@dataclass
class SimulationResult:
    """Recorded state and termination details for one simulation."""

    x_history: List[float]
    y_history: List[float]
    yaw_history: List[float]
    v_history: List[float]
    t_history: List[float]
    target_v_history: List[float]
    total_distance: float
    time: float
    stop_reason: str
    collision_count: int
    distance_to_goal: float


def run_simulation(
    env,
    planner,
    vehicle,
    controller,
    path,
    smooth_path,
    end_point=None,
    config=None,
    *,
    animate=True,
    plot=True,
    show=True,
):
    """Run the driving loop without changing the legacy step order."""
    if config is None:
        config = SimulationConfig()
    if end_point is None:
        end_point = env.end_point

    target_speed = config.target_speed
    dt = config.dt
    time = 0.0
    x_history = [vehicle.x]
    y_history = [vehicle.y]
    yaw_history = [vehicle.yaw]
    v_history = [vehicle.v]
    t_history = [0.0]
    target_v_history = [0.0]
    total_distance = 0.0
    collision_count = 0
    stop_reason = "timeout"

    print("开始模拟驾驶...")
    print(f"使用目标速度: {target_speed} m/s")
    while time < config.sim_time:
        delta, ai = controller.calculate_steering(
            vehicle, smooth_path, env.road_width
        )
        target_v_history.append(target_speed)

        next_x = vehicle.x + vehicle.v * np.cos(vehicle.yaw) * dt
        next_y = vehicle.y + vehicle.v * np.sin(vehicle.yaw) * dt
        if env.is_collision(
            next_x,
            next_y,
            radius=0,
            angle=vehicle.yaw,
            length=vehicle.length,
            width=vehicle.width,
        ):
            collision_count += 1
            if collision_count >= config.max_collision_count:
                print(f"发生碰撞或超出车道！时间: {time:.2f}秒")
                stop_reason = "collision"
                break
        else:
            collision_count = 0

        vehicle.update(ai, delta)

        if env.is_collision(
            vehicle.x,
            vehicle.y,
            radius=0,
            angle=vehicle.yaw,
            length=vehicle.length,
            width=vehicle.width,
        ):
            print(
                f"警告：位置({vehicle.x:.2f}, {vehicle.y:.2f})可能存在碰撞风险"
            )

        x_history.append(vehicle.x)
        y_history.append(vehicle.y)
        yaw_history.append(vehicle.yaw)
        v_history.append(vehicle.v)
        t_history.append(time)

        if len(x_history) >= 2:
            step_distance = np.hypot(
                x_history[-1] - x_history[-2],
                y_history[-1] - y_history[-2],
            )
            total_distance += step_distance

        if time % 1.0 < dt:
            print(f"时间: {time:.1f}s, 行驶距离: {total_distance:.2f}m")

        time += dt

        dist_to_goal = np.hypot(
            vehicle.x - end_point[0], vehicle.y - end_point[1]
        )
        if dist_to_goal < config.goal_distance:
            print(
                f"到达终点！总用时: {time:.2f}秒, "
                f"总行驶距离: {total_distance:.2f}米"
            )
            stop_reason = "goal"
            break

    print(f"模拟结束, 总行驶距离: {total_distance:.2f}米")
    planner_safety_distance = getattr(planner, "safety_distance", 1.5)

    if plot:
        from autodrive.simulation.plots import plot_simulation_result

        print("绘制模拟结果...")
        plot_simulation_result(
            env,
            path,
            smooth_path,
            x_history,
            y_history,
            v_history,
            t_history,
            target_v_history,
            planner_safety_distance,
            show=show,
        )

    if animate:
        from autodrive.simulation.animation import create_animation

        print("创建模拟动画...")
        create_animation(
            env,
            smooth_path,
            x_history,
            y_history,
            yaw_history,
            v_history,
            planner_safety_distance,
            show=show,
        )

    distance_to_goal = float(
        np.hypot(vehicle.x - end_point[0], vehicle.y - end_point[1])
    )
    return SimulationResult(
        x_history=x_history,
        y_history=y_history,
        yaw_history=yaw_history,
        v_history=v_history,
        t_history=t_history,
        target_v_history=target_v_history,
        total_distance=float(total_distance),
        time=time,
        stop_reason=stop_reason,
        collision_count=collision_count,
        distance_to_goal=distance_to_goal,
    )
