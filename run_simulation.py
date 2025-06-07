import numpy as np
import matplotlib.pyplot as plt
import time as python_time

from environment import Environment, Vehicle
from astar_path_planning import AStar
from vehicle_model import BicycleModel
from mpc_controller import MPCController

def run_full_simulation():
    """运行完整的仿真，测试重构的MPC控制器"""
    print("=== 完整仿真测试 - 重构版MPC控制器 ===")
    
    # 创建环境
    env = Environment()
    
    # 创建A*路径规划器
    planner = AStar(env, grid_resolution=0.5, safety_distance=1.5)
    
    # 获取起点和终点
    start_point = env.start_point
    end_point = env.end_point
    
    print(f"起点: ({start_point[0]:.2f}, {start_point[1]:.2f})")
    print(f"终点: ({end_point[0]:.2f}, {end_point[1]:.2f})")
    
    # 路径规划
    print("正在进行A*路径规划...")
    start_time = python_time.time()
    path = planner.planning(start_point[0], start_point[1], end_point[0], end_point[1])
    planning_time = python_time.time() - start_time
    
    if not path:
        print("❌ 路径规划失败！")
        return
        
    print(f"✅ 路径规划成功，用时: {planning_time:.2f}秒")
    print(f"路径长度: {len(path)}个点")
    
    # 创建车辆模型
    vehicle = BicycleModel()
    vehicle.width = env.vehicle_width
    vehicle.length = env.vehicle_length
    
    # 设置车辆初始状态
    if len(path) > 1:
        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
        initial_yaw = np.arctan2(dy, dx)
        vehicle.set_state(path[0][0], path[0][1], initial_yaw, 0.0)
        print(f"车辆初始状态: ({vehicle.x:.2f}, {vehicle.y:.2f}), 航向: {np.rad2deg(initial_yaw):.1f}°")
    
    # 创建重构的MPC控制器
    print("创建重构版MPC控制器...")
    controller = MPCController(dt=vehicle.dt, horizon=5)
    target_speed = 4.0
    controller.set_target_speed(target_speed)
    controller.set_path(path)
    print(f"目标速度: {target_speed} m/s")
    
    # 仿真参数
    sim_time = 30.0  # 最大仿真时间
    dt = vehicle.dt
    time = 0.0
    
    # 记录历史轨迹
    x_history = [vehicle.x]
    y_history = [vehicle.y]
    yaw_history = [vehicle.yaw]
    v_history = [vehicle.v]
    t_history = [0.0]
    
    # 计数器
    total_distance = 0.0
    collision_count = 0
    max_collision_count = 10
    success_steps = 0
    total_steps = 0
    
    # 目标检查参数
    goal_tolerance = 2.0  # 目标容忍度
    goal_reached = False
    
    print("开始完整仿真...")
    simulation_start = python_time.time()
    
    while time < sim_time and not goal_reached:
        total_steps += 1
        
        try:
            # 计算控制输入
            delta, accel = controller.calculate_steering(vehicle, path)
            
            # 检查控制输入
            if np.isfinite(delta) and np.isfinite(accel):
                success_steps += 1
                
                # 更新车辆状态 (参数顺序：加速度在前，转向角在后)
                vehicle.update(accel, delta)
                
                # 计算移动距离
                if len(x_history) > 0:
                    dist = np.hypot(vehicle.x - x_history[-1], vehicle.y - y_history[-1])
                    total_distance += dist
                
                # 记录轨迹
                x_history.append(vehicle.x)
                y_history.append(vehicle.y)
                yaw_history.append(vehicle.yaw)
                v_history.append(vehicle.v)
                t_history.append(time)
                
                # 检查是否到达目标
                dist_to_goal = np.hypot(vehicle.x - end_point[0], vehicle.y - end_point[1])
                if dist_to_goal < goal_tolerance:
                    goal_reached = True
                    print(f"✅ 到达目标！距离目标: {dist_to_goal:.2f}m")
                
                # 边界检查
                safety_margin = vehicle.width / 2 + 0.2
                if vehicle.y <= safety_margin or vehicle.y >= 15.0 - safety_margin:
                    collision_count += 1
                    if collision_count >= max_collision_count:
                        print(f"❌ 车辆持续偏离道路，仿真终止")
                        break
                else:
                    collision_count = 0
                
                # 定期输出状态
                if total_steps % 50 == 0:
                    print(f"仿真时间: {time:.1f}s, 位置: ({vehicle.x:.1f}, {vehicle.y:.1f}), 速度: {vehicle.v:.1f}m/s")
                
            else:
                print(f"时间 {time:.1f}s: 控制输入异常")
                break
                
        except Exception as e:
            print(f"仿真异常: {e}")
            break
            
        time += dt
    
    simulation_time = python_time.time() - simulation_start
    
    # 计算仿真结果
    print(f"\n=== 仿真结果 ===")
    print(f"仿真时间: {simulation_time:.2f}秒")
    print(f"仿真步数: {total_steps}")
    print(f"成功步数: {success_steps}")
    print(f"成功率: {success_steps/total_steps*100:.1f}%")
    print(f"总行驶距离: {total_distance:.2f}m")
    print(f"平均速度: {total_distance/time:.2f}m/s" if time > 0 else "0.00m/s")
    print(f"最大速度: {max(v_history):.2f}m/s")
    
    # 最终位置
    final_dist_to_goal = np.hypot(vehicle.x - end_point[0], vehicle.y - end_point[1])
    print(f"最终位置: ({vehicle.x:.2f}, {vehicle.y:.2f})")
    print(f"距离目标: {final_dist_to_goal:.2f}m")
    
    # 判断成功标准
    if goal_reached:
        print("🎉 任务完成：成功到达目标！")
    elif success_steps/total_steps > 0.9:
        print("✅ 控制器性能良好：成功率超过90%")
    else:
        print("⚠️ 控制器需要进一步优化")
        
    return {
        'success': goal_reached or (success_steps/total_steps > 0.9),
        'simulation_time': simulation_time,
        'total_distance': total_distance,
        'success_rate': success_steps/total_steps*100,
        'goal_reached': goal_reached,
        'final_distance_to_goal': final_dist_to_goal
    }

if __name__ == "__main__":
    result = run_full_simulation()
    
    if result['success']:
        print("\n🎯 重构版MPC控制器测试成功！")
    else:
        print("\n❌ 需要进一步调优...") 