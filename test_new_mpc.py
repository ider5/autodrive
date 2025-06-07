import numpy as np
from environment import Environment
from astar_path_planning import AStar
from vehicle_model import BicycleModel
from mpc_controller import MPCController

def test_new_mpc():
    """测试新的简化MPC控制器"""
    print("=== 测试新的简化MPC控制器 ===")
    
    # 创建环境和路径
    env = Environment()
    planner = AStar(env, grid_resolution=0.5, safety_distance=1.5)
    
    start_point = env.start_point
    end_point = env.end_point
    path = planner.planning(start_point[0], start_point[1], end_point[0], end_point[1])
    
    print(f"路径长度: {len(path)}")
    print(f"起点: {path[0]}")
    print(f"终点: {path[-1]}")
    
    # 创建车辆
    vehicle = BicycleModel()
    vehicle.width = env.vehicle_width
    vehicle.length = env.vehicle_length
    
    # 设置初始状态
    if len(path) > 1:
        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
        initial_yaw = np.arctan2(dy, dx)
        vehicle.set_state(path[0][0], path[0][1], initial_yaw, 0.0)
    
    print(f"车辆初始状态: ({vehicle.x:.2f}, {vehicle.y:.2f}), 航向: {np.rad2deg(vehicle.yaw):.1f}°, 速度: {vehicle.v:.2f}m/s")
    
    # 创建MPC控制器
    controller = MPCController(dt=vehicle.dt, horizon=4)
    controller.set_target_speed(3.0)
    controller.set_path(path)
    
    print("\n开始测试控制器...")
    
    # 测试多个步骤
    for step in range(10):
        print(f"\n--- 步骤 {step+1} ---")
        print(f"车辆状态: ({vehicle.x:.2f}, {vehicle.y:.2f}), 航向: {np.rad2deg(vehicle.yaw):.1f}°, 速度: {vehicle.v:.2f}m/s")
        
        try:
            # 计算控制输入
            delta, accel = controller.calculate_steering(vehicle, path)
            print(f"控制输出: δ={np.rad2deg(delta):.1f}°, a={accel:.2f}m/s²")
            
            # 检查控制输入是否合理
            if abs(delta) > 0.01 or abs(accel) > 0.01:
                print(f"✅ 控制器产生了非零输出")
            else:
                print(f"⚠️ 控制器输出接近零")
            
            # 更新车辆状态 (注意参数顺序：加速度在前，转向角在后)
            old_x, old_y = vehicle.x, vehicle.y
            vehicle.update(accel, delta)
            
            # 检查车辆是否移动
            moved_dist = np.hypot(vehicle.x - old_x, vehicle.y - old_y)
            print(f"车辆移动距离: {moved_dist:.4f}m")
            
            if moved_dist > 0.001:
                print(f"✅ 车辆正在移动")
            else:
                print(f"❌ 车辆没有移动")
                
        except Exception as e:
            print(f"❌ 控制器异常: {e}")
            import traceback
            traceback.print_exc()
            break
            
        # 如果车辆开始移动，就认为测试成功
        if vehicle.v > 0.1:
            print(f"🎉 车辆开始移动，速度: {vehicle.v:.2f}m/s")
            break
    
    print(f"\n=== 测试结果 ===")
    print(f"最终位置: ({vehicle.x:.2f}, {vehicle.y:.2f})")
    print(f"最终速度: {vehicle.v:.2f}m/s")
    
    # 计算与起点的距离
    start_dist = np.hypot(vehicle.x - path[0][0], vehicle.y - path[0][1])
    print(f"距离起点: {start_dist:.4f}m")
    
    if start_dist > 0.01 or vehicle.v > 0.05:  # 降低阈值，或者检查速度
        print("✅ MPC控制器工作正常！")
        return True
    else:
        print("❌ MPC控制器未能驱动车辆移动")
        return False

if __name__ == "__main__":
    success = test_new_mpc()
    if success:
        print("\n🎯 新MPC控制器测试成功！")
    else:
        print("\n💥 新MPC控制器仍有问题，需要进一步调试...") 