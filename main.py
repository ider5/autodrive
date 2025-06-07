import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import matplotlib.gridspec as gridspec
import argparse

from environment import Environment, Vehicle
from rrt_path_planning import RRT
from rrt_star_path_planning import RRTStar
from astar_path_planning import AStar
from vehicle_model import BicycleModel
from pure_pursuit_controller import CompatibleController
from font_support import set_chinese_font, labels, use_english_labels

def main():
    # 创建命令行参数解析器，只保留随机数种子参数
    parser = argparse.ArgumentParser(description='自动驾驶仿真系统')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')
    args = parser.parse_args()
    
    # 设置随机种子（如果提供）
    if args.seed is not None:
        np.random.seed(args.seed)
        import random
        random.seed(args.seed)
    
    print("自动驾驶系统启动...")
    
    # 用户选择路径规划算法
    print("\n请选择路径规划算法:")
    print("1. RRT (快速随机树)")
    print("2. A* (A星算法)")
    print("3. RRT* (优化随机树 - 具有渐近最优性)")
    
    while True:
        try:
            choice = int(input("请输入数字 (1、2 或 3): "))
            if choice in [1, 2, 3]:
                break
            else:
                print("请输入有效的数字 1、2 或 3")
        except ValueError:
            print("请输入有效的数字 1、2 或 3")
    
    # 设置中文字体
    set_chinese_font()
    global labels
    if labels is None:
        labels = use_english_labels()
    
    # 创建环境，使用固定障碍物位置
    env = Environment()
    
    # 根据选择创建路径规划器
    if choice == 1:
        print("使用RRT路径规划算法")
        planner = RRT(env, step_size=1.0, max_iter=20000, goal_sample_rate=30, max_turn_angle=10, safety_distance=1.5)
        filename = "rrt_path_planning.png"
    elif choice == 2:
        print("使用A*路径规划算法")
        planner = AStar(env, grid_resolution=0.5, safety_distance=1.5)
        filename = "astar_path_planning.png"
    else:  # choice == 3
        print("使用RRT*路径规划算法（运动学约束+平滑版本）")
        planner = RRTStar(env, step_size=1.0, max_iter=200000, goal_sample_rate=20, 
                         safety_distance=1.5, rewire_radius=3.0, early_stop_enabled=True,
                         no_improvement_limit=8000, improvement_threshold=0.1, 
                         target_quality_factor=1.15, smooth_iterations=3)
        filename = "rrt_star_path_planning.png"
    
    # 获取起点和终点
    start_point = env.start_point
    end_point = env.end_point
    
    # 路径规划
    algorithm_names = {1: "RRT", 2: "A*", 3: "RRT*"}
    algorithm_name = algorithm_names[choice]
    print(f"正在进行{algorithm_name}路径规划...")
    path = planner.planning(start_point[0], start_point[1], end_point[0], end_point[1])
    
    if not path:
        print("无法找到路径！")
        return
    
    # 路径处理 - 根据算法类型决定是否平滑
    if choice == 1:  # RRT算法
        print("RRT路径平滑...")
        smooth_path = planner.smooth_path(path, smoothness=0.3)  # 使用0.3的平滑系数，增加平滑程度
    elif choice == 2:  # A*算法
        print("A*算法使用原始路径，不进行平滑处理")
        smooth_path = path  # 直接使用原始路径
    else:  # RRT*算法
        print("RRT*算法使用约束+平滑路径")
        smooth_path = path  # RRT*生成约束路径并进行平滑处理
    
    # 保存并显示路径规划结果
    print("保存路径规划结果...")
    if choice == 3:  # RRT*算法
        planner.save_and_show_results(path, filename)
    else:
        planner.save_and_show_results(path, smooth_path, filename)
    
    # 创建车辆模型
    vehicle = BicycleModel()
    vehicle.width = env.vehicle_width
    vehicle.length = env.vehicle_length
    
    # 设置车辆初始状态
    initial_speed = 0.0  # 初始速度
    
    # 确保车辆中心点与路径规划起点完全重合
    # 直接使用路径的第一个点作为起点，确保精确匹配
    if path and len(path) > 1:
        # 从路径的前两个点计算初始航向角
        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
        initial_yaw = np.arctan2(dy, dx)  # 确保初始朝向与路径方向一致
        print(f"计算得到的初始航向角: {np.rad2deg(initial_yaw):.2f}度")
        vehicle.set_state(path[0][0], path[0][1], initial_yaw, initial_speed)
    else:
        initial_yaw = 0.0  # 默认初始朝向（沿x轴正方向）
        vehicle.set_state(start_point[0], start_point[1], initial_yaw, initial_speed)
    
    # 创建路径跟踪控制器
    target_speed = 4.0  # 目标速度设置为4.0 m/s
    
    # 创建路径跟踪控制器
    controller = CompatibleController(dt=vehicle.dt, horizon=8)
    # 设置目标速度
    controller.target_speed = target_speed
    # 设置环境引用
    controller.env = env
    # 将平滑路径提前传递给控制器，减少运行时计算
    controller.set_path(smooth_path)
    
    # 调试：打印路径和车辆初始信息
    print(f"设置控制器目标速度: {target_speed} m/s")
    print(f"车辆初始位置: ({vehicle.x:.2f}, {vehicle.y:.2f}), 航向: {np.rad2deg(vehicle.yaw):.2f}度")
    if smooth_path and len(smooth_path) >= 3:
        print(f"控制器路径前3点: {smooth_path[:3]}")
    path_types = {1: "RRT平滑路径", 2: "A*原始路径", 3: "RRT*约束+平滑路径"}
    path_type = path_types[choice]
    print(f"使用路径类型: {path_type}")
    
    # 仿真设置
    sim_time = 30.0  # 最大仿真时间 (s)
    dt = vehicle.dt  # 时间步长
    time = 0.0
    
    # 记录历史轨迹
    x_history = [vehicle.x]
    y_history = [vehicle.y]
    yaw_history = [vehicle.yaw]
    v_history = [vehicle.v]
    t_history = [0.0]
    target_v_history = [0.0]  # 记录目标速度历史
    
    # 添加总行驶距离变量
    total_distance = 0.0
    
    # 计数器，用于确保多次连续碰撞检测才触发终止
    collision_count = 0
    max_collision_count = 15  # 增加最大连续碰撞次数，允许更长时间通过复杂区域
    
    # 模拟驾驶
    print("开始模拟驾驶...")
    print(f"使用目标速度: {target_speed} m/s")
    while time < sim_time:
        # 使用控制器计算转向角和加速度
        delta, ai = controller.calculate_steering(vehicle, smooth_path, env.road_width)
        
        # 记录目标速度用于绘图
        target_v_history.append(target_speed)
        
        # 更新车辆状态前检查下一步是否会导致碰撞
        next_x = vehicle.x + vehicle.v * np.cos(vehicle.yaw) * dt
        next_y = vehicle.y + vehicle.v * np.sin(vehicle.yaw) * dt
        
        # 检查下一位置是否有碰撞风险或超出车道
        if env.is_collision(next_x, next_y, radius=0, yaw=vehicle.yaw, length=vehicle.length, width=vehicle.width):
            collision_count += 1
            if collision_count >= max_collision_count:
                print(f"发生碰撞或超出车道！时间: {time:.2f}秒")
                break
        else:
            collision_count = 0  # 重置计数器
            
        # 更新车辆状态
        vehicle.update(ai, delta)
        
        # 检查碰撞（用于记录，但不立即终止）
        if env.is_collision(vehicle.x, vehicle.y, radius=0, yaw=vehicle.yaw, length=vehicle.length, width=vehicle.width):
            print(f"警告：位置({vehicle.x:.2f}, {vehicle.y:.2f})可能存在碰撞风险")
        
        # 记录轨迹
        x_history.append(vehicle.x)
        y_history.append(vehicle.y)
        yaw_history.append(vehicle.yaw)
        v_history.append(vehicle.v)
        t_history.append(time)
        
        # 计算当前步骤行驶的距离
        if len(x_history) >= 2:
            step_distance = np.hypot(x_history[-1] - x_history[-2], y_history[-1] - y_history[-2])
            total_distance += step_distance
        
        # 每秒输出一次当前行驶距离
        if time % 1.0 < dt:  # 每秒大约输出一次
            print(f"时间: {time:.1f}s, 行驶距离: {total_distance:.2f}m")
        
        # 更新时间
        time += dt
        
        # 检查是否到达终点
        dist_to_goal = np.hypot(vehicle.x - end_point[0], vehicle.y - end_point[1])
        if dist_to_goal < 2.0:
            print(f"到达终点！总用时: {time:.2f}秒, 总行驶距离: {total_distance:.2f}米")
            break
    
    print(f"模拟结束, 总行驶距离: {total_distance:.2f}米")
    
    # 绘制并保存结果
    print("绘制模拟结果...")
    plot_simulation_result(env, path, smooth_path, x_history, y_history, v_history, t_history, target_v_history)
    
    # 创建动画
    print("创建模拟动画...")
    create_animation(env, smooth_path, x_history, y_history, yaw_history, v_history)

def plot_simulation_result(env, path, smooth_path, x_history, y_history, v_history, t_history, target_v_history=None):
    """绘制模拟结果"""
    # 创建带有特定布局的图形
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    
    # 1. 路径规划与跟踪 (左侧整列)
    ax1 = plt.subplot(gs[:, 0])  # 路径跟踪图占用左侧整列
    env.plot_environment(ax1)
    
    # 设置标题
    title = labels.get('路径规划与跟踪', 'Path Planning & Tracking')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # 安全距离设置（不再显示红色虚线）
    safety_distance = 1.5  # 安全距离设置为1.5米
    
    # 在图像上添加安全距离说明
    ax1.text(5, env.road_width - 0.5, 
             f"安全距离: {safety_distance:.2f}米", 
             fontsize=12, color='black', 
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
             zorder=15)
    
    # 绘制原始路径
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax1.plot(path_x, path_y, '--', color='navy', linewidth=1.5, 
            label=labels.get('原始路径', 'Original Path'), zorder=6)
    
    # 绘制平滑路径
    smooth_path_x = [p[0] for p in smooth_path]
    smooth_path_y = [p[1] for p in smooth_path]
    ax1.plot(smooth_path_x, smooth_path_y, '-', color='darkgreen', linewidth=2, 
            label=labels.get('平滑路径', 'Smoothed Path'), zorder=7)
    
    # 绘制车辆实际轨迹
    ax1.plot(x_history, y_history, '-', color='crimson', linewidth=2.5, 
            label=labels.get('车辆轨迹', 'Vehicle Trajectory'), zorder=8)
    
    # 添加起点和终点标记
    ax1.scatter([path[0][0]], [path[0][1]], color='green', s=100, marker='*', zorder=9)
    ax1.scatter([path[-1][0]], [path[-1][1]], color='red', s=100, marker='*', zorder=9)
    
    # 添加图例
    ax1.legend(loc='upper left', fontsize=10)
    
    # 2. 车辆速度 (右上)
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(t_history, v_history, '-', color='blue', linewidth=2, label=labels.get('实际速度', 'Actual Speed'))
    
    # 绘制目标速度曲线
    if target_v_history is not None and len(target_v_history) == len(t_history):
        ax2.plot(t_history, target_v_history, '--', color='red', linewidth=1.5, 
                label=labels.get('目标速度', 'Target Speed'))
    
    ax2.fill_between(t_history, 0, v_history, color='skyblue', alpha=0.3)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Speed [m/s]')
    ax2.set_title(labels.get('车辆速度', 'Vehicle Speed'), fontsize=12, fontweight='bold')
    ax2.legend()
    
    # 3. X-Y轨迹 (右下) - 改为轨迹热力图
    ax3 = plt.subplot(gs[1, 1])
    
    # 创建用于颜色映射的分段点
    points = np.array([x_history, y_history]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 使用时间作为颜色映射
    norm = plt.Normalize(0, t_history[-1])
    lc = plt.matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(np.array(t_history[:-1]))
    lc.set_linewidth(3)
    line = ax3.add_collection(lc)
    
    # 添加起点和终点标记
    ax3.scatter(x_history[0], y_history[0], color='green', s=80, marker='o', label='Start')
    ax3.scatter(x_history[-1], y_history[-1], color='red', s=80, marker='o', label='Goal')
    
    # 设置坐标范围和标签
    ax3.set_xlim(0, env.road_length)
    ax3.set_ylim(0, env.road_width)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_title(labels.get('轨迹时间分布', 'Trajectory Time Distribution'), 
                 fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 添加颜色条
    cbar = fig.colorbar(line, ax=ax3)
    cbar.set_label('Time [s]')
    
    # 添加图例
    ax3.legend(loc='upper right')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=120, bbox_inches='tight')
    plt.show()

def create_animation(env, path, x_history, y_history, yaw_history, v_history):
    """创建动画"""
    # 创建单独的图形对象
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 绘制环境
    env.plot_environment(ax)
    
    # 安全距离设置（不再显示红色虚线）
    safety_distance = 1.5  # 安全距离设置为1.5米
    
    # 在图像上添加安全距离说明
    ax.text(5, env.road_width - 0.5, 
            f"安全距离: {safety_distance:.2f}米", 
            fontsize=12, color='black', 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
            zorder=15)
    
    # 绘制路径
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, '-', color='darkgreen', linewidth=2, label='Path', zorder=5)
    
    # 车辆表示 - 使用矩形和箭头组合表示车辆
    car_length = env.vehicle_length
    car_width = env.vehicle_width
    
    # 使用自定义函数处理绘制，确保使用中心点坐标
    car = Rectangle((0, 0), car_length, car_width, fc='red', ec='black', alpha=0.8, zorder=10)
    ax.add_patch(car)
    
    # 辅助函数，根据中心点坐标和旋转角度设置矩形位置
    def update_car_position(rect, center_x, center_y, yaw):
        # 在matplotlib中，Rectangle的位置是指定其左下角
        # 需要计算从中心点到左下角的偏移，考虑旋转角度
        
        # 首先计算未旋转状态下的左下角相对于中心点的偏移
        dx = -car_length/2
        dy = -car_width/2
        
        # 应用旋转来计算实际左下角位置
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        corner_x = center_x + dx * cos_yaw - dy * sin_yaw
        corner_y = center_y + dx * sin_yaw + dy * cos_yaw
        
        # 设置Rectangle的位置和旋转角度
        rect.set_xy((corner_x, corner_y))
        rect.set_angle(np.rad2deg(yaw))
        
        # 绘制中心点标记（调试用）
        if hasattr(ax, 'center_point'):
            ax.center_point.remove()
        ax.center_point = ax.plot(center_x, center_y, 'ko', markersize=4)[0]
    
    # 车辆轨迹 - 使用渐变色显示
    trajectory_points = []
    trajectory_colors = []
    trajectory = plt.matplotlib.collections.LineCollection([], cmap='inferno', 
                                                        linewidths=2.5, zorder=7)
    ax.add_collection(trajectory)
    
    # 文本显示 - 美化信息面板
    info_panel = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, 
                                 boxstyle='round,pad=0.5'), zorder=20)
    
    # 添加标题
    title = labels.get('自动驾驶模拟', 'Autonomous Driving Simulation')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # 进度条
    progress_bar_bg = Rectangle((0, 0), 0, 0, fc='lightgray', ec='black', zorder=9)
    progress_bar = Rectangle((0, 0), 0, 0, fc='limegreen', ec=None, zorder=9)
    ax.add_patch(progress_bar_bg)
    ax.add_patch(progress_bar)
    
    def init():
        """初始化动画"""
        update_car_position(car, x_history[0], y_history[0], yaw_history[0])
        info_panel.set_text('')
        trajectory.set_segments([])
        
        # 初始化进度条
        progress_bar_bg.set_xy((10, env.road_width + 1))
        progress_bar_bg.set_width(env.road_length - 20)
        progress_bar_bg.set_height(0.5)
        
        progress_bar.set_xy((10, env.road_width + 1))
        progress_bar.set_width(0)
        progress_bar.set_height(0.5)
        
        return car, info_panel, trajectory, progress_bar_bg, progress_bar
    
    def animate(i):
        """更新动画帧"""
        # 更新车辆位置
        x = x_history[i]
        y = y_history[i]
        yaw = yaw_history[i]
        v = v_history[i]
        t = i * 0.1
        
        # 更新车辆位置和方向
        update_car_position(car, x, y, yaw)
        
        # 更新轨迹 - 使用渐变色显示
        if i > 0:
            points = np.array([x_history[:i+1], y_history[:i+1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            trajectory.set_segments(segments)
            
            # 使用速度作为颜色映射
            colors = np.array(v_history[:i])
            norm = plt.Normalize(0, max(v_history))
            trajectory.set_array(colors)
        
        # 更新信息面板
        time_label = labels.get('时间', 'Time')
        speed_label = labels.get('速度', 'Speed')
        distance_label = labels.get('行驶距离', 'Distance')
        
        # 计算已行驶距离
        if i > 0:
            distance = sum(np.hypot(x_history[j+1]-x_history[j], y_history[j+1]-y_history[j]) 
                         for j in range(i))
        else:
            distance = 0
            
        # 格式化信息面板
        info_text = f"{time_label}: {t:.1f}s\n{speed_label}: {v:.2f}m/s\n{distance_label}: {distance:.1f}m"
        info_panel.set_text(info_text)
        
        # 更新进度条
        total_time = len(x_history) * 0.1
        progress = t / total_time
        progress_bar.set_width((env.road_length - 20) * progress)
        
        return car, info_panel, trajectory, progress_bar_bg, progress_bar
    
    # 创建动画
    ani = animation.FuncAnimation(fig, animate, frames=len(x_history),
                                   init_func=init, blit=False, interval=50)
    
    # 保存动画
    try:
        ani.save('vehicle_animation.gif', writer='pillow', fps=20, dpi=100)
        print("动画已保存为 'vehicle_animation.gif'")
    except Exception as e:
        print(f"保存动画时出错: {e}")
        print("继续显示静态图片...")
    
    plt.show()

if __name__ == "__main__":
    main()