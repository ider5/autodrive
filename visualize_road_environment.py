import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from environment import Environment, Vehicle
from font_support import set_chinese_font

def visualize_road_environment():
    """可视化展示道路环境"""
    # 设置中文字体支持
    set_chinese_font()
    
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('自动驾驶模拟道路环境', fontsize=16)
    
    # 整理子图为一维数组
    axes = axes.flatten()
    
    # 创建环境实例
    env = Environment()
    
    # 1. 绘制完整道路环境 - 左上
    ax1 = axes[0]
    ax1.set_title('完整道路环境')
    env.plot_environment(ax1)
    
    # 2. 绘制车道结构和标记 - 右上
    ax2 = axes[1]
    ax2.set_title('车道结构和标记')
    
    # 绘制道路背景
    road = Rectangle((0, 0), env.road_length, env.road_width, 
                     facecolor='darkgray', alpha=0.5, zorder=0)
    ax2.add_patch(road)
    
    # 绘制车道背景色
    env._draw_lane_backgrounds(ax2)
    
    # 绘制车道标线
    env._draw_lane_markings(ax2)
    
    # 绘制道路标记（箭头等）
    env._draw_road_markings(ax2)
    
    # 绘制车道标签
    env._draw_lane_labels(ax2)
    
    # 添加车道中心线和标注
    for i in range(1, env.num_lanes + 1):
        center_y = env.get_lane_center(i)
        ax2.axhline(y=center_y, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax2.text(-3, center_y, f'车道{i}中心线', fontsize=9, 
                 ha='right', va='center', color='black',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        ax2.text(env.road_length+3, center_y, f'y={center_y:.1f}m', fontsize=9, 
                 ha='left', va='center', color='black',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # 标注车道宽度
    for i in range(env.num_lanes):
        y_bottom = i * env.lane_width
        y_top = (i+1) * env.lane_width
        mid_y = (y_bottom + y_top) / 2
        ax2.annotate('', xy=(-2, y_bottom), xytext=(-2, y_top),
                    arrowprops=dict(arrowstyle='<->', color='black'))
        ax2.text(-4, mid_y, f'{env.lane_width:.1f}m', fontsize=9, ha='center', va='center')
    
    # 设置图形范围
    ax2.set_xlim(-5, env.road_length + 5)
    ax2.set_ylim(-2, env.road_width + 2)
    ax2.set_aspect('equal')
    ax2.grid(False)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    
    # 3. 绘制障碍物车辆配置 - 左下
    ax3 = axes[2]
    ax3.set_title('障碍物车辆配置')
    
    # 绘制道路背景
    road = Rectangle((0, 0), env.road_length, env.road_width, 
                     facecolor='darkgray', alpha=0.5, zorder=0)
    ax3.add_patch(road)
    
    # 绘制车道背景色和标线
    env._draw_lane_backgrounds(ax3)
    env._draw_lane_markings(ax3)
    
    # 绘制车道标签
    env._draw_lane_labels(ax3)
    
    # 获取车道中心线位置（用于参考）
    lane1_center = env.get_lane_center(3)  # 第一车道中心（最下方车道）
    lane2_center = env.get_lane_center(2)  # 第二车道中心（中间车道）
    lane3_center = env.get_lane_center(1)  # 第三车道中心（最上方车道）
    
    # 绘制车道中心线（参考线）
    ax3.axhline(y=lane1_center, color='green', linestyle='--', alpha=0.3)
    ax3.axhline(y=lane2_center, color='green', linestyle='--', alpha=0.3)
    ax3.axhline(y=lane3_center, color='green', linestyle='--', alpha=0.3)
    
    # 绘制障碍物车辆并添加标注
    for i, vehicle in enumerate(env.obstacle_vehicles):
        vehicle.draw(ax3)
        
        # 自定义标注文本
        if i == 0:
            position_desc = "位于车道1中心"
        elif i == 1:
            position_desc = "位于车道2中心"
        elif i == 2:
            position_desc = "位于车道3中心"
        elif i == 3:
            position_desc = "位于车道3中心"
        
        # 添加障碍物车辆标注
        ax3.annotate(f'障碍物{i+1}: {position_desc}', (vehicle.x, vehicle.y), xytext=(0, 15), 
                     textcoords='offset points', ha='center', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # 添加位置标注
        ax3.annotate(f'({vehicle.x:.1f}, {vehicle.y:.1f})', (vehicle.x, vehicle.y), 
                    xytext=(0, -15), textcoords='offset points', ha='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # 添加偏移量标注（仅针对第一辆车）
        if i == 0:
            offset = vehicle.y - lane1_center
            ax3.annotate(f'偏移: {offset:.1f}m', (vehicle.x, vehicle.y), 
                        xytext=(0, -30), textcoords='offset points', ha='center', fontsize=8,
                        bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round,pad=0.2'))
            
            # 添加从中心线到车辆的标识线
            ax3.plot([vehicle.x, vehicle.x], [lane1_center, vehicle.y], 'r-', linewidth=1, alpha=0.5)
    
    # 设置图形范围
    ax3.set_xlim(-5, env.road_length + 5)
    ax3.set_ylim(-2, env.road_width + 2)
    ax3.set_aspect('equal')
    ax3.grid(False)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    
    # 4. 绘制起点和终点位置 - 右下
    ax4 = axes[3]
    ax4.set_title('起点和终点位置')
    
    # 绘制道路背景
    road = Rectangle((0, 0), env.road_length, env.road_width, 
                     facecolor='darkgray', alpha=0.5, zorder=0)
    ax4.add_patch(road)
    
    # 绘制车道背景色和标线
    env._draw_lane_backgrounds(ax4)
    env._draw_lane_markings(ax4)
    
    # 绘制起点和终点车辆
    env.start_vehicle.draw(ax4)
    env.end_vehicle.draw(ax4)
    
    # 添加详细标注
    # 起点标注
    ax4.annotate('起点', (env.start_point[0], env.start_point[1]), 
                 xytext=(0, 20), textcoords='offset points', ha='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    ax4.annotate(f'坐标: ({env.start_point[0]:.1f}, {env.start_point[1]:.1f})', 
                 (env.start_point[0], env.start_point[1]), 
                 xytext=(0, -20), textcoords='offset points', ha='center', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # 终点标注
    ax4.annotate('终点', (env.end_point[0], env.end_point[1]), 
                 xytext=(0, 20), textcoords='offset points', ha='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    ax4.annotate(f'坐标: ({env.end_point[0]:.1f}, {env.end_point[1]:.1f})', 
                 (env.end_point[0], env.end_point[1]), 
                 xytext=(0, -20), textcoords='offset points', ha='center', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # 标记起点终点连线（理想路径）
    ax4.plot([env.start_point[0], env.end_point[0]], 
             [env.start_point[1], env.end_point[1]], 
             'r--', linewidth=1.5, label='理想直线路径')
    
    # 计算并标注直线距离
    dist = np.hypot(env.end_point[0] - env.start_point[0], 
                     env.end_point[1] - env.start_point[1])
    mid_x = (env.start_point[0] + env.end_point[0]) / 2
    mid_y = (env.start_point[1] + env.end_point[1]) / 2
    ax4.annotate(f'直线距离: {dist:.1f}m', (mid_x, mid_y), 
                 xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # 添加图例
    ax4.legend(loc='lower right')
    
    # 设置图形范围
    ax4.set_xlim(-5, env.road_length + 5)
    ax4.set_ylim(-2, env.road_width + 2)
    ax4.set_aspect('equal')
    ax4.grid(False)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('road_environment_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("道路环境可视化已保存为'road_environment_visualization.png'")

if __name__ == "__main__":
    visualize_road_environment() 