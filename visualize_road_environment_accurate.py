import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Arrow
from environment import Environment, Vehicle
from font_support import set_chinese_font

def visualize_road_environment_accurate():
    """准确可视化展示道路环境，展示修改后的障碍物位置"""
    # 设置中文字体支持
    set_chinese_font()
    
    # 创建环境实例
    env = Environment()
    
    # 获取车道中心线位置（用于参考）
    lane1_center = env.get_lane_center(1)  # 第一车道中心（最下方车道）
    lane2_center = env.get_lane_center(2)  # 第二车道中心（中间车道）
    lane3_center = env.get_lane_center(3)  # 第三车道中心（最上方车道）
    
    # 创建大型画布
    plt.figure(figsize=(18, 12))
    
    # 创建主视图 - 整个道路环境
    plt.subplot(1, 1, 1)
    plt.title('自动驾驶道路环境精确模型', fontsize=16)
    
    # 绘制道路背景
    road = Rectangle((0, 0), env.road_length, env.road_width, 
                    facecolor='darkgray', alpha=0.5, zorder=0)
    plt.gca().add_patch(road)
    
    # 绘制车道背景色
    env._draw_lane_backgrounds(plt.gca())
    
    # 绘制车道标线
    env._draw_lane_markings(plt.gca())
    
    # 绘制道路标记
    env._draw_road_markings(plt.gca())
    
    # 绘制车道中心线
    plt.axhline(y=lane1_center, color='blue', linestyle='--', alpha=0.4, linewidth=1.5, label='车道中心线')
    plt.axhline(y=lane2_center, color='blue', linestyle='--', alpha=0.4, linewidth=1.5)
    plt.axhline(y=lane3_center, color='blue', linestyle='--', alpha=0.4, linewidth=1.5)
    
    # 绘制车道下边界线
    lane1_bottom = lane1_center - env.lane_width/2
    lane2_bottom = lane2_center - env.lane_width/2
    lane3_bottom = lane3_center - env.lane_width/2
    plt.axhline(y=lane1_bottom, color='red', linestyle='-.', alpha=0.3, linewidth=1, label='车道边界线')
    plt.axhline(y=lane2_bottom, color='red', linestyle='-.', alpha=0.3, linewidth=1)
    plt.axhline(y=lane3_bottom, color='red', linestyle='-.', alpha=0.3, linewidth=1)
    plt.axhline(y=env.road_width, color='red', linestyle='-.', alpha=0.3, linewidth=1)
    
    # 绘制车道标识
    plt.text(-5, lane1_center, f'车道1中心 (y={lane1_center:.1f}m)', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    plt.text(-5, lane2_center, f'车道2中心 (y={lane2_center:.1f}m)', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    plt.text(-5, lane3_center, f'车道3中心 (y={lane3_center:.1f}m)', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # 绘制起点和终点
    env.start_vehicle.draw(plt.gca())
    env.end_vehicle.draw(plt.gca())
    
    # 添加起点和终点标注
    plt.annotate('起点', (env.start_point[0], env.start_point[1]), 
                 xytext=(0, 20), textcoords='offset points', ha='center', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    plt.annotate('终点', (env.end_point[0], env.end_point[1]), 
                 xytext=(0, 20), textcoords='offset points', ha='center', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # 绘制标识线，表示起点和终点的横向位置
    plt.plot([env.start_point[0]-5, env.start_point[0]], [lane1_center, lane1_center], 'g-', linewidth=1.5, alpha=0.6)
    plt.plot([env.end_point[0]+5, env.end_point[0]], [lane3_center, lane3_center], 'g-', linewidth=1.5, alpha=0.6)
    
    # 绘制障碍物车辆，并展示其位置
    colors = ['green', 'blue', 'blue', 'blue']
    descriptions = [
        f'障碍物1: 位于车道1中心\n位置: (25.0, {env.obstacle_vehicles[0].y:.1f})',
        f'障碍物2: 位于车道2中心\n位置: (48.27, {env.obstacle_vehicles[1].y:.1f})',
        f'障碍物3: 位于车道3中心\n位置: (48.27, {env.obstacle_vehicles[2].y:.1f})',
        f'障碍物4: 位于车道3中心\n位置: (60.0, {env.obstacle_vehicles[3].y:.1f})'
    ]
    
    # 绘制障碍物车辆
    for i, vehicle in enumerate(env.obstacle_vehicles):
        # 如果是第一辆车，使用绿色高亮
        if i == 0:
            # 标记车辆位于车道中心
            plt.plot([vehicle.x-1, vehicle.x+1], [lane1_center, lane1_center], 'g-', linewidth=2, alpha=0.7)
            
            # 添加位置说明
            plt.text(vehicle.x + 6, vehicle.y, f'位于车道1中心', 
                     fontsize=10, color='green', ha='left', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # 绘制车辆
        rect = Rectangle(
            (vehicle.x - vehicle.length/2, vehicle.y - vehicle.width/2),
            vehicle.length, vehicle.width, angle=np.rad2deg(vehicle.yaw),
            facecolor=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5
        )
        plt.gca().add_patch(rect)
        
        # 添加描述标签
        plt.annotate(descriptions[i], (vehicle.x, vehicle.y), 
                    xytext=(0, 35), textcoords='offset points', ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # 强调第一辆障碍物位于车道中心
    highlight = Circle((env.obstacle_vehicles[0].x, env.obstacle_vehicles[0].y), 
                       radius=1.0, fill=False, edgecolor='green', linestyle='-',
                       linewidth=2, alpha=0.7, zorder=20)
    plt.gca().add_patch(highlight)
    
    # 绘制道路几何标注
    plt.annotate(f'道路长度: {env.road_length:.1f}m', (env.road_length/2, -1), 
                 ha='center', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    plt.annotate(f'道路宽度: {env.road_width:.1f}m', (-1, env.road_width/2), 
                 ha='right', va='center', rotation=90, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # 标记车道宽度
    plt.annotate(f'车道宽度: {env.lane_width:.1f}m', (-3, lane1_center), 
                 ha='center', va='center', rotation=90, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # 添加图例
    plt.legend(loc='upper left', fontsize=10)
    
    # 设置坐标范围
    plt.xlim(-10, env.road_length + 10)
    plt.ylim(-3, env.road_width + 3)
    plt.xlabel('X轴坐标 (m)')
    plt.ylabel('Y轴坐标 (m)')
    plt.grid(False)
    plt.axis('equal')
    
    # 添加标题和信息文本
    plt.title('自动驾驶仿真环境 - 更新后的障碍物位置', fontsize=16)
    info_text = (
        "说明：\n"
        "1. 第一辆障碍车辆 (绿色) 现已移动到车道1中心位置\n"
        "2. 所有障碍车辆都位于各自车道的中心线上\n"
        "3. 路径规划算法考虑了所有障碍物的实际位置\n"
        "4. 蓝色虚线表示车道中心线，红色点线表示车道边界线"
    )
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12, 
               bbox=dict(facecolor='lightyellow', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # 保存并显示图像
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('road_environment_updated.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("更新后的道路环境可视化已保存为'road_environment_updated.png'")

if __name__ == "__main__":
    visualize_road_environment_accurate() 