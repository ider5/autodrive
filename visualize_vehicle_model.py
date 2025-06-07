import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from vehicle_model import BicycleModel
from font_support import set_chinese_font

def visualize_bicycle_model():
    """可视化自行车模型"""
    # 设置中文字体支持
    set_chinese_font()
    
    # 创建画布和子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('自行车运动学模型(Bicycle Kinematic Model)可视化', fontsize=16)
    
    # 整理所有子图的数组为一维
    axes = axes.flatten()
    
    # 创建车辆模型实例
    vehicle = BicycleModel()
    
    # 1. 绘制自行车模型的基本结构 - 左上
    ax1 = axes[0]
    ax1.set_title('自行车模型基本结构')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.grid(True)
    
    # 绘制车辆外形
    car_length = vehicle.length
    car_width = vehicle.width
    car = Rectangle(
        (-car_length/2, -car_width/2),
        car_length, car_width, angle=0,
        facecolor='lightblue', alpha=0.7, edgecolor='blue', linewidth=2
    )
    ax1.add_patch(car)
    
    # 绘制车轮
    wheel_width = 0.2
    wheel_length = 0.5
    # 后轮
    rear_wheel = Rectangle(
        (-car_length/2 + wheel_length/2, -car_width/4 - wheel_width/2),
        wheel_length, wheel_width, angle=0,
        facecolor='black', alpha=0.8
    )
    ax1.add_patch(rear_wheel)
    
    # 前轮（可转向）
    front_wheel = Rectangle(
        (car_length/2 - wheel_length*1.5, -car_width/4 - wheel_width/2),
        wheel_length, wheel_width, angle=0,
        facecolor='black', alpha=0.8
    )
    ax1.add_patch(front_wheel)
    
    # 绘制轴距
    ax1.plot([-car_length/4, car_length/4], [0, 0], 'r-', linewidth=2)
    ax1.text(-0.5, 0.2, f'轴距 (L): {vehicle.L}m', fontsize=10)
    
    # 标注前后轮位置
    ax1.plot([-car_length/4], [0], 'ko', markersize=6)
    ax1.plot([car_length/4], [0], 'ko', markersize=6)
    ax1.text(-car_length/4 - 0.5, -0.4, '后轮', fontsize=10)
    ax1.text(car_length/4 - 0.3, -0.4, '前轮', fontsize=10)
    
    # 绘制航向角
    ax1.arrow(0, 0, 1.5, 0, head_width=0.2, head_length=0.3, fc='green', ec='green', linewidth=2)
    ax1.text(1.0, 0.3, 'yaw (航向角)', fontsize=10, color='green')
    
    # 2. 绘制不同转向角下的车辆运动 - 右上
    ax2 = axes[1]
    ax2.set_title('不同转向角下的车辆轨迹')
    ax2.set_xlim(-2, 8)
    ax2.set_ylim(-2, 6)
    ax2.set_aspect('equal')
    ax2.grid(True)
    
    # 绘制不同转向角下的轨迹
    steering_angles = [0, 5, 10, 15, 20]  # 度
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, angle in enumerate(steering_angles):
        # 将角度转换为弧度
        delta = np.deg2rad(angle)
        
        # 初始化车辆位置
        x, y = 0, 0
        yaw = 0
        v = 1.0  # 固定速度
        
        # 存储轨迹
        trajectory_x = [x]
        trajectory_y = [y]
        
        # 模拟车辆运动
        for t in range(30):
            # 更新状态
            if angle == 0:  # 直线运动
                x += v * np.cos(yaw) * 0.1
                y += v * np.sin(yaw) * 0.1
            else:  # 转向运动
                x += v * np.cos(yaw) * 0.1
                y += v * np.sin(yaw) * 0.1
                yaw += v * np.tan(delta) / vehicle.L * 0.1
            
            # 记录轨迹
            trajectory_x.append(x)
            trajectory_y.append(y)
        
        # 绘制轨迹
        ax2.plot(trajectory_x, trajectory_y, color=colors[i], linewidth=2, 
                 label=f'转向角: {angle}°')
    
    # 添加图例
    ax2.legend(loc='upper left')
    
    # 3. 绘制车辆状态更新方程 - 左下
    ax3 = axes[2]
    ax3.set_title('自行车模型状态更新方程')
    ax3.axis('off')  # 关闭坐标轴
    
    # 创建文本框来显示状态更新方程
    textstr = '\n'.join((
        '状态更新方程:',
        r'$x_{t+1} = x_t + v_t \cdot \cos(\theta_t) \cdot dt$',
        r'$y_{t+1} = y_t + v_t \cdot \sin(\theta_t) \cdot dt$',
        r'$\theta_{t+1} = \theta_t + \frac{v_t \cdot \tan(\delta_t)}{L} \cdot dt$',
        r'$v_{t+1} = v_t + a_t \cdot dt$',
        '',
        '其中:',
        r'$(x, y)$ - 车辆位置坐标',
        r'$\theta$ - 航向角 (yaw)',
        r'$v$ - 车辆速度',
        r'$\delta$ - 前轮转向角',
        r'$a$ - 加速度',
        r'$L$ - 轴距',
        r'$dt$ - 时间步长'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # 4. 绘制车辆物理约束 - 右下
    ax4 = axes[3]
    ax4.set_title('车辆物理约束参数')
    ax4.axis('off')  # 关闭坐标轴
    
    # 创建文本框来显示车辆约束参数
    constraints = '\n'.join((
        '车辆约束参数:',
        f'• 轴距 (L): {vehicle.L}m',
        f'• 车身尺寸: {vehicle.length}m × {vehicle.width}m',
        f'• 最大转向角: ±{np.rad2deg(vehicle.max_steer):.1f}°',
        f'• 最大转向角速度: {np.rad2deg(vehicle.max_delta_dot):.1f}°/s',
        f'• 速度范围: {vehicle.min_v} - {vehicle.max_v}m/s',
        f'• 最大加速度: ±{vehicle.max_a}m/s²',
        f'• 最大加加速度(jerk): {vehicle.max_jerk}m/s³',
        f'• 时间步长: {vehicle.dt}s',
        '',
        '高级特性:',
        '• 转向角变化率限制',
        '• 加速度变化率限制',
        '• 高速区动态降低可用加速度',
        '• 航向角规范化'
    ))
    ax4.text(0.05, 0.95, constraints, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('vehicle_model_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("车辆自行车模型可视化已保存为'vehicle_model_visualization.png'")

if __name__ == "__main__":
    visualize_bicycle_model() 