"""
车辆动力学模型

本模块实现用于车辆动力学仿真的自行车模型。
自行车模型是车辆运动的简化表示，能够捕捉自动驾驶应用中的基本运动学行为。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class BicycleModel:
    """
    车辆动力学仿真的自行车模型
    
    该类实现运动学自行车模型，使用简化的两轮动力学表示车辆运动。
    该模型在自动驾驶研究中常用，兼顾了精度和计算效率。
    
    属性:
        x (float): 车辆x坐标位置
        y (float): 车辆y坐标位置  
        yaw (float): 车辆航向角（弧度）
        v (float): 车辆速度（米/秒）
        length (float): 车辆长度（米）
        width (float): 车辆宽度（米）
        wheelbase (float): 前后轴距离
        dt (float): 数值积分时间步长
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, dt=0.1):
        """
        初始化自行车模型
        
        参数:
            x (float): 初始x坐标位置
            y (float): 初始y坐标位置
            yaw (float): 初始航向角（弧度）
            v (float): 初始速度（米/秒）
            dt (float): 积分时间步长
        """
        # 车辆参数
        self.L = 2.0  # 轴距 (m)
        self.max_steer = np.deg2rad(20.0)  # 降低最大转向角 (rad)，从30度降低到20度
        self.dt = dt  # 时间步长，恢复为0.1秒
        self.width = 1.8  # 车宽 (m)
        self.length = 4.0  # 车长 (m)
        
        # 车辆状态 [x, y, yaw, v]
        self.x = x  # 位置x
        self.y = y  # 位置y
        self.yaw = yaw  # 航向角
        self.v = v  # 速度
        
        # 控制输入
        self.delta = 0.0  # 前轮转向角
        self.a = 0.0  # 加速度
        
        # 车辆运动限制
        self.max_v = 7.0  # 最大速度 (m/s)，从8.0降低到7.0
        self.min_v = 0.0  # 最小速度 (m/s)
        self.max_a = 1.5  # 最大加速度 (m/s^2)，从2.0降低到1.5
        self.max_delta_dot = np.deg2rad(12.0)  # 最大转向角速度 (rad/s)，从15度/秒降低到12度/秒
        
        # 上一次的转向角，用于限制转向角变化率
        self.prev_delta = 0.0
        
        # 新增: 上一次的加速度，用于限制加速度变化率
        self.prev_a = 0.0
        self.max_jerk = 0.8  # 最大加加速度(jerk) (m/s^3)
    
    def set_state(self, x, y, yaw, v):
        """设置车辆状态"""
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
    
    def update(self, a, delta):
        """更新车辆状态"""
        # 限制转向角变化率
        delta_change = delta - self.prev_delta
        if abs(delta_change) > self.max_delta_dot * self.dt:
            delta_change = np.sign(delta_change) * self.max_delta_dot * self.dt
        
        # 应用变化率限制后的转向角
        delta = self.prev_delta + delta_change
        
        # 限制转向角在最大范围内
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        
        # 新增: 限制加速度变化率(jerk)
        a_change = a - self.prev_a
        if abs(a_change) > self.max_jerk * self.dt:
            a_change = np.sign(a_change) * self.max_jerk * self.dt
        
        # 应用jerk限制后的加速度
        a = self.prev_a + a_change
        
        # 限制加速度
        a = np.clip(a, -self.max_a, self.max_a)
        
        # 新增: 基于当前速度自适应调整加速度限制
        # 高速时减小可用加速度，提高安全性
        if self.v > 5.0:  # 高速区域
            speed_factor = 1.0 - 0.3 * min(1.0, (self.v - 5.0) / 2.0)  # 速度越高，可用加速度越小
            a = np.clip(a, -self.max_a * speed_factor, self.max_a * speed_factor)
        
        # 更新状态 (自行车模型的运动学方程)
        self.x += self.v * np.cos(self.yaw) * self.dt
        self.y += self.v * np.sin(self.yaw) * self.dt
        self.yaw += self.v * np.tan(delta) / self.L * self.dt
        self.v += a * self.dt
        
        # 限制速度
        self.v = np.clip(self.v, self.min_v, self.max_v)
        
        # 规范化航向角到 [-pi, pi]
        self.yaw = self.normalize_angle(self.yaw)
        
        # 保存控制输入
        self.delta = delta
        self.a = a
        
        # 更新上一次的转向角和加速度
        self.prev_delta = delta
        self.prev_a = a
        
        return self.x, self.y, self.yaw, self.v
    
    def normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]区间"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
            
        while angle < -np.pi:
            angle += 2.0 * np.pi
            
        return angle
    
    def get_state(self):
        """获取车辆当前状态"""
        return [self.x, self.y, self.yaw, self.v]
    
    def draw(self, ax=None):
        """绘制车辆"""
        if ax is None:
            _, ax = plt.subplots()
        
        # 车辆外形（矩形）
        car = Rectangle(
            (self.x - self.length/2 * np.cos(self.yaw) - self.width/2 * np.sin(self.yaw),
             self.y - self.length/2 * np.sin(self.yaw) + self.width/2 * np.cos(self.yaw)),
            self.length, self.width, angle=np.rad2deg(self.yaw),
            facecolor='green', alpha=0.7)
        ax.add_patch(car)
        
        # 画出车辆前进方向
        ax.arrow(self.x, self.y, 
                0.5 * np.cos(self.yaw), 
                0.5 * np.sin(self.yaw),
                head_width=0.3, head_length=0.3, 
                fc='red', ec='red')
        
        return ax