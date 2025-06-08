"""
Stanley路径跟踪控制器

本模块实现Stanley控制算法，用于车辆的精确路径跟踪。
Stanley控制器结合横向误差和朝向误差，具有良好的路径跟踪性能。
"""

import numpy as np
from vehicle_model import BicycleModel

class StanleyController:
    """
    Stanley路径跟踪控制器
    
    该类实现Stanley控制算法，通过同时考虑车辆的横向位置误差
    和朝向误差来计算转向控制量，实现高精度的路径跟踪。
    """
    
    def __init__(self, dt=0.1, horizon=8):
        """
        初始化Stanley控制器
        
        参数:
            dt (float): 采样时间步长
            horizon (int): 兼容参数，Stanley不需要预测时域
        """
        # 基本参数
        self.dt = dt
        
        # 车辆参数
        self.wheelbase = 2.0  # 轴距 (m)
        self.width = 1.8      # 车宽 (m)
        self.length = 4.0     # 车长 (m)
        
        # 控制约束
        self.max_steer = np.deg2rad(20.0)  # 最大转向角
        self.max_accel = 1.5               # 最大加速度
        self.max_decel = -2.5              # 最大减速度
        self.max_speed = 4.0               # 最大速度
        self.min_speed = 0.0               # 最小速度
        
        # Stanley控制器参数
        self.k_e = 0.3       # 横向误差增益（调节横向误差响应速度）
        self.k_v = 10.0      # 速度相关增益（速度越高，横向误差影响越小）
        self.k_soft = 1.0    # 软化因子（避免低速时的奇异性）
        
        # 目标速度
        self.target_speed = 4.0
        
        # 参考路径
        self.path = None
        self.env = None
        
        # 控制历史（用于平滑）
        self.prev_delta = 0.0
        self.prev_accel = 0.0
        
        # 路径跟踪参数
        self.lookahead_dist = 2.0  # 前瞻距离
        
        # 平滑参数
        self.alpha_steer = 0.7     # 转向平滑系数
        self.alpha_accel = 0.8     # 加速度平滑系数
        
    def set_target_speed(self, speed):
        """设置目标速度，确保不超过最大速度限制"""
        self.target_speed = min(speed, self.max_speed)
        print(f"Stanley目标速度设置为: {self.target_speed:.1f}m/s (最大限制: {self.max_speed:.1f}m/s)")
        
    def set_path(self, path):
        """设置参考路径"""
        self.path = path
        
    def _find_nearest_point(self, vehicle):
        """找到路径上距离车辆最近的点"""
        if self.path is None or len(self.path) < 2:
            return None, 0, 0
            
        min_dist = float('inf')
        min_idx = 0
        
        # 找到最近的路径点
        for i, point in enumerate(self.path):
            dist = np.hypot(point[0] - vehicle.x, point[1] - vehicle.y)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                
        return self.path[min_idx], min_idx, min_dist
        
    def _calculate_cross_track_error(self, vehicle, nearest_point, path_idx):
        """计算横向跟踪误差（Cross Track Error）"""
        if path_idx >= len(self.path) - 1:
            return 0.0, 0.0
            
        # 获取路径段
        p1 = np.array(self.path[path_idx])
        p2 = np.array(self.path[path_idx + 1])
        
        # 车辆位置
        vehicle_pos = np.array([vehicle.x, vehicle.y])
        
        # 路径方向向量
        path_vector = p2 - p1
        path_length = np.linalg.norm(path_vector)
        
        if path_length < 1e-6:
            return 0.0, 0.0
            
        # 单位路径方向向量
        path_unit = path_vector / path_length
        
        # 车辆到路径起点的向量
        vehicle_to_path = vehicle_pos - p1
        
        # 计算横向误差（垂直距离）
        cross_track_error = np.cross(vehicle_to_path, path_unit)
        
        # 路径朝向角
        path_yaw = np.arctan2(path_vector[1], path_vector[0])
        
        return cross_track_error, path_yaw
        
    def _calculate_heading_error(self, vehicle_yaw, path_yaw):
        """计算朝向误差"""
        heading_error = path_yaw - vehicle_yaw
        # 角度归一化到[-π, π]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        return heading_error
        
    def _stanley_steering_control(self, vehicle):
        """Stanley转向控制算法"""
        # 找到最近的路径点
        nearest_point, path_idx, min_dist = self._find_nearest_point(vehicle)
        
        if nearest_point is None:
            return 0.0
            
        # 计算横向误差和路径朝向
        cross_track_error, path_yaw = self._calculate_cross_track_error(vehicle, nearest_point, path_idx)
        
        # 计算朝向误差
        heading_error = self._calculate_heading_error(vehicle.yaw, path_yaw)
        
        # Stanley控制律
        # delta = heading_error + arctan(k_e * cross_track_error / (k_v + v))
        
        # 速度自适应项
        speed_term = self.k_v + max(vehicle.v, self.k_soft)
        
        # 横向误差项
        cross_track_term = np.arctan(self.k_e * cross_track_error / speed_term)
        
        # Stanley转向角
        delta = heading_error + cross_track_term
        
        # 边界保护
        if self.env is not None:
            delta = self._apply_boundary_protection(vehicle, delta)
        
        # 转向角限制
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        
        # 转向平滑
        delta = self.alpha_steer * self.prev_delta + (1 - self.alpha_steer) * delta
        self.prev_delta = delta
        
        return delta
        
    def _apply_boundary_protection(self, vehicle, delta):
        """边界保护逻辑"""
        y = vehicle.y
        margin = self.width / 2 + 0.5
        road_width = self.env.road_width
        
        # 边界检测和校正
        if y < margin:  # 接近下边界
            # 计算需要的校正角度
            boundary_error = margin - y
            correction_angle = np.arctan(boundary_error * 2.0)
            delta = min(delta, -correction_angle)  # 强制向上转向
            print(f"边界保护：强制向上转向 {np.rad2deg(correction_angle):.1f}度")
            
        elif y > (road_width - margin):  # 接近上边界
            # 计算需要的校正角度
            boundary_error = y - (road_width - margin)
            correction_angle = np.arctan(boundary_error * 2.0)
            delta = max(delta, correction_angle)  # 强制向下转向
            print(f"边界保护：强制向下转向 {np.rad2deg(correction_angle):.1f}度")
            
        return delta
        
    def _speed_control(self, vehicle):
        """速度控制"""
        speed_error = self.target_speed - vehicle.v
        
        # PID速度控制
        if vehicle.v >= self.max_speed:
            # 如果超过最大速度，强制减速
            a = min(-0.5, speed_error * 2.0)
        else:
            # 正常速度控制
            a = np.clip(speed_error * 1.5, self.max_decel, self.max_accel)
            
        # 加速度平滑
        a = self.alpha_accel * self.prev_accel + (1 - self.alpha_accel) * a
        self.prev_accel = a
        
        return a
        
    def calculate_steering(self, vehicle, path, road_width=None):
        """计算Stanley控制输入"""
        # 设置路径（如果还没设置）
        if self.path is None:
            self.set_path(path)
            
        # 检查路径有效性
        if self.path is None or len(self.path) < 2:
            return 0.0, 0.0
            
        # Stanley转向控制
        delta = self._stanley_steering_control(vehicle)
        
        # 速度控制
        a = self._speed_control(vehicle)
        
        return delta, a


class CompatibleStanleyController(StanleyController):
    """兼容现有接口的Stanley控制器封装"""
    
    def __init__(self, dt=0.1, horizon=8):
        super().__init__(dt, horizon)
        
    def calculate_steering(self, vehicle, path, road_width=None):
        """保持与现有控制器相同的接口"""
        return super().calculate_steering(vehicle, path, road_width)


# 向后兼容的车辆模型类（用于独立测试）
class VehicleModel:
    """简化的车辆模型类，用于兼容性"""
    
    def __init__(self, dt=0.1):
        self.dt = dt
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.width = 1.8
        self.length = 4.0
        
    def update(self, x, y, yaw, v, delta, a=0.0):
        """更新车辆状态"""
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v 