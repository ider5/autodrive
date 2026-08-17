"""
模型预测控制器（MPC）

本模块实现模型预测控制算法，用于车辆的路径跟踪控制。
MPC通过预测车辆未来状态并优化控制序列来实现精确的路径跟踪。
"""

import numpy as np

from autodrive.control.geometry import (
    nearest_index,
    path_yaw,
    signed_cross_track,
    wrap_angle,
)

class SimpleMPCController:
    """
    简化模型预测控制器
    
    该类实现简化的MPC算法，通过优化控制输入来最小化路径跟踪误差。
    控制器考虑车辆动力学约束和路径跟踪性能指标。
    """
    
    def __init__(self, dt=0.1, horizon=1):
        """
        初始化MPC控制器
        
        参数:
            dt (float): 采样时间步长
            horizon (int): 预测时域长度
        """
        self.dt = dt
        self.horizon = horizon
        
        # 车辆参数
        self.wheelbase = 2.0
        self.width = 1.8
        self.length = 4.0
        
        # 控制约束
        self.max_steer = np.deg2rad(25.0)     # 稍大的转向角
        self.max_accel = 2.0                  # 更大的加速度
        self.max_decel = -3.0                 # 更大的减速度
        self.max_speed = 4.0                  # 速度限制
        self.min_speed = 0.0                  # 最小速度
        
        # 代价函数权重 - 简化设计
        self.w_lat = 50.0      # 横向误差权重
        self.w_yaw = 10.0      # 航向误差权重  
        self.w_speed = 1.0     # 速度跟踪权重
        self.w_control = 0.1   # 控制输入权重
        self.w_smooth = 0.5    # 控制平滑权重
        
        # 目标速度
        self.target_speed = 3.0
        
        # 路径相关
        self.path = None
        self.current_index = 0
        self.env = None
        
        # 控制历史
        self.prev_delta = 0.0
        self.prev_accel = 0.0
        
        print("创建简化版MPC控制器")
        
    def set_target_speed(self, speed):
        """设置目标速度"""
        self.target_speed = min(speed, self.max_speed)
        print(f"MPC目标速度: {self.target_speed:.1f}m/s")
        
    def set_path(self, path):
        """设置参考路径"""
        self.path = path
        self.current_index = 0
        
    def _find_nearest_point(self, vehicle):
        """找到路径上最近的点"""
        if self.path is None:
            return 0

        return nearest_index(self.path, vehicle.x, vehicle.y)
        
    def _get_path_point(self, index):
        """获取路径点，超出范围时返回终点"""
        if self.path is None:
            return None
            
        if index >= len(self.path):
            return self.path[-1]
        elif index < 0:
            return self.path[0]
        else:
            return self.path[index]
            
    def _calculate_cross_track_error(self, vehicle, path_point):
        """计算横向跟踪误差"""
        if path_point is None:
            return 0.0
            
        # 简单的点到点距离作为横向误差
        return np.hypot(vehicle.x - path_point[0], vehicle.y - path_point[1])
        
    def _calculate_heading_error(self, vehicle, target_point, current_point):
        """计算航向误差"""
        if target_point is None or current_point is None:
            return 0.0
            
        if np.hypot(
            target_point[0] - current_point[0],
            target_point[1] - current_point[1],
        ) < 0.1:
            return 0.0

        desired_yaw = path_yaw(current_point, target_point)
        return wrap_angle(desired_yaw - vehicle.yaw)
            
    def calculate_steering(self, vehicle, path, road_width=None):
        """
        优化的直接几何控制方法 - 移除复杂优化，使用直接计算
        """
        try:
            # 设置路径
            if path is not None:
                self.set_path(path)
                
            if self.path is None or len(self.path) < 2:
                return self._emergency_control()
            
            # 找到最近的路径点
            nearest_idx = self._find_nearest_point(vehicle)
            
            # 简化前瞻逻辑 - 在短道路低速环境下减少预测
            # 根据车速和道路特点调整前瞻距离
            if vehicle.v > 3.0:
                lookahead_points = 2  # 即使高速也只看2个点
            elif vehicle.v > 1.5:
                lookahead_points = 1  # 中速只看1个点
            else:
                lookahead_points = 1  # 低速专注当前点
            target_idx = min(nearest_idx + lookahead_points, len(self.path) - 1)
            target_point = self.path[target_idx]
            
            # 计算期望航向角
            desired_yaw = path_yaw(
                [vehicle.x, vehicle.y], target_point
            )
            
            # 航向误差
            yaw_error = wrap_angle(desired_yaw - vehicle.yaw)
            
            # === 转向控制 ===
            # 改进的路径跟踪算法 - Stanley + Pure Pursuit组合
            
            # 1. 计算横向误差（相对于最近路径点的垂直距离）
            if nearest_idx < len(self.path) - 1:
                lateral_error = -signed_cross_track(
                    [vehicle.x, vehicle.y],
                    self.path[nearest_idx],
                    self.path[nearest_idx + 1],
                )
            else:
                lateral_error = 0.0
            
            # 2. 简化控制 - 减少复杂的组合控制
            # 直接基于目标点的几何关系进行控制
            k_lateral = 1.8  # 降低横向误差增益，避免过激反应
            k_yaw = 1.5      # 降低航向误差增益
            
            # 在低速短道路环境下，主要关注航向跟踪，减少横向修正
            if vehicle.v < 2.0:
                # 低速时主要跟踪航向，减少横向修正的权重
                lateral_control = np.arctan(k_lateral * 0.5 * lateral_error / max(vehicle.v, 0.3))
                heading_control = k_yaw * yaw_error
            else:
                # 中高速时正常控制
                lateral_control = np.arctan(k_lateral * lateral_error / max(vehicle.v, 0.5))
                heading_control = k_yaw * 0.8 * yaw_error  # 稍微降低航向增益
            
            # 4. 简化组合控制
            delta = lateral_control + heading_control
            
            # 5. 边界保护 - 防止车辆偏离车道
            road_upper = 12.0  # 道路上边界
            road_lower = 0.0   # 道路下边界
            safety_margin = 1.0  # 安全边界
            
            # 如果接近上边界，强制向下转向
            if vehicle.y > road_upper - safety_margin:
                boundary_correction = -2.0 * (vehicle.y - (road_upper - safety_margin))
                delta += boundary_correction
                
            # 如果接近下边界，强制向上转向
            elif vehicle.y < road_lower + safety_margin:
                boundary_correction = 2.0 * ((road_lower + safety_margin) - vehicle.y)
                delta += boundary_correction
            
            # 速度自适应调整
            if vehicle.v > 2.0:
                delta *= 0.8  # 高速时减小转向幅度
            elif vehicle.v < 0.5:
                delta *= 1.2  # 低速时适度增大转向幅度
                
            # 限制转向角
            delta = np.clip(delta, -self.max_steer, self.max_steer)
            
            # === 简化速度控制 ===
            # 在短道路低速环境下，简化速度控制逻辑
            speed_error = self.target_speed - vehicle.v
            
            # 简化的速度控制 - 减少复杂判断
            if vehicle.v < 0.1:
                # 启动阶段
                accel = 1.2
            elif speed_error > 0.5:
                # 速度明显不足：加速
                accel = 1.0
            elif speed_error > 0.1:
                # 速度略不足：轻加速
                accel = 0.6
            elif speed_error < -0.5:
                # 速度过高：减速
                accel = -0.4
            else:
                # 速度接近目标：简单比例控制
                accel = speed_error * 0.8
            
            # 简化的转向减速 - 只在转向角很大时减速
            if abs(delta) > np.deg2rad(15) and vehicle.v > 2.0:
                accel *= 0.8  # 大转向角时适度减速
            
            # 限制加速度
            accel = np.clip(accel, self.max_decel, self.max_accel)
            
            # === 控制平滑 ===
            # 与历史值平滑（进一步提高响应性）
            alpha = 0.5  # 更低的平滑系数，更高响应性
            delta = alpha * delta + (1 - alpha) * self.prev_delta
            accel = alpha * accel + (1 - alpha) * self.prev_accel
            
            # 最终约束
            delta = np.clip(delta, -self.max_steer, self.max_steer)
            accel = np.clip(accel, self.max_decel, self.max_accel)
            
            # 速度限制检查
            if vehicle.v >= self.max_speed * 0.95:
                accel = min(accel, -0.5)
                
            # 更新历史
            self.prev_delta = delta
            self.prev_accel = accel
            
            return delta, accel
                
        except Exception as e:
            print(f"MPC控制器异常: {e}")
            return self._emergency_control()
            
    def _emergency_control(self):
        """应急控制 - 改进版本，保持运动连续性"""
        # 保持一定的转向连续性
        delta = np.clip(self.prev_delta * 0.5, -0.2, 0.2)
        # 适中的加速度
        accel = 0.8
        return delta, accel


# 兼容性包装类
class MPCController(SimpleMPCController):
    """兼容性包装类"""
    
    def __init__(self, dt=0.1, horizon=1):
        super().__init__(dt, horizon)
        print("[OK] 已加载优化版MPC控制器 (几何直接控制，高响应性)")
        
    def calculate_steering(self, vehicle, path, road_width=None):
        """计算转向控制"""
        return super().calculate_steering(vehicle, path, road_width)