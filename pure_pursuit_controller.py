"""
Pure Pursuit路径跟踪控制器

本模块实现Pure Pursuit控制算法，用于车辆的路径跟踪控制。
Pure Pursuit是一种几何路径跟踪方法，通过追踪前瞻点实现路径跟踪。
"""

import numpy as np
from vehicle_model import BicycleModel

class PurePursuitController:
    """
    Pure Pursuit路径跟踪控制器
    
    该类实现Pure Pursuit算法，通过计算车辆当前位置到前瞻点的
    几何关系来确定转向角，实现平滑的路径跟踪。
    """
    def __init__(self, dt=0.1, horizon=8):
        """
        初始化控制器
        
        参数:
            dt (float): 采样时间步长
            horizon (int): 兼容参数，Pure Pursuit不需要预测时域
        """
        # 基本参数
        self.dt = dt
        
        # 车辆约束
        self.max_steer = np.deg2rad(20.0)  # 最大转向角
        self.max_accel = 1.5               # 最大加速度
        self.max_decel = -1.5              # 最大减速度
        
        # 目标速度
        self.target_speed = 5.0  # 默认目标速度 (m/s)
        
        # 参考路径
        self.path = None
        
        # 环境引用
        self.env = None
        
        # 安全参数
        self.safety_margin = 2.5  # 距边界的安全距离
        
        # 累积误差（速度控制用）
        self.error_sum = 0.0
        
    def set_target_speed(self, speed):
        """设置目标速度"""
        self.target_speed = speed

    def set_path(self, path):
        """设置参考路径，用于预先准备"""
        if path and len(path) > 2:
            # 预处理路径，增加安全边界
            self.path = self._preprocess_path(path)
        else:
            self.path = path
    
    def _preprocess_path(self, path):
        """预处理路径，增加安全边界防护"""
        if not path or len(path) < 3 or not self.env:
            return path
            
        # 创建新路径，保持原始路径的起点和终点
        safe_path = [path[0]]  # 复制起点
        
        # 安全距离（道路边界）
        min_y = self.safety_margin
        max_y = self.env.road_width - self.safety_margin
        
        # 处理中间点
        for i in range(1, len(path) - 1):
            x, y = path[i]
            
            # 约束Y坐标以保持安全距离
            safe_y = np.clip(y, min_y, max_y)
            
            # 添加安全点
            safe_path.append([x, safe_y])
        
        # 确保添加终点
        safe_path.append(path[-1])
        
        return safe_path
        
    def _get_nearest_point(self, x, y, points):
        """获取距离当前位置最近的路径点"""
        min_dist = float('inf')
        min_idx = 0
        
        for i, (px, py) in enumerate(points):
            dist = np.hypot(px - x, py - y)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                
        return min_idx, min_dist
    
    def calculate_steering(self, vehicle, path, road_width=None):
        """计算控制输入 - 路径跟踪控制"""
        # 使用当前路径或已存储的路径
        current_path = self.path if self.path is not None else path
        
        # 确保路径有效
        if current_path is None or len(current_path) < 2:
            return 0.0, 0.0
            
        # 在每次控制前进行安全检查
        if self.env is not None:
            # 如果车辆已经接近或超出安全边界，执行紧急校正
            if self._is_near_boundary(vehicle):
                return self._emergency_correction(vehicle, current_path)
        
        # 使用Pure Pursuit控制
        return self._pure_pursuit_control(vehicle, current_path)
    
    def _is_near_boundary(self, vehicle):
        """检查车辆是否接近道路边界"""
        if not self.env:
            return False
            
        # 车辆位置
        y = vehicle.y
        
        # 道路边界
        road_width = self.env.road_width
        
        # 考虑车辆宽度的一半
        half_width = vehicle.width / 2
        
        # 检查是否接近下边界或上边界
        if (y - half_width) < 0.5 or (y + half_width) > (road_width - 0.5):
            return True
            
        return False
        
    def _emergency_correction(self, vehicle, path):
        """紧急校正控制 - 在车辆接近边界时使用"""
        # 车辆位置
        y = vehicle.y
        v = vehicle.v
        
        # 道路边界
        road_width = self.env.road_width
        
        # 距离下边界和上边界的距离
        dist_to_lower = y
        dist_to_upper = road_width - y
        
        # 转向校正
        if dist_to_lower < 1.0:  # 非常接近下边界
            # 强制向上转向 - 根据接近程度调整强度
            correction_factor = max(0.1, min(0.5, 1.0 - dist_to_lower))
            delta = -self.max_steer * correction_factor  # 负值使车辆向上转向
            
            # 同时减速
            decel_factor = max(0.3, min(0.8, 1.0 - dist_to_lower))
            a = -self.max_accel * decel_factor
            
            print(f"紧急校正: 向上转向，减速! 距下边界: {dist_to_lower:.2f}m")
            return delta, a
            
        elif dist_to_upper < 1.0:  # 非常接近上边界
            # 强制向下转向
            correction_factor = max(0.1, min(0.5, 1.0 - dist_to_upper))
            delta = self.max_steer * correction_factor  # 正值使车辆向下转向
            
            # 同时减速
            decel_factor = max(0.3, min(0.8, 1.0 - dist_to_upper))
            a = -self.max_accel * decel_factor
            
            print(f"紧急校正: 向下转向，减速! 距上边界: {dist_to_upper:.2f}m")
            return delta, a
            
        # 如果不需要紧急校正，返回默认的Pure Pursuit控制
        return self._pure_pursuit_control(vehicle, path)
    
    def _pure_pursuit_control(self, vehicle, path):
        """Pure Pursuit 跟踪控制器"""
        # 当前状态
        x = vehicle.x
        y = vehicle.y
        yaw = vehicle.yaw
        v = vehicle.v
        L = 2.0  # 轴距
        
        # 前瞻距离根据速度动态调整
        # 增加最小前瞻距离，确保有足够的预见性
        lookahead_dist = max(3.0, min(5.0, 0.5 * v + 3.0))
        
        # 获取最近点
        min_idx, min_dist = self._get_nearest_point(x, y, path)
        
        # 横向偏差补偿 - 如果离路径太远，增加纠正能力
        cross_track_error = min_dist
        if min_idx < len(path) - 1:
            # 计算参考路径的朝向
            next_idx = min(min_idx + 1, len(path) - 1)
            path_yaw = np.arctan2(path[next_idx][1] - path[min_idx][1], 
                                 path[next_idx][0] - path[min_idx][0])
            
            # 正确计算横向误差（正值表示车辆在路径右侧，负值表示在左侧）
            lateral_error = -((x - path[min_idx][0]) * np.sin(path_yaw) - 
                            (y - path[min_idx][1]) * np.cos(path_yaw))
            
            # 打印调试信息
            if abs(lateral_error) > 1.0:  # 当横向误差较大时才打印
                print(f"横向误差: {lateral_error:.2f}m, 路径朝向: {np.rad2deg(path_yaw):.2f}度, 车辆朝向: {np.rad2deg(yaw):.2f}度")
        else:
            lateral_error = 0.0
        
        # 从最近点开始搜索满足前瞻距离的点
        target_idx = min_idx
        for i in range(min_idx, len(path) - 1):
            dist = np.hypot(path[i][0] - x, path[i][1] - y)
            if dist > lookahead_dist:
                target_idx = i
                break
            target_idx = i
        
        # 获取目标点
        tx, ty = path[target_idx]
        
        # 计算前瞻点与车辆的相对坐标（车身坐标系下）
        dx = tx - x
        dy = ty - y
        
        # 计算前瞻点在车身坐标系下的位置
        target_x = dx * np.cos(yaw) + dy * np.sin(yaw)
        target_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
        
        # 确保目标点在前方（如果不在前方，可能需要后退）
        if target_x < 0:
            # 目标在车后方，找一个替代目标
            for i in range(min_idx + 1, len(path)):
                tx_new, ty_new = path[i]
                dx_new = tx_new - x
                dy_new = ty_new - y
                target_x_new = dx_new * np.cos(yaw) + dy_new * np.sin(yaw)
                if target_x_new > 0:  # 找到前方的点
                    tx, ty = tx_new, ty_new
                    target_x = target_x_new
                    target_y = -dx_new * np.sin(yaw) + dy_new * np.cos(yaw)
                    break
                    
        # 计算曲率 k = 2 * y / L^2 (L是轴距，y是前瞻点在车身坐标系下的横向距离)
        # 注意：这里的y是车身坐标系下的横向距离，不是世界坐标系的y
        curvature = 2.0 * target_y / (lookahead_dist * lookahead_dist)
        
        # 转向角 = arctan(L * k)
        delta = np.arctan2(L * curvature, 1.0)
        
        # 添加横向误差修正
        delta += 0.1 * lateral_error  # 横向误差补偿系数调整
        
        # 限制转向角
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        
        # 道路边界安全检查
        if self.env is not None:
            road_width = self.env.road_width
            # 距下边界和上边界的距离
            dist_to_lower = y - vehicle.width/2
            dist_to_upper = road_width - (y + vehicle.width/2)
            
            # 边界避让
            if dist_to_lower < 2.0:
                # 向上修正（减小delta值，向左转向）
                boundary_avoid = max(0, 0.3 * (2.0 - dist_to_lower))
                delta -= boundary_avoid
                if dist_to_lower < 1.0:
                    print(f"边界避让: 接近下边界 {dist_to_lower:.2f}m, 修正: {boundary_avoid:.2f}rad")
            elif dist_to_upper < 2.0:
                # 向下修正（增大delta值，向右转向）
                boundary_avoid = max(0, 0.3 * (2.0 - dist_to_upper))
                delta += boundary_avoid
                if dist_to_upper < 1.0:
                    print(f"边界避让: 接近上边界 {dist_to_upper:.2f}m, 修正: {boundary_avoid:.2f}rad")
            
            # 重新限制转向角
            delta = np.clip(delta, -self.max_steer, self.max_steer)
        
        # 根据距离终点的距离调整速度
        dist_to_goal = np.hypot(path[-1][0] - x, path[-1][1] - y)
        
        # 速度控制
        if dist_to_goal < 10.0:  # 接近终点
            target_speed = min(self.target_speed, 0.4 * dist_to_goal + 1.0)
        else:
            # 基于路径曲率的速度调整
            current_speed = min(self.target_speed, 5.0 / (1.0 + 5.0 * abs(curvature)))
            
            # 边界接近时减速
            if self.env is not None:
                min_dist_to_boundary = min(dist_to_lower, dist_to_upper)
                if min_dist_to_boundary < 2.0:
                    boundary_factor = max(0.5, min(1.0, min_dist_to_boundary / 2.0))
                    current_speed *= boundary_factor
                    
            target_speed = current_speed
        
        # 速度控制器
        speed_error = target_speed - v
        
        # PI控制器
        Kp = 0.5
        Ki = 0.1
            
        # 累积误差（加入抗饱和措施）
        self.error_sum += speed_error * self.dt
        self.error_sum = np.clip(self.error_sum, -3.0, 3.0)  # 防止积分饱和
        
        # 计算加速度
        a = Kp * speed_error + Ki * self.error_sum
        
        # 限制加速度
        a = np.clip(a, self.max_decel, self.max_accel)
        
        return delta, a

class CompatibleController(PurePursuitController):
    """与原代码兼容的控制器接口"""
    def __init__(self, dt=0.1, horizon=8):
        super().__init__(dt, horizon)
        
    def calculate_steering(self, vehicle, path, road_width=None):
        """计算控制输入，使用预先存储的路径或新传入的路径"""
        # 如果传入了新路径且未设置内部路径，则使用传入的路径
        if self.path is None and path is not None:
            self.set_path(path)
        
        # 如果有环境信息，使用环境道路宽度
        if road_width is None and self.env is not None:
            road_width = self.env.road_width
            
        # 调用父类方法
        return super().calculate_steering(vehicle, self.path or path, road_width)

# 兼容性类 - 用于test_mpc.py文件
class VehicleModel:
    """兼容test_mpc.py的车辆模型封装"""
    def __init__(self, dt=0.1):
        self.dt = dt
        # 使用自行车模型作为内部模型
        self.bicycle_model = BicycleModel()
        self.bicycle_model.dt = dt
        
    def update(self, x, y, yaw, v, delta, a=0.0):
        """更新车辆状态"""
        # 设置初始状态
        self.bicycle_model.set_state(x, y, yaw, v)
        # 更新状态
        new_x, new_y, new_yaw, new_v = self.bicycle_model.update(a, delta)
        return new_x, new_y, new_yaw

