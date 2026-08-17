"""
RRT*路径规划算法实现

本模块实现RRT*（RRT-star）路径规划算法，这是RRT算法的改进版本。
RRT*通过重连优化操作实现渐近最优性，能够找到更优质的路径。
"""

import numpy as np
import random
import time

from .plotting import save_rrt_star_results

class RRTStar:
    """
    RRT*路径规划算法
    
    该类实现RRT*算法，在RRT的基础上增加了重连优化步骤，
    能够渐近收敛到最优路径。算法结合了智能采样策略和平滑性约束。
    """
    def __init__(self, env, step_size=1.5, max_iter=10000, goal_sample_rate=20, safety_distance=1.7, rewire_radius=3.0):
        """
        初始化RRT*路径规划器
        
        参数:
            env: 环境对象，包含道路信息和障碍物
            step_size (float): 随机树扩展步长（米）
            max_iter (int): 最大迭代次数
            goal_sample_rate (int): 目标点采样率（百分比）
            safety_distance (float): 与障碍物的安全距离（米）
            rewire_radius (float): 重连半径（米）
        """
        self.env = env
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.safety_distance = safety_distance
        self.rewire_radius = rewire_radius
        self.node_list = []
        
        print(f"初始化RRT*路径规划器")
        print(f"步长: {step_size}m, 最大迭代: {max_iter}, 目标采样率: {goal_sample_rate}%")
        print(f"安全距离: {safety_distance}m, 重连半径: {rewire_radius}m")
    
    class Node:
        """RRT*树节点"""
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None
            self.cost = 0.0  # 从起点到该节点的代价
    
    def planning(self, start_x, start_y, goal_x, goal_y):
        """RRT*路径规划主函数"""
        print(f"开始RRT*路径规划")
        print(f"起点: ({start_x:.1f}, {start_y:.1f})")
        print(f"终点: ({goal_x:.1f}, {goal_y:.1f})")
        
        start_time = time.time()
        
        # 创建起点和终点节点
        start_node = self.Node(start_x, start_y)
        goal_node = self.Node(goal_x, goal_y)
        
        # 初始化节点列表
        self.node_list = [start_node]
        
        best_goal_node = None
        best_cost = float('inf')
        
        # 计算直线距离和目标路径长度
        direct_distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        target_path_length = direct_distance * 1.15  # 允许15%的绕行
        print(f"直线距离: {direct_distance:.2f}m, 目标路径长度: {target_path_length:.2f}m")
        
        # RRT*主循环
        for i in range(self.max_iter):
            # 定期输出进度
            if i % 1000 == 0:
                print(f"RRT*迭代 {i}/{self.max_iter}, 节点数: {len(self.node_list)}")
            
            # 智能采样策略 - 优先道路中央直线行驶
            if random.randint(0, 100) > self.goal_sample_rate:
                sampling_strategy = random.random()
                
                if sampling_strategy < 0.4:
                    # 40%: 道路中央直线采样
                    road_center_y = self.env.road_width / 2
                    rnd_x = random.uniform(0, self.env.road_length)
                    # 在道路中央附近采样，偏差不超过1个车道宽度
                    center_variance = self.env.lane_width * 0.8
                    rnd_y = road_center_y + random.uniform(-center_variance, center_variance)
                    rnd_y = np.clip(rnd_y, 0, self.env.road_width)
                    rnd_node = self.Node(rnd_x, rnd_y)
                    
                elif sampling_strategy < 0.7 and len(self.node_list) > 1:
                    # 30%: 前向偏向采样（保持方向一致性）
                    latest_node = self.node_list[-1]
                    # 计算当前前进方向
                    if latest_node.parent is not None:
                        # 基于历史方向
                        current_direction = np.arctan2(latest_node.y - latest_node.parent.y, 
                                                     latest_node.x - latest_node.parent.x)
                    else:
                        # 基于目标方向
                        current_direction = np.arctan2(goal_y - latest_node.y, goal_x - latest_node.x)
                    
                    # 限制角度变化，保持方向一致性
                    angle_variance = np.pi / 6  # 30度的角度范围（更小的变化）
                    sample_angle = current_direction + random.uniform(-angle_variance, angle_variance)
                    sample_distance = random.uniform(3.0, 6.0)  # 更合理的采样距离
                    
                    rnd_x = latest_node.x + sample_distance * np.cos(sample_angle)
                    rnd_y = latest_node.y + sample_distance * np.sin(sample_angle)
                    
                    # 确保采样点在道路范围内
                    rnd_x = np.clip(rnd_x, 0, self.env.road_length)
                    rnd_y = np.clip(rnd_y, 0, self.env.road_width)
                    
                    rnd_node = self.Node(rnd_x, rnd_y)
                    
                else:
                    # 30%: 避障导向采样
                    # 检查是否需要避障
                    if len(self.node_list) > 1:
                        latest_node = self.node_list[-1]
                        # 检查前方是否有障碍物
                        obstacle_ahead = self._check_obstacle_ahead(latest_node, goal_x, goal_y)
                        if obstacle_ahead:
                            # 在避障方向采样
                            rnd_node = self._sample_for_obstacle_avoidance(latest_node, goal_x, goal_y)
                        else:
                            # 正常随机采样
                            rnd_x = random.uniform(0, self.env.road_length)
                            rnd_y = random.uniform(0, self.env.road_width)
                            rnd_node = self.Node(rnd_x, rnd_y)
                    else:
                        # 正常随机采样
                        rnd_x = random.uniform(0, self.env.road_length)
                        rnd_y = random.uniform(0, self.env.road_width)
                        rnd_node = self.Node(rnd_x, rnd_y)
            else:
                # 目标点采样
                rnd_node = self.Node(goal_x, goal_y)
            
            # 1. 找到最近节点
            nearest_ind = self._get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_ind]
            
            # 2. 从最近节点朝随机点扩展
            new_node = self._steer(nearest_node, rnd_node)
            if new_node is None:
                continue
            
            # 3. 检查碰撞
            if not self._is_collision_free(nearest_node, new_node):
                continue
            
            # 4. 在重连半径内寻找更优父节点
            near_nodes = self._find_near_nodes(new_node)
            best_parent = nearest_node
            min_cost = nearest_node.cost + self._calc_distance(nearest_node, new_node)
            
            # 检查所有近邻节点，寻找最优父节点（考虑路径平滑性和转向限制）
            for near_node in near_nodes:
                if not self._is_collision_free(near_node, new_node):
                    continue
                
                # 检查转向角度是否合理
                if not self._is_turn_angle_acceptable(near_node, new_node):
                    continue
                
                base_cost = near_node.cost + self._calc_distance(near_node, new_node)
                
                # 计算平滑性惩罚
                smoothness_penalty = self._calculate_smoothness_penalty(near_node, new_node)
                total_cost = base_cost + smoothness_penalty
                
                if total_cost < min_cost:
                    best_parent = near_node
                    min_cost = total_cost
            
            # 5. 设置新节点的父节点和代价
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.node_list.append(new_node)
            
            # 6. 重连附近节点
            self._rewire(new_node, near_nodes)
            
            # 7. 检查是否到达目标
            if self._can_connect_to_goal(new_node, goal_node):
                goal_cost = new_node.cost + self._calc_distance(new_node, goal_node)
                if goal_cost < best_cost:
                    best_cost = goal_cost
                    best_goal_node = self.Node(goal_x, goal_y)
                    best_goal_node.parent = new_node
                    best_goal_node.cost = goal_cost
                    
                    # 如果找到足够好的路径，可以提前结束
                    if best_cost <= target_path_length:
                        print(f"✅ 找到优质路径！迭代: {i}, 长度: {best_cost:.2f}m ≤ {target_path_length:.2f}m")
                        break
        
        # 生成最终路径
        if best_goal_node is not None:
            raw_path = self._extract_path(best_goal_node)
            planning_time = time.time() - start_time
            print(f"RRT*路径规划完成，用时: {planning_time:.2f}秒")
            print(f"找到原始路径，总长度: {best_cost:.2f}m，节点数: {len(raw_path)}")
            print(f"相对于直线距离的比值: {best_cost/direct_distance:.2f}")
            
            # 路径质量评价
            if best_cost/direct_distance <= 1.1:
                print("🌟 路径质量：优秀")
            elif best_cost/direct_distance <= 1.3:
                print("👍 路径质量：良好")
            else:
                print("⚠️ 路径质量：一般")
            
            # 应用路径平滑处理（参照RRT方法）
            print("应用RRT*路径平滑处理...")
            smooth_path = self.smooth_path(raw_path, smoothness=0.25)
            
            # 保存原始路径和平滑路径用于可视化
            self.raw_path = raw_path
            self.smoothed_path = smooth_path
            
            print(f"平滑后路径节点数: {len(smooth_path)}")
            
            return smooth_path
        else:
            print("❌ 未找到可行路径")
            return None
    
    def _get_nearest_node_index(self, node):
        """找到最近节点的索引"""
        min_dist = float('inf')
        nearest_ind = 0
        for i, n in enumerate(self.node_list):
            dist = self._calc_distance(n, node)
            if dist < min_dist:
                min_dist = dist
                nearest_ind = i
        return nearest_ind
    
    def _calc_distance(self, from_node, to_node):
        """计算两节点间的欧几里得距离"""
        return np.sqrt((from_node.x - to_node.x)**2 + (from_node.y - to_node.y)**2)
    
    def _steer(self, from_node, to_node):
        """从from_node朝to_node方向扩展step_size距离"""
        dist = self._calc_distance(from_node, to_node)
        if dist <= self.step_size:
            return to_node
        
        # 计算方向
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + self.step_size * np.cos(theta)
        new_y = from_node.y + self.step_size * np.sin(theta)
        
        new_node = self.Node(new_x, new_y)
        return new_node
    
    def _is_collision_free(self, from_node, to_node):
        """检查两节点间的路径是否无碰撞"""
        # 首先检查节点是否在道路范围内
        if (to_node.x < 0 or to_node.x > self.env.road_length or 
            to_node.y < 0 or to_node.y > self.env.road_width):
            return False
        
        # 检查与障碍物的碰撞
        num_checks = int(self._calc_distance(from_node, to_node) / 0.2) + 1
        for i in range(num_checks + 1):
            t = i / num_checks if num_checks > 0 else 0
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            
            # 使用环境对象的碰撞检测方法
            if self.env.is_collision(x, y, radius=self.safety_distance):
                return False
        
        return True
    
    def _find_near_nodes(self, node):
        """找到节点重连半径内的所有节点"""
        near_nodes = []
        for n in self.node_list:
            if self._calc_distance(n, node) <= self.rewire_radius:
                near_nodes.append(n)
        return near_nodes
    
    def _rewire(self, new_node, near_nodes):
        """重连附近节点以优化路径（考虑平滑性）"""
        for near_node in near_nodes:
            if not self._is_collision_free(new_node, near_node):
                continue
            
            # 检查转向角度是否合理
            if not self._is_turn_angle_acceptable(new_node, near_node):
                continue
            
            # 计算通过new_node到达near_node的代价（包含平滑性惩罚）
            base_cost = new_node.cost + self._calc_distance(new_node, near_node)
            smoothness_penalty = self._calculate_smoothness_penalty(new_node, near_node)
            new_cost = base_cost + smoothness_penalty
            
            if new_cost < near_node.cost:
                # 更新near_node的父节点
                old_cost = near_node.cost
                near_node.parent = new_node
                near_node.cost = new_cost
                
                # 递归更新所有子节点的代价
                self._update_children_cost(near_node)
    
    def _update_children_cost(self, parent_node):
        """递归更新子节点的代价"""
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = parent_node.cost + self._calc_distance(parent_node, node)
                self._update_children_cost(node)
    
    def _calculate_smoothness_penalty(self, parent_node, new_node):
        """计算路径平滑性惩罚 - 加强折角和急转弯惩罚"""
        if parent_node.parent is None:
            return 0.0  # 起点没有平滑性约束
        
        # 计算转向角度
        # 前一段的方向
        prev_angle = np.arctan2(parent_node.y - parent_node.parent.y, 
                               parent_node.x - parent_node.parent.x)
        # 当前段的方向
        curr_angle = np.arctan2(new_node.y - parent_node.y, 
                               new_node.x - parent_node.x)
        
        # 计算转向角度差
        angle_diff = abs(curr_angle - prev_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        # 增强的平滑性惩罚
        base_penalty = 0.0
        
        # 1. 基础角度惩罚（二次函数，角度越大惩罚越重）
        angle_penalty_weight = 3.0
        base_penalty += angle_penalty_weight * (angle_diff ** 2)
        
        # 2. 急转弯重惩罚（超过30度的转弯）
        sharp_turn_threshold = np.pi / 6  # 30度
        if angle_diff > sharp_turn_threshold:
            sharp_turn_penalty = 5.0 * (angle_diff - sharp_turn_threshold)
            base_penalty += sharp_turn_penalty
        
        # 3. 道路中央偏好奖励
        road_center_y = self.env.road_width / 2
        distance_from_center = abs(new_node.y - road_center_y)
        center_preference_weight = 1.0
        center_penalty = center_preference_weight * distance_from_center
        
        total_penalty = base_penalty + center_penalty
        
        return total_penalty
    
    def _check_obstacle_ahead(self, current_node, goal_x, goal_y):
        """检查前方是否有障碍物"""
        # 计算朝向目标的方向
        direction = np.arctan2(goal_y - current_node.y, goal_x - current_node.x)
        
        # 检查前方一定距离内是否有障碍物
        check_distance = 8.0  # 检查前方8米
        check_steps = 10
        
        for i in range(1, check_steps + 1):
            check_dist = (i / check_steps) * check_distance
            check_x = current_node.x + check_dist * np.cos(direction)
            check_y = current_node.y + check_dist * np.sin(direction)
            
            # 检查是否超出道路边界
            if check_x < 0 or check_x > self.env.road_length or check_y < 0 or check_y > self.env.road_width:
                continue
                
            # 检查是否有碰撞
            if self.env.is_collision(check_x, check_y, radius=self.safety_distance):
                return True
        
        return False
    
    def _sample_for_obstacle_avoidance(self, current_node, goal_x, goal_y):
        """为避障进行采样"""
        # 计算朝向目标的基本方向
        base_direction = np.arctan2(goal_y - current_node.y, goal_x - current_node.x)
        
        # 尝试左右两个方向的避障路径
        avoidance_angles = [base_direction + np.pi/3, base_direction - np.pi/3]  # ±60度避障
        
        for angle in avoidance_angles:
            sample_distance = random.uniform(4.0, 8.0)
            rnd_x = current_node.x + sample_distance * np.cos(angle)
            rnd_y = current_node.y + sample_distance * np.sin(angle)
            
            # 确保在道路范围内
            if 0 <= rnd_x <= self.env.road_length and 0 <= rnd_y <= self.env.road_width:
                # 检查这个方向是否安全
                if not self.env.is_collision(rnd_x, rnd_y, radius=self.safety_distance):
                    return self.Node(rnd_x, rnd_y)
        
        # 如果避障采样失败，返回随机采样点
        rnd_x = random.uniform(0, self.env.road_length)
        rnd_y = random.uniform(0, self.env.road_width)
        return self.Node(rnd_x, rnd_y)
    
    def _is_turn_angle_acceptable(self, parent_node, new_node):
        """检查转向角度是否在可接受范围内"""
        if parent_node.parent is None:
            return True  # 起点没有转向限制
        
        # 计算转向角度
        prev_angle = np.arctan2(parent_node.y - parent_node.parent.y, 
                               parent_node.x - parent_node.parent.x)
        curr_angle = np.arctan2(new_node.y - parent_node.y, 
                               new_node.x - parent_node.x)
        
        # 计算转向角度差
        angle_diff = abs(curr_angle - prev_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        # 设置最大允许转向角度（90度）
        max_turn_angle = np.pi / 2  # 90度
        
        return angle_diff <= max_turn_angle
    
    def _can_connect_to_goal(self, node, goal):
        """检查节点是否可以直接连接到目标"""
        dist = self._calc_distance(node, goal)
        if dist > self.step_size * 2:  # 如果距离太远，不尝试连接
            return False
        
        return self._is_collision_free(node, goal)
    
    def smooth_path(self, path, smoothness=0.25):
        """平滑路径处理（参照RRT算法）"""
        if not path or len(path) < 3:
            return path
        
        print(f"RRT*路径平滑处理，使用平滑系数: {smoothness}")
        
        # 转换为numpy数组便于操作
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        # 创建平滑路径
        smooth_x = x.copy()
        smooth_y = y.copy()
        
        # 第一级平滑：基础迭代平滑
        change = 1.0
        n_iterations = 120  # 基础平滑迭代次数
        
        for iteration in range(n_iterations):
            if change < 0.0005:  # 收敛阈值
                break
                
            change = 0.0
            
            # 不修改起点和终点
            for i in range(1, len(x) - 1):
                # 基本平滑
                old_x, old_y = smooth_x[i], smooth_y[i]
                
                # 计算平滑后的位置（考虑前后点的影响）
                smooth_x[i] += smoothness * (x[i] - smooth_x[i])
                smooth_x[i] += smoothness * (smooth_x[i-1] + smooth_x[i+1] - 2.0 * smooth_x[i])
                
                smooth_y[i] += smoothness * (y[i] - smooth_y[i])
                smooth_y[i] += smoothness * (smooth_y[i-1] + smooth_y[i+1] - 2.0 * smooth_y[i])
                
                # 计算变化量
                change += abs(old_x - smooth_x[i]) + abs(old_y - smooth_y[i])
                
                # 确保平滑后的点仍然在道路内且无碰撞
                if (smooth_x[i] < 0 or smooth_x[i] > self.env.road_length or 
                    smooth_y[i] < self.safety_distance or smooth_y[i] > self.env.road_width - self.safety_distance):
                    # 如果不满足约束，恢复原来的位置
                    smooth_x[i], smooth_y[i] = old_x, old_y
                    continue
                
                # 检查与障碍物的安全距离
                if self.env.is_collision(smooth_x[i], smooth_y[i], radius=self.safety_distance):
                    # 如果与障碍物的安全距离不满足，恢复原来的位置
                    smooth_x[i], smooth_y[i] = old_x, old_y
        
        # 第二级平滑：强化平滑处理
        second_smooth_x = smooth_x.copy()
        second_smooth_y = smooth_y.copy()
        
        # 二次平滑处理，使用更大范围的点进行平滑
        for iteration in range(60):
            for i in range(2, len(x) - 2):
                # 使用5点窗口进行高斯权重平滑
                old_x, old_y = second_smooth_x[i], second_smooth_y[i]
                
                second_smooth_x[i] = (0.05 * second_smooth_x[i-2] + 0.2 * second_smooth_x[i-1] + 
                                     0.5 * second_smooth_x[i] + 
                                     0.2 * second_smooth_x[i+1] + 0.05 * second_smooth_x[i+2])
                                    
                second_smooth_y[i] = (0.05 * second_smooth_y[i-2] + 0.2 * second_smooth_y[i-1] + 
                                     0.5 * second_smooth_y[i] + 
                                     0.2 * second_smooth_y[i+1] + 0.05 * second_smooth_y[i+2])
                
                # 确保平滑后的点仍然在道路内且无碰撞
                if (second_smooth_x[i] < 0 or second_smooth_x[i] > self.env.road_length or 
                    second_smooth_y[i] < self.safety_distance or second_smooth_y[i] > self.env.road_width - self.safety_distance or 
                    self.env.is_collision(second_smooth_x[i], second_smooth_y[i], radius=self.safety_distance)):
                    # 如果不满足约束，恢复原来的位置
                    second_smooth_x[i], second_smooth_y[i] = old_x, old_y
        
        # 第三级平滑：精细平滑处理
        third_smooth_x = second_smooth_x.copy()
        third_smooth_y = second_smooth_y.copy()
        
        # 应用三次平滑，使用更大的7点窗口
        for iteration in range(30):
            for i in range(3, len(x) - 3):
                old_x, old_y = third_smooth_x[i], third_smooth_y[i]
                
                # 使用7点窗口平滑
                third_smooth_x[i] = (0.02 * third_smooth_x[i-3] + 0.05 * third_smooth_x[i-2] + 
                                   0.15 * third_smooth_x[i-1] + 0.56 * third_smooth_x[i] + 
                                   0.15 * third_smooth_x[i+1] + 0.05 * third_smooth_x[i+2] + 
                                   0.02 * third_smooth_x[i+3])
                                  
                third_smooth_y[i] = (0.02 * third_smooth_y[i-3] + 0.05 * third_smooth_y[i-2] + 
                                   0.15 * third_smooth_y[i-1] + 0.56 * third_smooth_y[i] + 
                                   0.15 * third_smooth_y[i+1] + 0.05 * third_smooth_y[i+2] + 
                                   0.02 * third_smooth_y[i+3])
                
                # 确保平滑后的点仍然安全
                if (third_smooth_x[i] < 0 or third_smooth_x[i] > self.env.road_length or 
                    third_smooth_y[i] < self.safety_distance or third_smooth_y[i] > self.env.road_width - self.safety_distance or 
                    self.env.is_collision(third_smooth_x[i], third_smooth_y[i], radius=self.safety_distance)):
                    # 如果不满足约束，恢复原来的位置
                    third_smooth_x[i], third_smooth_y[i] = old_x, old_y
        
        # 合并多次平滑的结果
        final_smooth_path = []
        for i in range(len(x)):
            if i > 2 and i < len(x) - 3:
                # 对于中间点使用三次平滑结果
                final_smooth_path.append([third_smooth_x[i], third_smooth_y[i]])
            elif i > 1 and i < len(x) - 2:
                # 对于近端点使用二次平滑结果
                final_smooth_path.append([second_smooth_x[i], second_smooth_y[i]])
            else:
                # 对于端点使用一次平滑结果
                final_smooth_path.append([smooth_x[i], smooth_y[i]])
        
        print(f"RRT*路径平滑完成：原始{len(path)}点 → 平滑{len(final_smooth_path)}点")
        return final_smooth_path
    
    def _extract_path(self, goal_node):
        """从目标节点回溯提取路径"""
        path = []
        current = goal_node
        while current is not None:
            path.append([current.x, current.y])
            current = current.parent
        
        path.reverse()  # 反转路径，从起点到终点
        return path
    
    def save_and_show_results(self, path, filename, show=True):
        """保存并显示结果，显示原始路径和平滑路径，使用更美观的样式"""
        save_rrt_star_results(self, path, filename, show=show)