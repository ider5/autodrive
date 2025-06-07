import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle

class RRTStar:
    """RRT*路径规划算法 - RRT的优化版本，具有渐近最优性"""
    
    def __init__(self, env, step_size=1.0, max_iter=20000, goal_sample_rate=20, 
                 safety_distance=1.5, rewire_radius=3.0, early_stop_enabled=True,
                 no_improvement_limit=3000, improvement_threshold=0.1, 
                 target_quality_factor=1.1, smooth_iterations=3):
        """
        初始化RRT*路径规划器
        
        参数:
            env: environment object
            step_size: 每步的最大距离
            max_iter: 最大迭代次数
            goal_sample_rate: 目标采样率(%)
            safety_distance: 安全距离
            rewire_radius: 重连半径
            early_stop_enabled: 是否启用提前停止
            no_improvement_limit: 无改善迭代次数限制
            improvement_threshold: 改善阈值(米)
            target_quality_factor: 目标质量因子(相对于直线距离)
            smooth_iterations: 路径平滑迭代次数
        """
        self.env = env
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.safety_distance = safety_distance
        self.rewire_radius = rewire_radius
        
        # 提前停止相关参数
        self.early_stop_enabled = early_stop_enabled
        self.no_improvement_limit = no_improvement_limit
        self.improvement_threshold = improvement_threshold
        self.target_quality_factor = target_quality_factor
        
        # 车辆运动学约束参数
        self.max_turn_angle = np.deg2rad(25.0)  # 最大转向角度（度转弧度）- 严格限制确保平滑
        
        # 路径平滑参数
        self.smooth_iterations = smooth_iterations  # 平滑迭代次数
        
        # 搜索空间
        self.x_min = 0
        self.x_max = env.road_length
        self.y_min = 0
        self.y_max = env.road_width
        
        print(f"初始化RRT*路径规划器（运动学约束+平滑版本）")
        print(f"步长: {step_size}m, 最大迭代: {max_iter}, 目标采样率: {goal_sample_rate}%")
        print(f"安全距离: {safety_distance}m, 重连半径: {rewire_radius}m")
        print(f"最大转向角: {np.rad2deg(self.max_turn_angle):.1f}度")
        print(f"路径平滑迭代: {smooth_iterations}次")
        if early_stop_enabled:
            print(f"提前停止: 启用 (无改善限制: {no_improvement_limit}次, 改善阈值: {improvement_threshold}m)")
        else:
            print(f"提前停止: 禁用")
    
    class Node:
        """RRT*节点"""
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None
            self.cost = 0.0
            
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
        node_list = [start_node]
        
        # 用于跟踪最佳路径
        best_goal_node = None
        best_cost = float('inf')
        
        # 提前停止相关变量
        iterations_since_improvement = 0
        target_distance = self._distance(start_node, goal_node)  # 直线距离
        target_cost = target_distance * self.target_quality_factor  # 目标代价
        
        print(f"直线距离: {target_distance:.2f}m, 目标路径长度: {target_cost:.2f}m")
        
        for i in range(self.max_iter):
            if i % 2000 == 0:
                print(f"RRT*迭代 {i}/{self.max_iter}, 节点数: {len(node_list)}")
                if best_goal_node:
                    print(f"当前最佳路径长度: {best_goal_node.cost:.2f}m")
            
            # 采样随机点或目标点
            if np.random.rand() <= self.goal_sample_rate / 100.0:
                rnd_node = goal_node
            else:
                rnd_node = self._sample_random_node()
            
            # 找到最近的节点
            nearest_node = self._get_nearest_node(node_list, rnd_node)
            
            # 生成新节点
            new_node = self._steer(nearest_node, rnd_node)
            
            if new_node is None:
                continue
            
            # 碰撞检测
            if not self._is_collision_free_path(nearest_node, new_node):
                continue
            
            # 检查转向角度约束
            if not self._check_turn_angle_constraint(nearest_node, new_node):
                continue
            
            # RRT*的核心改进：选择最优父节点
            near_nodes = self._find_near_nodes(node_list, new_node)
            new_node = self._choose_parent(near_nodes, new_node)
            
            if new_node is None:
                continue
            
            # 添加新节点到树中
            node_list.append(new_node)
            
            # RRT*的核心改进：重连操作
            self._rewire(node_list, new_node, near_nodes)
            
            # 检查是否到达目标
            if self._is_near_goal(new_node, goal_node):
                # 尝试连接到目标
                if self._is_collision_free_path(new_node, goal_node):
                    # 检查到目标的转向角度约束
                    if self._check_turn_angle_constraint(new_node, goal_node):
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + self._distance(new_node, goal_node)
                        
                        # 更新最佳路径
                        if goal_node.cost < best_cost:
                            # 检查是否有显著改善
                            improvement = best_cost - goal_node.cost
                            if improvement >= self.improvement_threshold:
                                iterations_since_improvement = 0  # 重置计数器
                                print(f"找到更优路径！迭代: {i}, 长度: {goal_node.cost:.2f}m (改善: {improvement:.2f}m)")
                            else:
                                iterations_since_improvement += 1
                                print(f"路径微调：迭代: {i}, 长度: {goal_node.cost:.2f}m (改善: {improvement:.3f}m)")
                            
                            best_cost = goal_node.cost
                            best_goal_node = goal_node
                            
                            # 检查是否达到目标质量
                            if self.early_stop_enabled and best_cost <= target_cost:
                                print(f"✅ 达到目标路径质量！迭代: {i}, 长度: {best_cost:.2f}m ≤ {target_cost:.2f}m")
                                break
                        else:
                            iterations_since_improvement += 1
            
            # 提前停止检查
            if self.early_stop_enabled and iterations_since_improvement >= self.no_improvement_limit:
                print(f"⏹️ 提前停止：连续{self.no_improvement_limit}次迭代无显著改善")
                break
        
        planning_time = time.time() - start_time
        
        if best_goal_node:
            # 提取原始路径
            raw_path = self._extract_path(best_goal_node)
            
            # 应用路径平滑
            smoothed_path = self._smooth_path(raw_path)
            
            # 验证平滑后的路径
            if self._validate_smoothed_path(smoothed_path):
                final_path = smoothed_path
                print(f"✅ 路径平滑成功：原始{len(raw_path)}点 → 平滑{len(final_path)}点")
            else:
                final_path = raw_path
                print(f"⚠️ 路径平滑失效，使用原始路径")
            
            print(f"RRT*路径规划完成，用时: {planning_time:.2f}秒")
            print(f"找到路径，总长度: {best_cost:.2f}m，节点数: {len(final_path)}")
            print(f"相对于直线距离的比值: {best_cost/target_distance:.2f}")
            
            # 路径质量评估
            if best_cost <= target_cost:
                print("🌟 路径质量：优秀")
            elif best_cost <= target_distance * 1.3:
                print("👍 路径质量：良好")
            else:
                print("📈 路径质量：可接受")
            
            # 分析路径转向情况
            self._analyze_path_turns(final_path)
            
            return final_path
        else:
            print(f"RRT*路径规划失败，未找到路径")
            return None
    
    def _check_turn_angle_constraint(self, from_node, to_node):
        """检查转向角度约束"""
        if from_node.parent is None:
            return True  # 起点没有转向约束
        
        # 计算前一段和当前段的方向
        prev_angle = np.arctan2(from_node.y - from_node.parent.y, 
                               from_node.x - from_node.parent.x)
        curr_angle = np.arctan2(to_node.y - from_node.y, 
                               to_node.x - from_node.x)
        
        # 计算转向角度
        turn_angle = abs(curr_angle - prev_angle)
        if turn_angle > np.pi:
            turn_angle = 2 * np.pi - turn_angle
        
        return turn_angle <= self.max_turn_angle
    
    def _smooth_path(self, path):
        """路径平滑处理 - 使用移动平均算法抹去尖锐转角"""
        if len(path) < 3:
            return path
        
        smoothed_path = path.copy()
        
        # 多次迭代平滑
        for iteration in range(self.smooth_iterations):
            new_smoothed = [smoothed_path[0]]  # 保持起点
            
            # 对中间点进行平滑
            for i in range(1, len(smoothed_path) - 1):
                # 使用前后点的平均值进行平滑
                prev_point = smoothed_path[i-1]
                curr_point = smoothed_path[i]
                next_point = smoothed_path[i+1]
                
                # 计算平滑后的点（加权平均）
                weight_center = 0.5  # 中心点权重
                weight_neighbor = 0.25  # 邻点权重
                
                smooth_x = (weight_neighbor * prev_point[0] + 
                           weight_center * curr_point[0] + 
                           weight_neighbor * next_point[0])
                smooth_y = (weight_neighbor * prev_point[1] + 
                           weight_center * curr_point[1] + 
                           weight_neighbor * next_point[1])
                
                new_smoothed.append([smooth_x, smooth_y])
            
            new_smoothed.append(smoothed_path[-1])  # 保持终点
            smoothed_path = new_smoothed
        
        return smoothed_path
    
    def _validate_smoothed_path(self, path):
        """验证平滑后的路径是否有效"""
        if len(path) < 2:
            return False
        
        # 检查路径是否有碰撞
        for i in range(len(path) - 1):
            if not self._is_collision_free_segment(path[i], path[i + 1]):
                return False
        
        # 检查路径是否超出边界
        for point in path:
            if not (self.x_min <= point[0] <= self.x_max and 
                   self.y_min <= point[1] <= self.y_max):
                return False
        
        # 检查平滑后的路径是否仍满足转向角度约束
        max_turn_violations = len(path) * 0.1  # 允许10%的点超限
        turn_violations = 0
        
        for i in range(1, len(path) - 1):
            # 计算前一段方向
            prev_angle = np.arctan2(path[i][1] - path[i-1][1], 
                                   path[i][0] - path[i-1][0])
            # 计算当前段方向
            curr_angle = np.arctan2(path[i+1][1] - path[i][1], 
                                   path[i+1][0] - path[i][0])
            
            # 计算转向角度
            turn_angle = abs(curr_angle - prev_angle)
            if turn_angle > np.pi:
                turn_angle = 2 * np.pi - turn_angle
            
            if turn_angle > self.max_turn_angle * 1.2:  # 允许平滑后的路径稍微超限
                turn_violations += 1
        
        return turn_violations <= max_turn_violations
    
    def _is_collision_free_segment(self, p1, p2):
        """检查两点间线段是否无碰撞"""
        steps = int(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) / 0.5) + 1
        
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            # 检查是否与障碍物碰撞
            if self.env.is_collision(x, y, radius=self.safety_distance):
                return False
        
        return True
    
    def _analyze_path_turns(self, path):
        """分析路径的转向情况"""
        if len(path) < 3:
            return
        
        max_turn_angle = 0
        turn_violations = 0
        
        turn_angles = []
        for i in range(1, len(path) - 1):
            # 计算前一段方向
            prev_angle = np.arctan2(path[i][1] - path[i-1][1], 
                                   path[i][0] - path[i-1][0])
            # 计算当前段方向
            curr_angle = np.arctan2(path[i+1][1] - path[i][1], 
                                   path[i+1][0] - path[i][0])
            
            # 计算转向角度
            turn_angle = abs(curr_angle - prev_angle)
            if turn_angle > np.pi:
                turn_angle = 2 * np.pi - turn_angle
            
            turn_angles.append(turn_angle)
            
            if turn_angle > max_turn_angle:
                max_turn_angle = turn_angle
            
            if turn_angle > self.max_turn_angle:
                turn_violations += 1
        
        print(f"📊 路径转向分析：")
        print(f"   最大转向角: {np.rad2deg(max_turn_angle):.1f}度 (限制: {np.rad2deg(self.max_turn_angle):.1f}度)")
        if turn_angles:
            avg_turn = np.mean(turn_angles)
            print(f"   平均转向角: {np.rad2deg(avg_turn):.1f}度")
        print(f"   超限转向点: {turn_violations}/{len(turn_angles)}")
        
        if turn_violations == 0:
            print("   ✅ 所有转向都在车辆性能范围内")
        else:
            print(f"   ⚠️ {turn_violations}个转向点超出车辆性能限制")
    
    def _sample_random_node(self):
        """采样随机节点"""
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        return self.Node(x, y)
    
    def _get_nearest_node(self, node_list, rnd_node):
        """找到最近的节点"""
        distances = [self._distance(node, rnd_node) for node in node_list]
        min_index = distances.index(min(distances))
        return node_list[min_index]
    
    def _steer(self, from_node, to_node):
        """从from_node向to_node方向扩展step_size距离"""
        dist = self._distance(from_node, to_node)
        
        if dist <= self.step_size:
            new_node = self.Node(to_node.x, to_node.y)
        else:
            theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_x = from_node.x + self.step_size * np.cos(theta)
            new_y = from_node.y + self.step_size * np.sin(theta)
            new_node = self.Node(new_x, new_y)
        
        # 检查是否在边界内
        if not self._is_in_bounds(new_node):
            return None
        
        return new_node
    
    def _find_near_nodes(self, node_list, new_node):
        """找到新节点附近的节点"""
        near_nodes = []
        for node in node_list:
            if self._distance(node, new_node) <= self.rewire_radius:
                near_nodes.append(node)
        return near_nodes
    
    def _choose_parent(self, near_nodes, new_node):
        """选择最优父节点"""
        if not near_nodes:
            return None
        
        costs = []
        for near_node in near_nodes:
            t_cost = near_node.cost + self._distance(near_node, new_node)
            costs.append(t_cost)
        
        min_cost = min(costs)
        min_index = costs.index(min_cost)
        
        # 检查最优路径是否无碰撞
        if self._is_collision_free_path(near_nodes[min_index], new_node):
            new_node.cost = min_cost
            new_node.parent = near_nodes[min_index]
            return new_node
        
        return None
    
    def _rewire(self, node_list, new_node, near_nodes):
        """重连操作"""
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue
            
            # 计算通过new_node到达near_node的代价
            new_cost = new_node.cost + self._distance(new_node, near_node)
            
            if new_cost < near_node.cost:
                # 检查路径是否无碰撞
                if self._is_collision_free_path(new_node, near_node):
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    # 递归更新子节点的代价
                    self._update_cost_recursive(node_list, near_node)
    
    def _update_cost_recursive(self, node_list, parent_node):
        """递归更新子节点的代价"""
        for node in node_list:
            if node.parent == parent_node:
                node.cost = parent_node.cost + self._distance(parent_node, node)
                self._update_cost_recursive(node_list, node)
    
    def _distance(self, node1, node2):
        """计算两个节点之间的欧几里得距离"""
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def _is_in_bounds(self, node):
        """检查节点是否在边界内"""
        return (self.x_min <= node.x <= self.x_max and 
                self.y_min <= node.y <= self.y_max)
    
    def _is_collision_free_path(self, from_node, to_node):
        """检查两个节点之间的路径是否无碰撞"""
        # 检查起点和终点
        if not self._is_in_bounds(from_node) or not self._is_in_bounds(to_node):
            return False
        
        # 沿路径采样点进行碰撞检测
        dist = self._distance(from_node, to_node)
        steps = int(dist / 0.5) + 1  # 每0.5米采样一次
        
        for i in range(steps + 1):
            t = i / steps
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            
            # 检查是否与障碍物碰撞
            if self.env.is_collision(x, y, radius=self.safety_distance):
                return False
            
            # 检查是否过于接近边界
            if (y >= self.y_max - self.safety_distance or 
                y <= self.y_min + self.safety_distance):
                return False
        
        return True
    
    def _is_near_goal(self, node, goal_node):
        """检查节点是否接近目标"""
        return self._distance(node, goal_node) <= self.step_size * 2
    
    def _extract_path(self, goal_node):
        """从目标节点回溯提取路径"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append([current.x, current.y])
            current = current.parent
        
        # 反转路径，使其从起点到终点
        path.reverse()
        return path
    
    def save_and_show_results(self, path, filename):
        """保存和显示结果"""
        if path is None:
            print("没有找到路径，无法保存结果")
            return
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制道路边界
        ax.plot([0, self.env.road_length], [0, 0], 'k-', linewidth=2, label='道路边界')
        ax.plot([0, self.env.road_length], [self.env.road_width, self.env.road_width], 'k-', linewidth=2)
        
        # 绘制车道线
        lane_width = self.env.road_width / self.env.num_lanes
        for i in range(1, self.env.num_lanes):
            y = i * lane_width
            ax.plot([0, self.env.road_length], [y, y], 'k--', alpha=0.5)
        
        # 绘制障碍物车辆
        if hasattr(self.env, 'obstacle_vehicles'):
            for i, vehicle in enumerate(self.env.obstacle_vehicles):
                rect = Rectangle((vehicle.x - vehicle.length/2, vehicle.y - vehicle.width/2), 
                               vehicle.length, vehicle.width, 
                               facecolor='red', alpha=0.7)
                if i == 0:  # 只为第一个障碍物添加标签
                    rect.set_label('障碍物')
                ax.add_patch(rect)
        
        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.8, label='RRT*路径')
            ax.plot(path_x, path_y, 'bo', markersize=4, alpha=0.6)
            
            # 标记起点和终点
            ax.plot(path[0][0], path[0][1], 'go', markersize=8, label='起点')
            ax.plot(path[-1][0], path[-1][1], 'ro', markersize=8, label='终点')
        
        ax.set_xlim(-2, self.env.road_length + 2)
        ax.set_ylim(-2, self.env.road_width + 2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('RRT*路径规划结果（运动学约束+平滑版本）')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"RRT*路径规划结果已保存为 {filename}") 