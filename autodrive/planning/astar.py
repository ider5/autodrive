"""
A*路径规划算法实现

本模块实现A*算法用于自动驾驶车辆的路径规划。
A*是一种启发式搜索算法，能够找到从起点到终点的最优路径。
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from matplotlib.patches import Rectangle, Circle, Polygon

class AStar:
    """
    A*路径规划算法
    
    该类实现A*搜索算法，使用网格化地图进行路径规划。
    算法结合了代价函数和启发式函数，保证找到最优路径。
    """
    def __init__(self, env, grid_resolution=0.5, safety_distance=1.5):
        """
        初始化A*路径规划器
        
        参数:
            env: 环境对象，包含道路信息和障碍物
            grid_resolution (float): 网格分辨率（米）
            safety_distance (float): 与障碍物的安全距离（米）
        """
        self.env = env  # 环境对象
        self.grid_resolution = grid_resolution  # 栅格分辨率
        self.safety_distance = safety_distance  # 安全距离
        
        # 车道相关参数
        self.lane_width = env.lane_width
        self.num_lanes = max(1, env.num_lanes)
        self.lane_centers = []
        for i in range(self.num_lanes):
            try:
                self.lane_centers.append(env.get_lane_center(i+1))
            except Exception as e:
                print(f"警告: 获取车道 {i+1} 中心位置时出错: {e}")
        
        if not self.lane_centers:
            print("警告: 未能获取任何车道中心位置，使用默认值")
            self.lane_centers = [env.road_width / 2]
        
        # 路径平滑相关参数
        self.smoothing_strength = 0.2
        self.smoothing_window = 5
        
        # 车道约束参数
        self.lane_change_penalty = 5.0   # 变道惩罚系数（降低）
        self.off_lane_penalty = 10.0     # 偏离车道中心惩罚系数（降低）
        self.max_lane_deviation = self.lane_width * 0.4  # 允许最大偏离车道中心距离（增大）
        self.lane_change_angle_limit = np.deg2rad(25)  # 变道时的最大转角限制（增大）
        
        # 创建栅格地图
        self.grid_map = None
        self.lane_grid = None  # 车道栅格图
        self.x_min = 0
        self.x_max = env.road_length
        self.y_min = 0
        self.y_max = env.road_width
        self.width = int((self.x_max - self.x_min) / grid_resolution) + 1
        self.height = int((self.y_max - self.y_min) / grid_resolution) + 1
        
        # 修改运动模型：主要沿车道方向，限制变道角度
        self.motion = [
            [1, 0, 1.0],       # 前进（主要方向）
            [2, 0, 2.0],       # 快速前进
            [1, 1, 1.414],     # 前进+轻微上变道
            [1, -1, 1.414],    # 前进+轻微下变道
            [2, 1, 2.236],     # 快速前进+轻微上变道
            [2, -1, 2.236],    # 快速前进+轻微下变道
            [3, 1, 3.162],     # 更快前进+轻微上变道
            [3, -1, 3.162],    # 更快前进+轻微下变道
        ]
        
        print(f"初始化A*路径规划器, 栅格分辨率: {grid_resolution}m, 安全距离: {safety_distance}m")
        print(f"栅格地图大小: {self.width}x{self.height}")
        print(f"车道数量: {self.num_lanes}, 车道中心: {self.lane_centers}")
    
    class Node:
        """A*节点"""
        def __init__(self, x, y, cost, parent_index, lane_id=None):
            self.x = x  # 栅格x坐标
            self.y = y  # 栅格y坐标
            self.cost = cost  # 从起点到当前点的实际代价g(n)
            self.parent_index = parent_index  # 父节点索引
            self.lane_id = lane_id  # 所在车道ID
        
        def __str__(self):
            return f"({self.x}, {self.y}, {self.cost}, {self.parent_index}, lane:{self.lane_id})"
    
    def planning(self, start_x, start_y, goal_x, goal_y):
        """A*路径规划主函数"""
        print(f"开始A*路径规划: 从({start_x:.1f}, {start_y:.1f})到({goal_x:.1f}, {goal_y:.1f})")
        start_time = time.time()
        
        # 创建栅格地图
        self._create_grid_map()
        
        # 转换为栅格坐标
        start_node = self.Node(
            self._get_grid_index(start_x, self.x_min, self.grid_resolution),
            self._get_grid_index(start_y, self.y_min, self.grid_resolution),
            0.0, -1, self._get_lane_id(start_y)
        )
        
        goal_node = self.Node(
            self._get_grid_index(goal_x, self.x_min, self.grid_resolution),
            self._get_grid_index(goal_y, self.y_min, self.grid_resolution),
            0.0, -1, self._get_lane_id(goal_y)
        )
        
        # 检查起点和终点是否有效
        if not self._verify_node(start_node) or not self._verify_node(goal_node):
            print("起点或终点无效！")
            return None
        
        # 初始化开放列表和关闭列表
        open_set = dict()  # 开放列表 {index: node}
        closed_set = dict()  # 关闭列表 {index: node}
        
        # 优先队列，存储 (f_cost, index)
        pq = []
        
        # 计算起点的索引和启发式代价
        start_index = self._calc_grid_index(start_node)
        open_set[start_index] = start_node
        
        # 将起点加入优先队列
        h_cost = self._calc_heuristic(start_node, goal_node)
        f_cost = start_node.cost + h_cost
        heapq.heappush(pq, (f_cost, start_index))
        
        iterations = 0
        max_iterations = self.width * self.height  # 最大迭代次数
        
        while len(pq) > 0:
            iterations += 1
            
            if iterations % 1000 == 0:
                print(f"迭代 {iterations}/{max_iterations}, 开放列表大小: {len(open_set)}")
            
            if iterations > max_iterations:
                print("超过最大迭代次数，搜索失败")
                break
            
            # 从优先队列中取出f代价最小的节点
            _, current_index = heapq.heappop(pq)
            
            # 跳过已经不在开放列表中的节点（可能已被更好路径替代）
            if current_index not in open_set:
                continue
                
            current_node = open_set[current_index]
            
            # 将当前节点从开放列表移到关闭列表
            del open_set[current_index]
            closed_set[current_index] = current_node
            
            # 检查是否到达目标
            if self._is_goal(current_node, goal_node):
                print(f"找到路径！迭代次数: {iterations}")
                goal_node.parent_index = current_index
                goal_node.cost = current_node.cost + self._calc_distance(current_node, goal_node)
                break
            
            # 扩展当前节点
            for motion in self.motion:
                new_node = self.Node(
                    current_node.x + motion[0],
                    current_node.y + motion[1],
                    current_node.cost + motion[2],
                    current_index
                )
                
                # 计算新节点的世界坐标和车道ID
                world_x = new_node.x * self.grid_resolution + self.x_min
                world_y = new_node.y * self.grid_resolution + self.y_min
                new_node.lane_id = self._get_lane_id(world_y)
                
                new_index = self._calc_grid_index(new_node)
                
                # 检查节点是否有效（包括车道约束）
                if not self._verify_node_with_lane_constraint(new_node, current_node):
                    continue
                
                # 计算车道约束代价
                lane_cost = self._calc_lane_cost(new_node, current_node)
                new_node.cost += lane_cost
                
                # 检查是否在关闭列表中
                if new_index in closed_set:
                    continue
                
                # 检查是否在开放列表中
                if new_index in open_set:
                    # 如果新路径更好，更新节点
                    if open_set[new_index].cost > new_node.cost:
                        open_set[new_index] = new_node
                        # 重新加入优先队列
                        h_cost = self._calc_heuristic_with_lane(new_node, goal_node)
                        f_cost = new_node.cost + h_cost
                        heapq.heappush(pq, (f_cost, new_index))
                else:
                    # 新节点，加入开放列表
                    open_set[new_index] = new_node
                    h_cost = self._calc_heuristic_with_lane(new_node, goal_node)
                    f_cost = new_node.cost + h_cost
                    heapq.heappush(pq, (f_cost, new_index))
        
        # 提取路径
        if goal_node.parent_index == -1:
            print("无法找到路径！")
            return None
        
        # 构建路径
        path = self._extract_path(closed_set, goal_node, start_node)
        
        planning_time = time.time() - start_time
        print(f"A*路径规划完成，用时: {planning_time:.2f}秒")
        print(f"路径长度: {len(path)}个点")
        
        # 调试：打印路径的前几个点和后几个点
        if path and len(path) >= 4:
            print(f"路径起始点: {path[0]}")
            print(f"路径第二点: {path[1]}")
            print(f"路径倒数第二点: {path[-2]}")
            print(f"路径终点: {path[-1]}")
            
            # 检查路径方向
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
            print(f"路径开始方向: dx={dx:.2f}, dy={dy:.2f}")
            
            if dx < 0:
                print("警告：A*路径方向可能相反（X坐标递减）")
        
        return path
    
    def _create_grid_map(self):
        """创建栅格地图"""
        self.grid_map = np.zeros((self.height, self.width))
        
        # 标记障碍物
        for x_idx in range(self.width):
            for y_idx in range(self.height):
                # 转换为世界坐标
                world_x = x_idx * self.grid_resolution + self.x_min
                world_y = y_idx * self.grid_resolution + self.y_min
                
                # 检查是否有碰撞（考虑安全距离）
                if self.env.is_collision(world_x, world_y, radius=self.safety_distance):
                    self.grid_map[y_idx][x_idx] = 1  # 1表示障碍物
    
    def _get_grid_index(self, position, min_pos, resolution):
        """将世界坐标转换为栅格索引"""
        return int(round((position - min_pos) / resolution))
    
    def _calc_grid_index(self, node):
        """计算节点的栅格索引"""
        return node.y * self.width + node.x
    
    def _verify_node(self, node):
        """验证节点是否有效"""
        # 检查边界
        if node.x < 0 or node.x >= self.width:
            return False
        if node.y < 0 or node.y >= self.height:
            return False
        
        # 检查障碍物
        if self.grid_map[node.y][node.x] == 1:
            return False
        
        return True
    
    def _verify_node_with_lane_constraint(self, new_node, current_node):
        """验证节点是否有效（包含车道约束）"""
        # 基本有效性检查
        if not self._verify_node(new_node):
            return False
        
        # 检查是否在道路范围内（更严格的安全边界检查）
        world_y = new_node.y * self.grid_resolution + self.y_min
        # 增加更大的安全边距，特别是对于上边界
        safety_margin = max(self.safety_distance * 1.5, 2.0)  # 至少2米安全边距
        if world_y < safety_margin or world_y > (self.env.road_width - safety_margin):
            return False
        
        # 检查变道角度限制（只在实际变道时应用）
        if current_node and current_node.lane_id is not None:
            if abs(new_node.lane_id - current_node.lane_id) > 1:
                # 不允许跨越多个车道
                return False
            
            if new_node.lane_id != current_node.lane_id:
                # 发生变道，检查角度
                if not self._is_lane_change_angle_valid(new_node, current_node):
                    return False
        
        return True
    
    def _is_in_valid_lane(self, world_y):
        """检查Y坐标是否在有效车道范围内"""
        for lane_center in self.lane_centers:
            if abs(world_y - lane_center) <= self.max_lane_deviation:
                return True
        return False
    
    def _is_lane_change_angle_valid(self, new_node, current_node):
        """检查变道角度是否在允许范围内"""
        dx = new_node.x - current_node.x
        dy = new_node.y - current_node.y
        
        if dx == 0:
            return False  # 不允许纯横向移动
        
        angle = abs(np.arctan2(dy * self.grid_resolution, dx * self.grid_resolution))
        return angle <= self.lane_change_angle_limit
    
    def _get_lane_id(self, world_y):
        """根据Y坐标获取车道ID"""
        min_distance = float('inf')
        closest_lane = 0
        
        for i, lane_center in enumerate(self.lane_centers):
            distance = abs(world_y - lane_center)
            if distance < min_distance:
                min_distance = distance
                closest_lane = i
        
        return closest_lane
    
    def _calc_lane_cost(self, new_node, current_node):
        """计算车道相关代价"""
        cost = 0.0
        
        # 计算偏离车道中心的代价
        world_y = new_node.y * self.grid_resolution + self.y_min
        closest_lane_center = self.lane_centers[new_node.lane_id]
        deviation = abs(world_y - closest_lane_center)
        cost += self.off_lane_penalty * (deviation / self.max_lane_deviation) ** 2
        
        # 计算变道代价
        if current_node and current_node.lane_id is not None:
            if new_node.lane_id != current_node.lane_id:
                cost += self.lane_change_penalty
        
        return cost
    
    def _calc_heuristic_with_lane(self, node, goal):
        """计算包含车道信息的启发式代价"""
        # 基本欧几里得距离
        dx = node.x - goal.x
        dy = node.y - goal.y
        basic_cost = np.sqrt(dx**2 + dy**2) * self.grid_resolution
        
        # 车道偏好：如果不在目标车道，增加代价
        lane_penalty = 0.0
        if node.lane_id != goal.lane_id:
            lane_penalty = self.lane_change_penalty * abs(node.lane_id - goal.lane_id)
        
        return basic_cost + lane_penalty
    
    def _calc_heuristic(self, node1, node2):
        """计算启发式代价（欧几里得距离）"""
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        return np.sqrt(dx**2 + dy**2) * self.grid_resolution
    
    def _calc_distance(self, node1, node2):
        """计算两节点间的实际距离"""
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        return np.sqrt(dx**2 + dy**2) * self.grid_resolution
    
    def _is_goal(self, node, goal):
        """检查是否到达目标"""
        return abs(node.x - goal.x) <= 1 and abs(node.y - goal.y) <= 1
    
    def _extract_path(self, closed_set, goal_node, start_node):
        """提取路径"""
        path = []
        current_node = goal_node
        
        while current_node.parent_index != -1:
            # 转换为世界坐标
            world_x = current_node.x * self.grid_resolution + self.x_min
            world_y = current_node.y * self.grid_resolution + self.y_min
            path.append([world_x, world_y])
            
            current_node = closed_set[current_node.parent_index]
        
        # 添加起点
        world_x = start_node.x * self.grid_resolution + self.x_min
        world_y = start_node.y * self.grid_resolution + self.y_min
        path.append([world_x, world_y])
        
        # 反转路径（从起点到终点）
        path.reverse()
        
        return path
    
    def smooth_path(self, path, smoothness=0.30):
        """路径平滑处理 - A*版本不进行平滑"""
        if not path or len(path) < 3:
            return path
        
        print("A*路径不进行平滑处理，直接应用车道约束")
        
        # 只应用车道中心约束，不进行平滑
        constrained_path = self._apply_lane_center_constraint(path)
        
        return constrained_path
    
    def _apply_lane_center_constraint(self, path):
        """应用车道中心约束 - 改进版"""
        if not path or len(path) < 2:
            return path
        
        constrained_path = []
        current_lane_id = None
        
        for i, point in enumerate(path):
            x, y = point
            
            # 确定当前应该在哪个车道
            if current_lane_id is None:
                current_lane_id = self._get_lane_id(y)
            
            # 检查是否需要变道
            target_lane_id = self._get_lane_id(y)
            
            # 如果距离起始车道中心太远，考虑变道
            current_lane_center = self.lane_centers[current_lane_id]
            if abs(y - current_lane_center) > self.max_lane_deviation:
                # 检查是否应该变道
                if target_lane_id != current_lane_id:
                    # 平滑变道：逐渐过渡到新车道
                    if i > 0 and i < len(path) - 1:
                        # 计算变道进度
                        lane_change_progress = min(1.0, abs(y - current_lane_center) / (self.lane_width * 0.5))
                        target_lane_center = self.lane_centers[target_lane_id]
                        
                        # 插值计算新的Y位置
                        y = current_lane_center + lane_change_progress * (target_lane_center - current_lane_center)
                        current_lane_id = target_lane_id
            else:
                # 保持在当前车道中心附近
                y = current_lane_center + np.clip(y - current_lane_center, 
                                                  -self.max_lane_deviation, 
                                                  self.max_lane_deviation)
            
            # 确保不超出道路边界
            y = max(self.safety_distance, min(y, self.env.road_width - self.safety_distance))
            
            constrained_path.append([x, y])
        
        return constrained_path
    
    def save_and_show_results(self, path, smooth_path, filename):
        """保存并显示路径规划结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左图：显示栅格地图和原始路径
        ax1.set_title('A*路径规划 - 栅格地图', fontsize=14, fontweight='bold')
        
        # 绘制栅格地图
        ax1.imshow(self.grid_map, cmap='binary', origin='lower', 
                  extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        
        # 绘制原始路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax1.plot(path_x, path_y, 'r-', linewidth=2, label='A*原始路径')
        
        # 绘制起点和终点
        start_point = self.env.start_point
        end_point = self.env.end_point
        ax1.plot(start_point[0], start_point[1], 'go', markersize=10, label='起点')
        ax1.plot(end_point[0], end_point[1], 'ro', markersize=10, label='终点')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 右图：显示环境和平滑路径
        ax2.set_title('A*路径规划 - 车道约束路径', fontsize=14, fontweight='bold')
        self.env.plot_environment(ax2)
        
        # 绘制车道中心线和约束范围
        for i, center in enumerate(self.lane_centers):
            # 绘制车道中心线
            ax2.axhline(y=center, color='yellow', linestyle='-', linewidth=2, alpha=0.8, 
                       label=f'车道{i+1}中心' if i == 0 else "")
            
            # 绘制允许的偏离范围
            ax2.axhline(y=center + self.max_lane_deviation, color='orange', 
                       linestyle=':', alpha=0.5)
            ax2.axhline(y=center - self.max_lane_deviation, color='orange', 
                       linestyle=':', alpha=0.5)
            
            # 填充允许的车道范围
            ax2.fill_between([0, self.env.road_length], 
                           center - self.max_lane_deviation, 
                           center + self.max_lane_deviation, 
                           alpha=0.1, color='green', label='允许范围' if i == 0 else "")
        
        # 绘制原始路径和平滑路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax2.plot(path_x, path_y, '--', color='red', linewidth=1.5, 
                    label='A*原始路径', alpha=0.7)
        
        if smooth_path:
            smooth_x = [p[0] for p in smooth_path]
            smooth_y = [p[1] for p in smooth_path]
            ax2.plot(smooth_x, smooth_y, '-', color='blue', linewidth=3, 
                    label='A*车道约束路径')
        
        # 添加参数说明
        param_text = f"车道偏离限制: {self.max_lane_deviation:.2f}m\n变道角度限制: {np.rad2deg(self.lane_change_angle_limit):.1f}°"
        ax2.text(0.02, 0.98, param_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"A*路径规划结果已保存为 {filename}")
        plt.close() 