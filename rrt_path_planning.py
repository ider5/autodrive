import numpy as np
import matplotlib.pyplot as plt
import random
import time
from matplotlib.patches import Rectangle, Circle, Polygon

class RRT:
    """RRT路径规划算法 - 改进版"""
    def __init__(self, env, step_size=1.5, max_iter=30000, goal_sample_rate=40, max_turn_angle=30, safety_distance=0.5):
        """
        初始化RRT路径规划器
        
        参数:
            env: 环境对象
            step_size: 随机树扩展步长
            max_iter: 最大迭代次数
            goal_sample_rate: 目标点采样率 (%)
            max_turn_angle: 最大转向角度 (度)
            safety_distance: 与障碍物的安全距离 (m)
        """
        self.env = env  # 环境对象
        self.step_size = step_size  # 步长
        self.max_iter = max_iter  # 最大迭代次数
        self.goal_sample_rate = goal_sample_rate  # 目标采样率
        self.max_turn_angle = np.deg2rad(max_turn_angle)  # 最大转弯角度（弧度）
        self.safety_distance = safety_distance  # 安全距离
        self.node_list = []  # 节点列表
        
        # 车道相关参数
        self.lane_width = env.lane_width
        self.num_lanes = max(1, env.num_lanes)  # 确保至少有一个车道
        # 安全地创建车道中心列表
        self.lane_centers = []
        for i in range(self.num_lanes):
            try:
                self.lane_centers.append(env.get_lane_center(i+1))
            except Exception as e:
                print(f"警告: 获取车道 {i+1} 中心位置时出错: {e}")
        
        # 确保车道中心列表非空
        if not self.lane_centers:
            print("警告: 未能获取任何车道中心位置，使用默认值")
            self.lane_centers = [env.road_width / 2]  # 默认使用道路中心
        
        # 路径平滑相关参数
        self.smoothing_strength = 0.2  # 平滑强度
        self.smoothing_window = 5  # 平滑窗口大小
        
        # 反向打方向盘参数
        self.counter_steering_enabled = True  # 启用反向打方向盘
        self.counter_steering_threshold = np.deg2rad(15)  # 触发反向打方向盘的转向角阈值
        
        # 调试信息
        self.debug_info = {}
        
        print(f"初始化RRT路径规划器, 安全距离设置为: {self.safety_distance}米")
    
    class Node:
        """RRT树节点"""
        def __init__(self, x, y):
            self.x = x  # x坐标
            self.y = y  # y坐标
            self.path_x = []  # 路径x坐标列表
            self.path_y = []  # 路径y坐标列表
            self.parent = None  # 父节点
            self.yaw = 0.0  # 航向角
            self.lane_id = None  # 所在车道ID
            self.cost = 0.0  # 路径代价
    
    def planning(self, start_x, start_y, goal_x, goal_y):
        """路径规划主函数"""
        print(f"开始RRT路径规划: 从({start_x:.1f}, {start_y:.1f})到({goal_x:.1f}, {goal_y:.1f})")
        start_time = time.time()
        
        # 创建起点和终点节点
        start_node = self.Node(start_x, start_y)
        goal_node = self.Node(goal_x, goal_y)
        
        # 确定起点和终点所在车道
        start_node.lane_id = self._get_lane_id(start_y)
        goal_node.lane_id = self._get_lane_id(goal_y)
        
        # 初始化节点列表
        self.node_list = [start_node]
        
        # 添加变道控制变量 - 用于强制车辆先沿车道行驶
        min_distance_before_lane_change = 8.0  # 变道前至少沿车道行驶的最小距离，从15.0降低到8.0
        current_distance_in_lane = 0.0  # 当前已在车道内行驶的距离
        initial_lane = start_node.lane_id  # 初始车道
        allow_lane_change = False  # 是否允许变道
        
        # 强制沿着车道中心线行驶的参数
        lane_center_bias = 0.3  # 进一步减弱车道中心偏好系数，从0.5降低到0.3
        
        # 添加新参数，完全禁用车道约束
        apply_lane_constraint = False  # 设置为False来完全禁用车道中心约束
        
        # 迭代生成路径
        for i in range(self.max_iter):
            # 根据目标采样率决定是否采样目标点
            if random.randint(0, 100) > self.goal_sample_rate:
                # 随机采样点
                if not apply_lane_constraint:
                    # 完全自由采样，不考虑车道约束
                    rnd_x = random.uniform(0, self.env.road_length)
                    rnd_y = random.uniform(0, self.env.road_width)
                    rnd = self.Node(rnd_x, rnd_y)
                    rnd.lane_id = self._get_lane_id(rnd_y)
                else:
                    # 使用车道约束的采样
                    current_node = self.node_list[-1]
                    current_lane = current_node.lane_id
                    
                    # 确定可能的车道
                    possible_lanes = [current_lane]  # 当前车道始终是可选的
                    
                    # 检查是否允许变道
                    # 通过回溯计算在当前车道的行驶距离
                    if not allow_lane_change:
                        temp_node = current_node
                        distance_in_lane = 0.0
                        while temp_node.parent and temp_node.lane_id == initial_lane:
                            distance_in_lane += self._calc_distance(temp_node, temp_node.parent)
                            temp_node = temp_node.parent
                        
                        # 如果在初始车道行驶足够距离，允许变道
                        if distance_in_lane >= min_distance_before_lane_change:
                            allow_lane_change = True
                    
                    # 仅当允许变道时才添加相邻车道作为选项
                    if allow_lane_change and self.num_lanes > 0:
                        # 添加相邻车道（只能变换到临近车道）
                        if current_lane > 0 and current_lane < self.num_lanes - 1:
                            possible_lanes.append(current_lane - 1)  # 左侧车道
                            possible_lanes.append(current_lane + 1)  # 右侧车道
                        elif current_lane == 0 and self.num_lanes > 1:
                            possible_lanes.append(current_lane + 1)  # 只能向右变道
                        elif current_lane == self.num_lanes - 1 and self.num_lanes > 1:
                            possible_lanes.append(current_lane - 1)  # 只能向左变道
                    
                    # 如果没有允许变道且需要变道到目标车道，增加沿当前车道向前行驶的概率
                    if not allow_lane_change and initial_lane != goal_node.lane_id:
                        forward_bias = 0.8  # 80%的概率向前行驶
                        if random.random() < forward_bias:
                            # 在当前车道前方区域采样
                            rnd_x = current_node.x + random.uniform(1.0, 5.0)  # 向前采样
                            lane_center = self.lane_centers[current_lane]
                            lane_deviation = random.uniform(-self.lane_width/8, self.lane_width/8)  # 严格限制在车道中心附近
                            rnd_y = lane_center + lane_deviation
                            rnd = self.Node(rnd_x, rnd_y)
                            rnd.lane_id = current_lane
                        else:
                            # 标准采样 - 强化车道中心偏好
                            random_lane = random.choice(possible_lanes)
                            lane_center = self.lane_centers[random_lane]
                            
                            # 使用车道中心偏好参数，更多地在车道中心线附近采样
                            if random.random() < lane_center_bias:  # 50%概率靠近车道中心
                                lane_deviation = random.uniform(-self.lane_width/6, self.lane_width/6)  # 扩大偏移范围
                            else:
                                lane_deviation = random.uniform(-self.lane_width/2.5, self.lane_width/2.5)  # 显著扩大偏移范围
                                
                            rnd_y = lane_center + lane_deviation
                            rnd_x = random.uniform(0, self.env.road_length)
                            rnd = self.Node(rnd_x, rnd_y)
                            rnd.lane_id = random_lane
                    else:
                        # 标准采样 - 强化车道中心偏好
                        random_lane = random.choice(possible_lanes)
                        lane_center = self.lane_centers[random_lane]
                        
                        # 使用车道中心偏好参数，更多地在车道中心线附近采样
                        if random.random() < lane_center_bias:  # 50%概率靠近车道中心
                            lane_deviation = random.uniform(-self.lane_width/6, self.lane_width/6)  # 扩大偏移范围
                        else:
                            lane_deviation = random.uniform(-self.lane_width/2.5, self.lane_width/2.5)  # 显著扩大偏移范围
                            
                        rnd_y = lane_center + lane_deviation
                        rnd_x = random.uniform(0, self.env.road_length)
                        rnd = self.Node(rnd_x, rnd_y)
                        rnd.lane_id = random_lane
            else:
                # 直接使用目标点
                rnd = self.Node(goal_x, goal_y)
                rnd.lane_id = goal_node.lane_id
            
            # 查找最近节点
            nearest_ind = self._get_nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]
            
            # 从最近节点扩展到随机点
            new_node = self._steer(nearest_node, rnd)
            
            # 检查新节点是否有效
            if new_node and self._is_collision_free(new_node):
                # 计算新节点的代价
                new_node.cost = nearest_node.cost + self._calc_distance(nearest_node, new_node)
                
                # 添加车道保持/变道代价
                if new_node.lane_id != nearest_node.lane_id:
                    # 变道增加额外代价
                    new_node.cost += self.lane_width * 0.5  # 增加变道代价，从0.5增加到1.0
                
                # 添加到节点列表
                self.node_list.append(new_node)
                
                # 检查是否可以连接到目标
                if self._can_connect_to_goal(self.node_list[-1], goal_node):
                    final_node = self._steer(self.node_list[-1], goal_node)
                    if final_node and self._is_collision_free(final_node):
                        # 找到路径
                        print(f"找到路径！迭代次数: {i+1}")
                        self.debug_info['iterations'] = i+1
                        self.debug_info['time'] = time.time() - start_time
                        return self._extract_path(final_node)
            
            # 输出进度
            if i % 1000 == 0:
                print(f"迭代 {i}/{self.max_iter}, 当前节点数: {len(self.node_list)}")
        
        # 达到最大迭代次数仍未找到路径
        print(f"达到最大迭代次数 {self.max_iter}，未找到路径！")
        return None
    
    def _get_lane_id(self, y):
        """根据y坐标确定所在车道ID"""
        # 确保车道中心列表不为空
        if not self.lane_centers:
            return 0
            
        for i, center in enumerate(self.lane_centers):
            if abs(y - center) <= self.lane_width / 2:
                return i
        # 如果不在任何车道内，返回最近的车道
        distances = [abs(y - center) for center in self.lane_centers]
        return distances.index(min(distances))
    
    def _get_nearest_node_index(self, node):
        """获取距离目标节点最近的节点索引"""
        distances = [self._calc_distance(node, n) for n in self.node_list]
        return distances.index(min(distances))
    
    def _calc_distance(self, from_node, to_node):
        """计算两节点间距离"""
        return np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
    
    def _steer(self, from_node, to_node):
        """从一个节点转向另一个节点，考虑最大转弯角度限制"""
        # 创建新节点
        new_node = self.Node(from_node.x, from_node.y)
        new_node.parent = from_node
        new_node.lane_id = from_node.lane_id
        
        # 计算方向角
        angle = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        
        # 如果父节点有定义航向角，检查转弯角度限制
        if hasattr(from_node, 'yaw') and from_node.yaw is not None:
            # 计算转弯角度（相对于当前航向）
            turn_angle = self._normalize_angle(angle - from_node.yaw)
            
            # 限制转弯角度
            if abs(turn_angle) > self.max_turn_angle:
                # 如果超过最大转弯角度，则按最大角度转向
                angle = from_node.yaw + np.sign(turn_angle) * self.max_turn_angle
        
        # 计算步长距离内的新位置
        dist = min(self.step_size, self._calc_distance(from_node, to_node))
        new_node.x = from_node.x + dist * np.cos(angle)
        new_node.y = from_node.y + dist * np.sin(angle)
        new_node.yaw = angle  # 更新航向角
        
        # 更新路径
        new_node.path_x = [from_node.x, new_node.x]
        new_node.path_y = [from_node.y, new_node.y]
        
        # 更新所在车道
        new_node.lane_id = self._get_lane_id(new_node.y)
        
        # 检查是否只变换到临近车道
        if abs(new_node.lane_id - from_node.lane_id) > 1:
            return None  # 不允许跨车道变道
        
        return new_node
    
    def _normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]区间"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def _is_collision_free(self, node):
        """检查节点是否无碰撞"""
        # 检查节点是否在道路范围内
        if node.x < 0 or node.x > self.env.road_length or \
           node.y < self.safety_distance or node.y > self.env.road_width - self.safety_distance:
            return False
        
        # 检查是否与障碍物碰撞，考虑安全距离
        if hasattr(self.env, 'obstacle_vehicles'):
            for vehicle in self.env.obstacle_vehicles:
                # 计算距离
                dist_x = abs(node.x - vehicle.x)
                dist_y = abs(node.y - vehicle.y)
                
                # 判断是否碰撞（矩形碰撞检测），增加安全距离
                if (dist_x < (vehicle.length/2 + self.safety_distance) and
                    dist_y < (vehicle.width/2 + self.safety_distance)):
                    return False
        
        # 检查路径是否无碰撞
        if node.parent:
            # 对路径进行更细致的碰撞检查
            for i in range(15):  # 从10段增加到15段，进行更细致的检查
                t = i / 15.0
                x = node.parent.x + t * (node.x - node.parent.x)
                y = node.parent.y + t * (node.y - node.parent.y)
                
                # 检查道路边界
                if x < 0 or x > self.env.road_length or \
                   y < self.safety_distance or y > self.env.road_width - self.safety_distance:
                    return False
                
                # 检查障碍物
                if hasattr(self.env, 'obstacle_vehicles'):
                    for vehicle in self.env.obstacle_vehicles:
                        dist_x = abs(x - vehicle.x)
                        dist_y = abs(y - vehicle.y)
                        
                        if (dist_x < (vehicle.length/2 + self.safety_distance) and
                            dist_y < (vehicle.width/2 + self.safety_distance)):
                            return False
        
        return True
    
    def _can_connect_to_goal(self, node, goal):
        """检查是否可以连接到目标点"""
        # 计算距离
        dist = self._calc_distance(node, goal)
        
        # 如果距离小于步长的3倍，尝试连接
        if dist < self.step_size * 3:
            # 检查转弯角度
            angle = np.arctan2(goal.y - node.y, goal.x - node.x)
            turn_angle = self._normalize_angle(angle - node.yaw)
            
            # 如果转弯角度在限制范围内或距离很近时允许更大角度
            if abs(turn_angle) <= self.max_turn_angle * 1.2 or dist < self.step_size:
                return True
        
        return False
    
    def _extract_path(self, final_node):
        """从最终节点提取路径"""
        path = []
        node = final_node
        
        # 从终点回溯到起点
        while node:
            path.append([node.x, node.y])
            node = node.parent
        
        # 反转路径，使其从起点到终点
        return path[::-1]
    
    def smooth_path(self, path, smoothness=0.30):
        """平滑路径，减少锯齿，增加丝滑度"""
        if not path or len(path) < 3:
            return path
        
        print(f"平滑路径，使用平滑系数: {smoothness}")
        
        # 转换为numpy数组便于操作
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        # 创建平滑路径
        smooth_x = x.copy()
        smooth_y = y.copy()
        
        # 迭代平滑
        change = 1.0
        n_iterations = 150  # 从100增加到150以获得更平滑的路径
        
        for _ in range(n_iterations):
            if change < 0.0003:  # 降低收敛阈值以获得更好的平滑效果
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
                
                # 确保平滑后的点仍然在道路内且无碰撞，严格检查安全距离
                if smooth_x[i] < 0 or smooth_x[i] > self.env.road_length or \
                   smooth_y[i] < self.safety_distance or smooth_y[i] > self.env.road_width - self.safety_distance:
                    # 如果不满足约束，恢复原来的位置
                    smooth_x[i], smooth_y[i] = old_x, old_y
                    continue
                
                # 检查与障碍物的安全距离
                collision = False
                if hasattr(self.env, 'obstacle_vehicles'):
                    for vehicle in self.env.obstacle_vehicles:
                        dist_x = abs(smooth_x[i] - vehicle.x)
                        dist_y = abs(smooth_y[i] - vehicle.y)
                        
                        if (dist_x < (vehicle.length/2 + self.safety_distance) and
                            dist_y < (vehicle.width/2 + self.safety_distance)):
                            collision = True
                            break
                
                if collision:
                    # 如果与障碍物的安全距离不满足，恢复原来的位置
                    smooth_x[i], smooth_y[i] = old_x, old_y
        
        # 应用强化平滑 - 二次平滑处理
        second_smooth_x = smooth_x.copy()
        second_smooth_y = smooth_y.copy()
        
        # 二次平滑处理，增加平滑迭代次数
        for _ in range(80):  # 从50增加到80，提高平滑效果
            for i in range(2, len(x) - 2):
                # 使用更大范围的点进行平滑，使用高斯权重
                second_smooth_x[i] = 0.05 * second_smooth_x[i-2] + 0.2 * second_smooth_x[i-1] + \
                                    0.5 * second_smooth_x[i] + \
                                    0.2 * second_smooth_x[i+1] + 0.05 * second_smooth_x[i+2]
                                    
                second_smooth_y[i] = 0.05 * second_smooth_y[i-2] + 0.2 * second_smooth_y[i-1] + \
                                    0.5 * second_smooth_y[i] + \
                                    0.2 * second_smooth_y[i+1] + 0.05 * second_smooth_y[i+2]
                
                # 确保平滑后的点仍然在道路内且无碰撞
                if second_smooth_x[i] < 0 or second_smooth_x[i] > self.env.road_length or \
                   second_smooth_y[i] < self.safety_distance or second_smooth_y[i] > self.env.road_width - self.safety_distance or \
                   self.env.is_collision(second_smooth_x[i], second_smooth_y[i], radius=self.safety_distance):
                    # 如果不满足约束，恢复原来的位置
                    second_smooth_x[i], second_smooth_y[i] = smooth_x[i], smooth_y[i]
        
        # 三次平滑处理 - 添加三次平滑以获得更好的顺滑效果
        third_smooth_x = second_smooth_x.copy()
        third_smooth_y = second_smooth_y.copy()
        
        # 应用三次平滑，使用更大的窗口
        for _ in range(40):
            for i in range(3, len(x) - 3):
                # 使用7点窗口平滑
                third_smooth_x[i] = 0.02 * third_smooth_x[i-3] + 0.05 * third_smooth_x[i-2] + \
                                  0.15 * third_smooth_x[i-1] + 0.56 * third_smooth_x[i] + \
                                  0.15 * third_smooth_x[i+1] + 0.05 * third_smooth_x[i+2] + \
                                  0.02 * third_smooth_x[i+3]
                                  
                third_smooth_y[i] = 0.02 * third_smooth_y[i-3] + 0.05 * third_smooth_y[i-2] + \
                                  0.15 * third_smooth_y[i-1] + 0.56 * third_smooth_y[i] + \
                                  0.15 * third_smooth_y[i+1] + 0.05 * third_smooth_y[i+2] + \
                                  0.02 * third_smooth_y[i+3]
                
                # 确保平滑后的点仍然安全
                if third_smooth_x[i] < 0 or third_smooth_x[i] > self.env.road_length or \
                   third_smooth_y[i] < self.safety_distance or third_smooth_y[i] > self.env.road_width - self.safety_distance or \
                   self.env.is_collision(third_smooth_x[i], third_smooth_y[i], radius=self.safety_distance):
                    # 如果不满足约束，恢复原来的位置
                    third_smooth_x[i], third_smooth_y[i] = second_smooth_x[i], second_smooth_y[i]
        
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
        
        # 应用车道中心约束 - 确保在直线段落沿车道中心行驶
        # 这里可以添加一个标志来决定是否应用车道中心约束
        apply_lane_constraint = False  # 设置为False，禁用车道中心约束
        if apply_lane_constraint:
            smooth_path = self._apply_lane_center_constraint(final_smooth_path)
        else:
            # 如果不应用车道约束，直接使用平滑后的路径
            smooth_path = final_smooth_path
        
        # 应用反向打方向盘动作 - 模拟人类驾驶习惯
        if hasattr(self, 'counter_steering_enabled') and self.counter_steering_enabled:
            smooth_path = self._apply_counter_steering(smooth_path)
        
        return smooth_path
    
    def _apply_lane_center_constraint(self, path):
        """应用车道中心约束，确保在直线段落沿车道中心行驶"""
        if len(path) < 3:
            return path
            
        result_path = [path[0]]  # 保留起点
        
        # 计算每段路径的方向变化，用于识别直线和转弯
        directions = []
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            angle = np.arctan2(dy, dx)
            directions.append(angle)
            
        # 检测每个点的曲率和所在车道
        for i in range(1, len(path) - 1):
            prev_point = path[i-1]
            curr_point = path[i]
            next_point = path[i+1]
            
            # 计算当前段和下一段的方向变化
            angle_diff = 0
            if i < len(directions):
                angle_diff = self._normalize_angle(directions[i] - directions[i-1])
            
            # 确定当前点所在车道
            lane_id = self._get_lane_id(curr_point[1])
            lane_center = self.lane_centers[lane_id]
            
            # 强化车道中心约束 - 大幅提高靠近车道中心线的约束力度
            # 直线段几乎完全靠拢车道中心，弯道也有较强的靠拢力度
            if abs(angle_diff) < np.deg2rad(3):  # 直线段（角度变化很小）
                # 计算向车道中心的调整 - 降低约束力度
                adjustment = (lane_center - curr_point[1]) * 0.5  # 从0.9降低到0.5，降低直线段的车道中心约束力度
                adjusted_y = curr_point[1] + adjustment
            else:  # 弯道
                # 降低向车道中心靠拢的程度
                adjustment = (lane_center - curr_point[1]) * 0.3  # 从0.6降低到0.3，降低弯道中的靠拢力度
                adjusted_y = curr_point[1] + adjustment
            
            # 确保调整后的点仍然安全
            if adjusted_y >= self.safety_distance and \
               adjusted_y <= self.env.road_width - self.safety_distance and \
               not self.env.is_collision(curr_point[0], adjusted_y, radius=self.safety_distance):
                result_path.append([curr_point[0], adjusted_y])
            else:
                # 如果调整后不安全，保持原位但仍尝试轻微向中心靠拢
                min_adjustment = (lane_center - curr_point[1]) * 0.1  # 最小调整
                min_adjusted_y = curr_point[1] + min_adjustment
                
                if min_adjusted_y >= self.safety_distance and \
                   min_adjusted_y <= self.env.road_width - self.safety_distance and \
                   not self.env.is_collision(curr_point[0], min_adjusted_y, radius=self.safety_distance):
                    result_path.append([curr_point[0], min_adjusted_y])
                else:
                    result_path.append(curr_point)
        
        result_path.append(path[-1])  # 保留终点
        
        # 再次应用平滑处理
        final_path = []
        final_path.append(result_path[0])  # 保留起点
        
        # 对中间点应用窗口平滑
        for i in range(1, len(result_path) - 1):
            # 三点窗口平滑，保持中心点权重较大以保留车道中心约束效果
            x_smoothed = 0.25 * result_path[i-1][0] + 0.5 * result_path[i][0] + 0.25 * result_path[i+1][0]
            y_smoothed = 0.25 * result_path[i-1][1] + 0.5 * result_path[i][1] + 0.25 * result_path[i+1][1]
            
            # 确保平滑后的点仍然安全
            if y_smoothed >= self.safety_distance and \
               y_smoothed <= self.env.road_width - self.safety_distance and \
               not self.env.is_collision(x_smoothed, y_smoothed, radius=self.safety_distance):
                final_path.append([x_smoothed, y_smoothed])
            else:
                final_path.append(result_path[i])
        
        final_path.append(result_path[-1])  # 保留终点
        return final_path
    
    def _apply_counter_steering(self, path):
        """应用反向打方向盘动作，模拟人类驾驶习惯"""
        if len(path) < 5:  # 至少需要5个点才能检测转向模式
            return path
            
        result_path = path.copy()
        
        # 计算每个点的航向角
        yaws = []
        for i in range(len(path) - 1):
            yaw = np.arctan2(path[i+1][1] - path[i][1], path[i+1][0] - path[i][0])
            yaws.append(yaw)
        yaws.append(yaws[-1])  # 最后一点使用前一点的航向角
        
        # 检测大转向后的反向打方向盘机会
        for i in range(2, len(path) - 2):
            # 计算前后航向角变化
            prev_yaw_change = self._normalize_angle(yaws[i] - yaws[i-2])
            
            # 如果刚完成一个大转向
            if abs(prev_yaw_change) > self.counter_steering_threshold:
                # 计算反向打方向盘的目标点
                counter_steer_strength = 0.15 * np.sign(prev_yaw_change)  # 反向打方向盘强度
                
                # 修改后续2-3个点，实现反向打方向盘效果
                for j in range(1, 3):
                    if i + j < len(result_path):
                        # 计算垂直于当前航向的方向
                        perp_x = -np.sin(yaws[i])
                        perp_y = np.cos(yaws[i])
                        
                        # 根据距离衰减反向打方向盘强度
                        decay = (3 - j) / 3.0
                        
                        # 应用反向打方向盘调整
                        adjust_x = perp_x * counter_steer_strength * decay
                        adjust_y = perp_y * counter_steer_strength * decay
                        
                        # 创建调整后的点
                        new_x = result_path[i+j][0] + adjust_x
                        new_y = result_path[i+j][1] + adjust_y
                        
                        # 确保新点在道路内且安全
                        if 0 <= new_x <= self.env.road_length and \
                           0 <= new_y <= self.env.road_width and \
                           not self.env.is_collision(new_x, new_y, radius=self.safety_distance):
                            result_path[i+j] = (new_x, new_y)
        
        return result_path
    
    def save_and_show_results(self, path, smooth_path, filename):
        """保存并显示路径规划结果"""
        plt.figure(figsize=(12, 6))
        
        # 绘制环境
        ax = plt.gca()
        self.env.plot_environment(ax)
        
        # 绘制安全距离边界
        self._plot_safety_boundaries(ax)
        
        # 绘制RRT树
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, '-g', alpha=0.3)
        
        # 绘制原始路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b--', linewidth=2, label='原始路径')
        
        # 绘制平滑路径
        if smooth_path:
            smooth_path_x = [p[0] for p in smooth_path]
            smooth_path_y = [p[1] for p in smooth_path]
            plt.plot(smooth_path_x, smooth_path_y, 'r-', linewidth=2, label='平滑路径')
        
        # 添加起点和终点标记
        if path:
            plt.plot(path[0][0], path[0][1], 'go', markersize=10, label='起点')
            plt.plot(path[-1][0], path[-1][1], 'ro', markersize=10, label='终点')
        
        # 添加图例和标题
        plt.legend()
        plt.title(f'RRT路径规划结果 (安全距离: {self.safety_distance:.1f}m)')
        plt.axis('equal')
        plt.grid(True)
        
        # 保存图像
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"路径规划结果已保存为 {filename}")
        
        # 关闭图形（不显示）
        plt.close()
    
    def _plot_safety_boundaries(self, ax):
        """绘制安全距离边界线"""
        # 1. 绘制道路边界的安全距离
        # 下边界安全线
        lower_safe_boundary = self.safety_distance
        ax.plot([0, self.env.road_length], [lower_safe_boundary, lower_safe_boundary], 
                'r--', linewidth=2.0, label='安全距离', zorder=5)
        
        # 上边界安全线
        upper_safe_boundary = self.env.road_width - self.safety_distance
        ax.plot([0, self.env.road_length], [upper_safe_boundary, upper_safe_boundary], 
                'r--', linewidth=2.0, zorder=5)
        
        # 2. 绘制障碍物安全距离
        if hasattr(self.env, 'obstacles') and self.env.obstacles:
            for obs in self.env.obstacles:
                try:
                    if isinstance(obs, dict) and 'position' in obs and 'dimensions' in obs:
                        # 障碍物为字典格式
                        if 'radius' in obs['dimensions']:
                            # 圆形障碍物 - 绘制矩形安全距离边界
                            x, y = obs['position'][0], obs['position'][1]
                            radius = obs['dimensions']['radius']
                            
                            # 创建矩形安全边界（距离障碍物安全距离）
                            safety_rect = Rectangle(
                                (x - radius - self.safety_distance, y - radius - self.safety_distance),
                                2 * radius + 2 * self.safety_distance,
                                2 * radius + 2 * self.safety_distance,
                                fill=False,
                                edgecolor='r',
                                linestyle='--',
                                linewidth=2.0,
                                alpha=0.8
                            )
                            ax.add_patch(safety_rect)
                            
                        elif 'width' in obs['dimensions'] and 'length' in obs['dimensions']:
                            # 矩形障碍物
                            x, y = obs['position'][0], obs['position'][1]
                            width, length = obs['dimensions']['width'], obs['dimensions']['length']
                            
                            # 绘制安全距离矩形
                            safety_rect = Rectangle(
                                (x - length/2 - self.safety_distance, 
                                 y - width/2 - self.safety_distance),
                                length + 2 * self.safety_distance,
                                width + 2 * self.safety_distance,
                                fill=False,
                                edgecolor='r',
                                linestyle='--',
                                linewidth=2.0,
                                alpha=0.8
                            )
                            ax.add_patch(safety_rect)
                except Exception as e:
                    print(f"绘制障碍物安全边界时出错: {e}")
        
        # 3. 绘制障碍车辆的安全距离矩形
        if hasattr(self.env, 'obstacle_vehicles'):
            for vehicle in self.env.obstacle_vehicles:
                # 创建安全距离矩形
                safety_rect = Rectangle(
                    (vehicle.x - vehicle.length/2 - self.safety_distance, 
                     vehicle.y - vehicle.width/2 - self.safety_distance),
                    vehicle.length + 2 * self.safety_distance,
                    vehicle.width + 2 * self.safety_distance,
                    fill=False,
                    edgecolor='r',
                    linestyle='--',
                    linewidth=2.0,
                    alpha=0.8
                )
                ax.add_patch(safety_rect)
            
        # 4. 车道中心线（可选）
        for lane_center in self.lane_centers:
            ax.axhline(y=lane_center, color='g', linestyle='--', alpha=0.5)
            
        # 在图例中添加安全距离说明
        ax.text(5, self.env.road_width - 0.5, 
                f"安全距离: {self.safety_distance:.1f}米", 
                fontsize=12, color='red', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))