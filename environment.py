import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from font_support import set_chinese_font, labels
import random

class Vehicle:
    """车辆类（包括静态障碍物）"""
    def __init__(self, x, y, length=4.0, width=2.0, yaw=0.0, color='blue', is_static=True):
        self.x = x  # 中心x坐标
        self.y = y  # 中心y坐标
        self.length = length  # 车长
        self.width = width    # 车宽
        self.yaw = yaw        # 航向角
        self.color = color    # 颜色
        self.is_static = is_static  # 是否为静态障碍物
        
    def draw(self, ax):
        """绘制车辆"""
        # 计算车辆矩形的左下角坐标
        # 对于yaw=0的情况（沿x轴方向），左下角坐标应为(x-length/2, y-width/2)
        corner_x = self.x - self.length/2
        corner_y = self.y - self.width/2
        
        # 创建矩形表示车辆
        car = Rectangle(
            (corner_x, corner_y),
            self.length, self.width, angle=np.rad2deg(self.yaw),
            facecolor=self.color, alpha=0.8, edgecolor='black', linewidth=1)
        ax.add_patch(car)
        
        # 标记车辆中心点（用于调试）
        ax.plot(self.x, self.y, 'ko', markersize=3)
        
        # 如果不是静态车辆，画出车辆前进方向
        if not self.is_static:
            ax.arrow(self.x, self.y, 
                    0.5 * np.cos(self.yaw), 
                    0.5 * np.sin(self.yaw),
                    head_width=0.3, head_length=0.3, 
                    fc='red', ec='red')
            
        return car
    
    def update_position(self, x, y, yaw=None):
        """更新车辆位置"""
        self.x = x
        self.y = y
        if yaw is not None:
            self.yaw = yaw

class Environment:
    def __init__(self):
        # 设置中文字体
        set_chinese_font()
        
        # 道路参数 - 调整长度以更接近图片比例
        self.road_length = 85.0  # 道路长度 (m)，从100米缩减到85米，保持比例
        
        # 车辆参数 - 与vehicle_model.py保持一致
        self.vehicle_length = 4.0  # 车长 (m)
        self.vehicle_width = 1.8   # 车宽 (m)，与vehicle_model.py一致
        
        # 车道宽度设置为车宽的2.2倍
        self.lane_width = self.vehicle_width * 2.2  # 车道宽度 (m)，从2.5倍减少到2.2倍
        self.num_lanes = 3        # 车道数量
        self.road_width = self.lane_width * self.num_lanes  # 道路总宽度
        
        # 获取车道中心位置 - 交换车道编号：最上面的车道为道路3，最下面的车道为道路1
        # 修正车道编号和中心计算
        lane1_center = self.get_lane_center(1)  # 第一车道中心（最下方车道）
        lane2_center = self.get_lane_center(2)  # 第二车道中心（中间车道）
        lane3_center = self.get_lane_center(3)  # 第三车道中心（最上方车道）
        
        # 定义起点和终点位置 - 根据用户要求：起点在上方车道，终点在下方车道
        self.start_point = np.array([9.71, lane3_center])   # 起点在第三车道（最上方）左侧
        self.end_point = np.array([80.0, lane1_center])    # 终点在第一车道（最下方）右侧，X坐标向右移至80m
        
        # 使用固定的障碍车辆位置 - 更新布局以符合新的起点终点设置
        self.obstacle_vehicles = [
            Vehicle(25.0, lane3_center, self.vehicle_length, self.vehicle_width, 0.0, 'blue'),  # 第三车道（上方）- 起点车道障碍物
            Vehicle(48.27, lane2_center, self.vehicle_length, self.vehicle_width, 0.0, 'blue'),  # 第二车道（中间）
            Vehicle(48.27, lane1_center, self.vehicle_length, self.vehicle_width, 0.0, 'blue'),  # 第一车道（下方）- 终点车道障碍物
            Vehicle(60.0, lane1_center, self.vehicle_length, self.vehicle_width, 0.0, 'blue')   # 第一车道（下方）靠近终点
        ]
        
        # 为起点和终点创建车辆表示
        self.start_vehicle = Vehicle(self.start_point[0], self.start_point[1], 
                                     self.vehicle_length, self.vehicle_width, 0.0, 'green')
        self.end_vehicle = Vehicle(self.end_point[0], self.end_point[1], 
                                  self.vehicle_length, self.vehicle_width, 0.0, 'green')
        
        # 车道颜色
        self.lane_colors = ['#f0f0f0', '#e8e8e8', '#f0f0f0']  # 为每条车道设置不同的底色
    
    def _get_lane_id(self, y):
        """根据y坐标获取车道ID（0-based索引）"""
        for i in range(self.num_lanes):
            lane_center = self.get_lane_center(i+1)
            if abs(y - lane_center) <= self.lane_width / 2:
                return i
        # 如果不在任何车道内，返回最近的车道
        distances = [abs(y - self.get_lane_center(i+1)) for i in range(self.num_lanes)]
        return distances.index(min(distances))
    
    def update_obstacles(self, dt):
        """更新障碍物位置（保留空方法以兼容原有代码）"""
        pass
    
    def is_collision(self, x, y, radius=0.0, yaw=0.0, length=None, width=None):
        """检测给定位置是否与障碍物碰撞
        
        参数:
            x (float): 车辆中心x坐标
            y (float): 车辆中心y坐标
            radius (float): 用于简单碰撞检测的半径
            yaw (float): 车辆航向角
            length (float): 车辆长度，默认使用初始化时设置的值
            width (float): 车辆宽度，默认使用初始化时设置的值
            
        返回:
            bool: 如果碰撞返回True，否则返回False
        """
        # 容错距离，只有当实际距离小于此值时才认为碰撞
        tolerance = 0.05  # 5厘米容错距离，恢复为原始值
        
        # 使用默认车辆尺寸（如果未提供）
        if length is None:
            length = self.vehicle_length
        if width is None:
            width = self.vehicle_width
        
        # 如果提供了半径且大于0，使用简化的圆形碰撞检测（用于路径规划阶段）
        if radius > 0.0:
            # 检查道路边界
            if y - radius < 0 or y + radius > self.road_width:
                return True
                
            # 检查障碍物
            if hasattr(self, 'obstacle_vehicles'):
                for vehicle in self.obstacle_vehicles:
                    dist_x = abs(x - vehicle.x)
                    dist_y = abs(y - vehicle.y)
                    if (dist_x < (vehicle.length/2 + radius) and
                        dist_y < (vehicle.width/2 + radius)):
                        return True
            return False
            
        # 精确的矩形边界碰撞检测
        # 1. 计算运动车辆的四个角点坐标
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        half_length = length / 2
        half_width = width / 2
        
        # 车辆四个角点相对于中心点的坐标偏移
        corners_offsets = [
            [-half_length, -half_width],  # 左后
            [half_length, -half_width],   # 右后
            [half_length, half_width],    # 右前
            [-half_length, half_width]    # 左前
        ]
        
        # 计算旋转后的实际角点坐标
        corners = []
        for dx, dy in corners_offsets:
            rotated_dx = dx * cos_yaw - dy * sin_yaw
            rotated_dy = dx * sin_yaw + dy * cos_yaw
            corners.append([x + rotated_dx, y + rotated_dy])
        
        # 2. 检查是否与道路边界碰撞（边对边检测）
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            
            # 检查是否与下边界相交（考虑容错距离）
            if (y1 < -tolerance and y2 < -tolerance) or (min(y1, y2) < -tolerance and max(y1, y2) >= -tolerance):
                # 打印碰撞信息
                print(f"车辆与道路下边界碰撞! 坐标: ({x:.2f}, {y:.2f})")
                return True
                
            # 检查是否与上边界相交（考虑容错距离）
            if (y1 > self.road_width + tolerance and y2 > self.road_width + tolerance) or (min(y1, y2) <= self.road_width + tolerance and max(y1, y2) > self.road_width + tolerance):
                # 打印碰撞信息
                print(f"车辆与道路上边界碰撞! 坐标: ({x:.2f}, {y:.2f})")
                return True
        
        # 3. 检查是否与障碍物车辆碰撞（矩形之间的碰撞检测）
        if hasattr(self, 'obstacle_vehicles'):
            for obstacle in self.obstacle_vehicles:
                # 计算障碍物车辆的四个角点
                obs_half_length = obstacle.length / 2
                obs_half_width = obstacle.width / 2
                
                obstacle_corners = [
                    [obstacle.x - obs_half_length, obstacle.y - obs_half_width],
                    [obstacle.x + obs_half_length, obstacle.y - obs_half_width],
                    [obstacle.x + obs_half_length, obstacle.y + obs_half_width],
                    [obstacle.x - obs_half_length, obstacle.y + obs_half_width]
                ]
                
                # 检查两个矩形是否相交，考虑容错距离
                if self._check_rectangles_intersect(corners, obstacle_corners, tolerance):
                    print(f"车辆与障碍物碰撞! 坐标: ({x:.2f}, {y:.2f})")
                    return True
        
        # 没有碰撞
        return False
    
    def _check_rectangles_intersect(self, rect1_corners, rect2_corners, tolerance=0.0):
        """检查两个矩形是否相交
        
        参数:
            rect1_corners: 第一个矩形的四个角点坐标
            rect2_corners: 第二个矩形的四个角点坐标
            tolerance: 容错距离，只有当两个矩形距离小于此值时才认为相交
            
        返回:
            bool: 如果相交返回True，否则返回False
        """
        # 分离轴定理(SAT)检测两个凸多边形是否相交
        
        # 检查矩形1的投影是否与矩形2分离
        for i in range(4):
            p1 = rect1_corners[i]
            p2 = rect1_corners[(i + 1) % 4]
            
            # 计算法向量
            normal = [-(p2[1] - p1[1]), p2[0] - p1[0]]
            norm = np.sqrt(normal[0]**2 + normal[1]**2)
            if norm < 1e-10:  # 避免除以零
                continue
            normal = [normal[0]/norm, normal[1]/norm]
            
            # 计算rect1投影的最小和最大值
            min_r1 = float('inf')
            max_r1 = float('-inf')
            for j in range(4):
                dot_product = normal[0] * rect1_corners[j][0] + normal[1] * rect1_corners[j][1]
                min_r1 = min(min_r1, dot_product)
                max_r1 = max(max_r1, dot_product)
            
            # 计算rect2投影的最小和最大值
            min_r2 = float('inf')
            max_r2 = float('-inf')
            for j in range(4):
                dot_product = normal[0] * rect2_corners[j][0] + normal[1] * rect2_corners[j][1]
                min_r2 = min(min_r2, dot_product)
                max_r2 = max(max_r2, dot_product)
            
            # 检查投影是否分离，考虑容错距离
            if max_r1 + tolerance < min_r2 or max_r2 + tolerance < min_r1:
                return False  # 找到一个分离轴，矩形不相交
        
        # 检查矩形2的投影是否与矩形1分离
        for i in range(4):
            p1 = rect2_corners[i]
            p2 = rect2_corners[(i + 1) % 4]
            
            # 计算法向量
            normal = [-(p2[1] - p1[1]), p2[0] - p1[0]]
            norm = np.sqrt(normal[0]**2 + normal[1]**2)
            if norm < 1e-10:  # 避免除以零
                continue
            normal = [normal[0]/norm, normal[1]/norm]
            
            # 计算rect1投影的最小和最大值
            min_r1 = float('inf')
            max_r1 = float('-inf')
            for j in range(4):
                dot_product = normal[0] * rect1_corners[j][0] + normal[1] * rect1_corners[j][1]
                min_r1 = min(min_r1, dot_product)
                max_r1 = max(max_r1, dot_product)
            
            # 计算rect2投影的最小和最大值
            min_r2 = float('inf')
            max_r2 = float('-inf')
            for j in range(4):
                dot_product = normal[0] * rect2_corners[j][0] + normal[1] * rect2_corners[j][1]
                min_r2 = min(min_r2, dot_product)
                max_r2 = max(max_r2, dot_product)
            
            # 检查投影是否分离，考虑容错距离
            if max_r1 + tolerance < min_r2 or max_r2 + tolerance < min_r1:
                return False  # 找到一个分离轴，矩形不相交
        
        # 没有找到分离轴，矩形相交
        return True
    
    def is_within_lane(self, x, y, lane_id):
        """检查点(x,y)是否在指定车道内，不再使用安全边距"""
        lane_center = self.get_lane_center(lane_id)
        lane_boundary = self.lane_width / 2
        
        # 车道中心 ±（车道宽度/2）- 不再使用安全边距
        return abs(y - lane_center) <= lane_boundary
    
    def is_out_of_bounds(self, x, y):
        """检查点(x,y)是否超出道路边界"""
        # 检查是否在道路范围内，精确到边界
        if not (0 <= x <= self.road_length and 0 <= y <= self.road_width):
            return True
            
        return False
    
    def get_lane_center(self, lane_id):
        """获取车道中心线y坐标"""
        if 1 <= lane_id <= self.num_lanes:
            return (lane_id - 0.5) * self.lane_width
        return None
    
    def _draw_lane_markings(self, ax, interval=5.0, width=0.3):
        """绘制车道标线 - 增强可视性"""
        # 车道分隔线采用虚线，更加符合实际道路
        for i in range(1, self.num_lanes):
            y = i * self.lane_width
            # 绘制白色虚线，更宽更明显
            for x in np.arange(0, self.road_length, interval):
                line_length = min(3.0, self.road_length - x)
                if x + line_length <= self.road_length:
                    # 创建矩形表示车道线
                    lane_line = Rectangle((x, y - width/2), line_length, width, 
                                         color='white', alpha=1.0, zorder=2)
                    ax.add_patch(lane_line)
        
        # 边缘线采用更宽的实线，显著提高可见性
        edge_width = 0.4  # 减小边缘线宽度，从0.6减小到0.4
        edge_color = 'yellow'  # 使用黄色边缘线，增加对比度
        
        # 道路下边缘
        bottom_edge = Rectangle((0, 0), self.road_length, edge_width, 
                               color=edge_color, alpha=1.0, zorder=3)
        ax.add_patch(bottom_edge)
        
        # 道路上边缘
        top_edge = Rectangle((0, self.road_width - edge_width), self.road_length, edge_width, 
                             color=edge_color, alpha=1.0, zorder=3)
        ax.add_patch(top_edge)
    
    def _draw_lane_backgrounds(self, ax):
        """绘制车道背景色"""
        for i in range(self.num_lanes):
            y_bottom = i * self.lane_width
            lane_bg = Rectangle((0, y_bottom), self.road_length, self.lane_width, 
                              facecolor=self.lane_colors[i], alpha=0.6, zorder=1)
            ax.add_patch(lane_bg)
    
    def _draw_lane_labels(self, ax):
        """绘制车道标签"""
        for i in range(self.num_lanes):
            y = (i + 0.5) * self.lane_width
            # 根据新的车道编号，从上到下分别是车道3、车道2、车道1
            lane_id = self.num_lanes - i
            lane_text = f"车道 {lane_id}" if labels is None else f"Lane {lane_id}"
            
            # 将标签移到车道最右边，终点标志的右边
            x_pos = self.road_length + 5.0  # 在道路结束处右侧标记
            ax.text(x_pos, y, lane_text, ha='center', va='center', fontsize=10,
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.3'),
                   zorder=3)
                
    def _draw_road_markings(self, ax):
        """绘制道路标记，如导向箭头等"""
        # 在每条车道上绘制导向箭头
        for i in range(self.num_lanes):
            lane_center = self.get_lane_center(i+1)
            
            # 在道路中间位置绘制箭头 - 按比例调整
            arrow_positions = [22.77, 53.13]  # 按比例从100米长度调整到85米长度
            for x_pos in arrow_positions:
                # 绘制箭头
                arrow_length = 4.0
                arrow_width = 1.0
                
                # 创建箭头形状的多边形
                vertices = [
                    (x_pos, lane_center),  # 箭头尖端
                    (x_pos - arrow_length, lane_center - arrow_width/2),  # 左下
                    (x_pos - arrow_length * 0.7, lane_center - arrow_width/2),  # 内缩左下
                    (x_pos - arrow_length * 0.7, lane_center - arrow_width),  # 尾部左下
                    (x_pos - arrow_length * 0.3, lane_center - arrow_width),  # 尾部右下
                    (x_pos - arrow_length * 0.3, lane_center - arrow_width/2),  # 内缩右下
                    (x_pos - arrow_length, lane_center - arrow_width/2),  # 右下
                    (x_pos - arrow_length, lane_center + arrow_width/2),  # 右上
                    (x_pos - arrow_length * 0.3, lane_center + arrow_width/2),  # 内缩右上
                    (x_pos - arrow_length * 0.3, lane_center + arrow_width),  # 尾部右上
                    (x_pos - arrow_length * 0.7, lane_center + arrow_width),  # 尾部左上
                    (x_pos - arrow_length * 0.7, lane_center + arrow_width/2),  # 内缩左上
                    (x_pos - arrow_length, lane_center + arrow_width/2),  # 左上
                ]
                
                arrow = Polygon(vertices, closed=True, facecolor='white', edgecolor='white', alpha=0.8, zorder=2)
                ax.add_patch(arrow)
    
    def plot_environment(self, ax=None):
        """绘制环境，包括道路、车道、车辆等"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制道路背景 - 深灰色
        road = Rectangle((0, 0), self.road_length, self.road_width, 
                         facecolor='darkgray', alpha=0.5, zorder=0)
        ax.add_patch(road)
        
        # 绘制车道背景色
        self._draw_lane_backgrounds(ax)
        
        # 绘制车道标线
        self._draw_lane_markings(ax)
        
        # 绘制道路标记（箭头等）
        self._draw_road_markings(ax)
        
        # 绘制车道标签
        self._draw_lane_labels(ax)
        
        # 绘制起点和终点车辆
        self.start_vehicle.draw(ax)
        self.end_vehicle.draw(ax)
        
        # 添加起点和终点标签
        start_label = '起点' if labels is None else labels.get('起点', 'Start')
        end_label = '终点' if labels is None else labels.get('终点', 'End')
        
        ax.annotate(start_label, (self.start_point[0], self.start_point[1]), 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, boxstyle='round,pad=0.3'),
                    zorder=4)
        ax.annotate(end_label, (self.end_point[0], self.end_point[1]), 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, boxstyle='round,pad=0.3'),
                   zorder=4)
        
        # 绘制障碍物车辆
        for vehicle in self.obstacle_vehicles:
            vehicle.draw(ax)
            
        # 辅助调试：绘制更明显的车道中心线
        for i in range(1, self.num_lanes + 1):
            center_y = self.get_lane_center(i)
            ax.axhline(y=center_y, color='yellow', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # 设置图形范围和外观
        ax.set_xlim(-5, self.road_length + 5)
        ax.set_ylim(-2, self.road_width + 2)
        ax.set_aspect('equal')
        ax.grid(False)  # 移除网格线，避免干扰道路标线
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        title = '道路环境' if labels is None else labels.get('道路环境', 'Road Environment')
        ax.set_title(title)
        
        return ax