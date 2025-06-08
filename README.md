# 自动驾驶仿真系统

一个集成多种路径规划算法和控制器的自动驾驶仿真平台，支持obstacle避障、路径规划和车辆控制的完整仿真。

## 功能特性

- **多种路径规划算法**：支持RRT、A*、RRT*算法
- **多种控制器**：集成Pure Pursuit、MPC、Stanley控制器  
- **完整仿真环境**：三车道道路场景，包含静态障碍车辆
- **可视化功能**：实时路径显示、动画仿真、结果分析
- **中文界面**：完整的中文用户界面和文档

## 系统架构

```
自动驾驶仿真系统
├── 环境模块 (environment.py)
├── 车辆模型 (vehicle_model.py)  
├── 路径规划
│   ├── RRT算法 (rrt_path_planning.py)
│   ├── A*算法 (astar_path_planning.py)
│   └── RRT*算法 (rrt_star_path_planning.py)
├── 控制器
│   ├── Pure Pursuit (pure_pursuit_controller.py)
│   ├── MPC控制器 (mpc_controller.py)
│   └── Stanley控制器 (stanley_controller.py)
└── 主程序 (main.py)
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包列表

- numpy: 数值计算
- matplotlib: 绘图和可视化
- scipy: 科学计算和优化

## 使用方法

### 基本运行

```bash
python main.py
```

### 设置随机种子

```bash
python main.py --seed 42
```

### 交互式选择

运行程序后，系统会提示选择：

1. **路径规划算法**：
   - RRT (快速随机树)
   - A* (A星算法)  
   - RRT* (优化随机树)

2. **控制器类型**：
   - Pure Pursuit (纯跟踪控制器)
   - MPC (模型预测控制器)
   - Stanley (Stanley路径跟踪控制器)

## 算法说明

### 路径规划算法

- **RRT算法**：通过随机采样快速构建搜索树，适合复杂环境的路径规划
- **A*算法**：基于网格的启发式搜索，保证最优路径
- **RRT*算法**：RRT的改进版本，通过重连操作实现渐近最优

### 控制器

- **Pure Pursuit**：几何路径跟踪方法，通过追踪前瞻点实现控制
- **MPC**：模型预测控制，通过优化未来控制序列实现精确跟踪
- **Stanley**：结合横向误差和朝向误差的高精度控制器

## 仿真环境

- **道路规格**：85米长，三车道，每车道3.96米宽
- **车辆尺寸**：4.0米长，1.8米宽  
- **起点位置**：(9.71, 9.9) 第三车道
- **终点位置**：(80.0, 2.0) 第一车道
- **静态障碍物**：4辆静止车辆分布在不同位置

## 输出结果

仿真完成后会生成：

- **路径规划图**：显示算法生成的路径
- **仿真结果图**：包含路径跟踪、速度曲线等信息  
- **动画文件**：vehicle_animation.gif，展示车辆运动过程

## 项目结构

```
├── main.py                 # 主程序入口
├── environment.py          # 仿真环境
├── vehicle_model.py        # 车辆动力学模型
├── rrt_path_planning.py    # RRT路径规划
├── astar_path_planning.py  # A*路径规划  
├── rrt_star_path_planning.py # RRT*路径规划
├── pure_pursuit_controller.py # Pure Pursuit控制器
├── mpc_controller.py       # MPC控制器
├── stanley_controller.py   # Stanley控制器
├── font_support.py         # 字体支持
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明
```

## 参数配置

### 路径规划参数

- `step_size`: 搜索步长
- `max_iter`: 最大迭代次数  
- `safety_distance`: 安全距离
- `goal_sample_rate`: 目标采样率

### 控制参数

- `target_speed`: 目标速度 (4.0 m/s)
- `max_steer`: 最大转向角
- `dt`: 时间步长 (0.1s)

## 注意事项

1. 首次运行可能需要安装字体支持
2. 仿真过程中请勿关闭matplotlib窗口
3. 结果图片会保存在当前目录下
4. 建议在Python 3.7+环境下运行

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。