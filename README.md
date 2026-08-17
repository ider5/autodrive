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
├── main.py                     # 精简命令行入口
├── 根目录兼容层                 # 保留旧模块导入路径
├── autodrive/
│   ├── config.py               # 参数配置
│   ├── environment.py          # 仿真环境
│   ├── vehicle.py              # 车辆模型
│   ├── i18n.py                 # 界面文字与字体支持
│   ├── planning/               # 路径规划
│   ├── control/                # 路径跟踪控制
│   ├── simulation/             # 仿真会话、循环与输出
│   └── viz/                    # 可视化
└── tests/                      # 自动化测试
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包列表

- numpy: 数值计算
- matplotlib: 绘图和可视化
- scipy: 科学计算和优化

开发和测试依赖可通过以下命令安装：

```bash
pip install -r requirements-dev.txt
```

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
   - 1：RRT (快速随机树)
   - 2：A* (A星算法)
   - 3：RRT* (优化随机树)

2. **控制器类型**：
   - 1：Pure Pursuit (纯跟踪控制器)
   - 2：MPC（菜单沿用原名称，当前实现是已有的几何路径跟踪器）
   - 3：Stanley (Stanley路径跟踪控制器，推荐)

## 算法说明

### 路径规划算法

- **RRT算法**：通过随机采样快速构建搜索树，适合复杂环境的路径规划
- **A*算法**：基于网格的启发式搜索，保证最优路径
- **RRT*算法**：RRT的改进版本，通过重连操作实现渐近最优

### 控制器

- **Pure Pursuit**：几何路径跟踪方法，通过追踪前瞻点实现控制
- **MPC**：菜单沿用原名称，当前实现是已有的几何路径跟踪器
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
├── main.py                         # 精简命令行入口
├── environment.py                  # 兼容层
├── vehicle_model.py                # 兼容层
├── font_support.py                 # 兼容层
├── rrt_path_planning.py            # RRT兼容层
├── astar_path_planning.py          # A*兼容层
├── rrt_star_path_planning.py       # RRT*兼容层
├── pure_pursuit_controller.py      # Pure Pursuit兼容层
├── mpc_controller.py               # MPC兼容层
├── stanley_controller.py           # Stanley兼容层
├── visualize_vehicle_model.py      # 车辆模型可视化入口
├── visualize_road_environment.py   # 道路环境可视化入口
├── visualize_road_environment_accurate.py # 精确道路环境可视化入口
├── autodrive/
│   ├── config.py
│   ├── environment.py
│   ├── vehicle.py
│   ├── i18n.py
│   ├── planning/
│   ├── control/
│   ├── simulation/
│   └── viz/
├── tests/                           # pytest测试
├── requirements.txt                # 运行依赖
├── requirements-dev.txt            # 开发与pytest依赖
├── pyproject.toml                   # 项目与pytest配置
├── LICENSE                          # MIT许可证
└── README.md                        # 项目说明
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
4. 需要Python 3.9+环境

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。