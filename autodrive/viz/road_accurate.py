"""Detailed road environment visualization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from autodrive.environment import Environment, ObstacleVehicle
from autodrive.i18n import set_chinese_font


def visualize_road_environment_accurate(
    show=True, output_path="road_environment_updated.png"
):
    """准确可视化展示道路环境，展示修改后的障碍物位置。"""
    set_chinese_font()
    env = Environment()
    lane1_center = env.get_lane_center(1)
    lane2_center = env.get_lane_center(2)
    lane3_center = env.get_lane_center(3)

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("自动驾驶道路环境精确模型", fontsize=16)
    road = Rectangle(
        (0, 0),
        env.road_length,
        env.road_width,
        facecolor="darkgray",
        alpha=0.5,
        zorder=0,
    )
    ax.add_patch(road)
    env._draw_lane_backgrounds(ax)
    env._draw_lane_markings(ax)
    env._draw_road_markings(ax)
    ax.axhline(
        y=lane1_center,
        color="blue",
        linestyle="--",
        alpha=0.4,
        linewidth=1.5,
        label="车道中心线",
    )
    ax.axhline(
        y=lane2_center,
        color="blue",
        linestyle="--",
        alpha=0.4,
        linewidth=1.5,
    )
    ax.axhline(
        y=lane3_center,
        color="blue",
        linestyle="--",
        alpha=0.4,
        linewidth=1.5,
    )
    lane_bottoms = [
        center - env.lane_width / 2
        for center in (lane1_center, lane2_center, lane3_center)
    ]
    for index, lane_bottom in enumerate(lane_bottoms):
        ax.axhline(
            y=lane_bottom,
            color="red",
            linestyle="-.",
            alpha=0.3,
            linewidth=1,
            label="车道边界线" if index == 0 else None,
        )
    ax.axhline(
        y=env.road_width,
        color="red",
        linestyle="-.",
        alpha=0.3,
        linewidth=1,
    )
    for lane_id, center in enumerate(
        (lane1_center, lane2_center, lane3_center), start=1
    ):
        ax.text(
            -5,
            center,
            f"车道{lane_id}中心 (y={center:.1f}m)",
            fontsize=10,
            bbox=dict(
                facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"
            ),
        )

    env.start_vehicle.draw(ax)
    env.end_vehicle.draw(ax)
    ax.annotate(
        "起点",
        (env.start_point[0], env.start_point[1]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        fontsize=12,
        bbox=dict(
            facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"
        ),
    )
    ax.annotate(
        "终点",
        (env.end_point[0], env.end_point[1]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        fontsize=12,
        bbox=dict(
            facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"
        ),
    )
    ax.plot(
        [env.start_point[0] - 5, env.start_point[0]],
        [lane1_center, lane1_center],
        "g-",
        linewidth=1.5,
        alpha=0.6,
    )
    ax.plot(
        [env.end_point[0] + 5, env.end_point[0]],
        [lane3_center, lane3_center],
        "g-",
        linewidth=1.5,
        alpha=0.6,
    )

    colors = ["green", "blue", "blue", "blue"]
    descriptions = [
        f"障碍物1: 位于车道1中心\n位置: (25.0, {env.obstacle_vehicles[0].y:.1f})",
        f"障碍物2: 位于车道2中心\n位置: (48.27, {env.obstacle_vehicles[1].y:.1f})",
        f"障碍物3: 位于车道3中心\n位置: (48.27, {env.obstacle_vehicles[2].y:.1f})",
        f"障碍物4: 位于车道3中心\n位置: (60.0, {env.obstacle_vehicles[3].y:.1f})",
    ]
    for index, vehicle in enumerate(env.obstacle_vehicles):
        assert isinstance(vehicle, ObstacleVehicle)
        if index == 0:
            ax.plot(
                [vehicle.x - 1, vehicle.x + 1],
                [lane1_center, lane1_center],
                "g-",
                linewidth=2,
                alpha=0.7,
            )
            ax.text(
                vehicle.x + 6,
                vehicle.y,
                "位于车道1中心",
                fontsize=10,
                color="green",
                ha="left",
                va="center",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    boxstyle="round,pad=0.3",
                ),
            )
        rect = Rectangle(
            (
                vehicle.x - vehicle.length / 2,
                vehicle.y - vehicle.width / 2,
            ),
            vehicle.length,
            vehicle.width,
            angle=np.rad2deg(vehicle.angle),
            facecolor=colors[index],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.annotate(
            descriptions[index],
            (vehicle.x, vehicle.y),
            xytext=(0, 35),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            bbox=dict(
                facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"
            ),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

    highlight = Circle(
        (env.obstacle_vehicles[0].x, env.obstacle_vehicles[0].y),
        radius=1.0,
        fill=False,
        edgecolor="green",
        linestyle="-",
        linewidth=2,
        alpha=0.7,
        zorder=20,
    )
    ax.add_patch(highlight)
    ax.annotate(
        f"道路长度: {env.road_length:.1f}m",
        (env.road_length / 2, -1),
        ha="center",
        fontsize=12,
        bbox=dict(
            facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"
        ),
    )
    ax.annotate(
        f"道路宽度: {env.road_width:.1f}m",
        (-1, env.road_width / 2),
        ha="right",
        va="center",
        rotation=90,
        fontsize=12,
        bbox=dict(
            facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"
        ),
    )
    ax.annotate(
        f"车道宽度: {env.lane_width:.1f}m",
        (-3, lane1_center),
        ha="center",
        va="center",
        rotation=90,
        fontsize=12,
        bbox=dict(
            facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"
        ),
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(-10, env.road_length + 10)
    ax.set_ylim(-3, env.road_width + 3)
    ax.set_xlabel("X轴坐标 (m)")
    ax.set_ylabel("Y轴坐标 (m)")
    ax.grid(False)
    ax.axis("equal")
    ax.set_title("自动驾驶仿真环境 - 更新后的障碍物位置", fontsize=16)
    info_text = (
        "说明：\n"
        "1. 第一辆障碍车辆 (绿色) 现已移动到车道1中心位置\n"
        "2. 所有障碍车辆都位于各自车道的中心线上\n"
        "3. 路径规划算法考虑了所有障碍物的实际位置\n"
        "4. 蓝色虚线表示车道中心线，红色点线表示车道边界线"
    )
    fig.text(
        0.5,
        0.01,
        info_text,
        ha="center",
        fontsize=12,
        bbox=dict(
            facecolor="lightyellow",
            alpha=0.9,
            boxstyle="round,pad=0.5",
        ),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"更新后的道路环境可视化已保存为'{output_path}'")
    return output_path
