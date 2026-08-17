"""Road environment overview visualization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from autodrive.environment import Environment, ObstacleVehicle
from autodrive.i18n import set_chinese_font


def visualize_road_environment(
    show=True, output_path="road_environment_visualization.png"
):
    """可视化展示道路环境。"""
    set_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("自动驾驶模拟道路环境", fontsize=16)
    axes = axes.flatten()
    env = Environment()

    ax1 = axes[0]
    ax1.set_title("完整道路环境")
    env.plot_environment(ax1)

    ax2 = axes[1]
    ax2.set_title("车道结构和标记")
    road = Rectangle(
        (0, 0),
        env.road_length,
        env.road_width,
        facecolor="darkgray",
        alpha=0.5,
        zorder=0,
    )
    ax2.add_patch(road)
    env._draw_lane_backgrounds(ax2)
    env._draw_lane_markings(ax2)
    env._draw_road_markings(ax2)
    env._draw_lane_labels(ax2)
    for lane_id in range(1, env.num_lanes + 1):
        center_y = env.get_lane_center(lane_id)
        ax2.axhline(
            y=center_y,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )
        ax2.text(
            -3,
            center_y,
            f"车道{lane_id}中心线",
            fontsize=9,
            ha="right",
            va="center",
            color="black",
            bbox=dict(
                facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"
            ),
        )
        ax2.text(
            env.road_length + 3,
            center_y,
            f"y={center_y:.1f}m",
            fontsize=9,
            ha="left",
            va="center",
            color="black",
            bbox=dict(
                facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"
            ),
        )
    for index in range(env.num_lanes):
        y_bottom = index * env.lane_width
        y_top = (index + 1) * env.lane_width
        mid_y = (y_bottom + y_top) / 2
        ax2.annotate(
            "",
            xy=(-2, y_bottom),
            xytext=(-2, y_top),
            arrowprops=dict(arrowstyle="<->", color="black"),
        )
        ax2.text(
            -4,
            mid_y,
            f"{env.lane_width:.1f}m",
            fontsize=9,
            ha="center",
            va="center",
        )
    ax2.set_xlim(-5, env.road_length + 5)
    ax2.set_ylim(-2, env.road_width + 2)
    ax2.set_aspect("equal")
    ax2.grid(False)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    ax3 = axes[2]
    ax3.set_title("障碍物车辆配置")
    road = Rectangle(
        (0, 0),
        env.road_length,
        env.road_width,
        facecolor="darkgray",
        alpha=0.5,
        zorder=0,
    )
    ax3.add_patch(road)
    env._draw_lane_backgrounds(ax3)
    env._draw_lane_markings(ax3)
    env._draw_lane_labels(ax3)
    lane1_center = env.get_lane_center(1)
    lane2_center = env.get_lane_center(2)
    lane3_center = env.get_lane_center(3)
    ax3.axhline(
        y=lane1_center, color="green", linestyle="--", alpha=0.3
    )
    ax3.axhline(
        y=lane2_center, color="green", linestyle="--", alpha=0.3
    )
    ax3.axhline(
        y=lane3_center, color="green", linestyle="--", alpha=0.3
    )
    position_descriptions = [
        "位于车道1中心",
        "位于车道2中心",
        "位于车道3中心",
        "位于车道3中心",
    ]
    for index, vehicle in enumerate(env.obstacle_vehicles):
        assert isinstance(vehicle, ObstacleVehicle)
        vehicle.draw(ax3)
        ax3.annotate(
            f"障碍物{index + 1}: {position_descriptions[index]}",
            (vehicle.x, vehicle.y),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            bbox=dict(
                facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"
            ),
        )
        ax3.annotate(
            f"({vehicle.x:.1f}, {vehicle.y:.1f})",
            (vehicle.x, vehicle.y),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            bbox=dict(
                facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"
            ),
        )
        if index == 0:
            offset = vehicle.y - lane1_center
            ax3.annotate(
                f"偏移: {offset:.1f}m",
                (vehicle.x, vehicle.y),
                xytext=(0, -30),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                bbox=dict(
                    facecolor="yellow",
                    alpha=0.7,
                    boxstyle="round,pad=0.2",
                ),
            )
            ax3.plot(
                [vehicle.x, vehicle.x],
                [lane1_center, vehicle.y],
                "r-",
                linewidth=1,
                alpha=0.5,
            )
    ax3.set_xlim(-5, env.road_length + 5)
    ax3.set_ylim(-2, env.road_width + 2)
    ax3.set_aspect("equal")
    ax3.grid(False)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")

    ax4 = axes[3]
    ax4.set_title("起点和终点位置")
    road = Rectangle(
        (0, 0),
        env.road_length,
        env.road_width,
        facecolor="darkgray",
        alpha=0.5,
        zorder=0,
    )
    ax4.add_patch(road)
    env._draw_lane_backgrounds(ax4)
    env._draw_lane_markings(ax4)
    env.start_vehicle.draw(ax4)
    env.end_vehicle.draw(ax4)
    ax4.annotate(
        "起点",
        (env.start_point[0], env.start_point[1]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        bbox=dict(
            facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"
        ),
    )
    ax4.annotate(
        f"坐标: ({env.start_point[0]:.1f}, {env.start_point[1]:.1f})",
        (env.start_point[0], env.start_point[1]),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        bbox=dict(
            facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"
        ),
    )
    ax4.annotate(
        "终点",
        (env.end_point[0], env.end_point[1]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        bbox=dict(
            facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"
        ),
    )
    ax4.annotate(
        f"坐标: ({env.end_point[0]:.1f}, {env.end_point[1]:.1f})",
        (env.end_point[0], env.end_point[1]),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        bbox=dict(
            facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"
        ),
    )
    ax4.plot(
        [env.start_point[0], env.end_point[0]],
        [env.start_point[1], env.end_point[1]],
        "r--",
        linewidth=1.5,
        label="理想直线路径",
    )
    distance = np.hypot(
        env.end_point[0] - env.start_point[0],
        env.end_point[1] - env.start_point[1],
    )
    mid_x = (env.start_point[0] + env.end_point[0]) / 2
    mid_y = (env.start_point[1] + env.end_point[1]) / 2
    ax4.annotate(
        f"直线距离: {distance:.1f}m",
        (mid_x, mid_y),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        bbox=dict(
            facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"
        ),
    )
    ax4.legend(loc="lower right")
    ax4.set_xlim(-5, env.road_length + 5)
    ax4.set_ylim(-2, env.road_width + 2)
    ax4.set_aspect("equal")
    ax4.grid(False)
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(
        f"道路环境可视化已保存为'{output_path}'"
    )
    return output_path
